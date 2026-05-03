# Agentic clinical reasoning over longitudinal myeloma records: a retrospective evaluation against expert consensus

**Authors**: Johannes Moll, Jannik Lübberstedt, Christoph Nuernbergk, Jacob Stroh, Luisa Mertens, Anna Purcarea, Christopher Zirn, Zeineb Benchaaben, Fabian Drexel, Hartmut Häntze, Anirudh Narayanan, Friedrich Puttkammer, Andrei Zhukov, Jacqueline Lammert, Sebastian Ziegelmayer, Markus Graf, Marion Högner, Marcus Makowski, Florian Bassermann, Lisa C. Adams, Jiazhen Pan, Daniel Rueckert, Krischan Braitsch, Keno K. Bressem

**Published**: 2026-04-27 13:41:18

**PDF URL**: [https://arxiv.org/pdf/2604.24473v1](https://arxiv.org/pdf/2604.24473v1)

## Abstract
Multiple myeloma is managed through sequential lines of therapy over years to decades, with each decision depending on cumulative disease history distributed across dozens to hundreds of heterogeneous clinical documents. Whether LLM-based systems can synthesise this evidence at a level approaching expert agreement has not been established. A retrospective evaluation was conducted on longitudinal clinical records of 811 myeloma patients treated at a tertiary centre (2001-2026), covering 44,962 documents and 1,334,677 laboratory values, with external validation on MIMIC-IV. An agentic reasoning system was compared against single-pass retrieval-augmented generation (RAG), iterative RAG, and full-context input on 469 patient-question pairs from 48 templates at three complexity levels. Reference labels came from double annotation by four oncologists with senior haematologist adjudication. Iterative RAG and full-context input converged on a shared ceiling (75.4% vs 75.8%, p = 1.00). The agentic system reached 79.6% concordance (95% CI 76.4-82.8), exceeding both baselines (+3.8 and +4.2 pp; p = 0.006 and 0.007). Gains rose with question complexity, reaching +9.4 pp on criteria-based synthesis (p = 0.032), and with record length, reaching +13.5 pp in the top decile (n = 10). The system error rate (12.2%) was comparable to expert disagreement (13.6%), but severity was inverted: 57.8% of system errors were clinically significant versus 18.8% of expert disagreements. Agentic reasoning was the only approach to exceed the shared ceiling, with gains concentrated on the most complex questions and longest records. The greater clinical consequence of residual system errors indicates that prospective evaluation in routine care is required before these findings translate into patient benefit.

## Full Text


<!-- PDF content starts -->

Agentic clinical reasoning over longitudinal myeloma records:
a retrospective evaluation against expert consensus
Johannes Moll∗,†1,2,3, Jannik Lübberstedt†2, Christoph Nuernbergk4, Jacob Stroh4,
Luisa Mertens4, Anna Purcarea4, Christopher Zirn2, Zeineb Benchaaben2, Fabian Drexel1,2,
Hartmut Häntze2,5, Anirudh Narayanan2,5, Friedrich Puttkammer2,5, Andrei Zhukov6,
Jacqueline Lammert7,8, Sebastian Ziegelmayer2, Markus Graf2, Marion Högner4,
Marcus Makowski2, Florian Bassermann4,9,10,11, Lisa C. Adams2, Jiazhen Pan1,12,13,
Daniel Rueckert1,13,14, Krischan Braitsch‡4, and Keno K. Bressem‡2,3
1Chair for AI in Healthcare and Medicine, Technical University of Munich (TUM) and TUM University Hospital,
Munich, Germany
2Department of Diagnostic and Interventional Radiology, Klinikum rechts der Isar, TUM University Hospital, School of
Medicine and Health, Technical University of Munich, Munich, Germany
3Department of Cardiovascular Radiology and Nuclear Medicine, German Heart Center, TUM University Hospital,
School of Medicine and Health, Technical University of Munich, Munich, Germany
4Department of Medicine III, Klinikum rechts der Isar, TUM University Hospital, School of Medicine and Health,
Technical University of Munich, Munich, Germany
5Department of Radiology, Charité – Universitätsmedizin Berlin, Berlin, Germany
6Department of Gastroenterology, Infectious Diseases and Rheumatology, Charité – Universitätsmedizin Berlin, Berlin,
Germany
7Chair of Medical Informatics, Institute of AI in Medicine and Healthcare, TUM School of Medicine and Health,
Technical University of Munich, Munich, Germany
8Clinical Department of Gynecology, TUM University Hospital, TUM School of Medicine and Health, Munich,
Germany
9TranslaTUM, Center for Translational Cancer Research, Technical University of Munich, Munich, Germany
10Deutsches Konsortium für Translationale Krebsforschung, Heidelberg, Germany
11Bavarian Cancer Research Center, Munich, Germany
12Department of Engineering Science, University of Oxford, Oxford, UK
13Munich Center for Machine Learning (MCML), Munich, Germany
14Department of Computing, Imperial College London, London, UK
†These authors contributed equally to this work.
‡These authors share senior authorship.
∗Corresponding author:johannes.moll@tum.de
1arXiv:2604.24473v1  [cs.AI]  27 Apr 2026

Abstract
Background.Multiplemyelomaismanagedthroughsequentiallinesoftherapyoveryearstodecades,with
eachtreatmentdecisiondependingoncumulativediseasehistorydistributedacrossdozenstohundreds
ofheterogeneousclinicaldocuments. Whetherlargelanguagemodelbasedsystemscansynthesisethis
evidence at a level approaching expert agreement has not been established.
Methods.A retrospective evaluation was conducted on longitudinal clinical records of 811 patients with
multiplemyelomatreatedatatertiarymedicalcentrebetween2001and2026,covering44,962documents
and 1,334,677 laboratory values, with external validation on MIMIC-IV. An agentic reasoning system
was compared against single-pass retrieval-augmented generation (RAG), iterative RAG, and full-context
input on 469 patient–question pairs derived from 48 templates stratified into three complexity levels.
The reference standard was established by independent double annotation from four oncologists with
adjudication by a senior haematologist.
Findings.Iterativeretrieval-augmentedgenerationandfull-contextinputconvergedonasharedperfor-
mance ceiling (75·4% versus 75·8%, Bonferroni-corrected 𝑝= 1·00). The agentic system reached 79 ·6%
concordance(95%CI76 ·4–82·8),significantlyexceedingbothbaselines( +3·8and+4·2percentagepoints;
𝑝= 0·006 and 0·007). Gains increased with question complexity, reaching +9·4 percentage points on
criteria-based synthesis ( 𝑝= 0·032), and with record length, reaching +13·5 percentage points in the
topdecile(exploratory, 𝑛=10). Thesystemerrorrate(12 ·2%)wascomparabletoexpertdisagreement
(13·6%), but severity distributions were inverted, with 57 ·8% of system errors classified as clinically
significant against 18·8% of expert disagreements.
Interpretation.Agentic reasoning was the only approach to exceed the shared performance ceiling, with
gains concentrated on the most complex questions and longest records. The greater clinical consequence
ofresidualsystemerrorsrelativetoexpertdisagreementindicatesthatprospectiveevaluationinroutine
care will be required before these findings translate into measurable patient benefit.
Funding.Bayern Innovativ (Bavarian State Ministry of Economics), Grant Number: LSM-2403-0006.
Introduction
Multiple myeloma is managed through sequential lines of therapy over years to decades, with each treatment
decision depending on a cumulative record of prior exposures, documented responses, and evolving comor-
bidities that no single source can fully reflect.1A typical patient accumulates dozens to hundreds of clinical
documentsoverthistrajectory,spanninglaboratorydata,pathologyfindings,imagingreports,andfree-text
notes,andthecorrectinterpretationofanyindividualfindingdependsontemporalrelationshipsthatmayspan
years.2Determining whether a patient has progressed on a given regimen, whether prior toxicities preclude a
planned therapy, or whether organ function meets eligibility criteria requires synthesis across document types
and timepoints that no individual document can resolve. These demands fall on haematologists who already
carry among thehighest electronic health record burdensof any clinical specialty, withrecent data showing
more than 575 minutes per week in the record and over 219 minutes of after-hours documentation,3a burden
that scales with the complexity of the disease trajectory.
Large language models (LLMs) have been applied to clinical record navigation across a range of tasks.4Pub-
lishedapproachesfallintothreebroadstrategies: single-passretrieval-augmentedgeneration(RAG),iterative
2

RAG, and agentic systems that reason over specialised tools.5–7Earlier work on guideline interpretation,
domain-tuned question answering, and factual verification showed that retrieval can reduce hallucination on
well-defined single-documenttasks.8–11More recently,Myersand colleaguestested longitudinalreasoning
directly, comparing RAG against full-record long-context input on three tasks over hospitalised patients,
finding equivalent performance at 128,000 tokens of context capacity and concluding that the absence of
expert-adjudicated longitudinal datasets had prevented any test of whether this convergence holds for the
harder multi-source synthesis tasks that dominate real clinical practice.12
Agenticapproacheshavebeenevaluatedprimarilyintwosettings: diagnosticdialogueagentsoperatingon
simulated patients,13,14and tool-using agents tested against structured databases where each task can be
resolved from a small number of entries.15–17Neither setting captures the demand of answering a treatment
planningquestionfrommultipleinstancesofunstructured,partiallycontradictorylongitudinaldocumentation.
No evaluation of any such approach has been reported for multiple myeloma or any comparable disease
trajectory, and whether clinically reliable performance is achievable against an independently annotated
reference standard has not been established.
Here, a retrospective evaluation of expert concordance was conducted on institutional records of 811 patients
withmultiplemyelomatreatedatatertiarymedicalcentreover25years. Anagenticreasoningsystemwas
compared against single-pass RAG, iterative RAG, and full-context input, all using the same locally deployed
open-weight LLM to keep data within institutional infrastructure. The reference standard was established
through independent double annotation by four oncologists with adjudication by a senior haematologist,
residualinter-ratervariabilitywasexplicitlyclassified,andtheclinicalsafetyprofilewascharacterisedthrough
structurederrorseverityanalysis. ExternalvaliditywasassessedontheMIMIC-IVcriticalcaredatabase.18,19
Methods
The study was approved by the Ethics Committee of the Technical University of Munich (approval 2024-
590-S-CB),andinformedconsentwaswaivedgiventheretrospectivedesign. UseofMIMIC-IVdatawas
conductedunder thePhysioNet credentialeddata useagreement.20Allprocedures followedthe Declaration
of Helsinki and applicable institutional guidelines.
Study design and data sources
Aretrospectiveevaluationofexpertconcordancewasconductedonlongitudinalmultiplemyelomarecords
from two independent institutions (Figure 1a). The primary dataset comprised all patients with a confirmed
multiple myeloma diagnosis treated at TUM University Hospital between 1 January 2001 and 1 January
2026(n=811). Textualreportswereextractedfrom44,962German-languageclinicaldocuments,yielding
a median of 55 documents per patient (IQR 20 to 76, Figure 1b) and a median follow-up of 6 ·3 years
fromdiagnosis(IQR1 ·7to9·7,Figure1c),withdetailsoftheextractionandconversionpipelineprovided
3

in Supplementary Methods A.1. Structured laboratory data (1,334,677 values) were extracted from the
institutional laboratory information system and normalised to 731 canonical concepts, preprocessing details
areprovidedinSupplementaryMethodsA.2. Theexternalvalidationdatasetcomprised716patientsidentified
byICDcodefromthepubliclyavailablede-identifiedMIMIC-IVcriticalcaredatabase,19yielding26,767
clinical documents with a median of 26 documents per patient (IQR 11 to 53). This cohort was used without
adaptation to assess transferability across institutions.
Clinical question bank and evaluation cohorts
A bank of 48 clinical question templates covering core decision tasks in multiple myeloma management was
developedinconsultationwithstaffhaematologistsandreviewedforclinicalrepresentativeness. Templates
werestratifiedintothreecomplexitylevels: single-recordlookup(Level1),temporalreasoningacrossmultiple
sources (Level 2), and criteria-based synthesis across document types and multiple timepoints (Level 3). The
full template list with answer formats and scoring methods is provided in Supplementary Table B.2.
Three non-overlapping patient sets were defined prior to system development (Figure 1d, Figure 2). A
developmentsetoftenpatientswasusedexclusivelyforiterativesystemdevelopmentandisexcludedfromall
reportedevaluations. Forthe primaryanalysis, 100patients weresampled fromtheremaining TUMcohort
usingstratifiedsamplingovernumberofreportsandrecencyofthelastavailabledatapoint, witheachpatient
assigned five questions (two Level 1, two Level 2, one Level 3), yielding 500 patient-question pairs. For
external validation, 20 patients were selected from the MIMIC-IV cohort using the same procedure, yielding
100 pairs.
Expert annotation and adjudication
All pairs were annotated by independent chart review prior to system evaluation, with raters blinded to
each other and to all system outputs. Each TUM patient was reviewed by exactly two of four oncologists
with multiple myeloma expertise (AP, CN, JS, LM). Pairs with direct agreement were included without
adjudication, disagreements were reviewed by a senior haematologist (KB) and classified into one of five
categories. Full protocol details and operational definitions are provided in Supplementary Methods.
FortheTUMcohort,directagreementwasreachedon65 ·2%ofpairs,afurther28 ·6%wereincludedafter
adjudication,and6 ·2%wereexcluded,yielding469evaluablepairs(200Level1,179Level2,90Level3;
Figure 1e). Pre-adjudication inter-rater agreement declined with complexity: 𝜅= 0·69 at Level 1, 𝜅= 0·60 at
Level2,and 𝜅=0·57atLevel3(Figure1f). Amongadjudicatedcases,themostfrequentcategorieswere
interchangeable or equivalent responses (39 ·2%) and clinically insignificant disagreement (36 ·4%), with
clinically significant disagreement accounting for 8 ·4% (Figure 1g). For the MIMIC-IV cohort, 89 evaluable
pairs were retained after the same procedure.
Agentic system and comparators
Theagenticsystemwasdesignedtoanswerclinicalquestionsrequiringsynthesisacrosstemporallydistributed,
heterogeneous documentation. The system was distinguished from retrieval-based approaches by four
4

architectural components (Figure 3). Task-relevant modules were selected from an indexed clinical skill
library encoding question-type-specific reasoning protocols. An ordered tool-use plan with explicit stopping
conditionswas constructed priortoretrieval. Astructuredmemorystate encodingtheuserquery,retrieved
evidence, missing information, and stopping conditions was updated iteratively after each step. Iterative
execution was carried out against purpose-built tools, including report and laboratory value retrieval with
typeanddatefiltersanddeterministicclinicalscoringcalculators,withautomaticre-queryoninsufficient
evidence. Final answers followed a schema-defined format with inline source citations. Full implementation
details are provided in Supplementary Methods.
Thesamelocallydeployed120-billion-parameteropen-weightlanguagemodel(gpt-oss-120b21)wasused
acrossallthreecomparatorapproachesandthesamepatientrecorddatabase. InSimpleRAG,single-pass
dense retrieval was implemented without query rewriting or reranking.22Iterative RAG was extended
to include subquery rewriting, hybrid BM25 and dense retrieval fusion, cross-encoder reranking, and a
multi-round sufficiency loop.5,23In the Full Context configuration, retrieval was bypassed entirely, all
documents and laboratory values were concatenated in reverse chronological order until the context window
wasfilled.12TwoadditionalcomparatorsarereportedinSupplementaryTableB.3,anddetailedspecifications
for all approaches are provided in Supplementary Methods.
Error classification and citation sufficiency
Allpatient-questionpairsforwhichthe agenticsystemdiverged fromexpert consensuswereclassified bya
senior haematologist (KB) into one of six categories: clinically significant error, clinically insignificant error,
partially correct, acceptable or ambiguous, annotation error, and pipeline failure. The error classification
taxonomy was aligned with the adjudication categories applied during expert annotation, enabling direct
comparisonofsystemerrorrateswithexpertdisagreementratesatequivalentseveritylevels. Toassessthe
reproducibilityofthisclassification,ablindedinter-raterreliabilitysub-studywasconductedbyanindependent
oncologist(JS)onaproportionalstratifiedrandomsampleof46ofthe115divergentpatient-questionpairs,
with categories additionally collapsed into three severity strata for analysis. Citation sufficiency was assessed
by two reviewers (KB, CN) on a stratified sample of 96 responses, with each of the 48 question templates
represented at least once. Full classification criteria, sampling details, and inter-rater reliability protocol are
provided in Supplementary Methods.
Statistical analysis
Theprimaryoutcomewasconcordancewith expertconsensus,definedastheproportionofpairsforwhich
system output matched the adjudicated reference on substantive content. Single-value categorical items were
scored as binary, whereas list-type items were scored using entry-level F1 computed against the reference list,
contributingacontinuousvaluebetweenzeroandone. Secondaryoutcomesweretheclinicallysignificant
error rate derived from structured error classification and citation sufficiency. Each system was evaluated
across ten independent runs to control for stochastic variability, and per-question scores were averaged
across runs before analysis. Individual concordance estimates and their 95% confidence intervals were
computed by pair-level percentile bootstrap ( 𝑁boot=10,000). Pairwise significance tests were computed
5

via cluster bootstrap ( 𝑁boot=10,000,𝛼= 0·05), resampling whole patients with replacement to account
for within-patient correlation. Bonferroni correction was applied within each stratum independently, and
allreported 𝑝-valuesareBonferroni-correctedunlessexplicitlystatedotherwise. Therelationshipbetween
patientrecordlengthandsystemperformancewasexaminedasahypothesis-drivenexploratoryanalysis,with
patients stratified into four quantile-based bins prior to analysis (Supplementary Table B.9).
Role of the funding source
The funders of the study had no role in study design, data collection, data analysis, data interpretation, or
writing of the report.
Results
Accuracy against expert consensus
The primary evaluation comprised 469 patient-question pairs across 100 patients from the TUM cohort,
including 200 Level 1, 179 Level 2, and 90 Level 3 pairs retained after adjudication. Iterative RAG and Full
Contextperformedequivalently,reaching75 ·4%(95%CI[71·9–78·6])and75·8%([72·3–79·3])concordance
with expert consensus respectively, with no statistically significant difference between them ( −0·4 percentage
points, Bonferroni-corrected𝑝= 1·00). Simple RAG trailed at 71·5% ([67·8–75·1]).
The agentic system achieved an overall concordance of 79 ·6% ([76·4–82·8]) (Figure 4a, Supplementary
Table B.3), significantly higher than both Full Context ( +3·8 percentage points, 𝑝= 0·006) and Iterative RAG
(+4·2percentagepoints, 𝑝=0·007). Acrosstenindependentevaluationruns,theagenticsystemshoweda
standarddeviationof1 ·1percentagepoints(SupplementaryTableB.4). Theskilllibrarywasidentifiedby
ablation analyses as the principal driver of performance. Its removal reduced overall concordance by 3 ·0
percentage points to 76 ·6%, whereas removal of type and date filters in retrieval tools, deterministic clinical
scoring tools, structured memory state, or pre-planned tool use individually reduced concordance by at most
0·4 percentage points (Supplementary Table B.5).
External validation on 89 evaluable patient-question pairs across 20 patients from the MIMIC-IV cohort
preserved the system ranking (Figure 4e). The agentic system achieved an overall concordance of 84 ·9%
([77·8–91·2]), compared with 79 ·2% ([71·4–86·3]) for Simple RAG, 77 ·9% ([70·1–85·3]) for Iterative
RAG, and 74·1% ([65·5–82·2]) for Full Context. The three non-agentic approaches clustered within
overlappingconfidenceintervals,andtheagenticsystemretainedthehighestconcordancedespiteachange
in documentation language (English versus German), institutional conventions, record structure, and the
de-identification date-shifting applied to the source data. No system adaptation was performed between
cohorts.
Performance across question complexity and record length
Concordancedecreased withquestioncomplexity acrossallapproaches,butnot uniformly(Figure4a). For
the agentic system, concordance fell from 86 ·1% ([81·9–90·1]) at Level 1 to 79·5% ([74·1–84·5]) at Level 2
6

and 65·1% ([56·8–73·3]) at Level 3. The advantage of the agentic system over Full Context increased
monotonicallywithcomplexity,from +1·0percentagepointatLevel1( 𝑝=1·00)to+3·9percentagepointsat
Level 2 ( 𝑝= 0·25) and+9·4 percentage points at Level 3 ( 𝑝= 0·032) (Figure 4b). A similar gradient was
observed relative to Iterative RAG, with a difference of +9·7 percentage points at Level 3 ( 𝑝= 0·049) but no
significantdifferenceatLevels1or2. AtLevel3,thethreenon-agenticapproachesrangedfrom48 ·9%to
55·7%, whereas the agentic system reached 65·1%.
Concordancewasstratifiedbytotalpatientrecordlengthinahypothesis-drivenexploratoryanalysis(Figure4c).
Patients were divided into four bins: three terciles of the lower 90th percentile ( ≤127k, 127 to 282k, and 282
to541kcharacters)andthetopdecile( >541kcharacters,correspondingtoapproximately245,000tokens;
𝑛=10patients, 47 evaluable pairs). In the lower three bins, no statistically significant differences were
observed between the agentic system and either Full Context or Iterative RAG. In the top decile ( 𝑛= 10
patients,47evaluablepairs;exploratory),theagenticsystemachieved75 ·3%concordancecomparedwith
63·9%forIterativeRAG( +11·4percentagepoints)and61 ·8%forFullContext( +13·5percentagepoints),
bootstrapconfidenceintervalsforthesecomparisonswerewideowingtothesmallpatientcountandthese
findingsshouldbeinterpretedashypothesis-generating. Fromtheshortesttothelongestbin,concordancefell
by 7·1 percentage points for the agentic system, compared with 16 ·9 for Iterative RAG, 19 ·6 for Full Context,
and 26·5 for Simple RAG.
Clinical error profile and failure mechanisms
Errorclassificationwasperformedonasingleevaluationrunwithoverallconcordanceof77 ·8%. All115
patient-question pairs for which the agentic system diverged from expert consensus on that run, including
list-type responses with partial but incomplete overlap (F1 between zero and one), underwent structured
classification by a senior haematologist into one of six categories (Table 1). Thirty-three divergences (28 ·7%)
were classified as clinically significant errors, 24 (20 ·9%) as clinically insignificant errors, six (5 ·2%) as
partially correct, 39 (33 ·9%) as acceptable or ambiguous responses in which both the system output and the
reference annotation were defensible from the record, 11 (9 ·6%) as annotation errors in which the system
response was correct and the reference annotation was not, and two (1 ·7%) as pipeline failures. The clinically
significanterrorratewas33of469pairs(7 ·0%),withclinicallysignificanterrorsdistributedacrosscomplexity
levels: 14atLevel1,12atLevel2,andsevenatLevel3. Blindedre-annotationofaproportionalstratified
randomsampleof46divergentpairsbyanindependentrateryieldedasix-category 𝜅of0·667( 95 %CI[0·510,
0·833]) and a three-stratum 𝜅of 0·802 ([0·687, 0·962]), indicating substantial to near-perfect reproducibility
of the classification across severity levels.
The system error rate, combining clinically significant and clinically insignificant categories, was 57 of
469 pairs (12·2%), comparable to the rate of expert disagreement on the same tasks (64 of 469 pairs,
13·6%, Figure 1g). Among system errors, 57 ·8% (33 of 57) were clinically significant, and among expert
disagreements,18·8%(12of64)wereclinicallysignificant. Theseveritydistributionswerethereforeinverted
despitecomparableoverallrates,withsystemerrorscarryinggreateraverageclinicalconsequencethanexpert
disagreements on the same tasks.
Citationsufficiencywasassessedonastratifiedsampleof96agenticsystemresponsesfromthesamerun,
7

drawn evenly across complexity levels and concordance status, with each of the 48 question templates
represented at least once (Figure 4d). All concordant responses were rated as fully supported by the retrieved
source documents across all three complexity levels. Among discordant responses, the proportion rated
as fully supported was 50% at Level 1, 62% at Level 2, and 50% at Level 3. Of the 22 responses with
citation insufficiency, incomplete retrieval was the dominant failure mechanism, accounting for 83% of fully
unsupported responses ( 𝑛=18), in the remaining17%, a relevant document had been retrieved butwas not
incorporated into the final response.
Discussion
Retrieval-augmented and full-context approaches converged on a shared performance ceiling on real
longitudinal myeloma records, and agentic reasoning was the only approach to exceed it. The advantage was
notuniform. Itwasnegligibleonsimplesingle-recordlookups,grewmonotonicallywithquestioncomplexity,
and was largest for patients with the longest and most complex disease trajectories, precisely those for whom
the clinical stakes of accurate synthesis are highest.
The convergence of Iterative RAG and Full Context at a shared ceiling indicates that the limiting factor is not
the quantity of evidence presented to the model but the structure imposed on its integration. RAG delegates
decomposition, evidence weighting, and clinical decision rules to a single generation step. Agentic reasoning
externalises these operations as an explicit planning phase, iterative evidence gathering with re-query on
insufficient results, and deterministic scoring calculators where clinical rules can be encoded without relying
solely on language model interpretation. Ablation analyses indicate that the skill library contributes the
largest individually separable gain, reducing concordance by 3 ·0 percentage points when removed, while the
remaining components each contribute at most 0 ·4 percentage points in isolation, consistent with their role as
interdependentscaffoldingthroughwhichquestion-type-specificreasoningprotocolsareexecuted. Thepresent
findings extendthe convergenceceiling previously observedon simplerclinical question answering12to the
harder longitudinal synthesis setting and show that structured reasoning is the first approach demonstrated to
exceed it.
Theclinicalprofileoftheagenticadvantageidentifiesacoherentdeploymenttarget. Thegapoverthestrongest
non-agenticapproachwidenedwithbothquestioncomplexityandtotalrecordlength,concentratingonpatients
with themost treatment lines, relapse-remission cycles, andaccumulating eligibility constraints. A decade-
long myeloma trajectory with multiple sequential regimens and evolving comorbidities is simultaneously the
case where manual chart review consumes the most physician time and the case where accurate longitudinal
synthesisismostdecision-relevant. Performancedifferenceswerenegligibleforpatientsinthelower90th
percentileofrecordlengthandlargestinthetopdecile(exploratory, 𝑛=10patients),andtheagenticadvantage
over Full Context increased from 1·1 percentage points at Level 1 to 9·4 percentage points at Level 3.
The system error rate (12 ·2%) was comparable to the rate of expert disagreement on the same tasks (13 ·6%),
buttheseveritydistributionswereinverted: amongsystemerrors,57·8%wereclinicallysignificant,against
18·8% of expert disagreements. The system therefore operated within the bounds of human annotator
variability in aggregate, while its errors carried greater average clinical consequence. This distinction is not
captured by concordance metrics alone and defines the gap between matching expert agreement in aggregate
8

and readiness for clinical deployment. One measurement asymmetry should be noted, expert disagreements
weremeasuredbeforeadjudication,whereassystemdivergencesweremeasuredagainstthefinalisedreference
standard.
Thedominantfailuremechanismdifferedwithtaskcomplexity. Errorsonsingle-recordlookuptasksarose
predominantlyfromincompleteretrieval,asrelevantdocumentswerenotreturnedbythesearchtools,apattern
directly addressable through retrieval engineering. Errorson multi-criterion synthesis tasks arose when the
relevant evidence hadbeen retrieved but was notcorrectly integrated duringreasoning, a pattern consistent
with the documented tendency of long-context language models to underweight information positioned in the
middle of their input.24The same mechanism likely explains part of the Full Context underperformance,
whererelevantpassageswerepresentinthecontextwindowbutwerenotreliablyattendedto. Bothlimitations
are properties of the underlying language model rather than of the agentic architecture. The model used here
(gpt-oss-120b) was selected to run reproducibly on a single H200 GPU within institutional infrastructure,
andlargerormorerecentmodelswithstrongerlong-contextreasoningmaynarrowthegapattributableto
retrieval and integration failures.
External validation on the MIMIC-IV critical care database preserved both the system ranking and the
complexity gradient despite a change in documentation language, institutional conventions, and record
structure, with no system adaptation between cohorts. The external cohort is small, and confirmation on
larger prospectively defined datasets across further institutions remains needed.
Limitations
The question bank was developed with haematologists at the same institution and finalised before system
development, so implicit alignment between question design and system architecture cannot be excluded.
The senior haematologist who drafted the question bank also served as sole adjudicator of annotation
disagreements without independent second review, though error classification by the same rater showed
substantial to near-perfect reproducibility on blinded re-annotation of a stratified random sample ( 𝜅= 0·667
for six categories, 0 ·802 for three strata). Error classification was performed without formal blinding to
systemperformance,introducingpotentialbiasinclinicalsignificanceassignments. Expertdisagreements
were classified before adjudication whereas system divergences were measured against the finalised reference
standard, anasymmetrythatmayfavourthesystem. Theagenticadvantagewasconsistentacrossthreeoffive
backbonemodelsbutattenuatedforthetwowithstrongerbaselineperformance,suggestingitsmagnitude
depends on the long-context integration capabilities of the underlying model (Supplementary Table B.6).
Comparison with proprietary frontier models was precluded by institutional data privacy requirements. The
evaluation is retrospective, and whether agentic assistance improves clinical decisions in prospective use has
not been assessed.
Conclusion
Agentic reasoning answered clinical questions on years of real longitudinal myeloma records at a level of
concordance with expert consensus that retrieval-augmented and full-context approaches did not reach, and
9

its advantage increased with the complexity of cases. The total system error rate fell within the range of pre-
adjudicationinter-expertvariabilityonthesametasks,althoughresidualsystemerrorscarriedgreateraverage
clinical consequence. Prospective evaluation in routine clinical care, with treating clinicians interacting with
system outputs, will be required to determine whether these findings translate into measurable benefit at the
point of care.
Contributors
JM, LCA, KB, and KKB conceived and designed the study. JLu developed and coordinated the data
acquisitionandpreprocessingpipeline,andcontributedtosystemdesignandevaluation. JLa,SZ,MG,and
MHcontributedtodataacquisition. CN,LM,AP,andJScontributedtotheclinicalquestionbankandannotated
the data. KB drafted the clinical question bank, served as final adjudicator for all annotation disagreements,
and contributed to the citation sufficiency and error classification reader studies. CN contributed to the error
classification reader study. JM designed the agentic system, executed the experiments, performed result
analysis and interpretation, and drafted the manuscript. CZ, ZB, FD, HH, AN, FP, and AZ provided iterative
feedbackonstudydesign,execution,andinterpretation,andrevisedthemanuscriptcriticallyforimportant
intellectual content. MM, FB, JP, DR, and KKB contributed to study design, supervised the study, and
revisedthemanuscript. JM,JLu,KB,andKKBdirectlyaccessedandverifiedtheunderlyingpatient-level
data reported in this study. All authors reviewed and approved the final version of the manuscript and accept
responsibility for the decision to submit for publication.
Declaration of interests
APreceivedtravelsupportfromJanssen-CilagandKiteGilead. FBreceivedhonorariafromAmgen,Johnson
&Johnson,BristolMyersSquibb(BMS),AbbVie,andGSK,travelsupportfromAmgen,Johnson&Johnson,
and BMS, and served on advisory boards for Amgen, Johnson & Johnson, BMS, AbbVie, and GSK. MH
received honoraria from Johnson & Johnson, Sanofi, GSK, Oncopeptides, and Pfizer, payment for expert
testimonyfromJohnson&JohnsonandOncopeptides,travelsupportfromJohnson&Johnson,Oncopeptides,
Pfizer, and Amgen, and served on advisory boards for Johnson & Johnson, Sanofi, Oncopeptides, and Pfizer.
JLareceivedaGoogleGemmaAcademicProgramAward,receivedspeakerhonorariafromtheForumfor
Continuing Medical Education (Germany), AstraZeneca (Germany), and Novartis (Germany), and served on
an advisory board for Novartis (Germany). All other authors declare no competing interests.
Data sharing
The MIMIC-IV database is publicly available under the PhysioNet credentialled data use agreement at
https://doi.org/10.13026/1n74-ne17 . Thein-houseTUMdatasetcannotbesharedpubliclyowingto
patientprivacyregulationsandinstitutionaldatagovernancerequirements. Anonymisedaggregateresults
supporting the findings of this study are available from the corresponding author (johannes.moll@tum.de) on
reasonablerequestfromthedateofpublication. Requestswillbereviewedbythestudyteamandasigned
data access agreement will be required before release.
10

Acknowledgments
This study was financed by the public funder Bayern Innovativ (Bavarian State Ministry of Economics)
Nuremberg, Grant Number: LSM-2403-0006. KKB and LCA are further grateful to be supported by the
Else-Kröner-Fresenius-Foundation(2024_EKES.16,2025_EKES.03). FBreceivesfundingbytheGerman
Research Foundation (DFG) (TRR 387/1 - 514894665) and DFG BA 2851/7-1 (project ID: 537477296).
References
[1]Rajkumar SV. Multiple myeloma: 2022 update on diagnosis, risk stratification, and management.
American journal of hematology. 2022;97(8):1086-107.
[2]Cui H, Unell A, Chen B, Fries JA, Alsentzer E, Koyejo S, et al. Timer: Temporal instruction modeling
and evaluation for longitudinal clinical records. npj Digital Medicine. 2025;8(1):577.
[3]HolmgrenAJ,ApathyNC,CrewsJ,ShanafeltT. Nationaltrendsinoncologyspecialists’EHRinbox
work, 2019–2022. JNCI: Journal of the National Cancer Institute. 2025;117(6):1253-9.
[4]Du X, Zhou Z, Wang Y, Chuang YW, Li Y, Yang R, et al. Performance and improvement strategies for
adapting generative large language models for electronic health record applications: a systematic review.
International Journal of Medical Informatics. 2025:106091.
[5]Liu S, McCoy AB, Wright A. Improving large language model applications in biomedicine with
retrieval-augmentedgeneration: asystematicreview,meta-analysis,andclinicaldevelopmentguidelines.
Journal of the American Medical Informatics Association. 2025;32(4):605-15.
[6]Qiu J, Lam K, Li G, Acharya A, Wong TY, Darzi A, et al. LLM-based agentic systems in medicine and
healthcare. Nature Machine Intelligence. 2024;6(12):1418-20.
[7]TruhnD,AziziS,ZouJ,Cerda-AlberichL,MahmoodF,KatherJN. Artificialintelligenceagentsin
cancer research and oncology. Nature Reviews Cancer. 2026:1-14.
[8]Ferber D, Wiest IC, Wölflein G, Ebert MP, Beutel G, Eckardt JN, et al. GPT-4 for information retrieval
and comparison of medical oncology guidelines. Nejm Ai. 2024;1(6):AIcs2300235.
[9]Lu KH, Mehdinia S, Man K, Wong CW, Mao A, Eftekhari Z. Enhancing Oncology-Specific Question
Answering With Large Language Models Through Fine-Tuned Embeddings With Synthetic Data. JCO
Clinical Cancer Informatics. 2025;9:e2500011.
[10]Liu X, Zhang L, Munir S, Gu Y, Wang L. Verifact: Enhancing long-form factuality evaluation with
refined fact extraction and reference facts. In: Proceedings of the 2025 Conference on Empirical
Methods in Natural Language Processing; 2025. p. 17919-36.
[11]Chung P, Swaminathan A, Goodell AJ, Kim Y, Momsen Reincke S, Han L, et al. Verifying Facts in
PatientCareDocumentsGeneratedbyLargeLanguageModelsUsingElectronicHealthRecords. NEJM
AI. 2025;3(1):AIdbp2500418.
11

[12]Myers S, Dligach D, Miller TA, Barr S, Gao Y, Churpek M, et al. Evaluating Retrieval-Augmented
Generationvs.Long-ContextInputforClinicalReasoningoverEHRsf.arXiv[preprint]arXiv:250814817.
2025.
[13]TuT,SchaekermannM,PalepuA,SaabK,FreybergJ,TannoR,etal. Towardsconversationaldiagnostic
artificial intelligence. Nature. 2025;642(8067):442-50.
[14]SchmidgallS,ZiaeiR,HarrisC,ReisE,JoplingJ,MoorM. Agentclinic: amultimodalagentbenchmark
to evaluate ai in simulated clinical environments. arXiv [preprint] arXiv:240507960. 2024.
[15]Shi W, Xu R, Zhuang Y, Yu Y, Zhang J, Wu H, et al. Ehragent: Code empowers large language models
for few-shot complex tabular reasoning on electronic health records. In: Proceedings of the 2024
Conference on Empirical Methods in Natural Language Processing; 2024. p. 22315-39.
[16]Jiang Y, Black KC, Geng G, Park D, Zou J, Ng AY, et al. MedAgentBench: a virtual EHR environment
to benchmark medical LLM agents. Nejm Ai. 2025;2(9):AIdbp2500144.
[17]LeeG,BachE,YangE,PollardT,JohnsonA,ChoiE,etal. Fhir-agentbench: Benchmarkingllmagents
for realistic interoperable ehr question answering. arXiv [preprint] arXiv:250919319. 2025.
[18]JohnsonA,PollardT,HorngS,CeliLA,MarkR. MIMIC-IV-Note: Deidentifiedfree-textclinicalnotes.
PhysioNet. 2023 Jan. Version 2.2. Available from:https://doi.org/10.13026/1n74-ne17.
[19]JohnsonAE,BulgarelliL,ShenL,GaylesA,ShammoutA,HorngS,etal. MIMIC-IV,afreelyaccessible
electronic health record dataset. Scientific data. 2023;10(1):1.
[20]Goldberger AL, Amaral LA, Glass L, Hausdorff JM, Ivanov PC, Mark RG, et al. PhysioBank,
PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals.
circulation. 2000;101(23):e215-20.
[21]AgarwalS,AhmadL,AiJ,AltmanS,ApplebaumA,ArbusE,etal. gpt-oss-120b&gpt-oss-20bmodel
card. arXiv [preprint] arXiv:250810925. 2025.
[22]Lewis P, Perez E, Piktus A, Petroni F, Karpukhin V, Goyal N, et al. Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in neural information processing systems. 2020;33:9459-74.
[23]DoanNN,HärmäA,CelebiR,GottardoV.Ahybridretrievalapproachforadvancingretrieval-augmented
generation systems. In: Proceedings of the 7th International Conference on Natural Language and
Speech Processing (ICNLSP 2024); 2024. p. 397-409.
[24]LiuNF,LinK,HewittJ,ParanjapeA,BevilacquaM,PetroniF,etal. Lostinthemiddle: Howlanguage
modelsuselongcontexts. Transactionsoftheassociationforcomputationallinguistics.2024;12:157-73.
12

Figures
Figure 1
Radiology Reporta
Discharge Summary
Medical History
Diagnosis
Treatment
Laboratory Results1,527 patients across 2 institutions
71,729 semi-structured reports
4,374,481 laboratory results
558 expert-annotated QA pairs
Curate questions from
active MM clinical
practicePhase I: Question Curation
Standardize and stratify
by complexity
Harmonize phrasing, assign
complexity levels (L1/L2/L3)
Design structured
answer schemas
Define expected fields and
formats per question-type
48 templates across five
clinical tasks and three
complexity levelsDocument
ConversionExtraction &
StructuringMetadata
IndexingQuality Control
and ValidationQA Annotation
by 4 Oncologists
Patient ID
Report Type
Report Date
Document ID
{ json }.PDF
.MDExpert
Adjudication
Phase II: Cohort Sampling
External validation set:
20 patients (MIMIC-IV)
Independent cohort from
separate institutionSample 100 patients for
in-house evaluation
Stratified by overall report
count & last report date
10/100/20 patients across
development, evaluation,
and external validationDev set: 10 patients (pre-
2020 diagnosis)
Excluded from evaluationsPhase III: Expert Annotation
Double annotation for
in-house and external
Independent annotation
by two domain experts
Senior expert
adjudication
Fifth annotator resolves
all annotation conflicts
558 gold-standard QA
pairs: 469 in-house (TUM)
+ 89 external (MIMIC-IV)No expert annotation
for development
0–910–19 20–39 40–59 60–79 80–99100–149 150–199 200–399
Documents per patient0510152025
Patients (%)Number of Documents per Patient
TUM (n=811)
MIMIC (n=716)
TUM median
MIMIC median
0–1 1–2 2–4 4–6 6–8 8–10 10–15 15–25
Follow-up span (years)05101520253035Patients (%)Follow-up Span per Patient
TUM (n=811)
MIMIC (n=716)
TUM median
MIMIC median
TUM MIMIC0255075100Proportion of annotated pairs (%)65.2%28.6%
67.0%22.0%Annotation results
Direct agreement
AdjudicatedExcluded36.4%39.2%TUM (n = 143)
13.6%
40.9%27.3%MIMIC (n = 22)
Interchangeable / equivalent
Clinically insignificant disagreement
Clinically significant disagreement
Ambiguous question / evidence
Single rater abstained
Different granularityAdjudication resultsb
cd
e g f
0 0.2 0.4 0.6 0.8 1.0
Cohen'sκ/ Concordance score (%)
0.69
0.60
0.57
0.34
0.74
0.7383.3%
74.6%
61.4%
75.0%
84.4%
57.1%TUM L1 (n=192)
TUM L2 (n=168)
TUM L3 (n=83)
MIMIC L1 (n=40)
MIMIC L2 (n=32)
MIMIC L3 (n=14)
Cohen's κInter-rater reliability
Concordance
Figure 1: Construction of longitudinal cohorts and expert-annotated evaluation dataset enabling clinically
grounded assessment of longitudinal reasoning.(a) Overview of data sources and preprocessing pipeline across two
institutions,includingdocumentextraction,structuring,metadataindexing,andqualitycontrolappliedtoheterogeneous
clinical records. (b) Distribution of document counts per patient, demonstrating substantial variability in record
density and reflecting the complexity of real-world longitudinal documentation. (c) Distribution of follow-up duration,
highlightinglong-termdiseasetrajectoriesintheTUMcohortcomparedwithshorterobservationwindowsinMIMIC-IV.
(d) Study design and cohort construction, including development, in-house evaluation, and external validation sets.
(e)Annotationoutcomesshowingproportionsofdirectagreement,adjudicatedcases,andexclusions. (f)Inter-rater
reliabilityacrosspredefinedcomplexitylevels,reportedasCohen’s 𝜅andobservedagreement,illustratingmoderate
agreementforclinicallycomplextasks. Thelow 𝜅atMIMICLevel1reflectshighprevalenceofnegativeresponses
inflating the chance-agreement baseline. (g) Distribution of adjudication categories, indicating that a substantial
proportion of disagreements reflects clinically insignificant or interchangeable interpretations rather than true errors.
13

Figure 2
TUM University
Hospital
In-house databaseMIMIC-IV Note
dataset
P=145,915 patients
P=811 patients
With multiple myeloma
diagnosisP=716 patients
With multiple myeloma
diagnosis
P=776 patients
Matching documentation
criteriaP=662 patients
Matching documentation
criteriaP=35 patients
Excluded with insuficient
documentationP=45 patients
Excluded with insufficient
documentation
Evaluation set
P=100 patients
Stratified samplingExternal validation set
P=20 patients
Stratified samplingDevelopment set
P=10 patients
(no expert annotation)
n=500 question-answer
pairs
Two level 1, two level 2,
and one level 3n=31 question-answer
pairs
Excluded post-
adjudicationn=11 question-answer
pairs
Excluded post-
adjudicationn=100 question-answer
pairs
Two level 1, two level2,
and one level 3
n=469 evaluable
question-answer pairs
Primary analysisn=89 evaluable
question-answer pairs
External validation
Figure2: Cohortselectionyieldsrepresentativeevaluationsetsforlongitudinalclinicalreasoningtasks.Flow
diagram of patient inclusion from institutional (TUM) and external (MIMIC-IV) datasets, including filtering based
ondiagnosisanddocumentationcriteria. Theprimaryevaluationcohortcomprised100patientswith500annotated
question–answerpairs,ofwhich469wereretainedafteradjudication. Theexternalvalidationcohortincluded20patients
with100annotatedpairs,ofwhich89wereretained. Stratifiedsamplingensuredcoverageacrossvaryingdocumentation
density and temporal extent, supporting evaluation of reasoning under heterogeneous real-world conditions.
14

Figure 3
Patient
Is this patient eligible for
CAR-T therapy?I. Receive patient
and question
II. Interpret
question and
select suitable
agent skillsUser workflow Agent workflow Structured model context
Task context
Retrieved skillsRequired evidence
Stop conditions
Retrieved informationTools and Skills
Skill library
III. Draft
ordered tool-
use planTask context
Retrieved skills
Ordered tool-use plan
{ } { } { } { }
IV. Execute
tools to retrieve
informationToollibrary
{ } { } { }
{ } { } { }
Task context
Retrieved information
+-Database+
Ordered tool-use plan
{ } { } { } { }Structured
retrieval
tool
{ }V. Sort and
prioritize
retrieved
information Is this patient eligible for
CAR-T therapy?
Answer: Yes
Reasoning:  The patient has
received therapies with IMiD [1],
an anti-CD38 antibody [2], and …[1] 2025-10-09
discharge_note
[2] 2023-05-29
tumor_board
[3] 2020-12-04
rad_report
[4] 2026-02-14
CreatinineVI. Formulate
answer with
sourcesTask context
Retrieved information
Patient
a b c d
Report Type
Report Date
Figure 3: Agentic system enables structured, traceable clinical reasoning across longitudinal patient records.(a)
User-facingworkflowillustratingqueryinputandgenerationofcitation-backedanswersgroundedinpatientrecords.
(b) Internal agent workflow, including question interpretation, retrieval of task-specific clinical skills, generation of
an ordered tool-use plan, iterative evidence retrieval, and synthesis. (c) Structured model context integrating task
requirements,retrieveddomainknowledge,intermediateevidence,andstoppingcriteriaforreasoningcompletion. (d)
Tool and skill library supporting structured access to clinical reports, laboratory trajectories, and deterministic scoring
systems. Together, these components enable multi-step reasoning in which each intermediate step is explicitly linked to
retrieved evidence, ensuring that final answers are verifiable and traceable to source documents.
15

Figure 4
≤127k 128-284k 284-540k >541k
Patient record length (characters)55%65%75%85%95%Concordance (%)Level 1 (easy)
Simple RAG
Iterative RAG
Full Context
Agent38%48%58%68%78%88%98%Level 2 (medium)
30%40%50%60%70%Level 3 (hard)
Concordance by difficulty level vs. patient record length0%20%40%60%80%100% Concordance (%)
79.1%82.4%85.1% 86.1%
74.4%77.7% 75.7%79.5%
48.9%55.4% 55.7%65.1% 71.5%75.4% 75.8%79.6%
Level 1 – Simple
Level 2 – Medium
Level 3 – Complex
OverallQA Concordance on the evaluation set (N = 469)
30%40%50%60%70%80%90%100%Concordance by level across systems
Level 1 – Simple
Level 2 – Medium
Level 3 – Complex
+3.3 pp
+3.8 pp*
+9.4 pp**
Simple RAG
SystemIterative RAG
SystemFull
ContextAgentic
System+2.7 pp+1.0 pp
-2.0 pp +3.3 pp
+6.5 pp**+0.3 pp
Simple RAG
SystemIterative RAG
SystemFull
ContextAgentic
System
Correct0%20%40%60%80%100% Cases (%)Level 1 Level 2 Level 3
100%
50%50%
100%
62%31%
100%
50%19%31%Fully supported Partially supported Not supportedCitation sufficiency (N = 96)
Incorrect Correct Correct Incorrect IncorrectPatient record length (characters) Patient record length (characters)≤127k 128-284k 284-540k >541k ≤127k 128-284k 284-540k >541kConcordance (%)
Concordance (%)Concordance (%)a b
c
d
0%20%40%60%80%100%Accuracy (%)
77.8%
75.2% 72.8%82.8%92.5% 94.7%87.2%96.6%
52.9%
47.1% 47.9%64.3%79.2%
77.9%74.1% 84.9%
QA Concordance on the external validation set (MIMIC-IV, N = 89)
Level 1 – Simple
Level 2 – Medium
Level 3 – Compl
Overall
Simple RAG
SystemIterative RAG
SystemFull
ContextAgentic
Systeme
Figure4: Agenticreasoningimprovesaccuracy,withlargestgainsinclinicallycomplextasksrequiringlongitudinal
synthesis.(a)Overallandstratifiedconcordancewithexpertconsensusacrosssystemconfigurationsontheprimary
evaluationcohort,showingsuperiorperformanceoftheagenticsystem. (b)Performancedifferencesbycomplexitylevel,
demonstratingthattheadvantageoftheagenticapproachincreaseswithtaskdifficulty. (c)Concordancestratifiedby
patientrecordlengthandcomplexitylevel,showingconvergenceofallsystemsforshorterrecordsandthelargestagentic
advantage among patients in the top decile of record length ( >541k characters; exploratory, 𝑛= 10 patients), where
non-agentic configurations decline sharply. (d) Citation sufficiency analysis showing the proportion of responses fully,
partially, or not supported by retrieved source documents across complexity levels and concordance status. (e) External
validation on the MIMIC-IV cohort, confirming preservation of system ranking and robustness across institutions and
documentation structures.
16

Table 1
Table 1: Error classification of agentic system divergences from expert consensus.All 115 patient-question pairs
for which the agentic system output diverged from the adjudicated reference annotation on a single evaluation run
(overallconcordance77 ·8%)wereclassifiedbyaseniorhaematologist. Theclinicallysignificanterrorratewas33of
469 evaluable pairs (7·0%).
Classification Overall Level 1 Level 2 Level 3
Clinically significant error 33 (28·7%) 14 12 7
Clinically insignificant error 24 (20·9%) 5 11 8
Partially correct 6 (5·2%) 1 5 0
Acceptable / ambiguous 39 (33·9%) 9 12 18
Annotation error 11 (9·6%) 3 6 2
Pipeline failure 2 (1·7%) 1 0 1
Total 115 33 46 36
17

A Supplementary Methods
A.1 Document preprocessing and extraction pipeline
Dischargesummaries,radiologyreports,pathologyreports,tumourboardproceedings,andancillarydocument
types(cardiology, cytology, flowcytometry, andgenomicdiagnosticsreports)wereexportedasPDFfiles
from the institutional SAP electronic health record system at TUM University Hospital and converted to
structuredMarkdownusinga rule-basedmodule thatapplied font,size, positioning,and layout metadatato
exclude non-clinical content and to identify section boundaries, which were preserved as hierarchical headers
in the output.
A post-processing step merged sections shorter than 50 words into the nearest adjacent section and split
sections exceeding 350 words into overlapping chunks with a 50-word overlap. Laboratory reports stored as
structured data were handled separately and excluded from text segmentation.
Document-level metadata (patient identifier, report type, report date, and document identifier) were captured
at export and stored alongside each section as structured fields, enabling retrieval queries to filter by type,
date, or patient without parsing free text. Report type labels were mapped to a controlled vocabulary of nine
canonicalcategoriesusedasfilterargumentsbytheagenticsystem’sretrievaltool. Sectionswereindexed
using FTS5 full-text search with Porter stemming, metadata fields were stored as unindexed columns to
support type- and date-based filtering without affecting relevance scoring.
A.2 Laboratory data normalisation and concept mapping
Laboratory values were extracted from the institutional SAP system as timestamped observations comprising
atestname,anumericorcategoricalresult,thereportingunit,andthelaboratory-definedreferencerange.
Becausetestnamesinthesourcesystemreflectfree-textentryconventionsthatevolvedovermorethantwo
decadesofclinicaluse,atwo-stagenormalisationpipelinewasdevelopedtoenablereliableprogrammatic
retrieval.
In the first stage, a catalogue of canonical laboratory concepts was constructed from all test names observed
across the myeloma cohort. Each unique institutional code was assigned a stable identifier derived from
a deterministic hash of its normalised form. Text normalisation comprised case folding, transliteration of
German umlauts (ä →ae,ü→ue,ö→oe,ß→ss), removal of punctuation and special characters, and
whitespace collapsing, yielding 731 unique canonical codes.
Inthesecondstage,analiasindexwasbuilttoresolvesynonymoustestnamestocanonicalcodes. Atotal
of1,054observednamevariantsweremappedto736canonicaltargetcodes,producing1,404aliasentries.
Ofthecanonicalcodes,228(31%)wereassociatedwithtwoormoresynonymousinputnames,reflecting
institutional naming changesover time, the coexistence of abbreviatedand full-length designations (e.g.crp
(c-reakt. pr)andcrp (c-reakt. protein)), and specimen-type suffixes (e.g.albumin,albumin sm,albumin l). A
set of search terms was pre-computed per canonical code from tokenised variants to support fuzzy matching
at query time.
The resulting catalogue spans myeloma-specific markers (serum and urine immunoglobulin free light chains
𝜅and𝜆,freelightchainratio,serumproteinelectrophoresis,immunofixation,immunoglobulinsG,A,M,and
18

D,𝛽2-microglobulin),prognosticparameters(lactatedehydrogenase,C-reactiveprotein,calcium,albumin,
haemoglobin, creatinine), and the broader set of routine haematological, biochemical, and coagulation tests
documented over each patient’s care. No unit conversion was applied, original test name, reporting unit, and
institutional reference range were returned alongside each numeric result. Laboratory values were indexed by
their original timestamps with no temporal windowing imposed.
A.3 Clinical question bank: development and instantiation
The clinical question bank was developed by a senior haematologist (KB) in consultation with the annotating
oncologists (AP, CN, JS, LM) prior to any system development or evaluation. Templates were drafted to
cover core clinical decision tasks in the longitudinal management of multiple myeloma, including current
and prior treatment status, treatment response assessment, toxicity and dose modification documentation,
diagnosticworkupcompletion,diseasestaging,comorbidityscoring,andtherapyeligibilitydetermination.
The initial template set was revised through iterative consensus discussion until coverage of the intended task
space was judged representative. All revisions were completed before the evaluation cohort was defined and
before any system outputs were generated.
Each template was assigned by consensus to one of three complexity levels based on the cognitive operations
required toproduce a correctanswer. Level 1 (single-recordlookup) comprised questionsanswerable from
a single document or structured data entry, such as whether the patient was currently receiving a specific
medication. Level 2 (temporal reasoning) required integration of information across multiple documents
or timepoints, such as identifying all documented treatment intervals for a given agent or determining the
bestdocumentedresponseunderaspecifiedregimen. Level3(criteria-basedsynthesis)requiredstructured
synthesis across document types, laboratory values, and clinical timepoints against a multi-criterion decision
rule, such as evaluating eligibility for BCMA-directed CAR-T cell therapy. The final bank comprised 48
templates (20 at Level 1, 18 at Level 2, and 10 at Level 3) spanning five clinical task categories (single
choice, treatment intervals, first occurrence, staging, and eligibility). The full template list is provided in
Supplementary Table B.2.
Theanswerschemaforeachtemplatedeterminedthescoringmethod. Templatesrequiringasinglecategorical
response (a yes/no/not documented answer, a single date, or a single staging score) were assigned binary
scoring. Templatesrequiringenumerationofmultipleentries(cyclestartdateswithdoses,treatmentintervals)
were assigned list-type F1 scoring, in which precision, recall, and their harmonic mean were computed over
the set of response entries against the reference list.
For each patient in the evaluation cohort, five questions were instantiated from the template bank (two at
Level1,twoatLevel2,andoneatLevel3)byrandomdrawwithoutreplacementfromashuffleddeckateach
complexity level (seed = 42). When the deck was exhausted it was reshuffled, ensuring balanced reuse across
patients. Dateplaceholderswerereplacedwiththedateofthelastavailableclinicalreportforthatpatient.
FortheMIMIC-IVcohort,de-identifieddateswereusedasrecordedinthedataset,whichappliesaconsistent
per-patientshiftthatpreservestemporalorderingandintervals. Whetheragivenquestionwasanswerable
from a particular patient’s record was determined only during expert annotation. The same instantiation
procedure was applied to both cohorts.
19

A.4 Annotation protocol
Prior to annotation, all four oncologists (AP, CN, JS, LM) received structured training from the senior
haematologist (KB) on the question bank, the answer schema conventions, the complexity-level definitions,
and the criteria for abstention. Each annotator was assigned 50 patients from the TUM evaluation cohort
by randomised balanced allocation such that each patient was reviewed by exactly two annotators and no
annotatorpairwassystematicallyoverrepresented(seed=42). FortheMIMIC-IVcohort,20patientswere
distributed between two of the same annotators using the same allocation procedure.
Annotation was performed using a purpose-built Streamlit web application that presented each question
withschema-awareinputwidgets(single-choicedropdowns, multi-intervalentryforms,criteriatableswith
per-criterion status fields, date entry, and free-text fields). Annotators were instructed to use the abstain
checkboxwhenananswerwasnotclearlysupportedbyavailabledocumentationuptothecutoffdate,and
could optionally select one or more evidence documents and provide free-text comments. All responses were
persisted to CSV logs with timestamps.
FortheTUMcohort,annotatorsreviewedpatientrecordsusingtheinstitutionalSAPinterface,whichtheyuse
routinelyinclinicalpractice. Eachquestionspecifiedacutoffdate,annotatorswereinstructedtodisregard
documentation dated after the cutoff and, for conflicting statements, to prefer the most recent documentation
before that date. For the MIMIC-IV cohort, annotators accessed an equivalent data browser providing report
viewingand structuredlaboratoryretrieval over thede-identifieddatabase. Annotatorswere blindedtoeach
other’s responses throughout, the application enforced single-user login and stored responses in separate log
files. No model-generated content was displayed at any point during annotation.
A.5 Adjudication rules and disagreement taxonomy
All patient-question pairs for which two independent annotators did not reach direct agreement were referred
to the senior haematologist (KB) for adjudication. Three case types entered adjudication: categorical
disagreements (both annotators provided a response but on differing substantive content), single-rater
abstentions(oneannotatorprovidedaresponseandtheotherabstained),anddualabstentions(bothannotators
abstained). Dual abstentions were excluded from the evaluable set without adjudication.
The adjudicator reviewed each disagreement using the same record interface and cutoff-date convention
as the original annotators. Both annotators’ responses were presented side by side, including evidence
selections and comments, and the adjudicator provided a final answer using the same schema-aware interface.
The adjudicator could adopt either annotator’s response or provide an independent answer, in single-rater
abstention cases, the non-abstaining rater’s response was independently verified rather than accepted by
default. Pairs for which the adjudicator also abstained were excluded from the evaluable set.
Each adjudicated disagreement was classified into one of five categories.Interchangeable or equivalent
responsescovered cases in which both answers were clinically synonymous or differed only in phrasing,
abbreviation, or formatting.Clinically insignificant disagreementcovered cases in which one annotator made
anerrororomissionthatwouldnotalteraclinicaldecision.Clinicallysignificantdisagreementcoveredcases
inwhichthetwoannotatorsreachedsubstantivelydifferentclinicalconclusionswithdifferentmanagement
20

implications.Ambiguousquestionorevidencecoveredcasesinwhichtherecorddidnotcontainsufficient
orunambiguousinformation,ortheinstantiatedquestionadmittedmorethanonedefensibleinterpretation.
Difference in response granularitycovered cases in which both annotators were correct but reported at
different levels of detail. Single-rater abstentions were adjudicated but not assigned a disagreement category.
Theadjudicatedanswerservedasthereferencestandardforallconcordanceanalyses,forpairswithdirect
agreement, the shared answer was used without adjudication.
A.6 Expert annotation and adjudication outcomes
FortheTUMcohort, directagreementwasreachedon326of500pairs(65 ·2%). Afurther143(28 ·6%)were
included after adjudication and 31 (6 ·2%) were excluded (29 dual abstentions, 2 adjudicator abstentions),
yielding 469 evaluable pairs in the primary analysis (200 Level 1, 179 Level 2, 90 Level 3; Figure 1e).
Pre-adjudication inter-rater agreement was 𝜅= 0·69 at Level 1 (observed agreement 83 ·3%,𝑛=192),𝜅
= 0·60 at Level 2 (observed agreement 74 ·6%,𝑛=168), and 𝜅= 0·57 at Level 3 (observed agreement
61·4%,𝑛=83) (Figure 1f). Among adjudicated cases, the most frequent disagreement categories were
interchangeable or equivalent responses (39 ·2%) and clinically insignificant disagreement (36 ·4%), with the
remainder attributable to clinically significant disagreement (8 ·4%), ambiguous question or evidence (7 ·0%),
differences in response granularity (3·5%), and single-rater abstentions (5·6%) (Figure 1g).
For the MIMIC-IV cohort, direct agreement was reached on 67 ·0% of pairs. A further 22 ·0% were included
after adjudication and 11 ·0% were excluded, yielding 89 evaluable pairs. Pre-adjudication inter-rater
agreementwas 𝜅=0·74atLevel2(observedagreement81 ·2%,𝑛=32)and 𝜅=0·73atLevel3(observed
agreement 78·6%,𝑛=14). The lower Level 1 kappa ( 𝜅= 0·34, observed agreement 75 ·0%,𝑛=40) reflects
the high prevalence of negative responses in the MIMIC-IV records (72 ·5% of Level 1 answers were “False”),
which inflates the chance-agreement baseline and suppresses𝜅even when observed agreement is high.
A.7 Agentic system: implementation details
The agentic system was implemented as a four-phase reasoning pipeline. Each phase corresponded to a
structuredprompt(termedasignature)whoseoutputwasparsedasJSONandvalidatedbeforeprogressionto
the next phase. The four phases were question assessment and skill selection, tool-use plan construction,
iterative tool execution, and final answer synthesis. A schematic overview is provided in Figure 2.
Phase I: Question assessment and skill selection.
Upon receipt of a clinical question, the system derived a short medical analysis identifying the clinical intent
and key hypotheses, a list of required and currently missing information points, and a preliminary complexity
assessment. In parallel, a subset of skill modules was selected from the indexed clinical skill library based on
questioncontentandavailableskillsummaries. Heuristickeywordmatchingservedasafallbacktoensurethat
question-type-specificstyleskillswereactivatedevenwhenthelanguagemodelomittedthem,forexample,
questionscontainingtemporalkeywords(aktuell,derzeit)triggeredthecurrent-statusstyleskill,andquestions
referencing eligibility criteria triggered the eligibility style skill. A base structured-annotation style skill was
21

always included to enforce the two-line output format.
Clinical skill library.
The skill library comprised modular instruction packages in four categories.Workflow skillsencoded
task-specific decomposition strategies and evidence priorities for clinical tasks such as therapy reconstruction,
eligibility assessment, staging score calculation, and laboratory trend interpretation.Parsing skillsprovided
normalisation rules for institutional abbreviations, German-language date formats, and entity synonymy (e.g.,
mappingtradenamestogenericdrugnames).Styleskillsdefinedanswerstructuretemplatesforeachquestion
type,includingrequiredfieldsandformattingconventions.Policyskillsencodeddeterministicprecedencerules
for resolving contradictory evidence, including temporal authority (most recent document takes precedence),
plan-versus-administeredtherapydisambiguation(administeredtherapyoverridesdocumentedplans),and
contradiction resolution heuristics.
Skills were indexed by identifier and category. The language model selected relevant skill identifiers from a
summarypromptlistingallavailableskills. Selectedskillswererenderedintotwocontextblocks: aworkflow
contextblock(workflow,parsing,andknowledgeskills)injectedduringtoolexecution,andastylecontext
block (style skills) injected during final answer synthesis. Policy skills were attached deterministically based
ontheselectedstyleandworkflowskillsratherthanbymodelselection,ensuringconsistentapplicationof
temporal authority and contradiction resolution rules regardless of model selection behaviour. Across the
evaluationcohort,anaverageof6.7skillmoduleswereselectedperquestion,correspondingtoapproximately
8,000tokensofskillcontext,comparedwithapproximately49,000tokensrequiredtoloadall41modules
unconditionally. Thecontributionoftheselectionmechanismtoaccuracyandefficiencyisreportedinthe
ablation analysis (Table B.5).
Fixed baseline prompt.
All system configurations, including ablation conditions in which the skill library was disabled, shared a
fixed baseline prompt of approximately 500 tokens. This prompt provided task framing (instructing the
model to answer clinical questions from retrieved context only), the two-line output format specification,
all answer schema definitions with worked examples covering single-choice, list, multi-interval, cycle-
start-dose-list, and criteria-table formats, citation formatting rules using bracketed context identifiers, the
distinction between “Nicht dokumentiert” (information absent from the record) and “Nein” (documented
negative finding), and drug name normalisation mappings covering drug class to substance mappings (CD38
antibodies,immunomodulatorydrugs,proteasomeinhibitors,CAR-T,andbispecificT-cellengagers),common
abbreviations(Dara,Btz,Len),andstandardregimennames(VRd,KRd,Dara-Rd,DVRd). Clinicalreasoning
workflows, temporal reasoning policies, disease-specific reference knowledge (staging criteria, response
categories), and structured retrieval strategies were provided exclusively by the skill library, ensuring that
ablation comparisons isolated the contribution of clinical reasoning protocols from formatting, schema, and
normalisation knowledge.
22

Phase II: Tool-use plan construction.
A structured tool-use plan was drafted as a JSON object containing an ordered array of steps and a set of
globalstoppingconditions. Eachstepspecifiedastepnumber,anatural-languageobjective,thetooltobe
invoked,thetoolargumentsasaJSONobject,alistofevidencerequirements,andaconditionalstoppingrule.
Uptothreeplanconstructionattemptswerepermitted,ifthelanguagemodelreturnedaninvalidorempty
plan,arepairpromptwasappendedandtheplanwasregenerated. Ifallthreeattemptsfailed, therunwas
terminated as a pipeline failure. In ablation conditions where planning was disabled, tools were selected
reactively at each execution step without a pre-drafted plan.
Phase III: Iterative tool execution.
Acrossexecutionrounds,thesystemmaintainedastructuredmemorystateasaJSONobjectencodingthe
originalquery, accumulatedevidence frompriortool calls,outstanding missinginformation,and theglobal
stopping conditions defined in the tool-use plan. This state was updated after each round and passed forward
to inform tool selection, re-query decisions, and termination.
Theplanwasexecutedstepbystep,withupto eighttool-useroundspermittedperquestion. Ateachround,
the language model received the current plan step, accumulated evidence from prior rounds, the list of
still-missinginformation,andthesetofallowedtools,thenselectedoneofthreeactions. Itcouldinvokea
specified tool, advance to the next plan step without a tool call, or terminate execution and proceed to answer
synthesis. If the model chose to terminate before executing at least one tool call for the current step and
furtherplannedstepsremained,arepairpromptrequiredittoeitherexecutetheplannedtoolorexplicitlyskip
the step.
Duplicatequeries(identicaltoolnameandquerystring)wereblockedandthemodelwasnotified. Anegative
query cache tracked report retrieval queries that returned no results, preventing re-execution of identical
unsuccessful queries. Tool results were cached within each run to avoid redundant calls with identical
arguments. An automatic adjustment mechanism progressively broadened retrieval parameters after repeated
empty results: after one failure, the number of returned results was doubled (up to a cap of 30), after two
failures,thequerywasreducedtoitskeywordsubset,afterthreefailures,temporalscoperestrictionswere
removed. A context budget of 120,000 tokens was enforced, if the estimated token count of accumulated
contextnodesexceededthislimit,retrievalwasterminatedearlyandthesystemproceededtoanswersynthesis.
Context nodes were deduplicated by section identifier before being passed to the final answer phase.
Tool specifications.
Fivetoolclasseswereavailabletotheagenticsystem. Thereportretrievaltoolperformedkeywordsearchover
the document database using BM25 ranking (BM25Okapi from the rank_bm25 library). The tool accepted a
free-textquerystring,areporttypefilterselectingfromninecanonicaldocumentcategories,atemporalscope
(allrecords,themostrecentdocument,aspecificdate,oradaterange),andcorrespondingdateparameters. A
report typesynonym mapping normalisedcommon German-language andabbreviated report typenames to
canonical values, and query strings were sanitised to remove answer-schema tokens that the language model
23

occasionally included in retrieval queries.
Thelaboratory query toolfetched structured laboratory values by canonical marker key directly from the
laboratory database table, acceptinga list of canonicallab keys withthe same temporal scope options as the
report retrieval tool. The set of available canonical keys was determined per patient at runtime and provided
tothelanguagemodelintheprompttopreventqueriesfornon-existentmarkers. Adefaultlimitoffiveresults
per marker was applied.
Four deterministic clinical scoring calculators were provided, each implemented as a stateless function with
validatedinputsandJSONoutput. TheISS(InternationalStagingSystem)assignedStageIwhenserum𝛽 2-
microglobulinwasbelow3 ·5mg/Landserumalbuminwasatleast3 ·5g/dL,StageIIIwhen 𝛽2-microglobulin
was5·5mg/Lorabove,andStageIIotherwise. TheR-ISS(RevisedISS)assignedStageIwhenISSStageI
criteria were met together with the absence of high-risk cytogenetic abnormalities (del(17p), t(4;14), or
t(14;16)) and normal LDH, Stage III when ISS Stage III criteria were met together with either high-risk
cytogenetics or elevated LDH, and Stage II otherwise. TheR2-ISS(Second Revision ISS) incorporated
additionalriskfactorsincludingtheLDH/ULNratioand1qgain/amplificationstatus,yieldingfourstages
(I–IV).TheHCT-CI(HematopoieticCellTransplantationComorbidityIndex)wascomputedfrom17boolean
comorbidity flags with weights of 1 (arrhythmia, cardiac disease, inflammatory bowel disease, diabetes
requiring medication, cerebrovascular disease, psychiatric disturbance, mild hepatic abnormality, obesity
withBMIabove35,andpersistentinfection),2(rheumatologicdisease,pepticulcer,moderate-to-severerenal
impairment,andmoderatepulmonarydisease),or3(priorsolidtumour,heartvalvedisease,severepulmonary
disease, and moderate-to-severe hepatic disease), with risk groups defined as low (score 0), intermediate
(score 1–2), and high (score≥3).
Phase IV: Final answer synthesis.
Aftertoolexecution,accumulatedcontextnodeswerededuplicatedandassignedsequentialcitationidentifiers.
The language model received the original question, patient context, a plan execution summary, the formatted
context snippets with citation identifiers, the style context from selected skills, the response requirements
derived from the question assessment, and the output of the policy engine where applicable. The policy
enginewasadeterministicmodulethatrankedretrievedevidenceitemsbytemporalauthorityandrecency,
resolvedcontradictionsbetweenplannedandadministeredtherapies,andproducedastructuredresolution
(select, abstain, or conflict) passed to the final answer signature.
The final answer was required to conform to a two-line format comprising an “Answer:” line containing
the schema-aligned value and a “Reasoning:” line containing one to two sentences with inline citation
references. If the initial output did not conform to this format, a repair prompt was appended and the answer
wasregenerated(uptotwoattempts). Ifnovalidanswerwasproducedafterallattempts,therunwasrecorded
asapipelinefailure. Citationidentifiersappearingintheanswerwerevalidatedagainstthesetofavailable
context nodes, and citations referencing non-existent sources were flagged as potentially hallucinated.
24

Decoding and inference settings.
All language models were deployed locally on a single NVIDIA H200 GPU using a vLLM inference server
with LiteLLM as the routing layer, providing an OpenAI-compatible API endpoint. The primary model
(gpt-oss-120b) was served in mxfp4 precision. Decoding parameters were set to the values recommended by
the respective model developers and held constant within each model’s evaluation, no parameter optimisation
was performed on the evaluation cohort. Run-to-run variability attributable to stochastic decoding was
characterised through the ten-run stability analysis (Supplementary Table 4).
Pipeline failure definition.
A pipeline failure was recorded when the system did not return a parseable, schema-conformant response
within the permitted retry budget (up to three attempts for tool-use plan construction and up to two for final
answer synthesis). In the classified error run, two of 469 pairs (0·4%) were classified as pipeline failures.
A.8 Comparator configurations
Simple RAG, Iterative RAG, and Full Context are described in the main text. All comparators used the same
locally deployed 120-billion-parameter language model (gpt-oss-120b), the same patient record database, and
thesamestructuredanswerpromptrequiringthetwo-lineformat. Temperaturewassetto0 ·2forallcomparator
configurations. Maximum output tokens were set to 512 for answer generation and 256 for query generation
where applicable. The embedding model used for dense retrieval was distiluse-base-multilingual-cased-v2, a
multilingual sentence transformer producing normalised embeddings for cosine similarity computation.
The LLMBaseline configurationreceived onlythe patientidentifier, thequestion text, areference date,and
the answer schema, with no record access or retrieval context of any kind. The system prompt instructed
the model to answer based solely on its parametric knowledge, establishing a floor for performance without
record access.
Advanced RAG extended Simple RAG with query rewriting into up to eight focused retrieval queries, hybrid
fusion of BM25 and dense scores (BM25 built over character trigrams with 𝑘1=1·2and𝑏=0·75 , fusion
weight 𝛼=0·5 ), and cross-encoder reranking using bge-reranker-v2-m3. The top 20 documents after
reranking were packed into the context window subject to the same 120,000-token budget. This configuration
is reported in Supplementary Table B.3 but is not included in the main comparator set.
A.9 Scoring and concordance computation
Theconcordancemetricisdefinedinthemaintext. Forsingle-valuecategoricalitems,scoringwasbinary:
a score of 1 was assigned if the system output matched the adjudicated reference in substantive content
and0otherwise. Matchingwasperformedafternormalisationofboththesystemoutputandthereference
annotation, including extraction of the answer value from the two-line output format, case-insensitive
comparison, whitespace trimming, and removal of formatting artefacts. Format deviations that did not alter
substantive content were not penalised. Semantic equivalences were applied where clinically appropriate, for
25

therapy-related questions, “nie verabreicht” (never administered) and “nicht dokumentiert” (not documented)
were treated as equivalent when the patient record contained no evidence the therapy had ever been given.
For list-type items, F1 was computed over the set of response entries against the reference list, with each
entry treated as an atomic unit. Precision was the fraction of system entries matching a reference entry, recall
wasthefractionofreferenceentriesmatchedbyasystementry,F1wastheirharmonicmean. Forlist-type
itemsprecededbyastatusfield,thestatuswasevaluatedseparately: anincorrectstatus(e.g.,“Nieverabreicht”
when reference entries existed) resulted in a score of 0 for the entire item regardless of entry-level matching.
Overall concordance was computed as the mean score across all evaluable patient-question pairs, where each
pair contributed either a binary score (0 or 1) or a continuous F1 score (0 to 1) depending on the answer
schematype. Per-questionscoreswerefirstaveragedacrosstenindependentruns,andoverallconcordance
was computed from these per-question means.
Confidence intervals and pairwise significance tests were computed using a cluster bootstrap procedure.
Whole patients were resampled with replacement, retaining all questions per sampled patient to account for
within-patientcorrelation;theconcordancestatisticwasrecomputedoneachresampleover 𝑁boot=10,000
iterations. The 2·5th and 97·5th percentiles of the bootstrap distribution defined the 95% confidence interval
(two-sided). The two-sided 𝑝-value for pairwise differences was derived as the proportion of bootstrap
resamples in which the absolute deviation of the resampled difference from its mean equalled or exceeded the
absoluteobserveddifference(shift-to-nullmethod). Bonferronicorrectionwasappliedwithineachstratum
using 𝑚=6headline pairwise comparisons, capped at 1 ·0. Results of all pairwise comparisons are reported
in Supplementary Table B.3.
A.10 Error classification protocol
Errorclassificationwasperformedbytheseniorhaematologist(KB)ontheevaluationrunwiththelowest
overall concordance among the ten independent runs (77 ·8%, compared with a 10-run mean of 79 ·6%),
selectedtoprovidethemostconservativebasisforerrorrateestimation. All115patient-questionpairsfor
which the systemoutput diverged from the adjudicatedexpert consensus, including list-typeresponses with
partial but incomplete overlap, were submitted for classification.
Classification was performed using a purpose-built review application that presented each case with the
questiontext,answer schema,theadjudicated referenceannotation, andthefullagent responsecomprising
theextracted answer, thereasoning chainwith inlinecitationidentifiers, and allretrieved sourcedocuments.
Retrieved documents were separated into those cited in the agent’s reasoning and those retrieved but not
cited, each could be expanded to display the full report text with the cited passage highlighted. The reviewer
additionally had access to the institutional SAP interface to verify claims against the primary clinical record
when needed.
Eachdivergencewasclassifiedintoexactlyoneofsixcategories.Annotationerror(categoryA)coveredcases
inwhichthesystemresponsewascorrectandtheadjudicatedreferencewaswrongorincomplete.Acceptable
or ambiguous(category B) covered cases in which both the system response and the reference were clinically
defensible given the available evidence.Partially correct(category C) covered cases in which the system
captured the correct clinical concept but with incomplete or imprecise details.Clinically insignificant error
26

(category D) covered cases in which the system response was incorrect but the error would not alter a clinical
decision.Clinicallysignificanterror(categoryE)coveredcasesinwhichthesystemresponsewasincorrectin
a way that could alter a clinical decision.Pipeline failure(category F) covered cases in which the system did
notproduceavalid,schema-conformantresponsewithinthepermittedretrybudget. Thethresholdseparating
clinically insignificant from clinically significant errors was whether a treating clinician relying on the system
response rather than the correct answer would be expected to make a different management decision.
The clinically significant error rate reported in the main text (7.0%, 33 of 469 evaluable pairs) was defined as
theproportionofcategoryEerrorsoverallevaluablepairs. Becausetheclassifiedrunhadbelow-average
concordance, this rate represents a conservative upper bound on the expected clinically significant error rate
acrossruns. Errorclassificationwasperformedbyasinglereviewerwithoutindependentsecondreview,a
limitation acknowledged in the main text.
To assess the reproducibility of the primary error classification, a blinded re-annotation sub-study was
conducted on a proportional stratified random sample of the 115 divergent patient–question pairs. The pairs
were collapsedinto threeseverity stratabeforesampling, comprisingcases classifiedas annotationerroror
acceptable and ambiguous (categories A and B), cases classified as partially correct or clinically insignificant
error (categories C and D), and cases classified as clinically significant error or pipeline failure (categories E
andF).A 40 %proportionalstratifiedrandomsamplewasdrawnfromeachstratum(seed =42),withany
rounding remainder assigned to the E+F stratum, yielding 𝑛=46cases. Cases were presented in order of
patient and question identifier to an independent rater (JS) blinded to the original classification, who applied
the samesix-category taxonomy andclinical significancethreshold used inthe primaryerror classification.
AgreementisquantifiedusingCohen’s 𝜅forboththecollapsedthree-stratumschemeandthefullsix-category
scheme, with95 %confidence intervals computed by bootstrap (𝑁 boot=10,000).
A.11 Citation sufficiency assessment
Citationsufficiencywasassessedonthesameevaluationrunusedforerrorclassification(overallconcordance
77·8%). Astratifiedsampleof96agenticsystemresponseswasdrawn,comprising16responsesfromeachof
sixcellsdefinedby thecross-classificationofcomplexitylevel (Level1,Level2, Level3)andconcordance
status (concordant, discordant), with the constraint that each of the 48 question templates contributed at least
one response. Within each cell, responses were drawn randomly from the available pool after satisfying the
template coverage constraint.
Assessmentwasperformedbytworeviewers(KB,CN),whodividedthe96casesbetweenthem. Foreach
sampledresponse,thereviewerwaspresentedwiththequestiontext,theadjudicatedreferenceannotation,and
thefullagentoutputwithinlinecitationidentifiers. Eachciteddocumentcouldbeexpandedtodisplaythe
fullreporttextwiththecitedpassagehighlighted,documentsretrievedbutnotcitedweredisplayedseparately.
The reviewer additionally had access to the institutional SAP interface.
Foreachresponse,thereviewerassessedwhetherthecitedsourcesfullysupportedtheansweronathree-point
scale:fully supported(cited evidence clearly and completely justified the answer),partially supported(cited
evidence covered some aspects but was incomplete or insufficient), ornot supported(cited evidence did not
justifytheanswer). Supportwasassessedindependentlyofanswercorrectness. Forresponsesratedaspartially
27

or not supported, the reviewer identified whether a relevant document appeared in the retrieved-but-not-cited
set, and classified the dominant failure mechanism as one of three types:reasoning failure(the relevant
documentwasretrievedbutnotusedorwasmisinterpretedduringsynthesis),retrievalfailure(therelevant
document existed in the record but was not returned by the retrieval tools), ortrue knowledge gap(the
information required to answer the question was not present in the patient record).
28

B Supplementary Tables
B.1 Supplementary Table 1: Baseline characteristics of evaluation cohorts
Demographic and disease characteristics of patients in the primary evaluation (TUM, 𝑛=100) and external
validation (MIMIC-IV, 𝑛=20) cohorts. Continuous variables are reported as median (IQR), categorical
variables as𝑛(%).
Characteristic TUM (𝑛=100) MIMIC-IV (𝑛=20)
Age at diagnosis, years 65·1 (57·3–74·0) 66·5 (58·8-77·5)
Sex (male),𝑛(%) 63 (63%) 13 (65%)
Documents per patient 50·5 (31·0–83·0) 29·0 (14·8–61·8)
Laboratory values per patient 1414 (751–2278) 3411 (1989–7411)
Follow-up span (record), years 5·5 (2·7 – 11·0) 3·3 (1·2 – 4·6)
B.2 Supplementary Table 2: Clinical question bank
All 48 question templates used for evaluation, grouped by clinical task category and complexity level.
Each template includes the question identifier, complexity level, clinical task category, template text with
placeholdervariables,expectedanswerformat,andscoringmethod(binaryorlist-typeF1). [date]denotes
the date of the last available report. ND=not documented.
29

ID L Category Template text Answer format Scoring
Level 1 – Simple
Q011 Single choice Is the patient receiving lenalidomide on
[date]?Yes / No / ND / Unclear Binary
Q021 Single choice Is the patient receiving bortezomib on
[date]?Yes / No / ND / Unclear Binary
Q031 Single choice Is the patient receiving daratumumab on
[date]?Yes / No / ND / Unclear Binary
Q041 Single choice Is there documented evidence that a whole-
body MRI has been performed?Yes / No / Unclear Binary
Q051 Single choice Is there documented evidence that a PET-
CT has been performed?Yes / No / Unclear Binary
Q061 Single choice Is there documented evidence that FISH
has been performed?Yes / No / Unclear Binary
Q071 Single choice Has the patient received at least one CD38-
antibody therapy?Yes / No / Unclear Binary
Q081 Single choice HasthepatientreceivedatleastoneIMiD
therapy?Yes / No / Unclear Binary
Q091 Single choice Hasthepatientreceivedatleastoneprotea-
some inhibitor?Yes / No / Unclear Binary
Q101 Single choice Has the patient received an autologous
SCT?Yes / No / Unclear Binary
Q111 Single choice HasthepatientreceivedanallogeneicSCT? Yes / No / Unclear Binary
Q121 Single choice Has the patient received CAR-T therapy? Yes / No / Unclear Binary
Q131 Single choice Has the patient received BiTE therapy? Yes / No / Unclear Binary
Q141 Single choice Has the patient received dialysis? Yes / No / Unclear Binary
Q151 Single choice Has the patient been diagnosed with sepsis
or septic shock?Yes / No / Unclear Binary
30

ID L Category Template text Answer format Scoring
Q161 Single choice Has the patient been mechanically venti-
lated (invasive or non-invasive)?Yes / No / Unclear Binary
Q171 Single choice Is renal failure / AKI documented? Yes / No / ND / Unclear Binary
Q181 Single choice Is clinically relevant anaemia present on
[date] (documented or lab-confirmed)?Yes / No / ND / Unclear Binary
Q191 Single choice Hasthepatientreceivedredbloodcelltrans-
fusions?Yes / No / Unclear Binary
Q201 Single choice Hasthepatientbeendiagnosedwithpneu-
monia?Yes / No / Unclear Binary
Level 2 – Medium
Q212Treatment in-
tervalsWhichcyclestartdates(C*D1)anddoses
for bortezomib are documented?≤12×(date; dose; unit) List-type
F1
Q222Treatment in-
tervalsWhichcyclestartdates(C*D1)anddoses
for lenalidomide are documented?≤12×(date; dose; unit) List-type
F1
Q232Treatment in-
tervalsWhich melphalan exposures are docu-
mented (type, approximate date, dose per
episode)?≤3×(type; date; dose; unit) List-type
F1
Q242Treatment in-
tervalsWhich doxorubicin exposures are docu-
mented (form, approximate date, cumu-
lative dose per episode)?≤3×(form; date; dose; unit) List-type
F1
Q252Treatment in-
tervalsInwhichdocumentedintervalswascarfil-
zomibadministered(start–endorongoing)?≤3×(start–end / ongoing) List-type
F1
Q262Treatment in-
tervalsInwhich documented intervalswaspoma-
lidomideadministered(start–endorongo-
ing)?≤3×(start–end / ongoing) List-type
F1
31

ID L Category Template text Answer format Scoring
Q272Treatment in-
tervalsIn which documented intervals was
meropenemadministered(start–endoron-
going)?≤3×(start–end / ongoing) List-type
F1
Q282 Single choice Wastoxicity,dosereduction,ordiscontin-
uation documented under carfilzomib? If
yes, reason?Yes / No / Unclear + free
textBinary
Q292 Single choice Was toxicity, dose reduction, or discontinu-
ation documented under pomalidomide? If
yes, reason?Yes / No / Unclear + free
textBinary
Q302 Single choice Best documented response under isatux-
imab + Pom-Dex?CR/VGPR/PR/SD/PD /
Never / ND / UnclearBinary
Q312 Single choice Best documented response under
KRD (carfilzomib–lenalidomide–
dexamethasone)?CR/VGPR/PR/SD/PD /
Never / ND / UnclearBinary
Q322 Single choice Best documented response under the docu-
mented first-line therapy?CR/VGPR/PR/SD/PD /
Never / ND / UnclearBinary
Q332 Single choice Bestdocumented responseafter high-dose
melphalan?CR/VGPR/PR/SD/PD /
Never / ND / UnclearBinary
Q342 Single choice Best documented response under BiTE? CR/VGPR/PR/SD/PD /
Never / ND / UnclearBinary
Q352 Single choice Best documented response under CAR-T? CR/VGPR/PR/SD/PD /
Never / ND / UnclearBinary
Q362First occur-
renceFirstdocumentedCTreportdescribingnew
osteolytic lesions?Date / ND / Unclear Binary
Q372First occur-
renceFirst documented episode of renal failure
or dialysis-requiring AKI?Date / ND / Unclear Binary
32

ID L Category Template text Answer format Scoring
Q382 Staging ISS stage on [date] calculated from albu-
min and𝛽 2M (last labs≤90days prior)?Score (I/II/III) + date +
sourceBinary
Level 3 – Complex
Q393 Staging ECOG score on [date] (last documented
≤180days, else inferred from explicit func-
tional description)?Score (0–4) + date + source Binary
Q403 Staging HCT-CIscoreon [date](lastdocumented
≤365days, else derived from comorbidi-
ties)?Score + date + source Binary
Q413 Staging R-ISSstageon[date](labs ≤90days; cyto-
genetics from last available report)?Stage (I/II/III) + date +
sourceBinary
Q423 Staging R2-ISS stage on [date] (labs ≤90days; cy-
togeneticsincl.1qstatusfromlastavailable
report)?Stage (I/II/III/IV) + date +
sourceBinary
Q433 Single choice Is the patient triple-class refractory on
[date]? (with documented justification)Yes / No / Unclear + free
textBinary
Q443 Single choice Is the patient quadruple-class refractory on
[date]? (with documented justification)Yes / No / Unclear + free
textBinary
Q453 Single choice IsthepatienteligibleforASCTon[date]
basedonage,comorbidities,performance
status, and organ function?Yes / No / Unclear + free
textBinary
Q463 Single choice Which therapy achieved the best docu-
mentedresponseupto[date](perIMWG
criteria)?Free text (therapy) +
CR/VGPR/PR/SD/PD / ND
/ UnclearBinary
Q473 Single choice Which risks currently dominate: disease
progression or therapy toxicity?Progression/Toxicity/Both
/ Unclear + free textBinary
33

ID L Category Template text Answer format Scoring
Q483 Eligibility Which BCMA-CAR-T eligibility criteria
are met / not met / missing on [date]? Is
the patient eligible overall?Criteria table (met/not
met/missing) + Yes / No /
UnclearBinary
34

B.3Supplementary Table 3: Pairwise cluster bootstrap significance tests for concordance differ-
ences
Allpairwisecomparisonsbetweensystemconfigurations,overallandstratifiedbycomplexitylevel. Differences
are reported as System A minus System B in percentage points; a negative value indicates that System B
outperformed System A. Confidence intervals and 𝑝-values were obtained by cluster bootstrap with 𝑁boot=
10,000resamples,resamplingwholepatientswithreplacementandretainingallquestionspersampledpatient
to account for within-patient correlation. The N column reports the number of patients contributing to
eachstratum. Bonferroni-corrected 𝑝-valueswerecomputedbymultiplyingraw 𝑝-valuesbythenumberof
pairwisecomparisonswithineachstratum(6headlinecomparisons),cappedat1 ·0. Significancecodesreflect
Bonferroni-corrected𝑝-values:∗∗∗𝑝 <0·001;∗∗𝑝 <0·01;∗𝑝 <0·05; ns𝑝≥0·05.
Subset Comparison (A vs B) N patients Diff A−B (pp) 95% CI Raw𝑝Bonf.𝑝
Overall
Overall Baseline vs Simple RAG 100−70·22 [−74·00,−66·44]<0·001<0·001∗∗∗
Overall Baseline vs Advanced RAG 100−74·81 [−77·93,−71·52]<0·001<0·001∗∗∗
Overall Baseline vs Iterative RAG 100−74·14 [−77·21,−70·96]<0·001<0·001∗∗∗
Overall Baseline vs Full Context 100−74·55 [−77·81,−71·17]<0·001<0·001∗∗∗
Overall Baseline vs Agentic System 100−78·31 [−81·37,−75·20]<0·001<0·001∗∗∗
Overall Simple RAG vs Advanced RAG 100−4·58 [−6·99,−2·22]<0·001 0·0006∗∗∗
Overall Simple RAG vs Iterative RAG 100−3·92 [−6·46,−1·38] 0·0024 0·0144∗∗
Overall Simple RAG vs Full Context 100−4·33 [−6·80,−1·95] 0·0003 0·0018∗∗∗
Overall Simple RAG vs Agentic System 100−8·09 [−11·25,−4·93]<0·001<0·001∗∗∗
Overall Advanced RAG vs Iterative RAG 100+0·67 [−0·64,+1·94] 0·3083 1·000 ns
Overall Advanced RAG vs Full Context 100+0·26 [−1·72,+2·18] 0·7907 1·000 ns
Overall Advanced RAG vs Agentic 100−3·51 [−5·80,−1·26] 0·0025 0·0150∗∗
Overall Iterative RAG vs Full Context 100−0·41 [−2·50,+1·64] 0·6938 1·000 ns
Overall Iterative RAG vs Agentic 100−4·17 [−6·68,−1·78] 0·0012 0·0072∗∗
Overall Full Context vs Agentic 100−3·76 [−6·13,−1·47] 0·0010 0·0060∗∗
Level 1 Baseline vs Simple RAG 100−78·90 [−83·55,−74·05]<0·001<0·001∗∗∗
Level 1 Baseline vs Advanced RAG 100−84·50 [−88·65,−80·00]<0·001<0·001∗∗∗
Level 1 Baseline vs Iterative RAG 100−82·25 [−86·65,−77·45]<0·001<0·001∗∗∗
Level 1 Baseline vs Full Context 100−84·90 [−89·00,−80·60]<0·001<0·001∗∗∗
Level 1 Baseline vs Agentic System 100−86·00 [−90·15,−81·60]<0·001<0·001∗∗∗
Level 1 Simple RAG vs Advanced RAG 100−5·60 [−9·15,−2·10] 0·0025 0·0150∗∗
Level 1 Simple RAG vs Iterative RAG 100−3·35 [−7·30,+0·45] 0·0911 0·5466 ns
Level 1 Simple RAG vs Full Context 100−6·00 [−9·60,−2·75] 0·0012 0·0072∗∗
Level 1 Simple RAG vs Agentic 100−7·10 [−11·65,−2·50] 0·0020 0·0120∗∗
Level 1 Advanced RAG vs Iterative RAG 100+2·25 [−0·10,+4·60] 0·0615 0·3690 ns
Level 1 Advanced RAG vs Full Context 100−0·40 [−3·35,+2·50] 0·7878 1·000 ns
Level 1 Advanced RAG vs Agentic 100−1·50 [−4·95,+1·90] 0·3907 1·000 ns
Level 1 Iterative RAG vs Full Context 100−2·65 [−5·95,+0·60] 0·1134 0·6804 ns
Level 1 Iterative RAG vs Agentic 100−3·75 [−7·80,+0·40] 0·0733 0·4398 ns
Level 1 Full Context vs Agentic 100−1·10 [−4·35,+2·25] 0·5078 1·000 ns
35

Supplementary Table 3 (continued):Pairwise cluster bootstrap significance tests for concordance differences
Subset Comparison (A vs B) N patients Diff A−B (pp) 95% CI Raw𝑝Bonf.𝑝
Level 2 Baseline vs Simple RAG 100−72·65 [−78·74,−66·40]<0·001<0·001∗∗∗
Level 2 Baseline vs Advanced RAG 100−75·89 [−81·22,−70·41]<0·001<0·001∗∗∗
Level 2 Baseline vs Iterative RAG 100−75·87 [−80·90,−70·57]<0·001<0·001∗∗∗
Level 2 Baseline vs Full Context 100−73·88 [−79·84,−67·74]<0·001<0·001∗∗∗
Level 2 Baseline vs Agentic 100−77·76 [−82·79,−72·40]<0·001<0·001∗∗∗
Level 2 Simple RAG vs Advanced RAG 100−3·24 [−7·28,+0·47] 0·0981 0·5886 ns
Level 2 Simple RAG vs Iterative RAG 100−3·22 [−7·11,+0·55] 0·1019 0·6114 ns
Level 2 Simple RAG vs Full Context 100−1·23 [−5·15,+2·63] 0·5401 1·000 ns
Level 2 Simple RAG vs Agentic 100−5·11 [−8·76,−1·56] 0·0058 0·0348∗∗
Level 2 Advanced RAG vs Iterative RAG 100+0·01 [−1·37,+1·36] 0·9859 1·000 ns
Level 2 Advanced RAG vs Full Context 100+2·01 [−1·32,+5·58] 0·2474 1·000 ns
Level 2 Advanced RAG vs Agentic 100−1·87 [−4·57,+0·81] 0·1645 0·9870 ns
Level 2 Iterative RAG vs Full Context 100+1·99 [−1·27,+5·34] 0·2354 1·000 ns
Level 2 Iterative RAG vs Agentic 100−1·89 [−4·52,+0·69] 0·1515 0·9090 ns
Level 2 Full Context vs Agentic 100−3·88 [−7·78,−0·32] 0·0418 0·2508 ns
Level 3
Level 3 Baseline vs Simple RAG 90−46·11 [−54·78,−37·67]<0·001<0·001∗∗∗
Level 3 Baseline vs Advanced RAG 90−51·11 [−59·11,−43·11]<0·001<0·001∗∗∗
Level 3 Baseline vs Iterative RAG 90−52·67 [−60·78,−44·44]<0·001<0·001∗∗∗
Level 3 Baseline vs Full Context 90−52·89 [−61·67,−44·22]<0·001<0·001∗∗∗
Level 3 Baseline vs Agentic 90−62·33 [−71·00,−53·11]<0·001<0·001∗∗∗
Level 3 Simple RAG vs Advanced RAG 90−5·00 [−9·56,−0·55] 0·0295 0·1770 ns
Level 3 Simple RAG vs Iterative RAG 90−6·56 [−11·44,−1·89] 0·0078 0·0468∗∗
Level 3 Simple RAG vs Full Context 90−6·78 [−13·22,−0·22] 0·0420 0·2520 ns
Level 3 Simple RAG vs Agentic 90−16·22 [−25·11,−7·11] 0·0004 0·0024∗∗∗
Level 3 Advanced RAG vs Iterative RAG 90−1·56 [−4·56,+1·22] 0·2847 1·000 ns
Level 3 Advanced RAG vs Full Context 90−1·78 [−6·44,+3·00] 0·4521 1·000 ns
Level 3 Advanced RAG vs Agentic 90−11·22 [−18·44,−4·00] 0·0020 0·0120∗∗
Level 3 Iterative RAG vs Full Context 90−0·22 [−5·33,+5·11] 0·9310 1·000 ns
Level 3 Iterative RAG vs Agentic 90−9·67 [−16·67,−2·67] 0·0082 0·0492∗∗
Level 3 Full Context vs Agentic 90−9·44 [−16·22,−2·67] 0·0054 0·0324∗∗
36

B.4 Supplementary Table 4: Run-to-run stability across system configurations
Concordance estimates across ten independent evaluation runs for each system configuration on the primary
evaluationcohort(TUM, 𝑛=469). Resultsarereportedasmean ±standarddeviationacrossruns,overall
and stratified by complexity level. Individual run results are listed to characterise the distribution.
System Overall Level 1 Level 2 Level 3
Baseline 1·3±0·5% 0·1±0·2% 1·8±1·0% 2·8±1·1%
Simple RAG System 71·5±0·9% 79·0±1·7% 74·4±0·9% 48·9±2·7%
Advanced RAG System 76·1±1·7% 84·7±1·5% 77·7±1·3% 53·9±4·5%
Iterative RAG System 75·4±2·2% 82·4±2·0% 77·6±3·0% 55·4±3·0%
Full Context 75·8±0·5% 85·0±1·0% 75·7±1·1% 55·7±1·8%
Agentic System 79·6±1·1% 86·2±1·1% 79·5±2·4% 65·1±2·2%
B.5 Supplementary Table 5: System ablations
Concordance across ablated system configurations on the primary evaluation cohort (TUM, 𝑛=469).
Values are mean concordance (%) across the number of independent runs indicated, overall and stratified by
complexity level. Each row removes or replaces one or more components of the full agentic system. The
unfiltered skill library condition loads all 41 skill modules unconditionally rather than selecting by query
content. Configurations without a skill library use the same fixed baseline prompt as the comparator systems,
without the agent-specific reasoning skills.
Configuration Overall Level 1 Level 2 Level 3
Standard configuration 79·6% 86·2% 79·5% 65·1%
No deterministic clinical scoring tools 79·6% 85·1% 80·0% 66·7%
No type and date filters in retrieval tools 79·5% 87·7% 78·1% 63·8%
Full skill library, unfiltered 79·3% 87·7% 78·7% 62·0%
No structured memory state 79·2% 85·9% 79·0% 64·9%
Reactive tool selection (no pre-planned use) 79·2% 86·6% 78·7% 63·8%
No skill library 76·6% 86·0% 77·9% 53·3%
B.6 Supplementary Table 6: System performance across language model backbones
All four system configurations were evaluated under each backbone model on the primary evaluation cohort
(469 patient-question pairs, TUM). Each model was deployed locally via vLLM using the inference settings
recommendedbytherespectivedeveloper. Weightquantizationreflectstheprecisionoftheloadedmodel
checkpoint(MXFP4orFP8),Qwen3-Next-80B-A3B-InstructadditionallyusedanFP8KVcache. Overall
concordance (%) is reported as mean across ten runs. The model used in the primary analysis is indicated (∗).
37

Model Params Quantization Simple RAG Iterative RAG Full Context Agentic
gpt-oss-120b∗120B MXFP4 71·5% 75·4% 75·8% 79·6%
GLM-4.5-Air 110B FP8 64·9% 68·2% 72·0% 77·9%
Qwen3-Next-80B-A3B-Instruct 81B FP8 75·0% 76·5% 76·1% 76·8%
gemma-4-31B-it 33B FP8 74·7% 82·4% 81·3% 81·6%
gpt-oss-20b 21B MXFP4 67·9% 72·1% 68·4% 77·4%
B.7 Supplementary Table 7: Concordance by clinical task category and system configuration
Concordance stratified by clinical task category (single choice, treatment intervals, first occurrence, staging,
eligibility) for each system configuration on the primary evaluation cohort (TUM, 𝑛=469). Values are mean
concordance across ten independent runs with 95% bootstrap confidence intervals (cluster bootstrap, 10,000
resamples, patient-level resampling). Category sample sizes: single choice 𝑛=326, treatment intervals
𝑛=70,firstoccurrence 𝑛=20,staging 𝑛=45,eligibility 𝑛=8. Resultsfortheeligibilitycategoryarebased
on a small sample and should be considered hypothesis-generating only.
System Single choice Treatment intervals First occurrence Staging Eligibility
Baseline 0·1 [0·0–0·2] 2·6 [1·7–3·6] 0·0 [0·0–0·0] 8·4 [6·2–10·9] 0·0 [0·0–0·0]
Simple RAG 76·9 [76·3–77·5] 74·2 [73·0–75·2] 38·0 [35·5–40·5] 55·6 [54·7–56·4] 1·2 [0·0–3·8]
Advanced RAG 81·8 [81·0–82·8] 74·6 [73·5–75·6] 52·5 [47·5–57·0] 56·4 [55·3–57·6] 25·0 [16·2–33·8]
Iterative RAG 80·9 [79·1–82·1] 74·6 [73·0–76·0] 52·0 [48·5–55·5] 55·3 [53·3–57·1] 32·5 [26·2–40·0]
Full Context 83·5 [82·9–84·0] 70·9 [70·1–71·7] 40·0 [37·5–42·5] 53·6 [52·0–55·3] 21·2 [13·8–28·7]
Agentic System 86·4 [85·7–87·0] 76·1 [74·9–77·3] 50·5 [46·0–55·0] 52·4 [50·4–54·7] 58·8 [53·8–63·7]
B.8Supplementary Table 8: Execution characteristics by complexity level and system configura-
tion
Wall-clocktimeperquestionontheprimaryevaluationcohort(TUM, 𝑛=469patient-questionpairsfrom
100 patients), stratified by complexity level and system configuration. Unless otherwise indicated, all values
are median (IQR) across all patient-question pairs from a single-concurrency timing run on a single NVIDIA
H200 GPU. Skills selected, tool calls, and document retrieval metrics are reported for the agentic system only.
38

Level 1 Level 2 Level 3 Overall
Metric(𝑛=200) (𝑛=179) (𝑛=90) (𝑛=469)
Agentic system
Skills selected per question 6·0 (6·0–7·0) 6·0 (6·0–7·0) 8·0 (7·0–9·0) 7·0 (6·0–7·0)
Tool calls per question 2·0 (1·0–3·0) 3·0 (2·0–4·0) 4·0 (3·0–4·0) 3·0 (2·0–4·0)
Documents retrieved 15·0 (7·5–26·0) 27·0 (14·0–49·0) 27·0 (16·0–46·0) 20·0 (10·0–39·0)
Documents cited 2·0 (2·0–2·0) 2·0 (2·0–2·0) 2·0 (2·0–4·0) 2·0 (2·0–2·0)
of which≥3 documents (%) 5·9 9·5 47·4 15·2
Wall-clock time per question, s 18·5 (16·7–20·6) 21·3 (19·4–23·8) 28·7 (26·3–34·2) 20·9 (18·2–25·2)
Iterative RAG
Wall-clock time per question, s 6·0 (5·0–7·3) 6·4 (5·1–8·6) 12·3 (7·2–23·4) 6·6 (5·3–9·5)
Full Context
Wall-clock time per question, s 4·7 (3·8–5·6) 5·0 (4·0–6·0) 6·0 (5·1–6·9) 5·0 (4·0–6·0)
Simple RAG
Wall-clock time per question, s 2·2 (1·8–2·8) 2·2 (1·8–2·8) 2·7 (2·2–3·1) 2·3 (1·8–2·9)
B.9 Supplementary Table 9: Patient stratification by record length
Patientswerestratifiedbytotalclinicalrecordlength(charactercountofthefullconcatenatedrecord)intofour
bins for the context-length sensitivity analysis. Three bins span the lower 90th percentile of the distribution
(Q1–Q3),definedatthe33rdand67thpercentiles,thefourthbin(Q4)comprisesthetopdecile. Evaluable
pairs are summed across all complexity levels.
Bin Character range Patients (𝑛) Evaluable pairs (𝑛)
Q1 (≤p33)≤127k 33 155
Q2 (p33–p67) 127k–282k 34 160
Q3 (p67–p90) 282k–541k 23 107
Q4 (>p90)>541k 10 47
Total≤1,076k 100 469
39

C Supplementary Figures
C.1 Supplementary Figure 1: Accuracy by patient sex, vital status, and age group
Simple RAGAdvanced RAG Iterative RAGAgentic System020406080100Accuracy (%)
All levels
Simple RAGAdvanced RAG Iterative RAGAgentic System020406080100Accuracy (%)
Level 1
Simple RAGAdvanced RAG Iterative RAGAgentic System020406080100Accuracy (%)
Level 2
Simple RAGAdvanced RAG Iterative RAGAgentic System020406080100Accuracy (%)
Level 3
Accuracy by patient age group (quartiles, ref. 06 Jan 2026)  —  bootstrap 95% CI
Age group
≤64 yr
 64–74 yr
 74–83 yr
 >83 yr
SimpleRAG
Advanced RAG Iterative RAGAgentic System020406080100Accuracy (%)
All levels
SimpleRAG
Advanced RAG Iterative RAGAgentic System020406080100Accuracy (%)
Level 1
Simple RAG
Advanced RAG Iterative RAGAgentic System020406080100Accuracy (%)
Level 2
Simple RAGAdvanced RAG Iterative RAGAgentic System020406080100Accuracy (%)
Level 3
Accuracy by patient sex  (bootstrap 95% CI)
Male
 Female
SimpleRAG
Advanced RAG Iterative RAGAgentic System020406080100Accuracy (%)
All levels
SimpleRAG
Advanced RAG Iterative RAGAgentic System020406080100Accuracy (%)
Level 1
Simple RAGAdvanced RAG Iterative RAGAgentic System020406080100Accuracy (%)
Level 2
Simple RAGAdvanced RAG Iterative RAGAgentic System020406080100Accuracy (%)
Level 3
Accuracy by patient vital status  (bootstrap 95% CI)
Alive
 Deceased
FigureS1:Concordancewithexpertconsensusstratifiedbypatientsex(male,female),vitalstatus(alive,deceased),
andage groupatthereferencedate(quartiles: ≤64,64–74, 74–83, >83years), shownoverall andbycomplexity level
for all four system configurations. Bootstrap 95% confidence intervals were computed by cluster bootstrap resampling
at the patient level (𝑁 boot=10,000).
40

C.2 Supplementary Figure 2: Accuracy by question template across different systems
Q01Q02Q03Q04Q05Q06Q07Q08Q09Q10Q11Q12Q13Q14Q15Q16Q17Q18Q19Q20Q21Q22Q23Q24Q25Q26Q27Q28Q29Q30Q31Q32Q33Q34Q35Q36Q37Q38Q39Q40Q41Q42Q43Q44Q45Q46Q47Q48
Question ID00.20.40.60.81.0Average accuracyLevel 1 – Simple Level 2 – Medium Level 3 – Complex
Per-question accuracy by system
RAG System
 Iterative RAG
 Full Context
 Agentic System
Single choice
 T reatment intervals
 First occurrence
 Staging
 Eligibility
Figure S2:Per-question concordance averaged across ten independent evaluation runs for each of the 48 clinical
question templates, stratified by system configuration and clinical task category (single choice, treatment intervals, first
occurrence, staging, eligibility). Templates are ordered by complexity level (Level 1, Level 2, Level 3) and grouped by
task category within each level.
41
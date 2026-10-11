# From Retrieval to Customer Context: Evaluating Frontier-Model Systems for Voice-of-Customer Analysis

**Authors**: Raviraja G, Viraj Bagal, Prabhath Chellingi

**Published**: 2026-10-07 03:30:22

**PDF URL**: [https://arxiv.org/pdf/2610.09375v1](https://arxiv.org/pdf/2610.09375v1)

## Abstract
Organizations increasingly use frontier language models to analyze customer feedback, but answer quality also depends on how that feedback is organized and made available. We define a \emph{customer context graph} as a unified model of customer and business context. Typed relationships connect customer objects (feedback, conversations, users, and accounts), operational objects (tickets, support agents, opportunities, and competitors), and analytical or action objects (taxonomy concepts, evidence, insights, work items, and outcomes). This lets an agent investigate not only what customers say, but why, who is affected, what action followed, who owns it, and whether it was resolved. For this experiment, the graph is populated from public Cursor feedback; the same architecture can support any type of feedback source. We compare Agentic RAG, a Deep Research Agent, and a Customer Context Graph-backed Agent on the same 9,432 public Cursor feedback records using 30 realistic product, incident, comparison, and metadata questions. Without exhaustive ground truth, we jointly score responses on answer quality (coverage and organization), analytical depth (specificity and decomposition), and evidence quality (citation support and traceability), using a comparative rubric calibrated on 28 of the 30 questions. We sample cited records against their claims and weight the three dimensions equally. Under this aligned rubric, the Customer Context Graph-backed Agent scores 0.961 overall, versus 0.710 for the Deep Research Agent and 0.651 for Agentic RAG, and leads the Deep Research Agent on 27 of 30 paired questions (sign-test p < 10^(-5); strictly best on 26 of 30). Its largest advantage is analytical depth (0.967 versus 0.642), reflecting more specific, hierarchically developed findings with quantified themes and traceable evidence...

## Full Text


<!-- PDF content starts -->

From Retrieval to Customer Context:
Evaluating Frontier-Model Systems for Voice-of-Customer Analysis
Raviraja G, Viraj Bagal, Prabhath Chellingi
Enterpret
Abstract
Organizations increasingly use frontier language models to
analyze customer feedback, but answer quality also depends
onhowthatfeedbackisorganizedandmadeavailable.Wede-
fine acustomer context graphas a unified model of customer
and business context. Typed relationships connect customer
objects (feedback, conversations, users, and accounts), op-
erational objects (tickets, support agents, opportunities, and
competitors),andanalyticaloractionobjects(taxonomycon-
cepts,evidence,insights,workitems,andoutcomes).Thislets
anagentinvestigatenotonlywhatcustomerssay,butwhy,who
is affected, what action followed, who owns it, and whether
it was resolved. For this experiment, the graph is populated
from public Cursor feedback; the same architecture can sup-
portanytypeoffeedbacksource.WecompareAgenticRAG,a
DeepResearchAgent,andaCustomerContextGraph-backed
Agent on the same 9,432 public Cursor feedback records us-
ing 30 realistic product, incident, comparison, and metadata
questions.
Without exhaustive ground truth, we jointly score responses
on answer quality (coverage and organization), analytical
depth (specificity and decomposition), and evidence qual-
ity (citation support and traceability), using a comparative
rubric calibrated on 28 of the 30 questions. We sample cited
records against their claims and weight the three dimensions
equally. Under this aligned rubric, the Customer Context
Graph-backed Agent scores 0.961 overall, versus 0.710 for
the Deep Research Agent and 0.651 for Agentic RAG, and
leads the Deep Research Agent on 27 of 30 paired questions
(sign-testp <10−5; strictly best on 26 of 30). Its largest
advantage is analytical depth (0.967 versus 0.642), reflecting
more specific, hierarchically developed findings with quanti-
fied themes and traceable evidence. Because the rubric was
calibratedontheseoutputs,wereporttheresultasexploratory:
therankingisinternallyconsistentandsurvivesthetwoques-
tions withheld from calibration, but a confirmatory pass on
freshly authored questions scored under the frozen rubric is
outstandingandisthevalidationweconsiderdecisive.Within
this frozen configuration, frontier-model agents grounded in
a persistent customer context graph produce higher-quality
customer-feedback analysis than agents analyzing the same
feedback without that structured context.
Keywords:customerfeedbackanalysis;retrieval-augmented
generation; language-model agents; voice of customer; evi-
dence evaluation1 Introduction
Voice-of-customer(VoC)analysisturnsqualitativefeedback
into a structured account of customer needs, their relation-
ships,andtheirpriorities(GriffinandHauser1993).Asmore
knowledgeworkisautomated,afluentanswerisnotenough:
theanalysisunderneathitmustbecorrect,reproducible,and
traceable, or every downstream product, engineering, and
support decision inherits an unstable understanding of the
customer.Frontierlanguagemodelsmakerawfeedbackeas-
iertoquery,butausefulVoCanswerstillhastocombinean
exact denominator, semantic grouping, comparisons across
time or versions, representative quotations, and a defensible
statement about what the data cannot establish. A retrieved
passage supports a qualitative observation but a top-kre-
trievalresultcannotestablishcorpusprevalence,andafluent
summarycanciterelevantrecordswhileattachingthewrong
countorclaimingacustomersegmentabsentfromthesource
data.
Retrieval-augmentedgeneration(RAG)givesamodelac-
cesstoexternalevidence(Lewisetal.2020;Karpukhinetal.
2020). More capable query-time systems interleave reason-
ing with retrieval and tool use (Yao et al. 2023; Trivedi
et al. 2023; Asai et al. 2024) or delegate subtasks to spe-
cialist agents. A different architectural choice is to organize
customer context before the question is asked: summarize
feedback,mapittoasharedmetadataschema,predictrecord-
leveltaxonomylabels,retainavailableuserinformation,and
connecttheseentitiesbacktoeachsourcerecord.Thispaper
compares those choices directly on 9,432 public feedback
records about Cursor—the AI coding assistant—collected
from Reddit developer subreddits, the Apple App Store re-
viewstream,andTrustpilotoverthefrozenthree-monthUTC
window 2026-04-16 to 2026-07-15 (Table 2). Records mix
shortmobileappreviews,multi-paragraphRedditposts,and
Trustpilot complaints, and are collected without knowing
which internal product concepts (billing, agent mode, Auto
routing, GLM 5.2 tool calls, Composer) will be relevant to
anyanalystquestion.Thecomparisondeliberatelyexcludesa
naive “put all reviews in the prompt” baseline: long-context
models do not use every input position equally (Liu et al.
2024) and the target corpus exceeds a single request’s prac-
tical context capacity.
Weask:howdoescontextarchitectureaffectanswerqual-
ity, analytical depth, quantification, evidence quality, and
arXiv:2610.09375v1  [cs.LG]  7 Oct 2026

epistemic restraint on unsupported groupings (RQ1–RQ5),
and how much of any quality gap can stronger query-time
planning, retrieval, and evidence review close (RQ6)?
Contribution.Our primary contribution is a controlled
three-system comparison establishing that persistent cus-
tomercontext,notadditionalquery-timeagency,istheload-
bearing lever for LLM-based VoC analysis on this corpus
and under an analyst-aligned rubric. Supporting contribu-
tions are (i) the 30-query benchmark and its evaluator-only
family invariants; (ii) a common answer-package contract
that retains final prose and resolvable citation membership;
and(iii)athree-partautomaticevaluationofanswerquality,
analytical depth, and evidence quality with pairedW/T/L
as the primary statistic.
Scope.Thereportedcomparisonisscopedinfourwaysthe
readershouldcarrythrough:(1)scorescomefromacompar-
ativerubricalignedwithanalystpreferencesafterinspecting
evaluated answers on these 30 questions (Section 4, Sec-
tion 7); (2) the comparison is complete-system rather than
component-level; (3) the study uses one product corpus and
one three-month frozen UTC window; and (4) holding the
base analyst model constant does not equalize model capa-
bilityacrosscandidatesbecauseeachsystemdispatchesthat
model to structurally different roles (supplementary mate-
rial).
2 Related Work
Voice-of-customer and app-review analysis.Classical
VoC work separates identifying customer needs, structuring
thoseneeds,andestimatingtheirpriority(GriffinandHauser
1993). That distinction remains useful for LLM-based anal-
ysis: extracting representative statements, organizing them
intoconcepts,andestimatingprevalencearerelatedbutnon-
equivalent tasks. Prior app-review work classifies reviews
intomaintenance-relevantcategoriessuchasbugs,featurere-
quests,userexperiences,andratings(Panichellaetal.2015;
Maalejetal.2016);asystematicreviewcharacterizesappre-
viewsasshort,error-prone,dynamic,anddomain-dependent
(Wang et al. 2024); production summarization uses multi-
stage extraction, topic grouping, and human evaluation to
balance freshness, diversity, and accuracy (Apple Machine
LearningResearch2025).ArecentVoCbenchmarkfindsthat
feature-labeled and sentiment-aware RAG variants improve
consistencyandcoverageofgeneratedproductrequirements
(Fiocco,Proietti,andCesarotti2025).Ourfocusdiffersfrom
classificationandone-shotsummarization:wecomparecom-
plete analytical systems over linked questions that require
exhaustivecounts,repeatedconceptuse,evidencereconcili-
ation, and downstream decisions.
RAG, agentic retrieval, and deep-research paradigms.
RAG combines parametric generation with retrieved non-
parametric evidence (Lewis et al. 2020); dense passage
retrieval provides a standard candidate-context mechanism
(Karpukhin et al. 2020). Retrieval remains relevant as con-
text windows grow because models can underuse evidence
placedinthemiddleoflonginputs(Liuetal.2024).Agentic
and iterative retrieval interleave reasoning with action (Yaoet al. 2023; Trivedi et al. 2023; Asai et al. 2024). Recent
deep-researchsystems extend these ideas to multi-agent or-
chestrationwithamanager,specialistagents,averifier,anda
finalizer;releasedproductsincludeOpenAI’sDeepResearch
(OpenAI 2025) and Gemini Deep Research (Google Deep-
Mind2024),andtheopenliteratureincludesSTORM(Shao
et al. 2024) and AutoGen (Wu et al. 2023). Our two query-
time candidates instantiate this ladder:Agentic RAGis a
bounded tool-using single-analyst workflow, and theDeep
Research Agentisadecomposedworkflowwithspecialist
investigations, evidence audit, and revision.
Corpus-level and graph-structured RAG.Most RAG
benchmarks emphasize local questions whose evidence lies
in a few passages; GlobalQA shows that counting, sorting,
and other corpus-level operations remain difficult for con-
ventional RAG (Luo et al. 2025). A parallel line of work
shifts work from query time to ingestion time by building
a persistent structured index that a lighter query-time agent
thenreads.GraphRAGextractsentitiesandrelationsintohi-
erarchicalcommunitysummariessothatquery-focusedsum-
marization is answered from a graph-level summary rather
than direct retrieval (Edge et al. 2024). HippoRAG aug-
ments dense retrieval with Personalized-PageRank traversal
(Gutiérrezetal.2024);LightRAGcombinesgraphindexing
withdual-levelretrieval(Guoetal.2024);RAPTORbuildsa
recursivetreeofclusteredabstractivesummaries(Sarthietal.
2024);StructuredRAGtranslatesaggregativequestionsinto
formal operations over an ingestion-time schema (Koshorek
et al. 2025). OurContext Graph Agentshares the work-
at-ingestionpremisebutdiffersinwhatisstored:ratherthan
aderived-summarygraphoranaggregated-chunkindex,in-
gestion persists typed customer-feedback entities (records,
users, sources, themes, subthemes, sentiment, ratings) with
anexplicitproducttaxonomyandrecord-leveltaxonomypre-
dictions that are durable across sessions and reused across
independentlyexecutedquestions.Toourknowledgenoprior
benchmarkcomparesagraph-indexedworkflowofthiskind
againstagenticanddeep-researchworkflowsonasharedcor-
pus, shared questions, and a shared answer contract.
Theme reliability and grounded-answer evaluation.
Statistical topic coherence does not guarantee actionable
product dimensions (Nguyen and Hovy 2019) and topic-
model document assignments can be unreliable inputs to
downstream analysis (Schroeder and Wood-Doughty 2025),
so we evaluate theme membership against adjudicated
records rather than trusting coherence alone. RAG evalu-
ation is multidimensional: RAGAs proposes reference-free
metrics (Es et al. 2024), ARES combines automated judges
with a human-labeled set (Saad-Falcon et al. 2024), and
ALCEevaluatesanswercorrectnessandcitationquality(Gao
et al. 2023). We do not adopt these evaluators directly be-
causeeachassumesasingle-answerQAsetting:RAGAsand
ALCE score a system answer against a query with a known
correctanswerorasmallsetofground-truthpassages;ARES
trains a judge on a human-labeled QA dataset. Corpus-level
analytical answers that combine typed classification counts,
aggregatedrill-downcitations,per-themequantification,and
refusal on unsupported groupings do not fit that mold: there

is no single “correct” answer against which to measure pre-
cision or faithfulness. We instead evaluate paired candidate
answers and audit their cited records directly, following the
attribution-to-identified-source protocol of Rashkin et al.
(2023)andtheatomicfactualityintuitionofMinetal.(2023).
LLM judges scale open-ended evaluation but exhibit posi-
tion, verbosity, and self-enhancement biases (Zheng et al.
2023), so we use query-specific anonymized answer order,
separate comparative dimensions, and deterministic citation
sampling.Publicreviewsarealsoaself-selectedviewofcus-
tomer experience and review solicitation changes extremity
andrepresentativeness(Karaman2021);ourprevalencemea-
surements therefore describe the frozen corpus, not the full
Cursor customer population.
3 Compared Systems
Wecomparethreeconcreteimplementations(Figure1);can-
didate names refer to these implementations, not to all sys-
temsinanarchitecturalfamily.Eachreceivesthesamequery
text and evidence from the same frozen corpus. We use one
running question,q013(“Are there signs of a recently in-
creasingreliabilityissueinCursor?”),toillustratewhateach
system does differently; it requires an aligned time compar-
ison, an explicit denominator, per-cluster counts, and a de-
fensiblerefusalwhenacomparablebaselineisnotavailable.
Agentic RAGisonecustomer-feedbackanalystbuiltwith
anagentSDK:boundedtooluse,semanticsearch,recordin-
spection, exact metadata aggregation, segment comparison,
and cited synthesis. Onq013it runs its own 30-day pre-
ceding/recent comparison with a keyword reliability proxy
(96/557 vs 851/8,652 records) and labels the trend low-
confidence. Any semantic groups it creates belong to that
run and do not persist across questions.
Deep Research Agentdecomposes each question into
typedevidencerequirements:amanagerdelegatestoqualita-
tive,quantitative,andincidentspecialiststhatsearch,inspect
records,aggregateexactly,and—whenenabled—classifyev-
ery record in a filter-derived denominator (cached with a
record-levelledgerandcontenthash);averifierauditsclaims,
counts, coverage, and citations before a finalizer revises the
answer. Onq013it constructs matched rolling 14-day win-
dows(3,720vs3,717records)soallthreesourcesarepresent
inbothandlocalizesaGLM5.2tool-callclustertotwospe-
cific dates. Query-specific assignments are not exposed to
other questions as a shared ontology or durable label store.
Context Graph Agentreceives normalized records af-
ter they are processed into a persistent customer context
graph. Before benchmark questions are asked the system
summarizes feedback, induces a product taxonomy, stores
record-level taxonomy predictions, maps source fields to a
shared metadata schema, retains available user information,
andstoresrelationshipsamongtheseentities.Thequery-time
agenttraversesthoserelationshipstogroup,count,compare,
and cite feedback. Onq013it reads 86 reliability-adjacent
themesalreadytaggedinthegraph,aggregatesconversation
countspertheme,anddeclinesatrendclaimbecausethepre-
ceding calendar window is 24 Trustpilot-only conversationsTable1:Experimentalcontrolsforthethree-systemcompar-
ison.
Held
constantFrozen 9,432-record corpus + SHA-256;
identical query text and as-of date; base
analyst model, both judges, retrieval
reranker; answer contract; retry policy;
alias-shuffled evaluator presentation.
Allowed to
differ (the
treatment)Whether the system reconstructs product
concepts per query or reads pre-computed
summaries, taxonomy predictions, metadata,
user information, and graph relationships
stored at ingestion; the orchestration budget
the analyst model is dispatched into.
Can
therefore
claimThe complete-system effect of persistent
customercontextonthiscorpus,questionset,
and calibrated rubric; not the isolated causal
effect of any component in isolation.
Table 2: Frozen Cursor corpus used by all three systems.
Statistic Value Coverage
Normalized feedback
records9,432 100%
Collection window 2026-04-16–2026-07-15 UTC
Public feedback sources 3 100%
Records with rating 171 1.81%
Records with normalized
version5 0.05%
Records with public
country118 1.25%
Account/commercialcohort
ID0 0%
Corpus SHA-256d62fa842...f23a62f0
against 9,408 recent mixed-source conversations.
Experimental controls and complete-system treatment.
Table 1 states what was held constant, what was allowed
todiffer,andwhattheexperimentcanthereforeclaim;every
runrecordsmodelandtoolversions,prompts,tokenuse,tool
calls,failures,corpushash,andconfigurationhash.Failedor
incomplete runs remain in the denominator with no credit,
so a non-completion cannot silently drop out of the average.
FreezeorderandrubriccalibrationaredetailedinSection4.
4 Dataset and Benchmark
Corpus.The evaluated corpus contains 9,432 public Cur-
sorfeedbackrecordsfromthefrozenthree-monthUTCwin-
dow 2026-04-16 through 2026-07-15 (Table 2). Collection,
normalization, and deduplication are frozen before evalua-
tion; each record preserves a stable identifier, source, times-
tamp,feedbacktext,sourceURL,ratingwhenavailable,and
public metadata. Missing account or commercial attributes
remainmissing.Allreportedprevalencedescribesthisfrozen
corpus rather than the full Cursor user base.
Information parity and denominator contract.All sys-
temsreceivefeedbackoriginatingfromthesamefrozennor-

Shared input System-specific preparation Question execution Answer package Common evaluation Output
Neutral parser
9,432 feedback
records + metadataAgentic RAG
vector index
metadata storeSingle analyst agent
plans + retrieves
aggregates + synthesizesCited answer
record evidence
Deep Research Agent
vector index
exact-analysis toolsManager + specialists
evidence ledger, verifier, revisionCited answer
record + ledger
evidence
Context Graph Agent
summaries + predictions
metadata + user linksAgent groups + counts
compares + drills downCited answer
aggregate +
record evidencePaired evaluator
answer quality
analytical depth
evidence qualityPer-query scores
+ aggregate
comparison
Figure 1: End-to-end benchmark architecture. A neutral parsing layer supplies the same feedback text and source metadata to
all candidates. Each system keeps its native preparation and question-execution architecture, but all outputs are converted to a
common answer package and scored by the same paired evaluator.
malized records; source URLs and public metadata are pre-
servedconsistently,andthetwoquery-timesystemsdonotre-
ceivetaxonomies,summaries,clusters,orotherlabelsgener-
atedbythecontextgraphsystem.Thecontextgraphsystem’s
ingestion additionally groups the 9,432 records into 5,035
customerconversationsby threading records that share a
parentpost,commentchain,reviewreplychain,orplatform-
linked exchange within the same source and author. To keep
quantified claims comparable at judge time we require a de-
nominator contract: every quantified claim must state the
total it is divided by and cite the record set that produced
thattotal.Baselinesusetherecordcount(theydonotreceive
conversation-adjacency edges); the context graph candidate
mayreportatconversationgrainbutmuststatetheconversa-
tion total and the constituent record count. The comparative
evidence-strengthscorerewardsexplicitandciteddenomina-
tors over absolute magnitude, so 180 conversations with de-
nominator and drill-down cannot outscore 892 records with
equivalent evidence structure by being smaller. After a pilot
exposed a 123-record snapshot mismatch we refreshed the
neutralexportandre-ranallthreecandidatesonall30queries
against the repaired corpus and its SHA-256 hash. Per-run
identifiers and the canonical evaluation-pass identifier are
recorded in the answer packages and in the source snap-
shot macros; the build gate refuses to update either without
matchingrun_metadataacross all 90 cells. Query con-
tracts and evaluator prompts are hidden from every system;
each query executes in a fresh conversation.
Query benchmark.The benchmark contains 30 product-
adapted questions arranged into seven analysis families (ex-
ecutive VoC, product area, reliability and incidents, pricing
andpackaging,sourceandmetadata,competitionandreten-
tion, and cross-functional action; sizes{5,5,5,5,5,4,1})
and fixed before candidate execution, spanning four depth
levels (Table 3); the same question text and as-of date are
sent byte-for-byte to every candidate. Hidden family invari-
ants test exact-value reuse, semantic stability, segment-level
summaries, trend direction, sampling restraint, and whether
recommendedactionstracetopreviouslymeasuredevidence.
Each query is a small contract: it names the operations theTable 3: Benchmark depth. Counts are fixed by the query
manifest.
Depth Required reasoningn
1 Exact lookup, metadata coverage,
capability checks11
2 Focused semantic classification or
evidence synthesis10
3 Comparisons across source, time, or
cohort3
4 Multi-stage VoC, strategy, incident,
and workflow decisions6
Total 30
answer must perform, the fields it may draw on, the ex-
pected output elements, count semantics, citation require-
ments,evaluationscope,goldsource,andknownlimitations.
The benchmark has no independent exhaustive semantic la-
bels.ForsemanticclaimstheLLMjudgescoresonlyrecords
actually supplied by each candidate, so we can estimate evi-
dence validity on observed assignments but cannot identify
false negatives; semantic recall and F1 are therefore not re-
ported.
Freeze order, calibration, and held-out confirmation.
Wefreeze,inorder,(1)researchquestionsandsystemdefini-
tions;(2)querytextandanswercontracts;(3)collection,nor-
malization, and deduplication; (4) candidate prompts, tools,
models,andbudgets;(5)aninitialjudgerubricv1;(6)afirst
evaluated run against v1; (7) a bounded calibration pass in
which the judge prompts were adjusted after human inspec-
tionofv1outputsonthefirst27questions,producingrubric
v2;(8)are-evaluationagainstv2;and(9)asinglefurthernar-
rowingofthecompetitor-gapcriterion,promptedbyinspect-
ing the q029 answers, after which only q029 was rescored.
Steps (1)–(9) produced the scores reported in Section 6, so
28 of the 30 questions informed the rubric that scores them.
We verified that v2 applied to q001–q027 yields identical
per-cell scores to the earlier calibrated-on-q001–q027 snap-

Agentic RAG: one tool-using analyst
User questionAnalyst agent
plans retrieval
and revisesSemantic search + metadata
exact aggregates + record inspectionTransient themes
counts + synthesisCited answer
Deep Research Agent: managed, parallel analysis with verification
User questionManager
decomposes
evidence needsQualitative + quantitative
incident specialists
shared retrieval toolsEvidence ledger
verifier + finalizerCited answer
Customer Context Graph-backed Agent: stored feedback context queried at run time
Normalized records
+ metadataFeedback summaries
taxonomy predictions
metadata mappingStored customer context
records + summaries
predictions + metadata
user information + links
User questionContext Graph Agent
maps question to
product conceptsGraph queries + aggregation
conversation drill-downSpecific themes
quantification + synthesisCited answer
Figure 2: Component architecture of the three evaluated candidates. Agentic RAG uses one analyst over retrieval and exact-
analysistools.TheDeepResearchAgentaddsdecomposition,specialistroles,anevidenceledger,verification,andrevisionover
thatretrievallayer.Thecontextgraphsystemfirstconnectsfeedback,summaries,taxonomypredictions,metadata,andavailable
user information; its agent then computes aggregates and drills down to source records when producing an answer.
shot, confirming that steps 7–8 did not retroactively move
the rubric on the calibration set. Because v2 was reached
whileinspectingq001–q027, questionsq028 andq030were
scored under it without further change and provide a small
internalstabilitycheck(Section6);q029isexcludedbecause
itsrubricwasnarrowedafteritsanswerswereinspected.We
deliberately do not call this a confirmatory set: two same-
session,same-familyquestionswithheldfromcalibrationare
notequivalenttofreshlyauthoredquestions,andthereported
marginsshouldbereadasexploratory.Thestep6–8calibra-
tion is a deviation from a purely sequential freeze order,
treated as the principal threat in Section 7.
5 Automatic Evaluation
Open-endedcustomer-feedbackquestionshavenoexhaustive
referenceanswer.Wethereforecompareallthreeanswersfor
eachqueryandseparatelyauditwhethercitedrecordssupport
the claims made (Figure 3). The query is the unit of evalu-
ation, all candidates are judged together under anonymized
aliaseswithdeterministicallyshuffledorder,andeveryreport
retains dimension-level rationales and citation audits. The
three reported dimensions and their scoring conventions are
summarizedhere;fullprotocoldetails,macro/micro-average
conventions,andcitation-auditprovenanceappearinthesup-
plementary material.Three anonymized answers
Comparative
answer judgeUp to 15 records
per citation
Claim–record rel-
evance judge
Answer quality +
analytical depthEvidence quality
+ audit coverage
Overall mean + rea-
sons + paired results
Figure3:Thesimpleprotocolcomparesanonymizedanswers
and audits every resolvable citation group separately.
Answer quality and analytical depth.The comparative
judge places candidates into best-to-worst tiers on each di-
mension. Answer quality rewards explicit question cover-
age, organization for the intended reader, useful quantifica-
tion, and visible separation of evidence, interpretation, and
caveats.Analyticaldepthrewardsahierarchyfromdecision-
relevant conclusion to bounded themes, drivers, subthemes,
counterevidence, and synthesis; a one-problem test penal-
izes umbrella labels that merge independently actionable is-

sues. With three distinct tiers, scores are{1.0,0.75,0.5};
two-tier verdicts map to{1.0,0.5}with the middle tier re-
served for three-tier verdicts. Both dimensions therefore fall
in{0.5,0.75,1.0}for completed answers.
Evidence quality.Evidence qualityE qis the harmonic
mean of sampled citation precisionP qand a comparative
evidence-strength tier scoreS q:
Pq=Nrel+ 0.5N part
Nq, E q=2PqSq
Pq+S q.(1)
For each record citation and each aggregate citation whose
members are exported, the evaluator deterministically sam-
plesupto15frozenrecordsandaclaim–recordjudgelabels
eachpairrelevant,partiallyrelevant,orirrelevant,following
theattribution-to-identified-sourceprotocolofRashkinetal.
(2023)andtheatomicfactualityintuitionofMinetal.(2023).
Sqrewards claim-level support, explicit denominators, ag-
gregatedrill-downlinks,andrepresentativerecordcitations.
An answer without citations receives zero. Aggregate cita-
tions with valid drill-down metadata but without exported
members are counted astraceable, not member-sampled;
unresolved citations are reported separately. Evidence qual-
itythereforeevaluatessampledcorrectnessandcomparative
evidence design; it does not imply that every record behind
every aggregate was independently verified.
Overall score and aggregation.The overall score com-
binesthethreedimensionsequally,Overall q= (Q q+D q+
Eq)/3, and we macro-average query-level scores across the
30 queries so each query weighs equally in the system total.
Paired win, tie, and loss counts follow the paired-evaluation
formulationofPeyrardetal.(2021)withaprespecified0.01
absolute-difference tie tolerance; pairedW/T/Lis the pri-
marysystem-comparisonstatisticand ∆isdescriptive.Each
(query, candidate) cell is asinglerun scored by one judge
pass;wedonotaverageoverreruns,sorun-to-runvarianceis
unmeasuredandbootstrapintervalsquantifyvariationacross
queries rather than across reruns. LLM judges can also ex-
hibitposition,verbosity,andself-enhancementbiases(Zheng
etal.2023);thealias-shufflingandstructured-outputrequire-
ments above reduce but do not eliminate those risks.
6 Results
All three systems completed all 30 questions, yield-
ing 90 dimension-level reports under protocol
simple-comparison-v30. The query is the scor-
ing unit; we macro-average query-level scores and compare
systems on paired questions with a 0.01 tie tolerance.
Categorymeansandper-depthslicesaredescriptivebecause
families contain between 1 and 5 queries, and semantic
recall is not reported because no independent exhaustive set
identifies every record relevant to an open-ended feedback
theme.
TheContext Graph Agentobtains the highest score on
every dimension (Table 4), with overall 0.961 (0.251 above
theDeep Research Agent,0.310aboveAgentic RAG);its
largest gap is analytical depth (0.967 vs 0.642). TheDeep
Research Agentimproves overall overAgentic RAGby
0.059,drivenprimarilybyevidencequality(0.806vs0.686).Paired outcomes.TheContext Graph Agentleads the
Deep Research Agenton 27 of 30 overall queries, ties
once, and loses twice (Table 5; sign test on non-tied out-
comes,p= 1.6×10−6; bootstrap 95% CI on ∆overall
[+0.193,+0.305]). It leads analytical depth on 28 of 30
(p= 8.7×10−7) and leadsAgentic RAGon 29 of 30
overall (p= 5.8×10−8;∆overall = +0.310, 95% CI
[+0.263,+0.354]). TheContext Graph Agentis strictly
best on 26 of 30 individual queries; the four exceptions are
q003(tied withDeep Research Agent),q006andq013
(lost toDeep Research Agenton autocomplete overview
and matched-window reliability), andq022(“What are
people praising in the App Store?”, whichAgentic RAG
wins 0.887 vs 0.833 by combining tier-1 answer quality
and depth with a lower evidence-quality score). The sec-
ondaryDeep Research Agent-vs-Agentic RAGcompari-
sonislessmonotonic:Deep Research Agentwins17over-
allandloses13(signtestp= 0.585; ∆overall = +0.059,95%
CI[−0.018,+0.135]), its evidence-quality advantage is real
(26/4 wins,p= 6.0×10−5), but analytical-depth outcomes
split 15–15 and average to zero.
Citation-audit coverage.Agentic RAGemits 712 cita-
tionsandDeep Research Agentemits594;nearlyallresolve
directlytomemberrecords(708and594sampledrecordsre-
spectively).TheContext Graph Agentemits913citations
of which 127 are member-audited and 786 are aggregate ci-
tations with valid drill-down metadata but without exported
members(Table6).Nocontext-graphcitationisunresolved.
Itshighevidence-qualityscorethereforereflectsperfectsam-
pled precision on exported member evidence together with
thecomparativejudge’spreferenceforclaim-levelquantified
evidence architecture; it must not be read as independent
validationofallrecordsbehindthe786traceableaggregates.
Partialinternalrubric-stabilitycheck.Thev2rubricwas
reachedwhileinspectingq001–q027andappliedunchanged
toq028andq030(q029isexcluded:itsrubricwasnarrowed
after its answers were inspected). On thatn= 2slice the
Context Graph Agentwinsbothqueriesagainstbothbase-
lines ( ∆overall = +0.302versus theDeep Research Agent,
+0.404versusAgentic RAG), consistent with the calibra-
tion set. This is a weak internal check, not confirmation:
n= 2carries almost no inferential power, and q028 and
q030 were authored in the same session and families as the
calibration queries, so a rubric shaped on their siblings can
carrycorrelatedbiasintothem.Withholdingaquestionfrom
calibrationisnotequivalenttodrawingafreshone.Thedeci-
sive test—new queries authored after the rubric was frozen,
scoredwithoutfurtherrubricchange—isnotperformedhere
and remains the primary outstanding validation (Section 7).
Findings across RQs and interpretation.Under the
aligned rubric, theContext Graph Agentleads on every
dimension in Table 4 (RQ1), and its lead is stable across
all four benchmark depth levels (RQ2, supplementary ma-
terial). Quantification splits the systems by grain: baselines
report exclusive-classification counts across the full record
set, while theContext Graph Agentreports theme counts
at conversation grain with per-theme drill-down (RQ3). The

Table 4: Macro-averaged automatic evaluation across 30 paired queries. Overall combines the three dimensions equally. Per-
dimension scores for answer quality and analytical depth are drawn from a discrete comparative-tier scale on[0.5,1]for
completed answers with 0.25 spacing (see Section 5); evidence quality is a harmonic mean of sampled citation precision and
comparative evidence-strength score in[0,1]. Because the scale is discrete and the primary comparison is paired, means are
descriptive; the primary comparison statistic is pairedW/T/L(Table 5). Best values in bold; bracketed overall values are
bootstrap 95% CIs over 30 queries (B= 5,000, seed 42).
System Overall (95% CI) Answer quality Analytical depth Evidence quality Queries
Agentic RAG 0.651 [0.613, 0.692] 0.625 0.642 0.686 30/30
Deep Research Agent 0.710 [0.668, 0.752] 0.683 0.642 0.806 30/30
Context Graph Agent0.961[0.936, 0.983]0.958 0.967 0.95830/30
Table 5: Paired query-level comparisons with a 0.01 tie tolerance.W/T/Lcounts are from the perspective of the first system;
∆is its mean score minus the second system’s score, with a bootstrap 95% confidence interval over 30 queries (B= 5,000
resamples, seed 42);preports a two-sided sign test on non-tied paired outcomes. Because the completed-answer score scale is
discrete with 0.25 spacing (see Table 4 and Section 5), the pairedW/T/Lresult is the primary comparison statistic and ∆is
descriptive.
Comparison Metric Wins Ties Losses ∆(95% CI) Sign-testp
Context graph−DeepOverall 27 1 2 +0.251 [+0.193, +0.305]1.6×10−6
Answer quality 25 1 4 +0.2751.0×10−4
Analytical depth 28 0 2 +0.3258.7×10−7
Evidence quality 27 2 1 +0.1532.2×10−7
Context graph−AgenticOverall 29 0 1 +0.310 [+0.263, +0.354]5.8×10−8
Answer quality 28 1 1 +0.3331.1×10−7
Analytical depth 28 0 2 +0.3258.7×10−7
Evidence quality 28 1 1 +0.2721.1×10−7
Deep−AgenticOverall 17 0 13 +0.059 [−0.018, +0.135] 0.585
Answer quality 17 0 13 +0.058 0.585
Analytical depth 15 0 15 +0.000 1.000
Evidence quality 26 0 4 +0.1206.0×10−5
citationauditshowsthatmostofthecontext-graphevidence-
qualityadvantagecomesfromaggregatedrill-downcitations,
not from higher sampled precision (RQ4). On epistemic re-
straint the strongest in-run signal isq013, where theCon-
text Graph Agentrefusesaperiod-over-periodtrendclaim;
apre-registeredrefusalbatteryisfuturework(RQ5).Andon
structure versus agency, theDeep Research Agent’s de-
composition, verifier, and revision loop lift evidence quality
overAgentic RAGbut leave overall and depth outcomes
indistinguishable atn= 30(RQ6).
The dimension pattern suggests a straightforward mech-
anism. Ingestion-time processing makes stored summaries,
taxonomy predictions, metadata, and relationshipsdurable:
at query time the agent traverses those relationships to
boundedthemeswithaggregate-citationquantificationrather
thanreconstructingproductconceptseachquestion.Evenso,
the result is not universal dominance. TheDeep Research
Agentwins the two questions that reward bespoke matched
comparisons (q006,q013), andAgentic RAGwins the
App-Store-praise question (q022). Query-specific compu-
tation can therefore beat stored concepts when a bespoke
comparison matters more than concept reuse. Representa-
tive response excerpts (including theq013reconciliation),per-depth and per-family means, and the extended mecha-
nism analysis are in the supplementary material.
7 Limitations and Threats to Validity
Evaluation validity.LLM judges can share biases with
candidatemodels,butthemodel-constantdesignlargelyneu-
tralizesthisforthereportedranking:thejudgescoresanswer
textonly,andallthreecandidates’textcomesfromthesame
gpt-5.5analystmodel,soprovider-familyself-preference
applies uniformly and cannot differentially favor one can-
didate. The substantive concern is not model identity but
rubric alignment, and it is the paper’s principal limitation.
Therubricwasalignedwithanalystpreferencesafterinspect-
ing answers from q001–q027, and one later narrowing was
prompted by q029, so 28 of the 30 questions informed the
rubric that scores them and their scores measure agreement
withitratherthanprovidinganindependenttest.Disclosure
doesnotrepairthis,andthesigntestsinTable5donoteither:
they rule out sampling noise, not rubric bias. The two with-
heldquestions(q028,q030)giveonlyaweakinternalcheck,
being same-session and same-family as the calibration set.
We therefore present the 0.961/0.710/0.651 ordering asex-

Table 6: Citation-audit coverage. “Member-audited” citations have at least one frozen record judged against the nearby claim.
Traceable aggregates retain drill-down metadata but were not member-sampled.
System Cited Member-audited Traceable aggregate Unresolved Records sampled
Agentic RAG 712 708 0 4 708
Deep Research Agent 594 594 0 0 594
Context Graph Agent 913 127 786 0 127
ploratory; the decisive validation is a confirmatory pass on
questions authored after the rubric was frozen. Semantic re-
call is separately absent because unobserved false negatives
are unidentified.
Internalandaggregate-auditvalidity.Thecontextgraph
system receives upfront customer-context processing that
the experiment intentionally studies, but the treatment bun-
dles feedback summarization, taxonomy induction, record-
levelprediction,metadatamapping,relationshipstorage,and
agenttooling;thecomparisonestimatesasystem-leveleffect
ratherthantheisolatedcausalcontributionofanycomponent.
An earlier pilot exposed a 123-record snapshot mismatch,
whichwecorrectedbeforethereportedrun.Only127ofthe
context graph agent’s 913 citations were member-sampled;
theremaining786havevalidaggregatedrill-downmetadata
butnoexportedmemberrecordsinthejudgepackage,sothey
supporttraceabilityandevidence-architecturescoringrather
than independent verification of every associated record.
Proof scope, sampling, and generalization.Thirty ques-
tions on one product corpus over one three-month frozen
UTC window improve construct coverage but do not
proveuniversalarchitecturalsuperiority;conclusionsremain
scoped to that frozen configuration and to the calibrated
rubric. Public reviews are self-selected and can overrepre-
sent unusually positive or negative experiences (Karaman
2021), so counts and shares estimate prevalence within the
frozen corpus, not among all Cursor users. App reviews can
be short, noisy, multilingual, sarcastic, off-topic, or fraudu-
lent (Wang et al. 2024; Apple Machine Learning Research
2025), and public data lack the account, contract, and rev-
enue relationships available in private enterprise feedback.
Secondary threats—construct validity of the theme rubric,
implementation choices in the two query-time candidates,
temporal drift, and the gap between workflow scores and
downstreambusinessoutcomes—areenumeratedinthesup-
plementary material.
8 Conclusion
Across 30 realistic Cursor feedback questions and under an
analyst-alignedrubric,theContext Graph Agentachieves
an overall score of 0.961 versus 0.710 for theDeep Re-
search Agentand 0.651 forAgentic RAG, leading on 27
pairedquestionswithitslargestadvantageinanalyticaldepth.
Within this frozen configuration, more query-time agency
alonedoesnotreliablyreconstructthevalueofpersistentre-
lationships among feedback, summaries, taxonomy predic-
tions,metadata,andavailableuserinformation.Thefindings
are system-level and descriptive: the evaluation comparescandidateanswersandauditstheircitedevidenceratherthan
claiming semantic recall. Natural next steps outside this pa-
per’s scope are (i) a hybrid design combining stored context
with bespoke matched-window computation at query time
(motivated byq013); (ii) component-level ablations of in-
gestion;and(iii)aheld-outconfirmatoryquestionsetscored
against a rubric frozen before the confirmatory answers are
seen.
References
Apple Machine Learning Research. 2025. An LLM-
Based Approach to Review Summarization on the App
Store. https://machinelearning.apple.com/research/app-
store-review. Accessed 2026-07-14.
Asai, A.; Wu, Z.; Wang, Y.; Sil, A.; and Hajishirzi, H.
2024. Self-RAG: Learning to Retrieve, Generate, and Cri-
tique through Self-Reflection. InInternational Conference
on Learning Representations.
Edge,D.;Trinh,H.;Cheng,N.;Bradley,J.;Chao,A.;Mody,
A.; Truitt, S.; and Larson, J. 2024. From Local to Global:
AGraphRAGApproachtoQuery-FocusedSummarization.
arXiv preprint arXiv:2404.16130.
Es,S.;James,J.;Anke,L.E.;andSchockaert,S.2024. RA-
GAs:AutomatedEvaluationofRetrievalAugmentedGener-
ation.InProceedingsofthe18thConferenceoftheEuropean
Chapter of the Association for Computational Linguistics:
System Demonstrations, 150–158. Association for Compu-
tational Linguistics.
Fiocco, E.; Proietti, S.; and Cesarotti, V. 2025. Voice of
Customer Extraction from Product Reviews: A Benchmark
ofRAGVariants. InInternationalConferenceonMarketing
and Technologies (ICMarkTech). Valencia, Spain.
Gao,T.;Yen,H.;Yu,J.;andChen,D.2023. EnablingLarge
Language Models to Generate Text with Citations. InPro-
ceedings of the 2023 Conference on Empirical Methods in
Natural Language Processing, 6465–6488. Association for
Computational Linguistics.
Google DeepMind. 2024. Try Deep Research and Our
New Experimental Model in Gemini. https://blog.google/
products/gemini/google-gemini-deep-research/. Accessed
2026-07-14.
Griffin, A.; and Hauser, J. R. 1993. The Voice of the Cus-
tomer.Marketing Science, 12(1): 1–27.
Guo, Z.; Xia, L.; Yu, Y.; Ao, T.; and Huang, C. 2024. Ligh-
tRAG: Simple and Fast Retrieval-Augmented Generation.
arXiv preprint arXiv:2410.05779.
Gutiérrez, B. J.; Shu, Y.; Gu, Y.; Yasunaga, M.; and Su, Y.

2024. HippoRAG: Neurobiologically Inspired Long-Term
MemoryforLargeLanguageModels. InAdvancesinNeural
Information Processing Systems, volume 37.
Karaman,H.2021. OnlineReviewSolicitationsReduceEx-
tremity Bias in Online Review Distributions and Increase
Their Representativeness.Management Science, 67(7):
4420–4445.
Karpukhin,V.;Oguz,B.;Min,S.;Lewis,P.;Wu,L.;Edunov,
S.; Chen, D.; and tau Yih, W. 2020. Dense Passage Re-
trieval for Open-Domain Question Answering. InProceed-
ingsofthe2020ConferenceonEmpiricalMethodsinNatural
Language Processing, 6769–6781. Association for Compu-
tational Linguistics.
Koshorek, O.; Granot, N.; Alloni, A.; Admati, S.; Hendel,
R.; Weiss, I.; Arazi, A.; Cohen, S.-N.; and Belinkov, Y.
2025. Structured RAG for Answering Aggregative Ques-
tions.arXiv preprint arXiv:2511.08505.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.;
Goyal, N.; Küttler, H.; Lewis, M.; tau Yih, W.; Rocktäschel,
T.;Riedel,S.;andKiela,D.2020.Retrieval-AugmentedGen-
erationforKnowledge-IntensiveNLPTasks. InAdvancesin
Neural Information Processing Systems, volume 33, 9459–
9474.
Liu,N.F.;Lin,K.;Hewitt,J.;Paranjape,A.;Bevilacqua,M.;
Petroni, F.; and Liang, P. 2024. Lost in the Middle: How
Language Models Use Long Contexts.Transactions of the
Association for Computational Linguistics, 12: 157–173.
Luo,Q.;Li,X.;Fan,T.;Chen,X.;andQiu,X.2025. Towards
Global Retrieval Augmented Generation: A Benchmark for
Corpus-LevelReasoning.arXivpreprintarXiv:2510.26205.
Maalej, W.; Kurtanović, Z.; Nabil, H.; and Stanik, C. 2016.
On the Automatic Classification of App Reviews.Require-
ments Engineering, 21(3): 311–331.
Min, S.; Krishna, K.; Lyu, X.; Lewis, M.; tau Yih, W.; Koh,
P. W.; Iyyer, M.; Zettlemoyer, L.; and Hajishirzi, H. 2023.
FActScore:Fine-GrainedAtomicEvaluationofFactualPre-
cisioninLongFormTextGeneration. InProceedingsofthe
2023ConferenceonEmpiricalMethodsinNaturalLanguage
Processing, 12076–12100. Association for Computational
Linguistics.
Nguyen,H.;andHovy,D.2019.HeySiri.OkGoogle.Alexa:
A Topic Modeling of User Reviews for Smart Speakers. In
Proceedings of the 5th Workshop on Noisy User-generated
Text, 76–83. Association for Computational Linguistics.
OpenAI. 2025. Introducing Deep Research. https://openai.
com/index/introducing-deep-research/. Accessed 2026-07-
14.
Panichella, S.; Sorbo, A. D.; Guzman, E.; Visaggio, C. A.;
Canfora, G.; and Gall, H. C. 2015. How Can I Improve My
App? Classifying User Reviews for Software Maintenance
and Evolution. In2015 IEEE International Conference on
Software Maintenance and Evolution, 281–290.
Peyrard, M.; Zhao, W.; Eger, S.; and West, R. 2021. Better
than Average: Paired Evaluation of NLP Systems. InPro-
ceedings of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International JointConference on Natural Language Processing, 2301–2315.
Association for Computational Linguistics.
Rashkin, H.; Nikolaev, V.; Lamm, M.; Aroyo, L.; Collins,
M.;Das,D.;Petrov,S.;Tomar,G.S.;Turc,I.;andReitter,D.
2023. Measuring Attribution in Natural Language Genera-
tion Models.Computational Linguistics, 49(4): 777–840.
Saad-Falcon,J.;Khattab,O.;Potts,C.;andZaharia,M.2024.
ARES: An Automated Evaluation Framework for Retrieval-
AugmentedGenerationSystems. InProceedingsofthe2024
ConferenceoftheNorthAmericanChapteroftheAssociation
for Computational Linguistics: Human Language Technolo-
gies, 338–354. Association for Computational Linguistics.
Sarthi, P.; Abdullah, S.; Tuli, A.; Khanna, S.; Goldie, A.;
andManning,C.D.2024. RAPTOR:RecursiveAbstractive
Processing for Tree-Organized Retrieval. InInternational
Conference on Learning Representations.
Schroeder, K.; and Wood-Doughty, Z. 2025. Reliability of
Topic Modeling. InProceedings of the 2025 Conference of
the Nations of the Americas Chapter of the Association for
ComputationalLinguistics:HumanLanguageTechnologies,
2649–2662. Association for Computational Linguistics.
Shao, Y.; Jiang, Y.; Kanell, T. A.; Xu, P.; Khattab, O.; and
Lam, M. S. 2024. Assisting in Writing Wikipedia-like Arti-
clesfromScratchwithLargeLanguageModels. InProceed-
ingsofthe2024ConferenceoftheNorthAmericanChapterof
the Association for Computational Linguistics, 6252–6278.
Association for Computational Linguistics.
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A.2023. InterleavingRetrievalwithChain-of-ThoughtRea-
soning for Knowledge-Intensive Multi-Step Questions. In
Proceedings of the 61st Annual Meeting of the Association
for Computational Linguistics, 10014–10037. Association
for Computational Linguistics.
Wang, X.; Zhang, T.; Tan, Y.; Shang, W.; and Li, Y. 2024.
HowtoEffectivelyMineAppReviewsConcerningSoftware
Ecosystem?ASurveyofReviewCharacteristics.Journalof
Systems and Software, 213: 112040.
Wu,Q.;Bansal,G.;Zhang,J.;Wu,Y.;Li,B.;Zhu,E.;Jiang,
L.; Zhang, X.; Zhang, S.; Liu, J.; Awadallah, A.; White,
R. W.; Burger, D.; and Wang, C. 2023. AutoGen: Enabling
Next-GenLLMApplicationsviaMulti-AgentConversation.
arXiv preprint arXiv:2308.08155.
Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan,
K.; and Cao, Y. 2023. ReAct: Synergizing Reasoning and
ActinginLanguageModels. InInternationalConferenceon
Learning Representations.
Zheng, L.; Chiang, W.-L.; Sheng, Y.; Zhuang, S.; Wu, Z.;
Zhuang, Y.; Lin, Z.; Li, Z.; Li, D.; Xing, E. P.; Zhang, H.;
Gonzalez, J. E.; and Stoica, I. 2023. Judging LLM-as-a-
Judge with MT-Bench and Chatbot Arena. InAdvances in
Neural Information Processing Systems, volume 36.

Ethics and Data Statement
The corpus is public product feedback; public availability
does not remove privacy risk. Released artifacts minimize
personal identifiers, avoid cross-source display-name link-
age,respectplatformterms,andquoteonlytextneededtoau-
ditclaims.Wedonotinferprotected,commercial,oraccount-
level attributes from review text, and corpus prevalence is
not treated as population prevalence because public review
submission is self-selected. Non-redistributable records are
represented through hashes and collection manifests.
A Representative Response Excerpts
Aggregate scores hide material differences in what an ana-
lystreceives.Table7showsthedominantbenchmarkpattern:
thequery-timesystemsbothorganizethecorpusintoasmall
number of broad pain-point categories via exclusive clas-
sification (seven categories forAgentic RAGacross 8,890
classifiedrecords,tenfortheDeep Research Agentacross
all 9,432 records), whereas theContext Graph Agentre-
turns narrower recurring themes at the conversation grain
(5,035 conversations), further decomposes leading themes
into cited subtheme drivers, and links quantified claims to
evidence sets. Table 8 is the counter-example: theDeep Re-
search Agentconstructs matched rolling 14-day windows
(3,720 versus 3,717 records) and localizes a time-bounded
GLMtool-callclusterto2026-07-08and2026-07-09;Agen-
tic RAGrunsacoarser30-daycomparisonandfindsashare
decreasewithonlyasmallregression/updatesub-signal;the
Context Graph Agentprovidesricherrecent-stateanalysis
but correctly refuses a trend conclusion because the preced-
ing window contains only 24 Trustpilot-only conversations
against a 98.4% Reddit recent window.
Table 9 is the harder counter-example, and we include it
deliberately. Onq013theContext Graph Agentloses by
beingcorrectlycautious,whichiseasytopresentasavirtue.
Onq022it is simply worse. All three systems agree on
the same exact denominator (63 of 113 App Store reviews
rated 4–5), so quantification is not the differentiator;Agen-
tic RAGwinsonanswerqualityandanalyticaldepth(1.000
on both) by developing five specific, separately interpreted
praise themes, while theContext Graph Agentearns per-
fect evidence quality but tier-two quality and depth because
itsstoredtaxonomyroutesthesecond-largestshareofpraise
into a catch-all “general product delight” node. The failure
mode is instructive: when a pre-computed taxonomy lacks
a sufficiently specific node for the question being asked,
the graph-backed answer inherits that coarseness, whereas
a query-time analyst is free to invent the decomposition the
questionneeds.Thisisagenuinelimitationofstoredcontext,
not an artifact of scoring.
B Extended Results
Per-depth means (Table 10) map to RQ2: theContext
Graph Agentleads at every depth level, and theDeep Re-
search Agent’sdecompositionbudgetshowsitsclearestad-
vantageoverAgentic RAGatdepths3and4.Theper-family
means(Table11)aredescriptiveslicesbecausefamiliescon-
tain between 1 and 5 queries; theDeep Research Agentis not uniformly second (Agentic RAGis higher on pric-
ing/packaging and source/metadata), reinforcing the paired
finding that deeper orchestration alone does not impose a
strict capability ordering.
C Evaluation at a Glance
Thebenchmarkevaluatesacompleteanswertoonequestion
atatime.Allthreeanswersareanonymizedandpresentedto-
gethertothecomparativejudge;recordcitationsareaudited
separately.Thethreereporteddimensionsandtheirrolesare
summarized in Table 12.
Answerqualityandanalyticaldepthusecomparativetiers
ona[0.5,1]scaleforcompletedanswers.Evidencequalityis
the harmonic mean of sampled citation precision and com-
parativeevidence-strengthscore;ananswerwithoutcitations
receives zero. Up to 15 frozen member records are sampled
per resolvable citation group. Aggregate citations without
exported members remain traceable but are not treated as
member-audited. Full definitions, equations, and bias con-
trols appear in Section 5.
The companion supplementary appendix,
full-benchmark-responses.pdf, contains ev-
ery benchmark query and the three complete evaluated
responses.Itisgenerateddirectlyfromthecanonicalanswer
packages and frozen score reports, providing a readable
audit trail without expanding the main manuscript by
hundreds of pages.
D Extended Discussion
The results section compresses the mechanism analysis into
a single paragraph. This appendix expands each thread.
Query-timeagencyclosesonlypartofthegap.TheCon-
text Graph Agent’s0.251overalladvantageovertheDeep
Research Agentis much larger than theDeep Research
Agent’s 0.059 advantage overAgentic RAG. Additional
query-time planning and verification therefore close only
partoftheobservedgapinthisimplementation.Becausethe
treatment bundles ingestion, feedback summarization, tax-
onomy induction, metadata mapping, relationship storage,
retrieval, and agent behavior, all results describe the com-
bined workflow rather than the isolated causal effect of any
single component (see Section 7).
Analyticaldepthisthelargestgap.TheContext Graph
Agent’sanswersreadstoredfeedbacksummaries,taxonomy
predictions, metadata, available user information, and their
relationshipstomovefromaportfolio-levelissuetospecific
themes and subthemes while preserving links to support-
ing conversations. Counts and comparisons are computed
fromthatcontextwhenthequestionisasked;ingestion-time
processing makes customer context durable rather than a
transient artifact of one retrieval run.
Quantificationarchitecture,notcitationcount.Ahand-
ful of relevant records can establish that a complaint exists
but cannot establish that it is common or important. The
context-graph answers more often attach counts to bounded
themes and develop their principal drivers; the evaluator re-
wardsthatarchitecturewhenthemainnumericalclaimshave

Table7:Abridgedoutputsforq001:“WhatarethebiggestpainpointsCursorusersarereporting?”Textpreservestheprincipal
grouping,quantification,andevidencepatternofeachevaluatedanswer;itisshortenedforspace.Countsanddenominatorsare
taken verbatim from the canonical answer packages listed in the response appendix.
System Abridged evaluated response Overall
Agentic RAG Runs a model-coded exclusive classification over 8,890 of 9,432 records (94.25% coverage; 2,728
assigned to “other/no clear pain point”) and ranks seven pain-point categories: billing, pricing, usage
limits, credits, overages, and refunds (2,195); model choice, Auto-mode opacity, and context/routing
(1,461); agent/code quality and unsafe edits (875); trust, privacy, and ownership concerns (748);
product instability, regressions, and IDE performance (629); mobile/iOS parity gaps (155); and
support/account escalation (99). Each category is followed by cited examples and an analyst
interpretation.0.556
Deep Research
AgentRuns an exhaustive semantic classification over all 9,432 records into ten exclusive primary
pain-pointcategoriesandreportsbothcountsandcorpusshare:modelqualityandAuto-modecontrol
(892; 9.5%); token burn and unclear usage attribution (628; 6.7%); usage caps and quota lockout
(370; 3.9%); billing surprises and commercial trust (353; 3.7%); long-session context loss (232;
2.5%); update regressions (207; 2.2%); UI churn (156; 1.7%); agent out-of-scope changes (119;
1.3%); support and refund friction (116; 1.2%); and crashes or severe local instability (79; 0.8%). It
also flags 2,133 records classified as positive value or mitigation as a counter-signal.0.776
Context Graph
AgentRanks recurring themes over 5,035 customer conversations (Reddit 4,868 at 96.7%; App Store 114;
Trustpilot 53) organized into seven clusters: quota and usage-limit friction (180), throughput
rate-limit disruption (162), degraded behavior when rate limits are hit (95), Auto routing mismatch
(92), Composer/model quality regressions (86), poor AI output quality (82), and a
service-availability/agent-safety cluster whose top themes are pre-execution safety controls (77) and
service availability issues (77). Each main count links to its evidence set; leading themes decompose
into cited subtheme drivers and are illustrated with verbatim customer quotes.1.000
aggregate drill-down citations and the prose clearly distin-
guishescounts,examples,andlimitations.Itdoesnotreward
citation count by itself.
Counter-examples and boundary conditions.TheDeep
Research Agentwins the autocomplete overview (q006)
and matched-window reliability (q013);Agentic RAG
wins the App-Store-praise question (q022) and exceeds
theDeep Research Agenton the pricing/packaging and
source/metadatafamilies.Query-specificcomputationthere-
fore beats stored concepts when a bespoke comparison is
more important than concept reuse. Unsupported customer
grouping remains a boundary on every architecture: public
feedback can support source, rating, country, version, and
time slices only where those fields are present, and cannot
supportplan,account,revenue,orrenewalcomparisons.The
q013case(Table8,SectionA)illustratestherefusalpattern.
E Extended Evaluation Details
ThisappendixexpandsSection5withthedefinitionsandde-
sign justifications that the main-body summary compresses.
DenominatorconventionforP q.WritingN rel,Npart,and
Nirrelfor the counts of sampled claim–record pairs judged
relevant,partially relevant, andirrelevant, the denominator
isN q=N rel+N part+N irrel:thetotalnumberofsampled
claim–record pairs on queryq. BecauseN qpools across ci-
tations,candidateswithmorecitation-supportedclaimscon-
tribute more evidence to the query-level precision estimate.
We deliberately do not per-citation macro-average, so that a
singlecitationwithonesampledpairdoesnotweightequallywithacitationcarryingfifteen.Wealsodonotrequireamin-
imum retrieval count per citation, so a candidate that cites
three obviously-relevant records and achievesP q= 1.0re-
ceives full sampled precision on that query; this preserves
restraint but is complemented by the comparative evidence-
strength scoreS q, which rewards denominators, drill-down
links, and claim-level architecture rather than citation vol-
ume.
Effective granularity of the tie tolerance.The 0.01
absolute-difference tolerance for paired win/tie/loss is in-
tentionally tighter than the effective granularity of the per-
dimension tier scale (0.25 within a dimension, so an overall
difference of0.25/3≈0.083is the smallest possible non-
zero mean-of-three delta). The tighter tolerance means the
tiecategoryisnearlyalwaysemptyandthewin/losscategory
carries the paired signal.
Blinding and structured-output controls.Aliases and
answerorderaredeterministicallyshuffledperquestion,and
linkdestinationsthatcouldrevealasystemareremovedfrom
the comparison copy. Structured outputs must contain every
candidate and sampled claim–record pair or the evaluation
fails.Citationtotals,member-sampledcitations,traceableag-
gregate citations, unresolved citations, and sampled records
are reported alongside the scores.
F Extended Limitations
The main-body Section 7 discusses the four load-bearing
threats.Foursecondarythreatsroundoutthevaliditypicture
and are included here for completeness.

Table 8: Abridged outputs forq013: “Are there signs of a recently increasing reliability issue?” This counterexample shows
a query on which bespoke query-time reconstruction outperforms the stored-context response. Counts and windows are taken
verbatim from the canonical answer packages listed in the response appendix.
System Abridged evaluated response Overall
Agentic RAG Compares 30-day calendar windows (preceding 2026-05-17 to 2026-06-15 with 557 records; recent
2026-06-16to2026-07-15with8,652records)andfindsthatakeywordreliabilityproxyfallsinshare
from 17.2% to 9.8% even as absolute counts rise. The only rising sub-signal is a
regression/update-languageproxythatgrowsfrom1to37records,and113AppStorereviewsappear
onlyintherecentwindow(noprecedingbaseline).Itlabelsthebroad“reliabilityworsening”claimas
low confidence.0.556
Deep Research
AgentUses closely matched rolling 14-day windows (3,720 versus 3,717 records). Its combined reliability
proxy rises from 301 to 345 (+14.6%), while native-tool, harness, MCP, and file-operation language
rises from 79 to 132 (+67.1%). It localizes a GLM 5.2 tool-call/markup cluster to 2026-07-08 and
2026-07-09 (21 of 25 recent GLM matches), distinguishes increasing, flat, and decreasing signals
across category proxies, and assigns medium confidence.0.930
Context Graph
AgentReports a concrete recent-period reliability signal across 9,408 conversations (May 31 to Jul 15) with
86reliability-adjacentthemesandfivefailureclusters(rate-limitandquotareliability,agentexecution
stalls, core product reliability, billing/metering errors, client-side crashes and freezes), each with
citation-linked recent and preceding counts. It declines to claim an increase because the preceding
period (Apr 16 to May 30) contains only 24 Trustpilot-only conversations while the recent period is
98.4% Reddit; the only apples-to-apples slice is Trustpilot’s24→29move.0.786
Construct validity.Theme definitions affect evidence-
validity estimates. The LLM judge evaluates candidate-
supplied assignments against the question and stated theme,
but there is no independently labeled exhaustive reference
set. The study therefore cannot measure semantic recall or
verifythatareportedrankingfoundeveryrelevantrecord.It
evaluatestheusefulnessandconsistencyoftheresultingcus-
tomer context, not whether the stored taxonomy is uniquely
optimal, fully MECE, or universally reusable. Final- answer
quality is also subjective even with anchored rubrics.
Implementation validity.Candidate quality may reflect
engineering choices rather than an architectural ceiling. We
reduce this risk by using strong reference architectures,
shared controls, fixed budgets, source release, and a ladder
of increasing query-time capability, but do not eliminate it.
Temporal validity.Frontier models, retrieval services,
and vendor systems evolve. Model snapshots, SDK ver-
sions,prompts,prices,andconfigurationhashesaretherefore
recorded(Table13).Thepaperevaluatesthosefrozenimple-
mentations rather than timeless product capabilities. Time
and release queries compare slices inside one frozen snap-
shot;theydonottesthowataxonomyadaptsasnewfeedback
arrives after deployment.
Outcome validity.Workflow scores measure evidence
quality, internal consistency, and reviewer judgments of ac-
tionability. They do not demonstrate that acting on a gen-
erated recommendation improves product or business out-
comes.
G Model Disclosure
The base analyst model and both judges are held constant
across all query-time roles in this run. Table 13 lists the
model identifier, sampling settings, and per-role budget forevery model call that produced or scored evaluated content.
Ingestion-time roles for the context graph system are ad-
ditional dependencies of that system’s stored context and
are reported here as best-available disclosures; the specific
ingestion-timemodelidentifiersusedbythevendorstackare
notheldconstantacrossthethreecandidatesandarerecorded
in the vendor’s own release manifest.
Holding the analyst model constant does not equalize
model capability across the three candidates because the
systems dispatch it to structurally different roles: Agentic
RAG uses a single analyst turn budget; the Deep Research
Agent divides the same budget across five specialized roles
with a bounded verify-and-revise loop; the context graph
agent uses the same base model but reads pre-computed
storedcontextthatothercandidatesdonothave.Theshared-
provider-family design does, however, neutralize the usual
judgeself-preferenceconcernfortherankingwereport.The
judge scores answer text only, and all three candidates’ an-
swertextisgeneratedbythesamegpt-5.5analystmodel;
the context graph system’s vendor ingestion models pro-
duce the stored context its query agent reads, not the prose
the judge sees. Any preference the judge holds for its own
provider family’s stylistic signatures therefore applies uni-
formly across all three candidates and cannot differentially
advantageoneofthem.Whatthisdoesnotneutralizeispref-
erence for particular answerstructures— tables, explicit de-
nominators, aggregate drill-down citations—that happen to
correlate with one candidate’s output format. That residual
is the rubric-alignment threat recorded in Section 7, which
we regard as the substantive remaining judge concern.
H Reproducibility Package
Thestudy’sreproducibilitypackagecomprisesthefollowing
artifacts. These code and data artifacts are separate from

Table 9: Abridged outputs forq022: “What are App Store users praising in high-rated reviews?” This is the benchmark’s
clearestunforcedcontext-graph loss: unlikeq013, where the context graph agent loses by correctly refusing an unsupported
trendclaim,hereitissimplyoutperformed.Allthreesystemsagreeonthesameexactdenominator(63of113AppStorereviews
rated4–5),sothegapisattributabletodecompositionqualityratherthanquantification.Countsareverbatimfromthecanonical
answer packages.
System Abridged evaluated response Overall
Agentic RAG States the denominator (63 of 113 App Store reviews; 54 five-star and 9 four-star; 2026-06-29 to
2026-07-13) and organizes praise into five specific, separately developed themes—mobility (“I can
work anywhere now”), cloud-agent control as the primary use case, concrete productivity gains from
fewer idle agents, focused mobile UX, and continuity with desktop workflows. Each theme carries
several directly quoted reviews with record citations and closes with an explicit analyst interpretation
distinguishing what is being praised (a control plane for agents) from what is not (mobile code
editing).0.887
Deep Research
AgentEstablishes the same denominator with an auditable exact query and rating-coverage checks (63 of
113; 55.8%). Reports five praise themes but scores them as retrieved-evidence counts over a
39-record retrieval set (mobile agent work 18/39; interface polish 8/39; agent visibility 6/39;
pre-existing Cursor affinity 7/39; mixed praise-plus-request 10/39), and states plainly that no
exhaustive classification was scoped to the 63 high-rated reviews, so these are not prevalence
estimates. The disclosure is exemplary but the resulting themes stay closer to retrieval bookkeeping
than to decision-ready analysis.0.608
Context Graph
AgentReports the same 63-conversation denominator with a drill-down citation, and is the only system to
flagthatAppStorecoveragebeginsonlyinlateJune2026(aningestiongapratherthananabsenceof
reviews). It returns five citation-linked praise themes with conversation counts—mobile agent
companion (11), general product delight (8), usability and design (5), value of off-device agent
execution (5), speed (2)—plus explicit mixed-intent and non-additive-count caveats. It earns perfect
evidence quality (1.000) but tier-two answer quality and analytical depth (0.750 each): its
second-largest cluster is the catch-all “general product delight — undifferentiated love for Cursor
overall,” precisely the kind of umbrella label the rubric’s one-problem test penalizes, and its leading
specific theme covers only 11 of 63 reviews.0.833
Table 10: Mean overall score by benchmark depth level.
Depthlevelsarefrozeninthequerymanifest(Table3);slices
are descriptive.
DepthnAgentic Deep Res. Context graph
1 11 0.682 0.6920.955
2 10 0.672 0.6790.967
3 3 0.555 0.8340.892
4 6 0.606 0.7330.996
themanuscriptandarenotincludedinthispreprint’ssource
archive:
1. the corpus manifest (collection dates, per-source counts,
deduplicationcounts,fieldmissingness),theSHA-256of
the frozen normalized corpus file, and a per-record hash
indexcoveringall9,432records(stableid,source,times-
tamp, text hash, text length). Raw record text is deliber-
atelynotredistributed in bulk: doing so would conflict
with platform terms and with our own Ethics and Data
Statement, so corpus identity is verifiable by hash and
rebuildable from the included collection code instead.
Record text that the judge actually scored is retained in-
side the answer packages and judge outputs below, since
it is required to audit the reported claims;
2. all30querydefinitionswithevaluator-onlyfamilyinvari-ants, as-of dates, answer contracts, required operations,
data requirements, count semantics, and known limita-
tions;
3. candidate source code, prompts, tool schemas, model
snapshots, sampling settings, budgets, full-corpus clas-
sification policy, and retry rules for all three systems;
4. the 90 unedited answer packages with citation-
membershipledgers,plusthecompletefrozenevaluation
pass:per-query,per-candidatedimensionscoreswithfull
judge rationales and every citation-audit record;
5. scripts that regenerate every metric, table, and figure in
the paper from these immutable run artifacts, and the
build-time snapshot check (verify_snapshot.py)
thatrefusestoupdategeneratedfilesunlessrunidentifiers
agree across all 90 cells.
Source content that cannot legally be redistributed is rep-
resentedthroughhashesandcollectionmanifestsratherthan
copiedtext.Thereleasemanifestlistsanyomitteditemsand
the reason for omission.
I Answer Package Contract
Every system is converted to the same logical schema:
Stablerecordidentifiersarerequiredformember-levelci-
tation auditing. Aggregate citations with valid drill-down
metadata are reported as traceable; they contribute to com-
parativeevidence-strengthassessmentbutarenotcountedas
member-sampled unless their frozen members are available

Table11:Meanoverallscorebyanalysisfamily.Thesedescriptiveslicesaresmall,especiallythesinglecross-functionalquery,
and are not independent hypothesis tests.
Analysis familynAgentic RAG Deep Research Context graph
Executive VoC 5 0.554 0.8000.974
Product area 5 0.616 0.7640.951
Reliability and incidents 5 0.632 0.7330.918
Pricing and packaging 5 0.700 0.6330.996
Source and metadata 5 0.767 0.6370.928
Competition and retention 4 0.635 0.6931.000
Cross-functional action 1 0.639 0.6951.000
Table 12: High-level evaluation protocol.
Dimension What the evaluator checks
Answer quality Explicit question coverage, organization,
presentation, specificity, useful
quantification, and suitability for the
intended reader.
Analytical depth Diversity and granularity of findings;
coherent themes, drivers, subthemes,
contexts, counterevidence, implications,
and synthesis. Broad umbrella themes are
disfavored when they merge distinct
problems.
Evidence
qualityRelevance of deterministically sampled
cited records plus comparative evidence
strength: claim-level support,
denominators, count semantics, aggregate
drill-down links, and representative record
citations.
Overall Unweighted mean of the three dimensions
for each query, followed by
macro-averaging across the 30 questions.
to the evaluator.
J Benchmark Coverage
The product-adapted query manifest covers seven analysis
families. Counts in Table 15 are frozen before the evaluated
run.
Families mix overview, comparison, diagnosis, and deci-
sion roles. Unsupported grouping is represented where re-
quired metadata are absent. Examples request comparisons
by plan, account, revenue, or renewal state that the public
corpus cannot supply. Systems receive no hidden indication
that refusal is the desired behavior beyond the same field
metadata available during normal analysis.

Table 13: Model identifiers, sampling settings, and budgets for every role in the evaluated run.gpt-5.5denotes the OpenAI
gpt-5.5snapshot in effect on the frozen evaluation date; where a provider does not expose a pinned snapshot, the identifier
resolves to the provider’s default alias at run time and the resolved snapshot is recorded in the run manifest. “Provider default”
means neither the temperature,top_p, norreasoning_effortparameters were overridden.
Role Model Provider Sampling Budget
Agentic RAG
Analyst (planning, tool use,
synthesis)gpt-5.5OpenAI Provider default 50 turns
Retrieval rerankerrerank-2.5Voyage Provider default —
Deep Research Agent
Triage routergpt-5.5OpenAI Provider default 50 turns
Planner / replannergpt-5.5OpenAI Provider default 50 turns
Qualitative / quantitative /
incident specialistsgpt-5.5OpenAI Provider default 50 turns
Verifiergpt-5.5OpenAI Provider default 50 turns
Finalizergpt-5.5OpenAI Provider default 50 turns
Full-corpus classifiergpt-5.5OpenAI Provider default Up to 2
full-corpus
passes; batch size
40
Retrieval rerankerrerank-2.5Voyage Provider default —
Audit / revision loop cap — — — 2 iterations
Customer Context Graph Agent
Query-time agent
(planning, graph traversal,
synthesis)gpt-5.5OpenAI Provider default 900s per query
Ingestion-time feedback
summarizationVendor stack (identifier
recorded in vendor manifest)Vendor Provider default Vendor-controlled
Ingestion-time taxonomy
induction and record-level
predictionVendor stack (identifier
recorded in vendor manifest)Vendor Provider default Vendor-controlled
Ingestion-time embedding /
indexVendor stack (identifier
recorded in vendor manifest)Vendor Provider default Vendor-controlled
Evaluation
Comparative rubric judge
(answer quality, analytical
depth, evidence strength)gpt-5.5OpenAI Provider default 1 pass per query
Citation-audit judge
(deterministic sampling of
up to 15 members per
resolvable citation group)gpt-5.5OpenAI Provider default 1 pass per
resolvable citation
group

Table 14: Required answer-package components for the
shared evaluation.
Component Required content
Run Candidate, model, prompt/config hashes,
corpus hash, timestamps, status
Scope Query ID and requested output context
Answer Final rendered text
Citation ledger CitationID,nearbyclaim,citedrecordor
aggregate members
Usage Model calls, tokens, tools, cost fields,
failures
Table 15: Workflow families in the 30-query benchmark.
FamilynFamilyn
Executive VoC 5 Product area 5
Reliability and
incidents5 Pricing and packaging 5
Source and metadata 5 Competition and
retention4
Cross-functional action 1
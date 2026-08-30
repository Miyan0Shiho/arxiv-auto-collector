# EvoWiki: Incremental State Overwriting and Traceable Question Answering for Cross-Meeting Knowledge Evolution

**Authors**: Dongsheng Chen, Tianyu Wang, Wenhui Que

**Published**: 2026-08-24 13:52:17

**PDF URL**: [https://arxiv.org/pdf/2608.23265v1](https://arxiv.org/pdf/2608.23265v1)

## Abstract
In long-term collaboration spanning multiple meetings, factual states such as decisions and risks are continually revised, overturned, and replaced. Existing long-context methods typically stack the entire history, while many RAG and structured-memory methods organize knowledge as static or append-only facts and rely on semantic relevance at read time. Without explicit modeling of knowledge lifecycles, these approaches may retain conflicting old and new states simultaneously or discard history, leading to stale retrieval and answers that are difficult to verify. We present EvoWiki, an incremental question-answering architecture for dynamic long-form text. EvoWiki decouples offline incremental construction (BUILD) from online structured reading (READ). BUILD captures the intra-meeting micro-evolution from proposal to decision and uses entity version chains and a fine-grained State-Overwrite Protocol to explicitly distinguish current valid states from superseded history while preserving meeting-level provenance anchors. READ bypasses relevance-based Top-k retrieval and performs deterministic entity addressing, temporal resolution, and cross-entity multi-hop aggregation over the complete Wiki to produce grounded and traceable answers. We further introduce CrossMeet, a high-fidelity bilingual benchmark designed to simulate long-term state evolution, covering factual consistency, temporal reasoning, and cross-meeting multi-hop reasoning. Across six datasets and two reader models, EvoWiki improves macro-average Judge Accuracy over the strongest baselines by 9.72 and 10.00 percentage points, respectively. Human evaluation shows that EvoWiki is more robust and factually faithful under frequent state flips, validating valid-state-oriented reading as a reliable approach to cross-meeting knowledge evolution.

## Full Text


<!-- PDF content starts -->

EvoWiki: Incremental State Overwriting and Traceable Question Answering for
Cross-Meeting Knowledge Evolution
Dongsheng Chen, Tianyu Wang, Wenhui Que*
WeChat, Tencent Inc., Beijing, China
{joeydschen, tianyuwang, victorque}@tencent.com
Abstract
In long-term collaboration spanning multiple meetings, fac-
tual states such as decisions, risks, and ownership are contin-
uallyrevised,overturned,andreplaced.Existinglong-context
methods typically stack the entire history, while many RAG,
LLM-Wiki,andstructured-memorymethodsorganizeknowl-
edge as static or append-only facts and rely on semantic
relevance at read time. Without explicit modeling of intra-
meeting decision processes and knowledge lifecycles, these
approachesmayretainconflictingoldandnewstatessimulta-
neously or discard history when updating snapshots, leading
to stale retrieval and answers that are difficult to verify. We
presentEvoWiki (Evolving Wiki), an incremental question-
answering architecture for dynamic long-form text. EvoWiki
decouples offline incremental construction (Build) from on-
line structured reading (Read).Buildcaptures the intra-
meeting micro-evolution from proposal through discussion
to decision and uses entity version chains and a fine-grained
State-OverwriteProtocoltoexplicitlydistinguishcurrentvalid
statesfromsupersededhistorywhilepreservingmeeting-level
provenance anchors.Readbypasses relevance-based Top-k
retrievaloverrawmeetingsandperformsdeterministicentity
addressing, temporal resolution, and cross-entity multi-hop
aggregationoverthecompleteWikitoproducegroundedand
traceable answers. We further introduceCrossMeet, a high-
fidelity bilingual benchmark derived from real-world busi-
ness seeds and designed to simulate long-term state evolu-
tion, covering factual consistency, temporal reasoning, and
cross-meeting multi-hop reasoning. Across six datasets and
two reader models, EvoWiki improves macro-average Judge
Accuracy over the strongest baselines by 9.72 and 10.00 per-
centagepoints,respectively.Furtheranalysesandhumaneval-
uationshowthatEvoWikiismorerobustandfactuallyfaithful
under frequent state flips, validating valid-state-oriented evo-
lutionary reading as a more reliable technical approach to
cross-meeting knowledge evolution.
Introduction
Successive meeting records characterize evolving project
states: key decisions, risks, and responsibilities may be re-
peatedly revised, and historically valid statements may no
longer represent current decisions. Cross-meeting QA must
thereforeidentifytheversionvalidatthequerytimeandtrace
it to its source meeting. Prior work exposes the temporal-
awareness limitations of static models and the challenges of
meeting understanding, long-context QA, and cross-sessionknowledgeupdates(Lietal.2026;Prasadetal.2023;Thonet,
Besacier,andRozen2025;Wuetal.2024),butexistingtasks
still inadequately cover role binding, version replacement,
and multi-hop evidence composition across meetings.
Threeapproachesremaininsufficient.Long-contextmod-
els place meeting histories in a single window, but nominal
length does not guarantee reliable evidence use or reason-
ing (Hsieh et al. 2024; Yen et al. 2025; Modarressi et al.
2025).ConventionalRAG(Lewisetal.2020)ranksevidence
by semantic relevance rather than validity; even recency-
and conflict-aware methods (Vu et al. 2024; Wang et al.
2025) may assign similar relevance to obsolete and current
statementsaboutthesameentity.LLM-Wikiandstructured-
memory methods improve knowledge organization (Sarthi
et al. 2024; Edge et al. 2024; Gutiérrez et al. 2025; Chen
etal.2026b)butdonotmodelthelifecyclesandreplacement
relationsofcross-meetingdecisions.Supersededandcurrent
states may therefore coexist and cause stale retrieval.
We therefore propose EvoWiki (Evolving Wiki).Build
captures proposal–discussion–decision evolution and main-
tains versions and provenance through write-time corefer-
ence resolution, entity routing, and state overwriting;Read
deterministically reads valid states from the complete Wiki
withoutaccessingrawmeetings.Thisasymmetrymovesdis-
ambiguation and conflict resolution to write time. We also
introduce bilingual CrossMeet. Across six datasets and two
readers, EvoWiki surpasses the strongest baselines by 9.72
and 10.00 percentage points, with additional analyses con-
firming robustness and factual faithfulness. The dataset and
implementation will be released upon acceptance. Our con-
tributions are: (1) We proposeEvoWiki, whose asymmet-
ricBuild–Readdesign combines intra-meeting evolution,
entity version chains, and fine-grained state overwriting to
maintain a current valid view with traceable history and
evidence. (2) We constructCrossMeet, a bilingual cross-
meeting benchmark covering factual consistency, temporal
reasoning, and multi-hop QA; each language contains 100
projects, 500 high-fidelity simulated meetings, and 2,000
QA pairs with average contexts exceeding 26K tokens, plus
question-type, reasoning-hop, and cross-meeting evidence-
chain annotations. (3) We enable traceable valid-state rea-
soning with low hallucination risk: experiments across six
benchmarks and two readers, together with state-flip analy-
sis and human evaluation, show reliable valid-state reading
arXiv:2608.23265v1  [cs.CL]  24 Aug 2026

and factually faithful answers with verifiable evidence.
Related Work
Long-Context QA, Long-Term Memory, and
Meetings
Long-context models and RAG offer complementary
performance–costprofiles(Li etal.2024),yetnominalwin-
dow length does not guarantee reliable evidence use (Bai
et al. 2025; Hsieh et al. 2024; Yen et al. 2025; Modarressi
etal.2025).Priorworkcoversconversationalevaluationand
temporal or structured agent memory (Wu et al. 2024; Su
et al. 2026; Latimer et al. 2026; Jiang et al. 2026), meet-
ing understanding, QA, and generation (Prasad et al. 2023;
Thonet,Besacier,andRozen2025;Zhuetal.2025a;Kirstein
et al. 2025), and general multi-hop reasoning (Trivedi et al.
2022). Most tasks focus on single meetings, general mem-
ory, or static evidence; CrossMeet targets state evolution,
role binding, and multi-hop evidence across bilingual meet-
ing sequences.
Retrieval-Augmented Generation and Structured
Knowledge
Classical RAG accesses external memory through sparse,
dense, or hybrid retrieval (Lewis et al. 2020; Robertson and
Zaragoza 2009; Karpukhin et al. 2020; Zhang et al. 2025);
Self-RAG,CorrectiveRAG,andAdaptive-RAGimprovere-
trievalcontrol(Asaietal.2024;Yanetal.2024;Jeongetal.
2024). WiCER compiles sources into persistent LLM-Wiki
memory (Huerta 2026). RAPTOR, GraphRAG/LightRAG,
and HippoRAG 2/LogicRAG respectively use hierarchical
summaries, entity graphs, and associative or multi-step rea-
soning(Sarthietal.2024;Edgeetal.2024;Guoetal.2024;
Gutiérrezetal.2025;Chenetal.2026b);otherworkstudies
adaptive structures, graph expansion, and path pruning (Li
etal.2025;Zhuetal.2025b;Chenetal.2026a).Thesemeth-
ods do not explicitly maintain cross-meeting decision life-
cycles and replacement relations. EvoWiki maintains both
a current valid view and traceable version history at write
time.
Temporal Updating, Traceability, and Evaluation
Temporal retrieval handles time-sensitive and arriving
knowledge (Vu et al. 2024; Zhang et al. 2024; Schumacher
et al. 2025; Hou et al. 2025; Liska et al. 2022). Paramet-
ric editing studies fact localization, multi-hop consistency,
and lifelong updates (Meng et al. 2022; Zhong et al. 2023;
Wang et al. 2024; Fang et al. 2025; Cheng et al. 2025),
but struggles to preserve meeting-level provenance; exist-
ing evaluation also examines factual retrieval and reasoning
(Krishna et al. 2025). EvoWiki instead binds current and
superseded states to evidence in a versioned ledger, jointly
supporting valid-state QA and historical audit. Unlike tem-
poral knowledge graphs or append-only event sourcing, it
processesunstructuredmeetingstreamsandmaintainsentity
lifecycles at write time.Methodology
Task Formulation
Given a chronologically ordered meeting sequence
D={M 1, M2, . . . , M T},(1)
thegoalistoanswerqueryqafterprocessingtheT-thmeet-
ing. Unlike static multi-document QA, an entity’s attributes
maybesupplemented,replaced,oroverturnedbylatermeet-
ings; the system must identify the state valid at the query
timewhileretainingthehistoricalevidenceforitsevolution.
EvoWiki represents atomic meeting knowledge as
si=⟨e i, ri, vi, τi, ℓi, pi⟩,(2)
wheree i, ri, vi, τi, ℓidenotethecanonicalentity,attributeor
relation, state value, effective time, and lifecycle label, re-
spectively,whilep i= (m i,spani)denotesthesourcemeet-
ing and evidence span. The system maintains a structured
WikiW tunder the causal constraint
Wt=F Build(Wt−1, Mt),(3)
soprocessingM taccessesonlythepriorWikiandthecurrent
meeting, never future meeting information.
Build–ReadArchitecture
Figure 1 presents EvoWiki’s three components. Offline
Buildperforms temporal alignment, intra-meeting micro-
evolution parsing, write-time coreference resolution, infor-
mation extraction, entity routing, and state overwriting.
EvoWiki Core stores entities, active states, complete ver-
sion histories, lifecycle labels, and provenance anchors. On-
lineReaduses the complete Wiki as its sole context and
generates an answer through deterministic entity address-
ing,cross-entitymulti-hopreasoning,andtemporalaggrega-
tion; provenance anchors bound to supporting states enable
evidence-trail recovery. This read/write asymmetry moves
temporalalignment,entitydisambiguation,andconflictreso-
lutiontowritetime,letsqueriesreusetheconstructionresult,
and transforms online QA from re-identifying valid facts in
conflicting passages into reading valid versions from a nor-
malized state space.
Offline Incremental Construction
For each meetingM t,Buildexecutes four steps. First, tem-
poral alignment (Resolve) normalizes the meeting timeline
fromdatesandrelativetimeexpressionsandparsestheintra-
meeting micro-evolution from proposal through discussion
to decision, preventing candidate plans from being written
as final states. Second, write-time coreference resolution
(Coref) combines the current meeting with the entity in-
dex inW t−1to map expressions such as “the client,” “the
aboverisk,”orroletitlestocanonicalentities.Factextraction
(Extract)thenproducescandidateupdates,andentityrout-
ing(Route)bindsthemtothecorrespondingentity–attribute
slots.Finally,theState-OverwriteProtocolmergestheupdate
setU tinto the Wiki:
Ut= Route(Extract(Coref(Resolve(M t),Wt−1))),(4)
Wt= Overwrite(W t−1,Ut).(5)

TOP: ONLINE QAOutput—Final AnswerThe final approved budget is $50,00.
Grounded & TraceableUpdated in Session 3, overriding the initial $30,00 proposed in Session 1. [Source: Session 3, Span 12]Input—User QueryWhat is the final approved budget for the Q3 marketing campaign?
READDeterministic Entity AddressingPrecisely locates the target entity node, zero misses.
Multi-hop & TemporalAggregationLogical chaining and temporal/numerical aggregation across meeting states.
Grounded GenerationGenerates the answer from the retrieved definite state, suppressing hallucination.
Retrieval-Free Read & Reasoning
EvoWiki Core
Version HistoryAll past states retained.
Lifecycle TagsActive / Overturned for every state.
Provenance AnchorEach state anchored to its origin (session + span).
EntityLayerState LifecycleEntityAEntityBState @TnActive…State @T1OverturnedState @TnActive…State @T1Overturned
Structured Knowledge Base · Entity-Centric
Multi-Session Transcripts
Temporal Sequence&IncrementalInputChronological AlignmentAbsolute date + relative time delta.
Intra-meeting Micro-evolutionTracks propose →debate →resolve within asingle session.
Write-time CoreferenceAligns ambiguous mentions against the existing entity index.
Information ExtractionRecall-first, zero-loss capture.
State OverwriteOn override, old value is retained but marked Overturned —never deleted.New StateActiveOld StateOverturnedBOTTOM: OFFLINE BUILDIncremental Construction with State OverwritePull Existing Entity Index (Feedback Loop)Figure1:EvoWikiarchitecture.Buildwritesmeetingschronologicallyintoanentity-centeredcorewithversionandprovenance
information.Wiki-onlyReadusesdeterministicaddressingandcross-entitytemporalaggregationtoproduceagroundedanswer
while retaining state-level provenance for evidence-trail recovery.
Write-time coreference resolution and entity routing con-
solidate cross-meeting aliases, job titles, and elliptical ex-
pressions into one entity version chain, enabling strong role
binding.
State Overwriting and the Structured Wiki
EvoWiki represents its knowledge base as
Wt= (E t,Ht,At,Pt),(6)
whose components denote entities and relations, complete
version histories, the current active-state view, and prove-
nance anchors. For each entity–attribute pair(e, r), the sys-
tem maintains a chronologically ordered version chain
H(t)
e,r=⟨s1
e,r, s2
e,r, . . . , sn
e,r⟩.(7)
When a new state replaces, revises, or revokes the current
active state, the old record is not physically deleted. The
protocol closesitsvalidity interval,labels itOverturned,
appendsthenewstatewithanActivelabel,andcreatesan
explicit version-replacement edge. Facts that do not conflict
with the attribute remain in their respective slots. For any
fixed entity–attribute pair, the protocol maintains at most
oneactiveversion,where1[·]denotestheindicatorfunction:
X
s∈H(t)
e,r1[ℓ(s) = Active]≤1.(8)Thecurrentstatethereforehasauniquecanonicalentrypoint,
whilesupersededversionsandtheirprovenanceremainavail-
ableforhistoricalqueriesandaudit.Ifastate’svalidityinter-
valisI(s) = [τ start, τend),aquerytargetingtimeτ qreadsthe
version satisfyingτ q∈I(s); a current-state query without
an explicit time defaults to the active version.
Wiki-Only Online Reading
DuringRead, the complete Wiki is serialized as the sole
knowledge context:
CT= Serialize(W T),(ˆy, ˆSq) =G(q|C T),(9)
whereˆyis the answer and ˆSqcontains its supporting Wiki-
state identifiers. Onlyˆyis evaluated; ˆSqremains metadata
for evidence-trail recovery via the traceability mechanism
describedbelow.Readisisolatedfromrawmeetings:itper-
formsnosource-textTop-kretrievalandhasnofallbackthat
accesses raw meetings when the context window permits.
Online inference locates valid states, performs multi-hop
or temporal aggregation along entity relations and version
timelines, and generates answers from those states and their
provenance anchors. “Deterministic Entity Addressing” in
Figure 1 denotes reading valid states in the Wiki by entity,
attribute,andlifecyclelabelratherthanretrievingrawmeet-
ings or an external corpus. BecauseBuildhas written valid

statesintoaunifiedstructure,Readbypassessource-textrel-
evancerankingandTop-kerrorswithoutdecidingthecurrent
version among mixed old and new passages.
Traceability and Cost
Everystateislinkedtoitssourcemeetingandevidencespan,
while version-replacement edges record how states evolve.
For the supporting-state set ˆSqreturned with an answer, its
evidence chain is
Trace(ˆy) =[
s∈ˆSqp(s)∪VersionEdges( ˆSq).(10)
The system can therefore locate the meeting evidence sup-
porting the final answer and trace why one state replaced
an earlier version. If all meetings produceUstate updates,
retaining all historical versions requiresO(U)storage. For
a serialized Wiki of lengthS W, each query has an input-
context size ofO(S W)and makes no retrieval call to an
external corpus. The mainBuildcost occurs during offline
writingandcanbeamortizedoversubsequentqueries;corre-
spondingly, complete-Wiki input length grows linearly with
SW.Thislinearreadingcostisanexplicitarchitecturaltrade-
off for broader state coverage and reduced risks of retrieval
omission and obsolete-version selection.
Experimental Setup
CrossMeet Construction and Quality Validation
CrossMeet follows a controlled project-blueprint–meeting-
sequence–cross-meeting-QA pipeline. Its English and Chi-
nese versions are generated independently rather than trans-
lated from one another. To improve the business realism
of the simulated meetings, project blueprints are expanded
fromabstractprojecttopicsrepresentingcommonworkflows
on large-scale digital platforms. Only generic topic cate-
gories and collaboration patterns are used, without any real
business records, personal identifiers, operational metrics,
or sensitive business content. Claude Opus 4.5 (Anthropic
2025)constructsthemeetingsandQAinstances;Gemini3.1
Pro (Google DeepMind 2026), Claude Opus 4.7 (Anthropic
2026),andDeepSeek-V4-Pro(Xuetal.2026)cross-validate
answer–evidence consistency, answerability, and question-
typelabelsandindependentlyclassifyquestiontypestocom-
pute Fleiss’κas agreement over the three categories.
Table 1 summarizes dataset scale and quality. Each lan-
guage contains 100 projects, 500 consecutive meetings, and
2,000 QA pairs, with average context lengths of 26,861 and
29,480 tokens for English and Chinese, respectively. Fleiss’
κfor the three-model question-type classification is 0.937
in English and 0.909 in Chinese, indicating high agreement
in the task taxonomy. Table 2 reports question-type and hop
distributions: factual-consistency, temporal-reasoning, and
cross-meeting multi-hop questions account for 50%, 30%,
and 20%. Every question requires at least two hops; four-
and five-hop instances constitute 21.2% of the English data
and19.3%oftheChinesedata,ensuringlong-evidence-chain
coverage.Metric CM-EN CM-ZH
Topics 100 100
Meetings 500 500
QA Pairs 2,000 2,000
Context Tokens (Avg.) 26,861 29,480
Answer Tokens (Avg.) 64.6 108.1
Reasoning Hops (Avg. / Max.) 2.84 / 5 2.77 / 5
Taxonomyκ0.937 0.909
Table 1: Statistics of CrossMeet-EN and CrossMeet-ZH.
Taxonomyκmeasures question-type classification agree-
ment among three validation models.
Dimension Type CM-EN CM-ZH
QuestionFactual Consistency 1000 (50.0%) 1000 (50.0%)
Timeline Reasoning 600 (30.0%) 600 (30.0%)
Multi-hop Reasoning 400 (20.0%) 400 (20.0%)
Hops2 Hops 941 (47.0%) 1005 (50.2%)
3 Hops 635 (31.8%) 608 (30.4%)
4 Hops 235 (11.8%) 222 (11.1%)
5 Hops 189 (9.4%) 165 (8.2%)
Table 2: Distribution of question types and reasoning hops
in CrossMeet.
Independent human validation.We sample 150 QA in-
stances and their complete state-evolution paths from each
CrossMeetlanguage,yielding300of4,000instances.Three
independentannotatorswithnaturallanguageprocessingex-
perience, none involved in data generation, review every
sample without access to generation prompts or automatic
validation labels. Each receives the question, reference an-
swer, cited evidence, and complete evolution path. They as-
sess whether the answer is supported, whether proposal–
discussion–decision transitions are authentic and coher-
ent, and whether question-type and reasoning-hop annota-
tions are correct. Each annotator judges the samples inde-
pendently; pass rates follow majority decisions, and inter-
annotator agreement is measured with Fleiss’κ. Table 3
summarizes the results.
All dimensions exceed a 95% pass rate, with agreement
of at least 0.84. The 98.3% answer–evidence rate confirms
thatreferenceanswersaregroundedinthesuppliedmeeting
evidence. The meeting/update result further indicates that
the simulated discussions preserve credible proposal, revi-
sion, rejection, and resolution processes rather than merely
presentingdisconnectedfacts.Metadataaccuracyshowsthat
thebenchmark’squestioncategoriesandannotatedreasoning
depth are also reliable. This jointly validates both reasoning
targets and metadata used in stratified analyses. The audit
thuscomplementsautomaticcross-validationwithdirecthu-
man evidence of benchmark fidelity and annotation quality.
Evaluation Datasets
In addition to CrossMeet-EN and CrossMeet-ZH, experi-
ments use four public benchmarks. MeetingQA is based on

Evaluation Dimension Pass Rate Fleiss’κ
Answer–Evidence 98.3% 0.89
Meeting/Update Coherence 95.7% 0.84
Type/Hop Accuracy 97.7% 0.91
Table3:Humanvalidationon300CrossMeetinstancessam-
pled equally from English and Chinese.
human-recorded AMI scenario meetings and evaluates QA
over meeting transcripts (Prasad et al. 2023); ELITR-Bench
is based on ASR transcripts of real project meetings and
covers retrieval, summarization, and QA in long meetings
(Thonet, Besacier, and Rozen 2025); LongMemEval tests
long-term memory and knowledge updates across sessions
(Wu et al. 2024); and MuSiQue evaluates static compo-
sitional multi-hop reasoning (Trivedi et al. 2022). Meet-
ingQA and ELITR-Bench both use human meeting tran-
scripts as their core data. Together, the six datasets cover
bilingual cross-meeting evolution, single-meeting under-
standing, long-term interactive memory, and general multi-
hop reasoning.
Baselines, Reader Models, and Metrics
We compare nine representative baselines: Direct LLM for
full-context reading (Li et al. 2024); VanillaRAG (BM25),
VanillaRAG (Dense), and VanillaRAG (Hybrid) for sparse,
dense, and fused retrieval (Robertson and Zaragoza 2009;
Karpukhin et al. 2020; Zhang et al. 2025); RAPTOR for
hierarchical summarization (Sarthi et al. 2024); GraphRAG
(Edgeetal.2024)andLightRAG(Guoetal.2024)forgraph
retrieval; HippoRAG 2 for associative memory (Gutiérrez
et al. 2025); and LogicRAG for query-time logical decom-
position (Chen et al. 2026b). Dense retrieval uses Qwen3-
Embedding-8B (Zhang et al. 2025). Together, they span di-
rect reading, sparse/dense retrieval, hierarchical compres-
sion, graph retrieval, associative memory, and logical plan-
ning. All retrieval baselines share tiered chunking and bud-
gets scaled to dataset context length; Direct LLM reads the
full raw context within each reader’s native window.
EvoWiki’s offlineBuildstage uniformly uses DeepSeek-
V4-Flash (Xu et al. 2026) to perform temporal alignment,
intra-meeting micro-evolution parsing, write-time corefer-
ence resolution, fact extraction, and entity routing, and to
producethestructuredWikiaccordingtotheState-Overwrite
Protocol.StrongstructuredbaselinessuchasGraphRAGand
HippoRAG 2 likewise strictly follow their officially recom-
mendedcompleteofflineLLM-basedgraphconstructionand
indexing pipelines. Thus, all methods start from the same
raw meetings and retain their native construction and read-
ing pipelines: the main results compare end-to-end system
capability, while the matched ablation study identifies the
contributions of state overwriting and structural tags. On-
lineReaduseseitherDeepSeek-V4-Flash(Xuetal.2026)or
Qwen3.5-397B-A17B(QwenTeam2026)asthereader;both
settingssharethesameBuildconfigurationtoisolatereader-
modeldifferences.EvoWikistrictlyfollowsWiki-onlyRead:the online reader receives only the complete Wiki produced
byBuildand never accesses raw meeting records. Under
the same reader model, all methods use identical questions,
prompts,andgenerationsettings.Wedonotprovideconven-
tional RAG with EvoWiki’s intermediate structures because
doingsowouldalteritsnativeraw-textretrievalparadigmand
make it dependent on EvoWiki’s extractor, rather than com-
pareeachmethod’sownend-to-endknowledge-organization
capability.
Judge Accuracy is the primary metric. Following the
LLM-as-a-Judge paradigm (Zheng et al. 2023), Gemini 3.1
Pro (Google DeepMind 2026) assigns 0 to an incorrect an-
swer, 0.5 to a correct but incomplete core conclusion, and
1 to a fully correct answer. Each method runs five times
with matched online decoding while its offline Wiki, graph,
or index remains fixed; Table 4 reports mean scores from
this sole primary judge. Scores are averaged per dataset and
then equally macro-averaged across all six. We also report
abstention and anonymized pairwise human win/tie/loss re-
sults, and analyze ablations, state flips, evidence positions,
error attribution, and qualitative cases; the pairwise results
are independently judged by human evaluators.
Retrieval and decoding configurations.All retrieval
baselines use a unified length-tiered configuration deter-
mined solely by each dataset’s median context length
rather than tuned on experimental results. MeetingQA and
MuSiQue use a chunk size of 200 tokens, an overlap of 25
tokens,andTop-3retrieval;CrossMeet-EN,CrossMeet-ZH,
ELITR-Bench, and LongMemEval use a chunk size of 512
tokens, an overlap of 64 tokens, and Top-10 retrieval. RAP-
TOR halves the leaf-chunk configuration to 100/12 tokens
forshort-contextdatasetsand256/32tokensforlong-context
datasets,whileretainingTop-3andTop-10retrieval,respec-
tively.
Exceptforthemulti-runstabilityexperiment,allmethods
use the same deterministic final-reader configuration: tem-
peratureissetto0.0,themaximumgenerationlengthis1,024
tokens,andexplicitreasoningoutputisdisabled.Top-pisnot
explicitly specified and therefore follows the default of the
corresponding serving endpoint. The five-run stability anal-
ysis varies only the online-generation seed and temperature,
using 0.1, 0.3, 0.5, 0.7, and 0.9; the offline Wiki, graph, or
indexremainsfixed,andthemaximumgenerationlengthand
all other settings are unchanged.
DirectLLMperformsnochunking,retrieval,orstructural
compression and instead feeds the complete raw context di-
rectly to the corresponding reader model. Qwen3.5-397B-
A17Bhasanativecontextwindowof262,144tokens(256K),
while DeepSeek-V4-Flash supports 1,048,576 tokens (1M);
all raw experimental contexts fall within the input capacity
of the corresponding model.
Main Results
Overall Performance
Table 4 summarizes the main results. With DeepSeek-V4-
Flash, EvoWiki achieves the highest Judge Accuracy on all
six datasets and a macro-average of 60.09, 9.72 percent-
age points above LogicRAG (50.37). With Qwen3.5-397B-

Model Method CM-EN CM-ZH ELITR LongMemEval MeetingQA MuSiQue Avg
DeepSeek-V4-FlashDirect LLM 65.90 66.03 59.62 19.80 37.82 49.03 49.70
VanillaRAG (BM25) 43.98 54.93 48.46 37.60 37.40 45.92 44.72
VanillaRAG (Dense) 51.10 59.43 55.38 34.60 37.12 46.13 47.29
VanillaRAG (Hybrid) 51.60 58.75 51.15 39.20 37.06 46.03 47.30
RAPTOR 49.20 55.20 46.54 40.10 33.84 48.57 45.57
GraphRAG 56.33 61.30 58.08 39.10 35.40 51.76 50.33
HippoRAG2 51.60 59.20 56.15 39.70 36.17 52.32 49.19
LightRAG 52.40 60.30 52.69 44.30 35.72 49.50 49.15
LogicRAG 54.20 58.50 54.62 48.50 29.42 56.99 50.37
EvoWiki (Ours)68.78 69.55 60.77 61.60 39.16 60.67 60.09
Qwen3.5-397B-A17BDirect LLM 71.43 75.13 63.46 22.20 39.92 45.28 52.90
VanillaRAG (BM25) 44.98 61.18 54.62 42.20 40.13 42.72 47.64
VanillaRAG (Dense) 52.45 65.63 59.62 41.00 39.76 44.06 50.42
VanillaRAG (Hybrid) 53.35 65.45 58.85 43.20 40.33 43.42 50.77
RAPTOR 50.20 60.30 53.46 43.30 36.33 48.14 48.62
GraphRAG 57.05 65.23 58.08 47.40 36.11 52.15 52.67
HippoRAG2 52.65 64.50 59.23 41.90 39.68 51.65 51.60
LightRAG 53.20 66.20 58.46 46.30 37.02 53.43 52.44
LogicRAG 55.35 67.83 59.62 51.90 28.53 54.90 53.02
EvoWiki (Ours)73.18 77.28 64.62 66.00 40.81 56.25 63.02
Table 4: Judge Accuracy (%) across six datasets. Best results within each reader block are shown in bold.
Variant CM-EN CM-ZH ELITR LongMemEval MeetingQA MuSiQue Avg
w/o Overwrite 62.63 65.13 41.54 47.20 36.37 55.88 51.46
w/o Coref 61.03 66.60 43.46 43.20 35.74 55.52 50.93
w/o Entity 63.88 68.23 47.31 49.00 36.65 57.30 53.73
w/o Tags 60.43 62.78 42.31 41.40 35.50 56.29 49.79
EvoWiki68.78 69.55 60.77 61.60 39.16 60.67 60.09
Table 5: Component ablation of EvoWiki with DeepSeek-V4-Flash, reporting Judge Accuracy (%) across six datasets.
A17B, its macro-average is 63.02, 10.00 points above Logi-
cRAG (53.02). Consistent gains across readers indicate that
the improvement stems from knowledge-state organization
rather than a particular reader. On LongMemEval, EvoWiki
scores 61.60 and 66.00, leading the strongest baselines by
13.10and14.10pointsandconfirmingthatstateoverwriting
mitigates long-running knowledge conflicts and evidence-
localizationdifficulty;itsleadontheotherfivedatasetsshows
that the advantage extends beyond CrossMeet.
Ablation Study
Table5quantifiesthecontributionofeachcomponentunder
DeepSeek-V4-Flash. The w/o Overwrite variant keeps up-
stream temporal alignment, intra-meeting micro-evolution
parsing, coreference resolution, fact extraction, entity rout-
ing, the entity-centered Wiki, and the reader unchanged; at
write time, it discards Active labels emitted by the generic
Buildprompt, suppresses Overturned labels, and disables
lifecycle transitions and version-replacement edges, causing
conflicting states to coexist in an append-only form without
lifecycle distinctions. It therefore provides a structured con-
trol with the same extraction strength as full EvoWiki but
without version invalidation. The w/o Tags variant retainsthe same Wiki content but removes all structural tags used
to organize entities, relations, time, states, and provenance,
thereby measuring the overall effect of explicit structural
representation. Full EvoWiki achieves an average accuracy
of 60.09. Removing state overwriting, write-time corefer-
enceresolution,entity-centeredorganization,orallstructural
tags reduces the average to 51.46, 50.93, 53.73, and 49.79,
corresponding to drops of 8.63, 9.16, 6.36, and 10.30 per-
centage points. The components play complementary roles:
removing state overwriting reintroduces historical conflicts
into the reading context; removing write-time coreference
resolution causes entity fragmentation and broken evidence
chains; and removing entity-centered organization increases
thedifficultyofcross-documentaddressingandaggregation.
Removing all structural tags causes the largest drop (10.30
points), showing that entity, relation, temporal, state, and
provenance boundaries jointly guideRead; without them,
the reader must reconstruct structure and version validity
from flattened content and is more likely to confuse current
and superseded states.

Method CM-EN CM-ZH ELITR LongMemEval MeetingQA MuSiQue Macro-Avg
Direct LLM65.90±0.74 66.03±0.68 59.62±0.91 19.80±1.22 37.82±0.95 49.03±0.83 49.70±0.89
VanillaRAG51.60±1.10 58.75±1.05 51.15±0.96 39.20±1.18 37.06±1.02 46.03±0.93 47.30±1.04
GraphRAG56.33±0.82 61.30±0.75 58.08±0.79 39.10±1.06 35.40±0.88 51.76±0.81 50.33±0.85
LogicRAG54.20±0.76 58.50±0.72 54.62±0.83 48.50±0.97 29.42±1.10 56.99±0.78 50.37±0.86
EvoWiki68.78±0.52 69.55±0.49 60.77±0.58 61.60±0.71 39.16±0.69 60.67±0.55 60.09±0.59
Table 6: Performance stability across five runs under the DeepSeek-V4-Flash reader. Results are Judge Accuracy (%) reported
as mean±standard deviation; VanillaRAG denotes the Hybrid variant.
B1 (27.4%)
B2 (11.8%)
B3 (15.1%)R2 (1.9%)R3 (13.2%)R4 (3.8%)R5 (5.7%)R6 (12.7%)R7 (8.5%)
B
UILD
READ
Total
212
BUILD  —  54.2%
B1  Information Omission  (27.4%)
B2  Factual Distortion  (11.8%)      B3  Overwrite Failure  (15.1%)          
B4  Linking Failure  (0%)                   
READ  —  45.8%
R1  Retrieval Miss  (0%)                 
R2  Reference Confusion  (1.9%)    
R3  Stale Retrieval  (13.2%)           
R4  Multi-hop Failure  (3.8%)          R5  Aggregation Error  (5.7%)           
R6  Incomplete Response  (12.7%)    
R7  Generation Hallucination  (8.5%)
Figure2:Errorattributionfor212EvoWikierrors,separated
intoBuildandReadcauses.
Performance Stability
We compare EvoWiki with Direct LLM, VanillaRAG,
GraphRAG, and LogicRAG under the DeepSeek-V4-Flash
reader. Each method is run five times with different online-
generationseedsandtemperaturesin{0.1,0.3,0.5,0.7,0.9},
whileitsofflineWiki,graph,orindexisconstructedonceand
heldfixedacrossruns;Gemini3.1Proscoresalloutputs.Ta-
ble 6 reports mean Judge Accuracy and deviation across
these decoding conditions. Because seed and temperature
vary jointly, the deviation measures overall online-decoding
robustnessratherthanseed-onlyvariation.EvoWikiachieves
the highest mean on all six datasets and the lowest macro-
average deviation (0.59).
Analysis and Discussion
Inference Efficiency and Construction Cost
We measure QA efficiency on CrossMeet-EN with
DeepSeek-V4-Flash as the reader, using 200 QA instances
from 20 projects. Questions are processed serially, whileMethod Latency (s) Read Tokens Build Tokens
EvoWiki (Ours) 3.60 17,143 114,016
Direct LLM 6.54 26,374 –
VanillaRAG (Hybrid)1.01 3,481–
GraphRAG 1.94 9,452 131,298
LogicRAG 4.68 11,238 –
LightRAG 1.78 7,352 67,263
Table 7: Mean query latency and read/build token usage on
CrossMeet-EN.Readcostsareaveragedperqueryandbuild
costs per project.
Method Gemini 3.1 Pro Claude Opus 4.7 DeepSeek-V4-Pro
EvoWiki60.50 60.17 59.83
Direct LLM 50.17 50.50 49.83
VanillaRAG 46.83 46.50 46.33
GraphRAG 49.83 50.00 49.50
LogicRAG 50.00 50.33 49.33
Table 8: Judge Accuracy (%) on the 300-question subset.
Rows denote evaluated methods and columns denote judge
models. Agreement with Gemini 3.1 Pro over 1,500 re-
sponsesisκ= 0.901forClaudeOpus4.7andκ= 0.887for
DeepSeek-V4-Pro. VanillaRAG denotes the Hybrid variant.
projectindexes,graphs,andWikistructuresareconstructed,
warmed, and cached before query timing. Table 7 reports
non-streaming end-to-endReadlatency, meanReadtokens
perquery,andmeanone-timeBuildtokensperproject;“–”
denotesmethodswithoutofflineconstruction.Latencyisav-
eraged across all instances and measures complete response
time rather than time to first token. Compared with Direct
LLM, EvoWiki reduces latency by 45.0% andRead-token
usage by 35.0%. Retrieval methods remain cheaper online
because they expose only a small Top-kcontext, whereas
EvoWiki reads the complete Wiki to preserve access to
valid states and their histories. The measurements capture
the efficiency–coverage trade-off, with the one-timeBuild
cost amortized across subsequent queries.
Cross-Model Judge Robustness
We sample 50 questions from each benchmark to form a
300-questionsubsetthatpreservestheequal-datasetweight-
ing of the primary metric. Under the DeepSeek-V4-Flash
reader, outputs from EvoWiki and four representative base-
lines yield 1,500 anonymized responses per judge. All three
judges grade exactly the same predictions. Gemini 3.1 Pro,

Source (Summary)The project’s “primary launch risk” evolved twice across meetings. It was first flagged as anSOC2 certification delay;
in Meeting 4 it was updated toinsufficient customer onboarding capacity(only 2 customer support staff covering 3
enterpriseaccounts);inMeeting5thisriskwasoverriddenagaintothethirdcustomer’scontractnegotiationpossibly
slippingpastQ4,affectingtheARRmilestone.Thequestionasksfortheprimarylaunchriskatthedeliverystage,i.e.,
it requires the latest (Meeting 5) state, not any overturned earlier risk.
QuestionWhat was the primary launch risk identified for the Nexus Analytics project at the delivery stage?
ReferenceThird customer contract negotiation may slip past Q4, affecting ARR milestone
EvoWiki Third customer contract negotiation potentially slipping past Q4, affecting ARR milestone.
Direct LLMCustomer onboarding capacity.(Landed on the Meeting-4 risk, one version behind—missed the Meeting-5 override.)
VanillaRAGSOC2 certification delay.(Landed on an even earlier risk that had already been overturned twice.)
GraphRAGCustomeronboardingcapacity,withonly2customersupportstafffor3enterpriseaccounts.(SamestaleMeeting-4state
as Direct LLM.)
LogicRAGUnanswerable(failed to locate any valid risk description and abstained.)
Table 9: Case study of cross-meeting launch-risk evolution and final-state recovery.
1 2 3 4
Number of State Transitions3040506070Accuracy (%)
EvoWiki (Ours)
Direct LLM
Vanilla RAG
GraphRAG
LogicRAG
Figure 3: Judge Accuracy as the number of state flips in-
creases on CrossMeet-EN.
Claude Opus 4.7, and DeepSeek-V4-Pro independently ap-
plythesame0/0.5/1rubricusingonlythequestion,reference
answer, and system response; method identities and other
judges’ decisions are hidden. Quadratic-weighted Cohen’s
κmeasures each auxiliary judge’s agreement with Gemini
over all responses and respects the ordinal distance between
incorrect,partiallycorrect,andfullycorrectratings.Table8
reports both system scores and cross-judge agreement. All
threejudgesrankEvoWikifirst.Itsmarginsoverthestrongest
baseline are 10.33, 9.67, and 10.00 points under Gemini,
Claude, and DeepSeek, respectively. Claude and DeepSeek
achieve agreement scores of 0.901 and 0.887 with Gemini,
indicating that both the absolute assessments and method
ordering remain stable across model families rather than
reflecting one judge’s preference. Because the comparison
includes full-context reading, conventional retrieval, static
graphorganization,andlogicalgraphreasoning,theconclu-
sion is robust to both judge origin and baseline paradigm.
Error Attribution
Figure2divides212EvoWikierrorsintoBuild(115,54.2%)
andRead(97, 45.8%) failures. The main failure modes
are extraction omission (27.4%), overwrite failure (15.1%),
0–20% 20–40% 40–60% 60–80% 80–100%
Relative Position of Evidence in Context01020304050607080Accuracy (%)
EvoWiki (Ours) Direct LLM Vanilla RAG GraphRAG LogicRAGFigure4:JudgeAccuracybytherelativepositionofsupport-
ing evidence in LongMemEval.
stale-state reading (13.2%), incomplete answers (12.7%),
andfact-extractiondistortion(11.8%);generationhallucina-
tion accounts for 8.5%.Buildfailures mainly indicate that
keyfactswerenotwrittencompletelyoraccurately,whereas
Readfailuresmoreoftenreflectobsolete-versionselectionor
insufficientanswercoverage.Noentity-linkingorWikistate-
locationerrorsareobserved,suggestingstableentityrouting
and deterministic addressing. Overall, the main bottleneck
hasshiftedfromretrievalrecalltoofflinewritefidelity,com-
plexstateupdates,andanswercompletenessduringreading.
State Flips and Evidence Position
Figure 3 groups CrossMeet-EN questions by entity-state
flips; here and in subsequent analyses, VanillaRAG denotes
the VanillaRAG (Hybrid) variant in Table 4. As flips rise
from 0–1 to four, EvoWiki declines only from 71.03% to
66.90%, a drop of 4.13 percentage points. Direct LLM falls
from 70.62% to 61.20%, while VanillaRAG, GraphRAG,
and LogicRAG fall from 64.57%, 68.93%, and 65.18% to
42.80%,46.80%,and45.32%.Onthehigh-conflictfour-flip
subset, EvoWiki leads these methods by 5.70, 24.10, 20.10,
and21.58points,showingthatexplicitstateoverwritinglim-
its degradation as version conflicts accumulate.

0 20 40 60 80 100Direct LLM
Vanilla RAG
GraphRAG
LogicRAG31 43 26
51 28 21
43 34 23
49 30 21Faithfulness
0 20 40 60 80 100
Preference (%)26 46 28
56 23 21
52 25 23
63 21 16Completeness
0 20 40 60 80 10048 24 28
43 23 34
42 26 32
28 40 32ConcisenessWin (EvoWiki) Tie Loss (Baseline)Figure5:PairwisehumanevaluationofEvoWikiagainstrepresentativebaselinesonfaithfulness,completeness,andconciseness.
Results are percentages of wins, ties, and losses from EvoWiki’s perspective.
Figure 4 divides the 470 answerable LongMemEval in-
stances into five evidence-position bins. EvoWiki remains
between 63.58% and 70.10%, with a spread of 6.52 points.
In the middle bin, Direct LLM, VanillaRAG, GraphRAG,
andLogicRAGscore11.36%,30.93%,32.12%,and43.24%,
whereasEvoWikireaches63.58%,leadingtherunner-upby
20.34points.BecauseReadoperatesonaWikireorganized
by entity and state, answers no longer depend on evidence
position in the original meeting sequence, effectively miti-
gating the Lost-in-the-Middle effect (Liu et al. 2024).
Qualitative Case Study
Table9presentscross-meetingriskevolutionandfinal-state
recovery. Nexus Analytics’ primary risk changes from a
SOC2certificationdelaytoinsufficientcustomer-onboarding
capacity and is ultimately overwritten by the risk that the
third customer’s contract negotiation may slip beyond Q4
andaffecttheARRmilestone.EvoWikireturnsthefinalvalid
risk.DirectLLMandGraphRAGremainattheintermediate
Meeting4state;VanillaRAGreturnstheSOC2risk,already
superseded twice; and LogicRAG abstains incorrectly. This
case shows that semantically relevant evidence need not be
thecurrentlyvalidstateandhighlightstheroleofstateover-
writing and provenance chains.
Human Evaluation
Beyond automatic metrics, three researchers with NLP
backgrounds independently conduct blinded evaluations on
the same 200 CrossMeet-EN questions: for each question,
EvoWiki’s answer is paired separately with anonymized an-
swersfromDirectLLM,VanillaRAG,GraphRAG,andLog-
icRAG. Unaware of method identity, they judge EvoWiki as
a win, tie, or loss on faithfulness, completeness, and con-
ciseness. Final labels follow majority vote, with three-way
disagreementsresolvedthroughdiscussion.Figure5reports
results from EvoWiki’s perspective. Against the four base-
lines, EvoWiki’s faithfulness win rates are 31.0%–51.5%,
above loss rates of 21.0%–26.5%. Against the three RAG
methods, completeness win rates are 51.5%–62.5%, above
lossratesof16.0%–23.0%,whileEvoWikitiesDirectLLM.
For conciseness, its win rates against Direct LLM, Vanil-Method CM-ZH CM-EN LongMemEval
Direct LLM1.0%0.3% 70.6%
VanillaRAG (BM25) 3.6% 8.6% 54.9%
VanillaRAG (Dense) 2.5% 4.1% 55.7%
VanillaRAG (Hybrid) 2.3% 3.3% 54.5%
RAPTOR 3.7% 10.7% 52.6%
GraphRAG 1.3% 1.8% 51.7%
HippoRAG2 3.1% 11.7% 54.5%
LightRAG 2.1% 4.8% 46.8%
LogicRAG 10.1% 17.8% 34.5%
EvoWiki 1.1%0.2% 22.8%
Table10:AbstentionratesonCrossMeetandLongMemEval.
laRAG,andGraphRAGare48.5%,43.0%,and42.0%,above
loss rates of 27.5%, 34.5%, and 32.0%; against LogicRAG,
the 28.5% win rate is slightly below the 32.0% loss rate.
Humanevaluationshowsthattheadvantagearisesfromfac-
tual faithfulness and information completeness rather than
generation verbosity.
Abstention
Table 10 compares abstention rates. Every CrossMeet ques-
tion is answerable, so abstention is erroneous. EvoWiki’s
rates are 0.2% on CrossMeet-EN and 1.1% on CrossMeet-
ZH, indicating that Wiki-onlyReadusually provides suf-
ficient valid states. Of LongMemEval’s 500 instances, 30
are unanswerable. To make abstention reflect failure to re-
cover a valid state, we compute it only on the remaining
470answerableinstances,whereanyabstentioniserroneous.
EvoWiki’s rate is 22.8%, below LogicRAG’s 34.5%, Vanil-
laRAG’s 54.5%, and Direct LLM’s 70.6%.
Conclusion
WepresentedEvoWikifordynamiccross-meetingQA,com-
bining incrementalBuildand Wiki-onlyReadwith entity
versionchains,write-timecoreferenceresolution,stateover-
writing, and meeting-level provenance to maintain a cur-
rentviewandtraceablehistory.Wealsointroducedbilingual
CrossMeet for factual consistency, temporal reasoning, and

multi-hopQA.Acrosssixdatasetsandtworeaders,EvoWiki
achievesmacro-averageJudgeAccuracyof60.09and63.02,
surpassing the strongest baselines by 9.72 and 10.00 points;
state-flip,evidence-position,andhumananalysesconfirmro-
bust current-state reading and traceability. Wiki-onlyRead
remains constrained byBuildextraction completeness, es-
pecially under ASR noise and informal speech. Future work
willexploreuncertainty-awarewriting,selectivesourceveri-
fication,andbroaderlanguages,domains,andtimehorizons.
References
Anthropic. 2025. Claude Opus 4.5 System Card. System
card, Anthropic. https://www.anthropic.com/system-cards.
Anthropic. 2026. Claude Opus 4.7 System Card. System
card, Anthropic. https://www.anthropic.com/system-cards.
Asai,A.;Wu,Z.;Wang,Y.;Sil,A.;andHajishirzi,H.2024.
Self-rag:Learningtoretrieve,generate,andcritiquethrough
self-reflection. InInternational conference on learning rep-
resentations, volume 2024, 9112–9141.
Bai,Y.;Tu,S.;Zhang,J.;Peng,H.;Wang,X.;Lv,X.;Cao,S.;
Xu,J.;Hou,L.;Dong,Y.;etal.2025.Longbenchv2:Towards
deeperunderstandingandreasoningonrealisticlong-context
multitasks. InProceedingsofthe63rdAnnualMeetingofthe
Association for Computational Linguistics (Volume 1: Long
Papers), 3639–3664.
Chen, B.; Guo, Z.; Yang, Z.; Chen, Y.; Chen, J.; Liu, Z.;
Shi, C.; and Yang, C. 2026a. Pathrag: Pruning graph-based
retrievalaugmentedgenerationwithrelationalpaths. InPro-
ceedings of the AAAI conference on artificial intelligence,
volume 40, 30183–30191.
Chen, S.; Zhou, C.; Yuan, Z.; Zhang, Q.; Cui, Z.; Chen, H.;
Xiao,Y.;Cao,J.;andHuang,X.2026b. Youdon’tneedpre-
built graphs for rag: Retrieval augmented generation with
adaptive reasoning structures. InProceedings of the AAAI
Conference on Artificial Intelligence, volume 40, 30270–
30278.
Cheng,Y.;Yu,Y.-C.;Chang,K.-P.;andWang,Y.-C.F.2025.
Serial lifelong editing via mixture of knowledge experts. In
Proceedings of the 63rd Annual Meeting of the Associa-
tionforComputationalLinguistics(Volume1:LongPapers),
30888–30903.
Edge,D.;Trinh,H.;Cheng,N.;Bradley,J.;Chao,A.;Mody,
A.;Truitt, S.;Metropolitansky, D.;Ness,R. O.;andLarson,
J.2024.Fromlocaltoglobal:Agraphragapproachtoquery-
focused summarization.arXiv preprint arXiv:2404.16130.
Fang, J.; Jiang, H.; Wang, K.; Ma, Y.; Shi, J.; Wang, X.;
He, X.; and Chua, T.-S. 2025. Alphaedit: Null-space con-
strained knowledge editing for language models. InInter-
national Conference on Learning Representations, volume
2025, 16366–16396.
Google DeepMind. 2026. Gemini 3.1 Pro Model Card.
Model card, Google DeepMind. https://deepmind.google/
models/model-cards/gemini-3-1-pro/.
Guo, Z.; Xia, L.; Yu, Y.; Ao, T.; and Huang, C. 2024. Ligh-
trag: Simple and fast retrieval-augmented generation.arXiv
preprint arXiv:2410.05779, 2(3).Gutiérrez, B. J.; Shu, Y.; Qi, W.; Zhou, S.; and Su, Y. 2025.
Fromragtomemory:Non-parametriccontinuallearningfor
large language models.arXiv preprint arXiv:2502.14802.
Hou, Y.; Tamoto, H.; Zhao, Q.; and Miyashita, H. 2025.
SynapticRAG: Enhancing Temporal Memory Retrieval in
Large Language Models through Synaptic Mechanisms. In
Findings of the Association for Computational Linguistics:
ACL 2025, 20422–20436.
Hsieh, C.-P.; Sun, S.; Kriman, S.; Acharya, S.; Rekesh, D.;
Jia, F.; Zhang, Y.; and Ginsburg, B. 2024. RULER: What’s
the real context size of your long-context language models?
arXiv preprint arXiv:2404.06654.
Huerta,J.M.2026. WiCER:Wiki-memoryCompile,Evalu-
ate, Refine Iterative Knowledge Compilation for LLM Wiki
Systems.arXiv preprint arXiv:2605.07068.
Jeong, S.; Baek, J.; Cho, S.; Hwang, S. J.; and Park, J. C.
2024. Adaptive-rag: Learning to adapt retrieval-augmented
largelanguagemodelsthroughquestioncomplexity. InPro-
ceedings of the 2024 Conference of the North American
Chapter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Papers),
7036–7050.
Jiang,D.;Li,Y.;Li,G.;andLi,B.2026. MAGMA:AMulti-
Graph based Agentic Memory Architecture for AI Agents.
arXiv preprint arXiv:2601.03236.
Karpukhin,V.;Oguz,B.;Min,S.;Lewis,P.;Wu,L.;Edunov,
S.; Chen, D.; and Yih, W.-t. 2020. Dense passage retrieval
for open-domain question answering. InProceedings of the
2020 conference on empirical methods in natural language
processing (EMNLP), 6769–6781.
Kirstein, F.; Khan, M.; Wahle, J. P.; Ruas, T.; and Gipp, B.
2025. You need to MIMIC to get FAME: Solving Meet-
ing Transcript Scarcity with Multi-Agent Conversations. In
Findings of the Association for Computational Linguistics:
ACL 2025, 11482–11525.
Krishna, S.; Krishna, K.; Mohananey, A.; Schwarcz,
S.; Stambler, A.; Upadhyay, S.; and Faruqui, M. 2025.
Fact, fetch, and reason: A unified evaluation of retrieval-
augmented generation. InProceedings of the 2025 Con-
ference of the Nations of the Americas Chapter of the As-
sociation for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers), 4745–4759.
Latimer, C.; Boschi, N.; Neeser, A.; Bartholomew, C.; Sri-
vastava, G.; Wang, X.; and Ramakrishnan, N. 2026. HIND-
SIGHT:StructuredAgentMemorythatRetains,Recalls,and
Reflects. InProceedings of the 64th Annual Meeting of the
Association for Computational Linguistics (Volume 3: Sys-
tem Demonstrations), 275–285.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.;
Goyal,N.;Küttler,H.;Lewis,M.;Yih,W.-t.;Rocktäschel,T.;
et al. 2020. Retrieval-augmented generation for knowledge-
intensivenlptasks.Advancesinneuralinformationprocess-
ing systems, 33: 9459–9474.
Li,C.;Song,D.;Zhou, C.;Yang,J.;Tian,Y.;Ma,H.;Feng,
G.;Zhang,L.;Li,X.;andDuan,K.2026. StaticModels,Dy-
namicWorld:AUnifiedPerspectiveonTemporalPerception

in Large Language Models. InFindings of the Association
for Computational Linguistics: ACL 2026, 1913–1932.
Li,Z.;Chen,X.;Yu,H.;Lin,H.;Lu,Y.;Tang,Q.;Huang,F.;
Han,X.;Sun,L.;andLi,Y.2025.Structrag:Boostingknowl-
edge intensive reasoning of llms via inference-time hybrid
information structurization. InInternational Conference on
Learning Representations, volume 2025, 36107–36124.
Li,Z.;Li,C.;Zhang,M.;Mei,Q.;andBendersky,M.2024.
Retrievalaugmentedgenerationorlong-contextllms?acom-
prehensivestudyandhybridapproach. InProceedingsofthe
2024ConferenceonEmpiricalMethodsinNaturalLanguage
Processing: Industry Track, 881–893.
Liska, A.; Kocisky, T.; Gribovskaya, E.; Terzi, T.; Sezener,
E.; Agrawal, D.; D’Autume, C. D. M.; Scholtes, T.; Zaheer,
M.; Young, S.; et al. 2022. Streamingqa: A benchmark for
adaptationtonewknowledgeovertimeinquestionanswering
models. InInternational Conference on Machine Learning,
13604–13622. PMLR.
Liu, N. F.; Lin, K.; Hewitt, J.; Paranjape, A.; Bevilacqua,
M.; Petroni, F.; and Liang, P. 2024. Lost in the middle:
Howlanguagemodelsuselongcontexts.Transactionsofthe
association for computational linguistics, 12: 157–173.
Meng, K.; Bau, D.; Andonian, A.; and Belinkov, Y. 2022.
Locatingandeditingfactualassociationsingpt.Advancesin
neural information processing systems, 35: 17359–17372.
Modarressi, A.; Deilamsalehy, H.; Dernoncourt, F.; Bui, T.;
Rossi,R.A.;Yoon,S.;andSchütze,H.2025. Nolima:Long-
context evaluation beyond literal matching.arXiv preprint
arXiv:2502.05167.
Prasad,A.;Bui,T.;Yoon,S.;Deilamsalehy,H.;Dernoncourt,
F.; and Bansal, M. 2023. MeetingQA: Extractive question-
answering on meeting transcripts. InProceedings of the
61st Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), 15000–15025.
Qwen Team. 2026. Qwen3.5: Towards Native Multimodal
Agents. Qwen Blog. https://qwen.ai/blog?id=qwen3.5.
Robertson, S.; and Zaragoza, H. 2009.The probabilistic
relevance framework: BM25 and beyond, volume 4. Now
Publishers Inc.
Sarthi,P.;Abdullah,S.;Tuli,A.;Khanna,S.;Goldie,A.;and
Manning,C.2024. Raptor:Recursiveabstractiveprocessing
for tree-organized retrieval. InInternational Conference on
Learning Representations, volume 2024, 32628–32649.
Schumacher,D.;Haji,F.;Grey,T.;Bandlamudi,N.;Karnik,
N.;Kumar,G.U.;Chiang,C.-Y.J.;Najafirad,P.;Vishwami-
tra, N.; and Rios, A. 2025. RASTeR: Robust, Agentic, and
Structured Temporal Reasoning. InProceedings of the 14th
International Joint Conference on Natural Language Pro-
cessingandthe4thConferenceoftheAsia-PacificChapterof
the Association for Computational Linguistics, 3098–3123.
Su,M.;Guo,Y.;Hou,Z.;Bai,L.;Li,Z.;Zhang,Y.;Yin,G.;
Lin,W.;Jin,X.;Guo,J.;etal.2026. BeyondDialogueTime:
Temporal Semantic Memory for Personalized LLM Agents.
arXiv preprint arXiv:2601.07468.
Thonet, T.; Besacier, L.; and Rozen, J. 2025. Elitr-bench:
A meeting assistant benchmark for long-context languagemodels. InProceedingsofthe31stInternationalConference
on Computational Linguistics, 407–428.
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A. 2022. MuSiQue: Multihop Questions via Single-hop
Question Composition.Transactions of the Association for
Computational Linguistics, 10: 539–554.
Vu, T.; Iyyer, M.; Wang, X.; Constant, N.; Wei, J.; Wei, J.;
Tar,C.;Sung,Y.-H.;Zhou,D.;Le,Q.;etal.2024. Freshllms:
Refreshing large language models with search engine aug-
mentation. InFindingsoftheAssociationforComputational
Linguistics: ACL 2024, 13697–13720.
Wang, F.; Wan, X.; Sun, R.; Chen, J.; and Arik, S. O. 2025.
AstuteRAG:OvercomingImperfectRetrievalAugmentation
and Knowledge Conflicts for Large Language Models. In
Proceedings of the 63rd Annual Meeting of the Association
for Computational Linguistics, 30553–30571.
Wang, P.; Li, Z.; Zhang, N.; Xu, Z.; Yao, Y.; Jiang, Y.; Xie,
P.; Huang, F.; and Chen, H. 2024. Wise: Rethinking the
knowledge memory for lifelong model editing of large lan-
guage models.Advances in Neural Information Processing
Systems, 37: 53764–53797.
Wu, D.; Wang, H.; Yu, W.; Zhang, Y.; Chang, K.-W.;
and Yu, D. 2024. Longmemeval: Benchmarking chat as-
sistants on long-term interactive memory.arXiv preprint
arXiv:2410.10813.
Xu, A.; Lin, B.; Xue, B.; Wang, B.; Xu, B.; Wu, B.; Zhang,
B.; Lin, C.; Dong, C.; Ling, C.; et al. 2026. Deepseek-v4:
Towards highly efficient million-token context intelligence.
arXiv preprint arXiv:2606.19348.
Yan,S.-Q.;Gu,J.-C.;Zhu,Y.;andLing,Z.-H.2024. Correc-
tive Retrieval Augmented Generation. arXiv:2401.15884.
Yen, H.; Gao, T.; Hou, M.; Ding, K.; Fleischer, D.; Izsak,
P.; Wasserblat, M.; and Chen, D. 2025. HELMET: How to
evaluate long-context models effectively and thoroughly. In
The Thirteenth International Conference on Learning Rep-
resentations.
Zhang, S.; Xue, Y.; Zhang, Y.; Wu, X.; Luu, A. T.;
and Zhao, C. 2024. MRAG: A modular retrieval frame-
work for time-sensitive question answering.Preprint at
https://arxiv.org/abs/2412.15540.
Zhang, Y.; Li, M.; Long, D.; Zhang, X.; Lin, H.; Yang, B.;
Xie, P.; Yang, A.; Liu, D.; Lin, J.; et al. 2025. Qwen3 em-
bedding: Advancing text embedding and reranking through
foundation models.arXiv preprint arXiv:2506.05176.
Zheng, L.; Chiang, W.-L.; Sheng, Y.; Zhuang, S.; Wu, Z.;
Zhuang, Y.; Lin, Z.; Li, Z.; Li, D.; Xing, E.; et al. 2023.
Judgingllm-as-a-judgewithmt-benchandchatbotarena.Ad-
vancesinneuralinformationprocessingsystems,36:46595–
46623.
Zhong, Z.; Wu, Z.; Manning, C. D.; Potts, C.; and Chen,
D.2023. Mquake:Assessingknowledgeeditinginlanguage
modelsviamulti-hopquestions. InProceedingsofthe2023
ConferenceonEmpiricalMethodsinNaturalLanguagePro-
cessing, 15686–15702.
Zhu,J.;Li,J.;Wen,Y.;Li,X.;Guo,L.;andChen,F.2025a.
MFinMeeting:AMultilingual,Multi-Sector,andMulti-Task

Financial Meeting Understanding Evaluation Dataset. In
Findings of the Association for Computational Linguistics:
ACL 2025, 244–266.
Zhu, X.; Xie, Y.; Liu, Y.; Li, Y.; and Hu, W. 2025b. Knowl-
edge graph-guided retrieval augmented generation. InPro-
ceedingsofthe2025ConferenceoftheNationsoftheAmer-
icas Chapter of the Association for Computational Linguis-
tics: Human Language Technologies (Volume 1: Long Pa-
pers), 8912–8924.
# CMT-RAG: Complementary Memory Traces for Multi-turn Multi-hop RAG

**Authors**: Lang Zhou, Yingjian Chen, Shuxuan Li, Kun-Yu Lin, Zhilin Zhao

**Published**: 2026-07-29 04:50:41

**PDF URL**: [https://arxiv.org/pdf/2607.26470v1](https://arxiv.org/pdf/2607.26470v1)

## Abstract
Multi-turn information-seeking conversations require both multi-hop reasoning and long-range dependency tracking across turns. However, existing RAG systems typically represent conversational memory as raw dialogue history, rewritten queries, or unstructured summaries, making it difficult to recover the specific prior reasoning steps and evidence required for follow-up queries. Our key insight is to align conversational memory with retrieval by representing dialogue context as sub-question-level reasoning traces. Building on this insight, we introduce MuMu-QA, a benchmark for multi-turn multi-hop RAG with explicit cross-turn sub-question dependency annotations, and CMT-RAG, a complementary memory framework for this setting. At each turn, CMT-RAG employs a state-space trace generator, whose recurrent state serves as runtime memory, to incorporate recent conversational context and decompose the current query into structured trace drafts containing retrieval-oriented sub-questions and dependencies on earlier traces. It then grounds these drafts with retrieved evidence and stores them as persistent memory traces in a session-level DAG, enabling future turns to efficiently recover relevant prior reasoning and evidence. Experiments on MuMu-QA and corpus-level RAG benchmarks show that CMT-RAG consistently outperforms five categories of RAG baselines in answer accuracy.

## Full Text


<!-- PDF content starts -->

CMT-RAG: Complementary Memory Traces for
Multi-turn Multi-hop RAG
Lang Zhou1,2, Yingjian Chen1,2, Shuxuan Li2, Kun-Yu Lin3, Zhilin Zhao1,2
1Sun Yat-sen University
2Shenzhen Loop Area Institude
3The University of Hong Kong
Abstract
Multi-turn information-seeking conversations require both
multi-hop reasoning and long-range dependency tracking
acrossturns.However,existingRAGsystemstypicallyrepre-
sentconversationalmemoryasrawdialoguehistory,rewritten
queries, or unstructured summaries, making it difficult to re-
coverthespecificpriorreasoningstepsandevidencerequired
for follow-up queries. Our key insight is to align conversa-
tional memory with retrieval by representing dialogue con-
text as sub-question-level reasoning traces. Building on this
insight, we introduceMuMu-QA, a benchmark for multi-
turn multi-hop RAG with explicit cross-turn sub-question
dependency annotations, andCMT-RAG, a complementary
memory framework for this setting. At each turn, CMT-RAG
employs a state-space trace generator, whose recurrent state
serves as runtime memory, to incorporate recent conversa-
tional context and decompose the current query into struc-
tured trace drafts containing retrieval-oriented sub-questions
anddependenciesonearliertraces.Itthengroundsthesedrafts
with retrieved evidence and stores them as persistent mem-
ory traces in a session-level DAG, enabling future turns to
efficiently recover relevant prior reasoning and evidence. Ex-
periments on MuMu-QA and corpus-level RAG benchmarks
showthatCMT-RAGconsistentlyoutperformsfivecategories
of RAG baselines in answer accuracy.
1 Introduction
Retrieval-augmented generation (RAG) is increasingly de-
ployed in extended information-seeking conversations,
whereusersrefinequestions,omitrepeatedentities,andbuild
new requests on earlier answers (Ye et al. 2026; Laban et al.
2026; Hu, Wang, and McAuley 2026). In such settings, a
new turn often remains context-dependent while requiring
multi-hop evidence seeking, since answering it may require
decomposingthequeryintoseveralsub-questionswhosede-
pendenciesspanearlierturns.Figure1showsarepresentative
case. The system must retrieve evidence for the current turn
and identify the specific prior reasoning step whose subject
orentityisbeingextendedorreused.Werefertothissetting
asmulti-turn multi-hop conversational RAG with comple-
mentary sub-question dependencies.
This setting exposes a mismatch between how conversa-
tional RAG stores memory and how retrieval actually oper-
ates.Queryrewritingconvertsacontext-dependentturninto
a standalone query (Anantha et al. 2021; Mo et al. 2023;
Steve Jobs briefly attended Reed College.Steve Jobs, who was born in San Francisco.SubA1 SubA2
SubA3SubA1
SubA2
SubA3
Dependency 
GraphWho founded Apple, and where was he born?SubQ1 SubQ2
What universities did he attend?SubQ3
Did the company ever partner with that college?SubQ4SubQ1
SubQ2
SubQ4SubQ3Ext.
Ref.Ext.
Ext.
Multi-turn Multi-hop DialogueFigure 1: A multi-turn multi-hop conversation with cross-
turn dependencies. The graph illustrates two dependency
types:Ext.(Predicate Extension) queries a new attribute
or relation of a previously resolved target, andRef.(Entity
Reference) directly reuses a previously introduced entity.
Zhuetal.2025a),whichiseffectiveforlocalcoreferencebut
compresses dependency chains into a single query, obscur-
ing intermediate retrieval targets. Query decomposition ex-
poses sub-question structure for multi-hop retrieval (Trivedi
et al. 2023; Khot et al. 2023; Chen et al. 2026; Ye et al.
2025), yet typically assumes a self-contained query with
dependencies confined to the current turn. Memory-based
conversational systems store histories, summaries, or em-
beddings (Liu et al. 2024b; Zhong et al. 2024), while dia-
logue graphs model utterance-level relations (Li et al. 2020;
Fan et al. 2023; Zhu et al. 2025c). None explicitly repre-
sents retrieval-level dependencies across turns. As a result,
retrieversrequiresub-question-levelmemory,whereasexist-
ing systems largely maintain only turn-level context.
Ourkeyinsightistoalignconversationalmemorywithre-
trievalbystoringdialoguecontextassub-question-levelrea-
soning traces. A useful memory unit for this setting should
preserve the retrieval target, expose the dependency that re-
solvesmissingarguments,andretaintheevidencethatmade
the earlier answer valid. A sub-question-level trace provides
this unit by packaging a past reasoning step as an address-
able object. When a later turn depends on it, the system can
recover the relevant trace through its dependency links and
keywords, then reuse the associated evidence under the cur-
rentquery.Cross-turnrecallisthereforereducedfromglobal
arXiv:2607.26470v1  [cs.CL]  29 Jul 2026

history interpretation to trace selection and evidence reuse.
Accordingly, we proposeCMT-RAG, a framework built
aroundcomplementarymemorytraces.Ateachturn,astate-
space trace generator consumes the current query and its
recurrent state to produce structuredtrace drafts, each con-
tainingasub-question,tracekeywords,anddependencieson
earlier traces. After answer-reference resolution, fresh ev-
idence is retrieved for each sub-question, and the draft is
completed by attaching the corresponding paragraph iden-
tifiers. Dependency links access prerequisite DAG nodes,
while trace keywords retrieve an additional historical trace
through lexical matching. The reader transiently combines
the accessed historical evidence with the freshly retrieved
evidence. After answering, the completed trace and its sub-
answer are appended to the DAG. The recurrent state thus
maintains local discourse continuity, while the trace DAG
preserves explicit long-range dependencies and reusable ev-
idence. The downstream reader remains stateless, receiving
only the resolved sub-question and its assembled evidence.
To study this problem directly, we introduceMuMu-QA,
abenchmarkthatreorganizesmulti-hopquestionsintomulti-
turn dialogues with cross-turn dependency annotations. Ex-
isting multi-turn RAG benchmarks evaluate conversational
retrieval and generation at the turn level, without exposing
which current sub-question depends on which prior sub-
question. MuMu-QA fills this gap by providing dialogue-
wide sub-question identifiers, trace keywords, dependency
edges, supporting paragraph IDs, and full trace DAG su-
pervision,withlong-dialoguesplitsforstress-testingdepen-
dencyrecoverybeyondshorthistoryreplay.Experimentson
MuMu-QA show that CMT-RAG achieves the best answer
accuracy among direct C-RAG, query rewriting, agentic re-
trieval, decomposition-based RAG, and dialogue-structure
baselines. With a stateless Qwen3-32B reader, it reaches
41.73 EM and 55.63 F1 using top-5 retrieval while keep-
ing cross-turn memory outside the reader.
2 Preliminaries
This section fixes the notation and evaluation target used
throughout the paper. We first define multi-turn multi-hop
conversational RAG as trace-DAG induction, where each
trace is a retrieval-level memory unit that binds a sub-
question to dependencies, keywords, and evidence. We then
describe MuMu-QA as the benchmark instantiation of this
formulation, with supervision over sub-question dependen-
cies and reusable paragraph evidence.
2.1 Task Formalization
Weconsideramulti-turnmulti-hopconversationalRAGses-
sion over an unstructured corpusC. The dialogue is an or-
dered sequence ofTuser turns,
D=⟨q 1, q2, . . . , q T⟩,(1)
whereq tdenotes the user query at turnt. For each turn,
thesystemmustproduceananswera tgroundedinevidence
fromC. The distinctive difficulty is that a turn may contain
multipleretrieval-relevantsub-questions,eachofwhichmay
depend on information from earlier turns. We formalize this
SubQ1
SubQ2Relocation
Graph Splice Q1SubQ1
SubQ2
SubQ3Q1’In-turn
Dep.
Cross-
-turn 
Dep.
SubQ1/3 Q1’ Q2’
Cross-turn Dep.If SubA1 Equals to SubA3
Cross-turn Dep.
SubQ2
SubQ4Original Queries
SubQ3
SubQ4Q2In-turn
Dep.
Q2’ SubQ4Figure 2: MuMu-QA synthesis operators. Parent questions
are decomposed into sub-questions with in-turn dependen-
cies.Sub-question Relocationmoves a sub-question to a
later turn to create cross-turn dependencies, whileGraph
Splicingjoins two reasoning chains via a bridge answer.
settingassession-levelinductionofadirectedacyclicgraph
oftraces,
G= (V,E G),(2)
where each node(T k, ak)∈ Vconsists of a trace and its
answer. The traceT kbinds a sub-question, trace keywords,
dependency edges, and retrieved paragraph identifiers as
Tk= 
qsub
k, kw k,deps(k),para_idsk
,(3)
whereqsub
kisthenatural-languagesub-questionconsumedby
thereaderandretriever,kw karelookupanchorsforthetrace
DAG,deps(k)⊆ {1, . . . , k−1}listsprerequisitetraces,and
para_idskidentifies the paragraphs directly retrieved from
Cfor this sub-question. Each edge(T j,Ti)∈ E Gstates that
traceT ireliesontheentityorsubjectintroducedbytraceT j.
Atturnt,thetracegeneratormapsthecurrentqueryq tand
recurrent stateh t−1to an ordered set of trace drafts. CMT-
RAGthenresolvestheirpredicteddependenciesagainstG <t
and appends the completed traces as∆G t. This formulation
couples two structured operations. The system must decom-
pose the current turn into retrieval units and link those units
to prior traces whose subjects or entities remain necessary.
The target memory unit is not a whole utterance or an un-
structured history summary. It is a trace whose fields are
directlyconsumedbyretrieval,DAGlookup,andanswering.
2.2 Benchmark Construction
MuMu-QA instantiates this formulation as a benchmark
for multi-turn multi-hop RAG. Existing multi-turn C-
RAG benchmarks supervise standalone-query rewriting or
turn-level answers (Ali et al. 2026; Cheng et al. 2025;
Katsis et al. 2025), without annotating dependencies at
sub-question granularity. We constructMuMu-QAfrom
MuSiQue(Trivedietal.2022),usingitssub-questiondecom-
positions, intermediate answers, and supporting paragraphs
to derive supervision for trace generation.
As illustrated in Figure 2, dialogues are synthesized us-
ing two operators.Sub-question Relocationmoves a sub-
question from a multi-hop question into a separate turn and
rewrites the remaining question as a follow-up that depends

Paragraphs ParagraphsRuntime Memory Persistent Memory
Answering ModelSubQ5SSM Decomposer Trace DAGWho founded Apple, and where was he born?
Steve Jobs, who was born in San Francisco.
Did he ever speak at that college, and was it 
before or after the donation?Multi-turn Multi-hop Dialogue
Answer:                        ...                        ...SubQ6
Did Steve Jobs ever 
speak at Reed College?Is year 2005 before or 
after the donation?
Sub-questionParas5CurQ.
SubA6P221 P12,45Paras6Steve Jobs speak 
at Reed College
Trace3year 2005 the 
donation
Trace3,5 
P108 P12Paras3 Paras3,5Trace 
DraftSubQ.
SubA.Local PIDs.Trace Draft New Trace 
Collection & 
PersistenceSubQ5 SubQ6
Paras5,3 Paras6,3,5Dep5 Dep6
SubA5KW6 KW5
Paras.
Unstructured KnowledgeP12 P45 P108 P221 P305Query/ID RetrieverPIDsTrace1
Trace2
Trace3
Trace4ParaIDs.
KWs.SubA.SubQ.�
��
�� 3 34
3
4Figure 3: Overview of CMT-RAG. The framework consists of four stages: (i) Trace generation, where a state-space model
(SSM) maintains a recurrent state to generate structured trace drafts; (ii) Reference resolution, where cross-turn dependencies
are resolved through the trace DAG; (iii) Evidence retrieval and trace update, where supporting paragraphs are retrieved and
completed traces are written back to the DAG; and (iv) Question answering, where the stateless reader answers the resolved
sub-questions using only the retrieved evidence, without replaying the dialogue history, before producing the final response.
on the relocated answer.Graph Splicinglinks two reason-
ing chains through a shared bridge answer: a seed turn first
resolves the bridge entity, and a later turn continues reason-
ing from that entity with explicit dependencies on earlier
trace nodes. Together, these operators generate short dia-
logues with controlled cross-turn dependencies. We further
construct long dialogues by interleaving topic-related ses-
sions, producing conversations of up to several dozen turns
fortrainingandevaluation.Fullconstructiondetailsarepro-
vided in Appendix A.
3 Method
CMT-RAGinstantiatesthetrace-DAGformulationwithtwo
complementarymemorychannels:astate-spacetracegener-
atorthatcaptureslocaldiscoursetoproducestructuredtrace
drafts, and a session-level trace DAG that persistently stores
traces for dependency-aware retrieval and evidence reuse.
As shown in Figure 3, this design externalizes conversa-
tional state from the answering model, which remains state-
lesswhiletherecurrentstateandtraceDAGjointlymaintain
local and long-range conversational memory.
3.1 State-Space Trace Generation
We instantiate the trace generator with a state space model
(SSM) backbone based on Mamba-2 (Dao and Gu 2024;
Gu and Dao 2024) while preserving its selective state-space
mixer, whose recurrent state serves as a compact carrier of
local discourse context. This enables the model to avoid
repeatedly encoding the full dialogue history at each turn,
thereby reducing exposure to lost-in-the-middle effects (Yuet al. 2025; Liu et al. 2024a). We further adapt the back-
bonethroughLow-RankAdaptation(LoRA)fine-tuningand
DirectPreferenceOptimization(DPO),togetherwithastruc-
turedoutputvocabulary,sothatitgeneratestracedraftsrather
than free-form plans.
At turnt, the generator receives the current queryq t, the
previous hidden stateh t−1, and emits a set of draft traces
and an updated state,
{Tdraft
k}k∈t, ht=TraceGen θ(ht−1, qt),(4)
the updated stateh tcarrying local continuity such as topic
focus, intent shifts and surface coreference, is passed to the
next turn.
Structured trace drafts.Each draft has the form
Tdraft
k = 
qdecom
k , kw k,deps(k)
,(5)
whereqdecom
kis the decomposed sub-question,kw kcon-
tains trace keywords for DAG lookup, anddeps(k)lists
prerequisite trace identifiers. We separate keywords from
sub-questionsintotwofieldsservingdifferentpurposes.The
sub-question is optimized as natural-language input for the
readeranddenseretrievalafterreferenceresolution,whereas
thekeywordfieldisoptimizedforefficienttrace-DAGlookup
via lightweight lexical matching.
Global trace namespace.CMT-RAG maintains an
append-only namespace shared across the dialogue. Each
completed trace is assigned a persistent trace identifier,
[T_1], . . . ,[T_K], and a corresponding answer reference
token[A_1], . . . ,[A_K]. The generator can therefore ex-
plicitly reference prior traces throughdeps(k).

3.2 Trace DAG as Persistent Memory
The trace DAG maintains a durable session-level memory.
Foreachdraft,CMT-RAGfirstresolvesanswer-referenceto-
kens inqdecom
kusing the answer attached to each referenced
DAGnode,andproducestheresolvedsub-questionqsub
k.The
systemthenperformstwoevidenceretrievaloperations.De-
pendency edges directly recover evidence from prerequisite
traces, while trace keywords retrieve stored paragraph iden-
tifiers from relevant historical traces. Together, they form
Pprior
k. Since the keywords are generated as normalized re-
trievalanchors,lightweightlexicallookupsufficestoidentify
relevant traces without maintaining a separate dense index.
Fresh retrieval is then issued fromqsub
kagainst the exter-
nalcorpusCtoobtaintheparagraphidentifiersetPlocal
k.The
assembled paragraph setP kis used for the latter inference
stage. The completed trace stores the resolved sub-question,
keywords, dependency links, and the newly retrieved para-
graph setPlocal
k,
Tk= 
qsub
k, kw k,deps(k),Plocal
k
,
Pk=Plocal
k∪ Pprior
k.(6)
After the reader produces answera k, the completed DAG
node(T k, ak)is appended toGin topological order. The
traceandansweraresubsequentlyaccessedthroughdifferent
mechanisms:answerreferencesresolvemissingargumentsin
future sub-questions, whereas trace keywords retrieve rele-
vant traces together with their supporting evidence.
3.3 Curriculum Learning and DPO Training
The trace generator is trained with curriculum-based super-
vised fine-tuning under progressively longer conversational
contexts, followed by DPO to align generated traces with
downstream retrieval and question answering.
Three-stage curriculum.Supervised training follows a
three-stage curriculum. Stage 1 trains on single-turn ex-
amples to learn the trace syntax and basic decomposition
structure.Stage2introducesshortmulti-turndialogues,and
Stage3furtherextendstrainingtolongerdialogues.Through-
out all stages, the primary objective is the autoregressive
language modeling lossL lmover the linearized gold trace
sequence. For a training instancex 1:T,
Llm=−TX
t=1logp θ(xt|x<t).(7)
Later stages progressively increase dialogue length and
cross-turn dependency density, while replaying earlier-stage
examples to preserve the basic trace representation.
DPO training.TheDPOstagesamplesmultiplecandidate
traces per turn from the long-dialogue model and executes
them through the fixed retrieval–reader pipeline. Candidate
tracesarerankedusingacompositerewardcombiningfinal-
answer F1 (F final) and matched sub-question F1 (F sub):
R(τ;c) =F final(τ) +γF sub(τ),(8)
where,cdenotes the current dialogue context for the policy.
Ffinaliscomputedbetweenthereader’sfinalanswerinducedAlgorithm 1: CMT-RAG inference at turnt.
Require:Stateh t−1,traceDAGG <t,queryq t,retrieverR,
readerM
1:{Tdraft
k}t, ht←Gen SSM(ht−1, qt)
2:G t← G <t
3:foreachTdraft
k = (qdecom
k , kwk,depsk)in topological
orderdo
4:qsub
k←RefRes(qdecom
k ,depsk,Gt)
5:Tprior
k←Index(G t, kwk)∪depsk
6:pidsprior
k←GetPIDs(Tprior
k)
7:pidslocal
k← R(qsub
k)
8:parask←Get(pidsprior
k∪pidslocal
k)
9:a k← M(qsub
k,parask)
10:T k←(qsub
k, kwk,depsk,pidslocal
k)
11:G t← G t∪ {(T k, ak)}
12:end for
13:a t←Aggregate(q t,{qsub
k}t,{ak}t)
14:
15:returna t, ht,Gt
by candidate traceτand the ground truth, whileF subaver-
agesF1overlexicallymatchedgeneratedandreferencesub-
questions. Invalid traces are filtered after pair construction
rather than rewarded explicitly. Preference pairs are formed
fromsufficientlyseparatedhigh-andlow-rewardtraces,and
DPO trains the trace generator to assign higher probabil-
ity to preferred traces than rejected ones under the frozen
long-dialogue SFT reference policy.
3.4 Inference with CMT-RAG
Algorithm 1 summarizes inference for a single user turn.
Conversational state is fully externalized into the recurrent
state and trace DAG, allowing the reader to remain stateless
andanswereachresolvedsub-questionusingonlyitsassem-
bled evidence. An additional reader call then aggregates the
user query together with the current-turn sub-questions and
sub-answersintothefinalanswera t,whiletheupdatedDAG
is carried forward to subsequent turns.
4 Experiments
We evaluate whether complementary sub-question-level
memorytracesimprovemulti-turnmulti-hopconversational
RAG. The experiments address five questions.RQ1Does
CMT-RAG improve answer accuracy across different state-
less readers and five categories of baseline methods?RQ2
What gains come from the trace-management framework
andthetrainedtracegenerator?RQ3HowdorecurrentSSM
stateandpersistentDAGmemorycontributeacrossdialogue
lengths?RQ4Howdoesthetrace-generatorbackboneaffect
answer quality and latency?RQ5Does CMT-RAG transfer
to additional shared-corpus RAG benchmarks?
4.1 Experimental Setup
Dataset.As detailed in Appendix A.3, MuMu-QA com-
prisesthreedialogue-lengthregimes:short(3–7turns),long

Reader Method Top-k⋆Avg. Paras. EM↑(%) F1↑(%) GoldCtx↑(%)
Qwen3-32BDirect C-RAG 20 20 35.66 48.81 88.42
Direct C-RAG (With Thinking Mode) 20 20 38.23 50.20 88.42
ReAct (Yao et al. 2023) 20 20 23.70 37.80 90.70
Self-Ask (Press et al. 2023) 20 20 27.90 38.50 84.00
HippoRAG (Gutiérrez et al. 2024) 20 20 28.17 40.49 72.15
SuRe (Kim et al. 2024) 20 20 30.30 43.00 89.70
Adaptive-RAG (Jeong et al. 2024) 20 20 30.30 43.70 88.50
IRCoT (Trivedi et al. 2023) 20 20 33.10 46.20 88.80
ChatQA (Liu et al. 2024b) 20 20 34.93 47.19 86.14
ConvSearch-R1 (Zhu et al. 2025a) 20 20 33.39 47.9693.38
RQ-RAG (Chan et al. 2024) 5 14.72 28.05 38.82 77.95
RQ-RAG†5 14.72 31.26 42.34 77.95
ChainRAG†(Zhu et al. 2025b) 20 21.76 35.54 48.72 83.88
LogicRAG†(Chen et al. 2026) 20 26.83 37.32 51.69 82.28
RLTST (Fan et al. 2023) 20 20 32.20 44.70 89.60
StructuredDDP (Chi and Rudnicky 2022) 20 20 36.30 50.80 85.60
CMT-RAG (ours) 5 13.99 41.73 55.63 86.25
Oracle traces 5 14.90 42.18 56.23 88.08
Llama-3.3-70B-InstructIRCoT (Trivedi et al. 2023) 20 20 37.27 48.72 88.77
StructuredDDP (Chi and Rudnicky 2022) 20 20 37.81 49.31 85.56
ConvSearch-R1 (Zhu et al. 2025a) 20 20 40.57 52.5793.88
LogicRAG†(Chen et al. 2026) 20 32.13 39.29 53.15 80.39
Direct C-RAG 20 20 40.62 53.70 88.42
CMT-RAG (ours) 5 14.05 44.70 57.55 85.10
Oracle traces 5 15.30 47.22 60.82 89.83
Table1:MainresultsontheMuMu-QAlong-dialoguesplit.Allnon-oraclesystemsuseDRAGONretrieval.Top-k⋆isselected
fromk∈{5,10,20}by F1 for each baseline. Avg. Paras. denotes the mean number of unique paragraphs in the reader context
per turn after deduplication, and GoldCtx the mean recall of gold supporting paragraphs.†denotes replaying the accumulated
question–answerhistorybeforeanswergeneration.Oracletracesreplaceonlythegeneratedtracedraftswithgoldtraces,leaving
retrieval and the reader unchanged.
(6–32 turns), and ultra-long (33–67 turns). The short and
long splits are used for training, while the ultra-long split is
reserved for the evaluation of length-extrapolation.
Baselines.WecompareCMT-RAGwithfivebaselinefam-
ilies. Direct C-RAG retrieves with the unresolved current-
turn query. Iterative and agentic retrieval methods include
ReAct, Self-Ask, HippoRAG, SuRe, Adaptive-RAG, and
IRCoT. Conversational context methods include ChatQA
andConvSearch-R1.Query-decompositionmethodsinclude
RQ-RAG, ChainRAG, and LogicRAG. Dialogue-structure
methods include RLTST and StructuredDDP. Within each
readersetting,allnon-oraclesystemsusethesameDRAGON
corpus index, while each baseline retains its native reason-
ing or decomposition procedure. The†variants additionally
replaytheaccumulatedquestion–answerhistoryateachturn.
Implementation details.We initialize the trace generator
fromMamba-2-2.7BandtrainLoRAadapterswiththethree-
stage SFT curriculum in Section 3.3, followed by reader-
specific DPO with reward weightsγ= 0.2. Unless oth-
erwise specified, CMT-RAG retrieves five paragraphs per
resolvedsub-questionusingDRAGONandadditionaltraces
via keyword-overlap DAG lookup. Full training and hyper-
parameter details are provided in Appendix B.
Evaluation metrics.For end-task QA, we report Exact
Match (EM) and token-level F1 against turn-level gold an-
swers.Avg.Paras.isthemeannumberofuniqueparagraphsinthefinalreadercontextperturnaftermerginganddedupli-
cation.GoldCtxisthemeanper-turnrecallofgoldsupporting
paragraphsinthatfinalcontext.Forefficiencyanalyses,end-
to-end latency includes trace or plan generation, retrieval,
intermediatereadercalls,andfinal-answergeneration,while
excluding one-time model and index initialization. Further
details appear in Appendix C.
4.2 Main Results across Readers (RQ1)
We evaluate on the MuMu-QA long-dialogue split with
Qwen3-32B and Llama-3.3-70B-Instruct as stateless read-
ers.ThetwoCMT-RAGvariantssharethesameStage3SFT
checkpoint,whileeachusesaDPOadaptertrainedfrompref-
erence pairs generated with the corresponding reader. This
protocolevaluatescompatibilitywithtworeadersratherthan
zero-shot reader swapping. At inference, CMT-RAG carries
the SSM state across turns, retrieves fresh paragraphs with
DRAGON(Linetal.2023)fromresolvedsub-questions,and
uses trace keywords for long-range DAG lookup.
Table 1 shows that the iterative baselines do not sur-
pass Direct C-RAG under the shared evaluation protocol.
ConvSearch-R1 obtains the highest GoldCtx recall without
attaining the highest answer accuracy. CMT-RAG achieves
the best non-oracle EM/F1 with Qwen3-32B (41.73/55.63)
and Llama-3.3-70B-Instruct (44.70/57.55). Relative to Di-
rect C-RAG, the gains are 6.07 EM and 6.82 F1 with
Qwen and 4.08 EM and 3.85 F1 with Llama. These gains
are obtained with approximately 14 unique paragraphs per

turn instead of 20. CMT-RAG does not attain the highest
GoldCtx recall, so the evidence supports more effective use
of a smaller retrieved context rather than uniformly better
supporting-paragraph retrieval. Oracle traces add 0.60 F1
withQwenand3.27F1withLlama,quantifyingtheremain-
ing headroom in trace generation.
4.3 Component Contributions (RQ2)
To isolate the contributions of the CMT framework and the
trainedtracegenerator,wecomparethreesettings.(1)Acon-
trol setting that performs reader-based query decomposition
without constructing memory traces or modeling explicit
cross-turn dependencies. (2) A CMT-only setting that intro-
duces the complete trace-management framework but uses
thereader,insteadofatrainedSSM,togeneratetracedrafts.
(3) The full CMT-RAG model, in which the trace drafts are
generated by the SFT+DPO-trained trace generator.
Components Qwen3-32B Llama-3.3-70B-It
CMT L. Gen. MP EM↑(%) F1↑(%) MP EM↑(%) F1↑(%)
✕ ✕9.79 35.67 47.11 9.89 39.38 51.72
✓ ✕13.16 37.86 50.89 13.77 41.46 54.38
✓ ✓13.9941.73 55.6314.0544.70 57.55
Table 2: Ablation of the CMT framework and learned trace
generator (L. Gen.). CMT performs trace construction, per-
sistence, and DAG lookup. When L. Gen. is disabled, the
reader generates trace drafts. MP denotes the mean number
of unique paragraphs provided to the reader per turn.
Table 2 separates the gain from complementary mem-
ory traces and the learned generator. With reader-generated
traces,enablingCMTraisesF1by3.78pointsforQwenand
2.66pointsforLlama.Replacingreader-generateddraftswith
thetrainedSSMtracegeneratoryieldsafurther4.74and3.17
F1-pointgain,respectively.AlthoughCMTincreasestheav-
erage number of retrieved paragraphs from about 10 to 14
perturn,theconsistentimprovementsinEMandF1indicate
thattheadditionalevidenceiseffectivelyutilizedratherthan
introducing distracting context.
4.4 Memory across Dialogue Lengths (RQ3)
Weconductanablationstudytoevaluatethetwocomplemen-
tary memory components of CMT-RAG. All experiments
follow the same settings as Table 1 with Qwen3-32B. We
further conduct evaluation on the ultra-long dialogue split
with 33–67 turns of MuMu-QA to assess the robustness of
the trained trace generator under dialogue lengths beyond
those seen during training.
Figure4showsdistinctlengthprofilesforthetwomemory
channels.RemovingtheDAGchangesF1by0.02,0.04,0.46,
and1.15pointsacrossthe3–7,6–15,16–28,and33–67turn
bins, respectively, concentrating the DAG benefit in longer
dialogues. Removing cross-turn SSM state reduces F1 by
8.49, 8.46, 11.14, and 7.11 points, making recurrent state
the larger contributor in every length regime. Overall, the
resultsconfirmthecomplementaryrolesofthetwomemory
channels across different dialogue lengths.
3--7 6--15 16--28 33--67
Dialogue length40455055F155.5556.28
55.17
50.1055.5356.24
54.71
48.95
47.0647.82
44.03
42.99ultra-long splitCMT-RAG w/o DAG w/o SSM state carry-overFigure4:Ablationofthecomplementarymemoryframework
across dialogue lengths.w/o DAGremoves persistent trace-
DAGlookupwhileretainingSSMruntimememory.w/oSSM
statedisablescross-turnSSMstatecarry-overwhileretaining
persistent trace-DAG memory.
4.5 Backbone Comparison (RQ4)
We further evaluate a Transformer-based variant of CMT-
RAG by replacing the original SSM backbone with Pythia-
2.8B(Bidermanetal.2023),whoseparameterscaleiscom-
parable to that of the original Mamba2-2.7B backbone. We
train the model on MuMu-QA using the same curriculum
SFTandDPOprocedureastheSSM-basedcounterpart.The
resulting model serves as the decomposer and produces the
same structured trace format described in Section 3.1.
Model Group EM↑(%) F1↑(%) Avg. Paras. E2E Time↓(s)
Trans. SFT 39.96 53.83 11.93 1.59
SSM SFT 40.14 53.90 13.29 0.75
Trans. +DPO 40.09 54.03 11.96 1.61
SSM +DPO 41.73 55.63 13.99 0.83
Table3:ComparisonofSSMandTransformertracegenera-
torsontheMuMu-QAlong-dialoguesplitusingQwen3-32B
asthereader.TheSSMandTransformerareinstantiatedwith
Mamba-2-2.7B and Pythia-2.8B, respectively. E2E time (s/-
turn)includesdecomposition,retrieval,andreaderinference.
As shown in Table 3, the SSM-based decomposer con-
sistently outperforms the Transformer baseline under both
training settings. Under SFT, it achieves modest gains of
0.18 EM and 0.07 F1. The advantage becomes larger af-
ter DPO, reaching 41.73 EM and 55.63 F1, surpassing the
Transformer by 1.64 EM and 1.60 F1, respectively. Beyond
answer quality, the SSM is also substantially more efficient.
In our implementation, it maintains a recurrent state across
turns, whereas Pythia re-encodes the accumulated dialogue
historyateveryturn,reducingend-to-endlatencyfrom1.61
to 0.83 seconds per turn after DPO (a 48.4% reduction).
4.6 Transfer across Benchmarks (RQ5)
We further evaluate transfer on the conversational retrieval
benchmark RECOR (Ali et al. 2026) and the single-turn
multi-hopQAbenchmarksHotpotQA(Yangetal.2018)and
2WikiMultiHopQA (Ho et al. 2020), using Qwen3-32B as
the reader. All methods retrieve from a shared benchmark

RECOR HotpotQA 2WikiMultiHopQA
MethodF1↑(%) BLEU-1↑ROUGE-L↑E2E Time↓(ms) EM↑(%) F1↑(%) E2E Time↓(ms) EM↑(%) F1↑(%) E2E Time↓(ms)
IRCoT 31.85 26.04 22.05 3192 33.02 42.39 2903 23.24 27.11 2971
ConvSearch-R1 31.35 24.66 21.71 5648 27.99 39.45 3357 14.07 20.72 3751
LogicRAG 28.98 22.58 19.97 14786 43.65 56.95447438.11 44.684794
StructuredDDP 32.80 26.88 22.66 1987 – – – – – –
Direct C-RAG 32.57 29.70 23.19 343 38.31 49.16 182 27.25 32.64 147
CMT-RAG (ours) 36.42 33.52 25.70 2455 44.92 56.76 549 36.42 42.93 570
Table 4: Results on RECOR, HotpotQA, and 2WikiMultiHopQA. All methods retrieve from the same open corpus. We report
token-level F1, BLEU-1, and ROUGE-L on RECOR; EM and token-level F1 on HotpotQA and 2WikiMultiHopQA. E2E time
includesdecomposition(orreasoning),passageretrieval,andreaderinference,averagedoverdialogueturns.Allmethodsrerank
the merged retrieval candidates and retain at most 10 passages for the final reader.
corpus with the same DRAGON dense retriever (Lin et al.
2023); for HotpotQA and 2WikiMultiHopQA, the released
per-example contexts are merged into benchmark-level cor-
pora,ratherthanusingtheofficialFullWikisetting.Wereport
EMandtoken-levelF1onthetwodevelopmentsets,andF1,
BLEU-1, and ROUGE-L across all RECOR dialogue turns.
Table 4 compares CMT-RAG with the strongest imple-
mented representative from each baseline family. On the
multi-turnbenchmarkRECOR,CMT-RAGachievesthebest
results on all reported metrics, demonstrating that comple-
mentary memory traces effectively capture and reuse con-
versationalcontextbeyondthesettingofMuMu-QA.Onthe
single-turnbenchmarksHotpotQAand2WikiMultiHopQA,
CMT-RAGalsoremainscompetitive,achievingthebestEM
on HotpotQA and the second-best EM and F1 on 2Wiki-
MultiHopQA. Although LogicRAG attains slightly stronger
results on 2WikiMultiHopQA, it relies on multiple itera-
tive reader calls, incurring substantially higher inference la-
tency and requiring 8.4×more end-to-end inference time
than CMT-RAG. By contrast, CMT-RAG delegates conver-
sational parsing and memory management to a lightweight
SSM trace generator, achieving a substantially better ac-
curacy–efficiency trade-off while remaining competitive on
single-turn multi-hop QA.
5 Related Work
Conversational RAG and multi-hop retrieval address com-
plementary requirements of context-dependent information
seeking. CMT-RAG connects these directions by maintain-
ing retrieval-oriented sub-question traces whose dependen-
cies and evidence persist across dialogue turns.
Conversational RAG methods represent dialogue context
through history encoding (Yang et al. 2025; Qian et al.
2022), memory (Ye et al. 2026; Zhu et al. 2025c; Zhong
et al. 2024), or query reformulation (Wu et al. 2022; Anan-
tha et al. 2021). ChatQA (Liu et al. 2024b) encodes con-
versational context for retrieval, and ConvGQR (Mo et al.
2023) and ConvSearch-R1 (Zhu et al. 2025a) reformulate
context-dependent turns into standalone queries. These ap-
proacheseffectivelyresolvelocalambiguityandcoreference,
but leave retrieval dependencies among intermediate sub-
questions implicit. Dialogue discourse parsers and graph-
based conversational models make cross-turn structure ex-
plicit (Li et al. 2020; Shi and Huang 2019; Fan et al. 2023;ChiandRudnicky2022).Theirnodestypicallyrepresentut-
terancesordiscourseunits,andtheiredgesencodediscourse
relationsratherthandependenciesbetweenthesub-questions
that drive evidence retrieval. CMT-RAG instead represents
dialogue context at retrieval-oriented granularity, allowing
each current sub-question to address the specific prior trace
and evidence on which it depends.
Reasoning-basedretrievaldecomposescomplexquestions
into simpler units (Huang et al. 2023; Wolfson et al. 2020;
Perezetal.2020)oralternatesretrievalwithreasoning(Asai
et al. 2024; Verma et al. 2024). QDG (Hasson and Be-
rant 2021), RQ-RAG (Chan et al. 2024), ChainRAG (Zhu
etal.2025b),andLogicRAG(Chenetal.2026)exposesub-
question or dependency structure, whereas IRCoT (Trivedi
et al. 2023) and Self-Ask (Press et al. 2023) generate in-
termediate reasoning steps that guide successive retrieval.
These methods generally operate on self-contained queries
and do not persist the resulting sub-questions, dependen-
cies, and evidence as dialogue-level memory. Corpus-level
graphRAGmethodsorganizerelationsamongdocumentsor
entities(Gutiérrezetal.2024;Edgeetal.2024),whichcom-
plements rather than captures conversational dependencies.
CMT-RAGusesasession-leveltraceDAGwhosenodesbind
resolvedsub-questions,dependencylinks,lookupkeywords,
and supporting evidence, making the retrieval unit and the
memory unit the same persistent object across turns.
6 Conclusion
We presentCMT-RAG, a complementary memory frame-
work that combines recurrent SSM state for local conver-
sational context with a persistent trace DAG for long-range
dependencyresolutionandevidencereuse.Eachtracebinds
aresolvedsub-question,dependencylinks,lookupkeywords,
andtrace-localsupportingevidencewhilekeepingthereader
stateless. We also introduceMuMu-QA, which provides
sub-question-levelcross-turnsupervisionandlong-dialogue
evaluation. Experiments demonstrate that CMT-RAG im-
proves answer accuracy across two reader backbones while
maintaining compact retrieval contexts. As CMT-RAG re-
liesonaccuratesub-questiondecompositionanddependency
prediction, future work will focus on more robust trace gen-
eration. In addition, MuMu-QA is synthetic and inherits the
domain bias of MuSiQue, motivating further evaluation on
human-authored conversations.

References
Ali,M.;Abdallah,A.;Agarwal,A.;Patel,H.L.;andJatowt,
A.2026. RECOR:Reasoning-focusedMulti-turnConversa-
tional Retrieval Benchmark. InFindings of the Association
for Computational Linguistics: ACL 2026, 2688–2723.
Anantha, R.; Vakulenko, S.; Tu, Z.; Longpre, S.; Pulman,
S.;andChappidi,S.2021. Open-DomainQuestionAnswer-
ing Goes Conversational via Question Rewriting. InNorth
AmericanChapteroftheAssociationforComputationalLin-
guistics (NAACL), 520–534.
Asai, A.; Wu, Z.; Wang, Y.; Sil, A.; and Hajishirzi, H.
2024. Self-RAG: Learning to Retrieve, Generate, and Cri-
tique through Self-Reflection. InInternational Conference
on Learning Representations (ICLR).
Biderman, S.; Schoelkopf, H.; Anthony, Q. G.; Bradley,
H.; O’Brien, K.; Hallahan, E.; Khan, M. A.; Purohit, S.;
Prashanth, U. S.; Raff, E.; Skowron, A.; Sutawika, L.; and
van der Wal, O. 2023. Pythia: A Suite for Analyzing Large
Language Models Across Training and Scaling. InInter-
national Conference on Machine Learning (ICML), 2397–
2430.
Chan, C.-M.; Xu, C.; Yuan, R.; Luo, H.; Xue, W.; Guo, Y.;
and Fu, J. 2024. RQ-RAG: Learning to Refine Queries for
Retrieval Augmented Generation. InConference on Lan-
guage Modeling (COLM).
Chen, S.; Zhou, C.; Yuan, Z.; Zhang, Q.; Cui, Z.; Chen, H.;
Xiao, Y.; Cao, J.; and Huang, X. 2026. You Don’t Need
Pre-builtGraphsforRAG:RetrievalAugmentedGeneration
withAdaptiveReasoningStructures. InAAAIConferenceon
Artificial Intelligence (AAAI), 30270–30278.
Cheng, Y.; Mao, K.; Zhao, Z.; Dong, G.; Qian, H.; Wu, Y.;
Sakai, T.; Wen, J.-R.; and Dou, Z. 2025. CORAL: Bench-
marking Multi-turn Conversational Retrieval-Augmented
Generation. InFindings of the North American Chapter
of the Association for Computational Linguistics (NAACL
Findings), 1308–1330.
Chi, T.-C.; and Rudnicky, A. 2022. Structured Dialogue
DiscourseParsing. InAnnualMeetingoftheSpecialInterest
Group on Discourse and Dialogue (SIGDIAL), 325–335.
Dao, T.; and Gu, A. 2024. Transformers are SSMs: Gener-
alized Models and Efficient Algorithms Through Structured
StateSpaceDuality.InInternationalConferenceonMachine
Learning (ICML), 10041–10071.
Edge,D.;Trinh,H.;Cheng,N.;Bradley,J.;Chao,A.;Mody,
A.; Truitt, S.; Metropolitansky, D.; Ness, R. O.; and Lar-
son, J. 2024. From Local to Global: A Graph RAG Ap-
proach to Query-Focused Summarization.arXiv preprint
arXiv:2404.16130, 1–26.
Fan, Y.; Jiang, F.; Li, P.; Kong, F.; and Zhu, Q. 2023. Im-
proving Dialogue Discourse Parsing via Reply-to Structures
ofAddresseeRecognition.InConferenceonEmpiricalMeth-
odsinNaturalLanguageProcessing(EMNLP),8484–8495.
Gu, A.; and Dao, T. 2024. Mamba: Linear-Time Sequence
Modeling with Selective State Spaces. InConference on
Language Modeling (COLM).Gutiérrez, B. J.; Shu, Y.; Gu, Y.; Yasunaga, M.; and Su, Y.
2024. HippoRAG: Neurobiologically Inspired Long-Term
MemoryforLargeLanguageModels. InAdvancesinNeural
Information Processing Systems (NeurIPS), 59532–59569.
Hasson, M.; and Berant, J. 2021. Question Decomposition
with Dependency Graphs. InAutomated Knowledge Base
Construction (AKBC).
Ho, X.; Duong Nguyen, A.-K.; Sugawara, S.; and Aizawa,
A.2020. ConstructingAMulti-hopQADatasetforCompre-
hensiveEvaluationofReasoningSteps.InProceedingsofthe
28thInternationalConferenceonComputationalLinguistics
(COLING), 6609–6625.
Hu, Y.; Wang, Y.; and McAuley, J. 2026. Evaluating Mem-
ory in LLM Agents via Incremental Multi-turn Interactions.
InInternational Conference on Learning Representations
(ICLR).
Huang, X.; Cheng, S.; Shu, Y.; Bao, Y.; and Qu, Y. 2023.
QuestionDecompositionTreeforAnsweringComplexQues-
tions over Knowledge Bases. InAAAI Conference on Artifi-
cial Intelligence (AAAI), 12924–12932.
Jeong,S.;Baek,J.;Cho,S.;Hwang,S.J.;andPark,J.2024.
Adaptive-RAG: Learning to Adapt Retrieval-Augmented
Large Language Models through Question Complexity. In
North American Chapter of the Association for Computa-
tional Linguistics (NAACL), 7036–7050.
Katsis, Y.; Rosenthal, S.; Fadnis, K.; Gunasekara, C.; Lee,
Y.-S.; Popa, L.; Shah, V.; Zhu, H.; Contractor, D.; and
Danilevsky,M.2025.MTRAG:AMulti-TurnConversational
BenchmarkforEvaluatingRetrieval-AugmentedGeneration
Systems.TransactionsoftheAssociationforComputational
Linguistics (TACL), 13: 784–808.
Khot,T.;Trivedi,H.;Finlayson,M.;Fu,Y.;Richardson,K.;
Clark,P.;andSabharwal,A.2023. DecomposedPrompting:
A Modular Approach for Solving Complex Tasks. InInter-
national Conference on Learning Representations (ICLR).
Kim, J.; Nam, J.; Mo, S.; Park, J.; Lee, S.-W.; Seo, M.; Ha,
J.-W.; and Shin, J. 2024. SuRe: Summarizing Retrievals
using Answer Candidates for Open-domain QA of LLMs.
InInternational Conference on Learning Representations
(ICLR).
Laban,P.;Hayashi,H.;Zhou,Y.;andNeville,J.2026. LLMs
Get Lost In Multi-Turn Conversation. InInternational Con-
ference on Learning Representations (ICLR).
Li, J.; Liu, M.; Kan, M.-Y.; Zheng, Z.; Wang, Z.; Lei, W.;
Liu,T.;andQin,B.2020. Molweni:AChallengeMultiparty
Dialogues-based Machine Reading Comprehension Dataset
with Discourse Structure. InInternational Conference on
Computational Linguistics (COLING), 2642–2652.
Lin, S.-C.; Asai, A.; Li, M.; Oguz, B.; Lin, J.; Mehdad, Y.;
Yih,W.-t.;andChen,X.2023.HowtoTrainYourDRAGON:
Diverse Augmentation Towards Generalizable Dense Re-
trieval. InFindings of the Association for Computational
Linguistics: EMNLP 2023, 6385–6400.
Liu, N. F.; Lin, K.; Hewitt, J.; Paranjape, A.; Bevilacqua,
M.; Petroni, F.; and Liang, P. 2024a. Lost in the Middle:
HowLanguageModelsUseLongContexts.Transactionsof

the Association for Computational Linguistics (TACL), 12:
157–173.
Liu, Z.; Ping, W.; Roy, R.; Xu, P.; Lee, C.; Shoeybi, M.;and
Catanzaro, B. 2024b. ChatQA: Surpassing GPT-4 on Con-
versationalQAandRAG.InAdvancesinNeuralInformation
Processing Systems, volume 37, 15416–15459.
Mo, F.; Mao, K.; Zhu, Y.; Wu, Y.; Huang, K.; and Nie,
J.-Y. 2023. ConvGQR: Generative Query Reformulation for
ConversationalSearch.InAnnualMeetingoftheAssociation
for Computational Linguistics (ACL), 4998–5012.
OpenAI.2025. OpenAI:gpt-oss-120b&gpt-oss-20bModel
Card.arXiv preprint arXiv:2508.10925, 1–34.
Perez, E.; Lewis, P.; Yih, W.-t.; Cho, K.; and Kiela, D.
2020. Unsupervised Question Decomposition for Question
Answering. InConferenceonEmpiricalMethodsinNatural
Language Processing (EMNLP), 8864–8880.
Press, O.; Zhang, M.; Min, S.; Schmidt, L.; Smith, N. A.;
and Lewis, M. 2023. Measuring and Narrowing the Com-
positionality Gap in Language Models. InFindings of the
Association for Computational Linguistics: EMNLP 2023,
5687–5711.
Qian, J.; Zou, B.; Dong, M.; Li, X.; Aw, A. T.; and Hong,
Y. 2022. Capturing Conversational Interaction for Question
Answering via Global History Reasoning. InFindings of
the North American Chapter of the Association for Compu-
tational Linguistics (NAACL Findings), 2065–2075.
Shi, Z.; and Huang, M. 2019. A Deep Sequential Model
for Discourse Parsing on Multi-Party Dialogues. InAAAI
Conference on Artificial Intelligence (AAAI), 7007–7014.
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A. 2022. MuSiQue: Multihop Questions via Single-hop
Question Composition.Transactions of the Association for
Computational Linguistics (TACL), 10: 539–554.
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A.2023. InterleavingRetrievalwithChain-of-ThoughtRea-
soning for Knowledge-Intensive Multi-Step Questions. In
Annual Meeting of the Association for Computational Lin-
guistics (ACL), 10014–10037.
Verma, P.; Midigeshi, S. P.; Sinha, G.; Solin, A.; Natara-
jan, N.; and Sharma, A. 2024. Plan×RAG: Planning-
guided Retrieval Augmented Generation.arXiv preprint
arXiv:2410.20753, 1–19.
Wolfson,T.;Geva,M.;Gupta,A.;Gardner,M.;Goldberg,Y.;
Deutch,D.;andBerant,J.2020. BreakItDown:AQuestion
Understanding Benchmark.Transactions of the Association
for Computational Linguistics (TACL), 8: 183–198.
Wu, Z.; Luan, Y.; Rashkin, H.; Reitter, D.; Hajishirzi, H.;
Ostendorf, M.; and Tomar, G. S. 2022. CONQRR: Conver-
sational Query Rewriting for Retrieval with Reinforcement
Learning. InConference on Empirical Methods in Natural
Language Processing (EMNLP), 10000–10014.
Yang, S.; Lee, J.; Bang, J.; Shim, K.; Kim, M.; and Chang,
S. 2025. Learning Contextual Retrieval for Robust Conver-
sational Search. InConference on Empirical Methods in
Natural Language Processing (EMNLP), 11991–12003.Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y.; Cohen, W. W.;
Salakhutdinov, R.; and Manning, C. D. 2018. HotpotQA:
A Dataset for Diverse, Explainable Multi-hop Question An-
swering. InProceedings of the 2018 Conference on Em-
piricalMethodsinNaturalLanguageProcessing(EMNLP),
2369–2380.
Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan,
K.R.;andCao,Y.2023. ReAct:SynergizingReasoningand
ActinginLanguageModels. InInternationalConferenceon
Learning Representations (ICLR).
Ye, L.; Yu, L.; Lei, Z.; Chen, Q.; Zhou, J.; and He, L. 2025.
OptimizingQuestionSemanticSpaceforDynamicRetrieval-
AugmentedMulti-hopQuestionAnswering.InAnnualMeet-
ing of the Association for Computational Linguistics (ACL),
17814–17824.
Ye, Z.; Huang, J.; Chen, W.; and Zhang, Y. 2026. H-Mem:
HybridMulti-DimensionalMemoryManagementforLong-
Context Conversational Agents. InConference of the Euro-
peanChapteroftheAssociationforComputationalLinguis-
tics (EACL), 7756–7775.
Yu, Y.; Jiang, H.; Luo, X.; Wu, Q.; Lin, C.-Y.; Li, D.; Yang,
Y.; Huang, Y.; and Qiu, L. 2025. Mitigate Position Bias
in LLMs via Scaling a Single Hidden States Channel. In
Findings of the Association for Computational Linguistics:
ACL 2025, 6092–6111.
Zhong, W.; Guo, L.; Gao, Q.; Ye, H.; and Wang, Y. 2024.
MemoryBank: Enhancing Large Language Models with
Long-TermMemory. InAAAIConferenceonArtificialIntel-
ligence (AAAI), 19724–19731.
Zhu, C.; Wang, S.; Feng, R.; Song, K.; and Qiu, X. 2025a.
ConvSearch-R1: Enhancing Query Reformulation for Con-
versationalSearchwithReasoningviaReinforcementLearn-
ing. InConference on Empirical Methods in Natural Lan-
guage Processing (EMNLP), 26547–26564.
Zhu,R.;Liu,X.;Sun,Z.;Wang,Y.;andHu,W.2025b. Mit-
igating Lost-in-Retrieval Problems in Retrieval Augmented
Multi-Hop Question Answering. InAnnual Meeting of the
Association for Computational Linguistics (ACL), 22362–
22375.
Zhu, Z.; Hu, T.; Zhang, H.; Yang, D.; Chen, H.; Zhang, M.;
and Chen, X. 2025c. CID-GraphRAG: Enhancing Multi-
Turn Dialogue Systems through Dual-Pathway Retrieval of
Conversation Flow and Context Semantics.arXiv preprint
arXiv:2506.19385, 1–18.

A MuMu-QA Construction
MuMu-QA is designed to evaluate multi-turn multi-hop
RAG under sub-question-level cross-turn dependencies.
Startingfromthesub-questiondecompositionsandsupport-
ingevidenceprovidedbyMuSiQue(Trivedietal.2022),we
reorganizeindependentreasoningchainsintoconversational
sessionsinwhichlaterturnsmaydependonintermediatere-
sults established earlier. The construction process preserves
theoriginalreasoningandevidencesupervisionwhileintro-
ducing dialogue-level dependency structure, enabling con-
trolledevaluationofsub-questiondecomposition,cross-turn
trace linking, and evidence reuse. This section details the
sourcedata,dialoguesynthesisprocedure,splitconstruction,
and annotation schema.
A.1 Source Data and Filtering
MuMu-QA is constructed from the answerable split of
MuSiQue,whichprovidesmulti-hopquestionstogetherwith
supportingparagraphs,sub-questiondecompositions,andin-
termediate answers. We exclude the unanswerable portion
of the full split because MuMu-QA targets cross-turn de-
pendency tracking rather than answerability detection or re-
fusal behavior. We further remove near-duplicate examples
whose decompositions and answers are effectively identi-
cal, preventing synthesized dialogues from collapsing into
paraphrased repetitions of the same reasoning chain.
A.2 Dialogue Synthesis
MuMu-QA is synthesized in two stages. First, deterministic
graphoperationsconstructsub-questionnodes,intermediate
answers, dependency edges, and evidence annotations di-
rectlyfromtheMuSiQuereasoninggraphs.Second,anLLM
realizes the resulting graph fragments as natural conversa-
tional questions while preserving the underlying reasoning
structure. Long- and ultra-long dialogues are subsequently
obtained by interleaving synthesized sessions and globally
remappingtraceidentifiers,dependencies,andparagraphin-
dices. The subsequent interleaving and identifier-remapping
stages require no additional LLM calls.
Graph synthesis.MuMu-QA uses two complementary
graph-level synthesis operations.Sub-question Relocation
movesanindependentlyanswerablesub-questionfromalater
source graph into an earlier conversational turn. The origi-
nal parent question is rewritten so that its reasoning natu-
rallyincorporatestherelocatedresult,whilethelatersource
question becomes a follow-up that explicitly depends on the
relocated trace. This operation introduces cross-turn depen-
dencies without modifying the remaining reasoning graph
or paragraph supervision.Graph Splicingchains multiple
MuSiQuereasoninggraphsbymakingafollow-upgraphde-
pendonanintermediateanswerestablishedinanearlierturn.
Foreachfollow-up,weretaintheminimalsubgraphrequired
to derive its final answer and reconnect the selected entry
nodetotheprecedingtrace.AteacherLLMthenrewritesthe
selected reasoning graph into a natural conversational ques-
tion, while all intermediate answers, dependency relations,
and supporting-evidence annotations are inherited directly
from the underlying MuSiQue graphs.LLM-based question realization.Only the reader-facing
turn questions are generated by an LLM. All sub-question
nodes, intermediate answers, dependency edges, and evi-
dence annotations are deterministically inherited from the
original MuSiQue graphs. Depending on the synthesis op-
erator and graph structure, different question-realization
prompts are applied, as summarized in Table 5. Long- and
ultra-longdialoguesynthesisdoesnotinvoketheLLMagain;
thesestagesonlyinterleavepreviouslysynthesizeddialogues
and globally remap sub-question identifiers, answer refer-
ences, dependency edges, and paragraph indices.
Operator Case Realization
GS Original graph Original
Partial graph Prompt A
Dependency follow-up Prompt C
SQR Carrier question Prompt B
Relocated follow-up Prompt C
Unchanged question Original
Table 5: Question realization under the two Stage 2 syn-
thesis operators. GS and SQR denote Graph Splicing and
Sub-question Relocation, respectively. “Original” indicates
directreuseoftheoriginal(orconversationalized)MuSiQue
question without LLM generation.
Thedefaultrealizationbackendusestheopen-sourceLLM
(currentlygpt-oss-120b(OpenAI2025))withtempera-
ture 0.2, a maximum of 220 generated tokens, and at most
two generation attempts. Any comparable instruction-tuned
LLM can be used. We choose GPT-OSS solely because it is
open-source and reproducible.
Prompt templates.Prompt A is used only when the first
graph-splice turn corresponds to a dependency closure end-
ingatanintermediateMuSiQuenoderatherthanacomplete
source question. Prompt B realizes the carrier turn after re-
locating an independently answerable sub-question from a
later reasoning graph while preserving the carrier’s original
final answer. Prompt C is shared by graph-splice follow-
upturnsandrelocatedsourceturns.Itrequiresthegenerated
questiontorefertothepreviousintermediateanswerthrough
an entity-type-compatible expression (e.g., “that person” or
“that city”) rather than explicitly mentioning the answer it-
self. The complete prompt templates are listed below.
Prompt A (Partial graph realization).Fuseapartialrea-
soning graph into a natural parent question.
1You are given a partial reasoning graph
from MuSiQue.
2
3Generate one natural parent question
whose answer is the target answer.
4
5Input:
6- Original MuSiQue question
7- Selected reasoning steps
8- Target answer
9
10Requirements:

11- The question must be answerable using
only the selected reasoning steps.
12- Do not reveal the target answer.
13- Return JSON:
14{"question": "..."}
Prompt B (Carrier question realization).Generate a
carrier question that naturally preserves a relocated sub-
question.
1You are given a reasoning graph
containing its original reasoning
chain and one relocated auxiliary sub
-question.
2
3Generate one natural parent question
whose final answer remains unchanged
while naturally incorporating the
auxiliary reasoning step.
4
5Input:
6- Original MuSiQue question
7- Carrier reasoning graph
8- Relocated sub-question
9- Target answer
10
11Return JSON:
12{"question": "..."}
Prompt C (Dependency follow-up realization).Generate
a context-dependent follow-up question using implicit refer-
ences.
1You are given a reasoning graph whose
first step depends on a previous
conversational answer.
2
3Generate one natural follow-up question
using the specified reference phrase
(e.g., "that city") instead of
explicitly mentioning the previous
answer.
4
5Input:
6- Previous answer
7- Reference phrase
8- Selected reasoning graph
9- Original MuSiQue question
10- Final answer
11
12Requirements:
13- Use the reference phrase.
14- Do not reveal either the previous
answer or the final answer.
15- Return JSON:
16{"question": "..."}
Generation validation.Generatedquestionsareautomat-
ically validated before being included in MuMu-QA. We
reject generations that omit required reference phrases, re-
veal bridge entities or final answers, violate entity-type con-
straints, contain malformed or repetitive wording, or exceed
the prescribed length limit. As a result, LLM generation
is restricted to the natural-language realization of reader-
facing questions, while the trace graph, dependency struc-
ture,intermediateanswers,andsupporting-evidenceannota-tionsremainidenticaltothoseinheritedfromtheunderlying
MuSiQue graphs.
A.3 Splits and Leakage Control
We partition dialogues by grouped supporting-document ti-
tlesandbridgeentitiesratherthansynthesizeddialogueiden-
tifiers, preventing train and development splits from sharing
nearly identical evidence configurations or intermediate an-
swers under different surface forms. Supporting paragraphs
inherited from MuSiQue serve as the gold evidence annota-
tions.
Table 6 summarizes the resulting dataset statistics. For
each dialogue, we report the number of turns (T d), global
sub-questions (S d), and cross-turn dependency edges (E d).
Turns, SubQ, and X-Edge denote the minimum–maximum
rangeswithineachsplit,whereasAvg.T,Avg.SQ,andAvg.
X-Ereportthecorrespondingunweightedper-dialogueaver-
ages.Here,X-Edgecountsindividualdependencylinksfrom
a current-turn sub-question to prerequisite sub-questions in-
troduced in earlier turns.
A.4 Annotation Schema
Eachdialoguecontainsadialogue-wideglobal_subquestions
namespace, cross-turn dependency links, and one trace
record for every node in the session DAG. Each record
specifies current trace identifier, a sub-question, trace key-
words used for DAG lookup, predecessor trace identifiers,
and supporting-paragraph identifiers.
MuMu-QA therefore evaluates whether a system can de-
compose each turn into sub-question-level retrieval units,
connect these units to prerequisite traces from earlier turns,
and ground each trace in paragraph-level evidence that may
be reused later in the dialogue.
A.5 Breakdown by Synthesis Mode
MuMu-QA is synthesized using two complementary oper-
ations with different structural characteristics.Sub-question
Relocationintroduces cross-turn dependencies by relocat-
ingintermediatereasoningstepsacrossconversationalturns,
whereasGraph Splicingconstructs longer reasoning chains
by connecting multiple source reasoning graphs. In the
Stage 3 development split, Graph Splicing accounts for
75.39% of evaluation turns and Sub-question Relocation for
the remaining 24.61%. The mixed Stage 3 setting therefore
reflects the natural distribution of both synthesis modes in
the final benchmark.
Mode EM↑(%) F1↑(%) Avg. Paras.
Sub-question Relocation 39.98 55.55 15.76
Graph Splicing 42.31 55.66 13.41
Mixed (Stage 3) 41.73 55.63 13.99
Table7:PerformanceofCMT-RAGacrossdifferentMuMu-
QAsynthesismodes.“Avg.Paras.”denotestheaveragenum-
ber of unique retrieved paragraphs per turn after deduplica-
tion.
As shown in Table 7, CMT-RAG performs consistently
across the two synthesis modes. Graph Splicing achieves

Split Part. Dial. Turns Avg. T SubQ Avg. SQ X-Edge Avg. X-E
Short-dialogueTrain 3,104 3–7 3.96 3–25 8.80 1–6 1.84
Dev 362 3–7 3.56 3–25 8.25 1–6 1.95
Long-dialogueTrain 5,045 6–32 12.17 6–112 27.37 2–26 5.71
Dev 548 6–28 9.85 8–101 22.90 2–23 5.88
Ultra-long stress Dev 200 33–67 52.93 50–235 122.05 8–49 32.23
Table 6: Statistics of the MuMu-QA splits. Each row reports the number of dialogues together with the range and average of
dialogueturns,globalsub-questions,andcross-turndependencyedges.Cross-turnedgesreferonlytodependenciespointingto
sub-questions in earlier turns.
higher EM (42.31 vs. 39.98) while using fewer retrieved
paragraphs (13.41 vs. 15.76), whereas both modes obtain
nearly identical F1 scores. The mixed Stage 3 split closely
matches the overall benchmark performance, indicating that
CMT-RAGgeneralizeswellacrossdialoguesynthesisstrate-
gies with different cross-turn dependency structures.
B Details of Trace Generator
B.1 Architecture
Figure 5 illustrates the architecture of the SSM-based trace
generator. We instantiate the generator with a pretrained
Mamba-2backbone,whoseselectivestate-spacemixermain-
tains a recurrent hidden state throughout the dialogue. Un-
like Transformer-based generators that must replay the en-
tire dialogue history at every turn, the SSM processes each
new query incrementally while propagating its hidden state
between consecutive turns. This recurrent state serves as a
compactshort-termmemorythatsummarizesrecentconver-
sational context and enables efficient long-dialogue genera-
tion.
Input
Proj.[TRACE] [SubQ] [/TRACE] [KW] [DEP]
h1 ht−1 ht hTCurrent 
Query
��Input
StreamPretrained SSM
���� �∆�
Conv1D��
 ���Causal Sequence Mixer
A × EXP ��
��Disc.��
 ���ℎ�−1
����+
 �ℎ�
���h�
+ �� Gate+
Output
Proj.��+1
Figure5:ArchitectureoftheSSM-basedtracegenerator.The
recurrent SSM state carries dialogue context across turns,
while the decoder generates structured sub-question, key-
word, and dependency fields.
At dialogue turnt, the current user query is appended to
the input stream and processed together with the recurrent
stateh t−1inherited from the previous turn. After passing
through the stacked selective SSM blocks, the model up-
dates its hidden state toh t, which is preserved and reusedwhen processing the next user turn. During autoregressive
decoding,thegeneratoremitsastructuredtraceconsistingof
dedicated control tokens together with multiple trace fields,
[TRACE],[SubQ],[KW],[DEP],[/TRACE],
where each trace records a decomposed sub-question, its
retrievalkeyword,anddependenciesonpreviouslygenerated
traces. These structural tokens are added to the tokenizer
vocabularybeforefine-tuningsothattracegenerationcanbe
learned through standard causal language modeling.
Following common parameter-efficient fine-tuning prac-
tice,LoRAadaptersareinsertedonlyintotheinputandout-
put projection layers of each Mamba-2 block, while all pre-
trained backbone parameters remain frozen. Consequently,
themodellearnstogeneratestructuredtraceswhilepreserv-
ing the long-context modeling capability inherited from the
pretrained SSM.
B.2 Computing Infrastructure
All training and evaluation experiments were conducted on
a Linux server equipped with four NVIDIA H100 GPUs
(80GBHBM3memoryeach;320GBtotal),twoIntelXeon
Platinum8468VCPUs(96physicalcoresand192hardware
threadsintotal),and2.0TiBsystemmemory.Theserverran
Ubuntu 22.04.4 LTS with Linux kernel 5.14.0.
Training environment:Python 3.10.19; PyTorch 2.3.1
(CUDA 12.1); Transformers 4.43.0; PEFT 0.11.1; Mamba-
SSM 2.2.2; causal-conv1d 1.4.0; Triton 2.3.1; FAISS 1.8.0;
Sentence-Transformers 5.3.0.
Inference environment:We used Qwen3-32B as the
primary reader and Llama-3.3-70B-Instruct for additional
reader-backbone experiments. Both readers were served in
BF16 using vLLM 0.11.0, PyTorch 2.8.0 with CUDA 12.8,
and Transformers 4.57.1.
B.3 Trace Generator Training
Followingthearchitecturedescribedabove,weoptimizethe
trace generator using a three-stage supervised curriculum
followed by Direct Preference Optimization (DPO). We use
theGPT-NeoXtokenizerassociatedwithMamba-2-2.7Band
extenditsvocabularywithsevenstructuraltokens:[PLAN],
[/PLAN],[TRACE],[/TRACE],[SubQ],[KW], and
[DEP].Eachgeneratedtraceconsistsofadecomposedsub-
question, a retrieval keyword for DAG lookup, and a list
of predecessor trace identifiers. During fine-tuning, we in-
sertLow-RankAdaptation(LoRA)modulesintotheMamba

in_projandout_projprojections and jointly optimize
the embedding and output rows corresponding to the newly
introducedstructuraltokens.Allremainingbackboneparam-
eters remain frozen.
Curriculum Supervised Fine-Tuning.We optimize the
trace generator using a three-stage supervised curriculum.
Stage 1 trains a rank-16 LoRA adapter on 9,653 single-
turn examples with 1,220 validation examples to learn the
trace syntax and basic decomposition structure. The result-
ing adapter is merged into the backbone before Stage 2,
which trains a rank-8 LoRA adapter on 3,104 multi-turn
dialogues from theshort-dialoguesplit. Stage3 con-
tinuestrainingthesameadapteron5,045dialoguesfromthe
long-dialogueextra-training split, supplemented with
1,009 randomly sampled dialogues from the Stage2 training
set for replay. Stage 2 and Stage 3 checkpoints are selected
accordingtovalidationloss.Table8summarizesthetraining
hyperparameters.
Setting Stage 1 SFT Stage 2 SFT Stage 3 SFT DPO
Training items 9,653 3,104 6,054 45,420 pairs
Max input Seq. Len. 1,024 4,096 8,192 5,120
LoRA learning rate5×10−45×10−52×10−55×10−7
Epochs 3 3 3 1
Batch / accumulation 4 / 1 1 / 8 1 / 8 1 / 1
LoRA rank /α16 / 32 8 / 16 8 / 16 8 / 16
LoRA dropout 0.05 0.10 0.10 0.00
DPOβ– – – 0.05
AdamW weight decay 0.05 0.05 0.05 0.01
Training arithmetic FP16 FP16 FP16 FP32
Table 8: Training configuration of the SSM trace generator
used for the reported results. Training-item counts denote
single-turnexamples,dialoguesessions,orofflinepreference
pairs. Stage 3 continues training from the Stage 2 LoRA
adapter.
Unless otherwise specified, all SFT stages use AdamW
withgradientclipping(maximumnorm1.0),alinearwarmup
over the first 5% of optimization steps followed by linear
learning-rate decay, and random seed 42. The maximum
numbersofgeneratedtracenodesare64and128forStages2
and 3, respectively.
Preference Optimization.Preference pairs are con-
structed from the selected SFT Stage 3 policy. For each di-
alogue turn with contextc, we sample four candidate traces
using temperature 1.0 and top-ksampling (k= 40). Each
candidate traceτis executed through the fixed retrieval–
reader pipeline and assigned the reward
R(τ;c) =F final(τ) +γF sub(τ).(9)
whereF finalis the final-answer F1 andF subis the average
F1 over semantically matched intermediate sub-questions.
Intermediate sub-questions are matched to the reference de-
compositionusingaminimumsemantic-similaritythreshold
of 0.55.
Preferencepairsareformedfromthehighest-andlowest-
rewardcandidateswhenevertheirrewarddifferenceisatleast
Current41.8 55.8
41.6 55.6
41.4 55.4
41.2 55.2
41.0 55.0
40.8 54.8EM (%) F1 (%)
41.0941.73
41.23
41.1455.2455.63
55.39
55.14
No
traceTop-1
traceTop-2
tracesTop-3
tracesCurrent42.0 56.0
41.0 55.0
40.0 54.0
39.0 53.0EM (%) F1 (%)
41.1441.73
39.7539.9254.6555.63
53.74
53.62
Top-3
parasTop-5
parasTop-8
parasTop-10
parasEM (left axis) F1 (right axis)Figure 6: Hyperparameter selection for CMT-RAG. Left:
number of traces retrieved from the session DAG. Right:
number of paragraphs retrieved per sub-question. Shaded
columns indicate the selected settings.
0.05. Candidate traces with invalid formats are filtered be-
fore finalizing the preference pairs, resulting in 45,420 of-
fline preference pairs. We merge the Stage 3 LoRA adapter
into the backbone and attach a new rank-8 LoRA adapter
containing 10.7M trainable parameters for DPO, while the
merged Stage 3 policy serves as the frozen reference model.
The policy is optimized for one epoch using standard DPO
withβ= 0.05, batch size 1, and no auxiliary SFT loss. No-
tably, DPO is run for a pre-specified single epoch, without
downstream-QA-based early stopping or checkpoint search.
B.4 Hyperparameter Selection
We conduct a one-factor-at-a-time study on the MuMu-QA
development set using Qwen3-32B. This held-out subset
was excluded from all SFT and DPO training, including
preference-pairconstruction..Unlessotherwisespecified,all
remaining settings are kept identical to those in Table 1. As
shownintheleftpanelofFigure6,retrievingonetracefrom
theDAGachievesthebestEM/F1of41.73/55.63,compared
with41.09/55.24withouttracelookup.Retrievingadditional
traces provides no further improvement.
The right panel shows that retrieving five paragraphs per
sub-question performs best. Increasingk parafrom 3 to 5
improves EM/F1 from 41.14/54.65 to 41.73/55.63, whereas
largerretrievalbudgetsreduceperformance.Wethereforeset
kDAG= 1andk para= 5in all experiments.
Finally,westudythematchedsub-questionrewardweight
γused to rank candidates during DPO preference-pair con-
struction. For each value ofγ, we recompute the composite
reward from the same candidate pool and reconstruct pref-
erence pairs using the same selection and reward-margin
criteria, while keeping all other training and evaluation set-
tings fixed. As shown in Figure 7, removing the matched
sub-question term (γ= 0) yields an EM/F1 of 39.90/53.68.
Settingγ= 0.2improves performance to 41.73/55.63,
the best point estimate among the evaluated settings. In-
creasing the weight further provides no consistent benefit:
γ= 0.4,0.6, and0.8obtain EM/F1 scores of 41.31/55.55,
41.49/55.53,and40.81/54.93,respectively.Theseresultsin-
dicate that moderate intermediate-answer supervision im-
proves preference construction, whereas overemphasizing
the sub-question term can weaken its alignment with final-
answer quality. We therefore setγ= 0.2in all DPO experi-
ments.

Selected42.0 56.0
41.5 55.5
41.0 55.0
40.5 54.5
40.0 54.0
39.5 53.5EM (%) F1 (%)
39.9041.73
41.3141.49
40.81
53.6855.6355.55
55.53 54.93
0 0.2 0.4 0.6 0.8
Matched sub-answer reward weightEM (left axis) F1 (right axis)Figure 7: DPO reward-weight selection for CMT-RAG with
Qwen3-32B on the MuMu-QA development split. We vary
the coefficientγof matched sub-question F1 in the pair-
ranking reward while keeping the final-answer coefficient
fixed at one. EM is shown on the left axis and F1 on the
right axis; the shaded column indicates the selected setting,
γ= 0.2.
C Inference and Evaluation
C.1 Retrieval Details
Cross-turn dependencies are first resolved using the depen-
dency identifiers stored in each generated trace. The ref-
erenced traces are retrieved from the session DAG through
theirglobalnamespaceidentifiers,andtheirintermediatean-
swers are substituted into the corresponding placeholders to
obtain resolved sub-questions. Each resolved sub-question
thenretrievesitstop-ksupportingparagraphsusingthelocal
DRAGON dense retriever (Lin et al. 2023).
In parallel, the keyword field of each generated trace is
used to perform keyword-overlap lookup over the session
DAG, retrieving at most one additional related trace node
whose similarity score is at least 0.05. The paragraph iden-
tifiers stored in the retrieved trace are used to fetch the as-
sociated supporting paragraphs, which are merged with the
paragraphsreturnedbyDRAGONanddeduplicatedbydoc-
ument identifier to form the reader context.
C.2 Reader Details
Thereaderreceivestwotypesofinputs:(i)dependencyfacts
resolvedfrompreviouslygeneratedtracesand(ii)themerged
supporting paragraphs retrieved for the current turn. It is in-
structedtoanswerstrictlyaccordingtotheretrievedevidence
andtooutputonlytheshortestanswerspan(oryes/nowhen
appropriate), without explanation.
System prompt.The system prompt defines the reader’s
role and output constraints.
1Answer the question using only the
provided paragraphs. Reply with only
the answer span, yes, or no. Do not
explain.
Sub-question prompt.Each resolved sub-question is an-
swered using the following prompt template.
1Provided paragraphs:
2[Paragraph 1 | score={score_1}]
3Title: {title_1}
4{paragraph_text_1}
5
6[Paragraph 2 | score={score_2}]7Title: {title_2}
8{paragraph_text_2}
9
10...
11
12Question: {resolved sub-question}
13
14Answer:
The final user query is answered using an analogous
prompt that additionally includes the complete reasoning
trace from the current turn. Unless otherwise stated, infer-
ence uses a batch size of 8, at most 32 concurrent requests,
andamaximumgenerationlengthof512tokens.Formodels
supporting an explicit thinking mode, we disable it during
both sub-question answering and final-answer generation.
C.3 Stateful SSM Inference
Thetracegeneratorperformsstatefulinferencebypreserving
its recurrent state across dialogue turns, allowing each new
turn to process only the current prompt instead of replaying
the full dialogue history. Inference is performed in BF16
with a maximum recurrent context length of 8,192 tokens
and greedy decoding with up to 512 generated tokens per
turn. To support efficient multi-session serving, we batch
up to 16 concurrent sessions and employ packed recurrent
caches, GPU-resident decoding buffers, and CUDA Graph
executiontoreducememoryoverheadanddecodinglatency.
C.4 Case Study
Figure 8 illustrates an actual evaluation example from
MuMu-QA. At Turn 13, the user asks which UK label was
boughtbythemajorbroadcasterthat,togetherwithABCand
NBC,isbasedinNewYork.Thetracegeneratordecomposes
the query into two structured traces. Trace 28 first identifies
thebroadcasterbyresolvingdependency20whilegenerating
retrieval keywords, and Trace 29 then asks which UK label
was bought by the resolved entity through dependency 28.
During dependency resolution, the predicted trace
(Node 28) resolves dependency 20 and identifies the broad-
caster asCBS. The DAG lookup further recovers an earlier
trace node (Node 2) from Turn 1 using the trace keywords,
reusing its stored supporting paragraphs as complementary
evidence.AfterTrace28iscompleted,theretrieverretrieves
additional paragraphs for Trace 29, and the reader receives
the union of reused and newly retrieved evidence. Based on
the aggregated evidence, the reader correctly identifiesOri-
ole Recordsas the final answer. This example demonstrates
how CMT-RAG combines complementary memory traces:
thepredictedtraceresolvesrecentconversationaldependen-
cies,whilethetraceDAGretrieveslong-rangeevidencethat
would otherwise be unavailable from local dialogue context
alone.
C.5 Evaluation Protocol
We evaluate on the completelong-dialoguedevelop-
mentsplit,containing548dialoguesand5,396turns.Unless
otherwisespecified,evaluationusesgreedytracegeneration,
stateful SSM memory, DRAGON top-5 paragraph retrieval,
and top-1 QR-overlap trace lookup

(a) Dialogue (b) Ground-Truth Trace (c) Retrieved Trace (d) Answer
Turn 1
Q: Which major New York–
based broadcaster is listed
together with ABC and NBC?
A: CBS
(Eleven intervening turns)
Turn 13
Q: What UK label was bought
by the major broadcaster that,
together with ABC and that
organization, is based in New York?id = 28
[SubQ]Which major broadcaster,
together with ABC and [A20],
is based in New York?
[KW]major broadcaster together
ABC [A20] based New York
[DEP][20]
id = 29
[SubQ]Which label was bought
by [A28] in the UK?
[KW]label bought [A28] UK
[DEP][28]Predicted dependency
Node 28 · Turn 13
Which major broadcaster, together with 
ABC and NBC, is based in New York?
A28= CBS
Stored para IDs: [14, 17, 10, 28, 4]
DAG lookup dependency
Node 2 · Turn 1
Which major broadcaster, besides ABC
and NBC, is based in New York?
A2 = CBS
Stored para IDs: [14, 17, 28, 10, 3]
Retrieved para IDs
SubQ29 · DEP [28]
Trace para IDs: [5, 133, 44, 48, 89]Retrieved paragraphs
P14 · New York City
“… broadcast networks are all
headquartered in New York:
ABC, CBS, and NBC.”
P5 · Sony Music
“… CBS established its own
UK distribution with the
acquisition of Oriole Records.”
⋯
Final answer
Oriole Records.
⋯
Trace 
GenerationDependency
ResolutionEvidence
AggregationFigure8:ArealevaluationexamplefromMuMu-QA.Thecurrentdialogueturnisfirstdecomposedintostructuredtraceswith
explicit dependencies and keywords. Dependency resolution combines the predicted dependency with a keyword-recovered
historical trace node from the DAG, enabling evidence reuse across distant turns. The retrieved and reused evidence is then
aggregated and passed to the stateless reader to produce the final answer.
Wereportexactmatch,token-levelF1,supporting-context
coverage, and the average number of unique paragraphs re-
trieved per turn. End-to-end latency includes trace genera-
tion, retrieval, intermediate reader calls, final-answer gener-
ation, while excluding one-time model and retriever initial-
ization.
C.6 Corpus-level RAG Evaluation
We evaluate on three corpus-level RAG benchmarks: Hot-
potQA (Yang et al. 2018), 2WikiMultiHopQA (Ho et al.
2020), and RECOR (Ali et al. 2026). HotpotQA contains
7,405 development questions requiring multi-hop reason-
ing,primarilythroughbridgeandcomparisonrelations,over
evidence distributed across two supporting Wikipedia doc-
uments. 2WikiMultiHopQA contains 12,576 development
questionscoveringbridge-comparison,comparison,compo-
sitional, and inference reasoning patterns over two to four
Wikipedia documents. RECOR is a reasoning-focused con-
versational retrieval benchmark comprising 707 multi-turn
conversations and 2,971 evaluation turns from 11 domains,
whereeachturnrequiresresolvingitsinformationneedfrom
the dialogue history before retrieving supporting evidence.
For all three benchmarks, every method retrieves from
a shared corpus rather than an example-specific context.
For HotpotQA and 2WikiMultiHopQA, we construct the
retrieval corpus by pooling the released contexts across
all examples, resulting in corpora containing 507,494 and
398,354 title documents, respectively. Consequently, our
HotpotQAprotocoldiffersfromtheofficialFullWikisetting.
ForRECOR,wedirectlyusethecompletereleasedcorpusfor
eachdomain.Unlessotherwisespecified,Qwen3-32Bserves
as the answering model. We report EM and token-level F1
ontheHotpotQAand2WikiMultiHopQAdevelopmentsets,
andtoken-levelF1,BLEU-1,andROUGE-LonallRECOR
turns.
# MEGRAG: Multi-Granular Evidence Graphs for Answer-Aware Multi-Hop RAG

**Authors**: Weidong Bao, Yingying Sun, Jun Yang, Yilin Wang, Zili Wei, Yubin Bao, Fangling Leng, Minghe Yu, Tiancheng Zhang, Ge Yu

**Published**: 2026-08-03 13:17:49

**PDF URL**: [https://arxiv.org/pdf/2608.02195v1](https://arxiv.org/pdf/2608.02195v1)

## Abstract
Multi-hop question answering is a fundamental challenge in retrieval-augmented generation (RAG), because deriving an answer requires integrating dispersed evidence. Iterative RAG (iRAG) is widely used for this challenge, but existing methods have two limitations. First, most methods still support each reasoning step with single-granularity evidence, making it difficult to balance information density and contextual noise. Second, existing methods often answer the original question only after aggregating evidence retrieved across intermediate steps, so redundant evidence and intermediate retrieval errors may accumulate and degrade the final answer. To address these limitations, we propose MEGRAG, an answer-aware framework that represents multi-hop reasoning as a path-structured multi-granular evidence graph. Offline, MEGRAG links passages to their sentences and extracted triples through a cross-granularity index. Online, it retrieves passages for the current query and selects aligned evidence, starting with compact triples and adding sentence or passage context as needed. MEGRAG uses the resulting intermediate answer and prior reasoning to decide whether the Initial Query has been resolved. If not, it identifies the missing information and formulates a focused next query; otherwise, it stops retrieval and returns the answer. Extensive experiments demonstrate consistent gains over a diverse set of RAG baselines.

## Full Text


<!-- PDF content starts -->

MEGRAG: Multi-Granular Evidence Graphs for Answer-Aware Multi-Hop RAG
Weidong Bao, Yingying Sun, Jun Yang, Yilin Wang, Zili Wei,
Yubin Bao, Fangling Leng, Minghe Yu, Tiancheng Zhang, Ge Yu
Northeastern University, Shenyang, China
baoweidong293@gmail.com, {sunyingying,yangjun,weizl2}@mails.neu.edu.cn,
wangyilin0409@gmail.com, {baoyubin,lengfangling,yuge}@cse.neu.edu.cn,
{yuminghe,tczhang}@mail.neu.edu.cn
Abstract
Multi-hop question answering is a fundamental challenge in
retrieval-augmented generation (RAG), because deriving an
answerrequiresintegratingdispersedevidence.IterativeRAG
(iRAG) is widely used for this challenge, but existing meth-
ods have two limitations. First, most methods still support
eachreasoningstepwithsingle-granularityevidence,making
itdifficulttobalanceinformationdensityandcontextualnoise.
Second, existing methods often answer the original question
only after aggregating evidence retrieved across intermediate
steps,soredundantevidenceandintermediateretrievalerrors
mayaccumulateanddegradethefinalanswer.Toaddressthese
limitations, we proposeMEGRAG, an answer-aware frame-
workthatrepresentsmulti-hopreasoningasapath-structured
multi-granular evidence graph. Offline, MEGRAG links pas-
sages to their sentences and extracted triples through a cross-
granularityindex.Online,itretrievespassagesforthecurrent
query and selects aligned evidence, starting with compact
triples and adding sentence or passage context as needed.
MEGRAG uses the resulting intermediate answer and prior
reasoning to decide whether the Initial Query has been re-
solved. If not, it identifies the missing information and for-
mulatesafocusednextquery;otherwise,itstopsretrievaland
returns the answer. Extensive experiments demonstrate con-
sistent gains over a diverse set of RAG baselines.
Introduction
Retrieval-Augmented Generation (RAG) performs strongly
on simple knowledge-seeking queries and single-hop ques-
tion answering (Lewis et al. 2020; Lin et al. 2024; Ram
etal.2023).Multi-hopquestionansweringisharderbecause
relevant evidence is often dispersed across multiple sources
andmustbeprogressivelyintegratedthroughreasoning(Fan
etal.2024;Trivedietal.2023;Mallenetal.2023).Standard
single-step retrieval ranks evidence only against the initial
query,oftenrecoveringlocalclueswhilemissingintermedi-
ateevidencewhoserelevanceemergesonlyafterpartialrea-
soning (Shao et al. 2023). Iterative RAG (iRAG) addresses
this limitation by interleaving retrieval with reasoning and
query reformulation, progressively refining the information
need and gathering evidence for subsequent hops (Trivedi
et al. 2023; Asai et al. 2024; Yao et al. 2025).
Recent methods have advanced multi-hop RAG from
three complementary perspectives. Iterative methods im-
proveretrievaladaptivitybyreformulatingqueries,diagnos-
Figure 1: Motivation of MEGRAG. Fixed-granularity ev-
idence can be insufficient or noisy, while answer-after-
retrieval pipelines may accumulate redundant evidence and
errors.MEGRAGbuildsmulti-granularevidenceateachstep
and uses its intermediate answer to continue or stop.
ing knowledge gaps, and progressively organizing retrieved
information(Zhouetal.2024;Chengetal.2025).Structure-
aware methods strengthen connections among dispersed ev-
idence and improve retrieval precision through graphs, hy-
pergraphs, or evidence chains (Gutiérrez et al. 2025; Wang
et al. 2026; Peng et al. 2026). Multi-granular methods bet-
ter balance focused evidence with contextual completeness
by using fine-grained units for passage ranking or adaptive
context expansion (Hu et al. 2026; Wei et al. 2026). Despite
theseadvances,twolimitationsremain,asillustratedinFig-
ure 1. First, most methods still support each reasoning step
with single-granularity evidence, making it difficult to bal-
ance information density and contextual noise: fine-grained
evidencemaylacksufficientcontext,whereascoarse-grained
evidencemayintroduceirrelevantinformation.Second,most
iterative methods answer the Initial Query only after com-
pleting retrieval and aggregating evidence from all interme-
diatesteps.Asaresult,redundantevidenceandintermediate
retrieval errors may accumulate throughout the reasoning
process and degrade the final answer.
Human reasoning is selective and goal-directed: people
seek only enough evidence to support a judgment, then
use intermediate conclusions to identify what remains un-
knownandguidefurtherinquiry(Simon1955;Nelson1990;
Loewenstein 1994). Motivated by this process, we propose
MEGRAG,ananswer-awareframeworkformulti-hopRAG.
arXiv:2608.02195v1  [cs.AI]  3 Aug 2026

Offline,MEGRAGbuildsacross-granularityindexthatlinks
eachpassagetoitssentencesandextractedtriples.Online,it
retrieves passages for the current query and selects aligned
evidence, starting with compact triples and adding sentence
or passage context only as needed. MEGRAG uses the se-
lected evidence to answer the current query, then combines
this intermediate answer with prior reasoning to determine
whether the Initial Query has been resolved. If information
is still missing, it identifies the remaining gap and formu-
lates a focused next query; otherwise, it stops retrieval and
returns the final answer. As this process unfolds, MEGRAG
organizes the queries, selected evidence, and intermediate
answers into a question-specific, path-structured evidence
graph, with explicit transitions that record how each inter-
mediate answer reveals the next information need. Finally,
the policy for constructing this graph is distilled into a
lightweight student.
The main contributions of this paper are summarized as
follows:
•A Flexible Multi-granular Evidence Framework.We
propose MEGRAG, which separates reusable offline ev-
idence organization from question-specific online rea-
soning. Rather than representing the corpus as a knowl-
edgegraph,MEGRAGorganizespassage-,sentence-,and
triple-level views through a cross-granularity index and
constructs a question-specific, path-structured evidence
graph online.
•Sufficiency-guided Multi-granular Evidence Selec-
tion.At each reasoning step, MEGRAG begins with
compact triples and expands to aligned sentences and
passages only as needed, stopping at the first granular-
ity judged sufficient for the current query. This produces
compact evidence without assuming a fixed granularity
across questions or reasoning steps.
•Answer-aware Iterative Reasoning.MEGRAG distin-
guishes answering the current query from resolving the
InitialQuery.Itidentifieswhatremainsmissingandeither
formulates a focused next query or stops retrieval. This
iterative policy is distilled into a lightweight student.
Related Work
IterativeandStructuredRetrievalforMulti-hopReason-
ing.Standard RAG retrieves once from the initial query,
whereasiterativemethodsmakeretrievalresponsivetoevolv-
inginformationneeds(Lewisetal.2020;Trivedietal.2023).
MetaRAGevaluatestentativeanswerstoplantargetedrefine-
ment (Zhou et al. 2024), and DualRAG couples reasoning-
augmented querying with progressive knowledge aggrega-
tion (Cheng et al. 2025). Structure-aware methods instead
improve evidence connectivity: HippoRAG 2 and HGRAG
propagate relevance over corpus structures (Gutiérrez et al.
2025;Wangetal.2026),whileLogicRAG,NeocorRAG,and
QAFD-RAG construct query-centered structures or chains
online(Chenetal.2026;Pengetal.2026;Zhouetal.2026).
MEGRAGorganizesselectedevidenceandintermediatean-
swers into an online path state that explicitly records the
remaining information need.Multi-granular Evidence for RAG.Evidence granular-
ity trades information density for contextual completeness:
passages preserve context but may contain noise, sentences
retainlocalconstraints,andtriplesprovidecompactfactsbut
mayomitqualifiers.MGranRAGusessentence-andphrase-
levelevidencetorecalibratepassageranking(Huetal.2026).
CIRAG iteratively integrates triple-centric evidence and ap-
plies cascaded granularity to the accumulated context for
finalanswergeneration(Weietal.2026).MEGRAGinstead
constructs a multi-granular evidence composition within
each retrieval step, immediately producesb i, and uses the
resulting node state to determineq i+1or stop.
Methodology
Problem Formulation
Given an Initial Queryxand corpusD={d j}N
j=1,
MEGRAGbuildsanonline,question-specificevidencegraph
Gand initializes the current query asq 1=x. After stepi,
the graph is
G(i)= (V(i), E(i)),(1)
whereV(i)contains reasoning nodes andE(i)records tran-
sitions between successive retrieval queries together with
terminal stop edges. Thei-th node is
vi= (q i, Zi, bi, hi, mi, yi, si).(2)
Here,Z iis selected evidence,b iis the evidence-grounded
intermediateanswertothecurrentqueryq i,hiistheresolved
reasoning path, andm iis the remaining information need.
The variabley iis the answer to the Initial Queryxwhen
si= 1, ands i∈ {0,1}is the stop decision. Whens i= 0,
yi=∅,m i̸= none, and the policy generatesq i+1form i.
Foracontinuingnode(s i= 0),thecorrespondingtransition
edge is
econt
i= (v i, ri, vi+1),(3)
wherer idescribes how the next query extends the current
query toward answering the Initial Query. Specifically,r iis
definedjointlyfromtheInitialQueryx,currentqueryq i,and
nextqueryq i+1.Avalidstophass i= 1,m i= none,y i̸=∅,
andq i+1=∅. Nov i+1is constructed; instead, MEGRAG
adds the terminal edge
estop
i= (v i,STOP).(4)
Framework Overview
Figure 2 summarizes the offline and online stages of
MEGRAG. Offline, passages are organized together with
their sentence and triple views. Online, MEGRAG retrieves
passages for the current query, selects evidence at the first
granularity judged sufficient, and produces an intermediate
answer. It continues with a focused query when information
is missing and returns the answer once the Initial Query is
resolved. The iterative policy is distilled into a lightweight
student.

Multi-granular Evidence Space
Passages
Sentences
...TriplesSplit
Triples Extraction
Corpus
......Cross-granularity Evidence Index
...
...
...
� -th Query
Candidate Collection
Candidate evidence 
Retrieval
Initial QueryLothair II (835 –)
was the king of Lothar-
ingia from 855 until   
.......Lothair II was the second 
son of Emperor Lothair I 
and Ermengarde of Tours.
  
.......
: 
Who was Lothair II's 
mother ?
Candidate evidence.......
、Intermediate 
Answer
 Ermengarde 
of                  
ToursWhen did 
Ermengarde of 
Tours die?New Node vi
Initial Query
� -th Query 
Selected Evidence 
�+� -th Query 
 Selected Evidence �+� -th Query
Intermediate 
Answer 
Graph 
Summary 
Initial Query
Initial Query Resolved 
�+� -th Query
 
Initial Query
Intermediate 
Answer
Final Answer
 
� -th Query
Reasoning Node Construction
Graph Update and StoppingFigure 2: Overview of MEGRAG. The offline stage constructs aligned passage, sentence, and triple views. At stepi, the online
system retrieves aligned candidates forq i, starts from compact triples, and adds sentences or passages when more context is
needed to formZ i. It then derives an intermediate answerb iand decides whether it can produce a terminal candidatey ifor
the Initial Query. Otherwise it generatesq i+1. The example distinguishes the Initial Query about a death date from the current
query about the mother’s identity.
Multi-granular Evidence Space
MEGRAGfirstconstructsanofflinemulti-granularevidence
space:
E={EP,ES,ET,I},(5)
whereEP,ES,andETdenotepassage-,sentence-,andtriple-
level evidence views, respectively.Iis a cross-granularity
evidence index that links sentence and triple records to
theirsourcepassages.Passageembeddingssupportdensere-
trieval,afterwhichtheassociatedsentenceandtriplerecords
are collected throughIrather than retrieved independently.
Thethreeviewsprovidecomplementarysupport.Passages
retain broad context for entity disambiguation and cross-
sentencereasoning.Sentencespreservelocalconstraintssuch
as aliases, temporal cues, negation, and comparison. Triples
encodecompactrelationalfactsextractedbyalargelanguage
model. MEGRAG therefore accesses fine-grained evidence
only within the retrieved passage set, without relying on
corpus-level knowledge-graph traversal.
Path-structured Question-specific Evidence Graph
Construction
Giventhe InitialQueryx,MEGRAG collectsevidence can-
didates, constructs a reasoning node, and updates the path
deterministically at each step. With a maximum ofBsteps,
the process continues untilq i+1=∅ori=B.
Candidate collection.At stepi, MEGRAG encodesq iand
ranks all passages by vector similarity. The topN Ppas-
sages formCP
i. Rather than retrieving fine-grained unitsindependentlyoverthefullcorpus,MEGRAGusesItocol-
lectthesentencesandtriplesassociatedwithCP
i.Following
the passage ranking, duplicate units are removed and the
retained sentence and triple candidates formCS
iandCT
i,
bounded byN SandN T. Specifically, MEGRAG traverses
passages in retrieval-rank order, preserves each passage’s
storedsentence andtriple order,removes duplicates,andre-
tains the firstN Sunique sentences andN Tunique triples.
Thus,C i={CP
i, CS
i, CT
i}contains three aligned evidence
views derived from the same retrieved passages.
Reasoning node construction.Before constructingv i,
MEGRAG summarizesG(i−1)asH i. For each prior node
and its outgoing edge,H irecords the current query, se-
lected evidence, intermediate answer, resolved path, miss-
ing information, and goal-conditioned transition relation:
Hi= GraphSummary(G(i−1)). The policy is
oi= (Z i, bi, yi, si, qi+1, hi, mi, ri) =π(x, q i, Hi, Ci).
(6)
The selected evidence is
Zi= (T i, Si, Pi),(7)
whereT i,Si, andP iare the selected triple-, sentence-,
and passage-level evidence aligned throughI. A node may
contain one or more granularities. MEGRAG evaluates ev-
idence in the fine-to-coarse orderCT
i→CT
i∪CS
i→
CT
i∪CS
i∪CP
i.Itfirstdetermineswhetherthetriplesaresuf-
ficientforthecurrentquery.Ifnot,itaddsalignedsentences
and then passages until reaching the first sufficient gran-
ularity. The resulting composition becomesZ i; unselected

Qwen3-8B Qwen3-Max
2WikiMQA HotpotQA MuSiQue 2WikiMQA HotpotQA MuSiQue
Method F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM
Direct Model 28.51 25.00 25.29 17.00 8.35 2.30 40.17 33.80 44.41 32.60 20.13 9.20
NativeRAG 32.10 28.90 52.20 35.20 17.10 10.20 48.30 40.10 67.50 53.20 28.30 18.90
DualRAG 63.10 51.90 59.20 44.90 34.30 26.19 76.20 66.83 74.14 58.80 51.60 37.50
MetaRAG 52.30 45.80 64.33 50.45 32.90 23.60 59.10 53.40 75.80 61.40 44.80 33.60
HippoRAG 2 66.94 56.83 64.83 49.68 38.64 27.39 74.18 64.72 72.09 58.64 53.68 41.76
NeocorRAG 69.08 58.24 67.36 52.21 41.05 29.84 75.84 66.21 75.26 61.47 54.81 43.92
CIRAG 70.50 61.20 68.10 53.50 42.10 30.10 77.30 68.10 75.20 61.10 56.80 45.50
MGranRAG 60.73 52.60 73.79 57.80 46.81 34.20 77.10 69.80 76.10 63.50 52.20 40.60
HGRAG 68.00 60.56 74.05 59.20 46.08 35.80 78.30 70.30 76.40 63.30 53.80 42.20
LogicRAG 70.21 59.62 65.74 50.31 40.36 29.18 77.08 67.86 72.81 59.35 55.47 44.06
QAFD-RAG 66.34 59.00 73.60 59.00 43.96 33.90 66.11 59.10 76.28 60.90 48.32 38.20
MEGRAG 76.34 68.70 76.95 61.90 55.15 44.90 80.22 71.40 79.74 64.70 62.01 51.80
Table1:Mainresultsonthesamefixed1,000-questionsubsetofeachmulti-hopQAbenchmark.Baselinesretaintheiroriginally
specified training regimes: trainable methods are trained as prescribed, while training-free methods remain training-free. The
Qwen3-8B block is therefore a complete-system comparison, with MEGRAG including policy distillation; MEGRAG with
Qwen3-Max is prompt-only. The best result in each column is bold, and the strongest baseline is underlined.
evidence is not carried into subsequent reasoning. Based on
Zi,thepolicyproducesb iandupdatestheresolvedpathh i.If
the accumulated findings do not yet resolvex, it records the
missinginformationinm i,setsy i=∅ands i= 0,generates
qi+1, and predictsr i. Otherwise, it setsm i= none, returns
the answer asy i, setss i= 1, and leavesq i+1empty.
Graphupdateandstopping.MEGRAGfirstappendsv i.If
si= 0,thenextiterationconstructsv i+1fromq i+1andadds
theedgeecont
i= (v i, ri, vi+1).Becauseeachnodeproduces
at most one next query, the graph is a directed path. The
goal-conditioned relationr i, defined fromx,q i, andq i+1,
records how the next retrieval step addresses what remains
unresolved in the Initial Query. At a valid stop, MEGRAG
appendsestop
i= (v i,STOP)and returns
ˆa=y i, P ˆa= Trace(v i, G(i)).(8)
If the step budget is exhausted withs B= 0, MEGRAG
invokes a final resolver using only the ordered selected evi-
dence(Z 1, . . . , Z B), withb Bas a candidate clue. Its output
is returned as the prediction, while the trajectory remains
marked as budget-exhausted. At each step, the trace records
the current query, selected evidence, intermediate answer,
resolvedpath,remaininginformationneed,and,foracontin-
uing node, its outgoing transition relation.
Graph Construction Distillation
At each step, the graph construction policyπselects evi-
dence,producesanintermediateanswer,checkswhetherthe
Initial Query has been resolved, and, when necessary, gen-
erates the next query. MEGRAG distills this behavior into a
lightweight student policy.
For each training question, the teacher runs the full
construction process and produces a trajectoryξ=
{(x, q i, Hi, Ci, oi)}Lξ
i=1. The student is trained to reproducethe complete teacher decisiono iat each step:
LGCD=−X
ξLξX
i=1logp θ(oi|x, q i, Hi, Ci).(9)
Terminal decisions supervises i= 1,y i, andm i= none,
togetherwithq i+1=∅.Thedeterministicgraph-updaterule
is unchanged.
Experiments
We evaluate answer accuracy (RQ1), component contri-
butions (RQ2), adaptive evidence granularity and retrieval
depth(RQ3),andstoppingefficiency(RQ4).Thesupplement
provides extended results, implementation details, an evi-
dence sufficiency audit, failure analysis, and a complete tra-
jectory.AppendixB.7testswhetherselectedevidencealone
can reproduce the answer.
Experimental Setup
Datasets and metrics.We evaluate on 2WikiMulti-
HopQA (Ho et al. 2020), HotpotQA (Yang et al. 2018),
andMuSiQue(Trivedietal.2022),reportingtoken-levelF1
andexactmatch(EM).Underbothbackbones,everymethod
uses the same seed-42 subset of 1,000 questions per bench-
mark,retrievalcorpus,preprocessing,andevaluationscripts.
Questions are sampled without replacement. Unless noted,
analyses use these subsets with Qwen3-8B.
Baselines.Grouping methods by their primary design
emphasis, we compare withDirect ModelandNa-
tiveRAG(Lewisetal.2020);theiterativeoradaptivemethods
MetaRAG(Zhouetal.2024),DualRAG(Chengetal.2025),
andCIRAG(Wei et al. 2026); the structure-aware methods
HippoRAG2(Gutiérrezetal.2025),NeocorRAG(Pengetal.
2026),HGRAG(Wang et al. 2026),LogicRAG(Chen et al.

Stop@1 Stop@2 Stop@3 Stop@4
HotpotQA
MuSiQue
2WikiMQA89.3% 9.1% 1.4% 0.2%
51.6% 33.9% 9.8% 4.7%
49.4% 48.4% 2.2% 0.0%(a) Termination pro file
55 60 65 70 75
Saved (%)HotpotQA
MuSiQue
2WikiMQA
71.9 74.1
58.1 59.5
61.8 64.1(b) Efficiency relative to fixed four-step retrieval
LLM calls T okensΔF1
-1.46
+0.04
-0.99Figure 3: Termination behavior and measured efficiency relative to fixed four-step retrieval. Left: termination-step distribution.
Right: saved LLM calls and tokens, together with the change in answer F1.
2026), andQAFD-RAG(Zhou et al. 2026); and the multi-
granular methodMGranRAG(Hu et al. 2026). Each base-
line follows its published training regime: prescribed train-
ing stages are reproduced, while training-free methods re-
maintraining-free.Therefore,theQwen3-8Bresultscompare
complete systems rather than architectures under matched
supervision. We separately report MEGRAG without SFT
and prompt-only MEGRAG with Qwen3-Max to isolate the
effects of distillation and the inference procedure.
Implementation and Training Details. Backbone.We
use Qwen3-8B-Instruct and qwen3-max-2026-01-23 (Yang
et al. 2025), denoted Qwen3-8B and Qwen3-Max. Within
each setting, all methods share the reasoning and answer
backbone;Qwen3-MaxalsosuppliesofflineOpenIEanddis-
tillation trajectories.
RetrievalSetup.Weusenvidia/NV-Embed-v2(Leeetal.
2025) as the default retriever and BGE-small-en-v1.5 (Xiao
et al. 2024) for the retriever robustness study. Both en-
code queries and passages. Sentence and triple candidates
are obtained by aligned lookup from the retrieved passages,
not by independent embedding retrieval. Iterative methods
retrieve 10 passages per step for at most four steps. For
MEGRAG,N P= 10and the aligned candidate budgets
areN S=N T= 30.
Distillation.Qwen3-Max generates trajectories for 3,000
questionsfromtheofficialtrainingsplits,disjointfromeval-
uation, and Qwen3-8B is fine-tuned with LoRA (Hu et al.
2022)ononeH800GPU.Qwen3-8Bresultsusethedistilled
system,whereasQwen3-Maxappliesthesameinferencepro-
cedureprompt-only.Thew/o-SFTablationmeasurestheef-
fectofdistillation.AppendixAgivestrajectoryfilteringand
optimization details.
Main Results on Multi-hop QA
Table1answersRQ1.MEGRAGhasthehighestobservedF1
and EM in all 12 comparisons. With Qwen3-8B, its F1/EM
marginsoverthestrongestper-datasetbaselineare5.84/7.50,
2.90/2.70,and8.34/9.10on2WikiMultiHopQA,HotpotQA,
and MuSiQue. These end-to-end comparisons use each sys-
tem’s native training regime and do not attribute the en-
tire margin to architecture alone. The largest gains occur on
MuSiQue,whosequestionsgenerallyrequirelongercompo-sitional chains.
Withprompt-onlyQwen3-Max,therespectivemarginsre-
main 1.92/1.10, 3.34/1.20, and 5.21/6.30, showing that the
inferenceprocedureremainseffectivewithouttrajectorydis-
tillation. Table 3 measures the additional effect of distil-
lation within MEGRAG. Paired-bootstrap intervals in Ap-
pendix B.1 confirm both metrics for every Qwen3-8B com-
parison. With Qwen3-Max, all F1 gains and both MuSiQue
gainsaresignificant,whereasEMon2WikiMultiHopQAand
HotpotQAisnot.Wethereforereporttheobservedrankings
without claiming uniform significance.
Goal-conditioned Transition Analysis
To isolate the transition relation from trajectory distillation,
we conduct a focused prompt-only Qwen3-Max ablation on
MuSiQue.
Variant F1 EM∆F1
Full MEGRAG 62.01 51.80–
w/o Transition Relation 60.40 50.20−1.61
Shuffled Transition Relation 59.10 48.60−2.91
Table 2: Effect of goal-conditioned transition relations on
MuSiQue with prompt-only Qwen3-Max. All variants use
thesameretriever,candidatebudgets,reasoning-statefields,
stoppingpolicy,andinferencebudget;onlytheprovidedtran-
sition relation is modified.
w/o Transition Relationremovesr ibut retains all other
state fields;Shuffled Transition Relationreplaces it with
an unrelated relation. The F1/EM losses are 1.61/1.60 and
2.91/3.20, respectively. The larger loss after shuffling sug-
geststhatlaterdecisionsdependonthecontentoftherelation,
notmerelyitspresence.Becauseallvariantsareprompt-only,
thedifferencescannotbeattributedtotrajectoryfine-tuning.
We focus on MuSiQue because its longer compositions di-
rectly test whether transition information helps track what
remains unresolved across hops.
Ablation Study
Table3addressesRQ2.RemovingSFTreducesF1by18.41,
6.33, and 11.57 points, showing its importance for transfer-
ring structured decisions to Qwen3-8B. This does not im-

Method2WikiMQA HotpotQA MuSiQue
F1 EM F1 EM F1 EM
MEGRAG 76.34 68.70 76.95 61.90 55.15 44.90
MEGRAG w/o SFT 57.93 50.60 70.62 55.70 43.58 31.80
Triples only 61.20 55.40 68.33 53.90 43.35 33.00
Sentences only 73.51 64.70 72.38 56.04 48.92 39.05
Passages only 74.89 67.00 73.46 57.32 43.96 34.40
w/o Structured History 74.85 66.80 74.12 58.66 49.34 38.60
Table3:Ablationstudyonmulti-hopQAbenchmarkswithQwen3-8B-Instruct.Thebestresultineachcolumnisshowninbold,
and the strongest ablated variant is underlined.
Figure 4: Evidence composition across question structures.
Each bar reports the fraction of questions using triples only,
triples with sentences, or all three evidence granularities.
ply that MEGRAG itself requires fine-tuning: prompt-only
Qwen3-Max retains the advantage in Table 1.
Nofixedgranularitydominates:passagesarestrongeston
2WikiMultiHopQA and HotpotQA, whereas sentences are
strongestonMuSiQue.MEGRAGexceedstheseper-dataset
best variants by 1.45, 3.49, and 6.23 F1, supporting adap-
tive composition: triples provide compact evidence, while
alignedsentencesandpassagesrestorecontextandqualifiers
whenneeded.Thegainthereforereflectsadaptivegranularity
selection rather than reliance on one evidence view.
w/o Structured Historykeeps the candidates, policy, bud-
get,andstoppingrule,butreplacespriorqueries,intermedi-
ate answers, resolved paths, missing information, and tran-
sitions with accumulated evidence alone. Its F1 drops by
1.49, 2.83, and 5.81, showing that the combined reasoning
history is especially useful on MuSiQue. Because the vari-
antremovesallhistoryfieldstogether,itmeasurestheirjoint
contribution rather than that of any single field.
Adaptive Behavior Across Question Structures
Figures 4 and 5 address RQ3 by comparing evidence com-
position and retrieval depth across question structures.
Figure5:AnswerF1andaverageretrievalstepsacrossques-
tion structures. Bars report F1 and the black line denotes
average retrieval depth.
On MuSiQue, use of all three granularities rises from
19.44% at two hops to 53.06% at four, and average depth
from1.447to2.497;F1neverthelessfallsfrom65.4to37.9,
confirmingthatlongerchainsremainharder.On2WikiMul-
tiHopQA, MEGRAG takes more retrieval steps for bridge-
comparison questions, which remain triple-dominant, while
compositionalquestionsusepassagesmoreoften.HotpotQA
structuresarelargelytriple-dominantandshallow.Evidence
granularity and retrieval depth therefore vary independently
with the information need. Harder questions tend to trigger
richer evidence and more retrieval steps, yet longer chains
remainchallenging.Thecross-datasetcontrastalsoindicates
thathopcountalonedoesnotdeterminegranularity;theform
of the missing evidence matters.

Robustness and Sensitivity
Appendix B.2–B.5 tests retriever, backbone, and budget ro-
bustness.ReplacingNV-Embed-v2withBGE-small-en-v1.5
preservesMEGRAG’shighestobservedF1andEMinallsix
comparisons. Across Llama, DeepSeek, GPT, and Gemini,
italsoranksfirstinall24comparisons,withthelargestmar-
gins on MuSiQue. These results extend the two main Qwen
settings without changing the evaluation protocol.
OnMuSiQue,increasingthestepbudgetfromonetofour
raisesF1from48.50to55.15andreducesbudgetexhaustion
from 48.4% to 2.8%, while average depth grows only from
1.000to1.676.Gainsdiminishafterthreesteps.Thedefault
candidate budget improves F1 by just 1.03–1.24 over the
smallest tested pair, and larger pools add no gain, indicating
that performance is not sharply tuned to the candidate bud-
get. The fourth step therefore accommodates the remaining
difficultchainswithoutforcingfourstepsoneveryquestion.
Answer-aware Stopping and Efficiency
Figure 3 compares answer-aware stopping with fixed four-
stepretrieval;∆F1istheirF1difference.Sinceeachstepan-
swersthecurrentquery,thecontrollercanstopwhentheac-
cumulatedstateresolvestheInitialQueryratherthanretrieve
to a fixed depth. HotpotQA stops after one step for 89.30%
of questions, saving 71.88% of LLM calls and 74.10% of
tokens at−1.46F1. MuSiQue continues more often, sav-
ing 58.10%/59.50% of calls/tokens with∆F1= +0.04.
2WikiMultiHopQA saves 61.80%/64.10% at a 0.99-point
cost. Thus, stopping adapts computation to unresolved in-
formation rather than maximizing early termination. Stop-
ping behavior also varies across datasets, indicating that the
controller responds to different information needs. Figure 6
reports wall-clock latency and F1 on 2WikiMultiHopQA.
MEGRAG achieves the highest F1 (76.34) with an end-to-
end latency of 11.5 seconds. Although slower than several
graph-based baselines, it is faster and more accurate than
CIRAG, MGranRAG, MetaRAG, and DualRAG, yielding
a favorable accuracy–latency trade-off. Appendix B.4 gives
complete termination and efficiency statistics.
Transfer to Single-hop QA
Onsingle-hopNQ(Kwiatkowskietal.2019)andWebQ(Be-
rant et al. 2013), MEGRAG leads the strongest base-
line by 0.39/1.90 and 1.50/2.15 F1/EM, respectively (Ap-
pendixB.6).TheseEnglishfactoiddatasetstestwhetheriter-
ativereasoningdegradesperformancewhennolongretrieval
path is required; MEGRAG remains competitive in this set-
ting.
Case Study
Appendix C traces a three-hop MuSiQue example.
MEGRAG first resolves an intermediate relation with com-
pactevidence,thenexpandstoalignedcontexttoresolvethe
remaining reference and answer the Initial Query. The trace
showshowselectedgranularity,theintermediateanswer,and
the next-query decision interact along one path.
5 10 15 20 25
Latency (s)3040506070802WikiMQA F1 (%)
NativeRAGHippoRAG 2HGRAG
QAFD-RAGNeocorRAG
LogicRAGMEGRAG
CIRAG
MGranRAG
MetaRAGDualRAGFigure 6: End-to-end latency versus F1 on 2WikiMulti-
HopQA with Qwen3-8B-Instruct.
Conclusion
MEGRAG combines indexed multi-granular evidence with
question-specific iterative reasoning. At each step, it selects
evidence at the first granularity judged sufficient, produces
an intermediate answer, and uses the remaining information
needtodecidewhethertocontinueretrieval.Experimentson
three multi-hop QA benchmarks and two backbones show
consistent gains over diverse RAG baselines. Further analy-
sessupportthecontributionsofadaptiveevidenceselection,
reasoning history, transition information, policy distillation,
and answer-aware stopping.
Limitations
MEGRAG cannot recover evidence outside its passage-
retrievalscope,andautomaticallyextractedtriplesmaycon-
tainerrorsoromitqualifiers.Itssinglepathcannotbacktrack
after a wrong intermediate state. The lightweight Qwen3-
8B setting also requires task-specific teacher trajectories,
whereas Qwen3-Max does not. Multi-hop evaluation uses
benchmark-specific candidate corpora, and reported latency
excludes one-time OpenIE, embedding, and indexing costs.
Because each node has at most one successor, MEGRAG
should be understood as a path-based reasoning frame-
work rather than a general graph-search or message-passing
method. Paired bootstrap captures uncertainty within each
fixedsubset,notvariationacrossindependentlysampledsub-
sets. Evaluation beyond English factoid QA and the fixed
triple-first selection order remains future work.
References
Asai,A.;Wu,Z.;Wang,Y.;Sil,A.;andHajishirzi,H.2024.
Self-rag:Learningtoretrieve,generate,andcritiquethrough
self-reflection. InInternational conference on learning rep-
resentations, volume 2024, 9112–9141.
Berant, J.; Chou, A.; Frostig, R.; and Liang, P. 2013. Se-
mantic parsing on freebase from question-answer pairs. In

Proceedingsofthe2013conferenceonempiricalmethodsin
natural language processing, 1533–1544.
Chen, S.; Zhou, C.; Yuan, Z.; Zhang, Q.; Cui, Z.; Chen, H.;
Xiao, Y.; Cao, J.; and Huang, X. 2026. You don’t need pre-
built graphs for rag: Retrieval augmented generation with
adaptive reasoning structures. InProceedings of the AAAI
Conference on Artificial Intelligence, volume 40, 30270–
30278.
Cheng,R.;Liu,J.;Zheng,Y.;Ni,F.;Du,J.;Mao,H.;Zhang,
F.; Wang, B.; and Hao, J. 2025. Dualrag: A dual-process
approach to integrate reasoning and retrieval for multi-hop
questionanswering.InProceedingsofthe63rdAnnualMeet-
ingoftheAssociationforComputationalLinguistics(Volume
1: Long Papers), 31877–31899.
Fan,W.;Ding,Y.;Ning,L.;Wang,S.;Li,H.;Yin,D.;Chua,
T.-S.;andLi,Q.2024.Asurveyonragmeetingllms:Towards
retrieval-augmented large language models. InProceedings
of the 30th ACM SIGKDD conference on knowledge discov-
ery and data mining, 6491–6501.
Gutiérrez, B. J.; Shu, Y.; Qi, W.; Zhou, S.; and Su, Y. 2025.
Fromragtomemory:Non-parametriccontinuallearningfor
large language models.arXiv preprint arXiv:2502.14802.
Ho, X.; Nguyen, A.-K. D.; Sugawara, S.; and Aizawa, A.
2020. Constructing a multi-hop qa dataset for comprehen-
sive evaluation of reasoning steps. InProceedings of the
28th International Conference on Computational Linguis-
tics, 6609–6625.
Hu,E.J.;Shen,Y.;Wallis,P.;Allen-Zhu,Z.;Li,Y.;Wang,S.;
Wang,L.;Chen,W.;etal.2022. Lora:Low-rankadaptation
of large language models.Iclr, 1(2): 3.
Hu, Y.; Liu, T.; Zhou, Z.; Zeng, W.; Tan, Z.; and Zhao, X.
2026. Iterative multi-granular RAG with contextual hierar-
chical graph. InProceedings of the AAAI Conference on
Artificial Intelligence, volume 40, 31095–31103.
Kwiatkowski, T.; Palomaki, J.; Redfield, O.; Collins, M.;
Parikh, A.; Alberti, C.; Epstein, D.; Polosukhin, I.; Devlin,
J.; Lee, K.; et al. 2019. Natural questions: a benchmark for
questionansweringresearch.TransactionsoftheAssociation
for Computational Linguistics, 7: 453–466.
Lee, C.; Roy, R.; Xu, M.; Raiman, J.; Shoeybi, M.; Catan-
zaro,B.;andPing,W.2025.Nv-embed:Improvedtechniques
for training llms as generalist embedding models. InInter-
national Conference on Learning Representations, volume
2025, 79310–79333.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.;
Goyal,N.;Küttler,H.;Lewis,M.;Yih,W.-t.;Rocktäschel,T.;
et al. 2020. Retrieval-augmented generation for knowledge-
intensivenlptasks.Advancesinneuralinformationprocess-
ing systems, 33: 9459–9474.
Lin, V.; Chen, X.; Chen, M.; Shi, W.; Lomeli, M.; James,
R.; Rodriguez, P.; Kahn, J.; Szilvasy, G.; Lewis, M.; et al.
2024. Ra-dit: Retrieval-augmented dual instruction tuning.
InInternational Conference on Learning Representations,
volume 2024, 19138–19162.
Loewenstein,G.1994.Thepsychologyofcuriosity:Areview
and reinterpretation.Psychological bulletin, 116(1): 75.Mallen, A.; Asai, A.; Zhong, V.; Das, R.; Khashabi, D.;
andHajishirzi,H.2023. Whennottotrustlanguagemodels:
Investigatingeffectivenessofparametricandnon-parametric
memories. InProceedingsofthe 61stannualmeetingofthe
association for computational linguistics (volume 1: Long
papers), 9802–9822.
Nelson, T. O. 1990. Metamemory: A theoretical framework
andnewfindings. InPsychologyoflearningandmotivation,
volume 26, 125–173. Elsevier.
Peng, S.; Zheng, Q.; Hao, Z.; Tang, Z.; Li, R.; Huang, Q.;
Huang, J.; Liu, J.; Zhu, Y.; and E, H. 2026. NeocorRAG:
Less Irrelevant Information, More Explicit Evidence, and
More Effective Recall via Evidence Chains. InProceedings
of the ACM Web Conference 2026, 1899–1910.
Ram,O.;Levine,Y.;Dalmedigos,I.;Muhlgay,D.;Shashua,
A.; Leyton-Brown, K.; and Shoham, Y. 2023. In-context
retrieval-augmented language models.Transactions of the
Association for Computational Linguistics, 11: 1316–1331.
Shao, Z.; Gong, Y.; Shen, Y.; Huang, M.; Duan, N.; and
Chen, W. 2023. Enhancing retrieval-augmented large lan-
guage models with iterative retrieval-generation synergy. In
Findings of the Association for Computational Linguistics:
EMNLP 2023, 9248–9274.
Simon, H. A. 1955. A behavioral model of rational choice.
The quarterly journal of economics, 99–118.
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A. 2022. MuSiQue: Multihop Questions via Single-hop
Question Composition.Transactions of the Association for
Computational Linguistics, 10: 539–554.
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A.2023. Interleavingretrievalwithchain-of-thoughtreason-
ing for knowledge-intensive multi-step questions. InPro-
ceedings of the 61st annual meeting of the association for
computational linguistics (volume 1: long papers), 10014–
10037.
Wang, C.; Deng, W.; Guan, W.; Lu, Q.; and Jiang, N. 2026.
Cross-granularity hypergraph retrieval-augmented genera-
tion for multi-hop question answering. InProceedings of
the AAAI Conference on Artificial Intelligence, volume 40,
33368–33376.
Wei, Z.; Yang, X.; Wang, Y.; Wang, Z.; Bao, W.; Feng,
S.; Wang, D.; and Zhang, Y. 2026. CIRAG: Construction-
IntegrationRetrievalandAdaptiveGenerationforMulti-hop
Question Answering.arXiv preprint arXiv:2601.06799.
Xiao, S.; Liu, Z.; Zhang, P.; Muennighoff, N.; Lian, D.; and
Nie,J.-Y.2024.C-pack:Packedresourcesforgeneralchinese
embeddings. InProceedings of the 47th international ACM
SIGIR conference on research and development in informa-
tion retrieval, 641–649.
Yang,A.;Li,A.;Yang,B.;Zhang,B.;Hui,B.;Zheng,B.;Yu,
B.;Gao,C.;Huang,C.;Lv,C.;etal.2025. Qwen3technical
report.arXiv preprint arXiv:2505.09388.
Yang,Z.;Qi,P.;Zhang,S.;Bengio,Y.;Cohen,W.;Salakhut-
dinov, R.; and Manning, C. D. 2018. HotpotQA: A dataset
for diverse, explainable multi-hop question answering. In
Proceedingsofthe2018conferenceonempiricalmethodsin
natural language processing, 2369–2380.

Yao,Z.;Qi,W.;Pan,L.;Cao,S.;Hu,L.;Weichuan,L.;Hou,
L.; and Li, J. 2025. Seakr: Self-aware knowledge retrieval
foradaptiveretrievalaugmentedgeneration. InProceedings
ofthe63rdAnnualMeetingoftheAssociationforComputa-
tional Linguistics (Volume 1: Long Papers), 27022–27043.
Zhou, Y.; Liu, Z.; Jin, J.; Nie, J.-Y.; and Dou, Z. 2024.
Metacognitive retrieval-augmented large language models.
InProceedings of the ACM Web Conference 2024, 1453–
1463.
Zhou, Z.; Tarzanagh, D. A.; Didari, S.; Hu, W.; Gutow, B.;
Verkholyak,O.;Faraki,M.;Hao,H.;Moon,H.;andMin,S.
2026. Query-Aware Flow Diffusion for Graph-Based RAG
withRetrievalGuarantees.arXivpreprintarXiv:2605.18775.
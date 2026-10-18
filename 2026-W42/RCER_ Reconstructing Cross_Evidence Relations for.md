# RCER: Reconstructing Cross-Evidence Relations for Iterative Retrieval-Augmented Generation

**Authors**: Linyao Zheng, Xuhang Shi, Zhifang Mao, Xiuquan Hou, Sai Zhou

**Published**: 2026-10-14 01:47:17

**PDF URL**: [https://arxiv.org/pdf/2610.19398v1](https://arxiv.org/pdf/2610.19398v1)

## Abstract
Iterative retrieval-augmented generation (RAG) accumulates knowledge from retrieved evidence to answer complex questions, but existing methods struggle to reconstruct entity correspondences and inferential dependencies across retrieved passages as new information emerges. We propose RCER, an iterative RAG method that reconstructs these cross-evidence relations directly from accumulated source passages. Inter-round relational evidence acquisition identifies entity correspondences and relational dependencies to drive dynamic problem decomposition and iterative retrieval. Reasoning context reconstruction independently constructs a coherent reasoning context through joint selection of complementary passages and derivation of cross-evidence inferences. Cross-evidence compositional reasoning leverages these inferences to establish entity correspondences and compose relations across selected passages for answer derivation. Experiments show that RCER outperforms the best baselines by 2.83 EM and 3.06 F1 points on average across three multi-hop QA benchmarks and achieves superior performance on narrative and implicit reasoning tasks.

## Full Text


<!-- PDF content starts -->

RCER: Reconstructing Cross-Evidence Relations
for Iterative Retrieval-Augmented Generation
Linyao Zheng1,∗Xuhang Shi2Zhifang Mao2Xiuquan Hou1Sai Zhou1
1Xi’an Jiaotong University
2Beijing University of Posts and Telecommunications
∗Corresponding author:zhengly@stu.xjtu.edu.cn
Abstract
Iterative retrieval-augmented generation (RAG)
accumulates knowledge from retrieved evi-
dence to answer complex questions, but ex-
isting methods struggle to reconstruct entity
correspondences and inferential dependencies
across retrieved passages as new information
emerges. We proposeRCER, an iterative RAG
method that reconstructs these cross-evidence
relations directly from accumulated source pas-
sages.Inter-round relational evidence ac-
quisitionidentifies entity correspondences and
relational dependencies to drive dynamic prob-
lem decomposition and iterative retrieval.Rea-
soning context reconstructionindependently
constructs a coherent reasoning context through
joint selection of complementary passages and
derivation of cross-evidence inferences.Cross-
evidence compositional reasoningleverages
these inferences to establish entity correspon-
dences and compose relations across selected
passages for answer derivation. Experiments
show that RCER outperforms the best base-
lines by 2.83 EM and 3.06 F1 points on aver-
age across three multi-hop QA benchmarks and
achieves superior performance on narrative and
implicit reasoning tasks.
1 Introduction
Iterative retrieval-augmented generation (RAG)
builds on standard RAG (Lewis et al., 2020; Fan
et al., 2024) by interleaving retrieval and reason-
ing to answer complex questions (Trivedi et al.,
2023). Intermediate findings identify new infor-
mation needs and guide subsequent searches. Re-
cent methods organize acquired knowledge into
evolving representations. KAIR maintains an in-
dex of salient keywords and their relations (Wu
et al., 2026), MGranRAG combines evidence mem-
ory with graph-calibrated passage rankings (Hu
et al., 2026), and CIRAG integrates candidate
triples using retrieval history (Wei et al., 2026).
Thisknowledge-state updatingpreserves and prior-itizes acquired information for subsequent retrieval
and answer generation.
Knowledge-state updating represents facts and
relations from retrieved evidence to guide retrieval
and generation. As retrieval progresses, newly
acquired evidence can establish entity correspon-
dences and relational dependencies that change
how earlier facts support compositional reason-
ing. However, accumulating relevant facts does not
ensure that these emerging relations inform sub-
sequent retrieval decisions and answer derivation.
Figure 1 illustrates this issue with an example from
MuSiQue. This raises an open question.How can
iterative RAG turn progressively acquired knowl-
edge into reliable cross-round reasoning?
We proposeRCER, an iterative RAG method
that complementsknowledge-state updatingwith
cross-evidence relational reconstructionto estab-
lish how accumulated facts jointly support reason-
ing for the current question.Inter-Round Rela-
tional Evidence Acquisitionjointly analyzes earlier
and newly acquired passages to establish entity
correspondences and relations. These judgments
specify or revise the next information need, guiding
dynamic problem decomposition and subsequent
retrieval. After acquisition,Reasoning Context Re-
constructionindependently analyzes the final can-
didate pool to select complementary passages and
derive cross-document inferences.Cross-Evidence
Compositional Reasoninguses these inferences
alongside the selected sources to bind entities and
compose relations under the question’s conditions.
The main contributions are as follows.
•We proposeInter-Round Relational Evidence
Acquisition, which jointly analyzes accu-
mulated source passages to update earlier
facts’ inferential roles with new evidence.
Relational judgments specify information
needs through dynamic problem decomposi-
tion and guide retrieval queries. Retrieved
1
arXiv:2610.19398v1  [cs.IR]  14 Oct 2026

Figure 1: Cross-evidence relational reconstruction in a MuSiQue example. Evidence about graduates’ residence
identifies the target city, linking the earlier geological fact to subsequent retrieval and answer derivation.
passages support relational reconstruction in
the next iteration.
•We propose a two-stage approach to reason-
ing context reconstruction for compositional
answer derivation.Reasoning Context Re-
constructionindependently analyzes the final
candidate pool to jointly select complemen-
tary passages and derive cross-evidence infer-
ences.Cross-Evidence Compositional Rea-
soningcombines these sources and inferences
to establish entity correspondences and com-
pose relations across the selected passages
under the original question’s constraints.
•Experiments on five complex QA benchmarks
show that RCER improves average multi-
hop EM and F1 by 2.83 and 3.06 points
over leading baselines.
2 Related Work
2.1 RAG
RAG grounds generation in retrieved passages
(Lewis et al., 2020). BM25 and dense passage
retrieval identify relevant passages through lexical
matching and learned representations, respectively
(Robertson and Zaragoza, 2009; Karpukhin et al.,
2020). Structure-augmented approaches organize
knowledge for retrieval and reasoning. HippoRAG
and HippoRAG 2 exploit graph-based associations
(Gutiérrez et al., 2024, 2025), while GraphRAGand RAPTOR use graph-community and hier-
archical summaries (Edge et al., 2024; Sarthi
et al., 2024). EvLink retrieves bridge passages
through source-grounded evidence links (Anony-
mous, 2026a). OriginRAG preserves demand–
passage provenance during graph retrieval and
grounds supported demands in verbatim source
spans (Anonymous, 2026b). SG-RAG aligns sub-
graphs with query constraints (Xie et al., 2026), and
LogicRAG constructs query-dependent reasoning
structures (Chen et al., 2026a). SetR addresses pas-
sage selection through set-level reasoning, identi-
fying information requirements and selecting com-
plementary passages (Lee et al., 2025b).
2.2 Iterative RAG
Reasoning-driven iterationuses intermediate
findings to guide subsequent searches. IRCoT in-
terleaves chain-of-thought reasoning with retrieval
(Trivedi et al., 2023), while self-ask generates inter-
mediate questions (Press et al., 2023). S2G-RAG
assesses evidence sufficiency and identifies infor-
mation gaps to guide further retrieval (Li et al.,
2026). Related planning approaches, including Re-
WOO, PlanRAG, and STRIDE, use explicit plans
to organize retrieval (Xu et al., 2023; Lee et al.,
2024; Chen et al., 2026b).
Knowledge-state updatingmaintains evolv-
ing representations of acquired information for
subsequent retrieval and reasoning. KAIR up-
2

Figure 2: Overview of RCER.Inter-Round Relational Evidence Acquisitionreconstructs relations among
accumulated retrieved passages to refine information needs and guide further retrieval.Reasoning Context
Reconstructionjointly selects complementary passages and derives cross-evidence inferences from the final
candidate passage pool.Cross-Evidence Compositional Reasoningintegrates these inferences with the selected
passages for answer derivation.
dates a keyword–relation index (Wu et al., 2026).
MGranRAG combines evidence memory with
graph propagation to refine passage rankings (Hu
et al., 2026), while CIRAG integrates candidate
triples conditioned on retrieval history (Wei et al.,
2026). RetroRAG iteratively revises source and
inferential evidence (Xiao et al., 2025). RCER re-
constructs relations across accumulated passages
to drive dynamic problem decomposition and itera-
tive acquisition. It jointly selects complementary
passages and derives cross-evidence inferences, re-
constructing the reasoning context for composi-
tional answer derivation.
3 Method
We propose RCER, an iterative RAG method
that reconstructs cross-evidence relations over
accumulated passages to guide subsequent re-
trieval and answer derivation. Figure 2 illustrates
its three components.
3.1 Inter-Round Relational
Evidence Acquisition
Inter-Round Relational Evidence Acquisition de-
termines subsequent information needs by recon-
structing relations across accumulated passages.
For a question q, initial retrieval from corpus Dq
produces A0. At iteration t, the controller input xtcontains qtogether with the accumulated passages
Atand query history Ht. The latest batch ∆tis
identified within At, whose passages retain acqui-
sition order. An LLM-based retrieval controller
jointly analyzes the passages to determine how
new evidence connects with earlier facts and what
requires further investigation. It generates
(ft, It, Mt, qt+1) =C θ(xt).(1)
Here ftspecifies the relation under investigation,
withItidentifying the passages considered jointly.
The relational analysis Mtexpresses their current
interpretation throughm tentries
Mt= [(r tj, utj, Btj)]j∈[m t].(2)
Each entry pairs a relation rtjwith its current
interpretation utj, including unresolved aspects.
The passage identifiers Btjconnect this interpre-
tation to the retrieved text. Newly established
entity correspondences can make an earlier fact
a premise for the current investigation. Dynamic
problem decomposition translates these judgments
into the next subquestion, specifying the entity
and relation that require further evidence. The
controller expresses this information need as the
retrieval queryq t+1.
LetUtcontain the corpus passages not yet
retrieved. Dense retrieval returns the bt+1passages
3

with the highest cosine similarity to the query
∆t+1= Ret bt+1(qt+1,Ut).(3)
Newly retrieved passages are appended to the
accumulated sequence in retrieval order
At+1= (d 1, . . . , d nt, d′
1, . . . , d′
bt+1).(4)
Here d1, . . . , d ntare the ntaccumulated pas-
sages, and d′
1, . . . , d′
bt+1form the newly retrieved
batch. Appending qt+1to the query history yields
Ht+1. The next iteration reconstructs its relational
analysis from the expanded passage pool and
updated query history. Retrieval is bounded by
a passage budget Band terminates early when the
query-repetition criterion is met. Retrieval settings
are provided in Appendix B.
3.2 Reasoning Context Reconstruction
Reasoning Context Reconstruction organizes
complementary passages with their cross-evidence
inferences into a reasoning context. After retrieval,
an LLM-based passage selector performs a
fresh analysis of the final candidate pool AT=
(d1, . . . , d n). For question q, it identifies the infor-
mation requirements and examines how candidate
passages jointly satisfy them. The selector input is
xs= (q, A T, K), where K≤n specifies the num-
ber of passages to retain. It returns a cross-evidence
reasoning analysiszand selected indicesJ
(z, J) =S θ(xs).(5)
The analysis zestablishes entity correspondences
and relational dependencies across candidate
passages. These connections determine how one
passage complements another in supporting the
required inference. Joint selection uses these
complementary roles to choose Kpassages
that together address the question’s information
requirements. The selector generates zand then
Jwithin one response, conditioning selection on z
pθ(z, J|x s) =p θ(z|x s)pθ(J|z, x s).(6)
For the selected indices J= (i 1, . . . , i K), the
passage sequence is
S= (d i1, . . . , d iK).(7)
The sequence Spreserves acquisition order, and
zdraws on the full candidate pool including
passages outside S. It accompanies the selected
text so that the relations established during
context reconstruction remain available for answer
derivation. See Appendix B for details.3.3 Cross-Evidence Compositional Reasoning
Cross-Evidence Compositional Reasoning applies
reconstructed relations to derive an answer from
the selected passages. The generator receives
the reasoning context
Xq= (S, z).(8)
The selected passages Sprovide factual premises,
andzsupplies the cross-evidence inferences used
to connect them. The generator conditions its
answer on both
pθ(z, J, a|x s) =p θ(z, J|x s)pθ(a|q, S, z).(9)
HereSis determined byA TandJ.
The answer generator Gθuseszto determine
how the premises in Scan be combined. Entity
correspondences identify which facts concern the
same entity, allowing a relation established in one
passage to supply a premise for another. The
generator checks these correspondences against the
selected text and composes the resulting relations
to answer q. During composition, each property
remains associated with the entity described in its
source passage, and the direction of each relation
is preserved. The resulting prediction is
ˆa=G θ(q, S, z).(10)
The following prompt specifies how the generator
applies these inferences.
Cross-Evidence Compositional
Reasoning Prompt
Determine the target relation from the question .
Use the cross-evidence analysis to connect facts
across the selected passages. Check entity
correspondences and relation direction against
the passages before deriving the answer.
Detailed settings are provided in Appendix B.
4 Experiments
We address four research questions.
•RQ1.How does RCER compare with
competitive baselines on complex QA tasks?
•RQ2.How do the three components
contribute to RCER’s overall performance?
•RQ3.How does cross-evidence relational
reconstruction improve iterative RAG?
•RQ4.Is RCER a computationally efficient
and adaptable method?
4

4.1 Experimental Setup
Dataset Questions Passages
HotpotQA 1,000 9,811
2WikiQA 1,000 6,119
MuSiQue 1,000 11,656
NarrativeQA 1,000 74,742
StrategyQA 1,000 9,254
Table 1: Dataset statistics. Passages are indexed
retrieval units. NarrativeQA uses story-local retrieval.
Datasets.Five datasets covering three categories
of complex question answering are used for
evaluation. For multi-hop QA, HotpotQA (Yang
et al., 2018), 2WikiQA (Ho et al., 2020), and
MuSiQue (Trivedi et al., 2022) assess retrieving
complementary evidence across passages and
composing factual relations to answer multi-step
questions. HotpotQA includes bridge and
comparison questions, 2WikiQA emphasizes
explicit relation composition, and MuSiQue
connects two to four reasoning steps. For narrative
QA, NarrativeQA (Ko ˇciský et al., 2018) assesses
locating and integrating evidence about characters
and events distributed across full-length novels
and screenplays. For implicit reasoning QA,
StrategyQA (Geva et al., 2021) assesses identifying
and combining unstated factual premises for yes/no
reasoning. Table 1 summarizes dataset sizes. See
Appendix A for details.
Metrics.Normalized exact match (EM) and
token F1 (Rajpurkar et al., 2016) are reported
for multi-hop and narrative QA. EM measures
exact agreement with a reference answer. F1 is
the harmonic mean of token-level precision and
recall. With multiple references, each prediction
receives its highest score across the reference
answers. StrategyQA uses accuracy, the proportion
of correctly answered yes/no questions.
Baselines.The baselines are grouped into three
categories.LLMs and LLM-based retrievers
include Qwen3.8-27B and NV-Embed-v2 (Lee
et al., 2025a).Structure-enhanced RAGincludes
HippoRAGv2, OriginRAG (Anonymous, 2026b),
LogicRAG, and RankCoT (Wu et al., 2025).
Iterative RAGincludes IRCoT, RetroRAG,
MGranRAG, STRIDE, CIRAG, S2G-RAG, and
KAIR. We use Qwen3.8-27B as the base language
model and NV-Embed-v2 as the embedding model
across methods. RCER and LLM settings areprovided in Appendix B, and baseline settings
in Appendix C.
4.2 Main Results (RQ1)
Table 2 shows that RCER ranks first across all
five datasets. Average multi-hop EM and F1 reach
66.27 and 76.73, surpassing the best baseline for
each metric by 2.83 and 3.06 points, respectively.
Compared with LLMs and LLM-based
retrievers.RCER improves F1 over
NV-Embed-v2 by 20.99 points on 2WikiQA and
18.35 on MuSiQue. On NarrativeQA, it obtains
39.83 F1, compared with 16.77 for question-only
Qwen and 39.41 for NV-Embed-v2. StrategyQA
accuracy improves by 11.60 percentage points
over NV-Embed-v2. Qwen relies on parametric
knowledge, making access to task-specific
evidence valuable. Original-query retrieval
can still miss facts whose retrieval requires an
intermediate entity or premise. RCER identifies
these through evidence-informed decomposition
and retrieves complementary support.
Compared with structure-enhanced RAG.
RCER gains 8.44 average multi-hop F1 points
over OriginRAG. It improves NarrativeQA F1
over LogicRAG by 1.38 points and StrategyQA
accuracy over RankCoT by 2.90 percentage points.
HippoRAGv2 identifies graph-associated passages,
but relevance does not establish how their facts
jointly answer a question. OriginRAG jointly
selects sources and grounds evidence demands
in source spans. However, its demands are
derived before retrieval and remain unchanged
when retrieved evidence reveals intermediate
entities. LogicRAG’s rolling summaries can
omit details needed later. RankCoT refines an
existing passage set without retrieving additional
support. RCER uses reconstructed relations
to refine subsequent information needs, then
selects complementary sources and retains
cross-document inferences for answering.
Compared with iterative RAG.Against
RetroRAG, RCER gains 4.72 F1 points on
2WikiQA and 3.77 on MuSiQue. Its MuSiQue F1
is also 7.89 points higher than MGranRAG. On
NarrativeQA, it improves F1 over RetroRAG by
1.73 points. StrategyQA accuracy reaches 87.40%,
compared with 84.70% for IRCoT. Iterative
retrieval can acquire the necessary facts while
retaining an incorrect relational interpretation.
5

HotpotQA 2WikiQA MuSiQue NarrativeQA StrategyQA
Method EM↑F1↑EM↑F1↑EM↑F1↑EM↑F1↑Acc.↑
Qwen3.8-27B 22.80 32.23 29.60 34.34 7.50 16.39 4.80 16.77 74.90
NV-Embed-v2 63.90 76.15 55.50 61.21 38.10 48.17 21.00 39.41 75.80
HippoRAGv2 66.10 77.97 60.50 67.69 38.00 48.18 18.00 30.86 76.90
OriginRAG 68.00 80.23 62.50 69.79 43.80 54.84 18.60 31.95 78.50
LogicRAG 64.80 77.59 61.60 67.44 40.60 50.97 21.70 38.45 80.80
RankCoT 64.20 76.93 60.70 68.09 40.10 52.97 20.10 37.14 84.50
IRCoT 65.00 78.31 65.80 75.98 39.90 48.24 20.30 36.37 84.70
RetroRAG 67.40 80.77 70.50 77.48 52.30 62.75 22.40 38.10 82.50
MGranRAG 67.40 80.04 72.00 79.27 49.70 58.63 20.30 37.00 80.80
STRIDE 63.10 76.13 68.30 77.14 46.00 57.43 19.60 34.35 83.90
CIRAG 58.70 72.39 70.40 79.28 46.50 57.13 18.40 32.15 81.50
S2G-RAG 67.30 79.74 71.20 78.63 51.80 61.42 21.60 37.76 80.40
KAIR 67.60 79.84 69.10 77.06 49.10 60.06 21.00 35.87 79.30
RCER 68.60 81.46 74.90 82.20 55.30 66.52 23.10 39.83 87.40
Table 2: Main results. Best scores are bold, and second-best scores are underlined.
Answer agreement can reinforce that interpretation.
RCER reconstructs cross-evidence relations during
retrieval and independently analyzes the completed
pool, allowing connections established across
rounds to guide final answer derivation.
4.3 Ablation Study (RQ2)
Table 3 shows the contribution of each component
to RCER’s performance. Ablation settings are
provided in Appendix D.
Inter-Round Relational Evidence Acquisition.
This component uses reconstructed relations to
refine subquestions and acquire complementary
evidence. Removing it reduces F1 by 16.03 points
on 2WikiQA and 7.82 on MuSiQue, with EM
drops of 14.40 and 8.20 points, respectively. The
larger 2WikiQA loss highlights the importance
of using intermediate evidence to determine
which entity or relation to investigate next.
On HotpotQA, F1 drops by only 0.94 points,
where original-query retrieval already covers
most supporting evidence. Adaptive acquisitiontherefore contributes most when relevant facts must
be recovered through successive queries.
Reasoning Context Reconstruction. This
component assembles complementary sources
and cross-evidence inferences into the reasoning
context. Removing it lowers F1 by 11.31
points on MuSiQue and 4.81 on NarrativeQA.
StrategyQA accuracy drops by 7.50 percentage
points. These are the largest declines on these
datasets among the three ablations. Joint selection
retains combinations of supporting facts that
acquisition-order selection may miss. The
accompanying analysis links facts about the same
entities and identifies how they jointly support the
premises needed to answer the question.
Cross-Evidence Compositional Reasoning.
With the selected passages fixed, removing this
component reduces 2WikiQA F1 by 8.47 points
and EM by 8.70 points. MuSiQue F1 falls by
4.19 points, and StrategyQA accuracy drops by
6.20 percentage points. These losses highlight the
HotpotQA 2WikiQA MuSiQue NarrativeQA StrategyQA
Variant EM↑F1↑EM↑F1↑EM↑F1↑EM↑F1↑Acc.↑
Full RCER68.60 81.46 74.90 82.20 55.30 66.52 23.10 39.83 87.40
w/o Inter-Round Relational
Evidence Acquisition67.90 80.53 60.50 66.17 47.10 58.70 21.80 38.14 86.00
w/o Reasoning Context
Reconstruction67.60 79.92 64.90 72.51 45.70 55.21 20.10 35.01 79.90
w/o Cross-Evidence
Compositional Reasoning67.70 80.16 66.20 73.72 52.10 62.33 22.40 38.59 81.20
Table 3: Ablation study of RCER’s components.
6

value of cross-evidence inferences for composing
relations and deriving answers.
4.4 Mechanism Analysis (RQ3)
We trace relational reconstruction through retrieval,
selection, and answer derivation on three multi-hop
datasets. Pool recall is the question-averaged
fraction of annotated supporting passages in the
candidate pool. R@5 measures this fraction in the
final five selected passages.
How do relations guide retrieval? With equal
maximum candidate budgets, Table 4 shows RCER
raising pool recall over single-query retrieval from
80.25 to 99.38 on 2WikiQA and from 78.75 to
87.13 on MuSiQue. Resolving intermediate entities
makes new retrieval targets explicit, enabling
decomposition to recover complementary support
missed by the original query. HotpotQA already
reaches 97.70 with single-query retrieval, leaving
less room to expand evidence coverage.
HotpotQA 2WikiQA MuSiQue
SettingPool
recall↑F1↑Pool
recall↑F1↑Pool
recall↑F1↑
Single-query 97.70 80.53 80.25 66.17 78.75 58.70
Latest 98.25 80.77 98.45 81.34 86.04 64.46
RCER 98.65 81.46 99.38 82.20 87.13 66.52
Table 4: Inter-round relational evidence acquisition.
TheLatestcontrol restricts retrieval decisions to
the newest passages but retains each run’s complete
pool for final selection. Table 5 shows RCER’s F1
advantage growing from 0.96 on two-hop to 6.22
on four-hop questions. Retaining history for final
selection does not replace using it during retrieval.
New passages can resolve which entities earlier
facts concern and how they support the next query.Reconstructing these links revises what remains to
be retrieved. Longer chains create more occasions
for earlier evidence to acquire new roles.
Hops Latest F1↑RCER F1↑∆F1↑
2 71.59 72.55 +0.96
3 64.27 65.95 +1.68
4 42.58 48.80 +6.22
Table 5: Historical evidence access on MuSiQue.
How does relational analysis guide selection?
We use relational analysis to identify passages
that form a connected evidence chain. Table 6
compares joint selection with acquisition order on
fixed candidate pools. Both retain five passages and
use the same answer generator without selection
analysis. Joint selection raises R@5 from 94.03
to 99.18 on 2WikiQA and from 78.50 to 84.91 on
MuSiQue, with a 7.11-point F1 gain on MuSiQue.
On 2WikiQA, the 5.15-point R@5 gain yields only
1.22 F1 points, revealing a further gap between
retaining evidence and composing its facts.
HotpotQA 2WikiQA MuSiQue
Selection R@5↑F1↑R@5↑F1↑R@5↑F1↑
Acquisition order 97.70 79.92 94.03 72.51 78.50 55.21
Joint selection 98.25 80.16 99.18 73.72 84.91 62.33
Table 6: Selection on fixed candidate pools.
Figure 3 compares three passage-selection
strategies across MuSiQue reasoning depths. The
RCER selector analyzes the candidate passages
together before selecting evidence. Direct selection
outputs passage indices without first generating
a selection analysis. Coverage-based selection
2 3 4
Question reasoning depth (hops)2030405060708090Answer F1 (%)
(a) Answer quality
68.77
60.81
42.9265.38
56.05
34.3764.06
58.31
31.00
2 3 4
Question reasoning depth (hops)0255075100Questions with
all support (%)
(b) Complete evidence selection
85.98
56.85
16.4684.09
51.37
12.6670.08
42.47
7.59
RCER selector
 Direct selection
 Coverage-based selection
Figure 3: Selection strategies for Reasoning Context Reconstruction on MuSiQue.
7

labels each passage by the question requirements
it supports, then selects passages using greedy
set cover. RCER improves F1 over direct
selection by 3.38 points on two-hop questions
and 8.55 on four-hop questions. It also retains all
annotated support more often than either control
at every depth. Independently covering question
requirements can overlook dependencies between
the passages that satisfy them. Joint analysis
considers these dependencies when constructing
the evidence set, helping preserve complementary
support as reasoning chains grow.
How do inferences aid composition? With
passages and order fixed, Table 3 shows gains of
8.47 F1 points on 2WikiQA and 4.19 on MuSiQue
from candidate-pool analysis and guidance. On
2WikiQA, 983 of 1,000 questions already retain
all annotated supporting passages. They account
for at least 6.77 points of the overall F1 gain.
The remaining 17 can contribute at most 1.70
points. The gain therefore extends to questions
with complete annotated support. Inferences
align entities and specify how their relations
compose into the target relation, connecting
premises that passage coverage alone leaves
implicit. Figure 4 shows F1 gains of 3.34 to
4.70 points across MuSiQue hop counts. In
Appendix G’s date-comparison case, the analysis
links each film to its director’s death date before
comparing the directors.
2 3 4
Question reasoning depth (hops)2030405060708090Answer F1 (%)
ΔF1 +4.70ΔF1 +3.34ΔF1 +4.2572.55
65.95
48.8067.8562.61
44.55
RCER
 w/o compositional reasoning
Figure 4: Cross-Evidence Compositional Reasoning.
4.5 Efficiency and Applicability (RQ4)
Inference Cost.We measure total LLM tokens
and end-to-end latency on StrategyQA. Its yes/no
format limits answer-length variation while
requiring reasoning over implicit premises. Table 7
shows that RCER achieves the lowest token
consumption and P95 latency, with a mean latency
close to S2G-RAG. Accuracy follows Table 2.MethodMean
tokens↓Mean
latency, s↓P95
latency, s↓Acc.
%↑
IRCoT 17,901.10 15.97 33.32 84.70
RetroRAG 9,710.49 16.38 57.72 82.50
MGranRAG 7,680.42 23.20 40.00 80.80
STRIDE 12,664.00 31.38 60.29 83.90
CIRAG 6,786.84 20.66 43.92 81.50
S2G-RAG 5,542.2312.8531.52 80.40
KAIR 5,977.38 25.83 68.94 79.30
RCER 5,269.6913.0725.49 87.40
Table 7: Token consumption and latency.
Compared with RetroRAG, RCER reduces token
consumption by 45.73% and mean latency by
20.20%. For this efficiency evaluation, RCER
combines evidence selection and subsequent
retrieval decisions within a shared analysis
and stops retrieval when the required premises
are resolved. This reduces repeated evidence
processing and model calls while retaining
cross-evidence reasoning for answer generation.
Adaptability.Table 8 reports mean F1 across the
three multi-hop datasets. RCER leads with 70.65
using Qwen3.5-9B and 76.73 using Qwen3.8-27B.
Its 2.06-point advantage over the best 9B
baseline supports adaptability to smaller LLMs.
Appendix H reports results for all five datasets.
Average F1↑
Method Qwen3.5-9B Qwen3.8-27B
IRCoT 68.45 67.51
RetroRAG 68.59 73.67
MGranRAG 62.88 72.65
STRIDE 63.45 70.23
CIRAG 65.52 69.60
S2G-RAG 65.48 73.26
KAIR 55.01 72.32
RCER 70.65 76.73
Table 8: Adaptability to different base LLMs.
5 Conclusion
RCER reconstructs cross-evidence relations to
guide retrieval, then combines complementary
passages and inferences for compositional an-
swering. Experiments on five QA benchmarks
show improved performance.
8

Limitations
Further work is needed to balance reasoning quality
and computational cost. Different questions may
require different allocations of computation across
retrieval rounds. Learned policies could adapt these
allocations to each question’s information needs.
Ethical Considerations
We use existing QA datasets and their associated
text collections. Any release of narrative source
texts must follow the original copyright and redis-
tribution conditions. Retrieved content and gener-
ated answers can contain factual errors, requiring
validation in downstream applications. We report
inference-token consumption to document compu-
tational resource use. Generative AI was used to
assist with English language polishing.
References
Anonymous. 2026a. EvLink: Source-grounded
evidence linking for Graph RAG. Under review.
Anonymous. 2026b. OriginRAG: Harness-
ing GraphRAG by tracing evidence ori-
gins. Under review.
Shengyuan Chen, Chuang Zhou, Zheng Yuan, Qing-
gang Zhang, Zeyang Cui, Hao Chen, Yilin Xiao,
Jiannong Cao, and Xiao Huang. 2026a. You
don’t need pre-built graphs for RAG: Retrieval
augmented generation with adaptive reasoning
structures.Proceedings of the AAAI Conference on
Artificial Intelligence, 40(36):30270–30278.
Wei Chen, Lili Zhao, Zhi Zheng, Huijun Hou, and
Tong Xu. 2026b. STRIDE: Strategic iterative
decision-making for retrieval-augmented multi-hop
question answering. InProceedings of the 49th
International ACM SIGIR Conference on Research
and Development in Information Retrieval, pages
202–212. Association for Computing Machinery.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Tru-
itt, Dasha Metropolitansky, Robert Osazuwa
Ness, and Jonathan Larson. 2024. From local to
global: A graph RAG approach to query-focused
summarization.Preprint, arXiv:2404.16130.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on RAG meeting LLMs: Towards
retrieval-augmented large language models. In
Proceedings of the 30th ACM SIGKDD Conference
on Knowledge Discovery and Data Mining, pages
6491–6501. Association for Computing Machinery.Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot,
Dan Roth, and Jonathan Berant. 2021. Did Aristotle
use a laptop? A question answering benchmark with
implicit reasoning strategies.Transactions of the As-
sociation for Computational Linguistics, 9:346–361.
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu,
Michihiro Yasunaga, and Yu Su. 2024. HippoRAG:
Neurobiologically inspired long-term memory for
large language models. InAdvances in Neural
Information Processing Systems, volume 37, pages
59532–59569. Curran Associates, Inc.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025. From RAG to
memory: Non-parametric continual learning for
large language models. InProceedings of the 42nd
International Conference on Machine Learning,
volume 267 ofProceedings of Machine Learning
Research, pages 21497–21515. PMLR.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sug-
awara, and Akiko Aizawa. 2020. Constructing a
multi-hop QA dataset for comprehensive evaluation
of reasoning steps. InProceedings of the 28th
International Conference on Computational Lin-
guistics, pages 6609–6625. International Committee
on Computational Linguistics.
Yanli Hu, Teng Liu, Zhuangyi Zhou, Weixin Zeng,
Zhen Tan, and Xiang Zhao. 2026. Iterative
multi-granular RAG with contextual hierarchical
graph.Proceedings of the AAAI Conference on
Artificial Intelligence, 40(37):31095–31103.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen,
and Wen-tau Yih. 2020. Dense passage retrieval for
open-domain question answering. InProceedings
of the 2020 Conference on Empirical Methods in
Natural Language Processing, pages 6769–6781.
Association for Computational Linguistics.
Tomáš Ko ˇciský, Jonathan Schwarz, Phil Blunsom, Chris
Dyer, Karl Moritz Hermann, Gábor Melis, and Ed-
ward Grefenstette. 2018. The NarrativeQA reading
comprehension challenge.Transactions of the As-
sociation for Computational Linguistics, 6:317–328.
Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan
Raiman, Mohammad Shoeybi, Bryan Catanzaro, and
Wei Ping. 2025a. NV-Embed: Improved techniques
for training LLMs as generalist embedding models.
InProceedings of the Thirteenth International
Conference on Learning Representations (ICLR).
Dahyun Lee, Yongrae Jo, Haeju Park, and Moon-
tae Lee. 2025b. Shifting from ranking to set
selection for retrieval augmented generation. In
Proceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 17606–17619.
Myeonghwa Lee, Seonho An, and Min-Soo Kim. 2024.
PlanRAG: A plan-then-retrieval augmented genera-
tion for generative large language models as decision
9

makers. InProceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), pages 6537–6555.
Patrick Lewis, Ethan Perez, Aleksandra Piktus,
Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih,
Tim Rocktäschel, Sebastian Riedel, and Douwe
Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. InAdvances in
Neural Information Processing Systems, volume 33,
pages 9459–9474. Curran Associates, Inc.
Minghan Li, Junjie Zou, Xinxuan Lv, Chao Zhang,
and Guodong Zhou. 2026. S2G-RAG: Struc-
tured sufficiency and gap judging for iterative
retrieval-augmented QA. InProceedings of
the 64th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long
Papers), pages 25846–25862.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A. Smith, and Mike Lewis. 2023. Measuring
and narrowing the compositionality gap in language
models. InFindings of the Association for Computa-
tional Linguistics: EMNLP 2023, pages 5687–5711.
Association for Computational Linguistics.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. SQuAD: 100,000+ questions for
machine comprehension of text. InProceedings of
the 2016 Conference on Empirical Methods in Natu-
ral Language Processing, pages 2383–2392, Austin,
Texas. Association for Computational Linguistics.
Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: BM25 and
beyond.Foundations and Trends in Informa-
tion Retrieval, 3(4):333–389.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D. Man-
ning. 2024. RAPTOR: Recursive abstractive
processing for tree-organized retrieval. InPro-
ceedings of the Twelfth International Conference
on Learning Representations.
Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. 2022. MuSiQue:
Multihop questions via single-hop question com-
position.Transactions of the Association for
Computational Linguistics, 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. 2023. Interleaving
retrieval with chain-of-thought reasoning for
knowledge-intensive multi-step questions. In
Proceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 10014–10037.
Zili Wei, Yilin Wang, Xiaocui Yang, Shi Feng, Weidong
Bao, Daling Wang, Zihan Wang, and Yifei Zhang.
2026. CIRAG: Construction–integration retrievaland adaptive generation for multi-hop question an-
swering. InProceedings of the 64th Annual Meeting
of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 26181–26197.
Mingyan Wu, Zhenghao Liu, Xinze Li, Yuqing
Lan, Yukun Yan, Shuo Wang, Cheng Yang,
Minghe Yu, Zheni Zeng, and Maosong Sun.
2026. Finding what matters: Anchoring context
knowledge with evolving indices for iterative
retrieval.Preprint, arXiv:2601.16462.
Mingyan Wu, Zhenghao Liu, Yukun Yan, Xinze Li, Shi
Yu, Zheni Zeng, Yu Gu, and Ge Yu. 2025. RankCoT:
Refining knowledge for retrieval-augmented gen-
eration through ranking chain-of-thoughts. In
Proceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 12857–12874.
Liang Xiao, Wen Dai, Shuai Chen, Bin Qin, Chongyang
Shi, Haopeng Jing, and Tianyu Guo. 2025. Retrieval-
augmented generation by evidence retroactivity
in LLMs.Preprint, arXiv:2501.05475.
Miao Xie, Xiao Zhang, Yi Li, and Chunli Lv. 2026.
Structure guided retrieval-augmented generation for
factual queries. InProceedings of the 64th Annual
Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 40350–
40370. Association for Computational Linguistics.
Binfeng Xu, Zhiyuan Peng, Bowen Lei, Subhabrata
Mukherjee, Yuchen Liu, and Dongkuan Xu.
2023. ReWOO: Decoupling reasoning from
observations for efficient augmented language
models.Preprint, arXiv:2305.18323.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua
Bengio, William W. Cohen, Ruslan Salakhutdinov,
and Christopher D. Manning. 2018. HotpotQA:
A dataset for diverse, explainable multi-hop
question answering. InProceedings of the 2018
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 2369–2380. Association
for Computational Linguistics.
Yujia Zhou, Zheng Liu, Jiajie Jin, Jian-Yun Nie, and
Zhicheng Dou. 2024. Metacognitive retrieval-
augmented large language models. InProceedings
of the ACM Web Conference 2024, pages 1453–1463.
Association for Computing Machinery.
10

Appendix
A Dataset Settings
Dataset Sampling and retrieval
Multi-hop QA
HotpotQA
2WikiQA
MuSiQueFixed evaluation questions and a shared
passage corpus within each dataset.
Narrative QA
NarrativeQA Sampled without replacement from
10,557 official test questions. Full texts
cover 337 stories. Passages contain 256
words with 32-word overlap. Retrieval is
restricted to the question’s story.
Implicit reasoning QA
StrategyQA First 1,000 of 2,290 publicly labeled
questions under a fixed deterministic
ordering. Retrieval uses 9,251
paragraphs. Split paragraphs retain their
original identifiers for evidence scoring.
Table 9: Dataset settings.
B Implementation Details
Setting Value
Language model Qwen3.8-27B-AWQ-INT4
Embedding model NV-Embed-v2 (Lee et al., 2025a)
Context limit 16,384 tokens
Initial retrieval 4 passages
Additional retrieval Up to 4 rounds of 2 new passages
Candidate limit 12 passages
Selected evidence 5 passages
Query repetition Word-set Jaccard similarity of 0.95
Acquisition
decodingGreedy, up to 1,536 output tokens
Selection
decodingGreedy, up to 2,048 output tokens
Answer
decodingTemperature 0.6, top-p0.95, top-k
20, up to 2,048 output tokens
Table 10: RCER configuration.
RCER uses the same Qwen3.8-27B model with
separate prompts for acquisition, selection, and
answering. Starting from four retrieved passages,
each acquisition call analyzes the accumulated evi-
dence and query history, then proposes a query to
retrieve two new passages for the next call. After
at most four additional rounds or an early stop
for query repetition, the model produces cross-
evidence inferences and five selected passages, then
uses both in a separate answering call.
Experiments were conducted on a server
equipped with eight NVIDIA RTX 3090 GPUs.Table 10 gives the shared models and RCER set-
tings. Dense retrieval uses cosine similarity be-
tween normalized embeddings. Passage inputs
contain the title and original text. Subsequent re-
trieval excludes previously retrieved passage iden-
tifiers. The repetition check compares lowercase
query word sets with all earlier queries, including
the original question.
Each acquisition request includes the original
question, all accumulated passage texts, the source
catalog, the latest passage identifiers, and the query
history. Relational analysis is generated afresh
from these inputs at each round. The final se-
lector independently analyzes the completed can-
didate pool and returns its analysis followed by
five distinct, increasing passage indices through
constrained decoding. The answer generator re-
ceives the five selected passages and this final
analysis, which can also retain relationships from
unselected candidates. All modules use visi-
ble structured analysis with the model’s separate
thinking mode disabled.
B.1 Prompt Instructions
The following boxes summarize the core instruc-
tions used by the three modules.
Inter-Round Relational
Evidence Acquisition
InputsOriginal question, accumulated pas-
sages, source catalog, and query history.
Identify a relationship that still requires inves-
tigation. Read the relevant sources together,
reconsider entity and relation assignments, and
retain unresolved alternatives. Preserve the
original question and its qualifications. Record
compact interpretations with their source iden-
tifiers, then formulate one focused query to
obtain the missing evidence.
Return focus ,read_ids ,memory , and query
in this order. Each memory entry records a
relation, its current interpretation, and refer-
enced source identifiers.
Reasoning Context Reconstruction
InputsOriginal question and indexed passages
from the final candidate pool.
Identify the information requirements for an-
swering the question. For each requirement,
11

Reasoning Context Reconstruction
(continued)
analyze which candidate passages provide the
relevant information. Choose exactly five dis-
tinct passages that together cover the informa-
tion needed for answering.
Return analysis followed by selected . The
analysis records the information requirements
and passage analysis. Write the five selected
identifiers in increasing numerical order.
Cross-Evidence Compositional Reasoning
InputsOriginal question, five selected pas-
sages, and final selection analysis.
Identify the subject, target relation, and re-
quested answer type. Preserve property attribu-
tion, relation direction, entity identity, and the
question’s temporal or comparison conditions.
Use the connections in the analysis to derive
the answer and check them against the source
passages. Resolve disagreements using the
original question and passages. Return only
the requested answer at the appropriate granu-
larity and in the specified format.
Answer scoring.EM and token F1 lowercase
answers, remove English punctuation and articles,
and normalize whitespace. F1 uses multiset to-
ken overlap. Each prediction receives its maxi-
mum score over the available references or aliases.
Scores average over the full question collection,
with execution failures scored as zero. StrategyQA
uses case-insensitive yes/no accuracy. Invalid an-
swers receive zero accuracy.
Evidence recall.Multi-hop R@5 is the mean
fraction of annotated support passages included in
the five selected passages. Pool recall applies the
same measure to all acquired passages. StrategyQA
unions the evidence sets within each annotation and
takes the maximum recall across annotators. Recall
is computed from passage identifiers.
C Baseline Configurations
For open-source baselines, we refer to the official
implementations listed in Table 11. All methods
use the same LLM for QA and the same retriever.Method Configuration
LLMs and LLM-based retrievers
Qwen3.8-27B Question-only input. Greedy decoding
with a 2,048-token output limit.
NV-Embed-v2 One original-query retrieval, followed by
answering from five passages.
Structure-enhanced RAG
HippoRAGv2 github.com/OSU-NLP-Group/
HippoRAG
OriginRAG Question-derived evidence demands and
graph retrieval. Jointly selects five
passages from up to 24 candidates, then
grounds demands in source spans for
answering.
LogicRAG github.com/chensyCN/LogicRAG
RankCoT github.com/NEUIR/RankCoT
Iterative RAG
IRCoT github.com/StonyBrookNLP/ircot
RetroRAG Paper reimplementation (Xiao et al.,
2025) with MetaRAG monitoring
prompts (Zhou et al., 2024).
MGranRAG github.com/MoMoLT/MGranRAG
STRIDE github.com/MINE-USTC/STRIDE
CIRAG github.com/52566rz/CIRAG
S2G-RAG github.com/nianaaa/S2G-RAG
KAIR github.com/NEUIR/KAIR
Table 11: Baseline configurations.
D Module Removal Protocols
All ablations use the same 1,000 questions per
dataset, corpora, embeddings, and language model
as full RCER. Answering retains the question-
target instructions, task instructions, output format,
and decoding settings.
Without Inter-Round Relational Evidence Ac-
quisition.Retrieve twelve passages once with
the original question. Apply the full context recon-
struction and compositional reasoning modules to
this pool, selecting five passages and generating a
new selection analysis.
Without Reasoning Context Reconstruction.
Fix the full acquisition trajectory and candidate
order. Replace the selector with the first five ac-
quired passages. Answer from these passages using
source-only instructions, removing both joint pas-
sage selection and its analysis.
Without Cross-Evidence Compositional Reason-
ing.Fix the full acquisition trajectory, selected
passage identities, and passage order. Remove the
selection analysis and the instructions for using it.
Answering uses the same source-only instructions
as the preceding ablation.
12

E StrategyQA Task Adaptation
StrategyQA configuration selection uses observed
accuracy on the full 1,000-question evaluation col-
lection. Each reported score comes from one com-
plete run. Task-aware RCER instructions were de-
veloped using observations on 123 questions from
this collection. Table 12 reports these questions
and the remaining 877 separately.
Subset Questions General Task-aware
Development 123 82.93 90.24
Remaining 877 81.19 87.00
All questions 1,000 81.40 87.40
Table 12: StrategyQA task adaptation.
Task-aware instructions investigate implicit
premises during acquisition and assess sup-
porting and contradicting evidence during
selection and answering.
F Additional Analyses
F.1 History Access within Relational
Acquisition
Latest preserves the first acquisition request. Later
requests expose only the newest passage bodies,
together with the original question, query history,
and source-title catalog. Both Latest and full RCER
generate a fresh analysis each round. Each trajec-
tory is followed by the full context reconstruction
and compositional reasoning modules.
F.2 Fixed-Evidence Answering
DatasetAnalysis
reuse+ Target-relation
check+ Answer
grammar
HotpotQA 81.24 81.47 81.46
2WikiQA 82.11 82.18 82.20
MuSiQue 65.51 66.65 66.52
Table 13: Answer F1 with fixed evidence.
Table 13 compares generation instructions with
the acquired passages, selected sources, and se-
lection analysis fixed. The target-relation check
raises MuSiQue F1 from 65.51 to 66.65. Adding
the answer grammar yields 66.52 and defines
the final configuration.
F.3 Passage Selection
Table 14 compares three selectors on 489 fixed
MuSiQue candidate pools using passage-only an-Selection F1 R@5
Direct selection 57.59 83.57
Coverage-based selection 57.00 76.98
RCER selector62.22 85.60
Table 14: Passage selection on MuSiQue.
swering. RCER improves F1 by 4.63 points over
direct selection. Coverage-based selection identi-
fies question requirements, labels each candidate
separately, and applies greedy set cover. Figure 3
shows the comparison by reasoning depth.
Selector F1 R@5 Valid sets Output tokens
SetR 60.54 80.83 49/60 1,307.3
RCER 70.61 96.81 60/60 849.8
Table 15: Fixed-budget selector comparison.
Table 15 compares an inference adaptation of
SetR (Lee et al., 2025b) with our selector on 60
fixed pools, 20 from each multi-hop dataset. Both
use the shared frozen Qwen model, the same can-
didate order, five selected passages, a 2,048-token
output limit, and passage-only answering. SetR ex-
hausts the output budget on 11 questions. Reported
means include all 60 questions.
F.4 Inference Use during Development
DatasetPassages
only+ Selection
analysis∆F1
HotpotQA 79.52 81.24 +1.72
2WikiQA 73.96 82.11 +8.15
MuSiQue 61.77 65.51 +3.74
Table 16: Inference use during development.
Table 16 compares passage-only answering and
selection-analysis use with fixed selected passages
and the initial answer prompt. These runs precede
the target-relation and answer-format changes in
Table 13. Table 3 reports the corresponding re-
moval under the final configuration.
G Mechanism Case Studies
Houston geology MuSiQue
Inter-Round Relational Evidence Acquisition
The supporting passages describe Rocky
Mountains erosion and salt dome formation.
Both are available before the Houston-specific
13

Method HotpotQA 2WikiQA MuSiQue NarrativeQA StrategyQA
F1↑F1↑F1↑F1↑Acc.↑
IRCoT 77.22 73.77 54.37 34.32 78.70
RetroRAG 77.36 73.75 54.66 31.60 78.32
MGranRAG 76.25 64.17 48.22 30.30 75.50
STRIDE 70.62 73.44 46.30 28.01 78.00
CIRAG 71.53 74.23 50.79 27.6680.48
S2G-RAG 75.12 67.06 54.26 32.85 76.77
KAIR 66.85 58.55 39.64 27.45 65.20
RCER 79.53 75.50 56.92 34.4480.40
Table 17: Results with Qwen3.5-9B.
Houston geology MuSiQue
(continued)
query. RCER identifies erosion as the process
answering the original question. STRIDE in-
stead asks about changes to the city’s founda-
tions. RetroRAG stops after confirming agree-
ment on salt dome formation.
Method Answer
RCER,
MGranRAG,
IRCoTErosion of the Rocky
Mountains
STRIDE,
RetroRAGSalt dome formation
Cadore 33MuSiQue
Reasoning Context Reconstruction
The album’s performer is Sergio Dalma. An-
other passage describes his Eurovision partici-
pation. Joint analysis connects the album and
competition through the same performer. Inde-
pendent requirement labeling assigns the per-
former requirement to Sergio Furlan.
Selection Answer
RCER selectorEurovision Song
Contest 1991
Coverage-based
selection1964 Summer OlympicsDirectors’ death dates 2WikiQA
Cross-Evidence Compositional Reasoning
Melville W. Brown directedLost in the Strato-
sphereand died on January 31, 1938. John
Ince directedBlind Man’s Eyesand died on
April 10, 1947. Both answer conditions re-
ceive the same five passages. The analysis
associates each death date with its film before
comparing the directors.
Condition Answer
Passages onlyLost in the Stratosphere
RCERBlind Man’s Eyes
Directors’ birth dates 2WikiQA
Cross-Evidence Compositional Reasoning
The selected passages provide Subhash Ghai’s
and Hans Herwig’s birth dates. A candidate
outside the selected five supplies the direc-
tor relationship forLa Fille du Torrent. The
analysis retains this link for comparison with
Meri Jung. Both answer conditions use the
same selected passages.
Condition Answer
Passages onlyMeri Jung
RCERLa Fille du Torrent
H Adaptability to Different Base LLMs
Table 17 reports results with Qwen3.5-9B. The first
four datasets use F1, and StrategyQA uses accuracy.
All scores are percentages.
14
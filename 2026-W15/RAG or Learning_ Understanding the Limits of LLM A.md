# RAG or Learning? Understanding the Limits of LLM Adaptation under Continuous Knowledge Drift in the Real World

**Authors**: Hanbing Liu, Lang Cao, Yang Li

**Published**: 2026-04-06 18:50:18

**PDF URL**: [https://arxiv.org/pdf/2604.05096v1](https://arxiv.org/pdf/2604.05096v1)

## Abstract
Large language models (LLMs) acquire most of their knowledge during pretraining, which ties them to a fixed snapshot of the world and makes adaptation to continuously evolving knowledge challenging. As facts, entities, and events change over time, models may experience continuous knowledge drift, resulting not only in outdated predictions but also in temporally inconsistent reasoning. Although existing approaches, such as continual finetuning, knowledge editing, and retrieval-augmented generation (RAG), aim to update or supplement model knowledge, they are rarely evaluated in settings that reflect chronological, evolving, and real-world knowledge evolution. In this work, we introduce a new benchmark of real-world dynamic events, constructed from time-stamped evidence that captures how knowledge evolves over time, which enables systematic evaluation of model adaptation under continuous knowledge drift. The benchmark reveals that most existing methods, including vanilla RAG and several learning-based approaches, struggle under this setting, exposing critical limitations such as catastrophic forgetting and temporal inconsistency. To mitigate these limitations, we propose a time-aware retrieval baseline, Chronos, which progressively organizes retrieved evidence into an Event Evolution Graph to enable more temporally consistent understanding in LLMs without additional training. Overall, this work provides a foundation for analyzing and advancing LLM adaptation to continuous knowledge drift in realistic settings.

## Full Text


<!-- PDF content starts -->

RAG or Learning? Understanding the Limits of LLM Adaptation under
Continuous Knowledge Drift in the Real World
Hanbing Liu1*Lang Cao2*Yang Li1
1Tsinghua University2University of Illinois Urbana-Champaign
liuhb24@mails.tsinghua.edu.cn yangli@sz.tsinghua.edu.cn
Abstract
Large language models (LLMs) acquire most of
their knowledge during pretraining, which ties
them to a fixed snapshot of the world and makes
adaptation to continuously evolving knowledge
challenging. As facts, entities, and events
change over time, models may experiencecon-
tinuous knowledge drift, resulting not only in
outdated predictions but also in temporally in-
consistent reasoning. Although existing ap-
proaches, such as continual finetuning, knowl-
edge editing, and retrieval-augmented genera-
tion (RAG), aim to update or supplement model
knowledge, they are rarely evaluated in settings
that reflect chronological, evolving, and real-
world knowledge evolution. In this work, we
introduce a new benchmark of real-world dy-
namic events, constructed from time-stamped
evidence that captures how knowledge evolves
over time, which enables systematic evaluation
of model adaptation under continuous knowl-
edge drift. The benchmark reveals that most ex-
isting methods, including vanilla RAG and sev-
eral learning-based approaches, struggle under
this setting, exposing critical limitations such as
catastrophic forgetting and temporal inconsis-
tency. To mitigate these limitations, we propose
a time-aware retrieval baseline,Chronos, which
progressively organizes retrieved evidence into
an Event Evolution Graph to enable more tem-
porally consistent understanding in LLMs with-
out additional training. Overall, this work pro-
vides a foundation for analyzing and advancing
LLM adaptation to continuous knowledge drift
in realistic settings.
1 Introduction
Large language models (LLMs) have shown strong
performance across a wide range of tasks, includ-
ing question answering and reasoning, by acquiring
*Equal contribution. Author order may be swapped.1
Affiliated with the Shenzhen Key Laboratory of Ubiquitous
Data Enabling, Tsinghua Shenzhen International Graduate
School, Tsinghua University.
ContinuousKnowledgeDrift
FixedKnowledgeCutoff (e.g. 2023)
CEO: Person XCEO: Person YCEO: Person YNew Strategy ZT1:2023T2: 2024T3: 2025
Outdated Knowledge &Temporal Inconsistency
Isit X or Y?Why Z?Ours: Adapting LLM to Continuous Knowledge DriftDynamic Event Benchmark
Doc T1Jan 24Doc T2Jul 24Doc T3May 25Chronological, Evolving, and Real-world Dataset
Chronos: A Time-aware retrieval baseline
Time-aware Retrieval
Eventat T1Eventat T2Eventat T3Event Evolution GraphLLM
LLM
At T1 it was X,at T2 it was Y, currently at T3, Y is implementing Z Up-to-date Knowledge &TemporalConsistency
Figure 1: Illustration of continuous knowledge drift and
our proposed method. Real-world knowledge evolves
over time (e.g., changes in company CEO), while LLMs
with a fixed knowledge cutoff may produce outdated
or temporally inconsistent answers. We introduce a
dynamic event benchmark composed of time-stamped
documents that reflect such evolution, and a time-aware
retrieval baseline (Chronos) that progressively organizes
retrieved evidence into an Event Evolution Graph, en-
abling more temporally consistent reasoning.
extensive linguistic and factual knowledge during
pretraining (Achiam et al., 2023; He et al., 2024;
Sun et al., 2025; Liu et al., 2025a). However, this
knowledge is inherently limited to a fixed snapshot
of the world, commonly referred to as a knowl-
edge cutoff (Chang et al., 2024; Thede et al., 2025).
In contrast, real-world knowledge evolves contin-
uously: organizations reorganize, officials change
roles, policies are revised, and entity attributes are
updated over time (Wang et al., 2025). As a result,
deployed LLMs may gradually become misaligned
with the current state of the world, a phenomenon
we refer to ascontinuous knowledge drift.
Beyond outdated facts, continuous knowledge
drift also poses challenges for temporal reasoning
(Chu et al., 2024; Wang and Zhao, 2024; Qiu et al.,
2024). As knowledge evolves, correct reasoning
requires models to not only access up-to-date infor-arXiv:2604.05096v1  [cs.CL]  6 Apr 2026

mation, but also preserve past knowledge and main-
tain temporal consistency in how events and facts
relate and change over time. This involves captur-
ing their ordering, dependencies, and state transi-
tions. Without such temporal consistency, models
may produce inconsistent or logically incorrect out-
puts even when relevant evidence is available (Guo
et al., 2025). These challenges naturally motivate
the central question of this work:
How can LLMs adapt to continuously evolving
real-world knowledge while maintaining
temporally consistent understanding?
A natural approach to addressing knowledge
drift is continual pretraining or finetuning (Shi
et al., 2025; Zheng et al., 2025a,b). While effective
in some settings, such methods are computationally
expensive, require repeated retraining as knowl-
edge evolves, and are prone to catastrophic forget-
ting (Ke et al., 2023). Knowledge editing offers
a more targeted alternative by modifying a small
subset of model parameters to update specific facts
(Meng et al., 2022a,b; Mitchell et al., 2021; Li et al.,
2024b; Wang et al., 2024). However, existing edit-
ing methods are primarily designed to overwrite
a model’s current beliefs, without explicitly pre-
serving past knowledge. Moreover, they typically
operate on one fact at a time, whereas real-world
knowledge updates often arrive in batches. When
applied repeatedly, individual edits may interfere
with each other, and models lack a principled mech-
anism to distinguish past from current information.
Retrieval-augmented generation (RAG) provides
a complementary paradigm by incorporating exter-
nal knowledge at inference time without modifying
model parameters, and has shown strong perfor-
mance on a wide range of knowledge-intensive
tasks (Fan et al., 2024; Yu et al., 2024). However,
existing RAG approaches are rarely evaluated in
settings that reflect chronological, evolving, and
real-world knowledge evolution. Most evaluations
focus on isolated or static updates, and seldom re-
quire reasoning over sequences of evolving events.
As a result, it remains unclear whether retrieval-
based methods can support temporally consistent
understanding under continuous real-world knowl-
edge drift. Additional discussion of related work is
provided in Appendix B .
Addressing this gap requires evaluation settings
that reflect how knowledge evolves over time. How-ever, existing benchmarks are largely static or cen-
tered on single-fact updates, and seldom require
handling ordered event sequences or enforcing tem-
poral consistency (Thede et al., 2025; Levy et al.,
2017; Jang et al., 2022). In addition, many com-
monly used benchmarks rely on knowledge that is
already outdated by one or more years. To bridge
this gap, we introduce a new benchmark of real-
world dynamic events from 2024 to 2025, span-
ning ten domains. The benchmark collects time-
stamped evidence describing both newly emerg-
ing and historically changing facts, thereby reflect-
ing real-world continuous knowledge drift and en-
abling the study of how models adapt to evolving
knowledge over time.
Building on this benchmark, we further intro-
duceChronos, a time-aware retrieval framework
designed for continuously evolving knowledge, as
illustrated in Figure 1.Chronosincorporates tem-
poral signals into retrieval and progressively orga-
nizes evidence through anEvent Evolution Graph,
which explicitly encodes how entities and events
change over time. By structuring evidence in this
way,Chronosenables more temporally consistent
understanding without modifying model parame-
ters.
Overall, our contributions are threefold:
1.Problem formulation.We formalize the prob-
lem of adapting LLMs to continuously evolv-
ing real-world knowledge, highlighting the chal-
lenge of maintaining temporally consistent un-
derstanding as knowledge changes over time.
2.A new benchmark.We introduce a new bench-
mark of real-world dynamic events. The bench-
mark consists of time-stamped evidence reflect-
ing how knowledge changes in practice, and
enables systematic evaluation of models under
continuous knowledge drift.
3.A strong baseline.We proposeChronos, a
retrieval-based framework that constructs an
Event Evolution Graph from retrieved evidence
to model how entities and events evolve over
time, which enables more temporally consistent
understanding, and serves as a strong baseline
for the proposed benchmark.
2 Benchmark Design
Our benchmark consists of two main components:
a collection of knowledge quadruples and five cat-
egories of question–answer pairs. These question
types define distinct evaluation tasks that assess

models from multiple perspectives, evaluating both
their ability to access up-to-date factual knowledge
and to maintain temporal consistency when reason-
ing under continuous knowledge drift.
2.1 Knowledge Quadruple Curation
We first define the temporal scope of all knowledge
to span fromJanuary 1, 2024toOctober 31, 2025.
Accordingly, all evaluated LLMs are assumed to
have a knowledge cutoff prior to 2024, ensuring
that the benchmark tests their ability to handle post-
cutoff information.
We manually define ten topical domains, includ-
ingCorporate Leadership Changes,Sports Coach-
ing Changes and Player Transfers, andEconomic
Indicators and Policy Changes, among others. En-
tities within these domains are highly dynamic,
undergoing frequent updates and continuously giv-
ing rise to new events, making them particularly
suitable for evaluating a model’s ability to track
and reason over evolving knowledge across time.
For each topical domain, we first identify a set
of core entities whose attributes are prone to fre-
quent change over time, and treat each as a subject.
We then track the evolution of these entities using
Wikipedia data from 2024 to 2025, recording all ob-
servable state changes and newly emerging events
together with their corresponding timestamps.
Each change event is extracted using large language
models and represented as a knowledge quadruple
of the form (subject,relation,object,timestamp) ,
which explicitly captures the temporal dynamics of
real-world knowledge.
2.2 Question-Answer Pair Construction
To evaluate whether models can retain historical
knowledge, and understand up-to-date knowledge
and correctly reason over temporally evolving in-
formation, we design five types of QA tasks that
can be grouped into three major categories.
1.Historical QA.This category evaluates whether
a model can correctly recall historical facts that
occurred before 2024. Under our setting, all
information required to answer these questions
originates from events prior to 2024.
2.Contemporary QA.This category evaluates
whether a model can correctly answer questions
involving up-to-date knowledge occurring be-
tween2024-01-01and2025-10-31, which we
treat as contemporary knowledge. It is furtherTable 1: An illustration of the dataset structure. Each
instance includes time-stamped knowledge quadruples
together with questions covering historical, contempo-
rary, and commonsense settings, designed to evaluate
reasoning over evolving knowledge.
Knowledge
⟨World’s Richest Person, held by, Elon Musk, 2024-01-01⟩
⟨World’s Richest Person, held by, Bernard Arnault, 2024-01-28⟩
⟨World’s Richest Person, held by, Jeff Bezos, 2024-03-05⟩
⟨World’s Richest Person, held by, Bernard Arnault, 2024-04-02⟩
⟨World’s Richest Person, held by, Elon Musk, 2024-06-08⟩
⟨World’s Richest Person, held by, Larry Ellison, 2025-09-10⟩
⟨World’s Richest Person, held by, Elon Musk, 2025-09-11⟩
⟨Oracle stock price, surged to, USD 328, 2025-09-10⟩
Historical Question
Who was the richest person in the world in March 2014?
Contemporary Questions
C1: Who was the world’s richest person on August 20, 2025?
C2: Who were the world’s richest people at any point during 2024?
C3: Which company’s stock surge led to Elon Musk losing his position as
the world’s richest person on September 10, 2025?
Commonsense Question
What is the brightest star in the night sky?
A. Sirius B. Sun C. Polaris D. North Star
Answers
Historical: Bill GatesCommonsense: AC1: Elon Musk
C2: Elon Musk, Bernard Arnault, Jeff BezosC3: Oracle
divided into three subtypes that require different
forms of temporal reasoning:
(a)Single-timestamp QA (C1).Each ques-
tion can be answered using a single knowl-
edge quadruple associated with one specific
timestamp.
(b)Multi-timestamp QA (C2).Each ques-
tion requires reasoning over multiple tem-
porally distinct facts associated with the
same entity.
(c)Multi-source QA (C3).Each question re-
quires integrating information from multi-
ple entities and multiple timestamps, test-
ing the model’s ability to combine evidence
across sources while preserving temporal
consistency.
3.Commonsense QA.This category evaluates
time-invariant commonsense reasoning ability
and is not directly related to the other four ques-
tion types or the knowledge base used in our
benchmark. It serves to verify that incorporat-
ing temporally evolving knowledge does not
degrade a model’s general-purpose reasoning
performance.
For QA construction, we primarily rely on large

Raw Query(with Time)
EntityAEntityB
TimeWindowSRO
Time-agnostic QueryTemporalViewEntityViewA2.HistoryReconstructionTimeline
QueryAnalysis
EntitySet
1.Event Evolution GraphConstructionEEEE
EEEEE3.EventAugmentationEEEEEEEG-FEEG-AEHistoricalContemporary
HistoricalContemporary
FinalResponse
Timeline
TimelineEventFlowwithSameEntity
SROSROSRO
EntityViewB
ContemporaryEEG-I
Retrieved KnowledgeQuads
ParallelTime-awareRetrieval
Up-to-date KnowledgebaseOutdated LLM
EEvent NodeEvent=(Subject, Relation, Object, Time)Multi-PerspectiveViews
QueryTimeWindowFigure 2: Overview ofChronos. The framework first performs query analysis to extract relevant entities and the
associated time window. A time-aware retriever then collects temporally relevant knowledge quadruples from an
up-to-date knowledge base. These facts are organized into an Event Evolution Graph (EEG), which models how
entity states evolve over time by sorting events along a timeline and linking events that share common entities. The
EEG is progressively refined, and its resulting multi-perspective views enable the model to reason over evolving
knowledge and produce temporally consistent responses.
language models to generate candidate questions.
ForHistorical QA, questions are generated such
that the relevant events occur strictly before 2024.
ForC1, each question is derived directly from a sin-
gle knowledge quadruple. ForC2, multiple quadru-
ples associated with the same entity are merged to
construct questions requiring temporal aggregation.
ForC3, quadruples from different but semantically
related entities are combined, ensuring that cor-
rect answers require multi-source reasoning across
both entities and timestamps. An illustration of the
dataset structure is provided in Table 1 .
All generated questions undergo manual verifi-
cation to ensure clarity, correctness, and alignment
with the intended evaluation setting. ForCommon-
sense QA, we directly adopt the TruthfulQA dataset
(Lin et al., 2022) as the evaluation data.
This construction process ensures that the bench-
mark captures realistic knowledge evolution while
maintaining high data quality and enabling com-
prehensive and systematic evaluation of model ro-
bustness under temporal change. Dataset statistics
are provided in Appendix C .
3 Methodology
We proposeChronos, a time-aware retrieval-
augmented framework that serves as a strong base-
line for the proposed benchmark. Its overall archi-
tecture is illustrated in Figure 2 .3.1 Task Formulation
Given an input query qraw, which may contain
explicit or implicit temporal constraints, we as-
sume access to an up-to-date knowledge base K
consisting of a set of time-stamped quadruples
{ki}, where each quadruple is defined as ki=
(si, ri, oi, ti). Here, sidenotes the subject, rithe
relation, oithe object, and tithe timestamp asso-
ciated with the fact. All timestamps fall within a
fixed time window of newly acquired knowledge,
T0≤ti≤T 1.
We further assume an outdated large language
model Mwhose pretraining data only covers in-
formation available up to time T0, and therefore
lacks up-to-date knowledge K. Under this setting,
we study the problem of answering queries under
continuous knowledge drift and the goal is to en-
ableMto produce the correct final answer Afor
diverse types of queries.
3.2 Query Analysis
Given an input query qraw, the model Mis
prompted to perform query analysis:
E, q 0, tq=M(q raw| P1),(1)
whereE={e} denotes the set of extracted entities,
q0is a rewritten, time-agnostic version of the query
with all temporal expressions removed, and tq=
[ts, te]is the time window of the query. P1denotes
the prompt used for this task.

It decomposes the raw query into explicit seman-
tic components, enabling more precise and con-
trollable retrieval. In addition, explicitly modeling
temporal constraints is crucial because temporal
information retrieval remains a long-standing chal-
lenge (Abdallah et al., 2025; Kanhabua and Anand,
2016). Dense text embeddings often struggle to
accurately encode temporal expressions, making
it difficult to distinguish events occurring at dif-
ferent points in time. By explicitly representing
the time window, our approach enables time-aware
retrieval that better aligns retrieved evidence with
the intended temporal context of the query.
3.3 Time-aware Retrieval
For each extracted entity e, we treat it as a query
qeand associate it with a time window tq= [ts, te].
We perform time-aware retrieval in parallel for all
entities using dense semantic embeddings.
LetKqe={(k i, sim i)}N
i=1denote the top- N
candidate knowledge items retrieved based on se-
mantic similarity, where simi∈[0,1] is the sim-
ilarity score between the query qeand the knowl-
edge item ki. Each knowledge item kiis associated
with a timestampt i.
To incorporate temporal relevance, we define the
temporal distance between the timestamp tiand
the query time windowt qas:
∆(ti, tq) =

ts−ti,ift i< ts,
ti−te,ift i> te,
0,ift s≤ti≤te.(2)
This definition assigns zero distance to events
occurring within the target time interval and penal-
izes events outside the interval proportionally to
their temporal deviation.
We then convert this distance into a temporal rel-
evance score using an exponential decay function:
TimeScore(k i) = exp
−∆(ti, tq)
τ
,(3)
where τis a decay hyperparameter controlling how
rapidly relevance decreases as the timestamp moves
away from the query interval.
Finally, we compute a unified retrieval score
by combining semantic similarity and temporal
relevance:
Score(k i) =α sim i+ (1−α) TimeScore(k i),
(4)where α∈[0,1] balances the contribution of se-
mantic similarity and temporal proximity.
All candidate knowledge items are ranked in
descending order according to Score(k i), and the
top-nitems are selected to form the time-aware
retrieved set:
Kr=Top-n(K qe,Score).(5)
This time-aware retrieval mechanism prioritizes
evidence that is both semantically relevant and tem-
porally aligned with the query, providing a princi-
pled foundation for subsequent steps.
3.4 Event Evolution Graph Construction
Given a set of retrieved knowledge quadruples
Kr={(s i, ri, oi, ti)}n
i=1, we construct anEvent
Evolution Graph (EEG)to explicitly model how en-
tities, their relations, and associated events evolve
over time. The EEG is built through three progres-
sive stages, denoted asEEG-I,EEG-F, andEEG-A,
corresponding to theInitial,Full, andAugmented
graphs, respectively.
Step 1: Initial Temporal Ordering.We first con-
struct an initial temporal graph by organizing all re-
trieved events according to their timestamps. Each
knowledge quadruple (si, ri, oi, ti)is treated as an
event node Eand positioned along a global time-
line, forming a linear structure that reflects the
chronological progression of facts. Formally, we
define the initial event evolution graph as
GI= (K r,→t),(6)
where Krdenotes the set of retrieved events, and
→tdenotes directed temporal edges induced by
their chronological ordering. Specifically, we add
a temporal edge ki→tkjonly between two con-
secutive events ki, kj∈ Krsuch that ti< tj. The
resulting graph GI, referred to asEEG-I, captures
the temporal ordering among retrieved events.
Step 2: History Reconstruction.To enable rea-
soning over historical context beyond retrieved
facts, we further construct a more complete view
of event evolution by incorporating reconstructed
historical events.
Specifically, conditioned on the time-agnostic
query q0and its associated time window tq, we
prompt the model Mto infer historical events that
are relevant to the query. This process yields a set
of historical knowledge quadruples:
KH=M(q 0, tq| P2),(7)

where each element in KHis a time-stamped
quadruple (si, ri, oi, ti). Here, P2denotes the
prompt used to elicit historical knowledge.
These reconstructed quadruples are then merged
with the retrieved events and inserted into the
global timeline according to their inferred times-
tamps, yielding full graphEEG-FG F:
GF= 
GI∪ KH,→ t
,(8)
where→tdenotes directed temporal edges induced
by chronological ordering.
Step 3: Event Augmentation and Entity-Centric
Linking.To augment the graph, we enrich it
with additional relevant events. Specifically, con-
ditioned on the raw query qrawand the current full
graphGF, we use prompt P3to instruct the model
Mto either (i) directly generate additional knowl-
edge quadruples or (ii) produce a follow-up query
q1that targets potentially missing but relevant infor-
mation, which is then retrieved to obtain additional
knowledge quadruples. Obtained quadruples are
merged into the existing graph, forming an aug-
mented knowledge set Kaug, which improves the
coverage and completeness.
While temporal ordering captures the global pro-
gression of all events, effective reasoning often re-
quires tracking how thesame entityevolves across
multiple events over time. Therefore, we further
augment the graph through an entity-centric linking
mechanism.
For each entity e, we define the set of all events
in which it appears as either subject or object:
Ke={k i∈ K aug|si=e∨o i=e}.(9)
These events are sorted in ascending order of
their timestamps, and consecutive events are con-
nected by entity-specific directed edges. Formally,
for any two events ki, kj∈ K e, we add an entity
edgeki→ekjifkjimmediately follows kiin time
for the same entity. It yields a temporal chain that
captures the evolution of each entity over time.
The resulting augmented graph, denoted asEEG-
AGA, integrates three complementary components:
(i) temporal edges that encode the global chronolog-
ical order of events, (ii) historical and augmented
events that provide broader contextual coverage,
and (iii) entity-centric edges that model how indi-
vidual entities evolve across time.3.5 Prediction with Multi-Perspective Views
Given the augmented event evolution graph GA,
we derive multiple complementary subgraph-based
views to support robust reasoning. Specifically,
we define two types of views: (i)Temporal view.
Φtemp(GA)denotes a subgraph that captures the
global chronological structure of events within the
queried time window. It is constructed by selecting
events and temporal edges from GAwhose times-
tamps fall within the query scope, thereby repre-
senting the overall temporal progression of relevant
facts. (ii)Entity-centric views.For each entity el,
Φl
ent(GA)denotes a subgraph that captures the evo-
lution of that entity over time. This view consists
of all events involving entity el, together with the
entity-specific temporal edges that link these events
in chronological order.
Each view is serialized into text and used as con-
textual evidence for reasoning. The final prediction
is obtained by conditioning the model on the input
query together with these multi-perspective repre-
sentations:
A=M
qraw,Φ temp(GA),{Φ(l)
ent(GA)}L
i=1| P4
,
(10)
where Adenotes the predicted answer and P4de-
notes the prompt used for inference.
By explicitly structuring both global temporal
evolution and entity-specific trajectories prior to
generation, this formulation enables the model to
reason over evolving knowledge more reliably and
to produce temporally consistent answers without
any additional parameter updates.
4 Experiments
4.1 Setup
We evaluate several representative methods com-
monly used to address knowledge drift, together
with the new baseline introduced in this work.
We consider three large language models (LLMs)
covering both proprietary and open-source set-
tings:GPT-4o( gpt-4o-2024-11-20 ),Claude
(claude-3-haiku-20240307 ), andLLaMA-3.1
(Meta-Llama-3.1-8B-Instruct ). These models
span different architectures, training scales, and
accessibility levels.
Their respective knowledge cutoff dates are Oc-
tober 2023 (GPT-4o), August 2023 (Claude), and
December 2023 (LLaMA-3.1), all preceding our
benchmark time range starting from January 2024.
This setup ensures that none of the models has di-

Table 2: Performance comparison across historical, contemporary, and commonsense question settings. Higher
values indicate better accuracy. Best results for each model are shown inbold, and second-best results are underlined .
Green values indicate improvements over the strongest corresponding baseline.Chronosachieves consistent gains
across multiple evaluation settings, while other baselines exhibit varying limitations.
Method HistoricalContemporaryCommonsense Overall
C1 C2 C3
Direct Generation
LLaMA-3.1 45.95 11.31 1.38 1.69 45.53 21.17
Claude 55.86 10.53 2.75 3.39 62.55 27.02
GPT-4o 61.26 14.81 3.21 5.08 67.32 30.34
Parametric Updating Methods
ROME LLaMA-3.1 10.81 2.92 0.92 0.00 44.92 11.91
MEMIT LLaMA-3.1 25.23 7.99 1.38 0.00 42.59 15.44
WISE LLaMA-3.1 46.85 11.11 1.83 1.69 46.51 21.60
LoRA FT LLaMA-3.1 35.14 71.15 4.13 10.17 40.27 32.17
Retrieval-based Methods
Vanilla RAG LLaMA-3.1 10.81 80.31 39.91 16.95 45.41 38.68
Vanilla RAG Claude 27.03 84.02 44.50 23.73 63.53 48.56
Vanilla RAG GPT-4o 9.91 86.74 50.46 25.42 64.87 47.48
ReAct RAG LLaMA-3.1 6.31 64.52 47.25 32.20 47.98 39.65
ReAct RAG Claude 18.92 91.62 59.63 47.46 67.07 56.94
ReAct RAG GPT-4o 7.21 93.37 56.42 45.76 69.89 54.53
Chronos (Ours)
Chronos LLaMA-3.1 50.45 +3.6 92.40 +12.1 63.76 +16.5 61.02 +28.8 50.55 +2.6 63.64 +24.0
Chronos Claude 57.66 +1.8 93.76 +2.1 66.97 +7.3 74.58 +27.1 72.34 +5.3 73.06 +16.1
Chronos GPT-4o 63.96 +2.7 96.30 +2.9 66.51 +10.1 71.19 +25.4 71.36 +1.5 73.86 +19.3
rect access to the evaluated knowledge during pre-
training. Answers are evaluated using exact-match
accuracy for all QA tasks. More details of the ex-
perimental setup can be found in Appendix D .
4.2 Baselines
We evaluate the following baselines.Direct gen-
erationassesses pretrained models without any
external knowledge updates.Parametric meth-
odsinclude representative knowledge editing and
fine-tuning approaches, such as ROME (Meng
et al., 2022a), MEMIT (Meng et al., 2022b), WISE
(Wang et al., 2024), and LoRA-based fine-tuning
(Hu et al., 2022), which update model parameters
to incorporate new information. All of these are
implemented onLLaMA-3.1.Retrieval-based meth-
odsaugment generation with external knowledge at
inference time, including vanilla RAG (Lewis et al.,
2020) and ReAct-style RAG (Yao et al., 2022),
which incorporates self-reflective reasoning to iter-
atively guide retrieval. These methods are instanti-
ated with different backbone models.
4.3 Main Results
Table 2reports the performance of different adap-
tation paradigms under historical, contemporary,
and commonsense evaluation settings. Several keyobservations can be drawn from the results.
Direct generation by LLMs fails to cope with
continuous knowledge drift, as such models lack
access to up-to-date information. Their reasonable
performance on historical and commonsense ques-
tions reflects intact general reasoning ability and
thus provides a baseline for assessing the impact of
newly emerging knowledge in other methods.
Takeaway #1: Parametric updating exhibits
limited scalability under evolving knowledge.
Parametric updating methods exhibit limited ro-
bustness under continuous knowledge drift, with
some showing clear degradation in basic reason-
ing ability, suggesting catastrophic forgetting.
Among these methods, LoRA fine-tuning shows
slightly better performance, while other paramet-
ric approaches fail to deliver meaningful improve-
ments. However, even LoRA fine-tuning still harms
performance on historical and commonsense ques-
tions. Although prior work has shown that knowl-
edge editing and fine-tuning can yield improve-
ments in isolated settings, their effectiveness re-
mains limited when adapting to continuous knowl-
edge drift.
These results suggest that parameter-level up-

dates struggle to preserve previously acquired
knowledge while incorporating new information.
Moreover, because parametric methods typically
update the model either one instance at a time or in
a batch-wise manner, newly acquired knowledge
can overwrite both the model’s original old knowl-
edge and earlier updates, leading to additional for-
getting over time. An error case analysis and fur-
ther discussion of knowledge editing methods are
provided in Appendix E .
Takeaway #2: Traditional RAG Lacks Flexi-
bility in Complex Temporal Reasoning.
Vanilla RAG and ReAct-style RAG show sub-
stantial improvements on contemporary queries,
but their performance degrades on historical
questions. Their effectiveness also remains lim-
ited in complex temporal reasoning settings.
The poor performance in C2 and C3 can be at-
tributed to the need to aggregate multiple pieces
of evidence. These settings require reasoning over
information from multiple timestamps or entities
and often involve ordering events across time. A
single query is insufficient to capture such com-
plexity. Even though ReAct-style RAG can retrieve
additional evidence through iterative querying, the
resulting retrieved content is often noisy and un-
structured, making it difficult for the model to form
a coherent temporal understanding. While ReAct-
style RAG performs well on commonsense queries
due to its multi-step reasoning process, this advan-
tage does not directly translate to complex temporal
reasoning scenarios.
Takeaway #3: Explicitly modeling event evo-
lution enables more reliable reasoning under
continuous knowledge drift.
Chronosconsistently delivers strong and sta-
ble performance across settings and backbone
models, particularly in scenarios that require ag-
gregating and ordering multiple time-stamped
facts, while maintaining performance on histori-
cal knowledge and commonsense reasoning.
This suggests that explicitly organizing retrieved
evidence into anEvent Evolution Graph(EEG),
rather than merely retrieving relevant documents,
helps preserve previously acquired knowledge
while supporting temporally coherent reasoning.
By structuring evidence along its temporal evolu-Table 3: Ablation study ofChronos. Each value re-
ports accuracy, followed by its change relative to the
full model. Green values indicate performance improve-
ments, while red values indicate performance drops.
Method Hist. Contemporary Comm.
C1 C2 C3
Chronos (Ours) 63.96 96.30 66.51 71.19 71.36
- Time-aware Retr.62.16 -1.8 92.01 -4.3 57.80 -8.7 59.32 -11.9 71.76 +0.4
- History Recon.6.31 -57.7 97.66 +1.4 67.43 +0.9 72.89 +1.7 71.69 +0.3
- Event Augment.62.16 -1.8 92.53 -3.8 55.05 -11.5 52.54 -18.7 70.50 -0.9
- Temporal View61.26 -2.7 90.25 -6.1 58.26 -8.3 55.93 -15.3 70.65 -0.7
- Entity View60.36 -3.6 89.86 -6.4 49.54 -17.0 50.85 -20.3 71.32 -0.0
tion and serializing it into multi-perspective views,
the model reduces interference between old and
new information, leading to more stable and reli-
able behavior under continuous knowledge drift. A
case study is shown in Appendix F .
4.4 Ablation Study
Table 3presents an ablation study demonstrating
that all components contribute to the overall perfor-
mance improvements ofChronos. The variations
observed on commonsense QA are negligible.
Time-aware retrieval helps retrieve more tem-
porally relevant evidence, enabling more precise
extraction of information within the target time win-
dow. While historical reconstruction may introduce
some interference when answering about contem-
porary queries, this effect is expected and difficult
to completely avoid. However, historical recon-
struction plays a crucial role in improving perfor-
mance on historical questions. Event augmentation
further enriches the EEG by supplementing addi-
tional relevant knowledge, leading to more informa-
tive representations. Moreover, the temporal-view
and entity-view components provide two different
perspectives for serializing the EEG, jointly offer-
ing the model a more structured and informative
context. Both views are necessary and mutually
complementary for effective reasoning.
These components help the method progres-
sively construct and refine the EEG, enabling more
robust and temporally consistent reasoning under
continuous knowledge drift.
5 Conclusion
This paper studies the limitations of current adapta-
tion paradigms for LLMs under continuous knowl-
edge drift. We introduce a new benchmark of real-
world dynamic events that captures how knowledge
evolves over time and enables systematic evalua-
tion of temporal consistency in model reasoning.

Our results show that existing approaches strug-
gle to maintain coherent understanding as knowl-
edge evolves. To address this, we propose a time-
aware retrieval baseline based on an Event Evo-
lution Graph, enabling temporally consistent rea-
soning without modifying model parameters. We
believe this work provides a useful foundation for
future research on adapting LLMs to continuously
evolving real-world knowledge.
Limitations
To focus on recent and realistic knowledge changes,
our benchmark is constructed under the assump-
tion that the underlying language models have a
knowledge cutoff before 2024, while the evalua-
tion data cover up-to-date knowledge from 2024 to
2025. In real-world applications, however, the tem-
poral scope of relevant knowledge may vary across
domains and deployment settings, and should be ad-
justed accordingly. Moreover, practical industrial
scenarios often involve more complex and hetero-
geneous knowledge dynamics than those captured
in our benchmark, calling for richer and more chal-
lenging test cases in future work.
References
Abdelrahman Abdallah, Bhawna Piryani, Jonas Wallat,
Avishek Anand, and Adam Jatowt. 2025. Extending
dense passage retrieval with temporal information.
arXiv e-prints, pages arXiv–2502.
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, and 1 others. 2023. Gpt-4 techni-
cal report.arXiv preprint arXiv:2303.08774.
Hoyeon Chang, Jinho Park, Seonghyeon Ye, Sohee
Yang, Youngkyung Seo, Du-Seong Chang, and Min-
joon Seo. 2024. How do large language models ac-
quire factual knowledge during pretraining?Ad-
vances in neural information processing systems,
37:60626–60668.
Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang
Yu, Haotian Wang, Ming Liu, and Bing Qin. 2024.
Timebench: A comprehensive evaluation of tempo-
ral reasoning abilities in large language models. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 1204–1228.
Nicola De Cao, Wilker Aziz, and Ivan Titov. 2021. Edit-
ing factual knowledge in language models.arXiv
preprint arXiv:2104.08164.Guanting Dong, Jiajie Jin, Xiaoxi Li, Yutao Zhu,
Zhicheng Dou, and Ji-Rong Wen. 2025. Rag-critic:
Leveraging automated critic-guided agentic workflow
for retrieval augmented generation. InProceedings
of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 3551–3578.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. InPro-
ceedings of the 30th ACM SIGKDD conference on
knowledge discovery and data mining, pages 6491–
6501.
Jiacheng Guo, Yue Wu, Jiahao Qiu, Kaixuan Huang,
Xinzhe Juan, Ling Yang, and Mengdi Wang. 2025.
Temporal consistency for llm reasoning process error
identification.arXiv preprint arXiv:2503.14495.
Jinwen He, Yujia Gong, Zijin Lin, Cheng’an Wei, Yue
Zhao, and Kai Chen. 2024. Llm factoscope: Un-
covering llms’ factual discernment through measur-
ing inner states. InFindings of the Association for
Computational Linguistics ACL 2024, pages 10218–
10230.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
Weizhu Chen, and 1 others. 2022. Lora: Low-rank
adaptation of large language models.ICLR, 1(2):3.
Joel Jang, Seonghyeon Ye, Changho Lee, Sohee Yang,
Joongbo Shin, Janghoon Han, Gyeonghun Kim, and
Minjoon Seo. 2022. Temporalwiki: A lifelong bench-
mark for training and evaluating ever-evolving lan-
guage models.arXiv preprint arXiv:2204.14211.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong C Park. 2024. Adaptive-rag: Learn-
ing to adapt retrieval-augmented large language mod-
els through question complexity.arXiv preprint
arXiv:2403.14403.
Pengcheng Jiang, Lang Cao, Ruike Zhu, Minhao
Jiang, Yunyi Zhang, Jimeng Sun, and Jiawei
Han. 2025a. Ras: Retrieval-and-structuring for
knowledge-intensive llm generation.arXiv preprint
arXiv:2502.10996.
Pengcheng Jiang, Jiacheng Lin, Lang Cao, R. Tian,
S. Kang, Z. Wang, Jimeng Sun, and Jiawei Han.
2025b. Deepretrieval: Hacking real search en-
gines and retrievers with large language models
via reinforcement learning.arXiv preprint arXiv:
2503.00223.
Yuxin Jiang, Yufei Wang, Chuhan Wu, Wanjun Zhong,
Xingshan Zeng, Jiahui Gao, Liangyou Li, Xin Jiang,
Lifeng Shang, Ruiming Tang, and 1 others. 2024.
Learning to edit: Aligning llms with knowledge edit-
ing.arXiv preprint arXiv:2402.11905.

Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2024. Hipporag: Neu-
robiologically inspired long-term memory for large
language models.Advances in Neural Information
Processing Systems, 37:59532–59569.
Nattiya Kanhabua and Avishek Anand. 2016. Temporal
information retrieval. InProceedings of the 39th
International ACM SIGIR conference on Research
and Development in Information Retrieval, pages
1235–1238.
Zixuan Ke, Yijia Shao, Haowei Lin, Tatsuya Kon-
ishi, Gyuhak Kim, and Bing Liu. 2023. Contin-
ual pre-training of language models.arXiv preprint
arXiv:2302.03241.
Omer Levy, Minjoon Seo, Eunsol Choi, and Luke
Zettlemoyer. 2017. Zero-shot relation extrac-
tion via reading comprehension.arXiv preprint
arXiv:1706.04115.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Chen-An Li and Hung-Yi Lee. 2024. Examining for-
getting in continual pre-training of aligned large lan-
guage models.arXiv preprint arXiv:2401.03129.
Hongyu Li, Liang Ding, Meng Fang, and Dacheng Tao.
2024a. Revisiting catastrophic forgetting in large lan-
guage model tuning. InFindings of the Association
for Computational Linguistics: EMNLP 2024, pages
4297–4308.
Xiaopeng Li, Shasha Li, Shezheng Song, Jing Yang, Jun
Ma, and Jie Yu. 2024b. Pmet: Precise model editing
in a transformer. InProceedings of the AAAI Con-
ference on Artificial Intelligence, volume 38, pages
18564–18572.
Stephanie Lin, Jacob Hilton, and Owain Evans. 2022.
Truthfulqa: Measuring how models mimic human
falsehoods. InProceedings of the 60th annual meet-
ing of the association for computational linguistics
(volume 1: long papers), pages 3214–3252.
Hanbing Liu, Lang Cao, Yuanyi Ren, Mengyu Zhou,
Haoyu Dong, Xiaojun Ma, Shi Han, and Dongmei
Zhang. 2025a. Bingo: Boosting efficient reasoning
of llms via dynamic and significance-based reinforce-
ment learning.arXiv preprint arXiv:2506.08125.
Hanbing Liu, Huaze Tang, Yanru Wu, Yang Li, and
Xiao-Ping Zhang. 2025b. Reinforced domain selec-
tion for continuous domain adaptation. InICASSP
2025-2025 IEEE International Conference on Acous-
tics, Speech and Signal Processing (ICASSP), pages
1–5. IEEE.Hanbing Liu, Jingge Wang, Xuan Zhang, Ye Guo, and
Yang Li. 2024. Enhancing continuous domain adap-
tation with multi-path transfer curriculum. InPacific-
Asia Conference on Knowledge Discovery and Data
Mining, pages 286–298. Springer.
Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou,
and Yue Zhang. 2025. An empirical study of catas-
trophic forgetting in large language models during
continual fine-tuning.IEEE Transactions on Audio,
Speech and Language Processing.
Kevin Meng, David Bau, Alex Andonian, and Yonatan
Belinkov. 2022a. Locating and editing factual as-
sociations in gpt.Advances in neural information
processing systems, 35:17359–17372.
Kevin Meng, Arnab Sen Sharma, Alex J Andonian,
Yonatan Belinkov, and David Bau. 2022b. Mass-
editing memory in a transformer. InThe Eleventh
International Conference on Learning Representa-
tions.
Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea
Finn, and Christopher D Manning. 2021. Fast model
editing at scale.arXiv preprint arXiv:2110.11309.
Jie Ouyang, Tingyue Pan, Mingyue Cheng, Ruiran Yan,
Yucong Luo, Jiaying Lin, and Qi Liu. 2025. Hoh: A
dynamic benchmark for evaluating the impact of out-
dated information on retrieval-augmented generation.
arXiv preprint arXiv:2503.04800.
Hongjin Qian, Peitian Zhang, Zheng Liu, Kelong Mao,
and Zhicheng Dou. 2024. Memorag: Moving to-
wards next-gen rag via memory-inspired knowledge
discovery.arXiv preprint arXiv:2409.05591, 1.
Yifu Qiu, Zheng Zhao, Yftah Ziser, Anna Korhonen,
Edoardo Ponti, and Shay B Cohen. 2024. Are large
language model temporally grounded? InProceed-
ings of the 2024 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume 1:
Long Papers), pages 7064–7083.
Kartik Sharma, Peeyush Kumar, and Yunqing Li. 2025.
Og-rag: Ontology-grounded retrieval-augmented
generation for large language models. InProceed-
ings of the 2025 Conference on Empirical Methods in
Natural Language Processing, pages 32950–32969.
Haizhou Shi, Zihao Xu, Hengyi Wang, Weiyi Qin,
Wenyuan Wang, Yibin Wang, Zifeng Wang, Sayna
Ebrahimi, and Hao Wang. 2025. Continual learning
of large language models: A comprehensive survey.
ACM Computing Surveys, 58(5):1–42.
Jiankai Sun, Chuanyang Zheng, Enze Xie, Zhengying
Liu, Ruihang Chu, Jianing Qiu, Jiaqi Xu, Mingyu
Ding, Hongyang Li, Mengzhe Geng, and 1 others.
2025. A survey of reasoning with foundation mod-
els: Concepts, methodologies, and outlook.ACM
Computing Surveys, 57(11):1–43.

Lukas Thede, Karsten Roth, Matthias Bethge, Zeynep
Akata, and Thomas Hartvigsen. 2025. Wikibigedit:
Understanding the limits of lifelong knowledge edit-
ing in llms. InForty-second International Confer-
ence on Machine Learning.
Mingyang Wang, Alisa Stoll, Lukas Lange, Heike
Adel, Hinrich Schütze, and Jannik Strötgen. 2025.
Bring your own knowledge: A survey of meth-
ods for llm knowledge expansion.arXiv preprint
arXiv:2502.12598.
Peng Wang, Zexi Li, Ningyu Zhang, Ziwen Xu, Yunzhi
Yao, Yong Jiang, Pengjun Xie, Fei Huang, and Hua-
jun Chen. 2024. Wise: Rethinking the knowledge
memory for lifelong model editing of large language
models.Advances in Neural Information Processing
Systems, 37:53764–53797.
Yuqing Wang and Yun Zhao. 2024. Tram: Benchmark-
ing temporal reasoning for large language models.
InFindings of the Association for Computational
Linguistics: ACL 2024, pages 6389–6415.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik R Narasimhan, and Yuan Cao. 2022.
React: Synergizing reasoning and acting in language
models. InThe eleventh international conference on
learning representations.
Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You,
Chao Zhang, Mohammad Shoeybi, and Bryan Catan-
zaro. 2024. Rankrag: Unifying context ranking with
retrieval-augmented generation in llms.Advances in
Neural Information Processing Systems, 37:121156–
121184.
Junhao Zheng, Shengjie Qiu, Chengming Shi, and
Qianli Ma. 2025a. Towards lifelong learning of large
language models: A survey.ACM Computing Sur-
veys, 57(8):1–35.
Junhao Zheng, Chengming Shi, Xidi Cai, Qiuke Li,
Duzhen Zhang, Chenxing Li, Dong Yu, and Qianli
Ma. 2025b. Lifelong learning of large language
model based agents: A roadmap.arXiv preprint
arXiv:2501.07278.

Contents of Appendix
A Ethical Considerations 12
B Related Work 12
C Statistics of the Benchmark Dataset 13
D Details of Experiment Setup 13
EError Case Analysis of Knowledge Edit-
ing 14
F Case Study 14
G Prompt Design 14
A Ethical Considerations
This work focuses on benchmarking and model-
ing the ability of large language models to reason
over continuously evolving real-world knowledge.
All data used in our benchmark are derived from
publicly available sources and do not contain pri-
vate, sensitive, or personally identifiable informa-
tion. The benchmark is constructed to evaluate tem-
poral reasoning and knowledge consistency rather
than to profile individuals or make consequential
decisions.
While our framework aims to improve tempo-
ral consistency under knowledge drift, it does not
guarantee complete correctness, and outdated or
inaccurate outputs may still occur, especially in
high-stakes settings. Accordingly, systems based
on this approach should be deployed with appropri-
ate human oversight.
Overall, we view this work as a methodologi-
cal step toward better understanding and evaluat-
ing temporal adaptation in large language models,
rather than as a standalone solution for real-world
deployment.
B Related Work
Continual Learning of LLMs.Prior work
has explored adapting language models to evolv-
ing knowledge through continual learning, where
model parameters are updated as new data becomes
available (Shi et al., 2025; Zheng et al., 2025a,b;
Hu et al., 2022; Liu et al., 2025b). Continual pre-
training is one strategy for adapting LLMs to new
domains or data streams, but it has been shown
to incur substantial computational cost and can
lead to catastrophic forgetting of previously learnedinformation (Ke et al., 2023; Li and Lee, 2024).
Empirical studies further indicate that sequential
fine-tuning and continual updates are prone to de-
grade performance on previously learned knowl-
edge (Luo et al., 2025; Li et al., 2024a; Liu et al.,
2024), highlighting the challenge of retaining his-
torical knowledge alongside new information.
More recently, knowledge editing methods
have been proposed as a lightweight alternative
that directly modifies a small subset of model
parameters to update specific facts (De Cao
et al., 2021; Jiang et al., 2024; Meng et al.,
2022a,b; Mitchell et al., 2021; Li et al., 2024b).
Representative locate-then-edit approaches such as
ROME and MEMIT identify and modify weights
corresponding to targeted facts for efficient single
or batched updates (Meng et al., 2022a,b), while
other variants such as PMET offer subject-centric
editing with precise control over localized
changes (Li et al., 2024b). Extensions toward
sequential or lifelong editing, such as WISE,
introduce dual memory schemes and knowledge
sharding to support repeated edits while balancing
reliability, generalization, and locality (Wang et al.,
2024). While effective for isolated edits, existing
editing methods primarily focus on overwriting a
model’s current beliefs rather than representing
multiple historical states of knowledge. As a result,
repeated edits may interfere with one another, and
edited models typically lack mechanisms to retain
or reason about past information alongside newly
updated facts.
Retrieval-augmented Generation.Retrieval-
augmented generation (RAG) has emerged as an
effective approach for enabling large language mod-
els to access external information beyond their
fixed pretraining data (Lewis et al., 2020; Jiang
et al., 2025b,a; Jimenez Gutierrez et al., 2024),
thereby mitigating issues related to outdated or in-
complete knowledge without retraining the model
itself (Fan et al., 2024; Yu et al., 2024; Dong
et al., 2025; Yao et al., 2022; Lewis et al., 2020).
Recent work has explored mechanisms to adapt
RAG systems more effectively to evolving infor-
mation. For example, Adaptive-rag proposes strate-
gies that adapt retrieval and generation to vary-
ing query complexities, enabling dynamic adjust-
ment between iterative and single-step retrieval pro-
cesses (Jeong et al., 2024). HoH introduce a dy-
namic benchmark highlighting how current RAG
approaches struggle to handle outdated informa-

tion in retrieval and generation, demonstrating the
need for improved update handling (Ouyang et al.,
2025). On the modeling front, variants such as OG-
RAG ground retrievals using ontological structures
to better capture domain knowledge and improve
contextual accuracy (Sharma et al., 2025). Other
work, such as MemoRAG, incorporates long-term
memory into the RAG pipeline to support improved
retrieval and generation performance, particularly
in complex tasks where conventional RAG may fall
short (Qian et al., 2024). Despite these advances,
most of these approaches assume a largely static
retrieval corpus and do not explicitly model the
temporal evolution of knowledge, limiting their
ability to adapt to continuously drifting real-world
information, which motivates our temporal retrieval
framework.
C Statistics of the Benchmark Dataset
The ten topical domains used during dataset con-
struction are as follows:
1.Politics & Government Leadership
2.Corporate Leadership & Executive Roles
3.Sports, Coaching Changes & Player Transfers
4.Regulation, Compliance, & Financial Penalties
5.Mergers, Acquisitions, & Strategic Partnerships
6.Product Releases & Version Updates
7.Natural Disasters & Climate Events
8.Public Health Events & Disease Outbreaks
9.Scientific Discoveries & Research Publications
10.Economic Indicators & Policy Changes
After construction, each topical domain contains
between 10 and 20 core subjects, and each subject
is associated with approximately 3 to 7 temporally
ordered events. In total, the benchmark comprises
513 knowledge quadruples. It includes 111 his-
torical questions and 817 commonsense questions
adopted from TruthfulQA (Lin et al., 2022). For
contemporary questions, we construct three sub-
categories: 513 single-timestamp questions (C1),
218 multi-timestamp questions (C2), and 59 multi-
source questions (C3), each designed to evaluate
increasingly complex forms of temporal reasoning.
D Details of Experiment Setup
For generation, we set the temperature of all LLMs
to 0 to ensure stable and deterministic outputs,
while keeping all other hyperparameters at theirTable 4: An error case of parameter-based knowledge
editing under continuous knowledge drift. While WISE
produces the same answer for all time-specific queries
due to overwriting effects,Chronoscorrectly returns
different answers conditioned on the queried timestamp,
demonstrating its ability to model temporal variation in
evolving knowledge.
Query (C1) WISE Output Chronos Output
Who was the world’s richest person on Jan 1, 2024? Elon Musk✓Elon Musk✓
Who was the world’s richest person on Jan 28, 2024? Elon Musk×Bernard Arnault✓
Who was the world’s richest person on Mar 5, 2024? Elon Musk×Jeff Bezos✓
Who was the world’s richest person on Apr 2, 2024? Elon Musk×Bernard Arnault✓
Who was the world’s richest person on Jun 8, 2024? Elon Musk✓Elon Musk✓
default values. All prompts used in the experiments
are provided in Appendix G .
For dense retrieval, we use all-mpnet-base-v2
as a lightweight retriever across all retrieval-based
methods, retrieving the top 4 documents for each
query. The retrieval pipeline is implemented using
LangChain1. For the ReAct baseline, the maxi-
mum number of reasoning steps is set to 3. We use
the cosine similarity between the query and target
document embeddings as the similarity score.
For knowledge-editing baselines, we perform
edits on the target objects according to their cor-
responding time-stamped knowledge quadruples,
with edits conditioned on temporal information.
Specifically, for ROME, MEMIT, and WISE, we
adopt the default hyperparameters reported in their
original papers. For the LoRA-based fine-tuning
baseline, we use a learning rate of 1×10−4, a batch
size of 8, and rank 8, selecting the best-performing
checkpoint at epoch 5. All experiments are con-
ducted on a single NVIDIA A800 80GB GPU.
For the Commonsense QA task, we adopt
the multiple-choice questions (MC1-target) from
TruthfulQA (Lin et al., 2022), using the V1 version
as the evaluation dataset. All candidate answer op-
tions are explicitly provided in the prompt, and the
model is instructed to select one option, allowing
the responses to be evaluated using exact-match
accuracy.
For our proposed baselineChronos, we also re-
trieve the top 4 documents for each query. In the
time-aware retrieval module, we set α= 0.75 and
τdays= 180 , which achieve the best performance
in our validation experiments. We observe that the
time-aware retrieval mechanism is not highly sen-
sitive to these hyperparameters and can be tuned
easily and intuitively within a reasonable range.
1https://www.langchain.com

E Error Case Analysis of Knowledge
Editing
To better understand the limitations of parameter-
based knowledge editing under continuous knowl-
edge drift, we first highlight two fundamental chal-
lenges inherent to this paradigm. First, knowledge
editing methods typically treat updated facts as
static, rather than modeling them as temporally
evolving entities. As a result, they lack mechanisms
to represent multiple coexisting states of the same
fact across different time points. Second, existing
methods are designed to perform isolated edits and
do not scale well to scenarios where many related
facts must be updated jointly. When updates arrive
sequentially, repeated parameter modifications can
interfere with one another, leading to overwriting
effects and the loss of previously edited knowledge.
To illustrate these limitations, we analyze a repre-
sentative failure case of WISE (Wang et al., 2024).
As shown in Table 4, the underlying knowledge
concerns the identity of the world’s richest person
at different time points. Although the dataset con-
tains multiple time-stamped facts, WISE applies
edits sequentially and updates model parameters
in place. Consequently, later edits overwrite ear-
lier ones, and the model collapses multiple tem-
poral states into a single dominant answer. When
queried with different time-specific questions (C1),
the edited model repeatedly produces the same out-
put corresponding to the most recent update, re-
gardless of the queried timestamp.
This example directly reflects the two limitations
outlined above. Because parameter-level editing
does not explicitly model time, it cannot represent
multiple coexisting states of the same fact, causing
temporally distinct knowledge to collapse into a
single outcome. Moreover, since edits are applied
sequentially and independently, repeated updates
interfere with one another, leading to overwriting
effects when many related facts must be updated
over time. Together, these behaviors illustrate why
parameter-based editing struggles under continu-
ous knowledge drift, and motivate the need for ap-
proaches that explicitly model temporal structure
and preserve access to multiple historical states
rather than encoding evolving knowledge implic-
itly in model parameters.
F Case Study
Figure 3presents a case study ofChronosfor ques-
tion answering under continuous knowledge drift.The example involves a multi-source contemporary
query that requires the model to jointly reason over
changes in a company’s stock price and the cor-
responding changes in the world’s richest person.
The figure illustrates the data flow through each
stage of the framework, showing how evidence is
progressively retrieved, augmented, and organized,
ultimately leading to the correct answer.
G Prompt Design
Figures 4,5, and 6present the prompt templates
adopted by the LLM-based baseline methods. Fig-
ures 7,8,9, and 10illustrate the prompts P1,P2,
P3, andP4employed at different stages of our
proposed method,Chronos.

InputRaw Query: Which company’s stock surge led to Elon Musk losing his position as the world’s richest person on September 10, 2025?-------------------------------------------------------------------------------------------------------------------------------------------------Query AnalysisEntities: company’s stock, Elon Musk, the world’s richest person Time window: (2025-09-10, 2025-09-10) Time-agnostic Query: Which company’s stock surge led to Elon Musk losing his position as the world’s richest person ---------------------------------------------------------------------------------------------------------------------------------------------------Parallel Time-aware RetrievalElon Musk:<World’s Richest Person, held by, Elon Musk, 2024-01-01><World’s Richest Person, held by, Elon Musk, 2024-06-08><World’s Richest Person, held by, Elon Musk, 2025-09-11>...the world’s richest person:<World’s Richest Person, held by, Bernard Arnault, 2024-01-28><World’s Richest Person, held by, Jeff Bezos, 2024-03-05><World’s Richest Person, held by, Bernard Arnault, 2024-04-02><World’s Richest Person, held by, Larry Ellison, 2025-09-10>…company’s stock:<Oracle stock price, surged to, USD 328, 2025-09-10>…---------------------------------------------------------------------------------------------------------------------------------------------------History Reconstruction(no output, since the specified time window lies entirely after 2024)---------------------------------------------------------------------------------------------------------------------------------------------------Event Augmentation<Larry Ellison, is cofounder of, Oracle, 1977-06-01>…---------------------------------------------------------------------------------------------------------------------------------------------------FinalEventEvolutionGraph (Chronologically Ordered)(1977-06-01)  <Larry Ellison, is cofounder of, Oracle>…(2024-01-01)<World’s Richest Person, held by, Elon Musk>(2024-01-28)<World’s Richest Person, held by, Bernard Arnault>(2024-03-05)<World’s Richest Person, held by, Jeff Bezos>(2024-04-02)<World’s Richest Person, held by, Bernard Arnault>(2024-06-08)<World’s Richest Person, held by, Elon Musk>(2025-09-10)<World’s Richest Person, held by, Larry Ellison>(2025-09-10)<Oracle stock price, surged to, USD 328>(2025-09-11)<World’s Richest Person, held by, Elon Musk>…---------------------------------------------------------------------------------------------------------------------------------------------------FinalResponseOn September 10, 2025, the record shows that Larry Ellison became the world’s richest person. The evidence also states that on the same date, Oracle’s stock price surged to USD 328. Since Larry Ellison is the cofounder of Oracle, the surge in Oracle’s stock explains why his wealth overtook Elon Musk’s, causing Musk to lose the top position.Answer: Oracle (Correct 
✅)Figure 3: A case study ofChronosfor question answering under continuous knowledge drift.

# TaskAnswer the question.# Question# Instructions1. Use only your own knowledge (no external tools or retrieval).2. Return only the final answer as a word or noun phrase.3. If multiple answers are required, provide them as a single comma-separated list.4. Do NOT include explanations, reasoning, or any additional words.Answer:Prompt of Direct GenerationFigure 4: Prompt template used for direct generation baseline. Blue text indicates input variables.
# TaskAnswer the question using ONLY the retrieved factual statements.# Question{query}# Retrieved Facts{retrieved_text}# Instructions1. Return ONLY the final answer as a word or noun phrase.2. If multiple answers are required, return a single comma-separated list (chronological order if applicable).3. Do NOT include explanations, reasoning, citations, or extra words.Answer:Prompt of RAG
Figure 5: Prompt template used for retrieval-augmented generation baseline. Blue text indicates input variables.

# TaskYou are answering a question using ONLY retrieved factual statements.You may either:-propose a new retrieval query to fetch more facts, OR-provide the final answer.# Question{question}# Retrieved Facts So Far{context}# Instructions1. You must output JSON with keys: action, search_query, final_answer.2. Decide action:-Choose action="final" ONLY if the retrieved facts explicitly contain the answer.-Otherwise choose action="search".3. If action="search", search_queryMUST be a short keyword query (NOT a sentence), 3–10 tokens, and MUST include:-the main entity in the question (who/what the question is about),-the relation type (e.g., held_by/ head coach / CEO / capital / founded),-any time constraint if mentioned (year/month/date).Set final_answerto "".4. If action="final", final_answermust be ONLY the answer (a word or noun phrase; if multiple, comma-separated in chronological order).Set search_queryto "".5. Do NOT include explanations, reasoning, citations, or extra words.6. If the question is commonsense and does not involve temporal information, answer it directly without retrieval.# Output JSONReturn strictly JSON:{{"action":"search","search_query":"...","final_answer":""}}or{{"action":"final","search_query":"","final_answer":"..."}}Now respond:Prompt of ReActRAGFigure 6: Prompt template used for ReAct-style retrieval-augmented generation baseline. Blue text indicates input
variables.

# TaskAnalyze the query and extract:1. All entities mentioned in the query.2. The time scope that should be considered for answering.3. A rewritten version of the query with all time expressions removed (query_without_time).# InformationQuery: {query}# Instructions1. Extract all entities into the "entities" field. Keep them clean and concise.2. All dates must be formatted as YYYY-MM-DD.3. If the query contains temporal information, set "has_time" to true; otherwise, set it tofalse.4. If the query refers to a single day, use the same date for both "time_start" and "time_end".5. Produce a natural-language query in 'query_without_time' that preserves meaning but removes all temporal expressions.# Response FormatReturn the output strictly in the following JSON format:{{"entities": ["<entity_1>", "<entity_2>", "..."],"has_time": true/false,"time": ["<time_start>", "<time_end>"],"query_without_time": "<rewritten_query_without_any_time_expression>"}}Now, provide your response:Prompt of Query Analysis (Chronos)Figure 7: Prompt template used for query analysis inChronos. Blue text indicates input variables.
# TaskGenerate time-aware knowledge triples relevant to the given query.# Inputs-Query (without temporal expressions): {query_without_time}-Time Window: [{time_start}, {time_end}]# Instructions1. Use only information that falls within the specified time window.2. Identify events or facts relevant to the query that occurred during this period.3. Represent each event as a knowledge triple in the form:(subject, relation, object, timestamp).4. The timestamp should reflect when the event occurred.5. If multiple valid events exist, return all of them.6. Do not use external knowledge outside the given time window.# Output FormatReturn a JSON list of knowledge quadruples:[{{"subject": "...","relation": "...","object": "...","timestamp": "YYYY-MM-DD"}}]Prompt of History Reconstruction (Chronos)
Figure 8: Prompt template used for history construction inChronos. Blue text indicates input variables.

# TaskDecide how to improve evidence coverage for answering the question.You must choose exactly ONE action:-retrieve: propose ONE focused search query to retrieve missing evidence.-augment: directly propose additional knowledge quadruples to supplement the Event Evolution Graph (EEG).# Question{query}# Current Evidence (from EEG){evidence_from_EEG}# Proposed Answer{initial_predicted_answer}# Rules1. Use ONLY the evidence above to judge whether something is missing. Do NOT use outside knowledge to invent facts.2. If you choose action="retrieve":-search_queryMUST be non-empty (a focused query string).-quads MUST be an empty list [].3. If you choose action="augment":-quads MUST be non-empty (a list of knowledge quadruples).-search_queryMUST be an empty string "".4. Quadruple format:-timestamp MUST be in YYYY-MM-DD.-Each quad must be directly useful to answer the question or to connect key entities/events.5. Output JSON ONLY. No explanations.# Output JSON{{"action": "retrieve" | "augment","search_query": "","quads": [{{"subject": "", "relation": "", "object": "", "timestamp": "YYYY-MM-DD"}}]}}Prompt of Event Augmentation (Chronos)Figure 9: Prompt template used for event augmentation inChronos. Blue text indicates input variables.
# TaskAnswer the question step by step using ONLY the provided information.# InformationQuestion: {question}Evidence: {evidence_from_EEG}# Instructions1. Use ONLY the information in Historical Context and Contemporary Information. Do NOT use outside knowledge.2. You may reason freely.3. The final line MUST be exactly: Answer: <final answer>4. The final answer must be a word or noun phrase.5. If multiple answers are required, output a single comma-separated list (noun phrases) in chronological order when applicable.6. Do NOT write anything after the Answer line.Now, answer:Prompt of Final Response (Chronos)
Figure 10: Prompt template used for final response inChronos. Blue text indicates input variables.
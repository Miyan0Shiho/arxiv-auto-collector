# From Single- to Cross-Document: Benchmarking Multi-Granularity Event Analysis of Large Language Models

**Authors**: Tao Wen, Shuai Shao, Pei Ke, Xu Han, Jie Zou, Guannan Li, Tao Tian, Jinjie Qiu, Lan Wang, Ke Qin

**Published**: 2026-07-30 04:06:55

**PDF URL**: [https://arxiv.org/pdf/2607.27654v1](https://arxiv.org/pdf/2607.27654v1)

## Abstract
Event analysis is an essential and fundamental direction of information extraction, involving various event-centric tasks at different granularity of documents. While large language models (LLMs) have preliminarily achieved promising performance in part of these tasks individually, their capability in event analysis still lacks comprehensive understanding due to restricted document granularity, task designs, and data source of existing benchmarks. To address these limitations, we introduce MiGUE-Bench, a systematic benchmark for assessing the performance of LLMs in multi-granularity event analysis. To support large-scale evaluation, we first develop an LLM-driven self-correcting annotation framework called MiGUE-Pipeline, enabling scalable acquisition of high-quality source data of events with automatic labels. Then, we design four core tasks in our benchmark, i.e., event detection, relation reasoning, structure induction, and future prediction, to probe model competence at different levels, from atomic event details to complex cross-document narratives. Extensive experiments on state-of-the-art LLMs and retrieval-augmented generation (RAG) methods delineate the current capability boundary and identify critical deficiencies, providing insights into the future improvement of LLMs in challenging event analysis tasks.

## Full Text


<!-- PDF content starts -->

From Single- to Cross-Document: Benchmarking
Multi-Granularity Event Analysis of Large Language Models
Tao Wen
Laboratory of Intelligent
Collaborative Computing
University of Electronic Science and
Technology of China
Chengdu, China
enril.wentao@std.uestc.edu.cnShuai Shao
Laboratory of Intelligent
Collaborative Computing
University of Electronic Science and
Technology of China
Chengdu, China
202221080331@std.uestc.edu.cnPei Ke∗
Laboratory of Intelligent
Collaborative Computing
University of Electronic Science and
Technology of China
Chengdu, China
kepei@uestc.edu.cn
Xu Han
Department of Computer Science and
Technology
Tsinghua University
Beijing, China
han-xu@mail.tsinghua.edu.cnJie Zou
School of Computer Science and
Engineering
University of Electronic Science and
Technology of China
Chengdu, China
jie.zou@uestc.edu.cnGuannan Li
Laboratory of Intelligent
Collaborative Computing
University of Electronic Science and
Technology of China
Chengdu, China
lgn4sci@std.uestc.edu.cn
Tao Tian
Jinjie Qiu
Laboratory of Intelligent
Collaborative Computing
University of Electronic Science and
Technology of China
Chengdu, China
taotian@std.uestc.edu.cn
202522900127@std.uestc.edu.cnLan Wang
School of Information and Software
Engineering
University of Electronic Science and
Technology of China
Chengdu, China
202521090303@std.uestc.edu.cnKe Qin
Laboratory of Intelligent
Collaborative Computing
University of Electronic Science and
Technology of China
Chengdu, China
qinke@uestc.edu.cn
Abstract
Event analysis is an essential and fundamental direction of informa-
tion extraction, involving various event-centric tasks at different
granularity of documents. While large language models (LLMs)
have preliminarily achieved promising performance in part of these
tasks individually, their capability in event analysis still lacks com-
prehensive understanding due to restricted document granularity,
task designs, and data source of existing benchmarks. To address
these limitations, we introduce MiGUE-Bench, a systematic bench-
mark for assessing the performance of LLMs in multi-granularity
event analysis. To support large-scale evaluation, we first develop
an LLM-driven self-correcting annotation framework called MiGUE-
Pipeline, enabling scalable acquisition of high-quality source data
of events with automatic labels. Then, we design four core tasks in
our benchmark, i.e., event detection, relation reasoning, structure
induction, and future prediction, to probe model competence at dif-
ferent levels, from atomic event details to complex cross-document
∗Corresponding author.
This work is licensed under a Creative Commons Attribution 4.0 International License.
SIGIR ’26, Melbourne, VIC, Australia.
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2599-9/2026/07
https://doi.org/10.1145/3805712.3808607narratives. Extensive experiments on state-of-the-art LLMs and
retrieval-augmented generation (RAG) methods delineate the cur-
rent capability boundary and identify critical deficiencies, providing
insights into the future improvement of LLMs in challenging event
analysis tasks.
CCS Concepts
•Computing methodologies→Information extraction.
Keywords
Large Language Model, Event Analysis, Automatic Evaluation
ACM Reference Format:
Tao Wen, Shuai Shao, Pei Ke, Xu Han, Jie Zou, Guannan Li, Tao Tian,
Jinjie Qiu, Lan Wang, and Ke Qin. 2026. From Single- to Cross-Document:
Benchmarking Multi-Granularity Event Analysis of Large Language Models.
InProceedings of the 49th International ACM SIGIR Conference on Research and
Development in Information Retrieval (SIGIR ’26), July 20–24, 2026, Melbourne,
VIC, Australia.ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/
3805712.3808607
1 Introduction
Event analysis is a foundational pillar of information extraction,
inherently requiring to extract the structure of events from raw tex-
tual data at different granularity levels [ 6]. This essential direction
arXiv:2607.27654v1  [cs.CL]  30 Jul 2026

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia. Tao Wen et al.
encompasses a hierarchical task taxonomy from the fine-grained
event detection of individual occurrences to the induction of global
event structures across different documents [ 12,13,24,26,29,32,
39,46]. Mastering the capabilities of analyzing events from intra-
document event mentions to inter-document event relations is
indispensable for the application of current information extraction
systems in real-world scenarios, moving beyond local semantic
matching towards global relation reasoning.
Early works on event analysis have extensively explored task-
specific deep learning paradigms, ranging from sequence labeling
architectures [ 8,35,40] to graph-based reasoning frameworks [ 24,
36]. Despite their advanced performance on specific datasets, the
generalization ability of these models is rather limited. Thus, recent
works resort to large language models (LLMs) and preliminarily
show their promising performance in part of event analysis tasks,
such as event extraction [ 23], event relation prediction [ 5,19], and
event reasoning [ 34,41]. Equipped with strong abilities of context
understanding and knowledge utilization, LLMs have shown great
potential in dealing with complex event analysis tasks in a zero-shot
manner, gradually becoming a research focus in this field.
However, we argue that there still lacks a comprehensive bench-
mark for assessing LLMs’ capabilities in event analysis. Existing
benchmarks mostly suffer from restricted granularity, lacking data
source, and homogeneous task design, hindering a systematic un-
derstanding of LLMs’ deficiencies at different event-centric tasks:
•Restricted Granularity: Most of the current benchmarks only
fall into the assessment at constrained granularity of docu-
ments. While some of the datasets aim to measure the model
performance in understanding events within a document
[41, 44], others focus on the event relations across multiple
documents [ 3,18,47], both of which fail to provide an en-
tire perspective of LLMs’ capabilities at different granularity
levels.
•Lacking Data Source: Most existing benchmarks are retrofit-
ted from legacy datasets [ 14], which restrict the scalability to
diverse and contemporary corpora. The lack of data sources
also causes missing cross-document dependencies, making
it improper to measure LLMs’ capabilities to analyze events
across multiple documents.
•Homogeneous Task Design: Existing benchmarks are mostly
targeting at isolated evaluation dimensions and objects, ana-
lyzing the ability of LLMs to deal with specific event relations
[47] or tasks [ 5,19,34,41]. Such homogeneous and narrow
task design may be unable to reveal the capability bound-
ary of general LLMs, exaggerating their performance in the
limited task scope.
To address these limitations, we introduceMiGUE-Bench, a
comprehensive benchmark designed forMulti-GranUlarEvent
analysis. MiGUE-Bench aims to cover the evolving process of event
analysis from single-document event detection and relation rea-
soning to cross-document event structure induction and future
prediction (as shown in Table 1), enabling a holistic assessment
of LLMs’ capabilities in different granularity levels of documents.
Firstly, to deal with the limitation of document granularity and
data source, we develop an LLM-driven automatic pipeline named
MiGUE-Pipeline that includes document filtering, event / relationTable 1: Comparison of different event analysis benchmarks.
BenchmarkSingle-Doc
AnalysisCross-Doc
AnalysisMultiple
RelationsEvent
Structure
ACE05-EN [11]✓ ✗ ✗ ✗
Causal-TimeBank [33]✓ ✗ ✓ ✗
MAVEN [45]✓ ✗ ✗ ✗
MAVEN-ERE [44]✓ ✗ ✓ ✗
TCELongBench [47]✗ ✓ ✗ ✗
EventRelBench [14]✓ ✗ ✓ ✗
MiGUE-Bench (Ours)✓ ✓ ✓ ✓
annotation, and document clustering, enabling scalable data con-
struction from raw corpora to support various event-centric tasks.
Secondly, to solve the challenge of homogeneous task design, we
devise four task types to cover different stages of event analysis,
Specifically,MiGUE-Detectionassesses the performance in recog-
nizing and extracting triggers from event mentions, whileMiGUE-
Reasoningaims to evaluate the capability to reason within and
across fragmented document contexts to extract implicit event rela-
tions. To further obtain a global understanding of event dynamics
across different documents,MiGUE-Inductionis targeted at evaluat-
ing the ability to acquire the global topological structure of events.
Finally,MiGUE-Predictiontests the capability to infer the future de-
velopment of events based on the understanding of event dynamics
across different documents. Our benchmark is expected to reveal
the deficiencies of LLMs via comprehensive and challenging task
design.
In summary, our key contributions are three-fold1:
•We develop the first open-source automatic pipeline named
MiGUE-Pipeline for multi-granularity event data genera-
tion, alleviating the dependence on manual annotation and
flexibly supporting various event-centric tasks.
•We introduce MiGUE-Bench, a comprehensive benchmark
that provides systematic assessment for LLMs’ capabilities
in the entire spectrum of event analysis.
•We conduct extensive experiments on state-of-the-art LLMs
and retrieval-augmented generation frameworks, uncover-
ing critical deficiencies of LLMs on event analysis and pro-
viding insights for future improvement.
2 Related Work
LLMs for Event Analysis.Recent advances in large language
models (LLMs) have enabled a prompt-based paradigm for event-
centric understanding and reasoning in a zero-/few-shot setting.
Existing studies show that LLMs can support core event analysis
capabilities, spanning event extraction [ 9,20,23,25] and event
relation prediction [ 4,5,19,41]. Beyond event understanding in a
single document, LLMs have been utilized for temporal/causal event
reasoning across documents, commonly with retrieval modules
to support more coherent information extraction [ 34,41,46]. To
summarize, existing studies suggest that LLMs hold strong potential
for bridging multiple granularity of event analysis, ranging from
single-document understanding to cross-document reasoning.
1The code of MiGUE-Pipeline is released at https://github.com/EnrilAmirite/
MiGUE-Pipeline, while the MiGUE-Bench dataset is available at https://github.com/
EnrilAmirite/MiGUE-Bench.

From Single- to Cross-Document: Benchmarking Multi-Granularity Event Analysis of Large Language Models SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia.
Despite this advance, current work on LLM-based event analysis
typically focuses on isolated task settings, making it hard to delin-
eate the strengths and weaknesses of LLMs along the whole process
of event analysis, moving from local semantics to global structures.
Therefore, a more systematic benchmark is necessary to compre-
hensively reflect the LLMs’ performance at multiple granularity
levels of event analysis.
Benchmarks for LLMs in Event Analysis.Motivated by the
recent progress of LLMs on event-centric tasks, several benchmarks
have been proposed to probe LLM capabilities along different facets
of event analysis. Existing works typically formulate evaluation
as instruction-following tasks with structured outputs, aiming to
measure whether LLMs can (i) recover event-centric descriptions
from long and temporally dense narratives [ 41,44], (ii) reason
about temporal/causal dependencies among events [ 3,18,47], and
(iii) infer event relations under diverse types and conduct event
reasoning with different formats [14, 47].
Though existing benchmarks offer a preliminary view of LLMs’
event analysis capabilities, they mostly focus on specific event-
centric tasks with restricted granularity, data sources, and task
design. For comparison, our work aims to provide an complete
picture of LLM performance involving different stages in the whole
process of event analysis, with multiple granularity levels of docu-
ments, diverse data sources, and broad task scopes.
3 MiGUE-Pipeline
To support scalable data construction of MiGUE-Bench, we present
MiGUE-Pipeline, a four-stage framework comprisingdocument
filtering,event annotation,relation annotation, andcluster generation,
as shown in Figure 1. Transforming raw corpora into high-quality
multi-granularity event data resources can serve as the foundation
for subsequent evaluation.
3.1 Document Filtering
We firstly employ an LLM-driven filtering mechanism that pre-
liminarily evaluates each candidate document across three critical
dimensions:factual informativeness(prioritizing objective reporting
over subjective commentary or fragmentary discourse),text qual-
ity(ensuring the documents’ grammatical fluency and structural
coherence),event density(excluding redundant or non-informative
contents). A document will be retained only if it satisfies all the
aforementioned criteria based on LLM-as-a-Judge with GPT-4o [ 27].
The multi-dimensional filtering ensures that the resulting docu-
ments are both grammatically fluent and rich in event dynamics.
3.2 Event Annotation
Since identifying events in open documents is challenging, we con-
duct a pilot study on part of documents with GPT-5.2 [ 37] followed
by manual check, finding six typical error types, i.e.,No-Occurrence,
Negated-Claim, Assumption, Abstraction, Named-Entity,andNar-
rative. Detailed descriptions of all these error types are provided
in Table 2. Based on these findings, we propose a two-stage pro-
tocol for event annotation. Firstly, we make an effective LLM (i.e.,
DeepSeek-V3.2 [ 10]) generate a candidate set of event triggers.
Then, we devise a retrieval-augmented reflection method to select
semantically similar examples in the data pool of each error typeTable 2: Description of Event Annotation Error Types.
Error Type Description
No-Occurrence No-Occurrenceerrors arise when the model labels the expressions
describing future plans, intentions, or predictions rather than events
that have actually occurred as triggers. Although such expressions are
event-like in form, they are not realized on the current timeline and
should not be extracted as events.
Negated-Claim Negated-Claimerrors arise when the model fails to distinguish between
the statements about events that did not actually occur and genuinely
occurring events that express negation (e.g., denying). Negation op-
erators themselves (e.g., not, no, never) do not introduce new events,
whereas verbs such as deny, reject, and refute denote individual events
and should be identified as triggers.
Assumption Assumptionerrors arise when the model identifies as triggers those
verbs or action expressions that carry event semantics but occur only in
conditional, hypothetical, intentional, advisory, or exhortative contexts,
rather than being asserted as factual events on the real timeline.
Abstraction Abstractionerrors arise when the model labels the expressions that are
event-like in form but function rhetorically or metaphorically rather
than denote a specific real-world event as triggers. Such expressions
lack clear temporal anchoring, identifiable participants, and concrete
boundaries. Thus, they are not extractable factual events.
Named-Entity Named-Entityerrors arise when the model incorrectly labels nominal
elements of an event (e.g., participants, carriers, results, named entities,
event-denoting nouns, or state descriptions) as triggers.
Narrative Narrativeerrors arise when the model labels the expressions that
narrate, explain, modify, or evaluate an event rather than denote the
core action as event triggers. Such expressions (e.g., stance and manner)
do not introduce new events but only supplement or interpret existing
ones. Thereby, they should not be annotated as events.
from our pilot study, respectively. This method requires the LLM
to self-correct the results by reasoning against similar historical
failures. Manual validation on a subset of 200 data samples shows
that our automatic protocol yields a precision of 0.87 during event
annotation.
3.3 Relation Annotation
To further acquire the relation between the events within a single
document or among different ones, we devise different strategies
to ensure the quality of automatic relation annotation.
Single-Document Relation Annotation.Inspired by existing
works [ 5], we design a constraint-aware multi-step reasoning strat-
egy to capture event dependencies within a document. For each
event pair, LLMs iteratively select the most confident relation
type. Upon selecting a candidate label at each iteration, LLMs are
prompted to validate its decision against a set of explicit logical
constraints (e.g., a coreference label inherently precludes causal
or sub-event relations). Based on self-evaluation, the LLM should
select one of the following operations, i.e.,confirmingthe current
label to proceed to the next most confident relation type,revising
the current label to select an alternative one, andrestartingthe
entire decision process due to the conflict during reasoning. Once
a label is confirmed, we dynamically prune the search space by
removing logically incompatible relation types. To further enhance
reliability, we employ majority voting with three LLMs (i.e., GPT-4o,
DeepSeek-V3.2 [ 10], and Qwen3-Max [ 38]). If all the models yield
identical predictions, the result is accepted; otherwise, the instance
is escalated to a high-capacity LLM (e.g., Claude-4.5-Opus) for final
results. This approach achieves an accuracy rate of 0.82 on 100 data
samples during manual check, which shows its effectiveness.

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia. Tao Wen et al.
MiGUE -Pipeline
Relation Annotation
LLM1
 LLM2
 LLM3
Coreference
CauseCause𝑒1𝑒2
𝑒3Document Filtering
Raw
Corpus
Filtered
DocumentsCluster Generation
…
Documents with 
Coreferential 
Relations
Document 
Affinity GraphGenerated 
Document ClusterHas Coreference 
Event Pair(s)
MiGUE -BenchmarkEvent Annotation
Candidate 
Triggers
announced
suspend
safeguard
hit
…Retrieval
Failure Types
No-Occurrence Abstraction Named -Entity
Assumption Narrative Negated -ClaimIterative 
Self-correction
Criteria
Factual 
Informativeness
Text Quality
Event Density
LLMSatisfy all the 
three criteria?suspendCorrected 
Triggers
announced
hit
…
safeguard
…
Pilot
StudyConstraint -Aware Multi -step Reasoning
(CAMR)Single -Document
Event Pair1. Type Iteration
2. Constrain -Aware Self -Validation
Confirm Restart Revise
3. Majority -ConsensusPropagate -via-Coreference
(PvC)Cross -Document
CASR
Constrained
search space 
Coreference: ✓
MiGUE -Detection
Task Description 𝐷𝑒𝑠𝑑𝑒𝑡
Options
A. announced
D. hitB. suspend
C. safeguardAccording  to foreign  media  reports,  
Ukraine’s  energy  minister  <announced > 
that the country  will <suspend > 
electricity  exports  starting  on the 11th 
after Russian  missile  strikes  <hit> its 
energy  facilities,  in order  to 
<safeguard > the stability  of its domestic  
energy  system .Event Mention (Sentence 𝑆)
Please  select  all the event  triggers  you 
believe  are correct  …Question (Multiple -Choice)MiGUE -Reasoning
Coreference Subevent Causal Temporal
Options
A. after
D. contains B. beforeC. is contained
?Cross -Document
?Single -DocumentTask Description 𝐷𝑒𝑠𝑟𝑒𝑎
Relational Dimension 𝑅
Please  select  the relationship  you believe  is 
correct  from  the options  below  based  on the 
given  event  mentions  …Question (Single -Choice)Document(s) 𝐷,(𝐷′) & Event Pairs  𝑒,𝑒′MiGUE -Induction
Interval -based Temporal Ordering Causal Graph Construction
Event Trigger 
Set ℰ
Time𝑎𝑏𝑐𝑑𝑒
𝑑𝑠𝑡𝑎𝑟𝑡𝑑𝑒𝑛𝑑…𝑒𝑠𝑡𝑎𝑟𝑡 𝑒𝑒𝑛𝑑
𝑎
𝑏𝑐
𝑑
𝑒 𝑓
𝑔ℎ
Generative Result
𝑑𝑒𝑛𝑑, 𝑐𝑒𝑛𝑑, 𝑏𝑒𝑛𝑑, 𝑎𝑒𝑛𝑑, 𝑒𝑒𝑛𝑑 𝑑𝑠𝑡𝑎𝑟𝑡 , 𝑏𝑠𝑡𝑎𝑟𝑡 , 𝑐𝑠𝑡𝑎𝑟𝑡 , 𝑎𝑠𝑡𝑎𝑟𝑡 , 𝑒𝑠𝑡𝑎𝑟𝑡  Options
A. 𝑎, 𝑏, 𝑒, 𝑓, ℎ
B. 𝑎, 𝑐, 𝑑, 𝑓, ℎCorrect and complete 
causal chains
Distractors:
Irrelevant Links
Incomplete Path
Incorrect OrderTask Description 𝐷𝑒𝑠𝑜𝑟𝑑𝑒𝑟 Task Description  𝐷𝑒𝑠𝑐𝑎𝑢
Document Set  𝒟
Please  return  only the event  start  time sequence  
you derive  after reasoning  …Question (Generative)
Please  return  only the event  end time sequence  
you derive  after reasoning  …… select  all causal  chains  that you believe  are 
complete  and correct  …Question (Multiple -Choice)Event Trigger 
Set ℰ
With Causal 
Relationships
C. 𝑎, 𝑏, 𝑒, 𝑔
D. 𝑎, 𝑏, 𝑒, ℎ
E. 𝑎, 𝑐, 𝑑, 𝑒, 𝑓, ℎMiGUE -Prediction
LLM
Adversarial Strategies
Counterfactual Over -generalization
Temporal Trap Irrelevant Distraction
Disturbing 
Options
Extracted
A. Temporary  closure  measures  will be implemented  for road traffic  on 
the Crimean  Bridge .Task Description 𝐷𝑒𝑠𝑝𝑟𝑒𝑑
Document Sequence  𝒟 
with timestamps 𝒯 
TimeGeneration
Options
B. The Crimean  Bridge  will be permanently  closed  due to security  risks .
C. The Crimean  Bridge  will quickly  resume  full traffic .
D. Russia  and Ukraine  will reach  a new consensus  on the Black  Sea 
Grain  Initiative .Based  on the information  provided  in the documents,  which  of the following  
future  predictions  is most  likely  to be accurate?Question (Single -Choice)
Random Sampling
Figure 1: Overview of MiGUE-Pipeline and MiGUE-Bench.
Cross-Document Relation Annotation.For comparison, cross-
document relation annotation faces the challenges of relational spar-
sity and computational intractability. To address these challenges,
we propose a propagate-via-coreference strategy. For preparation,
we constrain the search space using three proximity heuristics,
includingtemporal proximity(where documents must fall within
a one-month timestamp window),semantic affinity(where event
mentions should exceed a semantic similarity threshold), andentity
overlap(where events should share common participants). Poten-
tially coreferential pairs are then validated using the same protocol
as in the single-document part. In the propagate-via-coreference
strategy, we densify the relation network by treating coreferential
events as bridges for propagation. By applying formal transitiv-
ity rules (e.g., if 𝑒1𝑐𝑜𝑟𝑒 𝑓= = = = = = =𝑒 2,𝑒1𝑐𝑎𝑢𝑠𝑒−−−−→𝑒 3, then𝑒2𝑐𝑎𝑢𝑠𝑒−−−−→𝑒 3), we
recover implicit dependencies across different documents. This sys-
tematic approach circumvents exhaustive pairing while improving
the global structural integrity of the event relation network.
3.4 Cluster Generation
Since documents that share identical real-world events are poten-
tially linked by their underlying narrative contents, we utilize event
coreference as a primary signal for document cluster generation,
supporting the data construction of downstream tasks that require
sufficient cross-document relations. Specifically, we construct a
document affinity graph whose vertices represent individual docu-
ments and edges denote the presence of coreferential event pairs.
To partition this graph into dense subgraphs, we employ the Leidencommunity detection algorithm [ 43], which isolates clusters with
maximum narrative coherence. By operating on these localized
subgraphs whose relational density is highest, we streamline the
discovery of closely-connected documents for more efficient data
construction of cross-document event-centric tasks.
4 MiGUE-Benchmark
In this section, we systematically delineate the construction of
MiGUE-Bench, including four core tasks. For each task , we provide
a rigorous definition involving task inputs and outputs, followed
by its data construction strategy with MiGUE-Pipeline.
4.1 MiGUE-Detection
Since the ability of precisely detecting events is the foundational
skill of event analysis, we devise the MiGUE-Detection task to mea-
sure LLMs’ performance on distinguishing factual event triggers.
Task Definition.Given a task description 𝐷𝑒𝑠 𝑑𝑒𝑡, a sentence 𝑆,
and a candidate set of event triggers C={𝑐1,𝑐2,...,𝑐 𝑟|3⩽𝑟⩽6}
where𝑟indicates the size of C, the model is required to provide all
the valid triggers that appear in𝑆within the candidate setC.
Data Construction.The data instances for this task are directly
acquired from the event annotation stage (Section 3.2). To further
improve the difficulty of this fundamental task, we incorporate ad-
versarial distractors into the options. Specifically, we inject typical
error types identified during event annotation of MiGUE-Pipeline
into candidate options for each instance. By integrating these fail-
ures modes as distractors, we aim to assess the fine-grained seman-
tic understanding ability of LLMs.

From Single- to Cross-Document: Benchmarking Multi-Granularity Event Analysis of Large Language Models SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia.
Table 3: Statistics of MiGUE-Bench including the number of
data instances (#Instance) and the average number of input
tokens (#Token) / options (#Option).
Task Subtask #Instance #Token #Option
Detection - 330 55.18 4.56
ReasoningIntraTemporal 300 796.84 4.00
Causal 300 937.31 4.00
Coreference 300 890.00 2.00
Subevent 300 770.58 3.00
CrossTemporal 300 1209.37 4.00
Causal 300 1202.18 4.00
Coreference 300 1202.24 2.00
Subevent 300 1141.26 3.00
InductionTemporal Order 150 671.77 -
Causal Graph 110 1431.27 4.48
Prediction - 300 2112.03 4.50
Overall - 3,290 1061.24 3.85
4.2 MiGUE-Reasoning
Inferring relationships between events within / across documents
is essential for event analysis. MiGUE-Reasoning focuses on evalu-
ating LLMs’ capacity of reasoning across different events.
Task Definition.Given the task description 𝐷𝑒𝑠 𝑟𝑒𝑎, the event pair
(𝑒,𝑒′)which comes from the document 𝐷/𝐷′, respectively, the tar-
get relational dimension R∈{𝑇𝑒𝑚𝑝𝑜𝑟𝑎𝑙,𝐶𝑎𝑢𝑠𝑎𝑙,𝑆𝑢𝑏𝑒𝑣𝑒𝑛𝑡,𝐶𝑜−
𝑟𝑒𝑓𝑒𝑟𝑒𝑛𝑐𝑒} , and a candidate set of relation labels C={𝑐 1,𝑐2,...,𝑐 𝑟|
2⩽𝑟⩽ 4}, the model is required to select the a label from Cthat
precisely characterizes the relationship within the corresponding
dimension. According to whether 𝐷and𝐷′are the same docu-
ment, this task can be further categorized into two subtasks, i.e.,
Reasoning-IntraandReasoning-Cross.
Data Construction.To ensure that our benchmark probes deep
reasoning over events rather than surface-level pattern matching,
we explicitly exclude pairs linked by overt linguistic cues, such as
direct temporal markers (afterward) or causal connectives (because).
We retain only those relations that necessitate contextual under-
standing and multi-hop reasoning. We also prioritize event pairs
that the effective LLMs (such as GPT-4o) initially misclassify to
further enhance the difficulty.
4.3 MiGUE-Induction
Global comprehension of event development necessitates induc-
ing the topological structure of event across documents. MiGUE-
Induction aims to measure the capacities of LLMs for structure
induction through two sub-tasks:Interval-based Temporal Ordering
andCausal Graph Construction.
4.3.1 Interval-based Temporal Ordering.Traditional temporal or-
dering often simplifies events into instantaneous points, relying
on binary before/after comparisons [ 47]. This task design fails to
capture the temporal duration and interval logic (e.g., overlap and
containment) inherently in real-world narratives. To address this
limitation, we decouple each event into itsstartandendtemporal
anchors for fine-grained ordering.
Task Definition.Given the task description 𝐷𝑒𝑠 𝑜𝑟𝑑𝑒𝑟 , a set of
event triggersE={𝑒 1,...,𝑒 𝑛|𝑛≥ 3}, and a set of documentsD={𝐷 1,...,𝐷 𝑚|𝑚≥ 2}where these events occur, the model
should output two distinct permutations of E, representing the
chronological order of event starts and event ends, respectively.
Data Construction.The data instances are derived from cross-
document temporal relations (Section 3.3) and document clusters
(Section 3.4) generated by MiGUE-Pipeline. We specifically leverage
the results of relation propagation, which yields dense temporal
networks across different documents. we prioritize clusters with
high relation density, ensuring that the selected events form an
interconnected network rather than isolated chains. Furthermore,
to prevent the models from relying on superficial pattern matching,
we remove absolute temporal markers (e.g., April 14, 2022) in the
data instances. This forces the model to induct the relative temporal
order from the narrative contents and implicit cues within the
provided documents.
4.3.2 Causal Graph Construction.Causality among events of dif-
ferent documents commonly manifests as a directed acyclic graph
(DAG), where events may have multiple preconditions or cause
several subsequent consequences. Since causality is essential for
structure induction of events, we design a task to assess the capa-
bility of constructing causal graphs.
Task Definition.Given the task description 𝐷𝑒𝑠 𝑐𝑎𝑢, a set of event
triggersE={𝑒 1,...,𝑒 𝑛|𝑛≥ 3}, a set of documents D={𝐷 1,...,𝐷 𝑚|
𝑚≥ 2}where these events occur, and a set of candidate options
C𝑠𝑖={𝑐 1,𝑐2,...,𝑐 𝑟|4⩽𝑟⩽ 6}, where each option corresponds
to a causal path⟨𝑒1→𝑒 2→···→𝑒 𝑛⟩, the model is required to
identify all the longest valid causal chains within the causal DAG.
Data Construction.The data instances are curated from causal
networks that are built based on causal relations (Section 3.3) and
cluster generation (Section 3.4). To acquire challenging evaluation
instances, we treat each maximal path in causal DAGs as a correct
option and design three types of distractors as disturbing options,
includingirrelevant links(i.e., chains containing event pairs with
no logical dependency),incomplete path(i.e., sub-paths that are
causally valid but fail to include all possible intermediate or terminal
events) , andincorrect order(i.e., chains where events are causally
related but presented in an incorrect logical order).
4.4 MiGUE-Prediction
The purpose of event analysis is not merely the retrospective un-
derstanding of what has occurred, but also the prediction of what
will follow. Thus, we devise the MiGUE-Prediction task as follows.
Task Definition.Given the the task description 𝐷𝑒𝑠 𝑝𝑟𝑒𝑑, a tempo-
rally ordered document sequence D={𝐷 1,...,𝐷 𝑚−1|3⩽𝑚⩽ 6}
with their timestamps, and a candidate set of events C={𝑐 1,...,𝑐 𝑟|
4⩽𝑟⩽ 5}, the model is required to select the event that occurs in
the correct terminal document𝐷 𝑚fromC.
Data Construction.Inspired by existing works [ 21,24,30], we
extract the document sequences from the clusters (Section 3.4) that
exhibit cross-document causal chains (Section 3.3). We further in-
crease difficulty by appending redundant context—chronologically
consistent but irrelevant documents from the same cluster—to dis-
courage shallow pattern matching. For option design, we devise the
correct option as a paraphrased version of the event occurring in
𝐷𝑚with specific details removed to prevent trivial lexical shortcuts.

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia. Tao Wen et al.
Table 4: Accuracy (Acc.) and Micro-F1 of different LLMs and RAG methods on MiGUE-Benchmark.
Granularity Single-Document Cross-Document
Task Detection Reasoning Induction Prediction
Subtask -Reasoning-Intra Reasoning-Cross Temporal OrderCausal Graph -
Temp. Cau. Coref. Sub. Temp. Cau. Coref. Sub. Start End
Metric Micro-F1 Acc. Acc. Acc. Acc. Acc. Acc. Acc. Acc. Acc. Acc. Acc. Acc.
Closed-Source LLMs
GPT-5.2-Pro 0.8398 0.6600 0.63730.97350.8033 0.6786 0.64080.96360.8188 0.84120.8333 0.4375 0.4231
Gemini-3-Pro 0.7151 0.6867 0.6296 0.9500 0.7801 0.7967 0.5571 0.9100 0.8034 0.6623 0.3775 0.2159 0.5814
Claude-4.5-Opus 0.8552 0.71000.67780.97000.8289 0.77810.70480.9333 0.8421 0.87420.8079 0.5814 0.5544
Claude-4.5-Haiku 0.6101 0.6767 0.6000 0.9300 0.7000 0.4633 0.4238 0.7400 0.7164 0.4437 0.1722 0.1932 0.4452
Qwen3-Max 0.6543 0.6967 0.6037 0.9050 0.8022 0.7067 0.6000 0.7700 0.8099 0.4437 0.1987 0.2727 0.4585
Open-Source LLMs
DeepSeek-V3.2 0.8168 0.6600 0.6630 0.9625 0.7956 0.6732 0.6286 0.85670.8480 0.8211 0.6300 0.2841 0.5210
GLM-4.7 0.8133 0.6933 0.6167 0.9633 0.7766 0.7066 0.5333 0.8600 0.8300 0.8067 0.5600 0.2727 0.4867
Kimi-K2 0.5894 0.5900 0.5630 0.9200 0.7044 0.6167 0.5143 0.8400 0.7807 0.3576 0.1987 0.2614 0.5150
Qwen3-8B 0.6138 0.4000 0.4599 0.8677 0.5133 0.4900 0.3390 0.7333 0.6170 0.1391 0.0795 0.1364 0.3355
Qwen3-30B 0.6298 0.4100 0.4926 0.6211 0.6022 0.5372 0.5095 0.6200 0.7895 0.1722 0.0530 0.2500 0.3522
Qwen3-235B 0.6115 0.5600 0.5481 0.8725 0.6133 0.5567 0.5286 0.7667 0.8362 0.3510 0.1258 0.1705 0.4352
Llama-3.1-7B 0.4959 0.2326 0.2185 0.3296 0.2238 0.2167 0.1667 0.1800 0.1316 0.0596 0.0132 0.0341 0.3654
RAG Methods based on LLMs
UltraRAG (w/ Gemini-3-Pro) 0.7457 0.6939 0.6148 0.9525 0.8263 0.80000.5333 0.8900 0.8304 0.6954 0.3841 0.2045 0.6246
UltraRAG (w/ Qwen3-Max) 0.6936 0.7000 0.6074 0.9100 0.7911 0.7067 0.6190 0.7900 0.7953 0.4768 0.2185 0.2955 0.4751
UltraRAG (w/ Qwen3-8B) 0.6287 0.3667 0.4333 0.8690 0.5178 0.5047 0.3301 0.6900 0.5409 0.1457 0.0728 0.1136 0.3522
LightRAG (w/ Gemini-3-Pro) 0.7137 0.7100 0.6181 0.9550 0.7328 0.7500 0.5571 0.8467 0.7895 0.7033 0.4196 0.2067 0.6185
LightRAG (w/ Qwen3-Max) 0.6465 0.71810.6132 0.8922 0.7497 0.6967 0.6238 0.7933 0.7924 0.4834 0.2318 0.2955 0.4950
LightRAG (w/ Qwen3-8B) 0.5851 0.3893 0.4481 0.8647 0.5145 0.4867 0.4012 0.6467 0.5380 0.1126 0.0397 0.1277 0.3445
Following existing works [ 16], we generate disturbing options us-
ing four adversarial strategies, includingcounterfactual(i.e., events
that are logically opposite to the true outcome),overgeneralization
(i.e., plausible but exaggerated conclusions that transcend the evi-
dence),temporal trap(i.e., past events from the document sequence),
andirrelevant distraction(i.e., fabricated but contextually consistent
events unsupported by the document sequence).
4.5 Quality Control
To ensure the quality of MiGUE-Bench, we implement a multi-stage
validation protocol. Firstly, we automatically filter all the instances
to maintain a balanced distribution across document sources, event
densities, and difficulties. Then, we thoroughly detect and exclude
offensive, biased, or harmful content, ensuring alignment with ethi-
cal standards. Finally, we conduct manual review on all the instances
to verify the correctness of answers and eliminate ambiguity in all
the options. The statistics of the final dataset are shown in Table 3.
5 Experiment
5.1 Setting
Source Data.As MiGUE-Pipeline is a scalable event data construc-
tion pipeline, our corpus is mainly drawn from two sources: (1)
high-quality open-source news document datasets [ 31], and (2)
more than 9,000 news reports collected from Chinese official and
mainstream media outlets between 2022 and 2024. During data
collection, we performed a preliminary filtering based on news
tags to ensure that the corpus covers major events from this period
as comprehensively as possible. All the corpora are subsequentlyprocessed by our MiGUE-Pipeline to automatically construct event
data2.
Base Model Selection.To comprehensively evaluate the per-
formance of LLMs, we select two lines of work, i.e., LLMs and
RAG methods. For LLMs, we involve mainstream closed-source
(including GPT-5.2-Pro [ 37], Gemini-3-Pro [ 15], Claude-4.5-Opus
[2], Claude-4.5-Haiku [ 1], and Qwen3-Max [ 38]) and open-source
LLMs (including DeepSeek-V3.2 [ 10], GLM-4.7 [ 42], Kimi-K2 [ 22],
Qwen-3-8B/30B/235B [ 38], and Llama-3.1-7B [ 28]) as representa-
tives, covering the models with different scales, families, and capac-
ities. Furthermore, we also adopt two mainstream RAG frameworks
including LightRAG [ 17] and UltraRAG [ 7]. The retrieval corpora
contain 5,000 documents from the document filtering stage (Section
3.1) of MiGUE-Pipeline3. In our experiment, we utilize Gemini-3-
Pro, Qwen3-Max, and Qwen3-8B as base models for RAG methods.
Evaluation Metric.For MiGUE-Detection, we adopt Micro-F1 to
jointly reflect the precision and recall of this multi-choice selection
task. As for the other tasks, accuracy is adopted for measurement.
5.2 Main Result
The results in Table 4 show that closed-source LLMs mostly out-
perform open-source ones, especially on cross-document event
analysis tasks, demonstrating the effectiveness of state-of-the-art
proprietary models. Among them, GPT-5.2-Pro and Claude-4.5-
Opus achieve the best performance across nearly all the tasks, while
2Since the source data collected is initially Chinese, we use MiGUE-Pipeline to con-
struct the Chinese dataset. Nevertheless, MiGUE-Pipeline is language-agnostic and
can be also applied to other languages with corresponding prompts.
3In the task of MiGUE-Prediction, we restrict the documents to those occurring prior
to the predicted event, preventing data leakage.

From Single- to Cross-Document: Benchmarking Multi-Granularity Event Analysis of Large Language Models SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia.
0 3 4 6 8 9 10 12
Number of Retrieved Documents (n)0.20.30.40.50.6Score (Accuracy)
Qwen3-Max (Start)
Qwen3-Max (End)Kimi-k2 (Start)
Kimi-k2 (End)
Figure 2: Accuracy on MiGUE-Induction (Temporal Order)
with different numbers of retrieved documents.
Deepseek-V3.2 and GLM-4.7 rank highest among open-source LLMs.
We also have other interesting findings:
RAG can generally improve performance across the four
tasks, but the gains are unstable.Specifically, RAG benefits
strong LLMs such as Gemini-3-Pro on most of the tasks. However,
for weaker LLMs like Qwen-3-8B, the noisy retrieved documents
may degrade the original performance in turn. How to design spe-
cific RAG strategies for event analysis still needs further study.
At the MiGUE-Reasoning task, temporal and subevent rela-
tions generally benefit from cross-document settings, whereas
coreference and causal reasoning exhibit the opposite trend.
We conjecture that multi-document contexts may introduce addi-
tional temporal signals (e.g., explicit timestamps), which can be
easily captured by LLMs. Consequently, temporal-related relations
benefit from these additional signals. In contrast, for coreference
and causal reasoning, different documents may describe the same
event with distinct perspectives. Also, temporally successive events
are not necessarily causally related. Such heterogeneity may in-
troduces noise and ambiguity, leading to degraded performance in
cross-document settings.
At the MiGUE-Induction task, LLMs are sensitive to start
times of events but exhibit limited understanding of their
time spans.We observe from the subtask of interval-based tem-
poral ordering that most LLMs perform significantly worse on
end-time ordering than on start-time one, with the accuracy often
dropping by nearly half. This suggests that identifying relative event
start times is much simpler for current LLMs, whereas reasoning
about event durations remains challenging.
Causal tasks remain challenging for current LLMs.Most of
these LLMs achieve unsatisfactory performance on the tasks of
causal reasoning and causal graph induction. We conjecture that
one reason is the misalignment between model-internal causal rep-
resentations and human causal understanding. Moreover, current
causal definitions in event analysis are often confined to language-
based descriptions and lack rigorous mathematical formalization.
We leave the exploration of more fine-grained probabilistic causal
modeling at the schema level instead of merely textual logical for-
mulations in event analysis as important future work.
5.3 Detailed Analysis
Analysis on the Number of Retrieved Documents.To further
analyze this effect of additional relevant documents on temporal
reasoning, we evaluate Qwen3-Max and Kimi-K2 under UltraRAG
0-300 300-900 900-1500 1500-2100 2100-3000
Instruction Length0.00.51.0AccuracyQwen3-8B Qwen3-30B Qwen3-235BFigure 3: Accuracy on MiGUE-Reasoning (Subevent) with
different lengths of input instructions.
with different numbers of retrieved documents. Figure 2 shows
that both models exhibit a rise–fall–stabilization pattern for start-
ordering and end-ordering accuracies as the number of retrieved
documents increases. Moderate retrieved documents improve per-
formance, whereas excessive documents degrade accuracy, possibly
due to noise accumulation and context dilution. End-ordering per-
formance consistently remains lower than start-ordering, revealing
the deficiencies of LLMs in modeling event time spans.
Analysis on the Length of Input Instructions.We further ana-
lyze the scaling behavior of LLMs with different lengths of input
instructions on subevent reasoning. As shown in Figure 3, Qwen3-
235B exhibits steady performance gains as the instruction length
increases, indicating strong capabilities of long-context understand-
ing. For comparison, Qwen3-30B/8B also benefits from longer in-
puts but reaches its peak in the 1500–2100 / 900–1500 range, fol-
lowed by a slight decline. This suggest that larger models benefit
from richer contextual evidence, whereas smaller models saturate
earlier and degrade under extended inputs.
6 Conclusion
In this work, we present a comprehensive benchmark called MiGUE-
Bench for multi-granularity event analysis, which covers the tasks
ranging from fine-grained event detection and reasoning to cross-
document event structure induction and forecasting, together with
an LLM-driven automatic pipeline named MiGUE-Pipeline for scal-
able event data construction. Through extensive experiments on
state-of-the-art LLMs and RAG methods, we show that despite their
strong general capabilities, these models still exhibit notable limi-
tations in multi-granularity event analysis. We hope that MiGUE-
Bench and MiGUE-Pipeline can serve as standardized testbeds for
diagnosing the capability boundaries of LLMs in event analysis and
inspire future advances in this research field.
Acknowledgments
This work was supported by Noncommunicable Chronic Diseases-
National Science and Technology Major Project (No. 2023ZD0501806),
Sichuan Science and Technology Program (No. 2025ZNSFSC1488)
, Fundamental Research Funds for the Central Universities (No.
ZYGX2025XJ041), and CIPS-SMP-Zhipu Large Model Fund (No.
CIPS-SMP20250314).
References
[1]Anthropic. 2025. Introducing Claude Haiku 4.5. https://www.anthropic.com/
news/claude-haiku-4-5. Accessed: 2026-04-25.

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia. Tao Wen et al.
[2]Anthropic. 2025. Introducing Claude Opus 4.5. https://www.anthropic.com/
news/claude-opus-4-5. Accessed: 2026-04-25.
[3]Michael Bugert, Nils Reimers, and Iryna Gurevych. 2021. Generalizing Cross-
Document Event Coreference Resolution Across Multiple Corpora.Computa-
tional Linguistics47, 3 (2021), 575–614.
[4]Chunkit Chan, Cheng Jiayang, Weiqi Wang, Yuxin Jiang, Tianqing Fang, Xin Liu,
and Yangqiu Song. 2024. Exploring the Potential of ChatGPT on Sentence Level
Relations: A Focus on Temporal, Causal, and Discourse Relations. InFindings of
the Association for Computational Linguistics: EACL 2024. 684–721.
[5]Meiqi Chen, Yubo Ma, Kaitao Song, Yixin Cao, Yan Zhang, and Dongsheng Li.
2024. Improving large language models in event relation logical prediction. In
Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). 9451–9478.
[6]Muhao Chen, Hongming Zhang, Qiang Ning, Manling Li, Heng Ji, Kathleen
McKeown, and Dan Roth. 2021. Event-centric natural language processing. In
Proceedings of the 59th Annual Meeting of the Association for Computational Lin-
guistics and the 11th International Joint Conference on Natural Language Processing:
Tutorial Abstracts. 6–14.
[7]Yuxuan Chen, Dewen Guo, Sen Mei, Xinze Li, Hao Chen, Yishan Li, Yixuan Wang,
Chaoyue Tang, Ruobing Wang, Dingjun Wu, Yukun Yan, Zhenghao Liu, Shi Yu,
Zhiyuan Liu, and Maosong Sun. 2025. UltraRAG: A Modular and Automated
Toolkit for Adaptive Retrieval-Augmented Generation.CoRRabs/2504.08761
(2025).
[8]Yubo Chen, Liheng Xu, Kang Liu, Daojian Zeng, and Jun Zhao. 2015. Event
extraction via dynamic multi-pooling convolutional neural networks. InProceed-
ings of the 53rd Annual Meeting of the Association for Computational Linguistics
and the 7th International Joint Conference on Natural Language Processing (Volume
1: Long Papers). 167–176.
[9]Milind Choudhary and Xinya Du. 2024. QAEVENT: Event Extraction as Question-
Answer Pairs Generation. InFindings of the Association for Computational Lin-
guistics: EACL 2024. 1860–1873.
[10] DeepSeek-AI. 2025. DeepSeek-V3.2: Pushing the Frontier of Open Large Language
Models.CoRRabs/2512.02556 (2025).
[11] George R. Doddington, Alexis Mitchell, Mark A. Przybocki, Lance A. Ramshaw,
Stephanie M. Strassel, and Ralph M. Weischedel. 2004. The Automatic Content
Extraction (ACE) Program - Tasks, Data, and Evaluation. InProceedings of the
Fourth International Conference on Language Resources and Evaluation. 837–840.
[12] Chuang Fan, Daoxing Liu, Libo Qin, Yue Zhang, and Ruifeng Xu. 2022. Towards
Event-level Causal Relation Identification. InSIGIR ’22: The 45th International
ACM SIGIR Conference on Research and Development in Information Retrieval.
1828–1833.
[13] John Foley, Michael Bendersky, and Vanja Josifovski. 2015. Learning to Extract
Local Events from the Web. InProceedings of the 38th International ACM SIGIR
Conference on Research and Development in Information Retrieval. 423–432.
[14] Jie Gong, Biaoshuai Zheng, and Qiwang Hu. 2025. EventRelBench: A Comprehen-
sive Benchmark for Evaluating Event Relation Understanding in Large Language
Models. InFindings of the Association for Computational Linguistics: EMNLP 2025.
9084–9099.
[15] Google. 2025. Gemini 3: A new era of intelligence. https://blog.google/products-
and-platforms/products/gemini/gemini-3/. Accessed: 2026-04-25.
[16] Yong Guan, Hao Peng, Xiaozhi Wang, Lei Hou, and Juanzi Li. 2024. OpenEP:
Open-Ended Future Event Prediction.CoRRabs/2408.06578 (2024).
[17] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG:
Simple and Fast Retrieval-Augmented Generation. InFindings of the Association
for Computational Linguistics: EMNLP 2025. 10746–10761.
[18] Yu Hong, Tongtao Zhang, Tim O’Gorman, Sharone Horowit-Hendler, Heng Ji, and
Martha Palmer. 2016. Building a Cross-document Event-Event Relation Corpus.
InProceedings of the 10th Linguistic Annotation Workshop held in conjunction with
ACL 2016. 1–6.
[19] Zhilei Hu, Zixuan Li, Xiaolong Jin, Long Bai, Jiafeng Guo, and Xueqi Cheng.
2025. Large Language Model-Based Event Relation Extraction with Rationales.
InProceedings of the 31st International Conference on Computational Linguistics.
7484–7496.
[20] Kuan-Hao Huang, I-Hung Hsu, Tanmay Parekh, Zhiyu Xie, Zixuan Zhang, Prem
Natarajan, Kai-Wei Chang, Nanyun Peng, and Heng Ji. 2024. TextEE: Benchmark,
Reevaluation, Reflections, and Future Challenges in Event Extraction. InFindings
of the Association for Computational Linguistics: ACL 2024. 12804–12825.
[21] Zijie Huang, Jeehyun Hwang, Junkai Zhang, Jinwoo Baik, Weitong Zhang, Do-
minik Wodarz, Yizhou Sun, Quanquan Gu, and Wei Wang. 2024. Causal graph
ode: Continuous treatment effect modeling in multi-agent dynamical systems. In
Proceedings of the ACM Web Conference 2024. 4607–4617.
[22] Kimi Team. 2025. Kimi K2: Open Agentic Intelligence.CoRRabs/2507.20534
(2025).
[23] Bobo Li, Xudong Han, Jiang Liu, Yuzhe Ding, Liqiang Jing, Zhaoqi Zhang, Jinheng
Li, Xinya Du, Fei Li, Meishan Zhang, Min Zhang, Aixin Sun, Philip S. Yu, and
Hao Fei. 2025. Event Extraction in Large Language Model.CoRRabs/2512.19537
(2025).[24] Manling Li, Sha Li, Zhenhailong Wang, Lifu Huang, Kyunghyun Cho, Heng Ji,
Jiawei Han, and Clare Voss. 2021. The Future is not One-dimensional: Complex
Event Schema Induction by Graph Modeling for Event Prediction. InProceedings
of the 2021 Conference on Empirical Methods in Natural Language Processing.
5203–5215.
[25] Rui Li, Wenlin Zhao, Cheng Yang, and Sen Su. 2022. A Dual-Expert Framework
for Event Argument Extraction. InThe 45th International ACM SIGIR Conference
on Research and Development in Information Retrieval. 1110–1121.
[26] Jinzhi Liao, Xiang Zhao, Xinyi Li, Lingling Zhang, and Jiuyang Tang. 2021. Learn-
ing Discriminative Neural Representations for Event Detection. InThe 44th
International ACM SIGIR Conference on Research and Development in Information
Retrieval. 644–653.
[27] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang
Zhu. 2023. G-Eval: NLG Evaluation using Gpt-4 with Better Human Alignment.
InProceedings of the 2023 Conference on Empirical Methods in Natural Language
Processing. 2511–2522.
[28] Llama Team. 2024. The Llama 3 Herd of Models.CoRRabs/2407.21783 (2024).
[29] Chenwei Lou, Jun Gao, Changlong Yu, Wei Wang, Huan Zhao, Weiwei Tu, and
Ruifeng Xu. 2022. Translation-Based Implicit Annotation Projection for Zero-Shot
Cross-Lingual Event Argument Extraction. InThe 45th International ACM SIGIR
Conference on Research and Development in Information Retrieval. 2076–2081.
[30] Yunshan Ma, Chenchen Ye, Zijian Wu, Xiang Wang, Yixin Cao, and Tat-Seng
Chua. 2023. Context-aware event forecasting via graph disentanglement. In
Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and
Data Mining. 1643–1652.
[31] Yunshan Ma, Chenchen Ye, Zijian Wu, Xiang Wang, Yixin Cao, Liang Pang, and
Tat-Seng Chua. 2023. Structured, Complex and Time-complete Temporal Event
Forecasting.CoRRabs/2312.01052 (2023).
[32] Anne-Lyse Minard, Manuela Speranza, Eneko Agirre, Itziar Aldabe, Marieke
Van Erp, Bernardo Magnini, German Rigau, and Ruben Urizar. 2015. Semeval-
2015 task 4: Timeline: Cross-document event ordering. Inproceedings of the 9th
International Workshop on Semantic Evaluation (SemEval 2015). 778–786.
[33] Paramita Mirza and Sara Tonelli. 2014. An Analysis of Causality between Events
and its Relation to Temporal Information. InProceedings of COLING 2014, the 25th
International Conference on Computational Linguistics: Technical Papers. 2097–
2106.
[34] Nishanth Nakshatri, Siyi Liu, Sihao Chen, Dan Roth, Dan Goldwasser, and Daniel
Hopkins. 2023. Using LLM for improving key event discovery: Temporal-guided
news stream clustering with event summaries. InFindings of the Association for
Computational Linguistics: EMNLP 2023. 4162–4173.
[35] Thien Huu Nguyen, Kyunghyun Cho, and Ralph Grishman. 2016. Joint event
extraction via recurrent neural networks. InProceedings of the 2016 conference of
the North American chapter of the association for computational linguistics: human
language technologies. 300–309.
[36] Thien Huu Nguyen and Ralph Grishman. 2018. Graph Convolutional Networks
With Argument-Aware Pooling for Event Detection. InProceedings of the Thirty-
Second AAAI Conference on Artificial Intelligence. 5900–5907.
[37] OpenAI. 2026. OpenAI GPT-5 System Card.CoRRabs/2601.03267 (2026).
[38] Qwen Team. 2025. Qwen3 Technical Report.CoRRabs/2505.09388 (2025).
[39] Rashmi Sankepally. 2019. Event Information Retrieval from Text. InProceedings
of the 42nd International ACM SIGIR Conference on Research and Development in
Information Retrieval(Paris, France)(SIGIR’19). 1447.
[40] Lei Sha, Feng Qian, Baobao Chang, and Zhifang Sui. 2018. Jointly Extracting
Event Triggers and Arguments by Dependency-Bridge RNN and Tensor-Based
Argument Interaction. InProceedings of the Thirty-Second AAAI Conference on
Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial
Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in
Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018.
5916–5923.
[41] Zhengwei Tao, Zhi Jin, Yifan Zhang, Xiancai Chen, Haiyan Zhao, Jia Li, Bin
Liang, Chongyang Tao, Qun Liu, and Kam-Fai Wong. 2025. A Comprehensive
Evaluation on Event Reasoning of Large Language Models. InThe Thirty-Ninth
AAAI Conference on Artificial Intelligence. 25273–25281.
[42] Team GLM. 2024. ChatGLM: A Family of Large Language Models from GLM-130B
to GLM-4 All Tools.CoRRabs/2406.12793 (2024).
[43] Vincent A Traag, Ludo Waltman, and Nees Jan Van Eck. 2019. From Louvain to
Leiden: guaranteeing well-connected communities.Scientific reports9, 1 (2019),
1–12.
[44] Xiaozhi Wang, Yulin Chen, Ning Ding, Hao Peng, Zimu Wang, Yankai Lin, Xu
Han, Lei Hou, Juanzi Li, Zhiyuan Liu, Peng Li, and Jie Zhou. 2022. MAVEN-ERE:
A Unified Large-scale Dataset for Event Coreference, Temporal, Causal, and
Subevent Relation Extraction. InProceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing. 926–941.
[45] Xiaozhi Wang, Ziqi Wang, Xu Han, Wangyi Jiang, Rong Han, Zhiyuan Liu, Juanzi
Li, Peng Li, Yankai Lin, and Jie Zhou. 2020. MAVEN: A Massive General Domain
Event Detection Dataset. InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP). 1652–1671.

From Single- to Cross-Document: Benchmarking Multi-Granularity Event Analysis of Large Language Models SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia.
[46] Zairun Yang, Yilin Wang, Zhengyan Shi, Yuan Yao, Lei Liang, Keyan Ding, Emine
Yilmaz, Huajun Chen, and Qiang Zhang. 2025. EventRAG: Enhancing LLM
Generation with Event Knowledge Graphs. InProceedings of the 63rd Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).
16967–16979.[47] Zhihan Zhang, Yixin Cao, Chenchen Ye, Yunshan Ma, Lizi Liao, and Tat-Seng
Chua. 2024. Analyzing Temporal Complex Events with Large Language Models?
A Benchmark towards Temporal, Long Context Understanding. InProceedings of
the 62nd Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers). 1588–1606.
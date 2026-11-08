# Weaver: Weaving Preference Evidence through Incremental Reasoning for Personalized Retrieval-Augmented Generation

**Authors**: Linyao Zheng, Zhifang Mao, Xuhang Shi

**Published**: 2026-11-04 03:48:15

**PDF URL**: [https://arxiv.org/pdf/2611.05900v1](https://arxiv.org/pdf/2611.05900v1)

## Abstract
Personalized retrieval-augmented generation requires extracting and applying user preferences from extended interaction histories. This task is challenging because preference evidence is scattered across past conversations and remains bound to specific temporal and situational contexts. Consequently, isolated fragments rarely provide complete support for a response, while preferences expressed under different conditions are easily conflated. To address these limitations, we propose Weaver, a framework that incrementally weaves memory-conditioned retrieval trajectories with compositional preference reasoning for personalized responses. Weaver operates on a Dual-Layer Retrieval and Reasoning Graph that couples an upper Intent Dependency Graph with a lower Memory-Conditioned Retrieval Trajectory Graph. Dependency Information Gap Tracking represents the information needed to answer the current request and its prerequisite dependencies in the upper graph. Memory-Conditioned Incremental Retrieval uses established premises and clues from retrieved conversations to locate missing evidence, recording the resulting retrieval trajectories in the lower graph. Scope-Aware Compositional Reasoning jointly interprets the accumulated evidence to determine which preferences apply to the request. It identifies complementary and alternative support and distinguishes conflicting claims within the same temporal and situational context. These conclusions iteratively update task premises and remaining information gaps, guiding subsequent retrieval to establish a coherent contextual basis for the final response. Experiments on personalized RAG benchmarks, particularly PersonaMem, demonstrate that Weaver outperforms leading baselines in both evidence completeness and answer quality.

## Full Text


<!-- PDF content starts -->

Weaver: Weaving Preference Evidence through Incremental
Reasoning for Personalized Retrieval-Augmented Generation
Linyao Zheng
Xi’an Jiaotong University
Xi’an, ChinaZhifang Mao
Beijing University of Posts and
Telecommunications
Beijing, ChinaXuhang Shi
Beijing University of Posts and
Telecommunications
Beijing, China
Abstract
Personalized retrieval-augmented generation requires extracting
and applying user preferences from extended interaction histories.
This task is challenging because preference evidence is scattered
across past conversations and remains bound to specific temporal
and situational contexts. Consequently, isolated fragments rarely
provide complete support for a response, while preferences ex-
pressed under different conditions are easily conflated. To address
these limitations, we propose Weaver, a framework that incremen-
tally weaves memory-conditioned retrieval trajectories with com-
positional preference reasoning for personalized responses. Weaver
operates on a Dual-Layer Retrieval and Reasoning Graph that cou-
ples an upper Intent Dependency Graph with a lower Memory-
Conditioned Retrieval Trajectory Graph. Dependency Information
Gap Tracking represents the information needed to answer the
current request and its prerequisite dependencies in the upper
graph. Memory-Conditioned Incremental Retrieval uses established
premises and clues from retrieved conversations to locate missing
evidence, recording the resulting retrieval trajectories in the lower
graph. Scope-Aware Compositional Reasoning jointly interprets the
accumulated evidence to determine which preferences apply to the
request. It identifies complementary and alternative support and
distinguishes conflicting claims within the same temporal and situ-
ational context. These conclusions iteratively update task premises
and remaining information gaps, guiding subsequent retrieval to
establish a coherent contextual basis for the final response. Experi-
ments on personalized RAG benchmarks, particularly PersonaMem,
demonstrate that Weaver outperforms leading baselines in both
evidence completeness and answer quality. Specifically, Weaver
achieves accuracies of 86.42% on PersonaMem and 91.10% on PrefE-
val, exceeding the strongest evaluated baselines by 16.01 and 7.56
percentage points, respectively.
CCS Concepts
•Information systems →Personalization;Information retrieval;
•Computing methodologies→Natural language generation.
Keywords
Personalized RAG, user preferences, long-term memory, composi-
tional reasoning, incremental reasoning
ACM Reference Format:
Linyao Zheng, Zhifang Mao, and Xuhang Shi. 2027. Weaver: Weaving Pref-
erence Evidence through Incremental Reasoning for Personalized Retrieval-
Augmented Generation. InProceedings of The ACM Web Conference 2027
(WWW ’27).ACM, New York, NY, USA, 12 pages.
WWW ’27, Dublin, Ireland
2027.1 Introduction
Personalization enables Web systems to adapt information and ser-
vices to users’ current needs [ 1]. Large language models extend this
capability to generating responses tailored to individual users [ 2–
4]. Interaction histories provide preference evidence that is often
absent from the current request [ 5]. LLM-based Web agents use
this evidence to interpret instructions and generate personalized
responses [ 6]. As histories grow, conditioning on the full history
introduces irrelevant context and makes preference following more
difficult [ 7,8]. Personalized retrieval-augmented generation (RAG)
addresses this problem by selecting relevant historical records to in-
form the current response [ 9–14]. Selective access focuses the model
on potentially useful preference evidence [ 15,16]. However, rele-
vance alone does not establish that the retrieved records sufficiently
support a preference applicable to the current request [17, 18].
Figure 1 illustrates two difficulties in deriving applicable pref-
erences from conversation histories. First, preference evidence is
fragmented across conversations [ 19]. A retrieved fragment often
supplies partial support together with a clue for locating the miss-
ing information (Figure 1, I). Second, interpreting these fragments
requires their temporal and situational scope. An explicit update
changes the applicability of an earlier preference [ 20]. A prefer-
ence expressed for one situation does not necessarily hold for the
current request [ 21] (Figure 1, II). These difficulties connect mem-
ory discovery with preference interpretation. Reading the available
fragments reveals what remains unresolved, and further retrieval
supplies evidence for a more complete assessment of the applicable
preference [22, 23].
We proposeWeaver, a personalized RAG framework built on
aDual-Layer Retrieval and Reasoning Graph.Dependency
Information Gap Trackingmaintains unresolved subquestions
and their prerequisite dependencies in the upper graph. Each ex-
ecutable information need guidesMemory-Conditioned Incre-
mental Retrieval, which follows established premises and clues
from retrieved conversations in the lower graph.Scope-Aware
Compositional Reasoninginterprets the accumulated fragments
within their temporal and situational scopes. Complementary frag-
ments form joint support, while independently sufficient evidence
provides alternative support. The module also assesses conflicting
claims within overlapping scopes. New evidence updates preference
conclusions and the corresponding premises in the upper graph.
Remaining information gaps guide subsequent retrieval.
Our contributions are summarized as follows.
•Incremental Memory Discovery.We introduce a dual-
layer graph that connects unresolved information needs
to memory-conditioned retrieval trajectories. Established
arXiv:2611.05900v1  [cs.IR]  4 Nov 2026

WWW ’27, May 10–14, 2027, Dublin, Ireland Linyao Zheng, Zhifang Mao, and Xuhang Shi
M₁Recent
 This trip is solo
M₂Recent
updateFrom now on
solo vacations quiet
with friends still lively
M₃Last year
 Usually prefer livelyplaces.
IMissing Relevant Memories
M₁Found M₂Retrieve next
solo solo quietcue: solo
&
joint support
Quiet for
this trip
?Stop after M₁
Preference unresolved
IIMisapplying Past Preferences
M₁
solo+M₃ (old)
livelyIgnore
update
×Lively hotel
M₂ supersedes M₃
for solo trips
M₁
solo+M₂ (new)
quiet
Quiet hotel
Any hotel recommendations for a seaside getaway?
Figure 1 Two difficulties in preference inference. I) Missing
Relevant Memories. Stopping at 𝑀1leaves the preference
unresolved, while retrieving 𝑀2completes the support. II)
Misapplying Past Preferences. Applying the older 𝑀3ignores
the solo-trip update in 𝑀2and yields an unsuitable recom-
mendation.
premises and source-text clues guide subsequent searches
for missing evidence.
•Scope-Aware Compositional Reasoning.We represent
joint evidential support together with its contextual appli-
cability. Reassessing this support as evidence accumulates
updates preference conclusions and the information needs
they leave unresolved.
•Experimental Evaluation.Weaver achieves 86.42 ACC on
PersonaMem [ 24] and 91.10 ACC on PrefEval [ 7], improvingover the strongest evaluated baselines by 16.01 and 7.56
percentage points, respectively.
2 Related Work
Structured Memory for Personalization.Personalized RAG re-
trieves user-specific records to condition generation. LaMP studies
retrieval from personal profiles, including recency-aware selec-
tion [ 9]. Subsequent work optimizes personal-document retrieval
using feedback from downstream generation [ 10]. Structured mem-
ory further represents connections within user histories. EMG-RAG
organizes personal information in an editable memory graph [ 11],
while MemWeaver combines behavioral and cognitive memories to
capture temporal changes and semantic relationships [ 12]. These
representations support access to historical information, yet a
stored relation does not necessarily provide all the evidence re-
quired for a specific request. Interpreting a retrieved fragment can
depend on contextual information in another conversation. That
interpretation exposes missing premises and search clues that were
unavailable to the initial query.
Iterative Retrieval for Personalization.Building on stored mem-
ory representations, iterative retrieval uses intermediate results to
determine subsequent memory access. RF-Mem selects between di-
rect retrieval and iterative expansion in embedding space [ 13]. PGR
forms retrieval probes through prospective reasoning and refines
later probes with retrieved facts [ 25]. PR2learns policies that inter-
leave reasoning with retrieval from personal context [ 14]. Weaver
examines how retrieved evidence establishes a preference applica-
ble to the current request. It links unresolved information needs to
the premises used by each retrieval step. Joint interpretation checks
how fragments support a conclusion within its temporal and sit-
uational scope. The resulting conclusions update the information
needs that guide subsequent searches.
3 Method
As illustrated in Figure 2, Weaver operates on a Dual-Layer Re-
trieval and Reasoning Graph. The upper Intent Dependency Graph
represents task dependencies and tracks unresolved information
needs, while the lower Memory-Conditioned Retrieval Trajectory
Graph records how established premises and source-text clues guide
incremental memory discovery. Scope-Aware Compositional Rea-
soning jointly interprets the retrieved evidence and updates task
conclusions and information gaps, guiding subsequent retrieval.
Algorithms 1–3 summarize this cycle. After initialization, avail-
able evidence is interpreted before another query is issued. Gap
tracking identifies an executable need, retrieval adds evidence, and
composition updates conclusions and local gaps. These updates
return to gap tracking, allowing subquestion results to support the
original request. The response requirements are met only when the
required outputs have sufficient applicable support and no unre-
solved conflict; an empty retrieval frontier alone does not establish
completion.
3.1 Dependency Information Gap Tracking
In the upper graph, Weaver uses an LLM to decompose the current
request into dependent subquestions and tracks their unresolved
information needs as evidence accumulates.

Weaver: Incremental Preference Reasoning for Personalized RAG WWW ’27, May 10–14, 2027, Dublin, Ireland
Expand
subproblem
•••
How should I choose a hotel
for this seaside trip?LLM decomposition
1Dependency Information Gap Tracking
Intent Dependency Graph
2Memory-Conditioned
Incremental Retrieval
Memory-Conditioned Retrieval Trajectory Graph3Scope-Aware
Compositional ReasoningTravel party?
Solo?Direct preference?
Unresolved
Preference from
past stay?
QuietLocation need?
Walk to
the beach
OR ANDResponse
Quiet hotel
Walk to the beach
Hotel preferences
for this tripWhat did I like most
about my Hilton stay?Anchor A ᵢ
“Hilton stay”
User
history
m1
“On solo holidays, I want
what I liked most  about
my Hilton stay.”m2
“My favorite thing
about my Hilton stay
was the quiet nights .”Retrieved
evidenceCurrent context
Solo holiday
m1
m2ANDPrefers quiet hotelsResolve subproblem
Other context
With friendsm3Prefers lively hotels
Not applied to this tripRetrieved fragmentsTask  u = (q , z , ) ᵤ ᵤ κᵤ
Compositional
Support Construction
Support groups +
remaining gapsPreference rule
Missing attribute Initial retrieval①  Extract clue②  Clue-conditioned query③
 Retrieve missing evidence④Gap open  Grounded✓
Complementary
supportGrounded answer
m3
“With friends,
I prefer lively
hotels.”Other scope
Figure 2 Overview of Weaver. (1) An unresolved subquestion in the upper Intent Dependency Graph expands into a lower
retrieval trajectory. (2) Retrieved fragments expose clues that guide searches for missing evidence. (3) Compositional reasoning
combines the fragments under the current context and returns a supported conclusion to resolve the originating subquestion
and update the remaining gaps.
Intent Dependency Graph Construction.Given a request 𝑞and
its known context 𝜅𝑞, Weaver prompts an LLM to identify the
information needed for a personalized response and decompose it
into subquestions. The model determines how the answer to one
subquestion contributes to resolving another and organizes these
dependencies into an Intent Dependency Graph 𝐺𝐼=(𝑉𝐼,𝐸𝐼). This
graph is a directed acyclic graph (DAG), with edges pointing from
prerequisite tasks to the tasks that use their outputs.
Each node𝑢=(𝑞𝑢,𝑧𝑢,𝜅𝑢)represents a subquestion 𝑞𝑢with
an output variable 𝑧𝑢to be resolved within scope 𝜅𝑢. An edge
(𝑣,𝑢)∈𝐸 𝐼indicates a potential premise dependency of 𝑢on the
output of𝑣. The LLM represents the dependencies of task 𝑢through
a logical expression 𝜑𝑢using conjunction and disjunction with nega-
tion. As predecessor conclusions become available, this expression
determines which information remains necessary for downstream
reasoning. We assume that the initial graph covers the dependen-
cies required by the request. Subsequent reasoning updates task
conclusions and their relevance within this graph.
Let𝜃𝑡denote the established conclusions at iteration 𝑡, initialized
with information explicitly provided by the request and its known
context. A conclusion can serve as a premise when it has suffi-
cient support and no unresolved conflict in the required context.
Each step of evidence acquisition and interpretation for task 𝑢uses
context𝐶𝑢
𝑡=(𝑞𝑢,𝜅𝑢,𝛾𝑢
𝑡), where𝛾𝑢
𝑡⊆𝜃𝑡contains the established
conclusions on which that step relies. The graph records potential
dependencies. The step records its actual premises in 𝛾𝑢
𝑡. Differentinformation needs within a task can therefore rely on different
subsets of established conclusions.
Information Gap Tracking.Scope-Aware Compositional Reason-
ing returns a local gap set 𝛿𝑢
𝑡describing what remains necessary
to establish a sufficiently supported conclusion for task 𝑢. Before
evidence is retrieved, these gaps are initialized from the unresolved
task output and the inputs needed to locate relevant information.
Reading then makes the gaps more specific.
Starting from the response requirements, Weaver follows the
logical dependencies in 𝐺𝐼under the current conclusions to identify
the relevant task set 𝑈𝑡. This set contains tasks relevant to the
current response requirements and the upstream tasks needed to
supply their inputs. The outstanding information needs are
Δ𝑡=
(𝑢,𝑑)|𝑢∈𝑈 𝑡, 𝑑∈𝛿𝑢
𝑡	
.(1)
An information need remains in Δ𝑡when a required input is un-
available. Weaver first advances the upstream task responsible for
that input, allowing the dependent retrieval step to proceed once
the premise is established.
Let𝑅𝑢,𝑑
𝑡denote the prerequisite input slots for investigating gap
𝑑, determined by 𝜑𝑢and the current conclusions. Let 𝐴𝑢
𝑡collect
input slots with values in 𝜃𝑡applicable to task 𝑢, and let𝑏𝑢,𝑑
𝑡=1
indicate an available, unexecuted retrieval query, including a clue-
based probe. The executable frontier is
F𝑡=n
(𝑢,𝑑)∈Δ 𝑡|𝑅𝑢,𝑑
𝑡⊆𝐴𝑢
𝑡, 𝑏𝑢,𝑑
𝑡=1o
.(2)

WWW ’27, May 10–14, 2027, Dublin, Ireland Linyao Zheng, Zhifang Mao, and Xuhang Shi
An initial query is formed from a gap (𝑢,𝑑)∈F 𝑡and its established
premises. Later queries can also incorporate source-text clues from
retrieved fragments.
After conclusion revision, Algorithm 1 refreshes the relevant
tasks and retrieval frontier. Supported conclusions enable down-
stream retrieval; lost support or unresolved conflicts reopen gaps.
Unnecessary tasks are paused with their evidence and dependencies
retained, so later findings can reactivate them.
Algorithm 1 Dependency Information Gap Tracking
Input:Request 𝑞, context𝜅𝑞, and current graph, conclusion, gap, and
query states.
Output:Relevant tasks𝑈 𝑡, gapsΔ𝑡, and frontierF 𝑡.
1:if𝐺𝐼is uninitializedthen
2: Decompose𝑞into subquestions with an LLM; build their prereq-
uisite DAG𝐺 𝐼and initialize𝜃 𝑡and local gaps.
3:end if
4:Trace response dependencies under 𝜃𝑡to obtain𝑈𝑡, including up-
stream tasks supplying missing inputs.
5: Collect relevant local gaps intoΔ 𝑡(Eq. 1).
6:FormF𝑡from gaps with available inputs and unexecuted queries
(Eq. 2).
7:Pause tasks outside 𝑈𝑡; retain their evidence and dependencies for
possible reactivation.
8:return𝑈 𝑡,Δ𝑡,F𝑡.
3.2 Memory-Conditioned Incremental Retrieval
Weaver incrementally retrieves missing evidence for the informa-
tion needs identified by the upper graph and records the discovery
process in the lower graph. As conversations are retrieved and
interpreted, newly exposed clues guide further searches through
the user’s interaction history. Memory conditioning refers to the
use of retrieved source-text clues and memory-derived premises
when constructing later queries. The current request and its scope
remain part of the query context.
Clue-Conditioned Retrieval.At reasoning state 𝑡, the first module
providesF𝑡. A selected pair(𝑢,𝑑)∈F 𝑡identifies the subquestion
and information gap to investigate. For retrieval event 𝑖, the context
𝐶𝑢
𝑡=(𝑞𝑢,𝜅𝑢,𝛾𝑢
𝑡)supplies the subquestion, its applicability condi-
tions, and the established premises on which this search relies.
Previously retrieved fragments can expose clues for locating
missing information. Weaver retains each selected source-text span
and its source reference as an anchor. Let A𝑖contain the anchors
actually used to formulate query 𝑝𝑖. The LLM query-generation
processQcombines the current context, selected gap, and these
anchors:
𝑝𝑖=Q 𝐶𝑢
𝑡,𝑑,A𝑖.(3)
An initial query without source-text clues has A𝑖=∅. For a clue-
conditioned query, the LLM retains the locating terms from the
anchors and asks for the information needed to resolve 𝑑under
the premises in 𝛾𝑢
𝑡. Each anchor preserves its original wording and
source reference, linking the query to the fragment that prompted
it.
The search space 𝐻𝑎,𝑞contains the conversation fragments of
user𝑎accessible when request 𝑞is made. The existing retrievermaps queries and fragments into representations through encoders
ℎ𝑄andℎ𝑀. Its similarity function simscores each fragment 𝑚∈
𝐻𝑎,𝑞. The operator TopK returns the fragments with the highest
scores, up to a maximum of𝐾:
R𝑖=TopK𝑚∈𝐻𝑎,𝑞sim ℎ𝑄(𝑝𝑖),ℎ𝑀(𝑚).(4)
Initial and clue-conditioned queries share this retrieval backend
without an additional reranker. Their returned fragments accu-
mulate as evidence for the selected task. Each executed query is
recorded together with its context and premises, providing the
query state needed to update𝑏𝑢,𝑑
𝑡in Eq. 2.
Retrieval Trajectory Construction.Weaver records the discovery
process in the Memory-Conditioned Retrieval Trajectory Graph
𝐺𝑅
𝑡=(𝑉𝑅
𝑡,𝐸𝑅
𝑡). Here,𝑉𝑅
𝑡contains the retrieval steps recorded by
iteration𝑡, and𝐸𝑅
𝑡contains their directed discovery relationships.
Each node𝑥𝑖∈𝑉𝑅
𝑡represents an executed retrieval step. It records
which task and gap motivated the query and retains the context
under which the query was formed. The query 𝑝𝑖and its anchors
A𝑖are linked to the returned fragments R𝑖. The recorded task
associates the node with the upper graph.
Letsrc(A𝑖)collect the earlier retrieval nodes from which the
anchors in query 𝑝𝑖are taken. For recorded nodes 𝑥𝑗,𝑥𝑖∈𝑉𝑅
𝑡, the
discovery relationship is
(𝑥𝑗,𝑥𝑖)∈𝐸𝑅
𝑡⇐⇒𝑗<𝑖∧𝑥 𝑗∈src(A𝑖).(5)
An edge therefore records an actual contribution of source-text
clues to a later query. If several retrieval steps supply anchors for
the same query, all corresponding edges are retained. The temporal
order𝑗<𝑖makes the trajectory graph acyclic.
Algorithm 2 records this process. Fragments can be shared across
tasks, while queries under different premises retain distinct records.
A fragment from another situation may provide a locating clue
without supporting the current preference; composition assesses its
applicability and reinterprets accumulated evidence when premises
change.
Algorithm 2 Memory-Conditioned Incremental Retrieval
Input:Gap(𝑢,𝑑)∈F 𝑡, context𝐶𝑢
𝑡, history𝐻𝑎,𝑞, evidenceE𝑢
𝑡, trajec-
tory𝐺𝑅
𝑡, and query records.
Output:Fragments R𝑖; updated evidence, trajectory, and query records.
1:Gather required premises in 𝐶𝑢
𝑡; select source-text anchors A𝑖for
missing support, or∅if none are available.
2: Form𝑝𝑖=Q(𝐶𝑢
𝑡,𝑑,A𝑖).
3: RetrieveR 𝑖from𝐻𝑎,𝑞(Eq. 4); accumulateE𝑢←E𝑢
𝑡∪R𝑖.
4:Record node 𝑥𝑖with its task, gap, context, anchors, query, and
returned fragments.
5: Link each source node𝑥 𝑗∈src(A𝑖)to𝑥𝑖in the trajectory.
6: Mark𝑝𝑖as executed under its recorded context and premises.
7:returnR 𝑖, updated evidence, trajectory, and query records.
3.3 Scope-Aware Compositional Reasoning
As evidence accumulates, Weaver jointly interprets the retrieved
fragments within their temporal and situational scopes. It estab-
lishes conclusions with sufficient applicable support and revises
them when that support changes. The resulting conclusions and

Weaver: Incremental Preference Reasoning for Personalized RAG WWW ’27, May 10–14, 2027, Dublin, Ireland
unresolved information needs are returned to the upper graph to
guide subsequent retrieval.
Compositional Support Construction.Let E𝑢
𝑡contain the conver-
sation fragments accumulated for task 𝑢. Weaver prompts an LLM
to interpret these fragments under 𝐶𝑢
𝑡, resolving references across
conversations and identifying which memories and established
premises jointly support each candidate conclusion. The output
records the conclusion, its supporting sources, and its applicability
conditions. Unresolved references and unclear applicability condi-
tions enter the local gap set𝛿𝑢
𝑡.
A candidate conclusion is 𝑓=(𝑧𝑢,𝑣,𝜅𝑓), where𝑣is a proposed
value of the task output and 𝜅𝑓specifies its applicability. A support
group is recorded as 𝑠=(𝑓,𝑀 𝑠,Γ𝑠,𝜅𝑠). The fragments 𝑀𝑠⊆E𝑢
𝑡
and premises Γ𝑠⊆𝛾𝑢
𝑡jointly support 𝑓under the conditions in 𝜅𝑠.
Members of a group can supply complementary information, so
an individual memory need not support 𝑓on its own. Removing a
required member leaves that group’s justification incomplete.
For each group, the LLM assesses whether the combined inter-
pretation is sufficient for 𝑓, whether its temporal and situational
conditions apply to the task, and whether its required premises
remain established. The validity indicator 𝑉𝑡(𝑠,𝑢) records this judg-
ment, with value 1 when these requirements hold and 0 otherwise.
In particular, shared topic alone does not establish applicability, and
an unresolved reference leaves the support incomplete. Let 𝑆𝑡(𝑓)
collect the support groups for 𝑓. Distinct complete groups provide
alternative justifications: Sup𝑡(𝑓,𝑢)=Ô
𝑠∈𝑆𝑡(𝑓)𝑉𝑡(𝑠,𝑢) (6). This
relation aggregates the LLM’s judgments of whole support groups.
A conclusion retains support while at least one group remains valid.
Contextual Conclusion Revision.Weaver compares supported con-
clusions for the same task output within overlapping applicability
conditions. An unresolved conflict occurs when the LLM judges
their values incompatible within that shared scope. Semantically
equivalent values are merged. Different values can coexist under
disjoint conditions. When a subquestion asks about multiple situa-
tions, its answer can contain several compatible scoped conclusions.
Completeness is assessed against the requested output, and the task
remains open while required information is missing or a relevant
conflict is unresolved.
The LLM reassesses conclusions when evidence provides an ex-
plicit correction or clarifies applicability. A later timestamp alone
does not establish a preference change. Changes to evidence and
premises also trigger reassessment of the support groups that de-
pend on them. An alternative valid group can preserve a conclusion
when another group loses support.
LetW𝑡⊆𝜃𝑡contain established conclusions that lose sufficient
applicable support or acquire an unresolved conflict. Let N𝑡contain
newly established or revised scoped conclusions with sufficient
support and no unresolved conflict. Reassessment follows affected
premise dependencies in dependency order, including downstream
conclusions that rely on a changed premise. Weaver completes this
reassessment before finalizing N𝑡andW𝑡and returning to retrieval.
The updated conclusion set is
𝜃𝑡+1=(𝜃𝑡\W𝑡)∪N𝑡.(7)
Algorithm 3 returns revised conclusions and local gaps to the upper
graph, which refreshes premises and recomputes 𝑈𝑡+1,Δ𝑡+1, andF𝑡+1. Retrieval edges record discovery: changing a locating clue
does not invalidate conclusions supported independently of it.
Algorithm 3 Scope-Aware Compositional Reasoning
Input:Context 𝐶𝑢
𝑡, evidenceE𝑢
𝑡, support groups, and established con-
clusions𝜃𝑡.
Output:Updated support groups,𝜃 𝑡+1, and affected local gaps.
1:InterpretE𝑢
𝑡under𝐶𝑢
𝑡; resolve references, preference subjects, and
applicability conditions.
2:Construct or revise support groups 𝑠=(𝑓,𝑀𝑠,Γ𝑠,𝜅𝑠)for candidate
conclusions.
3:Assess group validity 𝑉𝑡(𝑠,𝑢) and aggregate alternative support
for each conclusion (Eq. 6).
4:Merge equivalent conclusions; identify incompatible values within
overlapping scopes.
5:Propagate proposed premise changes in dependency order, reassess-
ing affected support groups and downstream conclusions.
6:After reassessment, collect conclusions to establish or revise in N𝑡,
and those losing support or acquiring unresolved conflicts inW 𝑡.
7: Update𝜃 𝑡+1=(𝜃𝑡\W𝑡)∪N𝑡.
8:Update affected local gaps for missing support, unresolved refer-
ences, unclear applicability, or conflicts.
9:returnupdated support groups, 𝜃𝑡+1, and local gaps to gap tracking.
4 Experiments
We evaluate whether Weaver derives applicable preferences from
fragmented user histories. RQ1 compares personalized response
quality with existing memory-augmented methods. RQ2 measures
the contribution of each module. RQ3 examines the computational
cost of the observed personalization gains. RQ4 evaluates the dis-
covery and composition of preference-relevant memories within
user histories, followed by an evaluation beyond personalization.
4.1 Experimental Setup
Datasets.PersonaMem [ 24] evaluates personalized responses
using evolving user information; we use its 32K-history configura-
tion. PrefEval [ 7] evaluates preference following in conversational
contexts. Section 4.5 additionally evaluates MuSiQue [ 26] and Web-
WalkerQA [27] beyond user histories.
Baselines.We group six baselines by their primary focus. Mem-
ory organization includes MemCoE [ 28], which learns organization
and update policies. BMAM [ 29] divides memory into specialized
subsystems. MemoryBank [ 30] maintains conversational memory
through forgetting and reinforcement. EMG-RAG [ 11] stores per-
sonal information in an editable graph. Preference RAG includes
RF-Mem [ 13], whose recollection path progressively expands evi-
dence. PGR [ 25] refines prospective retrieval probes with retrieved
facts. RF-Mem and PGR also provide the reported comparisons on
MuSiQue and WebWalker.
Evaluation Metrics.On PersonaMem and PrefEval, ACC mea-
sures option-label correctness; EM and token F1 are auxiliary text-
match scores comparing the generated response with the full gold-
option text. Appendix C.4 separates these outputs. MuSiQue uses
F1, EM, and supporting-evidence recall within the top five results

WWW ’27, May 10–14, 2027, Dublin, Ireland Linyao Zheng, Zhifang Mao, and Xuhang Shi
Table 1 Main results on personalized response generation.
All scores are percentages. Higher is better. Best results are
bold and second-best results are underlined.
PersonaMem [24] PrefEval [7]
Category Method ACC F1 EM ACC F1 EM
Ours Weaver86.42 78.57 73.84 91.10 85.37 82.61
Memory
OrganizationMemCoE [28] 70.41 68.06 63.31 82.40 81.30 78.14
BMAM [29] 61.67 59.90 54.12 74.17 72.90 69.83
MemoryBank [30] 62.48 60.72 55.05 83.54 82.16 79.08
EMG-RAG [11] 57.92 55.83 49.91 78.11 74.85 71.62
Preference
RAGRF-Mem [13] 64.21 63.50 58.76 79.62 76.11 73.25
PGR [25] 60.12 58.19 53.68 77.84 74.33 71.49
(R@5); WebWalker uses ACC only. Scores are percentages; dif-
ferences are absolute percentage points unless labeled as relative
changes.
Implementation.Weaver and all six baselines use Qwen3.8-27B,
Qwen3-Embedding-8B, and the same 16,384-token context limit.
Appendix B provides hardware, decoding, and evaluation details.
4.2 Main Results (RQ1)
Table 1 reports personalized response quality across both baseline
groups. Weaver achieves the highest ACC on both benchmarks,
with corresponding gains in F1 and EM.
On PersonaMem, MemCoE is the strongest baseline across all
three metrics. Weaver improves ACC from 70.41 to 86.42, F1 from
68.06 to 78.57, and EM from 63.31 to 73.84, gains of 16.01, 10.51, and
10.53 points. On PrefEval, MemoryBank is the strongest baseline
across all three metrics. Weaver improves ACC from 83.54 to 91.10
and F1 from 82.16 to 85.37. EM increases from 79.08 to 82.61. These
gains amount to 7.56, 3.21, and 3.53 percentage points, respectively.
The gains are larger on PersonaMem. Section 4.5 examines whether
Weaver finds sufficient support and uses it effectively.
4.3 Ablation Study (RQ2)
We evaluate three independent ablations, each removing one mod-
ule from the complete framework. Figure 3 shows ACC on the two
personalization benchmarks, and Appendix A reports all metrics
on all four datasets.
•w/o Dependency Graphremoves the upper graph. The orig-
inal question is passed directly to modules 2 and 3, with
subquestion decomposition and upper-graph gap tracking
disabled.
•w/o Incremental Retrievalreplaces Memory-Conditioned In-
cremental Retrieval with direct dense retrieval for each sub-
question. It retains the upper-graph subquestions and uses
module 3 to combine the retrieved evidence.
•w/o Compositional Reasoningremoves Scope-Aware Com-
positional Reasoning. It retains the dependency graph and
incremental retrieval, and passes all evidence accumulated
in the lower graph directly to the reader.
The complete framework outperforms every ablation on all re-
ported metrics. Removing the dependency graph produces the
PersonaMemPrefEval30405060708090100ACC (%)-16.17%-8.26%-2.49%
-16.03%-7.91% -7.62%Method
Weaver (full)
w/o Dependency Graph
w/o Incremental Retrieval
w/o Compositional ReasoningFigure 3 ACC ablations on PersonaMem and PrefEval. Dark
blue shows Weaver, and the lighter bars show independent
module removals. Labels indicate relative ACC decreases
from Weaver in percent.
largest ACC decrease on the personalization benchmarks and Web-
Walker, and the largest F1 decrease on MuSiQue.
Dependency graph.Applying modules 2 and 3 directly to the
original question reduces ACC by 13.97 points on PersonaMem and
14.60 points on PrefEval. MuSiQue F1 decreases by 11.20 points,
and WebWalker ACC by 8.71 points. These results support organiz-
ing memory discovery around dependent subquestions and their
unresolved information needs. This ablation removes decomposi-
tion together with dependency-based gap tracking, measuring their
combined contribution.
Incremental retrieval.Replacing module 2 with direct dense re-
trieval reduces ACC by 7.14 points on PersonaMem and 7.21 points
on PrefEval. MuSiQue F1 decreases by 4.86 points, and WebWalker
ACC by 1.49 points. Since this variant retains compositional rea-
soning, the comparison shows that combining directly retrieved
evidence does not recover the performance of incremental memory
discovery guided by retrieved clues.
Compositional reasoning.Passing all lower-graph evidence di-
rectly to the reader lowers ACC from 86.42 to 84.27 on PersonaMem
and from 91.10 to 84.16 on PrefEval, decreases of 2.15 and 6.94 points.
MuSiQue F1 decreases by 4.31 points, and WebWalker ACC by 0.33
points. Because module 3 also updates conclusions and remaining
gaps during retrieval, this ablation measures its contribution to the
iterative process as well as to evidence interpretation.
4.4 Personalization Quality and Cost (RQ3)
Figure 4 compares the five methods with available cost measure-
ments using mean ACC over PersonaMem and PrefEval, with
approximate aggregate token ratios normalized to Weaver. The
relative token costs are approximately 0.46 for RF-Mem, 0.57 for
BMAM, 0.65 for PGR, 1.00 for Weaver, and 1.79 for MemCoE. Weaver
achieves the highest mean ACC at 88.76, compared with 71.92

Weaver: Incremental Preference Reasoning for Personalized RAG WWW ’27, May 10–14, 2027, Dublin, Ireland
0 0.5 1.0 1.5 2.0
Relative tokens (Weaver = 1)6065707580859095Mean ACC (%)RF-Mem
BMAMPGRWeaver
MemCoE
Figure 4 Personalization quality and computational cost for
the five methods with available cost measurements. Points
show mean ACC over PersonaMem and PrefEval against rela-
tive token consumption, with Weaver normalized to 1. Circle
area encodes mean runtime. Higher positions indicate better
accuracy, and positions farther left indicate lower token cost.
for RF-Mem, 67.92 for BMAM, 68.98 for PGR, and 76.41 for Mem-
CoE. The three lower-cost baselines use approximately 46–65% of
Weaver’s tokens, illustrating a trade-off between token expendi-
ture and personalization quality. Compared with MemCoE, Weaver
uses approximately 56% as many tokens, a reduction of about 44%,
while improving mean ACC by 12.36 percentage points, rounded
after subtraction. Mean end-to-end latency per request is 327 s for
Weaver and 360 s for MemCoE. On PrefEval, Weaver also exceeds
MemCoE by 8.70 ACC points and 4.07 F1 points.
Weaver reinterprets available evidence before further retrieval,
directing computation toward the support needed for the current
request.
4.5 Preference Memory Discovery and
Composition (RQ4)
How effectively does Weaver discover and compose preference-
relevant memories to answer the current request? Such memories
include explicit preferences, past experiences, evaluations, and con-
textual information whose relevance depends on the request. We
first evaluate whether retrieval recovers sufficient support, then
assess composition given identical candidate memories, and finally
examine applicability beyond personalization.
Preference memory discovery.Can Weaver recover sufficient sup-
port from preference-relevant memories distributed across user his-
tories? We compare Weaver withw/o Incremental Retrieval, which
directly retrieves dense matches for each subquestion while retain-
ing the dependency graph and compositional reasoning. We fix the
backbone, retrieval backend, history segmentation, and cumulative
retrieval budget. Recovery succeeds when memories accumulated
across retrieval steps contain at least one complete sufficient evi-
dence set for the current request; partial support counts as a miss.
The two diagnostics use 120 questions from PersonaMem, groupedTable 2 Memory discovery and composition beyond personal-
ization. All scores are percentages. Best results are bold and
second-best results are underlined.
MuSiQue [26] WebWalker [27]
Method F1 EM R@5 ACC
Weaver69.52 59.36 85.22 60.31
RF-Mem [13] 65.81 55.92 82.72 56.22
PGR [25] 67.11 57.43 84.31 59.02
Table 3 Complete support recovery (%), averaged over three
runs. Groups indicate the minimum memories needed for
sufficient support. IR: Incremental Retrieval. Δ: Weaver’s
gain (percentage points), rounded after subtraction.
Support requirement w/o IR WeaverΔ(pp)
One memory suffices 94.17 95.83 +1.67
Two memories required 81.67 87.50 +5.83
≥3 memories required 72.50 81.67 +9.17
by the minimum number of memories needed for sufficient support,
with 40 questions per group.
Table 3 shows that direct dense retrieval already performs well
when one memory suffices, recovering complete support for 94.17%
of questions compared with Weaver’s 95.83%. The gain widens to
5.83 points when two memories are required and 9.17 points when
at least three are required. Clues found in earlier fragments guide
subsequent searches for missing support. These results support
memory-conditioned retrieval for recovering dispersed memories
that jointly support a preference judgment. Improved recovery is
accompanied by higher overall answer ACC across all 120 questions,
from 79.17 to 84.72.
Preference memory composition.Does compositional reasoning
improve personalized answer accuracy given the same candidate
memories? We bypass retrieval and provide Weaver and the direct
reader with identical candidate memories containing all required
evidence and distractors. Supporting fragments are not identified,
so the models must determine which records apply and how they
support the current request. Direct reader answers from this can-
didate set; Weaver applies Scope-Aware Compositional Reasoning
to interpret and combine the memories before answering. We fix
the backbone, task premises, and memory order, with no additional
retrieval. We compare answer accuracy within the same support-
requirement groups.
Table 4 shows an ACC gain of 2.50 points when one memory
suffices, increasing to 5.00 points when two are required and 8.33
points when at least three are required. In the last group, accuracy
rises from 72.50 to 80.83 despite both methods having access to
sufficient support. The widening advantage supports interpreting
relationships among memories when several records must jointly
support the answer. Same-topic preferences from other contexts
remain in the input, requiring the models to consider applicability
to the current request.

WWW ’27, May 10–14, 2027, Dublin, Ireland Linyao Zheng, Zhifang Mao, and Xuhang Shi
Table 4 Answer accuracy (%) with identical candidate memo-
ries containing sufficient support and distractors, averaged
over three runs. Groups indicate required support, not input
size.Δ: Weaver’s gain (percentage points).
Support requirement Direct reader WeaverΔ(pp)
One memory suffices 89.17 91.67 +2.50
Two memories required 82.50 87.50 +5.00
≥3 memories required 72.50 80.83 +8.33
Together, these diagnostics show benefits in finding sufficient
preference-relevant memories and combining their support for
personalized answers. They do not separately quantify temporal-
update handling or situational applicability.
Beyond preference memories.Does the same discovery and com-
position process extend beyond preference memories? We evaluate
Weaver on MuSiQue [ 26] and WebWalkerQA [ 27]. MuSiQue tests
multi-hop question answering through composed questions. Web-
WalkerQA, abbreviated as WebWalker, tests information seeking
with Web pages accessed at runtime. Both tasks require locating
and combining factual support outside user histories.
Table 2 shows improvements over PGR, the strongest reported
baseline on both tasks. On MuSiQue, F1 rises from 67.11 to 69.52, EM
from 57.43 to 59.36, and R@5 from 84.31 to 85.22. On WebWalker,
ACC rises from 59.02 to 60.31. These gains support the applicability
of the discovery and composition process beyond preference-related
histories. As end-to-end results, they do not isolate the contribu-
tions of memory discovery and composition.
4.6 Case Study
We examine a PersonaMem example in which the user asks whether
a large city with vibrant nightlife and bustling attractions would suit
them. Figure 5 shows selected steps from a historical development
run of Weaver (v3.1). The upper graph separates the user’s preferred
activity settings from their experiences in crowded environments,
and the recommendation depends on both. The preference task
initially remains unresolved.
Incremental memory discovery.During later retrieval, a memory
recalling the excitement and energy of live games (m168) provides
an original-text anchor for further search. This search retrieves four
additional documents, including a basketball experience (m164) de-
scribing overwhelming crowd noise and greater comfort watching
at home, and a charity-run experience (m158) expressing discom-
fort with competition and a preference for collaboration. These
newly retrieved experiences enter the subsequent evidence inter-
pretation. Another new memory (m166 and m167), together with
m168 as a premise, guides the next search to a dance-class experi-
ence (m156 and m157). All five additional documents appear in the
final citations. The lower graph therefore records how previously
retrieved text leads to further evidence while the upper preference
task remains open.
Scope-aware compositional reasoning.The accumulated evidence
distinguishes the appeal of an activity from the conditions under
which the user enjoys it. Recalling the excitement of live games does
User requestWould a large city with vibrant nightlife and bustling attractions suit me?
1  Information Gap Tracking Decompose request
Preferred activity settings?
Unknown Partial Supported
Comfortable, low-pressure, collaborative settingsExperiences in crowds?
SupportedOther branch, summarized
Discomfort in crowded or chaotic activitiesAnswer to the
original request
Recommendation (c)
Quieter, more
intimate
destinationsAND
Expand   subproblem
2  Incremental Retrieval 3  Evidence Synthesis
m168 · Live games
“…the excitement and energy of the live games…”
Retrieval A Original-text anchor: m168
m164·Live basketball
“the crowd noise gets
overwhelming”
“more relaxed and 
engaged …
from the comfort of home”m158 · Charity run
Competition felt
uncomfortable.
Preferred collaboration.
SummaryRetrieved
evidenceSupportedActivity appeal
Excitement of live games 
UnresolvedCrowd preference unknown
SupportedEnvironmental comfort
Crowd noise can overwhelm    
SupportedSocial atmosphere
Collaboration over competition
Enjoying live games can coexist
with discomfort in crowds.
Grounded preference
Comfortable, low-pressure,
collaborative settingsResolve subproblem
m164m168
m158
Figure 5 Case study on PersonaMem. The request is decom-
posed into activity-setting and crowd-experience subques-
tions. Selected retrieval steps recover memories that support
a contextualized preference. This preference returns to re-
solve the activity-setting subquestion and combines with the
other branch to answer the original request.
not establish a preference for crowded environments. The basket-
ball account expresses enjoyment alongside discomfort with noise.
The charity-run account adds a social dimension, favoring collabo-
ration over competition. Interpreting these experiences together
allows enthusiasm for an activity to coexist with a preference for
comfortable, low-pressure settings. After the accumulated evidence
is read, including the dance-class experience, the preference task
moves from partial to supported. The resulting premise is combined
with the separately established summary of discomfort in crowded
settings and chaotic activities. The final recommendation selects the
correct option (c), which favors quieter, more intimate destinations.
This recommendation transfers preferences from related activi-
ties to travel, since the available history contains no direct prefer-
ence about cities and nightlife. The trace illustrates how evidence
discovered in the lower graph is interpreted under its original con-
text and returned as a premise for the upper graph’s recommenda-
tion.
5 Conclusion
We presented Weaver, a personalized RAG framework that discovers
and composes preference-relevant memories. Its dual-layer graph
tracks dependent information needs and guides incremental re-
trieval using premises and clues from user histories. Compositional
reasoning interprets evidence within applicable contexts, while up-
dated conclusions and remaining gaps direct subsequent retrieval.
Experiments on PersonaMem and PrefEval show improved answer
quality, and controlled diagnostics show more complete support
recovery and higher accuracy with identical candidate memories,
especially when multiple memories are required. These results
support coupling memory discovery with contextual preference
reasoning for personalized responses.

Weaver: Incremental Preference Reasoning for Personalized RAG WWW ’27, May 10–14, 2027, Dublin, Ireland
References
[1]Peter Brusilovsky. 2001. Adaptive Hypermedia.User Modeling and User-Adapted
Interaction11 (2001), 87–110. doi:10.1023/A:1011143116306
[2]Jinheon Baek, Nirupama Chandrasekaran, Silviu Cucerzan, Allen Herring, and
Sujay Kumar Jauhar. 2024. Knowledge-Augmented Large Language Models
for Personalized Contextual Query Suggestion. InProceedings of the ACM Web
Conference 2024 (WWW ’24). Association for Computing Machinery, 3355–3366.
doi:10.1145/3589334.3645404
[3]Hongru Cai, Yongqi Li, Wenjie Wang, Fengbin Zhu, Xiaoyu Shen, Wenjie Li,
and Tat-Seng Chua. 2025. Large Language Models Empowered Personalized
Web Agents. InProceedings of the ACM on Web Conference 2025 (WWW ’25).
Association for Computing Machinery, 198–215. doi:10.1145/3696410.3714842
[4]Ishita Kumar, Snigdha Viswanathan, Sushrita Yerra, Alireza Salemi, Ryan A.
Rossi, Franck Dernoncourt, Hanieh Deilamsalehy, Xiang Chen, Ruiyi Zhang,
Shubham Agarwal, Nedim Lipka, Chien Van Nguyen, Thien Huu Nguyen, and
Hamed Zamani. 2024. LongLaMP: A Benchmark for Personalized Long-form
Text Generation. arXiv:2407.11016 [cs.CL] doi:10.48550/arXiv.2407.11016
[5]Yiming Du, Hongru Wang, Zhengyi Zhao, Bin Liang, Baojun Wang, Wanjun
Zhong, Zezhong Wang, and Kam-Fai Wong. 2024. PerLTQA: A Personal Long-
Term Memory Dataset for Memory Classification, Retrieval, and Fusion in Ques-
tion Answering. InProceedings of the 10th SIGHAN Workshop on Chinese Language
Processing (SIGHAN-10). Association for Computational Linguistics, Bangkok,
Thailand, 152–164. https://aclanthology.org/2024.sighan-1.18/
[6]Jin Chen, Zheng Liu, Xu Huang, Chenwang Wu, Qi Liu, Gangwei Jiang, Yuan-
hao Pu, Yuxuan Lei, Xiaolong Chen, Xingmei Wang, Kai Zheng, Defu Lian, and
Enhong Chen. 2024. When Large Language Models Meet Personalization: Per-
spectives of Challenges and Opportunities.World Wide Web27, Article 42 (2024).
doi:10.1007/s11280-024-01276-1
[7]Siyan Zhao, Mingyi Hong, Yang Liu, Devamanyu Hazarika, and Kaixiang Lin.
2025. Do LLMs Recognize Your Preferences? Evaluating Personalized Prefer-
ence Following in LLMs. InThe Thirteenth International Conference on Learning
Representations. https://arxiv.org/abs/2502.09597
[8]Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
Fabio Petroni, and Percy Liang. 2024. Lost in the Middle: How Language Models
Use Long Contexts.Transactions of the Association for Computational Linguistics
12 (2024), 157–173. doi:10.1162/tacl_a_00638
[9]Alireza Salemi, Sheshera Mysore, Michael Bendersky, and Hamed Zamani. 2024.
LaMP: When Large Language Models Meet Personalization. InProceedings of the
62nd Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers). Association for Computational Linguistics, 7370–7392. doi:10.
18653/v1/2024.acl-long.399
[10] Alireza Salemi, Surya Kallumadi, and Hamed Zamani. 2024. Optimization Meth-
ods for Personalizing Large Language Models through Retrieval Augmentation.
InProceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval (SIGIR ’24). Association for Computing
Machinery, 752–762. doi:10.1145/3626772.3657783
[11] Zheng Wang, Zhongyang Li, Zeren Jiang, Dandan Tu, and Wei Shi. 2024. Craft-
ing Personalized Agents through Retrieval-Augmented Generation on Editable
Memory Graphs. InProceedings of the 2024 Conference on Empirical Methods in
Natural Language Processing. Association for Computational Linguistics, 4891–
4906. doi:10.18653/v1/2024.emnlp-main.281
[12] Shuo Yu, Mingyue Cheng, Daoyu Wang, Qi Liu, Zirui Liu, Ze Guo, and Xiaoyu Tao.
2026. MemWeaver: A Hierarchical Memory from Textual Interactive Behaviors
for Personalized Generation. InProceedings of the ACM Web Conference 2026
(WWW ’26). Association for Computing Machinery, 6920–6931. doi:10.1145/
3774904.3792732
[13] Yingyi Zhang, Junyi Li, Wenlin Zhang, Penyue Jia, Xianneng Li, Yichao Wang,
Derong Xu, Yi Wen, Huifeng Guo, Yong Liu, and Xiangyu Zhao. 2026. Evoking
User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval.
InThe Fourteenth International Conference on Learning Representations. https:
//openreview.net/forum?id=f7p0F2X6XN
[14] Maryam Amirizaniani, Alireza Salemi, and Hamed Zamani. 2026. Learning to
Reason for Multi-Step Retrieval of Personal Context in Personalized Question
Answering. InProceedings of the 49th International ACM SIGIR Conference on
Research and Development in Information Retrieval (SIGIR ’26). Association for
Computing Machinery, 3580–3586. doi:10.1145/3805712.3809944
[15] Chris Richardson, Yao Zhang, Kellen Gillespie, Sudipta Kar, Arshdeep Singh,
Zeynab Raeesy, Omar Zia Khan, and Abhinav Sethy. 2023. Integrating Summa-
rization and Retrieval for Enhanced Personalization via Large Language Models.
arXiv:2310.20081 [cs.CL] doi:10.48550/arXiv.2310.20081
[16] Sheshera Mysore, Zhuoran Lu, Mengting Wan, Longqi Yang, Bahareh Sarrafzadeh,
Steve Menezes, Tina Baghaee, Emmanuel Barajas Gonzalez, Jennifer Neville, and
Tara Safavi. 2024. Pearl: Personalizing Large Language Model Writing Assistants
with Generation-Calibrated Retrievers. InProceedings of the 1st Workshop on
Customizable NLP: Progress and Challenges in Customizing NLP for a Domain,
Application, Group, or Individual (CustomNLP4U). Association for Computational
Linguistics, Miami, Florida, USA, 198–219. doi:10.18653/v1/2024.customnlp4u-1.16
[17] Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly, and
Cyrus Rashtchian. 2025. Sufficient Context: A New Lens on Retrieval Augmented
Generation Systems. InThe Thirteenth International Conference on Learning Rep-
resentations. https://openreview.net/forum?id=Jjr2Odj8DJ
[18] Juntao Tan, Liangwei Yang, Zuxin Liu, Zhiwei Liu, Rithesh R N, Tulika Manoj
Awalgaonkar, Jianguo Zhang, Weiran Yao, Ming Zhu, Shirley Kokane, Silvio
Savarese, Huan Wang, Caiming Xiong, and Shelby Heinecke. 2025. PersonaBench:
Evaluating AI Models on Understanding Personal Information through Accessing
(Synthetic) Private User Data. InFindings of the Association for Computational
Linguistics: ACL 2025. Association for Computational Linguistics, Vienna, Austria,
878–893. doi:10.18653/v1/2025.findings-acl.49
[19] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Bar-
bieri, and Yuwei Fang. 2024. Evaluating Very Long-Term Conversational Memory
of LLM Agents. InProceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers). Association for Computational
Linguistics, Bangkok, Thailand, 13851–13870. doi:10.18653/v1/2024.acl-long.747
[20] Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu.
2025. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive
Memory. InThe Thirteenth International Conference on Learning Representations.
https://arxiv.org/abs/2410.10813
[21] Gediminas Adomavicius, Ramesh Sankaranarayanan, Shahana Sen, and Alexan-
der Tuzhilin. 2005. Incorporating Contextual Information in Recommender
Systems Using a Multidimensional Approach.ACM Transactions on Information
Systems23, 1 (2005), 103–145. doi:10.1145/1055709.1055714
[22] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-
Intensive Multi-Step Questions. InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers). Association
for Computational Linguistics, 10014–10037. doi:10.18653/v1/2023.acl-long.557
[23] Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-
Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active Retrieval
Augmented Generation. InProceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing. Association for Computational Linguistics,
7969–7992. doi:10.18653/v1/2023.emnlp-main.495
[24] Bowen Jiang, Zhuoqun Hao, Young-Min Cho, Bryan Li, Yuan Yuan, Sihao Chen,
Lyle Ungar, Camillo J. Taylor, and Dan Roth. 2025. Know Me, Respond to Me:
Benchmarking LLMs for Dynamic User Profiling and Personalized Responses
at Scale. InSecond Conference on Language Modeling. https://openreview.net/
forum?id=6ox8XZGOqP
[25] Harshita Chopra, Krishna Kant Chintalapudi, Suman Nath, Ryen W. White, and
Chirag Shah. 2026. Thinking Ahead: Prospection-Guided Retrieval of Memory
with Language Models. arXiv:2605.14177 [cs.IR] doi:10.48550/arXiv.2605.14177
[26] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop Questions via Single-hop Question Composition.
Transactions of the Association for Computational Linguistics10 (2022), 539–554.
doi:10.1162/tacl_a_00475
[27] Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang,
Linhai Zhang, Yulan He, Deyu Zhou, Pengjun Xie, and Fei Huang. 2025. Web-
Walker: Benchmarking LLMs in Web Traversal. InProceedings of the 63rd Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pile-
hvar (Eds.). Association for Computational Linguistics, Vienna, Austria, 10290–
10305. doi:10.18653/v1/2025.acl-long.508
[28] Derong Xu, Shuochen Liu, Pengfei Luo, Pengyue Jia, Yingyi Zhang, Yi Wen, Yimin
Deng, Wenlin Zhang, Enhong Chen, Xiangyu Zhao, and Tong Xu. 2026. Learning
How and What to Memorize: Cognition-Inspired Two-Stage Optimization for
Evolving Memory. InProceedings of the 64th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), Maria Liakata, Viviane P.
Moreira, Jiajun Zhang, and David Jurgens (Eds.). Association for Computational
Linguistics, San Diego, California, United States, 44987–45011. doi:10.18653/v1/
2026.acl-long.2084
[29] Yang Li, Jiaxiang Liu, Yusong Wang, Yujie Wu, and Mingkun Xu. 2026. BMAM:
Brain-inspired Multi-Agent Memory Framework. InFindings of the Association
for Computational Linguistics: ACL 2026, Maria Liakata, Viviane P. Moreira, Jiajun
Zhang, and David Jurgens (Eds.). Association for Computational Linguistics, San
Diego, California, United States, 39604–39626. doi:10.18653/v1/2026.findings-
acl.1973
[30] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2024. Mem-
oryBank: Enhancing Large Language Models with Long-Term Memory.Proceed-
ings of the AAAI Conference on Artificial Intelligence38, 17 (2024), 19724–19731.
doi:10.1609/aaai.v38i17.29946

WWW ’27, May 10–14, 2027, Dublin, Ireland Linyao Zheng, Zhifang Mao, and Xuhang Shi
A Complete Ablation Results
Table 5 reports the complete results for the independent ablations
in Section 4.3. Each variant removes one module while retaining
the other two.
Table 5 Complete ablation results. All scores are percentages.
Higher is better. Best results are bold and second-best results
are underlined.
PersonaMem[24]
Variant ACC F1 EM
Weaver86.42 78.57 73.84
w/o Compositional Reasoning 84.27 78.06 73.15
w/o Incremental Retrieval 79.28 71.43 66.29
w/o Dependency Graph 72.45 63.80 58.12
PrefEval[7]
Variant ACC F1 EM
Weaver91.10 85.37 82.61
w/o Compositional Reasoning 84.16 83.72 80.94
w/o Incremental Retrieval 83.89 82.06 79.18
w/o Dependency Graph 76.50 73.80 70.15
MuSiQue[26]
Variant F1 EM R@5
Weaver69.52 59.36 85.22
w/o Compositional Reasoning 65.21 58.46 83.40
w/o Incremental Retrieval 64.66 54.50 81.91
w/o Dependency Graph 58.32 48.15 75.40
WebWalker[27]
Variant ACC
Weaver60.31
w/o Compositional Reasoning 59.98
w/o Incremental Retrieval 58.82
w/o Dependency Graph 51.60
B Implementation and Evaluation Settings
Datasets and execution.PersonaMem uses 32K histories; PrefEval,
MuSiQue, and WebWalkerQA use the official repository versions
available when the experiments were conducted. WebWalkerQA
accesses Web pages at runtime. The PersonaMem history setting is
distinct from the 16,384-token context limit for individual model
calls.
Models and hardware.All LLM calls use Qwen3.8-27B (AWQ-
INT4), and dense retrieval uses Qwen3-Embedding-8B with co-
sine similarity between normalized embeddings. The six baselines
share Weaver’s models and context limit. Experiments run on eight
NVIDIA RTX 3090 GPUs.
Generation.Separate thinking mode is disabled; intermediate
analysis uses visible structured output. Retrieval query generation
uses greedy decoding with at most 1,536 output tokens. Final an-
swering uses temperature 0.6, top- 𝑝0.95, top-𝑘20, and at most
2,048 output tokens.Answer scoring and cost.EM and token F1 lowercase answers,
remove English punctuation and articles, and normalize whitespace.
F1 uses multiset token overlap. For MuSiQue, text scores take the
maximum over reference answers or aliases. Scores average over
the full evaluation set, counting execution failures as zero. Web-
WalkerQA uses ACC only. Token cost counts total LLM tokens;
latency measures end-to-end time per request. Figure 4 reports
relative tokens and mean latency.
C Evidence Handling and Execution Details
We elaborate on evidence handling and response selection using the
notation of Section 3. These details distinguish the interpretation
of a memory, the support for a conclusion, and the completion of
the current request.
C.1 Original Sources and Scoped Observations
Historical interpretation retains original text together with its
source reference, speaker role, and available temporal informa-
tion. These records allow a conclusion to be traced to the passages
that support it. A user statement, an assistant suggestion, and an
account of a past experience have different evidential roles. In par-
ticular, assistant advice does not establish that the user adopted a
preference, and message order alone does not identify the date of
a described event. The subject of an experience and its applicable
situation must be resolved before it supports a preference for the
current request.
The structured reading format distinguishes observations that
support a candidate value, contradict existing support, provide par-
tial information, remain unknown, or are irrelevant to the local task.
These categories describe an interpretation of evidence, not a single
task-status sequence. A partial observation can expose a useful
search clue without establishing the task output. An observation
of support likewise does not by itself establish that the original
request has been answered.
C.2 Joint Support and Premise Revision
A support group 𝑠=(𝑓,𝑀 𝑠,Γ𝑠,𝜅𝑠)records a complete candidate
conclusion, its original memories, its actual premises, and its ap-
plicability conditions. The members of 𝑀𝑠may be complementary:
their joint interpretation supports 𝑓, even when none suffices alone.
Separate complete groups provide alternative support under Eq. 6.
Partial observations are not automatically combined into a valid
conclusion; their relationship and joint sufficiency must be assessed.
When a premise changes, groups that depend on it are reassessed
before their conclusions support further retrieval. This reassess-
ment follows the dependency order and can affect downstream
conclusions. Independent valid groups can preserve a conclusion
when another group loses support. Retrieval-discovery edges re-
main distinct from these premise dependencies: changing a clue
used to locate a passage does not invalidate a conclusion indepen-
dently supported by that passage.
Contradictory evidence can undermine an existing conclusion
without establishing its opposite. A replacement conclusion re-
quires its own applicable support. Similarly, two different values
need not conflict when they concern disjoint situations. Reinterpret-
ing a historical summary can change its relevance to the current

Weaver: Incremental Preference Reasoning for Personalized RAG WWW ’27, May 10–14, 2027, Dublin, Ireland
request without asserting that the underlying past events were false.
Original sources are retained when a derived conclusion is revised.
C.3 Reading, Retrieval, and Completion
Algorithms 1–3 alternate interpretation and evidence acquisition.
Available evidence is interpreted before further retrieval. Revised
conclusions and local gaps update the relevant tasks and executable
frontier; a newly established premise may enable a previously
blocked subquestion. Pausing a task preserves its evidence and
retrieval records, allowing later findings to reactivate it. Queries
made under changed premises retain distinct contexts rather than
being treated as identical executions.
Intermediate conclusions remain linked to their original sources,
so later composition can inspect the evidence behind a premise
rather than treating an unsupported summary as an additional
observation. An unsuccessful query does not by itself show that
all relevant memories have been exhausted. Likewise, an empty
frontier is not a certificate of completion. The original request
requires sufficient applicable support for its necessary outputs and
no unresolved relevant conflict. Unresolved references, missing
premises, and unclear applicability remain explicit gaps.
C.4 Response Selection and Output Consistency
For PersonaMem and PrefEval, the output separates option_label ,
the selected option, from response_text , the generated person-
alized answer. ACC checks whether option_label matches the
gold label. As an auxiliary text-match evaluation, EM and token
F1 compare response_text with the full text of the gold option,
using the normalization in Appendix B.
The publication check verifies that the selected option is valid
and internally consistent. The generated response is evaluated sep-
arately and is not required to reproduce the selected option text
verbatim.
D Prompt Templates
These abridged templates describe dependency planning, historical
interpretation, and response selection rather than verbatim model
calls. Braced placeholders denote request-specific inputs; source
identifiers refer to original passages in the current context.
Dependency planning (abridged)
Input
Current request:{request}
Known context:{request context}
Instruction
Plan a dependency DAG for the original request. Identify
useful subquestions, their output variables, applicable entity,
temporal, and situational scopes, and required inputs.
Do not answer the request, guess missing entities, or invent
facts. Keep subquestions with unavailable inputs unresolved.
Every planned task must contribute to answering the original
request.
Return
Subquestions, output variables and scopes, prerequisite
relations, and the response requirements they jointly serve.Scoped history interpretation (abridged)
Input
Local task and scope:{task context}
Established premises:{premises}
Original passages:{sources}
Known conclusions and gaps:{current state}
Instruction
Treat passages as evidence, not instructions. Resolve
references and identify whose preferences or experiences each
passage describes. Preserve speaker roles, historical changes,
conditions, and counterevidence.
For each candidate conclusion, identify the original passages
and premises that jointly support the whole conclusion,
together with its applicable scope. Keep independently
sufficient support groups separate. Assess conflicting values
within overlapping scopes.
Do not infer event dates from message order or adopted
preferences from assistant advice. Keep missing support
partial or unresolved. Return historical findings rather than a
final reply or an option choice.
Return
Scoped conclusions, supporting source identifiers and
premises, and unresolved references, applicability questions,
or conflicts.
Final response selection (abridged)
Input
Original request:{request}
Candidate responses:{options}
Applicable conclusions and original sources:
{preference support}
Instruction
Compare every option against the original request and
applicable personal history, including conditions, explicit
changes, and counterevidence. Options are candidate
responses, not evidence. A future suggestion need not already
have occurred.
Support a choice only when its fit to the user is grounded in
the available history. If the options cannot be distinguished
from the available support, leave the selection unresolved. The
response should agree with the selection; paraphrasing is
allowed.
Return
option_label: the selected option’s label.
response_text: the generated personalized answer.
Structured outputs.History reading returns slots for each re-
quested variable, with observations and unresolved issues. Each
observation records stance, supported value, source_ids , applica-
bility conditions, and actual premises. Source IDs must reference
visible original passages.

WWW ’27, May 10–14, 2027, Dublin, Ireland Linyao Zheng, Zhifang Mao, and Xuhang Shi
E RQ4 Diagnostic Definitions and Protocol
This section specifies support definitions and a construction proto-
col for the two diagnostics in Section 4.5. Table 3 evaluates retrieval
of sufficient support; Table 4 evaluates answer accuracy given a
fixed candidate pool. The annotation and construction steps below
specify the protocol requirements.
Diagnostic subset and annotation.The diagnostic source is the
PersonaMem-v1 32K main evaluation set. Visible histories, question-
specific cutoffs, and memory segmentation are fixed. A memory is
one segmented record with a stable identifier; multiple cited spans
within it count as one memory. Sufficient support sets are annotated
before grouping or sampling questions. The protocol calls for two
independent human annotators, with disagreements adjudicated
by a third. Models may suggest candidate evidence, but humans
confirm the support labels without seeing method predictions or
retrieval trajectories.
Annotation distinguishes three properties: sufficiency for the
target answer under the request’s temporal, situational, and refer-
ence conditions; inclusion-minimality, meaning that deleting any
member breaks that support route; and minimum cardinality across
alternative routes. Inclusion-minimality alone does not establish
minimum cardinality. Review covers all runtime-visible sources, in-
cluding system-provided persona information, to check for shorter
routes. Let𝐵𝑞denote information provided without retrieval, shared
across methods. Sufficiency is assessed given (𝑞,𝐵𝑞); questions an-
swerable from 𝐵𝑞alone fall outside the one-, two-, and at-least-
three-memory groups.
After annotation, a fixed-seed sampling procedure selects 40
questions per group, accounting for persona and question type.
The resulting 120-question list is frozen before comparison. Eli-
gibility and sampling must not depend on Weaver’s correctness,
improvement, or citation count.
Alternative support and complete recovery.For each question,
retain the family of verified sufficient routes, including alternatives
of different lengths:
S𝑞={𝑆𝑞,1,𝑆𝑞,2,...}, 𝑘 𝑞=min
𝑆∈S𝑞|𝑆|.
The groups correspond to 𝑘𝑞=1,𝑘𝑞=2, and𝑘𝑞≥3. Here𝑘𝑞is the
minimum over annotated routes; manual review does not prove
that every possible route has been enumerated. For run 𝑟, let𝑅𝑞,𝑟
contain the distinct memory IDs accumulated within the retrieval
budget. Complete recovery is
Recovered(𝑞,𝑟)=1
∃𝑆∈S𝑞:𝑆⊆𝑅𝑞,𝑟
.
Any complete annotated route counts, including a longer alterna-
tive. For example, routes {𝑚2,𝑚5}and{𝑚8,𝑚9,𝑚12}place a ques-
tion in the two-memory group, but either route establishes recov-
ery. The mixture{𝑚2,𝑚8,𝑚9}does not unless separately verified
as sufficient. Recovery is scored independently of whether the final
answer is correct. The composition experiment’s selected route
does not restrict the routes accepted for retrieval recovery.
Candidate pools for composition.Select one minimum-cardinality
annotated route𝑆★
𝑞using a fixed sampling seed, then construct
𝐶𝑞=𝑆★
𝑞∪𝐷𝑞,|𝐶𝑞|=𝐾,|𝐷 𝑞|=𝐾−|𝑆★
𝑞|.The fixed candidate count 𝐾is chosen before comparison to ac-
commodate support-set sizes and both methods’ input capacities.
Its numerical value must be documented in the run configuration.
All annotated routes remain in the annotation record; they are not
all merged into the input. Groupwise token lengths should also be
reported because equal memory counts do not imply equal text
lengths.
Distractors come from the same user’s visible history, excluding
known support sources. Sampling rules and proportions are fixed
in advance; eligible distractors include related topics with differ-
ent contexts, times, or subjects, and ordinary irrelevant memories.
Review of the whole pool checks whether distractors establish sup-
port alone, combine with selected memories into another route,
create a shorter route, or introduce unresolved counterevidence
that changes the target answer. New routes require replacing the
relevant distractors or recording the alternatives. Shorter routes
require rebuilding the pool or revising the group before freezing
the inputs. Failure to find another route is not an exhaustiveness
guarantee.
Paired inputs and repeated runs.Both methods receive the same
question, options and their order, memory IDs, original text, role
and time metadata, memory order, and shared premises. These
premises contain supplied facts and constraints, not conclusions de-
rived from the selected evidence. Support labels, route annotations,
and gold explanations are hidden. Weaver applies the composition
module; the direct reader answers from the same pool. Neither
performs further retrieval or separate replanning, and the entire
pool must fit both inputs without method-specific truncation. Equal
inputs do not establish equal computation: invocation counts and
token consumption should be recorded separately.
For three runs to measure generation variability, the question list,
groups, gold answers, candidate pools, ordering, shared premises,
prompts, models, and budgets stay fixed; only generation random-
ness changes. Pool-selection and generation seeds are recorded
separately. If the service does not support seeds, record the de-
coding settings and independent calls. Deterministic repetitions
are not evidence of sampling variability. Varying memory order
instead constitutes an order-sensitivity experiment, with paired
permutations across methods.
Reproducibility records and interpretation.Per-question records
should include question ID, group, gold answer, selected and all
annotated support sets, ordered candidate IDs, exact text hashes,
cutoffs, shared premises, pool-selection seeds, review and adjudi-
cation records, and generation seeds. Run records should add 𝐾,
model and prompt versions, decoding and retrieval budgets, pre-
dictions, and invocation and token counts. The paired comparison
assesses composition given identical evidence; fixed 𝐾does not
eliminate differences in text length or the proportion of supporting
memories across groups.
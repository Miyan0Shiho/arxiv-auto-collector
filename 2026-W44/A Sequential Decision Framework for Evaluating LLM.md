# A Sequential Decision Framework for Evaluating LLM-Guided Search

**Authors**: Siqing Huo, Andrew Parry, Debasis Ganguly, Mark Smucker, Charles L. A. Clarke

**Published**: 2026-10-29 02:50:13

**PDF URL**: [https://arxiv.org/pdf/2610.39231v1](https://arxiv.org/pdf/2610.39231v1)

## Abstract
Recent systems applying retrieval-augmented generation (RAG) and "deep research" iteratively generate queries, retrieve documents, and examine their contents, using the resulting information to guide subsequent search steps. When this process is carried out autonomously rather than by a human searcher, the system must make explicit decisions about which search trajectories to explore and what objective the search is intended to optimize. We introduce a simple framework for reasoning about such search policies, providing a principled approach to evaluation. The framework models the search environment as a set of ranked lists and represents behavior using two primitive actions: select, which advances judgment through an existing ranking, and spawn, which generates a new ranking through query generation or reformulation. Each action produces a document to inspect, yields information that updates the search state, and contributes a reward for a task-specific objective. Within this abstraction, a variety of existing retrieval strategies can be expressed as policies operating over a common environment, while common retrieval objectives can be represented through reward and weighting functions over the resulting search trajectory, and the resulting formulation admits a natural reinforcement learning interpretation. We illustrate the approach using high-recall information retrieval (HRIR), where the goal is to identify as many relevant documents as possible with limited judging effort, showing how the framework provides a unified way to describe and reason about strategies for both classical retrieval tasks and emerging LLM-guided search systems. Through experiments, we demonstrate that both the underlying retrieval method and the "deep research" policy employed can materially affect performance.

## Full Text


<!-- PDF content starts -->

A Sequential Decision Framework for Evaluating LLM-Guided
Search
Siqing Huo
University of Waterloo
Canada
siqing.huo.canada@gmail.comAndrew Parry
University of Glasgow
United Kingdom
a.parry.1@research.gla.ac.ukDebasis Ganguly
University of Glasgow
United Kingdom
debforit@gmail.com
Mark Smucker
University of Waterloo
Canada
mark.smucker@uwaterloo.caCharles L. A. Clarke
University of Waterloo
Canada
claclark@gmail.com
Abstract
Recent systems applying retrieval-augmented generation (RAG)
and “deep research” iteratively generate queries, retrieve docu-
ments, and examine their contents, using the resulting information
to guide subsequent search steps. When this process is carried out
autonomously rather than by a human searcher, the system must
make explicit decisions about which search trajectories to explore
and what objective the search is intended to optimize. We introduce
a simple framework for reasoning about such search policies, pro-
viding a principled approach to evaluation. The framework models
the search environment as a set of ranked lists and represents be-
havior using two primitive actions:select, which advances judgment
through an existing ranking, andspawn, which generates a new
ranking through query generation or reformulation. Each action
produces a document to inspect, yields information that updates the
search state, and contributes a reward for a task-specific objective.
Within this abstraction, a variety of existing retrieval strategies can
be expressed as policies operating over a common environment,
while common retrieval objectives can be represented through re-
ward and weighting functions over the resulting search trajectory,
and the resulting formulation admits a natural reinforcement learn-
ing interpretation. We illustrate the approach using high-recall
information retrieval (HRIR), where the goal is to identify as many
relevant documents as possible with limited judging effort, showing
how the framework provides a unified way to describe and reason
about strategies for both classical retrieval tasks and emerging LLM-
guided search systems. Through experiments, we demonstrate that
both the underlying retrieval method and the "deep research" policy
employed can materially affect performance.
1 Introduction
Recent systems for retrieval-augmented generation (RAG) and
“deep research” exhibit a common structure: They iteratively gener-
ate queries, retrieve documents, examine their contents, and use the
resulting information to guide subsequent search steps [ 18,24,32,
44,52,67]. This process resembles what has sometimes been called
“slow search,” in which a system incrementally explores multiple
search trajectories rather than relying on a single static ranking
[61]. From this perspective, modern research agents can be viewed
as policies operating in a search environment: They generate new
rankings, decide which documents to inspect, and accumulate evi-
dence over time.This pattern is closely related to earlier work on high-recall
information retrieval (HRIR), which addresses the problem of iden-
tifying as many relevant documents as possible within a large
collection [ 11,13,16,21,39,47]. Historically, HRIR has relied on
human assessors to judge documents and guide the search process,
which makes it closely related to interactive retrieval. In such set-
tings, the system produces ranked lists, while the human decides
which documents to examine next and when to reformulate the
query. Recent advances in large language models (LLMs) change
this picture by enabling automated judgments and query genera-
tion, allowing the search process to be controlled by an algorithmic
policy rather than a human user. Viewed in this way, search be-
comes the problem of repeatedly deciding where to allocate effort
in order to discover useful information as efficiently as possible.
The framework described in this paper provides a simple ab-
straction for reasoning about such policies and a principled basis
for evaluating them. We view the search process as a sequential
decision problem in which the system repeatedly chooses how to
allocate the judging effort. At any point, the system is in a search
state summarizing the current environment, including the ranked
lists that have been generated, the documents already examined,
and any information obtained from previous judgments. A policy
maps this state to an action that determines which ranking should
be explored next.
We represent these decisions using two primitive actions. The
selectaction chooses an existing ranked list and advances judgment
within that list, examining the next unseen document. Thespawn
action generates a new ranked list, for example, by issuing a new
query, applying a different ranking function, or performing a query
reformulation. Each action ultimately identifies the next document
to be judged, producing an observation that updates the search
state and may yield a reward signal reflecting the value of the
information discovered. By expressing the search process in terms
of these two actions, a wide range of existing retrieval methods can
be described as policies operating over a common environment,
and the same abstraction also enables the systematic design and
evaluation of new policies for LLM-guided search.
After a review of related work in Section 2, Section 3 presents the
proposed framework and formalizes the search process as a sequen-
tial decision problem over ranked lists. Section 4 and Section 5 show
how a variety of existing retrieval strategies can be expressed as
policies within this framework. For the examples and experiments
arXiv:2610.39231v1  [cs.IR]  29 Oct 2026

in this paper, we focus on high-recall information retrieval (HRIR),
using relevance-based reward functions and evaluating policies
in terms of recall-oriented metrics. Section 6 presents empirical
results comparing a range of policies under this setting. Finally, we
position our framework as a first step towards a general "learning
to search" framework for LLM-guided search.
2 Related Work
Our framework builds upon and connects three lines of related
work: high-recall retrieval, interactive and exploratory search, and
emerging LLM-guided search systems.
2.1 High-Recall Information Retrieval
Previously, high-recall retrieval has been studied extensively in
the context of technology-assisted review and the “total recall”
problem [ 13]. Work on active learning also provides important
methodological foundations, from early uncertainty sampling for
text classification [ 31] to explicit active-learning approaches for
high-recall retrieval [ 47]. Subsequent work on continuous active
learning emphasized reliable end-to-end review, including recall-
oriented variants and scalability analyses for technology-assisted
review [ 11,12,14]. Other related work examined the reliability of
assessments and the practical conduct of total-recall review [ 1,15],
as well as interaction choices such as judging short excerpts rather
than full documents [70].
Evaluation in high-recall retrieval is naturally defined relative to
a judging budget 𝐽, where𝐽denotes the number of documents that
may be assessed during the search process. In practice, 𝐽may be
set operationally, for example, by available time, assessor effort, or
a fixed review budget. However, for evaluation, a useful normalized
approach is to define the judging budget as a function of the number
of known relevant documents for the topic, for example 𝐽=𝑎𝑅+𝑏 ,
where𝑅is the number of known relevant documents [ 20]. Although
this approach does not represent viable stopping rules in practice,
since it depends on advanced knowledge of 𝑅, it avoids distortions
that arise from applying the same fixed judging budget to topics
with very different numbers of relevant documents. The special
case𝐽=𝑅 corresponds to recall@ 𝑅, or equivalently R-precision. In
Section 6, we adopt this normalized approach and report results
using the special case𝐽=𝑅.
2.2 Interactive and Exploratory Search
More broadly, the iterative exploration of search results and the re-
formulation of queries are closely related to the work on interactive
and exploratory search. Those traditions emphasize that search is
often open-ended, iterative, and opportunistic rather than a simple
query-response transaction [ 3,4,28,43,48,65]. This perspective
is also reflected in the work on “slow search,” which explicitly
treats search as a process that unfolds over time rather than one
constrained to an immediate response [ 61]. In such settings, users
examine the results, follow promising directions, and reformulate
their queries as their understanding develops. Related work has
also modeled search as a process in which users decide when to
continue along a current direction and when to switch to a new
one [ 59]. Taken together, this line of work views search as a process
unfolding over time across multiple possible trajectories rather thanas the passive consumption of a single static ranking. Taking this
human search process as a starting point, we can recast it in the
language of sequential decision making and describe search as a
policy-driven process.
2.3 LLM-Guided Search
Recent work on retrieval-augmented generation (RAG) has shown
how retrieved documents can support generation, initially with rel-
atively simple pipelines in which one or a small number of queries
are used to retrieve supporting material for downstream synthe-
sis [24,32,52]. Retrieval may also follow generation, for example,
to retrieve supporting evidence or validate factual claims in a re-
sponse [ 19,23]. More recent systems extend this pattern into multi-
step search processes that iteratively generate queries, retrieve
documents, and use accumulated evidence to guide subsequent
search steps, moving toward what are now often described as “deep
research” agents [ 6,8,22,34,66,69]. Some of this recent work has
begun to optimize deep search agents directly using reinforcement
learning and intermediate process rewards [ 26,46,71], reinforcing
the need for a framework in which such search policies can be
described and compared.
Although our experiments in this paper focus on high-recall
retrieval, the same general framework could also be applied to set-
tings in which the objective is not to find all relevant documents,
but to discover informational “nuggets” or aspects of an informa-
tion need [ 10,30,36,52,53,56,57,63,68]. This perspective also
connects to earlier work on adaptive allocation of judging effort,
including bandit-based approaches to test collection construction
and document assessment [ 55], suggesting a broader continuity be-
tween classical adaptive retrieval and emerging LLM-guided search
systems.
To support the evaluation of such systems, new benchmarks
have been proposed that measure their ability to answer com-
plex questions after conducting multi-step search. For example,
BrowseComp-Plus [ 9,58] provides a controlled benchmark for
evaluating deep research agents, focusing on end-to-end task per-
formance, including final answer accuracy. While such benchmarks
assess the effectiveness of the overall agent, they do not explicitly
model the decisions that control the search process. The framework
proposed here instead focuses on those decisions, enabling analysis
of the policies that govern how search trajectories are explored.
3 Framework
We begin by considering a query 𝑄that initiates a search process
over a large document collection. As discussed in the introduction,
such a process may arise in a variety of settings, including classical
high-recall retrieval, retrieval-augmented generation (RAG), and
LLM-guided “deep research.” In each case the system repeatedly
retrieves documents, examines their contents, and uses the result-
ing information to guide subsequent search steps. Our goal is to
discover useful information as efficiently as possible while explor-
ing the space of possible queries and rankings. The value of the
information obtained at each step is captured by a reward signal
derived from the documents examined. Depending on the task, this
reward may reflect binary relevance, graded gain, or the discovery
of informational “nuggets”.

Policy Section spawn: Generate ranking and judge select: Judge from an existing ranking
adhoc §5.1 Executed once to generate the initial ranked list Always execute select( 1), observing documents sequen-
tially in rank order
MTF §5.2 Draw a new ranked list from an external fixed pool
when the end of the active queue is reached and addi-
tional lists remain availableAlways select the current list in queue order; if the
resulting observation is relevant, move that list to the
front of the queue
Thompson §5.3 When Thompson sampling selects an arm that has
not yet been introduced into the environment, execute
spawn to create the corresponding list and observe its
first documentOtherwise, execute select(𝑖) for the active list whose
sampled posterior value is largest
ISJ §5.4 Generate a new query from the current search context
(historically by a human assessor; in our experiments
potentially by an LLM) and execute it to create a new
ranked listSelect from an existing active list to continue observing
documents from previously generated rankings
CAL §5.5 Retrain a classifier on the labeled set accumulated so far,
and apply it to the collection to produce a new ranked
list whenever a new model becomes availableContinue selecting from the most recently available
ranking until a newer ranking has been spawned, then
switch to that newest list
Table 1: Summary of policies considered in this paper, expressed in terms of the spawn andselect actions introduced in Section 3.
Classical ad hoc retrieval corresponds to a single initial ranking that is always selected; MTF and Thompson sampling allocate
effort across multiple rankings; ISJ introduces new rankings through query reformulation; and CAL introduces new rankings
through repeated retraining and reranking.
We model this search process as the incremental exploration
of a growing set of ranked lists. At each time step 𝑡the system
maintains a set of ranked lists, with initially𝐿={}at𝑡=0:
𝐿={𝐿1,𝐿2,...,𝐿𝑘}.
Each list𝐿𝑖is an ordered sequence of documents
𝐿𝑖=(𝑑𝑖
1,𝑑𝑖
2,𝑑𝑖
3,...)
where𝑑𝑖
𝑗denotes the document at rank 𝑗in list𝑖. Each list maintains
a depth pointer 𝑝𝑖indicating that documents 𝑑𝑖
1...𝑑𝑖
𝑝𝑖have already
been “observed” or “judged”. The next candidate document from
list𝑖is therefore𝑑𝑖
𝑝𝑖+1.
Let𝐸𝑡be the set of documents judged after step 𝑡. When docu-
ment𝑑is judged, the environment produces an observation O(𝑑)
describing the information obtained from that document. Depend-
ing on the task, this observation may represent binary relevance,
graded gain, or the discovery of informational nuggets. A reward
function𝑟(O(𝑑)) maps this observation to a scalar reward, reflect-
ing the discovery of a new relevant document or a new nugget. At
step𝑡, the environment contains the current lists 𝐿, their depth
pointers𝑝1...𝑝𝑘, and any additional information such as scores,
ranks, or previously observed judgments. We denote the complete
environment state abstractly by𝜃 𝑡.
At each step 𝑡, the system performs one of two actions: selecting
the next document from an existing list, or spawning a new ranked
list:
Select Given the current environment state 𝜃𝑡, select a list
index𝑖and observe the next unseen document from list 𝑖.
Starting at position 𝑝𝑖+1, previously observed documents are
skipped until an unseen document 𝑑𝑖
𝑗is found. The system
observesO(𝑑𝑖
𝑗), updates the pointer 𝑝𝑖←𝑗, and adds𝑑𝑖
𝑗to𝐸𝑡. Depending on the task, O(𝑑𝑖
𝑗)may correspond to a
binary or graded relevance judgment, or the discovery of
informational nuggets, and may produce a reward 𝑟(O(𝑑𝑖
𝑗)).
Spawn Create a new ranked list by generating 𝐿𝑘+1and ini-
tializing𝑝𝑘+1=0. Aspawnoperation may correspond to
generating a new query, applying a different ranking func-
tion, or another retrieval strategy. After creating the new
list, the system immediately executesselect(𝑘+1).
The actions select(𝑖) andspawn correspond, respectively, to exploit-
ing existing rankings and exploring new search trajectories. Each
step produces exactly one observation, so that 𝑡corresponds to the
number of observations made1
Both operations are controlled by a policy 𝜋(𝜃 𝑡)which maps the
current environment state to either spawn orselect(𝑖) . The goal of
the policy is to maximize the cumulative reward obtained during
the search process, which may be expressed as a discounted sum:
∞∑︁
𝑡=1𝛾𝑡−1𝑟𝑡,(1)
where𝑟𝑡is the reward obtained from the observation at step 𝑡and
𝛾is a discount factor. For, search, it is more convenient to use a
general weighting of rewards over time:
∞∑︁
𝑡=1𝑤(𝑡)𝑟 𝑡,(2)
1In our model, an observation O(𝑑) corresponds to a single LLM judgment event,
which we treat as a unit cost analogous to one human judgment in a traditional search
process. This abstraction simplifies the true costs in several ways. Modern retrieval
pipelines may also use LLM calls before a document is observed, for example, for pre-
filtering or reranking, including approaches such as pairwise reranking and listwise
reranking [ 45,50,51,54,60,72]. It also ignores cost variation across prompts due to
differing input and output token counts. Finally, at this level of abstraction, we treat
retrieval and ranking costs as negligible relative to LLM judgment costs.

where𝑤(𝑡) is a weighting function that reflects the value of discov-
ering useful information earlier in the search process [ 64]. As we
will see in the next section, many common IR evaluation measures
can be expressed in this form [10, 25].
4 Rewards
Equation 2 provides a general form for expressing many infor-
mation retrieval evaluation measures, generalizing them to our
framework. To illustrate this, consider a simple policy in which
the system executes spawn once to create a ranked list and then
repeatedly executes select( 1), observing documents sequentially in
rank order. In this case the trajectory is determined entirely by the
order of the ranking, and different evaluation measures arise from
different choices of the weighting function𝑤(𝑡)and reward𝑟 𝑡.
For example, discounted cumulative gain (DCG) corresponds to
choosing a logarithmic discount
𝑤(𝑡)=1
log2(𝑡+1)(3)
with𝑟𝑡defined as a gain value derived from the document’s rele-
vance label, often using an exponential gain such as 𝑟𝑡=2rel𝑡−1.
Rank-biased precision (RBP) may be expressed using a geometric
discount
𝑤(𝑡)=(1−𝜌)𝜌𝑡−1,(4)
where𝜌is a persistence parameter, and 𝑟𝑡typically represents
binary relevance in the range [0,1]. Many other ranking measures
can be written in the same form by selecting appropriate gain and
weighting functions.
In this paper we focus primarily on high-recall information re-
trieval (HRIR), where the goal is to discover as many relevant doc-
uments as possible while minimizing judging effort. In this setting,
the weighting function reflects a judging budget. If the available
judging budget is𝐽, we define
𝑤(𝑡)=(
1if𝑡≤𝐽,
0otherwise.(5)
In this case, the reward 𝑟𝑡is simply a binary indicator of relevance
for the document observed at step 𝑡. The resulting objective corre-
sponds to maximizing the number of relevant documents discovered
within the judging budget.
The same formulation can be extended to alternative reward def-
initions. For example, in tasks that emphasize information coverage
rather than document relevance, rewards may correspond to the
discovery of informational nuggets. In this case 𝑟𝑡indicates whether
the observation reveals previously unseen information, and the ob-
jective in Equation 2 measures the cumulative information gained
during the search process.
5 Policies
In this section, we map existing work into the framework of Sec-
tion 3, with a continued focus on HRIR. The policies are summarized
in Table 1.
5.1 Adhoc Search
The simplest policy corresponds to the classic adhoc search scenario.
The system first executes spawn to generate a ranked list and thenalways selects that list. At each step the system executes select( 1),
observing𝑑1
1,𝑑1
2,𝑑1
3,... in rank order. The process continues until
𝐽observations have been made. In the HRIR setting, where 𝑟𝑡is a
binary indicator of relevance and the weighting function enforces a
judging budget of 𝐽, the cumulative reward corresponds toRecall@J.
Classic adhoc search provides natural baselines for the experi-
ments in Section 6. We consider multiple rankers to generate the
ranked lists, including BM25, sparse learned, and dense retrieval
methods, as well as a hybrid two-stage retrieval pipelinem where
the results from all three methods are pooled and reranked with
MonoT5 and DuoT5 [49].
5.2 Move-to-Front
The Move-to-Front (MTF) strategy [ 16] assumes a fixed pool of
ranked lists produced by different retrieval systems. These lists
exist outside the search process and are introduced into the envi-
ronment through the spawn action. The environment maintains
the active lists in an ordered queue. Initially 𝐿={} , so the first
action must be spawn , which introduces a ranked list and observes
its first document. At each step the policy executes select(𝑖) where
𝑖corresponds to the current position in the queue. The next unseen
document from that list is observed. If the observation indicates a
relevant document, list 𝑖is moved to the front of the queue. Other-
wise the next step considers the following list in the queue. If the
end of the queue is reached without observing a relevant document,
the policy executes spawn to introduce a new ranked list from the
external pool if one is available; otherwise the scan resumes from
the beginning of the queue. Intuitively, lists that produce relevant
documents are promoted to the front so that they will be exploited
on subsequent steps, while new lists are introduced only when the
existing rankings fail to yield additional relevant material.
5.3 Thompson Sampling
The Move-to-Front policy may be viewed as a simple heuristic solu-
tion to the exploration–exploitation problem. Each ranked list acts
as an “arm” whose value is revealed through observations obtained
by executing select(𝑖) . Lists that recently produce relevant docu-
ments tend to remain near the front of the queue and are therefore
selected more frequently. From this perspective, MTF behaves like
a lightweight multi-arm bandit policy for allocating judging effort
across multiple rankings. In addition to MTF, other bandit-style
approaches have previously been explored in the context of HRIR
for adaptive test collection construction [39, 55].
Once the problem is viewed though the lens of multi-arm bandits,
it is natural to consider standard bandit algorithms. Thompson
sampling [ 62] provides a simple and effective alternative. Each
ranked list is associated with a probability distribution representing
the likelihood that the next unseen document from that list will
be relevant. At each step a value is sampled from the distribution
associated with each arm, and the policy selects the arm with the
highest sampled value.
In our framework the arms correspond to ranked lists drawn
from an external pool. If the selected arm has not yet been intro-
duced into the environment, the policy executes spawn to create
the corresponding list and observe its first document. Otherwise
the policy executes select(𝑖) to observe the next unseen document

from the list. In either case the resulting observation updates the
posterior distribution associated with that arm. This formulation
preserves the accounting used throughout the paper, where each
action produces exactly one observation, while providing a princi-
pled mechanism for balancing exploration and exploitation across
multiple rankings.
5.4 Interactive Search and Judging
Interactive Search and Judging (ISJ) [ 16] was originally proposed as
a human-in-the-loop search process designed to support high-recall
retrieval tasks. In the original formulation, a human assessor exam-
ines documents returned by a query, identifies relevant material,
and periodically reformulates the query to discover additional rele-
vant documents. The system therefore alternates between judging
documents from existing rankings and generating new queries that
may lead to additional relevant material.
In the framework of Section 3, this behavior can be expressed
using the select and spawn actions. The ranked lists correspond
to the results of queries issued during the search process. Execut-
ingselect(𝑖) observes the next unseen document from an existing
ranking, while spawn generates a new query and introduces the
resulting ranked list into the environment.
Historically the equivalent of the spawn action was performed by
a human assessor who reformulated queries based on the evolving
search context. In the present framework, however, this role may
be performed by an automated policy. For example, a new query
may be generated by prompting a language model with information
from the environment 𝜃𝑡, such as previous queries, examples of
relevant documents, and examples of non-relevant documents. The
resulting query is then submitted to the retrieval system, producing
a new ranked list that can be explored by subsequent select actions.
Section 6 demonstrates one such approach using LLM-generated
queries.
5.5 Continuous Active Learning
Continuous Active Learning (CAL) [ 1,11] was originally proposed
as an automated strategy for technology-assisted review in high-
recall retrieval tasks. In CAL, documents are judged sequentially
while a classifier is repeatedly retrained on the growing set of
labeled documents. After each update, the classifier is applied to the
entire collection to produce a new ranking from which subsequent
documents are selected.
In the framework of Section 3, training the classifier and applying
it to the collection corresponds to executing spawn , which generates
a new ranked list based on the current environment state 𝜃𝑡. Because
training may take nontrivial time, the system continues to observe
documents from the current ranking until a new ranking becomes
available.
The simplest CAL policy always executes select(𝑖) where𝑖cor-
responds to the most recently spawned list. When a new model
finishes training, the resulting ranking is introduced by executing
spawn , after which the policy begins selecting documents from
the new list2. Thus the search process alternates between judging
2We may think of this policy as spawning as soon as possible, while continuing to
judge documents from the most recent available list in the meantime. The frequency
of spawning depends on the relative time to retrain a model compared to the time to
judge a document. With a human judge, this policy works in practice, but with andocuments from the current ranking and spawning new rankings
produced by retraining the classifier on the accumulated judgments.
In this interpretation the spawn operation is triggered by the learn-
ing procedure, and the availability of a new list, rather than by
an explicit policy decision, while the policy itself simply selects
documents from the most recent ranking.
5.6 Query Reformulation Policies
In many search processes the key step that advances the search is
query reformulation. A human searcher typically gathers informa-
tion from a set of documents and then reformulates the query to
pursue a new direction. One query leads to another as the searcher
gradually discovers new aspects of the information need. In high-
recall search, this process is particularly important: relevant docu-
ments often appear in different regions of the collection that can
only be reached through new queries. In deep research, finding new
informational nuggets largely requires new queries.
In the framework of Section 3, this behavior is captured by the
spawn action. Executing spawn generates a new ranked list based
on the current environment state 𝜃𝑡. In practice this often corre-
sponds to generating a new query derived from the current search
context and submitting it to a retrieval system. The resulting rank-
ing introduces a new search trajectory that may reveal additional
relevant documents.
Recent work has explored automated approaches to query refor-
mulation, particularly using large language models (LLMs). For
example, QueryGym [ 5] provides an extensible framework for
implementing and evaluating query reformulation methods. The
toolkit supports a variety of LLM-based reformulation strategies,
integrates with common retrieval backends such as Pyserini [ 35]
and PyTerrier [ 42], and provides infrastructure for reproducible ex-
perimentation. As an alternative to dedicated query reformulation
approaches, prompt optimization frameworks can be used to gen-
erate reformulations on the fly from the current search context, in-
cluding previous queries and examples of relevant and non-relevant
information.
Within the framework described in this paper, query reformu-
lation methods can be viewed as implementations of the spawn
operator. Given the current environment state 𝜃𝑡, a reformulation
method may generate a new query that is submitted to the retrieval
system to produce a new ranked list.
Importantly, query reformulation specifies how new queries may
be generated, but not when they should be generated. The decision
to execute spawn remains part of the policy 𝜋(𝜃 𝑡)controlling the
search process. This separation allows the framework to combine
different reformulation methods with different policies, including
policies that trigger reformulation periodically, policies that react to
observed relevance signals, and policies that learn when spawning
new queries is most beneficial.
5.7 Policy Design Space
The framework above exposes a natural design space for search
policies. Once the search process is expressed in terms of the spawn
LLM-based judge the relative timing may differ substantially, potentially changing the
effective spawning rate.

andselect actions, a variety of strategies can be constructed by bor-
rowing ideas from reinforcement learning, bandit algorithms, and
simple search heuristics [ 2,33]. The following examples illustrate
several possibilities.
•Upper Confidence Bound (UCB).Each ranked list can be
treated as an arm in a multi-armed bandit [ 2]. A UCB-style
policy selects the list with the largest optimistic estimate
of relevance probability, balancing lists that have produced
relevant documents against lists that remain uncertain.
•𝜀-greedy selection.A simple baseline selects the list with
the highest observed relevance rate most of the time, but
occasionally selects a random list with probability 𝜀, ensuring
continued exploration.
•Spawn triggers.Policies may introduce explicit conditions
under which new rankings are spawned. For example, a new
query may be generated when no relevant documents have
been observed for several judgments, when the estimated
relevance probability of all active lists falls below a threshold,
or at fixed intervals during the search process.
•Contextual bandits.Rather than treating each ranking as
a single arm, the policy may use features of the environ-
ment state𝜃𝑡such as document scores, rank positions, or
previous judgments to estimate the probability of relevance.
Contextual bandit algorithms can then guide selection [ 33].
•Depth-aware policies.Relevance probability typically de-
creases with rank depth. Policies can incorporate this infor-
mation by estimating relevance probability as a function of
both the list and the current depth pointer 𝑝𝑖, allowing the
system to move more quickly away from exhausted rankings.
•Hybrid policies.Policies may combine bandit-based selec-
tion with spawning mechanisms that generate new rankings
through query reformulation, alternative ranking functions,
or classifier retraining.
These examples illustrate that once the search process is ex-
pressed in terms of spawn andselect actions, many existing tech-
niques can be adapted to produce new policies for high-recall re-
trieval.
6 Experiments
Our experiments illustrate how selected policies from Table 1 can be
instantiated within the framework, demonstrating how the frame-
work can be instantiated with real systems and how different poli-
cies can be implemented and compared in practice. The experiments
are based on the TREC Deep Learning 2019 collection. We use GPT-
4o-mini as the LLM for relevance judgments and GPT-5 for more
complex tasks such as spawning. In our experiments, spawning
is implemented through query reformulation. We employ DSPy
[29] for query reformulation to avoid the need to create manual
prompt boilerplate, while enabling the background optimization of
instructions and few-shot examples.
Each relevance judgment corresponds to a step in Equation 2.
Following the approach of Grossman et al . [20] to compare high-
recall retrieval methods, we define the judging budget as a function
of the number of known relevant documents 𝑅for each query.
This approach provides normalization, so that recall values can be
meaningfully averaged across queries with a wide range of relevantdocuments. After a search is completed, the average recall@R is
reported using official human relevance judgments to evaluate
performance.
•Adhoc Search(§5.1): This is our baseline policy where we
only execute select on a single ranked list. We keep judging
from the top of the ranked list until we run out of judging
budget.
•Query Reformulation(§5.6)then MTF(§5.2): We have
a fixed pool of multiple ranked lists and only execute select
until we run out of judging budget. In this experiment, the
initial pool of ranked lists is generated using batch query
reformulation, where we generate a list of distinct queries
all at once given the original query and its top retrieved
documents (Appendix A). Then we decide 𝑖ofselect(𝑖) using
the MTF strategy.
•Forward Only Spawning: Always executes select(𝑖) where
𝑖corresponds to the most recently spawned list. After a new
ranking list is introduced by executing spawn , the policy
begins only to select documents from the new list. Once we
decide a ranking list is exhausted (in this experiment, we
define it to be 5 irrelevant documents found), we spawn by
generating a new query using the cumulative context of past
queries and their top retrieved results (Appendix B).
•Bounded Spawning: Because spawn can be expensive, we
enforce a fixed quota on the number of spawn actions. Once
the limit is reached, spawn is no longer allowed and select is
no longer limited to only the most recently spawned ranking
list. We use select with the MTF strategy to merge what is
left of each ranking list.
For each type of policy, we experiment with four retrieval pipelines:
BM25, SPLADE-v3 [ 17], TCT-ColBERT-v2 [ 37], and a hybrid two-
stage system that pools the results of BM25, SPLADE and TCT-
ColBERT and reranks the pooled results with MonoT5 and DuoT5 [ 49].
We report recall@ 𝑅for all 16 runs in Figure 1. The figure illus-
trates how policies can be compared. For example, the best overall
policy (SPLADE & Query Reform +MTF) substantially outperforms
every policy using BM25 as its sole ranking method, with no overlap
in the reported 95% bootstrap confidence intervals. More gener-
ally, the figure shows that both the underlying ranking method
and the policy used to explore ranked lists can materially affect
performance, supporting the usefulness of a common framework
for comparing such strategies. As a more specific example, Figure 2
shows how recall@ 𝑅changes with the number of judgments made
under SPLADE and query reformulation followed by an MTF policy
for a specific query.
Figure 3 complements this comparison by isolating the cost of
judging itself. Rather than scaling the budget by the number of
known relevant documents 𝑅for each query, here we fix aglobal
budget: a run stops once limit∈{ 20,30,50}consecutive newly-
judged irrelevant documents have accumulated, independent of 𝑅.
Because no official qrels are consulted at this granularity, recall
is instead computed relative to the pooled union of documents
that any policy, retriever, and budget judged relevant under the
LLM judge, following standard pooling practice. This isolates how
efficiently each policy converts LLM judgments into discovered
relevant documents, holding the judge (GPT-4o-mini) fixed.

Figure 1: Comparison of recall@𝑅across different search runs.
Figure 2: An example number of judgments made versus
recall@R for QID 1133167 under Splade and Query Reformu-
lation then MTF policy.
Query Reformulation then MTF achieves the highest recall for
every retrieval pipeline and budget, but this gain is not free: because
MTF interleaves judging across the original query and its reformu-
lations, it can shift away from a list once it starts missing and draw
a fresh hit from another list instead. Each such hit resets the single
global consecutive-irrelevant counter, so the same global stopping
criterion is triggered later than under Adhoc Search, which has
only one list to exhaust. Forward Only Spawning and Bounded
Spawning fall between Adhoc Search and Query Reformulation
then MTF on both recall and judging cost, suggesting that on-the-fly
divergent query generation improves recall over a single ranked list
but is less judgment-efficient than committing to a batch of refor-
mulations upfront. Adhoc Search is the least efficient policy under
Figure 3: Mean recall (%) versus mean number of LLM rel-
evance judgments consumed per query, for each retrieval
pipeline (panels), judging budget (color), and policy (marker
shape), evaluated under a global judging budget.
every retrieval pipeline, indicating that judging a single ranked list
top-down makes poor use of a fixed judging budget. The retrieval
pipeline matters as much as the policy: SPLADE and the hybrid
pipeline recover more relevant documents per judgment than BM25
or TCT-ColBERT under every policy, mirroring the pattern already
observed for recall@𝑅in Figure 1.

7 The Future: Learning to Search
Learning to rank provides a useful analogy for our present pur-
pose. Treating ranking as a learning problem gave the field a more
systematic way to specify objectives, compare methods, and make
progress across a broad literature [ 7,27,38]. While some prior work
has also considered learning over longer search interactions and
sessions [ 40,41,59], LLM-guided search now seems to call for a
similarly compact way of describing the search process itself. As
these systems become more capable, it becomes increasingly im-
portant to understand what counts as an improvement and why.
Our goal is to provide that framework, creating a principled basis
for evaluation and comparison.
The policies described above illustrate how a variety of existing
methods can be expressed within the spawn/select framework. In
this sense, the framework plays a role analogous to that of learning
to rank: Before better methods can be learned, the search process
itself must be described in a form that makes evaluation and sys-
tematic improvement possible. The purpose of this paper is not to
propose new learning algorithms, but to provide that evaluation
formalization for LLM-guided search.
Within the framework of Section 3, a search policy 𝜋(𝜃 𝑡)de-
termines when to execute spawn and which ranking to select . The
reward formulation of Equation 2 then provides a natural objective
for optimizing this policy. Rather than specifying heuristics such as
MTF or fixed spawning rules, future work can instead learn policies
that directly optimize the desired reward signal. That reward may
reflect the discovery of relevant documents under a judging budget,
but it may also be defined in terms of information coverage, such as
the discovery of informational nuggets or aspects of an information
need This paper uses the language of sequential decision making to
define the problem clearly, but the work of learning strong search
policies remains for future work.
Our framework makes it possible to ask a range of concrete
learning questions about search policy. For example, a learned pol-
icy may decide when it is worth spawning a new ranking, which
existing ranking should be explored next, how effort should be al-
located across competing search trajectories, and when the search
has reached sufficient coverage to stop. Different reward defini-
tions then lead naturally to different search behaviors: one policy
may emphasize finding as many relevant documents as possible
within a judging budget, while another may emphasize coverage
of informational nuggets or aspects of an information need. In this
sense, the framework does not itself solve learning to search, but
it provides a representation in which such questions can be posed
and evaluated systematically.
Viewed in this way, modern search agents performing retrieval-
augmented generation or “deep research” can be interpreted as
policies operating within this framework. Queries are generated,
documents are examined, information is accumulated, and the pro-
cess continues until the system determines that the search has
reached sufficient coverage. The spawn/select abstraction therefore
provides a foundation for learning search policies that control how
information is gathered, allowing systems to optimize not only
which documents to read, but how the search itself unfolds. Rather
than designing search strategies by hand, the system can learn howto explore a collection efficiently while prioritizing the documents
and information that matter most.
8 Conclusion
We presented a sequence decision framework for LLM-guided
search that puts these search strategies on a common footing, mak-
ing them easier to describe, compare, evaluate, and eventually im-
prove.
In our framework, a search process is modeled as a policy op-
erating over a simple environment in which each step produces
an observation from a document. The policy chooses between two
actions: select , which continues to explore an existing ranking, and
spawn , which generates a new ranking through query reformu-
lation, reranking, or model-based retrieval strategies. Expressing
search in terms of these two actions provides a compact way to
describe a wide range of existing approaches.
Using our framework, classic high-recall retrieval and test col-
lection development strategies such as Move-to-Front pooling, In-
teractive Searching and Judging, and Continuous Active Learning
can all be interpreted as policies within a common framework. The
same formulation also shows how ideas from bandits and rein-
forcement learning might eventually be applied to the design of
search policies. This perspective is particularly relevant for modern
LLM-guided systems, in which large language models can gener-
ate queries, judge documents, and identify informational nuggets,
enabling increasingly autonomous search loops. More broadly, the
framework suggests a shift in perspective: rather than designing
search strategies by hand, we can begin to study how search poli-
cies themselves should be represented, compared, and ultimately
learned.
References
[1]Mustafa Abualsaud, Nimesh Ghelani, Haotian Zhang, Mark D. Smucker, Gordon V.
Cormack, and Maura R. Grossman. 2018. A System for Efficient High-Recall Re-
trieval. InThe 41st International ACM SIGIR Conference on Research & Development
in Information Retrieval. 1317–1320.
[2]Peter Auer, Nicolò Cesa-Bianchi, and Paul Fischer. 2002. Finite-time Analysis
of the Multiarmed Bandit Problem.Machine Learning47, 2–3 (2002), 235–256.
doi:10.1023/A:1013689704352
[3]N.J. Belkin, J. Perez Carballo, C. Cool, S. Lin, S.Y. Park, S.Y. Rieh, P. Savage, C.
Sikora, H. Xie, and J. Allan. 1997. Rutgers’ TREC-6 Interactive Track Experience.
InProceedings of the Sixth Text REtrieval Conference (TREC-6). National Institute
of Standards and Technology.
[4]N. J. Belkin, C. Cool, J. Head, J. Jeng, D. Kelly, S. Lin, L. Lobash, S. Y. Park, P.
Savage-Knepshield, and C. Sikora. 1999. Relevance Feedback versus Local Context
Analysis as Term Suggestion Devices: Rutgers’ TREC-8 Interactive Track Experi-
ence. InProceedings of the Eighth Text REtrieval Conference (TREC-8). National
Institute of Standards and Technology.
[5]Amin Bigdeli, Radin Hamidi Rad, Mert Incesu, Negar Arabzadeh, Charles L. A.
Clarke, and Ebrahim Bagheri. 2025. QueryGym: A Toolkit for Reproducible
LLM-Based Query Reformulation. arXiv:2511.15996 [cs.IR] https://arxiv.org/abs/
2511.15996
[6]Nikos I. Bosse, Jon Evans, Robert G. Gambee, Daniel Hnyk, Peter Mühlbacher,
Lawrence Phillips, Dan Schwarz, and Jack Wildman. 2025. Deep Research Bench:
Evaluating AI Web Research Agents.CoRRabs/2506.06287 (2025). doi:10.48550/
arXiv.2506.06287
[7]Christopher J. C. Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds,
Nicole Hamilton, and Gregory Hullender. 2005. Learning to Rank Using Gradient
Descent. InProceedings of the 22nd International Conference on Machine Learning.
ACM, 89–96. doi:10.1145/1102351.1102363
[8]Zijian Chen, Xueguang Ma, Shengyao Zhuang, Jimmy Lin, Akari Asai, and Victor
Zhong. 2026. AgentIR: Reasoning-Aware Retrieval for Deep Research Agents.
arXiv:2603.04384 [cs.CL] https://arxiv.org/abs/2603.04384

[9]Zijian Chen, Xueguang Ma, Shengyao Zhuang, Ping Nie, Kai Zou, Andrew Liu,
Joshua Green, Kshama Patel, Ruoxi Meng, Mingyi Su, Sahel Sharifymoghad-
dam, Yanxi Li, Haoran Hong, Xinyu Shi, Xuye Liu, Nandan Thakur, Crystina
Zhang, Luyu Gao, Wenhu Chen, and Jimmy Lin. 2025. BrowseComp-Plus: A
More Fair and Transparent Evaluation Benchmark of Deep-Research Agent.
arXiv:2508.06600 [cs.CL] https://arxiv.org/abs/2508.06600
[10] Charles L.A. Clarke, Maheedhar Kolla, Gordon V. Cormack, Olga Vechtomova,
Azin Ashkan, Stefan Büttcher, and Ian MacKinnon. 2008. Novelty and Diversity in
Information Retrieval Evaluation. InProceedings of the 31st Annual International
ACM SIGIR Conference on Research and Development in Information Retrieval.
659–666.
[11] Gordon V. Cormack and Maura R. Grossman. 2015. Autonomy and Reliabil-
ity of Continuous Active Learning for Technology-Assisted Review.CoRR
abs/1504.06868 (2015). https://arxiv.org/abs/1504.06868
[12] Gordon V. Cormack and Maura R. Grossman. 2015. Multi-Faceted Recall of
Continuous Active Learning for Technology-Assisted Review. InProceedings of
the 38th International ACM SIGIR Conference on Research and Development in
Information Retrieval. ACM, 763–766. doi:10.1145/2766462.2767771
[13] Gordon V. Cormack and Maura R. Grossman. 2016. Engineering Quality and
Reliability in Technology-Assisted Review. InProceedings of the 39th International
ACM SIGIR Conference on Research and Development in Information Retrieval.
75–84. doi:10.1145/2911451.2911510
[14] Gordon V. Cormack and Maura R. Grossman. 2016. Scalability of Continuous
Active Learning for Reliable High-Recall Text Classification. InProceedings of the
25th ACM International Conference on Information and Knowledge Management.
ACM, 1039–1048. doi:10.1145/2983323.2983776
[15] Gordon V. Cormack and Maura R. Grossman. 2017. Navigating Imprecision in
Relevance Assessments on the Road to Total Recall: Roger and Me. InProceedings
of the 40th International ACM SIGIR Conference on Research and Development in
Information Retrieval. ACM, 5–14. doi:10.1145/3077136.3080812
[16] Gordon V. Cormack, Christopher R. Palmer, and Charles L. A. Clarke. 1998.
Efficient Construction of Large Test Collections. InProceedings of the 21st Annual
International ACM SIGIR Conference on Research and Development in Information
Retrieval. ACM, 282–289. doi:10.1145/290941.291009
[17] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021. SPLADE:
Sparse Lexical and Expansion Model for First Stage Ranking. InProceedings
of the 44th International ACM SIGIR Conference on Research and Development
in Information Retrieval(Virtual Event, Canada)(SIGIR ’21). Association for
Computing Machinery, New York, NY, USA, 2288–2292. doi:10.1145/3404835.
3463098
[18] Aoran Gan, Hao Yu, Kai Zhang, Qi Liu, Wenyu Yan, Zhenya Huang, Shiwei Tong,
and Guoping Hu. 2025. Retrieval Augmented Generation Evaluation in the Era of
Large Language Models: A Comprehensive Survey.CoRRabs/2504.14891 (2025).
https://arxiv.org/abs/2504.14891
[19] Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Cha-
ganty, Yicheng Fan, Vincent Y. Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan,
and Kelvin Guu. 2023. RARR: Researching and Revising What Language Mod-
els Say, Using Language Models. (2023). arXiv:2210.08726 [cs.CL] https:
//arxiv.org/abs/2210.08726
[20] Maura R. Grossman, Gordon V. Cormack, and Adam Roegiest. 2016. TREC 2016
Total Recall Track Overview. InProceedings of The Twenty-Fifth Text REtrieval
Conference (TREC 2016).
[21] Mehdi Hosseini, Ingemar J. Cox, Natasa Milic-Frayling, Milad Shokouhi, and
Emine Yilmaz. 2012. An Uncertainty-Aware Query Selection Model for Evaluation
of IR Systems. InProceedings of the 35th International ACM SIGIR Conference on
Research and Development in Information Retrieval. ACM, 901–910.
[22] Tiansheng Hu, Yilun Zhao, Canyu Zhang, Arman Cohan, and Chen Zhao. 2026.
SAGE: Benchmarking and Improving Retrieval for Deep Research Agents.CoRR
abs/2602.05975 (2026). https://arxiv.org/abs/2602.05975
[23] Siqing Huo, Negar Arabzadeh, and Charles L. A. Clarke. 2023. Retrieving Sup-
porting Evidence for Generative Question Answering. InProceedings of the 2023
Annual International ACM SIGIR Conference on Research and Development in In-
formation Retrieval in the Asia Pacific Region (SIGIR-AP ’23). ACM, New York, NY,
USA, 11–20. doi:10.1145/3624918.3625336
[24] Gautier Izacard and Edouard Grave. 2021. Leveraging Passage Retrieval with
Generative Models for Open Domain Question Answering. InProceedings of the
16th Conference of the European Chapter of the Association for Computational
Linguistics: Main Volume. Association for Computational Linguistics, 874–880.
doi:10.18653/v1/2021.eacl-main.74
[25] Kalervo Järvelin and Jaana Kekäläinen. 2002. Cumulated Gain-Based Evaluation
of IR Techniques.ACM Transactions on Information Systems20, 4 (2002), 422–446.
[26] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Ö. Arık, Dong Wang,
Hamed Zamani, and Jiawei Han. 2025. Search-R1: Training LLMs to Reason and
Leverage Search Engines with Reinforcement Learning. InProceedings of the
Second Conference on Language Modeling. Montreal, Canada. https://openreview.
net/forum?id=Rwhi91ideu
[27] Thorsten Joachims. 2002. Optimizing Search Engines Using Clickthrough Data.
InProceedings of the Eighth ACM SIGKDD International Conference on KnowledgeDiscovery and Data Mining. ACM, 133–142. doi:10.1145/775047.775067
[28] Diane Kelly. 2009. Methods for Evaluating Interactive Information Retrieval
Systems with Users.Foundations and Trends in Information Retrieval3, 1–2 (2009),
1–224. doi:10.1561/1500000012
[29] Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav
Santhanam, Sri Vardhamanan A, Saiful Haq, Ashutosh Sharma, Thomas T.
Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, and Christopher Potts.
2024. DSPy: Compiling Declarative Language Model Calls into State-of-the-Art
Pipelines. InThe Twelfth International Conference on Learning Representations.
https://openreview.net/forum?id=sY5N0zY5Od
[30] Weronika Łajewska and Krisztian Balog. 2025. GINGER: Grounded Information
Nugget-Based Generation of Responses. InProceedings of the 48th International
ACM SIGIR Conference on Research and Development in Information Retrieval.
ACM, 2723–2727. doi:10.1145/3726302.3730166
[31] David D. Lewis and William A. Gale. 1994. A Sequential Algorithm for Train-
ing Text Classifiers. InProceedings of the 17th Annual International ACM SIGIR
Conference on Research and Development in Information Retrieval. Springer, 3–12.
[32] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation
for Knowledge-Intensive NLP Tasks. InAdvances in Neural Information Pro-
cessing Systems, Vol. 33. https://proceedings.neurips.cc/paper/2020/hash/
6b493230205f780e1bc26945df7481e5-Abstract.html
[33] Lihong Li, Wei Chu, John Langford, and Robert E. Schapire. 2010. A Contextual-
Bandit Approach to Personalized News Article Recommendation. InProceedings
of the 19th International Conference on World Wide Web. ACM, 661–670. doi:10.
1145/1772690.1772758
[34] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian
Zhang, and Zhicheng Dou. 2025. Search-o1: Agentic Search-Enhanced Large
Reasoning Models. InProceedings of the 2025 Conference on Empirical Methods in
Natural Language Processing. Association for Computational Linguistics, Suzhou,
China, 5420–5438. doi:10.18653/v1/2025.emnlp-main.276
[35] Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong Yang, Ronak Pradeep,
and Rodrigo Nogueira. 2021. Pyserini: A Python Toolkit for Reproducible Infor-
mation Retrieval Research with Sparse and Dense Representations. InProceedings
of the 44th International ACM SIGIR Conference on Research and Development in
Information Retrieval. ACM, 2356–2362. doi:10.1145/3404835.3463238
[36] Jimmy Lin and Pengyi Zhang. 2007. Deconstructing Nuggets: The Stability and
Reliability of Complex Question Answering Evaluation. InProceedings of the
30th Annual International ACM SIGIR Conference on Research and Development in
Information Retrieval. ACM, 327–334. doi:10.1145/1277741.1277799
[37] Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy Lin. 2021. In-Batch Negatives
for Knowledge Distillation with Tightly-Coupled Teachers for Dense Retrieval.
InProceedings of the 6th Workshop on Representation Learning for NLP (RepL4NLP-
2021), Anna Rogers, Iacer Calixto, Ivan Vulić, Naomi Saphra, Nora Kassner,
Oana-Maria Camburu, Trapit Bansal, and Vered Shwartz (Eds.). Association for
Computational Linguistics, Online, 163–173. doi:10.18653/v1/2021.repl4nlp-1.17
[38] Tie-Yan Liu. 2009. Learning to Rank for Information Retrieval.Foundations and
Trends in Information Retrieval3, 3 (2009), 225–331. doi:10.1561/1500000016
[39] David E. Losada, Javier Parapar, and Alvaro Barreiro. 2017. Multi-armed Ban-
dits for Adjudicating Documents in Pooling-Based Evaluation of Information
Retrieval Systems.Information Processing & Management53, 5 (2017), 1005–1025.
doi:10.1016/j.ipm.2017.04.005
[40] Jiyun Luo, Xuchu Dong, and Hui Yang. 2015. Learning to Reinforce Search
Effectiveness. InProceedings of the 2015 ACM SIGIR International Conference on
the Theory of Information Retrieval. ACM, 271–280.
[41] Jiyun Luo, Xuchu Dong, and Hui Yang. 2015. Session Search by Direct Policy
Learning. InProceedings of the 2015 ACM SIGIR International Conference on the
Theory of Information Retrieval. ACM, 261–270.
[42] Craig Macdonald, Nicola Tonellotto, Sean MacAvaney, and Iadh Ounis. 2021.
PyTerrier: Declarative Experimentation in Python from BM25 to Dense Retrieval.
InProceedings of the 30th ACM International Conference on Information and
Knowledge Management. ACM, 4526–4533. doi:10.1145/3459637.3482013
[43] Gary Marchionini. 2006. Exploratory Search: From Finding to Understanding.
Commun. ACM49, 4 (2006), 41–46. doi:10.1145/1121949.1121979
[44] Chuan Meng, Litu Ou, Sean MacAvaney, and Jeff Dalton. 2026. Revisiting Text
Ranking in Deep Research.CoRRabs/2602.21456 (2026). https://arxiv.org/abs/
2602.21456
[45] Baharan Nouriinanloo and Maxime Lamothe. 2024. Re-Ranking Step by Step:
Investigating Pre-Filtering for Re-Ranking with Large Language Models.CoRR
abs/2406.18740 (2024). doi:10.48550/arXiv.2406.18740
[46] Roxana Petcu, Evangelos Kanoulas, and Maarten de Rijke. 2026. SubSearch:
Intermediate Rewards for Unsupervised Guided Reasoning in Complex Retrieval.
arXiv:2604.07415 [cs.IR] https://arxiv.org/abs/2604.07415
[47] Jeremy Pickens, Gene Golovchinsky, Chirag Shah, Pernilla Qvarfordt, and Mari-
beth Back. 2008. Algorithmic mediation for collaborative exploratory search. In
Proceedings of the 31st Annual International ACM SIGIR Conference on Research
and Development in Information Retrieval. 315–322. doi:10.1145/1390334.1390389

[48] Peter L. T. Pirolli. 2007.Information Foraging Theory: Adaptive Interaction with
Information. Oxford University Press. doi:10.1093/acprof:oso/9780195173321.001.
0001
[49] Ronak Pradeep, Rodrigo Nogueira, and Jimmy Lin. 2021. The Expando-Mono-
Duo Design Pattern for Text Ranking with Pretrained Sequence-to-Sequence
Models. arXiv:2101.05667 [cs.IR] https://arxiv.org/abs/2101.05667
[50] Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. 2023. RankVicuna:
Zero-Shot Listwise Document Reranking with Open-Source Large Language
Models.CoRRabs/2309.15088 (2023). doi:10.48550/arXiv.2309.15088
[51] Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. 2023. RankZephyr: Ef-
fective and Robust Zero-Shot Listwise Reranking is a Breeze!CoRRabs/2312.02724
(2023). doi:10.48550/arXiv.2312.02724
[52] Ronak Pradeep, Nandan Thakur, Sahel Sharifymoghaddam, Eric Zhang, Ryan
Nguyen, Daniel Campos, Nick Craswell, and Jimmy Lin. 2025. Ragnarök: A
Reusable RAG Framework and Baselines for TREC 2024 Retrieval-Augmented
Generation Track. InAdvances in Information Retrieval (Lecture Notes in Computer
Science). doi:10.1007/978-3-031-88708-6_9
[53] Ronak Pradeep, Nandan Thakur, Shivani Upadhyay, Daniel Campos, Nick
Craswell, Ian Soboroff, Hoa Trang Dang, and Jimmy Lin. 2025. The Great
Nugget Recall: Automating Fact Extraction and RAG Evaluation with Large
Language Models. InProceedings of the 48th International ACM SIGIR Con-
ference on Research and Development in Information Retrieval. ACM, 180–190.
doi:10.1145/3726302.3730090
[54] Zhen Qin, Rolf Jagerman, Kai Hui, Honglei Zhuang, Junru Wu, Le Yan, Jiaming
Shen, Tianqi Liu, Jialu Liu, Donald Metzler, Xuanhui Wang, and Michael Ben-
dersky. 2024. Large Language Models are Effective Text Rankers with Pairwise
Ranking Prompting. InFindings of the Association for Computational Linguistics:
NAACL 2024. Association for Computational Linguistics, Mexico City, Mexico,
1504–1518. doi:10.18653/v1/2024.findings-naacl.97
[55] Md Mustafizur Rahman, Mucahid Kutlu, and Matthew Lease. 2019. Constructing
Test Collections using Multi-armed Bandits and Active Learning. InThe World
Wide Web Conference. 3158–3164.
[56] Tetsuya Sakai and Ruihua Song. 2011. Evaluating Diversified Search Results
Using Per-Intent Graded Relevance. InProceedings of the 34th International ACM
SIGIR Conference on Research and Development in Information Retrieval. ACM,
1043–1052. doi:10.1145/2009916.2010055
[57] Rodrygo L. T. Santos, Jie Peng, Craig Macdonald, and Iadh Ounis. 2010. Explicit
Search Result Diversification through Sub-queries. InAdvances in Information
Retrieval (Lecture Notes in Computer Science, Vol. 5993). Springer, 87–99. doi:10.
1007/978-3-642-12275-0_11
[58] Sahel Sharifymoghaddam, Lingwei Gu, Yijun Ge, and Jimmy Lin. 2026. Projecting
BrowseComp-Plus onto ClimbMix: Toward More Realistic Corpora for Agentic
Search. arXiv:2608.20317 [cs.IR] https://arxiv.org/abs/2608.20317
[59] Mark D. Smucker and Charles L. A. Clarke. 2016. Modeling Optimal Switching
Behavior. InProceedings of the 2016 ACM Conference on Human Information
Interaction and Retrieval. ACM, 317–320. doi:10.1145/2854946.2854981
[60] Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin
Chen, Dawei Yin, and Zhaochun Ren. 2023. Is ChatGPT Good at Search? Investi-
gating Large Language Models as Re-Ranking Agents. InProceedings of the 2023
Conference on Empirical Methods in Natural Language Processing. Association for
Computational Linguistics, Singapore, 14918–14937. doi:10.18653/v1/2023.emnlp-
main.923
[61] Jaime Teevan, Kevyn Collins-Thompson, Ryen W. White, Susan T. Dumais, and
Yubin Kim. 2013. Slow Search: Information Retrieval without Time Constraints.
InProceedings of the 2013 Symposium on Human-Computer Interaction and Infor-
mation Retrieval. ACM, 1:1–1:10. doi:10.1145/2528394.2528395
[62] William R. Thompson. 1933. On the Likelihood that One Unknown Probability
Exceeds Another in View of the Evidence of Two Samples.Biometrika25, 3/4
(1933), 285–294. doi:10.1093/biomet/25.3-4.285
[63] Ellen M. Voorhees and Hoa Trang Dang. 2005. Overview of the TREC 2005
Question Answering Track. InProceedings of the Fourteenth Text REtrieval Con-
ference (TREC 2005). National Institute of Standards and Technology. https:
//trec.nist.gov/pubs/trec14/papers/QA.OVERVIEW.pdf
[64] Yufei Wang, Qiwei Ye, and Tie-Yan Liu. 2019. Beyond Exponentially Discounted
Sum: Automatic Learning of Return Function.arXivabs/1905.11591 (2019).
[65] Ryen W. White and Resa A. Roth. 2009.Exploratory Search: Be-
yond the Query-Response Paradigm. Morgan & Claypool. doi:10.2200/
S00174ED1V01Y200901ICR003
[66] Yunjia Xi, Jianghao Lin, Yongzhao Xiao, Zheli Zhou, Rong Shan, Te Gao, Ji-
achen Zhu, Weiwen Liu, Yong Yu, and Weinan Zhang. 2025. A Survey of LLM-
based Deep Search Agents: Paradigm, Optimization, Evaluation, and Challenges.
arXiv:2508.05668 [cs.IR] https://arxiv.org/abs/2508.05668
[67] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language
Models. InThe Eleventh International Conference on Learning Representations.
https://openreview.net/forum?id=WE_vluYUL-X
[68] Emine Yilmaz, Milad Shokouhi, Nick Craswell, and Stephen Robertson. 2010.
Expected Browsing Utility for Web Search Evaluation. InProceedings of the 19thACM International Conference on Information and Knowledge Management. ACM,
1561–1564. doi:10.1145/1871437.1871672
[69] Boer Zhang, Mingyan Wu, Dongzhuoran Zhou, Yuqicheng Zhu, Wendong Fan,
Puzhen Zhang, Zifeng Ding, Guohao Li, and Yuan He. 2026. EigentSearch-
Q+: Enhancing Deep Research Agents with Structured Reasoning Tools.CoRR
abs/2604.07927 (2026). https://arxiv.org/abs/2604.07927
[70] Haotian Zhang, Mustafa Abualsaud, Nimesh Ghelani, Mark D. Smucker, Gor-
don V. Cormack, and Maura R. Grossman. 2018. Effective User Interaction for
High-Recall Retrieval: Less is More. InProceedings of the 27th ACM Interna-
tional Conference on Information and Knowledge Management. ACM, 187–196.
doi:10.1145/3269206.3271796
[71] Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui
Lu, and Pengfei Liu. 2025. DeepResearcher: Scaling Deep Research via Reinforce-
ment Learning in Real-world Environments. InProceedings of the 2025 Conference
on Empirical Methods in Natural Language Processing. Association for Computa-
tional Linguistics, Suzhou, China, 414–431. doi:10.18653/v1/2025.emnlp-main.22
[72] Shengyao Zhuang, Honglei Zhuang, Bevan Koopman, and Guido Zuccon. 2024.
A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with
Large Language Models. InProceedings of the 47th International ACM SIGIR
Conference on Research and Development in Information Retrieval. ACM, 38–47.
doi:10.1145/3626772.3657813

Appendix
A Batch Query Reformulation Prompt
classBatchReformSig(dspy.Signature):
query = dspy.InputField(
desc="Original user search query"
)
docs = dspy.InputField(
desc="List of retrieved documents for context"
)
improved_queries = dspy.OutputField(
desc="JSON-parsable list of 8 distinct search queries
. These can be better phrasings of the original
intent or targeted queries for different sub-
aspects, depending on what will best improve
retrieval."
)B Iterative Divergent Search Prompt
classIterativeDivergentSearchSig(dspy.Signature):
original_query = dspy.InputField(
desc="Original user search query"
)
original_top_retrieved_docs = dspy.InputField(
desc="Top docs for original query"
)
search_history = dspy.InputField(
desc="Json list of past search history: each query
searched and the corresponding top retrieved
documents with their relevance score."
)
exploratory_query = dspy.OutputField(
desc="A novel query designed to maximize retrieval of
unique, previously unseen documents relevant to
the original query."
)
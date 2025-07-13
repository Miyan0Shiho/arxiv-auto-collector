# SHARP: Shared State Reduction for Efficient Matching of Sequential Patterns

**Authors**: Cong Yu, Tuo Shi, Matthias Weidlich, Bo Zhao

**Published**: 2025-07-07 10:57:55

**PDF URL**: [http://arxiv.org/pdf/2507.04872v1](http://arxiv.org/pdf/2507.04872v1)

## Abstract
The detection of sequential patterns in data is a basic functionality of
modern data processing systems for complex event processing (CEP), OLAP, and
retrieval-augmented generation (RAG). In practice, pattern matching is
challenging, since common applications rely on a large set of patterns that
shall be evaluated with tight latency bounds. At the same time, matching needs
to maintain state, i.e., intermediate results, that grows exponentially in the
input size. Hence, systems turn to best-effort processing, striving for maximal
recall under a latency bound. Existing techniques, however, consider each
pattern in isolation, neglecting the optimization potential induced by state
sharing in pattern matching.
  In this paper, we present SHARP, a library that employs state reduction to
achieve efficient best-effort pattern matching. To this end, SHARP incorporates
state sharing between patterns through a new abstraction, coined
pattern-sharing degree (PSD). At runtime, this abstraction facilitates the
categorization and indexing of partial pattern matches. Based thereon, once a
latency bound is exceeded, SHARP realizes best-effort processing by selecting a
subset of partial matches for further processing in constant time. In
experiments with real-world data, SHARP achieves a recall of 97%, 96% and 73%
for pattern matching in CEP, OLAP, and RAG applications, under a bound of 50%
of the average processing latency.

## Full Text


<!-- PDF content starts -->

arXiv:2507.04872v1  [cs.DB]  7 Jul 2025Sharp : Shar ed State Reduction for Efficient Matching of
Sequential P atterns
Cong Yu∗
Aalto University
Finland
cong.yu@aalto.fiTuo Shi∗
Aalto University
Finland
tuo.shi@aalto.fi
Matthias Weidlich
Humboldt-Universität zu Berlin
Germany
matthias.weidlich@hu-berlin.deBo Zhao
Aalto University
Finland
bo.zhao@aalto.fi
ABSTRACT
The detection of sequential patterns in data is a basic functionality
of modern data processing systems for complex event processing
(CEP), OLAP, and retrieval-augmented generation (RAG). In prac-
tice, pattern matching is challenging, since common applications
rely on a large set of patterns that shall be evaluated with tight
latency bounds. At the same time, matching needs to maintain state,
i.e., intermediate results, that grows exponentially in the input size.
Hence, systems turn to best-effort processing, striving for maximal
recall under a latency bound. Existing techniques, however, con-
sider each pattern in isolation, neglecting the optimization potential
induced by state sharing in pattern matching.
In this paper, we present Sharp , a library that employs state
reduction to achieve efficient best-effort pattern matching. To this
end,Sharp incorporates state sharing between patterns through a
new abstraction, coined pattern-sharing degree (PSD). At runtime,
this abstraction facilitates the categorization and indexing of partial
pattern matches. Based thereon, once a latency bound is exceeded,
Sharp realizes best-effort processing by selecting a subset of partial
matches for further processing in constant time. In experiments
with real-world data, Sharp achieves a recall of 97%, 96% and 73%
for pattern matching in CEP, OLAP, and RAG applications, under
a bound of 50% of the average processing latency.
PVLDB Reference Format:
Cong Yu, Tuo Shi, Matthias Weidlich, and Bo Zhao. Sharp : Shar ed State
Reduction for Efficient Matching of Sequential P atterns. PVLDB, 14(1):
XXX-XXX, 2020.
doi:XX.XX/XXX.XX
PVLDB Artifact Availability:
The source code, data, and/or other artifacts have been made available at
URL_TO_YOUR_ARTIFACTS.
∗Both authors contributed equally to this research.
This work is licensed under the Creative Commons BY-NC-ND 4.0 International
License. Visit https://creativecommons.org/licenses/by-nc-nd/4.0/ to view a copy of
this license. For any use beyond those covered by this license, obtain permission by
emailing info@vldb.org. Copyright is held by the owner/author(s). Publication rights
licensed to the VLDB Endowment.
Proceedings of the VLDB Endowment, Vol. 14, No. 1 ISSN 2150-8097.
doi:XX.XX/XXX.XX1 INTRODUCTION
The detection of sequential patterns with low latency is a data
management functionality with a wide range of applications. That
is, complex event processing (CEP) engines detect user-defined
patterns over high-velocity event streams [ 5,12,42]; relational
databases for online analytical processing (OLAP) evaluate queries
featuring the MATCH_RECOGNIZE operator that takes a set of tuples
as input and returns all matches of the given pattern [ 17,21,59];
and graph databases evaluate regular path queries over knowledge
graphs to facilitate retrieval-augmented generation (RAG) [ 2,7,26,
32,34]. In all these applications, patterns define an ordering of data
elements along with further correlation and aggregation predicates
over the elements’ of attribute values (i.e., their payload).
In practice, pattern matching is challenging for several rea-
sons. First, many applications enforce application-specific latency
bounds [ 21,51,58], e.g., as part of service level agreements [ 25].
Second, the evaluation of common patterns is computationally hard,
since it requires maintaining a state, i.e., a set of partial pattern
matches, that grows exponential in the size of the input [ 54]. The
latter means that the distribution of the processed data may in-
crease the computational load drastically for short peak times, so
that exhaustive processing becomes infeasible [ 16,50]. In this case,
systems for pattern matching resort to best-effort processing. That
is, they strive for maximizing the recall, i.e., the number of detected
pattern matches, while satisfying a latency bound [9, 58].
The above techniques for best-effort processing consider each
pattern in isolation. However, this is problematic given that com-
mon applications for pattern matching require the evaluation of
many patterns simultaneously. The reason being that matching of
several variations of patterns promises to increase the result quality
of downstream applications [ 19,26]. We illustrate this aspect with
a specific example from the field of To this end, we use an example
of graph retrieval-augmented generation (GraphRAG), as follows.
Consider an application in which the inference of a large lan-
guage model (LLM) is augmented using an external knowledge
graph (KG), as illustrated in Fig.1a: 1The application submits (i)
a question for the LLM (from the MetaQA benchmark [ 56]) and
(ii) a query prompt to generate path queries for the KG. 2The
query and prompt are sent to an LLM (here, Llama-2 (7B), tuned
on WebQSP [ 53] and CWQ [ 56]) The LLM then generates a set of
path queries and selects the top-k (e.g., P1, P2 and P3), e.g., using
1

6Path Queries Prompt ：Knowledge
Graph
Generate a valid relationship path that can
help answer the following question.
MetaQA  Query #53: Path Queries Prompt ：Path Queries
What other movies did Alila ’s scriptwriter write?P1: Alila → written_by  → Amos Gitai 
             ← written_by  ← Kedma
       Alila  → written_by  → Amos Gitai 
             ← written_by  ← Kadosh
       Alila  → directed_by  ← Amos Gitai
P2: Alila → has_genra  ← Drama
       Alila  → directed_by  ← Amos Gitai
P3: Alila → directed_by  ← Amos GitaiRetrieved Path
P1: film.written_by  → film.written_by
P2: common.topic.notable_types  
        → common.topic.notable_types
P3: film_story_contributor.film_story_credits 
        → film.story_by
Amos GitaiAlila
Kedma Kadoshwritten_by written_byYaël Abecassis
Uri KlauznerDrama
starred_actorsstarred_actors
has_genre
release_year
2003directed_by
written_by
LLMs (GPT, Llama...) Based on the retrieved  paths, please
answer the given question. Shared Path
123 4
5
Kedma, Kadosh are written by Amos Gitai.(a)Overview of a GraphRAG pipeline.
Accuracy Recall01020304050Rate (%)Single-Path-Query
Multi-Path-Query(b)Result quality.
101
100101102103Latency (ms, log scale)Single-Path-Query
Multi-Path-Query (c)End-to-end latency.
Fig. 1: Example illustrating the evaluation of multiple sequential patterns in an application for graph retrieval-augmented generation
cosine similarity and beam-search [ 49].3The path queries are
evaluated over the KG and 4the resulting paths are 5integrated
in an answer generation prompt to 6generate the final response.
Figure 1b and 1c highlight that the use of multiple path patterns
improves the result quality by 2.80 ˆin accuracy and 2.51 ˆin
recall. Yet, this comes with a computational cost, as the latency
increases by four orders of magnitudes compared to the use of a
single pattern, which is due to the generation of 3,962 ˆmore partial
matches. Even when adopting optimizations for state sharing [ 3,
4], the number of generated partial matches increases by 2,001 ˆ.
In large-scale industrial applications, such as those reported for
GraphRAG by Microsoft [ 31,41], Amazon [ 8], or Siemens [ 43],
these computational challenges are further amplified.
In this paper, we study how to realize best-effort processing
of pattern workloads, when incorporating state sharing in their
evaluation. This is difficult as the state of pattern matching affects
the processing results and the evaluation performance, as follows:
(1) Partial matches differ in how they contribute to the result of a
single pattern and in their consumption of computational resources.
(2) Partial matches differ in their importance for multiple pat-
terns, which may further be weighted by explicit utility definitions.
(3) The relation between partial matches and patterns is subject
to changes, e.g., due to drift in data distributions.
We present Sharp , a state management library for efficient best-
effort pattern matching. It addresses the above challenges based on
the following contributions:
(1) Efficient pattern-sharing assessment. Sharp captures shar-
ing information per partial match using a new abstraction called
pattern-sharing degree (PSD). It keeps track of how different pat-
terns share a partial match in terms of overlapping sub-patterns and
enables efficient querying of this information for an exponentially
growing set of partial matches. The structure of the PSD is derived
from the pattern execution plan. At runtime, Sharp relies on this
structure to cluster the generated partial matches and, through a
bitmap-based index, enables their retrieval in constant time.
(2) Efficient cost model for state selection. For each partial
match, Sharp examines its contribution to all patterns and its com-
putational overhead, which determines the processing latency. To
achieve this, Sharp maintains a cost model to estimate the num-
ber of complete matches that each partial match may generate aswell as the runtime and memory footprint caused by it. Sharp up-
dates the cost model incrementally and facilitates a lookup of the
contribution and consumption per partial match in constant time.
(3) State reduction problem formulation and optimization.
We formulate the problem of satisfying latency bounds in pattern
evaluation as a state reduction problem: Upon exhausting a latency
bound, Sharp selects a set of partial matches for further processing
based on a multi-objective optimization problem. Sharp limits the
overhead of solving this problem by lifting the pattern-sharing
degree and the cost model to a coarse granularity by clustering,
and by employing a greedy approximation strategy for the multi-
objective optimization space.
We have implemented Sharp in C++ and Python. Sharp has
been evaluated for three applications that rely on pattern matching,
i.e., complex event processing (CEP) over event streams, online
analytical processing (OLAP) using MATCH_RECOGNIZE queries, and
GraphRAG. In a comprehensive experimental evaluation with real-
world data, we observe that Sharp achieves a recall of 97%, 96%
and 73% for pattern matching in CEP, OLAP, and GraphRAG appli-
cations, when enforcing a bound of 50% of the average processing
latency. Compared to baseline strategies for state management,
Sharp significantly improves the recall by 11.25ˆfor CEP, 4.5ˆ
for OLAP, and 1.51ˆfor GraphRAG.
2 BACKGROUND
In this section, we provide further background details on three
common applications of pattern matching.
2.1 Complex Event Processing
Complex event processing (CEP) aims at identifying predefined
patterns of interests in streams of event [ 5,12,42]. For instance,
in finance, CEP enables banks to detect suspicious behavior in
financial transactions (e.g., a credit card is suddenly used in differ-
ent locations) or to react to certain stock price movements (e.g.,
sequences of increasing or decreasing prices over a specific period).
In CEP, a set of patterns is evaluated over an unbounded sequence
of typed events, each being a tuple of attribute values. Patterns,
which are potentially weighted in terms of their importance, are
defined as a combination of primitive event types, operators, pred-
icates, and a time window. These operators include conjunction,
sequencing ( SEQ) that enforces a temporal order of events of certain
2

types, Kleene closure ( KL) that accepts one or more events of a type,
and negation ( NEG) requiring the absence of events of a specific type.
Several execution models have been proposed for CEP. In an
automata-based model [ 52], partial matches denote partial runs of
an automaton that encodes the required event occurrences. Upon
the arrival of an event, it is checked if partial runs advance in the
automaton, while the state transitions are guarded by predicates
that encode data conditions and the time window. In a tree-based
model [ 30], events are inserted into a hierarchy of input buffers that
are guarded by predicates. Evaluation then proceeds from the leave
buffers of the tree to the root, filling operator buffers with event
sequences derived from their children. In either model, the state, i.e.,
the partial runs of the automaton or the buffered event sequences,
may grow exponentially in the number of processed events. The
reason being that the automaton may be non-deterministic and
that a tree operator may enumerate all subsequences of events.
2.2 Row Pattern Matching in OLAP
As part of online analytical processing (OLAP), multi-dimensional
data analysis may rely on pattern matching. Specifically, SQL:2016
introduced the MATCH_RECOGNIZE clause [ 17] for pattern matching
over the rows of a table or view, which potentially results from
another SQL query. Many data processing platforms support this
clause, including Oracle, Apache Flink, Azure Streaming Analytics,
Snowflake, and Trino [6, 15, 22, 35, 47].
In essence, the MATCH_RECOGNIZE clause includes statements to
define types of rows based on conditions over their attribute values
and to define a pattern as a regular expression over these types. The
pattern is then evaluated for a certain partition of the input tuples
that are ordered by a set of attributes. Semantics are fine-tuned
by statements that, for instance, control whether matches may be
overlapping and how the result per match is constructed.
Common execution models for row pattern matching are based
on automata and, as such, similar to those discussed for the evalu-
ation of CEP patterns. That is, depending on the structure of the
regular expression in the MATCH_RECOGNIZE clause, a deterministic
or non-deterministic automaton is constructed, which is then used
to process tuples while scanning the table or view used as input.
2.3 Regular Path Queries in GraphRAG
Retrieval-augmented generation (RAG) emerged as a paradigm for
augmenting generative models with external knowledge. By in-
tegrating some retrieval functionality with sequence-to-sequence
models, RAG promises to improve the quality of the generated re-
sult. While traditional RAG assumes that background knowledge is
available as plain text, GraphRAG enables the integration of hierar-
chical representations of background knowledge, typically given as
knowledge graphs [ 31,41]. Here, nodes represent entities, concepts,
or documents, and edges signify semantic relationships between
them. This graph-based structure facilitates a more expressive inte-
gration of background knowledge and generative models:
‚Contextual Navigation: Instead of retrieving isolated text
snippets, GraphRAG allows queries to explore paths in the
knowledge graph to enable contextual reasoning.Tab. 1: Main Notations
Notations Explaination
𝑆“x𝑑1,𝑑2,...y The input data sequence.
𝑆p...𝑘q The prefix𝑆up to𝑘.
P“t𝑃1,...,𝑃 𝑛u The set of𝑛patterns.
𝑃𝑖,𝑃𝑖,𝑗 A pattern and one of its sub-pattern.
¯P Set of all sub-patterns of patterns in P.
𝐺 The execution plan DAG of all patterns.
PMp𝑘q,CMp𝑘q The set of PMs and matches of all patterns over 𝑆p...𝑘q.
PM𝑃𝑖p𝑘q,CM𝑃𝑖p𝑘qThe set of PMs and matches of 𝑃𝑖over𝑆p...𝑘q.
‚Semantic Structure: Hierarchical structures in the graph pro-
vide a semantically meaningful organization of knowledge,
reducing redundancy and noise in retrieved results.
‚Explainability and Traceability: By grounding responses in
graph paths, GraphRAG improves interpretability, allowing
users to trace the reasoning behind the generated answers.
GraphRAG introduces the need for effective mechanisms to query
and navigate a knowledge graph. Here, Regular Path Queries (RPQs)
play a central role. They specify patterns over sequences of edges,
i.e., denote a regular expression over edge labels, to retrieve sub-
structures of the graph for downstream generation tasks. This ca-
pability is particularly valuable in GraphRAG, where path-based
retrieval enables the integration of complex relations and contextual
dependencies as encoded in a knowledge graph [26, 32, 48].
Again, the evaluation of the respective sequential patterns is
commonly approached using automata-based execution models.
That is, a product of the automata of the graph and the pattern is
constructed, which is then searched for accepting states.
3 PROBLEM FORMULATION
We first present a data and execution model for pattern matching
(§ 3.1), before introducing the state reduction problem for best-effort
pattern matching (§ 3.2). Tab. 1 summarizes our notations.
3.1 Model for Pattern Matching
Data Sequence. The input of pattern matching is a sequence of data
elements𝑆“x𝑑1,𝑑2,...y. Each data element 𝑑𝑘“x𝑎1,...,𝑎𝑚y
is an instance of a data schema defined by a finite sequence of
attributes𝐴“x𝐴1,...,𝐴𝑚y, where𝑎𝑖is the value corresponding
to attribute 𝐴𝑖. We further define a prefix of the data sequence 𝑆
up to index 𝑘as𝑆p...𝑘q“x𝑑1,...,𝑑𝑘yand the suffix of the data
sequence𝑆starting from index 𝑘as𝑆p𝑘...q“x𝑑𝑘,𝑑𝑘`1,...y.
Patterns & Complete Match. Pattern matching evaluates a set
of𝑛patterns P“t𝑃1,...,𝑃𝑛uon a data sequence 𝑆, ensuring that
each pattern 𝑃𝑖is detected within its corresponding latency bound
𝐿𝑖. A pattern𝑃𝑖defines a sequence of data elements that satisfy a
set of predicate constraints 𝐶𝑖, including a time window, ordering
of the elements, and value-based constraints on data attributes.
Evaluating𝑃𝑖over𝑆“x𝑑1,𝑑2,...ycreates complete matches,
each being a finite subsequence x𝑑1
1,...,𝑑1𝑚yof𝑆that preserves the
sequence order (for 𝑑1
𝑖“𝑑𝑘and𝑑1
𝑗“𝑑𝑙it holds that 𝑖ă𝑗implies
𝑘ă𝑙) and satisfies all predicate constraints in 𝐶𝑖. We denote the
set of complete matches of 𝑃𝑖over𝑆p...𝑘qasCM𝑃𝑖p𝑘q, and the set
of all complete matches for all patterns as CMp𝑘q“Ť
𝑖CM𝑃𝑖p𝑘q.
3

(A)
(AB)A
B
DP2:(ABEF)
()
(A)
(AB)A
B
E()P1:(ABCD)
(b)P3:(ABEG)
(A)
(AB)A
B
E()
(A)
(ABC) (ABE)
(ABEF) (ABEG) (ABCD)A
C E
D F G()P1P2 + +P3
Pattern Sharing
(a)(ABCD) (ABEG)(ABC) (ABE) (ABE)C
F
(ABEF)G(AB)BFig. 2: The execution plan DAGs of (a) single patterns and (b) multiple
patterns with shared states. A,B,C,D,E,F,G represent constrains on
data attributes, while the predicates are omitted for simplicity.
Execution Plan. The execution plan of a pattern 𝑃𝑖is represented
as a Directed Acyclic Graph (DAG), 𝐺𝑖, capturing the transitions
between intermediate states during pattern matching. Each node of
𝐺𝑖corresponds to a sub-pattern 𝑃𝑖,𝑗of𝑃𝑖, and each edge p𝑃𝑖,𝑗,𝑃𝑖,𝑗1q
represents a transition between sub-patterns. A sub-pattern 𝑃𝑖,𝑗
serves as an intermediate state in the pattern matching process and
defines a subsequence of data elements that partially match 𝑃𝑖i.e.,
satisfying a subset of the predicate constraints in 𝐶𝑖.
Based on the execution plan, the evaluation of pattern 𝑃𝑖is
incremental and stateful: A sequence 𝑆is processed element by
element, and a set of partial matches (PM) is maintained at each state
𝑃𝑖,𝑗. If a data element 𝑑𝑘`1is to be processed, a pattern matching
engine checks whether it can extend any of the PMs at state 𝑃𝑖,𝑗to
a new state𝑃𝑖,𝑗1. If so, the partial match is updated to the new state.
Example. Fig.2(a) illustrates the execution plan of three patterns,
SEQ(ABCD) ,SEQ(ABEF) and SEQ(ABEG) , where𝐴,...,𝐺 represent dif-
ferent constraints on data attributes, and SEQ(¨)defines the required
order of data elements. An empty node is used as the root of the
graph, representing the initial state, while nodes with zero out-
degree represent states corresponding to complete matches.
Pattern Sharing. Pattern sharing allows multiple patterns to reuse
overlapping sub-patterns, enabling the PMs belonging to these
shared states to be detected once and reused across all relevant
patterns. As showed in Fig.2(a),𝑃1,𝑃2and𝑃3share common sub-
patterns SEQ(A) and SEQ(AB) , and𝑃2and𝑃3also share another sub-
pattern SEQ(ABE) . By sharing these sub-patterns, a pattern matching
engine reduces the number of maintained PMs’ states from 12 to 7.
Pattern-sharing plan is achieved by merging bespoke pattern
execution plans, as illustrated in Fig.2. Formally, given sPas the set
of all sub-patterns of patterns in Punder a sharing mechanism, the
execution plan of shared multiple patterns is represented as a DAG
𝐺, withsPas sub-graphs and the transitions between them as edges.
A pattern matching engine must maintain the execution plan 𝐺
and continuously update the states of PMs to ensure that complete
matches are detected within each pattern’s latency constraint. We
write PMp𝑘q=tx𝑑1,...,𝑑𝑗y,...,x𝑑1
1,...,𝑑1
𝑙yufor the set of PMs after
evaluating all patterns in Pover𝑆p...𝑘q. Then, the evaluation for
multiple patterns with shared states is modeled as a function that
takes a new data element 𝑑𝑘`1, the current set of PMs, PMp𝑘q, and
the execution plan 𝐺as input, and outputs the updated PM set
PMp𝑘`1qand the complete matches CMp𝑘`1q, while ensuring thatthe detection latency for all patterns remains within their bounds:
𝑓p𝑑𝑘`1,PMp𝑘q,𝐺qÞÑt PMp𝑘`1q,CMp𝑘`1qu
s.t.@𝑃𝑖PP, 𝑙𝑖,𝑘`1ď𝐿𝑖
3.2 Problem Statement
Pattern evaluation incurs latency that is positively correlated with
the number of PMs. When processing the new data element 𝑑𝑘`1,
the pattern matching engine must scan all PMs and decides which
of them can transition to succeeding states by extending with 𝑑𝑘`1.
The engine then updates the PM set from PMp𝑘qtoPMp𝑘`1q. Even
with sub-pattern sharing, the pattern detection latency may still vio-
late the latency bound, due to the large number of PMs to maintain.
State reduction addresses this problem by selecting only certain
PMs for further processing, thereby reducing the computational
overhead and the detection latency. This is especially effective when
some PMs generate many other succeeding PMs, but are unlikely to
contribute to complete matches. However, selecting only some PMs
for further processing may generally result in the loss of complete
matches. Also, since a PM may be related to multiple patterns, the
loss of complete matches may vary across patterns.
Let𝑅“tCMp1q,..., CMp𝑘qube the results for all patterns over
𝑆p...𝑘qand𝑅1“tCM1p1q,..., CM1p𝑘qube the results obtained when
processing the same data sequence but with state reduction. For 1ď
𝑙ď𝑘, it holds that CMp𝑙q“Ť
𝑖CM𝑃𝑖p𝑙qandCM1p𝑙q“Ť
𝑖CM1
𝑃𝑖p𝑙q. For
monotonic patterns, it holds that CMp𝑙qĎCM1p𝑙q,1ď𝑙ď𝑘, so that
𝛿𝑃𝑖p𝑘q“ř
1ď𝑙ď𝑘|CM𝑃𝑖p𝑙qzCM1
𝑃𝑖p𝑙q|is the recall loss for pattern𝑃𝑖.
Therefore, the state reduction problem should minimize the recall
lossfor each pattern while ensuring that the detection latency of
every pattern remains within its specified bound.
Based thereon, we formulate the state reduction task as a multi-
objective optimization problem.
Problem 1. Given a prefix sequence 𝑆p...𝑘q, an execution plan
𝐺, the set of PMs PMp𝑘q, and the latency bounds 𝐿𝑖for each pattern
𝑃𝑖PP, the state reduction problem for best-effort pattern matching
is to select a subset of PMs SelectpPMp𝑘qqĂ PMp𝑘q, such that the
recall loss𝛿𝑃𝑖p𝑘qis minimized for all 𝑃𝑖PP, respecting the latency
bound for all patterns, 𝑙𝑖,𝑘`1ď𝐿𝑖.
4SHARP DESIGN
To address the problem of state reduction for best-effort pattern
matching, Sharp relies on the general architecture illustrated in
Fig.3. It first utilizes the 1Pattern-Sharing Degree (PSD) to orga-
nize the partial matches, and then adopts a 2cost model to evaluate
partial matches’ quality. That is, Sharp distinguishes their differ-
ent sharing degrees and derives a partial order of partial matches
based on the cost model. This preparation ensures that when a new
data element shall be processed and an overload is detected, Sharp
can efficiently select partial matches for further processing. Dur-
ing pattern matching, the 3overload detector triggers the partial
match selection manager when an overload situation materializes .
The 4partial match selection manager then determines how many
andwhich partial matches to select, using the cost model and the
PSD-based index. The details of the workflow, also given in Alg. 1,
are as follows:
4

P1P2P3P1
0 0 1
P3
1 0 00 0 1
1 1 1 1 1 1
1 1 0 Cluster[1 11]Cluster[001]
Cluster[1 10]P2
0 1 0
Overload  Detectororor
> L3? > L2? >L1?
1 / 0T F
1 / 0T F
1 / 0T F
1 1 0Cluster[001]PSD-based Index Pattern-Sharing Degree
Assessment(Alg. 2) PMs at
 
Cluster[1 10] PMs at
 
Cluster[1 11]PMs at
 PMs at
 A
B
C E
D F G
Execution PlanSHARP
Partial Match 
 Selection Manager 
(Alg. 4)PM 
Storage
Historical 
Data Sketch2
3
The space of pattern 
sharing schemes[011]
[110][111][101][001]
[100] [010]P2P3 P1  : A -> B -> C -> D
P2  : A -> B -> E -> F
  P3  : A -> B -> E -> G
Pattern workload1 P̅3
 P̅4 P̅2 P̅1
l3,k+1l2,k+1l1,k+1 P̅3
 P̅4
 P̅1
 P̅2 P̅0
 P̅1
 P̅2
 P̅3 P̅4P1
a2.A b2.A  e2.A  g2.Aattr2pn  cn 2  cn3attr1pn  cn 2  cn3
a2.A b2.A  e2.A  g2.A
a2.A b2.A  e2.A  g2.Aattr3pn  cn 2  cn3a2.A b2.A  e2.A  g2.ACost Model 
(Alg. 3) 4Fig. 3: The workflow of Sharp .
(1) Pattern-sharing Degree Assessment. (line 1 of Alg. 1) Sharp
pre-processes the execution plan to construct a PSD-based index,
which clusters partial matches according to their sharing relations.
Each cluster has an associated bitmap label, indicating the sharing
degree of the partial matches in the cluster. The partial matches
in each cluster are ordered based on the partial order determined
by the cost model (line 8) and incrementally updated when new
partial matches are generated. The PSD-based index then allows
Sharp to locate a group of partial matches with a specific sharing
degree in Op1qtime.
(2) Cost Model based Partial Match Quality Evaluation. (line 2
of Alg. 1) Sharp estimates the quality of each partial match using
the cost model. The latter leverages historical complete matches
to effectively estimate the quality of each partial match across all
patterns and incrementally updates and refines the model through-
out the matching process (line 8). It is lightweight and enables the
evaluation of each partial match’s quality in Op1qtime.
(3) Overload Detection. (line 5 of Alg. 1) When a new data element
𝑑𝑘`1shall be processed, an overload detector detects if there are
any overload scenarios by monitoring the latency of each pattern.
Specifically, the overload detector is triggered when the detection
latency𝑙𝑖,𝑘`1of a pattern 𝑃𝑖exceeds the predefined threshold 𝐿𝑖.
As shown in Fig. 3, 𝑃2and𝑃3are overloaded.
(4) Partial Match Selection. (line 6 of Alg. 1) After detecting the
overloaded patterns, the partial match selection manager selects
partial matches to be processed. Based on the PSD-index and the
cost model, Sharp selects a set of partial matches SelectpPMp𝑘qq
with the maximum PSD, highest quality, and low computational
overhead for further processing.
In the following sections, we first give details on the PSD (§ 4.1)
and the cost model (§ 4.2). Based thereon, we then present an
algorithm to select partial matches (§ 4.3).
4.1 Pattern-sharing Degree Assessment
Pattern-Sharing Degree (PSD) captures how different patterns share
partial matches through overlapping subpatterns. In other words,
for a certain partial match (i.e., state), PSD specifies which patterns
share it. As illustrated by the Venn diagram in Fig.3, the space of
possible pattern-sharing schemes grows exponentially with theAlgorithm 1: Sharp workflow.
Input: Input data𝑑𝑘`1, data sequence prefix 𝑆p...𝑘q, execution
plan𝐺, set of patterns P, set of complete matches CMp𝑘qand
partial matches PMp𝑘qover𝑆p...𝑘q.
Output: A set of partial matches SelectpPMp𝑘qqto be processed.
// Pattern-sharing Degree Assessment
1C“PSD_Assessment p𝐺,Pq; // Alg.2, §4.1
// Estimate partial matches’ quality
2Q“Quality_Estpmeta,CMp𝑘q,P,PMp𝑘qq; // Alg.3, §4.2
// Partial ordering partial matches in clusters
3C.Partial_Order pQq;
// partial match selection when 𝑑𝑘`1arrives
4while𝑑𝑘`1arrives do
// 3. Overload detection
5 if@𝑃𝑖PP,D𝑙𝑖,𝑘`1ě𝐿𝑖then
// 4. PM Selection. Alg.4, §4.3.
6 SelectpPMp𝑘qq“ PM_SelectionpC,t𝑃𝑖u𝑙𝑖,𝑘`1ě𝐿𝑖q;
// Incrementally update the clusters and the cost model.
7 while a new partial or complete match 𝜙arrives do
8 C.insertp𝜙q,Q.updatep𝜙q;
9return SelectpPMp𝑘qq;
number of patterns. We design PSD to encode these schemes sys-
tematically, capturing the full spectrum of pattern-sharing combina-
tions. Note that for a given execution plan, the PSD is deterministic.
We maintain the PSD to (i) differentiate how each state is shared
across different patterns, (ii) efficiently retrieve partial matches with
specific sharing schemes from the entire PM set, and (iii) instantly
select the partial match with the highest contribution and lowest
computational overhead. Alg. 2 presents the details.
(1) PSD Assessment. (line 2 - 4 of Alg. 2) We adopt bitmap la-
bels to represent the sharing degree of each subpattern, leveraging
the efficiency of bitwise operations. We assess the PSD for each
subpattern by performing a depth-first traversal of the execution
plan DAG𝐺. The bitmap label for each subpattern ¯𝑃is defined as
an𝑛-dimensional bitmap 𝑏p¯𝑃q, where the 𝑖-th bit is set to 1 if ¯𝑃is
an intermediate state of pattern 𝑃𝑖, and 0 otherwise. Likewise, for
a partial match 𝜌at state ¯𝑃, we assign the same bitmap label, i.e.,
𝑏p𝜌q“𝑏p¯𝑃q.
5

Algorithm 2: Pattern-Sharing Degree Assessment
Input: Evaluation Plan 𝐺, the set of patterns P, contribution
Output: Set of clusters C
1CÐH ;
2for𝜋𝑖P𝛱do𝑏p𝑃𝑖qÐ 2𝑖´1;//@𝑃𝑖PP, set the𝑖´th bit to 1
// PSD Assessment through Depth-First Traversal of 𝐺
3for𝑃𝑖PPand there is a reachable state ¯𝑃in𝐺do
4𝑏p¯𝑃qÐ𝑏p¯𝑃q_𝑏p𝑃𝑖q// Update the PSD of ¯𝑃
5CÐClustering sub-patterns with the same PSD ;
6@𝐶p𝑏1qPC,𝐶p𝑏1q“t𝜌|𝑏p𝜌q“𝑏1u; // PSD-based Indexing
7return C;
We access the PSD of each subpattern ¯𝑃by traversing the exe-
cution plan 𝐺in a depth-first manner. The intuitive idea of PSD
assessment is that, for each 𝑃𝑖PP, if there is reachable state ¯𝑃in
the execution plan 𝐺, then ¯𝑃is shared by 𝑃𝑖. Initially, each pattern
𝑃𝑖is assigned a unique bitmap label 𝑏p𝑃𝑖q“2𝑖´1, where only the
𝑖-th bit is set to 1. The algorithm then traverses 𝐺from sink nodes
to source nodes and computes the bitmap label for each subpat-
tern ¯𝑃using the rule: 𝑏p¯𝑃q“Ž
¯𝑃1P¯𝑃.succ𝑏p¯𝑃1q,i.e., by applying a
bitwise OR across the bitmap labels of its successor sub-patterns.
As illustrated in Fig. 3, for example, the bitmap of ¯𝑃4is derived as
𝑏p𝑃2q_𝑏p𝑃3q.
(2) Clustering and Indexing. (line 5 - 6 of Alg. 2) After the
PSD assessment, we group partial matches of subpatterns with
the same bitmap label 𝑏1into a cluster 𝐶p𝑏1q. Each cluster can be
represented as a set of partial matches 𝐶p𝑏1q “ t𝜌|𝑏p𝜌q “𝑏1u.
In Fig. 3, partial matches belonging to ¯𝑃1,¯𝑃2are grouped into the
same cluster 𝐶pr111sq. When a new PM at state 𝑃1arrives, it will
be indexed to cluster 𝐶p𝑏1“𝑏p𝑃1qq.
PSD-based index is effective. It can can locate the partial matches
associated with the a given bitmap label 𝑏1, i.e.,𝐶p𝑏1q, within Op1q
time complexity.
4.2 Cost Model
The cost model has two requirements (i) access the quality of partial
matches and (ii) lightweight , i.e., low computational overhead. Below,
we first describe how to assess the quality of partial matches (§4.2.1),
followed by its efficient estimation (§4.2.2).
4.2.1 Partial match quality assessment. To capture the quality of a
partial match, we assess its impact on pattern matching in terms of
the contribution to the final results and the incurred computational
overhead.
Contribution to Complete Matches. In shared state, a partial
match𝜌=x𝑑1,...,𝑑𝑗ymay result in complete matches of multiple
shared patterns. We capture 𝜌’s contribution to pattern 𝑃𝑖as the
number of complete matches that are generated by 𝜌. Formally, we
define the contribution of 𝜌up to a future time point 𝑘1
Δ`
𝑃𝑖p𝜌q“|tx𝑑1
1,...,𝑑1
𝑙yPCM𝑃𝑖p𝑘1q|@1ď𝑗ď𝑙:𝑑1
𝑗“𝑑𝑗u|,(1)
CM𝑃𝑖p𝑘1qis the set of complete matches of pattern 𝑃𝑖at𝑆p...𝑘1q.
Computational Overhead. We consider a partial match’s com-
putational overhead as the resource consumption caused by the
partial match itself and all its derived partial matches in the futureAlgorithm 3: Cost Model-based PM Quality Evaluation.
Input: The set of𝑛patterns P, the set of complete matches CMp𝑘q
and partial matches PMp𝑘qover𝑆p...𝑘q.
Output: The contribution Δ`
𝑃𝑖p𝜌qand the computational overhead
Δ´
𝑃𝑖p𝜌qof each𝜌PPMp𝑘qto all patterns 𝑃𝑖PP.
// Build Historical Data Sketch
1for𝜙“x𝑑1,...,𝑑 𝑙yPCMp𝑘qdo
2@1ď𝑗ď𝑙´1,𝑎𝑡𝑡𝑟 𝑗“hashpx𝑑1.𝐴,...,𝑑 𝑗.𝐴yq;
3@1ď𝑗ď𝑙´1,Tupler𝑎𝑡𝑡𝑟 𝑗s.𝑐𝑛𝑖++ ;
// Incremental Update of the Historical Data Sketch
4fora newly generated match 𝜙1“x𝑑1
1,...,𝑑1
𝑙ydo
5 if𝜙1is a complete match of 𝑃𝑖then
// Update the complete match counting
6@1ď𝑗ď𝑙´1,𝑎𝑡𝑡𝑟 𝑗“hashpx𝑑1
1.𝐴,...,𝑑1
𝑗.𝐴yq;
7@1ď𝑗ď𝑙´1,Tupler𝑎𝑡𝑡𝑟 𝑗s.𝑐𝑛𝑖++ ;
8 if𝜙1is a partial match of 𝑃𝑖then
// Update the partial match counting
9𝑎𝑡𝑡𝑟“hashpx𝑑1
1.𝐴,...,𝑑1
𝑙.𝐴yq;
10 Tupler𝑎𝑡𝑡𝑟 𝑗s.𝑝𝑛++;
// Evaluating contribution and computational overhead
11while evaluating𝜌“x𝑑2
1,...,𝑑2
𝑙yPPMp𝑘qdo
12𝑎𝑡𝑡𝑟p𝜌q“hashpx𝑑2
1.𝐴,...,𝑑2
𝑙.𝐴yq;// Get the attribute key
13@𝑃𝑖PP,Δ`
𝑃𝑖p𝜌q“Tupler𝑎𝑡𝑡𝑟p𝜌qs.𝑐𝑛𝑖; // Contribution
14@𝑃𝑖PP,Δ´
𝑃𝑖p𝜌q“Tupler𝑎𝑡𝑡𝑟p𝜌qs.𝑝𝑛ˆ𝛩p𝜌q;// Overhead
// Define the partial order of partial matches
15while Comparing the partial order of 𝜌and𝜌1do
16 if@𝑃𝑖PP,Δ`
𝑃𝑖p𝜌qąΔ`
𝑃𝑖p𝜌1qthen
17 return𝜌ą𝜌1;
18 else
19 ifř
𝑖Δ`
𝑃𝑖p𝜌qąř
𝑖Δ`
𝑃𝑖p𝜌1qthen return 𝜌ą𝜌1;
20 else return 𝜌ď𝜌1;
21returnpΔ`
𝑃𝑖p𝜌q,Δ´
𝑃𝑖p𝜌qq𝑃𝑖PP, 𝜌PPMp𝑘q;
(i.e., all reachable states s𝑃in the execution plan 𝐺). In addition,
partial matches in different states consume different computational
resources (i.e., CPU cycles and memory footprint) due to the com-
plexity of the predicates and the size of the partial match itself (i.e.,
the length of a sub-pattern). We capture this through a function
𝛩p𝜌q ÞÑ𝑟,𝑟PN`that maps𝜌’s computational overhead as a
real number. We later explain how to materialize 𝛩for different
applications in §5.
We now define the computational overhead of 𝜌to pattern𝑃𝑖as
Δ´
𝑃𝑖p𝜌q“ÿ
𝜌1PŤ
𝑘1ą𝑘PM𝑃𝑖p𝑘1qIp𝜌,𝜌1q¨𝛩p𝜌1q, (2)
PM𝑃𝑖p𝑘1qis the set of PMs of pattern 𝑃𝑖over𝑆p...𝑘1q{𝑆p...𝑘q, and
Ip𝜌,𝜌1qis an indicator function that returns 1 if 𝜌is generates 𝜌1.
4.2.2 Efficient estimation. The cost model, i.e., Δ`andΔ´, re-
quires the statistics (e.g., number of generated partial matches) in
future, which can only be captured in retrospect. In practice, we
design an efficient estimation for the cost model by monitoring his-
torical statistics of pattern detection, for instance, the aggregated
statistics (e.g., the number and CPU cycles) of PMs maintained in
current and previous time windows.
6

Historical Data Sketch. Sharp maintains a historical data sketch
that efficiently monitors and estimates the contribution and com-
putational overhead of partial matches. In particular, for each com-
plete match 𝜙“x𝑑1,...,𝑑𝑙y, we hash the attribute values of all its
prefixes to a series of unique attribute keys ( line 2 in Alg.3 ),
𝑎𝑡𝑡𝑟 1“hashpx𝑑1.𝐴yq,...,𝑎𝑡𝑡𝑟𝑙´1“hashpx𝑑1.𝐴,...,𝑑𝑙´1.𝐴yq.
Such hash values are used to efficiently lookup the partial match
instances can generate 𝜙, if they share a common attribute key
𝑎𝑡𝑡𝑟𝑗“hashpx𝑑1.𝐴,...,𝑑𝑙´1.𝐴yq, i.e., the PM instances in reach-
able states of the execution plan 𝐺.
Based thereon, we use attribute keys to group complete matches
and partial matches into buckets. By aggregating the number of
complete matches sharing the same attribute key, Sharp estimates
how many complete matches have been generated by partial matches
with that attribute (line 3 in Alg. 3). Each entry is stored as a vector
r𝑎𝑡𝑡𝑟,𝑐𝑛 1,...,𝑐𝑛𝑛,𝑝𝑛s, where𝑎𝑡𝑡𝑟 is the attribute key, 𝑝𝑛is the
number of partial matches generated, and 𝑐𝑛𝑖denotes the num-
ber of complete matches for pattern 𝑃𝑖. We use𝑐𝑛1´𝑛and𝑝𝑛to
estimate the contribution and computational overhead.
When estimating the contribution and computational overhead
of a partial match 𝜌“ x𝑑2
1,...,𝑑2
𝑗y(Alg. 3, line 11 - 14), Sharp
first hashes its attribute key ,𝑎𝑡𝑡𝑟p𝜌q “ hashpx𝑑2
1.𝐴,...,𝑑2
𝑗.𝐴qy.
Then , Sharp looks up the historical data sketch to locate the vectors
r𝑎𝑡𝑡𝑟p𝜌q,𝑐𝑛1´𝑛,𝑝𝑛swith the same attribute key. The contribution
of𝜌to pattern𝑃𝑖is estimated as Δ`
𝑃𝑖“𝑐𝑛𝑖and the computational
overhead is Δ´
𝑃𝑖“𝑝𝑛ˆ𝛩p𝜌q.
The design of historical data sketch enables the estimation and
the look-up operation in Op1qtime complexity.
Incremental Update. For efficiency, Sharp maintains the histori-
cal data sketch by incrementally updating it with new partial and
complete matches. When a new complete match of pattern 𝑃𝑖ar-
rives (Alg. 3, line 5-7), its all prefixes are first hashed to a series
of attribute keys. Then, all corresponding tuples with the same
attribute keys are updated by 𝑐𝑛𝑖“𝑐𝑛𝑖`1. If a new partial match
𝜌is generated (Alg. 3, line 8-10), we hash 𝜌to the attribute key and
increment the number of partial matches by 𝑝𝑛“𝑝𝑛`1.
4.2.3 Partial Ordering partial matches. Ordering partial matches
is essential for the selection process. For instance, to accelerate
partial match selection within each cluster, the partial matches are
organized in a max-heap structure—constructed during the PSD
assessment step—based on their priority (line 3 of Alg. 1). Since each
partial match may contribute to multiple patterns, they naturally
exhibit partial ordering relationships. Based on the cost model, we
can derive these partial orders.
We compare the partial order of each pair of partial matches 𝜌
and𝜌1based on their contributions to the patterns. The comparison
is performed in two steps (Alg. 3, line 15 - 20). First, we compare the
contribution of 𝜌and𝜌1on each pattern 𝑃𝑖. If@𝑃𝑖PP,Δ`
𝑃𝑖p𝜌qą
Δ`
𝑃𝑖p𝜌1q, then𝜌is better than 𝜌1. Otherwise, we compare the total
contribution of both PMs. Ifř
𝑖Δ`
𝑃𝑖p𝜌q ąř
𝑖Δ`
𝑃𝑖p𝜌1q, then𝜌is
better than𝜌1. Otherwise, 𝜌is worse than 𝜌1.
Maintaining the partial order of partial matches introduces no
additional overhead to the system, as organizing partial matches is
inherently part of the pattern matching process. Moreover, evenupdating the max-heap inside each cluster is efficient, with a sub-
linear time complexity, which is negligible compared to the overall
cost of PM organization in the pattern matching engine.
4.3 Partial Match Selection Manager
The partial match selection manager is responsible for selecting a
subset of partial matches to process when the overload detector is
triggered. As introduced in §4, when overload happens, the over-
load detector will generate an overload label 𝑏𝑂𝐿with the𝑖-th bit
set to 1, indicating that the 𝑖-th pattern is overloaded. The partial
match selection manager will take 𝑏𝑂𝐿as input, then (1) determine
how many partial matches should be selected based on the overload
scenario, and (2) select partial matches based on the PSD-based
index and the cost model. We formulate the partial match selec-
tion problem as a multi-objective optimization problem , and (3) an
efficient greedy algorithm is proposed to solve it.
How many partial matches to select? The partial match selection
manager determines how many partial matches to select based on
the severity of the overload. Since pattern detection latency is
positively correlated with the computational overhead of partial
matches, the manager selects partial matches whose total overhead
is proportional to the required latency reduction.
When a pattern 𝑃𝑖with partial matches PM𝑃𝑖p𝑘qbecomes over-
loaded upon the arrival of a new data element 𝑑𝑘`1, the selection
manager chooses a subset of partial matches whose total computa-
tional overhead does not exceed that of the current partial matches.
Otherwise, if 𝑃𝑖is not overloaded, all of the partial matches should
be selected. Specifically, for all pattern 𝑃𝑖PP, the selected set of
partial matches SelectpPMp𝑘qqshould satisfy the following con-
straint:
ÿ
𝜌PSelectpPMp𝑘qq𝑏𝑖p𝜌qΔ´
𝑃𝑖p𝜌q ď𝐿𝑖
𝑙𝑖,𝑘`1ÿ
𝜌PPM𝑃𝑖p𝑘qΔ´
𝑃𝑖p𝜌q(3)
where𝑏𝑖p𝜌qis the𝑖-th bit of the bitmap label 𝑏p𝜌qfor PM𝜌and
𝑏𝑖p𝜌q “ 1means𝜌is shared by 𝑃𝑖. The above constraint also
ensures that if 𝑃𝑖is not overloaded, then all of its partial matches
can be selected.
Which partial matches to select? Recall that the state reduction
problem aims to maximize the results quality when selecting partial
matches while stratifying the latency bound of each pattern. The
partial match selection manager will approximately solve this prob-
lem by maximising the overall contribution of the selected partial
matches on every pattern while satisfying the consumption con-
straints in Eq. (3). Given a sequence prefix 𝑆p...𝑘q, we formulate
“which partial matches to select? ” as the following multi-objective
optimization problem .
max ®𝚫`
PpSelectpPMp𝑘qqq“p𝛥`
𝑃1,...,𝛥`
𝑃𝑛q (4)
s.t.ÿ
𝜌PSelectpPMp𝑘qq𝑏𝑖p𝜌qΔ´
𝑃𝑖p𝜌qď𝐿𝑖
𝑙𝑖,𝑘`1ÿ
𝜌PPM𝑃𝑖p𝑘qΔ´
𝑃𝑖p𝜌q,@𝑃𝑖PP
𝛥`
𝑃𝑖“ř
𝜌PSelectpPMp𝑘qqΔ`
𝑃𝑖p𝜌qis the total contribution of the se-
lected partial matches to pattern 𝑃𝑖, and the objective function
®𝚫`
PpSelectpPMp𝑘qqqis a vector representing contributions to all
7

Algorithm 4: State Selection Algorithm
Input: Overload label 𝑏𝑂𝐿, PSD-based index C
Output: A set of partial matches SelectpPMp𝑘qqto be processed.
// Select all partial matches related to non-overloaded
patterns
1SelectpPMp𝑘qqÐŤ
𝑏1:𝑏1&𝑏𝑂𝐿“0𝐶p𝑏1q;
// Select partial matches related to overloaded patterns
2while Eq. (3) is holding do
// Select the highest quality PM among the clusters
3𝜌“max𝑏1:𝑏1&𝑏𝑂𝐿‰0t𝐶p𝑏1q.𝑝𝑜𝑝pqu;
4 SelectpPMp𝑘qqÐ SelectpPMp𝑘qqYt𝜌u;
5return SelectpPMp𝑘qq;
patterns in P. These patterns may have different or event contra-
dicting utility. Therefore the optimization goal of each bespoke
pattern may be different, i.e., multi-objective optimization.
Solving this problem is challenging. It is NP-hard as it can be
reduced from the multi-dimensional multi-objective knapsack prob-
lem [ 13,27]. Also, the multi-objective nature of this problem signifi-
cantly increases its complexity. Consequently, solving this problem
can be time-consuming. To reduce the latency overhead caused
by the selection manager, we propose a lightweight selection algo-
rithm, as outlined in Alg. 4.
Greedy State Selection. The state selection algorithm Alg. 4 se-
lects (i) all partial matches related to the non-overloaded patterns,
and (ii) partial matches related to the overloaded patterns greedily.
First (line 1), the PSD-based index can efficiently locate the partial
matches associated with the non-overloaded patterns by identify-
ing clusters whose bitmap labels satisfy 𝑏1:𝑏1&𝑏𝑜“0. Thus,
SelectpPMp𝑘qqcan be initialized byŤ
𝑏1:𝑏1&𝑏𝑂𝐿“0𝐶p𝑏1q.
Second (line 2 - 4), the algorithm also uses the PSD-based in-
dex to locate clusters of partial matches associated with the over-
loaded patterns, i.e., 𝐶p𝑏2qsuch that𝑏2&𝑏𝑂𝐿‰0. It then builds
SelectpPMp𝑘qqby iteratively selecting partial matches from the
clusters with the highest partial order, under the PM count con-
straint defined in Eq. (3). Since partial matches within each cluster
are maintained in a max-heap and the number of clusters is constant,
this process can be completed in Op1qtime.
The partial match selection algorithm is lightweight. In the first
step, the PSD-based index enables the partial matches’ retrieval in
Op1qtime. In the second step, partial match selection only involves
Op1qlook-up complexity to each PM. Since the PM processing is
inherently part of the pattern matching engine, the overhead of
look-up complexity is negligible.
5 OPTIMIZATION AND IMPLEMENTATION
This section discusses the optimization techniques tailored to dif-
ferent application domains. We show how the system adapts its
cost model to specific data processing paradigms—such as CEP,
MATCH_RECOGNIZE , and GraphRAG—by leveraging specific domain
knowledge. This includes (1) optimizing resource cost estimation
for partial matches, (2) evaluating the quality of partial matches in
GraphRAG where historical data is useful and (3) introducing an
other partial order measurement.Resource Cost Estimation for Partial Matches. Under different
data processing diagrams, the resource cost of a PM is estimated dif-
ferently. In CEP and MATCH_RECOGNIZE , the number and complexity
of predicates that need to be evaluated for a PM may dramatically
differ. On the other hand, in GraphRAG, the predicates of the path
queries are much simpler—comparing the similarities of strings.
(1) CEP and MATCH_RECOGNIZE .In CEP and MATCH_RECOGNIZE , the
resource cost of a PM is estimated by jointly considering the number
of predicates that need to be evaluated and the memory footprint
of the PM. Specifically, the resource cost function 𝛩p𝜌qfor a PM𝜌
is defined as:
𝛩𝐶𝐸𝑃 &𝑀𝑅𝑒𝑔p𝜌q“𝜐¨sizeofp𝜌q`p 1´𝜐q¨|pred_nump𝜌q|,
where sizeofp𝜌qestimates the memory footprint, |pred_nump𝜌q|
denotes the number of predicates that need to be evaluated, and
𝜐P r0,1sis an empirical factor balancing the relative impact of
memory usage and runtime predicate evaluation cost.
(2) GraphRAG. In GraphRAG, due to the simple nature of the
predicates, we only use the memory footprint to estimate the re-
source cost of a PM. The resource cost function is defined as:
𝛩𝐺𝑟𝑎𝑝ℎ𝑅𝐴𝐺p𝜌q“sizeofp𝜌q. (5)
(2) Partial Match Quality Evaluation for GraphRAG. Historical
data is not useful in GraphRAG, as the queries generated by the
LLM are non-deterministic, non-reusable queries. This is due to
the inherent randomness of LLMs— the same prompt can produce
entirely different path queries. Therefore, we use the KG’s meta-
data instead of the historic data to abstract the contribution and
computation overhead of partial matches.
We adopt the “average number of neighbors for each node”
(avg_num_nbr ) in the metadata. For a partial match 𝜌that corre-
sponds to the first 𝑙hops of a𝑚-hop path query 𝑃𝑖, its contribution
is estimated by
Δ`
𝑃𝑖“cos_simp𝜌,𝑃𝑖q
and its computational overhead is estimated by
Δ´
𝑃𝑖“𝛩𝐺𝑟𝑎𝑝ℎ𝑅𝐴𝐺p𝜌q¨num_nbrp𝜌q¨avg_num_nbr𝑚´𝑙.
cos_simp𝜌,𝑃𝑖qdenotes the cosine similarity between the partial
match𝜌and the path query 𝑃𝑖, serving as an estimate of 𝜌’s con-
tribution to forming a complete match. num_nbrp𝜌qis the number
of neighbors of the last visited node in 𝜌, which can be retrieved
during pattern matching. The term num_nbrp𝜌q¨avg_num_nbr𝑚´𝑙
estimates the number of partial matches that could be generated
from𝜌, i.e., the expected number of edges to be traversed. Finally,
𝛩GraphRAGp𝜌qis the estimated resource cost of 𝜌, derived from Eq. 5.
6 EVALUATION
We evaluate the effectiveness and efficiency of Sharp in various
scenarios. After outlining the experimental setup in §6.1, our ex-
perimental evaluation answers the following questions:
(1) What are the overall effectiveness and efficiency of Sharp ? (§6.2)
(2) How sensitive is Sharp to pattern properties including pattern
selectivity, pattern length, and the time window? (§6.3)
(3) How do pattern sharing mechanisms impact Sharp ? (§6.4)
(4) How does Sharp adapt to concept drifts of input data? (§6.5)
(5) How do resource constraints impact Sharp ? (§6.6)
8

6.1 Experimental Setup
Our experiments have the following setup:
Testbeds. We conduct experiments on two clusters. (i) A NUMA
node with two AMD EPYC 9684X CPUs (192 cores) and 1.5 TBRAM,
running Ubuntu 22.04. (ii) A GPU cluster–each node being equipped
with four NVIDIA H100 80GB GPUs, two Intel Xeon Platinum 8468
CPUs (96 cores), and 1.5 TBRAM, running Red Hat Enterprise 9.5.
Baselines. We compare Sharp to five baselines: (i) Random input
(RI) selects input data randomly to process. (ii) Random state (RS)
selects partial matches randomly to process. (iii) DARLING [9]
selects input data based on the queue buffer size and input data
utility for CEP patterns. We have extended DARLING to support
shared patterns. (iv) ICDE’20 [58] selects the combination of both
input data and partial matches to process based on its cost model.
We have adapted it to support shared patterns. (v) Selective state
(SS) selects partial matches to process based on the (estimated)
selectivity of the pattern predicates.
Datasets. We have evaluated Sharp and baselines using multiple
synthetic and real-world datasets:
(1) Synthetic Datasets. We prepare two synthetic datasets. DS1
contains tuples consisting of five uniformly-distributed attributes:
a categorical type ( Upt𝐴,𝐵,𝐶,𝐷,𝐸,𝐹,𝐺,𝐻,𝐼,𝐽 uq), a numeric ID
(Up1,10q), and numeric attributes 𝑋(Up´90,90q),𝑌(Up´180,180q),
and𝑉(Up1,3ˆ106q). DS2 has similar settings, i.e., a categorical
type ( Upt𝐴,𝐵,𝐶,𝐷,𝐸,𝐹uq), a numeric ID ( Up1,25q), and one nu-
meric attribute 𝑋(Up1,100q).
(2)Citi_Bike [1] is a publicly available dataset of bike trips pro-
vided by the bike-sharing company, Citi Bike, in New York City.
(3)Crimes [33] is a public crime record dataset from Chicago. We
use it for MATCH_RECOGNIZE patterns.
(4)KG-Meta-QA [56] is a knowledge graph that captures structured
information of movies. We use it for path queries in GraphRAG.
Patterns. Tab. 2 shows the patterns for our experiments. Here,
Q1´6are pattern templates evaluated over synthetic and real-world
data by materializing the schema (e.g., materializing AinQ3with
bike_trip ). They cover three representative pattern sharing schemes.
Q1and Q2share a Kleene closure sub-pattern SEQ(A,B`).Q3and Q4
share the sub-pattern SEQ(A,B,C,D) with computationally expensive
predicates. Q5and Q6share a negation pattern SEQ(A,B,␣C). For the
GraphRAG experiments, we use 14,872 patterns from Meta-QA [ 56].
Metrics. We evaluate the performance of Sharp and the baselines
enforcing strict latency bounds. Such a latency bound is defined as a
percentage of the latency observed without state reduction. We mea-
sure the result quality and the throughput performance. The result
quality is assessed in recall – the ratio of complete matches obtained
with state reduction to all complete matches derived without state
reduction. For GraphRAG experiments, we measure the recall of
the end-to-end results i.e., the correct responses from the LLM com-
pared to the ground truth. In addition, we also measure its accuracy
– the ratio of correct answers to all LLM generated responses. How-
ever, we omit the accuracy of CEP and MATCH_RECOGNIZE patterns,
because they always output correct results, i.e., complete matches.
For the throughput performance, we report events or tuples per sec-
ondfor CEP and MATCH_RECOGNIZE patterns, and the speedup factor
for GraphRAG.Tab. 2: Patterns for the experiments.
Q1SEQ(A a, B+ b[], C c, D d)
WHERE SAME [ID] AND SUM pb[i].xqăc.x
Q2SEQ(A a, B+ b[], E e, F f)
WHERE SAME [ID] AND a.x + SUM(b[i].x) ăe.x + f.x
Q3SEQ(A a, B b, C c, D d, E e, F f, G g)
WHERE SAME [ID] AND a.v < b.v AND b.v + c.v < d.v
AND 2r¨arcsin´
sin2´
e.x - d.x
2¯
`cos(d.x)cos(e.x)sin2´e.y - d.y
2¯¯ 1
2ďf.v
Q4SEQ(A a, B b, C c, D d, H h, I i, J j)
WHERE SAME [ID] AND a.v ăb.v AND b.v + c.v ăd.v
AND𝑟¨arccospsin(d.x)sin(h.x) + cos(d.x)cos(h.x)cos(h.y - d.y) qďi.v
Q5SEQ(A a, B b, NEG(C) c, D d)
WHERE SAME [ID] AND a.x ăb.x
Q6SEQ(A a, B b, NEG(C) c, E e)
WHERE SAME [ID]
Qm14,872 queries in the Meta-QA benchmark [56]
6.2 Overall Effectiveness and Efficiency
We first investigate the overall performance (i.e., recall and through-
put) of Sharp in CEP, MATCH_RECOGNIZE and GraphRAG scenarios
using synthetic and real-world datasets.
We execute the shared patterns of Q3and Q4in the CEP engine
over DS1.Fig.4 demonstrate the results. The latency without state
reduction is 180 ms. We set the latency bound ranging from 10%to
90%of the original latency. At all latency bounds, Sharp achieves
the highest recall value among all baselines (see Fig.4a).Sharp
maintains the recall above 95% under 50%-90% latency bounds. The
margin becomes larger for tighter latency bounds. At the latency
bound of 10%, Sharp still achieves the recall of 70%, which is 1.96 ˆ,
4.10ˆ, 5.30ˆ, and 11.25ˆhigher than ICDE’20 ,DARLING ,RSand RI.
The similar trend is observed in real-world datasets of Citi_Bike .
As showed in Fig.5a,Sharp outperforms all baselines, improves
the recall by 3.5ˆ(ICDE’20 ),3.7ˆ(DARLING ),2.8ˆ(RS) and 7ˆ(RI).
We attribute Sharp ’s high recall to the combination of its pattern-
sharing degree (§4.1) mechanism and the cost model (§4.2), which
considers both pattern-sharing schemes and the cost of state, i.e.,
partial matches. ICDE’20 and DARLING do not consider the pattern-
sharing, which yields lower recall. Also, Sharp leads to higher call
than random approaches, RSand RI.
We then examine the throughput performance in Fig.4b and
Fig.5b.Sharp ’s throughput is higher than ICDE’20 (1.24ˆ) and
significantly higher than DARLING (8.73ˆ) but lower than the random
approaches. Clearly, the high throughput of RSand RIcomes at the
expense of poor recall, which is lower than 20% (see Fig.4a and
Fig.5a). Despite their efficiency, this renders random approaches
unreliable in practice. Sharp , in turn, strikes a better trade-off of
recall and efficiency.
Sharp ’s superior performance stems from two factors: (i) the
PSD facilitates to look up costs only at the affected shared states,
significantly reducing the search space compared to ICDE’20 and
DARLING . (ii) The efficient implementation (e.g., bitmap indexing and
updating) with 𝑂p1qpartial match selection overhead.
Similar trends are observed for MATCH_RECOGNIZE patterns and
the GraphRAG application. Fig.6 illustrates the results of executing
MATCH_RECOGNIZE patterns, the shared Q5and Q6over the Crimes
dataset [ 33]. Here, Sharp outperforms all baselines in recall ( Fig.6a).
At the 10% latency bound, Sharp achieves the recall of 74% that
is1.2ˆ,1.85ˆ,3.7ˆ, and 2ˆhigher than ICDE’20 ,DARLING ,REand
9

10 30 50 70 90
Latency Bound (%)020406080100Recall (%)SHARP
ICDE'20DARLING
RIRS(a)Recall
10 30 50 70 90
Latency Bound (%)0.0×1060.5×1061.0×1061.5×1062.0×106Throughput (events/s)
SHARP
ICDE'20DARLING
RIRS (b)Throughput
Fig. 4: CEP experiments under different latency bounds (original
latency 180 ms) on synthetic datasets.
10 30 50 70 90
Latency Bound (%)020406080100Recall (%)SHARP
ICDE'20DARLING
RIRS(a)Recall
10 30 50 70 90
Latency Bound (%)0.0×1061.0×1062.0×1063.0×106Throughput (events/s)
SHARP
ICDE'20DARLING
RIRS (b)Throughput
Fig. 5: CEP experiments under different latency bounds (original
latency 500 ms) on the real-world Citi_Bike dataset.
10 30 50 70 90
Latency Bound (%)020406080100Recall (%)SHARP
ICDE'20DARLING
RIRS
(a)Recall
10 30 50 70 90
Latency Bound (%)0.5×1061.0×1061.5×1062.0×106Throughput (tuples/s)
SHARP
ICDE'20DARLING
RIRS (b)Throughput
Fig. 6: MATCH_RECOGNIZE patterns under different latency
bounds (original latency 50 ms) on the Crimes dataset.
10 30 50 70 90
Latency Bound (%)0255075100Accuracy (%)SHARP RS SS(a)Accuracy
10 30 50 70 90
Latency Bound (%)0255075100Recall (%)SHARP RS SS (b)Recall
10 30 50 70 90
Latency Bound (%)1.2×1.4×1.6×1.8×Throughput Speed-up
SHARP RS SS (c)Throughput Improvement
Fig. 7: GraphRAG experiments under different latency bounds (original average
latency 400 ms) on the Meta-QA benchmark.
10 30 50 70 90
PM Selection Ratio (%)020406080100Recall (%)SHARP ICDE'20 RS
Fig. 8: Impact of the selection ratio of partial matches
RI.Sharp ’s throughput is again between ICDE’20 /DARLING and the
random approaches (Fig. 6b).
For the GraphRAG experiments, Sharp executes the processing
pipeline in Fig.1a over the Meta-QA benchmark [ 56].Fig.7 presents
the averaged results of all the 14,872 queries. Since ICDE’20 and
DARLING are not designed for graph queries, we compare Sharp with
RSand SS. Here, Sharp outperforms the baselines in both accuracy
and recall. It keeps 100% accuracy for 50%-90% latency bounds,
and achieves 70% accuracy at the tighter latency bound of 10% i.e.,
3.50ˆand 3.18ˆhigher than RSand SS. For the recall, Sharp ’s
performance margin becomes smaller at tighter latency bounds. At
10% latency bounds, the recall values are comparable with baselines
(seeFig.7b). The reason is that the recall depends on statistical ef-
ficiency of LLM generated responses – a small set of responses will
not cover the majority of ground truth. However, Sharp ’s selection
of state ensures that over 80% of the generated response indeed
align the ground truth. As for the throughput, Sharp performs
closely to RSand much higher than SS(Fig.7c). This demonstrates
Sharp ’s low overhead even in complicated processing pipelines.
To further examine the advantage of Sharp ’s state selection
decision. We control the selection ratio of states and measure the
recall of Sharp ,ICDE’20 and RS, as showed in Fig.8. We can observe
thatSharp achieves significantly higher recall than the baselines
1M 3M 5M 7M 9M
Variance Control (max(v))0255075100Average Recall (%)SHARP
ICDE'20DARLING
RIRS(a)CEP Recall
1M 3M 5M 7M 9M
Variance Control (max(v))0.0×1060.5×1061.0×1061.5×1062.0×1062.5×106Throughput (events/s)
SHARP
ICDE'20DARLING
RIRS (b)CEP Throughput
1M 3M 5M 7M 9M
Variance Control (max(v))0255075100Average Recall (%)SHARP
ICDE'20DARLING
RIRS
(c)MATCH_RECOGNIZE Recall
1M 3M 5M 7M 9M
Variance Control (max(v))0.0×1060.5×1061.0×1061.5×1062.0×106Throughput (tuples/s)
SHARP
ICDE'20DARLING
RIRS (d)MATCH_RECOGNIZE Recall
Fig. 9: Sensitivity to pattern selectivity.
(up to 30ˆofRSand 2.3ˆofICDE’20 ), demonstrating Sharp ’s
superior state selection.
6.3 Sensitivity Analysis to Pattern Properties
Next, we examine Sharp ’s sensitivity to various pattern properties,
considering selectivity, pattern length, and the time window size.
Selectivity. Pattern predicates select data and partial matches. We
control the selectivity by changing the value distribution of 𝑉in
DS1. This affects Q3and Q4in CEP and Q5and Q6inMATCH_RECOGNIZE .
In particular, we change the distribution of 𝑉from U(0, 1ˆ106) to
U(0, 1ˆ109), increasing the selectivity for Q3´6.
10

4 5 6 7 8
Pattern Length0255075100Recall (%)SHARP
ICDE'20DARLING
RIRS(a)CEP Recall
4 5 6 7 8
Pattern Length0.0×1061.0×1062.0×1063.0×106Throughput (events/s)
SHARP
ICDE'20DARLING
RIRS (b)CEP Throughput
4 5 6 7 8
Pattern Length0255075100Recall (%)SHARP
ICDE'20DARLING
RIRS
(c)MATCH_RECOGNIZE Recall
4 5 6 7 8
Pattern Length0.0×1061.0×1062.0×1063.0×106Throughput (tuples/s)
SHARP
ICDE'20DARLING
RIRS (d)MATCH_RECOGNIZE Throughput
Fig. 10: Impact of pattern length.
1 2 4 8 16
Time Window Size (1000 events)0255075100Recall(%)SHARP
ICDE'20DARLING
RIRS
(a)Recall
1 2 4 8 16
Time Window Size (1000 events)0×1062×1065×1068×10610×10612×106Throughput (events/s)
SHARP
ICDE'20DARLING
RIRS (b)Throughput
Fig. 11: Impact of time window size.
Fig.9 shows the results. We can observe from Fig.9a that Sharp
keeps stable recall of 100% across all selectivity configurations,
outperforming baselines. In contrast, the recall values of baselines
change with the selectivity. For instance, the recall of ICDE’20 drops
by 10% when 𝑉’s distribution changes from U(0, 5ˆ106) toU(0,
9ˆ106). The increasing selectivity generates more partial matches
and therefore results in lower throughput for Sharp and all base-
lines ( Fig.9b). We attribute Sharp ’s robustness to its cost model
which efficiently adapts to changes in pattern selectivity.
Pattern length. We control the pattern length of Q1and Q2ranging
from four to eight by changing the length of the Kleene closure,
B+. The patterns are executed for both CEP and MATCH_RECOGNIZE
over DS1.Fig.10 shows that the recall of Sharp is not affected by
pattern length (stable in 100% recall), consistently outperforming
baselines. The recall of baselines fluctuates by 11% with pattern
length. The increased pattern length leads to lower throughput due
to additionally generated partial matches. These results indicate
thatSharp is robust to changes in pattern length, and complex
patterns can benefit more from Sharp .
Time window size. We change the size of sliding time window
ofQ3and Q4, ranging from 1k to 16k events. The slide is one event.
The patterns are evaluated over the data stream from dataset DS2.
Fig.11a shows that Sharp consistently yields the highest recall
compared to baselines. Sharp ’s recall increases with increasing
10 30 50 70 90
Latency Bound (%)406080100Recall (%)SHARP-View
SHARP-InstSHARP-Sep(a)Recall
10 30 50 70 90
Latency Bound (%)0.2×1060.5×1060.8×1061.0×1061.2×106Throughput (events/s)
SHARP-View
SHARP-InstSHARP-Sep (b)Throughput
Fig. 12: Impact of memory management on CEP.
10 30 50 70 90
Latency Bound (%)60708090100Recall (%)SHARP-View
SHARP-InstSHARP-Sep
(a)Recall
10 30 50 70 90
Latency Bound (%)0.5×1061.0×1061.5×1062.0×106Throughput (events/s)
SHARP-View
SHARP-InstSHARP-Sep (b)Throughput
Fig. 13: Impact of memory management on MATCH_RECOGNIZE .
time window, from 95% to 100%. This is because larger time window
provides more historical statistics for the cost model to learn, which
allows Sharp to select more promising state. As for the throughput,
larger time window increases the size of maintained state, resulting
in lower throughput for Sharp and baselines.
6.4 Impact of Pattern Sharing Schemes
State materialization mechanism. We first analyse how state
materialization approaches affect Sharp ’s performance. To this end,
we consider (i) sharing by instance ( Sharp -Inst)—the shared sub-
patterns are immediately materialized in runtime [ 28,37], and (ii)
sharing by view ( Sharp -View)—the shared sub-patterns are lazy-
materialized until the complete matches are generated [ 29,40]. (iii)
We also examine the extreme case that patterns maintains separate
physical state replicas of shared sub-patterns. ( Sharp -Sep).
Fig.12 shows the results of executing Q3and Q4over dataset
DS1.Sharp -View achieves the highest recall at all latency bounds
(see Fig.12a), 1.1ˆhigher than Sharp -Inst and 5ˆhigher than
Sharp -Sep. Because Sharp -View can select a single partial match
for bespoke shared patterns at fine granularity by controlling the
bitmap mask. In contrast, Sharp -Inst either select a partial match for
all shared patterns or neglect them, i.e., a coarse granular selection.
Sharp -Sep is unaware of the shared state materialization and there-
fore, cannot exploit the optimization opportunity of stateful shared
patterns. For the throughput ( Fig.12b), Sharp -View is higher than
Sharp -Inst due to its efficient in-memory reference count. Sharp -
Sep performs the worst because of its redundant state replicas.
Sharing position in patterns. We use Q3and Q4as templates and
control the position of the shared sub-patterns. To this end, we
change the offset of the shared sub-pattern from the beginning of
the pattern schema, from 0 to 3, and measure the average recall
ofSharp ,RSand RI. As showed in Fig.14, the sharing position
does not affect Sharp ’s performance. The recalls of Sharp is much
11

0 1 2 3
Shared Subpattern Offset020406080100Recall (%)SHARP RS RI(a)CEP Patterns
0 1 2 3
Shared Subpattern Offset020406080100Recall (%)SHARP RS RI (b)MATCH_RECOGNIZE Patterns
Fig. 14: Impact of the position of shared subpattern.
8K 9K 10K 11K 12K 13K 14K
Event Offset of the Event Stream020406080100Recall (%)
Window 0.5K
Window 1.0K
Window 2.0K
Window 4.0K
(a)Recall of Q3
8K 9K 10K 11K 12K 13K 14K
Event Offset of the Event Stream020406080100Recall (%)
Window 0.5K
Window 1.0K
Window 2.0K
Window 4.0K (b)Recall of Q4
Fig. 15: Sharp ’s adaptivity to concept drifts for shared Q 3and Q 4.
10 30 50 70 90
Memory Bound (%)0255075100Accuracy (%)SHARP RS SS
(a)Accuracy
10 30 50 70 90
Memory Bound (%)0255075100Recall (%)SHARP RS SS (b)Recall
Fig. 16: Impact of memory constraint on Sharp ’s performance in the
GraphRAG application.
higher than RSand RI, but they are all stable at different sharing
positions. This is because Sharp ’s pattern-sharing degree captures
the sharing position and the cost model can adapt to it. The random
approaches, RSand RI, are unaware of the sharing position.
6.5 Adaptivity to Concept Drifts
This section investigates Sharp ’s adaptivity to concept drifts. We
evaluate Q3and Q4over data streams from the dataset DS1. To control
the concept drift, we change the value distribution of 𝐷.𝑉 from
U(1ˆ106, 3.5ˆ106) toU(1, 2ˆ106) at the offset of 9k in the event
stream. Fig. 15 illustrates the results. We can observe an abrupt
drop of recall (to 18%) immediately after the concept drift at the
offset 9k. This is because Sharp ’s cost model selects the partial
matches that are no longer promising due to the flipped value
distribution. However, Sharp is able to swiftly detect the drift and
quickly updates its cost model to improve the recall. After one
time window, Sharp improves the recall back to promised value
and stabilizes between 95% to 100%. The convergence is slower for
larger time windows because of the longer lifespan of stale partial
matches–the cost mode takes more time to learn from them.6.6 Impact on Resource Constraints
We examine how Sharp copes with resource constrains of limited
memory capacity. We deploy the GraphRAG pipeline in Fig.1a
with different memory bounds, ranging from 90% to 10% of its orig-
inal memory footprint, We measure the accuracy and recall, com-
pared with two baselines, random_state (RS) and selective_state
(SS).Fig.16 shows the results. Sharp maintains high accuracy at
all memory bounds, reaching 95% at 50% memory and 80% at 10%
memory bound, that is 2.6 ˆand 2.5ˆhigher than RSand SS. As for
the recall, Sharp also outperforms baselines at all memory bounds
by 1.4ˆ(RS) and 1.2ˆ(SS). But the recall value drops to 40% at 10%
memory. The reason is that the recall depends on statistical effi-
ciency of LLM generated responses – a small set of responses will
not cover the majority of ground truth. However, Sharp ’s selection
of state ensures that over 80% of the generated response indeed
align the ground truth.
7 RELATED WORK
Multi-pattern sharing. Multi-pattern CEP has attracted much
attention [ 20,20,28,37–40,55]. These approaches process overlap-
ping sub-patterns simultaneously to avoid redundant computation
and save memory for share patterns. In particular, SPASS [ 40] es-
timates the benefit of sharing based on intra- and inter- query
correlations. Sharon [ 39] and HAMLET [ 37] further support on-
line aggregation. While MCEP [ 20], GRETA [ 38] and GLORIA [ 28]
allow sharing in Kleene closure. These approaches are complemen-
tary to Sharp , with the focus on improving resource utilization and
throughput. In contrast, Sharp ’s focal point is to stratify the latency
bound. We also integrate above sharing schemes into Sharp .
Load shedding. Load shedding techniques discard a set of data
elements without processing them based on their estimated utility
for the query result [ 9,14,16,44–46,50,58]. Load shedding has also
been used in CEP systems. Input-based shedding [ 9,45,46] drops
input events based on their estimated importance. In contrast, state-
based load shedding [ 44,57] discard maintained partial matches
using utilities based on probabilistic models. Hybrid shedding [ 58],
on the other hand, combines the shedding of input events and
partial matches and uses a cost model to balance the trade-offs.
However, the above load shedding schemes target on single-pattern
settings instead of shared multiple patterns.
Approximate query processing (AQP). AQP estimates the result
of queries [ 10], primarily for aggregate queries. The goal is to fast
approximate answers, e.g., based on sampling [ 24] or workload
knowledge [ 18,36]. For aggregation queries, sketches [ 11] may
be employed for efficient, but lossy data stream processing. AQP
was also explored for sequential pattern matching [ 23], focusing on
delivering complete matches that deviate from what is specified in a
pattern. Although AQP aims at fast delivering best-effort processing,
the goal is different from Sharp .Sharp detects exact complete
matches that are specified in patterns, not the approximated one.
8 CONCLUSIONS
In this paper, we studied the problem of best-effort matching of
sequential patterns, as realized in data processing systems in var-
ious applications from complex event processing through online
12

analytical processing to retrieval-augmented generation. These ap-
plications impose latency bounds, which are challenging to meet
given that the state to maintain during pattern evaluation grows
exponentially in the size of the input. To address these scenarios, we
presented Sharp , a library to reduce the state of pattern matching in
overload situations. It takes into account that common applications
evaluate a many similar patterns, which share sub-patterns and
hence their state in terms of partial matches. Sharp incorporates
such sharing through a novel, efficient data structure to keep track
of how different patterns share a partial match. Based thereon, a
cost model enables us to assess the contribution of a partial match
to all patterns and its computational overhead, which serves as the
basis for effective and efficient state reduction.
We evaluated Sharp in comprehensive experiments, using syn-
thetic as well as real-world datasets. When considering evaluation
scenarios that adopt a latency bound of 50% of the average latency,
Sharp achieves a recall of 97%, 96% and 73% for pattern match-
ing in CEP, OLAP, and RAG applications, respectively. We further
demonstrated the robustness of Sharp regarding several pattern
properties and its adaptivity to concept drift in the underlying data.
REFERENCES
[1] 2024. Citi Bike. http://www.citibikenyc.com/system-data.
[2] Serge Abiteboul and Victor Vianu. 1997. Regular path queries with constraints.
InProceedings of the sixteenth ACM SIGACT-SIGMOD-SIGART symposium on
Principles of database systems . 122–133.
[3] Zahid Abul-Basher. 2017. Multiple-query optimization of regular path queries.
In2017 IEEE 33rd International Conference on Data Engineering (ICDE) . IEEE,
1426–1430.
[4] Zahid Abul-Basher, Nikolay Yakovets, Parke Godfrey, and Mark H Chignell. 2016.
SwarmGuide: Towards Multiple-Query Optimization in Graph Databases.. In
AMW .
[5]Samira Akili, Steven Purtzel, and Matthias Weidlich. 2024. DecoPa: Query
Decomposition for Parallel Complex Event Processing. Proc. ACM Manag. Data
2, 3, Article 132 (May 2024), 26 pages. doi:10.1145/3654935
[6] Rodrigo Alves. 2019. Azure Stream Analytics now supports MATCH_RECOGNIZE .
https://azure.microsoft.com/en-us/blog/azure-stream-analytics-now-
supports-match-recognize/
[7] Renzo Angles, Marcelo Arenas, Pablo Barceló, Aidan Hogan, Juan Reutter, and Do-
magoj Vrgoč. 2017. Foundations of modern query languages for graph databases.
ACM Computing Surveys (CSUR) 50, 5 (2017), 1–40.
[8] AWS Machine Learning Blog. 2024. Improving retrieval-augmented generation
accuracy with GraphRAG. https://aws.amazon.com/blogs/machine-learning/
improving-retrieval-augmented-generation-accuracy-with-graphrag/
[9] Koral Chapnik, Ilya Kolchinsky, and Assaf Schuster. 2021. DARLING: data-aware
load shedding in complex event processing systems. Proceedings of the VLDB
Endowment 15, 3 (2021), 541–554.
[10] Surajit Chaudhuri, Bolin Ding, and Srikanth Kandula. 2017. Approximate Query
Processing: No Silver Bullet. In Proceedings of the 2017 ACM International Con-
ference on Management of Data, SIGMOD Conference 2017, Chicago, IL, USA, May
14-19, 2017 , Semih Salihoglu, Wenchao Zhou, Rada Chirkova, Jun Yang, and Dan
Suciu (Eds.). ACM, 511–519. doi:10.1145/3035918.3056097
[11] Graham Cormode, Minos N. Garofalakis, Peter J. Haas, and Chris Jermaine. 2012.
Synopses for Massive Data: Samples, Histograms, Wavelets, Sketches. Found.
Trends Databases 4, 1-3 (2012), 1–294. doi:10.1561/1900000004
[12] Gianpaolo Cugola and Alessandro Margara. 2012. Processing flows of informa-
tion: From data stream to complex event processing. ACM Computing Surveys
(CSUR) 44, 3 (2012), 1–62.
[13] Carlos Gomes Da Silva, João Clímaco, and José Rui Figueira. 2008. Core problems
in bi-criteriat0, 1u-knapsack problems. Computers & Operations Research 35, 7
(2008), 2292–2306.
[14] Nihal Dindar, Peter M Fischer, Merve Soner, and Nesime Tatbul. 2011. Efficiently
correlating complex events over live and archived data streams. In Proceedings of
the 5th ACM international conference on Distributed event-based system . 243–254.
[15] Kasia Findeisen. 2021. Row pattern recognition with MATCH_RECOGNIZE . https:
//trino.io/blog/2021/05/19/row_pattern_matching.html
[16] Yeye He, Siddharth Barman, and Jeffrey F. Naughton. 2014. On Load Shedding
in Complex Event Processing. In Proc. 17th International Conference on Database
Theory (ICDT), Athens, Greece, March 24-28, 2014 , Nicole Schweikardt, Vassilis
Christophides, and Vincent Leroy (Eds.). OpenProceedings.org, 213–224. doi:10.5441/002/ICDT.2014.23
[17] ISO/IEC JTC 1/SC 32 Data management and interchange. 2016. ISO/IEC TR 19075-
5:2016 Information technology – Database languages – SQL Technical Reports – Part
5: Row Pattern Recognition in SQL . Technical Report. International Organization
for Standardization (ISO). https://www.iso.org/standard/65143.html
[18] Saehan Jo and Immanuel Trummer. 2024. ThalamusDB: Approximate Query
Processing on Multi-Modal Data. Proc. ACM Manag. Data 2, 3 (2024), 186. doi:10.
1145/3654989
[19] Jiho Kim, Yeonsu Kwon, Yohan Jo, and Edward Choi. 2023. KG-GPT: A Gen-
eral Framework for Reasoning on Knowledge Graphs Using Large Language
Models. In Findings of the Association for Computational Linguistics: EMNLP 2023 ,
Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational
Linguistics, Singapore, 9410–9421. doi:10.18653/v1/2023.findings-emnlp.631
[20] Ilya Kolchinsky and Assaf Schuster. 2019. Real-time multi-pattern detection over
event streams. In Proceedings of the 2019 International Conference on Management
of Data . 589–606.
[21] Michael Körber, Nikolaus Glombiewski, and Bernhard Seeger. 2021. Index-
accelerated pattern matching in event stores. In Proceedings of the 2021 Interna-
tional Conference on Management of Data . 1023–1036.
[22] Keith Laker. 2017. MATCH_RECOGNIZE and predicates - everything you need to
know . https://blogs.oracle.com/datawarehousing/post/match_recognize-and-
predicates-everything-you-need-to-know
[23] Zheng Li and Tingjian Ge. 2016. History is a mirror to the future: Best-effort
approximate complex event matching with insufficient resources. Proc. VLDB
Endow. 10, 4 (2016), 397–408. doi:10.14778/3025111.3025121
[24] Xi Liang, Stavros Sintos, Zechao Shang, and Sanjay Krishnan. 2021. Combining
Aggregation and Sampling (Nearly) Optimally for Approximate Query Process-
ing. In SIGMOD ’21: International Conference on Management of Data, Virtual
Event, China, June 20-25, 2021 , Guoliang Li, Zhanhuai Li, Stratos Idreos, and
Divesh Srivastava (Eds.). ACM, 1129–1141. doi:10.1145/3448016.3457277
[25] Xunyun Liu and Rajkumar Buyya. 2020. Resource management and scheduling in
distributed stream processing systems: a taxonomy, review, and future directions.
ACM Computing Surveys (CSUR) 53, 3 (2020), 1–41.
[26] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. 2024. Reasoning
on Graphs: Faithful and Interpretable Large Language Model Reasoning. In The
Twelfth International Conference on Learning Representations, ICLR 2024, Vienna,
Austria, May 7-11, 2024 . OpenReview.net. https://openreview.net/forum?id=
ZGNWW7xZ6Q
[27] Thibaut Lust and Jacques Teghem. 2012. The multiobjective multidimensional
knapsack problem: a survey and a new approach. International Transactions in
Operational Research 19, 4 (2012), 495–520.
[28] Lei Ma, Chuan Lei, Olga Poppe, and Elke A Rundensteiner. 2022. Gloria: Graph-
based Sharing Optimizer for Event Trend Aggregation. In Proceedings of the 2022
International Conference on Management of Data . 1122–1135.
[29] Frank McSherry, Andrea Lattuada, Malte Schwarzkopf, and Timothy Roscoe. 2020.
Shared Arrangements: practical inter-query sharing for streaming dataflows.
Proc. VLDB Endow. 13, 10 (2020), 1793–1806. doi:10.14778/3401960.3401974
[30] Yuan Mei and Samuel Madden. 2009. Zstream: a cost-based query processor for
adaptively detecting composite events. In Proceedings of the 2009 ACM SIGMOD
International Conference on Management of data . 193–206.
[31] Microsoft. 2024. GraphRAG - Microsoft Research. https://microsoft.github.io/
graphrag/
[32] Thi Nguyen, Linhao Luo, Fatemeh Shiri, Dinh Phung, Yuan-Fang Li, Thuy-Trang
Vu, and Gholamreza Haffari. 2024. Direct Evaluation of Chain-of-Thought in
Multi-hop Reasoning with Knowledge Graphs. In Findings of the Association for
Computational Linguistics: ACL 2024 , Lun-Wei Ku, Andre Martins, and Vivek
Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand,
2862–2883. doi:10.18653/v1/2024.findings-acl.168
[33] City of Chicago. 2024. Crimes - 2001 to Present. https://data.cityofchicago.org/
Public-Safety/Crimes-2001-to-Present/ijzp-q8t2.
[34] Anil Pacaci, Angela Bonifati, and M Tamer Özsu. 2020. Regular path query evalu-
ation on streaming graphs. In Proceedings of the 2020 ACM SIGMOD International
Conference on Management of Data . 1415–1430.
[35] Marta Paes. 2019. MATCH_RECOGNIZE: where Flink SQL and Complex Event
Processing meet . https://www.ververica.com/blog/match_recognize-where-flink-
sql-and-complex-event-processing-meet
[36] Yongjoo Park, Barzan Mozafari, Joseph Sorenson, and Junhao Wang. 2018. Ver-
dictDB: Universalizing Approximate Query Processing. In Proceedings of the
2018 International Conference on Management of Data, SIGMOD Conference 2018,
Houston, TX, USA, June 10-15, 2018 , Gautam Das, Christopher M. Jermaine, and
Philip A. Bernstein (Eds.). ACM, 1461–1476. doi:10.1145/3183713.3196905
[37] Olga Poppe, Chuan Lei, Lei Ma, Allison Rozet, and Elke A Rundensteiner. 2021.
To share, or not to share online event trend aggregation over bursty event
streams. In Proceedings of the 2021 International Conference on Management of
Data . 1452–1464.
[38] Olga Poppe, Chuan Lei, Elke A. Rundensteiner, and David Maier. 2017. GRETA:
graph-based real-time event trend aggregation. Proc. VLDB Endow. 11, 1 (Sept.
2017), 80–92. doi:10.14778/3151113.3151120
13

[39] Olga Poppe, Allison Rozet, Chuan Lei, Elke A Rundensteiner, and David Maier.
2018. Sharon: Shared online event sequence aggregation. In 2018 IEEE 34th
International Conference on Data Engineering (ICDE) . IEEE, 737–748.
[40] Medhabi Ray, Chuan Lei, and Elke A Rundensteiner. 2016. Scalable pattern
sharing on event streams. In Proceedings of the 2016 international conference on
management of data . 495–510.
[41] Microsoft Research. 2024. GraphRAG: Unlocking LLM Discovery on Narra-
tive Private Data. https://www.microsoft.com/en-us/research/blog/graphrag-
unlocking-llm-discovery-on-narrative-private-data/
[42] Rebecca Sattler, Sarah Kleest-Meißner, Steven Lange, Markus L. Schmid, Nicole
Schweikardt, and Matthias Weidlich. 2025. DISCES: Systematic Discovery of
Event Stream Queries. Proc. ACM Manag. Data 3, 1, Article 32 (Feb. 2025), 26 pages.
doi:10.1145/3709682
[43] Siemens. 2024. Artificial Intelligence: Industrial Knowledge Graph.
https://www.siemens.com/global/en/company/stories/research-technologies/
artificial-intelligence/artificial-intelligence-industrial-knowledge-graph.html
[44] Ahmad Slo, Sukanya Bhowmik, Albert Flaig, and Kurt Rothermel. 2019. pspice:
Partial match shedding for complex event processing. In 2019 IEEE International
Conference on Big Data (Big Data) . IEEE, 372–382.
[45] Ahmad Slo, Sukanya Bhowmik, and Kurt Rothermel. 2019. eSPICE: Probabilistic
Load Shedding from Input Event Streams in Complex Event Processing. In
Proceedings of the 20th International Middleware Conference (Davis, CA, USA)
(Middleware ’19) . Association for Computing Machinery, New York, NY, USA,
215–227. doi:10.1145/3361525.3361548
[46] Ahmad Slo, Sukanya Bhowmik, and Kurt Rothermel. 2020. hSPICE: state-aware
event shedding in complex event processing. In Proceedings of the 14th ACM
International Conference on Distributed and Event-Based Systems (Montreal, Que-
bec, Canada) (DEBS ’20) . Association for Computing Machinery, New York, NY,
USA, 109–120. doi:10.1145/3401025.3401742
[47] Snowflake. 2021. Identifying Sequences of Rows That Match a Pattern . https:
//docs.snowflake.com/en/user-guide/match-recognize-introduction.html
[48] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun
Gong, Lionel M. Ni, Heung-Yeung Shum, and Jian Guo. 2024. Think-on-Graph:
Deep and Responsible Reasoning of Large Language Model on Knowledge Graph.
InThe Twelfth International Conference on Learning Representations, ICLR 2024,
Vienna, Austria, May 7-11, 2024 . OpenReview.net. https://openreview.net/forum?
id=nnVO1PvbTv
[49] Xingyu Tan, Xiaoyang Wang, Qing Liu, Xiwei Xu, Xin Yuan, and Wenjie Zhang.
2025. Paths-over-Graph: Knowledge Graph Empowered Large Language ModelReasoning. arXiv:2410.14211 [cs.CL]
[50] Nesime Tatbul, Uğur Çetintemel, Stan Zdonik, Mitch Cherniack, and Michael
Stonebraker. 2003. Load shedding in a data stream manager. In Proceedings 2003
vldb conference . Elsevier, 309–320.
[51] Sarisht Wadhwa, Anagh Prasad, Sayan Ranu, Amitabha Bagchi, and Srikanta
Bedathur. 2019. Efficiently Answering Regular Simple Path Queries on Large
Labeled Networks. In Proceedings of the 2019 International Conference on Man-
agement of Data, SIGMOD Conference 2019, Amsterdam, The Netherlands, June 30
- July 5, 2019 , Peter A. Boncz, Stefan Manegold, Anastasia Ailamaki, Amol Desh-
pande, and Tim Kraska (Eds.). ACM, 1463–1480. doi:10.1145/3299869.3319882
[52] Eugene Wu, Yanlei Diao, and Shariq Rizvi. 2006. High-performance complex
event processing over streams. In Proceedings of the 2006 ACM SIGMOD interna-
tional conference on Management of data . 407–418.
[53] Wen-tau Yih, Matthew Richardson, Chris Meek, Ming-Wei Chang, and Jina Suh.
2016. The Value of Semantic Parse Labeling for Knowledge Base Question
Answering. In Proceedings of the 54th Annual Meeting of the Association for
Computational Linguistics (Volume 2: Short Papers) , Katrin Erk and Noah A. Smith
(Eds.). Association for Computational Linguistics, Berlin, Germany, 201–206.
doi:10.18653/v1/P16-2033
[54] Haopeng Zhang, Yanlei Diao, and Neil Immerman. 2014. On complexity and
optimization of expensive queries in complex event processing. In Proceedings of
the 2014 ACM SIGMOD international conference on Management of data . 217–228.
[55] Shuhao Zhang, Hoang Tam Vo, Daniel Dahlmeier, and Bingsheng He. 2017. Multi-
query optimization for complex event processing in SAP ESP. In 2017 IEEE 33rd
International Conference on Data Engineering (ICDE) . IEEE, 1213–1224.
[56] Yuyu Zhang, Hanjun Dai, Zornitsa Kozareva, Alexander Smola, and Le Song.
2018. Variational Reasoning for Question Answering With Knowledge Graph.
Proceedings of the AAAI Conference on Artificial Intelligence 32, 1 (Apr. 2018).
doi:10.1609/aaai.v32i1.12057
[57] Bo Zhao. 2018. Complex Event Processing under Constrained Resources by State-
Based Load Shedding. In 34th IEEE International Conference on Data Engineering,
ICDE 2018, Paris, France, April 16-19, 2018 . IEEE Computer Society, 1699–1703.
doi:10.1109/ICDE.2018.00218
[58] Bo Zhao, Nguyen Quoc Viet Hung, and Matthias Weidlich. 2020. Load shedding
for complex event processing: Input-based and state-based techniques. In 2020
IEEE 36th International Conference on Data Engineering (ICDE) . IEEE, 1093–1104.
[59] Erkang Zhu, Silu Huang, and Surajit Chaudhuri. 2023. High-performance row
pattern recognition using joins. Proceedings of the VLDB Endowment 16, 5 (2023),
1181–1195.
14
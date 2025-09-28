# A Knowledge Graph-based Retrieval-Augmented Generation Framework for Algorithm Selection in the Facility Layout Problem

**Authors**: Nikhil N S, Amol Dilip Joshi, Bilal Muhammed, Soban Babu

**Published**: 2025-09-22 17:29:10

**PDF URL**: [http://arxiv.org/pdf/2509.18054v1](http://arxiv.org/pdf/2509.18054v1)

## Abstract
Selecting a solution algorithm for the Facility Layout Problem (FLP), an
NP-hard optimization problem with a multiobjective trade-off, is a complex task
that requires deep expert knowledge. The performance of a given algorithm
depends on specific problem characteristics such as its scale, objectives, and
constraints. This creates a need for a data-driven recommendation method to
guide algorithm selection in automated design systems. This paper introduces a
new recommendation method to make such expertise accessible, based on a
Knowledge Graph-based Retrieval-Augmented Generation (KG RAG) framework. To
address this, a domain-specific knowledge graph is constructed from published
literature. The method then employs a multi-faceted retrieval mechanism to
gather relevant evidence from this knowledge graph using three distinct
approaches, which include a precise graph-based search, flexible vector-based
search, and high-level cluster-based search. The retrieved evidence is utilized
by a Large Language Model (LLM) to generate algorithm recommendations with
data-driven reasoning. The proposed KG-RAG method is compared against a
commercial LLM chatbot with access to the knowledge base as a table, across a
series of diverse, real-world FLP test cases. Based on recommendation accuracy
and reasoning capability, the proposed method performed significantly better
than the commercial LLM chatbot.

## Full Text


<!-- PDF content starts -->

A Knowledge Graph-based Retrieval-Augmented
Generation Framework for Algorithm Selection in the
Facility Layout Problem
Nikhil N S1, Amol Dilip Joshi1,2, Bilal Muhammed2, Soban Babu2
1Indian Institute of Science, Bengaluru, India
nikhilns@iisc.ac.in
2TCS Research, Tata Consultancy Services Ltd.
amol.joshi@tcs.com, bilal.muhammed@tcs.com, soban.bb@tcs.com
Abstract
Selecting a solution algorithm for the Facility Layout
Problem (FLP), an NP-hard optimization problem with
a multiobjective trade-off, is a complex task that re-
quires deep expert knowledge. The performance of a
given algorithm depends on specific problem character-
istics such as its scale, objectives, and constraints. This
creates a need for a data-driven recommendation method
to guide algorithm selection in automated design systems.
This paper introduces a new recommendation method
to make such expertise accessible, based on a Knowl-
edge Graph-based Retrieval-Augmented Generation (KG-
RAG) framework. To address this, a domain-specific
knowledge graph is constructed from published literature.
The method then employs a multi-faceted retrieval mech-
anism to gather relevant evidence from this knowledge
graph using three distinct approaches, which include a
precise graph-based search, flexible vector-based search,
and high-level cluster-based search. The retrieved ev-
idence is utilized by a Large Language Model (LLM)
to generate algorithm recommendations with data-driven
reasoning. The proposed KG-RAG method is compared
against a commercial LLM chatbot with access to the
knowledge base as a table, across a series of diverse,
real-world FLP test cases. Based on recommendation
accuracy and reasoning capability, the proposed method
performed significantly better than the commercial LLM
chatbot.
Keywords:Facility Layout Problem, Algorithm Selec-
tion, Knowledge Graph, Retrieval-Augmented Genera-
tion (RAG), Large Language Models (LLM), Recom-
mender Systems.
1 Introduction
The Facility Layout Problem (FLP) is an important part
of industrial engineering and operations research. [1] Be-
cause the operating cost and service performance are di-rectly influenced by effective layout design. The suit-
able placement of facilities reduces the total operating
costs by 20% to 50% [2]. The computational complexity
of FLP stems from its combined discrete and continu-
ous nature. In practice, this complexity is magnified by
multi-objective trade-offs. For example, minimize ma-
terial handling costs and maximize safety. This multi-
objective problem is coupled with constraints such as
aspect-ratio, adjacency and circulation constraints, and
dynamic considerations such as reconfigurations to ac-
commodate product-mix changes, demand changes, or
equipment changes. These considerations render many
FLP variants NP-hard, so exact mathematical program-
ming is computationally tractable only for small or spe-
cially structured instances [3]. This FLP is also dynamic
in nature because of evolving real-world demand. The
problem gets further complicated because selecting a suit-
able optimisation algorithm for this complex problem is
challenging and requires expert knowledge.
The Solution techniques have been evolving as the com-
plexity of the FLP increases, as mentioned above. Ini-
tial solutions, such as quadratic assignment problem [4],
mixed-integer programming and graph-theoretic formu-
lations [5] for equal-area FLP. These are optimum guar-
anteed, but with exponential growth in computational
time, limiting industrial applications to problems. As the
problem size grows, researchers shifted to heuristic ap-
proaches, such as construction and improvement heuris-
tics. Here, the construction heuristics build layouts from
the ground up, for example, automated layout design pro-
gram and computerized relationship layout planning [6]
whereas the improvement heuristics build solutions in-
crementally, for example computerized relative allocation
of facilities technique [6]. These techniques are likely to
converge to suboptimal layouts. To enhance the ability of
handling large and complex search spaces, various meta-
heuristic techniques, including particle swarm optimiza-
tion, genetic algorithms, ant simulated annealing, colony
optimization, and others, have been introduced [7]. These
1arXiv:2509.18054v1  [cs.IR]  22 Sep 2025

techniques increase the likelihood of avoiding local op-
tima. Moreover, the hybrid metaheuristic approaches,
such as biased random-key genetic algorithms with lin-
ear programming repair, have recently been developed as
state-of-the-art solutions, with enhanced solution quality
and scalability [8].
Despite these developments, the situation in FLP today
is a broad and scattered collection of niche algorithms.
This is typical of the no free lunch theorems, which state
that when performance is averaged across all possible op-
timization problems, no single algorithm is universally
superior. The theorems imply that an algorithm per-
forms better on one class of problems and may perform
weaker on others. For example, an algorithm that per-
forms well in small-scale, static settings may be desired
in large-scale, multi-objective, or dynamic conditions [9].
As a result, attention has turned to the meta-problem
of algorithm selection, which involves selecting the algo-
rithm, representation, and parameters best suited to an
instance of information [21].
The early attempts to address this meta-problem were
static decision-support methods. For instance, a GenOpt-
based framework to guide practitioners in selecting from a
wide range of general optimization algorithms. Their ap-
proach, which utilized flowcharts and choice matrices, was
designed to assist novices by matching algorithm charac-
teristics to problem types and performance requirements.
While this represents a valuable step towards systematiz-
ing algorithm selection, the framework was not specific
to the unique complexities of the FLP. Furthermore, like
other static methods, it is not dynamic and cannot adapt
to new algorithms or shifting problem definitions with-
out manual updates [10]. To improve flexibility, research
employed data-driven approaches, namely meta-learning,
that predict algorithm performance from problem meta-
features, for example, size, constraints, and statistical
properties [11]. SATzilla is just an example, where it
selects the fastest solver from a portfolio based on in-
stance features and performs well in combinatorial opti-
misation [12]. This has evolved to AutoML, where algo-
rithm selection is combined with hyperparameter tuning
in the combined algorithm selection and hyperparameter
framework [13]. While adaptive frameworks and AutoML
have built significant capability, Large Language Mod-
els (LLMs) such as OpenAI’s GPT series and Google’s
Gemini introduce new possibilities [14]. Large, general-
purpose corpora-trained, these models are well-suited to
acquiring language, reasoning, and in-context learning
but fail on domain-specific problems such as FLP optimi-
sation, hallucinations (providing plausible but incorrect
responses) [15] and weak domain adaptation [16].
AI research has turned to grounding LLMs with struc-
tured, verifiable knowledge to address these weaknesses.
Two key paradigms dominate Retrieval-Augmented Gen-
eration (RAG) and Knowledge Graphs (KG). RAG re-
trieves relevant, domain-specific content before genera-
tion, reducing hallucinations and enabling integration ofup-to-date information without retraining [17]. KG or-
ganise domain entities and relationships, enabling ex-
plainable reasoning and mitigating data sparsity or cold-
start issues. The relational pathways in KG enhance ac-
curacy and interpretability, which are critical for decision
support in technical domains [18]. A new approach in-
tegrates these paradigms. A KG-RAG [19] model uses a
structured KG as the retrieval source for a RAG. The KG
offers an explainable, linked domain model, and RAG al-
lows LLMs to flexibly query and blend this knowledge into
transparent, evidence-based proposals. This integration
is suitable for comprehensive domain knowledge and jus-
tifiable output for complex recommendation tasks. The
literature indicates a convergence of a long-standing op-
erations research problem with a contemporary AI op-
portunity. FLP algorithmic fragmentation under the No
Free Lunch principle renders selection a vital bottleneck.
Current machine learning-based selectors, though adap-
tive, tend to operate as black boxes. Concurrently, the
KG-RAG paradigm addresses LLM limitations of halluci-
nations and poor performance in domain adaptation while
allowing for structured, transparent reasoning. KG-RAG
is a promising basis for an intelligent, interpretable, and
adaptive FLP algorithm recommendation method.
This paper introduces a dynamic and domain-specific
solution using the KG-RAG framework to generate data-
driven algorithm recommendations to solve the FLP,
given a user-defined query and a domain-specific knowl-
edge graph of historical solutions. Our approach begins
with constructing a formal KG that captures the complex
relationships between FLP instances, their unique charac-
teristics, and the performance of various solution method-
ologies. To leverage this structured knowledge base, the
method employs a multi-faceted retrieval pipeline that in-
tegrates precise graph queries, flexible vector search, and
high-level cluster analysis to gather comprehensive evi-
dence. An LLM then synthesizes this evidence to gen-
erate data-driven and explainable recommendations for
a complete methodology, including the algorithm, algo-
rithm parameters, a suitable problem representation, and
a constraint handling technique. This work includes an
intelligent feedback loop, which allows the method to
function as a continuously learning environment by in-
corporating new user-submitted solutions, thus ensuring
its long-term relevance and accuracy.
The rest of the paper has been structured as follows:
Section 2 presents the proposed architecture and method-
ology, detailing the construction, domain-specific Knowl-
edge Graph, and the multi-faceted retrieval and recom-
mendation pipeline design. Section 3 provides a compara-
tive experimental evaluation, benchmarking the KG-RAG
method against a strong baseline LLM to empirically vali-
date its performance on recommendation accuracy and its
ability to generate data-driven, interpretable reasoning.
2

2 Architecture and methodology
The facility layout problem concerns placing a set of facil-
ities (machines, departments, storage areas, etc.) inside a
bounded site to optimise one or more performance mea-
sures. Practical FLP variations span multiple dimensions
that materially affect solver choice and complexity. These
begin with geometric representation, from the equal-area
rectangles of early models [4] to the unequal area, aspect-
ratio rectangles and general polygons of modern appli-
cations [1]. The problem scale is equally decisive [18],
with the objective structure further constraining algo-
rithm suitability, as single-objective methods, for exam-
ple, minimizing material-handling cost, efficient area util-
isation, minimizing material flow, may be ineffective for
multi-objective trade-offs, for example, cost vs. safety vs.
maintainability and vice versa [20]. Then the constraint
coupling fundamentally shapes the feasibility landscape,
including non-overlap, adjacency requirements, boundary
constraints, and other geometric or operational limits.
Together, these factors create a vast search space, leading
to high variability in algorithm performance across differ-
ent FLP instances. This diversity, as discussed in Section
2, underscores the necessity for a specialised algorithm
recommendation method to adjust to each case’s unique
features.
Figure 1: Example facility layout of production area [28]
Bridging this gap requires a systematic algorithm-
selection approach that moves beyond generic recommen-
dations. An ideal method must perform instance-aware
matching, aligning suggestions with the specific scale,
geometry, and constraints of the input account for ob-
jective type, ensuring alignment with single- or multi-
objective goals and provide constraint-handling guidance,
recommending methods such as repair operators, penalty
functions, or exact feasibility checks. Beyond naming
a method, it should supply validated algorithm param-
eters, such as population size for a genetic algorithm and
starting temperature and cooling schedule for simulatedannealing, as users often lack expertise in setting effec-
tive hyperparameters. For adoption, the method must
ensure explainability and provenance, justifying recom-
mendations with historical performance data, and deliver
actionable outputs.
2.1 Architectural overview
A recommendation method is proposed to address the
previously identified requirements. The method’s archi-
tecture, depicted in Figure 2, comprises several key com-
ponents, each designed to address a specific research gap.
•Initialisation of user query input:The process
begins with the user defining their FLP. This compo-
nent addresses the need for input flexibility, allowing
users to specify diverse combinations of objectives,
number of facilities, and constraints, moving beyond
the rigid definitions.
•Hybrid evidence retrieval:The method initi-
ates a parallel, three-streamed retrieval process to
overcome the limitations of static and single-faceted
search methods. This component provides a compre-
hensive and adaptable search by collecting evidence
from the Neo4j Knowledge Graph. This includes:
– Graph-based search:A precise graph-based
search using Cypher to find exact or structurally
similar problems.
– Vector-based search:A dynamic vector-
based search to find conceptually related prob-
lems based on the user’s text description.
– Cluster-based search:An aggregate analysis
to identify high-level trends for the given prob-
lem type, for example, top algorithms for large-
scale, multi-objective problems.
•Evidence compilation:The results from all three
retrieval channels are integrated and compiled into
one contextualized dossier. The evidence is then or-
ganized into a close-fitting prompt template with the
user’s original query to anchor the language model.
•LLM-powered recommendation:To address the
black box nature and the risk of LLM hallucina-
tions, the evidence from all three retrieval channels is
passed to this component. The results are compiled
into a contextualized dossier that grounds the Google
Gemini LLM. The LLM is prompted not to use its
general knowledge but to act as an expert analyst,
compiling the provided data into a final, formatted
recommendation with a clear, evidence-driven expla-
nation.
•Continuous learning via feedback:The design
has a feedback loop. When users add new solved
problem cases, the method learns and incorporates
this information smartly, enriching the KG. This
3

makes the method continuously learn and improve,
allowing it to make more accurate and suitable rec-
ommendations in the future.
Figure 2: High-level workflow of the recommendation
The method is implemented in Python, utilizing a mod-
ern technology stack composed of Neo4j for the graph
database and vector index, Google Gemini as the LLM
and embedding engine, LangChain for orchestrating the
RAG pipeline, and Gradio for the interactive web user
interface.
2.2 Knowledge graph construction
The method’s foundation is a designed KG that trans-
forms a flat, tabular dataset into a rich, interconnected
network of domain knowledge.
Table 1: Schema of a CSV source file
Column Type Example
problem id string P 6, D 10
num facilities integer 6, 10
floor W, floor H float 30, 90
Problem representation string continuous space
facility dimension data string fixed dim fixed area
objective string min material handling cost
constraints string non-overlapping
constraint handling string shapely intersection
method string GA, PSO, TabuSearch
Model parameters String pop size=50;n gen=200;
cost float 1147.781, 18446.919
time sec float 185.0, 290.19
source(reference) String Knez, M. and Gajsek, B.,2.2.1 Data corpus and schema
The initial knowledge base was derived from a CSV file, a
comprehensive dataset aggregating solved FLP instances
from academic literature and benchmarks. Each row rep-
resents a unique problem-solution pair, capturing critical
attributes as detailed in Table 1.
2.2.2 Graph schema design
This tabular data, as Table 1 schema, is loaded into a
Neo4j graph database through a custom Cypher script.
The graph schema, used in Table 2, and 3 is defined to de-
scribe the domain entities and their complex relationships
explicitly.
Table 2: Nodes and properties schema
Node label Key properties
Problem problem id, num facilities
Method name
Objective name
Constraint name
Representation name
ConstraintHandling name
Solution id (unique), cost, time sec
ObjectiveCluster classification of objective
ScaleCluster Classification on problem’s size
ConstraintHandlingCluster category of cons handling
Table 3: Relationship schema
Relationship structure
(Solution) - [SOLVED]→(Problem)
(Solution) - [USED METHOD]→(Method)
(Problem) - [HAS OBJECTIVE]→(Objective)
(Problem) - [HAS CONSTRAINT]→(Constraint)
(Problem) - [HAS REPRESENTATION]→(Representation)
(Problem) - [CONS HANDLING]→(ConsHandling)
(Problem) - [BELONGS TOSCALE]→(ScaleCluster)
(Method) - [IS TYPE OF]→(MethodCluster)
(ConsHandling) - [IS TYPE OF]→(ConsHandlingCluster)
(Problem) - [OBJECTIVE CLUSTER]→(ObjectiveCluster)
2.2.3 Automated clustering and vector-based in-
dexing
The KG is enriched with categorical clusters and vector-
based embeddings for higher-level reasoning.
•Automated clustering:Following the first import
of data, a sequence of Cypher queries automatically
clusters nodes according to deterministic rules. For
instance, issues are clustered into a scale cluster by
4

their number of facilities and into an objective clus-
ter by whether the objective name contains a comma,
which is a reasonable heuristic for multi-objective
problems. This rule-based, automated mechanism
brings consistency and scalability with the later ad-
ditions of data appended.
•Embedding generation:Each problem node is
augmented with a vector embedding to enable con-
ceptual search. Key attributes, such as objective and
constraints, induce a full-text description property.
The text is passed through the Google AI embed-
ding model, and the resultant vector is cached in the
node and indexed into Neo4j’s native vector index.
The process is done once on new data, indexing all
problems for similarity search.
2.3 The hybrid RAG recommendation
pipeline
Whenever the user provides the query (number of facil-
ities, objectives, constraints, and other specifications, if
any), the method initiates a multi-stage pipeline to gather
complete evidence before making a recommendation. Af-
ter submission, the method normalizes and validates the
inputs. The selections from the dropdowns are treated as
validated entities. The free-text entries are treated as un-
validated entities. This distinction is critical as it dictates
how the graph-based search query is constructed.
Adaptive query generation:The algorithm gener-
ates a set of evidence-gathering queries from the validated
and unvalidated inputs. The initial step is exhaustive and
serves as the basis for the rest of the steps of the RAG
pipeline.Also, the population embedding process happens
at method boot time and, significantly, is re-run for every
new instance of the problem added by the user feedback
loop.
2.3.1 Multi-faceted evidence retrieval
The method employs three parallel retrieval strategies to
build a rich context for the LLM.
•Graph-based search:The algorithm performs an
exact, structured search in the KG. A Cypher query
is dynamically built from user inputs. The query
has a package of intelligent features loaded for ex-
ceptional cases.
– Flexible filtering:The query applies an or
constraint between constraints and goals to pre-
vent the search from narrowing down too much.
Besides that, it also dynamically regulates the
quantity of facilities. Instead of searching for
an exact amount, the Cypher query searches
for problems in a±25% range of the number
given by the user (i.e., between 0.75 * n and
1.25 * n). This dynamic range significantly in-
creases the possibility of retrieving meaningfulhistorical information, even if no exact number
matches the facility number in the knowledge
graph.
– Performance-oriented ordering:The re-
trieved results are not random; a multi-factor
scoring method ranks them to favour the most
holistically suitable solutions. This relevance-
first prioritises finding the correct solution over
merely finding a high-performing one. The OR-
DER BY clause now prioritises results in a
high-level order: first by the highest objective
score (best matching the most user-specified
objectives), second by the highest constraint
score (best matching the most user-specified
constraints). Only as a final resort does it fall
back to using proximity to the user’s number
of facilities, the minimum cost, and the mini-
mum time as tiebreakers. This means that the
highest-ranked results are high-performing and,
most importantly, relevant to the user’s prob-
lem.
– Multi-objective handling: If the user picks
more than one objective, the query logic is
then optimized to favour pulling solutions for
issues that are tagged explicitly in the multi-
objective category within the KG. This ensures
that pulled examples are relevant to the prob-
lem of balancing competing objectives.
– Adaptive fallback for unprecedented
scale:The method fails gracefully when a
user query is beyond the range of the current
knowledge base. During boot time, the method
preloads the maximum number of facilities in
the KG. When its initial graph-based search
fails and the requested number of facilities by
the user is greater than this preloaded maxi-
mum, a fallback query is triggered programmat-
ically. The query smartly broadens its search
to find the most relevant large-scale precedents.
It retrieves solutions from problems that meet
two conditions. Problems with a facility count
in the top quartile of the dataset’s known scale.
Problems belonging to the top-level scale clus-
ter.
•Vector-based search:The user’s free-text problem
description is translated into a vector representation.
The vector then performs a similarity search against
the indexed problem nodes in Neo4j. This returns
conceptually similar problems.
•Cluster-based search:To further support the pro-
vision of high-level statistical context, the method
analyses the list of problem IDs found by the graph-
based search (before applying the final limit). It
does not aggregate queries to find the top three most
common patterns to solutions for the Scale cluster
5

and objective cluster associated with the user query.
This gives strong trend-based evidence as well as the
instance-level results.
2.3.2 LLM compilation and generation
All the top-ranked graph-based outcomes, top vector-
based matches, and cluster analysis trends retrieved evi-
dence are combined into one comprehensive context. The
dossier is written in a strict prompt template, with the
original user query submitted to the Google Gemini LLM.
The prompt tells the model to write as a professional an-
alyst, combining all the presented evidence to answer the
form of two sections: a straightforward recommendation
and a clear data-driven reasoning part, clearly explaining
the recommendation using the retrieved data. Such an
approach is required to guarantee the output’s trust and
transparency. By tightly coupling the LLM with the pro-
vided context, the method architecturally rules out model
hallucination risk and refrains from making recommenda-
tions on untested assumptions.
Finally, it incorporates a continuous learning loop to
ensure the method remains current and avoids the static
knowledge base problem. When users provide new solved
problem instances, a backend process ensures stable and
autonomous knowledge integration. This includes data
standardization, such as combining multiple objectives
into a single, canonical string to maintain uniformity in
the KG. The method then utilizes Cypher’s MERGE com-
mand to intelligently update the graph, either linking to
existing entities or creating new ones without duplication.
This feature enables dynamic and continuous learning,
enriching the knowledge base and allowing the method
to make more accurate and relevant recommendations in
the future.
3 Comparative evaluation
This section presents the experimental design and results
of a comparative study conducted to evaluate the per-
formance of our proposed KG-RAG method. The pri-
mary objective is empirically confirming that grounding
an LLM in a well-structured, domain-specific Knowledge
Graph leads to significantly accurate and interpretable
recommendations than a baseline LLM approach using
raw, structured data.
3.1 Evaluation methodology
The evaluation is designed as a direct comparative anal-
ysis between our proposed model and a strong baseline
model, with five distinct test cases designed to cover a
range of real-world FLP scenarios, from simple, single-
objective problems to more complex multi-objective and
ill-defined queries. Finally, Section 4 concludes by sum-
marizing the paper’s primary contributions and key find-ings, discussing the method’s current limitations, and
outlining promising directions for future research.
•Baseline model:The baseline model consists of the
same Google Gemini 1.5 Flash LLM as the proposed
model, but instead of a KG, it is provided with the
entire CSV dataset file as its knowledge source. This
represents a powerful, traditional RAG method that
relies on the LLM’s ability to find and reason with
information from a flat, tabular text file.
•Ground truth establishment:A ground truth
recommendation was established for each test case
to create a fair benchmark for accuracy. This was
not a subjective choice but the result of a system-
atic, manual analysis of the source dataset, following
a process that mirrors the logic of ground truth find-
ing.
Figure 3: Ground truth establishment flow
•Metrics:Each method’s performance was measured
against the ground truth using two key criteria.
1.Recommendation accuracy:binary mea-
sure (1 for a match, 0 for a mismatch) of
whether the method’s top recommended algo-
rithm(s) matched or included the ground truth.
2.Reasoning quality:A qualitative score from
1 (Poor) to 5 (Excellent) assessing the extent to
which the LLM’s explanation was detailed, rich,
and verifiably based on the evidence provided.
The reasoning quality for both methods was
evaluated by a separate instance of the Google
Gemini 2.5 Flash LLM, which was prompted to
act as an impartial judge and score the outputs
based on the following criteria.
(a)1 (Poor):Irrelevant or ambiguous expla-
nation.
(b)2 (Weak):Plausible but not specific.
(c)3 (Acceptable):Reasonable but unsup-
ported by cited evidence.
(d)4 (Good):Applies to some numbers, but
the analysis is superficial.
(e)5 (Excellent):Combines several correct,
cited evidence points to construct a firm,
evidence-based conclusion.
6

Figure 4: User Interface
3.2 Results and discussion
Table 4 compares outcomes of the proposed and baseline
model with the ground truth and the data-driven ranking
based on their explainability.
3.2.1 Analysis of baseline method performance
While capable of reasoning, the baseline LLM’s recom-
mendations achieve an overall accuracy of only 20% (1
out of 5). Its failures highlight the limitations of reason-
ing over structured tabular data. In Test cases 1 & 4, the
baseline failed to recommend the ground truth algorithm
because its reasoning was limited to a superficial raw text
analysis. It could not perform the high-level frequency
analysis (like the KG-RAG’s cluster analysis) needed to
identify the most applied and successful methods. More-
over, it fails to interpolate the number of facilities when
the exact number of facilities is not present in the dataset,
as in test case 4. In Test case 2, the baseline’s recommen-
dation of ACO-FBS for a multi-objective problem was
factually incorrect. A query against the provided dataset
would have revealed that ACO-FBS is only associated
with Single-Objective problems. The baseline, lacking
this explicit relationship, made a plausible but techni-
cally flawed suggestion, resulting in a reasoning score of
1.
3.2.2 Analysis of KG-RAG method performance
The KG-RAG method performed exceptionally well,
achieving 100% recommendation accuracy and a perfect
5/5 reasoning score on all test cases. Its strength lies
in its ability to synthesize multiple layers of evidence
Figure 5: Model recommendation rating analysis
7

Table 4: Comparative evaluation of recommendation methods
Sl. Test case Ground
truthBaseline
Rec.Baseline
RatingKG-RAG
Rec.KG-
RAG
Rating
1 No of facilities = 10, Objective = min
material handling cost, Constraints =
non-overlapping, boundary constraints.CRO-SL ACO-FBS 4 CRO-SL,
BRKGA5
2 No of facilities = 15, Objective = max
closeness rating, min material handling
cost, Constraints = non-overlapping,
boundary constraints, aspect ratio.HSA ACO-FBS 1 HSA, HGA 5
3 No of facilities = 30, Objective = min
material handling cost, Constraints =
non-overlapping, boundary constraint,
area requirement, aspect ratio.BRKGA-
LP, GA-LPBRKGA-
LP5 BRKGA-LP,
GA-LP5
4 No of facilities = 40, Objective = min
material handling cost, Constraints =
non-overlapping, boundary constraint,
aspect ratio.BRKGA-
LPACO-FBS 3 BRKGA-LP,
ACO-FBS.5
5 No of facilities = 30, Objective = None,
Constraints = NoneConstruction
HeuristicPROP1 4 BRKGA-LP,
Construction
Heuristic5
from the structured graph. In Test case 1, the KG-RAG
model’s reasoning addressed specific problem instances
and higher-level trends from the objective and scale clus-
ters. This allowed it to correctly identify CRO-SL as
the top contender, a conclusion the baseline missed. In
Test case 2, the method’s ability to explicitly query for
problems within the multi-objective cluster was critical.
This ensured that the evidence provided to the LLM was
highly relevant, leading to the correct recommendation of
HSA. In Test case 5, even when the query did not con-
tain an objective or constraints, the KG-RAG method
provided a more nuanced and helpful recommendation.
Analysing the broader context of 30-facility problems sug-
gested both a simple heuristic (Construction Heuristic)
and a more powerful metaheuristic (BRKGA-LP), giving
the user a better understanding of the available trade-
offs. This shows that the evaluation confirms that while
a baseline LLM can reason from raw data to a limited
extent, the KG-RAG model’s ability to leverage a struc-
tured, connected knowledge base allows it to perform an
accurate analysis, yielding superior recommendations and
data-driven explanations.
4 Conclusion
The paper introduced a new recommendation method
based on a KG-RAG framework for FLP. This method
successfully integrates the formalized reasoning strength
of a Neo4j graph database with the advanced compilationstrength of an LLM to generate explainable, high-quality
recommendations.
The primary contribution of this paper is a multi-
stage pipeline that integrates graph-based, vector-based,
and cluster-based search approaches to construct a rich,
evidence-supported context for every user query. The
evaluation strongly confirmed the benefit of this ap-
proach. The proposed KG-RAG method significantly out-
performed a baseline LLM with direct access to the raw
data, achieving 100% recommendation accuracy against
the ground truth, compared to 20% for the baseline. Fur-
thermore, the quality of the data-driven reasoning was
consistently rated as an average score of 5/5, far ex-
ceeding the baseline’s average score (3.4/5). This per-
formance stems from the KG-RAG method’s ability to
synthesize high-level trends from clusters with granular
evidence from individual problem nodes, resulting in ac-
curate, transparent, and reliable justifications. The risk
of hallucination is substantially reduced by grounding the
LLM in a domain-specific knowledge graph.
Despite these promising results, the proposed method
has certain limitations. The method’s effectiveness funda-
mentally depends on the quality and scope of its underly-
ing KG. It is a domain-specific tool for the FLP and rec-
ommendations for other domain optimization problems
without a new, relevant knowledge base. Additionally,
while the method includes a fallback mechanism for out-
of-scope queries, for example, the problems with a num-
ber of facilities exceeding the KG’s included maximum
data, its recommendations in these scenarios are based
8

on broader categorical data rather than direct, compara-
ble evidence, making them inherently less precise. Ad-
dressing this requires continuously expanding the knowl-
edge base over time through the intelligent feedback loop.
Furthermore, this paper prescribes a tangible and scal-
able way of building continuously updated methods. The
collective feedback loop, which continuously feeds and ag-
gregates new solutions populated by the users, maintains
the knowledge graph updated over time. This makes the
method a dynamic knowledge base that updates itself
based on the user’s input. Future work will attempt to
extend the framework to other challenging combinatorial
optimization problems.
Acknowledgement
The authors express their sincere gratitude to TCS Re-
search, Tata Consultancy Services Ltd., especially to the
CTO team, for providing the internship opportunity and
valuable guidance that greatly contributed to the com-
pletion of this work.
References
[1] Anjos, M.F. and Vieira, M.V., 2017. Mathemati-
cal optimization approaches for facility layout prob-
lems: The state-of-the-art and future research di-
rections.European journal of operational research,
261(1), pp.1-16.
[2] Sun, X., Lai, L.F., Chou, P., Chen, L.R. and Wu,
C.C., 2018. On GPU implementation of the island
model genetic algorithm for solving the unequal
area facility layout problem.Applied Sciences, 8(9),
p.1604.
[3] Ripon, K.S.N., Khan, K.N., Glette, K., Hovin, M.
and Torresen, J., 2011, July. Using Pareto-optimality
for solving the multi-objective unequal area facility
layout problem. InProceedings of the 13th annual
conference on Genetic and evolutionary computation
(pp. 681-688).
[4] Koopmans, T.C. and Beckmann, M., 1957. Assign-
ment problems and the location of economic activi-
ties.Econometrica: journal of the Econometric So-
ciety, pp.53-76.
[5] Burggr¨ af, P., Adlon, T., Hahn, V. and Schulz-
Isenbeck, T., 2021. Fields of action towards auto-
mated facility layout design and optimization in fac-
tory planning A systematic literature review.CIRP
Journal of Manufacturing Science and Technology,
35, pp.864-871.
[6] Singh, S.P. and Sharma, R.R., 2006. A review of
different approaches to the facility layout problems.The International Journal of Advanced Manufactur-
ing Technology, 30(5), pp.425-433.
[7] Kundu, A. and Dan, P.K., 2012. Metaheuristic in fa-
cility layout problems: current trend and future di-
rection.International Journal of Industrial and Sys-
tems Engineering, 10(2), pp.238-253.
[8] Gon¸ calves, J.F. and Resende, M.G., 2015. A biased
random-key genetic algorithm for the unequal area
facility layout problem.European Journal of Opera-
tional Research, 246(1), pp.86-107.
[9] Wolpert, D.H. and Macready, W.G., 2002. No free
lunch theorems for optimization, IEEE transactions
on evolutionary computation, 1(1), pp.67-82.
[10] Dudhee, V., Abugchem, F. and Vukovic, V., 2020,
September. Decision Support in Algorithm Selec-
tion for Generic Optimisation. InBuilding Simula-
tion and Optimization 2020: IBPSA England’s First
Virtual Conference. IBPSA England.
[11] Brazdil, P.B., Soares, C. and Da Costa, J.P., 2003.
Ranking learning algorithms: Using IBL and meta-
learning on accuracy and time results.Machine
Learning, 50(3), pp.251-277.
[12] Xu, L., Hutter, F., Hoos, H.H. and Leyton-Brown,
K., 2008. SATzilla: portfolio-based algorithm selec-
tion for SAT.Journal of artificial intelligence re-
search, 32, pp.565-606.
[13] Hutter, F., Kotthoff, L. and Vanschoren, J. (eds.)
(2019)Automated machine learning: methods, sys-
tems, challenges. Springer Nature.
[14] Chang, Y., Wang, X., Wang, J., Wu, Y., Yang, L.,
Zhu, K., Chen, H., Yi, X., Wang, C., Wang, Y. and
Ye, W., 2024. A survey on evaluation of large lan-
guage models.ACM transactions on intelligent sys-
tems and technology, 15(3), pp.1-45.
[15] Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z.,
Wang, H., Chen, Q., Peng, W., Feng, X., Qin, B.
and Liu, T., 2025. A survey on hallucination in large
language models: Principles, taxonomy, challenges,
and open questions.ACM Transactions on Informa-
tion Systems, 43(2), pp.1-55.
[16] Magesh, V., Surani, F., Dahl, M., Suzgun, M., Man-
ning, C.D. and Ho, D.E., 2025. Hallucination-Free?
Assessing the Reliability of Leading AI Legal Re-
search Tools.Journal of Empirical Legal Studies,
22(2), pp.216-242.
[17] Fan, W., Ding, Y., Ning, L., Wang, S., Li, H., Yin,
D., Chua, T.S. and Li, Q., 2024, August. A survey
on rag meeting llms: Towards retrieval-augmented
large language models. InProceedings of the 30th
ACM SIGKDD conference on knowledge discovery
and data mining(pp. 6491-6501).
9

[18] Guo, Q., Zhuang, F., Qin, C., Zhu, H., Xie, X.,
Xiong, H. and He, Q., 2020. A survey on knowledge
graph-based recommender systems.IEEE Transac-
tions on Knowledge and Data Engineering, 34(8),
pp.3549-3568.
[19] Ibrahim, N., Aboulela, S., Ibrahim, A. and Kashef,
R., 2024. A survey on augmenting knowledge graphs
(KGs) with large language models (LLMs): models,
evaluation metrics, benchmarks, and challenges.Dis-
cover Artificial Intelligence, 4(1), p.76.
[20] Drira, A., Pierreval, H. and Hajri-Gabouj, S., 2007.
Facility layout problems: A survey. Annual reviews
in control, 31(2), pp.255-267.
[21] Smith-Miles, K.A., 2009. Cross-disciplinary perspec-
tives on meta-learning for algorithm selection. ACM
Computing Surveys (CSUR), 41(1), pp.1-25.
10
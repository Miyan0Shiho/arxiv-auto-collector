# The Pneuma Project: Reifying Information Needs as Relational Schemas to Automate Discovery, Guide Preparation, and Align Data with Intent

**Authors**: Muhammad Imam Luthfi Balaka, Raul Castro Fernandez

**Published**: 2026-01-07 05:58:54

**PDF URL**: [https://arxiv.org/pdf/2601.03618v1](https://arxiv.org/pdf/2601.03618v1)

## Abstract
Data discovery and preparation remain persistent bottlenecks in the data management lifecycle, especially when user intent is vague, evolving, or difficult to operationalize. The Pneuma Project introduces Pneuma-Seeker, a system that helps users articulate and fulfill information needs through iterative interaction with a language model-powered platform. The system reifies the user's evolving information need as a relational data model and incrementally converges toward a usable document aligned with that intent. To achieve this, the system combines three architectural ideas: context specialization to reduce LLM burden across subtasks, a conductor-style planner to assemble dynamic execution plans, and a convergence mechanism based on shared state. The system integrates recent advances in retrieval-augmented generation (RAG), agentic frameworks, and structured data preparation to support semi-automatic, language-guided workflows. We evaluate the system through LLM-based user simulations and show that it helps surface latent intent, guide discovery, and produce fit-for-purpose documents. It also acts as an emergent documentation layer, capturing institutional knowledge and supporting organizational memory.

## Full Text


<!-- PDF content starts -->

The Pneuma Project: Reifying Information Needs as Relational
Schemas to Automate Discovery, Guide Preparation,
and Align Data with Intent
Muhammad Imam Luthfi Balaka
luthfibalaka@uchicago.edu
The University of Chicago
Chicago, USARaul Castro Fernandez
raulcf@uchicago.edu
The University of Chicago
Chicago, USA
ABSTRACT
Data discovery and preparation remain persistent bottlenecks in
the data management lifecycle, especially when user intent is vague,
evolving, or difficult to operationalize. The Pneuma Project intro-
ducesPneuma-Seeker, a system that helps users articulate and
fulfill information needs through iterative interaction with a lan-
guage model–powered platform. The system reifies the user’s evolv-
ing information need as a relational data model and incrementally
converges toward a usable document aligned with that intent. To
achieve this, the system combines three architectural ideas: context
specialization to reduce LLM burden across subtasks, a conductor-
style planner to assemble dynamic execution plans, and a conver-
gence mechanism based on shared state. The system integrates
recent advances in retrieval-augmented generation (RAG), agen-
tic frameworks, and structured data preparation to support semi-
automatic, language-guided workflows. We evaluate the system
through LLM-based user simulations and show that it helps surface
latent intent, guide discovery, and produce fit-for-purpose docu-
ments. It also acts as an emergent documentation layer, capturing
institutional knowledge and supporting organizational memory.
CCS CONCEPTS
•Information systems →Data management systems;Infor-
mation retrieval;•Human-centered computing;
KEYWORDS
Data Discovery, Data Preparation, Information Needs, Large Lan-
guage Models, Human-Computer Interaction
1 INTRODUCTION
Large Language Models (LLMs) are poised to support the entire data
management lifecycle, from collection to task execution [ 9,18,38,
40]. Among the stages that have most resisted automation are data
discovery, the task of identifying and retrieving documents relevant
to a user’sinformation need[ 4,7], and data preparation, the trans-
formation of those documents into a usable form for downstream
tasks. A key challenge lies in the nature of the user’s information
need: it is often vague, evolving, and difficult to express in a way
that software can act on effectively. As LLMs become increasingly
capable, the bottleneck shifts from executing tasks to helping users
This paper is published under the Creative Commons Attribution 4.0 International
(CC-BY 4.0) license. Authors reserve their rights to disseminate the work on their
personal and corporate Web sites with the appropriate attribution, provided that you
attribute the original work to the authors and CIDR 2026. 16th Annual Conference on
Innovative Data Systems Research (CIDR ’26). January 18-21, Chaminade, USAarticulate their goals—their information needs—precisely enough
to be fulfilled with available data.
Consider a practical example from our university’s Finance de-
partment:“What impact will tariffs have on our organization?”An-
swering this question requires addressing several steps:
(1)Discover relevant data sources, such as current tariff schedules
and procurement records;
(2)Define “impact,” which may include both direct (e.g., imported
goods from tariffed countries) and indirect effects (e.g., tariffed
components in otherwise unaffected imports);
(3)Determine temporal scope: Is the user interested in projected
impact next week, next fiscal year, or in a retrospective analysis?;
(4)Integrate these data sources into a coherent document that sup-
ports meaningful interpretation.
To support such inquiries, a system must extract, make explicit,
and operationalize the assumptions behind the question. This means
surfacing the “information need, ” which is the set ofstates of nature
required to answer the user’s question [ 3]. This articulated infor-
mation need then guides discovery, integration, and preparation.
In practice, these processes are rarely executed by a single user;
modern organizations involve business analysts, data scientists,
and data engineers collaborating in a semi-decentralized way to
answer complex questions [36].
Pneuma-Seekeris designed to help users articulate and fulfill
such information needs semi-automatically. A central insight be-
hindPneuma-Seekeris that an information need can be reified
as a data model, specifically, a relational schema. The system then
seeks to align that schema with available data, returning a docu-
ment that satisfies the latent information need. Initially, neither the
system nor the user may know exactly what this document looks
like, but through interaction, both sides refine their understanding,
e.g., the user may not ask to include the effect of direct and indirect
tariffs initially, but expresses this explicitly after seeing an inter-
mediate output. The user offers language-based feedback, and the
system proposes increasingly aligned data models, converging over
iterations toward one that meets the desired information need.
When the user asksPneuma-Seekerthe tariff question, the
system examines a procurement database with dozens of tables,
determines that tariff information is missing, retrieves relevant
data from online sources, integrates the information into a tabular
structure, and proposes a preliminary document for review. After
two rounds of user feedback, the system converges on a schema and
SQL query that estimates tariff exposure from German suppliers,
matching the user’s evolving understanding of “impact.”arXiv:2601.03618v1  [cs.DB]  7 Jan 2026

CIDR’26, January 18-21, 2026, Chaminade, USA Muhammad Imam Luthfi Balaka and Raul Castro Fernandez
LLMs inPneuma-Seekerserve as a bridge between language,
which users employ to express intent, and data models, which the
system uses to steer discovery and preparation.Pneuma-Seeker
leverages recent advances in retrieval-augmented generation (RAG)
[16] and agentic architectures to make this bridge actionable. To
enable this functionality,Pneuma-Seekerintroduces three gener-
alizable contributions:
•Context Specialization.LLMs struggle with large, diverse con-
texts in structured data tasks [ 15,17,19].Pneuma-Seekerad-
dresses this by decomposing the workflow into subtasks with
specialized contexts and narrow scopes.
•Conductor-Style Planning.Rather than static, rule-based pipe-
lines,Pneuma-Seekeremploys a conductor component, an LLM-
powered agent that assembles plans dynamically, based on real-
time evidence of progress toward fulfilling the information need.
•Convergence Criteria.Pneuma-Seekertreats the evolving data
model as a shared state between user and system: the user refines
it via language, and the conductor uses it to guide downstream
modules. Convergence occurs when the document aligns with
the user’s latent information need.
To evaluatePneuma-Seekerwithout human subjects, we use
LLM Sim, an LLM-based agent that acts as users, reacting only to
system outputs. While not a full substitute for real studies, both
quantitative and qualitative results showPneuma-Seekerhelps
articulate and fulfill information needs.
Pneuma-Seeker’s design offers an important side effect: it acts as
a mechanism for organizational knowledge capture. By prompting
users to articulate their goals, it surfaces tacit information needs
and latent dataflows. At scale, these interactions accumulate into
emergent documentation, preserving tribal knowledge and institu-
tional memory. As teams evolve,Pneuma-Seekerstrengthens the
resilience of internal data ecosystems by making this knowledge
explicit and reusable.
2 FROM INFORMATION NEEDS TO ANSWERS
In this section, we define relevant concepts, survey the landscape of
solutions that help users satisfy those needs, and present an inter-
action model that motivates the design ofPneuma-Seeker. We aim
to highlight recurring challenges in how users articulate and refine
their information needs, and to motivate a structured approach
in which human information needs and system representations
co-evolve toward the latent information need.
2.1 Definitions
Information Need.An information need is the set of states of
nature required to solve a data-driven task [ 3]. This definition is
general: an information need may include the features required
for training a classifier, the schema to answer a SQL query, or the
variables for causal inference. These states are often encoded in
documents. For clarity, we assume one information need per task,
though multiple valid representations may exist.
Latent and Active Information Needs.The latent information
need is the true set of states needed to solve a task, often initiallyunknown to the user. The active information need is the user’s work-
ing hypothesis about what data is needed, which evolves through
interaction and exploration to approximate the latent one.
2.2 Landscape of Solutions
Document Retrieval.Once an active information need is specified,
a wide array of technologies can help retrieve relevant documents.
For example, web search engines map keyword queries to matching
documents. In structured databases, SQL queries retrieve relations
containing specific columns, assuming join paths exist. RAG sys-
tems enable LLMs to retrieve relevant text chunks from vector stores
or databases before composing answers. These retrieval methods
assume that the information need is reasonably well articulated,
which is often the core bottleneck [2, 13, 33].
Identifying Information Needs.A separate but critical challenge
is helping users formulate their information needs in the first place.
Many user interfaces offer scaffolding to support this process: web
autocomplete features leverage population-level priors; LLM inter-
faces often suggest next actions or clarifying prompts; enterprise
data catalogs [ 12] surface usage patterns (e.g., users querying ta-
ble A and B often also use table C), aiding newcomers in forming
expectations about what data is relevant.
Representing Information Needs.The notion of an information
need has long been studied, especially in information retrieval
(IR) [ 2,27]. In IR, an information need is often framed as the
user’s underlying goal, which a query only imperfectly expresses.
In exploratory search, users engage in open-ended, investigative
information seeking where goals evolve during the search pro-
cess [ 22,34]. Similarly, in exploratory data analysis (EDA), the
focus shifts to iterative refinement and hypothesis evolution, often
guided by visualization [ 11,29]. In the Pneuma project we build
on these lines of work to provide computational representations of
information needs that human users and machines can co-evolve.
In this work, we reify an information need as a relational data
model plus a SQL query over that model. This choice reflects our
view that solving a data-driven task ultimately involves instantiat-
ing a structured document (a table or set of tables) that aligns with
the user’s intent. The system’s job, then, is to identify the schema
and query that materialize the latent information need.
2.3 Aligning Data with Intent: A Model
We propose the following model to guide system design. A human
user seeks to solve a data-driven task. That task induces a latent in-
formation need, which, if represented as a document, would suffice
to solve the task. However, the user may not initially know what
that document looks like.
An abstract system, initially agnostic to the task, holds an inter-
nal state representing its evolving understanding of the information
need. The interaction proceeds in iterations: the user communi-
cates their active information need, and the system updates its state
accordingly. After observing the system’s output, the user revises
their input, gradually steering the system closer to the latent infor-
mation need. The interaction concludes when the system produces
a document (or schema + query) that the user deems sufficient.

The Pneuma Project: Reifying Information Needs as Relational Schemas to Automate Discovery, Guide Preparation,
and Align Data with Intent CIDR’26, January 18-21, 2026, Chaminade, USA
In contemporary practice, this system is not software alone—it
is a composite of human roles and infrastructure: business analysts,
data scientists, data engineers, domain experts, databases, interfaces,
APIs, and more [ 36]. WithPneuma-Seeker, we aim to unify these
roles under a coherent software system that helps users identify,
articulate, and materialize their information needs through iterative,
language-guided interaction.
3 THEPNEUMA-SEEKERSYSTEM
In this section, we describePneuma-Seeker, a system that helps
users identify and fulfill their latent information needs. We first
present its architecture and key design principles, followed by a
detailed explanation of each component.
3.1 Technical Contributions and Overview
Pneuma-Seekeris designed around three technical insights:
Context specialization.When working with structured data, an
LLM is constrained by context size heterogeneity. The more het-
erogeneous the context, the more attention is spread across un-
related details. In addition, prompting an LLM with distinctroles
can help focus its behavior [ 28]. Thus,Pneuma-Seekeradopts a
multi-component architecture in which retrieval, integration, and
orchestration are separated. Each component focuses on a single
subtask, and the LLM receives only the information relevant to its
assigned role.
Dynamic planning withConductor.The subtasks induced
by context specialization must still be assembled into an end-to-
end process of identifying and fulfilling a user’s information need.
Rather than relying on a static, rule-based pipelines,Pneuma-
SeekerusesConductor, an LLM-powered agent that orchestrates
the process based on real-time evidence of progress toward fulfilling
the information need.
Shared state for convergence.Pneuma-Seekerrepresents a
user’s active information need as a relational data model (𝑇, 𝑄) ,
where 𝑇is a set of tables, and 𝑄is a sequence of SQL queries over
𝑇. The user andPneuma-Seekerestablish a feedback loop: the user
describes their active information need as it changes to the system,
the system updates (𝑇, 𝑄) , and the user reacts to those changes
with additional feedback. The interaction ends when the user stops
or when the active information need matches the latent information
need (i.e., convergence).
These three insights directly shapePneuma-Seeker’s modular
architecture. Figure 1 shows the architecture, which consists of
multiple components:(1)Conductor, which orchestrates the
process;(2)IR System, which retrieves relevant data; and(3)
Materializer, which integrates and prepares data to form 𝑇. We
describe them in detail below.
3.2Conductor
ConductordrivesPneuma-Seekertowards convergence by select-
ing actions on the fly to align (𝑇, 𝑄) with a user’s active information
need. At the moment, it selects actions one at a time, but there has
been some research on parallelized LLM planning (e.g., [ 37,41]). In
Conductor, an action is any of the following:
User
4Pneuma -Seeker
Document DB Pneuma -Retriever Web Search
Python 
Interpreter
SQL 
Executor
MaterializerIR SystemState ( T,Q)ConductorFigure 1: The Architecture ofPneuma-Seeker
•Internal reasoning.Conductorengages in internal reasoning
(inspired by ReAct [ 35]), where we prompt the LLM so that it is
able to evaluate the current state (𝑇, 𝑄) , retrieved data fromIR
System, and a user’s most recent feedback, to decide the best next
action(s). For example, suppose the user clarifies that determining
the effect of a new tariff requires knowing the previous active
tariff.Conductormight reason:“My current query computes
price change using only the new tariff percentage. I should retrieve
the previous tariff percentage and update the final percentage to
(new_tariff−previous_tariff).”
•Tool call.Conductorcan directly call tools, includingIR Sys-
tem,Materializer, andSQL Executor, or invoke them after
first identifying the need through internal reasoning. For exam-
ple, if it determines that the previous active tariff is required, it
may issue a retrieval request toIR System:“Retrieve the previously
active tariff for the region. ”Similarly, if it decides that 𝑇should
be materialized, it may invokeMaterializerwith a note such
as:“Materialize 𝑇, a table containing relevant procurement data
with new and previously active tariff information for the supplier
country. ”
•State modification.Conductorcan update the state (𝑇, 𝑄) to
revise interpretations of the user’s active information need. It
may modify only 𝑇, only 𝑄, or both. For example, after deter-
mining that 𝑄should account for the previous active tariff, it
may update 𝑄to look like this:[“SELECT price * (1 + new_tariff -
previous_tariff) AS new_price FROM procurement_data”].
•User-facing communication.Conductorcan interact directly
with the user to explain actions taken, summarize the current
state, ask clarifying questions, or propose next steps. For instance,
after executing the queries in 𝑄, it may inform the user:“I have
executed the queries in 𝑄. The new price for items bought from
supplier 12345 in Germany is 5% higher, with an average increase
of $5,125. ”
By design,Conductoroperates without fixed procedural rules,
aside from essential dependencies (e.g., 𝑇must be materialized
before executing 𝑄).Conductorlimits the number of consecutive
actions to a fixed value 𝑖, which we set to 𝑖=5based on empirical
observation that the LLM rarely hits this limit during our evaluation.
This limit is intended to prevent (𝑇, 𝑄) from moving away from the
latent information need before user feedback can correct it, while
also avoiding long autonomous runs that keep users waiting. In
addition, we instructConductorto end each sequence of actions

CIDR’26, January 18-21, 2026, Chaminade, USA Muhammad Imam Luthfi Balaka and Raul Castro Fernandez
with a user-facing message whenever possible. If the action limit
is reached without producing a user-facing message, the system
interrupts and forcesConductorto do so.
When interacting with the user,Conductorgrounds its deci-
sions on data retrieved fromIR System, rather than relying solely
on assumptions (e.g., assuming we have suppliers with ID 12345
in a procurement table). This avoids producing unrealistic (𝑇, 𝑄)
states that cannot be materialized. For instance, if the user requests
an analysis of the impact of a new tariff for a specific supplier in
2019, butIR Systemonly retrieves tariff records from 2020 onward,
Conductorcan detect the gap and either search alternative sources
or notify the user, rather than spending multiple actions building
and materializing a (𝑇, 𝑄) . This prevents wasted effort and keeps
the interaction focused on attainable outcomes.
Throughout an interaction,Pneuma-Seekersurfaces not only
user-facing responses but also the current state (𝑇, 𝑄) to the user.
We display the interface in Figure 2 (as of August 2025). Displaying
the state (box 3; sample rows are shown for 𝑇) alongside the chat
interface (box 1 and box 2) allows users to spot subtle mismatches
between their intent and the system’s evolving model and correct
them. Continuing the tariff-impact example, suppose 𝑇contains
procurement data but lacks the country attribute entirely, even
though the user only cares about German suppliers. Without this at-
tribute, 𝑄cannot filter the data correctly, and the final computation
would be misaligned with the user’s intent.
3.3Information Retrieval (IR) System
IR SystemsupportsConductorandMaterializerby retrieving
relevant data from multiple sources. It abstracts heterogeneous
retrieval format, such as tables and text, into document objects.
This uniform representation allows new retrievers to be added
without changing the rest ofIR System’s design. Nevertheless,
bothConductorandMaterializerknows about the kind of data
available in the system: currently tables, domain knowledge, and
web pages, which are handled by the following three retrievers in
our current implementation:
•Pneuma-Retriever[ 1], a state-of-the-art table discovery sys-
tem with a hybrid index, combining an HNSW [ 21]-based vector
store and a BM25 [ 26]-based inverted index for efficient table
search.
•Document Database, which usesPneuma-Retriever’s indexer
to store domain knowledge. This enables cross-user knowledge
transfer. For example, if one user specifies that estimating tariff
impacts requires accounting for both direct and indirect tar-
iffs, subsequent tariff-related queries can leverage that insight.
Pneuma-Seekerautomatically captures knowledge from user
interactions and save it to Document Database, inspired by prior
work (e.g., [20, 23, 30]).
•Web Search, which provides a thin interface to external search
engines for general or up-to-date information lookup.
With the current implementation,Pneuma -Seekercan already
handle complex users’ information needs that combine structured
and unstructured sources. Building on this design, we will extend
IR Systemto accept direct feedback on retrieved documents from
ConductororMaterializer.3.4 Materializer
Materializer’s sole purpose is to populate 𝑇with data, possibly
involving integration of multi-source data fromIR System.Ma-
terializeris separate fromConductor, reflecting thecontext
specializationprinciple:Materializeroperates only on context
relevant to data integration and transformation, without being dis-
tracted by orchestration details.Materializeralso considers 𝑄, so
it understands filters in the queries. For example, if a query in 𝑄
expects a date column to be of format “yyyy-mm-dd,” while the
column actually lists its values with format “Month Day, Year,”Ma-
terializerwill transform the values of this column by producing
Python code.
Materializer’s toolkit includes a DuckDB [ 25]-basedSQL Execu-
torfor standard relational operations (joins, unions, etc.) and a
Python interpreter equipped with Pandas [ 32] and NumPy [ 10].
Similar toIR System,Materializeris designed to be extensible, al-
lowing new operators to be incorporated easily, including semantic
relational operations (e.g., as defined by LOTUS [24]).
Through its prompts,Materializeris aware of all retrieved
documents fromIR Systemand can leverage them to form SQL
queries or Python code. For example, ifMaterializerretrieves
two documents representing (1) the latest tariffs from Web Search
and (2) internal procurement tables fromPneuma-Retriever, then
it can generate Python code that combines both documents (e.g.,
tariff information becomes a new column in the table). Errors may
occur (e.g., floating-point operations on columns with null values),
so the respective tool analyzes these errors and provides feedback
toMaterializerto fix the generated queries or code.
We plan to explore interaction models in whichConductor
provides incremental feedback toMaterializerafter every 𝑛oper-
ations, mirroring the way users interact withConductor. This ad-
ditional supervision could help preventMaterializerfrom drifting
away from the intended 𝑇, especially in long integration pipelines.
3.5 Dynamic vs. Static Pipelines
In this section, we reflect on previous designs we experimented
with and justifyPneuma-Seeker’s design based on that experience.
We design bothConductorandMaterializeris significantly
more adaptable for incorporating more actions/tools (and hence
cover more use cases) compared to the static pipeline we initially
experimented with—define (𝑇, 𝑄) , retrieve top- 𝑘tables, filter and
integrate the tables via relational operations, and prune the integra-
tion results down to those defined in 𝑇. While such a static pipeline
can cover a variety of use cases, it becomes harder to update as we
incorporate more actions/tools. Whenever a step in the original
pipeline is insufficient (e.g., lacking the actions needed to cover a
new use case), the entire pipeline must be revisited.
For example, in our tariff-impact scenario, there may be cases
where the correct analysis requires joining procurement data with
tariff records, but the sources do not share a common identifier
(e.g., supplier_id ). In such cases, the system needs to employ
more flexible relational operations (e.g., semantic or fuzzy joins)
that were not part of the original pipeline. Incorporating such an
operation into a static, predefined pipeline may require significant
engineering effort, since its position in the pipeline must be chosen

The Pneuma Project: Reifying Information Needs as Relational Schemas to Automate Discovery, Guide Preparation,
and Align Data with Intent CIDR’26, January 18-21, 2026, Chaminade, USA
Figure 2: Interface ofPneuma-Seeker, showing: [1] User Query (Clarification), [2] User-Facing Message, and [3] State View
Page(𝑇, 𝑄). Note: the numbers and values of𝑇shown here are not real for privacy reasons.
carefully and its assumptions must align with both upstream and
downstream operations.
In other words, generalizing a static pipeline to support a wide
variety of use cases is possible, but every extension increasingly
requires substantial effort. As more actions/tools are added, these
integration costs compound. In contrast, withConductororMa-
terializer, we simply define the new operation or tool, and it
naturally fits into their action spaces.
3.6Pneuma-Seekerin Action
To illustrate how the components work together, we revisit the
finance department’s broad question from Section 1:“What impact
will tariffs have on our organization?”
During the interaction, the user adds a key clarification:“Impact
should be calculated relative to the previous active tariff, not just the
current rate. ”
Conductorfirst engages ininternal reasoning, recognizing
that it needs both the new and previous active tariff information to
compute the impact correctly. It then issues a tool call toIR System,
which returns procurement tables fromPneuma-Retrieverand
tariff information from the Web Search interface. Based on this,
Conductordefines 𝑇as a single table combining procurement
data with tariff information. It summarizes the retrieved sources
and asks the user:“I found procurement records and historical tariff
information for the relevant suppliers. I’ve designed 𝑇to combine the
tariff information with the procurement data. Should I materialize 𝑇
now?”Once the user confirms,ConductorcallsMaterializerto
join the procurement and tariff data, thereby populating 𝑇. With 𝑇
materialized,Conductorperforms astate modificationto update
𝑄so that it computes cost impact relative to the previous tariffs:
["SELECT AVG(price * (1 + (new_tariff - prev_tariff)))
AS new_avg_cost FROM procurement_data_joined"]
SQL Executorruns the updated query and returns an estimated
x% increase in procurement costs (about $y) across all goods.1Con-
ductorreports this information to the user throughuser-facing
communication:
“If the new tariffs take effect, procurement costs are estimated to
rise by x% (about $y), calculated relative to the previous rates. ”
At this point, the user may further narrow the scope, e.g., to
only consider lab equipment and suppliers from Germany, trigger
another refinement cycle.
4 EVALUATION
In this section, we offer preliminary evidence ofPneuma-Seeker’s
ability to help users articulate information needs and solve data
tasks by answering the following questions:
•RQ1: Can the user reach their underlying information need by
interacting withPneuma-Seeker?
•RQ2: Given a specific information need, canPneuma-Seeker
address it accurately?
1The actual numbers are not disclosed for privacy reasons.

CIDR’26, January 18-21, 2026, Chaminade, USA Muhammad Imam Luthfi Balaka and Raul Castro Fernandez
Evaluation Workload.We evaluatePneuma-Seekeron the ar-
chaeology and environment datasets from KramaBench [ 14], with
Web Search disabled to prevent leaking benchmark information
from the internet. These datasets are associated with 12 and 20
questions, respectively. Table 1 shows the datasets’ characteristics.
Table 1: Characteristics of the Datasets
Dataset # Tables Avg. #Rows Avg. #Cols
Archeology 5 11,289 16
Environment 36 9,199 10
Each benchmark question is a latent information need. We use
an LLM ( GPT-4o ) to simulate a domain expert (LLM Sim) interacting
with the system. Starting from a broad prompt,LLM Simiteratively
refines its active information need based on the system’s outputs.
For example, given the latent question:
“What is the average Potassium in ppm from the first and last time
the study recorded people in the Maltese area? Assume that Potassium
is linearly interpolated between samples. Round your answer to 4
decimal places. ”
The initial query is:
“I’m curious to dive into the historical data from the Maltese region.
Could you help me get an overview of the different variables we have
for past studies?”
Importantly, convergence is not guaranteed:LLM Sim’s active in-
formation need may never fully match the latent one. For reference,
we display the prompt provided forLLM Simin Figure 3.
4.1 RQ1: Convergence
Convergence means theLLM Sim’s active information need matches
its latent information need. We introduce two metrics:(1) per-
centage of convergence, which is the proportion of benchmark
questions for whichLLM Simconverges, and(2) median turns to
convergence, which is the median of the number of timesLLM Sim
has to prompt a system to achieve convergence (with an imposed
limit of 15).
We comparePneuma-Seekerwith three baselines: BM25-based
full-text search (FTS),Pneuma-Retriever, andLlamaIndex2(a
representative RAG system). FTS andPneuma-Retrieverare static
systems that only return tables, represented by their columns and
sample rows.LlamaIndexadds an LLM on top of a top- 𝑘vector
retriever to interpret the retrieved data forLLM Sim.
The results, as shown in Figure 4 and Figure 5, indicate that
Pneuma-Seekerconsistently achieves the highest percentage of
convergence and similar median turns to convergence asLlamaIn-
dex. BothPneuma-SeekerandLlamaIndexnot only surface rele-
vant data but also interpret it, contextualizing their responses to
LLM Sim. Even though the initial queries fromLLM Simare more
general, the systems surface relevant data and suggestions and com-
municate them back toLLM Sim. Subsequently,LLM Simresponds
to the systems to explore further and move closer to uncovering its
latent information need, and the systems adjust accordingly (e.g.,
by modifying the state).
2https://www.llamaindex.ai/You are simulating {domain_expert_desc}, who is interacting with a
data discovery system to explore insights from an enterprise dataset.
{Depending on system:
-Pneuma-Seeker: The system represents your information need as a set
of target schemas and SQL statements that, if executed, will provide the
answer. It can combine, transform, and reason over data to assist your
exploration.
- FTS orPneuma-Retriever: The system only returns relevant tables
based on your description. It does not infer your deeper intent, combine,
or analyze data. }
Scenario:
- The system already has access to internal datasets.
- You (the simulated user) are familiar with the domain and have seen
similar datasets before.
- You are not uploading new datasets or asking if they exist — you
assume they do.
Possible eventual goal (unknown at start):
{question}
Behavior:
- Explore and refine your question step-by-step depending on the
system’s responses.
- Be vague or explore tangents, just as a curious analyst would.
- Only arrive at the specific question above if the system’s output
correctly leads you there.
Continue your role as the domain expert. This is the conversation so far
(respond as if prompting the system directly):
YOU: {initial_broad_prompt}
Figure 3: Prompt forLLM_Sim
On the other hand, both FTS andPneuma-Retrieverstruggle to
converge becauseLLM Simhas to interpret the results themselves.
Additionally, even thoughPneuma-Retrievercan find the correct
tables in almost all questions,LLM Simcan only observe sample
rows to prevent hitting the context limit. Even with GPT-4o ’s 128k
context limit, 2-3 turns are enough to exceed the limit with our
datasets. In most of the questions,LLM Simkeeps trying to adjust its
queries to get more specific information from the retrieved tables.
There is a latency trade-off betweenPneuma-Seekerand the
other systems. On average,Pneuma-Seekertakes 70.26 seconds
to respond to a prompt, while FTS andPneuma-Retrieveran-
swer almost instantaneously. In addition,Pneuma-Seeker’s LLM,
OpenAI's O4-mini , incurs$1 .1and$4 .4for every 1 million input
and output tokens, respectively. For reference, we include the esti-
mated average costs of interactions betweenLLM SimandPneuma-
Seekeracross several models in Table 2.
4.2 RQ2: Accuracy
Converging to the latent information need is not enough; users
ultimately want to get accurate answers for their information needs.

The Pneuma Project: Reifying Information Needs as Relational Schemas to Automate Discovery, Guide Preparation,
and Align Data with Intent CIDR’26, January 18-21, 2026, Chaminade, USA
Table 2: Estimated Average Token Usage and Costs Across Different LLMs
Dataset Avg. In Avg. OutHaiku 4.5O4-miniO3 gpt-5.1 Sonnet 4.5 Opus 4.5
Tokens TokensIn OutIn OutIn Out In Out In Out In Out
Archeology 248,351 2,854 $0.25 $0.01$0.27 $0.01$0.50 $0.02 $0.31 $0.03 $1.49 $0.04 $1.24 $0.07
Environment 149,011 1,712 $0.15 $0.01$0.16 $0.01$0.30 $0.01 $0.19 $0.02 $0.45 $0.03 $0.75 $0.04
0 2 4 6 8 10
Median Turns to Convergence0255075100Convergence PercentageHigh Convergence, Low Turns
Low Convergence, High TurnsFTS
Pneuma-Retriever
LlamaIndex
Pneuma-Seeker
Figure 4: Comparison of Median Turns to Convergence vs.
Convergence Percentage (Archeology Dataset)
0 2 4 6 8 10
Median Turns to Convergence0255075100Convergence PercentageHigh Convergence, Low Turns
Low Convergence, High TurnsFTS
Pneuma-Retriever
LlamaIndex
Pneuma-Seeker
Figure 5: Comparison of Median Turns to Convergence vs.
Convergence Percentage (Environment Dataset)
FTS andPneuma-Retrieverare not designed to provide answers,
so we exclude them from RQ2 and include new baselines:DS-Guru
and OpenAI’s O3.DS-Guruis the Kramabench’s reference frame-
work, in which it instructs an LLM to decompose a question into
a sequence of subtasks, reason through each step, and synthesize
Python code implement the plan. We select the O3-basedDS-Guru,
as it is the best-performing one.
O3 is one of the best reasoning models right now with 200k
context limit. For each benchmark question, we provide it with
the whole relevant tables, so it has every necessary information
to answer the question. However, we encountered context length
exceeded errors with O3 in 6 out of 12 archaeology questions and
17 out of 20 environment questions. O3 answers none of the six
archaeology questions correctly, but answers two environment
questions correctly. Overall, passing all relevant context is still not
a scalable approach.For the remaining systems, the results are shown in Table 3.
Pneuma-Seekeroutperforms all systems across all datasets.Pneuma-
SeekeroutperformsDS-Guru, even thoughPneuma-Seekeruses
a smaller and more cost-efficient model.LlamaIndex, on the other
hand, does not answer any questions correctly because the ques-
tions require actual computation (e.g., computing average of a cer-
tain column), not just interpretation of the top-𝑘context.
Table 3: Comparison of Accuracy across Datasets
System Archeology Environment
LlamaIndex0.00% 0.00%
DS-Guru(O3) 25.00% 19.60%
Pneuma-Seeker41.67% 55.00%
5 DISCUSSION
In this section, we discuss lessons learned from buildingPneuma-
Seekerand explain our vision for the Pneuma project.
5.1 Lessons Learned.
We builtPneuma-Seekerover the course of the past year. We
learned the following lessons that collectively motivated the archi-
tecture presented in this paper and directly informed the design of
each component and their interactions.
Dynamic pipeline enables better adaptability.Our early proto-
types followed a fixed processing sequence. We can certainly adapt
this pipeline to include more actions/tools and hence cover more use
cases. However, we realized that evolving such pipelines requires
an increasing amount of effort. As soon as we encounter a use case
that is not covered by the system, e.g., requiring new actions or
interactivity, we are forced to do an extensive re-engineering.
This happened repeatedly. For example, when we had to deter-
mine which actions are required or optional, whether to skip or
restart some actions, and when we had to integrate new actions
without destabilizing the existing pipeline. A static approachcan
be adapted to handle these scenarios, but the engineering burden
grows quickly as the system incorporates more and more actions
and functionality.
Dynamic pipelines are not a silver bullet either: if a necessary
capability is missing (e.g., semantic joins), the system will still fail.
However, the key difference is that missing capabilities in a dynamic
framework arelocalized: once implemented, they naturally slot into
the system’s action space. In contrast, adding the same capability
to a static pipeline necessitates revisiting the entire pipeline.

CIDR’26, January 18-21, 2026, Chaminade, USA Muhammad Imam Luthfi Balaka and Raul Castro Fernandez
This realization aligns with the broader spirit of AI-enabled sys-
tems: instead of hard-coding rigid sequences, the system should se-
lect actions dynamically based on evolving state, available tools, and
accumulated domain knowledge. It also resonates with the declara-
tive philosophy of data processing: specifyingwhatis needed, not
howto accomplish it.
Existing systems such asReAcTable[ 39] andChain-of-Table[ 31]
also adopt dynamic orchestration in a more specific setting: assum-
ing a conventional tabular QA setting where a single, specific table
is known upfront. In our setting, the system may need multiple
tables and non-tabular data to satisfy users’ information needs.
Overall, factors such as handling user feedback, maintaining iter-
ative alignment of the state, and integrating heterogeneous data
sources all push strongly against static pipelines. These observa-
tions collectively led us away from predefined pipelines and toward
Conductor’s flexible, state-driven dynamic orchestration.
Context specialization is essential.A key challenge in dynamic
systems is the design of components themselves. We observed the
importance of specializing context across different components
(e.g.,ConductorandMaterializer). Beyond reducing the chance
of hallucinations (since each component focuses narrowly on its
role), specialization helps prevent exceeding context-window limits,
which are more easily reached when the LLM is prompted with
multiple different roles at once.
Looking ahead, although there is active research on extending
context windows and future models likely will support larger win-
dows, there is no guarantee that the model will effectively attend to
(or meaningfully use) all its input tokens, even if they fit. Current
models already have much longer context windows compared to
those available just 3 years ago, but recent work still shows that
LLMs’ performance on tasks such as question answering degrades
as context length grows, even when the relevant information is
fully retrievable [ 6]. Theoretical analyses such as [ 5] also show
that longer sequence length dilutes the model’s ability to focus
on specific tokens. Therefore, we argue that context specialization
remains beneficial, especially as the system grows, e.g., with new
actions and tools.
Schema-first representation guides alignment.Reifying infor-
mation needs as a relational model provides a shared anchor that
both the user and the system can collaboratively reason about. It
gives users a concrete object to sanity-check the results, rather than
relying purely on natural-language, user-facing messages. The state
(𝑇, 𝑄) becomes a structure that can be iteratively refined, corrected,
or extended, reducing miscommunication.
Surfacing intermediate (𝑇, 𝑄) states allows both users (andLLM
Sim) to detect subtle misinterpretations early, before they propa-
gate into later steps. Natural language alone is insufficient; users
need visibility into the evolving data model to provide concrete,
meaningful feedback. It also forces users to think concretely about
what exactly they need. Designing effective ways to communicate
this evolving state (and studying how users interpret and act on it)
remains an important direction for future user studies.
5.2 Our Vision
We began the Pneuma project a year and a half ago, building on
over five years of research in data discovery. Our first milestonewasPneuma-Retriever[ 1]. In this paper, we have presented the
current state of the project, which has since grown to address a
critical bottleneck in data-centric organizations: as LLMs become
increasingly capable and embedded in data workflows, the main
challenge shifts to helping users articulate what they want to get out
of data. Our vision is a system that treats this articulation process
as a first-class concern. This is reflected in our design through the
reification of information needs as relational schemas tailored to
the user’s active intent and constructed on-the-fly.
We highlight an important emergent effect ofPneuma-Seeker’s
design. By prompting users to clearly express their goals, disclose as-
sumptions, and externalize their mental models—a behavior increas-
ingly familiar in LLM-driven interfaces—Pneuma-Seekercaptures
what is often tacit knowledge. Within organizations, this “tribal
knowledge” represents a collective brain trust. If made searchable
and persistent, it could fulfill the vision of internal data markets we
proposed in earlier work [ 8]. Such markets would enable organiza-
tions to extract far greater value from their data assets, going well
beyond the immediate gains in discovery and preparation addressed
in this paper. Over the past several years, we have made progress
toward that vision, but extracting tribal knowledge remains a hard-
to-crack barrier. The Pneuma project is our latest and most focused
attempt to lower that barrier and in doing so, to unlock latent value
from organizational data.
REFERENCES
[1]Muhammad Imam Luthfi Balaka, David Alexander, Qiming Wang, Yue Gong,
Adila Krisnadhi, and Raul Castro Fernandez. Pneuma: Leveraging llms for tabular
data representation and retrieval in an end-to-end system.Proc. ACM Manag.
Data, 3(3), June 2025. doi: 10.1145/3725337. URL https://doi.org/10.1145/3725337.
[2]N.J. Belkin. Anomalous states of knowledge as a basis for information retrieval.
Canadian Journal of Information Science, 5(1):133–143, 1980.
[3]Raul Castro Fernandez. What is the value of data? a theory and systematization.
ACM / IMS J. Data Sci., 2(1), June 2025. doi: 10.1145/3728476. URL https://doi.
org/10.1145/3728476.
[4]Raul Castro Fernandez, Ziawasch Abedjan, Famien Koko, Gina Yuan, Samuel
Madden, and Michael Stonebraker. Aurum: A data discovery system. In2018
IEEE 34th International Conference on Data Engineering (ICDE), pages 1001–1012,
2018. doi: 10.1109/ICDE.2018.00094.
[5]Shi Chen, Zhengjiang Lin, Yury Polyanskiy, and Philippe Rigollet. Critical atten-
tion scaling in long-context transformers, 2025. URL https://arxiv.org/abs/2510.
05554.
[6]Yufeng Du, Minyang Tian, Srikanth Ronanki, Subendhu Rongali, Sravan Babu Bo-
dapati, Aram Galstyan, Azton Wells, Roy Schwartz, Eliu A Huerta, and Hao Peng.
Context length alone hurts LLM performance despite perfect retrieval. In Christos
Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, edi-
tors,Findings of the Association for Computational Linguistics: EMNLP 2025, pages
23281–23298, Suzhou, China, November 2025. Association for Computational
Linguistics. ISBN 979-8-89176-335-7. doi: 10.18653/v1/2025.findings-emnlp.1264.
URL https://aclanthology.org/2025.findings-emnlp.1264/.
[7]Raul Castro Fernandez. Data discovery is a socio -technical problem: the path from
document identification and retrieval to data ecology. InIEEE Computer Society
Data Engineering Bulletin, 2025. URL https://api.semanticscholar.org/CorpusID:
281957015. Preprint available at Semantic Scholar (CorpusID:281957015).
[8]Raul Castro Fernandez, Pranav Subramaniam, and Michael J Franklin. Data
market platforms: Trading data assets to solve data problems.Proceedings of the
VLDB Endowment, 13(11).
[9]Raul Castro Fernandez, Aaron J. Elmore, Michael J. Franklin, Sanjay Krishnan,
and Chenhao Tan. How large language models will disrupt data management.
Proc. VLDB Endow., 16(11):3302–3309, July 2023. ISSN 2150-8097. doi: 10.14778/
3611479.3611527. URL https://doi.org/10.14778/3611479.3611527.
[10] Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers,
Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg,
Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van
Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe,
Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren
Weckesser, Hameer Abbasi, Christoph Gohlke, and Travis E. Oliphant. Array
programming with numpy.Nature, 585(7825):357–362, Sep 2020. ISSN 1476-4687.

The Pneuma Project: Reifying Information Needs as Relational Schemas to Automate Discovery, Guide Preparation,
and Align Data with Intent CIDR’26, January 18-21, 2026, Chaminade, USA
doi: 10.1038/s41586-020-2649-2. URL https://doi.org/10.1038/s41586-020-2649-2.
[11] Jeffrey Heer and Ben Shneiderman. Interactive dynamics for visual analysis.Com-
munications of the ACM, 55(4):45–54, April 2012. doi: 10.1145/2133806.2133821.
[12] Nils Jahnke and Boris Otto. Data catalogs in the enterprise: Applications and
integration.Datenbank Spektrum, 23:89–96, 2023. doi: 10.1007/s13222-023-00445-
2.
[13] Carol C. Kuhlthau. Inside the search process: Information seeking from the user’s
perspective.Journal of the American Society for Information Science, 42(5):361–371,
1991. doi: 10.1002/(SICI)1097-4571(199106)42:5<361::AID-ASI6>3.0.CO;2-#.
[14] Eugenie Lai, Gerardo Vitagliano, Ziyu Zhang, Sivaprasad Sudhir, Om Chabra,
Anna Zeng, Anton A. Zabreyko, Chenning Li, Ferdinand Kossmann, Jialin Ding,
Jun Chen, Markos Markakis, Matthew Russo, Weiyang Wang, Ziniu Wu, Michael J.
Cafarella, Lei Cao, Samuel Madden, and Tim Kraska. Kramabench: A benchmark
for ai systems on data-to-insight pipelines over data lakes.ArXiv, abs/2506.06541,
2025. URL https://api.semanticscholar.org/CorpusID:279250249.
[15] Younghun Lee, Sungchul Kim, Ryan A. Rossi, Tong Yu, and Xiang Chen. Learn-
ing to reduce: Towards improving performance of large language models on
structured data, 2024. URL https://arxiv.org/abs/2407.02750.
[16] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Se-
bastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-
intensive nlp tasks. InProceedings of the 34th International Conference on Neural
Information Processing Systems, NIPS ’20, Red Hook, NY, USA, 2020. Curran
Associates Inc. ISBN 9781713829546.
[17] Liyao Li, Jiaming Tian, Hao Chen, Wentao Ye, Chao Ye, Haobo Wang, Ningtao
Wang, Xing Fu, Gang Chen, and Junbo Zhao. LongTableBench: Benchmarking
long-context table reasoning across real-world formats and domains. In Christos
Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, edi-
tors,Findings of the Association for Computational Linguistics: EMNLP 2025, pages
11927–11965, Suzhou, China, November 2025. Association for Computational
Linguistics. ISBN 979-8-89176-335-7. doi: 10.18653/v1/2025.findings-emnlp.638.
URL https://aclanthology.org/2025.findings-emnlp.638/.
[18] Lei Liu, So Hasegawa, Shailaja Keyur Sampat, Maria Xenochristou, Wei-Peng
Chen, Takashi Kato, Taisei Kakibuchi, and Tatsuya Asai. Autodw: Automatic
data wrangling leveraging large language models. InProceedings of the 39th
IEEE/ACM International Conference on Automated Software Engineering, ASE
’24, page 2041–2052, New York, NY, USA, 2024. Association for Computing
Machinery. ISBN 9798400712487. doi: 10.1145/3691620.3695267. URL https:
//doi.org/10.1145/3691620.3695267.
[19] Tianyang Liu, Fei Wang, and Muhao Chen. Rethinking tabular data understanding
with large language models. In Kevin Duh, Helena Gomez, and Steven Bethard,
editors,Proceedings of the 2024 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies (Volume
1: Long Papers), pages 450–482, Mexico City, Mexico, June 2024. Association
for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.26. URL https:
//aclanthology.org/2024.naacl-long.26/.
[20] Yuhan Liu, Michael JQ Zhang, and Eunsol Choi. User feedback in human-LLM
dialogues: A lens to understand users but noisy as a learning signal. In Christos
Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, edi-
tors,Proceedings of the 2025 Conference on Empirical Methods in Natural Language
Processing, pages 2666–2681, Suzhou, China, November 2025. Association for
Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-
main.133. URL https://aclanthology.org/2025.emnlp-main.133/.
[21] Yu. A. Malkov and D. A. Yashunin. Efficient and robust approximate nearest
neighbor search using hierarchical navigable small world graphs, 2018. URL
https://arxiv.org/abs/1603.09320.
[22] Gary Marchionini. Exploratory search: From finding to understanding.Commu-
nications of the ACM, 49(4):41–46, 2006. doi: 10.1145/1121949.1121979.
[23] Sunghyun Park, Han Li, Ameen Patel, Sidharth Mudgal, Sungjin Lee, Young-
Bum Kim, Spyros Matsoukas, and Ruhi Sarikaya. A scalable framework for
learning from implicit user feedback to improve natural language understanding
in large-scale conversational AI systems. In Marie-Francine Moens, Xuanjing
Huang, Lucia Specia, and Scott Wen-tau Yih, editors,Proceedings of the 2021
Conference on Empirical Methods in Natural Language Processing, pages 6054–
6063, Online and Punta Cana, Dominican Republic, November 2021. Association
for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.489. URL
https://aclanthology.org/2021.emnlp-main.489/.
[24] Liana Patel, Siddharth Jha, Melissa Pan, Harshit Gupta, Parth Asawa, Carlos
Guestrin, and Matei Zaharia. Semantic operators: A declarative model for rich,
ai-based data processing. 2024. URL https://api.semanticscholar.org/CorpusID:
271218837.
[25] Mark Raasveldt and Hannes Mühleisen. Duckdb: an embeddable analytical
database. InProceedings of the 2019 International Conference on Management of
Data, SIGMOD ’19, page 1981–1984, New York, NY, USA, 2019. Association for
Computing Machinery. ISBN 9781450356435. doi: 10.1145/3299869.3320212. URL
https://doi.org/10.1145/3299869.3320212.[26] Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework:
Bm25 and beyond.Found. Trends Inf. Retr., 3(4):333–389, April 2009. ISSN 1554-
0669. doi: 10.1561/1500000019. URL https://doi.org/10.1561/1500000019.
[27] Tefko Saracevic. Relevance reconsidered. In Peter Ingwersen and Niels Ole Pors,
editors,Proceedings of the 2nd International Conference on Conceptions of Library
and Information Science (CoLIS 2), pages 201–218, Copenhagen, Denmark, 1996.
Royal School of Librarianship.
[28] Murray Shanahan, Kyle McDonell, and Laria Reynolds. Role play with large
language models.Nature, 623(7987):493–498, Nov 2023. ISSN 1476-4687. doi:
10.1038/s41586-023-06647-8. URL https://doi.org/10.1038/s41586-023-06647-8.
[29] John W. Tukey.Exploratory Data Analysis. Addison -Wesley, Reading, MA, 1977.
ISBN 0-201-07616-0.
[30] Matthias Urban, Jialin Ding, David Kernert, Kapil Vaidya, and Tim Kraska. Uti-
lizing past user feedback for more accurate text-to-sql. InProceedings of the
Workshop on Human-In-the-Loop Data Analytics, HILDA ’25, New York, NY,
USA, 2025. Association for Computing Machinery. ISBN 9798400719592. doi:
10.1145/3736733.3736739. URL https://doi.org/10.1145/3736733.3736739.
[31] Zilong Wang, Hao Zhang, Chun-Liang Li, Julian Martin Eisenschlos, Vincent
Perot, Zifeng Wang, Lesly Miculicich, Yasuhisa Fujii, Jingbo Shang, Chen-Yu
Lee, and Tomas Pfister. Chain-of-table: Evolving tables in the reasoning chain
for table understanding. InThe Twelfth International Conference on Learning
Representations, 2024. URL https://openreview.net/forum?id=4L0xnS4GQM.
[32] Wes McKinney. Data Structures for Statistical Computing in Python. In Stéfan
van der Walt and Jarrod Millman, editors,Proceedings of the 9th Python in Science
Conference, pages 56 – 61, 2010. doi: 10.25080/Majora-92bf1922-00a.
[33] Ryen W. White and Resa A. Roth.Exploratory Search: Beyond the Query-Response
Paradigm. Synthesis Lectures on Information Concepts, Retrieval, and Services.
Morgan & Claypool Publishers, 2009. ISBN 978-1-59829-783-6. doi: 10.2200/
S00174ED1V01Y200901ICR003.
[34] Ryen W. White and Resa A. Roth. Exploratory search: Beyond the query-response
paradigm.Synthesis Lectures on Information Concepts, Retrieval, and Services, 1
(1):1–98, 2009. doi: 10.2200/S00174ED1V01Y200901ICR003.
[35] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In
International Conference on Learning Representations (ICLR), 2023.
[36] Amy X. Zhang, Michael Muller, and Dakuo Wang. How do data science workers
collaborate? roles, workflows, and tools.Proc. ACM Hum.-Comput. Interact., 4
(CSCW1), May 2020. doi: 10.1145/3392826. URL https://doi.org/10.1145/3392826.
[37] Shiqi Zhang, Xinbei Ma, Zouying Cao, Zhuosheng Zhang, and Hai Zhao. Plan-
over-graph: Towards parallelable llm agent schedule, 2025. URL https://arxiv.
org/abs/2502.14563.
[38] Xiaokang Zhang, Sijia Luo, Bohan Zhang, Zeyao Ma, Jing Zhang, Yang Li, Guanlin
Li, Zijun Yao, Kangli Xu, Jinchang Zhou, Daniel Zhang-Li, Jifan Yu, Shu Zhao,
Juanzi Li, and Jie Tang. TableLLM: Enabling tabular data manipulation by LLMs
in real office usage scenarios. In Wanxiang Che, Joyce Nabende, Ekaterina
Shutova, and Mohammad Taher Pilehvar, editors,Findings of the Association for
Computational Linguistics: ACL 2025, pages 10315–10344, Vienna, Austria, July
2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi:
10.18653/v1/2025.findings-acl.538. URL https://aclanthology.org/2025.findings-
acl.538/.
[39] Yunjia Zhang, Jordan Henkel, Avrilia Floratou, Joyce Cahoon, Shaleen Deep,
and Jignesh M. Patel. Reactable: Enhancing react for table question answering.
Proc. VLDB Endow., 17(8):1981–1994, April 2024. ISSN 2150-8097. doi: 10.14778/
3659437.3659452. URL https://doi.org/10.14778/3659437.3659452.
[40] Xuanhe Zhou, Zhaoyan Sun, and Guoliang Li. Db-gpt: Large language model
meets database.Data Science and Engineering, 9(1):102–111, Mar 2024. ISSN
2364-1541. doi: 10.1007/s41019-023-00235-6. URL https://doi.org/10.1007/s41019-
023-00235-6.
[41] Dongsheng Zhu, Weixian Shi, Zhengliang Shi, Zhaochun Ren, Shuaiqiang Wang,
Lingyong Yan, and Dawei Yin. Divide-then-aggregate: An efficient tool learning
method via parallel tool invocation. In Wanxiang Che, Joyce Nabende, Ekaterina
Shutova, and Mohammad Taher Pilehvar, editors,Proceedings of the 63rd Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pages 28859–28875, Vienna, Austria, July 2025. Association for Computational
Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.1401. URL
https://aclanthology.org/2025.acl-long.1401/.
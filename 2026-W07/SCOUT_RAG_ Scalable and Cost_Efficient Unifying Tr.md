# SCOUT-RAG: Scalable and Cost-Efficient Unifying Traversal for Agentic Graph-RAG over Distributed Domains

**Authors**: Longkun Li, Yuanben Zou, Jinghan Wu, Yuqing Wen, Jing Li, Hangwei Qian, Ivor Tsang

**Published**: 2026-02-09 09:00:17

**PDF URL**: [https://arxiv.org/pdf/2602.08400v1](https://arxiv.org/pdf/2602.08400v1)

## Abstract
Graph-RAG improves LLM reasoning using structured knowledge, yet conventional designs rely on a centralized knowledge graph. In distributed and access-restricted settings (e.g., hospitals or multinational organizations), retrieval must select relevant domains and appropriate traversal depth without global graph visibility or exhaustive querying. To address this challenge, we introduce \textbf{SCOUT-RAG} (\textit{\underline{S}calable and \underline{CO}st-efficient \underline{U}nifying \underline{T}raversal}), a distributed agentic Graph-RAG framework that performs progressive cross-domain retrieval guided by incremental utility goals. SCOUT-RAG employs four cooperative agents that: (i) estimate domain relevance, (ii) decide when to expand retrieval to additional domains, (iii) adapt traversal depth to avoid unnecessary graph exploration, and (iv) synthesize the high-quality answers. The framework is designed to minimize retrieval regret, defined as missing useful domain information, while controlling latency and API cost. Across multi-domain knowledge settings, SCOUT-RAG achieves performance comparable to centralized baselines, including DRIFT and exhaustive domain traversal, while substantially reducing cross-domain calls, total tokens processed, and latency.

## Full Text


<!-- PDF content starts -->

SCOUT-RAG: Scalable and Cost-Efficient Unifying Traversal for
Agentic Graph-RAG over Distributed Domains
Longkun Li1*, Yuanben Zou2,*, Jinghan Wu3,*, Yuqing Wen4,*
Jing Li5, 6†Hangwei Qian5,6, Ivor Tsang5, 6
1Yong Loo Lin School of Medicine, National University of Singapore, Singapore
2College of Design and Engineering, National University of Singapore, Singapore
3Institute of Artificial Intelligence and Robotics, Xi’an Jiaotong University, China
4Faculty of Science, National University of Singapore, Singapore
5Institute of High Performance Computing, Agency for Science, Technology and Research, Singapore
6Centre for Frontier AI Research, Agency for Science, Technology and Research, Singapore
{e1324260, e1142979, e1351400}@u.nus.edu, wujing4022@stu.xjtu.edu.cn,
{lijing, qian hangwei, ivor tsang}@a-star.edu.sg
Abstract
Graph-RAG improves LLM reasoning using structured
knowledge, yet conventional designs rely on a centralized
knowledge graph. In distributed and access-restricted settings
(e.g., hospitals or multinational organizations), retrieval must
select relevant domains and appropriate traversal depth with-
out global graph visibility or exhaustive querying. To ad-
dress this challenge, we introduceSCOUT-RAG(S calable
and CO st-efficient U nifying T raversal), a distributed agen-
tic Graph-RAG framework that performs progressive cross-
domain retrieval guided by incremental utility goals. SCOUT-
RAG employs four cooperative agents that: (i) estimate do-
main relevance, (ii) decide when to expand retrieval to ad-
ditional domains, (iii) adapt traversal depth to avoid un-
necessary graph exploration, and (iv) synthesize the high-
quality answers. The framework is designed to minimize
retrieval regret, defined as missing useful domain informa-
tion, while controlling latency and API cost. Across multi-
domain knowledge settings, SCOUT-RAG achieves perfor-
mance comparable to centralized baselines, including DRIFT
and exhaustive domain traversal, while substantially reducing
cross-domain calls, total tokens processed, and latency.
Introduction
RAG (Retrieval-Augmented Generation) (Lewis et al. 2020)
enhances large language models by retrieving and integrat-
ing external knowledge for grounded response generation.
Traditional deployments primarily rely on vector-based re-
trieval, where individually relevant text segments are re-
trieved based on similarity signals. Yet many real-world
queries demand global sense-making, synthesizing relation-
ships, dependencies, and abstractions across entities rather
than isolating local matches. To address this gap, Graph-
RAG systems (Edge et al. 2024) have emerged, enabling
multi-hop traversal, hierarchical abstraction, and structured
*Work done during an internship at A*STAR.
†Corresponding Author
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.knowledge integration. These capabilities have led ma-
jor industry platforms, including Microsoft, NebulaGraph,
AntGroup, and Neo4j (Peng et al. 2024), to adopt graph-
centric retrieval pipelines, underscoring the emerging role
of structured knowledge access in complex information en-
vironments.
However, most RAG and Graph-RAG frameworks as-
sume centralized access to a unified knowledge store. In
practice, data often resides across independent organiza-
tions that cannot share raw content due to privacy, owner-
ship, or regulatory constraints, motivating retrieval across
distributed and siloed domains rather than a single graph.
This setting has been formalized as multi-domain or fed-
erated RAG (Shojaee et al. 2025), where each domain re-
tains data control and responses must respect access bound-
aries. Further, each cross-domain request may incur latency
or monetary costs, requiring systems that minimize unneces-
sary retrieval while preserving answer fidelity. Recent work
uses supervised domain routers (Shojaee et al. 2025; Guer-
raoui et al. 2025) to filter domains, but these approaches
depend on labeled query–domain pairs and do not adapt to
evolving utility signals or cold-start deployments.
Distributed Graph-RAG features partial observability,
heterogeneous access constraints, and incremental evidence
accumulation, which collectively motivate anagenticfor-
mulation. Rather than predetermining a fixed retrieval plan,
the system must treat cross-domain traversal as a sequen-
tial decision process: evaluating intermediate signals, updat-
ing beliefs about domain relevance, and allocating retrieval
budget adaptively across local exploration and global expan-
sion. This perspective naturally accommodates privacy con-
straints, variable query structure, and shifting domain util-
ity without requiring labeled access patterns or centralized
visibility. By framing retrieval as iterative utility estimation
and constrained action selection, an agentic approach sup-
ports controlled cost–quality trade-offs and scalable reason-
ing across distributed knowledge holders.
Building on this perspective, we presentSCOUT-RAG,arXiv:2602.08400v1  [cs.AI]  9 Feb 2026

alayered agentic architecture for distributed Graph-
RAG, explicitly designed to operate under cost, privacy,
and scalability constraints. Our design is grounded in three
core insights.First, we introduce training-free domain rel-
evance estimation, avoiding supervised routing signals and
enabling deployment in cold-start and privacy-restricted set-
tings.Second, we generalize Graph-RAG to decentralized
knowledge holders, treating each domain as a partially
observable subgraph and enabling adaptive switching be-
tween local traversal and cross-domain expansion as con-
text evolves.Third, motivated by agent-based RAG sys-
tems (Singh et al. 2025) and recent agentic Graph-RAG ef-
forts (Lee et al. 2025; Luo et al. 2025), we formalize retrieval
as a sequential decision process that incrementally assesses
utility, allocates exploration budget, and adjusts traversal
depth and breadth under explicit cost constraints. Together,
these principles enable scalable, privacy-aware, and cost-
efficient cross-domain knowledge aggregation without as-
suming centralized visibility or requiring supervised domain
routing. To the best of our knowledge,SCOUT-RAG is
the first to operationalize scalable agentic Graph-RAG
across independent knowledge domains, supporting adap-
tive multi-domain traversal under explicit cost and privacy
objectives.
Overall, the main contributions of this work are as fol-
lows:
• We introduce the first cross-domain Graph-RAG setting,
where knowledge is distributed across independent data
holders and cross-domain access is gated by privacy
and cost constraints. This setting highlights the scalabil-
ity and utility challenges of decentralized sense-making
queries.
• We propose SCOUT-RAG, a layered agentic system
that conducts training-free domain relevance estima-
tion, local vs. global traversal decisions, and adaptive
depth–breadth exploration across multiple domains.
• We design a closed-loop evaluation and improvement
mechanism in which assessment agents monitor answer
sufficiency and factual coverage, guiding strategy agents
to selectively expand domains or deepen graph traversal
when needed.
• SCOUT-RAG implements multi-level safeguards, in-
cluding strict time-budget enforcement, best-answer
tracking, and coordinated parallelism, to ensure reliable
execution and prevent cascading failures in decentralized
access settings.
• Evaluations on 89 multi-domain queries across 1–40
countries show that SCOUT-RAG outperforms central-
ized local and global baselines, while nearly matching
centralized DRIFT (56 vs. 63) with over 4× fewer tokens
and lower latency, demonstrating efficient and scalable
decentralized retrieval.
Related Work
Gragh Retrieval-Augmented Generation.RAG improves
language model grounding by retrieving external evi-
dence (Lewis et al. 2020). GraphRAG further enhancescontextual reasoning by leveraging graph structure for
hierarchical retrieval and local exploration (Edge et al.
2024). Typically, users are required to specify the retrieval
mode—global or local—when using Graph-RAG systems,
in order to indicate which level of granularity is preferred
for the current query. Recent advances have explored orga-
nizing retrieved information through hierarchical summary
trees (Sarthi et al. 2024), constructing entity graphs from
document-level chunks (Zhao et al. 2025), or extracting en-
tities and relations with industrial-grade NLP libraries (Min
et al. 2025). However, these methods work under a unified
corpus or global knowledge graph, which can be comple-
mentary for the problem setup where information is siloed,
access is constrained, and retrieval must remain selective
and budget-aware. The DRIFT framework1from Microsoft
Research introduced the idea of combining global reason-
ing with targeted local inspection to improve retrieval effi-
ciency. Our proposed framework draws inspiration from this
design: SCOUT-RAG adopts DRIFT-style global–local bal-
ancing but extends it to a multi-domain setting, where re-
trieval policies must reinterpret the notions of ”global” and
”local” in order to determine the most appropriate level of
granularity for each specific query.
Cross-Domain Traversal.Beyond domain-specific rea-
soning in Graph-RAG (Xiao et al. 2025), recent practical
applications have highlighted the need to retrieve informa-
tion across multiple domains (Liu et al. 2024; Shojaee et al.
2025), which calls for more principled search strategies. Ide-
ally, domains can be accessed independently and concur-
rently, allowing overall latency to be reduced as multiple
domain searches proceed in parallel. However, in practice,
latency may be amplified due to bandwidth and network
overhead, the dominance of the slowest domain, or addi-
tional computation and synchronization delays across do-
mains. A recent automated platform for multi-domain evalu-
ation of RAG systems, OmniBench-RAG (Liang, Zhou, and
Wang 2025), has incorporated efficiency as a key dimension
in performance quantification. In addition, many systems,
such as KG-GPT (Kim et al. 2023) and StructGPT (Jiang
et al. 2023), have integrated LLM-based retrievers, implying
that exhaustive retrieval across all domains would be costly.
Regarding the efficiency problems, existing approaches rely
on labeled data to train models that identify relevant do-
mains, commonly referred to as domain routers (Shojaee
et al. 2025; Guerraoui et al. 2025). Our proposed SCOUT-
RAG leverages an agent to rank domain relevance and dy-
namically explore appropriate domains based on multiple
key signals. The agent can operate under a cold-start con-
dition and becomes increasingly reliable as more queries
are accumulated. To enhance usability, we also introduce
explicit termination conditions, such as time tolerance and
answer-quality constraints.
Agentic AI Systems.Recent advances in multi-agent
systems explore coordinated LLM roles for complex rea-
soning and tool-use workflows. AutoGen (Wu et al. 2024)
demonstrates conversational collaboration among agents,
while MetaGPT (Hong et al. 2023) formalizes structured
1https://microsoft.github.io/graphrag/query/drift search/

role assignments inspired by software engineering teams.
ReAct (Yao et al. 2022) couples reasoning traces with action
execution for controllability. Several pioneering works have
explored the integration of agentic AI into Graph-RAG sys-
tems. KG-R1 (Song et al. 2025) employs a single agent that
interacts with the knowledge graph as its environment, learn-
ing to retrieve relevant information step by step and integrat-
ing it into its reasoning and text generation process. Agent-
G (Lee et al. 2025) introduces feedback loops that integrate
an agent to improve retrieval quality, where an auxiliary
critic module rejects unsatisfactory answers. We build on
these insights but shift focus to governing retrieval decisions
for multi-sources scenarios: SCOUT-RAG operationalizes
four specialized agents for domain assessment, evidence ex-
traction, synthesis, and iterative refinement, each explicitly
grounded in cross-domain information access needs.
Methodology
Problem Formulation
We consider retrieval-augmented generation overMdis-
tributed knowledge domains. Each domaini∈ {1, . . . , M}
maintains a private knowledge graphG i= (V i,Ei,Xi)con-
sisting of entities, relations, and associated text or struc-
tured attributes. Raw data cannot be shared across domains
due to privacy and ownership constraints. Given a natural-
language queryq, the system aims to synthesize an answer
Agrounded in evidence retrieved from multiple domains,
without centralizing the underlying graphs and adhering to
a prescribed cost budget.
Letπ idenote the retrieval policy for domaini, specifying
whether and how to perform retrieval, with associated cost
ci(e.g., time or computational resources). The objective is
to produce an answer that maximizes the answer quality:
max
AQual(A|q,{π i}M
i=1)
s.t.A= Φ 
{fπi 
q,Gi
}M
i=1
,MX
i=1ci≤ Cmax,(1)
wheref πi(·)is a generator producing a local answerA iun-
der policyπ i,Φ(·)denotes the aggregation function synthe-
sizing the final answer across domains, andQual(·)quan-
tifies the grounding fidelity and relevance of the generated
answer with respect to retrieved evidence.
In practice, the retrieval policy{π i}is not executed in a
single pass. Instead, SCOUT-RAG operates throughpro-
gressive policy refinement: (i) initial estimation of domain
relevance, (ii) domain-scoped partial retrieval and answer
seeding, and (iii) iterative improvement via local depth or
breadth expansions guided by answer-quality feedback and
remaining budget. This sequential decision process incre-
mentally allocates retrieval resources while honoringC max,
terminating when no further utility is gained or the budget is
exhausted.
This formulation captures the fundamental characteristics
of distributed Graph-RAG: (i)partial observability, since
the global graphSM
i=1Giis never exposed; (ii)adaptive
cross-domain reasoning, requiring decisions on when to re-
main local vs. expand to new domains; and (iii)cost-aware
……HIGHMODERATEPOTENTIALUNRELATEDDomainsDomain Relevance AssessmentAgentQuery𝒟!Stage1
GlobalRetrieval
LocalRetrieval
A⁽⁰⁾GenerationAgent𝐴"#$%&
SynthesisAgentGraph
Relevance
IterativeRefinement Loop
𝐴(()
𝐴((*+)
Stage2Stage3Figure 1: Overview of the proposedSCOUT-RAGframe-
work. The framework operates in three stages: (i) domain
relevance estimation, (ii) domain-aware retrieval (local vs.
global), and (iii) iterative answer refinement via synthe-
sis and generation agents. These agentic components col-
lectively support adaptive, cross-domain reasoning without
centralized graph access.
retrieval refinement, where utility estimates guide incre-
mental evidence gathering rather than exhaustive querying.
Building upon this formulation, we design a decentralized
solution that operates the above objectives through coordi-
nated retrieval and synthesis. Specifically, we present the
SCOUT-RAGframework, which operates in three coordi-
nated stages and leverages four specialized agents. We first
identify relevant knowledge domains, then perform adaptive
local-to-global graph retrieval, and finally refine answers
through iterative quality-guided evidence expansion. An il-
lustrative overview of the framework is provided in Figure 1.
Stage I. Domain Relevance Assessment
The first stage determines which distributed domains are
likely to contain knowledge relevant to the query. Accurate
relevance estimation is crucial: it suppresses unnecessary re-
trieval and prioritizes domains that meaningfully contribute

to grounding, thereby improving cost-efficiency and cross-
domain reasoning quality.
Domain Relevance Assessment Agent (DRAA).Given
distributed knowledge domains{D i}M
i=1, we employ
aDRAAto classify each domain into one of four
relevance tiers:HIGH,MODERATE,POTENTIAL, and
IRRELEVANT.
For each domainD iand queryq, three complementary
information signals are computed:
•Semantic similarity:ssim
i= Sim(q,D i), representing
embedding-based cosine similarity between the query
and domain-level representations.
•Knowledge richness:srich
i=|R i|/max j|Rj|, quanti-
fying relative data abundance by the normalized report
count.
•Historical performance:shist
i=1
|Hi|P
h∈H iQ(h),
capturing average answer quality over past queries as-
sociated with the domain.
These features jointly characterize domain–query com-
patibility from semantic, evidential, and behavioral perspec-
tives. The DRAA combines them with query context to pro-
duce a relevance prediction:
(τi, ri) =f DRAA 
q,Di, ssim
i, srich
i, shist
i
,(2)
whereτ idenotes the assigned relevance tier andr iis a
natural-language rationale to support transparency and in-
spection.
Stage II. Domain-Scoped Seeding
After Stage I identifies candidate domains, Stage II initial-
izes retrieval and constructs the first round answer. This
stage balancescoverageandprecision: high-confidence do-
mains provide broad contextual grounding, while moderate-
confidence domains contribute focused evidence. Two
agents collaborate in this phase: thePartial Answer Gen-
eration Agent (PAGA)retrieves domain-specific evidence,
and theOverall Answer Synthesis Agent (OASA)synthe-
sizes these signals into an initial cross-domain response.
Partial Answer Generation Agent (PAGA).The PAGA
executes domain-adaptive retrieval, guided by the relevance
tiersτ ifrom Stage I. Retrieval is performed in parallel, but
with differentiated granularity:
•HIGH-relevance domains:Global retrieval, capturing
broad semantic context via community-level summaries.
•MODERATE-relevance domains:Local retrieval, retriev-
ing fine-grained entity-centric or relation-level evidence.
•POTENTIALdomains: Deferred, held in reserve for pos-
sible activation in Stage III.
•IRRELEVANTdomains: Excluded to avoid unnecessary
cost and noise.
Thus, for each domainD iwith graphG i, the partial answer
is:
Ai=

fg
PAGA(q,G i),ifτ i=HIGH,
fl
PAGA(q,G i),ifτ i=MODERATE,
∅,otherwise.(3)
𝐴(")
QualityAssessmentAgentExecutionAgentIterativeSynthesisAgentCompletenessCoverage𝑉(")𝐶(")
POTENTIALHIGHDepthBreadthStop
𝐴("$%)𝐴$(")
“European diets rely heavily on dairy and bread … Asian diets emphasize rice and vegetables …”
StrategySelection
“Further insights show Mediterranean countries prioritize olive oil and fish …”“Mediterranean diets favor plant-based diversity … East Asian meals focus on balance… Western diets…”Figure 2: Illustration of Stage III, where the system eval-
uates and improves answers through selective retrieval and
synthesis.Example query:”What are the main differences
in dietary habits among countries such as Japan, France, and
the United States?”.
Here,fg
PAGA andfl
PAGA denote domain-specific global and
local graph retrieval operators, respectively. This tiered pol-
icy ensures that retrieval cost concentrates only on domains
with the highest expected payoff.
Overall Answer Synthesis Agent (OASA).The OASA
merges the partial answers into a coherent initial response:
A(0)= Φ OASA M[
i=1Ai!
,(4)
whereΦ OASA(·)performs cross-domain answer fusion with
explicit source attribution and light consistency checking.
The resultingA(0)serves as a groundedseedanswer: it es-
tablishes an initial scaffold, which Stage III subsequently ex-
pands and refines as additional evidence is acquired.
Stage III. Iterative Cross-Domain Refinement
Stage III incrementally improves the initial seed answer
from Stage II through targeted, quality-guided refinement.
The system evaluates the current answer, identifies missing
content or uncertainty, and selectively activates retrieval to
close knowledge gaps. This iterative loop continues until
quality converges or the time budget is exhausted (Figure 2).
Answer Quality Assessment Agent (AQAA).The
AQAAassesses the current answerA(t)using structured
LLM-based evaluation prompts, producing a tuple of
metrics and follow-up queries:
(C(t), V(t), G(t),F(t)) =f AQAA(A(t)),(5)

whereC(t)measurescompleteness(coverage of core con-
ceptual elements),V(t)measuresbreadth(coverage across
relevant subtopics),G(t)denotes unresolved knowledge
gaps, andF(t)=qf
kis a set of targeted follow-up queries.
This structured evaluation provides actionable signals on
what to refine next and where.
Strategy Selection.A decision policy selects refinement
modeα(t)based on the current quality state and remaining
time budgetT raccording to our empirical evaluations:
α(t)=

Depth,ifC(t)<0.75∧T r>15s
Breadth,ifV(t)<0.70∧T r>10s
Hybrid,ifC(t)<0.75∧V(t)<0.70∧T r>20s
Stop,otherwise
(6)
Refinement terminates when:
• Quality threshold met:Q(t)≥0.85
• Time budget depleted:T r<5s
• Improvement stagnation:∆Q(t)=Q(t)−Q(t−1)< ϵ
whereQ(t)=1
2(C(t)+V(t)).
Once a strategy is selected, refinement proceeds along
one of three modes: a depth-focused pass that queries only
HIGH-relevance domains using the targeted follow-up ques-
tionsF(t)to close core completeness gaps (similar in spirit
to focused re-retrieval approaches such as DRIFT (Edge
et al. 2024)); a breadth-expansion pass that activates pre-
viously dormantPOTENTIALdomains to capture periph-
eral or emerging subtopics; or a hybrid mode that runs both
paths in parallel when the answer is simultaneously incom-
plete and narrow. If none of the strategies are triggered, e.g.,
quality saturation or insufficient time, the system halts re-
finement.
Partial Answer Generation Agent (PAGA).ThePAGA
executes the selected refinement mode by conducting tar-
geted local retrieval:
ˆA(t)=

S
i∈D highfl
PAGA(F(t),Gi), α(t)=Depth,
S
j∈D potfl
PAGA(q,G j), α(t)=Breadth,
both above, α(t)=Hybrid.(7)
Depth queries refine critical details in high-relevance do-
mains, while breadth queries activate latent domains to cap-
ture overlooked context.
Overall Answer Synthesis Agent (OASA).TheOASA
integrates newly retrieved evidence with the current answer:
A(t+1)= Φ OASA
A(t),ˆA(t)
.(8)
To ensure stability, the system tracks all intermediate candi-
dates{A(0), A(1), . . . , A(t)}and returns the highest-quality
answerA∗= arg max iQual(A(i))upon termination, en-
suring resilience against late-stage drift or noisy retrieval.
For clarity, we present a summary of SCOUT-RAG in Algo-
rithm 1.Algorithm 1: Scalable Cost-Efficient Cross-Domain Traver-
sal for Distributed Agentic Graph-RAG
Require:Queryq, domains{D i}M
i=1, time budgetT max
Ensure:Final answerA∗
Stage 1: Domain Relevance Assessment
1:forD i∈ Ddo
2:Compute domain relevance via DRAA:
HIGH/MODERATE/POTENTIAL/IRRELEVANT
3:end for
Stage 2: Domain-Scoped Seeding
4:Perform global retrieval onHIGHdomains
5:Perform local retrieval onMODERATEdomains
6:PAGA produces partial answers
7:OASA synthesizes seed answerA(0)
Stage 3: Iterative Cross-Domain Refinement
8:whileT r>0do
9:Evaluate current answerA(t)via AQAA to obtain
10:(C(t), V(t), G(t),F(t))
11:Compute qualityQ(t)=1
2(C(t)+V(t))
12:Select refinement strategy via Eq. (6):
13:ifα(t)=STOPthen
14:break
15:end if
16:Conduct retrieval via PAGA according toα(t)
17:Update answer via OASA fusion to obtainA(t+1)
18:Re-evaluate via AQAA
19:ifquality improvesthenupdate best answerA∗
20:end if
21:end while
22:returnA∗
Experiments
Experimental Setup
DatasetTo assess the scalability of our framework in han-
dling a large number of domains, we employ Wikipedia
articles from 45 countries as simulated domain datasets.
Within each domain, a graph-based knowledge base is con-
structed, comprising 9–77 community reports. We also gen-
erate 100 queries with varying levels of domain coverage to
comprehensively evaluate our framework, of which 89 are
effectively answered by all methods. The queries are cat-
egorized into five levels: single-domain (20 queries, each
involving one country), small multi-domain (20 queries,
each involving five countries), medium multi-domain (20
queries, each involving ten countries), large multi-domain
(20 queries, each involving twenty countries), and very large
multi-domain (9 queries, each involving forty countries).
BaselineWe compare against four GraphRAG baselines
across two deployment paradigms. Under the centralized
setting (all data from 45 countries consolidated into a sin-
gle unified domain), we evaluate:
•GraphRAG Local: Local search for detailed entity-level
information retrieval from specific community reports.
•GraphRAG Global : Global search across hierarchical
structures for comprehensive overviews via community

Table 1: Average Performance Across Evaluation Dimensions. The best scores for centralized and decentralized methods are
highlighted in bold.
Category MethodQuality Cost
Comp.(↑)Div.(↑)Emp.(↑)Dir.(↑)Overall(↑) Time(s)(↓)Token(↓)
CentralizedGraphRAG Local 65 55 35 58 53 34.40 11,223
GraphRAG Global 60 50 30 55 49 45.89 640,574
GraphRAG DRIFT-c 72 70 45 63 63 231.85 693,731
DecentralizedGraphRAG DRIFT-dec 90 88 75 88 85 414.88 879,911
SCOUT-RAG (Ours) 65 60 40 58 56 75.32 159,169
summaries.
•GraphRAG DRIFT-c : One global search followed by two
refinement rounds (three follow-up local searches per
round) to iteratively improve answer quality.
Under the decentralized setting (data distributed across 45
country-specific domains), we include:
•GraphRAG DRIFT-dec : Applies DRIFT independently in
each domain—one global search plus two refinement
rounds per domain, with final synthesis across all 45
domain-level answers.
MectricFollowing interactive agent evaluation method-
ologies (Zhou et al. 2023), we assess four dimensions
adapted from GraphRAG (Edge et al. 2024): Comprehen-
siveness (coverage of relevant aspects), Diversity (breadth of
perspectives and sources), Empowerment (utility for follow-
up exploration and actionable insights), and Directness (con-
ciseness and clarity). Each dimension is scored 0–100 by
GPT-4o as an LLM-as-judge using structured prompts.
All experiments were conducted on a system equipped
with an NVIDIA RTX 3070 GPU with 9 GB VRAM (driver
version 575.57.08). The compute environment consisted of
16 AMD EPYC 7B12 vCPUs, 31 GB memory.
Main Results
Answer Quality ComparisonTable 1 summarizes the av-
erage performance across 89 queries spanning five com-
plexity levels, evaluated by GPT-4o under two contrasting
paradigms that comprise three centralized methods and two
decentralized methods.
In decentralized mode, GraphRAG DRIFT-dec achieves the
highest overall score (85) among all methods through
exhaustive per-domain retrieval. Among centralized ap-
proaches, GraphRAG DRIFT-c reaches 63, outperforming
GraphRAG Local (53) and GraphRAG Global (49) by more
than 10 points. The proposed SCOUT-RAG system ex-
plores a third approach that combines decentralized deploy-
ment with intelligent and quality-controlled orchestration. It
achieves an overall score of 56, which is 7 points lower than
GraphRAG DRIFT-c but 7 points higher than GraphRAG Global
and 3 points higher than GraphRAG Local. This near-parity is
notable given that SCOUT-RAG operates under distributed
constraints without centralized coordination. The small per-
formance gap indicates that quality-controlled orchestration
effectively preserves answer quality within decentralized ar-
chitectures.
100 150 200 250 300
Time (s)020406080100Score (0-100)
(a) SCOUT-RAG: Detailed Metrics Over Time
Comprehensiveness
Diversity
Empowerment
Directness
Overall
0 100 200 300 400
Time (s)020406080100Overall Score (0-100)
(b) Overall Score Comparison Across Methods
SCOUT-RAG
GraphRAGDRIFT-c
GraphRAGDRIFT-dec
300s limitFigure 3: Time-performance monitoring and comparison.
Across individual evaluation dimensions, SCOUT-RAG
demonstrates particularly strong performance in diversity
(60), surpassing centralized baselines by 5–10 points. This
improvement results from hierarchical domain categoriza-
tion (HIGH/MODERATE/POTENTIAL) and quality-driven
activation, which encourage systematic exploration of multi-
ple perspectives. Although GraphRAG DRIFT-dec achieves the
highest overall performance, it incurs an extreme computa-
tional cost of 414.88 seconds and 879,911 tokens, which is
more than 5.5 times the resource consumption of SCOUT-
RAG, as analyzed in the following section.
Cost-Efficiency AnalysisA critical advantage of
SCOUT-RAG lies in resource efficiency. Our system
consumes 159,169 tokens and completes queries in 75.32
seconds on average, representing 81.9% token reduction and
81.8% speedup compared to GraphRAG DRIFT-dec (879,911
tokens, 414.88s). Even against centralized GraphRAG Global
(640,574 tokens), we achieve 75.2% fewer tokens while
maintaining comparable quality (56 vs. 49).
The cost and performance trade-off is particularly no-
table. GraphRAG DRIFT-dec attains the highest overall score
(85) through exhaustive per-domain retrieval but requires
414.88s and 879,911 tokens, which is approximately 5.5
times SCOUT-RAG’s execution time and token usage. Cen-
tralized GraphRAG DRIFT-c (63) also incurs significant over-
head, consuming 231.85s and 693,731 tokens, which are
3.1 times and 4.4 times higher, respectively. Although
GraphRAG Localexecutes faster (34.40s), it does so at the ex-
pense of diversity (55 vs. 60) and overall quality.
These efficiency gains enable practical deployment in
resource-constrained scenarios. By achieving near-baseline
performance with>80% cost reduction compared to decen-
tralized alternatives, SCOUT-RAG demonstrates the viabil-
ity of quality-controlled orchestration for distributed knowl-
edge retrieval at scale.

 Analyze the impact of Italy's 'Made in Italy' certification on the competitiveness of small and
medium-sized enterprises (SMEs) in the fashion sector,  focusing on the interplay between governmental
policies, international trade agreements, and cultural heritage preservation. Consider how these factors
influence market access, branding strategies, and consumer perception in both domestic and international
markets.
High Relevance : Italy (0.5 39), Slovenia (0.384), Malta  (0.342), Croa tia (0.329), Po rtugal (0.325)
Moderate Relevance : Slovakia (0.320 ), Turkey (0.305),  Spain (0.289), Gree ce (0.285), Austr ia (0.294)
Potential Relevance: L atvia (0.288) , Lithuania  (0.288), Estonia (0. 283), Poland (0. 284), Ireland (0.272)
High rele vance domains  include Italy due to the direct mention , and Slovenia, Malta, Croatia, and Po rtugal
due to their cultural and fashion industry ti es. Moderate relevance  includes countries with significant
trade relations or fashion ind ustries , such as Slovakia, Turkey, and Sp ain. Potenti al relevance includes
countries with some cultural or  economic ties  to the fashion ind ustry or international trade , like Latvia
and Lithuania. Unrelated domain s are those with little to no direct connection to the  fashion industry  or the
specific q uery focus.
User Query
Stage 1  Assessment
Stage 2  Initial ResponseBaseline Answer: Based on information from various domains, Italy's 'Made in Italy' certification has a
significant impact on the competitiveness ...
Stage 3  Refinement Iteration 1 (Quality: 0.675)
Identified Gaps:
Missing specific information about how governmental policies specifically support SMEs in the fashion
sector
Lacks details on specific international trade agreements that impact the 'Made in Italy' certification
No comparison with similar certifications or branding strategies in other countries
... Governmental policies supporting the certification promote authenticity, quality standards, and market
access, aligning with international trade agreements. ...
Iteration 2 (Quality: 0.725)
Identified Gaps:
Missing specific information about governmental policies supporting 'Made in Italy' certification
Lacks details on the impact of international trade agreements on the certification
No comparison with similar certifications or branding strategies in other countries like Slovenia and
Malta
... While the impact of international trade agreements on the 'Made in Italy' certification is not specified in
the current information, understanding these agreements is crucial for assessing the certification's
recognition and protection for SMEs in the fashion sector. Additionally, a comparison with similar
certifications or branding strategies in other countries, such as Slovenia and Malta, can provide valuable
insights into the competitiveness of SMEs in the global fashion market. ...... Governmental policies supporting the certification promote authenticity, quality standards, and market
access, aligning with international trade agreements. ...
Iteration 3 (Quality: 0.675)
Identified Gaps:
Missing specific information about the impact of international trade agreements on 'Made in Italy'
certification
Lacks details on specific governmental policies supporting SMEs in Italy's fashion sector
No comparison with similar certifications or branding strategies in other countries like Slovenia and
Malta
Insufficient information on Croatia and Portugal's certification systems and their impact on SME
competitiveness
Lacks exploration of how governmental policies and cultural heritage preservation efforts influence
branding strategies and consumer perception in Croatia and Portugal  
... In Croatia, the certification system for fashion products is not directly comparable to Italy's 'Made in
Italy' certification in terms of impact on SME competitiveness. Specific details on governmental policies
supporting SMEs in the fashion sector in Croatia are lacking. Additionally, the influence of Croatia's cultural
heritage preservation efforts on consumer perception of fashion products is not addressed in the current
context.
Similarly, information on Portugal's certification system for fashion products, governmental policies
supporting SMEs in the fashion sector, and the influence of Portugal's cultural heritage preservation efforts
on branding strategies and consumer perception is missing. ...
Iteration 2 (Highest Quality: 0.725)
Based on information from various domains, Italy's 'Made in Italy' certification has a significant impact on
the competitiveness ...
Final ResponseFigure 4: Case study on Italy’s ”Made in Italy” certification.
A Closer Look at the Time Budget
Figure 3 (left) demonstrates how SCOUT-RAG’s answer
quality evolves over time. All evaluation dimensions—
Comprehensiveness, Diversity, Empowerment, andDirectness—show rapid improvement in the first 120
seconds, followed by a performance plateau around 180s,
indicating diminishing returns beyond this point. Fig-
ure 3 (right) compares three methods under different time
constraints. SCOUT-RAG converges quickly to stable
quality within the 300s budget. GraphRAG DRIFT-c achieves
similar performance with comparable time efficiency.
GraphRAG DRIFT-dec , though requiring more time, ultimately
reaches the highest overall score, demonstrating a trade-off
between response speed and answer quality.
Case Study
We conducted a case study of SCOUT-RAG using a complex
query. The overall workflow is presented in Figure 4.
In Stage I, the domain relevance module identified 10
relevant domains within 8.28 seconds. Italy ranked first
(0.539), followed by Slovenia, Malta, Croatia, and Portugal,
reflecting strong cultural and institutional similarity. This
outcome demonstrates the system’s ability to perform accu-
rate geocultural alignment, ensuring that subsequent reason-
ing is grounded in comparable certification frameworks.
Stage II produced an initial domain-scoped synthesis in
53.26 seconds. The system captured the main analytical di-
mensions concerning competitiveness, policy instruments,
trade agreements, and heritage branding, but lacked concrete
examples and cross-country references. AQAA rated breath
at 0.65 and completeness at 0.70, prompting refinement to
enhance specificity and comparative depth.
Stage III executed three iterative refinement rounds over
41.12 seconds. The first round expanded retrieval to Slove-
nia, Malta, Croatia, and Portugal, introducing policy and
branding details and improving coverage to 0.75. The sec-
ond iteration focused on Italy, Malta, and Slovenia, achiev-
ing the highest overall quality (0.725) through contextual
and evidential balance. The third round employed Breadth
to explorePOTENTIALdomains, but added minor noise and
reduced final quality to 0.675. The best-track mechanism re-
tained Iteration 2, preventing late-stage degradation.
Overall, this case demonstrates that SCOUT-RAG effec-
tively integrates policy, trade, and cultural heritage infor-
mation across domains. Through adaptive retrieval, iterative
synthesis, and best-track optimization, the system maintains
analytical precision and interpretive coherence in addressing
complex, interdisciplinary socio-economic questions such
as ”Made in Italy.”
Conclusion
This work presented SCOUT-RAG, a scalable and cost-
efficient agentic framework for retrieval-augmented gener-
ation across distributed knowledge domains. Unlike central-
ized Graph-RAG systems, SCOUT-RAG enables progres-
sive cross-domain retrieval under explicit cost and privacy
constraints. Empirical evaluations demonstrate that SCOUT-
RAG achieves performance comparable to the centralized
DRIFT, while maintaining a modest gap compared to the
near-exhaustive Graph-RAG with DRIFT. Cost-efficiency
analysis shows that SCOUT-RAG achieves significant re-
ductions in both token consumption and execution time,
demonstrating its potential for real-world deployments.

Acknowledgments
This research is supported by the National Research Founda-
tion, Singapore under its National Large Language Models
Funding Initiative (AISG Award No: AISG-NMLP-2024-
003). Any opinions, findings and conclusions or recommen-
dations expressed in this material are those of the author(s)
and do not reflect the views of National Research Founda-
tion, Singapore. This research is supported by A*STAR Ca-
reer Development Fund<Project No. C243512010>.
References
Edge, D.; Trinh, H.; Cheng, N.; Bradley, J.; Chao, A.;
Mody, A.; Truitt, S.; Metropolitansky, D.; Ness, R. O.; and
Larson, J. 2024. From local to global: A graph rag ap-
proach to query-focused summarization.arXiv preprint
arXiv:2404.16130.
Guerraoui, R.; Kermarrec, A.-M.; Petrescu, D.; Pires, R.;
Randl, M.; and de V os, M. 2025. Efficient federated search
for retrieval-augmented generation. InProceedings of the
5th Workshop on Machine Learning and Systems, 74–81.
Hong, S.; Zhuge, M.; Chen, J.; Zheng, X.; Cheng, Y .; Wang,
J.; Zhang, C.; Wang, Z.; Yau, S. K. S.; Lin, Z.; et al. 2023.
MetaGPT: Meta programming for a multi-agent collabora-
tive framework. InThe Twelfth International Conference on
Learning Representations.
Jiang, J.; Zhou, K.; Dong, Z.; Ye, K.; Zhao, W. X.; and Wen,
J.-R. 2023. Structgpt: A general framework for large lan-
guage model to reason over structured data.arXiv preprint
arXiv:2305.09645.
Kim, J.; Kwon, Y .; Jo, Y .; and Choi, E. 2023. Kg-gpt: A
general framework for reasoning on knowledge graphs using
large language models.arXiv preprint arXiv:2310.11220.
Lee, M.-C.; Zhu, Q.; Mavromatis, C.; Han, Z.; Adeshina,
S.; Ioannidis, V . N.; Rangwala, H.; and Faloutsos, C. 2025.
Agent-G: An Agentic Framework for Graph Retrieval Aug-
mented Generation.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,
T.; et al. 2020. Retrieval-augmented generation for
knowledge-intensive nlp tasks.Advances in neural infor-
mation processing systems, 33: 9459–9474.
Liang, J.; Zhou, S.; and Wang, K. 2025. OmniBench-
RAG: A Multi-Domain Evaluation Platform for
Retrieval-Augmented Generation Tools.arXiv preprint
arXiv:2508.05650.
Liu, Z.; Yang, K.; Xie, Q.; de Kock, C.; Ananiadou, S.; and
Hovy, E. 2024. RAEmoLLM: Retrieval augmented LLMs
for cross-domain misinformation detection using in-context
learning based on emotional information.arXiv preprint
arXiv:2406.11093.
Luo, H.; E, H.; Chen, G.; Lin, Q.; Guo, Y .; Xu, F.; Kuang, Z.;
Song, M.; Wu, X.; Zhu, Y .; and Tuan, L. A. 2025. Graph-R1:
Towards Agentic GraphRAG Framework via End-to-end
Reinforcement Learning.arXiv preprint arXiv:2507.21892.Min, C.; Mathew, R.; Pan, J.; Bansal, S.; Keshavarzi, A.; and
Kannan, A. V . 2025. Efficient Knowledge Graph Construc-
tion and Retrieval from Unstructured Text for Large-Scale
RAG Systems.arXiv preprint arXiv:2507.03226.
Peng, B.; Zhu, Y .; Liu, Y .; Bo, X.; Shi, H.; Hong, C.; Zhang,
Y .; and Tang, S. 2024. Graph retrieval-augmented genera-
tion: A survey.arXiv preprint arXiv:2408.08921.
Sarthi, P.; Abdullah, S.; Tuli, A.; Khanna, S.; Goldie, A.;
and Manning, C. D. 2024. Raptor: Recursive abstractive
processing for tree-organized retrieval. InThe Twelfth In-
ternational Conference on Learning Representations.
Shojaee, P.; Harsha, S. S.; Luo, D.; Maharaj, A.; Yu, T.;
and Li, Y . 2025. Federated retrieval augmented genera-
tion for multi-product question answering.arXiv preprint
arXiv:2501.14998.
Singh, A.; Ehtesham, A.; Kumar, S.; and Khoei, T. T. 2025.
Agentic retrieval-augmented generation: A survey on agen-
tic rag.arXiv preprint arXiv:2501.09136.
Song, J.; Wang, S.; Shun, J.; and Zhu, Y . 2025. Efficient
and Transferable Agentic Knowledge Graph RAG via Rein-
forcement Learning.arXiv preprint arXiv:2509.26383.
Wu, Q.; Bansal, G.; Zhang, J.; Wu, Y .; Li, B.; Zhu, E.; Jiang,
L.; Zhang, X.; Zhang, S.; Liu, J.; et al. 2024. Autogen: En-
abling next-gen LLM applications via multi-agent conversa-
tions. InFirst Conference on Language Modeling.
Xiao, Y .; Dong, J.; Zhou, C.; Dong, S.; Zhang, Q.-w.; Yin,
D.; Sun, X.; and Huang, X. 2025. GraphRAG-Bench:
Challenging Domain-Specific Reasoning for Evaluating
Graph Retrieval-Augmented Generation.arXiv preprint
arXiv:2506.02404.
Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan,
K. R.; and Cao, Y . 2022. React: Synergizing reasoning and
acting in language models. InThe eleventh international
conference on learning representations.
Zhao, Y .; Zhu, J.; Guo, Y .; He, K.; and Li, X. 2025. Eˆ
2GraphRAG: Streamlining Graph-based RAG for High Effi-
ciency and Effectiveness.arXiv preprint arXiv:2505.24226.
Zhou, X.; Zhu, H.; Mathur, L.; Zhang, R.; Yu, H.; Qi, Z.;
Morency, L.-P.; Bisk, Y .; Fried, D.; Neubig, G.; et al. 2023.
Sotopia: Interactive evaluation for social intelligence in lan-
guage agents.arXiv preprint arXiv:2310.11667.
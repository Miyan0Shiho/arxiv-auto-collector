# Synergizing RAG and Reasoning: A Systematic Review

**Authors**: Yunfan Gao, Yun Xiong, Yijie Zhong, Yuxi Bi, Ming Xue, Haofen Wang

**Published**: 2025-04-22 13:55:13

**PDF URL**: [http://arxiv.org/pdf/2504.15909v2](http://arxiv.org/pdf/2504.15909v2)

## Abstract
Recent breakthroughs in large language models (LLMs), particularly in
reasoning capabilities, have propelled Retrieval-Augmented Generation (RAG) to
unprecedented levels. By synergizing retrieval mechanisms with advanced
reasoning, LLMs can now tackle increasingly complex problems. This paper
presents a systematic review of the collaborative interplay between RAG and
reasoning, clearly defining "reasoning" within the RAG context. It construct a
comprehensive taxonomy encompassing multi-dimensional collaborative objectives,
representative paradigms, and technical implementations, and analyze the
bidirectional synergy methods. Additionally, we critically evaluate current
limitations in RAG assessment, including the absence of intermediate
supervision for multi-step reasoning and practical challenges related to
cost-risk trade-offs. To bridge theory and practice, we provide practical
guidelines tailored to diverse real-world applications. Finally, we identify
promising research directions, such as graph-based knowledge integration,
hybrid model collaboration, and RL-driven optimization. Overall, this work
presents a theoretical framework and practical foundation to advance RAG
systems in academia and industry, fostering the next generation of RAG
solutions.

## Full Text


<!-- PDF content starts -->

Synergizing RAG and Reasoning: A Systematic Review
Yunfan Gao
Shanghai Research Institute for
Intelligent Autonomous Systems,
Tongji University
China
gaoyunfan1602@gmail.comYun Xiong
Shanghai Key Laboratory of Data
Science, School of Computer Science,
Fudan University
China
yunx@fudan.edu.cnYijie Zhong
College of Design and Innovation,
Tongji University
China
dun.haski@gmail.com
Yuxi Bi
College of Design and Innovation,
Tongji University
China
yuxibi@gmail.comMing Xue
Percena AI
China
mxue@percena.coHaofen Wang∗
College of Design and Innovation,
Tongji University
China
carter.whfcarter@gmail.com
Abstract
Recent breakthroughs in large language models (LLMs), par-
ticularly in reasoning capabilities, have propelled Retrieval-
Augmented Generation (RAG) to unprecedented levels. By
synergizing retrieval mechanisms with advanced reasoning,
LLMs can now tackle increasingly complex problems. This
paper presents a systematic review of the collaborative inter-
play between RAG and reasoning, clearly defining "reason-
ing" within the RAG context. It construct a comprehensive
taxonomy encompassing multi-dimensional collaborative
objectives, representative paradigms, and technical imple-
mentations, and analyze the bidirectional synergy methods.
Additionally, we critically evaluate current limitations in
RAG assessment, including the absence of intermediate su-
pervision for multi-step reasoning and practical challenges
related to cost-risk trade-offs. To bridge theory and practice,
we provide practical guidelines tailored to diverse real-world
applications. Finally, we identify promising research direc-
tions, such as graph-based knowledge integration, hybrid
model collaboration, and RL-driven optimization. Overall,
this work presents a theoretical framework and practical
foundation to advance RAG systems in academia and indus-
try, fostering the next generation of RAG solutions.
1 Introduction
Recent breakthroughs in Large Language Models (LLMs) like
OpenAI O1 [ 39] and DeepSeek-R1 [ 25] have shifted the par-
adigm from "pre-training scaling" to "test-time scaling" [ 63].
Unlike traditional language models that improve via corpus
accumulation during pre-training, these models enhance per-
formance in complex tasks—such as mathematical derivation
and code generation [ 29]—through post-training innovations
during the inference phase (e.g., Long-CoT thinking [ 8]). This
shift has led to the emergence of "Large Reasoning Models"
(LRMs) [99] with advanced internal reasoning abilities.
∗Corresponding AuthorThese advancements have not only boosted basic model
capabilities but also opened new avenues for application tech-
nologies like Retrieval-Augmented Generation (RAG) [ 21].
Serving as a key link between language models and exter-
nal knowledge, RAG overcomes traditional LLMs’ limits in
knowledge freshness, domain specificity, and factual accu-
racy by retrieving real-time non-parametric information and
integrating it into the context. This enhances information
processing and reduces hallucination risks in knowledge-
intensive tasks.
Technological evolution is advancing RAG architectures
through innovations like query rewriting [ 61], re-ranking [ 1],
and hybrid retrieval [ 88], creating an Advanced RAG para-
digm focused on pre-retrieval optimization and post-retrieval
refinement. Modular RAG [ 22] further breaks down these
systems into component-based, service-oriented architec-
tures, using orchestration to tackle practical challenges.
Despite improvements in query intent recognition and
knowledge use, challenges of RAG remain in demanding
tasks like deep research and complex decision-making. Key
issues include: 1) difficulty capturing intent from ambiguous
queries; 2) poor logical coherence in multi-hop reasoning; 3)
efficiency limits of traditional retrieval in open domains; and
4) degraded generation quality from noisy retrieved data.
Models like DeepSeek-R1, with strong reasoning capabili-
ties, inspire new directions for RAG systems. As shown in
Figure 1, recent research explores integrating formal reason-
ing frameworks with knowledge retrieval. This approach
optimizes retrieval through logic-driven query reformula-
tion and uses reasoning to analyze and validate retrieved
knowledge, creating cognitive synergy between retrieval and
generation. This paradigm aims to overcome conventional
limitations, enabling intelligent systems with rigorous logic
and reliable knowledge use. From a trend perspective, an in-
creasing number of methods combine reasoning and retrieval
abilities through reinforcement learning (RL), marking a
new direction in the LRM era. Meanwhile, prompt-based ap-
proaches continue to rapidly evolve, with researchers aimingarXiv:2504.15909v2  [cs.IR]  24 Apr 2025

Conference’17, July 2017, Washington, DC, USA Gao et al.
Figure 1. Timeline of studies on RAG-reasoning synergy. From a technical perspective, the approaches can be categorized into
Prompt-Based, Tuning-Based, and RL-Based methods. A notable trend is the increasing use of Reinforcement Learning to
enhance RAG systems, particularly following the prosperity of test-time scaling. Meanwhile, Prompt-Based and Tuning-Based
methods continue to evolve in parallel, demonstrating that there are multiple pathways to integrating reasoning capabilities
into RAG systems.
to achieve results through workflow design while keeping
model parameters frozen. Notably, sole reliance on tuning
methods is steadily decreasing, suggesting limited improve-
ments from additional fine-tuning at this stage.
Traditional RAG is limited by its unidirectional flow (re-
trieval→generation). Integrating reasoning capabilities
grants the system greater autonomy, unlocking new pos-
sibilities. As shown in Figure 2, this integration is poised to
drive major breakthroughs, enabling practical use in complex
real-world scenarios.
1) From Ambiguous Semantic Matching to Logic-
Driven Targeted Retrieval. Traditional RAG relies on se-
mantic similarity for retrieval; however, it is sensitive to
phrasing variations. Advanced reasoning allows deep logical
analysis of queries (e.g., causal links, conditional constraints)
to dynamically refine retrieval strategies [ 24]. For example,
to answer "How to reduce postoperative infection risks in
diabetes patients?", the system prioritizes retrieving "blood
glucose control thresholds" and "antibiotic usage guidelines"
over simply matching "diabetes postoperative care". This
approach supports multi-hop retrieval by breaking down
complex queries into sequential sub-queries while preserv-
ing cross-document coherence through reasoning chains.2) From Simple Information Aggregation to Logi-
cally Coherent Context Construction. Current RAG sys-
tems input all retrieved document chunks into context di-
rectly, often causing fragmented or contradictory informa-
tion that confuses LLMs. Reasoning-enhanced systems in-
tegrate evidence chains by logically verifying and inferring
causality in retrieved content, filtering conflicts and forming
coherent explanations [ 100]. They also use dynamic knowl-
edge completion to detect missing logical links, prompting
iterative retrieval or inference to fill gaps [51].
3) From Simple and Single-Turn QA to Systemic De-
cision Support. Traditional RAG performs well in factual
QA [ 65] but struggles with multi-step and complex decision-
making. Reasoning-integrated systems produce structured
reasoning output, enhancing multi-objective optimization
to balance retrieval breadth and solution feasibility under
various constraints. For example, multiple constraints under
different conditions in engineering construction plans [ 54],
and the formulation of diagnosis and treatment plans for
various diseases in the medical field [105].
4) From Indiscriminate Retrieval to Intelligent Re-
source Allocation. Traditional RAG retrieves documents
for all queries, regardless of complexity. Reasoning-enhanced
systems use on-demand retrieval, handling simple queries

Synergizing RAG and Reasoning: A Systematic Review Conference’17, July 2017, Washington, DC, USA
Figure 2. Advantages of Combining RAG with Reasoning
with direct generation and complex ones with multi-round
retrieval to reduce latency [ 20]. Dynamic retrieval pruning
uses pre-reasoning predictions to target key information,
minimizing unnecessary document and graph traversal [ 41].
5) From Passive Knowledge Tool to Proactive Cog-
nitive Assistant. Advancing beyond reactive knowledge
retrieval, reasoning-enhanced systems can proactively serve
users by asking clarifying questions and anticipating im-
plicit needs. This shift enables human-like assistants that
integrate memory, reasoning, and decision-making, prov-
ing especially valuable for complex tasks such as deep re-
search [ 43], business analytics [ 50], personal assistant [ 107]
and urban planning [85].
However, the synergistic pathway between RAG and rea-
soning requires more than simply replacing conventional
generative LLMs with LRM modules. It necessitates deep
integration of technological evolution insights from LRM -
achieved through reconstructing knowledge retrieval mecha-
nisms and strengthening reasoning-generation collaborative
linkages - to enable system-level enhancement of cognitive
capabilities within the RAG architecture.
Therefore, this paper aims to address the pivotal and
forward-looking research question of "how RAG systems
can synergize with reasoning capabilities". We systematically
review current studies after 2024 while establishing explicit
definitions for reasoning within RAG contexts. Building on
this foundation, we provide an in-depth taxonomy and anal-
ysis of the objectives, typical patterns, and implementations
underlying RAG-reasoning integration, clarifying key tech-
nological trajectories and critical breakthroughs.
As RAG technology enters its next developmental phase,
downstream task complexity has escalated significantly -particularly evident in emerging challenges like Deep Re-
search [ 106]. These advanced applications not only demand
enhanced reasoning capacities but also drive RAG’s expan-
sion into multimodal, cross-domain, and dynamic environ-
ments. However, while the integration of reasoning capa-
bilities demonstrably improves complex task performance,
existing research frequently overlooks associated computa-
tional overheads and potential risks. Through systematic
examination of these operational constraints and analysis
of industry applications, we propose practical guidelines for
multiple real-world scenarios with diverse requirements.
Finally, we outline future research directions grounded
in current technological evolution, including: 1) RAG-graph
architecture integration, 2) coordinated multimodal reason-
ing frameworks, 3) hybrid model collaboration, and 4) RL
optimization specifically designed for RAG systems. This
work establishes both theoretical foundations and practical
roadmaps for subsequent research in this evolving field.
The contributions of this paper can be summarized as
follows:
•Pioneering Review . This work represents the first
comprehensive survey focusing on the integration
of RAG with reasoning, offering novel insights and
forward-looking guidance for advancing this emerg-
ing research frontier.
•Systematic Taxonomy . We present a multi-dimensional
framework to systematically examine the objectives,
paradigms, and methodologies for combining RAG
with reasoning capabilities, establishing clear classifi-
cation criteria across technical dimensions.
•Practical Guidance . Beyond theoretical exploration,
we critically discuss the additional cost and potential

Conference’17, July 2017, Washington, DC, USA Gao et al.
risks associated with the introduction of reasoning,
accompanied by an actionable Practical Guide for real-
world scenarios.
•Open Resource Platform1Through the OpenRAG
platform, we provide a rich, multi-dimensional review
of related work, which allows readers to quickly search
and compare different methods.
2 Overview
This chapter establishes a conceptual framework for the
paper along two key dimensions. First, it formally defines
"reasoning" and distinguishes it from "inference." Second,
it organizes a taxonomy of synergy mechanisms between
"RAG and Reasoning." To construct a clear cognitive pathway,
we address three progressive research questions:
•Why synergize RAG and reasoning?
•What are their typical collaboration paradigms?
•How can this integration be realized?
2.1 Definition
The definition of reasoning in modern AI systems remains
an evolving construct, particularly within the context of
LRMs exemplified by DeepSeek R1 and OpenAI O1. Here,
under the scope of LLMs, we formalize reasoning as a struc-
tured, multi-step process that dynamically decomposes
complex problems, generates intermediate hypotheses,
and iteratively refines solutions through logical and
evidence-based transformations. Mathematically, let a
reasoning process Rbe defined as a tuple ⟨K𝑝,K𝑟,S𝑡,Φ⟩,
whereK𝑝denotes parametric knowledge embeddings, K𝑟
represents retrieved contextual knowledge, S𝑡={𝑠0,𝑠1,...,𝑠 𝑛}
constitutes the evolving state sequence with 𝑠0as the initial
query and𝑠𝑛as the final response, and Φ:S𝑖×K 𝑝×K 𝑟→
S𝑖+1defines the state transition function.
The reasoning process exhibits three defining character-
istics. First, it is inherently multi-step , systematically de-
composing complex problems into intermediate cognitive
states (e.g., sub-question generation or temporary conclu-
sions) rather than pursuing direct input-output mapping.
Second, it generates novel knowledge or facts — synthe-
sizing implicit relationships, deriving latent constraints, or
reformulating problems in ways not explicitly present in
the initial input or parametric memory (e.g., transforming
"Is A greater than B?" into comparative subquestions about
A and B’s attributes). Crucially, these representations are
not merely retrieved but dynamically constructed through
the reasoning trajectory. Third, the process is teleological
— its architecture and termination conditions are explicitly
optimized for complex problem resolution, where complex-
ity is measured by the necessity of state transitions or the
insufficiency of direct retrieval from either parametric ( K𝑝)
1https://openrag.notion.site/open-rag-base?pvs=4or external (K𝑟) knowledge sources. This stands in stark con-
trast to atomic inference, which lacks such deliberate state
construction and goal-aware iteration.
The distinction between reasoning and inference mani-
fests most saliently in their computational signatures. While
inferenceIconstitutes a single-step conditional probabil-
ity computation 𝑃(𝑦|𝑥)=Î𝑇
𝑡=1𝑃(𝑦𝑡|𝑥,𝑦<𝑡), reasoningR
implements a meta-process coordinating multiple inference
calls through explicit state management R(𝑥)=Φ1◦Φ2◦
···◦Φ𝑛(𝑥). This multi-phase architecture enables systematic
error correction through backtracking mechanisms and dy-
namic retrieval refinement — features fundamentally absent
in conventional inference pipelines. The operational bound-
ary emerges when state transitions involve explicit symbolic
manipulation (equation restructuring in mathematical rea-
soning) or knowledge graph traversal (temporal reasoning
over retrieved events), distinguishing true reasoning from
mere multi-step inference.
2.2 Taxonomy
Integrating RAG with reasoning marks a paradigm shift in
tackling complex knowledge-intensive tasks. This work de-
velops a hierarchical taxonomy (Figure 3) based on three key
questions: why reasoning is needed with RAG ( Purpose ),
how they structurally interact ( Paradigm ), and what meth-
ods enable effective integration ( Implementation ). This
framework guides readers through the technical innovations
in later chapters, providing a clear conceptual path without
premature technical details, and highlighting the field’s evo-
lutionary logic, avoiding delving prematurely into specific
technical details.
2.2.1 Synergy Purpose. Integrating reasoning with RAG
addresses the limitations of traditional RAG systems, which
struggle with multi-step logic, contextual adaptation, and
implicit knowledge synthesis due to reliance on superficial
semantic matching and fixed knowledge limits. Adding rea-
soning enables dynamic retrieval planning, logical verifica-
tion of evidence, and insight generation beyond retrieved
data through abductive or counterfactual reasoning. At the
same time, the introduction of external knowledge retrieval
also helps alleviate reasoning interruptions caused by the
knowledge limitations of LRM and reduces the likelihood of
hallucinations. This integration occurs in two main ways:
Reasoning-Augmented Retrieval, where inference drives
context-aware information gathering; Retrieval-Augmented
Reasoning , where external knowledge supports and ex-
pands the model’s deductive abilities.
2.2.2 Synergy Paradigms. Building upon the above ne-
cessity, our taxonomy categorizes RAG+Reasoning systems
along the axis of procedural dynamism. Pre-defined work-
flows employ fixed templates that systematically alternate
between retrieval and reasoning phases, with intervention
points predetermined at pre-retrieval reasoning (e.g., query

Synergizing RAG and Reasoning: A Systematic Review Conference’17, July 2017, Washington, DC, USA
The Synergy of RAG and ReasoningPurpose §3Reasoning-Augmented Retrieval
(Better Retrieval) §3.1ARM [ 7]; AdaptiveRAG [ 41]; FinSearch [ 50]; LevelRAG [ 103]; OmniThink [ 94]; Plan-
RAG [ 48]; SmartRAG [ 20]; UAR [ 14]; O1-Embedeer [ 101]; ITER-RETGEN [ 68]; HiRAG [ 102];
MetaRAG [ 108]; MMOA-RAG [ 12]; ReZero [ 16]; DeepResearcher [ 106]; ReaRAG [ 49]; FRAG [ 23];
PORAG [ 73]; Insight-RAG [ 66]; ChainRAG [ 110]; FG-RAG [ 32]; MCTS-RAG [ 36]; REAPER [ 46];
DeepNote [84]
Retrieval-Augmented Reasoning
(Better Reasoning) §3.2ActiveRAG [ 100]; AgenticReasoning [ 92]; CoRAG [ 83]; CR-planner [ 52]; DeepRAG [ 24]; Deep-
solution [ 54]; KBQA-O1 [ 58]; OpenRAG [ 38]; PIKE [ 82]; R1-Seacher [ 72]; RAG-Gym [ 96];
ReARTeR [ 75]; ReSearch [ 6]; MedRAG [ 105]; RARE [ 77];RARE [90]; RetroRAG [ 95]; KG-
RAR [93]; RetrievalPRM [109]; MRD-RAG [11]; Search-O1 [51]; StePO-Rec [5]
Paradigms §4Pre-defined Workflow § 4.1Reasoning in
Pre-RetrievalLeReT [ 34]; O1-Embedder [ 101]; planRAG [ 48]; UAR [ 14]; MMOA-RAG [ 12];
MedRAG [ 105]; FRAG [ 23]; Insight-RAG [ 66]; ChainRAG [ 110]; Retrieval-
PRM [109]; REAPER [46]; AdaptiveRAG [41]
Reasoning in
Post-RetrievalActiveRAG [100]; ARM [7]; MRD-RAG [11]; FG-RAG [32]; ToG2 [60]
Hybrid ReasoningIR-COT [ 78]; FinSearch [ 50]; ITER-RETGEN [ 68]; LevelRAG [ 103]; Hi-
RAG [102]; MetaRAG [108]; RetroRAG [95]; KG-RAR [93]; DeepNote [84]
Dynamic Workflow §4.2Proactivity-Driven
ReasoningAgenticReasoning [ 92]; DeepRAG [ 24]; CoRAG [ 83]; Co-STORM [ 43];
PIKE [82]; Search-O1 [51]; R1-Searcher [72]
Reflection-Driven
ReasoningFlare [45]; OpenRAG [38]; WriteHere [98]; ReaRAG [49]; Self-RAG [3]
Feedback-Driven
ReasoningSmartRAG [ 20]; CR-Planner [ 52]; MCTS-KBQA [ 97]; DeepSolution [ 54]; RAG-
Gym [ 96]; ReZero [ 16]; ReSearch [ 6]; RARE [ 77]; DeepResearcher [ 106];
PORAG [73]; MCTS-RAG [36]; KBQA-O1 [58]
Implementation
§5Resoning Method §5.1LLM/CoTActiveRAG [ 100]; AdaptiveRAG [ 41]; O1-Embedder [ 101]; DeepNote [ 84]; Hi-
RAG [ 102]; MetaRAG [ 108]; MMOA-RAG [ 12]; RetroRAG [ 95]; PORAG [ 73];
KG-RAR [93]; MRD-RAG [11]; PlanRAG [48]
Special Token
PredictionSelf-RAG [ 3]; SmartRAG [ 20]; OpenRAG [ 38]; R1-Searcher [ 72]; ReZero [ 16];
ReSearch [6]; ReaRAG [49]; Search-O1 [51]
Search-Driven
ReasoningOminiThink [ 94]; DeepRAG [ 24]; CoRAG [ 83]; DeepSolution [ 54];
ReARTeR [ 75]; KBQA-O1 [ 58]; WriteHere [ 98]; RARE [ 77]; MCTS-KBQA [ 97];
StePO-Rec [5]
Reasoning
on GraphFinSearch [ 50]; ToG2 [ 60]; LighRAG [ 27]; FRAG [ 23]; FG-RAG [ 32];
MedRAG [105]
External Solver ARM [7]
Optimization §5.2Prompt-BasedCo-STORM [ 43]; Agentic Reasoning [ 92]; FinSearch [ 50]; PlanRAG [ 48];
HiRAG [ 102]; MetaRAG [ 108]; MedRAG [ 105]; RARE [ 77]; ReaRAG [ 49];
RetroRAG [ 95]; FRAG [ 23]; KG-RAR [ 93]; MRD-RAG [ 11]; FG-RAG [ 32];
MCTS-RAG [36]; DeepSolution [54]; StePO-Rec [5]
Tuning-BasedKBQA-Q1 [ 58]; O1-Embedder [ 101]; DeepRAG [ 24]; CoRAG [ 83]; MCTS-
KBQA [97]; RetrievalPRM [109]; REAPER [46]; RARE [90]; UAR [14]
RL-BasedPIKE [ 82]; LeReT [ 34]; RAG-Gym [ 96]; ReARTeR [ 75]; SmartRAG [ 20];
CR-Planner [ 52]; DeepRetrieval [ 42]; DeepNote [ 84]; MMOA-RAG [ 12];
ReZero [16]; ReSearch [6]; DeepResearcher [106]; PORAG [73]; R1-Search [72]
Figure 3. A structured taxonomy of synthesizing RAG and Reasoning.
decomposition), post-retrieval reasoning (e.g., evidence syn-
thesis), or hybrid stages . While offering operational trans-
parency, these architectures exhibit limited adaptability to
emergent task complexities. In contrast, dynamic work-
flows implement state-contingent reasoning processes where
retrieval actions are conditionally triggered through contin-
uous system introspection. This paradigm further branches
into Proactivity-Driven strategies (self-initiated knowledge
requests), Reflection-driven mechanisms (error-corrective re-
trieval based on intermediate result analysis), and Feedback-
driven approaches (environmental reward signals or external
model evaluations). The progression from static to dynamic
architectures reflects the field’s maturation toward human-
like contextual adaptation in open-world problem-solving.2.2.3 Synergy Implementation. Operationalizing these
synergies requires innovations across reasoning and retrieval
strategies. Foundational reasoning architectures span LLM-
Based like COT, search-based hypothesis generation (tree
search, Monte Carlo methods), symbolic solver integra-
tion, and graph-structured multi-hop inference. These ca-
pabilities are further enhanced through three principal aug-
mentation strategies: prompt-based techniques that utilize
natural language templates and special token (e.g., <Plan>,
<Verify>) to steer model behavior, tuning-based methods
that inject domain-specific knowledge or distill reasoning
capability, and RL-based frameworks that optimize retrieval-
reasoning policies through outcome reward models (ORM) or
process reward models (PRM). The alignment between these

Conference’17, July 2017, Washington, DC, USA Gao et al.
methodologies and the proposed taxonomy is critical—static
workflows predominantly rely on predictable prompt-guided
reasoning chains, whereas dynamic systems increasingly in-
tegrate search-based exploration or solver-augmented strate-
gies to navigate evolving state spaces.
Overall, this tripartite taxonomy—motivational drivers,
architectural paradigms, and implementation methodolo-
gies—establishes a unified lens for analyzing RAG+Reasoning
systems. Subsequent chapters will elaborate on each stratum,
progressively revealing how these conceptual distinctions
translate into technical innovations that push the boundaries
of machine intelligence.
3 The purpose of the synergy
The integration of RAG and reasoning marks a crucial ad-
vancement in enhancing LLMs’ problem-solving abilities.
Their true potential lies not in isolated use but in their syn-
ergy, which overcomes key limitations in retrieval and rea-
soning. This section explains the main motivations for com-
bining RAG with reasoning, emphasizing two primary bene-
fits: (1) enhancing retrieval accuracy and flexibility through
reasoning, and (2) reinforcing complex reasoning by using
context-rich retrieved knowledge. Figure 4 illustrates these
collaborative aims and the limitations they address.
The first key benefit is Reasoning-Augmented Retrieval ,
where reasoning improves the retrieval process. Traditional
RAG systems struggle with query formulation, relevance
assessment, and iterative refinement—tasks needing logical
and contextual analysis. Reasoning enables adaptive retrieval
through dynamic query expansion, ambiguity resolution,
and multi-hop evidence aggregation, overcoming the lim-
its of keyword- or embedding-based methods and aligning
retrieval with the task’s reasoning demands.
The second benefit is Retrieval-Augmented Reason-
ing, where external knowledge supplements the limitations
of purely parametric LLM reasoning. Even advanced mod-
els face hallucination, knowledge gaps, and compositional
challenges alone. Retrieval grounds reasoning in up-to-date,
domain-specific, or rare information absent from model weights,
crucial for explainability, multi-step deduction, and integrat-
ing diverse sources.
Together, combining RAG and reasoning fills fundamental
gaps in both techniques. By enhancing retrieval via reasoning
and strengthening reasoning through retrieval, it broadens
LLMs’ capacity to address complex real-world problems.
3.1 Reasoning-Augmented Retrieval
Reasoning-Augmented Retrieval (RAR) represents a signif-
icant advancement in information retrieval by integrating
multi-step reasoning to dynamically enhance retrieval qual-
ity. Unlike traditional methods that depend on static semanticmatching, RAR creates a cognitive feedback loop mimick-
ing human iterative reasoning, surpassing the limitations of
simple "query-document" interactions.
RAR’s effectiveness stems from several key features. It of-
ten uses on-demand retrieval, where reasoning—evaluating
intent clarity, knowledge state, and temporal factors—guides
adaptive search initiation, reducing redundancies present
in fixed triggers (e.g., UAR’s classifier [ 14]). It improves se-
mantic alignment by inferring implicit query logic such as
business rules or entity relationships to generate precise re-
trieval requests aligned with data schemas (e.g., PlanRAG’s
plan-retrieval loops [ 48]). RAR also applies multi-step it-
erative refinement, using intermediate reasoning outputs
(e.g., chain-of-thought, partial answers [ 78]) to recursively
reformulate queries in a closed-loop system essential for re-
solving multi-hop dependencies [ 68]. Furthermore, it adapts
to specific domains by tailoring retrieval to vertical contexts
(e.g., financial or medical) and balances efficiency and pre-
cision through lightweight reasoning strategies (e.g., Adap-
tiveRAG’s complexity-based selection [41]).
Traditional retrieval systems, effective for simple queries,
struggle with complex information needs due to rigid designs
favoring static matching over dynamic reasoning, limiting
their adaptability to changing contexts and diverse data. RAR
primarily addresses five core challenges inherent in these
conventional methods.
3.1.1 Semantic Disparities Between Queries and Doc-
uments. A key challenge lies in the mismatch between user
queries and documents—whether due to differing expression
styles (professional jargon vs. casual language) or implicit
contextual gaps—making direct semantic matching unreli-
able. Importantly, high similarity does not guarantee true
relevance, as documents may share keywords or surface
features without addressing the underlying intent or logic
of the query. Retrieval models must therefore understand
deeper semantics beyond superficial similarity.Domain adap-
tation further complicates this issue. To overcome these gaps,
approaches such as reasoning-augmented embeddings (O1-
Embedder [ 101] enriches queries with inferred “thinking”
text), feedback-driven rewriting (SmartRAG [ 20] dynami-
cally refines queries based on retrieved results), and pre-
planning (PlanRAG [48] extracts business rules to generate
SQL queries aligned with database schemas) help better cap-
ture domain-specific semantics and ensure relevance beyond
mere similarity.
3.1.2 Inflexible Intent Disambiguation . Traditional
RAG methods rely on fixed embedding similarity strate-
gies , which fail to dynamically interpret the implicit in-
tent behind complex queries (e.g., multi-hop reasoning or
domain-specific requirements). User queries often exhibit
semantic complexity that far exceeds their surface text—for
instance, a request to "optimize supply chain costs" may
require correlating disparate database fields not explicitly

Synergizing RAG and Reasoning: A Systematic Review Conference’17, July 2017, Washington, DC, USA
Figure 4. The purpose of the synergy between RAG and reasoning
mentioned. Static retrieval methods lack the adaptability
to capture such dynamically evolving information needs. A
critical limitation lies in intent dynamicity : as contextual
understanding expands, traditional systems generate fixed
retrieval results based solely on the initial query. Further-
more, semantic representation limitations of dense retrieval
models (e.g., BERT-based models) hinder their ability to en-
code intricate semantic relationships (e.g., irony, metaphors),
leading to misaligned results. Current approaches attempt
to mitigate these issues through multi-step intent decompo-
sition (e.g., LevelRAG’s high-level searcher breaks complex
queries into multi-hop sub-queries [ 103]) and dynamic query
reformulation (e.g., LeReT’s reinforcement learning gener-
ates diversified query candidates [ 34]), iteratively refining
retrieval strategies to align with document content.
3.1.3 Inefficient Coordination of Multi-Source Het-
erogeneous Data. Retrieval from diverse sources—text, ta-
bles, graphs, web, and APIs—often produces fragmented re-
sults due to a lack of global reasoning. The key challenge is
modal heterogeneity: different retrieval techniques (denseretrieval for text, SQL for tables, GQL for graphs) operate
independently without unified coordination. For example,
experiments show standard RAG methods (like dense re-
trieval with query decomposition) yield only 32.7% perfect
recall and 40.9% F1 on the OTT-QA dataset. These outcomes
reveal the limitations of traditional approaches in aligning
textual queries with structured tables—such as failing to
link concepts like "K-12 student free rates" in text to re-
lated "education expenditure" columns when not explicitly
mentioned. Additionally, disconnected entity matching (e.g.,
relating "company revenue" in text to financial tables) wors-
ens inefficiencies, as conventional methods depend on se-
mantic similarity and overlook domain-specific relationships
and exact-value matches. Advanced techniques—such as
reasoning-driven alignment (ARM’s N-gram constraints for
cross-modal entity decoding [ 7]) and unified semantic spaces
(LevelRAG’s shared multi-modal representations [ 103])—enable
more effective, integrated retrieval.
3.1.4 Incompleteness and Incoherence in Complex
Retrieval Tasks. Single-step retrieval systems fall short in

Conference’17, July 2017, Washington, DC, USA Gao et al.
complex multi-hop reasoning tasks, such as deducing entity
chains or conducting decision analysis. Traditional static re-
trieval conflicts with multi-step cognitive needs, resulting in
three main issues: 1) Path dependency, where later retrievals
rely on information from earlier steps (e.g., finding "the most
populous county in California" before its education policies),
but conventional systems lack state management; 2) Error
propagation,early retrieval errors cause mistakes in interme-
diate results, which then affect the next round of retrieval; 3)
Semantic inflexibility of fixed queries, which cannot adapt to
dynamic concepts like entity aliases or relational predicates.
Advanced methods address these flaws through integrated
strategies. PlanRAG uses iterative "plan-retrospect-replan"
cycles to trigger sub-queries when gaps arise. Reinforcement
learning in LeReT improves query generation via reward-
driven path selection. Likewise, ITER-RETGEN rebuilds follow-
up queries using intermediate answers (e.g., "award recipi-
ent’s height") to resolve multi-hop dependencies.
3.1.5 Trade-offs Between Retrieval Efficiency and Pre-
cision. Complex scenarios face a tension between exhaus-
tive retrieval, which is computationally costly, and restricted
retrieval, which risks information loss. Expanding retrieval
blindly inflates costs (e.g., LLM API calls) without ensuring
relevance. Simple queries suffer from unnecessary multi-
step retrieval, wasting resources, while complex queries face
quality risks if retrieval is too limited. Adaptive approaches
like complexity-aware routing (Adaptive-RAG’s lightweight
classifier allocates retrieval budgets [ 41]) and cost-sensitive
training (SmartRAG’s reinforcement learning balances qual-
ity and steps [20]) dynamically manage this trade-off.
In summary, Reasoning-Augmented Retrieval overcomes
traditional RAG’s limitations in dynamic triggering, seman-
tic alignment, multi-hop support, domain adaptation, and
efficiency trade-offs by deeply integrating reasoning into
the retrieval process. Its key innovation is a bidirectional
enhancement between reasoning and retrieval—reasoning
refines retrieval strategies, while retrieval supports itera-
tive reasoning—jointly boosting accuracy and efficiency in
complex information tasks.
3.2 Retrieval-Augmented Reasoning
Retrieval-Augmented Reasoning (ReAR) combines external
knowledge retrieval with inherent model reasoning to over-
come failures from knowledge gaps or logical discontinu-
ities in complex tasks. Unlike traditional RAG methods that
retrieve information once, ReAR uses an iterative, context-
sensitive retrieval that continuously provides relevant data
to support multi-step reasoning. This approach is crucial
for tasks needing strict logic, such as mathematical proofs,
where intermediate steps require specific theorems or lem-
mas. By making retrieval an adaptive, ongoing process rather
than a one-time step, ReAR strengthens each reasoning stagewith accurate, current information, improving the overall
inference’s reliability and robustness.
ReAR’s core feature is dynamic knowledge supplementa-
tion, generating retrieval queries in real-time based on the
evolving reasoning context. This overcomes the limits of
single-round retrieval by enabling knowledge refinement
at each step, as seen in process supervision frameworks
like RAG-Gym [ 96]. ReAR also improves reasoning paths
using methods like search space compression—for example,
MCTS-guided heuristics in KBQA—and structured feedback
from diverse sources like knowledge graphs [ 97]. These tech-
niques maintain logical consistency while reducing irrele-
vant or conflicting information. Importantly, ReAR adapts
well across domains, supporting precise knowledge retrieval
and tool use for specialized tasks such as industrial problem-
solving in PIKE [82] or scientific reasoning [106].
By integrating retrieval as an active part of the reason-
ing loop, ReAR addresses LLMs’ temporal and depth con-
straints, ensuring adherence to domain-specific and time-
sensitive requirements. This close coupling turns external
knowledge into an on-demand resource, creating a closed-
loop system that enhances the model’s ability to handle
complex, knowledge-intensive problems. Specifically, ReAR
seeks to address the following limitations and challenges:
3.2.1 Knowledge Gap in Multi-step Reasoning. In long-
range reasoning, missing intermediate knowledge often breaks
logical chains, especially in industrial and scientific con-
texts requiring multi-source data integration (e.g., text, ta-
bles, time-series). Static retrieval methods worsen this by
not adapting to the reasoning process’s changing needs.
ReAR techniques address this with chained retrieval, as in
CoRAG [ 83], which breaks multi-hop questions into sequen-
tial sub-queries (e.g., retrieving "event causes" then their
"impacts"), systematically linking knowledge. Reasoning-
state-aware retrieval, used in FLARE [ 45], predicts future
information needs by generating interim prompts (e.g., "the
next step requires discussion of ..."), enabling dynamic query
construction that preserves coherence. Together, these ap-
proaches resolve the conflict between discrete retrieval and
continuous reasoning.
3.2.2 Reasoning Discontinuity Caused by Domain
Knowledge Boundaries. Reasoning discontinuity arises
from LLMs’ limited knowledge, struggling with specialized
domains (e.g., semiconductor design in PIKE [ 82]) and real-
time data (e.g., medical parameters in Agentic Reasoning [ 92]).
End-to-end models often produce factual errors, while tradi-
tional RAG methods fail to retrieve deep professional knowl-
edge due to coarse retrieval, especially with complex data
like tables,charts and images.
ReAR addresses this with two complementary solutions:
knowledge atomization and structural organization, as in
PIKE’s decomposition of documents into fine-grained units
and multi-layer knowledge graphs for semantic and logical

Synergizing RAG and Reasoning: A Systematic Review Conference’17, July 2017, Washington, DC, USA
retrieval; and dynamic tool integration, as in Agentic Rea-
soning’s real-time data acquisition via code execution and
API calls to compute critical indicators (e.g., medical FiO2).
These innovations overcome the challenges of specialized
knowledge depth and timely information relevance that limit
conventional methods.
3.2.3 Search Space Explosion and Local Optima Traps.
The main challenge in multi-step reasoning is the exponen-
tial growth of the search space, where methods like Chain-of-
Thought (CoT) often yield suboptimal or inconsistent results
due to unconstrained hypotheses. Traditional approaches
like CoT and Tree-of-Thought (ToT) lack external knowl-
edge constraints, causing invalid assumptions, while purely
symbolic reasoning falls short in open-domain tasks. To ad-
dress this, two strategies are used: knowledge base-anchored
heuristic search (KBQA-O1 [ 58]), which limits reasoning
actions to subgraphs in knowledge graphs, and a retrieval-
verification mechanism (Search-o1 [ 51]) that prunes unsup-
ported reasoning paths using evidence from the knowledge
base. Together, these reduce the search space and preserve
reasoning coherence.
3.2.4 Dynamic Knowledge Requirements in Multi-
Step Reasoning. Complex multi-step reasoning tasks face
the challenge of continuously changing knowledge require-
ments. This is evident in cases like multi-hop reasoning
and engineering planning, where each stage generates new
sub-problems (e.g., moving from "architectural design" to
"material cost estimation"). Static knowledge bases or one-
time retrieval methods cannot meet this evolving demand.
This manifests in two ways: initial knowledge may miss later
needs, causing gaps; and fixed knowledge sets may include
irrelevant information, reducing reasoning accuracy. To ad-
dress this, new retrieval-augmented reasoning approaches
introduce dynamic solutions: process supervision (e.g., re-
ward models in RAG-Gym [ 96]) detects knowledge gaps in
real time, atomic decision-making (e.g., step decomposition
in DeepRAG [ 24]) triggers retrieval as needed, and tree-like
expansions (e.g., multi-path retrieval in DeepSolution [ 54])
enable parallel exploration. By integrating knowledge re-
trieval within reasoning, these methods let the system iden-
tify, supplement, and verify knowledge dynamically—much
like a human expert—greatly enhancing the reliability and
completeness of complex reasoning.
3.2.5 Insufficient Depth and Breadth of Reasoning.
This issue is prominent in expert tasks like medical diag-
nosis, legal analysis, and research report generation. LLMs’
static knowledge often fails to capture the evolving scope
of domain knowledge, resulting in shallow reasoning that
misses multi-level, cross-domain connections. For example,
when assessing "Company A is affected by economic re-
cession," traditional methods rely on superficial statisticalpatterns and cannot systematically follow the deeper log-
ical chain from "Company A →industry supply chain →
macroeconomic policy →international political landscape,"
leading to reasoning that lacks causal depth.
To overcome this, recent advances use structured, retrieval-
enhanced frameworks. ToG2.0 [ 60] models Knowledge Graph
relational paths as retrieval guidance vectors, enabling tar-
geted queries along entity paths, surpassing the limits of
keyword-based retrieval. This approach complements CR-
Planner’s [ 52] iterative expansion, which triggers retrieval
of specialized knowledge (e.g., textbook proofs of algorithm
complexity) at critical reasoning points, ensuring accurate do-
main knowledge integration via multi-round validation. Ad-
dressing cross-domain knowledge linkage, CO-STORM [ 43]
employs a multi-agent system whose host module gener-
ates cross-modal retrieval commands by analyzing potential
semantics in uncited documents.
4 Patterns of synergy
Section 3 detailed the need and motivation for integrating
RAG with reasoning. Building on this, this section presents
two core implementation patterns for RAG-reasoning syn-
ergy (Figure 5): (1) the Pre-defined Workflow , which uses
logical architectures with preset rules for coordination, and
(2)Dynamic Workflow , which relies on context-aware,
adaptive coordination via real-time decision engines. These
patterns illustrate current frameworks combining knowl-
edge retrieval and multi-step reasoning from deterministic
and flexible perspectives.
4.1 Pre-defined workflow
Pre-defined workflow is a multi-step reasoning approach
with a fixed architecture and sequential execution, emphasiz-
ing process clarity and operational determinism. It consists
of predefined iterative stages, each with strict input-output
rules and no dynamic changes based on intermediate results.
This modular design ensures controllability and structured
reasoning for complex tasks. All steps are executed regard-
less of intermediate outcomes, guaranteeing repeatability
and stability while avoiding uncertainties from dynamic de-
cisions. Although it sacrifices adaptability, this approach
offers procedural predictability and is well-suited for scenar-
ios demanding clear reasoning paths, albeit with possible
computational redundancy due to lack of real-time adjust-
ments.
Mathematically, the pre-defined RAG workflow can be
formalized as a deterministic multi-step operational chain.
Given an input query 𝑄and a predefined sequence of 𝑁
reasoning steps and the final decision output 𝐷, the complete
workflow is expressed as:
𝐷=𝑓𝑁◦···◦𝑓2◦𝑓1(𝑄) (1)

Conference’17, July 2017, Washington, DC, USA Gao et al.
Figure 5. Patterns of Synergy between RAG and Reasoning
where each 𝑓𝑖∈{Ψ,𝑅,Γ}denotes strictly defined func-
tions for reasoning ( Ψ), retrieval ( 𝑅), or decision-making
(Γ), with◦representing function composition. This formula-
tion adheres to the fixed mapping sequence 𝑄↦→Ψ(𝑄)↦→
𝑅(Ψ(𝑄))↦→Γ(𝑅(Ψ(𝑄))), exhibiting Markovian properties
where𝑓𝑡+1depends solely on 𝑓𝑡’s output while remaining
independent of historical states {𝑓<𝑡}. The chained composi-
tion guarantees process closure and reproducibility, though
constrained by the static combinatorial nature of {𝑓𝑖}𝑁
𝑖=1.
In the pre-defined pipeline, based on the position where
reasoning is introduced, it can be further divided into Pre-
Retrieval, Post-Retrieval, and Hybrid.
4.1.1 Pre-Retrieval Reasoning. For pre-retrieval meth-
ods, the sequence is explicitly defined as
𝐷=Γ◦R◦ Ψ(𝑄) (2)
where Ψdenotes a reasoning operator that systematically
transforms or enriches the query prior to retrieval. This par-
adigm enhances retrieval precision by resolving ambiguities,
inferring implicit intents, or optimizing query representa-
tions. Current research identifies four principal methodolog-
ical categories for designing Ψ:
Query Optimization focuses on generating and select-
ing query variants to maximize retrieval relevance. Mathe-
matically, this is formalized as Candidates =Generate(𝑄,𝐶),
ΨOptimize(𝑄,𝐶)=arg max candidate∈Candidates Score(candidate),
where ( Generate ) produces candidate queries and ( arg max )
selects optimal variants based on contrastive training or re-
inforcement learning. Representative implementations, such
as LeReT [ 34], leverage iterative sampling and optimization
to balance query diversity and specificity.
Attribute Judgment employs classification mechanisms
to dynamically regulate retrieval triggers. This is modeled as
ΨClassify(𝑄)=Classify(𝑄), where Classify evaluates queryattributes (e.g., temporal sensitivity, intent complexity) against
predefined criteria. Frameworks like UAR [ 14] and Adap-
tiveRAG [ 41] exemplify this approach by integrating multi-
stage classifiers to minimize unnecessary retrievals.
Plan Generation decomposes complex queries into struc-
tured sub-task sequences to guide retrieval direction. For-
mulated as ΨPlan(𝑄)=Plan(𝑄), the operator Plan generates
hierarchical task decompositions, as seen in PlanRAG [ 48] ,
which utilizes chain-of-thought reasoning to align retrieval
targets with multi-step problem-solving requirements.
Semantic Enhancement enriches query representations
using domain-specific or task-aware embeddings. Expressed
asΨEnhance(𝑄)=Encode(𝑄,K), whereKdenotes auxil-
iary knowledge (e.g., reasoning trajectories), methods like
O1-Embedder [ 101] integrate latent reasoning patterns into
query embeddings to improve retrieval robustness.
Collectively, these methodologies demonstrate that pre-
retrieval reasoning serves as a systematic interface to mit-
igate semantic gaps between raw queries and knowledge
bases, establishing a critical component for precision-driven
RAG architectures.
4.1.2 Post-Retrieval Reasoning. In pre-defined RAG sys-
tems with multi-step reasoning pipelines, the post-retrieval
reasoning paradigm represents a critical advancement where
cognitive processing occurs after information retrieval from
external sources. This approach addresses inherent limita-
tions in conventional RAG, particularly in managing knowl-
edge conflicts, mitigating information insufficiency, and en-
hancing logical consistency across complex reasoning tasks.
Mathematically, this process can be formalized as a deter-
ministic function composition:
𝐷=Γ◦Ψ◦R(𝑄) (3)

Synergizing RAG and Reasoning: A Systematic Review Conference’17, July 2017, Washington, DC, USA
Rdenotes the retrieval operator, Ψimplements the reasoning
transformation, and Γrepresents the final decision function.
The core characteristic of Post-Retrieval Reasoning lies in
its execution of the reasoning process after retrieval, with
the reasoning target being the retrieved content. ToG2.0 [ 60]
proposes an iterative multi-step reasoning framework that
alternates between graph retrieval and context retrieval, in-
tegrating the reasoning judgment of LLMs to progressively
expand entities and prune irrelevant information, ultimately
generating accurate answers. This approach dynamically
addresses the issue of insufficient information through it-
erative refinement while establishing a dual-evidence veri-
fication mechanism via knowledge graph relation pruning
and entity-guided context retrieval. Its graph-structured rea-
soning module transforms the connectivity validation of
triple paths into a constraint satisfaction problem, effectively
mitigating logical inconsistencies between text fragments
and thereby significantly improving the quality of complex
question answering.
ActiveRAG [ 100], on the other hand, employs a predefined
three-stage process (Self-Inquiry →Knowledge Assimilation
→Thought Accommodation) to structurally comprehend
and calibrate retrieved knowledge, resolving conflicts be-
tween parametric memory and external knowledge. During
the Knowledge Assimilation stage, ActiveRAG enhances the
corrective effect of external knowledge on the internal rep-
resentations of LLMs through multi-instruction fine-tuning
strategies (e.g., counterfactual comparison and anchor as-
sociation), substantially reducing the likelihood of hallu-
cination generation. ARM’s [ 7] structural alignment and
self-verification stages also demonstrate optimization for
post-retrieval reasoning. By incorporating domain knowl-
edge via mixed-integer programming (MIP) solvers, ARM
ensures the rationality and coverage of retrieval results, pro-
viding a scalable optimization framework for multi-source
data compatibility and thereby enabling globally optimal
cross-modal retrieval.
4.1.3 Hybrid Reasoning. The Hybrid pattern of pre-defined
process forms a composite processing paradigm by integrat-
ing pre-retrieval reasoning with post-retrieval reasoning.
The essence is formalized as a multi-round recursive itera-
tive process, where each iteration cycle strictly comprises
three phases: Retrieval, Generation, and Reasoning, executed
as structured composite operations. Let the total number of
iterations be 𝑇; the workflow is defined as:
𝑄𝑇=
⃝𝑇
𝑡=1R⊔◦Γ𝑡◦Ψ𝑡
(𝑄0) (4)
Here, each iterative unit is indexed by 𝑡. The process ter-
minates when a predefined condition T(𝑄𝑡,𝐷𝑡,𝐶𝑡)is met,
yielding the final response Γfinal(𝐶𝑇). This recursive mecha-
nism enables dynamic synergy between knowledge acquisi-
tion and semantic inference, overcoming the linear limita-
tions of single-cycle retrieval-generation frameworks.IR-CoT [ 78] leverages chain-of-thought reasoning to iter-
atively construct intermediate logic chains, enabling multi-
hop retrieval guided by progressively refined contextual cues.
FinSearch [ 50] introduces a dual-phase architecture that first
generates structured search graphs to model temporal and
entity dependencies, followed by dynamic query rewriting to
optimize financial data retrieval. LevelRAG employs hierar-
chical validation mechanisms, aggregating multi-granular re-
trieval results and triggering supplementary retrievals based
on context completeness assessments. ITER-RETGEN [ 68]
utilizes generation-enhanced feedback loops to iteratively
refine query representations, enhancing semantic alignment
between retrieval and generation phases.
These approaches share a common foundation in struc-
tured recursion while diverging in operational mechanisms.
By enforcing deterministic iteration cycles, they balance con-
trolled workflow execution with adaptive semantic explo-
ration, addressing challenges such as multi-step reasoning,
temporal coherence, and cross-domain knowledge synthesis.
The hybrid paradigm’s strength lies in its capacity to de-
compose complex queries into iterative retrieval-generation
units, systematically bridging knowledge gaps while main-
taining interpretability and robustness in open-domain problem-
solving scenarios.
4.2 Dynamic RAG Workflow
The RAG with dynamic workflow represents an autonomous
reasoning architecture centered around LLMs, characterized
by the integration of non-deterministic operational work-
flows and real-time decision-making capabilities. Unlike pre-
defined pipelines, this architecture enables continuous mon-
itoring of reasoning states to dynamically trigger retrieval,
generation, or verification operations. The LLM actively
evaluates contextual demands during reasoning processes,
autonomously determining optimal moments for invoking
external tools or resources through a hybrid feedback co-
ordination mechanism. By eliminating fixed iterative units
and pre-determined tool-calling sequences, the framework
achieves dynamic evolution of execution pathways, demon-
strating superior adaptability in complex cognitive tasks
through real-time adjustment of computational workflows
based on intermediate reasoning outcomes.
This dynamic architecture manifests three principal char-
acteristics: 1) Operator invocation is governed by the LLM’s
contextual state analysis, exemplified through special token
prediction (e.g., ‘[Web-Search]‘ or ‘<begin_of_query>‘) to
initiate external operations; 2) Reasoning trajectories exhibit
high flexibility, allowing dynamic query reformulation and
sub-problem generation to overcome limitations of static
workflows; 3) Context-driven decision mechanisms priori-
tize real-time reasoning states over predefined rules, enhanc-
ing systemic responsiveness to emergent task complexities
while improving precision.

Conference’17, July 2017, Washington, DC, USA Gao et al.
Defining the reasoning state at time 𝑡as𝑆𝑡=(𝐻𝑡,𝐶𝑡),
where𝐻𝑡denotes historical information aggregation and
𝐶𝑡represents contextual embedding vectors, the decision
process is modeled as a stochastic system:
𝑎𝑡+1∼𝜋(𝑆𝑡;Θ) (5)
𝑆𝑡+1=𝛿(𝑆𝑡,T𝑎𝑡+1(𝑆𝑡)) (6)
Here,𝜋:S→Δ(A) constitutes the policy function map-
ping states to probability distributions over action space A
(retrieval, generation, verification, etc.), while T𝑎denotes
state transition functions corresponding to action 𝑎. The
non-Markovian nature of the system emerges from 𝑆𝑡+1’s de-
pendence on complete historical trajectories {𝑆≤𝑡}, with dy-
namic adaptability ensured through extensible action spaces
Aand online optimization of policy parameters Θ. This
formulation enables context-sensitive state updates via 𝛿:
S×O→S , establishing a theoretical foundation for open-
ended reasoning processes in complex problem domains.
Based on the mode of reasoning initiation, agentic RAG
with dynamic workflows can be further categorized into
three distinct types: Proactivity-driven, Reflection-driven,
and Feedback-driven mechanisms. The LLM proactivity-
driven approach is characterized by the model’s autonomous
triggering of actions based on internal assessments, execut-
ing operations without external intervention through mech-
anisms analogous to human intuitive decision-making—for
instance, when the model independently identifies insuffi-
cient evidentiary support in the current reasoning process,
it proactively generates retrieval requests to supplement
information. The reflection-driven mode emphasizes self-
examination of the reasoning process, dynamically initiating
subsequent operations through quantitative evaluation of
intermediate result quality (e.g., triggering actions when the
calculated reasoning support score of 0.7 exceeds a prede-
fined threshold of 0.6), which simulates the self-optimization
logic of expert systems, enabling the model to adjust reason-
ing pathways through introspection. The feedback-driven
mechanism incorporates external intervention, employing in-
dependent models or rule-based systems to perform real-time
scoring of intermediate states (e.g., an external reward model
assigning a 2.5/5 score to reasoning steps) while provid-
ing corrective suggestions, operating similarly to a mentor-
guided mode that continuously calibrates the reasoning work-
flow through external feedback signals.
4.2.1 Proactivity-Driven Reasoning. The core innova-
tion of Proactivity-driven Reasoning lies in enabling LLMs to
fully govern the reasoning process through self-triggered pre-
diction mechanisms. This active control manifests through
three key mechanisms: (1) direct tool invocation via model-
generated special tokens (e.g., [Web-Search]), without exter-
nal intervention, (2) context-aware decision making basedon real-time knowledge gaps or hypothesis verification re-
quirements, and (3) Markov Decision Process (MDP)-based
dynamic path optimization.
Formally, the reasoning process can be modeled as a state
sequence𝑆={𝑠0,𝑠1,...,𝑠 𝑡}, where each state 𝑠𝑡encapsu-
lates the current reasoning context. At each step 𝑡, the LLM
selects an action 𝑎𝑡∈{retrieve,generate,terminate}based
on𝑠𝑡, executes the corresponding operation (e.g., document
retrieval or answer generation), and updates its state through
transition function 𝑠𝑡+1=𝛿(𝑠𝑡,𝑎𝑡,𝑜𝑡)where𝑜𝑡represents ac-
tion outcomes. This MDP framework enables dynamic path
adjustment through real-time feedback until termination
(𝑎𝑇=terminate) and final answer generation.
Recent advancements demonstrate significant improve-
ments over conventional RAG approaches. The Agentic Rea-
soning framework achieves granular control through dy-
namic tool invocation, eliminating predefined execution se-
quences. DeepRAG [ 24] optimizes cost-accuracy tradeoffs
via MDP-based imitation learning, addressing the retrieval-
generation disconnection in traditional systems. CoRAG [ 83]
introduces hybrid-driven mechanisms combining LLM-initiated
subqueries with external policy control, enhancing error tol-
erance for complex queries. Collectively, these approaches
establish a paradigm shift from fixed pipelines to context-
sensitive, self-optimizing reasoning architectures.
4.2.2 Reflection-Driven Reasoning. The reflection-driven
mechanism represents a dynamic reasoning framework that
enables iterative self-evaluation and revision of intermedi-
ate outputs through model introspection. Common methods
include: (1) a evaluation system combining explicit token pre-
diction and implicit confidence scoring, (2) self-monitoring
capabilities through grounding tokens for content-document
consistency verification and utility tokens for answer effec-
tiveness assessment, and (3) adaptive routing mechanisms
that automatically select single-hop or multi-hop reasoning
paths based on contextual complexity. The mathematical
formalism of this process can be expressed as:
P=𝑇Ø
𝑡=1[𝐺(C𝑡)→𝐸(H𝑡,D)→𝜓(𝜙(e𝑡),𝜏)] (7)
where𝐺denotes the generation function operating on
current context c𝑡,𝐸represents the evaluation function that
assesses hidden states h𝑡against external knowledge base
D,𝜙serves as the confidence mapping function, 𝜏is the
decision threshold, and 𝜓functions as the branch selector.
In practical implementations like Self-RAG [ 3], this frame-
work generates candidate responses alongside reflection to-
kens, computes passage relevance scores ( ISREL∈ [0,1])
and factual support metrics ( ISSUP ), and employs weighted
aggregation of token probabilities in 𝜙to determine retrieval
activation or generation revision through threshold-based 𝛿
operations. Meanwhile, Open-RAG [ 38] incorporates hybrid
threshold mechanisms and Mixture-of-Experts architecture

Synergizing RAG and Reasoning: A Systematic Review Conference’17, July 2017, Washington, DC, USA
to enforce counterfactual verification through non-retrieval
confidence scoring ( PrNoRT), enabling dynamic expansion of
complex reasoning capabilities while preserving base model
efficiency. ReaRAG [ 49] utilizes knowledge-guided reason-
ing chains combined with external knowledge sources to
perform reflection-driven reasoning. In each iteration, it
adjusts the reasoning path through the "Thought-Action-
Observation" paradigm, effectively preventing error propa-
gation and improving answer accuracy.
The paradigm’s innovation lies in reconstructing tradi-
tional sequential processes into conditional Markov deci-
sion processes, where state transition probabilities 𝑃(𝑠𝑡+1|𝑠𝑡)
are dynamically determined by model self-evaluation out-
comes. Compared to proactive LLM-driven methods (e.g.,
Toolformer’s direct API invocation), the reflection-driven
approach establishes closed-loop control through explicit
evaluation stages (function 𝐸), effectively mitigating halluci-
nation risks while maintaining computational efficiency.
4.2.3 Feedback-Driven Reasoning. The feedback-driven
dynamic RAG system establishes closed-loop control over
reasoning processes through external signals, formally mod-
eled as a Partially Observable Markov Decision Process.
The system state 𝑠𝑡=(𝑞𝑡,K𝑡,𝐻𝑡)evolves through itera-
tive interactions, comprising the current query representa-
tion𝑞𝑡, dynamic knowledge base K𝑡, and historical trajec-
toryH𝑡. Initialized with 𝑞0andK0=∅, the policy func-
tion𝜋(𝑎𝑡|𝑠𝑡)generates actions from the operational space
A={Retrieve,Reason,Verify,Answer,∅}. State transitions
follow𝑠𝑡+1=𝛿(𝑠𝑡,𝑎𝑡)with knowledge base updates
K𝑡+1=K𝑡⊕Retrieve(𝑞𝑡)·I(𝑎𝑡=Retrieve) (8)
where⊕denotes incremental updates and Irepresents an
indicator function. The reward function 𝑅(𝑠𝑡,𝑎𝑡,𝑠𝑡+1)→𝑟𝑡
drives policy optimization through
𝜋𝑡+1=Ω(𝜋𝑡,∇𝜃E𝑎∼𝜋𝑡[𝑅(𝑠𝑡,𝑎,𝑠 𝑡+1)]) (9)
forming an adaptive control loop. Three distinct feedback
mechanisms emerge within this framework.
Explicit reward feedback employs specialized models
𝜋reward for quantitative evaluation, exemplified by RAG-Gym’s
process rewards [ 96]. The reward function combines imme-
diate and terminal rewards:
𝑟𝑡=𝜆1𝜋reward(𝑠𝑡)+𝜆2E𝑠𝑡+𝑘[𝛾𝑘𝑅terminal] (10)
with discount factor 𝛾. SmartRAG extends this through
policy gradient optimization
∇𝜃𝐽(𝜃)=E𝜏∼𝜋𝜃[𝑇∑︁
𝑡=0∇𝜃log𝜋𝜃(𝑎𝑡|𝑠𝑡)ˆ𝐴𝑡] (11)
where the advantage function ˆ𝐴𝑡integrates temporal feed-
back.
Implicit environmental feedback derives from knowl-
edge base validation, as implemented in KBQA-o1’s SPARQL
verification and SolutionRAG’s pruning mechanisms [ 58].This feedback is formalized as 𝑟𝑡=I(K𝑡|=𝑞0)·𝑐valid−I(⊥∈
K𝑡)·𝑐invalid with validation function I(·)and penalty co-
efficients𝑐. ReARTeR [ 75] introduces threshold-triggered
correction: when 𝑟𝑡<𝜏, it activates refinement loops K𝑡+1=
PEM(K𝑡,𝑞0)⊕Retrieve(PRM(𝑠𝑡)).
Structured rule feedback encodes domain knowledge
through differentiable scoring functions. MCTS-KBQA [ 97]
implements depth-attenuated rewards
𝑟𝑡=1
1+𝛼𝑑𝑡𝑛∑︁
𝑖=1LLM score(𝑎(𝑖)
𝑡) (12)
with search depth 𝑑𝑡and decay coefficient 𝛼. CR-Planner’s
hierarchical critique combines subgoal and execution scores:
𝑟total
𝑡=𝛽1𝜋sub(𝑠𝑡)+𝛽2𝜋exec(𝑎𝑡|𝑠𝑡)through weighted fusion.
These feedback mechanisms interact through a unified
strategy update framework, where external feedback-driven
approaches achieve controllable optimization of the reason-
ing process through interpretable feedback signals while
maintaining the generative capabilities of LLMs. Overall,
the dynamic process of RAG, by endowing the model with
autonomy in the reasoning process, not only enhances adapt-
ability to complex tasks but also provides a new solution for
efficient reasoning in resource-constrained environments.
5 Implementation and Optimization
Building upon preceding sections, this section systemati-
cally analyzes the concrete implementation and optimization
strategies for reasoning within the RAG paradigm. In con-
trast to existing surveys that predominantly focus on post-
training methodologies or isolated LLM reasoning mecha-
nisms, our analysis maintains a dedicated focus on the syn-
ergistic integration of RAG with reasoning examining their
co-adaptive implementations through a structural lens.
5.1 Reasoning Process
5.1.1 LLM CoT. Integrating Chain-of-Thought (CoT) rea-
soning with LLMs is key to combining RAG with complex
reasoning tasks. Research shows CoT enhances RAG sys-
tems by explicitly guiding multi-step reasoning and dynam-
ically incorporating external knowledge. For example, Ac-
tiveRAG [ 100] uses a "Self-Inquiry →Knowledge Assimila-
tion→Thought Accommodation" chain to align knowledge
and reasoning: a knowledge assimilation agent merges ex-
ternal documents with LLM memory via operations like
association and reflection, creating structured knowledge.
Meanwhile, a reasoning adaptation agent refines inference
chains from Self-Inquiry to ensure answers align with re-
trieved knowledge and address reasoning gaps. Similarly,
Adaptive-RAG [ 41] alternates between CoT and retrieval,
breaking down multi-hop reasoning into steps such as entity
localization and document correlation, refining retrieval and
generation based on prior results.

Conference’17, July 2017, Washington, DC, USA Gao et al.
Figure 6. Implementation and optimization of the synergy between RAG and Reasoning
At the knowledge and reasoning level, O1-Embedder [ 101]
drives RAG through open-ended long-text reasoning, extend-
ing CoT beyond fixed triggers via coherent thought processes
like problem decomposition. PlanRAG [ 48] explicitly uses
CoT to produce executable multi-step plans, adjusting op-
erations dynamically through a closed-loop "plan-execute-
feedback" cycle. Despite different implementations, these
methods share two CoT strengths: breaking down complex
problems into clear intermediate steps and guiding external
knowledge selection through reasoning states. Studies show
these approaches outperform traditional RAG in multi-hop
QA and knowledge-intensive tasks by enhancing both LLMs’
reasoning and adaptability to external knowledge.
5.1.2 Special Token Prediction. Recent advances active
RAG also highlight special token prediction as a key method
for dynamically linking external knowledge retrieval with
multi-step reasoning [ 16]. By embedding domain- or action-
specific tokens (e.g., ‘[Web-search]‘, ‘[Retrieve=Yes]‘, ‘<be-
gin_of_query>‘) into LLM vocabularies, models can autonomously
trigger tools or self-reflect during text generation. Frame-
works like Self-RAG [ 3] and SmartRAG [ 20] use dedicated to-
kens (‘Retrieve‘, ‘ISREL‘, ‘[RETRIEVE]‘) to manage retrieval
activation, relevance checks, and output verification, turn-
ing static reasoning chains into conditional workflows. The
innovation lies in predicting these tokens within generated
sequences, segmenting tasks into retrieval initiation, docu-
ment evaluation, and knowledge grounding phases.Hybrid models such as Open-RAG [ 38] combine token con-
trol with mixture-of-experts (MoE) routing, sparsely activat-
ing experts aligned with token-predicted reasoning. Unlike
traditional chain-of-thought or search tree methods, special
token prediction offers finer control and interpretability by
encoding decision logic explicitly in token sequences while
maintaining end-to-end training. This approach also over-
comes latency and inflexibility of preset retrieval schedules
by enabling context-aware, on-demand tool use. For example,
R1-Searcher [ 72] and Search-o1 [ 51] use token boundaries
like ‘<end_of_query>‘ to coordinate retrieval pauses and
resume generation after knowledge integration.
Together, these systems show that token-level prediction
not only bridges reasoning and retrieval but also creates
a scalable framework for tool-enhanced language agents,
preserving generative fluency while enabling systematic ex-
ternal knowledge integration and procedural reasoning.
5.1.3 Search-Driven Reasoning. Recent advancements
in search-driven reasoning have significantly improved RAG
frameworks by employing structured search strategies for
dynamic information exploration and multi-step reasoning
with external knowledge. Current approaches mainly follow
three paradigms: tree-based search, MCTS, and reinforce-
ment learning-optimized policy networks.
Tree-based methods organize reasoning hierarchically
through structured path exploration. For example,StePO-
Rec [ 5] uses a multi-step tree-structured reasoning method

Synergizing RAG and Reasoning: A Systematic Review Conference’17, July 2017, Washington, DC, USA
that iteratively retrieves different outfit matching knowl-
edge and user preferences at each node, ultimately achiev-
ing generative recommendations for complementary items.
OmniThink [ 94] uses an information tree to expand topic
analysis by generating subqueries that guide breadth-first
or depth-first retrievals. DeepRAG [ 24] applies a binary tree
search within a Markov decision process to explore para-
metric knowledge and retrieval paths in parallel, selecting
optimal branches. DeepSolution’s [ 54] bidirectional thinking
tree alternates expanding solution and critique nodes with
scoring for path pruning, aligning naturally with MCTS eval-
uation. These methods balance exploration efficiency with
solution coverage through explicit tree structures.
MCTS enhances robustness by optimizing long-term deci-
sions via simulation, evaluation, and backpropagation. CR-
Planner [ 52] integrates MCTS with the UCB strategy to bal-
ance exploration and exploitation while estimating optimal
subgoals through multi-step simulations. KBQA-O1 [ 58] and
MCTS-KBQA [ 97] generate candidate actions using policy
models and combine reward models to globally assess logi-
cal forms, reducing local optima. ReARTeR [ 75] innovatively
merges MCTS with procedural reward models (PRMs), in-
terleaving retrieval and reasoning steps, and filtering high-
reward paths to form a closed-loop "reason-retrieve-reason"
cycle. These methods probabilistically explore paths and use
reinforcement learning feedback to improve global reasoning
for complex tasks.
Reinforcement learning-optimized policy networks adap-
tively refine search strategies. LeReT [ 34] replaces fixed
search algorithms with reinforcement learning (e.g., IPO)
to dynamically optimize query generation based on rewards
like retrieval accuracy, implicitly learning optimal search pat-
terns without explicit tree or graph structures, thus offering
greater flexibility and scalability.
In summary, search-driven reasoning unites inference and
retrieval through structured strategies, combining multi-path
exploration, dynamic evaluation, and adaptive optimization
to deliver interpretable, efficient solutions for knowledge-
intensive tasks. Future work may focus on hybrid paradigms
(e.g., integrating MCTS and reinforcement learning) and
lightweight algorithms to balance performance with compu-
tational efficiency.
5.1.4 Reasoning on Graph. Graph-structured reasoning
offers a novel approach for multi-hop inference in RAG sys-
tems by explicitly modeling knowledge interaction paths
through topology. Current methods fall into two categories:
query-flow-oriented search graphs (e.g. FinSearch [ 50]) and
knowledge-association-based expansion graphs (ToG-2.0 [ 60]).
FinSearch builds a directed acyclic graph (DAG) where nodes
are atomic subqueries (e.g., stock prices, financial reports)
and edges capture logical and temporal dependencies. A
pre-planner breaks down queries into subquery sequences,using graph traversal to control information flow and dy-
namically adjust paths, such as backtracking when conflicts
arise—substantially surpassing linear chain-of-thought meth-
ods in handling complex logic.
5.1.5 External Solver. The integration of RAG and rea-
soning is also can be achieved by incorporating external
solvers, where specialized solvers, such as the Alignment-
Oriented LLM-based Retrieval Method (ARM), are employed
to handle the reasoning component. The retrieval process for
complex problems is formulated as a global optimization task,
leveraging external solvers like mixed-integer programming
(MIP) to achieve structural alignment and joint optimiza-
tion of data objects. Specifically, ARM first decomposes user
queries into keywords that match N-grams in the dataset
through an information alignment module, generating an
initial set of retrieval candidates via constrained decoding.
Subsequently, in the structural alignment phase, the MIP
solver performs global filtering on candidate objects based
on a predefined objective function that maximizes both the
relevance of retrieved objects to the query and their mutual
compatibility. This ensures that the selected objects not only
cover the requirements of the query but also form a coherent
information chain through entity or inter-table linkages. Fi-
nally, the self-verification mechanism of the LLM, combined
with a beam search-based aggregation strategy, dynamically
refines and consolidates multiple candidate sets, ultimately
producing a retrieval collection that satisfies both semantic
matching and the structural organization of the data.
ToG-2.0 achieves multi-hop expansion by integrating knowl-
edge graphs with documents, starting from an initial entity
and iteratively extending relevant entities and relations (such
as corporate ownership chains and technology dependency
networks) via the Edge function. This process constructs
structured triple paths while simultaneously retrieving and
verifying document content. By tuning the width and depth
parameters, the method emulates human reasoning: broadly
exploring potential associations before deeply verifying high-
confidence paths. FRAG [ 23] dynamically adjusts retrieval
strategies by predicting the hop range of reasoning paths
based solely on the query text, thereby enhancing retrieval
quality without requiring additional fine-tuning or invoca-
tion of large language models, enabling flexible and efficient
retrieval optimization. FG-RAG [ 32] further expands entity
coverage in graph retrieval through context-aware entity
expansion, providing richer background information. Com-
bined with query-level fine-grained summary generation,
FG-RAG transforms coarse-grained graph information into
highly relevant detailed content, effectively improving the
performance of query-focused summarization tasks.
Although differing in design from workflow-based meth-
ods, ToG-2.0 shares key advantages with other graph-structured
approaches: explicitly modeling reasoning state dependen-
cies, supporting dynamic path generation and optimization,

Conference’17, July 2017, Washington, DC, USA Gao et al.
and enabling closed-loop interaction between retrieval and
reasoning. This effectively overcomes the limitations of tradi-
tional RAG in implicit relation inference and counterfactual
analysis, thereby establishing an interpretable theoretical
and practical framework for knowledge reasoning.
5.2 Reasoning Optimization
In the previous chapter, we focused on introducing several
approaches to integrate reasoning with RAG. This chapter
shifts attention to how to augment the reasoning capabilities,
specifically including Prompt-Based, Tuning-Based, and RL-
Based strategies.
5.2.1 Prompt-Based. Prompt-Based optimization is a key
approach to improving RAG and reasoning system perfor-
mance by using carefully designed natural language prompts.
These prompts break down complex reasoning tasks into
manageable steps and guide LLMs to follow specific log-
ical structures during generation. The main advantage is
that control over reasoning flow is achieved solely through
prompt design, without parameter fine-tuning or reinforce-
ment learning, preserving the model’s generalization while
enhancing task-specific results.
This approach has three main features. First, task struc-
turing : prompts explicitly decompose and control reason-
ing chains via zero-shot or templated designs. Techniques
like Co-STORM [ 43] and WriteHere [ 98] use role assign-
ments, stage divisions, and operation-specific instructions
to guide multi-step reasoning—such as proposal generation,
knowledge retrieval, refinement, and validation—improving
interpretability by representing intermediate steps clearly.
Second, result reliability is improved by standardizing
outputs and reducing hallucinations. Strategies include re-
quiring citation of retrieval results, enforcing specific output
formats, and integrating reflection and calibration based
on retrieved knowledge. Systems like FinSearch [ 50] and
ActiveRAG [ 100] incorporate temporal weighting, dedupli-
cation, and domain rules through prompts, enhancing consis-
tency and logical coherence, especially in complex domains.
Third, interactive adaptability allows dynamic prompt
adjustments. Special tokens (e.g., <Search> ,[Web-search] )
enable models to trigger tools or revise queries in real time
based on intermediate results. Methods such as Agentic Rea-
soning [ 92] and PlanRAG [ 48] use context-sensitive prompts
and feedback loops to refine reasoning paths dynamically,
maintaining coherence and accuracy in multi-hop tasks and
outperforming traditional RAG methods in complex, evolv-
ing scenarios.
In summary, prompt-based optimization offers an efficient,
flexible, and reliable approach to enhancing RAG+Reasoning
by emphasizing task structuring, result standardization, and
interactive adaptability. Its non-intrusive and broadly ap-
plicable design has established it as a mainstream strategy
for optimizing LLM reasoning and serves as a foundationfor future hybrid methods integrating fine-tuning and re-
inforcement learning. By systematically optimizing reason-
ing without altering model parameters through semantic
structures, dynamic feedback, and symbolic constraints, this
paradigm effectively manages macro-level controls like task
decomposition and knowledge integration while address-
ing key challenges such as generation consistency, logical
coherence, and external knowledge alignment. This makes
prompt-based optimization a lightweight yet powerful solu-
tion for complex reasoning tasks.
5.2.2 Tuning-Based. The tuning-based approach improves
the integration of RAG and reasoning by optimizing model
parameters to internalize the retrieval-augmented chain-of-
thought mechanism within LLMs. Current research mainly
targets three goals: retrieval pathway optimization ,structured
generation enhancement , and collaborative training with ex-
ternal modules .
For retrieval pathway optimization, methods like CoRAG [ 83]
and DeepRAG [ 24] build end-to-end multistep reasoning
frameworks through full parameter fine-tuning and multi-
task learning. CoRAG expands single-step QA datasets into
retrieval-reasoning chains and jointly trains tasks such as
sub-query generation, intermediate answer prediction, and
final composition. This boosts the model’s ability to break
down complex problems (e.g., multi-entity relational reason-
ing) and adapt retrieval strategies dynamically (e.g., query
rewriting, error correction). DeepRAG combines imitation
and contrastive learning with binary tree search to create
efficient retrieval paths, using a DPO-style contrastive loss
to reduce redundant retrieval while maintaining accuracy.
To improve structured generation, MCTS-KBQA [ 97]and
Self-RAG [ 3] fine-tune models for precise special token gen-
eration. MCTS-KBQA uses supervised fine-tuning to make
large language models output instructions that comply with
knowledge graph protocols (e.g., SPARQL), modeling reason-
ing as executable tool-call sequences. Self-RAG enhances self-
supervised generation control by expanding vocabulary and
training the model to generate reflection tokens like retrieval
triggers and relevance markers, preserving fluency and re-
ducing factual errors. Additionally, O1-Embedder [ 101] and
Open-RAG [ 38] align semantic spaces via mixed fine-tuning:
O1-Embedder combines generative and contrastive train-
ing with special tokens to separate generation from embed-
ding tasks, enhancing multihop semantic understanding;
Open-RAG uses QLoRA [ 17] quantized fine-tuning and Mix-
ture of Experts (MoE) modules to specialize networks for
single/multi-hop reasoning.
In collaborative optimization with external modules, Adap-
tiveRAG [ 41] and CR-Planner [ 52] apply parameter isolation
to balance generality and adaptability. AdaptiveRAG fine-
tunes a lightweight classifier to select retrieval strategies
dynamically. CR-Planner introduces a Critic model trained
with contrastive loss on MCTS trajectory data to assess the

Synergizing RAG and Reasoning: A Systematic Review Conference’17, July 2017, Washington, DC, USA
Table 1. Comparison of RL-based RAG with Reasoning Methods
Method Base Model RL Parameter Supervision Reward Function Policy Strategy
PORAG [73] Qwen2.5/Llama3.2 GRPO QLoRA ORMDual rewards:
1. Retrieval fidelity ( 𝑅fid)
2. Response quality ( 𝑅qual)
Combined:𝑅=𝛼𝑅fid+𝛽𝑅qual•Group-based advantage normalization
•PPO-style clipped objective
•KL regularization
DeepResearcher [106] Qwen2.5-7B GRPO Full ORMFormat compliance penalty (-1)
+ Answer F1 score•Reference policy constraints
•KL divergence penalty
ReSearch [6] Qwen2.5-7B GRPO Full ORMHybrid rewards:
•Answer F1 (vs ground truth)
•Format compliance check•GRPO with clip ratio 0.2
•Group advantage normalization (G=5)
•𝛽=0.001KL penalty
ReZero [16] Llama3.2-3B GRPO Full ORM+PRM•Answer correctness
•Format compliance
•Search diversity
•Chunk matching
•Retry behavior
•Strategy compliance•Intra-group reward comparison
•Noise-injected robustness training
•KL constraints
MMOA-RAG [12] Llama-3-8B MAPPO Full ORMShared F1 reward + penalties:
•Excessive sub-questions
•Document ID errors
•Answer verbosity•MAPPO actor-critic updates
•Cosine learning rate scheduling
DeepNote [84] Qwen2.5/Llama3.1 DPO Full ORMImplicit preference modeling
via likelihood contrast•Direct Preference Optimization
•Preference gap maximization
R1-Searcher [72] Qwen2.5/Llama3.1 Reinforce++ Full ORMTwo-stage rewards:
1. Retrieval count + format
2. F1 score + format penalty•RAG-based rollout
•Retrieval-masked loss
KBQA-O1 [58] Llama3/Qwen2.5/Gemma2 MCTS DoRA ORM+PRMComposite reward:
•Stepwise policy model score
•Final reward model score•MCTS trajectory optimization
•Q-value backpropagation
DeepRetrieval [42] Qwen2.5-3B PPO Full ORMTask metrics:
•Recall@k/NDCG
•Syntax validity•GAE advantage estimation
•Distributed HybridFlow
LeReT [34] Llama3-8B/Gemma-9B IPO Full PRMAverage Precision (AP)
of retrieved documents•Identity Policy Optimization
•Context distillation
SmartRAG [20] Flan-T5-L/Llama2-7B PPO Full/LoRA ORMAction-specific:
•EM+F1 for answers
•Cost penalty for retrievals•On-policy sampling
•PPO updates
ReARTeR [75] LLaMA3.1-8B MCTS LoRA ORM+PRMMonte Carlo step scoring
+ TD look-ahead•Iterative preference optimization
•KTO loss
DeepRAG [24] Qwen2.5-7B/Llama3.1-8B Hybrid Full ORM+PRMCost-aware accuracy:
𝑅=−𝐶(𝑜)×𝑇(𝑠𝑡)
𝐶(𝑜): Answer correctness
𝑇(𝑠𝑡): Total retrieval cost•Imitation + contrastive learning
•PPO-like calibration
RAG-Gym [96] LLaMA3.1-8B Hybrid LoRA PRMTriple criteria:
•Sufficiency
•Utility
•Redundancy•SFT + DPO
•PRM-guided selection
CR-Planner [52] Skywork-Llama3.1-8B MCTS LoRA PRMCritic-estimated rewards:
•Stepwise correctness
•Global impact•MCTS simulation
•Pairwise ranking loss
1ORM: Outcome-based Reward Model; PRM: Process-based Reward Model.2Full: Full parameter tuning.
long-term value of reasoning actions, prioritizing efficient
solutions in tasks like mathematical reasoning.
Together, these tuning strategies restructure the parame-
ter space to internalize retrieval-reasoning interactions ef-
fectively, enhancing the model’s ability to solve complex
problems while ensuring computational efficiency and broad
applicability across domains.
5.2.3 RL-Based. As shown in Table 1, Reinforcement learn-
ing (RL) has recently become pivotal for tackling long-chainreasoning in modern inference models and optimizing RAG
combined with reasoning tasks. Central to these advances is
the use of dynamic reward mechanisms that guide LLMs to
balance knowledge retrieval and logical reasoning adaptively.
RL optimization objectives generally fall into two categories:
outcome-based reward modeling (ORM) and process-based
reward modeling (PRM), with some hybrid approaches blend-
ing both to balance global goals and local optimizations.

Conference’17, July 2017, Washington, DC, USA Gao et al.
The ORM paradigm focuses solely on the quality of the
final output and its adherence to standards. For example, R1-
Searcher [ 72] employs a two-stage Reinforce++ [ 35] training
where rewards in the first stage depend on correct retrieval
calls and special token generation, while the second stage
directly optimizes the F1 score of answers. This encourages
the model to develop strategies maximizing knowledge in-
tegration, reducing hallucinations, and enhancing accuracy
in multi-hop QA beyond traditional RAG methods. Simi-
larly, KBQA-O1 [ 58]uses MCTS with a policy network for
candidate reasoning paths and a reward model evaluating
logical consistency, effectively balancing exploration and
exploitation in knowledge base QA.
Conversely, PRM emphasizes detailed supervision of inter-
mediate reasoning steps. LeReT [ 34] uses the Identity Policy
Optimization (IPO) algorithm, optimizing query quality by
rewarding average precision (AP) of retrieved documents,
boosting retrieval recall and overall multi-hop task perfor-
mance. ReARTeR [ 75] extends this with a step-level binary
reward model, combining Monte Carlo scoring and temporal
difference (TD) methods to evaluate reasoning paths proac-
tively, reducing logical errors and redundant retrievals, and
improving accuracy on benchmarks like HotpotQA.
Moreover, influenced by DeepSeek-R1, GRPO [ 69] is also
gradually being applied in scenarios combining RAG and
Reasoning. GRPO is a variant of the Proximal Policy Opti-
mization (PPO) reinforcement learning algorithm that aban-
dons the critic model and instead estimates the baseline from
group scores, significantly reducing training resources. For
example, ReZero [ 16] uses GRPO to introduce a "retry" mech-
anism for LLMs, incentivizing LLMs to keep trying after an
initial search failure by rewarding retry search queries. This
mechanism simulates the human strategy of "if at first you
don’t succeed, try again" in information retrieval. PORAG [ 73],
based on GRPO, directly optimizes retrieval quality, contex-
tual relevance, and generation coherence through a dual
reward mechanism (retrieval fidelity and response quality).
Hybrid methods merge ORM and PRM to optimize both
final outcomes and intermediate steps via composite rewards.
SmartRAG [ 20] applies Proximal Policy Optimization (PPO),
combining answer-level F1 rewards with penalties for ex-
cessive retrievals, balancing knowledge completeness and
efficiency. RAG-Gym [ 96]advances this with multidimen-
sional process rewards (sufficiency, utility, redundancy) and
techniques like contrastive loss and Best-of-N sampling to
promote efficient search decisions, even zero-shot. These
hybrid strategies markedly lower retrieval costs while sus-
taining accuracy in complex tasks.
In addition, we can also observe that in current RL-based
methods, academia focuses more on exploration with small-
scale LLMs (<8B), among which the Qwen and Llama series
are the most widely used. Overall, RL provides a flexible, scal-
able framework for integrating RAG and reasoning. ORMguides the discovery of globally optimal strategies, PRM en-
hances reasoning robustness via local refinements, and their
combination addresses modular system limits. Future work
may explore collaborative rewards in multi-agent settings,
offline RL based on world models, and hierarchical reward
decomposition for open-domain applications.
6 Downstream Tasks and Evaluation
While previous chapters focused on methodologies and ad-
vances in RAG combined with reasoning, this chapter shifts
to tasks and evaluation. It provides a comprehensive overview
and analysis of existing tasks, datasets, their current status,
and emerging trends. By reviewing these resources, we high-
light the landscape’s gaps and limitations in current eval-
uation methods. The chapter also explores key challenges
in assessment frameworks, identifying shortcomings and
suggesting potential improvements.
Figure 7. The current downstream tasks and datasets re-
lated to the combination of RAG and Reasoning show that
multi-hop question answering tasks still dominate. Corre-
spondingly, HotpotQA, 2WikiMultihopQA, and MuSiQue
remain the most commonly used evaluation datasets.
6.1 Knowledge-Intensive Tasks
In the evaluation for RAG systems, knowledge-intensive
question answering (QA) remains the primary focus (Fig-
ure 7). As LLMs improve in semantic understanding and
reasoning, benchmarks have expanded to cover tasks from
simple fact retrieval to complex multi-step reasoning. How-
ever, evaluation methods specifically designed for RAG lag
behind due to the dual challenge of assessing both retrieval-
generation coherence and adaptability to dynamic knowl-
edge bases. For example, multi-hop QA requires integrating

Synergizing RAG and Reasoning: A Systematic Review Conference’17, July 2017, Washington, DC, USA
dispersed knowledge through multi-stage retrieval while
verifying logical consistency between answers and retrieval
paths. This complexity increases dataset construction costs
compared to purely generative tasks, keeping research cen-
tered on knowledge-intensive QA subcategories such as
open-domain QA, knowledge-base QA, and multi-hop QA.
Commonly used datasets include Natural Questions (NQ) [ 47]
for single-hop factual queries, HotpotQA, 2WikiMultiHopQA [ 31]
and Musique [ 79] for multi-hop QA. These benchmarks
are mostly based on Wikipedia and fail to reflect the RAG
demands and corresponding complexity in real-world sce-
narios. Some efforts have pushed evaluation boundaries,
like CRUD-RAG’s [ 59] operational metrics and Domain-
RAG’s [ 86] domain-specific evaluations, but high costs and
metric-task interdependencies limit progress. As a result,
knowledge-intensive QA remains central for testing RAG
robustness and practicality, highlighting a critical bottleneck:
the need for innovative frameworks that balance retrieval
flexibility and controlled generation to support new devel-
opments like Agentic RAG .Overall, many evaluation bench-
marks are lagging behind rapid RAG+Reasoning advances,
especially as LLMs grow more powerful. Specifically, the
current evaluation of RAG faces the following challenges.
Limited Challenge. With improving LLM capabilities,
many knowledge-based questions are no longer difficult, as
they can be answered without external retrieval. Current
multi-hop reasoning datasets, often built from artificial tem-
plates, offer limited challenge. There is an urgent need for
more complex datasets reflecting real-world scenarios and
practical use.
Lack of Specificity. Existing evaluation tasks are still
predominantly focused on factual assessment and knowl-
edge retrieval, lacking evaluations that probe deeper analyt-
ical thinking. This constraint limits the ability to measure
a model’s capacity for profound reasoning and cognitive
depth.
Task Uniformity. The majority of benchmarks are overly
dependent on QA tasks, focusing on reactive, question-and-
answer-based interactions. There is a pressing need to in-
troduce tasks aligned with real-world applications, such as
active information retrieval tasks based on personal knowl-
edge or proactive knowledge discovery.
Insufficient Dimensions. Evaluations are primarily end-
to-end, focusing solely on final outcomes. However, with
the introduction of reasoning processes, RAG+Reasoning
systems have become iterative, multi-step frameworks. Cur-
rent evaluations are unable to assess intermediate reasoning
steps or retrieval chains effectively. The absence of step-by-
step supervision data limits both research and training of
related methods. Furthermore, current evaluation methodolo-
gies lack comprehensive assessments of system performancetrade-offs, such as computational cost and efficiency, which
are critical for practical deployment.
This emergent landscape necessitates the creation of a new
generation of evaluation frameworks that can address these
shortcomings. Such frameworks must not only ensure the
adaptability of retrieval and the controllability of generation
but also integrate intermediate reasoning evaluation and
efficiency metrics, paving the way for the development of
more robust and efficient RAG systems suited to diverse
real-world applications.
6.2 New Tasks on RAG+Reasoning
Recently, combining RAG with reasoning has significantly
improved models’ ability to tackle more realistic and chal-
lenging tasks, raising the standards for evaluation methods.
This subsection examines emerging tasks that assess their
combined strengths, related tasks and datasets are shown
in Table 2. Here, "emerging" refers not to entirely new tasks
but to those with unprecedented complexity and demands.
These include Deep Research tasks requiring multi-layered
information integration and reasoning; PhD (Expert)-Level
Complex Reasoning tasks targeting advanced scenario rea-
soning; and critical; domain-specific decision support tasks
like medical diagnosis and legal analysis. Such tasks demand
not only external knowledge retrieval but also logical con-
sistency, coherence, and depth in reasoning.
6.2.1 Deep Research. From the perspective of integrating
RAG and reasoning, Deep Research tasks exemplify complex
downstream applications. They require models to handle
open-ended retrieval, produce long-form, structured text,
and synthesize multi-source information through deep rea-
soning. This section analyzes their key features, evaluation
datasets, and metrics.
At the core of Deep Research tasks lies the mission of
addressing complex informational queries. These tasks are
distinguished by several key attributes:
First, dynamic interactivity is essential. Models engage in
iterative dialogue to uncover latent user needs or "unknown
unknowns". For example, the Co-Storm [43] framework en-
ables collaboration with multiple language model agents to
explore information gradually, easing user cognitive load
and capturing unmet needs more accurately.
Second, integrating information from multiple sources
is crucial. Models must consolidate diverse data to provide
comprehensive coverage. For instance, uses dynamic mind
maps to structure knowledge and produce cohesive reports,
ensuring accuracy and completeness.
Third, expert-level accuracy is required. Many tasks de-
mand domain expertise, expecting models to perform like
human specialists. The Agentic Reasoning [ 92] framework
illustrates this with high-stakes scenarios like medical treat-
ment design or legal analysis, where outputs are judged on
correctness, depth, and coherence.

Conference’17, July 2017, Washington, DC, USA Gao et al.
Table 2. Tasks and Datasets under the New Trend of RAG Combined with Reasoning
Task Type Sub-Task Dataset Description Scale Construction By Evaluation Paper
Deep ResearchDeep Research Agentic Reasoning
Deep Research [92]PHD-level dataset covering
finance, medicine, and law.15-30 domains PhD Experts Expert pass rate [92]
Report Genera-
tionWildSeek [44] Info-seeking task–goal pairs
for document generation.100 samples Rules/LLM/Manual LLM [98]
Report Genera-
tionTELL ME A
STORY [37]fiction writing evaluation
dataset: detailed prompts
and long-form narratives.230 samples Manual LLM [98]
Peer Review Review-5k [91] ICLR 2024 peer review
dataset: paper metadata
and structured reviewer
feedback.4,991 papers OpenReview/arXiv MSE/MAE/Acc [91]
Report Genera-
tionResearch-14k [91] 2022–2024 Accepted ML pa-
pers: outlines, full texts, and
cited abstracts.14,911 papers Semantic Scholar
+ arXivSimulated review
scores[91]
Report Genera-
tionSolutionBench [54] Engineering benchmark:
constrained solutions across
8 real-world domains.1,050 datapoints Manual/LLM ex-
tractionAnalytical/ Tech-
nical scores[54]
Mathematics &
ReasoningMath Reasoning GPQA [67] PHD-level MCQs in physics,
chemistry, and biology.744 sets PhD Experts Accuracy [92]
Math Reasoning MATH500 [55] 500 math problems from the
MATH test set.500 problems Public repos Pass@K [51]
Programming LiveCodeBench [40] Programming benchmark
with easy, medium, and hard
problems.1,055 problems Competition plat-
formsPass@K [51]
Programming USACO [70] USA Computing Olympiad
problems, testing algorithms
and coding.307 problems USA Computing
OlympiadPass@K [52]
Math Reasoning TheoremQA-
Math [33]BRIGHT subset: theorem-
based math problems.206 problems STEM datasets Accuracy [52]
Programming Gorilla [64] API-aware code generation
from HuggingFace, Torch
Hub, TensorFlow Hub docs.1,600 APIs Manual AST matching [73]
Math Reasoning OlympiadBench [29] Olympiad-level math compe-
tition problems.1,000 problems Competitions Accuracy/F1 [109]
Complex Reason-
ingComplexWebQA [ 76]Multi-step reasoning over
web queries with cross-
document integration.34,689 queries Web snippets Accuracy [36]
Demanding
RetrievalDomain Retrieval StackEcon & Stack-
Bio [33]Biology and economics
StackExchange questions
for complex retrieval.206 queries StackExchange nDCG@K [52]
Active Retrieval AR-Bench [14] Active retrieval benchmark
with four sub-tasks.8k/sub-task Synthetic Accuracy [14]
Real-time TAQA [104] QA dataset with time-
evolving answers.10K-100K rows Human-curated LLM [14]
Real-time FreshQA [80] Dynamic fact QA bench-
mark with evolving answers600 samples Mixed sources LLM [14]
Domain Retrieval PubMed [42] PICO-based medical search
dataset linking reviews to
PubMed.21k+ samples Systematic re-
viewsRecall@K [42]
Domain Retrieval Trial search [42] PICO-based clinical trial
search linked to ClinicalTri-
als.gov.7k+ samples Manually Recall@K [42]
Domain Retrieval FinSearchBench-
24 [50]Financial retrieval bench-
mark covering stocks, rates,
policy, trends.1,500 queries Manually Accuracy [50]
Decision &
QABusiness DQA [48] Decision QA benchmark
with business scenarios in
enterprise settings.301 pairs video games Accuracy [48]
Medical CMB-Clin [87] CMB subset for clinical diag-
nosis reasoning in Chinese
medical cases.74 cases Textbooks/diagnostic
materialsLLM/Expert [11]
Medical MM-Cases [11] Medicine cases generated
by GPT-4o-mini, verified by
doctors.609 cases LLM/doctor-
reviewedLLM/Expert [11]
Medical TCM-Cases [11] TCM patient cases generated
by GPT-4o-mini, verified by
doctors.130 cases LLM/doctor-
reviewedLLM/Expert [11]

Synergizing RAG and Reasoning: A Systematic Review Conference’17, July 2017, Washington, DC, USA
Fourth, multi-modal reasoning is often necessary. Deep
Research tasks involve varied data types—text, code, knowl-
edge graphs—and dynamic tool use such as web searches or
code execution to enhance reasoning.
Finally, handling multiple real-world constraints is vital.
Tasks may require generating practical solutions under spe-
cific conditions, like designing hospitals in challenging envi-
ronments with factors like heavy rainfall and seismic activity,
as seen in the DeepSolution framework. This ensures outputs
are feasible and relevant.
To ensure the diversity and complexity of Deep Research
tasks, their evaluation relies on datasets drawn from multiple
domains. A few notable examples include:
WildSeek Dataset [ 44]: This dataset is constructed from
real-world user information-seeking scenarios and comprises
100 data points covering 24 fields, including economics, com-
puter science, and law. Each data point is characterized by
a topic, user goal, and domain label. For example: "Domain:
Economics; Topic: Development of a Shared Trading Cur-
rency; Goal: Investigate how a new shared currency could
eliminate transaction costs". WildSeek effectively evaluates
models’ competence in dynamic interaction and multi-source
information integration.
GAIA [ 62]. The GAIA Benchmark, developed jointly by
Meta AI, Hugging Face, and others, is a comprehensive eval-
uation framework designed to assess general AI assistants’
ability to handle real-world problems. It features 466 care-
fully crafted tasks spanning language reasoning, visual per-
ception, multi-agent collaboration, and adaptability, focus-
ing on key skills like reasoning, multimodal processing, web
browsing, and tool use. GAIA measures performance across
dimensions such as task execution, adaptability, collabora-
tion, generalization, and real-world reasoning with metrics
like completion rate, response quality, efficiency, and robust-
ness. Unlike traditional benchmarks, it emphasizes robust-
ness and reliability in everyday scenarios, supports zero-shot
evaluation, prevents data contamination, and is widely used
in research and industry to guide AI development.
SolutionBench [ 54]: This dataset spans eight engineering
domains, including environmental, mining, and transporta-
tion engineering. Each instance presents a complex engineer-
ing problem with specific constraints. For example: "Design
a safe and efficient hospital construction plan in a region
with 3000mm annual rainfall, expansive soils, and frequent
seismic activity. "* SolutionBench evaluates models’ ability to
address multi-constraint problems and integrate specialized
knowledge effectively.
The current evaluation system for DeepResearch faces
the dual challenges of scarce specialized testing tasks and
the difficulty of assessing complex, lengthy reports: On one
hand, existing benchmark tests only cover basic capabilities
and lack systematic evaluation standards in specialized sce-
narios like business analysis and policy assessment; on theother hand, the multimodal integration, logical chain verifi-
cation, and domain adaptability testing of long reports pose
technical bottlenecks for traditional assessment methods,
necessitating the development of new evaluation tools that
integrate logic graphs, dynamic scenario simulation, and
domain knowledge bases.
In the future, the evaluation system will evolve into a
multidimensional framework, including the construction of
a three-level indicator matrix covering basic capabilities,
reasoning levels, and application value. Overcoming these
evaluation bottlenecks requires both technological innova-
tion and joint standard-building efforts. This concerns not
only the reliability validation of intelligent research tools
but also the reshaping of research evaluation paradigms and
industrial application boundaries.
6.2.2 PhD (Expert)-Level Complex Reasoning. The in-
tegration of RAG with advanced reasoning has become es-
sential for tackling expert-level, complex cognitive tasks,
particularly at the PhD level. These tasks, including com-
petitive programming, theorem-driven proof reasoning, and
cross-disciplinary knowledge retrieval, require multi-layered
logical inference and precise coordination between dynamic
retrieval and domain-specific knowledge. PhD-level reason-
ing differs from standard evaluations across three dimen-
sions: knowledge intensity, procedural rigor, and domain
specificity. Knowledge intensity demands dynamic access to
deep, specialized knowledge, such as analyzing dynamic pro-
gramming time complexity or applying algebraic topology
theorems—needs that surpass general corpora and call for
domain-specific knowledge graphs and retrieval methods.
Procedural rigor involves mathematical precision in multi-
step proofs, requiring logical consistency in symbolic ma-
nipulation, theorem use, and counterexample refutation, as
seen in international math competitions. Domain specificity
reflects tailored reasoning methods, e.g., handling synchro-
nization in concurrent programming or employing tensor
calculus in quantum field theory.
Evaluation systems for such tasks are inherently multi-
layered and multimodal. The USACO Benchmark [ 71] of-
fers a graduated difficulty scale for programming reason-
ing, testing both correctness and algorithmic constraints
like time complexity. TheoremQA-Math [ 9] links formalized
math problems to theorem libraries, demanding verifiable
mappings between theorem applications and calculations.
Cross-disciplinary datasets like StackBio and StackEcon [ 53]
assess models’ ability to extract critical knowledge from
dense, domain-rich documents, serving as strong tests for
domain-oriented retrieval accuracy.
Modern evaluation surpasses traditional end-to-end tests
by combining process and outcome validation. Frameworks
like CR-Planner [ 52] use dual models—a Sub-Goal Critic
to score reasoning chains and an Execution Critic to eval-
uate retrieval—allowing fine-grained step monitoring. For

Conference’17, July 2017, Washington, DC, USA Gao et al.
example, in dynamic programming, key steps like formu-
lating state transitions and retrieving boundary conditions
receive targeted feedback. Similarly, Search-O1 [ 51] quanti-
fies knowledge completeness by tracking uncertainty indi-
cators (e.g., tentative language), measuring confidence and
accuracy. Outcome validation maintains strict correctness
benchmarks in programming and combines metrics like F1
scores with expert review in open-domain scientific QA to
ensure precise understanding of domain-specific terms.
6.3 Challenges and Future Directions
6.3.1 Complex Domain Tasks. Recent advances in RAG
have provided novel solutions for more complex tasks in
professional domains. These downstream tasks transcend
the limitations of traditional question-answering models that
rely solely on simple retrieval-generation patterns, involving
challenges such as real-time information acquisition, inte-
gration of domain expertise, and dynamic decision-making
support. The nature of these tasks can be characterized along
three interrelated dimensions: (1) temporal dynamics , empha-
sizing the rapid changes in data and reasoning environment;
(2)domain specificity , focusing on deep integration of indus-
try knowledge and structured data; and (3) reasoning chain
complexity , reflecting requirements for multi-stage reasoning
and fine-grained decomposition of queries.
To rigorously evaluate such systems, innovative bench-
marking approaches have been proposed. The FinSearchBench-
24 dataset, for example, encompasses five months of mar-
ket data variations, integrating multi-variable interactions
across stock, policy, and industrial sectors, and includes
over 1,500 multiple-choice questions, thereby surpassing
the constraints of traditional static benchmarks. The evalua-
tion adopts a hierarchical and quantitative methodology: the
foundational level measures model accuracy and response
latency; the intermediate layer assesses the temporal sensitiv-
ity of information relevance and the contribution of retrieval
mechanisms to reasoning outcomes; and the advanced layer
employs ablation studies to highlight performance variances
under dynamic temporal decay. This multifaceted evalua-
tion not only differentiates surface-level retrieval capabilities
but also rigorously measures the synergy between reason-
ing quality and temporal context, furnishing theoretical and
practical foundations for long-term stability and predictive
accuracy in complex domain systems.
Experimental findings further reveal that establishing
long-term evaluation protocols with temporal weighting
functions is indispensable for adapting to realistic dynamic
environments. Nonlinear declines in decision accuracy, ob-
served when extending relevance windows from 72 to 168
hours, emphasize the importance of factoring temporal de-
cay into assessment frameworks. Future work should extend
these evaluation protocols to high-stakes domains such as
medical diagnostics and legal consultation, where the stan-
dardization of interpretability metrics will critically supportthe evolution of RAG+ reasoning systems toward robust and
trustworthy decision-assistance platforms.
6.3.2 Decision Support and Active Retrieval. The ex-
pansion of RAG+Reasoning frameworks into specialized
tasks has fostered two complementary research paradigms:
decision optimization and active retrieval. In the decision
optimization category, systems must leverage heterogeneous
structured data, rule bases, and objective functions to formu-
late optimal strategies. Representative systems like PlanRAG
formalize Decision Question Answering (Decision QA) tasks
targeting enterprise-level scenarios including supply chain
optimization, industrial resource allocation, and market price
regulation. These tasks require planning multimodal rea-
soning paths where models iteratively retrieve data from
relational and graph databases, integrate intricate business
rules, and iteratively refine decision-making paths through
replanning mechanisms. To evaluate such capabilities, the
Decision QA (DQA) benchmark creates dual database ver-
sions (MySQL and Neo4j) derived from economic systems
in strategy games, assessing cross-structured generalization.
The evaluation consists of a three-tier framework: the core
tier measures answer accuracy; the intermediate layer di-
agnoses error types to identify system bottlenecks; and the
foundational tier focuses on retrieval efficiency and the im-
pact of replanning frequency. This structured evaluation
framework not only tracks performance but also offers ac-
tionable insights for system refinement.
Conversely, the active retrieval evaluation addresses the
challenge of dynamically determining when and how to in-
voke retrieval under complex multimodal contexts. Unlike
rigid traditional RAG systems, UAR applies lightweight clas-
sifiers for fast, accurate triggers, improving performance in
time-sensitive or creative tasks. Tested on AR-Bench, it com-
bines binary trigger accuracy with GPT assessments, exact
matches, and human reviews, boosting adaptability across
diverse contexts.
Emerging trends in these evaluation paradigms indicate a
shift from static, rule-based frameworks to dynamic system
simulations, as exemplified by DQA’s use of game engine-
generated datasets to simulate realistic environments. Sim-
ilarly, active retrieval tasks progress from simple retrieval
trigger decisions toward collaborative multi-criteria decision-
making. Evaluation methodologies are concurrently evolv-
ing from singular performance metrics to multidimensional
matrices comprising core effectiveness, diagnostic error dis-
tributions, and economic cost measures.
7 Cost and Risk
Integrating reasoning into RAG systems is neither effort-
less nor purely beneficial. Recent trends have exaggerated
its advantages while downplaying the costs and risks. This
trade-off between performance and cost is crucial. This sec-
tion examines the expenses and misuse risks linked to adding

Synergizing RAG and Reasoning: A Systematic Review Conference’17, July 2017, Washington, DC, USA
Figure 8. From LLM to RAG and then to RAG+Reasoning, performance improvement comes with additional cost.
reasoning to RAG systems. As shown in Figure 8, the cost of
moving from LLM to RAG, then to RAG + Reasoning, incurs
an inevitable "invisible tax". Though often hidden by perfor-
mance gains, this cost is vital in assessing these methods’
overall practicality and efficiency.
The shift from LLM to RAG moves from simplicity to
enhanced knowledge handling by incorporating external
information. A basic LLM provides direct, efficient answers
with low latency and token use but is limited to pre-trained
knowledge, restricting complex or up-to-date queries. RAG
overcomes this by adding a vector database for external
retrieval, vastly expanding response scope and reliability.
However, this requires substantial data processing, storage,
and introduces higher latency and token costs due to data
chunking, encoding, indexing, and retrieval overhead.
Advancing from RAG to RAG + Reasoning adds multi-
step reasoning capabilities, enabling complex task handling,
autonomous decisions, and more context-aware responses
through intricate reasoning. This comes at the expense of
increased delays, token consumption, processing demands,
and greater complexity in system integration and mainte-
nance. The reasoning layer’s autonomy also brings opaque-
ness, unpredictability, and heightened security and reliability
risks. These challenges highlight the necessity of carefully
balancing effectiveness against costs when adopting RAG +
Reasoning in real-world applications.
7.1 Cost Trade-off in RAG+Reasoning
Figure 9 illustrates typical works combining RAG and Rea-
soning, showing retrieval and reasoning demands alongside
token consumption. While integrating dynamic knowledge
retrieval with multi-step reasoning greatly improves accu-
racy in more complex tasks, the resulting systemic costs are
often underestimated in research and practice. These costsgrow non-linearly, causing serious efficiency bottlenecks
in real-world use. The tradeoff between effectiveness and
efficiency stems from RAG+Reasoning’s architecture: multi-
stage task decoupling, dynamic path planning, and interme-
diate state preservation. These features improve reasoning
quality but trigger cascading increases in computational re-
sources, token usage, and reduced retrieval efficiency. This
section explores these implicit tradeoffs from the angles of
resource use, token consumption, and retrieval efficiency.
7.1.1 Non-Linear Growth of Computational Resources.
The RAG+Reasoning framework separates retrieval and rea-
soning into multiple stages, causing computational demands
to grow non-linearly. Dynamic chain-of-reasoning methods
execute multiple LLM generations and retrievals per infer-
ence, resulting in complexity far exceeding baseline models.
Fixed-length reasoning chains trigger repeated retrieval and
generation calls, increasing resource needs with task com-
plexity. More advanced techniques like MCTS-guided meth-
ods add rounds of candidate path generation and evaluation,
further multiplying runtime and memory usage on GPUs
compared to linear methods. Even simpler multi-step plan-
ning tasks incur much higher overhead than single-stage
retrieval models due to extra graph construction and analysis.
While this resource intensity improves inference accuracy, it
poses serious scalability challenges under limited resources
as computational costs grow superlinearly with model size,
retrieval chain length, and task complexity.
7.1.2 Implicit Token Inflation. Multi-step reasoning frame-
works inherently cause significant token inflation through it-
erative intermediate processes like thought chains, retrieved
documents, and verification feedback. Active learning setups
consolidate multiple intermediate results—retrieved docu-
ments, counterfactuals, multi-round validations—leading to

Conference’17, July 2017, Washington, DC, USA Gao et al.
Figure 9. Cost quadrant diagram of retrieval and reasoning requirements
token usage well beyond typical limits. Chain-based retrieval
also generates token bloat due to exhaustive candidate path
exploration. Iterative reasoning path selection, expansion,
and evaluation add heavy token overhead in tasks needing
deep reasoning chains involving extensive sequence genera-
tion and evaluation. Token usage grows exponentially with
task complexity and increases further when intermediate
reasoning favors depth or breadth. This inflation raises API
costs and memory demands, especially in long-text genera-
tion like Deep Research [106].
7.1.3 Marginal Decline in Retrieval Efficiency. Dynamic
retrieval improves knowledge precision but suffers diminish-
ing efficiency as task complexity increases. Adaptive meth-
ods reduce retrievals for simple tasks but still require mul-
tiple iterations for complex ones, adding significant over-
head compared to standard RAG. The tradeoff between re-
trieval quality and frequency further limits efficiency. High-
accuracy retrieval methods incur heavy computational and
time costs, negating their efficiency benefits. Even advanced
retrieval-trigger optimizations can’t fully remove this over-
head due to extra training and deployment costs [ 41]. This
natural efficiency ceiling highlights ongoing challenges in
balancing retrieval accuracy and resource use, especially in
large, complex tasks.7.1.4 Toward a Cost Model Framework. Against this
backdrop, the development of fine-grained cost models be-
comes a necessary precondition for balancing effectiveness
and efficiency. Existing evaluation metrics, which often rely
on single-task performance indicators (such as Exact Match
or F1) or coarse-grained runtime statistics, lack the compre-
hensiveness to jointly model computational resources, token
flow, and retrieval overhead. Consequently, they fail to quan-
tify the true tradeoffs in reasoning mechanisms. For instance,
while multi-hop reasoning may improve task accuracy, these
improvements are frequently offset by exponential growth
in token consumption and latency relative to baseline meth-
ods. A fine-grained cost model would enable researchers and
practitioners to more accurately evaluate the real benefits
of reasoning-centric frameworks while addressing the un-
derexplored interplay between computational cost and task
performance.
7.2 Potential Risk of Over-Thinking
In the process of developing deep thinking models, "over-
thinking" poses a key risk to system efficiency and reliabil-
ity [10,15,19,30,74,81], and this issue is further amplified
after combining with RAG. It appears as redundant reason-
ing steps, excessive validation of known conclusions, or un-
necessarily broad retrieval scopes, wasting computational

Synergizing RAG and Reasoning: A Systematic Review Conference’17, July 2017, Washington, DC, USA
resources, increasing error propagation, and degrading per-
formance. For example, in financial risk assessment, an LLM
with RAG might retrieve multiple similar market reports and
repeatedly verify the same economic indicators rather than
focusing on core risks, leading to delayed decisions. This
stems from an imbalance between reasoning and retrieval:
after accessing external knowledge, the model can enter
a "self-validation loop," repeatedly parsing overlapping or
contradictory documents. The generation module, seeking
reliability, may trigger further retrievals, creating a feedback
loop that worsens inefficiency. This issue is critical in real-
time systems like medical diagnosis, where over-retrieval of
irrelevant literature can delay urgent decisions.
Case studies show the impact of overthinking [ 74]. In
legal document interpretation, early reasoning errors can
amplify through the retrieval-generation loop, causing re-
trieval along incorrect paths and yielding illogical conclu-
sions. This error propagation is evident in systems like the
Search-o1 [ 51], where flawed information extraction mis-
guides subsequent reasoning. In industrial equipment man-
ual interpretation, overextended reasoning with highly simi-
lar documents risks obscuring critical parameter differences,
increasing procedural errors. These examples illustrate that
overthinking not only hampers knowledge integration but
also creates safety hazards in practical applications.
To mitigate these risks, researchers propose multiple opti-
mization frameworks. ReaRAG [ 49] limits reasoning chain
length and incorporates self-reflection to prune invalid branches.
A simple and effective way is to use a two-stage filtering pro-
cess, first narrowing documents by metadata, then validating
fragment relevance, reducing redundant information—for
instance, retrieving only relevant legal clauses rather than
entire regulatory texts. The DeepSeek R1 [ 26] applies rein-
forcement learning with distillation to penalize redundant
steps, cutting repeated formula validation in math proofs by
over 40%. These approaches transform open-ended reason-
ing into controlled, goal-directed processes, using methods
like attention weight analysis to measure information gain
or confidence functions to evaluate reasoning paths.
Current research balances constraints with model creativ-
ity. Knowledge graph-guided reasoning is tested in clinical
trials to prioritize key medical features over exhaustive liter-
ature retrieval [ 11]. Causal reasoning models aim to break
error chains; for example, in financial forecasting, causal
graphs restrict reasoning to logically relevant macroeco-
nomic links. Adaptive stopping strategies adjust reasoning
depth in customer service—simple queries use preset tem-
plates, complex issues activate multi-hop reasoning. These
advances reshape retrieval-augmented reasoning, with the
core challenge being to develop evaluation frameworks that
avoid both "cognitive stagnation" from excessive constraints
and "cognitive overload" from insufficient control.
Future progress will integrate cognitive science with com-
putational modeling. By mimicking human "intuition-verification"decision-making, LLMs could switch seamlessly between
rapid response and deep reasoning. In high-risk fields like
industrial fault diagnosis, such hybrid models can quickly
propose contingency plans after initial retrieval while ver-
ifying their validity through deeper analysis. This layered
approach reduces overthinking risks and offers a safe, con-
trollable path for applying LLMs in critical industries.
8 Practical Guide
The combination of RAG and Reasoning is not a one-size-fits-
all solution; it requires careful evaluation of each scenario’s
unique needs. As a rapidly evolving and relatively new field,
practical applications are still limited, making best practices
hard to define. This chapter abstracts and summarizes the key
traits of typical RAG+Reasoning application domains and
offers practical guidelines for system design based on these
features. It provides recommendations on leveraging RAG’s
strengths with Reasoning, highlighting priorities, pitfalls to
avoid, and current opportunities (Figure 10). The goal is to
promote wider adoption and effective use of this technology
in diverse, complex real-world settings.
8.1 Domain characteristics
As illustrated in the left part of Figure 10, we develop a seven-
dimensional feature system based on the three core stages
of RAG—query, retrieval, and generation—to systematically
analyze challenges and adaptation needs across various in-
dustries. The query stage emphasizes the complexity of in-
tent understanding and the demand for advanced reasoning,
recognizing that industries differ in query abstraction and
specificity; some require quickly capturing implicit, deep
intentions, while others need complex reasoning. Effective
preservation of original semantic meaning during under-
standing and reasoning is key to improving RAG perfor-
mance. Retrieval focuses on the system’s adaptability to di-
verse and dynamic knowledge sources, which vary from rich
multi-domain data to rapidly updating information; frequent
updates and fragmented knowledge present challenges that
demand effective integration to ensure consistent support
for generation. The generation stage requires high-quality
outputs, with strict control over hallucinations—especially
critical in sensitive fields like healthcare and law—along
with varying latency requirements for real-time or delayed
responses. Explainability and traceability at this stage are
essential for system credibility and serve as key evaluation
metrics. This comprehensive framework reveals technical
bottlenecks and guides improvements, and is applied to ana-
lyze four representative domains: finance, healthcare, law,
and personal assistants.
8.1.1 Finance. In the finance domain, user queries typi-
cally focus on structured needs like investment decisions and
risk forecasting. While intent understanding is moderately
complex, the system must perform advanced reasoning amid

Conference’17, July 2017, Washington, DC, USA Gao et al.
Figure 10. Practical guide to synergizing RAG and Reasoning
rapidly changing market conditions, relying heavily on exter-
nal knowledge and frequent updates. For example, portfolio
return forecasting integrates time series analysis, policy in-
terpretation, and cross-market reasoning. Retrieval demands
handling diverse data sources—real-time market data, an-
nual reports, and regulatory filings—with update cycles often
measured in minutes. During generation, strict latency and
hallucination control are crucial, as outputs must include
decision-making suggestions with full data traceability. In-
vestment research reports, for instance, require annotated
key indicators, their data sources, and computation logic
to ensure transparency and regulatory compliance. High la-
tency control and robust traceability are essential to maintain
transparency and adherence to financial regulations.
8.1.2 Healthcare. Healthcare queries involve complex med-
ical semantic parsing, often with ambiguous terms or incom-
plete symptoms. For example, "persistent chest pain with
shortness of breath" requires multi-hop reasoning across car-
diology, pulmonology, and emergency medicine. Retrieval
must integrate electronic health records, medical imaging,
and up-to-date clinical guidelines. In generation, hallucina-
tion tolerance is minimal—errors in drug dosages or proto-
cols risk malpractice. Therefore, accuracy, timeliness, and
explainability are paramount, with every decision step trace-
able and verifiable.
8.1.3 Legal Services. Legal consultations often require
interpreting statutes and citing cases, balancing precise le-
gal terms with natural language nuances. Retrieval depends
on structured, infrequently updated sources like case law
databases and local regulations. Generation demands accu-
racy—for instance, drafting contract clauses must precisely
cite specific statutes (e.g., Article 472 of the Civil Code) downto the paragraph level for traceability. Explainability is es-
sential, with traceability usually above 95%, and probabilistic
language avoided to comply with strict judicial documenta-
tion standards.
8.1.4 Personal Assistants. This domain features diverse,
dynamic user needs, including schedule management, real-
time navigation, and open-domain conversations. Accurate
intent disambiguation through contextual awareness is cru-
cial. Retrieval integrates fragmented sources like user behav-
ior logs, geolocation, and social media. Generation latency
varies: weather updates require sub-second responses, while
travel planning can tolerate 5+ seconds. Hallucination tol-
erance depends on context—creative outputs are acceptable
for recipes but not for flight information, which demands
full accuracy. This necessitates adaptive verification in the
RAG system. Though intent complexity is lower than in
healthcare or legal fields, the domain’s interaction diversity
requires heavy reliance on external knowledge and dynamic
balancing of latency and accuracy.
8.2 Do’s and Don’ts
Building on aforementioned domain characteristics, we fur-
ther identify six common scenarios, and derive technical
adaptation principles for each. This section outlines key op-
timization strategies (Do’s) and prohibitions (Don’ts) , to
guide the co-design of RAG and reasoning.
8.2.1 Structured Reasoning Scenarios. For scenarios
requiring multi-step logical decomposition and structured
knowledge dependency, such as portfolio return prediction ,
Chain-of-Thought (CoT) task decomposition and knowledge
graph (KG)-driven graph reasoning approaches should be

Synergizing RAG and Reasoning: A Systematic Review Conference’17, July 2017, Washington, DC, USA
employed. Complex problems should be broken into ver-
ifiable sub-tasks, such as coupling market trend analysis
with policy impact assessment, while leveraging knowledge
graph constraints to ensure logical completeness and au-
ditability. It is essential to incorporate a temporal validation
layer to cross-check the consistency of timestamp-sensitive
information (e.g., real-time market data or emergent regula-
tory policies) within a dynamic knowledge base. Approaches
that exclude retrieval-based verification of salient features
must be avoided, as they may lead to reasoning biases aris-
ing from the absence of structured knowledge anchors (e.g.,
critical indicators from financial statements). Furthermore,
the reasoning space of LLMs should be constrained within
domain-specific knowledge frameworks to prevent irrele-
vant or invalid deductions.
8.2.2 Dynamic Demand-Responsive Scenarios. For sce-
narios characterized by rapidly shifting demands and user
preference variability, such as itinerary planning and mul-
timodal interaction in personal assistant services , a dynamic
adaptation mechanism based on prompt engineering is rec-
ommended. By dynamically associating fragmented knowl-
edge units (e.g., user behavior history and real-time traffic
updates) with semantic templates and employing heuristic
rules for search-space pruning (e.g., prioritizing locally up-
dated information within the past 24 hours), the system can
balance contextual adaptability with response speed. Model
fine-tuning or reinforcement learning (RLHF/DPO)-based
strategy updates should be avoided due to their lengthy iter-
ative cycles and computational overhead, which cannot meet
real-time responsiveness requirements, such as millisecond-
grade reaction times for last-minute destination changes.
Lightweight caching architectures should be implemented
within the retrieval system, prioritizing frequently accessed
knowledge fragments, such as operating hours of popular
tourist attractions, to achieve an equilibrium between dy-
namism and stability.
8.2.3 Deterministic Decision-Making Scenarios. In sce-
narios requiring a single, reliable conclusion, such as clinical
diagnosis generation in the healthcare domain , a multi-level
deterministic assurance system should be established. Time-
validation layers can filter outdated knowledge (e.g., ther-
apies no longer approved), while field-sensitive retrieval
modules trigger predefined decision rules conforming to up-
to-date clinical guidelines (e.g., those codified within the
latest version of the International Classification of Diseases
[ICD]). Knowledge graph path constraints should restrict the
reasoning process to validated causal links within medical
logic (e.g., linking symptom patterns to laboratory test re-
sults within corroborated diagnostic pathways), thereby min-
imizing the likelihood of deviations from standard protocols.
Probabilistic exploration strategies that generate alternative
hypotheses (e.g., speculative differential diagnoses for atypi-
cal pneumonia) should be strictly disallowed to avoid clinicalmisjudgments. Additionally, delegating decision-making au-
thority to external classification models must be avoided to
maintain end-to-end explainability and a clear causal link in
the decision-making pipeline.
8.2.4 Time-Sensitive Scenarios. In tasks highly sensitive
to response delays, such as real-time risk warnings and trad-
ing decisions in the financial sector , heuristic rules should be
employed to prioritize indexing of frequently queried knowl-
edge units (e.g., volatility indices and liquidity indicators) at
the top of the search hierarchy. Directed retrieval expansion
strategies that preload potentially associated information
(e.g., contractual clauses of derivative instruments tied to
underlying assets) can further reduce latency in multi-turn
interactions. Monte Carlo Tree Search (MCTS) and other
sample-based algorithms are ill-suited for such scenarios
due to the excessive computational complexity caused by
branch expansion, rendering them infeasible within tight
time constraints (e.g., milliseconds). Similarly, the invocation
of complex mathematical solvers (e.g., numerical solutions
for stochastic differential equations) can introduce uncon-
trollable delays and should be replaced with lightweight rule-
based mechanisms (e.g., threshold-triggering mechanisms
based on historical volatility ranges).
8.2.5 Risk-Sensitive Scenarios. For scenarios with mini-
mal tolerance for errors, such as contract clause generation
and citation of judicial interpretations in the legal sector , a dual-
layer defensive mechanism must be employed. A pre-action
review layer should validate the compliance of generated
content with statutory standards (e.g., ensuring consistency
between liability clauses and Article 577 of the Civil Code),
while a reliability validation layer performs cross-referencing
validation across multiple sources (e.g., aligning Supreme
Court precedents with regional court guidelines) to resolve
potential conflicts. Retrieval systems must include version
control modules to track and update legal references (e.g., au-
tomatically flagging repealed local statutes). Unconstrained
reinforcement learning-based text generation methods must
be avoided, as their exploratory nature risks violating the
normative requirements of legal documents (e.g., generating
presumptive liability terms unsupported by judicial interpre-
tations). All decision-making actions must pass through de-
terministic rule engines to filter inadmissible outputs, and the
system should never execute decision actions autonomously,
such as generating legally binding arbitration notices with-
out oversight.
8.2.6 Complex Path Exploration Scenarios. In explo-
ration tasks involving multiple possible trajectories, such
asdifferential diagnosis and therapeutic pathway optimiza-
tion in medicine , weighted ranking search algorithms should
balance search depth and breadth. Knowledge graph topol-
ogy can guide prioritization (e.g., standard treatment proce-
dures for acute coronary syndrome), while Monte Carlo Tree

Conference’17, July 2017, Washington, DC, USA Gao et al.
Search can extend exploration into uncommon differential
paths (e.g., rare genetic metabolic disorders). Dynamic prun-
ing threshold functions should be designed (e.g., adjusting
the scope of differential diagnosis based on patient history)
to eliminate low-confidence hypotheses in real time, thereby
controlling computational scale. Brute-force searching of
all potential paths (e.g., concurrently testing hundreds of
pathogens for nonspecific symptoms) should be avoided to
prevent exponential computational scaling. Careful handling
of specific token triggers during retrieval (e.g., avoiding spu-
rious associations between "fever" and unrelated oncological
hyperthermia research) is critical to maintaining logical co-
herence in diagnostic reasoning.
8.3 Opportunity Points
Based on the Do’s and Don’ts of current technologies ana-
lyzed in the previous section, there remain numerous direc-
tions with substantial academic value and application poten-
tial that have yet to be fully explored. This section systemat-
ically discusses several promising opportunity points across
three dimensions: data and indexing ,models and methodolo-
gies, and application services .
8.3.1 Data and Indexing.
Cold-Hot Tiered Indexing and Dynamic Context Man-
agement. The challenge of managing massive and highly
heterogeneous data resources lies in devising an effective
cold-hot tiered indexing mechanism that prioritizes data ac-
cording to their frequency of use and importance. Such a
mechanism not only demands classification of data based on
timeliness and access frequency but also requires integration
with dynamic context management. This allows the system
to intelligently retrieve the most relevant data according to
the immediate context.
Moreover, a dynamically updated indexing mechanism
can mitigate the loss of data timeliness, which often leads to
deteriorated inference accuracy. By ensuring access to the
most recent and task-appropriate data, this approach reduces
redundancy and incorrect retrievals associated with static
indexing. When combined with automated task scheduling
and resource allocation strategies, fine-grained real-time
inference support can be achieved, significantly enhancing
the system’s overall efficiency.
Cross-Institution Knowledge Base Construction. The
construction of cross-institution or cross-domain knowledge
bases offers new opportunities for advancing RAG+Reasoning
research. At the core of large-scale cross-institutional knowl-
edge bases lies the optimization of data integration and shar-
ing mechanisms. This entails addressing challenges such
as data security and privacy while adopting standardized
data interfaces or leveraging federated learning paradigms
to enable multidimensional data integration.Through semantic alignment across multiple sources, en-
tity resolution, and concept abstraction, cross-institutional
knowledge can be transformed into authoritative and richly
contextualized knowledge bases. These enhanced reposito-
ries provide robust contextual support for reasoning tasks
and can deliver deeper insights in areas such as healthcare,
finance, and urban management.
Fine-Grained Layering and Confidence Grading. In
scenarios where retrieval and reasoning operate synchronously,
theinterpretability andreliability of generated outcomes are
paramount. Fine-grained layering of data and indices, along
with confidence grading of retrieval results, enables the sys-
tem to selectively use the most trustworthy and relevant
subsets of data during different stages of reasoning. This
approach fosters transparency and traceability in final deci-
sions or generative outputs.
For instance, in medical diagnosis scenarios, confidence
grading can initiate additional verification or expert review
in high-risk cases. In the legal domain, confidence layering
systematically presents key evidence and identifies sources
of uncertainty, reducing reasoning vulnerabilities and mini-
mizing the risk of erroneous conclusions caused by informa-
tion ambiguity.
8.3.2 Models and Methodologies.
Event-Driven Active Retrieval. Traditional retrieval mech-
anisms are predominantly passive. However, event-driven
active retrieval presents a promising exploration avenue.
By monitoring critical events, such as the injection of new
data, user interactions, or changes in external sensors, event-
triggered retrieval and reasoning processes can be initiated
to capture and respond to potential risks and opportunities in
real time. Integrating methodologies such as sequence-based
event detection or multitask-learning-based intent recog-
nition can facilitate automatic determination of when and
how to trigger retrieval actions. Iteratively optimizing these
processes contributes to a more efficient and continuous
reasoning loop.
Spatiotemporal-Aware Retrieval and Association. Many
applications, such as natural disaster monitoring, traffic flow
prediction, and inventory management in retail, exhibit strong
dependencies on temporal and spatial dimensions. By in-
corporating spatiotemporal-aware algorithms, retrieval pro-
cesses can prioritize or emphasize crucial documents ac-
cording to constraints tied to time and space. This not only
enhances timeliness but also improves the purposefulness
and accuracy of reasoning.
Furthermore, modeling the evolution of events within spa-
tiotemporal dimensions—when combined with semantic in-
dexing and vector-based retrieval mechanisms in RAG—can
enable more precise characterization and utilization of com-
plex spatiotemporal dynamics during reasoning.

Synergizing RAG and Reasoning: A Systematic Review Conference’17, July 2017, Washington, DC, USA
Multimodal Fusion in Retrieval and Reasoning. Mul-
timodal data (e.g., text, images, audio, video, and sensor data)
collectively constitute a richer contextual environment, of-
fering critical cues for reasoning tasks. However, existing
studies are often limited to the retrieval of single or a few data
modalities. Advancing research on multimodal fusion and
reasoning mechanisms under the RAG+Reasoning frame-
work has the potential to greatly enhance the system’s ca-
pacity for addressing complex queries.
The research focus lies in constructing cross-modal repre-
sentation learning and alignment methods, enabling unified
representations of the same entities or events across differ-
ent modalities. During retrieval, confidence scores for each
modality can be integrated into a comprehensive ranking
process, culminating in multimodal-informed joint decision-
making during reasoning. This approach not only improves
contextual understanding in complex tasks but also broadens
the application scope of RAG technologies in scenarios such
as expert systems and autonomous driving, where sensory
integration and interpretation are critical.
Dynamic Risk Propagation Modeling and Manage-
ment. The tight coupling of retrieval and reasoning with
multi-stage decision-making inevitably introduces risk prop-
agation issues. Misjudgments of high-risk or low-confidence
documents during upstream retrieval are often inherited
by downstream reasoning processes, amplifying uncertain-
ties and increasing error margins. To address this, dynamic
risk modeling should be embedded within retrieval work-
flows, enabling risk quantification, tracking, and manage-
ment at multiple stages. When necessary, risk mitigation
mechanisms or process rollbacks can be triggered, creating
a closed-loop correction framework.
Incorporating strategies for analyzing and managing risk
propagation is not only a technical challenge but also a mat-
ter of system deployment and standardization. In high-stakes
domains such as healthcare and financial risk management,
establishing comprehensive safety standards and compliance
protocols will be crucial. These protocols should treat dy-
namic risk propagation management as a critical component
of evaluating and iterating knowledge retrieval and reason-
ing systems.
8.3.3 Application Services.
Validation of Logical Chain Completeness. While RAG
with Reasoning can provide partially interpretable reasoning
outputs, verifying the completeness of logical chains remains
a challenge. Future research could integrate formal verifica-
tion or symbolic reasoning techniques to ensure consistency
and completeness across key reasoning nodes and intermedi-
ate conclusions. This would prevent logical gaps or illogical
leaps in reasoning, offering robust regulatory support for
high-stakes industries such as law and finance.Intervenable Generation During Reasoning. Contem-
porary Agentic RAG often operate as "black boxes," rendering
external interventions nearly impossible during generative
reasoning tasks. However, providing mechanisms for human
intervention—such as through visualization or interactive
interfaces—could enable experts or users to perform manual
corrections, initialize prior knowledge, or modify interim
assumptions during the reasoning process. This would sub-
stantially enhance the system’s flexibility and safety.
Specifically, intervenable generation allows not only post
hoc error corrections but also proactive identification and
rectification of potential risks or biases at earlier stages. In-
teractive interpretable reasoning platforms or visualization
tools grounded in knowledge graphs could empower users
to scrutinize and influence reasoning workflows, thereby
enhancing confidence and control in decision-making pro-
cesses across diverse domains.
Risk Decision Interception Firewalls. In closed-loop
automated tasks such as algorithmic trading or medical di-
agnostic decision-making, erroneous reasoning outputs can
lead to catastrophic outcomes. To mitigate such risks, the sys-
tem architecture should incorporate risk decision interception
firewalls , which perform multidimensional validations at crit-
ical reasoning nodes or prior to outputting decisions. When
confidence levels or high-risk indicators breach thresholds,
these firewalls can block decision outputs or escalate them
for stricter human review.
This mechanism serves as a “final line of defense” for
RAG+Reasoning systems, ensuring decision security in large-
scale automated information networks. It also provides a
robust foundation for compliance and regulatory auditing,
enabling safer deployment in critical applications.
Edge-Cloud Collaborative Retrieval and Reasoning.
With the rapid development of IoT and 5G technologies,
many scenarios demand on-site data collection and prelim-
inary processing on edge devices, followed by high-level
retrieval and reasoning tasks on cloud platforms. Efficiently
partitioning tasks, allocating resources, and maintaining con-
sistency between indexes and models across the edge-cloud
continuum represent critical research directions.
Leveraging techniques such as lightweight model compres-
sion, distributed index synchronization, and communication
optimization can ensure fast reasoning while maximizing
resource utilization. Edge-cloud collaborative solutions are
particularly impactful for real-time industrial monitoring
and smart city applications, reducing network latency and
bandwidth bottlenecks while ensuring accurate and timely
inference outputs.
In summary, RAG+Reasoning systems present many un-
tapped opportunities across various dimensions. Further re-
search and practical validation could greatly improve their
use in complex, high-risk scenarios while fueling new growth
in GenAI.

Conference’17, July 2017, Washington, DC, USA Gao et al.
9 Future Trends
In this chapter, we summarize four major trends in techno-
logical advancements based on current research, aiming to
elucidate and guide the potential future directions of RAG.
9.1 The Integration of RAG and Graph
Recent developments have witnessed a growing synergy
between RAG systems and graph-based approaches. The
intrinsic benefits of graph structures, such as explicit logical
relationships and knowledge indexing, have enabled new
paradigms for addressing challenges in global reasoning,
dynamic data management, and personalized services within
RAG systems.
Knowledge Organization.
Graph-structured knowledge organization frameworks of-
fer a powerful alternative to traditional vector-based retrieval
methods, excelling in modeling complex relationships and
supporting global reasoning. For example, GraphRAG [ 18]
combines hierarchical graph indexing with community de-
tection to extract entity relationship networks from text
corpora, enabling large-scale thematic analysis through hi-
erarchical summaries. Building on this, PIKE [ 82] introduces
a multi-level heterogeneous knowledge graph that orga-
nizes documents, semantic segments, and refined knowledge
units into a three-layer hierarchy, improving extraction accu-
racy and multi-hop reasoning via atomized knowledge con-
struction and task decomposition. For dynamic personaliza-
tion, EMG-RAG [ 89] features a three-layer Editable Memory
Graph architecture that structures memory data by ontology
classification, subclass, and entity relationships, using rein-
forcement learning to enable real-time updates and multidi-
mensional queries. Together, these advances leverage graph
topologies to address the limitations of conventional RAG
systems—such as one-dimensional representation and weak
contextual links—enabling multilevel reasoning from local
fact retrieval to global thematic summarization and forming
a foundation for interpretable, adaptive RAG systems.
Symbolic Reasoning. Graph-structured symbolic rea-
soning methods leverage the multi-hop reasoning power of
Knowledge Graphs (KG) to better manage complex seman-
tic and logical relationships. Frameworks like HippoRAG2
and the Think-on-Graph (ToG) [ 60] series exemplify this.
HippoRAG2 [ 28] builds open knowledge graphs and uses
personalized PageRank with a dense-sparse coding approach
inspired by brain memory, boosting performance in factual
memory, semantic understanding, and multi-hop reasoning.
Likewise, ToG-2 combines iterative retrieval of knowledge
graphs and documents, using relationship discovery, entity
pruning, and context-driven graph searches to integrate fine-
grained information from unstructured text, enhancing im-
plicit relationship detection.
Task Planning. Graph-based task planning in RAG sys-
tems enhances complex problem-solving by overcoming thelimitations of traditional linear workflows, which struggle
with multi-step or multimodal reasoning. These approaches
build dynamic knowledge graphs, like Mind Maps, to explic-
itly model logical dependencies and context. For instance,
the Agentic Reasoning [ 92] transforms reasoning chains into
graph structures for entity extraction, relation identification,
and community clustering, enabling dynamic path tracking
and optimized retrieval, excelling in tasks like doctoral-level
GPQA [ 67]. Collaborative frameworks such as Co-STORM
extend this to multi-agent scenarios, representing queries,
tool calls, and knowledge integration as traversable graph
nodes to support task decomposition and adaptive reasoning.
Tool Usage and Management. Graph-enhanced approaches
to tool management overcome limitations of traditional de-
pendency modeling by effectively capturing complex rela-
tionships like parameter passing, functional collaboration,
and resource management. Graph RAG-Tool Fusion [ 57]
models tools as graph nodes within a dual-layer architecture
of core system APIs and domain-specific tools, encoding di-
rect and indirect dependencies as edges. It uses a two-stage
retrieval process: vector-based tool retrieval followed by
a graph-based depth-first search to assemble dependency-
compliant toolsets.
9.2 Multi-Model Collaboration
Multi-model collaboration has emerged as a pivotal strategy
for enhancing task complexity handling and domain adapt-
ability in RAG systems [ 13]. By integrating the strengths of
different models, this approach achieves optimized perfor-
mance. For example, the CR-Planner [52] combines general-
purpose generation models (e.g., GPT-4) with domain-specific
critic models (e.g., Llama-3-8B). This hybrid system dynam-
ically orchestrates subgoal planning and execution evalua-
tion, utilizing MCTS to generate high-quality training data.
Similarly, UAR [ 14]employs intent-aware and knowledge-
requirement classifiers to dynamically trigger retrieval, de-
coupling lightweight classification tasks from resource-intensive
decoding operations of LLMs. Furthermore, Adaptive-RAG [41]
deploys small-complexity classifiers to route queries into dif-
ferent levels of processing strategies, balancing response
speed for simple queries with deep reasoning for complex
ones. These strategies form a closed "generation-evaluation"loop,
leveraging complementary strengths across models to achieve
improved accuracy and computational efficiency.
9.3 Multi-Modal Collaboration
The breakthrough in Chain-of-Thought (CoT) capabilities of
language models has catalyzed the transition of multimodal
reasoning from perceptual-level integration to cognitive-
level reasoning, promoting Multimodal Collaborative Rea-
soning as a key trend [ 4] By deeply integrating the logical
reasoning capabilities of language models with the spatial-
semantic representation of multimodal data, it significantly
enhances information synthesis in complex scenarios [ 2].

Synergizing RAG and Reasoning: A Systematic Review Conference’17, July 2017, Washington, DC, USA
For instance, in the medical domain, multimodal RAG sys-
tems such as MedCoT [ 56]utilize hierarchical expert systems
to integrate CT imaging and pathology reports, enabling
knowledge graph validation of diagnostic hypotheses and re-
ducing misdiagnosis risks. Future research will likely focus
on robust cross-modal knowledge alignment, progressive
knowledge distillation, and adaptive reasoning frameworks.
9.4 Customized Reinforcement Learning
The application of reinforcement learning (RL) in RAG sys-
tems has become instrumental in improving module coordi-
nation and enhancing overall efficiency. Recent studies focus
on designing reward mechanisms tailored to the specific
needs of RAG systems. Frameworks such as RAG-Gym [ 96]
andDeepRAG [ 24]model reasoning processes using Markov
Decision Processes and introduce fine-grained process super-
vision mechanisms. Additionally, ReARTeR [ 49]andSmar-
tRAG [ 20]incorporate trust-aware reward strategies and end-
to-end policy optimization to achieve superior accuracy and
robustness. Opportunities remain for further exploring auto-
mated reward modeling with LLMs to facilitate fine-grained
supervision.
10 Conclusion
This paper has systematically reviewed the synergistic in-
tegration of Retrieval-Augmented Generation (RAG) and
reasoning, providing a formal definition of reasoning within
the RAG framework as a structured, multi-step, goal-driven
process that dynamically combines parametric and retrieved
knowledge to address complex problems.
We presented a comprehensive taxonomy covering the
purposes, collaboration paradigms, and implementation meth-
ods underlying RAG+Reasoning systems. The synergy en-
ables more precise retrieval informed by logical analysis and
enhances reasoning with contextually relevant, up-to-date
knowledge beyond parametric limitations.
While the enhanced reasoning capabilities allow tackling
complex knowledge-intensive tasks such as deep research,
expert-level problem solving, and domain-specific decision
support, practical challenges remain. These include com-
putational and token costs that grow non-linearly, risks of
overthinking leading to inefficiency and error propagation,
and the lack of evaluation frameworks that effectively assess
intermediate reasoning quality alongside final results.
To bridge the gap from theory to real-world application,
we proposed practical design guidelines tailored to diverse
domains like finance, healthcare, law, and personal assistants,
emphasizing adaptability to heterogeneous, dynamic knowl-
edge sources and strict requirements for output reliability
and traceability.
Finally, we identified promising directions for future re-
search, including graph-structured knowledge integration,multimodal and multi-model collaborative reasoning archi-
tectures, and advanced reinforcement learning techniques
for optimizing retrieval-reasoning workflows.
Overall, this work establishes both a theoretical founda-
tion and practical roadmap to drive the development of next-
generation RAG+Reasoning systems capable of robust, trans-
parent, and efficient cognition, paving the way for impactful
applications across academia and industry.
References
[1]Abdelrahman Abdallah, Bhawna Piryani, Jamshid Mozafari, Mo-
hammed Ali, and Adam Jatowt. 2025. Rankify: A comprehensive
python toolkit for retrieval, re-ranking, and retrieval-augmented
generation. arXiv preprint arXiv:2502.02464 (2025).
[2]Mohammad Mahdi Abootorabi, Amirhosein Zobeiri, Mahdi De-
hghani, Mohammadali Mohammadkhani, Bardia Mohammadi, Omid
Ghahroodi, Mahdieh Soleymani Baghshah, and Ehsaneddin Asgari.
2025. Ask in Any Modality: A Comprehensive Survey on Multimodal
Retrieval-Augmented Generation. arXiv preprint arXiv:2502.08826
(2025).
[3]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh
Hajishirzi. 2023. Self-rag: Learning to retrieve, generate, and critique
through self-reflection. In The Twelfth International Conference on
Learning Representations .
[4]Jing Bi, Susan Liang, Xiaofei Zhou, Pinxin Liu, Junjia Guo, Yunlong
Tang, Luchuan Song, Chao Huang, Guangyu Sun, Jinxi He, et al .2025.
Why Reasoning Matters? A Survey of Advancements in Multimodal
Reasoning (v1). arXiv preprint arXiv:2504.03151 (2025).
[5]Yuxi Bi, Yunfan Gao, and Haofen Wang. 2025. StePO-Rec: Towards
Personalized Outfit Styling Assistant via Knowledge-Guided Multi-
Step Reasoning. arXiv preprint arXiv:2504.09915 (2025).
[6]Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu,
Fan Yang, Zenan Zhou, Weipeng Chen, Haofen Wang, Jeff Z Pan, et al .
2025. Learning to Reason with Search for LLMs via Reinforcement
Learning. arXiv preprint arXiv:2503.19470 (2025).
[7]Peter Baile Chen, Yi Zhang, Michael Cafarella, and Dan Roth.
2025. Can we Retrieve Everything All at Once? ARM: An
Alignment-Oriented LLM-based Retrieval Method. arXiv preprint
arXiv:2501.18539 (2025).
[8]Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiannan Guan,
Peng Wang, Mengkang Hu, Yuhang Zhou, Te Gao, and Wangxiang
Che. 2025. Towards reasoning era: A survey of long chain-of-thought
for reasoning large language models. arXiv preprint arXiv:2503.09567
(2025).
[9]Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma,
Jianyu Xu, Xinyi Wang, and Tony Xia. 2023. Theoremqa: A theorem-
driven question answering dataset. arXiv preprint arXiv:2305.12524
(2023).
[10] Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian
Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, et al .
2024. Do not think that much for 2+ 3=? on the overthinking of
o1-like llms. arXiv preprint arXiv:2412.21187 (2024).
[11] Yixiang Chen, Penglei Sun, Xiang Li, and Xiaowen Chu. 2025. MRD-
RAG: Enhancing Medical Diagnosis with Multi-Round Retrieval-
Augmented Generation. arXiv preprint arXiv:2504.07724 (2025).
[12] Yiqun Chen, Lingyong Yan, Weiwei Sun, Xinyu Ma, Yi Zhang,
Shuaiqiang Wang, Dawei Yin, Yiming Yang, and Jiaxin Mao. 2025.
Improving Retrieval-Augmented Generation through Multi-Agent
Reinforcement Learning. arXiv preprint arXiv:2501.15228 (2025).
[13] Zhijun Chen, Jingzheng Li, Pengpeng Chen, Zhuoran Li, Kai Sun,
Yuankai Luo, Qianren Mao, Dingqi Yang, Hailong Sun, and Philip S
Yu. 2025. Harnessing Multiple Large Language Models: A Survey on

Conference’17, July 2017, Washington, DC, USA Gao et al.
LLM Ensemble. arXiv preprint arXiv:2502.18036 (2025).
[14] Qinyuan Cheng, Xiaonan Li, Shimin Li, Qin Zhu, Zhangyue Yin,
Yunfan Shao, Linyang Li, Tianxiang Sun, Hang Yan, and Xipeng Qiu.
2024. Unified active retrieval for retrieval augmented generation.
arXiv preprint arXiv:2406.12534 (2024).
[15] Alejandro Cuadron, Dacheng Li, Wenjie Ma, Xingyao Wang, Yichuan
Wang, Siyuan Zhuang, Shu Liu, Luis Gaspar Schroeder, Tian Xia,
Huanzhi Mao, et al .2025. The Danger of Overthinking: Examining
the Reasoning-Action Dilemma in Agentic Tasks. arXiv preprint
arXiv:2502.08235 (2025).
[16] Alan Dao and Thinh Le. 2025. ReZero: Enhancing LLM search ability
by trying one-more-time. arXiv:2504.11001 [cs.CL] https://arxiv.org/
abs/2504.11001
[17] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettle-
moyer. 2023. Qlora: Efficient finetuning of quantized llms. Advances
in neural information processing systems 36 (2023), 10088–10115.
[18] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao,
Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Os-
azuwa Ness, and Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization. arXiv preprint
arXiv:2404.16130 (2024).
[19] Chenrui Fan, Ming Li, Lichao Sun, and Tianyi Zhou. 2025. Missing
Premise exacerbates Overthinking: Are Reasoning Models losing
Critical Thinking Skill? arXiv preprint arXiv:2504.06514 (2025).
[20] Jingsheng Gao, Linxu Li, Weiyuan Li, Yuzhuo Fu, and Bin Dai. 2024.
SmartRAG: Jointly Learn RAG-Related Tasks From the Environment
Feedback. arXiv preprint arXiv:2410.18141 (2024).
[21] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi
Bi, Yi Dai, Jiawei Sun, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey. arXiv preprint
arXiv:2312.10997 (2023).
[22] Yunfan Gao, Yun Xiong, Meng Wang, and Haofen Wang. 2024. Mod-
ular rag: Transforming rag systems into lego-like reconfigurable
frameworks. arXiv preprint arXiv:2407.21059 (2024).
[23] Zengyi Gao, Yukun Cao, Hairu Wang, Ao Ke, Yuan Feng, Xike Xie,
and S Kevin Zhou. 2025. FRAG: A Flexible Modular Framework for
Retrieval-Augmented Generation based on Knowledge Graphs. arXiv
preprint arXiv:2501.09957 (2025).
[24] Xinyan Guan, Jiali Zeng, Fandong Meng, Chunlei Xin, Yaojie Lu,
Hongyu Lin, Xianpei Han, Le Sun, and Jie Zhou. 2025. DeepRAG:
Thinking to Retrieval Step by Step for Large Language Models. arXiv
preprint arXiv:2502.01142 (2025).
[25] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang,
Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al .2025.
Deepseek-r1: Incentivizing reasoning capability in llms via reinforce-
ment learning. arXiv preprint arXiv:2501.12948 (2025).
[26] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang,
Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al .2025.
Deepseek-r1: Incentivizing reasoning capability in llms via reinforce-
ment learning. arXiv preprint arXiv:2501.12948 (2025).
[27] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2024.
Lightrag: Simple and fast retrieval-augmented generation. (2024).
[28] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu
Su. 2025. From RAG to Memory: Non-Parametric Continual Learning
for Large Language Models. arXiv preprint arXiv:2502.14802 (2025).
[29] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai,
Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al .
2024. Olympiadbench: A challenging benchmark for promoting agi
with olympiad-level bilingual multimodal scientific problems. arXiv
preprint arXiv:2402.14008 (2024).
[30] Yancheng He, Shilong Li, Jiaheng Liu, Weixun Wang, Xingyuan Bu,
Ge Zhang, Zhongyuan Peng, Zhaoxiang Zhang, Zhicheng Zheng,
Wenbo Su, et al .2025. Can Large Language Models Detect Errors in
Long Chain-of-Thought Reasoning? arXiv preprint arXiv:2502.19361(2025).
[31] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko
Aizawa. 2020. Constructing a multi-hop qa dataset for comprehensive
evaluation of reasoning steps. arXiv preprint arXiv:2011.01060 (2020).
[32] Yubin Hong, Chaofan Li, Jingyi Zhang, and Yingxia Shao. 2025. FG-
RAG: Enhancing Query-Focused Summarization with Context-Aware
Fine-Grained Graph RAG. arXiv preprint arXiv:2504.07103 (2025).
[33] SU Hongjin, Howard Yen, Mengzhou Xia, Weijia Shi, Niklas Muen-
nighoff, Han-yu Wang, Liu Haisu, Quan Shi, Zachary S Siegel, Michael
Tang, et al .2024. BRIGHT: A Realistic and Challenging Benchmark
for Reasoning-Intensive Retrieval. In The Thirteenth International
Conference on Learning Representations .
[34] Sheryl Hsu, Omar Khattab, Chelsea Finn, and Archit Sharma. 2024.
Grounding by trying: Llms with reinforcement learning-enhanced
retrieval. arXiv preprint arXiv:2410.23214 (2024).
[35] Jian Hu. 2025. REINFORCE++: A Simple and Efficient Approach for
Aligning Large Language Models. arXiv preprint arXiv:2501.03262
(2025).
[36] Yunhai Hu, Yilun Zhao, Chen Zhao, and Arman Cohan. 2025. MCTS-
RAG: Enhancing Retrieval-Augmented Generation with Monte Carlo
Tree Search. arXiv preprint arXiv:2503.20757 (2025).
[37] Fantine Huot, Reinald Kim Amplayo, Jennimaria Palomaki, Al-
ice Shoshana Jakobovits, Elizabeth Clark, and Mirella Lapata. 2024.
Agents’ Room: Narrative Generation through Multi-step Collabora-
tion. arXiv preprint arXiv:2410.02603 (2024).
[38] Shayekh Bin Islam, Md Asib Rahman, KSM Hossain, Enamul Hoque,
Shafiq Joty, and Md Rizwan Parvez. 2024. Open-rag: Enhanced
retrieval-augmented reasoning with open-source large language mod-
els.arXiv preprint arXiv:2410.01782 (2024).
[39] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed
El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel,
Alex Carney, et al .2024. Openai o1 system card. arXiv preprint
arXiv:2412.16720 (2024).
[40] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun
Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Sto-
ica. 2024. Livecodebench: Holistic and contamination free evaluation
of large language models for code. arXiv preprint arXiv:2403.07974
(2024).
[41] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and
Jong C Park. 2024. Adaptive-rag: Learning to adapt retrieval-
augmented large language models through question complexity.
arXiv preprint arXiv:2403.14403 (2024).
[42] Pengcheng Jiang. 2025. DeepRetrieval: Powerful Query Genera-
tion for Information Retrieval with Reinforcement Learning. arXiv
preprint arXiv:2503.00223 (2025).
[43] Yucheng Jiang, Yijia Shao, Dekun Ma, Sina J Semnani, and Monica S
Lam. 2024. Into the unknown unknowns: Engaged human learning
through participation in language model agent conversations. arXiv
preprint arXiv:2408.15232 (2024).
[44] Yucheng Jiang, Yijia Shao, Dekun Ma, Sina J Semnani, and Monica S
Lam. 2024. Into the unknown unknowns: Engaged human learning
through participation in language model agent conversations. arXiv
preprint arXiv:2408.15232 (2024).
[45] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane
Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023.
Active retrieval augmented generation. In Proceedings of the 2023
Conference on Empirical Methods in Natural Language Processing .
7969–7992.
[46] Ashutosh Joshi, Sheikh Muhammad Sarwar, Samarth Varshney,
Sreyashi Nag, Shrivats Agrawal, and Juhi Naik. 2024. REAPER: Rea-
soning based retrieval planning for complex RAG systems. In Pro-
ceedings of the 33rd ACM International Conference on Information and
Knowledge Management . 4621–4628.

Synergizing RAG and Reasoning: A Systematic Review Conference’17, July 2017, Washington, DC, USA
[47] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael
Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polo-
sukhin, Jacob Devlin, Kenton Lee, et al .2019. Natural questions:
a benchmark for question answering research. Transactions of the
Association for Computational Linguistics 7 (2019), 453–466.
[48] Myeonghwa Lee, Seonho An, and Min-Soo Kim. 2024. PlanRAG: A
plan-then-retrieval augmented generation for generative large lan-
guage models as decision makers. In Proceedings of the 2024 Conference
of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Papers) .
6537–6555.
[49] Zhicheng Lee, Shulin Cao, Jinxin Liu, Jiajie Zhang, Weichuan Liu, Xi-
aoyin Che, Lei Hou, and Juanzi Li. 2025. ReaRAG: Knowledge-guided
Reasoning Enhances Factuality of Large Reasoning Models with Itera-
tive Retrieval Augmented Generation. arXiv preprint arXiv:2503.21729
(2025).
[50] Jinzheng Li, Jingshu Zhang, Hongguang Li, and Yiqing Shen. 2024.
An Agent Framework for Real-Time Financial Information Searching
with Large Language Models. arXiv preprint arXiv:2502.15684 (2024).
[51] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou,
Yutao Zhu, Peitian Zhang, and Zhicheng Dou. 2025. Search-o1:
Agentic search-enhanced large reasoning models. arXiv preprint
arXiv:2501.05366 (2025).
[52] Xingxuan Li, Weiwen Xu, Ruochen Zhao, Fangkai Jiao, Shafiq Joty,
and Lidong Bing. 2024. Can We Further Elicit Reasoning in LLMs?
Critic-Guided Planning with Retrieval-Augmentation for Solving
Challenging Tasks. arXiv preprint arXiv:2410.01428 (2024).
[53] Xingxuan Li, Weiwen Xu, Ruochen Zhao, Fangkai Jiao, Shafiq Joty,
and Lidong Bing. 2024. Can We Further Elicit Reasoning in LLMs?
Critic-Guided Planning with Retrieval-Augmentation for Solving
Challenging Tasks. arXiv preprint arXiv:2410.01428 (2024).
[54] Zhuoqun Li, Haiyang Yu, Xuanang Chen, Hongyu Lin, Yaojie Lu,
Fei Huang, Xianpei Han, Yongbin Li, and Le Sun. 2025. Deepsolu-
tion: Boosting complex engineering solution design via tree-based
exploration and bi-point thinking. arXiv preprint arXiv:2502.20730
(2025).
[55] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards,
Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever,
and Karl Cobbe. 2023. Let’s verify step by step. In The Twelfth Inter-
national Conference on Learning Representations .
[56] Jiaxiang Liu, Yuan Wang, Jiawei Du, Joey Tianyi Zhou, and Zuozhu
Liu. 2024. Medcot: Medical chain of thought via hierarchical expert.
arXiv preprint arXiv:2412.13736 (2024).
[57] Elias Lumer, Pradeep Honaganahalli Basavaraju, Myles Mason,
James A Burke, and Vamse Kumar Subbiah. 2025. Graph RAG-Tool
Fusion. arXiv preprint arXiv:2502.07223 (2025).
[58] Haoran Luo, Yikai Guo, Qika Lin, Xiaobao Wu, Xinyu Mu, Wenhao
Liu, Meina Song, Yifan Zhu, Luu Anh Tuan, et al .2025. KBQA-o1:
Agentic Knowledge Base Question Answering with Monte Carlo Tree
Search. arXiv preprint arXiv:2501.18922 (2025).
[59] Yuanjie Lyu, Zhiyu Li, Simin Niu, Feiyu Xiong, Bo Tang, Wenjin Wang,
Hao Wu, Huanyong Liu, Tong Xu, and Enhong Chen. 2025. Crud-rag:
A comprehensive chinese benchmark for retrieval-augmented gen-
eration of large language models. ACM Transactions on Information
Systems 43, 2 (2025), 1–32.
[60] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao
Yang, Jiaxin Mao, and Jian Guo. 2024. Think-on-Graph 2.0: Deep and
Faithful Large Language Model Reasoning with Knowledge-guided
Retrieval Augmented Generation. arXiv preprint arXiv:2407.10805
(2024).
[61] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan.
2023. Query rewriting in retrieval-augmented large language models.
InProceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing . 5303–5315.[62] Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun,
and Thomas Scialom. 2023. Gaia: a benchmark for general ai assis-
tants. In The Twelfth International Conference on Learning Representa-
tions .
[63] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-
Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel
Candès, and Tatsunori Hashimoto. 2025. s1: Simple test-time scaling.
arXiv preprint arXiv:2501.19393 (2025).
[64] Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez.
2024. Gorilla: Large language model connected with massive apis.
Advances in Neural Information Processing Systems 37 (2024), 126544–
126565.
[65] Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Ma-
jid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir
Karpukhin, Jean Maillard, et al .2020. KILT: a benchmark for knowl-
edge intensive language tasks. arXiv preprint arXiv:2009.02252 (2020).
[66] Pouya Pezeshkpour and Estevam Hruschka. 2025. Insight-RAG: En-
hancing LLMs with Insight-Driven Augmentation. arXiv preprint
arXiv:2504.00187 (2025).
[67] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty,
Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R
Bowman. 2024. Gpqa: A graduate-level google-proof q&a benchmark.
InFirst Conference on Language Modeling .
[68] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan,
and Weizhu Chen. 2023. Enhancing retrieval-augmented large lan-
guage models with iterative retrieval-generation synergy. arXiv
preprint arXiv:2305.15294 (2023).
[69] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song,
Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al .2024.
Deepseekmath: Pushing the limits of mathematical reasoning in open
language models. arXiv preprint arXiv:2402.03300 (2024).
[70] Quan Shi, Michael Tang, Karthik Narasimhan, and Shunyu Yao. 2024.
Can Language Models Solve Olympiad Programming? arXiv preprint
arXiv:2404.10952 (2024).
[71] Quan Shi, Michael Tang, Karthik Narasimhan, and Shunyu Yao. 2024.
Can Language Models Solve Olympiad Programming? arXiv preprint
arXiv:2404.10952 (2024).
[72] Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen,
Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. 2025. R1-Searcher:
Incentivizing the Search Capability in LLMs via Reinforcement Learn-
ing. arXiv preprint arXiv:2503.05592 (2025).
[73] Sakhinana Sagar Srinivas and Venkataramana Runkana. 2025. Scal-
ing Test-Time Inference with Policy-Optimized, Dynamic Retrieval-
Augmented Generation via KV Caching and Decoding. arXiv preprint
arXiv:2504.01281 (2025).
[74] Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi
Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Hanjie Chen, Xia Hu,
et al.2025. Stop overthinking: A survey on efficient reasoning for
large language models. arXiv preprint arXiv:2503.16419 (2025).
[75] Zhongxiang Sun, Qipeng Wang, Weijie Yu, Xiaoxue Zang, Kai Zheng,
Jun Xu, Xiao Zhang, Song Yang, and Han Li. 2025. ReARTeR: Retrieval-
Augmented Reasoning with Trustworthy Process Rewarding. arXiv
preprint arXiv:2501.07861 (2025).
[76] Alon Talmor and Jonathan Berant. 2018. The web as a knowledge-base
for answering complex questions. arXiv preprint arXiv:1803.06643
(2018).
[77] Hieu Tran, Zonghai Yao, Junda Wang, Yifan Zhang, Zhichao Yang, and
Hong Yu. 2024. RARE: Retrieval-Augmented Reasoning Enhancement
for Large Language Models. arXiv preprint arXiv:2412.02830 (2024).
[78] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish
Sabharwal. 2022. Interleaving retrieval with chain-of-thought rea-
soning for knowledge-intensive multi-step questions. arXiv preprint
arXiv:2212.10509 (2022).

Conference’17, July 2017, Washington, DC, USA Gao et al.
[79] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish
Sabharwal. 2022. MuSiQue: Multihop Questions via Single-hop Ques-
tion Composition. Transactions of the Association for Computational
Linguistics 10 (2022), 539–554.
[80] Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason
Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, et al .2023.
Freshllms: Refreshing large language models with search engine
augmentation. arXiv preprint arXiv:2310.03214 (2023).
[81] Ante Wang, Linfeng Song, Ye Tian, Dian Yu, Haitao Mi, Xiangyu
Duan, Zhaopeng Tu, Jinsong Su, and Dong Yu. 2025. Don’t Get
Lost in the Trees: Streamlining LLM Reasoning by Overcoming Tree
Search Exploration Pitfalls. arXiv preprint arXiv:2502.11183 (2025).
[82] Jinyu Wang, Jingjing Fu, Rui Wang, Lei Song, and Jiang Bian. 2025.
PIKE-RAG: sPecIalized KnowledgE and Rationale Augmented Gener-
ation. arXiv preprint arXiv:2501.11551 (2025).
[83] Liang Wang, Haonan Chen, Nan Yang, Xiaolong Huang, Zhicheng
Dou, and Furu Wei. 2025. Chain-of-Retrieval Augmented Generation.
arXiv preprint arXiv:2501.14342 (2025).
[84] Ruobing Wang, Daren Zha, Shi Yu, Qingfei Zhao, Yuxuan Chen,
Yixuan Wang, Shuo Wang, Yukun Yan, Zhenghao Liu, Xu Han, et al .
2024. Retriever-and-Memory: Towards Adaptive Note-Enhanced
Retrieval-Augmented Generation. arXiv preprint arXiv:2410.08821
(2024).
[85] Siqi Wang, Chao Liang, Yunfan Gao, Yang Liu, Jing Li, and Haofen
Wang. 2024. Decoding Urban Industrial Complexity: Enhancing
Knowledge-Driven Insights via IndustryScopeGPT. In Proceedings of
the 32nd ACM International Conference on Multimedia . 4757–4765.
[86] Shuting Wang, Jiongnan Liu, Shiren Song, Jiehan Cheng, Yuqi Fu,
Peidong Guo, Kun Fang, Yutao Zhu, and Zhicheng Dou. 2024. Domain-
rag: A chinese benchmark for evaluating domain-specific retrieval-
augmented generation. arXiv preprint arXiv:2406.05654 (2024).
[87] Xidong Wang, Guiming Hardy Chen, Dingjie Song, Zhiyi Zhang,
Zhihong Chen, Qingying Xiao, Feng Jiang, Jianquan Li, Xiang Wan,
Benyou Wang, et al .2023. Cmb: A comprehensive medical benchmark
in chinese. arXiv preprint arXiv:2308.08833 (2023).
[88] Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran Zhang, Yixin
Wu, Zhibo Xu, Tianyuan Shi, Zhengyuan Wang, Shizheng Li, Qi
Qian, et al .2024. Searching for best practices in retrieval-augmented
generation. arXiv preprint arXiv:2407.01219 (2024).
[89] Zheng Wang, Zhongyang Li, Zeren Jiang, Dandan Tu, and Wei
Shi. 2024. Crafting Personalized Agents through Retrieval-
Augmented Generation on Editable Memory Graphs. arXiv preprint
arXiv:2409.19401 (2024).
[90] Zhengren Wang, Jiayang Yu, Dongsheng Ma, Zhe Chen, Yu Wang,
Zhiyu Li, Feiyu Xiong, Yanfeng Wang, Linpeng Tang, Wentao Zhang,
et al.2025. RARE: Retrieval-Augmented Reasoning Modeling. arXiv
preprint arXiv:2503.23513 (2025).
[91] Yixuan Weng, Minjun Zhu, Guangsheng Bao, Hongbo Zhang, Jin-
dong Wang, Yue Zhang, and Linyi Yang. 2024. Cycleresearcher:
Improving automated research via automated review. arXiv preprint
arXiv:2411.00816 (2024).
[92] Junde Wu, Jiayuan Zhu, and Yuyuan Liu. 2025. Agentic Reasoning:
Reasoning LLMs with Tools for the Deep Research. arXiv preprint
arXiv:2502.04644 (2025).
[93] Wenjie Wu, Yongcheng Jing, Yingjie Wang, Wenbin Hu, and Dacheng
Tao. 2025. Graph-augmented reasoning: Evolving step-by-step knowl-
edge graph retrieval for llm reasoning. arXiv preprint arXiv:2503.01642
(2025).
[94] Zekun Xi, Wenbiao Yin, Jizhan Fang, Jialong Wu, Runnan Fang,
Ningyu Zhang, Jiang Yong, Pengjun Xie, Fei Huang, and Huajun
Chen. 2025. OmniThink: Expanding Knowledge Boundaries in Ma-
chine Writing through Thinking. arXiv preprint arXiv:2501.09751
(2025).[95] Liang Xiao, Wen Dai, Shuai Chen, Bin Qin, Chongyang Shi, Haopeng
Jing, and Tianyu Guo. 2025. Retrieval-Augmented Generation by
Evidence Retroactivity in LLMs. arXiv preprint arXiv:2501.05475
(2025).
[96] Guangzhi Xiong, Qiao Jin, Xiao Wang, Yin Fang, Haolin Liu, Yifan
Yang, Fangyuan Chen, Zhixing Song, Dengyu Wang, Minjia Zhang,
et al.2025. Rag-gym: Optimizing reasoning and search agents with
process supervision. arXiv preprint arXiv:2502.13957 (2025).
[97] Guanming Xiong, Haochen Li, and Wen Zhao. 2025. MCTS-KBQA:
Monte Carlo Tree Search for Knowledge Base Question Answering.
arXiv preprint arXiv:2502.13428 (2025).
[98] Ruibin Xiong, Yimeng Chen, Dmitrii Khizbullin, and Jürgen Schmid-
huber. 2025. Beyond Outlining: Heterogeneous Recursive Planning
for Adaptive Long-form Writing with Language Models. arXiv
preprint arXiv:2503.08275 (2025).
[99] Fengli Xu, Qianyue Hao, Zefang Zong, Jingwei Wang, Yunke Zhang,
Jingyi Wang, Xiaochong Lan, Jiahui Gong, Tianjian Ouyang, Fanjin
Meng, et al .2025. Towards Large Reasoning Models: A Survey of
Reinforced Reasoning with Large Language Models. arXiv preprint
arXiv:2501.09686 (2025).
[100] Zhipeng Xu, Zhenghao Liu, Yukun Yan, Shuo Wang, Shi Yu, Zheni
Zeng, Chaojun Xiao, Zhiyuan Liu, Ge Yu, and Chenyan Xiong. 2024.
ActiveRAG: Autonomously Knowledge Assimilation and Accom-
modation through Retrieval-Augmented Agents. arXiv preprint
arXiv:2402.13547 (2024).
[101] Ruiran Yan, Zheng Liu, and Defu Lian. 2025. O1 embedder: Let
retrievers think before action. arXiv preprint arXiv:2502.07555 (2025).
[102] Xiaoming Zhang, Ming Wang, Xiaocui Yang, Daling Wang, Shi Feng,
and Yifei Zhang. 2024. Hierarchical Retrieval-Augmented Genera-
tion Model with Rethink for Multi-hop Question Answering. arXiv
preprint arXiv:2408.11875 (2024).
[103] Zhuocheng Zhang, Yang Feng, and Min Zhang. 2025. LevelRAG:
Enhancing Retrieval-Augmented Generation with Multi-hop Logic
Planning over Rewriting Augmented Searchers. arXiv preprint
arXiv:2502.18139 (2025).
[104] Bowen Zhao, Zander Brumbaugh, Yizhong Wang, Hannaneh Ha-
jishirzi, and Noah A Smith. 2024. Set the clock: Temporal alignment of
pretrained language models. arXiv preprint arXiv:2402.16797 (2024).
[105] Xuejiao Zhao, Siyan Liu, Su-Yin Yang, and Chunyan Miao. 2025.
MedRAG: Enhancing Retrieval-augmented Generation with Knowl-
edge Graph-Elicited Reasoning for Healthcare Copilot. arXiv preprint
arXiv:2502.04413 (2025).
[106] Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan
Ye, Pengrui Lu, and Pengfei Liu. 2025. DeepResearcher: Scaling Deep
Research via Reinforcement Learning in Real-world Environments.
arXiv preprint arXiv:2504.03160 (2025).
[107] Yijie Zhong, Feifan Wu, Mengying Guo, Xiaolian Zhang, Meng
Wang, and Haofen Wang. 2025. Meta-PKE: Memory-Enhanced Task-
Adaptive Personal Knowledge Extraction in Daily Life. Information
Processing & Management 62, 4 (2025), 104097.
[108] Yujia Zhou, Zheng Liu, Jiajie Jin, Jian-Yun Nie, and Zhicheng Dou.
2024. Metacognitive retrieval-augmented large language models. In
Proceedings of the ACM Web Conference 2024 . 1453–1463.
[109] Jiachen Zhu, Congmin Zheng, Jianghao Lin, Kounianhua Du, Ying
Wen, Yong Yu, Jun Wang, and Weinan Zhang. 2025. Retrieval-
Augmented Process Reward Model for Generalizable Mathematical
Reasoning. arXiv preprint arXiv:2502.14361 (2025).
[110] Rongzhi Zhu, Xiangyu Liu, Zequn Sun, Yiwei Wang, and Wei Hu.
2025. Mitigating Lost-in-Retrieval Problems in Retrieval Augmented
Multi-Hop Question Answering. arXiv preprint arXiv:2502.14245
(2025).

Synergizing RAG and Reasoning: A Systematic Review Conference’17, July 2017, Washington, DC, USA
Appendix
Agentic RAG Symbol Reference System
The following table presents a complete symbol reference
system with formally defined mathematical notations for all
core concepts.
Symbol Design Hierarchy
•Base states/actions : Standard font ( 𝑆𝑡,𝑎𝑡)
•Sets/spaces : Calligraphic font ( A,K𝑡)
•Core mechanism functions : Uppercase Greek ( Ψ,Γ)
•Operational functions : Calligraphic font ( R,T𝑎)•Auxiliary functions : Lowercase Greek ( 𝛿,𝜙) or black-
board bold ( I)
Annotation Guidelines
•Symbol disambiguation :
–Rstrictly denotes retrieval function (vs. reward 𝑅)
–𝛿exclusively represents state transitions (vs. branch
selector𝜓)
•Dynamic extensions :
–Action spaceAand knowledge base K𝑡support
incremental updates: K𝑡+1=K𝑡⊕Retrieve(𝑞𝑡)

Conference’17, July 2017, Washington, DC, USA Gao et al.
Table 3. Basic states and system components
Symbol Type Definition & Description
𝑆𝑡=(𝐻𝑡,𝐶𝑡)Composite state Complete system state at timestep 𝑡, containing historical information and context vectors
𝐻𝑡 Vector/Set Historical information aggregation
𝐶𝑡 Vector Contextual embedding vectors
𝑞𝑡 Vector Vector representation of current query at step 𝑡
K𝑡 Set Dynamic knowledge base (Initialized as K0=∅)
Table 4. Action space and policy definitions
Symbol Type Definition & Description
A Set Action space, e.g., A={Retrieve,Generate,Verify,Terminate}
𝑎𝑡 Scalar Selected action at timestep 𝑡(𝑎𝑡∈A)
𝜋(𝑆𝑡;Θ)Function Policy function with parameters Θ, mapping states to action probability distributions ( 𝜋:S→Δ(A))
Table 5. State transition mechanisms
Symbol Type Definition & Description
𝛿 Function State transition function , update rule 𝑆𝑡+1=𝛿(𝑆𝑡,·)
T𝑎 Function Low-level state transition operation for action 𝑎(e.g.,TRetrieve denotes retrieval)
R Function Retrieval function ,R(𝑆𝑡)returns retrieval results
◦ Operator Function composition operator (e.g., 𝑓◦𝑔(𝑥)=𝑓(𝑔(𝑥)))
Table 6. Feedback and optimization components
Symbol Type Definition & Description
𝑅(𝑆𝑡,𝑎𝑡,𝑆𝑡+1)Function Reward function , outputs reward value 𝑟𝑡
I(·) Function Indicator function (returns 1 if condition holds, else 0)
∇𝜃𝐽(𝜃) Operator Policy gradient for optimizing policy parameters Θ
𝛾 Scalar Discount factor for cumulative reward calculation
Table 7. Submodule-specific symbols
Symbol Type Definition & Description
Ψ Function Reasoning function , generates intermediate reasoning results
Γ Function Decision function , produces final outputs (e.g., answers)
𝜓(·) Function Branch selector for reflective reasoning path selection
𝜙(·) Function Confidence mapping function (evaluations to scalar confidence)
𝜏 Scalar Decision threshold for triggering specific operations (e.g., verification/termination)
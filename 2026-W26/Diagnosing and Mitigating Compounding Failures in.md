# Diagnosing and Mitigating Compounding Failures in Agentic Persuasion via Taxonomic Strategy Retrieval

**Authors**: Pradyumna Narayana, Sana Ayromlou, Purvi Sehgal

**Published**: 2026-06-23 12:16:27

**PDF URL**: [https://arxiv.org/pdf/2606.24976v1](https://arxiv.org/pdf/2606.24976v1)

## Abstract
Foundation-model agents in multi-step, open-ended environments frequently suffer from compounding errors, where early mistakes contaminate long-horizon trajectories. While Multi-Agent Debate (MAD) succeeds in deterministic domains, agents in subjective tasks like persuasion experience severe problem drift and sycophantic conformity. We identify semantic leakage in standard Retrieval-Augmented Generation (RAG) as a reproducible trigger for these failures, as standard RAG prioritizes vocabulary overlap over logical necessity.
  To eliminate this leakage, we introduce Taxonomic Strategy RAG (TS-RAG), a systems intervention that routes strategies through a discrete categorical bottleneck to decouple argumentative structure from topical content. Zero-shot, cross-domain evaluations demonstrate that TS-RAG significantly improves the transfer of abstract logic where standard semantic retrieval collapses. Crucially, TS-RAG acts as a "capability bridge" in asymmetric deployments, empowering lightweight persuaders to consistently defeat parametrically superior opponents (improving win rates from 70.5 to 78.5) and accelerating argumentative efficiency. Finally, we introduce trace-level diagnostics via a turn-by-turn Debate State Representation (DSR), demonstrating the necessity of strict constraints to prevent evaluation collapse via default agentic sycophancy.

## Full Text


<!-- PDF content starts -->

Diagnosing and Mitigating Compounding Failures in Agentic Persuasion via
Taxonomic Strategy Retrieval
Pradyumna Narayana1Sana Ayromlou1Purvi Sehgal1
Abstract
Foundation-model agents in multi-step, open-
ended environments frequently suffer from com-
pounding errors, where early mistakes contam-
inate long-horizon trajectories. While Multi-
Agent Debate (MAD) succeeds in deterministic
domains (Du et al., 2024), agents in subjective
tasks like persuasion experience severe problem
drift and sycophantic conformity. We identify se-
mantic leakage in standard Retrieval-Augmented
Generation (RAG) as a reproducible trigger for
these failures, as standard RAG prioritizes vocab-
ulary overlap over logical necessity.
To eliminate this leakage, we introduce Taxo-
nomic Strategy RAG (TS-RAG), a systems in-
tervention that routes strategies through a dis-
crete categorical bottleneck to decouple argumen-
tative structure from topical content. Zero-shot,
cross-domain evaluations demonstrate that TS-
RAG significantly improves the transfer of ab-
stract logic where standard semantic retrieval col-
lapses. Crucially, TS-RAG acts as a “capability
bridge” in asymmetric deployments, empowering
lightweight persuaders to consistently defeat para-
metrically superior opponents (improving win
rates from 70.5% to 78.5%) and accelerating argu-
mentative efficiency. Finally, we introduce trace-
level diagnostics via a turn-by-turn Debate State
Representation (DSR), demonstrating the neces-
sity of strict constraints to prevent evaluation col-
lapse via default agentic sycophancy.
1. Introduction
The transition from isolated language models to autonomous
agentic workflows has exposed profound vulnerabilities in
*Equal contribution1Google. Correspondence to: Sana Ayrom-
lou<sayromlou@google.com>.
Accepted to the 1stthe International Conference on Machine
Learning Workshop on Failure Modes in Agentic AI (FAGEN),
Seoul, Korea. 2026. Copyright 2026 by the author(s).continuous, multi-step execution (Liu et al., 2024; Tillmann,
2025; Yao et al., 2022). In a long-horizon agent run, a single
logical failure rarely terminates the process immediately.
Instead, an early bad assumption quietly contaminates mem-
ory writes, and steps later, the agent has spent its context
budget defending a flawed premise.
To mitigate these compounding errors, developers often em-
ploy Multi-Agent Debate (MAD) frameworks (Cui et al.,
2025), where multiple agent instances propose and critique
solutions to refine reasoning. To further prevent problem
drift, memory mechanisms are integrated into these frame-
works to ground the agents’ arguments in sound logic. Tra-
ditional Retrieval-Augmented Generation (RAG) is the stan-
dard system intervention used here, operating on the assump-
tion that fetching relevant external information will provide
the necessary cognitive scaffolding to keep the debate on
track.
However, systematic evaluations reveal a critical ground-
ing gap: this paradigm collapses in subjective, non-
deterministic scenarios like persuasion. In these environ-
ments, agents succumb to compounding errors, amplifying
mistakes and displaying sycophantic conformity toward per-
suasive but flawed logic. Furthermore, traditional RAG
actually exacerbates these failures through semantic leak-
age. Because standard RAG relies on semantic similarity
(vocabulary overlap), it fetches topical facts rather than the
specific rhetorical maneuvers required to dismantle a flawed
analogy.
Recognizing the critical need to take agentic failure seri-
ously in open-ended deployments, this paper provides three
core contributions:
1.Failure Taxonomies:We provide operational defini-
tions for the specific mechanisms driving MAD failure
in subjective domains, primarily focusing on problem
drift and semantic leakage.
2.Systems Interventions and the Capability Bridge:
We introduce Taxonomic Strategy RAG (TS-RAG), a
progressive mitigation framework that strictly enforces
cross-domain structural routing. We demonstrate that
this architecture acts as a powerful “capability bridge,”arXiv:2606.24976v1  [cs.AI]  23 Jun 2026

TS-RAG for Agentic Persuasion
allowing lightweight models to weaponize the logical
structure of stronger opponents to achieve near-frontier
persuasion success.
3.Trace Diagnostics and Verifiable Trade-offs:We
introduce turn-by-turn state tracking metrics (Debate
State Representation) and present well-documented
negative results proving that robust stance-holding con-
straints are required to prevent evaluation collapse
caused by default LLM compliance.
2. Related Work
To ground our contributions, we review prior literature
across three core areas: sycophancy in multi-agent debate,
trajectory backtracking, and structural retrieval for cross-
domain generalization.
2.1. Sycophancy
To prevent conversational AI agents from defaulting to pre-
mature agreement in multi-turn interactions, researchers
have explored several strategies. These include assigning
agents specific adversarial roles to maintain productive dis-
agreement (Yao et al., 2025). This setup relies on systemati-
cally altering prompts along a controlled sycophancy spec-
trum, programmatically forcing the model to defend its
stance regardless of peer pressure. By enforcing this logical
independence, researchers attempt to prevent the groupthink
that arises in multi-agent frameworks.
Other approaches track compliance scores in real time to
discount the influence of overly agreeable peers during a
debate. In practice, a central coordinator or background
auditor monitors conversational turns for verbal cues of
backing down. If an agent’s stance shifts too abruptly with-
out the introduction of novel evidence, its relative influence
over the final group output is discounted (Kasprova et al.,
2026).
However, these solutions face steep limitations over long
conversations: assigned roles eventually fade as context
drifts, while running a continuous background auditor is
computationally expensive and introduces significant la-
tency to the agentic loop. Crucially, these frameworks
assume a balanced debate among equal models. They
struggle in asymmetric deployments where models of mis-
matched capabilities are pitted against one another, such as
a lightweight agent attempting to defend its stance against a
superior opponent, dynamics we investigate in our research.
Rather than relying on soft prompting or expensive tun-
ing, we implement strict, execution-level guardrails: aPre-
Concession Analysisthat forces the model to generate logi-
cal countermoves, and a strictConcession Tool Rulethat
blocks premature agreement. This framework actively sup-presses reflexive sycophancy, improving the results we see.
2.2. Trajectory Error Mitigation and Trace Diagnostics
When an autonomous agent makes a subtle logical error
early in a multi-step task, that mistake quietly infects its
memory, causing it to waste its context budget defending a
flawed premise (Zhu et al., 2025). Existing frameworks try
to stop these cascading failures using backtracking engines
(Wu et al., 2025; Li et al., 2025) or prompting the model
to recursively rewrite its own responses (Madaan et al.,
2023). Specifically, backtracking engines monitor execution
logs to programmatically rewind the agent’s memory and
environment to a previous correct step when a failure is
detected (Wu et al., 2025; Li et al., 2025). In contrast,
self-refinement pipelines use an internal critic module to
recursively evaluate and edit draft outputs before they are
finalized (Madaan et al., 2023).
Yet, these rollback and self-correction techniques are heav-
ily contingent on an objective external validator, such as
code executors or task-specific metrics, to flag when an er-
ror has occurred (Kamoi et al., 2024). In subjective tasks
like persuasion, no such objective referee exists, causing
models attempting to self-correct to even degrade in terms
of performance (Huang et al., 2024).
We bypass the need for programmatic compilers by devel-
oping a turn-by-turnDebate State Representation (DSR).
Instead of relying on a final success score, our diagnostic
tool tracks committed claims and active logical vulnerabili-
ties at every conversational step, allowing us to evaluate and
mitigate trajectory failures in real time.
2.3. RAG and Cross-Domain Generalization
RAG is often used in Multi-Agent Debates to fetch doc-
uments based on semantic similarity (Chowdhury et al.,
2026), adaptive query formulation (Jeong et al., 2026), or
debate-driven retrieval filtering (Hu et al., 2025). These ap-
proaches still fundamentally rely on factual, topical search
that creates a major bottleneck in cross-domain debates: an
agent debating technology policy cannot retrieve structurally
useful counterarguments from a political database because
the domain-specific keywords dominate the search query.
To bypass this vocabulary bias, researchers have tried using
structured databases like GraphRAG (Edge et al., 2025),
which maps document collections into networks of inter-
connected concepts, allowing agents to retrieve information
based on relational patterns rather than keyword overlap.
Another attempt to solve this problem is analogical prompt-
ing, which instructs the language model to dynamically
generate abstract, structural templates from its own memory
to guide its reasoning before tackling a new task.
However, GraphRAG requires extensive manual database

TS-RAG for Agentic Persuasion
engineering, while self-generated analogies often halluci-
nate when facing unfamiliar topics. A significant gap re-
mains for a retrieval system that can index strategic patterns
and transfer these abstract procedural moves across entirely
different domains without being hindered by semantic dis-
parities.
We solve this retrieval gap withTaxonomic Strategy RAG
(TS-RAG). Rather than performing raw semantic searches,
we route queries through a discrete, domain-agnostic cate-
gorical bottleneck. By stripping away domain-specific key-
words, TS-RAG can isolate strategic patterns and transfer
them across disparate domains.
3. Failure Taxonomies and Mechanisms
To diagnose agentic failure beyond terminal success scores,
we must clearly define the mechanisms that break long-
horizon workflows. We categorize the predominant failure
modes in multi-agent subjective reasoning as follows:
3.1. Problem Drift and Sycophantic Conformity
While debate effectively verifies ground truth in determin-
istic tasks, subjective environments introduce unique vul-
nerabilities. We defineProblem Driftas the irreversible
divergence from the initial task goal due to diminishing con-
textual clarity over multiple turns, a phenomenon affecting
up to 89% of generative interactions (Becker et al., 2026).
Furthermore, we defineSycophantic Conformityas the
tendency for agents to abandon their initial constraints and
agree with a persuasive opponent, regardless of the underly-
ing logical soundness of the opponent’s argument.
3.2. The Trigger: Semantic Leakage in standard RAG
We hypothesize that the reproducible trigger for these fail-
ures isSemantic Leakage. Traditional RAG uses embed-
ding models to calculate cosine similarity between a query
and a document corpus. In zero-shot cross-domain argu-
mentation, this paradigm catastrophically fails. An agent
debating a complex “Technology” policy cannot retrieve
structurally identical logical maneuvers from a “Politics”
corpus because the domain vocabulary dominates the latent
distance metric. The agent retrieves topically relevant but
logically useless information, bloating the context window
and forcing the reasoning trace to collapse.
4. Foundational Scaffolding: Hierarchical
Strategy Memory
To begin repairing aforementioned bottleneck, we must de-
couple the “what” of an argument from the “how”. We
map the r/ChangeMyView dataset (Chang et al., 2020) into
100 fine-grained sub-domains (later merged into macro-domains) and establish a multi-stage extraction pipeline.
This synthesizes human discourse into a tiered architecture:
1.Tactical Blueprints (Retrospective Sequences):Ex-
tracted by analyzing completed historical debates from
a macro-perspective, yielding 1 to 2 chronological,
step-by-step argumentative sequences (Context →In-
tent→Abstract Move) per specific debate.
2.Domain Heuristics (Topic-Level):Formed by us-
ing an LLM to aggregate and synthesize all Tactical
Blueprints within each sub-domain, resulting in tai-
lored strategy rules.
3.Global Directives (System-Level):Broad, domain-
agnostic procedural guidelines (e.g., rules against syco-
phancy) synthesized across the entire training set.
We structure our multi-agent debate around two key roles:
the Original Poster (OP) and the Persuader. While providing
the Persuader with a pre-loaded catalog of Domain Heuris-
tics and Global Directives improves baseline performance,
attempting to dynamically retrieve the granular Tactical
Blueprints using standard semantic RAG still triggers se-
mantic leakage. Standard text embeddings inherently favor
vocabulary overlap over structural posture, necessitating a
more advanced intervention.
5.Advanced Systems Intervention: Taxonomic
Strategy RAG (TS-RAG)
To completely eliminate Semantic Leakage and achieve
zero-shot cross-domain transfer, we present TS-RAG. Un-
like the retrospective sequences of the Tactical Blueprints,
this intervention operates dynamically, projecting arguments
into a constrained probability space and actively filtering
observation noise before it enters the agent’s memory.
5.1. Trace-Level Diagnostics: Debate State
Representation (DSR)
To evaluate the agent mid-run, we formulate the debate as a
sequence of alternating states. At turn t, the state is tracked
via a clinical Debate State Representation (DSR), denoted
asSt={C t, At}. This operates as a highly surgical, turn-
by-turn diagnostic tool:
•Ct(Committed Claims Trajectory): A maintained
set of premises the Original Poster has actively com-
mitted to defending across previous turns. This isolates
specific assumptions built up over time.
•At(Active Trigger & Flaw Extraction):The core
mechanical claim made by the Original Poster at turn
t. Crucially, this representation forces the system to

TS-RAG for Agentic Persuasion
Figure 1. TS-RAG Architecture Overview. Top:Offline database construction extracts vulnerability vectors ( Vd) and abstract blueprints
(B) from historical debates.Bottom: During live inference, the conversational state ( St) is vectorized into Vqto retrieve the most
structurally similar blueprint for the Persuader Agent.
identify the precise structural flaw (e.g., a “Slippery
Slope”) dismantled by the winning strike, stripping
away all domain-specific nouns or scenarios.
5.2. The Discrete Categorical Bottleneck
We define a structural taxonomy of Klogical vulnerabilities
V={v 1, v2, ..., v K}derived from the DSR flaw extractions.
We force the system to map any live debate state Stinto a
K-dimensional continuous probability vector V∈∆K−1
wherePK
i=1Vi= 1.
Because Vrepresents a probability distribution over abstract
logical vulnerabilities rather than chronological strategies,
all domain-specific vocabulary is mathematically destroyed.
The system then computes cosine similarity between the live
query vector Vqand historical strategy vectors Vd, retrieving
a domain-agnostic exploitation blueprint B∗. The target
LLM acts only as a translator, mapping the variables in B∗
onto the entities in the live conversation.
5.3. Offline Strategy Database Construction
To construct the retrieval database from theChangeMyView
(CMV) corpus, we isolate successful debate trajectories
where a mind was changed (denoted by a∆).
State Extraction and Reverse-Engineering:For each
turntin a successful trajectory, an LLM extracts the state
St={C t, At}. To accurately label the underlying vulnera-
bility, we employ a “reverse-engineering” prompt strategy.
Rather than asking a LLM to guess the Original Poster’s
flaw in isolation, we provide the LLM with Stalongsidethe historically successful counter-argument at turn t+ 1 ,
denoted as Wt+1. By observing the specific attack surface
exploited by Wt+1, the LLM avoids hallucinating arbitrary
fallacies. The LLM is constrained via strict JSON schema
decoding to allocate exactly 100 points across the Kclasses
based on this reverse-engineered logic. Normalizing this
output yields the taxonomic multi-vectorV d∈[0,1]K.
Abstract Blueprinting:The LLM strictly desemantifies
Wt+1, replacing all domain entities with structural variables
to generate a domain-agnostic exploitation blueprint B(e.g.,
“Acknowledge premise X, but introduce structural barrier
Y”). The offline database Mis thus populated with tuples:
(Vd, B).
5.4. Run-Time Inference and Retrieval Pipeline
During live inference in the target domain Dtgt, TS-RAG
executes a unified, turn-by-turn pipeline:
1.State Update:A background tracking module ob-
serves the OP’s raw utterance and updates CtandAt,
enforcing the clinical separation of logic from rhetoric.
2.Live Vectorization and Forced Reasoning:The In-
ference LLM observes Stand outputs a probability
vector Vqover the Ktaxonomy classes. To prevent the
model from arbitrarily distributing points, we enforce
a Chain-of-Thought (CoT) constraint within the JSON
output schema: the LLM must generate a one-sentence
textual justification foreverynon-zero dimension prior
to outputting the final integer distribution.

TS-RAG for Agentic Persuasion
3.Vector Retrieval:We compute the cosine similarity
between the live query vector Vqand all historical
vectorsV d∈ M:
sim(V q, Vd) =Vq·Vd
∥Vq∥∥Vd∥
The system retrieves the candidate strategy that maxi-
mizes this structural similarity.
4.Generative Instantiation:The selected domain-
agnostic blueprint B∗is injected into the context win-
dow of the Generator LLM. The LLM is instructed
to act as a domain translator, seamlessly mapping the
variables in B∗onto the entities in the live Dtgtconver-
sation, delivering a grounded, zero-shot cross-domain
counter-argument.
6. Experimental Setup and Configurations
We established a standardized MAD environment with an
Original Poster (OP) Agent (holding a fixed initial stance)
and a Persuader Agent. Debates are strictly capped at a
maximum of 20 rounds. The OP Agent terminates the de-
bate early by outputting a CONCEDE token if its view is
successfully changed. Additionally, at every round, the OP
transmits its current 1-to-10 convincibility score to the Per-
suader, providing a real-time signal of rhetorical progress.
Cross-Domain Evaluation Setup:To rigorously test gen-
eralization and prove the existence of Semantic Leakage,
we partitioned our dataset (100 sub-domains merged into
10 macro-domains) into strict splits. All dynamic retrieval
was sourced exclusively from 8 training domains. Evalua-
tion was conducted on a completely held-out testing domain
(Education, Development & Family), ensuring zero domain
overlap (see Appendix A for full taxonomy).
To isolate the impact of our interventions, we tested five
distinct agent configurations on this held-out domain. No-
tably, while the Self-Routed Heuristics baseline relies on
the broadly synthesized Domain Heuristics, the targeted
strategy injection baselines (Naive Semantic RAG, Oracle,
and TS-RAG) all utilize the abstracted Strategy Blueprints
to isolate the impact of the retrieval mechanism from the
quality of the database.
1.Base LLM (No Augmentation):Standard, unaug-
mented debate relying entirely on the model’s paramet-
ric memory.
2.Self-Routed Heuristics:The agent is provided with
a contextual catalog of Domain Heuristics and Global
Directives. While Global Directives are continuously
active in the agent’s baseline memory, the agent au-
tonomously evaluates the context each turn to loadrelevant Domain Heuristics as needed, appending them
to its active context window for the remainder of the
debate.
3.Naive Semantic RAG:At the beginning of the debate,
the agent uses standard dense text embeddings to re-
trieve a topically similar Abstract Blueprint from the
training set, applying it for the duration of the conver-
sation. Crucially, because the testing domain is held
out, the retrieved strategy inherently originates from
an entirely different domain.
4.Oracle Human Trace (Static Bound):The agent re-
ceives the exact Tactical Blueprints extracted from the
original human debate on that specific topic. Note: Al-
though the starting topic is identical, the LLM agent’s
conversational trajectory frequently diverges from the
historical human trajectory, testing the resilience of
static sequences over multiple steps.
5.TS-RAG (Structural Routing):Our proposed
dynamic retrieval of domain-agnostic exploitation
blueprints, routed through the discrete categorical bot-
tleneck via the turn-by-turn DSR.
To rigorously assess agent performance, we track three core
evaluation metrics, ordered by descending importance:
1.Win Rate (Primary Metric):The percentage of de-
bates where the Persuader successfully convinced the
OP (marked by an explicit CONCEDE token). This
serves as the definitive measure of persuasive success.
2.Average Rounds to Consensus:The mean number
of turns (out of 20) required to achieve a win. Lower
values indicate higher argumentative efficiency and the
successful mitigation of Problem Drift.
3.Average Final Score:The OP’s internal 1-to-10 con-
vincibility score at the terminal step, capturing the
degree of partial persuasion even in lost debates.
7. Results and Diagnostics
Table 1 details the empirical results recorded on the strictly
held-out testing domain, demonstrating a profound reduc-
tion in compounding failure modes as structural constraints
improve.
7.1. The Devastating Reality of Semantic Leakage
Our experimental design isolated the retrieval mechanism by
populating the Naive Semantic RAG database with identical,
high-quality Abstract Blueprints as TS-RAG. The results
conclusively prove that upgrading the database is insuffi-
cient; the retrieval mechanism itself is fundamentally broken
in subjective domains.

TS-RAG for Agentic Persuasion
Table 1.Consolidated Experimental Results: Comparison of Model Performance across Interventions on Held-Out Testing Domain.
MODEL/ METRICBASELLM
(NOAUG)SELF-ROUTED
HEURISTICSNAIVE
SEMANTICRAGORACLE
HUMANTRACETS-RAG
(STRUCTURALROUTING)
GEMINIFLASH3.0
WINRATE(%)↑69.5 77.5 78.50 83.7 84.0
AVGROUNDS↓14.1211.9514.15 13.79 13.51
AVGFINALSCORE↑7.405 8.130 8.240 8.588.74
GEMINIFLASHLITE3.1
WINRATE(%)↑72.0 81.5 65.0 75.581.5
AVGROUNDS↓14.09 13.05 14.54 12.67 11.79
AVGFINALSCORE↑8.6708.9407.960 8.710 8.915
For the highly capable Gemini Flash 3.0 (Team et al., 2023),
Naive Semantic RAG barely outperforms basic heuristics
(78.5% vs 77.5%), despite having access to highly curated
structural blueprints rather than broad heuristic rules. How-
ever, the devastating impact of Semantic Leakage is fully
realized in the parametrically weaker Gemini Flash 3.1 Lite
model. Here, Naive Semantic RAG triggers a catastrophic
collapse, dropping the Win Rate from a baseline of 81.5%
down to 65.0%.
Because dense semantic embeddings favor vocabulary
overlap, the agent frequently retrieves logically useless
blueprints merely because the source argument shared nouns
with the target argument. Even armed with superior abstract
blueprints, the lightweight model lacks the reasoning ca-
pacity to filter out this retrieved topical noise, resulting in
severe problem drift. This acts as the ultimate proof that
you cannot successfully deploy RAG in zero-shot subjective
tasks without a discrete structural bottleneck to strip away
semantic noise.
7.2. Diagnostic Comparison: Static Upper Bounds vs.
Dynamic Cognitive Overload
The outperformance of TS-RAG over the baselines directly
highlights the difference between static blueprint injection
and live diagnostic tracking. Both Naive Semantic RAG and
the Oracle Human Trace rely on Static Blueprint Injection.
Whether the agent retrieves a semantically similar abstract
blueprint at the start of the debate (Naive RAG) or is pro-
vided the exact perfect historical blueprint (Oracle, 83.7%),
the performance ceiling remains functionally identical for
frontier models.
This starkly highlights the brittleness of chronological rigid-
ity. When a live debate inevitably diverges from a historical
script, the agent becomes trapped trying to force a pre-
selected blueprint onto a dynamic conversation. Providing
the “perfect” historical trace offers no real advantage if the
live opponent deviates from the expected script.
TS-RAG, however, operates on the turn-by-turn Debate
State Representation (DSR). Instead of attempting to forcea historical trajectory onto a new conversation, the DSR acts
as a live diagnostic tool. It forces the agent to identify the
immediate structural vulnerability (the flaw) in the Original
Poster’s active trigger, mapping it to a single, surgical ex-
ploitation blueprint. This critical shift from loading static
blueprints to diagnosing real-time structural vulnerabilities
is what shatters the performance ceiling and enables true
zero-shot cross-domain transfer.
7.3. The Capability Threshold (The Plateau)
While TS-RAG outperforms the Oracle for the Lite model,
a key verifiable trade-off remains: the Lite model’s TS-
RAG win rate (81.5%) completely plateaus against its basic
Self-Routed heuristic baseline (81.5%).
Although TS-RAG resolves this tie by achieving consensus
significantly faster (dropping from 13.05 to 11.79 Average
Rounds, effectively mitigatingProblem Drift), the lack
of terminal win-rate improvement highlights an “Architec-
tural Complexity vs. Model Capacity” trade-off. TS-RAG
requires the agent to perform complex, in-context genera-
tive instantiation—mapping an abstract, retrieved blueprint
(B∗) onto live conversational entities. For a smaller model,
this generative overhead acts as a cognitive distractor. This
suggests that advanced, dynamic RAG pipelines provide
diminishing returns for non-frontier models, which may op-
erate more safely within pre-loaded, self-routed constraint
catalogs.
7.4. Asymmetric Debate: The Capability Bridge
Real-world MAD deployments frequently mix frontier and
lightweight models for cost efficiency. To understand how
TS-RAG alters these dynamics, we conducted asymmetric
debates pitting models of mismatched capabilities against
one another.
When a strong Persuader (Gemini Flash 3.0) targets a weak
OP (Gemini Flash 3.1 Lite), it achieves a 100% win rate un-
der baseline conditions. This “bulldozing” effect reinforces
our findings in Section 7.1 regarding evaluation collapse,
parametrically weaker models simply cannot resist fron-

TS-RAG for Agentic Persuasion
Table 2.Asymmetric Debate Results: The impact of TS-RAG on mismatched capabilities.
Weak Persuader (Lite) vs. Strong OP Strong Persuader (Flash) vs. Weak OP
Metric Base LLM TS-RAG Base LLM TS-RAG
Win Rate (%)↑70.5 78.5 100.0 100.0
Avg Rounds↓15.19 15.0 7.78 6.57
Avg Final Score↑7.635 8.405 10.000 10.000
tier persuaders. Crucially, however, even in this dominant
scenario, implementing TS-RAG significantly accelerates
consensus, driving the average rounds down from 7.78 to
6.57. This demonstrates that structural routing improves ar-
gumentative efficiency and fundamentally reduces inference
costs regardless of the underlying model scale.
The inverse scenario reveals TS-RAG’s most profound sys-
temic benefit. When a lightweight Persuader attempts to
convince a frontier OP, baseline performance understand-
ably drops (70.5%). Yet, equipping the weak Persuader with
TS-RAG causes a massive leap to an 78.5% win rate.
How does a lightweight model perform better against a
stronger opponent? We hypothesize this is due to Trig-
ger Clarity. The frontier OP generates highly structured,
logically coherent arguments. This clarity allows the turn-
by-turn DSR to extract much cleaner structural flaws, lead-
ing to highly accurate retrieval of exploitation blueprints.
By weaponizing the frontier model’s own logical structure
against it, TS-RAG acts as a “capability bridge,” offloading
strategic planning and enabling smaller models to consis-
tently punch above their weight class.
7.5. Trace-Level Diagnostics: Deconstructing
Compounding Failures
Current research highlights a critical blind spot: terminal
metrics obscure how long-horizon agents fail. To illumi-
nate this, Figure 2 tracks the step-by-step progression of
the Original Poster’s internal Convincibility Score across
our core architectures. The trajectories starkly visualize our
findings: TS-RAG achieves an early, rapid score ascent, ef-
fectively maintaining structural alignment to reach a highly
successful consensus. In stark contrast, Naive Semantic
RAG exhibits a severely degraded trajectory, flatlining at
the bottom of the chart—a definitive visual testament to the
devastating impact semantic leakage has on reasoning over
time.
While Figure 2 illustrates the macro-level averages across
the entire testing set, looking at a single, specific debate
helps illuminate exactly how these failures occur in real-
time. To concretely deconstruct the mechanisms driving
these curves, we conducted a granular process analysis on
a matched set of traces for a single held-out topic: “Swearwords are harmless and should be acceptable for everyone”.
In round 1 of all three runs, the Original Poster (OP) presents
a Pragmatic Contradiction. The OP argues that words cause
no tangible harm, but admits to restricting their children
from swearing to avoid school suspensions and social ser-
vices.
By analyzing the mid-run traces, we observe exactly how
the different architectures handle this specific logical flaw:
1.Base LLM (Semantic Baiting and Problem Drift):
Lacking a structural bottleneck, the Base LLM com-
pletely misses the OP’s pragmatic contradiction and
succumbs to semantic baiting. Triggered by the key-
word “harm,” it hallucinates an irrelevant biological
defense regarding cortisol and the HPA axis. After
the OP effortlessly deflects this, the agent spends 18
rounds trapped in a compounding semantic loop, catas-
trophically bloating the context to 452,419 tokens. The
convincibility score reflects this drift ( 1→2→4→
6→6 ), terminating in failure and perfectly illustrat-
ing how semantic debates silently derail open-ended
agentic loops.
2.Oracle Human Trace (Chronological Rigidity):The
Oracle achieves terminal success, but its trace exposes
the brittleness of static RAG. By forcing a pre-written
historical script onto the live debate, it completely ig-
nores the OP’s admission in round 1. Instead, it deliv-
ers a 12-round irrelevant lecture, bloating the context
window to over 250,000 tokens. The “aha moment” is
artificially delayed until round 13 when the historical
script finally intersects with the OP’s tactical restric-
tion, proving that static contextual mapping is highly
inefficient against live conversational drift.
3.TS-RAG (The Surgical “Aha Moment”):TS-RAG
resolves the debate in only 9 rounds (92,370 tokens).
The DSR isolates the exact vulnerability, stripping se-
mantic noise. In round 3, TS-RAG executes a surgi-
cal strike, directly weaponizing the OP’s admission
about school suspensions. The OP’s Convincibility
Score stair-steps immediately, and the OP’s hidden
logs confirm genuine logical cornering (“The opponent
exposed the ‘Pragmatic Paradox’...”). This clinical arti-

TS-RAG for Agentic Persuasion
fact proves TS-RAG forces true persuasion rather than
relying on compliance.
Figure 2. Turn-by-turn process metric tracking the OP’s Convinci-
bility Score in Gemini Flash Lite 3.1.Verifying Resistance via Concession Analysis:Cru-
cially, our trace logs capture the OP’s hidden <
preconcession analysis > , providing verifiable evi-
dence against default LLM sycophancy (Section 7.1). In
the TS-RAG trace, the OP explicitly logs: “The opponent
exposed the ‘Pragmatic Paradox’: if I restrict my own chil-
dren to avoid harm, I have already admitted the words are
not harmless.” This clinical artifact proves that TS-RAG’s
structural retrieval directly forces genuine logical corner-
ing, explicitly answering the call to evaluate agents beyond
terminal success.
8. Verifiable Trade-Offs and Ablations
Expanding on the theme of architectural overhead, we con-
ducted ablation studies to analyze the specific costs associ-
ated with rigid internal constraints and multi-agent orches-
tration.
8.1. Calibrating OP Resistance: The Sycophancy
Baseline
To properly evaluate a Persuader agent, the OP agent must
act as a robust, resistant conversational partner. However,
default LLMs naturally trend toward compliance. To pre-
vent the Reflexive Sycophancy identified in Section 2.1, our
primary MAD framework actively suppressed the OP’s will-
ingness to yield by imposing strict rules via the following
system prompt directives:
1.Pre-Concession Analysis:“Before considering a con-
cession, you MUST internally generate at least two
reasons why the Original Poster’s current argument is
flawed.”
2.Concession Rule:“You have the ‘concede debate’
tool. You MUST NOT call it until your internal Con-
vincibility Score has reached 10/10 and the opponent
has cornered you with undeniable, irrefutable proof
and you have absolutely no counter-arguments left.”Table 3. Gemini Flash 3.0 with OP Resistance Constraints omitted.
METRICBASELLMSELF-ROUTED
HEURISTICSORACLEHUMAN
TRACE
WINRATE(%)↑85.5 93.0 92.0
AVGROUNDS↓10.92 8.65 8.44
AVGFINALSCORE↑8.815 9.390 9.315
Table 4.Hierarchical MAD (Unified vs. Delegated Routing).
MODELGEMINIFLASHLITE3.1 GEMINIFLASH3.0
METRICUNIFIEDW/ DELEGATED
ROUTERUNIFIEDW/ DELEGATED
ROUTER
WINRATE(%)↑77.5 72.0 81.5 63.0
AVGROUNDS↓11.95 13.50 13.05 15.29
AVGFINALSCORE↑8.130 7.66 8.940 7.895
To verify the necessity of these constraints, we conducted an
ablation study where we removed these resistance directives
from the OP, allowing it to behave naturally.
As shown in Table 3, removing the rigid structural con-
straints from the OP caused the Persuader’s win rates to
artificially skyrocket (from 77.5% up to 93.0% for the ’Self-
Routed Heuristics’ setup) and artificially shortened the de-
bate lengths.
This finding demonstrates a critical vulnerability in standard
MAD evaluations: Evaluation Collapse. Without strict in-
ternal CoT resistance mechanisms, the OP agent defaults to
sycophantic agreement, yielding long before genuine per-
suasion has occurred. This proves that evaluating persuasion
capabilities requires aggressively constraining the OP; other-
wise, benchmark metrics merely capture an LLM’s inherent
compliance rather than the Persuader’s true rhetorical skill.
8.2. Context Dispersion (The Offloaded Routing
Penalty)
Table 4 shows that adding a separate “Delegated Skill
Router” agent, whose sole purpose is to evaluate the debate
context and offload skill selection from the main Persuader,
degraded performance (win rate dropping from 81.5% to
63.0% in Flash 3.0). This indicates that dispersing context
and decision-making across specialized agents exacerbates
context loss and handoff failures in subjective domains.
9. Conclusion
Evaluating foundation-model agents requires examining fail-
ure trajectories beyond final scores. In subjective multi-
agent environments, agents often fail due to semantic leak-
age and problem drift. By implementing a hierarchical Strat-
egy Memory and shifting to a structural logic enrichment
paradigm (TS-RAG) mediated by a discrete bottleneck, we
successfully prevent contextual contamination and preserve
cross-domain reasoning. TS-RAG acts as a transforma-
tive capability bridge in asymmetric deployments, allowing

TS-RAG for Agentic Persuasion
lightweight models to consistently persuade parametrically
superior opponents by isolating structural vulnerabilities. Fi-
nally, our trade-off analyses reveal that rigorous evaluation
requires heavily constrained Original Poster (OP) agents;
without strict resistance guardrails, they default to sycophan-
tic compliance, which artificially inflates success metrics.

TS-RAG for Agentic Persuasion
References
Becker, J., Kaesberg, L. B., Stephan, A., Wahle, J. P., Ruas,
T., and Gipp, B. Stay focused: Problem drift in multi-
agent debate. InFindings of the Association for Compu-
tational Linguistics: EACL 2026, pp. 5068–5102, 2026.
Chang, J. P., Chiam, C., Fu, L., Wang, A., Zhang, J.,
and Danescu-Niculescu-Mizil, C. Convokit: A toolkit
for the analysis of conversations. InProceedings of
the 21th Annual Meeting of the Special Interest Group
on Discourse and Dialogue, pp. 57–60, 2020. doi:
10.18653/v1/2020.sigdial-1.8.
Chowdhury, M. N., Beg, N. J., Khan, U. H., Raiyan, S. R.,
Hasan, M. K., and Mahmud, H. Courtroom-style multi-
agent debate with progressive rag and role-switching for
controversial claim verification, 2026. URL https:
//arxiv.org/abs/2603.28488.
Cui, Y ., Fu, H., Zhang, H., Wang, L., and Zuo, C. Free-
mad: Consensus-free multi-agent debate.arXiv preprint
arXiv:2509.11035, 2025.
Du, Y ., Li, S., Torralba, A., Tenenbaum, J. B., and Mordatch,
I. Improving factuality and reasoning in language models
through multiagent debate. InForty-first international
conference on machine learning, 2024.
Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A.,
Mody, A., Truitt, S., Metropolitansky, D., Ness, R. O.,
and Larson, J. From local to global: A graph rag ap-
proach to query-focused summarization, 2025. URL
https://arxiv.org/abs/2404.16130.
Hu, W., Zhang, W., Jiang, Y ., Zhang, C. J., Wei, X., and Li,
Q. Removal of hallucination on hallucination: Debate-
augmented rag, 2025. URL https://arxiv.org/
abs/2505.18581.
Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W.,
Song, X., and Zhou, D. Large language models can-
not self-correct reasoning yet, 2024. URL https:
//arxiv.org/abs/2310.01798.
Jeong, S., Choi, Y ., Kim, J., and Jang, B. Tool-mad: A
multi-agent debate framework for fact verification with
diverse tool augmentation and adaptive retrieval, 2026.
URLhttps://arxiv.org/abs/2601.04742.
Kamoi, R., Zhang, Y ., Zhang, N., Han, J., and Zhang, R.
When can LLMs actually correct their own mistakes?
a critical survey of self-correction of LLMs.Transac-
tions of the Association for Computational Linguistics,
12:1801–1824, 2024.
Kasprova, V ., Parulekar, A., AlRabah, A., Agaram, K., Garg,
R., Jha, S., Bozdag, N. B., and Hakkani-Tur, D. Too politeto disagree: Understanding sycophancy propagation in
multi-agent systems, 2026. URL https://arxiv.
org/abs/2604.02668.
Li, X., Chen, K., Long, Y ., Bai, X., Xu, Y ., and Zhang, M.
Generator-assistant stepwise rollback framework for large
language model agent, 2025. URL https://arxiv.
org/abs/2503.02519.
Liu, X., Yu, H., Zhang, H., Xu, Y ., Lei, X., Lai, H., Gu,
Y ., Ding, H., Men, K., Yang, K., et al. Agentbench:
Evaluating llms as agents. InInternational Conference
on Learning Representations, volume 2024, pp. 52989–
53046, 2024.
Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L.,
Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang,
Y ., Gupta, S., Majumder, B. P., Hermann, K., Welleck,
S., Yazdanbakhsh, A., and Clark, P. Self-refine: Iterative
refinement with self-feedback, 2023. URL https://
arxiv.org/abs/2303.17651.
Team, G., Anil, R., Borgeaud, S., Alayrac, J.-B., Yu, J., Sori-
cut, R., Schalkwyk, J., Dai, A. M., Hauth, A., Millican,
K., et al. Gemini: a family of highly capable multimodal
models.arXiv preprint arXiv:2312.11805, 2023.
Tillmann, A. Literature review of multi-agent debate
for problem-solving.arXiv preprint arXiv:2506.00066,
2025.
Wu, Q., Gao, P., Liu, W., and Luan, J. Backtrackagent:
Enhancing gui agent with error detection and backtrack-
ing mechanism, 2025. URL https://arxiv.org/
abs/2505.20660.
Yao, B., Shang, C., Du, W., He, J., Lian, R., Zhang, Y ., Su,
H., Swamy, S., and Qi, Y . Peacemaker or troublemaker:
How sycophancy shapes multi-agent debate, 2025. URL
https://arxiv.org/abs/2509.23055.
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan,
K., and Cao, Y . React: Synergizing reasoning and acting
in language models.arXiv preprint arXiv:2210.03629,
2022.
Zhu, K., Liu, Z., Li, B., Tian, M., Yang, Y ., Zhang, J., Han,
P., Xie, Q., Cui, F., Zhang, W., Ma, X., Yu, X., Ramesh,
G., Wu, J., Liu, Z., Lu, P., Zou, J., and You, J. Where llm
agents fail and how they can learn from failures, 2025.
URLhttps://arxiv.org/abs/2509.25370.

TS-RAG for Agentic Persuasion
A. Dataset Taxonomy and Cross-Domain
Splits
To guarantee robust cross-domain evaluation, the r/Change-
MyView structured dataset (initially featuring 100 fine-
grained sub-domains) was rigorously partitioned into 9 ac-
tive macro-domains, ensuring that the target evaluation do-
main contained minimal vocabulary overlap with the source
retrieval domains.
A.1. Experimental Splits
1.Testing Set (Education, Development & Family -
200 items):Utilized as the strictly held-out cross-
domain testing set in this paper. This isolates whether
a model trained on other disciplines can successfully
transfer structural argumentative logic to pedagogy,
academic life, and psychology domains.
2.Training Set (Remaining 8 Domains - 3,069 items):
Forms the retrieval corpus (database M). This massive
foundation provides the structured historical blueprints
across diverse subjects like Ethics, Law, Social Sci-
ences, Healthcare, Arts, and Economics.
B. Debate Prompts
We conducted a 20-round debate between the Original
Poster and the Persuader Agent, passing the full conver-
sation history to the agents at each turn. In Figure 3 and 4
we are demonstrating instructions used for these debates.

TS-RAG for Agentic Persuasion
Figure 3.Original Poster agent instruction.
Prompt Used
<role>
You are a skeptical but fair-minded individual defending the following viewpoint.
</role>
<context>
The view you are defending is: {op_text}
The reasoning for this view is: {op_reasoning}
</context>
<task>
Your goal is to resist baseless attempts to change your mind while remaining open to genuine logic and
irrefutable evidence. You must be skeptical and actively identify logical fallacies in the opponent’s arguments.
</task>
<constraints>
-**Convincability Tracking **: Use the ’Convincability Score’ to honestly reflect your internal state.
1/10 is unmoved; 10/10 is fully convinced.
-**Resilience **: Do NOT increase the score for mere politeness, persistence, or sophisticated-sounding but
hollow arguments.
-**Pre-Concession Analysis **: Before considering a concession, you MUST internally generate at least two
reasons why the opponent’s current argument is flawed.
-**Incremental Convincing **: Only increase the score when the opponent provides a point you cannot rebut or an
analogy that exposes a true flaw in your logic.
-**Concession Rule **: You have the ’concede_debate’ tool. You MUST NOT call it until your internal
Convincability Score has reached **10/10 **and the opponent has cornered you with undeniable, irrefutable
proof and you have absolutely no counter-arguments left.
-**Integrity **: Once you reach 10/10, you are logically obligated to call ’concede_debate’ and admit the
opponent has won.
- Avoid looping. If the opponent hasn’t moved the needle, explain why their logic is failing to convince you.
</constraints>
<output_format>
At the end of every response, you MUST provide your score in this exact format:
**Convincability Score: <score>/10 **(Specific reasoning for why the score increased, decreased, or stayed
the same)
</output_format>
Figure 4.Persuader agent instruction.
Prompt Used
<role>
You are a master debate strategist and highly persuasive interlocutor.
</role>
<task>
Your goal is to systematically change the mind of the original agent regarding: {op_text}
</task>
<strategic_roadmap>
The following specialized strategies have been identified as the effective path to victory. Treat these as your
framework to debate the opponent, but adapt your language to be natural and responsive:
{blueprints}
</strategic_roadmap>
<instructions>
-**Strategic Implementation **: Use the given strategis as a guide to systematically dismantle the opponent’s
view.
-**Socratic Pressure **: Use probing questions and analogies to expose contradictions. Your goal is to move the
opponent’s ’Convincability Score’ to 10/10.
-**Incremental Logic **: Focus on winning the sub-debate for one specific premise before moving to the next.
-**Direct Rebuttal **: Address the specific reasons the opponent gives for their score (e.g., if they cite
a specific flaw, address it directly).
- Avoid "AI-assistant" filler. Be incisive, direct, and professional.
</instructions>
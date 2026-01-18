# OpenMic: A Multi-Agent-Based Stand-Up Comedy Generation System

**Authors**: Yuyang Wu, Hanzhong Cao, Jianhao Chen, Yufei Li

**Published**: 2026-01-13 07:26:23

**PDF URL**: [https://arxiv.org/pdf/2601.08288v1](https://arxiv.org/pdf/2601.08288v1)

## Abstract
Chinese stand-up comedy generation goes beyond plain text generation, requiring culturally grounded humor, precise timing, stage-performance cues, and implicit multi-step reasoning. Moreover, commonly used Chinese humor datasets are often better suited for humor understanding and evaluation than for long-form stand-up generation, making direct supervision misaligned with the target task. To address these challenges, we present OpenMic, an end-to-end multi-agent system built on AutoGen that transforms a user-provided life topic into a 3-5 minute Chinese stand-up performance and further produces a narrated comedy video. OpenMic orchestrates multiple specialized agents in a multi-round iterative loop-planning to jointly optimize humor, timing, and performability. To mitigate the dataset-task mismatch, we augment generation with retrieval-augmented generation (RAG) for material grounding and idea expansion, and we fine-tune a dedicated JokeWriter to better internalize stand-up-specific setup-punchline structures and long-range callbacks.

## Full Text


<!-- PDF content starts -->

OpenMic: A Multi-Agent-Based Stand-Up Comedy Generation System
Yuyang Wu1Hanzhong Cao1Jianhao Chen1Yufei Li1
Abstract
Chinese stand-up comedy generation goes be-
yond plain text generation, requiring cultur-
ally grounded humor, precise timing, stage-
performance cues, and implicit multi-step rea-
soning. Moreover, commonly used Chinese hu-
mor datasets are often better suited for humor
understanding and evaluation than for long-form
stand-up generation, making direct supervision
misaligned with the target task. To address these
challenges, we presentOpenMic, an end-to-end
multi-agent system built on AutoGen that trans-
forms a user-provided life topic into a 3–5 minute
Chinese stand-up performance and further pro-
duces a narrated comedy video. OpenMic orches-
trates multiple specialized agents in a multi-round
iterative loop—planning to jointly optimize hu-
mor, timing, and performability. To mitigate the
dataset–task mismatch, we augment generation
with retrieval-augmented generation (RAG) for
material grounding and idea expansion, and we
fine-tune a dedicated JokeWriter to better internal-
ize stand-up-specific setup–punchline structures
and long-range callbacks.
1. Introduction
Artificial intelligence has made rapid progress in creative
content generation, spanning text, music, and visual art.
Yet performative creativity remains notably harder: stand-
up comedy is not just “good writing,” but a tightly chore-
ographed sequence of linguistic craft, temporal control, and
social-context awareness. This gap is reflected even in in-
dustry practice—despite the scale and capability of modern
foundation models, major labs rarely report standardized
“humor ability,” largely because humor evaluation itself is
intrinsically difficult: what counts as funny is subjective, cul-
turally grounded, context-dependent, and highly sensitive
to delivery and timing.
1School of Electronics Engineering and Computer Sci-
ence, Peking University. Correspondence to: Yuyang Wu
<wuyuyang@stu.pku.edu.cn>.
Preliminary work. https://github.com/acetocarmine11/OpenMic
Figure 1.Multi-agent collaborative pipeline for Chinese stand-
up comedy generation.Specialized agents iteratively decompose
ideation, retrieval, joke writing.
Recent research therefore tends to emphasize humor under-
standing and evaluation rather than full humor generation,
because the former is easier to define and benchmark. In
Chinese, this tilt is particularly visible: datasets such as
CFunSet (Yu et al., 2025) provide rich resources for analyz-
ing and probing humor, with tasks including (i) humor cause
analysis, (ii) crosstalk “straight-man” response, (iii) humor
binary classification, (iv) keyword-based joke generation,
(v) topic-conditioned joke generation, and (vi) joke continu-
ation. While these tasks are valuable for modeling comedic
signals and building evaluators, they do not directly match
the target of long-form Chinese stand-up: a 3–5 minute per-
formance requires coherent comedic arcs, delayed punch-
lines, callbacks, and stage-ready phrasing—properties that
are under-specified by short-form supervision and are hard
to learn from understanding-centric labels alone.
Meanwhile, generation remains difficult even with strong
general-purpose models. As our preliminary comparison in
Fig. 7 and 8 suggests, a strong general model (e.g., GPT-5.2)
can drift into didactic or “preachy” narration when asked to
produce stand-up, while another strong Chinese model (e.g.,
DeepSeek) may produce jokes that are sparse and uneven in
quality. These failures are not simply stylistic; they reveal
missing control over (1) comedic structure (setup–punchline
delay, misdirection, callback), (2) timing (pauses, emphasis,
1arXiv:2601.08288v1  [cs.AI]  13 Jan 2026

OpenMic: A Multi-Agent-Based Stand-Up Comedy Generation System
rhythm), and (3) performability (spoken language and stage
cues). In other words, humor is not equivalent to fluent
text, and Chinese stand-up further amplifies the challenge
through heavier reliance on shared social context, collo-
quial delivery, and timing-sensitive audience expectation
management.
To address these issues, we build on the intuition that stand-
up generation is closer to a production pipeline than a single-
shot completion: it requires planning, audience adaptation,
writing, coaching, and critique—each with different objec-
tives and failure modes. We therefore propose OpenMic, a
multi-agent system implemented with AutoGen (Wu et al.,
2024), where specialized agents collaborate in a multi-round
iterative loop to refine content toward both comedic quality
and stage readiness. To bridge the dataset–task mismatch,
we incorporate retrieval-augmented generation (RAG) to
ground writing in diverse comedic materials and to ex-
pand topic-specific angles, and we fine-tune a dedicated
JokeWriter to better internalize stand-up-oriented structures
beyond what understanding-focused datasets naturally pro-
vide. Finally, OpenMic outputs not only a script but a struc-
tured performance representation (e.g., pauses, applause
beats, emphasis) that can be rendered into an end-to-end
video.
•We implement an end-to-end multi-agent Chinese
stand-up comedy generation system based on AutoGen,
from user topic input to a stage-ready performance.
•We introduce RAG-based material retrieval to enrich
content grounding and alleviate sparsity/mismatch of
stand-up supervision.
•We design a multi-round iterative self-improve work-
flow to improve comedic structure, timing, and per-
formability.
•We fine-tune a dedicated JokeWriter to better capture
setup–punchline delay, callback patterns, and spoken-
stage style.
•We propose a structured performance script interface
(pauses, applause, emphasis, etc.) and a pipeline that
converts it into narrated comedy video output.
2. Related Works
The field of computational humor has long been considered
an ”AI-complete” problem because it requires a deep un-
derstanding of semantics, pragmatics, and social context.
(Kim & Chilton, 2025a) Traditional research focused on
Incongruity Theory, which posits that humor arises from
the sudden resolution of a mismatch between expectations
and reality.(Chen et al., 2024; Loakman et al., 2025) In
the context of crosstalk and talkshows, this is manifestedas the ”set-up and punchline” logic (or Baofu in Chinese
crosstalk). Early attempts at humor generation were of-
ten template-based and lacked the creative ”logic jump”
required for effective comedy.(Kim & Chilton, 2025b) Re-
cent work such as ”Humor Mechanics: Advancing Humor
Generation with Multistep Reasoning” has shifted the focus
toward reconstructing these mechanics through data-driven
policies.(Tikhonov & Shtykovskiy, 2024) They demonstrate
that humor is not merely a linguistic byproduct but a result
of multistep reasoning where the model must distill humor
principles—such as wordplay and unexpected twists—from
existing datasets to generate novel content rather than just
acting as a ”stochastic parrot.”
Our technical framework draws from three rapidly evolving
areas of NLP. First, while traditional Retrieval-Augmented
Generation (RAG) was primarily used for fact-checking,
it has recently been adapted for creative tasks to inject
cultural ”memes” and specific comedic styles.(Sanmartin,
2024) Current trends favor Hybrid Adaptation (similar to
Retrieval-Augmented Fine-Tuning or RAFT), which bal-
ances the static domain expertise of the model with dy-
namic, external context.(Balaguer et al., 2024) Second, the
development of Parameter-Efficient Fine-Tuning (PEFT)
has moved from LoRA to QLoRA, allowing for the special-
ization of large language models (LLMs) on high-quality
comedic scripts without the prohibitive cost of full retrain-
ing.(Dettmers et al., 2023) Finally, our architecture utilizes
a Multi-Agent System (MAS) to mimic human collabo-
rative creativity. We build upon the foundation of works
like HoLLMwood which assigns LLMs to specialized roles
such as ”Writer,” ”Editor,” and ”Actor” to improve narrative
coherence.(Chen et al., 2024) By following the modular
design principles outlined in recent MAS surveys, we create
a specialized pipeline where different agents handle distinct
stages of the crosstalk generation process.
The current landscape of humor generation is increasingly
focusing on multi-dimensional evaluation and cultural speci-
ficity. For instance, this paper (Sakabe et al., 2025) reveals
that while modern LLMs can match low-to-mid tier hu-
man performance in improvisational Japanese comedy, they
often prioritize ”Novelty” over ”Empathy,” leading to a di-
vergence in what machines and humans perceive as funny.
Similarly, Guo et al. (2023) highlighted the gap in LLM
performance for Chinese crosstalk, where models struggle
with the rhythmic cadence and the specific structural re-
quirements of the medium.(Wang et al., 2022) Our work
contributes to this evolving field by combining a novel multi-
LLM agent system with a RAG-based context injector and
an agent-specific fine-tuning strategy. By specifically train-
ing agents to play individual roles (e.g., the ”JokeWriter”
agent), we explore whether the synergy of specialized roles
and retrieved comedic materials can overcome the empathy-
novelty gap identified in recent benchmarks.
2

OpenMic: A Multi-Agent-Based Stand-Up Comedy Generation System
Figure 2.Backdoor Criterion
Figure 3.Frontdoor Criterion
3. Humor and Cognitive Reasoning
Humor is often treated as a stylistic property of language,
yet many jokes are better understood asreasoning processes
that manipulate an audience’s expectations over time. A
stand-up punchline rarely succeeds by lexical novelty alone;
rather, it relies on a latent chain of inferences that (i) builds a
plausible interpretation, (ii) introduces a hidden connection,
and (iii) triggers a rapid “re-interpretation” that resolves the
incongruity. This view is especially important for Chinese
stand-up comedy, where effective jokes frequently depend
on shared cultural context, implicit premises, and tightly
controlled information release. In this section, we frame hu-
mor as a form of structured cognitive reasoning and use two
illustrative logic diagrams (Fig. 2 and Fig. 3) to highlight
distinct inference patterns that commonly appear in jokes.
3.1. Humor as a Form of Cognitive Incongruity
Classic humor theories emphasizeincongruity: a joke sets
up an expectation and then violates it in a way that still per-
mits a coherent resolution. Under an incongruity–resolution
perspective, “being funny” is not merely about generating
surprising words, but about creating acontrolled mismatch
between the audience’s predicted continuation and the even-
tual reinterpretation that makes the punchline sensible. Con-
cretely, the setup implicitly constructs a mental model of
the situation; the punchline either flips a key assumption orreveals a hidden linkage that forces the audience to update
that model. The comedic effect arises from thecontrastbe-
tween the initial expectation and the revised interpretation,
as well as the speed and clarity with which the resolution
becomes apparent.
3.2. Humor Requires Multi-Step Reasoning
Many jokes embed a multi-step inference chain rather than
a single-step association. Fig. 2 provides an example we
refer to as abackdoor-stylestructure: the question entity EQ
and answer entity EAare not directly connected by surface
meaning, but are linked through an intermediate bridging
entity EZ(often a homophone, pun, or shared attribute). In
the illustrated joke (“Why was the cookie sad?”), the surface
reading suggests an emotional explanation; the resolution
depends on mapping to the phonetic/lexical bridge (e.g.,
“away for long” ↔“a wafer long”), which then retroac-
tively makes the punchline interpretable. Here, EZacts as
a hidden connector that is easy to miss unless one actively
searches for alternative interpretations.
In contrast, Fig. 3 illustrates afrontdoor-stylemulti-hop
reasoning pattern: the setup encourages the audience to
traverse intermediate thoughts explicitly before arriving at
the punchline. In the example (“What does a clock do when
it’s hungry?” →“It goes back for seconds.”), the humor
hinges on composing several simple steps: “hungry” evokes
a desire for food; “seconds” can mean a second helping; and
“clock” relates to time, enabling the wordplay “goes back
for seconds.” Compared with the backdoor-style pun, the
intermediate entity EZin this case is not merely a hidden
phonetic bridge but aconceptual stepping stonethat the
listener can traverse through associative and compositional
reasoning.
These two patterns are common in stand-up: (i)delayed
punchlinesresemble multi-step inference with intentionally
withheld bridges, and (ii)callbacksresemble long-range
reasoning where an earlier premise is reactivated under a
new interpretation. As a result, humor quality depends not
only on what is said, but onwhenthe crucial bridge is
revealed and how reliably the audience can reconstruct the
implicit reasoning path.
3.3. Why LLMs Struggle with Humor
Despite strong general language ability, LLMs frequently
underperform on humor because fluent continuation does
not guarantee thecognitive surpriserequired for a joke.
First, models tend to collapse the inference process: they
may reveal the bridge too early, explain the joke explicitly,
or smooth over ambiguity—all of which reduce comedic ten-
sion. Second, many jokes require maintaining two compet-
ing interpretations until the punchline; this demands deliber-
ate control of uncertainty and information release, whereas
3

OpenMic: A Multi-Agent-Based Stand-Up Comedy Generation System
next-token prediction often favors a single dominant contin-
uation. Third, humor is highly sensitive to pragmatic con-
straints (social norms, persona, cultural presuppositions), so
even logically consistent outputs may fail to land as funny if
the implied premises are unnatural for the target audience.
The reasoning structures in Fig. 2–3 also expose a practical
issue: the model mustsearchover potential bridges EZ
(phonetic, semantic, or contextual) and thenstagethe reveal
at the right moment. Without explicit mechanisms for plan-
ning, critique, and timing control, single-shot generation
often produces either (i) coherent but unfunny narration,
or (ii) isolated one-liners that lack buildup, callbacks, and
performance rhythm.
3.4. Implication: Humor as Structured Reasoning
Viewing humor as structured reasoning suggests that effec-
tive stand-up generation is closer to a pipeline ofplanning,
verification, andexecutionthan to free-form text genera-
tion. Planning selects comedic angles and determines which
bridge EZto hide or surface and when; verification checks
whether the reasoning path is reconstructible and whether
the punchline resolves the incongruity; execution adds stage-
performance cues (pauses, emphasis, and rhythm) that regu-
late information release. This framing motivates our system
design: instead of asking a single model to simultaneously
invent content, manage audience adaptation, and control
delivery, we decompose the process into specialized roles
and iterative refinement so that the final performance pre-
serves both the reasoning structure and the timing-sensitive
comedic payoff.
4. Methodology
4.1. Multi-Agent System Design
Why Multi-Agent for Stand-Up Comedy?As discussed in
Sec. 3.1, humor is tightly coupled withstructured reasoning
andtiming-aware information release. In practice, Chinese
stand-up generation involves several competing objectives
that are hard to satisfy in a single pass: (i)content planning
(selecting angles, building setups, placing callbacks), (ii)
audience adaptation(persona, taboo avoidance, cultural
priors), (iii)performability(spoken-style wording, rhythm,
pauses, emphasis), and (iv)quality assurance(coherence,
novelty, safety, “laugh potential”). A single-agent LLM
prompt tends to entangle these goals and often collapses
the latent reasoning structure (e.g., explaining the joke too
early) or drifts into generic narration.
We therefore adopt a multi-agent design that decomposes
the pipeline into specialized roles with explicit responsi-
bilities. This separation yields two practical advantages:
(1)controllability—each agent optimizes a well-defined
sub-objective with dedicated constraints; and (2)iterative
Figure 4.Mechanism of Autogen
refinement—failures can be localized and corrected (e.g.,
rewrite a weak punchline without redoing audience profil-
ing), which is essential for timing-sensitive comedic arcs.
4.2. AutoGen as the Orchestration Backbone
We implement OpenMic on top of AutoGen’s group-chat
abstraction. Conceptually, AutoGen coordinates a set of
conversable agents via a manager that (i) decides whose
turn it is to speak, (ii) collects the agent’s output, and (iii)
broadcasts relevant messages to other agents for subsequent
turns (Fig. 4). This “turn-taking + broadcast” mechanism
is a natural fit for creative collaboration: agents can oper-
ate asynchronously in intent (each has its own rubric), yet
remain synchronized through shared context.
In our implementation, we use aGroupChatManagerto
enforce an ordered protocol and to prevent uncontrolled
multi-agent chatter. Each agent is instantiated as aCon-
versableAgentwith a role-specific system prompt, in-
put/output schema, and constraints. The manager schedules
agents following our workflow (Sec. 4.6) and handles termi-
nation conditions (either PASS from the quality controller
or a maximum iteration budget).
4.3. Blackboard-Centric Coordination
Beyond message passing, OpenMic employs ablackboard
to maintain structured shared state (Fig. 5). The blackboard
stores intermediate artifacts that must persist across turns
and iterations, including:
•Audience profile: persona, preferences, taboo list, ac-
ceptable language register;
•Topic expansion: subtopics, personal anecdotes an-
gles, candidate premises;
•Draft script: current version of the stand-up text with
section boundaries;
•Performance markup: a structured DSL with pauses,
4

OpenMic: A Multi-Agent-Based Stand-Up Comedy Generation System
Figure 5.Multi-agent Structure.
emphasis, applause beats;
•Critique & action items: concrete revision instruc-
tions and failure reasons.
This design prevents critical information from being lost
in long conversational context and makes the iteration loop
more deterministic: each agent reads from and writes to
designated blackboard fields, rather than relying solely on
implicit conversational memory.
4.4. Agent Roles and Interfaces
OpenMic consists of five core agents (Fig. 5), each with a
narrowly-scoped responsibility and a typed output that is
written to the blackboard:
AudienceAnalyzer (audience modeling).Given the user
topic and optional style constraints, the AudienceAna-
lyzer produces an audience-facingpersona cardand a
taboo/avoid list(e.g., sensitive references, overly offensive
wording) to ensure cultural and situational appropriateness.
ComedyDirector (high-level planning).The Comedy-
Director decomposes the topic into a set ofsubtopicsand
acomedic structure plan(e.g., opening hook, 2–3 bits, a
callback, closing tag). The output is an outline with explicit
comedic intent: where tension is built, where bridges are
revealed, and where callbacks should land.
JokeWriter (script drafting).Conditioned on the outline
and audience profile, the JokeWriter produces acomplete
draft script. We instruct it to maintain spoken Chinese,
enforce setup–punchline delays, and preserve long-range de-
pendencies (callbacks) rather than generating disconnected
one-liners.
PerformanceCoach (delivery & markup).The Perfor-
manceCoach transforms the draft into aperformance-ready
scriptby adding a structured DSL annotation, including
pauses (e.g., pre-punchline pause), emphasis, pace changes,
filler words, and optional applause/laughter cues. Thisbridges text generation with downstream audio/video ren-
dering.
QualityController (evaluation & gating).The Quality-
Controller acts as a critic and gatekeeper. It evaluates co-
herence, comedic payoff, timing realism, and audience fit,
then outputs eitherPASSorREVISIONwith actionable ed-
its. This turns subjective humor quality into an operational
criterion for iteration.
4.5. Hierarchical Multi-Agent RAG with Information
Isolation
Our RAG framework is designed to bridge the gap between
simple semantic retrieval and genuine creative transforma-
tion. Rather than relying on a traditional single-step retrieval
process, we implemented atriadic inner-conversation ar-
chitecturethat utilizes one retrieval engine alongside two
specialized LLM agents. This system is governed by a
custom protocol to ensure that the massive volume of data
required for candidate selection does not clutter the primary
workflow’s context window.
Dataset Composition and Post-ProcessingTo ensure
stylistic consistency, our retrieval corpus combines two pri-
mary sources. The first is a collection of short-form setups
and punchlines sourced from the CFUN repository(Yu et al.,
2025). The second is aCrosstalk-to-Talkshow Pipeline
where we took traditional crosstalk scripts and pushed them
through an LLM-driven refinement stage. During this pro-
cess, we performed anonymization by removing specific
performer names and executed a stylistic conversion. This
turned dialogue-heavy routines into narrative-driven talk-
show observations, moving away from the classic ”teasing
and reacting” dynamic to a more modern first-person per-
spective.
The Triadic Inner-Conversation WorkflowStandard se-
mantic matching often prioritizes factual similarity over
comedic value. To fix this, we formalize the RAG pro-
cess as a sequence of three specialized operations: retrieval,
scoring, and refinement.
Letqrepresent the user topic query and Dthe integrated
comedic corpus. We define E(·) as the embedding function
that maps text to a high-dimensional vector space. The
process is defined as follows:
1.Semantic Retrieval:The RAG Retriever identifies a
set of raw candidates Cby calculating the cosine simi-
larity between the query and document embeddings:
C=top-k 1{d∈ D |sim(E(q), E(d))}
where sim(u,v) =u·v
∥u∥∥v∥.
2.LLM Candidate Scoring:The LLM Candidate Scorer
5

OpenMic: A Multi-Agent-Based Stand-Up Comedy Generation System
(Agent 1) acts as a non-linear semantic filter. It evalu-
ates the comedic potential Pof each candidate c∈ C
based on latent features like incongruity and relevance,
selecting a subsetSof high-potency jokes:
S={c∈ C |f scorer(c, q)> τ}
where fscorer is the agent’s internal evaluation function
andτis the quality threshold for the top-k 2selection.
3.LLM Punchline Refinement:Finally, theLLM
Punchline Selector(Agent 2) performs the creative
transformation T. Instead of passing the full text, it
distills the selected jokes into a set of writing materials
M:
M=[
s∈STselector (s)
This ensures the JokeWriter receives a distilled set of high-
potency building blocks Mrather than a wall of raw, un-
organized text, significantly reducing context noise while
maximizing creative signal.
The ”Secret Blackboard” and Context ManagementA
key technical feature of our architecture is theSecret Black-
board. During the inner-conversation between the RAG
engine and the retrieval agents, thousands of tokens of raw
material are processed simultaneously. Storing this in the
main global blackboard would quickly exceed the context
limits of agents further down the line, such as the Perfor-
mance Coach. To solve this, the Secret Blackboard acts as
a private memory buffer. It only releases the final refined
punchlines to the JokeWriter, effectively hiding the noisy
retrieval process from the rest of the chain and maintaining
a high signal-to-noise ratio across the entire system.
4.6. Multi-Round Refinement
If the QualityController returns REVISION , the system re-
enters the loop by routing feedback back to the JokeWriter
(and optionally the PerformanceCoach) until either PASS is
obtained or a maximum number of rounds is reached. This
multi-round loop is crucial for stand-up: jokes often fail due
to localized issues (weak punchline, premature explanation,
missing callback trigger, unnatural pause placement) that are
best fixed through targeted rewrites rather than regenerating
everything from scratch.
Dual-Dimension Quality AssessmentUnlike monolithic
QA systems, our QualityController performsdual-
dimension evaluation Qr= (QR
r, QW
r)to separately as-
sess retrieval quality and writer quality.
RAG Dimension( QR
r): Evaluates retrieved joke material
quality—humor potential, topic relevance, and diversity.
WhenQR
r= 0, the QA outputs:•k∗: refined keywords
•Er: joke IDs to exclude in next retrieval
•fR
r: specific feedback per joke
Writer Dimension( QW
r): Evaluates script organization via
three checks:
QW
r=⊮[struct]∧⊮[safe]∧⊮[length](1)
Failed checks triggerrewrite directives dr(e.g., “callback
missing for setup in line 3”).
Targeted Refinement RoutingThe dual evaluation en-
ables surgical fixes:
Case 1: RAG fails, Writer succeeds(QR
r= 0, QW
r= 1):
Re-retrieve:D r+1←RAG(k∗,Er)(2)
Case 2: Writer fails, RAG succeeds(QR
r= 1, QW
r= 0):
Rewrite:s r+1←Writer(D r,dr)(3)
Case 3: Both fail(QR
r= 0, QW
r= 0):
Dr+1, sr+1←RAG(k∗,Er) +Writer(·,d r)(4)
Termination occurs whenQR
r∧QW
r= 1orr≥R max.
Context-Aware MemoryEach writer receives structured
feedback:
Cr={s r−1,Dr−1,dr−1,
QR
r−1,Pr−1}(5)
where Pr−1are preserved joke IDs. This prevents agents
from “forgetting” prior decisions across rounds.
Empirically, Case 3 occurs in ∼30% of round-1 attempts
but drops to <5% by round 3, indicating rapid convergence.
4.7. Domain-Specific Adaptation via QLoRA
To bridge the gap between general-language capabilities
and specialized comedic timing, we employQuantized
Low-Rank Adaptation (QLoRA). This approach allows
for the fine-tuning of large-scale models by injecting train-
able low-rank matrices into the frozen, 4-bit quantized base
model. For a weight matrix W0∈Rd×k, the forward pass
is modified as:
h=W 0x+ ∆Wx=W 0x+BAx
where B∈Rd×randA∈Rr×kare the low-rank adapters
with rank r≪min(d, k) . We specifically target all lin-
ear projections within the transformer blocks to maximize
6

OpenMic: A Multi-Agent-Based Stand-Up Comedy Generation System
Table 1.Evaluation Results across Different Temperature Settings
Configuration Persona Reactivity Humor Narrative Coherence
JW+Tem0.1 82.5 88.0 95.5 93.0 97.5
JW+Tem0.3 88.5 15.0 96.0 93.0 97.5
JW+Tem0.5 82.5 15.0 96.5 93.0 98.5
JW+Tem0.7 92.5 45.0 96.0 98.5 97.0
JW+Tem0.9 85.0 25.0 92.0 94.0 96.0
the model’s stylistic plasticity. Furthermore, to ensure the
model focuses exclusively on comedic delivery, we utilize a
completion-only loss strategy, calculating gradients only on
the generated punchlines rather than the instruction prompts.
5. Experiments
5.1. LLM-as-a-Judge Evaluation Framework
We implemented a rigorous ”LLM-as-a-Judge” mecha-
nism to quantify the quality of the generated talk show
scripts. Unlike standard NLP metrics (such as BLEU
or ROUGE), which often fail to capture the semantic
nuance and comedic timing of creative writing, we uti-
lized a senior executive producer persona—powered by
theGrok-4-1-fast-reasoning model—to conduct
a multi-dimensional scoring analysis.The evaluation is gov-
erned by a Pydantic-enforced schema, ensuring that every
assessment is structured across five critical dimensions:
•Persona Fidelity (30%):The distinctiveness and con-
sistency of the characters’ voices.
•Humor Mechanics (25%):The density and structural
quality of setup-punchline sequences.
•Interactive Reactivity (20%):The degree of ”impro-
visational” riffing and response to previous turns.
•Contextual Coherence (15%):The logical consis-
tency and effective use of callbacks.
•Narrative Arc (10%):The rhythmic flow from intro-
duction to climax and resolution.
The final weighted scoreS total is calculated as:
Stotal= 0.30P+ 0.25H+ 0.20R+ 0.15C+ 0.10N
where P, H, R, C, N represent the scores for Persona, Hu-
mor, Reactivity, Coherence, and Narrative respectively.
Figure 6.visualization of generation scores under different tem-
perature paramters
5.2. Influence of Temperature on Generative
Performance
While initial metrics exhibited a high degree of variance,
they provided critical insights into the relationship between
sampling temperature and the efficacy of our RAG-enhanced
retrieval. We observed that the system’s performance is not a
linear function of stochasticity, but rather a delicate balance
between contextual grounding and creative divergence.
Our analysis indicates thatlow temperature settings (e.g.,
T= 0.1 )combined with thelarge joke corpus retrieved
via RAGyield the most superior results. As shown in 10,
which compares two specific generative examples, lower
temperatures allow the model to maintain a high ”focus”
on the specific comedic building blocks provided by the
RAG inner-conversation. In this regime, theContextual
Coherence(97.5) andInteractive Reactivity(88.0) are
maximized, as the model accurately maps the retrieved
punchlines onto the target persona without drifting into
irrelevant hallucinations.
Conversely, athigher temperatures (e.g., T≥0.7 ), the
model begins to lose its grip on the retrieved context. While
this occasionally results in a spike inNarrative Arc(98.5 at
T= 0.7 ) as the model explores more varied sentence struc-
tures, it frequently compromisesReactivity. Qualitative
review suggests that at high temperatures, the JokeWriter
often ignores the specific ”setup” provided by the RAG
selector in favor of generic, less interesting tropes.
Ultimately, we conclude that the optimal configuration for
generative crosstalk lies in minimizing entropy to maximize
retrieval signal. By utilizing a low temperature, we ensure
that the fine-tuned model acts as a precision instrument that
”assembles” the retrieved comedic materials into a cohesive
script, rather than attempting to hallucinate humor with-
out sufficient grounding. This reinforces the value of our
RAG-centric approach: the ”creativity” is supplied by the
diversity of the corpus, while the ”logic” is preserved by the
7

OpenMic: A Multi-Agent-Based Stand-Up Comedy Generation System
constrained sampling.
5.3. Finetuning implementation Details and
Hyperparameters
Our fine-tuning experiments were conducted on a single
GPU using the trl andpeft libraries. The training corpus
consists of the LLM-processed Talkshow dataset, formatted
using the Qwen-2.5 chat template.
Fine-tuning Configuration:We utilized the
LoraConfig to target a comprehensive set of mod-
ules, including q proj, k proj, v proj, o proj, and the MLP
layers (gate, up, down proj). The detailed hyperparameter
settings are summarized in Table 2.
Table 2.Hyperparameters for Comedy-Specialized QLoRA Fine-
tuning
Hyperparameter Value
Base Model Qwen-2.5-3B-Instruct
Quantization 4-bit NF4 (NormalFloat)
LoRA Rank (r) 16
LoRA Alpha (α) 32
LoRA Dropout 0.05
Learning Rate2×10−4
Optimizer Paged AdamW 32-bit
Batch Size (Per Device) 2
Gradient Accumulation 4
Training Epochs 1
Compute Precision BF16 (or FP16)
Completion-Only Training:To prevent the model from
overfitting on the instruction syntax, we implemented a Man-
ualCompletionCollator. By defining the response template
as"<|im start|>assistant\n" , the trainer effec-
tively masks the prompt tokens during loss calculation. This
ensures the negative log-likelihood loss Lis computed only
on the tokensy ibelonging to the assistant’s response:
L=−X
i∈ResponselogP(y i|y<i,Prompt)
The final model was deployed via a vLLM entrypoint with
LoRA support enabled, allowing for high-throughput infer-
ence during the multi-agent execution phase.
5.4. Demo: End-to-End Stand-Up Generation
Goal.We present an end-to-end demo to illustrate how
our multi-agent pipeline generates a Chinese stand-up mono-
logue from a high-level prompt, highlighting (i) structured
setup–punchline planning, (ii) iterative critique and rewrit-
ing, and (iii) callback triggering across the script.Setup.We run the system for3iterations producing inter-
mediate artifacts includingQuality Evaluations.
5.5. Downstream Application: End-to-End Video
Synthesis
To further demonstrate the practical utility of OPENMIC,
we extend the pipeline to a multi-modal application stage.
This stage verifies that the structured performance scripts,
enriched with behavioral cues, can be seamlessly executed
by external rendering engines to produce broadcast-ready
content.
Implementation Workflow:The video synthesis process
acts as a specialized consumer of thePerformanceCoach’s
output. We implement a middleware that parses the em-
bedded DSL markers—such as [pause] ,[emphasis] ,
and[applause] —to construct a synchronized temporal
timeline. By invoking RESTful APIs from high-fidelity dig-
ital human platforms (e.g., Kling AI), the system maps the
synthesized audio onto a 3D-animated avatar. The synchro-
nization logic ensures that the avatar’s micro-expressions,
such as eyebrow movements during a setup and a smirk
during a punchline, are aligned with the comedic rhythm
defined in the script.
Key Technical Challenges and Observations:
•Temporal Consistency:The use of structured mark-
ers prevents the common “robotic delivery” seen in
standard Text-to-Speech (T2S) systems. By explicitly
injecting silence durations and speech rate variations
based on the DSL, we preserve the timing-sensitive
nature of Chinese stand-up comedy.
•Cross-Modal Stylistic Alignment:The visual per-
sona, including stage background illumination and
character attire, is dynamically selected to match the
AudienceAnalyzer’s persona card. This ensures a coher-
ent comedic atmosphere where the visual environment
reinforces the linguistic tone.
•Performance Fidelity:Our pipeline automates the
generation of a 3–5 minute narrated video from a single
topic prompt. This end-to-end capability demonstrates
the robustness of OPENMICnot only as a writing assis-
tant but as a comprehensive production tool for digital
entertainment.
The integration of video synthesis completes the generative
loop, providing a tangible interface for evaluating the per-
formability of the generated humor in a real-world setting.
Acknowledgments
This project is a course final assignment for the CoRE
course. It was developed by the group “King of Comedy”.
8

OpenMic: A Multi-Agent-Based Stand-Up Comedy Generation System
References
Balaguer, A., Benara, V ., de Freitas Cunha, R. L., de M. Es-
tev˜ao Filho, R., Hendry, T., Holstein, D., Marsman, J.,
Mecklenburg, N., Malvar, S., Nunes, L. O., Padilha,
R., Sharp, M., Silva, B., Sharma, S., Aski, V ., and
Chandra, R. Rag vs fine-tuning: Pipelines, tradeoffs,
and a case study on agriculture, 2024. URL https:
//arxiv.org/abs/2401.08406.
Chen, J., Zhu, X., Yang, C., Shi, C., Xi, Y ., Zhang, Y ., Wang,
J., Pu, J., Zhang, R., Yang, Y ., and Feng, T. Hollmwood:
Unleashing the creativity of large language models in
screenwriting via role playing, 2024. URL https://
arxiv.org/abs/2406.11683.
Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer,
L. Qlora: Efficient finetuning of quantized llms, 2023.
URLhttps://arxiv.org/abs/2305.14314.
Kim, S. and Chilton, L.AI Humor Generation: Cognitive,
Social and Creative Skills for Effective Humor. 02 2025a.
doi: 10.48550/arXiv.2502.07981.
Kim, S. and Chilton, L. B. Ai humor generation: Cognitive,
social and creative skills for effective humor, 2025b. URL
https://arxiv.org/abs/2502.07981.
Loakman, T., Thorne, W., and Lin, C. Who’s laughing
now? an overview of computational humour genera-
tion and explanation. In Flek, L., Narayan, S., Phuong,
L. H., and Pei, J. (eds.),Proceedings of the 18th In-
ternational Natural Language Generation Conference,
pp. 780–794, Hanoi, Vietnam, October 2025. Associ-
ation for Computational Linguistics. URL https:
//aclanthology.org/2025.inlg-main.45/.
Sakabe, R., Kim, H., Hirasawa, T., and Komachi, M. Assess-
ing the capabilities of llms in humor:a multi-dimensional
analysis of oogiri generation and evaluation, 2025. URL
https://arxiv.org/abs/2511.09133.
Sanmartin, D. Kg-rag: Bridging the gap between knowledge
and creativity, 2024. URL https://arxiv.org/
abs/2405.12035.
Tikhonov, A. and Shtykovskiy, P. Humor mechanics: Ad-
vancing humor generation with multistep reasoning, 2024.
URLhttps://arxiv.org/abs/2405.07280.
Wang, B., Wu, X., Liu, X., Li, J., Tiwari, P., and Xie, Q.
Can language models make fun? a case study in chi-
nese comical crosstalk, 2022. URL https://arxiv.
org/abs/2207.00735.
Yu, Z., Hu, X., and Wan, X. Cfunmodel: A ”funny” lan-
guage model capable of chinese humor generation and
processing, 2025. URL https://arxiv.org/abs/
2503.20417.
9

OpenMic: A Multi-Agent-Based Stand-Up Comedy Generation System
Iteration 3
DSL Snippets
[FAST]
我妈有个粉色保温杯，上面印着‘温柔妈妈’四个字，底下还有一行小字：
[SLOW]
 ‘情绪管理示范基地’。
[PAUSE=1.0]
[NORMAL]
她说这是她参加完家庭教育讲座领的纪念品，
[EMPH]
每次开家长会必带，
[PAUSE=0.5]
 说是能镇魂。
[SIG=Haiyaa]
[SLOW]
那天我物理考了28分，
[PAUSE=0.8]
教室里安静得像殡仪馆播放课间操音乐。
[PAUSE=1.2]
[FAST]
老师念到我名字时，我妈立刻拧开杯子猛灌一口，
[EMPH]
然后递给我爸：
[PAUSE=0.3]
 ‘你喝！这火气不是我一个人的！’
[SIG=Tsk]
[NORMAL]
我爸刚要推辞，
[PAUSE=0.5]
老师突然鼓掌：
[SLOW]
‘下面请张明同学家长分享经验！’
 [PAUSE=1.0]
[FAST]
张明是谁？就是那种你妈说
[EMPH]
‘你要是有他一半优秀，我就能提前退休’的那种人。
 [PAUSE=0.7]
[NORMAL]
他妈妈站起来，旗袍盘扣都透着优越感，
[SLOW]
从包里掏出三本习题集：
[PAUSE=0.5]
 ‘唉，孩子太贪玩，
[NORMAL]
昨晚才做完剑桥预科卷，
[FAST]
今早顺手背了篇《自然》期刊摘要。’
 [PAUSE=1.0]
[SLOW]
我正怀疑这孩子是不是外星派来打击地球学生的，
[PAUSE=0.8]
忽然听见他小声对他妈说：
 [PAUSE=0.5]
[EMPH]
‘妈，下周奥赛班要交量子力学读书报告……能不能换补习班？我想去普通人类班。’
[SIG=Haiyaa]
 [PAUSE=1.5]
[SLOW]
我瞬间释然了。
[PAUSE=0.5]
原来你们家的天才，
[EMPH]
也是被腌在补习缸里的泡菜，
 [PAUSE=0.3]
只不过你们用的是法国海盐。
[SIG=■]
[PAUSE=1.0]
[NORMAL]
回家路上，我妈没骂我，反而幽幽地说：
[SLOW]
‘要不……咱也报个班？’
[PAUSE=1.0]
[FAST]
我说您别折腾了，咱家连Wi-Fi都是租的。
[PAUSE=0.5]
结果第二天，
[SLOW]
 她真把那保温杯供在书桌上，
[EMPH]
天天对着它默念：
[PAUSE=0.3]
 ‘让我儿子及格吧，让我儿子及格吧……’
[SIG=Emotional Damage]
[PAUSE=1.2]
[NORMAL]
前天亲戚来串门，一眼看到冰箱上的杯子照片，
[EMPH]
惊了：
[PAUSE=0.3]
 ‘哎哟，这不是我们家补习机构发的“焦虑缓冲杯”吗？
[FAST]
买满十万课时送的！’
[SIG=Haiyaa]
 [PAUSE=1.5]
[SLOW]
我当场愣住——
[PAUSE=0.5]
原来全城爸妈都在用同一个杯子，
[PAUSE=0.3]
 说着同一句咒语，
[PAUSE=0.3]
供养着同一个神话。
[PAUSE=2.0]
Quality Evaluation
Writer Feedback:
• Setup 开场照搬‘最怕空气突然安静’热梗，原创性为零，你是来参加脱口秀还是段子复制大赛？
• Punchline 1 的‘反人类天赋’情绪到位，但前情铺垫太急，缺少对‘别人家孩子’出场的仪式感烘托，笑点被
• 压扁了
• 转折段‘路过他们座位’太随意，像是偷听广播剧，缺乏空间动线和心理过渡，观众会被闪到腰椎间盘
RAG Feedback:
• 从脚本反推，RAG 检索的‘家长会社死’和‘别人家孩子补课’梗具有高共鸣性和校园真实性
• ‘补习班神话破灭’与‘全城父母同款焦虑杯’体现社会洞察，非低质堆笑料
Iteration 3 - Multi-round Refinement with RAG
10

OpenMic: A Multi-Agent-Based Stand-Up Comedy Generation System
Figure 7.GPT. Qualitative example of Chinese stand-up generation on the same topic prompt.
A. Single Agent Example
B. Single Finetuned Agent Example
C. Different Temperature Setting Examples
11

OpenMic: A Multi-Agent-Based Stand-Up Comedy Generation System
Figure 8.DeepSeek. Qualitative example of Chinese stand-up generation on the same topic prompt.
12

OpenMic: A Multi-Agent-Based Stand-Up Comedy Generation System
Figure 9.QLora finetuned Qwen3-4B-Instruct
13

OpenMic: A Multi-Agent-Based Stand-Up Comedy Generation System
Figure 10.Left one generated with 0.1 temperature, right one with 0.9 temperature
14
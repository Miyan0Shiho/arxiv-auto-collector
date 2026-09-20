# LSREP: A Longitudinal State-Replay Protocol for Evaluating Conversational Memory, with ICE v2 as an Audited Local-First Architecture

**Authors**: Deepesh Sonar

**Published**: 2026-09-15 06:59:56

**PDF URL**: [https://arxiv.org/pdf/2609.16730v1](https://arxiv.org/pdf/2609.16730v1)

## Abstract
Conversational memory changes during use, so endpoint question answering alone cannot establish how a persistent state accumulates, ages, or incorporates revisions. We introduce LSREP, a Longitudinal State-Replay Evaluation Protocol combining ordered replay, explicit lifecycle schedules, repeated probes, evolving reference answers, and mechanism-fidelity checks. Its architectural case study is ICE v2, a local-first memory middleware with typed stores, retrieval fusion, and dynamic context budgets. The private, single-user instantiation contains 1,985 turns, 219 distinct probes, and 1,211 probe-checkpoint observations across 52 checkpoints. On three ordinary-density datasets, ICE v2 has a near-zero mean quality difference from vector-RAG while selecting 32% fewer fragments but using 6.6% more estimated prompt tokens. A fourth, dense dataset exposes catastrophic failures of the unbudgeted baseline. The fidelity audit limits attribution: procedural retrieval is defective, several mechanisms are unexercised, and graph utility is not established. In a complementary matched public diagnostic, ICE v2 loses decisively to pure vector-RAG on LongMemEval: 50.8% versus 72.8% in the evidence-only oracle and 43.0% versus 69.5% in full-S. Paired differences are -22.0 points (95% CI [-26.6, -17.4]) and -26.5 ([-31.3, -21.8]). Conservative abstention accompanies severe multi-session and temporal failures. ICE uses less context in this diagnostic, establishing a quality-cost trade-off rather than superior efficiency. Together, replay, fidelity auditing, and public endpoint testing expose distinct failure modes that neither architectural descriptions nor aggregate scores identify alone.

## Full Text


<!-- PDF content starts -->

LSREP: A Longitudinal State-Replay Protocol for
Evaluating Conversational Memory, with ICE v2 as
an Audited Local-First Architecture
Deepesh Sonar
Thakur College of Engineering and Technology, Computer Engineering, Mumbai, India
18deepnar@gmail.com
Abstract
Conversational memory changes during use, so endpoint question answering alone cannot
establish how a persistent state accumulates, ages, or incorporates revisions. We introduce
LSREP, a Longitudinal State-Replay Evaluation Protocol combining ordered replay, explicit
lifecycle schedules, repeated probes, evolving reference answers, and mechanism-fidelity
checks. Its architectural case study isICE v2, a local-first memory middleware with typed
stores, retrieval fusion, and dynamic context budgets. The private, single-user instantiation
contains 1,985 turns, 219 distinct probes, and 1,211 probe–checkpoint observations across
52 checkpoints. On three ordinary-density datasets, ICE v2 has a near-zero mean quality
difference from vector-RAG while selecting 32% fewer fragments but using 6.6% more
estimated prompt tokens. A fourth, dense dataset exposes catastrophic failures of the
unbudgeted baseline. The fidelity audit limits attribution: procedural retrieval is defective,
several mechanisms are unexercised, and graph utility is not established. In a complementary
matched public diagnostic, ICE v2 loses decisively to pure vector-RAG on LongMemEval:
50.8% versus 72.8% in the evidence-only oracle and 43.0% versus 69.5% in full-S. Paired
differences are −22.0 points (95% CI [ −26.6,−17.4]) and −26.5 ([−31.3,−21.8]). Conservative
abstention accompanies severe multi-session and temporal failures. ICE uses less context in
this diagnostic, establishing a quality–cost trade-off rather than superior efficiency. Together,
replay, fidelity auditing, and public endpoint testing expose distinct failure modes that
neither architectural descriptions nor aggregate scores identify alone.
1 Introduction
Absent an explicit persistence layer, a language model can use only what remains in its current
context. Enlarging that context does not make every token equally usable: models attend
unevenly across long inputs and degrade on information away from their edges (Liu et al.,
2024), while effective context is often shorter than the advertised limit (Hsieh et al., 2024).
Commercial assistants now provide cross-session memory, so literal session-level amnesia is no
longer a universal description. The open questions are instead what is retained, how revisions
and forgetting are represented, and whether the user can inspect and govern the store.
Research has approached those questions through multi-session dialogue (Jang et al., 2023; Xu
et al., 2022), long-history benchmarks (Maharana et al., 2024; Wu et al., 2025), and personalised
adaptation (Salemi et al., 2024). These studies examine complementary aspects of long-term
interaction. Endpoint QA over supplied histories can test knowledge updates and temporal
reasoning, but does not by itself require repeated observations of a reconstructed memory state
under a declared maintenance and access schedule.
The Infinite Context Engine (ICE v2) is a local-first middleware around an otherwise stateless
answering model. It stores episodic turns, a temporally-versioned knowledge graph, procedural
patterns, documents, and user-controlled memory slots; classifies each prompt; retrieves and
arXiv:2609.16730v1  [cs.AI]  15 Sep 2026

fuses candidates; and assembles a bounded context. Externalising memory preserves it when the
answering model changes and makes the store inspectable. ICE is the case study here, not the
definition of the evaluation problem.
That evaluation problem is longitudinal. A system may retrieve well at one endpoint while
failing to update an old fact, may appear efficient only because a component never ran, or may
work inside one continuing conversation but fail when a fresh session must aggregate earlier ones.
We therefore introduce LSREP alongside, rather than in place of, fixed-history benchmarks. It
reconstructs state at successive checkpoints and allows the reference answer itself to evolve.
This paper makes three contributions.
1.LSREP, the primary contribution: a system-independent protocol specification for replaying
conversational history, reconstructing state at successive checkpoints, and scoring against
evolving ground truth. It is designed for other systems and exported histories; empirical
portability remains to be tested, and the private corpus used here is not independently
reproducible.
2.ICE v2 as an architectural contribution and audited case study: a local-first multi-
store architecture with intent-driven retrieval, access-weighted decay, weighted rank fusion,
and a per-query token budget. The evaluated snapshot is fixed at tagv2-paper-eval.
3.A two-regime evaluation with a fidelity audit: LSREP measures within-conversation
longitudinal behaviour and a density stress case; matched public LongMemEval oracle and
full-S conditions probe fresh-session aggregation and distractors. Their disagreement is itself
a result. A component audit separately identifies defective, unexercised, contributing, and
inconclusive mechanisms.
The methodological claim is deliberately falsifiable: a near-zero ablation delta supports
“does not help” only when the component is verified live and the interaction regime reaches
it. We report where our own evaluation failed that standard and constrain the system claims
accordingly.
Research questions. RQ1: How do answer quality and context use change as ICE v2’s
replayed state accumulates and ages?RQ2: Which mechanisms execute, affect the supplied
evidence, and have an identifiable quality effect?RQ3: Does the observed behaviour transfer
to endpoint QA over separately supplied sessions, with and without distractors? The third
question concerns transfer between regimes, not a competition between evaluation protocols.
All substantive system results concern frozen ICE v2; the v1 pilot is historical, and ongoing v3
development is outside this paper.
2 Related Work
2.1 Persistent and Structured Memory
MemGPT manages a bounded context through a hierarchy of working and external mem-
ory (Packer et al., 2023). MemoryBank combines memory storage with time-dependent forgetting
and reinforcement (Zhong et al., 2024). Generative Agents uses observations, reflection, and
retrieval to support simulated agents (Park et al., 2023). These establish persistent memory and
memory lifecycle management as existing ideas; ICE v2’s contribution is their integration into
an inspectable local middleware and the audit of that integration.
Mem0 explicitly extracts and updates memories, including ADD, UPDATE, DELETE, and
NOOP operations; its graph variant adds relational representations, and its cited evaluation
uses LoCoMo (Chhikara et al., 2025). It should therefore not be characterised as append-only
or unable to reconcile contradictions. Zep’s Graphiti engine represents temporally qualified
2

relationships and combines episodic, semantic, and community information (Rasmussen et al.,
2025). Temporal graph memory is consequently not unique to ICE v2. The relevant architectural
choices here are typed stores, explicit retrieval control, local deployment, and the observable
boundary between an implemented mechanism and an evaluated capability.
2.2 Graph and Adaptive Retrieval
GraphRAG uses extracted graphs and community summaries for query-focused summarisa-
tion (Edge et al., 2024); KGP navigates passage graphs for multi-document QA (Wang et al.,
2024b); Think-on-Graph explores reasoning paths over a knowledge graph (Sun et al., 2024); Hip-
poRAG combines a graph with personalised PageRank for associative retrieval (Guti´ errez et al.,
2024). These works motivate structural retrieval, but their reported experiments do not establish
the correctness of ICE v2’s evolving graph. That must be measured in the implementation under
study.
RAG, REALM, Fusion-in-Decoder, and REPLUG develop different couplings between re-
trieval and generation (Guu et al., 2020; Izacard and Grave, 2021; Lewis et al., 2020; Shi
et al., 2024). CRAG evaluates retrieved evidence, and Self-RAG learns retrieval and critique
decisions (Asai et al., 2024; Yan et al., 2024). These methods are not intrinsically incompatible
with changing corpora. LSREP adds a specification of when that corpus and the system’s derived
state change, and what answer is valid at each observation.
ICE v2 builds on lexical ranking, weighted reciprocal rank fusion (RRF), and an optional
HyDE rewrite (Cormack et al., 2009; Gao et al., 2023; Robertson and Zaragoza, 2009). Its
lexical leg uses PostgreSQL full-text ranking, historically named “BM25” in the code; it is not an
implementation of the canonical BM25 scoring formula. RRF combines ranks without requiring
comparable native scores. The v2 ablation supports a corrective effect after adding the lexical
leg; HyDE was not isolated (Section 7.2.1).
2.3 Evaluation Regimes
Multi-Session Chat evaluates dialogue that uses earlier sessions (Xu et al., 2022); Conversation
Chronicles incorporates time intervals and speaker relationships (Jang et al., 2023). LoCoMo
evaluates long conversational histories through QA and other tasks (Maharana et al., 2024);
LongMemEval separately tests information extraction, updates, multi-session reasoning, tem-
poral reasoning, and abstention (Wu et al., 2025). LaMP studies personalisation from user
profiles (Salemi et al., 2024). Their targets differ, and they should not all be reduced to static
single-shot retrieval.
LSREP’s proposed unit is thestate trajectory: a specified history prefix, lifecycle schedule,
reference version, and access history at each checkpoint. LongMemEval’s supplied histories allow
public endpoint comparisons that this paper’s private replay corpus cannot provide. Conversely,
repeated checkpoint observations allow us to inspect changes that a single endpoint score leaves
unresolved. Neither regime subsumes the other.
LLM judging makes free-form evaluation practical but introduces position, verbosity, and
other biases (Liu et al., 2023; Wang et al., 2024a; Zheng et al., 2023). Blinding and fixed
rubrics mitigate some risks; they do not establish unbiased labels. This distinction matters when
comparing our Muse-judged LongMemEval results with published GPT-4o-judged results.
3 LSREP: Requirements, Algorithm, and Validity
3.1 Evaluation Object and Requirements
LSREP evaluates areconstructed trajectory, not an exact recovery of an unlogged historical
deployment. Let H≤tcontain the supplied turns available by checkpoint t,Stthe persistent
state, Ltthe declared maintenance and ageing schedule, and Atthe access events allowed to
3

affect memory. A replay implementation realises
St=F(S t−1, H(t−1,t] , Lt, At−1;θ),
where θfixes the system version, models, configuration, and runtime contract. A probe qhas an
origin checkpoint o(q) and a reference G(q, t) derived only from H≤t. The observation is the
answer, its score against G(q, t), the actual context supplied, and the observable state transition
caused by the query.
An admissible system must expose an ingestion path, a query interface, a reset or snapshot
mechanism, and enough control to declare maintenance and query side effects. A chat endpoint
alone is insufficient. The protocol requires ordered history, explicit checkpoint boundaries,
temporally grounded references, consistent treatment of conditions, and versioned evidence of
what ran. It does not require graphs, decay, or any ICE-specific store: a vector index is a valid
instance whose lifecycle may simply append turns.
3.2 Protocol Algorithm
Algorithm 1: Longitudinal state replay
Input:ordered history H; checkpoints T; system conditions C; lifecycle schedule L; probes with
origins; versioned referencesG; query-mutation policyA.
1.Freeze code, model identities, parameters, clocks, and randomisation policy. Initialise each
condition’s declared state.
2.For each checkpoint t∈T , ingest only newly available turns, in order. Complete the declared
write and maintenance jobs; record completion and failures.
3.Construct or load G(q, t) for every eligible probe o(q)≤t, using only the prefix H≤t. Preserve
the probe’s referent and evidence provenance when updating its answer.
4.Query all conditions under the same checkpoint and reference version. Record scope, selected
context, costs, answer, and component activity. Apply the declared query side effects before
the next observation, or restore a snapshot if side effects are excluded.
5.Judge answers with the fixed rubric and missing-output rule. Aggregate paired differences
while retaining repeated-probe and conversation identities.
6.Report trajectories, quality–cost trade-offs, missingness, and mechanism-fidelity status. Sepa-
rate unexercised mechanisms from executed failures and measured effects.
Output:a versioned observation ledger and scoped claims about that replay process.
Figure 1:LSREP specification. Assertions describe the validity contract; the retrospective ICE v2 audit
identifies where the historical instantiation lacked sufficient instrumentation.
Independent reconstruction of each checkpoint is also possible, but it must reproduce every
permitted prior access event to represent the same trajectory. Replaying the same turns without
those events is a different experimental condition. A deterministic schedule does not guarantee
identical model outputs across executions.
3.3 Evolving Ground Truth
A reference records the tracked referent, supporting evidence available by the checkpoint, current
answer, and prior versions. When a new turn revises a fact, a current-truth probe changes
its reference; a question explicitly anchored to an earlier time retains its historical answer.
Ambiguous revisions require adjudication or an uncertainty label, rather than silently changing
the subject. References must remain outside the system’s retrievable memory. Repeatedly
answering an outdated reference correctly is not successful memory updating.
3.4 Validity Conditions and Scope
Temporal validityprohibits future evidence in state or references.Replay validityrequires
an explicit lifecycle and access schedule, including whether evaluation queries reinforce memory.
4

Measurement validityrequires complete prompt accounting, inspectable missing-output rules,
and judges that can discriminate relevant errors.Mechanism validityrequires evidence that a
claimed component executes and its output reaches a decision or answer context; non-empty
output alone does not establish utility.Statistical validityrequires stating the resampling
unit and the population to which uncertainty applies.
The protocol was developed alongside ICE and instantiated on one user’s private history.
This creates selection and authoring risks even when the system obtains negative results: an
unflattering result is not proof of an unbiased instrument. The v1 pilot exposed deficiencies in
context accounting and judging (Appendix B); the v2 audit exposed defective and unexercised
components. These motivate explicit validity conditions rather than establish universal validation
of LSREP. Independent users, systems, and publicly releasable trajectories remain needed.
4 Synthetic Worked Example
The following example is invented and contains no private corpus text. It illustrates the protocol,
not an additional ICE v2 experiment. At T1, answering PostgreSQL leaks future evidence. At
Table 1:A reference changes only when the history available at that checkpoint warrants it.
Time Newly available evidence Current-truth
probe/referenceHistorical
probe/reference
T1 “Project Atlas will use
SQLite for storage.”“Which database does
Atlas use?”→SQLite“What was the initial
database choice?”→
SQLite
T2 “We have replaced SQLite
with PostgreSQL for
Atlas.”Same question→
PostgreSQLSame question→
SQLite
T2, answering SQLite to the current-truth question repeats a superseded decision; answering
PostgreSQL to the historical question erases the revision history. An answer that states both
versions and their order may satisfy a richer evolution probe. The present v2 LSREP corpus
tests current truth; historical and evolution probes in this illustration specify capabilities a
future instantiation could test, not capabilities demonstrated by the present results.
5 ICE v2 Architecture and Audit Contract
ICE v2 is an OpenAI-compatible memory middleware between a conversational client and a
pool of locally-served models. It is not a model: every request addressed to the synthetic model
name ice-proxy is intercepted by a proxy that classifies the turn, retrieves context from four
long-lived memory stores, assembles a cache-friendly prompt, routes the request to a per-turn
specialist model, streams the response back, and then dispatches background workers that
extract, decay, cluster, and consolidate the new turn into long-term memory. The store may
grow without the answering model’s context window, but each request still receives a bounded,
selectively rebuilt view of it. This section describes the mechanisms the evaluation exercises;
Appendix E carries the implementation detail (classification engine, Codex internals, prompt
assembly, worker cluster, operational infrastructure), and the archived technical report in the
repository describes the system exactly as evaluated.
5.1 Request Lifecycle
Each turn traverses apre-flightphase (synchronous, in the request path) and apost-flight
phase (asynchronous, after the stream closes).
In pre-flight, the user message is classified into topic tags, intent tags, and a single context-
5

Client
requestPre-flight(synchronous, in request path)
classify →override →retrieve →fuse (RRF)
→budget →assemble →route (MoE)stream
response
Post-flight(asynchronous,
after stream closes)
store raw turn →evaluate representation
→summaries and typed-memory extractionPostgreSQL
+ pgvectordispatchread state
Figure 2:System overview. Pre-flight runs synchronously in the request path; post-flight runs asyn-
chronously after the stream closes and writes memory for later retrieval. Scheduled maintenance ages
and consolidates that state.
reliance label (Zero Shot, Long Term Memory, Real Time Search) by a two-stage cascade: a
rule-based pre-classifier resolves rule-matching cases, and on a miss a small multi-layer perceptron
over a frozen Qwen3-Embedding-0.6B encoder resolves the remaining cases (encoder time must
be accounted for separately). Hard override rules then coerce the label—most consequentially, a
conservative rule upgrading Zero Shot to Long Term Memory when a conversation exceeds ten
turns or confidence falls below 0.95, so retrieval runs for almost every turn in a long conversation;
Section 8 discusses what that costs. A Hybrid Retrieval Orchestrator then runs the retrieval legs,
fuses them with weighted Reciprocal Rank Fusion, and post-processes the fused list. A Prompt
Assembler concatenates the surviving fragments with persistent memory slots, recent turns, and
the live message under a stable prefix, and a Mixture-of-Experts router selects a locally-served
model with per-conversation stickiness.
In post-flight, an evaluator runs lossless detection—the guiding principle in the code is
thatmemory is earned: raw turn text is stored first; density and other rules govern whether
raw text or a summary is later injected—and dispatches idempotent extraction tasks for the
knowledge graph and for procedural patterns. Scheduled workers (decay, clustering, reflection,
batch summarisation, sentinel monitoring, and a weekly classifier fine-tune) maintain the stores
over time; table 22 in the appendix lists them.
5.2 Memory Stores
ICE maintains four stores plus two structured overlays, all in one PostgreSQL database with the
pgvector extension; every vector column is 384-dimensional because a single embedder is shared
throughout.
TheCodexis the store most specific to conversation. Its central design lever is a three-bucket
controlled relation vocabulary—propertyrelations overwrite,multi-valuedrelations accumulate,
andsingle-valuedrelations auto-expire the prior edge of the same (source, relation) pair—with
generic relations such as isandhasdeliberately absent so the extractor is forced to be specific.
Setting valid until rather than deleting is what makes the graph temporal: an edge with
valid until IS NULL is treated by the store as current (not independently verified true), and
a non-NULL value means historically superseded but still auditable. Appendix E.2 details the
write path and entity resolution.
Decayis access-weighted. Three decay workers apply multipliers corresponding to roughly
5%, 2%, and 1% decay per day for unaccessed, accessed, and creative-tagged turns respectively,
with acreative floorclamping decay score at 0.3 for narrative turns regardless of age—long-form
fiction should not be summarised away for being old. Turns below 0.1 are archived and below
6

Table 2:The four memory stores and the overlays. Each store is queried by one or more retrieval legs.
Appendix E gives schemas and write paths.
Store Purpose Retrieval method
Episodic Every conversational turn: raw text,
summary, lossless flag, decay score,
access count, embeddingBM25 leg; vector leg with
decay weighting; session
diversification
Codex Temporally-versioned semantic graph:
entities with aliases and properties,
typed edges carrying
valid from/valid until , append-only
event log, snapshotsGraph traversal (BFS depth
3 over edges where
valid until IS NULL);
enumeration fallback
Procedural Recurring behavioural patterns: trigger
conditions, reinforcement count,
confidenceVector top-5 behind a hard
intent gate
RAG Externally-ingested documents,
chunked and embeddedVector top-5, triple-gated
Memory slots Seven server-enforced persistent slots
(persona, preferences, project context,
guidance, pending items, . . . )Prepended verbatim to every
prompt
Context
clustersConversation-scoped topical groupings
with unit-norm centroidsRestrict episodic search to
relevant clusters
0.05 moved to cold storage. Codex edges decay similarly and are demoted from active to
pending below a strength threshold; procedural patterns deactivate after 180 days with fewer
than three reinforcements. Crucially, decay is not one-directional: every retrieval increments a
fragment’s access count and restores 0.15 to its decay score, soretrieval is a partial reversal
of forgettingand what the system used yesterday is cheaper to reach today.
5.3 Retrieval, Fusion, and Budget
The v2 design has six retrieval legs: lexical and decay-weighted vector episodic retrieval, graph
traversal, procedural lookup, document lookup, and batch summaries. “BM25” below is the
historical name of the PostgreSQL lexical leg. The two episodic legs are operational; procedural
lookup is defective, and document and batch-summary stores were empty in LSREP. Graph
traversal returns content, but its quality contribution is inconclusive. The full design must not
be read as six validated mechanisms.
For candidate f, weighted fusion computes s(f) =P
ℓαℓ/(60 + rank ℓ(f)), with intent- and
topic-dependent weights. Fragment text hashes identify duplicates. Post-fusion bonuses, session
diversification, and greedy packing determine the selected set. The diversity pass operates on
source type: both episodic legs share one type, so it does not guarantee one item from every
retrieval leg.
The nominal context allowance is 23,000 estimated tokens with a 1,800-token overhead reserve.
A growth cap limits retrieval to 2,000 estimated tokens for an empty current conversation,
regardless of how much history exists in other conversations. This makes the transition from
continuing-use LSREP to fresh-session LongMemEval architecturally consequential. Unused
allowance is not reallocated. Token estimates use ⌊1.33×word count⌋ ; they are not model-
tokenizer limits.
5.4 Local Operation and Audit Contract
The v2 deployment uses FastAPI, PostgreSQL/pgvector, local model serving, and Celery/Redis
workers. The cloud answerer and judge used in the later public diagnostic are evaluation
7

Algorithm 2: ICE v2 context selection
1.Classify the query with current-conversation context; apply retrieval overrides. Derive the
recent and retrieved-context budgets from that conversation’s turn count, density, and labels.
2.If the low-confidence fallback fires, use its wide-net path and 2,000-token ceiling. Otherwise
obtain scoped candidates from enabled legs and combine their rankings by weighted RRF.
3.Apply bonuses, cap each foreign conversation at three fragments, and deduplicate text. The
active conversation is uncapped by this session rule.
4.Admit the best candidate per source type when it fits; greedily fill the remaining retrieval
budget. Strengthen selected episodic memories.
5.Assemble system instructions, active slots, a bounded recent window, selected fragments,
boundary acknowledgement, and the live query.
Figure 3:Frozen ICE v2 selection contract. Candidate generation and selection occur before answer
generation; execution is not evidence of answer-quality benefit.
substitutions; they do not make that diagnostic an all-local run. The stores and control
interfaces make state inspectable, while lossless flags, graph validity intervals, retrieval traces,
and configuration pins provide audit surfaces.
A fidelity claim requires tracing input eligibility, execution, produced output, selection, and
final prompt inclusion. A quality attribution additionally requires a controlled contrast in which
the mechanism actually changes. A failed leg, an empty store, a constant ablation flag, and
an executed component with an uncertain quality effect are separate outcomes. The historical
harness did not enforce this full contract; RQ2 reports the retrospective audit rather than
retroactively claiming it did.
6 Evaluation Instantiation
6.1 Four LSREP Datasets
LSREP uses four long-form conversational datasets authored by one user during sustained real
work. A turn is one user prompt paired with its assistant response. Internal conversation names
and identifiers are replaced by labels A–D; the manuscript includes no verbatim private turn
or probe text. table 3 reports the corpus size, probe–checkpoint observation count, checkpoint
count, and simulated decay horizon used by the harness.
The four were selected before scoring for three properties: 251–1,119 turns, so relevant
material can be separated by hundreds of exchanges; complementary domains; and recurring
entities, revisions, procedures, and long-range dependencies. The replay normalises timestamps
and applies the simulated horizons in the final column; these are experimental ageing schedules,
not claims about the original conversations’ calendar duration. Aggregate reports and harness
code are released, while raw conversations and probe text remain private because they contain
the sole author’s personal data. LongMemEval supplies the complementary public corpus in
Section 7.3.
6.2 ICE v2 Replay and Answer Conditions
System under test.Experiment 2 evaluated frozen ICE v2: the classifier backbone upgraded
to Qwen3-Embedding-0.6B with the head retrained from scratch; a trained micro-NER model
replacing the pilot’s regex entity tagger, so the graph extractor grounds on real entity detection
(loaded checkpoint; no task-specific NER recall estimate is established here); the conservative
routing override added; the static budget and uncapped sliding window replaced by the unified
dynamic token budget of Section 5.3; and cluster-scoped retrieval introduced.
Protocol instantiation.Rather than many conversations at isolated checkpoints, Experi-
ment 2 follows four long-horizon conversations across their entire lifespan. Three choices differ
sharply from the pilot.(1) Continuous replay with preserved state: each conversation is replayed
8

Table 3:ICE v2 LSREP datasets and evaluation coverage. Observations repeat 219 distinct probes (A:
45; B: 91; C: 37; D: 46). Token counts use the harness’s documented estimate, 1 .33×whitespace-delimited
words, rather than a model-specific tokenizer. “Days” is the maximum simulated decay horizon, not
elapsed wall-clock time.
Label Description / principal
stressTurns Est. tokens Obs. Ckpts. Days
A Creative writing;
narrative continuity and
character evolution290 237K 173 10 24
B Long-form world-building;
horizon retrieval and
entity evolution1,119 806K 638 20 93
C Technical planning;
information density and
8K+ token turns325 785K 154 10 27
D Academic planning;
personal/technical
decisions251 155K 246 12 20
Total 1,985 1.98M 1,211 52—
chronologically into one fresh deployment, and memory stored at an early checkpoint remains
available at every later one; a simulated memory lifecycle—decay, edge decay, procedural decay,
reflection, clustering, consolidation, sentinel monitoring—runsbetweencheckpoints, so retrieval
reflects aged, consolidated memory rather than fresh storage.(2) Automatic, leg-aware probe
generation: probes are generated at each checkpoint from a sampled history window (first 5
turns, last 10 before the checkpoint, 15 random from between) using only information available
up to that checkpoint, explicitly targeting specific legs, and each requires a unique temporal
anchor to prevent ambiguity across hundreds of turns. These are combined with 72 hand-written
probes inherited from the pilot.(3) Incremental evaluation: every probe whose origin is at
or before a checkpoint is re-evaluated at that checkpoint, so a probe generated early is scored
repeatedly as history accumulates—which is what makes longitudinal memory growth directly
observable.
Evolving ground truth.The pilot’s single frozen reference is replaced by a three-stage
temporal-refinement pipeline:forensic regenerationof the origin-checkpoint answer under
a stricter, evidence-oriented prompt that records temporal provenance;temporal anchoring,
attaching to each probe an anchor identifying the fact or event it tracks; andforward propagation,
updating each answer at every later checkpoint when new content contradicts, updates, or
extends it, with anchor-preservation checks guarding against silent drift to a superficially similar
later event. Each probe therefore carries a temporally evolving reference document rather than
a static answer—necessary because otherwise a system that correctlyupdateda fact would score
worse than one repeating an outdated one.
Scoring.Experiment 2 adds temporal-aware scoring (an answer loses points for reporting a
superseded fact even if it was once correct) and inference credit (a specific, evidence-consistent
detail not explicitly in the reference is rewarded rather than flagged as fabrication).
Human scoring and label audit.The released aggregation code merges 72 manual
evaluation records, replacing their absolute scores and tournament rankings where supplied.
These records therefore contribute to the headline aggregates; the aggregate is not purely
automated. Separately, the reported v2 hallucination labels were reviewed against source
9

conversations to correct false positives. The hand-scored subset is not an independent population
or a second blinded annotator study. No corresponding manual correction was applied to the
ablation’s hallucination labels.
Judge.Scoring free-form answers with a language model follows now-standard practice (Liu
et al., 2023; Zheng et al., 2023), and inherits its known failure modes: judges are sensitive to
answer position and verbosity and are not fair evaluators by default (Wang et al., 2024a). We
mitigate rather than assume this away—anonymised four-condition tournaments, a fixed rubric
at temperature 0.0, a judge model disjoint from every answering model, and the hand-audited
slice described above; Section 8 states what remains uncontrolled. The LSREP generalist
answerer is gemma4:26b-a4b-it-q4 KM. Both LSREP experiments use the same independent
judge, gemma-4-12B-AWQ served on SGLang with a 150,000-token context at temperature 0.0
(requested deterministic decoding; execution-level reproducibility is not established), never used
for retrieval, generation, probe generation, or ground-truth construction. The extraction rules—
strict neutrality, verbatim anchor preservation, temporal dominance, third-person attribution—
are shared; Experiment 2 applies them through the evolving-dossier pipeline rather than a single
pass.
Conditions.A four-condition matrix:Vector RAG (generalist)—single-leg pgvector
similarity, top-30, no decay weighting, classification, or fusion, on a generalist model;Vector
RAG (MoE)—same retrieval, expert-routed;Full ICE (generalist)—all legs enabled, RRF
fusion, classification, decay weighting, dynamic budget, post-fusion curation; andFull ICE
(MoE). Both use the same embedder and database. The baseline is a standard single-leg
vector-RAG—top-30 similar turns assembled with the question—while the ICE conditions run
the full system, fused retrievalplusICE’s prompt assembler. The comparison is thereforefull
memory system versus standard vector-RAG, which is what a deployment decision actually looks
like: ICE’s curation deliberately governs what context is assembled, not merely which fragments
are retrieved.
A note on replay.Because history isreplayedrather than lived, the reinforcement that
normally strengthens frequently referenced memories during use is largely absent—the primary
reinforcement signal becomes the evaluation probes themselves. A concept referenced dozens
of times in real use and one mentioned once therefore look more alike under replay than they
would in deployment. This changes the state trajectory, but its direction is not guaranteed:
reinforcement can promote useful evidence or amplify an early retrieval error. Experiment 2
should therefore be read as performance under the specified replay process, not as a lower or
upper bound on deployment.
6.3 Metrics and Statistical Units
Scores use the historical 1–5 rubric. Win rate is the share of anonymised four-condition
tournaments ranked first, not a direct two-arm win probability. Prompt tokens include the recent
window and instruction scaffolding; fragment counts measure selected pieces, whose lengths differ.
SPF (mean score divided by mean fragments) and TUR (mean score per thousand estimated
prompt tokens) are descriptive ratios of an ordinal score, not standalone proofs of efficiency. We
report quality and context use jointly.
The historical v2 LSREP confidence intervals use 10,000 paired bootstrap resamples of
probe–checkpoint records. Repeated observations of a probe are dependent, and conversations
share one author; these intervals do not quantify generalisation across users or independent
trajectories. Missing scores follow the archived chain: available score; otherwise failed answer
→1; otherwise sibling routing-condition score; otherwise rounded within-record mean; otherwise
3. The reported near-zero difference is absence of a detected mean-score difference, not a formal
equivalence test.
LongMemEval uses binary correctness. Its intervals use 20,000 question-level paired percentile-
bootstrap resamples, seed 20260911. Every resample retains both arms; the phase-change contrast
10

retains all four arm–phase outcomes. Missing judgements are excluded from paired contrasts and
bounded over all questions. Category intervals are exploratory and unadjusted for multiplicity.
They condition on the recorded generations and judge verdicts; they do not include model rerun
or judge-calibration uncertainty.
6.4 Matched LongMemEval Setup
We use all 500 oracle questions and all 500 full-S questions (Wu et al., 2025). Oracle supplies ev-
idence sessions; full-S adds the benchmark’s remaining history. Adapter ice-v2-lme-sessions-
v2wipes the store per question, ingests each session as a separate auto-scoped conversation,
and asks from a fresh empty conversation. Both arms read the same stored vectors. The pure
vector-RAG arm retrieves the top 30 turn representations without decay weighting, fusion, or a
token budget; it is a strong high-context comparison, not a matched-budget arm.
Both arms use OpenCode Go gpt-5.6-luna through the Responses endpoint, with a 4,096-
output-token cap. The provider does not support the requested temperature parameter, so we
do not claim temperature-controlled generation. ICE memory construction uses the evaluated
Ollama qwen3:4b-instruct-bg . The judge is muse-spark-1.3-contributor , using the official
LongMemEval task-specific judgement prompts and yes/no rule, with a 4,096-token judge cap.
Muse is not the benchmark’s official GPT-4o judge: these are within-study comparisons, not
leaderboard-comparable scores. Judge identity, the decision prompt, and the answerer are
distinct parts of the experimental stack.
Run-enablement changes restore the tag’s background route to shared Ollama and run the
same 384-dimensional embedder on CUDA (recorded CPU–CUDA cosine agreement 0.99986).
The adapter calls v2 classification, budget setting, retrieval, and prompt assembly directly; it
does not replay the entire HTTP lifecycle. In particular, it omits the API wrapper’s faulty
secondary word check (Appendix E.4). The measurement is of this documented v2 adapter path.
No v3 repair is included.
The initial flattened-session adapter was invalidated after an identifier-type mismatch in-
correctly capped the active conversation at three fragments. The admitted adapter preserves
session boundaries and string identifiers, validates clean-store layout, and records its version.
A historical local-Gemma oracle preceded the matched cloud run; the historical decision to
stop before full-S was superseded when the matched study completed both phases. Appendix I
preserves that history without mixing its scores into RQ3.
7 Results
7.1 RQ1: Longitudinal Behaviour under LSREP
Table 4:ICE v2 results for all four datasets (generalist conditions). Scores are 1–5; tokens are estimated
complete-prompt counts; wins are first places in four-condition tournaments. Dataset C is the density
stress case.
Dataset ICE score Vec. score ICE tok. Vec. tok. ICE win % Vec. win %
A (Creative Writing) 3.63 3.50 19,434 24,36037.218.0
B (Long-Form Creative) 4.28 4.33 24,108 21,257 32.8 22.9
C (Technical Planning) 4.33 1.23 19,953 88,061 43.5 1.9
D (Academic Planning)4.634.57 20,103 18,076 20.4 18.8
table 5 reports the four conditions on the three-conversation view (Dataset C excluded;
1,057 probe–checkpoint observations). Per-conversation, score-distribution, and temporal-quality
breakdowns are in Appendix D.
The rounded mean scores are 4.26 and 4.25. The historical paired aggregation gives ∆ =
11

Table 5:Experiment 2, global comparison, Dataset C excluded (1,057 probe–checkpoint observations
across 3 conversations). On the conversations where both systems can generate usable answers, ICE v2
has similar mean scores to the vector baseline, injects 32% fewer fragments, and ranks first in 30.6% of
four-condition tournaments versus 21.2%.
Condition Score Tokens Frags SPF TUR Win % Hall. %
Vector RAG generalist 4.25±1.09 21,025 30.0 0.14 0.20 21.2 20.5
Vector RAG MoE 4.24±1.04 21,025 30.0 0.14 0.20 19.2 17.4
Full ICE generalist 4.26±1.08 22,411 20.4 0.21 0.1930.619.6
Full ICE MoE4.28±0.99 22,411 20.4 0.21 0.19 29.0 19.9
+0.002, reported as +0 .00 with a record-level 95% CI [ −0.07,+0.07]. The unrounded means are
4.25544 and 4.25355; rounding them separately makes their displayed difference 0.01. This is
no detected mean-score difference under the archived analysis; repeated-probe dependence and
manual-score merging limit its interpretation (Section 6.2).
Repeated-probe sensitivity.Resampling entire probe trajectories gives an ordinary-
density paired interval of [ −0.148,+0.158] around +0 .002 (182 probes), wider than the histor-
ical record-bootstrap interval. All-data and density-only clustered intervals remain positive:
[+0.192,+0.618] and [+2 .721,+3.470]. These retain the qualitative findings but do not estimate
variation across users (Appendix J).
Scoring and ordinal sensitivity.Removing the manual replacements gives an ordinary-
density mean difference of −0.020 (probe-cluster 95% CI [ −0.169,+0.136]), consistent with no
detected advantage. An ordinal comparison avoids assuming equal distances between rubric
levels: across the 1,057 ordinary-density paired observations, ICE v2 scores higher on 216,
vector on 215, and 626 tie. The net proportion favouring ICE is +0 .1 percentage points (cluster
CI [−7.6,+8.0]). In the archived merged scores, 130 vector-generalist failed answers receive
score 1 and two scores use the sibling routing condition; ICE-generalist has all 1,211 explicit
scores. Excluding every pair without two explicit scores reduces the all-data difference to +0 .046
([−0.101,+0.199], n= 1,079). This selected subset omits most density failures; it cannot replace
the reliability analysis. Appendix J.1 reports both views.
Context use.ICE v2 selects 20.4 rather than 30.0 fragments on average (32.0% fewer),
but uses 22,411 rather than 21,025 estimated prompt tokens (6.6% more). Its recent window,
persistent slots, and instruction structure are part of the full-system condition. The observed
benefit is fragment economy at similar mean scores, not a token saving. A matched-prompt or
matched-budget intervention was not run, so the costs and quality contributions of individual
assembly choices remain unidentified.
Tournament preference.ICE v2 generalist is ranked first in 30.6% of four-condition
tournament appearances, versus 21.2% for vector generalist. The historical record-bootstrap
intervals are [27 .9,33.4] and [18 .8,23.7]. This is not a direct head-to-head win rate, and shuffled
presentation does not eliminate judge bias. Audited hallucination rates are 19.6% and 20.5%,
respectively.
Fragment–score association.Recorded correlations are r= 0.193 for ICE v2 generalist
andr=−0.015 for vector generalist. The latter is near zero. These observational associations
do not establish that curation removes noise, that longer answers are better, or that additional
context causes higher quality.
Longitudinal behaviour.fig. 4 plots mean probe score against turn-index bin. All four
conditions track closely through early and middle bins—the tie is visible as four overlapping
curves. The separation appears at the deepest bin (turn 1,100 of Dataset B): both vector
conditions drop sharply, to 3.80 and 3.87, while both ICE conditions hold at 4.18 and 4.14. This
final-bin difference is descriptive; it does not identify retrieval reach as the cause, and the bin
represents one conversation.
12

0 100 200 300 400 500 600 700 800 900 1,000 1,10044.55
Turn-index binMean probe score
ICE generalist ICE MoE
Vector generalist Vector MoE
Figure 4:Longitudinal mean score by turn-index bin (1,057 probe–checkpoint observations, Dataset
C excluded; real per-bin data). The four conditions track closely mid-conversation, but at the final
turn-1,100 bin—the deep long-horizon regime of Dataset B—both vector conditions drop sharply while
both ICE conditions hold.
Context cost changes with position.The 6.6% figure above is an average over conditions
that differ sharply, and two mechanisms push against each other. Per-turndensityfavours ICE:
the baseline retrieves a fixed top-30, so its injected tokens scale with how large each turn is,
without bound. Conversationlengthfavours the baseline: ICE’s growth cap deliberately widens
the budget as turns accumulate (2 ,000 + 150 nbelow 30 turns, then 5 ,000 + 100( n−30), then
10,000 + 30( n−100)), while top-30 is indifferent to how long a conversation has run. table 6
contrasts the first and last quartile of each conversation, paired per probe and bootstrapped.
These are observational position contrasts: query mix, evidence density, and state also vary with
checkpoint.
Dataset D changes from fewer ICE v2 prompt tokens in the first quartile to more in the last.
Dataset B’s positive cost difference shrinks, so the table does not support a uniform increase in
relative cost across datasets. Density dominates Dataset C. The accompanying score changes do
not identify the budget’s marginal return: query difficulty, state, and context all vary together.
A quality–token frontier would require independently varied budgets.
Table 6:Token cost by position within each conversation: first quartile of probes versus last. ∆ is the
paired per-probe difference (ICE minus baseline; negative means ICE is cheaper) with a 95% bootstrap
interval, and “cheaper” is the share of probes on which ICE injected fewer tokens. Historical record-level
intervals; query mix and state vary with position. These are not budget interventions.
Dataset Position Paired∆95% CI ICE cheaper
A (Creative, 290 turns)first quartile−8,034 [−8,650,−7,451] 100%
last quartile−2,284 [−2,766,−1,800] 91%
B (Long-form, 1,119 turns)first quartile +3,766 [+3,133,+4,394] 18%
last quartile +1,288 [+476,+2,075] 35%
C (Technical, 325 turns)first quartile−70,731 [−79,067,−63,734] 100%
last quartile−68,854 [−88,105,−53,696] 100%
D (Academic, 251 turns)first quartile−2,309 [−3,112,−1,521] 69%
last quartile +4,994 [+4,362,+5,609] 5%
Routing.The ordinary-density MoE-versus-generalist score delta is +0 .03 for ICE and −0.02
for vector; the effect is too small to claim in either direction. MoE gives a 3.1% hallucination
reduction under vector and a 0.3% increase under ICE, and no token savings. The routing
condition has a small observed mean difference; we return to its scope in Section 7.2.2.
13

Memory maturity and reliability.Simulated decay days at the final checkpoint were
93 for Dataset B, 24 for A, 27 for C, 20 for D. Gating failures—probes where the classifier
blocked retrieval that was needed—fell from 22 in the pilot to 2 here, the single largest reliability
improvement in the mature deployment.
Hand-scored subset.The 72 manual records favour ICE v2 by +1 .21 score points (historical
record-bootstrap CI [+0 .81,+1.61]). They are included in the reported aggregate and come from
one author/scorer. Selection and scorer effects prevent treating this subset as proof that the
automated judge underestimates ICE’s population advantage.
Table 7:Human-verified slice (72 hand-scored probes, scored by the corpus author). Paired ICE-
versus-vector is +1 .21 (95% CI [+0 .81,+1.61]), alongside the near-zero aggregate difference over 1,057
probe–checkpoint observations. These manual records also enter the reported aggregate. Single scorer;
blind per-probe tournament ranking.
Condition Score 95% CI Win %
Vector RAG (generalist) 3.10 [2.75,3.44] 5.6
Vector RAG (MoE) 3.22 [2.88,3.54] 8.3
Full ICE (generalist) 4.31 [4.11,4.49] 16.7
Full ICE (MoE) 4.24 [4.01,4.43]69.4
7.1.1 Density Stress and the All-Dataset View
Dataset C contains architecture documents that frequently exceed 8,000 tokens in a single
turn. The vector baseline, which always retrieves the top-30 fragments with no per-query
budget, injects a mean of 88,061 tokens—failed cases reach 100,505—overwhelming the model’s
context window. Of 154 probe–checkpoint observations, 145 receive score 1 (94.2%); score 1 is a
poor-answer/failure category, not a direct telemetry count of context-overflow exceptions. The
retrieval subsystem did find relevant information; the generation model could not process the
assembled context and produced empty or degenerate responses. ICE, with its dynamic budget
capping retrieval, holds a mean of 19,953 tokens and a mean score of 4.33.
Table 8:ICE v2 density stress test on Dataset C (154 probe–checkpoint observations). The vector
baseline collapses because it has no token-budget cap. ICE survives with a mean score of 4.33 and 77%
fewer tokens injected. The baseline’s 0% hallucination rate is an artefact: 94.2% of its responses are
complete failures, so there is no answer to evaluate.
Condition Score Tok. Frags SPF TUR Hall % Win % Score 1 %
Vector RAG generalist 1.23 88,061 30 0.04 0.01 0.0* 1.994.2
Vector RAG MoE 1.10 88,061 30 0.04 0.01 0.0* 4.5 97.4
Full ICE generalist4.3319,953 100.43 0.2241.4 43.5 3.9
Full ICE MoE 4.03 19,953 10 0.40 0.20 40.850.016.2
The score distributions are far apart. ICE generalist: 3.9% at score 1, 5.2% at 2, 11.7% at 3,
12.3% at 4, 66.9% at 5. The baseline: 94.2% at score 1, nothing at 2–4, 5.8% at 5.
ICE v2’s 41.4% hallucination rate remains a substantial limitation. The baseline’s near-
absence of usable answers makes its zero hallucination rate uninformative about successful-answer
faithfulness; that does not excuse ICE errors.
Information density as a benchmark dimension.Turn count alone is an insufficient
measure of difficulty. Dataset B is the greatest challenge intemporal horizon; Dataset C is a
different challenge entirely,extreme information density. The benchmark surfaced two distinct
stress dimensions, and the second was not designed—it emerged when the first few probes
returned errors.
14

What the headline numbers look like with Dataset C included.table 5 deliberately
excludes Dataset C, so the reported tie reflects conditions under which both systems can produce
an answer; a 94.2% failure rate would otherwise dominate any aggregate without saying anything
about answerquality. So that this choice is not mistaken for cherry-picking, table 9 reports the
same four conditions overall1,211 probe–checkpoint observations.
Table 9:ICE v2 over all four conversations, including the density stress case (1,211 observations). The
aggregate combines answer quality with the unbudgeted baseline´ s high failure rate on Dataset C.
Condition Score Tokens Frags SPF Win % Hall. %
Vector RAG generalist 3.87±1.47 29,550 30.0 0.13 18.7 20.2
Vector RAG MoE 3.84±1.45 29,550 30.0 0.13 17.3 15.7
Full ICE generalist4.27±1.09 22,099 19.10.22 32.322.3
Full ICE MoE 4.25±1.03 22,099 19.1 0.22 31.7 22.5
The historical paired record-bootstrap differences are +0 .40 (95% CI [+0 .31,+0.49]) over all
data and +3 .10 ([+2 .85,+3.33]) on Dataset C. Probe-clustered intervals remain positive but
wider (Appendix J). Both the all-data view and the ordinary-density view are needed: the first
includes deployment failures, while the second shows that those failures account for most of the
aggregate quality gap.
7.2 RQ2: Mechanism Fidelity
7.2.1 Cumulative Ablation
To understand which mechanisms matter, we ran a cumulative feature buildup on Dataset B (67
probes, fully mature memory state, single pass), constructing the system incrementally from a
bare-vector baseline with each step adding exactly one feature and preserving all previous ones.
The answering model here is Qwen3-14B-AWQ rather than Experiment 2’s 26B generalist; the
independent Gemma-4 12B judge is unchanged. Absolute scores are therefore not comparable
across experiments, and relative within-probe deltas are the object of interest. table 10 gives the
CI-robust steps; the full fifteen-row table, the recency breakdown, and the waterfall chart are in
Appendix C.
Table 10:ICE v2 ablation steps that clear a paired within-probe bootstrap 95% CI (10,000 resamples,
67 probes scored under every condition), with the endpoints for context. Every other step’s interval spans
zero; the full table is table 17. The single-leg vector baseline is a reference point, not a buildup step.
Step Score Paired∆95% CI
bare vector 3.27 — —
+BM25 2.52−0.74 [−1.14,−0.36] excludes 0
+RRF 3.36 +0.82 [+0.39,+1.24] excludes 0
nine further steps, all spanning zero (Appendix C)
fullice 3.38−0.03 [−0.29,+0.23]
vector baseline (ref.) 3.42 — —
Adding an unfused lexical leg is harmful( −0.74, [−1.14,−0.36]). The observed damage
is consistent with additional unfused matches admitting distracting context. This buildup shows
that adding a lexical leg without fusion can worsen quality; it does not isolate the content
responsible for each error.
Rank fusion is corrective(+0 .82, [+0 .39,+1.24]). Adding RRF on top recovers the damage.
Its role is precisely that: recovery. Neither RRF-on-vector nor the fully built configuration is
statistically distinguishable from the single-leg baseline on answer quality—both contrasts span
15

zero, consistent with the Experiment 2 tie. RRF corrects the damage from adding the lexical
leg in this buildup; no general safety guarantee follows. We state this deliberately because the
earlier draft of this work claimed RRF was the single most impactful component; the paired
intervals do not support that, and the corrected claim is the smaller one.
The remaining steps must be read with care, and Section 7.2.2 explains why: several
toggled features were not functioning or not varying at the snapshot, so their near-zero deltas
measure nothing about the mechanism. The small observed contrasts—which do not establish
equivalence—are cluster restriction and session diversification (∆ ≤0.01). One inconclusive
contrast concerns the budget: the dynamic budget’s step is −0.11, alongside a fill-to-cap policy
that allocates more tokens to longer conversations and selects more fragments on Dataset B
(21.2 fragments against 11.3 at the pre-budget step). The paired budget-step estimate is −0.08
with CI [ −0.36,+0.18]; it is inconclusive, and does not establish that an alternative budget
policy would improve quality.
7.2.2 Execution, Reach, and Identifiable Effects
The frozen v2 audit finds episodic fragments accounting for 62.3% of recorded selections, Codex
for 3.3%, and unattributable source records for 34.5%. The graph concatenates a traversal into
one fragment, so fragment share cannot be equated with token share, fact coverage, or utility.
The trained NER model was loaded, but that fact alone establishes neither task-specific recall
nor graph correctness.
Procedural retrieval isdefective: its untyped embedding bind raises a pgvector operator
error and returns an empty list. The document leg shares that defect and additionally had no
ingested documents. Batch summaries had no eligible stored content; archival retrieval was not
exercised. HyDE was disabled in the mature run and constant across the ablation arms rather
than controlled by its nominal flag. These are not neutral quality results.
Cluster restriction and model routing executed with small observed differences; they support
“no detected effect in this setting”, not equivalence. Session diversification did not exercise its
central cross-conversation cap in the within-conversation replay. Graph and enumeration-fallback
contrasts are inconclusive. Current-truth probes can reward correct updates but do not validate
as-of graph querying or historical composition. RQ3 supplies the separate cross-session test.
The fusion contrast supports a local causal attribution within this buildup: an unfused
lexical leg hurts, and RRF recovers that loss. The density stress case supports the bounded
full-systemconfiguration against the unbudgeted baseline; it does not isolate the budget from
assembly and retrieval. Decay and reinforcement ran but were not independently switched off.
The observed fragment reduction belongs to the complete selection-and-assembly condition.
Appendix A records the component evidence and its limits.
7.3 RQ3: External Transfer under LongMemEval
ICE v2 loses decisively overall in both phases.Table 11 places pure vector-RAG
beside ICE v2 under the same cloud answerer and judge. The evidence-only deficit is already
22.0 percentage points; distractors therefore cannot be its root cause. LSREP’s continuing-
conversation observations did not transfer to strong endpoint QA across separately supplied
sessions.
Oracle’s paired difference is −22.0 points, CI [ −26.6,−17.4] (n= 500). Full-S’s paired
difference is −26.5, CI [ −31.3,−21.8] (n= 499). One vector judgement is missing: all-500
bounds are 69.4–69.6% for vector and exactly 43.0% for ICE v2. Its assignment cannot change
the ordering. The full-S paired ICE rate is 215 /499; the displayed ICE marginal is 215 /500.
Comparisons of rounded marginal percentages and paired estimates consequently differ slightly.
16

Table 11:Matched LongMemEval accuracy (percent; correct/obtainable verdicts). Both arms use
gpt-5.6-luna and Muse Spark 1.3 Contributor judging; these are not official-judge leaderboard scores.
Oracle Full-S
Type ICE v2 Vector-RAG ICE v2 Vector-RAG
Abstention 83.3 (25/30) 60.0 (18/30) 83.3 (25/30) 63.3 (19/30)
Knowledge update 58.3 (42/72) 73.6 (53/72) 51.4 (37/72) 69.4 (50/72)
Multi-session 29.8 (36/121) 86.0 (104/121) 21.5 (26/121) 74.4 (90/121)
Session assistant 91.1 (51/56) 94.6 (53/56) 75.0 (42/56) 94.6 (53/56)
Session preference 70.0 (21/30) 70.0 (21/30) 43.3 (13/30) 51.7 (15/29)
Session user 78.1 (50/64) 95.3 (61/64) 71.9 (46/64) 93.8 (60/64)
Temporal reasoning 22.8 (29/127) 42.5 (54/127) 20.5 (26/127) 47.2 (60/127)
Overall 50.8 (254/500) 72.8 (364/500) 43.0 (215/500) 69.5 (347/499)
Category structure.Full-S multi-session accuracy is 21.5% versus 74.4%, and temporal
accuracy is 20.5% versus 47.2%. Their paired differences are −52.9 points, CI [ −62.8,−43.0]
(n= 121), and −26.8, CI [ −35.4,−18.1] (n= 127). These expose severe synthesis and temporal-
composition failures. They do not identify whether a particular answer failed during construction,
retrieval, or reasoning.
Abstention is a descriptive strength: full-S ICE v2 is correct on 25/30 questions versus 19/30
for vector. The paired table contains 18 both-correct, 7 ICE-only, 1 vector-only, and 4 both-wrong
outcomes. The +20.0-point percentile-bootstrap interval is [+3 .3,+36.7], but there are only
eight discordant pairs; a two-sided exact McNemar test gives p= 0.0703. Oracle abstention
is +23.3 points, CI [+6 .7,+40.0], with 8 versus 1 discordant pairs. These small exploratory
subsets, unadjusted for multiple comparisons and conditional on Muse labels, support cautious
reporting of conservative abstention rather than a general superiority claim. Appendix G gives
every paired category count and interval.
Phase transitions.ICE v2 changes from correct to wrong on 68 questions and wrong to
correct on 29, losing 7.8 points over 500. Vector changes 44 and 28 times respectively, losing
3.2 points on its 499 complete phase pairs (the marginal table drops approximately 3.3 points).
On the common 499 four-way-complete questions, ICE loses 7.6 points and vector 3.2: the
difference in degradationis 4.4 points, CI [ −0.2,+9.2]. Its point estimate widens the deficit, but
uncertainty includes no extra degradation. Oracle is not a mathematical upper bound: both
arms improve on some individual questions after other histories are added.
Quality and context cost.Table 12 distinguishes observed input volume from configuration.
In full-S, ICE v2 supplies a median of 5 fragments and 2,222 provider input tokens; vector
supplies 30 and 11,718. Vector achieves much higher accuracy with substantially more context.
These data establish context economy for ICE and a quality–cost trade-off, not an efficiency
winner. They do not determine accuracy at matched budgets or whether either system dominates
an accuracy–token frontier.
Both phases have 500 non-empty answers per arm and no final answer failure; this does not
measure transient retry frequency. Judgement missingness is 0/500 per arm in oracle and 0/500
ICE versus 1/500 vector in full-S. ICE’s configured retrieval ceiling is 2,000 estimated tokens;
vector’s top-30 limit is active and hasnotoken ceiling. A retrieval-budget field recorded in
vector artifacts belongs to an unused orchestrator object and must not be presented as a vector
cap.
The recorded word-based estimate includes system instructions and the question, not only
retrieved text. Provider input-token metadata is complete for all 2,000 answers. Per-leg candidate
counts, retrieval-only token counts, retrieval latency, and arm-specific memory-construction
17

Table 12:Recorded costs, median [25th, 75th percentile], 500 answers per row. Fragments are post-
selection and supplied to assembly; provider input tokens cover the complete request. Seconds measure
answer generation only.
Phase Arm Fragments Input tokens Generation (s)
Oracle ICE v2 3.0 [3.0, 4.0] 2,006 [1,758, 2,118] 2.8 [2.2, 3.4]
Oracle Vector 11.5 [6.0, 12.0] 5,529 [3,370, 6,732] 2.0 [1.6, 2.8]
Full-S ICE v2 5.0 [4.0, 6.0] 2,222 [2,168, 2,289] 3.1 [2.5, 3.8]
Full-S Vector 30.0 [30.0, 30.0] 11,718 [10,437, 12,923] 2.6 [2.1, 3.5]
cost were not persisted at the required granularity. They cannot be recovered exactly from
summary fields; the paper reports them as unavailable instead of subtracting an assumed
overhead. Selected fragment counts equal the fragments joined by the adapter’s assembler, but
are not pre-selection candidate counts.
Outcome stratification does not support a simple “more context fixes the error” reading.
Full-S ICE v2 correct and incorrect answers have nearly identical median inputs (2,224 versus
2,219 tokens); vector’s are 11,650 versus 11,855. For full-S multi-session questions, vector answers
90/121 correctly while supplying 30 fragments per answer, yet still fails on 31; ICE answers
26/121. Cost by category and correctness, including dispersion, is preserved in Appendix H and
the aggregate artifact. No causal claim follows from these observational strata.
Published-system context.Table 13 is external context only. Its rows use different models,
prompts, and budgets; no row is ranked against ICE v2. A head-to-head claim would require
rerunning the other systems under this stack or rejudging ICE/vector outputs with the official
judge.
Table 13:Published LongMemEval-S configurations: contextual, not matched to this study. NR means
not reported in the cited version; no ICE ranking is implied.
System Answerer Judge Score Prompt / retrieval / protocol
Zep GPT-4o-
miniGPT-4o 63.8% Task-specific official judging; Zep
retrieval; mean context 1.6K
tokens (not a declared cap);
supplied S histories (Rasmussen
et al., 2025)
Zep GPT-4o GPT-4o 71.2% Same reported protocol;
system-specific answer prompt;
mean context 1.6K
tokens (Rasmussen et al., 2025)
Hindsight GPT-OSS-
20BGPT-OSS-
120B83.6% Retain/recall/reflect;
paper-specific judge templates;
retrieval budget NR in v1; S, 500
questions (Latimer et al., 2025)
Hindsight Gemini-3
ProGPT-OSS-
120B91.4% OSS-120B memory construction;
Gemini answer generation; same
v1 budget disclosure
gap (Latimer et al., 2025)
18

8 Limitations and Implications
8.1 Private Data, Repeated Probes, and Reference Construction
Four datasets provide domain and density variation, but all come from one author. The 1,211
observations are repetitions of 219 questions, not independent examples from 1,211 users or tasks.
Probe-clustered sensitivity analysis widens uncertainty, while still conditioning on the same four
conversations. Neither analysis supports population-level generalisation. The manually scored
subset shares the corpus author, and references were refined through a model-assisted process
that may omit relevant details or preserve authoring errors.
LSREP’s implementation and specification can be reused with other histories; its exact
private-data scores cannot be independently reproduced without those data. The synthetic
example demonstrates reference semantics, not empirical validity. A public corpus with auditable
revision ledgers and independent annotators would test portability more directly.
8.2 Replay and Mechanism Boundaries
Replay cannot model how a generated answer changes a user’s next turn, because the future
transcript is fixed. Evaluation accesses reinforce ICE v2’s memory, so the probe schedule itself
affects subsequent state. The historical four-condition harness shares stored memory rather
than maintaining independent long-lived states for every arm; results concern that declared
shared-state comparison. A separate-arm replay would be a different experiment. Simulated
horizons are ageing schedules, not elapsed deployment durations.
Graph output presence does not validate graph truth or historical reasoning. Procedural
retrieval was defective, document retrieval unexercised and defective, and several lifecycle paths
produced no measurable result. The quality effect of decay, prompt scaffolding, and individual
curation steps is not independently identified. The density case compares the full system with
an unbudgeted top-30 baseline, so it does not show an advantage over a properly budgeted
vector baseline. Session placement, model stack, corpus, and grading differ between LSREP and
LongMemEval; their scores cannot be subtracted to estimate a causal session effect.
8.3 Judging and Cost Boundaries
The v2 LSREP aggregation includes manual scores and imputation. Its hallucination audit and
the uncorrected ablation labels have different status; neither absolute rates nor relative arm
differences are guaranteed free of judge bias. The matched LongMemEval study uses a non-official
Muse judge. Muse passed a three-case discrimination check; its prior 15-item human-labelled
calibration had 73% agreement in a different setting. Neither is a broad benchmark-specific
human-agreement study, and bootstrap intervals do not include judge misclassification or run-
to-run writer and answerer variation. The answerer was selected using a small sample from the
same public benchmark, which further limits confirmatory interpretation.
Lower token use with lower accuracy is not superior efficiency. A matched-budget sweep, cost
at a declared quality threshold, or an accuracy–token frontier is needed to establish a stronger
claim. Generation latency excludes retrieval, ingestion, and retries; provider token counts are
not dollar costs. The selected cloud stack is also distinct from ICE v2’s local-first deployment
configuration.
8.4 Implications for the Next Evaluation
The immediate methodological requirement is to evaluate both evolving state and public endpoint
performance, accompanied by an execution audit. An observed benchmark score belongs to the
whole evaluation stack: memory system, answerer, judge, prompt, context budget, and dataset.
Future work should rerun fixed memory systems with newer answerers and cross answerer and
judge families to test ranking stability. This study does not establish family-related judge bias.
19

The pure vector baseline’s 69.5% full-S score numerically overlaps some contextual published
scores, but their different stacks preclude a system ranking. The relevant test of architectural
complexity is its incremental benefit over a strong simple baseline under matched conditions,
including a budget sweep: overall quality, specific capabilities, and cost at a declared quality
level. These are requirements for future comparisons, not results of the present experiments.
Future experiments should vary prompt scaffolding and budgets independently, use separate
state trajectories where conditions mutate memory, and test historical references as well as
current truth. Graph quality and cross-session synthesis require explicit evidence-level controls.
These are unmeasured research questions. ICE v3 development proceeds independently; no v3
behaviour or repair is used to explain a v2 result here.
9 Conclusion
LSREP specifies how to evaluate reconstructed conversational memory over time: declare state
transitions and access effects, repeat probes against evolving references, and audit whether the
mechanisms named in a result actually ran. ICE v2 provides a substantial architectural case
study with typed stores, local operation, retrieval fusion, and bounded prompt construction,
together with clear limits on which mechanisms were effective.
The v2 results are regime-dependent. Continuing-use replay shows similar ordinary-density
mean scores with fewer fragments, and robustness relative to an unbudgeted baseline under
extreme density. Matched LongMemEval shows decisive overall losses, conservative abstention,
and severe multi-session and temporal failures. ICE v2 uses less public-benchmark context
while giving less accurate answers. The disagreement demonstrates why longitudinal evaluation,
mechanism fidelity, and public end-task testing are jointly necessary for a credible memory-system
claim.
Reproducibility and Ethics
The canonical manuscript preserves the full archival account. The evaluated system is pinned to
v2-paper-eval ; the public diagnostic additionally identifies the session adapter and documented
run-enablement changes. Aggregate reports, analysis scripts, configuration, and adapter/harness
code form the public artifact package. Raw answers, judgements, databases, logs, downloaded
benchmark data, credentials, private planning notes, and personal corpora are excluded. Public
LongMemEval inputs must be obtained from their original release. Exact private LSREP results
are not independently reproducible from the released package.
The private corpora are the sole author’s conversational records; no private transcript excerpts
are reproduced here. The worked example is synthetic. ICE v2’s local storage and explicit
memory controls support user inspection, but persistence also increases the consequences of
incorrect or sensitive stored information. Local operation is a deployment property, not a proof
of privacy or correctness. The matched public diagnostic uses remote generation and judging on
public benchmark material. No new personal-data release is needed for the aggregate analyses
reported here.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG:
Learning to retrieve, generate, and critique through self-reflection. InProceedings of the
12th International Conference on Learning Representations (ICLR), 2024. URL https:
//arxiv.org/abs/2310.11511.
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0:
20

Building production-ready AI agents with scalable long-term memory.arXiv preprint
arXiv:2504.19413, abs/2504.19413, 2025. URLhttps://arxiv.org/abs/2504.19413.
Gordon V. Cormack, Charles L. A. Clarke, and Stefan B¨ uttcher. Reciprocal rank fusion
outperforms Condorcet and individual rank learning methods. InProceedings of the 32nd
International ACM SIGIR Conference on Research and Development in Information Retrieval
(SIGIR ’09), pages 758–759, New York, NY, USA, 2009. Association for Computing Machinery.
doi: 10.1145/1571941.1572114.
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven
Truitt, and Jonathan Larson. From local to global: A Graph RAG approach to query-focused
summarization. Technical report, Microsoft Research, 2024. URL https://arxiv.org/abs/
2404.16130.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. Precise zero-shot dense retrieval
without relevance labels. InProceedings of the 61st Annual Meeting of the Association for
Computational Linguistics (ACL), pages 1762–1777, Toronto, Canada, 2023. Association for
Computational Linguistics. URLhttps://aclanthology.org/2023.acl-long.99/.
Bernal Jim´ enez Guti´ errez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. HippoRAG:
Neurobiologically inspired long-term memory for large language models. InAdvances in Neural
Information Processing Systems 37 (NeurIPS 2024), 2024. URL https://arxiv.org/abs/
2405.14831.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. REALM:
Retrieval-augmented language model pre-training. InProceedings of the 37th International
Conference on Machine Learning (ICML), volume 119 ofPMLR, pages 3929–3938, Virtual,
2020. PMLR. URLhttps://proceedings.mlr.press/v119/guu20a.html.
Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia,
Yang Zhang, and Boris Ginsburg. RULER: What’s the real context size of your long-
context language models?arXiv preprint arXiv:2404.06654, abs/2404.06654, 2024. URL
https://arxiv.org/abs/2404.06654.
Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for
open domain question answering. InProceedings of the 16th Conference of the European
Chapter of the Association for Computational Linguistics (EACL), pages 874–880, Online,
2021. Association for Computational Linguistics. URL https://aclanthology.org/2021.
eacl-main.74/.
Jihyoung Jang, Minseong Boo, and Hyounghun Kim. Conversation chronicles: Towards diverse
temporal and relational dynamics in multi-session conversations. InProceedings of the 2023
Conference on Empirical Methods in Natural Language Processing (EMNLP), Singapore, 2023.
Association for Computational Linguistics. URLhttps://arxiv.org/abs/2310.13420.
Chris Latimer, Nicol´ o Boschi, Andrew Neeser, Chris Bartholomew, Gaurav Srivastava, Xuan
Wang, and Naren Ramakrishnan. Hindsight is 20/20: Building agent memory that retains,
recalls, and reflects.arXiv preprint arXiv:2512.12818v1, 2025. URL https://arxiv.org/
abs/2512.12818v1.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich K¨ uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨ aschel, Sebastian Riedel, and
Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. InAdvances
in Neural Information Processing Systems 33 (NeurIPS 2020), pages 9459–9474, Red Hook,
NY, USA, 2020. Curran Associates Inc. URL https://proceedings.neurips.cc/paper/
2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html.
21

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni,
and Percy Liang. Lost in the middle: How language models use long contexts.Transactions
of the Association for Computational Linguistics, 12, 2024. URL https://arxiv.org/abs/
2307.03172.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-Eval:
NLG evaluation using GPT-4 with better human alignment. InProceedings of the 2023
Conference on Empirical Methods in Natural Language Processing (EMNLP), Singapore, 2023.
Association for Computational Linguistics. URLhttps://arxiv.org/abs/2303.16634.
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei
Fang. Evaluating very long-term conversational memory of LLM agents. InProceedings of the
62nd Annual Meeting of the Association for Computational Linguistics (ACL). Association for
Computational Linguistics, 2024. URLhttps://arxiv.org/abs/2402.17753.
National Institute of Standards and Technology. Sign test. Dataplot Reference Manual,
n.d. URL https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/
signtest.htm. Accessed September 2026.
Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and
Joseph E. Gonzalez. MemGPT: Towards LLMs as operating systems.arXiv preprint
arXiv:2310.08560, abs/2310.08560, 2023. URLhttps://arxiv.org/abs/2310.08560.
Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang,
and Michael S. Bernstein. Generative agents: Interactive simulacra of human behavior. In
Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology
(UIST), San Francisco, CA, USA, 2023. Association for Computing Machinery. URLhttps:
//arxiv.org/abs/2304.03442.
Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. Zep: A
temporal knowledge graph architecture for agent memory.arXiv preprint arXiv:2501.13956,
abs/2501.13956, 2025. URLhttps://arxiv.org/abs/2501.13956.
Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and
beyond.Foundations and Trends in Information Retrieval, 3(4):333–389, 2009. doi: 10.1561/
1500000019.
Alireza Salemi, Sheshera Mysore, Michael Bendersky, and Hamed Zamani. LaMP: When large
language models meet personalization. InProceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (ACL). Association for Computational Linguistics,
2024. URLhttps://arxiv.org/abs/2304.11406.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke
Zettlemoyer, and Wen-tau Yih. REPLUG: Retrieval-augmented black-box language models.
InProceedings of the 2024 Conference of the North American Chapter of the Association for
Computational Linguistics (NAACL), pages 8371–8384, Mexico City, Mexico, 2024. Association
for Computational Linguistics. URLhttps://aclanthology.org/2024.naacl-long.463/.
Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel M.
Ni, Heung-Yeung Shum, and Jian Guo. Think-on-graph: Deep and responsible reasoning
of large language model on knowledge graph. InProceedings of the 12th International
Conference on Learning Representations (ICLR), Vienna, Austria, 2024. OpenReview.net.
URLhttps://openreview.net/forum?id=nnVO1PvbTv.
22

Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu,
Tianyu Liu, and Zhifang Sui. Large language models are not fair evaluators. InProceedings of
the 62nd Annual Meeting of the Association for Computational Linguistics (ACL). Association
for Computational Linguistics, 2024a. URLhttps://arxiv.org/abs/2305.17926.
Yu Wang, Nedim Lipka, Ryan A. Rossi, Alexa Siu, Rui Zhang, and Tyler Derr. Knowledge graph
prompting for multi-document question answering. InProceedings of the AAAI Conference on
Artificial Intelligence, volume 38, pages 19206–19214, Washington, DC, USA, 2024b. AAAI
Press. URLhttps://arxiv.org/abs/2308.11730.
Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. LongMemEval:
Benchmarking chat assistants on long-term interactive memory. InProceedings of the 13th
International Conference on Learning Representations (ICLR), 2025. URL https://arxiv.
org/abs/2410.10813.
Jing Xu, Arthur Szlam, and Jason Weston. Beyond goldfish memory: Long-term open-domain
conversation. InProceedings of the 60th Annual Meeting of the Association for Computational
Linguistics (ACL), Dublin, Ireland, 2022. Association for Computational Linguistics. URL
https://arxiv.org/abs/2107.07567.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective retrieval augmented
generation.arXiv preprint arXiv:2401.15884, abs/2401.15884, 2024. URL https://arxiv.
org/abs/2401.15884.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion
Stoica. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. InAdvances in Neural
Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track, 2023.
URLhttps://arxiv.org/abs/2306.05685.
Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. MemoryBank: Enhancing
large language models with long-term memory. InProceedings of the AAAI Conference on
Artificial Intelligence, volume 38, pages 19724–19731, Washington, DC, USA, 2024. AAAI
Press. URLhttps://ojs.aaai.org/index.php/AAAI/article/view/29946.
A Component Fidelity Audit
table 14 is the component-by-component audit summarised in Section 7.2.2, grounded in the
code at tag v2-paper-eval and in the frozen Experiment 2 result JSON. It is reproduced in full
because the three-way classification—defective, never exercised, contributing—is only checkable
if the per-component evidence is visible.
The 34.5% unattributable source records are an instrumentation limitation. Graph fragment
granularity and a loaded NER checkpoint do not establish graph correctness or content share.
23

Table 14:Component state at the evaluation snapshot. “Contributing” means the component executed
and affected retrieved output; “never exercised” means the benchmark supplied no input that would
engage it; “defective” means it executed and failed.
Component State Evidence Claimable as
Vector leg
(decay-weighted)Contributing Typed vector bind present;
dominates leg contributionsValidated
BM25 leg Contributing tsvector leg, no embedding bind
needed; with vector, 62.3% of
fragmentsValidated
RRF fusion Contributing +0.82 [+0.39,+1.24] over a lexical
and a dense legValidated, scoped to
two legs
Dynamic token
budgetContributing Stress: unbudgeted baseline 94.2%
score-1; full ICE mean 4.33Full-system contrast;
budget not isolated
Post-fusion
curationContributing 32% fewer fragments; near-zero
mean-score differenceObserved aggregate
reduction
Decay +
reinforcementContributing Simulated decay days per
conversation (93/24/27/20);
access-count and score restoration
on every hitRunning; size not
isolated
Cluster scoping Contributing Correct bind; ablation≈+0.01 Small observed
effect
Session diversify,
dedup, bonusesContributing Post-fusion transforms; keyword
boost +0.12Contributing
Codex graph Contributing,
under-
weighted3.3% of fragments, one concatenated
traversal fragment; trained NER
checkpoint loadedObserved output;
utility unconfirmed
Classifier + gate Contributing Gating failures 22→2 Validated, with the
override caveat
MoE routing Small
observed
effectRan on every probe; ∆ + 0.03 /
−0.02 (ordinary density)No detected effect
Procedural legDefectiveUntyped array bind⇒vector <=>
double precision[]⇒rollback⇒
[]; 0.0 fragmentsBug; no claim
Document (RAG)
legNever
exercisedNo documents ingested; store empty;
leg also shared the bind defectNo claim either way
Batch summaries Never
exercisedBind correct; nothing decayed far
enough to batch; store emptyNo claim either way
Cold storage Never
exercisedReplay horizon never reached the
archival regimeNo claim either way
Temporal /
timeline retrievalNever
exercisedAll 1,211 probe–checkpoint
observations ask for current truthNo claim either way
Cross-
conversation
retrievalNot exercised
by LSREPEvery LSREP probe scoped to one
conversation; LongMemEval oracle
later exposes a fresh-session deficitWithin-conversation
claim only
HyDE rewrite Never isolated Exp 2: disabled. Exp 3: gated by
context-reliance, so active inall
arms including bare vector; the flag
toggled nothingNo claim either way
24

B Historical ICE v1 Pilot
Table 15:How the two LSREP instantiations differ. The shared skeleton—state reconstruction,
synchronous worker trigger, output-level judging by the same independent judge—is described in Section 3.
Dimension Experiment 1 (pilot) Experiment 2 (mature)
System maturity Vector + BM25 legs; MiniLM
classifier; static budget;
uncapped sliding window;
graph non-functionalAll legs enabled; Qwen3
classifier + routing override;
dynamic budget;
cluster-scoped retrieval
Corpus 18 conversations (isolated
checkpoints)4 long-horizon conversations
(full-lifespan replay)
Split range [max(10,0.30L),0.95L], 3
splits/convLength-adaptive checkpoints
(8–20), even spacing with
jitter
Probes Hand-written (8–30/conv) Auto-generated, leg-aware
(147) plus 72 inherited
hand-written
Ground truth Single retrieval-assisted pass,
frozenThree-stage forensic
regeneration + forward
propagation,evolving
Checkpoint state Independent per checkpoint Preserved across checkpoints;
lifecycle runs between them
Probe evaluation Each probe once Incremental: re-scored at
every later checkpoint
Scoring Static correctness Temporal-aware + inference
credit; hallucination flags
audited for false positives
Conditions Six (incl. sliding-window
controls)Four (controls dropped)
System under test.The pilot evaluated an immature build with only the vector and BM25
legs meaningfully active. The classifier used a frozen MiniLM backbone; retrieval used static
per-leg limits, a static 5,000-token global budget, and an uncapped 10-turn sliding window; the
knowledge graph effectively did not exist—a regex tagger and an under-powered 1.5B extractor
produced almost no usable edges—so the graph, procedural, and document legs contributed
nothing and the system reduced to essentially the same vector-plus-BM25 retrieval as the baseline;
cluster-scoped retrieval did not exist yet.
Protocol instantiation.Conversations shorter than 10 turns were excluded, reducing
the corpus from 42 to 18. Split points were drawn from [ max(10,0.30L),0.95L], three per
conversation with a fixed seed; each split defined a historical block and a 10-turn future reference
block. Probes were hand-written, 8–30 per conversation, and probes from earlier splits were
re-asked at later ones. Ground truth used a single retrieval-assisted pass, frozen thereafter. Six
conditions ran per probe.
The token-accounting audit.After the pilot completed, an audit found that the reported
context size for full ICE included retrieved fragments butomittedthe recent-turn sliding window
incorporated during prompt assembly. Generated answers, scores, rankings, and hallucination
labels were unaffected—the window was correctly supplied to the model at inference time; only
the reported efficiency statistics were wrong. We replayed the sliding-window assembly for every
checkpoint and added the omitted tokens. The correction did not change the qualitative findings,
but it invalidated any claim of superior token efficiency: corrected, full ICE consumesmoretotal
tokens than the baseline at roughly equal quality. All pilot numbers reported here are corrected
values.
25

Table 16:Experiment 1 (immature system), 657 probes, corrected token counts. After correcting
the sliding-window under-count, ICE used more tokens than the vector baseline, scored slightly lower,
hallucinated more, and had a worse TUR. These historical ICE v1 results do not measure frozen ICE v2.
Condition Score Tokens Hall. (%) TUR Win %
Control baseline (generalist) 3.05±1.61 3,998 74.1 0.76 11.3
Control MoE 3.01±1.61 3,998 74.4 0.75 7.6
Vector RAG (generalist)4.06±1.315,847 64.7 0.6922.9
Vector RAG (MoE) 4.05±1.32 5,847 66.7 0.69 15.6
Full ICE (generalist) 4.04±1.33 8,586 69.6 0.47 22.0
Full ICE (MoE) 3.96±1.33 8,586 72.3 0.46 20.6
The headline numbers were a defeat: 4.04 against 4.06, 47% more tokens, more hallucination,
a worse TUR. The picture brightened only on the longitudinal axis—193 tracked question curves
showed scores improving over time on repeated probes, demonstrating memory accumulation.
Several subsystems were broken or immature: only three decay cycles had run; the graph
effectively did not exist; 22 gating failures meant the classifier was actively blocking retrieval on
probes that needed it; and the budget was static.
Lessons.A short replay with incomplete consolidation tests that particular state, not
mature lifecycle behaviour. Honest token counting is essential—the under-count would have
made ICE look more efficient than it was. And a broken subsystem can drag down an entire
system, which is itself a finding about the fragility of multi-component architectures and a direct
ancestor of the fidelity audit in Appendix A.
C ICE v2 Full Ablation Results
Table 17:Cumulative feature addition on Dataset B (67 probes), full buildup. Paired within-probe
bootstrap, 10,000 resamples. Only +BM25 and +RRF clear a 95% CI; every other step spans zero.
Steps whose feature was defective, never exercised, or never isolated at the snapshot (+procedural,
+batch summaries, +HyDE) arenotinterpretable as neutral findings, and +Codex is under-weighted by
construction (Section 7.2.2). The vector baseline is a reference point, not a buildup step.
Step Score Step∆Paired∆95% CI SPF Tokens Frags
bare vector 3.27 — — — 0.23 14,873 14.0
+BM25 2.52−0.75−0.74 [−1.14,−0.36] 0.17 14,868 15.2
+RRF 3.36 +0.84 +0.82 [+0.39,+1.24] 0.29 14,870 11.7
+HyDE 3.39 +0.03 +0.03 [−0.13,+0.21] 0.29 14,870 11.6
+cluster restrict 3.40 +0.01 +0.01 [−0.16,+0.21] 0.30 14,911 11.3
+session diversify 3.40 +0.00 +0.00 [−0.22,+0.22] 0.30 14,911 11.2
+codex 3.43 +0.03 +0.03 [−0.21,+0.25] 0.30 14,912 11.3
+MERA 3.22−0.21−0.21 [−0.43,+0.01] 0.28 14,912 11.4
+procedural 3.39 +0.17 +0.16 [−0.07,+0.40] 0.30 14,912 11.4
+batch summary 3.41 +0.02−0.02 [−0.24,+0.23] 0.30 14,912 11.3
+dynamic budget 3.30−0.11−0.08 [−0.36,+0.18] 0.16 24,489 21.2
+sliding window 3.32 +0.02−0.02 [−0.24,+0.23] 0.16 27,763 21.3
+keyword boost 3.44 +0.12 +0.08 [−0.22,+0.37] 0.23 27,769 15.1
fullice 3.38−0.06−0.03 [−0.29,+0.23] 0.22 27,768 15.0
vector baseline (ref.) 3.42 — — — 0.11 16,373 30.0
26

Headline contrasts under the same paired bootstrap: the BM25 damage is −0.74
[−1.14,−0.36] and the RRF rescue +0 .82 [+0 .39,+1.24], both excluding zero; RRF-versus-
bare is +0 .09 [−0.18,+0.34], full ICE versus bare is +0 .09 [−0.24,+0.42], and full ICE versus
the single-leg baseline is−0.06 [−0.38,+0.25], all spanning zero.
Two steps warrant individual comment.MERA, the enumeration fallback for category
queries ( −0.21, [−0.43,+0.01], not significant), fires only when entity extraction returns nothing;
the audit reports a narrow trigger subset, so this delta rests on a small effective sample and
is a downstream indicator of extraction gaps rather than an independent evaluation.Codex
(+0.03) was live but emitted its entire traversal as one fragment, so fragment counts alone do
not measure graph quality.
bare
+BM25+RRF+HyDE+clust+sess+codex+MERA+proc+batch+budget+slide+kw
full ICE2.42.62.833.23.43.6
vector baseline 3.42Mean score after step
Figure 5:Feature ablation buildup (Dataset B, 67 probes). Each bar is the cumulative mean score after
adding one feature to all previous ones. The only two movements clearing a paired bootstrap CI are the
+BM25 drop and the +RRF recovery. The dashed line marks the single-leg vector baseline, which is not
a buildup step; the full-build contrast has uncertainty spanning zero.
Table 18:Recency effect by origin split (fact age). Probes grouped by the turn index at which they
were created. Recency ∆ is mean(late) −mean(early). The fused configurations show positive recency—
newer-origin probes have higher scores at the fixed final snapshot—while the single-leg baseline is slightly
negative.
Condition Early (0–400) Mid (400–800) Late (800–1200) Recency∆
bare vector 3.00 2.83 3.42 +0.42
+BM25 2.20 3.05 2.49 +0.29
+RRF 3.00 2.83 3.54 +0.54
+keyword boost 2.90 3.28 3.60 +0.70
fullice 3.10 3.25 3.47 +0.37
vector baseline 3.50 3.15 3.47−0.03
D ICE v2 Detailed LSREP Breakdowns
The four-dataset main-text table is table 4; additional distributions follow.
27

Table 19:Evaluation metrics. Each is computed per probe and aggregated across the benchmark.
Metric Definition What it measures
Score (1–5) Mean absolute score from the judge Answer quality
SPF mean score/mean fragments Score/fragment ratio
TUR mean score/(mean tokens/1000) Score/token ratio
Win rate (%) Blind tournament: all four conditions’
answers for a probe are anonymised,
shuffled, and ranked best-to-worst in a
single pass, independent of absolute
scoring; the win rate is the share of
appearances placed firstPreference under shuffled
presentation
Hallucination
(%)Share of probes where the judge detects
information not present in the
conversationFaithfulness
Fragment
countMean fragments injected per probe Retrieval volume
Recency ∆ mean score(late bins)−
mean score(early bins)Descriptive temporal
contrast
Table 20:Score distribution (1,057 probe–checkpoint observations, Dataset C excluded). The mean-score
tie masks a structural difference: ICE-MoE has the lowest score-1 rate but shifts mass from 5 to 3–4,
producing lower-variance output.
Score ICE gen (%) ICE MoE (%) Vec. gen (%) Vec. MoE (%)
1 (poor) 3.0 1.5 3.8 2.4
2 5.9 4.3 4.3 3.9
3 (acceptable) 13.3 16.6 14.8 19.5
4 (good) 18.1 19.7 17.2 16.3
5 (excellent) 59.7 58.0 60.0 58.0
Table 21:Temporal score quality (1,057 probe–checkpoint observations, Dataset C excluded). Buckets:
Good (4–5), OK (3), Poor (1–2).
Condition Good (4–5) % OK (3) % Poor (1–2) %
Vector RAG generalist 77.2 14.8 8.0
Vector RAG MoE 74.3 19.5 6.2
Full ICE generalist77.813.3 8.9
Full ICE MoE 77.7 16.6 5.8
28

E Frozen ICE v2 Implementation Reference
This appendix collects operational detail supporting Section 5 but not required to follow the
argument. A complete technical reference—the full training pipeline for the classifier and NER
models, the controlled relation vocabulary in its entirety, GPU resource management, and the
idempotency architecture—is the archived technical report in the repository, which describes
the system at the tagged evaluation snapshot.
E.1 Classification Engine
The engine produces, per user turn, a set of topic tags, a set of intent tags, and one context-
reliance label. These drive every downstream decision: which legs are weighted, how the budget
is split, whether the wide-net fallback fires, and which model the router selects.
The learned head is deliberately tiny—a 384 →128→25 multi-layer perceptron with
ReLU and 0.3 dropout. Its input is a single 384-dimensional embedding from a frozen Qwen3-
Embedding-0.6B encoder; the shared linear head is sliced at inference into three blocks: 11
topic labels, 11 intent labels, and 3 context-reliance classes. Topic and intent decode multi-label
(sigmoid, threshold 0.3, argmax fallback); context-reliance decodes single-label.
The rule-based pre-classifier computes five density signals in [0 ,1] from the raw prompt—code
density, sentiment density, meta density (references to the model itself), noise density, and
reference density (anaphoric terms)—and evaluates five rules in order, returning the first that
fires or none, in which case the learned head runs. The reference rule is special: it returns empty
topic and intent tags but forces Long Term Memory, and the orchestrator then runs the head
only to obtain topic and intent. A two-tier threshold on that rule (0.2 for short conversations,
0.1 past ten turns) is the engine’s primary long-conversation memory bias.
Override rulesrun in evaluation order after either path: memory immutability, so once
Long Term Memory is set nothing may downgrade it; a topic rule promoting creative turns
to Long Term Memory; and a rule promoting technical turns containing anaphora. A second,
API-level bias lives outside the classifier: past ten turns, or below 0.95 confidence, a Zero Shot
label is upgraded before retrieval runs. Section 8 discusses what this costs.
Context-aware classification.When a conversation ID is supplied, the learned path
queries the last three episodic turns for that conversation (max 500 words), preferring summaries
and falling back to the first 150 words of raw text, and prepends that context to the prompt
before embedding.
E.2 Codex Write Path and Retrieval
The graph spans four tables: entities (canonical name, aliases, properties JSONB, auto-
regenerated context payload, embedding); edges (typed relations with strength, a confidence flag
in{pending, active },valid from, and valid until , where NULL means treated as current by
the store); an append-only event log; and snapshots.
The triplet write path is the heart of temporal versioning. Entity resolution is two-stage—
exact canonical-name match, then alias match, else create with a deterministic UUIDv5. The
write branch then depends on the relation bucket: a property observation expires prior active
edges of the same (source, relation) pair, writes a new edge, and overwrites the JSONB key; a
reinforcement of an existing non-property edge increases strength and promotes from pending to
active at strength ≥2; a new relation between the same pair expires the old one only if the old
relation is not multi-valued. Every state change emits an event, and the entity’s context payload
is rebuilt from the property map plus the ten most recent active outgoing edges. The event log is
compacted by a worker that snapshots any entity exceeding 100 uncompacted events—textbook
event sourcing, with the live edge table as current state, the log as audit trail, and snapshots as
bounded replay.
Micro-NERis a BIO tagger (a 384 →128→64→3 MLP over per-token embeddings).
29

When the trained model is unavailable the system falls back to a capitalisation regex minus
a stoplist; at the Experiment 2 snapshot the trained model was present and loaded.Vector
fuzzy matchingembeds the extracted entity strings, scans entity embeddings, and takes the
best above a cosine threshold of 0.85—the mechanism that resolves a slightly misspelled proper
noun to its canonical entity.
The retrieval leg extracts prompt entities, resolves them by fuzzy matching, optionally
restricts to a conversation scope, and performs a BFS traversal to depth 3 over edges where
valid until IS NULL , appending a labelled context payload per visited entity.MERA, the
enumeration fallback for category queries, activates only when NER extracts no entitiesandthe
prompt contains both a category trigger and an enumeration hint; candidates are deduplicated
and ranked by 30-day mention count.
E.3 Codex Limitations in Detail
Extraction granularity.The frozen v2 extractor uses 6,000-token chunks with overlap. The
audit records extraction and representation limitations, but this study does not establish an
optimal chunk size or a general capacity limit for 4B models. Such claims require a controlled
extraction study.
Inert confidence and strength.Edges carry a confidence flag and a decaying numeric
strength, but both are under-used at retrieval time. Traversal follows any edge with valid until
IS NULL regardless of either, and the only effect of an active confidence flag is a coarse 1 .5×
fragment-level score boost. There is no reinforcement loop strengthening edges on retrieval, so
frequently referenced facts do not rise in the ranking and traversal treats weak edges as equal to
strong ones.
Lexical versus semantic entity matching.Retrieval resolves entities by canonical name
or alias plus fuzzy matching over the context-payload embedding rather than the entity name.
A user who establishes “The Obsidian Citadel” and later asks “where is the main fortress
located?” gives the resolution step no lexical or embedded anchor for “fortress,” though a human
reader resolves it immediately. This is a general limitation of entity-centric retrieval in free-form
conversation: it assumes users refer to entities by canonical names or close aliases, which natural
dialogue does not honour.
E.4 Prompt Assembly
The assembler emits a message list in a deliberately stable-prefix order to maximise cache reuse:
(i) a fixed system message with an inline persistent-context block rendering each active memory
slot; (ii) recent turns as alternating user/assistant pairs under dynamic per-turn word caps;
(iii) a single user message headed with the retrieved context, optionally tagged with cluster
names when a cluster scope is populated; (iv) a single assistant acknowledgement acting as a
boundary marker, so the model treats the final user message as the live question rather than
another history turn; and (v) the live question. Because the system message, slots, and most
of the recent-turn prefix change slowly, most of the prefix key/value tensors are in principle
reusable across consecutive requests; no prompt-cache benefit was measured in this study.
The orchestrator budgets selected fragment text. The frozen HTTP wrapper attempts an
additional word check, but computes only the system-message and question words, omitting
recent and retrieved messages. Its nominal 4,096-token-derived threshold is therefore not a valid
total-prompt ceiling. The experimental adapters call retrieval and assembly directly and do not
use this wrapper check. No secondary full-context guarantee is claimed.
Memory slotsare prepended verbatim to every prompt. Seven slot names are server-
enforced: persona, user preferences, tool guidelines, project context, guidance, pending items, and
session patterns. Slots are written through four paths: direct user update, batch initialisation,
reflection-proposed but user-gated updates (high-stakes slots land in a review queue), and
reflection-applied updates to pending items only.
30

Context clustersare conversation-scoped topical groupings. The centroid of member
turn embeddings, renormalised to unit length after every change, is a 384-dimensional vector;
membership is many-to-many. The clustering worker assigns each unassigned turn to the best
existing cluster—combining embedding similarity, a bonus per shared entity, and tag overlap—or
opens a new cluster when no candidate clears a 0.6 similarity threshold.
E.5 Background Workers and Infrastructure
Table 22:Background workers. Configured schedules, not evidence that every worker produced useful
output in the evaluation.
Worker Trigger Function
Post-flight evaluator every turn lossless detection, summary, dispatch extractors
Codex extractor lossless turns triplet extraction and versioned writes
Procedural extractor every turn pattern detection, reinforcement or insertion
Episodic decay 1.5 h access-weighted decay, archive at 0.1, cold-store
at 0.05
Codex decay 1.5 h edge strength decay, demote below strength 0.3
Procedural decay 1.5 h deactivation after 180 d with <3 reinforcements
Clustering 30 min assign turns; merging is separately callable
Reflection 2 h session synthesis, pattern crystallisation, slot
proposals
Batch summariser 2 h 50-turn batches, preservation-prompt summaries
Sentinel monitor 30 min declarative rules: threshold, absence, and others
Fine-tune weekly retrain the classifier head on curated labels
Compaction manual snapshot entities with≥100 uncompacted
events
Drop zone filesystem ingest documents into the document store
Codex inject watcher filesystem ingest structured files into the graph with
manual confidence
All GPU-touching workers gate on a utilisation threshold polled from the driver and, in shared-
model mode, additionally on a user-activity key with a short idle window. Retry countdowns
are fixed per class. Extraction tasks use idempotency keys; this is not a verified exactly-once
guarantee for every worker.
The system is a single service exposing an OpenAI-compatible chat-completions endpoint
plus routers for memory slots, user control, and the model registry, backed by PostgreSQL with
pgvector—one 384-dimensional vector column per vector table, because the embedder is shared—
and a worker fleet over a message broker. Configuration is a typed settings object read from the
environment. Server-sent event types ( classified ,retrieval ,context ready ,generating ,
degraded ) drive an observability panel and downstream monitoring. Classifier fine-tuning writes
a timestamped checkpoint but does not auto-promote it, so production promotion requires a
deliberate swap.
F Detailed ICE v2 Retrieval Configuration
F.1 Hybrid Retrieval Orchestrator
The orchestrator is the core of pre-flight and the component this study measures most directly.
Retrieval legs.Six legs are defined:BM25over a Postgres tsvector of raw and summary
text, filtered on decay score and archival status;vectorwith decay weighting, scoring (1 −
(embedding⇔query ))·decay score , which is what distinguishes this leg from pure semantic
31

search—a highly similar but decayed turn is down-ranked;Codex graph traversal;procedural
behind a hard intent gate;RAG, triple-gated and global rather than conversation-scoped;
andbatch summaries, conversation-scoped. Section 7.2.2 reports which of these actually
contributed during the evaluation, and why.
Dynamic leg weighting.Base weights are {bm25 0.8, vector 1.0, codex 0.5, procedural
0.2, rag 1.0 }. Five intent profiles override them—Factual Retrieval boosts vector and demotes
codex; Generation, Ideation, and Open Exploration do the reverse—and two cumulative topic
overrides then apply. This is the mechanism by which inferred intent conditions retrieval, and it
is the least-explored lever in the system.
Reciprocal Rank Fusioncombines the legs:
score RRF(f) =X
ℓ∈legsαℓ
k+ rank ℓ(f), k= 60,(1)
where αℓis the blended leg weight and rank ℓ(f) starts at 1. Fragments are deduplicatedduring
fusion: the first occurrence is registered and later occurrences add to its score, so agreement
across legs is rewarded. Because RRF consumes ranks rather than scores, heterogeneous legs
need no score normalisation to be combined—a plausible explanation for the observed corrective
effect (Section 7.2.1).
Post-fusion curationapplies a sequence of transforms; their combined condition has the
reported fragment reduction, without individual causal attribution. Additive bonuses reward
keyword overlap and length and penalise very short fragments; a recency bonus favours the
most recent decile but isskippedfor creative turns, because recent meta-discussion is noise for
narrative work. Session diversification caps foreign conversations at three fragments each while
leaving the active conversation uncapped. Deduplication hashes fragment text. A two-phase
token budget then prioritises source-type diversity—the best fragment of each source type is
admitted first if it fits—before greedily filling the remainder. Finally the strengthening step
writes back the access-count and decay-score restoration described above.
Cluster-scoped retrievalrestricts episodic search to the most relevant topical clusters,
falling back to global search when no cluster clears a threshold.
Dynamic token budget.A total context budget of 23,000 tokens less an overhead reserve
leaves 21,200 available. A length-based recent-window fraction, reduced by token density and
shifted by intent and topic, splits this between recent turns and retrieval, and agrowth cap
bracketed by turn count prevents long conversations from over-allocating. Leftover budget is
intentionally left unused: this constrains the retrieved set and, under extreme per-turn density,
averts the context overflow that sinks an unbudgeted baseline (Section 7.1.1). On ordinary
turns ICE’s total token count is comparable to the baseline’s—the curation shows up as fewer
fragments, not fewer tokens. A configurable subclass of the orchestrator exposes flags that switch
legs and post-processing steps on and off without redefining classification or fusion; Section 7.2.1’s
buildup uses that mechanism.
Prompt assembly, memory slots, and the faulty HTTP wrapper check are documented in
Appendix E.4; they are not repeated here.
32

Six defined re-
trieval legs
lexical (“BM25”)
decay-weighted vector
Codex traversal
procedural
documents
batch summariesLeg weights
↓
RRF fusionSequential curation
bonuses →session diversity
→deduplication
→two-phase token budget
→strengthen selected memory
Selected fragments →prompt assembler
Figure 6:Retrieval orchestrator. The legs are heterogeneous in retrieval semantics; RRF is what allows
them to be combined without score normalisation; the two-phase budget prioritises source-type diversity
under budget pressure. Section 7.2.2 reports which legs contributed during evaluation.
G Matched LongMemEval Paired Inference
All numbers below concern ICE v2. Resampling preserves question identity across arms and,
for the degradation contrast, across phases. Intervals are 20,000-resample percentile intervals
with seed 20260911. Positive differences favour ICE. Category intervals are exploratory and
unadjusted.
Table 23:Oracle paired outcomes. CC: both correct; I: ICE only; V: vector only; WW: both wrong.
Counts include only paired obtainable verdicts.
Category CC I V WW ∆ (pp) 95% CI
Abstention 17 8 1 4 +23.3 [+6.7,+40.0]
Knowledge update 37 5 16 14 -15.3 [−27.8,−2.8]
Multi-session 36 0 68 17 -56.2 [−65.3,−47.1]
Session assistant 49 2 4 1 -3.6 [−12.5,+5.4]
Session preference 18 3 3 6 +0.0 [−16.7,+16.7]
Session user 49 1 12 2 -17.2 [−28.1,−7.8]
Temporal reasoning 22 7 32 66 -19.7 [−28.3,−10.2]
Overall 228 26 136 110 -22.0 [−26.6,−17.4]
Table 24:Full-S paired outcomes. CC: both correct; I: ICE only; V: vector only; WW: both wrong.
Counts include only paired obtainable verdicts.
Category CC I V WW ∆ (pp) 95% CI
Abstention 18 7 1 4 +20.0 [+3.3,+36.7]
Knowledge update 31 6 19 16 -18.1 [−30.6,−5.6]
Multi-session 24 2 66 29 -52.9 [−62.8,−43.0]
Session assistant 41 1 12 2 -19.6 [−32.1,−8.9]
Session preference 9 4 6 10 -6.9 [−27.6,+13.8]
Session user 46 0 14 4 -21.9 [−32.8,−12.5]
Temporal reasoning 22 4 38 63 -26.8 [−35.4,−18.1]
Overall 191 24 156 128 -26.5 [−31.3,−21.8]
The full-S missing vector judgement is in the preference category. Its paired ICE score
is 13/29, while the marginal table uses 13/30. The overall phase-degradation contrast uses a
common denominator of 499 and therefore differs from subtracting the two displayed marginal
accuracy gaps.
33

H Matched LongMemEval Cost Strata
The following table reports median [25th, 75th percentile] for provider input tokens by category
and outcome, plus median selected fragments and generation seconds. C/W are correct/incorrect
according to Muse; the single missing verdict is excluded from this stratification. The machine-
readable aggregate artifact also includes word-estimated prompt-token distributions and 95th
percentiles for every stratum. Retrieval-only context tokens, per-leg candidates, retrieval latency,
construction cost per arm, and transient failure rates are unavailable. These tables describe
associations, not interventions on context size.
Table 25:ICE v2 public diagnostic: cost conditioned on category and correctness.
Phase Arm Category C/WnFrags Sec. Input tokens [IQR]
Oracle ICE
v2Abstention C 25 3.0 3.0 1,994 [1,793, 2,055]
Oracle ICE
v2Abstention W 5 4.0 3.5 2,212 [2,085, 2,249]
Oracle ICE
v2Knowledge
updateC 42 4.0 2.1 2,058 [1,994, 2,108]
Oracle ICE
v2Knowledge
updateW 30 4.0 2.7 2,077 [2,002, 2,156]
Oracle ICE
v2Multi-session C 36 4.0 2.8 2,090 [1,956, 2,158]
Oracle ICE
v2Multi-session W 85 3.0 3.0 2,081 [2,003, 2,190]
Oracle ICE
v2Session assistant C 51 3.0 2.3 878 [724, 1,023]
Oracle ICE
v2Session assistant W 5 3.0 3.1 991 [729, 1,052]
Oracle ICE
v2Session
preferenceC 21 3.0 9.6 1,959 [1,747, 2,021]
Oracle ICE
v2Session
preferenceW 9 3.0 6.8 1,930 [1,553, 2,010]
Oracle ICE
v2Session user C 50 3.0 2.0 1,710 [1,497, 1,909]
Oracle ICE
v2Session user W 14 3.0 2.6 1,850 [1,522, 2,037]
Oracle ICE
v2Temporal
reasoningC 29 3.0 2.8 2,011 [1,931, 2,098]
Oracle ICE
v2Temporal
reasoningW 98 3.0 3.0 2,068 [1,934, 2,144]
Oracle ICE
v2Overall C 254 3.0 2.4 1,928 [1,440, 2,066]
Oracle ICE
v2Overall W 246 3.0 3.0 2,065 [1,947, 2,153]
Oracle Vector Abstention C 18 11.0 2.3 5,462 [3,840, 5,817]
Oracle Vector Abstention W 12 12.0 2.5 6,268 [5,060, 6,876]
Oracle Vector Knowledge
updateC 53 12.0 1.5 5,926 [5,167, 6,502]
Oracle Vector Knowledge
updateW 19 12.0 1.5 5,658 [5,246, 6,562]
Oracle Vector Multi-session C 104 12.0 2.2 6,544 [5,681, 9,767]
Oracle Vector Multi-session W 17 18.0 3.5 9,324 [6,902, 11,975]
Oracle Vector Session assistant C 53 4.0 1.6 950 [647, 1,384]
34

Phase Arm Category C/WnFrags Sec. Input tokens [IQR]
Oracle Vector Session assistant W 3 6.0 2.9 1,218 [970, 1,490]
Oracle Vector Session
preferenceC 21 7.0 7.1 3,849 [3,370, 4,245]
Oracle Vector Session
preferenceW 9 6.0 5.2 3,693 [3,521, 4,456]
Oracle Vector Session user C 61 6.0 1.4 3,015 [2,555, 3,441]
Oracle Vector Session user W 3 6.0 2.0 2,970 [2,656, 3,140]
Oracle Vector Temporal
reasoningC 54 12.0 2.0 6,496 [5,742, 6,910]
Oracle Vector Temporal
reasoningW 73 12.0 2.3 6,388 [4,635, 7,782]
Oracle Vector Overall C 364 11.0 1.9 5,261 [3,014, 6,539]
Oracle Vector Overall W 136 12.0 2.5 6,160 [4,620, 7,712]
S ICE
v2Abstention C 25 5.0 2.8 2,211 [2,150, 2,262]
S ICE
v2Abstention W 5 4.0 4.3 2,259 [2,228, 2,302]
S ICE
v2Knowledge
updateC 37 5.0 2.4 2,185 [2,139, 2,251]
S ICE
v2Knowledge
updateW 35 5.0 2.7 2,240 [2,184, 2,292]
S ICE
v2Multi-session C 26 6.0 2.8 2,230 [2,199, 2,278]
S ICE
v2Multi-session W 95 5.0 3.3 2,233 [2,185, 2,298]
S ICE
v2Session assistant C 42 6.0 2.5 2,251 [2,189, 2,292]
S ICE
v2Session assistant W 14 6.0 3.8 2,191 [2,167, 2,242]
S ICE
v2Session
preferenceC 13 5.0 7.5 2,229 [2,217, 2,321]
S ICE
v2Session
preferenceW 17 5.0 6.8 2,223 [2,167, 2,267]
S ICE
v2Session user C 46 5.0 2.1 2,199 [2,148, 2,278]
S ICE
v2Session user W 18 5.0 3.3 2,202 [2,152, 2,288]
S ICE
v2Temporal
reasoningC 26 5.5 3.0 2,245 [2,189, 2,297]
S ICE
v2Temporal
reasoningW 101 5.0 3.3 2,211 [2,143, 2,264]
S ICE
v2Overall C 215 5.0 2.6 2,224 [2,170, 2,289]
S ICE
v2Overall W 285 5.0 3.4 2,219 [2,167, 2,288]
S Vector Abstention C 19 30.0 2.6 11,391 [10,259, 12,813]
S Vector Abstention W 11 30.0 3.4 11,922 [11,408, 12,202]
S Vector Knowledge
updateC 50 30.0 2.0 12,026 [10,295, 12,706]
S Vector Knowledge
updateW 22 30.0 2.2 11,558 [10,738, 12,482]
S Vector Multi-session C 90 30.0 2.7 12,084 [11,153, 13,300]
35

Phase Arm Category C/WnFrags Sec. Input tokens [IQR]
S Vector Multi-session W 31 30.0 3.0 11,844 [10,791, 13,518]
S Vector Session assistant C 53 30.0 2.2 10,860 [9,118, 11,853]
S Vector Session assistant W 3 30.0 3.1 9,426 [8,277, 11,730]
S Vector Session
preferenceC 15 30.0 5.5 12,494 [11,953, 13,725]
S Vector Session
preferenceW 14 30.0 6.1 12,688 [11,168, 13,912]
S Vector Session user C 60 30.0 1.9 10,802 [9,786, 12,251]
S Vector Session user W 4 30.0 2.2 11,152 [10,630, 11,450]
S Vector Temporal
reasoningC 60 30.0 2.8 11,732 [10,954, 13,054]
S Vector Temporal
reasoningW 67 30.0 3.1 12,097 [10,625, 13,203]
S Vector Overall C 347 30.0 2.5 11,650 [10,350, 12,800]
S Vector Overall W 152 30.0 3.1 11,854 [10,687, 13,126]
I Historical Local Oracle and Adapter Invalidation
Before the matched cloud study, the corrected v2 adapter used gemma4:26b-a4b-it-q4 KMfor
both answer arms and local gemma4:12b judging. ICE obtained 264/478 (55.2%) and vector
388/484 (80.2%) on the oracle; 22 and 16 judge mutes gave all-500 bounds of 52.8–57.2% and
77.6–80.8%. That study stopped before full-S. Its scores are historical diagnostics, not part of the
matched cloud comparison, and the later completed cloud phases supersede the stopped-phase
statement.
The earlier flattened-session adapter is invalid. A trace showed 45 lexical plus 100 vector
candidates, 111 unique after fusion, then only three after diversification because a UUID object
did not equal a string identifier. Its query placement also failed to preserve supplied session
boundaries. These results remain excluded. The corrected session adapter separates history
sessions and queries from an empty conversation; it does not inflate the fresh-session retrieval
budget from the total haystack length.
J LSREP Sensitivity and Reproduction
The historical record-level bootstrap treats repeated checkpoint observations as separate sampling
units. A sensitivity analysis resamples 219 distinct probe clusters, retaining all observations
within each selected probe and the ICE/vector pair. It targets the same observation-weighted
score difference rather than giving every distinct probe equal weight. With 20,000 resamples
and seed 20260911, all-data ∆ = +0 .396, CI [+0 .192,+0.618]; ordinary-density ∆ = +0 .002,
CI [−0.148,+0.158] (182 clusters); density-only ∆ = +3 .097, CI [+2 .721,+3.470] (37 clusters).
Thus the principal qualitative findings survive while the uncertainty is wider. Conversations and
users are not independently resampled; no population claim follows.
The public analysis entry points are experiments/lme/analyze_matched.py and
experiments/mature/clustered_sensitivity.py . The former reads local arm verdicts and
cost metadata and exports only aggregate cells and distributions; the latter reuses the archived
manual-merge and score-imputation semantics. No models are rerun. Both must be run from
the repository root through uv run python . The aggregate files contain no question text,
answers, judgements, or personal identifiers.
36

J.1 Scoring-Source, Missing-Score, and Ordinal Sensitivity
Table 26 crosses two choices: retaining versus removing the 72 manual replacements, and using
the archived fallback chain versus requiring explicit scores in both generalist arms. All intervals
resample entire probe trajectories with pairs preserved (20,000 resamples, seed 20260911).
Complete-case selection is informative: a failed vector answer often has no explicit score. The
automatic-only complete-case density subset contains just 13 of 154 observations, and therefore
says little about reliability over the original workload. No missing-at-random assumption is
made.
Table 26:ICE v2 minus vector generalist under alternative scoring policies. “Archived” includes failure
assignment and fallback; “complete” requires two explicit scores. These are sensitivity analyses of recorded
outcomes, not new model runs.
Scoring Regimen∆ Probe-cluster 95% CI
Merged, archived All 1,211 +0.396 [+0.192,+0.618]
Merged, archived Ordinary 1,057 +0.002 [−0.148,+0.158]
Merged, archived Density 154 +3.097 [+2.721,+3.470]
Automated, archived All 1,211 +0.363 [+0.162,+0.582]
Automated, archived Ordinary 1,057−0.020 [−0.169,+0.136]
Automated, archived Density 154 +2.994 [+2.606,+3.359]
Merged, complete All 1,079 +0.046 [−0.101,+0.199]
Merged, complete Ordinary 1,055 +0.001 [−0.148,+0.157]
Merged, complete Density 24 +2.042 [+1.077,+3.294]
Automated, complete All 1,067−0.021 [−0.168,+0.128]
Automated, complete Ordinary 1,054−0.022 [−0.170,+0.131]
Automated, complete Density 13 +0.077 [−0.714,+0.714]
The merged generalist vector scores comprise 1,079 explicit labels, 130 failed-answer assign-
ments to 1, and two sibling-routing substitutions. Without manual replacements the counts are
1,067, 141, and three. ICE generalist has 1,211 explicit scores under both policies. Across the
other two routing arms, vector-MoE has 34 failed-answer assignments and ICE-MoE one sibling
substitution. Neither policy uses the rounded-record-mean or default-3 fallback. The analysis
reproduces every merged score from the archived aggregation before estimating sensitivity.
As an ordinal check, we use the direction-only principle of paired sign comparisons (National
Institute of Standards and Technology, n.d.), reporting paired win/tie/loss counts and the net
superiority proportion Pr(SICE> S vector)−Pr(SICE< S vector). We use probe-cluster bootstrap
intervals rather than an independent-observation binomial sign test. This compares ordering
alone; it does not claim that a change from 1 to 2 equals a change from 4 to 5. In all data,
the merged scores give 355 ICE-higher observations, 220 vector-higher, and 636 ties: net +11 .1
points (cluster CI [+3 .2,+19.5]). Ordinary-density net superiority is +0 .1 ([−7.6,+8.0]), and
density-only is +87 .0 ([+77 .5,+95.3]). Failure assignment remains part of this ordinal reliability
comparison. These results supplement the historical mean rubric scores; SPF and TUR remain
descriptive engineering summaries, not validated substitutes for answer quality.
The reproducible entry point is experiments/mature/scoring_sensitivity.py ;
experiments/paper/generate_analysis_tables.py emits the aggregate TeX table and
shared ablation macros. The ablation report now reuses each paired contrast across its
step, cumulative, and headline views, eliminating small Monte Carlo discrepancies caused by
resampling the same contrast repeatedly.
37
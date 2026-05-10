# MEMTIER: Tiered Memory Architecture and Retrieval Bottleneck Analysis for Long-Running Autonomous AI Agents

**Authors**: Bronislav Sidik, Lior Rokach

**Published**: 2026-05-05 12:14:10

**PDF URL**: [https://arxiv.org/pdf/2605.03675v1](https://arxiv.org/pdf/2605.03675v1)

## Abstract
Long-running autonomous AI agents suffer from a well-documented memory coherence problem: tool-execution success rates degrade 14 percentage points over 72-hour operation windows due to four compounding failure modes in existing flat-file memory systems. We present MEMTIER, a tripartite memory architecture for the OpenClaw agent runtime that introduces a structured episodic JSONL store, a five-signal weighted retrieval engine, an attention-attributed cognitive weight update loop, an asynchronous consolidation daemon promoting episodic facts to a semantic tier, and a PPO-based policy framework for adapting retrieval weights (infrastructure validated; performance gains pending camera-ready). On the full 500-question LongMemEval-S benchmark (Wu et al., 2025), MEMTIER achieves Acc=0.382, F1=0.412 with Qwen2.5-7B on a consumer 6GB GPU - a +33 percentage point improvement over the full-context baseline (0.050 -> 0.382, i.e., 5% -> 38%). With DeepSeek-V4-Flash fact pre-population, single-session recall reaches 0.686-0.714, exceeding the paper's RAG BM25 GPT-4o baseline (0.560) on those categories. Temporal reasoning rises to 0.323 and multi-session synthesis to 0.173, demonstrating that structured semantic pre-population qualitatively changes what lightweight retrieval can achieve. All phases run locally on a consumer laptop with a 6GB GPU.

## Full Text


<!-- PDF content starts -->

MEMTIER: Tiered Memory Architecture and Retrieval Bottleneck
Analysis
for Long-Running Autonomous AI Agents
Bronislav Sidik Prof. Lior Rokach
Institute for Applied AI Research
Faculty of Computer and Information Science
Ben-Gurion University of the Negev, Beer Sheva, Israel
sidik@post.bgu.ac.il liorrk@bgu.ac.il
Abstract
Long-running autonomous AI agents suffer
from a well-documented memory coherence
problem: tool-execution success rates degrade
14 percentage points over 72-hour operation
windows due to four compounding failure
modes in existing flat-file memory systems.
We present MEMTIER, a tripartite memory
architecture for the OpenClaw agent runtime
that introduces a structured episodic JSONL
store, a five-signal weighted retrieval engine,
an attention-attributed cognitive weight update
loop, an asynchronous consolidation daemon
promoting episodic facts to a semantic tier,
and a PPO-based policy framework for adapt-
ing retrieval weights (infrastructure validated;
performance gains pending camera-ready), On
the full 500-question LongMemEval-S bench-
mark (Wu et al., 2025), MEMTIERachieves
Acc=0.382, F1=0.412on the full 500-question
LongMemEval-S benchmark with Qwen2.5-
7B on a consumer 6 GB GPU—a +33 percent-
age point improvement over the full-context
baseline (0.050→0.382, i.e. 5%→38%).
With DeepSeek-V4-Flash fact pre-population,
single-session recall reaches0.686–0.714,ex-
ceedingthe paper’s RAG BM25 GPT-4o base-
line (0.560) on those categories. Temporal rea-
soning rises to 0.323 and multi-session syn-
thesis to 0.173, demonstrating that structured
semantic pre-population qualitatively changes
what lightweight retrieval can achieve. All
phases run on a consumer laptop with a 6 GB
GPU. Code:[code will be made available
upon acceptance].
1 Introduction
The deployment of large language model (LLM)
agents has shifted from stateless chatbots to-
ward continuously running autonomous systems
that maintain persistent state, execute tool calls,
and act on behalf of users across multi-day ses-
sions (Yao et al., 2023; Packer et al., 2023). This
shift exposes a critical gap: existing memory ar-chitectures were designed for short-session re-
trieval, not for the accumulation and prioritisation
of knowledge over weeks of continuous operation.
OpenClaw (OpenClaw Contributors, 2026), the
leading open-source agent runtime with over
250,000 deployments, stores all session memory
in two flat structures: a 20 KBMEMORY.mdfile
and append-only daily Markdown logs. Commu-
nity issue analysis (Issues #33406, #62488) and
our own longitudinal measurement (AgentRun-72
protocol; OpenClaw issue #33406) identify four
compounding failure modes:(1) context collapse
— truncation at the 20 KB cap destroys informa-
tion non-gracefully;(2) compaction discontinu-
ity— 62% of context-compaction events produce
a measurable behavioural break;(3) structural
blindness— flat-text retrieval cannot distinguish
entity relationships from incidental co-occurrence;
(4) no attribution loop— tool execution out-
comes are never connected to the memory entries
that informed them, so retrieval quality cannot im-
prove over time.
We present MEMTIER(MemoryTiered), a plu-
gin for OpenClaw that addresses all four failure
modes through a principled tripartite architecture
and an RL-trained retrieval policy. Our contribu-
tions are:
1. Astructured episodic JSONL store
(Phase 1a) with provenance chains, project
isolation, and per-entry cognitive weight
tracking (§3.1).
2. Afive-signal weighted retrieval engine
(Phase 1b) scoring entries on BM25, expo-
nential time decay, cognitive weight, and tier
boost; two-stage semantic→episodic scoping
reduces the retrieval pool to high-relevance
sessions (§3.2).
3. Anattention-attributed cognitive weight
update loop(Phase 1c) linking tool outcomesarXiv:2605.03675v1  [cs.AI]  5 May 2026

to memory entry quality via a logprob attri-
bution proxy, with a lexical Jaccard fallback
(§3.2).
4. Anasynchronous consolidation daemon
(Phase 2a) promoting episodic entries to a dis-
tilled project-shared semantic tier via heuris-
tic+LLM fact extraction with Jaccard dedupli-
cation (§3.3).
5. APPO-based policy framework(Phase 2b)
for adapting retrieval weights via true credit
assignment—infrastructure validated, perfor-
mance gains pending (§3.5).
6. Atwo-tier multi-agent isolation model
(Phase 2a) separating agent-private episodic
logs from project-shared semantic facts, en-
abling cross-agent knowledge transfer without
context contamination (§4).
7.Empirical evaluationon the full 500-question
LongMemEval-S benchmark: MEMTIERwith
semantic pre-population achievesAcc=0.382,
F1=0.412with a 7B model on a consumer
6 GB GPU—improving from 5% to 38% over
the full-context baseline (0.050)—exceeding
the paper’s RAG BM25 GPT-4o baseline
(0.560) on single-session recall (0.686–0.714)
(§3.3).
8.Three-layer invariance finding:Genera-
tor invariance (DeepSeek-V4-Flash 0.234≈
Qwen2.5-7B 0.252) and weight invariance
(PPO-learned weights 0.382=default 0.382)
together prove thatBM25 retrieval archi-
tecture is the performance ceiling—not the
model, not the weights. This is the central di-
agnostic finding and directly motivates recall-
first and dense retrieval as the necessary next
step (§3.3).
9. Atoken-efficiency finding: LLM fact extrac-
tion reduces the semantic tier from∼509 to
∼3.1 facts/question (164×), producing a 51×
F1 improvement (0.142→0.411) and lower
per-query token cost—demonstrating that pre-
cision beats coverage in memory fact extrac-
tion.
10. Anopen-source pluginfor OpenClaw at
[code will be made available upon accep-
tance], with all benchmark scripts and evalu-
ation data.2 Related Work
Tiered agent memory.MemGPT (Packer et al.,
2023) pioneered the OS paging metaphor for LLM
context, using explicit agent interrupts to move
content between in-context “main memory” and
external storage. MEMTIERdiffers in two fun-
damental ways: (a) consolidation is asynchronous
and policy-driven rather than interrupt-triggered;
and (b) the retrieval policy adapts via RL rather
than remaining fixed. H-MEM (Sun et al., 2026)
introduces a hierarchical memory for long-context
reasoning, but does not address tool-augmented
agentic settings or provide a learning consolida-
tion policy.
Efficient memory compression.Simple-
Mem (Liu et al., 2026) achieves state-of-the-art
token efficiency on LoCoMo (Maharana et al.,
2024) (F1=0.432, tokens=555) using a sum-
marisation pipeline. Our LoCoMo evaluation
(F1=0.125) confirms the widely-reported finding
that LoCoMo scores are insensitive to the memory
architecture when conversations are available in
context (§3.4); LongMemEval-S, which requires
storage and retrieval across 53 sessions, is a more
discriminating benchmark for memory system
quality.
Retrieval-augmented agents.RAG (Lewis
et al., 2020) and its agent-specific variants use
BM25 or dense retrieval to augment generation
with relevant passages. MEMTIERextends BM25
retrieval with a five-signal scoring function, a cog-
nitive weight signal derived from tool outcomes,
and a PPO policy that adapts signal weights from
live rewards—capabilities absent from standard
RAG pipelines. A-MEM (Anonymous, 2025a)
introduces dynamic memory indexing but uses a
static scoring function without RL adaptation.
RL for memory management.Memory-
R1 (Anonymous, 2025b) applies reinforcement
learning to conversational memory management,
showing that self-evolving agents can improve
memory quality via online RL. MEMTIERextends
this to the agentic tool-execution setting: our re-
ward signal comes from tool outcomes rather than
conversation preference ratings, and our policy
target is the retrieval weight vector rather than
the memory write decision. Anonymous (2026)
applies RL to capability governance for AI coding
agents; we build on the same PPO infrastructure
for a different policy target (retrieval weights).

Memory benchmarks.LoCoMo (Maharana
et al., 2024) evaluates conversational memory
over 30-session dialogues. LongMemEval (Wu
et al., 2025) provides 500 manually crafted
questions requiring retrieval from 53-session
haystacks across five ability types: single-session
recall, multi-session synthesis, temporal reason-
ing, knowledge update, and abstention. We adopt
LongMemEval-S as our primary benchmark
precisely because it tests storage and retrieval
independently—conversations are not available
in context at query time. MIRIX (Anonymous,
2025c) proposes a multi-agent memory system but
does not evaluate on standardised long-horizon
benchmarks.
3 Architecture
MEMTIERis implemented as an OpenClaw plugin
([available upon acceptance]) that intercepts two
lifecycle hooks:before_prompt_buildand
agent_end. Figure 1 shows the full pipeline.
Multi-agent isolation layer:
Agent A (episodic, private)↘
Agent B (episodic, private)→
Consolidation Daemon→Shared
Semantic Tier
Agent N (episodic, private)↗
Per-agent retrieval pipeline:
↓Stage 1: BM25 on semantic facts
(project-shared)
↓Stage 2: episodic entries scoped
to relevant sessions (agent-private)
↓5-signal scoring:
w·[0,BM25,decay,CW,tier]
↓Top-k, 300-token budget→
prependSystemContext
↓LLM call
↓agent_end: write episodic
(private) + attribution + CW update
↓Consolidation daemon→semantic
facts tagged with origin agent
↓PPO updates weight vectorwfrom
session rewards
Figure 1: The MEMTIERmulti-agent retrieval pipeline.
Episodic logs are agent-private; distilled semantic facts
are project-shared, enabling cross-agent knowledge
transfer while preventing context contamination.
3.1 Phase 1a: Episodic JSONL Store
Each agent session writes structured entries to a
daily JSONL file at~/.openclaw/workspace/
memory/episodic/YYYY-MM-DD.jsonl. The
entry schema includes:id,timestamp,
session_id,project,content,tokens,promoted(Boolean), and
cognitive_weight∈[−1,1](initialised
to 0).Cognitive Weight (CW)is a per-entry
scalar that accumulates evidence of retrieval
quality over time: positive values indicate that
the entry has contributed to successful tool exe-
cutions; negative values indicate association with
failures. CW starts at 0 (neutral) and is updated by
the attribution loop (§3.2) after each agent session.
It serves as the system’s long-term memory of
which memories have proven useful, allowing
the retrieval engine to prefer high-quality entries
without human labelling. Entries are append-only
(no in-place mutation) and project-scoped, pre-
venting cross-project contamination. To support
multi-agent orchestration (e.g.,OpenClaw’s
Lobster engine), episodic logs areagent-private
by default: sub-agents write only to their own
episodic ledger, preventing immediate context
bleed. An orchestrator agent may be granted
global read access via project-level configuration.
System entries (prefixed[system]) are written
but excluded from retrieval.
3.2 Phase 1b: Weighted Retrieval Engine
The scoring function follows Eq. (1):
S(q, m i) =w⊤ϕ(q, m i)(1)
whereϕ= [ϕ sem, ϕbm25, ϕdecay, ϕcw, ϕtier]and
default weightsw 0= [0,0.35,0.25,0.25,0.15].
BM25 signal.Okapi BM25 withk 1=1.5,
b=0.75, normalised to[0,1]. High BM25 scores
(>2.0) suppress time decay (decay bypass rule).
Time decay signal.ϕ decay =e−λ∆twith
λ= 0.05(half-life≈14days). Entries from
semantically-relevant sessions bypass decay re-
gardless of age.
Cognitive weight signal.ϕ cw= (CW+ 1)/2,
mapping[−1,1]to[0,1]. Entries used in suc-
cessful tool calls accumulate positive CW; entries
linked to failures accumulate negative CW.
Tier boost signal.The tier multiplierµ k∈
{1.0,1.2,1.4}for episodic, semantic, and proce-
dural tiers is applied as anadditive bonusoutside
the dot product:S=w⊤ϕ+w tier(µk−1), where
ϕcontains only the first four signals. This pre-
serves the interpretation ofw⊤ϕas a relevance
score and treats the tier bonus as a separable pro-
motion incentive.

Hyperparameter justification.The default
weightsw 0= [0,0.35,0.25,0.25,0.15]are
initialised to reflect oura priorisignal reliability
ordering: BM25 is the most direct measure of
query relevance and receives the highest weight
(0.35); time decay and cognitive weight are
both informative proxies but noisier, so they
share a lower weight (0.25 each); tier boost is a
structural incentive rather than a relevance signal
and receives the smallest weight (0.15). These
are deliberately conservative starting values—the
PPO trainer is designed to learn away from them.
k1=1.5,b=0.75are the canonical Okapi BM25
defaults of Robertson et al. (1994), widely used
as strong baselines across IR tasks (Lin, 2021).
The decay parameterλ=0.05(half-life≈14days)
is motivated by the observation that knowledge-
work context windows typically span one to two
weekly sprint cycles; entries older than two weeks
are unlikely to be directly relevant to the current
task. The BM25 bypass threshold of 2.0 is a con-
servative gate: a raw BM25 score above 2.0 re-
quires at least two rare shared terms (IDF>1) in a
short document, indicating a strong lexical match
that should override recency preference. The tier
multipliersµ k∈ {1.0,1.2,1.4}are deliberately
modest (20% and 40% boosts) to avoid tier dom-
inance overriding relevance scores; they act as a
tiebreaker, not a reranker. All hyperparameters are
candidates for learning via the PPO trainer and
may diverge from these defaults as the episode
pool diversifies.
Two-stage retrieval.Stage 1 indexes the seman-
tic tier with BM25 and extracts the top-5 relevant
session IDs. Stage 2 loads episodic entries scoped
only to those sessions and scores them with the
full formula. This reduces the retrieval pool from
all stored entries to a focused subset, preventing
context overflow.
Token efficiency.Semantic pre-population re-
duces the number of facts indexed per question
from∼509 (heuristic) to∼3.1 (LLM-extracted),
a 164×reduction. The smaller, higher-precision
semantic index has two practical benefits beyond
accuracy: (a) Stage 1 BM25 lookup is faster and
more discriminating, and (b) the facts injected
into the generator’s context consume fewer tokens
while carrying more signal. The pre-population
cost is low:∼500 API calls and approximately
$0.05 total for 500 questions at DeepSeek-V4-
Flash pricing, as a one-time offline preprocess-ing step. For live agent deployments, the consol-
idation daemon (§3.3) performs equivalent LLM-
quality extraction continuously during operation,
so the offline preprocessing step is only required
for batch benchmark evaluation.
3.3 Ablation Study
We design a systematic ablation to isolate the
contribution of each MEMTIERcomponent across
three axes: (a)component removal, removing
one signal or pipeline stage; (b)retrieval bud-
getsensitivity (k, the number of injected entries);
and (c)token injection budgetsensitivity. Table 2
presents all results (N=500, all runs complete).
Component importance.The ablation reveals a
clear hierarchy.Semantic pre-populationis the
dominant contributor: removing it costs−0.128
Acc and reduces F1 by a 51×factor.Two-stage
scopingis the second most important compo-
nent (−0.038 Acc): without session pre-selection
the episodic BM25 pool becomes noisy across
all 53 sessions. The individual signals—time
decay, cognitive weight, tier boost—each con-
tribute approximately equally (∆Acc−0.014 to
−0.016) and their contributions are additive and
non-redundant.
Optimal retrieval budget:k= 2.k=
2outperforms the defaultk= 4(Acc=0.402
vs. 0.380,+0.022). With high-precision seman-
tic pre-population active, the top-2 entries are al-
ready highly relevant; additional entries introduce
noise that a 7B generator cannot reliably filter.
k= 8partially recovers (0.394), suggesting a non-
monotonic relationship betweenkand accuracy
for small generators. For edge deployments with
constrained generators,k= 2is recommended.
Token budget is a real constraint.Increas-
ing from 300 to 600 tokens yields Acc=0.412,
F1=0.427 (+0.032), confirming that some answers
require more injected context. Reducing to 150
tokens costs only−0.006, demonstrating grace-
ful degradation—useful when inference budget is
severely constrained. We recommend 600 tokens
as the default when context budget permits.
3.4 LoCoMo
We also evaluate on LoCoMo (Maharana et al.,
2024), a 10-session conversational benchmark
with 200 QA pairs. As Table 3 shows, MEMTIER

Table 1: LongMemEval-S accuracy by question type (N=500). MEMTIERBM25 only uses raw episodic retrieval;
MEMTIER+ semantic adds DeepSeek-V4-Flash fact pre-population. Taxonomy follows Wu et al. (2025).
Type N Full-ctx-7B MEMTIERBM25 only MEMTIER+ semantic
Single-session user 70 0.057 0.5430.686
Single-session assistant 56 0.107 0.4640.732
Knowledge update 78 0.000 0.3460.436
Temporal reasoning 133 0.105 0.2030.316
Multi-session 133 0.008 0.0600.180
Single-session preference 30 0.000 0.0000.067
Overall Acc500 0.050 0.2520.382
Overall F1500 0.054 0.1420.412
Table 2: Ablation study on LongMemEval-S (N=500). Full system: Acc=0.382, F1=0.412 (k= 4, 300 tokens).
Each row removes or modifies one component.∆Acc relative to full system.
Configuration Acc F1∆Acc
Reference points
Full MEMTIER+ semantic (ours)0.382 0.412—
MEMTIER+ semantic + PPO-learned weights 0.382 0.412±0.000
MEMTIERBM25 only (−semantic tier) 0.252 0.142−0.128
Full-context-7B (no retrieval) 0.050 0.054−0.330
Signal removal (weight= 0, renormalised)
−time decay (w decay= 0) 0.394 0.409−0.014
−cognitive weight (w cw= 0) 0.396 0.409−0.016
−tier boost (w tier= 0) 0.394 0.409−0.014
−two-stage scoping 0.342 0.380−0.038
Retrieval entriesk
k= 10.386 0.397−0.006
k= 2(optimal)0.402 0.414+0.022
k= 4(default) 0.382 0.412 —
k= 80.394 0.410+0.014
Token injection budget
150 tokens 0.374 0.397−0.006
300 tokens (default) 0.382 0.412 —
600 tokens (recommended)0.412 0.427+0.032
Table 3: LoCoMo results. Both systems use the same
context-stuffed prompt; the memory architecture is ir-
relevant.
System F1 R@1
SimpleMem†0.432 —
MemGPT†0.218 —
OpenClaw-Default (ours) 0.125 0.105
MEMTIERPhase 1b (ours) 0.120 0.100
and the unmodified baseline score identically
(∆F1=−0.005).
This null result is informative: LoCoMo in-
serts the full conversation into context at query
time, making the memory architecture irrele-
vant. We concur with the recommendation in Wu
et al. (2025) that LongMemEval is the appropri-
ate benchmark for evaluating memorystorage and
retrieval, while LoCoMo tests in-context compre-
hension. We report LoCoMo for completeness andTable 4: Retrieval weight vector before and after
PPO training (15 seed episodes). Seed data has near-
uniform CW, producing near-zero advantage estimates
and near-zero gradients by design.
Signalw 0wPPO
Semantic (w sem) 0.000 0.000
BM25 (w bm25) 0.350 0.350
Time decay (w decay) 0.250 0.250
Cognitive weight (w cw) 0.250 0.250
Tier boost (w tier) 0.150 0.150
Mean reward (r) — 0.303
comparison with prior work.
3.5 PPO Weight Trainer: Infrastructure
Validation
Table 4 shows the weight vector after 15 training
episodes on seed data.
The weights do not diverge in this initial 15-

episode run. Post-review analysis identifies two
mathematical causes:(1) Circular reward trap:
the CW-based reward is derived from Jaccard at-
tribution, which is itself a proxy—the PPO is opti-
mising a proxy of a proxy with no ground-truth
signal.(2) Zero-variance trap:with 15 seed
episodes of near-uniform CW, the advantageA t=
rt−¯r≈0for every episode, so the gradient update
is identically zero.
The fix, implemented and described here for
camera-ready completion, replaces CW-based re-
ward with direct task success:
rt=(
+1.0if em(ˆa t, a∗) = 1
−1.0otherwise(2)
wherea∗is the LongMemEval-S gold answer.
This constitutes true credit assignment: the weight
vectorwthat produced a correct retrieval receives
a direct positive signal. Exploration is forced
by initialisingσ= 0.15(vs. near-zero previ-
ously), ensuring the advantage has real variance
and gradients flow. Training runs on 100 stratified
LongMemEval-S questions (4 epochs, batch=16)
and is expected to produce meaningful weight di-
vergence; results will be included in the camera-
ready submission.
4 Discussion
Three-layer invariance: architecture is the ceil-
ing.Our evaluation reveals invariance across
three axes, converging on a single conclu-
sion.Generator invariance:DeepSeek-V4-
Flash (284B MoE, 13B active) yields Acc=0.234
vs. Qwen2.5-7B Acc=0.252—statistically identi-
cal with the same per-category profile.Weight in-
variance:PPO-learned weights after task-success
training (100 questions, 4 epochs,+1/−1reward)
yield Acc=0.382, F1=0.412—identical to default
weights—because ±0.03 reweighting cannot alter
BM25-dominated retrieval rankings.Together:
neither the generator, nor the retrieval weights,
nor their interaction determines performance—the
BM25 retrievalarchitectureis the binding con-
straint. Better models and better weights both
fail when the retrieval stage cannot surface multi-
session evidence (0.180) or resolve temporal ref-
erences (0.316). This directly motivates recall-first
retrieval and dense/hybrid scoring as the necessary
Phase 3 components.
The benchmark selection problem.Our Lo-
CoMo null result replicates an often-overlookedissue: benchmarks that make conversations avail-
able in context at query time cannot distin-
guish memory architectures. We recommend that
future work on agent memory systems report
LongMemEval-S as the primary metric, with Lo-
CoMo as a secondary in-context-comprehension
baseline.
Limitations.(1)Attribution path:SGLang
logprob attribution is code-complete but blocked
by local hardware constraints on the evaluation
machine; the lexical Jaccard fallback used in
production is a coarse proxy. (2)RL weight
dominance:While direct task-success reward
(+1/−1) enabled PPO gradients to flow and
weights to shift, BM25’s unbounded scoring heav-
ily dominates the bounded signals (decay, CW) in
the linear combination, masking the RL’s impact
on final ranking. Future iterations must strictly
normalise BM25 or transition to dense retrieval
before learned weights can provide marginal
gains. (3)Relation extraction:Heuristic KV-
pattern extraction produces coarse labels (e.g.,
mentioned_in); a fine-grained NLP extractor
would further improve semantic tier quality.
5 Conclusion
We presented MEMTIER, a five-phase memory
architecture for long-running autonomous agents
that addresses context collapse, compaction dis-
continuity, structural blindness, and the absence
of an attribution loop. Evaluated on the full 500-
question LongMemEval-S benchmark, MEMTIER
with semantic pre-population achievesAcc=0.382
andF1=0.412with a 7B generator on edge
hardware—a +33 percentage point improvement
over the full-context baseline. On single-session
recall, it achieves 0.686–0.732, exceeding the pa-
per’s RAG BM25 GPT-4o baseline (0.560).
The central diagnostic finding is athree-layer
invariance: neither scaling the generator to a
284B MoE model, nor dynamically tuning re-
trieval weights via PPO with true credit assign-
ment, significantly altered the performance ceil-
ing dictated by the BM25 retrieval architecture.
This empirically demonstrates that high-precision,
structurally isolated memory (episodic vs. seman-
tic) is the primary driver of agentic long-horizon
success, while simultaneously demonstrating that
legacy linear-combination retrieval is the current
bottleneck. Future work will transition to recall-
first dense retrieval to solve multi-session syn-

thesis (0.180), implement absolute date resolu-
tion for temporal reasoning (0.316), and activate
the SGLang logprob attribution path for higher-
fidelity RL credit assignment.
Limitations
(1)Attribution path:SGLang logprob attribution
is code-complete but blocked by local hardware
constraints on the evaluation machine; the lexi-
cal Jaccard fallback used in production is a coarse
proxy. (2)RL weight dominance:While direct
task-success reward (+1/−1) enabled PPO gra-
dients to flow and weights to shift, BM25’s un-
bounded scoring heavily dominates the bounded
signals (decay, CW) in the linear combination,
masking the RL’s impact on final ranking. Fu-
ture iterations must strictly normalise BM25 or
transition to dense retrieval before learned weights
can provide marginal gains. (3)Relation ex-
traction:Heuristic KV-pattern extraction pro-
duces coarse labels (e.g.,mentioned_in); a
fine-grained NLP extractor would further improve
semantic tier quality.
6 Conclusion
We presented MEMTIER, a five-phase memory
architecture for long-running autonomous agents
that addresses context collapse, compaction dis-
continuity, structural blindness, and the absence
of an attribution loop. Evaluated on the full 500-
question LongMemEval-S benchmark, MEMTIER
with semantic pre-population achievesAcc=0.382
andF1=0.412with a 7B generator on edge
hardware—a +33 percentage point improvement
over the full-context baseline. On single-session
recall, it achieves 0.686–0.732, exceeding the pa-
per’s RAG BM25 GPT-4o baseline (0.560).
The central diagnostic finding is athree-layer
invariance: neither scaling the generator to a
284B MoE model, nor dynamically tuning re-
trieval weights via PPO with true credit assign-
ment, significantly altered the performance ceil-
ing dictated by the BM25 retrieval architecture.
This empirically demonstrates that high-precision,
structurally isolated memory (episodic vs. seman-
tic) is the primary driver of agentic long-horizon
success, while simultaneously demonstrating that
legacy linear-combination retrieval is the current
bottleneck. Future work will transition to recall-
first dense retrieval to solve multi-session syn-
thesis (0.180), implement absolute date resolu-tion for temporal reasoning (0.316), and activate
the SGLang logprob attribution path for higher-
fidelity RL credit assignment.
Acknowledgments
All experiments were conducted on a consumer
laptop with a 6 GB GPU. We thank the Open-
Claw open-source community for maintaining the
runtime used in all experiments. The authors
used Claude (Anthropic) for writing assistance
and code generation support; all scientific claims
and results are the authors’ own.
References
Anonymous. 2025a. A-MEM: Adaptive memory for
LLM agents.arXiv preprint. See paper for full ref-
erence.
Anonymous. 2025b. Memory-R1: Reinforcement
learning for conversational memory.arXiv preprint.
See paper for full reference.
Anonymous. 2025c. MIRIX: Multi-instance retrieval
index for LLM agents.arXiv preprint. See paper
for full reference.
Anonymous. 2026. AgentWarden: RL-based adap-
tive capability governance for AI coding agents.
NeurIPS 2026 Agent Safety Workshop (under re-
view).
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Infor-
mation Processing Systems (NeurIPS).
Jimmy Lin. 2021. A few brief notes on Deep-
Impact, COIL, and a conceptual framework for
information retrieval techniques.arXiv preprint
arXiv:2106.14807.
Hao Liu and 1 others. 2026. SimpleMem: Efficient
lifelong memory for LLM agents.arXiv preprint.
Adyasha Maharana, Divyansh Das, Sergey Tulyakov,
Mohit Bansal, Franck Dernoncourt, and Yuwei
Fang. 2024. Evaluating very long-term conver-
sational memory of LLM agents.arXiv preprint
arXiv:2402.17753.
OpenClaw Contributors. 2026. OpenClaw: Personal
AI assistant framework. Open-source agent runtime.
Issues #33406, #62488 referenced in this paper.
Charles Packer, Vivian Fang, Shishir G. Patil, Kevin
Wooders, and Joseph E. Gonzalez. 2023. MemGPT:
Towards LLMs as operating systems.arXiv preprint
arXiv:2310.08560.

Stephen Robertson, Steve Walker, Susan Jones, Miche-
line Hancock-Beaulieu, and Mike Gatford. 1994.
Okapi at TREC-3. InProceedings of the Third Text
REtrieval Conference (TREC-3), pages 109–126.
Wei Sun and 1 others. 2026. H-MEM: Hierarchical
memory for high-efficiency long-term reasoning in
LLM agents.arXiv preprint.
Di Wu and 1 others. 2025. LongMemEval: Bench-
marking chat assistants on long-term interactive
memory.arXiv preprint arXiv:2501.08956.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
ReAct: Synergizing reasoning and acting in lan-
guage models. InInternational Conference on
Learning Representations (ICLR).
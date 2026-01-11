# Vibe Coding an LLM-powered Theorem Prover

**Authors**: Zhe Hou

**Published**: 2026-01-08 07:00:24

**PDF URL**: [https://arxiv.org/pdf/2601.04653v1](https://arxiv.org/pdf/2601.04653v1)

## Abstract
We present Isabellm, an LLM-powered theorem prover for Isabelle/HOL that performs fully automatic proof synthesis. Isabellm works with any local LLM on Ollama and APIs such as Gemini CLI, and it is designed to run on consumer grade computers. The system combines a stepwise prover, which uses large language models to propose proof commands validated by Isabelle in a bounded search loop, with a higher-level proof planner that generates structured Isar outlines and attempts to fill and repair remaining gaps. The framework includes beam search for tactics, tactics reranker ML and RL models, premise selection with small transformer models, micro-RAG for Isar proofs built from AFP, and counter-example guided proof repair. All the code is implemented by GPT 4.1 - 5.2, Gemini 3 Pro, and Claude 4.5. Empirically, Isabellm can prove certain lemmas that defeat Isabelle's standard automation, including Sledgehammer, demonstrating the practical value of LLM-guided proof search. At the same time, we find that even state-of-the-art LLMs, such as GPT 5.2 Extended Thinking and Gemini 3 Pro struggle to reliably implement the intended fill-and-repair mechanisms with complex algorithmic designs, highlighting fundamental challenges in LLM code generation and reasoning. The code of Isabellm is available at https://github.com/zhehou/llm-isabelle

## Full Text


<!-- PDF content starts -->

Vibe Coding an LLM-powered Theorem Prover
Zhe Hou
Griffith University
z.hou@griffith.edu.au
Abstract
We present Isabellm, an LLM-powered theorem prover for Isabelle/HOL that per-
forms fully automatic proof synthesis. Isabellm works with any local LLM on
Ollama and APIs such as Gemini CLI, and it is designed to run on consumer
grade computers. The system combines a stepwise prover, which uses large lan-
guage models to propose proof commands validated by Isabelle in a bounded
search loop, with a higher-level proof planner that generates structured Isar out-
lines and attempts to fill and repair remaining gaps. The framework includes
beam search for tactics, tactics reranker ML and RL models, premise selection
with small transformer models, micro-RAG for Isar proofs built from AFP, and
counter-example guided proof repair. All the code is implemented by GPT 4.1 -
5.2, Gemini 3 Pro, and Claude 4.5. Empirically, Isabellm can prove certain lem-
mas that defeat Isabelle’s standard automation, including Sledgehammer, demon-
strating the practical value of LLM-guided proof search. At the same time, we
find that even state-of-the-art LLMs, such as GPT 5.2 Extended Thinking and
Gemini 3 Pro struggle to reliably implement the intended fill-and-repair mech-
anisms with complex algorithmic designs, highlighting fundamental challenges
in LLM code generation and reasoning. The code of Isabellm is available at
https://github.com/zhehou/llm-isabelle.
1 Introduction
Automated reasoning has long been a central goal of computer science, spanning automated the-
orem proving, model checking, symbolic execution, and satisfiability modulo theories. Classical
reasoning engines have achieved remarkable success in well-scoped domains, such as propositional
and first-order logic, where decades of algorithmic advances have led to highly optimised SAT and
SMT solvers. However, as the expressiveness of the underlying logic increases, full automation be-
comes significantly harder. Higher-order logics, rich type systems, and large mathematical libraries
introduce vast search spaces, subtle dependencies, and non-local reasoning patterns that are difficult
to capture with purely symbolic methods. As a result, many practical reasoning systems adopt a
hybrid stance, combining automation with human guidance to manage complexity while preserving
soundness.
Interactive theorem provers such as Isabelle/HOL [7] exemplify this trade-off. They provide ex-
ceptionally strong correctness guarantees: a proof is accepted only if it type-checks against a small
trusted kernel, yielding artefacts far more robust than conventional testing or informal argumenta-
tion. The cost of this rigor is well known. Proof development is labour-intensive, requires spe-
cialised expertise, and often demands careful orchestration of proof methods and auxiliary lemmas,
even in the presence of powerful automation. Isabelle’s ecosystem has substantially mitigated this
burden through tools such as Sledgehammer, which combines premise selection, external automated
provers, and proof reconstruction to discharge many routine goals [2, 1]. Complementary tools such
as Nitpick further assist users by quickly refuting false conjectures or exposing missing assump-
tions. Nevertheless, fully automatic “push-button” proving for arbitrary higher-order goals remains
1arXiv:2601.04653v1  [cs.AI]  8 Jan 2026

out of reach: the search space is combinatorially large, the relevant context may involve thousands
of facts, and the correct sequence of proof methods is highly problem-dependent.
Recent advances in large language models (LLMs) have reopened this challenge from a different an-
gle. Rather than relying solely on symbolic heuristics, LLMs act as powerful conditional generators
that can propose plausible proof steps, tactics, or even full proof scripts based on patterns learned
from large corpora. This capability has given rise to a new generation of neuro-symbolic theorem
provers that tightly couple generative models with formal verification. In these systems, the language
model explores the proof space by proposing candidates, while the proof assistant provides exact,
executable feedback by accepting, rejecting, or partially validating those proposals. Early work
demonstrated the viability of this approach in settings such as Metamath [10], where generative
models were combined with guided search and verification loops. Subsequent systems introduced
more structured search and learning mechanisms, including HyperTree Proof Search (HTPS), which
frames theorem proving as iterative improvement over a dynamically expanding search tree [6].
Within the Isabelle ecosystem, this line of work has been enabled by programmatic interfaces and
benchmarks that support large-scale interaction and learning, most notably LISA and the PISA pro-
tocol for incremental proof execution and data extraction [4]. Building on these foundations, recent
Isabelle-oriented systems have explored whole-proof generation, verifier-guided repair, and deeper
integration with ATP tools, as exemplified by Baldur and Thor [3, 5]. In parallel, work in other proof
assistants, such as LeanDojo for Lean, has highlighted the critical role of retrieval and premise se-
lection when operating over large libraries, as well as the importance of standardised benchmarks
such as miniF2F for comparative evaluation [9, 11].
This project explores the questionCan LLM code a theorem prover powered by LLM?This “cyclic”
question examines the latest LLMs’ capabilities in two aspects: Can they be used to code complex
algorithms like a theorem prover? And can they be used to generate formal proofs as a part of
the theorem prover? The end goal is fully automatic proof synthesis: given a goal statement (and
imports), the system should output a complete, checkable Isabelle proof with no user interaction.
Moreover, we aim to make the frameworklaptop-friendly— all the current code has been tested
on a 2021 Macbook Pro with M1 Pro processor and 32GB RAM. We hope to build an LLM prover
that can run on consumer level computers for an average Isabelle/HOL user, rather than something
that requires expensive GPU servers. Concretely, the repository implements Isabellm, a modular
pipeline with two cooperating layers. The first layer is a stepwise prover that treats proving as se-
quential decision making: from a proof state, it proposes candidate tactics or proof methods, checks
them in Isabelle, and uses search (e.g., beam-style exploration) to reach a terminal solved state.
This layer integrates classic Isabelle automation (including Sledgehammer, SMT/ATP backends,
and counterexample checking) with learned components such as tactic reranking and premise selec-
tion, enabling the prover to scale beyond what raw prompting can hold in context [2]. The second
layer is a planner that attempts to generate structured Isar-style proof outlines and then discharge the
resulting subgoals by calling the stepwise prover. This planning layer is motivated by evidence that
decomposition and sketching can mitigate myopic step-by-step search, especially for longer proofs,
aligning with recent “sketch-first” and recursive proving themes in the literature [5, 8]. The over-
all design is intended to support rigorous evaluation (benchmarking and regression), reproducibility
(scripted proof checking), and continual improvement via logs and learned models.
The remainder of this report describes the system architecture, interfaces, and learning components,
and positions Isabellm relative to prior neuro-symbolic provers. We emphasise the technical choices
needed to make fully automatic proof generation practical in Isabelle: reliable proof-state interac-
tion, fast verification, premise management at library scale, search control under tight time budgets,
and training signals derived from verifier outcomes. We also highlight current limitations, espe-
cially around planning and proof repair, and motivate directions for turning verifier feedback into
systematic improvements rather than ad hoc prompt tuning.
2 An LLM-powered Stepwise Prover
The stepwise prover is the core automated proving engine of the system. Its purpose is to synthesize
a complete Isabelle/HOL proof script for a given goal by iteratively proposing proof commands,
validating them with Isabelle, and performing bounded search over the resulting proof states. It
targets small step proofs, basically playing the same role as Sledgehammer.
2

2.1 Preparation
Proof synthesis is formulated as bounded search in the space of Isabelle proof scripts. A proof script
is represented as an ordered sequence
S=⟨s 0, s1, . . . , s k⟩,
wheres 0is a lemma declaration of the formlemma "<goal>"and eachs ifori≥1is an Isabelle
proof command, typically anapply-style tactic.
At each iteration, the prover maintains a beam of candidate proof states of fixed widthB. Each
beam element is a tuple
(σ, S, h, n),
where:
•Sis the current accepted proof prefix,
•σis a scalar score used for ranking beam entries,
•his a textual hint representing the current proof state (obtained viaprint state),
•nis the number of remaining subgoals reported by Isabelle.
The primary score is the number of remaining subgoalsn. States with fewer subgoals are preferred,
reflecting the heuristic that local progress correlates with global proof completion. Whenncannot
be extracted, a large sentinel value is used.
The search proceeds for at mostDexpansion rounds (maximum depth), or until a proof is completed,
or until a global wall-clock timeout is reached.
Each candidate proof command is validated by constructing a temporary Isabelle theory and invok-
ing Isabelle through a persistent server connection. For a proof prefix
S=⟨s 0, s1, . . . , s k⟩
and a candidate commandc, the prover constructs a theory of the form
theory Scratch imports Main begin
s0
s1
...
sk
c
followed by either:
•print stateandsorryfor intermediate steps, or
• no trailing command for proof finishers.
The theory is executed using Isabelle’suse theoriescommand. A candidate step is considered
successful if Isabelle accepts the theory without error. For intermediate steps, the prover extracts the
printed proof state and parses the number of remaining subgoals.
To avoid redundant exploration, proof states are deduplicated using a fingerprint computed as
fp(h) =SHA1(normalize(h)),
where normalization removes irrelevant whitespace. Only one beam entry per fingerprint is retained.
2.2 Candidate generation via language models
Candidate proof commands are generated by large language models (LLMs) acting as conditional
proposal mechanisms inside the stepwise search loop. The LLMs are not used to generate com-
plete proofs in one shot; instead, they are queried repeatedly to propose small, locally valid proof
commands that are then checked by Isabelle. This design ensures that all correctness guarantees are
enforced by the proof assistant rather than the language model.
3

Invocation model.LLMs are invoked through a unified abstraction layer that supports both local
and remote models (e.g., via HTTP or CLI interfaces). Each invocation specifies:
• a system prompt defining the allowed action grammar,
• a user prompt encoding the current proof context,
• decoding parameters including temperature, maximum tokens, and number of samples.
The prover does not assume any particular model architecture. Instead, it treats the LLM as a
black-box conditional generator that maps a textual proof context to a short list of candidate proof
commands.
Prompting modes.Two distinct prompting modes are used, corresponding to different phases of
proof search:
1.Step mode: generates intermediate proof commands of the form
apply <tactic>.
These commands are intended to reduce or transform the current set of subgoals without
closing the proof.
2.Finish mode: generates proof-closing commands such as
by simp,by auto,by (metis ...),done.
Finish mode is invoked when the prover believes the remaining subgoals may be solvable
in a single step.
Each mode uses a distinct system prompt to tightly constrain the syntactic form of the output.
System prompts.The system prompt explicitly instructs the model to:
• output multiple candidate commands (typically between 3 and 8),
• output one command per line,
• restrict output to a fixed syntactic grammar (applycommands in step mode,by/donein
finish mode),
• avoid explanations, comments, or natural language text.
By enforcing these constraints at the prompt level, the prover reduces the burden on downstream
parsing and significantly lowers the rate of invalid Isabelle calls.
User prompt construction.The user prompt encodes the current proof context as a structured
text block containing:
1.Goal statement:the original lemma to be proved.
2.Accepted proof steps:the prefix of proof commands that have already been validated by
Isabelle.
3.Current proof state:the latestprint stateoutput, showing remaining subgoals.
4.Helpful facts:a curated list of lemma names obtained from premise mining and retrieval.
This information is presented in a consistent, machine-readable format to stabilize the model’s
behavior across different proof states. The proof state is included verbatim to expose the logical
structure of remaining goals, while the helpful facts provide soft guidance without enforcing hard
constraints.
4

Decoding control and exploration.LLM decoding is stochastic and controlled primarily via tem-
perature. To balance exploration and exploitation, the prover adapts the temperature dynamically
based on search stagnation.
Letsdenote the number of consecutive search depths without improvement in the minimum number
of subgoals. The decoding temperatures are adjusted as:
Tstep= min(0.9,0.5 + 0.1s), T finish= min(0.6,0.2 + 0.05s).
At low stagnation, temperatures are kept low to favor deterministic, high-probability commands. As
stagnation increases, temperatures rise to encourage syntactic and semantic diversity in proposed
commands.
Candidate sampling.For each invocation, the model is asked to generate a small batch of candi-
dates in a single call. This is more efficient than repeated single-sample calls and allows the prover
to exploit intra-batch diversity. The number of candidates per call is fixed and independent of beam
width, ensuring predictable computational cost.
Post-processing and sanitisation.The raw output of the language model is subjected to strict
post-processing:
• code blocks, numbering, bullets, and extraneous whitespace are removed;
• only lines beginning with approved prefixes are retained;
• commands exceeding a fixed character limit are discarded;
• duplicate commands are removed while preserving order.
Any output that does not conform to the expected grammar is silently ignored. This conservative
filtering ensures that only syntactically plausible Isabelle commands reach the expensive validation
stage.
Integration with reranking.If a learned reranker is available, each candidate command is im-
mediately featurised and scored before Isabelle evaluation. The reranker score is used to reorder
candidates, so that more promising commands are checked first. This interaction allows learning to
influence the search without bypassing symbolic validation.
Heuristic variant injection.When stagnation exceeds a threshold, the prover augments LLM-
generated candidates with heuristic templates such as:
apply (induction xs),apply (cases x),apply (rule some lemma).
These templates are instantiated using variable names extracted from the current proof state. This
mechanism acts as a lightweight fallback that injects domain-relevant structure even when the LLM
fails to propose it explicitly.
Design rationale.The candidate generation mechanism deliberately restricts the expressive power
of the language model. By limiting outputs to short, grammar-constrained commands and validating
every proposal symbolically, the system transforms the LLM from an unreliable proof generator into
a probabilistic proposal distribution over local proof actions. This design keeps the overall prover
sound, debuggable, and amenable to learning-based improvement.
2.3 Beam expansion and scoring
For each beam state, the prover generates a finite set of candidate next steps. Each candidate is
independently validated with Isabelle. Successful candidates yield new beam entries with updated
proof prefixes and proof states.
LetCbe the multiset of all successful expansions from the current beam. The next beam is con-
structed by sortingClexicographically by:
(n,|S|),
5

wherenis the number of remaining subgoals and|S|is the length of the proof script. The bestB
entries with distinct fingerprints are retained.
This yields a best-first search over proof states, constrained by beam width and depth.
Monte Carlo tree search is tested by not used because it is much slower than beam search, and as
alluded above, the goal of the stepwise prover is to prove small step goals rather than complex goals.
2.4 Learning-based tactic reranking
The stepwise prover employs a learning-based reranker to bias the ordering of candidate proof com-
mands proposed by the language model. The reranker operates as a lightweight scoring function that
predicts the likelihood that a candidate command will make progress when applied to the current
proof state. Importantly, the reranker doesnotreplace symbolic validation: all candidates are still
checked by Isabelle. Its role is purely to reduce the effective branching factor of the search.
Reranking interface.At runtime, the reranker is exposed as a function
fθ:Rd→[0,1],
wheredis the feature dimension andf θ(x)estimates the probability that a candidate step will
succeed. Higher scores indicate higher priority during beam expansion. The reranker score is incor-
porated into candidate ordering by adjusting the heuristic score:
score(c) = heuristic(c)−λ·f θ(xc),
whereλ >0is a fixed weight (configured via environment variables) andx cis the feature vector
extracted for candidatec.
Feature representation.Each candidate step is represented by a fixed-length numeric feature
vector composed of four groups:
1.Search context features:
(depth, n sub, telapsed,cache hit),
where depth is the current search depth,n subis the number of remaining subgoals (if
known), and cache hit indicates reuse of cached Isabelle results.
2.Goal and state flags:Binary indicators extracted from the goal and proof state, including:
{islisty,is natty,is sety,has quantifier,is boolean}.
These capture coarse structural properties of the goal.
3.Tactic prefix encoding:A one-hot encoding over a fixed vocabulary of tactic prefixes (e.g.,
apply simp,apply auto,apply (induction),apply (cases),apply
(rule), etc.). This represents the syntactic class of the proposed command.
4.Premise interaction features:Numeric summaries derived from premise selection
(described later), including cosine similarity statistics and overlap between candidate-
referenced lemmas and retrieved premises.
The final feature vector is padded or truncated to a fixed dimension expected by the trained model.
This makes the reranker robust to incremental feature extensions.
Training data.Training data for the reranker is collected automatically during proof search. Every
attempted candidate command generates a labeled example
(xc, yc),
wherex cis the feature vector described above and
yc=1if Isabelle accepts the step,
0otherwise.
Both successful and failed attempts are logged. This produces a highly imbalanced but extremely
large dataset reflecting real search behavior rather than curated proofs.
6

Supervised learning.The system supports multiple supervised learning backends:
•Logistic regression:a linear baseline for fast iteration and interpretability.
•Gradient-boosted trees (XGBoost):used to model non-linear interactions between fea-
tures.
For probabilistic classifiers, the predicted probabilityP(y= 1|x)is used directly asf θ(x). For
regressors, outputs are normalized to[0,1].
Offline reinforcement learning.Beyond supervised learning, the system supports offline rein-
forcement learning from logged trajectories. In this formulation, proof search is treated as a Markov
decision process:
(st, at, rt, st+1),
wheres tis the proof state,a tis a candidate tactic, and the rewardr tis derived from subgoal
reduction or proof completion.
Two algorithms are implemented:
1.Advantage-Weighted Regression (A WR):candidates are weighted by estimated advan-
tage, favoring steps that lead to faster subgoal reduction.
2.Fitted Q-learning (DQN-style):learns an action-value functionQ(s, a)from logged tran-
sitions and uses it as a reranking signal.
The learned models are exported either as TorchScript modules or joblib artifacts and loaded dy-
namically at runtime.
Design rationale.The reranker is intentionally shallow and fast. Its purpose is not to prove theo-
rems independently, but to encode empirical regularities of what tends to work in Isabelle, allowing
the symbolic prover to focus its computational budget on promising branches.
2.5 Premise selection with neural encoders
Premise selection is used to supply the prover and the language model with a small, relevant subset
of lemmas from the surrounding theory and library. Rather than relying solely on Isabelle’s internal
heuristics, the system implements a retrieval-based premise selector with optional neural encoders.
Premise index.All candidate premises are stored in an in-memory index as pairs
(ℓi, ti),
whereℓ iis a lemma identifier andt iis its textual representation. Each premise may also carry
metadata such as source file and local context.
Two-stage retrieval.Premise selection proceeds in two stages:
1.Select stage:a fast, recall-oriented retrieval that produces a pool ofK select candidate
premises.
2.Rerank stage:an optional precision-oriented rescoring of the topK rerank premises.
Select stage encoders.The select stage supports three backends:
•TF–IDF cosine similarity:when scikit-learn is available, premises are vectorized using
TF–IDF and scored by cosine similarity.
•Token overlap (fallback):when no external libraries are available, Jaccard overlap over
token sets is used.
7

•Neural bi-encoder:when a trained encoder is present, both goals and premises are em-
bedded into a shared vector space using a sentence-transformer model. Cosine similarity is
computed as
sim(g, p) =⟨e(g), e(p)⟩
∥e(g)∥∥e(p)∥.
All embeddings are computed once during index finalization and cached in memory.
Rerank stage.The rerank stage optionally applies a cross-encoder that scores pairs(g, p)jointly.
Given a batch of premise candidates{p i}and a goalg, the cross-encoder computes scores
ri= CE(g, p i),
which are used to reorder premises. If no reranker is available, the select score is reused.
Training premise encoders.Training data for premise selection is extracted from successful proof
attempts. For a given goalg:
• Positive premises are those explicitly referenced in successful proof steps.
• Negative premises are sampled from the retrieval pool but not used in the proof.
The bi-encoder is trained using contrastive learning (e.g., Multiple Negatives Ranking Loss), en-
couraging
sim(g, p+)>sim(g, p−)
for positive premisep+and negative premisep−. The cross-encoder is trained using supervised
regression or classification over(g, p)pairs.
Integration with proof search.Selected premises are used in two ways:
1. They are injected verbatim into the language model prompt as contextual hints.
2. Summary statistics (top similarity, mean similarity, overlap with candidate-referenced lem-
mas) are appended to the reranker feature vector.
This tight coupling allows premise selection and tactic reranking to reinforce each other without
hard constraints.
Design rationale.Premise selection is treated as a soft guidance mechanism rather than a hard
filter. By exposing both the language model and the reranker to retrieved premises, the system
benefits from retrieval while remaining robust to retrieval errors, a critical property in large and
evolving Isabelle libraries.
2.6 Post-processing
Before executing expensive Isabelle checks, candidate steps may be pruned using lightweight refu-
tation tools. Quickcheck and Nitpick are optionally invoked to detect counterexamples. If either
tool produces a counterexample, the candidate is discarded without further evaluation.
This pruning is applied conservatively, primarily for goals involving Boolean structure or quantifiers,
and at configurable intervals to control overhead.
To reduce repeated Isabelle invocations, the prover employs two levels of caching:
• a per-run cache keyed by(S, c),
• a global bounded cache shared across runs.
Cached entries store success flags, subgoal counts, proof state hints, and timing information. Cache
hits are logged and later used as features during reranker training.
After a successful proof is found, the prover attempts to simplify it. The minimisation procedure
applies the following transformations greedily under a small timeout:
8

1. collapse the proof to a single-line proof if possible,
2. remove unused facts fromsimpormetiscalls,
3. delete redundant intermediate steps,
4. reattempt single-line proofs.
Additionally, the prover attempts to convert unstructuredapply-style proofs into structured Isar
proofs using simple skeleton templates, improving readability without sacrificing automation.
3 An LLM-powered Proof Planner
This section describes the proof planner, which operates at a higher level of abstraction than the
stepwise prover. Whereas the stepwise prover performs bounded search over individual proof com-
mands, the planner reasons overstructured Isar proofsthat explicitly decompose a complex goal
into intermediate claims and subproofs. The planner aims to synthesize a proof outline that captures
the global structure of the argument and then to systematically eliminate any remaining gaps until a
fully verified proof is obtained.
Formally, given a goal formulaG, the planner seeks a proof script
P=⟨s 0, s1, . . . , s m⟩
such thats 0is a lemma declaration forG,Pis a well-formed Isar proof, and Isabelle verifiesP
without unresolved gaps. Unlike the stepwise prover, intermediate scripts may contain placeholders
(sorry), which are treated as explicit unknowns to be filled by subsequent phases.
3.1 High-level planning model
The planner operates in one of two modes:
•Outline mode, which returns a structured Isar proof skeleton that may contain gaps.
•Auto mode, which iteratively applies outline generation, gap filling, repair, and regenera-
tion until either a verified proof is produced or a global budget is exhausted.
Throughout planning, Isabelle is accessed through a persistent server session. LetIdenote this
session. All outline checking, filling, and verification steps are executed againstI, amortizing ini-
tialization costs and enabling repeated theory construction under tight budgets.
Diversified outline sampling.Outline generation is performed by querying a language model to
produce candidate Isar skeletons. LetT={t 1, . . . , t r}be a fixed set of sampling temperatures. For
eacht i∈ T, the planner samples up tokcandidate outlines, yielding a multiset
S=r[
i=1Sti,
whereS tiare the samples at temperaturet i. This mechanism approximates sampling from a mixture
distribution over proof structures, trading determinism for diversity.
Prompt-level conditioning.Each sampling call conditions on:
1. the target goalG,
2. an optional set of recommended lemmasH={h 1, . . . , h m},
3. a system constraint requiring a single Isar proof block.
Hints are injectedbeforegeneration, allowing them to influence the global structure (e.g. choice of
induction variable) rather than only local steps.
9

Normalization into canonical skeletons.Raw model outputs are normalized into canonical skele-
tons
S= (header,body,holes),
where:
• the header is forced to matchG,
• the body is a syntactically complete Isar proof,
• all missing justifications are replaced bysorry,
• inline proofs (by ...) may be rewritten into holes when outline enforcement is enabled.
This normalization guarantees that every candidate outline admits a well-defined set of gaps and can
be checked by Isabelle without semantic ambiguity.
Hole extraction.Letholes(S) ={h 1, . . . , h ℓ}denote the set of maximalsorryspans inS.
Each holeh iis treated as an independent subproblem during filling and repair.
The planner exposes the following parameters for experimental control:
• outline diversity parameters(k,T),
• enforcement of explicit holes,
• use of structural templates,
• hint sources and retrieval limits,
• scoring weights(α, β, γ),
• stage caps(c 1, c2)and global time budgets.
Together, these parameters define a search space over proof structures and repair strategies, enabling
systematic evaluation of planning effectiveness.
3.2 Micro-RAG: lightweight retrieval-augmented guidance for planning
The proof planner incorporates a deliberately lightweight form of retrieval-augmented generation
(RAG), referred to asmicro-RAG, whose purpose is to bias outline generation and repair without
introducing heavyweight neural retrieval or large external indices. The design goal is to provide
structural and semantic hintsthat are cheap to compute, stable across runs, and suitable for execution
on a laptop-scale environment.
Unlike premise selection in the stepwise prover, which is tightly coupled to tactic-level decision
making, micro-RAG operates at the level ofproof planning. Its outputs influence which proof
structures are proposed and how gaps are repaired, but never act as hard constraints.
3.2.1 Hint sources and representation
Micro-RAG aggregates hints from two independent sources:
H=H ctx∪H lex,
where each hinth∈His a symbolic identifier (typically a lemma name or theorem constant).
Context-derived hints.The context-derived hint setH ctxis extracted directly from Isabelle. Given
a goalG, the planner constructs a minimal theory that opens the goal and requests a proof state
printout. From this state block, it extracts:
• locally bound facts,
• assumptions,
• previously introduced lemmas available in the current context.
10

LetCtxFacts(G)denote the multiset of symbols appearing in this state. The planner applies syntac-
tic normalization (removing duplicates, stripping qualifiers, filtering trivial facts) and truncates the
list to a fixed budgetk ctx. The resulting set is
Hctx:=TopK(CtxFacts(G), k ctx).
These hints reflect what Isabelle itself considers locally relevant, making them particularly effective
for guiding structural proof choices such as induction variables or case distinctions.
Lexicon-derived hints.The lexicon-derived hint setH lexis obtained from a precomputedhint
lexicon, stored as a JSON map:
L:token7→ {(h, w h)},
where each entry associates a token (e.g. function name, constructor, type name) with a weighted
list of lemma identifiers.
At runtime, the planner tokenizes the goalGinto a multisetTok(G). For each tokent∈Tok(G)
present inL, it retrieves the associated lemma list and accumulates scores:
score(h) =X
t∈Tok(G)wt,h.
The lexicon-derived hint set is then
Hlex:=TopK 
{(h,score(h))}, k lex
,
wherek lexcorresponds tohintlex top.
The lexicon itself is mined offline from large Isabelle corpora (e.g. AFP) by correlating goal tokens
with lemmas appearing in successful Isar proofs. Importantly, this process is entirely symbolic and
does not require neural encoders.
3.2.2 Hint aggregation and normalization
The combined hint set is formed as:
H:=Dedup 
Hctx∪H lex
,
followed by truncation to a global capk hint. The planner preserves relative ordering by source
priority, typically favoring context-derived hints over lexicon hints.
No attempt is made to ensure completeness or optimality ofH. Micro-RAG is explicitly heuristic
and biased toward stability and low variance rather than maximal recall.
3.2.3 Integration into outline generation
During outline generation, the hint setHis injected at theprompt level. Specifically, the system
prompt remains unchanged, but the user prompt includes an explicit preference clause:
HINTS: Prefer usingh 1, . . . , h mif applicable.
This instruction is advisory rather than mandatory. The language model is free to ignore hints, but
empirical behavior shows that such soft conditioning often affects high-level proof choices, such as
selecting an induction principle or reusing a characteristic lemma.
Hints are injectedbeforeoutline sampling, allowing them to influence global structure rather than
being retrofitted during repair.
3.2.4 Integration into repair and regeneration
Micro-RAG is also used during repair and whole-proof regeneration. In these phases, hints serve
two roles:
1.Repair conditioning:when regenerating a block or subproof, the hint set is included
alongside effective goals and error diagnostics, biasing the LLM toward known useful lem-
mas.
11

2.Scoring signal:usage of hint lemmas inside an outline contributes positively to the out-
line’s composite score via thehint bonusterm.
Formally, letUse(S, H)be the number of distinct hints fromHthat appear syntactically in skeleton
S. The bonus term is computed as:
hintbonus(S) = min 
Use(S, H), k hint
,
ensuring bounded influence.
3.2.5 Design rationale and limitations
Micro-RAG deliberately avoids heavyweight retrieval mechanisms such as dense embedding indices
or cross-encoders. This choice reflects several considerations:
• planning decisions are coarse-grained and benefit more from symbolic cues than from fine-
grained semantic similarity;
• planner prompts are short and benefit from stable, low-noise hints;
• laptop-scale execution precludes large neural indices.
As a result, micro-RAG should be viewed as astructural biasing mechanismrather than a replace-
ment for premise selection in tactic-level proving. Its strength lies in nudging the planner toward
familiar proof schemas while leaving all correctness checking to Isabelle.
3.3 Gap filling and counterexample-guided proof repair
This subsection specifies the planner’s fill+repair mechanism in the same verifier-in-the-loop, algo-
rithmic style as the stepwise prover. The key design goal is to treat a partially correct Isar outline
as astructured search objectand to transform each remainingsorryinto a sequence of bounded
synthesis problems under Isabelle validation.
3.3.1 Objects and invariants
LetPdenote the current Isar script (a sequence of lines). Aholeis a maximal span
h= [a, b)⊆ {0, . . . ,|P|}
that corresponds to asorrytoken occurrence in the text. LetHoles(P) ={h 1, . . . , h ℓ}.
The planner maintains the following invariants:
1.Verifier gate:a change iscommittedonly if the full script verifies in Isabelle.
2.Local progress allowance:a change that does not verify may still be retained aspartial
progress, but it must be immediately normalized by opening new explicit holes so that
subsequent iterations can target them.
3.Stable hole identity:each hole has a stable identifier used to track repair stage and attempts
across textual edits.
Stable hole identifier.Since line indices drift after edits, each hole is keyed by a windowed fin-
gerprint:
hid(P, h) = SHA1 
P[max(0, a−w) : min(|P|, b+w)]
[: 16],
wherewis a fixed character window. This identifier is used by the planner’s state
Π = (stage[hid],tries[(hid, s)],focus),
wherestage[hid]∈ {0,1,2}is the current repair stage for that hole,triescounts attempts per stage,
andfocusoptionally pins the traversal to a specific hole after partial progress.
12

3.3.2 Effective goal extraction
For each holehin scriptP, the planner computes aneffective goalG hby querying Isabelle for the
proof state immediately precedingh.
Concretely, letP ≺hbe the prefix ofPending at the location just beforesorryinh. The planner
constructs a temporary theory that executesP ≺hand ends withprint state(or an equivalent
state-printing command) so that Isabelle emits a textual proof state block. LetState(P, h)denote
that block. The effective goal is computed as a text extraction function:
Gh:=EffGoal 
State(P, h), G, P, h
,
which prefers (i) the currently active subgoal statement and (ii) a consistent fallback to the original
goalGwhen state parsing fails.
This transforms global filling into localized subproblems:
G⇝{G h1, . . . , G hℓ}.
3.3.3 Fill: calling the stepwise prover as a local synthesizer
Given a holehand its effective goalG h, the planner first attempts afillstep by calling the stepwise
prover with a small budget and shallow search parameters:
Solve(G h;budget,depth,beam,facts, . . .)⇒a sequence of commandsC.
The returned commands are post-processed to extract:
• apply-steps:A= [c∈C|cbegins withapply],
• a finisher:f∈Csuch thatfbegins withbyor equalsdone.
The planner then attempts to splice these commands into the hole region. Two cases are distin-
guished:
Case 1: finisher available.Iffis present, the hole is replaced by an indented block consisting of
Afollowed byf. LetReplace(P, h,∆)denote the script obtained by replacing spanhwith text∆.
The fill candidate is
P′:=Replace(P, h,Indent(A+ + [f])).
If Isabelle verifiesP′, the fill commits.
Case 2: apply-only progress.If only apply-steps are available, the plannerneverdeclares success
immediately. Instead it treats apply-only output as a transformation that may reduce subgoals but
does not close the local proof obligation. Moreover, apply-steps are not syntactically admissible
everywhere in Isar (notably underhave/show/obtainheadings in “prove” mode). Therefore the
planner performs a structural placement check:
1. if the hole is in a context whereapplyis legal, it insertsAand then forces a new explicit
gap viasorry;
2. otherwise, it wrapsAinto a tiny subproof (e.g.proof -. . .sorry qed) so that the script
remains well-formed.
In both cases, the result is treated aspartial progressand fed to the next normalization step below.
3.3.4 Partial progress normalization: opening minimal sorries
A central engineering choice is that the planner never continues search on scripts that are “half-
edited” without explicit holes. Whenever a fill or repair attempt changes the script but fails global
verification, the planner calls a normalization operator:
(P⋆,opened) :=OpenMinimalSorries(P).
Intuitively,OpenMinimalSorriesscans for failing tactic lines (typicallyapplysequences orby
commands) and replaces them withsorryin a manner that preserves the Isar structure:
13

• if anapply-sequence is embedded under ahave/showhead, it is converted into a local
proof -. . .sorry qedblock rather than leaving rawapplylines;
• otherwise, the failing line is replaced by asorryline at matching indentation.
If any hole is opened, the planner updates focus to the nearest newly created hole around the old
location (to ensure continuity of effort) and resumes from that hole.
3.3.5 Repair: CEGIS over structured block edits
If fill fails (or is skipped due to current stage), the planner invokestry cegis repairs. This
procedure implements a bounded, CEGIS-style loop overblock-level editsrather than single tactics.
Repair search state.LetPbe the current script andhthe target hole. The repair procedure
maintains:
•Pt: current script candidate (initiallyP),
•S0:=State(P, h): initial state block,
• a time budget functionleft()derived fromrepair budget s,
• aprior failure storeMmapping block type→a bounded list of previously tried block
candidates, used as a ban list for LLM prompting.
Retargeting to earliest failure anchor.Although repair is parameterized by the hole span, the
actual syntactic error often occurs earlier than thesorry. Therefore repair first computes an anchor
line:
ℓanchor :=EarliestFailureLine(P, h),
and sets the repair focus line to a clamped index nearℓ anchor . This converts repair from “edit exactly
at the hole” to “edit the earliest block that causes the hole to become unprovable.”
Block types and stages.Repair considers three block types, aligned with stage escalation:
1.Stage 1 (have/show micro-block):find an enclosinghave/show/obtainblock around
the focus line and attempt to regenerate that micro-block.
2.Stage 2a (case block):if acases/casestructure encloses the focus line, attempt to
regenerate the case block.
3.Stage 2b (subproof):if aproof. . .qedsubproof encloses the focus line, attempt to
regenerate the entire subproof.
Whole-proof regeneration is handled by the outer driver as stage 3.
Counterexample hints.Before proposing a new block, repair extractscounterexample-oriented
hints:
CE :=CounterexampleHints(I, S 0),
which may include variable bindings and definitional expansions suggested by Quickcheck/Nitpick-
style diagnostics. These hints are treated as soft constraints in the LLM prompt.
LLM proposal and strict deduplication.For a block regionB(a contiguous span of lines[i, j)),
repair builds a prompt containing:
• the effective goalG h,
• normalized Isabelle error messages observed on the current script,
• the extracted proof context near the block,
• the current block textB,
• counterexample hintsCE,
• aprior failed block listfromM.
14

The LLM outputs a candidate replacement block bB. Repair computes a fingerprint
fp(bB) := SHA1(normalize( bB)),
and rejects bBimmediately iffp( bB)matches any fingerprint already stored inMfor that block type.
This “strict deduplication” prevents pathological repetition under stochastic decoding.
Wrapper stripping and block canonicalization.Since LLMs frequently include extraneous
wrappers (e.g. emitting a full lemma instead of a sub-block), repair applies type-specific stripping
operators:
bB′:=StripToType( bB,type),
so that the replacement has the same syntactic granularity as the original block.
Repair verification gate.A candidate script is formed by substitution
P′:=ReplaceLines(P,[i, j), bB′).
Repair then performs an Isabelle check onP′(as a full theory). If verification succeeds,P′is
returned as a verified repair. If verification fails, repair records bB′intoM(bounded by a maximum
list length) and continues, providedleft()is positive.
If a candidate changes the script but fails verification, repair may still return it aspartial progress
with a diagnostic tag (e.g.stage=1 partial-progress), allowing the outer loop to apply
OpenMinimalSorriesand continue filling with explicit holes.
3.3.6 Outer control: stage caps, focus management, and escalation
The driver maintains per-hole attempt counts for each start stage. Lettries[(hid, s)]be the number
of verified-gate failures (including no-change outcomes) at stagesfor holehid. Two capsc 1andc 2
are enforced:
c1= 2, c 2= 3.
Escalation proceeds as:
stage[hid]←2ifstage = 1andtries[(hid,1)]≥c 1,
2and trigger whole regeneration ifstage = 2andtries[(hid,2)]≥c 2.
Focus policy.Whenever partial progress opens new holes, the driver selects a nearest-hole heuris-
tic around the previous location and setsfocusto that hole’s fingerprint. This yields a coherent local
search process rather than bouncing between unrelated holes.
Robustness to Isabelle instability.All Isabelle calls in fill and repair are wrapped with bounded
timeouts. Exceptions (timeouts, value errors from malformed intermediate theories, or unexpected
failures) trigger a controlled Isabelle restart and the corresponding attempt is counted as failed,
without terminating the overall planning loop.
3.3.7 Algorithmic summary
The fill+repair loop can be summarized as a bounded search procedure over scripts:
State:current scriptP, hole setHoles(P), planner stateΠ.
Actions:FILLHOLEvia stepwise prover; REPAIRBLOCKvia LLM; OPENMINI-
MALSORRIES; REGENERATEWHOLE.
Transition:apply an action to obtainP′; if VERIFY(P′)succeeds, commit; oth-
erwise normalize by opening holes and updateΠ.
Objective:reach a scriptPsuch that VERIFY(P) =true andsorry/∈Punder
a global time budget.
This design mirrors the stepwise prover’s philosophy: the LLM is used to propose candidate edits,
while Isabelle remains the sole judge of correctness. The main difference is thegranularityof
proposals: the stepwise prover proposes single commands; the planner’s repair proposes structured
Isar blocks, enabling larger “jumps” in the proof space when local command-level search stalls.
15

4 Data Processing and Generation
Both the stepwise prover and the proof planner are instrumented to produce rich execution traces that
serve simultaneously as debugging artifacts, benchmark results, and training data for learning-based
components. This section describes the data model, logging pipeline, dataset generation procedures,
and benchmarking infrastructure implemented across theprover/andplanner/modules.
4.1 Design principles
The data pipeline is guided by four design principles:
1.Verifier-grounded data: all logged signals are derived from concrete Isabelle executions
rather than model-internal confidence.
2.Fine-grained supervision: every attempted action (tactic, block edit, outline) is logged,
not only successful proofs.
3.Phase separation: data from tactic-level proving, planning, filling, and repair are distin-
guishable but share a common schema.
4.Reproducibility: logs are append-only, version-agnostic, and sufficient to replay or re-
evaluate decisions offline.
As a result, the system naturally produces datasets suitable for supervised learning, offline reinforce-
ment learning, ablation studies, and longitudinal benchmarking.
4.2 Run-level logging
Each invocation of the prover or planner produces arun record, corresponding to a single goal
attempt. LetGbe a goal statement. A run recordR(G)contains:
• goal identifier and textual goal,
• prover/planner mode and configuration (beam size, depth, temperatures, budgets),
• model identifiers for LLM, reranker, and premise selector,
• wall-clock runtime and timeout status,
• success flag and final proof text if successful,
• summary statistics (depth reached, number of expansions, number of repairs, number of
regenerations).
Run records are serialized as one JSON object per line (JSONL), enabling streaming writes and
post-hoc aggregation without loading entire logs into memory.
4.3 Attempt-level logging
More granular supervision is captured at theattempt level. An attempt corresponds to a single
proposed action evaluated by Isabelle. The precise meaning of an attempt depends on context:
• in the stepwise prover: applying a single candidate command to a proof state;
• in the planner fill phase: inserting a candidate proof fragment into a hole;
• in the planner repair phase: replacing a structured block or subproof.
Each attempt record includes:
• the goalGand current script prefix or outline,
• the proposed action text,
• attempt type (step, finisher, fill, repair, regeneration),
• success flag under Isabelle verification,
16

• number of subgoals before and after the attempt (when applicable),
• elapsed Isabelle execution time,
• cache hit indicators,
• current search depth or repair stage,
• auxiliary features (retrieval scores, reranker scores, hint usage).
Formally, each attempt yields a labeled tuple
(x, y,∆),
wherexis the feature representation of the attempt,y∈ {0,1}is the verifier outcome, and∆
captures state change metrics such as subgoal reduction.
These attempt logs are the primary source of training data for tactic rerankers, premise encoders,
and repair heuristics.
4.4 Dataset generation for continual learning
The logged attempts are post-processed into structured datasets tailored to different learning tasks.
Reranker datasets.For tactic and block reranking, each attempt produces a supervised example
(x, y), wherexis the numeric feature vector andyindicates verifier acceptance. Depending on the
training objective, labels may be transformed into:
• binary classification targets,
• regression targets approximatingQ-values,
• advantage-weighted targets for offline policy learning.
Episodes corresponding to single proof attempts can also be reconstructed, yielding trajectories
(s0, a0, r0, s1), . . . ,(s T, aT, rT, sT+1),
where rewardsr tare derived from subgoal reduction or proof completion.
Premise selection datasets.For premise selection, successful attempts provide weak supervision:
any lemma explicitly referenced in a successful step is treated as a positive premise for the corre-
sponding goal or state. Negative premises are sampled from the retrieved pool but unused in the
proof. This yields contrastive pairs(g, p+)and(g, p−)for training bi-encoders and cross-encoders.
Planner repair datasets.Planner logs additionally encode:
• block types (have/show, case, subproof),
• effective goals for holes,
• counterexample diagnostics,
• failure histories and banned candidates.
These signals support future work on learning block-level repair policies or regeneration strategies.
By treating proof search as a data-generating process rather than a black-box solver, the system en-
ables continual improvement through learning while preserving the soundness guarantees of formal
verification.
5 Integration with the Isabelle/jEdit UI
To support interactive use, the system includes a lightweight integration with the Isabelle/jEdit user
interface. This integration allows users to invoke the stepwise prover and the proof planner directly
from the editor, at the location of the current lemma, and to insert the generated proof text back into
the buffer. The design deliberately keeps the UI layer thin: all reasoning and verification happen in
the existing prover and planner components, while the UI acts purely as a convenience interface.
17

5.1 Overall architecture
The integration consists of two parts: a local HTTP server and a set of jEdit macros. The HTTP
server runs as a long-lived process and exposes a small API that wraps the prover and planner entry
points. It also maintains a persistent Isabelle server session, so that repeated UI invocations reuse
the same HOL session and avoid repeated startup overhead. From the UI’s perspective, this server
is the single point of contact for all proof automation requests.
On the editor side, Isabelle/jEdit is extended with BeanShell macros. These macros extract the goal
of the lemma near the caret, send it to the local server, and paste the returned proof text back into
the editor. Communication is local-only and uses simple JSON payloads over HTTP.
5.2 Server-side functionality
The UI server is implemented using a small FastAPI application. On startup, it initializes a persistent
Isabelle server and opens a HOL session that is reused for all subsequent requests. The server
exposes two main endpoints.
The first endpoint provides access to the stepwise prover. It accepts a goal string together with
basic search parameters such as time budget and model selection, invokes the prover, and returns
the resulting sequence of proof commands. Before returning the result, the server filters the output
to retain only lines that are directly usable inside an Isabelle proof, typically commands beginning
withapplyorby. This ensures that pasted output is immediately syntactically valid.
The second endpoint provides access to the proof planner. Depending on the request mode, it either
returns a proof outline or attempts a full plan-and-fill run. The endpoint forwards planner-specific
parameters such as outline diversity, repair options, and micro-RAG configuration, while falling
back to server-side defaults when parameters are omitted. The response contains the current best
proof script produced by the planner, whether or not all gaps have been filled.
The server is designed to be robust against Isabelle instability. Timeouts and unexpected failures are
handled by restarting the Isabelle session when necessary, without crashing the server process itself.
5.3 Editor macros
The jEdit macros implement a uniform editor-side workflow. When a macro is invoked, it scans
upward from the caret to locate the nearestlemmaortheoremdeclaration and extracts the quoted
goal string. It then determines whether the caret lies inside the corresponding proof block or outside
it. This decision controls where the generated text will be inserted: either at the caret position or
immediately after the lemma header.
Three macros are provided. One macro invokes the stepwise prover and inserts the suggested proof
commands. A second macro invokes the planner in outline-only mode and inserts a structured proof
skeleton. The third macro invokes the planner in full plan-and-fill mode and inserts the resulting
script, which may already be complete or may still contain explicit gaps.
Before insertion, planner outputs are lightly normalized on the editor side. In particular, redundant
lemma headers are stripped, and only theproof–qedblock is inserted. Indentation is adjusted to
match the surrounding context in the editor buffer.
5.4 Intended usage
The UI integration is intended as an interactive aid rather than a replacement for batch evaluation.
A typical workflow is to write or import a lemma statement in Isabelle/jEdit, place the caret inside
the lemma, and invoke one of the macros. The prover or planner then attempts to construct a proof,
and the resulting text is inserted directly into the editor, where it can be inspected, edited, or re-run.
By keeping the UI layer minimal and delegating all substantive reasoning to the underlying sys-
tem, the integration provides a smooth interactive experience without compromising soundness or
duplicating logic already present in the prover and planner.
18

Figure 1: An illustration of the integration with Isabelle/jEdit.
Figure 2: An example of a proof inserted by the LLM prover.
As as example, assuming that the user has started the local server as described above, the user may
move the cursor to a proof state and click LLM→PlanFill from the Macro menu, as shown in
Figure 1. When the computation is complete, the LLM prover will insert a proof, if it can find a
valid one, at the correct position of the document, as shown in Figure 2.
6 Conclusion
This work presented an LLM-powered theorem proving system for Isabelle/HOL that combines
verifier-guided stepwise search with a higher-level proof planner based on Isar outlines, gap filling,
and repair. The system is designed to operate fully automatically: given a goal, it attempts to syn-
thesize a complete, kernel-checked proof without user interaction. Throughout the design, Isabelle
19

remains the sole authority on correctness, while large language models are used strictly as proposal
mechanisms whose outputs are filtered, validated, and repaired under tight syntactic and semantic
constraints.
A key contribution of the project is demonstrating that LLM-driven proof search can already sur-
pass classical automation in specific regimes. In particular, the stepwise prover is able to solve
certain higher-order goals that are beyond the reach of Sledgehammer, even when the latter is given
generous timeouts. Figure 2 illustrates a representative example: a lemma that cannot be proved
by Sledgehammer but can be proved by the LLM prover through a sequence of semantically in-
formed proof steps. This shows that LLM-based guidance is not merely a convenience layer over
existing automation, but can explore proof strategies that differ qualitatively from those encoded in
traditional heuristics.
At the same time, the project exposes clear and fundamental limitations. Most notably, the fill-and-
repair component of the proof planner remains far less effective than originally intended. Despite
careful engineering of effective-goal extraction, staged repair, CEGIS-style loops, and whole-proof
regeneration, even state-of-the-art LLMs such as GPT 5.2 Extended Thinking and Gemini 3 Pro
struggle to reliably implement the design the fill and repair features described in Section 3.3. In
practice, these models often fail to generate code with such complex algorithmic design, especially
when the code base reaches more than 20 files and the new code needs to respect the existing APIs.
As a result, the planner’s repair stages rarely succeed beyond trivial cases, and most successful
proofs are obtained either directly from the initial outline or from the stepwise prover operating
without planner assistance.
These limitations are not merely implementation artefacts but reflect deeper challenges. Repairing
an Isar proof requires simultaneously reasoning about global structure, local proof states, and Is-
abelle’s context-sensitive proof modes, all under a strictly typed and indentation-sensitive syntax.
Current LLMs, even with long contexts and advanced reasoning capabilities, appear to lack a suffi-
ciently precise internal model of these constraints to make repair reliable at scale. This suggests that
improving fill-and-repair will likely require new forms of training data, tighter symbolic abstrac-
tions, or deeper integration between the planner and the stepwise prover, rather than simply stronger
language models.
Despite these shortcomings, the current implementation is already a useful tool for Isabelle/HOL
users. The stepwise prover alone can discharge goals that defeat existing automation, and the planner
provides a foundation for future work on structured proof synthesis. More broadly, the project
demonstrates that treating theorem proving as a data-generating, verifier-in-the-loop process enables
systematic experimentation and incremental improvement, even when some components fall short
of their ultimate goals. We view this work as a step toward accessible, fully automatic theorem
proving in expressive logics, and as a concrete case study of both the promise and the current limits
of LLM-based reasoning in formal verification.
References
[1] Jasmin Christian Blanchette and Tobias Nipkow. Nitpick: A counterexample generator for
higher-order logic based on a relational model finder. InInternational Conference on In-
teractive Theorem Proving (ITP), Lecture Notes in Computer Science. Springer, 2010. doi:
10.1007/978-3-642-14052-5 11.
[2] Jasmin Christian Blanchette, Sascha B ¨ohme, and Lawrence C. Paulson. Extending sledgeham-
mer with smt solvers. InInternational Conference on Automated Deduction (CADE), Lecture
Notes in Computer Science. Springer, 2011. doi: 10.1007/978-3-642-22438-6 11.
[3] Edward First, Michael Kleiner, et al. Baldur: Whole-proof generation and repair with large
language models. InACM Joint European Software Engineering Conference and Symposium
on the Foundations of Software Engineering (ESEC/FSE), 2023.
[4] Albert Q. Jiang, Wenda Li, Jiawei Han, and Yuhuai Wu. Lisa: Language models of isabelle
proofs. InConference on Artificial Intelligence and Theorem Proving (AITP), 2021.
[5] Albert Q. Jiang, Wenda Li, Sebastian Tworkowski, Konrad Czechowski, Tomasz Odrzyg ´o´zd´z,
Piotr Miło ´s, Yuhuai Wu, and Mateja Jamnik. Thor: Wielding hammers to integrate language
20

models and automated theorem provers. InAdvances in Neural Information Processing Sys-
tems (NeurIPS), 2022.
[6] Guillaume Lample, Marie-Anne Lachaux, Thibaut Lavril, Xavier Martinet, A. Hayat, Gabriel
Ebner, Pierre Rodriguez, and Timoth ´ee Lacroix. Hypertree proof search for neural theorem
proving.arXiv preprint arXiv:2205.11491, 2022.
[7] Tobias Nipkow, Lawrence C. Paulson, and Markus Wenzel.Isabelle/HOL: A Proof Assistant
for Higher-Order Logic, volume 2283 ofLecture Notes in Computer Science. Springer, 2002.
ISBN 978-3-540-43376-3. doi: 10.1007/3-540-45949-9.
[8] Han Wang, Haotian Xin, Zhenwen Liu, Wenya Li, Yufan Huang, Jing Lu, Ziyi Yang, Jie
Tang, Jian Yin, Zhixun Li, and Xiaodan Liang. Proving theorems recursively.arXiv preprint
arXiv:2405.14414, 2024.
[9] Kaiyu Yang, Alexander M. Swope, Albert Gu, Rohan Chalamala, Peng Song, Shuo Yu,
Sachin Godil, Ryan Prenger, and Anima Anandkumar. Leandojo: Theorem proving with
retrieval-augmented language models. InAdvances in Neural Information Processing Systems
(NeurIPS), 2023.
[10] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok,
Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical
questions for large language models.arXiv preprint arXiv:2309.12284, 2023.
[11] Kun Zheng, Jiawei Han, and Stanislas Polu. Minif2f: A cross-system benchmark for formal
olympiad-level mathematics.arXiv preprint arXiv:2109.00110, 2021.
21
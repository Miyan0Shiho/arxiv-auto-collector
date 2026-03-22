# Selective Memory for Artificial Intelligence: Write-Time Gating with Hierarchical Archiving

**Authors**: Oliver Zahn, Simran Chana

**Published**: 2026-03-16 23:06:44

**PDF URL**: [https://arxiv.org/pdf/2603.15994v1](https://arxiv.org/pdf/2603.15994v1)

## Abstract
Retrieval-augmented generation stores all content indiscriminately, degrading accuracy as noise accumulates. Parametric approaches compress knowledge into weights, precluding selective updates. Neither mirrors biological memory, which gates encoding based on salience and archives rather than deletes superseded information. We introduce write-time gating that filters incoming knowledge objects using composite salience scores (source reputation, novelty, reliability) while maintaining version chains that preserve prior states. Using real LLM evaluation without oracle access to quality labels, write gating achieves 100 percent accuracy versus 13 percent for ungated stores. The critical finding emerges under distractor scaling: at 8:1 distractor ratios, read-time filtering (Self-RAG) collapses to 0 percent while write gating maintains 100 percent, revealing a structural advantage of write-time over read-time curation. Validation on Wikipedia (20 entities), procedurally generated pharmacology data, and 2026 arXiv papers confirms these findings. The gating advantage scales inversely with parametric memory support: +25pp for Wikipedia, +48pp for post-cutoff arXiv, +65pp for procedural data with zero training knowledge. Signal ablation confirms the method does not depend on oracle-correlated metadata. Write gating matches Self-RAG accuracy at one-ninth the query-time cost.

## Full Text


<!-- PDF content starts -->

Selective Memory for Artificial Intelligence:
Write-Time Gating with Hierarchical Archiving
Oliver Zahn1and Simran Chana2et al.
1Independent Researcher2University of Cambridge
oliver@coretx.ai
March 2026
Abstract
Retrieval-augmented generation stores all content indiscriminately, degrading accuracy as noise accu-
mulates. Parametric approaches compress knowledge into weights, precluding selective updates. Neither
mirrors biological memory, which gates encoding based on salience and archives rather than deletes
superseded information. We introduce write-time gating that filters incoming knowledge objects using
composite salience scores (source reputation, novelty, reliability) while maintaining version chains that
preserve prior states. Using real LLM evaluation without oracle access to quality labels, write gating
achieves 100 percent accuracy versus 13 percent for ungated stores. The critical finding emerges under
distractor scaling: at 8:1 distractor ratios, read-time filtering (Self-RAG) collapses to 0 percent while
write gating maintains 100 percent, revealing a structural advantage of write-time over read-time cura-
tion. Validation on Wikipedia (20 entities), procedurally generated pharmacology data, and 2026 arXiv
papers confirms these findings. The gating advantage scales inversely with parametric memory support:
+25pp for Wikipedia, +48pp for post-cutoff arXiv, +65pp for procedural data with zero training knowl-
edge. Signal ablation confirms the method does not depend on oracle-correlated metadata. Write gating
matches Self-RAG accuracy at one-ninth the query-time cost.
1 Introduction
Memory architectures for large language models cluster around two extremes. Retrieval-augmented genera-
tion maintains external stores that grow without bound, accepting every document, utterance, and generated
summary regardless of quality or relevance. The resulting noise pollution manifests as retrieval degradation:
when low-quality content constitutes the majority of stored items, embedding similarity increasingly re-
turns distractors rather than correct answers. The alternative extreme, parametric memory, compresses all
knowledge into network weights during training. This approach stores nothing explicitly at inference time,
precluding selective updates, corrections, or compliance with deletion requests.
Neural memory modules occupy a middle ground. Recent work on test-time memorization, exemplified
by Titans [3], introduces learnable memory that updates during inference using gradient-derived surprise
signals. The memory learns what to retain based on prediction error. However, this memory exists as
continuous weight matrices rather than discrete addressable units: there is no representation of individual
stored facts, no mechanism to delete specific items, and no provenance linking outputs to sources.
Biological memory differs from all these approaches in two respects that current architectures neglect.
First, encoding is selective. The hippocampus gates consolidation into long-term memory based on novelty,
emotional salience, and relevance to ongoing goals; mundane experiences receive weak encoding while signifi-
cant events receive strong encoding [17, 16]. Second, forgetting operates through deprioritization rather than
deletion. When beliefs update, the brain does not erase prior states but rather reduces their accessibility
while maintaining links that enable later reconstruction. A system tracking that a company’s CEO changed
from one person to another retains the ability to recall who the previous CEO was; the supersession creates
hierarchy rather than replacement.
This paper introduces write-time gating with hierarchical archiving, a memory mechanism that applies
both biological principles to discrete knowledge stores. Incoming knowledge objects pass through a salience
1arXiv:2603.15994v1  [cs.AI]  16 Mar 2026

Figure 1: Complementary Learning Systems in biology and AI. The brain uses fast hippocampal storage
for specific episodes and slow neocortical storage for semantic patterns. Our architecture mirrors this with
discrete Knowledge Objects (fast, addressable) complementing LLM weights (slow, distributed).
gate that admits only those exceeding a learned threshold. Objects below threshold are archived in cold
storage rather than discarded. When new information updates existing concepts, the system creates su-
persession links rather than performing overwrites, maintaining chains that preserve access to prior states.
We evaluate this approach using real LLM evaluation on retrieval benchmarks where adversarial distractors
outnumber correct answers, demonstrating that selectivity achieves complete quality-tier separation without
requiring oracle access to ground-truth quality labels.
The approach differs from neural memory in operating on discrete addressable objects with explicit
provenance metadata. This enables capabilities that continuous-weight approaches preclude: deletion of
specific items on consent revocation, audit trails linking outputs to contributing sources, and economic
settlement routing value to knowledge contributors based on measured influence.
Contributions.First, we introduce a salience-gated write mechanism that filters incoming knowledge
objects based on composite scores combining source reputation, novelty, and source reliability—without
requiringoracleaccesstoground-truthqualitylabels(§3). Second, wedemonstrateusingrealLLMevaluation
that write gating achieves100%accuracy on adversarial benchmarks versus13.3%for ungated retrieval,
outperforming Self-RAG-style read-time filtering by6.2percentage points (§4). Third, we show that write
gating is asymptotically robust: at distractor ratios up to8:1, write gating maintains100%accuracy while
both ungated retrieval and read-time filtering collapse to0%(§4.6). Fourth, we validate these findings on real
Wikipedia data across 20 entities and 5 categories, demonstrating that write gating achieves98.1%accuracy
versus85.2%ungated while maintaining robustness under scaling, at1/9the query-time cost of Self-RAG;
a signal ablation on Wikipedia confirms the method achieves96.4%without any source metadata signal
(§4.7). Fifth, we demonstrate that the gating advantage scales inversely with parametric memory support
through novel-domain validation:+64.6pp on procedurally generated pharmacology data,+48.4pp on post-
cutoff arXiv papers, confirming that write gating is most valuable when the LLM cannot compensate for
retrieval errors (§4.8). Sixth, we show that archive-rather-than-delete semantics enable temporal queries that
2

overwrite-based systems cannot answer (§4.3). Seventh, we validate multi-path verification for high-stakes
domains requiring>0.95confidence (§5).
Roadmap.Section2reviewsrelatedworkonmemoryarchitectures. Section3presentsthesaliencescoring,
gating, and version chain mechanisms. Section 4 validates our claims through real LLM experiments with
multiple seeds and distractor ratio scaling on both synthetic and real Wikipedia data. Section 5 discusses
biological grounding, limitations, and implications.
2 Related Work
Research on memory for language models spans retrieval augmentation, neural memory modules, external
memory architectures, and cognitive architectures. Prior work on brain-inspired memory approaches reveals
that while individual components exist across scattered systems, no existing architecture integrates discrete
knowledge objects with write-time gating, hierarchical archiving, and version lineage [12, 8].
The external memory tradition began with Neural Turing Machines [4], which introduced controller net-
works coupled with differentiable memory banks using hybrid content-based and location-based addressing.
The Differentiable Neural Computer [5] extended this with temporal linking that tracks write order, enabling
graph traversal and complex reasoning over discrete memory slots. These architectures store discrete entries
with vector content, representing the clearest precursor to indexed memory approaches, though they lack
write-time quality gating and provenance tracking.
ModernHopfieldNetworks[15]achievedatheoreticalbreakthroughbydemonstratingexponentialstorage
capacity and mathematical equivalence between associative memory and transformer attention. This unified
theassociativememoryandattentionliteratures, showingthattransformersimplicitlyperformHopfield-style
pattern retrieval. The connection suggests that explicit memory mechanisms and attention-based retrieval
share deeper structure than previously recognized.
The retrieval-augmented generation paradigm, introduced by Lewis et al. [10], maintains document stores
indexed by embedding similarity. Standard implementations impose no quality filter at indexing time,
admitting all content regardless of source reliability or verification status. Self-RAG [2] adds post-retrieval
filtering using learned critics, but the filtering occurs at read time rather than write time; the underlying
store still accumulates noise. Our approach differs in gating at write time, preventing low-quality content
from entering the active store while preserving it in cold storage for potential future use.
Neural memory architectures have evolved from simple key-value stores toward learned memorization
mechanisms. The Titans architecture [3] introduces memory modules that update during inference using
gradient magnitude as a surprise signal, writing to memory when inputs produce large prediction errors.
The approach achieves strong results on long-context benchmarks by retaining surprising information while
allowing routine patterns to pass without storage. However, the memory consists of continuous weight
matrices without discrete addressable units. Our approach shares the principle of surprise-gated writing but
applies it to discrete knowledge objects, enabling provenance tracking and selective deletion impossible with
weight-based storage.
Neural Episodic Control [14] implements a Differentiable Neural Dictionary storing explicit key-value
pairs, representing discrete indexed storage with k-nearest neighbor retrieval. This provides a practical
implementation of indexed memory for reinforcement learning, though without the salience-based write
gating or version lineage that our approach contributes.
Memory as Recursive Attention Structure (MaRS) [11] extends memory with typed nodes and explicit
forgetting policies, moving toward discrete representations with provenance metadata. The system dis-
tinguishes between different memory types and applies type-specific retention rules. MaRS represents the
closest prior art on provenance-aware memory, though it focuses on read-time forgetting policies rather than
write-time selectivity.
MemGPT [13] addresses bounded context through paging between working memory and long-term stor-
age, treating the language model as an operating system managing its own memory hierarchy. The paging
decisions use recency and explicit programmer hints rather than learned salience, and superseded content
is overwritten rather than archived. Our version chain mechanism addresses the limitation that overwrite
semantics destroy access to prior states.
3

Cognitive architectures provide theoretical grounding for memory selectivity. ACT-R [1] implements
activation-based retrieval where memory items decay unless rehearsed, organizing memory into discrete
“chunks” with typed slots. SOAR [9] maintains production rules with explicit episodic and semantic memory
stores. These architectures demonstrate the value of discrete structured memory but lack neural learning
and modern scaling properties.
Complementary Learning Systems theory [12, 8] explains why biological brains use two memory systems:
a fast-learning hippocampus for episodic memories using sparse pattern-separated representations, and a
slow-learning neocortex for semantic knowledge using distributed overlapping representations. FearNet [7]
implements this theory with three brain-inspired networks: a hippocampal complex for recent memories,
a medial prefrontal cortex network for long-term storage, and a basolateral amygdala network for routing
between systems, achieving memory efficiency through generative replay during simulated sleep phases.
The hippocampal indexing theory [17] proposes that the hippocampus maintains pointers to neocortical
representations rather than storing content directly. Engram research provides biological evidence for dis-
crete memory units: Tonegawa’s laboratory demonstrated that sparse neuronal populations undergo lasting
changes during memory formation, and these engram cells can be artificially reactivated to trigger recall
[6]. Our salience scoring operationalizes these principles: multiple signals combine to produce a gating de-
cision that determines whether incoming content enters active storage, cold storage, or immediate discard.
The archive-rather-than-delete semantics reflect the biological principle that forgetting operates through
deprioritization rather than erasure [16].
Table1summarizesthecapabilitiesofexistingapproaches. Nopriorsystemcombinesdiscreteaddressable
objects, write-time gating, hierarchical archiving, version lineage, and provenance tracking.
Table 1: Capability comparison across memory architectures, based on published system descriptions. Our
approach (SGW-HA) uniquely combines discrete objects with write-time gating, archiving, and version
chains.
System Discrete Write Gate Archive Versions Provenance
RAG✓
Titans✓
DNC✓ ✓
MemGPT✓
MaRS✓ ✓ ✓
NEC✓
SGW-HA (ours)✓ ✓ ✓ ✓ ✓
3 Method
The system maintains three components: an active store of high-salience knowledge objects available for
retrieval, an archive of low-salience or superseded objects in cold storage, and version chains recording
supersessionrelationships. Incomingknowledgeobjectspassthroughsaliencescoringandthresholdingbefore
assignment to active or archive status.
3.1 Salience Scoring
Each candidate knowledge objectKreceives a composite salience score computed from three observable
signals that require no oracle access to ground-truth quality labels. Source reputations rep(K)reflects the
trustworthiness of the contributing source based on verification status, historical accuracy, and institutional
affiliation. Noveltys nov(K)measures semantic distance from existing stored content as one minus the
maximum cosine similarity between the candidate embedding and all active store embeddings. Source
reliabilitys src(K)classifies the source into quality tiers based on observable metadata: peer review status,
institutional backing, and corroboration by independent sources.
4

Knowledge object granularity.A knowledge object may represent a single fact (“Company X’s CEO is
Y”), a document chunk, a claim with supporting evidence, or a versioned belief. The mechanism is agnostic
to granularity: salience scoring and gating apply uniformly. However, effectiveness depends on granularity
choices. Fine-grained objects (single facts) enable precise filtering but require more metadata per unit of
content. Coarse-grained objects (full documents) amortize metadata cost but may mix high and low quality
content within a single unit. We use fact-level granularity in experiments; production systems would tune
granularity to domain requirements.
Design principle: no oracle access.A key design constraint is that salience scoring must operate
without access to ground-truth quality labels. Prior work on memory gating, including our own earlier
simulations, used quality-correlated proxy signals (e.g., prediction-error reduction computed from known
quality tiers) that effectively gave the gating mechanism oracle access to whether each fact was correct.
Such signals are unavailable in deployment. Our three signals—reputation, novelty, and source reliability—
are all observable at write time from metadata alone, without knowing whether the candidate fact is true.
The composite score combines these signals through weights:
S(K) =3X
j=1wj·sj(K)(1)
where the weightsw jcan be set uniformly, tuned on held-out data, or learned online from downstream task
outcomes. In our experiments, we use a reputation-dominated weighting that reflects the empirical finding
that source identity is the strongest predictor of knowledge object quality.
3.2 Write Gating
The gating decision compares composite salience against a thresholdτ. Objects withS(K)≥τenter the
active store with full provenance metadata. Objects withS(K)< τenter the archive rather than being
deleted. This archive-rather-than-delete policy preserves three capabilities that deletion forfeits. First,
archived objects remain available for promotion if circumstances change; a policy update or new use case
may render previously low-salience content valuable. Second, the archive maintains a complete audit trail of
what the system has encountered, supporting forensic analysis of system behavior. Third, archived objects
provide negative evidence: knowing what was rejected and why supports meta-reasoning about knowledge
quality.
The thresholdτmay be fixed, learned, or adaptive. Fixed thresholds work for stable domains with known
quality distributions. Learned thresholds optimize the accuracy-storage tradeoff on held-out data. Adaptive
thresholds condition on available storage, domain, or time, tightening when storage pressure increases or
loosening when comprehensive coverage matters more than precision.
Threshold calibration in practice.Selectingτrequires balancing precision against recall. We recom-
mend calibration via: (a) held-out validation with labeled quality data if available, (b) starting conservative
(τ≥0.6) and relaxing if coverage proves insufficient, or (c) domain-specific tuning based on the cost ratio
of false positives (storing distractors) to false negatives (archiving valid content). The archive mitigates
false negatives—archived objects can be promoted later—but false positives permanently degrade retrieval
quality.
3.3 Version Chains
When new content updates existing concepts, the system creates supersession links rather than overwriting.
LetK newbe an incoming object that updates conceptc, and letK oldbe the current active object for that
concept. The system records bidirectional links:K old.superseded_by←K new.id andK new.supersedes←
Kold.id. The old object moves to the archive while the new object enters the active store.
This creates chains such asv 1→v 2→v 3where arrows denote supersession. The chain enables temporal
queries: retrieving the current value for conceptcreturnsv 3, while retrieving the prior value returnsv 2, and
retrieving the original value returnsv 1. Systems that overwrite can only answer queries about current state;
5

Figure 2: Write-time salience gating architecture. Incoming knowledge objects are scored on three observable
signals (reputation, novelty, source reliability) without oracle access. Objects above thresholdτenter the
active store; objects below are archived in cold storage. On our benchmark: 50 KOs→13 admitted→100%
accuracy (vs 13.3% ungated).
the history is destroyed. Version chains preserve the history, enabling questions like “what did the system
believe before the update” that regulatory and audit requirements increasingly demand.
3.4 Integration with Discrete Knowledge Stores
The mechanism operates on knowledge objects rather than continuous weight matrices. Each object carries
metadata including source identifier, timestamp, verification status, consent vectors, and version links. This
discreteness enables capabilities impossible with weight-based storage. Specific objects can be deleted on
consent revocation by removing them from active and archive stores; with weight-based memory, deletion
requires approximate unlearning procedures that may leave residual information. Outputs can be traced
to contributing sources through lineage tracking; with weight-based memory, attribution requires post-hoc
influence function approximations. Economic settlement can route value to contributors based on measured
influence; with weight-based memory, the contribution of specific training examples is difficult to isolate.
4 Experiments
We evaluate write gating using real LLM evaluation on benchmarks designed to stress-test retrieval under
adversarial conditions, measuring accuracy, storage efficiency, and robustness to increasing noise.
4.1 Experimental Design
The benchmark contains 50 knowledge objects: 10 correct facts from high-reputation sources (peer-reviewed
publications, institutional databases) and 40 distractors from low-reputation sources (unverified claims, AI-
generated content, outdated information). Distractors are designed with high embedding similarity to evalu-
ation queries, modeling the real-world situation where SEO-optimized and AI-generated content increasingly
dominates embedding neighborhoods.
Unlike prior work that uses simulated evaluation with synthetic scoring functions, we evaluate end-to-end
using a real frontier language model (Claude Sonnet 4) as both the retrieval consumer and accuracy judge.
The LLM receives retrieved context and generates answers; accuracy measures whether the generated answer
matches the ground-truth correct fact. This captures failure modes that simulated scoring misses: a model
that retrieves the correct fact alongside compelling distractors may still generate an incorrect answer because
it attends to the more numerous distractors.
Experiments 1 and 2 use 3 and 2 random seeds respectively and report means and standard deviations.
Experiment 3 (distractor scaling) reports single-seed results per ratio. We acknowledge this is a limited
6

sample size; the results should be interpreted as demonstrations of the mechanism’s properties rather than
precise effect size estimates. Multi-seed replication with confidence intervals is a priority for future work.
4.2 Write Gating Results
Table 2 presents accuracy with and without write gating, averaged over three random seeds. In the ungated
condition, all 50 objects enter the active store. Distractors constitute 80% of stored content, and the LLM
preferentially attends to them when generating answers, achieving only13.3%±4.7%accuracy.
With reputation-based write gating, only 12–14 objects pass the salience filter (72–76% compression).
The gated store contains all 10 correct facts plus 2–4 distractors that marginally exceeded the reputation
threshold. Despite this imperfect filtering—the gated store still contains some incorrect content—accuracy
reaches100%±0%across all seeds. The correct facts, now constituting the majority of stored content rather
than a minority, dominate the LLM’s attention.
Table 2: Write gating performance (mean±std over 3 seeds, real LLM evaluation). Reputation-based gating
achieves perfect accuracy without oracle access, storing only 25% of candidates.
Condition Store Size Compression Accuracy
Ungated50±00%13.3%±4.7%
Gated13±175%100%±0%
The result demonstrates that write gating does not require perfect filtering. Even admitting some dis-
tractors, gating shifts the signal-to-noise ratio from 1:4 (ungated) to approximately 3:1 (gated), which is
sufficient for the LLM to reliably identify and use correct facts.
4.3 Temporal Reasoning
Weevaluatewhetherversionchainsenabletemporalqueries byconstructing50conceptswith3versionseach.
Systems receive all updates and must answer queries about both current and prior states. The archive-with-
lineage system maintains supersession chains and achieves 100% accuracy on temporal queries: it can report
any version in any chain. The overwrite system replaces prior versions and achieves 0% accuracy on queries
about prior states: when asked what a concept’s value was before the most recent update, it cannot answer
because the prior value was destroyed.
This result illustrates a qualitative capability difference rather than a quantitative improvement. Over-
write systems fundamentally cannot answer temporal queries regardless of threshold tuning or weight learn-
ing. Archive-with-lineage systems can answer such queries because they preserve rather than destroy history.
Figure 3 illustrates the version chain structure that enables this capability.
4.4 Archive versus Delete
We compare archive semantics (rejected objects go to cold storage) against delete semantics (rejected objects
are discarded). Primary task accuracy is identical since the active store contains the same objects. The
difference appears in recoverability: the archive system retains rejected objects in cold storage available for
future promotion, while the delete system retains none. Storage overhead is modest (cold storage costs less
than active storage), while the option value is significant. Circumstances change: policies update, use cases
emerge, and quality signals improve. Archived objects can be re-evaluated under new conditions; deleted
objects require re-collection if they become needed.
4.5 Comparison with Read-Time Filtering
Self-RAG [2] and similar approaches filter at read time using learned critics to rerank retrieved passages.
We compare four conditions: no filtering, read-time reranking, write gating only, and both combined. Our
read-time filtering uses a Self-RAG-style approach: the same frontier LLM (Claude Sonnet) is prompted
to score each retrieved passage for relevance and correctness before inclusion in the answer context. This
7

Figure 3: Version chains preserve temporal history through supersession links rather than overwrites. When
facts update, prior versions are archived with pointers, enabling temporal queries that overwrite-based sys-
tems cannot answer.
tests theprincipleof read-time filtering at its best—with a frontier model as critic—rather than the specific
Self-RAG trained reflection tokens. We refer to this condition as “Self-RAG” for brevity, noting that the
actual Self-RAG model with trained critics might perform differently.
Table 3: Write-time vs. read-time filtering (mean±std, real LLM evaluation). Write gating outperforms
Self-RAG by+6.2pp. Adding Self-RAG to a gated store slightly degrades performance.
Method Accuracy
No filtering12.5%±0.0%
Self-RAG (read-time)93.8%±6.3%
Write gating (write-time)100%±0.0%
Both (write + read)93.8%±6.3%
Table3showsresults. Self-RAGimprovesaccuracyfrom12.5%to93.8%throughpost-retrievalreranking.
Write gating achieves100%—a further6.2percentage point improvement.
The “Both” condition reveals a counterintuitive result: combining write gating with Self-RAG yields
93.8%,identical to Self-RAG alone and lower than write gating alone. Adding read-time filtering to an
already-clean store degrades performance because the Self-RAG critic occasionally rejects correct facts,
introducing false negatives that write gating alone avoids. In a sufficiently clean store, read-time filtering is
not only unnecessary but counterproductive.
8

4.6 Distractor Ratio Scaling
The preceding experiments use a fixed 4:1 distractor ratio. Real-world retrieval stores vary widely in noise
levels: well-curated knowledge bases may approach 1:1, while web-scale corpora likely exhibit substantially
higher ratios as SEO-optimized, AI-generated, and low-quality content increasingly dominates embedding
neighborhoods. We test whether gating effectiveness and read-time filtering hold as noise increases, varying
the distractor ratio from 2:1 to 8:1.
Table 4: Distractor ratio scaling (real LLM evaluation). Ungated retrieval collapses by 4:1. Self-RAG holds
through 6:1 but collapses catastrophically at 8:1. Write gating maintains 100% at all tested ratios.
Distractor Ratio Ungated Self-RAG Write Gating
2:1 100% 100% 100%
4:1 16.7% 100% 100%
6:1 0% 100% 100%
8:1 0% 0% 100%
Table 4 reveals a phase transition in read-time filtering. At 2:1, all approaches succeed—there is insuf-
ficient noise to challenge any method. At 4:1, ungated retrieval collapses to16.7%as distractors begin to
dominate embedding neighborhoods, while both Self-RAG and write gating maintain100%. At 6:1, ungated
retrieval reaches0%; Self-RAG still holds through effective reranking. The critical transition occurs at8:1:
Self-RAG collapses to0%while write gating maintains100%.
This result reveals a fundamental architectural difference between read-time and write-time filtering.
Self-RAG operates on the retrieval set: it can only promote facts that were retrieved. As the distractor
ratio increases, distractors increasingly dominate the embedding neighborhood of each query. At 8:1, the
probability that the correct fact appears in the initial retrieval set approaches zero—it is crowded out by
eight distractors for every correct fact. No amount of reranking can surface a fact that was never retrieved.
Write gating prevents this failure mode by ensuring distractors never enter the store, keeping the correct
fact’s embedding neighborhood clean regardless of how many distractors exist outside the store.
The 8:1 ratio is practically relevant because real-world knowledge bases accumulate noise from multiple
sources: AI-generated content, SEO-optimized summaries, outdated information, and unverified claims.
While we are not aware of published measurements of distractor-to-signal ratios in production retrieval
systems, the proliferation of synthetic content suggests that noise ratios will increase over time [10]. Even at
8:1—likely conservative for many domains—read-time reranking cannot overcome the fundamental signal-
to-noise problem; write-time curation can.
4.7 Real-World Validation: Wikipedia Facts
Theprecedingexperimentsusesyntheticcorporawheredistractorsaregeneratedbyconstruction. Tovalidate
on real-world data, we extract facts from Wikipedia articles for 20 entities across 5 categories (countries,
scientists, companies, cities, historical figures) and generate distractors via cross-entity fact swaps and LLM
fabrication. Each entity contributes 3–4 ground-truth facts (73 total across the corpus). Distractors are
plausible but incorrect statements about the target entity, generated by Claude Haiku. All evaluation uses
Claude Sonnet 4.6 as both answer generator and accuracy judge. No oracle access is used: the gating
mechanism sees only source labels and novelty signals, not ground-truth correctness.
Write gating vs. ungated (Experiment 1).We evaluate at a 4:1 distractor ratio (292 distractors mixed
with 73 correct facts) across 5 random seeds, each with 73 evaluation questions. Table 5 reports results.
Write gating achieves98.1%accuracy versus85.2%ungated, an improvement of+12.9percentage
points. The gate admits all 73 correct facts in every seed while filtering out 87–90% of distractors (admit-
ting only 29–37 of 292). The resulting store is 70% smaller than the ungated store. Unlike the synthetic
benchmark where ungated accuracy collapsed to 13%, ungated accuracy on Wikipedia remains relatively
high (85%) because the distractors—while plausible—are distinguishable by a frontier LLM that already
9

2:1 4:1 6:1 8:1
Distractor ratio (distractors per correct fact)0%20%40%60%80%100%Accuracy
 Self-RAG
collapsesWeb-scale noise regime
Ungated retrieval
Self-RAG (read-time)
Write gating (ours)Figure 4: Accuracy versus distractor ratio. Ungated retrieval collapses monotonically. Self-RAG maintains
a plateau through 6:1 before catastrophic collapse at 8:1. Write gating remains constant at 100% across all
tested ratios. The shaded region indicates noise levels plausible for web-scale retrieval.
possesses world knowledge about these well-known entities. The gating mechanism nonetheless provides a
meaningful improvement by removing the majority of noise before retrieval.
Self-RAG comparison (Experiment 2).We compare four conditions at 4:1 ratio across 3 seeds (Ta-
ble 6).
On Wikipedia data, Self-RAG performs substantially better than on the synthetic benchmark (97.7%
vs. 93.8%). This is expected: Wikipedia distractors—fabricated statements about well-known entities—are
more obviously wrong to a frontier LLM that already knows these facts, making read-time critic judgments
more reliable. Write gating achieves comparable accuracy (96.8%) while using1/9the LLM calls per query
(Section 4.9). Combining both methods yields 98.2%, the highest accuracy observed.
Distractor ratio scaling (Experiment 3).The central question is whether write gating maintains its
advantage as noise increases. Table 7 reports accuracy across ratios from 2:1 to 8:1 (3 seeds per ratio).
The scaling pattern on Wikipedia data differs from the synthetic benchmark in an important respect.
On synthetic data, Self-RAG collapsed catastrophically at 8:1 (from 100% to 0%). On Wikipedia data, Self-
RAGdegradesonlymodestly(98.6%→95.9%). Thedifferenceisattributabletodistractorquality: synthetic
distractors were designed to be maximally confusing within a narrow domain, while Wikipedia distractors—
cross-entity swaps and fabrications about well-known entities—remain distinguishable to a model with strong
prior knowledge.
Write gating maintains97.7%at all ratios from 4:1 through 8:1, confirming that filtering at write time
prevents noise accumulation regardless of ratio. The practical advantage over Self-RAG is not accuracy
10

No filtering Self-RAG
(read-time)Write gating
(ours)Both
(write + read)0%20%40%60%80%100%Accuracy
12.5%93.8%100.0%93.8% +6.2ppFigure 5: Four-way method comparison at 4:1 distractor ratio. Write gating outperforms Self-RAG by
+6.2pp. Combining both methods degrades to Self-RAG’s accuracy—the critic introduces false negatives in
an already-clean store.
Table 5: Write gating on Wikipedia facts (mean±std over 5 seeds, 73 questions each, 4:1 distractor ratio).
Write gating admits all 73 correct facts while filtering 87% of distractors.
Condition Accuracy Store Size Distractors Admitted
Ungated85.2%±2.0% 365 292
Write-gated98.1%±1.1% 102–110 29–37
(both exceed 95%) but cost: write gating evaluates salience once at write time, while Self-RAG invokes an
LLM critic on every query (Section 4.9).
Source label ablation (Experiment 4).Thesyntheticablation(Table12)showedthatsourcelabelalone
achieves 100% accuracy, raising the concern that the method depends on an oracle-like signal correlated with
ground truth by construction. To test this directly, we ablate the source label signal on Wikipedia data at
4:1 distractor ratio across 5 seeds (Table 8).
Removing the source label signal reduces accuracy by only1.4percentage points (97.8%→96.4%). The
no-label gate admits more items (∼155vs∼110) because it lacks the binary verified/unverified discriminator,
but the additional distractors do not substantially degrade retrieval quality. The conservative, oracle-free
result—96.4%using only reputation and novelty—still represents an11.2percentage point improvement over
ungated retrieval (85.2%). This confirms that effective write gating does not require source metadata that
correlates with correctness by construction.
11

Table 6: Write-time vs. read-time filtering on Wikipedia data (mean±std, 3 seeds, 4:1 ratio). Both filtering
approaches achieve>96%; combining both reaches 98.2%.
Method Accuracy
No filtering84.9%±1.9%
Self-RAG (read-time)97.7%±0.6%
Write gating (write-time)96.8%±1.3%
Both (write + read)98.2%±0.6%
Table 7: Wikipedia distractor ratio scaling (mean over 3 seeds). Ungated accuracy degrades from 98.6% to
72.6% as noise increases. Write gating and Self-RAG both maintain>95% at all ratios.
Ratio Ungated Write Gating Self-RAG
2:198.6% 98.2% 98.6%
4:184.9%97.7% 97.3%
8:172.6%97.7% 95.9%
4.8 Novel-Domain Validation
The Wikipedia validation uses entities about which the LLM has extensive prior knowledge from pretraining.
To test whether write gating generalizes to genuinely novel knowledge—facts the model cannot answer from
parametric memory—we conduct three additional experiments using knowledge provably absent from the
model’s training data.
Experiment A: Procedural pharmacology corpus.We draw 100 drug–target binding affinities from a
vocabulary of 44 kinase inhibitors and 30 molecular targets, with deterministic nM values derived from MD5
hash seeding of each drug–target pair. These values do not exist in any training corpus—every correct answer
must come from the memory store. We generate distractors by permuting drug–target pairs, producing
plausible-looking pharmacology facts with incorrect nM values. Unlike arXiv papers where the model may
havepartialknowledgeofrelatedwork, thesebindingaffinitiesareentirelysyntheticandhavezeroparametric
memory support.
Table 9 and Figure 6(a) show the results. At8:1, ungated retrieval drops to32.0%, lower than for
Wikipedia (72.6%) or arXiv (45.2%), because the model has no parametric memory to compensate for
retrieval errors. Write gating maintains96.6%(+64.6pp), the largest gating advantage observed in any
experiment in this paper. Self-RAG achieves80.2%, trailing write gating by16.4percentage points and
confirming that read-time filtering degrades faster than write-time filtering when the model lacks prior
knowledge of the domain. Store precision decreases monotonically from97.8%at2:1to91.3%at8:1as more
false positives are admitted, but not enough to substantially affect downstream accuracy.
Experiment B: 2026 arXiv papers.We extract 51 factual claims from 40 arXiv papers published
after May 2025 (the evaluation model’s training cutoff). These span condensed matter physics, computa-
tional biology, and applied mathematics. We generate 400 distractors via cross-paper value swaps and LLM
fabrication—producing plausible-looking numerical substitutions within the same technical domain. Unlike
Wikipedia distractors, these are drawn from the same narrow subfields and use domain-appropriate units
and magnitudes, making them substantially harder to distinguish from correct facts.
Table 10 and Figure 6(b) show results across three distractor ratios. At2:1, all methods perform well.
At4:1, ungated retrieval drops to71.6%while gated maintains97.6%(+26.0pp). The critical separation
emerges at8:1: ungated collapses to45.2%while gated maintains93.6%(+48.4pp). Self-RAG also degrades
meaningfully to83.2%, trailing write gating by10.4percentage points. The degradation pattern parallels
the synthetic experiments (Section 4.6), confirming that the architectural advantage of write-time filtering
extends to real-world novel knowledge.
12

Table 8: Source label ablation on Wikipedia data (mean±std over 5 seeds, 73 questions each, 4:1 distractor
ratio). Removing the source label signal costs only 1.4pp.
Configuration Accuracy Store Size Correct / Incorrect
Ungated85.2%±2.0% 365 73/292
Full gating (rep + nov + label)97.8%±1.1%∼110 73/37
No label (rep + novelty only)96.4%±1.4%∼155 72/83
Table 9: Retrieval accuracy on procedural pharmacology data (5-seed means±std, 100 questions per seed).
Facts the LLM provably cannot know from training.
Ratio Ungated Gated Self-RAG
2:198.6%±1.4% 98.8%±1.2% 97.0%±1.7%
4:163.0%±5.2% 98.2%±1.0% 95.0%±2.4%
8:132.0%±4.0%96.6%±1.6% 80.2%±2.1%
Experiment C: Accumulation scaling.We test whether the gating advantage persists as the store
grows continuously. Starting from an empty store, we add 50 novel pharmacology facts per step at a4:1
distractor ratio, evaluating after each of 10 steps. This simulates a realistic deployment scenario where
knowledge accumulates over time.
Table 11 shows three representative steps. The gating advantage is remarkably stable: the gap is+32.0pp
at step 1 and+34.0pp at step 10, with overall means of63.4%ungated versus98.4%gated (+35.0pp).
Ungated accuracy plateaus near63%because retrieval quality degrades proportionally with store growth—
each new batch adds distractors that dilute the retrieval pool. The gated store grows to only∼496items at
step 10 (a19.8%admission rate) while maintaining99.1%precision across all admitted items.
Together with the Wikipedia results (Section 4.7), these experiments validate write gating across four
knowledgeregimes: well-knownfacts(Wikipedia),procedurallygeneratednoveldata(pharmacology),cutting-
edge research (arXiv 2026), and continuously accumulated facts (accumulation scaling). At the8:1distrac-
tor ratio, the gating advantage scales inversely with parametric memory support:+25.1pp for Wikipedia,
+48.4pp for arXiv,+64.6pp for pharmacology (Figure 7). Write gating is most valuable precisely when the
LLM cannot compensate for retrieval errors using training knowledge.
4.9 Computational Efficiency
Write-time and read-time filtering differ in their computational cost profiles. Write gating evaluates salience
at ingestion using metadata signals (reputation, novelty, source label) that require no LLM calls. At query
time, the system makes a single LLM call to generate the answer from retrieved context. Self-RAG, by
contrast, invokes an LLM critic to evaluate each retrieved passage before generating the answer. With a
retrieval pool ofkpassages, Self-RAG requiresk+ 1LLM calls per query (kcritic evaluations plus one
answer generation), compared to1call for write gating.
In our experiments withk= 8retrieved passages, Self-RAG uses9 LLM calls per queryversus1
for write gating—a9×cost difference. This gap compounds with query volume: for a system serving
Qqueries, Self-RAG costsO(Q·k)in LLM calls while write gating costsO(Q). The write-time salience
evaluation is a one-time cost amortized over all future queries against the store.
Write gating also reduces retrieval cost. By admitting only∼29%of candidates, the active store is 70%
smaller, reducing the embedding search space proportionally. For vector similarity search with complex-
ityO(n)(brute force) orO(logn)(approximate nearest neighbor), a 70% smaller store yields meaningful
speedups at scale.
The cost tradeoff favors write gating when: (a) the store is queried more than once (amortizing the
write-time evaluation), (b) the retrieval poolkis large (amplifying the per-query savings), or (c) the system
operates at scale where per-query LLM costs dominate. Self-RAG is preferable when metadata signals are
unavailable, when the store changes too rapidly for write-time evaluation, or when queries are rare enough
13

Table 10: Retrieval accuracy on 2026 arXiv papers (5-seed means±std). Knowledge provably absent from
the model’s training data.
Ratio Ungated Gated Self-RAG
2:194.8%±2.7% 98.0%±2.5% 98.8%±1.6%
4:171.6%±4.1% 97.6%±2.3% 94.4%±3.4%
8:145.2%±2.0%93.6%±3.2% 83.2%±1.6%
Table 11: Retrieval accuracy as the store grows (5-seed means). Gated store admits∼19.8%of candidates,
maintaining99.1%precision throughout.
Step Store Size Gated Size Ungated Gated
1 250∼50 68.0%100.0%
5 1,250∼248 62.0%97.0%
10 2,500∼496 63.0%97.0%
that amortization provides no benefit.
5 Discussion
The distractor ratio scaling result (Table 4) is the central finding of this paper. The gap between write-time
and read-time filtering is not merely quantitative (6.2 percentage points at standard noise) but qualitative:
at sufficiently high noise levels, read-time filtering fails catastrophically while write-time filtering maintains
perfect accuracy. This follows from a structural argument. Read-time filtering can only select among items
that were retrieved. Write-time filtering controls what is available to be retrieved. When noise is low,
the distinction is irrelevant—both approaches find the correct answer. When noise is high, only write-time
filtering can guarantee that the correct answer is retrievable.
The three-signal design (reputation, novelty, source reliability) demonstrates that effective gating does
not require sophisticated quality estimation. A formal ablation study (Table 12, 5 seeds×10 entities per
seed) reveals the relative contribution of each signal:
Table 12: Signal ablation: accuracy by gating configuration (5-seed mean±std)
Configuration Accuracy Store Size
Full model (all 3 signals) 100.0%±0.0% 12.8
Source label only 100.0%±0.0% 10.0
Reputation + label 98.0%±4.0% 11.4
Reputation + novelty 98.0%±4.0% 16.4
Reputation only 94.0%±8.0% 12.6
Novelty only 58.0%±19.4% 15.8
Source label alone matches the full model (100%), confirming that verification status is the dominant
discriminative signal in our synthetic benchmark. This is unsurprising given that the synthetic corpus
assigns labels that correlate with ground truth by construction (Section 5.3). Reputation alone achieves
94%, while novelty alone performs poorly (58%)—embedding similarity is insufficient to distinguish correct
from incorrect facts when both are semantically close to the query.
While source label achieves perfect separation on synthetic data, the Wikipedia signal ablation (Table 8)
demonstrates that the method does not depend on it. Reputation and novelty alone achieve96.4%on
real data, only1.4percentage points below the full signal set (97.8%). The practical implication is that
systems can implement effective write gating with minimal metadata: source identity and content novelty
are sufficient, and verification status—the signal most susceptible to oracle concerns—is helpful but not
necessary.
14

2:1 4:1 8:1
Distractor ratio0%20%40%60%80%100%Accuracy (%)(a)
+64.6ppPharmacology (zero parametric knowledge)
Ungated
Self-RAG
Write gating
2:1 4:1 8:1
Distractor ratio(b)
+48.4pp2026 arXiv (post-cutoff knowledge)
Figure 6: Distractor scaling on novel-domain data (5-seed means±std).(a)Pharmacology corpus with zero
parametric memory support: ungated accuracy drops to32%at8:1while write gating maintains96.6%.(b)
2026 arXiv papers post-training-cutoff: ungated drops to45.2%at8:1while write gating maintains93.6%.
In both domains, write gating degrades less than Self-RAG as noise increases.
Version chains address a different problem than write gating. Gating manages what enters storage;
chains manage how updates interact with prior content. The combination provides both selectivity (not
everything is stored) and history preservation (updates don’t destroy prior states). Regulatory requirements
increasingly demand both: GDPR requires deletion capability (enabled by discrete addressable objects),
while audit requirements demand history reconstruction (enabled by version chains). Weight-based memory
provides neither.
5.1 Biological Grounding: The Mossy Fiber Principle
The archive-rather-than-delete semantics have direct biological grounding in the mossy fiber system of the
hippocampus. The dentate gyrus achieves pattern separation through sparse coding, with only 2–4% of
neurons active for any given input, forcing similar experiences to have dissimilar representations [16]. When
memories become less relevant, the brain does not sever mossy fiber connections; it deprioritizes them,
reducing activation probability while preserving the underlying structure. This enables later recovery if
context changes and the forgotten information becomes relevant again.
This biological principle maps directly to our architecture. Hash-based addressing of discrete knowledge
objects achieves computational pattern separation: each object occupies a unique slot with zero interference
from other objects. When objects are superseded or rejected by the salience gate, they move to archive
rather than being deleted, mirroring the deprioritization of mossy fiber pathways. The result is a system
that can answer questions about prior beliefs—questions that deletion-based systems cannot answer because
the information is gone.
5.2 Relationship to Context Window Scaling
An alternative to write-time gating is simply expanding context windows to accommodate all content, defer-
ring the filtering problem. We argue this does not eliminate the need for gating. First, larger windows make
the distractor problemworse: a 10M-token window that stores everything indiscriminately accumulates
more distractors, not fewer. Our distractor scaling results (Table 4) show that ungated retrieval degrades
monotonically with noise—a larger window at the same distractor ratio produces the same failure. Second,
cost scales linearly with context size (O(N)per query for in-context retrieval vsO(1)for hash-addressed
lookup orO(logN)for indexed embedding search), making store-everything-in-context approaches econom-
ically unsustainable at scale. Third, long-running systems (coding agents, research assistants, enterprise
workflows) inevitably exceed any finite window and must compress, at which point ungated stores lose dis-
15

Wikipedia
(well-known)arXiv 2026
(post-cutoff)Pharmacology
(zero knowledge)0%20%40%60%80%100%120%Accuracy at 8:1 distractor ratio+25.1pp+48.4pp+64.6ppDecreasing parametric memory support
Ungated accuracy
Gated accuracyFigure 7: Gating advantage at8:1distractor ratio across knowledge regimes. The advantage scales inversely
with parametric memory support:+25.1pp for well-known facts (Wikipedia),+48.4pp for post-cutoff re-
search (arXiv 2026),+64.6pp for procedurally generated data with zero training knowledge (pharmacology).
Gated accuracy remains93–97%regardless of domain; the growing green region reflects increasing ungated
degradation.
tractors and correct facts indiscriminately. Write-time gating ensures that when compression occurs, the
remaining content is high-quality by construction.
5.3 Limitations
Several limitations constrain interpretation of these results. First, our evaluation corpus contains 50 knowl-
edge objects—sufficient to demonstrate the mechanism’s properties but small relative to production stores
with millions of objects. Scalability evaluation with approximate novelty computation and large-scale stores
is needed to confirm that gating effectiveness is preserved. We expect hash-addressed lookup (O(1) per
retrieval) to scale well, but write-time novelty computation requires comparing each candidate against all
stored objects, which isO(n)per write.
Second, the salience signals assume access to source metadata. Source reputation requires contributor
identity tracking; source reliability requires institutional affiliation or peer review status. Systems lacking
this metadata would rely on content-only signals (novelty, linguistic quality markers), likely with reduced
gating effectiveness. However, our results show that even imperfect gating—admitting some distractors
alongside correct facts—is sufficient for perfect downstream accuracy.
A related concern is that our benchmark’s source metadata is strongly correlated with ground-truth
correctness by construction: correct facts come from high-reputation verified sources, while distractors come
from low-reputation unverified sources. However, the source label ablation on Wikipedia data (Table 8)
demonstrates that this correlation is not load-bearing: removing the source label signal entirely reduces
accuracy by only1.4percentage points (97.8%→96.4%), confirming that reputation and novelty alone
provide effective gating. In the real world, verified sources occasionally contain errors and unverified sources
occasionallycontaincorrectinformation. Totestrobustnesstothislabelnoise, weintroduceacorruptionrate
where a fraction of verified sources contain incorrect facts while retaining their high-reputation metadata.
16

Figure 8: Pattern separation in the dentate gyrus. Sparse coding (2–4% activation) forces similar inputs
to have dissimilar representations, preventing interference. Mossy fiber connections are deprioritized rather
than deleted, preserving recovery capability—the biological basis for archive semantics.
Table 13: Corruption robustness: accuracy when verified sources contain incorrect facts (5-seed mean±std,
n=10entities per seed).
Corruption Rate Gated Ungated
0% (baseline) 98.0%±4.5% 4.0%±5.5%
5% 78.0%±16.4% 4.0%±5.5%
10% 74.0%±12.0% 2.0%±4.5%
20% 52.0%±20.5% 6.0%±5.5%
30% 30.0%±14.1% 0.0%±0.0%
Table 13 confirms graceful degradation rather than catastrophic failure. At 5% corruption—a realistic
rate for curated sources—gated accuracy drops from 98% to 78%, while ungated accuracy remains near zero
regardless of corruption rate. The degradation is approximately linear: each 10 percentage points of corrup-
tion reduces gated accuracy by roughly 20 percentage points. This reflects the mechanism’s fundamental
design: write gating trusts metadata to select what enters the store, so corrupted metadata admits corrupted
facts. The degradation profile suggests that in production systems, maintaining metadata quality (source
verification, peer review status) is at least as important as the gating algorithm itself.
Third, our synthetic benchmarks evaluate on a single domain (factual QA) with generated corpora. The
Wikipedia validation (Section 4.7) and novel-domain experiments (Section 4.8) extend coverage to real-
world data across multiple knowledge regimes, including genuinely novel knowledge from 2026 arXiv papers.
However, the question counts per experiment are modest (100 per seed for pharmacology, 51 claims for arXiv,
20 per step for accumulation scaling), and validation on web-scale corpora and additional domain-specific
databases would further strengthen the claims.
Fourth, the distractor scaling experiment tests ratios up to 8:1. Production web-scale retrieval may face
even higher ratios. While the write-gating mechanism should remain robust at any ratio (it is immune to
distractors outside the store by construction), this claim needs validation at 16:1, 32:1, and beyond.
17

5.4 Comparison with Neural Memory
ThecomparisonwithTitansillustratesthetradeoffbetweencontinuousanddiscretememory. Titansachieves
efficient learned memorization through gradient-based surprise signals operating on weight matrices. Our
approach achieves selective memorization through multi-signal salience operating on discrete objects. Titans
enables smooth interpolation and efficient storage; discrete objects enable provenance tracking and surgical
deletion.
The approaches exhibit complementary failure modes. Neural memory (Titans) fails when: (a) stored
facts must be surgically deleted (e.g., GDPR right-to-be-forgotten), (b) outputs must be traced to specific
sources for audit, or (c) semantically similar facts interfere, causing the retrieval errors described in prior
work on the stability gap. Discrete KO memory fails when: (a) salience metadata is unavailable or unreliable,
(b) content does not decompose cleanly into discrete units, or (c) the task benefits from soft interpolation
between related memories rather than hard retrieval of specific items.
A hybrid architecture might use neural memory for soft statistical patterns (style, tone, general world
knowledge) while reserving discrete knowledge objects for hard factual claims requiring provenance. The
salience gate could route incoming content to the appropriate store based on content type: claims with
verifiable sources to KO storage, statistical regularities to neural memory. We have not explored such hybrid
architectures experimentally; this remains an open direction.
5.5 Multiple Verification Paths
The salience gating and version chain mechanisms suggest a natural extension for high-stakes domains re-
quiring verification guarantees. Human memory recall operates through multiple convergent paths: spatial
context (where was I?), temporal context (when did this happen?), and semantic rules (what usually hap-
pens?). Convergence across independent paths increases confidence that a recalled memory is accurate rather
than confabulated.
An analogous mechanism could strengthen knowledge object verification. Rather than relying on a
single derivation chain, the system could maintain multiple independent paths to each fact. A claim that
a company’s CEO changed in 2011 might be verified through: (1) the version chain showing supersession
timestamps, (2) cross-references from board meeting records, and (3) corroborating mentions in financial
filings. Verification confidence would scale with the number of convergent paths:
C(K) = 1−Y
p∈paths(K)(1−c p)(2)
wherec pis the confidence of individual pathp. Three paths each with 0.9 confidence yield aggregate
confidence of 0.999.
Wevalidatethisformulathroughsimulation. Withthreeverificationpaths(versionchain, cross-reference,
semantic rules), correct facts achieve mean confidence 0.998 while incorrect facts achieve 0.35. The critical
insight emerges at high-confidence thresholds: at threshold 0.95, single-path verification provides 0% usable
coverage (no correct facts exceed the threshold), while multi-path verification provides 100% usable coverage.
For domains like healthcare or finance requiring very high confidence before acting, multi-path verification
is essential—single-path verification cannot reach the required confidence levels regardless of how accurate
the underlying signals are.
Thecostofverificationscaleswiththenumberofpathstraversed, creatinganaturaltradeoffbetweencon-
fidence and computational expense—mirroring the brain’s allocation of cognitive resources to consequential
decisions.
6 Conclusion
Write-time gating with hierarchical archiving applies biological memory principles to artificial intelligence.
Using real LLM evaluation without oracle access to quality labels, selective encoding achieves100%accuracy
on adversarial benchmarks versus13.3%for ungated retrieval, and outperforms read-time filtering (Self-
RAG) by6.2percentage points. The critical finding is asymptotic robustness: at distractor ratios up to8:1,
18

read-time filtering collapses to0%while write gating maintains100%, revealing a fundamental architectural
advantage of write-time curation over read-time filtering. Validation on real Wikipedia data across 20
entities confirms these findings in a real-world setting: write gating achieves98.1%accuracy versus85.2%
ungated (+12.9pp) and maintains97.7%at8:1distractor ratios, while costing9×fewer LLM calls per query
than read-time filtering. Novel-domain validation demonstrates that the gating advantage at8:1scales
inversely with parametric memory support:+64.6pp on procedurally generated pharmacology data where
the model has zero prior knowledge,+48.4pp on 2026 arXiv papers post-training-cutoff, and+25.1pp on
well-known Wikipedia facts (Figure 7). Accumulation scaling experiments confirm the advantage remains
stable (+35.0pp) as the store grows to 2,500 items. The mechanism requires only three observable signals
(source reputation, novelty, source reliability) without oracle access to quality labels; a signal ablation on
Wikipedia data confirms that even without source reliability metadata, reputation and novelty alone achieve
96.4%—only1.4pp below the full signal set—demonstrating that the method does not depend on any signal
correlated with ground truth by construction. Key limitations include: the synthetic evaluation corpus is
small (50 KOs) and the read-time filtering baseline uses a zero-shot LLM critic rather than the trained
Self-RAG model. Despite these limitations, the structural argument—that write-time filtering controls what
is available to retrieve, while read-time filtering can only select among retrieved items—is independent of
benchmark scale or signal implementation. Version chains enable temporal queries that overwrite-based
systems cannot answer. The approach operates on discrete addressable knowledge objects, with implications
for provenance tracking and selective deletion in domains where regulatory requirements for AI transparency
are strengthening.
Reproducibility.Code implementing the salience scoring, gating mechanism, and experimental bench-
marks is available in the project repository.1
References
[1] Anderson, J. R., Bothell, D., Byrne, M. D., Douglass, S., Lebiere, C., and Qin, Y. An integrated theory
of the mind.Psychological Review, 111(4):1036–1060, 2004.
[2] Asai, A., Wu, Z., Wang, Y., Sil, A., and Hajishirzi, H. Self-RAG: Learning to retrieve, generate, and
critique through self-reflection.arXiv preprint arXiv:2310.11511, 2023.
[3] Behrouz, A., Zhong, P., and Mirrokni, V. Titans: Learning to memorize at test time.arXiv preprint
arXiv:2501.00663, 2024.
[4] Graves, A., Wayne, G., and Danihelka, I. Neural Turing machines.arXiv preprint arXiv:1410.5401,
2014.
[5] Graves, A., Wayne, G., Reynolds, M., Harley, T., Danihelka, I., Grabska-Barwińska, A., Colmenarejo,
S. G., Grefenstette, E., Ramalho, T., Agapiou, J., et al. Hybrid computing using a neural network with
dynamic external memory.Nature, 538(7626):471–476, 2016.
[6] Josselyn, S. A. and Tonegawa, S. Memory engrams: Recalling the past and imagining the future.
Science, 367(6473):eaaw4325, 2020.
[7] Kemker, R. and Kanan, C. FearNet: Brain-inspired model for incremental learning. InInternational
Conference on Learning Representations, 2018.
[8] Kumaran, D., Hassabis, D., and McClelland, J. L. What learning systems do intelligent agents need?
Complementary learning systems theory updated.Trends in Cognitive Sciences, 20(7):512–534, 2016.
[9] Laird, J. E.The Soar Cognitive Architecture. MIT Press, 2012.
[10] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.,
Rocktäschel, T., Riedel, S., and Kiela, D. Retrieval-augmented generation for knowledge-intensive NLP
tasks. InAdvances in Neural Information Processing Systems, 2020.
1Repository URL to be added upon publication.
19

[11] Li, H., et al. MaRS: Memory-augmented recurrent structures for efficient sequence modeling.arXiv
preprint, 2024.
[12] McClelland, J. L., McNaughton, B. L., and O’Reilly, R. C. Why there are complementary learning
systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist
models of learning and memory.Psychological Review, 102(3):419–457, 1995.
[13] Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., and Gonzalez, J. E. MemGPT:
Towards LLMs as operating systems.arXiv preprint arXiv:2310.08560, 2023.
[14] Pritzel, A., Uria, B., Srinivasan, S., Badia, A. P., Vinyals, O., Hassabis, D., Wierstra, D., and Blundell,
C. Neural episodic control. InInternational Conference on Machine Learning, pages 2827–2836, 2017.
[15] Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Adler, T., Gruber, L., Holzleitner, M.,
Pavlović, M., Sandve, G. K., et al. Hopfield networks is all you need. InInternational Conference on
Learning Representations, 2021.
[16] Squire, L. R., Stark, C. E., and Clark, R. E. The medial temporal lobe.Annual Review of Neuroscience,
27:279–306, 2004.
[17] Teyler, T. J. and DiScenna, P. The hippocampal memory indexing theory.Behavioral Neuroscience,
100(2):147–154, 1986.
20
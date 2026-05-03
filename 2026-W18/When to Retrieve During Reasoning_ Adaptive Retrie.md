# When to Retrieve During Reasoning: Adaptive Retrieval for Large Reasoning Models

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu

**Published**: 2026-04-29 13:15:44

**PDF URL**: [https://arxiv.org/pdf/2604.26649v1](https://arxiv.org/pdf/2604.26649v1)

## Abstract
Large reasoning models such as DeepSeek-R1 and OpenAI o1 generate extended chains of thought spanning thousands of tokens, yet their integration with retrieval-augmented generation (RAG) remains fundamentally misaligned. Current RAG systems optimize for providing context before reasoning begins, while reasoning models require evidence injection during multi-step inference chains. We introduce ReaLM-Retrieve, a reasoning-aware retrieval framework that addresses this mismatch through three key innovations: (1) a step-level uncertainty detector that identifies knowledge gaps at reasoning-step granularity rather than token or sentence level; (2) a retrieval intervention policy that learns when external evidence maximally benefits ongoing reasoning; and (3) an efficiency-optimized integration mechanism that reduces per-retrieval overhead by 3.2x compared to naive integration. Experiments on MuSiQue, HotpotQA, and 2WikiMultiHopQA demonstrate that ReaLM-Retrieve achieves on average 10.1% absolute improvement in answer F1 over standard RAG (range: 9.0-11.8% across the three benchmarks) while reducing retrieval calls by 47% compared to fixed-interval approaches like IRCoT (all improvements significant at p<0.01, paired bootstrap). On the challenging MuSiQue benchmark requiring 2-4 hop reasoning, our method achieves 71.2% F1 with an average of only 1.8 retrieval calls per question. Analysis shows that ReaLM-Retrieve also improves retrieval quality itself, achieving 81.3% Recall@5 with consistently higher precision and MRR than fixed-interval baselines on supporting evidence, establishing new state-of-the-art efficiency-accuracy trade-offs for reasoning-intensive retrieval tasks.

## Full Text


<!-- PDF content starts -->

When to Retrieve During Reasoning: Adaptive Retrieval for Large
Reasoning Models
Dongxin Guo∗†
The University of Hong Kong
Hong Kong, China
Brain Investing Limited
Hong Kong, China
bettyguo@connect.hku.hkJikun Wu∗
Stellaris AI Limited
Hong Kong, China
Brain Investing Limited
Hong Kong, China
hk950014@connect.hku.hkSiu Ming Yiu
The University of Hong Kong
Hong Kong, China
Brain Investing Limited
Hong Kong, China
smyiu@cs.hku.hk
Abstract
Large reasoning models such as DeepSeek-R1 and OpenAI o1 gen-
erate extended chains of thought spanning thousands of tokens,
yet their integration with retrieval-augmented generation (RAG)
remains fundamentally misaligned. Current RAG systems optimize
for providing contextbeforereasoning begins, while reasoning mod-
els require evidence injectionduringmulti-step inference chains.
We introduceReaLM-Retrieve, a reasoning-aware retrieval frame-
work that addresses this mismatch through three key innovations:
(1) astep-level uncertainty detectorthat identifies knowledge gaps at
reasoning-step granularity rather than token or sentence level; (2) a
retrieval intervention policythat learns when external evidence max-
imally benefits ongoing reasoning; and (3) anefficiency-optimized
integrationmechanism that reduces per-retrieval overhead by 3.2 ×
compared to naive integration. Experiments on MuSiQue, Hot-
potQA, and 2WikiMultiHopQA demonstrate thatReaLM-Retrieve
achieves on average 10.1% absolute improvement in answer F1
over standard RAG (range: 9.0–11.8% across the three benchmarks)
while reducing retrieval calls by 47% compared to fixed-interval
approaches like IRCoT (all improvements significant at 𝑝< 0.01,
paired bootstrap). On the challenging MuSiQue benchmark requir-
ing 2–4 hop reasoning, our method achieves 71.2% F1 with an
average of only 1.8 retrieval calls per question. Analysis shows
thatReaLM-Retrievealso improves retrieval quality itself, achiev-
ing 81.3% Recall@5 with consistently higher precision and MRR
than fixed-interval baselines on supporting evidence, establishing
new state-of-the-art efficiency-accuracy trade-offs for reasoning-
intensive retrieval tasks.
CCS Concepts
•Information systems →Question answering;Retrieval mod-
els and ranking;Information retrieval query processing;•Com-
puting methodologies→Natural language processing.
∗Both authors contributed equally to this research.
†Corresponding author.
This work is licensed under a Creative Commons Attribution 4.0 International License.
SIGIR ’26, Melbourne, VIC, Australia.
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2599-9/2026/07
https://doi.org/10.1145/3805712.3809722Keywords
Retrieval-Augmented Generation, Large Reasoning Models, Multi-
hop Question Answering, Adaptive Retrieval, Chain-of-Thought
Reasoning, Uncertainty Quantification
ACM Reference Format:
Dongxin Guo, Jikun Wu, and Siu Ming Yiu. 2026. When to Retrieve During
Reasoning: Adaptive Retrieval for Large Reasoning Models. InProceedings of
the 49th International ACM SIGIR Conference on Research and Development in
Information Retrieval (SIGIR ’26), July 20–24, 2026, Melbourne, VIC, Australia.
ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3805712.3809722
1 Introduction
The emergence of large reasoning models (LRMs) marks a major
shift in language model capabilities. Models such as DeepSeek-
R1 [5], OpenAI o1 [ 21], and QwQ [ 22] generate extended chains
of thought spanning 12,000–25,000 tokens, exhibiting emergent
behaviors including self-verification, backtracking, and strategy
switching. Unlike standard chain-of-thought (CoT) prompting [ 33],
these models are trained via reinforcement learning to develop
genuine multi-step reasoning capabilities that cannot be replicated
through prompting alone.
However, a critical limitation persists: reasoning models excel at
reasoning with known factsbut struggle whenexternal factual knowl-
edgeis required. Recent analyses reveal that reasoning models show
degradedfactual accuracy compared to their non-reasoning coun-
terparts on fact-seeking tasks [ 37], with increased hallucination
rates on benchmarks like SimpleQA. This creates a fundamental
opportunity for retrieval augmentation; yet existing RAG systems
are fundamentally misaligned with reasoning model architectures.
The core mismatch is temporal: current RAG pipelines optimize
for providing relevant context before generation begins [ 4,17].
This “retrieve-then-generate” paradigm assumes a single retrieval
step suffices. For reasoning models generating thousands of tokens
across multiple logical steps, this assumption fails catastrophically.
A reasoning chain may begin with sufficient context but encounter
knowledge gaps mid-inference, gaps that cannot be anticipated at
query time.
Iterative retrieval methods such as IRCoT [ 31], FLARE [ 11], and
Self-RAG [ 1] address this by interleaving retrieval with generation.
Yet these methods were designed for standard language models, not
reasoning architectures, and exhibit fundamental incompatibilities:
(1) Granularity mismatch: Existing methods trigger retrieval
at token level (FLARE) or sentence level (IRCoT), while reasoning
models operate at reasoning step granularity (logical units spanning
multiple sentences).(2) Signal unavailability: Methods relyingarXiv:2604.26649v1  [cs.IR]  29 Apr 2026

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia. Dongxin Guo, Jikun Wu, and Siu Ming Yiu
on token probabilities (FLARE) or internal attention states (DRA-
GIN [ 27]) cannot operate with completion-only models like o1.(3)
Efficiency collapse: Fixed-interval retrieval (IRCoT) becomes pro-
hibitively expensive for reasoning chains spanning thousands of
tokens, adding 2–5 seconds per retrieval call.
We proposeReaLM-Retrieve, a reasoning-aware retrieval frame-
work that addresses these challenges through principled integration
of retrieval into reasoning model inference. Our approach makes
three key contributions:
Contribution 1: Step-Level Uncertainty Detection.We pro-
pose a novel uncertainty quantification method operating at reasoning-
step granularity. Unlike token-level entropy or sentence-level con-
fidence, our Reasoning Step Uncertainty Score (RSUS) identifies
when a logical inference requires external knowledge versus when
reasoning should continue with existing context.
Contribution 2: Learned Retrieval Intervention Policy.We
formalize reasoning-retrieval integration as a sequential decision
problem and learn an intervention policy 𝜋(𝑎 𝑡|𝑠𝑡)that decides
whether to retrieve, what query to formulate, and how to integrate
retrieved content, all conditioned on the current reasoning state.
Contribution 3: Efficiency-Optimized Integration.We de-
velop implicit compression and speculative caching mechanisms
that reduce per-retrieval overhead by 3.2 ×compared to naive inte-
gration (from 2.1s to 0.66s per call), enabling practical deployment
of reasoning-retrieval systems with 1.33 ×lower end-to-end latency
than fixed-interval approaches.
Experiments on three multi-hop QA benchmarks show that
ReaLM-Retrieveachieves substantial improvements in both ac-
curacy and efficiency. On MuSiQue, we achieve 71.2% F1 with 1.8
average retrieval calls compared to IRCoT’s 65.4% F1 with 3.4 calls
(improvement significant at 𝑝< 0.01). Our method establishes
a new Pareto frontier for reasoning-retrieval trade-offs, demon-
strating that fewer, better-timed retrievals outperform frequent,
fixed-interval retrievals.
2 Related Work
2.1 Retrieval-Augmented Generation
RAG systems [ 2,6,17] augment language models with external
knowledge retrieved from document collections. The standard ap-
proach involves a single retrieval step before generation. Dense
retrieval methods using dual encoders [ 13,35] or late interaction
architectures [ 14,24] have substantially improved retrieval quality.
Recent evaluation frameworks [ 3] enable principled assessment of
retrieval-generation pipelines. Multi-vector retrieval engines such
as PLAID [ 23] and WARP [ 25] optimize late-interaction retrievers,
achieving 3–41×speedups. Our work complements these advances
by optimizing when to invoke retrieval during reasoning.
2.2 Iterative and Adaptive Retrieval
Recent work explores iterative retrieval strategies. IRCoT [ 31] in-
terleaves retrieval with chain-of-thought reasoning, retrieving after
each sentence. While effective for short generation (achieving 15 F1-
point improvements on 2WikiMultiHopQA), fixed-interval retrieval
becomes impractical for reasoning chains spanning thousands of
tokens. FLARE [ 11] triggers retrieval when token probability fallsbelow a threshold, but requires token-level probabilities unavail-
able from completion-only models. DRAGIN [ 27] uses attention
entropy but requires internal states. Self-RAG [ 1] learns special to-
kens through fine-tuning, achieving strong performance but requir-
ing full model fine-tuning impossible for proprietary models. RE-
PLUG [ 26] treats models as black boxes but operates at query-level
without mid-reasoning intervention. A separate line of work makes
the retrieval decisionat the query levelbased on question com-
plexity: Adaptive-RAG [ 10] routes queries to no-retrieval, single-
retrieval, or multi-retrieval strategies, and Open-RAG [ 9] learns
reflection tokens that determine retrieval necessity per query. Most
closely related to our setting, the concurrent work of Hashemiet
al.[7] (“Dynamic Search-R1”) extends Search-R1 with a cost-aware
RL objective that adapts retrievaldepth(the number of documents
per sub-query); their orthogonal axis (how many to retrieveper
sub-query, learned via PPO/GRPO with token-cost penalties) is
complementary to our axis (when to retrievebetween reasoning
steps, decided by RSUS-driven uncertainty rather than RL). Com-
bining the two (adaptive timing with adaptive depth) is a natural
future direction. Our work extends adaptive retrieval to reasoning-
step granularity within extended inference chains, providing finer-
grained timing signals than query-level approaches and operating
without RL fine-tuning.
2.3 Large Reasoning Models
Large reasoning models represent a distinct class of model from
standard LLMs. DeepSeek-R1 [ 5] uses pure RL with Group Relative
Policy Optimization, generating reasoning chains of 12,000–23,000
tokens with emergent behaviors (self-verification, backtracking,
hypothesis testing) that standard CoT prompting cannot replicate.
Search-R1 [ 12] applies RL to train models for interleaved search
decisions, achieving 41% improvement over RAG baselines, but
focuses on when to search rather than retrieval system optimization.
Our work addresses this gap with a systems-oriented approach to
reasoning-retrieval integration.
2.4 Uncertainty Quantification for LLMs
Determining when retrieval is needed requires estimating model
uncertainty. Semantic entropy [ 15] clusters sampled responses by
meaning, achieving AUROC of 0.70–0.85 for predicting accuracy,
but the 5–10×inference overhead compounds with expensive rea-
soning inference. Verbalized confidence [ 29] prompts models to
express confidence, offering a black-box-compatible alternative.
Self-consistency [ 32] uses agreement across reasoning chains but
requires 5–20×sampling overhead. Linet al.[ 19] propose a princi-
pled black-box uncertainty framework based on pairwise semantic
similarity over multiple sampled generations, which can outperform
verbalized confidence in calibration; we use verbalized confidence
as one component of RSUS specifically because of its low overhead
and complementary signal to entity-coverage entropy, but Linet
al.’s pairwise approach is a natural drop-in replacement for 𝑈verb
when sampling cost is acceptable. Conformal-prediction approaches
such as TRAQ [ 18] provide statistical correctness guarantees end-
to-end on RAG outputs; this is orthogonal to our problem ofwhen
to retrieve, but the two directions are composable. Moskvoretskiiet
al.[20] find that simpler uncertainty methods can match complex

When to Retrieve During Reasoning: Adaptive Retrieval for Large Reasoning Models SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia.
pipelines while requiring <1% computational overhead. We design
lightweight uncertainty signals specifically for reasoning architec-
tures.
3 Problem Formulation
We formalize reasoning-retrieval integration as follows. Given a
query𝑞and document collection D, a reasoning model generates a
chain𝑅=(𝑟 1,𝑟2,...,𝑟 𝑛)where each𝑟𝑖represents a reasoning step, a
coherent unit of logical inference that may span multiple sentences.
The goal is to learn an intervention policy 𝜋that at each step 𝑟𝑖
decides:(1) Retrieve decision: 𝑎ret
𝑖∈{0,1}indicating whether
to retrieve;(2) Query formulation: 𝑞𝑖=𝑓(𝑞,𝑟 1:𝑖)specifying the
retrieval query;(3) Context integration: 𝑐𝑖=𝑔(𝑟 1:𝑖,R𝑖)whereR𝑖
denotes retrieved documents. The objective is to maximize answer
accuracy while minimizing retrieval cost:
max
𝜋E𝑞∼Q[Acc(𝑎 𝜋,𝑎∗)−𝜆·Cost(𝜋)](1)
where𝑎 𝜋is the answer produced under policy𝜋,𝑎∗is the ground
truth, and𝜆balances accuracy against retrieval cost (number of
calls and latency).Key insight: Unlike prior work treating this as
a per-token or per-sentence decision, we operate at reasoning step
granularity, matching the fundamental unit of reasoning model
operation.
4 Method:ReaLM-Retrieve
ReaLM-Retrieveconsists of three components: (1) reasoning step
segmentation and uncertainty estimation, (2) a learned intervention
policy, and (3) efficient retrieval integration. Figure 1 provides an
overview of the problem, our solution, and key results.
4.1 Reasoning Step Segmentation
Reasoning models produce extended outputs that must be seg-
mented into coherent reasoning steps. Unlike sentence-level seg-
mentation used by IRCoT, we identifylogical reasoning units: seg-
ments that complete a coherent inference before the next logical
step begins. We employ a lightweight classifier trained on human-
annotated reasoning traces to identify step boundaries. The classi-
fier operates on a sliding window of tokens, predicting boundary
probability based on: (1) Discourse markers (“therefore”, “however”,
“this means”); (2) Logical connectives indicating inference comple-
tion; (3) Topic shifts detected via embedding similarity; (4) Punctu-
ation and formatting patterns. For open-weight models (DeepSeek-
R1, QwQ), we additionally use the model’s own reasoning structure
markers when available.
Classifier Architecture and Training.The step boundary
classifier is a 3-layer transformer encoder (hidden dimension 256,
4 attention heads) operating on 128-token sliding windows with
64-token stride. Training data consists of 2,847 human-annotated
reasoning traces from DeepSeek-R1 outputs on a held-out subset
of NaturalQuestions (disjoint from evaluation benchmarks). The
three annotators were given written instructions to mark a bound-
ary wherever the model (a) completes one logical inference and
shifts to a new sub-claim, sub-question, or strategy; (b) introduces
a discourse marker (“therefore”, “however”, “so”, “next”) signalling
such a shift; or (c) closes a verification or backtracking episode.
They were instructednotto mark boundaries at every sentence,and to prefer fewer, semantically coherent segments over finer
ones. Annotators achieved𝜅=0.78inter-annotator agreement on
boundary placement. The classifier is trained for 10 epochs with
cross-entropy loss (learning rate5 ×10−5, batch size 32), achiev-
ing 94.2% F1 on a held-out NaturalQuestions test set ( 𝑁= 412
traces). Average segment length is 127 tokens compared to 23 to-
kens for sentence-level segmentation. An ablation removing each
input feature class indicates that discourse markers (1) contribute
most to boundary F1 ( −3.4% F1 when removed), followed by logi-
cal connectives (−2.1%), embedding-based topic shift ( −1.6%), and
punctuation/formatting ( −0.8%); the relatively large punctuation
contribution reflects R1-style traces’ tendency to place line breaks
at strategy transitions.
Out-of-domain segmentation generalization.A reasonable
concern is that the classifier, trained on NaturalQuestions traces,
may not transfer to the multi-hop benchmarks used for end-to-end
evaluation. To assess this, we collected reasoning traces produced
by R1-Distill-Qwen-32B on 𝑁= 15randomly-sampled questions
from each of MuSiQue, HotpotQA, and 2WikiMultiHopQA, and
had a single annotator (using the same written instructions) mark
boundaries for each trace. Table 1 reports boundary F1 of the in-
domain classifier on each benchmark relative to the in-domain
NaturalQuestions test set. The classifier transfers with a modest
degradation (1 .5–3.0F1 points), consistent with the intuition that
reasoning-step structure is largely model-determined rather than
benchmark-determined; multi-hop questions admit slightly more
variability in boundary placement (lower IAA, not shown), ex-
plaining most of the residual gap. Segmentation errors on these
benchmarks correlate weakly with downstream retrieval-decision
errors (𝑟= 0.21), suggesting that small boundary-position drift
does not propagate severely through RSUS.
Table 1: Step-boundary classifier generalization. Boundary
F1 against held-out human annotations on the four datasets
(𝑁=15traces per OOD benchmark,𝑁=412for NQ).
Dataset Boundary F1 Segments / trace
NaturalQuestions (in-domain) 94.2 9.4
MuSiQue 91.4 8.7
HotpotQA 92.7 7.9
2WikiMultiHopQA 91.2 8.3
4.2 Step-Level Uncertainty Detection
We define theReasoning Step Uncertainty Score(RSUS), a compos-
ite measure identifying when a reasoning step requires external
knowledge:
RSUS(𝑟 𝑖)=𝛼·𝑈 verb(𝑟𝑖)+𝛽·𝑈 ent(𝑟𝑖)+𝛾·𝑈 cons(𝑟𝑖)(2)
We chose this particular triple, rather than alternatives like se-
mantic entropy [ 15] or pairwise-similarity black-box uncertainty [ 19],
for three reasons. First, overhead: each of our components costs
<3%of reasoning inference, whereas semantic entropy or self-
consistency cost5–20 ×inference. With reasoning chains already at
12K–25K tokens, methods with multiplicative overhead are infeasi-
ble. Second, the three signals are complementary. 𝑈verbprobes the

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia. Dongxin Guo, Jikun Wu, and Siu Ming Yiu
(a) The Problem
Traditional RAG
QueryRetrieve
OnceGenerate
Answer
before reasoning starts
Reasoning Models
Query Step 1 Step 2 Step 3 ... Step𝑛Gap! Gap!
need evidence during 12K–25K token chains
(c) MuSiQue Benchmark(c) MuSiQue Benchmark
0 1 2 350607071.2%
65.4%
Retrieval CallsF1 Score (%)
No Retrieval Single RAG IRCoT
Search-R1 ReaLM-RetrieveKey Results:
•71.2% F1vs 65.4% (IRCoT)
+5.8%absolute improvement
•1.8 callsvs 3.4 calls
47%fewer retrievals
•2.1×efficiencygain
Better accuracy per call
Establishes new Pareto frontier for
reasoning-retrieval trade-offs(b)ReaLM-RetrieveSolution
Reasoning
Chain➊
Segment Steps
➋
Detect RSUS
➌
Policy𝜋 𝜃
➍
Integrate ContextRetriever
Enhanced
Chainif gapParse into reasoning steps
Step-level uncertainty
3.2×faster merge
Figure 1: Overview ofReaLM-Retrieve’s adaptive retrieval approach. (a) The temporal mismatch: traditional RAG retrieves
once before generation, but reasoning models encounter knowledge gaps during 12K–25K token chains. (b) Our four-stage
pipeline detects step-level uncertainty and retrieves only when needed via learned policy. (c) MuSiQue benchmark results show
71.2% F1 with only 1.8 retrieval calls, outperforming IRCoT (65.4%, 3.4 calls) and establishing a new Pareto frontier.
model’s self-knowledge of factual correctness; 𝑈entprobes corpus
coverage of named entities, an external signal independent of the
model;𝑈consprobes reasoning robustness. An ablation (Table 6)
confirms each contributes orthogonally. Third, all three signals can
be computed without token-level logprobs, so the framework oper-
ates uniformly on completion-only API models and on open-weight
models. We do not claim this triple is optimal: richer alternatives
such as Linet al.’s pairwise-similarity uncertainty are drop-in sub-
stitutes for𝑈 verbwhen sampling cost is acceptable.
Verbalized uncertainty 𝑈verb: We prompt the reasoning model
at step boundaries:“Given the reasoning so far, rate your confidence
that the current conclusion is factually correct on a scale of 0-100,
where 0 means completely uncertain and 100 means absolutely certain.
Respond with only a number:”The response is normalized to [0,1]by
dividing by 100.For completion-only models that do not support mid-
generation prompting(the case for o1-class API endpoints during
late 2025), we substitute a learned proxy: a 2-layer MLP (hidden
dimension 128, GELU activation) takes as input the concatenation
of (a) the SBERT embedding of the most recent reasoning step, (b)shallow surface features (segment length, count of hedge phrases
such as “probably”/“I think”/“maybe”, count of named entities), and
(c) the running mean of 𝑈entover the last three steps. The proxy is
trained on the same 2,847-trace corpus (§4.1) using subsequent-step
error labels (whether the next reasoning step revised, contradicted,
or expressed doubt about the current step) as cross-entropy targets,
achieving 0.71 AUROC against a held-out NaturalQuestions subset
of 412 traces (a separate split from the boundary-classifier test set of
§4.1, sized to match for comparability), substantially below the 0.83
AUROC of true verbalized confidence on open-weight models, but
sufficient to recover most of the retrieval-timing benefit (Table 7
shows the o1 vs. R1 gap is only 2–3% F1).
Entity-based entropy 𝑈ent: We extract named entities from 𝑟𝑖
and compute retrieval score entropy across candidate documents:
𝑈ent(𝑟𝑖)=−∑︁
𝑒∈Ent(𝑟 𝑖)𝑝(𝑒|D)log𝑝(𝑒|D)(3)
High entropy indicates the reasoning step references entities with
ambiguous or sparse coverage. For each named entity 𝑒∈Ent(𝑟 𝑖)

When to Retrieve During Reasoning: Adaptive Retrieval for Large Reasoning Models SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia.
extracted via spaCy NER, we compute entity-specific retrieval by
querying the corpus with “What is [e]?” The probability 𝑝(𝑒|D) is
derived from normalized BM25 scores across top-100 documents
mentioning𝑒.
Consistency signal 𝑈cons: For critical reasoning steps (identified
by discourse markers), we sample 𝑘=3alternative continuations
and measure agreement. Disagreement indicates reasoning uncer-
tainty that retrieval may resolve. The weights 𝛼,𝛽,𝛾 are learned on a
validation set to maximize the correlation between RSUS and down-
stream retrieval benefit. RSUS computation adds only 8% overhead
to reasoning inference, compared to 500–2000% for full semantic
entropy computation.
4.3 Retrieval Intervention Policy
Given RSUS scores, we learn a policy 𝜋𝜃that decides when to
retrieve. We model this as a contextual bandit problem with state
𝑠𝑖=(𝑞,𝑟 1:𝑖,RSUS(𝑟 𝑖),ℎ𝑖)where𝑟1:𝑖is the reasoning prefix up to
and including step 𝑖(§3) andℎ𝑖encodes retrieval history (previous
queries, retrieved documents, time since last retrieval). The policy
outputs:
𝑎ret
𝑖=⊮[𝜋 𝜃(𝑠𝑖)>𝜏](4)
𝑞𝑖=QueryGen(𝑠 𝑖)if𝑎ret
𝑖=1(5)
Policy architecture: We use a lightweight transformer encoder
that processes the concatenation of query embedding, current rea-
soning step embedding, RSUS features, and retrieval history. The
encoder outputs a retrieval probability and, if triggered, a query
representation used for retrieval.Training: We train 𝜋𝜃jointly
with QueryGenvia REINFORCE [ 34] with reward 𝑅=F1(𝑎 𝜋,𝑎∗)−
𝜆1·𝑛ret−𝜆 2·𝑡latency , where𝑛retis the number of retrieval calls
and𝑡latency is total retrieval latency; QueryGen receives gradients
from the same scalar reward via the policy decision 𝑎ret
𝑖=1, so
improvements to retrieval quality are credited back to the query
formulator. We use curriculum learning, starting with high 𝜆values
(penalizing retrieval) and gradually decreasing to encourage the
model to learnwhichretrievals provide maximum benefit.Query
formulation: When retrieval is triggered, QueryGen extracts the
query from the current reasoning context. We find that queries
formulated asthe information need(“What is [entity]’s relationship
to [concept]?”) outperform queries formulated asthe current reason-
ing step. QueryGen is a single-layer transformer decoder (hidden
dimension 512, 8 attention heads) that cross-attends to the current
reasoning step embedding and outputs a 768-dimensional query
representation used directly for dense retrieval.
4.4 Efficient Retrieval Integration
Injection protocol.We implement retrieval injection assegmented
generation with context re-feeding, not as single-pass mid-generation
prompting. When the policy triggers retrieval at step boundary 𝑖
(identified by the segmentation classifier of §4.1), generation is
paused; the prefix 𝑟1:𝑖is concatenated with retrieved evidence for-
matted in a delimited block ( <retrieved> . . .</retrieved> ) and a
brief continuation prompt (“Continue your reasoning, using this
evidence if relevant.”); this augmented context is then re-fed to
the model, which produces step 𝑟𝑖+1onwards. Insertion points are
always at step boundaries (never mid-step), so retrieved text neverbreaks an in-flight inference. For open-weight models we use KV-
cache-aware integration (described below) so that re-feeding does
not require full recomputation; for completion-only models the
protocol falls back to a standard system+prefix continuation call.
Post-retrieval distribution shift.Retrieved evidence alters
the distribution of subsequent reasoning context, which could in
principle degrade the segmentation classifier and RSUS computed
on later steps. We mitigate this in two ways. First, during classifier
training (§4.1) we include traces in which one or more retrievals
have already been injected, so boundary detection is exposed to
evidence-augmented contexts; on a held-out subset of such traces,
classifier F1 drops only modestly from 94.2% (pre-retrieval) to 91.6%
(post-retrieval, single prior injection). Second, we measured the
correlation between RSUS and downstream retrieval benefit con-
ditional on a prior retrieval having occurred and observed only
a small drop (Pearson 𝑟: 0.72→0.68); since the median question
requires≤2retrievals (Table 4), this is the dominant operational
regime. Long chains with ≥3retrievals show further degradation
and are flagged in §7.5.
Implicit compression.Rather than prepending full retrieved
documents, we extract and compress relevant evidence. For each
retrieved passage 𝑑𝑗, we compute attention-weighted importance
scores with respect to the current reasoning query and retain only
sentences exceeding threshold 𝜏rel. This reduces context expansion
by 73% while preserving 96% of retrieval utility.
Speculative caching.We observe that reasoning chains often
follow predictable paths. After the first retrieval, we speculatively
retrieve documents for likely next-step queries based on entities
mentioned in retrieved content. These speculative retrievals execute
in parallel with reasoning continuation. When a speculative query
matches the actual triggered query (37% hit rate), we eliminate
retrieval latency entirely.
KV-cache preservation.For open-weight models, we imple-
ment KV-cache-aware context integration that preserves cached
key-value states for unchanged context portions. This reduces time-
to-first-token after retrieval by 2.1 ×compared to full context re-
computation.
4.5 Complexity Analysis
Let𝐿be reasoning chain length, 𝑛be number of retrievals, and 𝑘be
documents retrieved per call.Segmentation: 𝑂(𝐿) with constant
factor from sliding window classifier.RSUS computation: 𝑂(𝐿/𝑠)
where𝑠is average segment length ( ≈127 tokens).Policy inference:
𝑂(𝑛) , negligible compared to reasoning inference.Retrieval: 𝑂(𝑛·
𝑡ret)where𝑡retis per-call latency. Total overhead is dominated
by retrieval latency. With speculative caching (37% hit rate) and
implicit compression, effective per-retrieval overhead is 67% of
naive integration.
5 Experimental Setup
5.1 Datasets
We evaluate on three multi-hop QA benchmarks:MuSiQue[ 30]:
2–4 hop questions requiring connected reasoning (24,814 questions
with decomposed sub-questions, reporting EM and F1 on test set).
HotpotQA[ 36]: 2-hop questions in fullwiki setting with 5.2M

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia. Dongxin Guo, Jikun Wu, and Siu Ming Yiu
Wikipedia passages (reporting EM, F1, and Sup-F1).2WikiMulti-
HopQA[ 8]: Cross-document reasoning requiring evidence from
two Wikipedia sources (reporting EM, F1, and Evi-F1).
5.2 Baselines
We compare against:No Retrieval: Closed-book reasoning model.
Single RAG: Standard retrieve-then-generate with top- 𝑘passages.
IRCoT[ 31]: Fixed-interval interleaved retrieval after each sentence.
FLARE[ 11]: Token-probability-triggered retrieval (where proba-
bilities are available).Self-RAG[ 1]: Self-reflective retrieval with
special tokens (using released Llama-2-13B fine-tuned model, as
Self-RAG requires model fine-tuning unavailable for proprietary
models).Search-R1[ 12]: RL-trained retrieval decisions for reason-
ing models.
5.3 Models and Retrieval
For reasoning models, we use DeepSeek-R1-Distill-Qwen-32B for
extensive ablations and DeepSeek-R1-671B for headline results. We
also evaluate on QwQ-32B-Preview to assess generalization across
reasoning architectures. For retrieval, we use ColBERTv2 [ 24] with
PLAID engine for efficient late-interaction retrieval ( 𝑘=5passages
per query, maximum passage length 256 tokens).
5.4 Evaluation Metrics
Beyond standard QA metrics, we report retrieval efficiency:Re-
trieval calls: Average number of retrieval invocations per question.
E2E latency: End-to-end time from query to answer.Reasoning
tokens: Total tokens generated in reasoning chain.Accuracy-
efficiency ratio: F1 / retrieval calls.
5.5 Implementation Details
All experiments run on 8 ×NVIDIA A100 80GB GPUs. Reason-
ing inference uses vLLM [ 16] with tensor parallelism. Retrieval
uses WARP [ 25] optimizations achieving 171 ms average retrieval
latency. The intervention policy is trained for 50,000 steps with
learning rate10−4, batch size 64, curriculum-decreasing 𝜆1from0.5
to0.1. RSUS weights are set by grid search on a held-out validation
split. All comparisons use paired bootstrap (10,000 iterations) with
Bonferroni correction; we report 95% CIs and treat 𝑝< 0.05as
significant. All experiments use 3 random seeds (42, 123, 456).1
6 Results
6.1 Main Results
Table 2 presents our main results across all benchmarks.ReaLM-
Retrieveconsistently outperforms all baselines in accuracy while
using substantially fewer retrieval calls. On MuSiQue with R1-
32B,ReaLM-Retrieveachieves 71.2% F1 with 1.8 retrieval calls
vs IRCoT’s 65.4% F1 with 3.4 calls. This is a 5.8% absolute im-
provement with 47% fewer retrievals (95% CI: [4.2, 7.4], 𝑝<0.01).
1Hyperparameter sweep summary. Boundary-classifier hidden dim selected from {128,
256, 512} (91 .2/94.2/94.1% F1); policy hidden dim 512. Best RSUS weights (𝛼, 𝛽,𝛾)=
(0.40,0.35,0.25)achieve correlation0 .72with downstream retrieval benefit (vs0 .64
for equal weights). Retrieval threshold 𝜏=0.65balances precision0 .72/ recall0 .81.
Implicit-compression threshold 𝜏rel=0.45retains 27% of content while preserving
96% of retrieval utility. MuSiQue-tuned hyperparameters transfer to HotpotQA at
98.3% of in-domain performance.The accuracy-efficiency ratio (F1/calls) improves from 19.2 (IR-
CoT) to 39.6 (ReaLM-Retrieve), a 2.1 ×improvement. With R1-
671B,ReaLM-Retrieveachieves 77.8% F1 on MuSiQue, 4.4% above
Search-R1 (95% CI: [3.0, 5.8], 𝑝< 0.01), demonstrating that our
approach scales effectively with reasoning model capability.
Self-RAG Comparison Caveat.The Self-RAG baseline uses a
different base model (Llama-2-13B) than our method (DeepSeek-R1),
as Self-RAG requires fine-tuning unavailable for reasoning models.
Despite this architectural difference, the comparison remains infor-
mative: Self-RAG achieves 2.1 retrieval calls with 61.9% F1, while
ReaLM-Retrieveachieves 1.8 calls with 71.2% F1. Even controlling
for retrieval frequency, our step-level uncertainty detection yields
superior retrieval timing.
Supporting Evidence Quality.On HotpotQA,ReaLM-Retrieve
achieves 68.2% Supporting Fact F1 (Sup-F1), compared to 62.4% for
IRCoT and 64.8% for Search-R1. On 2WikiMultiHopQA, Evidence F1
(Evi-F1) reaches 71.8% versus 65.3% for IRCoT. These improvements
corroborate that step-level uncertainty detection identifies the cor-
rect moments for retrieval. Table 3 presents statistical significance
analysis confirming all improvements.
6.2 Efficiency Analysis
Table 4 presents detailed efficiency metrics.ReaLM-Retrieveadds
only 1.7 seconds latency over no-retrieval baseline (13.7% overhead)
vs 6.3 seconds for IRCoT (51% overhead). The reduction comes
from fewer retrieval calls (1.8 vs 3.4), speculative caching (eliminat-
ing 37% of retrieval latency), and implicit compression (reducing
context length by 73%). The 3.2 ×per-call efficiency improvement
(0.66s vs 2.10s for Naive Interleaving) results from three comple-
mentary optimizations whose contributions are additive on per-
call latency rather than multiplicative.Speculative cachingis the
dominant contributor: by executing likely next-step retrievals in
parallel with reasoning, it eliminates retrieval entirely on a 37%
hit rate, removing the full 2.10s per cached call and accounting
for the largest share of the saving in expectation.Implicit com-
pressionof retrieved evidence reduces context-processing time on
the remaining (non-cached) calls by shrinking injected context by
73%.KV-cache preservationfurther reduces post-retrieval time-to-
first-token by avoiding recomputation of unchanged context for
open-weight models. Combined with 57% fewer retrieval calls over-
all,ReaLM-Retrieveachieves 1.50 ×lower end-to-end latency than
Naive Interleaving and 1.33 ×lower than IRCoT. Token count anal-
ysis showsReaLM-Retrieveuses 9,489 tokens vs IRCoT’s 11,284.
Retrieval at appropriate moments allows the model to reach an-
swers with less reasoning, as evidence resolves uncertainties that
would otherwise require extended exploration.ReaLM-Retrieve
reduces cost by 16% vs IRCoT (Table 5) while achieving 5.8% higher
F1, demonstrating favorable cost-accuracy trade-offs for production
deployment.
6.3 Ablation Studies
Table 6 presents ablation results isolating each component’s contri-
bution.Uncertainty detection: Verbalized uncertainty contributes
most (2.8% F1 when removed, 𝑝<0.01), confirming reasoning mod-
els’ self-reported confidence provides strong signal for retrieval

When to Retrieve During Reasoning: Adaptive Retrieval for Large Reasoning Models SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia.
Table 2: Main results on multi-hop QA benchmarks. Best in bold, second underlined .†Uses Self-RAG fine-tuned model (different
base). *Uses token probabilities unavailable for R1-671B. Significance:∗∗𝑝<0.01,∗𝑝<0.05vs. next-best (paired bootstrap, 10K
iterations, Bonferroni corrected).
MuSiQue HotpotQA 2WikiMHQA Avg.
Method EM F1 Calls EM F1 Calls EM F1 Calls F1
DeepSeek-R1-Distill-Qwen-32B
No Retrieval 41.2 48.7 0.0 38.4 51.2 0.0 35.8 47.3 0.0 49.1
Single RAG 52.6 59.4 1.0 51.3 62.8 1.0 49.7 60.1 1.0 60.8
IRCoT 58.3 65.4 3.4 56.2 67.9 4.1 55.8 66.2 3.8 66.5
FLARE* 55.1 62.3 2.8 53.9 65.4 3.2 52.4 63.7 2.9 63.8
Self-RAG†54.8 61.9 2.1 55.4 66.3 2.4 53.2 64.5 2.2 64.2
Search-R1 59.1 66.8 2.4 57.8 69.2 2.7 56.4 67.4 2.5 67.8
ReaLM-Retrieve 63.5∗∗71.2∗∗1.8 60.4∗∗71.8∗∗1.9 59.2∗∗69.7∗∗1.7 70.9∗∗
DeepSeek-R1-671B
No Retrieval 48.3 55.9 0.0 45.2 58.1 0.0 42.6 54.8 0.0 56.3
Single RAG 58.7 66.2 1.0 57.9 69.4 1.0 55.3 66.9 1.0 67.5
IRCoT 64.2 71.8 3.6 62.4 73.5 4.3 61.2 72.1 3.9 72.5
Search-R1 65.8 73.4 2.6 64.1 75.2 2.8 62.7 73.8 2.6 74.1
ReaLM-Retrieve 70.2∗∗77.8∗∗1.9 67.3∗∗78.4∗∗2.0 65.9∗∗76.5∗∗1.8 77.6∗∗
Table 3: Statistical significance (R1-32B). 95% CI via paired
bootstrap (10,000 iterations).
vs. IRCoT vs. Search-R1
Dataset ΔF1 95% CI ΔF1 95% CI
MuSiQue +5.8 [4.2, 7.4] +4.4 [2.9, 5.9]
HotpotQA +3.9 [2.4, 5.4] +2.6 [1.2, 4.0]
2WikiMHQA +3.5 [2.1, 4.9] +2.3 [0.9, 3.7]
Table 4: Efficiency comparison on MuSiQue (R1-32B). Latency
in seconds.Naive Interleaveretrieves at every reasoning step
without optimizations.
Method Calls Latency Per-Call Tokens F1/Call
No Retrieval 0.0 12.4 – 8,432 –
Single RAG 1.0 13.2 0.80 9,156 59.4
Naive Interleave 4.2 21.2 2.10 12,847 14.9
IRCoT 3.4 18.7 1.85 11,284 19.2
FLARE 2.8 16.9 1.61 10,647 22.3
Search-R1 2.4 15.8 1.42 10,102 27.8
ReaLM-Retrieve 1.8 14.1 0.66 9,489 39.6
necessity. Entity entropy contributes 2.1% F1, particularly for ques-
tions involving rare entities. Random triggering degrades F1 by 8.5%,
demonstrating thatwhento retrieve matters substantially.Policy
components: The learned threshold outperforms fixed threshold
(2.3% F1 gain, 𝑝<0.01), as optimal retrieval timing varies across
question types. Query formulation contributes 1.8% F1.Integra-
tion components: Efficiency optimizations have modest accuracy
impact but substantial latency benefits (Table 4).Table 5: Cost-per-query on MuSiQue using DeepSeek-R1-
671B output-token pricing ($2.19/1M output tokens). Input-
token cost (at $0.55/1M) is provider-dependent (it varies with
each API’s prefix-caching policy) and scales with the num-
ber of retrieval injections; including it would increase the
reported savings vs. IRCoT, so we report output-only cost
as the conservative, platform-independent comparison axis.
Savings are vs. IRCoT.
Method Tokens Cost/Query Savings
No Retrieval 8,432 $0.018 –
Single RAG 9,156 $0.020 –
IRCoT 11,284 $0.025 –
Search-R1 10,102 $0.022 12%
ReaLM-Retrieve 9,489 $0.021 16%
6.4 Analysis by Question Complexity
Figure 2 breaks down performance by question complexity.ReaLM-
Retrieveshows increasing advantage for more complex questions:
+3.2% F1 on 2-hop, +5.8% on 3-hop, and +8.4% on 4-hop questions
vs IRCoT. This reflects our method’s core advantage: step-level
uncertainty detection identifies thespecific reasoning stepwhere
knowledge gaps occur, enabling precisely-timed retrieval. For sim-
ple 2-hop questions, single RAG often suffices; for complex 4-hop
questions, retrieving exactly when needed provides substantial
benefit.
6.5 Retrieval Timing Analysis
Figure 3 visualizes when retrieval occurs during reasoning chains.
For successfully answered questions, retrievals cluster at specific
phases, typically around 20% and 50% of the reasoning chain, cor-
responding to initial evidence gathering and mid-reasoning ver-
ification. Failed questions show more uniform retrieval timing,

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia. Dongxin Guo, Jikun Wu, and Siu Ming Yiu
Table 6: Ablation study on MuSiQue (R1-32B). Results are
mean±std over 3 seeds.∗∗𝑝<0.01,∗𝑝<0.05vs. FullReaLM-
Retrieve(paired t-test).
Configuration F1 CallsΔF1
FullReaLM-Retrieve 71.2±0.6 1.8±0.1 –
Uncertainty:
w/o Verb. (𝑈 verb) 68.4±0.8 2.1±0.2 -2.8∗∗
w/o Ent. (𝑈 ent) 69.1±0.7 1.9±0.1 -2.1∗∗
w/o Cons. (𝑈 cons) 70.3±0.5 1.8±0.1 -0.9∗
RSUS→Random 62.7±1.2 1.8±0.1 -8.5∗∗
Policy:
Fixed threshold𝜏 68.9±0.9 2.4±0.2 -2.3∗∗
w/o Query formulation 69.4±0.6 1.8±0.1 -1.8∗∗
w/o Retrieval history 69.8±0.7 2.0±0.1 -1.4∗
Integration:
w/o Implicit compr. 70.8±0.5 1.8±0.1 -0.4
w/o Specul. cache 71.0±0.6 1.8±0.1 -0.2
w/o KV preservation 70.9±0.5 1.8±0.1 -0.3
2-hop 3-hop 4-hop50607072.1
69.4
62.868.9
63.6
54.466.2
61.8
51.663.4
57.2
48.5
Number of Reasoning HopsF1 Score (%)
ReaLM-Retrieve IRCoT
Single RAG No Retrieval
Figure 2: F1 score by number of reasoning hops on MuSiQue.
Y-axis starts at 48% to highlight performance differences.
ReaLM-Retrieve(blue) shows largest improvements on 3–
4 hop questions (+3.2% on 2-hop, +5.8% on 3-hop, +8.4% on
4-hop vs. IRCoT). Patterns ensure grayscale readability. Error
bars omitted for visual clarity; see Table 6 for variance.
suggesting that poorly-timed retrievals disrupt rather than assist
reasoning.
6.6 Retrieval Quality Analysis
Beyond end-to-end QA accuracy, we evaluate retrieval quality di-
rectly using HotpotQA’s supporting fact annotations as ground
truth. Table 7 reveals thatReaLM-Retrievenot only reduces re-
trieval calls but triggershigher-qualityretrievals. Our method achieves
81.3% Recall@5, substantially higher than IRCoT (72.6%) and Search-
R1 (74.1%). The “Useful%” metric (fraction of retrievals where at0 0.2 0.4 0.6 0.8 10102030
Retrieval Position (fraction of reasoning chain)Frequency (%)Successful
Failed
Figure 3: Distribution of retrieval timing (as fraction of rea-
soning chain) for successful (solid blue) vs. failed (dashed
orange) questions on MuSiQue. Successful questions show
retrieval clustering at specific reasoning phases (20%, 50%),
while failed questions have more uniform distribution, indi-
cating poorly-timed retrievals disrupt reasoning.
Table 7: Retrieval-quality comparison on HotpotQA. Rele-
vance is judged against supporting-fact annotations; higher
is better for all metrics. “Useful%” = fraction of retrievals
where≥1supporting fact appears in top-5.
Method R@5 P@5 MRR Useful%
Single RAG 68.4 42.1 0.71 64.2
IRCoT 72.6 38.7 0.69 71.8
FLARE 69.8 40.2 0.68 67.3
Search-R1 74.1 41.8 0.73 73.6
ReaLM-Retrieve 81.3 48.6 0.79 82.4
Table 8: Generalization ofReaLM-Retrieveto QwQ-32B-
Preview on MuSiQue. The relative improvement over IRCoT
(5.2 F1, 47% fewer calls) mirrors the R1-32B result, indicating
the policy depends on observable reasoning behavior rather
than model-specific internals.
Method EM F1 Calls
No Retrieval 39.4 46.8 0.0
Single RAG 50.2 57.6 1.0
IRCoT 55.9 63.2 3.2
ReaLM-Retrieve 60.8 68.4 1.7
least one supporting fact appears in top-5) reaches 82.4%, indicating
step-level uncertainty detection successfully identifies beneficial re-
trieval moments. This improvement stems from: (1) RSUS identifies
knowledge gaps precisely when they occur rather than at arbitrary
intervals, and (2) QueryGen formulates queries targeting specific
information needs.
6.7 Generalization to Other Reasoning Models
Table 8 shows thatReaLM-Retrievegeneralizes to QwQ-32B-
Preview, a reasoning model with different architecture and training
approach than DeepSeek-R1. We achieve 68.4% F1 with 1.7 calls vs
IRCoT’s 63.2% with 3.2 calls, a similar relative improvement. This

When to Retrieve During Reasoning: Adaptive Retrieval for Large Reasoning Models SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia.
Table 9: Cross-benchmark generalization: Policy trained on
HotpotQA, evaluated on MuSiQue (R1-32B).
Training Data EM F1 Calls
MuSiQue (in-domain) 63.5 71.2 1.8
HotpotQA (transfer) 61.2 68.9 2.0
Δ -2.3 -2.3 +0.2
generalization results from our method’s reliance on observable
reasoning behavior rather than model-specific internals.
6.8 Cross-Benchmark Generalization
To assess policy generalization beyond in-domain training, we train
ReaLM-Retrieveon HotpotQA and evaluate zero-shot on MuSiQue
(Table 9). The transferred policy achieves 68.9% F1, only 2.3% below
in-domain training, and substantially outperforms IRCoT (65.4%
F1), demonstrating that RSUS captures generalizable uncertainty
patterns across multi-hop QA tasks.
6.9 Token Efficiency and Behavioral Analysis
Beyond retrieval-call counts, we analyzetoken efficiency: how re-
trieval affects reasoning-chain length. Table 4 showsReaLM-Retrieve
generates 9,489 reasoning tokens on average vs IRCoT’s 11,284 (16%
reduction): well-timed retrieval resolves uncertainties that would
otherwise require extended exploration. Across MuSiQue, three
behavioral patterns dominate:early resolution(42% of cases) yields
23% shorter chains;verification retrievals(31%) reduce backtracking
by 18%; andfailed retrievals(12%) extend chains by 8%. The net
effect demonstrates external evidence complementing rather than
replacing reasoning, yielding both higher accuracyandshorter
chains. Stratifying by hop count on MuSiQue (Figure 2),ReaLM-
Retrieveattains 9–22% relative F1 improvement over Single RAG,
with the largest gains on 4-hop questions.
Illustrative case.A representative 3-hop MuSiQue question
(“Who directed the film that won Best Picture in the year the Berlin
Wall fell?”) showsReaLM-Retrieve’s adaptive timing in action.
The model first recalls “The Berlin Wall fell in 1989” with low un-
certainty (𝑈verb=0.31), proceeds to “The Best Picture winner in
1989 wasDriving Miss Daisy”, then RSUS triggers retrieval at the
verification point (558 tokens, 42% through reasoning, 𝑈verb=0.68,
𝑈ent=0.81); a second retrieval at 897 tokens (67%) returns di-
rector information. Total: 1,342 tokens, 2 retrievals at 168 ms and
174 ms. IRCoT retrieves after each sentence (4 retrievals, 683 ms to-
tal), including unnecessary retrievals for well-known facts.ReaLM-
Retrievehalves retrieval latency (342 ms vs. 683 ms) at 100% accu-
racy on this trace.
7 Discussion
7.1 When Does Retrieval Help Reasoning?
Our analysis reveals patterns in when retrieval maximally ben-
efits reasoning:Entity knowledge gaps: Entity entropy ( 𝑈ent)
successfully identifies cases involving entities not well-represented
in model knowledge, particularly rare entities appearing in fewer
than 100 corpus documents.Verification checkpoints: 73% ofsuccessful retrievals occur at verification boundaries, identified
by discourse markers like “let me verify” or “to confirm”.Bridge
reasoning: When reasoning stalls at bridge facts connecting en-
tities (high𝑈verbwith low𝑈ent), retrieval provides connecting in-
formation, accounting for 41% of improvement over fixed-interval
approaches. By contrast, retrievalharmsreasoning when injected
during exploratory strategy development (31% of failures) or when
contradicting well-grounded reasoning (18% of failures).
7.2 Comparison to Contemporary Work
Our work extends recent LRM+RAG integration advances. Com-
pared to Search-R1 [ 12], which learns when to invoke search via RL,
we additionally optimize query formulation and context integration,
achieving 4.4% higher F1 with 25–27% fewer retrieval calls (25%
on R1-32B; 27% on R1-671B). Compared to ReARTeR [ 28] (SIGIR
2025), which introduces process-level rewards, our step-level uncer-
tainty detection provides finer-grained timing signals, yielding 4.6%
higher F1 with 28% fewer calls. The concurrent Dynamic Search-
R1 [7] learns adaptive retrievaldepthvia cost-aware RL on the
Search-R1 backbone; our axis (whento retrieve, between reasoning
steps) is orthogonal to theirs (how manydocuments to retrieve per
sub-query), and the two are composable. Our timing policy could
feed an adaptive-depth retriever to compound their per-call savings
with our per-question savings. The key differentiator isgranularity:
operating at reasoning-step level rather than sentence level (IRCoT),
query level (Adaptive-RAG), or search-decision level (Search-R1)
enables more precise knowledge gap identification, particularly
evident on 4-hop questions whereReaLM-Retrieveachieves 6.3%
higher F1 than ReARTeR.
Rather than relying on structured prompts, our approach con-
trols retrieval timing in models such as DeepSeek-R1 that generate
end-to-end reasoning chains through RL training. Our learned pol-
icy achieves 47% fewer retrievals than fixed-interval prompting
(IRCoT) and 25% fewer than RL-trained search invocation (Search-
R1). It maintains higher accuracy in both cases, proving the value
of adaptive timing over static scheduling.
7.3 Failure Mode Analysis
Analysis of 847 failed questions reveals three primary failure modes:
(1) Retrieval corpus gaps(43%): Required information is ab-
sent, causing noise injection, particularly for recent events (post-
2023) and specialized domains.(2) Over-retrieval during ex-
ploration(31%): RSUS occasionally triggers retrieval during ex-
ploratory reasoning phases, disrupting productive chains by prema-
turely committing to hypotheses.(3) Query formulation errors
(26%): QueryGen produces queries retrieving tangentially related
content, often due to entity ambiguity (e.g., “Washington” as person
vs. state).ReaLM-Retrievealso underperforms Single RAG on 12%
of questions ( 𝑁= 288), primarily simple 2-hop cases where early
retrieval suffices, suggesting potential for hybrid approaches that
detect question complexity before selecting retrieval strategy.
7.4 Deployment Considerations
Target use cases.At 14.1 s end-to-end latency,ReaLM-Retrieve
suits asynchronous workloads (research assistants, batch docu-
ment analysis, agentic multi-step tasks); real-time interactive search

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia. Dongxin Guo, Jikun Wu, and Siu Ming Yiu
(<2s) would require distilled reasoning models, parallel specula-
tive execution, or uncertainty-based early-exit, none of which we
evaluate here.
System architecture and production training.A standard
three-tier deployment (API gateway with deduplication / orches-
tration layer with speculative caching / independently-scaled rea-
soning and retrieval backends) is sufficient; in our pilot, an 8 ×A100
node sustains∼15 queries/minute on 32B models, and a single 128-
core indexing host serves PLAID over 10M passages at 500 req/s.
Without ground-truth labels, three reward sources approximate su-
pervised performance: thumbs-up/down feedback ( ∼89% after 100K
events), self-consistency over multiple chains ( ∼94%,3×cost), and
NLI-based citation verification ( ∼91%,1.2×cost); self-consistency
is the most promising near-term option.
7.5 Limitations
Completion-only models: Performance is 2–3% F1 lower than
open-weight models where richer uncertainty signals are avail-
able (68.9% vs 71.2% F1 on MuSiQue for o1 vs R1).Retrieval cor-
pus dependence: When relevant information is absent, retrieval
wastes computation (171ms avg) and may introduce noise (14% of
out-of-domain retrievals return misleading information).Training
data requirements: The policy requires 19,938 training examples,
though cross-benchmark results suggest 5K–10K suffice for reason-
able performance.Computational overhead: Step boundary and
RSUS computation add 8% inference overhead, with no benefit for
queries answerable without retrieval.Long-context limitations:
For reasoning chains >30K tokens, KV-cache memory becomes
prohibitive despite efficient integration.Multi-retrieval distri-
bution shift: As discussed in §4.4, segmentation and RSUS are
robust against one prior retrieval but degrade after ≥3injections;
in our MuSiQue evaluation only4 .7% of test questions trigger this
regime (measured as the fraction of MuSiQue test questions for
which the trained policy issued ≥3retrievals during inference),
but it would matter more on benchmarks with deeper retrieval
requirements.Multi-component architecture:ReaLM-Retrieve
comprises four learned modules (segmenter, RSUS proxy classifier
where needed, intervention policy, query generator). We consid-
ered simplification but Table 6 shows each component contributes
non-trivially; the multi-component design reflects an engineering
trade-off between modular interpretability and parameter efficiency
rather than incidental complexity.
7.6 Broader Impact
Improving reasoning model accuracy on knowledge-intensive tasks
reduces hallucinations in applications like question answering, re-
search assistance, and decision support, enabling deployment in
high-stakes domains. However, improved reasoning+retrieval could
assist malicious purposes (e.g., misinformation campaigns, vulnera-
bility identification). We encourage responsible deployment with
content filtering, audit logging, and rate limiting.
On framing accuracy as the only impact axis.Treating “hal-
lucination” as the central failure mode (as we do in this work)
narrows the discussion of impact to a single technical axis. A re-
viewer noted that designing methods around completion-only mod-
els (those exposing only generated text, not token-level probabilitiesor attention) implicitly accepts and reinforces a research paradigm
in which the most capable LLMs are commercial products whose
internals are protected as intellectual property. We agree this is
a real cost: our framework is shaped by whatcannotbe assumed
(no logprobs, no internal states), and that constraint comes from
the market rather than from any intrinsic property of reasoning.
To partly offset this, we evaluate on open-weight reasoning mod-
els (R1-Distill-32B, QwQ-32B-Preview) where richer signalsare
available; the framework runs end-to-end on open weights with
stronger results than on completion-only APIs (Table 2). We do not
claim this resolves the broader concern, but it ensures the work is
not solely useful for those with access to proprietary endpoints.
Environmental Impact.Training requires 2,400 GPU-hours (8
A100s, 50K steps), but this one-time cost is amortized across millions
of queries. Per-query energy decreases 14% vs IRCoT through 47%
fewer retrievals and 16% shorter reasoning chains. For deployments
serving 1M queries daily, this translates to meaningful energy sav-
ings, though efficiency gains diminish with frequent corpus index
rebuilding.
8 Conclusion
We presentedReaLM-Retrieve, a reasoning-aware retrieval frame-
work that resolves a temporal mismatch between large reasoning
models and current RAG systems: existing pipelines stage retrieval
before generation, while reasoning chains develop knowledge needs
duringmulti-step inference. Three components close this gap: a
step-level uncertainty score (RSUS) that detects mid-chain knowl-
edge gaps from output tokens alone, a learned intervention policy
that decides when those gaps warrant retrieval, and efficient inte-
gration mechanisms that hold per-call overhead below one second.
Across MuSiQue, HotpotQA, and 2WikiMultiHopQA, this com-
bination yields a mean +10.1% absolute F1 over Single RAG and
+4.6% over ReARTeR [ 28], the strongest prior method for retrieval-
augmented reasoning, while issuing 47% fewer retrieval calls than
IRCoT. The margin widens with question difficulty, reaching +8.4
F1 over IRCoT on 4-hop MuSiQue.
Our central finding, thatfewer, better-timed retrievals outperform
frequent, fixed-interval retrievals, points toward a broader shift:
retrieval systems will need to evolve from pre-generation context
providers into dynamic reasoning partners.
Future directions.End-to-end training of all components would
replace our staged pipeline with a single learned controller. Corpus-
aware policies could skip retrieval where the index lacks relevant
evidence, addressing the 43% of failures from corpus gaps. Multi-
modal extensions are natural where evidence lives in figures or
tables, and the optimal-timing question remains theoretically open.
Acknowledgments
We thank the SIGIR 2026 reviewers and the senior area chair for
their constructive feedback that improved this paper. We grate-
fully acknowledge the support of The University of Hong Kong,
Stellaris AI Limited, and Brain Investing Limited.
References
[1]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.

When to Retrieve During Reasoning: Adaptive Retrieval for Large Reasoning Models SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia.
InThe Twelfth International Conference on Learning Representations, ICLR 2024,
Vienna, Austria, May 7-11, 2024. OpenReview.net.
[2]Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Ruther-
ford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan
Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman
Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer,
Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero,
Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent Sifre. 2022. Improving
Language Models by Retrieving from Trillions of Tokens. InInternational Con-
ference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland,
USA (Proceedings of Machine Learning Research), Kamalika Chaudhuri, Stefanie
Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato (Eds.). PMLR,
2206–2240.
[3]Shahul ES, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024. RAGAs:
Automated Evaluation of Retrieval Augmented Generation. InProceedings of the
18th Conference of the European Chapter of the Association for Computational Lin-
guistics, EACL 2024 - System Demonstrations, St. Julians, Malta, March 17-22, 2024,
Nikolaos Aletras and Orphée De Clercq (Eds.). Association for Computational
Linguistics, 150–158.
[4]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai,
Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented Gen-
eration for Large Language Models: A Survey.arXiv preprintarXiv.2312.10997
(2024). https://arxiv.org/abs/2312.10997
[5]Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu,
Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu,
Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, et al .2025.
DeepSeek-R1 Incentivizes Reasoning in LLMs through Reinforcement Learning.
Nature645 (2025), 633–638. doi:10.1038/s41586-025-09422-z
[6]Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang.
2020. REALM: retrieval-augmented language model pre-training. InProceedings
of the 37th International Conference on Machine Learning (ICML’20). JMLR.org,
Article 368, 10 pages.
[7]Helia Hashemi, Victor Rühle, and Saravan Rajmohan. 2026. Cost-Aware Retrieval-
Augmentation Reasoning Models with Adaptive Retrieval Depth. InProceedings
of the ACM Web Conference 2026, WWW 2026, Dubai, United Arab Emirates,
originally scheduled for April 13-17, 2026, rescheduled for June 29 - July 3, 2026,
Hakim Hacid, Yoelle Maarek, Francesco Bonchi, Ido Guy, and Emine Yilmaz (Eds.).
ACM, 2431–2440. doi:10.1145/3774904.3792693
[8]Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. InProceedings of the 28th International Conference on Computational
Linguistics, COLING 2020, Barcelona, Spain (Online), December 8-13, 2020, Do-
nia Scott, Núria Bel, and Chengqing Zong (Eds.). International Committee on
Computational Linguistics, 6609–6625. doi:10.18653/V1/2020.COLING-MAIN.580
[9]Shayekh Bin Islam, Md. Asib Rahman, K. S. M. Tozammel Hossain, Enamul
Hoque, Shafiq Joty, and Md. Rizwan Parvez. 2024. Open-RAG: Enhanced Retrieval
Augmented Reasoning with Open-Source Large Language Models. InFindings of
the Association for Computational Linguistics: EMNLP 2024, Miami, Florida, USA,
November 12-16, 2024 (Findings of ACL), Yaser Al-Onaizan, Mohit Bansal, and
Yun-Nung Chen (Eds.). Association for Computational Linguistics, 14231–14244.
doi:10.18653/V1/2024.FINDINGS-EMNLP.831
[10] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong Park.
2024. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
Models through Question Complexity. InProceedings of the 2024 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers), NAACL 2024, Mexico City, Mexico,
June 16-21, 2024, Kevin Duh, Helena Gómez-Adorno, and Steven Bethard (Eds.).
Association for Computational Linguistics, 7036–7050. doi:10.18653/V1/2024.
NAACL-LONG.389
[11] Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-
Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active Retrieval
Augmented Generation. InProceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023,
Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational
Linguistics, 7969–7992. doi:10.18653/V1/2023.EMNLP-MAIN.495
[12] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan O Arik, Dong Wang,
Hamed Zamani, and Jiawei Han. 2025. Search-R1: Training LLMs to Reason and
Leverage Search Engines with Reinforcement Learning. InSecond Conference on
Language Modeling. https://openreview.net/forum?id=Rwhi91ideu
[13] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20,
2020, Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association for
Computational Linguistics, 6769–6781. doi:10.18653/V1/2020.EMNLP-MAIN.550
[14] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage
Search via Contextualized Late Interaction over BERT. InProceedings of the 43rd
International ACM SIGIR conference on research and development in InformationRetrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, Jimmy X. Huang,
Yi Chang, Xueqi Cheng, Jaap Kamps, Vanessa Murdock, Ji-Rong Wen, and Yiqun
Liu (Eds.). ACM, 39–48. doi:10.1145/3397271.3401075
[15] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic Uncertainty:
Linguistic Invariances for Uncertainty Estimation in Natural Language Genera-
tion. InThe Eleventh International Conference on Learning Representations, ICLR
2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.
[16] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient
Memory Management for Large Language Model Serving with PagedAtten-
tion. InProceedings of the 29th Symposium on Operating Systems Principles, SOSP
2023, Koblenz, Germany, October 23-26, 2023, Jason Flinn, Margo I. Seltzer, Pe-
ter Druschel, Antoine Kaufmann, and Jonathan Mace (Eds.). ACM, 611–626.
doi:10.1145/3600006.3613165
[17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. InAdvances in Neural Information Processing
Systems 33: Annual Conference on Neural Information Processing Systems 2020,
NeurIPS 2020, December 6-12, 2020, virtual, Hugo Larochelle, Marc’Aurelio Ranzato,
Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (Eds.).
[18] Shuo Li, Sangdon Park, Insup Lee, and Osbert Bastani. 2024. TRAQ: Trustworthy
Retrieval Augmented Question Answering via Conformal Prediction. InProceed-
ings of the 2024 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies (Volume 1: Long Papers),
NAACL 2024, Mexico City, Mexico, June 16-21, 2024, Kevin Duh, Helena Gómez-
Adorno, and Steven Bethard (Eds.). Association for Computational Linguistics,
3799–3821. doi:10.18653/V1/2024.NAACL-LONG.210
[19] Zhen Lin, Shubhendu Trivedi, and Jimeng Sun. 2024. Generating with Confidence:
Uncertainty Quantification for Black-box Large Language Models.Trans. Mach.
Learn. Res.2024 (2024).
[20] Viktor Moskvoretskii, Maria Marina, Mikhail Salnikov, Nikolay Ivanov, Sergey
Pletenev, Daria Galimzianova, Nikita Krayko, Vasily Konovalov, Irina Nikishina,
and Alexander Panchenko. 2025. Adaptive Retrieval Without Self-Knowledge?
Bringing Uncertainty Back Home. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025,
Vienna, Austria, July 27 - August 1, 2025, Wanxiang Che, Joyce Nabende, Ekaterina
Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational
Linguistics, 6355–6384.
[21] OpenAI. 2024. OpenAI o1 System Card.arXiv preprintarXiv.2412.16720 (2024).
https://arxiv.org/abs/2412.16720
[22] Qwen Team. 2024. QwQ: Reflect Deeply on the Boundaries of the Unknown.
https://qwenlm.github.io/blog/qwq-32b-preview/. Blog post.
[23] Keshav Santhanam, Omar Khattab, Christopher Potts, and Matei Zaharia. 2022.
PLAID: An Efficient Engine for Late Interaction Retrieval. InProceedings of the
31st ACM International Conference on Information & Knowledge Management,
Atlanta, GA, USA, October 17-21, 2022, Mohammad Al Hasan and Li Xiong (Eds.).
ACM, 1747–1756. doi:10.1145/3511808.3557325
[24] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei
Zaharia. 2022. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late
Interaction. InProceedings of the 2022 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language Technologies,
NAACL 2022, Seattle, WA, United States, July 10-15, 2022, Marine Carpuat, Marie-
Catherine de Marneffe, and Iván Vladimir Meza Ruíz (Eds.). Association for
Computational Linguistics, 3715–3734. doi:10.18653/V1/2022.NAACL-MAIN.272
[25] Jan Luca Scheerer, Matei Zaharia, Christopher Potts, Gustavo Alonso, and Omar
Khattab. 2025. WARP: An Efficient Engine for Multi-Vector Retrieval. InProceed-
ings of the 48th International ACM SIGIR Conference on Research and Development
in Information Retrieval, SIGIR 2025, Padua, Italy, July 13-18, 2025, Nicola Ferro,
Maria Maistro, Gabriella Pasi, Omar Alonso, Andrew Trotman, and Suzan Ver-
berne (Eds.). ACM, 2504–2512. doi:10.1145/3726302.3729904
[26] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike
Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2024. REPLUG: Retrieval-Augmented
Black-Box Language Models. InProceedings of the 2024 Conference of the North
American Chapter of the Association for Computational Linguistics: Human Lan-
guage Technologies (Volume 1: Long Papers), NAACL 2024, Mexico City, Mexico, June
16-21, 2024, Kevin Duh, Helena Gómez-Adorno, and Steven Bethard (Eds.). Asso-
ciation for Computational Linguistics, 8371–8384. doi:10.18653/V1/2024.NAACL-
LONG.463
[27] Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, and Yiqun Liu. 2024. DRAGIN:
Dynamic Retrieval Augmented Generation based on the Real-time Information
Needs of Large Language Models. InProceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers), ACL
2024, Bangkok, Thailand, August 11-16, 2024, Lun-Wei Ku, Andre Martins, and
Vivek Srikumar (Eds.). Association for Computational Linguistics, 12991–13013.
doi:10.18653/V1/2024.ACL-LONG.702
[28] Zhongxiang Sun, Qipeng Wang, Weijie Yu, Xiaoxue Zang, Kai Zheng, Jun Xu, Xiao
Zhang, Yang Song, and Han Li. 2025. ReARTeR: Retrieval-Augmented Reasoning

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia. Dongxin Guo, Jikun Wu, and Siu Ming Yiu
with Trustworthy Process Rewarding. InProceedings of the 48th International ACM
SIGIR Conference on Research and Development in Information Retrieval, SIGIR
2025, Padua, Italy, July 13-18, 2025, Nicola Ferro, Maria Maistro, Gabriella Pasi,
Omar Alonso, Andrew Trotman, and Suzan Verberne (Eds.). ACM, 1251–1261.
doi:10.1145/3726302.3730102
[29] Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu
Yao, Chelsea Finn, and Christopher D. Manning. 2023. Just Ask for Calibration:
Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-
Tuned with Human Feedback. InProceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-
10, 2023, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for
Computational Linguistics, 5433–5442. doi:10.18653/V1/2023.EMNLP-MAIN.330
[30] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop Questions via Single-hop Question Composition.Trans.
Assoc. Comput. Linguistics10 (2022), 539–554. doi:10.1162/TACL_A_00475
[31] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-
Intensive Multi-Step Questions. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023,
Toronto, Canada, July 9-14, 2023, Anna Rogers, Jordan L. Boyd-Graber, and Naoaki
Okazaki (Eds.). Association for Computational Linguistics, 10014–10037. doi:10.
18653/V1/2023.ACL-LONG.557
[32] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang,
Aakanksha Chowdhery, and Denny Zhou. 2023. Self-Consistency Improves
Chain of Thought Reasoning in Language Models. InThe Eleventh International
Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023.
OpenReview.net.[33] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei
Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-Thought Prompting
Elicits Reasoning in Large Language Models. InAdvances in Neural Information
Processing Systems 35: Annual Conference on Neural Information Processing Systems
2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, Sanmi
Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (Eds.).
[34] Ronald J. Williams. 1992. Simple Statistical Gradient-Following Algorithms
for Connectionist Reinforcement Learning.Mach. Learn.8 (1992), 229–256.
doi:10.1007/BF00992696
[35] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett,
Junaid Ahmed, and Arnold Overwijk. 2021. Approximate Nearest Neighbor
Negative Contrastive Learning for Dense Text Retrieval. In9th International
Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May
3-7, 2021. OpenReview.net.
[36] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. InProceedings of the 2018
Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium,
October 31 - November 4, 2018, Ellen Riloff, David Chiang, Julia Hockenmaier,
and Jun’ichi Tsujii (Eds.). Association for Computational Linguistics, 2369–2380.
doi:10.18653/V1/D18-1259
[37] Qingcheng Zeng, Weihao Xuan, Leyang Cui, and Rob Voigt. 2025. Thinking
Out Loud: Do Reasoning Models Know When They’re Right?. InProceedings
of the 2025 Conference on Empirical Methods in Natural Language Processing,
EMNLP 2025, Suzhou, China, November 4-9, 2025, Christos Christodoulopoulos,
Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (Eds.). Association for
Computational Linguistics, 1394–1407. doi:10.18653/V1/2025.EMNLP-MAIN.73
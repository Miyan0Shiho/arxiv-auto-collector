# SilentRetrieval: Hijacking Retrieval-Augmented Generation via Semantically-Preserving Adversarial Data Poisoning

**Authors**: Jiachen Qian

**Published**: 2026-05-27 07:30:30

**PDF URL**: [https://arxiv.org/pdf/2605.28074v1](https://arxiv.org/pdf/2605.28074v1)

## Abstract
Retrieval-Augmented Generation (RAG) mitigates LLM hallucinations but introduces a critical vulnerability: corpus integrity. We present SilentRetrieval, a two-stage data poisoning attack that hijacks RAG systems through adversarially crafted yet fluent documents. Stage 1 uses Coordinated Beam Search, a multi-token joint optimization method with a fluency-similarity objective, to keep a poisoned host document retrievable while constraining perplexity. Stage 2 uses Context-Adaptive Trigger Generation, a lightweight trigger-fusion step driven by a frozen LLM, to integrate manipulation triggers into document content. Under a one-poisoned-document-per-query evaluation with synthetic target answers, SilentRetrieval achieves 84.6%/81.3% HR@10 and 57.5%/54.8% ASR-LLM on Natural Questions and MS MARCO, while maintaining near-benign perplexity. Cross-model evaluation across four target LLMs shows nontrivial effectiveness under a fixed trigger generator, and transfer tests against unseen retrievers, including ColBERT and commercial embedding models, yield 64.7% average HR@10 under the same injected-corpus protocol. In a sampled Wikipedia-scale evaluation, SilentRetrieval retains 74.2% HR@10 at a 0.016% poisoning ratio. Combined retrieval-side and generation-side defenses reduce attack success substantially but incur a latency trade-off. Human evaluation shows substantially lower flag rates than disfluent baselines, while remaining numerically more suspicious than benign content at the current sample size.

## Full Text


<!-- PDF content starts -->

SilentRetrieval: Hijacking Retrieval-Augmented Generation via
Semantically-Preserving Adversarial Data Poisoning
Jiachen Qian
City University of Hong Kong
Hong Kong, Hong Kong
72510756@cityu-dg.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) mitigates LLM hallucina-
tions but introduces a critical vulnerability: corpus integrity. We
presentSilentRetrieval, a two-stage data poisoning attack that
hijacks RAG systems through adversarially crafted yet fluent docu-
ments. Stage 1 introducesCoordinated Beam Search(CBS), a multi-
token joint optimization with a penalized fluency–similarity objec-
tive that preconditions a topically relevant host document to remain
retrievable after payload insertion while constraining perplexity.
Stage 2 employsContext-Adaptive Trigger Generation(CATG), a
lightweight trigger-fusion step that uses a frozen LLM to generate
triggers contextually integrated with document content. Under a
one-poisoned-document-per-query evaluation with synthetic tar-
get answers,SilentRetrievalachieves84.6%/81.3% HR@10and
57.5%/54.8% ASR-LLMon Natural Questions (NQ, 361K-passage
subset;notthe standard 21M DPR corpus) and MS MARCO (8.8M
passages), while maintaining near-benign perplexity (32.4 vs. 28.4).
Cross-model evaluation across four target LLMs shows nontrivial
effectiveness under a fixed CATG generator (48.6–57.5% ASR-LLM).
Surrogate-transfer evaluation against unseen retrievers—including
ColBERT and rebuilt indexes using commercial embedding models—
yields 64.7% average HR@10 under the same injected-corpus pro-
tocol. In a sampled large-corpus evaluation built from a Wikipedia-
scale 21M-passage construction,SilentRetrievalretains 74.2%
HR@10 at a 0.016% poisoning ratio, characterizing large-corpus
behavior under the sampled protocol. Combined retrieval-side and
generation-side defenses reduce ASR-LLM to 25.6% at a 6 ×latency
trade-off in our evaluated setting, and to 21.3% under the strongest
evaluated configuration; adaptive attacks recover 6.2% HR@10
in the matched MiniLM-L6-v2 reranker setting. Human evalua-
tion (𝑛=600 documents, Krippendorff’s 𝛼=0.74) shows substantially
lower flag rates than disfluent baselines, while remaining numeri-
cally more suspicious than benign content at the current sample
size (𝑝≈0.064).
CCS Concepts
•Security and privacy →Systems security;•Information sys-
tems→Retrieval models and ranking;•Computing method-
ologies→Natural language processing.
This work is licensed under a Creative Commons Attribution 4.0 International License.
KDD ’26, Jeju Island, Republic of Korea
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2259-2/2026/08
https://doi.org/10.1145/3770855.3818186Keywords
Retrieval-Augmented Generation, Data Poisoning, Adversarial At-
tacks, Dense Retrieval, Large Language Models
ACM Reference Format:
Jiachen Qian. 2026. SilentRetrieval: Hijacking Retrieval-Augmented Genera-
tion via Semantically-Preserving Adversarial Data Poisoning. InProceedings
of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Min-
ing V.2 (KDD ’26), August 09–13, 2026, Jeju Island, Republic of Korea.ACM,
New York, NY, USA, 12 pages. https://doi.org/10.1145/3770855.3818186
1 Introduction
Retrieval-Augmented Generation (RAG) mitigates LLM hallucina-
tions and knowledge staleness by grounding generation in external
corpora [ 4,10,17,24]. RAG has become a widely used paradigm
for knowledge-intensive NLP, powering enterprise QA, search, and
conversational AI [ 12,14]. This widespread adoption makes under-
standing RAG’s security implications critically important.
Motivation & Challenge.RAG assumes corpus integrity, yet
knowledge bases are often sourced from publicly editable platforms
(e.g., Wikipedia), web crawls, or third-party providers with limited
verification [ 19]. Since dense retrievers match queries to documents
via embedding similarity [ 11,14], an adversary who can inject docu-
ments into the corpus can hijack retrieval and, consequently, gener-
ation. However, prior adversarial attacks produce easily detectable
outputs—gibberish text or obviously suspicious content [ 8,22]. A
practical RAG poisoning attack mustsimultaneously(1) achieve
retrieval hijacking, (2) manipulate LLM generation, and (3) remain
substantially less suspicious than prior attack documents under
automated filters and human inspection.
Research Gap.Prior corpus poisoning methods face a persistent
trade-off among retrieval effectiveness, generation manipulation,
and stealthiness. While several recent or concurrent methods target
subsets of these desiderata, no evaluated baseline in our unified
single-document protocol demonstrates strong performance across
all three simultaneously. Gradient-based optimization [ 3,23,34]
achieves strong retrieval hijacking (and GASLITE additionally pre-
serves fluency) but omits generation-phase manipulation. Joint
retriever-generator attacks [ 25,35] manipulate generation effec-
tively but produce high-perplexity text (PPL >150) that is highly
exposed to simple PPL filtering; augmenting these methods with flu-
ency constraints remains unexplored. Adversarial decoding [ 32] pre-
serves fluency but operates at generation-time rather than through
corpus poisoning. This gap motivatesSilentRetrieval: a frame-
work that jointly optimizes across all three objectives, combining
fluency-constrained retrieval optimization with context-adaptive
generation manipulation.arXiv:2605.28074v1  [cs.CR]  27 May 2026

KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea Jiachen Qian
Figure 1: Teaser: Stealthy RAG Poisoning Attack. For each
target query, an adversary injects a crafted adversarial doc-
ument into the knowledge corpus. When the correspond-
ing query is issued, the poisoned document is retrieved and
causes the LLM to generate manipulated misinformation,
while appearing substantially less suspicious than disfluent
attack baselines in our human study.
Our Contributions.We presentSilentRetrieval, a data poi-
soning attack against RAG systems that jointly addresses retrieval
hijacking, generation manipulation, and fluency preservation. Our
key contributions:
•Two-Stage Attack Framework.We proposeCoordinated
Beam Search(CBS), a multi-token joint optimization with a
penalized fluency–similarity objective that preconditions a
topically relevant host under a fluency budget before pay-
load insertion (Section 4.1), andContext-Adaptive Trigger
Generation(CATG), which leverages a frozen LLM to pro-
duce triggers contextually integrated with document content
(Section 4.2).
•Comprehensive Evaluation.Experiments on NQ (361K)
and MS MARCO (8.8M) under our unified single-document,
synthetic-target evaluation show the strongest retrieval hi-
jacking among evaluated baselines at near-benign perplexity,
nontrivial effectiveness across four target LLMs under a fixed
CATG generator, sampled large-corpus results up to 21M pas-
sages, surrogate-transfer evidence to unseen retrievers, and
substantially lower human flag rates than disfluent baselines
(𝑛=600 documents, Krippendorff’s𝛼=0.74; Section 5).
•Defense and Adaptive Attack Analysis.We evaluate
retrieval-side (reranking, hybrid retrieval) and generation-
side (passage isolation) defenses, showing that combined
defenses substantially reduce attack success while adaptive
attacks partially recover effectiveness in a matched reranker
setting (Section 6).
2 Related Work
2.1 Corpus Poisoning Attacks
RAG systems augment LLMs with dense retrieval from external
corpora [ 11,14,17], making corpus integrity a critical security
concern. Existing corpus poisoning methods face a persistent trade-
off among retrieval effectiveness, generation manipulation, andstealthiness (see Table 10 in Appendix D for a comprehensive
comparison across 14 methods).Optimization-focusedmethods
(AGGD [ 23], GASLITE [ 3]) achieve strong retrieval hijacking with
fluency awareness but omit generation-phase manipulation.Joint
retriever-generatormethods (Joint-GCG [ 25], PoisonedRAG [ 35])
manipulate generation effectively but produce high-perplexity text
that is highly exposed to simple PPL filtering.Black-boxapproaches
(CorruptRAG [ 30], RIPRAG [ 26], MIRAGE [ 5]) avoid white-box as-
sumptions but with weaker optimization or different threat-model
trade-offs. Our central comparison is therefore not a binary fea-
ture checklist, but whether a method achieves a strong stealth–
effectiveness trade-off under a unified single-document RAG poi-
soning protocol.
2.2 Defenses against RAG Poisoning
RobustRAG [ 27] provides certifiable robustness via isolate-then-
aggregate generation; RSB [ 31] benchmarks 13 attacks against 7
defenses, finding that expanded corpora reduce attack effectiveness.
Other directions include cross-encoder reranking, query augmenta-
tion with answer-consistency checking, and rationale-driven pas-
sage selection [ 1]. Our defense evaluation (Section 6) empirically
covers reranking, hybrid retrieval, and passage isolation.
Positioning.SilentRetrievalcontributes atask-specific
methodological advancerather than a wholly new optimization
primitive: Coordinated Beam Search (CBS) for fluency-constrained
multi-token optimization and Context-Adaptive Trigger Generation
(CATG) as a lightweight LLM-based trigger-fusion step. Compared
to concurrent black-box approaches,SilentRetrievaltrades the
white-box retriever assumption for stronger optimization; transfer
results (Section 5) partially bridge this gap.
3 Problem Formulation
3.1 Notation and Setup
We consider a standard RAG pipeline [ 14,17]: given a user query
𝑞, a dense retriever with encoder 𝐸returns the top- 𝑘documents
D𝑞from corpusD={𝑑 1,...,𝑑𝑛}ranked by embedding similarity
𝑠(𝑞,𝑑𝑖)=sim(𝐸(𝑞),𝐸(𝑑 𝑖)), and a generatorGproduces an answer
𝑎=G(𝑞,D 𝑞).
3.2 Threat Model
Adversary’s Goal.The adversary aims to manipulate the RAG
system’s output for a target query 𝑞target, causing it to generate a
predetermined target answer 𝑎target instead of the correct response.
Adversary’s Capability.The adversary can inject a limited
number of documents into D, modeling scenarios such as editing
public knowledge bases (e.g., Wikipedia), publishing crawlable web
content, or uploading to enterprise systems. A small controlled
reviewer-perception study ( 𝑛=150, Fleiss’ 𝜅=0.68) suggests that flu-
ent adversarial passages are judged less likely to be flagged than
disfluent baselines under a simplified review scenario (78% ofSilen-
tRetrievalpassages rated “likely to survive review” vs. 12% for
Zhong et al.); this should be interpreted as preliminary evidence
about perceived plausibility rather than real-world injection feasi-
bility. Details and limitations are in Appendix A.
Adversary’s Knowledge.We considerwhite-box(full retriever
access for gradient-based optimization) andlimited-access black-box

SilentRetrieval: Hijacking RAG via Semantically-Preserving Poisoning KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea
settings without retriever gradients, which we study through sur-
rogate transfer rather than fully adaptive query-only optimization.
3.3 Attack Objectives
The adversary constructs 𝑑advto satisfy three objectives: (1)Re-
trieval Hijacking: Rank(𝑑 adv,𝑞target)≤𝑘 ; (2)Generation Ma-
nipulation:G(𝑞 target,{𝑑adv}∪D′
𝑞)≈𝑎 target; and (3)Stealthiness:
PPL(𝑑 adv)≤𝜏 ppl, enforcing a perplexity budget calibrated on be-
nign documents as a proxy for fluency.
4 Methodology
We propose a two-stage attack framework that first optimizes docu-
ments for retrieval hijacking while maintaining fluency, then injects
triggers for generation manipulation.
4.1 Stage 1: Coordinated Beam Search (CBS)
Given a seed document 𝑑seedtopically related to the target query
𝑞target, we optimize tokens under a fluency-constrained retrieval
objective. Importantly, Stage 1 is not intended to outperform the
untouched top-1 seed in raw retrievability; rather, it preconditions
the host so that Stage 2 can inject a target-bearing payload without
collapsing rank. Unlike prior single-token greedy approaches [ 8,23],
CBS performsmulti-token joint optimizationthat captures inter-
position dependencies.
Penalized Fluency–Similarity Objective.We formulate re-
trieval optimization as a penalized maximization problem that
trades off embedding similarity against a fluency violation penalty:
𝑑opt=arg max
𝑑∈Tsim(𝐸(𝑞 target),𝐸(𝑑))−𝜆·max(0,PPL(𝑑)−𝜏 ppl)(1)
where𝜆controls the fluency penalty weight (we use 𝜆=0.1). This
relaxation augments the hard PPL rejection used in prior work [ 3]:
retrieval similarity is optimized through gradients, while PPL con-
tributes a heuristic proposal signal and final beam candidates are
filtered by the PPL budget.
Gradient-Guided Candidate Generation.At each step𝑡, we
use gradients to propose candidate token edits, and then score the
resulting discrete candidates with the exact objective in Eq. (1). The
proposal score combines the retriever-similarity gradient with a
heuristic PPL proposal signal:
𝑔𝑖=∇𝑒𝑖sim(𝐸(𝑞 target),𝐸(𝑑(𝑡)))−𝜆e∇PPL
𝑖 (2)
where𝑒𝑖is the retriever-side embedding of token 𝑖in𝑑(𝑡), ande∇PPL
𝑖
denotes a text-aligned proposal signal for the PPL violation penalty
rather than an exact gradient in the retriever embedding space.
Since the retriever and PPL model use different tokenizers, this
signal is used only for candidate proposal; final beam decisions are
made by exact re-evaluation of generated candidates under Eq. (1)
(details in Appendix B). For each position 𝑖, we project the proposal
vector onto the vocabulary embedding matrix 𝑊𝑉:𝑠𝑖,𝑗=𝑔⊤
𝑖𝑊𝑉[𝑗],
and retain the top-𝐾candidates (default𝐾=10).
Coordinated Multi-Token Beam Search.The key novelty of
CBS is simultaneous perturbation of multiple correlated positions.
At each iteration, we:
(1)Rank all positions by gradient magnitude ∥𝑔𝑖∥and select the
top-𝑀positions (default𝑀=3).(2)Construct a beam of width 𝐵= 5over the joint space of
𝑀×𝐾candidate replacements.
(3)For each beam candidate (a set of 𝑀simultaneous token
replacements), evaluate the full objective including PPL via
GPT-2 [20].
(4)Select the beam candidate with highest objective value; reject
if PPL exceeds𝜏 ppl=50.
This multi-position coordination captures token interdependencies
that single-token greedy methods miss—for example, replacing a
noun often requires adjusting its associated adjective and verb to
maintain fluency.
Perplexity Computation.Document perplexity is measured
by GPT-2:
PPL(𝑑)=exp 
−1
|𝑑||𝑑|∑︁
𝑖=1log𝑃(𝑤𝑖|𝑤<𝑖)!
(3)
We set𝜏ppl=50from the 95th percentile of NQ benign docu-
ment perplexities (mean: 28.4, std: 12.7, 95th: 48.6), and reuse it on
MS MARCO as a fixed cross-dataset attack budget.1
Seed Document Selection.Seed documents are automatically
selected by retrieving the top-1 passage from the corpus for each
target query and length-normalizing to 100 ±20 tokens. We evaluate
sensitivity to seed selection in Section 6.7.
Convergence.Optimization proceeds for 𝑇= 500iterations
with early stopping when the objective improvement falls below
𝜖=10−4for 20 consecutive steps.
Optimization Dynamics.Empirically, CBS converges within
300–400 iterations for 87% of documents; see Appendix C for con-
vergence analysis.
4.2 Stage 2: Context-Adaptive Trigger
Generation (CATG)
Rather than appending fixed trigger templates [ 35], CATG lever-
ages a frozen LLM to generate triggers that are contextually inte-
grated with the optimized document’s content. In our ablation, this
lightweight fusion step modestly improves ASR-LLM over static
templates while preserving comparable perplexity.
LLM-Based Trigger Synthesis.Given the optimized document
𝑑optand target answer 𝑎target, we prompt a frozen LLM M(Llama-
2-7B-Chat) to generate a context-adaptive trigger:
trigger=M(prompt(𝑑 opt,𝑎target))(4)
where the prompt instructs Mto “Continue the following
passage with a sentence that naturally incorporates
the claim: {𝑎 target }. Match the passage’s style, domain
vocabulary, and narrative voice. Passage: {𝑑 opt}”. The
final adversarial document is:
𝑑adv=𝑑opt⊕trigger (5)
Although CATG does not explicitly optimize the retrieval objective,
it is conditioned on 𝑑optrather than the raw seed. The two stages are
therefore coupled: CBS reshapes the lexical and semantic anchors
1We additionally report Llama-2-7B PPL as an auxiliary metric. On a 500-document
subsample, Spearman 𝜌=0.84 with GPT-2 PPL; 91.3% of accepted documents also pass
the Llama-2-7B threshold.

KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea Jiachen Qian
Algorithm 1SilentRetrieval: Two-Stage RAG Poisoning Attack
Require: Query𝑞𝑡, answer𝑎𝑡, seed𝑑seed, encoder𝐸, threshold𝜏ppl,
LLMM
Ensure:Adversarial document𝑑 adv
1:// Stage 1: CBS—𝑑(0)←𝑑 seed
2:for𝑡=0to𝑇−1do
3:𝑔𝑖←∇𝑒𝑖sim(𝐸(𝑞𝑡),𝐸(𝑑(𝑡)))−𝜆e∇PPL
𝑖,∀𝑖
4: Select top-𝑀positions by∥𝑔𝑖∥; get top-𝐾proposal candi-
dates per position via𝑔⊤
𝑖𝑊𝑉
5: Beam search ( 𝐵=5):𝑑best←arg max 𝑐∈Bobj(𝑐) using Eq. (1)
6:𝑑(𝑡+1)←𝑑 bestif PPL(𝑑 best)≤𝜏 ppl, else𝑑(𝑡)
7:ifconverged (Δobj<𝜖for 20 steps)then
8:break
9:end if
10:end for
11:// Stage 2: CATG—𝑑 opt←𝑑(𝑡+1)
12:Sample 5 triggers via M(prompt(𝑑 opt,𝑎𝑡)); filter by entailment,
coherence, PPL
13:𝑑 adv←𝑑 opt⊕triggerbest
14:return𝑑 adv
that CATG continues from, which in turn changes the embedding
drift induced by the added payload.
Semantic Coherence Filtering.We filter generated triggers
through three quality gates: (1)Entailment check: the trigger must
entail𝑎target (verified via NLI model, threshold >0.8). (2)Coherence
score: cosine similarity between 𝑑optand trigger embeddings must
exceed 0.7, encouraging topical consistency. (3)PPL gate: the con-
catenated document 𝑑advmust satisfy PPL(𝑑 adv)≤𝜏 pplunder the
attacker-side scoring pipeline used during trigger selection. We
sample up to 5 candidate triggers and select the one with highest
coherence score passing all gates.
Authoritative Framing.The prompt encourages authoritative
phrasing (“verified sources”, “research confirms”) that leverages
LLMs’ tendency to defer to confident language in retrieved context.
Unlike static templates, CATG adapts framing to the document’s
domain—e.g., producing citation-style language for academic pas-
sages and journalistic language for news articles. We compare
CATG against static templates in our ablation study (Section 6.7).
4.3 Algorithm Summary
Algorithm 1 summarizes the completeSilentRetrievalpipeline.
5 Experiments
5.1 Experimental Setup
Datasets.We evaluate on two benchmarks: (1)Natural Ques-
tions (NQ)[ 15] with 3,452 test queries and a 361K-passage corpus;2
(2)MS MARCO[ 2] with 6,980 dev queries and the full 8.8M passage
corpus. The two datasets span factoid (Wikipedia) and web-search
query types. For controlled evaluation, each query is paired with
a synthetic target answer rather than a real-world poisoned fact.
2The 361K corpus combines DPR positive/hard-negative passages ( ∼109K) with strat-
ified Wikipedia samples, providing a controlled retrieval setting for rapid iteration.
This is smaller than the standard 21M DPR corpus; scalability to 21M is analyzed in
Table 8 (74.2% HR@10).Unless otherwise stated, “one poisoned document per query” means
that each target query is paired with its own generated adversarial
document; corpus-level poison ratios are computed when the full
set of target-query-specific poisoned documents is injected into the
evaluation corpus.
Models.For retrieval, we use Contriever [ 11] as the primary
dense retriever. For generation, we evaluate four LLMs: Llama-2-
7B-Chat [ 24] (primary), GPT-3.5-Turbo, Mistral-7B-Instruct-v0.2,
and Qwen-7B-Chat, to assess cross-model attack generalizability.
Baselines.We compare against:
•Zhong et al.[ 34]: Gradient-based corpus poisoning via
HotFlip-style [ 8] token substitution optimized for retrieval
hijacking, without fluency constraints or generation-phase
manipulation
•PoisonedRAG[ 35]: Retrieval poisoning with trigger-based
generation manipulation
•GASLITE[ 3]: SEO-style corpus poisoning with fluency-
aware multi-coordinate ascent
•Joint-GCG[ 25]: Unified gradient-based attack jointly opti-
mizing retrieval and generation
Re-implementation Protocol.All baselines are re-implemented
under our unified protocol (single poisoned document per query,
synthetic target answers, Contriever retriever, Llama-2-7B-Chat
generator, GPT-4 judge) on our 361K NQ corpus. These controlled
comparison numbers are therefore not direct substitutes for head-
line results reported in original papers: they differ due to single vs.
multi-document poisoning, synthetic targets, string-match vs. LLM-
judge ASR, and corpus size differences.3For retrieval-only baselines,
downstream ASR reflects exposure to the common generator–judge
pipeline rather than a standalone generation-control module. We
report the main protocol details needed to support reproduction.
Metrics.We report:
•HR@k: Hit Rate at 𝑘—percentage of queries where adver-
sarial document appears in top-𝑘results
•ASR-SM: Attack Success Rate via string matching—target
answer appears as a substring in the generated output (case-
insensitive, after whitespace normalization)
•ASR-LLM: Attack Success Rate via LLM-as-judge [ 33]—GPT-
4 evaluates if the outputgenuinely endorsesthe target an-
swer.4
•ASR-P: Attack Success Rate Persistence—target answer en-
dorsed across 3 follow-up query rephrasings (ASR-P ≤ASR-
LLM by construction).
•PPL: Perplexity of adversarial documents measured by both
GPT-2 (PPL-G2, primary) and Llama-2-7B (PPL-L2, auxiliary).
See Section 4.1 for calibration discussion
5.2 Main Results
Table 1 presents results on both datasets (bootstrap 95% CIs from
1,000 resamples; significance is computed with paired bootstrap
3For example, PoisonedRAG reports 97% ASR with 5 documents and substring match-
ing [ 35]; our protocol yields 48.2% ASR-LLM. Joint-GCG reports near-100% ASR [ 25];
ours yields 62.8% ASR-LLM. Concurrent methods (CorruptRAG [ 30], RIPRAG [ 26],
EYES-ON-ME [ 6], MIRAGE [ 5], POISONCRAFT [ 21]) are excluded due to code
availability.
4Inter-run std: 0.3% over 3 evaluations on 500 queries. GPT-4 vs. Claude-3-Opus
agreement: Cohen’s𝜅=0.81. We report prompts and aggregate evaluation protocols.

SilentRetrieval: Hijacking RAG via Semantically-Preserving Poisoning KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea
Table 1: Main results on NQ (361K corpus, 3,452 queries) and
MS MARCO (8.8M corpus, 6,980 queries). One poisoned doc-
ument per query; unified protocol (Contriever, Llama-2-7B-
Chat, GPT-4 judge). PPL-G2/L2 = GPT-2/Llama-2 perplexity
(NQ benign: 28.4/22.7; MARCO benign: 30.2/24.1).
Dataset Method HR@10 ASR-SM ASR-LLM PPL-G2 PPL-L2
NQZhong et al. 72.3 51.2 38.4 847.2 612.4
PoisonedRAG 78.6 64.3 48.2 412.6 287.3
GASLITE 76.8 58.7 44.1 38.6 31.2
Joint-GCG 81.2 78.4 62.8 156.3 118.7
SilentRetrieval(Ours) 84.668.9 57.532.4 26.1
MARCOZhong et al. 67.8 46.3 34.1 912.4 678.3
PoisonedRAG 73.4 59.1 44.7 478.3 312.6
GASLITE 72.1 53.8 40.6 41.3 33.8
Joint-GCG 76.4 72.6 58.3 173.2 131.4
SilentRetrieval(Ours) 81.364.2 54.833.1 27.4
Zhong
et al.Poisoned
RAGGASLITE Joint-
GCGSilent
Retrieval020406080100Percentage (%)72.378.676.881.284.6
38.448.2
44.162.8
57.5Attack Effectiveness
HR@10 ASR-LLM
Zhong
et al.Poisoned
RAGGASLITE Joint-
GCGSilent
Retrieval101102103PPL-G2 (log scale)
Benign 28.4τppl=50847.2
412.6
38.6156.3
32.4Perplexity Comparison
Figure 2: Attack Effectiveness and Perplexity Comparison.
Left: HR@10 and ASR-LLM (LLM-as-judge) across retrieval
attack methods on NQ (361K corpus). Right: Perplexity (PPL)
comparison on a log scale; lower is better.SilentRetrieval
achieves the highest HR@10 (84.6%) with competitive ASR-
LLM (57.5%) while maintaining near-benign perplexity (32.4
vs. 28.4 benign). Joint-GCG achieves higher ASR-LLM (62.8%)
but at 4.8×higher perplexity.
tests over queries). On NQ,SilentRetrievalachieves the highest
HR@10 (84.6%, CI: [83.2%, 86.0%], 𝑝<0.01 vs. Joint-GCG) with
near-benign perplexity (PPL-G2: 32.4 vs. 28.4 benign). Joint-GCG
achieves higher ASR-LLM (62.8%) but at 4.8 ×higher perplexity,
making it much more exposed under our simple independent PPL
audit (94.2% detection at 𝜏=50 vs. 8.7% forSilentRetrieval; Ta-
ble 11 in Appendix). Compared to GASLITE—the strongest PPL-
comparable baseline—SilentRetrievalachieves +7.8% HR@10
(𝑝<0.01) and +13.4% ASR-LLM ( 𝑝<0.001), demonstrating CBS’s
advantage over multi-coordinate ascent under comparable fluency.
Results on MS MARCO (8.8M passages, 24 ×larger) preserve the
same method ranking, with broadly similar but dataset-dependent
decreases across methods, consistent with a corpus-level dilution
effect.SilentRetrieval’s stealth-aware Pareto efficiency (PES; Ap-
pendix I) is highest among the evaluated baselines on both datasets.
5.3 Cross-Model Generalizability
We evaluateSilentRetrieval’s effectiveness across four target
LLMs to assess whether the attack generalizes beyond Llama-2-7B-
Chat.Table 2: Cross-model evaluation on NQ (361K corpus). ASR
metrics vary by generator robustness; retrieval is fixed across
target LLMs.
Target LLM ASR-SM ASR-LLM ASR-P
Llama-2-7B-Chat 68.9 57.5 38.9
Mistral-7B-Instruct 65.7 54.2 35.8
Qwen-7B-Chat 61.3 48.6 31.4
GPT-3.5-Turbo 59.4 51.7 36.1
Average63.8 53.0 35.6
Table 3: Surrogate-transfer attack results. Surrogate: Con-
triever. Target systems include open-source retrievers (bi-
encoder and late-interaction) and commercial embedding
models.
Target System Type HR@10 ASR-SM ASR-LLM
Contriever (white-box) Bi-enc. 84.6 68.9 57.5
DPR Bi-enc. 72.8 55.4 44.1
BGE-base Bi-enc. 67.3 50.2 40.6
ColBERTv2 Late-int. 61.4 47.1 37.2
text-embedding-ada-002 Commercial 64.7 49.6 39.8
Cohere embed-v3 Commercial 57.2 43.8 34.6
Average (transfer)64.7 49.2 39.3
Table 2 shows thatSilentRetrievalmaintains nontrivial ef-
fectiveness across all four LLMs (ASR-LLM: 48.6–57.5%). Llama-2-
7B-Chat shows the highest ASR-LLM (57.5%), which is consistent
with partial trigger–generator affinity because it is also used for
CATG. Qwen-7B-Chat shows the lowest ASR-LLM (48.6%); this
may reflect differences in instruction tuning, safety behavior, or
how the model resolves conflicts between retrieved context and
parametric knowledge. GPT-3.5-Turbo exhibits lower ASR-SM but
relatively higher ASR-LLM, suggesting that it may endorse target
claims in less extractive, more paraphrased forms under our judge;
controlled model-family comparisons would be needed to isolate
this effect.
5.4 Surrogate Transfer
We evaluateSilentRetrieval’s transferability in a surrogate-
transfer setting where the adversary optimizes against a surrogate
retriever and then attacks a different target system without access
to target retriever gradients. For commercial embedding models,
we fix the generated adversarial documents, rebuild the target
index with the same corpus and injected passages, and evaluate
retrieval with the same top-𝑘and generation-judge protocol.
Table 3 shows thatSilentRetrievalachieves 64.7% average
HR@10 under surrogate transfer across five diverse retrievers
in this fixed-corpus transfer protocol. DPR shows the strongest
transfer (72.8% HR@10) due to architectural similarity with Con-
triever, while ColBERTv2’s late-interaction architecture is more
resilient (61.4%). Attacks retain nontrivial effectiveness across all
systems (34–44% ASR-LLM), but these results should be interpreted

KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea Jiachen Qian
Table 4: Retrieval-side defense evaluation on NQ. Top:
pipeline configurations with MiniLM-L6-v2 reranker. Bot-
tom: reranker scale comparison.
Pipeline HR@5 HR@10 ASR-LLM Latency
Bi-encoder only 76.8 84.6 57.5 —
+ Cross-encoder reranker 62.4 73.2 46.8 +120ms
Hybrid BM25 + Dense 68.1 79.2 52.3 —
+ Cross-encoder reranker 54.7 67.3 41.5 +120ms
Reranker scale comparison (bi-encoder + reranker)
MiniLM-L6-v2 (22M) — 73.2 — +120ms
MiniLM-L12-v2 (33M) — 68.9 — +180ms
MonoT5-base (220M) — 68.4 — +340ms
as surrogate-transfer evidence rather than fully black-box adaptive
optimization.
6 Defense Evaluation
We evaluateSilentRetrievalagainst both retrieval-side defenses
(cross-encoder reranking, hybrid retrieval) and generation-side de-
fenses (passage isolation with majority voting). We further evaluate
adaptive attacks that incorporate defense-aware optimization. A
layered defense architecture overview is summarized in Appen-
dix E.
6.1 Cross-Encoder Reranking
Cross-encoder rerankers [ 9,18] compute fine-grained relevance
scores by jointly encoding query-document pairs, potentially de-
tecting adversarially optimized passages.
Results.Table 4 shows that cross-encoder reranking decreases
HR@10 by 11.4% (84.6% →73.2%) and ASR-LLM by 10.7%. Combin-
ing hybrid retrieval with reranking achieves the strongest mitiga-
tion (HR@10: 67.3%, ASR-LLM: 41.5%). Defender-side scale com-
parison shows that stronger rerankers further reduce HR@10 from
73.2% with MiniLM-L6-v2 to 68.9% with MiniLM-L12-v2 and 68.4%
with MonoT5-base, at higher latency cost. We therefore view larger
rerankers as a stronger defense rather than a failure case, but note
that the adaptive study below is evaluated only against MiniLM-L6-
v2. We recommend deploying reranking as a defense layer (+100–
200ms overhead).
6.2 Adaptive Attack against Rerankers
We implement an adaptive attack that incorporates cross-encoder
scores into the CBS optimization objective:
𝑑opt=arg max
𝑑𝜆1𝑠bi(𝑞,𝑑)+𝜆 2𝑠ce(𝑞,𝑑)−𝜆 3·max(0,PPL(𝑑)−𝜏 ppl)
(6)
where𝑠biand𝑠ceare bi-encoder and cross-encoder scores, respec-
tively. The adaptive objective uses the same cross-encoder family as
the defender-side reranker. We select the loss weights on a held-out
validation set to maximize HR@10 under reranking while main-
taining the fluency constraint. For the hybrid+reranker adaptive
setting, we additionally incorporate a lightweight lexical score term
to encourage overlap that survives hybrid retrieval.Table 5: Adaptive attack results against cross-encoder rerank-
ing defense. Δdenotes recovery from the non-adaptive attack
under the same defense.
Configuration HR@10 ASR-LLM PPL Time/doc
No defense (standard) 84.6 57.5 32.4 15min
+ Reranker (non-adaptive) 73.2 46.8 32.4 15min
+ Reranker (adaptive) 79.4 51.2 34.8 42min
ΔRecovery +6.2 +4.4 +2.4 +27min
+ Hybrid + Reranker (non-adaptive) 67.3 41.5 32.4 15min
+ Hybrid + Reranker (adaptive) 71.6 44.1 35.7 51min
ΔRecovery +4.3 +2.6 +3.3 +36min
Table 6: Generation-side defense evaluation. Passage isola-
tion uses top- 𝑘retrieved passages; HR@5/HR@10 report
pre-isolation retrieval exposure.
Defense Configuration HR@5 HR@10 ASR-LLM Latency
No defense (standard,𝑘=5) 76.8 84.6 57.5 1×
Passage isolation + vote (𝑘=5) 76.8 84.6 38.2 5×
Passage isolation + vote (𝑘=10) 76.8 84.6 31.4 10×
Combined: Hybrid + Reranker + Isolation (𝑘=5) 54.7 67.3 25.6 6×
Combined: All defenses (𝑘=10) 54.7 67.3 21.3 11×
Table 5 shows that the adaptive attack recovers 6.2% HR@10
against the matched MiniLM-L6-v2 reranker defense (73.2% →79.4%)
and 4.3% against the combined defense (67.3% →71.6%), at the cost
of∼3×longer optimization time (due to cross-encoder inference)
and moderately degraded fluency (+2.4/+3.3 PPL). The smaller
recovery in the combined setting reflects the additional difficulty of
simultaneously satisfying bi-encoder, BM25, and cross-encoder ob-
jectives within the PPL budget. These results suggest an arms-race
pattern in the matched MiniLM-L6-v2 setting; extending adaptive
attacks to larger rerankers remains future work.
6.3 Generation-Side Defense: Passage Isolation
Inspired by RobustRAG [ 27], we implement a passage isolation
defense that processes each retrieved passage independently before
aggregating responses via majority voting.
Defense Mechanism.For each retrieved passage 𝑑𝑖∈ D𝑞,
the LLM generates an independent answer 𝑎𝑖=G(𝑞,{𝑑 𝑖}).
The final answer is determined by majority voting: 𝑎final=
MajorityVote({𝑎 1,...,𝑎𝑘}), where semantic similarity (cosine
>0.85) is used to cluster equivalent answers.
Results.Table 6 shows that passage isolation is effective at
reducing generation manipulation in our evaluated setting: ASR-
LLM drops from 57.5% to 38.2% with 𝑘=5 (33.6% relative reduction)
and to 31.4% with 𝑘=10. Combining all defenses (hybrid retrieval +
reranker + passage isolation at 𝑘=10) reduces ASR-LLM to 21.3%,
though at significant latency cost (11 ×). Since isolation is post-
retrieval, HR@5 indicates exposure within the 𝑘=5 generator input
and HR@10 is shown for retrieval-side comparability. The com-
bined defense at 𝑘=5 reduces ASR-LLM to 25.6% at 6 ×latency, a
potentially practical trade-off point within this evaluated configu-
ration.

SilentRetrieval: Hijacking RAG via Semantically-Preserving Poisoning KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea
Table 7: Human evaluation results ( 𝑛=600 documents, 10
annotators, Krippendorff’s 𝛼=0.74). Ratings on 1–5 scale
(lower=better for suspiciousness); document-level aggregated
suspiciousness ratings≥4 are flagged.
Doc Type Fluency↑Coher.↑App. Plaus.↑Susp.↓Flag%
Benign 4.52 4.41 4.38 1.42 7.3%
Zhong et al. 1.87 2.14 2.31 4.24 83.3%
PoisonedRAG 2.83 2.96 3.12 3.58 56.7%
SilentRetrieval4.03 3.84 3.71 1.94 14.7%
6.4 Detection-Based Defenses
SilentRetrievalachieves substantially lower detection rates (8.7–
34.2%) than prior methods (67–94% for Zhong et al.) under the
defenses we evaluate, with an independent PPL audit at the same
threshold (𝜏=50) detecting only 8.7%. Here, “independent” denotes
a defender-side audit computed by a separately implemented full-
passage scoring pipeline rather than the attacker-side CATG accep-
tance score in Section 4.2; it can still catch normalization, tokeniza-
tion, scoring-window, or passage-boundary cases that pass attacker-
side quality control. Learned detectors (a simplified RAGuard-style
detector and Mahalanobis scoring) show moderate detection (31.6–
34.2%), suggesting that fluency-preserving attacks can often pass
simple fluency filters and may require stronger detection strate-
gies. Full detection results are in Table 11 (Appendix F). Remain-
ing untested defenses include query augmentation with answer-
consistency checking and domain-tuned detectors [1, 31].
6.5 Human Evaluation
We conducted a human study to assess whether adversarial docu-
ments are distinguishable from benign content under document-
level aggregation, and to compare their flag rates against disfluent
baselines.
Study Design.10 annotators with NLP research experience
evaluated 600 documents (150 per condition: benign, Zhong et
al., PoisonedRAG,SilentRetrieval) in randomized order. Each
document was rated along four dimensions on a 1–5 scale: (1)
fluency, (2)coherence, (3)apparent factual plausibility(whether the
passage appears fact-like to a reader, not whether the target claim
is true), and (4)overall suspiciousness. We report 𝑛as the number of
documents rather than the number of individual ratings, aggregate
annotator ratings at the document level, and flag a document when
its aggregated suspiciousness rating is ≥4. This design is powered
to detect larger document-level flag-rate differences but should not
be interpreted as proving indistinguishability from benign content.
Table 7 shows thatSilentRetrievaldocuments achieve flag
rates (14.7%) substantially lower than PoisonedRAG (56.7%) and
Zhong et al. (83.3%). The difference betweenSilentRetrieval
(14.7%) and benign documents (7.3%) is not statistically significant
at𝛼= 0.05(𝑝≈0.064, Fisher’s exact test on document-level ag-
gregated flags; 150 benign vs. 150SilentRetrievaldocuments),
though the trend suggests that a larger sample may reveal a de-
tectable gap. Across all four dimensions,SilentRetrievalratings
are directionally closer to benign documents than the disfluent base-
lines, but they still retain moderate gaps relative to benign content:
fluency (4.03 vs. 4.52, Δ=0.49), coherence (3.84 vs. 4.41, Δ=0.57),Table 8: Scalability analysis: attack effectiveness vs. corpus
size. One adversarial document is generated for each of the
3,452 NQ target queries and injected into the corpus when
computing the poison ratio. ΔHR@10 denotes absolute drop
from the 100K baseline.†Rows marked with†report retrieval
and generation metrics on a 1,500-query stratified evaluation
subset; 95% CI≈±2–2.5%.
Corpus Size Poison % HR@10 ASR-LLMΔHR@10
100K 3.45% 86.4 59.8 —
361K (main) 0.96% 84.6 57.5−1.8
500K 0.69% 83.6 56.2−2.8
1M 0.35% 81.4 53.4−5.0
5M†0.069% 79.3 49.1−7.1
10M†0.035% 77.8 46.2−8.6
21M†(sampled eval) 0.016% 74.2 41.5−12.2
†Metrics are evaluated on 1,500 stratified sampled queries, while the poison
percentage is computed from the full set of 3,452 injected adversarial documents. In
this sampled construction, both HR@10 and ASR-LLM show an overall decline with
corpus size, reflecting increased retrieval competition.
apparent plausibility (3.71 vs. 4.38, Δ=0.67), and suspiciousness
(1.94 vs. 1.42, Δ=0.52). Inter-annotator agreement (Krippendorff’s
𝛼=0.74) indicates substantial agreement and supports the reliability
of the aggregated ratings.
6.6 Scalability Analysis
We evaluateSilentRetrievalon corpora of varying sizes (100K to
21M passages) to assess large-corpus behavior. These experiments
are designed to characterize dilution trends under our sampled
Wikipedia-scale construction rather than to replicate the standard
full-corpus DPR evaluation end-to-end.
Corpus Construction.For scalability experiments, we sample
from the full Wikipedia dump (21M passages) while maintaining
approximately uniform query-to-relevant-passage ratios across cor-
pus sizes. The 100K pilot uses a downsampled controlled corpus;
for𝑁≥ 361K, we include the full 361K controlled NQ corpus and
add category-stratified Wikipedia samples to reach𝑁.
Results.Table 8 reports attack effectiveness across corpus scales.
Analysis.Attack effectiveness declines under the sampled scal-
ing protocol: HR@10 drops from 86.4% (100K) to 74.2% (21M, 95% CI:
[72.2%, 76.2%]), a 12.2% absolute decrease across ∼210×corpus ex-
pansion. ASR-LLM declines more than HR@10 ( −18.3% vs.−12.2%),
suggesting legitimate context may dilute adversarial influence even
when poisoned documents are retrieved. This pattern is broadly
consistent with RSB [ 31] findings, indicating that corpus expansion
provides partial natural defense but is insufficient as a standalone
strategy given the 74.2% HR@10 observed at 21M.
6.7 Ablation Study
We conduct ablation experiments to validate the contribution of
each component inSilentRetrieval.
Table 9 validates each component’s contribution. Removing
Stage 2 leaves a retrieval-only host with 21.3% incidental ASR-
LLM under the original competing context, but this is not an active

KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea Jiachen Qian
100K 361K 500K 1M5M†10M†21M†
Corpus Size70.072.575.077.580.082.585.087.590.0HR@10 (%)
Main
setting~210x corpus expansion
12.2% HR@10 drop
† 1,500 stratified queries; 95% CI about +/-2--2.5%.Attack Effectiveness vs. Corpus Size
HR@10
ASR-LLM
Sampled eval
4045505560
ASR-LLM (%)
Figure 3: Scalability Analysis. Attack effectiveness vs. cor-
pus size (100K–21M). HR@10 drops 12.2% absolute while
ASR-LLM drops 18.3% across ∼210×corpus expansion. ASR-
LLM degrades faster, reflecting dilution from competing le-
gitimate context. Results for ≥5M based on 1,500 stratified
queries (95% CI≈±2–2.5%).
Table 9: Ablation study on NQ (361K corpus, 3,452 queries).
Each row removes or modifies one component from the
fullSilentRetrievalpipeline. All values are means across
queries; bootstrap 95% CIs for HR@10 are ±1.5–2.0% (omitted
for space).
Configuration HR@10 ASR-LLM PPL
SilentRetrieval(full: CBS + CATG) 84.6 57.5 32.4
Stage ablation
CBS host only (retrieval-only, no target payload) 84.6 21.3 30.8
Seed document + CATG (no CBS) 38.2 14.7 26.1
CBS vs. single-token greedy (Stage 1)
Single-token greedy + PPL reject 79.8 53.1 33.7
CBS (𝑀=3,𝐵=5, default) 84.6 57.5 32.4
CATG vs. static template (Stage 2)
Best static template 84.6 54.2 33.1
CATG (default) 84.6 57.5 32.4
PPL threshold sensitivity
𝜏ppl=3076.3 49.2 24.7
𝜏ppl=50(default) 84.6 57.5 32.4
𝜏ppl=8087.9 59.1 53.7
generation-manipulation mechanism and cannot replace a target-
bearing payload. Directly applying CATG to the original seed doc-
ument (i.e., no CBS) yields only 38.2% HR@10 and 14.7% ASR-LLM.
Because the untouched seed is selected as the top-1 host and car-
ries no malicious claim, it is not the relevant baseline for the final
attack object. The key comparison is between the payload-bearing
variants: Seed document + CATG (no CBS) collapses to 38.2%
HR@10, whereas full CBS + CATG preserves 84.6% HR@10, with
theCBS host only row confirming that Stage 1 preserves retriev-
ability before payload insertion. Rows that only vary Stage 2 reuse
20 40 60 80 150 300
Adversarial Document PPL (Perplexity)75808590HR@10 (%)
48.6Benign range: mean 28.4; 95th pct 48.6
τ=30τ=40τ=50τ=60τ=80 τ=∞
Selected
threshold
τppl settings
Selected (τ=50)
No PPL constraint
Benign PPL rangeFigure 4: PPL Threshold Trade-off. HR@10 vs. adversarial
document PPL at different 𝜏pplvalues. The shaded region
indicates PPL comparable to benign documents (mean 28.4,
95th percentile 48.6). 𝜏ppl=50(star) balances effectiveness
and stealthiness. The figure includes additional sweep points
beyond the three thresholds summarized in Table 9.
the same CBS-optimized host, so HR@10 can remain unchanged
while ASR-LLM changes. Thus, Stage 1 is better understood as
payload-robust preconditioning: it may sacrifice some of the raw
seed’s original rank, but it preserves the retrievability of the final
adversarial document after trigger insertion. This also explains why
a locally relevant trigger can still hurt dense retrieval: rank depends
on the embedding of the entire passage, so semantic fit of the added
sentence does not guarantee global rank stability. CBS outperforms
single-token greedy by +4.8% HR@10 at comparable PPL. CATG
improves ASR-LLM by +3.3% over the best static template. For
the PPL threshold (Figure 4), 𝜏ppl=50balances effectiveness and
stealthiness: tightening to 30 costs 8.3% HR@10, while relaxing to
80 gains only 3.3%. Additional ablations (CBS settings, iteration
count, seed sensitivity, multi-query attacks) are in Appendix H.
Computation Cost.SilentRetrievalis more efficient than
the joint optimization baseline while achieving the best HR@10,
requiring roughly2 .3×less per-document optimization time than
Joint-GCG in our implementation. A compact cost comparison is
in Table 12 (Appendix G).
6.8 Qualitative Case Studies
We summarize three case studies (entertainment, medical, science)
in Appendix J. Key observations: CATG-generated triggers natu-
rally integrate domain-specific language, producing fluent adver-
sarial documents across domains. A notablefailure modeoccurs for
well-established factual constants (e.g., speed of light), where the
LLM’s parametric knowledge overrides adversarial context ( ∼18%
of ASR-LLM failures)—consistent with prior observations on knowl-
edge conflict resolution [28].

SilentRetrieval: Hijacking RAG via Semantically-Preserving Poisoning KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea
7 Ethical Considerations
This work documents RAG vulnerabilities to motivate stronger
defenses and follows a responsible-disclosure posture. All exper-
iments are derived from standard academic benchmarks and use
synthetic targets—no real-world knowledge bases were poisoned.
Concrete defense recommendations are provided in Section 6; a
detailed ethical discussion is in Appendix K.
8 Limitations
•Comparison & Methodological Scope.Our four baselines
are re-implemented under a unified protocol (single docu-
ment, LLM judge, 361K corpus) that differs from original
settings; several concurrent black-box methods are absent
due to code availability. CBS requires white-box retriever
gradients—a stronger assumption; our transfer results (64.7%
HR@10) rely on simple surrogate transfer. The underlying
optimization builds on established gradient-guided substi-
tution [ 3,8]; augmenting Joint-GCG with a PPL penalty for
iso-fluency comparison is left to future work.
•Evaluation Scope.We evaluate on two datasets (NQ,
MS MARCO), both centered on knowledge-intensive QA
/ search-oriented RAG rather than the full space of open-
ended RAG applications such as long-form summarization or
document extraction. Additional domains (e.g., biomedical,
legal) would further validate generalizability. Scalability
experiments at≥5M use 1,500 stratified queries rather than
the full 3,452. Cross-model evaluation varies the target gen-
erator across four LLMs, but it does not replace a controlled
trigger-generator swap; stronger target models (GPT-4,
Claude-3) remain untested.
•Fluency & Practical Assumptions.Perplexity remains
a proxy for fluency; domain-specific style detectors could
identify subtle anomalies that our metrics miss. The attack
requires white-box retriever access, topically-relevant seeds,
and nontrivial per-document optimization. Our Wikipedia-
style reviewer-perception study (Appendix A) provides pre-
liminary evidence about perceived plausibility under a sim-
plified review scenario, but it does not model the full injec-
tion pipeline. In addition, the current study fixes the CATG
generator to Llama-2-7B-Chat, and the adaptive reranker
experiment is matched only to MiniLM-L6-v2; varying the
trigger generator and extending adaptive attacks to larger
rerankers are important next steps.
9 Conclusion
We presentedSilentRetrieval, a two-stage data poisoning attack
against RAG systems featuring Coordinated Beam Search (CBS)
for multi-token joint optimization and Context-Adaptive Trigger
Generation (CATG) for lightweight semantic trigger fusion. Under
our unified single-document, synthetic-target evaluation on NQ
and MS MARCO,SilentRetrievalachieves the strongest stealth–
effectiveness trade-off among the evaluated baselines (PES: 1,386
on NQ, 1,262 on MS MARCO), with near-benign perplexity and
nontrivial effectiveness across the evaluated target generators and
datasets. Our defense evaluation shows that combined defenses
reduce ASR-LLM to 25.6% at a 6 ×latency trade-off in our evaluatedsetting, and to 21.3% under the strongest evaluated configuration,
while adaptive attacks partially recover effectiveness in the matched
reranker setting.
Key Takeaways for the Community.(1)Corpus integrity is a
first-class security concern: in our evaluation, fluency-preserving at-
tacks often pass a simple PPL-based audit at the calibrated threshold
(8.7% detection) and are much harder for human annotators to flag
than prior disfluent attacks, while still showing a higher flag-rate
trend than benign content (14.7% vs. 7.3%, 𝑝≈0.064). (2)Layered
defenses appear more robust than any single tested defense: combin-
ing retrieval-side reranking, hybrid retrieval, and generation-side
passage isolation provides the strongest mitigation among the con-
figurations we evaluate. (3)The stealth–effectiveness frontier is a
useful evaluation lens in this setting: raw ASR comparisons without
PPL normalization can obscure practical differences in detectability
under simple fluency auditing.
Future Directions.Several promising directions remain: (i) de-
veloping detectors specifically trained on fluency-preserving adver-
sarial examples; (ii) extending query-augmentation defenses with
answer-consistency checking to dilute single-document influence;
(iii) designing sparse-attention generation mechanisms that limit
trigger effectiveness; (iv) establishing unified attack–defense bench-
marks (e.g., RSB [ 31]) to enable fair cross-method comparison; and
(v) investigating multi-modal RAG poisoning as vision-language
retrieval systems gain adoption.
References
[1]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
InInternational Conference on Learning Representations. OpenReview.net, Vienna,
Austria, 30 pages. https://openreview.net/forum?id=hSyW5go0v8
[2]Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong
Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir
Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. 2016. MS
MARCO: A Human Generated MAchine Reading COmprehension Dataset.arXiv
preprint arXiv:1611.09268abs/1611.09268 (2016), 16 pages. doi:10.48550/arXiv.
1611.09268
[3]Matan Ben-Tov and Mahmood Sharif. 2025. GASLITEing the Retrieval: Exploring
Vulnerabilities in Dense Embedding-based Search. InProceedings of the 2025
ACM SIGSAC Conference on Computer and Communications Security (CCS). ACM,
Taipei, Taiwan, 4364–4378. doi:10.1145/3719027.3765095
[4]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al .2020. Language models are few-shot learners.Advances in neural
information processing systems33 (2020), 1877–1901.
[5]Tailun Chen, Yu He, Yan Wang, Shuo Shao, Haolun Zheng, Zhihao Liu, Jinfeng Li,
Zhizhen Qin, Yuefeng Chen, Zhixuan Chu, Zhan Qin, and Kui Ren. 2025. MIRAGE:
Misleading Retrieval-Augmented Generation via Black-box and Query-agnostic
Poisoning Attacks.arXiv preprint arXiv:2512.08289abs/2512.08289 (2025), 1–14.
doi:10.48550/arXiv.2512.08289
[6]Yen-Shan Chen, Sian-Yao Huang, Cheng-Lin Yang, and Yun-Nung Chen. 2025.
Eyes-on-Me: Scalable RAG Poisoning through Transferable Attention-Steering
Attractors.arXiv preprint arXiv:2510.00586abs/2510.00586 (2025), 1–13. doi:10.
48550/arXiv.2510.00586
[7]Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu, Wei Du, Ping Yi, Zhu-
osheng Zhang, and Gongshen Liu. 2024. TrojanRAG: Retrieval-Augmented
Generation Can Be Backdoor Driver in Large Language Models.arXiv preprint
arXiv:2405.13401abs/2405.13401 (2024), 1–12. doi:10.48550/arXiv.2405.13401
[8]Javid Ebrahimi, Anyi Rao, Daniel Lowd, and Dejing Dou. 2018. HotFlip: White-
box adversarial examples for text classification. InProceedings of the 56th Annual
Meeting of the Association for Computational Linguistics. Association for Compu-
tational Linguistics, Melbourne, Australia, 31–36. doi:10.18653/v1/P18-2006
[9]Michael Glass, Gaetano Rossiello, Md Faisal Mahbub Chowdhury, Ankita Naik,
Pengshan Cai, and Alfio Gliozzo. 2022. Re2G: Retrieve, Rerank, Generate. In
Proceedings of the 2022 Conference of the North American Chapter of the Association
for Computational Linguistics. Association for Computational Linguistics, Seattle,
United States, 2701–2715. doi:10.18653/v1/2022.naacl-main.194

KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea Jiachen Qian
[10] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang.
2020. REALM: Retrieval augmented language model pre-training. InInternational
Conference on Machine Learning. PMLR, Virtual, 3929–3938. https://proceedings.
mlr.press/v119/guu20a.html
[11] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bo-
janowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised dense infor-
mation retrieval with contrastive learning.Transactions on Machine Learning
Research(2022), 21 pages. https://openreview.net/forum?id=jKN1pXi7b0
[12] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo
Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave.
2023. Atlas: Few-shot learning with retrieval augmented language models.Journal
of Machine Learning Research24, 251 (2023), 1–43. https://jmlr.org/papers/v24/23-
0037.html
[13] Yang Jiao, Xiaodong Wang, and Kai Yang. 2025. PR-Attack: Coordinated Prompt-
RAG Attacks on Retrieval-Augmented Generation in Large Language Models
via Bilevel Optimization. InProceedings of the 48th International ACM SIGIR
Conference on Research and Development in Information Retrieval. ACM, Padua,
Italy, 656–667. doi:10.1145/3726302.3730058
[14] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing. Association for Computational Linguis-
tics, Online, 6769–6781. doi:10.18653/v1/2020.emnlp-main.550
[15] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton
Lee, et al .2019. Natural questions: a benchmark for question answering research.
Transactions of the Association for Computational Linguistics7 (2019), 453–466.
doi:10.1162/tacl_a_00276
[16] Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. 2018. A simple unified
framework for detecting out-of-distribution samples and adversarial attacks. In
Advances in Neural Information Processing Systems, Vol. 31. Curran Associates,
Inc., Montréal, Canada, 7167–7177.
[17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al .
2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. In
Advances in Neural Information Processing Systems, Vol. 33. Curran Associates,
Inc., Virtual, 9459–9474.
[18] Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT.
arXiv preprint arXiv:1901.04085abs/1901.04085 (2019), 1–4. doi:10.48550/arXiv.
1901.04085
[19] Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani,
Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard,
et al .2021. KILT: a benchmark for knowledge intensive language tasks. In
Proceedings of the 2021 Conference of the North American Chapter of the Association
for Computational Linguistics. Association for Computational Linguistics, Online,
2523–2544. doi:10.18653/v1/2021.naacl-main.200
[20] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya
Sutskever. 2019. Language models are unsupervised multitask learners.Ope-
nAI Technical Report(2019), 24 pages. https://cdn.openai.com/better-language-
models/language_models_are_unsupervised_multitask_learners.pdf
[21] Yangguang Shao, Xinjie Lin, Haozheng Luo, Chengshang Hou, Gang Xiong,
Jiahao Yu, and Junzheng Shi. 2025. POISONCRAFT: Practical Poisoning of
Retrieval-Augmented Generation for Large Language Models.arXiv preprint
arXiv:2505.06579abs/2505.06579 (2025), 1–11. doi:10.48550/arXiv.2505.06579
[22] Congzheng Song, Alexander M Rush, and Vitaly Shmatikov. 2020. Adversarial
semantic collisions. InProceedings of the 2020 Conference on Empirical Methods in
Natural Language Processing. Association for Computational Linguistics, Online,
4198–4210. doi:10.18653/v1/2020.emnlp-main.344
[23] Jinyan Su, Preslav Nakov, and Claire Cardie. 2025. Corpus Poisoning via Approx-
imate Greedy Gradient Descent. InFindings of the Association for Computational
Linguistics: ACL 2025. Association for Computational Linguistics, Vienna, Austria,
4274–4294. doi:10.18653/v1/2025.findings-acl.222
[24] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yas-
mine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhos-
ale, et al .2023. Llama 2: Open foundation and fine-tuned chat models.arXiv
preprint arXiv:2307.09288abs/2307.09288 (2023), 1–77. doi:10.48550/arXiv.2307.
09288
[25] Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan
Wang, and Qing Wang. 2026. Joint-GCG: Unified Gradient-Based Poisoning
Attacks on Retrieval-Augmented Generation Systems. InProceedings of the AAAI
Conference on Artificial Intelligence. Association for the Advancement of Artificial
Intelligence, Singapore, 35793–35801. doi:10.1609/aaai.v40i42.40893
[26] Meng Xi, Sihan Lv, Yechen Jin, Guanjie Cheng, Naibo Wang, Ying Li, and Jianwei
Yin. 2025. RIPRAG: Hack a Black-box Retrieval-Augmented Generation Question-
Answering System with Reinforcement Learning.arXiv preprint arXiv:2510.10008
abs/2510.10008 (2025), 1–10. doi:10.48550/arXiv.2510.10008
[27] Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner, Danqi Chen, and Prateek
Mittal. 2024. Certifiably Robust RAG against Retrieval Corruption.arXiv preprintarXiv:2405.15556abs/2405.15556 (2024), 1–24. doi:10.48550/arXiv.2405.15556
[28] Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. 2024. Adaptive
Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models
in Knowledge Conflicts. InInternational Conference on Learning Representations.
OpenReview.net, Vienna, Austria, 24 pages. https://openreview.net/forum?id=
auKAUJZMO6
[29] Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun Chen, and Qian Lou. 2024.
BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large
Language Models.arXiv preprint arXiv:2406.00083abs/2406.00083 (2024), 1–13.
doi:10.48550/arXiv.2406.00083
[30] Baolei Zhang, Yuxi Chen, Zhuqing Liu, Lihai Nie, Tong Li, Zheli Liu, and
Minghong Fang. 2025. Practical Poisoning Attacks against Retrieval-Augmented
Generation.arXiv preprint arXiv:2504.03957abs/2504.03957 (2025), 1–15. doi:10.
48550/arXiv.2504.03957
[31] Baolei Zhang, Haoran Xin, Jiatong Li, Dongzhe Zhang, Minghong Fang, Zhuqing
Liu, Lihai Nie, and Zheli Liu. 2025. Benchmarking Poisoning Attacks against
Retrieval-Augmented Generation.arXiv preprint arXiv:2505.18543abs/2505.18543
(2025), 1–14. doi:10.48550/arXiv.2505.18543
[32] Collin Zhang, Tingwei Zhang, and Vitaly Shmatikov. 2026. Adversarial Decoding:
Generating Readable Documents for Adversarial Objectives. InFindings of the As-
sociation for Computational Linguistics: EACL 2026. Association for Computational
Linguistics, Rabat, Morocco, 2053–2068. doi:10.18653/v1/2026.findings-eacl.108
[33] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu,
Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P Xing, et al .2023. Judg-
ing LLM-as-a-Judge with MT-Bench and Chatbot Arena. InAdvances in Neural
Information Processing Systems, Vol. 36. Curran Associates, Inc., New Orleans,
LA, 46595–46623.
[34] Zexuan Zhong, Ziqing Huang, Alexander Wettig, and Danqi Chen. 2023. Poison-
ing Retrieval Corpora by Injecting Adversarial Passages. InProceedings of the 2023
Conference on Empirical Methods in Natural Language Processing. Association for
Computational Linguistics, Singapore, 13764–13775. doi:10.18653/v1/2023.emnlp-
main.849
[35] Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia. 2025. Poisone-
dRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of
Large Language Models. In34th USENIX Security Symposium (USENIX Security
25). USENIX Association, Seattle, WA, 3827–3844. https://www.usenix.org/
conference/usenixsecurity25/presentation/zou-poisonedrag
A Wikipedia-Style Reviewer Perception Study
To ground the threat model, we conducted a small controlled
reviewer-perception analysis simulating one aspect of Wikipedia-
style review. We created 50SilentRetrievaladversarial passages
(along with 50 Zhong et al. and 50 PoisonedRAG passages) for-
matted as plausible Wikipedia paragraphs and submitted them
to three volunteer reviewers with Wikipedia editing experience,
who independently evaluated whether each passage would be
likely to survive a typical “new page patrol” review using a binary
rating (“likely to survive” vs. “likely to be flagged”). Results: 78% of
SilentRetrievalpassages were rated as “likely to survive initial
review” compared to 12% for Zhong et al. and 34% for PoisonedRAG
(inter-rater agreement: Fleiss’ 𝜅=0.68). We interpret this only as
evidence about perceived plausibility under a simplified moderation
scenario, not as direct evidence of real-world injection success.
Limitations:The sample size is small ( 𝑛=150 passages, 3 raters),
volunteer reviewers may not be representative of active Wikipedia
patrollers, and the study does not model the full injection pipeline
(account creation, edit history, community norms, page context,
and follow-up community moderation). These results should be
interpreted as suggestive evidence about reviewer perception rather
than definitive evidence of real Wikipedia injection feasibility. This
analysis is intended to motivate defense research, not to provide
an injection toolkit.
B Cross-Tokenizer Gradient Computation
Since the retriever (Contriever) and the PPL model (GPT-2) use
different tokenizers and embedding spaces, we compute the two

SilentRetrieval: Hijacking RAG via Semantically-Preserving Poisoning KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea
gradient terms in the Stage 1 gradient objective separately. The
similarity gradient ∇𝑒𝑖sim(·) is computed through Contriever’s en-
coder. The PPL term is incorporated through a text-space alignment
heuristic: we retokenize the current document with the GPT-2 to-
kenizer, compute∇PPL with respect to GPT-2 embeddings, and
aggregate the resulting signal over aligned sub-token spans when
ranking candidate edit positions. Since generated token candidates
are ultimately evaluated by theexactdiscrete objective during beam-
level re-evaluation, this alignment heuristic does not change the
scoring rule used to rank generated candidates. It can, however, af-
fect the candidate proposal set, so we treat it as a heuristic proposal
mechanism rather than an exact cross-tokenizer gradient estimator.
C CBS Optimization Dynamics
CBS relies on a HotFlip/GCG-style first-order relaxation: gradi-
ents are computed with respect to continuous token embeddings,
then projected onto the discrete vocabulary via dot-product scor-
ing. This relaxation introduces an approximation gap—the discrete
replacement that maximizes the projected score may not maxi-
mize the true objective. We mitigate this in two ways. First, by
retaining𝐾=10 candidates per position (rather than the greedy
top-1), CBS explores a broader neighborhood and reduces sensi-
tivity to noisy gradient directions. Second, the PPL penalty term
𝜆·max( 0,PPL(𝑑)−𝜏 ppl)introduces a non-smooth hinge at 𝜏ppl,
which can cause gradient oscillation near the boundary. In practice,
the beam-level re-evaluation acts as a stabilizer: candidates are
scored by theexactobjective (including true PPL), not solely by
gradient projections, so the hinge non-smoothness affects candi-
date ranking but not final selection. Empirically, CBS converges
within 300–400 iterations for 87% of documents, with a median
objective variance of <0.01 over the final 50 iterations, indicating
stable convergence despite the discrete relaxation.
D Comparison Table
Table 10 provides a feature-level comparison of corpus poisoning
methods across five dimensions.
E Defense Architecture
The evaluated defense stack follows a layered pipeline: lightweight
PPL filtering first removes highly disfluent candidates, embedding-
space detectors such as Mahalanobis scoring catch distributional
outliers, cross-encoder reranking reorders suspicious passages us-
ing query-document interaction, and LLM-side passage isolation
reduces the influence of any single retrieved document. Empiri-
cally, the lightweight filters alone are insufficient againstSilen-
tRetrieval, while combining retrieval-side and generation-side
defenses provides the strongest mitigation.
F Detection-Based Defense Results
G Computation Cost
H Additional Analyses
Seed sensitivity and multi-target settings.In a separate seed-
selection sensitivity run with the same full CBS+CATG pipeline,
initializing from the default top-1 retrieved seed achieves 85.2%Table 10: Comparison with related corpus poisoning meth-
ods. Ret.=Retrieval optimization, Gen.=Generation manip-
ulation, Flu.=Fluency constraints, Low%=Demonstrated at
<0.1% poisoning, BB=Black-box transfer support.∗Evaluated
at 0.069% poisoning (5M corpus) with reduced effectiveness
(79.3% HR@10; see Table 8).
Method Ret. Gen. Flu. Low% BB
Zhong et al. [34]✓– –✓–
PoisonedRAG [35]✓ ✓– – –
AGGD [23]✓– – – –
GASLITE [3]✓–✓ ✓–
TrojanRAG [7]✓ ✓– – –
BadRAG [29]✓ ✓– – –
AdvDec [32] –✓ ✓– –
Joint-GCG [25]✓ ✓– – –
EYES-ON-ME [6]✓ ✓– –✓
MIRAGE [5]✓ ✓ ✓–✓
POISONCRAFT [21]✓ ✓– –✓
PR-Attack [13]✓ ✓–✓–
CorruptRAG [30]✓ ✓–✓ ✓
RIPRAG [26]✓ ✓– –✓
SilentRetrieval(Ours)✓ ✓ ✓(∗) (†)
†Black-box support via surrogate transfer (Section 5); primary optimization requires
white-box retriever access.
Table 11: Detection rates of defense methods against corpus
poisoning attacks. Lower = more stealthy.‡Simplified re-
implementation trained on Zhong et al. examples.
Defense Zhong et al. PoisonedRAG GASLITESilentRetrieval
Independent PPL audit (𝜏=50) 94.2% 78.6% 12.3%8.7%
LLM-based filter (GPT-4) 89.3% 71.4% 24.6%14.2%
RAGuard‡67.8% 52.3% 41.7%34.2%
Mahalanobis [16] 72.4% 58.9% 38.4%31.6%
Table 12: Relative computation cost per document.
Method Time/doc Relative Time HR@10
Zhong et al. 8 min 0.5×72.3
PoisonedRAG 12 min 0.8×78.6
GASLITE 18 min 1.2×76.8
Joint-GCG 35 min 2.3×81.2
SilentRetrieval15 min 1.0×84.6
HR@10, close to the main-test result of 84.6%. Replacing it with ran-
dom topical passages reduces HR@10 by 6.3%, and random corpus
passages reduce HR@10 by 22.1%. Optimizing one document for
multiple related queries introduces a trade-off: two queries main-
tain 76.3% HR@10, while five queries degrade to 58.4%. Injecting
𝑁= 3documents per query raises HR@10 to 94.3% and ASR-LLM
to 69.2%.
ASR-P failure analysis.Of ASR-LLM successes, 32.3% fail un-
der query rephrasing. Manual analysis of 200 non-persistent cases

KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea Jiachen Qian
identifies query reformulation shifting retrieval (43.5%), LLM hedg-
ing under paraphrase (31.0%), competing context dominance (24.5%),
and other/rounding cases (1.0%) as the main failure modes.
I Pareto Efficiency Analysis
We define PES=(HR@10×ASR-LLM)/ln(PPL+ 1)to jointly
evaluate effectiveness and stealthiness. On NQ,SilentRetrieval
achieves PES = 1,386, outperforming Joint-GCG (1,008), GASLITE
(921), PoisonedRAG (629), and Zhong et al. (412). The same ranking
holds on MS MARCO, whereSilentRetrievalreaches PES = 1,262.
J Qualitative Case Studies
Across entertainment, medical, and science queries, CATG-generated
triggers integrate domain-specific language while preserving flu-
ency. Successful cases include target substitutions such as attribut-
ing “Bohemian Rhapsody” to Elton John or shifting a diabetestreatment answer toward sulfonylureas. A representative failure
occurs for well-established constants such as the speed of light,
where the LLM’s parametric knowledge overrides adversarial
context.
K Ethical Considerations (Extended)
We present this attack to support defense development and follow
a responsible-disclosure posture. The techniques are dual-use, but
documenting them together with concrete mitigations is preferable
to relying on security through obscurity. All experiments use aca-
demic benchmarks with synthetic target answers; no real-world
knowledge bases were poisoned. Recommended mitigations include
cross-encoder reranking, hybrid BM25+dense retrieval, passage iso-
lation with majority voting, fluency-aware moderation, adaptive-
robust defense design, and treating corpus integrity as a first-class
security requirement.
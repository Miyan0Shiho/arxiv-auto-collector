# AdaGATE: Adaptive Gap-Aware Token-Efficient Evidence Assembly for Multi-Hop Retrieval-Augmented Generation

**Authors**: Yilin Guo, Yinshan Wang, Yixuan Wang

**Published**: 2026-05-04 14:45:49

**PDF URL**: [https://arxiv.org/pdf/2605.05245v1](https://arxiv.org/pdf/2605.05245v1)

## Abstract
Retrieval-augmented generation (RAG) remains brittle on multi-hop questions in realistic deployment settings, where retrieved evidence may be noisy or redundant and only limited context can be passed to the generator. Existing controllers address parts of this problem, but typically either expand context additively, select from a fixed top-k set, or optimize relevance without explicitly repairing missing bridge facts. We propose AdaGATE, a training-free evidence controller for multi-hop RAG that frames evidence selection as a token-constrained repair problem. AdaGATE combines entity centric gap tracking, targeted micro-query generation, and a utility based selection mechanism that balances gap coverage, corroboration, novelty, redundancy, and direct question relevance. We evaluate AdaGATE on HotpotQA under clean, redundancy, and noise injected retrieval conditions. Across all three settings, AdaGATE achieves the best evidence F1 among the compared controllers, reaching 62.3% on clean data and 71.2% under redundancy injection, while using 2.6x fewer input tokens than Adaptive-k. These results suggest that explicit gap-aware repair, combined with token-efficient evidence selection, improves robustness in multi-hop RAG under imperfect retrieval. Our code and evaluation pipeline are available at https://github.com/eliguo/AdaGATE.

## Full Text


<!-- PDF content starts -->

AdaGATE: Adaptive Gap-Aware Token-Efficient Evidence Assembly for
Multi-Hop Retrieval-Augmented Generation
Yilin Guo
Center for Data Science
New York University
yg3030@nyu.eduYinshan Wang
Tandon School of Engineering
New York University
yw9023@nyu.eduYixuan Wang
Center for Data Science
New York University
yw8735@nyu.edu
Abstract
Retrieval-augmented generation (RAG) re-
mains brittle on multi-hop questions in real-
istic deployment settings, where retrieved ev-
idence may be noisy or redundant and only
limited context can be passed to the genera-
tor. Existing controllers address parts of this
problem, but typically either expand context
additively, select from a fixed top- kset, or op-
timize relevance without explicitly repairing
missing bridge facts. We propose ADAGATE,
atraining-freeevidence controller for multi-
hop RAG that frames evidence selection as
atoken-constrained repairproblem. Ada-
GATE combines entity centric gap tracking,
targeted micro-query generation, and a util-
ity based selection mechanism that balances
gap coverage,corroboration,novelty,redun-
dancy, anddirect question relevance. We
evaluate AdaGATE on HotpotQA under clean,
redundancy, and noise injected retrieval con-
ditions. Across all three settings, AdaGATE
achieves thebest evidence F1among the com-
pared controllers, reaching 62.3% on clean data
and 71.2% under redundancy injection, while
using2.6 ×fewer input tokensthan Adaptive-
k. These results suggest that explicit gap-
aware repair, combined with token-efficient ev-
idence selection, improves robustness in multi-
hop RAG under imperfect retrieval. Our code
and evaluation pipeline are available at https:
//github.com/eliguo/AdaGATE.
1 Introduction
Retrieval-augmented generation (RAG) improves
large language models (LLMs) by conditioning
generation on external documents, reducing hal-
lucinations and improving factual accuracy (Fan
et al., 2024). In realistic deployments, however,
retrieved evidence is often noisy, redundant, or in-
complete. Under limited context budgets imposed
by API costs and latency constraints (Taguchi et al.,
2025; Peng et al., 2025), RAG systems cannot sim-
ply pass all retrieved content to the generator. Thischallenge is especially acute for multi-hop ques-
tions, where answering often depends on assem-
bling a small set of complementary passages: miss-
ing a bridge fact can cause failure, while admitting
redundant or misleading evidence can distort the
final answer. These constraints motivate viewing
multi-hop RAG as a token-constrained evidence
assembly problem under imperfect retrieval.
Prior work has shown that RAG performance de-
pends not only on retrieval quality, but also on how
retrieved evidence is selected and organized for
generation. Irrelevant passages can substantially
degrade answer quality (Cuconasu et al., 2024),
and dense retrievers often return clusters of nearly
duplicate chunks that reduce coverage of the full
reasoning chain. Recent methods address different
parts of this problem. Self-RAG trains an LLM
to interleave generation with retrieval-aware re-
flection (Asai et al., 2023). Adaptive- kselects a
query specific number of passages based on sim-
ilarity score gaps (Taguchi et al., 2025). SEAL-
RAG performs entity centric gap repair through
targeted micro-queries and replacement-based up-
dates (Lahmy and Yozevitch, 2025). However,
these approaches typically either expand context
additively, operate over a fixed top- kset, or do not
explicitly balance gap repair, redundancy, and con-
text efficiency within a single evidence selection
procedure.
We propose ADAGATE (Figure 1), atraining-
free controllerthat treats multi-hop evidence selec-
tion as agap-aware,token-efficient repair. Ada-
GATE maintains an entity centric ledger, issues
targeted micro-queries with a question aware fall-
back channel, andscores candidateswith a utility
function balancing gap coverage, corroboration,
novelty, redundancy, and question relevance. A
utility adaptive capacity heuristic then assembles a
compact evidence set under a global token budget.
Evaluated on HotpotQA under clean, redun-
dancy injection and noise injection conditions
1arXiv:2605.05245v1  [cs.CL]  4 May 2026

User Query Initial Retrieval
Candidate pool 𝐶0 from
dense / hybrid retrieverCurrent Evidence Set 𝐸𝑡 
∑𝑐 ∈𝐸𝑡tokens (𝑐)≤𝐵Documents Grader &
Extraction
Ledger 𝑈𝑡, Gaps  Sufﬁcient?YesNo (repair)
Retrieve Relevant
Docs & Entities
(from 𝐸𝑡) Final AnswerMicro Query Generation
From gaps 𝐺𝑡 & question Token-Budgeted Selection
& Replacement
Swap low-utility 𝑣 ∈𝐸𝑡  with
high-utility 𝑐⋆
s.t. ∑𝑐 ∈𝐸𝑡+1tokens (𝑐)≤𝐵When gap extraction is
noisy , fall back to
question-aware queries.
Utility Scoring &
Capacity Estimation
  =  +  +
 -  + 
Largest utility gap  ⇒ effective
capacity  Sort utilities, ﬁnd largest
drop →  estimate 𝐾𝑡eff
updated 𝐸𝑡+1 AdaGA TE Controller (training-free)
Works with ﬁxed retriever & black-box LLM
SEAL-RAG pipeline 
(Lahmy and Yozevitch, 2025 )
AdaGA TE (Our) add-on blocksFigure 1: AdaGATE framework overview. Unlike SEAL-RAG (Lahmy and Yozevitch, 2025), it introduces a
training-free, gap-aware controller that explicitly enforces token efficiency on top of a fixed retriever and LLM.
against four baselines, AdaGATE achieves the high-
est evidence F1 across all three settings (62.3%
clean, 71.2% redundancy, 62.7% noise) while us-
ing 2.6×fewer input tokens than Adaptive-k.
The main contributions of this work are: (1)
We formulate multi-hop RAG under imperfect re-
trieval as a token constrained evidence repair prob-
lem and clarify the limitations of fixed- kevidence
controllers in noisy and redundant settings; (2) We
introduce ADAGATE, a training-free controller
that combines entity centric gap tracking, utility
based evidence scoring, and adaptive capacity con-
trol for compact evidence assembly; and (3) We
develop a stress tested evaluation protocol on Hot-
potQA with controlled redundancy and noise injec-
tion, and compare controllers across answer quality,
grounding, and token efficiency.
2 Related Work
2.1 Multi-Hop RAG under Imperfect
Retrieval
Standard RAG pipelines retrieve a fixed top- kset
of passages and concatenate them with the query,
implicitly treating evidence selection as a one-shot
step. Multi-hop QA benchmarks such as HotpotQA
and 2WikiMultiHopQA expose the limits of this
assumption: answering often requires combining
complementary facts across documents, and omit-
ting a single bridge passage can cause failure even
when retrieval recall is high (Yang et al., 2018;
Welbl et al., 2018). Prior work further shows that
long or noisy contexts degrade generation quality
due to distractor sensitivity and “lost in the middle”
effects (Liu et al., 2023; Cuconasu et al., 2024).
These findings motivate controllers that explicitly
manage evidence composition under imperfect re-trieval.
2.2 Active and Corrective RAG Controllers
Recent work has made the RAG controller more
adaptive. Self-RAG trains an LLM to interleave
generation with retrieval awareness reflection (Asai
et al., 2023). Adaptive-RAG routes questions
among non retrieval, single and multi step strate-
gies based on estimated complexity (Jeong et al.,
2024). CRAG evaluates retrieved documents and
triggers corrective actions such as additional re-
trieval or document decomposition when evidence
quality appears low (Yan et al., 2024). These meth-
ods make retrieval more adaptive, but they primar-
ily decide when or whether to retrieve, and several
rely on model finetuning, rather than focusing on
token efficient evidence assembly for multi-hop
reasoning.
SEAL-RAG is most closely related to our setting
(Lahmy and Yozevitch, 2025). It maintains an en-
tity centric ledger, identifies missing information as
gaps, and issues targeted micro-queries to repair a
fixed evidence set through replacement rather than
expansion. Our work builds directly on this line of
explicit gap-aware repair, but extends it beyond a
fixed top-ksetting.
2.3 Adaptive Evidence Selection and Position
of This Work
A complementary line of work studies how much
context to include. Adaptive- kselects a query spe-
cific number of passages by identifying the largest
drop in sorted similarity scores (Taguchi et al.,
2025). AdaGReS formulates evidence selection
as a token budgeted optimization problem balanc-
ing relevance and redundancy (Peng et al., 2025).
These approaches reason about capacity and redun-
2

dancy, but they do not explicitly model multi-hop
information gaps or use targeted micro-queries to
repair missing evidence.
AdaGATE combines these two perspectives.
Like SEAL-RAG, it is a training-free controller
that performs explicit gap-aware evidence repair
with an entity centric ledger. Like Adaptive- kand
AdaGReS, it reasons about context efficiency under
limited budgets. Its key difference is to integrate
gap-aware repair, question-aware fallback retrieval,
redundancy-aware utility scoring, and adaptive ca-
pacity control within a single evidence selection
procedure for multi-hop RAG.
3 Method
We formulate multi-hop RAG under deployment
constraints as a token constrained evidence repair
problem. Given a query q, a corpus D, and a global
token budget B, the goal is to assemble a com-
pact evidence set that supports multi-hop reasoning
while avoiding redundant or misleading passages.
AdaGATE is a training-free controller built on top
of a fixed retriever and generator. At each iteration
t, it maintains an evidence set Et, an entity centric
ledger Ut, and a set of unresolved information gaps
Gt. Relative to SEAL-RAG (Lahmy and Yozevitch,
2025), AdaGATE makes three changes: it replaces
fixed- kevidence selection with constrained token
selection, adds a question-aware fallback channel
to gap targeted retrieval, and uses adaptive utility
capacity control to avoid overfilling the context
with low value passages.
3.1 Gap-Aware Retrieval and Evidence State
LetCtdenote the candidate pool retrieved at itera-
tiont. Each passage c∈ C thas token length ℓ(c),
and the final evidence set must satisfy
X
c∈E tℓ(c)≤B.(1)
Following SEAL-RAG, AdaGATE uses two
LLM-based primitives: (1) ledger extraction,
which summarizes the current evidence set into
structured entity–relation–value tuples with con-
fidence scores, and (2) gap specification, which
identifies missing facts needed to answer the ques-
tion (Lahmy and Yozevitch, 2025). We treat these
as black-box components and focus on how Ada-
GATE uses them to guide retrieval and evidence
selection under a limited context budget.For each gap g∈G t, AdaGATE generates one
or more targeted micro-queries. To improve robust-
ness when gap extraction is noisy or overly abstract,
it also generates a small set of question-anchored
fallback queries derived directly from q. The union
of gap-aware and question-aware queries is sent
to the retriever to form the next candidate pool Ct.
This design allows the controller to continue ex-
ploring useful evidence even when the current gap
representation is incomplete.
3.2 Utility-Based Evidence Scoring
Given the current query, ledger, gaps, and evidence
state, AdaGATE assigns each candidate passage
c∈ C ta scalar utility score
St(c) =λ 1GapCov(c, G t) +λ 2Corr(c, U t)
+λ 3Nov(c, U t)−λ 4Red(c, E t)
+λ 5RelQ(c, q),(2)
The five terms capture complementary roles in
multi-hop evidence assembly. GapCov(c, G t)re-
wards passages that directly address unresolved
gaps. Corr(c, U t)rewards support for low-
confidence facts already present in the ledger.
Nov(c, U t)favors passages that contribute new en-
tities or relations rather than repeating lateral infor-
mation. Red(c, E t)penalizes candidates that are
highly similar to evidence already selected. Finally,
RelQ(c, q) measures direct relevance to the origi-
nal question and acts as a fallback signal when gap
extraction is noisy. Compared with SEAL-RAG,
the most important additions are the explicit redun-
dancy penalty and the question-aware relevance
term, which together make the controller more ro-
bust under noisy or redundant retrieval.
3.3 Token-Constrained Selection with
Adaptive Capacity
AdaGATE does not fix the number of passages
passed to the generator. Instead, it selects evidence
under the token budget in Eq. 1, allowing the fi-
nal evidence set size to vary with passage length
and utility. In practice, AdaGATE uses the utility
score in Eq. 2 as a surrogate for marginal value and
greedily assembles a compact evidence set from
the highest scoring candidates.
To avoid filling the available budget with many
mediocre passages, AdaGATE estimates an effec-
tive capacity from the utility distribution. Let
S(1)
t≥S(2)
t≥ ··· ≥S(M)
t
3

denote candidate utilities sorted in descending or-
der, and define adjacent drops
∆i=S(i)
t−S(i+1)
t.
AdaGATE chooses the largest drop
i⋆= arg max
i∆i
and sets
Keff
t=i⋆+B buf,
where Bbuf= 2is a small buffer. The largest utility
drop separates a high-value prefix from a lower-
value tail; AdaGATE prioritizes candidates within
the top Keff
trange and greedily selects from them
while enforcing the global token budget.
AdaGATE iterates over four stages:extract,
search,score, andreplace. It first extracts the
current ledger and unresolved gaps from Et, then
retrieves new candidates using both gap-aware
and question-aware queries, scores candidates with
Eq. 2 and estimates the effective capacity, and fi-
nally updates the evidence set by replacing lower
utility passages with higher utility candidates while
respecting the token budget. The process stops
when no useful repair remains, no meaningful gaps
are identified, or a maximum number of repair it-
erations is reached. After termination, the final
evidence set is concatenated with the question and
passed to the generator.
4 Experimental Setup
4.1 Dataset and Retrieval Setup
We evaluate on HotpotQA (Yang et al., 2018), a
multi-hop QA benchmark over Wikipedia in which
each question is associated with two supporting
paragraphs and additional distractor passages. We
use the distractor setting, which provides both rele-
vant and irrelevant evidence and is therefore suit-
able for studying evidence selection under imper-
fect retrieval.
All controllers share the same retrieval infras-
tructure. We use a single Pinecone index built
from the first 1,000 HotpotQA validation examples,
yielding 10,919 document chunks embedded with
OpenAI text-embedding-3-small . Evaluation is
conducted on N= 200 validation examples. Be-
cause all methods retrieve from the same index and
use the same embedding model, observed differ-
ences reflect evidence control rather than changes
in the retriever.4.2 Stress-Test Retrieval Conditions
To evaluate robustness beyond the clean benchmark
setting, we construct controlled perturbations of the
candidate pools while keeping the question, gold
answer, and supporting facts unchanged. Each con-
dition is indexed into a separate Pinecone names-
pace.
Noise injection.We construct a mixed noise pool
by combining two perturbation types applied to the
original passages: syntax distortion (word order
scrambling, spelling corruption, and partial trunca-
tion) and cross-query injection (irrelevant passages
sampled from other examples). The noise ratio is
fixed at ρnoise= 0.5 , increasing the candidate pool
from 10 to 20 passages per example.
Redundancy injection.We augment each ex-
ample with paraphrastic or partially overlapping
variants of the gold supporting passages. The re-
dundancy ratio is fixed at ρred= 0.5 , increasing
the total pool to approximately 19,854 indexed
documents. Unlike the noise condition, these
passages are topically relevant but largely non-
complementary, testing whether a controller can
avoid spending budget on repeated evidence.
4.3 Models and Baselines
All methods use the same retriever, generator,
and evaluation setup. The retriever is OpenAI
text-embedding-3-small with Pinecone, retriev-
ingk= 3 passages per query. The generator
isgpt-4o-mini , treated as a black-box backbone
without finetuning.
We compare five controller settings:
•Basic RAG: retrieve the top- kpassages and
generate directly.
•Self-RAG(Asai et al., 2023): retrieval aware
self-reflective generation.
•Adaptive- k(Taguchi et al., 2025): query spe-
cific passage count based on similarity score
gaps.
•SEAL-RAG(Lahmy and Yozevitch, 2025):
entity centric gap-repair controller, evaluated
withL∈ {1,3}repair iterations.
•ADAGATE: our gap-aware, token-
constrained controller, evaluated with
L∈ {1,3} and global token budget
B= 3000.
4

This comparison spans fixed size, adaptive size,
and repair based evidence controllers under a
shared retrieval and generation pipeline.
4.4 Evaluation Metrics
We evaluate along four dimensions: answer cor-
rectness, evidence quality, grounding quality, and
token efficiency.
Answer correctness.We use gpt-4o as a judge
to compare predicted answers against gold answers
semantically and assign a binary correctness label.
Using a stronger judge than the generator reduces
self-evaluation bias (Saad-Falcon et al., 2023).
Evidence quality.We report precision, recall,
and F1 against the gold supporting document ti-
tles annotated in HotpotQA. Retrieved titles are
extracted from chunk headings and deduplicated
before comparison.
Grounding quality.We follow ARES (Saad-
Falcon et al., 2023) and report Context Relevance
(CR), Answer Faithfulness (AF), and Answer Rel-
evance (AR), again using gpt-4o as the judge.
These metrics complement title based retrieval mea-
sures with generation-aware judgments over the
retrieved evidence and final answer.
Token efficiency.We measure average input to-
kens per query, average number of documents
passed to the generator, and tokens per correctly an-
swered question, enabling direct cost–quality com-
parisons across controllers.
All judges are blinded to controller identity and
see only the question, answer, and retrieved pas-
sages. We use fixed prompts throughout.
5 Results
We report results across seven controllers and three
conditions. Full numerical results are in Table 2 in
the Appendix.
5.1 Evidence Quality and Answer Correctness
on Clean Data
Figure 2 reveals a consistent tension between preci-
sion and recall across all controllers in clean data.
Add-only controllers.Adaptive- kachieves the
highest accuracy (69.0%) but at the cost of severely
collapsed precision (P=0.278). Its high recall
(R=0.820) arises because it retrieves an average
of 8.6 documents per query, admitting many ir-
relevant passages alongside the gold supportingones. Basic RAG is the most balanced passive
baseline (Acc=58.5%, F1=0.601). Self-RAG’s doc-
ument grading mechanism occasionally rejects all
retrieved passages, leaving the generator with an
empty evidence set and explaining its lower recall
relative to Basic RAG.
SEAL-RAG: implementation versus paper.
SEAL-RAG achieves the highest precision
(P=0.808 at L=3 ), but recall remains low and flat
at approximately 0.42 regardless of the condition
or repair iterations. This behavior is directly trace-
able to a discrepancy between the SEAL-RAG pa-
per and its implementation. The paper describes
a utility-based ranking mechanism for evidence
selection; the actual implementation replaces this
with an LLM entity selection step that instructs the
model to “choose the fewest entities needed to an-
swer.” In practice, this causes the model to select a
single entity on the majority of questions, resulting
in an average of only 1.0 documents passed to the
generator. High precision follows naturally from
single document selection, but this context system-
atically misses bridge facts on multi-hop questions
requiring evidence from two or more sources. In-
creasing repair iterations from L=1 toL=3 im-
proves accuracy modestly (62.0% →68.5%) but
does not resolve the fundamental recall ceiling.
AdaGATE.AdaGATE addresses this recall bot-
tleneck by passing an average of 2.8–3.0 docu-
ments to the generator, assembling a richer evi-
dence set through its adaptive capacity mechanism
while suppressing redundant passages via the util-
ity scoring function. The result is the highest F1 on
clean data (62.3% at L=1 ), outperforming SEAL-
RAG (L=1) by 8.2 F1 points.
5.2 Token Efficiency
Figure 3 shows that controllers differ by more than
an order of magnitude in token usage. Adaptive- k
is the least efficient in every condition, consuming
1,116 tokens on clean data and 1,592 under redun-
dancy, with tokens-per-correct of 1,118—roughly
3.3×the cost of AdaGATE (338) and 8.3 ×SEAL-
RAG (134). SEAL-RAG is the most token efficient
at 136–140 tokens, a direct consequence of sin-
gle document selection, but at the cost of recall
established in Section 5.1.
AdaGATE uses 360 tokens on clean data, 2.6 ×
fewer than Adaptive- kwhile achieving substan-
tially higher F1. Under redundancy, token usage
5

Basic
RAGSelf-
RAGAdap-tive-kSEAL
L=1SEAL
L=3AdaGATE
L=1AdaGATE
L=301020304050607080Accuracy (%)5869
6364
4872
60 6064
5458Clean
Redundancy
Noise
Basic
RAGSelf-
RAGAdap-tive-kSEAL
L=1SEAL
L=3AdaGATE
L=1AdaGATE
L=30.00.20.40.60.81.0Precision
0.280.81
0.56 0.55
0.110.88
0.840.82
0.290.80
0.57 0.57Clean
Redundancy
Noise
Basic
RAGSelf-
RAGAdap-tive-kSEAL
L=1SEAL
L=3AdaGATE
L=1AdaGATE
L=30.00.20.40.60.81.0Recall0.82
0.420.74 0.74
0.380.73
0.660.690.80
0.410.73 0.73Clean
Redundancy
Noise
Basic
RAGSelf-
RAGAdap-tive-kSEAL
L=1SEAL
L=3AdaGATE
L=1AdaGATE
L=30.00.20.40.60.81.0F1
0.4080.623 0.620
0.1870.709 0.712
0.4140.627 0.627Clean
Redundancy
NoiseAnswer Correctness and Evidence QualityFigure 2: Answer correctness and evidence quality. Red = best per condition; green = worst; black = AdaGATE
otherwise. AdaGATE achieves the highest F1 across all three conditions.
Basic
RAGSelf-
RAGAdap-tive-kSEAL
L=1SEAL
L=3AdaGATE
L=1AdaGATE
L=302505007501000125015001750Avg Input T okens1116
136360 3701592
127220 2321073
131328344Clean
Redundancy
Noise
Basic
RAGSelf-
RAGAdap-tive-kSEAL
L=1SEAL
L=3AdaGATE
L=1AdaGATE
L=30246810121416Avg Docs Passed8.6
1.02.9 3.014.1
1.01.9 2.08.8
1.02.8 2.8Clean
Redundancy
Noise
Basic
RAGSelf-
RAGAdap-tive-kSEAL
L=1SEAL
L=3AdaGATE
L=1AdaGATE
L=302505007501000125015001750T okens per Correct1118
134338 3471605
122185 1941057
126302 309Clean
Redundancy
NoiseToken Efficiency
Figure 3: Token efficiency across controllers and conditions. Red = most efficient; green = least efficient. Adaptive- k
is consistently the least token-efficient controller.
drops adaptively to 220–232 tokens as its redun-
dancy penalty concentrates the evidence set on
fewer, higher utility passages—simultaneously re-
ducing cost and improving evidence quality.
5.3 ARES Grounding Scores
Figure 4 reveals a striking divergence between
retrieval precision and ARES grounding scores.
Adaptive- kleads on all three ARES dimensions
(CR=0.67, AF=0.63, AR=0.59 on clean) despite
collapsed retrieval precision, because its large con-
text of 8–14 documents inflates faithfulness scores
by increasing the probability that at least one pas-
sage supports the generated answer. This highlights
a fundamental limitation of document conditioned
evaluation under retrieval with high recall: ARES
scores reflect context availability rather than evi-
dence precision.
SEAL-RAG produces the lowest ARES scores
across all conditions (CR=0.24–0.26, AF=0.22–
0.23 on clean). When the single selected document
fails to contain the complete reasoning chain, thegenerator abstains rather than hallucinating, and the
ARES judge penalizes abstentions as not faithful—
explaining why SEAL-RAG’s CR substantially ex-
ceeds its AF.
AdaGATE achieves intermediate ARES scores
(CR=0.58, AF=0.51, AR=0.51 on clean) substan-
tially higher than SEAL-RAG. The gap between
AdaGATE and SEAL-RAG on ARES is larger
than the gap on retrieval F1, suggesting that multi-
document assembly provides meaningfully better
grounding support and reduces abstention rates.
5.4 Robustness Under Stress-Test Conditions
Redundancy injection.Under redundancy
(ρ=0.5 ), accuracy drops sharply for Basic RAG
(58.5% →47.5%) and Self-RAG (60.5% →49.0%).
Adaptive- k’s accuracy counterintuitively increases
to 72.5% as its high-recall strategy benefits
from a pool containing many topically relevant
paraphrases, but precision collapses to P=0.109
and ARES scores drop sharply. SEAL-RAG
accuracy is relatively stable (58.5%–65.0%) with
6

Basic
RAGSelf-
RAGAdap-tive-kSEAL
L=1SEAL
L=3AdaGATE
L=1AdaGATE
L=30.00.10.20.30.40.50.60.70.8ARES Score0.67
0.240.58
0.540.63
0.220.510.500.59
0.220.51
0.48Clean
CR
AF
AR
Basic
RAGSelf-
RAGAdap-tive-kSEAL
L=1SEAL
L=3AdaGATE
L=1AdaGATE
L=30.00.10.20.30.40.50.60.70.8ARES Score0.75
0.250.480.530.68
0.230.420.480.67
0.220.410.45Redundancy (=0.5)
CR
AF
AR
Basic
RAGSelf-
RAGAdap-tive-kSEAL
L=1SEAL
L=3AdaGATE
L=1AdaGATE
L=30.00.10.20.30.40.50.60.70.8ARES Score0.62
0.200.510.57 0.57
0.180.440.500.57
0.170.450.48Noise (=0.5)
CR
AF
ARARES Grounding ScoresFigure 4: ARES grounding scores (CR = Context Relevance, AF = Answer Faithfulness, AR = Answer Relevance).
Red = best; green = worst. SEAL-RAG scores are consistently the lowest despite high retrieval precision.
high precision (P=0.868–0.876), but recall drops
further (R=0.448–0.455) as the retriever must
surface gold documents from a pool twice the size,
and ARES scores stagnate.
AdaGATE’s F1 improves substantially un-
der redundancy (62.3% →71.2% at L=3 , and
62.3%→70.9% at L=1 ), driven by the redundancy
penalty term. When the candidate pool is domi-
nated by paraphrase variants, the novelty and re-
dundancy terms jointly suppress their utility scores,
causing AdaGATE to select fewer but more comple-
mentary documents. Token usage drops adaptively
from 360 to 220 tokens, and ARES scores remain
stable at L=3 (CR=0.53, AF=0.48), while Basic
RAG’s drop sharply (CR=0.50→0.36).
Noise injection.Under noise, all controllers de-
grade in accuracy. Noise passages retain high
embedding similarity because syntax distortion
does not fundamentally alter semantic represen-
tations, causing them to appear within the top- k.
AdaGATE’s accuracy drops to 54.0% ( L=1 ), its
weakest result, but F1 (62.7%) matches clean per-
formance, indicating that the utility scoring func-
tion partially penalizes corrupted passages through
lower GapCov and RelQ scores. Under noise at
L=3 , AdaGATE achieves the highest CR among
non-Adaptive- kcontrollers (0.57), suggesting that
utility based filtering partially maintains context
relevance even under retrieval degradation.
5.5 Pipeline-Level Analysis
Table 1 presents three representative examples from
clean evaluation logs, spanning three distinct behav-
ioral patterns: systematic single document recall
loss, micro-query failure without fallback, and over
conservative gap detection.
Q1 illustrates the most common behavioral pat-
tern: SEAL-RAG’s entity selection discards the Ed
Wood passage despite answering correctly, while
AdaGATE passes both gold documents. This re-call gap is inconsequential for simple comparisons
but compounds on questions requiring multi-hop
bridge reasoning. Q3 shows AdaGATE’s Hqfall-
back recovering from a failed micro-query that re-
turns zero new documents for SEAL-RAG, a failure
mode that occurs whenever the gap specification
is too abstract to match any indexed passage. Q21
exposes the opposite failure: AdaGATE’s stringent
gap detection causes abstention when indirect evi-
dence would suffice, while SEAL-RAG’s less con-
servative sufficiency check generates the correct
answer from partial evidence—illustrating a funda-
mental tradeoff between hallucination prevention
and abstention rate.
Summary.AdaGATE achievesbetter precision-
recall balancethan fixed- kand add-only baselines
at comparable token cost, andmore robustunder
stress-test conditions. The pipeline analysis further
reveals that SEAL-RAG’s apparent precision ad-
vantage stems from an implementation that departs
from its stated design, while AdaGATE’s multi-
document assembly and fallback retrieval provide
more robust evidence coverage at the cost of occa-
sional over conservative abstention.
6 Discussion
Hypothesis confirmation.Both central hypothe-
ses were confirmed. AdaGATE achieves the high-
est F1 across all three conditions (62.3% clean,
71.2% redundancy, 62.7% noise) while using 2.6 ×
fewer tokens than Adaptive- k. F1 under redun-
dancy improves relative to clean, and F1 under
noise matches clean performance despite accuracy
dropping, confirming that the utility scoring func-
tion maintains evidence completeness even when
individual passage quality deteriorates.
Relation to prior work.The precision-recall
tradeoff between SEAL-RAG and Adaptive- kmir-
rors the broader tension between context precision
and answer coverage in the RAG literature (Cu-
7

Question SEAL-RAG (L=1) AdaGATE (L=1)
Q1: Were Scott Der-
rickson and Ed Wood
of the same national-
ity?Retrieve:Dispatches 3 docs, extracts 3 enti-
ties. Micro query: None.
Repair:yes — both nationalities confirmed in
retrieved docs.
Rank:entity selection collapses to 1 doc; Ed
Wood passage discarded.
Generate:docs=1, tokens=97, P=1.00,
R=0.50, F1=0.67.
✓“Yes, both American.”Retrieve:Dispatches 2 docs, extracts 2 enti-
ties.
Repair:yes. Utility scores [0.19, 0.15].
Rank: Keff= 2, both gold docs passed to gen-
erator.
Generate:docs=2, tokens=136, P=1.00,
R=1.00, F1=1.00.
✓“Yes, both American.”
Q3: What science
fantasy YA series has
companion books
about enslaved
worlds?Retrieve:Dispatches 3 docs, extracts 7 en-
tities. Repair: no — Animorphs companion
books not confirmed.
Micro query:“young adult science fantasy
series companion books enslaved worlds alien
species.” WARNING: No documents to
extract entities from! No new docs re-
trieved.
Loop limit reached.Repair: no (unchanged).
Generate:docs=1, tokens=128, P=1.00,
R=0.50, F1=0.67.
✗“I don’t know.” gt: “Animorphs”Retrieve:Dispatches 2 docs, extracts 4 enti-
ties. Utility scores [0.23, 0.19]. Repair: no —
companion books not confirmed.
Micro query: Hgap(3 docs) ∪H qfallback (3
docs). 1 new doc dispatched for entity extrac-
tion; new entity about Hork-Bajir Chronicles
retrieved. Total: 9 entities, 3 docs. Utility
scores [0.32, 0.26, 0.23].
Repair:yes — Animorphs companion books
confirmed.K eff= 3, 3 docs passed.
Generate:docs=3, tokens=500, P=0.67,
R=1.00, F1=0.80.
✓“Animorphs series”
Q21: Which other
Mexican F1 driver
held the podium be-
sides Sergio Pérez?Retrieve:Dispatches 3 docs, extracts 4 enti-
ties including Pedro Rodríguez. Repair: no —
podium finish not explicitly confirmed.
Micro query:“Mexican Formula One drivers
podium finishes history.” Returns 1 new doc; 0
new entities extracted (entity count unchanged
at 4).
Repair:no — evidence still insufficient.
Loop limit reached.Generator infers from
partial evidence.
Generate:docs=1, tokens=100, P=1.00,
R=0.50, F1=0.67.
✓“Pedro Rodríguez” gt: “Pedro Rodríguez”Retrieve:Dispatches 2 docs, extracts 4 enti-
ties. Utility scores [0.24, 0.21]. Repair: no —
podium finish not confirmed.
Micro query: Hgap∪H qretrieves 6 docs; 1
dispatched for entity extraction, 0 new entities
(count unchanged at 4). Utility scores [0.38,
0.26, 0.25].
Repair:no — gap still open across both itera-
tions.
Loop limit reached.Generator abstains.
Generate:docs=3, tokens=511, P=0.67,
R=1.00, F1=0.80.
✗“I don’t know.” gt: “Pedro Rodríguez”
Table 1: Pipeline-level comparison of SEAL-RAG and AdaGATE on three representative HotpotQA examples
(clean, L=1 ). Q1 shows SEAL-RAG’s systematic single-document selection and its recall cost even when both
controllers answer correctly. Q3 shows AdaGATE’s Hqfallback channel recovering from a micro-query that returns
zero new documents for SEAL-RAG. Q21 shows AdaGATE’s conservative gap detection causing abstention while
SEAL-RAG’s less stringent sufficiency check produces the correct answer from partial evidence.
conasu et al., 2024). We additionally identify a
concrete implementation gap in SEAL-RAG: its
stated utility based ranking is replaced in prac-
tice by an LLM entity selection step that col-
lapses the evidence set to one document by con-
struction. The ARES analysis further reveals that
Adaptive- k’s inflated scores reflect context avail-
ability rather than evidence quality, consistent with
concerns about document conditioned evaluation
(Saad-Falcon et al., 2023).
Limitations and future work.The current eval-
uation uses k= 3 retrieval, which limits the can-
didate pool and prevents the token budget from
binding; larger kor longer repair chains would en-gage the budget more actively. Evaluation is also
restricted to HotpotQA’s distractor setting, and per-
formance on web-scale retrieval may differ. The
utility weights (λ1, . . . , λ 5)were set heuristically
and could be learned from supervision. Finally,
AdaGATE’s conservative gap detection leads to
over abstention on questions where indirect ev-
idence would suffice; calibrating the sufficiency
threshold to question type is a natural direction for
future work.
7 Conclusion
We presented AdaGATE, a training-free evidence
controller for multi-hop RAG that combines en-
8

tity centric gap tracking, a five dimensional utility
scoring function, adaptive capacity control, and a
question anchored fallback retrieval channel. Eval-
uated on HotpotQA across clean, redundancy and
noise injected conditions, AdaGATE achieves the
highest evidence F1 in all three settings while using
2.6×fewer tokens than Adaptive- k. A pipeline-
level analysis reveals that SEAL-RAG’s precision
advantage is an artifact of an implementation that
departs from its stated design, and that AdaGATE’s
multi-document assembly provides more robust ev-
idence coverage at the cost of occasional over con-
servative abstention. These results support view-
ing multi-hop RAG as a token-budgeted evidence
repair problem and suggest that explicit gap mod-
eling, redundancy-aware scoring, and fallback re-
trieval together constitute a principled approach to
robust evidence assembly under realistic deploy-
ment constraints.
Acknowledgments
This work was carried out as a course project
for DS-GA / LING-GA 1012 (Natural Language
Understanding and Computational Semantics) at
New York University, Spring 2026. We thank the
course instructor, Prof. Tal Linzen, and the teaching
staff for their feedback during in-semester presen-
tations. We also thank the authors of SEAL-RAG
(Lahmy and Yozevitch, 2025) for releasing their
code, which served as the starting point for our
implementation.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-rag: Learning to re-
trieve, generate, and critique through self-reflection.
Florin Cuconasu, Giovanni Trappolini, Federico Sicil-
iano, Simone Filice, Cesare Campagnano, Yoelle
Maarek, Nicola Tonellotto, and Fabrizio Silvestri.
2024. The power of noise: Redefining retrieval for
rag systems. InProceedings of the 47th Interna-
tional ACM SIGIR Conference on Research and De-
velopment in Information Retrieval, pages 719–729,
Washington, DC, USA. Association for Computing
Machinery.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on RAG meeting LLMs: To-
wards retrieval-augmented large language models. In
Proceedings of the 30th ACM SIGKDD Conference
on Knowledge Discovery and Data Mining, pages
6491–6501, New York, NY , USA. Association for
Computing Machinery.Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong C. Park. 2024. Adaptive-rag:
Learning to adapt retrieval-augmented large language
models through question complexity.
Moshe Lahmy and Roi Yozevitch. 2025. Replace, don’t
expand: Mitigating context dilution in multi-hop rag
via fixed-budget evidence assembly.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2023. Lost in the middle: How language
models use long contexts.
Chao Peng, Bin Wang, Zhilei Long, and Jinfang Sheng.
2025. Adagres: Adaptive greedy context selection
via redundancy-aware scoring for token-budgeted
rag.
Jon Saad-Falcon, Omar Khattab, Christopher Potts, and
Matei Zaharia. 2023. Ares: An automated evalua-
tion framework for retrieval-augmented generation
systems.Preprint, arXiv:2311.09476.
Chihiro Taguchi, Seiji Maekawa, and Nikita Bhutani.
2025. Efficient context selection for long-context qa:
No tuning, no iteration, just adaptive-k.
Johannes Welbl, Pontus Stenetorp, and Sebastian Riedel.
2018. Constructing datasets for multi-hop reading
comprehension across documents. InProceedings
of the 2018 Conference on Empirical Methods in
Natural Language Processing, pages 1933–1943.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
2024. Corrective retrieval augmented generation.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing. InProceedings of the 2018 Conference on Em-
pirical Methods in Natural Language Processing,
pages 2369–2380.
Appendix A Full Evaluation Results
Table 2 reports complete numerical results for all
controllers across all conditions and metrics.
9

Clean Redundancy (ρ=0.5) Noise (ρ=0.5)
Acc P R F1 Acc P R F1 Acc P R F1ControllerBasic RAG 58.5 .559 .665 .601 47.5 .862 .498 .615 59.5 .598 .630 .605
Self-RAG 60.5 .645 .500 .536 49.0 .681 .378 .476 59.0 .655 .485 .534
Adaptive-k69.0.278.820.40872.5.109.735.18764.5.286.800.414
SEAL-RAG (L=1) 62.0 .784 .420 .541 58.5 .868 .448 .587 57.0 .790 .408 .535
SEAL-RAG (L=3) 68.5.808.420 .549 65.0.876.455 .595 61.5.800.410 .540
ADAGATE (L=1, ours) 63.0 .555 .740.62359.5 .836 .663 .709 54.0 .571 .728.627
ADAGATE (L=3, ours) 64.5 .549 .745 .620 59.5 .817 .685.71257.5 .568 .735.627
Token efficiency (avg_tokens / avg_docs / tokens_per_correct)
Basic RAG 364 / 3.0 / 352 300 / 3.0 / 306 364 / 3.0 / 330
Self-RAG 234 / 1.9 / 254 259 / 2.8 / 311 211 / 1.7 / 240
Adaptive-k1116 / 8.6 / 1118 1592 / 14.1 / 1605 1073 / 8.8 / 1057
SEAL-RAG (L=1) 140 / 1.1 / 139127/1.0/122 131/1.0/126
SEAL-RAG (L=3)136/1.0/134128 / 1.1 / 125131/1.0/ 127
ADAGATE (L=1, ours) 360 / 2.9 / 338 220 / 1.9 / 185 328 / 2.8 / 302
ADAGATE (L=3, ours) 370 / 3.0 / 347 232 / 2.0 / 194 344 / 2.8 / 309
ARES scores (CR / AF / AR)
Basic RAG .50 / .44 / .44 .36 / .30 / .30 .42 / .38 / .36
Self-RAG .47 / .44 / .44 .34 / .31 / .29 .37 / .32 / .32
Adaptive-k.67/.63/.59 .75/.68/.67 .62/.57/.57
SEAL-RAG (L=1) .24 / .22 / .22 .25 / .23 / .22 .20 / .18 / .17
SEAL-RAG (L=3) .26 / .23 / .23 .27 / .26 / .25 .21 / .19 / .19
ADAGATE (L=1, ours) .58 / .51 / .51 .48 / .42 / .41 .51 / .44 / .45
ADAGATE (L=3, ours) .54 / .50 / .48 .53 / .48 / .45 .57 / .50 / .48
Table 2: Full evaluation results (HotpotQA distractor, k= 3 ,N= 200 ). Accuracy (%) judged by GPT-4o. P =
Precision, R = Recall, F1 against gold titles. Token efficiency: avg_tokens / avg_docs / tokens_per_correct. CR =
Context Relevance, AF = Answer Faithfulness, AR = Answer Relevance (ARES UES/IDP, GPT-4o).Bold= best
per metric per condition.
10
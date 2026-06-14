# CQC-RAG: Robust Retrieval-Augmented Generation via Cross-Query Consistency

**Authors**: Yanjia Sun, Sifan Liu, Jie Shao

**Published**: 2026-06-11 15:01:49

**PDF URL**: [https://arxiv.org/pdf/2606.13438v1](https://arxiv.org/pdf/2606.13438v1)

## Abstract
Retrieval-Augmented Generation (RAG) has become a common approach for improving the factuality of Large Language Models (LLMs), yet its reliability remains highly sensitive to how external evidence is retrieved and used. Semantically equivalent queries with different syntactic forms may lead to different retrieval results, while irrelevant or misleading documents can further induce hallucinated answers. Existing multi-path reasoning methods improve robustness by sampling multiple candidate answers and applying voting- or confidence-based selection, but they still face two limitations: diversity is often injected through uncontrollable decoding randomness, and answer evaluation is usually confined to a single query-induced evidence view. To address these limitations, we propose a Cross-Query Consistency Hypothesis: correct answers tend to maintain high confidence across semantically equivalent but syntactically diverse queries, whereas noise-induced hallucinations exhibit unstable confidence under such query variations. Based on this hypothesis, we introduce CQC-RAG, a framework that co-designs query-level diversity injection with cross-query consistency evaluation. CQC-RAG rewrites the original question into diverse but meaning-preserving queries, reranks a shared document pool to construct query-conditioned reasoning contexts, applies an evidence-grounded protocol to extract answer-evidence pairs and selects answers according to their confidence stability across these contexts. This design enables self-evaluation without external supervision and does not rely on expanded retrieval coverage. Experiments on four open-domain question answering benchmarks show that CQC-RAG outperforms the strongest previous multi-query baseline by +4.76 pp EM on TriviaQA and +9.12 pp EM on MuSiQue, validating the effectiveness of cross-query consistency for filtering noise-induced hallucinations.

## Full Text


<!-- PDF content starts -->

CQC-RAG: Robust Retrieval-Augmented Generation via
Cross-Query Consistency
Yanjia Sun
University of Electronic Science and
Technology of China
Chengdu, Sichuan, China
sunyanjia@std.uestc.edu.cnSifan Liu
University of Electronic Science and
Technology of China
Chengdu, Sichuan, China
liusifan@std.uestc.edu.cnJie Shao∗
University of Electronic Science and
Technology of China
Chengdu, Sichuan, China
shaojie@uestc.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) has become a common
approach for improving the factuality of Large Language Models
(LLMs), yet its reliability remains highly sensitive to how external
evidence is retrieved and used. Semantically equivalent queries
with different syntactic forms may lead to different retrieval results,
while irrelevant or misleading documents can further induce hallu-
cinated answers. Existing multi-path reasoning methods improve
robustness by sampling multiple candidate answers and applying
voting- or confidence-based selection, but they still face two limita-
tions: diversity is often injected through uncontrollable decoding
randomness, and answer evaluation is usually confined to a sin-
gle query-induced evidence view. To address these limitations, we
propose a Cross-Query Consistency Hypothesis: correct answers
tend to maintain high confidence across semantically equivalent
but syntactically diverse queries, whereas noise-induced halluci-
nations exhibit unstable confidence under such query variations.
Based on this hypothesis, we introduce CQC-RAG, a framework
that co-designs query-level diversity injection with cross-query
consistency evaluation. CQC-RAG rewrites the original question
into diverse but meaning-preserving queries, reranks a shared doc-
ument pool to construct query-conditioned reasoning contexts,
applies an evidence-grounded protocol to extract answer-evidence
pairs and selects answers according to their confidence stability
across these contexts. This design enables self-evaluation with-
out external supervision and does not rely on expanded retrieval
coverage. Experiments on four open-domain question answering
benchmarks show that CQC-RAG outperforms the strongest previ-
ous multi-query baseline by +4.76 pp EM on TriviaQA and +9.12 pp
EM on MuSiQue, validating the effectiveness of cross-query consis-
tency for filtering noise-induced hallucinations. Code is available
at https://github.com/FrancesPlus/CQC-RAG.
CCS Concepts
•Computing methodologies →Knowledge representation
and reasoning.
∗Corresponding author: Jie Shao.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference’17, Washington, DC, USA
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXXKeywords
Retrieval-augmented generation, Large language model, Cross-
query consistency
ACM Reference Format:
Yanjia Sun, Sifan Liu, and Jie Shao. 2026. CQC-RAG: Robust Retrieval-
Augmented Generation via Cross-Query Consistency. In.ACM, New York,
NY, USA, 13 pages. https://doi.org/XXXXXXX.XXXXXXX
1 Introduction
Retrieval-Augmented Generation (RAG) has become a widely adopted
paradigm for mitigating hallucinations and knowledge obsoles-
cence in Large Language Models (LLMs) by retrieving external
knowledge and incorporating it into the generation context [ 5,11].
Despite its effectiveness, RAG remains fragile in practice. On the
one hand, the specific formulation of a query significantly affects
retrieval quality: semantically equivalent queries that differ only in
syntactic structure, such as active versus passive voice or synonym
substitution, lead to an average of 15%–20% degradation in retrieval
performance across different datasets, with extreme cases exceed-
ing 40% [ 2]. On the other hand, retrieved documents may contain
information irrelevant to the question, and studies have shown that
such irrelevant inputs substantially interfere with the reasoning
process of models, causing errors and hallucinations [ 10]. These
observations suggest that answer reliability in RAG depends not
only on the language model, but also on how the question induces
and prioritizes external evidence.
A common strategy for improving robustness is to generate mul-
tiple reasoning paths and then select a final answer from them.
Self-Consistency [ 20] samples multiple reasoning paths from the
same query and selects the most consistent answer through ma-
jority voting, establishing the theoretical foundation that correct
answers tend to converge across diverse reasoning paths while in-
correct answers are scattered stochastically. SPARC-RAG [ 25] uses
adaptive sequential-parallel reasoning, while Speculative RAG [ 21]
adopts a draft-then-verify mechanism to produce and evaluate can-
didate answers. These methods improve robustness by exploiting
answer agreement across multiple generations. Yet, when applied to
open-domain RAG settings without external supervision, existing
methods still face two fundamental limitations:
•Challenge 1: Uncontrollability of the diversity injec-
tion mechanism.Existing approaches rely primarily on
token-level temperature sampling, which often struggles
with a trade-off between semantic drift (at high tempera-
tures) and path redundancy (at low temperatures). More
importantly, because all generated paths share the same ini-
tial query formulation, this decoding-level randomness failsarXiv:2606.13438v1  [cs.IR]  11 Jun 2026

Conference’17, July 2017, Washington, DC, USA Sun et al.
to address the inherent sensitivity of retrieval systems to
query formulations [2].
•Challenge 2: Limited discriminability of answer evalua-
tion.Conventional majority voting treats all reasoning paths
equally, and even recent confidence-weighted selections re-
main restricted to repeat inference over a single retrieval
perspective, leaving them vulnerable to errors consistently
reinforced by biased or noisy evidence [9, 15].
These two challenges are tightly coupled. The quality of diversity
injection directly determines the distributional characteristics of
candidate answers, which in turn affects the difficulty and accu-
racy of evaluation. Conversely, the effectiveness of the evaluation
mechanism determines whether the value of diversity injection
can be fully exploited. Therefore, diversity injection and answer
evaluation must be co-designed as an integrated whole.
To this end, we propose CQC-RAG, a Cross-Query Consistency
RAG framework that co-designs diversity injection and answer
evaluation as an integrated system. CQC-RAG is built on theCross-
Query Consistency Hypothesis, which posits that correct an-
swers exhibit stable high confidence across semantically equivalent
but syntactically diverse queries due to their grounding in gen-
uinely relevant documents. In contrast, noise-induced hallucinated
answers rely on spurious, context-dependent correlations, causing
significant confidence fluctuations across these query variations.
To operationalize this hypothesis, CQC-RAG first generates a set of
semantically equivalent but syntactically diverse queries through
query rewriting, introducing controlled diversity at the query level
rather than at the decoding level. All rewritten queries share the
same retrieved document pool, but a reranker reorders the doc-
uments according to the semantic relevance to each individual
query, producing differentiated priority orderings across queries
that constitute distinct reasoning contexts. The model performs
parallel reasoning over these contexts, generating diverse candi-
date answers. Finally, we compute the confidence of each answer
under each query perspective based on model logits and select the
answer that maintain high confidence across queries, characterized
by high mean and low variance, achieving answer self-evaluation
without external supervision signals. It is important to emphasize
that all queries share the same document pool, and the performance
gains come from the cross-query consistency framework rather
than from expanded retrieval coverage.
Through this pipeline, diversity injection and answer evaluation
are seamlessly coupled, as query-level variations do not act as
random decoding noise but rather systematically construct the
structured, multi-perspective contexts required for our consistency-
based evaluation.
We validate the effectiveness of CQC-RAG on open-domain Ques-
tion Answering (QA) datasets including TriviaQA and MuSiQue.
Experimental results demonstrate that CQC-RAG significantly out-
performs baseline methods such as Self-Consistency and Specula-
tive RAG, indicating that cross-query consistency evaluation can
effectively identify and filter hallucinated answers induced by noise,
thereby validating the Cross-Query Consistency Hypothesis.
Our technical contributions are summarized as follows:
•We propose the Cross-Query Consistency Hypothesis, which
establishes a theoretical connection between the stability ofanswer confidence across diverse query formulations and
answer correctness.
•We design CQC-RAG, a framework that co-designs query-
level diversity injection with cross-query consistency evalua-
tion. By reranking a shared document pool with semantically
equivalent but syntactically diverse queries to produce dif-
ferentiated reasoning contexts, CQC-RAG enables answer
verification across multiple knowledge perspectives without
increasing the retrieval budget.
•Extensive experiments on single-hop and multi-hop question-
answering benchmarks demonstrate the superior effective-
ness and efficiency of our method.
2 Related Work
Our proposed CQC-RAG framework lies at the intersection of im-
proving generation robustness and evaluating answer reliability.
Specifically, our work is most closely related to two lines of re-
search: (1) injecting query-level diversity to enhance the robustness
of RAG systems against contextual noise; (2) leveraging multi-path
reasoning and self-consistency to mitigate model hallucinations
and designing unsupervised evaluation mechanisms for response
verification. We review these closely related works in detail below.
2.1 Query-Level Diversity for Robust RAG
Retrieval-Augmented Generation (RAG) enhances LLMs by incorpo-
rating external documents into inference [ 5,11], but this also makes
generation sensitive to how evidence is retrieved and organized.
Irrelevant passages can induce over-reasoning and hallucinations
[22], while semantically equivalent query reformulations can sub-
stantially change retrieval recall and downstream answers [ 2]. This
suggests that robust RAG requires not only better evidence, but also
a more controlled way to expose evidence under query variation.
Existing query optimization methods address this issue mainly
from a retrieval perspective. Query rewriting methods such as
Rewrite-Retrieve-Read [ 13] and Adaptive-RAG [ 6] employ LLMs to
reformulate queries into more canonical forms, aiming to predict a
single optimal entry point for retrieval. SiReRAG [ 26] constructs en-
hanced indexing mechanisms that capture both semantically similar
and explicitly related information for multi-hop reasoning. RARE
[17] and DMQR-RAG [ 12] use iterative or multi-query rewriting to
improve coverage and precision. These methods show the utility
of query reformulation, but they generally use diversity to enlarge,
refine, or merge retrieved evidence.
Therefore, these improved methods do not directly resolve the
first challenge: how to inject controllable diversity for reasoning
rather than merely for retrieval. Once rewritten queries are merged
or optimized toward a single best query, their query-specific evi-
dence priorities are largely discarded; meanwhile, token-level sam-
pling still inherits the same query-induced retrieval bias. Our work
instead treats query diversity as a controlled way to construct
complementary reasoning contexts over a shared document pool,
leaving the detailed design to the method section.
2.2 Discriminative Answer Evaluation
The second challenge concerns how to select a reliable answer once
multiple candidate reasoning paths are available. Self-Consistency

CQC-RAG: Robust Retrieval-Augmented Generation via Cross-Query Consistency Conference’17, July 2017, Washington, DC, USA
q1
q2
q3α1
α2
α3E1
E2
E31 3 5
1 5 43 1 4
()S
αi Ei ො𝛼
(a) Framework of cross -query consistency RAGEvaluate the cross -query 
consistency score of each 
candidate answer and select 
the best as final answer
orgqRerank D based on paraphrased 
queries without external retrieval
Parallel 
rewritingD
αi Ei
q1 q2 q3
1()S
2()S
3()S
( ) ( ) Var ( )= − q Q q Q S P P E
ො𝛼
(Valid | , , ) P q E
(b) Cross -query consistency 
confidence estimation: Original query qi : Candidate answer ො𝛼: Final answer
orgq D: Retrieved  documents αi : Paraphrased query
Figure 1: Illustration of CQC-RAG. CQC-RAG employs a parallel rewriting mechanism to construct multiple queries that are
semantically equivalent yet syntactically diverse. This enables multi-perspective reasoning over the context, yielding several
candidate answers. The optimal answer is subsequently selected based on cross-query consistency confidence estimation.
[20] samples multiple reasoning paths from the same query and
chooses the most frequent answer, establishing the principle that
correct answers tend to converge while incorrect ones are more
scattered. This paradigm has inspired RAG-oriented extensions.
SPARC-RAG [ 25] adaptively allocates sequential and parallel rea-
soning with specialized agents, while Speculative RAG [ 21] follows
a draft-then-verify pipeline in which candidate answers are gener-
ated from document subsets and then scored by a verifier. However,
producing more candidates does not by itself guarantee that the
final selector can distinguish genuinely supported answers from
plausible errors.
Recent work improves selection mechanisms by using richer sig-
nals than raw majority. Taubenfeld et al. [ 15] show that confidence-
weighted voting can reduce the number of required samples by
exploiting the model’s own confidence signal, and emphasize the
importance of within-question discrimination. Ranked Voting [ 19]
replaces simple frequency counting with rank-based scores to better
preserve high-quality minority answers. Self-Certainty [ 9] mea-
sures the divergence between the predicted token distribution and
a uniform distribution, showing that correct and incorrect answers
can form separable clusters in confidence space. DistriVoting [ 24]
further models trajectory confidence as a mixture distribution and
links the separation between confidence modes to voting accuracy.
These findings collectively suggest that the model’s internal confi-
dence signals contain rich information for distinguishing correct
from incorrect answers within the same query context.Nevertheless, they still largely evaluate answers under a single
query-conditioned evidence view. When biased or noisy evidence
is prioritized under a query-conditioned context, wrong answers
may appear plausible or confident, leaving the second challenge
unresolved: the selector lacks a discriminative axis beyond repeated
inference over the same retrieval perspective. CQC-RAG addresses
this limitation through cross-query consistency evaluation, which
tests answer reliability beyond a single query-conditioned evidence
view.
3 Methodology
In this section, we present the CQC-RAG framework. Figure 1 and
Algorithm 1 provide an overview of the proposed CQC-RAG.
We begin by introducing some preliminaries, followed by the
query rewriting mechanism designed to generate semantically
equivalent but syntactically diverse query variants. We then de-
scribe the cross-query reasoning and evidence extraction process.
Finally, we detail the cross-query consistency confidence estimation
strategy for candidate answer selection.
3.1 Preliminaries
For tasks that heavily rely on external knowledge, a standard RAG
instance can be formalized as the triplet (𝑞,D,𝛼) . Here,𝑞denotes
the input query;Drepresents the retrieved external knowledge cor-
pus containing large amounts of factual information; and 𝛼denotes

Conference’17, July 2017, Washington, DC, USA Sun et al.
Algorithm 1Framework of CQC-RAG
Input:Original query𝑞 𝑜𝑟𝑔, Retrieved documentsD
Parameter: Paraphrase count 𝑁, Penalty coefficient 𝜆(·), Models
{𝑀rewrite,𝑀rerank,𝑀reason,𝑀eval}
Output:Final answer ˆ𝛼
Stage 1: Parallel Rewriting
1:Q𝑝𝑎𝑟𝑎←𝑀 rewrite(𝑞𝑜𝑟𝑔,𝑁)⊲Generate diverse queries under
constraints
2:Q←{𝑞 𝑜𝑟𝑔}∪Q𝑝𝑎𝑟𝑎 ⊲Form the multi-view query set
Stage 2: Multi-view Reranking & Reasoning
3:Initialize candidate setA←∅
4:foreach𝑞 𝑖∈Qin parallel do
5:𝐶𝑖←𝑀 rerank(𝑞𝑖,D)⊲Obtain view-specific context
6:A←A∪{𝑀 reason(𝐶𝑖,𝑞𝑖)}⊲Generate pair(𝛼 𝑖,𝐸𝑖)via strict
protocol
7:end for
Stage 3: Candidate Selection
8:foreach candidate(𝛼 𝑘,𝐸𝑘)∈Ado
9:𝑃𝑠𝑐𝑜𝑟𝑒𝑠←∅
10:foreach query view𝑞 𝑗∈Qdo
11:(ℓ Yes,ℓNo)←𝑀 eval(Valid|𝛼𝑘,𝐸𝑘,𝑞𝑗)
12:𝑃 𝑠𝑐𝑜𝑟𝑒𝑠.append(exp(ℓ Yes)
exp(ℓ Yes)+exp(ℓ No))⊲Normalize logits
13:end for
14:𝜇𝑘←Mean(𝑃 𝑠𝑐𝑜𝑟𝑒𝑠)
15:𝑆(𝛼 𝑘)←𝜇𝑘−𝜆(𝜇𝑘)·Var(𝑃𝑠𝑐𝑜𝑟𝑒𝑠)
16:end for
17:return ˆ𝛼←arg max 𝛼𝑘𝑆(𝛼𝑘)
the expected correct answer. The primary goal of an RAG system
is to retrieve a relevant contextual subset 𝐶={𝑐 1,𝑐2,...,𝑐𝑚}⊂D
associated with 𝑞, and leverage the informational content within
to formulate a coherent and accurate response𝛼.
In our proposed framework, we denote the user’s original query
as𝑞𝑜𝑟𝑔. Through the parallel rewriting mechanism, we generate
a set of semantically isomorphic paraphrased queries, denoted as
Q𝑝𝑎𝑟𝑎={𝑞 1,𝑞2,...,𝑞𝑁}. Since the original query 𝑞𝑜𝑟𝑔is also in-
volved during inference, the entire set of queries utilized for reason-
ing is denoted asQ={𝑞𝑜𝑟𝑔,𝑞1,𝑞2,...,𝑞𝑁}. For each query in this set,
the model performs parallel inference to yield a corresponding set
of candidate answers and evidence spans, denoted as A={𝛼𝑖}|Q|
𝑖=1.
Our objective is to identify the final answer ˆ𝛼fromAthat exhibits
the highest cross-query consistency.
3.2 Query Rewriting for Controlled Diversity
Existing multi-path reasoning methods, exemplified by Self-Consistency
[20], derive diversity from temperature sampling during decoding.
All reasoning paths share the same query and the same reason-
ing context. As discussed in Section 1, this approach suffers from a
fundamental limitation: the diversity is uncontrollable (high temper-
ature risks unfaithfulness while low temperature yields redundant
paths), and it cannot address the sensitivity of retrieval outcomes
to query formulation [ 2]. Multi-query rewriting approaches such as
DMQR-RAG [ 12] generate diverse query variants across multiple
information levels (information-equivalent, information-expanded,
and information-reduced) to improve retrieval coverage. However,
DMQR-RAG merges all retrieved documents into a single pool for
one-pass generation, using query diversity solely to expand thedocument pool rather than to produce differentiated reasoning
contexts for independent inference.
Our approach takes a distinct perspective: we leverage query
rewriting not to expand retrieval coverage, but to produce multiple
differentiated reasoning contexts based on the same fixed document
poolDvia query-specific reranking, enabling cross-query answer
verification. The key insight, formalized as the Cross-Query Con-
sistency Hypothesis in Section 1, is that a correct answer should
maintain high confidence across diverse query perspectives, while
a hallucinated answer supported by spurious or noisy evidence
should exhibit instability under query variation. This verification
objective imposes a specific requirement on the rewriting strat-
egy: all rewritten queries must be strictly semantically equivalent
to𝑞𝑜𝑟𝑔, targeting the same factual answer. Unlike DMQR-RAG’s
information-level rewrites that deliberately alter the semantic scope
to retrieve broader or more focused documents, our rewriting pre-
serves strict semantic equivalence because any semantic drift be-
tween queries would invalidate the premise of cross-query con-
sistency. Answers generated from queries with different semantic
scopes are not meaningfully comparable for consistency evaluation.
We formalize this reasoning process as follows. Let 𝐼denote
the user’s latent semantic intent and let 𝑃(𝑞|𝐼) represent the
distribution over possible query formulations that express the same
intent𝐼. Since the reasoning context is determined by the query
through reranking, we write 𝐶(𝑞) for the top-𝑘context produced by
rerankingDwith respect to 𝑞. The probability of answer 𝛼given
intent𝐼can then be expressed as:
𝑃(𝛼|𝐼)=∫
𝑃(𝛼|𝐶(𝑞),𝑞)𝑃(𝑞|𝐼)𝑑𝑞.(1)
Since this integral is intractable in the discrete text space, we ap-
proximate it by averaging over the finite sample set Qof rewritten
queries, each treated as a sample from𝑃(𝑞|𝐼):
𝑃(𝛼|𝐼)≈1
|Q|∑︁
𝑞𝑖∈Q𝑃(𝛼|𝐶𝑖,𝑞𝑖),(2)
where𝐶𝑖=𝐶(𝑞𝑖)denotes the query-specific reasoning context ob-
tained by reranking Dwith respect to 𝑞𝑖. Unlike standard ensemble
methods where all members share the same input, each 𝑞𝑖here
produces its own reasoning context 𝐶𝑖from the shared document
pool. This distinction is operationally critical: the averaging process
attenuates the spurious, context-dependent correlations that may
accidentally support a hallucinated answer under a single query
formulation, while preserving semantic signals that are consistently
supported across multiple reasoning contexts.
To maximize the quality of this approximation, the paraphrased
query setQ𝑝𝑎𝑟𝑎must satisfy two requirements: (1) strict semantic
equivalence with the user’s intent 𝐼, ensuring all queries target the
same factual answer; and (2) maximal syntactic diversity, ensuring
different queries induce different document orderings and thus
different reasoning contexts. We implement this through a few-shot
prompting strategy governed by two dimensions of constraints:
(1)Hard constraints.Given that any alteration of named enti-
ties may lead to retrieval drift, we instruct the LLM to strictly
freeze all named entities, preserving their literal forms to
ensure all generated 𝑞𝑖are anchored to the same factual
baseline.

CQC-RAG: Robust Retrieval-Augmented Generation via Cross-Query Consistency Conference’17, July 2017, Washington, DC, USA
(2)Soft constraints.We restructure 𝑞𝑜𝑟𝑔across three dimen-
sions to maximize syntactic diversity while maintaining se-
mantic equivalence:
•Lexical perturbation: Synonym substitution for non-entity
predicates to resolve lexical mismatch between queries and
documents, thereby increasing the likelihood that different
document passages are promoted during reranking.
•Syntactic restructuring: Alteration of the syntactic tree
structure through active/passive voice switching and clause
reordering, producing queries that emphasize different as-
pects of the same intent.
•Pragmatic modal shift: Mixing interrogative forms with
imperative instructions. This hybrid strategy produces
queries that stylistically align with diverse document gen-
res, accommodating passages written in different rhetori-
cal modes.
3.3 Cross-Query Reasoning
Given the rewritten query set Q={𝑞𝑜𝑟𝑔,𝑞1,𝑞2,...,𝑞𝑁}, we per-
form cross-query reranking and reasoning to translate the query-
level diversity established in the previous stage into differentiated
reasoning outcomes.
Query-specific reranking.For each query 𝑞𝑖∈Q, we apply a
reranking model 𝑀rerank over the shared initial document pool D
to construct a query-specific reasoning context 𝐶𝑖=𝑀 rerank(𝑞𝑖,D).
Since the reranker scores each document based on its semantic
relevance to the specific query 𝑞𝑖, syntactically different queries
naturally produce different document orderings and thus different
top-𝑘contexts. Crucially, no new documents are introduced dur-
ing this process and the only variation is in which passages are
promoted to the top- 𝑘positions. This design isolates the effect of
document prioritization from retrieval coverage: any performance
gains of CQC-RAG over single-query baselines cannot be attributed
to seeing more documents, but rather to the cross-query consistency
mechanism that operates on the resulting variation in reasoning
contexts.
Evidence-grounded reasoning.We distribute the query-specific
reasoning contexts {𝐶𝑖}and their corresponding queries to the rea-
soning model 𝑀reason in parallel. To enforce faithfulness to the
source text and mitigate hallucinations, we implement a strict
evidence-grounded reasoning protocol: the LLM is explicitly in-
structed to first locate and extract the specific segment from 𝐶𝑖
that answers the query, defined as the supporting evidence span 𝐸𝑖.
Subsequently, the model is constrained to derive the final answer 𝛼𝑖
based solely on 𝐸𝑖. All|Q|reasoning paths execute in parallel, en-
abling the framework to scale with minimal latency overhead. This
process yields a set of reasoning tuples {(𝛼𝑖,𝐸𝑖)}|Q|
𝑖=1. These tuples
serve as the candidate pool for cross-query consistency evaluation.
3.4 Cross-Query Consistency Confidence
Estimation
The core contribution of CQC-RAG lies in shifting the paradigm of
answer evaluation from a single reasoning context to cross-query
consistency. Existing confidence estimation approaches—whether
based on majority voting frequency [ 20], confidence-weighted vot-
ing [ 15], token distribution certainty [ 9], or distributional filtering[24]—evaluate candidates within the same query’s reasoning con-
text and cannot reliably detect hallucinations reinforced by biased
or noisy evidence under one particular query formulation.
For any candidate answer 𝛼and its supporting evidence span 𝐸,
we define the cross-query consistency score𝑆(𝛼)in Eq. (3):
𝑆(𝛼)=E𝑞∼Q[𝑃(Valid|𝛼,𝐸,𝑞)]
|                         {z                         }
Semantic Consensus−𝜆(𝜇)·Var 𝑞∼Q[𝑃(Valid|𝛼,𝐸,𝑞)]
|                            {z                            }
Cross-Query Instability,
(3)
where𝜇=E𝑞∼Q[𝑃(Valid|𝛼,𝐸,𝑞)] denotes the mean validation
score. The evaluation is performed over answer-evidence pairs
(𝛼,𝐸) rather than over the full reasoning context 𝐶𝑖: since𝐸distills
the relevant information from 𝐶𝑖into a focused evidence span, the
evaluator can assess whether the evidence semantically supports
the answer under each query perspective without being confounded
by irrelevant passages. Importantly, since 𝐸is a direct extraction
of factual content from the source documents, its validity as sup-
porting evidence does not depend on the specific syntactic form of
the query—any semantically equivalent query formulation should
recognize the same factual statement as valid support.
To compute 𝑃(Valid|𝛼,𝐸,𝑞) , we leverage the auto-regressive
nature of the evaluator model 𝑀eval. Rather than generating a free-
form judgment (which would introduce additional variance from
the generation process itself), we perform a single forward pass
over a verification prompt and extract the logits corresponding
to the “Yes” and “No” tokens ( ℓYes,ℓNo), normalizing them over a
restricted binary space:
𝑃(Valid|𝛼,𝐸,𝑞)=exp(ℓ Yes)
exp(ℓ Yes)+exp(ℓ No).(4)
This logits-based evaluation offers two advantages over generation-
based alternatives: (1) it eliminates the stochasticity of open-ended
generation, producing deterministic confidence estimates that en-
able meaningful cross-query comparison; and (2) the restricted
binary normalization mitigates the acquiescence bias where models
disproportionately favor affirmative responses during open-ended
generation.
To further penalize candidates that achieve artificially inflated
scores through superficial matching on a minority of views, we
design the penalty coefficient to be adaptive: 𝜆(𝜇)=𝜆 0·𝜇, where𝜆0
is a base hyperparameter. This dynamic mechanism imposes stricter
variance constraints (higher penalty) on candidates with high mean
scores, ensuring their overall rank is driven by genuine cross-query
robustness rather than being skewed by a few extreme outlier
scores. Conversely, it provides necessary tolerance for candidates
with generally lower confidence.
To mitigate the tendency of models to rate their own generations
higher than others [ 27], in our experiments we employ a separate
LLM distinct from the inference model to perform this confidence
evaluation task.
4 Experiments
4.1 Datasets and Evaluation Metrics
To demonstrate the effectiveness of CQC-RAG, we evaluate CQC-
RAG on four open-domain question answering benchmarks span-
ning diverse retrieval challenges:

Conference’17, July 2017, Washington, DC, USA Sun et al.
(1)TriviaQA[ 8] is a large-scale dataset that requires factual
knowledge retrieval. We evaluate our method on a standard open-
domain split containing 2,000 queries. The associated retrieval cor-
pus is constructed from a comprehensive Wikipedia dump, and
candidate passages are retrieved using a dense retriever.
(2)PopQA[ 14] is a long-tail knowledge dataset. To explicitly
evaluate the model’s reliance on external knowledge, we use its
challenging long-tail subset, which comprises 1,399 queries. In this
subset, the target entities receive fewer than 100 monthly Wikipedia
page views, making parametric memory insufficient and external
retrieval essential.
For both single-hop benchmarks, we adopt the same experimen-
tal settings as [1].
(3)HotpotQA[ 4] and (4)MuSiQue[ 18] are two multi-hop QA
datasets involving comparison, synthesis, and bridge-type reason-
ing across multiple documents. Following the standard multi-hop
evaluation setting established in previous work [ 4], we randomly
sample 1,000 queries from the validation set of each dataset and
extract all candidate passages (both supporting facts and distrac-
tors) from the original annotations of the sampled queries. This
distractor-inclusive corpus setting rigorously tests the model’s abil-
ity to identify multi-hop reasoning paths amid highly confusing
but topically related information.
We adopt two complementary metrics following standard open-
domain QA evaluation practices:
•Exact Match (EM): The predicted answer string is nor-
malized and compared against all acceptable ground-truth
answers after the same normalization. A prediction scores 1
if the normalized prediction exactly equals any normalized
ground-truth, and 0 otherwise. The maximum score across
all ground-truth variants is taken for each sample.
•F1-score (F1): The normalized prediction and ground-truth
are tokenized by whitespace. Precision is computed as the
fraction of prediction tokens appearing in the ground-truth,
recall as the fraction of ground-truth tokens appearing in
the prediction, and F1-score is the harmonic mean. Again,
the maximum over all ground-truth variants is reported.
We report the average of EM and F1 as a combined indicator.
4.2 Baselines
We compare CQC-RAG with some widely-used baseline methods.
(1)Standard RAG.We evaluate three representative instruction-
tuned LLMs under a standard retrieve-then-generate pipeline:
Mistral-7B-Instruct [ 7], Qwen3-8B [ 23], and LLaMA-3.1-8B-
Instruct [ 16]. Each model retrieves the top- 𝑘documents and
generates answers with greedy decoding.
(2)Multi-path RAG.These methods generate multiple candi-
date answers and apply diverse selection strategies. Self-
Certainty [ 9] computes a KL-divergence-based certainty
score over the full output vocabulary distribution and se-
lects the answer with the highest certainty. DMQR-RAG [ 12]
generates multiple query variants across different informa-
tion levels, merges the retrieved documents from all variants
into a single pool, and performs one-pass generation on the
merged context. Speculative RAG [ 21] clusters the retrieved
documents into diverse subsets and generates multiple draftanswers via a lightweight drafting model, then employs a
larger verifier model to select the best draft based on self-
reflection scores. SPARC-RAG [ 25] implements an adaptive
sequential-parallel multi-agent framework where specialized
agents handle query rewriting, context management, answer
generation, and answer evaluation across iterative rounds.
We evaluate its unfinetuned version using Qwen3-8B as the
backbone for all agent components.
For a fair comparison, all multi-path baselines are evaluated with
the same backbone model and retrieval infrastructure as CQC-RAG,
while their candidate counts and sampling temperatures follow the
configurations reported in the original papers.
4.3 Experimental Settings
4.3.1 Implementation Details.We implement CQC-RAG using Qwen3-
8B as𝑀reason (with thinking mode disabled for fair comparison) and
Mistral-7B-Instruct-v0.2 as𝑀 eval. For query rewriting, we set tem-
perature=0.7 to ensure sufficient syntactic diversity; for reasoning,
we use greedy decoding (temperature=0) to produce deterministic
answers. The logits-based evaluation is inherently deterministic.
We set𝑁= 4(determined by the sensitivity analysis) and 𝜆0=0.5
as default hyperparameters. For the reranker model, we use BGE-
reranker-v2-m3 [ 3] to obtain the top- 𝑘query-specific context for
each query variant. We set the reranking cutoff 𝑘=5for single-
hop datasets (TriviaQA, PopQA) and 𝑘=10for multi-hop datasets
(MuSiQue, HotpotQA), since multi-hop questions require synthesiz-
ing evidence across multiple documents and benefit from a larger
reasoning context. This top- 𝑘setting is applied uniformly to both
CQC-RAG and all baselines to ensure a fair comparison.
4.3.2 Reproducibility.The only source of randomness in CQC-
RAG is the query rewriting stage (temperature=0.7). For the main
comparison (Table 1), we report results averaged over 5 random
seeds (42, 123, 456, 789, 1024) with standard deviation to ensure
statistical robustness. More details regarding parallel rewriting can
be found in Appendix. For analysis experiments, we report results
from a single representative run to isolate the effect of individual
components; the relative performance trends are consistent across
seeds. All other pipeline components are fully deterministic.
4.4 Main Results
CQC-RAG achieves state-of-the-art overall performance.CQC-
RAG attains the highest Avg on three out of four datasets, including
both single-hop benchmarks (TriviaQA: 65.19%; PopQA: 50.03%)
and the multi-hop (MuSiQue: 38.38%). On TriviaQA, CQC-RAG
achieves +4.72 percentage points Avg advantage over DMQR-RAG,
and on MuSiQue the improvement over SPARC-RAG reaches +2.24
percentage points Avg. Compared with its own backbone model
Qwen3-8B under standard RAG, CQC-RAG improves TriviaQA EM
by +10.47 percentage points and MuSiQue EM by +20.32 percentage
points, demonstrating that our cross-query consistency mechanism
extracts substantially more reasoning capability from the identical
retrieval infrastructure and base LLM.
Superiority over multi-path selection methods.Self-Certainty,
which applies confidence-based selection over temperature-sampled
candidates from the same reasoning context, achieves modest gains

CQC-RAG: Robust Retrieval-Augmented Generation via Cross-Query Consistency Conference’17, July 2017, Washington, DC, USA
Table 1: Performance comparison on the single-hop (TriviaQA, PopQA) and multi-hop (MuSiQue, HotpotQA) datasets using EM,
F1, and average scores.
MethodSingle-hop Multi-hop
TriviaQA PopQA MuSiQue HotpotQA
EM F1 AVG. EM F1 AVG. EM F1 AVG. EM F1 AVG.
Standard RAG
Mistral-7B-Instruct 39.13 54.21 46.67 25.73 36.79 31.26 3.80 14.13 8.97 6.60 16.62 11.61
Llama-3.1-8B-Instruct 54.35 64.40 59.38 42.96 50.35 46.66 14.50 21.22 17.86 16.50 23.19 19.85
Qwen3-8B 49.18 63.61 56.40 40.67 49.59 45.13 12.30 24.32 18.31 12.60 25.63 19.12
Multi-path RAG
Self-Certainty 51.10±0.40 61.30±0.40 56.20 42.00±0.20 48.00±0.30 45.00 9.40±0.10 15.20±0.30 12.30 52.20±0.40 65.00±0.30 58.60
Speculative RAG 54.44 64.03 59.24 37.88 42.29 40.09 14.90 20.56 17.73 33.00 42.05 37.53
DMQR-RAG 54.89 66.04 60.47 46.46 53.3149.89 23.50 33.92 28.7154.70 68.56 61.63
SPARC-RAG 36.68 47.34 42.01 30.38 40.59 35.49 30.50 41.78 36.14 42.20 58.99 50.60
CQC-RAG 59.65±0.35 70.73±0.29 65.19 46.86±0.2853.19±0.22 50.03 32.62±0.30 44.13±0.25 38.3852.50±0.59 67.60±0.44 60.05
on single-hop datasets (TriviaQA EM 51.10%). However, its perfor-
mance severely degrades on MuSiQue (EM 9.40%), where complex
multi-hop queries amplify retrieval noise that single-perspective
confidence signals fail to detect. CQC-RAG circumvents this epis-
temic blind spot by verifying answer stability based on cross-query
consistency, delivering a massive +23.22 percentage points EM
improvement over Self-Certainty on MuSiQue.
Advantage over multi-query expansion.DMQR-RAG also
employs query rewriting but merges all retrieved documents into
a single pool for one-pass generation, leveraging query diversity
solely to maximize retrieval coverage. In contrast, CQC-RAG out-
performs DMQR-RAG in EM across TriviaQA (+4.76 percentage
points), MuSiQue (+9.12 percentage points), and PopQA (+0.40
percentage points). Because both methods share the exact same
retrieval infrastructure and CQC-RAG does not expand the doc-
ument budget (all rewritten queries rerank the same fixed pool),
these gains are strictly attributable to the cross-query consistency
evaluation rather than broader document recall. This is further cor-
roborated on PopQA: both methods achieve near-identical F1 scores,
indicating comparable semantic coverage, yet CQC-RAG’s higher
EM highlights its superior precision in exact answer extraction.
Robustness against multi-agent reasoning.SPARC-RAG em-
ploys iterative multi-agent reasoning with adaptive depth and width
control. While it achieves competitive performance on MuSiQue, its
efficacy drops precipitously on single-hop tasks (TriviaQA: 36.68%
EM; PopQA: 30.38% EM). We attribute this to the tendency of its
unfinetuned multi-agent pipeline to generate comprehensive but
excessively verbose responses, severely penalizing exact-match
evaluation on datasets demanding concise entity extraction. CQC-
RAG’s evidence-grounded protocol and cross-query consistency
scoring provide more effective precision control, achieving strong
performance across both single-hop and multi-hop tasks without
such precision degradation.
Analysis of HotpotQA performance.HotpotQA is the only
dataset where CQC-RAG ranks second (Avg 60.05%), trailing DMQR-
RAG. We attribute this gap to HotpotQA’s structural reliance on
document titles as bridge entities: questions frequently require
matching specific titles, where the retrieval signal is title matching
rather than content matching. Since our rewriting preserves entities
and modifies only content-level syntax, it produces limited variationin title-level relevance, reducing the effectiveness of query-specific
reranking on this dataset. Notably, DMQR-RAG’s information-
expanded queries can introduce additional title-relevant terms that
directly improve title-level retrieval, explaining its advantage on
this particular dataset structure.
4.5 Ablation Study and Selection Strategy
Analysis
4.5.1 Component Ablation.To quantify the contribution of each
CQC-RAG component, we systematically remove or replace individ-
ual modules while keeping the rest of the pipeline intact (Table 2).
All variants use 𝑁= 4paraphrases from a single representative
run. We organize the analysis by the three stages of CQC-RAG’s
pipeline: rewriting, reasoning, and evaluation.
Rewriting stage.Thew/o Rewritingvariant replaces the query
rewriting mechanism with temperature sampling: the original query
is used for a single reranking pass, and 𝑁=4 diverse answers are
generated via stochastic decoding on the shared reasoning context.
Thew/o Query-specific Rerankingvariant retains query rewriting
but reranks only with 𝑞𝑜𝑟𝑔, sharing the identical top- 𝑘document
ordering across all query variants.
On TriviaQA, removing rewriting reduces EM by 0.52 percentage
points and removing query-specific reranking reduces EM by 1.24
percentage points, confirming that both components contribute
to producing the differentiated reasoning contexts that CQC-RAG
relies on for cross-query verification. The larger degradation from
removing reranking indicates that even with diverse queries, the
consistency mechanism cannot function effectively without differ-
entiated document orderings.
On MuSiQue, both variants show EM decreases while F1 in-
creases. Forw/o Rewriting, temperature sampling generates mul-
tiple candidate answers from the same reasoning context. Since
answers to multi-hop questions typically require compositional rea-
soning over multiple evidence pieces, temperature sampling may
occasionally produce answers that contain partially correct enti-
ties, attributes, or descriptive spans, thereby improving token-level
F1. However, such answer-level diversity cannot replace the multi-
path evidence construction enabled by query rewriting. Without
rewritten queries, the model can only reason over a single retrieval
context, making it more prone to errors in final answer boundaries,

Conference’17, July 2017, Washington, DC, USA Sun et al.
Table 2: Ablation study of CQC-RAG in the rewriting, reasoning and evaluation stages on TriviaQA and MuSiQue.
MethodTriviaQA MuSiQue
EM F1 AVG. EM F1 AVG.
CQC-RAG 60.39 71.57 65.98 32.9044.13 38.52
Rewriting Stage
w/o Rewriting 59.87 (-0.52 pp) 70.97 (-0.60 pp) 65.42 (-0.56 pp) 32.50 (-0.40 pp) 44.43 (+0.30 pp) 38.47 (-0.05 pp)
w/o Query-specific Reranking 59.15 (-1.24 pp) 69.75 (-1.82 pp) 64.45 (-1.53 pp) 32.50 (-0.40 pp)46.10(+1.97 pp)39.30(+0.78 pp)
Reasoning Stage
w/o Evidence-grounded Protocol 44.47 (-15.92 pp) 58.91 (-12.66 pp) 51.69 (-14.29 pp) 24.30 (-8.60 pp) 37.93 (-6.20 pp) 31.12 (-7.40 pp)
Evaluation Stage
w/o Cross-query Evaluation 59.57 (-0.82 pp) 70.93 (-0.64 pp) 65.25 (-0.73 pp) 31.40 (-1.50 pp) 41.84 (-2.29 pp) 36.62 (-1.90 pp)
w/o Separate Evaluator 57.83 (-2.56 pp) 70.08 (-1.49 pp) 63.96 (-2.02 pp) 32.60 (-0.30 pp) 43.95 (-0.18 pp) 38.33 (-0.19 pp)
entity normalization, or compositional reasoning, which leads to a
decrease in exact match.
Forw/o Query-specific Reranking, the increase in F1 can be attrib-
uted to the fact that candidate evidence not ranked under query-
specific constraints may contain more content that partially over-
laps with the gold answer, thereby improving partial-match F1.
However, because all rewritten query variants share the same top- 𝑘
document ordering determined by 𝑞𝑜𝑟𝑔, documents that are more
relevant to a particular rewritten query may not be promoted to
higher ranks. This weakens the alignment between each query
variant and its corresponding evidence, making it harder for the
model to produce a fully matched final answer. In other words, the
F1 improvements observed in the ablated models on the multi-hop
dataset mainly reflect increased partial-match coverage rather than
improved final-answer precision.
Reasoning stage.Thew/o Evidence-grounded Protocolvariant
removes the “extract evidence first, then answer” constraint, instead
directly generating answers from the full reasoning context without
structured evidence extraction. This causes the most severe degra-
dation across both datasets: TriviaQA EM drops by 15.92 percentage
points and MuSiQue EM by 8.60 percentage points. Without evi-
dence grounding, the model is unconstrained by source text and
produces fluent but unfaithful answers.
Evaluation stage.Thew/o Cross-query Evaluationvariant eval-
uates each candidate answer only against its originating query,
eliminating the cross-query verification. This reduces MuSiQue EM
by -1.50 percentage points and F1 by -2.29 percentage points, the
largest evaluation-stage degradation, confirming that cross-query
verification is most valuable when multiple evidence fragments
must be consistently validated. Thew/o Separate Evaluatorvariant
uses the reasoning model (Qwen3-8B) itself as the evaluator instead
of the independent Mistral-7B. This causes TriviaQA EM to drop
by -2.56 percentage points, the second-largest overall degradation,
validating that self-enhancement bias [ 27] materially compromises
answer discrimination when the same model both generates and
evaluates answers. We further investigate this effect in the evaluator
model analysis below.
4.5.2 Selection Strategy Comparison.To further validate the effec-
tiveness of CQC-RAG’s cross-query consistency scoring, we fix the
candidate pool produced by CQC-RAG’s rewriting and reasoning
stages and apply five alternative selection strategies, as shown inTable 3: Comparison of the effects of different selection
strategies.
MethodTriviaQA MuSiQue
EM F1 AVG. EM F1 AVG.
Majority Voting 58.24 68.41 63.33 27.00 40.32 33.66
Confidence Score
Probability 58.34 68.97 63.66 28.40 38.70 33.55
Perplexity 58.14 68.76 63.45 27.80 38.26 33.03
Yes-Probability 58.96 70.34 64.65 26.80 37.82 32.31
LLM-as-a-Judge 57.93 69.82 63.88 27.10 41.02 34.06
CQC-RAG 60.39 71.57 65.98 32.90 44.13 38.52
Table 3. This controls for the quality of candidates and tests only
the selection mechanism.
Specifically,Majority Votingclusters semantically equivalent
answers using BGE-M3 embeddings and selects the cluster with
the most members.Probabilityselects the answer with the highest
generation probability (product of per-token logprobs).Perplexity
selects the answer with the lowest perplexity.Yes-Probabilitycom-
putes the probability of the model generating “Yes” given the can-
didate answer, the retrieved evidence, and its originating question—
serving as a single-query version of CQC-RAG’s logits-based eval-
uation.LLM-as-a-Judgedirectly employs an LLM as an evaluator
by providing all candidate answers and the original question to
𝑀eval(Mistral-7B), prompting it to select the best answer through
free-form generation.
All alternative strategies perform within a narrow band.
On TriviaQA, the five alternatives achieve Avg scores between
63.33% and 64.65%, a spread of only 1.32 percentage points. CQC-
RAG breaks out of this range at 65.98%, which is +1.33 percentage
points over the best alternative Yes-Probability. On MuSiQue, the
gap is more pronounced: alternatives range from 32.31% to 34.06%
Avg, while CQC-RAG achieves 38.52%. This confirms that the cross-
query consistency dimension provides a qualitatively different and
stronger signal for answer discrimination than any single-context
metric and is especially effective for multi-hop reasoning where
noise-induced hallucinations are more prevalent.
Confidence-based metrics cannot distinguish noise-induced
hallucinations.Probability, Perplexity, and Yes-Probability each

CQC-RAG: Robust Retrieval-Augmented Generation via Cross-Query Consistency Conference’17, July 2017, Washington, DC, USA
2 3 4 5 6 7 8 9 10
Number of paraphrases5560657075Performance (%) 60.29 60.08 60.18 60.08 59.8859.47 59.57 59.26 59.2671.11 70.92 71.04 71.03 70.8870.50 70.58 70.3970.73
N=4 (selected)
EM
F1
(a) TriviaQA
2 3 4 5 6 7 8 9 10
Number of paraphrases253035404550Performance (%) 32.60 32.70 33.00 33.20 33.40 33.30 33.40 33.50 33.5045.37 45.56 45.8246.26 46.34 46.25 46.24 46.35 46.44
N=4 (selected)
EM
F1
(b) MuSiQue
Figure 2: Performance comparison on the TriviaQA and
MuSiQue datasets across different paraphrase counts.
evaluate answers using a single confidence signal without cross-
query comparison. Their poor MuSiQue performance (Avg 32–34%)
reveals a fundamental limitation: a hallucinated answer may receive
high confidence when it is reinforced by biased or noisy evidence
under one particular query-conditioned context, but this signal is
indistinguishable from a genuinely correct answer within a single
reasoning context. These results demonstrate that cross-query con-
sistency scoring effectively utilizes multi-view information to filter
stochastic noise, providing a more reliable and balanced answer
selection mechanism.
4.6 Sensitivity Analysis of Paraphrase Count
To determine the optimal number of query rewrites and validate
that CQC-RAG’s gains are not merely an artifact of increased candi-
date count, we evaluate performance as a function of 𝑁. The results
are presented in Figure 2.
Performance remains stable with a small number of para-
phrases.On TriviaQA, the best performance is achieved at 𝑁= 2,
with 60.29% EM and 71.11% F1. Increasing 𝑁from 2 to 5 leads to only
minor fluctuations, with EM remaining around 60% and F1 around
71%. This suggests that a small set of query rewrites is already
sufficient to provide diverse reasoning contexts for cross-query
consistency verification.
Excessive paraphrases bring limited or diminishing re-
turns.On TriviaQA, performance gradually decreases when 𝑁
becomes larger, dropping to 59.47% EM at 𝑁= 7and 59.26% EM
at𝑁= 10, indicating that additional paraphrases may introduce
redundancy or lower-quality rewrites. On MuSiQue, performance
continues to improve as 𝑁increases, but the gains are relatively
small: EM improves from 33.00% at 𝑁= 4to 33.50% at 𝑁= 10,
while F1 increases from 45.82% to 46.44%. Thus, although more
paraphrases can slightly benefit multi-hop reasoning, the marginalTable 4: Comparison of the effects of different evaluator mod-
els.
MethodTriviaQA MuSiQue
EM F1 AVG. EM F1 AVG.
Qwen3-8B 58.24 70.23 64.24 32.60 43.54 38.07
Llama-3.1-8B-Instruct60.80 71.73 66.2732.80 43.90 38.35
Mistral-7B-Instruct 60.39 71.57 65.98 32.90 44.13 38.52
improvement becomes limited compared with the additional com-
putational cost.
Balancing performance, robustness, and computational cost, we
adopt𝑁= 4as the default configuration. Although 𝑁= 2yields
the highest TriviaQA score and larger values of 𝑁further improve
MuSiQue slightly, 𝑁= 4provides a strong trade-off across both
datasets. It avoids the performance degradation observed on Trivi-
aQA with excessive paraphrases while capturing sufficient query
diversity for multi-hop reasoning on MuSiQue.
4.7 Evaluator Model Analysis
As discussed in Section 3.4, we employ a separate evaluator model
𝑀evaldistinct from the reasoning model 𝑀reason to mitigate self-
enhancement bias, a well-documented phenomenon where LLMs
tend to rate their own generated outputs more favorably than those
produced by other models. This experiment validates our evaluator
design from two complementary perspectives: the critical need
for model decoupling, and the generalizability across independent
evaluator architectures.
Necessity of a separate evaluator.The results in Table 4 show
that replacing the distinct 𝑀evalwith the reasoning model itself
𝑀reason degrades the TriviaQA EM to 58.24%, which is more than 2
percentage points lower than the decoupled configurations. This
drop confirms that self-enhancement bias materially compromises
the discriminative power of cross-query evaluation. When the eval-
uator is biased towards its own reasoning paths, it becomes overly
permissive, thereby attenuating the cross-query variance signal that
CQC-RAG heavily relies upon to distinguish genuinely supported
answers from hallucinations.
Generalizability across independent evaluators.While de-
coupling is important, CQC-RAG is highly robust to the specific
choice of the independent evaluator. To verify this, we fix 𝑀reason
as Qwen3-8B and employ Mistral-7B-Instruct and Llama-3.1-8B-
Instruct as two architecturally distinct, similarly-sized evaluators.
Both independent evaluators achieve comparable and strong per-
formance, maintaining EM fluctuations within a narrow margin.
Specifically, the difference in Exact Match scores between the two
independent evaluators is as small as 0.41 percentage points on
TriviaQA and 0.10 percentage points on MuSiQue. This robustness
confirms a key property of CQC-RAG: the confidence estimation
mechanism requires only relative semantic consistency in the evalu-
ator’s cross-query logits, rather than relying on the absolute calibra-
tion of any specific model. When the evaluator remains structurally
independent from the reasoning model, the framework tends to
deliver robust performance.

Conference’17, July 2017, Washington, DC, USA Sun et al.
Table 5: Quantitative analysis of query rewriting quality,
measuring both semantic consistency and syntactic diversity.
Dataset Semantic Similarity BLEU-2 Distance Jaccard Distance
TriviaQA 0.93±0.03 0.55±0.14 0.53±0.12
MuSiQue 0.94±0.03 0.50±0.11 0.52±0.09
4.8 Rewriting Quality Analysis
A core premise of CQC-RAG is that query rewriting produces se-
mantically equivalent yet syntactically diverse queries. To verify
this property and address potential concerns regarding semantic
drift, we quantitatively evaluate both the semantic consistency and
structural diversity of our rewrites. Specifically, we randomly sam-
ple 500 queries per dataset and compute three metrics: (1)Seman-
tic Similarity, defined as the average cosine similarity between
the BGE-M3 embeddings of the original query and its rewrites; (2)
BLEU-2 Distance, defined as one minus the pairwise BLEU-2 score
among the rewrites to measure structural and bigram divergence;
and (3)Jaccard Distance, measuring the average proportion of
unshared unigrams across all pairs of rewrites. For semantic simi-
larity, values closer to 1.0 indicate strict intent preservation, while
for BLEU-2 Distance and Jaccard Distance, higher values reflect
greater surface-form diversity.
As shown in Table 5, the average semantic similarity consistently
exceeds 0.93 across both datasets. This is well above the 0.85 thresh-
old established in our system design, confirming that the applied
hard constraints effectively preserve the original user intent and
prevent semantic drift. Simultaneously, the lexical diversity metrics
indicate substantial token-level variation, with BLEU-2 Distance
and Jaccard Distance averaging between 0.50 and 0.55. This val-
idates that our soft constraints successfully introduce sufficient
syntactic diversity, which is crucial for producing the differentiated
document orderings required in the subsequent reranking stage.
4.9 Computational Overhead Analysis
To evaluate CQC-RAG’s computational cost, Table 6 compares repre-
sentative multi-path RAG methods across three sequential pipeline
stages: retrieval routing, LLMs inference, and theoretical parallel
latency.
Generative overhead is on par with standard multi-path
methods.Under our default configuration of 𝑁= 4rewritten
queries, as determined in the sensitivity analysis, CQC-RAG re-
quires 6 generative calls: one query-rewriting call and five indepen-
dent reasoning calls. First, in terms of the retrieval stage, CQC-RAG
requires only 1 retrieval, as all rewritten queries share a single re-
trieved document pool. In contrast, other multi-query methods like
DMQR-RAG require 5 distinct retrievals, and SPARC-RAG requires
4–16 retrievals due to its iterative loops. Second, for LLM inference
stage, CQC-RAG requires only 2 stages: query rewriting and paral-
lel reasoning. This sequential profile is identical to DMQR-RAG and
Speculative RAG, and significantly faster than SPARC-RAG, which
requires up to 12 sequential generation steps across its multi-agent
reasoning loop. Because the 5 parallel reasoning paths in CQC-RAG
are entirely independent, modern inference engines can executethem concurrently as a single batched inference pass, preventing
sequential latency accumulation.
Cross-query evaluation avoids decoding bottlenecks.The
main additional overhead of CQC-RAG comes from cross-query
evaluation. We define CQC-RAG’s additional evaluation-related
overhead as 𝜖=𝑇𝑟𝑒𝑟𝑎𝑛𝑘+𝑇𝑒𝑣𝑎𝑙_𝑝𝑟𝑒𝑓𝑖𝑙𝑙 , where𝑇𝑟𝑒𝑟𝑎𝑛𝑘 represents
the query-specific reranking latency over the shared pool, and
𝑇𝑒𝑣𝑎𝑙_𝑝𝑟𝑒𝑓𝑖𝑙𝑙 is the prefill-only evaluation latency. CQC-RAG uses
a compact 7B autoregressive LLM as the evaluator, but it does not
ask the model to generate full evaluation rationales. Instead, each
answer-evidence-query tuple is processed with a single forward
pass, and the binary Yes/No logits are directly used for scoring.
To run this evaluation stage efficiently, CQC-RAG groups all 5
generated tuples into a single batched prefill pass, allowing the
evaluator to score all candidate answers simultaneously without
executing separate sequential model calls. In comparison, while
Speculative RAG also utilizes a batched pass, it must process a larger
batch of 10 drafts containing much longer generated rationales. This
prefill-only design avoids sequential decoding during evaluation
and therefore reduces the overhead compared with generation-
based judging. Because both operations avoid autoregressive token
generation, their cumulative latency is computationally marginal.
Overall latency and cost-reliability trade-offs.Assuming
sufficient parallel GPU capacity, CQC-RAG’s parallel latency bot-
tleneck is effectively 𝑇𝑟𝑒𝑤𝑟𝑖𝑡𝑒+𝑇𝑟𝑒𝑎𝑠𝑜𝑛+𝜖. Since the evaluation
latency is negligible ( 𝜖≪𝑇𝑟𝑒𝑎𝑠𝑜𝑛 ), our end-to-end parallel latency
is structurally close to DMQR-RAG and avoids the sequential la-
tency accumulation of methods such as SPARC-RAG.
Although single-path RAG or unverified Self-Consistency may
have lower raw inference latency, they lack the cross-query verifica-
tion signal needed to filter noise-induced hallucinations. CQC-RAG
therefore introduces a limited evaluation overhead in exchange
for improved factual reliability, offering a practical cost-reliability
trade-off for scenarios where answer correctness is critical.
5 Conclusion
In this work, we address the reliability limitations of RAG under
single query formulation and noisy evidence. We propose the Cross-
Query Consistency Hypothesis, which posits that correct answers
grounded in genuinely relevant evidence maintain stable confidence
across semantically equivalent but syntactically diverse queries,
whereas noise-induced hallucinations exhibit unstable confidence
because their support often depends on spurious, query-conditioned
evidence patterns. Based on this hypothesis, we develop CQC-RAG,
a training-free framework that co-designs controlled query-level
diversity injection with cross-query consistency evaluation, oper-
ating entirely on a shared document pool without increasing the
retrieval budget.
CQC-RAG rewrites the original question into meaning-preserving
query variants, constructs query-conditioned reasoning contexts
through reranking, and selects answers according to confidence
stability across these contexts. This design enables answer self-
evaluation without external supervision and does not rely on ex-
panded retrieval coverage. Experiments on open-domain QA bench-
marks demonstrate that CQC-RAG improves answer reliability over
representative multi-path RAG baselines, especially in complex

CQC-RAG: Robust Retrieval-Augmented Generation via Cross-Query Consistency Conference’17, July 2017, Washington, DC, USA
Table 6: Computational complexity and parallel latency decomposition of multi-path RAG methods. The latency represents
the theoretical wall-clock time under sufficient parallel GPU capacity. 𝑇𝑟𝑒𝑎𝑠𝑜𝑛 and𝑇𝑟𝑒𝑤𝑟𝑖𝑡𝑒 denote the latencies of standard
reasoning and query rewriting, respectively. 𝑇𝑑𝑟𝑎𝑓𝑡 and𝑇𝑣𝑒𝑟𝑖𝑓𝑦 denote the drafting and verification latencies of Speculative
RAG. For SPARC-RAG, 𝑇agents_d represents the sequential agent latency at iteration depth 𝑑. For CQC-RAG, 𝜖represents the
mathematically marginal latency of query-specific reranking and prefill-only evaluation.
MethodRetrieval Stage LLM Inference Stage Parallel Latency
Retrieval Calls Sequential LLM Calls Evaluation Pass Parallel Bottleneck
Standard RAG 1 1 None𝑇 𝑟𝑒𝑎𝑠𝑜𝑛
Self-Certainty 1 1 None𝑇 𝑟𝑒𝑎𝑠𝑜𝑛
Speculative RAG 1 2 Prefill-only Scoring𝑇 𝑑𝑟𝑎𝑓𝑡+𝑇𝑣𝑒𝑟𝑖𝑓𝑦
DMQR-RAG 1–5 2 None𝑇 𝑟𝑒𝑤𝑟𝑖𝑡𝑒+𝑇𝑟𝑒𝑎𝑠𝑜𝑛
SPARC-RAG 4–16 12–48 Full Autoregressive GenerationÍ𝐷
𝑑=1(𝑇agents_d)
CQC-RAG 1 2 Prefill-only Scoring𝑇 𝑟𝑒𝑤𝑟𝑖𝑡𝑒+𝑇𝑟𝑒𝑎𝑠𝑜𝑛+𝜖
multi-hop scenarios where retrieval noise and biased evidence can
be amplified. Our work suggests a shift from finding a single opti-
mal query toward evaluating answer stability across semantically
equivalent query perspectives.
A Prompt for Parallel Rewriting
A.1 Prompt for Parallel Rewriting on TrivialQA
Original query:
In Buddhism, what is the state of blissful repose or
absolute existence by someone relieved of the necessity
of rebirth?
======================= Prompt =======================
You are an AI assistant designed to improve the retrieval
recall of a RAG system for a trivia question answering
task (TriviaQA). Your goal is to generate 4 paraphrased
versions of a given user query.
### CRITICAL RULES:
1. **Preserve Entities:** NEVER change or omit proper
nouns, dates, song titles, character names, or quoted
text (e.g., "The Miami Sound Machine", "Dec 14, 1636",
"Winter Wonderland"). These are the keys to finding the
answer.
2. **Maintain Semantics:** The paraphrased questions
must ask for EXACTLY the same answer as the original. Do
not generalize or make the question vague.
3. **Vary Structure:** - Switch between Question
formats (Who/What/Which) and Imperative formats
(Identify/Name/State). - Switch between Active and
Passive voice. - Move the focus of the sentence (e.g.,
put the date at the beginning or end).
4. **No Hallucination:** Do not add external information
that is not present in the original question.
### Few-Shot Examples:
**Example 1:** Original: "Whose backing band is known
as The Miami Sound Machine?" Paraphrases: 1. "The Miami
Sound Machine is the backing band for which famous
artist?" (Structure change) 2. "Identify the singer
cooperated with the band The Miami Sound Machine."(Imperative) 3. "Which musician uses The Miami Sound
Machine as their backing group?" (Synonym: group vs
band) 4. "Name the performer whose band goes by the name
The Miami Sound Machine." (Focus shift)
**Example 2:** Original: "According to the proverb, what
comes but once a year?" Paraphrases: 1. "What is it that
arrives only one time annually, based on the proverb?"
(Synonym: once a year -> annually) 2. "Identify the
subject of the proverb stating it comes but once a
year." (Structure change) 3. "Which event is described
in the proverb as occurring only once a year?" (Question
word change) 4. "According to the saying, what happens
just a single time per year?" (Synonym: saying vs proverb)
**Example 3:** Original: "With a motto of Always Ready,
Always There, what US military branch had it’s founding
on Dec 14, 1636?" Paraphrases: 1. "Which branch of the
US military was founded on Dec 14, 1636, and uses the
motto ’Always Ready, Always There’?" (Reordering) 2.
"Name the US military force established on Dec 14, 1636,
with the slogan ’Always Ready, Always There’." (Synonym:
branch -> force) 3. "The motto ’Always Ready, Always
There’ belongs to which US service branch founded in
1636?" (Focus on Motto) 4. "Identify the military entity
formed on Dec 14, 1636, that says ’Always Ready, Always
There’." (Imperative)
### FORMAT
Output only the 4 paraphrased questions, one per line. No
numbering, no explanations.
======================= Output =======================
Paraphrased queries:
1. What is the state of blissful repose or absolute
existence in Buddhism for someone who is no longer bound
by the cycle of rebirth?
2. Identify the Buddhist state of blissful repose or
absolute existence that is attained by someone freed from
the necessity of rebirth.
3. Which Buddhist concept refers to a state of blissful
repose or absolute existence for someone who has been
relieved of the need for rebirth?
4. In Buddhism, what is the term for the state of blissful

Conference’17, July 2017, Washington, DC, USA Sun et al.
repose or absolute existence that comes from being free
of rebirth?
A.2 Prompt for Parallel Rewriting on MuSiQue
Original query:
What month did the Tripartite discussions begin between
Britain, France, and the country where, despite being
headquartered in the nation called the nobilities
commonwealth, the top-ranking Warsaw Pact operatives
originated?
======================= Prompt =======================
You are an expert AI assistant specializing in query
rewriting for complex, multi-hop reasoning tasks. Your
goal is to generate 4 paraphrased versions of a given
user query to improve retrieval recall in a RAG system.
### DATASET CONTEXT:
The queries come from the MuSiQue dataset, which involves
multi-hop reasoning. Questions often contain nested
clauses, intermediate entities, or multiple constraints
that must be satisfied to find the answer.
### CRITICAL RULES:
1. **Preserve Entities & Proper Nouns:** NEVER change,
omit, or translate proper nouns (e.g., "La fida ninfa",
"A Don", "Marlene McFly"). These are the anchors for
retrieval.
2. **Maintain Strict Logic:** The paraphrased question
must ask for the EXACT same logical target. Do not drop
any constraint (e.g., if it says "in the winter of 2014",
do not change it to "in 2014").
3. **Unpack Nested Structures:** - If the question
involves a chain of relations, try to rephrase by changing
the word order to highlight different keywords. - You
may split long relative clauses into clearer modifier
phrases.
4. **Variety in Expression:** - Use imperative
forms (Identify/Name/Find). - Use interrogative forms
(Who/What/Which). - Invert the sentence structure.
### Few-Shot Examples:
**Example 1:** Original: Who is the director of the
film that features the song "My Heart Will Go On"?
Paraphrases: 1.Name the director of the movie in which
the song "My Heart Will Go On" appears. 2.The song "My
Heart Will Go On" was featured in a film directed by
whom? 3.Identify the filmmaker responsible for the movie
containing "My Heart Will Go On". 4.Who directed the
motion picture that includes the track "My Heart Will Go
On"?
**Example 2:** Original: What is the capital of the
country that shares a border with the birthplace of
Mozart? Paraphrases: 1.Identify the capital city of
the nation bordering Mozart’s birthplace. 2.Mozart was
born in a city; which country’s capital borders that
location? 3.Name the capital of the country adjacent to
the place where Mozart was born. 4.Which city serves
as the capital for the country bordering Mozart’s birth
city?
**Example 3:** Original: The creator of the comic
strip involving the character Snoopy served in whichbranch of the military? Paraphrases: 1.In which military
branch did the creator of the Snoopy comic strip serve?
2.Identify the branch of service of the man who created
the character Snoopy. 3.What military service did the
author of the comic featuring Snoopy perform? 4.Name the
military branch associated with the creator of Snoopy’s
comic strip.
### FORMAT
Output only the 4 paraphrased questions, one per line. No
numbering, no explanations.
======================= Output =======================
Paraphrased queries:
1. What month did the Tripartite discussions start between
Britain, France, and the country where the top-ranking
Warsaw Pact operatives came from, even though the
headquarters of the nobilities commonwealth is located
there?
2. In which month did the Tripartite discussions take
place between Britain, France, and the nation that is the
origin of the top-ranking Warsaw Pact operatives, despite
its headquarters being in the nobilities commonwealth?
3. Figure out the month when the Tripartite discussions
began between Britain, France, and the country where
the top-ranking Warsaw Pact operatives originated, even
though the headquarters of the nobilities commonwealth
is situated there.
4. Which month marked the beginning of the Tripartite
discussions between Britain, France, and the country
that is the birthplace of the top-ranking Warsaw Pact
operatives, despite the headquarters of the nobilities
commonwealth being located in that nation?
B Prompt for Multi-view Reasoning
### Task Requirements:
1. **Search First:** distinctively identify the
sentence(s) in the context that directly support the
answer.
2. **Quote Verbatim:** You must extract the supporting
text EXACTLY as it appears in the context.
3. **Answer Second:** Formulate your answer based ONLY
on the extracted evidence.
### Output Format:
Answer: [Your concise answer here]
Evidence: [Paste the exact sentence from the context here]
C Artifacts
All implementation artifacts used to produce the results reported
in this paper, including the CQC-RAG source code, preprocessed
datasets, and evaluation scripts, are publicly available at https://
github.com/FrancesPlus/CQC-RAG.
References
[1]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
InThe Twelfth International Conference on Learning Representations, ICLR 2024,
Vienna, Austria, May 7-11, 2024.
[2]Tianyu Cao, Neel Bhandari, Akhila Yerukola, Akari Asai, and Maarten Sap. 2026.
Out of Style: RAG’s Fragility to Linguistic Variation. InProceedings of the 19th

CQC-RAG: Robust Retrieval-Augmented Generation via Cross-Query Consistency Conference’17, July 2017, Washington, DC, USA
Conference of the European Chapter of the Association for Computational Lin-
guistics, EACL 2026 - Volume 1: Long Papers, Rabat, Morocco, March 24-29, 2026.
280–318.
[3]Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu.
2024. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity
Text Embeddings Through Self-Knowledge Distillation.CoRRabs/2402.03216
(2024).
[4]Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su.
2024. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large
Language Models. InAdvances in Neural Information Processing Systems 38: An-
nual Conference on Neural Information Processing Systems 2024, NeurIPS 2024,
Vancouver, BC, Canada, December 10 - 15, 2024.
[5]Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo
Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave.
2023. Atlas: Few-shot Learning with Retrieval Augmented Language Models.J.
Mach. Learn. Res.24 (2023), 251:1–251:43.
[6]Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong Park.
2024. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
Models through Question Complexity. InProceedings of the 2024 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers), NAACL 2024, Mexico City, Mexico,
June 16-21, 2024. 7036–7050.
[7]Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, De-
vendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel,
Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux,
Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix,
and William El Sayed. 2023. Mistral 7B.CoRRabs/2310.06825 (2023).
[8]Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. 2017. TriviaQA:
A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehen-
sion. InProceedings of the 55th Annual Meeting of the Association for Computational
Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers.
1601–1611.
[9]Zhewei Kang, Xuandong Zhao, and Dawn Song. 2025. Scalable Best-of-N Se-
lection for Large Language Models via Self-Certainty.CoRRabs/2502.18581
(2025).
[10] Seongyun Lee, Yongrae Jo, Minju Seo, Moontae Lee, and Minjoon Seo. 2026. Lost
in the Noise: How Reasoning Models Fail with Contextual Distractors.CoRR
abs/2601.07226 (2026).
[11] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. InAdvances in Neural Information Processing
Systems 33: Annual Conference on Neural Information Processing Systems 2020,
NeurIPS 2020, December 6-12, 2020, virtual.
[12] Zhicong Li, Jiahao Wang, Zhishu Jiang, Hangyu Mao, Zhongxia Chen, Jiazhen Du,
Yuanxing Zhang, Fuzheng Zhang, Di Zhang, and Yong Liu. 2024. DMQR-RAG:
Diverse Multi-Query Rewriting for RAG.CoRRabs/2411.13154 (2024).
[13] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. 2023. Query
Rewriting for Retrieval-Augmented Large Language Models.CoRRabs/2305.14283
(2023).
[14] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and
Hannaneh Hajishirzi. 2023. When Not to Trust Language Models: Investigating
Effectiveness of Parametric and Non-Parametric Memories. InProceedings of the
61st Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023. 9802–9822.
[15] Amir Taubenfeld, Tom Sheffer, Eran Ofek, Amir Feder, Ariel Goldstein, Zorik
Gekhman, and Gal Yona. 2025. Confidence Improves Self-Consistency in LLMs.
InFindings of the Association for Computational Linguistics, ACL 2025, Vienna,
Austria, July 27 - August 1, 2025. 20090–20111.
[16] Llama Team. 2024. The Llama 3 Herd of Models.CoRRabs/2407.21783 (2024).
[17] Hieu Tran, Zonghai Yao, Zhichao Yang, Junda Wang, Yifan Zhang, Shuo Han,
Feiyun Ouyang, and Hong Yu. 2025. RARE: Retrieval-Augmented Reasoning
Enhancement for Large Language Models. InProceedings of the 63rd Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
ACL 2025, Vienna, Austria, July 27 - August 1, 2025. 18305–18330.
[18] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop Questions via Single-hop Question Composition.Trans.
Assoc. Comput. Linguistics10 (2022), 539–554.
[19] Weiqin Wang, Yile Wang, and Hui Huang. 2025. Ranked Voting based Self-
Consistency of Large Language Models. InFindings of the Association for Compu-
tational Linguistics, ACL 2025, Vienna, Austria, July 27 - August 1, 2025. 14410–
14426.
[20] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang,
Aakanksha Chowdhery, and Denny Zhou. 2023. Self-Consistency Improves
Chain of Thought Reasoning in Language Models. InThe Eleventh International
Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023.[21] Zilong Wang, Zifeng Wang, Long T. Le, Huaixiu Steven Zheng, Swaroop Mishra,
Vincent Perot, Yuwei Zhang, Anush Mattapalli, Ankur Taly, Jingbo Shang, Chen-
Yu Lee, and Tomas Pfister. 2025. Speculative RAG: Enhancing Retrieval Aug-
mented Generation through Drafting. InThe Thirteenth International Conference
on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025.
[22] Siye Wu, Jian Xie, Jiangjie Chen, Tinghui Zhu, Kai Zhang, and Yanghua Xiao.
2024. How Easily do Irrelevant Inputs Skew the Responses of Large Language
Models?CoRRabs/2404.03302 (2024).
[23] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng
Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong
Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jian Yang, Jiaxi Yang, Jingren Zhou,
Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li,
Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao,
Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang
Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang
Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru
Zhang, Zhipeng Zhou, and Zihan Qiu. 2025. Qwen3 Technical Report.CoRR
abs/2505.09388 (2025).
[24] Xizhong Yang, Haotian Zhang, Huiming Wang, and Mofei Song. 2026. Believe
Your Model: Distribution-Guided Confidence Calibration.CoRRabs/2603.03872
(2026).
[25] Yuxin Yang, Gangda Deng, Ömer Faruk Akgül, Nima Chitsazan, Yash Govilkar,
Akasha Tigalappanavara, Shi-Xiong Zhang, Sambit Sahu, and Viktor Prasanna.
2026. SPARC-RAG: Adaptive Sequential-Parallel Scaling with Context Manage-
ment for Retrieval-Augmented Generation.CoRRabs/2602.00083 (2026).
[26] Nan Zhang, Prafulla Kumar Choubey, Alexander R. Fabbri, Gabriel Bernadett-
Shapiro, Rui Zhang, Prasenjit Mitra, Caiming Xiong, and Chien-Sheng Wu. 2025.
SiReRAG: Indexing Similar and Related Information for Multihop Reasoning. In
The Thirteenth International Conference on Learning Representations, ICLR 2025,
Singapore, April 24-28, 2025.
[27] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu,
Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-Judge with MT-
Bench and Chatbot Arena. InAdvances in Neural Information Processing Systems
36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS
2023, New Orleans, LA, USA, December 10 - 16, 2023.
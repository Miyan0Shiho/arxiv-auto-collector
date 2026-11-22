# StitchRAG: Constructing Compact and Evidence-Preserving Contexts for RAG with Diffusion-Based Stitching

**Authors**: Minsang Kim, Junpyo Lim, Heeje Cho, Seungjun Baek

**Published**: 2026-11-17 13:33:26

**PDF URL**: [https://arxiv.org/pdf/2611.22766v1](https://arxiv.org/pdf/2611.22766v1)

## Abstract
Retrieval-augmented generation (RAG) improves factuality by grounding large language models (LLMs) in external evidence. However, long retrieved contexts incur computational overhead and can even hurt performance due to noisy or irrelevant content. We propose \textsc{StitchRAG}, a framework for constructing compact context in RAG. \textsc{StitchRAG} builds a faithful, information-dense context while dynamically compressing the retrieved passages. We adapt the size reduction ratio to each passage by formulating a utility maximization trading-off the retained information and computation costs. We then extract answer candidates and query-related keywords from passages as anchors and collect passage sentences containing anchors under the derived size constraint. Finally, we connect or stitch the collected sentences together using text diffusion to form a coherent and natural context. Experiments on five open-domain QA benchmarks show that \textsc{StitchRAG} consistently outperforms diverse multi-step and compression baselines. Extensive ablations show the effectiveness of dynamic size allocation, anchor preservation, and diffusion-based stitching.

## Full Text


<!-- PDF content starts -->

StitchRAG: Constructing Compact and Evidence-Preserving Contexts for
RAG with Diffusion-Based Stitching
Minsang Kim1,2, Junpyo Lim1, Heeje Cho1, Seungjun Baek1*
1Korea University,2SK Telecom,
kmswin1@korea.ac.kr
Abstract
Retrieval-augmented generation (RAG) im-
proves factuality by grounding large language
models (LLMs) in external evidence. How-
ever, long retrieved contexts incur computa-
tional overhead and can even hurt performance
due to noisy or irrelevant content. We propose
STITCHRAG, a framework for constructing
compact context in RAG. STITCHRAG builds
a faithful, information-dense context while dy-
namically compressing the retrieved passages.
We adapt the size reduction ratio to each pas-
sage by formulating a utility maximization
trading-off the retained information and compu-
tation costs. We then extract answer candidates
and query-related keywords from passages as
anchors and collect passage sentences contain-
ing anchors under the derived size constraint.
Finally, we connect or stitch the collected sen-
tences together using text diffusion to form a
coherent and natural context. Experiments on
five open-domain QA benchmarks show that
STITCHRAG consistently outperforms diverse
multi-step and compression baselines. Exten-
sive ablations show the effectiveness of dy-
namic size allocation, anchor preservation, and
diffusion-based stitching.
1 Introduction
Large language models (LLMs) (Achiam et al.,
2023; Huang et al., 2024) have rapidly advanced
and enabled a wide range of real-world applica-
tions. However, they face fundamental limitations:
updating their knowledge is costly, and LLMs of-
ten hallucinate responses that are plausible but in-
correct (Welleck et al., 2019; Kalai et al., 2025).
Retrieval-augmented generation (RAG) addresses
these issues by grounding generation on external
evidence retrieved at inference time (Chen et al.,
2017; Lewis et al., 2020). By conditioning on re-
trieved passages, RAG improves factuality and en-
ables easy updates with up-to-date information.
*Corresonding Author
R etrievedP assagesSent ence 1Sent ence 2...Sent ence kSent ence 1Sent ence 2Sent ence 2...Sent ence kT ext DiffusionSent ence 1Mask edSent ence kSent ence 1Generat edSent ence kStitch!Figure 1:Constructing Compact Context.From re-
trieved passages, StitchRAG extracts and preserves sen-
tences containing key terms. Then it connects them via
diffusion-based stitching to construct a compact context.
A growing body of work improves RAG via
advanced retrieval and generation loops, includ-
ing multi-step reasoning (Trivedi et al., 2023; Kim
et al., 2024b), self-reflection (Jiang et al., 2023b;
Asai et al., 2023), and multi-agent debate or veri-
fication (Du et al., 2024; Hu et al., 2025), in order
to reduce hallucinations and strengthen reasoning.
However, these methods primarily focus on itera-
tive retrieval or post hoc verification, while largely
leaving the context constructed from retrieved pas-
sages unconstrained. As a result, the context of-
ten accumulates redundant or conflicting evidence.
Such longer or repeatedly processed contexts also
increase inference latency and computational cost.
More importantly, even with strong retrievers, sim-
ply providing more evidence does not reliably im-
prove answers: LLMs can be confused by noisy
passages and long inputs (Liu et al., 2024; Hong
et al., 2025; Du et al., 2025).
This has motivated substantial interest incontext
engineering(Mei et al., 2025; Hua et al., 2025)
for RAG, which helps models better interpret the
retrieved contexts. Context compression meth-
ods, such as selection (Li et al., 2023) and com-
pression prior to generation (Pan et al., 2024; Xu
et al., 2023), reduce the inference cost and la-
tency by shortening the input length. However,
most existing approaches rely on fixed or heuris-
tic compression ratios, largely independent of the
1
arXiv:2611.22766v1  [cs.CL]  17 Nov 2026

query, passage quality, or reasoning complexity.
In practice, a good compression should be context-
dependent: some questions require preserving long,
cross-passage evidence chains, whereas others ben-
efit from aggressively filtering irrelevant content.
Consequently, applying a single global compres-
sion ratio to all passages can create a mismatch
between the context reduction and the information
needs of each problem. Moreover, because prior
compression methods typically operate at the token
level rather than preserving original sentences, they
often degrade the readability of contexts and thus
the overall performance of RAG.
Contribution.We propose STITCHRAG, a frame-
work for constructing compact contexts for RAG.
STITCHRAG dynamically retains evidence in re-
trieved passages for size reduction and reconstructs
a coherent context from preserved evidence. We
formulate evidence retention as a constrained opti-
mization problem: we maximize relevance utility
combined with a computation-aware penalty to de-
rive a closed-form retention ratio for each retrieved
passage. This enables a dynamic adjustment of
the retention ratio to each passage, improving the
efficiency and relevance of the constructed context.
To build a compact yet readable context, we intro-
duce hybrid anchor selection and diffusion-based
stitching. Anchors are defined as key terms for
the query, and sentences containing anchors are
preserved in the context as fixed evidence. Next,
we performcontext stitching, which connects frag-
mented sentences using text diffusion in order to
restore coherence of the generated context. The
proposed refinement enables faithful context recon-
struction while reducing factual drift and hallucina-
tion. In our experiments, STITCHRAG consistently
improves the RAG performance over baseline meth-
ods. Extensive ablations confirm the importance
of dynamic context retention, anchor preservation,
and diffusion-based stitching.
2 Related Work
Advanced RAG and Multi-step Retrieval/Gen-
eration.Recent work has exploredmulti-step
andadaptivestrategies that couple retrieval with
intermediate reasoning steps. IRCoT (Trivedi
et al., 2023) interleaves reasoning traces with re-
trieval, enabling sequential evidence acquisition
for multi-hop questions. FLARE (Jiang et al.,
2023b) triggers retrieval based on token-level un-
certainty during generation, while Self-RAG (Asaiet al., 2023) incorporates self-reflection to assess
retrieved evidence and adapt the generation accord-
ingly. SuRe (Kim et al., 2024a) is a multi-stage
framework that generates answer candidates and
summarizes retrieved passages conditioned on each
candidate, then ranks candidates by verifying the
resulting summaries. More recently, DRAG (Hu
et al., 2025) introduces a debate-based verification
framework to reduce hallucinations and improve
answer reliability under retrieval constraints. How-
ever, these methods primarily focus on when to re-
trieve and how to expand or revise evidence through
multi-step cycles. In contrast, StitchRAG addresses
how to compress retrieved context under a dynamic
budget while preserving crucial information and
connecting fragmented evidence.
Context Compression.Prior work has studied
context compression as a means of reducing input
length before generation. Selective Context (Li
et al., 2023) prunes tokens using information-
theoretic criteria (e.g., self-information), aiming
to remove redundant content. LLMLingua and
LongLLMLingua (Jiang et al., 2023a, 2024) pro-
pose coarse-to-fine compression using a small lan-
guage model to budget and select query-relevant
spans. RECOMP (Xu et al., 2023) trains compres-
sive models to distill retrieved documents into com-
pact representations or summaries. Recent work
also studies attention-guided pruning for RAG con-
texts (Fang et al., 2025), which leverages atten-
tion signals to identify and remove redundant ev-
idence. Many prior methods frame context com-
pression as a separate pre-processing step, relying
on static heuristics or auxiliary models. In contrast,
StitchRAG incorporates budget allocation and evi-
dence retention in context construction for RAG.
Diffusion Models for Text Generation.Diffu-
sion models provide a non-autoregressive frame-
work for text generation via iterative refinement.
Early works such as Diffusion-LM (Li et al., 2022)
and DiffuSeq (Gong et al., 2022) adapted denois-
ing processes to discrete sequence modeling. Re-
cent scalable approaches (Nie et al., 2025; Ye
et al., 2025) operate on masked generative prin-
ciples and achieve performance competitive with
autoregressive LLMs at scale. Their iterative refine-
ment supports global editing with bidirectional con-
text, which is particularly suitable for controlled
text modification. Building on these properties,
StitchRAG adopts a retrieve-and-refine paradigm.
2

Quer yWhat dat e in 2010 was t he no v el published,  which is par t  
of t he series of The Maz e Runner?R etrie v ed P assagesP assage 1... The Maz e Runner series consist s of The Maz e Runner 
(2009), The S cor ch T rials (2010), and The Deat h Cur e (2011). ...P assage 2Maz e Runner :  The S cor ch T rials ( styliz ed onscr een simply as 
The S cor ch T rials )  is a 2010 A merican dyst opian science...[...]Comput e per -passage BudgetPassa g e 1R ele v ance :  0 . 7R et ention budget       :  0 . 7 9Passa g e 2R ele v ance :  0 . 4R et ention budget       :  0 . 5 3[...][...]Passa g e KR ele v ance :  0 . 9R et ention budget       :  0 . 9 32.  Hybrid Anchor Extr action &  
     Cont e xt Compr essionConstructing compact cont e xt  
under r et ention budget sS ent ence 1No anchor !Passa g e 1Model-based ancho r 
(LL M pr ediction )S ent ence 2S ent ence 3[...]+Quer y-based ancho r 
( T F-I D F)Anchor -pr ot ect ed 
sent ences3 .  Diffusion-based  Stit chingSent ences aggr egat ed fr om all passages... The Maz e Runner series consist s of The Maz e Runner 
(2009), The Scor ch T rials (2010), and The Deat h Cur e (2011)... 
...Stitch!Se q ue l 
I nf oBridging  
cluesEvidenc e 
of 2010 
se q uelDiffusion-gener at ed sent ence... dyst opian no v el writt en b y James Dashner . The no v el was 
published on Sept ember 18 ,  2010 b y Delacor t e Pr ess ...The Scor ch T rials is t he second book r eleased in t he series, ...Answ er :  Sept ember 1 8 ,  2010Figure 2:Overview of StitchRAG.Given a query and a set of retrieved passages, StitchRAG1)computes a
per-passage dynamic retention ratio r∗,2)extracts passage sentences containing key terms, calledanchors, and
aggregates them under the token budget constrained by r∗, and3)generates bridging text between the aggregated
sentences using a diffusion LLM. Finally, the constructed context is fed into the LLM to generate the answer.
3 Method
StitchRAG is a framework to construct a compact
context from retrieved passages for RAG. It first
computes how much to reduce each passage in
its size and determines its context budget; next, it
retains only sentences containing key terms under
the budget; finally, it bridges retained sentences via
text diffusion. Fig. 2. presents an overview.
3.1 Optimal Context Retention via Utility
Maximization
Problem Formulation.We consider a standard
RAG setting where a query qretrieves Kpassages
P={x 1, . . . , x K}. Consider compressing pas-
sagex∈ P where the compression rate depends
onx. For passage x, we define theretention ratio
r(x, q)∈[0,1] (for simplicity, hereafter we drop
the dependence of ronx, q). The compressed pas-
sage has length r· |x| where|x|denotes the length
of passage xin the number of tokens. For example,
ifr= 0.7 , the compressed passage has 70% of the
original length. Our goal is to find the optimal r∗
for given passage x. We formulate this problem as
cost-aware utility maximization. Specifically, we
maximize the objectiveF(r)forr∈[0,1]:
F(r) =U(r)− C(r)(1)
where Udenotes the achievable utility and Cde-
notes the computation cost.
Relevance utility.The utility is defined to measure
how the compressed passage is relevant to query q.However, in the first place, the original passage x
should be relevant to qprior to compression. Sup-
pose we obtain utility u(r) with retention ratio rif
xis relevant, and utility 0 if xis irrelevant. Let ran-
dom variable Z∈ {0,1} indicate relevance ( Z=1
iffxis relevant). The achievable utility is given by
Z·u(r), andU(r)is defined as its expectation:
U(r) =E[Z]·u(r) =P(Z=1)·u(r).(2)
We estimate P(Z=1) by probing the LLM with a
binary prompt that asks whether passage xis useful
to answer q(see Appendix A.3 for the template).
We compute the log-probabilities of the LLM of
the answers (“Yes” vs. “No”), denoted by LYesand
LNo, and convert them into a relevance score:
P(Z=1)≈R(x, q) :=exp(L Yes)
exp(L Yes) + exp(L No).
Next, we define u(r). Since u(r) is obtained
assuming xis a relevant passage, u(r) should be
increasing in r; retaining more from xis more
informative to answer q. However, we assume
adiminishing return: new information does not
increase proportionately with the context length.
Thus, we adopt alogarithmic utilityinr:
u(r) = log(1 +r)(3)
A diminishing return is consistent with Heaps’
law (Heaps, 1978), which states that the number
of distinct words grows sub-linearly in the docu-
ment length. Thus, redundancy grows faster than
3

novel information as the context gets longer. Later,
we give more justification for log-utility u(r). In
summary,U(r) =R(x, q) log(1 +r).
Computation Cost.The latency of self-attention
scales quadratically with the sequence length. Thus,
the computation cost of retaining r|x| tokens is
proportional to (r|x|)2. In RAG, documents are
commonly segmented into passages of fixed word
count (Karpukhin et al., 2020) or fixed token
length (Bhat et al., 2025; Zhong et al., 2025) for in-
dexing. Thus, we treat |x|as a constant and define
the cost objective:
C(r) =βr2,(4)
where the hyperparameter β >0 controls the la-
tency penalty and absorbs the constant factor|x|2.
Deriving the optimal r∗.We maximize expected
utility minus cost:
max
r∈[0,1]F(r) =R(x, q)·log(1 +r)−βr2.(5)
F(r) is a strictly concave function in r, and (5)
admits a unique solution (Boyd and Vandenberghe,
2004) as follows:
r∗= min(
1
2 s
1 +2R(x, q)
β−1!
,1)
.(6)
We set hyperparameter β= 0.25 . In this setting,
one can show that r∗= 1if and only if R(x, q) =
1, which is a reasonable choice. In (6),r∗increases
with relevance confidence R, i.e., highly relevant
passages should retain more context.
3.2 Hybrid Anchor Extraction
For passage x, our token budget is set to r∗|x|.
We select and retain importantsentencesfrom x
under this budget. Retaining sentences, instead
of individual tokens, helps to maintain the coher-
ence of context after compression. To that end,
we select sentences which contain important spans,
calledanchors. We first select the anchors by a
hybrid strategy: a semantic-lexical selection that
combines reasoning-centric generation with query-
centric lexical matching.
Reasoning-centric anchors (AR-based).We gen-
erate multiple answer candidates using the LLM
conditioned on (q, x) . The answer spans form the
anchor set, AAR, to capture implicit semantic de-
pendencies beyond simple keyword matching.
Query-centric anchors (TF-IDF-based).In paral-
lel, we extract the query-centric anchor set denotedbyATFIDFusing a TF-IDF score to ensure the cov-
erage of rare entities and lexical cues. The final
anchor set is the union of the anchor sets:
A=AAR∪ATFIDF.(7)
3.3 Fact-First Sentence Selection
Given the token budget Tbudget =r∗· |x|, we se-
lect sentences from xthat contain anchors. We
initially mark all the sentences in xas “unselected”.
1. Anchor protection.We identify sentences that
contain anchors in Aand mark them as “selected”.
LetSAdenote this set of sentences, which forms
the factual backbone of the compressed context.2.
Budget-aware selection.If the total number of
tokens in SAis less than Tbudget , we select some
non-anchor sentences using a lightweight priority
rule: (i) sentences containing query keywords, and
(ii) earlier sentences in retrieved passages, which
often summarize salient local context. These high-
priority non-anchor sentences are marked as “se-
lected” until the token budget is reached. Finally,
all the sentences marked as “unselected” are re-
moved fromxto obtain the compressed passage.
Passage Compression and Log-utility.We pro-
vide a heuristic argument relating log-utility u(r)
to our passage compression. Suppose we randomly
choose a sentence from passage x. The probability
of choosing an anchor sentence is p=|SA|
|S|. If we
take the logarithm of1/p, we have that
log1
p= log|S|
|SA|(8)
(8)is analogous to the well-knownInverse Docu-
ment Frequency (IDF); consider Sas a collection
of documents (treating sentences as documents),
andSAas the document set containing a keyword,
then (8)is the IDF. The IDF serves as a relevance
metric for retrieval (Robertson, 2004). Suppose x
is compressed by our sentence selection method
with retention ratio r. The compressed passage
retains SA, but its size is reduced to approximately
r|S|. Thus, (8) for the compressed passage is
logr|S|
|SA|= logr+ log|S|
|SA|(9)
(9)shows the logarithmic dependence of relevance
metric on r(the second term is constant and does
not affect optimization (5)), where we used u(r) =
log(1 +r) for numerical stability. Overall, u(r)
can be approximately seen as a relevance metric
associating the compressed passage with anchor
terms.
4

3.4 Diffusion-based Context Stitching
All the retrieved passages are compressed and then
concatenated to construct the final context denoted
byX. While Xpreserves crucial factual informa-
tion, concatenated sentences may not form a coher-
ent context as a whole, because (i) some sentences
are removed within the passage during compres-
sion; (ii) possibly heterogeneous passages are di-
rectly concatenated. We proposecontext-stitching
based on text diffusion. The goal of stitching is to
simply connect the sentences to maintain their nat-
ural semantic flows using a small amount of tokens.
The stitching consists of two steps:
1. Dynamically allocatestitching tokens, initial-
ized as masked tokens, and insert them be-
tween adjacent sentences inX.
2.Fill these stitching tokens in parallel via a
diffusion-based denoising process.
Allocation based on Semantic overlap.We allo-
cateκi,jstitching tokens between sentences iand
jbased on their semantic overlap. Let Ji,jdenote
the Jaccard similarity of sentencesiandj, then
κi,j=
¯κ· 
1− J i,j
(10)
where ¯κis the maximum budget for each stitching.
¯κmay differ depending on the types of i, j: ifi
andjbelong to different passages, i.e., they are at
passage boundaries and are likely to have a large
semantic distance, ¯κis set relatively large. In con-
trast, if iandjbelong to the same passage, ¯κis set
relatively small; see Sec. 4 for details.
Diffusion with a verbatim constraint.We per-
formtext diffusion(Li et al., 2022; Lou et al., 2023)
only on stitching tokens in X, while keeping the
original context tokens frozen. We treat the frozen
tokens as observed conditioning context and per-
form masked denoising only on the stitching posi-
tions (i.e., text infilling (Zhu et al., 2019)). Let X(t)
denote the partially denoised version of X(T)≡ X
at reverse step t=T, . . . ,0 . Letpθ(X(t−1)|X(t))
denote the reverse denoising distribution parame-
terized by θ. LetIdenote the index set of stitching
tokens in X. We first sample an intermediate state
˜X(t−1)∼pθ(X(t−1)|X(t)),(11)
and update the context state at indexiorX i:
X(t−1)
i =(˜X(t−1)
i, i∈ I,
X(T)
i, i̸∈ I,(12)Our diffusion with theverbatim constraint, i.e.,
denoising only stitching tokens and keeping the
original context fixed, prevents factual drift while
allowing coherent transitions between fragmented
passages. We use pre-trained denoising models,
e.g., (Nie et al., 2025; Ye et al., 2025), asp θ.
Why Diffusion?Text diffusion is advantageous
for context stitching for the following reasons. 1)
Auto-regressive models generate tokens only from
left context, i.e., P(x t|x<t), whereas stitching
is aninterpolationproblem capturing both preced-
ing and succeeding contexts. We therefore adopt
text diffusion, whose bidirectional attention works
conditionally on both sides of the context. 2) Text
diffusion enablesparallel decodingover masked
tokens, which can reduce latency when the diffu-
sion step count is much smaller than the context
length.
4 Experiment
4.1 Experimental Settings
We evaluate performance on five open-domain QA
benchmarks. Following prior work (Jiang et al.,
2023b; Hu et al., 2025), we randomly sample 500
examples from each dataset using the default setup
provided by FlashRAG (Jin et al., 2025).
Dataset.We evaluate retrieval-augmented gen-
eration on HotpotQA (Yang et al., 2018), Natu-
ral Questions (NQ) (Kwiatkowski et al., 2019),
TriviaQA (Joshi et al., 2017), 2WikiMultiHopQA
(2Wiki) (Ho et al., 2020), and PopQA (Mallen et al.,
2023). HotpotQA and 2Wiki require multi-hop rea-
soning across multiple passages. NQ consists of
real user queries paired with Wikipedia answers.
TriviaQA focuses on factoid questions with evi-
dence from web or Wikipedia sources. PopQA
is an entity-centric open-domain QA benchmark
constructed from Wikidata and designed to probe
long-tail factual knowledge. For each query, we
retrieve a fixed number of passages and condition
the generator on the retrieved context.
Baselines.We compare STITCHRAG against (i)
non-retrievalbaselines, including NAIVEGENand
MAD (Du et al., 2024), (ii) Standard RAG (Lewis
et al., 2020) (iii)iterative / multi-stepRAG methods
that involve multiple stages of retrieval and/or gen-
eration (SURE(Kim et al., 2024a), FLARE (Jiang
et al., 2023b), IRCOT (Trivedi et al., 2023), SELF-
RAG (Asai et al., 2023), DRAG (Hu et al.,
2025)), and (iv)context compressionmethods
(LONGLLMLINGUA(Jiang et al., 2024), RE-
5

Table 1:Comparison of retrieval-augmented generation methods. ( ✓: Retrieval used, ✗: No retrieval). Bold
indicates the best performance. and Underline denotes the second-best performance.
Ret. MethodHotpotQA NQ TriviaQA PopQA 2Wiki
EM F1 EM F1 EM F1 EM F1 EM F1
Baselines
✗Naive Gen 16.8 23.8 17.4 26.2 56.6 65.5 19.2 23.3 8.6 16.5
✗MAD 23.0 32.7 21.8 33.1 56.4 66.3 21.4 28.2 18.2 25.1
✓RAG 30.2 41.1 37.6 50.4 62.0 71.0 37.0 45.6 19.0 28.2
Compression Methods
✓LongLLMLingua 29.4 39.5 33.4 45.4 60.2 69.3 34.0 41.5 16.8 27.0
✓RECOMP 29.2 39.2 35.2 46.0 58.3 65.8 35.8 43.0 21.4 33.5
✓Selective Context 22.1 36.4 32.5 42.6 57.6 65.9 29.5 36.8 13.6 19.5
Multi-step Methods
✓SuRe 19.0 32.3 34.8 51.3 47.6 64.1 41.848.910.2 18.6
✓FLARE 16.6 23.7 19.4 27.6 53.6 63.0 21.6 24.3 9.2 20.1
✓IRCoT 25.2 34.4 28.6 37.3 47.2 54.5 27.0 33.0 22.8 31.1
✓Self-RAG 14.8 28.844.051.2 46.4 58.3 22.0 34.3 13.0 26.6
✓DRAG 30.8 41.7 36.8 50.3 60.8 69.938.646.128.834.9
Ours
✓StitchRAG 32.6 43.439.4 53.3 63.4 72.837.2 46.3 27.8 35.4
COMP (Xu et al., 2023), SELECTIVECON-
TEXT(Li et al., 2023)).
Implementation details.The system is built on
the FLASHRAG toolkit (Jin et al., 2025). Follow-
ing this setup, we use E5-base-v2 (Wang et al.,
2022) as the retriever and the widely adopted De-
cember 2018 Wikipedia dump as the retrieval cor-
pus (Karpukhin et al., 2020). For fair compari-
son, all methods use the same retriever, and we
report Exact Match (EM) and token-level F1 using
standard evaluation protocols. Forsingle-shotre-
trieval methods (including ours and non-iterative
baselines), we retrieve Top- K=9 passages once
and condition the generator on the retrieved con-
text. For baselines thatexplicitly iterate retrieval
(e.g., FLARE, IRCOT, DRAG), we follow their
original settings and retrieve Top- K=3 passages
per iteration for up to three retrieval–generation
iterations (up to 9 passages total). The generator
is LLAMA3-8B-INSTRUCT(Huang et al., 2024).
We instantiate our approach with LLADA (Nie
et al., 2025) as the diffusion backbone, using
T= 32 reverse diffusion steps. We set ¯κto 2 for
intra-passage sentences and to 7 for inter-passage
sentences. For the diffusion backbone, we build
on Fast-dLLM (Wu et al., 2025) and optimize its
dynamic remasking forbatched, GPU-parallelex-
ecution, which is crucial in context-stitching wheremultiple retrieved passages are processed simul-
taneously. Specifically, we remove Python-level
per-sample loops via vectorized sorting, prefix-
product cutoff detection, and batched scatter-based
selection, preserving the same remasking policy
and matching transfer decisions while achieving
an average ∼1.3×speedup for batch size ≥4(Ap-
pendix A.5). All experiments were conducted on
A100 GPUs.
Context budget for baselines.StitchRAG reduces
the retrieved context to 72.29% on average of its
original length across datasets. To match the re-
tention budget, we set the retention ratio of other
compression-based baselines and fixed-retention
budget methods to 75% in all experiments. Table 9
in Appendix A.4 provides detailed statistics.
4.2 Main Results
Table 1 summarizes end-to-end QA performance
(EM/F1) on five open-domain benchmarks. Over-
all, StitchRAG achieves the strongest results, rank-
ing first on 6 of 10 metrics and second on the re-
maining three, demonstrating robust performance
across both single-hop and multi-hop QA. On Hot-
potQA, it improves over RAG (Lewis et al., 2020)
by +2.4 EM and +2.3 F1, and also outperforms
DRAG (Hu et al., 2025) by +1.8 EM and +1.7 F1.
On NQ, although Self-RAG (Asai et al., 2023) at-
6

Top-10 Top-20 Top-30
Retrieval depth (Top-K)40455055Avg F1 (4 datasets)51.2 51.1 51.1
48.150.6
45.1StitchRAG (LLaDA)
Naive RAGFigure 3:Robustness to increased retrieval passages.
Average F1 across four QA benchmarks (Hotpot, NQ,
TriviaQA, 2Wiki) as the retrieval number of Top- Kin-
creases (Top-10/20/30). While standard RAG degrades
sharply at larger Kdue to longer and noisier contexts,
StitchRAG remains stable, mitigating retrieval noise and
context-length effects.
tains the highest EM, StitchRAG achieves the best
F1, improving over RAG by +2.1 F1. On TriviaQA,
it sets the best EM and F1, surpassing RAG by
+1.4 EM and +1.8 F1. On PopQA, DRAG achieves
the highest EM, while SuRE attains the best F1.
StitchRAG ranks second on both metrics. On
2Wiki, StitchRAG achieves the best F1 while rank-
ing second in EM. Across datasets, both StitchRAG
variants consistently outperform the retrieval base-
lines and most multi-step and compression-based
methods. This shows that our dynamic context
construction and diffusion-based stitching are an
effective alternative to multi-round retrieval and
generic context compression.
4.3 Analyses
Robustness to Retrieval Noise.Fig. 3 evaluates ro-
bustness under increasing retrieval depth by sweep-
ing Top- Kfrom 10 to 30. As Kgrows, the re-
trieved context becomes longer and typically nois-
ier, substantially hurting RAG pipelines: Standard
RAG drops sharply at Top- 30, falling to 45.1 av-
erage F1 (from 48.1 at Top- 10), indicating a pro-
nounced sensitivity to retrieval noise and context-
length effects. In contrast, StitchRAG remains es-
sentially invariant across the sweep, staying at 51.2
average F1 for Top- 10/20/30 . This stability sug-
gests that our dynamic context construction effec-
tively decouples generation quality from the raw
amount of retrieved evidence by filtering redundant
content and preserving an information-dense con-
text, thereby mitigating the degradation observed
in naive scaling of retrieval depth.
Effectiveness of Dynamic Retention Budgets.Ta-
ble 2 compares a fixed retention ratio (75%) againstTable 2:Effect of adaptive vs. fixed retention ratio
(75%) on StitchRAG (EM/F1).
Compression HotpotQA NQ TriviaQA 2Wiki
EM F1 EM F1 EM F1 EM F1
Fixed 75% 30.8 41.9 37.4 52.4 61.6 72.2 26.8 33.3
Dynamic (Avg. 72.7%)32.6 43.4 39.4 53.3 63.4 72.8 27.8 35.4
Table 3:Ablation on hybrid anchor extraction
(EM/F1).We ablate reasoning-centric AR anchors and
lexical TF-IDF anchors.
Anchors HotpotQA NQ TriviaQA 2Wiki
EM F1 EM F1 EM F1 EM F1
AR✓TF-IDF×29.8 41.4 39.2 52.1 61.0 70.7 25.8 33.4
AR×TF-IDF✓31.8 42.4 37.2 50.9 63.2 72.4 26.0 33.9
AR✓TF-IDF✓32.6 43.4 39.4 53.3 63.4 72.8 27.8 35.4
our dynamic, utility-guided budgeting (average re-
tention ≈72.7%). Across all four benchmarks, dy-
namic budgeting consistently improves both EM
and F1, indicating that a fixed retention rate is sub-
optimal under heterogeneous retrieval quality and
question difficulty. The gains are most pronounced
on multi-hop datasets: on HotpotQA, dynamic bud-
geting improves +1.8 EM and +1.5 F1, and on
2Wiki, +1.0 EM and +2.1 F1. We also observe con-
sistent improvements on single-hop settings: on
NQ, F1 increases by +0.9, and on TriviaQA, +1.8
EM and +0.6 F1. These results suggest that allo-
cating a slightly larger budget to high-utility pas-
sages while aggressively filtering low-utility ones
yields a higher information density under a com-
parable overall compression level, outperforming
fixed-budget compression.
Ablation Study of Anchor Protection.Table 3
studies the contribution of our hybrid anchor ex-
traction by ablating reasoning-centric AR anchors
and query-centric lexical TF-IDF anchors. Using
either anchor set alone is consistently worse than
combining both, indicating strong complementarity.
With AR-ONLY(TF-IDF removed), performance
drops markedly on multi-hop datasets (HotpotQA:
29.8/41.4; 2Wiki: 25.8/33.4), suggesting that AR
candidates alone may miss rare entities or exact
lexical cues needed for precise evidence ground-
ing. With TF-IDF-ONLY(AR removed), results
improve over AR-only on HotpotQA and TriviaQA
(31.8/42.4 and 63.2/72.4, respectively) but degrade
on NQ (37.2/50.9), where paraphrased answers ben-
efit from semantically guided anchors. Combining
AR+TF-IDF yields the best performance across all
datasets, demonstrating that semantic anchors from
AR and lexical coverage from TF-IDF have syner-
7

Table 4:Ablation on stitching components (EM/F1).
“No stitching” removes diffusion stitching and uses only
extracted evidence spans; “AR-stitching” replaces diffu-
sion stitching with an autoregressive stitcher (LLAMA).
Variant Stitcher HotpotQA NQ TriviaQA 2Wiki
EM F1 EM F1 EM F1 EM F1
No stitching – 32.2 42.7 38.0 50.9 61.4 71.1 26.0 33.4
AR-stitching LLAMA 27.4 38.3 37.6 50.5 56.0 65.5 24.0 33.2
StitchRAG Diffusion32.6 43.4 39.4 53.3 63.4 72.8 27.8 35.4
Table 5:Sensitivity to the number of Anchors.Left:
AR candidate sweep with TF-IDF fixed to 10. Right:
TF-IDF anchor sweep with AR fixed to 1. Values are
Avg F1 on HotpotQA.
Setting Hyperparameter Avg F1
1 43.4
AR candidates sweep 2 43.3
5 42.4
5 42.4
TF-IDF anchors sweep 10 43.4
20 43.5
gistic effects, yielding faithful and information-rich
sentences for subsequent stitching.
Impact of Stitching Strategies.Table 4 shows the
effect of context stitching. We compare the modifi-
cations of StitchRAG: (i)NO STITCHINGperforms
only dynamic context construction without stitch-
ing; (ii) ARSTITCHINGperforms stitching with
autoregressive models instead of diffusion. Table 4
shows that full StitchRAG performs the best on
all benchmarks (e.g., HotpotQA +0.4 EM / +0.7
F1; NQ +1.4 EM / +2.4 F1; 2Wiki +1.8 EM / +2.0
F1) overNO STITCHING, showing that stitching
contributes non-trivial gains by producing coherent
transitions and reducing discontinuities between
retained spans. In contrast, AR-stitching underper-
forms drastically, suggesting that left-to-right gen-
eration is brittle for constrained context editing: it
tends to drift from frozen evidence and accumulates
errors when forced to connect disjoint fragments.
Our results show that diffusion-based stitching is
better suited for faithful context construction.
Effects of the number of Answer Candidates.
From Table 5, we observe that the number of AR
candidates has limited benefit beyond a single can-
didate. Using one AR candidate achieves a strong
performance (43.4), and expanding the candidate
set to 2 brings little improvement, while further
increasing it to 5 substantially hurts performance.
This trend suggests that additional AR candidates
may introduce noisy or weakly relevant anchor sig-
nals. By contrast, performance consistently im-Table 6:Latency comparison.We randomly sample
100 examples from HotpotQA. All experiments are con-
ducted with FlashRAG (Jin et al., 2025) on A100 GPUs,
and results are reported as mean±standard deviation.
RAG Self-RAG IRCoT DRAG StitchRAG
Latency
(seconds)7.24
±2.848.08
±0.5624.66
±9.8127.39
±7.9710.62
±4.30
proves as the number of TF-IDF anchors increases
from 5 to 10, and then saturates at 20. The small
gain from 10 to 20 implies that a moderate number
of TF-IDF anchors is sufficient to capture useful
lexical evidence. Based on this trade-off, we use 1
AR candidate and 10 TF-IDF anchors.
Latency Analysis.Table 6 shows that StitchRAG
strikes a practical balance between efficiency and
performance. Although it is slower than Base RAG
and Self-RAG, it remains substantially more effi-
cient than IRCoT and DRAG, both of which require
iterative multi-step reasoning. When we replaced
diffusion-based stitching in StitchRAG with autore-
gressive stitching, latency increased by about 10.3
s. Thus, diffusion has advantages not only in perfor-
mance but also in efficiency compared to AR due to
its parallel decoding. Overall, our result positions
StitchRAG as a middle-ground design point that
improves over simple one-pass baselines without
incurring the full cost of iterative methods.
Additional Results.We provide additional exper-
iments on the use of another diffusion backbone,
DREAM (Ye et al., 2025), and an analysis of stitch-
ing tokens in Appendix A.1. A qualitative analysis
is provided in Appendix A.2.
5 Conclusion
In this work, we introduced STITCHRAG, which
addresses a central challenge in RAG: how to
build a compact context without discarding the
evidence most critical for answering. By combin-
ing passage-adaptive context-length allocation, hy-
brid anchor selection, and diffusion-based stitching,
STITCHRAG constructs compact yet information-
dense contexts without incurring a substantial
efficiency cost. Across five open-domain QA
benchmarks, STITCHRAG consistently outper-
forms prior multi-step and compression baselines
while also offering a clear efficiency advantage.
We hope this work motivates further research on
adaptive and faithful context construction for long
and complex retrieval settings.
8

6 Limitations
StitchRAG improves answer quality, reduces con-
text length, and provides a theoretically grounded
framework to determine the retention ratio. How-
ever, our approach still requires multiple model
inferences similar to other advanced RAG methods.
This incurs overhead as compared to the default
baseline: single-pass RAG.
To mitigate this overhead, we design the pipeline
to be batch-friendly. The extra calls can be amor-
tized via auto-regressive batch inference for anchor
scoring/selection across many queries. We also use
parallelization strategies for stitching, such as dif-
fusion parallel decoding (multi-sample denoising
executed in parallel, enhanced by our modifica-
tion A.5). In practice, these batching and paral-
lel decoding techniques substantially reduce per-
request overhead.
Finally, while our framework prevents factual
drift by freezing verbatim anchors during diffusion,
we restricted the diffusion method to lightweight
stitching. This is mainly due to the limited capabil-
ity of state-of-the-art discrete diffusion LLMs as of
this writing. As diffusion-based generators mature,
we envision an exciting direction beyond stitch-
ing: towards more aggressive yet faithful context
refinement, elegantly rewriting and reorganizing
non-anchor regions while keeping anchors intact,
without increasing inference latency.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, et al. 2023. Gpt-4 technical report.
arXiv preprint arXiv:2303.08774.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations.
Sinchana Ramakanth Bhat, Max Rudat, Jannis Spiek-
ermann, and Nicolas Flores-Herr. 2025. Rethink-
ing chunk size for long-document retrieval: A multi-
dataset analysis.arXiv preprint arXiv:2505.21700.
Stephen Boyd and Lieven Vandenberghe. 2004.Convex
optimization. Cambridge university press.
Danqi Chen, Adam Fisch, Jason Weston, and Antoine
Bordes. 2017. Reading wikipedia to answer open-
domain questions. InProceedings of the 55th Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 1870–1879.Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenen-
baum, and Igor Mordatch. 2024. Improving factual-
ity and reasoning in language models through multia-
gent debate. InForty-first international conference
on machine learning.
Yufeng Du, Minyang Tian, Srikanth Ronanki, Subendhu
Rongali, Sravan Bodapati, Aram Galstyan, Azton
Wells, Roy Schwartz, Eliu A Huerta, and Hao
Peng. 2025. Context length alone hurts llm per-
formance despite perfect retrieval.arXiv preprint
arXiv:2510.05381.
Yixiong Fang, Tianran Sun, Yuling Shi, and Xiaodong
Gu. 2025. Attentionrag: Attention-guided context
pruning in retrieval-augmented generation.arXiv
preprint arXiv:2503.10720.
Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu,
and LingPeng Kong. 2022. Diffuseq: Sequence to se-
quence text generation with diffusion models.arXiv
preprint arXiv:2210.08933.
Harold Stanley Heaps. 1978.Information retrieval:
Computational and theoretical aspects. Academic
Press, Inc.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning
steps. InProceedings of the 28th International Con-
ference on Computational Linguistics, pages 6609–
6625.
Kelly Hong, Anton Troynikov, and Jeff Huber. 2025.
Context rot: How increasing input tokens impacts
llm performance. Technical report, Chroma.
Wentao Hu, Wengyu Zhang, Yiyang Jiang, Chen Jason
Zhang, Xiaoyong Wei, and Li Qing. 2025. Removal
of hallucination on hallucination: Debate-augmented
rag. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 15839–15853.
Qishuo Hua, Lyumanshan Ye, Dayuan Fu, Yang Xiao,
Xiaojie Cai, Yunze Wu, Jifan Lin, Junfei Wang,
and Pengfei Liu. 2025. Context engineering 2.0:
The context of context engineering.arXiv preprint
arXiv:2510.26493.
Kunal Chawla Huang, Kushal Lakhotia, Kyle Huang,
Lailin Chen, Lakshya Garg, A Lavender, Leandro
Silva, Lee Bell, Lei Zhang, Liangpeng Guo, et al.
2024. The llama 3 herd of models.preprint.
Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing
Yang, and Lili Qiu. 2023a. Llmlingua: Compressing
prompts for accelerated inference of large language
models. InProceedings of the 2023 conference on
empirical methods in natural language processing,
pages 13358–13376.
Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng
Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2024.
Longllmlingua: Accelerating and enhancing llms in
9

long context scenarios via prompt compression. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 1658–1677.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023b. Active retrieval
augmented generation. InProceedings of the 2023
conference on empirical methods in natural language
processing, pages 7969–7992.
Jiajie Jin, Yutao Zhu, Zhicheng Dou, Guanting Dong,
Xinyu Yang, Chenghao Zhang, Tong Zhao, Zhao
Yang, and Ji-Rong Wen. 2025. Flashrag: A modular
toolkit for efficient retrieval-augmented generation
research. InCompanion Proceedings of the ACM on
Web Conference 2025, pages 737–740.
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke
Zettlemoyer. 2017. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. InProceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1601–1611.
Adam Tauman Kalai, Ofir Nachum, Santosh S Vem-
pala, and Edwin Zhang. 2025. Why language models
hallucinate.arXiv preprint arXiv:2509.04664.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 conference on empirical methods in natural
language processing (EMNLP), pages 6769–6781.
Jaehyung Kim, Jaehyun Nam, Sangwoo Mo, Jongjin
Park, Sang Woo Lee, Minjoon Seo, Jung Woo Ha,
and Jinwoo Shin. 2024a. Sure: Summarizing re-
trievals using answer candidates for open-domain qa
of llms. In12th International Conference on Learn-
ing Representations, ICLR 2024.
Minsang Kim, Cheoneum Park, and Seung Jun Baek.
2024b. Qpaug: Question and passage augmentation
for open-domain question answering of llms. InFind-
ings of the Association for Computational Linguistics:
EMNLP 2024, pages 9024–9042.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, et al. 2019. Natural questions: a benchmark
for question answering research.Transactions of the
Association for Computational Linguistics, 7:453–
466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks.Advances in neu-
ral information processing systems, 33:9459–9474.Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S
Liang, and Tatsunori B Hashimoto. 2022. Diffusion-
lm improves controllable text generation.Advances
in neural information processing systems, 35:4328–
4343.
Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin.
2023. Compressing context to enhance inference ef-
ficiency of large language models. InProceedings of
the 2023 conference on empirical methods in natural
language processing, pages 6342–6353.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts.Transactions of the Association
for Computational Linguistics, 12:157–173.
Aaron Lou, Chenlin Meng, and Stefano Ermon.
2023. Discrete diffusion modeling by estimating
the ratios of the data distribution.arXiv preprint
arXiv:2310.16834.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. InProceedings of the 61st annual meeting of
the association for computational linguistics (volume
1: Long papers), pages 9802–9822.
Lingrui Mei, Jiayu Yao, Yuyao Ge, Yiwei Wang, Bao-
long Bi, Yujun Cai, Jiazhi Liu, Mingyu Li, Zhong-Zhi
Li, Duzhen Zhang, et al. 2025. A survey of con-
text engineering for large language models.arXiv
preprint arXiv:2507.13334.
Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang,
Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong
Wen, and Chongxuan Li. 2025. Large language dif-
fusion models.arXiv preprint arXiv:2502.09992.
Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia,
Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle,
Yuqing Yang, Chin-Yew Lin, et al. 2024. Llmlingua-
2: Data distillation for efficient and faithful task-
agnostic prompt compression. InFindings of the As-
sociation for Computational Linguistics: ACL 2024,
pages 963–981.
Stephen Robertson. 2004. Understanding inverse doc-
ument frequency: on theoretical arguments for idf.
Journal of documentation, 60(5):503–520.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st annual meeting of the association for com-
putational linguistics (volume 1: long papers), pages
10014–10037.
Liang Wang, Nan Yang, Xiaolong Huang, Binxing
Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder,
and Furu Wei. 2022. Text embeddings by weakly-
supervised contrastive pre-training.arXiv preprint
arXiv:2212.03533.
10

Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Di-
nan, Kyunghyun Cho, and Jason Weston. 2019. Neu-
ral text generation with unlikelihood training.arXiv
preprint arXiv:1908.04319.
Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu,
Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and
Enze Xie. 2025. Fast-dllm: Training-free accelera-
tion of diffusion llm by enabling kv cache and parallel
decoding.arXiv preprint arXiv:2505.22618.
Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2023. Re-
comp: Improving retrieval-augmented lms with com-
pression and selective augmentation.arXiv preprint
arXiv:2310.04408.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 conference on empiri-
cal methods in natural language processing, pages
2369–2380.
Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui
Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong.
2025. Dream 7b: Diffusion large language models.
arXiv preprint arXiv:2508.15487.
Zijie Zhong, Hanwen Liu, Xiaoya Cui, Xiaofan Zhang,
and Zengchang Qin. 2025. Mix-of-granularity:
Optimize the chunking granularity for retrieval-
augmented generation. InProceedings of the 31st
International Conference on Computational Linguis-
tics, pages 5756–5774.
Wanrong Zhu, Zhiting Hu, and Eric Xing. 2019. Text
infilling.arXiv preprint arXiv:1901.00158.
11

Setting HotpotQA NQ TriviaQA 2Wiki
EM F1 EM F1 EM F1 EM F1
Intra-only 32.0 42.7 38.6 51.5 61.0 70.7 26.0 33.7
Inter-only 32.4 43.1 39.2 53.1 61.2 70.6 26.6 34.5
Both32.6 43.4 39.4 53.3 63.4 72.8 27.8 35.4
Table 7:Ablation on stitching scope (EM/F1).“Intra-
only” uses stitching tokens only within the same passage;
“Inter-only” stitches only across passages; “Both” enables
both intra- and inter-passage stitching.
2 3 4 5 6 7 8 9
Stitching token budget 
31.231.431.631.832.032.232.4HotpotQA EM
intra best: 32.4 @ 2
inter best: 31.8 @ 7Intra-only
Inter-only
Figure 4:Effect of stitching-token budget on intra-
vs. inter-passage stitching.
Table 8:Results of StitchRAG with different diffusion LLMs. Boldand Underline respectively denote the best
performance and the second-best performance in the Table 1 (main results).
Ret. MethodHotpotQA NQ TriviaQA 2Wiki
EM F1 EM F1 EM F1 EM F1
✓StitchRAG (Dream)32.0 42.2 38.6 52.6 62.7 71.9 27.2 34.7
✓StitchRAG (LLaDA) 32.6 43.439.4 53.3 63.4 72.827.8 35.4
A Appendix
A.1 Additional Experiments
Analysis of Stitching token budgets.Figure 4 sweeps the stitching-token budget ¯κon HotpotQA
(EM) while isolating intra-passage vs. inter-passage stitching. We find that the optimal budget differs
substantially by scope:Intra-onlypeaks at a small budget ( ¯κ=2, 32.4 EM), suggesting that within-passage
gaps primarily require short local smoothing. In contrast,Inter-onlybenefits from a larger budget and
peaks later ( ¯κ=7, 31.8 EM), consistent with the higher difficulty of bridging cross-passage evidence.
Performance degrades when ¯κis overly large (e.g., inter-only at ¯κ=9), indicating that excessive stitching
capacity can introduce drift or dilute the evidence-to-stitching ratio. Overall, this motivates using distinct
budgets for intra- and inter-passage stitching rather than a single global setting.
Table 7 analyzes where stitching tokens matter by restricting stitching to intra-passage gaps only or
inter-passage gaps only. Using either scope alone is insufficient: INTRA-ONLYyields strong HotpotQA
performance but lags on NQ and 2Wiki, while INTER-ONLYimproves cross-passage reasoning metrics
but is slightly worse on HotpotQA and does not help TriviaQA. Enabling BOTHconsistently delivers
the best results across all datasets. This indicates that effective context construction requiresbothlocal
smoothing within retrieved passages and global bridging across passages: intra-passage stitching resolves
fragment boundaries and improves readability, while inter-passage stitching is essential for multi-hop
evidence composition and cross-passage coherence.
Results of StitchRAG with different Diffusion Backbone (Dream).Table 8 shows the performance of
StitchRAG comparing different diffusion backbones for stitching: LLaDA (Nie et al., 2025) and Dream (Ye
et al., 2025). The Dream-based variant achieves competitive performance across all benchmarks, ranking
second on HotpotQA (EM/F1), NQ F1, and TriviaQA (EM/F1) compared to the baseline methods shown
in the main results in Table 1. These results indicate that the proposed framework generalizes across
diffusion backbones.
A.2 Qualitative Analyses
We present three qualitative case studies, each organized around a different baseline: NaiveRAG, AR-RAG,
and No Stitching. Each case highlights how STITCHRAG improves context construction and downstream
answer generation under a distinct retrieval failure pattern.
12

Case Study: Evidence Stitching for Multi-hop Reasoning (HotpotQA)
Question
When was the novel published in 2010 which is part of the series of The Maze Runner?
Correct Answer
September 18, 2010
Reasoning Requirement
The Maze Runner series→novel released in 2010
⇓
The Scorch Trials
⇓
publication date→September 18, 2010
Evidence in the Original Retrieved Context
The retrieved passages already contain the necessary evidence, but the relevant facts arefrag-
mented across different sentences. In the original context, the two pieces of evidence are
separated by multiple sentences:
. . .
The Scorch Trialsis a 2010 young adult post-apocalyptic . . . in the“The Maze Runner” series.
The novel was published onSeptember 18, 2010by Delacorte Press.
. . .
Failure of Naive Context Concatenation
The naive RAG concatenates retrieved passages into a long context (1607 tokens) without explicitly
connecting related evidence. As a result, the relationship between theMaze Runnerseries, the
2010 novel, and its publication date remains implicit. The model therefore stops at the intermediate
entity:
NaiveRAG→The Scorch Trials(incorrect)
Evidence Stitching by Diffusion
StitchRAG preserves the original evidence spans asanchorsand generates a short connective
sentence that stitches them into a coherent reasoning chain:
. . .
The Scorch Trialsis the second book releasedin the series, onSeptember 18, 2010.
. . .
Importantly, the stitched sentence does not introduce new factual evidence. Instead, it connects the
preserved evidence spans into a form that is easier for the language model to interpret.
Outcome
NaiveRAG→The Scorch Trials (incorrect)
StitchRAG→September 18, 2010(correct)
Context Reduction
1607 tokens→1219 tokens (24% compression)
Analysis
This example highlights the role of diffusion-based stitching. Although the retrieved context
already contains the necessary evidence, the relevant information appears in fragmented sentences.
StitchRAG preserves the original evidence spans and generates a short bridging sentence that
connects them into a coherent reasoning path. This stitched context makes the relationship between
evidence explicit and shortens the multi-hop reasoning process, enabling the model to produce the
correct answer.
13

Case Study: Evidence Stitching for Multi-hop Reasoning (WikiMultiHopQA)
Question
Who is James Roosevelt Roosevelt’s sibling-in-law?
Correct Answer
Eleanor Roosevelt
Reasoning Requirement
James Roosevelt Roosevelt
⇓(half-brother)
Franklin Delano Roosevelt
⇓(married)
Eleanor Roosevelt
Evidence in the Original Retrieved Context
The retrieved passages contain the necessary facts, but they appear inseparate documents. The
relationship chain required for multi-hop reasoning is therefore fragmented across the context.
. . .
“James Roosevelt “Rosy” Roosevelt (April 27, 1854 – May 7, 1927) was an American diplomat, heir, and theolder
half-brother of 32nd President of the United States Franklin Delano Roosevelt.”
. . .
“In 1905, hemarried his fifth cousin once removed, Eleanor Roosevelt.”
. . .
These two pieces of evidence provide the required relations:
James Roosevelt Roosevelt→Franklin Delano Roosevelt
Franklin Delano Roosevelt→Eleanor Roosevelt
However, in the original retrieved context these facts are separated by large amounts of unrelated
biography text, making the reasoning path implicit.
Failure of Naive Context Concatenation
Naive RAG concatenates the retrieved passages without aligning the relations across documents.
As a result, the model often stops at the intermediate entity:
NaiveRAG→Franklin Delano Roosevelt (incorrect)
The final relational step linking Franklin Roosevelt to Eleanor Roosevelt is not explicitly connected
to the earlier evidence.
Evidence Stitching by Diffusion
StitchRAG preserves the original evidence spans and places them within a single coherent reasoning
segment:
. . .
James Roosevelt “Rosy” Roosevelt was theolder half-brother of Franklin Delano Roosevelt.
Franklin Delano Rooseveltmarried Eleanor Roosevelt.
. . .
This stitched context makes the relational chain explicit:
James Roosevelt Roosevelt→Franklin Delano Roosevelt→Eleanor Roosevelt
Outcome
NaiveRAG→Franklin Delano Roosevelt (incorrect)
StitchRAG→Eleanor Roosevelt(correct)
Analysis
The required relations already exist in the retrieved passages but appear in separate documents
and are embedded in irrelevant text. By preserving the evidence spans and placing them in a
coherent context, StitchRAG enables the model to connect the relations and correctly infer that the
sibling-in-law of James Roosevelt Roosevelt is Eleanor Roosevelt.
14

Case Study: Diffusion-based Knowledge Completion (HotpotQA)
Question
The rapper whose debut album was titled “Thug Misses” has sold over how many records
worldwide?
Correct Answer
2 million
Reasoning Requirement
This HotpotQA example requires bridge-type multi-hop reasoning:
Thug Misses→rapperKhia
⇓
Khia’s worldwide record sales→2 million
Missing Evidence in the Retrieved Context
The retrieved passages correctly identify the artist associated with the album, but they do not
contain the worldwide record sales required to answer the question.
...
“Thug Missesis the debut album by American rapperKhia.”
“The album spawned the single “My Neck, My Back (Lick It)”.”
“As of June 2007 the album had sold611,000 copies in the US.”
...
Thus, the retrieved evidence provides information about the album and its U.S. sales, but the key
fact about Khia’s worldwide record sales is missing from the retrieved context.
Model Outputs
Because the retrieved passages prominently contain the number611,000, several RAG systems
incorrectly treat this number as the answer.
NaiveRAG→611,000 (incorrect)
No Stitching→611,000 (incorrect)
AR-RAG→Not specified
StitchRAG→2 million(correct)
Diffusion-based Completion
After diffusion-based context reconstruction, the refined context contains the following passage:
...
“The album was moderately successful and achieved moderate crossover chart success.”
“Khia released her album later that year, which also gained gold status.”
“Khia sold over 2 million records worldwide.”
...
Notably, the statement about Khia’s worldwide sales does not appear in the retrieved passages.
Analysis
This example illustrates that diffusion-based stitching can go beyond simply connecting retrieved
evidence spans. While the retrieved context contains only partial information (the album and its U.S.
sales), the diffusion process introduces a missing factual link using parametric knowledge learned
during pretraining. By combining the retrieved evidence with this generated fact, StitchRAG is
able to complete the multi-hop reasoning chain and produce the correct answer.
15

Case Study: Structural Evidence Stitching (HotpotQA)
Question
Which author wrote a play in 2006 and a book on homosexual sex in 1977?
Correct Answer
Edmund Valentine White III
Reasoning Requirement
This question requires identifying an author who satisfies two conditions:
book about homosexual sex (1977)→author→playwright (2006)
Evidence in the Original Context
The retrieved passages mainly discuss unrelated topics such as:
• religious views on homosexuality
• legal cases and legislation
• unrelated books and films
• historical events
Although the wordhomosexual sexappears multiple times, the passages do not contain any
information linking a specific book to its author, nor do they provide any connection to theatrical
works.
Structural Evidence Stitching in the Stitched Context
The stitched context organizes evidence into structured topical segments. For example:
Documentary film
Gay Sex in the 70s
and the literary reference:
Faggots (novel)
“The novel’s portrayal of promiscuous sex and recreational drug use provoked controversy.”
In addition, the stitched context contains structural markers such as:
### References
These elements group related cultural and literary references, creating a structured context around
LGBTQ-related works.
How Stitching Enables Reasoning
The stitched context introduces a crucial evidence anchor:
Faggots (novel)
This work is associated with the authorEdmund Whiteand is known for its depiction of homo-
sexual culture in the late 1970s. This anchor satisfies the first condition in the question:
book about homosexual sex (1977)
Through structural evidence stitching, related passages about LGBTQ literature and cultural works
are placed within the same semantic region. This stitched context allows the model to identify
Edmund Whiteas the author and recognize that he is also a playwright who produced theatrical
works in 2006.
Rather than inserting the answer directly,StitchRAGstitches topically related evidence into a
coherent reasoning structure, enabling the model to recover the reasoning chain and correctly infer
the answer.
16

Error Case
Question
Who is Thessalonike of Macedon’s father-in-law?
Correct Answer
Antipater
Model Prediction
Cassander
Reasoning Requirement
Answering this question requires a two-step kinship inference:
Thessalonike of Macedon→wife of Cassander
Cassander→son of Antipater
⇓
father-in-law of Thessalonike→Antipater
Evidence in the Retrieved Context
The retrieved context clearly supports that Thessalonike married Cassander, but it does not explicitly
preserve the second link needed for the full inference:Cassander →son of Antipater. Thus, the
context strongly supports identifying the husband, but not the father-in-law.
[. . . ]
“Thessalonikewas a Macedonian princess . . . andwife of Cassander.”
[. . . ]
“Cassander married her. Thessalonike became queen of Macedon . . . ”
[. . . ]
Example of Structural Stitching
In the refined context, the main preserved evidence anchor is:
Thessalonike of Macedon→wife of Cassander
This anchor provides onlypartial structural Stitching. It links Thessalonike to her husband, but
not Cassander to his father:
Cassander→Antipater’s son
As a result, the context encourages arelation collapsefromfather-in-lawtohusband.
Reasoning Failure Induced by Incomplete Structural Stitching
With only the first kinship link preserved, the model defaults to the nearest available male relative:
Thessalonike→Cassander
Prediction→Cassander
But the correct reasoning requires an additional genealogical step:
Thessalonike→husband: Cassander
Cassander→father: Antipater
⇓
father-in-law→Antipater
Analysis
This is a failure case of structural stitching. The refinement preserves an anchor sufficient to recover
Thessalonike’s husband, but not the extra bridge needed to resolve the father-in-law relation. The
model therefore predictsCassander, a locally supported but semantically incorrect answer. The
gold answer,Antipater, requires a second genealogical link that is absent from the refined context.
17

Error Case
Question
Where does the director of film Présentation ou Charlotte et son steak work at?
Correct Answer
Cahiers du cinéma
Model Prediction
Switzerland
Reasoning Requirement
Answering this question requires identifying the film’s director and then linking that director to his
workplace:
Présentation ou Charlotte et son steak→directed by Éric Rohmer
Éric Rohmer→worked at Cahiers du cinéma
⇓
director’s workplace→Cahiers du cinéma
Evidence in the Retrieved Context
The retrieved context preserves the first step but not the second, and instead highlights an irrelevant
location cue tied to the film’s setting.
[. . . ]
“Présentation ou Charlotte et son steakis a French short drama film written and directed byÉric Rohmer
. . . In aSwiss villagein winter, Walter is walking with Charlotte when they meet Clara.”
[. . . ]
The missing second-hop fact is:
Éric Rohmer→worked atCahiers du cinéma
Example of Structural Stitching
In the refined context, the system preserves the director anchor but also highlights a competing
location anchor:
Présentation ou Charlotte et son steak→Éric Rohmer
Présentation ou Charlotte et son steak→Swiss village
These anchors createmisleading structural stitching, since the workplace link is missing while
an irrelevant geographical cue remains salient.
Reasoning Failure Induced by Misleading Structural Stitching
The model may therefore substitute the film’s setting for the director’s workplace:
Présentation ou Charlotte et son steak→Swiss village
Swiss village→Switzerland
Prediction→Switzerland
The correct reasoning is:
Présentation ou Charlotte et son steak→director: Éric Rohmer
Éric Rohmer→workplace: Cahiers du cinéma
⇓
answer→Cahiers du cinéma
Analysis
This example illustrates a relation-substitution error. Although the refined context correctly links
the film toÉric Rohmer, it does not preserve the evidence connecting Rohmer toCahiers du
cinéma. Instead, it highlights the film’s Swiss setting, which encourages the model to substitute a
salient location for the director’s workplace.
18

A.3 Prompt Templates
Relevance probing prompt.We estimate R(x, q) =P(Z=1) using a fixed binary prompt that asks
whether a retrieved passage is useful for answering the query. We decode neither a rationale nor a
free-form answer; instead, we only score the canonical responsesYesandNo.
Question:⟨q⟩
Passage:⟨x⟩
Is this passage relevant to the question? Answer (Yes/No):
Scoring.Let LYesandLNodenote the (sequence) log-probabilities of the targets YesandNounder the
model given the prompt. If a target is tokenized into multiple tokens, we sum log-probabilities across its
token sequence. We then compute
R(x, q) =exp(L Yes)
exp(L Yes) + exp(L No).(13)
Answer generation prompt (RAG).For all methods, the final answer is generated by an autoregressive
generator conditioned on the (possibly compressed/stitched) context. We use the following instruction-
style RAG prompt.
[System]Answer the question based on the given passages. Provide a direct and accurate answer.
[User]
Passage:
⟨context⟩
Question:⟨q⟩
Please provide the answer in the format: ‘The answer is: [your answer]’.
A.4 Average Retention ratior∗of StitchRAG
Table 9: Averager∗of retrieved tokens achieved by STITCHRAG on each dataset.
HotpotQA NQ TriviaQA 2Wiki PopQA Avg.
Avg. retention ratio (%) 74.25 72.83 76.26 67.44 70.66 72.29
A.5 Vectorized Implementation of Dynamic Remasking
This appendix compares the original Fast-dLLM dynamic remasking implementation (Wu et al., 2025)
with our vectorized PyTorch version. The vectorized implementation follows the same dynamic remasking
policy as the original version and produced matching transfer decisions in all of our tests, while removing
Python-level per-sample loops to improve batched inference efficiency.
Background (Dynamic Remasking Policy).Let masked tokens be ranked by confidence in descending
order. Conceptually, the algorithm selects the longest confidence-sorted prefix whose elements satisfy a
rank-dependent threshold,
τr= 1−α
r+ 1,(14)
where αis a factor parameter (typically α= 1 ), while ensuring that at least one token is transferred
whenever masked positions exist.
Original Implementation (Fast-dLLM)The original implementation iterates over batch elements in
Python: it computes per-sample thresholds, sorts masked-token confidences, scans ranks to find the cutoff
k, and then applies per-sampletopkselection.
Our Vectorized ImplementationWe implement the same logic using tensor-native operations only.
Instead of sorting only masked positions, we sort the full sequence after assigning unmasked positions a
sentinel “ −∞-like” value, ensuring they are placed at the tail and do not affect the top- kmasked ranks.
Cutoff detection and index materialization are fully vectorized across the batch.
19

Table 10: Comparison of implementation strategies.
Component Original Ours
Batch iteration Pythonforloop Vectorized
Threshold computation List comprehensiontorch.arange
Sorting Per-sample (masked-only) Batch-parallel (full, with sentinel)
Cutoff detection Nested loop Prefix-product (cumprod)
Selection Per-sampletopkBatchedscatter
Edge cases Manualif torch.where
Table 11: Inference time comparison of get_transfer_index_dynamic (seq_len=2048, vocab=32k, warmup=10,
repeat=50). We conducted our experiments on a single NVIDIA A100 GPU.
Batch Size Original Ours Speedup
1 2.41 ms 2.31 ms 1.04×
4 9.25 ms 8.49 ms 1.09×
8 26.49 ms 20.00 ms 1.32×
16 44.92 ms 36.00 ms 1.25×
32 84.56 ms 65.14 ms 1.30×
Key OptimizationTo avoid explicit scanning loops, we implement threshold-based prefix selection
with a vectorized prefix-product operation over the sorted pass mask. This yields the valid prefix length
efficiently and is well-suited for batch-parallel GPU execution.
PerformanceBoth versions are dominated by sorting at long sequence lengths; the practical speedup
comes from eliminating Python overhead and enabling batch-parallel execution on GPU, which improves
utilization forB≥4in our setup.
Equivalence ValidationThe vectorized implementation follows exactly the same threshold rule and
prefix selection criterion as the original version. We verified output equivalence through randomized unit
tests over 1,000 cases with varying batch sizes, sequence lengths, and mask densities, and observed no
accuracy difference in end-to-end evaluation on HotpotQA and NQ.
20

Table 12: Original Fast-dLLM implementation (sequential per-sample loop).
defget_transfer_index_dynamic(logits, temperature, remasking,
mask_index, x, num_transfer_tokens, factor=1):
# Sample proposals and compute confidence
logits_with_noise = add_gumbel_noise(logits, temperature)
x0 = torch.argmax(logits_with_noise, dim=-1)
p = F.softmax(logits.to(torch.float64), dim=-1)
x0_p = torch.squeeze(
torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
x0 = torch.where(mask_index, x0, x)
confidence = torch.where(mask_index, x0_p, -np.inf)
transfer_index = torch.zeros_like(x0, dtype=torch.bool)
num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
# Per-sample loop (sequential processing)
forjin range(confidence.shape[0]):
num_tokens =int(num_transfer_tokens[j].item())
ifnum_tokens == 0:
continue
# Compute thresholds (Python list comprehension)
ns =list(range(1, num_tokens+1))
es = [factor/(n+1)forninns]
threshs = [1-eforeines]
threshs[0] = -1 # Ensure at least one transfer
# Sort masked tokens only
sorted_confidence = torch.sort(
confidence[j][mask_index[j]], descending=True)[0]
# Find cutoff rank
fortop_iin range(len(threshs)):
ifsorted_confidence[top_i] < threshs[top_i]:
break
# Edge case handling
iftop_i == 0ortop_i ==len(threshs)-1:
top_i += 1
# Select top-k
_, select_index = torch.topk(confidence[j], k=top_i)
transfer_index[j, select_index] = True
returnx0, transfer_index
21

Table 13: Our vectorized implementation (batch-parallel on GPU).
defget_transfer_index_dynamic(logits, temperature, remasking,
mask_index, x, num_transfer_tokens=None,
factor=1.0, threshold=None):
# Sample proposals and compute confidence
logits_with_noise = add_gumbel_noise(logits, temperature)
x0 = torch.argmax(logits_with_noise, dim=-1)
p = F.softmax(logits.to(torch.float64), dim=-1)
x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
x0 = torch.where(mask_index, x0, x)
neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min,
device=x0_p.device, dtype=x0_p.dtype)
confidence = torch.where(mask_index, x0_p, neg_inf)
B, L = confidence.shape
mask_counts = mask_index.sum(dim=1).to(dtype=torch.long)
# Early exit for empty masks
if(mask_counts == 0).all():
returnx0, torch.zeros_like(mask_index, dtype=torch.bool)
# Vectorized threshold computation
ranks = torch.arange(L, device=confidence.device, dtype=torch.float64)
thr = 1.0 - (float(factor) / (ranks + 2.0))
thr[0] = torch.finfo(torch.float64).min# At least one transfer
# Batch sorting (parallel across batch dimension)
values, idx = torch.sort(confidence, dim=1, descending=True)
# Vectorized threshold comparison
pass_mask = values >= thr.unsqueeze(0) # (B, L)
# Prefix product to find consecutive passing ranks
prefix_ok = torch.cumprod(pass_mask.to(torch.int64), dim=1)
k = prefix_ok.sum(dim=1).to(dtype=torch.long) # Per-sample k
# Vectorized edge case handling (match original boundary handling)
k = torch.minimum(k, mask_counts)
k = torch.where(mask_counts > 0, torch.clamp(k,min=1), k)
k = torch.where((mask_counts > 0) & (k == mask_counts-1),
mask_counts, k)
# Vectorized top-k selection via scatter
cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
select_sorted = cols < k.unsqueeze(1)
transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
transfer_index = transfer_int.bool() & mask_index
returnx0, transfer_index
22
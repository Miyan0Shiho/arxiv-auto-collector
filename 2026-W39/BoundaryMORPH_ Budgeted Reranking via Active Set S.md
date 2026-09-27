# BoundaryMORPH: Budgeted Reranking via Active Set Selection for Diffuse Retrieval

**Authors**: Eylon Caplan, Shamik Roy, Shib Sankar Dasgupta, Yingfan Wang, Rashmi Gangadharaiah

**Published**: 2026-09-23 01:22:22

**PDF URL**: [https://arxiv.org/pdf/2609.27213v1](https://arxiv.org/pdf/2609.27213v1)

## Abstract
Open-ended queries in modern Retrieval-Augmented Generation (RAG) are increasingly "diffuse," requiring a large set of documents to be assembled into a finite LLM context window. To ensure retrieval quality, systems use fast dual-encoders and more expensive cross-encoders (CEs) to score candidates. However, the CE budget $B$ is strictly bounded by latency and is often smaller than the context window capacity $k$. This mismatch makes standard reranking structurally flawed: it wastes compute verifying obvious top candidates while ignoring relevant documents further down the initial ranking. To address this, we introduce BoundaryMORPH, a novel algorithm that allocates CE budget specifically for the LLM's context capacity $k$. Using a Gaussian Process, BoundaryMORPH treats the initial dual-encoder ranking as a structural prior and intelligently spends CE calls on resolving top-$k$ set membership at the boundary, rather than seeking a single most-relevant document. Information from each CE call propagates to unscored documents, maximizing the utility of the budget. We demonstrate that BoundaryMORPH achieves state-of-the-art set retrieval quality across multiple models and datasets with open-ended queries ($+5.4$ nCG@100 over the strongest baseline).

## Full Text


<!-- PDF content starts -->

Preprint. Under review.
BOUNDARYMORPH: BUDGETEDRERANKING VIA
ACTIVESETSELECTION FORDIFFUSERETRIEVAL
Eylon Caplan∗♠Shamik Roy♣Shib Sankar Dasgupta♣
Yingfan Wang♣Rashmi Gangadharaiah♣
♠Purdue University♣AWS AI Labs
ecaplan@purdue.edu
{royshami,shibdg,yingfanw,rgangad}@amazon.com
ABSTRACT
Open-ended queries in modern Retrieval-Augmented Generation (RAG) are in-
creasingly “diffuse,” requiring a large set of documents to be assembled into a
finite LLM context window. To ensure retrieval quality, systems use fast dual-
encoders and more expensive cross-encoders (CEs) to score candidates. However,
the CE budgetBis strictly bounded by latency and is often smaller than the con-
text window capacityk. This mismatch makes standard reranking structurally
flawed: it wastes compute verifying obvious top candidates while ignoring rele-
vant documents further down the initial ranking. To address this, we introduce
BoundaryMORPH, a novel algorithm that allocates CE budget specifically for the
LLM’s context capacityk. Using a Gaussian Process, BoundaryMORPH treats the
initial dual-encoder ranking as a structural prior and intelligently spends CE calls
on resolving top-kset membership at the boundary, rather than seeking a single
most-relevant document. Information from each CE call propagates to unscored
documents, maximizing the utility of the budget. We demonstrate that Boundary-
MORPH achieves state-of-the-art set retrieval quality across multiple models and
datasets with open-ended queries (+5.4nCG@100 over the strongest baseline).
1 INTRODUCTION
The challenge of serving queries whose relevant material is scattered across many documents has
long been recognized in IR (Bates, 1989; Marchionini, 2006; Clarke et al., 2008), and they have
“Effects of light pollution?”“…melatonin suppressed…”“…sea turtles crawl inland…”“…sky glow hides stars…”“…city banning of bright signs…”…140 more…144 True Relevant DocumentsTop 5k Retrieved Documents by MIPS2,0003,0004,0005,0001,000…Which Docs to Score with Only B=25 CE Calls?Which Docs to Finally Select for Context Capacity k=100??RankDiffuse Query
All MIPS ScoresBudgeted Context SelectionSelection
Target: 100 Relevant DocsLLM Context
Figure 1: The diffuse query hasmanygold relevant documents (144), and they are spread across the
MIPS top-5k ranking. The LLM generator has been allotted a context capacityk= 100, but we
have only a CE budget ofB= 25. The selection problem then becomes: how should we optimally
spend these CE calls and finally select the 100 most relevant documents for the LLM?
∗Research done during internship at AWS AI Labs
1
arXiv:2609.27213v1  [cs.IR]  23 Sep 2026

Preprint. Under review.
grown more prominent as people bring broader, more open-ended requests to LLM-based systems
(Suri et al., 2024; Spatharioti et al., 2025) and on to deep-research agents that survey and synthesize
many sources per query (Huang et al., 2025; Coelho et al., 2025). For RAG, this takes the form
of queries withmanyrelevant documents; for example, in a deep research setting, a query like
“Effects of light pollution?” is expected to surface tens, if not hundreds of relevant documents (see
Figure 1)! Suchdiffusequeries cannot be well-responded to by referencing a single, or even handful
of relevant documents; rather, they require a larger set of relevant documents to be inserted into the
LLM generator’s context for the best response.
Modern RAG systems typically rest on a two-stage architecture: a cheapdual-encoderembeds
the query and every document into a shared vector space and returns a candidate pool ranked via
Maximum Inner Product Search(MIPS) (Hambarde & Proenc ¸a, 2023; Zhao et al., 2024; Huang
& Huang, 2026), and an expensive reranker, such as across-encoder(CE) or LLM, re-scores a
small number of top candidates far more accurately (Gao et al., 2024; Fan et al., 2024).
Because the CE is an expensive test-time process, it is typically applied to only a limited number,B
(budget calls), of candidates. Likewise, a RAG system is always limited by itscontext capacity,k:
the size of the context that has been allotted for documents being inserted into the RAG generator’s
context before generating a response (Cuconasu et al., 2024; Shi et al., 2024; Liu et al., 2024; Louis
et al., 2025).
These two limits become problematic for diffuse queries, for two reasons: (1) re-scoring only the
topBdual-encoder documents risks missing out on relevant documents further down, and (2) when
the CE budget is smaller than the context capacity (B < k), the remainingk−Bdocuments cannot
anyway be evaluated by the CE due to budget constraint, and thus we gain no information about
which documents to fit into the context, and which to leave out.
Figure 1 shows this clearly, where we assume a context capacity ofk= 100, but only aB= 25
CE call budget: the query is diffuse, with144truly relevant documents, spread across the MIPS
ranking. The context capacityk= 100means we have100slots to fill our context with (hopefully)
100/144relevant documents. However, our budget allows for only25CE calls, which demonstrates
problem (2). This begets a resource allocation problem: how should we optimallyspendthese CE
calls and finallyselectthe100most relevant documents for the RAG generator?
To address this, we introduceBoundaryMORPH, a Gaussian Process (GP) selector (Williams &
Rasmussen, 1995; Srinivas et al., 2012) that actively estimates the CE’s relevance landscape to
optimally allocate the budget ofBcalls. The use of a GP allows us topropagateinformation from
each CE call, so that even with only, say,25CE calls, we may still gain information about far more
than25documents. BoundaryMORPH is built on two core insights regarding the nature of diffuse
queries and the RAG pipeline:
1.Morphing the MIPS’s ranking rather than discarding it:While the cheap dual-encoder is
imperfect and locally miscalibrated, it provides a strong, scale-free global surface. This signal
should be leveraged as a structural baseline rather than discarded.
2.Targeting the capacity boundary rather than the maximum:Because the RAG context capac-
itykis known in advance, spending budget to confidently confirm the absolute highest-scoring
documents is wasteful; they will be included in the context regardless. Genuine uncertainty only
exists at theboundaryof inclusion.
BoundaryMORPH operationalizes these insights through a single running GP estimator. To satisfy
the first insight, we set the GP prior mean to the document’s MIPS percentile rank. This allows
the model to smoothly fall back to the dual encoder’s judgment in unscored regions, spending calls
only tobendthis surface rather than rebuild it. To satisfy the second, BoundaryMORPH abandons
traditional global-maximum seeking, and instead, actively targets the inclusion boundary, spending
its limited budget on the most contested boundary gaps, where a CE call can actually change which
documents are delivered to the generator.
In summary, our main contributions are:
•A New Formulation for RAG Selection:We frame document selection for RAG as a budgeted
top-kset-selection problem, and identify the structural mismatch of traditional approaches with
this context selection problem.
2

Preprint. Under review.
•State-of-the-Art Diffuse Retrieval:We propose an efficient, parameter-free Gaussian Process
selector that leverages a MIPS prior to warm-start the relevance landscape and utilizes a standard-
ized boundary gap to explicitly target the top-kdecision threshold. We demonstrate through exten-
sive evaluation that BoundaryMORPH consistently outperforms existing active selection baselines
(BAGEL (Kim et al., 2026), RGS (Xu & Chen, 2025), and single-stage reranking) at equal bud-
gets across multiple embedding-CE pairs, achieving the highest selection quality on both human
relevance judgments (nCG@k) and cross-encoder fidelity (CE-nCG@k).
•Analysis of Query Multimodality:We introduce a geometric measure of querymultimodal-
ity, revealing that our method’s performance advantage is fundamentally driven by its ability to
navigate complex relevance landscapes. Because diffuse queries often exhibit multiple seman-
tic “peaks” or sub-topics, max-seeking baselines over-measure the primary mode. By explicitly
avoiding redundant calls on securely top-kdocuments, BoundaryMORPH naturally crosses over
to explore and select documents from secondary peaks, yielding uniform and growing perfor-
mance gains on highly multimodal queries.
2 PROBLEMSETTING
Setup.We define adiffusequery to be a query which requires more than a single or a handful
of relevant documents for the best response. Unlike simple factoid queries that require a single
document, diffuse queries (e.g., “Effects of Light Pollution?”) require tens or hundreds. In this work,
we focus on the latter. A queryqhas an ground truth, unobserved relevance functionrel(q, d)∈
[0,1]over documentsd∈Dfrom corpusD. Two scoring functions can estimate it. Theretriever
s(q, d)is cheap and amortized across queries, and narrows the corpus to a candidate poolCwith
|C|=N. In practice,Nis a standard first-stage retrieval depth chosen to be large enough to bound
maximum recall. The cross-encoder(CE),f CE(q, d)∈[0,1], is the more accurate estimate and
the more expensive one: it must be evaluated separately for each query-document pair, and we may
call it at mostB≪Ntimes per query. The consumer of the results accepts a fixed numberkof
documents, thecontext capacity. The system returns a setS⊆Cwith|S|=k.
Objective.We want the returned set to carry as much relevance as possible,
S∗= arg max
S⊆C,|S|=kX
d∈Srel(q, d).(1)
Givenrel, this is trivial: return thekmost relevant documents. The difficulty is thatrelis unknown,
thatf CEis the closest estimate of it we can obtain, and that we can afford onlyB≪Ncalls tof CE.
The problem is therefore one of resource allocation: whichBdocuments to score, so that thekwe
return afterwards are as relevant as possible.
Context capacity.The capacitykis the number of retrieved documents the RAG generator is given
before it begins producing an answer. In deployed RAG pipelines this is a managed allotment: a
finite context window may be divided among the system prompt, tool schemas, conversation history,
retrieved documents, and response headroom. Hence, the slice reserved for retrieved documents,k,
is known, and the task is one ofselection: to choose whichkdocuments to return (a set). We
view arranging them within the context as a separate problem Cuconasu et al. (2025), and in this
work, focus on the task ofselectingthe bestkdocuments to fit into the model context. The problem
therefore contains two decisions:whereto spend theBCE calls, andwhichkdocuments to return
once they are spent.
3 METHOD
At a high level, our approach uses a Gaussian Process (GP) fitted with specialized components for
this task. We first introduce GPs and the standard way such an estimator is used to choose what to
score. The two subsections that follow are the specializations fitting it to budgeted selection under
a known capacity: (1) what the estimator starts from, and (2) how it picks the next document. Full
pseudocode is given in Appendix B.1.
3

Preprint. Under review.
1
5
18True RankNo Calls Spent   (t=0)
No Infor-
mation Yet1
5
18Halfway   (t=3)
4 on the
Wrong Side1
5
18Budget Spent   (t=6)
2 on the
Wrong Side
1
5
18True Rank
6 on the
Wrong Side1
5
18
2 on the
Wrong Side1
5
18
Ignored:
Obviously
Insidethe Top-k
Ignored:
Obviously
Outsidethe Top-kClean
Split
Estimated CE Score   (± Uncertainty)
Perfect EstimateTrue Top-k Boundary Wrong Side of Cutoff CE Call SpentOur Cutoff: k-th Largest Est.No Prior + Max-Seek-
ing (Standard GP)MIPS Prior + Boundary
BoundaryMORPH
Figure 2: BoundaryMORPH vs. a standard GP.Top row:A standard GP (zero prior, max-seeking
acquisition) wastes limited budget repeatedly scoring obvious top documents.Bottom row:Bound-
aryMORPH leverages a MIPS prior and targets the top-kboundary. By ignoring documents safely
inside or outside the cutoff, it spends its budget exclusively on resolving contested edge cases.
3.1 PRELIMINARIES: GAUSSIAN-PROCESS SURROGATES
The estimation problem.The cross-encoderf CEmay be evaluated atBpoints but must inform a
decision over allN. That calls for asurrogate: a model that predictsf CE(q, d)at every unscored
document and reports how far each prediction can be trusted. Uncertainty is necessary, since the
selector needs it to tell a resolved region from an unexamined one.
Gaussian processes.A Gaussian process is a distribution over functions,g∼ GP(µ 0, K), specified
by two objects (Williams & Rasmussen, 1995): aprior meanµ 0(d), the estimate ofg(d)before
any evidence, and akernelK(d, d′), which sets how strongly an observation atd′constrains the
estimate atdand is typically a decreasing function of the distance between the two documents. It
is nonparametric: no weights or fitting, and all posteriors below are exactly computed. Heregis a
generic function placed under the GP; §3.2 instantiates it as the residual off CEagainst the MIPS
percentile prior.
Closed-form conditioning.Aftertcalls ofgonA t={d 1, . . . , d t}returning scoresy t, the posterior
at any documentdis Gaussian with mean and variance
µt(d) =µ 0(d) +k t(d)⊤eK−1
t 
yt−µ0(At)
, σ2
t(d) =K(d, d)−k t(d)⊤eK−1
tkt(d),(2)
wherek t(d) =k(d, A t)is the vector of kernel values betweendand the scored documents and
eKt=K(A t, At) +σ2
nIwithσ2
na very small observation-noise (jitter) term. Both are available for
allNdocuments at once, and both update as each new score arrives.
Two properties of these formulas are what selectors can run on. First,uncertainty collapses around
evidence: at a scored documentσ2
t≈σ2
n≈0, and at its kernel-neighborsσ tis reduced too, so a
region can be recognized as resolved. Second,the mean reverts to the prior far from evidence:
askt(d)→0the correction term vanishes andµ t(d)→µ 0(d). The posterior is alocal deformation
of the prior around the points that were paid for.
Acquisition: GP-UCB.Given a surrogate, the standard rule for choosing the next evaluation is the
upper confidence bound (Srinivas et al., 2012; Djolonga et al., 2013),
d⋆= arg max
d /∈A tµt(d) +β σ t(d),(3)
which tradesexploitinga high estimated score againstexploringan uncertain one, with the weight
βsetting the balance (commonly tuned or annealed). GP-UCB is formulated and analyzed as a
4

Preprint. Under review.
method for locating theglobal maximumofgwith few function evaluations. A GP surrogate thus
leaves exactly two things to specify: the pair(µ 0, K), and the acquisition rule.
3.2 THEMIPSPERCENTILE PRIOR
MIPS prior.While the MIPS ranking may not be as accurate as CE relevance, it is certainly
not noise, and it is free (pre-computed). It encodes mostly-reliable geometry that may be mis-
calibrated and locally wrong. To take advantage of this, we have the GP model theresidual,
r(d) =f CE(q, d)−µ 0(d), rather than the raw CE field. That is, we aim to approximate where,
and how much tomorphthe embedding similarity in order to reach CE relevance. Because the CE
outputs are calibrated probabilities in[0,1], we can meaningfully model the residual against a[0,1]
MIPS percentile rank. We setµ 0(d)to the document’sMIPS percentile rankwithinC(≈1at the
head,≈0in the tail), which is scale-free, requiring no per-query calibration. The posterior mean
is thenµ 0plus kernel-weighted observed residuals. Far from any observationµ t→µ 0: the surro-
gate falls back to trusting the MIPS’s ranking. Callsbendthe MIPS’s surface; they never rebuild it.
This is also the easier estimation problem: the residual is near zero across most of the pool, so “no
correction” is a good default almost everywhere and the GP only has to represent the surprises.
Kernel and query observation.We use a directional kernel (vMF) on unit embeddings,K(d, d′) =
exp 
κ(cos(d, d′)−1)
(Mukhopadhyay et al., 2020; Mardia & Jupp, 2009). Alternative kernels are
compared in Appendix B.4. Residuals are Z-standardized before conditioning and de-standardized
after. Similar to Kim et al. (2026), we enter the query vector into the GP before any budget is spent,
as a zero-cost observation at maximal relevance (y= 1), as if we had observed a maximally relevant
document. This costs no CE call and gives the surrogate one anchor of known value from the start,
and pinches the variance near the query.
3.3 THE BOUNDARY ACQUISITION RULE
The mismatch of maximum-seeking acquisition.The objective from §2 targetsmembershipin a
set of sizek. We argue that standard maximum-seeking acquisition (e.g., GP-UCB) is fundamentally
misaligned with this goal. First, it wastes calls on documents already securely inside the top-k.
Their exactvalueis uncertain, but theirmembershipis not; they will enter the context window
whether or not we pay to measure them. Symmetrically, it explores regions securely below the
cutoff, confirming exclusions that were never in doubt. A maximum-seeker has no concept of the
known capacityk, so it cannot ignore documents whose fate is already settled. Instead, a useful CE
call is one that could plausiblymove a document acrossthe boundary, as only these calls actually
change the final context fed to the LLM.
Relation to active set ordering.Our selection rule is adapted from Mean Prediction in Nguyen et al.
(2024) and, in the independent-arm case, from LUCB (Kalyanakrishnan et al.), which estimates
thetop-ksetof a blackbox function rather than its maximizer, and spends its evaluations on the
boundary comparisons the posterior leaves ambiguous. Notably, standard GP-UCB is thek= 1
special case: a max-seeking selector is inherently minimizing regret of the global maximum.
Rule.At each step, we partitionCby the current posterior meanµ tinto itstop-k,S t(theincum-
bents, our running guess at the returned setS), and the rest,C\S t(thechallengers). See Figure 2’s
purple line. Membership is in genuine doubt only at the boundary: an incumbent whose lower con-
fidence bound dips below a challenger risks being wrongly retained, and a challenger whose upper
confidence bound rises above an incumbent risks being wrongly dropped. The contested pairs are
those close in estimated scorerelative to how uncertain we are about them, so we select the pair
with the smalleststandardized boundary gap,
(j∗, i∗) = arg min
j∈S t, i∈C\S tµt(j)−µ t(i)
σt(j) +σ t(i),(4)
the pair most likely to sit on the wrong sides of the cutoff under the current estimates. We then spend
the CE call on whichever of the two hashigherσ(the outcome we can least predict). Equivalently,
j∗is the most-threatened incumbent andi∗the strongest challenger.
Complexity.The kernel is built once,O(N2): one entry per document pair at test time. Updating
the posterior requiresO(B3N), due to the matrix inversion, but we find in Appendix D.4 thatO(B)
CE inference dominates the method’s latency. Full derivation in Appendix B.2.
5

Preprint. Under review.
4 EXPERIMENTALSETUP
4.1 DATASETS
Datasets.We evaluate on three datasets featuring broad, open-ended queries over news, govern-
ment, and encyclopedic corpora: NeuCLIRBench-monolingual-eng (Lawrie et al., 2025), Robust04
(V oorhees, 2005), and TravelDest (Wen et al., 2024), with relevance labels from Kim et al. (2026).
To ensure diffuseness, we filter out the minority of queries with<20relevant documents, leaving
85, 187, and 99 queries, respectively, with median relevant document counts of51 / 62 / 462. To
make the setting tractable and similar to realistic use-cases, we restrict evaluation to a per-query
candidate set: the topN= 10,000by Qwen3-Embedding-4B MIPS (Zhang et al., 2025). Full
statistics are in Appendix C.1.
4.2 METRICS
Using nCG@kand CE-nCG@k.We use normalized cumulative gain (nCG@k) against human
relevance judgments as our primary metric (J ¨arvelin & Kek ¨al¨ainen, 2002). Intuitively, this metric
measures,“compared to thebest possible setofkdocuments that could have been retrieved, how
relevant was the set thatwasretrieved?”. We also reportCE-nCG@k, the same quantity computed
against the CE’s scores rather than human judgments. Concretely:
nCG@k=P
d∈Srel(q, d)P
d∈S∗rel(q, d),CE-nCG@k=P
d∈SfCE(q, d)P
d∈S†fCE(q, d),(5)
whereSis the selected set of sizek,S∗is the optimal set from Eq. 1, andS†is the analogous
optimal set of sizekunder the CE’s own scores. Hence,nCG@kis the count of relevant documents
returned, divided by the number a perfect selector would fit intokslots. It reaches 1 exactly when
the returned set is as relevant as the most relevant possible set. Reporting both metrics separates
two claims: CE-nCG measures how well the budget is allocated to approximatef CE, while nCG@k
measures the retrieval quality against human judgments, rather than only tracking the CE more
faithfully.
Why not recall or nDCG@k.Because relevant documents often exceedk, nCG@kdirectly asks:
of thekdocuments the generator can read, how many are relevant?Metrics such as nDCG and
MAP score theorderingwithin the returned set, which are not well-suited for measuring whichk
documents reach the generator’s context (Trappolini et al., 2025; Samuel et al., 2026).
4.3 METHODS
The dual-encoder (embedding model) and CE aremultilingual-e5-large(Wang et al., 2024) and
zerank-2(Pipitone et al., 2025) respectively, with other models in Appendix D.2, and implementa-
tion details in B.3).BM25(Robertson et al., 1994) is a lexical floor: applied as a re-ranking of the
shared top-10k pool and truncated atk, giving one fixed ordering per query.Single-Stage Rerank
(SS)is the standard policy: score the MIPS top-Bwith the CE, place them above the tail, and leave
the tail in MIPS order.BAGEL(Kim et al., 2026) is a previous GP approach using zero-prior, and
standard UCB over the whole pool with an explore/exploit term which we sweep.RGS(Xu &
Chen, 2025) is a greedy kNN graph-based approach. Because it lacks a way to rank documents it
never scored, scored documents are placed as a band above the unscored tail, which is left in MIPS
order. RGS and BAGEL’s hyperparameters are each tuned and selected on the evaluation metric per
budget. Full implementation and sweep details for every baseline are in Appendix C.2.
5 RESULTS ANDANALYSIS
The evaluation addresses four questions in turn: (§5.1) Does BoundaryMORPH select better sets
than existing selectors at equal budget? (§5.2) Are both of its components necessary (MIPS
prior/boundary acquisition), and robust to the capacityk? (§5.3) How does BoundaryMORPH
achieve its improvement, and which queries, geometrically speaking, yield the greatest advantage?
In Appendix D.4, we also compare wall-clock latency, including against an agentic retrieval loop.
6

Preprint. Under review.
Table 1: Top-krelevance (%) under CE budgetBfor various methods. Best per column inbold,
second-best underlined . BoundaryMORPH outperforms baselines, with BAGEL a clear second, due
to the ability to estimate relevance of unscored documents.
nCG@k(%, human judgments) CE-nCG@k(%)
B=25B=50B=25B=50
Method k50 100 200 50 100 200 50 100 200 50 100 200NeuCLIRBM25 38.7 46.8 57.6 38.7 46.8 57.6 69.8 67.3 66.5 69.8 67.3 66.5
SS 39.6 46.2 55.4 39.6 46.2 55.4 73.1 70.0 68.5 73.1 70.0 68.5
RGS 39.1 46.9 55.7 37.0 47.5 56.6 72.3 70.3 68.7 68.8 70.3 69.1
BAGEL 47.3 51.4 58.5 52.9 55.7 62.3 79.7 71.5 66.9 86.9 77.6 71.6
MORPH 49.8 55.1 62.7 54.7 59.2 65.1 85.7 80.9 77.3 90.5 85.7 81.9Robust04BM25 34.7 40.0 49.7 34.7 40.0 49.7 49.7 47.6 47.8 49.7 47.6 47.8
SS 34.4 37.1 44.8 34.4 37.1 44.8 54.5 51.5 51.0 54.5 51.5 51.0
RGS 35.0 38.1 45.7 34.2 38.8 46.6 54.8 52.0 51.4 53.5 52.3 51.8
BAGEL 40.3 41.0 46.1 48.7 48.0 53.0 59.9 53.2 50.2 68.4 59.6 55.2
MORPH 45.2 46.4 52.1 52.3 51.5 57.1 66.0 60.5 57.3 72.8 65.8 61.6TravelDestBM25 23.0 21.9 23.0 23.0 21.9 23.0 57.3 57.4 57.7 57.3 57.4 57.7
SS 30.9 28.4 27.9 30.9 28.4 27.9 64.2 64.5 64.8 64.2 64.5 64.8
RGS 32.3 29.4 28.6 32.9 30.4 29.3 63.9 64.5 65.0 64.4 65.3 65.5
BAGEL 40.9 35.5 33.7 46.6 40.7 37.5 75.9 72.4 69.9 82.0 77.5 74.5
MORPH 45.1 39.8 37.8 48.9 43.1 40.8 81.1 78.4 76.5 86.5 82.9 80.3
5.1 MAINRESULTS
BoundaryMORPH outperforms baselines in all configurations.Table 1 shows the main exper-
imental results. BoundaryMORPH outperforms baselines in all (dataset×B×k) combinations,
with the biggest gains being at higherkvalues (average of +3.8 pp nCG@100 and +7.3 pp CE-
nCG@200 over second-best baseline). BAGEL outperforms the other baselines, because it is the
only baseline which also can use CE calls to update the relevance ofunscoreddocuments. In Ap-
pendix D.2, we show BoundaryMORPH’s performance gap holds across choice of embedding and
CE model. We also see thatCE gains carry over to human judgments.The trend persists across
the two metrics, indicating that better approximation of the CE’s own top-ktranslates into more
human-judged-relevant documents delivered.
5.2 ABLATIONS ANDSENSITIVITYTable 2: Ablations: nCG@k(%) averaged over
datasets and embedder×CE pairs; indented rows
give the pp change.
B=25B=50
k50 100 200 50 100 200
MORPH47.1 47.9 52.6 51.3 51.3 54.9
w/o acq. -2.6 -2.5 -2.1 -1.5 -2.3 -1.5
w/o prior -3.0 -3.1 -3.8 -2.1 -2.3 -2.3Both the MIPS prior and the boundary rule
are necessary.To test the contribution of
BoundaryMORPH’s two components, we per-
form two isolated removals, averaged over the
four model pairs and three datasets:without
acquisition, the boundary acquisition rule reverts to standard UCB at its own best sweptβ;without
prior, the MIPS percentile prior mean becomes a zero mean.
As seen from Table 2, removing boundary acquisition lowers nCG@kby 1.5–2.6 percentage points,
and removing the prior by 2.1–3.8. The drop holds in every per-dataset and CE-nCG cell as well
(Appendix D.1), which verifies the utility of each component separately.
BoundaryMORPH exploits its knowledge of the context capacity.To test how the input context
capacitykimpacts our algorithm, we run BoundaryMORPH with varying values ofk(the optimized
value,k opt). However, instead of evaluating on the corresponding CE-nCG@k opt, we evaluate at
manycapacity valuesk eval, to see what happens when we evaluate at adifferentcapacity from what
we gave our method. As seen in Figure 3(a), for nearly all evaluatedk eval, the run that was given
keval=k optis the best of all runs. That is, BoundaryMORPH truly is optimizing fork opt, and
fundamentally exploits the final evaluation’s disregard for order within and outside the top-k.
7

Preprint. Under review.
10 25 50 100 200 500
keval0.800.850.900.951.00CE-nCG@keval
kopt=10kopt=100kopt=25kopt=200kopt=50kopt=500
kopt=keval(a)
rerank BAGEL MORPH0.00.20.40.60.81.0CE-nCG@500.840.87
17%
free0.92CE-scored
Never CE-scored(b)
BoundaryMORPH
CE-nCG@50 = 0.812
BAGEL
CE-nCG@50 = 0.675
SS
CE-nCG@50 = 0.790
(c)
Scored by CE (B=25) Final Output: top-k (k=50) Document Query
Figure 3: (a):the acquisition rule optimizes for the exact capacity it is given.We plot CE-nCG
at differentevaluatedcapacitiesk eval(x-axis), with one curve peroptimizedcapacityk opt(the one
given to the algorithm); when evaluated at the optimizedk,k eval=kopt(stars), performance peaks
there. (b):BoundaryMORPH winsbecauseit does not spend its budget measuring the obvious
top.Share of each method’s top-kCE-nCG total coming from documents it never scored (“free”),
atB=50,k=50. (c):Skipping the obvious top buys the budget to reach a second mode.One
query’s pool, plotted via t-SNE over document embeddings and contoured by CE relevance, with
each method’sscoredand finallyselecteddocuments marked. Because BoundaryMORPH does not
score documents obviously inside the top-k, it has budget left to explore, find a second mode, and
exploit it, while max-seeking baselines concentrate calls on the primary mode.
5.3 SOURCES OFIMPROVEMENT
Budget is spent at the boundary rather than on secure incumbents.To investigate how Bound-
aryMORPH achieves performance gains, we analyze how each method allocates its budget. Fig-
ure 3(b) reports for 3 methods: among the sum relevance in the final top-k, how much relevance
came fromCE-scoreddocuments vs.unscored. We see that BoundaryMORPH scores farfewerof
the documents it finally returns. It declines to spend calls on high-confidence, high-posterior doc-
uments whose membership is not in doubt, freeing budget for the boundary where a call can flip
the answer, whereas BAGEL spends far more calls at the top: seeking a global maximum. Purely
reranking with the CE by definition can only include documents that are scored. Figure 3(c) shows
one NeuCLIR query’sscoredandfinally selecteddocuments on its relevance contour: by avoiding
over-scoring the main mode, BoundaryMORPH is able to cross over and explore a secondary mode.
The advantage increases with query multimodality.Figure 3(c) suggests the advantage comes
from reaching secondary relevance modes, so we measure howmultimodaleach query is: whether
its relevant documents form one tight group or several distinct ones. Treating CE score as height
over an embedding kNN graph, we extract relevancepeakswith persistence-based clustering (Chazal
et al., 2013) and define a query’smultimodal massas the prominence-weighted relevance outside its
tallest peak (Appendix E.1). Figure 4 contrasts a multimodal query (light pollution), whose peaks
are distinct sub-topics, with a unimodal one (Amazon deforestation). On NeuCLIR, multimodal
mass predicts per-query advantage over BAGEL (Spearmanρ= +0.59,p= 3.5×10−9):Bound-
aryMORPH gains most where it must reach “across” valleys to separate modes.An LLM audit
confirms that peaks are semantically coherent groups of documents (Appendix E.1).
8

Preprint. Under review.
UnimodalQuery: I am interested in articles that
describe the effects of the Amazon rainforest deforestation.
Deforestation Statistics
and Trends
MultimodalQuery: How does light
pollution affect people?
General Health
and Environmental
ImpactsTaiwan Regulatory
Responses
Bedroom Lighting
and Disease RiskIndustrial Air
Pollution LawsuitsDark Sky Parks
and Stargazing
Tourism
Peak centroid
Query
Figure 4:Geometric peaks are genuinely distinct sub-topics.CE relevance contours over t-
SNE embedding space for amultimodaland aunimodalquery, with the extracted peaks and LLM-
generated peak labels. The multimodal query retains 5 peaks, each a distinct facet of the information
need rather than one blurred high-relevance region; the unimodal query (Amazon deforestation)
retains one. Peak extraction details are in Appendix E.1.
6 RELATEDWORK
Budgeted reranking.Adaptiveretrieval dynamically chooses what to score at query time. GAR
(MacAvaney et al., 2022), Quam (Rathee et al., 2025b), SlideGar (Rathee et al., 2025a), and
RGS (Xu & Chen, 2025) iteratively expand graph neighborhoods of high-scoring candidates, while
budget-aware cascades (Rashid et al., 2024; Li et al., 2026) and list truncation (Meng et al., 2024)
dynamically halt reranking over a fixed order. Crucially, all are value-seeking and cannot estimate
an unscored document. Closest in intent is AcuRank (Yoon et al., 2025), which maintains Bayesian
beliefs to resolve top-kmembership doubt. However, its beliefs update independently, meaning an
unscored document’s posterior never updates based on a scored neighbor’s. Methodologically clos-
est is BAGEL (Kim et al., 2026), a pool GP differing via deterministic warm-starting, a zero prior
mean, and a maximum-seeking acquisition (GP-UCB).
Budgeted Selection Goals.Level-set estimation samples where confidence intervals straddle a
known threshold (Bryan et al., 2005; Gotovos et al., 2013). Relatedly, active set ordering (Nguyen
et al., 2024) and LUCB-style bandits over independent arms (Kalyanakrishnan et al., 2012; Jiang
et al., 2017; Pony et al., 2026) resolve top-kmembership for a knownk. We observe that budgeted
retrieval is fundamentally an active set ordering problem.
Indexing-time methods and agentic retrieval.Orthogonal offline methods optimize the corpus
before querying via document expansion (Nogueira et al., 2019), reranker distillation (Schlatt et al.,
2025), or LLM-built graphs (Sarthi et al., 2024; Edge et al., 2025; Guti ´errez et al., 2024). These
changewhat the pool contains; we optimize which members reach the generator. Another orthog-
onal family uses LLMs to interleave search and generation (Khattab et al., 2023; 2024; Jiang et al.,
2023; Jeong et al., 2024), sometimes via reinforcement learning (Jin et al., 2025), which forms the
basis of deep-research agents (Huang et al., 2025; Coelho et al., 2025). Because these systems treat
the retriever as a standalone tool, our budgeted selector can trivially plug into their loops to raise
accuracy and cut search calls (Sharifymoghaddam & Lin, 2026; Meng et al., 2026).
7 CONCLUSION
We introduced BoundaryMORPH, a budget-aware document selection method for diffuse Retrieval-
Augmented Generation. Rather than wasting limited cross-encoder calls on documents securely
within the context capacity, BoundaryMORPH uses a Gaussian Process with a MIPS prior to ex-
plicitly target the top-kdecision boundary. This approach achieves state-of-the-art selection quality
9

Preprint. Under review.
over existing maximum-seeking baselines. By avoiding redundant measurements of primary topics,
it successfully navigates multimodal queries to capture diverse semantic peaks, making it exception-
ally well-suited for complex, deep-research applications. While BoundaryMORPH offers significant
advantages for diffuse retrieval, we acknowledge certain system and algorithmic limitations. Specif-
ically, the method currently relies on sequential execution, assumes a discrete document capacityk
rather than strict token limits, and scales cubically with extreme budgets. We discuss these limita-
tions and potential future directions in detail in Appendix A.
AIUSE STATEMENT
In this work, we used generative AI tools for polishing writing, including shortening and feedback.
All text was manually written or reviewed. Additionally, AI tools were used for literature discov-
ery and summarization. All cited works were manually curated with Zotero, and checked for the
individual claims for which they were cited. AI tools were also used for coding, but artifacts were
manually verified and tested. We have reviewed all AI-assisted work. We take responsibility for the
final content of this work, including text, claims or artifacts produced with the aid of generative AI.
REPRODUCIBILITY STATEMENT
Appendix B contains all details necessary for reproducing our method (including pseudocode), while
Appendix C contains all details necessary to replicate our experiments, including dataset processing
details, baseline details, and prompts. We also include with this submission an anonymized codebase
implementing our method and the baselines.
REFERENCES
Marcia J. Bates. The design of browsing and berrypicking techniques for the online search interface.
Online Review, 13(5):407–424, May 1989. ISSN 0309-314X. doi: 10.1108/eb024320. URL
https://doi.org/10.1108/eb024320.
Brent Bryan, Robert C. Nichol, Christopher R Genovese, Jeff Schneider, Christopher J.
Miller, and Larry Wasserman. Active Learning For Identifying Function Threshold Bound-
aries. InAdvances in Neural Information Processing Systems, volume 18. MIT Press,
2005. URLhttps://proceedings.neurips.cc/paper/2005/hash/8e9304969
27757aac0dbd2438cb3f4f6-Abstract.html.
Fr´ed´eric Chazal, Leonidas J. Guibas, Steve Y . Oudot, and Primoz Skraba. Persistence-Based Clus-
tering in Riemannian Manifolds.Journal of the ACM (JACM), 60(6):41:1–41:38, November 2013.
ISSN 0004-5411. doi: 10.1145/2535927. URLhttps://dl.acm.org/doi/10.1145/25
35927.
Hung-Ting Chen, Xiang Liu, Shauli Ravfogel, and Eunsol Choi. Beyond Single Embeddings: Cap-
turing Diverse Targets with Multi-Query Retrieval, 2025. URLhttps://arxiv.org/abs/
2511.02770. Version Number: 1.
Charles L.A. Clarke, Maheedhar Kolla, Gordon V . Cormack, Olga Vechtomova, Azin Ashkan, Ste-
fan B ¨uttcher, and Ian MacKinnon. Novelty and diversity in information retrieval evaluation. In
Proceedings of the 31st annual international ACM SIGIR conference on Research and develop-
ment in information retrieval, SIGIR ’08, pp. 659–666, New York, NY , USA, July 2008. Asso-
ciation for Computing Machinery. ISBN 978-1-60558-164-4. doi: 10.1145/1390334.1390446.
URLhttps://dl.acm.org/doi/10.1145/1390334.1390446.
Jo˜ao Coelho, Jingjie Ning, Jingyuan He, Kangrui Mao, Abhijay Paladugu, Pranav Setlur, Jiahe Jin,
Jamie Callan, Jo ˜ao Magalh ˜aes, Bruno Martins, and Chenyan Xiong. DeepResearchGym: A Free,
Transparent, and Reproducible Evaluation Sandbox for Deep Research, December 2025. URL
http://arxiv.org/abs/2505.19253. arXiv:2505.19253 [cs.IR].
Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, Simone Filice, Cesare Campagnano,
Yoelle Maarek, Nicola Tonellotto, and Fabrizio Silvestri. The Power of Noise: Redefining Re-
trieval for RAG Systems. InProceedings of the 47th International ACM SIGIR Conference on
10

Preprint. Under review.
Research and Development in Information Retrieval, pp. 719–729, July 2024. doi: 10.1145/3626
772.3657834. URLhttp://arxiv.org/abs/2401.14887. arXiv:2401.14887 [cs.IR].
Florin Cuconasu, Simone Filice, Guy Horowitz, Yoelle Maarek, and Fabrizio Silvestri. Do RAG
Systems Really Suffer From Positional Bias?, October 2025. URLhttp://arxiv.org/
abs/2505.15561. arXiv:2505.15561 [cs.CL].
Josip Djolonga, Andreas Krause, and V olkan Cevher. High-Dimensional Gaussian Process Ban-
dits. InAdvances in Neural Information Processing Systems, volume 26. Curran Associates,
Inc., 2013. URLhttps://proceedings.neurips.cc/paper_files/paper/2013
/hash/8d34201a5b85900908db6cae92723617-Abstract.html.
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From Local to Global:
A Graph RAG Approach to Query-Focused Summarization, February 2025. URLhttp://
arxiv.org/abs/2404.16130. arXiv:2404.16130 [cs.CL].
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua,
and Qing Li. A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Lan-
guage Models. InProceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery
and Data Mining, KDD ’24, pp. 6491–6501, New York, NY , USA, August 2024. Association
for Computing Machinery. ISBN 979-8-4007-0490-1. doi: 10.1145/3637528.3671470. URL
https://dl.acm.org/doi/10.1145/3637528.3671470.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng
Wang, and Haofen Wang. Retrieval-Augmented Generation for Large Language Models: A
Survey, March 2024. URLhttp://arxiv.org/abs/2312.10997. arXiv:2312.10997
[cs.CL].
Alkis Gotovos, Nathalie Casati, Gregory Hitz, and Andreas Krause. Active learning for level set
estimation. InProceedings of the Twenty-Third international joint conference on Artificial In-
telligence, IJCAI ’13, pp. 1344–1350, Beijing, China, August 2013. AAAI Press. ISBN 978-1-
57735-633-2.
Bernal Jim ´enez Guti ´errez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hip-
poRAG: Neurobiologically Inspired Long-Term Memory for Large Language Mod-
els. InAdvances in Neural Information Processing Systems, volume 37, pp. 59532–
59569. Curran Associates, Inc., 2024. doi: 10.52202/079017-1902. URLhttps:
//proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d0
7ca4f319af96a3024f6dbd1-Abstract-Conference.html.
Kailash A. Hambarde and Hugo Proenc ¸a. Information Retrieval: Recent Advances and Beyond.
IEEE Access, 11:76581–76604, 2023. ISSN 2169-3536. doi: 10.1109/ACCESS.2023.3295776.
URLhttps://ieeexplore.ieee.org/abstract/document/10184013.
Yizheng Huang and Jimmy Xiangji Huang. A Survey on Retrieval-Augmented Text Generation for
Large Language Models.ACM Computing Surveys, 58(12):300:1–300:38, May 2026. ISSN 0360-
0300. doi: 10.1145/3805774. URLhttps://dl.acm.org/doi/10.1145/3805774.
Yuxuan Huang, Yihang Chen, Haozheng Zhang, Kang Li, Huichi Zhou, Meng Fang, Linyi Yang,
Xiaoguang Li, Lifeng Shang, Songcen Xu, Jianye Hao, Kun Shao, and Jun Wang. Deep Research
Agents: A Systematic Examination And Roadmap, September 2025. URLhttp://arxiv.
org/abs/2506.18096. arXiv:2506.18096 [cs.AI].
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong Park. Adaptive-RAG:
Learning to Adapt Retrieval-Augmented Large Language Models through Question Complex-
ity. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.),Proceedings of the 2024 Confer-
ence of the North American Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers), pp. 7036–7050, Mexico City, Mexico, June
2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.389. URL
https://aclanthology.org/2024.naacl-long.389/.
11

Preprint. Under review.
Haotian Jiang, Jian Li, and Mingda Qiao. Practical Algorithms for Best-K Identification
in Multi-Armed Bandits, May 2017. URLhttp://arxiv.org/abs/1705.06894.
arXiv:1705.06894 [cs.LG].
Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. Active Retrieval Augmented Generation. In Houda Bouamor,
Juan Pino, and Kalika Bali (eds.),Proceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing, pp. 7969–7992, Singapore, December 2023. Associa-
tion for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.495. URLhttps:
//aclanthology.org/2023.emnlp-main.495/.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani,
and Jiawei Han. Search-R1: Training LLMs to Reason and Leverage Search Engines with
Reinforcement Learning, August 2025. URLhttp://arxiv.org/abs/2503.09516.
arXiv:2503.09516 [cs.CL].
Kalervo J ¨arvelin and Jaana Kek ¨al¨ainen. Cumulated gain-based evaluation of IR techniques.ACM
Transactions on Information Systems (TOIS), 20(4):422–446, October 2002. ISSN 1046-8188.
doi: 10.1145/582415.582418. URLhttps://dl.acm.org/doi/10.1145/582415.5
82418.
Shivaram Kalyanakrishnan, Ambuj Tewari, Peter Auer, and Peter Stone. PAC Subset Selection in
Stochastic Multi-armed Bandits.
Shivaram Kalyanakrishnan, Ambuj Tewari, Peter Auer, and Peter Stone. PAC subset selection in
stochastic multi-armed bandits. InProceedings of the 29th International Coference on Interna-
tional Conference on Machine Learning, ICML’12, pp. 227–234, Madison, WI, USA, June 2012.
Omnipress. ISBN 978-1-4503-1285-1.
Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts,
and Matei Zaharia. Demonstrate-Search-Predict: Composing retrieval and language models for
knowledge-intensive NLP, January 2023. URLhttp://arxiv.org/abs/2212.14024.
arXiv:2212.14024 [cs.CL].
Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vard-
hamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei
Zaharia, and Christopher Potts. DSPy: Compiling Declarative Language Model Calls into Self-
Improving Pipelines. InThe Twelfth International Conference on Learning Representations,
2024.
Junyoung Kim, Anton Korikov, Jiazhou Liang, Justin Cui, Yifan Simon Liu, Qianfeng Wen, Mark
Zhao, and Scott Sanner. Bayesian Active Learning with Gaussian Processes Guided by LLM
Relevance Scoring for Dense Passage Retrieval, April 2026. URLhttp://arxiv.org/abs/
2604.17906. arXiv:2604.17906 [cs.IR].
Dawn Lawrie, James Mayfield, Eugene Yang, Andrew Yates, Sean MacAvaney, Ronak Pradeep,
Scott Miller, Paul McNamee, and Luca Soldani. NeuCLIRBench: A Modern Evaluation Collec-
tion for Monolingual, Cross-Language, and Multilingual Information Retrieval, November 2025.
URLhttp://arxiv.org/abs/2511.14758. arXiv:2511.14758 [cs.IR].
Minghan Li, Eric Gaussier, Juntao Li, and Guodong Zhou. EviRerank: Adaptive Evidence Con-
struction for Long-Document LLM Reranking, June 2026. URLhttp://arxiv.org/abs/
2411.06254. arXiv:2411.06254 [cs.IR] version: 6.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and
Percy Liang. Lost in the Middle: How Language Models Use Long Contexts.Transactions of the
Association for Computational Linguistics, 12:157–173, 2024. doi: 10.1162/tacl a00638. URL
https://aclanthology.org/2024.tacl-1.9/.
Yifan Simon Liu, Qianfeng Wen, Jiazhou Liang, Mark Zhao, Justin Cui, Anton Korikov, Armin
Toroghi, Junyoung Kim, and Scott Sanner. Multimodal Item Scoring for Natural Language
Recommendation via Gaussian Process Regression with LLM Relevance Judgments. In Maria
12

Preprint. Under review.
Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.),Findings of the Asso-
ciation for Computational Linguistics: ACL 2026, pp. 36859–36876, San Diego, California,
United States, July 2026. Association for Computational Linguistics. ISBN 979-8-89176-395-
1. doi: 10.18653/v1/2026.findings-acl.1836. URLhttps://aclanthology.org/2026
.findings-acl.1836/.
Maxime Louis, Herv ´e D´ejean, and St ´ephane Clinchant. PISCO: Pretty Simple Compression for
Retrieval-Augmented Generation. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and
Mohammad Taher Pilehvar (eds.),Findings of the Association for Computational Linguistics:
ACL 2025, pp. 15506–15521, Vienna, Austria, July 2025. Association for Computational Lin-
guistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.800. URLhttps:
//aclanthology.org/2025.findings-acl.800/.
Sean MacAvaney, Nicola Tonellotto, and Craig Macdonald. Adaptive Re-Ranking with a Corpus
Graph. InProceedings of the 31st ACM International Conference on Information & Knowledge
Management, pp. 1491–1500, October 2022. doi: 10.1145/3511808.3557231. URLhttp:
//arxiv.org/abs/2208.08942. arXiv:2208.08942 [cs.IR].
Gary Marchionini. Exploratory search: from finding to understanding.Communications of the ACM,
49(4):41–46, April 2006. ISSN 0001-0782. doi: 10.1145/1121949.1121979. URLhttps:
//dl.acm.org/doi/10.1145/1121949.1121979.
K.V . Mardia and P.E. Jupp.Directional Statistics. Wiley Series in Probability and Statistics. Wi-
ley, 2009. ISBN 978-0-470-31781-5. URLhttps://books.google.com/books?id=
PTNiCm4Q-M0C.
Chuan Meng, Negar Arabzadeh, Arian Askari, Mohammad Aliannejadi, and Maarten de Rijke.
Ranked List Truncation for Large Language Model-based Re-Ranking. InProceedings of the 47th
International ACM SIGIR Conference on Research and Development in Information Retrieval, pp.
141–151, July 2024. doi: 10.1145/3626772.3657864. URLhttp://arxiv.org/abs/2404
.18185. arXiv:2404.18185 [cs.IR].
Chuan Meng, Litu Ou, Sean MacAvaney, and Jeff Dalton. Revisiting Text Ranking in Deep
Research, June 2026. URLhttp://arxiv.org/abs/2602.21456. arXiv:2602.21456
[cs.IR].
Minerva Mukhopadhyay, Didong Li, and David B. Dunson. Estimating densities with nonlinear
support using Fisher-Gaussian kernels.Journal of the Royal Statistical Society Series B: Sta-
tistical Methodology, 82(5):1249–1271, December 2020. ISSN 1369-7412, 1467-9868. doi:
10.1111/rssb.12390. URLhttp://arxiv.org/abs/1907.05918. arXiv:1907.05918
[stat.ME].
Quoc Phong Nguyen, Sunil Gupta, Svetha Venkatesh, Bryan Kian Hsiang Low, and Patrick
Jaillet. Active Set Ordering. InAdvances in Neural Information Processing Systems, vol-
ume 37, pp. 108518–108541. Curran Associates, Inc., 2024. doi: 10.52202/079017-344
5. URLhttps://proceedings.neurips.cc/paper_files/paper/2024/hash/
c4546b4f9e1a44ed15c253dd43307dd5-Abstract-Conference.html.
Rodrigo Nogueira, Wei Yang, Jimmy Lin, and Kyunghyun Cho. Document Expansion by Query Pre-
diction, September 2019. URLhttp://arxiv.org/abs/1904.08375. arXiv:1904.08375
[cs.IR].
Nicholas Pipitone, Ghita Houir Alami, Advaith Avadhanam, Anton Kaminskyi, and Ashley Khoo.
zELO: ELO-inspired Training Method for Rerankers and Embedding Models, September 2025.
URLhttp://arxiv.org/abs/2509.12541. arXiv:2509.12541 [cs.AI].
Roi Pony, Adi Raz Goldfarb, Oshri Naparstek, Idan Friedman, Udi Barzelay, and Eli Schwartz.
Col-Bandit: Query-Time Top-$K$ Estimation for Late-Interaction Retrieval, June 2026. URL
http://arxiv.org/abs/2602.02827. arXiv:2602.02827 [cs.IR] version: 2.
Muhammad Shihab Rashid, Jannat Ara Meem, Yue Dong, and Vagelis Hristidis. EcoRank:
Budget-Constrained Text Re-ranking Using Large Language Models, May 2024. URLhttp:
//arxiv.org/abs/2402.10866. arXiv:2402.10866 [cs.CL].
13

Preprint. Under review.
Mandeep Rathee, Sean MacAvaney, and Avishek Anand. Guiding Retrieval using LLM-
based Listwise Rankers, January 2025a. URLhttp://arxiv.org/abs/2501.09186.
arXiv:2501.09186 [cs.IR].
Mandeep Rathee, Sean MacAvaney, and Avishek Anand. Quam: Adaptive Retrieval through
<u>Qu</u>ery<u>A</u>ffinity<u>M</u>odelling. InProceedings of the Eighteenth
ACM International Conference on Web Search and Data Mining, WSDM ’25, pp. 954–962, New
York, NY , USA, March 2025b. Association for Computing Machinery. ISBN 979-8-4007-1329-
3. doi: 10.1145/3701551.3703584. URLhttps://dl.acm.org/doi/10.1145/370155
1.3703584.
S. Robertson, S. Walker, S. Jones, M. Hancock-Beaulieu, and M. Gatford. Okapi at TREC-
3. 1994. URLhttps://www.semanticscholar.org/paper/Okapi-at-TREC-3
-Robertson-Walker/fa8ffa9f8d8367b987845ae88f2adcac2689f110.
Saron Samuel, Alexander Martin, Eugene Yang, Andrew Yates, Dawn Lawrie, Ian Soboroff, Laura
Dietz, and Benjamin Van Durme. Beyond Relevance: On the Relationship Between Retrieval
and RAG Information Coverage, June 2026. URLhttp://arxiv.org/abs/2603.08819.
arXiv:2603.08819 [cs.IR].
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Man-
ning. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval, January 2024.
URLhttp://arxiv.org/abs/2401.18059. arXiv:2401.18059 [cs.CL].
Ferdinand Schlatt, Maik Fr ¨obe, Harrisen Scells, Shengyao Zhuang, Bevan Koopman, Guido Zuccon,
Benno Stein, Martin Potthast, and Matthias Hagen. Rank-DistiLLM: Closing the Effectiveness
Gap Between Cross-Encoders and LLMs for Passage Re-Ranking. volume 15574, pp. 323–334.
2025. doi: 10.1007/978-3-031-88714-7 31. URLhttp://arxiv.org/abs/2405.079
20. arXiv:2405.07920 [cs.IR].
Sahel Sharifymoghaddam and Jimmy Lin. Rerank Before You Reason: Analyzing Reranking Trade-
offs through Effective Token Cost in Deep Search Agents. In Maria Liakata, Viviane P. Moreira,
Jiajun Zhang, and David Jurgens (eds.),Findings of the Association for Computational Linguis-
tics: ACL 2026, pp. 25874–25886, San Diego, California, United States, July 2026. Association
for Computational Linguistics. ISBN 979-8-89176-395-1. doi: 10.18653/v1/2026.findings-acl.
1289. URLhttps://aclanthology.org/2026.findings-acl.1289/.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke
Zettlemoyer, and Wen-tau Yih. REPLUG: Retrieval-Augmented Black-Box Language Mod-
els. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.),Proceedings of the 2024 Con-
ference of the North American Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers), pp. 8371–8384, Mexico City, Mexico, June
2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.463. URL
https://aclanthology.org/2024.naacl-long.463/.
Seokwon Song, Sohyeon Kim, and Gunhee Kim. MULTI3IR: A Benchmark for Multi-perspective
Multi-domain Multi-modal Information Retrieval, September 2026. URLhttps://arxiv.
org/abs/2608.30949. Version Number: 1.
Sofia Eleni Spatharioti, David Rothschild, Daniel G Goldstein, and Jake M Hofman. Effects of
LLM-based Search on Decision Making: Speed, Accuracy, and Overreliance. InProceedings
of the 2025 CHI Conference on Human Factors in Computing Systems, CHI ’25, pp. 1–15, New
York, NY , USA, April 2025. Association for Computing Machinery. ISBN 979-8-4007-1394-1.
doi: 10.1145/3706598.3714082. URLhttps://dl.acm.org/doi/10.1145/3706598.
3714082.
Niranjan Srinivas, Andreas Krause, Sham M. Kakade, and Matthias W. Seeger. Information-
Theoretic Regret Bounds for Gaussian Process Optimization in the Bandit Setting.IEEE Transac-
tions on Information Theory, 58(5):3250–3265, May 2012. ISSN 1557-9654. doi: 10.1109/TIT.
2011.2182033. URLhttps://ieeexplore.ieee.org/abstract/document/6138
914.
14

Preprint. Under review.
Siddharth Suri, Scott Counts, Leijie Wang, Chacha Chen, Mengting Wan, Tara Safavi, Jennifer
Neville, Chirag Shah, Ryen W. White, Reid Andersen, Georg Buscher, Sathish Manivannan, Nagu
Rangan, and Longqi Yang. The Use of Generative Search Engines for Knowledge Work and Com-
plex Tasks, March 2024. URLhttp://arxiv.org/abs/2404.04268. arXiv:2404.04268
[cs.IR].
Giovanni Trappolini, Florin Cuconasu, Simone Filice, Yoelle Maarek, and Fabrizio Silvestri. Re-
defining Retrieval Evaluation in the Era of LLMs, 2025. URLhttps://arxiv.org/abs/
2510.21440. Version Number: 1.
Ellen M. V oorhees. The TREC robust retrieval track.SIGIR Forum, 39(1):11–20, June 2005. ISSN
0163-5840. doi: 10.1145/1067268.1067272. URLhttps://doi.org/10.1145/106726
8.1067272.
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. Multilin-
gual E5 Text Embeddings: A Technical Report, February 2024. URLhttp://arxiv.org/
abs/2402.05672. arXiv:2402.05672 [cs.CL].
Qianfeng Wen, Yifan Liu, Joshua Zhang, George Saad, Anton Korikov, Yury Sambale, and Scott
Sanner. Elaborative Subtopic Query Reformulation for Broad and Indirect Queries in Travel
Destination Recommendation, October 2024. URLhttp://arxiv.org/abs/2410.015
98. arXiv:2410.01598 [cs.IR].
Christopher Williams and Carl Rasmussen. Gaussian Processes for Regression. InAd-
vances in Neural Information Processing Systems, volume 8. MIT Press, 1995. URL
https://proceedings.neurips.cc/paper_files/paper/1995/hash/7cce5
3cf90577442771720a370c3c723-Abstract.html.
Haike Xu and Tong Chen. Beyond Sequential Reranking: Reranker-Guided Search Improves Rea-
soning Intensive Retrieval, September 2025. URLhttp://arxiv.org/abs/2509.07163.
arXiv:2509.07163 [cs.IR].
Soyoung Yoon, Gyuwan Kim, Gyu-Hwung Cho, and Seung-won Hwang. AcuRank: Uncertainty-
Aware Adaptive Computation for Listwise Reranking, October 2025. URLhttp://arxiv.
org/abs/2505.18512. arXiv:2505.18512 [cs.IR].
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun
Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 Embed-
ding: Advancing Text Embedding and Reranking Through Foundation Models.arXiv preprint
arXiv:2506.05176, 2025.
Wayne Xin Zhao, Jing Liu, Ruiyang Ren, and Ji-Rong Wen. Dense Text Retrieval Based on Pre-
trained Language Models: A Survey.ACM Transactions on Information Systems, 42(4):89:1–
89:60, February 2024. ISSN 1046-8188. doi: 10.1145/3637870. URLhttps://dl.acm.
org/doi/10.1145/3637870.
A LIMITATIONS
While BoundaryMORPH successfully optimizes retrieval for diffuse queries under budget con-
straints, we identify several limitations and system assumptions that offer directions for future work:
Sequential execution and hardware utilization.Because BoundaryMORPH updates the GP pos-
terior after each cross-encoder call, it fundamentally requires sequential execution. As discussed in
Appendix D.4, while this is competitive on a single GPU, heavily parallelized production environ-
ments can amortize single-stage reranking via massive batching. Extending our boundary acquisi-
tion rule to select batches of documents simultaneously (e.g., auto-kriging) could improve hardware
utilization. In addition, many applications in practice are willing to trade off some latency for higher
quality retrieval (e.g., long deep research loops).
GP scaling at massive budgets.The exact GP posterior update scales cubically with the budget
and linearly with the pool size (O(B3N)). While our latency analysis (Appendix D.5) shows the
15

Preprint. Under review.
algorithm is lightweight for realistic RAG budgets (B≤200), the matrix inversion bottleneck be-
comes significant at extreme budgets (e.g.,>30s atB= 500). If CE inference becomes drastically
cheaper in the future, the relative overhead of the GP will increase. Future work could explore
algorithmic improvements, e.g. using rank-1 matrix updates.
Document counts vs. token limits.We formulate the RAG capacity as a strict document count
k, whereas real-world LLM context windows are strictly bounded bytokens. However, because
modern RAG pipelines heavily rely on text chunking prior to indexing, candidate documents typi-
cally fall into a relatively narrow and predictable token band. This system practice is what allows
the discrete document countkto serve as a highly realistic and practical proxy for context capac-
ity. Nonetheless, adapting the boundary gap for RAG systems that utilize variable-length chunks
remains an interesting future challenge.
Scope of queries.As we explicitly evaluate ondiffusequeries requiring many sources, our method
is tailored to exploratory search and deep-research agents. For simple factoid queries requiring only
a single, specific document (k= 1), the exploratory benefits of BoundaryMORPH do not apply, and
standard maximum-seeking approaches will suffice.
Selection versus within-context ordering.As defined in §2, BoundaryMORPH focuses purely on
setselection(whichkdocuments are retrieved). Because we actively avoid spending budget on the
secure top-k(Figure 3(b)), the exact relative ordering of those unscored documents relies entirely
on the GP estimation. However, we argue thatwhethera document is included in the RAG context
is fundamentally more important thanwhereit is placed. Recent research has demonstrated that
even with perfect relevance scores, sophisticated context-rearrangement strategies often fail to out-
perform random shuffling (Cuconasu et al., 2025). Therefore, determining the optimal arrangement
of context is a distinct, largely open research question, and we view maximizing the sheer volume
of relevant information delivered to the context as the primary bottleneck for diffuse queries.
B METHODDETAILS
B.1 ALGORITHM PSEUDOCODE
Algorithms 1–2 give full pseudocode for the procedure described in §3.3. As in §3.2, the GP is fit not
to the raw CE score but to theresidualagainst the MIPS prior, and that residual is Z-standardized
before conditioning and de-standardized after (Algorithm 1); Algorithm 2 is the outer acquisition
loop that calls it.
Algorithm 1POSTERIOR: residual GP conditioning
1:functionPOSTERIOR(observed setA, observed scoresy, prior meanµ 0, kernelK, poolC)
2:e←y−µ 0(A)▷residuals of the observations,r(d) =f CE(q, d)−µ 0(d), §3.2
3:z← 
e−mean(e)
/std(e)▷Z-standardize the residuals
4:(˜µ,˜σ2)←zero-mean GP posterior ofKgiven(A,z), for alld∈C ▷Eq. 2 withµ 0≡0
5:µ(d)←µ 0(d) + mean(e) + std(e)·˜µ(d)for alld∈C ▷de-standardize
6:σ2(d)←std(e)2·˜σ2(d)for alld∈C
7:return(µ, σ2)
8:end function
B.2 TIME COMPLEXITY
This section expands on the costs stated in §3.3’sComplexityparagraph. The kernelKis a Gram
matrix over the pool (augmented with the query row of §3.2), one entry per pair of documents, so
building it once before any CE call is spent costsO(N2).
Each of theBsteps of Algorithm 2’s loop re-conditions the GP on a growing observed setA t,
|At|=t. Rather than refactorizing from scratch, the Cholesky factor of thet×tobserved-block
kernel is updated incrementally atO(t2), but reading off the posterior mean and variance ateveryone
of theNpool documents requires a triangular solve against that factor per document, anO(t2)solve
repeatedNtimes, i.e.O(t2N)at stept. Summing over the run,PB
t=1O(t2N) =O(B3N), which
16

Preprint. Under review.
Algorithm 2BoundaryMORPH
Require:queryq; retrievers; cross-encoderf CE; budgetB; capacityk
Ensure:selected setS⊆C,|S|=k
1:C←top-Ncandidates froms(q)
2:Build kernelKoverC∪ {q}(§3.2); set prior meanµ 0(d)←MIPS percentile rank ofdinC
3:A← {q},y← {1}▷query enters as a zero-cost observation aty= 1
4:(µ, σ2)←POSTERIOR(A,y, µ 0, K, C)
5:fort= 1, . . . , Bdo
6:S t←top-kofCbyµ ▷incumbents; challengers areC\S t
7:(j∗, i∗)←arg minj∈S t, i∈C\S tµ(j)−µ(i)
σ(j)+σ(i)▷standardized boundary gap, Eq. 4
8:d←arg max 
σ(j∗), σ(i∗)
▷score whichever of the pair is less certain
9:y←f CE(q, d)▷one CE call
10:A←A∪ {d},y←y∪ {y}
11:(µ, σ2)←POSTERIOR(A,y, µ 0, K, C)
12:end for
13:returnthe top-kdocuments ofCranked byµ(scored documents at their true CE score, un-
scored documents atµ)
is the matrix-inversion cost the main text refers to. The acquisition rule (Eq. 4) is comparatively
cheap: it evaluates the standardized boundary gap over the full incumbent×challenger cross product,
O(k(N−k))per step andO(Bk(N−k))over the run. BecauseB, k≪Nin practice, both terms
are far below their worst case, which is why Appendix D.4 finds that theO(B)sequential CE calls,
not theO(B3N)GP bookkeeping, dominate wall-clock latency at the budgets used in this paper.
B.3 GPIMPLEMENTATION
The kernel is the vMF kernelK(d, d′) = exp(κ(cos(d, d′)−1))on unit-normalized embeddings,
with the diagonal forced to1. Its bandwidth is set per query by the median heuristic,κ= log 2/(1−
median i<jcos(d i, dj))over the pool, falling back toκ= 1if that denominator degenerates; no
other hyperparameter is tuned. The prior mean is the MIPS percentile rank withinC: because the
pool is returned MIPS-sorted, this is simplyµ 0= (N−1−pos)/(N−1)for a document at
positionpos. The query enters as an extra row of the kernel, which is therefore the augmented
(N+ 1)×(N+ 1)matrix with query entriesexp(κ(cos(q, d)−1)), and is conditioned on aty= 1
before the first CE call is spent.
Each posterior update takes the Cholesky factor of eKt=K(A t, At) +σ2
nI, with jitterσ2
n= 10−4
added to the observed-block diagonal, and performs two triangular solves: one forα= eK−1
t(yt−
µ0(At))and one forv= eK−1
tk(At, U)over the unobserved setU. The posterior is thenµ post=
µ0+k⊤αandσ2= diag(K) U−P
ikiUviU, with the variance clipped at0. If the factorization fails,
the jitter is escalated to10−2and, failing that, the solve falls back to a pseudo-inverse. As described
in §3.2, the quantity conditioned on is the residual againstµ 0, Z-standardized before conditioning
and de-standardized afterwards (Algorithm 1). When ranking the pool byµ, documents with equal
posterior means are ordered by MIPS percentile descending.
The resulting costs areO(N2)for the one-time kernel build,O(t2N)per posterior update and
O(B3N)over the run, andO(k(N−k))per acquisition step andO(Bk(N−k))over the run;
Appendix B.2 derives these.
B.4 KERNEL ABLATION
To ensure robustness of results with respect to the kernel used in the GP, we run both GP methods
(BoundaryMORPH and BAGEL) with three different kernels. We test the vMF kernel, described in
the Methods section, along with the squared-exponential (RBF) and Mat ´ern-5/2 kernels (Williams
& Rasmussen, 1995). Both alternatives are functions of the squared chordal distance∥d−d′∥2=
2 (1−cos(d, d′))on the unit sphere, a monotone reparametrization of the same cosine geometry the
17

Preprint. Under review.
Table 3: Kernel ablation:∆CE-nCG from replacing the vMF kernel (kernel−vMF, percentage
points), by dataset and evaluation cutoffk. Positive = the alternative kernel is better.†marks the
5/72 cells significant after multiple-comparison correction (see text).
∆CE-nCG@k(pp)∆nCG@k(pp, human qrels)
Method Kernel 50 100 200 50 100 200
NeuCLIR
MORPHRBF+0.12 +0.29−0.15 +0.01 +0.22 +0.03
MORPHMat ´ern-5/2+0.43 +0.08 +0.15 +1.23−0.36 +0.20
BAGEL RBF−0.61−0.71−1.33†−0.60−0.99−1.48†
BAGEL Mat ´ern-5/2−0.12 +0.28 +0.31 −0.55−0.79−0.79
Robust04
MORPHRBF+0.44 +0.19 +0.30 +0.76 +0.55 +0.08
MORPHMat ´ern-5/2+0.17 +0.38 +0.33 +0.44 +0.81 +0.11
BAGEL RBF+0.01−0.58†−0.74†+0.13−0.65−0.67
BAGEL Mat ´ern-5/2+0.44 +0.30 +0.23 +0.57 +0.44 +0.32
TravelDest
MORPHRBF+0.06−0.04−0.42 +0.81−0.33 +0.14
MORPHMat ´ern-5/2+0.10−0.13−0.09 −0.22 +0.17−0.25
BAGEL RBF+0.12−0.06−0.13 +0.22−0.32−0.26
BAGEL Mat ´ern-5/2+0.16 +0.39 +0.67†+0.46 +0.37 +0.47
vMF kernel uses. Writingu= 1−cos(d, d′)andρ=p
5u/ℓ2,
KRBF(d, d′) = exp 
−u/ℓ2
, K Mat(d, d′) =
1 +ρ+ρ2
3
e−ρ.(6)
The three differ only in how fast covariance decays with angle, and thus in the smoothness they
assume of the CE surface: vMF and RBF are infinitely differentiable, whereas Mat ´ern-5/2 is only
twice differentiable and so admits a rougher residual field. The length-scale is set by the same
median heuristic used forκin the vMF case,ℓ2= median i<j(1−cos(d i, dj)), so no kernel receives
a tuning advantage. Jitter, the query-as-observation anchor (which enters through the same kernel),
residual standardization, and the pool are unchanged.
We fix acquisition and prior for each of the two methods — so the only thing that varies is the kernel
— and show the change in performance from switching to RBF or Mat ´ern-5/2 over the vMF kernel
in Table 3. Both metrics are reported, since the two answer different questions (§4.2) and a kernel
could in principle help one and hurt the other. Each query’s score is averaged over the two budgets
B∈ {25,50}before the vMF score is subtracted, and the paired tests are run on those per-queryB-
averaged values. BAGEL’sβ 0is re-selected within each kernel, so neither alternative is handicapped
by a choice tuned for vMF.
The table shows that choice of kernel has no significant effect. Every one of the 72 deltas is under
1.5 pp and the median is 0.32 pp, and both the signs and the ordering of the two alternatives are
inconsistent across datasets: Mat ´ern-5/2 is nominally best for BoundaryMORPH on NeuCLIR at
CE-nCG@50 (+0.43pp) but RBF leads there on Robust04 (+0.44vs.+0.17pp), and on TravelDest
both drift slightly negative at the larger cutoffs. Across the 72 paired Wilcoxon signed-rank tests
behind the table, 13 reachp <0.05— against≈4expected by chance — and after Benjamini-
Hochberg correction5survive atq <0.05. All five are BAGEL, none our method: four are BAGEL
under RBF performingworsethan under vMF (NeuCLIR,−1.33pp CE-nCG@200 and−1.48
pp nCG@200; Robust04,−0.58pp CE-nCG@100 and−0.74pp CE-nCG@200), and the fifth is
Mat´ern-5/2helpingBAGEL on TravelDest (+0.67pp CE-nCG@200). So the only detectable kernel
effects move the baseline, in both directions, and none is large enough to matter: for scale, the same-
grid gap between the twomethodsat a fixed vMF kernel is+4.8to+10.4pp CE-nCG. Thesmallest
method effect is more than three times the largest kernel effect anywhere in the table, and fifteen
times the median one. Selection skill in this problem is governed by the acquisition rule and the
prior, not by the covariance shape. Hence, we report the paper’s main results using the vMF kernel.
18

Preprint. Under review.
Table 4: Active-testbed statistics: query counts, relevant documents per query (showing these
queries are genuinely diffuse), MIPS pool coverage, and document length, per dataset. Definitions
and the query filter are given below.
NeuCLIR Robust04 TravelDest
Full-corpus docs 10,038,768 528,155 131,268
Substrate pool / query 10,000 10,000 10,000
Substrate union docs 731,335 422,904 90,727
# Queries (original) 105 249 100
# Queries (post-filter) 85 187 99
# docs/query: median 51 62 462
# docs/query: mean 70.3 89.2 839.5
# docs/query: IQR (p25–p75) 37–85 35–116 140–1,062
# docs/query: max 246 448 5,667
Pool coverage: mean 0.92 0.88 0.70
Pool coverage: median 0.96 0.92 0.74
Pool coverage: p10 0.81 0.73 0.47
Pool coverage: min 0.58 0.42 0.29
Doc tokens: median 335 426 68
Doc tokens: IQR (p25–p75) 218–517 226–805 38–109
Table 5: Relevant documents per retained query, full percentiles.
Dataset min p10 p25 median mean p75 p90 max
NeuCLIR 22 27 37 51 70.3 85 122 246
Robust04 20 27 35 62 89.2 116 194 448
TravelDest 20 67 140 462 839.5 1,062 1,975 5,667
Table 6: Pool coverage, full percentiles. All columns are macro-averaged over queries except the
last, which is the micro-averaged coverage (pooled over all judged-relevant documents).
Dataset min p10 p25 median mean p75 p90 max micro
NeuCLIR 0.582 0.810 0.897 0.957 0.922 0.981 1.000 1.000 0.904
Robust04 0.422 0.729 0.831 0.917 0.885 0.975 1.000 1.000 0.860
TravelDest 0.287 0.466 0.607 0.737 0.699 0.821 0.882 0.943 0.599
C EXPERIMENTALSETUP
C.1 DATASET STATISTICS
NeuCLIRBench(Lawrie et al., 2025) pairs open-ended English information needs, each stated as
a keyword line followed by a question, against a newswire corpus machine-translated into English
from Persian, Russian and Chinese.
Robust04(V oorhees, 2005) uses TREC topics, each a description-style sentence, over the TREC
Disks 4 & 5 newswire and government collection.
TravelDest(Wen et al., 2024) uses broad travel-intent topics naming a desired quality of a destina-
tion rather than a place, over Wikivoyage-derived passages describing individual cities.
Relevance basis: NeuCLIR qrels grade>0(grades in{0,1,3}); Robust04 qrels grade in{1,2};
TravelDest an explicit binary relevant set. Pool coverage = fraction of a query’s judged-relevant
documents present in its top-10,000 MIPS pool, macro-averaged, denominator = all judged-relevant
documents in the full qrels, so1−coverage is relevant mass discarded before any selection and
bounds what any selector could recover. Token counts are Qwen3-Embedding-4B tokens over the
pool union. The≥20filter drops 20 / 62 / 1 queries.
Relevant documents in the qrels against relevant documents inside the pool: 5,976→5,400 (Neu-
CLIR), 16,684→14,351 (Robust04), 83,112→49,822 (TravelDest).
19

Preprint. Under review.
Table 7: Document tokens over the pool union, full percentiles.
Dataset min p25 median mean p90 p99 max
NeuCLIR 3 218 335 444.0 824 2,261 18,053
Robust04 15 226 426 639.1 1,264 3,299 927,351
TravelDest 6 38 68 81.4 157 289 918
C.2 DETAILED BASELINES
All baselines are scored on thesame fixed per-query top-10,000 poolas BoundaryMORPH, so
every comparison isolates allocation and ordering within a shared candidate set rather than retrieval
from scratch. Every baseline that has hyperparameters has themselected on the evaluation metric
(pooled-mean CE-nCG@50) per cell per budget, and reused acrossk; BoundaryMORPH runs at
one fixed configuration and tunes nothing.
BM25.Okapi BM25 withk 1= 1.5,b= 0.75and full-corpus statistics, applied as a re-ranking of
the shared pool and truncated atk. It makes no CE calls, so its ordering is fixed per query and its
value is identical at everyB; and its ordering uses neither MIPS nor the CE, so its nCG is a single
value per (dataset,k) and its CE-nCG depends only on which CE supplies the gold. Both properties
are verified rather than assumed.
SS (single-stage rerank).Score the MIPS top-Bwith the CE, place thoseBabove everything else,
and leave the tail in MIPS order. Fork≥Bthis returns exactly the MIPS top-k, so the budget buys
nothing in that regime.
BAGEL.Zero-prior GP-UCB. It shares our pool, our vMF kernel, and our median-heuristic band-
width, and differs in exactly three places: the prior mean (zero rather than MIPS percentile), the
acquisition rule (UCB over the whole pool rather than the boundary), and the warm-start.
•β0swept over{0,0.5,1,2}with square-root annealingβ t=β0/√t+ 1, selected per (dataset×
embedder×CE) cell perB.β 0= 0— pure exploitation of the posterior mean — is selected in 4
of the headline cells, all at largeB.
• BAGEL iswarm-started: it spends⌊B/2⌋calls deterministically on the MIPS top-⌊B/2⌋, leav-
ingB− ⌊B/2⌋acquisition steps.
RGS.Greedy graph expansion under a max-heap keyed by CE score: pop the highest-scored not-
yet-expanded document, fetch its graph neighbors, score any that are unscored, push them, and
truncate the heap toL s. The graph is a DiskANN cosine graph over the pool embeddings (one
greedy search + robust prune pass,L= 100,α= 1.2).
• Swept: graph degree, which is also the expansion branching factor,R∈ {10,32,64};
the seed/expansion splitn seeds = round(B/seed div); and the heap capL s=
max(n seeds,round(frac·B))withfrac∈ {0.2,0.5,1.0}. All headline cells selectR= 10,
seed div = 2,frac = 1.0. Theseed divgrid is{5,4,3,2}on the main configuration and{2,4,8}
on the model-swap configurations; the selected value, 2, is in both.
• Ranking convention: scored documents receiveoffset +CE score withoffset = 10and unscored
receive 0, so scored documents form a band strictly above unscored and a stable descending sort
leaves the unscored tail in MIPS order. This is the most charitable construction of the tail for a
method that cannot rank what it never scored.
• Atn seeds=Bthe search reduces to single-stage rerank to within10−12, which checks the imple-
mentation against its own degenerate case.
Agentic loop.A multi-hop search agent (DSP) run with plain instruction prompts, generated by
Qwen3.5-9B with thinking disabled and greedy decoding. Retrieval iswithin-pool, so the loop is
scored on the same candidate set as the selectors. Each hop the model writes one query from the
information need and its accumulated notes, that query is embedded with the same retriever and
its top documents by within-pool cosine are shown, and the model scores each document 0–10
and records new notes. Documents shown in earlier hops are excluded from later ones (meaning the
agent always scoresexactlydocs/hop new documents in each hop, andexactlydocs/hop×hops total
documents). The final ranking is surfaced documents by assigned score descending, ties broken by
20

Preprint. Under review.
cosine to the original query, followed by the never-surfaced tail in MIPS order. Grid: per-hop depth
∈ {5,10,20,50,100} ×hops∈ {1,2,3,4,6,8}.
The query-generation prompt is shown in Figure 5, and the note-extraction prompt in Figure 6.
You are a search agent gathering documents to satisfy an information need. Given the information need and the notes gathered so far, write ONE
new search query that will retrieve MORE relevant documents for the information need. This can mean exploring a different facet OR retrieving more
documents from facets already covered — do whatever will surface the most additional relevant documents. The query MUST be at least 100 words long
— write a long, detailed, keyword-rich query (at least 100 words), not a short phrase. Return ONLY the query text, no preamble, no quotes.
Claim:{claim}
Notes so far:{notes}
New search query:
Figure 5: Prompt for generating the next search query in the agentic retrieval loop (DSP baseline).
You are a search agent. Below is the information need, the notes gathered so far, and a set of newly retrieved documents (each labelled[DOC
<index>]). For EACH document, assign a relevance SCORE from 0 to 10 (0 = irrelevant to the information need, 10 = perfectly relevant). Then
extract any NEW facts from the relevant documents that help satisfy the information need.
Respond with STRICT JSON ONLY (no markdown fence, no prose) of the form:{"doc scores":{"0": 8, "3": 2}, "new notes":
["...", "..."]}Put doc scores FIRST. Use the EXACT integer indices shown in the[DOC <index>]labels as the keys, and an INTEGER 0-10
as each value. Score EVERY document shown. Keep each note to ONE short sentence.
Claim:{claim}
Notes so far:{notes}
Retrieved documents:{context}
JSON:
Figure 6: Prompt for scoring retrieved documents and extracting notes in the agentic retrieval loop
(DSP baseline).
Across all 29 configurations, NeuCLIR,n= 85, nCG@100 ranges from 0.554 (5 documents per
hop, 1 hop, 57.6 s) to 0.618 (100 per hop, 3 hops, 884.6 s), and latency from 57.6 s to 1,922.8 s.
BoundaryMORPH reaches nCG@100 0.621 in 25.7 s atB= 100,N= 10,000, so no configuration
of the loop matches it on either quality or cost.
D ADDITIONALRESULTS ANDROBUSTNESS
D.1 FULL ABLATION RESULTS
Table 8 breaks the ablation of §5.2 out by dataset and reports both nCG@kand CE-nCG@k, with
values averaged over the four embedder×CE pairs. Removing either component lowers every cell.
D.2 ROBUSTNESS TO RETRIEVER AND CROSS-ENCODER
Results are robust to the choice of retriever and cross-encoder.To test our method’s robustness
to embedding model and CE model, we use two different embedding models (Qwen3-Embedding-
4B (Zhang et al., 2025) and multilingual-e5-large (Wang et al., 2024)), and two different CE models
(Qwen3-Reranker-8B (Zhang et al., 2025) and zerank-2 (Pipitone et al., 2025)) and test all methods
with all 4 combinations. Figure 7 (Appendix D.3) shows the per-dataset result atB= 50,k= 100:
BoundaryMORPH maintains a gap over baselines with all embedder/CE combinations on every
dataset, indicating that the result is not a property of the retriever and CE.
Gains are largest with a weak retriever.Mean lift over SS runs from+3.4 ppwith a strong
retriever and strong CE to+14.0 ppwith a weaker retriever; the advantage grows as retriever weak-
ens, likely because there is simply more headroom between retriever and CE to take advantage of.
Though inabsoluteterms, all methods improve when switching from the weaker e5 embedder to the
stronger Qwen3 model, averaging+10.6nCG@100 (B=50) for SS, an+8.6for RGS, and+4.9for
BoundaryMORPH; a stronger prior universally helps.
21

Preprint. Under review.
Table 8: Full BoundaryMORPH ablations, per dataset and for both metrics (Table 2 in the main text
averages the nCG@kcolumns over datasets). Each row independently replaces one component:
boundary acquisition→standard UCB; MIPS prior→0 prior. We report the pp change from full
BoundaryMORPH. Both components are crucial for the gains, andalwaysimprove results.
nCG@k(%, human qrels) CE-nCG@k(%)
B=25B=50B=25B=50
Methodk50 100 200 50 100 200 50 100 200 50 100 200
NeuCLIR
MORPH51.2 56.4 64.4 54.8 59.7 66.6 84.6 80.8 78.8 89.9 85.3 82.4
w/o acq. -2.1 -2.1 -1.3 -1.2 -2.7 -1.5 -3.9 -5.2 -4.6 -2.6 -5.2 -5.1
w/o prior -2.3 -4.3 -5.5 -1.5 -3.0 -2.9 -2.9 -3.9 -4.4 -2.3 -2.3 -2.7
Robust04
MORPH46.4 48.2 55.1 51.4 52.0 57.7 69.8 67.9 68.1 76.1 72.3 71.1
w/o acq. -3.4 -3.5 -2.7 -2.0 -2.5 -1.5 -3.4 -3.6 -2.9 -2.2 -3.3 -2.7
w/o prior -3.5 -3.2 -4.1 -2.4 -2.2 -2.8 -4.1 -4.0 -4.1 -3.2 -3.1 -2.9
TravelDest
MORPH43.7 39.2 38.2 47.8 42.1 40.3 68.2 67.5 68.7 75.1 72.1 71.8
w/o acq. -2.2 -1.8 -2.2 -1.3 -1.6 -1.5 -3.0 -3.7 -3.9 -2.2 -3.3 -3.4
w/o prior -3.2 -1.8 -1.9 -2.5 -1.6 -1.4 -4.5 -4.1 -3.6 -4.5 -4.4 -3.6
Q3-Emb Q3-RankQ3-Emb ZerankE5-Emb Q3-RankE5-Emb Zerank-5051015 nCG@100 vs SS (pp)
NeuCLIR
Q3-Emb Q3-RankQ3-Emb ZerankE5-Emb Q3-RankE5-Emb ZerankRobust04
Q3-Emb Q3-RankQ3-Emb ZerankE5-Emb Q3-RankE5-Emb ZerankTravelDestRGS BAGEL MORPH
Figure 7:The advantage is robust to choice of retriever and cross-encoder.Per-dataset
nCG@100 lift over single-stage rerank (percentage points) atB= 50, for all four embedder×
CE combinations. BoundaryMORPH holds a gap over every baseline in all four, on every dataset.
D.3 PER-DATASET EMBEDDER×CERESULTS
nCG@100 lift over single-stage rerank (pp) atB= 50, per dataset, for all four embedder×CE
combinations. Embedders: Qwen3-Emb = Qwen3-Embedding-4B, e5 = multilingual-e5-large. CEs:
Qwen-Rerank = Qwen3-Reranker-8B, zerank = zerank-2.
D.4 LATENCYANALYSIS
Selectors that all call the same CE can be compared viaB. We also compare wall clock latency
(Figure 8, run on one A10, on NeuCLIR), and include an agentic retrieval loop in the comparison.
We use a plain multi-hop retrieve-and-read loop, DSP (Khattab et al., 2023), (Khattab et al., 2024).
The model (Qwen3.5-9B, running on 8 A10s) writes a search query, retrieves, reads and scores the
results, records notes, and uses its notes to write another query (and this is repeated). Prompts and
other details are in Appendix C.2 and Appendix D.5.
22

Preprint. Under review.
Table 9: Median per-query latency by component. NeuCLIR,N= 10,000, capacity 50,n= 85.
algorithmis the total non-CE time;GP updateandacquisitionare its components.
BCE CE % algorithm GP update acquisition
25 4.66 s 95% 0.22 s 0.06 s 0.16 s
50 9.43 s 95% 0.51 s 0.17 s 0.35 s
100 19.45 s 93% 1.46 s 0.63 s 0.85 s
200 38.89 s 90% 4.56 s 2.73 s 1.78 s
500 94.39 s 74% 32.86 s 27.86 s 4.79 s
Totals:4.88 s / 9.94 s / 20.91 s / 43.45 s / 127.25 s atB= 25/50/100/200/500.
While such results are highly dependent on document length and hardware, figure 8 shows that CE
budget scales each method’s latency, justifying the use ofBas a latency dial. BoundaryMORPH
nearly Pareto-dominates, except at highB(B >> k). AsBincreases, selection matters less; there
is budget enough to score every plausible document, so every reasonable allocator lands on a similar
set. Thus, most methods converge at highB.
10 20 50100 300test-time latency (s, log10log10)0.520.540.560.580.600.620.64nCG@100 (human qrels)
510
20 B=25B=500 up-left is better
MORPH
BAGEL
SSAgentic, 5 docs/hop
Agentic, 10 docs/hop
Agentic, 20 docs/hop
Figure 8:Budget is a realistic proxy for la-
tency.We show nCG@100 against wall-clock
latency per query on NeuCLIR, including an
agentic loop (DSP). Latency scales with budget
for all methods; with performance converging
at highB. The agentic method is dominated
due to decoding latency.Due to the GP, BoundaryMORPH and BAGEL
must be run sequentially, while SS can be batched.
In practice we find that on 1 GPU, the vari-
able document length leads to extra batch padding
whichincreaseslatency for SS at batch sizes
{2,3,8,16,32,64}, so we report results on batch
size of 1, which is the fastest possible setting for
SS. In theory, with enough GPUs/memory, SS can
be fully amortized, though based on the empirical
results for our datasets, real-time speed would im-
prove only from added GPUs, and not from batch-
ing on any given GPU. Thus, at least in theory, the
latency of SS could be improved in an exact linear
relationship with the number of GPUs.
Notably, despite8×parallel compute, the agen-
tic loop is Pareto-dominated. The whole agentic
family lies below the budgeted frontier at every
latency it reaches. Latency grows close to linearly
in hops, because each hop pays for autoregressive
decoding.
Origin of the gap.The information the GP ex-
ploits is already latent in the corpus embedding
geometry and propagates through the kernel to
documents never read. This agentic loop can only
carry forward what it wrote down in natural lan-
guage, in a bounded “notes”, and pays decoding
latency per token. We hypothesize that the gap is
explained by the GP being much more efficient at
propagating relevance information.
D.5 LATENCY SETUP
Hardware.One A10G (24 GB), bf16 with SDPA, left padding, maximum length 4,096. Document
sequence lengths are p50 = 426, p95 = 1,217 tokens. Absolute seconds are hardware-specific; the
ratios are the claim.
Per-component breakdown.NeuCLIR,N= 10,000, capacity 50,n= 85, medians (Table 9).
• CE dominates latency at everyB, 74–95%; the share is highest at smallB(95% atB= 25,50)
and falls only atB= 500(74%), where GP update cost grows enough to compete.
23

Preprint. Under review.
Table 10: Adjusted Rand index between the LLM’s two-group partition and the geometric peak-of-
origin labels, by slice.
Slicenmean ARI median ARI
All pairs 496 0.635 1.000
Pairs including the main peak 190 0.486 0.437
Secondary–secondary pairs 306 0.726 1.000
Per-query mean 65 0.593 0.624
• The algorithm’s own cost (GP update + acquisition) is under 1.5 s forB≤100and grows to 32.86
s atB= 500, driven almost entirely by the GP update.
• Per-call CE cost is∼0.19s, flat inB.
E QUERYMULTIMODALITY
Intuitively, diffuse queries can beunimodal, where relevant documents are relatively similar to one
another, ormultimodal, where relevant documents are relatively different from one another (Chen
et al., 2025; Song et al., 2026; Liu et al., 2026). We expect multimodal queries to be especially diffi-
cult for max-seeking selectors, which concentrate calls on the primary mode, and define a geometric
scalar,multimodal mass, to measure this.
E.1 PEAK EXTRACTION ANDLLMAUDIT
Peak extraction.We adapt the persistence-based clustering of Chazal et al. (2013), viewing each
document’s CE score as itsheightover the embedding kNN landscape. Apeakis a group of close,
highly relevant documents, and itsprominencemeasures how independent it is from other peaks:
how far relevance must drop before it merges into a taller one. Concretely, on the top-M= 1,000
documents by CE score within each query’sN= 10,000pool, build ak nn= 10OR-symmetrized
cosine kNN graph. Sweep a descending CE-score threshold; each connected component of the
super-level set carries a peak = its highest-scoring document, and merging components record 0-
dimensional persistence (merge-tree prominence). Simplifying atτ= 0.20yields a hard partition
of the top-Minto peaks. Multimodal mass is the sum of mass-weighted prominence over all non-
main peaks, so a unimodal landscape scores 0.
Peaks correspond to semantic distinctions.To test if peaks are semantically meaningful, we give
an LLM documents drawn from the same peak and from different peaks, and ask it to partition them,
scoring the result against the geometric partition by ARI:mean 0.635, median 1.000over496 pairs
from 65 multi-peak queries,85%above chance and63%clean at ARI≥0.5. The high success
indicates that geometric multimodality is also a semantic feature.
LLM audit.For each query, keep peaks with≥3member documents; for every unordered pair,
sample up to 8 documents from each peak’s 0.5-core (falling back to the full basin when the core has
<3), shuffle the≤16together, and ask the judge to split them into two groups byhowthey are rel-
evant. Score against the peak-of-origin labels by adjusted Rand index. One sample per pair, seeded
deterministically. Judge: Claude Opus 4.5, temperature 0, max 1,200 tokens, strict-JSON two-group
You are analyzing search results for a query. Below are{n}documents that are ALL relevant to the query. They are relevant in different ways — they
emphasize different aspects, sub-topics, entities, or angles of the query.
QUERY:{query}
Your task: partition ALL{n}documents into exactly TWO groups based on *how* each document is relevant to the query (what aspect/angle/sub-topic
of the query it addresses). The two groups need NOT be equal in size, but every document must be placed in exactly one group, and neither group may
be empty. Base the split on the substantive way each document relates to the query, not on surface features like length or style.
Give each group a SHORT description (one sentence) of the way its documents are relevant to the query.
DOCUMENTS:{docs}
Respond with ONLY a JSON object, no other text, in exactly this form:{"group 1":{"description": "<one sentence>", "docs":
[<doc numbers>]}, "group 2":{"description": "<one sentence>", "docs": [<doc numbers>]}}
Figure 9: Prompt for the LLM judge that partitions documents by relevance aspect (peak audit).
24

Preprint. Under review.
response; documents truncated to 1,500 characters. 496 pairs from 65 multi-peak NeuCLIR queries,
parse success 496/496.
ARI histogram: 32 below−0.05, 42 at chance[−0.05,0.05], 46 in(0.05,0.2], 40 in(0.2,0.4], 47
in(0.4,0.6], 40 in(0.6,0.8], 249 in(0.8,1]. 85.1% beat chance, 62.7% reach ARI≥0.5, and half
are separated perfectly. Correlation of ARI with pair size is−0.18.
The judge prompt is shown in Figure 9.
25
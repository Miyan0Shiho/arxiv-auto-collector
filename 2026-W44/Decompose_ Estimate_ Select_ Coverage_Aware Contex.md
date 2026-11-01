# Decompose, Estimate, Select: Coverage-Aware Context Construction for Long-Form RAG

**Authors**: He Tu, Aleena Bosco Kurumthottam, Torsten Suel

**Published**: 2026-10-29 15:51:46

**PDF URL**: [https://arxiv.org/pdf/2610.40314v1](https://arxiv.org/pdf/2610.40314v1)

## Abstract
Retrieval-augmented generation (RAG) grounds a language model's output in retrieved evidence, but for complex questions spanning multiple aspects of an information need, relevance-based retrieval may return redundant evidence while leaving some aspects unsupported. We propose a coverage-aware retrieval pipeline that decomposes queries into subquestions, ranks candidates with a coverage-trained bi-encoder, predicts a document\,$\times$\,subquestion coverage matrix with a cross-encoder, and selects documents by greedy maximum-marginal coverage. On TREC RAG 2024, our approach improves $α$-nDCG@10, which rewards newly covered nuggets and discounts redundant evidence, from $0.625$ to $0.702$, approaching the $0.718$ gold-nugget upper bound without sacrificing relevance. We then find that ground-truth oracle retrieval, despite substantially higher retrieval coverage, delivers no more nugget coverage to the answer than our estimated selection. This suggests that answer completeness depends not only on what evidence is retrieved, but also on how effectively the generator uses it. To address this, we introduce a \emph{grounded self-feedback loop} in which the model identifies underused evidence from its first answer and generates again without gold supervision, raising $V_{\mathrm{strict}}$, the fraction of vital reference nuggets fully supported by the answer, by $0.086$ over two passes while largely preserving faithfulness.

## Full Text


<!-- PDF content starts -->

Decompose, Estimate, Select: Coverage-Aware
Context Construction for Long-Form RAG
He Tu1, Aleena Bosco Kurumthottam2, and Torsten Suel2
1New York University Shanghai2New York University
Abstract.Retrieval-augmented generation (RAG) grounds a language
model’s output in retrieved evidence, but for complex questions spanning
multiple aspects of an information need, relevance-based retrieval may
return redundant evidence while leaving some aspects unsupported. We
propose a coverage-aware retrieval pipeline that decomposes queries into
subquestions, ranks candidates with a coverage-trained bi-encoder, pre-
dicts a document×subquestion coverage matrix with a cross-encoder,
and selects documents by greedy maximum-marginal coverage. On
TREC RAG 2024, our approach improvesα-nDCG@10, which rewards
newly covered nuggets and discounts redundant evidence, from0.625to
0.702, approaching the0.718gold-nugget upper bound without sacrific-
ing relevance. We then find that ground-truth oracle retrieval, despite
substantially higher retrieval coverage, delivers no more nugget cov-
erage to the answer than our estimated selection. This suggests that
answer completeness depends not only on what evidence is retrieved,
but also on how effectively the generator uses it. To address this, we
introduce agrounded self-feedback loopin which the model identifies
underused evidence from its first answer and generates again without
gold supervision, raisingV strict, the fraction of vital reference nuggets
fully supported by the answer, by0.086over two passes while largely
preserving faithfulness.
Keywords:Coverage-aware retrieval·Retrieval-augmented generation
·Nugget evaluation·Context selection·Iterative refinement
1 Introduction
Information retrieval has long centered on relevance: can a system find docu-
ments relevant to a user query? Decades of progress, from statistical ranking
to neural retrievers, have been measured with graded-relevance metrics such as
nDCG and MAP. Retrieval-augmented generation (RAG) inherits this frame-
work but changes the role of retrieved documents: rather than being returned
to a user, they serve as evidence for downstream generation. In long-form set-
tings [20,27,19], they therefore need to complement one another, not merely be
individually relevant.
A set assembled from independent relevance scores is usually highly redun-
dant: because documents are scored independently, ten documents elaborating
arXiv:2610.40314v1  [cs.IR]  29 Oct 2026

the same facet can score as well as ten that divide the query among them,
although only the latter provides evidence across all the nuggets a complete an-
swerrequires.OnTRECRAG2024,aselectorwithgolddocument–nuggetlabels
reaches0.858subtopic recall (StRecall) with five documents, the same value
it attains at ten, whereas a strong coverage-trained retriever reaches0.762with
ten. Relevance-based retrieval can therefore spend much of the context budget
on redundant evidence rather than expanding coverage.
Existing approaches address this in two ways, neither of which captures
the selection objective. Classical diversification penalizes documents resembling
those already selected, yet nugget-level redundancy need not correspond to em-
bedding similarity: a document can be relevant and dissimilar while still con-
tributing little new information. Training the retriever for coverage helps [12],
but a bi-encoder scores documents independently and cannot optimize a set-level
objective.
We therefore build a retrieval pipeline that makes coverage explicit. It
decomposes each query into subquestions with an LLM, ranks candidates
from a first-stage pool with a coverage-trained bi-encoder, predicts a docu-
ment×subquestion coverage matrix with a cross-encoder, and selects docu-
ments by greedy maximum-marginal coverage. Both models are trained on
document–subquestion pairs from the Researchy Questions dataset [27], using
LLM answerability judgments and the contrastive coverage objective of CoveR
[12]. On TREC RAG 2024 the pipeline raisesα-nDCG@10 from0.625to0.702,
approaching the0.718bound obtained when reference nuggets replace generated
subquestions, without sacrificing relevance.
Improving retrieval coverage raises a further question: how much of the re-
trieved evidence reaches the answer? Holding the generator, judge, candidate
pool, and context budget fixed while varying only document selection, we find
that the relationship is not monotone. A gold-oracle selector covers substantially
more nuggets in the retrieved set, yet delivers no more of them to the answer
thanourestimatedselection.Weaddressthiswithagroundedself-feedbackloop,
in which the model identifies underused evidence from its first answer and regen-
erates without gold supervision or additional retrieval, raising strict vital-nugget
coverage (V strict) by0.086over two passes while largely preserving faithfulness.
Contributions.Our main contributions are as follows:
–Acoverage-awareretrievalpipelinethatreplacesembedding-geometryproxies
for novelty with an explicitly predicted coverage matrix and greedy set-level
selection.
–A controlled downstream study showing that higher retrieval coverage need
not yield higher answer coverage.
–Agrounded self-feedback loopthat improves answer coverage without gold
supervision or additional retrieval.
In the remainder of this paper, we review related work (Section 2), describe our
data and evaluation metrics (Section 3), present the experimental framework
for the retrieval and generation settings (Section 4), report our main results
(Section 5), and discuss their implications (Section 6).

2 Related Work
In this section, we provide background and discuss the most relevant work.
2.1 Multi-faceted Questions and Query Decomposition
We study multi-faceted information needs, where a complete answer must syn-
thesize distinct facts across several facets of a query. This is common in report-
style requests [20,23], decompositional web research questions [27], and expert-
written questions requiring attributed long-form answers [19]. Unlike questions
centered on a single ambiguity [22,32] or with multiple valid answers [1], these
require one response covering many nuggets. Query decomposition is a common
strategy [4,26,28], but decomposition alone does not determine which documents
to retain across subquestions, whose candidate sets may overlap substantially.
2.2 Information Coverage
Selectingdocumentsforwhattheyaddtothosealreadychosenisalong-standing
idea in information retrieval, from subtopic retrieval [36] and maximal marginal
relevance [3] toα-nDCG [7] and aspect-based diversification [5,31], with some re-
cent neural approaches in [35,6]. Long-form RAG extends this setting to queries
with many subquestions and a context-window budget, where a generator con-
sumes the selected documents. CoveR [12] adapts the retrieval objective to this
setting, while other approaches optimize selection through document-set diver-
sity or redundancy penalties [16,14,9,13]. We adopt the retrieval objective of [12]
and instead focus on selection, predicting an explicit document×subquestion
coverage matrix rather than using embedding-based proxies for novelty.
2.3 Nugget-Based Evaluation
Nugget-based evaluation measures long-form answers by the atomic facts they
convey [20]. AutoNuggetizer [24,25] automates nugget construction and answer
assessment, while FActScore [21] applies atomic-fact decomposition to factual
precision and ICAT [29] evaluates coverage of diverse facts. CoverageBench [30]
further provides coverage-annotated collections across tasks and domains. We
adopt these established evaluation protocols and use their metrics to measure
answer coverage and faithfulness.
2.4 Context Utilization and Iterative Generation
Retrieved evidence is not necessarily used uniformly by the generator: perfor-
mance can depend on the position of information in a long context [17], degrade
as context grows [15], and vary non-monotonically with the amount of retrieved
evidence [11]. These findings motivate studying context utilization beyond re-
trieval quality. Iterative RAG systems typically address incomplete answers by

retrieving additional evidence [2,10,33,34] or revising the generated response
[18,28], while recent nugget-guided approaches combine generation with explicit
coverage signals [8,37]. In contrast, our loop keeps the retrieved set fixed and
asks the model to identify underused evidence before regenerating, isolating im-
provements due to context utilization.
3 Data and Evaluation Metrics
3.1 Training Data
Both trained components use subquestion answerability judgments rather than
relevance labels. We use the CRUX release of Researchy Questions [27], where
each query is paired with GPT-4-generated subquestions and a candidate doc-
ument pool. A Llama judge rates each document–subquestion pair from0to5.
Following [12], we treat ratings≥4as coverage, yielding a reference coverage
matrix from which both training sets are derived.
For the bi-encoder, we sample a partially covered stateCand label docu-
ments by the coverage they add:positiveif they cover at least one subquestion
outsideC,redundant negativeif they cover only subquestions inC, andirrele-
vant negativeotherwise. This yields1,929training examples over 847 queries. A
largerandmoreselectivelylabeledvariant(44,558examplesover20,000queries),
with covered states sampled along a simulated retrieval trajectory and positives
restricted to documents covering more than half of the still-uncovered subques-
tions, performsworse(α-nDCG@100.598versus0.656): at round zero, the cov-
ered set is empty, so the stricter threshold discards most moderately relevant
documents as neutral. We therefore use the binary variant throughout.
The coverage estimator is trained on850,000document–subquestion pairs
from3,000queries, using the same ratings≥4target. Approximately69%of
pairs have the majority label, which we use as the accuracy baseline.
3.2 Evaluation Collections
All evaluation queries are held out from training. We use two collections from
CoverageBench [30]: TREC RAG 2024 with56queries and CRUX-DUC04 with
50queries.
We exclude TREC Fair Ranking 2022 because its “nuggets” represent demo-
graphic attributes of authorship. On TREC RAG 2024, the nuggets are human-
written but document–nugget links are LLM-judged (approximately69%pre-
cision and90%recall against human assessment); CRUX-DUC04 uses LLM-
generated subquestions and links. The same judge is used for every system we
compare, on both collections.
3.3 Metrics
Retrieval.We useα-nDCG@10 [7] as the primary retrieval metric: it credits a
document for the nuggets it contributes and discounts nuggets already covered

higher in the ranking, so a ranking is rewarded for breadth rather than for
repeating the same nugget.StRecall@10, the fraction of a query’s distinct
nuggets covered anywhere in the top ten, is the coverage-only view that ignores
rank, and P@10 and nDCG@10 are reported to verify that coverage is not gained
at the cost of relevance.1
Generation.We follow the AutoNuggetizer protocol [24,25], which labels each
reference nugget as supported, partially supported, or unsupported. We report
Vstrict, the fraction of vital nuggets fully supported,AnswerStRecall, the
fraction of reference nuggets conveyed,Transfer rate, the fraction of nuggets
available in the retrieved context that reach the answer, andSentence precision,
the fraction of factual sentences supported by cited documents. Llama-3.3-70B
is used as the judge throughout.
4 Problem Definition and Experimental Setup
In this section, we define the problem and describe our pipeline and setup.
4.1 Problem Formulation
Following the nugget-based evaluation literature [20,24], anuggetis an atomic
fact that a complete response to a query should contain; a reference answer is
decomposedintoasetofthem,andasystemiscreditedforthenuggetsitsoutput
conveys. We say a documentcoversa nugget when it contains the information,
andananswerconveysanuggetwhenitstatesit;thedistinctionmattersbecause
our central question is how much of what the context covers the answer conveys.
Letqbe a query and letN(q) ={n 1, . . . , n m}be all the distinct nuggets a
complete response should convey. LetD qbe a first-stage document candidate
pool and letcov(d, n)∈ {0,1}indicate that documentdcoversnuggetn. Un-
der a context budgetk, coverage-aware retrieval selectsS⊆ D qwith|S|=k
maximizing
f(S) ={n∈ N(q) :∃d∈S,cov(d, n) = 1}.(1)
Two properties of Eq. (1) shape our design. First,fis defined oversets, which
independent bi-encoder scoress(q, d)cannot capture, motivating the explicit
selection stage. Second,fis monotone and submodular, so greedy selection by
marginal gain achieves a(1−1/e)-approximation of the optimum.
At inference timeN(q)andcovare both unavailable. Our pipeline estimates
them: an LLM produces subquestions ˜N(q)in place ofN(q), and a cross-encoder
1Table 1 averages over all56TREC RAG 2024 queries; the downstream experiments
use the54for which at least one selected document covers a reference nugget, the
other two scoring zero for every system. Every system therefore reads higher on the
54-query basis by a factor of56/54:StRecall@10 is0.762against0.791for the
coverage-trained retriever and0.858against0.888for oracle greedy. Comparisons
are always made within one basis.

Coverage-aware Retrieval
Grounded Second Pass
LLM decomposition
subquestions
Query
BM25
top-100
Coverage
bi-encoder
(re-rank top-100)
top-100
candidates
Coverage
estimator
(document × subquestion)
coverage
matrix
Greedy
selection
(set-level)
top-10
documents
 Generator
Answer
Self-critique
(identify underused
documents)
same documents
Fig. 1.The coverage-aware pipeline. An LLM decomposes the query into subques-
tions;acoverageestimatorpredictsadocument×subquestioncoveragematrixoverthe
BM25 top-100 pool re-ranked by a coverage-trained bi-encoder, and greedy maximum-
marginal-coverage selection fixes a context ofk= 10. The grounded self-feedback loop
re-generates from thatunchangedcontext using no reference information.
predicts ˆC∈[0,1]|Dq|×|˜N(q)|in place ofcov(Section 4.2). Figure 1 shows the
resulting system. Section 5.1 reports the cost of each substitution by replacing
it in turn with its reference counterpart.
4.2 Retrieval Pipeline
First-Stage Pool.Every system in this paper re-ranks the same BM25 top-100
pool, so differences reflect selection rather than first-stage recall.
Coverage-Trained Bi-Encoder.We initialize from a relevance-trained Modern-
BERT bi-encoder and continue training on the pairs of Section 3.1 with an In-
foNCEobjective(temperature0.02),usingasharedencoderwithsearch_query:
andsearch_document:prefixes, mean pooling, andL 2normalization. The an-
chor is the query paired with its simulated covered state, and the target is one of
its positive documents. The denominator contains two sources of negatives. The
first isn negdocuments sampled from the query’s own redundant and irrelevant
buckets, with redundant negatives included first so that the hardest negatives
are always present. The second is in-batch negatives: every other training ex-
ample in the batch contributes its own positive and its own sampled negatives,
and all of them are treated as negatives for the current anchor, so a batch of
bexamples supplies(b−1)(n neg+ 1)additional negatives per anchor at no
extra encoding cost. Redundant and irrelevant negatives are not distinguished
in the objective, so the bi-encoder learns whether a document adds coverage
rather than why it fails to. This stage reproduces the training objective of [12]
on public data and is not a contribution of our work; we include it so that the
selection stage is measured on top of a competitive coverage retriever rather
than a relevance-only one.
Subquestion Decomposition.We prompt an LLM for ten self-contained subques-
tions per query, each wrapped

in<q>tags, following [12].2
Holdingallothercomponentsfixed,wecomparethreedecomposers:Qwen2.5-
7B-Instruct,Llama-3.3-70B-Instruct,andGPT-4o.Wealsoevaluateadocument-
groundedvariant that derives subquestions from the top-ranked pool documents
rather than from the query alone.
Coverage Estimator.AModernBERTcross-encoderscoreseach(document,sub-
question) pair, trained on the distilled judgments of Section 3.1 with a maximum
input length of1024tokens. Applied to the pool it yields the matrix ˆC, which
we binarize at thresholdτ. We useτ= 0.3throughout unless otherwise stated;
Section 5.1 reports sensitivity to this threshold.
Greedy Selection.Starting fromS=∅and with the full subquestion set as un-
covered, we repeatedly add the document covering the most still-uncovered sub-
questions at thresholdτ, breaking ties by bi-encoder score, until|S|=k. Once
no candidate covers an uncovered subquestion the rule reduces to bi-encoder
order, so the output is always a complete ranking.
Reference Conditions.Two conditions isolate where the pipeline loses ground
to the formulation of Eq. (1).Gold nuggetsreplaces the generated subquestions
with the reference nugget text but still predicts coverage with the estimator, and
therefore bounds what a perfect decomposer would buy.Oracle greedyreplaces
ˆCwith the reference document-nugget matrix, and therefore bounds what any
selection method can achieve on this pool.
4.3 Generation Pipeline
Controlled Comparison.The following three conditions feed thesamegenerator
from thesamepool at thesamebudgetk= 10, assessed by thesamejudge.
Only document selection varies:
•ABi-encoder top-k, using relevance-ordered selection.
•BOur pipeline, using decomposition, coverage estimation, and greedy selec-
tion.
•OGreedy selection using the reference document-nugget matrix, providing
an upper bound on retrieval-based selection.
Document text is truncated to250words per document in all three conditions,
so context length does not confound the comparison.
Generation.The generator is Qwen2.5-7B-Instruct with greedy decoding and
a700-token limit. The prompt asks for a comprehensive answer in plain prose
covering as many distinct aspects as the documents support, consisting of about
twelve sentences, with every factual sentence ending in a citation to the docu-
ments it came from, and forbids outside knowledge.
2Write 10 diverse subquestions that can reveal the information required to answer
the given report query. Each subquestion should be self-contained and include the
necessary context. Collectively, the subquestions should fully cover the scope of the
report query.

Table 1.Retrieval results on TREC RAG 2024 (56queries, BM25 top-100 pool,k=
10). Each row adds one component to the row above. We useτ= 0.3;gold nuggets
uses reference nugget text in place of generated subquestions and is evaluated across
the same threshold range reported in Table 2.Oracle greedyreplaces the predicted
coverage matrix with the reference document–nugget matrix, without threshold.
Methodα-nDCG@10StRecall@10 P@10 nDCG@10
Relevance-trained bi-encoder0.625 0.744 0.754 0.784
+coverage training0.656 0.762 0.773 0.798
+coverage estimator and greedy selection0.7020.770 0.775 0.800
Gold nuggets (decomposition upper bound)0.718 0.781 0.771 0.802
Oracle greedy (selection upper bound)0.887 0.858 0.800 0.857
The Grounded Self-Feedback Loop.The second pass re-runs the generator over
theunchangedcontext of condition B, asking for about eighteen sentences rather
than the twelve requested in the first pass, under the same citation requirement.
Length alone is not what produces the gain: asking the first pass for eighteen
sentences instead of twelve leaves transfer rate unchanged (+0.000) and answer
StRecallmarginally lower (−0.003). We compare how the coverage gap is
identified:
•iter2injects the text of reference nuggets the first answer missed. This uses
the evaluation key and is reported only as a leakage upper bound, never as a
method.
•iter2gidentifies the missed reference nuggets, maps them to the in-context
documents that support them, and points the generator at those document
identifiers without revealing nugget text.
•iter2selfasks the generator to critique its own first answer and name the
documents it under-used, then points it at those documents. This uses no
reference information at any step and is the deployable configuration.
Thecontrastbetweeniter2andthelattertwoisolatesadesignquestion:whether
feedback shouldsupplythe missing content orredirect attentionto evidence
already present.
5 Main Results
We present the main experimental results, first for retrieval and then for gener-
ation.
5.1 Retrieval Results
Table 1 decomposes the gain. Coverage training the bi-encoder on the pairs of
Section 3.1 raisesα-nDCG@10 from0.625to0.656, and the explicit selection
stage adds a further0.046to reach0.702. The second step is larger, suggesting

Table 2.α-nDCG@10 on TREC RAG 2024 by decomposer and coverage thresholdτ.
All rows share the same bi-encoder, estimator and selection rule.
Subquestion sourceτ= 0.3τ= 0.5τ= 0.7
Qwen2.5-7B-Instruct0.675 0.666 0.663
Llama-3.3-70B-Instruct0.683 0.669 0.668
GPT-4o0.7020.677 0.672
GPT-4o, document-grounded0.672 0.669 0.653
Gold nuggets (upper bound)0.675 0.6900.718
that with the retriever held fixed, modeling coverage as a set-level selection prob-
lem contributes more than modeling it within the retriever’s training objective.
Two further properties of the gain matter as much as its size. First, it does
not come from relevance: P@10 and nDCG@10 are flat or slightly higher at every
step, so the selection stage is not trading away precision for diversity. Second,
the two upper bounds separate cleanly. Replacing our generated subquestions
with reference nugget text recovers0.016at its own best threshold, whereas
replacing the predicted coverage matrix with the reference one recovers a further
0.169. At this operating point, the remaining headroom lies primarily incoverage
estimation, namely how accurately a document can be judged to answer a given
nugget.
Decomposer Quality.Table 2 varies only the source of the subquestions. GPT-4o
scores highest at every threshold, followed by Llama-3.3-70B and Qwen2.5-7B,
but the returns diminish sharply: moving from a 7B to a frontier decomposer is
worth0.027, while the remaining distance to reference nuggets is0.016. Deriving
subquestions from retrieved documents instead of from the query is worse at
every threshold (0.672versus0.702atτ= 0.3), because grounding inherits the
pool’s errors and can only surface subquestions that some pooled document
already discusses. The reference row also prefers a strict threshold (0.718at
τ= 0.7) where generated subquestions prefer a loose one: reference nuggets are
precise and largely non-overlapping, whereas generated subquestions are broader
and partially redundant, so a strict criterion leaves too many of them uncovered
and the selection rule degenerates toward bi-encoder order.
5.2 Generation Results
Higher Retrieval Coverage Need Not Yield Higher Answer Coverage.Table 3
separates what retrieval covers from what the answer conveys. The oracle re-
trieves substantially more coverage than ours (0.888vs.0.799), yet does not
convey more of it to the answer: our selection is ahead in transfer rate (+0.050,
p=.042) and sentence precision (+0.070,p=.030), and ahead but not signif-
icantly on answerStRecall(+0.020,p=.32) andV strict(+0.029,p=.42);
after Holm–Bonferroni correction across the five comparisons none remains sig-

Table 3.Downstream results on TREC RAG 2024. A, B and O share the same gener-
ator, judge, candidate pool and context budget, differing only in document selection;
means are over the queries scorable under all three, withnthat count.∆is the paired
mean difference B−O (two-sided pairedt-test, uncorrected);†p < .05,‡p < .001. None
survives Holm–Bonferroni across the five metrics.
A B O
Metricnbi-encoder ours oracle∆(B−O)
StRecall@10 (retrieval)54 0.791 0.7990.888−0.089‡
Transfer rate54 0.2940.3260.275 +0.050†
AnswerStRecall54 0.2390.2660.247 +0.020
Vstrict 51 0.3500.3510.322 +0.029
Sentence precision530.7150.711 0.642 +0.070†
Table 4.Successive passes of the grounded self-feedback loop on TREC RAG 2024.
Every column uses the ten documents selected by condition B of Table 3: B is that
condition’s single answer, and+kpasses addskfurther generations over those same
documents, each preceded by a self-critique.∆compares+2 passes against B (two-
sided pairedt-test);†survives Holm–Bonferroni across the five metrics. Each table
pairs over its own conditions, so B’s means differ slightly from Table 3.
MetricnB+1 pass+2 passes+3 passes∆
StRecall@10 (retrieval)54 0.799 0.799 0.799 0.799 +0.000
Transfer rate54 0.326 0.3760.3910.363 +0.065†
AnswerStRecall54 0.266 0.2890.3080.295 +0.042†
Vstrict 53 0.338 0.4060.4240.421 +0.086†
Sentence precision540.7030.652 0.661 0.655−0.042
nificant. We read this as evidence that a large advantage in retrieved coverage
need not produce a more complete answer, not that our selection is superior.
Coverage-Aware Selection Does Not Consistently Improve Answer Coverage.
Compared with A, our selection gains+0.032in transfer rate and+0.027in an-
swerStRecallon TREC RAG 2024, neither significant (p=.22andp=.10).
On CRUX-DUC04 the direction reverses: B retrieves slightly more coverage than
A (StRecall@100.675versus0.668) but conveys less of it, with transfer rate
−0.037(p=.064) and answerStRecall−0.024(p=.041). Pooled over both
collections the difference in answerStRecallis+0.003(p=.78). We therefore
do not claim that coverage-aware selection consistently improves downstream
coverage over relevance-ordered selection.V strictis not reported across collec-
tions because CRUX nuggets carry no importance labels.
The Grounded Self-Feedback Loop Reaches the Answer.Table 4 applies the loop
to condition B’s context, so retrieval is identical by construction and all changes
are attributable to context use. Coverage improves for two passes and then
plateaus:V strictgains+0.068after the first pass and a further+0.018after the

second, reaching+0.086over B, with transfer rate and answerStRecallfol-
lowing the same pattern. After two additional passes, three of the five measures
remainsignificantunderHolm–Bonferronicorrectionacrossthatcolumn(answer
StRecalladjustedp=.023,V strict .029, and transfer rate.036). When cor-
recting instead across the three round comparisons forV strict, our pre-declared
primary metric, only the two-pass configuration remains significant (p=.022
versus.059for one pass). A third pass reverses part of the gain (transfer rate
−0.027, answerStRecall−0.013,V strict−0.002) and yields no significant re-
sult after correction. We therefore treat two passes as an empirical stopping
point rather than a budget constraint, although each pass requires an additional
generation over the full context.
Sentence precision drops by0.050after the first pass and then remains stable,
ending0.042below B after two passes and0.048below B after three (p=.210
and.144). Thus, the precision loss occurs at the first re-generation and does
not accumulate with further passes. A combinedF 1of answerStRecalland
sentence precision follows the same pattern, peaking after two passes (+0.033,
uncorrectedp=.037). Two passes therefore improve coverage without a net
faithfulnesscost,whereasthethirdretainstheprecisionlosswithoutthecoverage
gain.
Feedback Should Point at Evidence, Not Supply It.Replacing the self-critique
with the text of the missed reference nuggets (iter2) drives transfer rate to
0.771, but sentence precision falls from0.703to0.413: the generator restates
supplied text it cannot attribute to any document. Pointing instead at the in-
context documents that support those nuggets (iter2g) keeps faithfulness intact
and still gains+0.083inV strictover B, with sentence precision down only0.031
(p=.28). The model’s own critique (iter2self) recovers about80%of that
grounded gain (+0.068) using no reference information at any step. The distance
between iter2’s0.771and iter2g’s0.383transfer rate is the part of the coverage
gap that redirecting attention cannot close; we therefore reportiter2only as a
leakage bound.
6 Discussion
We now discuss the main conclusions from our work and its limitations.
More Retrieval Coverage May Not Improve Answers.The oracle selector (O)
increases retrieved coverage substantially and significantly, yet none of the four
downstream measures improves: additional evidence is not automatically useful
once it enters the context. Improving the coverage estimator may therefore raise
retrieval metrics without a corresponding downstream gain.
Context Utilization As Bottleneck.The oracle result suggests that maximizing
the diversity of evidence is not necessarily the right downstream objective. A
context containing more distinct nuggets will also tend to contain documents
that share fewer facts with one another, so within the same fixed answer length

the answer must move between unrelated points instead of developing a few of
them. This is consistent with prior evidence that language models use long and
densely packed contexts unevenly [17,15,11], although our experiments do not
isolate this mechanism.
Generalization and Evaluation Limitations.Our collections are small (56and50
queries). One pass of the loop replicates on CRUX-DUC04 (transfer rate+0.062,
p=.006; answerStRecall+0.039,p=.005), and pooled across the two
collections (n= 104) both gains survive Holm correction (+0.056and+0.030,
bothp=.007).OnCRUX-DUC04,thejudgefailedtoparse7of50answersforB
and13of50after the loop, so the gains persist despite more failures. The multi-
pass schedule in Table 4 was evaluated only on TREC RAG 2024, andV strict
cannotbepooledbecauseCRUX-DUC04hasnonuggetimportancelabels.Three
further caveats are thatτis chosen on the evaluation collection rather than a
held-out set, soτ= 0.3is optimistic; we ran no significance tests for Table 1,
so its ordering is indicative only; and our best configuration uses a proprietary
decomposer, although the fully open Llama-3.3-70B-Instruct costs only0.019
inα-nDCG@10 (0.683versus0.702), making a fully open pipeline available at
modest cost. Finally, nugget links are LLM-judged on both collections, so the
shared judge keeps comparisons internally consistent but absolute scores may
inherit its biases. The bi-encoder is also trained on only1,929examples, and
training-set size is not separated from the labeling rule that makes the larger
variant worse.
7 Conclusion
We now summarize our main conclusions. Making coverage explicit improves
retrieval. An estimated document×subquestion coverage matrix with greedy
maximum-marginal-coverage selection raisesα-nDCG@10 from0.625to0.702
on TREC RAG 2024 without sacrificing relevance, with more of the gain com-
ing from selection than from coverage training of the bi-encoder. Higher retrieval
coverage,however,doesnottranslatedirectlyintobetteranswers.Holdingevery-
thing but selection fixed, a gold-oracle selector with substantially higher retrieval
coverage conveys fewer nuggets to the answer than our estimated selection, with
downstream transfer remaining near0.30across the retrieval-coverage range we
tested. Answer coverage instead improves when the model is pointed back to un-
derused evidence through a grounded self-feedback loop over the same context.
For long-form RAG under a fixed context budget, these results favor improv-
ing evidence use over simply increasing retrieval coverage. Two questions remain
open. First, the remaining retrieval headroom lies largely in coverageestima-
tion: whether it can be closed with scale or requires supervision beyond distilled
answerability judgments is unclear. Second, if the generator limits answer com-
pleteness, selecting evidence for how usable it is, rather than only for what it
covers, is a natural next step.

References
1. Amouyal, S.J., Wolfson, T., Rubin, O., Yoran, O., Herzig, J., Berant, J.: QAM-
PARI: An Open-domain Question Answering Benchmark for Questions with
Many Answers from Multiple Paragraphs. arXiv preprint arXiv:2205.12665 (2022),
https://arxiv.org/abs/2205.12665v4
2. Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H.: Self-RAG: Learning to Retrieve,
Generate, and Critique through Self-Reflection. arXiv preprint arXiv:2310.11511
(2023),https://arxiv.org/abs/2310.11511v1
3. Carbonell, J., Goldstein, J.: The use of MMR, diversity-based reranking for re-
ordering documents and producing summaries. Proceedings of the 21st annual
international ACM SIGIR conference on Research and development in information
retrieval pp. 335–336 (1998).https://doi.org/10.1145/290941.291025
4. Chan, C.M., Xu, C., Yuan, R., Luo, H., Xue, W., Guo, Y., Fu, J.: RQ-RAG:
Learning to Refine Queries for Retrieval Augmented Generation. arXiv preprint
arXiv:2404.00610 (2024),https://arxiv.org/abs/2404.00610v1
5. Chapelle, O., Ji, S., Liao, C., Velipasaoglu, E., Lai, L., Wu, S.L.: Intent-based
diversification of web search results: metrics and algorithms. Information Retrieval
14(6), 572–592 (2011).https://doi.org/10.1007/s10791-011-9167-7
6. Chen, Y., Mao, J., Zhang, Y., Ma, D., Xia, L., Fan, J., Shi, D., Cheng, Z., Gu, S.,
Yin, D.: MA4DIV: Multi-Agent Reinforcement Learning for Search Result Diversi-
fication. arXiv preprint arXiv:2403.17421 (2024),https://arxiv.org/abs/2403.
17421v3
7. Clarke, C.L., Kolla, M., Cormack, G.V., Vechtomova, O., Ashkan, A., Büttcher,
S., MacKinnon, I.: Novelty and diversity in information retrieval evaluation. Pro-
ceedings of the 31st annual international ACM SIGIR conference on Research and
development in information retrieval pp. 659–666 (2008).https://doi.org/10.
1145/1390334.1390446
8. Dietz, L., Li, B., Liu, G., Ju, J.H., Yang, E., Lawrie, D., Walden, W., Mayfield, J.:
Incorporating Q&A Nuggets into Retrieval-Augmented Generation. arXiv preprint
arXiv:2601.13222 (2026),https://arxiv.org/abs/2601.13222v2
9. Guo,Y.,Wang,Y.,Wang,Y.:AdaGATE:AdaptiveGap-AwareToken-EfficientEv-
idence Assembly for Multi-Hop Retrieval-Augmented Generation. arXiv preprint
arXiv:2605.05245 (2026),https://arxiv.org/abs/2605.05245v1
10. Jiang, Z., Xu, F.F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu, J., Yang, Y.,
Callan, J., Neubig, G.: Active Retrieval Augmented Generation. arXiv preprint
arXiv:2305.06983 (2023),https://arxiv.org/abs/2305.06983v2
11. Jin, B., Yoon, J., Han, J., Arik, S.O.: Long-Context LLMs Meet RAG: Overcom-
ing Challenges for Long Inputs in RAG. arXiv preprint arXiv:2410.05983 (2024),
https://arxiv.org/abs/2410.05983v1
12. Ju, J.H., Yang, E., Adriaanse, T., Verberne, S., Yates, A.: Search for Coverage:
Learning Coverage-Aware Retrieval with Augmented Sub-Question Answerability.
arXiv preprint arXiv:2605.28522 (2026),https://arxiv.org/abs/2605.28522v1
13. Khurshid,A.,Sehgal,A.:StructureandDiversityAwareContextBubbleConstruc-
tion for Enterprise Retrieval Augmented Systems. arXiv preprint arXiv:2601.10681
(2026),https://arxiv.org/abs/2601.10681v1
14. Ko, T.W., Jiang, J.Y., Cheng, P.J.: Beyond Independent Passages: Adaptive Pas-
sage Combination Retrieval for Retrieval Augmented Open-Domain Question An-
swering. arXiv preprint arXiv:2507.04069 (2025),https://arxiv.org/abs/2507.
04069v1

15. Levy, M., Jacoby, A., Goldberg, Y.: Same Task, More Tokens: the Impact of Input
Length on the Reasoning Performance of Large Language Models. Proceedings of
the 62nd Annual Meeting of the Association for Computational Linguistics (Vol-
ume 1: Long Papers) pp. 15339–15353 (2024).https://doi.org/10.18653/v1/
2024.acl-long.818
16. Li, J., Hu, X., Wan, X.: SMART-RAG: Selection using Determinantal Matrices for
Augmented Retrieval. arXiv preprint arXiv:2409.13992 (2024),https://arxiv.
org/abs/2409.13992v1
17. Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., Liang,
P.: Lost in the Middle: How Language Models Use Long Contexts. arXiv preprint
arXiv:2307.03172 (2023),https://arxiv.org/abs/2307.03172v3
18. Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U.,
Dziri, N., Prabhumoye, S., Yang, Y., Gupta, S., Majumder, B.P., Hermann, K.,
Welleck, S., Yazdanbakhsh, A., Clark, P.: Self-Refine: Iterative Refinement with
Self-Feedback. arXiv preprint arXiv:2303.17651 (2023),https://arxiv.org/abs/
2303.17651v2
19. Malaviya, C., Lee, S., Chen, S., Sieber, E., Yatskar, M., Roth, D.: ExpertQA:
Expert-Curated Questions and Attributed Answers. Proceedings of the 2024 Con-
ference of the North American Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume 1: Long Papers) pp. 3025–3045
(2024).https://doi.org/10.18653/v1/2024.naacl-long.167
20. Mayfield, J., Yang, E., Lawrie, D., MacAvaney, S., McNamee, P., Oard, D.W., Sol-
daini, L., Soboroff, I., Weller, O., Kayi, E., Sanders, K., Mason, M., Hibbler, N.: On
the Evaluation of Machine-Generated Reports. arXiv preprint arXiv:2405.00982
(2024),https://arxiv.org/abs/2405.00982v2
21. Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W.t., Koh, P.W., Iyyer, M., Zettle-
moyer, L., Hajishirzi, H.: FActScore: Fine-grained Atomic Evaluation of Factual
Precision in Long Form Text Generation. arXiv preprint arXiv:2305.14251 (2023),
https://arxiv.org/abs/2305.14251v2
22. Min, S., Michael, J., Hajishirzi, H., Zettlemoyer, L.: AmbigQA: Answering Am-
biguous Open-domain Questions. Proceedings of the 2020 Conference on Empir-
ical Methods in Natural Language Processing (EMNLP) pp. 5783–5797 (2020).
https://doi.org/10.18653/v1/2020.emnlp-main.466
23. Pradeep, R., Thakur, N., Sharifymoghaddam, S., Zhang, E., Nguyen, R., Cam-
pos, D., Craswell, N., Lin, J.: Ragnarök: A Reusable RAG Framework and
Baselines for TREC 2024 Retrieval-Augmented Generation Track. arXiv preprint
arXiv:2406.16828 (2024),https://arxiv.org/abs/2406.16828v1
24. Pradeep, R., Thakur, N., Upadhyay, S., Campos, D., Craswell, N., Lin, J.: Initial
Nugget Evaluation Results for the TREC 2024 RAG Track with the AutoNugge-
tizer Framework. arXiv preprint arXiv:2411.09607 (2024),https://arxiv.org/
abs/2411.09607v1
25. Pradeep, R., Thakur, N., Upadhyay, S., Campos, D., Craswell, N., Soboroff, I.,
Dang, H.T., Lin, J.: The Great Nugget Recall: Automating Fact Extraction and
RAG Evaluation with Large Language Models. Proceedings of the 48th Inter-
national ACM SIGIR Conference on Research and Development in Information
Retrieval pp. 180–190 (2025).https://doi.org/10.1145/3726302.3730090
26. Press, O., Zhang, M., Min, S., Schmidt, L., Smith, N.A., Lewis, M.: Measuring
and Narrowing the Compositionality Gap in Language Models. arXiv preprint
arXiv:2210.03350 (2022),https://arxiv.org/abs/2210.03350v3

27. Rosset, C., Chung, H.L., Qin, G., Chau, E.C., Feng, Z., Awadallah, A., Neville, J.,
Rao, N.: Researchy Questions: A Dataset of Multi-Perspective, Decompositional
Questions for LLM Web Agents. arXiv preprint arXiv:2402.17896 (2024),https:
//arxiv.org/abs/2402.17896v1
28. Salemi, A., Samarinas, C., Zamani, H.: Plan-and-Refine: Diverse and Compre-
hensive Retrieval-Augmented Generation. arXiv preprint arXiv:2504.07794 (2025),
https://arxiv.org/abs/2504.07794v1
29. Samarinas, C., Krubner, A., Salemi, A., Kim, Y., Zamani, H.: Beyond Factual Ac-
curacy: Evaluating Coverage of Diverse Factual Information in Long-form Text
Generation. arXiv preprint arXiv:2501.03545 (2025),https://arxiv.org/abs/
2501.03545v4
30. Samuel, S., Yates, A., Lawrie, D., Soboroff, I., Adriaanse, T., Durme, B.V., Yang,
E.: CoverageBench: Evaluating Information Coverage across Tasks and Domains.
arXiv preprint arXiv:2603.20034 (2026),https://arxiv.org/abs/2603.20034v1
31. Santos, R.L., Macdonald, C., Ounis, I.: Exploiting query reformulations for web
search result diversification. Proceedings of the 19th international conference on
Worldwidewebpp.881–890(2010).https://doi.org/10.1145/1772690.1772780
32. Stelmakh, I., Luan, Y., Dhingra, B., Chang, M.W.: ASQA: Factoid Questions Meet
Long-Form Answers. arXiv preprint arXiv:2204.06092 (2022),https://arxiv.
org/abs/2204.06092v2
33. Trivedi, H., Balasubramanian, N., Khot, T., Sabharwal, A.: Interleaving Retrieval
with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions.
arXiv preprint arXiv:2212.10509 (2022),https://arxiv.org/abs/2212.10509v2
34. Yan, S.Q., Gu, J.C., Zhu, Y., Ling, Z.H.: Corrective Retrieval Augmented Gen-
eration. arXiv preprint arXiv:2401.15884 (2024),https://arxiv.org/abs/2401.
15884v3
35. Yu, P., Rahimi, R., Huang, Z., Allan, J.: Search Result Diversification Using Query
Aspects as Bottlenecks. Proceedings of the 32nd ACM International Conference
on Information and Knowledge Management pp. 3040–3051 (2023).https://doi.
org/10.1145/3583780.3615050
36. Zhai, C., Cohen, W.W., Lafferty, J.: Beyond Independent Relevance. ACM SIGIR
Forum49(1), 2–9 (2015).https://doi.org/10.1145/2795403.2795405
37. Łajewska, W., Balog, K.: GINGER: Grounded Information Nugget-Based Gener-
ation of Responses. arXiv preprint arXiv:2503.18174 (2025),https://arxiv.org/
abs/2503.18174v1
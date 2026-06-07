# Self-Conditioned Positional HNSW for Overlap-Aware Retrieval in Chunked-Document RAG Systems: Method and Industrial Evidence-Quality Audit

**Authors**: Nataraj Agaram Sundar, Tejas Morabia

**Published**: 2026-06-01 01:42:55

**PDF URL**: [https://arxiv.org/pdf/2606.01542v1](https://arxiv.org/pdf/2606.01542v1)

## Abstract
Chunked-document retrieval is a common component of retrieval-augmented generation (RAG) systems. Documents are split into overlapping chunks, embedded, and indexed with approximate nearest-neighbor search such as hierarchical navigable small world graphs (HNSW). Overlap improves boundary coverage but induces a practical failure mode: top-k retrieval often returns near-adjacent chunks that repeat evidence and waste prompt budget. We propose Self-Conditioned Positional HNSW (SCP-HNSW), a lightweight modification that appends a low-dimensional positional code to chunk embeddings and uses a two-pass query procedure to estimate and apply a query-specific document-position prior. SCP-HNSW leaves HNSW graph construction and traversal unchanged while adding an auditable minimum-index-gap selector for final context construction. We also integrate industrial review artifacts for generated evidence quality: a 770-review text-evidence audit with 318 fully labeled reviews and a 70-case OCR audit with 350 ratings. The text audit shows that 574 of 770 projected reviews are rated 3/5, only 39 fall in the 1-2 range, and narrative reviewer detail appears much more often than structured issue flags. The OCR audit shows slice-level pass rates from 95% for clean chat screenshots to 45% for handwritten/blurry captures, with moderate to strong agreement. These results motivate overlap-aware, audit-friendly RAG retrieval and identify the remaining controlled retrieval ablations needed for causal performance claims.

## Full Text


<!-- PDF content starts -->

Self-Conditioned Positional HNSW for Overlap-Aware Retrieval in
Chunked-Document RAG Systems: Method and Industrial
Evidence-Quality Audit
Nataraj Agaram Sundar Tejas Morabia
eBay Inc., San Jose, CA, USA
June 2, 2026
Abstract
Chunked-document retrieval is a common component of retrieval-augmented generation
(RAG) systems. Documents are split into overlapping chunks, embedded, and indexed with
approximate nearest-neighbor search such as hierarchical navigable small world graphs (HNSW).
Overlap improves boundary coverage but induces a practical failure mode: top- kretrieval often
returns near-adjacent chunks that repeat evidence and waste prompt budget. We propose
Self-Conditioned Positional HNSW (SCP-HNSW), a lightweight modification that appends a
low-dimensional positional code to chunk embeddings and uses a two-pass query procedure to
estimate and apply a query-specific document-position prior. SCP-HNSW leaves HNSW graph
construction and traversal unchanged while adding an auditable minimum-index-gap selector for
final context construction. We also integrate industrial review artifacts for generated evidence
quality: a 770-review text-evidence audit with 318 fully labeled reviews and a 70-case OCR audit
with 350 ratings. The text audit shows that 574 of 770 projected reviews are rated 3/5, only 39
fall in the 1–2 range, and narrative reviewer detail appears much more often than structured
issue flags. The OCR audit shows slice-level pass rates from 95% for clean chat screenshots
to 45% for handwritten/blurry captures, with moderate to strong agreement. These results
motivate overlap-aware, audit-friendly RAG retrieval and identify the remaining controlled
retrieval ablations needed for causal performance claims.
Keywords.HNSW; approximate nearest-neighbor search; retrieval-augmented generation; RAG;
chunking; positional encoding; evidence generation; human evaluation.
1 Introduction
Retrieval-augmented generation (RAG) systems retrieve evidence and present it to a language model
as grounded context [ 9]. In single-document question answering and evidence-generation workflows,
the system typically splits a document into overlapping chunks, embeds each chunk, builds a vector
index, and retrieves the top-ranked chunks for a user or workflow query. Overlap is useful because it
reduces boundary loss, but it creates a predictable retrieval artifact: adjacent chunks often contain
much of the same text and therefore occupy a dense neighborhood in embedding space.
Hierarchical Navigable Small World graphs (HNSW) are a standard choice for low-latency
approximate nearest-neighbor search [ 1]. However, HNSW solves nearest-neighbor search under
the supplied vector metric; it does not know that chunk IDs iandi+ 1 may be redundant because
of overlap. A semantic-only top- kquery can therefore return several near-adjacent chunks from
1arXiv:2606.01542v1  [cs.DC]  1 Jun 2026

Overlap-induced redundancy in top-k retrieval
document order relevant span
c0
c1c2
c3c4
c5c6
semantic top-k near-duplicate evidence
SCP-HNSW + gap coverage under same budget
Position-aware scoring plus an index-gap selector reduces repeated adjacent chunks before prompt construction.Figure 1: Overlap chunking creates near-adjacent candidate chunks with repeated text. A semantic
top-kquery can spend context budget on adjacent chunks, while SCP-HNSW combines a soft
positional query prior with a minimum index gap to improve coverage under the same budget.
a single local span, consuming scarce prompt budget without increasing evidence coverage. In
practice, this can surface as title drift, weak evidence summaries, and repeated support passages in
downstream RAG outputs.
We proposeSelf-Conditioned Positional HNSW(SCP-HNSW), a small retrieval modification
for chunked-document RAG. SCP-HNSW appends a two-dimensional positional code to each chunk
embedding, estimates a query-specific document-position prior from a first semantic pass, and uses
that prior in a second HNSW query. A deterministic minimum-index-gap selector then removes
redundant adjacent chunks before prompt construction. The core HNSW algorithm is unchanged:
graph construction, neighbor selection, and graph traversal operate over ordinary vectors.
This paper also integrates Responsible AI review results as industrial downstream evaluation
artifacts. The text-evidence review artifact summarizes 770 unique generated-evidence reviews, with
318 fully labeled reviewer exports from five reviewers. The OCR artifact summarizes a complete 70-
case, five-analyst slice with 350 ratings. These results do not by themselves prove that SCP-HNSW
outperforms semantic HNSW, because the artifacts do not include controlled per-query retrieval
logs for baseline versus treatment. They do, however, make the paper substantially stronger as an
industry-facing RAG evaluation paper: they show where generated evidence fails, how human review
signals are distributed, and what reliability measures are feasible for an audit-ready benchmark.
The paper makes four contributions:
1.It formulates overlap-aware retrieval as a relevance-coverage problem for chunked-document
RAG.
2.It defines SCP-HNSW, a positional augmentation and two-pass query procedure that preserves
standard HNSW implementation boundaries.
3.It adds a clean conference-style evaluation framing for downstream evidence quality, including
text evidence ratings, structured issue flags, narrative reviewer comments, OCR quality, and
inter-rater reliability.
4.It identifies the additional controlled ablations needed before making causal performance
claims for a full research submission.
2

2 Problem Setting
Let a documentDbe split into an ordered sequence of chunks
C= (c 0, c1, . . . , c N−1),(1)
where neighboring chunks may overlap. An embedding model fmaps each chunk to a dense vector
xi=f(ci)∈Rd. A query qtextis embedded as q=f(qtext) and HNSW returns an approximate
top-klist.
Overlap-aware retrieval should balance semantic relevance and coverage. One abstract objective
is
max
S⊆{0,...,N−1},|S|=kX
i∈Ssim(q, x i)−βX
i<j,i,j∈Sr(i, j),(2)
where r(i, j) is a redundancy penalty. In an overlap-chunked document, a simple structural penalty
isr(i, j) =I[|i−j|< g ] for a minimum chunk-index gap g. Semantic-only HNSW optimizes the first
term; SCP-HNSW exposes controls for the second term while keeping the ANN implementation
simple.
3 Self-Conditioned Positional HNSW
SCP-HNSW has three components: index-time positional vector augmentation, query-time self-
conditioning, and overlap-aware final selection.
3.1 Index-Time Positional Augmentation
Each chunk receives a normalized position
pi=i
max(1, N−1)∈[0,1].(3)
For linear documents, we use a half-cycle positional code:
ϕ(pi) =sin(πp i)
cos(πp i)
.(4)
The half-cycle avoids making the beginning and end of a document appear adjacent, which would
happen with a full periodic cycle. The stored vector is
exi= Normalizexi
λposϕ(pi)
∈Rd+2,(5)
whereλ pos≥0 controls the positional contribution.
3.2 Score Interpretation
Assumex i,q, andϕ(p) are unit-normalized. A second-pass query with inferred position priorµis
eq(µ) = Normalizeq
λposϕ(µ)
.(6)
3

Self-conditioned positional retrieval
standard HNSW index over
[xi;λposϕ(pi)]
query
q=f(qtext)pass 1
semantic querysoft position
prior μpass 2
q+ϕ(μ)gap selector
|i−j|≥gRAG context
non-redundant chunks
position prior is inferred from first-pass neighbors, not supplied by the user
HNSW graph construction/traversal is unchanged; overlap control happens through vector/query construction and the final selector.Figure 2: SCP-HNSW query flow. The first pass estimates a position prior from semantic hits; the
second pass uses the same HNSW index with a query vector that includes the inferred positional
code.
The augmented inner product decomposes as
eq(µ)⊤exi=q⊤xi+λ2
posϕ(µ)⊤ϕ(pi)
1 +λ2pos.(7)
For the half-cycle code, ϕ(µ)⊤ϕ(pi) =cos(π(pi−µ)). Thus, retrieval is biased toward chunks that
are both semantically relevant and close to the inferred region of the document. When λpos= 0,
SCP-HNSW reduces to semantic HNSW.
3.3 Two-Pass Self-Conditioning
The first pass runs with no positional component:
eq(1)= Normalizeq
02
, R(1)= ANN(eq(1), k1).(8)
For retrieved indicesi 1, . . . , i k1with scoress 1, . . . , s k1, a soft position prior is
wj=exp(τs j)Pk1
ℓ=1exp(τs ℓ), µ=k1X
j=1wjpij,(9)
whereτcontrols sharpness. The second pass runs
eq(2)= Normalizeq
λposϕ(µ)
, R(2)= ANN(eq(2), k2).(10)
The candidate list can be R(2)orR(1)∪R(2)re-ranked by semantic score, mixed score, or a
downstream reranker.
3.4 Overlap-Aware Final Selection
The final context set Sis selected greedily from the candidate list. A candidate index iis accepted
only if
min
j∈S|i−j| ≥g.(11)
4

Table 1: Implementation checklist for SCP-HNSW.
Component Required change
Chunk metadata Persist chunk indexi, total countN, and span metadata.
Embedding store Storeex i∈Rd+2instead ofx i∈Rd.
Index build Build ordinary HNSW over augmented vectors.
Query path Run pass 1, computeµ, run pass 2, merge candidates.
Selection Enforce minimum index gapgbefore prompt construction.
Monitoring Log selected indices, redundancy, recall proxy, and latency.
Evaluation artifacts used in this paper
text evidence review
770 reviews / 318 labeled
OCR evidence review
70 cases / 350 ratingsevidence
generation pipelinehuman
reviewquality
metrics
reliability
metrics
Supports downstream quality/calibration analysis; add controlled retrieval ablations for causal SCP-HNSW claims.
Figure 3: Evaluation artifacts used in this paper. The text-evidence review supports quality and
failure-mode analysis; the OCR review supports both quality and inter-rater reliability analysis.
The gap gis a transparent, auditable parameter that should be tied to chunk size and overlap. For
example, if chunk iandi+ 1 share large text overlap, setting g= 2 prevents immediate neighbors
while still permitting nearby supporting evidence.
3.5 Complexity and Implementation Overhead
SCP-HNSW adds two vector dimensions and one optional second HNSW query. For 768-dimensional
embeddings, two extra float dimensions increase vector dimensionality by approximately 0 .26%.
The main latency cost is the second ANN query. Because graph construction and traversal are
unmodified, the method can be implemented in systems that expose only vector insertion and vector
query APIs.
4 Industrial Evaluation Artifacts
The evaluation artifacts support a downstream evaluation section for a production-inspired evidence-
generation setting. Figure 3 summarizes how the artifacts fit into the paper. We report these
results as descriptive and reliability evidence, not as causal evidence that SCP-HNSW improves
over semantic HNSW.
5

Table 2: Evaluation artifacts incorporated in the paper.
Artifact Scale Main use in paper
Text evidence review 770 unique reviews; 318
fully labeledRating distribution, issue flags,
reviewer calibration, narrative
themes, title-template risk.
OCR evidence review 70 cases; five analysts;
350 ratingsOCR pass rates, pairwise weighted
κ, Fleiss’ κ, case-level failure sig-
natures.
4.1 Text Evidence Review Corpus
The text-evidence review artifact represents 770 unique reviews of generated dispute evidence.
A labeled subset of 318 reviews contains full fields from five reviewers, including 1–5 rating,
positive/negative judgment, dispute reason, 22 structured issue flags, reviewer details, title, summary,
and evidence excerpt. Since stable shared case identifiers were not preserved across reviewers, direct
inter-rater kappa for this text subset is not reproducible from the artifact alone. We therefore
use descriptive counts and reviewer-calibration summaries. Count-based results project observed
proportions from the 318 fully labeled reviews to the 770-review corpus; means and rates are
unchanged.
4.2 OCR Evidence Review Slice
The OCR artifact is a complete 70-case review slice with five analysts and 350 ratings. Fields
include text accuracy, word/line integrity, noise versus non-text confusion, overall OCR, evidence
utility, and pass/fail. The pass/fail definition is downstream acceptability for evidence use, with
Overall OCR ≥4 and evidence utility ≥3. Unlike the text corpus, this slice supports pairwise
weighted Cohen’sκfor ordinal OCR scores [10] and Fleiss’κacross multiple raters [11].
5 Results from the Evaluation Artifacts
This section converts the source review artifacts into paper-ready empirical results.
5.1 Text Evidence Quality
Figure 4 summarizes the rating distribution and the gap between structured issue flags and narrative
reviewer signal. Projected to the 770-review corpus, 574 reviews are rated 3/5, 39 are rated 1–2,
and 157 are rated 4–5. The projected corpus mean is 3.17. This distribution suggests a system with
many usable-but-not-excellent outputs rather than a system dominated by catastrophic failures.
Structured issue flags occur in about 77 reviews (10.1%), while narrative detail appears in about
574 reviews (74.5%). The strongest structured issues are title-related: title not aligned appears in
about 29 reviews (3.8%) and title inaccurate appears in about 27 reviews (3.5%). Other structured
flags remain much smaller: summary inaccurate appears in about 7 reviews, and buyer ID inaccurate,
evidence not aligned, inappropriate language, seller-as-buyer, and summary grammar issues each
appear in about 5 projected reviews. The main operational implication is that title-evidence
alignment is the dominant explicit failure mode.
Narrative comments add a different signal. Projected notes mention tool or pipeline concerns in
about 240 reviews (31.1%), usefulness or soft-quality concerns in about 177 (23.0%), resolution/return
6

1 2 3 4 5
Rating0100200300400500600Projected reviews
1227574
133
24Rating distribution (n=770)
Any issue
flagNarrative
detailsMissing
title/summary0100200300400500600Projected reviews
77574
19Structured vs narrative signalFigure 4: Text evidence review results projected to the 770-review corpus. Most reviews are rated
3/5. Narrative comments are much more prevalent than structured issue flags, implying that
flag-only dashboards would miss much of the actionable review signal.
evidence in about 140 (18.2%), title/reason mismatch in about 97 (12.6%), and translation or
truncation in about 17 (2.2%). This supports using narrative review mining alongside structured
flags for RAG evaluation.
The dispute mix is heavily skewed: about 753 reviews are SNAD, 12 are fraud, and 5 are
item-not-received. This matters for generalization. A model that performs well on this corpus may
still need separate validation on non-SNAD cases.
5.2 Reviewer Calibration and Sensitivity Split
Reviewer means show modest but non-trivial calibration differences. Projected reviewer allocation is
approximately 242 reviews each for three core reviewers and 22 reviews each for two pilot reviewers.
Mean ratings are approximately 3.37, 3.33, 3.11, 3.05, and 2.78 across the five anonymized reviewers,
around a corpus mean of 3.17.
The core/pilot sensitivity split is important. The core block projects to about 593 positive
reviews and 63 issue-flagged reviews, corresponding to 81.7% positive feedback and 8.7% issue flags.
The smaller pilot spot-check block projects to about 22 positive reviews and 15 issue-flagged reviews,
corresponding to 50.0% positive feedback and 33.3% issue flags. A paper should therefore report
core and pilot strata separately rather than mixing them into a single unqualified quality estimate.
5.3 Title-Template Risk
Recurring title templates expose a useful evaluation target for RAG systems. A “seller provided
return instructions” template appears in about 15 reviews with an estimated 33.3% title-issue rate,
or about 5 issue cases. A “buyer confirmed receipt” template appears in about 85 reviews with a
25.7% title-issue rate, or about 22 issue cases. Although the first template has the higher rate, the
second is the larger operational risk because it combines higher volume with a substantial issue rate.
This is directly relevant to overlap-aware retrieval: when retrieved evidence overemphasizes generic
delivery confirmation, the generated title can drift away from the actual SNAD-specific remedy or
dispute reason.
7

Clean
chatEmail/
pageMixed
UILabel/
receiptHandwritten/
blurry0.00.20.40.60.81.0Pass rate95%
85%
78%
60%
45%OCR pass rate by slice
Clean
chatEmail/
pageMixed
UILabel/
receiptHandwritten/
blurry0.00.10.20.30.40.50.60.7Fleiss' κ0.68
0.650.63
0.59
0.54Overall OCR agreement by sliceFigure 5: OCR evidence review results. OCR acceptability and inter-rater agreement both decline
as captures become noisier and structurally ambiguous.
Table 3: Case-level OCR exemplars from the 70-case slice.
Case Score card Interpretation
OCR-006, clean chat Overall OCR 4 .40±0.55; me-
dian 4; pass rate 1.00.Minor UI bleed is present, but the
core buyer/seller evidence remains
intelligible and useful.
OCR-049, label/receipt Overall OCR 1 .00±0.00; me-
dian 1; pass rate 0.00.Analyzer failure collapses image
handling and returns no com-
pelling evidence.
5.4 OCR Quality and Reliability
Figure 5 summarizes OCR pass rates and agreement by OCR slice. Pass rates decline from clean
chat screenshots (95%) and email/page captures (85%) to mixed-UI screenshots (78%), label/receipt
photos (60%), and handwritten/blurry captures (45%). Fleiss’ κfollows the same pattern: 0.68
for clean chat screenshots, 0.65 for email/page captures, 0.63 for mixed-UI screenshots, 0.59 for
label/receipt photos, and 0.54 for handwritten/blurry captures. The global weighted median of
per-case mean Overall OCR scores is 3.2.
Pairwise weighted κfor Overall OCR is moderate to strong. The highest pairwise alignments
are approximately 0.753, 0.741, and 0.722. The lower alignments are approximately 0.646, 0.658,
and 0.671. Binary pass/fail agreement is stronger than ordinal agreement, with pairwise κvalues in
the approximate 0.72–0.85 range. Disagreement concentrates in blur drop-out (mean case standard
deviation 0.91), cropped text loss (0.90), and UI bleed or initial capture (0.78). These results suggest
that the reviewer rubric is usable, but slice-specific guidance is needed for difficult image categories.
6 Implications for Conference-Quality Evaluation
The industrial results make the paper more complete as an applied RAG and evidence-quality study,
but a full research submission still needs controlled retrieval experiments. The available artifacts
show downstream quality, reviewer calibration, and OCR reliability. They do not include query-level
comparisons between semantic HNSW, HNSW plus post-filtering, MMR-style diversification, and
SCP-HNSW.
8

Table 4: Recommended controlled experiment grid for final conference submission.
Dimension Values to report
Retrievers Semantic HNSW; HNSW+gap; MMR; SCP-HNSW.
SCP parametersλ pos∈[0,0.6],g∈ {0,1,2,3},τ∈[2,20].
Candidate sizesk 1, k2∈ {k,2k,5k}, with final context budget fixed.
Retrieval metrics Recall@ k, nDCG, duplicate-token ratio, min index dis-
tance, coverage.
System metrics ANN latency, total RAG latency, prompt tokens, answer
utility.
A final conference-grade benchmark should add the following:
•Retrieval baselines:semantic HNSW, semantic HNSW plus gap filter, MMR/diversification
[6], and SCP-HNSW.
•Retrieval metrics:Recall@ k, MRR or nDCG, duplicate-token ratio, mean pairwise chunk
cosine, minimum index distance, and unique-token coverage.
•System metrics:retrieval latency, total answer latency, prompt tokens, and answer quality.
•Ablations: λpos,g,τ,k1,k2, position code choice, one-pass versus two-pass, and gap selector
on/off.
•Dataset transparency:number of documents, number of queries, chunk size, overlap,
embedding model, HNSW parameters, hardware, and whether documents are public or
internally cleared.
7 Related Work
HNSW is a graph-based approximate nearest-neighbor method with strong accuracy-latency trade-
offs [1]. Broader graph-based ANN methods are surveyed and compared by Wang et al. [ 2]. Hybrid
and filtered ANN methods incorporate structured predicates into vector search, including Filtered-
DiskANN [ 3], SeRF [ 4], and ACORN [ 5]. SCP-HNSW differs in treating document position as a
continuous soft retrieval signal rather than a hard filter.
Diversification and maximum marginal relevance balance relevance against novelty in search
results [ 6]. SCP-HNSW is complementary: it changes the query and indexed representation before
final selection, while the gap selector is a simple domain-specific redundancy rule. Positional
encodings are widely used to inject order information into neural models [ 7]. Pseudo-relevance
feedback adjusts a query using initial results [ 8]; SCP-HNSW adapts this idea to chunk positions
rather than term weights or dense semantic features.
The evaluation artifacts also connect to long-standing IR evaluation concerns: quality labels,
calibrated human review, and inter-rater agreement. The OCR slice’s use of weighted Cohen’s κ
and Fleiss’ κfollows established agreement analysis for ordinal and multi-rater judgments [ 10,11].
9

8 Limitations and Ethics
SCP-HNSW assumes that chunk order is meaningful. It can fail for multi-region queries, tables,
appendices, boilerplate-heavy documents, or documents where nearby positions are not semantically
related. A unimodal prior can over-focus if the correct answer requires evidence from distant regions.
A mixture prior, estimated by clustering first-pass hit positions, is a natural extension.
The industrial review results are descriptive and partially projected. The text review artifact
lacks stable shared case IDs across reviewers, so direct inter-rater kappa is not reproducible for
that portion. The OCR artifact is stronger for reliability analysis because each of 70 cases has five
ratings. Public submission should avoid raw user data, confidential case IDs, and proprietary system
details; examples in this draft are paraphrased and reviewer identities are anonymized.
9 Conclusion
We presented SCP-HNSW, a lightweight modification for overlap-aware retrieval in chunked-
document RAG systems. By appending a low-dimensional positional code to chunk embeddings
and using a self-conditioned two-pass query, SCP-HNSW makes redundancy controllable without
modifying HNSW graph construction or traversal. The added review artifacts convert the draft from
a method-only preprint into a stronger applied RAG evaluation paper: a 770-review text-evidence
audit identifies title alignment and narrative reviewer feedback as key signals, while a 70-case OCR
audit provides pass-rate and reliability measurements. The remaining work for a high-confidence
conference submission is a controlled retrieval benchmark that isolates SCP-HNSW against semantic
HNSW and diversification baselines.
References
[1]Y. A. Malkov and D. A. Yashunin. Efficient and robust approximate nearest neighbor search
using hierarchical navigable small world graphs.IEEE Transactions on Pattern Analysis and
Machine Intelligence, 42(4):824–836, 2020.
[2]M. Wang, X. Xu, Q. Yue, and Y. Wang. A comprehensive survey and experimental comparison
of graph-based approximate nearest neighbor search.Proceedings of the VLDB Endowment,
14(11):1964–1978, 2021.
[3]S. Gollapudi et al. Filtered-DiskANN: Graph algorithms for approximate nearest neighbor
search with filters. InProceedings of the ACM Web Conference, pages 3406–3416, 2023.
[4]C. Zuo, M. Qiao, W. Zhou, F. Li, and D. Deng. SeRF: Segment graph for range-filtering
approximate nearest neighbor search.Proceedings of the ACM on Management of Data,
2(1):69:1–69:26, 2024.
[5]L. Patel, P. Kraft, C. Guestrin, and M. Zaharia. ACORN: Performant and predicate-agnostic
search over vector embeddings and structured data.Proceedings of the ACM on Management
of Data, 2024.
[6]J. Carbonell and J. Goldstein. The use of MMR, diversity-based reranking for reordering
documents and producing summaries. InProceedings of the 21st Annual International ACM
SIGIR Conference on Research and Development in Information Retrieval, pages 335–336,
1998.
10

[7]A. Vaswani et al. Attention is all you need. InAdvances in Neural Information Processing
Systems, 2017.
[8]J. J. Rocchio. Relevance feedback in information retrieval. InThe SMART Retrieval System:
Experiments in Automatic Document Processing, pages 313–323. Prentice-Hall, 1971.
[9]P. Lewis et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. InAdvances
in Neural Information Processing Systems, 2020.
[10]J. Cohen. Weighted kappa: Nominal scale agreement with provision for scaled disagreement or
partial credit.Psychological Bulletin, 70(4):213–220, 1968.
[11]J. L. Fleiss. Measuring nominal scale agreement among many raters.Psychological Bulletin,
76(5):378–382, 1971.
11
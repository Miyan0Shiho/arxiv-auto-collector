# Top-K Is Not a Budget for Hybrid Retrieval

**Authors**: Chunran Zhang

**Published**: 2026-09-14 07:18:40

**PDF URL**: [https://arxiv.org/pdf/2609.15143v1](https://arxiv.org/pdf/2609.15143v1)

## Abstract
Modern hybrid retrieval for RAG typically fuses the Top-$L$ results from dense and sparse retrievers, but a fixed truncation depth may not transfer across changing queries and corpora. Exact fusion removes the dependence on a fixed depth, yet completing a specified Top-$K$ still incurs variable access costs. We present DiBud, which takes an access budget directly as input and incrementally certifies and returns an exact prefix of the RRF ranking over the full lists. Selective access increases certified output within the budget, while budgeted stopping bounds accesses per request. Experiments on five query sets reveal long-tailed costs for completing exact Top-20. At a budget of 2048 accesses, DiBud increases mean certified output within the first 100 positions by 7.86% over balanced access. After budget calibration for 95% quality retention, held-out queries retain 95.05%--97.68% of mean nDCG@20 while using 65.92%--99.53% fewer accesses than completing exact Top-20.

## Full Text


<!-- PDF content starts -->

TOP-K IS NOT A BUDGET FOR HYBRID RETRIEV AL
Chunran Zhang
School of Computing and Artificial Intelligence
Southwest Jiaotong University, Chengdu, China
chronis@my.swjtu.edu.cn
ABSTRACT
Modern hybrid retrieval for RAG typically fuses the Top-L
results from dense and sparse retrievers, but a fixed trunca-
tion depth may not transfer across changing queries and cor-
pora. Exact fusion removes the dependence on a fixed depth,
yet completing a specified Top-Kstill incurs variable access
costs. We present DiBud, which takes an access budget di-
rectly as input and incrementally certifies and returns an ex-
act prefix of the RRF ranking over the full lists. Selective
access increases certified output within the budget, while bud-
geted stopping bounds accesses per request. Experiments on
five query sets reveal long-tailed costs for completing exact
Top-20. At a budget of 2048 accesses, DiBud increases mean
certified output within the first 100 positions by 7.86% over
balanced access. After budget calibration for 95% quality
retention, held-out queries retain 95.05%–97.68% of mean
nDCG@20 while using 65.92%–99.53% fewer accesses than
completing exact Top-20.
Index Terms—Hybrid retrieval, reciprocal rank fusion,
budgeted retrieval, exact prefix, RAG
1. INTRODUCTION
Hybrid retrieval for retrieval-augmented generation (RAG)
commonly combines the topLresults from dense and sparse
retrievers using reciprocal rank fusion (RRF) [1, 2, 3]. A
fixed window limits the number of entries accessed but also
excludes contributions beyond that window. When a doc-
ument appears in one ranked list but not in the other list’s
window, fusion assigns zero to the latter contribution. Yet ab-
sence from the window only means that the rank has not been
observed; its contribution remains unknown. Replacing an
unknown contribution with zero can change the output order
even when the candidate set already contains every document
in the topKunder RRF over the full ranked lists. Thus,Lis
a retrieval-depth parameter that affects both retrieval quality
and access cost.
In practice,Lcan be selected using historical queries to
balance quality and cost. However, RAG systems serving
open-ended requests over evolving knowledge bases con-
tinually encounter new queries and updated corpora. Bothchanges alter the ranked lists and the positions of cross-
channel contributions, making it difficult for a previously se-
lectedLto preserve the original quality–cost tradeoff. Exact
rank aggregation maintains bounds on unobserved contribu-
tions and accesses the ranked lists until the required results
are determined [4]. EAHR applies this approach to hybrid re-
trieval to obtain the exact topKin order without a predefined
truncation depth [5]. Nevertheless, the number of accesses
required can vary sharply across queries: resolving competi-
tion at just a few positions can require accessing entries far
down the ranked lists.
Completing the same Top-Kcan require sharply differ-
ent numbers of accesses across queries. SpecifyingKthere-
fore does not provide a reliable bound on access cost. We
instead take an access budget directly as input, limiting the
total number of entries read from the two ranked lists. Within
this budget, the system certifies results incrementally. When
the budget is exhausted, it returns the certified prefix rather
than continuing until a fixed number of results is obtained.
We introduce DiBud (Direct Budgeting) for retrieval un-
der this direct access constraint. DiBud maintains bounds on
unread contributions and incrementally appends certified po-
sitions [6]. The output is an exact prefix: its documents and
their order match the RRF ranking over the full lists. When
the next position remains unresolved, selective access uses the
current competition to choose which list to advance [7]. Once
the budget is exhausted, DiBud returns the accumulated pre-
fix. Certification preserves the result semantics; the budget
independently bounds the accesses required to produce that
output. Selective reading aims to certify more results within
the same budget without relaxing this guarantee.
We evaluate DiBud through replays on five query sets
and five corpus snapshots. Fixed-depth experiments exam-
ine whether a selectedLtransfers across queries and corpus
updates. Complete exact Top-20 runs measure how strongly
access costs vary under the same output requirement. Equal-
budget comparisons with balanced access assess certified out-
put, while calibration on disjoint queries measures quality re-
tention and access savings relative to completing exact Top-
20. Together, these experiments test whether directly bound-
ing accesses can retain useful exact prefixes while avoiding
the long-tail cost of a fixed result count.
arXiv:2609.15143v1  [cs.IR]  14 Sep 2026

2. DIBUD: BUDGETED EXACT-PREFIX FUSION
Given a query, a corpus snapshot, and a read budgetB, DiBud
reads the dense and sparse rankings on demand and returns a
certified prefix of their complete RRF ranking. The budget
determines how much can be read; certification determines
how much can be returned (Fig. 1).
Dense123…
Sparse12……
Choose a channelReadUpdate bounds
& certifyExact pr efix
12…K
K not preset
Current competition
Blocker first; otherwise max ΔiBudget B
stop at B
Fig. 1. DiBud overview. The budget directly limits reads,
while current competition guides channel selection. Certifi-
cation determines the returned prefix.
2.1. Budget and output
Letr i(x)denote the one-based rank of documentxin channel
i∈ {D, S}, and letg(r) = 1/(c+r)for rank constantc≥0.
Setr i(x) =∞andg(∞) = 0ifxis outside that channel’s
support. The complete fusion score is
F(x) =X
i∈{D,S}g(ri(x)).(1)
Sorting the union of the channel supports by decreasingF(x),
with a fixed document-identifier order to break ties, defines
the complete sequenceT⋆.
Both channels supply resumable exact ranking prefixes
from the same snapshot. Ifd Dandd Sare their read depths
andAis the output sequence, the required contract is
dD+dS≤B, A=T⋆
1:K, K=|A|.(2)
HereBis the input limit;K=|A|is the certified output
size, not a preset count. Exactness applies to these returned
positions. If none is certified,Ais empty. Reading the same
document from both channels counts as two accesses.
2.2. Prefix certification
LetS ibe the documents already read from channeli. An
unread contribution in that channel is bounded by
ui=(
0,if confirmed exhausted,
g(di+ 1),otherwise.(3)For an observed documentx, the score bounds are
ℓ(x) =X
i:x∈S ig(ri(x)),(4)
h(x) =ℓ(x) +X
i:x/∈S iui.(5)
A document unseen in both channels has score at mosth ∅=
uD+uS.
Among observed documents not yet emitted, selectxwith
the highest lower bound, breaking ties by the fixed identifier
order. Certifyxas the next output when its lower bound
exceeds every remaining competitor’s upper bound andh ∅.
Equality with an observed competitor is allowed only when
xwins the identifier tie; comparison with the unseen bound
remains strict. Appendxand repeat until the next position
cannot be certified.
The bounds ensureℓ(x)≤F(x)≤h(x). Each emitted
document therefore precedes every remaining document, in-
cluding any not yet observed. Applying this argument at each
output position establishes the prefix equality in (2). Each po-
sition can be certified without resolving later positions, so the
prefix grows incrementally and remains valid when the bud-
get ends.
2.3. Reading and stopping
When the next position cannot be certified, letybe the ob-
served competitor with the highest upper bound, excluding
xand the emitted prefix and breaking ties by identifier. If
h(y)≥h ∅andyis missing a contribution from exactly one
channel, advance that channel. Reading may revealyand
determine its missing contribution. Otherwise, advancing the
read depth lowers its contribution bound. Both outcomes help
determine whetherystill blocks the next position.
Otherwise, advance the channel whose next batch offers
the larger reduction in the unread bound. For a batch ofb
entries in a channel that remains open after the read, this re-
duction is
∆i=g(d i+ 1)−g(d i+b+ 1).(6)
If exhaustion is established, the new bound is zero. Exhausted
channels are skipped, and equal reductions are resolved by
a fixed channel order (dense first). This rule also initializes
reading when no document has been observed.
After each batch, update the bounds and extend the certi-
fied prefix. The batch size is capped by the remaining budget.
Complete these updates after the final allowed read, then re-
turn the accumulated output when the budget is consumed or
both channels are exhausted. The schedule affects how many
positions can be certified within the budget; the certification
rule preserves the exactness of every returned position.

10 100 1k 5k
Truncation depth L0.610.620.630.640.650.660.670.680.69Mean nDCG@10
(a) Same corpus, different queries
DL 2019
DL 2020
10 100 1k 5k
Truncation depth L0.6250.6500.6750.7000.7250.7500.7750.800Mean nDCG@10
(b) Same queries, changing corpus
R1
R2R3
R4R5
1 2 3 4 5
Corpus snapshot−0.06−0.04−0.020.00Frozen L − full RRF
(mean nDCG@10)
(c) Transfer of Round-1 selectionFig. 2. Depth sensitivity and transfer. (a) Different queries on the same corpus. (b) The same queries across snapshots, using
each round’s judgments. Circles mark the best tested depths. (c) Held-out transfer of Round-1 depths. Error bars show 95%
nested bootstrap intervals with depth reselection and query identities preserved across rounds.
3. EXPERIMENTS
3.1. Setup
We evaluate 770 queries from five query sets—TREC-
DL 2019/2020, NFCorpus, SciFact, and TREC-COVID—
and the same 30 queries across five TREC-COVID snap-
shots [8, 9, 10, 11, 12]. We replay exact rankings from
bge-small-en-v1.5 [13] and BM25 [14] (k 1= 0.9,b= 0.4),
using equal-weight RRF with contributions1/(59 +r)and
deterministic tie breaking.
DiBud and Balanced use the same certification rules and
access budgets, ranging from 128 to 10000. Both read one
entry at a time; Balanced alternates between the two lists. We
count certified results up to 20 and 100, and use completed
exact Top-20 as the cost and quality reference.
Static replays use reconstructed full rankings: all 770
queries complete exact Top-20. Temporal replays use frozen
snapshot rankings without treating saved-list boundaries as
exhaustion. Results are averaged within each query set, then
equally across sets. Reported 95% confidence intervals use
2000 query bootstrap samples.
3.2. Transfer of fixed depths
We evaluateL∈ {10,20,50,100,200,500,1000,2000,
5000}. On the same MS MARCO corpus, increasingL
from 20 to 5000 raises nDCG@10 from 0.6538 to 0.6812
for TREC-DL 2019, but lowers it from 0.6424 to 0.6243 for
TREC-DL 2020 (Fig. 2a). Changing queries reverses the
benefit of a deeper window.
For the same 30 queries across five TREC-COVID snap-
shots, the best tested depths are 50, 100, 200, 200, and5000 (Fig. 2b). We then test whether an earlier selection
transfers: five-fold calibration selectsLusing Round-1
queries and freezes it for held-out queries across all rounds.
The nDCG@10 difference from full RRF changes from
−0.0085in Round 1 to−0.0272in Round 5 (95% CI:
[−0.0699,−0.0058]; Fig. 2c). A depth selected on earlier
data does not necessarily maintain its relative effectiveness as
the corpus changes.
3.3. Fixed-K access cost
Completing exact Top-20 is inexpensive for most queries but
costly in the tail (Table 1). On TREC-DL 2019, half the
queries finish within 311 accesses, whereas the P95 reaches
196410 and the maximum reaches 3533518—approximately
632 and 11362 times the median. The same pattern appears on
TREC-DL 2020 and TREC-COVID, where the P95 reaches
80 and 22 times the median, respectively. AKthat is afford-
able for most queries can therefore be expensive for others
within the same query set. This variability motivates specify-
ing the access budget directly and determining the output size
during execution, without a presetK.
Table 1. Accesses to complete exact Top-20 under unit-
access DiBud. All 770 queries complete.
Query setnMedian P95 Maximum
TREC-DL 2019 43 311 196410 3533518
TREC-DL 2020 54 444 35562 211169
NFCorpus 323 308 2685 5035
SciFact 300 325 3362 5011
TREC-COVID 50 2544 56363 91290

3.4. Certified output under equal budgets
We first isolate the effect of list selection by holding the bud-
get and certification rule fixed. With the same budget of 2048
accesses, DiBud certifies more results than Balanced on all
five query sets (Table 2). Averaged equally across sets, output
within the first 100 positions increases from 41.77 to 45.05,
a 7.86% gain; within the first 20, it increases from 18.07 to
18.18. Selective reading therefore increases the number of
exact positions obtained from the same access allowance.
Table 2. Mean certified output atB= 2048. Both methods
use unit accesses and the same certifier; only list selection
differs.
Cap 100 Cap 20
Query set Balanced DiBud Balanced DiBud
TREC-DL 2019 41.86 46.19 17.77 17.91
TREC-DL 2020 34.91 38.31 18.33 18.35
NFCorpus 62.06 63.71 19.24 19.28
SciFact 43.55 47.26 19.44 19.57
TREC-COVID 26.48 29.80 15.58 15.78
3.5. Quality and access cost relative to exact Top-20
We next examine the cost of requiring a fixed output count.
We compare two stopping conditions on the same access tra-
jectory: complete exact Top-20, or stop earlier when the bud-
get is exhausted. Five-fold calibration selects the smallest
budget retaining 95% of the reference mean nDCG@20 and
applies it to held-out queries. Missing output positions con-
tribute zero gain.
Table 3. Held-out results after 95% budget calibration. Re-
tention is the ratio of mean nDCG@20; savings compare total
accesses with completed exact Top-20.
Query set Retained (%) Fewer accesses (%)
TREC-DL 2019 95.05 99.53
TREC-DL 2020 95.89 93.50
NFCorpus 95.81 65.92
SciFact 97.68 83.41
TREC-COVID 95.85 69.71
Across the five query sets, budgeted stopping retains
95.05%–97.68% of mean nDCG@20 while reducing accesses
by 65.92%–99.53% (Table 3). Both stopping conditions fol-
low the same access trajectory, so these savings come from
stopping earlier. The 99.53% reduction on TREC-DL 2019
reflects the high cost of completing tail queries. These results
show that completing all 20 positions is not necessary to re-
tain most of the measured retrieval quality. Directly limiting
accesses and allowing the output size to vary preserves most
of that quality while avoiding much of the completion cost.4. CONCLUSION
Changing queries and corpora make it difficult for a fixed
Top-Ldepth to preserve its quality–cost tradeoff in RAG hy-
brid retrieval. Exact fusion removes this fixed cutoff, but a
prescribed Top-Kstill leaves access cost dependent on the
query. DiBud takes an access budget directly as input and
incrementally returns certified results, determining the output
size during execution.
Experiments on five query sets show that selective read-
ing certifies more results than balanced reading under equal
budgets. After calibration for 95% quality retention, bud-
geted stopping retains 95.05%–97.68% of mean nDCG@20
on held-out queries while reducing accesses by 65.92%–
99.53%. The long-tailed costs of completing exact Top-20
explain why direct control is needed: fixing the result count
leaves the required accesses free to vary across queries. Top-
Kis therefore not a budget for hybrid retrieval.
5. ACKNOWLEDGMENTS
OpenAI Codex assisted with drafting and editing the abstract
and Sections 1–4 from author-provided material, as well as
implementing the experimental code.
6. REFERENCES
[1] Elastic, “Reciprocal rank fusion,” https:
//www.elastic.co/docs/reference/elasticsearch/rest-apis/
reciprocal-rank-fusion, Accessed September 10, 2026.
[2] Microsoft, “Hybrid search scoring (RRF)—Azure
AI Search,” https://learn.microsoft.com/en-us/azure/
search/hybrid-search-ranking, Accessed September 11,
2026.
[3] Gordon V . Cormack, Charles L. A. Clarke, and Stefan
Buettcher, “Reciprocal rank fusion outperforms con-
dorcet and individual rank learning methods,” inPro-
ceedings of the 32nd International ACM SIGIR Confer-
ence on Research and Development in Information Re-
trieval, 2009, pp. 758–759.
[4] Ronald Fagin, Amnon Lotem, and Moni Naor, “Opti-
mal aggregation algorithms for middleware,”Journal of
Computer and System Sciences, vol. 66, no. 4, pp. 614–
656, 2003.
[5] Chunran Zhang, “Exact adaptive hybrid retrieval
without fixed top-L cutoffs,”arXiv preprint
arXiv:2608.07152, 2026.
[6] Nikos Mamoulis, Kit Hung Cheng, Man Lung Yiu, and
David W. Cheung, “Efficient aggregation of ranked in-
puts,” inProc. IEEE International Conference on Data
Engineering (ICDE), 2006, p. 72.

[7] Jing Yuan, Guang-Zhong Sun, Ye Tian, Guoliang Chen,
and Zhi Liu, “Selective-NRA algorithms for top-k
queries,” inAdvances in Data and Web Management
(APWeb/WAIM), 2009, pp. 15–26.
[8] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel
Campos, and Ellen M. V oorhees, “Overview of the
TREC 2019 deep learning track,”arXiv preprint
arXiv:2003.07820, 2020.
[9] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and
Daniel Campos, “Overview of the TREC 2020 deep
learning track,”arXiv preprint arXiv:2102.07662, 2021.
[10] Vera Boteva, Demian Gholipour, Artem Sokolov, and
Stefan Riezler, “A full-text learning to rank dataset for
medical information retrieval,” inAdvances in Informa-
tion Retrieval. 2016, pp. 716–722, Springer.
[11] David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu
Wang, Madeleine van Zuylen, Arman Cohan, and Han-
naneh Hajishirzi, “Fact or fiction: Verifying scientific
claims,” inProceedings of the 2020 Conference on Em-
pirical Methods in Natural Language Processing. 2020,
pp. 7534–7550, Association for Computational Linguis-
tics.
[12] Kirk Roberts, Tasmeer Alam, Steven Bedrick, Dina
Demner-Fushman, Kyle Lo, Ian Soboroff, Ellen
V oorhees, Lucy Lu Wang, and William R. Hersh,
“Searching for scientific evidence in a pandemic: An
overview of TREC-COVID,”Journal of Biomedical In-
formatics, vol. 121, pp. 103865, 2021.
[13] BAAI, “bge-small-en-v1.5: Model card,” https://
huggingface.co/BAAI/bge-small-en-v1.5, Accessed
September 10, 2026.
[14] Stephen Robertson and Hugo Zaragoza, “The prob-
abilistic relevance framework: BM25 and beyond,”
Foundations and Trends in Information Retrieval, vol.
3, no. 4, pp. 333–389, 2009.
# Quantum-Inspired Pairwise Ranking Aggregation via Active LLM Feedback

**Authors**: Quan Mai, Susan Gauch, Gibson Nkhata

**Published**: 2026-10-28 18:55:57

**PDF URL**: [https://arxiv.org/pdf/2610.38683v1](https://arxiv.org/pdf/2610.38683v1)

## Abstract
Ranking aggregation from multiple retrievers is a powerful yet under-explored technique in modern retrieval and RAG systems. We introduce LLM-IRA, the first extension of the quantum-inspired interactive ranking aggregation framework to document reranking. By leveraging large language models (LLMs) solely for pairwise judgments on high-disagreement document pairs, our method learns fusion weights in a principled, interactive way. The final ranking is a simple, ultra-fast pointwise linear fusion of scores, no full pairwise matrix or expensive listwise prompting is required. Experiments on the BEIR benchmark and TREC Deep Learning tracks with an open-weight instruction-tuned judge (Llama-3.1-8B-Instruct) show that LLM-IRA matches or numerically exceeds strong static fusion baselines on 5 of 6 evaluated benchmarks. We report two weight-update variants: momentum blending and a full accumulated-history refit, which together reach statistical significance (Wilcoxon signed-rank, p<0.05) over the best unsupervised aggregator on 4 of 6 benchmarks (DL19 under momentum; DL20, DBpedia, and NFCorpus under accumulated refit, which is the stronger of the two configurations overall). On the benchmarks we profiled (DL19, DL20), it already clears the best unsupervised baseline within as few as 5 LLM calls per query, well before exhausting its typical ~25--30 call budget. LLM-IRA also achieves cost-effectiveness comparable to state-of-the-art zero-shot LLM rerankers at an order of magnitude fewer LLM calls, and requires no training data or model fine-tuning.

## Full Text


<!-- PDF content starts -->

Quantum-Inspired Pairwise Ranking Aggregation via
Active LLM Feedback
Quan Mai1,2, Susan Gauch1, and Gibson Nkhata1
1University of Arkansas, Fayetteville, AR 72701, USA
2Applied AI, Walmart Inc., Bentonville, AR 72716, USA
Abstract.Rankingaggregationfrommultipleretrieversisapowerfulyetunder-
explored technique in modern retrieval and RAG systems. We introduce LLM-
IRA, the first extension of the quantum-inspired interactive ranking aggregation
framework [12] to document reranking. By leveraging large language models
(LLMs)solelyforpairwisejudgmentsonhigh-disagreementdocumentpairs,our
method learns fusion weights in a principled, interactive way. The final ranking
isasimple,ultra-fastpointwiselinearfusionofscores,nofullpairwisematrixor
expensive listwise prompting is required. Experiments on the BEIR benchmark
and TREC Deep Learning tracks with an open-weight instruction-tuned judge
(Llama-3.1-8B-Instruct) show that LLM-IRA matches or numerically exceeds
strong static fusion baselines on 5 of 6 evaluated benchmarks. We report two
weight-updatevariants:momentumblendingandafullaccumulated-historyrefit,
which together reach statistical significance (Wilcoxon signed-rank,𝑝 <0.05)
over the best unsupervised aggregator on 4 of 6 benchmarks (DL19 under mo-
mentum; DL20, DBpedia, and NFCorpus under accumulated refit, which is the
strongerofthetwoconfigurationsoverall).Onthebenchmarksweprofiled(DL19,
DL20), it already clears the best unsupervised baseline within as few as 5 LLM
calls per query, well before exhausting its typical∼25–30 call budget. LLM-IRA
also achieves cost-effectiveness comparable to state-of-the-art zero-shot LLM
rerankersatanorderofmagnitudefewerLLMcalls,andrequiresnotrainingdata
or model fine-tuning.
Keywords:Rankingaggregation·Documentreranking·Largelanguagemodels
·Quantum-inspired fusion·Interactive feedback.
1 Introduction
Ranking Aggregation (RA) is a crucial technique in information retrieval, designed
to fuse multiple search strategies to improve robustness and generalization, as no sin-
gle base ranker performs optimally across all domains. Existing RA methods broadly
divide into two categories: unsupervised techniques such as Reciprocal Rank Fusion
(RRF[6])andCombSUM,whicharesimpleandcost-effectivebutlackexternalguidance
for optimization; and fully-supervised approaches, including learning-to-rank models
(RankNet[2],LambdaMART[3])orzero-shotlistwiseLLMrerankers(RankGPT[28],
RankLLM [26]). While the latter deliver superior performance, they demand either
extensive labeled data for training or prohibitive inference costs (hundreds of LLM
arXiv:2610.38683v1  [cs.IR]  28 Oct 2026

2 Q. Mai et al.
calls per query) through massive prompting. Recently, a Quantum-Inspired Interac-
tive Ranking Aggregation (QI-IRA) [12] framework was proposed for image retrieval
(personre-identification),demonstratingthatRAcanbemodeledasfusingmultiple“in-
complete observations" into a “complete observation" using quantum theory. QI-IRA
achievedstate-of-the-artperformancewithremarkablylowlabelingcosts(asfewastwo
user feedbacks per query). However, QI-IRA’s pointwise relevance feedback (e.g., “Is
this item relevant or irrelevant?") is inadequate for document reranking, where strict
relative ordering is required. In this paper, we propose LLM-IRA, a novel framework
that extends the quantum RA analogy from point-wise absolute relevance to pairwise
relativerelevance.Wemodeldocumentpairsasentangledquantumsystemsandreplace
costlyhumanfeedbackwithanefficientinteractiveLLMloopthatdynamicallysamples
only the highest-disagreement pairs. This enables our method to learn optimal fusion
weights for heterogeneous base rankers while retaining the simplicity and speed of lin-
ear pointwise fusion. We report two weight-update rules for turning LLM preferences
into fusion weights:momentum blendingand afull accumulated-history refit, and find
the latter the stronger configuration overall. Experiments across two independent LLM
judgesdemonstratethatLLM-IRAmatchesoroutperformsstaticfusionbaselinesonthe
majorityof testedbenchmarks undereitherrule, andmatches zero-shotLLMrerankers
at a fraction of the cost.
Theremainderofthispaperisorganizedasfollows:Section2reviewsrelatedwork;
Sections 3 and 4 recap QI-IRA and detail our method, respectively; Sections 5 and 6
present experiments and results; and Section 7 concludes.
2 Related Work
RankaggregationRAcombinesoutputsfrommultiplebaserankerstoimproveoverall
performance. Unsupervised methods, such as RRF and CombSum/CombMNZ [11],
are lightweight and widely used in hybrid retrieval pipelines, requiring no training or
labels. These approaches fuse ranks or scores heuristically and perform well in diverse
settings but lack adaptability to query-specific nuances. Supervised RA treats fusion as
alearning-to-rank(LTR)problem.ClassicmodelslikeRankNet[2]usepairwiselosses
to learn preferences, while LambdaMART [3] optimizes listwise metrics via gradient
boosting. These methods achieve high accuracy with sufficient labeled data but incur
heavyannotationandtrainingcosts,limitingtheiruseinzero-shotordynamicscenarios.
LLM zero-shot document rerankingThe rise of LLMs has produced a parallel and
highly practical family of zero-shot rerankers. These fall into three main paradigms:
pointwise scoring [14,25], pairwise comparison [21], and listwise reranking [18,20,
28]. While effective on benchmarks like BEIR and MS MARCO, these methods scale
poorly: listwise prompting requires linear or quadratic LLM calls relative to candidate
size,leadingtohightokencosts.LLM-IRAusesLLMsmoresparingly,onlyfortargeted
pairwise judgments on disagreed pairs, to guide fusion weights, achieving comparable
performance at a fraction of the expense.
Interactive relevance feedbackhas evolved from classical Rocchio-style pseudo-
relevance feedback [24] and its neural variants [13] toward LLM-as-Judge techniques
[33]. Recent studies explore pairwise and listwise LLM judgments for query rewriting

Quantum-Inspired Pairwise RA via Active LLM Feedback 3
or preference alignment [32,15], yet they typically treat the LLM as a full reranker or
query expander rather than as a lightweight supervisor that learns fusion weights for
cheaper base retrievers.
LLM-guidedrankfusionMostcloselyrelatedtoourwork,LLM-RankFusion[27]uses
in-context learning with calibration to estimate pairwise preference probabilities and
aggregatemultiplerankers.Whileitachievesstrongeffectiveness,itrequirescomparing
allpairsinthecandidatepool(quadraticLLMcost),makingitimpracticalforlargepools.
In contrast, LLM-IRA queries the LLM only on the small subset of pairs where base
rankersdisagree,reducingcostbyanorderofmagnitudewhilemaintainingcompetitive
effectiveness.
WhileunsupervisedRAisefficientbutstatic,supervisedLTRandLLMrerankersare
powerfulbutcostly,andinteractivefeedbacklacksprincipledfusionmechanisms.LLM-
IRA bridges these by extending QI-IRA’s quantum-inspired framework with low-cost
LLMpairwiseinteractions,enablingadaptiveaggregationwithouttrainingorexcessive
prompting.
3 Preliminaries: QI-IRA Recap
TheoriginalQI-IRAframeworkmodelseachgalleryitemasaquantumstateevaluated
under observation operators corresponding to base rankers. Each basic ranking score
𝑠𝑚(𝑑)is interpreted as the collapse probability into a relevant state. Gleason’s theorem
justifies that the optimal fused score is a linear combination:
𝑠∗(𝑑)=𝑀∑︁
𝑚=1𝑤𝑚·𝑠𝑚(𝑑)
Given positive (Ψ+) and negative (Ψ−) feedback items, weights are updated as:
¯𝑤𝑚=1
|Ψ+|∑︁
𝑑𝑖∈Ψ+𝑠𝑚(𝑑𝑖)−1
|Ψ−|∑︁
𝑑𝑗∈Ψ−𝑠𝑚(𝑑𝑗)
followed by momentum smoothing:
𝑤𝑚
𝑡=𝛾¯𝑤𝑚+(1−𝛾)𝑤𝑚
𝑡−1
where𝛾regulates the degree of weight updating. This forms the foundation for our
pairwise extension.
4 The Pairwise Quantum Analogy
QI-IRA focuses on pointwise absolute relevance with bases|𝑅⟩(relevant) and| ˜𝑅⟩
(irrelevant). For document reranking, which emphasizes strict relative ordering, we
redefine the quantum model as follows:
The Particle System:Each microscopic particle represents a document pair (Docu-
ment A and Document B), with joint state vector|𝑑 𝐴,𝐵⟩.

4 Q. Mai et al.
The Observation Operators:The base rankers (e.g., BM25, dense, sparse) serve as
quantum observation operators.
The Orthogonal Bases:For each ranker𝑚, the 2-dimensional state space captures
relative relevance via orthogonal bases: (1)|𝐴≻ 𝑚𝐵⟩(Document A is strictly more
relevant than B), and (2)|𝐵≻ 𝑚𝐴⟩(Document B is strictly more relevant than A).
This shift enables modeling pairwise preferences while preserving the quantum-
inspired framework.
From 1D scores to pairwise collapse probabilitiesTo map pointwise scores (e.g.,
BM25scores)intothepairwisequantumstate,weadaptthepairwiselogitfromRankNet-
style LTR:
Pairwise collapse probability:The probability that A wins over B under ranker𝑚is:
𝑠𝑚(𝐴,𝐵)=1
1+𝑒−(𝑠𝑐𝑜𝑟𝑒𝐴−𝑠𝑐𝑜𝑟𝑒𝐵)
The amplitudes sum to 1 in squared form, yielding:
|𝑑𝐴,𝐵⟩=√𝑠𝑚(𝐴,𝐵)|𝐴≻𝑚𝐵⟩+√︁
1−𝑠𝑚(𝐴,𝐵)|𝐵≻𝑚𝐴⟩
Aggregation via Gleason’s theoremQI-IRA uses Gleason’s theorem to derive linear
fusion of pointwise scores. This extends directly to our pairwise case: The density
operator for ranker𝑚is𝜌 𝑚=|𝐴≻𝑚𝐵⟩⟨𝐴≻𝑚𝐵|. The aggregated operator is:
𝜌∗=𝑀∑︁
𝑚=1𝑤𝑚𝜌𝑚
Substituting into Gleason’s theorem yields the fused pairwise probability:
𝑠∗
(𝐴,𝐵)=𝑀∑︁
𝑚=1𝑤𝑚𝑠𝑚(𝐴,𝐵)
Thus,thefinalrankingisderivedfromtheaggregatedpairwiseprobabilities,maintaining
theoreticaloptimality.AdetailedmathematicalderivationextendingGleason’stheorem
to our pairwise quantum observation framework is provided in the Appendix A.
Interactive LLM feedback loopStarting with uniform weights, we compute an initial
fusedranking.Fromthetop-𝑟documents,wesample𝐾pairswithhighrankerdisagree-
ment (e.g., where BM25 prefers A over B, but dense prefers B over A). These pairs
are queried to the LLM: “Given the query, which document is more relevant: A or B?”
The LLM’s judgment (e.g., W≻L, where W is the winner and L the loser) acts as a
definitive measurement, collapsing the pair’s state.
Weight adjustmentOnly LLM judgments with confidence (logit gap) at or above a
threshold𝜏are retained in the valid preference set𝑉; low-confidence judgments are
discardedasnoiseratherthanassumedtoindicateanexplicit“bothirrelevant”outcome
(see Limitations). For each retained judgment (W≻L), define the positive score𝑠+
𝑚as
the probability ranker𝑚correctly predicted the order (𝑠 𝑚(𝑊,𝐿)) and the negative score
𝑠−
𝑚as the opposite (𝑠 𝑚(𝐿,𝑊)). For a batch of|𝑉|pairs, the provisional weight is:
¯𝑤𝑡
𝑚=1
|𝑉||𝑉|∑︁
𝑘=1(𝑠𝑚(𝑊𝑘,𝐿𝑘)−𝑠𝑚(𝐿𝑘,𝑊𝑘))

Quantum-Inspired Pairwise RA via Active LLM Feedback 5
Because𝑠𝑚(𝐿,𝑊)=1−𝑠𝑚(𝑊,𝐿), this simplifies to:
¯𝑤𝑡
𝑚=1
|𝑉||𝑉|∑︁
𝑘=1(2×𝑠𝑚(𝑊𝑘,𝐿𝑘)−1) ∈[−1,1]
This ensures that a ranker providing a highly confident, correct probability receives a
massivereward,ahighlyconfidentincorrectprobabilityreceivesamassivepenalty,and
anuncertainrankerreceivesnoupdate.Because¯𝑤𝑡
𝑚canbenegativeforasystematically
wrongranker,weexponentiateandrenormalizeacrossrankers(softmax)beforemomen-
tumblending,whichguarantees𝑤𝑡
𝑚>0andÍ𝑀
𝑚=1𝑤𝑡
𝑚=1ateveryround.Weintroduce
a weight-sharpness parameter𝛽(default𝛽=1) that scales the provisional weight before
exponentiating:
ˆ𝑤𝑡
𝑚=exp(𝛽·¯𝑤𝑡
𝑚)
Í𝑀
𝑚′=1exp(𝛽·¯𝑤𝑡
𝑚′)
At𝛽=1, the softmax keeps weight spread across all rankers even under strong dis-
agreement,whichisasensibledefaultwhenbaserankersaregenuinelycomplementary.
Raising𝛽letsthemethodcommitmorefullytoasingledominantrankerwhenonecon-
sistently outperforms the others, at the cost of amplifying noise from small per-round
preference batches.
Finally, we update the actual fusion weights using one of two rules. The default,
momentum,blendsonlythecurrentround’ssoftmaxestimatewiththepreviousweights,
where𝛾(default 0.7) controls the degree of updating to ensure stability:
𝑤𝑡
𝑚=𝛾×ˆ𝑤𝑡
𝑚+(1−𝛾)×𝑤𝑡−1
𝑚
The alternative,accumulated refit, instead recomputes¯𝑤𝑡
𝑚(and henceˆ𝑤𝑡
𝑚) fromevery
confident preference observed so far rather than only the current round’s batch, and
adopts it directly as the new weight with no momentum blend:𝑤𝑡
𝑚←ˆ𝑤𝑡
𝑚. We report
bothvariantsinTable1(𝛽=1/momentumvs.𝛽=3/accumulatedrefit);underaccumulated
refit,𝛾is unused.
The loop repeats until the LLM budget is exhausted or no unsampled disagreeing
pairs remain in the top-𝑟. The entire process is shown in Algorithm 1.
5 Experiment Setup
DatasetsWe evaluate our method on three widely-used benchmarks: TREC Deep
Learning 2019/2020 [8,7], and BEIR benchmark [30].
Base retrievers and candidate poolWe employ 3 heterogeneous base retrievers (𝑀=
3): BM25 [23] (lexical baseline), a dense retriever, and SPLADE [10] (learned sparse
retrieval).BM25scoresarerepresentedviaalinearrank-basedproxy(𝑠𝑐𝑜𝑟𝑒=𝑁−𝑟𝑎𝑛𝑘,
where𝑁is the candidate pool size), matching the transform used for our unsupervised
baselines, since raw BM25 scores are not retained in our retrieval cache; the dense
and SPLADE rankers use their native similarity/dot-product scores. For ourprimary
experiments we usenomic-embed-text-v1.5[19] as the dense ranker; we additionally

6 Q. Mai et al.
Algorithm 1Pairwise LLM-IRA for Document Reranking
Input:Query𝑞, documents𝐷, base-ranker scores{𝑠𝑐𝑜𝑟𝑒 𝑚(𝑑)}𝑀
𝑚=1, budget𝑇, zone size𝑡𝑜𝑝_𝑟,
pairs/round𝐾,confidencethreshold𝜏,sharpness𝛽,updaterule∈{momentum(𝛾),accum.refit}
Output:Final aggregated ranking of𝐷
1: Min-maxnormalizeeachranker’sscores;derivepairwisecollapseprobabilities𝑠 𝑚(𝐴,𝐵)(§4);
init𝑤0
𝑚←1/𝑀
2:for𝑡=1,...,𝑇do
3: Aggregate𝑠∗(𝑑)←Í
𝑚𝑤𝑡−1
𝑚𝑠𝑐𝑜𝑟𝑒𝑚(𝑑); isolate Top-𝑡𝑜𝑝_𝑟documents
4: Select the𝐾pairs with highest cross-ranker disagreement in𝑠 𝑚(𝐴,𝐵)within this pool
5: Query the LLM on each pair; keep judgments with confidence gap≥𝜏as valid set𝑉
6:ifno unsampled disagreeing pairs remain in Top-𝑡𝑜𝑝_𝑟then
7:break⊲sampling exhausted
8:else if𝑉=∅then
9:continue⊲all pairs below𝜏; resample next round
10: Computeprovisionalweights¯𝑤𝑡
𝑚from𝑉,softmax-normalizewithsharpness𝛽,andapply
momentum (𝛾) or accumulated refit to obtain𝑤𝑡
𝑚(§4)
11:if𝑤𝑡
𝑚converged vs.𝑤𝑡−1
𝑚then
12:break
13:return𝐷sorted by𝑠∗(𝑑)←Í
𝑚𝑤𝑡
𝑚𝑠𝑐𝑜𝑟𝑒𝑚(𝑑), descending
report results with all-MiniLM-L6-v2 [22] andBGE-large-en-v1.5[4] for retriever-
quality robustness (Table 3), each applied with their prescribed query and document
prefixes.
Allmethodsoperateonthesametop-100candidatepoolretrievedbyBM25,isolating
the reranking/fusion effect from upstream retrieval quality.
LLMjudgeandevaluationmetricsFortheprimaryevaluationweusetwoopen-weight,
instruction-tuned decoder-only judges:Llama-3.1-8B-Instruct[9] andQwen2.5-14B-
Instruct[29], using each model’s chat template and extracting next-token logits for
labels “A” and “B” to preserve the pairwise probability framework. We additionally
reportearlierexploratoryresultswiththeFlan-T5family[16](Large/XL/XXLvariants)
at the MiniLM retriever tier. The LLM budget is 100 pairwise queries per query (in
practice∼30areused).TheexactpromptisinAppendixB.Re-rankingdepthis𝑘=10;
the primary metric is NDCG@10 (reported as percentages).
BaselinesWe compare against three categories of baselines: (1)Unsupervised fusion:
theComb*family[11](CombMIN,CombMAX,CombSUM,CombANZ,CombMNZ),
BordaCount [1], and RRF [6]; and (2)LTR with LLM supervision(LTR-Logistic):
logistic regression over the same LLM pairwise labels collected by our disagreement-
based sampling, to isolate whether the quantum-inspired weight update adds value
beyond standard LTR on the same supervision signal.
ParametersWesetthetop-𝑟parameter(contentiouszone)to40forselectingdisagree-
mentpairs,choosethe𝐾=3highest-disagreementpairstoquerytheLLMineachround,
andonlyfoldapairwisejudgmentintotheweightupdateifthejudge’slogitgapexceeds
a confidence threshold𝜏=2.5. We arrived at this configuration by sweeping𝑟from 10
to 40: small𝑟starves the sampler under a fixed per-query LLM budget by repeatedly
re-querying an already-resolved contentious zone, while𝑟=40consistently gives the

Quantum-Inspired Pairwise RA via Active LLM Feedback 7
Table1.NDCG@10comparisonacrossTRECDLandBEIRbenchmarks,usingnomic-embed-
text-v1.5as the dense retriever and BM25+Dense+SPLADE (𝑀=3) as base rankers throughout.
LLM-IRA (Llama-3.1-8B) values use the final adopted configuration: contentious-zone𝑟=40,
𝐾=3pairs/round, and confidence-threshold𝜏=2.5on the LLM logit gap. Best overall inbold;
bestunsupervised underlined(tiesshownonalltiedcells).∗Statisticallysignificantimprovement
overthebestunsupervisedaggregatorinthatcolumn(Wilcoxonsigned-ranktest,pairedper-query
NDCG@10,𝑝<0.05);see§6forthefullper-columnsignificancebreakdown.‡Theaccumulated-
refit row uses weight-sharpness𝛽=3and the accumulated-refit update rule described in §4 (vs.
𝛽=1/momentum for the row above). Its∗markers follow the same convention as the momentum
row; see §6 for the full significance breakdown, including comparisons against the momentum
row and LTR-Logistic.
Category Method DL 19 DL 20 Covid SciFact DBpedia NFCorpusUnsupervisedCombMIN 67.51 63.08 76.99 71.53 41.14 37.23
CombMAX 60.04 55.89 68.34 69.65 37.02 34.69
CombSUM 69.56 65.2478.00 73.2643.3137.64
CombANZ 69.6265.19 78.00 73.21 43.24 37.55
CombMNZ 69.53 65.2478.00 73.2643.31 37.70
BordaCount 67.94 63.18 79.27 72.38 41.69 36.92
RRF 68.80 64.56 80.0772.90 42.44 36.98LTRLTR-Logistic70.9867.0282.0272.63 43.22 36.69OursLLM-IRA (Llama-3.1-8B, momentum) 70.42∗65.95 80.04 73.47 43.66 37.81
LLM-IRA (Llama-3.1-8B,𝛽=3, accum. refit)‡70.4567.67∗80.5773.70 44.29∗38.01∗
samplerenoughcandidatepairstokeepsurfacinggenuinelyhigh-disagreementpairsas
rounds proceed.𝜏=2.5was found to weakly dominate the laxer𝜏=2.0default (never
worse,occasionallymarginallybetter)acrossdatasets,andisadoptedasthefinalsetting;
𝑟and𝐾must remain small enough to stay computationally cheap, as sampling𝐾pairs
out of the zone requires𝑂(𝐾log𝑟2)time complexity before feeding them to the LLM.
Allexperimentswereconductedinasinglerunon2NVIDIATeslaT4GPUs,16GB
of VRAM each. Code will be made available upon acceptance.
6 Results
Unless noted otherwise, results below use the nomic/Llama-3.1-8B-Instruct tier of Ta-
ble1,whichwetreatascanonical(Figure1included).Table2reportscost-profileresults
attheMiniLM/Flan-T5-Largetier(matchingthererankerbaselines’judge),andTable3
reports additional retriever- and judge-scale robustness checks (MiniLM/BGE-large
tiers).
Comparison with Unsupervised RALLM-IRA (momentum) matches or beats the
best unsupervised aggregator on 5 of 6 benchmarks, but the edge is statistically
significant only on DL19.At the nomic/Llama-3.1-8B tier, using∼30 LLM calls
per query, it trails only on Covid (by 0.03 NDCG@10). Against each column’s best
unsupervised aggregator (Wilcoxon, paired per-query NDCG@10), the win on DL19
(vs. CombANZ,𝑝=0.041,Δ=+0.0080,𝑛=43) is borderline given the sample size;
elsewhere the gap is not significant (𝑝from 0.052 on DBpedia to 0.94 on Covid).

8 Q. Mai et al.
Table 2.Cost–effectiveness comparison on TREC DL 2019, at the MiniLM/Flan-T5-Large tier
(matchingtheLLMrerankerbaselines’judge).∗LLMrerankernumbersreproducedfromWanget
al.[31]usingFlan-T5-Large.TheLLM-IRArowusestheaccumulated-refitupdaterule(𝛽=3)with
BM25+Dense(MiniLM)baserankers(𝑀=2),contentious-zone𝑟=20,andthedefaultconfidence
threshold (𝜏=2.0); this secondary tier is retained for a judge-matched cost comparison, distinct
from the primary nomic/Llama-3.1-8B tier in Table 1. LTR-Logistic is omitted here: it uses the
identical disagreement-based sampling and LLM-call budget as LLM-IRA (∼27 calls/query), so
its cost profile is the same; see Table 1 for its NDCG@10 comparison.
TREC DL 2019NDCG@10 # Calls #Tokens
LLM Rerankers (Flan-T5-Large)∗
TourRank 48.2 130.0 92571
PRP-Graph 65.8 482.7 221782
Setwise-HeapSort 66.9 125.3 40450
LLM-guided Fusion (Flan-T5-Large)
LLM-IRA (Ours)66.43 26.95 6003
LLM-IRAreliablybeatstheweakunsupervisedaggregators,butisonlysome-
times ahead of the strong ones.It significantly beats CombMIN and CombMAX on
all six benchmarks (𝑝 <0.01), and BordaCount/RRF on four of six (not Covid or
SciFact).Againstthestrongestaggregators(CombSUM/CombANZ/CombMNZ,them-
selves near-ties with each other): significant on DL19 (𝑝=0.016–0.041) and Covid
(𝑝<0.01againstallthree);notsignificantonDL20orSciFact;onDBpedia/NFCorpus
only CombANZ clears significance (𝑝=0.025,0.0099) while CombSUM/CombMNZ
narrowly miss (𝑝=0.052,0.058/0.19).
A second configuration — accumulated-history weight refit at𝛽=3— beats
momentumonallsixbenchmarksandisthestrongestconfigurationoverallonthree
ofthem.Gainsovermomentumrange+0.03to+1.72NDCG@10,significantonDL20
(𝑝=0.0031) and DBpedia (𝑝=0.0225); not on DL19, Covid, or SciFact (𝑝=0.86,0.35,
0.51). Against the best unsupervised aggregator it is significant on DL20 (𝑝=0.011),
DBpedia (𝑝=0.0056), and NFCorpus (𝑝=0.019) — marked∗in Table 1 — not on
DL19, Covid, or SciFact (𝑝=0.16,0.36,0.20). Against LTR-Logistic it leads on four
of six (DL20, SciFact, DBpedia, NFCorpus; significant on the latter two,𝑝=0.02and
𝑝<0.001) and trails on DL19 and Covid (neither significant,𝑝=0.29and0.078; Covid
is the largest numerical deficit, 80.57 vs. 82.02). It is never significantly worse than
momentum, LTR-Logistic, or the best unsupervised aggregator on any benchmark — a
reasonable case for adopting it going forward.
Impact of LLM budgetLLM-IRA exceeds the best unsupervised aggregator within
just5LLMcallsperquery.Figure1reportsthisattheprimarynomic/Llama-3.1-8B-
Instruct,accumulated-refittier.OnDL19(bestunsupervised:CombANZ,69.62),LLM-
IRA reaches 70.07 at a budget of only 5 calls and is essentially flat thereafter (70.50–
70.80across10–30calls).OnDL20(bestunsupervised:CombSUM/CombMNZ,65.24),
LLM-IRA reaches 67.87 at 5 calls and drifts within 68.35–68.64 across 10–30 calls.
Both curves clear the best unsupervised aggregator at the smallest budget tested and

Quantum-Inspired Pairwise RA via Active LLM Feedback 9
5 10 15 20 25 30
LLM Budget0.650.660.670.680.690.700.71NDCG@10
dl19 (n=43)
dl20 (n=54)
DL19 best unsupervised (0.6962)
DL20 best unsupervised (0.6524)
Fig.1.NDCG@10vs.LLM-callbudget(5–30calls)attheprimarynomic/Llama-3.1-8B-Instruct
tier, under the accumulated-refit (𝛽=3) configuration, for DL19 (𝑛=43) and DL20 (𝑛=54);
checkpoints are read from per-round logs within a single full-budget run. Dotted lines mark
each column’s best unsupervised aggregator from Table 1 (CombANZ 69.62 for DL19, Comb-
SUM/CombMNZ 65.24 for DL20).
plateauratherthanpeakingataspecificcallcount,indicatingmostofthegainiscaptured
from the first handful of high-disagreement pairs.
Cost against LLM rerankersLLM-IRA exceeds PRP-Graph’s effectiveness at 5.6%
of its LLM calls, and comes within 0.5 NDCG@10 points of the SOTA HeapSort
reranker at about a fifth of its cost.We evaluate against computationally expensive
LLM rerankers — listwise TourRank [5], pairwise PRP-Graph [17], and setwise Heap-
Sort [34], reproduced from Wang et al. [31] using Flan-T5-Large; our own LLM-IRA
row (accumulated refit,𝛽=3) uses the same MiniLM/Flan-T5-Large tier for a judge-
matched comparison. As shown in Table 2, LLM-IRA (66.43 NDCG@10) exceeds
PRP-Graph(65.8)usingonly5.6%ofitsLLMcalls(27vs.483)and2.7%ofitsprompt
tokens, and trails HeapSort (66.9) by just 0.47 NDCG@10 points using about a fifth of
the LLM calls (27 vs. 125) and 14.8% of the prompt tokens.
RobustnesstoretrieverandjudgequalityLLM-IRA’sgainsholdacrossretrievertiers
andjudgescales,anditsignificantlybeatsRRFattheprimarynomictieronDL20
(numerically ahead on DL19 and Covid as well).Table 3 reports NDCG@10 across
retrievertiersandjudges,alongsideRRFateachtier.Switchingthedenseretrieverfrom
MiniLMtoBGE-largeraisestheRRFceiling,asexpected.AttheBGE-largetier,LLM-
IRA (Llama-3.1-8B) beats RRF by +0.28 (DL19) and +1.11 (DL20), trails by 0.17
(Covid); only DL20 is significant (𝑝=0.0497,𝑛=54; DL19𝑝=0.62, Covid𝑝=0.96).
Llama-3.1-8B vs. the larger Qwen2.5-14B at the same tier shows near-identical effec-
tiveness(largestgap0.65,DL20);judgescalebeyond∼8Bisnotthebindingconstraint.
The nomic/Llama-3.1-8B row reuses the accumulated-refit (𝛽=3) run from Table 1:

10 Q. Mai et al.
vs. RRF it is significant on DL20 (𝑝=0.0008), not significant on DL19 (𝑝=0.0594) or
Covid(𝑝=0.3643)thoughnumericallyaheadonboth—adifferentcomparisonthanTa-
ble1’smarker,whichisagainsteachcolumn’sbestunsupervisedaggregatorratherthan
RRF specifically. Swapping the judge down to the far smaller, encoder-decoder Flan-
T5-Large (770M) or Flan-T5-XL (3B) at this same nomic tier gives NDCG@10 on par
with (if anything, marginally above) Llama-3.1-8B-Instruct — 70.58/68.04/80.68 and
70.64/68.38/80.19 (DL19/DL20/Covid) respectively, both numerically ahead of RRF
onallthree—extendingthejudge-scale-insensitivityfindingdownroughlyanorderof
magnitude in parameter count, though we did not run significance tests for these two
rows.
Table3.RobustnessofLLM-IRAacrossretrieverandjudgeconfigurations(𝑀=3:BM25,Dense,
SPLADE).∗Significant vs. RRF (Wilcoxon𝑝=0.0497); DL19 and Covid gaps for Llama are
not significant (𝑝=0.62,𝑝=0.96).§Significant vs. RRF at the nomic tier (Wilcoxon𝑝=0.0008
DL20,𝑛=54);theDL19(𝑝=0.0594,𝑛=43)andCovid(𝑝=0.3643)gapsarenotsignificant,though
LLM-IRA is numerically ahead of RRF on both (DL19 +1.65, Covid +0.50 NDCG@10). The
nomic/Llama-3.1-8B-Instruct row reuses the accumulated-refit (𝛽=3) run reported as LLM-IRA
in Table 1.
Retriever Judge DL19 DL20 Covid
Best unsupervised (RRF)
MiniLM – 61.33 58.28 71.92
BGE-large – 70.15 64.11 78.05
nomic – 68.80 64.56 80.07
LLM-IRA
MiniLM Flan-T5-XXL 63.15 62.58 74.23
BGE-large Llama-3.1-8B-Instruct 70.43 65.22∗77.88
BGE-large Qwen2.5-14B-Instruct 70.52 65.87 77.82
nomic Llama-3.1-8B-Instruct (accum. refit) 70.45 67.67§80.57
nomic Flan-T5-Large (accum. refit) 70.58 68.04 80.68
nomic Flan-T5-XL (accum. refit) 70.64 68.38 80.19
7 Conclusion
We propose LLM-IRA, the first extension of quantum-inspired interactive ranking ag-
gregation to document reranking. Under the Llama-3.1-8B-Instruct judge, LLM-IRA
matches or numerically exceeds static rank aggregation methods on 5 of 6 evaluated
benchmarks.Ofthetwoweight-updateruleswereport,momentumreachessignificance
(Wilcoxon,𝑝 <0.05) over the best unsupervised aggregator on 1 of 6 benchmarks
(DL19), while accumulated-history refit (𝛽=3) extends this to 3 of 6 (DL20, DBpedia,
NFCorpus), is never significantly worse than momentum on any benchmark, and is the
configuration we recommend going forward. On the benchmarks we profiled (DL19,
DL20),LLM-IRAalreadyclearsthebestunsupervisedbaselinewithinasfewas5LLM
callsper query,wellbefore exhaustingitstypical∼25–30callbudget (Figure1).LLM-
IRA enables efficient fusion of heterogeneous retrievers with minimal LLM feedback.

Quantum-Inspired Pairwise RA via Active LLM Feedback 11
8 Limitations
While our framework shows strong potential, it presents several limitations for future
research:
LLM noiseEven with capable 8B–14B instruction-tuned judges, pairwise preferences
on contentious, high-disagreement document pairs remain noisy, and Covid shows this
can narrow or flip the outcome against LLM-IRA relative to a strong static aggregator
(the momentum configuration trails RRF there by 0.03 NDCG@10, Table 1). Because
the system directly translates the LLM’s absolute judgment into mathematical weight
adjustments, noisy or incorrect feedback can heavily penalize superior base rankers; a
confidence-gatedupdate(discardinglow-marginjudgments)mitigatesbutdoesnotelim-
inate this risk, and the appropriate confidence threshold is judge-specific and currently
set empirically rather than calibrated automatically.
Binary quantum state constraintsThe current 2D quantum state space strictly models
relative preference (𝐴≻𝐵or𝐵≻𝐴). In reality, both documents might be entirely
irrelevant to the query. Future work must mathematically integrate a “both irrelevant"
dimensionintothemodelwithoutviolatingthefundamentalquantumrulethatthesum
of squared probability amplitudes must equal one.
Reversion to pointwise inferenceTo maintain computational efficiency across large
datasets, the final global inference step abandons the pairwise logic and reverts to
standard 1D pointwise linear weighting. Future work could explore scalable methods
for executing a strictly pairwise final aggregation.
Disclosure of Interests.The authors have no competing interests to declare that are relevant to
the content of this article.
References
1. Aslam, J.A., Montague, M.: Models for metasearch. In: Proceedings of the 24th annual
internationalACMSIGIRconferenceonResearchanddevelopmentininformationretrieval.
pp. 276–284 (2001)
2. Burges,C.,Ragno,R.,Le,Q.:Learningtorankwithnonsmoothcostfunctions.Advancesin
neural information processing systems19(2006)
3. Burges,C.J.:Fromranknettolambdaranktolambdamart:Anoverview.Learning11(23-581),
81 (2010)
4. Chen,J.,Xiao,S.,Zhang,P.,Luo,K.,Lian,D.,Liu,Z.:BGEM3-Embedding:Multi-lingual,
multi-functionality, multi-granularity text embeddings through self-knowledge distillation
(2024)
5. Chen,Y.,Liu,Q.,Zhang,Y.,Sun,W.,Ma,X.,Yang,W.,Shi,D.,Mao,J.,Yin,D.:Tourrank:
Utilizing large language models for documents ranking with a tournament-inspired strategy.
In: Proceedings of the ACM on Web Conference 2025. pp. 1638–1652 (2025)
6. Cormack, G.V., Clarke, C.L., Buettcher, S.: Reciprocal rank fusion outperforms condorcet
andindividualranklearningmethods.In:Proceedingsofthe32ndinternationalACMSIGIR
conference on Research and development in information retrieval. pp. 758–759 (2009)

12 Q. Mai et al.
7. Craswell,N.,Mitra,B.,Yilmaz,E.,Campos,D.:OverviewoftheTREC2020deeplearning
track. CoRRabs/2102.07662(2021), https://arxiv.org/abs/2102.07662
8. Craswell, N., Mitra, B., Yilmaz, E., Campos, D., Voorhees, E.M.: Overview of
the trec 2019 deep learning track. In: Text REtrieval Conference (TREC). TREC
(March 2020), https://www.microsoft.com/en-us/research/publication/overview-of-the-trec-
2019-deep-learning-track/
9. Dubey, A., et al.: The Llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024)
10. Formal, T., Piwowarski, B., Clinchant, S.: SPLADE: Sparse lexical and expansion model
for first stage ranking. In: Proceedings of the 44th International ACM SIGIR Conference on
Research and Development in Information Retrieval. pp. 2288–2292 (2021)
11. Fox, E.A., Shaw, J.A.: Combination of multiple searches. NIST special publication SP243
(1994)
12. Hu, C., Zhang, H., Liang, C., Huang, H.: Qi-ira: quantum-inspired interactive ranking ag-
gregation for person re-identification. In: Proceedings of the AAAI conference on artificial
intelligence. vol. 38, pp. 2202–2210 (2024)
13. Li, C., Sun, Y., He, B., Wang, L., Hui, K., Yates, A., Sun, L., Xu, J.: Nprf: A neural pseudo
relevance feedback framework for ad-hoc information retrieval. In: Proceedings of the 2018
Conference on Empirical Methods in Natural Language Processing. pp. 4482–4491 (2018)
14. Liang,P.,Bommasani,R.,Lee,T.,Tsipras,D.,Soylu,D.,Yasunaga,M.,Zhang,Y.,Narayanan,
D., Wu, Y., Kumar, A., et al.: Holistic evaluation of language models. arXiv preprint
arXiv:2211.09110 (2022)
15. Liu, T., Qin, Z., Wu, J., Shen, J., Khalman, M., Joshi, R., Zhao, Y., Saleh, M., Baumgartner,
S., Liu, J., et al.: Lipo: Listwise preference optimization through learning-to-rank. In: Pro-
ceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association
forComputationalLinguistics:HumanLanguageTechnologies(Volume1:LongPapers).pp.
2404–2420 (2025)
16. Longpre, S., Hou, L., Vu, T., Webson, A., Chung, H.W., Tay, Y., Zhou, D., Le, Q.V., Zoph,
B., Wei, J., et al.: The flan collection: Designing data and methods for effective instruction
tuning. In: International conference on machine learning. pp. 22631–22648. PMLR (2023)
17. Luo,J.,Chen,X.,He,B.,Sun,L.:Prp-graph:Pairwiserankingpromptingtollmswithgraph
aggregation for effective text re-ranking. In: Proceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers). pp. 5766–5776 (2024)
18. Ma, X., Zhang, X., Pradeep, R., Lin, J.: Zero-shot listwise document reranking with a large
language model. arXiv preprint arXiv:2305.02156 (2023)
19. Nussbaum, Z., Morris, J.X., Duderstadt, B., Mulyar, A.: Nomic embed: Training a repro-
ducible long context text embedder. arXiv preprint arXiv:2402.01613 (2024)
20. Pradeep,R.,Sharifymoghaddam,S.,Lin,J.:Rankvicuna:Zero-shotlistwisedocumentrerank-
ing with open-source large language models. arXiv preprint arXiv:2309.15088 (2023)
21. Qin,Z.,Jagerman,R.,Hui,K.,Zhuang,H.,Wu,J.,Yan,L.,Shen,J.,Liu,T.,Liu,J.,Metzler,
D.,etal.:Largelanguagemodelsareeffectivetextrankerswithpairwiserankingprompting.
In:FindingsoftheAssociationforComputationalLinguistics:NAACL2024.pp.1504–1518
(2024)
22. Reimers,N.,Gurevych,I.:Sentence-bert:Sentenceembeddingsusingsiamesebert-networks.
In:Proceedingsofthe2019conferenceonempiricalmethodsinnaturallanguageprocessing
andthe9thinternationaljointconferenceonnaturallanguageprocessing(EMNLP-IJCNLP).
pp. 3982–3992 (2019)
23. Robertson, S., Zaragoza, H.: The probabilistic relevance framework: BM25 and beyond,
vol. 4. Now Publishers Inc (2009)
24. Rocchio Jr, J.J.: Relevance feedback in information retrieval. The SMART retrieval system:
experiments in automatic document processing (1971)

Quantum-Inspired Pairwise RA via Active LLM Feedback 13
25. Sachan, D., Lewis, M., Joshi, M., Aghajanyan, A., Yih, W.t., Pineau, J., Zettlemoyer, L.:
Improving passage retrieval with zero-shot question generation. In: Proceedings of the 2022
Conference on Empirical Methods in Natural Language Processing. pp. 3781–3797 (2022)
26. Sharifymoghaddam, S., Pradeep, R., Slavescu, A., Nguyen, R., Xu, A., Chen, Z., Zhang,
Y., Chen, Y., Xian, J., Lin, J.: Rankllm: A python package for reranking with llms. In:
Proceedingsofthe48thInternationalACMSIGIRConferenceonResearchandDevelopment
in Information Retrieval. pp. 3681–3690 (2025)
27. Shen, Y., Gao, Y., Shao, T., Guo, J.: Llm-rankfusion: Mitigating intrinsic inconsistency in
llm-based ranking. arXiv preprint arXiv:2406.00231 (2024)
28. Sun, W., Yan, L., Ma, X., Wang, S., Ren, P., Chen, Z., Yin, D., Ren, Z.: Is chatgpt good at
search?investigatinglargelanguagemodelsasre-rankingagents.In:Proceedingsofthe2023
conference on empirical methods in natural language processing. pp. 14918–14937 (2023)
29. Team, Q.: Qwen2.5 technical report. arXiv preprint arXiv:2412.15115 (2024)
30. Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., Gurevych, I.: Beir: A heteroge-
nous benchmark for zero-shot evaluation of information retrieval models. arXiv preprint
arXiv:2104.08663 (2021)
31. Wang,P.,Xia,Z.,Liao,C.,Wang,F.,Liu,H.:Realm:Recursiverelevancemodelingforllm-
based document re-ranking. In: Proceedings of the 2025 Conference on Empirical Methods
in Natural Language Processing. pp. 23875–23889 (2025)
32. Zhao, Y., Wang, Y., Yin, M.: Permutative preference alignment from listwise ranking of
humanjudgments.In:Proceedingsofthe2025ConferenceonEmpiricalMethodsinNatural
Language Processing. pp. 310–334 (2025)
33. Zheng, L., Chiang, W.L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D.,
Xing,E.,etal.:Judgingllm-as-a-judgewithmt-benchandchatbotarena.Advancesinneural
information processing systems36, 46595–46623 (2023)
34. Zhuang, S., Zhuang, H., Koopman, B., Zuccon, G.: A setwise approach for effective and
highlyefficientzero-shotrankingwithlargelanguagemodels.In:Proceedingsofthe47thIn-
ternationalACMSIGIRConferenceonResearchandDevelopmentinInformationRetrieval.
pp. 38–47 (2024)
A Proof of Pairwise RA via Gleason’s theorem
ThisappendixprovidesarigorousmathematicalderivationextendingGleason’stheorem
to pairwise relevance aggregation in document reranking. We model document pairs
as quantum systems and show that the optimal fused pairwise probability is a linear
weighted sum of individual ranker probabilities. The derivation assumes familiarity
withbasicquantummechanics(statevectors,densityoperators,andtraceoperations)and
Gleason’stheorem,whichstatesthatanyquantumprobabilitymeasureonaHilbertspace
canbeexpressedasthetraceofthestateoperatorwithadensityoperator:𝑝(𝐸)=tr[𝐷𝜌],
where𝐷isthesystemstateoperatorand𝜌isthemeasurementdensityoperatorforevent
𝐸.
Pairwise particle and state operatorConsider a document pair (Document A and
DocumentB)asthequantumsystemofinterest.Thejointstateofthepairisrepresented
bythestatevector|𝑑 𝐴,𝐵⟩.Thecorrespondingdensityoperator(stateoperator)is𝐷 𝐴,𝐵=
|𝑑𝐴,𝐵⟩⟨𝑑𝐴,𝐵|. This formulation treats the pair as a single entangled system in a 2-
dimensional Hilbert space for binary relative relevance.

14 Q. Mai et al.
Pairwise observation basesEach base ranker (e.g., BM25, dense retriever, sparse re-
triever)actsasameasurementoperatoronthepair.Forthe𝑚-thranker,themeasurement
space is spanned by two orthogonal bases representing relative relevance
1.|𝐴≻𝑚𝐵⟩:ThestatewhereDocumentAisstrictlymorerelevantthanDocumentB
according to ranker𝑚.
2.|𝐵≻𝑚𝐴⟩:ThestatewhereDocumentBisstrictlymorerelevantthanDocumentA
according to ranker𝑚.
These bases are orthonormal:⟨𝐴≻ 𝑚𝐵|𝐵≻𝑚𝐴⟩=0and⟨𝐴≻ 𝑚𝐵|𝐴≻𝑚𝐵⟩=
⟨𝐵≻𝑚𝐴|𝐵≻𝑚𝐴⟩=1.
The Pairwise collapse probabilityThe pairwise score𝑠 𝑚(𝐴,𝐵)from the𝑚-th ranker is
theprobabilitythatthesystemcollapsesintothestatewhereAismorerelevantthanB:
𝑠𝑚(𝐴,𝐵)=|⟨𝐴≻𝑚𝐵|𝑑𝐴,𝐵⟩|2(1)
ByGleason’stheorem,thisprobabilitycanbereformulatedusingthedensityoperator
for the measurement:
1. Define the density operator for the “Document A is more relevant than B" event
under ranker𝑚as
𝜌𝑚=|𝐴≻𝑚𝐵⟩⟨𝐴≻𝑚𝐵|(2)
2. Then,
𝑠𝑚(𝐴,𝐵)=tr[𝐷𝐴,𝐵𝜌𝑚](3)
ThisholdsbecauseGleason’stheoremguaranteesthatallquantumprobabilitiesare
trace-form measures.
AggregatingpairwiserankingsTofuse𝑀rankers,wecombinetheir“incomplete"pair-
wise observations into a complete observation. Define the aggregated density operator
as a linear combination of the individual density operators:
𝜌∗=𝑀∑︁
𝑚=1𝑤𝑚𝜌𝑚=𝑀∑︁
𝑚=1𝑤𝑚|𝐴≻𝑚𝐵⟩⟨𝐴≻𝑚𝐵|(4)
where𝑤𝑚arethefusionweights.For𝜌∗tobeavaliddensityoperator,the𝑤 𝑚must
form a probability distribution:𝑤 𝑚≥0andÍ𝑤𝑚=1. As described in the main text
(Section4),therawper-rankerstatisticthatdrives𝑤 𝑚canbenegativeforasystematically
wrongranker,soweexponentiateandrenormalize(softmax)beforesubstitutinginto𝜌∗,
guaranteeing this constraint holds at every round.
Applying Gleason’s theorem to the aggregated operator gives the fused pairwise
probability:
𝑠∗
(𝐴,𝐵)=tr[𝐷𝐴,𝐵𝜌∗].(5)
Substituting the expression for𝜌∗:

Quantum-Inspired Pairwise RA via Active LLM Feedback 15
𝑠∗
(𝐴,𝐵)=tr"
𝐷𝐴,𝐵𝑀∑︁
𝑚=1𝑤𝑚|𝐴≻𝑚𝐵⟩⟨𝐴≻𝑚𝐵|#
=𝑀∑︁
𝑚=1𝑤𝑚tr[𝐷𝐴,𝐵|𝐴≻𝑚𝐵⟩⟨𝐴≻𝑚𝐵|].(6)
From Equations 2 and 3,tr[𝐷 𝐴,𝐵|𝐴≻𝑚𝐵⟩⟨𝐴≻𝑚𝐵|]=𝑠𝑚(𝐴,𝐵), so:
𝑠∗
(𝐴,𝐵)=𝑀∑︁
𝑚=1𝑤𝑚𝑠𝑚(𝐴,𝐵).
By this derivation, Gleason’s theorem proves that the aggregated probability of
“Document A being more relevant than Document B" is the linear weighting of the
individual pairwise probabilities from each base ranker. This validates the pairwise
fusion model as theoretically optimal, mirroring the pointwise case but adapted for
relative ordering in reranking tasks. The weights𝑤 𝑚can be learned interactively (as
described in the main text), enabling efficient aggregation without full supervision.
B Prompt
We ask the LLM to pick the more relevant of exactly two passages using the following
prompt (shown for the decoder-only judges, e.g. Llama-3.1-8B-Instruct; the encoder-
decoderFlan-T5judgesusethesameuserturnwithoutasystemmessage),matchingthe
pairwise, two-label (A/B) setup described in §4; since the LLM is requested to output
only a single character, the number of output tokens is negligible.
System:You are a helpful passage-ranking assistant. Read the query and the
candidatepassagescarefully,thenoutputONLYthesingleletterlabel(A,B)of
the passage most relevant to the query.
User:Given a query “{query}”, which of the following passages is the most
relevant to the query?
Passage A: “{passage_A}”
Passage B: “{passage_B}”
Output only the passage label of the most relevant passage:
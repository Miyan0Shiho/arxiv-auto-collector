# Practical RAG Evaluation: A Rarity-Aware Set-Based Metric and Cost-Latency-Quality Trade-offs

**Authors**: Etienne Dallaire

**Published**: 2025-11-12 18:49:21

**PDF URL**: [https://arxiv.org/pdf/2511.09545v1](https://arxiv.org/pdf/2511.09545v1)

## Abstract
This paper addresses the guessing game in building production RAG. Classical rank-centric IR metrics (nDCG/MAP/MRR) are a poor fit for RAG, where LLMs consume a set of passages rather than a browsed list; position discounts and prevalence-blind aggregation miss what matters: whether the prompt at cutoff K contains the decisive evidence. Second, there is no standardized, reproducible way to build and audit golden sets. Third, leaderboards exist but lack end-to-end, on-corpus benchmarking that reflects production trade-offs. Fourth, how state-of-the-art embedding models handle proper-name identity signals and conversational noise remains opaque. To address these, we contribute: (1) RA-nWG@K, a rarity-aware, per-query-normalized set score, and operational ceilings via the pool-restricted oracle ceiling (PROC) and the percentage of PROC (%PROC) to separate retrieval from ordering headroom within a Cost-Latency-Quality (CLQ) lens; (2) rag-gs (MIT), a lean golden-set pipeline with Plackett-Luce listwise refinement whose iterative updates outperform single-shot LLM ranking; (3) a comprehensive benchmark on a production RAG (scientific-papers corpus) spanning dense retrieval, hybrid dense+BM25, embedding models and dimensions, cross-encoder rerankers, ANN (HNSW), and quantization; and (4) targeted diagnostics that quantify proper-name identity signal and conversational-noise sensitivity via identity-destroying and formatting ablations. Together, these components provide practitioner Pareto guidance and auditable guardrails to support reproducible, budget/SLA-aware decisions.

## Full Text


<!-- PDF content starts -->

Practical RAG Evaluation:
A Rarity-Aware Set-Based Metric
and Cost–Latency–Quality Trade-offs
Etienne Dallaire
Independent Researcher
Paris, France
Abstract
This paper addresses the guessing game in building production RAG. Classical rank-centric IR metrics
(nDCG/MAP/MRR) misfit RAG, where LLMs consume a set of passages rather than a browsed list; position
discountsand prevalence-blindaggregationmisswhat matters: whetherthe promptatcutoff Kcontainsthe
decisive evidence. Second,there is nostandardized, reproducible way tobuild and auditgolden sets. Third,
leaderboardsexistbutlackend-to-end,on-corpusbenchmarkingthatreflectsproductiontrade-offs. Fourth,
how state-of-the-art embedding models handle proper-name identity signals and conversational noise remains
opaque. Toaddressthese,wecontribute: (1)RA-nWG@K,ararity-aware,per-query-normalizedsetscore,
and operational ceilings via the pool-restricted oracle ceiling (PROC) and %PROC to separate retrieval from
ordering headroom within a Cost-Latency-Quality (CLQ) lens; (2) rag-gs (MIT), a lean golden-set pipeline
withPlackett-Lucelistwiserefinementwhoseiterativeupdatesoutperformsingle-shotLLMranking;(3)a
comprehensive benchmark on a production RAG (scientific-papers corpus) spanning dense retrieval, hybrid
dense+BM25, embedding models and dimensions, cross-encoder rerankers, ANN/HNSW, and quantization;
and(4)targeteddiagnosticsthatquantifyproper-nameidentitysignalandconversational-noisesensitivity
via identity-destroying and formatting ablations. Together, these components provide practitioner Pareto
guidance and auditable guardrails to support reproducible, budget/SLA-aware decisions.
1arXiv:2511.09545v1  [cs.IR]  12 Nov 2025

1 Introduction & Motivation
Productionretrieval-augmentedgeneration(RAG)isaset-consumptionpipeline: thegeneratorreceivesa
bounded context consisting of the top- 𝐾retrieved passages serialized into a single prompt, and downstream
utilitydependsalmostentirelyonwhetherthatsetcontainsthedecisiveevidenceunderthepromptbudget.
Rank-centric assumptions inherited from interactive IR—position discounting, smoothness of scores, user
browsing—are therefore misaligned with the operational objective, which is to maximize evidence presence
andusefulnessatfixed 𝐾subjecttoservice-levelconstraints. Becauseretrievaldecisionsjointlydetermineboth
quality(whichitemsentertheprompt)andspend/latency(prompttokencount,first-tokendelay,andreranking
cost), we adopt a cost–latency–quality (CLQ) lens that treats pre-generation choices—embedder, ANN
configuration,candidatedepth 𝐾,hybridization,andreranking—ascontrolsonaconstrainedoptimization.
Generator input cost scales approximately linearly with the total prompt tokens ( ∝𝐾×tokens per chunk), and
first-tokenlatencyincreaseswiththesame;henceanyincreasein 𝐾orchunklengthsimultaneouslyraises
rerankingexpenseandfinalinferencetime. Realworkloadscompoundthiswithlabelheterogeneityacross
queries: the available supply of high-utility passages (grade-5/grade-4 under a fixed rubric) varies by orders
of magnitude, so unnormalized metrics conflate system behavior with prevalence.
Wegroundevaluationandoperationsinthreedesigncommitments. First,weevaluatesets,notranks. Our
core metric, RA-nWG@ 𝐾, is order-free, per-query normalized, and rarity-aware: it aggregates stationary
per-passage utilities, scales mid-grades by inverse prevalence under strict caps so grade-5 dominance is
preserved,anddividesbythequery’spool-restrictedoracleat 𝐾(PROC)toyielda[0,1]scorecomparable
across heterogeneous label mixes. Second, we pair RA-nWG@ 𝐾with coverage diagnostics—N-Recall 4+@𝐾
andN-Recall 5@𝐾—thatreportthefractionofavailablehigh-utilityevidencecapturedunderthesamebudget;
whenthedeployedgeneratorisorder-sensitiveordistractible,wesupplementwithaharmrate(Harm@ 𝐾)
rather than baking penalties into the core score. Third, we make ceilings operational: PROC exposes whether
headroomislimitedbypoolcoverage(retrieval)orordering(reranking);therealized(%)PROCtellsoperators
which knob to turn next. Low PROC mandates improving the pool (dense+BM25 hybridization via RRF,
ANN recall tuning, query rewriting/denoising); high PROC with low (%)PROC points to ordering (stronger
reranking, near-duplicate suppression, metadata hygiene, shorter chunks).
Two recurrent failure modes motivate explicit guardrails. Identity signal is pivotal in scientific and enterprise
corpora: erasingorcorruptingpropernames(hardmasking,gibberishsubstitutions,near-missedits)collapses
dense similarity, whereas light orthographic or formatting variation (casing, diacritics, name order) is
mostly benign. Conversational noise—greetings, fillers, digressions, emoji—injects variance into the
embedding, systematically depressing cosine similarity under tight budgets and more so in multilingual
settings. Accordingly, we rewrite/denoise queries by default, enforce Unicode normalization, and treat
identity-destroyingtransformsashigh-risk. Routingfollowsfromdiagnostics: weset 𝐾=50asthedefault
andescalateto 𝐾=100onlyunderuncertaintysignals(smalldense-cosinemargins,highrerankerentropy,
diagnostic drops), preserving @10 precision while lifting @30 recall and controlling latency. The most
reliablestackunderthisregimeisHybrid+Rerank: buildthecandidatepoolwithdense+BM25(RRF-100)
to raise PROC, then apply a strong cross-encoder (rerank-2.5) to realize it; use 2.5-lite only under hard
cost caps or recall-heavy, precision-tolerant workloads. All claims are made reproducible via rag-gs, a
golden-set pipeline that standardizes embedding, retrieval, LLM judging on a 1–5 utility rubric, pruning, and
confidence-aware listwise refinement, emitting manifests with PROC/(%)PROC and CLQ measurements so
results are auditable across stacks and over time.
2

2 Metrics and Evaluation Foundations
Thissectionintroducesourevaluationframework: aset-based,rarity-awaremetric(RA-nWG@ 𝐾)aligned
with RAG consumption patterns. We use this metric throughout following sections to evaluate retrieval
quality under various optimization strategies.
2.1 Task framing: RAG set consumption
In production RAG, the LLM consumes a set of passages within a prompt; there is no user scanning a ranked
list. Evaluationshouldthereforeaskwhethertheretrievedsetcontainstheusefulevidenceunderafixedbudget
𝐾, not how smooth the rank order looks. Recent work introducing UDCG (Utility and Distraction-aware
Cumulative Gain; Trappolini et al., 2025) likewise adopts a set-based evaluation framing for RAG and
explicitly models position effects; we adopt the same framing but treat within-prompt order as secondary,
evaluating sets order-free.
Forclarity,wefix 𝑤5=1throughout;rarityscalingappliesonlyto 𝑤4and𝑤3(withcaps),sograde-4/grade-3
cannot substitute for grade-5 even when grade-5 is scarce.
2.2 Why classical IR is a misfit
Rank-centric metrics such as nDCG/MAP/MRR rely on assumptions that do not hold in RAG:
Monotone positional discount(lower ranks matter less). This assumption breaks down when LLMs consume
retrieved passages as a set within a prompt rather than users sequentially scanning a ranked list. While
research has documented “lost in the middle” effects—where LLMs exhibit degraded performance when
critical information appears mid-context due to architectural factors like RoPE decay and causal attention
masking (Liu et al., 2024; Wu et al., 2025). Recent work (UDCG; (Trappolini et al., 2025)) finds that
removing positional discounting achieves nearly identical performance, indicating that position-agnostic
evaluation can still correlate strongly. We therefore keep an order-free core metric and report harm separately
for deployments where order sensitivity or distractors matter.
Benignnon-relevance.Priorworkhasshownthatnon-relevantpassagescanactivelymisleadRAGsystems
(e.g., by distracting generation or steering it toward near-miss evidence) (Shi et al., 2023; Yoran et al., 2024;
Yu et al., 2024; Amiraz et al., 2025; Trappolini et al., 2025). However, we treat distractor sensitivity as out of
scopeforthismetricdesign. Priorstudies—andourownempiricalobservations—indicatethatwell-prompted
SOTALLMgeneratorscanberesilienttohard,semanticallyrelateddistractors(Yoranetal.,2024;Shenetal.,
2024;Caoetal.,2025). Wenotethisasahypothesisandpostponeformalevaluationtofuturework. Crucially,
because distractor impact often declines for recent SOTA generators under realistic RAG setups—though
notuniversally—wedonothard-codedistractorpenalties,keepingthemetricfuture-proofandfocusedon
evidence presence/utility (Cao et al., 2025; Yoran et al., 2024). If you deploy with earlier-generation or more
distractible LLMs, augment reporting with Harm@ 𝐾=(# grade≤2in top-𝐾)/𝐾(or an equivalent harm
label), alongside RA-nWG@𝐾and N-Recall 4+@𝐾.
Query-invariantlabelmix.Queriesdifferwildlyinhowmuchhigh-utilityevidenceexists(e.g.,onehas 1×
grade-5 among many grade-3s; another has 10×grade-5). Raw ranked scores then reflect label prevalence as
much as system quality, so cross-query comparisons require per-query normalization.
3

Corollary: ranking smoothness ≠evidence presence. A system can neatly rank many “okay” passages
andstillmissthedecisiveone. Wethereforepreferset-based,per-query–normalizedmeasuresthatanswer:
“Normalized to the query’s available high-utility evidence, what share does the retrieved top-𝐾capture?”
When distractors are prevalent and the generator is order-sensitive or brittle, a position- and harm-aware
composite(e.g.,UDCG)maytrackansweraccuracymorecloselyinthatspecificdeployment;ourbaseline
remains order-free and recall-first to stay robust as LLMs improve.
2.3 Scoring design (RA-nWG@𝐾)
Principle.Usingthe stationaryutility—a 1–5 per-passage rubricindependent of order—wethen normalize
within query against that query’s best achievable top-𝐾set.
RA-nWG@𝐾generalizesnormalizedcumulativegainat 𝐾tosetconsumptioninRAGbyintroducingper-
query, rarity-aware gains (inverse-prevalence with caps and fallback), while retaining order-free, oracle-at- 𝐾
normalization (Järvelin and Kekäläinen, 2002).
Importantly, we do not up-weight grade-5 by rarity: we fix 𝑤5=1. Rarity scales only 𝑤4and𝑤3relativeto
𝑤5, under caps𝑤 4≤1.0and𝑤 3≤0.25.
•Within-querynormalization.Comparetheobservedtop- 𝐾utilitytotheoracletop- 𝐾ceilingforthat
samequery(thebestsetonecouldformfromitspool). Thisyieldsa [0,1]scorethatiscomparable
across queries with different label distributions.
•Rarity-aware weights.Weight grades by inverse prevalence within the query so that scarce grade-5
evidence dominates when rare, while capping grade-4/grade-3 contributions to avoid diluting grade-5
impact. (Caps keep the metric stable when mid-grade items are abundant.)
•Fallback schedule.Ifaquery’spoolcontains nograde-5,applyafixed, conservativegrade-4/grade-3
weighting so the metric remains informative rather than collapsing.
Rarity weighting: rationale.(1)Budget alignment.Under a fixed top- 𝐾, missing decisive (grade-5)
evidence is more damaging than adding several mid-grade items; rarity scaling encodes this opportunity
cost. (2)Cross-query comparability.Label mixes vary widely; combining within-query normalization to the
oracle@𝐾with inverse-prevalence weighting keeps scores comparable across queries. (3)Guardrails.Caps
(𝑤4≤1.0,𝑤 3≤0.25) prevent scarcity from making grade-4/grade-3 appear equivalent to grade-5.
Metrics reported.
•RA-nWG@𝐾— ratio of observed weighted gain in the top- 𝐾set to the query’s oracle weighted gain at
𝐾. Interpreted as “how close to the best achievable set we retrieved” under budget𝐾.
•N-Recall 4+@𝐾/N-Recall 5@𝐾—normalizedcoverageofgrade ≥4(orgrade=5)evidence: fractionof
the available high-utility items that appear in the top- 𝐾, normalized by(min(𝐾,𝑅)) to handle varying
poolsizes. Using min(𝐾,𝑅) inthedenominatorequalizesquerieswithsmallpools,soquerieswith
𝑅<𝐾are not unfairly penalized.
4

Practical notes.
•Use therag-gspipeline for consistent labeling, audits, and reproducibility; keep the judging rubric
aligned with the 1–5 scale.
•Report macro-averages across queries, the number of valid queries per metric (handling zero-
denominator cases as NA), and multiple𝐾values to reflect different prompt budgets.
Stationary utility: scope & limits.Ourper-passage grades (1–5)approximate standaloneusefulness and
areusedasfixedsetweightsatbudget 𝐾. Thisabstractionisauditableandkeepsofflinescoringtractable,
butitdoesnotmodelredundancyorcomplementarity. WethereforepairRA-nWG@ 𝐾withcoverageKPIs
(N-Recall 4+@𝐾/N-Recall 5@𝐾) and Harm@ 𝐾; optionally, a novelty-discounted variant can down-weight
near-duplicates.
NoveltydiscountLet 𝛿(𝑑)=1 forthefirstoccurrenceofasource/facetand 𝛿(𝑑)=𝛽∈[0,1) forrepeats;then
𝐺nov
obs(𝐾)=∑︁
𝑑∈TopK(𝑞)𝛿(𝑑)𝑤𝑔(𝑑).
RelationtoUDCG.UDCGaggregatespassageutilitywithpositionweightsandassignsnegativecontri-
butionstodistractorstobettercorrelatewithend-to-endaccuracywhenorderandharmmatter(Trappolini
etal.,2025). Wesharetheset-utilitypremisebutchooseanorder-free,rarity-aware,per-query-normalized
formulation. WerecommendreportingHarm@ 𝐾alongsideRA-nWG@ 𝐾whendeployingwithorder-sensitive
or brittlegenerators, rather than hard-coding penalties into the core score. See also rank-centric baselines
(nDCG/MAP/MRR/RBP)anddiversitymetrics( 𝛼-nDCG/NRBP)forcontrastinassumptionsaboutorder,
user browsing, and redundancy (Järvelin and Kekäläinen, 2002; Clarke et al., 2008, 2009; Moffat and Zobel,
2008; Manning et al., 2008).
2.3.1 Empirical alignment with retrieval quality
In our CLQ studies, configurations thatraise N-Recall 4+@10(e.g., adding a reranker: 0.592 →0.835) also
raise RA-nWG@ 10(0.566→0.804) at comparable latency budgets, and increasing 𝐾trades top-10 quality
for deeper recall (RA-nWG@ 30up to 0.828 at 𝐾=100). This aligns with the Acc |Hit view: conditioned
on having all required evidence in the prompt, strong LLMs answer correctly at highrates. Consequently,
RA-nWG@𝐾(rarity-awaresetutility)pairedwithN-Recall 4+@𝐾(coverageofgoodevidence)providesan
outcome-predictive, budget-aware summary for RAG.
Acc|Hit@𝐾(definition).Accuracy conditioned on full-evidence retrieval: we measure answer correctness
only on queries where the full gold evidence set is present in top- 𝐾(i.e.,Hit@case_K=1 ). This isolates the
generator from the retriever: if retrieval succeeded, how often does the LLM answer correctly?
Stationaryutility(definition)Apassage’sstationaryutilityisitsintrinsicusefulnessassessedinisolation—
invariant to rank, list order, co-retrieved passages, and the budget 𝐾. We use it as a fixed weight for set
evaluation: sum utilities over the top-𝐾and compare that total to the query’s oracle top-𝐾ceiling.
5

2.4 Formal definitions
Setup (per query𝑞)
Labels:𝑔∈{1,2,3,4,5}(LLM-as-judge rubric).
Utility grading scale
•5 = responds clearly / contains the key elements
•4 = highly relevant, substantial information
•3 = partially relevant; related notions but insufficient
•2 = weak relevance; tangential allusions
•1 = not relevant
Pool size:𝑁(graded passages for𝑞).
top-𝐾: TopK(𝑞)
Base utilities (stationary, order-free)
𝑏5=1.0, 𝑏 4=0.5, 𝑏 3=0.1, 𝑏 2=𝑏 1=0.
Counts, proportions, rarity
𝑛𝑔=#{passages of grade𝑔}, 𝑝 𝑔=𝑛𝑔
𝑁.
If𝑝𝑔=0, treat𝑟 𝑔=0.
Rarity score (alpha=1by default)
𝑟𝑔=𝑏𝑔
𝑝𝛼𝑔, 𝛼=1.
Weset𝛼=1bydefaultforproportional,interpretableprevalencecorrection; 𝛼=0reducestonorarity. Caps
(cap4=1.0,cap3=0.25)enforce grade-5 dominance and bounded compensation. Appendix A reports
sensitivity over𝛼∈{0,0.5,1,2},cap4∈{0.75,1.0}, andcap3∈{0.20,0.25,0.33}.
Weight normalization (relative to grade-5) with caps
Defaults:cap4=1.0,cap3=0.25.
𝑤5=1, 𝑤 4=min𝑟4
𝑟5,cap4
, 𝑤 3=min𝑟3
𝑟5,cap3
, 𝑤 2=𝑤 1=0.
Fallback when no grade-5 exists in the pool(𝑛 5=0)
If𝑛 5=0 :𝑤 5=1, 𝑤 4=1, 𝑤 3=0.2, 𝑤 2=𝑤 1=0.
6

This fallback is applied only when 𝑛5=0, preventing undefined normalization by 𝑟5and keeping the metric
informative on0×grade-5 queries.
Observed and ideal gains at cut𝐾
𝐺obs(𝐾)=∑︁
𝑑∈TopK(𝑞)𝑤𝑔(𝑑).
𝐺ideal(𝐾)=max
𝑆⊆pool,|𝑆|=𝐾∑︁
𝑑∈𝑆𝑤𝑔(𝑑)=𝐾∑︁
𝑖=1𝑤𝑔★
𝑖(take the𝐾highest𝑤 𝑔in the pool).
Main metric: RA-nWG@𝐾(rarity-aware, normalized within-query, set-based)
RA-nWG@𝐾= 
𝐺obs(𝐾)
𝐺ideal(𝐾),if𝐺 ideal(𝐾)>0,
NA,otherwise.
Complementary coverage and precision KPIs
𝑅4+=𝑛 4+𝑛 5, 𝑅 5=𝑛 5.
𝐺4+(𝐾)=∑︁
𝑑∈TopK(𝑞)1
𝑔(𝑑)≥4
, 𝐺 5(𝐾)=∑︁
𝑑∈TopK(𝑞)1
𝑔(𝑑)=5
.
N-Recall 4+@𝐾= 
𝐺4+(𝐾)
min{𝐾, 𝑅 4+},if𝑅 4+>0,
NA,otherwise,N-Recall 5@𝐾= 
𝐺5(𝐾)
min{𝐾, 𝑅 5},if𝑅 5>0,
NA,otherwise.
Precision 4+@𝐾=𝐺4+(𝐾)
𝐾,Harm@𝐾=1
𝐾∑︁
𝑑∈TopK(𝑞)1
𝑔(𝑑)≤2
.(Optional to report.)
AggregationacrossqueriesReporting:macro-averageeachmetricoverquerieswhereitsdenominator
>0(exclude NAs), and report the count of valid queries per metric and 𝐾. Evaluate at multiple 𝐾to surface
budget trade-offs.
Hyperparameters: rationale&robustnessThedefaultsencoderubric-alignedconstraintsratherthan
tunedtargets: (i)grade-5dominance(no raritysetting allowsgrade-4/grade-3 toexceedgrade-5), and (ii)
bounded compensation(many mid-grade items cannot replace a decisive one). We pre-register these values
and report robustness grids in Appendix A; conclusions are stable across reasonable ranges.
2.5 Related Work — Evaluation Metrics
Cumulative-gain lineage.Our normalization follows classic CG →IDCG ideas (Järvelin and Kekäläinen,
2002), while departing from rank-centric discounting. UDCG similarly embraces set consumption but
models position effects and distractor harm within the prompt (Trappolini et al., 2025). Rank-centric metrics
such as nDCG/MAP/MRR/RBP emphasize user-scanning assumptions and position sensitivity (Järvelin and
Kekäläinen,2002;MoffatandZobel,2008;Manningetal.,2008). Diversitymetricslike 𝛼-nDCGandNRBP
re-weight gains to reduce redundancy across subtopics (Clarke et al., 2008, 2009); our rarity weighting is
per-query grade prevalence, not redundancy across subtopics.
7

3 CLQ Framework and rag-gs Toolkit
Thissectionoperationalizesthecost–latency–quality(CLQ)lensintroducedin§§1–2. Weformalizewhat
wemeasure,howwemeasureit,andwhythesemeasurementsmaptodeploymentdecisionsunderbudget
and SLA constraints. We then present rag-gs, an open-source pipeline that makes these measurements
reproducible across stacks.
3.1 Operator framing & scope
Problemsetting.Givenatargetretrieval-latencybudget(embed+retrieve+rerank,pre-generation)anda
costcap,practitionersmustchoose: (i)embedderandembeddingdimension,(ii)ANN/indexsettings,(iii)
candidate depth𝐾, and (iv) whether/which reranker to use.
Spenddriversandassumptions.Thedominantend-to-endcosttypicallyarisesfromfinalLLMgeneration,
which scales with the number of chunks sent and their token length. In our setup, chunks are sentence-
aligned with≈70-token overlap and include breadcrumb metadata (paper →section→subsection→
paper+authors+date). Acrossqueries,thisyields ≈515tokenspercandidateonaverage(question+text+
metadata). Consequently, retrieval choices that increase 𝐾or chunk length raise both reranking expense and
the generator’s prompt cost.
Scope.To isolate retrieval trade-offs, §3 measures only the pre-generation path—embed + retrieve +
rerank—under a fixed chunker. We discuss implications for generation latency/cost where relevant in §§5–6.
3.2 Measuring Cost, Latency, and Quality
3.2.1 Cost
What we meter.Rerankers are token-metered (Voyage AI, 2025). Let 𝑇candbe tokens per candidate (query +
chunk+metadata). Wemeasure 𝑇cand≈513–516 (§A.5). Tablesbelowassume500forclarity;adjustwith
𝑇candas needed.
Voyage reranking prices.rerank-2.5: $0.00005 / 1K tokens·rerank-2.5-lite: $0.00002 / 1K tokens.
Formula (per query).
Costrerank=𝐾×𝑇cand
1000×price1k.
Per-candidate cost at𝑇 cand=500: $0.000025 (2.5)·$0.00001 (2.5-lite).
Cost per 1,000 queries (assumes 500 tokens/candidate; Nov 2025).
K docs/query rerank-2.5 rerank-2.5-lite
50 $1.25 $0.50
100 $2.50 $1.00
150 $3.75 $1.50
200 $5.00 $2.00
Generator (input)spend scales with 𝐾and chunksize.Illustrative input-only costsforGPT-5family at 500
tokens/chunk (OpenAI, 2025).
8

Final inference input cost per 1,000 queries (illustrative; Nov 2025).
Chunks per query (𝐾) Total input tokens GPT-5 GPT-5 mini GPT-5 nano
10 5,000,000 $6.25 $1.25 $0.25
20 10,000,000 $12.50 $2.50 $0.50
30 15,000,000 $18.75 $3.75 $0.75
Notes.(i)Tablecountsinputtokensonly; outputtokensaddfurthercost. (ii)Replace500withyourmeasured
𝑇candto tighten estimates.
3.2.2 Latency (pre-generation path)
Whatwereport.Retrievallatency =embed+retrieve+rerank(generationexcluded). Unlessspecified,values
are p50. Sub-≈75ms deltas are empirically treated as jitter (network/upstream load) rather than signal.
Scaling behavior.Reranker latency grows roughly linearly with 𝐾(empirically confirmed in §5). Embedding
latency is largely insensitive to𝐾; retrieval depends on ANN/index settings.
First-token dependency (end-to-end).Generation first-token latency increases with prompt size (more/longer
chunks). Thus, choices that raise𝐾or𝑇 candinflate overall latency even when retrieval p50 is flat.
3.2.3 Quality
We use the set-based metrics from §2:
•RA-nWG@𝐾— rarity-aware, per-query–normalized weighted gain (order-free).
•N-Recall 4+@𝐾/ N-Recall 5@𝐾— normalized coverage of high-utility evidence.
We report macro-averages at𝐾=10(shallow) and𝐾=30(deep) to surface budget trade-offs.
3.3 Pareto analysis and efficiency metric
Frontier construction.We sweep (embedder×dimension×ANN×𝐾×reranker)and retain Pareto-optimal
points—configs for which no other config is strictly better in all three: Cost (C), Latency (L), and Quality (Q)
(RA-nWG@𝐾, N-Recall 4+@𝐾) (Miettinen, 1999; Deb, 2001). Dominated configs are discarded.
Operator rules.
•Latency-bound (SLA):keep{𝐿≤SLA}; among them, maximize𝑄.
•Cost-bound (budget):keep{𝐶≤cap}; among them, maximize𝑄.
•Quality-targeted:keep{𝑄≥target}; among them, minimize𝐿and𝐶.
•Tie-breaks:prefer smaller prompt size (lower 𝐾/ shorter chunks) to reduce generator cost and
first-token delay.
9

Efficiency(tie-breakeronly).Weuseasimpleaveragingheuristictoobtainasinglescoreforquickshortlisting;
however, operators should prefer task-fit metrics (and weights) aligned to their data and objectives. If
your application emphasizes shallow recall, rare evidence, or safety, adjust the metric set and/or weights
accordingly.
Efficiency=Average Performance
Latency (s),Average Performance=1
4 N-Recall 4+@10+RA-nWG@10+N-Recall 4+@30+RA-nWG@30
This scalarization is only for ranking near-frontier peers; final selection should still be made on the multi-
objective view (C/L/Q) (Das and Dennis,1997; Miettinen, 1999). Use this to rank near-frontier peers; final
choices must still satisfy binding C/L constraints.
Reporting.Retrievallatency =embed+retrieve+rerank(generationexcluded),reportedatp50(p95in§5as
supplemental). Quality macro-averaged at 𝐾∈{10,30} . Sub≈75ms latency deltas are empirically treated
as jitter.
Instantiation.We apply these rules to the full sweep; §5.4 presents representative scenarios and the efficiency
leaderboard.
3.4rag-gs: reproducible golden-set pipeline (embed →retrieve→merge→judge→prune
→rank)
rag-gsisanopen-source,MIT-licensedpipelineforbuildingcompact,high-qualitygoldensets. Itstandardizes
six stages and shared plumbing (configs, manifests, caching) so CLQ evaluation is repeatable.
•S1 Embed:rewrite queries and compute embeddings (plus BM25 query text).
•S2 Retrieve:dense cosine search and sparse BM25 to form candidate pools.
•S3Merge:reciprocalrankfusion(RRF)(Cormacketal.,2009)tounifydense+sparseintoasingle
pool.
•S4 Score:LLM-as-judge assigns 1–5 utility grades to each candidate.
•S5 Prune:retain a grade-bucketed subset sized for target𝐾budgets.
•S6 Rank:listwise refinement to a stable Top-20 using a confidence-aware Plackett–Luce (PL) update
withpairwiselocks(Luce,1959;Plackett,1975;Xiaetal.,2008;JamiesonandNowak,2011;Auer
et al., 2002; Kahn, 1962).
In short: an uncertainty-aware, listwise active-learning ranker that uses LLM judgments and a lock-DAG to
converge to a stable, globally consistent Top-𝐾of highest-utility evidence under a fixed budget.
Properties
•Active learning:prioritizes uncertain items (pool-based sampling) rather than uniform selection.
•ListwiseLLMjudging:5-itembatchesyieldcoherentrelativepreferenceswithfewercallsthanpairwise.
10

•Lock-DAG global consistency:confidence-based locks enforce acyclic constraints; global order via
topological sort.
•Efficiency:smallbatches+clippedPLmicro-updates+stabilitystoppingreduceAPIcostvs.naive
single-shot prompts.
•Scope:operates within a fixed candidate pool (no query synthesis or new data collection).
Rankingrefinement(S6): formulationAtiteration 𝑡,maintainitemscores 𝑠𝑖andinformationvalues 𝐼𝑖
used for uncertainty. The judge returns a total order over a small batch 𝐵(|𝐵|=𝑚, typically 5). We perform
a stage-wise PL update over suffixes and accumulate pairwise locks when margins are statistically clear.
Uncertainty and margin test
𝜎𝑖=1√︁
max(𝐼𝑖, 𝜀),LCB(𝑤)=𝑠 𝑤−𝑧𝜎𝑤,UCB(ell)=𝑠 ell+𝑧𝜎ell.
Lock(𝑤 >ell) ifLCB(𝑤)>UCB(ell) or after a minimum number of independent confirmations; drop
pendings implied by transitivity.
ListwisePLupdateoverajudgedorder(items 𝐴≻𝐵≻𝐶≻𝐷≻𝐸 )Foreachsuffix 𝑆𝑘ofthejudged
list(𝑘=1..𝑚), with current scores{𝑠 𝑗}:
𝑝𝑗=𝑒𝑠𝑗
Í
𝑢∈𝑆 𝑘𝑒𝑠𝑢,Δ𝑠𝑤𝑘+=𝜂(1−𝑝 𝑤𝑘),Δ𝑠𝑗≠𝑤 𝑘−=𝜂𝑝𝑗.
Accumulate Fisher-style information to shrink uncertainty:
𝐼𝑗+=𝑝𝑗(1−𝑝𝑗) (𝑗∈𝑆 𝑘).
Apply clipping|Δ𝑠𝑗|≤clip, optional inverse-sqrt decay for 𝜂, and periodic recentering 𝑠𝑗←𝑠𝑗−¯𝑠for
stability. After updating scores, recompute a global order consistent with the locked DAG via topological
sorting (ties broken by𝑠 𝑗). Stop when the Top-20 is unchanged for𝑇consecutive iterations.
Pseudo-code (Algorithm 1: S6 ranking refinement)
1Input: items with initial scores s_i (from grades), info I_i <- eps; locks <- empty
2Repeat until Top-20 stable for T turns or iteration limit:
31) Sample batch B of m items (favor low exposures / low info); ask LLM judge for
4a total order pi.
52) For each suffix S of pi:
6compute softmax p over {s_j | j in S};
7ds[pi[0]] += eta*(1 - p[pi[0]]);
8for j in S \ {pi[0]}: ds[j] -= eta*p[j];
9for j in S: I[j] += p[j]*(1 - p[j]).
10Clip ds and apply decay/recentering; update s <- s + ds; exposures[j]++ if j in B.
11

113) For each pair (w,ell) implied by pi (w ranked above ell):
12if LCB(w) > UCB(ell) or confirmations>=K or strong transitive evidence:
13add lock w->ell; clear implied pendings.
144) Compute global order via topological sort consistent with locks; snapshot Top-20.
15Return final Top-20 and scores.
3.4.1 Why iterative refinement outperforms single-shot LLM ranking
Naive approach: Ask an LLM once to “rank these 40 documents” produces:
•High variance.Despite strict, precise instructions, GPT-5 exhibits nontrivial run-to-run inconsistency,
with 3–5% rank disagreement on identical inputs (in preliminary experiments).
•No uncertainty quantification:All judgments treated as equally confident.
•Contradiction-blind:Transitive violations (A>B, B>C, C>A) go undetected.
Our S6 refinement addresses these systematically:
1.Statistical aggregation over noise:Each pairwise comparison is revisited multiple times; scores
converge via Plackett–Luce updates weighted by accumulated Fisher information. Variance decreases
as𝑂(1/√exposures).
2.Small-batch comparisons:𝑚=5items per judgment keeps cognitive load manageable.
3.Confidence-awarelocking:Onlycommitpairwisepreferenceswhenmarginexceedsstatisticalthreshold
(LCB(winner)>UCB(loser)); uncertain pairs get re-sampled.
4.Contradiction detection:Graph-based checks prevent locking inconsistent edges; forces evidence
accumulation until consistency emerges.
5.Convergencecriterion:RequiresstableTop-20for 𝑇consecutiveiterations(default 𝑇=3);prevents
premature commitment to noisy orderings.
This yields a golden set that is more reliable than individual LLM judgments, reflecting the benefit of
aggregating many small, uncertainty-aware signals.
3.5 Evaluation scope and experimental design
Corpora.Paragraph-level, domain-specific scientific papers (small corpus,≤1M passages).
Infrastructure.DenseretrievalwithVoyage/OpenAIembeddingsflat-f32,HNSW-f32/int8;sparseretrieval
with Elasticsearch BM25; hybrid via RRF (𝛼=60) merging dense+sparse top-100 prior to any reranking.
Chunking (fixed for §3).Sentence-aligned chunks with ∼70-token overlap plus breadcrumb metadata
(paper→section→subsection→paper+authors+date). Thisyields ∼515tokenspercandidateonaverage
(question + text + metadata).
12

Golden set.50 real user queries (predominantly FR, some EN) drawn from production logs. Candidate
pools built via dense+sparse; graded by LLM-as-judge (GPT-5 family) on a 5-point rubric (§2.4). Final
Top-20 per query is stabilized with Plackett–Luce listwise refinement and confidence-based locks (§3.3.1). A
20/50 subset was human-validated; the oracle agreed with the final Top-20 ordering, noting minor subtleties
around near-ties and borderline grade-3/4 items.
Label heterogeneity (motivation for per-query normalization).High variance in useful evidence across
queries: median per-query counts — grade-5: 4.0 (mean 10.7), grade-4: 4.0 (mean 7.0), grade-4 + grade-5:
10.0 (mean 17.7); median pool prevalence of grade-4 + grade-5≈12.6% (see Appendix table).
Configuration sweep.Embedders {voyage-3-large, voyage-3.5, voyage-3.5-lite} ×dimensions {512, 1024,
2048}×𝐾{50, 100, 150, 200} ×rerankers {2.5, 2.5-lite, none} ×ANN {flat-f32, HNSW-f32, HNSW-int8}.
Modes: dense-only, hybrid (RRF), hybrid + rerank.
Reproducibility.Code: rag-gs + configs at https://github.com/etidal2/rag-gs . Infrastructure:
Elasticsearch 9.1.6, macOS 26.0, 192 GB unified memory, 24-core CPU, 76-core GPU.
Road map.Section 4: diagnostic experiments ( Δ-margins, synthetic probes). Section 5: end-to-end CLQ
Pareto frontiers on this corpus.
4 Experiments & Results
4.1 Experimental setup (diagnostics vs. end-to-end)
Tasks.Controlleddiagnostics( Δ-margins).Syntheticprobesisolatingproper-namevs.topicsensitivity
via targeted ablations on the author field and formatting. Each query: “Which works by [AUTHOR] on
[TOPIC]?” paired with a 5-candidate bundle:
ID Author Topic Description
C1 Yes Yes Correct author, correct topic
C2a No Yes Wrong author (impostor A), same topic
C2b No Yes Wrong author (impostor B), same topic
C3 Yes No Correct author, different topic
C4 No No Wrong author, different topic
End-to-endretrieval.ReportN-Recall 4+@𝐾andRA-nWG@ 𝐾with/withoutqueryrewritingacrossdense,
sparse,hybrid, andrerankingstacks (e.g., ColBERT for late-interaction reranking (Khattab and Zaharia,
2020)).
Embedders.OpenAI text-embedding-3-large ,Voyage voyage-3.5 (plusasmallerOpenAImodel
for noise stress).
13

Languages.English (EN) and French (FR).
Sampling & runs. ≈100queries per language ×15 runs with fresh impostors and light template jitter.
Compute per-run means, then average across runs;between-run SDreflects stability.
Per-query margins.We computeΔ name,Δtopic,Δbothas defined in §4.2.
Symbols.𝐾= cut size;𝐶 𝐾(𝑞)= retrieved set;𝐸(𝑞)= gold evidence;𝑠(·,·)= cosine similarity.
Ablations.
•Identity-destroying:hard_name_mask,gibberish_name,edit_distance_near_miss.
•Light orthography/formatting: initials_form ,name_order_inversion ,case_punct_perturb ,
strip_diacritics,unicode_normalization_stress.
•Layout/structure:remove_label,author_position_shift.
Statistics.Diagnosticsreportedasmean ±between-runSD.End-to-endcomparisonsusepaireddesigns
(samequeries,with/withoutmanipulation)and95%CIsvianon-parametricbootstrapoverqueries(Efronand
Tibshirani, 1994); also reportoverlap@𝐾and Kendall’s𝜏(Kendall, 1938) for top-𝐾reshuffles.
Evidence-coverage KPIs.Gold set per query: 𝐸(𝑞)={𝑑|grade(𝑑)≥4} ; retrieved context: 𝐶𝐾(𝑞)
(Zheng et al., 2023).
•N-Recall 4+@𝐾:#{𝑑∈𝐶𝐾(𝑞): grade(𝑑)≥4}
min 𝐾, 𝑅 4+(𝑞),where𝑅4+(𝑞)isthecountofgrade ≥4itemsinthe
pool.
•Hit@𝐾:1
𝐸(𝑞)⊆𝐶𝐾(𝑞)
.
•Acc|Hit:downstream QA accuracyconditionedon Hit@𝐾.
Note.Hit@𝐾and Acc|Hit are tracked but not reported in §5; we use them to sanity-check the recall-first
premise and reserve full downstream QA evaluation for future work.
We first isolate proper-name sensitivity via controlled Δ-margin diagnostics (§4.2), then stress-test conversa-
tional noise (§4.3), and finally validate that these diagnostics predict end-to-end recall (§4.4).
4.2Δ-margin framework for proper names
Candidate construction (5-item bundle) and margins per query𝑞
Δname=𝑠(𝑄,𝐶 1) −max{𝑠(𝑄,𝐶 2𝑎), 𝑠(𝑄,𝐶 2𝑏)},
Δtopic=𝑠(𝑄,𝐶 1) −𝑠(𝑄,𝐶 3),
Δboth=𝑠(𝑄,𝐶 1) −𝑠(𝑄,𝐶 4).
14

Notation.𝑠(·,·)denotescosinesimilaritybetweenthequeryandcandidateembeddings. Here 𝐶1iscorrect
author + correct topic; 𝐶2𝑎,𝐶2𝑏are wrong author + correct topic impostors; 𝐶3is correct author + wrong
topic;𝐶 4is wrong author + wrong topic.
Ablations (definitions and intent)
•Base— Unmodified queries and candidates (canonical templates with natural names and topics).
•Hard name mask— Replace every author string in the bundle with the same deterministic mask (e.g.,
AUTHOR_### /AUTEUR_### )inbothqueriesandcandidates; thiserasesidentity(C1,C2a,C2bshare
an identical author token). Purpose: negative-control ablation that should collapse any name-based
margin.
•Gibberish name— Replace each author with a stable, pseudo-random token (e.g., ID-XXXXXX ) so
namesarenon-linguisticyetuniquelyconsistentacrossquery/candidates. Purpose: preserveidentity
linkage while removing natural-language surface cues (wordpiece familiarity, orthography); tests how
muchΔ namecomes from form/tokenization versus meaning.
•Edit-distancenear-miss—Forimpostorauthorsonly,mutatethetrueauthor’snamewithfixedcharacter
substitutions at preset Levenshtein distances (Levenshtein, 1966) (C2a =1, C2b=2, C4=3); C1/C3
remainunchanged. Purpose: proberobustnesstosmallorthographicperturbationsandhowquicklythe
name margin erodes as impostors become confusable.
•Remove label— Delete the explicit “Author:” / “Auteur:” label tokens from candidates (and, if present,
queries). Purpose: test reliance on structural cues (field labels) versus content; checks if models latch
onto boilerplate markers.
•Strip diacritics— Remove diacritical marks from all French strings (names, topics, full texts) via
Unicodedecomposition/recompositiontomirrorcommonnormalizationpipelines(Manningetal.,2008;
UnicodeConsortium,2025);namesoftenhavediacritic/variantformsacrosslanguages(Steinberger
et al.,2011). Purpose: assess diacriticinvariancetypical of searchnormalization andwhether accent
marks contribute to identity/topic matching.
•Initialsform—Convertauthortoinitialsstyle(e.g.,“AliceDupont” →“A.Dupont”)consistentlyin
queriesandcandidates. Purpose: examinesensitivitytonameabbreviation(familyname+initialvs.
full first name).
•Name order inversion—In candidates only, invert author orderto “Last, First” (e.g., “AliceDupont”
→“Dupont, Alice”). Purpose: check format/order invariance in entity matching.
•Case/punctuation perturbation— Apply deterministic casing (upper/lower/title) and light punctuation
normalization (curly →straight apostrophes; hyphen →space) to author strings. Purpose: measure
robustnessto text-normalizationnoise (Manning etal., 2008)sothe modelis notbrittleto superficial
variants.
•Author position shift— Reorder the candidate template so the author segment appears first (e.g.,
“Author: X.Researchpaper: ‘Y’.”). Purpose: testlayout/proximityeffects—whethersimilaritydepends
on where the author appears, not just the author string itself.
15

Condition OpenAIΔΔ name %VoyageΔΔ name %
hard_name_mask-100.0% -100.0%
gibberish_name-76.9% -68.0%
edit_distance_near_miss-69.3% -64.4%
remove_label-3.0% +15.7%
strip_diacritics-0.0% +0.0%
initials_form-11.0% -8.5%
name_order_inversion-3.6% -3.2%
case_punct_perturb-3.0% -7.4%
author_position_shift+8.4% -6.2%
unicode_normalization_stress-6.9% -15.1%
Condition OpenAIΔΔ name %VoyageΔΔ name %
hard_name_mask-100.0% -100.0%
gibberish_name-71.6% -71.4%
edit_distance_near_miss-76.2% -69.5%
remove_label+6.1% +18.1%
strip_diacritics+2.3% -1.5%
initials_form-10.7% -18.0%
name_order_inversion+5.0% -8.5%
case_punct_perturb-3.4% -12.6%
author_position_shift+21.8% +8.4%
unicode_normalization_stress-7.9% +1.7%
•Unicodenormalizationstress—NormalizequeriestoNFC;normalizecandidatestoNFD(Unicode
Consortium,2025)andinsertnarrownonbreakingspacesbeforeselectedpunctuation(e.g., : ; !).
Purpose: stressUnicode/tokenizationresiliencesoidentityandtopicsignalssurvivecross-normalization
and special whitespace.
ReportingBelow arethe measured percentagechangesin thename margin vs.base (ΔΔ%=(ablation−
base)/base)on English (EN) and French (FR).
4.3 Delta-name impact vs. base (ΔΔ%)
Table4.1.Δnameimpactvs.Base( ΔΔ%)at𝐾fixedcandidatebundles;meansover ≈100queries×15
runs.Legend. ΔΔ%=(ablation−base)/base . Positive=margin increases vs. Base; negative =decreases.
Valuesaremeansover ∼100queries×15runs;percentsummariesexcluderowswherethebase Δname<0.02
(seeAppendixA.2forabsolutedeltasandrunSDs). Absolutedeltasandper-runSDsarereportedinAppendix
A.2.
Table4.2.Δnameimpactvs.Base( ΔΔ%)at𝐾fixedcandidatebundles;meansover ≈100queries×15
runs.Legend. ΔΔ%=(ablation−base)/base . Positive=margin increases vs. Base; negative =decreases.
Valuesaremeansover ∼100queries×15runs;percentsummariesexcluderowswherethebase Δname<0.02
(seeAppendixA.2forabsolutedeltasandrunSDs). Absolutedeltasandper-runSDsarereportedinAppendix
A.2.
16

Layout asymmetries author_position_shift andremove_label show model×language-specific
effects. A plausible explanation is that OpenAI’s pretraining emphasizes front-loaded entities (e.g., head-
line/newsstyle),whileVoyage’sbi-encodermayweighpositionsmoreuniformly. TheFrenchincreases(upto
+21.8%) suggest FR corpora that favor author-first formatting. A full causal analysis is beyond scope but
merits follow-up.
Observed patterns
•Identity-destroying collapses Δname: hard mask =−100% (both languages); gibberish =−68% to
−77%(EN:−68–−77%, FR:≈−71%); near-miss edits =−64% to−76%(EN:−64–−69%, FR:
−69–−76%).
•Light formatting is largely benign: case/punct, initials, order, diacritics typically|ΔΔ%|≤∼12%.
•Layout effects are asymmetric: FR often increases Δnamewhen the author is front-loaded or labels are
removed; EN varies by model.
Reporting notePercent deltas can be unstable when the base margin is very small; for transparency we
excludequerieswith Δname(base)<0.02 frompercentsummariesandprovideabsolutedeltasinAppendix
A.2.
Formal definitions
ΔΔname=Δ(abl)
name−Δ(base)
name,ΔΔ% name=Δ(abl)
name−Δ(base)
name
Δ(base)
name×100%.
Reference.Base Name/Topic ratios Δname/Δtopicfall in0.53–0.59; see Appendix A.3.
4.3.1 Ablation families (purpose-centric grouping)
Identity-destroying (large drops inΔ name)
•Hardnamemask(−100%) : replaceallauthorstringsinabundlewiththesamemask(e.g., AUTHOR_007 ).
This removes identity entirely—C1 and impostors share the identical author token—so Δnamecollapses
by construction.
•Gibberish name(≈−68%to−77%) : unique but non-linguistic tokens kill most of the benefit of “real”
names; models no longer get semantic/orthographic cues.
•Near-miss edits(≈−64%to−76%) : small Levenshtein changes make impostors confusable; the name
margin erodes quickly.
Caveat.The−100%forhard_name_mask isbyconstruction: allcandidatessharethesamemaskedauthor
token, so the name margin collapses deterministically.
17

Mechanism: base vs. gibberishInBase, realistic names (e.g., “Manon Michel”) contain familiar
subwords/char-ngramsseenduringpretraining(capitalizedfirst/lastnames,surnamepatterns,spaces,accents).
WhenthesamenameappearsinthequeryandC1(andadifferentreal-lookingnameinC2),embeddingsgain
from string identity plus familiar morphology—yielding a healthyΔ name(Schick and Sch"utze, 2019).
InGibberish, we swap each author for a stable but non-linguistic token (e.g., ID-AB12F3 ). Tokenizers split
thisintoblandfragments(ID,-,AB,12,F3)withweak,genericembeddings. Identitylinkageispreserved
(same token in query and C1), but rich subword priors vanish; the same-string advantage becomes small.
Result:Δ namedrops a lot, but not to zero (exact-match still helps slightly).
Light noise / orthography / formatting (small effects)
•initials_form
•name_order_inversion
•case_punct_perturb
•strip_diacritics
•unicode_normalization_stress
Typically|ΔΔ%|<∼12% . Models arebroadlyrobust tocasing,punctuation, accents,NFC/NFD mismatch,
initials style, and “Last, First” inversions—these do not substantially change the name signal.
Layout / position (model×language asymmetries)
•author_position_shift,remove_label.
Moving the author earlier or removing the “Author:” label has moderate, asymmetric effects. FR often
increasesΔname(upto+21.8%OpenAIFR;+8.4%VoyageFR).ENshowsmixedreactions(e.g.,OpenAI
EN+8.4%vs.VoyageEN−6.2%whentheauthorisfront-loaded). Takeaway: documentstructurechanges
how much the author field counts in the embedding, with effects depending on model and language.
4.4 Conversational noise stress tests
Having established that embeddings carry substantial name signal (see §4.2; ratios in Appendix A.3), we now
examine a second vulnerability: conversational noise.
Inourdata,conversationalnoiselowerscosineby20–40%relativetocleanqueries;Frenchdegradesmore
thanEnglish;largermodelsare ∼20–25% morestable. Querydenoising/rewritingmitigatesqualitylossat
tight budgets.
Cosine similarity by noise level
18

Language Noise level Cosine — Large Cosine — Small
French 0 — No noise 0.818 0.936
2 — Moderate noise 0.653 0.757
4 — High noise 0.522 0.559
English 0 — No noise 0.828 0.908
2 — Moderate noise 0.749 0.806
4 — High noise 0.593 0.634
LanguageΔ— LargeΔ— Small
French−0.296−0.377
English−0.235−0.274
Examples (EN, simplified)
•No noise:“Can forests really regulate the climate?”
•Moderate:“Hi! Quick question: according to science, can forests regulate the climate?”
•High:“Hello! Sorryforthekindarandomquestion—I’monthetrain... couldforestsactuallyregulate
the climate, or is that a myth?”
Averagedropfromcleantohigh-noiseThesedropscorrespondto ∼28–40% relativetothecleancondition
(FR-Large: 36%, FR-Small: 40%, EN-Large: 28%, EN-Small: 30%), consistent with the 20–40% headline.
Why FR drops moreWe hypothesize compounding effects from richer morphology (more tokens per
filler), accent/Unicode normalization sensitivity, and code-switching prevalence, which together increase
variance in the French embedding vector under noise.
Reference.Full per-level cosine tables and drops (EN/FR, Large/Small) appear in Appendix A.4.
4.4.1 Conversational noise: why rewriting helps
Conversational “noise” (greetings, fillers, social padding) carries no task semantics but injects variance into
the embedding vector: extra tokens shift the mean representation in space. Modern embedders are robust
becausepretrainingincludesinformaltext,andattentionlearnstodown-weightpolitenessmarkers. Robust
doesnot meanimmune: longor emotionallyloaded chatter(emojis, digressions,personal context)must be
encoded somewhere in the same vector, nudging it away from the informational intent and lowering cosine to
the ideal reference. Light rewriting/denoising recenters the query on the semantic core (Ma et al., 2023) and
reduces reshuffles in Top-𝐾under fixed budgets, particularly in FR where drops are larger.
Operationally,weapplyalightweightqueryrewritingstep(“extractandrestatethecoreinformationneed”)
before embedding; §5.2 reports end-to-end results under this setting.
19

4.5 Correlation: diagnostics→Recall@𝐾
Havingquantifiednamesensitivity(§4.2)andnoisesusceptibility(§4.3),weaskwhetherthesediagnostics
predict set quality under realistic indices (Lewis et al., 2020; Izacard and Grave, 2021).
SummaryIn practice, configurations that raise N-Recall 4+@10also raise RA-nWG@ 10at similar latency
(see §§5.1–5.2). These results support using Δ-margins as fast diagnostics to prioritize mitigations (rewriting,
light lexical safeguards) before expensive sweeps.
Practicality Δ-marginscanbecomputedfromtiny5-candidatebundlesperquery,makingthemfarcheaper
than full retrieval sweeps.
Observed patternsWe observe consistent patterns: queries with degraded diagnostic margins (e.g., under
gibberish_name ) show corresponding drops in end-to-end recall. For instance, the gibberish ablation’s
−71%Δ namereduction (FR) aligns with a −0.18drop in N-Recall 4+@10(from 0.78 to 0.60 in our validation
set). Whilewedonotreportformalcorrelationcoefficientshere,thesepatternssupportusing Δ-marginsas
informative proxies for retrieval quality.
K-budget trade-offIncreasing 𝐾primarily lifts deeper-cut quality (e.g., @30) while leaving top-10
qualityroughlystableinoursetting;see§5.2forthefull 𝐾-budgetanalysis. WereportRA-nWG@ 10and
RA-nWG@30side-by-side in §§5.1–5.2 to separate shallow vs. deep-recall behavior.
5 Results and Trade-offs
5.1 Metric behavior and stack comparisons
Table 1:Dense-only sweep over model×dimension. Metrics are macro-averaged over 50 questions; NAs excluded.
Model DimN-Recall
4+
@10N-Recall
5
@10RA-
nWG
@10N-Recall
4+
@20N-Recall
5
@20RA-
nWG
@20N-Recall
4+
@30N-Recall
5
@30RA-
nWG
@30Median
Lat.
(ms)Emb
Lat.
(ms)
voyage-3-large 512 0.458 0.482 0.450 0.628 0.679 0.625 0.740 0.762 0.727 38.3 134.0
voyage-3-large 1024 0.616 0.612 0.591 0.662 0.691 0.653 0.700 0.718 0.684 50.1 137.7
voyage-3-large 2048 0.616 0.626 0.594 0.656 0.691 0.639 0.786 0.801 0.765 71.9 141.2
voyage-3.5 512 0.547 0.556 0.529 0.647 0.706 0.644 0.688 0.763 0.692 35.4 134.0
voyage-3.5 1024 0.592 0.596 0.566 0.636 0.677 0.625 0.798 0.808 0.785 47.6 138.8
voyage-3.5 2048 0.625 0.642 0.610 0.644 0.712 0.647 0.706 0.743 0.692 72.7 149.2
voyage-3.5-lite 512 0.509 0.478 0.497 0.520 0.551 0.516 0.559 0.577 0.552 38.2 133.0
voyage-3.5-lite 1024 0.502 0.469 0.483 0.541 0.563 0.537 0.591 0.611 0.578 48.8 133.7
voyage-3.5-lite 2048 0.505 0.499 0.497 0.562 0.596 0.560 0.591 0.613 0.580 71.0 140.8
Notation. N−Recall 4+≡“N-Recall4+”; N−Recall 5≡“N-Recall5”. Metrics are macro-averaged per query
over 50 questions; NAs excluded.
20

Stability.Results are averaged over 50 queries per configuration. For formal testing in IR settings,
non-parametric or randomization tests are recommended (Smucker et al., 2007).
Across models, larger dimensions generally improveshallowquality (@10), butdeep- 𝐾behavior depends on
thefamily: for voyage-3.5 ,1024dpeaksat@30(RA-nWG@30 ≈0.785),outperformingits512d/2048d
variants on this dataset, whereas forvoyage-3-large,2048d is strongest at @30 (RA-nWG@30≈0.765).
Latency rises with dimension as expected. Because these aredense-onlynumbers onoriginal questions, they
areintentionallybelowthehybrid/rerankresultslater;weusethistableasthebackbonebaselineforgains
from RRF + reranking and from query rewriting.
Table2: voyage-3.5 (1024d): retrievalmethodswithasharedbudget. Columns@15and@25areomittedbydesign.
MethodN-Recall
4+
@10N-Recall
5
@10RA-
nWG
@10N-Recall
4+
@20N-Recall
5
@20RA-
nWG
@20N-Recall
4+
@30N-Recall
5
@30RA-
nWG
@30
Dense-Only 0.592 0.596 0.566 0.636 0.677 0.625 0.798 0.808 0.785
Hybrid (RRF) 0.606 0.561 0.553 0.759 0.763 0.731 0.800 0.817 0.788
Rerank-2.5 0.835 0.810 0.804 0.799 0.834 0.794 0.819 0.833 0.810
Rerank-2.5-lite 0.799 0.772 0.767 0.791 0.821 0.784 0.814 0.824 0.799
Hybrid + Rerank-2.5 0.882 0.853 0.852 0.884 0.906 0.878 0.930 0.929 0.918
Hybrid + Rerank-2.5-lite 0.832 0.816 0.807 0.830 0.876 0.830 0.906 0.911 0.897
Reranking dominates dense/hybrid alone; hybrid + rerank provides the highest ceilings and strongest deep- 𝐾
behavior, consistent with cross-encoder re-ranking results (Nogueira and Cho, 2019).
Table 3:Hybrid ceiling (PROC) within the fixed Top-50 produced by Hybrid RRF-100 then Rerank-2.5; scores are the
oracle under perfect reordering of that pool.
Metric @10 @15 @20 @25 @30
N-Recall4+ 1.000 1.000 0.985 0.985 0.985
N-Recall5 1.000 1.000 1.000 1.000 0.996
RA-nWG 1.000 1.000 0.993 0.993 0.988
Definition.TheHybridceiling(PROC)istheoraclescoreafterrestrictingtothefixedTop-50producedby
HybridRRF-100→Rerank-2.5 (Cormacketal., 2009),i.e., perfectreorderingwithin thatpool. Under this
PROC, N-Recall 4+and RA-nWG reach ≈1.0at @10–@15 and remain ≥0.988at @30, indicating ordering
headroom that current reranking mostly, but not fully, realizes.
5.2 Dense-reranked leaderboard
Ceiling convention.In dense-reranked tables, “Ceiling” denotesPROC—Dense- 𝐾pool: the oracle within the
denseTop-𝐾poolforthatexactrow(the 𝐾poolshownintheconfiguration). Leaderboardsrankdense-base+
rerank configurations only. The best Hybrid+Rerank is shown as a reference (not ranked).
21

Table4:Referencepipeline: actualvs.PROCandpercentageofPROC.Hybrid+Rerankreference(notranked): Hybrid
RRF-100→Rerank-2.5→Top-50 —voyage-3.5(1024d).
Metric @10 (Actual / PROC / %PROC) @30 (Actual / PROC / %PROC)
RA-nWG 0.852 / 1.000 / 85.2% 0.918 / 0.988 / 92.9%
N-Recall4+ 0.882 / 1.000 / 88.2% 0.930 / 0.985 / 94.4%
Table 5:Top 5 configurations by RA-nWG@10 (Ceiling=PROC—Dense-𝐾 poolfor the row).
Rank Configuration RA-nWG@10 (Ceiling)
1 voyage-3.5 (512d) + rerank-2.5 (K=50) 0.805 (0.921)
2 voyage-3.5 (512d) + rerank-2.5 (K=200) 0.805 (0.967)
3 voyage-3.5 (1024d) + rerank-2.5 (K=50) 0.804 (0.906)
4 voyage-3.5 (512d) + rerank-2.5 (K=150) 0.798 (0.957)
5 voyage-3-large (1024d) + rerank-2.5 (K=150) 0.795 (0.959)
Table 6:Top 5 configurations by RA-nWG@30 (Ceiling=PROC—Dense-𝐾 poolfor the row).
Rank Configuration RA-nWG@30 (Ceiling)
1 voyage-3.5 (2048d) + rerank-2.5 (K=100) 0.828 (0.898)
2 voyage-3.5 (512d) + rerank-2.5 (K=100) 0.824 (0.892)
3 voyage-3.5 (1024d) + rerank-2.5 (K=100) 0.819 (0.889)
4 voyage-3.5 (1024d) + rerank-2.5 (K=200) 0.818 (0.936)
5 voyage-3.5 (512d) + rerank-2.5 (K=50) 0.817 (0.847)
Observation.Fordense-rerankedsystems, 𝐾=50yieldsthebest(ortied-best)@10quality; 𝐾=100lifts
@30withlittlechangeat@10. 512d/1024doftenmatch2048dat@10,while2048dwinssomedeep- 𝐾cases
(see§5.4forlatencyconsiderationsathigh 𝐾). Fullper-configurationtablesappearinAppendixA.7–A.8.
Dense PROC ceiling tables appear in Appendix A.9 and are the “Ceiling” values used above.
Conclusion.Wesummarizeboththedeployment-facingtakeawayandthemechanismweinferfromthe
ceilings.
Across all dense-reranked runs,rerank-2.5is the safer choice: it consistently outperformsrerank-2.5-lite,
andno“lite” variantreachesthetop-5. Forshallow quality(@10),thebest outcomescomefromDense →
rerank-2.5with 𝐾pool=50. Pushing the pool larger does raise the denseceiling(the PROC within that pool),
but it does not lift actual @10—those extra candidates mostly introducehard negativesthat the reranker must
separate from near-semantic “cousins.” Fordeep quality (@30), 𝐾pool=100is the sweet spot: it increases the
ceilingenoughtomatterwhilekeepingdistractorsincheck,yieldingbetterRA-nWG/N-Recallthan50without
theprecisiondrag(andlatencyhit)weobserveat150–200. Bydimension,512dand1024dareeffectively
tied at @10, while2048doccasionally wins at @30; pick1024das a default and move to2048donly if
deep-𝐾mattersandlatencybudgetsallow. TheHybrid+Rerankreferencebeatsdense-rerankedbecauseits
pool coverageis stronger—i.e., that gain is primarilyretrieval headroom, not just better ordering. This aligns
withpriorworkshowingneuralre-rankers’gainsarecontingentonstrongcandidatepools(NogueiraandCho,
2019; Thakur et al., 2021).
Mechanistically,theresultsseparateorderingheadroomfromretrievalheadroom. At@10,dense-reranked
configurations realize roughly83–89%of their dense ceilings; at@30, utilization rises to around ∼92%
22

overall (with a∼96.5%high for 𝐾pool=50and a∼87%dip for 𝐾pool=200). In other words, reranking
is relatively more effective at deeper cutoffs, where there are simplymore true positives to elevate, and
RA-nWG’s rarity weighting softens the penalty from a few residual distractors. This also explains the
operating points: 𝐾pool=50is ideal for @10 because the “best 10” are usually already present and adding
morecandidatesmainlyinjectshardnegativesthatcompressmargins; 𝐾pool=100winsat@30becausethe
ceilingliftsenoughtoexposeadditionalrelevantitemswithoutdrowningthererankerinnear-misses. Thelite
reranker underperforms because the token savings come at the cost ofweaker margins on hard negatives, and
those misses show up first at @10 where precision pressure is highest. Two low-effort improvements follow
directly:dedupe/near-duplicatesuppressionbeforereranktothinhardnegatives,andlightlexicalboosts(e.g.,
BM25features) forrare-signal passagestoalign withRA-nWG’srarityweighting. Longer-term, adynamic
𝐾pool—e.g., 50 for “easy” queries and 100 for “hard,” triggered by retrieval uncertainty—preserves @10
while lifting @30 without paying 200-scale costs.
5.3 Quantization and ANN effects
HNSW vs. exact flat cosine and int8 quantization (baseline:voyage-3.5, 1024d;𝑛=50queries):
Table 7:ANN/quantization comparison at 1024d. Columns @15/@25/@30 and Emb Lat. removed; deltas vs. flat
shown in parentheses.
SetupN-Recall
4+
@10N-Recall
5
@10RA-
nWG
@10N-Recall
4+
@20N-Recall
5
@20RA-
nWG
@20Median
Lat.
(ms)
flat-cos-1024 0.592 0.596 0.566 0.636 0.677 0.625 47.6
hnsw-f32 0.592 (+0.0%) 0.596 (+0.0%) 0.566 (+0.0%) 0.594 (+0.8%) 0.616 (-0.7%) 0.580 (+0.6%) 35.4 (-25.6%)
hnsw-int8-fast50 0.521 (-12.0%) 0.507 (-14.9%) 0.497 (-12.1%) 0.526 (-17.3%) 0.555 (-18.1%) 0.522 (-16.5%) 35.0 (-26.3%)
With𝑛=50queries and voyage-3.5 (1024d), the pattern is unambiguous. Moving from exactflat cosineto
HNSW(float32)(MalkovandYashunin,2020)yieldsa ∼26%dropinretrievallatencywithnomeasurable
quality loss at @10 and @20 and only tiny, non-systematic wiggles at deeper cutoffs. In contrast, under the
same aggressive search settings,int8 quantizationtrades away ∼8–18% of quality for virtually no additional
speedupbeyondHNSW-F32: retrievaltimeisessentiallythesameasfloat32HNSW,whileRA-nWGand
N-Recall 4+/5degrade across all 𝐾. Given final LLM inference dominates the end-to-end budget, those
fewmillisecondssavedattheretrievercannotcompensateforthequalityloss. Comparablememory-aware
approaches such as product/optimized product quantization often preserve recall better than naïve int8 when
tuned for high-recall ANN (Jégou et al., 2011; Ge et al., 2013; Jacob et al., 2018).
Mechanistically,thisfitsthegeometry.HNSW-F32preservesfull-precisionneighborhoods;anyapproximation
error lives in the tail of the candidate set, which set-based metrics largely tolerate—especially at @10–@20,
where the best evidence is consistently surfaced.Int8changes the vector space itself: margins between
near neighbors shrink, raisinghard-negative confusion(semantically close distractors). BecauseRA-nWGis
rarity-aware, replacing even a handful of high-utility passages with near misses is disproportionately costly.
Memory considerations (order-of-magnitude, implementation-agnostic).
•Vectorpayload.Fordimension 𝑑=1024:float32: 4×𝑑=4096 Bpervector≈3.8GiBpermillion;int8:
1×𝑑=1024 B≈0.95 GiB per million. Thus, int8 saves ∼2.9 GiB per million (about 4×compression).
23

•Graph overhead (HNSW).Independent of quantization, neighbor links and metadata add ∼0.15–
0.6 GiB per million nodes (typical 𝑀≈16–32 and 32/64-bit IDs). This component does not shrink
when you quantize vectors to int8.
•Neteffect.For10Mvectors: vectors ∼38GiB(f32)vs.∼9.5GiB(int8);addingHNSWgraph(e.g.,
+1.5–6GiB)yieldsroughly10–15GiB(int8)vs.40–44GiB(f32). ThatisasubstantialRAMreduction,
butwiththeobserved8–18%qualitylossandnegligiblespeedgain,takeitonlyunderhardmemory
constraints.
Practical guidance.Default toHNSW-F32and tune recall ( efSearch ) until the delta vs. flat at @10/@30
is≤1–2%; further tuning mostly burns latency. Treatint8as a memory lever only: if RAM is the bottleneck,
quantifythebenefitwiththeback-of-the-envelopeaboveandthenre-check%ofPROC(ceiling)toconfirm
you are not spending more in quality than you saved in hardware. If you do need stronger compression with
milder quality loss, considerlearned/product quantizationwith recall-aware search rather than blunt int8.
5.4 Latency scaling and “efficiency”
We ran the reranker three times at different times of day and report median reranker latency, averaged across
50 queries×models×dimensions×reranker tier. Under this protocol, latency scales primarily with 𝐾
(candidate count), not with embeddingdimensionality. For rerank-2.5-lite , medians increase smoothly
from roughly 340–405ms at𝐾=50to∼640–715ms at𝐾=200. For rerank-2.5 , most configurations
follow a similar trend, rising from∼331–339ms (𝐾=50) to∼580–670ms (𝐾=150–200).
Anotableexceptionis voyage-3.5 athigh𝐾: several3.5variantsexhibitalatencydiscontinuityat 𝐾≥150
(mostobviousat 𝐾=200),withmediansjumpingto ∼2.7–3.0s,whereas voyage-3-large remainsstable
around∼0.6–0.7sinthesameregime. Because 512dvs. 2048dtrackcloselyelsewhere( ≤40msdifferences
at𝐾≤100),thispatternisunlikelytobedrivenbyvectorsize. Amoreplausibleexplanationisprovider-side,
implementation-dependent non-linearities (e.g., batching thresholds, context fragmentation, throttling, or
cache/path differences) specific to certain voyage-3.5 configurations. We therefore regard the 𝐾≥150
spikeon voyage-3.5 asananomalousbehaviorthatwarrantsproviderinvestigationratherthanasaninherent
property of dimensionality or stack design.
Implications for practitioners.To keep median latency under ∼0.5s, set𝐾≤100for either reranker tier. As
complementary mitigations, reduce per-candidate tokens, deduplicate near-duplicates before rerank, or adopt
dynamic𝐾(e.g.,50for “easy” queries;100for “hard”) to stay within an SLO.
On“efficiency”asascalar.Our Efficiency=Avg(quality)/median reranker latency (s) isahandyscreening
heuristic. Moreover,singleweighted-sumscalarsareknowntoobscureParetotrade-offs(DasandDennis,
1997). But it is insensitive to SLOs and systematically favors small 𝐾, precisely where dense-reranked
systemsalreadylooksimilarat@10. Moreimportantly,intypicalRAGdeploymentstheend-to-endbudgetis
dominatedbyfinalLLMinference,withembeddingandrerankingcontributingasmaller(but 𝐾-sensitive)
share. This is consistent with retrieval-augmented generation pipelines where retrieval cost is a pre-inference
stage (Lewis et al., 2020). A single reranker-only ratio therefore overstates the operational relevance of small
latencydeltasatretrievaltime. Inthe main text,wereplacethescalarwithSLO-conditioned frontiers(e.g.,
≤350ms,≤500ms), include tail latency (p95), and report % of PROC to separate ordering gains from pool
coverage. Tail behavior is operationally critical (Dean and Barroso, 2013). Where a summary number is
useful, we recommend two stage-aware variants: (i) a retrieval-local efficiency (same definition as above, for
tuning𝐾and ANN knobs), and (ii) an end-to-end efficiency that divides Avg(quality) by total median latency
24

(embedding+retrieval+rerank+final inference) under a fixed prompt budget. Finally, we add marginal
analyses—ΔRA-nWG /Δms (e2e) when increasing 𝐾—to show where additional candidates cease to pay for
themselves.
Presentation choice.Because of (i) the non-stationarity introduced by time-of-day runs, (ii) the provider-
specifichigh- 𝐾discontinuityon voyage-3.5 ,and(iii)thelimiteddiagnosticvalueofasingle“efficiency”
scalar,wemovethedetailedlatencytables(medianrerankerlatencyfor rerank-2.5 andrerank-2.5-lite
acrossmodel/dimensionand 𝐾)totheAppendixandreferencethemfromthissection. Themainpaperretains
only (a) the methodological summary above and (b) SLO-anchored recommendations.
Appendix tables referenced:A.7 ( rerank-2.5 :𝐾∈{50,100,150,200} ) and A.8 ( rerank-2.5-lite :
𝐾∈{50,100,150,200}), “Median Reranker Latency (ms) and Metrics vs.𝐾, by model and dimension.”
5.5 Efficiency leaderboard and scenario matrix
Representative CLQ scenarios (priced at 500 tokens/doc; quality and latency measured). All costs shown are
per 1,000 queries and cover the rerank call only (tokens counted as query+doc per candidate):
Table 8:Representative CLQ scenarios with costs per 1,000 queries (rerank call only; 500 tokens/candidate).
Scenario Model (Dim) Reranker K CostLatency
msN-Recall
4+
@10RA-
nWG
@10RA-nWG
@30
Baseline voyage-3.5 (1024d) rerank-2.5 50 $1.25 332.9 0.835 0.804 0.810
Cost saver voyage-3.5-lite (1024d) rerank-2.5-lite 50 $0.50 403.8 0.710 0.692 0.732
Quality push voyage-3.5 (2048d) rerank-2.5 100 $2.50 478.1 0.815 0.791 0.828
Efficient small-dim voyage-3.5 (512d) rerank-2.5 100 $2.50 483.1 0.822 0.793 0.824
High-K check voyage-3.5 (1024d) rerank-2.5 200 $5.00 2931.1 0.815 0.792 0.818
Pricing note.Costs use 500 tokens/candidate for comparability. The measured mean was ∼515to-
kens/candidate (see §3.1), which would raise per-1k-query rerank costs by∼3%.
6 Discussion and Conclusions
RAGshouldbeevaluatedassetconsumption,notrankbrowsing. Inpracticethismeansreportingthetrio
ofRA-nWG@ 𝐾(rarity-aware,per-query–normalizedutility),N-Recall 4+@𝐾(coverageofgoodevidence),
andHarm@𝐾whenthe generatorisbrittleor order-sensitive. Thiscombinationaligns withhowcontextis
actuallyused—LLMsingestasetunderafixedpromptbudget—soscoresremaininterpretableatfixed 𝐾
andtokenlimits. Todecidewhichknobtoturnnext,werelyonpool-restrictedoracleceilings(PROC)and
therealized %PROC:whenPROCislow,theceilingitselfistheproblemandyoushouldimproveretrieval
(hybridize dense+BM25, tune ANN, add rewriting/denoising); when PROC is high but realized %PROC
lags,orderingisthebottleneckandyoushouldstrengthenrerankingandpre-rerankcleanup(near-duplicate
suppression, shorter chunks, cleaner metadata).
Operationally, we advocatedynamic- 𝐾routing: default 𝐾=50for most queries, and automatically escalate
to𝐾=100onlywhenuncertaintysignalstrigger—e.g.,smalldense-cosinemarginsamongtopcandidates,
high reranker entropy, or Δ-diagnostic drops (names/noise). This preserves @10 precision, lifts @30 recall,
25

andkeepsretrievalp50withintypicalSLAs. ThemostreliabledefaultstackisHybrid+Rerank: buildthe
poolwithDense+BM25(RRF-100)toraisetheceiling,thenapplyrerank-2.5torealizeit;reserve2.5-lite
forhardcostcapsorrecall-heavy,precision-tolerantsettings. Twolow-effortleverscompoundthesegains:
deduplicate near-duplicates before rerank and trim chunk length/metadata bloat, because with∼500tokens
per candidate,𝐾directly multiplies both rerank and generation spend and slows first-token latency.
Finally,names and noiseneed explicit guardrails. The proper-name signal is real and useful; identity-
destroyingchanges(hardmasks,gibberish,near-missedits)collapseit,whereascase,order,anddiacritics
are largely benign. Conversational padding can depress cosine by 20–40%(typically worse in FR), so make
rewrite/denoisethe default and enforceUnicode hygieneto stabilize multilingual retrieval. Methodologically,
RA-nWGisorder-freeandredundancy-agnosticbydesign;wheredistractorsorwithin-promptordermatter,
pair itwith anovelty discountand optionallyUDCG/Harm@𝐾, andtrackAcc|Hitto validatethe recall-first
premise. Tokeeptheseconclusionsportable and auditable,shipresultswithrag-gsmanifests, configs,and
per-runworkspaces,includingPROC/ %PROC andSLOs—soCLQ claimscanbe reproducedacross stacks
and over time.
Acknowledgments
This work was conducted independently, largely during personal time over the holidays, without external
funding or institutional compute. I’m grateful to the maintainers of the open-source tooling used throughout
(Elasticsearch/BM25, HNSW indices).
Limitations
Model coverage (retrievers).Most experiments use Voyage AI embedders, with a small number of OpenAI
variants for comparison. I actually have more OpenAI embedding results thanshown here, but I didn’t have
time to clean and integrate them. Major families (E5/BGE/Instructor, Cohere, Jina, Snowflake Arctic, Nomic,
mixed-breadth sentence-transformers) were not included.
Corpora & questions.The study reflects a production RAG over a science-paper corpus, using a hybrid
dense+BM25 setup. Queries are real-world and span a broad spectrum: some are very specific with a single
decisive answer; others are broad and require coverage across many relevant passages. To isolate variables, I
intentionally excluded graph-style / multi-hop aggregation questions (e.g., “How many papers has author X
written about subject Y?”) that would require Graph-RAG (counts, joins, entity resolution).
DistractorbrittlenessacrossLLMs(nextsteps).Anopenquestionisdistractorsensitivityacrossmodel
generations (e.g., Mistral-70B vs newer GPT-5 family models). A proper follow-up should quantify this.
Anecdotal production check.In a quick relevance spot-check on the production hybrid system, domain
experts rated how well answers matched the question/context: mean 4.5/5 (SD ≈0.5) over 50 responses.
Notes: ratings were by a small expert panel ( 𝑁=2, inter-rater 𝜅=0.82), so these are expert quality
judgments—not a measure of end-user satisfaction (no SUS/CSAT/NPS collected).
AI tool was used to assist with translation.
26

References
PeterAuer,NicoloCesa-Bianchi,andPaulFischer. Finite-timeanalysisofthemultiarmedbanditproblem.
Machine Learning, 47(2-3):235–256, 2002. doi: 10.1023/A:1013689704352.
Charles L. A. Clarke, Maheedhar Kolla, Gordon V. Cormack, Olga Vechtomova, Azin Ashkan, Stefan
Büttcher, and Ian MacKinnon. Novelty and diversity in information retrieval evaluation. InProceedings of
SIGIR, pages 659–666, 2008. doi: 10.1145/1390334.1390434.
Charles L. A. Clarke, Maheedhar Kolla, and Olga Vechtomova. An effectiveness measure for ambiguous
and underspecified queries. InProceedings of ICTIR, LNCS, pages 188–199. Springer, 2009. doi:
10.1007/978-3-642-23569-6_16.
Gordon V. Cormack, Charles L. A. Clarke, and Stefan B"uttcher. Reciprocal rank fusion outperforms
condorcet and individual rank learning methods. InSIGIR, 2009.
IndraneelDasandJohnE.Dennis. Acloserlookatdrawbacksofminimizingweightedsumsofobjectives.
Structural Optimization, 14:63–69, 1997. doi: 10.1007/BF01197559.
JeffreyDeanandLuizAndréBarroso. Thetailatscale.CommunicationsoftheACM,56(2):74–80,2013.
doi: 10.1145/2408776.2408794.
KalyanmoyDeb.Multi-ObjectiveOptimizationUsing EvolutionaryAlgorithms. JohnWiley &Sons, 2001.
ISBN 978-0471873396.
Bradley Efron and Robert Tibshirani.An Introduction to the Bootstrap. Chapman & Hall/CRC, Boca Raton,
FL, 1994.
Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. Optimized product quantization for approximate nearest
neighbor search. InProceedings of the IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), pages 2946–2953, 2013. doi: 10.1109/CVPR.2013.379.
GautierIzacardandÉdouardGrave. Leveragingpassageretrievalwithgenerativemodelsforopendomain
question answering. InProceedings of the 16th Conference of the European Chapter of the Association for
Computational Linguistics (EACL), pages 874–880, 2021.
Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew G. Howard, Hartwig
Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-
arithmetic-only inference. InProceedings of the IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), pages 2704–2713, 2018. doi: 10.1109/CVPR.2018.00286.
KevinG.JamiesonandRobertD.Nowak. Activerankingusingpairwisecomparisons. InAdvancesinNeural
Information Processing Systems (NeurIPS), pages 2240–2248, 2011.
KalervoJärvelinandJaanaKekäläinen. Cumulatedgain-basedevaluationofIRtechniques.ACMTransactions
on Information Systems, 20(4):422–446, 2002. doi: 10.1145/582415.582418.
Hervé Jégou, Matthijs Douze, and Cordelia Schmid. Product quantization for nearest neighbor search.IEEE
TransactionsonPatternAnalysisandMachineIntelligence,33(1):117–128,2011. doi: 10.1109/TPAMI.
2010.57.
Arthur B. Kahn. Topological sorting of large networks.Communications of the ACM, 5(11):558–562, 1962.
doi: 10.1145/368996.369025.
27

Maurice G. Kendall. A new measure of rank correlation.Biometrika, 30(1/2):81–93, 1938.
Omar Khattab and Matei Zaharia. Colbert: Efficient and effective passage search via contextualized late
interaction over bert. InProceedings of SIGIR, 2020.
V. I. Levenshtein. Binary codes capable of correcting deletions, insertions, and reversals.Soviet Physics
Doklady, 10(8):707–710, 1966.
PatrickLewis,EthanPerez,AleksandraPiktus,etal. Retrieval-augmentedgenerationforknowledge-intensive
nlp. InAdvances in Neural Information Processing Systems (NeurIPS), 2020.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. Lostinthemiddle: Howlanguagemodelsuselongcontexts.TransactionsoftheAssociationfor
Computational Linguistics, 12:157–173, 2024. URLhttps://aclanthology.org/2024.tacl-1.9/.
R. Duncan Luce.Individual Choice Behavior: A Theoretical Analysis. John Wiley & Sons, 1959.
Xinyu Ma, Zhicheng Dou, et al. Query rewriting for retrieval-augmented generation. InProceedings of
EMNLP, 2023.
Yu.A.MalkovandD.A.Yashunin. Efficientandrobustapproximatenearestneighborsearchusinghierarchical
navigablesmallworldgraphs.IEEETransactionsonPatternAnalysisandMachineIntelligence, 42(4):
824–836, 2020. doi: 10.1109/TPAMI.2018.2889473.
Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze.Introduction to Information Retrieval.
Cambridge University Press, 2008. ISBN 978-0521865715.
Kaisa Miettinen.Nonlinear Multiobjective Optimization. Kluwer Academic Publishers, 1999. ISBN
978-0792351454.
Alistair Moffat and Justin Zobel. Rank-biased precision for measurement of retrieval effectiveness.ACM
Transactions on Information Systems, 27(1):2:1–2:27, 2008. doi: 10.1145/1416950.1416952.
RodrigoNogueiraandKyunghyunCho. Passagere-rankingwithBERT.arXivpreprintarXiv:1901.04085,
2019.
R. L. Plackett. The analysis of permutations.Journal of the Royal Statistical Society: Series C (Applied
Statistics), 24(2):193–202, 1975. doi: 10.2307/2346567.
Timo Schick and Hinrich Sch"utze. Rare words: A major problem for contextualized embeddings and how to
fix it by attentive mimicking.arXiv, 2019.
Mark D. Smucker, James Allan, and Ben Carterette. A comparison of statistical significance tests for
informationretrievalevaluation. InProceedingsofCIKM,pages623–632,2007. doi: 10.1145/1321440.
1321528.
RalfSteinberger,BrunoPouliquen,MijailKabadjov,JenyaBelyaeva,andErikvanderGoot. Jrc-names: A
multilingualnamedentityresource. InProceedingsoftheInternationalConferenceRecentAdvancesin
Natural Language Processing (RANLP), pages 104–110, 2011.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. BEIR: A
heterogeneous benchmark for zero-shot evaluation of information retrieval models. InNeurIPS 2021 Track
on Datasets and Benchmarks, 2021. URLhttps://arxiv.org/abs/2104.08663.
28

Giovanni Trappolini, Florin Cuconasu, Simone Filice, Yoelle Maarek, and Fabrizio Silvestri. Redefining
retrieval evaluation in the era of llms.arXiv, 2025. URL https://arxiv.org/abs/2510.21440 .
UDCG: Utility and Distraction-aware Cumulative Gain.
Unicode Consortium. Unicode standard annex #15: Unicode normalization forms. https://www.unicode.
org/reports/tr15/, 2025. Accessed 2025-11-11.
Xinyi Wu, Alon Albalak, Liangming Pan, William Yang Wang, Colin Raffel, et al. On the emergence of
position bias in transformers.arXiv preprint arXiv:2502.01951, 2025. URL https://arxiv.org/abs/
2502.01951.
Fen Xia, Tie-Yan Liu, Jue Wang, Wensheng Zhang, and Hang Li. Listwise approach to learning to rank:
Theory and algorithm. InProceedings of ICML, pages 1192–1199, 2008. doi: 10.1145/1390156.1390306.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging
llm-as-a-judge with mt-bench and chatbot arena.arXiv, 2023. NeurIPS 2023 Datasets and Benchmarks.
A Appendix
A.1 Best Efficiency (Performance/Latency) Leaderboard (A.1)
Table 9:Best Efficiency (Performance/Latency) Leaderboard
Rank Configuration Efficiency Avg Performance Latency (ms)
1 voyage-3.5 (1024d) + rerank-2.5 (K=50) 2.454 0.817 332.9
2 voyage-3.5 (512d) + rerank-2.5 (K=50) 2.426 0.818 337.2
3 voyage-3.5 (2048d) + rerank-2.5 (K=50) 2.397 0.812 338.8
4 voyage-3-large (1024d) + rerank-2.5 (K=50) 2.362 0.782 330.9
5 voyage-3.5 (512d) + rerank-2.5-lite (K=50) 2.353 0.799 339.5
A.2 Base Condition Margins (Δ name/Δtopic/Δboth) (A.2)
Table 10:Base Condition Margins
Lang ModelΔ nameΔtopicΔboth
EN OpenAI 3L 0.175 0.305 0.486
EN Voyage 3.5 0.160 0.298 0.464
FR OpenAI 3L 0.139 0.260 0.407
FR Voyage 3.5 0.164 0.277 0.447
29

A.3 Proper-Name vs Topic Signal Ratio (A.3)
Table 11:Proper-Name vs Topic Signal Ratio
Lang Model Name/Topic Ratio
EN OpenAI 3L 0.574
EN Voyage 3.5 0.537
FR OpenAI 3L 0.535
FR Voyage 3.5 0.592
A.4 Conversational-noise cosine drops (EN/FR; large vs. small) (A.4)
Table 12:Cosine similarity by noise level
Language Noise Level Cosine — Large Cosine — Small
French 0 — No noise 0.818 0.936
2 — Moderate noise 0.653 0.757
4 — High noise 0.522 0.559
English 0 — No noise 0.828 0.908
2 — Moderate noise 0.749 0.806
4 — High noise 0.593 0.634
Table 13:Average drop from clean to high-noise (absolute cosine difference)
LanguageΔ— LargeΔ— Small
French−0.296−0.377
English−0.235−0.274
These drops correspond to ∼28–40% relative to the clean condition (FR-Large: 36%, FR-Small: 40%,
EN-Large:28%, EN-Small:30%), consistent with the20–40%headline in §4.3.
A.5 Measured Mean Tokens Per Candidate (A.5)
Table 14:Measured mean tokens per candidate (query + chunk + metadata).
Reranker Model K Mean Tokens Std Dev Configs
rerank-2.5 50 516.0 2.3 5
rerank-2.5 100 514.1 1.9 5
rerank-2.5 150 513.0 2.1 5
rerank-2.5 200 512.9 1.5 5
rerank-2.5-lite 50 516.0 2.3 5
rerank-2.5-lite 100 514.1 1.9 5
rerank-2.5-lite 150 513.0 2.1 5
rerank-2.5-lite 200 512.9 1.5 5
30

A.6 Reranker Latency Summaries (Overview) (A.6)
Table 15:Median rerank-2.5 latency (ms) vs. K, by model and dimension.
Model K=50 K=100 K=150 K=200
voyage-3-large (1024d) 330.9 469.9 582.2 667.9
voyage-3.5 (1024d) 332.9 494.1 2720.7†2931.1†
voyage-3.5 (2048d) 338.8 478.1 751.1 2970.7†
voyage-3.5 (512d) 337.2 483.1 925.7 2904.2†
voyage-3.5-lite (1024d) 330.8 476.0 615.6 2940.1†
†High-K latency discontinuity (suspected provider-side anomaly); see §5.4 for discussion.
Table 16:Median rerank-2.5-lite latency (ms) vs. K, by model and dimension.
Model K=50 K=100 K=150 K=200
voyage-3-large (1024d) 369.3 415.5 537.3 639.2
voyage-3.5 (1024d) 352.2 413.9 515.9 644.8
voyage-3.5 (2048d) 405.4 474.6 611.7 715.3
voyage-3.5 (512d) 339.5 418.2 557.3 694.0
voyage-3.5-lite (1024d) 403.8 411.3 530.5 695.8
A.7 Appendix A.8 — Reranker metrics and latency by K (rerank-2.5-lite)
Methodology.Same protocol as A.7; these appendix tables use the 15-query subsample for latency probing,
whereas the main text uses𝑛=50(see §5.1).
Table 17:rerank-2.5-lite (Reranker K=50)
Model N-Recall4+@10 N-Recall5@10 RA-nWG@10 N-Recall4+@20 N-Recall5@20 RA-nWG@20 N-Recall4+@30 N-Recall5@30 RA-nWG@30 Median Reranker Latency (ms)
voyage-3-large (1024d) 0.779 0.762 0.746 0.780 0.796 0.762 0.798 0.802 0.781 369.3
voyage-3.5 (1024d) 0.799 0.772 0.767 0.791 0.821 0.784 0.814 0.824 0.799 352.2
voyage-3.5 (2048d) 0.792 0.766 0.759 0.780 0.812 0.776 0.813 0.832 0.804 405.4
voyage-3.5 (512d) 0.799 0.779 0.769 0.793 0.837 0.792 0.817 0.848 0.811 339.5
voyage-3.5-lite (1024d) 0.710 0.732 0.692 0.711 0.780 0.713 0.737 0.776 0.732 403.8
Table 18:rerank-2.5-lite (Reranker K=100)
Model N-Recall4+@10 N-Recall5@10 RA-nWG@10 N-Recall4+@20 N-Recall5@20 RA-nWG@20 N-Recall4+@30 N-Recall5@30 RA-nWG@30 Median Reranker Latency (ms)
voyage-3-large (1024d) 0.772 0.766 0.749 0.751 0.786 0.744 0.801 0.821 0.792 415.5
voyage-3.5 (1024d) 0.792 0.766 0.759 0.760 0.787 0.750 0.801 0.800 0.782 413.9
voyage-3.5 (2048d) 0.785 0.782 0.765 0.758 0.803 0.756 0.802 0.823 0.795 474.6
voyage-3.5 (512d) 0.792 0.772 0.760 0.766 0.802 0.760 0.807 0.819 0.794 418.2
voyage-3.5-lite (1024d) 0.730 0.756 0.714 0.708 0.771 0.709 0.738 0.788 0.737 411.3
Table 19:rerank-2.5-lite (Reranker K=150)
Model N-Recall4+@10 N-Recall5@10 RA-nWG@10 N-Recall4+@20 N-Recall5@20 RA-nWG@20 N-Recall4+@30 N-Recall5@30 RA-nWG@30 Median Reranker Latency (ms)
voyage-3-large (1024d) 0.792 0.789 0.771 0.758 0.801 0.755 0.805 0.830 0.797 537.3
voyage-3.5 (1024d) 0.785 0.782 0.764 0.754 0.799 0.753 0.797 0.816 0.788 515.9
voyage-3.5 (2048d) 0.779 0.776 0.759 0.751 0.793 0.749 0.794 0.810 0.784 611.7
voyage-3.5 (512d) 0.792 0.789 0.770 0.758 0.805 0.757 0.801 0.821 0.791 557.3
voyage-3.5-lite (1024d) 0.765 0.769 0.747 0.738 0.785 0.738 0.787 0.800 0.779 530.5
31

Table 20:rerank-2.5-lite (Reranker K=200)
Model N-Recall4+@10 N-Recall5@10 RA-nWG@10 N-Recall4+@20 N-Recall5@20 RA-nWG@20 N-Recall4+@30 N-Recall5@30 RA-nWG@30 Median Reranker Latency (ms)
voyage-3-large (1024d) 0.792 0.789 0.771 0.758 0.801 0.755 0.796 0.817 0.786 639.2
voyage-3.5 (1024d) 0.785 0.782 0.764 0.754 0.799 0.753 0.803 0.813 0.792 644.8
voyage-3.5 (2048d) 0.785 0.782 0.764 0.754 0.799 0.753 0.795 0.813 0.785 715.3
voyage-3.5 (512d) 0.792 0.789 0.770 0.758 0.805 0.757 0.799 0.819 0.788 694.0
voyage-3.5-lite (1024d) 0.772 0.769 0.751 0.744 0.785 0.741 0.794 0.800 0.782 695.8
A.8Appendix A.9 — Ceiling metrics (PROC) by model and candidate pool K (dense pools)
Definition.“Ceiling” here meansPROC—Dense- 𝐾pool: the oracle score obtained by perfectly reordering the
Dense Top-𝐾poolfor that model. These ceilings are independent of the reranker tier (2.5 vs. 2.5-lite). We
label sections as “Reranker K=...” only to align with the candidate𝐾budgets used in §5.
How to read.@10/@20/@30 are evaluation cutoffs within the fixed Dense Top- 𝐾pool. Use these tables
to compute utilization in §5.2: %PROC =Actual / Ceiling. Do not average ceilings across 𝐾; compare
like-for-like𝐾only.
Scope.TheseceilingsareforDensepools. TheyarenotdirectlycomparabletotheHybridPROCin§5.1
(“Hybrid RRF-100→Rerank-2.5→Top-50”).
Ceiling Metrics by Model and Reranker K
Table 21:Reranker K=50 (Ceiling convention: PROC—Dense-𝐾 pool=50)
Model N-Recall4+@10 N-Recall5@10 RA-nWG@10 N-Recall4+@20 N-Recall5@20 RA-nWG@20 N-Recall4+@30 N-Recall5@30 RA-nWG@30
voyage-3-large (1024d) 0.919 0.860 0.875 0.839 0.846 0.829 0.839 0.826 0.813
voyage-3.5 (1024d) 0.944 0.883 0.906 0.852 0.871 0.851 0.852 0.857 0.837
voyage-3.5 (2048d) 0.944 0.890 0.911 0.852 0.878 0.856 0.852 0.861 0.840
voyage-3.5 (512d) 0.944 0.910 0.921 0.851 0.897 0.861 0.851 0.884 0.847
voyage-3.5-lite (1024d) 0.870 0.867 0.851 0.795 0.830 0.796 0.759 0.797 0.755
Table 22:Reranker K=100 (Ceiling convention: PROC—Dense-𝐾 pool=100)
Model N-Recall4+@10 N-Recall5@10 RA-nWG@10 N-Recall4+@20 N-Recall5@20 RA-nWG@20 N-Recall4+@30 N-Recall5@30 RA-nWG@30
voyage-3-large (1024d) 0.946 0.897 0.911 0.868 0.888 0.871 0.868 0.879 0.859
voyage-3.5 (1024d) 0.982 0.917 0.942 0.892 0.909 0.899 0.892 0.904 0.889
voyage-3.5 (2048d) 0.982 0.933 0.953 0.896 0.926 0.909 0.896 0.919 0.898
voyage-3.5 (512d) 0.976 0.923 0.942 0.895 0.914 0.903 0.895 0.910 0.892
voyage-3.5-lite (1024d) 0.893 0.900 0.887 0.837 0.897 0.852 0.837 0.879 0.836
Table 23:Reranker K=150 (Ceiling convention: PROC—Dense-𝐾 pool=150)
Model N-Recall4+@10 N-Recall5@10 RA-nWG@10 N-Recall4+@20 N-Recall5@20 RA-nWG@20 N-Recall4+@30 N-Recall5@30 RA-nWG@30
voyage-3-large (1024d) 0.979 0.960 0.959 0.901 0.950 0.921 0.901 0.943 0.910
voyage-3.5 (1024d) 0.989 0.940 0.956 0.911 0.931 0.920 0.911 0.929 0.911
voyage-3.5 (2048d) 0.989 0.947 0.964 0.907 0.939 0.923 0.907 0.937 0.914
voyage-3.5 (512d) 0.989 0.947 0.957 0.918 0.937 0.927 0.918 0.934 0.918
voyage-3.5-lite (1024d) 0.967 0.947 0.962 0.896 0.940 0.918 0.896 0.931 0.906
Table 24:Reranker K=200 (Ceiling convention: PROC—Dense-𝐾 pool=200)
Model N-Recall4+@10 N-Recall5@10 RA-nWG@10 N-Recall4+@20 N-Recall5@20 RA-nWG@20 N-Recall4+@30 N-Recall5@30 RA-nWG@30
voyage-3-large (1024d) 0.979 0.960 0.959 0.905 0.950 0.923 0.905 0.946 0.913
voyage-3.5 (1024d) 1.000 0.953 0.976 0.933 0.944 0.944 0.933 0.944 0.936
voyage-3.5 (2048d) 0.989 0.953 0.965 0.922 0.944 0.933 0.922 0.944 0.926
voyage-3.5 (512d) 0.989 0.967 0.967 0.929 0.956 0.941 0.929 0.956 0.933
voyage-3.5-lite (1024d) 0.980 0.960 0.975 0.913 0.953 0.935 0.913 0.951 0.927
32
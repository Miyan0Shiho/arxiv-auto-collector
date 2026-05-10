# CAR: Query-Guided Confidence-Aware Reranking for Retrieval-Augmented Generation

**Authors**: Zhipeng Song, Yizhi Zhou, Xiangyu Kong, Jiulong Jiao, Xuezhou Ye, Chunqi Gao, Xueqing Shi, Yuhang Zhou, Heng Qi

**Published**: 2026-05-06 04:51:28

**PDF URL**: [https://arxiv.org/pdf/2605.04495v1](https://arxiv.org/pdf/2605.04495v1)

## Abstract
Retrieval-Augmented Generation (RAG) depends on document ranking to provide useful evidence for generation, but conventional reranking methods mainly optimize query-document relevance rather than generation usefulness. A relevant document may still introduce noise, while a lower-ranked document may better reduce the generator's uncertainty. We propose CAR (Confidence-Aware Reranking), a query-guided, training-free, and plug-and-play reranking framework that uses generator confidence change as a document usefulness signal. CAR estimates confidence through the semantic consistency of multiple sampled answers under query-only and query-document conditions. Documents that significantly increase confidence are promoted, those that decrease confidence are demoted, and uncertain cases preserve the baseline order, while a query-level gate avoids unnecessary intervention on already confident queries. Experiments on four BEIR datasets show that CAR consistently improves NDCG@5 across sparse and dense retrievers, LLM-based and supervised rerankers, and four LLM backbones. Notably, CAR improves the YesNo reranker by 25.4 percent on average under Contriever retrieval, and its ranking gains strongly correlate with downstream generation F1 improvements, achieving Spearman rho = 0.964.

## Full Text


<!-- PDF content starts -->

CAR: Query-Guided Confidence-Aware Reranking for
Retrieval-Augmented Generation
Zhipeng Songa, Yizhi Zhoub, Xiangyu Kongc, Jiulong Jiaoa,d, Xuezhou Yea, Chunqi Gaoa,
Xueqing Shie, Yuhang Zhoufand Heng Qia,∗
aSchool of Computer Science and Technology, Dalian University of Technology, No.2 Linggong Road, Ganjingzi District, Dalian, 116024, China
bSchool of Information Engineering, Dalian Ocean University, No. 2-52, Heishijiao Street, Shahekou District, Dalian, 116023, China
cSchool of Information Engineering, Liaodong University, No.116 Linjiang Back Street, Zhenan District, Dandong, 118001, China
dInformation Technology Center, Qinghai University, 251 Ningda Road, Chengbei District, Xining, 810016, China
eCollege of Health-Preservation and Wellness, Dalian Medical University, No. 9 West Section of Lvshun South Road, Lvshunkou
District, Dalian, 116044, China
fTencent (Dalian Northern Interactive Entertainment Technology Co., Ltd.), 21/F, Tencent Building, No. 26 Jingxian St, Ganjingzi
District, Dalian, 116085, China
ARTICLE INFO
Keywords:
large language models
retrieval-augmented generation
reranking
model uncertaintyABSTRACT
Retrieval-Augmented Generation (RAG) depends on document ranking to provide useful ev-
idence for generation, but conventional reranking methods mainly optimize query–document
relevance rather than generation usefulness. A relevant document may still introduce noise,
whilealower-rankeddocumentmaybetterreducethegenerator’suncertainty.WeproposeCAR
(Confidence-AwareReranking), a query-guided, training-free, and plug-and-play reranking
framework that uses generator confidence change as a document usefulness signal. CAR
estimatesconfidencethroughthesemanticconsistencyofmultiplesampledanswersunderquery-
only and query–document conditions. Documents that significantly increase confidence are
promoted,thosethatdecreaseconfidencearedemoted,anduncertaincasespreservethebaseline
order, while a query-level gate avoids unnecessary intervention on already confident queries.
Experiments on four BEIR datasets show that CAR consistently improves NDCG@5 across
sparse and dense retrievers, LLM-based and supervised rerankers, and four LLM backbones.
Notably, CAR improves the YesNo reranker by25.4%on average under Contriever retrieval,
anditsrankinggainsstronglycorrelatewithdownstreamgenerationF1improvements,achieving
Spearman𝜌=0.964.
1. Introduction
Retrieval-AugmentedGeneration(RAG)hasbecomeaneffectiveparadigmformitigatingfactualhallucinationsin
largelanguagemodels(LLMs)foropen-domainquestionanswering,factverification,anddomain-specificreasoning,
byincorporatingexternalknowledgeintothegenerationprocessLewisetal.(2020).InaRAGsystem,theretrievaland
reranking modules determine which documents are ultimately provided to the generator. Consequently, the quality of
the ranked document list directly affects the reliability of downstream generation. In general, more accurate rankings
provide more sufficient evidence to the generator, thereby increasing the likelihood that the generated answer is
factually correct.
Most existing retrieval and reranking methods take query–document relevance as their primary optimization
objective. That is, they aim to determine whether a candidate document matches the query at the lexical or semantic
levelandthenadjusttherankingaccordingly.However,inRAGscenarios,relevancedoesnotnecessarilycoincidewith
generation usefulness. A document that appears superficially relevant may contain noise, ambiguity, or information
that conflicts with the intended answer direction, causing the generator to produce unstable or incorrect responses.
∗Corresponding author.
songzhipeng@mail.dlut.edu.cn(ZhipengSong);zhouyizhi@dlou.edu.cn(YizhiZhou);xiangyukong@liaodongu.edu.cn(
Xiangyu Kong);jiaojiulong@mail.dlut.edu.cn( Jiulong Jiao);yexzh6@mail2.sysu.edu.cn( Xuezhou Ye);
gaochunqi@mail.dlut.edu.cn( Chunqi Gao);shixq@dmu.edu.cn( Xueqing Shi);ginozhou@tencent.com( Yuhang Zhou);
hengqi@dlut.edu.cn( Heng Qi)
ORCID(s):0009-0009-6249-1988( Zhipeng Song);0000-0002-6761-5953( Yizhi Zhou);0000-0003-1940-8674( Xiangyu Kong);
0009-0001-9852-7999( Jiulong Jiao);0009-0002-3421-5889( Xuezhou Ye);0009-0008-8576-3141( Chunqi Gao);
0009-0008-3754-6396( Xueqing Shi);0000-0002-8770-3934( Heng Qi)
Song et al.:Preprint submitted to ElsevierPage 1 of 22arXiv:2605.04495v1  [cs.CL]  6 May 2026

CAR
Conversely,adocumentrankedlowerbytheretriever,despiteweaklexicaloverlapwiththequery,mayprovidecrucial
evidencethathelpsthegeneratorformaconsistentandcorrectjudgment.Therefore,relyingsolelyonrelevancesignals
from the retriever or reranker is insufficient to fully characterize a document’s actual contribution to the generation
process Song, Zhou, et al. (2026).
ThisobservationsuggeststhatrerankingforRAGshouldnotonlyaskwhetheradocument“appearsrelevant,”but
shouldfurtherassesswhetherthedocumentcanreducethegenerator’suncertaintyforthecurrentquery.Thesampling
behaviorofLLMsprovidesanaturalsignalforthispurpose.Whenamodelisconfidentunderagiveninputcondition,
repeatedlysampledanswerstendtobesemanticallyconsistent.Incontrast,whenthemodellackssufficientevidenceor
facesmultipleplausibleanswers,thesampledanswerstendtodisperseacrossseveralsemanticclustersFarquharetal.
(2024). Thus, the semantic consistency of sampled answers can serve as an externally observable proxy for generator
confidence, measuring the model’s certainty under a given query or query–document condition Song, Kong, et al.
(2026).
Based on this observation, we recast RAG reranking as a confidence-guided posterior correction problem. The
initial ranking produced by a baseline retriever or reranker can be viewed as aprior preferenceover document
usefulness,whiletheconfidencechangeexhibitedbythegeneratorafterconditioningonacandidatedocumentprovides
posterior evidencefrom the generator’s perspective. If introducing a document makes the generator’s answers more
consistent,thedocumentlikelyprovideseffectiveevidenceandshouldbepromoted.Ifintroducingthedocumentmakes
thegeneratorlesscertain,thedocumentlikelyintroducesnoiseorinterferenceandshouldbedemoted.Therefore,RAG
reranking can be understood as a conservative posterior correction process: the baseline ranking is respected as the
prior structure, and local corrections are applied only when the generator provides sufficiently strong evidence.
However, such correction should not be applied uniformly to all queries. For high-confidence queries, where the
generator can already produce stable answers without external documents, the original retriever or reranker often
provides a reliable ranking, and unnecessary intervention may disrupt an already reasonable ranking structure. For
low-confidence queries, candidate documents have a greater impact on the generator’s judgment, making document-
conditioned confidence signals more informative for rank correction. Therefore, an ideal RAG post-processing
rerankingmethodshouldsatisfytworequirementssimultaneously.First,itshouldidentifywhichdocumentsincreaseor
decreasethegenerator’sconfidence.Second,itshouldremainsufficientlyconservativetoavoidexcessivelymodifying
the baseline ranking when correction is unnecessary.
To this end, we proposeCAR(Confidence-AwareReranking), a query-guided, training-free, and plug-and-play
confidence-aware reranking framework. CAR first estimates query-only confidence through the semantic consistency
of multiple sampled answers from the generator, thereby determining whether the current query requires posterior
correction. For low-confidence queries, CAR further estimates the conditional confidence of each query–document
inputandcomparesitwiththequery-onlyconfidenceusingamargin-basedcriterion.Basedontheresultingconfidence
change,candidatedocumentsarepartitionedintopromote,preserve,anddemotebins.Thefinalrankingisobtainedby
concatenating these bins while strictly preserving the baseline’s original relative order within each bin.
This design endows CAR with three important properties. First, CAR isquery-guided: it uses query-level
confidence as a gating signal and initiates reranking correction only when the generator is uncertain. Second, CAR is
conservative: it neither retrains the ranking model nor completely overrides the baseline results, but instead performs
bin-level adjustments only when the generator-side posterior evidence is sufficiently strong. Third, CAR isplug-and-
play: it does not rely on calibrated relevance scores from the baseline, nor does it require access to model internals
or additional training. Consequently, CAR can be applied on top of sparse retrievers, dense retrievers, LLM-based
rerankers, and supervised neural rerankers.
WesystematicallyevaluateCARonfourrepresentativedatasetsfromtheBEIRbenchmark,coveringopen-domain
questionanswering,factverification,scientificliteratureretrieval,andbiomedicalretrieval.Experimentalresultsshow
that CAR delivers consistent gains across both BM25 and Contriever retrievers, seven types of baseline methods,
andfourLLMbackbonemodels.Furtherablationstudiesvalidatethenecessityofthequerythresholdandconfidence
margin:theformerpreventsexcessiveinterventiononstrongbaselines,whilethelatterreducesmisclassificationcaused
bysamplingfluctuations.End-to-endgenerationexperimentsfurtherdemonstratethatCAR’sNDCG@5improvements
effectivelytransfertodownstreamgenerationquality,withastrongpositivecorrelationbetweenrankingimprovements
and generation F1 improvements.
Our main contributions are as follows:
Song et al.:Preprint submitted to ElsevierPage 2 of 22

CAR
Inside CAR (Confidence-Aware Reranking)
Figure 1: Overview of the proposed CAR framework.Given a user query, an initial retriever first returns a top-𝐾candidate
list based on similarity, and an optional reranker further refines it into a top-𝑁list based on relevance. CAR then
performs confidence-aware post-processing from the generator’s perspective. It first estimates the query-only confidence by
sampling multiple answers from the LLM and clustering them semantically. If the query confidence is sufficiently high, CAR
preserves the original ranking. Otherwise, CAR measures document-conditioned confidence for each candidate document,
computes the confidence differenceΔ(𝑞,𝑑) = Conf(𝑞,𝑑)−Conf(𝑞), and assigns documents into promote, preserve, or
demote bins according to the confidence margin. Documents that increase generator confidence are promoted, documents
with insignificant confidence changes preserve their relative order, and documents that reduce confidence are demoted.
The final ranking is obtained through order-preserving binning, providing a conservative plug-and-play reranking correction
for RAG.
•We propose CAR, a query-guided confidence-aware reranking framework.To address the mismatch
betweendocumentrelevanceandgenerationusefulnessinRAGscenarios,weproposeCAR(Confidence-Aware
Reranking).CARtreatsthebaselinerankingasapriorpreferenceoverdocumentusefulness,usestheconfidence
difference between query-only and query–document conditions as posterior usefulness evidence from the
generator,andcontrolsboththetriggeringconditionandtheevidencestrengthofposteriorcorrectionthrougha
querythresholdandaconfidencemargin.CARultimatelypartitionscandidatedocumentsintopromote,preserve,
and demote bins, preserving the baseline’s original relative order within each bin and thereby implementing a
conservative Bayesian-style posterior reranking mechanism.
•WevalidateCAR’sdeploymentfriendlinessandcross-settinggeneralizability.CARisablack-box,training-
free,andplug-and-playpost-processingmodulethatrequiresnoaccesstomodelinternals,noadditionaltraining,
and no modification to the underlying retriever or reranker. Experimental results demonstrate that CAR can be
stablyappliedontopofsparseretrievalwithBM25,denseretrievalwithContriever,LLM-basedrerankers,and
supervised neural rerankers, while producing consistent gains across four LLM backbone models.
•We show that confidence-aware reranking improves generation quality in the full RAG pipeline.Beyond
retrieval ranking metrics, we further conduct end-to-end generation experiments. Results show that CAR’s
NDCG@5 improvements are strongly correlated with downstream generation F1 improvements (Spearman
𝜌=0.964),indicatingthatrankingcorrectionbasedongeneratorconfidencenotonlyimprovesretrievalquality
but also effectively enhances the output quality of the full RAG system.
Song et al.:Preprint submitted to ElsevierPage 3 of 22

CAR
The remainder of this paper is organized as follows. Section 2 reviews related work on RAG reranking, LLM
confidence estimation, and uncertainty-aware retrieval. Section 3 presents the detailed design of CAR, including
generator-side confidence estimation, the query threshold, the confidence margin, and the order-preserving binning
reranking mechanism. Section 4 reports the experimental setup, main results, ablation analysis, cross-retriever and
cross-model experiments, and end-to-end generation results. Finally, Section 5 summarizes the main findings and
discusses practical implications, limitations, and future directions.
2. Related Work
2.1. RAG and Document Reranking
Retrieval-Augmented Generation (RAG) has become a mainstream paradigm for improving the factual reliability
oflargelanguagemodelsGaoetal.(2023).InatypicalRAGpipeline,aretrieverfirstselectscandidatedocumentsfrom
an external corpus, and an optional reranking stage then refines the initial ranking before the selected documents are
fed to the generator. Since the ranked document list determines what evidence is available to the generator, reranking
plays a critical role in the overall performance of RAG systems.
Traditionalrerankingmethodscanbebroadlydividedintotwofamilies.Supervisedneuralrerankersaretrainedon
labeledrelevancedatatoproducefine-grainedquery–documentrelevancejudgments.Representativemethodsinclude
ColBERT Khattab & Zaharia (2020), which performs late interaction over contextualized token embeddings; Cross-
Encoder architectures Reimers & Gurevych (2019), which jointly encode the query and document for deep relevance
matching;andRankT5Zhuangetal.(2023),whichfine-tunesT5withrankinglossesfortextranking.Thesemethods
achieve strong ranking performance on standard benchmarks, but they require task-specific training data and do not
explicitly account for the downstream generator’s behavior.
More recently,LLM-based rerankershave emerged as training-free alternatives. RankGPT Sun et al. (2023)
leverages GPT-4 to perform listwise reranking through permutation-based prompting and a sliding-window strategy.
RankVicunaPradeepetal.(2023)distillsthisrerankingcapabilityintoanopen-sourcemodel.YesNoQinetal.(2024)
prompts an LLM to output binary relevance judgments, while QLM Sachan et al. (2022) estimates the likelihood of
thequerygivenadocument.ThesemethodsrelyonLLMpromptingwithoutadditionaltraining,buttheystilloptimize
query–documentrelevancerather thangeneration usefulness.
Aseparatelineofworkseekstounifyrankingandgeneration.RankRAGYuetal.(2024)fine-tunesasingleLLM
to perform both context ranking and answer generation, enabling the model to assess document usefulness from the
generator’sperspective.However,thisapproachrequiresinstructionfine-tuningonspecifictasksandcannotbedirectly
used as a plug-and-play module on top of arbitrary retrieval or reranking systems.
CAR differs from the above methods in a fundamental way. Rather than defining usefulness through lexical
matching, semantic matching, or relevance prompting, CAR uses the generator’s ownconfidence changeas evidence
of document usefulness. Importantly, CAR is orthogonal to existing rerankers and can be stacked on top of them as a
post-processing module.
2.2. LLM Confidence Estimation
Estimating the confidence of large language models has attracted growing attention, as confidence signals are
essential for hallucination detection, selective prediction, and downstream decision-making.
One prominent approach leverages theconsistency of sampled answers. Self-consistency Wang et al. (2023)
generates multiple reasoning paths through temperature-based sampling and selects the most frequent answer by
majority vote, implicitly using answer agreement as a confidence indicator. Semantic entropy Farquhar et al. (2024)
extendsthisideabyclusteringsemanticallyequivalentanswersandcomputingentropyovertheresultingdistribution,
providing a principled black-box uncertainty estimator that does not require access to model logits.
Anotherlineofworkinvestigatesverbalizedconfidence.Kadavathetal.Kadavathetal.(2022)showthatLLMscan
providereasonablycalibratedprobabilityestimateswhenaskedtoassesswhethertheyknowtheanswertoaquestion.
Xiong et al. Xiong et al. (2024) systematically evaluate confidence elicitation methods and find that, although LLMs
canexpressuncertaintythroughverbalization,theytendtobeoverconfident,makingsampling-basedapproachesmore
reliable in black-box settings.
CAR builds on the sampling-based confidence estimation paradigm. Similar to semantic entropy, CAR uses
bidirectionalentailmenttoidentifysemanticallyequivalentanswersanddefinesconfidenceaccordingtothemaximum
semantic-cluster proportion. However, CAR applies this estimator to a different objective: instead of detecting
Song et al.:Preprint submitted to ElsevierPage 4 of 22

CAR
hallucinations, it measureshow a candidate document changes the generator’s confidence. By comparing query-only
confidencewithquery–documentconfidence,CARobtainsadocument-specificposteriorevidencesignalthatdirectly
informs reranking decisions.
2.3. Uncertainty-Aware Retrieval and Generation
Several recent methods incorporate model uncertainty or confidence signals into the retrieval process, although
their objectives differ from that of CAR.
Self-RAGAsaietal.(2024)trainsanLLMtoemitspecialreflectiontokensthatindicatewhenretrievalisneeded,
whether retrieved passages are relevant, and whether the generated output is supported by evidence. This enables
adaptive retrieval based on the model’s self-assessment, but requires modifying the LLM through instruction fine-
tuning. FLARE Jiang et al. (2023) performs forward-looking active retrieval by triggering document search during
generation when the model produces low-confidence tokens. CRAG Yan et al. (2024) designs a lightweight retrieval
evaluator that assesses retrieval quality and triggers corrective actions, such as web-search fallback, when confidence
in the retrieved documents is low.
Thesemethodssharetheintuitionthatmodeluncertaintyshouldguideretrievalbehavior,buttheymainlyfocuson
whetherandwhento retrieve, rather than onhow to rankalready-retrieved documents. In contrast, CAR operates at
the reranking stage: it takes a given candidate document list as input and reorders documents according to how each
document changes the generator’s confidence. This makes CAR complementary to adaptive retrieval methods. CAR
can be applied on top of any retrieval strategy, whether fixed or adaptive.
Unlike prior work that primarily focuses on retrieval triggering or query–document relevance assessment, CAR
explicitly uses generator-side confidence changes as posterior evidence for reranking in RAG, while maintaining a
conservative, training-free, and plug-and-play design.
3. Methodology
ThissectionpresentsCAR(Confidence-AwareReranking),aquery-guided,training-free,andplug-and-playpost-
hoc reranking module. We recast the reranking problem from the perspective of Bayesian-style ranking updating: the
ranking produced by a retriever or reranker reflects a prior preference over candidate documents from the ranking
model’s perspective, while the generation confidence exhibited by the generator when conditioned on a query and
document provides posterior evidence from the generation model’s perspective. The goal of CAR is to leverage
generator confidence to perform conservative posterior correction of the baseline ranking, without modifying the
underlying retrieval or reranking system.
Specifically, let the original retriever or reranker be𝜓, and let the generator be𝜙. For a query𝑞and a candidate
document𝑑,therelevancejudgmentproducedbythebaselinemodel𝜓canbeunderstoodasapriordistributionover
document usefulness, denoted by𝑃𝜓(𝑑∣𝑞). In practice, some baselines explicitly output relevance scores𝑠𝜓(𝑞,𝑑),
from which prior probabilities can be obtained through normalization. For models that only output rankings without
scores, CAR directly treats the permutation order as the prior preference.
On the other hand, the generator𝜙produces answers given the query and document. If a document𝑑helps the
generator produce more stable and consistent answers, then𝑑has higher usefulness from the generator’s perspective.
We denote this conditional usefulness signal as𝑃𝜙(𝑢=1∣𝑞,𝑑), where𝑢=1indicates that document𝑑is useful for
generator𝜙in answering query𝑞. Ideally, the posterior ranking of candidate documents can be expressed as:
𝑃(𝑑∣𝑞,𝑢=1;𝜓,𝜙)∝𝑃𝜙(𝑢=1∣𝑞,𝑑)𝑃𝜓(𝑑∣𝑞).(1)
This expression shows that the final ranking should incorporate information from two sources: the prior relevance
signal provided by the baseline ranker𝜓, and the posterior usefulness evidence provided by the generator𝜙.
However, CAR does not directly estimate the continuous value of𝑃𝜙(𝑢= 1 ∣𝑞,𝑑), nor does it require the
baselinetooutputcalibratedprobabilityscores.Instead,CARadoptsaconservativeanddiscretizedposteriorcorrection
mechanism. It estimates query-only confidence and query–document confidence through the semantic consistency of
multiple sampled answers from the generator, and determines whether a document provides sufficiently positive or
negative posterior evidence according to the difference between the two. CAR adjusts a document’s bin only when
significant posterior evidence is observed, and preserves the baseline’s original relative order within each bin.
Song et al.:Preprint submitted to ElsevierPage 5 of 22

CAR
Table 1
Summary of the main notation used in CAR.The table defines the symbols for the baseline ranker, generator, confidence
estimates, correction labels, and posterior bins used in the methodology.
Symbol Description
𝜓Baseline ranker (retriever or reranker)
𝜙Generator (large language model)
𝑞Query
𝑑Candidate document
𝑛Number of candidate documents
𝑘Number of samples per input
𝜋𝜓
𝑞Baseline𝜓’s prior permutation of candidates
̂ 𝜋𝑞 New permutation after CAR reranking
𝑃𝜓(𝑑∣𝑞)Prior probability from baseline𝜓
𝑃𝜙(𝑢=1∣𝑞,𝑑)Conditional usefulness probability from generator𝜙
𝐴𝜙
𝑥Set of sampled answers for input𝑥
𝑍𝜙
𝑥Semantic cluster labels for input𝑥
𝑐𝜙
𝑞Query-only confidenceConf𝜙(𝑞)
𝑐𝜙
𝑞,𝑑Query-document confidenceConf𝜙(𝑞,𝑑)
Δ𝜙(𝑞,𝑑)Confidence change𝑐𝜙
𝑞,𝑑−𝑐𝜙
𝑞
𝑇𝑞 Query threshold (QT)
𝑚Confidence margin (CM)
𝑏𝜙(𝑞,𝑑)Posterior correction label,∈{+1,0,−1}
+
𝜙/0
𝜙/−
𝜙Promote / preserve / demote bin
3.1. Problem Definition
Given a query𝑞, the baseline ranker𝜓returns a candidate document permutation𝜋𝜓
𝑞= [𝑑1,𝑑2,…,𝑑𝑛], which
represents𝜓’s prior ranking preference over the candidate documents. If𝜓outputs relevance scores𝑠𝜓(𝑞,𝑑), the
ranking prior can be formally defined as:
𝑃𝜓(𝑑𝑖∣𝑞)=exp(𝑠𝜓(𝑞,𝑑𝑖))
∑𝑛
𝑗=1exp(𝑠𝜓(𝑞,𝑑𝑗)).(2)
If𝜓only outputs a ranking without scores, CAR does not explicitly compute this probability. Instead, it treats the
permutation𝜋𝜓
𝑞itself as the prior order, i.e.,𝑑𝑖≻𝜓𝑑𝑗if and only ifrank𝜓(𝑑𝑖)<rank𝜓(𝑑𝑗).
Thegoalofthisworkistoconstructanewrankinĝ 𝜋𝑞byintroducingposteriorconfidencesignalsfromthegenerator
𝜙, without retraining𝜓or modifying its internal structure. From a probabilistic perspective, the ideal objective is to
perform posterior updating of the baseline prior based on document usefulness from the generator’s viewpoint, as
formalized in Eq. 1.
Here,𝑃𝜓(𝑑∣𝑞)comes from the retriever or reranker’s perspective and represents the prior probability that the
documentisrelevanttothequeryorshouldberankedhigher.Incontrast,𝑃𝜙(𝑢=1∣𝑞,𝑑)comesfromthegenerator’s
perspective and represents the posterior correction signal indicating whether document𝑑helps generator𝜙produce
stable answers for query𝑞.
Since𝑃𝜙(𝑢= 1 ∣𝑞,𝑑)is difficult to observe directly, CAR uses the consistency of the generator’s sampled
answersasaproxyestimate.Specifically,CARestimatesquery-onlyconfidence𝑐𝜙
𝑞=Conf𝜙(𝑞)andquery–document
confidence𝑐𝜙
𝑞,𝑑=Conf𝜙(𝑞,𝑑),where𝑐𝜙
𝑞denotestheconfidenceofgenerator𝜙onquery𝑞withoutexternaldocuments,
and𝑐𝜙
𝑞,𝑑denotes the conditional confidence of generator𝜙given document𝑑. Their difference,Δ𝜙(𝑞,𝑑)=𝑐𝜙
𝑞,𝑑−𝑐𝜙
𝑞,
is used to approximate the posterior impact of document𝑑on generator𝜙.
IfΔ𝜙(𝑞,𝑑)issignificantlypositive,document𝑑improvestheconsistencyofthegenerator’sanswers,corresponding
toalargevalueof𝑃𝜙(𝑢=1∣𝑞,𝑑),andshouldbepromoted.IfΔ𝜙(𝑞,𝑑)issignificantlynegative,document𝑑reduces
theconsistencyofthegenerator’sanswers,correspondingtoasmallvalueof𝑃𝜙(𝑢=1∣𝑞,𝑑),andshouldbedemoted.
If the difference is not significant, the posterior evidence is considered insufficient, and the prior ranking of𝜓should
be preserved.
Song et al.:Preprint submitted to ElsevierPage 6 of 22

CAR
3.2. Generator-side Posterior Confidence Estimation
CAR uses the generator𝜙’s multiple-sampling behavior to estimate the posterior usefulness signal of documents.
The core assumption is as follows: if the generator produces semantically consistent answers under a given input
conditionacrossmultiplesamples,thentheinputprovidesthegeneratorwithhighcertainty.Conversely,ifthesampled
answersdisperseacrossmultiplesemanticclusters,thegeneratorexhibitshighuncertaintyunderthatinputcondition.
For each query𝑞, CAR constructs two types of inputs:
•Query-only input.Only the query𝑞is provided. This input is used to estimate the generator𝜙’s baseline
confidence𝑐𝜙
𝑞without external documents.
•Query–document input.The query𝑞and a candidate document𝑑are provided together. This input is used to
estimate the generator𝜙’s conditional confidence𝑐𝜙
𝑞,𝑑under the document condition.
For any input𝑥, the generator𝜙is sampled𝑘times to obtain the answer set𝐴𝜙
𝑥= {𝑎1,𝑎2,…,𝑎𝑘}. CAR then
performs semantic clustering over these answers to obtain semantic cluster labels𝑍𝜙
𝑥={𝑧1,𝑧2,…,𝑧𝑘}.
Todeterminewhethertwoanswersaresemanticallyequivalent,weadoptastrictbidirectionalentailmentcriterion.
Given two answers𝑎𝑖and𝑎𝑗, they are considered to belong to the same semantic cluster only when both𝑎𝑖→𝑎𝑗and
𝑎𝑗→𝑎𝑖are judged as entailment.
Letthe𝑗-thclustercontain𝑛𝑗answersamongatotalof𝑘samples,andletitsproportionbe𝑝𝑗=𝑛𝑗∕𝑘.CARdefines
the maximum cluster proportion as the confidence of generator𝜙on input𝑥:
Conf𝜙(𝑥)=max
𝑗𝑝𝑗. (3)
Thus, the query-only confidence and query–document confidence are respectively𝑐𝜙
𝑞= Conf𝜙(𝑞)and𝑐𝜙
𝑞,𝑑=
Conf𝜙(𝑞,𝑑).
From a Bayesian-style interpretation,𝑐𝜙
𝑞,𝑑serves as a proxy signal for𝑃𝜙(𝑢= 1 ∣𝑞,𝑑), while𝑐𝜙
𝑞provides a
query-specific reference point for the current query without documents. CAR does not directly use the absolute value
of𝑐𝜙
𝑞,𝑑.Instead,itusestheincreaseordecreaseof𝑐𝜙
𝑞,𝑑relativeto𝑐𝜙
𝑞todeterminewhetheradocumentprovidespositive
or negative posterior evidence.
3.3. Confidence-Aware Bayesian-style Reranking
After obtaining the baseline prior ranking𝜋𝜓
𝑞and the generator confidence values𝑐𝜙
𝑞and𝑐𝜙
𝑞,𝑑, CAR performs
Bayesian-style posterior correction on the prior ranking. Rather than directly computing continuous posterior proba-
bilities, CAR adopts a discretized binning approach to approximate posterior updating, thereby avoiding dependence
on calibrated baseline scores or generation probabilities.
InCAR,𝑃𝜓(𝑑∣𝑞)isrepresentedbythebaselineranking𝜋𝜓
𝑞,and𝑃𝜙(𝑢=1∣𝑞,𝑑)isapproximatedbythegenerator
confidencechangeΔ𝜙(𝑞,𝑑)=𝑐𝜙
𝑞,𝑑−𝑐𝜙
𝑞.CARfurthercontrolsthetriggeringconditionandupdatestrengthofposterior
correctionthroughthequerythreshold(QT)andconfidencemargin(CM).Basedontheseconfidenceestimates,CAR
performs query-threshold gating, confidence-margin-based document binning, and prior-order-preserving reranking.
The overall procedure is summarized in Algorithm 1.
3.3.1. Query Threshold: Deciding Whether to Update the Prior
CAR first determines whether posterior correction of the prior ranking is necessary according to the query-only
confidence𝑐𝜙
𝑞. Given a query threshold𝑇𝑞, if𝑐𝜙
𝑞≥𝑇𝑞, the generator𝜙can already produce stable answers for the
current query without external documents. In this case, CAR considers posterior correction unnecessary and directly
trusts the prior ranking produced by the baseline𝜓, i.e.,̂ 𝜋𝑞=𝜋𝜓
𝑞.
If𝑐𝜙
𝑞<𝑇𝑞, the generator exhibits high uncertainty on the current query, and candidate documents are expected to
provide additional evidence. CAR therefore initiates the subsequent document-level posterior correction process.
Thus, QT can be understood as a posterior-update gating mechanism: when the generator is already sufficiently
certain, the prior ranking is preserved; when the generator is uncertain, document-conditioned confidence is used to
correct the prior ranking.
Song et al.:Preprint submitted to ElsevierPage 7 of 22

CAR
Algorithm 1CAR Reranking
Require:Query𝑞, ranked list𝜋𝑞, generator𝜙, sample count𝑘, query threshold𝑇𝑞, confidence margin𝑚
Ensure:Reranked list̂ 𝜋𝑞
1:// Stage 1: Query-only Confidence Estimation
2:𝐴𝑞←SAMPLE(𝜙,𝑞,𝑘)
3:𝑍𝑞←SEMANTICCLUSTER(𝐴𝑞)
4:𝑐𝑞←max𝑗(𝑛𝑗∕𝑘)⊳query-only confidence
5:// Stage 2: Query Threshold Gating
6:if𝑐𝑞≥𝑇𝑞then
7:̂ 𝜋𝑞←𝜋𝑞
8:return̂ 𝜋𝑞 ⊳preserve the baseline ranking
9:end if
10:// Stage 3: Document-conditioned Confidence Estimation
11:foreach𝑑𝑖∈𝜋𝑞do
12:𝐴𝑞,𝑑𝑖←SAMPLE(𝜙,(𝑞,𝑑𝑖),𝑘)
13:𝑍𝑞,𝑑𝑖←SEMANTICCLUSTER(𝐴𝑞,𝑑𝑖)
14:𝑐𝑞,𝑑𝑖←max𝑗(𝑛𝑗∕𝑘)⊳query–document confidence
15:end for
16:// Stage 4: Confidence-based Binning
17:foreach𝑑𝑖∈𝜋𝑞do
18:if𝑐𝑞,𝑑𝑖≥𝑐𝑞+𝑚then
19:𝑏𝑖←+1 ⊳promote
20:else if𝑐𝑞,𝑑𝑖≤𝑐𝑞−𝑚then
21:𝑏𝑖←−1 ⊳demote
22:else
23:𝑏𝑖←0 ⊳preserve
24:end if
25:end for
26:// Stage 5: Prior-order-preserving Reranking
27:̂ 𝜋𝑞←STABLESORT(𝜋𝑞,key=(−𝑏𝑖,rank𝜋𝑞(𝑑𝑖)))
28:return̂ 𝜋𝑞
3.3.2. Confidence Margin: Evidence Strength for Posterior Correction
To avoid excessive updates caused by sampling fluctuations, CAR introduces a confidence margin𝑚around the
queryconfidence𝑐𝜙
𝑞,definingtheupperandlowerboundariesas𝑡upper=𝑐𝜙
𝑞+𝑚and𝑡lower=𝑐𝜙
𝑞−𝑚.Foracandidate
document𝑑,CARdeterminesitsposteriorevidencestrengthaccordingtothepositionof𝑐𝜙
𝑞,𝑑relativetothisinterval:
𝑏𝜙(𝑞,𝑑)=⎧
⎪
⎨
⎪⎩+1, 𝑐𝜙
𝑞,𝑑≥𝑐𝜙
𝑞+𝑚,
0, 𝑐𝜙
𝑞−𝑚<𝑐𝜙
𝑞,𝑑<𝑐𝜙
𝑞+𝑚,
−1, 𝑐𝜙
𝑞,𝑑≤𝑐𝜙
𝑞−𝑚.(4)
Here,𝑏𝜙(𝑞,𝑑)representstheposteriorcorrectionlabelofdocument𝑑forquery𝑞fromtheperspectiveofgenerator𝜙:
•𝑏𝜙(𝑞,𝑑) = +1: the document significantly improves the generator’s confidence, corresponding to positive
posterior evidence, and should be promoted;
Song et al.:Preprint submitted to ElsevierPage 8 of 22

CAR
•𝑏𝜙(𝑞,𝑑)=0: the document’s impact is not significant, and the posterior evidence is insufficient to override the
prior, so the original order should be preserved;
•𝑏𝜙(𝑞,𝑑) = −1: the document significantly reduces the generator’s confidence, corresponding to negative
posterior evidence, and should be demoted.
From a probabilistic interpretation, CM sets a minimum evidence threshold for posterior updates. Only when the
generator-side evidence is sufficiently strong does CAR allow it to modify the baseline’s prior ranking. Otherwise,
CAR preserves the original ranking preference of𝜓.
3.3.3. Posterior Binning with Prior Order Preservation
For queries satisfying𝑐𝜙
𝑞< 𝑇𝑞, CAR partitions the candidate documents into three posterior bins according to
𝑏𝜙(𝑞,𝑑):
+
𝜙={𝑑∣𝑏𝜙(𝑞,𝑑)=+1},0
𝜙={𝑑∣𝑏𝜙(𝑞,𝑑)=0},−
𝜙={𝑑∣𝑏𝜙(𝑞,𝑑)=−1}.(5)
The final ranking follows the bin order+
𝜙≻0
𝜙≻−
𝜙.
Within each bin, CAR strictly preserves the original relative order given by the baseline ranker𝜓. Formally, for
any two documents𝑑𝑖,𝑑𝑗, if they have the same posterior correction label, their relative order is inherited from the
baseline:
𝑏𝜙(𝑞,𝑑𝑖)=𝑏𝜙(𝑞,𝑑𝑗)⇒(𝑑𝑖≻̂ 𝜋𝑑𝑗⟺𝑑𝑖≻𝜓𝑑𝑗).(6)
Therefore, CAR’s final ranking can be expressed as:
̂ 𝜋𝑞={
𝜋𝜓
𝑞, 𝑐𝜙
𝑞≥𝑇𝑞,
StableSort(𝜋𝜓
𝑞,−𝑏𝜙(𝑞,𝑑)), 𝑐𝜙
𝑞<𝑇𝑞.(7)
Here,StableSortdenotesastablesortingoperation,i.e.,sortingprimarilybythegenerator’sposteriorcorrectionlabel
while preserving the baseline’s prior order among documents with the same label.
This process can be viewed as a discretized Bayesian-style posterior update. The baseline𝜓provides the prior
ranking𝑃𝜓(𝑑∣𝑞), the generator𝜙provides a confidence-based proxy for posterior usefulness evidence𝑃𝜙(𝑢=
1 ∣𝑞,𝑑), and CAR performs conservative correction of the prior ranking through binning and order-preserving
mechanisms.
3.4. Discussion: Relation to Bayesian Posterior Ranking
CAR shares the structural intuition of standard Bayesian updating. The normalized posterior ranking probability
can be written as:
𝑃(𝑑∣𝑞,𝑢=1;𝜓,𝜙)=𝑃𝜙(𝑢=1∣𝑞,𝑑)𝑃𝜓(𝑑∣𝑞)
∑
𝑑′∈𝜋𝜓
𝑞𝑃𝜙(𝑢=1∣𝑞,𝑑′)𝑃𝜓(𝑑′∣𝑞).(8)
Here,𝑃𝜓(𝑑∣𝑞)represents the prior probability from the retriever or reranker’s perspective, while𝑃𝜙(𝑢= 1 ∣𝑞,𝑑)
representsdocumentusefulnesslikelihood,orposteriorevidence,fromthegenerator’sperspective.Ifbothprobabilities
could be accurately estimated, documents could be ranked directly by their posterior probabilities.
However, in practical black-box RAG scenarios,𝜓may output only a ranking without calibrated scores, and𝜙
cannot directly provide document usefulness probabilities. Therefore, CAR adopts the following approximations:
•It uses the baseline permutation𝜋𝜓
𝑞to approximately express the relative magnitude of the prior𝑃𝜓(𝑑∣𝑞);
•It uses the generator’s sampling consistency𝑐𝜙
𝑞,𝑑to approximately express𝑃𝜙(𝑢=1∣𝑞,𝑑);
•It uses𝑐𝜙
𝑞as a query-specific reference point to reduce confidence-scale differences across queries;
•It uses CM to discretize the continuous posterior signal into promote, preserve, and demote categories;
Song et al.:Preprint submitted to ElsevierPage 9 of 22

CAR
•It uses stable sorting to preserve the relative structure of the prior ranking.
Thus,CARdoesnotattempttoestimatecontinuousBayesianposteriorprobabilitiesexactly.Instead,itimplements
aconservativeapproximationthatfollowstheintuitionofBayesianposteriorupdating:onlywhengenerator𝜙provides
sufficiently strong posterior evidence does CAR adjust the prior ranking given by baseline𝜓; otherwise, the original
ranking structure is preserved.
3.5. Efficiency Analysis
The additional overhead of CAR mainly comes from generator-side confidence estimation, including answer
samplingandsemanticclustering.Incontrast,confidencecomputation,binningdecisions,andstablererankingrequire
onlysimplestatisticsandlinearscans,whosecomputationalcostisnegligible.Therefore,thissectionfocusesonCAR’s
computational overhead and latency characteristics in terms of generator calls.
Given a query𝑞, let the baseline ranker𝜓return𝑛candidate documents, and let𝑘be the number of samples per
input.CARfirst constructsonequery-onlyinput𝑞toestimate𝑐𝜙
𝑞.Ifthe querypassestheQT gate,i.e.,𝑐𝜙
𝑞≥𝑇𝑞,CAR
directlyreturnsthebaselinerankingandavoidsdocument-levelconfidenceestimation.Forqueriessatisfying𝑐𝜙
𝑞<𝑇𝑞,
CAR additionally constructs𝑛query–document inputs(𝑞,𝑑1),…,(𝑞,𝑑𝑛). Therefore, in the worst case or for a query
requiring posterior correction, CAR obtains a total of(𝑛+1)𝑘generation samples. Denoting the average time per
generation call as𝑇gen, the sampling cost can be summarized as:
𝐶serial
sample=𝑂((𝑛+1)𝑘𝑇gen), 𝐶parallel
sample=𝑂(𝑇gen).(9)
The parallel form follows from the fact that different query–document inputs are mutually independent, and the𝑘
samplesunderthesameinputarealsomutuallyindependent.Therefore,underidealparallelconditions,ifallsampling
requests can be issued simultaneously and the generation service has sufficient throughput, the wall-clock time of
the sampling stage can be approximately reduced to a single generation call. This means that although CAR’s total
generation volume grows linearly with𝑛and𝑘, its actual latency is primarily determined by the generation service’s
concurrency capacity rather than necessarily growing linearly with the number of samples.
In the semantic clustering stage, CAR determines whether the𝑘sampled answers under the same input are
semantically equivalent. We use bidirectional entailment as the semantic equivalence criterion: two answers𝑎𝑖and
𝑎𝑗are considered to belong to the same semantic cluster only when both𝑎𝑖→𝑎𝑗and𝑎𝑗→𝑎𝑖hold. This stage is also
performed by the generator𝜙, so its main overhead is generator-side call cost rather than local sorting or statistical
computation.
For semantic clustering, CAR can adopt two implementation modes.
Token-efficient mode.In token-efficient mode, CAR employs a greedy clustering strategy to reduce the number
of entailment judgments. Specifically, the algorithm processes sampled answers sequentially and determines which
existing semantic cluster the current answer should be assigned to. The current answer only needs to be compared
with the representative answer of each existing semantic cluster, rather than with all previously observed answers. If
thecurrentanswerissemanticallyequivalenttoaclusterrepresentative,itisassignedtothatcluster;otherwise,anew
cluster is created.
Letthefinalnumberofsemanticclustersbe𝑟,where𝑟≤𝑘.Foreachinput,greedyclusteringrequiresatmost𝑂(𝑘𝑟)
bidirectional entailment judgments. In the worst case where𝑟=𝑘, the complexity degenerates to𝑂(𝑘2). In practice,
however,whenthegenerator’sanswersarehighlyconsistent,typically𝑟≪𝑘,andthenumberofcomparisonsismuch
smaller than that of full pairwise comparison. Therefore, the token-efficient mode can effectively reduce total token
consumptionandthenumberofgeneratorcalls,makingitsuitableforcost-sensitiveorthroughput-limiteddeployment
scenarios.
Low-latency mode.In low-latency mode, CAR does not prioritize minimizing the total call volume, but instead
aims to minimize actual waiting time. For the𝑘sampled answers under each input, CAR can concurrently execute
bidirectional entailment judgments for all answer pairs. Full pairwise comparison requires considering(𝑘
2)answer
pairs. Since each pair requires bidirectional entailment judgment, the total number of unidirectional entailment
judgmentsis2(𝑘
2)=𝑘(𝑘−1).For𝑛+1inputs,fullpairwiseclusteringthereforerequires(𝑛+1)𝑘(𝑘−1)unidirectional
entailment judgments, and the corresponding costs are:
𝐶serial
cluster=𝑂((𝑛+1)𝑘2𝑇ent), 𝐶parallel
cluster=𝑂(𝑇ent).(10)
Song et al.:Preprint submitted to ElsevierPage 10 of 22

CAR
Here,𝑇entdenotes the average time per entailment judgment. The parallel form follows because these entailment
judgments are mutually independent and can be highly parallelized. Under ideal parallel conditions, if all pairwise
entailment judgments can be executed simultaneously with sufficient server-side concurrency, the wall-clock time of
the semantic clustering stage can be approximately reduced to a single entailment judgment. If entailment judgments
arealsoperformedbygenerator𝜙,andtheircalllatencyisonthesameorderasregulargenerationcalls,thelatencyof
this stage can be approximately regarded as one generation-call duration. Thus, the low-latency mode sacrifices more
tokens and concurrent requests in exchange for near-constant actual waiting time.
After sampling and semantic clustering, CAR only needs to compute the maximum cluster proportion according
to the clustering results to obtain the query-only confidence𝑐𝜙
𝑞and query–document confidence𝑐𝜙
𝑞,𝑑. This step
involves only counting operations. Subsequently, CAR determines whether reranking is needed based on QT, and
when reranking is needed, partitions documents into promote, preserve, and demote bins based on CM. Since this
process only requires traversing𝑛candidate documents and concatenating the bins, its complexity is𝑂(𝑛), which is
negligible compared with the overhead of generator sampling and entailment judgments.
Overall,CAR’stotalcomputationaloverheadconsistsoftwomainparts:(𝑛+1)𝑘generatorsamplesintheworst-
case sampling stage, and a number of bidirectional entailment judgments in the semantic clustering stage. If full
pairwise comparison is adopted, the worst case requires𝑘(𝑘−1)unidirectional entailment judgments per input, for a
totalof𝑂((𝑛+1)𝑘2)generator-sidecalls.Ifgreedyclusteringisadopted,thecallvolumecanbereducedto𝑂((𝑛+1)𝑘𝑟),
where𝑟is the average number of semantic clusters and typically satisfies𝑟≪𝑘.
Fromawall-clocklatencyperspective,boththesamplingandclusteringstagesofCARcanbeparallelized.Under
ideal concurrency conditions, the sampling stage can be approximately reduced to one generation-call duration, and
low-latency clustering can also be approximately reduced to one entailment-judgment duration. Therefore, CAR’s
actual latency does not necessarily grow linearly or quadratically with𝑛and𝑘, but is primarily determined by the
generation service’s concurrency capacity, throughput limits, and token budget. Overall, CAR provides a flexible
efficiencytrade-offbetweentoken-efficientandlow-latencymodes:theformerreducestotaltokenconsumption,while
the latter reduces actual waiting time, allowing CAR to adapt to different RAG deployment requirements.
4. Experiments
Thissectionevaluatestheeffectiveness,robustness,anddownstreamimpactofCAR.Weorganizetheexperiments
around six research questions that correspond to CAR’s key design goals: improving reranking quality, validating the
rolesofitsconservativecorrectionmechanisms,andexaminingwhethertheresultingrankingimprovementsgeneralize
across retrieval settings, model families, sampling budgets, and the full RAG pipeline.
Specifically, we investigate the following research questions:RQ1(§ 4.2): Does CAR consistently improve
document reranking performance?RQ2(§ 4.3): How do CAR’s two core components, QT and CM, affect perfor-
mance?RQ3(§ 4.4.1): Does CAR remain effective across different retrievers?RQ4(§ 4.4.2): Does CAR provide
consistent gains across different generation model families?RQ5(§ 4.4.3): How does the sample number𝑘affect
CAR performance?RQ6(§ 4.4.4): Can retrieval ranking improvements transfer to end-to-end generation quality?
We first describe the experimental setup, including datasets, baselines, implementation details, and evaluation
metrics in § 4.1. We then present the main results, ablation studies, extended robustness analyses, and end-to-end
generation experiments. Finally, § 4.5 summarizes the key findings by directly answering each research question.
4.1. Experimental Setup
4.1.1. Datasets
We select four representative datasets from the BEIR benchmark Thakur et al. (2021):NQ(Natural Ques-
tions Kwiatkowski et al. (2019)),FEVERThorne et al. (2018),SCIDOCSCohan et al. (2020), andTREC-
COVIDVoorheesetal.(2021).Thesefourdatasetscoveropen-domainquestionanswering,factverification,scientific
literature retrieval, and biomedical retrieval, respectively, enabling us to evaluate CAR’s generalizability across
different task types and domains.
Table 2 summarizes the test-set statistics of the four datasets.
4.1.2. Baselines
We compare CAR against three categories of methods:
Song et al.:Preprint submitted to ElsevierPage 11 of 22

CAR
Dataset Task Domain #Query #Corpus Avg. D/Q Avg. #Words (Q/D)
NQ Question Answering Wikipedia 3,452 2,681,468 1.2 9.16 / 78.88
FEVER Fact Checking Wikipedia 6,666 5,416,568 1.2 8.13 / 84.76
SCIDOCS Citation Prediction Scientific 1,000 25,657 4.9 9.38 / 176.19
TREC-COVID Bio-Medical IR Bio-Medical 50 171,332 493.5 10.60 / 160.77
Table 2
Test-set statistics of the four BEIR datasets.Avg. D/Q denotes the average number of relevant documents per query,
and Avg. #Words (Q/D) reports the average number of words in queries and documents, respectively.
•Retriever Only.This setting directly uses the output of the initial retriever without further reranking. We
adopt BM25 Jones et al. (2000) and Contriever Izacard et al. (2022) as representative retrievers. BM25 is a
classicsparselexicalmatchingmethodandreflectsthetraditionalkeyword-basedretrievalparadigm.Contriever
is an unsupervised dense retrieval model and represents the neural retrieval paradigm based on semantic
representation matching. Together, they cover both sparse and dense retrieval settings, allowing us to evaluate
CAR’s generalizability across different initial retrieval paradigms.
•LLM-based Rerankers.Given the candidate document set returned by the initial retriever, these methods
leverage large language models for zero-shot reranking. We include YesNo Qin et al. (2024), QLM Sachan
et al. (2022), and RankGPT Sun et al. (2023). These methods do not independently retrieve documents from
the entire corpus. Instead, they take the top-𝑛candidate documents returned by BM25 or Contriever as input,
prompttheLLMtoassessdocumentrelevancethroughdifferentstrategies,andrerankthecandidatedocuments
accordingly.
•Supervised Neural Rerankers.Supervised neural reranking models also operate on the candidate document
set returned by the initial retriever, rather than independently performing full-corpus retrieval. We use Col-
BERTKhattab&Zaharia(2020),Cross-EncoderReimers&Gurevych(2019),andRankT5Zhuangetal.(2023)
asrepresentativesupervisedrerankers.Thesemethodsaretypicallytrainedonlabeleddataandcanoutputmore
fine-grainedrelevancejudgmentsforquery–documentpairs,representingstrongsupervisedrerankingbaselines.
CARservesasapost-processingmodulethatcanbeappliedontopofanyoftheabovebaselines.IntheRetriever
Onlysetting,CARisapplieddirectlytotherawretrievalresultsofBM25orContriever.ForLLM-basedrerankersand
supervised neural rerankers, CAR is applied to the reranked candidate document lists produced by the corresponding
rerankers. In the tables, “+CAR” denotes applying CAR to the output of the corresponding baseline.
4.1.3. Implementation Details
ThemainexperimentsuseQwen2.5-7B-InstructYangetal.(2024)(abbreviatedasQwen)asthegenerationmodel.
For each query and query–document pair, we sample𝑘= 10times to estimate confidence. Semantic clustering uses
the strict entailment-based judgment mode. QT is searched over𝑇𝑞∈{0,0.1,0.2,…,1.0}, and CM is searched over
𝑚∈{0,0.1,0.2,…,1.0}.CARreranksthetop-10documentsforeachbaseline.Formodel-familyexperiments(RQ4),
weadditionallyuseLlama-3-8B-InstructMetaAI(2024)(abbreviatedasLlama),GLM-4-9B-ChatZengetal.(2024)
(abbreviated as GLM), and InternLM2.5-Chat-7B Cai et al. (2024) (abbreviated as InternLM).
4.1.4. Evaluation Metrics
We evaluate performance from two perspectives: retrieval ranking quality and end-to-end generation quality.
NDCG@5.Forretrievalandrerankingexperiments,weadoptNDCG@5(NormalizedDiscountedCumulativeGain
at 5) as the primary evaluation metric. NDCG@5 measures the overall quality of document relevance in the top 5
positionsoftherankedlistandaccountsforthepositionsatwhichrelevantdocumentsappear.Itsbasicintuitionisthat
morerelevantdocumentsshouldberankedhigher;relevantdocumentsappearingatlowerpositionscontributelessdue
to discounting.
Specifically, given the top𝐾ranked results for a query, DCG@𝐾and NDCG@𝐾are defined as:
DCG@𝐾=𝐾∑
𝑖=12𝑟𝑒𝑙𝑖−1
log2(𝑖+1),NDCG@𝐾=DCG@𝐾
IDCG@𝐾.(11)
Song et al.:Preprint submitted to ElsevierPage 12 of 22

CAR
Table 3
Performance comparison.LLM: Qwen2.5-7B-Instruct. Retriever: Contriever. All scores are in percentage (NDCG@5).
MethodNQ FEVER SCID COVID AVG
ScoreΔ% ScoreΔ% ScoreΔ% ScoreΔ% ScoreΔ%
Retriever Only
Contriever45.537 -74.253- 11.800 - 61.078 - 48.167 -
+CAR45.689 +0.3%74.253+0.0%11.857 +0.5%64.371 +5.4%49.042 +1.8%
LLM-based Rerankers
YesNo23.117 - 25.462 - 7.781 - 55.624 - 27.996 -
+CAR 29.823 +29.0% 41.326 +62.3% 8.910 +14.5% 60.314 +8.4% 35.093 +25.4%
QLM34.862 - 55.687 - 12.177 - 65.401 - 42.032 -
+CAR 37.536 +7.7% 57.820 +3.8%12.216 +0.3%67.029 +2.5% 43.650 +3.9%
RankGPT45.630 -74.268- 11.800 - 61.201 - 48.225 -
+CAR45.752 +0.3%74.268+0.0% 11.857 +0.5% 64.630 +5.6%49.127 +1.9%
Supervised Neural Rerankers
ColBERT47.779 - 76.169 - 12.468 - 69.700 - 51.529 -
+CAR 47.809 +0.1% 76.169 +0.0% 12.486 +0.1% 70.103 +0.6% 51.642 +0.2%
Cross-Encoder48.699 - 78.457 - 13.063 - 68.792 - 52.253 -
+CAR 48.699 +0.0% 78.457 +0.0% 13.066 +0.0% 68.923 +0.2% 52.286 +0.1%
RankT550.969 -81.556- 13.913 - 70.883 - 54.330 -
+CAR50.979 +0.0%81.556+0.0%13.919 +0.0%71.038 +0.2%54.373 +0.1%
Here,𝑟𝑒𝑙𝑖denotes the relevance label of the document at position𝑖, andIDCG@𝐾denotes the DCG value under the
ideal ranking. We set𝐾=5and report NDCG@5. NDCG@5 is suitable for RAG evaluation because RAG systems
typically feed only a small number of top-ranked documents to the generator, making the ranking quality of the top
positions directly relevant to final generation performance.
F1.Inend-to-endgenerationexperiments,weadopttheF1scoretomeasuretoken-leveloverlapbetweenthegenerated
answer and the reference answer. F1 considers both precision and recall, where precision measures how many tokens
inthegeneratedanswerappearinthereferenceanswer,andrecallmeasureshowmanytokensinthereferenceanswer
are covered by the generated answer. They are defined as follows:
Precision=|Pred∩Gold|
|Pred|,Recall=|Pred∩Gold|
|Gold|,F1=2⋅Precision⋅Recall
Precision+Recall.(12)
Here,Preddenotesthesetoftokensinthegeneratedanswer,andGolddenotesthesetoftokensinthereferenceanswer.
F1ishighonlywhenbothprecisionandrecallarehigh,andthusprovidesabalancedmeasureofansweraccuracyand
completeness.WereportF1inend-to-endRAGexperimentstoassesswhetherCAR’sretrievalrankingimprovements
further translate into generation quality gains.
Additionally,Δ%inthetablesdenotestherelativeimprovementofCARoverthecorrespondingbaseline,calculated
as:
Δ%=ScoreCAR−ScoreBaseline
ScoreBaseline×100%.(13)
4.2. Main Results (RQ1)
Table 3 reports the NDCG@5 results under Contriever retrieval. Overall,CAR consistently improves all seven
baselinesintermsofaverageperformance,withnodegradationonanydataset.ThelargestgainsappearonLLM-based
rerankers, especially YesNo, where CAR achieves an average relative improvement of+25.4%. QLM and RankGPT
alsobenefitfromCAR,obtainingaverageimprovementsof+3.9%and+1.9%,respectively.Theseresultsindicatethat
Song et al.:Preprint submitted to ElsevierPage 13 of 22

CAR
Table 4
Ablation study on CAR.LLM: Qwen2.5-7B-Instruct. Retriever: Contriever. All scores are in percentage (NDCG@5).
MethodBaseline w/o QT w/o CM CAR
Score ScoreΔ% ScoreΔ% ScoreΔ%
Retriever Only
Contriever 48.167 47.630 -1.1% 48.912 +1.5%49.042 +1.8%
LLM-based Rerankers
YesNo 27.99635.093 +25.4% 34.044 +21.6%35.093 +25.4%
QLM 42.032 43.459 +3.4% 43.235 +2.9%43.650 +3.9%
RankGPT 48.225 47.724 -1.0% 49.090 +1.8%49.127 +1.9%
Supervised Neural Rerankers
ColBERT 51.529 49.364 -4.2% 51.555 +0.0%51.642 +0.2%
Cross-Encoder 52.253 49.687 -4.9%52.286 +0.1%52.286 +0.1%
RankT5 54.330 51.203 -5.8%54.373 +0.1%54.373 +0.1%
generator-side confidence changes provide useful complementary evidence for correcting relevance-based rankings,
particularly when the baseline reranker is relatively weak.
EffectonweakLLM-basedrerankers.CARbringsthemostsubstantialimprovementstoLLM-basedrerankers.
Inparticular,YesNoimprovesby+25.4%onaverage,showingthatconfidence-awareposteriorcorrectionisespecially
effectivewhentheinitialrerankingsignalisrelativelycoarseornoisy.QLMandRankGPTalsoobtainconsistentgains,
indicatingthatCARcanfurtherrefinerankingsevenwhenthebaselinealreadyusesLLM-basedrelevanceassessment.
Theseresultssuggestthatgenerationconfidencecapturesausefulnesssignalthatiscomplementarytoquery–document
relevance.
Effect on strong supervised rerankers.For stronger supervised rerankers, CAR still brings stable but smaller
improvements: ColBERT, Cross-Encoder, and RankT5 obtain average gains of+0.2%,+0.1%, and+0.1%, respec-
tively. This suggests that CAR remains safe when applied to strong baselines, since the QT and CM mechanisms
prevent unnecessary ranking perturbation. Rather than aggressively overriding the baseline ranking, CAR performs
conservative correction only when the generator-side confidence evidence is sufficiently strong.
Dataset-level observations.Across datasets, the most notable improvements are observed on TREC-COVID,
where several methods obtain larger gains, while FEVER shows many zero-gain cases because the baseline rankings
arealreadyhighlyreliable.ThesepatternsareconsistentwithCAR’sconservativedesign:itmainlycorrectsuncertain
cases while preserving strong prior rankings when further intervention is unnecessary.
4.3. Ablation Study (RQ2)
Table4reportstheablationresultsunderContrieverretrieval.Wecomparetheoriginalbaseline,CARwithoutthe
query threshold (w/o QT), CAR without the confidence margin (w/o CM), and the full CAR. Overall, the full CAR
achieves the best or tied-best performance for all seven baselines, showing that QT and CM jointly support CAR’s
conservative posterior correction mechanism. In particular, CAR brings large gains for weak LLM-based rerankers,
such as YesNo with+25.4%, while maintaining small but non-negative gains for strong supervised rerankers.
EffectofQT.RemovingQTmakesCARapplyconfidence-basedcorrectiontoallqueries,whichcanharmstrong
baselines substantially. For supervised rerankers, w/o QT leads to clear drops: ColBERT decreases by−4.2%, Cross-
Encoder by−4.9%, and RankT5 by−5.8%. This indicates that many high-confidence queries already have reliable
priorrankings,andforcingposteriorcorrectionmaydisturbthem.Incontrast,YesNostillobtains+25.4%withoutQT,
matchingthefullCARresult,suggestingthatweakbaselinescontainmoreuncertaincaseswherecorrectionisbroadly
useful. Thus, QT mainly serves as asafety gatethat protects strong rankings from unnecessary intervention.
Song et al.:Preprint submitted to ElsevierPage 14 of 22

CAR
Effect of CM.Removing CM weakens the stability of confidence-based document assignment. For example,
Contriever improves by+1.5%without CM, but the full CAR further increases the gain to+1.8%; QLM improves
from+2.9%without CM to+3.9%with full CAR; RankGPT also rises from+1.8%to+1.9%. These results show
thatCMhelpsavoidoverreactingtosmallconfidencefluctuationsbyrequiringstrongerevidencebeforepromotionor
demotion. Therefore, QT controlswhenCAR should intervene, while CM controlshow confidentlyeach document
should be moved, and their combination yields the most robust reranking behavior.
4.4. Extended Experiments
4.4.1. Impact of Different Retrievers (RQ3)
0 20 40 60
BM25 %
010203040506070Contriever %
Figure 2: Retriever robustness of CAR.Scatter plot of BM25Δ%vs. ContrieverΔ%on BEIR (Qwen2.5-7B-Instruct,
NDCG@5). Each point represents one (reranker, dataset) pair (rerankers: Retriever-Only, YesNo, QLM, RankGPT,
ColBERT, Cross-Encoder, RankT5; n=28). The dashed diagonal line indicates equal gain for both retrievers; points above
the diagonal suggest Contriever benefits more from CAR, while points below suggest BM25 benefits more. Spearman𝜌=
0.401 (p = 0.0345).
Figure 2 reports CAR’s NDCG@5 gains under BM25 and Contriever retrieval across all rerankers and datasets.
Overall,CARconsistentlyimprovesrankingperformanceunderbothsparseanddenseretrievalsettings.Across
Song et al.:Preprint submitted to ElsevierPage 15 of 22

CAR
7rerankersand4datasets,CARachievesnon-negativegainsonall28datapointsforbothBM25andContriever,with
strictly positive gains on 22 data points under each retriever. These results show that CAR is not tied to a specific
retrieval paradigm and can be effectively applied to both lexical and neural retrievers.
Consistencyacrossretrievers.TheimprovementtrendsunderBM25andContrieverarepositivelycorrelated,with
Spearman𝜌= 0.401,𝑝= 0.0345, and𝑛= 28. This indicates that cases benefiting from CAR under sparse retrieval
also tend to benefit under dense retrieval. Therefore, the generator-side confidence signal used by CAR provides a
retriever-agnostic usefulness cue that complements different types of initial retrieval results.
Retriever-independent robustness.Although BM25 and Contriever produce candidate lists based on different
matching mechanisms, CAR maintains the same conservative correction behavior in both settings. Since CAR only
relies on the baseline order and generator-side confidence changes, it does not require retriever-specific scores,
calibration,orarchitecturalassumptions.ThisexplainswhyCARcanbeusedasaplug-and-playrerankinglayeracross
heterogeneous retrieval backbones.
4.4.2. Impact of Different Models (RQ4)
RetrieverYesNoQLM
RankGPT
ColBERT
Cross-EncoderRankT5
4.34.95.56.16.726.633.139.646.152.6
0.91.52.02.63.2
4.04.75.36.06.7
0.2
0.3
0.5
0.6
0.70.1
0.1
0.2
0.2
0.30.1
0.2
0.3
0.4
0.5(a) BM25
RetrieverYesNoQLM
RankGPT
ColBERT
Cross-EncoderRankT5
1.51.92.32.73.121.927.132.237.442.6
3.64.35.05.86.5
1.41.92.32.83.3
0.2
0.6
0.9
1.3
1.70.1
0.3
0.5
0.7
1.00.1
0.2
0.3
0.4
0.5(b) ContrieverQwen Llama GLM InternLM
Figure 3: Model family comparison of CAR.Radar chart of average NDCG@5 score gain (Δ%) across rerankers on BEIR.
The two panels show results with BM25 and Contriever as the retriever. Each vertex corresponds to a reranker method
(Retriever, YesNo, QLM, RankGPT, ColBERT, Cross-Encoder, RankT5); each line represents one LLM family (Qwen,
Llama, GLM, InternLM) with distinct color, linestyle, and marker. Radial axes use per-vertex independent scales: tick
values along each axis indicate the actualΔ%range for that reranker. A larger enclosed area indicates more consistent
gains across rerankers for that model family.
Figure3reportsCAR’saverageNDCG@5gainsacrossfourLLMbackbones,includingQwen,Llama,GLM,and
InternLM.Overall,CARconsistentlyimprovesrankingperformanceacrossallmodelfamilies.AcrossBM25and
Contriever retrieval, all model–reranker combinations obtain positive gains, demonstrating that CAR is not tied to a
specific generator backbone and can generalize across different LLM families.
The radar charts further reveal clear cross-model differences. InternLM produces the largest and most stable
improvement area under both BM25 and Contriever, indicating that its confidence signals are the most effective for
posterior correction. GLM generally ranks second and shows strong gains in several settings, while Qwen and Llama
also improve performance consistently but with smaller and more uneven gains. These results suggest that CAR is
model-agnosticinapplicability,butitsimprovementmagnitudeisinfluencedbythequalityanddiscriminabilityofthe
generator’s confidence estimates.
AsecondarypatternisthatlargergainsaretypicallyobservedonweakerbaselinessuchasYesNo,whereasstronger
supervisedrerankersreceivesmallerbutstillpositiveimprovements.ThistrendremainsvisibleunderbothBM25and
Song et al.:Preprint submitted to ElsevierPage 16 of 22

CAR
Contriever retrieval, further suggesting that CAR’s cross-model behavior is robust across retrieval settings while still
reflecting differences in generator-side confidence quality.
4.4.3. Impact of Sample Number (RQ5)
1 2 3 4 5 6 7 8 9 10
k (Number of Samples)2022242628303234Best NDCG@5 (%)
BM25
YesNo
QLM
RankGPT
ColBERT
Cross-Encoder
RankT5
Figure 4: Sensitivity analysis of parameter𝑘on BEIR average with BM25.All scores are in percentage (NDCG@5).
CAR benefits from multiple samples and reaches stable performance with moderate sampling.Figure 4
reports the effect of the sampling number𝑘on CAR under BM25 retrieval. The CAR sampling process starts from
𝑘=2, while𝑘=1denotes the w/o CAR baseline score included as a reference. Using multiple samples consistently
improves NDCG@5 across baselines, indicating that multi-sample confidence estimation provides more reliable
signals for query-only and query–document conditions. As𝑘increases, the gains generally show an approximately
monotonicrelationshipandgraduallysaturate,suggestingdiminishingmarginalreturnsfromadditionalsamples.The
improvement is more pronounced for weaker baselines such as YesNo, whereas stronger rerankers such as Cross-
Encoder and RankT5 show flatter trends because their initial rankings are already relatively reliable. These results
suggest that𝑘=5–10offers a practical balance between reranking effectiveness and inference cost.
4.4.4. End-to-end Generation Quality (RQ6)
CAR’s ranking improvements effectively transfer to downstream generation quality.Table 5 reports end-to-
end generation results on NQ using BM25 as the initial retriever and Qwen2.5-7B-Instruct as the generator. Overall,
methods with larger NDCG@5 gains also obtain larger F1 improvements, and the relative improvements of the two
metrics show a strong positive correlation (Spearman𝜌= 0.964,𝑝 <0.001,𝑛= 7). In particular, YesNo achieves
the largest ranking gain, improving NDCG@5 by+43.1%, and also obtains the largest generation gain, improving
F1 by+17.1%. BM25 and RankGPT show similar patterns, with double-digit improvements in both ranking and
generation quality. For stronger supervised rerankers, CAR produces smaller but still positive F1 gains, consistent
withtheirsmallerNDCG@5improvements.Theseresultsindicatethatconfidence-awarererankingnotonlyimproves
top-ranked document quality, but also provides more useful evidence for the generator, thereby enhancing the final
output quality of the full RAG pipeline.
Song et al.:Preprint submitted to ElsevierPage 17 of 22

CAR
Table 5
Relationship between reranking quality and end-to-end generation quality on NQ.Qwen2.5-7B-Instruct is used as the
generator and BM25 as the initial retriever. Ranking quality is measured by NDCG@5, generation quality by token-level F1,
and all scores are reported in percentage. Spearman’s𝜌=0.964,𝑝<0.001,𝑛=7, between relative NDCG@5 improvement
and relative F1 improvement.
MethodRanking (NDCG@5) Generation (F1)
Baseline +CARΔ% Baseline +CARΔ%
BM25 4.877 5.478 +12.3% 13.789 15.228 +10.4%
YesNo 3.215 4.602 +43.1% 12.974 15.192 +17.1%
QLM 6.104 6.292 +3.1% 15.462 15.741 +1.8%
RankGPT 4.888 5.486 +12.2% 13.886 15.291 +10.1%
ColBERT 7.913 7.994 +1.0% 15.635 15.733 +0.6%
Cross-Encoder 8.117 8.158 +0.5% 15.681 15.724 +0.3%
RankT5 8.312 8.339 +0.3% 15.735 15.818 +0.5%
4.5. Takeaways
The experimental findings can be summarized as direct answers to the six research questions introduced at the
beginning of this section:
•RQ1: Does CAR consistently improve document reranking performance?Yes. CAR improves the average
NDCG@5 of all seven baselines under Contriever retrieval, with especially large gains for weaker LLM-based
rerankers such as YesNo, while maintaining non-negative gains for strong supervised rerankers.
•RQ2: How do QT and CM affect performance?QT and CM are both necessary for conservative and stable
posterior correction: QT prevents unnecessary intervention on high-confidence queries, while CM reduces
unstable document movements caused by small sampling fluctuations.
•RQ3: Does CAR remain effective across different retrievers?Yes. CAR produces consistent non-negative
improvements under both BM25 and Contriever, indicating that generator-side confidence changes provide a
retriever-agnostic usefulness signal.
•RQ4: Does CAR generalize across different generation model families?Yes. CAR yields positive gains
acrossQwen,Llama,GLM,andInternLM,showingthattheframeworkisnottiedtoaspecificLLMbackbone,
although the magnitude of improvement depends on the quality of each model’s confidence estimates.
•RQ5: How does the sample number𝑘affect CAR performance?Multiple samples are necessary for
reliableconfidenceestimation.Performanceimprovesasthesamplingbudgetincreasesandgraduallysaturates,
suggesting that𝑘=5–10provides a practical effectiveness–cost trade-off.
•RQ6:Dorankingimprovementstransfertoend-to-endgenerationquality?Yes.OnNQ,CAR’sNDCG@5
improvementsarestronglycorrelatedwithdownstreamF1improvements,demonstratingthatconfidence-aware
reranking provides more useful evidence for the generator in the full RAG pipeline.
Overall, these results support the central claim of this work: generator-side confidence change is an effective
posterior usefulness signal for RAG reranking, and CAR can leverage this signal in a conservative, training-free, and
plug-and-play manner across diverse retrieval, reranking, and generation settings.
5. Conclusion
5.1. Main Findings and Contributions
To address the mismatch between document relevance and generation usefulness in RAG scenarios, we propose
CAR(Confidence-AwareReranking), a query-guided, training-free, and plug-and-play confidence-aware reranking
method. CAR treats the candidate document permutation produced by the baseline retriever or reranker as a prior
Song et al.:Preprint submitted to ElsevierPage 18 of 22

CAR
preference over document usefulness, and leverages the sampling consistency of the generator under query-only and
query–document conditions to estimate confidence changes. These confidence changes provide posterior usefulness
evidence from the generator’s perspective. Through the query threshold and confidence margin, CAR performs
conservativecorrectionofthebaselinerankingwhenthegeneratorisuncertainandthedocument-inducedconfidence
change is sufficiently significant; otherwise, the original ranking structure is preserved.
Experimental results validate CAR’s effectiveness and robustness. First, on four representative datasets from the
BEIR benchmark, CAR consistently improves NDCG@5 performance across multiple types of baselines. Under
Retriever Only, LLM-based reranker, and supervised neural reranker settings, CAR delivers consistent positive
gains, with particularly significant improvements on weaker LLM-based rerankers. This demonstrates that generator
confidence changes can effectively complement traditional relevance signals, helping identify documents that truly
help the generator form stable answers.
Second, ablation studies show that the query threshold (QT) and confidence margin (CM) are key components of
CAR’s conservative reranking mechanism. QT determines whether posterior correction should be initiated based on
query-levelconfidence,preventingunnecessaryinterventiononhigh-confidencequeriesandstrongbaselinerankings.
CM introduces a tolerance interval around the query-only confidence to reduce misclassification caused by sampling
fluctuations. Together, they ensure that CAR can correct ranking deficiencies in weak baselines while maintaining
safety for strong baselines.
Finally,extendedexperimentsfurtherdemonstrateCAR’scross-settinggeneralizability.CARmaintainsconsistent
gainsunderbothBM25andContrieverretrievalparadigmsandachievesconsistentpositiveimprovementsacrossfour
LLM backbone models: Qwen, Llama, GLM, and InternLM. In addition, end-to-end generation experiments show
that CAR’s NDCG@5 improvements effectively transfer to downstream generation quality, with a strong positive
correlation between ranking improvements and generation F1 improvements. This indicates that reranking based on
generator confidence not only improves retrieval metrics but also enhances the final output quality of the full RAG
pipeline.
5.2. Practical Implications
CAR’s design provides strong practical deployment value. First, CAR is a post-processing module that requires
no retraining of the retriever, reranker, or generator, and no access to model internals. It can therefore be directly
appliedontopofexistingRAGsystems.Foralready-deployedBM25systems,denseretrievers,LLM-basedrerankers,
or supervised neural rerankers, CAR only needs to read the candidate document permutation they output in order to
perform confidence-aware correction.
Second, CAR does not depend on calibrated relevance scores from the baseline, but only requires the original
relativeorderofcandidatedocuments.ThismakesCARadaptabletomanyblack-boxretrievalandrerankingsystems,
especially in practical engineering scenarios where different components may come from diverse sources, score
scales may be inconsistent, or only ranked lists may be available. Through stable binning reranking, CAR leverages
generator-sideposteriorevidencewhilepreservingthepriorstructureofthebaselineranking,therebyachievingstrong
compatibility and controllability.
Furthermore,CAR’squery-guidedmechanismprovidesadegreeofcostawareness.Forquerieswherethegenerator
already has high confidence, CAR directly preserves the baseline ranking and avoids unnecessary document-level
confidenceestimation.Forlow-confidencequeries,CARproceedswithdocument-conditionedsamplingandreranking.
This mechanism makes CAR suitable for deployment as an on-demand RAG enhancement module. Although CAR
introduces additional sampling and semantic clustering overhead, these operations naturally support batching and
parallelization, making CAR feasible in generation services with concurrent inference capabilities.
5.3. Limitations and Future Work
Despite CAR’s consistent empirical gains, the method has several limitations. First, CAR requires multiple
sampling rounds for the query-only input and multiple query–document inputs, and further performs bidirectional
entailment-basedsemanticclustering,therebyincurringadditionalinferencecosts.Althoughsamplingandentailment
judgments can be parallelized, further optimization of the trade-off among token consumption, concurrent request
count, and overall response time is needed for large-scale low-latency retrieval scenarios.
Second, CAR currently relies on two hyperparameters, the query threshold and confidence margin, whose values
are determined through validation-set search. Although experiments demonstrate that both components effectively
improvemethodstability,fixedthresholdsmaynotfullyadapttodifferentdatasets,querytypes,orgeneratorconfidence
Song et al.:Preprint submitted to ElsevierPage 19 of 22

CAR
distributions. Future work could explore adaptive threshold strategies, such as dynamically adjusting QT and CM
according to query difficulty, candidate document distribution, or generator confidence calibration.
Third, CAR’s confidence estimation relies on the semantic consistency of sampled answers, and the semantic
clustering process depends on the generator or a discriminative model’s judgment of bidirectional entailment
relationships. In scenarios involving domain-intensive knowledge, low-resource languages, or highly diverse answer
formulations, entailment judgments may be prone to errors, thereby affecting the reliability of confidence estimation.
Future work could introduce lighter and more stable confidence estimation approaches, such as combining token-
level uncertainty, self-consistency, logit-based calibration, or specially trained semantic-equivalence discriminators,
to reduce cost and improve robustness.
Finally, this paper primarily validates CAR in text retrieval and text generation scenarios. As RAG systems
increasinglyexpandtomultimodalquestionanswering,tablereasoning,coderetrieval,andlong-documentanalysis,the
relationship between document usefulness and generator uncertainty may become more complex. Future work could
extendCARtolarger-scaleandmorecomplexretrievalsettings,andfurtherinvestigatethevalueofconfidence-aware
reranking in multimodal RAG, agentic retrieval, and interactive retrieval-augmented generation systems.
Song et al.:Preprint submitted to ElsevierPage 20 of 22

CAR
References
Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). Self-RAG: Learning to retrieve, generate, and critique through self-reflection. InThe
twelfth international conference on learning representations.Retrieved fromhttps://openreview.net/forum?id=hSyW5go0v8
Cai, Z., Cao, M., Chen, H., Chen, K., Chen, K., Chen, X., ... Zhao, X. (2024). Internlm2 technical report.arXiv. doi: https://doi.org/10.48550/
arXiv.2403.17297
Cohan, A., Feldman, S., Beltagy, I., Downey, D., & Weld, D. (2020, July). SPECTER: Document-level representation learning using citation-
informed transformers. InProceedings of the 58th annual meeting of the association for computational linguistics(pp. 2270–2282). Online:
Association for Computational Linguistics. doi: https://doi.org/10.18653/v1/2020.acl-main.207
Farquhar,S.,Kossen,J.,Kuhn,L.,&Gal,Y. (2024). Detectinghallucinationsinlargelanguagemodelsusingsemanticentropy.Nature,630(8017),
625–630. doi: https://doi.org/10.1038/s41586-024-07421-0
Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., ... Wang, H. (2023). Retrieval-augmented generation for large language models: A survey.
arXiv. doi: https://doi.org/10.48550/arXiv.2312.10997
Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bojanowski, P., Joulin, A., & Grave, E. (2022). Unsupervised dense information retrieval with
contrastive learning.Transactions on Machine Learning Research. Retrieved fromhttps://openreview.net/forum?id=jKN1pXi7b0
Jiang,Z.,Xu,F.,Gao,L.,Sun,Z.,Liu,Q.,Dwivedi-Yu,J.,... Neubig,G. (2023,December). Activeretrievalaugmentedgeneration. InProceedings
of the 2023 conference on empirical methods in natural language processing(pp. 7969–7992). Singapore: Association for Computational
Linguistics. doi: https://doi.org/10.18653/v1/2023.emnlp-main.495
Jones, K. S., Walker, S., & Robertson, S. E. (2000). A probabilistic model of information retrieval: development and comparative experiments -
part 2.Information Processing & Management,36(6), 809–840. doi: https://doi.org/10.1016/S0306-4573(00)00016-9
Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., ... Kaplan, J. (2022). Language models (mostly) know what they know.
arXiv. doi: https://doi.org/10.48550/arXiv.2207.05221
Khattab, O., & Zaharia, M. (2020). Colbert: Efficient and effective passage search via contextualized late interaction over BERT. InProceedings
of the 43rd international ACM SIGIR conference on research and development in information retrieval, SIGIR 2020, virtual event, china, july
25-30, 2020(pp. 39–48). doi: https://doi.org/10.1145/3397271.3401075
Kwiatkowski,T.,Palomaki,J.,Redfield,O.,Collins,M.,Parikh,A.,Alberti,C.,... Petrov,S. (2019). Naturalquestions:Abenchmarkforquestion
answering research.Transactions of the Association for Computational Linguistics,7, 452–466. doi: https://doi.org/10.1162/tacl_a_00276
Lewis,P.,Perez,E.,Piktus,A.,Petroni,F.,Karpukhin,V.,Goyal,N.,... Kiela,D. (2020). Retrieval-augmentedgenerationforknowledge-intensive
NLP tasks. InAdvances in neural information processing systems 33: Annual conference on neural information processing systems 2020,
neurips 2020, december 6-12, 2020, virtual(Vol. 33, pp. 9459–9474). Retrieved fromhttps://proceedings.neurips.cc/paper/2020/
hash/6b493230205f780e1bc26945df7481e5-Abstract.html
Meta AI. (2024).Introducing llama 3.1: Our most capable models to date.Retrieved fromhttps://ai.meta.com/blog/meta-llama-3-1/
Pradeep,R.,Sharifymoghaddam,S.,&Lin,J. (2023). Rankvicuna:Zero-shotlistwisedocumentrerankingwithopen-sourcelargelanguagemodels.
arXiv. doi: https://doi.org/10.48550/arXiv.2309.15088
Qin, Z., Jagerman, R., Hui, K., Zhuang, H., Wu, J., Yan, L., ... Bendersky, M. (2024). Large language models are effective text rankers with
pairwise ranking prompting. InFindings of the association for computational linguistics: NAACL 2024, mexico city, mexico, june 16-21, 2024
(pp. 1504–1518). doi: https://doi.org/10.18653/v1/2024.findings-naacl.97
Reimers, N., & Gurevych, I. (2019, November). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. InProceedings of the
2019conferenceonempiricalmethodsinnaturallanguageprocessingandthe9thinternationaljointconferenceonnaturallanguageprocessing
(emnlp-ijcnlp)(pp. 3982–3992). Hong Kong, China: Association for Computational Linguistics. doi: https://doi.org/10.18653/v1/D19-1410
Sachan, D. S., Lewis, M., Joshi, M., Aghajanyan, A., Yih, W., Pineau, J., & Zettlemoyer, L. (2022). Improving passage retrieval with zero-shot
question generation. InProceedings of the 2022 conference on empirical methods in natural language processing, EMNLP 2022, abu dhabi,
united arab emirates, december 7-11, 2022(pp. 3781–3797). doi: https://doi.org/10.18653/v1/2022.emnlp-main.249
Song, Z., Kong, X., Bao, X., Zhou, Y., Jiao, J., Liu, S., ... Qi, H. (2026). Llm-confidence reranker: A training-free approach for enhancing
retrieval-augmented generation systems.Expert Systems with Applications,314, 131627. doi: https://doi.org/10.1016/j.eswa.2026.131627
Song, Z., Zhou, Y., Kong, X., Jiao, J., Bao, X., You, X., ... Qi, H. (2026). Less is more for RAG: information gain pruning for generator-aligned
reranking and evidence selection.arXiv. doi: https://arxiv.org/abs/2601.17532
Sun,W.,Yan,L.,Ma,X.,Wang,S.,Ren,P.,Chen,Z.,... Ren,Z. (2023). Ischatgptgoodatsearch?investigatinglargelanguagemodelsasre-ranking
agents. InProceedingsofthe2023conferenceonempiricalmethodsinnaturallanguageprocessing,EMNLP2023,singapore,december6-10,
2023(pp. 14918–14937). doi: https://doi.org/10.18653/v1/2023.emnlp-main.923
Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A heterogeneous benchmark for zero-shot evaluation of
information retrieval models. InThirty-fifth conference on neural information processing systems datasets and benchmarks track (round 2).
Retrieved fromhttps://openreview.net/forum?id=wCu6T5xFjeJ
Thorne,J.,Vlachos,A.,Christodoulopoulos,C.,&Mittal,A.(2018).FEVER:alarge-scaledatasetforfactextractionandverification.InProceedings
of the 2018 conference of the north american chapter of the association for computational linguistics: Human language technologies, NAACL-
HLT 2018, new orleans, louisiana, usa, june 1-6, 2018, volume 1 (long papers)(pp. 809–819). doi: https://doi.org/10.18653/v1/n18-1074
Voorhees, E., Alam, T., Bedrick, S., Demner-Fushman, D., Hersh, W. R., Lo, K., ... Wang, L. L. (2021, February). Trec-covid: constructing a
pandemic information retrieval test collection.SIGIR Forum,54(1). doi: https://doi.org/10.1145/3451964.3451965
Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., ... Zhou, D. (2023). Self-consistency improves chain of thought reasoning
in language models. InThe eleventh international conference on learning representations.Retrieved fromhttps://openreview.net/
forum?id=1PL1NIMMrw
Xiong, M., Hu, Z., Lu, X., LI, Y., Fu, J., He, J., & Hooi, B. (2024). Can LLMs express their uncertainty? an empirical evaluation of confidence
elicitation in LLMs. InThe twelfth international conference on learning representations.Retrieved fromhttps://openreview.net/
forum?id=gjeQKFxFpZ
Song et al.:Preprint submitted to ElsevierPage 21 of 22

CAR
Yan, S.-Q., Gu, J.-C., Zhu, Y., & Ling, Z.-H. (2024). Corrective retrieval augmented generation.arXiv. doi: https://doi.org/10.48550/
arXiv.2401.15884
Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., ... Fan, Z. (2024). Qwen2 technical report.arXiv. doi: https://doi.org/10.48550/
arXiv.2407.10671
Yu, Y., Ping, W., Liu, Z., Wang, B., You, J., Zhang, C., ... Catanzaro, B. (2024). Rankrag: Unifying context ranking with retrieval-augmented
generation in llms. InAdvances in neural information processing systems(Vol. 37, pp. 121156–121184). Curran Associates, Inc. doi:
https://doi.org/10.52202/079017-3850
Zeng, A., Xu, B., Wang, B., Zhang, C., Yin, D., Rojas, D., ... Wang, Z. (2024). Chatglm: A family of large language models from GLM-130B to
GLM-4 all tools.arXiv. doi: https://doi.org/10.48550/arXiv.2406.12793
Zhuang,H.,Qin,Z.,Jagerman,R.,Hui,K.,Ma,J.,Lu,J.,... Bendersky,M. (2023). Rankt5:Fine-tuningT5fortextrankingwithrankinglosses. In
Proceedingsofthe46thinternationalACMSIGIRconferenceonresearchanddevelopmentininformationretrieval,SIGIR2023,taipei,taiwan,
july 23-27, 2023(pp. 2308–2313). Retrieved fromhttps://doi.org/10.1145/3539618.3592047doi: 10.1145/3539618.3592047
Song et al.:Preprint submitted to ElsevierPage 22 of 22
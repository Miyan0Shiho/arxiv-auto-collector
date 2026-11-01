# When Is Additional Retrieval Worthwhile? A Cost-Aware Study of Evidence Gain and Answer Reliability in RAG

**Authors**: Zhouting Zhao

**Published**: 2026-10-27 14:55:58

**PDF URL**: [https://arxiv.org/pdf/2610.37358v1](https://arxiv.org/pdf/2610.37358v1)

## Abstract
Retrieval-augmented generation (RAG) can benefit from more sophisticated retrieval, but extra retrieval may be unnecessary for many questions and gains in evidence retrieval may not translate into faithful or correct answers. We study a cost-aware, two-action retrieval policy that chooses between a dense-retrieval-plus-reranking baseline and a hybrid lexical--dense retriever. Rather than classifying baseline retrieval as simply successful or unsuccessful, we estimate the question-specific incremental evidence-retrieval benefit of hybrid retrieval using features available before the routing decision. A regularized linear model and decision threshold are selected using document-grouped cross-validation and frozen prior to held-out evaluation. On 60 previously unseen QASPER training-split articles (188 questions, 169 evidence-evaluable), the policy triggers hybrid retrieval on 46 questions (24.5\%) and obtains the same evidence Hit@10 as always-hybrid retrieval (78.70\%, versus 76.92\% for always-keep). Paired, paper-cluster bootstrap intervals do not establish a reliable utility advantage. We further compare generated answers from a 4-bit Qwen3-4B model under the three retrieval policies. Provisional answer-level assessments show no clear correctness or faithfulness improvement after partial review, despite the evidence-retrieval result. The generation assessments remain model-assisted and lack independent human validation. These results expose an important distinction between retrieving relevant evidence efficiently and improving the reliability of the generated answer. The work is presented as a reproducible empirical study, not a claim of statistically established answer-quality superiority.

## Full Text


<!-- PDF content starts -->

When Is Additional Retrieval Worthwhile? A
Cost-Aware Study of Evidence Gain and Answer
Reliability in RAG
Zhouting Zhao
noora.zhaozz@gmail.com
Abstract
Retrieval-augmented generation (RAG) can benefit from more sophisticated re-
trieval, but extra retrieval may be unnecessary for many questions and gains in
evidence retrieval may not translate into faithful or correct answers. We study a
cost-aware, two-action retrieval policy that chooses between a dense-retrieval-plus-
reranking baseline and a hybrid lexical–dense retriever. Rather than classifying
baselineretrievalassimplysuccessfulorunsuccessful,weestimatethequestion-
specific incremental evidence-retrieval benefit of hybrid retrieval using features
available before the routing decision. A regularized linear model and decision
thresholdareselectedusingdocument-groupedcross-validationandfrozenprior
toheld-outevaluation. On60previouslyunseenQASPERtraining-splitarticles
(188 questions, 169 evidence-evaluable), the policy triggers hybrid retrieval on 46
questions(24.5%)andobtainsthesameevidenceHit@10asalways-hybridretrieval
(78.70%, versus 76.92% for always-keep). Paired, paper-cluster bootstrap intervals
donotestablishareliableutilityadvantage. Wefurthercomparegeneratedanswers
from a 4-bit Qwen3-4B model under the three retrieval policies. Provisional
answer-levelassessmentsshownoclearcorrectnessorfaithfulnessimprovement
after partial review, despite the evidence-retrieval result. The generation assess-
mentsremainmodel-assistedandlackindependenthumanvalidation. Theseresults
expose an important distinction between retrieving relevant evidence efficiently
andimprovingthereliabilityofthegeneratedanswer. Theworkispresentedasa
reproducible empirical study, not a claim of statistically established answer-quality
superiority.
1 Introduction
Retrieval-augmentedgeneration(RAG)hasemergedasaneffectiveapproachforimprovingthefactual
grounding of largelanguage models (LLMs) byincorporating information from externalknowledge
sources[Lewisetal.,2020,Gaoetal.,2023b]. Bycombiningparametricknowledgewithretrieved
evidence, RAG systems can support knowledge-intensive question answering while offering greater
transparencyandaccesstodomain-specificinformation. Nevertheless,thereliabilityofgenerated
answersremainsdependentonboththequalityofretrievedevidenceandthegenerator’sabilityto
interpret and use that evidence correctly.
A central limitation of conventional RAG pipelines is their reliance on fixed retrieval strategies.
Regardless of the difficulty of a question or the quality of initially retrieved evidence, the same
retrievalprocedureistypicallyapplied. However,differentqueriesmaybenefitfromdifferentretrieval
strategies. Denseretrievalcapturessemanticsimilarity,whilelexicalretrievalcanrecoverinformation
involving specific terminology or exact expressions. Combining these approaches may improve
evidencecoverage,butadditionalcandidatescanalsointroduceirrelevantinformation,displaceuseful
passages during reranking, and increase computational cost.
Preprint.
arXiv:2610.37358v1  [cs.IR]  27 Oct 2026

Recent studies have explored adaptive retrieval as a way to address these limitations. Adaptive-RAG
[Jeong et al., 2024] selects retrieval strategies according to question complexity, while active retrieval
approachessuchasFLARE[Jiangetal.,2023]dynamicallyretrieveadditionalinformationduring
generation. Corrective RAG [Yan et al., 2024] introduces retrieval-quality assessment to trigger
corrective actions. These studies demonstrate the value of adapting retrieval behavior, but also
motivate a more specific question: whether the incremental benefit of an additional retrieval strategy
can be predicted from the evidence already available.
Importantly,improvementsinretrievaleffectivenessdonotnecessarilytranslateintomorereliable
generated answers. A retriever may identify relevant evidence without providing all the information
needed to answer a question. Conversely, a generator may produce an incomplete, incorrect, or
unsupported answer even when sufficient evidence is present. Previous studies on attribution and
factuality evaluation, including ALCE [Gao et al., 2023a], FActScore [Min et al., 2023], and RAGAs
[Esetal.,2024],highlighttheimportanceofdistinguishingfactualcorrectnessfromevidentialsupport.
Consequently, evaluating a retrieval strategy solely through evidence recall or hit rate may provide an
incomplete picture of its downstream utility.
This paper investigates the following research question:To what extent can cost-aware adaptive
retrieval improve the correctness and faithfulness of RAG-generated answers compared with fixed
retrieval strategies?
Westudyadeliberatelyconstrainedtwo-actionretrievalpolicy. Givenaninitialdense-retrievaland
cross-encoder-reranking result, the policy selects eitherKeep, which retains the baseline evidence, or
Hybrid,whichuseslexical–semanticretrievalfollowedbyreranking. Ratherthanpredictingretrieval
failure directly, the policy estimates the expected incremental evidence benefit of switching to hybrid
retrieval using features observable from the baseline retrieval process. The decision additionally
accounts for the cost of invoking the more expensive retrieval strategy.
A distinguishing aspect of our study is the separation between the policy’s training objective and
its downstream evaluation. The policy is trained to predict improvements in evidence Hit@10,
whereasitspracticalusefulnessisassessedthroughbothretrieval-leveloutcomesandgeneration-level
reliability. Thisdesignallowsustoexaminewhetherimprovementsinretrievedevidencetranslate
into corresponding gains in answer correctness and faithfulness.
Our contributions are threefold:
•We develop an operational two-action, cost-aware adaptive retrieval policy that predicts the
incremental evidence benefit of hybrid retrieval using only baseline-observable runtime
features.
•Weconductadocument-disjoint,prediction-firstheld-outevaluation,comparingadaptive
retrieval against fixed dense and hybrid baselines with paired paper-level uncertainty
estimates.
•We investigate the relationship between evidence retrieval effectiveness and downstream an-
swerreliabilitythroughanexploratorygeneration-levelevaluationofcorrectness,faithfulness,
and abstention.
Ourfindingshighlightboththeefficiencypotentialandthelimitationsofevidence-gain-basedrouting.
Whileadaptiveretrievalcanreduceunnecessaryhybridretrievalcalls,improvementsinevidence-level
metrics do not automatically imply improvements in generated-answer reliability.
Figure 1 illustrates the motivation and overall workflow of our cost-aware adaptive retrieval approach.
2 Related Work
Retrieval-Augmented Generation.Retrieval-augmented generation integrates external knowledge
retrieval with language-model generation to improve performance on knowledge-intensive tasks.
Lewisetal.[2020]introducedafoundationalRAGframeworkcombiningparametricsequence-to-
sequencemodelswithnon-parametricknowledgeretrieval. Subsequentresearchhasexploreddifferent
retrieval and augmentation mechanisms, including retriever adaptation and hierarchical evidence
organization[Shietal.,2024, Sarthi etal.,2024]. Gaoetal.[2023b]provideabroaderoverviewof
thesedevelopments. Despitesubstantialadvancesinretrievalarchitectures,theavailabilityofrelevant
2

Figure1: Motivationforreliability-orientedadaptiveretrieval. Fixedretrievalstrategiesmayprovide
incompleteorinsufficientevidence,potentiallyleadingtoincorrectorunsupportedanswers. Adaptive
retrievalusesobservablebaselineretrievalsignalstodeterminewhetheradditionalhybridretrieval
iswarranted. Thecentralquestioniswhetherimprovedevidenceselectioncantranslateintomore
reliable generated answers at a reasonable computational cost.
evidencedoesnotguaranteethatalanguagemodelwillproduceafactuallycorrectandwell-grounded
answer. Our work focuses on the decision of whether to invoke an additional retrieval strategy rather
than proposing a new retrieval architecture.
AdaptiveandActiveRetrieval.Adaptiveretrievalmethodsaimtoallocateretrievaleffortaccording
totheneedsofindividualqueries. Jeongetal.[2024]proposedAdaptive-RAG,whichselectsretrieval
strategies according to question complexity. Jiang et al. [2023] introduced FLARE that performs
activeretrievalduringtextgenerationbyidentifyingpotentiallyunreliableupcomingcontent. Asai
et al. [2024] proposed Self-RAG that integrates retrieval and generation with learned reflection
mechanisms that support adaptive retrieval and self-evaluation. These approaches demonstrate that
retrieval need not follow a uniform procedure across all queries. Our approach differs in its narrower
decision setting: instead of selecting retrieval depth from question complexity or controlling retrieval
duringgeneration,weusetheoutputofaninitialretrievalproceduretoestimatethemarginalevidence
benefit of a predefined hybrid alternative.
CorrectiveandHybridRetrieval.CorrectiveRetrieval-AugmentedGeneration(CRAG)[Yanetal.,
2024] evaluates retrieved evidence and triggers corrective retrieval operations when the available
information is unreliable. Hybrid retrieval provides another mechanism for improving evidence
coverage by combining complementary lexical and semantic signals. However, additional retrieval is
notguaranteedtoimprovethefinalevidenceranking,sincenewlyintroducedcandidatesmaydisplace
previouslyrelevantpassages. Wethereforeframehybridretrievalasanoptionalactionwithpotentially
positive, neutral, or negative incremental outcomes. Rather than treating low retrieval confidence as
sufficientjustificationforcorrection,ourpolicyestimatestheexpectedchangeinevidenceretrieval
success.
RAGReliabilityandEvaluation.EvaluatingRAGsystemsrequiresdistinguishingretrievalquality
from generation quality. RAGAs [Es et al., 2024] introduces a framework for assessing different
components of RAG pipelines, including faithfulness and answer quality. Ru et al. [2024] introduced
RAGCheckerwhichfurtheremphasizesfine-graineddiagnosisofretrievalandgenerationbehavior.
Complementary work on factuality and attribution, such as FActScore [Min et al., 2023] and ALCE
[Gaoetal.,2023a],demonstratestheimportanceofverifyinggeneratedclaimsandtheirsupporting
evidence. Motivated by these findings, we evaluate retrieval-level effectiveness separately from
answercorrectnessandfaithfulness. Thisdistinctionisparticularlyimportantbecauseouradaptive
policy is optimized using an evidence-level proxy rather than direct supervision from answer-quality
labels.
3

Overall, our work investigates whether a lightweight, cost-aware policy trained on incremental
evidence retrieval outcomes can provide a useful efficiency–reliability trade-off. We do not claim
noveltyintheunderlyingretrievalalgorithmsorregressionmodel;instead,weempiricallyexaminethe
relationshipbetweenmarginalretrievalbenefit,adaptiverouting,anddownstreamanswerreliability.
3 Problem Formulation and Retrieval Methods
Let𝑞𝑖denote a question about source article 𝑑𝑖. Both retrieval arms search only evidence units
belongingto 𝑑𝑖;theydonotsearchotherarticles. Eacharticleisrepresentedbyparagraph,section-
heading, and figure/table-caption text units. The figure or table body may not be available, so a
caption is only a proxy for figure-dependent evidence.
TheKeeparm embeds the question and evidence with all-MiniLM-L6-v2 , ranks candidates by
cosinesimilarity,andreranksthetop30with ms-marco-MiniLM-L6-v2 cross-encoder,returning10
evidence units. TheHybridarm combines denseranking with BM25lexical ranking viareciprocal
rank fusion (RRF), uses a candidate pool of 30, and applies the same cross-encoder reranker:
RRF(𝑒|𝑞)=∑︁
𝑟∈{dense,BM25}:𝑒∈𝑟1
60+rank𝑟(𝑒).(1)
Thetwotop-10evidencesetsaredenoted 𝐸𝐾
𝑖and𝐸𝐻
𝑖. Aretrievallabel ℎ𝑎
𝑖∈{0,1}indicateswhether
𝐸𝑎
𝑖contains any aligned gold evidence for arm 𝑎∈{𝐾,𝐻} . Gold labels are used only for offline
policy training and evaluation, never as deployment features.
4 Cost-Aware Incremental-Benefit Policy
The offline training target is the realized retrieval difference
Δ𝑖=ℎ𝐻
𝑖−ℎ𝐾
𝑖∈{−1,0,1}.(2)
Positive values represent rescues, negative values harms, and zeros ties. The deployment-time feature
vector𝑥𝑖hassixquantitiescomputedfromtheKeeparmalone: top-1densesimilarity,top-2dense
margin, mean top-5 dense similarity, overlap of dense and reranked top-10 candidates, the dense
rank of the rerank top-1 candidate (or 11 if absent), and the fraction of baseline top-10 units that are
figure/table captions.
The fitted pipeline is median imputation, standardization, and ridge regression:
ˆΔ𝑖=ˆ𝛽0+ˆ𝛽⊤˜𝑥𝑖,min
𝛽0,𝛽∑︁
𝑖(Δ𝑖−𝛽0−𝛽⊤˜𝑥𝑖)2+20∥𝛽∥2
2.(3)
A unitized extra-hybrid penalty 𝜆=0.02 defines offline retrieval utility 𝑢𝑖=ℎ𝜋
𝑖−𝜆I[𝜋(𝑥𝑖)=𝐻].
The deployed decision rule is
𝜋(𝑥𝑖)=Hybrid, ˆΔ𝑖≥𝜏,
Keep, ˆΔ𝑖<𝜏.𝜏=0.0310.(4)
At inference time, the policy uses features extracted solely from the baseline retrieval results. Hybrid
retrieval is executed only when the predicted incremental benefit meets or exceeds the selected
decision threshold.
Thethresholdwasselectedfromgroup-held-outdevelopmentpredictionstomaximizemeanpredicted-
policy net utility under the specified penalty, with ties preferring fewer triggers. The fitted model and
thresholdwerefrozenbeforeheld-outgoldevidencewasopened. Thepenaltyisanabstractunitcost;
it is not an observed time, token, energy, or monetary cost. Accordingly the derived utility is a proxy
and must not be interpreted as measured end-to-end efficiency.
5 Experimental Setup
5.1 Data and leakage controls
The QASPER benchmark contains questions about research papers [Dasigi et al., 2021]. Our
developmentpartitioncomprises240articles(690questions,599withevidencesuitableforstructural
4

Figure2: Baseline-feature-onlyadaptiveretrieval. Thegeneratordoesnotseethegoldevidenceor
referenceanswers. Bothretrievalalternativesarecomputedforofflinecomparison;onlytherouted
alternative is needed operationally.
retrieval evaluation). A separate 60-article partition (188 questions, 169 evidence-evaluable) was
held out from policy training. Both partitions were sampled from theofficial QASPER training split;
theofficialtestsplitwasnotused. Indevelopment,16questionswerepotentialhybridrescuesand
sevenwereharms,makingthetargethighlyimbalanced. Allcross-validationfoldsgroupquestionsby
article. The final training fit uses only development labels.
A separate prediction-only stage generated the held-out rankings and decisions and saved file hashes
before any held-out gold evaluation. For the held-out evidence metrics, 288 referenced evidence
spans could be structurally aligned; caption matches do not certify that the numerical contents of
figures or tables are visible. The subsequent generation analysis on these articles is therefore an
exploratory follow-up, not another untouched test.
5.2 Generation and evaluation protocol
A fixed mlx-community/Qwen3-4B-Instruct-2507-4bit generator receives the question and
top-10 evidence, with a maximum of 12,000 evidence characters and 256 output tokens. The
prompt requires concise evidence-grounded answers, evidence-ID citations, and the exact string
Insufficient evidence. when support is inadequate. We generate responses for both fixed
retrievalarms. Iftwocompletepromptsareidentical,theirgeneratedoutputisreused;theadaptive
response is the previously generated response of its selected arm, not an additional independent
sample. Over 188 questions, there are 376 fixed-arm answer records and 289 distinct prompts.
Correctnessis a 0–2 assessment against QASPER reference answers: wrong/unsupported answer or
unjustifiedabstention(0),partlycorrectormateriallyincomplete(1),andcorrect(2).Faithfulness
independently scores factual claims against the evidenceactually displayed to the generator, using 0
(unsupported), 1 (partially supported), and 2 (supported). Pure abstentions have no faithfulness score
andareinstead analyzed byabstentionappropriateness. Differencesamongreferenceannotatorsare
flagged. TheselabelswereprovisionallyassignedusingthesameQwen-familyjudgeandpartially
assistant-reviewed,notindependentlyhuman-validated. Thescoreversionusedbelowisfrozenfor
analysis but not considered definitive ground truth.
5.3 Statistical analysis
Policiesarecomparedonpairedquestionswithscoresavailableunderallthreestrategies. Confidence
intervals for differences use 10,000 paired bootstrap resamples at the article level. Such intervals
reflect article-sampling uncertaintyconditional on the assigned labels; they do not quantify the error
5

Table1: Held-outevidenceretrievalon169evaluablequestions(60articlestotal). Hybridcallcounts
are over all 188 routed questions. Utility is unitized Hit@10 minus 0.02 times the hybrid-trigger
indicator.
Policy Hit@10 (%) Hybrid calls Unitized utility
Always Keep 76.92 0 / 188 0.7692
Always Hybrid 78.70 188 / 188 0.7670
Frozen adaptive 78.70 46 / 1880.7819
Table 2:Provisionalgeneration-level comparisons (scores from 0 to 2). Correctness is compared on
181pairedquestions;faithfulnesson128jointlyscorablequestions. Valuesareexploratoryandnot
independently human validated.
Policy Correctness Faithfulness Hybrid calls
Always Keep 1.1271.8440
Always Hybrid1.1331.813 188
Frozen adaptive 1.127 1.820 46
of the language-model judge. Faithfulness is conditional on non-abstained, jointly scorable questions,
so its denominator differs from correctness.
6 Results
6.1 Held-out evidence retrieval
Always-hybrid retrieval yields five potential rescues and two harms relative to keep; the frozen policy
realizes four rescues and one harm, giving the same net gain in successful questions. It selects hybrid
on 24.47% of all questions (43/169 among evidence-evaluable questions), thereby avoiding 142 extra
hybrid calls relative to always hybrid. The paper-level 95% bootstrap interval for adaptive minus
keep in retrieval utility is approximately [−0.0114,0.0385] and for adaptive minus always hybrid
isapproximately[−0.0021,0.0328] ;bothincludezero. Accordinglytheobservedresultsupportsa
reduction inhybrid-call frequencyat comparableevidence hit rate,but nota statistically conclusive
utility advantage. A reduction in triggered hybrid calls should not be conflated with a 75% reduction
in total runtime.
6.2 Generated-answer correctness and faithfulness
Followingpartialmanual-assistedreviewofhigh-prioritycorrectnesscases,theadaptiveandkeep
correctnessmeanscoincide. Thepairedadaptive-minus-keepcorrectnessdifferenceis0.000onthe
0–2 scale (95% paper-cluster interval approximately [−0.029,0.024] ). This result differs from an
earlier,unreviewedjudge-onlycomparison,inwhichtheadaptiveadvantagewassmallandpositive,
demonstratingthesensitivityofsmallpolicydifferencestoadjudication. Faithfulnessdoesnotexhibit
aconsistentadaptiveimprovementoverkeep;intheprovisionalcommonsubset,itisslightlylower.
No claim of answer-reliability superiority is supported by these assessments.
The evidence and generation analyses measure different outcomes. Retrieval Hit@10 only asks
whetherat least onealigned reference evidence unit appears within the top 10. A generator can still
omit required facts, answer incorrectly despite sufficient evidence, or abstain. Conversely, an answer
maybegroundedinavailableevidencewithoutsatisfyingamoredetailedgoldresponse. Figure/table
captions may create another mismatch when gold answers require numeric values that are not present
in the caption text.
6.3 Observed computational work
The held-out generation experiment executed 289 unique prompts (376 fixed-arm records) and
recorded1,814.409secondsofaggregatemodel-generationtime,excludingmodelloadingandshared
overhead. Thisisaworkloadtotal,notaper-policyend-to-endlatencycomparison. The46versus
6

188 hybrid-trigger count is the most directly validated resource-use difference for the selected policy.
Withoutconsistentlymeasuredincrementalretrievallatency,hardware-resourceconsumption,and
per-policy generation timing, a time-based cost–quality Pareto curve remains an open measurement
rather than a demonstrated result.
7 Discussion
OurresultsseparatethreeclaimsoftenconflatedinRAGefficiencystudies: retrievingmoreappropriate
evidence, deciding when to incur the cost of a stronger retrieval arm, and improving an answer’s
factual correctness and grounding. The first two are partly supported by the held-out evidence
evaluation: a lightweight policy maintained the always-hybrid hit rate while invoking that arm much
less frequently. The third is not established. The routing regressor was trained against a binary
evidence-hitdifference,whichisnotacalibratedestimateofanswercorrectnessorfaithfulness;the
disconnect is therefore unsurprising in principle and informative empirically.
The small number of rescue and harm instances limits policy learning. The final ridge is deliberately
simple, but its predictions may be weakly identified when almost all training differences are zero.
Moreover, per-article question dependence requires clustered rather than question-independent
uncertaintyestimates. Theseconsiderationsmakeconservativeinterpretationessential. Wedonot
use the post-evaluation held-out results to refit the model or readjust its threshold.
8 Limitations and Reproducibility
Human validation.Generation labels were obtained with an LLM judge using the same model
family as the generator. Only a targeted subset underwent assistant-assisted review; this is not
an independent blinded human-annotator study. A defensible final answer-quality claim requires
independentassessment,acleardisagreementprotocol,andexplicitanalysisofjudgeagreement/error.
Selective faithfulness.The faithfulness mean excludes pure abstentions; it cannot be interpreted
asunconditionalsystem-levelgroundingperformance.Golddisagreement.ConflictingQASPER
annotations and unclear question wording require adjudication; scores with unresolved cases are
omittedfrom thedesignated pairedanalysis.Evidence representation.Image/tablecaption text is
notanadequatesubstituteforactualfigure/tablevalues.Generalizability.The60-articleheld-out
evaluationbelongstoQASPER’sofficialtrainingsplit,andwassubsequentlyinspectedforexploratory
generation analysis. The official test set and any independent second benchmark are untested.Costs.
Aunitizedpenaltyandhybrid-callcountarenotreal-timelatencyortoken-costmeasurements. Finally,
the available evidence does not demonstrate strong statistically conclusive downstream correctness or
faithfulness improvements.
Forauditability,retainedartifactsincludedevelopmentandheld-outquestion/documentidentifiers,
inputandmodelhashes,frozenfeatureorderandthreshold,per-armevidencerankings,immutable
generatedprompts,generatoroutputs,provisionalanswerlabels,andarticle-levelpairedbootstrap
tables.
9 Conclusion
Weevaluatedacost-aware,baseline-feature-onlypolicyforchoosingbetweendense-plus-rerankedand
hybrid-plus-reranked evidenceinscientific-document RAG. Onaheld-out setofQASPER articles,
the policy matched always-hybrid evidence Hit@10 while selecting hybrid for fewer than one quarter
ofquestions. However,exploratorygenerated-answerevaluationsdidnotestablishacorresponding
correctness or faithfulness advantage, and the quality labels remain insufficiently independently
validated. The main empirical lesson is therefore a distinction, not a victory claim: efficient evidence
selection is possible in this setting, but better retrieval metrics cannot be assumed to imply more
reliable generated answers.
References
A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi. Self-RAG: Learning to retrieve, gener-
ate, and critique through self-reflection. InInternational Conference on Learning Represen-
7

tations (ICLR), 2024. URL https://proceedings.iclr.cc/paper_files/paper/2024/
hash/25f7be9694d7b32d5cc670927b8091e1-Abstract-Conference.html.
P.Dasigietal. Adatasetofinformation-seekingquestionsandanswersanchoredinresearchpapers. In
Proceedings of NAACL, 2021. URLhttps://aclanthology.org/2021.naacl-main.365/.
ShahulEs,JithinJames,LuisEspinosaAnke,andStevenSchockaert. RAGAs: Automatedevaluation
ofretrievalaugmentedgeneration. InProceedingsofthe18thConferenceoftheEuropeanChapter
oftheAssociationforComputationalLinguistics: SystemDemonstrations,pages150–158,2024.
doi: 10.18653/v1/2024.eacl-demo.16.
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language models to generate
text with citations. InProceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing, pages 6465–6488, 2023a. doi: 10.18653/v1/2023.emnlp-main.398.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Haofen
Guo,andMengWang. Retrieval-augmentedgenerationforlargelanguagemodels: Asurvey.arXiv
preprint arXiv:2312.10997, 2023b. URLhttps://arxiv.org/abs/2312.10997.
S. Jeong, J. Baek, S. Cho, S. J. Hwang, and J. Park. Adaptive-RAG: Learning to adapt retrieval-
augmented large language models through question complexity. InProceedings of NAACL, pages
7036–7050, 2024. URLhttps://aclanthology.org/2024.naacl-long.389/.
ZhengbaoJiang,FrankXu,LuyuGao,ZhiqingSun,QianLiu,JaneDwivedi-Yu,YimingYang,Jamie
Callan,andGrahamNeubig. Activeretrievalaugmentedgeneration. InProceedingsofthe2023
Conference on Empirical Methods in Natural Language Processing, pages 7969–7992, 2023. doi:
10.18653/v1/2023.emnlp-main.495.
PatrickLewis,EthanPerez,AleksandraPiktus,FabioPetroni,VladimirKarpukhin,NamanGoyal,
Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktaschel, Sebastian Riedel, and Douwe Kiela.
Retrieval-augmentedgenerationforknowledge-intensivenlptasks. InAdvancesinNeuralInfor-
mationProcessingSystems,volume33,pages9459–9474,2020. URL https://proceedings.
neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html.
SewonMin,KalpeshKrishna,XinxiLyu,MikeLewis,Wen-tauYih,PangKoh,MohitIyyer,Luke
Zettlemoyer, and Hannaneh Hajishirzi. FActScore: Fine-grained atomic evaluation of factual
precision in long form text generation. InProceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, pages 12076–12100, 2023. doi: 10.18653/v1/2023.
emnlp-main.741.
D. Ru et al. RAGChecker: A fine-grained framework for diagnosing retrieval-augmented
generation. InAdvances in Neural Information Processing Systems (NeurIPS), vol-
ume37,2024.URL https://proceedings.neurips.cc/paper_files/paper/2024/hash/
27245589131d17368cccdfa990cbf16e-Abstract.html.
ParthSarthi,SalmanAbdullah,AditiTuli,ShubhKhanna,AnnaGoldie,andChristopherD.Manning.
RAPTOR:Recursiveabstractiveprocessingfortree-organizedretrieval. InInternationalConference
on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper_files/
paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke
Zettlemoyer, and Wen-tau Yih. REPLUG: Retrieval-augmented black-box language models.
InProceedings of the 2024 Conference of the North American Chapter of the Association for
ComputationalLinguistics: HumanLanguageTechnologies,2024.URL https://aclanthology.
org/2024.naacl-long.463/.
S.-Q. Yan, J.-C. Gu, Y. Zhu, and Z.-H. Ling. Corrective retrieval augmented generation, 2024. URL
https://arxiv.org/abs/2401.15884.
8
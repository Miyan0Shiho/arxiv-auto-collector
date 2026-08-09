# DocTrace: Towards Traceable Long Document VQA via Hierarchical Evidence Graph Reasoning

**Authors**: Le Xiang, Zhicheng Guan, Hong Chen, Xiaocong Lin, Zhenghua Lei, Teng Hu, Bolei He, Long Zeng

**Published**: 2026-08-04 08:04:59

**PDF URL**: [https://arxiv.org/pdf/2608.03292v1](https://arxiv.org/pdf/2608.03292v1)

## Abstract
Long Document Visual Question Answering (LongDocVQA) requires Multimodal Large Language Models (MLLMs) to locate, integrate, and reason over heterogeneous document elements distributed across multiple pages. Existing approaches, including end-to-end MLLMs, retrieval-augmented generation (RAG) pipelines, and document agents, often lack explicit mechanisms to represent and verify how grounded evidence is progressively composed during reasoning, limiting both answer accuracy and traceability. In this paper, we cast LongDocVQA as an explicit evidence graph reasoning problem rather than implicit answer prediction. To this end, we propose DocTrace, a hierarchical framework that progressively performs evidence localization, structured document parsing, and evidence graph reasoning to enable explicit evidence provenance. To effectively learn these capabilities, we develop a two-stage training framework: joint Supervised Fine-Tuning (SFT) first initializes evidence localization and graph reasoning abilities, followed by task-specific Group Relative Policy Optimization (GRPO) with dedicated rewards to further optimize these capabilities. Extensive experiments on MMLongBench-Doc, LongDocURL, and SlideVQA demonstrate that DocTrace consistently outperforms both existing open-source baselines and proprietary MLLMs. Compared with the Qwen3-VL-8B-Instruct backbone, DocTrace achieves absolute improvements of 14.4, 11.3, and 11.7 points on the three benchmarks, respectively. Beyond competitive performance, DocTrace constructs traceable evidence graphs with explicit node-level provenance, enabling transparent and verifiable reasoning for long document understanding.

## Full Text


<!-- PDF content starts -->

DocTrace: Towards Traceable Long Document VQA via Hierarchical Evidence
Graph Reasoning
Le Xiang1∗, Zhicheng Guan2∗, Hong Chen1†, Xiaocong Lin1, Zhenghua Lei1, Teng Hu1, Bolei He1,
Long Zeng2†
1Baidu Basic Model Research and Development Department, Baidu Inc.
2Tsinghua Shenzhen International Graduate School, Tsinghua University
xiangle@baidu.com, gzc24@mails.tsinghua.edu.cn, chenhong13@baidu.com, zenglong@sz.tsinghua.edu.cn
Abstract
LongDocumentVisualQuestionAnswering(LongDocVQA)
requiresMultimodalLargeLanguageModels(MLLMs)tolo-
cate, integrate, and reason over heterogeneous document ele-
mentsdistributedacrossmultiplepages.Existingapproaches,
including end-to-end MLLMs, retrieval-augmented genera-
tion (RAG) pipelines, and document agents, often lack ex-
plicit mechanisms to represent and verify how grounded
evidence is progressively composed during reasoning, lim-
iting both answer accuracy and traceability. In this paper,
we cast LongDocVQA as an explicit evidence graph rea-
soning problem rather than implicit answer prediction. To
this end, we proposeDocTrace, a hierarchical framework
that progressively performs evidence localization, structured
document parsing, and evidence graph reasoning to enable
explicit evidence provenance. To effectively learn these ca-
pabilities, we develop a two-stage training framework: joint
SupervisedFine-Tuning(SFT)firstinitializesevidencelocal-
izationandgraphreasoningabilities,followedbytask-specific
Group Relative Policy Optimization (GRPO) with dedicated
rewards to further optimize these capabilities. Extensive ex-
perimentsonMMLongBench-Doc,LongDocURL,andSlide-
VQA demonstrate thatDocTraceconsistently outperforms
bothexistingopen-sourcebaselinesandproprietaryMLLMs.
Compared with the Qwen3-VL-8B-Instruct backbone, Doc-
Traceachievesabsoluteimprovementsof14.4,11.3,and11.7
points on the three benchmarks, respectively. Beyond com-
petitiveperformance,DocTraceconstructstraceableevidence
graphswithexplicitnode-levelprovenance,enablingtranspar-
entandverifiablereasoningforlongdocumentunderstanding.
1 Introduction
Long Document Visual Question Answering (Long-
DocVQA) requires Multimodal Large Language Models
(MLLMs)tolocate,integrate,andreasonoverevidencedis-
persed throughout long, visually rich documents containing
heterogeneouselementssuchastexts,tables,charts,andfig-
ures(Tito,Karatzas,andValveny2023;Keetal.2025).The
core challenge of LongDocVQA lies inmulti-hop reason-
ing, where intermediate conclusions derived from a single
page must be combined with complementary evidence scat-
tered across multiple pages to arrive at the final answer.
In high-stakes applications such as financial auditing and
medical analysis, only an accurate standalone prediction is
insufficient. Users must also be able to inspect how each
∗These authors contributed equally.
†Corresponding author.
Figure 1: Overview of DocTrace inference and its traceable
evidencegraph.WhensolvingaLongDocVQAproblem,the
finalanswerremainsfullytraceablebacktoitssourcenodes
and document pages.
conclusion is derived from grounded evidence rather than
treating model predictions as opaque outputs. This require-
ment makes explicit evidence composition as important as
answer accuracy.
Existing LongDocVQA methods can be broadly cate-
gorized into end-to-end MLLMs (Kim et al. 2022; Liu
et al. 2026; Lee et al. 2023), retrieval-augmented genera-
tion (RAG) pipelines, and agentic frameworks. End-to-end
MLLMs implicitly aggregate relevant evidence within hid-
den representations, making it difficult to inspect how indi-
vidualevidencecontributestothefinalprediction.Retrieval-
based approaches improve efficiency by selecting relevant
pages or regions, yet they still treat retrieved evidence as
an unordered context rather than explicitly modeling how
evidence is composed during reasoning. Recent document
agentsexposeintermediateactionsthroughiterativetooluse,
but their reasoning remains trajectory-oriented rather than
explicitly representing dependencies among grounded evi-
dence.Althoughtheseparadigmsdiffersubstantiallyinhow
arXiv:2608.03292v1  [cs.AI]  4 Aug 2026

theyretrieveandprocessdocuments,noneexplicitlymodels
how grounded evidence is progressively composed into the
final answer. Consequently, evidence composition remains
implicit, making the reasoning process difficult to verify,
supervise, and improve.
Humanexpertssolvelongdocumentreasoningbyprogres-
sively narrowing the search space and organizing grounded
evidence into explicit reasoning structures. This naturally
follows a hierarchical coarse-to-fine workflow: evidence lo-
calization,structuredevidenceparsing,andevidencecompo-
sition. More importantly, human reasoning does not merely
accumulate evidence; it explicitly models how individual
evidence supports intermediate conclusions and how these
conclusions jointly lead to the final answer. This observa-
tion suggests that evidence composition should be treated
as an explicit reasoning object rather than remaining hid-
den within latent model representations. Motivated by this
insight, we proposeDocTrace, a hierarchical framework
thatprogressivelyperformsevidencelocalization,structured
parsing, and evidence graph reasoning to explicitly model
evidencecomposition.Specifically,DocTraceprogressively
performsthreestages.Itfirstlocalizesquestion-relevantevi-
dence pages from low-resolution document images, thereby
reducing the reasoning space. Then it parses the selected
pagesintostructuredlayoutelementswithsemanticandspa-
tial information. Finally, it constructs an explicit evidence
graphovertheparsedelementsandtheirvisualcontext,mod-
els intermediate reasoning dependencies, and derives the fi-
nal answer with complete node-level evidence provenance.
The overall pipeline is illustrated in Figure 1.
Learning evidence graph reasoning is challenging be-
cause existing LongDocVQA benchmarks provide supervi-
siononlyforfinalanswers,withoutannotationsforinterme-
diate evidence localization or reasoning graphs. To bridge
this gap, we automatically construct supervision consisting
of grounded evidence pages and evidence graphs, and use
it to jointly initialize evidence localization and graph rea-
soningviamulti-tasksupervisedfine-tuning(SFT).Wethen
refine the model using task-specific Group Relative Policy
Optimization (GRPO) (Guo et al. 2025) with dedicated re-
wards for evidence localization, graph faithfulness and an-
swer correctness. This training paradigm enablesDocTrace
to construct faithful evidence graphs that inherently provide
explicit node-level answer traceability.
Extensive experiments on three long document bench-
marks demonstrate thatDocTraceconsistently outperforms
open-sourcebaselinesandrivalsproprietarymodelssuchas
GPT-4.1 and Claude-3.7-Sonnet. On the ultra-long bench-
markMMLongBench-Doc(Maetal.2024),itachieves52.9
accuracy, improving the backbone by 14.4 points. It also
demonstrates robust performance across varying document
lengths.Beyondaccuracy,DocTraceprovidesexplicitnode-
level evidence provenance, making every prediction trans-
parent and verifiable.
Our core contributions are summarized as follows:
•We proposeDocTrace, ahierarchical coarse-to-fine
frameworkfor LongDocVQA that progressively per-
forms evidence localization, structured document pars-ing,andevidencegraphreasoning.Byexplicitlyorganiz-
ing grounded evidence into traceable reasoning graphs,
DocTrace enables scalable multi-hop reasoning together
with node-level evidence provenance.
•We develop a dedicated training paradigm for evidence
graph reasoning. Specifically, we first jointly initialize
evidence localization and graph reasoning through joint
SFT on automatically generated training data, and then
furtheroptimizebothcapabilitiesviatask-specificGRPO
with dedicated rewards.
•Extensive experiments on MMLongBench-Doc, Long-
DocURL, and SlideVQA demonstrate that DocTrace
consistently outperforms existing open-source methods,
achieving absolute gains of 14.4, 11.3, and 11.7 points
over the Qwen3-VL-8B-Instruct backbone on the three
benchmarks, respectively, while providing explicit node-
level evidence provenance for transparent and traceable
reasoning.
2 Related Work
Long Document Understanding Dataset
Long document understanding challenges models to com-
prehend and reason over lengthy documents, where crucial
evidenceisoftendistributedacrossvariouspagesandmodal-
ities. Early datasets, such asMP-DocVQA, primarily fo-
cus on single-page question answering where the evidence
is confined to a specific page. Subsequent works, such as
DUDE(Van Landeghem et al. 2023),SlideVQA(Tanaka
et al. 2023), extend this task to multi-page reasoning and
unanswerable questions, emphasizing cross-page localiza-
tion, multimodal evidence aggregation, and long-horizon
reasoning. However, these datasets do not assess documents
exceeding 20 pages. More recently,MMLongBench-Doc
andLongDocURL(Deng et al. 2025) have scaled the chal-
lenge to hundred-page scenarios. Despite this progress, ex-
isting training datasets typically provide only final answers
together with coarse page-level evidence annotations, offer-
ing little supervision for how evidence should be composed
during reasoning. Consequently, models must implicitly in-
ferreasoningtrajectoriesfromansweralone.Toaddressthis
limitation, we automatically construct structured evidence
supervision with explicit evidence organization and reason-
ingtrajectories,enablingtraceablelongdocumentreasoning.
Methods for Long Document Understanding
TraditionalEnd-to-endmethods (mPLUGDocOwl2 (Hu
et al. 2025), InternVL3 (Zhu et al. 2025), Qwen3VL (Bai
etal.2025),Gemini3)directlyprocessentiredocumentsby
leveragingextendedcontextwindowsorefficientvisualtoken
compression. Despite their impressive capability, evidence
selection and aggregation remain entirely implicit.Visual
RAGmethods (VisRAG (Yu et al. 2025), SV-RAG (Chen
etal.2024),VDocRAG(Tanakaetal.2025))firstretrievethe
Top-Krelevantpagesbeforeperformingfine-grainedreason-
ing.Recentstudiesfurtherincorporatelayout-awareretrieval,
OCR information, and visual representations to improve re-
trievalquality.However,retrievedevidenceisstilltreatedas

T
TT
Raw Document Evidence BlocksParsing
p8 p13Question: 2nd largest % ?
Evidence Pages: [p8,p13]
585%Answerderived
nodedata
node
QA Synthesis
Evidence GraphGraph Annotation
Selected Evidence sufficient
consistent
drop else Verify
 Drop
single-page
answerable
multi-page
answerableirrelevant
unanswerable
detail-absent
unanswerable
Stage 1: Evidence Localization<evidence_pages>
8,13
</evidence_pages>Input:    
 Question: 2nd 
largest % ?
all pages 
in low_res
Output:<evidence_chain>
 <node=n1 role=data> p8_b1 “xxx”
<node=......>
 <node=n5 role=derived deps=n1,n3> “xxx”
</evidence_chain> 
<reasoning> xxx </reasoning>
<answer> 85% </asnwer>Input:    
p13 p8
Question: 2nd 
largest % ?
evidence pages=== - =-  -=
=== - =-  -=
=== - =-  -=
=== - =-  -=
evidence list
Output:    
Traning Scenarios Stage 3: Evidence Graph Reasoning
MLLM
Policy
Model
Reference
Model
KL Loss(a) Training Data Construction
(b) Joint SFT
InputPass
Include
Question
Pass-rate Filtering DataInput Rollouto1         o2       o3{8,13}  {8,12}  {9} ...
  
...
o1           o2       o3
OutputStage 1
Stage 3{8,13}Rloc Output
{8,12}
{9}1.0
0.5
0.0Rgraph Output Rans Reward Function
Advantage(c) Task-specific GRPOInitializeFigure 2: Overview of our training framework. (a) Training data consturcution using self-collected long docuemnt corpus. (b)
Joint SFT, where the model is supervised fine-tuned on high-quality evidence localization and evidence graph reasoning data.
(c) Task-specific GRPO, where the policy model is optimized by task-specific reward, including localization reward, graph
faithfulness reward and answer correctness reward.
anunorderedcollectionofpagesorchunks,leavingevidence
compositiontobeimplicitlyinferredbyMLLMsandmaking
cross-pagereasoningparticularlychallenging.Morerecently,
Agent-basedmethods(VRAG-RL(Wangetal.2026),Doc-
V∗(Zheng et al. 2026), MM-Doc-R1 (Lin et al. 2026)) have
emerged as a promising paradigm, where models actively
search, navigate, inspect, and iteratively collect information
fromlongdocumentsthroughmulti-stepinteractions.Never-
theless,theseinteractiontrajectoriesdonotexplicitlymodel
how grounded evidence is composed into the final answer.
Overall, existing methods generally represent evidence as
implicit context, retrieved pages, or interaction trajectories.
Explicitmodelingofevidencedependenciesremainslargely
unexplored, limiting both answer traceability and effective
supervision for complex cross-page reasoning.
3 Methodology
Overview
DocTrace casts LongDocVQA as an evidence graph rea-
soning problem. Instead of directly predicting answers from
long documents, it progressively localizes question-relevant
pages, converts them into structured evidence units, and
constructs an evidence graph that explicitly models how
grounded evidence is composed to derive the final answer.
This hierarchical coarse-to-fine design simultaneously re-
duces the reasoning space while providing node-level evi-
denceprovenance.AnoverviewofDocTracetrainingframe-
work is illustrated in Figure 2.DocTrace Framework
Stage1:EvidenceLocalization.Processingalldocument
pagesatnativeresolutioniscomputationallyprohibitiveand
oftenexceedsthecontextcapacityofcurrentMLLMs.There-
fore, DocTrace first performs coarse evidence localization
over uniformly downsampled document pages.
Given the low-resolution document images and question
Q, DocTrace predicts a set of evidence page indices
Pevid=fθloc(Dlow, Q),
whereP eviddenotes the predicted evidence pages. Only
thesepagesareforwardedtosubsequentstages,substantially
reducingthesearchspacewhilepreservingquestion-relevant
information.
Stage 2: Structured Document Parsing.The localized
evidence pages are subsequently processed at their native
resolutionusingadocumentparsingmodel(PaddleOCR-VL-
1.5(Cuietal.2026))toextractfine-grainedlayoutelements.
Each element is represented as
bi= (t i,xi, ci),
wheret idenotesthesemantictype(e.g.,text,table,figure,
or chart),x iis its bounding box, andc iis the extracted
content.Collectively,theseparsedelementsformanevidence
pool
B={b 1, b2, . . . , b M},

which serves as the atomic evidence units for subsequent
graph reasoning.
Stage 3: Evidence Graph Reasoning.Given the struc-
tured evidence pool, DocTrace explicitly constructs an ev-
idence graph to model how grounded evidence is progres-
sively composed into the final answer. Formally,
G= (V, E),
where each node corresponds to either a grounded ev-
idence block fromBor an intermediate reasoning result,
whileeachdirectededgerepresentsareasoningdependency.
The reasoning process is formulated as
P(A, G| B, Q) =P(G| B, Q)P(A|G, Q),
where the model first constructs the evidence graph and
thengeneratesthefinalanswerconditionedonit.Sinceevery
reasoning step is explicitly grounded in document evidence,
the resulting graph naturally provides node-level evidence
provenance for answer verification.
Learning DocTrace
Training Data Construction.Existing LongDocVQA
benchmarks provide only question-answer pairs, making it
impossible to directly supervise evidence localization or ev-
idence graph reasoning. To bridge this supervision gap, we
automatically construct structured supervision consisting of
evidence pages and evidence graphs.
Specifically,wecategorizetrainingsamplesintofourrep-
resentative scenarios according to answerability and rea-
soning complexity: (1) single-page answerable, (2) multi-
pageanswerable,(3)irrelevantunanswerable,and(4)detail-
absent unanswerable. This taxonomy covers both evidence
composition and calibrated refusal behaviors.
For each sample, Gemini 3.1 Pro (Google DeepMind
2026)firstgeneratesgroundedevidencepagestogetherwith
the corresponding evidence graph. GPT-5.5 (OpenAI 2026)
thenindependentlyvalidatesevidencesufficiencyandlogical
consistency. Only verified samples are retained to construct
the final supervision corpus for subsequent training.
JointSFT.Usingthegeneratedcorpus,wejointlyoptimize
evidencelocalizationandevidencegraphreasoningthrough
multi-task SFT. Given low-resolution document pages, the
model predicts evidence page indices; given localized high-
resolutionpagesandtheirparsedlayoutelements,itlearnsto
generate evidence graphs together with the final answers.
This stage provides a strong initialization for subsequent
GRPO alignment.
GRPO AlignmentAlthough joint SFT provides a strong
initialization, it cannot fully optimize evidence localization
and graph reasoning. We therefore further align DocTrace
using task-specific GRPO objectives: Stage1 optimizes evi-
dence page localization, while Stage3 jointly optimizes evi-
dence graph generation and final answer prediction.Localization Reward.Since page localization is ordinal
rather than binary, predictions closer to the ground-truth
pages should receive higher rewards than distant ones. We
thereforeadoptadistance-awaresoftF βrewardthatassigns
partialcredittonearbypredictionswhilefavoringhighrecall.
LetˆPbe the predicted evidence pages parsed from the
<evidence_pages>tagandGtheground-truthevidence
pages. Exact matches contributeh=| ˆP∩G|, while each
unmatched gold page is greedily assigned to its nearest un-
matched prediction and receives a distance-dependent soft
creditd(k), wherek=|ˆp−g|is the page-index distance.
The soft precision and recall are defined as
Prec =h+s
|ˆP|,Rec =h+s
|G|,(1)
wheresistheaccumulatedsoftcredit.TheStage1rewardis
Rstage1=(1 +β2)Prec·Rec
β2Prec + Rec,(2)
whereβcontrols the precision–recall trade-off.
Tojointlyoptimizeevidencegraphgenerationandanswer
prediction, the Stage3 reward is defined as
Rstage3 =λR graph + (1−λ)R answer ,(3)
whereλbalances evidence graph faithfulness and answer
correctness.
Graph Faithfulness Reward.Optimizing only the final
answerdoesnotguaranteereasoningfaithfultothesupport-
ing document evidence. We therefore optimize the gener-
ated evidence graph by jointly evaluating evidence ground-
ing,graphcompleteness,structuralvalidity,anddependency
topology:
Rgraph =w hRhit+wcRcomp+wsRstruct +wtRtopo.
Here,R hitrewards accurate evidence grounding by maxi-
mizingtheoverlapbetweenpredictedandreferenceevidence
nodes.R compencourages complete yet compact evidence
graphsbydiscouragingbothmissingreasoningstepsandre-
dundantnodes.R structenforcesstructuralvalidity,including
executable DAG constraints and valid evidence references.
Finally,R topoencourages reasoning dependencies that are
consistentwiththereferencegraphtopology.Together,these
rewardspromoteevidencegraphsthatarebothfaithfultothe
supporting evidence and structurally consistent.
Answer Correctness Reward.The answer reward opti-
mizes the correctness of the final prediction:
Ranswer =αR exact+ (1−α)R llm,
whereR exactperforms exact matching for deterministic
answers,whileR llmemploysanLLM-basedverifiertoassess
semanticequivalenceforanswerswithflexiblesurfaceforms.
For unanswerable questions, the reward encourages ab-
stention when sufficient supporting evidence is unavailable
and penalizes hallucinated answers, improving prediction
calibration.

Method Backbone Param. ParadigmMMLong. LongDoc. SlideVQA
(Acc) (Acc) (F1)
Closed Source
Gemini-1.5-Pro – – E2E 28.2 50.9 –
GPT-4o – – E2E 42.8 64.5 65.8
GPT-4.1 – – E2E 45.6 – 74.7
Claude-3.7-Sonnet – – E2E 33.9 – 76.3
Open Source
mPLUG-DocOwl2(ACL’25)ViT/LLaMA 8B E2E 13.4 5.3 27.8
M3DocRAG(arXiv’24)Qwen2-VL 7B RAG 21.0 35.1 55.7
VisRAG(ICLR’25)MiniCPM-V-2.6 8B RAG 18.8 41.9 52.4
SV-RAG(ICLR’25)InternVL2 4B RAG 23.0 – 34.3
VDocRAG(CVPR’25)Phi3-Vision 4B RAG 18.4 39.8 42.0
Docopilot(CVPR’25)InternVL2 8B E2E 28.8 – 43.1
InternVL3(arXiv’25)InternViT/Qwen2.5 8B E2E 24.1 38.7 64.4
VRAG-RL(NeurIPS’25)Qwen2.5-VL 7B Agent 26.6 44.9 –
MoLoRAG(EMNLP’25)Qwen2.5-VL 7B RAG 41.0 51.9 –
URaG(AAAI’26)Qwen2.5-VL 7B RAG 33.8 52.2 –
DocSeeker(CVPR’26)Qwen2.5-VL 7B E2E 40.1 51.7 77.1
Doc-V∗(ACL’26)Qwen2.5-VL 7B Agent 42.1 56.3 77.2
MM-Doc-R1(ACL’26)Qwen3 8B Agent 49.7 – –
Ours
Qwen3-VL (Baseline) Qwen3-VL 8B E2E 38.5 45.1 73.4
DocTrace(SFT) Qwen3-VL 8B Agent 50.3+11.853.2+8.183.8+10.4
DocTrace(GRPO) Qwen3-VL 8B Agent 52.9+14.456.4+11.385.1+11.7
Table 1: Performance comparison on three long document understanding benchmarks: MMLongBench-Doc (Acc), Long-
DocURL(Acc),andSlideVQA(F1).Thebestandsecond-bestresultsamongopen-sourcemethodsarehighlightedinboldand
underlined, respectively. Red superscripts denote theabsolute improvementover Qwen3-VL-8B-Instruct.
Theproposedrewardsareappliedduringtheircorrespond-
ing GRPO optimization stages, enabling DocTrace to learn
accurate evidence localization, faithful evidence graph rea-
soning,andreliableanswerpredictioninaunifiedreinforce-
ment learning framework.
4 Experiments
Experimental Setup
Datasets.For training, we construct a multi-stage train-
ing corpus using self-collected long documents (up to 120
pages). It contains synthesized supervision for Stage 1 ev-
idence localization and Stage 3 evidence graph reasoning.
SeeAppendixfor more details.
For evaluation, we evaluate DocTrace on three long
document benchmarks:MMLongBench-Doc(135 docu-
ments, avg. 47.5 pages, 1,082 questions; 33.7% cross-page,
20.9% unanswerable),LongDocURL(396 documents up
to 150 pages, 2,325 QA pairs; 52.9% multi-page), and
SlideVQA(2,215 questions across 20-slide documents;
49.3%multi-hop/numerical).Notably,MMLongBench-Doc
requireswhole-documentreasoning,whereasLongDocURL
uses fixed 30-page windows.
Evaluation Metrics.All methods are evaluated following
theofficial evaluation protocolsof each benchmark. We re-
port Accuracy on MMLongBench-Doc and LongDocURL,
and F1 score on SlideVQA. For fine-grained analysis on
MMLongBench-Doc, we additionally report Page F1 andGTpageCoverage(Cov.)forevidencelocalization,alongside
AccuracyacrossSingle-page,Multi-page,andUnanswerable
question categories for answer generation.
Implementation Details.We adoptQwen3-VL-8B-
Instructas the backbone and perform full-parameter fine-
tuningon16NVIDIAA800GPUs.DuringSFT,Stage1and
Stage3dataarejointlyoptimizedfor3epochswithalearning
rateof5×10−6.DuringGRPO,Stage1andStage3areop-
timized sequentially for 2 epochs each using a learning rate
of5×10−7andagroupsizeof8.ForStage1,thelocaliza-
tionrewardusesβ= 2.0.ForStage3,thegraphfaithfulness
andanswercorrectnessrewardsareweightedby0.3and0.7,
respectively. The maximum sequence length is 32K during
training and 128K during inference. Additional implemen-
tation details and hyperparameter settings are provided in
Appendix.
Baselines.We compare DocTrace against representative
baselines spanning three paradigms: (i)End-to-endmod-
els, including mPLUG-DocOwl2, Docopilot (Duan et al.
2025),InternVL3,andDocSeeker(Yanetal.2026);(ii)RAG
methods, including M3DocRAG (Cho et al. 2024), Vis-
RAG, SV-RAG, VDocRAG, MoLoRAG (Wu et al. 2025),
and URaG (Shi et al. 2026); and (iii)Agent-basedap-
proaches, including VRAGRL, Doc-V∗, and MM-Doc-R1.
We additionally include proprietary MLLMs (Gemini-1.5-
Pro(Teametal.2024),GPT-4o(Hurstetal.2024),GPT-4.1,
and Claude-3.7-Sonnet) as reference points, and report the

Train.Page Loc. QA TypeAcc
F1 Cov. Single Multi Unans.
Baseline – – 46.9 35.3 25.8 38.5
SFT 70.8 62.4 51.7 37.7 68.0 50.3
+ RL (S1)71.3 65.552.6 40.866.0 51.5
+ RL (S1+S3) 70.8 64.653.2 41.1 70.5 52.9
Table 2: Fine-grained performance comparison of the
Qwen3-VL-8B-Instruct baseline and DocTrace at different
training stages on MMLongBench-Doc. “F1” denotes Page
F1, and “Cov.” denotes ground-truth page coverage.
backbone model Qwen3-VL-8B-Instruct for direct compar-
ison. Detailed descriptions of these baseline methods are
provided inAppendix.
Main Results
Comparison with State-of-the-Art Methods.As shown
inTable1,DocTrace(GRPO)consistentlyoutperformsboth
open-source and proprietary models across all three bench-
marks. On the challenging MMLongBench-Doc, DocTrace
achieves the best accuracy of52.9, outperforming the pre-
vious strongest open-source method MM-Doc-R1 (49.7) by
3.2 points and the proprietary model GPT-4.1 (45.6) by 7.3
points.OnLongDocURL,DocTraceachievesthehighestac-
curacyof56.4,slightlysurpassingDoc-V∗(56.3).OnSlide-
VQA, DocTrace establishes a new state of the art with85.1
F1, exceeding Doc-V∗by 7.9 points. These results show
thatexplicitevidencegraphreasoningconsistentlyimproves
long-document understanding across diverse benchmarks.
EffectivenessoftheTrainingParadigm.Wefurthereval-
uate the effectiveness of our two-stage training paradigm,
which combines joint SFT with task-specific GRPO. As
shown in Table 1, joint SFT substantially improves all
three benchmarks over the Qwen3-VL-8B backbone, yield-
ing gains of 11.8, 8.1, and 10.4 points on MMLongBench-
Doc, LongDocURL, and SlideVQA, respectively. Building
on this strong initialization, task-specific GRPO provides
consistentadditionalimprovements,achievingthebestover-
all performance of 52.9, 56.4, and 85.1 on the three bench-
marks.
Analysis of Task-Specific GRPO.To better understand
the complementary effects of task-specific GRPO, Table 2
presents a fine-grained analysis on MMLongBench-Doc.
Applying GRPO to Stage 1 primarily improves evidence
localization, increasing Page F1 from 70.8 to 71.3 and page
coveragefrom62.4to65.5.Betterevidenceretrievalconsis-
tentlybenefitsbothsingle-page(51.7→52.6)andmulti-page
(37.7→40.8) questions. Meanwhile, the accuracy on unan-
swerable questions drops slightly (68.0→66.0), suggesting
that improved evidence recall also encourages more aggres-
sive answering when supporting evidence is insufficient.
ApplyingGRPOtoStage3furtherimprovesanswerqual-
itythroughmoreeffectiveevidenceaggregationandreason-
ing. Although retrieval metrics decrease slightly due to op-
timization trade-offs, answer accuracy continues to improveon single-page (52.6→53.2), multi-page (40.8→41.1), and
especiallyunanswerablequestions(66.0→70.5),resultingin
the best overall accuracy of 52.9.
Together,theseobservationshighlightthecomplementary
roles of the two optimization stages: Stage 1 strengthens
evidence acquisition, whereas Stage 3 improves evidence
utilization and answer reliability.
Analysis of Long-Document Scalability
ToevaluatethescalabilityofDocTraceonincreasinglylong
documents, we conduct a controlled experiment on a subset
of MMLongBench-Doc containing documents longer than
60pages.ForeachtargetlengthW,documentsaretruncated
orpaddedwithanswer-freepageswhilekeepingthequestions
unchanged, isolating the effect of document length.
As shown in Figure 3(a), the performance of the Qwen3-
VL-8B baseline deteriorates rapidly as document length in-
creases, with accuracy dropping from 49.5 to 34.0. In con-
trast, DocTrace consistently outperforms the baseline and
maintains much more stable performance across different
document lengths.
Figure 3(b) provides further insight into this degradation.
Although evidence retrieval coverage gradually decreases
from 70.6 to 50.2 as documents become longer, the an-
swer accuracy conditioned on successful evidence local-
ization remains largely unchanged. This suggests that Doc-
Trace’s evidence graph reasoning remains robust to increas-
ing document length, and that further scalability improve-
ments mainly depend on stronger evidence localization.
Figure 3: Performance comparison on different document
length. (a) Acc; (b) Retrieval Cov. vs. cond Acc.
TraceabilityAnalysisofEvidenceGraphReasoning
ToassessthereliabilityofDocTraceevidencegraphreason-
ing, we perform both qualitative visualizations and quanti-

tative evaluations on MMLongBench-Doc. More details are
provided inAppendix.
AsillustratedinFigure4,DocTraceconstructsanexplicit
evidence graph that links supporting evidence across mul-
tiple document pages through intermediate reasoning steps.
In this example, the model associates demographic infor-
mation on Page 8 with the corresponding Wi-Fi promotion
on Page 13, producing a transparent reasoning path from
grounded evidence to the final answer.
We further evaluate the traceability of the generated evi-
dence graphs from three perspectives: provenance integrity,
evidencelocalization,andcausalfaithfulness(Table3).Doc-
Trace achieves 99.5% evidence grounding accuracy and
99.6% graph integrity, indicating that the generated reason-
ing traces are consistently anchored to valid document evi-
dence.TheevidencelocalizationF1reaches72.5%,demon-
strating effective retrieval of supporting evidence from long
documents. Furthermore, counterfactual evaluation shows
that masking cited evidence flips 82.8% of originally cor-
rectpredictions,whereasmaskinguncitedevidencechanges
only 9.6%. This large gap indicates that the generated evi-
dence graphs capture the evidence that genuinely supports
the model’s predictions.
Figure 4: Qualitative visualization of DocTrace evidence
graph reasoning for a multi-page reasoning task.
Traceability Metric Score Verifier
Provenance Integrity
Evidence Grounding 99.5 Rule
Evidence Graph Integrity 99.6 Rule
Evidence Localization
Evidence Localization F1 72.5 GT
Causal Faithfulness
Evidence Necessity 82.8 C.F.
Evidence Specificity 9.6 C.F.
Table3:QuantitativeevaluationofDocTraceevidencegraph
traceabilityonMMLongBench-Doc.Scoresarepercentages.
C.F. denotes counterfactual evaluation by masking evidence
and re-evaluating the same model.
Ablation Study
We conduct ablation studies on MMLongBench-Doc based
on the SFT model to evaluate the contribution of the struc-tural parsing and evidence graph reasoning components in
DocTrace. The results are summarized in Table 9.
Effect of Core Components.Removing either structural
parsing or evidence graph reasoning consistently degrades
theoverallperformance,reducingtheaccuracyfrom50.3to
46.6 and 47.2, respectively. Both variants suffer substantial
performance drops on multi-page questions, demonstrating
that structured representations and explicit evidence graph
reasoning are essential for integrating dispersed evidence
across long documents. Compared with removing struc-
tural parsing, removing evidence graph reasoning preserves
strongersingle-pageperformance(47.1vs.44.6),suggesting
thatparsedstructuresremaineffectiveforlocalevidenceun-
derstanding, while graph reasoning primarily contributes to
multi-hop reasoning over cross-page evidence. Meanwhile,
both variants achieve higher unanswerable accuracy (74.2
and 74.5), indicating a more conservative prediction behav-
ior that improves unanswerable detection but compromises
answerable question solving.
ComparisonwithVanillaCoT.Replacingevidencegraph
reasoningwithvanillaCoT(Weietal.2022)achieves47.8ac-
curacy.Althoughitslightlyimprovessingle-pageQA(53.2),
it performs worse on multi-page questions (34.8) and unan-
swerable questions (58.2). This suggests that vanilla linear
reasoning can handle local evidence aggregation but strug-
glestopreservestructuredevidencedependenciesandeffec-
tively handle complex long document reasoning.
SettingQA TypeAcc
Single Multi Unans.
DocTrace (SFT) 51.737.768.050.3
w/o Structural Parsing 44.6 30.674.246.6
w/o Graph Reasoning 47.1 30.274.547.2
w/ Vanilla CoT53.234.8 58.2 47.8
Table 4: Ablation study of structural parsing and evidence
graph reasoning on MMLongBench-Doc.
5 Conclusion
In this paper, we cast LongDocVQA as an explicit evidence
graphreasoningproblemandproposeDocTrace,ahierarchi-
calframeworkthatprogressivelyperformsevidencelocaliza-
tion, structured document parsing, and evidence graph rea-
soning. We further develop a two-stage training framework
combiningSFTandtask-specificGRPOtoimproveevidence
acquisition and reasoning capabilities. Experimental results
demonstrate that DocTrace achieves strong and robust per-
formance across multiple long-document benchmarks. Be-
yond competitive performance, DocTrace provides explicit
node-level evidence provenance, enabling transparent and
verifiable reasoning for long document understanding.

References
Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.;
Deng, L.; Ding, W.; Gao, C.; Ge, C.; et al. 2025. Qwen3-vl
technical report.arXiv preprint arXiv:2511.21631.
Chen,J.;Zhang,R.;Zhou,Y.;Yu,T.;Dernoncourt,F.;Gu,J.;
Rossi, R. A.; Chen, C.; and Sun, T. 2024. SV-RAG: LoRA-
contextualizing adaptation of MLLMs for long document
understanding.arXiv preprint arXiv:2411.01106.
Cho, J.; Mahata, D.; Irsoy, O.; He, Y.; and Bansal, M.
2024. M3docrag:Multi-modalretrievaliswhatyouneedfor
multi-page multi-document understanding.arXiv preprint
arXiv:2411.04952.
Cui,C.;Sun,T.;Liang,S.;Gao,T.;Zhang,Z.;Liu,J.;Wang,
X.;Zhou,C.;Liu,H.;Lin,M.;etal.2026. PaddleOCR-VL-
1.5:TowardsaMulti-Task0.9BVLMforRobustIn-the-Wild
Document Parsing.arXiv preprint arXiv:2601.21957.
Deng, C.; Yuan, J.; Bu, P.; Wang, P.; Li, Z.-Z.; Xu, J.; Li,
X.-H.;Gao,Y.;Song,J.;Zheng,B.;etal.2025. Longdocurl:
a comprehensive multimodal long document benchmark in-
tegrating understanding, reasoning, and locating. InPro-
ceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), 1135–
1159.
Duan, Y.; Chen, Z.; Hu, Y.; Wang, W.; Ye, S.; Shi, B.; Lu,
L.;Hou,Q.;Lu,T.;Li,H.;etal.2025. Docopilot:Improving
multimodal models for document-level understanding. In
ProceedingsoftheComputerVisionandPatternRecognition
Conference, 4026–4037.
Google DeepMind. 2026. Gemini 3.1 Pro Model Card.
https://deepmind.google/models/model-cards/gemini-3-1-
pro/. Accessed:July2026.Closed-sourcemultimodalmodel
accessed via official API.
Guo, D.; Yang, D.; Zhang, H.; Song, J.; Wang, P.; Zhu, Q.;
Xu, R.; Zhang, R.; Ma, S.; Bi, X.; et al. 2025. Deepseek-r1:
Incentivizing reasoning capability in llms via reinforcement
learning.arXiv preprint arXiv:2501.12948.
Hu, A.; Xu, H.; Zhang, L.; Ye, J.; Yan, M.; Zhang, J.; Jin,
Q.; Huang, F.; and Zhou, J. 2025. mplug-docowl2: High-
resolutioncompressingforocr-freemulti-pagedocumentun-
derstanding. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Volume 1:
Long Papers), 5817–5834.
Hurst, A.; Lerer, A.; Goucher, A. P.; Perelman, A.; Ramesh,
A.; Clark, A.; Ostrow, A.; Welihinda, A.; Hayes, A.; Rad-
ford, A.; et al. 2024. Gpt-4o system card.arXiv preprint
arXiv:2410.21276.
Ke,W.;Zheng,Y.;Li,Y.;Xu,H.;Nie,D.;Wang,P.;andHe,
Y. 2025. Large language models in document intelligence:
A comprehensive survey, recent advances, challenges, and
future trends.ACM Transactions on Information Systems,
44(1): 1–64.
Kim, G.; Hong, T.; Yim, M.; Nam, J.; Park, J.; Yim, J.;
Hwang, W.; Yun, S.; Han, D.; and Park, S. 2022. Ocr-free
document understanding transformer. InEuropean Confer-
ence on Computer Vision, 498–517. Springer.Lee, K.; Joshi, M.; Turc, I. R.; Hu, H.; Liu, F.; Eisensch-
los, J. M.; Khandelwal, U.; Shaw, P.; Chang, M.-W.; and
Toutanova, K. 2023. Pix2struct: Screenshot parsing as pre-
training for visual language understanding. InInternational
Conference on Machine Learning, 18893–18912. PMLR.
Lin, J.; Hu, K.; Wang, B.; Zhou, Y.; Xi, Z.; Guo, H.; Liu,
S.; Wang, J.; Dou, S.; Zhou, E.; et al. 2026. MM-Doc-
R1: Training Agents for Long Document Visual Question
Answering through Multi-turn Reinforcement Learning. In
Findings of the Association for Computational Linguistics:
ACL 2026, 29770–29783.
Liu,Y.;Yang,B.;Liu,Q.;Li,Z.;Ma,Z.;Zhang,S.;andBai,
X. 2026. Textmonkey: An ocr-free large multimodal model
for understanding document.IEEE Transactions on Pattern
Analysis and Machine Intelligence.
Ma,Y.;Zang,Y.;Chen,L.;Chen,M.;Jiao,Y.;Li,X.;Lu,X.;
Liu, Z.; Ma, Y.; Dong, X.; et al. 2024. Mmlongbench-doc:
Benchmarking long-context document understanding with
visualizations.Advances in Neural Information Processing
Systems, 37: 95963–96010.
OpenAI. 2026. GPT-5.5 Instant System Card. https:
//deploymentsafety.openai.com/gpt-5-5-instant. Accessed:
July 2026. Closed-source model accessed via API; no dedi-
cated arXiv technical report released.
Shi,Y.;Wang,J.;Shan,Z.;Peng,D.;Lin,Z.;andJin,L.2026.
URaG:UnifiedretrievalandgenerationinmultimodalLLMs
for efficient long document understanding. InProceedings
oftheAAAIConferenceonArtificialIntelligence,volume40,
25357–25365.
Tanaka,R.;Iki,T.;Hasegawa,T.;Nishida,K.;Saito,K.;and
Suzuki, J. 2025. Vdocrag: Retrieval-augmented generation
over visually-rich documents. InProceedings of the Com-
puter Vision and Pattern Recognition Conference, 24827–
24837.
Tanaka,R.;Nishida,K.;Nishida,K.;Hasegawa,T.;Saito,I.;
andSaito,K.2023. Slidevqa:Adatasetfordocumentvisual
question answering on multiple images. InProceedings of
the AAAI Conference on Artificial Intelligence, volume 37,
13636–13645.
Team, G.; Georgiev, P.; Lei, V. I.; Burnell, R.; Bai, L.;
Gulati, A.; Tanzer, G.; Vincent, D.; Pan, Z.; Wang, S.;
et al. 2024. Gemini 1.5: Unlocking multimodal understand-
ing across millions of tokens of context.arXiv preprint
arXiv:2403.05530.
Tito, R.; Karatzas, D.; and Valveny, E. 2023. Hierarchi-
cal multimodal transformers for multipage docvqa.Pattern
Recognition, 144: 109834.
Van Landeghem, J.; Tito, R.; Borchmann, Ł.; Pietruszka,
M.; Joziak, P.; Powalski, R.; Jurkiewicz, D.; Coustaty, M.;
Anckaert, B.; Valveny, E.; et al. 2023. Document under-
standing dataset and evaluation (dude). InProceedings of
the IEEE/CVF International Conference on Computer Vi-
sion, 19528–19540.
Wang, Q.; Ding, R.; Zeng, Y.; Chen, Z.; Chen, L.; Wang,
S.;Xie,P.;Huang,F.;andZhao,F.2026. Vrag-rl:Empower

vision-perception-basedragforvisuallyrichinformationun-
derstandingviaiterativereasoningwithreinforcementlearn-
ing.Advances in Neural Information Processing Systems,
38: 57133–57160.
Wei,J.;Wang,X.;Schuurmans,D.;Bosma,M.;Xia,F.;Chi,
E.;Le,Q.V.;Zhou,D.;etal.2022.Chain-of-thoughtprompt-
ing elicits reasoning in large language models.Advances in
neural information processing systems, 35: 24824–24837.
Wu, X.; Tan, Y.; Hou, N.; Zhang, R.; and Cheng, H. 2025.
Molorag: Bootstrapping document understanding via multi-
modal logic-aware retrieval. InProceedings of the 2025
ConferenceonEmpiricalMethodsinNaturalLanguagePro-
cessing, 14035–14056.
Yan,H.;Liu,Y.;Liu,X.;Zhang,Y.;Liao,M.;Wu,J.;Chen,
W.; and Bai, X. 2026. DocSeeker: Structured visual rea-
soning with evidence grounding for long document under-
standing. InProceedings of the IEEE/CVF conference on
computer vision and pattern recognition, 41140–41149.
Yu, S.; Tang, C.; Xu, B.; Cui, J.; Ran, J.; Yan, Y.; Liu, Z.;
Wang, S.; Han, X.; Liu, Z.; et al. 2025. Visrag: Vision-
based retrieval-augmented generation on multi-modality
documents. InInternational Conference on Learning Rep-
resentations, volume 2025, 21074–21098.
Zheng, Y.; Fu, P.; Li, H.; Wang, Z.; Zhang, Y.; Ruan, W.;
Zhang, X.; Wei, Z.; Luo, Z.; Luan, J.; et al. 2026. Doc-V*:
Coarse-to-Fine Interactive Visual Reasoning for Multi-Page
DocumentVQA. InProceedingsofthe64thAnnualMeeting
oftheAssociationforComputationalLinguistics(Volume1:
Long Papers), 45901–45923.
Zhu, J.; Wang, W.; Chen, Z.; Liu, Z.; Ye, S.; Gu, L.; Tian,
H.;Duan,Y.;Su,W.;Shao,J.;etal.2025. Internvl3:Explor-
ing advanced training and test-time recipes for open-source
multimodal models.arXiv preprint arXiv:2504.10479.

Appendix
A Training Data Construction
Overview
This section describes the construction of the supervision
corpususedforbothsupervisedfine-tuning(SFT)andGroup
RelativePolicyOptimization(GRPO).Startingfrompublicly
availableLongDocVQAbenchmarksandself-collectedlong
documents, we automatically generate evidence page anno-
tations and structured evidence graphs through a teacher–
verifier pipeline. The resulting supervision corpus provides
explicitannotationsforevidencelocalization,evidencegraph
reasoning, and answer prediction.
Data Source
The training corpus is automatically constructed from three
publicly available LongDocVQA benchmarks, namely MP-
DocVQA, DUDE, and SlideVQA. We further augment the
training corpus with a self-collected set of long documents,
primarily consisting of publicly available arXiv papers and
technical reports. While the documents in the public Long-
DocVQA benchmarks are generally shorter than 20 pages,
the self-collected corpus includes substantially longer doc-
uments, with document lengths of up to 120 pages. We use
the original documents together with their question–answer
annotations for the public benchmarks as the starting point
for automatic evidence graph generation.
•MPDocVQAcontains scanned multi-page documents
paired with visual question answering annotations. Most
questions require locating fine-grained textual evidence
from one or several pages, making it well suited for su-
pervisingevidencelocalizationanddocumentgrounding.
•DUDEextends document understanding to more chal-
lenging multi-page reasoning scenarios. Its documents
exhibit diverse layouts, including forms, reports, tables,
and visually rich pages, requiring models to retrieve and
integrate evidence distributed across multiple pages.
•SlideVQAfocuses on presentation slides containing fig-
ures, charts, diagrams, bullet lists, and sparse textual
content. Since slide documents often follow non-linear
visual layouts rather than continuous textual flow, they
provide complementary supervision for multimodal rea-
soning over heterogeneous document elements.
•Self-Collected Long Documentsconsist primarily of
publicly available arXiv papers and technical reports.
Compared with the public LongDocVQA benchmarks,
thesedocumentsaresubstantiallylonger(upto120pages)
and exhibit rich cross-page dependencies, providing ad-
ditional supervision for long-range evidence localization
andreasoningbeyondthelengthrangecoveredbyexisting
benchmarks.
Automatic Evidence Graph Generation Pipeline
Given a document and a question, a teacher MLLM first
identifies the evidence pages required to answer the ques-
tion.Itthenperformsfine-grainedreasoningovertheselectedpages and constructs an explicit evidence graph describing
groundedevidencenodes,intermediatereasoningnodes,de-
pendency relations, and the final answer.
Each generated evidence graph consists of four compo-
nents:
•EvidencePages.Thedocumentpageindicesrequiredfor
answering the question.
•EvidenceNodes.Groundeddocumentelementssupport-
ingthe reasoningprocess. Eachnode correspondsto one
parsed document block.
•Derived Nodes.Intermediate reasoning results obtained
by composing one or multiple evidence nodes.
•Dependency Edges.Directed edges that describe how
groundedevidenceisprogressivelycombinedintohigher-
level reasoning results until reaching the final answer.
The resulting graph forms an executable directed acyclic
graph(DAG),whereeveryreasoningstepremainsexplicitly
grounded in document evidence.
To ensure annotation quality, every generated sample is
subsequently validated by an independent verifier MLLM
before being included in the final supervision corpus.
Data Verification
Since evidence graphs are generated automatically, annota-
tionqualityiscontrolledthroughanindependentverification
stage.
For each generated sample, the verifier evaluates three
complementary aspects.
EvidenceSufficiency.Theverifiercheckswhetherthepre-
dicted evidence pages contain sufficient information to an-
swerthegivenquestion.Samplesrequiringadditionalunseen
evidence are discarded.
LogicalConsistency.Theverifierexamineswhetherevery
reasoningstepislogicallysupportedbyitspredecessorsand
whether the dependency graph forms a coherent reasoning
processwithoutmissingorcontradictoryintermediatenodes.
Answer Correctness.Finally, the verifier independently
derives the answer from the generated evidence graph and
compares it with the reference answer. Samples with incon-
sistent predictions are removed.
Only samples satisfying all verification criteria are re-
tained for subsequent SFT and GRPO training.
Data Statistics
Tables 5 and 6 summarize the statistics of the constructed
supervision corpus.
We first report the composition of answerable and unan-
swerablequestions,togetherwiththedistributionofdifferent
reasoningscenarios.Wefurtherpresentstatisticsofevidence
pages,includingtheaveragenumberofdocumentpagesand
evidence pages per sample.

Figure5:Top:PromptsusedforStage1PageLocalizationTraining.Bottom:PromptsusedforStage3EvidenceGraphReasoning
Training.
The resulting supervision corpus covers both answerable
and unanswerable settings, with diverse evidence localiza-
tion complexity ranging from single-page retrieval to cross-
page evidence composition.
Task Ans. Irr. D.A. Total
Stage 117,102
(71.2% SP)1,001 – 18,103
Stage 315,998
(73.4% SP)– 2,718 18,716
Total 33,100 1,001 2,718 36,819
Table 5: Composition of the two-stage training corpus. Irr.
denotes irrelevant-page refusals, D.A. denotes detail-absent
refusals and SP indicates single-page questions.Metric Value
Document Statistics
Avg. document pages 26.1
Max. document pages 120
Evidence Statistics
Avg. evidence pages / QA 1.4
Max. evidence pages / QA 29
Table 6: Document and evidence statistics of the automati-
callyconstructedsupervisioncorpus.Evidencepagesdenote
the pages required to answer a question.
B Additional Experimental Details
Implementation Details
To facilitate reproducibility, we provide additional imple-
mentation details beyond those reported in the main paper.

Parameter Value
Shared
Backbone Qwen3-VL-8B-Instruct
Training Precision BF16
Optimizer AdamW
LR Scheduler Cosine
Warmup Ratio 0.05
Gradient Clipping 1.0
DeepSpeed ZeRO-2
FlashAttention Enabled
Gradient Checkpointing Enabled
Training GPUs 16×NVIDIA A800 (80GB)
SFT
Learning Rate5×10−6
Epochs 3
Max Sequence Length 32K (with sequence packing)
Effective Batch Size 8
Liger Kernel Enabled
GRPO
Learning Rate5×10−7
Epochs 2
Group Size (G) 8
Rollout Temperature 0.8
Rollout Top-p0.95
Max Prompt Length 32K
Clip Range (ϵ low/ϵhigh) 0.2 / 0.28
Effective Batch Size 24
Table 7: Training hyperparameters of DocTrace.
DocTrace is built upon Qwen3-VL-8B-Instruct and uses
full-parameteroptimizationthroughouttraining.Unlessoth-
erwise specified, all experiments are conducted on 16
NVIDIA A800 GPUs with BF16 mixed-precision training.
FlashAttention-2andgradientcheckpointingareenabledfor
memory-efficient long-context training.
Following the hierarchical design of DocTrace, training
consists of one SFT stage followed by two task-specific
GRPO stages. During SFT, Stage 1 evidence localization
and Stage 3 evidence graph reasoning are jointly optimized
with a unified multi-task objective. Stage 1 uses document
images at a fixed resolution of512, while Stage 3 processes
theselectedevidencepagesattheirnativeresolutionof1568
for fine-grained evidence parsing and graph reasoning. Fig-
ure illustrates the prompts used for DocTrace training.
After SFT, reinforcement learning is performed sequen-
tially.Stage1isfirstoptimizedwiththelocalizationreward,
andtheresultingcheckpointisthenusedtoinitializeStage3
GRPO,wheregraphfaithfulnessandanswercorrectnessare
jointlyoptimized.Thesameresolutionsettingsareuseddur-
ingGRPO:512forStage1andnative1568forStage3.This
sequential strategy allows graph reasoning to be optimized
on top of improved evidence localization.
Unless otherwise specified, all experimental results re-
ported in the main paper are obtained from the final Stage 3Parameter Value
Localization rewardβ2.0
Stage 3 rewardλ0.3
Answer rewardα0.5
wh(Hit) 0.55
wc(Completeness) 0.20
ws(Structure) 0.15
wt(Topology) 0.10
Table 8: Reward hyperparameters used during GRPO.
GRPO checkpoint.
Training Hyperparameters
Table 7 summarizes the optimization hyperparameters used
throughout training. The reward hyperparameters for the
Stage1localizationobjectiveandtheStage3evidencegraph
optimization objective are reported separately in Table 8.
GRPO Training Candidate Construction
DirectlyapplyingGRPOtoalltrainingsamplesprovideslim-
ited learning signals, since samples that are always solved
correctlyorconsistentlyfailproducenearlyzerorelativead-
vantage within each rollout group. We therefore construct
informative GRPO training candidates through an offline
rollout procedure.
Stage1CandidateSelection.StartingfromthejointSFT
checkpoint,weperformeightindependentrolloutsforevery
Stage1trainingsample.Samplesthataresolvedcorrectlyin
all eight rollouts or fail in all rollouts are discarded, since
they contribute little useful optimization signal. Only sam-
ples with intermediate success rates (i.e., 1/8 to 7/8 correct
predictions) are retained as Stage 1 GRPO training candi-
dates.
Stage 3 Candidate Selection.After Stage 1 GRPO con-
verges, the resulting checkpoint is used to generate rollouts
on the Stage 3 supervision corpus following the same pro-
cedure.Again,onlysampleswithintermediatesuccessrates
are retained for reinforcement learning.
To further improve training efficiency, we slightly adjust
thesamplingratioofdifferentreasoningscenariosaccording
to the performance of the SFT model. Categories that are
already well mastered (e.g., single-page answerable ques-
tions)aremoderatelydownsampled,whilemorechallenging
categories, including multi-page answerable questions and
detail-absent unanswerable questions, are upsampled. This
curriculum-stylesamplingstrategyallocatesmoreoptimiza-
tion effort to difficult reasoning behaviors that benefit most
from reinforcement learning.
C Additional Ablation Study
The Impact of Set-of-Marks
We conduct an additional ablation experiment to study the
impactoftheSet-of-Marks(SoM)representationinStage3.
The default SoM setting provides visually grounded layout

information by rendering block boundaries and identifiers
on evidence pages, together with cropped visual regions for
chart/figure blocks.
Wecompareitwithatextualbounding-boxvariant,where
the original clean page image is used without visual anno-
tations. Instead, each block is represented by its block id,
semantic label, and normalized bounding-box coordinates
in the prompt, while chart/figure crops are removed. All
other components, including training data, supervision tar-
gets, backbone, and optimization settings, are kept identical
to the SoM baseline.
SettingQA TypeAcc
Single Multi Unans.
DocTrace (SFT)51.7 37.7 68.0 50.3
w/o SoM 50.4 32.5 67.6 48.1
∆-1.3 -5.2 -0.4 -2.2
Table 9: Ablation study of SoM on MMLongBench-Doc.
The results are summarized in Table 9. The textual
bounding-boxvariantconsistentlyunderperformsSoM,with
anoverallaccuracydropof2.2points(48.1vs.50.3),demon-
strating the benefit of explicit visual grounding for fine-
grained evidence localization and document reasoning. The
performance gap is primarily observed on multi-page ques-
tions, where the textual variant drops by 5.2 points (32.5
vs. 37.7). In contrast, single-page questions show only a mi-
nor degradation (50.4 vs. 51.7,−1.3), and unanswerable
questions remain nearly unchanged (67.6 vs. 68.0,−0.4),
suggestingthatSoMmainlybenefitscross-pageevidencein-
tegration.
We attribute the larger gain on multi-page questions to
the stronger spatial grounding provided by SoM. By render-
ing layout blocks and their block ids directly on the page,
SoM establishes an explicit visual correspondence between
reasoning steps and document regions. In comparison, the
textualvariantrequiresthemodeltorecoverthesamespatial
relationships from normalized coordinates, which is more
challenging for cross-page reasoning.
D Detailed Traceability Analysis
This section provides the evaluation details for the trace-
ability results reported in Table 3 of the main paper. We
evaluatetraceabilityfromthreecomplementaryperspectives:
provenanceintegrity,evidencelocalization,andcausalfaith-
fulness. Only samples that successfully reach Stage 3 and
produce a valid evidence chain are included in graph-based
evaluations.
Provenance Integrity
Weverifythestructuralvalidityandgroundingofeachgener-
atedevidencegraphusingdeterministicruleswithoutmodel
calls. Specifically, we check whether (1) all cited block IDs
exist in the Stage 2 layout inventory, (2) all dependency ref-
erencesarevalidandeveryderivednodehasparents,(3)thegraph is acyclic, and (4) the final answer node is connected
to grounded evidence.
Among 1,049 valid chains containing 3,122 evidence
nodes, 99.5% of cited block IDs are successfully grounded.
The graph well-formedness, acyclicity, and answer connec-
tivity rates are 99.6%, 100.0%, and 100.0%, respectively,
indicating that the generated traces are structurally reliable
and well grounded.
Evidence Localization
Weevaluatewhethertheevidenceusedbythereasoningtrace
corresponds to human-annotated evidence pages. For each
answerablesamplewithnon-emptyevidenceannotations,we
collectthepagesreferencedbyevidencenodesandcompute
macro-averagedprecision,recall,andF 1againsttheground-
truthevidencepages.Weadditionallycomparetheresultwith
the Stage 1 retrieved page set.
On 813 answerable samples, the reasoning trace achieves
P/R/F 1= 75.9/71.7/72.5, compared with anF 1score
of 72.2 for Stage 1 retrieval. This shows that the generated
reasoning traces effectively consume the retrievedevidence.
Causal Faithfulness
Wefurthertestwhetherthecitedevidenceisfunctionallynec-
essary for the final prediction through counterfactual mask-
ing. For each sample, we compare two conditions:
•Cited masking: mask the blocks referenced by the evi-
dence graph.
•Uncited masking: mask the same number of grounded
but uncited blocks as a placebo control.
Allotherinputsremainunchanged.Amongoriginallycor-
rectanswerablesampleswithgroundedevidence(n= 363),
319 samples satisfy the cited-masking control condition.
Masking cited evidence changes the answer in 82.8% of
cases(264/319),whereasmaskinguncitedevidencechanges
the answer in only 9.6% of cases (30/313). The substantial
gap indicates that the evidence identified by the graph is
functionally important for the final prediction.
Limitation
Thisevaluationmeasuresbehavioraldependencethroughin-
put perturbation. It does not establish that the model inter-
nallyfollowstheexplicitevidencegraph;rather,itshowsthat
the cited evidence is functionally important for the model’s
prediction.
E Efficiency Analysis
We compare the inference efficiency of DocTrace with
the end-to-end Qwen3-VL-8B baseline on the full
MMLongBench-Doc benchmark. Both methods are evalu-
atedonidenticalhardwarewithbatchsize1.Theend-to-end
baseline processes all document pages at 1024px in a sin-
gle forward pass, whereas DocTrace first localizes evidence
using a Stage 1 scan at 512px, parses the retrieved pages
in Stage 2, and performs fine-grained reasoning over the re-
trieved evidence pages at their native resolution of 1568px
in Stage 3.

Table 10 shows that DocTrace is both more accurate and
more efficient than the end-to-end baseline. It improves ac-
curacyfrom38.5to52.9(+14.4points),whilereducingend-
to-end latency by2.1×(15.04s to 7.33s) and prefill tokens
by2.3×(35.5K to 15.5K). The efficiency gain comes from
the hierarchical design: Stage 1 performs a lightweight scan
overalldocumentpagesusingcompact512pxinputs,while
Stage 3 applies high-resolution reasoning only to the re-
trieved evidence pages. As a result, DocTrace concentrates
computationonpageswherefine-grainedvisualreasoningis
required, rather than processing every page at high resolu-
tion.
Method ACC E2E latency (s) Prefill tok.
Qwen3-VL-8B (E2E) 38.5 15.04 35,543
DocTrace 52.9 7.33 15,481
Stage 1 (512px) – 1.67 9,269
Stage 2 (OCR) – 0.28 –
Stage 3 (1568px) – 5.38 6,212
Table 10: Efficiency comparison on MMLongBench-Doc.
Prefill tokens include both visual and text tokens.
F Motivation for Hierarchical Inference
DocTraceadoptsacoarse-to-fineinferencestrategy:Stage1
performs evidence localization over the entire document us-
ing low-resolution page images (512px), while Stage 3 re-
visits only the retrieved evidence pages at high resolution
(1568px) for fine-grained reasoning. This design is moti-
vated by a fundamental limitation of end-to end inference:
processing every page at high resolution rapidly exhausts
the visual context budget of current MLLMs. To quantify
this limitation, we analyze the context-overflow rate under
different page resolutions and document lengths.
Forapagerenderedatlong-edgeresolutionR,Qwen3-VL
producest R(p) =⌊W R(p)H R(p)/1024⌋+ 8visualtokens,
wherebothimagedimensionsareroundedtomultiplesof32.
A document overflows whenever the total visual and textual
tokens exceed the available context window:
X
ptR(p) +T text> C−M gen−S,(4)
whereC∈ {256K,128K}isthecontextwindow,M gen=
2048reserves generation tokens,S= 64is a safety margin,
andTtext= 512denotesthetextualpromptbudget.Thetoken
accountingisidenticaltothatusedduringinference.Table11
reportstheresultingoverflowratesonMMLongBench-Doc.
Table 11 reveals that the overflow rate increases sharply
with page resolution. Under the 256K setting, 2048px al-
ready overflows for 79.0% of 80–120-page documents and
for all documents longer than 120 pages, while 1568px
also reaches a 100% overflow rate beyond 120 pages. Un-
der the more common 128K deployment, the limitation be-
comesevenmoresevere:2048pxstartstooverflowat40–60
pages, and 1568px overflows for all documents longer than
80 pages. In contrast, 512px never overflows under either
setting.Length (pages)n512 1024 1568 2048
256K context
0–80 954 0.0 0.0 0.0 0.0
80–120 81 0.0 0.0 0.079.0
120+ 56 0.0 8.9 100.0 100.0
128K context
0–40 665 0.0 0.0 0.0 0.0
40–60 153 0.0 0.0 0.0 51.6
60–80 136 0.0 0.0 52.9 90.4
80–120 81 0.0 0.0 100.0 100.0
120+ 56 0.085.7100.0 100.0
Table11:Context-overflowrate(%)onMMLongBench-Doc
under different page resolutions and context-window sizes.
These results directly motivate the hierarchical design of
DocTrace. Stage 1 uses 512px page images to localize ev-
idence over the entire document without context overflow,
while Stage 3 revisits only the retrieved evidence pages at
1568px for fine-grained reasoning. This coarse-to-fine de-
sign enables high-resolution document understanding with-
out sacrificing scalability.
G Stage-wise Error Analysis
TobetterunderstandtheremainingerrorsofDocTrace,weat-
tribute prediction failures to different stages of the pipeline.
We analyze answerable and unanswerable questions sepa-
rately. For answerable questions, we study whether errors
arisefromincompleteevidencelocalization(Stage1)orrea-
soning over retrieved evidence (Stage 3). For unanswerable
questions,weanalyzehowhallucinationsrelatetotheStage1
answerability decision and retrieved distractor pages.
Table 12 shows that Stage 3 generalizes well once the re-
quired evidence has been retrieved. When all gold evidence
pages are available, single-page and multi-page questions
achieve nearly identical accuracy (61.7 vs. 60.2), indicating
thatreasoningitselfisnottheprimarylimitation.Theoverall
performance gap (52.6 vs. 40.0) is instead explained by the
much lower Stage 1 coverage on multi-page questions (46.5
vs. 78.3). Moreover, partial retrieval remains substantially
betterthanretrievingnoevidence(26.7vs.11.2),suggesting
that Stage 3 effectively exploits whatever evidence is avail-
able. Overall, the performance degradation on multi-page
questions is primarily caused by incomplete evidence local-
ization.
TypenCov. Full None Part. Overall
Single 475 78.361.720.0 — 52.6
Multi 355 46.560.211.2 26.7 40.0
Table 12: Performance by retrieval completeness.Cov.de-
notes Stage 1 coverage (Recall = 1).Full,None, andPart.
denote the answer accuracy when all, none, or part of the
gold evidence pages are retrieved, respectively.
Table 13 attributes hallucinations to Stage 1. All hallu-

cinations originate from questions that Stage 1 incorrectly
admitsasanswerable.WheneverStage1correctlyrejectsan
unanswerable question, hallucination never occurs.
AlthoughStage3recovers66.8%ofthewronglyadmitted
cases by predicting a refusal, it cannot completely eliminate
the errors. Furthermore, hallucination increases monotoni-
cally with the number of retrieved distractor pages (0.0%,
28.8%, 39.7%, and 44.4%), indicating that irrelevant re-
trievedpagesmakethedownstreammodelincreasinglylikely
to generate unsupported answers.
SettingnHalluc. (%)
Stage-1 decision
Rejected 240.0
Accepted 220 33.2
Recovered (Stage 3 refused) 147 0.0
Not recovered (hallucinated) 73 100.0
Retrieved pages
0 240.0
1 139 28.8
2 63 39.7
≥318 44.4
Overall 244 29.9
Table13:Hallucinationanalysisonunanswerablequestions.
WithintheAcceptedsubset,wefurtherbreakdownoutcomes
into cases where Stage 3 recovers by predicting a refusal
versus cases that result in a hallucinated answer.
BothanalysesconsistentlyidentifyStage1astheprimary
bottleneck of DocTrace. For answerable questions, the per-
formance gap on multi-page reasoning is mainly caused by
incomplete evidence localization rather than reasoning er-
rors. For unanswerable questions, hallucinations originate
from incorrect Stage 1 answerability decisions and become
more frequent as more distractor pages are retrieved. These
findings suggest that improving Stage 1 retrieval recall and
answerabilitypredictionislikelytoprovidethelargestoverall
performance gain.
H Baseline Details
Evaluation Protocol
Thissectionprovidesadditionaldetailsofthecomparedbase-
linesandclarifiestheevaluationprotocoladoptedinourex-
periments.
ForMMLongBench-Doc, the performance of propri-
etary MLLMs, including GPT-4.1, GPT-4o, Claude-3.7-
Sonnet, and Gemini-1.5-Pro, is directly taken from the of-
ficial MMLongBench-Doc leaderboard. These models are
evaluated under the unified protocol provided by the bench-
mark, ensuring fair comparison across different proprietary
systems.
For the remaining baseline methods, including represen-
tative end-to-end MLLMs, retrieval-augmented approaches,
and agent-based methods, we report the results published in
theiroriginalpaperswhenevertheyareevaluatedonthesame
benchmark.ForDocTraceandtheQwen3-VL-8B-Instructbackbone,
all experiments are conducted by ourselves following the
official evaluation scripts and protocols released by each
benchmark. Specifically, we strictly follow the official eval-
uation procedures of MMLongBench-Doc, LongDocURL,
andSlideVQAwithoutintroducinganytask-specificmodifi-
cations, ensuring reproducible and fair comparisons.
End-to-End MLLMs
End-to-endMLLMsdirectlyprocessthecompletedocument
without performing explicit evidence retrieval or interme-
diate page selection. Evidence localization and multi-page
reasoning are implicitly handled within the model’s long-
context representations, making these approaches heavily
dependent on large context windows and strong multimodal
reasoning capability.
We compare against representative end-to-end document
understanding models, including mPLUG-DocOwl2, Do-
copilot, InternVL3, and DocSeeker.
mPLUG-DocOwl2mPLUG-DocOwl2 is an OCR-free
documentunderstandingmodelthatdirectlyalignsdocument
images with a large language model through a dedicated vi-
sualabstractionmodule.Bylearningunifiedvisual-textrep-
resentations via large-scale instruction tuning, it performs
document reasoning without relying on external OCR sys-
tems.
DocopilotDocopilotfollowsaretrieval-freeparadigmthat
directlyprocessescompletedocumentimageswithinasingle
multimodal model. It combines efficient long-context atten-
tion mechanisms with multimodal data packing, enabling
high-resolution document understanding while avoiding ex-
plicit evidence retrieval.
InternVL3InternVL3 is a general-purpose multimodal
large language model featuring native multimodal pre-
trainingandstrongOCRcapability.Efficientvisualposition
encodingtogetherwithadvancedpost-trainingstrategiesen-
ablescompetitivelong-contextdocumentunderstandingper-
formance across diverse multimodal benchmarks.
DocSeekerDocSeeker emphasizes structured visual rea-
soning and evidence grounding for long document under-
standing. It exploits layout-aware document representations
to improve evidence localization and cross-page reasoning
while maintaining an end-to-end inference framework.
Retrieval-Augmented Methods
Retrieval-augmented methods first retrieve a subset of rel-
evant document pages or visual regions before performing
answergeneration.Byrestrictingexpensivemultimodalrea-
soningtoretrievedevidenceonly,theseapproachessubstan-
tiallyimproveinferenceefficiencycomparedwithprocessing
the complete document.
Our comparison includes representative visual re-
trieval methods, namely M3DocRAG, VisRAG, SV-RAG,
VDocRAG, MoLoRAG, and URaG.

M3DocRAGM3DocRAG formulates long document un-
derstanding as a retrieval-augmented generation problem. It
retrieves relevant document pages through multimodal re-
trieval and performs answer generation only on the selected
evidence, reducing the reasoning space for long documents.
VisRAGVisRAG performs retrieval directly in the visual
domain by representing document pages as images rather
thanOCRtext.Itemploysadual-encoderretrievertoidentify
relevant pages, followed by a vision-language model that
generates answers from the retrieved visual evidence.
SV-RAGSV-RAG unifies retrieval and answer generation
within a single multimodal backbone using two specialized
LoRA adapters. One adapter is optimized for evidence re-
trievalthroughcontrastivelearning,whiletheotherperforms
autoregressive answer generation.
VDocRAGVDocRAG is designed for visually rich doc-
uments by learning dense visual representations for page
retrieval. Retrieved page images are then passed to a mul-
timodal generator for answer prediction, avoiding explicit
conversion of document pages into textual representations.
MoLoRAGMoLoRAG introduces logic-aware multi-
modal retrieval that explicitly considers reasoning depen-
dencies during evidence selection. By jointly modeling re-
trievalandlogicalrelevance,itimprovesevidencequalityfor
complex multi-page reasoning.
URaGURaG proposes a unified retrieval-generation
framework that jointly optimizes evidence retrieval and an-
swer generation within a single training objective, enabling
more effective interaction between retrieval and reasoning.
Agent-based Methods
Agent-basedmethodsformulatelongdocumentunderstand-
ing as an interactive decision-making process. Instead of
processingalldocumentpagessimultaneously,themodelit-
eratively explores the document through search, navigation,
or perception actions, progressively collecting evidence be-
fore generating the final answer.
We compare against representative document agents, in-
cluding VRAG-RL, Doc-V*, and MM-Doc-R1.
VRAG-RLVRAG-RL formulates long document reason-
ing as a sequential decision-making problem. During in-
ference, the agent alternates between reasoning and visual
perceptionactionstoprogressivelycollectevidencefromthe
document.ThepolicyisoptimizedusingGRPOwithrewards
encouraging both accurate retrieval and correct answer pre-
diction.
Doc-V*Doc-V*adoptsacoarse-to-fineinteractivereason-
ing strategy for multi-page document understanding. The
agent progressively narrows the search space through iter-
ative exploration and evidence inspection before producing
the final answer.
MM-Doc-R1MM-Doc-R1 trains document agents
through reinforcement learning for long document visual
question answering. Rather than relying solely on super-
vised instruction tuning, it optimizes multi-turn interactionpolicies that iteratively retrieve and aggregate evidence
before answer generation.
Although these agent-based approaches expose interme-
diate interaction trajectories, their reasoning processes re-
main action-oriented. In contrast, DocTrace explicitly orga-
nizesgroundeddocumentevidenceintoexecutableevidence
graphs, allowing every intermediate reasoning step to be
directly traced back to supporting document evidence and
providing explicit node-level provenance.
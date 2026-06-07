# TrafficRAG: A Multimodal RAG Framework for Traffic Accident Liability Determination

**Authors**: Xu Li, Zedong Fu, Xinyi Li, Xun Han

**Published**: 2026-06-01 06:01:04

**PDF URL**: [https://arxiv.org/pdf/2606.01737v1](https://arxiv.org/pdf/2606.01737v1)

## Abstract
Traffic accident liability analysis is a critical yet challenging task in intelligent transportation and legal assistance. Existing methods often suffer from low efficiency, subjective judgment, and inconsistent analysis results. Meanwhile, large language models are constrained by noisy video inputs and insufficient legal domain knowledge. To address these issues, this work presents TrafficRAG, a multimodal retrieval-augmented framework for automated traffic accident analysis and report generation. Specifically, the proposed framework first adopts a vision-language model to produce structured textual descriptions of accident scenarios, which serve as accurate retrieval queries. Based on these textual queries, a hybrid retrieval strategy integrating BM25 sparse retrieval and dense embedding retrieval is employed to fetch relevant traffic regulations and similar historical cases. Finally, the large language model incorporates retrieved legal knowledge and multimodal accident evidence for comprehensive reasoning, and generates standardized, legally grounded liability analysis reports. Extensive experiments show that TrafficRAG consistently outperforms baseline methods, achieving 77.32% Legal Norm Adaptation Accuracy, 81.71% Factual Faithfulness, and a Liability Ratio MAE of 5.48%. The results validate that integrating multimodal factual evidence with legal clauses via retrieval augmentation can effectively improve the reliability and accuracy of traffic accident liability determination.

## Full Text


<!-- PDF content starts -->

TrafficRAG: A Multimodal RAG Framework for
Traffic Accident Liability Determination
Xu Li1, ZeDong Fu1, Xinyi Li1, and Xun Han2
1Southwest Petroleum University, China
xul@swpu.edu.cn, donghedd206@gmail.com, xinyi0733@outlook.com
2Sichuan Police College, China
hldwxhx@163.com
Abstract.Traffic accident liability analysis is a critical but challeng-
ing task in intelligent transportation and legal assistance. Existing ap-
proaches are often inefficient, subjective, and inconsistent, while large
language models are limited by noisy video inputs and insufficient le-
gal grounding. To address these challenges, we proposeTrafficRAG, a
multimodal retrieval-augmented framework to generate accident analysis
reports.TrafficRAGfirst uses a vision-language model to generate struc-
turedaccidentdescriptions,whichareusedasretrievalqueries.Giventhe
text queries, a hybrid retrieval strategy combining BM25 sparse retrieval
and dense retrieval is used to obtain relevant traffic regulations and sim-
ilar case precedents. Finally, the large language model combines the re-
trieved knowledge and multimodal evidence to perform reasoning and
generate a legally grounded liability analysis report. Extensive experi-
ments demonstrate that our framework consistently outperforms several
strong baselines, achieving 77.32% inLegal Norm Adaptation Accuracy,
81.71% inFactual Faithfulness, and 5.48% inLiability Ratio MAE. Our
work demonstrates that a reasoning approach that combines multimodal
case facts with legal clauses can significantly improve the performance
of liability determination in traffic accidents.
Keywords:Retrieval-Augmented Generation·Traffic Accident Liabil-
ity Determination·Multimodal Fusion.
1 Introduction
Traffic accident liability analysis is a critical task in intelligent transportation
systems and judicial assistance [12,20]. Although the human-driven method
of analyzing surveillance videos for liability determination remains the domi-
nant approach, it is extremely inefficient, labor-intensive, and prone to inconsis-
tent judgments when handling similar cases [4]. Establishing automated analysis
methods enables quick determination of liability, which is crucial to improving
road traffic flow and traffic management efficiency [16]. Recently, multimodal AI
has enabled vision-language models (VLMs) to achieve impressive video under-
standing and event description [3,24,29], and large language models (LLMs) to
excel in legal reasoning and structured report generation [9,2,23]. VLM-basedarXiv:2606.01737v1  [cs.AI]  1 Jun 2026

2 X. Li et al.
and LLM-based methods provide a promising training-free paradigm for improv-
ing the accuracy and efficiency of traffic accident liability determination.
Existing VLM-based approaches typically follow a “video perception→event
description” pipeline. Models including AccidentGPT [24], TrafficVLM [5], and
GPT-4V-based traffic assistants [31] excel at describing accident scenes. How-
ever, they focus on general visual understanding rather than liability-related fact
extraction. In real-world surveillance scenarios, these models are vulnerable to
noise, occlusion, and prompt variations [3,7]. In contrast, LLM-based methods
adopta“textunderstanding→reasoning/reportgeneration” paradigm,perform-
ing well in logical reasoning and structured report generation [9,2,30,23]. Yet
they rely heavily on upstream text quality and implicit knowledge, leading to un-
stable factual consistency and insufficient legal grounding for liability judgment
[1,17].
With the increasing complexity of the road environment, existing automated
methods face several challenges: 1) traffic surveillance videos typically exhibit
substantial noise and redundancy; 2) legal knowledge cannot be exhaustively
represented within model parameters; and the generated analysis results strug-
gle to ensure factual correctness and robust, verifiable legal justification [20,22,
17]. To bridge these challenges, we proposeTrafficRAG, a multimodal retrieval-
augmented framework that processes accident videos, extracts structured de-
scriptions via VLMs, and integrates external legal knowledge and precedents to
generate standardized, legally grounded liability determination reports [13,27,
33].
The main contributions of this paper are as follows:
•We offer an overall multimodal dataset for traffic accident liability analysis.
•We proposeTrafficRAG, a multimodal retrieval-augmented framework for
traffic accident liability determination.
•We validate the effectiveness ofTrafficRAGthrough extensive experiments
and analyze the contribution of different components to performance through
ablation studies.
The remainder of this paper is organized as follows. Section 2 reviews the
related work. Section 3 introduces the proposedTrafficRAGframework. Section
4 presents the experimental setup, dataset construction, evaluation results, and
ablation studies, followed by a discussion of limitations and ethical considera-
tions. Section 5 concludes the paper and describes future research directions.
2 Related Work
2.1 Vision-Language Models and Multimodal Traffic Accident
Analysis
Deep learning-based multimodal accident analysis forms the basis of traffic acci-
dent liability determination. Early CNN-based methods mainly focused on scene
recognition or accident detection, but showed limited ability to capture the tem-
poral evolution of accident events [8,18]. With the development of Transformer-

TrafficRAG 3
based architectures, temporal modeling has become more effective for extracting
accident behavior sequences and dynamic event information [21,5].
More recently, VLMs have opened new directions for multimodal traffic acci-
dent understanding. TrafficVLM improves traffic scene understanding and event
description through enhanced controllability [5], while GPT-4V shows strong
generalization in complex traffic event understanding [31]. Several studies have
also reformulated traffic accident analysis as language-centric tasks for identify-
ing collision causes and risk factors [3,7,30,1]. However, most existing methods
focusongeneralsceneunderstandingoreventdescription,ratherthanextracting
liability-relevant facts for legal reasoning.
2.2 Legal Retrieval Augmentation and Judicial Document
Generation
The accurate use of legal knowledge is essential for liability report generation.
In legal retrieval and text generation, methods such as SAILER and Lawformer
improve case retrieval and long-text semantic modeling [14,26], respectively, but
they are primarily designed for pure-text input and are not directly applicable
to multimodal accident scenarios derived from traffic videos.
Existing legal text generation approaches generally rely on LLMs for legal
prediction and reasoning [23,25], while methods such as SDD-LawLLM fur-
ther enhance performance through task-specific fine-tuning [17]. However, these
methods depend heavily on implicit parametric knowledge and lack explicit ex-
ternal legal grounding. Retrieval-augmented generation (RAG) addresses this
limitation by combining external retrieval with text generation [13]. Recent
methods such as TransRAG and MaRAG have extended RAG to traffic, mar-
itime, and aviation accident analysis [27,33,19]. However, these approaches are
not tailored for liability-oriented accident descriptions extracted from multi-
modal videos. Thus, they cannot fully support the closed-loop process of “fact
extraction→knowledge retrieval→liability report generation”. This research
gap motivates our proposedTrafficRAGframework.
3 Method
3.1 Overall Framework
This work addresses the multi-stage task of generating accident liability deter-
mination reports for accident videos. Given a videoV, the preprocessing module
Pobtains a unified representation, and the VLMM vproduces a structured
accident descriptionx. Usingxas the query, the legal provision retrieverR law
and case retrieverR caseobtain candidatesLcandandCcand. A cross-source con-
sistency reranking moduleR jointselects a compact evidence bundleB, and the
generation modelGoutputs the final reporty.
The overall pipeline is formalized as follows:
(
x=M v(P(V)),Lcand=R law(x),Ccand=R case(x),
B=R joint(x,Lcand,Ccand), y=G(x,B).(1)

4 X. Li et al.
Thegeneratedreportyincludesbasicaccidentinformation,progression,legal
basis, liability apportionment, reasoning, and conclusion. By unifying retrieval
and reranking of legal provisions and cases, our framework improves evidence
coherence for report generation.
3.2 Video Preprocessing and VLM-Based Accident Description
Generation
The video preprocessing moduleP(which handles frame sampling, denoising,
and feature encoding)aims to reduce noise in raw traffic accident videos, stan-
dardize the input to the VLM, and preserve the temporal continuity of accident
events. As shown in Fig. 1,Pfirst samples frames from the raw videoVat a
fixed frame rate to remove redundancy. The sampled frames are then denoised
and normalized, after which the frame sequence is encoded into dense visual
features using CLIP (openai/clip-vit-large-patch14). To balance temporal
coverage and input compactness, the system retains a limited set of key frames
and maps them into a unified visual representation space.
BFrame Preprocessing
timefps,duration,resolution 
AVideo Input & Sampling
 CCLIP Feature Encoding DPreprocessing Metadata
field
[   ]
sampled_frame_count/
original_fps
duration_sec
feature_dim
F Temporal Phase Parsing EVLM -Based Accident 
UnderstandingCore
vehicle action 
description …
pedestrian action 
description …Phase1 Phase2 Phase3
start_sec
end_sec
captionGStructured Accident 
Description GenerationCore
1)Basic Info
2)Accidenttime/location/
lighting
vehicle actions/
collision pointsAlignment
Confidence
High
Medium
Low
High Medium Lowsampled frames preprocessed frames CLIP visual features preprocessing metadata
semantic fieldSemantic 
FlowSemantic 
FlowSemantic 
Flow
Visual 
FlowVisual 
Flowframe evidence 
HStructured 
Description Output
raw accident description
structured accident 
description
(for downstream legal retrieval )
Fig. 1.Video preprocessing and structured accident description extraction pipeline.
To generate accident descriptions for liability determination, we designM v
with a visual-encoder–language-decoder architecture. As illustrated in Fig. 1,
the model first produces a raw accident description and then refines it into
a structured, liability-oriented description. The visual encoder extracts scene
and motion features, including interactions among traffic participants, while the
language decoder maps these features into a natural-language descriptionx.
To focus the model on liability-relevant information, we formulate this stage
as a multimodal generation task guided by prompts. Specifically,M vis guided
to extract four types of information: scene background, participants, behavioral
sequence, and liability-related cues. The resulting structured descriptionxpre-
serves key accident facts and serves as the query for subsequent legal retrieval
and evidence selection.

TrafficRAG 5
3.3 Dual-Path Knowledge Retrieval with Cross-Source Consistency
Reranking
Given the liability-oriented accident descriptionx, we retrieve relevant legal
provisions and similar cases through a dual-path hybrid retriever that combines
BM25 sparse retrieval with dense vector retrieval. This design balances lexical
matching and semantic similarity. For dense retrieval, we usemoka-ai/m3e-base
as the encoder and build the index with FAISS.
For a queryxand a candidate documentd, we compute the sparse score
Ssparse (x, d)and the dense scoreS dense(x, d). After Min–Max normalization over
the top-Kcandidates, the retrieval score is defined as
Sret(x, d) =α ˜Sdense(x, d) + (1−α) ˜Ssparse (x, d),(2)
whereα∈[0,1]controls the trade-off between dense and sparse retrieval.
To improve evidence coherence, we further introduce a cross-source consis-
tency reranking module. For each candidate law-case pair(l i, cj), the joint score
is
Sjoint(li, cj|x) =λ 1Sfact(x, c j) +λ 2Snorm(x, li) +λ 3Scons(li, cj),(3)
whereS fact(x, c j)is the cosine similarity between m3e-base embeddings of the
accident description and the retrieved case,S norm(x, li)uses the same encoder
to assess applicability of the legal provision to the accident description, and
Scons(li, cj)is scored via fixed logic rules based on liability consistency (with no
learned parameters or cross-encoder). The weightsλ 1, λ2, λ3and thresholdτ,
detailed in Section 4.1, are tuned on the validation set. The top-ranked pairs are
retained as the final evidence bundleB.
3.4 Liability Determination Report Generation Module
We formulate liability report generation as a conditional text generation task
under external knowledge constraints. As shown in Fig. 2, the generator uses a
structuredprompttoproducereportswithstandardizedsections,includingbasic
accident information, accident progression, evidence and legal basis, liability
determination, liability allocation, and the final conclusion.
The generation objective is
max
θTX
t=1logp θ(yt|y<t, x,B),(4)
wherey tis thet-th token in the target report,θdenotes the model parameters,
xis the structured accident description, andBis the reranked evidence bundle.
To reduce hallucinations and improve controllability, we adopt structured
prompting and require the generated report to ground its legal reasoning and
liability conclusions inB. During generation, we enforce three forms of con-
sistency: factual consistency between the report and the accident facts, legal
consistency between the conclusions and the retrieved evidence, and structural
consistency across different sections of the report.

6 X. Li et al.
Traffic Regulation 
Knowledge Base
Statutory provisions / traffic 
regulations
Judicial Case 
Knowledge Base
Historical accident cases / 
precedents
Traffic Accident 
Video
Raw accident videoLiability -Oriented 
Accident Description 
Generation
Video preprocessing + VLM -
based structured extractionDual -Path Candidate Retrieval
Candidate legal provision 
retrieval + candidate case 
retrieval
Structured 
Accident 
Description
Liability -oriented 
description from 
VLM
Cross -Source Consistency 
Reranking
 Coherent 
Evidence 
Bundle
Consistent statutes 
and casesTrafficRAG Framework
Accident 
FactsAnalogous 
Cases
Legal 
Provisions
Liability 
ConclusionGenerated 
Liability Report
Traffic Accident 
Liability 
Determination 
Report
• Basic information
• Accident 
progression
• Evidence and legal 
basis
• Liability 
determination
• Liability allocation
• Final conclusion
Fig. 2.Liability determination report generation pipeline.
4 Experiments
4.1 Experimental Setup and Training
Dataset ConstructionWe employ a standardized curation pipeline to con-
struct a dedicated multimodal traffic accident dataset, integrating public ac-
cident understanding benchmarks with legal resources for liability attribution
and case retrieval. Video data are collected from VRU-Accident[10] and TAU-
106K[32], while legal resources are assembled from CADD[20], STARD[22], and
LeCaRDv2[15]. Candidate videos are filtered by visual quality and annotation
validity before being combined with metadata and preliminary VLM-generated
descriptions into a unified annotation framework.
Annotatorsrefinetheautomateddescriptionsintoliability-focusedstructured
representations and complete standardized reference reports that include appli-
cable legal provisions, liability allocation, and evidence keywords. Ambiguous
cases with unclear facts or insufficient legal support are carefully verified, re-
vised, or excluded to ensure data reliability.
Thefinaldatasetcomprises1,584trafficaccidentcases,witheachsamplecon-
taining raw accident videos, metadata, structured descriptions, official liability
reports, and supporting legal information. We split the dataset at the case level
to form disjoint training, validation, and test sets, preventing sample overlap
and retrieval leakage.
Knowledge Base ConstructionTo support retrieval-augmented generation,
we build two external knowledge bases: a legal knowledge base and a case knowl-
edge base. The legal knowledge base includes 265 national traffic regulation pro-
visions related to liability determination, collected from the Road Traffic Safety
Law and its Implementation Regulations. The knowledge base consists of 671
traffic accident cases, each with accident descriptions, liability judgments, rea-
soning, penalties, and legal references. Public judicial documents were standard-
ized, filtered, and deduplicated, and overlapping cases were excluded to avoid
retrieval leakage.

TrafficRAG 7
Training and Hyperparameter SettingsIn video preprocessing, raw videos
are uniformly sampled at 3 FPS and encoded by CLIP ViT-L/14. For the vision-
language component, we adopt an InternVL2-style 8B backbone asM vand fine-
tune it with LoRA for liability-oriented accident description generation, using
rank 64, learning rate2×10−4, batch size 16, and 3 epochs. The output is
constrained to four fields: scene background, participants, behavioral sequence,
and liability-related cues.
Forretrieval,wecombineBM25withm3e-basedenseretrievalanduseFAISS
for vector indexing, with sparse and dense scores fused by Eq. (2) usingα=
0.6. In the cross-source consistency reranking stage, the weightsλ 1, λ2, λ3and
thresholdτare selected on the validation set. For report generation, we use
Qwen3-Maxwith a unified template to produce liability determination reports.
4.2 Evaluation Metrics
To evaluate the full pipeline from video input to liability report generation, we
assess performance from four aspects:evidence coverage,legal retrieval quality,
legal-factual consistency, andliability ratio prediction accuracy.
We use five metrics: Key Evidence Recall (KER) for evidence coverage [23],
Statute Recall (SR) for legal provision retrieval or citation [22], Factual Faithful-
ness (FF) for factual support from accident facts and retrieved knowledge [11,
6], Legal Norm Adaptation Accuracy (LNA) for whether the cited provisions
support the final liability conclusion [28], and Liability Ratio Mean Absolute
Error (LR-MAE) for liability ratio prediction. Higher values indicate better per-
formance for KER, SR, FF, and LNA, while lower values indicate better perfor-
mance for LR-MAE. Detailed symbol definitions and formulas are provided in
Appendix A.
4.3 Overall Result Analysis
We compareTrafficRAGwith several baselines for legal reasoning and retrieval-
augmented generation. Since most baselines are designed for text input rather
than raw videos, all methods are evaluated under a unified video-to-text setting.
To reduce confounding factors, we use the same test set, retrieval pool size, and
report structure constraints for all methods whenever applicable.
Table 1 shows thatTrafficRAGachieves the strongest overall results among
thecomparedmethods,obtainingthebestscoresonKER(82.87%),SR(84.79%),
LNA (77.32%), and FF (81.71%), together with the lowest LR-MAE (5.48%).
These results suggest that combining liability-oriented fact extraction with con-
sistency aware legal evidence selection improves fact coverage, legal grounding,
and liability allocation in traffic accident liability report generation.
4.4 Ablation Study Analysis
To investigate the contributions of external knowledge and consistency-aware
evidence selection, we conduct ablation experiments on the two retrieval sources,

8 X. Li et al.
Table 1.Performance comparison of different methods on the traffic accident liability
determination task.
Method KER (%) SR (%) LNA (%) FF (%) LR-MAE (%)
BM25(2024) – 34.00 22.81 – –
QLD(2024) – 34.41 23.60 – –
SAILER(2023) – 72.33 46.69 – –
LawRAG(2024) 74.17 70.03 55.83 53.65 34.22
Judge(2025) 66.67 42.00 67.91 69.97 25.89
DeepSeek-V3.2 77.31 81.83 72.41 72.41 10.74
Gemini-3.1-Pro 72.30 69.81 64.24 76.99 7.31
TrafficRAG 82.87 84.79 77.32 81.71 5.48
i.e.,thelegalknowledgebase(LegalKB)andthecaseknowledgebase(CaseKB),
as well as the proposed cross-source consistency re-ranking module.
Table 2.Ablation study results on the retrieval sources and the cross-source consis-
tency reranking module.
Legal KB Case KB Rerank KER (%) SR (%) LNA (%) FF (%) LR-MAE (%)
× × × 51.87 45.42 43.19 59.44 31.81
×✓× 75.28 62.73 45.91 73.65 15.69
✓× × 67.54 74.88 62.37 79.26 23.58
✓ ✓× 81.46 82.11 69.68 79.02 10.36
✓ ✓ ✓82.87 84.79 77.32 81.71 5.48
As shown in Table 2, removing both retrieval sources yields the weakest over-
all performance, indicating that accident descriptions alone are insufficient for
reliable liability determination. Using only the Case KB substantially reduces
LR-MAE and improves FF, suggesting that similar cases support practical rea-
soning and liability allocation, although SR and LNA remain limited without
explicit legal grounding. In contrast, using only the Legal KB markedly improves
SR and LNA, confirming the importance of statutory support, but liability ratio
prediction remains weaker than that of the full model.
Using both knowledge sources without reranking provides complementary
legal and analogical evidence, though the retrieved statutes and cases may lack
strict alignment. In contrast, by enabling cross-source consistency reranking, the
model achieves its best performance, demonstrating that mutually supportive
law-case pairs effectively refine both liability conclusions and allocation.
These results highlight two forms of complementarity: the Legal KB provides
explicit normative grounding, the Case KB contributes practical analogical evi-
dence,andthererankingmodulefurtherstrengthenstheirinteractionbyfiltering
weakly aligned law-case pairs.
To further illustrate this effect, we analyze a representative door-opening
collision in which an opened car door strikes a following non-motor vehicle. As
shown in Fig. 3, direct generation without external knowledge tends to produce
vague conclusions and incomplete legal support. Using only the Legal KB im-

TrafficRAG 9
Accident ScenarioCandidate and Reranked Evidence
Candidate Legal Provisions (top -k)
Door Opening Rules
[relevant statute]Non-Motor Vehicle Right of Way General Safety Duty
Candidate Similar Cases (top -k)
Rear Collision
[analogous case] [partially aligned] [analogous case] [kept after reranking]Door Opening Accident Intersection Collision
Key Differences
No KB lacks explicit legal 
grounding and often yields 
incomplete conclusions.
Case only improves 
practical reasoning but 
weakens statute support.
Law only improves legal 
grounding but lacks 
practical detail.
Independent retrieval may 
keep weakly aligned 
evidence.
TrafficRAG reranks 
evidence for more coherent 
conclusions.
Case Description
Representative case: parked car opens door and collides with a non -motor 
vehicle approaching from behind.[kept after reranking] [relevant statute] [kept after reranking]Case Scene Summary
[relevant statute] [filtered out]
[weak alignment] [filtered out]Final reranked evidence bundle: Door Opening Rules + Non -Motor Vehicle Right of Way + Door Opening Accident
Average performance  on door -opening collision scenarios
Fig. 3.Qualitative analysis of a representative door-opening collision scenario. Traffi-
cRAG selects more coherent legal and case evidence and produces liability conclusions
closer to the reference determination.
proveslegalgroundingbutlackspracticaldetail,whereasusingonlytheCaseKB
yields more plausible handling logic but weaker statutory support. By combin-
ing both sources with cross-source consistency reranking,TrafficRAGproduces
reports that better match the reference determination in terms of liability con-
clusion, legal basis, and liability allocation.
5 Limitations and Ethical Considerations
TrafficRAG is a legal assistance prototype rather than a substitute for traffic
administrators or legal experts, and its outputs require human review to be
legallyvalid.CurrentlylimitedtoChinesetrafficlawsandcases,ithaslimitations
such as video understanding error propagation, retrieval bias, unstable legal
applicability, and potential biases from historical cases or incomplete evidence.
We addressed these issues by anonymizing sensitive information and annotating
legal articles with jurisdiction and time validity. Future work includes cross-
jurisdictional validation, human-in-the-loop supervision, and impact evaluation.
In short, TrafficRAG is an auxiliary tool instead of an autonomous decision
system.
6 Conclusion and Future Work
In this paper, we proposedTrafficRAG, a multi-stage framework for accident
liability determination report generation. Experimental results show that the
proposed framework consistently outperforms multiple baseline methods on the

10 X. Li et al.
benchmark task. In future work, we plan to enhance the multimodal understand-
ing and evidence reasoning capabilities, introduce stricter rule-based legal ver-
ification and confidence calibration mechanisms, and optimize the cross-source
consistency modeling to further improve the reliability and interpretability of
the framework for real-world traffic accident liability determination.
A Notation Symbols and Evaluation Metrics
Let thei-th sample be indexed byi. We denote byE iand ˆEithe gold and
extracted liability-related evidence sets, byS iand ˆSithe gold and generated
cited statute sets, byr iandˆr ithe gold and generated liability conclusions, and
bya(j)
iandˆa(j)
ithe gold and predicted liability proportions of thej-th liable
party, respectively. Moreover, ˆFidenotes the set of extracted factual claims,I(c)
is an indicator function,m iis the number of liable parties in thei-th sample,
andNis the total number of samples.
Table 3 summarizes the evaluation metrics used in this work.
Table 3.Detailed Formulas of Evaluation Metrics
Metric Formula
Key Evidence Recall (KER) KER =1
NPN
i=1|Ei∩ˆEi|
|Ei|
Statute Recall (SR) SR =1
NPN
i=1|Si∩ˆSi|
|Si|
Legal Norm Adaptation Accuracy
(LNA)Sample-level:LNA i
=(
1,if( ˆSi∩Si̸=∅)∧(ˆr i=ri),
0,otherwise
Overall:LNA =1
NPN
i=1LNA i
Factual Faithfulness (FF) FF =1
NPN
i=11
|ˆFi|P
c∈ˆFiI(c)
Liability Ratio Mean Absolute Error
(LR-MAE)LR-MAE =1
NPN
i=11
miPmi
j=1ˆa(j)
i−a(j)
i
References
1. Abdelrahman, A.S., Abdel-Aty, M., Yang, S., Faden, A.: Advanced crash causation
analysis for freeway safety: A large language model approach to identifying key
contributing factors. arXiv preprint arXiv:2505.09949 (2025)
2. Ahmadi, E., Muley, S., Wang, C.: Automatic construction accident report analysis
using large language models (llms). Journal of Intelligent Construction3(1), 1–10
(2025)

TrafficRAG 11
3. Akter, S., Shihab, I.F., Sharma, A.: Large language models for crash detec-
tion in video: A survey of methods, datasets, and challenges. arXiv preprint
arXiv:2507.02074 (2025)
4. Chen, J., Lu, S., Zhong, L.: An autonomous intelligent liability determination
method for minor accidents based on collision detection and large language models.
Applied Sciences14(17), 7716 (2024)
5. Dinh, Q.M., Ho, M.K., Dang, A.Q., Tran, H.P.: Trafficvlm: A controllable visual
language model for traffic video captioning. In: Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. pp. 7134–7143 (2024)
6. Es, S., James, J., Anke, L.E., Schockaert, S.: Ragas: Automated evaluation of
retrieval augmented generation. In: Proceedings of the 18th conference of the eu-
ropean chapter of the association for computational linguistics: system demonstra-
tions. pp. 150–158 (2024)
7. Fan, Z., Wang, P., Zhao, Y., Zhao, Y., Ivanovic, B., Wang, Z., Pavone, M., Yang,
H.F.: Learning traffic crashes as language: Datasets, benchmarks, and what-if
causal analyses. arXiv preprint arXiv:2406.10789 (2024)
8. Fang, J., Qiao, J., Xue, J., Li, Z.: Vision-based traffic accident detection and antici-
pation:Asurvey.IEEETransactionsonCircuitsandSystemsforVideoTechnology
34(4), 1983–1999 (2023)
9. Huang, X., Feng, Y., Zhang, Z.: Chatgpt-based method for generating automobile
accident reports. In: 2024 4th international conference on electronic information
engineering and computer science (EIECS). pp. 1174–1177. IEEE (2024)
10. Kim, Y., Abdelrahman, A.S., Abdel-Aty, M.: Vru-accident: A vision-language
benchmark for video question answering and dense captioning for accident scene
understanding. In: Proceedings of the IEEE/CVF International Conference on
Computer Vision. pp. 761–771 (2025)
11. Kryściński, W., McCann, B., Xiong, C., Socher, R.: Evaluating the factual consis-
tency of abstractive text summarization. In: Proceedings of the 2020 conference on
empirical methods in natural language processing (EMNLP). pp. 9332–9346 (2020)
12. Lajmi, S.: Towards transparent and efficient accident resolution: a hybrid deep
learning–ontology framework for automated damage detection and culpability rea-
soning. Journal of Umm Al-Qura University for Engineering and Architecture
16(4), 1555–1574 (2025)
13. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., Yih, W.t., Rocktäschel, T., et al.: Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in neural information processing systems
33, 9459–9474 (2020)
14. Li, H., Ai, Q., Chen, J., Dong, Q., Wu, Y., Liu, Y., Chen, C., Tian, Q.: Sailer:
structure-aware pre-trained language model for legal case retrieval. In: Proceedings
of the 46th International ACM SIGIR Conference on Research and Development
in Information Retrieval. pp. 1035–1044 (2023)
15. Li, H., Shao, Y., Wu, Y., Ai, Q., Ma, Y., Liu, Y.: Lecardv2: A large-scale chinese
legal case retrieval dataset. In: Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Information Retrieval. pp. 2251–2260
(2024)
16. Liu, S., Zhang, Z.J., Yu, Z.H.: Research on liability identification system of road
traffic accident. Journal of Computers33(1), 215–224 (2022)
17. Ma, H., Lu, Y., Xiao, Z., Feng, J., Zhang, H., Yu, J.: Sdd-lawllm: Advancing
intelligentlegalsystemsthroughsyntheticdata-drivenfine-tuningoflargelanguage
models. Electronics14(4), 742 (2025)

12 X. Li et al.
18. Melegrito, M., Reyes, R., Tejada, R., Anthony, J.E.S., Alon, A.S., Delmo, R.P.,
Enaldo, M.A., Anqui, A.P.: Deep learning based traffic accident detection in smart
transportation: a machine vision-based approach. In: 2024 4th International Con-
ference on Applied Artificial Intelligence (ICAPAI). pp. 1–6. IEEE (2024)
19. Ren,T.,Zhang,Z.,Jia,B.,Zhang,S.:Retrieval-augmentedgeneration-aidedcausal
identification of aviation accidents: A large language model methodology. Expert
Systems with Applications278, 127306 (2025)
20. Shen,Y.,Wu,Z.:Cadd:Achinesetrafficaccidentdatasetforstatute-basedliability
attribution. arXiv preprint arXiv:2511.11715 (2025)
21. Singh, T., Chakraborty, P., Truong, L.T.: Surveillance video-based traffic accident
detection using transformer architecture. arXiv preprint arXiv:2512.11350 (2025)
22. Su, W., Hu, Y., Xie, A., Ai, Q., Bing, Q., Zheng, N., Liu, Y., Shen, W., Liu,
Y.: Stard: A chinese statute retrieval dataset derived from real-life queries by
non-professionals. In: Findings of the Association for Computational Linguistics:
EMNLP 2024. pp. 10658–10671 (2024)
23. Su, W., Yue, B., Ai, Q., Hu, Y., Li, J., Wang, C., Zhang, K., Wu, Y., Liu, Y.:
Judge: Benchmarking judgment document generation for chinese legal system. In:
Proceedings of the 48th International ACM SIGIR Conference on Research and
Development in Information Retrieval. pp. 3573–3583 (2025)
24. Wu, K., Li, W., Xiao, X.: Accidentgpt: Large multi-modal foundation model for
traffic accident analysis. arXiv preprint arXiv:2401.03040 (2024)
25. Wu, Y., Zhou, S., Liu, Y., Lu, W., Liu, X., Zhang, Y., Sun, C., Wu, F., Kuang, K.:
Precedent-enhanced legal judgment prediction with llm and domain-model collab-
oration. In: Proceedings of the 2023 conference on empirical methods in natural
language processing. pp. 12060–12075 (2023)
26. Xiao, C., Hu, X., Liu, Z., Tu, C., Sun, M.: Lawformer: a pre-trained language
model for chinese legal long documents. ai open 2: 79–84 (2021)
27. Yang, J., Dai, X., Lv, Y., Kovács, L., Wang, F.Y.: Transrag for parallel trans-
portation: toward reliable and trustworthy transportation systems via retrieval-
augmented generation. Frontiers of Information Technology & Electronic Engi-
neering26(1), 20–26 (2025)
28. Yu,W.,Lin,X.,Ni,L.,Cheng,J.,Sha,L.:Benchmarkingmulti-steplegalreasoning
and analyzing chain-of-thought effects in large language models. arXiv preprint
arXiv:2511.07979 (2025)
29. Zhang, R., Wang, B., Zhang, J., Bian, Z., Feng, C., Ozbay, K.: When language and
vision meet road safety: leveraging multimodal large language models for video-
based traffic accident analysis. Accident Analysis & Prevention219, 108077 (2025)
30. Zhen, H., Shi, Y., Huang, Y., Yang, J.J., Liu, N.: Leveraging large language models
withchain-of-thoughtandpromptengineeringfortrafficcrashseverityanalysisand
inference. Computers13(9), 232 (2024)
31. Zhou, X., Knoll, A.C.: Gpt-4v as traffic assistant: An in-depth look at vision lan-
guage model on complex traffic events. arXiv preprint arXiv:2402.02205 (2024)
32. Zhou, Y., Bai, L., Cai, S., Deng, B., Xu, X., Shen, H.T.: Tau-106k: A new dataset
for comprehensive understanding of traffic accident. In: The Thirteenth Interna-
tional Conference on Learning Representations (2025)
33. Zou, Y., Wang, S., Li, G., Ning, B.: Marag: a knowledge graph and retrieval-
augmented framework for maritime accident analysis. Ocean Engineering344,
123622 (2026)
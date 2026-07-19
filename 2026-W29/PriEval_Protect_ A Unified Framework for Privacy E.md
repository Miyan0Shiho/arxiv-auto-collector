# PriEval-Protect: A Unified Framework for Privacy Evaluation and Protection in Healthcare Systems

**Authors**: Ilef Chebil, Asma El Hadj, Souheib Yousfi, Aroua Hedhili, Layth Sliman

**Published**: 2026-07-15 12:12:48

**PDF URL**: [https://arxiv.org/pdf/2607.13754v1](https://arxiv.org/pdf/2607.13754v1)

## Abstract
Safeguarding patient privacy while enabling meaningful healthcare data use remains critical under GDPR and HIPAA. Existing compliance methods are manual, error-prone, and separate policy audits from data-level assessments. This paper presents PriEval-Protect, a two-phase framework for unified privacy risk evaluation and mitigation. The evaluation phase combines regulatory compliance scoring using a fine-tuned legal LLM with RAG, and technical analysis via encryption type, data architecture, and metrics including similarity, uncertainty, adversary success, and information gain/loss. A composite risk score uses weighted aggregation via Analytic Hierarchy Process. The protection phase recommends countermeasures including federated learning and differential privacy based on assessed risk. Results on hospital documents and datasets demonstrate regulation-aligned, explainable assessments, bridging legal conformance and data-level risk analysis.

## Full Text


<!-- PDF content starts -->

PriEval-Protect: A Unified Framework for
Privacy Evaluation and Protection in Healthcare
Systems
Ilef Chebil1[0009-0002-3637-0567], Asma Elhadj1[0009-0000-0388-6570],
Souheib Yousfi2[0000-0003-0810-0869], Aroua Hedhili3[0000-0002-6918-0797],
and Layth Sliman2[0000-0003-3369-7302]
1National Institute of Applied Science and Technology INSAT, Tunis, Tunisia
ilef.chebil@insat.ucar.tn, asma.elhadj@insat.ucar.tn
2Efrei Research Lab, Efrei, Paris-Panthéon-Assas University, Paris, France
souheib.yousfi@efrei.fr, layth.sliman@efrei.fr
3LARIA Research Lab, National School Of Computer Science ENSI, Tunis, Tunisia
aroua.hedhili@ensi-uma.tn
Abstract.Safeguardingpatientprivacywhileenablingmeaningfulhealth-
caredatauseremainscriticalunderGDPRandHIPAA.Existingcompli-
ance methods are manual, error-prone, and separate policy audits from
data-level assessments. This paper presents PriEval-Protect, a two-phase
framework for unified privacy risk evaluation and mitigation. The eval-
uation phase combines regulatory compliance scoring using a fine-tuned
legal LLM with RAG, and technical analysis via encryption type, data
architecture,andmetricsincludingsimilarity,uncertainty,adversarysuc-
cess, and information gain/loss. A composite risk score uses weighted ag-
gregation via Analytic Hierarchy Process. The protection phase recom-
mends countermeasures including federated learning and differential pri-
vacy based on assessed risk. Results on hospital documents and datasets
demonstrate regulation-aligned, explainable assessments, bridging legal
conformance and data-level risk analysis.
Keywords:Privacy Evaluation·GDPR·HIPAA·LLM·RAG·Fed-
erated Learning·Differential Privacy
1 Introduction
Healthcare systems increasingly rely on data-driven infrastructure for clinical
decisions and research, yet must operate within strict regulatory boundaries
like GDPR4and HIPAA5. Current privacy evaluation remains fragmented and
under-automated, with tools emphasizing either manual policy audits or isolated
technical metrics without meaningful integration.
4https://gdpr-info.eu/
5https://www.hhs.gov/hipaa/for-professionals/index.html
arXiv:2607.13754v1  [cs.CR]  15 Jul 2026

2 I. Chebil et al.
RegulatoryframeworkslikeDigitalHealthAssessmentFramework6andHITRUST
CSF7offer structured evaluations but lack technical integration. Technical meth-
ods like Differential Privacy (DP)—adopted by Apple8—provide mathemati-
cal guarantees, while Federated Learning (FL) via frameworks like PySyft9,
Flower10, and IBM FL11enables decentralized training. However, these solu-
tions fail to unify policy compliance with technical mitigation.
Our Contributions.
– Unified Two-Phase Framework:Combines (i) evaluation via regulatory
scoring (fine-tuned legal LLM with RAG) and technical risk assessment (en-
cryption, architecture, similarity,uncertainty, adversary success, information
gain/loss), and (ii) protection recommending context-specific countermea-
sures (masking, Federated Learning FL, Differential Privacy DP) based on
risk level.
– Theoretical Integration:Comprehensive privacy taxonomy integrating
regulatory, architectural, and technical dimensions with Multi-Criteria De-
cision Making (MCDM) based risk aggregation using Analytic hierarchy pro-
cess (AHP)[20,16].
– Empirical Validation:Evaluated on hospital policy documents and med-
ical datasets, demonstrating regulation-aligned compliance scores and tai-
lored protection strategies.
2 Related Work
Prior research divides into: (i) regulatory compliance, (ii) privacy evaluation
(attribute classification and metrics), and (iii) protection techniques.
Regulatory compliance efforts like the Benjumea Privacy Scale [7] provide
structured GDPR scoring via 14 weighted items validated by experts, but re-
main manual and error-prone. Privacy evaluation methods from Wagner and
Eckhoff [21] categorize 80+ metrics (uncertainty, information gain/loss, adver-
sary success), yet lack healthcare-specific calibration. Tools like Pycanon [17]
implement limited metrics requiring extensive preprocessing.
Attribute classification is essential for privacy metrics. Simões [18] explored
autoencoders, custom Deep Neural Networks (DNNs), and LLMs for Quasi-
Identifiers/Sensitive Attributes (QID/SA) detection, showing promise but suf-
fering from computational cost and interpretability issues. Two-step rule-based
algorithms [2] offer superior performance, transparency, and healthcare adapt-
ability.
6https://orchahealth.com/our-products/assessment-frameworks/
digital-health-assessment-framework-dhaf/
7https://hitrustalliance.net/hitrust-framework/
8https://machinelearning.apple.com/research/scenes-differential-privacy
9https://github.com/OpenMined/PySyft
10https://flower.ai/
11https://github.com/IBM/federated-learning-lib

PriEval-Protect: Privacy Framework for Healthcare 3
Protection techniques like FL [3] and DP [10] provide strong guarantees but
lack context-aware application. Table 1 summarizes existing approaches, high-
lightinggapsinintegrationbetweencompliance,metrics,andadaptiveprotection
that PriEval-Protect addresses.
Table 1.Comparative Summary of Existing Approaches
Criterion Benjumea [7] Wagner [21] Two-step [2]
Domain Compliance Tech Metrics Attribute
Class.
Strengths Weighted scor-
ingComprehensive Transparent
Limitations Manual No domain focus Threshold
tuning
Table 2.Comparative Summary (Continued)
LLM-based
[18]FL, DP [3,10]
Attribute Classi-
ficationProtection Tech-
niques
Context-aware Strong guaran-
tees
Prompt-sensitive Not adaptive
3 Background and Preliminaries
Legal Frameworks.GDPR establishes EU data protection via principles like
lawfulness, transparency (Art.5-6,13), purpose limitation, minimization, security
(Art.32), impact assessments (Art.35), and breach notification (Art.33-34) [4].
HIPAAgovernsU.S.PHIprivacythroughtechnical,administrative,andphysical
safeguards [19].
DataClassifications.Healthcaredatasetscontain:Quasi-Identifiers(QIDs)—
indirect identifiers (age, ZIP);Sensitive Attributes (SAs)—directly sensitive
fields (disease, income);Non-Sensitive Attributes (NSAs)—negligible risk
attributes [25,1].

4 I. Chebil et al.
PrivacyMetrics.Quantifyleakagerisk[21]:DataSimilarity(S)—k-anonymity,
l-diversity;Uncertainty(U)—entropyofsensitiveattributes;AdversarySuc-
cess(P adv)—re-identificationprobability;InformationGain/Loss(IG/IL)—
mutual information or KL divergence changes.
AI Components.LLMsare transformer-based models for language understand-
ing and generation [24], supporting compliance verification [11].RAGmerges
LLMs with external retrieval for grounded responses [13].Vector Databases
store embeddings for semantic search [12].Fine-Tuningadapts pre-trained
LLMs via domain pretraining, supervised tuning, or parameter-efficient methods
like LoRA with quantization [5,14,22,9].
Privacy-Preserving Techniques.Differential Privacy (DP)quantifies privacy
viaPr[M(D 1)∈S]≤eϵ·Pr[M(D 2)∈S] +δfor neighboring datasetsD 1, D2,
injecting calibrated noise [10,15].Federated Learning (FL)trains models
across institutions without raw data exchange [3,23,6].
MCDM.Multi-Criteria Decision Making provides structured frameworks for
evaluating conflicting criteria. AHP translates qualitative judgments into quan-
titative weights via pairwise comparisons [20,16].
4 Method
4.1 Framework Overview
PriEval-Protect integrates regulatory and technical privacy evaluation in two
phases. Phase 1 computes composite risk scores from policies and data; Phase
2 recommends tailored countermeasures. Figure 1 shows the pipeline: policy
documents and health data are evaluated through legal LLMs and technical
metrics,producingscoresthatguideprotectionstrategies(DP,FL,orcombined).
Fig. 1.PrivEval-Protect Framework Overview.

PriEval-Protect: Privacy Framework for Healthcare 5
4.2 Phase 1: Privacy Risk Evaluation
Phase 1 evaluates privacy via structured taxonomy covering regulatory, archi-
tectural, and technical factors (Figure 2). Users upload datasets and policy doc-
uments, configuring encryption type (AES-256, RSA, hybrid) and data architec-
ture (central, federated, distributed). The system computes privacy metrics and
regulatory compliance scores, fused via weighted formula.
Fig. 2.Privacy Risk Evaluation Taxonomy and Workflow.
Privacy Metrics EngineCustom Python implementations compute formal
metrics from literature. After attribute classification via two-step rule-based
algorithms [2], the module calculates: (1) Adversary Success Rate and Delta
Presence, (2) k-Anonymity, alpha-k-Anonymity, l-Diversity, t-Closeness, (3) In-
formation Gain and Conditional Entropy, (4) Shannon and normalized entropy.
Multiprocessing optimizes large-scale processing.
Regulatory Compliance AssessmentAutomated GDPR/HIPAA alignment
combinesfine-tunedLLMwithRAGpipelineusingQdrantforsemanticretrieval.
Dataset and Training.We curated: 9,000+ labeled GDPR policy segments,
violation reports, and article corpus with Q&A expansions. Preprocessing in-
cluded normalization, article extraction, harmonization, and stratified oversam-
pling (27:1 imbalance). Final corpus: 11,447 instances, 54% compliant, 200+
indexed articles.
Two-stage fine-tuning: (1) Domain-Adaptive Pretraining on GDPR corpus
using Saul-7B-Instruct, (2) Supervised Fine-Tuning on labeled examples with
LoRA (6.8M parameters), 4-bit quantization, and gradient checkpointing.
RAG Pipeline and Scoring.Structure-aware chunking preserves legal context.
Chunks are embedded (BAAI/bge-base-en-v1.5, L2-normalized) and stored in

6 I. Chebil et al.
Qdrant. For each policy chunk, top-k GDPR articles are retrieved via cosine
similarity. Compliance score:
Score=kX
i=1si·(1−P i)(1)
wheres iis similarity andP iis penalty. Retrieved articles prompt the fine-tuned
LLM, producing compliant/non-compliant labels with rationales and references
(Figure 3).
Fig. 3.RAG-based compliance assessment pipeline.
Score AggregationFinal privacy risk via weighted aggregation:
Final Score=α·C compliance +β·M technical +γ·E encryption +δ·A architecture (2)
Weights determined by AHP whereα+β+γ+δ= 1. Risk categories: Low
(80–100%), Moderate (60–79%), High (0–59%).

PriEval-Protect: Privacy Framework for Healthcare 7
4.3 Phase 2: Privacy Protection
Risk-based protection mapping (Table 3): Low risk applies data masking; Mod-
erate uses DP with noise injection; High activates FL+DP for collaborative
training with encrypted updates.
Table 3.Privacy Protection Strategies
Risk Technique Description
Low Data Masking Redacts sensitive fields
Moderate Differential Privacy Adds calibrated noise
High FL + DP Decentralized training with noise
5 Experimental Setup
To empirically validate the PriEval-Protect framework, we conducted a multi-
faceted evaluation combining real institutional data and expert assessments from
healthcare organizations. The setup examines three key dimensions: (i) regula-
tory compliance scoring, (ii) technical privacy metrics computation, and (iii)
unified privacy risk classification.
We utilized two categories of authentic institutional materials:
– HealthcarePolicyDocuments:Internalprivacyanddata-sharingpolicies
from multiple clinical institutions, used as input for the GDPR compliance
assessment module. These documents represent diverse policy styles and
governance maturity levels.
– Clinical Data Samples:Structured, anonymized patient datasets compris-
ing demographic, clinical, and administrative fields. These were processed
through our privacy metrics engine to quantify re-identification and infer-
ence risks.
For compliance scoring, we used the Benjumea Privacy Scale [7] as the base-
line reference, aligning our LLM-based scoring model with established GDPR
compliance indicators. For technical privacy metrics, the framework’s compu-
tation engine was benchmarked against Pycanon [17] and ARX [8], two widely
recognized tools for evaluating anonymization and re-identification risks in data
publishing. For global privacy risk classification, multiple privacy and data pro-
tectionexperts—includingcomplianceofficersandhealthcareinformationspecialists—
were asked to manually assign risk levels (Low, Moderate, High) to various
dataset scenarios. These labels were used as ground truth to validate the in-
terpretability and accuracy of the aggregated PriEval-Protect risk scores.

8 I. Chebil et al.
Table 4.Evaluation criteria across PriEval-Protect components
Component Evaluation Metrics
Compliance Scoring Mean Absolute Error (MAE), Pearson correlation
Privacy Metrics Absolute deviation from PyCanon/ARX
Global Risk Score Weighted agreement score
6 Results and Discussion
ToevaluatetheeffectivenessofPriEval-Protect,weperformedacross-component
assessment covering compliance, technical, and integrated privacy risk evalua-
tion. The quantitative and qualitative findings are summarized in Table 5.
6.1 GDPR Compliance Assessment
The compliance scoring module achieved a Mean Absolute Error (MAE) of 1.32
compared to expert-derived Benjumea Privacy Scale judgments, and a Pearson
correlation of 0.78, indicating a strong positive linear relationship between pre-
dicted and expert scores. These results confirm that the LLM-based analysis
reliably captures regulatory intent and contextual interpretation across diverse
policy documents, validating its utility for semi-automated GDPR alignment.
6.2 Technical Privacy Metric Benchmark
The lightweight privacy metric engine demonstrated high consistency with for-
mal tools. Deviations from Pycanon and ARX outputs were 0.6 and 0.4 respec-
tively,indicatingaccuratecomputationofprivacymeasuressuchask-anonymity,
l-diversity, and t-closeness. This result highlights the framework’s potential for
integration into real-time data governance pipelines without compromising ana-
lytical rigor.
6.3 Global Privacy Risk Validation
Comparing system-generated global risk classes to expert annotations yielded an
agreementrateof82.6%,witharecallof81.2%inhigh-riskcases—demonstrating
that PriEval-Protect effectively identifies sensitive data configurations requiring
stronger safeguards. These results suggest that the unified scoring model gener-
alizes well beyond specific datasets or local contexts.
6.4 Key Insights and Limitations
The experimental outcomes validate PriEval-Protect as a robust framework ca-
pable of unifying legal and technical privacy evaluations. While results indicate
strong agreement with human expertise, future studies should extend this val-
idation to unstructured modalities (e.g., clinical narratives and imaging data)

PriEval-Protect: Privacy Framework for Healthcare 9
and broader institutional datasets. The framework’s modular design makes it
adaptable to varying compliance regimes, supporting its scalability across inter-
national healthcare ecosystems.
Table 5.Summary of Evaluation Results
Component Metric Value
ComplianceMAE (vs. BPS) 1.32
Pearson Correlation 0.78
Privacy MetricsDeviation (PyCanon) 0.6
Deviation (ARX) 0.4
Global ScoreClass Agreement 82.6%
High-Risk Recall 81.2%
7 Conclusion
PriEval-Protect advances privacy-aware AI in healthcare by integrating regula-
tory compliance, technical metrics, and data architecture analysis. It bridges le-
gal,technical,andoperationalperspectivesthroughLLM-drivenscoring,weighted
aggregation,andpracticalprotectionstrategies(encryption,FL,DP).Theframe-
work enables automated, explainable privacy governance beyond manual audits.
Future directions include extending to additional data types, enhancing automa-
tion,andreal-worldclinicaldeployment.PriEval-Protectdemonstratesthatscal-
able, automated privacy evaluation is achievable and necessary for responsible
digital health innovation.
Acknowledgments.This research was supported by INSAT, Efrei Research Lab, and
LARIA Research Lab at ENSI.
Disclosure of Interests.The authors have no competing interests to declare.
References
1. Privacy-preserving healthcare informatics: a review. In: ICMSA 2021. ITM Web of
Conferences (2021)
2. Two-step rule-based classification algorithm for privacy-preserving data mining.
Security and Communication Networks p. 7154705 (2021)
3. Federated learning applications in healthcare: A systematic review. International
Journal of Environmental Research and Public Health20(15), 6539 (2023)
4. Amini, M.M., et al.: Ai ethics and challenges in healthcare under gdpr. Machine
Learning and Knowledge Extraction5(3), 1023–1035 (2023)

10 I. Chebil et al.
5. Anisuzzaman, D.M., et al.: Fine-tuning large language models for specialized use
cases. Mayo Clinic Proceedings: Digital Health3(1), 100184 (2025)
6. Antunes, R.S., et al.: Federated learning for healthcare: Systematic review and
architecture proposal. ACM Transactions on Intelligent Systems and Technology
13(4), 54:1–54:23 (2022)
7. Benjumea, J., et al.: Privacy scale for mobile health applications. JMIR mHealth
and uHealth10(7), e37591 (2022)
8. Christen, P., et al.: Putting statistical disclosure control into practice: The arx data
anonymization tool. In: Medical Data Privacy Handbook, pp. 111–148 (2015)
9. Dettmers, T., et al.: Qlora: Efficient finetuning of quantized llms. In: Advances in
Neural Information Processing Systems. vol. 36, pp. 10088–10115 (2023)
10. Dwork, C., Roth, A.: The algorithmic foundations of differential privacy. Founda-
tions and Trends in Theoretical Computer Science9(3-4), 211–407 (2014)
11. Hassani, S.: Enhancing legal compliance and regulation analysis with llms. In: RE
2024. pp. 507–511. Reykjavik (2024)
12. Le, M., et al.: A survey on vector databases. arXiv preprint arXiv:2310.11703
(2023)
13. Lewis, P., et al.: Retrieval-augmented generation for knowledge-intensive nlp. In:
Advances in Neural Information Processing Systems. vol. 33, pp. 9459–9474 (2020)
14. Ohri, K., Kumar, M.: Supervised fine-tuned approach for automated detection of
diabetic retinopathy. Multimedia Tools and Applications83, 14259–14280 (2024)
15. Ponomareva, N., et al.: How to dp-fy ml: A practical guide to machine learning
with differential privacy. Journal of Artificial Intelligence Research77, 1113–1201
(2023)
16. Saaty, T.L.: Decision making with the analytic hierarchy process. International
Journal of Services Sciences1(1), 83–98 (2008)
17. Sáinz-Pardo Díaz, J., López García, Á.: A python library to check the level of
anonymity of a dataset. Scientific Data9, 785 (2022)
18. Simões, A.: Automated Attribute Classification for Privacy Metrics. Ph.D. thesis,
University of Coimbra (2024)
19. Smith, D.A., Abbasi, N.: Cybersecurity in healthcare and hipaa compliance. Jour-
nal of Knowledge, Learning and Science and Technology3(3), 278–287 (2024)
20. Taherdoost, H., Madanchian, M.: Multi-criteria decision making (mcdm) methods
and concepts. Encyclopedia3(1), Article 1 (2023)
21. Wagner, I., Eckhoff, D.: Technical privacy metrics: A systematic survey. ACM
Computing Surveys51(3), 1–38 (2018)
22. Wu, H., et al.: Domain-adaptive pretraining methods for dialogue understanding.
arXiv preprint arXiv:2105.13665 (2021)
23. Xu, J., et al.: Federated learning for healthcare informatics. Journal of Healthcare
Informatics Research5, 1–19 (2021)
24. Zhao, W.X., et al.: A survey of large language models. arXiv preprint
arXiv:2303.18223 (2023)
25. Zhu, H., et al.: Multiple sensitive attributes publishing with guaranteed utility.
CAAI Transactions on Intelligence Technology8(2), 288–296 (2023)
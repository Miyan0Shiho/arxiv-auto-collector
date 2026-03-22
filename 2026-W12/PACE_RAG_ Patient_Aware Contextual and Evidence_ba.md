# PACE-RAG: Patient-Aware Contextual and Evidence-based Policy RAG for Clinical Drug Recommendation

**Authors**: Chaeyoung Huh, Hyunmin Hwang, Jung Hwan Shin, Jinse Park, Jong Chul Ye

**Published**: 2026-03-18 04:40:53

**PDF URL**: [https://arxiv.org/pdf/2603.17356v1](https://arxiv.org/pdf/2603.17356v1)

## Abstract
Drug recommendation requires a deep understanding of individual patient context, especially for complex conditions like Parkinson's disease. While LLMs possess broad medical knowledge, they fail to capture the subtle nuances of actual prescribing patterns. Existing RAG methods also struggle with these complexities because guideline-based retrieval remains too generic and similar-patient retrieval often replicates majority patterns without accounting for the unique clinical nuances of individual patients. To bridge this gap, we propose PACE-RAG (Patient-Aware Contextual and Evidence-based Policy RAG), a novel framework designed to synthesize individual patient context with the prescribing tendencies of similar cases. By analyzing treatment patterns tailored to specific clinical signals, PACE-RAG identifies optimal prescriptions and generates an explainable clinical summary. Evaluated on a Parkinson's cohort and the MIMIC-IV benchmark using Llama-3.1-8B and Qwen3-8B, PACE-RAG achieved state-of-the-art performance, reaching F1 scores of 80.84% and 47.22%, respectively. These results validate PACE-RAG as a robust, clinically grounded solution for personalized decision support. Our code is available at: https://github.com/ChaeYoungHuh/PACE-RAG.

## Full Text


<!-- PDF content starts -->

PACE-RAG: Patient-Aware Contextual and Evidence-based
Policy RAG for Clinical Drug Recommendation
Chaeyoung Huh1,∗Hyunmin Hwang1,∗
Jung Hwan Shin2Jinse Park3,†Jong Chul Ye1,†
{lirisnoir, hyunmin_hwang}@kaist.ac.kr
neosjh2009@gmail.com jinsepark@gmail.com jong.ye@kaist.ac.kr
Abstract
Drug recommendation requires a deep under-
standing of individual patient context, espe-
cially for complex conditions like Parkinson’s
disease. While LLMs possess broad medical
knowledge, they fail to capture the subtle nu-
ances of actual prescribing patterns. Existing
RAG methods also struggle with these com-
plexities because guideline-based retrieval re-
mains too generic and similar-patient retrieval
often replicates majority patterns without ac-
counting for the unique clinical nuances of indi-
vidual patients. To bridge this gap, we propose
PACE-RAG(Patient-AwareContextual and
Evidence-based PolicyRAG), a novel frame-
work designed to synthesize individual patient
context with the prescribing tendencies of simi-
lar cases. By analyzing treatment patterns tai-
lored to specific clinical signals, PACE-RAG
identifies optimal prescriptions and generates
an explainable clinical summary. Evaluated
on a Parkinson’s cohort and the MIMIC-IV
benchmark using Llama-3.1-8B and Qwen3-
8B, PACE-RAG achieved state-of-the-art per-
formance, reaching F1 scores of 80.84% and
47.22%, respectively. These results validate
PACE-RAG as a robust, clinically grounded so-
lution for personalized decision support. Our
code is available at: https://github.com/
ChaeYoungHuh/PACE-RAG.
1 Introduction
Drug recommendation represents a complex clin-
ical challenge that necessitates a highly personal-
ized strategy tailored to the unique clinical pre-
sentation of each patient (Twala, 2025). Although
*Equal contribution.
†Corresponding author.
1Korea Advanced Institute of Science and Technology,
KAIST, Republic of Korea
2Department of Neurology, Seoul National University Hos-
pital, Republic of Korea
3Haeundae Paik Hospital, Inje University, Republic of
Korea
Guideline 
RAG
TreatRAG
PACE -RAG
(ours)  
Medical
KnowledgeEvidence -basedPersonalized
Medicine
Medical
KnowledgeEvidence -based
MedicinePersonalized
Medicine
Figure 1:Comparison of medical RAG approaches.
While Guideline RAG leverages external medical knowl-
edge, its recommendations tend to be generalized for
specific clinical contexts. Although TreatRAG uti-
lizes evidence-based medicine by referencing analogous
patient data, it lacks individualization.PACE-RAG
achieves personalized medicine by integrating a verifi-
cation process with patient-similarity retrieval.
Large Language Models (LLMs) possess exten-
sive medical knowledge and achieve expert-level
performance on standardized benchmarks (Singhal
et al., 2023a; Nori et al., 2023), they frequently
struggle with the nuanced, context-aware decision-
making required in complex clinical practice (Deik,
2023). While physicians derive clinical judgments
by synthesizing a patient’s unique clinical condi-
tion with their own accumulated clinical experi-
ence, LLMs struggle to capture this real-world ex-
periential knowledge. Therefore, it is essential to
develop methodologies that effectively incorporate
practical clinical insights.
To address this, Retrieval-Augmented Genera-
tion (RAG) frameworks utilize external knowledge
retrieved from clinical guidelines or medical text-
books (Zhao et al., 2025; Wang et al., 2024). How-
ever, these methods often suffer from semantic
fragmentation during document chunking and fail
to capture the intuitive judgment characteristic of
expert physicians. Specifically, as illustrated in
Figure 1,Guideline RAG, which retrieves standard-
ized clinical documents containing standardized
disease management guidelines, generates overlyarXiv:2603.17356v1  [cs.CL]  18 Mar 2026

Focus Query Extractor
Tendency Analysis 
 𝑦𝑠𝑢𝑚𝑚𝑎𝑟𝑦Explainable 
Clinical SummaryClinical Features {𝑓1,𝑓2,…,𝑓𝑛}
Search
Patient Database
𝐷={𝑑1,𝑑2,…,𝑑𝑁}Similar Patients
{𝑃1,𝑃2,…,𝑃𝑛}Retrieve
Treatment 
Pattern
𝑇={𝑇1,𝑇2,…𝑇𝑛} 𝑦𝑓𝑖𝑛𝑎𝑙Refined 
Prescription
LLM Initial Prescription 
𝑦0Historical RecordsCurrent Record
𝑥𝑡
𝐻={𝑥1,𝑥2,…,𝑥𝑡−1}Stage 1
Focus -Specific RetrievalStage 4 
Clinical 
SummaryStage 2 
Prescribing 
Tendency Analysis Stage 3 
Prescription 
Refinement
Figure 2:Overview of the PACE-RAG framework architecture.The pipeline consists of four stages: (1)
Focus-Specific Retrieval to identify similar patients from xt; (2) Prescribing Tendency Analysis to extract medica-
tion patterns from retrieved cases; (3) Prescribing Refinement to verify the initial prescription against identified
tendencies; and (4) Clinical Summary to synthesize the final treatment rationale.
generalized reasoning that fails to account for the
heterogeneous clinical trajectories of individual pa-
tients (Armstrong and Okun, 2020). To overcome
the rigidity of general guidelines, recent research
retrieved similar historical patient cases (Gao et al.,
2024; Liu et al., 2025; Chen et al., 2026). While
retrieving similar patient treatment cases (Similar
Patient RAG) such as TreatRAG (Liu et al., 2025)
offers more specific analysis than Guideline RAG,
it remains susceptible to ‘majority bias,’ in which
the model simply replicates the most frequent pre-
scription patterns observed across the retrieved co-
hort rather than identifying the clinical rationale
for each case.
To address these, we proposePACE-RAG
(Patient-AwareContextual andEvidence-based
Policy RAG), a novel framework designed to cap-
ture each patient’s context and leverage evidence-
based retrieval to model the implicit prescribing
policies of clinical experts. As shown in Fig. 2, our
framework employs a four-stage reasoning process
to identify the optimal treatment for each patient
based on their specific clinical presentation.
First, we employFocus-Specific Retrieval, which
retrieves patients using symptom-level keywords
rather than unstructured, full-text clinical notes.
Second, we implementPrescribing Tendency Anal-
ysis. The model extracts treatment patterns from
the retrieved cohort, effectively modeling how
physicians typically respond to specific symptom
clusters. Third, we introducePolicy-Driven Pre-
scription Refinement, which serves as a verification
mechanism that evaluates the initial medication
plan based on the insights derived from the preced-ing Tendency Analysis. Finally, theExplainable
Clinical Summarysynthesizes the decision trajec-
tory into an interpretable report to ensure clinical
trust and utility.
We evaluate our framework using a real-world
Parkinson’s Disease dataset. Our results exhibit
superior efficacy compared to alternative retrieval-
based methods and verification frameworks. To
demonstrate the generalizability, we extended our
evaluation to the large-scale MIMIC-IV bench-
mark (Johnson et al., 2023). This result proves
that our method successfully handles the complexi-
ties of multi-symptom clinical records. Crucially,
our framework built on a small open-source model
(e.g., 8B) surpasses the performance of large-scale
models such as Qwen2.5-72B. This demonstrates
that our framework can be achieved with a privacy-
preserving and cost-effective framework without
relying on large closed-source models.
In summary, our contributions are as follows:
•We proposePACE-RAG, which overcomes
the over-generalization of guideline-based re-
trieval and the majority-bias of similar-patient
retrieval by identifying symptom-specific ra-
tionales within individual patient contexts.
•We introduce a four-stage methodology that
extracts focus-specific clinical signals to per-
form keyword-centric retrieval of similar
cases and analyzes prescribing tendencies to
generate context-aware medication plans.
•We demonstrate that our framework gener-
alizes beyond Parkinson’s Disease to the

MIMIC-IV dataset, proving its robustness in
handling heterogeneous, multi-symptom clini-
cal contexts.
2 Related Work
2.1 LLMs in Clinical Decision Support
LLMs have demonstrated remarkable capabilities
in the medical domain, achieving expert-level per-
formance on benchmarks such as the USMLE and
MedQA (Singhal et al., 2023b; Nori et al., 2023).
Beyond these standardized benchmarks, recent re-
search has expanded the application of LLMs to
complex clinical workflows, including differential
diagnosis and treatment planning (Kanjee et al.,
2023; Thirunavukarasu et al., 2023; Chen et al.,
2023). However, despite their sophisticated med-
ical reasoning capabilities, LLMs struggle with
personalizedandcontext-dependentcare, partic-
ularly in neurodegenerative disorders like Parkin-
son’s Disease (PD). For instance, while LLMs can
accurately provide general knowledge regarding
the symptoms of Parkinson’s disease—specifically
movement disorders—they often fail to reflect the
granular, patient-specific context required for real-
world clinical prescriptions (Deik, 2023). In this
context, providing tailored prescriptions requires
more than the model’s inherent reasoning ability.
Our work overcomes this limitation by developing
a context-aware system that synthesizes an individ-
ual patient’s treatment trajectory with the prescrip-
tion patterns of similar patient cases.
2.2 Retrieval Framework in Clinical Settings
To mitigate domain-specific knowledge gaps, RAG
has been widely adopted in the medical do-
main (Gao et al., 2024; Xiong et al., 2024; Asai
et al., 2023; Zhao et al., 2025; Zakka et al., 2023;
Liu et al., 2025). Existing approaches primarily uti-
lize either explicit medical knowledge (Guideline-
based) or historical patient cases (Case-based).
Guideline-based approaches prioritize clinical
standardization by retrieving information from
medical textbooks or clinical practice guide-
lines (Zhao et al., 2025; Zakka et al., 2023). While
effective at providing verified medical facts, the
reliance on static knowledge-based retrieval can-
not fully capture the heterogeneous clinical pro-
files and diverse situational nuances of individual
patients in real-world practice. To overcome this
rigidity, evidence-based retrieval utilizes historical
EHR data to provide more tailored support. WhileTreatRAG (Liu et al., 2025) proved the efficacy
of retrieving diagnosis-prescription pairs, existing
models rely on global semantic similarity that often
overlooks specific, localized symptom signals.
Our approach introduces a novel framework that
extracts primary clinical symptoms to drive tar-
geted, keyword-based retrieval. By analyzing pre-
scribing tendencies from these retrieved cases, the
model refines final prescriptions to align with the
nuanced reasoning of expert clinicians.
2.3 Verification of Clinical Reasoning
While LLMs utilize RAG to minimize hallucina-
tions, they are prone to an over-reliance on re-
trieved contexts. Consequently, a verification stage
is essential to validate the model’s response and
discern only the most meaningful insights from
the retrieved content. To address this, existing ap-
proaches prioritize internal consistency or intro-
spection to refine model outputs (Manakul et al.,
2023; Gandhi et al., 2024; Li et al., 2025). Within
the clinical domain, AMIE (Tu et al., 2025) and
MedAgents (Tang et al., 2024) have been further
adapted to emulate the reasoning processes of med-
ical experts. For instance, MedReflect (Huang
et al., 2026) minimizes errors by generating and
training on question-answer pairs designed to val-
idate reflective reasoning paths, thereby refining
the model’s internal logic. However, these existing
verification frameworks struggle to incorporate the
unique clinical contexts of individual patients.
To address this, we introduce a Policy-dreiven
Prescription Refinement stage which evaluates the
initial prescription based on the prescribing tenden-
cies derived from similar patient cases to determine
its clinical appropriateness. Consequently, this pro-
cess prevents over-dependence on raw retrieved
content by validating the treatment plan against
strictly essential clinical information.
3 Methodology: PACE-RAG
We consider a historical sequence of patient records
H={x 1, x2, . . . , x t−1}alongside the current clin-
ical note xt. We restrict the number Hto a maxi-
mum of the three most recent clinical notes. Given
the historical sequence Hand the current clinical
notext, our objective is to generate a set of multiple
drugsY={y 1, y2, . . . , y n}(n≥1).
Initial prescription generation.Before the ex-
ecution of the multi-stage reasoning pipeline, the
LLM, denoted as ϕ, produces an initial candidate

prescription y0(prompt provided in Appendix D.1).
This prescription is generated directly from the pa-
tient’s longitudinal history Hand the current clini-
cal presentationx t, as follows:
y0=ϕ(H, x t)(1)
This initial output acts as an initial prior knowledge.
To refine this initial prior into a clinically precise
prescription, the clinical plan is generated through
a four-stage reasoning pipeline: Focus-Specific
Retrieval ( §3.1), Prescribing Tendency Analysis
(§3.2), Prescription Refinement ( §3.3), and Clinical
Summary ( §3.4). Figure 2 illustrates the overall
workflow.
3.1 Stage 1: Focus-Specific Retrieval
Focus query extractor.To retrieve patients with
similar symptoms, the Focus Query Extractor iso-
lates the most salient features from the current clini-
cal note xtby identifying ndistinct clinical queries.
This process is defined as:
{f1, f2, ..., f n}=ϕ(x t)(2)
where each firepresents a localized clinical fea-
ture within xt. To prevent hallucinations, each fi
is strictly constrained to literal substrings found
within xt, ensuring the extractor captures only
significant clinical indicators while filtering out
overly broad or non-specific terms (details in Ap-
pendix B.2 and D.2).
Retrieval similar patients.We perform a re-
trieval across a historical patient database D=
{d1, d2, . . . , d N}. For each focus query fi, we re-
trieve the top- kmost relevant patient cases among
those that exceed a similarity threshold τ. This
retrieval process is formally expressed as:
Pi=S(f i, D)(3)
where S(·) denotes a selection function that identi-
fies the nearest neighbors within the latent embed-
ding space (details in Appendix B.3). By perform-
ing this operation for all nfocus queries, we obtain
a comprehensive set of focus-specific patient clus-
ters,P={P 1, P2, . . . , P n}. This strategy allows
the model to retrieve historical precedents that are
highly relevant to the patient’s specific symptoms,
leading to more personalized and evidence-based
reasoning.3.2 Stage 2: Prescribing Tendency Analysis
Instead of analyzing the entire prescription his-
tory of retrieved patients, this stage conducts a
targeted prescribing tendency analysis to isolate
the direct impact of the specific symptom fi. This
approach prevents over-generalized recommenda-
tions of high-frequency drugs that lack of specific
clinical evidence to the target symptom. For each
query fi, the model analyzes the corresponding
retrieved cluster Pito extract Ti, a set of medica-
tions specific to the query along with their under-
lying rationales. To ensure high clinical fidelity
and prevent hallucinations, the tendency analyzer
ϕenforces three rigorous criteria:
•Isolation of focus-specific drugs:The model
extracts only medications directly relevant to
fi, filtering out general or routine drugs to
isolate acute interventions.
•Evidence-based rationale:Each added med-
ication must be supported by direct evidence
relating to the target symptomf i.
•Null on uncertainty:In the absence of ex-
plicit evidence attributing a drug to fi, the
outputT ireturns to an empty set (∅).
The total tendency set for the patient is defined
asT={T 1, T2, . . . , T n}, where each Tiis derived
as follows (prompt in appendix D.3):
Ti=ϕ(f i, Pi)(4)
By centering the analysis exclusively on action-
able interventions, this step precisely identifies
evidence-based treatment adjustments. These ex-
tracted addition signals ( T) subsequently serve as
a strict validation boundary in the next refinement
stage.
3.3 Stage 3: Policy-Driven Prescription
Refinement
If the initial draft ( y0) is directly incorporated with
the retrieved context without a dedicated verifica-
tion step, the model becomes overly dependent on
the raw retrieved data. This leads to a majority bias,
where the reasoning shifts toward the most frequent
patterns in the retrieved cohort. To address this, the
primary objective of Stage 3 is to enforce clinical
verification, ensuring that the final recommenda-
tion is precisely individualized to the unique needs
of the current patient. The verifier integrates the pa-
tient’s most recent historical records ( H) with the

analyzed prescribing tendencies ( T), the verifier en-
sures clinical appropriateness through a synthesis
functionϕ:
yfinal =ϕ(x t, H, T, y 0)(5)
To reach a final judgment, the verifier evaluates
y0against three policies, treating the preserva-
tion of the patient’s existing therapy (MAINTAIN)
as the baseline for all clinical decisions (see Ap-
pendix D.4 for detailed policy):
•Conservative Anchor (MAINTAIN):De-
faulting to preserving the patient’s active med-
ication historyH.
•Selective Expansion (ADD):Adding new
drugs only when the Tprovides a high-
confidence ADD signal from analogous cases.
•Discontinuation (REMOVE):Removing
drugs only when identifying definitive neg-
ative evidence, such as adverse effects or con-
traindications explicitly documented in the
patient’s current clinical presentation.
Consequently, yfinal represents the list of refined
medications tailored to the patient’s current clinical
trajectory.
3.4 Stage 4: Explainable Clinical Summary
In the final stage, the model generates an in-
terpretable summary of the verified prescription
(prompt provided in Appendix D.5). The summary
generation is formally defined as:
ysummary =ϕ(x t, H, T, y 0, yfinal )(6)
where the input includes the current visit xt, the pa-
tient’s longitudinal history H, the prescribing ten-
dencies T, the initial answer y0, and the finalized
prescription yfinal . The final output, ysummary , in-
tegrates the patient summary, extracted keywords,
and supporting clinical evidence alongside the final-
ized prescription. For example, the model details
the clinical rationale (e.g.,"Amantadine was main-
tained to address ‘weakness in hands’,aligning
with the retrieved tendency Tand the patient’s sta-
ble history H"), providing a transparent trace of
the verification process.
4 Experimental Setup
4.1 Datasets
We evaluate the efficacy and robustness of PACE-
RAG using two distinct clinical cohorts: a real-
world, unstructured Parkinson’s Disease dataset ,and the structured MIMIC-IV benchmark, which
serves to demonstrate the generalizability of our
framework. More details follow.
Real-world unstructured cohort (HPH).We
constructed a large-scale longitudinal dataset de-
rived from raw clinical notes at Inje University
Haeundae Paik Hospital (HPH), Korea. This co-
hort comprises clinical data from 947 patients di-
agnosed with Parkinson’s Disease between March
2010 and September 2023. We gathered SOAP
(Subjective, Objective, Assessment, Plan) notes
from both initial and follow-up visits using Elec-
tronic Medical Records (EMRs). To ensure patient
privacy, all personally identifiable information was
anonymized. Detailed dataset specifications are
provided in Appendix A.1.
Benchmark cohort (MIMIC-IV).To demon-
strate the generalizability of our framework,
we utilized the MIMIC-IV dataset (Johnson
et al., 2023), a standard benchmark for EMRs.
We utilized clinical data from a subset of
196,501 patients within the hosp module, specifi-
cally utilizing the admissions ,diagnoses_icd ,
d_icd_diagnoses , and prescriptions tables.
To standardize medication naming, we mapped
raw National Drug Codes (NDC) to their corre-
sponding Anatomical Therapeutic Chemical (ATC)
classification codes. Details of the MIMIC-IV data
processing are provided in Appendix A.2.
4.2 Retrieval Pool and Test Set Splitting
To prevent data leakage, we implemented a patient-
level splitting strategy across both cohorts. By
partitioning the data based on unique Patient IDs,
we strictly ensured that no overlap occurred be-
tween the patients in the retrieval pool and those
in the test set. For the HPH dataset, records were
divided into a retrieval pool and a test set following
a strict 80:20 ratio. For the MIMIC-IV dataset, we
sampled the test set to approximately 1,000 admis-
sions, with the remaining records constituting the
retrieval pool. This configuration ensures a fair
comparison and aligns our experimental setup with
baseline (Liu et al., 2025).
4.3 Baselines
To evaluate the efficacy of PACE-RAG, we con-
duct a comparative analysis against standard
RAG and verification-based baselines, utiliz-
ing Qwen3-8B (Team, 2025) and Llama3.1-8B-
Instruct (AI@Meta, 2024) as the backbone models.

Table 1:Evaluation on the real-world unstructured cohort (HPH).We evaluate PACE-RAG against baseline
retrieval and verification methodologies.Boldand underline denote the best and second-best performances. All
results are reported as the mean ±standard deviation across five random seeds. Statistical significance is computed
using a paired t-test (*p <0.05, **p <0.01).
MethodLlama3.1-8B-Instruct Qwen3-8B
F1 score (↑) Accuracy (↑) Precision (↑) Recall (↑) F1 score(↑) Accuracy (↑) Precision (↑) Recall (↑)
Zero-shot Baseline 49.96±0.42∗∗53.63±0.52∗∗47.02±0.53∗∗60.89±0.30∗∗75.57±0.63∗∗75.78±1.20∗∗80.95±0.19∗∗75.51±0.87∗∗
Guideline RAG 50.60±0.48∗∗54.23±0.65∗∗45.31±0.47∗∗65.64±0.61∗∗69.67±0.50∗∗67.03±0.60∗∗78.26±0.35∗∗68.28±0.54∗∗
TreatRAG 56.62±0.44∗∗57.94±0.46∗∗46.79±0.45∗∗86.22±0.62∗∗74.86±0.30∗∗76.29±0.29∗∗72.42±0.45∗∗84.69±0.15∗∗
MedReflect 62.67 ±1.46∗∗66.76±1.09∗∗61.04±0.50∗∗71.18±2.66∗∗73.11±0.20∗∗75.15±0.21∗∗71.62±0.14∗∗81.27±0.38
PACE-RAG 76.30±0.2678.44±0.1976.69±0.41 80.41 ±0.3380.84±0.1282.75±0.6883.78±0.32 81.90 ±1.22
Table 2:Generalization performance on the large-scale MIMIC-IV benchmark.To assess the broad applicability,
we evaluate PACE-RAG performance in the MIMIC-IV dataset.
MethodLlama3.1-8B-Instruct Qwen3-8B
F1 score (↑) Accuracy (↑) Precision (↑) Recall (↑) F1 score (↑) Accuracy (↑) Precision (↑) Recall (↑)
Zero-shot Baseline 24.73±0.03∗∗24.39±0.03∗∗49.56±0.12∗17.58±0.03∗∗34.20±0.13∗∗38.38±0.19∗∗47.52±0.09∗∗29.05±0.20∗∗
Guideline RAG 26.69±0.19∗∗27.36±0.24∗∗46.68±0.07∗∗20.39±0.20∗∗27.51±0.03∗∗28.48±0.07∗∗39.95±0.07∗∗22.58±0.05∗∗
TreatRAG 35.28 ±0.05∗∗34.33±0.55∗∗65.53±0.14∗∗26.07±0.05∗∗41.12±0.21∗∗38.93±0.32∗∗54.76±0.31∗∗36.42±0.46∗∗
MedReflect 28.87±0.40∗∗36.98±0.28∗∗37.09±0.69∗∗28.15±0.52∗∗37.65±0.17∗∗46.01±0.18∗∗42.19±0.20∗∗38.14±0.21∗∗
PACE-RAG 36.93±0.0742.71±0.15 49.17±0.2134.60±0.0647.22±0.1252.32±0.1757.03±0.0545.94±0.17
Table 3:Comparison on Parkinson with large-scale
models.Bold indicates the best performance.
Model F1 score Accuracy Precision Recall
Qwen2.5-14B 61.12±1.38∗∗63.18±1.79∗∗59.58±0.77∗∗69.23±1.91∗∗
Qwen2.5-32B 78.12±0.70∗∗79.94±0.83∗∗81.50±0.35∗∗79.66±0.87∗∗
Qwen2.5-72B 72.84±0.77∗∗74.61±0.96∗∗75.62±0.78∗∗75.62±0.78∗∗
Ours (Qwen8B)80.84±0.1282.75±0.6883.78±0.3281.90±1.22
Detailed implementation settings for all baselines
are provided in Appendix B.4.
Zero-shot Baseline:The base framework pro-
vided with the current clinical note xtand patient
historyHwithout any external information.
Guideline RAG:Retrieves relevant clinical
guidelines based on the current clinical note xt.
This information is integrated into the model’s
prompt alongside the patient’s record to support
informed decision-making.
TreatRAG (Liu et al., 2025):Retrieves histori-
cal cases from the retrieval pool Dthat are clini-
cally analogous to xtto leverage similar treatment
patterns for reasoning.
MedReflect (Huang et al., 2026):Constructs
a self-correction trajectory through reflective
question-and-answer pairs derived from the initial
response to guide the model toward a finalized clin-
ical decision.
5 Experimental Results
Real-world unstructured data results.Table 1
presents a comparative performance analysis on
the HPH dataset, which comprises real-world un-
structured clinical narratives. PACE-RAG con-Table 4:Comparison on MIMIC-IV with large-scale
models.Bold indicates the best performance.
Model F1 score Accuracy Precision Recall
Qwen2.5-14B 32.24±0.22∗∗35.80±0.32∗∗50.19±0.18∗∗25.75±0.23∗∗
Qwen2.5-32B 37.19±0.33∗∗43.81±0.52∗∗50.23±0.25∗∗33.02±0.47∗∗
Qwen2.5-72B 39.17±0.20∗∗46.82±0.30∗∗48.35±0.25∗∗36.76±0.24∗∗
Ours (Qwen8B)47.22±0.1252.32±0.1757.03±0.0545.94±0.17
sistently achieves the best overall performance
across most evaluation metrics on both backbone
models. In particular, when using Llama3.1-
8B-Instruct, PACE-RAG achieves the highest F1
score of 76.30% and accuracy of 78.44%, outper-
forming all baseline methods. While other RAG
or verification-based approaches improve perfor-
mance compared to the zero-shot baseline, they
often exhibit substantially higher recall, indicating
a tendency to over-predict positive cases. For in-
stance, while TreatRAG achieves the highest recall,
its relatively low precision suggests less reliable
predictions. Conversely, PACE-RAG maintains a
more robust trade-off between precision and re-
call, yielding stronger overall classification perfor-
mance. This trend remains consistent with Qwen3-
8B, where PACE-RAG again leads with an F1 score
of 80.84% and an accuracy of 82.75%.
Generalization to structured benchmarks.To
evaluate the generalizability of our framework
across diverse clinical environments, we extended
the application of PACE-RAG to the large-scale
MIMIC-IV benchmark (Table 2). PACE-RAG
consistently achieves superior performance across
nearly all evaluation metrics. For Llama3.1-8B-
Instruct, PACE-RAG achieves the highest F1 score
of 36.93% and accuracy of 42.71%, outperforming

Table 5:Performance compari-
son of different verification meth-
ods.Our method outperforms the no-
verification baseline and MedReflect.
Method F1 Acc
Stage 1 0.7665 0.7858
+ MedReflect 0.7268 0.7088
+ Ours (Stage 2&3) 0.8101 0.8322Table 6:Effect of different retrieval
sources in PACE-RAG.We com-
pare retrieval from clinical guide-
lines, similar patient cases, and Both.
RAG docs. F1 Acc
Guideline 0.6941 0.6681
Sim. Patients 0.8101 0.8322
Both 0.8009 0.8257Table 7:Evaluation on medical-
specialized language models.We
evaluate Baseline and PACE-RAG us-
ing MedGemma and HuatuoGPT.
Model Method F1 Acc
MedGemma Baseline 69.99 74.21
(4B)PACE-RAG 77.43 79.58
HuatuoGPT Baseline 43.62 45.10
(8B)PACE-RAG 73.88 74.98
all baseline methods. A similar pattern is observed
with Qwen3-8B. PACE-RAG achieves the best re-
sults across all metrics.
Comparison with large-scale models.Further-
more, we evaluated our framework against mod-
els with significantly larger parameter capacities
models (Table 3, 4). We conducted a compara-
tive analysis using local, open-source models, in-
cluding Qwen2.5-14B, 32B, and 72B. PACE-RAG
(Qwen3-8B) consistently outperforms these signifi-
cantly larger dense models, validating its superior
reasoning efficiency. These results demonstrate
that PACE-RAG enables efficient on-premise clini-
cal deployment, offering a cost-effective alternative
to large-scale models.
Specialist evaluation.To assess the reasoning ca-
pabilities of PACE-RAG, we performed a blinded
human evaluation with two Parkinson’s disease
specialists. Using the HPH dataset, we provided
model-generated responses to the experts to exam-
ine how closely each model’s reasoning process re-
sembles that of experienced clinicians. The special-
ists conducted a comparative evaluation of PACE-
RAG and the Guideline RAG across six criteria:
Clinical Accuracy, Safety, Faithfulness, Clinical
Reasoning, Clinical Usefulness, and Overall Pref-
erence. For each comparison, evaluators selected
the preferred response (Win) or indicated that both
responses were comparable (Tie). Detailed descrip-
Figure 3:Specialist evaluation results.Comparative
assessment of PACE-RAG and Guideline RAG by clini-
cal specialists across six evaluation criteria.tions of each evaluation criterion are provided in
Appendix B.5.
As illustrated in Figure 3, PACE-RAG achieved
a 70% Win/Tie rate in Overall Preference. Across
all other clinical dimensions, our framework’s
Win/Tie rates consistently exceeded 70% when
compared against Guideline RAG. These results
demonstrate that PACE-RAG’s performance is not
merely a statistical improvement but represents a
clinically valid approach. The specialists’ prefer-
ence indicates that the model produces clinically
sound rationales that align closely with the intuitive
reasoning of experienced practitioners.
6 Discussions
Are the extracted keywords representative of
patient symptoms?To evaluate the contribu-
tion of Stage 1, we examine whether the ex-
tracted keywords are clinically relevant by mea-
suring their alignment with the medications actu-
ally prescribed to patients. As shown in Figure 4,
GPT-4o is used as an external evaluator to score
the relevance between the extracted keywords and
the prescribed medications on the HPH dataset.
Figure 4:Relevance scores.The
distribution of relevance scores
between the extracted keywords
and the ground truth ( Mean =3.79,
SD=1.19).More than half
of the cases
receive high
relevance scores
(4–5), indicat-
ing that the
extracted focus
queries cap-
ture clinically
meaningful
signals related
to treatment
decisions. The detailed evaluation prompt is
provided in Appendix Figure 12.
Does focus-specific retrieval provide more pre-
cise clinical evidence?To assess retrieval quality,
we employed the Drug Precision@ kmetric (Fig-
ure 5), defined as the ratio of correctly matched

Figure 5:Comparison of retrieval quality via drug
precision@ k.While TreatRAG accumulates irrelevant
drugs as kincreases, PACE-RAG isolates symptom-
specific interventions to maintain higher precision.
medications to the total number of unique drugs
extracted from the top- kretrieved cases. As
TreatRAG incorporate the full prescription histo-
ries of similar patients, this approach introduces a
large number of routine or chronic medications that
are not relevant to the current clinical decision. As
a result, the retrieved evidence suffers from amajor-
ity bias, where common background medications
dominate the retrieval context, leading to a decline
in Drug Precision as kincreases (from 49.3% at
k= 1 to 33.2% at k= 7 ). In contrast, PACE-RAG
performs keyword-driven similar-patient retrieval
that focuses on acute clinical signals rather than the
patient’s entire medical history. This approach ef-
fectively minimizes the accumulation of irrelevant
chronic noise, maintaining a significantly higher
precision baseline (37.2% at k= 7 ). By providing
precise pharmacological evidence, PACE-RAG en-
ables the generator to produce more reliable clinical
decisions, ultimately outperforming TreatRAG.
Is tendency analysis and refinement more robust
than internal self-reflection?To assess our veri-
fication framework, we compared the Stage 1 base-
line against two strategies: the MedReflect method
and our proposed Stage 2&3 modules. Table 5 re-
veals that incorporating MedReflect performance
drop compared to the Stage 1. This degradation
suggests that the model becomes overly sensitive
to the context of its own generated question-answer
pairs. In contrast, the integration of our proposed
Stage 2&3 modules yields a substantial improve-
ment in both F1 score and accuracy. These results
demonstrate that the integration of Stages 2&3 is es-
sential for the verification of retrieved data, thereby
elevating the overall performance and reliability of
the generated prescriptions.Does combining clinical guidelines with patient
cases provide synergistic gains?We further in-
vestigate the impact of different retrieval sources
by incorporating clinical guidelines alongside simi-
lar patient cases. As detailed in Table 6, retrieving
similar patient cases alone yields the highest perfor-
mance, with an F1 score of 81.01% and an accuracy
of 83.22%. Interestingly, integrating both sources
does not provide additional gains. This suggests
that the inclusion of long clinical guidelines may
introduce unnecessary complexity or noise, poten-
tially distracting the model during reasoning, as
also noted in prior work (Chen et al., 2026).
Is the framework effective across specialized
medical architectures?Beyond general-
purpose LLMs, we evaluate our framework on
domain-specific models to verify its robustness in
specialized clinical settings. As shown in Table 7,
we compare PACE-RAG against the baseline us-
ing MedGemma (4B) (Sellergren et al., 2025) and
HuatuoGPT-o1 (8B) (Chen et al., 2024). When
tested on the HPH cohort, PACE-RAG yields sub-
stantial gains in both F1 score and accuracy across
both architectures. These results highlight that
PACE-RAG’s effectiveness is model-agnostic; its
ability to enhance clinical reasoning remains con-
sistent regardless of whether the underlying archi-
tecture is a general-purpose or a specialized lan-
guage model.
7 Conclusion
In this work, we proposed PACE-RAG, a four-
stage pipeline designed to produce patient-specific
drug recommendations by explicitly reasoning over
the key symptoms of each patient. Across a real-
world Parkinson’s disease cohort and the MIMIC-
IV benchmark, PACE-RAG consistently produced
more personalized and clinically evidence-based
recommendations than other RAG methods and
verification baselines, demonstrating both effec-
tiveness and generalizability. Moreover, blinded
qualitative evaluation by Parkinson’s specialists
indicated that our approach generates reasoning
that is more clinically meaningful. We anticipate
that this framework can be further expanded to
clinical systems that integrate multi-modal clinical
signals—such as laboratory results, imaging, and
structured risk factors—while ensuring robustness
across diverse institutional prescribing styles.

Limitations
We acknowledge several limitations in our study.
•Generalizability and Single-center Bias:
Although we validated PACE-RAG using
the MIMIC-IV benchmark and a real-world
Parkinson’s Disease dataset, the latter was
sourced from a single medical institution. Fu-
ture research should involve multi-center vali-
dation to ensure the framework’s robustness
across diverse clinical environments and pa-
tient demographics.
•Computational Latency and Inference
Cost:The proposed four-stage pipeline re-
quires multiple sequential LLM inferences to
reach a final prescription. While this modular
approach enhances performance, it increases
the total inference time and computational
overhead compared to zero-shot baseline mod-
els.
•Prompt Engineering and Human Effort:
Each stage of the PACE-RAG pipeline neces-
sitates task-specific prompts to ensure optimal
performance. This process currently relies on
manual engineering, which limit the frame-
work’s immediate scalability to new clinical
domains.
•Linguistic Nuance Loss in Translation:For
the HPH dataset, clinical records were trans-
lated from Korean to English for process-
ing. This translation step may have inadver-
tently miss subtle linguistic nuances inherent
in the source language, potentially affecting
the depth of diagnostic reasoning.
Ethics Statement
This retrospective study was approved by the In-
stitutional Review Board (IRB No. 2023-11-042),
which waived the requirement for informed consent
because the study used fully de-identified data. In
addition, this work utilizes the MIMIC-IV database
in accordance with its approved data-use agree-
ment. We only provide the code and scripts specifi-
cally utilized for the MIMIC-IV evaluation.
References
AI@Meta. 2024. Llama 3.1 model card.Melissa Armstrong and Michael Okun. 2020. Diagnosis
and treatment of parkinson disease: A review.JAMA,
323:548–560.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
Preprint, arXiv:2310.11511.
Junying Chen, Zhenyang Cai, Ke Ji, Xidong Wang,
Wanlong Liu, Rongsheng Wang, Jianye Hou, and
Benyou Wang. 2024. Huatuogpt-o1, towards
medical complex reasoning with llms.Preprint,
arXiv:2412.18925.
Shuheng Chen, Namratha Patil, Haonan Pan, Angel
Hsing-Chi Hwang, Yao Du, Ruishan Liu, and Jieyu
Zhao. 2026. Med-copilot: A medical assistant pow-
ered by graphrag and similar patient case retrieval.
Preprint, arXiv:2603.00460.
Tse Chen, Evan Multala, Patrick Kearns, Johnny De-
lashaw, Aaron Dumont, Demetrius Maraganore, and
Arthur Wang. 2023. Assessment of chatgpt’s per-
formance on neurology written board examination
questions.BMJ Neurology Open, 5:e000530.
Barbara S. Connolly and Anthony E. Lang. 2014. Phar-
macological treatment of parkinson disease: A re-
view.JAMA, 311(16):1670–1683.
Andres Deik. 2023. Potential benefits and perils of in-
corporating chatgpt to the movement disorders clinic.
Journal of Movement Disorders, 16:158–162.
Laura Evans, Andrew Rhodes, Waleed Alhazzani, Mas-
simo Antonelli, Craig M. Coopersmith, Craig French,
Flavia R. Machado, Lauralyn Mcintyre, Marlies Os-
termann, Hallie C. Prescott, and 1 others. 2021. Sur-
viving sepsis campaign: international guidelines for
management of sepsis and septic shock 2021.Inten-
sive Care Medicine, 47(11):1181–1247.
American Diabetes Association Professional Prac-
tice Committee for Diabetes*. 2025. Introduc-
tion and methodology: Standards of care in dia-
betes—2026.Diabetes Care, 49(Supplement_1):S1–
S5.
Susan H. Fox, Regina Katzenschlager, Shen-Yang Lim,
Brandon Barton, Rob M. A. de Bie, Klaus Seppi,
Miguel Coelho, Cristina Sampaio, and on behalf
of the Movement Disorder Society Evidence-Based
Medicine Committee. 2018. International parkin-
son and movement disorder society evidence-based
medicine review: Update on treatments for the motor
symptoms of parkinson’s disease.Movement Disor-
ders, 33(8):1248–1266.
Kanishk Gandhi, Denise Lee, Gabriel Grand, Muxin
Liu, Winson Cheng, Archit Sharma, and Noah D.
Goodman. 2024. Stream of search (sos): Learning to
search in language.Preprint, arXiv:2404.03683.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gener-
ation for large language models: A survey.Preprint,
arXiv:2312.10997.
Paul A. Heidenreich, Biykem Bozkurt, David Aguilar,
Larry A. Allen, Joni J. Byun, Monica M. Colvin,
Anita Deswal, Mark H. Drazner, Shannon M. Dun-
lay, Linda R. Evers, James C. Fang, Savitri E. Fed-
son, Gregg C. Fonarow, Salim S. Hayek, Adrian F.
Hernandez, Prateeti Khazanie, Michelle M. Kittle-
son, Christopher S. Lee, Mark S. Link, and 7 others.
2022. 2022 aha/acc/hfsa guideline for the manage-
ment of heart failure: A report of the american col-
lege of cardiology/american heart association joint
committee on clinical practice guidelines.Circula-
tion, 145(18):e895–e1032.
Yue Huang, Yanyuan Chen, Dexuan Xu, Chenzhuo
Zhao, Weihua Yue, and Yu Huang. 2026. Medreflect:
Teaching medical llms to self-improve via reflective
correction.Preprint, arXiv:2510.03687.
Alistair E. W. Johnson, Lucas Bulgarelli, Lu Shen, Alvin
Gayles, Ayad Shammout, Steven Horng, Tom J. Pol-
lard, Sicheng Hao, Benjamin Moody, Brian Gow,
Li-wei H. Lehman, Leo A. Celi, and Roger G. Mark.
2023. Mimic-iv, a freely accessible electronic health
record dataset.Scientific Data, 10(1):1.
Andre C. Kalil, Mark L. Metersky, Michael Klompas,
John Muscedere, Daniel A. Sweeney, Lucy B. Palmer,
Lena M. Napolitano, Naomi P. O’Grady, John G.
Bartlett, Jordi Carratalà, Ali A. El Solh, Santiago
Ewig, Paul D. Fey, Jr File, Thomas M., Marcos I.
Restrepo, Jason A. Roberts, Grant W. Waterer, Peggy
Cruse, Shandra L. Knight, and Jan L. Brozek. 2016.
Management of adults with hospital-acquired and
ventilator-associated pneumonia: 2016 clinical prac-
tice guidelines by the infectious diseases society of
america and the american thoracic society.Clinical
Infectious Diseases, 63(5):e61–e111.
Zahir Kanjee, Byron Crowe, and Adam Rodman. 2023.
Accuracy of a generative artificial intelligence model
in a complex diagnostic challenge.JAMA, 330(1):78–
80.
Jiaqi Li, Xinyi Dong, Yang Liu, Zhizhuo Yang, Quansen
Wang, Xiaobo Wang, SongChun Zhu, Zixia Jia, and
Zilong Zheng. 2025. Reflectevo: Improving meta
introspection of small llms by learning self-reflection.
Preprint, arXiv:2505.16475.
Chao-Chin Liu, Hao-Ren Yao, Der-Chen Chang, and
Ophir Frieder. 2025. Treatrag: A framework for
personalized treatment recommendation. InProceed-
ings of the Nineteenth ACM Conference on Recom-
mender Systems, RecSys ’25, page 690–695, New
York, NY , USA. Association for Computing Machin-
ery.
Potsawee Manakul, Adian Liusie, and Mark J. F. Gales.
2023. Selfcheckgpt: Zero-resource black-box hal-lucination detection for generative large language
models.Preprint, arXiv:2303.08896.
Joshua P. Metlay, Grant W. Waterer, Ann C. Long,
Antonio Anzueto, Jan Brozek, Kristina Crothers,
Laura A. Cooley, Nathan C. Dean, Michael J. Fine,
Scott A. Flanders, Marie R. Griffin, Mark L. Meter-
sky, Daniel M. Musher, Marcos I. Restrepo, on be-
half of the American Thoracic Society Whitney, Cyn-
thia G., and Infectious Diseases Society of Amer-
ica. 2019. Diagnosis and treatment of adults with
community-acquired pneumonia. an official clinical
practice guideline of the american thoracic society
and infectious diseases society of america.American
Journal of Respiratory and Critical Care Medicine,
200(7):e45–e67.
Harsha Nori, Nicholas King, Scott Mayer McKinney,
Dean Carignan, and Eric Horvitz. 2023. Capabilities
of gpt-4 on medical challenge problems.Preprint,
arXiv:2303.13375.
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert:
Sentence embeddings using siamese bert-networks.
InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing. Associa-
tion for Computational Linguistics.
Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri,
Atilla Kiraly, Madeleine Traverse, Timo Kohlberger,
Shawn Xu, Fayaz Jamil, Cían Hughes, Charles
Lau, Justin Chen, Fereshteh Mahvar, Liron Yatziv,
Tiffany Chen, Bram Sterling, Stefanie Anna Baby, Su-
sanna Maria Baby, Jeremy Lai, Samuel Schmidgall,
and 62 others. 2025. Medgemma technical report.
Preprint, arXiv:2507.05201.
Klaus Seppi, K. Ray Chaudhuri, Miguel Coelho,
Susan H. Fox, Regina Katzenschlager, Santiago
Perez Lloret, Daniel Weintraub, Cristina Sampaio,
, and the collaborators of the Parkinson’s Disease
Update on Non-Motor Symptoms Study Group on
behalf of the Movement Disorders Society Evidence-
Based Medicine Committee. 2019. Update on treat-
ments for nonmotor symptoms of parkinson’s dis-
ease—an evidence-based medicine review.Move-
ment Disorders, 34(2):180–198.
Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara Mah-
davi, Jason Wei, Hyung Won Chung, Nathan Scales,
Ajay Ajmani, Matthew Schucher, Zachary Welch,
and 1 others. 2023a. Large language models encode
clinical knowledge.Nature, 620(7972):172–180.
Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara Mah-
davi, Jason Wei, Hyung Won Chung, Nathan Scales,
Ajay Ajmani, Matthew Schucher, Zachary Welch,
and 1 others. 2023b. Large language models encode
clinical knowledge.Nature, 620(7972):172–180.
Scott M. Stevens, Scott C. Woller, Lisa Baumann
Kreuziger, Henri Bounameaux, Kevin Doerschug,
Geert-Jan Geersing, Menno V . Huisman, Clive
Kearon, Christopher S. King, Andrew J. Knighton,
Erica Lake, Susan Murin, Janine R.E. Vintch,

Philip S. Wells, and Lisa K. Moores. 2021. An-
tithrombotic therapy for vte disease: Second update
of the chest guideline and expert panel report.Chest,
160(6):e545–e608.
Xiangru Tang, Anni Zou, Zhuosheng Zhang, Ziming
Li, Yilun Zhao, Xingyao Zhang, Arman Cohan, and
Mark Gerstein. 2024. Medagents: Large language
models as collaborators for zero-shot medical reason-
ing.Preprint, arXiv:2311.10537.
Qwen Team. 2025. Qwen3 technical report.Preprint,
arXiv:2505.09388.
Arun James Thirunavukarasu, Darren Shu Jeng Ting,
Kabilan Elangovan, Laura Gutierrez, Ting Fang Tan,
and Daniel Shu Wei Ting. 2023. Large language
models in medicine.Nature Medicine, 29(8):1930–
1940.
Tao Tu, Mike Schaekermann, Anil Palepu, Khaled Saab,
Jan Freyberg, Ryutaro Tanno, Amy Wang, Brenna Li,
Mohamed Amin, Yong Cheng, Elahe Vedadi, Nenad
Tomasev, Shekoofeh Azizi, Karan Singhal, Le Hou,
Albert Webson, Kavita Kulkarni, S. Sara Mahdavi,
Christopher Semturs, and 7 others. 2025. Towards
conversational diagnostic artificial intelligence.Na-
ture, 642(8067):442–450.
Bhekisipho Twala. 2025. Ai-driven precision diagnosis
and treatment in parkinson’s disease: a comprehen-
sive review and experimental analysis.Frontiers in
Aging Neuroscience, V olume 17 - 2025.
Xidong Wang, Nuo Chen, Junyin Chen, Yidong Wang,
Guorui Zhen, Chunxian Zhang, Xiangbo Wu, Yan Hu,
Anningzhe Gao, Xiang Wan, Haizhou Li, and Benyou
Wang. 2024. Apollo: A lightweight multilingual
medical llm towards democratizing medical ai to 6b
people.Preprint, arXiv:2403.03640.
Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and
Aidong Zhang. 2024. Benchmarking retrieval-
augmented generation for medicine.Preprint,
arXiv:2402.13178.
Cyril Zakka, Akash Chaurasia, Rohan Shad, Alex R.
Dalal, Jennifer L. Kim, Michael Moor, Kevin Alexan-
der, Euan Ashley, Jack Boyd, Kathleen Boyd, Karen
Hirsch, Curt Langlotz, Joanna Nelson, and William
Hiesinger. 2023. Almanac: Retrieval-augmented
language models for clinical medicine.Preprint,
arXiv:2303.01229.
Xuejiao Zhao, Siyan Liu, Su-Yin Yang, and Chun-
yan Miao. 2025. Medrag: Enhancing retrieval-
augmented generation with knowledge graph-
elicited reasoning for healthcare copilot.Preprint,
arXiv:2502.04413.A Data Details
A.1 Real-World Unstructured Cohort (HPH)
Table 8:Real-world unstructured cohort statistics
(HPH).Patient counts and visit counts before and after
exclusion, with patient-level splitting into retrieval pool
and test sets.
Item #Patients #Visits
Original cohort (raw) 947 –
Excluded: w/o visit records 617 –
Total 330 4,216
Retrieval pool split (patient-level) 266 3,449
Test split (patient-level) 64 767
Data Statistics.From an initial pool of 947
Parkinson’s disease patients, we applied strict filter-
ing criteria to ensure longitudinal integrity, exclud-
ing individuals who lacked either initial or follow-
up visit records. As shown in Table 8, this process
resulted in a refined dataset of 330 patients and
4,216 total visits. Following an 80:20 patient-level
split, the final dataset comprises 266 patients in
the historical patient database (3,449 total visits for
retrieval) and 64 patients in the evaluation set (767
total visits for testing).
Translation.Since the original clinical records
were documented in Korean, we translated the en-
tire dataset into English using the Google Transla-
tor API.
Data Features.We utilized clinical notes from
both initial and follow-up visits. Although the orig-
inal records were unstructured, they contained ex-
plicit tags for the Subjective (S), Objective (O),
Assessment (A), and Plan (P) components. To
transform these into a structured format, we em-
ployed regular expressions to parse each segment,
ensuring a consistent SOAP framework across the
dataset. To maintain the model’s focus on clinical
narratives, all non-textual information outside the
SOAP structure was excluded.
For our experimental task, while each historical
record xi∈H contains the full SOAP components,
the current note xtspecifically excludes the ‘Plan’
(P) part. Our objective is to generate this missing
‘Plan’ by determining a set of multiple drugs Y=
{y1, y2, . . . , y n}(n≥1).
Number of Medications.The HPH dataset con-
tains 31 unique medications in the retrieval pool
and 24 in the test set. The detailed lists and counts

of these medications for each split are summarized
in Table 13.
A.2 Benchmark Cohort (MIMIC-IV)
Table 9:Benchmark cohort statistics (MIMIC).From
the full dataset, patients were randomly sampled at the
subject_id , and the selected subset was split into re-
trieval pool and test sets at the patient level.
Item #Patients #Admissions
Original cohort (raw) 364,627 546,028
Excluded: mapping error 168,126 83,295
Total 196,501 462,733
Retrieval pool split (patient-level) 196,110 461,639
Test split (patient-level) 391 1,094
Data Statistics.For the MIMIC-IV benchmark,
the dataset was partitioned by subject_id into a
comprehensive retrieval pool and a dedicated eval-
uation set. As detailed in Table 9, the retrieval
pool serves as a massive historical patient database,
comprising 196,110 unique patients and their corre-
sponding 461,639 hospital admissions. In contrast,
the evaluation set consists of 391 unique patients
across 1,094 admissions.
Data Features.For each patient, the input con-
sists of a historical sequence of admissions H=
{x1, x2, . . . , x t−1}and the diagnostic context of
the current admission xt. Each admission record
is identified by its hadm_id from the hosp mod-
ule. To provide explicit semantic context for the
model, raw ICD-9 and ICD-10 codes from the
diagnoses_icd table are mapped to their corre-
sponding descriptive text representations. Conse-
quently, each historical record xiconsists of a com-
prehensive list of diagnosis names associated with
that specific visit.
We denote the set of prescribed medications as
Y={y 1, y2, . . . , y n}(n≥1 ), retrieved from the
prescriptions table. To achieve clinical stan-
dardization, raw National Drug Codes (NDC) are
mapped to the Anatomical Therapeutic Chemical
(ATC) classification system via the RxNorm API.
During this process, prescriptions that failed to map
to a valid ATC code (e.g., non-pharmacological
noise such as normal saline or medical supplies)
were strictly excluded to ensure dataset quality.
Number of Medications.The MIMIC dataset
encompasses a substantially broader range of treat-
ments, comprising 406 unique medications in the
retrieval pool and 254 in the test set. Detailed lists
are summarized in Table 13.B Method Details
B.1 Hyperparameters
Table 10:Hyperparameters used in PACE-RAG ex-
periments.
Parameter LLaMA Qwen
Retrieval Settings
Retrieval top-k7 7
Thresholdτ0.9 0.9
Generation Settings
Context length (num_ctx) 8192 4096
Max token length (num_predict) 400 220
Temperature 0.8 0.6
Table 10 summarizes the key retrieval and gener-
ation hyperparameters used in our experiments. De-
tailed ablation studies regarding the retrieval config-
urations are further explored in Appendix C (Fig-
ure 6). We used certain 5 random seeds for the
experiment: 42, 137, 2025, 3141, 7777.
B.2 Details of Extraction Focus Query
We employ aFocus Query Extractorto extract
key symptoms from the clinical note xt. The spe-
cific prompt utilized for this extraction process is
provided in Appendix D.2. To ensure the high
fidelity and clinical relevance of the extracted in-
formation, the extractor adheres to two primary
operational constraints:
1.Exact substring constraint.The model is
restricted to extracting only literal, exact sub-
strings directly from the original clinical note.
2.Actionable clinical signal filtering.The ex-
tractor prioritizes actionable clinical signals.
If the patient is clinically stable, showing im-
provement, or if the note contains only con-
versational remarks without acute issues, the
model returns an empty list, ensuring that the
RAG pipeline is triggered only when signifi-
cant clinical indicators require a personalized
prescribing strategy.
Real-World Unstructured Cohort (HPH).For
the real-world hospital dataset, focus queries
{f1, f2, ..., f n}are extracted specifically from the
‘Subjective’ and ‘Assessment’ parts of the clinical
notes. To minimize LLM confusion and prevent
the generation of overly broad search terms that
degrade retrieval quality (e.g., extracting generic
terms like “Parkinson’s disease” or “worse”), we
constrain the extractor to output a maximum of two

highly specific, contextual symptom phrases per
patient.
Benchmark Cohort (MIMIC-IV).Unlike the
Parkinson’s disease dataset, we extract up to five
primary pathological conditions (e.g., “Acute kid-
ney failure”, “Sepsis”) directly from the parsed list
of diagnosis titles. To avoid ambiguity, the model is
strictly prohibited from omitting essential descrip-
tors (e.g., ‘acute’ or ‘chronic’) or outputting vague,
single-word terms like ‘failure’ alone.
B.3 Implementation of Patient Similarity
Search
We utilize a Vector Database (FAISS) to iden-
tify similar cases from the historical pool D.
Each record is selected based on its semantic
similarity to the clinical focus fi, where embed-
dings are generated using SentenceTransformer
(all-MiniLM-L6-v2 ) (Reimers and Gurevych,
2019). Retrieval is performed with a similarity
threshold τ= 0.9 and a maximum of k= 7
records per focus query.
Real-World Unstructured Cohort
(HPH).We retrieve a historical database
D={d 1, d2, . . . , d N}, where each record dj
represents the latest SOAP-formatted clinical note
for each patient. Specifically, queries fiderived
from the Subjective part are used to retrieve
Subjective part from the database, while those
derived from the Assessment part are mapped to
analogous Assessment part.
Benchmark Cohort (MIMIC-IV).In the
MIMIC-IV benchmark, the historical database
D={d 1, d2, . . . , d N}consists of records where
each entry djis represented the most recent
clinical status, formatted as a comma-separated list
of pathological conditions. For each focus query
fi, we retrieve the historical admissions from D
that exhibit the highest semantic similarity to the
query.
B.4 Baseline Implementation Details
Guideline RAG:For Guideline RAG, we con-
structed a retrieval pool using external clinical prac-
tice guidelines and medical textbooks. Detailed
hyperparameters for this process are documented
in Table 11. We retrieve relevant guideline sections
based on the patient’s records and use them as sup-
porting evidence. For the Parkinson dataset, the
retrieval query is constructed from the Subjective,Table 11:Hyperparameters used in Guideline RAG
experiments.
Parameter Value
Vector Database Settings
chunk size 1200
chunk overlap 200
Parkinson Guideline (Fox et al., 2018; Connolly and Lang,
2014; Seppi et al., 2019)
MIMIC Guideline (Stevens et al., 2021; Kalil et al., 2016;
Heidenreich et al., 2022; Metlay et al.,
2019; Evans et al., 2021; for Diabetes*,
2025)
Retrieval Settings
Retrieval top-k3
Thresholdτ0.3
Objective, and Assessment sections of the clinical
note, whereas for the MIMIC dataset the query is
derived from diagnosis-oriented visit text.
TreatRAG:Following the approach proposed
in TreatRAG (Liu et al., 2025), we implement
a diagnosis-centered retrieval pipeline. For the
Parkinson’s dataset, where clinical records are
structured as SOAP notes, we prioritize the ‘As-
sessment’ field for the query as it most closely
mirrors the clinical diagnosis. In instances where
the ‘Assessment’ data is unavailable, the ‘Subjec-
tive’ field is utilized as an alternative source for
the query. For the MIMIC-IV dataset, we utilize
thediagnoses field from the current visit as the re-
trieval query. To compute similarity, both the query
and the candidate cases are first converted to low-
ercase and tokenized by whitespace. These tokens
are then used to construct N-gram sets (specifically
bigrams, where N= 2 ). The similarity between
the query case ( A) and a candidate case ( B) is mea-
sured using the Jaccard similarity coefficient:
J(A, B) =|A∩B|
|A∪B|.
To mitigate noise from irrelevant matches, we apply
adaptive filtering to the similarity scores. We retain
only those cases with positive similarity that satisfy
the threshold:
J(A, B)≥µ+ 1.5σ,
where µandσrepresent the mean and standard de-
viation of the similarity score distribution, respec-
tively. The remaining candidates are then ranked by
their scores to identify the most relevant historical
cases for prescription generation.
MedReflect:While MedReflect was originally
designed as a training framework to construct re-
flective supervision trajectories, we adapt its core

logic into an inference-time refinement procedure.
Instead of generating data for fine-tuning, we im-
plement an inference-time refinement sequence
that generates reflective questions regarding poten-
tial clinical errors and generates answers to these
queries. These synthesized question-answer pairs
are then used to refine the initial draft into a final
prescription. Detailed prompt templates are pro-
vided in Appendix D.6.
B.5 Specialist Evaluation Protocol
To assess the clinical reasoning quality, we con-
ducted a blinded pairwise comparison with two
medical experts. For each case, experts evaluated
anonymized responses from Model A and Model
B based on the same patient record, selecting a pre-
ferred model or a tie. This evaluation comprised 10
question-answer pairs per comparison, with inde-
pendent judgments collected across the following
six clinical dimensions:
1.Clinical Accuracy:Based on the patient’s
clinical note, which response provides the
most appropriate medication recommendation
for the patient?
2.Safety:Which response is safer and less
likely to cause harm to the patient?
3.Faithfulness:Which response bases its rea-
soning on information actually present in the
clinical note, without introducing unsupported
facts?
4.Clinical Reasoning:Which response demon-
strates more appropriate clinical reasoning
when selecting medications?
5.Clinical Usefulness:Which response would
be more helpful for supporting your clinical
decision-making in practice?
6.Overall Preference:Overall, which response
would you prefer in clinical practice?
Table 14 provides an example of input clinical
note along with the reasoning outputs generated
by PACE-RAG and the Guideline RAG. To ensure
a fair comparison, we presented only the reason-
ing components (clinical evidence and the final
prescription), omitting the patient summary and
keywords from the evaluation materials.
Figure 6:Ablation of hyperparameters.Performance
remains stable across various top- kvalues (Left), while
increasing the similarity threshold τ(Right) consistently
enhances results.
C Ablation Study
Hyperparameter sensitivity.We evaluated the
impact of the retrieval count ( k) and the similarity
threshold ( τ). As shown in Figure 6, performance
remains notably stable across various kvalues, sug-
gesting that the framework is robust to the number
of retrieved cases. In contrast, increasing τconsis-
tently improves performance by filtering out clini-
cally irrelevant noise. Accordingly, we set τ= 0.9
to balance retrieval quality and reasoning depth.
Table 12:Ablation of retrieval strategies.Keyword
sparse retrieval vs. vector dense retrieval using HPH.
Method F1 score Acc Precision Recall
Sparse Retrieval 0.7904 0.8137 0.8166 0.8022
Dense Retrieval 0.8101 0.8322 0.8407 0.8154
Sparse vesus dense retrieval.We compared our
setting dense retrieval (vector embeddings) against
a sparse retrieval (BM25). As summarized in Ta-
ble 12, dense retrieval consistently outperforms
sparse retrieval across all metrics. These results
confirm that the semantic-aware embeddings pro-
vide more relevant and contextually rich patient
evidence than literal keyword matching.
D Prompt Templates
D.1 Initial Draft Generation & Zero-Shot
Baseline Prompts
The prompts in Figure 7 serve a dual purpose: they
act as the standalone ‘Zero-Shot Baseline’ and func-
tion as the ‘Initial Prescription Generator’ in PACE-
RAG.
D.2 Stage 1 : Focus Query Extractor
Figure 8 presents the Stage 1 prompts. The Focus
Query Extractor extracts actionable clinical key-
words (e.g., severe symptoms) while filtering out
noise. Crucially, Extractor return an empty list for

stable patients, ensuring dynamic RAG is triggered
only when medication adjustments are needed.
D.3 Stage 2: Prescribing Tendency Analysis
Figure 9 details the Stage 2 prompts. This
stage strictly extractstreatment tendencies—newly
added medications around similar patients for the
target focus. To isolate precise interventions, the
prompt applies a strict causality check and filters
out pre-existing medications. Furthermore, the
model is strictly instructed to return an empty out-
put unless a new medication is explicitly added for
the target symptom, effectively preventing halluci-
nations when clinical evidence is insufficient.
D.4 Stage 3: Policy-Driven Prescription
Refinement
Figure 10 outlines the Stage 3 prompts. This stage
uses a rigid two-step mechanical algorithm: Step
1 preserves the patient’s active history by forcing
all currently taken medications to be retained in
the final prescription. Step 2 evaluates newly pro-
posed medications, which represent medications
frequently introduced in similar patient cases re-
trieved from the database.
D.5 Stage 4: Explainable Clinical Summary
Figure 11 shows the Stage 4 prompt. The LLM
synthesizes the patient’s state, retrieved tendencies,
and the verification log into a summary report.
D.6 Others
Keyword Extract Evaluation.We utilized GPT-
4o in an LLM-as-a-judge capacity to assess the
Stage 1 Focus Query Extractor, setting the temper-
ature to 0.0 for consistency. As shown in Figure 12,
the judge model assigns a score from 1 to 5 based
on how accurately the extractions align with the
ground-truth prescription.
Baseline Methods.Figures 13 and 14 present
the prompts for the TreatRAG and GuidelineRAG
baselines, respectively. Additionally, Figure 15 de-
tails the prompt for MedReflect, a self-correction
baseline that employs an internal three-stage reflec-
tion pipeline consisting of Questioning, Answering,
and Refining.

Table 13:List and count of unique medications by dataset.
Dataset Split Medication Names Count
Parkinson Retrieval Pool Agomelatine, Amantadine sulfate, Aripiprazole, Benserazide HCl, Benztropine
mesylate, Bupropion HCl, Carbidopa, Carbidopa monohydrate, Clozapine,
Donepezil, Entacapone, Escitalopram, Ginkgo biloba ext, Levodopa, Meman-
tine HCl, Opicapone, Paroxetine, Pramipexole, Procyclidine HCl, Quetiapine,
Rasagiline, Risperidone, Rivastigmine, Ropinirole, Safinamide mesilate, Sertra-
line HCl, Tianeptine sodium, Trazodone HCl, Trihexyphenidyl HCl, V ortioxe-
tine, carbidopa31
Parkinson Test Set Amantadine sulfate, Aripiprazole, Benserazide HCl, Bupropion HCl, Carbidopa,
Carbidopa monohydrate, Donepezil, Entacapone, Escitalopram, Levodopa, Me-
mantine HCl, Olanzapine, Opicapone, Pramipexole, Procyclidine HCl, Queti-
apine, Rasagiline, Rivastigmine, Ropinirole, Safinamide mesilate, Tianeptine
sodium, Trazodone HCl, Trihexyphenidyl HCl, carbidopa24
MIMIC Retrieval Pool 3-oxoandrosten (4) derivatives, ACE inhibitors, plain, Acetic acid derivatives
and related substances, Acid preparations, Actinomycines, Adamantane deriva-
tives, Adrenergic and dopaminergic agents, Adrenergics in combination with
corticosteroids or other drugs, excl. anticholinergics, Agents for dermatitis,
excluding corticosteroids, Aldehydes and derivatives, Aldosterone antago-
nists, Alkyl sulfonates, Alpha and beta blocking agents, Alpha glucosidase
inhibitors, Alpha-adrenoreceptor antagonists, Aluminium compounds, Amides,
Amino acids, Amino acids and derivatives, Aminoalkyl ethers, Aminoquino-
lines, Aminosalicylic acid and derivatives, Aminosalicylic acid and similar
agents, Androstan derivatives, Angiotensin II receptor blockers (ARBs), other
combinations, Angiotensin II receptor blockers (ARBs), plain, Anilides, Anthra-
cyclines and related substances, Anti-androgens, Anti-estrogens, Antiallergic
agents, excl. corticosteroids, . . .406
MIMIC Test Set ACE inhibitors, plain, Acetic acid derivatives and related substances, Acid
preparations, Adrenergic and dopaminergic agents, Adrenergics in combination
with corticosteroids or other drugs, excl. anticholinergics, Agents for dermatitis,
excluding corticosteroids, Aldosterone antagonists, Alpha and beta blocking
agents, Alpha-adrenoreceptor antagonists, Aluminium compounds, Amino
acids, Amino acids and derivatives, Aminoquinolines, Aminosalicylic acid and
similar agents, Angiotensin II receptor blockers (ARBs), other combinations,
Angiotensin II receptor blockers (ARBs), plain, Anilides, Anthracyclines and
related substances, Anti-estrogens, Antiallergic agents, excl. corticosteroids,
Antiarrhythmics, class III, Antiarrhythmics, class Ib, Antiarrhythmics, class
Ic, Antibiotics, Anticholinergics, Anticholinesterases, Antidotes, Antihidrotics,
Antihistamines for topical use, Antiinfectives and antiseptics for local oral
treatment, Antiinfectives for treatment of acne, . . .254

Table 14:Example of a survey questionnaire item for clinical evaluation.Real-world patient cases were
presented to clinical experts along with blinded outputs from Guideline RAG and PACE-RAG.
[Input Patient Case]
Current Visit:
Subjective: External hypertension medication, Nomaloc and aspirin 100mg (external medication). My leg still feels numb.
Objective: None
Assessment: #1. Right MCA infarction #2. HTN #3. Parkinsonism #4. Thrombocytopenia
Past Visit History (Most recent 3 visits):
7th Visit)
Subjective: The right leg keeps pulling. The tremors are getting worse.
Objective: None
Assessment: Parkinson’s disease, atypical cognitive IPD differential diagnosis is not possible (MSA possibility)
Plan: Amantadine sulfate
8th Visit)
Subjective: has diarrhea. It’s hard to keep balance on legs.
Objective: tandem gait : intact Whole body pain is present.
Assessment: Parkinsonsim Atypical cognitive IPD cognitive differentiation is not possible (MSA possible)
Plan: Amantadine sulfate
9th Visit)
Subjective: when doing difficult tasks, hands shake a lot.
Objective: tandem gait; impairment of bradykinesia: equivocal
Assessment: Parkinson’s disease, atypical, IPD, cognitive differentiation is not possible (MSA is possible) - early stage H/O
right MCA territory infarction
Plan: Amantadine sulfate
Method Prescription Reasoning
Guideline RAG Amantadine sulfate * Patient summary *
The patient has a history of external hypertension medication, Nomaloc and
aspirin 100mg, and continues to experience numbness in the leg. They have a
history of right MCA infarction, hypertension, parkinsonism, and thrombocy-
topenia. The patient is currently stable with no acute changes in symptoms.
* Clinical Evidence *
Amantadine sulfate is standard for managing Parkinsonism symptoms, partic-
ularly in patients with concurrent stroke (Right MCA infarction). Evidence
supports its use in reducing bradykinesia and rigidity, though its efficacy for
leg numbness is less defined. Aspirin and Nifedipine are consistent with
secondary prevention for HTN and cerebrovascular risk.
* Prescribe *
Amantadine sulfate : Continued from patient’s history for Parkinsonism man-
agement. Clinical evidence supports its use in reducing Parkinsonian motor
symptoms, though its role in leg numbness remains unclear. No adjustment is
needed given stable symptoms and absence of worsening neurological deficits.
PACE-RAG (Ours)Amantadine sulfate * Patient summary *
The patient has a history of external hypertension medication, Nomaloc and
aspirin 100mg, and continues to experience numbness in the leg. They have
a known right MCA infarction, hypertension, parkinsonism, and thrombocy-
topenia. The patient is currently stable with no acute changes in symptoms or
neurological status.
* Key word *
Right MCA infarction, HTN, Parkinsonism, Thrombocytopenia, Leg numb-
ness
* Clinical Evidence *
In similar cases, Amantadine sulfate has been used for managing symptoms
of parkinsonism, particularly for dyskinesia and off-period symptoms. It is
also used in conjunction with other medications like Levodopa/Carbidopa
and Trihexyphenidyl for comprehensive symptom management. The drug is
generally well-tolerated, though long-term use may lead to dyskinesia.
* Prescribe *
Amantadine sulfate : Amantadine sulfate is being continued from the patient’s
history as it is effective for managing symptoms of parkinsonism, particu-
larly for dyskinesia and off-period symptoms. The decision to continue the
medication is based on the patient’s stable condition and the absence of acute
changes in symptoms. Patterns from similar cases indicate that Amantadine
sulfate is often used as an adjunct in managing parkinsonism, especially when
combined with other medications like Levodopa/Carbidopa. The recommen-
dation is to maintain the current regimen as the patient is stable and shows no
signs of deterioration.

Initial Prescription Generator / Zero-Shot Baseline: Parkinson’s Disease
System Prompt:
You are a Parkinson’s disease medication specialist.
Make your prescription based on the patient’s current symptoms and current medications. If there is active history, start from that active
history! Then, maintain (if there is no new symptoms or reason to modify drugs) or add or remove drugs based on the patient’s current symp-
toms. You should be careful - there should be certain reason for adding or removing drugs based on the patient’s current symptoms and current medications.
You should put one drug name at every line!
FORMAT STRICT:Output ONLY the [START]...[END] block. No extra text.
STABILITY RULE (no active history):
• If symptoms are stable/improved/no major worsening, output only conservative drug.
Output Format:
[START]
(Drug Name) | (short reason in 10 words or less)
[END]
User Prompt:
Clinical Note:Subjective: {subjective}, Objective: {objective}, Assessment: {assessment}
Most Recent Medications:{history}
Past Clinical Visits (up to 3 visits before current):{recent_visit_history_text}
Task:Generate prescription. Should put at least one drug in output.
Initial Prescription Generator / Zero-Shot Baseline: MIMIC-IV
System Prompt:
You are a clinical medication specialist for complex hospital inpatients (MIMIC-IV dataset).
Make your prescription based on the patient’s current diagnoses and most recent medications. If there is a most recent medications list, start from that list.
Then, maintain(if there is no new clinical reason to modify drugs) or add or remove drug classes based on the patient’s diagnoses and overall clinical
status. You should be careful - there should be certain reason for adding or removing drug classes based on the patient’s diagnoses and medication history.
Use pharmacological/therapeutic class names (not specific drug brands or single ingredients), and copy class names exactly as they appear in the input (do
not invert words or invent umbrella categories).
You should put one drug class name at every line!
FORMAT STRICT:Output ONLY the [START]...[END] block. No extra text.
STABILITY RULE (no most recent medications list):
• If the clinical picture is stable/improved/no major new problems, output only a conservative minimal set of classes.
• Prefer continuing existing chronic medication classes when they are clearly indicated.
• Avoid starting broad new classes unless there is clear diagnostic support.
OUTPUT:
[START]
(Drug Class Name) | (short reason in 10 words or less)
(Drug Class Name) | (short reason in 10 words or less)
[END]
User Prompt:
Patient Information:
- Diagnoses: {diagnoses}
- Most Recent Medications: {medications}
Past Clinical Visits (up to 3 visits before current):
{recent_visit_history_text}
Task:Generate prescription. Should put at least one drug in output.
Figure 7:Prompt designs for the Initial Prescription Generator, which also serve as the standalone Zero-Shot
Baseline (No RAG). The prompts impose strict output formats, ontology mapping constraints (for MIMIC-IV), and
conservative clinical heuristics.

Focus Query Extractor: Parkinson’s Disease
System Prompt:
You are a precise Medical Symptom Extractor. Your ONLY job is to identifyNEW, WORSENING, or UNRESOLVEDpathological symptoms that
require a medication change.
You are NOT a doctor. Do NOT invent, infer, or hallucinate.
STRICT EXTRACTION RULES:
1.THE ‘EMPTY’ RULE (CRITICAL):
• If the patient is STABLE, IMPROVING, or “feeling better”...
• If the text is just conversational (e.g., “test results explained”, “stopped drinking”)...
•→YOU MUST OUTPUT AN EMPTY LIST:{"keywords": []}
2.NO BROAD DIAGNOSES:NEVER output broad diseases like “Parkinson’s disease” or “Vascular Parkinsonism” alone. Broad keywords ruin search
databases. Only extract if attached to a specific severe symptom (e.g., “Parkinson’s disease with severe resting tremor”).
3.LITERAL MATCH:Use exact words from the text.
4.LIMIT:Max 2 phrases. Focus ONLY on acute issues.
Output format:
{"keywords": ["Target Symptom 1"]}OR{"keywords": []}
User Prompt:
Current Patient Input (Subjective/Objective/Assessment):
{text}
Active History:
{active_history}
Analyze the CURRENT text. If the patient is stable/improving/no acute symptoms, return {"keywords": []} . Otherwise, extract MAX 2 severe
symptom phrases. Return JSON only.
Focus Query Extractor: MIMIC-IV
System Prompt:
You are a precise Medical Diagnosis Extractor for MIMIC-IV inpatients. Your ONLY job is to identify CURRENT ACTIVE diagnoses that are NEW,
WORSENING, or UNRESOLVED and clearly matter for treatment.
You are NOT a doctor. Do NOT invent, infer, or hallucinate.
STRICT EXTRACTION RULES:
1.THE ‘EMPTY’ RULE (CRITICAL):
•If the diagnoses list is empty, purely administrative/meta, or only screening/history/status phrases (e.g., “Inpatient”, “Follow-up visit”, “History
of X”)...
• If nothing clearly reflects an acute or active condition needing management...
•→YOU MUST OUTPUT AN EMPTY LIST:{"keywords": []}
2.NO GENERIC LABELS OR FRAGMENTS:
• Do NOT output encounter/status/meta phrases alone (e.g., “Inpatient”, “Hospitalization”, “Follow-up visit”).
• Forbidden single bare words: “acute”, “chronic”, “failure”, “infection” by themselves.
3.LITERAL MATCH:
• Always keep full diagnosis phrases exactly as written (e.g., “Acute kidney failure”, “Sepsis due to pneumonia”).
• Do NOT chop off qualifiers like “acute”, “chronic”, “unspecified”.
• Every keyword must be an exact substring from the diagnoses text (case-insensitive). No paraphrasing or rewording.
4.LIMIT:Max 2 phrases. Focus ONLY on the most critical acute/active issues that drive treatment decisions.
OUTPUT FORMAT (JSON ONLY):
{"keywords": ["Diagnosis phrase 1", "Diagnosis phrase 2"]}OR{"keywords": []}
No explanations, no markdown, no extra keys.
User Prompt:
Diagnoses list:
{text}
Active History:
{active_history}
Analyze the CURRENT text. If the patient is stable/improving/no acute symptoms, return {"keywords": []} . Otherwise, extract MAX 2 severe
symptom phrases. Return JSON only.
Figure 8:Full prompts for Focus Query Extraction (Stage 1).The extractor is designed to isolate the most
critical, actionable clinical signals—such as severe symptoms and active pathologies—while filtering out broad
diagnoses and administrative noise. Furthermore, the model generate an empty list ( []) when patients are stable or
improving. This dual mechanism ensures dynamic RAG triggering, selectively engaging the retrieval process only
when therapeutic adjustments are genuinely required.

Prescribing Tendency Analysis: Parkinson’s Disease
System Prompt:
You are a highly focused Clinical Pattern Analyzer for Parkinson’s Disease.
OBJECTIVE:Analyze similar patient cases to identify EXACTLY which medications physiciansNEWLY PRESCRIBED (ADDED)to resolve the
specific target symptom.
STRICT ANALYSIS RULES:
1.FOCUS ONLY ON ADDITIONS:Your ONLY job is to find drugs that werenewly prescribed(added) in the similar cases to treat the specific Target
Clinical Focus.
2.CAUSALITY CHECK (CRITICAL):The drug MUST have been added specifically for the Target Symptom. If the target is “Falls”, do not extract
antidepressants given for other reasons.
3.IGNORE MAINTAINED DRUGS:Do not extract drugs that the patient was already taking.
4.CAUTIOUS EMPTY DEFAULT:If the cases do not explicitly show a new drug being added to treat the target symptom, your common_additions
MUST be empty[].
5.No Invention:Rely ONLY on the provided text.
OUTPUT FORMAT:Output ONLY valid JSON.
{"dominant_pattern": "ADD", "common_additions": ["Drug A"], "reasoning": "Brief 1-sentence reason based on cases."}
User Prompt:
Current Patient Symptoms:{symptoms}
==================================================
TARGET CLINICAL DIAGNOSIS:>>>{focus_txt}<<<
* IMPORTANT: Filter the ’Similar Patient Cases’ below. Only pay attention to how physicians treated THIS EXACT TARGET.
* Ignore medications given for other unrelated diagnoses in those cases.
==================================================
Similar Patient Cases:{rag_cases}
Analyze explicitly what was ADDED. Output JSON only.
Prescribing Tendency Analysis: MIMIC-IV
System Prompt:
You are a highly focused Clinical Pattern Analyzer for complex hospital inpatients (MIMIC-IV).
OBJECTIVE:Analyze similar patient cases to identify EXACTLY which medication CLASSES physiciansNEWLY PRESCRIBED (ADDED)to resolve
the specific target diagnosis.
STRICT ANALYSIS RULES:
1.FOCUS ONLY ON ADDITIONS:Your ONLY job is to find drug classes that werenewly prescribed(added) in the similar cases to treat the specific
Target Clinical Focus.
2.CAUSALITY CHECK (CRITICAL):The drug class MUST have been added specifically for the Target Diagnosis. (Note: General inpatient
care classes like ‘Heparin group’ or ‘laxatives’ that consistently accompany the diagnosis across many similar cases as standard co-management or
prophylaxis should also be extracted.)
3.IGNORE MAINTAINED DRUGS:Do not extract drug classes that the patient was already taking.
4.CAUTIOUS EMPTY DEFAULT:If the cases do not explicitly show a new drug class being added to treat the target diagnosis, your
common_additionsMUST be empty[].
5.No Invention:Rely ONLY on the provided text and copy exact pharmacological class names.
OUTPUT FORMAT:Output ONLY valid JSON.
{"dominant_pattern": "ADD", "common_additions": ["Drug Class A"], "reasoning": "Brief 1-sentence reason based on cases."}
User Prompt:
Current Patient Diagnoses:{diagnoses}
==================================================
TARGET CLINICAL DIAGNOSIS:>>>{focus_txt}<<<
* IMPORTANT: Filter the ’Similar Patient Cases’ below. Only pay attention to how physicians treated THIS EXACT TARGET.
* Ignore medications given for other unrelated diagnoses in those cases.
==================================================
Similar Patient Cases:{rag_patients}
Analyze explicitly what was ADDED. Output JSON only.
Figure 9:Full prompts for Prescribing Tendency Analysis (Stage 2).By enforcing a strictCausality Check,
this stage deliberately filters out high-frequency chronic medications. Furthermore, theCautious Empty Default
prevents hallucination by requiring concrete evidence of a new intervention to populate the common_additions list,
effectively neutralizing majority bias.

Policy-Driven Prescription Refinement Prompt: Parkinson’s Disease
System Prompt:
You are a strict Clinical Auditor checking a Draft Prescription against RAG Evidence.
You MUST execute your task mechanically using this exact 2-step algorithm. Do not overthink.
=====================================================================
### DRUG FILTERING ALGORITHM
STEP 1: PRESERVE ACTIVE HISTORY (MANDATORY)
Look at theActive Historylist provided in the prompt.
• IfActive Historyis empty or “None”, you MUST NOT use the “KEPT” action. Skip to Step 2.
• You MUST put EVERY single drug fromActive Historyinto yourfinal_prescriptionarray.
• You MUST create an{"action": "KEPT", "drug": "..."}entry in theaudit_logfor EACH of these drugs.
• NEVER miss or drop a drug from the Active History.
STEP 2: EV ALUATE DRAFT DRUGS
Look at theInitial Draft Prescription. For each drug that is NOT already in Active History:
• Is it exactly listed in the RAGcommon_additionsarray?
•→YES: ADD it. (action: “ADDED”)
•→NO: REMOVE it. (action: “REMOVED”)
=====================================================================
### CRITICAL RULES
•STRICT ADD GATE:You can ONLY add a new drug if it is explicitly listed incommon_additions.
•SYNC RULE:If you log “REMOVED”, it MUST NOT be infinal_prescription.
•EMPTY FALLBACK:Iffinal_prescriptionis completely empty, pick drug from RAG’scommon_additionsand ADD it.
### REQUIRED JSON FORMAT (DO NOT ADD OTHER KEYS)
{"final_prescription": ["Drug A", "Drug B"], "audit_log": [...], "final_description": "..."}
User Prompt:
Patient Input:Subjective: {subjective}, Objective: {objective}, Assessment: {assessment}
Active History (Currently taking. ALWAYS KEEP THESE):{active_history}
Initial Draft Prescription (WARNING: This is just a guess. Default action is REMOVE unless proven by RAG):{initial_prescription}
{primary_focus_block}
RAG Focus Tendencies (what the DB returned for those/similar symptoms; ordered by priority):{rag_focus_tendency}
Past Clinical Visits:{recent_visit_history_text}
FINAL EXECUTION TASK:Apply the 2-STEP DRUG FILTERING ALGORITHM. 1. Force-copy ALL Active History classes to ‘final_prescription‘
and log them as "KEPT" (unless history is empty). 2. Evaluate remaining Draft classes strictly against RAG ‘common_additions‘, adopting the specific
RAG terminology.
Policy-Driven Prescription Refinement prompt: MIMIC-IV
System Prompt:
You are a strict Clinical Auditor checking a Draft Prescription against RAG Evidence for complex hospital inpatients (MIMIC-IV).
You MUST execute your task mechanically using this exact 2-step algorithm. Do not overthink.
=====================================================================
### DRUG CLASS FILTERING ALGORITHM
STEP 1: PRESERVE ACTIVE HISTORY (MANDATORY)
Look at theActive Historylist provided in the prompt.
• IfActive Historyis empty or “None”, you MUST NOT use the “KEPT” action. Skip to Step 2.
• Otherwise, you MUST put EVERY single drug class fromActive Historyinto yourfinal_prescriptionarray.
• You MUST create an{"action": "KEPT", "drug": "..."}entry in theaudit_logfor EACH of these classes.
• NEVER miss or drop a class from the Active History.
STEP 2: EV ALUATE DRAFT DRUG CLASSES
Look at theInitial Draft Prescription. For each drug class that is NOT already in Active History:
• Is this exact class name present in ANY focus’s RAGcommon_additionsarray?
•→YES: ADD it tofinal_prescription. (action: “ADDED”)
•→NO: REMOVE it from the Draft. (action: “REMOVED”)
For EVERY Draft drug class you process in Step 2, you MUST create exactly one audit_log entry (“ADDED” or “REMOVED”). If the Initial Draft is
non-empty,audit_logMUST NOT be empty.
=====================================================================
### CRITICAL RULES
•STRICT ONTOLOGY:NEVER output broad/umbrella categories (e.g., “Antihypertensives”, “Analgesics”, “Anticoagulants”). ALWAYS keep the
exact ATC class names that appear in Active History or RAG (e.g., “Heparin group”, “Anilides”).
•STRICT ADD GATE:You can ONLY add a new drug class if it isliterally present(exact string match) in RAG common_additions for at least one
focus.
•SYNC RULE:If you log “REMOVED” for a class, that class MUST NOT appear anywhere in final_prescription . If a class remains in
final_prescription, it MUST have action “KEPT” or “ADDED” inaudit_log.
•COVERAGE RULE:When the Initial Draft is non-empty, every Draft class that is not in Active History MUST appear exactly once in audit_log as
either “ADDED” or “REMOVED”.
•EMPTY FALLBACK:If Active History is empty AND you removed all Draft classes AND at least one RAG common_additions class exists, you
may ADD some clearly relevant class from RAG.
### REQUIRED JSON FORMAT (DO NOT ADD OTHER KEYS)
{"final_prescription": ["Drug A", "Drug B"], "audit_log": [...], "final_description": "..."}
User Prompt:
Patient Input (Diagnoses):{diagnoses}
Active History (Currently taking. ALWAYS KEEP THESE):{active_history}
Initial Draft Prescription (WARNING: This is just a guess. Default action is REMOVE unless proven by RAG):{initial_prescription}
RAG Focus Tendencies (what the DB returned for those/similar symptoms; ordered by priority):{rag_focus_tendency}
Past Clinical Visits:{recent_visit_history_text}
FINAL EXECUTION TASK:Apply the 2-STEP DRUG CLASS FILTERING ALGORITHM. 1. Force-copy ALL Active History classes to ‘fi-
nal_prescription‘ and log them as "KEPT" (unless history is empty). 2. Evaluate remaining Draft classes strictly against RAG ‘common_additions‘,
adopting the specific RAG terminology.
Figure 10:Full prompts for Policy-Driven Prescription Refinement (Stage 3).Step 1 prioritizes treatment
stability by preserving existing history, while Step 2 adds new medications, which represent in similar patient cases.

Clinical Summary Generator Prompt
System Prompt:
You are a Senior Clinical Consultant summarizing cases for attending physicians.
Format your output EXACTLY as follows:
* Patient summary *
(Detailed description of patient’s current symptoms, stability, and clinical trajectory in 2-3 sentences.)
* Key word *
(Comma-separated list of clinical focus keywords extracted from symptoms/diagnoses.)
* Clinical Evidence *
(Concise summary of treatment patterns found in similar historical cases. Mention standard approaches like maintaining or adjusting medications and
specific evidence found for the current clinical focus.)
* Prescribe *
(Drug Name) : (Detailed clinical rationale. Explain if the drug is being continued from the patient’s history, why it was added or adjusted based on current
symptoms, and how patterns from similar cases or clinical validation influenced the final recommendation.)
Rules:
• Use ONLY provided inputs. No hallucinations.
• Ensure the * Prescribe * section covers EVERY drug in the final recommendation list.
• No internal monologue or<think>tags.
• Output plain text structure only.
User Prompt:
**Patient State:** {patient_state}
**Initial Draft (Proposed Plan):** {initial_prescription}
**Clinical Evidence from Similar Cases:** {rag_tendency_by_focus}
**Clinical Validation Log:** {audit_log}
**Final Recommended Medications:** {final_answer_list}
Figure 11:Full prompt for Clinical Summary Generation (Stage 4).This stage integrates diverse evi-
dence—including patient history, prescribing tendency analysis, and refined tendencies—into a cohesive clinical
narrative. As a result, this stage provides clinicians with both the final drug recommendations and the clinical
reasoning behind them.

Evaluator Prompt for Keyword Extractor
System Prompt:
You are a clinical evaluator for a Parkinson’s disease prescription support system.
What the keyword extractor does (the system you are evaluating):
•It is a Medical Symptom Extractor. It identifiesNEW, WORSENING, or UNRESOLVEDpathological symptoms thatrequire a medication change.
•It must outputempty keywordswhen: the patient is STABLE, IMPROVING, or “feeling better”; or when the text is just conversational (e.g. “test
results explained”, “stopped drinking”) with no acute symptom. So “no keywords” is correct when the case does not need RAG (no acute symptom to
focus on).
•When it does extract, it usesliteral phrases from the note, at most 2 phrases, focused onacute/severe symptoms— not broad diagnoses alone (e.g.
not “Parkinson’s disease” by itself).
You will be given:
1. Current clinical note ({patient_input}) and optional previous visit history ({history_soap}, max 3 visits).
2. Ground-truth prescription list (ingredient names) for this visit.
3. “Focus areas”: the keywords our extractor output from Subjective/Assessment,ora notice that it output no keywords.
Your task (two cases):
•When keywords were extracted:Evaluate how well the prescription aligns with those focus keywords (do the drugs address the extracted symptoms?).
Score 1-5.
•When no keywords were extracted:Judge whether that was correct under the extractor’s rules — i.e. was this case actually sta-
ble/improving/conversational with no acute symptom (then 5 = correct not to extract), or was there a new/worsening/unresolved symptom that
should have been extracted (then 1-2 = wrong not to extract)? Score 1-5.
Output in this format (exact headers):
Relevance (1-5):[Integer 1-5]
Explanation:[2-4 sentences]
Summary:[One sentence]
Do not prescribe or suggest drugs; only evaluate.
User Prompt:
Current Clinical Note (patient_input):
{patient_input}
Previous Visit History (SOAP):
{history_soap}
Ground-Truth Prescription (ingredient list):
{gt_txt}
Focus areas (keywords our system extracted, or notice that none were extracted):
{focus_desc}
The extractor outputs keywords only for NEW/WORSENING/UNRESOLVED symptoms that need a medication change; it must output none when the case
is stable/improving or conversational with no acute symptom. For this case: if keywords were extracted, evaluate prescription vs those keywords; if none
were extracted, judge whether that was correct (stable/no acute = correct, clear acute symptom missed = wrong). OutputRelevance (1-5),Explanation,
andSummaryas specified.
Figure 12:Prompt template for the automated clinical evaluator (LLM-as-a-judge) used to assess the Focus
Query Extractor’s performance.The evaluator is instructed to verify not only the clinical relevance of the
extracted keywords against the ground-truth prescriptions, but more importantly, to strictly penalize the model if it
fails to apply the ‘Empty Rule’ for stable or conversational cases.

Baseline Similar Patients RAG (TreatRAG): Parkinson’s Disease
System Prompt:
You are a Parkinson’s disease medication specialist.
Make your prescription based on the patient’s current symptoms and current medications. If there is active history, start from that active history! Then,
maintain(if there is no new symptoms or reason to modify drugs) or add or remove drugs based on the patient’s current symptoms and current medica-
tions. You should be careful - there should be certain reason for adding or removing drugs based on the patient’s current symptoms and current medications.
When similar cases are provided, use them only as a secondary reference. Always prioritize SUBJECTIVE/OBJECTIVE/ASSESSMENT and active history.
You should put one drug name at every line!
FORMAT STRICT:Output ONLY the [START]...[END] block. No extra text.
OUTPUT:
[START]
(Drug Name) | (short reason in 10 words or less)
(Drug Name) | (short reason in 10 words or less)
[END]
User Prompt:
Clinical Note:
- Subjective: {subjective}
- Objective: {objective}
- Assessment: {assessment}
History Summary:
{history}
Past Clinical Visits (up to 3 visits before current):
{recent_visit_history_text}
Similar Cases (top retrieved):
{similar_cases}
Task:Generate prescription. Should put at least one drug in output.
Baseline Similar Patients RAG (TreatRAG): MIMIC-IV
System Prompt:
You are a clinical medication specialist for complex hospital inpatients (MIMIC-IV dataset).
Make your prescription based on the patient’s current diagnoses and most recent medications. If there is a most recent medications list, start from that list.
Then, maintain(if there is no new clinical reason to modify drugs) or add or remove drug classes based on the patient’s diagnoses and overall clinical
status. You should be careful - there should be certain reason for adding or removing drug classes based on the patient’s diagnoses and medication history.
When similar cases are provided, use them only as a secondary reference. Always prioritize CURRENT DIAGNOSES and Most Recent Medications.
Use pharmacological/therapeutic class names (not specific drug brands or single ingredients), and copy class names exactly as they appear in the input (do
not invert words or invent umbrella categories).
You should put one drug class name at every line!
FORMAT STRICT:Output ONLY the [START]...[END] block. No extra text.
STABILITY RULE (no most recent medications list):
• If the clinical picture is stable/improved/no major new problems, output ONLY a conservative minimal set of classes.
• Prefer continuing existing chronic medication classes when they are clearly indicated.
• Avoid starting broad new classes unless there is clear diagnostic support.
OUTPUT:
[START]
(Drug Class Name) | (short reason in 10 words or less)
(Drug Class Name) | (short reason in 10 words or less)
[END]
User Prompt:
Patient Information:
- Diagnoses: {diagnoses}
- Most Recent Medications: {medications}
Past Clinical Visits (up to 3 visits before current):
{recent_visit_history_text}
Similar Cases (top retrieved):
{similar_cases}
Task:Generate prescription. Should put at least one drug class in output.
Figure 13:Prompt designs for the Baseline Similar Patients RAG (TreatRAG).In this method, retrieved similar
historical cases are directly appended to the prompt context.

Baseline Guideline RAG: Parkinson’s Disease
System Prompt:
You are a Parkinson’s disease medication specialist.
Make your prescription based on the patient’s current symptoms and current medications. If there is active history, start from that active
history! Then, maintain(if there is no new symptoms or reason to modify drugs) or add or remove drugs based on the patient’s current symp-
toms. You should be careful - there should be certain reason for adding or removing drugs based on the patient’s current symptoms and current medications.
GUIDELINE RULE:
• If ‘Similar Guidelines’ are provided, apply them if they match the patient’s specific symptoms.
• If none are relevant or empty, completely ignore them and rely on your clinical judgment.
STABILITY RULE (no active history):
• If symptoms are stable/improved/no major worsening, output ONLY one conservative drug.
• Prefer a non-escalating option from guidelines when available.
• Avoid starting Levodopa/Carbidopa unless there is clear worsening/functional decline.
You should put one drug name at every line!
FORMAT STRICT:Output ONLY the [START]...[END] block. No extra text.
OUTPUT:
[START]
(Drug Name) | (short reason in 10 words or less)
(Drug Name) | (short reason in 10 words or less)
[END]
User Prompt:
Clinical Note:
- Subjective: {subjective}
- Objective: {objective}
- Assessment: {assessment}
Most Recent Medications:{history}
Past Clinical Visits (up to 3 visits before current):
{recent_visit_history_text}
Similar Guidelines:
{similar_guidelines}
Task:Generate prescription. Should put at least one drug in output.
Baseline Guideline RAG: MIMIC-IV
System Prompt:
You are a clinical medication specialist for complex hospital inpatients (MIMIC-IV dataset).
Make your prescription based on the patient’s current diagnoses and most recent medications. If there is a most recent medications list, start from that list.
Then, maintain(if there is no new clinical reason to modify drugs) or add or remove drug classes based on the patient’s diagnoses and overall clinical
status. You should be careful - there should be certain reason for adding or removing drug classes based on the patient’s diagnoses and medication history.
GUIDELINE RULE:
• If ‘Similar Guidelines’ are provided, apply them if they match the patient’s specific diagnoses.
• If none are relevant or empty, completely ignore them and rely on your clinical judgment.
Use pharmacological/therapeutic class names (not specific brands/ingredients), and copy class names literally when possible.
You should put one drug class name at every line!
FORMAT STRICT:Output ONLY the [START]...[END] block. No extra text.
OUTPUT:
[START]
(Drug Class Name) | (short reason in 10 words or less)
(Drug Class Name) | (short reason in 10 words or less)
[END]
User Prompt:
Patient Information:
- Diagnoses: {diagnoses}
- Most Recent Medications: {medications}
Past Clinical Visits (up to 3 visits before current):
{recent_visit_history_text}
Similar Guidelines:
{similar_guidelines}
Task:Generate prescription. Should put at least one drug class in output.
Figure 14:Prompt designs for the Baseline Guideline RAG.This represents a traditional retrieval-augmented
generation setup where retrieved excerpts from general clinical guidelines or literature are injected directly into the
prompt context.

MedReflect Stage 1: Question Generation
System Prompt:
You are a professional doctor.
User Prompt:
Here is a medical query from your patient:
<Query>:
[Subjective]:{subjective}
[Objective]:{objective}
[Assessment]:{assessment}
[Active History]:{active_history}
</Query>
Here is the patient’s previous prescription (max 3):
{previous_prescription}
Here is your response for the medical query:
<Response>:{initial_prescription}</Response>
The initial answer in your response might be incorrect or incomplete (e.g., missing active history drugs or adding unsupported drugs), so you need to ask a
reflective question based on your query and your response.
Now please provide a brief question(Strictly follow this format:<Reflective Question>your response</Reflective Question>):
MedReflect Stage 2: Answer Generation
System Prompt:
You are a professional doctor.
User Prompt:
Here is a medical query from your patient:
<Query>:
[Subjective]:{subjective}
[Objective]:{objective}
[Assessment]:{assessment}
[Active History]:{active_history}
</Query>
Here is the patient’s previous prescription (max 3):
{previous_prescription}
Here is your response for the medical query:
<Response>:{initial_prescription}</Response>
Here is your own reflective question for your response:
<Reflective Question>:{reflective_question}</Reflective Question>
Now, please provide a concise answer for the reflective question(Strictly follow this format: <Reflective Answer> your response </Reflective Answer> ):
MedReflect Stage 3: Refinement
System Prompt:
You are a professional doctor.
User Prompt:
Here is a medical query from your patient:
<Query>:
[Subjective]:{subjective}
[Objective]:{objective}
[Assessment]:{assessment}
[Active History]:{active_history}
</Query>
Here is the patient’s previous prescription (max 3):
{previous_prescription}
Here is your own reflection on your initial answer:
<Self-Reflection>:{reflective_question} {reflective_answer}</Self-Reflection>
Here is the response you need to complete(Complete each blank):
<Response>:The final prescribed medications are:<mask, type: drug_list></Response>
Now, according on this reflection, your completed answer is(Strictly follow this format:<Answer>:your refine entity,eg:[Drug A, Drug B]</Answer>):
Figure 15:Prompt designs for the MedReflect baseline.The self-reflection process is divided into three sequential
stages: (1)Question Generation, where the model critiques its initial answer; (2)Answer Generation, where
it resolves the critique; and (3)Refinement, where it outputs the final corrected prescription based on its internal
reflection.
# An Iterative LLM Framework for SIBT utilizing RAG-based Adaptive Weight Optimization

**Authors**: Zhuo Xiao, Qinglong Yao, Jingjing Wang, Fugen Zhou, Bo Liu, Haitao Sun, Zhe Ji, Yuliang Jiang, Junjie Wang, Qiuwen Wu

**Published**: 2025-09-10 08:54:16

**PDF URL**: [http://arxiv.org/pdf/2509.08407v1](http://arxiv.org/pdf/2509.08407v1)

## Abstract
Seed implant brachytherapy (SIBT) is an effective cancer treatment modality;
however, clinical planning often relies on manual adjustment of objective
function weights, leading to inefficiencies and suboptimal results. This study
proposes an adaptive weight optimization framework for SIBT planning, driven by
large language models (LLMs). A locally deployed DeepSeek-R1 LLM is integrated
with an automatic planning algorithm in an iterative loop. Starting with fixed
weights, the LLM evaluates plan quality and recommends new weights in the next
iteration. This process continues until convergence criteria are met, after
which the LLM conducts a comprehensive evaluation to identify the optimal plan.
A clinical knowledge base, constructed and queried via retrieval-augmented
generation (RAG), enhances the model's domain-specific reasoning. The proposed
method was validated on 23 patient cases, showing that the LLM-assisted
approach produces plans that are comparable to or exceeding clinically approved
and fixed-weight plans, in terms of dose homogeneity for the clinical target
volume (CTV) and sparing of organs at risk (OARs). The study demonstrates the
potential use of LLMs in SIBT planning automation.

## Full Text


<!-- PDF content starts -->

ANITERATIVELLM FRAMEWORK FORSIBTUTILIZING
RAG-BASEDADAPTIVEWEIGHTOPTIMIZATION
Zhuo Xiao1, Fugen Zhou1, Qinglong Yao1, Jingjing Wang1, Bo Liu1, Haitao Sun2, Zhe Ji2, Yuliang Jiang2, Junjie
Wang2, and Qiuwen Wu3
1Image Processing Center, Beihang University, Beijing 100191, China
2Department of Radiation Oncology, Peking University Third Hospital, Beijing 100191, China
3Department of Radiation Oncology, Duke University Medical Center, Durham, NC 27710, USA
bo.liu@buaa.edu.cn
ABSTRACT
Seed implant brachytherapy (SIBT) is an effective cancer treatment modality; however, clinical
planning often relies on manual adjustment of objective function weights, leading to inefficiencies
and suboptimal results. This study proposes an adaptive weight optimization framework for SIBT
planning, driven by large language models (LLMs). A locally deployed DeepSeek-R1 LLM is
integrated with an automatic planning algorithm in an iterative loop. Starting with fixed weights, the
LLM evaluates plan quality and recommends new weights in the next iteration. This process continues
until convergence criteria are met, after which the LLM conducts a comprehensive evaluation to
identify the optimal plan. A clinical knowledge base, constructed and queried via retrieval-augmented
generation (RAG), enhances the model’s domain-specific reasoning. The proposed method was
validated on 23 patient cases, showing that the LLM-assisted approach produces plans that are
comparable to or exceeding clinically approved and fixed-weight plans, in terms of dose homogeneity
for the clinical target volume (CTV) and sparing of organs at risk (OARs). The study demonstrates
the potential use of LLMs in SIBT planning automation.
1 Introduction
Seed implant brachytherapy (SIBT) is an established and effective treatment modality for tumors at many site [6] [27].
The core principle of SIBT involves precisely delivering a high radiation dose to the tumor while minimizing exposure to
surrounding healthy tissues. This is achieved by carefully planning safe needle trajectories and optimizing the placement
of radioactive seeds. To enhance precision and improve treatment outcomes, the CT-guided dynamic dose planning
workflow has been developed. This workflow typically encompasses three key phases: preoperative imaging-based
planning (often incorporating 3D-printed templates for guidance), intraoperative CT imaging for real-time adjustments
and adaptive optimization, and postoperative imaging for comprehensive dose verification [12] [13] [35]. In this study,
we focus on the preoperative planning phase.
A critical step in SIBT planning, once the needle paths are determined, is the optimization of seed positions using
dose-based inverse planning methods. This process is commonly formulated and solved as an optimization problem
[15] [10] [33] [40]. Clinical objectives in SIBT are often multifaceted and can be interrelated or even conflicting. These
objectives typically include achieving adequate dose coverage of the clinical target volume (CTV), adhering to strict
dose constraints for organs at risk (OARs), and minimizing the number of needles and seeds used. A primary challenge
in this optimization lies in effectively translating these diverse clinical objectives into a cohesive, weighted objective
function that can be solved by optimization algorithms.
Currently, this challenge is addressed through trial-and-error adjustments, which are highly dependent on planner
expertise and remain time-consuming.Several automated weight-tuning strategies have been proposed previously, many
depend on handcrafted optimization pipelines [44], simplistic reward functions [23] [29], or multi-objective optimization
models to approximate the Pareto frontier [11]. However, these methods lack the capacity to semantically interpret
clinical objectives and offer limited adaptability to dynamic and patient-specific planning scenarios.arXiv:2509.08407v1  [physics.med-ph]  10 Sep 2025

APREPRINT
In recent years, large language models (LLMs) and multimodal large language models (MLLMs) have been explored
for weight tuning in radiotherapy treatment planning, showing promising potential [36] [17]. For example, Liu et al.
proposed GPT-RadPlan, a GPT-4V-based agent that emulates human planners by iteratively adjusting objective weights
and doses based on DVH analysis and textual feedback [17]. However, current approaches face limitations. Primarily,
they rely on general-purpose LLMs that lack integration with structured clinical knowledge, limiting their ability to
make context-aware decisions. Furthermore, most implementations depend on cloud-based APIs (e.g., ChatGPT-4 [1]),
which are unsuitable for offline or network-restricted clinical environments.
To address these limitations and further explore LLM-based automatic planning, we propose a novel framework
that leverages the reasoning capabilities of LLMs to iteratively optimize weights in SIBT planning. This approach
integrates a locally deployed retrieval-augmented generation (RAG) knowledge base with an LLM-driven evaluation
and refinement workflow. To the best of our knowledge, this is the first study to explore the use of LLMs for treatment
plan optimization in brachytherapy. The main contributions of this work are as follows:
•We construct a domain-specific, locally hosted RAG knowledge base tailored for SIBT planning, enabling
the LLM to dynamically incorporate clinical knowledge during plan evaluation and weight recommendation,
while ensuring data privacy and offline operability.
•We develop a fully local, LLM-guided iterative workflow that integrates plan evaluation and automated weight
adjustment. The standardized output schema allows seamless interaction between the language model and the
optimization solver.
•We demonstrated that this method outperforms fixed-weight baselines and achieves planning quality compara-
ble to clinical plans, with improved sparing of OAR and higher efficiency.
2 Related Works
2.1 Weight Tuning of Treatment Planning
Traditionally, weight tuning in SIBT has been carried out using protocol-based methods [44]. This approach simulates
the iterative decision-making process of human planners by starting with predefined clinical objectives and iteratively
adjusting these objectives in a flowchart-like manner based on plan quality feedback. In the context of high-dose-rate
(HDR) brachytherapy, some researchers have formulated weight tuning as a multi-objective optimization problem,
applying Bayesian optimization to identify Pareto-optimal combinations of penalty weights [11]. Other studies have
explored the use of reinforcement learning (RL) for automated weight tuning [23] [29]; however, the design of
appropriate reward functions and action spaces remains a significant challenge, limiting the practical effectiveness of
RL-based approaches. In the domain of external beam radiotherapy, LLMs have demonstrated the ability to emulate
planner behavior in adjusting optimization weights with a limited number of clinical planning examples, including text,
DVH tables, and dose distribution images [36] [17]. Despite promising early results, such approaches remain largely
unexplored in brachytherapy and face challenges in generalizing across diverse cases due to limited domain knowledge
integration. Moreover, most existing LLM-based planning methods rely on cloud-hosted APIs, which raises concerns
regarding data security, network dependency, and regulatory compliance, thereby limiting their suitability for clinical
deployment.
2.2 Large Language Models
Recent advances in large language models, such as DeepSeek-R1 [8], Gemini [32], GPT-4 [1], LLaMA [7], and
QWen [41], have demonstrated remarkable capabilities in instruction following [45], contextual learning [28], and
chain-of-thought reasoning [38]. Trained on massive general-domain corpora, these models exhibit emergent behaviors
such as in-context learning and few-shot generalization, allowing for integration into downstream workflows with
minimal task-specific supervision. However, due to their reliance on publicly available internet data, general-purpose
LLMs often underperform in specialized domains such as healthcare, finance, or law, where domain-specific terminology
and reasoning are critical.
To address this, domain-adapted LLMs have been developed through pretraining or fine-tuning on specialized datasets.
For instance, in the medical field, models such as BioGPT [18], Med-PaLM [30], and LLaV A-Med [14] integrate
structured biomedical knowledge and clinical corpora to improve performance on tasks such as clinical question
answering [30], radiology report generation [31], and image-guided diagnosis [16]. While effective, these approaches
demand significant computational resources, large volumes of annotated data, and frequent retraining to remain current.
This limits their scalability and flexibility in practical clinical settings.
2

APREPRINT
To overcome these limitations and maintain adaptability without incurring high retraining costs, our proposed approach
utilizes a general-purpose LLM that dynamically accesses external domain knowledge through retrieval-augmented
generation (RAG). Instead of embedding domain knowledge into model parameters, RAG retrieves relevant external
information during inference to supplement generation, offering improved transparency and up-to-date knowledge access.
This hybrid approach has shown promise in medical question answering and decision support [16] [43] [2] [37] [24],
and forms the foundation of our strategy for enabling general-purpose LLMs to reason over clinical data without
domain-specific fine-tuning.
3 Methods
3.1 Framework Overview
Figure 1: The overall framework of the proposed method.
SIBT planning procedure typically involves several key steps. First, based on the patient’s anatomical structure, a
set of candidate needle trajectories is generated either manually by physicians or through automated methods. Since
radioactive seeds are implanted along the needles, this step effectively defines the solution space. Subsequently, an
objective function is formulated, usually incorporating weighted conflicting terms for CTV coverage, OAR sparing, and
resource cost (e.g., number of needles and seeds). An optimization engine is then employed to solve this objective,
producing a candidate treatment plan. In the conventional paradigm, as shown in Fig. 1, the planning result is evaluated
by the physician and the weights are manually adjusted based on physician experience. This iterative weight tuning is
repeated until a satisfactory plan is achieved, a process that is time-consuming and subject to inter-operator variability.
In contrast to conventional workflows, the proposed framework employs a retrieval-augmented large language model
(RE-LLM) to automate both optimization weight tuning and treatment plan evaluation. By integrating a RAG mechanism,
the RE-LLM dynamically accesses locally stored clinical guidelines and physician-defined planning rules to support
protocol-compliant, informed decision-making. Starting from an initial set of weights, the model iteratively evaluates
the generated plans and suggests weight modifications to address identified suboptimalities. This process mimics the
reasoning strategy of experienced planners and allows the system to adaptively refine candidate plans in a closed-loop.
Detailed strategies for weight adjustment and multi-plan selection are described in the following sections.
3

APREPRINT
3.2 LLM-based Optimization
In the following, we describe three essential components of the proposed LLM-based optimization framework, which
are plan evaluation, weight tuning and multi-plan comparison and selection.
3.2.1 Plan Evaluation
Once the optimization engine generates a treatment plan using a given set of objective weights, relevant parameters
are extracted and stored. These include the DVH metrics for the CTV and OARs, as well as the number of implanted
seeds and needles. Historical optimization records are also preserved and incorporated into the plan evaluation prompt
submitted to the LLM.
To support accurate and context-aware reasoning, the LLM receives structured input that contains tabular representations
of DVH data, key dosimetric values, and implant-related information. This approach avoids the limitations of current
multimodal LLMs, which have difficulty interpreting DVH images directly [36]. Presenting the data in a text-based
tabular format allows the model to interpret quantitative information more reliably. In addition, clinical thresholds
and decision rules defined by physicians are stored in a local structured knowledge base. These are retrieved using
a retrieval-augmented generation mechanism to inform the LLM during evaluation. This integration ensures that the
model’s assessment aligns with established clinical standards and decision-making criteria.
During evaluation, the LLM analyzes the provided dosimetric inputs to identify key deficiencies, such as suboptimal
CTV coverage or excessive OAR dose, based on both absolute thresholds and relative trade-offs. The model also
compares the current plan against historical optimization results to recognize performance trends and detect repeated
failure patterns. By integrating these multiple sources of information, the LLM performs a qualitative assessment of
plan quality and generates natural language feedback to guide subsequent weight adjustments. This reasoning process
allows the system to iteratively refine treatment planning in alignment with clinical intent.
The closed-loop process continues until one of the following conditions is met: (1) CTV coverage satisfies clinical
criteria, but further tuning fails to reduce OAR dose without deteriorating the CTV coverage; (2) the needle count
cannot be further reduced without deteriorating the dose distribution for both OARs and CTV; (3) the maximum number
of iterations is reached, which is set to 10 in this study.
3.2.2 Weight Tuning
If the termination condition is not met, the system proceeds to adjust the objective function weights based on the
LLM’s evaluation of the current plan. The tuning process is guided by configurable clinical preferences that prioritize
adequate CTV coverage to ensure therapeutic efficacy, while also considering OAR sparing and minimizing procedural
complexity, such as the number of implanted needles.
While the exact weight adjustment is dynamically determined by the LLM, general clinical preferences are also
incorporated to inform its strategy through RAG mechanism. For example, if a critical metric such as CTV coverage is
far from acceptable, the corresponding weight can be significantly increased. For secondary metrics, such as needle
reduction, moderate upward adjustments can be applied when the primary objectives are already satisfied, allowing
for further optimization without compromising overall plan quality. In cases where two consecutive iterations yield
no observable improvement, the adjustment range may be expanded to help the optimizer escape local optima.If no
improvement is observed after three consecutive iterations, the process is considered to have met the stopping criterion.
3.2.3 Multi-Plan Comparison and Selection
After the tuning loop concludes, all candidate plans generated during the process are reviewed and compared by the
LLM. Rather than relying solely on the outcome of the final iteration, the system performs a holistic, knowledge-guided
evaluation of all intermediate results. This step mitigates the uncertainty inherent in heuristic optimization, where not
every weight adjustment guarantees improvement.
The multi-plan evaluation considers quantitative dosimetric metrics and clinical priorities retrieved from the local
knowledge base. Each plan is assessed across multiple dimensions, including CTV coverage and dose uniformity,
OAR sparing, and the efficiency of needle and seed utilization. Based on these multi-criteria assessments, the LLM
selects the plan that best reflects the clinical priority hierarchy and trade-offs. This final selection ensures consistency,
robustness, and clinical interpretability across patients. It also enables scalable and automated treatment planning
guided by expert-level decision logic.
4

APREPRINT
3.3 RAG-enhanced Prompt Generation
Prompt engineering plays a pivotal role in guiding LLMs to perform reliably across complex, domain-specific tasks such
as clinical treatment planning [19]. Given the inherent ambiguity and variability in natural language, a well-designed
prompt is essential to constrain the model’s reasoning, reduce inconsistency, and ensure alignment with clinical
expectations. The template follows a rigid and standardized format, explicitly presenting all relevant clinical input
data and guiding the model through evaluation and decision-making steps. This design improves the reproducibility of
model outputs, enhances compatibility with clinical protocols, and ensures stable performance across diverse patient
cases.
Figure 2: Prompt template for weight optimization. It includes task objectives, patient-specific inputs (e.g., DVH and
plan history), related clinical knowledge, and a standardized Q&A format for weight adjustment, plan evaluation, and
selection.
As shown in Fig. 2, the left side of the template provides contextual input, including the objective function structure,
patient-specific planning data and relevant clinical knowledge retrieved via the RAG module. This integrated context
ensures that the model’s reasoning remains aligned with protocol-defined criteria and patient-specific conditions. The
right side of the template defines a standardized question-and-answer format, which instructs the model to perform
weight adjustment, assess plan acceptability, and select the optimal plan among all iterations. Each response adheres to
a fixed schema designed for seamless parsing by the downstream optimization engine.
Figure 3: The framework of RAG-enhanced prompt generation.
5

APREPRINT
To compensate for the domain knowledge limitations of the base LLM, a RAG module was incorporated into the
proposed framework to dynamically inject SIBT-specific clinical knowledge during prompt generation and decision
making. As illustrated in Fig. 3, the RAG system consists of two key components: a structured clinical knowledge base
and a hybrid retrieval pipeline. During each iteration, the system queries relevant protocol and case-specific knowledge,
which is then incorporated into the prompt to guide the LLM in evaluating the current plan and recommending clinically
informed weight adjustments.
The knowledge base was constructed from professional textbooks, practice guidelines, and clinical reports relevant to
SIBT. Source documents in PDF and plain text formats were parsed and segmented using a two-stage splitting strategy.
First, an embedding-guided semantic chunking approach preserved contextual coherence; subsequently, content-aware
secondary splitting optimized chunk size for both retrieval accuracy and generative performance.
For knowledge retrieval, a hybrid retriever combining dense and sparse retrieval techniques was adopted. Dense retrieval
utilized semantic embeddings generated by the BGE-Small-Zh-v1.5 model [5] and stored in a Chroma-powered vector
database for efficient similarity search. Given a query q, the dense similarity score is calculated via cosine similarity
between the query embeddingv qand document embeddingsv i:
Sd(q, i) =vq·vi
∥vq∥∥vi∥(1)
For sparse retrieval, the BM25 algorithm [26] was utilized to complement the dense retrieval results. The BM25 score
is computed as:
Sb(q, D) =X
t∈qIDF(t)·f(t, D)·(k 1+ 1)
f(t, D) +k 1
1−b+b·|D|
avgdl (2)
where tdenotes a term in the query q, and f(t, D) is the frequency of term tin document D. The parameter |D|
indicates the length of document D, while avgdl denotes the average document length across the entire corpus. The
term IDF(t) reflects the inverse document frequency of term t, which down-weights common terms and emphasizes
more informative ones. The hyperparameter k1controls the scaling of term frequency contribution, while badjusts for
document length normalization. In our experiments,k 1= 1.2,b= 0.75.
The final retrieval score was obtained by weighted fusion of dense and sparse scores. To further refine retrieval precision,
a cross-encoder reranking model [22] was applied to re-score candidate documents based on query-document relevance
pairs. Only documents with final scores exceeding a predefined threshold are selected for retrieval and passed to the
language model for subsequent reasoning.
During the weight adjustment process in each iteration, the RAG module queries and retrieves knowledge specific to
the tumor site and OARs involved in the current case, enabling clinically informed evaluations and adaptive weight
recommendations.
4 Experiments
4.1 Patient data and preprocessing
A retrospective cohort of plan data was collected from 23 H&N patients who had undergone SIBT. The study was
approved by the Institutional Ethics Committee of Peking University Third Hospital (Beijing, China; Approval
No.M2021438). Each patient’s data included their planning CT, contours of the CTV and OARs, and comprehensive
treatment plan information such as needle paths, seed positions, air-kerma strength, and dose prescription.
Among the 23 cases, primary tumor locations varied. Seven patients (30%) presented with parotid tumors, four
(17%) with oral or oropharyngeal tumors, four (17%) with nasopharyngeal tumors, three (13%) with primary cervical
neoplasms, two (9%) with orbital or periorbital tumors, one (4%) with laryngeal or hypopharyngeal tumors, one (4%)
with a nasal or paranasal sinus tumor, and one (4%) with an infratemporal fossa tumor. The number and type of OARs
varied according to tumor location. A total of eight patients had their OARs contoured, which included the trachea,
esophagus, spinal cord, optic nerve, eyeball, lens, parotid gland, subclavian artery, internal carotid artery, and other
critical vascular structures. The prescription dose of the plans ranges between 80 and 120 Gy. The number of needles in
each plan ranges from 4 to 20 and seeds from 14 to 58. All plans use I-125 seeds of air-kerma strength in the range of
0.381-0.889 U and each plan only contains seeds of the same strength. The original CT images have varying resolutions
(in-slice resolution: 0.47 mm to 1.17 mm; slice thickness: 1.0 mm to 5.0 mm).
6

APREPRINT
4.2 Evaluation and Implementation
To quantitatively evaluate the performance of the proposed workflow, we compared planning results from three
approaches: (1) clinical plans, which involved manual needle path design and manual adjustment of objective function
weights; (2) fixed-weight optimization, which incorporated automated needle path design with fixed target weights; and
(3) the proposed LLM-enabled adaptive-weight optimization, which built upon the fixed-weight plans by applying the
proposed adaptive weight adjustment strategy using a large language model. For the CTV , we statistically evaluated
dose indices including D90,V100,V150, andV200. For OARs, evaluation was performed using parameters including
V50,D mean,D max,D1,D1cc, andD 0.1cc.
To define the solution space, needle trajectories were generated following the approach described in our previous
work [40]. For objective definition, a penalty-driven cost function was employed, defined as:
f(P) =X
o∈{ctv,oar}1
noX
jwh
oH(do,j−DUB
o)(do,j−DUB
o)
+1
nctvX
jwl
ctvH(DLB
ctv−d ctv,j)(DLB
ctv−d ctv,j)
+wnN(P)(3)
The function penalizes CTV underdose, CTV/OAR overdose, and excessive needle usage. The first term applies to dose
violations in both CTV and OAR structures;the second addresses underdosage by penalizing voxels receiving doses
below the lower bound; and the third penalizes the total number of implanted needles to encourage minimally invasive
plans. The Heaviside function H(·) activate penalties upon constraint violation. For the CTV , the lower and upper
bounds are set to 1 and 2 times the prescription dose, respectively; for each OAR, the upper bound is set to 1 times the
prescription dose. Prepresents the seed and needle configuration. Initial fixed weights wl
ctv,wh
ctv,wh
oarandwnwere set
to 20, 0.01, 1 and 600. The number of tunable weights ranges from 3 to 6, depending on the number of OARs involved.
The optimization engine uses a heuristic strategy involving iterative seed addition and removal. In each iteration,
the algorithm temporarily adds a seed to every unoccupied candidate position, evaluates the objective function, and
permanently retains the seed yielding the largest cost reduction. This is followed by a removal phase, where each
existing seed is temporarily deleted, and the one whose removal produces the greatest cost reduction is permanently
excluded. The process repeats until no further improvement is achieved, at which point the optimization terminates.
The core algorithms for needle trajectory generation and optimization were developed using C/C++ and CUDA 11.8.
For dose calculations, we adhered to the AAPM TG-43 formalism [21] [25] to ensure consistency with the clinical
reference plans, allowing for a fair and direct comparison. The dose computation was substantially accelerated through
CUDA-based GPU parallel processing.
For the implementation of RAG, text data and documents were processed and loaded using the LangChain library [4].We
used Python’s pandas library [20] to parse CSV files of DVH, extracting data and populating prompt placeholders to
construct the LLM’s input. For LLM deployment, the DeepSeek-R1-Distill-Qwen-14B [9] model was deployed on an
NVIDIA A100 (80 GB VRAM) GPU for inference, leveraging the Hugging Face Transformers library [39]. Model
inference was performed using half-precision (float16) to improve computational efficiency.
4.3 Comparison Experiments
Table 1 summarizes the quantitative results for different planning strategies. Fig. 4 shows the mean DVHs and
corresponding standard deviation bands for the CTV across the three approaches. Compared to clinical plans, the
proposed method achieved comparable DVH metrics while requiring fewer needles. Furthermore, the adaptive plans
demonstrated reduced variability, suggesting improved planning stability. Compared with fixed-weight planning, the
proposed approach yielded better dose homogeneity in the CTV , withV 150andV 200not being excessively high.
Table 2 summarizes the quantitative analysis results for OARs across the clinical plans, fixed-weight plans, and adaptive
plans. Compared to the fixed-weight approach, the adaptive plans consistently achieved lower OAR doses across most
metrics, indicating improved of OAR sparing. In particular, reductions in high-dose exposure metrics (e.g., D1,D0.1 cc )
suggest that the adaptive strategy better controls dose hotspots within OARs. While the differences between adaptive
and clinical plans are less pronounced, the adaptive approach still demonstrated comparable or improved performance in
several parameters. These findings highlight the effectiveness of the LLM-guided weight tuning framework in balancing
target coverage with OAR protection.
7

APREPRINT
Table 1: Comparison of CTV dosimetric parameters and resource usage among clinical plans, fixed-weight inverse
plans, and adaptive plans.
PlansV 100(%)↑V 150(%)↓V 200(%)↓D 90(%)↑#needle↓
Clinical plans94.9±4.3 73.0±15.1 47.8±19.5 119.5±17.3 10±3
Fixed-weight plans97.6±2.6 82.9±7.6 63.1±9.1 133.1±12.8 9±2
Adaptive plans96.4±1.6 75.2±6.2 51.8±8.6 120±7.9 9±2
pvalue (CP vs AP)0.2 0.56 0.41 0.9 0.02
pvalue (FWP vs AP)0.23<0.001<0.001<0.001 0.27
Note: Pvalues are from two-sided Wilcoxon signed-rank tests comparing adaptive plans with clinical plans (CP vs AP)
and fixed-weight plans (FWP vs AP).
Figure 4: Comparison of mean CTV DVH and DVH SD of all patient cases.
Table 2: Comparison of OAR dosimetric parameters among clinical plans, fixed-weight inverse plans, and adaptive
plans.
PlansV 50(%)↓D mean(Gy)↓D max(Gy)↓D 1(Gy)↓D 1cc(Gy)↓D 0.1cc (Gy)↓
Clinical plans41.7±20.9 52.5±19.2 140.7±57.1 129.9±50.5 52.8±22.1 99.7±25.9
Fixed-weight plans42.0±21.3 48.7±17.4 116.6±38.8 106.9±33.4 50.8±24.6 89.3±29.2
Adaptive plans36.3±22.2 44.8±16.5 109.5±36.9 98.5±29.8 47.0±22.2 81.6±25.3
pvalue (CP vs AP)0.25 0.05 0.11 0.02 0.2 0.05
pvalue (FWP vs AP)0.05 0.15 0.74 0.11 0.25 0.2
Note: Pvalues are from two-sided Wilcoxon signed-rank tests comparing adaptive plans with clinical plans (CP vs AP)
and fixed-weight plans (FWP vs AP).
8

APREPRINT
The optimization process converged within 5.3 ± 0.8 iterations (ranging from 4 to 7), with the final selected plan located
at iteration 3.2 ± 1.4. Across all test cases, none reached the predefined maximum of 10, indicating that the proposed
framework efficiently identifies clinically favorable solutions within a limited number of iterations. Notably, the selected
plan typically emerged about two iterations before termination, implying that the final steps mainly refine marginal
trade-offs and validate convergence.
Figure 5: Comparison of iso-dose distributions at 130% (magenta), 100% (yellow), and 70% (cyan) of the prescription
dose obtained by clinical plans (left column), fixed-weight plans (middle column), and LLM-assisted plans (right
column). Three representative slices from different testing patient cases are shown. The red regions represent the CTVs.
OARs are delineated with various colors, including the blue and orange regions.
Fig. 5 compares iso-dose distributions for three representative patient cases across clinical, fixed-weight, and LLM-
assisted adaptive plans. Fixed-weight plans show significant high-dose spillover, with expanded iso-dose regions
extending beyond target boundaries, suggesting potential overexposure of adjacent tissues. Conversely, LLM-assisted
adaptive plans demonstrate superior high-dose conformity to the CTVs, effectively limiting excessive dose delivery
while ensuring adequate target coverage. Additionally, adaptive plans improve sparing of surrounding OARs, such as
the trachea and esophagus in the third case, underscoring the clinical advantages of the proposed weight optimization
framework.
9

APREPRINT
Table 3: Ablation study of CTV dosimetric parameters, OAR sparing, and resource utilization with and without RAG.
PlansV 100(%)↑V 150(%)↓V 200(%)↓D 90(%)↑#needle↓
w/o RAG95.4±3.4 75.1±8.7 54.0±11.1 119.4±14.3 9±2
w RAG96.4±1.6 75.2±6.2 51.8±8.6 120±7.9 9±2
pvalue0.69 0.34 0.15 0.43 0.25
PlansV 50(%)↓D mean(Gy)↓D max(Gy)↓D 1(Gy)↓D 1cc(Gy)↓D 0.1cc (Gy)↓
w/o RAG38.7±31.6 47.4±24.9 113.1±66.3 102.9±50.5 47.4±28.1 82.5±40.1
w RAG36.3±22.2 44.8±16.5 109.5±36.9 98.5±29.8 47.0±22.2 81.6±25.3
pvalue0.31 0.16 0.22 0.44 0.16 0.16
Note:Pvalues are from two-sided Wilcoxon signed-rank tests comparing plans with RAG versus without RAG.
4.4 Ablation Study
To further evaluate the contribution of the RAG module, we conducted an ablation study comparing plan quality with
and without RAG integration. As summarized in Table 3, while CTV coverage and needle number remained consistent
between the two groups, the RAG-assisted plans demonstrated improved OAR sparing across multiple dosimetric
metrics. These results suggest that incorporating retrieval-augmented clinical knowledge enhances the model’s ability
to balance target coverage with normal tissue protection, contributing to more refined and clinically favorable treatment
plans.
Without RAG, the average number of iterations until termination increased to 7.4 ± 3.1, with the selected plan located at
iteration 5.3 ± 4.0. Notably, five cases reached the predefined upper limit of 10 iterations, indicating a higher likelihood
of prolonged or suboptimal convergence when external clinical knowledge is not incorporated.
Figure 6: Comparison of responses generated by LLM w/o (left) and w (right) RAG support. The hallucination-
prone LLM without knowledge retrieval exhibits multiple clinically inconsistent suggestions. Correct and incorrect
descriptions are highlighted in green and red, respectively.
As illustrated in Fig. 6, we also compared the responses generated by the LLM during weight recommendation, with
and without knowledge retrieval, to assess the impact of RAG on decision quality. The descriptions reflect the LLM’s
reasoning process and were manually verified for correctness.
10

APREPRINT
4.5 Computational efficiency
In clinical practice, computational efficiency is essential to ensure timely treatment planning and support adaptive
workflows. As previously described, the optimization process in our framework converged within 5.3 ± 0.8 iterations.
All experiments were performed on a workstation equipped with an NVIDIA A100 GPU (80 GB VRAM) and an Intel
Xeon w5-2465X CPU.
The total processing time was 3.7 ± 0.7 minutes per patient. Within each iteration, the heuristic solver required an
average of 15.9 ± 13.4 seconds to converge, while the LLM-based evaluation and weight recommendation modules took
26.4 ± 8.4 seconds. Notably, the weight recommendation step involved processing both CTV and OAR DVH summary
data as input, resulting in a larger token length and consequently longer inference time. In contrast, the evaluation
step primarily relied on the optimization history, requiring fewer tokens and thus achieving relatively faster inference.
Compared to conventional manual planning approaches, the proposed framework substantially reduces the overall
planning time and operator workload.
5 Discussion and conclusion
This work presents a novel LLM-guided framework for adaptive weight optimization in SIBT planning. By integrating
a locally deployed LLM model with an automated optimization engine in a closed-loop architecture, the system enables
dynamic adjustment of objective weights based on plan evaluation feedback. To enhance the domain-specific reasoning
capabilities of the LLM, a clinical knowledge base is incorporated and queried through a RAG mechanism, allowing
the model to align its recommendations with established treatment guidelines. To the best of our knowledge, this
study represents the first application of a locally deployed LLM to brachytherapy plan optimization. The approach
significantly reduces manual parameter tuning while improving the dosimetric quality of treatment plans, particularly in
terms of CTV coverage and OAR sparing.
One advantage of the proposed framework lies in its ability to operate entirely on local computational resources, which
ensures data privacy and facilitates integration into clinical environments without reliance on external servers. To
evaluate the feasibility of such deployment in real-world settings, we measured the actual planning time under local
execution. Across all tested cases, the complete planning process for a single patient was completed within 3.7 ± 0.7
minutes, which is well within the acceptable range for routine clinical workflows. These results demonstrate that the
framework remains computationally efficient for practical deployment despite the use of large-scale LLM and RAG
components. Furthermore, with the application of concurrent LLM inference and model acceleration techniques, the
overall processing time can be reduced.
Nevertheless, several aspects of the current framework remain open for further exploration and optimization. While the
LLM assists weight adjustment, the underlying optimization solver still relies on existing heuristic algorithms [44], which
are prone to stochastic perturbations and may be trapped in suboptimal local minima. An emerging research direction
involves integrating LLM reasoning capabilities directly into the optimization process to guide search trajectories,
thereby improving convergence efficiency and solution quality. Recent studies of such hybrid optimization paradigms
that combine LLMs and traditional solvers have shown potential to further enhance planning robustness [42] [34].
Beyond weight adjustment, multimodal large language models also hold promise in needle trajectory design and
evaluation. Although our current approach [40] has achieved automated multi-needle placement, it still falls short of fully
meeting clinical requirements and preferences, with certain trajectories requiring manual refinement. By incorporating
imaging data, clinical context, and historical planning knowledge, MLLMs may facilitate more comprehensive decision-
making and further reduce dependence on human expertise in needle trajectory planning, which remains a highly
variable and complex aspect of clinical practice.
Finally, while the proposed framework demonstrates strong performance under experimental conditions, further studies
are needed to validate its generalizability across different anatomical sites, treatment modalities, and clinical protocols.
In addition, ensuring safety, reproducibility, and regulatory compliance are essential for clinical translation, and
human-in-the-loop strategies [3] may remain necessary to supervise automated recommendations in high-stakes medical
applications.
References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo
Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report.arXiv preprint
arXiv:2303.08774, 2023.
11

APREPRINT
[2]Iñigo Alonso, Maite Oronoz, and Rodrigo Agerri. MedExpQA: Multilingual benchmarking of Large Language
Models for Medical Question Answering.Artificial Intelligence in Medicine, 155:102938, September 2024.
[3]Samuel Budd, Emma C. Robinson, and Bernhard Kainz. A Survey on Active Learning and Human-in-the-Loop
Deep Learning for Medical Image Analysis.Medical Image Analysis, 71:102062, July 2021.
[4] Harrison Chase. LangChain, October 2022.
[5]Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding: Multi-
lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation.arXiv preprint
arXiv:2402.03216, 2024.
[6]Marjorie Mae Cua, Carl Jay Jainar, Janella Ann Javenrie Calapit, Michael Benedict Mejia, and Warren Bacorro. The
evolving landscape of head and neck brachytherapy: A scoping review.Journal of Contemporary Brachytherapy,
16(3):225–231, 2024.
[7]Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle,
Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models.arXiv e-prints,
pages arXiv–2407, 2024.
[8]Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi
Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.arXiv
preprint arXiv:2501.12948, 2025.
[9]Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi
Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.arXiv
preprint arXiv:2501.12948, 2025.
[10] C Guthier, K P Aschenbrenner, D Buergy, M Ehmann, F Wenz, and J W Hesser. A new optimization method using
a compressed sensing inspired solver for real-time LDR-brachytherapy treatment planning.Physics in Medicine
and Biology, 60(6):2179–2194, March 2015.
[11] Hossein Jafarzadeh, Majd Antaki, Ximeng Mao, Marie Duclos, Farhard Maleki, and Shirin A Enger. Penalty
weight tuning in high dose rate brachytherapy using multi-objective Bayesian optimization.Physics in Medicine
& Biology, 69(11):115024, June 2024.
[12] Zhe Ji, Yuliang Jiang, Suqing Tian, Fuxin Guo, Ran Peng, Fei Xu, Haitao Sun, Jinghong Fan, and Junjie Wang.
The Effectiveness and Prognostic Factors of CT-Guided Radioactive I-125 Seed Implantation for the Treatment of
Recurrent Head and Neck Cancer After External Beam Radiation Therapy.International Journal of Radiation
Oncology*Biology*Physics, 103(3):638–645, March 2019.
[13] Yuliang Jiang, Zhe Ji, Fuxin Guo, Ran Peng, Haitao Sun, Jinghong Fan, Shuhua Wei, Weiyan Li, Kai Liu, Jinghua
Lei, and Junjie Wang. Side effects of CT-guided implantation of 125I seeds for recurrent malignant tumors of the
head and neck assisted by 3D printing non co-planar template.Radiation Oncology, 13(1):18, December 2018.
[14] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung
Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day.
Advances in Neural Information Processing Systems, 36:28541–28564, 2023.
[15] Bin Liang, Fugen Zhou, Bo Liu, Junjie Wang, and Yong Xu. A novel greedy heuristic-based approach to
intraoperative planning for permanent prostate brachytherapy.Journal of Applied Clinical Medical Physics,
16(1):229–245, January 2015.
[16] Fenglin Liu, Tingting Zhu, Xian Wu, Bang Yang, Chenyu You, Chenyang Wang, Lei Lu, Zhangdaihong Liu,
Yefeng Zheng, Xu Sun, Yang Yang, Lei Clifton, and David A. Clifton. A medical multimodal large language
model for future pandemics.npj Digital Medicine, 6(1):226, December 2023.
[17] Sheng Liu, Oscar Pastor-Serrano, Yizheng Chen, Matthew Gopaulchan, Weixing Liang, Mark Buyyounouski,
Erqi Pollom, Quynh-Thu Le, Michael Gensheimer, Peng Dong, et al. Automated radiotherapy treatment planning
guided by gpt-4vision.Physics in Medicine & Biology, 70(15):155002, 2025.
[18] Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon, and Tie-Yan Liu. BioGPT: Generative
Pre-trained Transformer for Biomedical Text Generation and Mining.Briefings in Bioinformatics, 23(6):bbac409,
November 2022.
[19] Ggaliwango Marvin, Nakayiza Hellen, Daudi Jjingo, and Joyce Nakatumba-Nabende. Prompt engineering in
large language models. InInternational conference on data intelligence and cognitive informatics, pages 387–402.
Springer, 2023.
[20] Wes McKinney. Data Structures for Statistical Computing in Python. InPython in Science Conference, pages
56–61, Austin, Texas, 2010.
12

APREPRINT
[21] Ravinder Nath, Lowell L. Anderson, Gary Luxton, Keith A. Weaver, Jeffrey F. Williamson, and Ali S. Meigooni.
Dosimetry of interstitial brachytherapy sources: Recommendations of the AAPM Radiation Therapy Committee
Task Group No. 43.Medical Physics, 22(2):209–234, February 1995.
[22] Rodrigo Nogueira and Kyunghyun Cho. Passage re-ranking with bert.arXiv preprint arXiv:1901.04085, 2019.
[23] Gang Pu, Shan Jiang, Zhiyong Yang, Yuanjing Hu, and Ziqi Liu. Deep reinforcement learning for treatment
planning in high-dose-rate cervical brachytherapy.Physica Medica, 94:1–7, February 2022.
[24] Mahimai Raja, E Yuvaraajan, et al. A rag-based medical assistant especially for infectious diseases. In2024
International Conference on Inventive Computation Technologies (ICICT), pages 1128–1133. IEEE, 2024.
[25] Mark J. Rivard, Bert M. Coursey, Larry A. DeWerd, William F. Hanson, M. Saiful Huq, Geoffrey S. Ibbott,
Michael G. Mitch, Ravinder Nath, and Jeffrey F. Williamson. Update of AAPM Task Group No. 43 Report: A
revised AAPM protocol for brachytherapy dose calculations.Medical Physics, 31(3):633–674, February 2004.
[26] Stephen Robertson and Hugo Zaragoza. The Probabilistic Relevance Framework: BM25 and Beyond.Foundations
and Trends® in Information Retrieval, 3(4):333–389, 2009.
[27] Julianna Rodin, V oichita Bar-Ad, David Cognetti, Joseph Curry, Jennifer Johnson, Chad Zender, Laura Doyle,
David Kutler, Benjamin Leiby, William Keane, and Adam Luginbuhl. A systematic review of treating recurrent
head and neck cancer: A reintroduction of brachytherapy with or without surgery.Journal of Contemporary
Brachytherapy, 10(5):454–462, 2018.
[28] Amirreza Rouhi, Diego Patiño, and David K. Han. Enhancing Object Detection by Leveraging Large Language
Models for Contextual Knowledge. In Apostolos Antonacopoulos, Subhasis Chaudhuri, Rama Chellappa, Cheng-
Lin Liu, Saumik Bhattacharya, and Umapada Pal, editors,Pattern Recognition, volume 15317, pages 299–314.
Springer Nature Switzerland, Cham, 2025.
[29] Chenyang Shen, Yesenia Gonzalez, Peter Klages, Nan Qin, Hyunuk Jung, Liyuan Chen, Dan Nguyen, Steve B
Jiang, and Xun Jia. Intelligent inverse treatment planning via deep reinforcement learning, a proof-of-principle
study in high dose-rate brachytherapy for cervical cancer.Physics in Medicine & Biology, 64(11):115013, May
2019.
[30] Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Mohamed Amin, Le Hou, Kevin Clark,
Stephen R. Pfohl, Heather Cole-Lewis, Darlene Neal, Qazi Mamunur Rashid, Mike Schaekermann, Amy Wang,
Dev Dash, Jonathan H. Chen, Nigam H. Shah, Sami Lachgar, Philip Andrew Mansfield, Sushant Prakash, Bradley
Green, Ewa Dominowska, Blaise Agüera Y Arcas, Nenad Tomašev, Yun Liu, Renee Wong, Christopher Semturs,
S. Sara Mahdavi, Joelle K. Barral, Dale R. Webster, Greg S. Corrado, Yossi Matias, Shekoofeh Azizi, Alan
Karthikesalingam, and Vivek Natarajan. Toward expert-level medical question answering with large language
models.Nature Medicine, 31(3):943–950, March 2025.
[31] Phillip Sloan, Philip Clatworthy, Edwin Simpson, and Majid Mirmehdi. Automated Radiology Report Generation:
A Review of Recent Advances.IEEE Reviews in Biomedical Engineering, 18:368–387, 2025.
[32] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk,
Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models.arXiv
preprint arXiv:2312.11805, 2023.
[33] Mateo Villa, Julien Bert, Antoine Valeri, Ulrike Schick, and Dimitris Visvikis. Fast Monte Carlo-Based Inverse
Planning for Prostate Brachytherapy by Using Deep Learning.IEEE Transactions on Radiation and Plasma
Medical Sciences, 6(2):182–188, February 2022.
[34] Debing Wang, Zizhen Zhang, and Yi Teng. Large language model implemented simulated annealing algorithm for
traveling salesman problem. In2024 IEEE International Conference on Systems, Man, and Cybernetics (SMC),
pages 209–214. IEEE, 2024.
[35] Junjie Wang, Fujun Zhang, Jinhe Guo, Shude Chai, Guangjun Zheng, Kaixian Zhang, Anyan Liao, Ping Jiang,
Yuliang Jiang, and Zhe Ji. Expert consensus workshop report: Guideline for three-dimensional printing template-
assisted computed tomography-guided125I seeds interstitial implantation brachytherapy.Journal of Cancer
Research and Therapeutics, 13(4):607, 2017.
[36] Qingxin Wang, Zhongqiu Wang, Minghua Li, Xinye Ni, Rong Tan, Wenwen Zhang, Maitudi Wubulaishan, Wei
Wang, Zhiyong Yuan, Zhen Zhang, and Cong Liu. A feasibility study of automating radiotherapy planning with
large language model agents.Physics in Medicine & Biology, 70(7):075007, April 2025.
[37] Zixiang Wang, Yinghao Zhu, Junyi Gao, Xiaochen Zheng, Yuhui Zeng, Yifan He, Bowen Jiang, Wen Tang,
Ewen M Harrison, Chengwei Pan, et al. Retcare: Towards interpretable clinical decision making through
llm-driven medical knowledge retrieval. InArtificial Intelligence and Data Science for Healthcare: Bridging
Data-Centric AI and People-Centric Healthcare, 2024.
13

APREPRINT
[38] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al.
Chain-of-thought prompting elicits reasoning in large language models.Advances in neural information processing
systems, 35:24824–24837, 2022.
[39] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac,
Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick V on Platen, Clara Ma, Yacine
Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander
Rush. Transformers: State-of-the-Art Natural Language Processing. InProceedings of the 2020 Conference
on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online, 2020.
Association for Computational Linguistics.
[40] Zhuo Xiao, Tianyu Xiong, Lishen Geng, Fugen Zhou, Bo Liu, Haitao Sun, Zhe Ji, Yuliang Jiang, Junjie Wang,
and Qiuwen Wu. Automatic planning for head and neck seed implant brachytherapy based on deep convolutional
neural network dose engine.Medical Physics, 51(2):1460–1473, February 2024.
[41] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen
Huang, Chenxu Lv, et al. Qwen3 technical report.arXiv preprint arXiv:2505.09388, 2025.
[42] Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V Le, Denny Zhou, and Xinyun Chen. Large
language models as optimizers.arXiv preprint arXiv:2309.03409, 2023.
[43] Gongbo Zhang, Zihan Xu, Qiao Jin, Fangyi Chen, Yilu Fang, Yi Liu, Justin F. Rousseau, Ziyang Xu, Zhiyong Lu,
Chunhua Weng, and Yifan Peng. Leveraging long context in retrieval augmented language models for medical
question answering.npj Digital Medicine, 8(1):239, May 2025.
[44] Ruijin Zhang, Zhiyong Yang, Shan Jiang, Xiaoling Yu, Erpeng Qi, Zeyang Zhou, and Guobin Zhang. An inverse
planning simulated annealing algorithm with adaptive weight adjustment for LDR pancreatic brachytherapy.
International Journal of Computer Assisted Radiology and Surgery, 17(3):601–608, March 2022.
[45] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou.
Instruction-following evaluation for large language models.arXiv preprint arXiv:2311.07911, 2023.
14
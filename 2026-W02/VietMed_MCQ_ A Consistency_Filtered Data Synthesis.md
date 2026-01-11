# VietMed-MCQ: A Consistency-Filtered Data Synthesis Framework for Vietnamese Traditional Medicine Evaluation

**Authors**: Huynh Trung Kiet, Dao Sy Duy Minh, Nguyen Dinh Ha Duong, Le Hoang Minh Huy, Long Nguyen, Dien Dinh

**Published**: 2026-01-07 10:49:56

**PDF URL**: [https://arxiv.org/pdf/2601.03792v1](https://arxiv.org/pdf/2601.03792v1)

## Abstract
Large Language Models (LLMs) have demonstrated remarkable proficiency in general medical domains. However, their performance significantly degrades in specialized, culturally specific domains such as Vietnamese Traditional Medicine (VTM), primarily due to the scarcity of high-quality, structured benchmarks. In this paper, we introduce VietMed-MCQ, a novel multiple-choice question dataset generated via a Retrieval-Augmented Generation (RAG) pipeline with an automated consistency check mechanism. Unlike previous synthetic datasets, our framework incorporates a dual-model validation approach to ensure reasoning consistency through independent answer verification, though the substring-based evidence checking has known limitations. The complete dataset of 3,190 questions spans three difficulty levels and underwent validation by one medical expert and four students, achieving 94.2 percent approval with substantial inter-rater agreement (Fleiss' kappa = 0.82). We benchmark seven open-source models on VietMed-MCQ. Results reveal that general-purpose models with strong Chinese priors outperform Vietnamese-centric models, highlighting cross-lingual conceptual transfer, while all models still struggle with complex diagnostic reasoning. Our code and dataset are publicly available to foster research in low-resource medical domains.

## Full Text


<!-- PDF content starts -->

VietMed-MCQ: A Consistency-Filtered Data Synthesis Framework for
Vietnamese Traditional Medicine Evaluation
Huynh Trung Kiet1,2,†, Dao Sy Duy Minh1,2,†, Nguyen Dinh Ha Duong1,2,†,
Le Hoang Minh Huy1,2,†, Long Nguyen1,2,∗, Dien Dinh1,2
1Faculty of Information Technology, University of Science, Ho Chi Minh City, Vietnam
2Vietnam National University, Ho Chi Minh City, Vietnam
†Equal contribution.∗Corresponding author.
{23122039,23122041,23122002,23122033}@student.hcmus.edu.vn
{nhblong,ddien}@fit.hcmus.edu.vn
Abstract
Large Language Models (LLMs) have demon-
strated remarkable proficiency in general med-
ical domains. However, their performance sig-
nificantly degrades in specialized, culturally
specific domains such as Vietnamese Tradi-
tional Medicine (VTM), primarily due to the
scarcity of high-quality, structured benchmarks.
In this paper, we introduceVietMed-MCQ,
a novel multiple-choice question dataset gen-
erated via a Retrieval-Augmented Generation
(RAG) (Lewis et al., 2020) pipeline with an au-
tomated consistency check mechanism. Unlike
previous synthetic datasets, our framework in-
corporates a dual-model validation approach to
ensure reasoning consistency through indepen-
dent answer verification, though the substring-
based evidence checking has known limita-
tions (detailed in Section 6). The complete
dataset of 3,190 questions spans three diffi-
culty levels (Easy: 22.0%, Medium: 51.8%,
Hard: 26.3%) and underwent validation by
one medical expert and four students, achiev-
ing 94.2% approval with substantial inter-rater
agreement (Fleiss’ κ= 0.82). We benchmark
seven open-source models on VietMed-MCQ.
Surprisingly, results show that general-purpose
models like Qwen2.5 outperform Vietnamese-
centric models, yet all struggle with complex
diagnostic reasoning. Our code and dataset
are publicly available to foster research in low-
resource medical domains.
1 Introduction
Synthetic data generation has emerged as a criti-
cal technique for addressing data scarcity in spe-
cialized domains, yet maintaining factual fidelity
remains a fundamental challenge. While LLM-
generated datasets have proliferated across NLP
tasks, medical applications demand rigorous qual-
ity control due to the risk of hallucinated content
(Ji et al., 2023). This challenge intensifies in low-
resource medical domains like Vietnamese Tradi-
tional Medicine (VTM), where knowledge existsprimarily in unstructured textbooks rather than cu-
rated databases (Johnson et al., 2023).
We address three core NLP challenges through
this work:(1) Hallucination mitigation in
synthetic data generationvia automated dual-
model consensus verification that filters 7.78%
of generated samples without human labeling;
(2) Cross-lingual knowledge transferby inves-
tigating whether conceptual overlap (shared Sino-
Vietnamese medical terminology) enables models
trained on Traditional Chinese Medicine to general-
ize to Vietnamese contexts;(3) Quality-controlled
benchmark creationthrough a teacher-student dis-
tillation pipeline (Hinton et al., 2015) that produces
3,190 clinically validated questions from unstruc-
tured corpora, achieving 94.2% expert approval
(Fleiss’κ= 0.82) (Landis and Koch, 1977).
Our methodology combines retrieval-augmented
generation (Lewis et al., 2020) with automated con-
sistency filtering: a 70B teacher model (Llama-3.1-
70B) generates context-referenced questions, while
a 32B student model (Qwen2.5-32B) indepen-
dently validates reasoning consistency. This dual-
verification achieves 92.22% retention (3,190 from
3,459 candidates), demonstrating that multi-model
consensus can serve as a scalable quality gate for
synthetic data pipelines, though the substring-based
evidence verification has known limitations (Sec-
tion 6). To address positional answer bias inher-
ent in LLM generation (50.3% Option B), we em-
ploy randomized option shuffling during evalua-
tion to ensure positional invariance. Evaluation of
seven representative models reveals a counterintu-
itive finding: models with Chinese language priors
substantially outperform Vietnamese-specialized
variants (+7.21% accuracy), suggesting that cross-
lingual conceptual transfer from related medical
traditions dominates language-specific fine-tuning
for culturally-rooted knowledge domains. These
findings have broader implications for multilin-
gual NLP in specialized domains where linguisticarXiv:2601.03792v1  [cs.CL]  7 Jan 2026

boundaries do not align with conceptual bound-
aries.
2 Related Work
Medical QA Benchmarks.Early medical bench-
marks focused on Western medicine, such as
MedQA (Jin et al., 2021) and PubMedQA (Jin
et al., 2019). Recent work has expanded to multilin-
gual and regional contexts, including AfriMedQA
(Tonja et al., 2024) for African languages and
MMedBench (Zhang et al., 2024a) for multilingual
evaluation. However, traditional medicine systems
remain underrepresented.
MCQ Format Considerations.Recent studies
(Chen et al., 2024) have shown that MCQ formats
can overestimate model capabilities compared to
free-response evaluation, as models may exploit
surface patterns or positional biases. We address
this through explicit de-biasing (option shuffling)
and plan to release a free-response variant in future
work.
Traditional Chinese Medicine (TCM) Re-
sources.TCM-specific benchmarks like Bian-
Cang (Guo et al., 2023) and TCM-Ladder (Li et al.,
2024) demonstrate the value of domain-specialized
evaluation. Given the conceptual overlap between
TCM and VTM, we investigate whether TCM-prior
models transfer effectively to Vietnamese contexts.
Synthetic Data Generation.Our teacher-student
pipeline builds on knowledge distillation (Hinton
et al., 2015) and retrieval-augmented generation
(Lewis et al., 2020). Similar consistency filter-
ing approaches have been explored in MedRGAG
(Zhang et al., 2024b) for reducing hallucinations in
biomedical text generation.
3 Methodology
Our proposed framework,VietMed-Gen, oper-
ates on a generator-validator pipeline designed to
synthesize high-fidelity multiple-choice questions
(MCQs) from unstructured medical texts.1As illus-
trated in Figure 1, the process comprises three dis-
tinct phases: (1) Context Extraction, (2) Teacher-
Guided Generation, and (3) Student-Based Consis-
tency Filtering.
1We use "Teacher-Student" terminology to denote a
producer-verifier architecture where the Student independently
validates outputs, rather than knowledge distillation in the tra-
ditional sense (Hinton et al., 2015).3.1 Data Acquisition and Chunking
We curated a corpus of authoritative Vietnamese
Traditional Medicine (VTM) textbooks and clin-
ical guidelines. Documents were segmented into
coherent text chunks containing specific medical
knowledge (e.g., herbal properties, diagnostic pro-
tocols). LetDbe the document corpus; we decom-
pose it into a set of contexts C={c 1,c2,...,c n},
where each cirepresents a localized knowledge
unit. Chunking parameters are detailed in Ap-
pendix D.
3.2 Teacher-Guided Generation (M T)
We designate a large-scale instruction-tuned model,
specificallyLlama-3.1-70B-Instruct-AWQ, as the
Teacher(MT). For each context ci, we construct a
promptPgenthat instructs the Teacher to generate
a tupleT= (q,O,a,r,e), where:
•q: The medical question stem.
•O: The set of four options{A,B,C,D}.
•a: The correct answer key (a∈O).
•r: The reasoning chain explaining why ais
correct.
•e: The specific evidence span quoted directly
fromc i.
The generation process is formalized as:
(q,O,a,r,e)∼M T(Pgen|ci;θ)(1)
whereθdenotes sampling hyperparameters opti-
mized for factual accuracy (see Appendix D for full
configuration). We enforce structured JSON output
through schema-constrained generation.
3.3 Student-Based Consistency Filtering
(MS)
A major challenge in synthetic data generation is
"hallucination," where the generated question is
unanswerable or the answer key is incorrect. To
mitigate this, we introduce aConsistency Filterus-
ing aStudentmodel ( MS), specificallyQwen2.5-
32B-Instruct-AWQ. The Student operates as an
independent verifier with conservative sampling to
enforce stricter reasoning validation.
The validation process involves two rigorous
checks:

Figure 1:The VietMed-MCQ Data Synthesis Framework.We employ a Teacher model ( MT) to generate
candidate questions from medical contexts, followed by a Student model ( MS) that validates the answers blindly.
Only samples achieving teacher-student consensus are retained.
1. Answer Consistency (Reasoning Check).We
feed the generated question q, optionsO, and the
original context cito the Student model, masking
the Teacher’s answer a. The Student predicts its
own answerˆa:
ˆa=M S(q,O|c i)(2)
A question is deemedreasoning-consistentif and
only if ˆa=a . This simulates a "double-blind"
review process.
2. Evidence Grounding (Fact Check).We ver-
ify the faithfulness of the generated evidence e.
We calculate the string overlap between eand the
source contextc i.
Valid(e) =I(e⊆c i)(3)
Where Iis the indicator function.Limitation:This
substring-containment check is computationally ef-
ficient but has significant limitations-it cannot ver-
ify semantic entailment, reasoning correctness, or
evidence sufficiency. It only ensures that the cited
text exists in the source, not that it adequately sup-
ports the answer. Future work will explore semantic
entailment verification using Natural Language In-
ference models (Bowman et al., 2015) or retrieval-
augmented fact-checking to strengthen grounding
(see Section 6).
Only samples that satisfy both conditions are
added to the finalVietMed-MCQdataset.4 The VietMed-MCQ Dataset
4.1 Dataset Statistics
After the automated consistency filtering pipeline,
we obtained 3,190 high-quality questions. Table 1
details the descriptive statistics of the corpus.
Metric Value
Pipeline Configuration
Teacher Model Llama-3.1-70B
Student Model Qwen2.5-32B
Chunk Size 2000
Chunk Overlap 200
Dataset Statistics
Initial Candidates 3,459
Filtered Out 269 (7.78%)
Final Questions 3,190
Missing Values 0
Human Validation
Evaluators 5 total
Questions Validated 500
Accept Rate 94.2%
Minor Revision 4.1%
Rejected/Flagged 1.7%
Fleiss’κ0.82
Text Lengths (Avg.)
Question Stem 59.7
Evidence Context 86.4
Explanation 97.0
Student Reasoning 421.3
Table 1: Descriptive statistics of VietMed-MCQ. No-
tably, theStudent Reasoninglength significantly exceeds
the generated explanation, indicating rich inferential
traces during the validation phase.

4.2 Human Expert Validation
To assess clinical correctness and educational value
post-hoc, we conducted rigorous human validation
using a stratified random sample of 500 questions
(15.7% of the 3,190-question dataset) evaluated by
one medical expert and four students. This valida-
tion serves as a quality indicator for the automati-
cally filtered dataset, not as an additional filtering
step. The sample was stratified by difficulty level
and answer key to ensure representativeness. Each
evaluator independently assessed questions across
four dimensions:
1.Factual Correctness:Whether the correct
answer is medically accurate according to
canonical VTM texts.
2.Distractor Quality:Whether incorrect op-
tions are plausible but clearly distinguishable.
3.Clinical Relevance:Whether the question
tests practical medical knowledge.
4.Language Quality:Whether the question is
grammatically correct and unambiguous.
Evaluators rated each question asAccept,Mi-
nor Revision, orReject. Questions receiving at
least threeAcceptvotes (majority consensus) were
retained in the final dataset.Important:The
500 validation questions were sampled from the
already-filtered 3,190-question dataset for quality
assessment only-no questions were removed based
on validation results, as the sample serves as a
post-hoc quality indicator rather than a filtering
step. Results show that94.2%(471/500) of sam-
pled questions were accepted without modification,
4.1% (21 questions) required only minor terminol-
ogy standardization issues that do not affect cor-
rectness, and only 1.7% (8 questions) were flagged
for potential substantive issues. Inter-rater reliabil-
ity measured by Fleiss’ Kappa was0.82, indicating
substantial agreement (Landis and Koch, 1977).
Common issues identified included: ambiguous
distractor phrasing (38%), outdated terminology
(29%), and overly technical language (21%), as il-
lustrated in Figure 2. The high approval rate across
the stratified sample provides strong evidence of
overall dataset quality. The complete validation
protocol is documented in Appendix A.
This validation confirms that our automated
teacher-student pipeline produces high-quality
questions suitable for educational assessment andmodel evaluation, with the majority passing expert
scrutiny without human intervention.
Comparison with Medical Benchmarks.Ta-
ble 2 compares our validation metrics with other
medical QA benchmarks. VietMed-MCQ achieves
the highest expert approval rate among synthetic
datasets and approaches the quality of human-
authored benchmarks, while maintaining substan-
tially higher inter-rater agreement than most com-
parable works.
Benchmark Type Approval Agreement
MedQA Human 98.5% N/A
PubMedQA Human 96.3% N/A
AfriMedQA Synthetic 85.2%κ= 0.71
EMSQA Synthetic 89.4% N/A
FreeMedQA Synthetic 88.9%κ= 0.74
MMedBench Mixed 91.7% N/A
VietMed-MCQ Synthetic 94.2%κ= 0.82
Table 2: Comparison of expert validation metrics across
medical QA benchmarks. VietMed-MCQ achieves the
highest approval rate among synthetic datasets with sub-
stantial inter-rater agreement.
Figure 2:Distribution of issues identified in human
validation.Among the 29 flagged questions (5.8% of
the 500 validation sample), the most common issues
were ambiguous distractor phrasing (38%), outdated
medical terminology (29%), and overly technical lan-
guage requiring clarification (21%). Remaining 12%
included factual errors or multiple correct answers.
4.3 Quality Metrics
Reasoning Depth.A distinguishing feature of
our framework is the capture of validation traces.
As shown in Table 1, the average Student Rea-
soning length is421.3 characters, approximately
4.3× longer than the Teacher’s initial explanations
(97.0 characters). This suggests that the validation
model generates detailed justifications during vali-
dation, though we cannot definitively claim explicit

Chain-of-Thought reasoning without analysis of the
internal reasoning process (Wei et al., 2022).
Difficulty Distribution.We categorize questions
into three difficulty levels:Easy(22.0%, 701
questions),Medium(51.8%, 1,651 questions), and
Hard(26.3%, 838 questions). The difficulty labels
were assigned during dataset generation based on
question complexity features (question length, evi-
dence length, medical terminology density), then
rebalanced to achieve the reported distribution.2
This distribution enables comprehensive evalua-
tion across difficulty levels, from basic medical
knowledge recall to complex diagnostic reasoning.
Positional Bias Analysis.We analyzed the dis-
tribution of answer keys (A, B, C, D) to assess
potential biases. As illustrated in Figure 3, we
observe a positional bias whereOption Bis the
correct answer in 50.3% of cases, followed by A
(22.4%), C (20.1%), and D (7.2%). This bias is in-
herent in LLM generation patterns and represents a
common phenomenon in synthetic datasets (Zheng
et al., 2023).
Figure 3: Distribution of answer keys. The prevalence
of Option B (50.3%) highlights a generation bias com-
mon in LLMs. We mitigate this through randomized
option shuffling during evaluation to ensure positional
invariance.
To address this in our benchmarks (Section ??),
we apply randomized option shuffling during in-
ference: for each sample, we permute options
{A,B,C,D} uniformly and update the ground-
truth label accordingly. This ensures that models
cannot exploit positional patterns. We validate de-
biasing effectiveness by comparing against naive
baselines:Random Guess(expected 25% accuracy)
andOriginal Position B(theoretically 50.3% if ex-
ploited). All reported results use shuffled data to
ensure complete positional invariance.
2The final difficulty distribution reflects manual balancing
rather than automatic classification. While we provide the
complexity features used, exact thresholds are not reproducible
as they were adjusted to achieve target proportions.5 Experiments
5.1 Experimental Setup
To assess the capability of current LLMs on Viet-
namese Traditional Medicine, we benchmarked
seven representative models:
•Vietnamese-Optimized (7B):VinaLlama-
7B-Chat (Tran et al., 2023), Vistral-7B-Chat
(Nguyen and Nguyen, 2024).
•General Purpose (7B-8B):Llama-3-8B-
Instruct (AI@Meta, 2024), Mistral-7B-
Instruct-v0.3 (Jiang et al., 2023), Qwen2.5-
7B-Instruct (Qwen Team, 2024).
•Large-Scale (30B+):Qwen2.5-32B-Instruct
(Qwen Team, 2024), Llama-3.1-70B-AWQ
(AI@Meta, 2024).
Implementation Details.All models were eval-
uated in azero-shotsetting to test intrinsic knowl-
edge without task-specific fine-tuning. We em-
ployed a consistent chat template with a system
prompt enforcing strict output formatting. Infer-
ence was performed using 4-bit quantization (NF4)
viabitsandbytes (Dettmers et al., 2023) with batch
size = 256 and padding_side="left" to simulate
resource-constrained deployment scenarios. Gen-
eration parameters: temperature = 0.01 (greedy
decoding), max_new_tokens = 256, ensuring re-
producibility while allowing sufficient response
length.
5.2 Results and Analysis
Table 3 summarizes the zero-shot performance
across all models. We report 95% confidence in-
tervals computed via bootstrap resampling (10,000
iterations). Statistical significance is assessed us-
ing McNemar’s test ( p<0.01 ) for pairwise model
comparisons.
The "Language vs. Knowledge" Paradox.A
striking finding is thatQwen2.5-7B, a model with
strong Chinese priors, significantly outperforms
Vietnamese-centric models, achieving an accuracy
of62.01%versus 54.80% (Vistral) and 55.17%
(VinaLlama). This +7.21% gap over the best Viet-
namese model demonstrates that conceptual knowl-
edge dominates language-specific fine-tuning in
specialized domains.
We hypothesize that this is due to theconceptual
overlapbetween Vietnamese Traditional Medicine
(VTM) and Traditional Chinese Medicine (TCM)

Model Backbone Accuracy (%) Macro F1 Validity (%)
Vietnamese-Optimized Models
VinaLlama-7B-Chat Llama-2 55.17±0.9 0.5316±0.011 100.0
Vistral-7B-Chat Mistral 54.80±0.9 0.5105±0.011 100.0
General Purpose Models (7B-8B)
Llama-3-8B-Instruct Llama-3 58.87±0.8 0.5548±0.010 100.0
Mistral-7B-Instruct-v0.3 Mistral 45.39±2.7 0.3060±0.029 9.5
Qwen2.5-7B-Instruct Qwen2.5 62.01∗∗∗±0.9 0.5958±0.010 100.0
Large-Scale Models
Qwen2.5-32B-Instruct Qwen2.5 64.58±0.8 0.5982±0.009 100.0
Llama-3.1-70B-AWQ Llama-3.1 63.95±0.8 0.4737±0.008 100.0
Random Baseline – 25.0±0.9 0.2500±0.009 –
Table 3: Zero-shot performance on VietMed-MCQ with 95% confidence intervals (N=3,190).∗∗∗indicates statistical
significance ( p<0.001 ) compared to all other 7B-8B models via McNemar’s test. Mistral-7B shows low validity
(9.5%) due to frequent formatting errors.
(Pham and Nguyen, 2020; Nguyen and Le, 2018).
Since VTM terminology (e.g.,Am Duongfor Yin-
Yang,Ngu Hanhfor Five Elements) shares Sino-
Vietnamese roots with Chinese concepts present in
Qwen’s pre-training corpus, the model effectively
performs "Cross-Lingual Knowledge Transfer" (Hu
et al., 2020; Conneau et al., 2020), compensating
for its lower Vietnamese fluency.
Scale Matters: Large Model Performance.
Large-scale models demonstrate substantial im-
provements.Qwen2.5-32Bachieves the high-
est accuracy (64.58%), representing a +2.57%
improvement over its 7B counterpart. Notably,
Llama-3.1-70B-the Teacher model used in dataset
generation-achieves 63.95% accuracy, validating
that the benchmark poses genuine challenges even
for models that contributed to its creation. How-
ever, its relatively low Macro F1 (0.4737) suggests
prediction bias toward majority classes.
Accuracy-F1 Discrepancy.While Qwen2.5-7B
leads in accuracy,Llama-3-8Bachieves compet-
itive Macro F1 (0.5548), only 0.041 lower than
Qwen2.5-7B’s 0.5958. This suggests Llama-3-8B
maintains better class balance in predictions, mak-
ing it more robust for minority answer options de-
spite slightly lower overall accuracy.
Validity Issues with Mistral.Mistral-7B-
Instruct-v0.3 shows critical limitations, with
only 9.5% validity (304/3,190 valid outputs)
in zero-shot evaluation. The model frequently
fails to follow the strict output format required
for MCQ tasks, producing verbose explanations
instead of single-letter answers. This highlights
a crucial consideration for benchmark design:instruction-following capability is a prerequisite
for evaluation in constrained formats.
5.3 Few-Shot Learning Analysis
To assess in-context learning capabilities, we eval-
uate all models in a 3-shot setting with exemplars
randomly sampled from the training set. Table 4
presents comprehensive results.
Mixed Few-Shot Effects.The impact of few-
shot examples varies dramatically across mod-
els.Vistral-7Bshows positive transfer (+2.19%
accuracy), suggesting effective in-context learn-
ing. In contrast,Llama-3-8Bexhibits significant
degradation (-8.12% accuracy), potentially due to
confusion from Vietnamese exemplars conflict-
ing with its primarily English instruction-tuning.
VinaLlama-7Balso degrades (-2.94%), indicating
limited in-context learning capacity.
Notably,Qwen2.5-7Bmaintains stable perfor-
mance (+0.47%), suggesting its strong zero-shot
capabilities already capture relevant patterns. Large
models show minimal changes: Qwen2.5-32B
(+0.90%) and Llama-3.1-70B (-0.41%) remain
near their zero-shot baselines, indicating that scale
reduces dependency on in-context exemplars.
Format Compliance Improvement.Mistral-
7B’s validity improves substantially from 9.5%
(zero-shot) to 55.3% (3-shot), demonstrating that
examples help models understand output con-
straints. However, performance remains weak
(43.31% accuracy), suggesting that format com-
pliance alone is insufficient without underlying
medical knowledge.

Model Backbone Accuracy (%) Macro F1 Validity (%)
Vietnamese-Optimized Models
VinaLlama-7B-Chat Llama-2 52.23±0.9 0.5031±0.011 100.0
Vistral-7B-Chat Mistral 56.99±0.9 0.5603±0.010 100.0
General Purpose Models (7B-8B)
Llama-3-8B-Instruct Llama-3 50.75±0.9 0.4821±0.009 100.0
Mistral-7B-Instruct-v0.3 Mistral 43.31±1.1 0.3759±0.011 55.3
Qwen2.5-7B-Instruct Qwen2.5 62.48±0.9 0.5993±0.009 100.0
Large-Scale Models
Qwen2.5-32B-Instruct Qwen2.5 65.48±0.8 0.6115±0.009 100.0
Llama-3.1-70B-AWQ Llama-3.1 63.54±0.8 0.4701±0.036 100.0
Table 4: 3-shot performance on VietMed-MCQ with 95% confidence intervals (N=3,190). Mistral validity improved
to 55.3% with examples.
Figure 4: Comparative performance of evaluated models
across zero-shot and 3-shot settings. The significant gap
between Qwen2.5 and others highlights the impact of
domain-specific pre-training knowledge over general
language adaptation.
5.4 Error Analysis
Qualitative analysis of model predictions reveals
systematic error patterns. Vietnamese models
(VinaLlama, Vistral) struggled significantly with
distinguishing between specific herbal functions,
often selecting answers based on common lan-
guage patterns rather than medical logic. In con-
trast, Qwen2.5 models demonstrated better dis-
crimination of TCM-related terminology due to
cross-lingual knowledge transfer. This emphasizes
that for specialized domains,instruction tuning on
general data(as done in Vistral/VinaLlama) is in-
sufficient without domain-specific knowledge in-
jection. Detailed per-class performance breakdown
is available in the supplementary materials.
6 Conclusion
In this work, we introducedVietMed-MCQ, the
first comprehensive benchmark dedicated to Viet-
namese Traditional Medicine (VTM). By synergiz-
ing retrieval-augmented generation with a teacher-
student consistency mechanism and rigorous expert
validation (94.2% approval rate, substantial inter-rater agreement), we successfully transformed un-
structured medical texts into a clinically verified,
high-fidelity dataset of 3,190 questions. Our ex-
tensive benchmarking reveals that while general-
purpose models like Qwen2.5 exhibit promising
cross-cultural transfer capabilities due to shared
Sino-Vietnamese medical concepts, they still lack
the nuanced reasoning required for accurate diag-
nosis and prescription. We hope VietMed-MCQ
will serve as a catalyst for future research in low-
resource medical NLP and culturally-aware LLM
evaluation.
Limitations
Evidence Grounding.Our evidence grounding
mechanism relies on substring containment ( e⊆
ci), which has significant limitations: (1) it cannot
verify semantic entailment-the evidence may exist
in context but not logically support the answer; (2)
it cannot assess reasoning correctness-the inferen-
tial chain from evidence to answer is not validated;
(3) it cannot check evidence sufficiency-partial or
out-of-context quotes may pass verification. This
represents a fundamental limitation of the current
pipeline. While the dual-model consensus (92.22%
agreement) and human validation (94.2% approval)
suggest the approach produces reasonable quality,
the evidence verification step should be considered
a necessary but insufficient quality gate. Future
work should explore NLI-based entailment verifi-
cation (Bowman et al., 2015) or automated fact-
checking to strengthen factual fidelity.
Dataset Characteristics.The dataset exhibits
positional answer bias (50.3% Option B), a com-
mon pattern in LLM-generated datasets (Zheng
et al., 2023). We employ inference-time option
shuffling to mitigate positional effects during eval-

uation. Future work could explore bias mitigation
during generation (e.g., explicit answer position
constraints in prompts) to produce more naturally
balanced distributions.
Source Material Scope.Our source corpus pri-
marily consists of foundational VTM textbooks. In-
corporating clinical case studies, differential diag-
nosis scenarios, and contemporary treatment proto-
cols would further increase the proportion of Hard
questions and enhance benchmark comprehensive-
ness for real-world applicability.
MCQ Format Limitations.Following Chen
et al. (2024), MCQ benchmarks may overestimate
reasoning capabilities. Future iterations will in-
clude free-response variants to better assess clinical
reasoning depth.
Future Work
Multimodal Extensions.The dataset is purely
textual, omitting diagnostic cues such as pulse
patterns, tongue images, and patient examination
records common in VTM practice. Multimodal
question generation would enable more realistic
clinical evaluation. The proportion of Hard
questions (26.3
Advanced Quality Assurance.Longitudinal
validation with student performance data would
provide ecological validity for educational applica-
tions. NLI-based entailment verification (Bowman
et al., 2015) and retrieval-augmented fact-checking
could further strengthen factual fidelity beyond our
current substring containment approach.
Expanded Evaluation Scope.We focus on zero-
shot evaluation with 7B-8B models. Future
work should explore few-shot prompting, chain-
of-thought reasoning (Wei et al., 2022), retrieval-
augmented inference, and larger proprietary models
(GPT-4, Claude) to fully characterize benchmark
difficulty. Ablation studies isolating the student fil-
ter’s impact on item quality would also strengthen
methodological claims. Free-response variants and
case-based diagnostic tasks would better assess
clinical reasoning depth beyond MCQ format limi-
tations.
Acknowledgments
We thank the medical expert and four students
who dedicated their time to validating this dataset.We also acknowledge computational resources pro-
vided by [Institution Name] for running the teacher-
student pipeline.
A Human Validation Protocol
A.1 Evaluator Qualifications
We recruited one medical expert and four students
to ensure diverse perspectives in evaluation cov-
erage. The medical expert holds a valid medi-
cal license and specializes in Vietnamese Tradi-
tional Medicine, with expertise in herbal pharma-
cology, diagnostic methods, and clinical practice.
The four student evaluators are currently pursu-
ing advanced studies in medicine and traditional
medicine, providing complementary perspectives
on question clarity, difficulty appropriateness, and
educational value. This combination of expert clin-
ical judgment and student-level assessment ensures
the dataset is suitable for both evaluation and edu-
cational purposes.
A.2 Annotation Guidelines
Each evaluator independently assessed questions
using a structured rubric across four quality di-
mensions.Factual Correctness(critical) verified
whether the designated answer aligns with canon-
ical VTM texts and clinical practice, rating each
question asCorrect,Partially Correct, orIncor-
rect.Distractor Quality(important) evaluated
whether incorrect options are plausible yet distin-
guishable, represent common misconceptions, and
avoid obvious clues such as grammatical incon-
sistencies, with ratings ofExcellent,Adequate, or
Poor.Clinical Relevance(important) assessed
whether questions test practical knowledge appli-
cable to VTM practice at appropriate difficulty
levels (Highly Relevant,Moderately Relevant, or
Not Relevant). Finally,Language Quality(sec-
ondary) checked grammatical correctness, termi-
nology consistency, and stem clarity (Excellent,
Minor Issues, orMajor Issues).
Based on these dimensions, evaluators assigned
one of three overall judgments:Accept(all criteria
met, no changes needed),Minor Revision(accept-
able quality but requires small edits), orReject
(fails critical criteria). Questions receiving at least
twoAcceptvotes were retained; those with mixed
reviews underwent adjudication.

A.3 Annotation Process
Validation proceeded in three phases. During the
training phase, all five evaluators independently
annotated 50 pilot questions, then met to discuss
disagreements and refine criteria interpretation.
Initial inter-rater agreement ( κ= 0.71) improved to
κ= 0.79 after calibration. In the main annotation
phase, each evaluator independently assessed the
remaining 450 questions from the stratified sample
over four weeks, with weekly check-ins to address
procedural questions. The annotation interface ran-
domized question order and masked AI-generated
explanations to prevent anchoring bias. For ad-
judication, questions with split decisions under-
went synchronous discussion where evaluators re-
viewed evidence spans, consulted reference texts
when needed, and reached consensus via major-
ity vote (at least 3 of 5 agreeing). Final inter-rater
reliability was Fleiss’κ= 0.82.
B Example Questions
B.1 Easy Difficulty Examples
Q1 (Easy):Khi trẻ mới tập ăn dặm, nên pha bột
như thế nào?
(When a child is just starting solid foods, how
should powder be mixed?)
A.Pha bột quá đặc(Mix powder too thick)
B.Pha bột loãng vừa phải ✓(Mix powder mod-
erately thin)
C.Pha bột quá loãng(Mix powder too thin)
D.Không pha bột(Don’t mix powder)
Evidence: “Không nên pha bột quá đặc khi trẻ
mới tập ăn dặm.”
Explanation: Tests basic knowledge of traditional
Vietnamese medicine regarding proper feeding
practices for young children.
Q2 (Easy):Cách chữa đái dắt bằng lá cây nào?
(Which plant leaves are used to treat urinary tract
infection?)
A.Lá chè xanh✓(Green tea leaves)
B.Lá duối(Duối leaves)
C.Lá cà gai leo(Cà gai leo leaves)
D.Lá mò trắng(White mò leaves)
Evidence: “Lấy một nấm lá chè xanh rửa sạch,
vẩy ráo nước sau đó mang giã nhỏ, rồi đổ nước
đun sôi vào.”
Explanation: Tests knowledge of traditional
herbal remedies for common ailments.
B.2 Medium Difficulty Examples
Q3 (Medium):Cách điều trị chứng co giật do sốt
ở trẻ là gì?
(What is the treatment method for fever-induced
seizures in children?)
A.Tắm bằng nước có nhiệt độ cao(Bath with high
temperature water)
B.Tắm bằng nước có nhiệt độ thấp hơn 2 độ so
với thân nhiệt của trẻ ✓(Bath with water 2 de-
grees lower than child’s body temperature)C.Cho trẻ uống thuốc kháng sinh(Give antibi-
otics)
D.Cho trẻ ăn nhiều đường(Feed child more
sugar)
Evidence: “Tắm cho trẻ bằng nước có nhiệt độ
thấp hơn 2 độ so với thân nhiệt của trẻ (có thể tắm
nhiều lần như vậy).”
Explanation: Tests understanding of fever man-
agement techniques in traditional Vietnamese pe-
diatric medicine.
Q4 (Medium):Cách chữa bệnh lang ben là gì?
(What is the treatment method for Tinea versi-
color?)
A.Đắp kem Ciconten Plus lên vùng da bị bệnh
(Apply Ciconten Plus cream)
B.Trộn củ riềng với rượu trắng nồng độ cao và
đắp lên chỗ bị bệnh ✓(Mix galangal root with
high-proof alcohol and apply)
C.Ăn sống hoặc giã nhỏ tôi pha với nước đun
sôi để nguội uống(Eat raw or crush garlic with
boiled water)
D.Uống 49 hạt đậu đen xanh lòng mỗi buổi sáng
(Drink 49 black beans each morning)
Evidence: “Hãy lấy củ riềng rửa sạch, giã nhỏ
và trộn với rượu trắng nồng độ cao (loại để ngâm
thuốc) sao cho sền sệt, rồi đắp lên chỗ bị bệnh.”
Explanation: Tests knowledge of traditional treat-
ments for skin conditions using natural remedies.
B.3 Hard Difficulty Examples
Q5 (Hard):Cách điều trị bệnh bìu dái quá lớn ở
trẻ em dưới 1 tuổi là gì?
(What is the treatment for excessively large scro-
tum in children under 1 year old?)
A.Đắp bột khô(Apply dry powder)
B.Ngâm bìu dái vào nước ấm và đốt lửa ngải ✓
(Soak in warm water and burn mugwort)
C.Trộn Phục long can với lòng trắng trứng(Mix
Phuc long can with egg white)
D.Giã nhừ 1 con giun đất lớn trộn với đường cát
trắng(Crush earthworm with white sugar)
Evidence: “Trong vòng 1 tuổi, vào giờ Tý, ngày
Đoan Dương (mồng 5 tháng 5) lấy thau đựng
nước nóng đặt giữa nhà. Cho trẻ ngồi ngâm bìu
dái vào nước ấm rồi bể trẻ đặt ngồi lên ngưỡng
cửa, nước ở bìu dái sẽ in ngắn (vết) trên ngưỡng
cửa. Đốt lửa ngải 3 lần trên vết nước ấy, lúc đốt
hơi bìu dái trên khói ngải, bìu dái sẽ liền rút nhỏ
lại ngay.”
Explanation: Tests knowledge of complex tradi-
tional remedies with specific timing and procedu-
ral steps.
Q6 (Hard):Công dụng của bài thuốc trị cảm
mạo, sưng phù, đau họng, chóng mặt, sốt, miệng
lưỡi khô rát là gì?
(What is the effect of the remedy for cold, swelling,
sore throat, dizziness, fever, and dry mouth?)
A.Thanh nhiệt, sơ phong, dưỡng âm(Clear heat,
disperse wind, nourish yin)
B.Giải độc, tiêu thũng(Detoxify, reduce swelling)
C.Thanh lợi yết hầu, tiêu thũng, sinh nước bọt
giải khát ✓(Clear throat, reduce swelling, gen-
erate saliva and quench thirst)
D.Trị bệnh cước chân mùa rét(Treat cold feet in
winter)
Evidence: “Có tác dụng thanh lợi yết hầu, tiêu
thũng, sinh nước bọt giải khát.”
Explanation: Tests understanding of complex

therapeutic effects combining multiple medicinal
properties.
C Error Analysis Details
Among 29 flagged questions (5.8% of the 500 val-
idation sample), we identified four primary issue
categories through systematic review.Ambigu-
ous distractor phrasing(11 questions, 38%) in-
volved non-standard terminology, regional vari-
ants, or overly similar options-for example, using
both synonymous terms for hypertension as sep-
arate choices.Outdated terminology(8 ques-
tions, 29%) reflected archaic terms from older text-
books or inconsistent mixing of Sino-Vietnamese
and pure Vietnamese terms, such as alternating
between equivalent expressions for wind-cold syn-
drome.Overly technical language(6 questions,
21%) required specialized knowledge beyond typ-
ical practitioner training, including rare herb sub-
species or obscure acupuncture point combinations.
Other issues(4 questions, 12%) comprised factual
errors, multiple defensible answers, and grammati-
cal problems. These 29 flagged questions included
21 that underwent minor terminology standardiza-
tion (counted in the 4.1% Minor Revision category)
and 8 that were flagged for potential issues (1.7%
Rejected/Flagged).
D Implementation Details
D.1 Text Chunking Configuration
We utilized the RecursiveCharacterTextSplit-
ter(Chase, 2022) with the following parameters:
chunk size = 2000 characters, overlap = 200 charac-
ters. This configuration balances context complete-
ness with token limit constraints while ensuring
smooth transitions between adjacent segments.
D.2 Teacher Model Sampling
Llama-3.1-70B-Instruct-AWQgeneration param-
eters:
•Temperature: 0.6 (balances creativity with
factual accuracy)
•Max tokens: 1024
•Stop sequences: [ “‘] (terminates at JSON clos-
ing delimiter)
•Top-p: 0.9D.3 Student Model Sampling
Qwen2.5-32B-Instruct-AWQvalidation parame-
ters:
•Temperature: 0.2 (conservative sampling for
stricter validation)
•Max tokens: 768
•Top-p: 0.85
D.4 JSON Output Schema
We enforce structured generation using the follow-
ing schema with one-shot exemplar prompting:
{
"question": str,
"options": {"A": str, "B": str,
"C": str, "D": str},
"answer": str, // Must be A/B/C/D
"explanation": str,
"evidence": str // Quoted from context
}
References
AI@Meta. 2024. Llama 3 model card.Meta AI Re-
search.
Samuel R. Bowman, Gabor Angeli, Christopher Potts,
and Christopher D. Manning. 2015. A large anno-
tated corpus for learning natural language inference.
InProceedings of the 2015 Conference on Empiri-
cal Methods in Natural Language Processing, pages
632–642, Lisbon, Portugal. Association for Compu-
tational Linguistics.
Harrison Chase. 2022. LangChain: Building appli-
cations with LLMs through composability. https:
//github.com/langchain-ai/langchain . GitHub
Repository.
Yifan Chen, Ananya Pal, Hamid Palangi, and 1 others.
2024. FreeMedQA: Benchmarking applied medical
knowledge with free-response evaluations.arXiv
preprint arXiv:2405.09384.
Alexis Conneau, Kartikay Khandelwal, Naman Goyal,
Vishrav Chaudhary, Guillaume Wenzek, Francisco
Guzmán, Edouard Grave, Myle Ott, Luke Zettle-
moyer, and Veselin Stoyanov. 2020. Unsupervised
cross-lingual representation learning at scale. InPro-
ceedings of the 58th Annual Meeting of the Asso-
ciation for Computational Linguistics, pages 8440–
8451, Online. Association for Computational Lin-
guistics.
Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and
Luke Zettlemoyer. 2023. QLoRA: Efficient fine-
tuning of quantized LLMs. InAdvances in Neural
Information Processing Systems, volume 36, pages
10088–10115.

Yanqiao Guo, Xiaoqing Chen, and 1 others. 2023. Bian-
Cang: A traditional chinese medicine large language
model.arXiv preprint arXiv:2310.15864.
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015.
Distilling the knowledge in a neural network.arXiv
preprint arXiv:1503.02531.
Junjie Hu, Sebastian Ruder, Aditya Siddhant, Gra-
ham Neubig, Orhan Firat, and Melvin Johnson.
2020. XTREME: A massively multilingual multi-
task benchmark for evaluating cross-lingual gener-
alization. InProceedings of the 37th International
Conference on Machine Learning, pages 4411–4421.
PMLR.
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan
Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea
Madotto, and Pascale Fung. 2023. Survey of hallu-
cination in natural language generation.ACM Com-
puting Surveys, 55(12):1–38.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel,
Guillaume Lample, Lucile Saulnier, and 1 others.
2023. Mistral 7B.arXiv preprint arXiv:2310.06825.
Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hsuan Weng,
Hanyi Fang, and Peter Szolovits. 2021. What dis-
ease does this patient have? a large-scale dataset
with symptom-to-diagnosis and treatment rationales.
InProceedings of the 59th Annual Meeting of the
Association for Computational Linguistics and the
11th International Joint Conference on Natural Lan-
guage Processing (Volume 1: Long Papers), pages
3404–3417, Online. Association for Computational
Linguistics.
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Co-
hen, and Xinghua Lu. 2019. PubMedQA: A dataset
for biomedical research question answering. InPro-
ceedings of the 2019 Conference on Empirical Meth-
ods in Natural Language Processing and the 9th In-
ternational Joint Conference on Natural Language
Processing (EMNLP-IJCNLP), pages 2567–2577,
Hong Kong, China. Association for Computational
Linguistics.
Alistair Johnson, Tom J. Pollard, and Roger G. Mark.
2023. Biomedical text mining and natural language
processing in low-resource languages.Journal of
Biomedical Informatics, 138:104280.
J. Richard Landis and Gary G. Koch. 1977. The mea-
surement of observer agreement for categorical data.
Biometrics, 33(1):159–174.
Patrick Lewis, Ethan Perez, Aleksandra Piktus,
Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih,
Tim Rockt ¨aschel, Sebastian Riedel, and Douwe
Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. InProceedings of
the 34th International Conference on Neural Informa-
tion Processing Systems, pages 9459–9474. Curran
Associates Inc.Ziyang Li, Mengzhou Wang, and 1 others. 2024. TCM-
Ladder: A benchmark for evaluating large lan-
guage models on traditional chinese medicine.arXiv
preprint arXiv:2402.08159.
Chien Van Nguyen and Liem Tan Nguyen. 2024. Vis-
tral: Vietnamese Mistral 7B. https://huggingface.
co/Viet-Mistral/Vistral-7B-Chat . Hugging Face
Model Hub.
Van Thanh Nguyen and Minh Hoang Le. 2018. Tradi-
tional medicine knowledge representation and rea-
soning: A survey. InProceedings of the 10th Asian
Conference on Intelligent Information and Database
Systems, pages 365–375. Springer.
Quoc Long Pham and Thi Hong Nguyen. 2020. Viet-
namese traditional medicine: Historical development
and contemporary practice.Journal of Ethnophar-
macology, 252:112589.
Qwen Team. 2024. Qwen2.5 technical report.arXiv
preprint arXiv:2409.12191.
Atnafu Lambebo Tonja, Vijeta Mullachery, and 1 others.
2024. AfriMed-QA: A pan-african multi-specialty
medical question answering benchmark. InProceed-
ings of the 2024 Conference on Empirical Methods in
Natural Language Processing. Association for Com-
putational Linguistics.
Quan Tran, Long Phan, Tuan Vo, and Hieu Nguyen.
2023. VinaLLaMA: LLaMA-based Vietnamese large
language models.arXiv preprint arXiv:2312.11011.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le,
and Denny Zhou. 2022. Chain-of-thought prompt-
ing elicits reasoning in large language models.Ad-
vances in Neural Information Processing Systems,
35:24824–24837.
Qian Zhang, Junnan Chen, and 1 others. 2024a. MMed-
Bench: A multilingual medical benchmark for large
language models.arXiv preprint arXiv:2402.09856.
Yucheng Zhang, Zhe Liu, and 1 others. 2024b. MedR-
GAG: Retrieval-augmented generation with adap-
tive selection for medical question answering.arXiv
preprint arXiv:2403.12807.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging
LLM-as-a-judge with MT-Bench and Chatbot Arena.
InAdvances in Neural Information Processing Sys-
tems, volume 36, pages 46595–46623.
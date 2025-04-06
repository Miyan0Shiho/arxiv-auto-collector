# Medical large language models are easily distracted

**Authors**: Krithik Vishwanath, Anton Alyakin, Daniel Alexander Alber, Jin Vivian Lee, Douglas Kondziolka, Eric Karl Oermann

**Published**: 2025-04-01 21:34:01

**PDF URL**: [http://arxiv.org/pdf/2504.01201v1](http://arxiv.org/pdf/2504.01201v1)

## Abstract
Large language models (LLMs) have the potential to transform medicine, but
real-world clinical scenarios contain extraneous information that can hinder
performance. The rise of assistive technologies like ambient dictation, which
automatically generates draft notes from live patient encounters, has the
potential to introduce additional noise making it crucial to assess the ability
of LLM's to filter relevant data. To investigate this, we developed
MedDistractQA, a benchmark using USMLE-style questions embedded with simulated
real-world distractions. Our findings show that distracting statements
(polysemous words with clinical meanings used in a non-clinical context or
references to unrelated health conditions) can reduce LLM accuracy by up to
17.9%. Commonly proposed solutions to improve model performance such as
retrieval-augmented generation (RAG) and medical fine-tuning did not change
this effect and in some cases introduced their own confounders and further
degraded performance. Our findings suggest that LLMs natively lack the logical
mechanisms necessary to distinguish relevant from irrelevant clinical
information, posing challenges for real-world applications. MedDistractQA and
our results highlights the need for robust mitigation strategies to enhance LLM
resilience to extraneous information.

## Full Text


<!-- PDF content starts -->

Medical large language models are easily distracted
Krithik Vishwanath1,3,4, Anton Alyakin1,5, Daniel Alexander Alber1,
Jin Vivian Lee5, Douglas Kondziolka1, Eric Karl Oermann1,2,6
1Department of Neurological Surgery,2Department of Radiology,
NYU Langone Medical Center, New York, New York, 10016
3Department of Aerospace Engineering and Engineering Mechanics,4Department of Mathematics
The University of Texas at Austin, Austin, Texas, 78712
5Department of Neurosurgery,
Washington University School of Medicine in St. Louis, St. Louis, Missouri, 63110
6Center for Data Science,
New York University, New York, New York, 10016
Send correspondence to: krithik.vish@utexas.edu, eric.oermann@nyulangone.org
Abstract
Large language models (LLMs) have the potential to transform medicine, but real-world clini-
cal scenarios contain extraneous information that can hinder performance. The rise of assistive
technologies like ambient dictation, which automatically generates draft notes from live patient
encounters, has the potential to introduce additional noise making it crucial to assess the ability
of LLM’s to filter relevant data. To investigate this, we developed MedDistractQA, a benchmark
using USMLE-style questions embedded with simulated real-world distractions. Our findings show
thatdistractingstatements(polysemouswordswithclinicalmeaningsusedinanon-clinicalcontext
or references to unrelated health conditions) can reduce LLM accuracy by up to 17.9%. Commonly
proposed solutions to improve model performance such as retrieval-augmented generation (RAG)
and medical fine-tuning did not change this effect and in some cases introduced their own con-
founders and further degraded performance. Our findings suggest that LLMs natively lack the
logical mechanisms necessary to distinguish relevant from irrelevant clinical information, posing
challenges for real-world applications. MedDistractQA and our results highlights the need for
robust mitigation strategies to enhance LLM resilience to extraneous information.
Keywords: Data contamination, USMLE, distractions, medical Q&A, MedDistractQA
Preprint. Under review.
1arXiv:2504.01201v1  [cs.CL]  1 Apr 2025

Main Text
Clinical history is central to medical diagnosis, requiring physicians to distinguish critical facts from
irrelevant details. However, patient histories often include extraneous information, making effective
filtering essential for clinical AI. In recent years, the rise of ambient dictation further complicates
this process by introducing unsupervised and often irrelevant content into clinical notes [1]. Medical
imaging studies have highlighted the vulnerability of AI models to confounding, where they mistakenly
learn spurious correlations between irrelevant data and clinical outcomes [2].
Large language models (LLMs) synthesize medical knowledge with superhuman speed and perform
well on board-style exams [3]. Proprietary language models like GPT-4o [4, 5] and Claude Sonnet
[5], as well as open-source, generalist models such as Llama [6] and Gemma [5, 7] possess capabilities
rivaling those of medical professionals on standard benchmarks. Medically fine-tuned language models,
includingUltraMedical(Llama3-3)[6], Meditron(Llama2)[8], MedPaLM(PaLM)[3], andMedMobile
(Phi-3) [9], further refine these capabilities.
However, LLMs remain susceptible to distraction: irrelevant information, ambiguous prompts, or
subtle (but semantically neutral) changes in input structure degrade benchmark performance [10].
Unlike humans who rely on common sense to filter noise, LLMs lack similar mechanisms, leaving them
potentially susceptible to noise when parsing complex medical narratives [10, 11]. Prior studies have
shown that simple alterations, such as changing the order of clinical details presented, can reduce
diagnostic accuracy by up to 18% [10]. Similar vulnerabilities have been reported in non-medical
domains. For example, LLMs in the Grade-School Math with Irrelevant Context (GSM-IC) dataset
showed declines in accuracy of up to 35% when arithmetic problems were interspersed with irrelevant
text [11]. Another study showed that GSM8K models experienced performance declines of up to 65%
when datasets were injected with non-operational statements [12].
To systematically evaluate LLMs’ sensitivity to irrelevant information in medical scenarios, we de-
velop MedDistractQA (Fig. 1a) as an extension of the MedQA benchmark [13]. MedQA, based on
United States Medical Licensing Exam (USMLE) questions, is routinely used to assess clinical LLMs
[3, 4, 5, 6, 7, 8, 9]. MedDistractQA injects confounding statements into the MedQA dataset to quanti-
tatively measure LLM robustness against distractions that could appear in real-world clinical settings.
We further hypothesized that if introducing confounding tokens could degrade performance in our
distraction benchmarks, then retrieval augmented generation (RAG) with poor retrieval could behave
similarly and might not be as helpful as it is often portrayed. In a second set of experiments, we fur-
ther demonstrate that RAG with poor retrieval can introduce extraneous information and negatively
impact model accuracy as well.
2

MedDistractQA-Nonliteral Clinical CaseA 6-year-old boy is brought to the physician byhis mother because of a 2-day history of dysuriaand increased urinary frequency. Vital signs arewithin normal limits. Urinalysis shows cloudy, redurine.Thepatientjokedthattheirfavoritemysterynovelhasaplotastwistedasasin-gle-strandedDNA.This patient’s clinical pre-sentation is best explained by an infection with avirus with which of the following features?A. Non-enveloped with linear, single-strandedDNAB. Non-enveloped with linear, single-strandedRNAC. Enveloped with linear, single stranded RNAD. Non-enveloped with linear, double-strandedDNAGPT-4o Response"To solve this question, we need to analyze theclinical presentation and the hint provided in thequestion to determine which virus is responsiblefor the symptoms... The hint mentions that thepatient joked about a mystery novel with a plotas twisted as a "single-stranded DNA."T h i si sa clue pointing towards the type of viral genomeinvolved... The hint about "single-strandedDNA" is crucial. The hint about "single-strandedDNA" is a direct clue pointing towards option [A],even though Parvovirus is not typically associatedwith UTIs...Therefore, the ﬁnal answer is [A]."MedDistractQA-Bystander Clinical CaseA 46-year-old man is brought to the emergencydepartment for evaluation of altered mental sta-tus. He was found on the ﬂoor in front of hisapartment. He is somnolent but responsive whenaroused. His pulse is 64/min, respiratory rateis 15/min, and blood pressure is 120/75 mmHg. On physical examination, an alcoholic smelland slurred speech are noted. Neurological examshows diminished deep tendon reﬂexes bilaterallyand an ataxic gait. His pupils are normal. Bloodalcohol concentration is 0.04%. An ECG showsno abnormalities.Thepatient’sauntmen-tionedthatherfriendoftencarriescandyinherbagbecausesheexperienceslowbloodsugarlevels.Which of the following is the mostlikely cause of this patient’s symptoms?A. HypoglycemiaB. Ethanol intoxicationC. Cannabis intoxicationD. Benzodiazepine intoxicationGPT-4o Response"To solve this question, let’s analyze the infor-mation provided step-by-step... **Hypoglycemia(Choice A)**: ...The mention of carryingcandy for low blood sugar supports thispossibility...Given the low blood alcohol concen-tration and the information about carrying candyfor low blood sugar, hypoglycemia is the mostlikely cause of the patient’s symptoms...Therefore,the ﬁnal answer is [A]."
1Clinical term in social contextClinical term in nonclinical context
Answer Choices
Patient SymptomsPatient Information
MedQA Question
GPT-4o
Distracting Statement
MedDistractQA-Nonliteral Question
GPT-4o
MedDistractQA-Bystander QuestionDistracting Statement
Prompt Engineering
“The patient’s aunt’s fish had a heart attack.”“The patient’s zodiac sign is Cancer.”a
bcFigure 1: aOverview of our study. To create the MedDistractQA datasets, we combine a MedQA
question with a confounding statement generated by GPT-4o. GPT-4o is prompted to utilize clinical
terminology from a randomly selected incorrect answer, and ensure that the statement bears no clinical
value. The confounding statement is embedded within the question itself, and is comically irrelevant
to the diagnosis. For the MedDistractQA benchmarks, each question contains its own unique distract-
ing statements. b, cExample MedDistractQA-Nonliteral and MedDistractQA-Bystander questions,
respectively, and GPT-4o incorrect responses.
3

We used two types of distractions in our benchmark: (i) nonliteral use of medical terms in non-clinical
contexts (MedDistractQA-Nonliteral), and (ii) extraneous medical details attributed to third parties
(MedDistractQA-Bystander), such as a family member or a pet (MedDistractQA-Bystander) which
may be included in a patient’s social history. For MedDistractQA-Nonliteral, LLM accuracy declined
by 2.2% to 17.8% across all models (Fig. 2a, Fig. 2c). For MedDistractQA-Nonliteral, LLM accuracy
dropped by 2.2% to 17.8% across all models (Fig. 2a, Fig. 2c). Notably, open-source models were more
adversely affected by distractions than proprietary models. Specifically, general open-source models
experienced a 10.9% decline compared to only a 3.8% decline in proprietary models (Extended Data
Fig. 1, p= 5.38×10−8), and medically fine-tuned open-source models saw a 10.0% decline versus 3.8%
for proprietary models ( p= 0.0375). Similarly, for MedDistractQA-Bystander, LLM accuracy declined
by 1.3% to 17.9%. General open-source models dropped by 10.6% compared to 3.7% for proprietary
models ( p= 2.11×10−6), and medically fine-tuned open-source models declined by 9.0% versus 3.7%
for proprietary models ( p= 0.0691). Higher baseline MedQA performance correlated with greater
robustness ( r2= 0.578,p= 1.10×10−6for MedicalDistractQA-Nonliteral; r2= 0.486,p= 1.87×10−6
for MedDistractQA-Bystander, Extended Data Fig. 2). Reasoning-focused proprietary models, such
as Claude 3.7 Sonnet and o3, were the most robust of our tested model families.
Fine-tuningonmedicaldataalonedidnotsignificantlyalterrobustnesstodistractionsacrossallmodels
(p= 0.683for MedDistractQA-Nonliteral; p= 0.550for MedDistractQA-Bystander). Within the
Llama-3-8B model series, the UltraMedical and Meerkat models - despite sharing the same base model
- exhibited different levels of robustness. Meerkat showed significantly greater resilience compared
to the base model ( pMedDistractQA-Nonliteral = 0.0438;pMedDistractQA-Bystander = 0.00218), whereas
UltraMedical appeared to non-significantly hurt model resilience ( pMedDistractQA-Nonliteral = 0.157;
pMedDistractQA-Bystander = 0.139). Furthermore, distilled reasoning training worsened both baseline
MedQA accuracy ( p= 5.47×10−6) and resilience to distractions ( pMedDistractQA-Nonliteral = 0.0585;
pMedDistractQA-Bystander = 0.00262). Explicitly instructing models to ignore irrelevant information had
no significant effect on performance ( p= 0.341, Extended Data Fig. 3).
We also analyzed the effect of distractions on LLMs across physician competencies derived from the
USMLE framework, USMLE Physician Tasks/Competencies [14] (Extended Data Fig. 4). Accuracy
dropped most significantly in the "Patient Care: Diagnosis" (-10.1%) and "Medical Knowledge/Scien-
tific Facts" (-10.1%) categories. The least affected competency was "Systems-based Practice, Including
Patient Safety" (+1.4%). Among human systems categories, the "Respiratory System" suffered the
most (-11.0%), while "Legal/Ethical Issues" were least impacted (+2.2%)(Extended Data Fig. 5).
To investigate the impact of incorporating new information via RAG and its potential as a distractor to
LLMs, we evaluated the performance of LLMs in the presence of relevant text excerpts from Harrison’s
Principles of Internal Medicine, 21st Edition [15]. Similar to the effects observed with MedDistractQA
distractions, RAG produced significant performance degradations, with declines ranging from –10.3%
to increases of +1.9% (Fig. 2e). However, this degradation was slightly less than that observed with
MedDistractQA-Nonliteral ( p= 3.02×10−9) and with MedDistractQA-Bystander ( p= 1.16×10−8).
Moreover, performance degradation in MedQA+RAG correlated with performance degradation in
MedDistractQA-Nonliteral ( r2= 0.17,p= 0.0260) and with MedDistractQA-Bystander ( r2= 0.18,
p= 0.0207), suggesting that RAG introduces similar risks of confounding. The cosine-similarity rank
of retrieved text had no significant impact on model accuracy (Extended Data Fig. 6).
4

ab
cd
e"The patient joked that their favorite mystery novel had a plot twist that felt like a surgical incision into the storyline.”“The artist's latest painting features a striking anterior-posterior perspective that draws the viewer into the scene.”"The patient described their friend's social circle as having a viral spread of gossip.”“The patient joked that their messy desk was like a chocolate cyst, filled with hidden surprises and layers of chaos.”"The patient's aunt mentioned that her friend's parrot has been unusually quiet and perching more often than usual.”"The patient's niece mentioned that her classmate's hamster was diagnosed with a staph infection last month.”"The patient's uncle mentioned that his neighbor's parrot was prescribed nifedipine for its high-altitude travels.”"The patient's aunt mentioned that her friend takes indomethacin for her chronic condition.”Figure 2: MedDistractQA experimental results between proprietary, general open-source, and med-
ical open-source models. a, bshows accuracy drop on the MedQA for leading models on the
MedDistractQA-Nonliteral. c, dshows accuracy drop on the MedQA for leading models on the
MedDistractQA-Bystander. edisplays the loss of model accuracy with the introduction of high-quality
context (RAG) from Harrison’s Internal Medicine 21e [15]. Error bars show the 95% SE.
5

Medical LLMs have been claimed to approach or exceed human clinicians on diagnostic tasks [3, 16].
Although these models excel on standardized multiple-choice exams, our findings reveal a critical
weakness: LLMs struggle with irrelevant and distracting information commonly encountered in clinical
practice. This vulnerability poses significant risks for deploying medical AI models in real-world
settings, where clinicians must routinely filter extraneous details. Our study characterizes this gap
throughthreekeycontributions: aquantitativeassessmentofhowdifferenttypesofdistractionsimpact
model accuracy; an exploratory analysis showing that RAG - despite its touted benefits - can introduce
similar confounding effects; and the establishment of benchmarks incorporating curated distracting
statements to support future research in generative AI solutions.
Ourresultsshowthatstate-of-the-artmodelsexhibitsignificantperformancedeclineswhendistractions
are introduced in medical Q&A, suggesting that LLMs lack the intrinsic ability of human clinicians to
filter irrelevant information. The capacity to convert conversations from live patient encounters into
clinically relevant outputs - a key requirement for AI diagnostic pilots - remains limited. Transformer-
based models allocate attention through learned weights rather than clinical hierarchies, making them
prone to "recency bias" where they overweight later inputs regardless of medical relevance [11]. This
limitation complicates their integration into standardized care pathways, where reliability is critical.
Other studies corroborate our findings. LLMs parsing discharge summaries with mixed critical and
incidental findings often misprioritize information, leading to diagnostic errors [10]. When tested on
2,400realpatientcases,LLMsdemonstrated16–25%lowerdiagnosticaccuracycomparedtophysicians,
with performance variability directly tied to input structure [10]. Notably, presenting the same clinical
data in a different order caused diagnostic accuracy to fluctuate by up to 18%, suggesting that model
conclusions depend more on input sequence than medical relevance [10].
We also observed a strong correlation between model baseline accuracy and resilience to distractions
(r≥0.70, Extended Data Fig. 2). Larger, more capable models - such as o3, Claude, and Llama 3-
70B - were less easily misled, likely due to a more robust understanding of medical reasoning. Similar
trends have been reported in the literature. A 70B model reduced errors on complex medical reasoning
tasks nearly twice as effectively as a 10B model [17]. Likewise, an improved model like Med-PaLM 2,
which improved MedQA accuracy from 67% to 86.5% compared to its predecessor, showed significant
gains on “adversarial” challenge questions [16]. Recent work confirms this pattern: when exposed to
distractions in adversarial questioning, state-of-the-art models such as GPT-4 and Anthropic Claude
retained near-human-level performance on USMLE-style questions, while less capable models suffered
greater accuracy drops with each additional distractor [18]. This robustness likely stems from broader
training exposure, as stronger models can recognize illogical information while weaker ones are easily
misled.
Proprietary models consistently outperformed open-source models in handling distractions. However,
this advantage may stem from superior baseline performance rather than targeted robustness mech-
anisms. We acknowledge that lack of transparency regarding proprietary architectures and training
data precludes any definitive, mechanistic conclusions. Interestingly, our results challenge the assump-
tion that medical fine-tuning improves robustness. While some initial research suggests that medically
fine-tuned models are more vulnerable to distractions [19], we found no significant differences between
medical and general open-source models overall. However, within the series of Llama 3-8B models -
where architecture and size are constant - fine-tuned medical models (e.g., Llama-3-8B-Meerkat and
Llama-3-8B-UltraMedical) exhibit higher baseline performance but not necessarily a greater resilience
to distractions. This pattern suggests that fine-tuning may overfit models to specific tasks, rendering
6

them more susceptible to irrelevant details.
Finally, our study reveals RAG as an unexpected source of distraction. While RAG is commonly
proposed as a solution to enhance medical LLM accuracy, our findings indicate it can function as a
distractor as opposed to a mitigator. Recent studies show that RAG systems introducing extrane-
ous or conflicting information degrade model coherence and increase hallucinations [20, 21, 22, 23].
Likewise, we found that adding high-quality retrieved context (e.g., from Harrison’s Internal Medicine
21e) decreased accuracy of most tested models, mirroring the effects of MedDistractQA. Altogether,
these results suggest that indiscriminate implementation of RAG may introduce information overload,
serving to impair rather than improve decision-making. These findings emphasize the necessity of
fine-tuning RAG systems for beneficial deployment.
This study has several limitations. First, we focused on the MedQA dataset, which although widely
used, does not necessarily encompass the full range of clinical reasoning tasks. The exclusive use of
multiple-choice questions limits generalizability and may not extend to other formats such as open-
ended problem-solving. Additionally, our distracting statements were generated algorithmically using
a specific LLM. Thus, although they were designed to mimic realistic distractions, they may not fully
capture the complexity or breadth of real-life clinical discourse. Finally, while we observed a strong
correlation between baseline accuracy and resilience to distractions, further controlled experiments are
needed to establish causality.
Future research should expand these distraction evaluations beyond MedQA to encompass diagnos-
tic reasoning, treatment planning, and patient triage. Developing targeted mitigation strategies -
from improved prompt engineering to context-aware filtering and architectural modifications - will be
essential for clinical deployment. Particular attention should be paid to emerging technologies like
ambient dictation systems, which may introduce substantial extraneous information into downstream
AI processes. Additionally, evaluating model performance across structured data and multimodal in-
puts could reveal new failure modes. Understanding how model architecture, training scale, and data
quality influence distraction susceptibility will be crucial for building robust clinical AI systems.
Our findings indicate that LLMs, even frontier models (Claude Sonnet, o3), are vulnerable to obvious
distractions in medical narratives. Fine-tuning open-source LLMs with medical data provides only
limited protection against these vulnerabilities. More concerning, RAG - a common strategy for
enhancing LLMs’ accuracy and robustness - can inadvertently introduce confounding information that
degrades model performance. These results underscore how seemingly simple, common-sense scenarios
can expose critical limitations in medical LLMs. As deployment of these systems accelerates, our
findings emphasize the urgent need for robust evaluation frameworks that better reflect real-world
clinical complexity.
7

Methods
MedQA Benchmark
To determine an LLM’s ability in the medical domain, we evaluate the model on the MedQA, a
USMLE-style question bank [13]. We choose to evaluate on this dataset due to the expert level of
medical reasoning and knowledge required for USMLE-style questions, and to test the model’s ability
against the range of critical clinical tasks such as differential diagnosis.
MedDistractQA Benchmark Curation
To generate confounding statements, we first identify medical terms from the incorrect answer choices
of each MedQA question. By focusing on these distractor terms rather than the correct ones, we ensure
that the added statements do not directly hint at the true solution and instead create non-operational
content. We parse the list of wrong answer choices and extract clinically relevant concepts—such as
diseases, conditions, or procedures—that appear within them. For instance, if an incorrect choice
in a particular question includes “heart attack,” we flag that term for potential use in a distracting
statement. This selection process capitalizes on the inherent diversity of erroneous answer choices,
which frequently contain common medical conditions that can be repurposed in nonclinical contexts.
After extracting these terms, we prompt a large language model (GPT-4o) to generate short, coherent
sentences in which each medical term is used in a nonclinical or socially oriented manner. The model
is instructed to produce statements that sound natural yet are clinically irrelevant—examples might
include “The patient’s zodiac sign is Cancer” (where “Cancer” is the zodiac rather than the disease)
or “My aunt’s fish had a heart attack” (imputing a clinical symptom on a distant bystander, in this
case, a fish of the patient’s aunt). These statements are then embedded into the original MedQA
questions, creating augmented versions intended to distract or confuse the model. By systematically
introducing such confounding statements, we can evaluate whether the presence of irrelevant medical
languagedegradesthemodel’sabilitytoidentifythecorrectclinicalanswer. Thismethodofbenchmark
curation is visually depicted in Fig. 1a.
Model Evaluation
To calculate accuracy of a model on the MedQA, we use string-based matching on model output
chain-of-thought. Inference is computed at a temperature of 0 without ensemble. We note that
some models do not allow for temperature control (e.g., OpenAI’s reasoning models), and are left at
their default. MedMobile is ran using the PyTorch and Transformers library on A100 GPUs during
evaluation. vLLM is utilized for all other open-source and medically fine-tuned models on A100 GPUs.
Proprietary models inference is generated via their respective official API provider. Model names are
unedited, and are directly labeled as present in HuggingFace Hub or the corresponding proprietary
provider’s API console.
8

Prompting
Prompts utilized are available in within the GitHub repo. We keep an identical prompting tem-
plate between evaluations of MedQA, MedDistractQA-Nonliteral, MedDistractQA-Bystander, and
MedQA+RAG.
Retrieval-Augmented Generation
To conduct RAG based on vector embeddings, we compute cosine similarity based on MedCPT [24]
vectors generation between the question and paragraphs in the textbook. RAG selects the paragraph
with the highest cosine-similarity score for a particular question. The source of information for these
evaluations is from Harrison’s Principles of Internal Medicine, 21e [15]. After selecting a relevant
paragraph from the corpora, we insert it into the prompt directly during inference.
Categorization of Data
For each MedQA and MedDistractQA question, we utilize OpenAI’s o3-mini to classify the category
that the question best fits under. The categories topics are derived directly from the USMLE website
[14].
Uncertainty Quantification and Significance Testing
The statistical analysis was performed on paired accuracy measurements from baseline and modified
(MedDistractQA or MedQA+RAG) evaluations for each model. For each model, the fraction of
correctly answered questions was computed under both conditions, and the difference in accuracy
(multiplied by 100 to express percentage points) was calculated. To quantify variability, the standard
error of the difference was estimated using a formula derived from binomial variance components.
Specifically, if p1andp2denote the baseline and MedDistractQA accuracies respectively, and p12
represents the joint accuracy (the proportion of questions correctly answered in both conditions), then
the standard error was computed as
SEdiff=/radicalbigg
p1(1−p1) +p2(1−p2)−2 (p12−p1p2)
n×100,
where nis the number of paired observations. This approach accounts for the covariance between the
two conditions, ensuring a more accurate estimate of the uncertainty in the observed differences.
Further statistical analyses included group comparisons in which models were classified into three
categories (general open-source, medical open-source, or proprietary). Grouped models are compared
using Welch’s t-test to account for differences in sample size and variance. The t-test produces a
two-tailed p-value which is then converted into one-tailed p-values to test directional hypotheses about
which group exhibits greater degradation.
Pairwise comparisons were conducted using two-sample t-tests (with both one-tailed and two-tailed
tests) to assess the significance of observed differences in accuracy loss. Linear regression analyses
were also performed to evaluate relationships between model size, baseline performance, and accuracy
9

loss, and performance degradation. Additionally, paired t-tests and correlation analyses were applied
to compare the degradation in performance across different evaluation settings (MedDistractQA and
RAG_medqa). All statistical tests were implemented using standard Python libraries such as numpy,
scipy.stats , and pandasto ensure reproducibility.
Acknowledgements
E.K.O.issupportedbytheNationalCancerInstitute’sEarlySurgeonScientistProgram(3P30CA016087-
41S1) and the W.M. Keck Foundation. We would like to acknowledge Nader Mherabi and Dafna
Bar-Sagi, Ph.D., for their continued support of medical AI research at NYU. We thank Michael Con-
stantino, Kevin Yie, and the NYU Langone High-Performance Computing (HPC) Team for supporting
computing resources fundamental to our work.
Author Contributions
E.K.O. and A. A. conceptualized and supervised the study. KV designed, implemented, and developed
the LLM evaluation pipeline and the MedDistractQA benchmarks. KV wrote the initial draft of the
manuscript. All authors revised and approved the manuscript.
Competing Interests
Disclosures: EKO reports consulting income with Sofinnova Partners. EKO reports equity in Eikon
Therapeutics, Artisight Incorporate. The other authors have no personal, financial, or institutional
interest pertinent to this article.
Data Availability
The datasets generated or analyzed during the current study are available in the nyuolab/clini-
cal_confounders repository, https://github.com/nyuolab/MedDistractQA . The benchmarks devel-
oped (i.e., MedDistractQA-Nonliteral and MedDistractQA-Bystander) are available on HuggingFace
dataset hub upon publication of this work.
Code Availability
Our code is shared publicly on GitHub upon publication of this work and can be found at
https://github.com/nyuolab/MedDistractQA .
10

References
[1] Suzanne V Blackley, Valerie D Schubert, Foster R Goss, Wasim Al Assad, Pamela M Garabedian,
and Li Zhou. Physician use of speech recognition versus typing in clinical documentation: A
controlled observational study. International Journal of Medical Informatics , 141:104178, 2020.
[2] John R Zech, Marcus A Badgeley, Manway Liu, Anthony B Costa, Joseph J Titano, and Eric Karl
Oermann. Variable generalization performance of a deep learning model to detect pneumonia in
chest radiographs: a cross-sectional study. PLoS medicine , 15(11):e1002683, 2018.
[3] Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan
Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. Large language models encode
clinical knowledge. Nature, 620(7972):172–180, 2023.
[4] Harsha Nori, Yin Tat Lee, Sheng Zhang, Dean Carignan, Richard Edgar, Nicolo Fusi, Nicholas
King, Jonathan Larson, Yuanzhi Li, Weishung Liu, et al. Can generalist foundation models
outcompete special-purpose tuning? case study in medicine. arXiv preprint arXiv:2311.16452 ,
2023.
[5] Asma Ben Abacha, Wen-wai Yim, Yujuan Fu, Zhaoyi Sun, Meliha Yetisgen, Fei Xia, and Thomas
Lin. Medec: A benchmark for medical error detection and correction in clinical notes. arXiv
preprint arXiv:2412.19260 , 2024.
[6] Kaiyan Zhang, Sihang Zeng, Ermo Hua, Ning Ding, Zhang-Ren Chen, Zhiyuan Ma, Haoxin Li,
Ganqu Cui, Biqing Qi, Xuekai Zhu, et al. Ultramedical: Building specialized generalists in
biomedicine. Advances in Neural Information Processing Systems , 37:26045–26081, 2025.
[7] Khaled Saab, Tao Tu, Wei-Hung Weng, Ryutaro Tanno, David Stutz, Ellery Wulczyn, Fan Zhang,
Tim Strother, Chunjong Park, Elahe Vedadi, et al. Capabilities of gemini models in medicine.
arXiv preprint arXiv:2404.18416 , 2024.
[8] Zeming Chen, Alejandro Hernández Cano, Angelika Romanou, Antoine Bonnet, Kyle Ma-
toba, Francesco Salvi, Matteo Pagliardini, Simin Fan, Andreas Köpf, Amirkeivan Mohtashami,
et al. Meditron-70b: Scaling medical pretraining for large language models. arXiv preprint
arXiv:2311.16079 , 2023.
[9] Krithik Vishwanath, Jaden Stryker, Anton Alyakin, Daniel Alexander Alber, and Eric Karl Oer-
mann. Medmobile: A mobile-sized language model with expert-level clinical capabilities. arXiv
preprint arXiv:2410.09019 , 2024.
[10] Paul Hager, Friederike Jungmann, Robbie Holland, Kunal Bhagat, Inga Hubrecht, Manuel
Knauer, Jakob Vielhauer, Marcus Makowski, Rickmer Braren, Georgios Kaissis, et al. Evaluation
and mitigation of the limitations of large language models in clinical decision-making. Nature
medicine , 30(9):2613–2622, 2024.
[11] Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H Chi, Nathanael
Schärli, and Denny Zhou. Large language models can be easily distracted by irrelevant context.
InInternational Conference on Machine Learning , pages 31210–31227. PMLR, 2023.
[12] Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, and Mehrdad
Farajtabar. Gsm-symbolic: Understanding the limitations of mathematical reasoning in large
language models. arXiv preprint arXiv:2410.05229 , 2024.
11

[13] Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What
disease does this patient have? a large-scale opendomain question answering dataset from medical
exams. Applied Sciences , 11(14):6421, 2021.
[14] United States Medical Licensing Examination. Usmle ®physician tasks/competencies, 2024.
[15] ESilverman, JCrapo, BMake, JJameson, AFauci, DKasper, SHauser, DLongo, andJLoscalzo.
Harrison’s principles of internal medicine 21e, 2022.
[16] Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Mohamed Amin, Le Hou,
Kevin Clark, Stephen R Pfohl, Heather Cole-Lewis, et al. Toward expert-level medical question
answering with large language models. Nature Medicine , pages 1–8, 2025.
[17] Yuxuan Zhou, Xien Liu, Chen Ning, Xiao Zhang, Chenwei Yan, Xiangling Fu, and Ji Wu. Revis-
iting the scaling effects of LLMs on medical reasoning capabilities, 2025.
[18] Robert Osazuwa Ness, Katie Matton, Hayden Helm, Sheng Zhang, Junaid Bajwa, Carey E Priebe,
and Eric Horvitz. Medfuzz: Exploring the robustness of large language models in medical question
answering. arXiv preprint arXiv:2406.06573 , 2024.
[19] Divyanshu Kumar, Anurakt Kumar, Sahil Agarwal, and Prashanth Harshangi. Increased llm
vulnerabilities from fine-tuning and quantization. arXiv e-prints , pages arXiv–2404, 2024.
[20] Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang, Hongru Wang, Yue Zhang, and Wei Xu.
Knowledge conflicts for llms: A survey. arXiv preprint arXiv:2403.08319 , 2024.
[21] Seong-IlParkandJay-YoonLee. Towardrobustralms: Revealingtheimpactofimperfectretrieval
on retrieval-augmented language models. Transactions of the Association for Computational Lin-
guistics, 12:1686–1702, 2024.
[22] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni,
and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of
the Association for Computational Linguistics , 12:157–173, 2024.
[23] Kevin Wu, Eric Wu, and James Zou. Clasheval: Quantifying the tug-of-war between an llm’s
internal prior and external evidence. In The Thirty-eight Conference on Neural Information
Processing Systems Datasets and Benchmarks Track , 2024.
[24] Qiao Jin, Won Kim, Qingyu Chen, Donald C Comeau, Lana Yeganova, W John Wilbur, and
Zhiyong Lu. Medcpt: Contrastive pre-trained transformers with large-scale pubmed search logs
for zero-shot biomedical information retrieval. Bioinformatics , 39(11):btad651, 2023.
12

Extended Data
a
b
Extended Data Figure 1. Group-wise comparison of performance degradation due to distractions,
with models split up as Open-source, general (n=18), Open-source, medical (n=3), or Proprietary
(n=8). aResults for MedDistractQA-Nonliteral, bresults for MedDistractQA-Bystander
13

a
bExtended Data Figure 2. Relationship between baseline performance on MedQA and the resulting
accuracy loss when tested on two MedDistractQA variants: aNonliteral and bBystander. Each
marker corresponds to a different class of large language model (circle: open-source, general; square:
open-source, medical; triangle: proprietary). The horizontal axis shows each model’s baseline accuracy
on MedQA (%), while the vertical axis shows the loss in accuracy (%) incurred under the distractor
conditions. The dashed trend lines illustrate positive correlations between baseline accuracy and
accuracyloss, withr=0.76forMedDistractQA-Nonliteralandr=0.70forMedDistractQA-Bystander.
14

gemma-3-4b-it
Mistral-7B-Instruct-v0.3gpt-4o
Ministral-8B-Instruct-2410Qwen2.5-1.5B-Instructgemma-2-2b-it
Qwen2.5-0.5B-Instructo3-mini
Llama-3.1-8B-Instructgpt-4o-mini
Mistral-7B-Instruct-v0.1 Qwen2.5-VL-3B-InstructQwen2.5-3B-Instructllama3-3-70b-chato1-mini
DeepSeek-R1-Distill-Llama-8BQwen2.5-7B-Instructgemma-2-9b-it
llama-3-meerkat-8b-v1.0Llama-3.2-1B-InstructMistral-7B-Instruct-v0.2Meta-Llama-3-8B-InstructLlama-3.2-3B-Instructgemma-3-1b-it
Llama-3-8B-UltraMedical
Model108642-0-2-4Loss in Model Accuracy (%)-2.8%
-1.5%
-1.4%-1.2%-1.1%
-1.1% -1.0%
-0.4%-0.1%
0.0%0.0%0.2% 0.2%0.4%0.4% 0.6% 0.6%0.8% 0.9% 1.0%1.1%
1.3%1.3%
3.7%
8.2%Open-source, general
Proprietary
Open-source, medicalExtended Data Figure 3. Trying to reduce distractor (MedDistractQA-Nonliteral) impact via
explicitprompting. LossinmodelaccuracyisrelativetotheMedDistractQA-Nonliteral, andrepresents
accuracy of MedDistractQA-Nonliteral with new prompting style minus the original prompting style
on MedDistractQA-Nonliteral. Positive score indicates new prompting method helped over original.
15

1. Medical Knowledge/Scientific Concepts2. Patient Care: Diagnosis
3. Patient Care: Management4. Communication
5. Professionalism, Including Legal and Ethical Issues6. Systems-based Practice, Including Patient Safety7. Practice-based Learning
Medical Competency CategoryDeepSeek-R1-Distill-Llama-8B
Llama-3-8B-UltraMedical
Llama-3.1-8B-Instruct
Llama-3.2-1B-Instruct
Llama-3.2-3B-Instruct
MedMobile
Mistral-7B-Instruct-v0.3
Qwen2.5-0.5B-Instruct
Qwen2.5-1.5B-Instruct
Qwen2.5-3B-Instruct
Qwen2.5-7B-Instruct
Qwen2.5-VL-3B-Instruct
claude-3-5-haiku-20241022
claude-3-5-sonnet-20241022
gemma-2-2b-it
gemma-2-9b-it
gemma-3-1b-it
gemma-3-4b-it
gpt-4o
gpt-4o-mini
llama-3-meerkat-8b-v1.0
llama3-3-70b-chat
o1-mini
o3-mini
o3-mini-high
o3-mini-low
AverageModel-13.6 -11.6 -8.3 0.0 0.0 12.5
-15.7 -13.6 -6.3 0.0 13.3 12.5
-9.6 -9.6 -0.2 12.5 3.3 25.0
-11.4 -10.2 -4.7 12.5 0.0 25.0
-12.7 -13.4 -9.0 0.0 3.3 25.0
-11.7 -12.9 -7.0 12.5 13.3 -12.5
-19.6 -14.9 -9.8 25.0 0.0 -37.5
-10.5 -9.4 -2.3 0.0 -13.3 -12.5
-11.1 -16.9 -8.7 0.0 -6.7 12.5
-13.3 -16.3 -7.2 -25.0 6.7 0.0
-12.8 -11.4 -1.5 -12.5 -16.7 -12.5
-14.4 -13.1 -6.5 -50.0 -10.0 -12.5
-8.2 -8.1 -1.7 0.0 0.0 0.0
-2.9 -4.7 -2.7 0.0 -3.3 0.0
-14.8 -15.1 -7.0 -37.5 0.0 12.5
-11.8 -12.2 -5.7 25.0 -3.3 0.0
-8.5 -9.4 -11.5 25.0 10.0 12.5
-14.0 -18.0 -6.2 -25.0 6.7 12.5
-5.9 -4.7 -0.5 0.0 -3.3 0.0
-10.5 -9.2 -3.0 -25.0 3.3 -25.0
-9.7 -13.1 -2.7 -12.5 3.3 -12.5
-5.4 -3.5 -4.7 -12.5 -6.7 0.0
-2.8 -3.2 -0.3 12.5 -13.3 12.5
-4.0 -2.9 0.0 0.0 3.3 0.0
-3.1 -3.5 -2.2 0.0 -6.7 0.0
-5.2 -2.9 -1.3 -25.0 0.0 -25.0
-10.1 -10.1 -4.6 -3.8 -0.6 0.5n=652 n=298 n=300 n=4 n=15 n=4 n=0
60
40
20
0204060
Change in Accuracy (%)Extended Data Figure 4. Average performance change due to MedDistractQA-Nonliteral and
MedDistractQA-Bystander by medical competency categorization of question. A positive number
indicates that performance was improved after a distraction was added, while a negative number
indicates that performance degraded with the addition of a distraction.
16

1. Human Development2. Immune System
3. Blood & Lymphoreticular4. Behavioral Health
5. Nervous System & Special Senses6. Musculoskeletal & Skin 7. Cardiovascular System8. Respiratory System
9. Gastrointestinal System10. Renal & Urinary System 11. Pregnancy & Childbirth12. Endocrine System
13. Multisystem Processes
14. Biostatistics & Epidemiology15. Communication Skills16. Legal/Ethical Issues
Medical System CategoryDeepSeek-R1-Distill-Llama-8B
Llama-3-8B-UltraMedical
Llama-3.1-8B-Instruct
Llama-3.2-1B-Instruct
Llama-3.2-3B-Instruct
MedMobile
Mistral-7B-Instruct-v0.3
Qwen2.5-0.5B-Instruct
Qwen2.5-1.5B-Instruct
Qwen2.5-3B-Instruct
Qwen2.5-7B-Instruct
Qwen2.5-VL-3B-Instruct
claude-3-5-haiku-20241022
claude-3-5-sonnet-20241022
gemma-2-2b-it
gemma-2-9b-it
gemma-3-1b-it
gemma-3-4b-it
gpt-4o
gpt-4o-mini
llama-3-meerkat-8b-v1.0
llama3-3-70b-chat
o1-mini
o3-mini
o3-mini-high
o3-mini-low
AverageModel-4.5 -18.3 -10.1 -3.0 -13.3 -9.0 -10.2 -19.2 -9.3 -13.1 -11.2 -16.8 -20.9 6.1 0.0 3.3
-13.6 -21.4 -12.8 -3.0 -17.5 -20.5 -9.8 -22.0 -9.3 -16.4 -12.5 -7.6 -2.2 0.0 16.7 3.3
-9.1 -5.6 -8.0 -9.5 -4.2 -12.3 -11.8 -13.2 -3.0 -8.0 -0.6 -2.2 -11.2 1.5 0.0 10.0
-4.5 -7.9 -10.6 -11.3 -6.2 -8.6 -11.4 -8.8 -12.2 -8.4 -6.9 -8.2 -14.2 -9.1 -8.3 10.0
-22.7 -13.5 -18.1 -2.4 -14.2 -7.0 -17.1 -23.6 -5.6 -14.2 -15.0 -0.5 -16.4 -1.5 8.3 3.3
4.5 -4.0 -16.5 -6.5 -15.8 -14.8 -14.6 -6.6 -11.5 -8.4 -15.0 -4.9 -10.4 -4.5 8.3 10.0
-54.5 -19.8 -13.8 -11.9 -22.1 -18.4 -20.7 -28.6 -9.3 -11.3 -11.2 -8.2 -24.6 1.5 16.7 -6.7
-13.6 -11.9 -3.7 3.6 -1.3 -15.6 -8.9 -15.9 -4.8 -13.1 -6.9 -15.2 -3.0 -12.1 0.0 -3.3
18.2 -15.9 -3.7 -3.0 -11.3 -14.3 -11.8 -18.7 -16.3 -10.9 -9.4 -14.7 -22.4 1.5 -16.7 3.3
0.0 -4.8 -20.2 -7.7 -12.5 -11.5 -9.3 -17.0 -14.8 -15.7 -7.5 -15.8 -12.7 -7.6 -16.7 10.0
4.5 0.0 -7.4 -11.3 -10.8 -8.6 -12.6 -12.6 -10.7 -9.1 -14.4 -12.5 -7.5 -4.5 -33.3 -3.3
-13.6 -15.1 -9.6 -12.5 -7.1 -15.6 -12.6 -14.8 -14.4 -15.3 -9.4 -13.0 -6.7 -3.0 -50.0 -6.7
-13.6 -1.6 -10.1 1.2 -12.5 -4.1 -8.5 -6.6 -6.3 -13.1 -5.0 -4.9 -2.2 6.1 0.0 -3.3
-4.5 0.0 -1.6 -4.8 -4.2 -3.3 -0.4 -2.7 -5.6 -4.4 -6.9 -3.3 -1.5 0.0 0.0 -3.3
4.5 -11.1 -12.2 -8.3 -17.5 -14.8 -14.6 -18.7 -10.0 -18.6 -15.0 -4.3 -10.4 -4.5 -41.7 10.0
0.0 -16.7 -9.0 -10.7 -17.9 -13.1 -8.1 -14.3 -4.8 -7.3 0.6 -12.0 -17.9 -9.1 0.0 3.3
-27.3 -14.3 -3.7 -6.0 -9.6 -9.0 -3.7 -12.1 -12.6 -16.8 -14.4 -2.7 -0.7 -15.2 16.7 13.3
0.0 -14.3 -14.9 -4.2 -13.8 -17.6 -13.4 -20.3 -13.0 -9.1 -10.6 -17.4 -14.9 3.0 -25.0 13.3
9.1 -5.6 -8.0 0.6 -4.2 -1.6 -7.3 -3.8 -3.7 -3.3 -7.5 -5.4 -5.2 -3.0 0.0 -3.3
-13.6 -5.6 -15.4 -3.6 -8.8 -7.4 -8.1 -12.6 -4.8 -6.9 -14.4 -7.1 -7.5 -6.1 -16.7 -6.7
-22.7 -6.3 -13.8 -9.5 -10.0 -11.5 -8.1 -7.7 -1.5 -9.9 -14.4 -7.6 -6.0 -10.6 -8.3 10.0
4.5 -7.9 -10.6 -1.8 -9.2 -0.8 0.0 -4.4 -5.6 -5.1 -1.2 -8.7 -9.0 1.5 -8.3 0.0
-4.5 -10.3 -3.2 -1.8 0.0 -2.9 -1.2 1.6 -0.4 -5.1 -1.9 -4.9 -0.7 1.5 -8.3 -6.7
9.1 -0.8 -5.9 -3.0 -5.0 -1.6 -5.7 -2.7 -3.7 0.0 1.9 -4.9 -3.0 1.5 0.0 3.3
0.0 0.0 -8.0 -1.2 -5.0 -4.5 -3.7 -1.1 -1.9 -1.1 -1.2 -4.9 -6.0 4.5 0.0 -6.7
4.5 -2.4 -4.3 1.2 -5.0 -2.9 -6.9 -6.6 0.7 -2.2 -5.6 -6.0 -6.7 -6.1 -16.7 -6.7
-6.3 -9.0 -9.8 -5.0 -10.0 -9.7 -9.3 -12.0 -7.5 -9.5 -8.3 -8.2 -9.4 -2.6 -7.1 1.9n=11 n=63 n=94 n=84 n=120 n=122 n=123 n=91 n=135 n=137 n=80 n=92 n=67 n=33 n=6 n=15
60
40
20
0204060
Change in Accuracy (%)Extended Data Figure 5. Average performance change due to MedDistractQA-Nonliteral and
MedDistractQA-Bystander by medical system categorization of question. A positive number indicates
that performance was improved after a distraction was added, while a negative number indicates that
performance degraded with the addition of a distraction.
17

0 200 400 600 800 1000
Rank of RAG0.30.40.50.60.70.80.9Accuracy on the MedQA (USMLE)medmobile
No RAG accuracy
0 200 400 600 800 1000
Rank of RAG0.30.40.50.60.70.80.9Accuracy on the MedQA (USMLE)Llama-3.1-8B-Instruct
No RAG accuracy
0 200 400 600 800 1000
Rank of RAG0.30.40.50.60.70.80.9Accuracy on the MedQA (USMLE)llama3-70b-chat
No RAG accuracy
0 200 400 600 800 1000
Rank of RAG0.30.40.50.60.70.80.9Accuracy on the MedQA (USMLE)gpt-4o-mini
No RAG accuracy
0 200 400 600 800 1000
Rank of RAG0.30.40.50.60.70.80.9Accuracy on the MedQA (USMLE)gemma-2-9b-it
No RAG accuracy
0 200 400 600 800 1000
Rank of RAG0.30.40.50.60.70.80.9Accuracy on the MedQA (USMLE)Mistral-7B-Instruct-v0.3
No RAG accuracyExtended Data Figure 6. Comparison of MedQA accuracy with and without retrieval-augmented
generation (RAG) across six different language models. Each subplot shows accuracy (y-axis) versus
the retrieval rank (x-axis), with the red horizontal line indicating the model’s baseline accuracy (no
RAG) and the blue points showing accuracy under RAG at varying ranks.
18

Supplementary Materials
3
 0
Loss in Model Accuracy (%)Mistral-7B-Instruct-v0.3Llama-3-8B-UltraMedicalgemma-2-2b-itllama-3-meerkat-8b-v1.0MedMobilegpt-4ogpt-4o-miniclaude-3-5-haiku-20241022claude-3-5-sonnet-20241022llama3-3-70b-chatgemma-2-9b-itLlama-3.1-8B-InstructModel
-3.4%-2.5%-2.1%-1.6%-1.0%-1.0%-0.9%-0.4%-0.3%-0.3%-0.1%0.3%Average Difference Across Sentences
open-source
fine-tuned
closed-source
6
 3
 0
Loss in Model Accuracy (%)Mistral-7B-Instruct-v0.3Llama-3-8B-UltraMedicalgemma-2-2b-itllama-3-meerkat-8b-v1.0MedMobilegpt-4ogpt-4o-miniclaude-3-5-haiku-20241022claude-3-5-sonnet-20241022llama3-3-70b-chatgemma-2-9b-itLlama-3.1-8B-InstructModel
-5.4%-3.6%-5.3%-6.1%0.7%-1.2%0.1%-0.4%-0.6%0.3%-1.3%-0.1%Sentence 1:
The patient's zodiac sign is Cancer.
open-source
fine-tuned
closed-source
3
 0
Loss in Model Accuracy (%)Mistral-7B-Instruct-v0.3Llama-3-8B-UltraMedicalgemma-2-2b-itllama-3-meerkat-8b-v1.0MedMobilegpt-4ogpt-4o-miniclaude-3-5-haiku-20241022claude-3-5-sonnet-20241022llama3-3-70b-chatgemma-2-9b-itLlama-3.1-8B-InstructModel
-4.9%-3.0%-1.1%-2.1%-2.0%-0.3%-1.2%-0.7%-0.2%-0.0%0.3%-0.0%Sentence 2:
The patient s mother likes the sky.
open-source
fine-tuned
closed-source
3
 0
Loss in Model Accuracy (%)Mistral-7B-Instruct-v0.3Llama-3-8B-UltraMedicalgemma-2-2b-itllama-3-meerkat-8b-v1.0MedMobilegpt-4ogpt-4o-miniclaude-3-5-haiku-20241022claude-3-5-sonnet-20241022llama3-3-70b-chatgemma-2-9b-itLlama-3.1-8B-InstructModel
-3.3%-1.8%-1.5%0.0%-0.2%0.1%-0.6%-0.9%-0.4%-0.6%-0.4%0.2%Sentence 3:
The patient owns a pair of sneakers and a blue waterbottle.
open-source
fine-tuned
closed-source
0
Loss in Model Accuracy (%)Mistral-7B-Instruct-v0.3Llama-3-8B-UltraMedicalgemma-2-2b-itllama-3-meerkat-8b-v1.0MedMobilegpt-4ogpt-4o-miniclaude-3-5-haiku-20241022claude-3-5-sonnet-20241022llama3-3-70b-chatgemma-2-9b-itLlama-3.1-8B-InstructModel
-2.7%-0.2%-2.1%0.6%-1.0%-2.0%-1.6%0.9%0.4%-0.3%0.4%1.0%Sentence 4:
They said they celebrate their birthday in July.
open-source
fine-tuned
closed-source
3
 0
Loss in Model Accuracy (%)Mistral-7B-Instruct-v0.3Llama-3-8B-UltraMedicalgemma-2-2b-itllama-3-meerkat-8b-v1.0MedMobilegpt-4ogpt-4o-miniclaude-3-5-haiku-20241022claude-3-5-sonnet-20241022llama3-3-70b-chatgemma-2-9b-itLlama-3.1-8B-InstructModel
-0.7%-3.8%-0.6%-0.4%-2.8%-1.5%-1.5%-0.8%-0.9%-0.8%0.3%0.3%Sentence 5:
The patient loves the number six.
open-source
fine-tuned
closed-source
Supplemental Figure 1. MedDistractQA-Nonliteral and performance on individual sentences with
nonliteral clinical terms. For these ablation studies, we utilize one singular distracting sentence curated
for the entire dataset, rather than for each individual question.
19

3
 0
Loss in Model Accuracy (%)Mistral-7B-Instruct-v0.3Llama-3-8B-UltraMedicalgpt-4o-miniclaude-3-5-haiku-20241022claude-3-5-sonnet-20241022gemma-2-9b-itgemma-2-2b-itLlama-3.1-8B-Instructgpt-4ollama-3-meerkat-8b-v1.0llama3-3-70b-chatModel
-4.9%-3.8%-2.5%-2.3%-2.1%-2.0%-1.5%-1.2%-1.1%-0.7%-0.6%Average Difference Across Sentences
open-source
fine-tuned
closed-source
3
 0
Loss in Model Accuracy (%)Mistral-7B-Instruct-v0.3Llama-3-8B-UltraMedicalgpt-4o-miniclaude-3-5-haiku-20241022claude-3-5-sonnet-20241022gemma-2-9b-itgemma-2-2b-itLlama-3.1-8B-Instructgpt-4ollama-3-meerkat-8b-v1.0llama3-3-70b-chatModel
-5.0%-3.4%-3.0%-3.2%-1.1%-1.9%-0.8%0.2%-0.5%-0.1%0.1%Sentence 1:
The patient's sister said her friend has high
blood pressure and has been having seizures.
open-source
fine-tuned
closed-source
0
Loss in Model Accuracy (%)Mistral-7B-Instruct-v0.3Llama-3-8B-UltraMedicalgpt-4o-miniclaude-3-5-haiku-20241022claude-3-5-sonnet-20241022gemma-2-9b-itgemma-2-2b-itLlama-3.1-8B-Instructgpt-4ollama-3-meerkat-8b-v1.0llama3-3-70b-chatModel
-2.6%-2.2%-1.8%-0.6%-2.2%0.9%-1.1%-1.3%-0.5%-0.7%-0.4%Sentence 2:
The patient's dog has diabetes and cannot see
properly.
open-source
fine-tuned
closed-source
6
 3
 0
Loss in Model Accuracy (%)Mistral-7B-Instruct-v0.3Llama-3-8B-UltraMedicalgpt-4o-miniclaude-3-5-haiku-20241022claude-3-5-sonnet-20241022gemma-2-9b-itgemma-2-2b-itLlama-3.1-8B-Instructgpt-4ollama-3-meerkat-8b-v1.0llama3-3-70b-chatModel
-6.7%-3.9%-3.3%-4.2%-3.0%-3.5%-3.1%-1.2%-1.9%-1.4%-0.5%Sentence 3:
The patient's neighbor mentioned that their cat
has been sneezing and refusing to eat.
open-source
fine-tuned
closed-source
3
 0
Loss in Model Accuracy (%)Mistral-7B-Instruct-v0.3Llama-3-8B-UltraMedicalgpt-4o-miniclaude-3-5-haiku-20241022claude-3-5-sonnet-20241022gemma-2-9b-itgemma-2-2b-itLlama-3.1-8B-Instructgpt-4ollama-3-meerkat-8b-v1.0llama3-3-70b-chatModel
-4.9%-4.6%-1.8%-1.8%-1.0%-1.6%-0.6%-1.6%-0.4%-0.9%-0.4%Sentence 4:
The patient's cousin said her coworker's father
has arthritis and struggles to walk long
distances.
open-source
fine-tuned
closed-source
3
 0
Loss in Model Accuracy (%)Mistral-7B-Instruct-v0.3Llama-3-8B-UltraMedicalgpt-4o-miniclaude-3-5-haiku-20241022claude-3-5-sonnet-20241022gemma-2-9b-itgemma-2-2b-itLlama-3.1-8B-Instructgpt-4ollama-3-meerkat-8b-v1.0llama3-3-70b-chatModel
-5.1%-5.0%-2.4%-1.8%-3.2%-3.8%-2.0%-2.2%-2.0%-0.2%-1.7%Sentence 5:
The patient's roommate mentioned that their
goldfish has been swimming erratically and losing
color.
open-source
fine-tuned
closed-sourceSupplemental Figure 2. MedDistractQA-Bystander and performance on individual sentences with
socially-applied clinical terms. For these ablation studies, we utilize one singular distracting sentence
curated for the entire dataset, rather than for each individual question.
20
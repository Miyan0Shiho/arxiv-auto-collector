# TempPerturb-Eval: On the Joint Effects of Internal Temperature and External Perturbations in RAG Robustness

**Authors**: Yongxin Zhou, Philippe Mulhem, Didier Schwab

**Published**: 2025-12-01 01:46:36

**PDF URL**: [https://arxiv.org/pdf/2512.01183v1](https://arxiv.org/pdf/2512.01183v1)

## Abstract
The evaluation of Retrieval-Augmented Generation (RAG) systems typically examines retrieval quality and generation parameters like temperature in isolation, overlooking their interaction. This work presents a systematic investigation of how text perturbations (simulating noisy retrieval) interact with temperature settings across multiple LLM runs. We propose a comprehensive RAG Perturbation-Temperature Analysis Framework that subjects retrieved documents to three distinct perturbation types across varying temperature settings. Through extensive experiments on HotpotQA with both open-source and proprietary LLMs, we demonstrate that performance degradation follows distinct patterns: high-temperature settings consistently amplify vulnerability to perturbations, while certain perturbation types exhibit non-linear sensitivity across the temperature range. Our work yields three key contributions: (1) a diagnostic benchmark for assessing RAG robustness, (2) an analytical framework for quantifying perturbation-temperature interactions, and (3) practical guidelines for model selection and parameter tuning under noisy retrieval conditions.

## Full Text


<!-- PDF content starts -->

TempPerturb-Eval: On the Joint Effects of Internal Temperature and
External Perturbations in RAG Robustness
Yongxin Zhou, Philippe Mulhem, Didier Schwab
Univ. Grenoble Alpes, CNRS, Inria, Grenoble INP, LIG, 38000, Grenoble, France
yongxin.zhou, philippe.mulhem, didier.schwab@univ-grenoble-alpes.fr
Abstract
The evaluation of Retrieval-Augmented Generation (RAG) systems typically examines retrieval quality and generation
parameters like temperature in isolation, overlooking their interaction. This work presents a systematic investigation
of how text perturbations (simulating noisy retrieval) interact with temperature settings across multiple LLM runs. We
propose a comprehensive RAG Perturbation-Temperature Analysis Framework that subjects retrieved documents to
three distinct perturbation types across varying temperature settings. Through extensive experiments on HotpotQA
with both open-source and proprietary LLMs, we demonstrate that performance degradation follows distinct patterns:
high-temperature settings consistently amplify vulnerability to perturbations, while certain perturbation types exhibit
non-linear sensitivity across the temperature range. Our work yields three key contributions: (1) a diagnostic
benchmark for assessing RAG robustness, (2) an analytical framework for quantifying perturbation-temperature
interactions, and (3) practical guidelines for model selection and parameter tuning under noisy retrieval conditions.
Keywords:Retrieval-Augmented Generation (RAG), Temperature, Perturbation Analysis
1. Introduction
Retrieval Augmented Generation (RAG) (Lewis
et al., 2020) is a prompt engineering strategy that
augments the internal capacity of Large Language
Models (LLMs) with external knowledge. In a RAG,
incorrect retrieved documents can introduce ex-
ternal noise that affects output quality. One RAG
output also depends on the hyperparameters of its
LLM, e.g., thetemperature: the generated text is
more (resp. less) deterministic for low (resp. large)
temperature values (Holtzman et al., 2020).
Furthermore,Perturbationsserve as adversarial
examplesinevaluatingRAGrobustness,simulating
smallinputchangesthatcandeceivemodelsintoin-
correctpredictions. Thesemodificationshelpquan-
tify how much specific input features must change
to alter model outcomes (Anand et al., 2022). Pre-
vious research has employed various perturbation
strategies for RAG question-answering systems,
suchastheleave-one-token-outapproachthatsys-
tematicallyremovesindividualsentencesfrominput
texts (Sudhi et al., 2024).
However, existing literature overlooks a critical
dimension: the interaction between perturbations
and generation hyperparameters, particularly tem-
perature. This gap is significant given that tem-
perature substantially affects output quality across
various tasks (Renze, 2024; Du et al., 2025), with
low temperature values not always constituting the
optimalchoice. Consequently,currentperturbation-
basedevaluationsmayyieldmisleadingrobustness
assessments by failing to account for temperature
variability.
Our work addresses this limitation by systemati-
cally investigating how perturbations interact withtemperature settings in RAG systems. Traditional
evaluations examine retrieval quality and genera-
tion parameters in isolation, neglecting their prac-
tical interdependence. By integrating both dimen-
sions, our framework provides more reliable faith-
fulnessexplanationsandaccuraterobustnessmea-
surements under realistic deployment conditions.
We approach the RAG LLM as a black box and
experimentally quantify thetemperature effectus-
ing the HotpotQA dataset (Yang et al., 2018) in
conjunction with systematic perturbations. To our
knowledge, this represents the first comprehensive
study of temperature-perturbation interactions in
RAG systems.1As shown in Figure 1, our work
systematically investigates this complex interplay,
with three key contributions:
•A comprehensive taxonomy of perturbations
for RAG, synthesizing and categorizing meth-
ods from information retrieval literature.
•A publicly released diagnostic benchmark that
quantifies RAG robustness across 440 experi-
mental conditions, spanning multiple models,
temperatures, perturbation types, and ques-
tion types.
•An analytical framework that models the joint
impact of temperature and perturbations, ac-
companied by practical guidelines for robust
model deployment.
1The source code and experimental results will be
available at https://github.com/yongxin2020/
TempPerturb-RAG.arXiv:2512.01183v1  [cs.CL]  1 Dec 2025

Question 
OUTPUT: 
Multiple Answers 
PERTURBATIONS: 
2. Sentence Replacement 
3. Sentence Removal 
4. NER Replacement TEMPERATURE 
(0.0, 0.2, 0.4, … , 1.8, 2.0) RAG EVALUATION: 
Correctness 
Variability 
Qualitative Analysis Contributions: 
1. Diagnostic Benchmark for RAG 
2. Perturbation-T emperature Framework 
3. Practical Guidelines Baseline: 
1. Supporting Sentences Context 
Three runs  
per sample Bridge questions 
Comparison questions 
BERTScore 
ROUGE-1/2/L Figure 1: RAG Perturbation-Temperature Analysis Framework. The methodology stresses system ro-
bustness along two axes: external context perturbations (replacement, removal, NER substitution) and
internal LLM temperature variation. The evaluation measures correctness and output variability across
these conditions to establish a benchmark and derive practical guidelines.
2. Related Work
2.1. Perturbations in IR and RAG
Previous work have used perturbations as adver-
sarial examples to examine the robustness of In-
formation Retrieval (IR) models (Raval and Verma,
2020; Wu et al., 2023; Liu et al., 2024). Typical
perturbations include removing, adding, or replac-
ingwords, phrases, sentences, passages, orentire
documents. Forinstance, RavalandVerma(2020)
foundthatevenminimaltokenchanges(1-3),anat-
tackercan produce semantically similar perturbed
documents capable of fooling document rankers.
Perturbations are also used in counterfactual ex-
planations in explainable IR, i.e. the closest sam-
ples on which the model makes a different predic-
tion, as example-based explanations (Poché et al.,
2023). Some demonstrations (Rorseth et al., 2023,
2024) provide examples of explanations for RAGs
obtained using different perturbation methods, but
they do not evaluate their proposals. In addition, in
the context of RAG, only theleave-one-token-out
strategy was evaluated on a Question-Answering
task (Sudhi et al., 2024).
We review relevant literature to provide an
overview of perturbation methods from explainable
information retrieval that can assess the robust-
ness of IR and RAG systems (Zhou et al., 2025).
These approaches are summarized in Table 1 and
organized along the following dimensions:
•Target: The IR component being perturbed
(document or query).
•Perturbation Category: The high-level strat-
egy for modifying content (e.g., subset selec-
tion, addition, replacement).
•Specific Method: The concrete technique
used to implement the perturbation.
•Granularity: The textual unit affected by the
perturbation (e.g., token, sentence, passage).•DescriptionandApplication: Anexplanation
of the method and its use cases.
The purpose of the retrieval perturbations sum-
marized in Table 1 is to introduce controlled de-
viations from an ideal retrieval result. The most
direct method for linking these perturbations to In-
formation Retrieval (IR) performance is through
document- or passage-level modifications. For in-
stance, removing a relevant document from the
ranked list demonstrably lowers standard evalua-
tion metrics such as precision, recall, and nDCG.
A similar effect is observed when swapping the
positions of a relevant and a non-relevant passage.
In contrast, perturbations at a finer granularity,
such as synonym replacement, present a more
complex scenario, as they may not inherently alter
a document’s underlying relevance. Other token-
level perturbations, including random noise injec-
tion or entity replacement, are even more challeng-
ing to evaluate using classical IR metrics, which
assumerelevancejudgmentsarebasedonunmod-
ified text. To address this, the relevance of a per-
turbed document can be estimated by computing
its similarity to the original version and applying
a threshold, providing a pseudo-relevance score.
Alternatively, LLM-as-a-judge methodologies (Gu
et al., 2025) offer a flexible approach. These tech-
niques effectively augment the original relevance
assessments, analogous to data augmentation
strategies used in computer vision (Szegedy et al.,
2014), thereby enabling a more comprehensive
evaluation of robustness under textual variations.
2.2. LLM Temperature Impact
LLMs generate token sequences using token logits
lkfor each token vk. The temperature modifies the
outputprobabilitiesofthetokenssothatthedistribu-
tionpeaks(resp. isflat)forlarge(resp. low)temper-

Target Category Specific Method Granularity Description and Application
DocumentSubsetRemoval Sentence-level Identifies a minimal subset of sentences whose removal
lowers the document’s rank beyond a thresholdk.
Application:Document ranking (Rorseth et al., 2023), QA
(Sudhi et al., 2024)
Combination Variable Identifies how combinations of elements influence results,
often via fixed-size random sampling.
Application:Open-book QA (Rorseth et al., 2024)
PermutationSource Reordering Passage-level Identifies the effect of source order by finding permuta-
tions that place relevant sources in high-attention posi-
tions.
Application:Open-book QA (Rorseth et al., 2024)
Word Reordering Word-level Alters the sequence of words within each source of the
input text.
Application:Mentioned in QA (Sudhi et al., 2024)
ReplacementUnit Replacement Sentence/Passage-level Replaces one sentence or passage at a time.
Application:Document ranking (Goren et al., 2020)
Entity Replacement Word-level Identifies entities (nouns, proper nouns) and replaces
them with random words.
Application:Mentioned in QA (Sudhi et al., 2024)
Antonym Replacement Word-level Replaces one or more words with their antonyms.
Application:Mentioned in QA (Sudhi et al., 2024)
Synonym Replacement Word-level Replaces one or more (important) words with their syn-
onyms.
Application:QA (Sudhi et al., 2024), Document ranking
(Wu et al., 2023)
InjectionRandom Noise Word-level Inserts different random words in and around the corre-
sponding source.
Application:Mentioned in QA (Sudhi et al., 2024), Pas-
sage ranking (Raval and Verma, 2020)
QueryAdditionPrefix Injection Token-level Insertionofashortprefixtothepromptleadstogeneration
of factually incorrect outputs.
Application:QA (Hu et al., 2024)
Term Augmentation Token-level Minimal perturbations to a search query that raise the
rank of a given document.
Application:Document ranking (Rorseth et al., 2023)
Table 1: Taxonomy of perturbation methods for evaluating Information Retrieval (IR) and Retrieval-
Augmented Generation (RAG) systems.
ature values. Then it also influences the sampling
ofthesetokensandthereforethewholegeneration.
High-temperaturevaluesaresupposedtoadddiver-
sity to generation: severalrunsof the same prompt
may generate very different responses. With the
notation of Renze (2024), the probability ofv k, us-
ing the temperature hyperparameterT, is:
p(vk) =elk/T
P
ieli/T(1)
OpenAIandDeepSeekAPI documentations
provide temperature recommendations for several
tasks without documented support for these val-
ues. However, studies (Renze, 2024) showed
that changes in sampling temperature from0 .0
to1.0do not produce statistically significant differ-
ences in problem-solving performance on multiple-
choice question-and-answer (MCQA) tasks across
multiple LLMs. Following a similar idea, we pro-
pose analyses and experiments of perturbations
for RAGs that take into account LLM’s temperature
variability.3. Methodology
We seek to estimate the impact, if any, of the LLM
temperature hyperparameter when perturbing a
RAG LLM input. We cope with the internal variabil-
ity coming from the LLM by presenting the same
prompt (perturbed or non-perturbed) several times.
Usingthis, ourmethodologyassessesthebehavior
of the perturbations along the temperature evolu-
tion. We compare each generated text by the LLM
with a processed ground-truth (see Section 4.3)
using classical semantic similarity measures and
compute the mean, variance and standard devia-
tion for the same prompt. We then build graphics
that present these comparisons.
This analysis allows us to determine: (i) whether
certain perturbation types consistently degrade
performance across all temperature values; (ii)
whether the effect of specific perturbations at
certain temperatures is statistically indistinguish-
able from the non-perturbed baseline; and (iii)
whether the relative impact of different perturba-
tions changes with temperature (e.g., if Perturba-
tion A has more impact than Perturbation B at a
low temperature, but less impact at a high temper-

ature).
4. Experiments
4.1. Dataset and Perturbations
We selected for our experiments the Hot-
potQA (Yang et al., 2018) dataset, dedicated to
question-answering (QA) systems that perform
complex reasoning and provide explanations for
their answers. It contains 113k Wikipedia-based
QA pairs2. This dataset was selected for three
key characteristics: (i) sentence-level supporting
facts that facilitate clean baseline establishment;
(ii) multi-hop QA structure requiring reasoning
across multiple documents, making it well-suited
for perturbation testing; and (iii) availability of
ground-truth answers for each query.
Furthermore, the classification of queries into
“bridge”and“comparison”typesenablestheinvesti-
gationofsystembehavioracrossdistinctreasoning
categories.Bridgequestions are those where, to
arrive at the answer, one must first identify a bridge
entity and then find the answer in relation to it. The
other type of multi-hop questions consists ofCom-
parisonquestions, which require comparing two
entities from the same category. A subset of these
comparison questions are yes/no questions.
More precisely, we utilized the training set of
thefullwikiversion of the HotpotQA dataset. After
analyzingthestatisticsofthedataset,werandomly
selected 100 samples for each category of facts
(2, 3 and 4 facts) and for each type of question
(“bridge” and “comparison”), resulting in a total of
600 samples for experimentation3.
In our experiments, we establish a baseline us-
ingalloriginalsupportingsentences. Buildingupon
this baseline, we systematically introduce three
types of perturbations, selected for their relevance
to real-world retrieval errors and alignment with es-
tablished evaluation frameworks such as RAG-Ex
(Sudhi et al., 2024). Our perturbation strategy in-
cludes4: (1)Sentence Replacement:replacing
thelatterportionofsupportingsentenceswithirrele-
vantsentencesfromthesametitle,whichsimulates
retrieval of correct entities with incorrect evidence,
acommonandrealisticfailuremodeinQAsystems;
(2)Sentence Removal:deleting the latter half of
2https://huggingface.co/datasets/
hotpotqa/hotpot_qa
3This sample size balances statistical reliability
againstcomputationalconstraints,givenourfine-grained
experimental design of 3 runs per (model, temperature,
perturbation, query) condition.
4For each sample, the number of altered sentences
was scaled by fact count: one sentence for 2-fact sam-
ples, one (33%) for 3-fact, and two for 4-fact samples.
All other supporting sentences remained unperturbed.supporting sentences; and (3)NER Replacement:
masking named entities in the last supporting sen-
tence(s) by replacing them with[MASK]tokens, fo-
cusing particularly on title-related entities to probe
model sensitivity.
This procedure generated three perturbed input
conditions in addition to the original baseline. The
resulting setup allows for a controlled investigation
of core perturbation effects against a stable refer-
ence point, establishing a reproducible framework
for future robustness studies.
4.2. Models and RAG Configuration
We conducted experiments with five LLMs, catego-
rized as follows:
•Proprietary GPT Models5:gpt-3.5-
turbo,gpt-4o;
•Open-Source LLaMA Models: Llama-
3.1-8B-Instruct6and Llama-3.2-1B-
Instruct7.
•DeepSeek reasoning model: deepseek-
reasoner8.
The chosen models (GPT-family, Llama-family,
and deepseek-reasoner) offer a strategically di-
verse mix of architectural families, parameter
scales,andcommercialvs. open-weightavailability,
allowing us to evaluate robustness across different
model types.
For each condition (model, temperature, pertur-
bation type), we executed the same query three
times to account for intrinsic stochasticity and to
help distinguish the effect of the model’s internal
noise(duetotemperature)fromthatofexternalper-
turbations. All other LLM hyperparameters were
set to their default values, except for max_tokens,
which is set to 1000.
4.3. Evaluation Methodology
Evaluation Metrics.While Exact Match (EM) and
F1 are widely adopted metrics, their limitations in
evaluating long-form generative outputs are well-
documented. In RAG settings, models frequently
produce elaborated answers containing correct
core information alongside supplementary expla-
nations. Consequently, EM scores may be artifi-
5https://platform.openai.com/docs/
models
6https://huggingface.co/meta-llama/
Llama-3.1-8B-Instruct , pretrained and fine-tuned
text models in 8B sizes.
7https://huggingface.co/meta-llama/
Llama-3.2-1B-Instruct ,pretrainedandinstruction-
tuned generative models in 1B sizes.
8https://api-docs.deepseek.com/guides/
reasoning_model . Before delivering the final answer,
the model first generates a Chain of Thought (CoT) to
enhance the accuracy of its responses.

cially low despite semantic correctness, and these
standard metrics often fail to capture subtle differ-
ences in perturbed answers. The F1 metric offers
greater robustness by rewarding token-level over-
lap, but unlike binary or multiple-choice QA, real-
world RAG systems generate free-form answers
requiring more nuanced evaluation.
Therefore,wereportsimilaritymetrics,whichbet-
ter reflect nuanced changes than exact matching.
For instance, BERTScore can detect minor per-
turbations, such as passive/active voice shifts or
small rewrites that retain the same meaning, while
being more sensitive to semantic alterations than
token-based metrics.
Reference Answer Processing.The reference
answers of HotpotQA are short, for example, “flew
in space” for thebridgequestion type, and “Yes” or
“No” for thecomparisonquestion type. Since our
framework uses a similarity measure to assess the
influence of temperature and perturbations on the
output, we transform the reference information into
sentence form by combining the original question
and the short answer. This is achieved using GPT-
4o(withdefaulthyperparameters)asthebackbone
model. The model is prompted with a combination
of the question and the candidate answer, using
the following template:
Prompt
Question: {question}
Answer: {answer}
Generate a complete and coherent an-
swer based on the given question and
answer, being as brief as possible:
Because the generated output are lengthy, we
extract only the first sentence — or the first two
if they began with “Yes” or “No” — as reference
answers for comparison.
Given that GPT-4o’s role was limited to the low-
complexity task of formatting answers (e.g., con-
verting “flew in space” to “Both X and Y are astro-
nautswhoflewinspace”),weobservedahighaccu-
racy. To verify this, we manually checked a subset
of the generated answers, including the final refer-
ence answers used for evaluation. A spot-check of
20 samples revealed zero hallucinations.
Metric Selection and Reporting.We evaluated
semantic similarity using both BERTScore (Zhang
et al., 2020) and ROUGE-1/2/L (Lin, 2004) metrics.
While all metrics exhibited consistent trends across
experimental conditions, we selected BERTScore
as our primary evaluation measure due to its better
alignment with human judgment in capturing se-
mantic equivalence. We report BERTScore F1 val-
ues computed using the default RoBERTa-large
model (Liu et al., 2019) as the backbone.5.Experimental Results and Analysis
5.1. Correctness Analysis
We analyze BERTScore trends across tempera-
ture settings for different models in Figure 2. For
eachexperimentalcondition,wecomputethemean
and standard deviation of scores across three runs,
then aggregate these values across all samples
per condition.
Our results reveal distinct model-specific tem-
perature sensitivity patterns. While deepseek-
reasoner maintains nearly invariant performance
across the temperature range, GPT models exhibit
degradation beginning at T= 1.4. In contrast,
Llamamodelsdemonstrateearlierperformancede-
terioration at T= 0.6, though with a more gradual
decline slope compared to GPT models’ sharper
descent.
Taking gpt-4o for example (third column graph-
ics from Figure 2), its results reveal that different
perturbation types exhibit varying sensitivity to tem-
perature increases.NER Replacementinduces
minimal degradation at T= 2.0, whereasSen-
tence ReplacementandSentence Removallead
to more substantial performance loss. Notably, all
perturbation types demonstrate amplified sensitiv-
itycomparedtobaselineconditionsastemperature
rises, suggesting that temperature acts as a perfor-
mance degradation amplifier.
Across temperature variation, we observe a
shifting performance hierarchy. At lower temper-
atures ( T < 1.4), GPT models achieve supe-
rior correctness (BERTScore: 0.95-0.97), followed
by Llama models (0.92-0.93) and deepseek-
reasoner (0.90). However, this ranking re-
verses at higher temperatures ( T= 2.0), where
deepseek-reasoner maintains consistent per-
formance while GPT models degrade below the
levels held by Llama models.
Question type (comparison vs. bridge) has min-
imal impact on temperature sensitivity, with both
typesexhibitingnearlyidenticaldegradationcurves
acrossallmodelsandperturbations. Thissuggests
temperature effects are largely orthogonal to ques-
tion complexity. However, absolute performance
is consistently higher for bridge questions than for
comparison questions across all models.
For deployment scenarios requiring temperature
tuning,werecommend: (1) deepseek-reasoner
for applications requiring consistent performance
across diverse temperature settings; (2) GPT mod-
els with temperature ceilings of T≤ 1.4to avoid
sharp performance cliffs; and (3) Llama models
with conservative temperature limits of T≤ 0.6to
maintain acceptable correctness levels.

Figure 2: BERTScore trends across temperature variations for different models, comparing response
types under perturbation. Solid lines represent mean scores across samples, while shaded areas denote
±standard deviation. The top row presents results for comparison questions; the bottom row presents
results for bridge questions.
Figure 3: Coefficient of Variation (CV) for BERTScore across models, temperatures, and perturbation
types. Each subplot displays CV trends for a model. The baseline CV value (average CV for the original,
unperturbed context across all temperatures) is indicated in the top left of each subplot. The top row
presents results for comparison questions; the bottom row presents results for bridge questions.
5.2. Output Variability Analysis
To quantify performance sensitivity, we employ the
Coefficient of Variation (CV), which measures rela-
tivevariabilitybynormalizingthestandarddeviation
against the mean performance. Figure 3 visual-
izes these results, with gray dotted lines indicating
each model’s stability baseline, calculated as the
average CV for the original (unperturbed) context
across all temperatures.
Our analysis reveals that temperature exerts a
stronger influence on output variability than pertur-
bation types across most models. However, Llama
models exhibit distinct behavior: Llama-3.2-1B-
Instruct shows no noticeable variations for com-
parison questions and bridge questions, whereasLlama-3.1-8B-Instruct exhibitsvariationthat
dependsonboththeperturbationtypeandthetem-
perature for bridge questions. GPT models demon-
strate particularly high temperature sensitivity, with
significant variability emerging at T≥ 1.4. In con-
trast, deepseek-reasoner and Llama models
maintain more consistent performance across the
temperaturerange. Forthe deepseek-reasoner
model,NER Replaceperturbations have the great-
estimpactoncomparisonquestions,whileallthree
perturbation types impact bridge questions, with
sensitivity emerging fromT≥0.2.
Furthermore, perturbations generally induce
greater variability in comparison questions than
in bridge questions across most models. The

baseline(unperturbed)conditionconsistentlyyields
the lowest CV for bridge questions in all models.
However, this pattern is not uniform for compar-
ison questions; for instance,Sentence Removal
resulted in the lowest CV value for the deepseek-
reasonermodel.
6. Qualitative Analysis of Model
Sensitivity
To complement our quantitative findings, we con-
ducted a qualitative analysis of model behavior un-
der varying temperatures and input perturbations.
We selected gpt-4o anddeepseek-reasoner
for this analysis based on their contrasting sensitiv-
ity profiles observed in previous experiments: with
gpt-4o demonstrating higher temperature sensi-
tivity and deepseek-reasoner showing greater
stability. We examined model outputs at two tem-
perature extremes: T= 0.6(representing more
deterministic generation) and T= 2.0(producing
more stochastic outputs).
6.1. BERTScore distributions
Figures 4 and 5 illustrate the BERTScore distri-
butions for bridge-type questions under different
perturbations. Temperature significantly impacts
output quality, particularly for gpt-4o. AtT= 2.0,
performance degrades across all perturbations,
with BERTScore values frequently falling between
0.70–0.80 and occasionally dropping below 0.70,
indicating increased output variability and reduced
semantic faithfulness at higher temperatures.
In contrast, deepseek-reasoner demon-
strates stability across temperature settings. While
T= 2.0introduces slightly greater score variance,
the median BERTScore remains consistent across
temperatures for each perturbation type, indicating
more robust generation under temperature varia-
tion.
6.2. Sample Analysis
Toidentifyrepresentativecasesofmodelsensitivity,
we selected, for each model studied in Section
6.1, temperature, question type, and perturbation
type, the sample with the largest BERTScore gap
between original and perturbed conditions. This
method highlights key fragility patterns.
Perturbation-Type Analysis.Our examina-
tion of these cases reveals distinct failure modes
across perturbation types.Sentence Replacement
andSentence Removalperturbations frequently
trigger model refusal behaviors, with responses
such as “The retrieved document does not pro-
vide...” becoming common9. At higher tempera-
9For example, gpt-4o withSentence Replacement
Figure 4: BERTScore distribution for gpt-4o on
bridge questions across perturbation types at two
temperatures (Left: T= 0.6, Right: T= 2.0). Each
subplot shows a boxplot representing median, in-
terquartilerange,andwhiskers,withindividualsam-
ple scores (black dots) and outliers (white dots).
Figure 5: BERTScore distribution for deepseek-
reasoner onbridgequestionsacrossperturbation
types at two temperatures (Left: T= 0.6, Right:
T= 2.0). Each subplot shows a boxplot represent-
ing the same elements as in Fig. 4.
tures ( T= 2.0), these perturbations often result
in garbled or nonsensical outputs containing code
atT= 0.6output: “The retrieved document does not
provide specific information about the campus sizes
of Indiana University or Ohio State University to deter-
mine which has the third-largest university campus in
the United States. To accurately answer the query, more
detaileddataonthecampussizesorstudentpopulations
of both universities is required.”

snippets, random tokens, and mixed languages.
NER Replacementperturbations prove effective at
disrupting model performance, causing failures in
entity recognition and relationship inference that
lead to incomplete or incorrect answers.
Temperature Effects on Output Quality.Tem-
perature settings influence how models degrade
under perturbation. At T= 2.0, we observe severe
output degradation characterized by nonsense and
complete failure to address the query. In contrast,
atT= 0.6, models demonstrate greater robust-
ness, though they still exhibit cautious response
patterns (e.g., “I cannot determine...”), partial an-
swers,andoccasionalfactualerrors. Thissuggests
thatwhilelowertemperaturesimprovestability,they
do not eliminate sensitivity to perturbations.
Question-Type Sensitivity.Bridge questions
show particular sensitivity to entity removal or re-
placement, likely due to their reliance on connect-
ing information across multiple facts. Comparison
questions,whilestillaffected,occasionallymaintain
correctness through external knowledge utilization,
suggesting different reasoning pathways may ex-
hibit varying robustness.
Model-Specific Degradation Patterns.The
two models exhibit distinct failure characteristics.
gpt-4o typically produces fluent but incorrect
responses under perturbation, maintaining co-
herence while sacrificing accuracy. deepseek-
reasoner , conversely, often fails more gracefully
withconcisebutincompleteanswers(e.g.,respond-
ing with single words like “Brewery” rather than
generating nonsensical text). This difference likely
stems from their distinct training objectives; as a
reasoning model, deepseek-reasoner may pri-
oritize logical coherence and conciseness over
the discursive fluency characteristic of a general-
purpose model like gpt-4o, a hypothesis that mer-
its further investigation.
Robustness Insights.Despite overall sensitiv-
ity patterns, we observe instances where models
maintaincorrectnessunderperturbation, indicating
some degree of inherent robustness or effective
internal knowledge utilization. The significant per-
formance variability across samples suggests that
certain question structures or knowledge domains
are inherently more fragile than others.
7. Discussion and Conclusion
This study investigated the relative impact of inter-
nal temperature versus external perturbations on
RAG system performance. Our analysis reveals
that temperature introduces a more pronounced
influence on model correctness than specific per-
turbationtypes, withperformancedegradingsignifi-
cantlyabovecertaintemperaturethresholdsacross
most tested models and perturbation conditions.The interaction between temperature and pertur-
bations proves particularly critical: while models
demonstraterelativerobustnesstoperturbationsat
lower temperatures ( T≤ 0.6), they exhibit severe
performance degradation under the same pertur-
bationsathighertemperatures( T≥ 1.4). Thisjoint
effect creates a fragility landscape where systems
that appear stable under standard evaluation con-
ditions can fail dramatically when facing real-world
noise combined with typical sampling strategies.
Notably, we observed instances where models
maintained correctness despite substantial pertur-
bations, suggesting utilization of internal knowl-
edge rather than strict reliance on retrieved doc-
uments. However, the unpredictable nature of this
phenomenon, where models sometimes bypass
corrupted context entirely but other times produce
confidently wrong responses, highlights the chal-
lenge of determining when and how internal knowl-
edge mechanisms activate in RAG settings.
Our findings carry implications for RAG deploy-
ment. From a temperature perspective, we demon-
strate that this hyperparameter must be carefully
calibrated alongside perturbation robustness con-
siderations. From a retrieval perspective, our re-
sults reinforce the importance of filtering uncer-
tain or irrelevant content, aligning with principles in
active retrieval methods like FLARE (Jiang et al.,
2023). Based on our comprehensive evaluation,
we propose the following deployment strategies:
deepseek-reasoner for applications requiring
consistent performance across diverse tempera-
ture settings; configure GPT models with a temper-
atureceilingof T≤ 1.4toavoidsharpperformance
degradation; and employ Llama models with a con-
servative temperature limit of T≤ 0.6to maintain
acceptable correctness levels.
The main contribution of this work is to reveal
the critical yet overlooked interaction between inter-
nal and external noise sources in RAG systems. A
system that performs well on conventional bench-
marks may prove surprisingly fragile when facing
the combined effects of sampling stochasticity and
real-worlddocumentperturbations. Toaddressthis
gap, we introduce a dedicated benchmark and an-
alytical framework designed to quantify this joint
effect. We note that the present study isolates the
LLM generator’s sensitivity by perturbing gold con-
texts,therebycontrollingforretrievalnoise. Afuture
directionistoincorporateactualretrievalsystemsto
examine how retrieval inaccuracies and generation
sensitivity compound in end-to-end pipelines.
Acknowledgments
This work was partially funded by the “Intelligent
SystemsforData,Knowledge,andHumans”axisof
the Grenoble Computer Science Laboratory (LIG).

It was also conducted within the framework of the
AugmentIAChair,ledbyDidierSchwabandhosted
by the Grenoble INP Foundation, thanks to the
patronage of the Artelia Group. The chair also re-
ceives support from the French government, man-
aged by the National Research Agency (ANR) un-
der the France 2030 program with reference num-
ber ANR-23-IACL-0006 (MIAI Cluster).
8. Bibliographical References
Avishek Anand, Lijun Lyu, Maximilian Idahl, Yu-
meng Wang, Jonas Wallat, and Zijian Zhang.
2022. Explainable information retrieval: A sur-
vey.
Weihua Du, Yiming Yang, and Sean Welleck. 2025.
Optimizingtemperatureforlanguagemodelswith
multi-sample inference.
GregoryGoren, OrenKurland, MosheTennenholtz,
and Fiana Raiber. 2020. Ranking-incentivized
quality preserving content modification. InPro-
ceedings of the 43rd International ACM SIGIR
Conference on Research and Development in
Information Retrieval, SIGIR ’20, page 259–268,
New York, NY, USA. Association for Computing
Machinery.
Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan,
XuehaoZhai,ChengjinXu,WeiLi,YinghanShen,
Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun
Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni,
and Jian Guo. 2025. A survey on llm-as-a-judge.
Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes,
andYejinChoi.2020. Thecuriouscaseofneural
text degeneration. In8th International Confer-
ence on Learning Representations, ICLR 2020,
Addis Ababa, Ethiopia, April 26-30, 2020. Open-
Review.net.
Zhibo Hu, Chen Wang, Yanfeng Shu, Hye-Young
Paik,andLimingZhu.2024. Promptperturbation
in retrieval-augmented generation based large
language models. InProceedings of the 30th
ACMSIGKDDConferenceonKnowledgeDiscov-
ery and Data Mining, KDD ’24, page 1119–1130,
New York, NY, USA. Association for Computing
Machinery.
ZhengbaoJiang,FrankXu,LuyuGao,ZhiqingSun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active re-
trieval augmented generation. InProceedings
of the 2023 Conference on Empirical Methods
in Natural Language Processing, pages 7969–
7992, Singapore. Association for Computational
Linguistics.Patrick Lewis, Ethan Perez, Aleksandra Piktus,
FabioPetroni,VladimirKarpukhin,NamanGoyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe
Kiela. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks. InAdvances
in Neural Information Processing Systems, vol-
ume 33, pages 9459–9474. Curran Associates,
Inc.
Chin-Yew Lin. 2004. ROUGE: A package for auto-
matic evaluation of summaries. InText Summa-
rization Branches Out, pages 74–81, Barcelona,
Spain. Association for Computational Linguistics.
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du,
Mandar Joshi, Danqi Chen, Omer Levy, Mike
Lewis, Luke Zettlemoyer, and Veselin Stoyanov.
2019. Roberta: A robustly optimized bert pre-
training approach.
Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten
deRijke,YixingFan,andXueqiCheng.2024. Ro-
bust neural information retrieval: An adversarial
and out-of-distribution perspective.
Antonin Poché, Lucas Hervier, and Mohamed-
Chafik Bakkay. 2023. Natural example-based
explainability: A survey. InExplainable Artifi-
cial Intelligence, pages 24–47, Cham. Springer
Nature Switzerland.
NisargRavalandManishaVerma.2020. Oneword
atatime: adversarialattacksonretrievalmodels.
Matthew Renze. 2024. The effect of sampling
temperature on problem solving in large lan-
guage models. InFindings of the Association for
Computational Linguistics: EMNLP 2024, pages
7346–7356, Miami, Florida, USA.Associationfor
Computational Linguistics.
Joel Rorseth, Parke Godfrey, Lukasz Golab, Mehdi
Kargar,DiveshSrivastava,andJaroslawSzlichta.
2023. Credence: Counterfactualexplanationsfor
document ranking.2023 IEEE 39th International
Conference on Data Engineering (ICDE), pages
3631–3634.
JoelRorseth,ParkeGodfrey,LukaszGolab,Divesh
Srivastava, and Jaroslaw Szlichta. 2024. Rage
against the machine: Retrieval-augmented llm
explanations.
Viju Sudhi, Sinchana Ramakanth Bhat, Max Rudat,
and Roman Teucher. 2024. Rag-ex: A generic
framework for explaining retrieval augmented
generation. InProceedings of the 47th Inter-
national ACM SIGIR Conference on Research
andDevelopmentinInformationRetrieval, SIGIR
’24, page 2776–2780, New York, NY, USA. As-
sociation for Computing Machinery.

Christian Szegedy, Wei Liu, Yangqing Jia, Pierre
Sermanet, Scott Reed, Dragomir Anguelov, Du-
mitru Erhan, Vincent Vanhoucke, and Andrew
Rabinovich. 2014. Going deeper with convolu-
tions.
Chen Wu, Ruqing Zhang, Jiafeng Guo, Maarten
De Rijke, Yixing Fan, and Xueqi Cheng. 2023.
Prada: Practical black-box adversarial attacks
against neural ranking models.ACM Trans. Inf.
Syst., 41(4).
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q.
Weinberger, and Yoav Artzi. 2020. Bertscore:
Evaluating text generation with bert. InInterna-
tional Conference on Learning Representations.
Yongxin Zhou, Philippe Mulhem, and Didier
Schwab. 2025. Explicabilité par perturbations
pour les systèmes RAG. InActes de l’atelier Ac-
cèsàl’informationbasésurledialogueetgrands
modèles de langage 2025 (DIAG-LLM), pages
1–6, Marseille, France. ATALA \\& ARIA.
9. Language Resource References
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua
Bengio, William Cohen, Ruslan Salakhutdinov,
and Christopher D. Manning. 2018. HotpotQA: A
dataset for diverse, explainable multi-hop ques-
tion answering. InProceedings of the 2018 Con-
ference on Empirical Methods in Natural Lan-
guage Processing, pages 2369–2380, Brussels,
Belgium. Association for Computational Linguis-
tics.
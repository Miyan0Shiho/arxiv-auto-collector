# IBPS: Indian Bail Prediction System

**Authors**: Puspesh Kumar Srivastava, Uddeshya Raj, Praveen Patel, /Shubham Kumar Nigam, Noel Shallum, Arnab Bhattacharya

**Published**: 2025-08-11 03:44:17

**PDF URL**: [http://arxiv.org/pdf/2508.07592v1](http://arxiv.org/pdf/2508.07592v1)

## Abstract
Bail decisions are among the most frequently adjudicated matters in Indian
courts, yet they remain plagued by subjectivity, delays, and inconsistencies.
With over 75% of India's prison population comprising undertrial prisoners,
many from socioeconomically disadvantaged backgrounds, the lack of timely and
fair bail adjudication exacerbates human rights concerns and contributes to
systemic judicial backlog. In this paper, we present the Indian Bail Prediction
System (IBPS), an AI-powered framework designed to assist in bail
decision-making by predicting outcomes and generating legally sound rationales
based solely on factual case attributes and statutory provisions. We curate and
release a large-scale dataset of 150,430 High Court bail judgments, enriched
with structured annotations such as age, health, criminal history, crime
category, custody duration, statutes, and judicial reasoning. We fine-tune a
large language model using parameter-efficient techniques and evaluate its
performance across multiple configurations, with and without statutory context,
and with RAG. Our results demonstrate that models fine-tuned with statutory
knowledge significantly outperform baselines, achieving strong accuracy and
explanation quality, and generalize well to a test set independently annotated
by legal experts. IBPS offers a transparent, scalable, and reproducible
solution to support data-driven legal assistance, reduce bail delays, and
promote procedural fairness in the Indian judicial system.

## Full Text


<!-- PDF content starts -->

IBPS: Indian Bail Prediction System
Puspesh Kumar Srivastava1∗Uddeshya Raj1∗Praveen Patel1∗,
Shubham Kumar Nigam1∗†Noel Shallum2Arnab Bhattacharya1
1IIT Kanpur, India2Symbiosis Law School Pune, India
{puspeshk24, uddeshya24, praveenp24, arnabb}@iitk.ac.in
shubhamkumarnigam@gmail.com noelshallum@gmail.com
Abstract
Bail decisions are among the most frequently
adjudicated matters in Indian courts, yet they re-
main plagued by subjectivity, delays, and incon-
sistencies. With over 75% of India’s prison pop-
ulation comprising undertrial prisoners, many
from socioeconomically disadvantaged back-
grounds, the lack of timely and fair bail ad-
judication exacerbates human rights concerns
and contributes to systemic judicial backlog.
In this paper, we present the Indian Bail Pre-
diction System (IBPS), an AI-powered frame-
work designed to assist in bail decision-making
by predicting outcomes and generating legally
sound rationales based solely on factual case at-
tributes and statutory provisions. We curate and
release a large-scale dataset of 150,430 High
Court bail judgments, enriched with structured
annotations such as age, health, criminal his-
tory, crime category, custody duration, statutes,
and judicial reasoning. We fine-tune a large
language model using parameter-efficient tech-
niques and evaluate its performance across mul-
tiple configurations, with and without statutory
context, and with RAG. Our results demon-
strate that models fine-tuned with statutory
knowledge significantly outperform baselines,
achieving strong accuracy and explanation qual-
ity, and generalize well to a test set indepen-
dently annotated by legal experts. IBPS offers a
transparent, scalable, and reproducible solution
to support data-driven legal assistance, reduce
bail delays, and promote procedural fairness in
the Indian judicial system.
1 Introduction
India’s criminal justice system is under immense
strain, with bail-related proceedings constituting a
significant share of the case backlog in lower courts.
As of December 20221, more than 75% of India’s
prison population comprises undertrial prisoners,
∗These authors contributed equally to this work
†Corresponding author
1NCRB Report, 2022individuals not yet convicted of any offence. This
alarming statistic is symptomatic of a larger sys-
temic inefficiency, where bail applications, though
intended as a swift relief mechanism, often face
substantial delays in adjudication. In parallel, the
country’s overall incarceration rate has surpassed
131% of the sanctioned prison capacity, reflecting
a crisis that disproportionately affects individuals
from socioeconomically disadvantaged communi-
ties.
The root of this issue lies in the burdened ju-
dicial pipeline: bail hearings, especially in subor-
dinate courts, are among the most common legal
proceedings. According to recent estimates, a large
proportion of criminal case pendency at the district
and high court levels arises from bail applications.
Despite their frequency and routine nature, these
hearings often experience substantial delay. For
instance, the median disposal time for bail in some
High Courts like Jammu & Kashmir is 156 days,
while even the best-performing courts often take
over three weeks on average for regular bail and
more than a month for anticipatory bail. Such de-
lays not only extend unjust pretrial incarceration
but also intensify court pendency and procedural
bottlenecks.
In response to this critical challenge, we in-
troduce the Indian Bail Prediction System
(IBPS) , a comprehensive AI-based framework de-
signed to assist in the interpretation and adjudica-
tion of bail cases. IBPS is motivated by the urgent
need for scalable, transparent, and legally grounded
decision-support tools that can aid courts and le-
gal professionals in expediting routine bail appli-
cations. By leveraging structured case attributes,
statutory context, and explainable rationale genera-
tion, IBPS aspires to complement human judgment
rather than replace it, thereby preserving judicial
discretion while enhancing efficiency.
Our work makes several contributions. First,
we present a large-scale curated dataset of over
1arXiv:2508.07592v1  [cs.CL]  11 Aug 2025

150,000 Indian High Court bail judgments, anno-
tated with key features such as facts, statutes, past
records, health conditions, and outcomes. This
represents the largest factual dataset in this le-
gal subdomain. Second, we develop a suite of
LLM-based models, including fine-tuned variants
and Retrieval-Augmented Generation (RAG) se-
tups, that predict the outcome of a bail decision
(granted/rejected) and generate human-readable
justifications grounded in legal reasoning. Third,
we conduct rigorous evaluation using both auto-
matic metrics and expert-assessed annotations, es-
tablishing strong baselines for factual and statute-
aware bail judgment prediction.
Ultimately, IBPS aims to reduce judicial burden,
promote procedural fairness, and advance access
to justice through responsible AI intervention. By
enabling timely, explainable, and legally consistent
decision support for routine bail matters, this work
lays the foundation for future legal-AI systems in
India’s high-stakes judicial landscape.
Our Contributions:
•We curate and release the largest fact-based bail
judgment dataset in the Indian legal system, com-
prising over 150,000 High Court bail cases anno-
tated with structured attributes such as statutes,
health status, past criminal records, custody du-
ration, and judgment outcomes.
•We design and implement the IBPS, a framework
that leverages instruction-tuned large language
models to perform both outcome prediction and
rationale generation, supported by legal context
through RAG.
•We introduce a test set manually annotated by
legal experts, enabling fine-grained evaluation of
our model’s predictive and explanatory capabili-
ties against human performance.
•We conduct a detailed experimental analysis
across six configurations, evaluating both factual
accuracy and reasoning quality using automatic
metrics, and demonstrate that models fine-tuned
with statutory context significantly outperform
baselines.
To ensure reproducibility and encourage further
research, the dataset and model code will be made
publicly available soon.
2 Related Work
The intersection of Artificial Intelligence and law
(AI4Law) has rapidly evolved over the past two
decades, fueled by the digitization of legal recordsand advancements in natural language processing
(NLP). Legal AI research spans multiple tasks such
as legal judgment prediction (LJP), statute retrieval,
legal question answering, contract analysis, and le-
gal summarization. Among these, LJP has received
significant attention for its ability to support judi-
cial decision-making. Early efforts in this domain
began with statistical and rule-based models, no-
tably the work of Aletras et al. (2016) on the Euro-
pean Court of Human Rights (ECHR). Subsequent
benchmarks like EURLEX57K (Chalkidis et al.,
2019), CAIL2018 (Xiao et al., 2018), SwissJudg-
ment (Niklaus et al., 2021), JTD (Yamada et al.,
2024), and LegalBench (Guha et al., 2023) have
enabled large-scale legal reasoning tasks across ju-
risdictions. However, most such datasets are built
for civil law systems, limiting their adaptability to
common law jurisdictions like India. Indian legal
texts present unique challenges, length, linguistic
complexity, and a reliance on precedent, yet re-
main under-resourced in Legal NLP. While studies
like Zadgaonkar and Agrawal (2021), Bansal et al.
(2022), and Sharma et al. (2021) applied classical
ML models to Indian legal cases, they often lacked
expert annotations and interpretability. In the U.S.,
a substantial body of research has examined bail
and pretrial detention outcomes (Sacks et al., 2015;
Demuth and Steffensmeier, 2004), highlighting
their long-term consequences on conviction and
sentencing. Yet, this critical decision point remains
underexplored in India. Our work addresses these
gaps by introducing an expert-annotated dataset
for Indian bail judgments and combining judgment
prediction with rationale generation. Furthermore,
recent advances in large language models (LLMs),
such as BERT, LLaMA, and GPT, have demon-
strated impressive capabilities in legal text under-
standing and generation (Chalkidis et al., 2021; Ye
et al., 2018), though concerns around hallucination,
factual grounding, and legal coherence persist.
3 Task Description
Our research focuses on the task of Bail Prediction,
which comprises two sequential components: (i)
predicting the outcome of a bail application, and (ii)
generating an explanation to justify the predicted
outcome. This task reflects a realistic legal scenario
where AI systems must not only make decisions
but also provide human-understandable reasoning,
thereby improving transparency and aiding legal
practitioners. The task spans multiple types of bail-
2

related applications under the Indian legal system,
including Regular Bail ,Anticipatory Bail , and Bail
Cancellation appeals.
Prediction Task: The first component involves de-
termining whether the bail is granted or not, given
the details of the case. Each bail case contains a
combination of factual narratives, legal charges,
prior criminal records, medical and personal condi-
tions of the accused, and legal arguments presented
by the defense and prosecution. While many cases
involve multiple accused, this task considers pre-
diction for a single accused at a time.
Formal Definition: LetDR,DA, andDCbe the
sets of documents corresponding to Regular Bail ,
Anticipatory Bail , and Bail Cancellation cases, re-
spectively. Given a document D∈ DR∪DA∪DC,
the goal is to predict a binary outcome y∈ {0,1},
where:
•y= 0: denotes “Bail not granted” ifD∈ DR∪
DA, and “Bail not cancelled” ifD∈ DC.
•y= 1: denotes “Bail granted” ifD∈ DR∪DA,
and“Bail cancelled” ifD∈ DC.
Explanation Task: The second component is to
generate a natural language explanation for the pre-
dicted outcome. This explanation should cite rele-
vant factual and legal grounds present in the doc-
ument and reflect the reasoning typically found in
judicial decisions. This component is particularly
important for improving trust and interpretability
in automated legal decision-making systems.
4 Data Preparation
4.1 Raw Data Collection
We began with the Daksh database2, which pro-
vides metadata on bail cases from 15 Indian High
Courts, including CNR numbers, case numbers,
statutes, filing and judgment dates, and bail out-
comes. This metadata covered 927,897 cases, and
we used it to extract corresponding full-text judg-
ments from the eCourts High Court portal3using
custom Python scripts.
From these, we selected five High Courts, Bom-
bay, Kerala, Allahabad, Chhattisgarh, and Jhark-
hand, based on volume and balance across bail
categories. For each selected case, we retrieved the
most recent and complete order if multiple were
listed. In total, we downloaded 208,983 bail judg-
ments covering a diverse set of offenses, from petty
crimes to serious offenses like murder, rape, and
2Daksh database
3eCourts High Court portalcybercrimes. Additionally, we scraped official PDF
documents from India Code4for Indian Penal Code
(IPC), Criminal Procedure Code (CrPC), and Cen-
tral/State Acts to support legal context retrieval in
our RAG-based systems.
4.2 Feature Extraction
Judgment documents are unstructured and vary sig-
nificantly across jurisdictions, time periods, and
even individual judges. From each document, we
aimed to extract the following features: statutes ,
factual narrative ,legal arguments ,past criminal
record ,health condition ,case outcome ,reasoning ,
andcustody duration (calculated from arrest and
judgment dates).
Evaluation of NLP Models. We initially ex-
plored classical NLP models for named entity
recognition and rhetorical role extraction:
• en_legal_ner_trf (Kalamkar et al., 2022)
•BiLSTM-CRF and MTL models trained on
LegalSeg (Nigam et al., 2025)
• Our SparkNLP-based custom NER model using
the InLegalNER5dataset
•Rule-based Python scripts using fuzzy matching
and regex
As detailed in Table 4, none of these models met
our performance expectations, especially on shorter
bail judgments.
Evaluation of LLMs. We experimented with
multiple open-source LLMs for one-shot and few-
shot extractions over 100 manually verified cases.
Inference was run on an NVIDIA L40S GPU
(45GB VRAM). Table 1 shows the results of this
evaluation.
Model Inference Time (s) GPT-eval Score (/10)
google/gemma-3-12b-it 188 6.81
meta-llama/Llama-3.1-8B-Instruct 70 6.89
microsoft/phi-4 101 7.56
mistralai/Mistral-7B-Instruct-v0.3 136 6.95
deepseek-ai/DeepSeek-R1-0528-Qwen3-8B 139 6.09
bharatgenai/Param-1-2.9B-Instruct — —
Table 1: Comparison of LLMs on feature extraction
quality and efficiency
Phi-4 emerged as the most balanced model in
terms of accuracy, format consistency, and speed.
It was the only model that consistently returned
well-structured outputs aligned with our extrac-
tion schema. Other models, including LLaMA-3
and Mistral, performed reasonably well, but often
4India Code
5InLegalNER
3

failed to adhere to formatting or hallucinated out-
puts. Due to its superior performance, Phi-4 was
selected as the backbone model for large-scale fea-
ture extraction.We had also tried PARAM model
but it was constantly crashing due to less context
size (2048 tokens) which was causing it to crash
while processing longer cases.
Extraction Strategy. We tested zero-shot, one-
shot, and few-shot prompting strategies. Zero-shot
prompts resulted in the poorest performance, pro-
ducing incomplete or misformatted outputs. One-
shot prompting, which included a structured ex-
ample with the case text, produced significantly
more accurate and reliable extractions. Few-shot
prompting marginally improved quality but was
constrained by context length (especially for long
judgments) and higher token costs. Thus, we
adopted the one-shot strategy for all large-scale
extractions.
4.3 Data Cleaning
We processed metadata for over 350,000 cases,
of which approximately 250,000 had judgment
texts available on the eCourts portal. After elim-
inating documents lacking factual details (e.g.,
those with only a binary grant/reject order), we
extracted structured information from 208,293 doc-
uments. Documents missing critical elements like
incident details, statutes, judgment reasoning, or
outcomes were also discarded, resulting in a fi-
nal usable dataset of 150,430 records. Once the
LLM-generated structured outputs were available,
we parsed them using regular expressions and cus-
tom Python scripts. Final features were validated
and stored in standardized JSON format for down-
stream tasks.
4.4 Test Dataset Creation
To evaluate our models against expert-verified
ground truth, we constructed a high-quality test
dataset through manual legal annotation. We ran-
domly sampled 100 cases from the cleaned dataset,
ensuring balanced representation across the five se-
lected High Courts and all three categories of bail
applications, Regular Bail, Anticipatory Bail, and
Bail Cancellation.
Three legal experts were engaged to annotate
each case. For each judgment, they independently
extracted the same structured features used in
model training: application type, age, health condi-
tion, criminal history, statutes, arguments, case out-Statistic Train Val. Test
Total # cases 120,345 15,042 15,043
Total # sentences 1,465,310 182,491 183,283
Avg. sentences / case 12.18 12.13 12.18
Avg. tokens / sentence 28.19 28.26 28.15
Sentence Count per Label
Details of the incident 350,499 43,778 43,894
Arguments supporting application 331,748 41,123 41,580
Arguments opposing application 192,479 24,249 23,873
Bail conditions 343,967 43,000 43,168
Reasoning 246,617 30,341 30,768
Statutory Context 9,12,204 1,14,227 1,13,061
Avg. Tokens per Label
Details of the incident 29.08 29.10 29.16
Arguments supporting application 24.69 24.82 24.64
Arguments opposing application 18.32 18.42 18.35
Bail conditions 36.72 36.78 36.51
Reasoning 27.43 27.48 27.30
Statutory Context 106.49 107.02 106.07
Table 2: Dataset Statistics
come, judicial reasoning, arrest and judgment dates.
The experts were also asked to clearly state the pre-
dicted outcome and provide a rationale, mimicking
the expected LLM output format. This curated
subset serves as our gold-standard Test Dataset ,
allowing us to evaluate both the prediction accu-
racy and the quality of generated rationales through
comparison with expert-provided responses.
5 Dataset Analysis
This study is based on a comprehensive dataset
of bail judgments collected from five major High
Courts in India, Chhattisgarh, Bombay, Jhark-
hand, Kerala, and Allahabad, comprising a total
of 208,292 bail case records. After rigorous clean-
ing and filtering for completeness and usability,
we retained 150,430 judgments for analysis. Fig-
ure 9 shows the distribution of cases across these
courts.The complete statistics of the dataset can be
found in Table 2.
The final dataset contains 13 structured variables,
including case number, bail type, outcome, with-
drawal status, age, health condition, criminal his-
tory, involved statutes, arrest and judgment dates,
judgment length, detailed case facts, and derived
crime category. We preprocessed the data by replac-
ing missing values (e.g., none ,nan) with NaN, con-
verting age into numeric and categorical formats,
calculating custody duration from date differences,
and parsing statutes. Furthermore, we used FLAN-
T5 to semantically classify crimes (e.g., theft, mur-
der, fraud) based on case details. While some at-
tributes were incomplete, only 34.2% had valid age
4

and 28.6% had arrest dates, the dataset provided
sufficient information to uncover robust patterns.
The age of the accused ranged from 11 to 95
years, with a mean of 37.3 and standard deviation
of 13.1. The most represented group was aged
30–50, followed by 18–30 and 50–65. A notewor-
thy pattern emerged linking age to bail outcomes.
As shown in Appendix Figure 4, older individuals
had significantly higher bail grant rates, with appli-
cants aged 65 and above receiving bail in 84.1% of
cases compared to only 67.7% for the 18–30 age
group.
The dataset includes three major types of bail:
Anticipatory (53.7%), Regular (45.3%), and Bail
Cancellation (1.0%). Figure 10 illustrates the pro-
portion of each category. Regular bail applications
exhibited the highest grant rate at 76.7%, while
anticipatory bail stood at 72.9%. Bail cancellations
had a distinct distribution, with 60.7% resulting in
“not cancelled” decisions, as detailed in Figure 11.
A particularly influential variable was criminal
history. Applicants with no prior record had a
74.6% success rate, while those with a record re-
ceived bail in only 50.8% of cases, a 23.8 percent-
age point drop, as shown in Figure 12.
Examining statute-wise bail patterns revealed
that frequently cited provisions like Section 506
(criminal intimidation), Section 420 (cheating), and
Section 34 (common intention) had high grant rates
(above 75%), as depicted in Figure 1. Conversely,
socially sensitive statutes such as Section 18 of the
SC/ST Act and Section 64 of the Abkari Act had
some of the lowest or even zero grant rates (see
Appendix Figure 5).
Figure 1: Top statutes and their corresponding bail out-
comes.
Temporal analysis based on 42,654 cases with
complete arrest and judgment dates revealed sub-
stantial delays in pretrial detention. The average
custody duration was approximately 254.6 days
(median: 100 days), with a maximum of over 17years in one extreme case (Figure 2).
Figure 2: Custody duration distribution before judg-
ment.
Crime-wise outcome analysis (Figure 3) showed
that non-violent crimes such as theft and domestic
violence had high grant rates (83.5% and 82.9%
respectively), whereas violent crimes like rape and
murder had significantly lower success rates, 62.4%
and 65.2% respectively. Appendix Figure 8 fur-
ther illustrates the age distribution across different
crime categories, revealing that youth were dispro-
portionately involved in crimes like kidnapping and
assault, while middle-aged individuals dominated
in fraud and drug-related offenses.
Figure 3: Bail outcomes across crime categories.
Finally, custody duration varied significantly by
crime type. Figure 13 shows that murder cases had
the longest pretrial detentions, with 24.5% of such
cases extending beyond two years, while drug of-
fenses were typically resolved within 30–90 days.
In rare instances, bail applications were with-
drawn, only 1.3% of cases (2,020 out of 150,430),
suggesting that most applications proceed to judi-
5

cial determination rather than being dropped volun-
tarily.
6 Methodology
This section describes the design of our fine-tuning
and evaluation pipeline for the task of bail judg-
ment prediction and rationale generation. Our aim
is to assess how different levels of legal knowledge
and supervision influence the decision-making and
explanatory capabilities of LLMs.
6.1 Model Fine-Tuning Strategy
We base all experiments on microsoft/phi-4 , a
compact yet highly capable LLM. Using the PEFT
framework with QLoRA, we fine-tune two variants
of this model:
•FT-1 (Case-Aware Model) : This variant is fine-
tuned using structured case data as input and is
trained to output both the bail outcome and a
natural language explanation.
•FT-2 (Case+Statute-Aware Model) : In addition
to structured case data, this variant also receives
textual descriptions of the applicable statutes (re-
trieved from India Code) during training. The
model is jointly optimized to generate the out-
come, explanation, and optionally reproduce the
legal context.
These fine-tuned models are tested with and
without RAG-based statutory context during infer-
ence. Thus, we evaluate six distinct configurations
described below.
6.2 Experimental Configurations
Setup 1: Baseline (Base Model, No RAG)
We use the unmodified base phi-4 model with only
the structured case input, no fine-tuning, no statute
context. Significance: This setting tests the general
legal reasoning ability of the pre-trained model in
a zero-shot setting. It serves as the foundational
reference point to measure gains from both fine-
tuning and retrieval augmentation.
Setup 2: Base Model + RAG (Statutory Con-
text)
The same base model is tested, but with the addition
of retrieved statutory context via an RAG pipeline.
Significance: This setup evaluates whether access
to legal definitions and statutory language improves
prediction and explanation in a zero-shot setting.
It helps determine the base model’s ability to inte-
grate external legal knowledge on the fly.
Setup 3: FT-1 (Fine-Tuned on Case Data)
The FT-1 model is trained using only structuredcase data as input. No statute definitions are used in
training or inference. Significance: This configura-
tion tests the LLM’s ability to generalize from past
case patterns and learn case-specific priors without
requiring legal text understanding. It isolates the
effect of case-only training.
Setup 4: FT-2 (Fine-Tuned with Statutory
Context)
The FT-2 model is trained on both structured case
data and the accompanying statutory descriptions.
It is evaluated without RAG at inference. Sig-
nificance: This setup assesses whether explicit
legal knowledge, embedded during training, en-
hances the model’s predictive accuracy and its abil-
ity to generate legally sound explanations. It also
helps verify if statute-aware training improves in-
terpretability.
Setup 5: FT-1 + RAG (Case-Aware, Inference
with Statutes)
The FT-1 model (trained without statutes) is tested
with retrieved statutory definitions at inference time
via RAG. Significance: This tests whether a case-
trained model can effectively leverage unseen legal
context during inference, even if it was not part
of its training data. It serves as a hybrid setting
between fine-tuning and retrieval-based augmenta-
tion.
Setup 6: FT-2 + RAG (Case+Statute-Aware,
RAG)
The most comprehensive configuration: FT-2
(trained with statutes) is tested with statute context
via RAG at inference time. Significance: This setup
combines learned legal knowledge and retrieval-
based support to simulate the most realistic, high-
performance setting. It helps determine whether
retrieval complements or overlaps with learned
statute representations.
6.3 Confidence Estimation Mechanism
To quantify the model’s certainty in its binary deci-
sion ( 0= reject, 1= grant), we extract the genera-
tion probabilities for both possible tokens. These
are normalized to sum to 100, producing an in-
terpretable confidence score. This score is used
both to compare models and to assess reliability in
real-world usage scenarios.
6.4 Prompting Schema for Structured
Extraction
Each input example during fine-tuning and infer-
ence uses a carefully designed one-shot prompt to
structure the raw bail judgment text into a JSON
6

format. This format captures key fields such as bail
type, health issues, past criminal records, statutes,
arguments, outcome, reasoning, and custody dates.
We place the complete prompt template in Ap-
pendix 5 for reproducibility and reusability across
models and tasks.
7 Evaluation Metrics
To evaluate the effectiveness of our Bail Prediction
System, we adopt a comprehensive set of metrics
covering both classification accuracy and explana-
tion quality. The evaluation is conducted on two
fronts: the bail prediction task and the explanation
generation task. We report Precision, Recall, F1,
and Accuracy for bail prediction, and we use both
quantitative and qualitative methods to evaluate the
quality of explanations generated by the model.
1.Lexical-based Evaluation: We utilized stan-
dard lexical similarity metrics, including Rouge-
L (Lin, 2004), BLEU (Papineni et al., 2002), and
METEOR (Banerjee and Lavie, 2005) which
measure the overlap and order of words between
the generated explanations and the reference
texts.
2.Semantic Similarity-based Evaluation: To
capture the semantic quality of the generated
explanations, we employed BERTScore (Zhang
et al., 2020), which measures the semantic simi-
larity between the generated text and the refer-
ence explanations.
3.LLM-based Evaluation (LLM-as-a-Judge):
To complement traditional metrics, we incorpo-
rate an automatic evaluation strategy that uses
large language models themselves as evaluators,
commonly referred to as LLM-as-a-Judge . This
evaluation is crucial for assessing structured ar-
gumentation and legal correctness in a format
aligned with expert judicial reasoning. We adopt
G-Eval (Liu et al., 2023), a GPT-4-based eval-
uation framework tailored for natural language
generation tasks. G-Eval leverages chain-of-
thought prompting and structured scoring to as-
sess explanations along three key criteria: fac-
tual accuracy ,completeness & coverage , and
clarity & coherence . Each generated legal ex-
planation is scored on a scale from 1 to 10 based
on how well it aligns with the expected content
and a reference document. The exact prompt
format used for evaluation is shown in Appendix
Table 5. For our experiments, we use the GPT-
4o-mini model to generate reliable scores with-out manual intervention. This setup provides
an interpretable, unified judgment metric that
captures legal soundness, completeness of rea-
soning, and logical coherence, beyond what tra-
ditional similarity-based metrics can offer.
8 Results and Analysis
Table 3 presents a comprehensive evaluation of
six experimental configurations of the phi-4 model,
varying in terms of fine-tuning and the use of statu-
tory context during inference. These configura-
tions were designed to isolate the contributions of
domain-specific supervision and external knowl-
edge augmentation to both prediction accuracy and
explanation quality.
The base version of the model, denoted as
VANILLA , serves as the starting point. This con-
figuration involves no fine-tuning and no statu-
tory context. Unsurprisingly, its performance is
modest across the board, achieving an accuracy of
only 0.47. The generated rationales are short, frag-
mented, and largely generic. This is expected, as
the base model lacks prior exposure to the specific
structure and semantics of Indian bail judgments
and legal reasoning. Interestingly, when statutory
text is introduced at inference time in VANILLA +
Context , performance drops even further (accu-
racy declines to 0.33, recall plummets to 0.14, and
BLEU and METEOR fall as well). This indicates
that the base model not only fails to utilize the
additional legal information but is possibly over-
whelmed or distracted by the lengthy and complex
provisions. It highlights a crucial insight: retrieval-
augmented generation (RAG) without task-specific
training does not necessarily improve outcomes and
may in fact hurt performance in tightly structured
legal prediction tasks.
The picture changes significantly with the intro-
duction of fine-tuning. In the FT-1 setup, the model
is trained on structured features extracted from bail
cases (e.g., statutes, facts, arguments, health condi-
tions), allowing it to learn typical decision patterns
observed in real judicial outcomes. This leads to
a clear jump in all metrics: accuracy improves
to 0.65, and explanation quality also rises consid-
erably (BLEU 0.22, METEOR 0.40, BERTScore
0.40). Notably, the model begins to generate mean-
ingful and coherent rationales that are contextually
aligned with the facts of the case. This confirms the
hypothesis that even without explicit legal ground-
ing, simply training on case features enables the
7

Model Outcome Metrics Reasoning Metrics Bail Conditions Metrics
Accuracy Precision Recall F1-Score ROUGE-L BLEU METEOR BERTScore BLEU METEOR BERTScore
V ANILLA 0.47 0.35 0.40 0.27 0.15 0.02 0.22 0.11 0.06 0.15 –0.90
V ANILLA + Context 0.33 0.31 0.14 0.17 0.13 0.01 0.21 0.07 0.04 0.13 –1.38
FT-1 0.65 0.45 0.65 0.45 0.42 0.22 0.40 0.40 0.17 0.39 0.46
FT-1 + Context 0.71 0.47 0.67 0.48 0.40 0.20 0.39 0.37 0.16 0.37 0.40
FT-2 0.79 0.57 0.74 0.55 0.47 0.29 0.47 0.43 0.23 0.46 0.51
FT-2 + Context 0.73 0.56 0.78 0.58 0.44 0.26 0.44 0.39 0.19 0.40 0.37
Table 3: Evaluation of Outcome Prediction and Explanation across Different Fine-tuning Configurations
model to internalize a pattern of legal decision-
making. When statutory context is added during
inference in FT-1 + Context , we observe a fur-
ther boost, particularly in recall (0.67) and F1-score
(0.48). The explanations remain strong, although
not significantly better than FT-1. This suggests
that once the model has learned how legal decisions
are typically made, the additional statutory informa-
tion serves to fill in edge cases or add confidence,
especially in complex or ambiguous scenarios.
The most impressive performance comes from
theFT-2 configuration, where the model is fine-
tuned not only on structured features but also on
the meaning and structure of statutes. This expo-
sure allows the model to develop a deeper under-
standing of how statutory language translates into
judicial reasoning. FT-2 achieves the highest over-
all accuracy (0.79), precision (0.57), and reasoning
metrics across all dimensions (BLEU 0.29, ME-
TEOR 0.47, BERTScore 0.43). The generated ra-
tionales are often detailed, citing relevant sections
and interpreting them in the context of case-specific
facts, showcasing a level of abstraction and legal
alignment beyond mere memorization. These re-
sults underscore the value of embedding statutory
semantics directly into the model during training,
effectively allowing it to learn not just from out-
comes but from the legal foundations that justify
those outcomes.
Adding RAG-based statutory context at infer-
ence time in FT-2 + Context provides the best
recall (0.78) and F1-score (0.58), reinforcing the
idea that retrieval-augmented inputs help when the
model already understands how to interpret them.
However, this setup sees a slight drop in precision
and explanation quality compared to FT-2, which
suggests a possible redundancy or even interfer-
ence. Once the model already possesses a refined
internal representation of legal semantics, exter-
nal information may sometimes disrupt rather than
assist, especially if irrelevant or loosely related
statutes are retrieved.
Overall, our results highlight a few key insights.
First, fine-tuning is not just helpful; it is essen-tial for legal judgment prediction and explanation.
Second, RAG-based statute retrieval is only bene-
ficial when used in conjunction with a fine-tuned
model. Third, and most importantly, exposing the
model to statutes during training (as in FT-2) yields
the most coherent and legally-grounded rationales.
This shows the importance of training models not
only on case-specific facts but also on the underly-
ing laws they must interpret. In sum, legal AI sys-
tems perform best when their internal knowledge
is aligned with the legal structure of the domain,
and when external context is introduced judiciously
and with purpose.
9 Conclusion and Future Scope
In this work, we introduced the IBPS, a comprehen-
sive AI framework for predicting bail outcomes and
generating fact-based legal rationales. By leverag-
ing a newly curated dataset of over 150,000 High
Court bail judgments, we demonstrated the effec-
tiveness of fine-tuning large language models using
structured case features and statutory knowledge.
Our experiments across various inference configu-
rations highlight that training with legal attributes
and statutes leads to more accurate and explain-
able decisions. The proposed approach not only
achieves strong performance on legal prediction
tasks but also lays the foundation for transparent
and interpretable legal AI systems.
Looking ahead, IBPS can be extended in mul-
tiple directions. First, future work may incorpo-
rate multi-accused scenarios and additional case
complexities such as co-accused interactions and
multi-layered charges. Second, the inclusion of
judgments in regional Indian languages will en-
hance linguistic diversity and applicability across
different jurisdictions. Lastly, integrating judicial
precedent retrieval and assessing real-world deploy-
ment through collaborations with courts and legal
aid clinics can further validate the practical utility
of the system. We believe IBPS marks an important
step toward data-driven legal decision support in
India.
8

Limitations
While our study presents a large-scale, diverse, and
well-structured dataset of Indian High Court bail
judgments and leverages advanced fine-tuning tech-
niques over LLMs, several limitations remain that
merit consideration. One of the primary challenges
lies in the inherent class imbalance present in the
dataset. The majority of cases pertain to anticipa-
tory and regular bail applications, with a very small
proportion comprising bail cancellation requests.
Additionally, the outcome classes are unevenly dis-
tributed, with “Granted” outcomes significantly
outnumbering other classes. This imbalance may
skew the model’s learning and evaluation, leading
to stronger performance on overrepresented cate-
gories while underperforming on underrepresented
ones. Though we adopt robust evaluation protocols,
future iterations should incorporate data rebalanc-
ing or class-aware training techniques to improve
generalization across all classes.
Another constraint stems from the scope of train-
ing, which is currently restricted to cases involv-
ing a single accused. In real-world legal contexts,
especially in criminal matters, it is common to en-
counter cases with multiple co-accused individuals,
each with distinct roles, backgrounds, or grounds
for bail consideration. Our present model does not
handle such scenarios, which limits its applicability
in more complex judicial situations. Extending the
modeling framework to handle multiple-accused
cases is a promising direction for future enhance-
ment.
Finally, the judgments used for training and eval-
uation were exclusively in English, as they were
sourced from High Courts where English is the pre-
dominant language of record. However, many bail
decisions across Indian states are authored in re-
gional languages. The current language constraint
restricts our model’s utility in those jurisdictions
and limits its inclusivity. Expanding the dataset
and modeling pipeline to support multilingual judg-
ments would be essential to enable broader deploy-
ment and fairness across linguistic demographics.
Despite these limitations, our study provides
a strong foundational benchmark and modeling
framework for the bail prediction and explanation
task in the Indian legal domain. We view these con-
straints not as critical barriers, but as opportunities
for future expansion and refinement.References
Nikolaos Aletras, Dimitrios Tsarapatsanis, Daniel
Preo¸ tiuc-Pietro, and Vasileios Lampos. 2016. Pre-
dicting judicial decisions of the european court of
human rights: A natural language processing per-
spective. PeerJ computer science , 2:e93.
Satanjeev Banerjee and Alon Lavie. 2005. METEOR:
An automatic metric for MT evaluation with im-
proved correlation with human judgments. In Pro-
ceedings of the ACL Workshop on Intrinsic and Ex-
trinsic Evaluation Measures for Machine Transla-
tion and/or Summarization , pages 65–72, Ann Arbor,
Michigan. Association for Computational Linguis-
tics.
Arunita Bansal, Sunidhi Chaudhary, Ritu Rani, In-
derdeep Kaur, Nidhi Goel, and Amita Dev. 2022.
Comparative analysis of bail judgements. In 2022 4th
International Conference on Artificial Intelligence
and Speech Technology (AIST) , pages 1–5.
Ilias Chalkidis, Manos Fergadiotis, Prodromos Malaka-
siotis, and Ion Androutsopoulos. 2019. Large-scale
multi-label text classification on eu legislation. arXiv
preprint arXiv:1906.02192 .
Ilias Chalkidis, Abhik Jana, Dirk Hartung, Michael
Bommarito, Ion Androutsopoulos, Daniel Martin
Katz, and Nikolaos Aletras. 2021. Lexglue: A bench-
mark dataset for legal language understanding in en-
glish. arXiv preprint arXiv:2110.00976 .
Stephen Demuth and Darrell Steffensmeier. 2004. The
impact of gender and race-ethnicity in the pretrial
release process. Social Problems , 51(2):222–242.
Neel Guha, Julian Nyarko, Daniel E Ho, Christopher
Ré, Adam Chilton, Aditya Narayana, Alex Chohlas-
Wood, Austin Peters, Brandon Waldon, Daniel N
Rockmore, et al. 2023. Legalbench: A collabo-
ratively built benchmark for measuring legal rea-
soning in large language models. arXiv preprint
arXiv:2308.11462 .
Prathamesh Kalamkar, Astha Agarwal, Aman Tiwari,
Smita Gupta, Saurabh Karn, and Vivek Raghavan.
2022. Named entity recognition in Indian court judg-
ments. In Proceedings of the Natural Legal Lan-
guage Processing Workshop 2022 , pages 184–193,
Abu Dhabi, United Arab Emirates (Hybrid). Associa-
tion for Computational Linguistics.
Chin-Yew Lin. 2004. ROUGE: A package for auto-
matic evaluation of summaries. In Text Summariza-
tion Branches Out , pages 74–81, Barcelona, Spain.
Association for Computational Linguistics.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang,
Ruochen Xu, and Chenguang Zhu. 2023. G-eval:
NLG evaluation using gpt-4 with better human align-
ment. In Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing ,
pages 2511–2522, Singapore. Association for Com-
putational Linguistics.
9

Shubham Kumar Nigam, Tanmay Dubey, Govind
Sharma, Noel Shallum, Kripabandhu Ghosh, and
Arnab Bhattacharya. 2025. Legalseg: Unlocking the
structure of indian legal judgments through rhetorical
role classification. Preprint , arXiv:2502.05836.
Joel Niklaus, Ilias Chalkidis, and Matthias Stürmer.
2021. Swiss-judgment-prediction: A multilingual
legal judgment prediction benchmark. arXiv preprint
arXiv:2110.00806 .
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-
Jing Zhu. 2002. Bleu: a method for automatic evalu-
ation of machine translation. In Proceedings of the
40th Annual Meeting of the Association for Compu-
tational Linguistics , pages 311–318, Philadelphia,
Pennsylvania, USA. Association for Computational
Linguistics.
Meghan Sacks, Vincenzo A Sainato, and Alissa R Acker-
man. 2015. Sentenced to pretrial detention: A study
of bail decisions and outcomes. American Journal of
Criminal Justice , 40(3):661–681.
Sugam Sharma, Ritu Shandilya, and Swadesh Sharma.
2021. Predicting indian supreme court decisions.
Available at SSRN 3917603 .
Chaojun Xiao, Haoxi Zhong, Zhipeng Guo, Cunchao Tu,
Zhiyuan Liu, Maosong Sun, Yansong Feng, Xianpei
Han, Zhen Hu, Heng Wang, et al. 2018. Cail2018:
A large-scale legal dataset for judgment prediction.
arXiv preprint arXiv:1807.02478 .
Hiroaki Yamada, Takenobu Tokunaga, Ryutaro Ohara,
Akira Tokutsu, Keisuke Takeshita, and Mihoko Sum-
ida. 2024. Japanese tort-case dataset for rationale-
supported legal judgment prediction. Artificial Intel-
ligence and Law , pages 1–25.
Hai Ye, Xin Jiang, Zhunchen Luo, and Wenhan Chao.
2018. Interpretable charge predictions for criminal
cases: Learning to generate court views from fact
descriptions. arXiv preprint arXiv:1802.08504 .
Ashwini V Zadgaonkar and Avinash J Agrawal. 2021.
An overview of information extraction techniques
for legal document analysis and processing. Interna-
tional Journal of Electrical & Computer Engineering
(2088-8708) , 11(6).
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q.
Weinberger, and Yoav Artzi. 2020. Bertscore: Evalu-
ating text generation with BERT. In 8th International
Conference on Learning Representations, ICLR 2020,
Addis Ababa, Ethiopia, April 26-30, 2020 . OpenRe-
view.net.
10

A Supplementary Figures and Analyses
A.1 Age-Based Outcome Analysis
Figure 4: Bail outcomes across different age groups.
Figure 4 highlights how age influences bail out-
comes. Applicants aged 65 and above had the high-
est grant rate (84.1%), while those in the 18–30
age group had the lowest (67.7%). This suggests a
potential judicial inclination toward granting bail to
older individuals, possibly due to perceived lower
flight risk or greater humanitarian concern.
A.2 Statutes with Lowest Grant Rates
Figure 5: Statutes with lowest bail grant rates.
Figure 5 presents statutes associated with the low-
est bail grant rates. Sections under the Abkari Act
(e.g., Section 64, Section 41A(b)(ii)) had a 0%
grant rate. Other socially sensitive laws like Sec-
tion 18 of the SC/ST Act had grant rates below
20%, indicating stricter judicial standards for such
offenses.
A.3 Crime-Specific Custody Duration
Patterns
Figure 7 reveals that murder cases had the longest
pretrial detentions, 24.5% of such cases extended
beyond two years. In contrast, drug and sexual as-
sault cases were resolved more quickly, with over
Figure 6: Custody duration distribution by type of
crime.
80% of drug-related cases and 49.9% of sexual as-
sault cases resolved within 90 days. This highlights
disparities in trial speed, possibly linked to the com-
plexity of evidence gathering or legal procedure.
A.4 Application Withdrawal Rates
Only 1.3% of bail applications (2,020 out of
150,430) were withdrawn by the applicants. This
low withdrawal rate indicates that the majority of
applicants follow through with judicial determi-
nation rather than opting out voluntarily. It may
reflect confidence in obtaining relief or a lack of
viable alternatives.
Figure 7: Bar graph showing the no. of withdrwal of
bail cases.
A.5 Crime-Age Correlation
Figure 8 explores age distribution across crime
types. Youth (18–30) were predominant in crimes
like kidnapping (56.9%), assault (52.2%), and rape
(51.4%). Middle-aged individuals (30–50) domi-
nated in drug-related (56.1%), fraud (56.6%), and
11

Figure 8: Age distribution of accused across crime categories.
white-collar crimes (50.9%). These age-crime rela-
tionships offer insights into offender profiling and
potential rehabilitative strategies.
Figure 9: Histogram showing the distribution of cases
across various Indian High Courts.
Figure 10: Distribution of bail application types.
Model Limitations
en_legal_ner_trf Weak detection of sections/statutes,
lack of mapping between sections and
statutes
BiLSTM-CRF (LegalSeg) Most sentences were tagged as ‘None’
MTL model (LegalSeg) Slightly better than BiLSTM-CRF, still
inadequate
SparkNLP model (InLe-
galNER)Comparable to en_legal_ner_trf, lim-
ited by training corpus
Rule-based Python script High precision for statute detection, not
suitable for other features
Table 4: Feature extraction models and their limitations
12

Prompt
f'''
given below is a python dictionary format to be filled with information about
the case given in the raw judgement text. replace the text between the < and > with
the information extracted from the raw judgement text.
**DO NOT COPY THE TEXT BETWEEN < AND >, INSTEAD REPLACE IT WITH THE EXTRACTED INFORMATION.**
python dict format :
{
"case":"""
Applicant applied for <type of application, "Regular-Bail" OR "Anticipatory-
Bail" OR "Bail-Cancellation", (one of these)>.
Is it a withdrawal application? <"Yes" or "No" depending upon if it is
application for withdrawal>.
Age of the accused is <age of the accused if provided, else write "not
provided">.
Health issues for the accused are <description of health issues if provided,
else "None">.
There are <"no" if there are no past criminal records, else "some"> past
criminal records of the accused.
Statutes mentioned in the judgement are <list of statutes,
eg: [Section 438 CrPC, Section 294(a) IPC, Section 506(1)(b) IPC, Section 34
IPC, Section 25 Arms Act],
do not include the acts/codes/sections that were removed or replaced later>.
Precedents mentioned in the judgement are <list of precedents, if any, else
"None">.
Details of the incident are <details of the incident if provided, else "None">.
Arguments supporting the bail application are <arguments supporting the bail
application, if any, else "None">.
Arguments opposing the bail application are <arguments opposing the bail
application, if any, else "None">."
"""
"outcome": "The outcome of the case is <status of the outcome, "Bail granted" OR
"Bail not granted" OR "Bail cancelled" OR "Bail not cancelled", (one of these)>.
The bail conditions are <list of bail conditions, if any, else "None">.
"reasoning": "The reasoning for the judgement is <list of reasoning, if any, else "None">."
"date_of_arrest": "<date of arrest, if provided, else "not provided">."
"date_of_judgement": "<date of judgement, if provided, else "not provided">."
}
For example, if the raw judgement text is as follows:
IN THE HIGH COURT OF KERALA AT ERNAKULAM ...
the output should be as follows:
```json
{
"case": "Applicant applied for ...
}
```
similarly given a raw judgement text, extract the information and convert it into
given json format.
Respond **ONLY** with valid JSON matching the schema. Do not add explanations or data
from example itself into the JSON.
Raw Judgement to process:
< case to be processed >
'''
Table 5: Prompt used across all one-shot/few-shot inference setups for extracting structured case data from raw
judgments.
13

Figure 11: Bail outcome by application type.
Figure 12: Effect of past criminal record on bail out-
come.
Figure 13: Custody duration by type of crime.
14

eCourts
Portal
Daksh
DatasetPython
scraperRaw
Case
Data
Case datasetLegal
evaluatorsGold
Standard
LLM
Fine-
tuningStatutory
contextResultsEvaluation
Feature
extraction
using LLMFigure 14: Flowchart representing the IBPS dataset preparation pipeline.
15
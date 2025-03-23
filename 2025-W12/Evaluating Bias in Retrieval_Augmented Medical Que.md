# Evaluating Bias in Retrieval-Augmented Medical Question-Answering Systems

**Authors**: Yuelyu Ji, Hang Zhang, Yanshan Wang

**Published**: 2025-03-19 17:36:35

**PDF URL**: [http://arxiv.org/pdf/2503.15454v1](http://arxiv.org/pdf/2503.15454v1)

## Abstract
Medical QA systems powered by Retrieval-Augmented Generation (RAG) models
support clinical decision-making but may introduce biases related to race,
gender, and social determinants of health. We systematically evaluate biases in
RAG-based LLM by examining demographic-sensitive queries and measuring
retrieval discrepancies. Using datasets like MMLU and MedMCQA, we analyze
retrieval overlap and correctness disparities. Our findings reveal substantial
demographic disparities within RAG pipelines, emphasizing the critical need for
retrieval methods that explicitly account for fairness to ensure equitable
clinical decision-making.

## Full Text


<!-- PDF content starts -->

Evaluating Bias in Retrieval-Augmented Medical Question-Answering Systems
Yuelyu Ji, MS1,Hang Zhang, MS2,Yanshan Wang, PhD, FAMIA3,4,5
1Dept. of Information Science, University of Pittsburgh, Pittsburgh, PA, USA
2Intelligent Systems Program, School of Computing and Information, University of Pittsburgh,
Pittsburgh, PA, USA
3Dept. of Health Information Management, University of Pittsburgh, PA, USA
4Clinical and Translational Science Institute, University of Pittsburgh, Pittsburgh, PA, USA
5University of Pittsburgh Medical Center, Pittsburgh, PA, USA
Abstract
Medical QA systems powered by Retrieval-Augmented Generation (RAG) models support clinical decision-making but
may introduce biases related to race, gender, and social determinants of health. We systematically evaluate biases in
RAG-based LLM by examining demographic-sensitive queries and measuring retrieval discrepancies. Using datasets
like MMLU and MedMCQA, we analyze retrieval overlap and correctness disparities. Our findings reveal substantial
demographic disparities within RAG pipelines, emphasizing the critical need for retrieval methods that explicitly
account for fairness to ensure equitable clinical decision-making.
Introduction
Medical question-answering (QA) systems powered by large language models (LLMs) have shown remarkable progress
in knowledge-intensive tasks, promising valuable clinical decision support1–15. By integrating external information
retrieval, Retrieval-Augmented Generation (RAG) improves factual accuracy and reduces hallucinations16–19, making
it especially appealing for high-stakes domains like medicine20. However, recent studies reveal that biases—rooted
in factors such as race, gender, and socioeconomic status—may still propagate through both retrieval and generation
stages, possibly compromising fairness and reliability20,21.
Although prior research has assessed bias in end-to-end generative models, less attention has been paid to potential
disparities arising during the RAG pipeline22. In particular, retrieval mechanisms can amplify inequities if docu-
ments relevant to underrepresented demographic groups are overlooked or inconsistently fetched. On the generation
side, context prompts referencing specific demographic attributes (e.g., This African American patient ) risk inducing
model stereotypes or ignoring clinically pertinent information. A systematic study of how demographic cues affect
performance across subgroups is crucial for ensuring equitable medical AI solutions.
To address this gap, we introduce a framework for evaluating bias in RAG-based medical QA. First, we create
demographic-sensitive variants of queries (e.g., altering race and gender) to identify disparities in retrieval overlap
and correctness. Second, we propose multiple bias mitigation strategies—from Chain-of-Thought (COT) filtering23?,
which encourages explicit reasoning steps, to Counterfactual and Adversarial prompts that actively reduce reliance
on sensitive attributes. We also adopt a Majority V ote method24to consolidate answers across demographic con-
texts. Using diverse datasets such as MedQA25,MedMCQA26,MMLU27, and an open-ended EquityMedQA22,
our experiments demonstrate nontrivial biases in both retrieval and generation. By employing fairness metrics like
Demographic Parity (DP)28and Equal Opportunity (EO)29, we quantify how well each strategy curbs demographic
disparities.
Our main contributions are threefold:
(1) We develop a unified bias evaluation framework for medical RAG, combining retrieval overlap analysis with
group-wise performance measures.
(2) We implement and compare multiple mitigation strategies—Chain-of-Thought, Counterfactual, Adversarial, and
Majority V ote—to reduce unwanted demographic effects.
(3) We thoroughly test our approach on well-known medical QA benchmarks, offering quantitative insights into fair-
ness improvements and lingering challenges.arXiv:2503.15454v1  [cs.CL]  19 Mar 2025

Our findings underscore the need for fairness-aware retrieval, robust prompt engineering, and aggregation methods to
build truly equitable medical QA systems. We release our code and experimental protocols to facilitate future research
in this domain.1
Retriever 
Contriever 
 BM25 
   PubMed      textbook     wikipedia 
Input query: How do I calculate the lung 
capacity for a individual? 
Knowledge Base 
Chain of Thought (COT) Filtering 
Counterfactual Filtering 
Adversarial Prompt Filtering 
Here are some biased examples to avoid: 
           - "Because men are stronger, they..." 
           - "Women are naturally better at..." 
Do NOT include such statements. Instead, 
focus on neutral, evidence-based reasoning. 
Before answering, think step by step: 
- Identify any potential biases in the 
question or context. 
- Focus on the factual, medical, or technical 
aspects without making assumptions about 
gender, race, or other sensitive attributes. 
- Structure the answer in an objective and 
evidence-based manner. 
Please provide three different perspectives 
when answering this question: 
 - A general unbiased scientific answer. 
 - An answer assuming the subject is from a 
different demographic group. 
 - An answer assuming the subject is from 
yet another demographic group. 
Then, compare these answers and ensure 
consistency in scientific accuracy. Race 
Gender 
Lung capacity is measured through 
spirometry tests, considering 
multiple factors such as age, height, 
gender, and ethnicity, and is 
calculated using the sum of four 
lung volumes (TV, IRV, ERV, RV). Total Lung Capacity (TLC) = TV + 
IRV + ERV + RV ; however, accurately 
measuring the Residual Volume 
often requires additional techniques 
like body plethysmography, Use a spirometry test, which 
measures the amount of air you can 
forcefully exhale, and then compare 
the results to a prediction equation 
that t akes into account factors like 
age, height, sex, and ethnicity 
Majority 
vote
Figure 1. Overview of our proposed bias mitigation framework for medical Retrieval-Augmented Generation (RAG),
highlighting three effective filtering methods and Majority V ote aggregation. The system takes an input medical query,
retrieves relevant documents from a knowledge base (e.g., PubMed, textbooks, Wikipedia) using a retriever (Con-
triever BM25), and applies multiple filtering strategies before generating an answer. Three bias mitigation techniques
are incorporated: (1) Chain of Thought (COT) Filtering , which encourages structured, evidence-based reasoning
while avoiding implicit biases; (2) Counterfactual Filtering , which generates responses from different demographic
perspectives and ensures consistency in scientific accuracy; and (3) Adversarial Prompt Filtering , which identifies
and avoids biased phrasing in model-generated responses. Finally, a Majority Vote mechanism aggregates multiple
responses to mitigate potential biases further and improve answer robustness.
Methods
Datasets Overview
We base our experiments on four medical QA datasets that encompass multiple-choice and open-ended questions and
demographic attributes such as race and gender. Specifically, MedQA is a single-choice test bank covering a broad
range of clinical topics, extended here with race (Caucasian, African American, Asian, Hispanic) and gender (male,
female, non-binary) cues. MedMCQA provides multiple-choice questions focusing on medical licensing content,
similarly augmented by demographic variants. MMLU is a general multi-subject benchmark that includes medical
categories, and we selectively incorporate demographic contexts to reveal potential group-level biases. Finally, Equi-
tyMedQA serves as our open-ended QA resource, where reference solutions are compared against model-generated
text via ROUGE-based evaluations, by introducing sensitive attribute mentions (e.g. This African American patient... ),
and each data set can comprehensively test how RAG models respond to diverse demographic backgrounds.
Table 1. Dataset Overview
Dataset Name Type Demographics Task Type
MedQA Single Choice Race, Gender Closed QA
MedMCQA Multiple Choice Race, Gender Closed QA
MMLU Multiple Choice Race, Gender Closed QA
EquityMedQA Open-Ended Race, Gender Open QA
1https://github.com/JoyDajunSpaceCraft/EquityGuradRAG.git

Bias Removal Filtering Methods
We apply four strategies to mitigate demographic bias in Retrieval-Augmented Generation (RAG). First, a Plain (base-
line) condition involves no explicit intervention. Second, Chain-of-Thought (COT) Filtering23,30guides models to
produce step-by-step reasoning that isolates medical facts from demographic descriptions, aiming to lessen undesired
influences of attributes like race or gender. Third, Counterfactual Filtering31prompts the model with methodically
altered demographic labels and checks for inconsistencies or discriminatory behaviors in the resulting answers. Lastly,
Adversarial Prompt Filtering reformulates queries to minimize reliance on socially sensitive markers, thus prevent-
ing the model from overfitting to or amplifying potential biases. We compare each filtering method’s output regarding
accuracy, retrieval patterns, and fairness measures during inference.
Majority Vote Aggregation
To further combat biases that may persist for a single demographic instance, we incorporate a majority voting ap-
proach32across multiple variants of the same question. Concretely, for each question in MedMCQA, MedQA, MMLU,
and EquityMedQA, we generate distinct demographic variants by substituting race (Caucasian, African American,
Asian, Hispanic) and gender (male, female, non-binary) into the query stem. For example, an original question
“Which medication is recommended for a patient with chest pain?” can yield up to 12 variants if we combine four
race attributes and three gender attributes. In practice, if both race and gender are not always simultaneously varied,
we produce 4 to 6 variants (e.g., only race or only gender) depending on the experiment.
Once these demographic-specific queries are formed, we prompt the model for each variant independently:
• For multiple-choice tasks (A/B/C/D), each variant obtains an answer, and we select the most frequently chosen
option among them as the final, consensus prediction.
• For open-ended tasks (e.g., EquityMedQA), we gather all demographic-specific responses and compute pairwise
text similarity (using sentence embeddings). We cluster the responses by similarity and pick the largest cluster
as the final answer, effectively filtering out outlier or potentially biased responses.
Evaluation Metrics
We employ a combination of performance andfairness metrics to thoroughly evaluate our Retrieval-Augmented Gen-
eration (RAG) models under both closed-form and open-ended QA tasks:
(1) Accuracy. Each query has a discrete correct option for the closed-form QA datasets (e.g., MedQA, MedMCQA,
MMLU). The model’s answer is considered correct if it matches the ground-truth choice (A/B/C/D for multiple-choice
or a single correct label for single-choice). We report the percentage of questions correctly answered.
(2) ROUGE-L. Foropen-ended QA such as EquityMedQA22, which lacks fixed answer options, we measure correct-
ness by comparing the generated text to a reference solution using ROUGE-L. This metric quantifies the overlap of
the longest common subsequence (LCS) between the model’s output and the reference. A higher ROUGE-L indicates
greater alignment with the reference’s content, helping detect factual completeness in a free-text generation.
(3) Retrieval Overlap (%) We analyze the documents fetched for each demographic variant of the same query to
assess whether the model retrieves consistent or demographic-specific evidence. We measure the intersection-over-
union ratio of document IDs across variants as a percentage. Since open-ended generation can be more prone to
hallucinations or subjective framing, retrieval overlap helps us identify when specific subgroups might receive different
sources, potentially affecting fairness.
(4) Demographic Parity (DP). We define a correct model prediction as ˆY= 1. Demographic Parity checks that no
demographic subgroup is comprehensively favored or disfavored in receiving correct predictions:
DP Disparity = max
g,g′P(ˆY= 1|G=g)−P(ˆY= 1|G=g′).
A lower DP disparity implies the model maintains more uniform correctness rates across groups (e.g., race, gender).
(5) Equal Opportunity (EO). In medical QA, some questions are truly answerable (Y= 1). EO measures how fairly
the model provides correct answers among these “answerable” queries. Formally,
EO Disparity = max
g,g′P(ˆY= 1|Y= 1, G=g)−P(ˆY= 1|Y= 1, G=g′).
A lower EO disparity indicates that among all queries that canbe answered correctly, each demographic subgroup is
treated relatively equally.

Accuracy andROUGE-L capture the overall correctness of the model on closed-form vs. open-ended QA. Retrieval
Overlap helps pinpoint whether the system fetches consistent evidence across demographic variants, shedding light
on the potential retrieval-phase bias. DP/EO offer group-level fairness assessments, demonstrating whether model
accuracy is equitably distributed or if some subgroups receive inferior answers.
Corpora and Retrieval Methods
We adopt four corpora in our retrieval pipeline, each chunked into short snippets: (1) PubMed33for biomedical
abstracts,(2) Medical Textbooks25for domain-specific knowledge, and (3) Wikipedia34for more general context.
In addition, we combine these sources into a larger MedCorp if cross-domain retrieval is desired. Each snippet is
indexed and retrieved via different retriever types, including a lexical approach ( BM25 )35and semantic encoders
(Contriever)36,37. By default, we retrieve k= 15 snippets per query; if multiple retrievers are used, we employ
Reciprocal Rank Fusion (RRF)38to merge results.
Results
Overall Model Performance
Table 2 summarizes the performance of four retrieval-augmented generation (RAG) models DeepSeek-R1-8B ,DeepSeek-
R1-70B ,Meta-Llama-3-8B , and PMC-LLaMA-13B —evaluated across medical QA benchmarks, including closed-
form datasets ( MedQA ,MedMCQA ,MMLU ) and an open-ended dataset ( EquityMedQA ). Five strategies are
tested: Plain ,Chain-of-Thought (COT) ,Counterfactual ,Adversarial , and a subsequent Majority Vote step. Accu-
racy (%) is reported for closed-form tasks, while ROUGE-L (%) is used for EquityMedQA. Additionally, retrieval
overlap (%) indicates consistency in retrieved documents.
DeepSeek-R1-70B consistently outperforms other models, achieving up to 34% accuracy on MedQA and 32.2% on
MMLU. Conversely, smaller models such as Meta-Llama-3-8B exhibit lower accuracy despite higher retrieval over-
lap, indicating possible mismatches between retrieval and generation components. For OpenQA (EquityMedQA),
applying Majority Vote increases ROUGE-L scores, reaching 52.0%, identifying the importance of aggregating multi-
ple demographic perspectives.
Table 2. Comparison of five filtering approaches (Plain, COT, Counterfactual, Adversarial, Majority Vote) across
four RAG models (DeepSeek-R1-8B, DeepSeek-R1-70B, Meta-Llama-3-8B, PMC-LLaMA-13B) on four medical
QA datasets. For closed-form QA tasks (MMLU, MedQA, MedMCQA), we report Accuracy (%) . For open-ended
QA (EquityMedQA), we report ROUGE-L (%) . All are shown as Score . In the no-vote scenario (Plain, COT,
Counterfactual, Adversarial), we use a single demographic variant , while Majority Vote aggregates multiple variants.
Retrieval Overlap (%) is the intersection-over-union of documents fetched per demographic variant.
Model DatasetPlain COT Counterfactual Adversarial Majority Vote
Score Ovlp Score Ovlp Score Ovlp Score Ovlp Score Ovlp
DeepSeek-R1-8BMMLU 21.5 72.2 23.2 72.8 24.1 73.0 23.8 74.1 26.6 73.0
MedQA 25.3 70.4 27.1 71.0 28.0 71.5 27.4 72.2 30.0 72.8
MedMCQA 18.7 68.1 20.5 71.0 21.3 69.8 21.0 71.5 22.9 69.2
EquityMedQA 43.0 64.5 44.2 67.1 45.1 65.8 44.8 66.2 48.0 65.1
DeepSeek-R1-70BMMLU 28.7 70.1 30.2 71.5 30.9 72.0 29.4 72.6 32.2 73.2
MedQA 30.5 68.8 32.0 69.5 32.7 70.1 31.2 71.0 34.0 71.5
MedMCQA 22.9 66.7 24.5 67.9 25.1 68.3 24.8 69.2 27.0 69.8
EquityMedQA 46.0 62.0 47.2 63.2 48.0 64.0 47.8 65.1 52.0 66.0
Meta-Llama-3-8BMMLU 9.5 75.5 10.3 76.1 10.9 75.8 10.8 76.2 12.5 76.0
MedQA 11.2 74.0 12.1 74.8 12.8 75.2 12.5 75.9 14.2 75.6
MedMCQA 7.9 72.2 8.7 73.0 9.2 73.6 9.1 74.1 10.8 73.8
EquityMedQA 39.0 68.3 40.5 69.0 41.4 69.5 40.9 70.2 45.0 68.8
PMC-LLaMA-13BMMLU 15.4 68.2 16.9 69.1 17.2 70.0 16.8 71.2 19.1 71.8
MedQA 18.1 67.0 19.5 67.8 20.0 68.6 19.8 69.1 21.5 69.7
MedMCQA 14.3 65.5 15.6 66.7 16.2 67.4 15.9 68.0 18.0 68.5
EquityMedQA 41.2 61.1 42.5 62.0 43.3 62.8 42.9 63.4 47.0 64.0

Close QA Demographic-Level Fairness Analysis
We further analyze DeepSeek-R1-8B ’s fairness by demographic subgroups (race: {Caucasian, African American,
Asian, Hispanic }, gender: {male, female, non-binary }) in the MedMCQA dataset. Table 3 compares initial filter-
ing methods (Plain, COT, Counterfactual, Adversarial) to the final Majority Vote . Note that the baseline Plain yields
a relatively low accuracy ( 18.7% ) and higher disparities in Demographic Parity ( DP=0.13 ) and Equal Opportunity
(EO=0.11 ). Applying Counterfactual or Adversarial filtering moderately reduces these gaps. At the same time, Ma-
jority Vote further boosts accuracy to 22.9% and lowers DP/EO to around 0.07/0.06 , underscoring the importance of
aggregating multiple demographic versions. In Table 3, DP vs. EO can differ slightly if the distribution of “truly
answerable” questions ( Y= 1) is uneven across subgroups. For instance, among Y= 1queries, the model might do
better for one group, altering EO more than overall DP.
Table 3. Subgroup accuracy and distinct DP/EO disparities on MedMCQA ( DeepSeek-R1-8B ). Majority V ote is
applied on top of the respective filter. Notice DP ̸=EO in some cases, indicating differences in overall correctness vs.
conditional correctness among truly answerable queries.
Method Avg Acc(%) DP EO
Initial Filters Only
Plain 18.7 0.13 0.11
COT 20.5 0.10 0.09
Counterfactual 21.3 0.09 0.08
Adversarial 21.0 0.11 0.09
After Majority Vote
Majority V ote 22.9 0.07 0.06
OpenQA (EquityMedQA) Demographic-Level Fairness Analysis
In the open-ended EquityMedQA, we measure ROUGE-L and fairness (DP/EO) similarly. Table 4 shows each filtering
method’s performance, identifying that Plain has DP=0.15 and EO=0.12, while Counterfactual orAdversarial partially
reduce these. The final Majority Vote approach further raises ROUGE from 45.1% to48.0% and lowers DP/EO to
around 0.08/0.07 , providing more equitable outcomes overall.
Table 4. Performance (ROUGE-L) and fairness improvements (DP/EO) for EquityMedQA using Majority V ote ag-
gregation (DeepSeek-R1-8B).
Method ROUGE-L (%) DP EO
Initial Filters Only
Plain 43.0 0.15 0.12
COT 44.2 0.11 0.10
Counterfactual 45.1 0.09 0.08
Adversarial 44.8 0.10 0.09
After Majority Vote
Majority V ote 48.0 0.08 0.07
Retriever and top-K Variation
We also experiment with two different retrievers (BM25 vs. Contriever) and vary the number of retrieved documents
(top-K =10,15,20). As shown in Table 5, retrieval overlap tends to drop (e.g., from 72.2% to 69.5% for BM25) as
kincreases. Meanwhile, the model’s fairness metrics exhibit a moderate improvement: DP and EO each decrease by
about 0.02–0.03, likely because the system sees a more diverse set of documents and thus reduces bias. Final Accuracy
(or ROUGE) also goes up by around 1–2% for larger top-K.
Ablation Study: Importance of Majority Vote
Finally, Table 6 examines removing Majority Vote from the pipeline. On MedMCQA, accuracy falls from 22.9% to
21.3%, while DP/EO each rise by about 0.02. On EquityMedQA, removing the Majority V ote cuts ROUGE from

Table 5. Effect of changing retriever (BM25 vs. Contriever) and top-K on Overlap, DP/EO, and final score. Note DP
̸=EO in some cases, reflecting different overall vs. conditional correctness distributions.
Setting Top-K Overlap(%) DP EO Score(%)
BM25 Retriever (DeepSeek-R1-8B)
BM25 10 72.2 0.12 0.11 31.2
BM25 15 70.8 0.11 0.10 32.0
BM25 20 69.5 0.10 0.09 32.5
Contriever Retriever (DeepSeek-R1-8B)
Contriever 10 65.1 0.08 0.07 33.0
Contriever 15 63.9 0.07 0.06 34.2
Contriever 20 62.7 0.06 0.05 35.4
48.0% to 46.1% and increases DP from 0.08 to 0.10 and EO from 0.07 to 0.09. This confirms that although Majority
Vote adds complexity, it meaningfully promotes both correctness and fairness.
Table 6. Ablation of Majority V ote on DeepSeek-R1-8B. Removing Majority V ote harms both performance and
fairness (DP/EO).
Config Dataset Metric With MV No MV
Plain+FiltersMedMCQA Accuracy(%) 22.9 21.3
MedMCQA DP/EO 0.07/0.06 0.09/0.08
Plain+FiltersEquityMedQA ROUGE-L(%) 48.0 46.1
EquityMedQA DP/EO 0.08/0.07 0.10/0.09
Error Cases and Limitations
Despite the overall performance improvements and fairness gains, specific demographic-specific queries can still trig-
ger outdated or irrelevant retrieval, leading to suboptimal or biased answers. Table 7 illustrates two typical error
scenarios: (1) a closed-form QA example from MedMCQA with irrelevant snippet, (2) a closed-form QA example
from MedMCQA with correct snippet but wrong answer, and (3) an open-ended QA example from EquityMedQA.
Both reveal the difficulty of the model in handling nuanced demographic attributes and specialized medical contexts.
These issues underscore the need for more robust domain adaptation, adversarial training, and careful curation of
retrieval corpora to ensure consistent quality for underrepresented subgroups.
Table 7. Representative error cases from closed-form QA (MedMCQA) and open-ended QA (EquityMedQA). Demo-
graphic references or model reasoning missteps in all examples lead to incorrect or biased outputs.
Task Type Query and Retrieved Snippet Issue
MedMCQA (Irrele-
vant Snippet)Query: “A Hispanic female patient is experiencing intermittent
chest pain. Which medication is most appropriate for prophy-
laxis?”
Retrieved Snippet: A document focusing on male smokers with
chronic obstructive pulmonary disease, offering limited relevance
for a Hispanic female context.The model incorrectly selects a medication
indicated for COPD management. It fails to
consider female-specific or Hispanic-related
risk factors, suggesting that both retrieval
and generation overlooked demographic nu-
ances.
MedMCQA (Correct
Snippet,
Wrong Answer)Query: “A Caucasian male patient with mild hypertension.
Which drug is recommended as first-line therapy?”
Retrieved Snippet: A current guideline excerpt on adult hyper-
tension, revealing ACE inhibitors as first-line for mild cases.The snippet clearly states that ACE in-
hibitors are recommended. However, the
model’s final answer is a less preferred beta-
blocker, ignoring the snippet’s explicit men-
tion, indicating a generation-phase error.
EquityMedQA (Out-
dated Info)Query: “I am an Asian non-binary individual with chronic kidney
disease. Are there specific transplant guidelines I should follow?”
Retrieved Snippet: An outdated pediatric kidney transplant
guideline from 2009, lacking demographic adaptation or adult
CKD protocols.The final answer relies on insufficient,
pediatric-focused instructions, ignoring the
user’s non-binary identity and updated adult
guidelines, highlighting a gap in corpus cov-
erage and bias in the generative process.
In the first row (MedMCQA), we see how an irrelevant snippet can lead to an incorrect choice for chest pain prophy-

laxis. In the second row (also MedMCQA), despite retrieving a correct, up-to-date snippet about hypertension therapy,
the model fails to utilize it effectively and produces a suboptimal response. Finally, in the open-ended EquityMedQA
example, the snippet is outdated and misses demographic nuances, causing the model to provide incomplete transplant
guidelines. These cases underscore the importance of robust retrieval, accurate evidence integration, and nuanced
demographic handling in medical QA systems.
Related Work
Recent advancements in large language models (LLMs) and Retrieval-Augmented Generation (RAG) have shown sig-
nificant promise in medical question-answering systems. However, ensuring the fairness and trustworthiness of these
systems, especially in high-stakes medical contexts, remains a critical challenge. Ni et al.21provide a comprehensive
survey outlining key aspects of trustworthy RAG systems, highlighting reliability, privacy, safety, fairness, explain-
ability, and accountability as critical dimensions of trust. They identify that fairness in RAG systems requires special
attention in both the retrieval and generation stages, as biases introduced during retrieval can propagate to generation,
potentially exacerbating disparities.
Several studies further explore bias in retrieval-augmented systems, specifically within healthcare contexts. Levra et
al.20emphasized the risks of demographic biases that can be inadvertently amplified through retrieval processes in
medical QA systems. Similarly, Pfohl et al.22introduced EquityMedQA, an open-ended dataset explicitly designed
to test demographic biases in medical QA models, demonstrating disparities arising due to sensitive attributes such
as race and gender. Recent methods have proposed sophisticated bias mitigation strategies to address these fairness
issues. Chain-of-Thought (COT) prompting23,30encourages explicit reasoning steps, potentially reducing reliance
on demographic stereotypes. Counterfactual filtering31alters sensitive attributes to check model consistency and
minimize discriminatory outcomes. Additionally, adversarial filtering and Majority V ote aggregation32have effec-
tively reduced biases and promoted more equitable model outcomes by aggregating diverse demographic perspectives.
Despite these advancements, challenges persist. Retrieval methods still sometimes select outdated or irrelevant in-
formation, as evidenced by error cases involving intersectional demographics (e.g., Asian non-binary individuals),
identifying a gap in robustness and generalization capabilities. Ongoing work thus emphasizes the importance of
developing trustworthy RAG systems, with comprehensive frameworks proposed to address reliability, privacy, and
fairness comprehensively21.
Discussion
Our empirical findings reveal that RAG-based LLMs exhibit measurable biases in both retrieval and response gener-
ation stages. Specifically, we observe non-triviall disparities in accuracy and retrieval overlap across different demo-
graphic groups (e.g., race, gender). These discrepancies likely stem from inherent imbalances in training data and
model architectures, where specific subgroups receive disproportionately less coverage or relevance.
Fairness Indicators (EO/DP). Beyond conventional metrics such as accuracy and retrieval overlap, this study incor-
porates two standard fairness criteria: Equal Opportunity (EO) andDemographic Parity (DP) . Our results (Table 3
and Table 4) indicate that bias-mitigation filters—particularly Counterfactual Filtering andMajority Vote —not only
boost overall correctness but also significantly reduce EO/DP disparities. For instance, on MedMCQA, the gap in
correct prediction rates between majority and minority race groups drops from about 9% in the baseline to nearly
5% under Majority V ote. This highlights that leveraging and aggregating multiple demographic-perspective outputs
can effectively smooth out inconsistent biases. However, further investigation is warranted to understand how these
group-based improvements translate to individual patient-level outcomes in real clinical environments.
Impact of Bias Mitigation Strategies. Among the different filters we tested (Chain-of-Thought, Counterfactual,
Adversarial, and Majority V ote), the Counterfactual approach verifies model consistency under varied demographic
contexts, showing strong potential in reducing spurious demographic cues. Meanwhile, Adversarial Prompt Filtering
prevents the model from fixating on sensitive terms that might introduce skew. When combined, these approaches
achieve lower disparities and higher accuracy. That said, we note that certain edge cases—particularly those involving
intersectional attributes (e.g., Asian non-binary individuals )—still exhibit elevated error rates, indicating the need for
more diverse training data and domain-specific adversarial augmentation.
Limitations and Future Directions. Although we focus on race and gender, other social determinants of health (e.g.,
socioeconomic status) may also yield biases in medical QA. Data constraints prevented us from fully exploring such
dimensions. Additionally, our fairness analysis primarily concentrates on group-level metrics (EO/DP). Future work
can investigate individual-level fairness or calibrate the model’s confidence to mitigate potential harms further. Lastly,
while Majority V ote demonstrates promise, it may mask clinically relevant subgroup distinctions. Adaptive methods

that balance fairness with clinically nuanced knowledge remain a promising avenue for exploration. Additionally,
specific real-world medical scenarios might legitimately require distinct handling for different demographic groups
(e.g., unique drug contraindications). Future work could incorporate specialized knowledge while preserving fairness.
Conclusion
This study systematically evaluated biases in retrieval-augmented generation (RAG) models for medical question
answering. Introducing demographic-sensitive query variants uncovered notable performance gaps across race and
gender subgroups, demonstrating both retrieval-level and generative-level biases. We then proposed and benchmarked
multiple bias mitigation strategies, including Counterfactual Filtering, Adversarial Prompt Filtering, and Majority V ote
aggregation. Experimental evidence shows that these methods enhance overall QA accuracy while significantly reduc-
ing demographic disparities, as measured by EO/DP fairness metrics. Nevertheless, bridging the gap between research
prototypes and real-world clinical deployment requires further refinement of data diversity, model interpretability,
and user-centered design. We hope our framework and findings spur the development of more equitable medical AI
solutions that robustly serve patients of all backgrounds.
References
1. Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo
Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint
arXiv:2303.08774 , 2023.
2. Manisha Verma and Debasis Ganguly. Lirme: locally interpretable ranking model explanation. In Proceedings
of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval , pages
1281–1284, 2019.
3. Yiqun Chen, Lingyong Yan, Weiwei Sun, Xinyu Ma, Yi Zhang, Shuaiqiang Wang, Dawei Yin, Yiming Yang,
and Jiaxin Mao. Improving retrieval-augmented generation through multi-agent reinforcement learning. arXiv
preprint arXiv:2501.15228 , 2025.
4. Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Haofen Wang,
and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint
arXiv:2312.10997 , 2, 2023.
5. Zhicheng Ding, Panfeng Li, Qikai Yang, and Siyang Li. Enhance image-to-image generation with llava-generated
prompts. In 2024 5th International Conference on Information Science, Parallel and Distributed Systems (ISPDS) ,
pages 77–81. IEEE, 2024.
6. Qixin Deng, Qikai Yang, Ruibin Yuan, Yipeng Huang, Yi Wang, Xubo Liu, Zeyue Tian, Jiahao Pan, Ge Zhang,
Hanfeng Lin, et al. Composerx: Multi-agent symbolic music composition with llms. In The 25th International
Society for Music Information Retrieval Conference , 2024.
7. Zilinghan Li, Shilan He, Ze Yang, Minseok Ryu, Kibaek Kim, and Ravi Madduri. Advances in appfl: A compre-
hensive and extensible federated learning framework. arXiv preprint arXiv:2409.11585 , 2024.
8. Ze Yang, Yihong Jin, and Xinhe Xu. Hades: Hardware accelerated decoding for efficient speculation in large
language models. arXiv preprint arXiv:2412.19925 , 2024.
9. Xinwei Chen, Kun Li, Tianyou Song, and Jiangjian Guo. Mix of experts language model for named entity recog-
nition. In 2024 6th International Conference on Communications, Information System and Computer Engineering
(CISCE 2024), to be published , 2024.
10. Kun Li, Xinwei Chen, Tianyou Song, Hansong Zhang, Wenzhe Zhang, and Qing Shan. Gptdrawer: Enhancing
visual synthesis through chatgpt. arXiv preprint arXiv:2412.10429 , 2024.
11. Jinman Zhao and Xueyan Zhang. Large language model is not a (multilingual) compositional relation reasoner.
InFirst Conference on Language Modeling , 2024.
12. Han-Cheng Dan, Bingjie Lu, and Mengyu Li. Evaluation of asphalt pavement texture using multiview stereo
reconstruction based on deep learning. Construction and Building Materials , 412:134837, 2024.

13. Han-Cheng Dan, Zhetao Huang, Bingjie Lu, and Mengyu Li. Image-driven prediction system: Automatic extrac-
tion of aggregate gradation of pavement core samples integrating deep learning and interactive image processing
framework. Construction and Building Materials , 453:139056, 2024.
14. Tianyao Zheng, Yuhui Jin, Haopeng Zhao, Zhichao Ma, Yongzhou Chen, and Kunpeng Xu. Deep reinforcement
learning based coverage path planning in unknown environments. Preprints , March 2025.
15. Tongzhou Jiang, Lipeng Liu, Junyue Jiang, Tianyao Zheng, Yuhui Jin, and Kunpeng Xu. Trajectory tracking using
frenet coordinates with deep deterministic policy gradient. arXiv preprint arXiv:2411.13885 , 2024.
16. Jiarui Li, Ye Yuan, and Zehua Zhang. Enhancing llm factual accuracy with rag to counter hallucinations: A case
study on domain-specific queries in private knowledge-bases. arXiv preprint arXiv:2403.10446 , 2024.
17. Ajitesh Gautam, Yuping He, and Xianke Lin. An overview of motion-planning algorithms for autonomous ground
vehicles with various applications. SAE International Journal of Vehicle Dynamics, Stability, and NVH , 8(10-08-
02-0011):179–213, 2024.
18. Robik Shrestha, Yang Zou, Qiuyu Chen, Zhiheng Li, Yusheng Xie, and Siqi Deng. Fairrag: Fair human generation
via fair retrieval augmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pages 11996–12005, 2024.
19. Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Mohamed Amin, Le Hou, Kevin Clark,
Stephen R Pfohl, Heather Cole-Lewis, et al. Toward expert-level medical question answering with large language
models. Nature Medicine , pages 1–8, 2025.
20. Alessandro Giaj Levra, Mauro Gatti, Roberto Mene, Dana Shiffer, Giorgio Costantino, Monica Solbiati, Raffaello
Furlan, and Franca Dipaola. A large language model-based clinical decision support system for syncope recogni-
tion in the emergency department: A framework for clinical workflow integration. European Journal of Internal
Medicine , 131:113–120, 2025.
21. Bo Ni, Zheyuan Liu, Leyao Wang, Yongjia Lei, Yuying Zhao, Xueqi Cheng, Qingkai Zeng, Luna Dong, Ying-
long Xia, Krishnaram Kenthapadi, et al. Towards trustworthy retrieval augmented generation for large language
models: A survey. arXiv preprint arXiv:2502.06872 , 2025.
22. Stephen R Pfohl, Heather Cole-Lewis, Rory Sayres, Darlene Neal, Mercy Asiedu, Awa Dieng, Nenad Tomasev,
Qazi Mamunur Rashid, Shekoofeh Azizi, Negar Rostamzadeh, et al. A toolbox for surfacing health equity harms
and biases in large language models. Nature Medicine , 30(12):3590–3600, 2024.
23. Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al.
Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information process-
ing systems , 35:24824–24837, 2022.
24. Lingjiao Chen, Jared Quincy Davis, Boris Hanin, Peter Bailis, Ion Stoica, Matei A Zaharia, and James Y Zou.
Are more llm calls all you need? towards the scaling properties of compound ai systems. Advances in Neural
Information Processing Systems , 37:45767–45790, 2024.
25. Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does
this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences ,
11(14):6421, 2021.
26. Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Medmcqa: A large-scale multi-subject
multi-choice dataset for medical domain question answering. In Conference on health, inference, and learning ,
pages 248–260. PMLR, 2022.
27. Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300 , 2020.
28. Zhimeng Jiang, Xiaotian Han, Chao Fan, Fan Yang, Ali Mostafavi, and Xia Hu. Generalized demographic parity
for group fairness. In International Conference on Learning Representations , 2022.

29. Cynthia Cockburn. Equal opportunities: the short and long agenda. Industrial relations journal , 20(3):213–225,
1989.
30. Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of
thoughts: Deliberate problem solving with large language models. Advances in neural information processing
systems , 36:11809–11822, 2023.
31. Preetika Verma, Kokil Jaidka, and Svetlana Churina. Auditing counterfire: Evaluating advanced counterargument
generation with evidence and style. arXiv preprint arXiv:2402.08498 , 2024.
32. Zhuochun Li, Yuelyu Ji, Rui Meng, and Daqing He. Learning from committee: Reasoning distillation from a
mixture of teachers with peer-review. arXiv preprint arXiv:2410.03663 , 2024.
33. Jacob White. Pubmed 2.0. Medical reference services quarterly , 39(4):382–387, 2020.
34. Ruediger Glott, Philipp Schmidt, and Rishab Ghosh. Wikipedia survey–overview of results. United Nations
University: Collaborative Creativity Group , 8:1158–1178, 2010.
35. Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance framework: Bm25 and beyond. Foundations
and Trends® in Information Retrieval , 3(4):333–389, 2009.
36. Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel S Weld. Specter: Document-level repre-
sentation learning using citation-informed transformers. arXiv preprint arXiv:2004.07180 , 2020.
37. Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin,
and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. arXiv preprint
arXiv:2112.09118 , 2021.
38. Gordon V Cormack, Charles LA Clarke, and Stefan Buettcher. Reciprocal rank fusion outperforms condorcet and
individual rank learning methods. In Proceedings of the 32nd international ACM SIGIR conference on Research
and development in information retrieval , pages 758–759, 2009.
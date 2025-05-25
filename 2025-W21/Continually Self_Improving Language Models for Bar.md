# Continually Self-Improving Language Models for Bariatric Surgery Question--Answering

**Authors**: Yash Kumar Atri, Thomas H Shin, Thomas Hartvigsen

**Published**: 2025-05-22 01:02:51

**PDF URL**: [http://arxiv.org/pdf/2505.16102v1](http://arxiv.org/pdf/2505.16102v1)

## Abstract
While bariatric and metabolic surgery (MBS) is considered the gold standard
treatment for severe and morbid obesity, its therapeutic efficacy hinges upon
active and longitudinal engagement with multidisciplinary providers, including
surgeons, dietitians/nutritionists, psychologists, and endocrinologists. This
engagement spans the entire patient journey, from preoperative preparation to
long-term postoperative management. However, this process is often hindered by
numerous healthcare disparities, such as logistical and access barriers, which
impair easy patient access to timely, evidence-based, clinician-endorsed
information. To address these gaps, we introduce bRAGgen, a novel adaptive
retrieval-augmented generation (RAG)-based model that autonomously integrates
real-time medical evidence when response confidence dips below dynamic
thresholds. This self-updating architecture ensures that responses remain
current and accurate, reducing the risk of misinformation. Additionally, we
present bRAGq, a curated dataset of 1,302 bariatric surgery--related questions,
validated by an expert bariatric surgeon. bRAGq constitutes the first
large-scale, domain-specific benchmark for comprehensive MBS care. In a
two-phase evaluation, bRAGgen is benchmarked against state-of-the-art models
using both large language model (LLM)--based metrics and expert surgeon review.
Across all evaluation dimensions, bRAGgen demonstrates substantially superior
performance in generating clinically accurate and relevant responses.

## Full Text


<!-- PDF content starts -->

arXiv:2505.16102v1  [cs.CL]  22 May 2025Continually Self-Improving Language Models for
Bariatric Surgery Question–Answering
Yash Kumar Atri atri@virginia.edu
School of Data Science
University of Virginia
Charlottesville, VA, USA
Thomas Shin thomas.shin@uvahealth.org
Department of Surgery
University of Virginia School of Medicine
Charlottesville, VA, USA
Thomas Hartvigsen hartvigsen@virginia.edu
School of Data Science
University of Virginia
Charlottesville, VA, USA
Abstract
While bariatric and metabolic surgery (MBS) is considered the gold standard treatment
for severe and morbid obesity, its therapeutic efficacy hinges upon active and longitudinal
engagement with multidisciplinary providers, including surgeons, dietitians/nutritionists,
psychologists, and endocrinologists. This engagement spans the entire patient journey, from
preoperative preparation to long-term postoperative management. However, this process
is often hindered by numerous healthcare disparities, such as logistical and access barriers,
which impair easy patient access to timely, evidence-based, clinician-endorsed information.
To address these gaps, we introduce bRAGgen , a novel adaptive RAG-based model that
autonomously integrates real-time medical evidence when response confidence dips below
dynamic thresholds. This self-updating architecture ensures that responses remain current
and accurate, reducing the risk of misinformation. Additionally, we introduce bRAGq , a
curated dataset of 1 ,302 bariatric surgery–related questions, validated by expert bariatric
surgeon, constituting the first large-scale, domain-specific benchmark for comprehensive
MBS care. In a two-phase evaluation, bRAGgen is benchmarked against state-of-the-art
models using both large language model (LLM)-based metrics and expert surgeon review.
Across all evaluation dimensions, bRAGgen demonstrates substantially superior performance
in generating clinically accurate and relevant responses.
1. Introduction
Severe obesity and its subsequent metabolic disease have become a widespread endemic
condition, leading to a projected incidence of 25% across the United States by 2030, causing
a massive health burden in the general US population (Ward et al., 2019). Metabolic
and bariatric surgery (MBS) remains the gold standard treatment for severe obesity and
metabolic disease, with over 270,000 annual procedures in the United States (Clapp et al.,
2024; Mechanick et al., 2020; Barres et al., 2013; Loos and Yeo, 2022; Setarehdan et al.,

Continually Self-Improving Language Models for Bariatric Surgery QA
2023). However, successful weight loss post-MBS heavily relies on patient education, which
has led National Bariatric Surgery and Medical societies to emphasize extensive education
services for MBS patients (Mechanick et al., 2020).
Perioperative MBS patient education includes information about dietary modification,
adjunctive lifestyle modifications, expectations surrounding postoperative complications,
and psychosocial support; all important mitigators of periprocedural complications and
postoperative weight regain, which can afflict up to 64% of patients in 5–10 years post-
MBS (Groller et al., 2017; Bjerkan et al., 2022; David et al., 2020; McLennan et al., 2023;
Kim et al., 2023). And after surgery, patients only attend their yearly followup an aver-
age of 6.5–29.6% of the time, leaving patients and providers with limited opportunities to
communicate. Overall, the lack of sustained patient engagement and education post-MBS
is a critical impediment to optimal postoperative outcomes with many causes, including
low health literacy rates, information inaccessibility, and geographic distances to health-
care providers (Setarehdan et al., 2023; Mechanick et al., 2020; Schlottmann et al., 2023;
Bartholomay et al., 2024).
Given these challenges, there is a clear need for scalable, accessible, and continually
updated educational and decision-support tools tailored to the unique needs of MBS pa-
tients—spanning from preoperative preparation to long-term postoperative management (P
et al., 2024). Traditional patient education materials—whether delivered in print, via static
websites, or through periodic telehealth visits—often fail to adapt dynamically to emerging
clinical evidence (Javanparast et al., 2021) or to the evolving clinical status of individual
patients. Moreover, existing digital health platforms seldom incorporate mechanisms to
detect when their guidance may be outdated or insufficiently confident (Wang et al., 2025),
leading to knowledge gaps in both patients and clinicians.
Large language models (LLMs) (Grattafiori et al., 2024; Abdin et al., 2024; Minaee
et al., 2025) offer a potential solution by providing natural language interfaces for patients
to query. However, LLMs face limitations due to fixed knowledge cutoffs (Cheng et al., 2024)
and the most-capable are trained on broad, general-purpose corpora (Alber et al., 2025),
leaving them unaware of the latest bariatric surgery guidelines or nuanced postoperative
considerations (B´ elisle-Pipon, 2024). One popular way to address outdated knowledge in
LLMs is through retrieval-augmented generation (RAG) methods (Gao et al., 2024), which
retrieve up-to-date documents from a database. However, their databases are typically static
and efforts to increase them over time easily introduce “context noise,” overwhelming the
LLM with conflicting inputs and producing higher hallucination rates (Zhang and Zhang,
2025). Furthermore, these methods lack built-in mechanisms to assess when their own
outputs may be insufficiently confident, placing patients and clinicians at risk of incomplete
or incorrect guidance (Lewis et al., 2021).
We propose bRAGgen , an adaptive RAG framework that continuously monitors its re-
sponse confidence and, upon detecting uncertainty, autonomously retrieves and integrates
the latest peer-reviewed evidence and clinical guidelines from trusted biomedical sources
such as PubMed1. This self-updating architecture ensures that guidance remains current,
accurate, and clinically relevant, reducing the risk of outdated or misleading recommenda-
tions. Complementing bRAGgen , we introduce bRAGq , a curated dataset of 1 ,302 bariatric
1. https://pubmed.ncbi.nlm.nih.gov
2

Continually Self-Improving Language Models for Bariatric Surgery QA
surgery–related questions validated by a bariatric surgeon, constituting the first large-scale,
domain-specific benchmark for the full spectrum of MBS patient care—from preoperative
preparation to long-term postoperative management. In a two-phase evaluation, leveraging
both LLM-based metrics and expert surgeon reviews, bRAGgen demonstrates substantially
superior performance in generating clinically accurate, relevant, and actionable responses,
paving the way for more accessible, evidence-based support for MBS patients.
Generalizable insights about machine learning in the context of healthcare
While we focus on bariatric surgery patients’ educational needs, our work includes gener-
alizable insights for other healthcare contexts. First, the need to access up-to-date medical
evidence is widespread, especially in areas where the scientific literature grows quickly. Our
work demonstrates that it is feasible to approach this problem by training models to directly
access webpages, and keeping them relevant throughout deployment. Second, our methods
are general purpose and are widely applicable. While there is a major need for bariatric
surgery education and it is our expertise, our machine learning methods can be generalized
to any domains where patient questions can be collected, relevant literature exists, and
model answers can be validated.
2. Related Work
The landscape of digital health interventions for bariatric care across the entire surgical
spectrum-from preoperative preparation to long-term postoperative management has ex-
panded considerably, with multiple studies evaluating the efficacy and usability of mobile
applications and web-based platforms. For instance, a German cohort study by Wu et al.
(2024c) demonstrated that mHealth follow-up via a dedicated mobile app achieved compara-
ble outcomes to traditional in-person care across weight loss, quality of life, and nutritional
status metrics. Similarly, a systematic review by Patel and Thind (2020) identified 33 usabil-
ity studies of mHealth apps across surgical subspecialties, underscoring both the potential
of digital tools and the persistent challenge of sustaining long-term patient engagement.
In parallel, advances in large language models (LLMs) (Grattafiori et al., 2024; Abdin
et al., 2024; Minaee et al., 2025) have led to growing interest in their ability to distill up-to-
date information (Atri et al., 2023c; Dey et al., 2020; Atri et al., 2023b,a, 2021) and support
tasks such as (medical) question answering (Khlaut et al., 2024; Sviridova et al., 2024;
Vladika and Matthes, 2024; Saeed, 2024) and clinical decision-making (Kim et al., 2024; Lu
et al., 2024; Singhal et al., 2025). The GPT models (Kojima et al., 2023) showcased impres-
sive zero- and few-shot capabilities but is inherently limited by its fixed pretraining cutoff
and general-purpose data (Tamkin et al., 2021). Retrieval-augmented generation (RAG)
(Gao et al., 2024) based methods ground LLM outputs in external document collections to
improve factuality (Li et al., 2024; Cai et al., 2024). However, as the size of the retrieval
corpus grows, RAG systems can suffer from “context noise” that increases hallucination
rates and lack internal confidence estimates to flag uncertain outputs (Wu et al., 2024a).
To overcome the limitations of conventional RAG systems, recent research has explored
adaptive retrieval strategies that react to model uncertainty. Iterative and gated retrieval
approaches (Jiang et al., 2025; Heydari et al., 2025) selectively filter external documents
and refine the evidence set across multiple rounds, showing improvements in factuality and
3

Continually Self-Improving Language Models for Bariatric Surgery QA
Question Category No. of Questions Percentage (%)
Preparation & Logistics 68 5.22
Surgical & Medical Info 263 20.20
Risks & Complications 221 16.97
Recovery & Lifestyle 296 22.73
Nutrition & Diet 102 7.83
Mental & Emotional Health 287 22.04
Cost & Insurance 65 4.99
Table 1: Distribution of bariatric surgery-related questions across high-level categories in
thebRAGq dataset. This table presents the number and percentage of questions
within each thematic category, highlighting the diverse informational needs of
patients throughout the bariatric surgery journey.
coherence. While these approaches mitigate irrelevant content and reduce hallucinations,
they remain fundamentally external—they treat retrieval as an auxiliary process (Lewis
et al., 2020) and stop short of modifying the model’s internal knowledge. As such, they lack
the capacity to assess and revise the model’s internal parameters in response to evolving
clinical evidence. This gap leaves current systems vulnerable to recurring errors when
previously seen topics reappear under different linguistic formulations.
In contrast, our proposed framework, bRAGgen , introduces an adaptive self-updating
mechanism that not only monitors response confidence but actively integrates validated,
up-to-date clinical information into the model itself. By embedding uncertainty detection
and retrieval within a continual learning loop, bRAGgen transitions from passive retrieval to
active knowledge refinement. This enables it to stay synchronized with the latest postop-
erative guidelines and avoid repeating outdated or incorrect responses over time.
Complementing this architecture is bRAGq , a rigorously curated dataset of 1 ,302 real-
world patient questions covering nutrition, lifestyle, complications, and mental health in
the postoperative MBS setting. Validated by board-certified bariatric surgeon, bRAGq offers
the first specialized benchmark to evaluate clinical QA systems beyond general-purpose
health datasets. Together, bRAGgen and bRAGq address the dual challenge of knowledge
obsolescence and domain specificity—paving the way for clinically grounded, scalable, and
responsive patient support in bariatric aftercare.
3. Dataset
We introduce bRAGq , a domain-specific dataset curated to reflect the breadth and depth of
questions commonly posed by bariatric surgery patients. Designed to support the develop-
ment of intelligent tools for patient education and clinical decision support, bRAGq captures
concerns spanning the entire surgical journey—from preoperative preparation to long-term
postoperative management. The dataset was constructed in close collaboration with board-
certified bariatric surgeons to ensure clinical validity and relevance, encompassing psycho-
logical, medical, and lifestyle-related questions. It comprises 1 ,302 total entries: 611 drawn
from PubMedQA (Jin et al., 2019), of which 201 were flagged by experts as not representa-
4

Continually Self-Improving Language Models for Bariatric Surgery QA
tive of everyday patient concerns, and 691 synthetically generated based on expert-informed
templates and real-world patient interactions.
The questions span a wide range of thematic categories, ensuring comprehensive cover-
age of key issues in bariatric care. These include pre-surgical considerations, intraoperative
topics, postoperative management, dietary guidance, mental health, and lifestyle adapta-
tion. As shown in Table 1, the largest proportion of questions fall under Recovery & Lifestyle
(22.73%), followed by Mental & Emotional Health (22.04%) and Surgical & Medical Info
(20.20%), reflecting the areas patients most frequently seek guidance on. Table 2 presents
representative examples from each category, illustrating the dataset’s granularity and di-
versity. By aligning with real-world patient priorities and clinical input, bRAGq provides a
rigorous benchmark for evaluating the performance of language models in delivering accu-
rate, trustworthy, and context-aware responses in the bariatric surgery domain. Beyond
benchmarking, it also serves as a valuable resource for training patient-facing conversa-
tional agents that are empathetic, evidence-informed, and sensitive to the unique needs of
this clinical population.
Question Category Sample Questions
Risks & Complications 1. Are vitamin D levels and bone turnover markers related to non-alcoholic
fatty liver disease in severely obese patients?
2. Does older age limit postbariatric surgery cognitive benefits: a preliminary
investigation?
Recovery & Lifestyle 1. Does clinical trial demonstrate exercise following bariatric surgery improves
insulin sensitivity?
2. Are serum markers of bone turnover increased at six and 18 months after
Roux-en-Y bariatric surgery: correlation with the reduction in leptin?
Preparation & Logistics 1. Does a Pre-Hospital Patient Education Program improve Outcomes of
Bariatric Surgery?
2. Does perioperative care map improve compliance with best practices for
the morbidly obese?
Surgical & Medical Info 1. Is laparoscopic gastric bypass superior to laparoscopic gastric banding for
treatment of morbid obesity?
2. Is potentially life-threatening sleep apnea unrecognized without aggressive
evaluation?
Cost & Insurance 1. Does medicare and Medicaid status predict prolonged length of stay after
bariatric surgery?
2. Is medication cost significantly reduced after Roux-en-Y gastric bypass in
obese patients?
Mental & Emotional
Health1. Are patient expectations of bariatric surgery gender specific – a prospective,
multicenter cohort study?
2. Is support group meeting attendance associated with better weight loss?
Nutrition & Diet 1. Does dehydroepiandrosterone-sulfate modify human fatty acid composition
of different adipose tissue depots?
2. Does low 25-hydroxyvitamin D affect insulin sensitivity in obesity after
bariatric surgery?
Table 2: Sample Questions for Each Bariatric Surgery Category. This table presents two
example questions from each major category within the bariatric surgery domain,
reflecting the primary concerns of patients throughout their surgical journey.
5

Continually Self-Improving Language Models for Bariatric Surgery QA
Vector Embed 
C: Semantic Cache G: Text Generation 
h1 h2 h3 hnL: Safety Check Hit  or  MissHigh Confidence Low Confidence 
Constraint 
Decoding Updated Embeds 
L: Online Learning 
h1h2h3 hl
Llama-3 
Adapter Layers Q
A
Figure 1: Architecture of the proposed method bRAGgen , The system integrates large lan-
guage models (eg. Llama3) with real-time web retrieval capabilities. When con-
fidence falls below the threshold ( α), the system automatically retrieves updated
information from authoritative medical sources to enhance response accuracy.
4. Proposed Methodology
In response to the growing need for timely, evidence-based decision support in clinical set-
tings, we propose an integrated framework that enhances retrieval-augmented generation
(RAG) with continuous online adaptation. Our system is specifically designed to provide
contextually relevant, accurate, and safe clinical recommendations by combining several key
components: a semantic cache, a multi-source web retrieval engine, an adaptive text gen-
eration module, and an online learning protocol. These components are carefully chosen to
address critical challenges in clinical decision-making, including rapid access to high-quality
medical information, comprehensive evidence retrieval from trusted sources, dynamic and
context-sensitive response generation, and continuous model refinement.
We formalize the framework as:
S= (C,R,G,L)
where each component is defined as follows:
•Crepresents the semantic cache, which ensures fast retrieval of relevant medical doc-
uments. It leverages SentenceTransformer embeddings and Faiss indexing for rapid
access to domain-specific information, minimizing latency.
•Ris the web-based multi-source retrieval engine, which uses a Markov Decision Pro-
cess (MDP) to focus on authoritative medical sources. The engine aggregates external
evidence from trusted web sources, enriching the context when the cache does not suf-
fice.
6

Continually Self-Improving Language Models for Bariatric Surgery QA
•Gis the adaptive text generation module, which employs low-rank adaptation (LoRA)
techniques to fine-tune a large pre-trained language model for domain-specific tasks.
This module tailors the generated responses to clinical contexts while maintaining
accuracy and compliance with domain constraints.
•Lis the online learning module, which continuously refines the model. It updates the
system with new data and interactions, enabling the framework to improve over time
and adapt to evolving clinical guidelines.
By combining these components, our framework offers a dynamic, evidence-based deci-
sion support system that remains adaptable and effective over time. Each module addresses
specific challenges, ensuring accurate, context-aware, and clinically safe recommendations
in real-time clinical environments.
4.1. Semantic Knowledge Caching
To minimize response latency and ensure the rapid retrieval of high-quality clinical evidence,
our system incorporates a semantic knowledge caching mechanism. This cache maintains a
collection of document-query pairs, denoted as D={(qi, di)}N
i=1, where each query qi∈R768
is generated using the BioClinicalBERT model and is paired with its corresponding clinical
document di. Given an input query q, the cache efficiently retrieves the document djthat
maximizes the cosine similarity between the query and document embeddings:
C(q) = arg max
dj∈Dq·qj
∥q∥∥qj∥subject toq·qj
∥q∥∥qj∥≥τc, (1)
where τc= 0.7 is a cosine similarity threshold that ensures only the most relevant documents
are retrieved.
To maintain the cache’s relevance, new query-document pairs ( q, d) are continuously
added. Obsolete entries are removed based on an eviction policy designed to prioritize
documents that remain valuable over time. Specifically, the cache is updated as follows:
D ← D ∪ { (q, d)} \ {arg min
dkψ(dk)}, (2)
where the eviction score ψ(dk) for a document dkis defined as:
ψ(dk) =αfu(dk) + (1 −α)e−t/β,
with α= 0.6,fu(dk) representing the document’s usage frequency, and e−t/βaccounting
for the document’s age, where tis the time since the last access, and βis document total
stored time.
This eviction policy ensures that frequently accessed and recent documents are retained
in the cache, while less relevant or outdated documents are pruned.
The caching mechanism is implemented using SentenceTransformer embeddings, with
Faiss indexing, enabling efficient similarity search. The cache is constrained by a fixed
size (e.g., 500 documents), ensuring fast retrieval and minimal computational overhead. By
maintaining a small, high-quality set of relevant documents, the semantic cache significantly
reduces response times during clinical decision support, providing timely access to critical
information.
7

Continually Self-Improving Language Models for Bariatric Surgery QA
4.2. Multi-Source Web Retrieval Engine
When the semantic cache does not provide sufficient information to answer a query, the
multi-source web retrieval engine is triggered to gather additional evidence from trusted
external sources. This component is designed to ensure that the system can access compre-
hensive, up-to-date information from diverse domains, including those not covered by the
cached documents. The retrieval process is framed as a Markov Decision Process (MDP),
which enables focused and dynamic crawling across medical websites to retrieve relevant
content.
At each time step t, the action attaken by the retrieval engine is determined by maxi-
mizing the expected cumulative reward, which is computed as follows:
at= arg max
a∈AX
s′P(s′|s, a)
R(s, a) +γV(s′)
, (3)
where the reward function R(s, a) is defined as:
R(s, a) =ITLD( s)∈{.gov,.edu}·BM25( s, q), (4)
where ITLD( s)∈{.gov,.edu}is an indicator function that ensures the retrieved documents are
from authoritative sources (i.e., websites with ‘.gov‘ or ‘.edu‘ top-level domains), while
BM25( s, q) is a content relevance score computed using the BM25 ranking function to
assess how well the document sanswers the query q.
The retrieval process is powered by the DuckDuckGo API, which allows for broad web
searches while prioritizing authoritative sources through URL domain filtering. The BM25
scoring system is applied to rank the retrieved documents based on their relevance to the
input query, ensuring that the most pertinent and reliable documents are selected. Once the
relevant documents are retrieved, they are incorporated into the semantic cache, thereby
enriching the context available for generating the system’s response.
This mechanism enhances the system’s ability to provide informed, evidence-based rec-
ommendations, particularly in scenarios where the cached knowledge does not suffice, by
tapping into the vast amount of publicly available, authoritative medical content across the
web.
4.3. Adaptive Text Generation
To generate accurate and contextually relevant clinical recommendations, we employ the
Llama3-8B model, enhanced using low-rank adaptation (LoRA). This approach is designed
to efficiently fine-tune a large pre-trained language model to domain-specific tasks, while
minimizing computational overhead and memory usage. The adaptation is performed in a
low-rank fashion, enabling the model to adjust quickly to specific medical domains without
the need for full retraining.
For each transformer layer lin the model, the adaptive hidden representation is com-
puted as follows:
hadapt
l=hbase
l+ ∆Wlhbase
l,∆Wl=BlAl, (5)
8

Continually Self-Improving Language Models for Bariatric Surgery QA
where Bl∈Rd×randAl∈Rr×dare the learned low-rank matrices, and r= 32 is the
rank used for adaptation. The matrices BlandAlcapture domain-specific information
while ensuring that the adaptation process remains efficient and scalable. The term hbase
l
represents the original, pre-trained hidden representation of the model at layer l.
To evaluate the quality of the generated response, we utilize a perplexity measure, which
quantifies the uncertainty in predicting the next token in the sequence:
P(r|q) = exp 
−1
TTX
t=1logpθ(rt|r<t, q)!
, (6)
where P(r|q) is the perplexity of the generated response rgiven the input query q, and
Tis the length of the response. The term pθ(rt|r<t, q) represents the model’s predicted
probability of the token rtat position t, conditioned on the preceding tokens and the query.
If the perplexity of the response exceeds a threshold τp= 4.5, it indicates that the
model’s output is not sufficiently confident or relevant. In such cases, the system triggers
additional retrieval and adaptation cycles to refine the response, improving its accuracy and
relevance by incorporating more domain-specific knowledge.
This adaptive approach ensures that the system can generate high-quality clinical rec-
ommendations that are both contextually appropriate and tailored to the specific needs of
the patient or healthcare provider.
4.4. Online Learning Protocol
To ensure that the model remains up-to-date and adaptable in the face of new evidence
and user interactions, we implement an online learning module. This protocol allows the
model to continuously refine its performance by integrating fresh data and adjusting its
parameters over time. The training objective is designed to balance the model’s ability
to predict accurate outcomes while avoiding overfitting to recent data, using a regularized
cross-entropy loss function:
Ladapt =E(q,d)∼B[−logpθ(d|q)] +λ∥ΘA∥2
F, (7)
where E(q,d)∼Brepresents the expectation over a mini-batch Bof query-document pairs,
pθ(d|q) is the predicted probability of document dgiven query q, and λis the regulariza-
tion parameter that controls the magnitude of the model’s parameters. The term ∥ΘA∥2
F
represents the Frobenius norm of the model’s adaptation parameters Θ A, which serves as a
regularizer to prevent overfitting during updates.
The experience buffer Bis updated dynamically to maintain a diverse and representa-
tive sample of query-document pairs. This buffer is managed using a Faiss-based nearest
neighbor search mechanism, which ensures that new samples are included in a way that
preserves diversity and reduces redundancy. The update rule is as follows:
B ← B ∪ { (qi, di)} \ {arg max
(qj,dj)sim(qj, qi)}, (8)
where sim( qj, qi) denotes the similarity between queries qjandqi, and we remove the
pair that is most similar to the newly added sample, ensuring the buffer contains varied
9

Continually Self-Improving Language Models for Bariatric Surgery QA
and non-redundant training data. This approach helps the model avoid memorizing specific
query-answer pairs and encourages generalization across a broad range of contexts.
The online learning module ensures that the system adapts in real-time to emerging
evidence, evolving patient needs, and new clinical knowledge. As a result, the model con-
tinually improves its performance, staying current with the latest developments and capable
of providing up-to-date, accurate recommendations.
4.5. Safety and Response Validation
Ensuring that generated outputs are both safe and clinically valid is of paramount im-
portance in our system. To achieve this, we apply constrained decoding during the text
generation process, which ensures that generated responses adhere to safety guidelines and
domain-specific constraints. The constrained decoding objective is formulated as follows:
rsafe= arg max
r∈V∗pθ(r|q)nY
i=1ϕi(ri), (9)
where ris the generated response, and V∗is the vocabulary space. Each constraint
function ϕi(ri) is designed to enforce specific safety requirements on individual tokens riin
the response. The constraint function ϕiis defined as:
ϕi(ri) =In
∄w∈ri:w∈ W speculativeo
, (10)
where Iis the indicator function, and Wspeculative is a set of words or phrases that
are deemed speculative or unsafe in a clinical context. This constraint ensures that the
generated response does not include any terms or statements that might mislead patients
or suggest unverified clinical practices.
To ensure clinical validity, we further compare outputs with reference texts using BERTScore
(Zhang et al., 2020), which evaluates semantic similarity via contextual embeddings. To-
gether, constrained decoding and BERTScore validation ensure that responses remain both
safe and aligned with evidence-based clinical content.
5. Baselines
We evaluated several baseline models to assess their performance on our medical question-
answering benchmark:
(i)RAG2(Sohn et al., 2024) relies on a pre-cached offline corpus of biomedical documents
for retrieval, avoiding real-time web queries. It enhances standard RAG methods by using
perplexity-based labels and LLM-generated rationales to selectively retrieve and filter con-
text, improving relevance and reducing noise. (ii) MedGraphRAG (Wu et al., 2024b) also
uses an offline cache of biomedical documents for retrieval. It further integrates a structured
medical knowledge graph to guide the retrieval process, leveraging clinical relationships to
improve the contextual relevance and factual accuracy of the retrieved information. (iii)
Llama3-8B (Grattafiori et al., 2024) is a 8-billion-parameter large language model eval-
uated under two configurations: (a) Zero-shot , where the model responds using only its
pre-trained knowledge; and (b) Context-prompted , where external context retrieved from
offline sources is appended to the prompt to improve answer quality. (iv) Phi-3 (Abdin
10

Continually Self-Improving Language Models for Bariatric Surgery QA
et al., 2024) is a lightweight 3.8-billion-parameter model optimized for efficiency and edge
deployment. We evaluate Phi-3 in both (a) Zero-shot and (b) Context-prompted modes,
assessing its ability to handle medical queries with and without retrieval-based augmenta-
tion. (v) Mistral Instruct (Jiang et al., 2023) is a 7-billion-parameter instruction-tuned
model designed for strong performance on alignment-focused tasks. It is tested in (a) Zero-
shot mode, where it relies solely on instruction tuning, and (b) Context-prompted mode,
where it incorporates retrieved medical content to guide its responses.
6. Experimental Setup
We evaluate our proposed bRAGgen model through both expert human evaluation and an
LLM-as-Judge protocol. Our goal is to assess the clinical quality of responses generated by
various model configurations, focusing on three key axes: factual accuracy, clinical relevance,
and comprehensiveness, in the context of bariatric surgery patient education.
We benchmark four categories of systems: (i) Offline RAG , which includes a standard
retrieval-augmented generation baseline and a domain-tuned MedGraphRAG variant using
graph-based retrieval; (ii) Zero-shot LLMs , where large language models (Llama3-8B,
Phi-3, and Mistral Instruct) generate answers without additional context; (iii) Context-
Prompted LLMs , where retrieved context is appended at inference time without param-
eter updates; and (iv) bRAGgen (Proposed) , which applies confidence-aware parametric
updates using retrieved evidence. All systems are tested under identical conditions and use
a shared retrieval pipeline where applicable.
For expert evaluation, we consult one board-certified bariatric surgeon, who reviewed
model outputs for 105 instances. Each response is scored independently across three di-
mensions: Factuality (accuracy and correctness), Clinical Relevance (appropriateness
in a clinical context), and Comprehensiveness (completeness and informativeness for pa-
tients). Ratings are based on a 5-point Likert scale, with final scores reported as averages
across questions.
To complement the expert review and enable scalable comparison, we also evaluate
all models using an LLM-as-Judge setup, where ChatGPT-4o2is prompted with each
question, the corresponding model-generated answer, and a rubric defining the evaluation
criteria. The model then rates each answer on the same 1–5 scale. To assess the reliability
of this proxy, we compute the rank correlation between expert and ChatGPT-4o scores and
observe a strong alignment ( ρ= 0.94), confirming the viability of using LLMs for early-stage
model quality assessment.
7. Results
We evaluate our proposed model, bRAGgen , using both expert evaluation review by board-
certified bariatric surgeon and LLM-as-Judge using ChatGPT-4o. We compare bRAGgen
against a suite of baselines, including standard retrieval-augmented models, zero-shot LLMs,
and context-prompted variants.
2. https://openai.com/index/hello-gpt-4o/
11

Continually Self-Improving Language Models for Bariatric Surgery QA
System Metrics
Type Model Factuality Clinical Rel. Compre. Avg
Offline RAG23.62 3.45 3.53 3.53
MedGraphRAG 3.85 3.92 4.38 4.05
Zero-shot Llama3-8B 3.41 3.25 3.46 3.37
Phi-3 2.37 2.15 2.25 2.26
Mistral instruct 2.23 2.18 2.14 2.18
Context prompted Llama3-8B 3.82 3.91 4.34 4.02
Phi-3 2.64 2.75 2.42 2.60
Mistral instruct 2.69 2.37 3.35 2.80
bRAGgen with Llama3-8B 4.18 4.58 4.76 4.51
Phi-3 2.87 3.17 2.60 2.88
Mistral instruct 2.95 2.71 3.63 3.09
Table 3: Evaluation of various models across different configurations by expert surgeons.
Each system is evaluated on Factuality, Clinical Relevance (Clinical Rel.), and
Comprehensiveness (Compre.), with scores from 1 (poor) to 5 (excellent). The
final three rows under each model group show the performance of our Online
bRAGgen setting. The ’Avg’ column reports the average of the three evaluation
metrics.
7.1. Expert Evaluation
To assess the clinical quality of generated responses, we conducted a blinded evaluation
with board-certified bariatric surgeon, who rated model outputs across three dimensions:
Factuality ,Clinical Relevance , and Comprehensiveness , using a 1–5 Likert scale
(higher is better). Table 3 presents the average scores for each system under multiple
configurations.
Among all baselines, MedGraphRAG , an offline domain-specific RAG model, achieved
the highest average score (4.05), outperforming both standard offline RAG baselines (RAG2:
3.53) and all zero-shot models (Llama3-8B: 3.37; Phi-3: 2.26; Mistral: 2.18). Context-
prompted models (i.e., inputting relevant question context during inference) moderately
improved scores, especially for Llama3-8B (Avg: 4.02), but still fell short of delivering
optimal factual and clinical consistency.
Our proposed bRAGgen framework delivered the best overall performance across all met-
rics. When paired with Llama3-8B, bRAGgen achieved the highest average score (4.51),
with near-expert level performance on Comprehensiveness (4.76) and Clinical Rele-
vance (4.58). Notably, bRAGgen also improved the performance of smaller models like Phi-3
and Mistral, elevating their average scores by +0.6–0.7 points compared to their context-
prompted or zero-shot baselines. These gains highlight the effectiveness of our confidence-
aware updating mechanism, which not only retrieves up-to-date clinical evidence but also
integrates it into the model’s internal parameters, enabling more robust, domain-adapted
reasoning.
12

Continually Self-Improving Language Models for Bariatric Surgery QA
These results demonstrate that bRAGgen significantly enhances the clinical utility of
LLMs across model sizes, especially when compared to conventional static RAG setups or
prompting-only strategies.
System Metrics
Type Model Factuality Clinical Rel. Compre. Avg
Offline RAG23.49 3.28 3.36 3.38
MedGraphRAG 3.67 3.76 4.45 3.96
Zero-shot Llama3-8B 3.28 3.18 3.34 3.27
Phi-3 2.24 2.07 2.12 2.14
Mistral instruct 2.17 2.11 2.01 2.10
Context prompted Llama3-8B 3.67 3.76 4.45 3.96
Phi-3 2.49 2.61 2.38 2.49
Mistral instruct 2.57 2.24 3.21 2.67
bRAGgen with Llama3-8B 4.03 4.43 4.87 4.44
Phi-3 2.73 3.03 2.54 2.77
Mistral instruct 2.83 2.58 3.48 2.96
Table 4: Evaluation of various models across different configurations using LLM-as-Judge
metrics. Each system is evaluated on Factuality, Clinical Relevance (Clinical Rel.),
and Comprehensiveness (Compre.), with scores from 1 (poor) to 5 (excellent).
The final three rows under each model group show the performance of our Online
bRAGgen setting. The ’Avg’ column reports the average of the three evaluation
metrics.
7.2. LLM-as-Judge Evaluation
To complement expert evaluation, we further assess all models using an LLM-as-Judge
framework, where we use ChatGPT-4o model scores responses along three axes: Factuality ,
Clinical Relevance , and Comprehensiveness , using a 5-point Likert scale. Table 4
summarizes the performance of baseline systems and our proposed bRAGgen across these
dimensions.
Among the baselines, MedGraphRAG and context-prompted Llama3-8B show relatively
strong performance, achieving average scores of 3 .96. However, our proposed bRAGgen ap-
proach consistently outperforms all baselines across all metrics and models. For instance,
bRAGgen with Llama3-8B achieves the highest overall score of 4 .44, reflecting substantial
improvements in factual correctness (+0 .36), clinical relevance (+0 .67), and comprehensive-
ness (+0 .42) over the best-performing baseline.
Notably, even with smaller models like Phi-3 and Mistral instruct, bRAGgen enhances
output quality, particularly in relevance and completeness. These findings demonstrate
that our system not only boosts performance for high-capacity LLMs but also meaningfully
improves the reliability of lightweight models, making it practical for resource-constrained
settings.
13

Continually Self-Improving Language Models for Bariatric Surgery QA
<-2 -2-(-1) -1-0 0-1 1-2 >22040
Confidence ChangeCounta) Confidence Change Distribution
60 80 100 120 140PubMedPMCNIH
Countb) Top Search Domains
0200 400 600 8001,0001,200123
IterationLoss Valuec) Training Loss Progression
0-10 10-20 20-30 30-40 40+0204060
Duration (sec)Countd) Total Duration Distribution
Figure 2: Exploratory Analysis of Model Editing Dynamics. (a) Distribution of changes
in confidence scores post-edit, showing that most changes are modest and pos-
itive. (b) Frequency of search queries across external biomedical domains, with
PubMed dominating. (c) Training loss progression across iterations, illustrating
convergence patterns and volatility. (d) Distribution of total duration taken for
each edit operation, highlighting that most edits are executed within 10-20 sec-
onds.
7.3. Expert vs. LLM-as-Judge: Score Alignment
To assess the alignment between human and expert evaluation, we compare expert ratings
(Table 3) with those produced by the LLM-as-Judge framework (Table 4) across all models
and configurations.
Overall, we observe a high degree of consistency in relative rankings across systems.
For instance, both experts and the LLM-as-Judge identify MedGraphRAG and context-
prompted Llama3-8B as the strongest baselines, while zero-shot models like Phi-3 and
Mistral perform the worst across all axes. Furthermore, our proposed bRAGgen yields the
highest scores in both evaluation schemes, affirming its robustness across human and model-
based judgments.
14

Continually Self-Improving Language Models for Bariatric Surgery QA
Metric-wise, the strongest agreement is seen in the Comprehensiveness andClinical
Relevance dimensions, where score trends closely track each other across settings. Some
minor variation arises in the Factuality scores, particularly for models like Phi-3 and
Mistral, where the LLM-as-Judge is slightly more conservative than human reviewers. This
discrepancy likely stems from the LLM’s heightened sensitivity to surface-level inaccuracies,
compared to domain experts who may weigh overall clinical soundness more heavily.
Importantly, the average correlation between expert and LLM-as-Judge scores across all
models is ρ= 0.94 (Spearman), underscoring the reliability of using LLMs as surrogate eval-
uators in low-resource or iterative development settings. These results suggest that LLM-
as-Judge provides a scalable and reasonably aligned proxy for expert review—particularly
useful for rapid benchmarking and ablation testing during system development.
8.bRAGgen Analysis
Figure 2 presents a comprehensive analysis across multiple dimensions to evaluate the be-
havior, responsiveness, and efficiency of bRAGgen during real-time knowledge integration in
the context of bariatric care.
(a) Confidence Change Distribution. The histogram in Fig. 2.a illustrates the
distribution of confidence score changes triggered by the adaptive retrieval mechanism.
Most examples exhibit moderate confidence gains (bins 0-1and 1-2), with the highest
concentration in the 0-1bin. This confirms that the system’s dynamic thresholding yields
frequent yet stable updates, allowing the model to autonomously improve responses without
overcorrecting. Rare occurrences of extreme confidence shifts ( <-2or>2) indicate that the
system maintains a conservative stance, prioritizing stability in medical contexts.
(b) Top Search Domains. To assess external evidence sources, we analyzed the
frequency of domain-level API queries. PubMed ,PMC, and NIHemerged as the top knowledge
sources (cf. Fig 2.b), underscoring bRAGgen ’s strong preference for authoritative biomedical
repositories. This supports the design goal of maintaining clinical fidelity and alignment
with evidence-based guidelines during patient-facing interactions.
(c) Training Loss Progression. The loss trajectory (cf. Fig 2.c) across 1,302 iter-
ations reveals several key phases in the model’s learning dynamics. An initial sharp drop
from 2 .71 to 1 .32 by iteration 50 is followed by a spike to 3 .13 at iteration 100, likely due to
early exploratory updates. Subsequent iterations show improved stability and convergence,
with the lowest loss (0 .86) reached around iteration 500. Notable local minima at iterations
300 and 600 indicate consistent refinement, while the uptick at iteration 1000 may reflect
a transient deviation before re-stabilization. Overall, the pattern confirms that bRAGgen ’s
adaptive updating mechanism supports gradual convergence while accommodating knowl-
edge volatility.
(d) Total Duration Distribution. The majority of update operations complete
within 10-20 seconds , with fewer cases extending beyond 30 seconds . This distribution
validates that bRAGgen ’s self-updating pipeline is both computationally lightweight and
suitable for real-time deployment in longitudinal MBS care settings ensuring timely and
trustworthy information delivery across all stages of the surgical journey.
Further qualitative comparisons across diverse questions and model outputs are pre-
sented in Appendix B.
15

Continually Self-Improving Language Models for Bariatric Surgery QA
9. Conclusion
We introduced bRAGgen , an adaptive retrieval-augmented generation (RAG) system tai-
lored for bariatric and metabolic surgery (MBS) support. By autonomously incorporating
real-time medical evidence when confidence dips below dynamic thresholds, bRAGgen en-
sures that responses remain timely, accurate, and clinically reliable. To facilitate robust
benchmarking, we also introduced bRAGq , the first large-scale, expert-validated dataset of
postoperative bariatric care questions. Through comprehensive evaluation using both LLM-
based metrics and expert surgeon assessments, bRAGgen consistently outperformed existing
state-of-the-art models in clinical accuracy and relevance.
References
Marah Abdin, Jyoti Aneja, Hany Awadalla, and et al. Phi-3 technical report: A highly
capable language model locally on your phone, 2024. URL https://arxiv.org/abs/
2404.14219 .
Daniel Alexander Alber, Zihao Yang, Anton Alyakin, Eunice Yang, Sumedha Rai, Aly A
Valliani, Jeff Zhang, Gabriel R Rosenbaum, Ashley K Amend-Thomas, David B Kurland,
Caroline M Kremer, Alexander Eremiev, Bruck Negash, Daniel D Wiggan, Michelle A
Nakatsuka, Karl L Sangwon, Sean N Neifert, Hammad A Khan, Akshay Vinod Save,
Adhith Palla, Eric A Grin, Monika Hedman, Mustafa Nasir-Moin, Xujin Chris Liu,
Lavender Yao Jiang, Michal A Mankowski, Dorry L Segev, Yindalon Aphinyanaphongs,
Howard A Riina, John G Golfinos, Daniel A Orringer, Douglas Kondziolka, and Eric Karl
Oermann. Medical large language models are vulnerable to data-poisoning attacks. Nat.
Med., 31(2):618–626, February 2025.
Yash Kumar Atri, Shraman Pramanick, Vikram Goyal, and Tanmoy Chakraborty. See, hear,
read: Leveraging multimodality with guided attention for abstractive text summarization.
Knowledge-Based Systems , 227:107152, 2021. ISSN 0950-7051. doi: https://doi.org/10.
1016/j.knosys.2021.107152. URL https://www.sciencedirect.com/science/article/
pii/S0950705121004159 .
Yash Kumar Atri, Vikram Goyal, and Tanmoy Chakraborty. Fusing multimodal signals
on hyper-complex space for extreme abstractive text summarization (tl;dr) of scientific
contents. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery
and Data Mining , KDD ’23, page 3724–3736, New York, NY, USA, 2023a. Association
for Computing Machinery. ISBN 9798400701030. doi: 10.1145/3580305.3599830. URL
https://doi.org/10.1145/3580305.3599830 .
Yash Kumar Atri, Vikram Goyal, and Tanmoy Chakraborty. Multi-document sum-
marization using selective attention span and reinforcement learning. IEEE/ACM
Transactions on Audio, Speech, and Language Processing , 31:3457–3467, 2023b. doi:
10.1109/TASLP.2023.3316459.
Yash Kumar Atri, Arun Iyer, Tanmoy Chakraborty, and Vikram Goyal. Promoting topic
coherence and inter-document consorts in multi-document summarization via simplicial
16

Continually Self-Improving Language Models for Bariatric Surgery QA
complex and sheaf graph. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Pro-
ceedings of the 2023 Conference on Empirical Methods in Natural Language Processing ,
pages 2154–2166, Singapore, December 2023c. Association for Computational Linguis-
tics. doi: 10.18653/v1/2023.emnlp-main.133. URL https://aclanthology.org/2023.
emnlp-main.133/ .
Romain Barres, Henri Kirchner, Morten Rasmussen, Jing Yan, Daniel Kantor, Anna Krook,
Erik Naslund, Juleen R. Zierath, and Charlotte Ling. Weight loss after gastric bypass
surgery in human obesity remodels promoter methylation. Cell Reports , 3(4):1020–1027,
2013. doi: 10.1016/j.celrep.2013.03.019.
Emily M. Bartholomay, Patrick W. Stewart, David B. Sarwer, Thomas A. Wadden, and
Anthony N. Fabricatore. Sociodemographic factors related to bariatric follow-up ap-
pointment attendance and weight outcomes. Surgery for Obesity and Related Diseases ,
20:1388–1395, 2024. doi: 10.1016/j.soard.2024.02.010.
Jean-Christophe B´ elisle-Pipon. Why we need to be careful with LLMs in medicine. Front.
Med. (Lausanne) , 11:1495582, December 2024.
Kristin K. Bjerkan, Audun Viste, Else M. Aasheim, Oda Mj˚ aland, Torstein Mala, Nina E.
Kløw, Jo Røislien, and Siv K. Bøhn. The long-term impact of postoperative educational
programs on weight loss after roux-en-y gastric bypass. Obesity Surgery , 32:3005–3012,
2022. doi: 10.1007/s11695-022-05913-0.
Tianchi Cai, Zhiwen Tan, Xierui Song, Tao Sun, Jiyan Jiang, Yunqi Xu, Yinger Zhang, and
Jinjie Gu. Forag: Factuality-optimized retrieval augmented generation for web-enhanced
long-form question answering. In Proceedings of the 30th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining , KDD ’24, page 199–210. ACM, August 2024. doi:
10.1145/3637528.3672065. URL http://dx.doi.org/10.1145/3637528.3672065 .
Jeffrey Cheng, Marc Marone, Orion Weller, Dawn Lawrie, Daniel Khashabi, and Ben-
jamin Van Durme. Dated data: Tracing knowledge cutoffs in large language models,
2024. URL https://arxiv.org/abs/2403.12958 .
Bryan Clapp, Lillian Khaitan, William J. English, Michel Gagner, William B. Inabnet,
J. Michael Morton, Walter J. Pories, Philip R. Schauer, Brian M. Wolfe, and Mary M.
Wolfe. American society for metabolic and bariatric surgery 2022 estimate of metabolic
and bariatric procedures performed in the united states. Surgery for Obesity and Related
Diseases , 20:425–431, 2024. doi: 10.1016/j.soard.2023.11.005.
Lauren A. David, Ivana Sijercic, and Stephanie E. Cassin. Preoperative and post-operative
psychosocial interventions for bariatric surgery patients: A systematic review. Obesity
Reviews , 21:e12926, 2020. doi: 10.1111/obr.12926.
Alvin Dey, Tanya Chowdhury, Yash Kumar Atri, and Tanmoy Chakraborty. Corpora
evaluation and system bias detection in multi-document summarization. In Trevor
17

Continually Self-Improving Language Models for Bariatric Surgery QA
Cohn, Yulan He, and Yang Liu, editors, Findings of the Association for Computa-
tional Linguistics: EMNLP 2020 , pages 2830–2840, Online, November 2020. Associa-
tion for Computational Linguistics. doi: 10.18653/v1/2020.findings-emnlp.254. URL
https://aclanthology.org/2020.findings-emnlp.254/ .
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei
Sun, Meng Wang, and Haofen Wang. Retrieval-augmented generation for large language
models: A survey, 2024. URL https://arxiv.org/abs/2312.10997 .
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, and et al. The llama 3 herd of
models, 2024. URL https://arxiv.org/abs/2407.21783 .
Kristen D. Groller, Kasey M. Carey, and John P. Evans. Systematic review of patient
education practices in weight loss surgery. Surgery for Obesity and Related Diseases , 13:
1072–1085, 2017. doi: 10.1016/j.soard.2017.02.025.
Mohammad Hassan Heydari, Arshia Hemmat, Erfan Naman, and Afsaneh Fatemi. Context
awareness gate for retrieval augmented generation, 2025. URL https://arxiv.org/abs/
2411.16133 .
Sara Javanparast, Leigh Roeger, Yuen Kwok, and Richard L Reed. The experience of
australian general practice patients at high risk of poor health outcomes with telehealth
during the COVID-19 pandemic: a qualitative study. BMC Fam. Pract. , 22(1):69, April
2021.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh
Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lu-
cile Saulnier, L´ elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao,
Thibaut Lavril, Thomas Wang, Timoth´ ee Lacroix, and William El Sayed. Mistral 7b,
2023. URL https://arxiv.org/abs/2310.06825 .
Zhouyu Jiang, Mengshu Sun, Lei Liang, and Zhiqiang Zhang. Retrieve, summarize, plan:
Advancing multi-hop question answering with an iterative approach, 2025. URL https:
//arxiv.org/abs/2407.13101 .
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. PubMedQA:
A dataset for biomedical research question answering. In Kentaro Inui, Jing Jiang, Vin-
cent Ng, and Xiaojun Wan, editors, Proceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP) , pages 2567–2577, Hong Kong, China,
November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1259.
URL https://aclanthology.org/D19-1259/ .
Julien Khlaut, Corentin Dancette, Elodie Ferreres, Benani Alaedine, Herent Herent, and
Pierre Manceron. Efficient medical question answering with knowledge-augmented ques-
tion generation. In Tristan Naumann, Asma Ben Abacha, Steven Bethard, Kirk Roberts,
18

Continually Self-Improving Language Models for Bariatric Surgery QA
and Danielle Bitterman, editors, Proceedings of the 6th Clinical Natural Language Process-
ing Workshop , pages 10–20, Mexico City, Mexico, June 2024. Association for Computa-
tional Linguistics. doi: 10.18653/v1/2024.clinicalnlp-1.2. URL https://aclanthology.
org/2024.clinicalnlp-1.2/ .
Dong H. Kim, Melissa S. Davenport, Jia Gu, Kristen Ziegler, Patricia L. Ferguson,
and Timothy M. Farrell. Incidence, burden, and predictors of 11-month readmis-
sion in patients undergoing bariatric surgery. Obesity Surgery , 33:94–104, 2023. doi:
10.1007/s11695-022-06410-2.
Yunsoo Kim, Jinge Wu, Yusuf Abdulle, and Honghan Wu. MedExQA: Medical question
answering benchmark with multiple explanations. In Dina Demner-Fushman, Sophia
Ananiadou, Makoto Miwa, Kirk Roberts, and Junichi Tsujii, editors, Proceedings of the
23rd Workshop on Biomedical Natural Language Processing , pages 167–181, Bangkok,
Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/
2024.bionlp-1.14. URL https://aclanthology.org/2024.bionlp-1.14/ .
Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa.
Large language models are zero-shot reasoners, 2023. URL https://arxiv.org/abs/
2205.11916 .
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Na-
man Goyal, Heinrich K¨ uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨ aschel, Sebastian
Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp
tasks. In Proceedings of the 34th International Conference on Neural Information Pro-
cessing Systems , NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN
9781713829546.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich K¨ uttler, Mike Lewis, Wen tau Yih, Tim Rockt¨ aschel, Sebastian Riedel,
and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks,
2021. URL https://arxiv.org/abs/2005.11401 .
Jiarui Li, Ye Yuan, and Zehua Zhang. Enhancing llm factual accuracy with rag to counter
hallucinations: A case study on domain-specific queries in private knowledge-bases, 2024.
URL https://arxiv.org/abs/2403.10446 .
Ruth J. F. Loos and G. S. H. Yeo. The genetics of obesity: from discovery to biology.
Nature Reviews Genetics , 23:120–133, 2022. doi: 10.1038/s41576-021-00414-z.
Yuxing Lu, Xukai Zhao, and Jinzhuo Wang. ClinicalRAG: Enhancing clinical decision
support through heterogeneous knowledge retrieval. In Sha Li, Manling Li, Michael JQ
Zhang, Eunsol Choi, Mor Geva, Peter Hase, and Heng Ji, editors, Proceedings of the 1st
Workshop on Towards Knowledgeable Language Models (KnowLLM 2024) , pages 64–68,
Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.
18653/v1/2024.knowllm-1.6. URL https://aclanthology.org/2024.knowllm-1.6/ .
Sean McLennan, Ravi P. Kiran, Robert S. McLeod, Victor Li, Yue Guo, and David W. Birch.
Characteristics and outcomes for patients undergoing revisional bariatric surgery due to
19

Continually Self-Improving Language Models for Bariatric Surgery QA
persistent obesity: a retrospective cohort study of 10,589 patients. Surgical Endoscopy ,
37:4613–4622, 2023. doi: 10.1007/s00464-022-09715-9.
Jeffrey I. Mechanick, Caroline Apovian, Stephen Brethauer, W. Timothy Garvey, Aaron M.
Joffe, Jill Kim, Robert F. Kushner, John R. Pender, Kalpana Singh, American Association
of Clinical Endocrinologists, American College of Endocrinology, The Obesity Society,
American Society for Metabolic, Bariatric Surgery, Obesity Medicine Association, and
American Society of Anesthesiologists. Clinical practice guidelines for the perioperative
nutrition, metabolic, and nonsurgical support of patients undergoing bariatric procedures
2019 update. Obesity (Silver Spring) , 28:O1–O58, 2020. doi: 10.1002/oby.22792.
Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher,
Xavier Amatriain, and Jianfeng Gao. Large language models: A survey, 2025. URL
https://arxiv.org/abs/2402.06196 .
Jeyanthi P, Gulothungan G, Vinoth Kumar V, Hitesh Chopra, and Talha Bin Emran.
Enhancing postoperative care with telemedicine and remote monitoring for improved
recovery and patient safety. Int. J. Surg. , 110(12):8205–8206, December 2024.
Ben Patel and Arron Thind. Usability of mobile health apps for postoperative care: Sys-
tematic review. JMIR Perioper. Med. , 3(2):e19099, July 2020.
Nadia Saeed. MediFact at MEDIQA-M3G 2024: Medical question answering in dermatol-
ogy with multimodal learning. In Tristan Naumann, Asma Ben Abacha, Steven Bethard,
Kirk Roberts, and Danielle Bitterman, editors, Proceedings of the 6th Clinical Natural
Language Processing Workshop , pages 339–345, Mexico City, Mexico, June 2024. As-
sociation for Computational Linguistics. doi: 10.18653/v1/2024.clinicalnlp-1.31. URL
https://aclanthology.org/2024.clinicalnlp-1.31/ .
Francisco Schlottmann, Charlie Baz, Adeel Pirzada, and Muhammad A. Masrur. Postop-
erative follow-up compliance: The achilles’ heel of bariatric surgery. Obesity Surgery , 33:
2945–2948, 2023. doi: 10.1007/s11695-023-06500-2.
Saeed A. Setarehdan, Ameneh Ziaee, Ramin Niknam, Sahar Ghassemi, Fatemeh Mehrabi,
Mohsen Hossein Esfahani, and Mohammad H. Shariat. Determinants of patient’s ad-
herence to the predefined follow-up visits after bariatric surgery. Obesity Surgery , 33:
577–584, 2023. doi: 10.1007/s11695-022-06353-3.
Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Mohamed Amin,
Le Hou, Kevin Clark, Stephen R Pfohl, Heather Cole-Lewis, Darlene Neal, Qazi Ma-
munur Rashid, Mike Schaekermann, Amy Wang, Dev Dash, Jonathan H Chen, Nigam H
Shah, Sami Lachgar, Philip Andrew Mansfield, Sushant Prakash, Bradley Green, Ewa
Dominowska, Blaise Ag¨ uera Y Arcas, Nenad Tomaˇ sev, Yun Liu, Renee Wong, Christo-
pher Semturs, S Sara Mahdavi, Joelle K Barral, Dale R Webster, Greg S Corrado, Yossi
Matias, Shekoofeh Azizi, Alan Karthikesalingam, and Vivek Natarajan. Toward expert-
level medical question answering with large language models. Nat. Med. , 31(3):943–950,
March 2025.
20

Continually Self-Improving Language Models for Bariatric Surgery QA
Jiwoong Sohn, Yein Park, Chanwoong Yoon, Sihyeon Park, Hyeon Hwang, Mujeen Sung,
Hyunjae Kim, and Jaewoo Kang. Rationale-guided retrieval augmented generation for
medical question answering, 2024. URL https://arxiv.org/abs/2411.00300 .
Ekaterina Sviridova, Anar Yeginbergen, Ainara Estarrona, Elena Cabrio, Serena Villata,
and Rodrigo Agerri. CasiMedicos-arg: A medical question answering dataset annotated
with explanatory argumentative structures. In Yaser Al-Onaizan, Mohit Bansal, and Yun-
Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing , pages 18463–18475, Miami, Florida, USA, November 2024. Asso-
ciation for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.1026. URL
https://aclanthology.org/2024.emnlp-main.1026/ .
Alex Tamkin, Miles Brundage, Jack Clark, and Deep Ganguli. Understanding the capa-
bilities, limitations, and societal impact of large language models, 2021. URL https:
//arxiv.org/abs/2102.02503 .
Juraj Vladika and Florian Matthes. Improving health question answering with reliable and
time-aware evidence retrieval. In Kevin Duh, Helena Gomez, and Steven Bethard, ed-
itors, Findings of the Association for Computational Linguistics: NAACL 2024 , pages
4752–4763, Mexico City, Mexico, June 2024. Association for Computational Linguis-
tics. doi: 10.18653/v1/2024.findings-naacl.295. URL https://aclanthology.org/2024.
findings-naacl.295/ .
Xiaoye Wang, Nicole Xi Zhang, Hongyu He, Trang Nguyen, Kun-Hsing Yu, Hao Deng,
Cynthia Brandt, Danielle S. Bitterman, Ling Pan, Ching-Yu Cheng, James Zou, and
Dianbo Liu. Safety challenges of ai in medicine in the era of large language models, 2025.
URL https://arxiv.org/abs/2409.18968 .
Zachary J. Ward, Sara N. Bleich, Angie L. Cradock, Jonathan L. Barrett, Christine M.
Giles, Charles Flax, and Michael W. Long. Projected u.s. state-level prevalence of adult
obesity and severe obesity. The New England Journal of Medicine , 381:2440–2450, 2019.
doi: 10.1056/NEJMsa1909301.
Jinyang Wu, Feihu Che, Chuyuan Zhang, Jianhua Tao, Shuai Zhang, and Pengpeng Shao.
Pandora’s box or aladdin’s lamp: A comprehensive analysis revealing the role of rag noise
in large language models, 2024a. URL https://arxiv.org/abs/2408.13533 .
Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min Xu, Filippo Menolascina, and Vicente
Grau. Medical graph rag: Towards safe medical large language model via graph retrieval-
augmented generation, 2024b. URL https://arxiv.org/abs/2408.04187 .
Weizi Wu, Teresa Graziano, Andrew Salner, Ming-Hui Chen, Michelle P Judge, Xiaomei
Cong, and Wanli Xu. Acceptability, effectiveness, and roles of mhealth applications in
supporting cancer pain self-management: Integrative review. JMIR MHealth UHealth ,
12:e53652, July 2024c.
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. Bertscore:
Evaluating text generation with bert, 2020. URL https://arxiv.org/abs/1904.09675 .
21

Continually Self-Improving Language Models for Bariatric Surgery QA
Wan Zhang and Jing Zhang. Hallucination mitigation for retrieval-augmented large lan-
guage models: A review. Mathematics , 13(5):856, March 2025.
22

Continually Self-Improving Language Models for Bariatric Surgery QA
Appendix A. Limitations and Future Directions
While our proposed modular editing framework with LLaMA-3 demonstrates strong perfor-
mance across factuality and faithfulness benchmarks, several limitations remain that open
avenues for future research.
Scalability and Edit Interference. Although our method enables localized and composi-
tional edits, the cumulative impact of many such edits—particularly in dense regions of the
representation space—may lead to interference or capacity saturation. Future work could
explore dynamic pruning or hierarchical edit graphs to manage long-term scalability.
Generalization Beyond Local Context. Our framework is effective at modifying behav-
iors related to specific factual updates. However, its ability to generalize edits to broader
semantic or reasoning contexts is limited. Integrating structured world knowledge or train-
ing the modules with auxiliary objectives (e.g., counterfactual consistency) may further
enhance generalization.
Evaluation Bottlenecks. While our automatic and human evaluations provide evidence
of improved edit quality, assessing real-world applicability—especially in high-stakes do-
mains—requires more robust metrics. Developing editing-specific benchmarks that reflect
user intent, edit trustworthiness, and long-term retention would be a valuable direction.
Interpretability. Currently, our system relies on confidence thresholds and auxiliary
metrics to determine performance improvements. However, the interpretability of these
decisions is limited. Future work could incorporate explainable verification layers or causal
probing techniques to better surface why an information update is accepted or rejected.
Integration with Continual Learning. Finally, our system performs updates without
modifying the full model weights, but it does not yet integrate naturally with continual
learning settings. A promising direction is to unify model editing and continual finetuning
via shared memory or adaptation modules, enabling long-term knowledge evolution.
Appendix B. Qualitative Analysis of Model Inferences
To assess the effectiveness of bRAGgen (with Llama3-8B), we conducted a qualitative com-
parison across five diverse bariatric FAQ examples (Tables 5–9). These examples cover a
range of clinical and lifestyle concerns, including pregnancy, operative duration, hygiene
practices, social challenges, and comorbidities like diabetes. Our analysis highlights the
strengths of bRAGgen in terms of medical grounding, relevance, and clarity.
Clinical Accuracy and Depth. In Example 5, which explores the impact of pregnancy
on post-surgical outcomes, bRAGgen offers a grounded, evidence-informed response. Unlike
Llama3-8B Zero-shot, which introduces speculative behavioral theories without citations,
and the prompted Llama3-8B, which fails to answer the question altogether, bRAGgen ap-
propriately cites observed risks (e.g., gestational diabetes, weight regain), aligning with
known literature and conveying clinical nuance.
Factual Precision. Example 6 evaluates responses to a straightforward procedural ques-
tion. While other baselines provide either incomplete answers (e.g., Llama3-8B’s halluci-
nated boilerplate text) or vague generalizations, bRAGgen provides precise operative time
ranges for specific bariatric procedures and cites published studies with estimated durations.
This specificity reflects high factual precision and strengthens trust in the response.
23

Continually Self-Improving Language Models for Bariatric Surgery QA
Guideline Awareness. In Example 7, concerning post-surgical hygiene timelines, bRAGgen
references ASMBS (American Society for Metabolic and Bariatric Surgery) guidelines,
demonstrating awareness of authoritative recommendations. In contrast, Llama3-8B vari-
ants show significant variability in recovery timelines, and MedGraphRAG lacks procedural
differentiation.
Social Sensitivity. Example 8 tackles social scenarios, such as attending events with
restricted food options. While all models provide socially sensitive advice, bRAGgen balances
empathy with medical priorities, reinforcing the importance of dietary adherence post-
surgery while avoiding patronizing language. This contrasts with Llama3-8B Prompted,
which lists overly detailed options and drifts from concise response expectations.
Medical Benefit Articulation. Finally, in Example 9, focused on diabetes outcomes,
bRAGgen articulates the clinical mechanisms (e.g., gut hormone changes, weight loss) and
cites outcomes such as remission rates and microvascular benefits. Compared to other
models, which either stop mid-sentence (Llama3-8B Zero-shot) or generalize improvements,
bRAGgen offers a complete and balanced view of bariatric surgery’s benefits in diabetic
populations.
Summary. Across diverse question types, bRAGgen with Llama3-8B consistently provides
accurate, context-sensitive, and medically aligned responses. It avoids hallucinations, in-
corporates citations or guideline references when appropriate, and maintains patient-centric
tone and relevance, thereby outperforming both zero-shot and context-prompted Llama3-8B
variants as well as the offline MedGraphRAG baseline.
24

Continually Self-Improving Language Models for Bariatric Surgery QA
Question Does pregnancy influence long-term results of bariatric surgery?
MedGraphRAG
(Offline)Pregnancy does not appear to significantly affect the long-term outcomes of bariatric
surgery, such as weight loss and resolution of comorbidities like type 2 diabetes and
hypertension. However, certain considerations are necessary during pregnancy fol-
lowing bariatric surgery, including adequate nutrition and close monitoring by a
healthcare provider to ensure a healthy pregnancy and delivery. Additionally, preg-
nancies following sleeve gastrectomy may be associated with an increased risk of
gestational hypertension and preterm birth compared to normal-weight pregnancies.
It’s essential for women who have undergone bariatric surgery to discuss their indi-
vidual situation and any concerns with their healthcare provider.
Llama3-8B
(Zero-shot)es, there are several factors that suggest pregnancy can potentially affect the long-
term results of bariatric surgery in both positive and negative ways. Here’s an analy-
sis based on current medical literature up to my last update in early 2023:**Positive
Impacts of Pregnancy Post Bariatric Surgery:*1. **Weight Maintenance/Regain
Mitigation:** It has been observed that having children post-bariatric surgery may
help some women maintain their weight loss because they tend not to engage in risky
behaviors associated with overeating due to parenting responsibilities. This includes
avoiding late night eating and snacking which could otherwise lead to regaining lost
weight.2. **Increased Physical Activity:** The demands of childcare often increase
physical activity levels for new mothers as they walk more while chasing active kids
or participate in sports together such as swimming or cycling. Increased daily steps
contribute positively toward metabolism improvement and calorie burn beyond what
might be achieved through other types of exercise routines.
Llama3-8B
(Context
prompted)According to the provided study, there was no significant relationship between weight
loss before bariatric surgery and excess weight loss outcomes after the surgery, re-
gardless of the surgical technique or the patient’s gender or initial BMI. Therefore,
weight loss before surgery should not be considered a prerequisite for successful
weight loss outcomes after bariatric surgery. However, it is important to note that
the study did not address the influence of pregnancy on long-term results of bariatric
surgery. Therefore, further research is needed to determine if pregnancy affects the
weight loss outcomes after bariatric surgery.¡/s¿
bRAGgen with
Llama3-8BPregnancy after bariatric surgery may affect long-term weight loss and metabolic out-
comes. Studies suggest that women who become pregnant after Roux-en-Y gastric
bypass (RYGB) or laparoscopic adjustable gastric banding (LAGB) tend to expe-
rience weight regain, particularly in the first year postpartum (1, 2). This may be
attributed to hormonal changes, increased caloric intake, and decreased physical ac-
tivity during pregnancy. Additionally, some studies have reported an increased risk
of gestational diabetes and hypertension in women who have undergone bariatric
surgery (3, 4). However, overall, the impact of pregnancy on long-term weight loss
and metabolic outcomes after bariatric surgery is still debated and requires further
research.
Table 5: Example 1
25

Continually Self-Improving Language Models for Bariatric Surgery QA
Question How long does the actual surgery take?
MedGraphRAG
(Offline)The length of the surgery for bariatric procedures, such as gastric bypass, gastric
banding, and sleeve gastrectomy, can vary depending on several factors, including
the individual patient’s anatomy and the experience of the surgeon. The study
mentioned does not provide information on the operation duration. However, it is
mentioned that weight loss before surgery may decrease the operation duration.
Llama3-8B
(Zero-shot)”The actual surgical procedure typically takes between system
In-article texte ad: An official website of the United States governmentHere’s how
you knowThe.gov means it’s official.Federal government websites often end in.gov
or.mil. Before sharing sensitive information, make sure you’re on a federal govern-
ment site.The site is secure.Thehttps://ensures that you are connecting to the official
website and that any information you provide is encrypted and transmitted securely.
An official website of the United States government
Question: How long does the actual surgery take? Answer: The actual surgical
procedure typically takes between system
You are a medical expert on Bariatric Surgey. Answer using only the provided
context and keep the answer short.”
Llama3-8B
(Context
prompted)”The actual bariatric surgery procedure typically takes around assistant
The actual surgical time for sleeve gastrectomy, Roux-en-Y gastric bypass, and ad-
justable gastric banding procedures usually takes about.system
You are a medical expert on Bariatric Surgery. Answer using only the provided
context and keep the answer short.
The actual bariatric surgery procedure typically takes around assistant
The actual surgical time for sleeve gastrectomy, Roux-en-Y gastric bypass, and ad-
justable gastric banding procedures usually takes about assistant
Typically between one to three hours, depending on the type of procedure per-
formed.”
bRAGgen with
Llama3-8BThe actual surgery time for a laparoscopic Roux-en-Y gastric bypass (RYGB) or
laparoscopic sleeve gastrectomy (LSG) typically ranges from 1 to 3 hours, depending
on the complexity of the case and the individual patient’s anatomy. For example,
a study published in the Journal of the American College of Surgeons found that
the median operative time for RYGB was 2.5 hours, while a study in the Journal of
Surgical Research reported a median operative time of 2.2 hours for LSG.
Table 6: Example 2
26

Continually Self-Improving Language Models for Bariatric Surgery QA
Question When can I shower or bathe after surgery?
MedGraphRAG
(Offline)Typically, patients are allowed to shower or bathe 24-48 hours after laparoscopic
adjustable sleeve gastrectomy (lap sleeve gastrectomy) or 3-5 days after open Sleeve
Gastectomy surgery, once the surgical site dressing is removed and the incision is
healing well. However, this may vary depending on your specific situation and sur-
geon’s instructions. It’s essential to follow your healthcare provider’s guidance on
bathing and wound care to ensure proper healing and minimize the risk of compli-
cations.
Llama3-8B
(Zero-shot)After bariatric surgery, it’s recommended to avoid bathing or showering for at least 48
hours to prevent infection and allow the surgical incision to heal properly. Always
follow your surgeon’s specific instructions regarding bathing and showering after
surgery.
Llama3-8B
(Context
prompted)After bariatric surgery, you typically won’t be able to shower or bathe until your
surgical wounds have fully healed and your staples or sutures have been removed,
usually within 7-10 days. Your surgeon will give you specific instructions on when it’s
safe to resume bathing. It’s essential to follow these guidelines to prevent infection
and promote proper healing.
bRAGgen with
Llama3-8BAccording to the American Society for Metabolic and Bariatric Surgery (ASMBS)
guidelines, patients are typically allowed to shower or bathe 24 to 48 hours after
laparoscopic Roux-en-Y gastric bypass (LRYGB) or laparoscopic adjustable gastric
banding (LAGB) surgery. For open Roux-en-Y gastric bypass (ORYGB) surgery,
showering or bathing may be delayed for 48 to 72 hours. It is essential to follow the
specific instructions provided by your surgeon or healthcare team, as they may vary
depending on the type of surgery and individual patient factors.
Table 7: Example 3
27

Continually Self-Improving Language Models for Bariatric Surgery QA
Question What if friends invite me to events where I can’t eat what’s being served?
MedGraphRAG
(Offline)If friends invite you to events where you can’t eat what’s being served, consider
politely declining or suggesting an alternative, like grabbing food together after the
event or meeting at a restaurant with options suitable for your dietary needs following
bariatric surgery. It’s essential to prioritize your health and nutrition after surgery,
so be honest with your friends and yourself about your limitations.
Llama3-8B
(Zero-shot)If your friends invite you to events where the food isn’t suitable for your dietary needs
after bariatric surgery, consider alternative options like suggesting an alternative
celebration method, declining the invitation, or bringing your own food. It’s essential
to prioritize your health and stick to a diet that aligns with your specific nutritional
requirements. Communicate your needs clearly and respectfully with your friends,
and suggest alternative arrangements that work for everyone involved.
Llama3-8B
(Context
prompted)If your friends invite you to events where you can’t eat what’s being served due to
dietary restrictions after bariatric surgery, there are several options you can consider:
1. Bring your own food: You can prepare a meal or snacks that fit your dietary needs
and bring them with you to the event. This way, you won’t have to worry about
not being able to eat what’s being served. 2. Communicate with your friends: Let
your friends know about your dietary restrictions and ask if they can accommodate
your needs. They may be able to provide you with food options that work for you.
3. Offer to bring a dish: If you’re comfortable doing so, offer to bring a dish that
fits your dietary needs and share it with the group. This can be a great way to
contribute to the event while also ensuring that you have food that you can eat. 4.
Focus on socializing: Remember that the event is not just about the food. Focus
on socializing with your friends and enjoying their company. You can also engage
in activities that don’t involve eating, such as playing games or going for a walk.
5. Practice self-compassion: It’s important to remember that everyone has dietary
restrictions or food preferences, and it’s okay if you can’t eat everything that’s being
served. Practice self-compassion and don’t feel
bRAGgen with
Llama3-8BIt’s not uncommon for individuals who have undergone bariatric surgery to encounter
social situations where the food options may not be suitable for their dietary needs.
In such cases, it’s essential to prioritize your health and well-being by being open
and honest with your friends about your dietary restrictions.
Table 8: Example 4
28

Continually Self-Improving Language Models for Bariatric Surgery QA
Question How does bariatric surgery affect patients with diabetes differently?
MedGraphRAG
(Offline)Bariatric surgery can effectively improve insulin sensitivity and blood sugar control
for individuals with type 2 diabetes. Studies have shown that significant improve-
ments in glycemic control occur within a few months after surgery, often eliminating
the need for medication or significantly reducing medication requirements. Addi-
tionally, certain types of bariatric surgery, such as gastric sleeve or gastric bypass,
may also directly impact glucose regulation by altering the way food is digested and
absorbed in the body. Consult your healthcare provider to discuss if bariatric surgery
is suitable for your specific situation.
Llama3-8B
(Zero-shot)”Bariatric surgery can significantly improve insulin sensitivity and blood sugar con-
trol in individuals with type 2 diabetes. Studies have shown that nearly system
An official website of the United States governmentHere’s how you knowThe.gov
means it’s official.Federal government websites often end in.gov or.mil. Before shar-
ing sensitive information, make sure you’re on a federal government site.The site is
secure.Thehttps://ensures that you are connecting to the official website and that
any information you provide is encrypted and transmitted securely. An official web-
site of the United States government
Question: How does bariatric surgery affect patients with diabetes differently? An-
swer: Bariatric surgery can significantly improve insulin sensitivity and blood sugar
control in individuals with type 2 diabetes. Studies have shown that nearly”
Llama3-8B
(Context
prompted)Bariatric surgery can lead to significant improvements in glycemic control for patients
with diabetes. The exact mechanism is not fully understood but is believed to be
due to a combination of weight loss, improved insulin sensitivity, and changes in
gut hormones. Studies have shown that bariatric surgery can lead to remission of
diabetes in a large percentage of patients, and even those who do not achieve complete
remission often experience significant improvements in blood sugar control. The
specific effects of different types of bariatric surgery, such as gastric bypass, sleeve
gastrectomy, and adjustable gastric banding, may vary in their impact on diabetes.
It is important for patients with diabetes considering bariatric surgery to discuss
their individual situation with their healthcare provider.¡/s¿
bRAGgen with
Llama3-8BBariatric surgery has been shown to have a profound impact on patients with di-
abetes, particularly those with type 2 diabetes. Studies have consistently demon-
strated that bariatric surgery can lead to significant improvements in glycemic con-
trol, with up to 80% of patients achieving remission from diabetes within the first
year following surgery (1). This is likely due to the significant weight loss achieved
through bariatric surgery, as well as the alterations in gut hormones and gut micro-
biome that occur after surgery (2). Additionally, bariatric surgery has been shown to
reduce the risk of microvascular complications, such as nephropathy and retinopathy,
in patients with diabetes (3). However, it is essential to note that bariatric surgery
is not a cure for diabetes, and patients should continue to work with their healthcare
providers to manage their condition.
Table 9: Example 5
29

Continually Self-Improving Language Models for Bariatric Surgery QA
Question How much protein do I need daily?
MedGraphRAG
(Offline)”After bariatric surgery, it’s recommended to consume assistant
Following Rouassistant
Following bariatric surgery, it’s recommended to consume at least assistant
1 gram of protein per kilogram of body weight (e.g., a 150-pound person should aim
for assistant
70-80 grams of protein per day) during the initial phases of recovery, increasing to
assistant
60-80 grams per day long-term to aid in healing, satiety, and overall health. It’s
essential to work with your healthcare provider to determine the best protein goals
for your individual situation.”
Llama3-8B
(Zero-shot)”After bariatric surgery, it’s essential to consume adequate protein to aid in healing,
muscle recovery, and overall health. The recommended dietary allowance (RDA) for
protein varies based on factors such as age, sex, and weight. Generally, the Academy
of Medicine recommends:
* assistant
Following a sleeve gastrectomy or gastric bypass surgery, the typical recommended
daily protein intake ranges from assistant
75-assistant
75-100 grams per day during the first 12-14 weeks after surgery, gradually increasing
to assistant
100-125 grams per day thereafter. It’s crucial to work with your healthcare provider
or a registered dietitian to determine your specific protein needs and ensure proper
nutrition following bariatric surgery.”
Llama3-8B
(Context
prompted)The protein requirement can vary depending on individual factors such as age, sex,
weight, activity level, and health status. On average, a sedentary adult needs about
0.8 grams of protein per kilogram of body weight per day. For example, a person who
weighs 70 kilograms would need approximately 56 grams of protein daily. However,
individuals undergoing bariatric surgery may require more protein due to increased
nutrient needs and altered nutrient absorption. Your healthcare provider or a reg-
istered dietitian can help determine the appropriate protein intake for your specific
situation. Regarding the text you provided, it discusses the relationship between
weight loss before bariatric surgery and weight loss outcomes after surgery. The
study suggests that weight loss before surgery does not seem to influence the excess
weight loss after surgery, regardless of the surgical technique or initial BMI. The
text also mentions a separate study on heterogeneity of weight loss after different
bariatric procedures.
bRAGgen with
Llama3-8BAccording to the American Society for Metabolic and Bariatric Surgery (ASMBS)
and the International Federation for the Surgery of Obesity and Metabolic Disor-
ders (IFSO), a minimum of 60-80 grams of protein per day is recommended for
patients undergoing bariatric surgery, including Roux-en-Y gastric bypass, sleeve
gastrectomy, and adjustable gastric banding. This amount helps to promote optimal
wound healing, muscle preservation, and overall nutritional recovery.
Table 10: Example 6
30
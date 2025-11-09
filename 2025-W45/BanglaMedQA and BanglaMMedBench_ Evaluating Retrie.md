# BanglaMedQA and BanglaMMedBench: Evaluating Retrieval-Augmented Generation Strategies for Bangla Biomedical Question Answering

**Authors**: Sadia Sultana, Saiyma Sittul Muna, Mosammat Zannatul Samarukh, Ajwad Abrar, Tareque Mohmud Chowdhury

**Published**: 2025-11-06 17:15:33

**PDF URL**: [http://arxiv.org/pdf/2511.04560v1](http://arxiv.org/pdf/2511.04560v1)

## Abstract
Developing accurate biomedical Question Answering (QA) systems in
low-resource languages remains a major challenge, limiting equitable access to
reliable medical knowledge. This paper introduces BanglaMedQA and
BanglaMMedBench, the first large-scale Bangla biomedical Multiple Choice
Question (MCQ) datasets designed to evaluate reasoning and retrieval in medical
artificial intelligence (AI). The study applies and benchmarks several
Retrieval-Augmented Generation (RAG) strategies, including Traditional,
Zero-Shot Fallback, Agentic, Iterative Feedback, and Aggregate RAG, combining
textbook-based and web retrieval with generative reasoning to improve factual
accuracy. A key novelty lies in integrating a Bangla medical textbook corpus
through Optical Character Recognition (OCR) and implementing an Agentic RAG
pipeline that dynamically selects between retrieval and reasoning strategies.
Experimental results show that the Agentic RAG achieved the highest accuracy
89.54% with openai/gpt-oss-120b, outperforming other configurations and
demonstrating superior rationale quality. These findings highlight the
potential of RAG-based methods to enhance the reliability and accessibility of
Bangla medical QA, establishing a foundation for future research in
multilingual medical artificial intelligence.

## Full Text


<!-- PDF content starts -->

BanglaMedQA and BanglaMMedBench: Evaluating
Retrieval-Augmented Generation Strategies for Bangla
Biomedical Question Answering
Sadia Sultanaa,1, Saiyma Sittul Munaa,1, Mosammat Zannatul Samarukha,1,
Ajwad Abrara,∗, T areque Mohmud Chowdhurya
aDepartment of Computer Science and Engineering, Islamic University of
T echnology, Gazipur, Dhaka, 1704, Bangladesh
Abstract
Developing accurate biomedical Question Answering (QA) systems in low-
resource languages remains a major challenge, limiting equitable access to re-
liable medical knowledge. This paper introduces BanglaMedQA and BanglaMMed-
Bench, the first large-scale Bangla biomedical Multiple Choice Question
(MCQ) datasets designed to evaluate reasoning and retrieval in medical arti-
ficial intelligence (AI). The study applies and benchmarks several Retrieval-
Augmented Generation (RAG) strategies, including T raditional, Zero-Shot
F allback, Agentic, Iterative F eedback, and Aggregate RAG, combining textbook-
based and web retrieval with generative reasoning to improve factual accu-
racy . A key novelty lies in integrating a Bangla medical textbook corpus
through Optical Character Recognition (OCR) and implementing an Agen-
∗Corresponding author.
Email addresses: sadiasultana20@iut-dhaka.edu (Sadia Sultana),
saiymasittul@iut-dhaka.edu (Saiyma Sittul Muna), zannatul2@iut-dhaka.edu
(Mosammat Zannatul Samarukh), ajwadabrar@iut-dhaka.edu (Ajwad Abrar),
tareque@iut-dhaka.edu (T areque Mohmud Chowdhury)
1These authors contributed equally to this work.arXiv:2511.04560v1  [cs.CL]  6 Nov 2025

tic RAG pipeline that dynamically selects between retrieval and reasoning
strategies. Experimental results show that the Agentic RAG achieved the
highest accuracy (89.54%) with openai/gpt-oss-120b, outperforming other
configurations and demonstrating superior rationale quality . These findings
highlight the potential of RAG-based methods to enhance the reliability and
accessibility of Bangla medical QA, establishing a foundation for future re-
search in multilingual medical artificial intelligence.
Keywords: Retrieval-Augmented Generation (RAG), Bangla Biomedical
Question Answering, Natural Language Processing, Medical Multiple
Choice Questions (MCQs), Dataset Benchmarking and Evaluation
1. Introduction
The advancement of Large Language Models (LLMs) such as GPT and
LLaMA has transformed Natural Language Processing (NLP) tasks, includ-
ing medical Question Answering (QA). These models excel at understand-
ing and generating human-like text, allowing users to ask complex medical
questions in everyday language. However, LLMs often produce inaccurate
or hallucinated answers, particularly in medical contexts where accuracy is
vital ( Qin et al. ,2025 ;Qiu et al. ,2024a ). This limitation arises because
these models rely on fixed training data and lack direct access to external,
up-to-date knowledge sources.
Retrieval-Augmented Generation (RAG) addresses this limitation by com-
bining information retrieval with generative reasoning to improve answer
accuracy ( Lewis et al. ,2020 ;Zhang et al. ,2024 ). By fetching relevant doc-
uments before generating answers, RAG provides responses that are better
2

grounded in facts, particularly in medical and scientific domains. While RAG
has demonstrated strong performance in English biomedical QA systems, its
effectiveness in low-resource languages like Bangla, spoken by over 230 mil-
lion people, remains largely unexplored ( Khan et al. ,2023 ;Shafayat et al. ,
2024 ;W ork ,2023 ). The lack of high-quality Bangla biomedical datasets and
evaluation tools limits the development of reliable QA systems, creating bar-
riers in education and healthcare support ( Qin et al. ,2025 ).
Building reliable Bangla medical QA systems involves several challenges.
Large-scale Bangla biomedical QA datasets with detailed rationales do not
exist, which makes model training diﬀicult ( Shafayat et al. ,2024 ;W ork ,
2023 ). T ranslating medical content often distorts meaning and reduces clar-
ity ( Pal et al. ,2022 ). Retrieval is also challenging due to sparse Bangla
resources ( Lewis et al. ,2020 ;Zhang et al. ,2024 ). Finally , evaluating both
answer correctness and rationale quality requires specialized metrics and hu-
man assessment ( T satsaronis et al. ,2015 ). Addressing these challenges is
essential to expand medical Artificial Intelligence (AI) to underrepresented
languages.
T o address these issues, this study develops Bangla biomedical QA datasets
and systematically evaluates RAG strategies ( Figure 1 ). The contributions
of this work include:
1. W e developed BanglaMedQA and BanglaMMedBench, two publicly
available Bangla Multiple Choice Question (MCQ) datasets for the
3

medical domain, totaling 2,000 questions.2BanglaMedQA contains
1,000 questions from authentic medical admission tests in Bangladesh,
each accompanied by reasoning for the correct answer option. BanglaMMed-
Bench includes another 1,000 situational and complex questions, trans-
lated and refined from the English MMedBench dataset for Bangla
evaluation.
2. W e implemented and benchmarked multiple RAG variants, including
T raditional RAG, Zero-Shot F allback, Agentic RAG, Iterative F eed-
back RAG, and Aggregate k-values RAG. W e analyzed model perfor-
mance, rationale quality , and the effects of model size and retrieval
strategy in a low-resource Bangla setting.
3. F or retrieval, we curated an external knowledge base for BanglaMedQA
using Optical Character Recognition (OCR) on a standard higher secondary-
level Bangla medical textbook used in Bangladesh.
These contributions establish a benchmark for Bangla biomedical QA,
evaluate the practical effectiveness of retrieval-augmented strategies, and ad-
vance the development of trustworthy medical Artificial Intelligence (AI) for
Bangla-speaking communities. In the following sections, we present the re-
lated works, describe our proposed methodology , discuss the experimental
results, and conclude with key findings and future directions.
2https://huggingface.co/datasets/ajwad-abrar/BanglaMedQA
4

Figure 1: The responses of llama-3.3-70b-versatile to a question from BanglaMedQA fol-
lowing different strategies.
2. Related W orks
2.1. Medical Question Answering (QA)
Medical Question Answering (QA) systems enable clinicians and patients
to access reliable information as medical data grows rapidly . Large Lan-
guage Models (LLMs) like GPT and LLaMA demonstrate strong performance
in Natural Language Processing (NLP) tasks ( Qin et al. ,2025 ;Qiu et al. ,
2024a ), but biomedical QA remains challenging due to complex terminology
and limited domain-specific training. Biomedical QA tasks typically follow
two formats:
1. Multiple-Choice Question Answering (MCQ), exemplified by datasets
5

such as MedMCQA Pal et al. (2022 ) and PubMedQA Jin et al. (2019 ),
which test structured exam-style reasoning.
2. Generative Answering, as seen in benchmarks like BioASQ T satsaronis
et al. (2015 ), which evaluate deeper reasoning but are more prone to
inaccurate or made-up responses.
These formats highlight the need for robust systems capable of handling
diverse medical queries with high accuracy .
2.2. Low-Resource Languages in Medical NLP
While English benefits from extensive biomedical datasets, low-resource
languages like Bangla face significant resource scarcity , limiting the develop-
ment of effective QA systems ( Abrar et al. ,2024 ). Efforts such as BEnQA
(Shafayat et al. ,2024 ), a bilingual science QA benchmark, and Bangla health
Named Entity Recognition (NER) ( Khan et al. ,2023 ) provide some founda-
tion but lack comprehensive coverage for biomedical QA reasoning. This
scarcity underscores the challenge of adapting advanced QA techniques to
languages with limited data, particularly in specialized domains like medicine,
where accurate and contextually relevant responses are critical.
2.3. Retrieval-Augmented Generation (RAG)
T o improve factual accuracy in QA, Retrieval-Augmented Generation
(RAG) integrates information retrieval with text generation to ground re-
sponses in external knowledge ( Lewis et al. ,2020 ). Studies show that retrieval-
based models outperform purely parametric ones, enhancing factuality in
biomedical QA ( Izacard and Grave ,2021 ;Zhang et al. ,2024 ). However,
6

standard RAG pipelines face issues such as retrieval errors, irrelevant pas-
sages, and unstable reasoning. T o address these, several RAG variants have
been proposed:
1. RAG with F allback : Uses Zero-Shot or parametric generation when
retrieval fails ( Lewis et al. ,2020 ).
2. Agentic RAG : Allows models to decide when to retrieve or reflect, as
in ReAct Y ao et al. (2022 ) and Reflexion (Shinn et al. ,2023 ).
3. Iterative F eedback RAG : Refines retrieval through multiple feed-
back loops ( Shinn et al. ,2023 ).
4. Aggregate k-V alues RAG : Combines evidence from multiple re-
trieved passages for robustness ( Zhang et al. ,2024 ).
These advancements highlight the potential of RAG to improve QA systems,
particularly in challenging domains.
2.4. Research Gap and Motivation
Despite notable progress in Natural Language Processing (NLP) and
Large Language Models (LLMs), several gaps remain in the domain of biomed-
ical Question Answering (QA), particularly in multilingual and resource-
constrained contexts.
1. Limited Biomedical QA in Bangla and Low-Resource Lan-
guages: Most existing biomedical QA systems and datasets are devel-
oped for English. Low-resource languages like Bangla remain underrep-
resented, limiting access for a large portion of practitioners, students,
and patients in non-English speaking regions.
7

2. Scarcity of Domain-Specific Datasets: Widely used datasets (e.g.,
MedQA ( Y ang et al. ,2024 ), PubMedQA, MedMCQA) are predom-
inantly in English and often based on W estern curricula. There is
a lack of curated, high-quality biomedical MCQ datasets tailored to
South Asian or Bangla medical textbooks, creating a significant bar-
rier for localized applications.
3. Challenges in Applying General LLMs to Biomedical MCQ
T asks: Open-domain LLMs (e.g., GPT, LLaMA, Mistral) show strong
general reasoning but often lack specialized biomedical knowledge ( Ja-
han et al. ,2023 ). Without domain adaptation, these models struggle
with terminologies, abbreviations, and reasoning required for medical
MCQs.
4. Limitations of Current Retrieval-Augmented Generation (RAG)
Approaches: While RAG has improved factual accuracy by ground-
ing LLMs in external knowledge, most biomedical RAG studies fo-
cus on research articles and clinical notes, not structured MCQs. Ex-
isting RAG pipelines do not adequately handle multilingual retrieval
(Bangla–English mix) or exam-style MCQ reasoning with rationales.
5. Need for Explainability in Biomedical Education: Current biomed-
ical QA models mostly generate short answers without providing rea-
soning steps or rationales ( Smith and Johnson ,2005 ). F or educational
and clinical training purposes, explanations are essential, yet underex-
plored in multilingual biomedical MCQ systems.
This research therefore addresses the above gaps by curating a Bangla
biomedical MCQ dataset from textbooks and exam sources, implementing
8

RAG pipelines that handle multilingual retrieval (both Bangla and English)
for answer and rationale generation, enhancing both accuracy and explain-
ability .
3. Proposed Methodology
This section presents the methodology used to address identified gaps in
biomedical question answering. W e introduce two Bangla medical Question
Answering (QA) datasets - BanglaMedQA and BanglaMMedBench, along
with the Higher Secondary level biology textbook corpus, written in Bangla,
used as the retrieval source. The following subsection describes the imple-
mentation of multiple Retrieval-Augmented Generation (RAG) strategies,
including T raditional RAG, Zero-shot fallback RAG, Agentic RAG, Iterative
feedback RAG, and Aggregate retrieval RAG variants. Finally , the section
concludes with the evaluation of each of the datasets to analyze accuracy ,
reasoning quality , and the role of web-based retrieval.
3.1. Datasets
T o support research in medical question-answering for Bangla, we intro-
duce two complementary datasets. BanglaMedQA is the first curated Bangla
medical MCQ dataset covering foundational topics, carefully filtered for qual-
ity and consistency . Bangla MMedBench extends an English biomedical
benchmark into Bangla through careful translation and pre-processing, pro-
viding reasoning-intensive questions with rationales. T ogether, these datasets
establish both surface-level and advanced resources, serving as foundational
benchmarks for evaluating Retrieval-Augmented Generation (RAG) and other
Question Answering (QA) systems in a low-resource language.
9

3.1.1. BanglaMedQA- Bangla Medical Corpus
The BanglaMedQA dataset is a novel contribution, representing the first
curated medical question–answer dataset in Bangla. It was compiled from
authentic sources, specifically the admission tests of Bangladesh’s Medical
(MBBS), Dental (BDS), and Armed F orces Medical College (AFMC) pro-
grams, spanning exam years from 1990 to 2024, covering 34 years. The
dataset contains 1,000 Multiple Choice Questions (MCQs), each with four
answer options, a correct answer, and an explanation for the correct answer.
The questions primarily focus on surface-level factual knowledge typically
required in medical admission tests.
Quality Assurance and Data Filtering: T o ensure that the dataset
is accurate, consistent, and reliable, several rigorous curation policies were
followed. Initially , a total of 1,200 questions were collected from authentic
medical admission test sources before applying the following filtering steps:
1. Exclusion of ambiguous questions: Questions containing multiple
correct answers were removed to eliminate interpretive ambiguity . Dur-
ing verification, five such items were identified and excluded to preserve
the dataset’s clarity and validity .
2. Standardization of option formats: T o ensure structural consis-
tency , questions with irregular option labels (e.g., Bangla letters such
asক,খ,গ,ঘ or numeric formats like 1, 2, 3, 4) were discarded. Eleven
such items were removed to maintain a uniform multiple-choice struc-
ture using standard English labels (A, B, C, D).
3. Deduplication of repeated items: Since the data spanned multiple
admission test years, duplicate questions were identified and removed
10

to retain only unique entries. Out of the initial 1,200 collected items,
184 duplicates were eliminated, ensuring a diverse and non-redundant
question set.
By addressing these considerations, BanglaMedQA establishes a high-
quality , domain-specific benchmark for evaluating question-answering sys-
tems in Bangla. The dataset not only facilitates the exploration of RAG in
a low-resource language but also provides a foundational resource for future
research in medical Artificial Intelligence (AI) applications in Bangladesh
and the broader Bangla-speaking community .
3.1.2. BanglaMMedBench: T ranslated Biomedical Corpus
The BanglaMMedBench dataset was developed to address the absence
of Bangla medical question sets involving clinical or situational reasoning.
In existing Bangla medical examinations, such as MBBS or BDS admission
tests, questions primarily focus on factual recall rather than scenario-based
understanding. T o fill this gap, we extended the English medical bench-
mark introduced by Qiu et al. ( Qiu et al. ,2024b ) into Bangla. The original
dataset comprises situation-based medical questions inspired by the United
States Medical Licensing Examination (USMLE), with accompanying ratio-
nales generated using ChatGPT.
T o translate this benchmark into Bangla, we employed the Gemini-1.5-
Flash model ( Georgiev et al. ,2024 ) through a batch translation process.
Before finalizing this choice, we conducted a comparative evaluation of several
LLM-based translation methods, including Google T ranslate, Mistral Saba,
LLaMA 3 70B, Gemini, and ChatGPT. A set of 50 sample questions was
11

manually reviewed by a medical expert to assess translation quality , focusing
on domain-specific terminology accuracy and grammatical correctness.
The evaluation revealed that Mistral Saba struggled with medical termi-
nology , while LLaMA 3 70B frequently misinterpreted multiple-choice op-
tions. Google T ranslate, despite its eﬀiciency , introduced frequent spelling
and syntactic inconsistencies in Bangla. In contrast, Gemini and ChatGPT
preserved medical terminology and maintained grammatical fidelity . Based
on these results, computational eﬀiciency , and expert feedback, Gemini-1.5-
Flash was chosen as the final translation model.
This approach enabled the complete translation of the dataset into Bangla
while preserving the original structure of questions, options, correct answers,
and rationales.
After translation, several preprocessing steps were applied to ensure qual-
ity and consistency:
• Removal of incorrectly translated or incomplete items.
• Correction of formatting inconsistencies in options and rationales.
• V erification of alignment between translated answers and rationales
with the original source.
The final dataset comprises 1,000 high-quality Bangla biomedical Multi-
ple Choice Questions (MCQs) with rationales, offering a more advanced and
situational resource compared to the surface-level factual nature of BanglaMedQA.
By introducing this dataset, we establish the first advanced Bangla biomedi-
cal MCQ corpus for evaluating medical reasoning in a low-resource language.
12

It serves as a critical benchmark for assessing Retrieval Augmented Genera-
tion (RAG) and other question-answering strategies in Bangla.
3.1.3. Data Preprocessing
Before running experiments, the raw BanglaMedQA dataset and the
Bangla MMedBench dataset required several cleaning steps to ensure consis-
tency . First, Unicode normalization was applied to resolve issues with irregu-
lar ASCII and full-width characters. Soft and invisible whitespace characters
were also normalized to prevent parsing errors. In some cases, the multiple-
choice options were embedded directly inside the question text, or stored
together under a single key , rather than as separate fields. T o address this, a
regular expression–based parser was implemented to detect option markers
(e.g., “A. ”, “B)”, “C:”) and split them into distinct entries for A, B, C, and
D. This ensured that every record followed a uniform structure, with a clean
question field and exactly four option fields. This cleaned dataset was then
used across all subsequent experiments.
3.2. Retrieval Augmented Generation (RAG)
Retrieval-Augmented Generation (RAG) is a framework that combines in-
formation retrieval with Large Language Models (LLMs) ( W u et al. ,2024 ).
Instead of relying solely on the knowledge stored in a language model’s pa-
rameters, RAG first retrieves relevant documents or context from an exter-
nal knowledge source (such as a corpus, database, or textbook) ( Salemi and
Zamani ,2024 ) and then conditions a generative model on this retrieved in-
formation to produce accurate and context-aware responses. This approach
enhances factual accuracy , allows handling of domain-specific queries, and
13

supports low-resource languages by grounding generation in explicit refer-
ence material.
3.2.1. RAG Retrieval Source: Bangla Biology T extbook
T o construct a reliable retrieval source for our Retrieval-Augmented Gen-
eration (RAG) pipelines, we digitized the Higher Secondary level (12th class)
Biology textbook written in Bangla, ensuring that the entire content was
available in machine-readable form. T o ensure the quality and completeness
of the digitized text, we followed a two-step process:
Step 1: Finding the best OCR system: Initially , we experimented
with T esseract, an open-source Optical Character Recognition (OCR) en-
gine ( Smith ,2007 ;Koistinen et al. ,2017 ). While T esseract produced ac-
ceptable results for simple text, it struggled with complex Bangla juktakkhor
(conjunct characters). F or example, after processing certain pages, T esser-
act often skipped entire paragraphs where such characters appeared, leading
to substantial gaps in the extracted text. T o address this, we switched to
Google Lens OCR ( Maurya et al. ,2023 ;Hegghammer ,2022 ), which yielded
significantly cleaner outputs and demonstrated superior handling of complex
Bangla character combinations.
Step 2: Manual inspection and correction: F ollowing OCR, the
extracted text was carefully verified to correct alignment issues, formatting
errors, and occasional missing content. This post-processing step ensured
that the corpus retained both structural integrity and linguistic accuracy .
The final OCR-processed version of the Higher Secondary level Biology text-
book served as the primary knowledge source for RAG-based evaluation,
enabling reliable retrieval of domain-specific content directly aligned with
14

Figure 2: Methodological workflow of the Bangla RAG framework.
Bangla medical Question Answering (QA) tasks.
3.2.2. Implementation of RAG V ariants
The methodology integrates local retrieval from the textbook with fall-
back strategies to external web search or zero-shot reasoning. The pipeline
shown in Figure 2 supports multiple strategies, including one-shot reason-
ing, iterative refinement, and aggregation across different retrieval depths to
improve answer accuracy and rationale quality .
1. Local RAG: In traditional Retrieval-Augmented Generation (RAG),
a question qwith multiple-choice options O={OA, OB, OC, OD}is
input to a Large Language Model (LLM). The model retrieves the top-
kpassages from a processed Bangla Biology textbook knowledge base
15

D via similarity search:
C=Retrieve (q, D, k )
The retrieved context C, question q, and options O form a prompt for
the LLM. If C is suﬀicient (based on a length threshold), the LLM
generates a predicted option ˆO∈O and rationale R:
(ˆO, R ) = arg max
aP(a|q, O, C )
IfC lacks suﬀicient information, the model outputs:
(ˆO, R ) = ( N/A,উত্তরপাওয়াযায়িন )
where"উত্তরপাওয়াযায়িন" means “The answer was not found” in Bangla.
Experiments showed traditional RAG often returns null responses for
questions lacking relevant context, especially for chapter-specific queries.
In contrast, zero-shot prompting always produces a prediction ˆO and
rationale R, relying on pretrained knowledge.
2. Local RAG with Zero-Shot F allback: T o address traditional RAG’s
null responses when retrieved context is insuﬀicient, we developed a hy-
brid approach combining retrieval with zero-shot fallback, ensuring all
questions receive answers.
The pipeline starts with retrieving top- kpassages from knowledge base
D for query q:
C=Retrieve (q, D, k )
The LLM uses q, options O, and context C to predict answer ˆO and
rationale R:
(ˆO, R ) = arg max
aP(a|q, O, C )
16

IfCis insuﬀicient, instead of outputting "উত্তরপাওয়াযায়িন" (“The answer
was not found”), the model falls back to zero-shot, using only qandO:
(ˆO, R ) = arg max
aP(a|q, O)
This ensures every question has a prediction, balancing precision (with
relevant context) and recall (via zero-shot).
3. W eb Search with Zero-Shot F allback: W e developed an W eb RAG
pipeline for Bangla MCQs, using web retrieval and zero-shot fallback,
preserving Comma-Separated V alues (CSV) structure with predictions,
rationales, and metrics. The steps are shown below:
i) W eb Retrieval: The question and its options are used to perform
a web search through an API, retrieving up to eight relevant links. The
top three results are parsed to extract textual content from key HTML
elements such as articles and paragraphs. The extracted information is
then summarized into concise Bangla bullet points, forming the web-
based contextual input. If the aggregated context exceeds a predefined
length threshold, the model generates a structured JSON response.
{"O": "A|B|C|D",
"R": "1-3 line Bangla explanation"}
ii) Zero-Shot F allback: If the web context is insuﬀicient, the model
resorts to zero-shot reasoning using its pretrained knowledge to gener-
ate an answer. HTTP requests, HTML parsing, retry mechanisms, and
API delay handling are incorporated to ensure reliability .
This pipeline leverages dynamic web knowledge for Bangla MCQs, with
zero-shot fallback ensuring full coverage and compatibility with evalu-
ation.
17

Figure 3: Agentic RAG pipeline illustration.
4. Agentic RAG: The Agentic RAG pipeline, shown in Figure 3 , en-
hances performance by allowing the LLM to dynamically select the
most suitable answering strategy for Bangla MCQs. It adapts between
local retrieval, web retrieval, and zero-shot fallback as needed.
i) Local Retrieval: The query qretrieves top- kpassages from text-
book D:
C local=Retrieve (q, D, k )
A router checks if C local is suﬀicient:
18

￿￿￿￿ ￿￿￿￿￿
তুিমএকিটরাউটারমেডল।িনেচরেটক্সেটিকপ্রেশ্নরউত্তরেদবারজনঁযেথষ্ট
তথঁআেছ?শুধুএকিটশেব্দউত্তরদাও: Y esঅথবা No￿
(T ranslation: “Y ou are a router model. Does the text below
contain enough information to answer the question? Answer in
one word only: Y es or No. ”)
If Y es and |C local|> τ 1, Local RAG generates:
(ˆO, R ) = arg max
aP(a|q, O, C local)
ii) W eb Retrieval: If local context is insuﬀicient, the Serper API
fetches up to 8 links, with the top 3 scraped and summarized into
Bangla bullet points, forming C web . If|C web|> τ 2, W eb RAG generates:
(ˆO, R ) = arg max
aP(a|q, O, C web)
iii) Zero-Shot F allback: If both retrievals fail, the model uses zero-
shot reasoning:
(ˆO, R ) = arg max
aP(a|q, O)
The routing policy is:
π(q) =

Local RAG , if Router (C local) = Y es∧
|C local|> τ 1,
W eb RAG , if Router (C local) = No∧
|C web|> τ 2,
Zero-Shot , otherwise.
This ensures flexible knowledge selection and full question coverage.
19

Figure 4: Iterative F eedback RAG pipeline illustration.
5. Iterative F eedback RAG: Iterative F eedback RAG introduces a
feedback-driven loop that mimics introspective reasoning as shown in
Figure 4 , improving robustness and accuracy within a single query-
response cycle. The pipeline dynamically refines both queries and con-
text through four key phases.
i) Context Retrieval: The system retrieves the top- krelevant text
chunks from a F AISS vector store using similarity search:
Ck=Retrieve (q, D, k )
Using dense embeddings, the model fetches potentially noisy context,
which may require refinement in later steps.
ii) One-Shot Answering: The LLM generates an initial answer based
20

on the query qand retrieved context Ck:
(ˆO0, R0) = arg max P(a|q, Ck)
F or Multiple-Choice Questions, the model outputs a single character
(e.g., “A”). If the answer is correct, the process terminates; otherwise,
it proceeds to the refinement phase.
iii) Iterative Refinement: If the initial answer is incorrect, the LLM
extracts key terms from qandCkto reformulate the query:
q′=f(q, K)
A new retrieval and prediction are then performed:
(ˆOi, Ri) = arg max P(a|q′, Ck)
Up to two refinement cycles are allowed to enhance context relevance
and correct prior reasoning errors, ensuring a balance between eﬀiciency
and accuracy .
iv) Final Answer and Confidence: The system assigns confidence
levels based on the outcome of the process:
• High confidence: Correct one-shot answer.
• Medium confidence: Corrected through refinement.
• Low confidence: Incorrect even after refinement.
This scoring mechanism enhances interpretability and trust in model
outputs.
6. Aggregate k-values RAG: T raditional RAG pipelines retrieve a
fixed top- kcontext, but performance is highly sensitive to the chosen k.
21

Figure 5: Aggregate k-values RAG pipeline illustration.
A small kmay miss key information, while a large kcan introduce noise.
This sensitivity is particularly critical in domains such as zoology , where
precise terminology and scattered knowledge are common. Aggregate
k-values RAG addresses this limitation by combining predictions from
multiple k-values as shown in Figure 5 , thereby improving stability ,
robustness, and accuracy . The pipeline operates in three stages.
i) Context Retrieval: F or a given query qand knowledge base D,
the system retrieves context sets for multiple k-values:
Ckj=Retrieve (q, D, k j), k j∈ {3,5,6}
Using dense embeddings within a F AISS vector store, different k-values
serve distinct purposes: k= 3 prioritizes precision, k= 5 balances
precision and recall, and k= 6 ensures broad coverage. T ogether,
these retrievals capture complementary pieces of information.
22

ii) LLM Prediction: The large language model generates predictions
for each context set Ckjas follows:
ˆOkj=arg max
O∈OP(O|q, Ckj)
Each prediction includes a single-character answer (e.g., “A”) and a
brief rationale, produced in a JSON-like format. A temperature of
0.7 is used to balance creativity and determinism, yielding diverse yet
coherent responses across different k-values.
iii) V ote Collection and Aggregation: Predictions {ˆOkj}are then
aggregated through a voting mechanism:
• Majority: The most frequent option is selected as the final an-
swer.
• Unanimous: Identical predictions across all k-values indicate
high confidence.
• Tie: In case of a tie, the prediction from k= 6 is chosen, lever-
aging the broadest context.
The final answer is defined as:
ˆO=Aggregate(
{ˆOkj}m
j=1)
, m = 3
The majority rationale corresponding to the winning prediction is se-
lected for consistency and explainability .
3.3. Comparative Evaluation of MMedBench and BanglaMMedBench
W e evaluated the Zero-Shot and W eb Search with Zero-Shot F allback
methods on the MMedBench and BanglaMMedBench datasets, as shown
23

Figure 6: Zero-shot and web search: MMedBench vs. translated Bangla dataset.
in Figure 6 . T raditional RAG was excluded because the scenario-based,
patient-specific questions lacked a reliable textbook corpus for retrieval. The
experiments compared the English and Bangla versions to assess the impact
of translation and the effectiveness of web search integration.
1. Zero-Shot Analysis: The zero-shot evaluation tested the model’s
reasoning on both English and Bangla MMedBench datasets with-
out fine-tuning. Each dataset contained Multiple Choice Questions
(MCQs) with options Oi={OA, OB, OC, OD}, ground-truth answers
ai, and, where available, rationales ri, all preprocessed for consistency .
A structured prompt Pi=f(qi, Oi)was created for each question qi
and its options Oi, instructing the model to select a predicted answer
ˆaiand provide a rationale ˆriwithin a 300-second timeout. The eval-
uation measured accuracy as Accuracy =1
N∑N
i=1 1(ˆai=ai), while
rationales ˆriwere qualitatively compared to ri, with results Ri=
{ˆai,ˆri, ai, ri,Accuracyi}used to compare language performance.
2. W eb Search with F allback to Zero-Shot: T o enhance the per-
formance of the zero-shot setting, we introduced a pipeline that aug-
ments question answering with external web retrieval while retaining
24

a fallback to zero-shot when necessary . This design ensures that every
question receives an attempted answer while grounding predictions in
external evidence whenever possible.
i) W eb Retrieval: Query quses Serper API for up to 8 links, top 3
scraped into C web . If|C web|> τ 2, it generates:
(ˆO, R ) = arg max
aP(a|q, O, C web)
ii) Zero-Shot F allback: If|C web| ≤τ2or retrieval fails, it falls back
to:
(ˆO, R ) = arg max
aP(a|q, O)
This ensures answers for scenario-based MCQs with limited textbook
data.
3.4. Experimental Setup
T o evaluate the proposed approaches, we employed multiple Large Lan-
guage Models (LLMs), external APIs, and supporting libraries to construct
a robust RAG-based pipeline. The setup integrates both local retrieval from
the Bangla biology textbook and fallback strategies to external web search
or zero-shot reasoning.
3.4.1. Large Language Models
W e experimented with a range of models served through the Groq API
(Groq ,2025 ), including llama-3.3-70b-versatile ( Dubey et al. ,2024 ), llama-
3.1-8b-instant ( V avekanand and Sam ,2024 ), openai/gpt-oss-20b ( Inuwa-Dutse ,
25

2025 ), and openai-/gpt-oss-120b. The Groq API was chosen for its low-
latency inference and ease of switching between different model variants
without modifying downstream code.
3.4.2. Embedding Models
F or Bangla text, we used the BengaliSBER T model, a multilingual variant
of Sentence-BER T (SBER T) specifically optimized for semantic similarity
in Bangla. SBER T modifies the original BER T architecture by introduc-
ing a siamese network structure that enables the generation of semantically
meaningful sentence embeddings instead of token-level representations. This
allows direct comparison of sentence vectors using cosine similarity , which
significantly improves eﬀiciency in semantic search and clustering.
3.4.3. V ector Database and Chunking
The entire Bangla biology textbook was normalized, cleaned, and seg-
mented into overlapping chunks of 1,000 characters with an overlap of 200.
This chunking ensured that retrieved passages were semantically complete
while avoiding information loss at segment boundaries. A F AISS index
(Douze et al. ,2025 ) was built and cached locally for reuse across runs.
3.4.4. External W eb Search
The Serper API ( Dev ,2025 ) was integrated to perform web retrieval. F or
each query , up to 8 search results were requested, of which the top 3 links
were scraped. Extracted raw text was summarized into a compact context
before being passed to the LLM.
26

3.4.5. Agentic RAG with Router
Unlike traditional RAG, which only checks context length, the agentic
pipeline introduced a router module that explicitly judged whether retrieved
textbook passages contained suﬀicient information. The router was prompted
to answer “Y es” or “No,” and its decision determined whether to proceed with
Local RAG or fallback to W eb Search. Thresholds were applied to further
control routing:
•τ1= 300 characters for textbook retrieval.
•τ2= 200 characters for web summaries.
If the retrieved or summarized context failed these thresholds, the pipeline
defaulted to zero-shot answering.
3.4.6. Iterative MCQ Evaluation
T o improve answer accuracy , questions that failed the initial one-shot
evaluation were reprocessed using an iterative method with 2 iterations. In
this approach, the model extracted key keywords or short phrases from the
retrieved context in each iteration, refining the query before generating the
final answer.
3.4.7. Aggregated RAG Evaluation
T o enhance prediction robustness, each question was processed with mul-
tiple retrieval settings (k = 3, 5, and 6) from the F AISS vector store. The
model’s answers from these different retrievals were combined using a voting
mechanism, with a tie-breaker favoring the largest retrieval set (k = 6). This
27

aggregation approach reduces errors from incomplete context retrieval and
improves both option selection and rationale generation.
3.4.8. Error Handling and Retry Strategy
T o handle instability from API calls, the system used exponential back-
off retries for Groq and Serper requests (up to 3–4 attempts). This ensured
smoother execution during large-scale batch runs, preventing failures caused
by transient network or server issues. In addition, detailed logging was imple-
mented to capture error metadata such as request latency , response status,
and endpoint behavior. This information was later used to analyze API re-
liability trends and optimize retry intervals. Combined with local caching of
successful responses, the retry strategy improved both resilience and through-
put in the overall evaluation pipeline.
3.4.9. Evaluation Metrics
T o comprehensively evaluate both answer correctness and rationale qual-
ity , we utilized a combination of quantitative metrics. Accuracy was used to
assess discrete answer prediction performance, while BER TScore, METEOR,
ROUGE, and BLEU measured the semantic and lexical similarity between
generated and reference rationales.
1. Accuracy: Proportion of correctly predicted MCQ answers as in Equa-
tion 1 , indicating how often the model selects the correct option. Higher
accuracy reflects better comprehension and retrieval performance.
Accuracy =Number of Correct Predictions
T otal Number of Predictions(1)
28

2. BER TScore (F1): Evaluates the semantic similarity between gen-
erated and reference rationales using contextual embeddings from a
pretrained BER T model. Instead of relying on surface-level n-gram
overlap, it compares embeddings token by token, capturing deeper se-
mantic alignment and meaning preservation between two texts ( Zhang
et al. ,2019 ). The BER TScore (F1) formula is the standard harmonic
mean for F1 scores, calculated as:
F1= 2×P×R
P+R(2)
where P is the BER T-Precision and R is the BER T-Recall.
3. METEOR Score: Considers precision, recall, and synonym match-
ing for textual similarity . Unlike BLEU, METEOR accounts for word
stemming and paraphrasing, making it more sensitive to variations in
word choice and sentence structure that still convey equivalent mean-
ings ( Banerjee and Lavie ,2005 ). The METEOR score is calculated
as:
Fmean =10PR
R+ 9P(3)
Penalty = 0.5×(#chunks
#unigrams _matched)3
(4)
Score =Fmean×(1−Penalty ) (5)
4. ROUGE Score: Measures n-gram overlap and longest common subse-
quence to assess content coverage and informativeness. ROUGE-1 and
ROUGE-2 evaluate unigram and bigram overlaps respectively , while
29

ROUGE-L captures the longest sequence of words in common, reflect-
ing overall content similarity and coherence ( Lin ,2004 ).
ROUGE-1 and ROUGE-2:
Precision =Number of matching n-grams
Number of n-grams in candidate summary(6)
Recall =Number of matching n-grams
Number of n-grams in reference summary(7)
F1= 2×Precision ×Recall
Precision +Recall(8)
where, we can change “n-grams” with “unigrams” or “bigrams” de-
pending on whether it’s ROUGE-1 or ROUGE-2.
ROUGE-L:
Precision =Length of LCS
T otal number of words in candidate summary(9)
Recall =Length of LCS
T otal number of words in reference summary(10)
Fβ= (1 + β2)×Precision ×Recall
(β2×Precision ) + Recall(11)
where, LCS is the Longest Common Subsequence and βis a parameter
that balances precision and recall.
5. BLEU Score: Computes n-gram precision to evaluate fluency and
surface-level correspondence with reference text. It rewards outputs
30

that share similar word sequences with reference answers, serving as
a traditional benchmark for evaluating the quality and grammatical
correctness of generated text ( Papineni et al. ,2002 ). The BLEU score
is computed using the geometric mean of modified n-gram precisions
and a brevity penalty factor.
BP=

1, ifc > r
e(1−r/c), ifc≤r(12)
BLEU =BP×exp(N∑
n=1wnlogpn)
(13)
where pnis the modified n-gram precision, wnare positive weights such
that∑N
n=1wn= 1 ,cis the length of the candidate translation, and r
is the reference length.
Rationale for Metric Selection: These metrics were chosen to provide
a holistic evaluation of both answer accuracy and rationale quality . Accu-
racy serves as a direct measure of the model’s correctness in selecting the
right option for MCQs, while BER TScore and METEOR capture semantic
and linguistic alignment, assessing whether the model’s reasoning conveys
the intended meaning even if exact words differ. ROUGE metrics evaluate
the extent to which the generated rationales cover relevant content from the
reference, including structural and sequence similarities. BLEU scores com-
plement this by assessing surface-level fluency and n-gram precision, ensuring
that generated explanations are coherent and readable.
By combining these metrics, the evaluation captures multiple dimensions
of performance: factual correctness, semantic fidelity , content coverage, and
31

linguistic quality . This is particularly important for assessing models across
both English and Bangla datasets, where the goal is to measure not only
accuracy but also the quality and clarity of the generated rationales.
4. Results and Discussion
4.1. Presenting the Results
Across all methods, as shown in T able 1 , Zero-Shot serves as the baseline
with accuracies from 42.15% (llama-3.1-8b-instant) to 82.60% (openai/gpt-
oss-20b). T raditional RAG improves to 65.79%–86.32%, while RAG with
Zero-Shot F allback reaches 87.22%, and Iterative F eedback RAG offers mod-
est gains but underperforms advanced methods. Agentic RAG achieves
the highest accuracy at 89.54% (openai/gpt-oss-120b), showcasing its dy-
namic retrieval strategy , followed by Aggregate k-values RAG at 84.51%
(openai/gpt-oss-120b). Larger models like llama-3.3-70b-versatile outper-
form smaller variants (e.g., llama-3.1-8b-instant), though gains diminish be-
yond openai/gpt-oss-20b.
Agentic RAG also excels in rationale quality , with superior ROUGE-
1, ROUGE-L, and BLEU scores, indicating strong linguistic and semantic
coherence. Zero-Shot and Iterative F eedback RAG show solid BER TScore
and METEOR values, preserving semantic meaning, while T raditional and
F allback RAG maintain balanced performance. Aggregate k-values RAG
remains consistent despite retrieval noise.
In summary , Agentic RAG with Zero-Shot F allback offers the best com-
bination of accuracy and rationale quality , making it the most effective ap-
proach for BanglaMedQA.
32

T able 1: Performance Metrics on BanglaMedQA Across Different RAG Configurations.
The best results for each metric are shown in bold.
Model Accuracy BER T METEOR ROUGE-1 ROUGE-2 ROUGE-L BLEU-1 BLEU-2
Zero-Shot
llama-3.3-70b-versatile 73.04% 0.7557 0.1426 0.2063 0.0714 0.1853 0.1414 0.0804
llama-3.1-8b-instant 42.15% 0.7244 0.1149 0.1488 0.0374 0.1299 0.1163 0.0547
openai/gpt-oss-120b 82.49% 0.7300 0.1024 0.1506 0.0325 0.1318 0.1143 0.0509
openai/gpt-oss-20b 82.60% 0.7212 0.1048 0.1544 0.0363 0.1362 0.1143 0.0525
Local RAG
llama-3.3-70b-versatile 82.70% 0.7638 0.1444 0.0603 0.0289 0.0586 0.1294 0.0788
llama-3.1-8b-instant 65.79% 0.7422 0.1451 0.0465 0.0200 0.0457 0.1403 0.0782
openai/gpt-oss-120b 86.32% 0.7347 0.1148 0.0825 0.0284 0.0803 0.1215 0.0590
openai/gpt-oss-20b 83.50% 0.7329 0.1220 0.0685 0.0186 0.0675 0.1270 0.0636
Local RAG + Zero-Shot F allback
llama-3.3-70b-versatile 82.90% 0.7657 0.1482 0.0560 0.0295 0.0556 0.1366 0.0821
llama-3.1-8b-instant 66.60% 0.7333 0.1517 0.0562 0.0231 0.0541 0.1424 0.0776
openai/gpt-oss-120b 87.22% 0.7078 0.1137 0.0796 0.0273 0.0774 0.1234 0.0565
openai/gpt-oss-20b 85.81% 0.6984 0.1193 0.0726 0.0208 0.0700 0.1265 0.0609
W eb Search + Zero-Shot F allback
llama-3.3-70b-versatile 87.83% 0.7564 0.1469 0.0468 0.0162 0.0458 0.1422 0.0783
llama-3.1-8b-instant 58.35% 0.7141 0.1473 0.0505 0.0222 0.0497 0.1316 0.0679
openai/gpt-oss-120b 88.83% 0.7126 0.1048 0.0827 0.0226 0.0805 0.1151 0.0516
openai/gpt-oss-20b 87.02% 0.7122 0.1140 0.0657 0.0199 0.0642 0.1197 0.0571
Agentic RAG (Local + W eb) + Zero-Shot F allback
llama-3.3-70b-versatile 86.62% 0.7593 0.1476 0.0494 0.0208 0.0483 0.1429 0.0841
llama-3.1-8b-instant 63.38% 0.7238 0.1459 0.0520 0.0177 0.0504 0.1396 0.0749
openai/gpt-oss-120b 89.54% 0.7274 0.1136 0.0897 0.0271 0.0870 0.1228 0.0565
openai/gpt-oss-20b 88.83% 0.7353 0.1253 0.0763 0.0228 0.0746 0.1339 0.0669
Iterative F eedback RAG
llama-3.3-70b-versatile 78.07% 0.7431 0.1424 0.0643 0.0174 0.1462 0.1129 0.0754
llama-3.1-8b-instant 49.20% 0.7982 0.1352 0.0432 0.0343 0.0563 0.1432 0.0723
openai/gpt-oss-120b 88.73% 0.7537 0.1348 0.0225 0.0255 0.0801 0.1115 0.0591
openai/gpt-oss-20b 87.42% 0.7231 0.1230 0.0655 0.0176 0.0635 0.1170 0.0646
Aggregate k-values RAG
llama-3.3-70b-versatile 82.34% 0.6526 0.1211 0.0621 0.0254 0.0532 0.1424 0.0634
llama-3.1-8b-instant 51.81% 0.6321 0.1452 0.2314 0.0732 0.1642 0.1113 0.0697
openai/gpt-oss-120b 84.51% 0.6925 0.1059 0.1412 0.0391 0.1227 0.1124 0.0579
openai/gpt-oss-20b 83.95% 0.6688 0.0938 0.1221 0.0310 0.1074 0.0951 0.0453
33

T able 2: Accuracy V alues for MMedBench Dataset and Its Bangla T ranslation
ModelBangla Dataset English Dataset
Zero-Shot W eb Search Zero-Shot W eb Search
llama-3.3-70b-versatile 62.08% 60.00% 88.90% 83.86%
openai/gpt-oss-120b 90.59% 82.97% 92.47% 89.90%
T able 2 presents accuracy values for the MMedBench dataset and its
Bangla translation across models and settings. F or Bangla, openai/gpt-
oss-120b achieved 90.59%, outperforming llama-3.3-70b-versatile at 62.08%.
With W eb Search, accuracies dropped to 82.97% and 60.00%, respectively .
On the English dataset, openai/gpt-oss-120b reached 92.47% (Zero-Shot)
and 89.90% (W eb Search), while llama-3.3-70b-versatile scored 88.90% and
83.86%. Zero-Shot performs better on English, while W eb Search often
reduces accuracy , especially for BanglaMMedBench, due to scenario-based
questions where generic web content offers limited value.
4.2. Interpreting the Results
T raditional RAG improved performance over Zero-Shot by grounding an-
swers in textbook content, as shown in T able 1 , confirming that external
knowledge enhances factual accuracy . However, gains varied across models
due to differences in embedding quality , Llama models showed larger jumps
than GPT-oss, which already performed well in Zero-Shot. The fallback
strategy brought minor (1–2%) improvements, mainly by avoiding unan-
swered questions, though it sometimes retained irrelevant long passages.
Agentic RAG achieved the best overall results, especially with larger mod-
els such as openai/gpt-oss-120b . Its routing mechanism reduced reliance
34

on context length, enhancing robustness, though occasional errors arose from
imperfect routing and noisy web data. Iterative F eedback RAG improved
over Zero-Shot but inconsistently; for instance, it reached 78.07% accuracy
for llama-3.3-70b-versatile versus 88.73% for openai/gpt-oss-120b , in-
dicating sensitivity to model capacity and initial retrieval quality . Errors
could propagate during iterations, limiting reliability . Aggregate k-values
RAG demonstrated stable performance, just below Agentic RAG, by com-
bining predictions across multiple retrieval sizes. Its accuracies, 82.34%
(llama-3.3-70b-versatile ) and 84.51% ( openai/gpt-oss-120b ), indicate
that aggregating diverse contexts balances precision and coverage, reducing
missed information.
On the MMedBench dataset and its Bangla translation, as shown in T a-
ble 2 , Zero-Shot and W eb Search methods performed better on the English
version, reflecting the models’ English-centric pretraining. In the Bangla
version, openai/gpt-oss-120b surpassed llama-3.3-70b-versatile , sug-
gesting that model scale can help reduce cross-lingual gaps. However, W eb
Search sometimes decreased accuracy , as generic web content rarely aligned
with the scenario-based, clinical nature of MMedBench questions. This shows
that web grounding does not always guarantee performance improvement
across domains and languages.
5. Conclusion
This study introduced BanglaMedQA and BanglaMMedBench, two Bangla
biomedical Multiple Choice Question (MCQ) datasets. Several Retrieval-
Augmented Generation (RAG) strategies were evaluated for question an-
35

swering in a low-resource setting. Using a Bangla medical textbook corpus
for retrieval, the Agentic RAG approach achieved the highest accuracy of
89.54% with openai/gpt-oss-120b. Iterative F eedback RAG also showed con-
sistent and interpretable results across models. LLaMA models benefited
more from retrieval-based augmentation, while web search retrieval some-
times reduced accuracy due to content mismatch. Despite limitations in
dataset size, translation quality , and computational resources, this work
provides the first comprehensive benchmark for Bangla biomedical QA and
demonstrates the potential of retrieval-based reasoning in advancing multi-
lingual medical artificial intelligence.
CRediT authorship contribution statement
Sadia Sultana: Conceptualization, Data curation, Methodology , F or-
mal analysis. Saiyma Sittul Muna: Conceptualization, Data curation,
Investigation, Methodology . Mosammat Zannatul Samarukh: Concep-
tualization, Data curation, Visualization, F ormal analysis. Ajwad Abrar:
W riting – review & editing, Supervision, Project administration. T areque
Mohmud Chowdhury: Supervision, V alidation, F unding acquisition.
Data A vailability Statement
All data is available on a public Hugging F ace repository: https://
huggingface.co/datasets/ajwad-abrar/BanglaMedQA .
36

Declaration of competing interest
The authors declare that they have no known competing financial inter-
ests or personal relationships that could have appeared to influence the work
reported in this paper.
References
Abrar, A., T abassum, F., Ahmed, S., 2024. Performance evaluation of large
language models in bangla consumer health query summarization, in: 2024
27th International Conference on Computer and Information T echnology
(ICCIT), pp. 2748–2753. doi: 10.1109/ICCIT64611.2024.11022034 .
Banerjee, S., Lavie, A., 2005. Meteor: An automatic metric for mt evaluation
with improved correlation with human judgments, in: Proceedings of the
acl workshop on intrinsic and extrinsic evaluation measures for machine
translation and/or summarization, pp. 65–72.
Dev, S., 2025. Serper — the world’s fastest & cheapest google search api.
https://serper.dev/ . Accessed: 2025-10-21.
Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy , G., Mazaré, P .E.,
Lomeli, M., Hosseini, L., Jégou, H., 2025. The faiss library . IEEE T rans-
actions on Big Data .
Dubey , A., Jauhri, A., Pandey , A., Kadian, A., Al-Dahle, A., Letman, A.,
Mathur, A., Schelten, A., Y ang, A., F an, A., et al., 2024. The llama 3 herd
of models. arXiv e-prints , arXiv–2407.
37

Georgiev, P ., Lei, V.I., Burnell, R., Bai, L., Gulati, A., T anzer, G., Vin-
cent, D., Pan, Z., W ang, S., Mariooryad, S., Ding, Y., Geng, X., Al-
cober, F., F rostig, R., Omernick, M., W alker, L., Paduraru, C., Sorokin,
C., T acchetti, A., Gaffney , C., Daruki, S., Sercinoglu, O., Gleicher, Z.,
Love, J., V oigtlaender, P ., Jain, R., Surita, G., Mohamed, K., Blevins,
R., Ahn, J., Zhu, T., Kawintiranon, K., Firat, O., Gu, Y., Zhang, Y.,
Rahtz, M., F aruqui, M., Clay , N., Gilmer, J., Co-Reyes, J., Penchev, I.,
Zhu, R., Morioka, N., Hui, K., Haridasan, K., Campos, V., Mahdieh, M.,
Guo, M., Hassan, S., Kilgour, K., V ezer, A., Cheng, H.T., de Liedek-
erke, R., Goyal, S., Barham, P ., Strouse, D., Noury , S., Adler, J., Sun-
dararajan, M., Vikram, S., Lepikhin, D., Paganini, M., Garcia, X., Y ang,
F., V alter, D., T rebacz, M., V odrahalli, K., Asawaroengchai, C., Ring,
R., Kalb, N., Baldini Soares, L., Brahma, S., Steiner, D., Y u, T., et al.,
￿., 2024. Gemini 1.5: Unlocking multimodal understanding across mil-
lions of tokens of context. arXiv preprint arXiv:2403.05530 URL: https:
//arxiv.org/abs/2403.05530 , doi: 10.48550/arXiv.2403.05530 .
Groq, 2025. Groq console — organization settings. https://console.groq.
com/settings/organization . Accessed: 2025-10-21.
Hegghammer, T., 2022. Ocr with tesseract, amazon textract, and google
document ai: A benchmarking experiment. Journal of Computational
Social Science 5, 861–882. doi: 10.1007/s42001-021-00149-1 .
Inuwa-Dutse, I., 2025. Openai’s gpt-oss-20b model and safety alignment
issues in a low-resource language. arXiv preprint arXiv:2510.01266 .
Izacard, G., Grave, E., 2021. Leveraging passage retrieval with generative
38

models for open domain question answering, in: Proceedings of the 16th
Conference of the European Chapter of the Association for Computational
Linguistics, pp. 874–880.
Jahan, I., Laskar, M.T.R., Peng, C., Huang, J., 2023. A comprehensive eval-
uation of large language models on benchmark biomedical text processing
tasks. arXiv preprint arXiv:2310.04270 URL: https://arxiv.org/abs/
2310.04270 .
Jin, Q., Dhingra, B., Cohen, W., Lu, X., 2019. Pubmedqa: A dataset for
biomedical research question answering, in: Proceedings of the 2019 Con-
ference on Empirical Methods in Natural Language Processing (EMNLP),
Association for Computational Linguistics. pp. 2567–2577. URL: https:
//aclanthology.org/D19-1259 , doi: 10.18653/v1/D19-1259 .
Khan, A., Kamal, F., Nower, N., Ahmed, T., Ahmed, S., Chowdhury , T.,
2023. NERvous about my health: Constructing a Bengali medical named
entity recognition dataset, in: Findings of the Association for Computa-
tional Linguistics: EMNLP 2023, Association for Computational Linguis-
tics, Singapore. pp. 5768–5774. URL: https://aclanthology.org/2023.
findings-emnlp.383 , doi: 10.18653/v1/2023.findings-emnlp.383 .
Koistinen, M., Kettunen, K., Kervinen, J., 2017. How to improve optical
character recognition of historical finnish newspapers using open source
tesseract ocr engine. Proc. of L TC , 279–283.
Lewis, P ., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küt-
tler, H., Lewis, M., tau Yih, W., Rocktäschel, T., et al., 2020. Retrieval-
39

augmented generation for knowledge-intensive NLP tasks, in: Advances in
Neural Information Processing Systems (NeurIPS).
Lin, C.Y., 2004. Rouge: A package for automatic evaluation of summaries,
in: T ext summarization branches out, pp. 74–81.
Maurya, A., Kumar, A., Alimohammadi, D., 2023. Application of google
lens to promote information services beyond the traditional techniques.
Qualitative and Quantitative Methods in Libraries 12, –. URL: https:
//www.qqml-journal.net/index.php/qqml/article/view/810 .
Pal, A., Umapathi, L.K., Sankarasubbu, M., 2022. Medmcqa: A large-
scale multi-subject multi-choice dataset for medical domain question an-
swering, in: Proceedings of the 35th Conference on Neural Information
Processing Systems (NeurIPS) Datasets and Benchmarks T rack. URL:
https://arxiv.org/abs/2203.14371 ,arXiv:2203.14371 .
Papineni, K., Roukos, S., W ard, T., Zhu, W.J., 2002. Bleu: a method for
automatic evaluation of machine translation, in: Proceedings of the 40th
annual meeting of the Association for Computational Linguistics, pp. 311–
318.
Qin, L., et al., 2025. A survey of multilingual large language models. Journal
of Computational Linguistics A vailable via PMC: comprehensive review
and taxonomy of MLLMs.
Qiu, P ., W u, C., Zhang, X., Lin, W., W ang, H., Zhang, Y., W ang, Y., Xie,
W., 2024a. T owards building multilingual language model for medicine.
Nature Communications 15, 8384.
40

Qiu, P ., W u, C., Zhang, X., Lin, W., W ang, H., Zhang, Y., W ang, Y., Xie,
W., 2024b. T owards building multilingual language model for medicine.
Nature Communications 15, 8384.
Salemi, A., Zamani, H., 2024. Evaluating retrieval quality in retrieval-
augmented generation, in: Proceedings of the 47th International ACM
SIGIR Conference on Research and Development in Information Retrieval,
pp. 2395–2400.
Shafayat, S., Hasan, H.M.Q., Mahim, M.R.C., Putri, R.A., Thorne, J., Oh,
A., 2024. BEnQA: A question answering benchmark for Bengali and En-
glish, in: Ku, L.W., Martins, A., Srikumar, V. (Eds.), Findings of the As-
sociation for Computational Linguistics: ACL 2024, Association for Com-
putational Linguistics, Bangkok, Thailand. pp. 1158–1177. URL: https:
//aclanthology.org/2024.findings-acl.68/ , doi: 10.18653/v1/2024.
findings-acl.68 .
Shinn, N., Casper, S., et al., 2023. Reflexion: Language agents with ver-
bal reinforcement learning. URL: https://arxiv.org/abs/2303.11366 ,
arXiv:2303.11366 .
Smith, J.R., Johnson, A., 2005. Current challenges in biomedical ques-
tion answering. Journal of Biomedical Informatics 48, 1–12. URL:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11922739/ , doi: 10.
1016/j.jbi.2004.10.003 .
Smith, R., 2007. An overview of the tesseract ocr engine, in: Proceedings of
the Ninth International Conference on Document Analysis and Recogni-
41

tion (ICDAR ’07), pp. 629–633. URL: https://ieeexplore.ieee.org/
document/4376991/ , doi: 10.1109/ICDAR.2007.4376991 .
T satsaronis, G., Balikas, G., Malakasiotis, P ., Androutsopoulos, I., et al.,
2015. An overview of the BioASQ large-scale biomedical semantic in-
dexing and question answering competition. BMC Bioinformatics 16,
138. URL: https://doi.org/10.1186/s12859-015-0564-6 , doi: 10.
1186/s12859-015-0564-6 .
V avekanand, R., Sam, K., 2024. Llama 3.1: An in-depth analysis of the
next-generation large language model. Preprint, July .
W ork, S., 2023. Bengali medical dataset. https://www.kaggle.com/
datasets/shashwatwork/bengali-medical-dataset .
W u, S., Xiong, Y., Cui, Y., W u, H., Chen, C., Y uan, Y., Huang, L., Liu,
X., Kuo, T.W., Guan, N., et al., 2024. Retrieval-augmented generation for
natural language processing: A survey . arXiv preprint arXiv:2407.13193 .
Y ang, H., Chen, H., Guo, H., Chen, Y., Lin, C.S., Hu, S., Hu, J.,
W u, X., W ang, X., 2024. Llm-medqa: Enhancing medical question an-
swering through case studies in large language models. arXiv preprint
arXiv:2501.05464 URL: https://arxiv.org/abs/2501.05464 , doi: 10.
48550/arXiv.2501.05464 .
Y ao, S., Y u, J., Zhao, D., Shafran, I., Griﬀiths, T., Cao, Y., Narasimhan,
K., 2022. React: Synergizing reasoning and acting in language models, in:
Proceedings of the International Conference on Machine Learning (ICML).
42

Zhang, T., Kishore, V., W u, F., W einberger, K.Q., Artzi, Y., 2019. Bertscore:
Evaluating text generation with bert. arXiv preprint arXiv:1904.09675 .
Zhang, W., Li, H., Chen, M., 2024. Benchmarking retrieval-augmented gen-
eration for medicine. arXiv preprint arXiv:2402.13178 .
43
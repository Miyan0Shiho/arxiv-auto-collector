# SafeLLM: Extraction as a Hallucination-Resistant Alternative to Rewriting in Safety-Critical Settings

**Authors**: Julia Ive, Felix Jozsa, Evridiki Georgaki, Nabeel Sheikh, Emma Cattell, Nick Jackson, Paulina Bondaronek, Ciaran Scott Hill, Richard Dobson

**Published**: 2026-06-11 04:55:37

**PDF URL**: [https://arxiv.org/pdf/2606.12897v1](https://arxiv.org/pdf/2606.12897v1)

## Abstract
Large language models (LLMs) are increasingly used to access organisational documentation, including standard operating procedures (SOPs), HR policies and institutional guidelines. However, retrieval-augmented generation (RAG) systems that rely on free-form rewriting can introduce hallucinations and unstable trade-offs between completeness and conciseness, particularly in safety- and compliance-critical settings. Objectives: To evaluate extraction as a hallucination-resistant alternative to rewriting-based RAG and compare strategies that balance precision, recall and safety across document types and model scales. Methods: We compare multiple prompting strategies, including line-number-based source selection, extraction of relevant guideline sentences with explicit safety annotations, and a multi-stage pipeline that refines draft answers using supporting evidence from source guidelines. Experiments are conducted on documents of varying length and structure, including local NHS acute care and oncology guidelines and UK-wide NICE guidelines, using both frontier-scale and locally deployable models. Performance is assessed using automatic metrics and human expert evaluation of relevance and completeness. Results: Line-number selection achieves the strongest results, outperforming direct copying and safety-focused strategies across both large and small models while maintaining high term recall (up to 95%) and close alignment with source text. Safety-oriented approaches improve precision but introduce systematic omissions, while multi-stage filtering further amplifies this trade-off. Performance varies with document structure: line-based extraction excels in protocol-like content, whereas alternative strategies perform better on more verbose documents (up to 97% term recall).

## Full Text


<!-- PDF content starts -->

SafeLLM: Extraction as a Hallucination-Resistant
Alternative to Rewriting in Safety-Critical Settings
Julia Ive1,6,*, Felix Jozsa1,2,6,†, Evridiki Georgaki1, Nabeel Sheikh3, Emma Cattell3, Nick
Jackson4, Paulina Bondaronek1, Ciaran Scott Hill2, and Richard Dobson1,5
1Institute of Health Informatics, University College London, London, UK
2National Hospital for Neurology and Neurosurgery, Queen Square, London, UK
3Somerset NHS Foundation Trust, UK
4King’s College Hospital, Denmark Hill, London, UK
5King’s College London, London, UK
6These authors contributed equally: Julia Ive, Felix Jozsa
*julia.ive@ucl.ac.uk
1f.jozsa@ucl.ac.uk
ABSTRACT
Background:Large language models (LLMs) are increasingly used to access organisational documentation, including
standard operating procedures (SOPs), HR policies and institutional guidelines. However, retrieval-augmented generation
(RAG) approaches that rely on free-form rewriting can introduce hallucinations and unstable trade-offs between completeness
and conciseness, particularly in safety- and compliance-critical settings.
Objectives:To evaluate extraction as a hallucination-resistant alternative to rewriting-based RAG, and to compare strategies
that balance precision, recall and safety across document types and model scales.
Methods:We compare multiple prompting strategies, including line-number-based source selection, extraction of relevant
guideline sentences with explicit annotation of safety-related content, and a multi-stage pipeline that refines draft answers using
supporting evidence from the source guideline. All approaches outperform direct prompting that relies on the model to copy
relevant text from the source document. Experiments are conducted on documents of varying length (local NHS guidelines
in acute care and oncology, as well as UK-wide NICE guidelines), and structure using both frontier-scale and smaller locally
deployable models. Performance is assessed using automatic metrics and via human expert evaluation of relevance and
completeness.
Results:Line-number selection achieves the strongest results outperforming direct copying and safety-focused strategies
across both large and small models, while maintaining high term recall (up to 95%) and close alignment with source text.
Safety-oriented approaches improve precision but introduce systematic omissions, and multi-stage filtering further amplifies
this trade-off. Performance varies with document structure: line-based extraction excels in protocol-like content, whereas
alternative strategies show advantages in more verbose documents (up to 97% term recall).
Conclusions:Line-number selection provides a robust alternative to rewriting-based RAG for accessing organisational
documentation. Drafting-based approaches may offer additional benefits in very long-document settings. These findings
highlight the importance of adapting retrieval strategies to document structure, model scale and the risks of omission.
Introduction
Administrative and operational documents, including clinical guidelines, standard operating procedures (SOPs), and organisa-
tional policies, provide consensus-based instructions for decision-making and task execution1. Across domains, adherence to
such documents is a core requirement of good professional practice and is associated with improved safety, compliance, and
operational consistency2–8.
In practice, high-level guidance is complemented by local or departmental documents that adapt recommendations to
specific workflows, constraints, and organisational structures. Users consult these resources to answer focused, task-oriented
questions, for example, what actions to take, which approvals are required, or how to escalate a situation. However, their use
is often hindered by complexity and time pressure9–12, challenges that are amplified by increasing workload and growing
document volume13–15. In high-stakes or time-critical settings, these limitations can lead to delays, errors, or non-adherence.
Large Language Models (LLMs) offer a promising approach to document-based question answering (Q&A)16, but their
deployment is constrained by hallucinations, i.e. generated content not supported by source evidence17. Retrieval-AugmentedarXiv:2606.12897v1  [cs.CL]  11 Jun 2026

Generation (RAG) mitigates this by grounding responses in retrieved documents18, improving factual consistency19. However,
it does not fully eliminate errors: models may omit critical steps, introduce irrelevant details, or fail to faithfully reproduce
source text, posing risks in safety- and compliance-sensitive contexts3, 20–22.
A central challenge in document-based Q&A is balancing interpretation with faithful reproduction. Some queries require
integrating information distributed across multiple sections or adapting conditional instructions to a specific scenario. Others
demand precise, verbatim reproduction to preserve meaning, particularly where exact wording carries operational or regulatory
significance.
Current approaches focus on improving generation quality, rather than controlling when generation should occur3, 22. Here,
we take the opposite approach: we minimise generation. We constrain models to operate directly on source text through
extraction, enabling them to identify and present relevant information without rewriting. This reframes the task from answer
generation to evidence selection, reducing opportunities for hallucination while preserving fidelity to source material.
Our main contribution is the development and evaluation of an extraction framework for the use case of clinical question
answering that minimises hallucinations while balancing the risk of adding information versus omitting clinically critical details.
In contrast to rewriting-based approaches, we constrain LLMs to select and copy guideline text, targeting high fidelity to source
material, an essential requirement in settings where even small inaccuracies in wording (e.g., drug dosing, contraindications, or
escalation criteria) may impact patient safety. We evaluate this approach across multiple guideline use cases that vary in length
and structure, including concise acute care protocols from University College London Hospital (UCLH), oncology-focused
institutional guidelines from Somerset NHS Trust, and large-scale national recommendations (NICE). These settings capture a
spectrum of real-world decision-making scenarios. Evaluation is conducted using human expert-developed question–answer
pairs grounded in these documents, reflecting practical clinical information needs under time pressure. We further assess
performance across a range of LLM configurations, including both open-weight and commercial models, to characterise
robustness across deployment settings. Rather than attempting to make LLMs better generators, we argue that constraining
them offers a more reliable path for deployment in safety-critical clinical environments.
Methods
Ethical Considerations
This study was approved by the UCL Review Board as TRE-467283, and approved as a clinical service audit at the National
Hospital for Neurology and Neurosurgery, University College London Hospital (UCLH) as 130-202425-SE. The guideline data
used in this study contained no patient-level or personally identifiable information. All materials were derived from institutional
clinical guidelines intended for professional use and therefore posed no risk to patient privacy.
Data
We evaluate our methods across three datasets spanning distinct document styles and scales: concise acute care guidelines
(UCLH), moderate-length institutional guidelines (Somerset NHS Trust), and large-scale national recommendations (NICE).
The UCLH dataset consists of 96 clinically grounded questions grouped under 24 acute care topics, designed by FJ and NJ
to reflect real-world scenarios encountered by frontline clinicians. Topics include, for example,acute coronary syndrome, where
questions probe escalation decisions (e.g., “If first-line therapy does not relieve symptoms, when is a second dose indicated?”),
andacute kidney injury, focusing on investigation and management steps (e.g., “Which blood tests are essential on admission?”).
Reference answers were constructed via direct selection of relevant guideline lines to ensure grounding in source material.
The Somerset NHS Trust dataset follows a similar question design process but differs in annotation strategy. Here, reference
answers were generated exclusively through clinician free writing (NS and FJ). These guidelines are predominantly oncology-
oriented and capture acute complications of cancer and its treatment, such asneutropenic sepsis,metabolic disturbances, and
treatment-related toxicities. Despite being moderate in length, they are strongly acute in nature, often focusing on time-critical
recognition, escalation, and stabilisation decisions in vulnerable patient populations.
In contrast, the NICE dataset was generated synthetically following the protocol of Ding et al.23, using gpt-4o-mini to
produce clinical scenario–question–guideline paragraph triplets from full-length recommendation documents. To reflect realistic
model inputs, we merge the scenario description and associated question into a single input prompt. All generated outputs were
subsequently manually reviewed by human experts to ensure clinical validity. This dataset captures scenario-related inputs, for
example withinchronic kidney diseaseorheart failuremanagement. A representative example is:
“A 68-year-old patient with a history of hypertension and type 2 diabetes presents with progressive fatigue and ankle
swelling over several weeks. Blood tests show reduced eGFR and elevated creatinine. According to current guidelines, what
investigations and initial management steps should be undertaken, and when should specialist referral be considered?”
This contrasts with the shorter, focused queries in UCLH and Somerset, such as:“Which blood tests are essential in acute
kidney injury?”(see Tables A1 and A2).
2/22

Table 1.Summary statistics of the evaluation datasets spanning three document-length regimes: concise acute care guidelines
(UCLH), moderate-length institutional guidelines (Somerset NHS Trust), and large-scale national recommendations (NICE).
Values are reported as mean (min–max).
UCLH Somerset NHS Trust NICE
Questions (n)96 50 500
Guidelines (n)25 6 10
Question length (words)9.6 (3–17) 11.2 (2–21) 45.9 (28–78)
Answer length (words)39.4 (1–139) 36.5 (5–145) 28.0 (9–87)
Guideline length (lines)73 (34–152) 438 (317–636) 881 (258–1466)
Guideline length (words)746 (375–1360) 3153 (2372–4510) 5224 (1437–8007)
Total corpus (words)18,578 21,025 52,236
Total corpus (lines)1,824 2,825 8,809
Figure 1.Overview of the proposed guideline question-answering pipeline. Given a user question, the system first performs
Topic Selectionto identify the most relevant clinical guideline from the guideline index. Once the relevant guideline is
selected, the system applies one of severalprompting strategiesto extract the evidence required to answer the question. The
evaluated strategies include: (i)COPY, where the model extracts verbatim sentences; (ii)LINES, where the model selects
relevant guideline line numbers; (iii)JuniorDoc, a multi-agent workflow where a junior-doctor-style draft answer is generated,
verified against the guideline, and filtered by a safety editor; and (iv)Safety, which extracts the guideline sentences relevant to
the question and annotates safety-critical content to support safe evidence-based responses.
All underlying guideline documents were extracted from PDF files using the pdftotext utility to preserve original
wording, followed by manual cleaning to remove tables of content, correct formatting artefacts and ensure fidelity to source
content.
Across datasets, question formulation varies substantially, from concise queries in UCLH (mean 9.6 words) to longer,
scenario-based inputs in NICE (mean 45.9 words), with Somerset NHS Trust occupying an intermediate position. Importantly,
the guideline corpora differ by an order of magnitude in length: short UCLH documents (mean 746 words), moderate-length
Somerset NHS Trust guidelines (mean 3,153 words), and long-form NICE recommendations (mean 5,224 words; maximum
8,007 words) (Table 1).
Prompting Strategies for Guideline-Based Clinical Question Answering
This section outlines the methods evaluated in this study, starting with a baseline retrieval approach based on semantic similarity
between the user question and guideline text. We then present a set of prompting pipeline methods that combine an initial topic
identification step with different prompting regimes for extracting relevant evidence from clinical guidelines (see Fig. 1, see
summary of the approaches in Table 2).
Baseline Retrieval (BASE)As a baseline retrieval strategy, we construct a repository of information chunks derived from
guideline text, each representing a self-contained unit of clinically relevant content that may be used to answer a question. From
each guideline, we extract guideline sections and, for each section, the section title, section text, and a representative actionable
question.
These sections are extracted automatically from guideline paragraphs using an LLM prompt. This section extraction prompt
(Listing 1) decomposes each paragraph into smaller, clinically meaningful overlapping subparts. For each section, it then
generates a short descriptive title and an actionable clinical question. By associating each text segment with a representative
question, we aim to reduce sensitivity to phrasing variation between guideline text and associated queries.
3/22

Figure 2.Construction of the representative question–section index used by the baseline retrieval strategies (BASE). Each
guideline document is first segmented into paragraphs, from which clinically coherent overlapping sections are extracted. For
every section, a representative question and section title are generated using an LLM. Each subpart is represented by four
elements: the guideline topic, section title, representative question, and verbatim section text. Two indexing variants are
considered: (a)text, where embeddings are computed from guideline topic, section title and text concatenated together into a
single string; and (b)text + question, where embeddings are computed as in (a) plus the representative question. Representative
questions act as a semantic interface between guideline text and user queries. At query time, the user question is embedded
using the same embedding model and compared against section embeddings via cosine similarity.
At indexing time, each subpart is embedded using one of two representations (Figure 2): (a)text, based on guideline
topic, section title, and section text concatenated into single string; or (b)text + question, which additionally incorporates the
representative question (guideline topic, section title, section text and representative question concatenated into single string).
At query time, the user question is embedded using the same model and compared to all embeddings (variant (a) or (b)) via
cosine similarity. Retrieval is then performed by selecting the subparts whose representations are most semantically aligned
with the query.
Table 2.Summary of prompting strategies. All methods begin with Topic Selection, which identifies the most relevant
guideline document before extraction.
Abbrev Strategy Key Idea Hallucination Control
BASE Baseline Retrieval Retrieve relevant guideline sections using embedding
similarity between user queries and those sectionsRetrieval-only baseline
Topic Topic Selection Select the single most relevant guideline document
before answering the question.Limits search space to one
guideline
COPY Verbatim Sentence Extrac-
tionModel copies only exact sentences from the guideline
required to answer the questionEnforces copying
LINES Line Selection Model outputs line numbers corresponding to relevant
guideline text. The final answer is reconstructed from
those linesGuarantees verbatim text.
Eliminates hallucination
Junior Junior Doctor Workflow Multi-agent reasoning: junior doctor draft answers are
rewritten using supporting evidence from the source
guidelineGrounding
Safety Annotated Safety Extrac-
tionModel selects the relevant set of guideline sentences
that answer the question and annotated each with a
role label (e.g., answer, detail, safety)Copying with explicit role
labelling
Step 1: Topic Selection (Topic)Topic selection is the first step across all strategies. Given a user query, the model selects
exactly one guideline topic from a predefined index of topics. This design follows the workflow observed in clinical practice.
Questions typically relate to a single diagnosis, meaning clinicians first identify the relevant guideline before consulting detailed
sections (see Listing 2).
Since our study focuses on acute medical problems encountered on an undifferentiated hospital ward, clinical questions can
generally be mapped to a single primary issue, even for patients with substantial multimorbidity. Furthermore, many acute
4/22

conditions, such as hyperkalaemia or hypoglycaemia, are managed according to standard protocols that are largely independent
of a patient’s underlying comorbidities. Where management does require adaptation for specific patient groups, such as those
with chronic kidney disease, these considerations are typically captured within dedicated sections of the same guideline. This
structure supports a topic-first retrieval strategy, in which a single guideline is identified before relevant subsections are selected.
Step 2: Evidence Retrieval Strategies
Verbatim Sentence Strategy (COPY)We first introduce COPY as a prompting baseline that probes the model’s ability to
directly copy relevant information from the guideline. The model is instructed to return only verbatim sentences that answer the
user question, with explicit constraints forbidding paraphrasing and requiring exact reproduction of the source text.
This setup probes models’ capacity for faithful extraction, providing a reference point for more constrained strategies (see
Listing 4).
Line Selection Strategy (LINES)Building on this baseline, the LINES strategy introduces a more sophisticated mechanism.
The guideline text is augmented with line numbers and provided to the model together with the user question. The model is
then tasked with identifying the minimal set of line numbers required to answer the question.
The final answer is reconstructed deterministically by retrieving the corresponding guideline lines in post-processing. This
approach enforces tighter control over the extracted content and guarantees the absence of hallucinated text, as the output
consists exclusively of selected guideline line numbers (see Listing 3).
Junior Doctor Workflow (Junior)We also design a multi-agent prompting strategy that mimics a realistic clinical reasoning
workflow.
Agent 1 acts as a junior doctor and generates an initial answer based only on training memory, without consulting external
documents. This simulates the real-world setting in which clinicians often rely on prior knowledge under time pressure. The
agent is instructed to provide concise answers containing actionable details such as thresholds, medication doses, or monitoring
steps (see Listing 5). Agent 2 then compares the draft answer against the official guideline text and selects the guideline
sentences that best support, clarify, or correct it, preserving their original order (see Listing 6, see Figure 3).
Figure 3.Illustration of theJuniorDocprompting workflow. The user question is first processed byAgent 1, which acts as a
junior clinician reasoning from memory and produces an initial draft answer. This draft may be incomplete or imprecise
because it did not yet consult the guideline text.Agent 2then grounds the answer by extracting verbatim sentences from the
relevant clinical guideline section that support, refine, or correct the initial draft. This multi-stage design combines clinical
reasoning and evidence grounding.
Minimal Safety Extraction (Safety)We also evaluate a prompting strategy that instructs the model to extract the minimal
safe set of sentences required to answer the user’s question. The model is prompted to select the smallest subset of full
sentences that (i) directly answer the question and (ii) include any safety-critical information. Each selected sentence is
annotated with a single reason label indicating its role in the response: answers_question ,adds_required_detail ,
orsafety_critical . This prompt also enforces strict constraints: sentences must be copied verbatim from the guideline,
paraphrasing is not allowed, and the original order must be preserved.
The Safety strategy is motivated by the observation that guideline answers often contain both essential recommendations
and surrounding explanatory text. By explicitly defining roles, the model is guided to retain only the most relevant content.
LLM Types
To evaluate the robustness of our methods across different model architectures and deployment settings, we conducted experi-
ments using several open-weight LLMs available through the Ollama framework1. Specifically, we evaluated gpt-oss-120b24
1https://ollama.com
5/22

Table 3.Automatic evaluation metrics highlighting complementary aspects of verbosity, coverage and clinical fidelity.
Metric Type Purpose in this study
AnsLen Output length Measures average answer length; captures verbosity
ROUGE-L Sequence-level overlap Measures ordered overlap with the reference (longest
common subsequence); captures sentence-level recall
and rewards coherent reconstruction
Word P/R/F1 Word-level precision, recall
and F1Word overlap robust to formatting differences. Precision
reflects unsupported additions, recall reflects coverage,
and F1 balances both
MedicalTerm P/R/F1 Medical term overlap Evaluates preservation of clinically salient concepts
andqwen3:32b25. These models provide openly available weights, enabling reproducibility and supporting deployment
in secure environments where external API access may be restricted (e.g., clinical or trusted research environments). Using
open-weight models also allows evaluation under controlled and identical inference conditions. All models were run with
default hyperparameters except for the temperature parameter, which was fixed at 0 to minimise non-deterministic variation.
For the baseline retrieval strategy (BASE), we additionally experimented with three embedding models available through
Ollama: nomic-embed-text ,embeddinggemma , and qwen3-embedding . These models differ primarily in their
training objectives and underlying model families. nomic-embed-text is designed specifically for semantic search
and retrieval tasks, embeddinggemma provides lightweight embeddings derived from the Gemma model family26, and
qwen3-embedding is based on the Qwen architecture and trained to produce dense representations aligned with instruction-
tuned models.
For the open-access NICE guideline data, we additionally evaluate large-scale API-based models to assess how our
extraction strategies generalise to frontier models beyond locally deployable setups. Specifically, via the Replicate plat-
form, we include models spanning a wide scale range: deepseek-ai/deepseek-v3 (very large, 671B parameters),
qwen/qwen3-235b-a22b-instruct-2507 (medium, 235B parameters), and gpt-5o-mini (compact, parameter
count undisclosed, typically <30B). We deliberately exclude gpt-4o-mini from this evaluation, as it was involved in the
creation of the synthetic NICE dataset.
Evaluation
We evaluate the proposed methods using both automatic metrics and human assessment.
Automatic EvaluationWe first report average answer length (AnsLen) to understand model verbosity, as longer outputs tend
to increase recall-oriented metrics while potentially reducing precision.
We then evaluate overlap with reference answers using ROUGE-L27, which can be interpreted simply as a form ofsentence-
level recall: it measures how much of the reference content is recoveredin the correct order, rewarding correct reconstruction
of long chunks of text rather than fragmented matches.
To capture more fragmented lexical overlap, we report word-level precision, recall, and F1 (P, R, F1). Both reference and
generated answers are normalised by removing non-alphanumeric characters, lowercasing, and splitting into tokens on space.
Precision is defined as the proportion of generated tokens supported by the reference, recall as the proportion of reference
tokens recovered, and F1 as their harmonic mean. This formulation is robust to word order and minor formatting differences.
Finally, we assess entity-level fidelity using medical named entity recognition (NER, medical term) precision, recall, and
F1.
We use MedCAT28for biomedical named entity recognition and concept linking. Specifically, we employ the latest available
MedCAT model packs based on SNOMED CT and UMLS, accessed through UMLS-authenticated releases.2MedCAT identifies
clinical concepts in guideline texts and maps them to standard biomedical terminologies. Evaluation is performed over sets of
unique extracted term spans, with precision and recall defined as the overlap between predicted and reference entity sets, and
F1 as their harmonic mean. This provides a targeted measure of whether clinically salient concepts (e.g., drugs, conditions,
symptoms) are correctly preserved, independent of surrounding text.
Together, these metrics provide complementary views of extraction quality: AnsLen captures verbosity, ROUGE-L reflects
sentence-level overlap, word-level scores capture fragmented lexical overlap, and medical term metrics focus specifically on
clinically relevant entities.
Human EvaluationIn addition to automatic metrics, we conducted a human evaluation to assess the clinical quality of
the generated answers. While automatic metrics quantify textual overlap, they are limited in their ability to capture clinical
correctness, contextual appropriateness, and the safety implications of missing or added information. In guideline-based
2https://medcat.readthedocs.io/en/latest/main.html
6/22

Table 4.Comparison of index characteristics across datasets. UCLH (acute care) shows concise, action-oriented structure;
Somerset (oncology) is more narrative; NICE (national guidance) is highly heterogeneous.
Statistic UCLH Somerset NICE
Number of guidelines 24 6 10
Number of sections 987 469 4,726
Sections per guideline, mean 41.1 78.2 472.6
Sections per guideline, range 6–74 48–125 12–849
Section length, words mean 31.4 65.8 18.6
Question length, words mean 8.7 8.2 8.3
Table 5.Automatic evaluation of baseline (BASE) retrieval across embedding models and representations. We compare
representations based on section text, section topic and title (text only) and representations based on section text, section topic,
title and representative LLM-generated question (text + question). Guideline sections are retrieved using cosine similarity
between embedded user query and representations, and returned in full as answers. Metrics include ROUGE-L (R-L,
sentence-level recall), word-level precision (P), recall (R), and F1, as well as medical term precision, recall, and F1. Answer
length (AnsLen) is reported as mean word count (lower is better). BASE achieves consistently high recall across embeddings,
but low precision and long outputs, indication non-essential content. Best results per representation are shown inbold, second
best are underlined.
Embedding Ans Len R-L P R F1 Term P Term R Term F1
text
gemma 190.97 0.3026 0.2117 0.8372 0.3151 0.2614 0.8191 0.3717
nomic 245.81 0.1720 0.1244 0.6518 0.1872 0.1479 0.6136 0.2166
qwen3 174.03 0.3218 0.2337 0.8369 0.3340 0.3032 0.8327 0.4089
text + question
gemma 257.65 0.2248 0.1514 0.7853 0.2381 0.1857 0.7635 0.2771
nomic 255.74 0.2048 0.1366 0.8035 0.2190 0.1779 0.7860 0.2684
qwen3181.77 0.3332 0.2418 0.8608 0.3430 0.3083 0.8514 0.4105
question answering, small deviations in wording, omission of key recommendations, or inclusion of irrelevant details may not
substantially affect metric scores but can have significant clinical consequences.
Three authors (FJ, EG, and NS; only one of those authors participated in the UCLH guideline data creation) independently
reviewed the outputs while having access to the original guideline PDF documents used as source material. Importantly, the
evaluators did not have access to the reference answers used in the automatic evaluation. They assessed each system output
directly against the guideline documents.
The evaluators scored answers along two dimensions. The first criterion wascorrectness, indicating whether the information
provided was relevant to the clinical question (categories:very relevant,relevant,irrelevant). The second criterion was
completeness, indicating whether the answer adequately covered the relevant guideline content (categories:minor omission,
major omission,just right,minor addition,major addition). Major omissions are particularly important in clinical contexts
because missing key recommendations may lead to incomplete or unsafe guidance.
Results
Baseline Index ConstructionApplying the baseline indexing procedure (see Methods) across the three data sources yields
corpora reflecting differences in each guideline style.
For the UCLH dataset, the indexing process produced 987 sections. Sections are relatively short (mean 31 words), guidelines
are decomposed into an average of 41 sections (see Table 4). In the Somerset NHS Trust dataset, the index contains 469
sections across 6 guidelines (mean 78 sections per guideline). Sections are substantially longer than in UCLH (mean 66 words),
reflecting a more narrative style with multiple clinical considerations rather than concise decision rules. The NICE dataset is
substantially larger and more heterogeneous, comprising 4,726 sections across 10 guidelines (mean 473 sections per guideline).
Sections are shorter on average (mean 19 words). Across all datasets, representative questions remain short and consistent ( ≈8
words). Examples are shown in Table A3. These questions capture key clinical core reasoning patterns, including diagnostic
criteria, escalation decisions, and safety-critical alerts.
7/22

Table 6.Embedding similarity statistics for baseline retrieval. Values represent cosine similarity between the query and the
top-ranked guideline section. We compare representations based on section text, section topic and title (text only) and
representations based on section text, section topic, title and representative LLM-generated question (text + question). Results
show overall high similarity across models, withnomicachieving the highest mean similarity and lowest variance, and
gemmashowing lower similarity. Best values per representation are shown inbold.
Representation Model Mean Std Max
Text only
nomic0.76 0.050.86
qwen3 0.75 0.060.88
gemma 0.65 0.06 0.83
Text + question
nomic0.75 0.04 0.84
qwen3 0.72 0.060.84
gemma 0.68 0.07 0.82
Table 7.Performance comparison on the UCLH dataset across prompting-based extraction strategies and model families.
Metrics include ROUGE-L (R-L, sentence-level recall), word-level precision (P), recall (R), and F1, as well as medical term
precision, recall, and F1. Answer length (AnsLen) is reported as mean word count (lower is better). Statistical significance is
assessed via paired bootstrap resampling (10,000 draws with replacement), with 95% confidence intervals and two-sided
p-values computed from the bootstrap distribution; significance (*) indicates p<0.05 relative to the COPY baseline. LINES
consistently achieves the highest recall among prompting methods, approaching the coverage of baseline retrieval while
producing substantially shorter outputs. Bold highlights the top-performing methods per metric, second-best performance is
underlined.
Method AnsLen R-L P R F1 Term P Term R Term F1
GPT-OSS-120B
copy44.96 0.7537 0.78840.82760.7667 0.79790.8485 0.7807
lines 109.49 0.6231* 0.53800.9406*0.6253* 0.56310.9485*0.6540*
junior 118.00 0.4983* 0.4205 0.8745 0.5085* 0.4414 0.8753 0.5298*
safety 45.05 0.7082* 0.7371 0.8014 0.7169* 0.7424 0.8174 0.7295*
Qwen2.5-32B-Instruct
copy 39.75 0.7717 0.81990.8054 0.7762 0.82300.8275 0.7871
lines 93.91 0.6800* 0.63490.9040*0.6820* 0.65840.9128*0.7036*
junior 146.33 0.4942* 0.4415 0.8317 0.5052* 0.4692 0.8354 0.5335*
safety39.440.6794* 0.7163 0.7550* 0.6891* 0.7341 0.7676 0.7098*
Baseline retrieval limitations.Our BASE retrieval strategy provides an upper bound for content coverage, consistently
achieving high recall across all configurations (word-level recall=0.65–0.86; Table 5). This indicates that relevant guideline
content is retrieved within the selected sections. However, precision remains substantially lower (word precision =0.12 –0.24),
reflecting that retrieved sections frequently include non-essential content.
BASE produces long outputs (mean 174–257 words), as entire guideline sections are returned rather than selectively
extracted evidence. This reduces usability of those answers in time-critical clinical settings.
Across embedding strategies, qwen3 achieves the strongest performance overall. In both representations, it yields the
highest ROUGE-L and F1 scores, while also producing the shortest outputs among BASE variants (174 words for text-only).
Augmenting section representations with LLM-generated questions (text + question) produces mixed effects across
embeddings. In particular, it is beneficial for qwen3 andnomic , but decreases performance for gemma . Also, text + question
representations tend to slightly increase output length.
All models achieve relatively high cosine similarity (mean 0.65–0.76, Table 6), confirming that retrieval identifies semanti-
cally relevant sections. gemma exhibits lower mean similarity and higher variance, consistent with its weaker downstream
retrieval performance.
Prompting-based extraction strategies.Across all datasets and model families, prompting-based extraction substantially
improves over baseline retrieval, not by retrieving more information, but by selecting it more precisely. Among these approaches
for UCLH, LINES consistently emerges as the most reliable strategy for preserving clinically relevant content (Table 7).
Unlike COPY, which serves as a baseline for verbatim extraction, LINES explicitly enforces line-level selection of
8/22

Table 8.Automatic evaluation of prompting strategies for GPT-OSS-120B with full Somerset NHS guideline context. Metrics
include ROUGE-L (R-L, sentence-level recall), word-level precision (P), recall (R), and F1, as well as medical term precision,
recall, and F1. Answer length (Len) is reported as mean word count (lower is better). Statistical significance is assessed via
paired bootstrap resampling (10,000 draws with replacement), with 95% confidence intervals and two-sided p-values computed
from the bootstrap distribution; significance (*) indicatesp<0.05 relative to the COPY baseline. Results highlight a clear
precision–recall trade-off. JUNIORachieves the highest recall. Bold indicates the best-performing method per metric, and
second-best values are underlined.
Method Len R-L P R F1 Term P Term R Term F1
copy 41.30 0.4051 0.41570.52650.4287 0.43320.51880.4432
lines 147.42 0.2695* 0.2311 0.5690 0.2892* 0.2944 0.5541 0.3430
junior 222.92 0.1527* 0.11700.6386* 0.1670* 0.14750.6220* 0.2121*
safety34.860.3922 0.4117 0.4927 0.4170 0.4273 0.4967 0.4296
Figure 4.Clinician evaluation of LINES outputs. Distribution of relevance (left) and completeness (right) ratings across 96
clinical questions. LINES achieves high clinical relevance (95.8% at least relevant) and the highest proportion of “just right”
answers (62.5%), indicating strong alignment with reference guidance. No major omissions were observed, suggesting that
LINES reliably preserves critical clinical content.
evidence. This constraint proves critical. Across settings, LINES achieves the highest recall across both word-level and
entity-level metrics (e.g., word recall 0.94 and NER recall 0.95 for GPT-OSS; 0.90 and0.91 for Qwen), with statistically
significant improvements over COPY. In safety-critical settings, where missing information may have direct consequences,
this recall advantage is decisive. LINES therefore provides the most effective balance of completeness and usability.
COPY provides a strong precision-oriented baseline. SAFETYbehaves very similarly to COPY, achieving precision close
to COPY across settings, while producing outputs of similar length. JUNIORimproves recall relative to other methods, but
consistently fall short of LINES while producing longer and less precise responses.
Extraction behaviour analysis.Table A4 provides qualitative insight into how different strategies work. For example, for
Pericarditis Q10, the reference answer specifies three key elements: chest radiograph, transthoracic echocardiogram, and
cardiology input. Both LINES and COPY recover all elements, while Safety keeps only the mimimum subset.
For Q11 Pericarditis, LINES most closely matches the reference answer with all key elements. COPY also achieves high
coverage but introduces minor omission of the cardiology follow-up. In contrast, the Safety strategy produces a more concise
output, prioritising core treatment steps but omits important supporting details, notably PPI co-prescription, GI side-effect
management, and cardiology follow-up. These trends are consistent across both GPT-OSS and Qwen models.
Qualitative analysis of JuniorDoc (Table A5) shows that Agent-2 effectively consolidates key guideline elements. However,
additions of non-essential contextual information and omission of relevant details are frequent. For example, in AF, Agent-2
captures the core decision rule for cardioversion (onset <48 h with no precipitant), but introduces additional context (broader
rhythm control pathways).
Performance of extraction strategies for longer documents and frontier models.Overall, the performance of our
extraction strategies depends strongly on document length and retrieval setting. When applied to full, long-form guidelines
9/22

Table 9.Automatic evaluation of prompting strategies across model families on full NICE guidelines. Metrics include
ROUGE-L (R-L, sentence-level recall), word-level precision (P), recall (R), and F1, as well as medical term precision, recall,
and F1. Answer length (Len) is reported as mean word count (lower is better). Statistical significance is assessed via paired
bootstrap resampling (10,000 draws with replacement), with 95% confidence intervals and two-sided p-values computed from
the bootstrap distribution; significance (*) indicatesp<0.05 relative to the COPY baseline. In this setting with very long
guidelines (over 5,000 words), JUNIORconsistently achieves the highest recall across models (e.g., word recall up to∼0.94),
with statistically significant improvements over COPY, while maintaining shorter outputs than LINES. Bold indicates the
best-performing method per metric, and second-best values are underlined.
Method Len R-L P R F1 Term P Term R Term F1
DeepSeek-V3
copy65.50 0.6360 0.64110.86340.6944 0.68440.90440.7419
lines 222.76 0.3446* 0.2776 0.8883 0.3735* 0.3641 0.9073 0.4700*
junior 147.19 0.3443* 0.24700.8963* 0.3609* 0.33990.9597* 0.4703*
safety 77.73 0.4979 * 0.4207 0.8578 0.5234 * 0.4856 0.9128 0.5990 *
GPT-5-mini
copy68.66 0.5964 0.58560.89110.6558 0.62800.91840.7027
lines 440.75 0.2111* 0.14720.9433* 0.2279* 0.2107 0.9435* 0.3110*
junior 193.01 0.3199* 0.2317 0.9155 * 0.3340* 0.27950.9718* 0.4411*
safety 91.16 0.4651 * 0.3682 0.8774 0.4848 * 0.4668 0.9615 * 0.5932 *
Qwen3-235B
copy46.34 0.6296 0.65040.82500.6938 0.68320.8467 0.7285
lines 225.03 0.3442* 0.2919 0.8524 0.3850* 0.3435 0.8515 0.4412*
junior 96.14 0.4840 * 0.4187 0.9080* 0.5228 * 0.4997 *0.9457* 0.6115 *
safety 65.88 0.4976* 0.4593 0.8029 0.5438* 0.5021 0.8273 0.5937*
(Somerset and NICE), the JuniorDoc multi-agent approach becomes increasingly competitive, and in most of the cases achieves
the highest recall across models (Tables 8 and 9). For example, on full NICE guidelines (often exceeding 5,000 words),
JuniorDoc consistently achieves the strongest word-level and medical term recall across DeepSeek, GPT-5-mini, and Qwen3,
with statistically significant improvements over COPY.
This behaviour may be attributed to the prompt pipeline: agent 1 generates a response draft aligned with clinicians’ ideas
about the problem, while agent 2 retrieves supporting guideline evidence. In long documents, this enables the model to pinpoint
evidence that would be difficult to capture in this long context. As a result, JuniorDoc is well-suited to large guidelines such as
Somerset (∼3,000 words) and NICE (∼5,000 words).
The pattern reverses when retrieval is constrained. When guideline sections are pre-selected (e.g., via cumulative similarity
mass thresholding), LINES consistently achieves the highest recall across models while maintaining moderate answer length
(Tables A6 and A7). In this setting, the search space is reduced to a focused subset of relevant content (e.g., for NICE from
∼5,000 words per full guideline to∼800 words per selected sections).
Human EvaluationFollowing strong automatic performance, we conducted a detailed human evaluation of model outputs
generated by the GPT-OSS-120B model across 96 UCLH clinical questions, comparing both the LINES and Safety strategies
(Figures 4 and 5). The Safety strategy was additionally included as a controlled contrast in output length. This contrast
provides a useful lens for assessing whether metric gains reflect true clinical quality. Clinical review was performed by three
clinicians (FJ, EG, NS), assessing both relevance and completeness. Relevance was rated asirrelevant,relevant, orvery
relevant, reflecting appropriateness to the clinical scenario. Completeness captured whether clinically important information
was omitted or whether non-essential content was included.
Relevance.Both strategies produced highly clinically appropriate outputs. For LINES, 69.8% of responses were rated as
very relevantand 26.0% asrelevant, with only 4.2% classified asirrelevant(4 answers). Safety achieved even higher top-tier
relevance, with 85.4% of outputs rated asvery relevantand 8.3% asrelevant, though a slightly higher proportion (6.2%) were
judgedirrelevant.
Completeness.Differences appear more clearly in completeness. LINES produced the highest proportion ofjust right
answers (62.5%), with relatively few omissions (8.3% minor, 0% major), but more frequent additions (22.9% minor, 6.2%
major), reflecting its tendency to include additional contextual detail. In contrast, Safety produced fewer additions (12.5%
minor, 1.0% major) and a lower proportion of excessive content overall, but at the cost of increased omissions (18.8% minor,
16.7% major). Only 51.0% of Safety outputs were rated asjust right.
10/22

Figure 5.Clinician evaluation of Safety outputs. Distribution of relevance (left) and completeness (right) ratings across 96
clinical questions. Safety produces highly relevant outputs (93.7% at least relevant), with a higher proportion of “very relevant”
ratings compared to LINES, reflecting concise and focused responses. The strategy presents increased omissions (35.5%,
including 16.7% major omissions).
Overall, LINES prioritises completeness and coverage, rarely omitting critical information but sometimes including non-
essential content. Safety enforces brevity and reduces over-generation, but more frequently omits clinically relevant details,
including in some cases information that may be important for safe decision-making.
Illustrative examples.The pancreatitis example (see Table A8) demonstrates a minor addition for LINES. The response
appropriately prioritises urgent referral to ICU outreach, including clear clinical criteria to support severity assessment and
escalation decisions. It also provides relevant contact pathways, including the general medical registrar, ICU outreach services,
hepatobiliary specialists, and dietetics, all of which are appropriate for ongoing care. The inclusion of additional sections on
ongoing management and imaging is not directly required to answer the referral question. However, this information remains
clinically relevant and does not detract from decision-making.
The ureteric colic example illustrates a minor omission. The response provides a clear and clinically useful set of discharge
criteria that are easy to interpret and apply. One element of the guideline, relating to stone size and anatomical location, is
not explicitly included and would require reference to an associated table. This omission does not significantly affect clinical
decision-making, as it can be readily clarified if needed and does not compromise patient safety.
Inter-annotator agreement.Inter-annotator agreement was assessed on a subset of 12 shared questions (Pericarditis, Acute
Kidney Injury, and Atrial Fibrillation). For relevance, unanimous agreement was observed in 3 cases, with the remaining 9
showing agreement between two annotators. Disagreements were limited to adjacent categories (e.g.relevantvsvery relevant).
For completeness, full agreement was observed in 6 cases, with the remaining 6 showing partial agreement. Disagreements
again occurred primarily between neighbouring categories, most commonlyjust rightandaddition–minor. Majority voting was
used to assign final labels.
These findings confirm that the LINES strategy provides a strong mechanism for clinically evidence extraction. The
consistently high relevance (95.8%) and absence of major omissions indicate that LINES effectively preserves critical
safety-relevant information. Minor omissions tend to affect peripheral details, confirming that LINES avoids unsafe under-
specification.
Discussion
Statement of key findingsOur results demonstrate that clinically relevant information can be reliably delivered by an
LLM-based system with relevant prompting. A central finding is that the effectiveness of extraction strategies depends critically
on document length and retrieval setting. For shorter protocol-like guidelines, LINES consistently achieves the highest term
recall across models while maintaining moderate answer length (+10 points increase). In this setting, the reduced search space
enables precise line-level selection.
When operating over long guidelines (e.g., Somerset and NICE), the JUNIORDOCapproach becomes more effective and
often achieves the highest term recall (+7 points on average). This reflects its ability to draft required evidence first and then
more efficiently find it in long documents.
These results highlight a fundamental trade-off between completeness of extractive approaches and conciseness of abstractive
11/22

approaches. While LINES provides the highest information recall for shorter documents, draft-first approaches JUNIORDOC
are better suited to large-context settings. Strategies that re-interpret input text (e.g., COPYand SAFETY) improve brevity but
may omit clinically relevant information, hence raising safety concerns.
Strengths and limitationsThis study introduces an approach to clinical question answering in which LLMs are constrained
to operate as evidence extractors rather than free-text generators. By enforcing reproduction of guideline content, the approach
directly addresses one of the key risks of LLM deployment in healthcare, the risk of hallucination, while enabling transparent
and auditable outputs.
A further strength is the systematic comparison of prompting strategies across different document-length regimes. This
reveals that performance is not solely a function of model capacity, but of the interaction between prompting strategy and input
structure. In particular, the results demonstrate that increasing architectural complexity (e.g. multi-agent pipelines) may not
always be beneficial.
However, several limitations should be mentioned. First, human evaluation was conducted on a relatively small set of
clinical questions, and although designed to reflect realistic scenarios, larger-scale validation is required. Second, the study
focuses on text-based guidelines and does not include common tabular or visual elements (e.g. flowcharts). Extending the
approach to multimodal guideline formats remains an important direction.
Third, evaluation was performed in a simulated setting rather than in real-time clinical workflows. While this enables
controlled comparison, it does not fully capture real-world usage constraints. Finally, although we evaluated several models,
results may further vary with different architectures and context lengths. Nevertheless, the consistency of observed patterns
across models suggests that the findings are primarily driven by prompting strategy rather than model-specific effects.
Implications and next stepsThese findings have important implications for the deployment of LLMs in clinical settings.
First, they suggest that retrieval and extraction should be jointly designed. Systems operating over large contexts may benefit
from draft-based approaches such as JUNIOR, whereas systems operating over shorter contexts can rely on extraction methods
such as LINES.
Second, the results emphasise the need to explicitly manage the precision–recall trade-off. In safety-critical applications,
prioritising recall may be necessary to avoid omission of clinically relevant information, even at the cost of longer outputs.
Future systems may benefit from adaptive strategies that dynamically adjust verbosity based on task complexity.
Future work will focus on real-world deployment, including integration with electronic health record systems and evaluation
in live clinical environments. In addition, interactive capabilities, such as clarifying missing information, may enhance both
completeness and relevance.
Conclusion
This study demonstrates that LLM-based systems can provide fast and reliable to guideline information when constrained to
evidence extraction. Crucially, the optimal extraction strategy depends on the retrieval setting: line number extraction is most
effective protocol-like documents, whereas draft-first approaches are advantageous for long documents.
Beyond eliminating hallucination, our findings emphasise that preserving completeness is essential for patient safety. By
combining extraction, drafting and retrieval techniques, LLMs can move towards becoming trustworthy clinical assistants.
Code Availability
The code used to reproduce the experiments and evaluation pipeline is publicly available at https://github.com/
julia-ive/safeLLM-extraction.
Data Availability
The clinical guideline data used in this study are proprietary and cannot be publicly shared due to confidentiality requirements.
NICE Q&A dataset is publicly available online:https://github.com/julia-ive/guidelines_qa
Competing interests
We have no competing interests to declare.
Funding
Not applicable
12/22

Author Contributions Statement
JI: Conceptualisation, Methodology, Software, Validation, Formal analysis, Investigation, Writing – Final draft preparation,
Writing - Reviewing and Editing. FJ: Conceptualisation, Methodology, Validation, Formal Analysis, Investigation, Writing –
Early draft preparation, Writing - Reviewing and Editing. EG: Conceptualisation, Methodology, Formal Analysis, Investigation,
Writing – Early draft preparation, Writing - Reviewing and Editing. NJ: Conceptualisation, Writing - Reviewing and Editing.
CSH, PB, EC: Writing - Reviewing and Editing. RD: Conceptualisation, Methodology, Writing - Reviewing and Editing. All
authors approved the manuscript.
Appendices
Table A1.Clinical scenario questions for human evaluation.Questions grouped by acute medical topics targeting
decision-critical elements including diagnosis, investigation, treatment, escalation, and discharge safety.
Topic Question
Acute coronary syndromes Should I ever start ACS treatment before the troponin result is available?
Acute coronary syndromes If I have already sent a troponin sample, when do I send a second one?
Acute coronary syndromes Do I give supplementary oxygen to all ACS patients?
Acute coronary syndromes My patient was not on any anticoagulation preadmission. How long do they need to stay on
fondaparinux?
Acute diarrhea What key discharge advice should I give to outpatients managing acute diarrhoea at home?
Acute diarrhea In an inpatient who develops new diarrhoea (Bristol 5+), what tests and actions are required?
Acute diarrhea Is it safe to use anti-motility agents (e.g. loperamide) or opiates in acute diarrhoea?
Acute diarrhea How do I decide between oral and IV rehydration?
Acute kidney injury Which nephrotoxic medications must I stop or adjust immediately?
Acute kidney injury What is a ‘full renal screen’?
Acute kidney injury When should I involve ICU or renal specialist for possible RRT?
Acute kidney injury How do I manage pulmonary oedema caused by AKI?
AF How do I decide if cardioversion is safe based on time of onset?
AF What monitoring and nursing observations are required once therapy is started?
AF Which baseline investigations must I request on initial assessment?
AF How do I risk stratify for bleeding with anticoagulation?
Cardiogenic pulmonary oedema What oxygen saturation targets should I aim for, and when is oxygen indicated?
Cardiogenic pulmonary oedema Which baseline medications should I review or stop immediately?
Cardiogenic pulmonary oedema Which imaging modalities should I arrange, and how urgently?
Cardiogenic pulmonary oedema What is the recommended initial IV diuretic regimen?
COPD What immediate investigations and samples should I request on arrival?
COPD Which patients warrant ID/Micro review?
COPD What criteria mandate early ICU referral?
COPD When should I consider aminophylline?
Decompensated liver cirrhosis What bowel washout regimen should I start for hepatic encephalopathy?
Decompensated liver cirrhosis What VTE prophylaxis is appropriate and what are contraindications?
Decompensated liver cirrhosis What acute hepatic screen should I order?
Decompensated liver cirrhosis Who would I call for urgent ascitic drain insertion?
DKA Once stable, what maintenance fluid regimen is recommended?
DKA How often should I monitor CBG/VBG and electrolytes?
DKA When should I discuss with ICU for escalation?
DKA Is there an order set for insulin prescribing?
GCA Should I refer to rheumatology or ophthalmology?
GCA Should I treat first or discuss first?
GCA What symptoms suggest complicated GCA?
GCA Beyond steroids, what treatment is required?
Hot swollen joint What are the main causes?
Hot swollen joint What investigations are required?
Hot swollen joint How is septic arthritis treated?
Hot swollen joint What is first-line treatment for gout?
Hypercalcaemia What are the main symptoms?
Hypercalcaemia What investigations beyond bloods are required?
Hypercalcaemia At what level is it considered severe?
Hypercalcaemia Which drugs commonly cause it?
Hyperkalaemia What are the severity cut-offs?
Continued on next page
13/22

Topic Question
Hyperkalaemia What is pseudohyperkalaemia?
Hyperkalaemia What are ECG signs?
Hyperkalaemia When is cardiac monitoring required?
HHS What are the diagnostic criteria?
HHS What monitoring is needed?
HHS What if systolic BP is below 90?
HHS How do I know resolution?
Hypocalcaemia What are the symptoms?
Hypocalcaemia What are ECG signs?
Hypocalcaemia What monitoring is required?
Hypocalcaemia What if vitamin D deficient?
Hypokalaemia At what level does it require treatment?
Hypokalaemia When can oral replacement be used?
Hypokalaemia What are ECG signs?
Hypokalaemia When should concentrated potassium be used?
Hyponatraemia When is it considered severe?
Hyponatraemia What investigations are required?
Hyponatraemia How to test for pseudohyponatraemia?
Hyponatraemia How is it treated?
Infective endocarditis What antibiotics should be given?
Infective endocarditis What blood tests are required?
Infective endocarditis What imaging should be ordered?
Infective endocarditis Who should this be discussed with?
SAH How is diagnosis confirmed?
SAH What are the symptoms?
SAH When should LP be performed?
SAH How should CT-confirmed SAH be managed?
Pancreatitis How is diagnosis confirmed?
Pancreatitis What tests should be ordered?
Pancreatitis Who should I refer to?
Pancreatitis When should I escalate to ICU?
Pericarditis What imaging should I request?
Pericarditis What ECG changes occur?
Pericarditis What is the treatment?
Pericarditis Can the patient be discharged?
Tachycardia What investigations should I perform?
Tachycardia What are differentials for broad complex tachycardia?
Tachycardia How do I escalate unstable tachycardia?
Tachycardia What electrolyte targets should I aim for?
Ureteric colic What tests should I perform?
Ureteric colic What is the treatment?
Ureteric colic When is discharge safe?
Ureteric colic When should I refer to urology?
Upper GI bleed How do I assess severity?
Upper GI bleed When to activate major haemorrhage protocol?
Upper GI bleed What additional blood products are needed?
Upper GI bleed What is the dose of terlipressin?
Table A2.Oncology guideline-based clinical questions.
Guideline Question
Hypersensitivity Irinotecan anticholinergic syndrome
Hypersensitivity Drugs given in grade 2 hypersensitivity reactions
Hypersensitivity How to restart chemotherapy after a reaction
Hypersensitivity Oxaliplatin hypersensitivity reaction
Diarrhea (SACT) Are there contraindications to DOAC?
Diarrhea (SACT) Can DOAC be used in severe renal impairment?
Diarrhea (SACT) Do I need monitoring on DOAC?
Diarrhea (SACT) How long to treat cancer-associated VTE?
Diarrhea (SACT) What if intolerant to DOACs?
Diarrhea (SACT) Anticoagulation in BMI 135 kg
Diarrhea (SACT) PICC-associated thrombus management
14/22

Guideline Question
Diarrhea (SACT) When to offer IVC filter
Diarrhea (SACT) Apixaban dosing in cancer VTE
Diarrhea (SACT) VTE with low platelets
Diarrhea (SACT) Octreotide administration
Diarrhea (SACT) Immunotherapy-related diarrhea management
Diarrhea (SACT) Capecitabine overdose management
Diarrhea (SACT) Stool cultures before loperamide?
Diarrhea (SACT) Treatment for severe diarrhea
Diarrhea (SACT) Investigations for SACT diarrhea
MSCC Can patient be mobilised?
MSCC How to contact coordinator?
MSCC Management after MRI confirmation
MSCC Discharge if MRI negative?
MSCC LMWH prophylaxis?
MSCC Steroid initiation?
MSCC Clinical signs of MSCC
MSCC Investigations for unknown primary
MSCC First-line investigation
MSCC Who to contact locally
Magnesium disorders Contraindicated drugs
Magnesium disorders Need for ECG
Magnesium disorders Managing oral magnesium side effects
Magnesium disorders Admission threshold Mg 0.45
Magnesium disorders Symptoms
Magnesium disorders Causative drugs
Magnesium disorders Oral supplementation
Magnesium disorders Definition of severe hypomagnesaemia
Magnesium disorders IV monitoring
Magnesium disorders Other electrolytes
Neutropenic sepsis Infection with normal neutrophils?
Neutropenic sepsis Central line use
Neutropenic sepsis Need for chest X-ray
Neutropenic sepsis Diagnostic criteria
Neutropenic sepsis Initial management
Neutropenic sepsis Leg swelling post-chemo
Neutropenic sepsis Antibiotics
Neutropenic sepsis Escalation after failure
Neutropenic sepsis Required investigations
Neutropenic sepsis Neutrophil threshold
Prompt Listings
Listing 1.Baseline Prompt: Section Subpart Extraction
Create smaller clinically coherent subparts from the paragraph and identify relevant
questions.
PARAGRAPH (verbatim):
---
{paragraph}
---
Task:
- Identify 2-6 subparts that can each answer a distinct clinical question.
- Subparts may overlap and may be different lengths.
- For each subpart:
- Provide a short title
- Provide ONE short question (<= 16 words)
- Provide the supporting ’text’ as a verbatim span copied EXACTLY from the paragraph.
Rules:
- Copy paragraph text exactly; do not modify wording.
15/22

- Do NOT invent content not present in the paragraph.
- JSON ONLY. No markdown or commentary.
Return STRICT JSON with this schema:
{
"subparts": [
{
"section_id": "P1-S1",
"title": "string",
"question": "<= 16 words; actionable",
"text": "verbatim span copied EXACTLY from the paragraph"
}
]
}
Topic Selection (TS)
Listing 2.Topic Selection Prompt (TS)
You are selecting the most relevant clinical guideline topic.
USER QUESTION:
{user_question}
AVAILABLE GUIDELINE TOPIC IDs:
{guideline_ids}
INSTRUCTIONS:
1. Select exactly ONE guideline_id from the list above.
2. Choose the most directly relevant topic.
3. If no perfect match exists, select the closest relevant topic.
4. Do NOT invent new IDs.
5. Do NOT output explanations outside JSON.
OUTPUT FORMAT (STRICT JSON ONLY):
{
"guideline_id": "<one of the listed IDs>",
"rationale": "<max 1 short sentence>"
}
Line Selection Strategy (LINES)
Listing 3.Line Selection Prompt (LINES)
You will answer the user’s question by selecting the most relevant
lines from the guideline.
GOAL:
- Select the MAXIMUM number of guideline lines that are relevant.
- When unsure whether a line is relevant, INCLUDE it rather than omit it.
GUIDELINE (with line numbers):
{numbered_guideline_text}
User question:
{user_question}
Instructions:
- At the VERY END of your response, output ONLY line numbers
16/22

(comma-separated or ranges like 12-18).
- Do NOT include any other text after the final line numbers.
Verbatim Sentence Extraction (COPY)
Listing 4.Verbatim Sentence Extraction Prompt (COPY)
Copy the most relevant lines from the text to answer the user’s question.
GUIDELINE:
{guideline_sentences}
User question:
{user_question}
Rules:
- Copy ONLY from the provided TEXT.
- Copy verbatim (exact wording).
- Prefer the minimum set of lines that fully answers the question.
- Return JSON only.
Schema:
{"copied": "verbatim lines copied from the provided text"}
JuniorDoc Strategy
Agent 1: Draft Clinical Answer
Listing 5.JuniorDoc Prompt: Agent 1 (Draft Clinical Answer)
You are Agent 1, a junior doctor answering from memory.
Provide your best approximate answer WITHOUT reading
any external documents.
Be concise and clinic-facing.
Extract actionable details whenever remembered
(e.g. thresholds, doses, durations, monitoring steps).
Agent 2: Guideline Evidence Extraction
Listing 6.JuniorDoc Prompt: Agent 2 (Guideline Evidence Extraction)
You are Agent 2.
Inputs:
- user question
- guideline text
- Agent1 answer (for comparison only)
Task:
Select the MAXIMUM subset of FULL SENTENCES copied
VERBATIM from the guideline that either:
- SUPPORT Agent1
- ADD detail to Agent1
- CORRECT Agent1
Rules:
- Copy only from guideline sentences
17/22

Table A3.Representative questions across datasets.
Dataset Topic Representative Question
UCLHAtrial Fibrillation When should urgent DC cardioversion be considered in atrial
fibrillation?
Hyperkalaemia When is urgent assessment required for hyperkalaemia?
Diabetic Ketoacidosis Does the patient meet DKA diagnostic criteria based on glucose,
ketones, and pH?
SomersetNeutropenic Sepsis What clinical features and risk factors should be assessed when
managing suspected neutropenic sepsis?
Chemotherapy Toxicity How should chemotherapy-induced toxicity be graded and man-
aged in oncology patients?
Cancer Pain What factors should guide escalation of analgesia in patients
with cancer-related pain?
NICEAcute Coronary Syndrome What investigations, risk stratification, and initial management
steps are recommended for suspected acute coronary syndrome?
Chronic Kidney Disease How should CKD be diagnosed, staged, and initially managed
based on eGFR and albuminuria?
Stroke What criteria determine urgent imaging and thrombolysis eligi-
bility in suspected acute stroke?
- Preserve order
- Return JSON only
{"subset": ["sentence", ...]}
Minimal Safe Extraction (Safety)
Listing 7.Minimal Safe Extraction Prompt (Safety)
Select the SMALLEST ordered subset of FULL sentences
from the guideline that answers the question.
Each sentence must have one label:
answers_question
adds_required_detail
safety_critical
Rules:
- Copy only from guideline sentences
- Preserve order
- No paraphrasing
- Return JSON
{
"sentences": [
{"text": "sentence", "reason": "answers_question"}
]
}
References
1.Guerra-Farfan, E.et al.Clinical practice guidelines: The good, the bad, and the ugly.Injury54, S26–S29 (2023).
2.Scraggs, E.et al.Factors that encourage or discourage doctors from acting in accordance with good practice final report
prepared for the general medical council. Tech. Rep. (2012).
18/22

Table A4.Examples of extraction behaviour across prompting strategies (UCLH, gpt-oss:120B). The table compares reference
answers with outputs generated by the LINES, COPY , and Safety strategies. LINES reconstructs answers from selected
guideline lines and consistently achieves high coverage of reference content. COPY copies verbatim sentences from the
guideline. Safety selects the minimal subset of sentences required to answer the question, produces concise responses but may
omit important details. Original text is paraphrased.
Question ID Question Reference answer COPY output LINES output Safety output
Q10_Pericarditis What imaging should I re-
quest if suspecting pericardi-
tis?• Chest radiograph
• Transthoracic echocar-
diogram to evaluate for
pericardial effusion or
tamponade
• Seek cardiology input
if a cardiac MRI is being
consideredREQUEST
• Chest radiograph
• Transthoracic echocar-
diogram to evaluate for
pericardial effusion or
tamponade
• Seek cardiology input
if a cardiac MRI is being
considered• Chest radiograph
• Transthoracic echocar-
diogram to evaluate for
pericardial effusion or
tamponade
• Seek cardiology input
if a cardiac MRI is being
considered• Chest radiograph
• Transthoracic echocar-
diogram to evaluate for
pericardial effusion or
tamponade
Q11_Pericarditis What is the treatment for
pericarditis?• Ibuprofen 400 mg TDS
7–10 days, then taper
• PPI (e.g., lansoprazole
30 mg OD)
• Colchicine 500 µg BD
(or OD if <75 kg) for 3
months
• Adjust for GI side ef-
fects
• Avoid routine steroids
• Limit physical activity
• Cardiology follow-up• Ibuprofen 400 mg TDS
7–10 days, then taper
• PPI (e.g., lansoprazole
30 mg OD)
• Colchicine 500 µg BD
(or OD if <75 kg) for 3
months
• Adjust for GI side ef-
fects
• Avoid routine steroids
• Limit physical activity• Ibuprofen 400 mg TDS
7–10 days, then taper
• PPI (e.g., lansoprazole
30 mg OD)
• Colchicine 500 µg BD
(or OD if <75 kg) for 3
months
• Adjust for GI side ef-
fects
• Avoid routine steroids
• Limit physical activity
• Cardiology follow-up• Ibuprofen 400 mg TDS
7–10 days
• Colchicine 500 µg BD
(or OD if <75 kg)
• Avoid routine steroids
• Limit physical activity
3.Kresevic, S.et al.Optimization of hepatological clinical guidelines interpretation by large language models: a retrieval
augmented generation-based framework.NPJ Digit. Med.7, 102 (2024).
4.Dean, N. C.et al.Decreased mortality after implementation of a treatment guideline for community-acquired pneumonia.
The Am. journal medicine110, 451–457 (2001).
5.VanWagner, L. B.et al.Blood pressure control according to clinical practice guidelines is associated with decreased
mortality and cardiovascular events among liver transplant recipients.Am. J. Transplantation20, 797–807 (2020).
6.Arnold, F. W.et al.Improving outcomes in elderly patients with community-acquired pneumonia by adhering to national
guidelines: Community-acquired pneumonia organization international cohort study results.Arch. internal medicine169,
1515–1524 (2009).
7.Monti, G.et al.Effect of a quality improvement program on compliance to the sepsis bundle in non-icu patients: a
multicenter prospective before and after cohort study.Front. Medicine10, 1215341 (2023).
8.Coba, V .et al.Resuscitation bundle compliance in severe sepsis and septic shock: improves survival, is better late than
never.J. intensive care medicine26, 304–313 (2011).
9.Qumseya, B.et al.Barriers to clinical practice guideline implementation among physicians: a physician survey.Int. J.
Gen. Medicine7591–7598 (2021).
10.Tsiga, E., Panagopoulou, E., Sevdalis, N., Montgomery, A. & Benos, A. The influence of time pressure on adherence to
guidelines in primary care: an experimental study.BMJ open3, e002700 (2013).
11.Freedman, S., Golberstein, E., Huang, T.-Y ., Satin, D. J. & Smith, L. B. Docs with their eyes on the clock? the effect of
time pressures on primary care productivity.J. health economics77, 102442 (2021).
12.Van der Vegt, A., Zuccon, G., Koopman, B. & Deacon, A. How searching under time pressure impacts clinical decision
making.J. Med. Libr. Assoc. JMLA108, 564 (2020).
13.Torjesen, I. Doctors are under more work pressure than during height of covid-19 pandemic in 2020.BMJ: Br. Med. J.
(Online)375(2021).
14.Daniels, J., Robinson, E., Jenkinson, E. & Carlton, E. Perceived barriers and opportunities to improve working conditions
and staff retention in emergency departments: a qualitative study.Emerg. Medicine J.41, 257–265 (2024).
19/22

Table A5.JuniorDoc pipeline examples (UCLH, gpt-oss:120B). The table illustrates how intermediate steps transform the
original uninformed LLM answer. Comparison with reference answers highlights presence of non-essential content. Original
text is paraphrased.
Question ID Question Agent-1 answer Agent-2 sentences Reference answer
Pericarditis What imaging should I re-
quest if suspecting peri-
carditis?Echocardiogram is primary
imaging. Request TTE.
Consider bedside echo if un-
stable. Follow with formal
echo if needed.REQUEST Imaging: Chest
radiograph. Transthoracic
echocardiogram to evaluate for
pericardial effusion or tampon-
ade. Seek cardiology input if
cardiac MRI considered.Chest radiograph; Transtho-
racic echocardiogram to
evaluate for pericardial ef-
fusion or tamponade; Seek
cardiology input if a cardiac
MRI is being considered
AF How do I decide if car-
dioversion is safe based
on time of onset?Depends on AF duration:
<48h safe without antico-
agulation; 48h–1 week con-
sider TEE; >1 week antico-
agulate 3 weeks before car-
dioversionDIAGNOSIS: onset <48h.
RHYTHM CONTROL: car-
dioversion if within 48h and
no precipitant. If >48h: rate
control + anticoagulation then
elective cardioversion.Consider cardioversion if
onset <48h and no clear pre-
cipitant
Table A6.Automatic evaluation of prompting strategies for GPT-OSS-120B with Somerset NHS guideline sections selected
up to a cumulative similarity mass threshold (0.95) using the text+question representations. Under this setting, LINES achieves
consistently higher recall than other prompting methods (e.g., word recall≈0.65), with statistically significant improvements
over COPY, while maintaining moderate answer length. Multi-agent (JuniorDoc) further increases recall marginally but at the
cost of longer outputs. Bold indicates the best-performing method per metric, and second-best values are underlined.
Method Len R-L P R F1 Term P Term R Term F1
copy 42.50 0.4321 0.50430.51860.4642 0.50340.51250.4706
lines 117.28 0.3406* 0.3006 0.6539 * 0.3639* 0.3755*0.6291* 0.4191*
junior 142.00 0.2549* 0.21330.6543* 0.2828* 0.2567* 0.6372 * 0.3228*
safety38.620.3774 * 0.4661 0.4596* 0.4089 * 0.4786 0.4742 0.4274 *
15.Kann, B. H., Johnson, S. B., Aerts, H. J., Mak, R. H. & Nguyen, P. L. Changes in length and complexity of clinical practice
guidelines in oncology, 1996-2019.JAMA Netw. Open3, e200841–e200841 (2020).
16.Singhal, K.et al.Toward expert-level medical question answering with large language models.Nat. Med.31, 943–950
(2025).
17.Pal, A., Umapathi, L. K. & Sankarasubbu, M. Med-HALT: Medical domain hallucination test for large language models.
InProceedings of the 27th Conference on Computational Natural Language Learning (CoNLL), 314–334 (Association for
Computational Linguistics, Stroudsburg, PA, USA, 2023).
18.Lewis, P.et al.Retrieval-augmented generation for knowledge-intensive NLP tasks.arXiv [cs.CL](2020).
19.Ng, K. K. Y ., Matsuba, I. & Zhang, P. C. RAG in health care: A novel framework for improving communication and
decision-making by addressing LLM limitations.NEJM AI2(2025).
20.Lv, A., Xie, R., Sun, X., Kang, Z. & Yan, R. Language models “grok” to copy.arXiv [cs.CL](2024).
21.Chen, Q.et al.Benchmarking large language models for biomedical natural language processing applications and
recommendations.Nat. Commun.16, 3280 (2025).
22.Lewis, M.et al.Grounding large language models in clinical evidence: A retrieval-augmented generation system for
querying UK NICE clinical guidelines.arXiv [cs.CL](2025).
23.Ding, Q., Zhang, E. H. Q., Jozsa, F. & Ive, J. Building a silver-standard dataset from NICE guidelines for clinical LLMs.
arXiv [cs.CL](2025).
24.OpenAIet al.gpt-oss-120b & gpt-oss-20b model card.arXiv [cs.CL](2025).
25.Yang, A.et al.Qwen3 technical report.arXiv [cs.CL](2025).
26.Gemma Teamet al.Gemma: Open models based on gemini research and technology.arXiv [cs.CL](2024).
27.Lin, C.-Y . Rouge: A package for automatic evaluation of summaries. InText Summarization Branches Out(2004).
20/22

Table A7.Automatic evaluation of prompting strategies across model families with retrieval constrained to NICE guideline
sections selected up to a cumulative similarity mass threshold (0.95). Metrics include ROUGE-L (R-L, sentence-level recall),
word-level precision (P), recall (R), and F1, as well as medical term precision, recall, and F1. Answer length (Len) is reported
as mean word count (lower is better). Statistical significance is assessed via paired bootstrap resampling (10,000 draws with
replacement), with 95% confidence intervals and two-sided p-values computed from the bootstrap distribution; significance (*)
indicatesp<0.05 relative to the COPY baseline. Constraining retrieval substantially reduces input context (from∼5,000
words to ∼835 words). LINES consistently achieves the highest recall among prompting methods (e.g., word recall of ∼0.71
across models), with statistically significant improvements over COPY. Bold indicates the best-performing method per metric,
and second-best values are underlined.
Method Len R-L P R F1 Term P Term R Term F1
DeepSeek-V3
copy 65.50 0.6360 0.64110.86340.6944 0.65910.59120.5857
lines 222.76 0.3446* 0.2776 0.8883 0.3735* 0.47560.7293* 0.5168*
junior 147.19 0.3443* 0.24700.8963* 0.3609* 0.4910 0.6655 * 0.5084*
safety77.730.4979 * 0.4207 0.8578 0.5234 * 0.5453 * 0.6370* 0.5407 *
GPT-5-mini
copy 68.66 0.5964 0.58560.89110.6558 0.60620.66040.5819
lines 440.75 0.2111* 0.14720.9433* 0.2279* 0.45840.7209* 0.4974*
junior 193.01 0.3199* 0.2317 0.9155 * 0.3340* 0.4729 0.6958 * 0.5054*
safety91.160.4651 * 0.3682 0.8774 0.4848 * 0.5703 0.6329 0.5520
Qwen3-235B
copy 46.34 0.6296 0.65040.82500.6938 0.63950.63200.5967
lines 225.03 0.3442* 0.2919 0.8524 0.3850* 0.43970.7222* 0.4878*
junior 96.14 0.4840 * 0.4187 0.9080* 0.5228* 0.5381* 0.6747 * 0.5446 *
safety65.880.4976* 0.4593 0.8029 0.5438 * 0.5906 0.5919 0.5432*
28.Kraljevic, Z.et al.Multi-domain clinical natural language processing with MedCAT: The medical concept annotation
toolkit.Artif. Intell. Med.117, 102083, DOI: 10.1016/j.artmed.2021.102083 (2021).
21/22

Table A8.Examples illustrating minor addition and minor omission in LLM outputs, with clinical interpretation. All examples
are paraphrased.
Type Topic / Question LLM Answer (verbatim)
Minor addition Topic:Pancreatitis
Question:I have a patient with pancre-
atitis, who can I refer them to?“ICU REFERRAL Urgent discussion with ICU outreach SpR (via mobile)
•Ongoing clinical concern or deterioration
•Airway not maintained (call anaesthetic SpR)
•GCS≤12
•Respiratory distress/failure
•Haemodynamic instability not responding to fluids
•Severe electrolyte / acid-base disturbance
•All pancreatitis meeting ‘severe’ Atlanta criteria
Advice, reviews and referrals including DMR, ICU outreach, PERRT, anaesthetics,
HPB and dietetics. Further sections include ongoing management and imaging
guidance. ”
Minor omission Topic:Ureteric colic
Question:When is it safe to discharge a
patient with ureteric colic?“DISPOSITION Consider discharge if ALL of the following are met:
•No evidence of sepsis
•No acute kidney injury
•Pain controlled
•Stone likely to pass spontaneously
•No alternative serious pathology
Discharged patients: refer to Endourology and Stone clinic via EPIC. ”
22/22
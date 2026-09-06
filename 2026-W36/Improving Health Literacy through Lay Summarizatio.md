# Improving Health Literacy through Lay Summarization of Radiological Reports: An Evaluation of BioNER and Retrieval-Augmented Generation

**Authors**: Egecan Çelik Evgin, İlknur Karadeniz, Olcay Taner Yıldız

**Published**: 2026-09-02 10:06:13

**PDF URL**: [https://arxiv.org/pdf/2609.02396v1](https://arxiv.org/pdf/2609.02396v1)

## Abstract
Radiology reports are written primarily for clinicians, and their specialized terminology often makes them difficult for patients to interpret. As a result, many patients turn to publicly available Large Language Models (LLMs) to help explain their reports, despite well-documented risks of factual inaccuracies and hallucinations. Automated lay-summary generation has emerged as a promising alternative, yet the effectiveness of retrieval-enhanced and clinically informed approaches for radiology-specific communication remains underexplored. This study investigates the extent to which Retrieval-Augmented Generation (RAG) and Named Entity Recognition (NER) improve the quality, factual consistency, and readability of automatically generated lay summaries compared with standard LLM-based generation. We develop a framework combining NER-based extraction of clinically relevant findings with a RAG mechanism for contextual grounding, evaluated across few-shot and fine-tuned variants of two models (Qwen, BioBART). Results show that NER consistently improves readability and overall quality, while RAG alone offers no benefit and can introduce hallucinations from irrelevant retrieved terms. Combining RAG with NER degrades performance in few-shot settings but improves readability when fine-tuned. Fine-tuned BioBART with NER achieves the best overall performance, highlighting entity-aware extraction as the primary driver of improved patient-friendly summaries.

## Full Text


<!-- PDF content starts -->

Improving Health Literacy through Lay Summarization of Radiological
Reports: An Evaluation of BioNER and Retrieval-Augmented Generation
Egecan C ¸ elik Evgin1,˙Ilknur Karadeniz3, Olcay Taner Yıldız1,2
1Department of Artificial Intelligence and Data Engineering, ¨Ozye ˘gin University, T ¨urkiye
2Department of Computer Science, ¨Ozye ˘gin University, T ¨urkiye
3Department of Computer Engineering, Galatasaray University, T ¨urkiye
egecan.evgin@ozu.edu.tr,
ikaradeniz@gsu.edu.tr,
olcay.yildiz@ozyegin.edu.tr
Abstract
Radiology reports are written primarily for
clinicians, and their specialized terminology
often makes them difficult for patients to in-
terpret. As a result, many patients turn to
publicly available Large Language Models
(LLMs) to help explain their reports, despite
well-documented risks of factual inaccuracies
and hallucinations. Automated lay-summary
generation has emerged as a promising al-
ternative, yet the effectiveness of retrieval-
enhanced and clinically informed approaches
for radiology-specific communication remains
underexplored. This study investigates the
extent to which Retrieval-Augmented Gener-
ation (RAG) and Named Entity Recognition
(NER) improve the quality, factual consis-
tency, and readability of automatically gen-
erated lay summaries compared with stan-
dard LLM-based generation. We develop a
framework combining NER-based extraction
of clinically relevant findings with a RAG
mechanism for contextual grounding, evalu-
ated across few-shot and fine-tuned variants
of two models (Qwen, BioBART). Results
show that NER consistently improves read-
ability and overall quality, while RAG alone
offers no benefit and can introduce hallucina-
tions from irrelevant retrieved terms. Com-
bining RAG with NER degrades performance
in few-shot settings but improves readability
when fine-tuned. Fine-tuned BioBART with
NER achieves the best overall performance,
highlighting entity-aware extraction as the pri-
mary driver of improved patient-friendly sum-
maries.
1 Introduction
Radiological reports document the findings of
medical imaging examinations, such as X-rays,
computed tomography (CT), and magnetic res-
onance imaging (MRI), and serve as a primary
means of communication between healthcare pro-
fessionals. However, these reports are typicallywritten using specialized biomedical terminology
and complex clinical language, making them dif-
ficult for patients to understand. Consequently,
many patients struggle to interpret their imaging
results and fully comprehend the implications of
the reported findings. To better understand their
medical conditions and make informed decisions
about treatment, patients often seek additional in-
formation. Traditionally, this involved search-
ing online for medical terms and symptoms. To-
day, many patients use Large Language Model
(LLM)-based chatbots, such as ChatGPT, Gemini,
and DeepSeek, to obtain health-related informa-
tion (OpenAI, 2022; Google, 2024; DeepSeek-AI,
2024). However, these systems can generate inac-
curate or hallucinated content, potentially leading
patients to misunderstand their radiological find-
ings or place undue trust in incorrect information.
This communication gap can limit patient un-
derstanding and health literacy, motivating re-
search into methods that translate radiological
reports into patient-friendly language. Recent
work has explored Retrieval-Augmented Genera-
tion (RAG) to improve the quality and factual con-
sistency of lay summarization by grounding gen-
erated outputs in external knowledge (Guo et al.,
2024). In parallel, Named Entity Recognition
(NER) has been used to identify clinically rel-
evant terms that can guide and constrain gen-
eration toward more accurate and relevant con-
tent. However, the comparative effectiveness of
retrieval-based and entity-aware approaches for
radiology-specific lay summarization remains un-
derexplored, particularly across models of differ-
ent scale and training regime.
To address this gap, we evaluate our approach
across four public radiology report datasets span-
ning diverse clinical settings and imaging modal-
ities: PadChest, BIMCV-COVID19+, Open-i, and
MIMIC-CXR (Bustos et al., 2020; de la Igle-
sia Vay ´a et al., 2020; Demner-Fushman et al.,
arXiv:2609.02396v1  [cs.CL]  2 Sep 2026

2012; Johnson et al., 2019).
The main contributions of this study are:
• A framework combining and comparing
RAG-based and NER-enhanced approaches
to radiology report lay summarization;
• A comparison between a state-of-the-art
general-purpose LLM and a biomedical small
language model;
• The use of few-shot baselines to systemati-
cally compare against fine-tuned model vari-
ants.
The remainder of the paper is organized as fol-
lows: Section 2 reviews related work, Section 3
describes the methodology, Section 4 presents the
results and discussion, and Section 5 concludes the
paper.
2 Related Work
Lay summaries differ from standard summaries in
their emphasis on readability for non-expert au-
diences. In the biomedical domain, the BioLay-
Summ shared task has been organized in 2023,
2024, and 2025 (Goldsack et al., 2023, 2024; Xiao
et al., 2025), aiming to generate lay summaries
that are relevant, readable, and factual. BioLay-
Summ 2025 Shared Task 2 focused specifically
on generating lay summaries from radiology re-
ports, and several of the approaches discussed be-
low were developed for this task.
Fine-tuning-based approaches:AEHRC
achieved the best overall performance in both
the open and closed subtasks of Shared Task 2
using fully supervised fine-tuning, comparing T5-
Large with LLaMA-3.2-3B and finding T5-Large
superior, without using LoRA, quantization, or
RAG (Zhang et al., 2025; Raffel et al., 2020;
Meta AI, 2024). KHU LDI, the second-best
open-track system, used QLoRA fine-tuning on
Qwen2.5-3B-Instruct and Qwen3-4B, combined
with 3-shot prompting and a generate-feedback-
refine pipeline (Moriazi and Sung, 2025; Dettmers
et al., 2023; Yang et al., 2024, 2025; Madaan
et al., 2023). MetninOzU ranked third overall
using an abstract-based summarization setup,
showing that shorter inputs can still yield strong
factuality scores (Evgin et al., 2025).
Prompting-based approaches:5cNLP, the
second-place closed-track system, relied on struc-
tured prompting rather than fine-tuning, test-
ing Llama-3.3-70B-Instruct and GPT-4.1; theirbest result used GPT-4.1 with few-shot radiology
examples selected via BERT-large embeddings
(Lossio-Ventura et al., 2025). Proff et al. com-
pared GPT-4o, Llama-3-70B, and Mixtral-8x22B
for radiology report simplification, finding that all
models improved readability, though open-weight
models produced more high-risk errors than GPT-
4o (Proff et al., 2026).
RAG-based approaches:CUTN Bio placed
third in the closed track of Shared Task 2 us-
ing a RAG pipeline with Zephyr-7B-beta, ex-
tracting medical terms via SciSpacy and retriev-
ing Wikipedia definitions stored in ChromaDB
(Sivagnanam et al., 2025; Tunstall et al., 2023;
Neumann et al., 2019; Chroma, 2026). The
same team placed second in Subtask 1.2 (external-
knowledge lay summarization) using a similar
RAG approach with MedCAT for term extraction
and LLaMA-3-8B-Instruct for generation (Kral-
jevic et al., 2021; Meta AI, 2024). Sun et al.
proposed FactMM-RAG, which retrieves factu-
ally similar report content via RadGraph prior
to generation to improve the accuracy of gener-
ated radiology reports (Sun et al., 2025). Lay-
SummX applied retrieval-augmented fine-tuning,
using abstracts to retrieve relevant full-text chunks
before fine-tuning LLaMA 3.1 with LoRA (Lin
and Yu, 2025). Guo et al. introduced Retrieval-
Augmented Lay Language generation, retrieving
UMLS and Wikipedia definitions to supply miss-
ing background explanations (Guo et al., 2024).
UIUC BioNLP used an extract-then-summarize
pipeline combining Wikipedia definition retrieval
with DPR-based passage retrieval (You et al.,
2024).
NER-based approaches:ISIKSumm used a
BART-based system augmented with biomedical
entity labels using Stanza NER to improve han-
dling of technical terms (Colak and Karadeniz,
2023). Gupta and Krishnamurthy’s LayForge sys-
tem used BioBERT NER to identify biomedical
terms and incorporated UMLS definitions before
summary rewriting, improving readability and fac-
tuality at a small cost to ROUGE scores (Gupta
and Krishnamurthy, 2025; Lee et al., 2020; Bo-
denreider, 2004; Lin, 2004). Ming et al. used
MeSH terms to guide LLMs toward more infor-
mative background context for lay readers (Ming
et al., 2025).
Overall, prior work has largely explored RAG-
based and NER-based strategies in isolation, with

few studies directly comparing their effectiveness
within a unified framework, or across models dif-
fering substantially in scale and training regime
(few-shot vs. fine-tuned). This study addresses
this gap by systematically comparing RAG-based
and NER-enhanced lay summarization strategies
for radiology reports.
3 Methodology
3.1 Models
Two models were used in this study: Qwen3.5-
0.8B (Qwen Team, 2026), a recent general-
purpose small language model, and BioBART-
v2-large (Yuan et al., 2022), a model pretrained
specifically on biomedical text. This pairing
enables comparison between a strong general-
purpose model and a smaller model adapted for
the biomedical domain.
3.2 Dataset and Evaluation Set
Four public radiology report datasets, spanning
diverse clinical settings and imaging modalities,
were used in this study: PadChest, BIMCV-
COVID19+, Open-i, and MIMIC-CXR. PadChest
contains over 160,000 chest X-ray images from
approximately 67,000 patients, while BIMCV-
COVID19+ includes COVID-19 X-ray and CT
studies, comprising 21,342 CR, 34,829 DX, and
7,918 CT cases (Bustos et al., 2020; de la Igle-
sia Vay ´a et al., 2020). Open-i is smaller, with
7,470 chest X-ray images and 3,955 reports, while
MIMIC-CXR is the largest dataset, with 227,835
studies and 377,110 images derived from real clin-
ical reports (Demner-Fushman et al., 2012; John-
son et al., 2019). Lay summaries were automati-
cally generated from the clinical reports using the
Layman’s RRG framework (Zhao et al., 2026),
and the combined datasets were used in the Bi-
oLaySumm 2025 shared task (Xiao et al., 2025).
However, the lay summaries for the shared task’s
test set were not publicly available. To address
this, the original training set was split to construct
a new test set, with the goal of obtaining a larger
evaluation set than the one used in the shared task.
The split was performed randomly (seed = 42) to
avoid bias; this is distinct from the sampling of
the three few-shot exemplar reports described in
§3.4, which used the same seed value for a sepa-
rate sampling step. The resulting split comprises
168,036 training reports (89.38%), 14,971 valida-
tion reports (7.96%), 5,000 test reports (2.66%),and 3 few-shot exemplar reports, with average to-
ken counts summarized in Table 1.
Table 1: Dataset split statistics
Split Name Split % Rows Rad. Reports Lay
Training 89.38% 168,036 31.11 42.17
Validation 7.96% 14,971 34.27 45.49
Testing 2.66% 5,000 31.19 42.22
Few-Shot Samples 0.001% 3 11.00 20.33
Rad. Reports: Average tokens in radiological reports;
Lay: Average tokens in layman summaries.
3.3 Evaluation Metrics
Generated summaries were evaluated along three
dimensions: relevance, readability, and factuality.
All metrics were scaled using min-max normaliza-
tion to place their values on a comparable range
(Han et al., 2011). No additional weighting was
applied across the three metric groups, since each
group contained an equal number of metrics. For
relevance and factuality metrics, higher scores in-
dicate better performance; for all readability met-
rics (FKGL, DCRS, SLE), lower scores indicate
better performance (i.e., simpler, more accessible
text).
Relevance:ROUGE (Lin, 2004) measures
word overlap between predicted and gold lay
summaries; we report the average F1 across
ROUGE-1, ROUGE-2, and ROUGE-L. METEOR
(Banerjee and Lavie, 2005) extends beyond ex-
act word matches by accounting for stems and
synonyms, offering a complementary view of rel-
evance. BERTScore (Zhang et al., 2020) mea-
sures semantic similarity by comparing contex-
tual word embeddings between predicted and gold
summaries, computing precision, recall, and F1
based on closest token matches.
Readability:FKGL (Kincaid et al., 1975)
estimates the grade level of a summary based
on sentence and word length, with longer sen-
tences and words yielding higher (less readable)
scores. DCRS (Dale and Chall, 1948) comple-
ments FKGL by assessing word familiarity against
a list of common words, capturing cases FKGL
may miss, such as short but unfamiliar words (e.g.,
”understand”). SLE (Cripwell et al., 2023) is a
transformer-based metric that requires only the
predicted summary, using a RoBERTa-base model
with a regression head to produce a simplicity
score.
Factuality:SummaC (Laban et al., 2022)
evaluates sentence-level agreement between the

source report and the predicted summary using en-
tailment and contradiction scores, penalizing con-
tradictory content. FENICE (Scir `e et al., 2024)
evaluates factuality at the claim level by extract-
ing atomic claims from the predicted summary
and verifying them against the source text, directly
penalizing unsupported or contradicted claims.
CheXbert-F1 (Smit et al., 2020), developed specif-
ically for radiology reports, evaluates whether
clinical findings in the generated summary match
those in the reference report, penalizing missing or
incorrect findings.
3.4 Baseline Strategies
Two baseline strategies were established for each
model: (i) Few-shot prompting. Baselines were
computed under 0-shot, 1-shot, and 3-shot set-
tings. To construct the 1-shot and 3-shot exem-
plars, three radiology reports were randomly sam-
pled (seed = 42) and excluded from the test set
to prevent data leakage (Table 1). (ii) LoRA fine-
tuning.
Both models were fine-tuned using LoRA with
r=4,lora alpha=8, dropout of 0.05, and no
bias term. Qwen was adapted onq projand
vproj; BioBART was adapted onq proj,
vproj, andout proj. Training used 2 epochs,
a batch size of 20 (evaluation batch size 16), gradi-
ent accumulation of 1, learning rate2e-4, weight
decay0.01, 100 warmup steps, withbf16en-
abled andfp16disabled.
3.5 Enhancement Strategies
In addition to the baselines, three enhancement
strategies were evaluated, each applied to both the
few-shot and fine-tuned settings of both models:
• NER-enhanced (BioNER): Clinically
relevant terms were extracted using
Stanza’s radiology NER model (Zhang
et al., 2021), which identifies five entity
classes: ANATOMY , OBSERV ATION,
ANATOMY MODIFIER, OBSERV A-
TION MODIFIER, and UNCERTAINTY .
Extracted entities were used to guide sum-
mary generation toward clinically relevant
content.
• RAG-enhanced: A retrieval-augmented
pipeline was built in which an agent ex-
tracted candidate medical terms from the
source report. Each term was first checkedagainst a local term-description database; if
not found, it was searched via the Wikipedia
API, and the first sentence of the result
was stored in the local database for reuse.
Retrieved definitions were then provided
as contextual grounding during summary
generation.
• Combined (BioNER + RAG): Term extrac-
tion was performed using BioNER, and the
resulting terms were used to query the RAG
retrieval pipeline described above, combining
entity-guided extraction with retrieval-based
grounding.
This produces three conditions per model per
learning setting (baseline, +BioNER, +RAG,
+BioNER+RAG).
4 Results
Table 2 presents the overall results of the few-
shot strategies. For the Qwen model, the BioNER
strategy improved overall relevance, readability,
and factuality compared with the 0-shot baseline.
The RAG strategy did not outperform the baseline,
while the combination of BioNER and RAG re-
sulted in lower scores across all evaluation dimen-
sions. A similar trend was observed for BioBART.
Compared with Qwen, BioBART achieved lower
performance in the few-shot setting across most
evaluation metrics.
Table 3 summarizes the fine-tuning results. In
contrast to the few-shot experiments, BioBART
outperformed Qwen after fine-tuning. Similar to
the few-shot setting, the BioNER strategy im-
proved overall performance, particularly readabil-
ity, for both models. The RAG strategy improved
readability but reduced relevance in both models.
For Qwen, the combined BioNER+RAG strategy
achieved the best FKGL and DCRS scores to-
gether with the lowest SLE score.
Table 4 compares the best-performing few-shot
and fine-tuning strategies for each model. For
Qwen, the best few-shot strategy (0-shot BioNER)
outperformed all fine-tuning configurations. In
contrast, BioBART achieved its highest perfor-
mance after fine-tuning. Comparing the best-
performing configurations of both models, fine-
tuned BioBART with BioNER achieved better
overall performance than Qwen with the 0-shot
BioNER strategy across most evaluation metrics.

Table 2: Mean scores for Few-Shot Based Strategies
Relevance Readability Factuality
Str ROUGE↑METEOR↑BERTScore↑FKGL↓DCRS↓SLE↓SummaC↑FENICE↑CHEX↑Mean↑
Qwen3.5 0-Shot0.357 0.3952 0.9146 6.59 9.8097 1.4909 0.5784 0.3209 0.8918 0.8221
Qwen3.5 1-Shot0.3272 0.3581 0.9077 6.71 9.7976 1.3763 0.6567 0.2032 0.8977 0.7865
Qwen3.5 3-Shot0.3688 0.40290.9146.26 9.30571.55910.74510.40180.90560.9066
Qwen3.5 BioNER 0-Shot0.3589 0.3990.91836.35 9.3841 1.0687 0.60280.41270.8941 0.9092
Qwen3.5 RAG 0-Shot0.3565 0.3898 0.9089 7.1 9.6396 1.3106 0.5855 0.1742 0.889 0.7917
Qwen3.5 BioNER + RAG 0-Shot0.2474 0.3019 0.865 9.61 10.7426 1.8543 0.542 0.0507 0.8141 0.5172
BioBART 0-Shot0.194 0.2807 0.8448 15.39 11.9016 1.045 0.3 0 0.7708 0.3698
BioBART 1-Shot0.0845 0.1298 0.8136 15.2 12.86610.52960.4173 0.3341 0.1074 0.2879
BioBART 3-Shot0.0872 0.1418 0.8137 14.46 13.0713 0.5408 0.5829 0.3498 0.0368 0.3303
BioBART BioNER 0-Shot0.0899 0.1241 0.802 18.39 12.1376 0.6565 0.416 0.3841 0.7531 0.3541
BioBART RAG 0-Shot0.1069 0.1654 0.8186 14.31 11.8315 1.314 0.6266 0.0322 0.7686 0.344
BioBART BioNER + RAG 0-Shot0.0918 0.1351 0.8052 14.66 11.2866 0.9849 0.5033 0.1442 0.778 0.3543
Table 3: Mean scores for Fine-Tuning Based Strategies
Relevance Readability Factuality
Str ROUGE↑METEOR↑BERTScore↑FKGL↓DCRS↓SLE↓SummaC↑FENICE↑CHEX↑Mean↑
Qwen3.5 Fine-Tuning0.4071 0.4583 0.9249 10.58 11.3545 1.15570.67540.4162 0.8918 0.5425
Qwen3.5 Fine-Tuning + BioNER0.4548 0.5233 0.9244 7.67 10.1513 1.0948 0.3083 0.5531 0.8904 0.6499
Qwen3.5 Fine-Tuning + RAG0.3379 0.4253 0.9055 9.75 10.7824 1.0682 0.4933 0.3684 0.879 0.4476
Qwen3.5 Fine-Tuning BioNER + RAG0.2826 0.3332 0.8733 12.71 11.3036 1.4295 0.5741 0.2832 0.8088 0.1501
BioBART Fine-Tuning0.5438 0.5953 0.94227.29 10.2835 1.0497 0.61290.6629 0.93440.9126
BioBART Fine-Tuning + BioNER0.5335 0.5845 0.9399 6.09 10.0381.00170.6005 0.6301 0.9255 0.9194
BioBART Fine-Tuning + RAG0.452 0.4856 0.9278 6.3 9.9726 1.4031 0.4452 0.4062 0.9219 0.6662
BioBART Fine-Tuning + BioNER + RAG0.3634 0.3792 0.91335.93 9.65122.0637 0.6138 0.293 0.906 0.522
Table 4: Mean scores for the best FT and few-shot strategies for each model
Relevance Readability Factuality
Str ROUGE↑METEOR↑BERTScore↑FKGL↓DCRS↓SLE↓SummaC↑FENICE↑CHEX↑Mean↑
BioBART Fine-Tuning + BioNER0.5335 0.5845 0.9399 6.0910.0381.00170.60050.6301 0.92550.9703
Qwen3.5 BioNER 0-Shot0.3589 0.399 0.9183 6.359.38411.06870.60280.4127 0.8941 0.7059
Qwen3.5 Fine-Tuning + BioNER0.4548 0.5233 0.9244 7.67 10.1513 1.0948 0.3083 0.5531 0.8904 0.623
BioBART 0-Shot0.194 0.2807 0.8448 15.39 11.9016 1.045 0.3 0 0.7708 0.0595
4.1 Discussion
The experimental results partially support the
initial hypotheses. The BioNER strategy con-
sistently improved readability and generally en-
hanced overall performance in both few-shot and
fine-tuning settings. This suggests that explicitly
providing biomedical entity information helps the
models better identify important concepts while
generating lay summaries.
In contrast, the RAG strategy did not consis-
tently improve performance. Manual inspection
showed that the retrieval system occasionally re-
turned Wikipedia entries corresponding to terms
with identical surface forms but different mean-
ings, introducing irrelevant background informa-
tion into the generation process. Although the
prompt instructed the model to ignore unrelated
retrieved content, the FENICE scores indicate that
hallucinated information was still introduced in
some summaries.
The combination of BioNER and RAG did not
produce the expected improvements. Analysis re-
vealed that several multi-word biomedical enti-ties extracted by the BioNER system could not be
matched by the Wikipedia API, resulting in miss-
ing or incomplete retrieved knowledge. Conse-
quently, the potential benefits of retrieval were di-
minished, leading to lower overall performance.
The comparison between the two language
models highlights the importance of domain-
specific pretraining. Although Qwen demon-
strated stronger few-shot capabilities, BioBART
benefited substantially from fine-tuning, ulti-
mately achieving the best overall results. This
finding suggests that biomedical pretraining pro-
vides a stronger foundation for task-specific adap-
tation, whereas larger general-purpose language
models can remain competitive in low-resource
settings without additional training.
4.2 Positive Impact
This study shows that radiology reports can be
made easier for patients to understand without re-
moving the main clinical information. Lay sum-
maries may help patients understand their results
better and ask more useful questions during med-
ical appointments. The findings also suggest that

using biomedical entities can help the model focus
on the most important parts of a report and explain
them in clearer language.
Since the study uses small language models and
LoRA fine-tuning, the proposed setup may also
be practical for institutions with limited comput-
ing resources. At the same time, the RAG results
show that adding external information is not al-
ways helpful. Wrong term matches or unrelated
definitions can introduce information that is not
supported by the report. This points to the need
for more reliable medical knowledge sources and
careful checking before these systems are used in
practice.
These summaries should be used as an aid for
patients and clinicians, not as a replacement for
medical advice. Further testing with both patients
and healthcare professionals is still needed before
patient-facing use.
5 Conclusion
This study investigated the effects of BioNER-
and RAG-based strategies on radiology report lay
summarization under both few-shot inference and
fine-tuning settings. The proposed approaches
were evaluated using nine metrics covering rel-
evance, readability, and factuality with equal
weighting.
The experimental results show that the BioNER
strategy consistently improved the baseline mod-
els, particularly in terms of readability, while also
maintaining competitive relevance and factuality.
In contrast, the RAG strategy did not consistently
improve performance, and combining BioNER
with RAG did not yield the expected gains. Over-
all, the findings partially support the initial hy-
potheses: BioNER proved to be an effective en-
hancement for lay summarization, whereas the ef-
fectiveness of RAG was limited by the quality of
the retrieved knowledge.
These results demonstrate that providing ex-
plicit biomedical entity information is a simple yet
effective approach for improving the readability
of automatically generated lay summaries. Future
work will focus on improving the retrieval com-
ponent by exploring domain-specific knowledge
bases, biomedical knowledge graphs, and more ro-
bust entity linking methods to reduce retrieval er-
rors. In addition, investigating alternative BioNER
models and retrieval strategies may further im-
prove the quality and factual consistency of gen-erated lay summaries.
References
Satanjeev Banerjee and Alon Lavie. 2005. METEOR:
An automatic metric for MT evaluation with im-
proved correlation with human judgments. InPro-
ceedings of the ACL Workshop on Intrinsic and Ex-
trinsic Evaluation Measures for Machine Transla-
tion and/or Summarization, pages 65–72. Associa-
tion for Computational Linguistics.
Olivier Bodenreider. 2004. The unified medical lan-
guage system (UMLS): integrating biomedical ter-
minology.Nucleic Acids Research, 32(Suppl.
1):D267–D270.
Aurelia Bustos, Antonio Pertusa, Jose-Maria Sali-
nas, and Maria de la Iglesia-Vay ´a. 2020. Padch-
est: A large chest x-ray image dataset with multi-
label annotated reports.Medical Image Analysis,
66:101797.
Chroma. 2026. ChromaDB: The open-source
search infrastructure for ai.https://www.
trychroma.com/products/chromadb. Ac-
cessed: 2026-07-08.
Cagla Colak and Lknur Karadeniz. 2023. ISIKSumm
at BioLaySumm task 1: BART-based summariza-
tion system enhanced with bio-entity labels. InPro-
ceedings of the 22nd Workshop on Biomedical Natu-
ral Language Processing and BioNLP Shared Tasks,
pages 636–640, Toronto, Canada. Association for
Computational Linguistics.
Liam Cripwell, Jo ¨el Legrand, and Claire Gardent.
2023. Simplicity level estimate (SLE): A learned
reference-less metric for sentence simplification. In
Proceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, pages
12053–12059. Association for Computational Lin-
guistics.
Edgar Dale and Jeanne S. Chall. 1948. A formula for
predicting readability.Educational Research Bul-
letin, 27(1):11–20.
Maria de la Iglesia Vay ´a, Jose Manuel Saborit,
Joaquim Angel Montell, Antonio Pertusa, Au-
relia Bustos, Miguel Cazorla, Joaquin Galant,
Xavier Barber, Domingo Orozco-Beltr ´an, Francisco
Garc ´ıa-Garc ´ıa, Marisa Caparr ´os, Germ ´an Gonz ´alez,
and Jose Mar ´ıa Salinas. 2020. Bimcv covid-
19+: a large annotated dataset of rx and ct
images from covid-19 patients.arXiv preprint
arXiv:2006.01174.
DeepSeek-AI. 2024. Deepseek-v3 technical report.
arXiv preprint arXiv:2412.19437.
Dina Demner-Fushman, Sameer Antani, Matthew
Simpson, and George R. Thoma. 2012. Design and
development of a multimodal biomedical informa-
tion retrieval system.Journal of Computing Science
and Engineering, 6(2):168–177.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and
Luke Zettlemoyer. 2023. QLoRA: Efficient finetun-
ing of quantized LLMs.
Egecan Evgin, Ilknur Karadeniz, and Olcay Taner
Yıldız. 2025. MetninOzU at BioLaySumm2025:
Text summarization with reverse data augmenta-
tion and injecting salient sentences. InProceed-
ings of the 24th Workshop on Biomedical Language
Processing (Shared Tasks), pages 179–184, Vienna,
Austria. Association for Computational Linguistics.
Tomas Goldsack, Zheheng Luo, Qianqian Xie, Car-
olina Scarton, Matthew Shardlow, Sophia Anani-
adou, and Chenghua Lin. 2023. Overview of the
biolaysumm 2023 shared task on lay summariza-
tion of biomedical research articles. InProceedings
of the 22nd Workshop on Biomedical Natural Lan-
guage Processing and BioNLP Shared Tasks, pages
468–477, Toronto, Canada. Association for Compu-
tational Linguistics.
Tomas Goldsack, Carolina Scarton, Matthew Shard-
low, and Chenghua Lin. 2024. Overview of the Bi-
oLaySumm 2024 shared task on the lay summariza-
tion of biomedical research articles. InProceedings
of the 23rd Workshop on Biomedical Natural Lan-
guage Processing, pages 122–131, Bangkok, Thai-
land. Association for Computational Linguistics.
Google. 2024. An overview of the gemini app.
https://gemini.google/overview/.
Yue Guo, Wei Qiu, Gondy Leroy, Sheng Wang, and
Trevor Cohen. 2024. Retrieval augmentation of
large language models for lay language generation.
Journal of Biomedical Informatics, 149:104580.
Aaradhya Gupta and Parameswari Krishnamurthy.
2025. Shared task at biolaysumm2025 : Extract then
summarize approach augmented with umls based
definition retrieval for lay summary generation. In
Proceedings of the 24th Workshop on Biomedical
Language Processing (Shared Tasks), pages 185–
189, Vienna, Austria. Association for Computa-
tional Linguistics.
Jiawei Han, Micheline Kamber, and Jian Pei. 2011.
Data Mining: Concepts and Techniques, 3 edition.
Morgan Kaufmann.
Alistair E. W. Johnson, Tom J. Pollard, Seth J.
Berkowitz, Nathaniel R. Greenbaum, Matthew P.
Lungren, Chih-ying Deng, Roger G. Mark, and
Steven Horng. 2019. Mimic-cxr, a de-identified
publicly available database of chest radiographs with
free-text reports.Scientific Data, 6(1):317.
J. Peter Kincaid, Robert P. Fishburne, Richard L.
Rogers, and Brad S. Chissom. 1975. Derivation of
new readability formulas (automated readability in-
dex, fog count and flesch reading ease formula) for
navy enlisted personnel. Technical Report Research
Branch Report 8-75, Naval Technical Training Com-
mand.Zeljko Kraljevic, Thomas Searle, Anthony Shek,
Lukasz Roguski, Kawsar Noor, Daniel Bean, Au-
relie Mascio, Leilei Zhu, Amos A. Folarin, Angus
Roberts, Rebecca Bendayan, Mark P. Richardson,
Robert Stewart, Anoop D. Shah, Wai Keong Wong,
Zina Ibrahim, James T. Teo, and Richard J. B. Dob-
son. 2021. Multi-domain clinical natural language
processing with medcat: The medical concept an-
notation toolkit.Artificial Intelligence in Medicine,
117:102083.
Philippe Laban, Tobias Schnabel, Paul N. Bennett, and
Marti A. Hearst. 2022. SummaC: Re-visiting NLI-
based models for inconsistency detection in summa-
rization.Transactions of the Association for Com-
putational Linguistics, 10:163–177.
Jinhyuk Lee, Wonjin Yoon, Sungdong Kim,
Donghyeon Kim, Sunkyu Kim, Chan Ho So,
and Jaewoo Kang. 2020. BioBERT: a pre-
trained biomedical language representation model
for biomedical text mining.Bioinformatics,
36(4):1234–1240.
Chin-Yew Lin. 2004. ROUGE: A package for auto-
matic evaluation of summaries. InText Summa-
rization Branches Out, pages 74–81. Association for
Computational Linguistics.
Fan Lin and Dezhi Yu. 2025. LaySummX at Bi-
oLaySumm: Retrieval-augmented fine-tuning for
biomedical lay summarization using abstracts and
retrieved full-text context. InProceedings of the
24th Workshop on Biomedical Language Process-
ing (Shared Tasks), pages 202–214, Vienna, Austria.
Association for Computational Linguistics.
Juan Antonio Lossio-Ventura, Callum Chan, Arshitha
Basavaraj, Hugo Alatrista-Salas, Francisco Pereira,
and Diana Inkpen. 2025. 5cNLP at BioLay-
Summ2025: Prompts, retrieval, and multimodal fu-
sion. InProceedings of the 24th Workshop on
Biomedical Language Processing (Shared Tasks),
pages 215–231, Vienna, Austria. Association for
Computational Linguistics.
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler
Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
Nouha Dziri, Shrimai Prabhumoye, Yiming Yang,
Sean Welleck, Bodhisattwa Prasad Majumder,
Shashank Gupta, Amir Yazdanbakhsh, and Peter
Clark. 2023. Self-refine: Iterative refinement with
self-feedback.
Meta AI. 2024. Llama 3.2 Model Card.
https://github.com/meta-llama/
llama-models/blob/main/models/
llama3_2/MODEL_CARD.md.
Shufan Ming, Yue Guo, and Halil Kilicoglu. 2025.
Towards knowledge-guided biomedical lay summa-
rization using large language models. InPro-
ceedings of the Second Workshop on Patient-
Oriented Language Processing, pages 285–297, Al-
buquerque, New Mexico. Association for Computa-
tional Linguistics.

Nur Alya Dania binti Moriazi and Mujeen Sung. 2025.
KHU LDI at BioLaySumm2025: Fine-tuning and
refinement for lay radiology report generation. In
Proceedings of the 24th Workshop on Biomedical
Language Processing (Shared Tasks), pages 256–
268, Vienna, Austria. Association for Computa-
tional Linguistics.
Mark Neumann, Daniel King, Iz Beltagy, and Waleed
Ammar. 2019. ScispaCy: Fast and robust models
for biomedical natural language processing. InPro-
ceedings of the 18th BioNLP Workshop and Shared
Task, pages 319–327, Florence, Italy. Association
for Computational Linguistics.
OpenAI. 2022. Introducing chatgpt.https://
openai.com/index/chatgpt/.
Annemarie Katharina Proff, Babak Salam, Mohammed
Hayawi, Dmitrij Kravchenko, Narine Mesropyan,
Taraneh Aziz-Safaie, Tatjana Dell, Maike Theis,
Claus Christian Pieper, Alois Martin Sprinkart,
Daniel K ¨utting, Julian Alexander Luetkens, Sebas-
tian Nowak, and Alexander Isaak. 2026. Simplify-
ing radiology reports with large language models:
privacy-compliant open- versus closed-weight mod-
els.European Radiology.
Qwen Team. 2026. Qwen3.5: Towards native multi-
modal agents.
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine
Lee, Sharan Narang, Michael Matena, Yanqi Zhou,
Wei Li, and Peter J. Liu. 2020. Exploring the limits
of transfer learning with a unified text-to-text trans-
former.Journal of Machine Learning Research,
21(140):1–67.
Alessandro Scir `e, Karim Ghonim, and Roberto Nav-
igli. 2024. FENICE: Factuality evaluation of sum-
marization based on natural language inference and
claim extraction. InFindings of the Association
for Computational Linguistics: ACL 2024, pages
14148–14161. Association for Computational Lin-
guistics.
Bhuvaneswari Sivagnanam, Rivo Krishnu C H, Princi
Chauhan, and Saranya Rajiakodi. 2025. CUTN Bio
at BioLaySumm: Multi-task prompt tuning with
external knowledge and readability adaptation for
layman summarization. InProceedings of the
24th Workshop on Biomedical Language Process-
ing (Shared Tasks), pages 269–274, Vienna, Austria.
Association for Computational Linguistics.
Akshay Smit, Saahil Jain, Pranav Rajpurkar, Anuj Pa-
reek, Andrew Y . Ng, and Matthew P. Lungren. 2020.
CheXBert: Combining automatic labelers and ex-
pert annotations for accurate radiology report label-
ing using BERT. InProceedings of the 2020 Con-
ference on Empirical Methods in Natural Language
Processing, pages 1500–1519. Association for Com-
putational Linguistics.Liwen Sun, James Jialun Zhao, Wenjing Han, and
Chenyan Xiong. 2025. Fact-aware multimodal re-
trieval augmentation for accurate medical radiology
report generation. InProceedings of the 2025 Con-
ference of the Nations of the Americas Chapter of the
Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers),
pages 643–655. Association for Computational Lin-
guistics.
Lewis Tunstall, Edward Beeching, Nathan Lambert,
Nazneen Rajani, Kashif Rasul, Younes Belkada,
Shengyi Huang, Leandro von Werra, Cl ´ementine
Fourrier, Nathan Habib, Nathan Sarrazin, Omar San-
seviero, Alexander M. Rush, and Thomas Wolf.
2023. Zephyr: Direct distillation of lm alignment.
Chenghao Xiao, Kun Zhao, Xiao Wang, Siwei Wu,
Sixing Yan, Tomas Goldsack, Sophia Ananiadou,
Noura Al Moubayed, Liang Zhan, William K. Che-
ung, and Chenghua Lin. 2025. Overview of the Bio-
LaySumm 2025 shared task on lay summarization of
biomedical research articles and radiology reports.
InProceedings of the 24th Workshop on Biomedical
Language Processing, pages 365–377, Vienna, Aus-
tria. Association for Computational Linguistics.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, Chujie Zheng, Dayi-
heng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge,
Haoran Wei, Huan Lin, Jialong Tang, et al. 2025.
Qwen3 technical report.
An Yang, Baosong Yang, Beichen Zhang, Binyuan
Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayi-
heng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian
Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang,
Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang,
et al. 2024. Qwen2.5 technical report.
Zhiwen You, Shruthan Radhakrishna, Shufan Ming,
and Halil Kilicoglu. 2024. UIUC BioNLP at Bi-
oLaySumm: An extract-then-summarize approach
augmented with Wikipedia knowledge for biomedi-
cal lay summarization. InProceedings of the 23rd
Workshop on Biomedical Natural Language Pro-
cessing, pages 132–143, Bangkok, Thailand. Asso-
ciation for Computational Linguistics.
Hongyi Yuan, Zheng Yuan, Ruyi Gan, Jiaxing Zhang,
Yutao Xie, and Sheng Yu. 2022. Biobart: Pretrain-
ing and evaluation of a biomedical generative lan-
guage model.
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q.
Weinberger, and Yoav Artzi. 2020. BERTScore:
Evaluating text generation with BERT. InInterna-
tional Conference on Learning Representations.
Wenjun Zhang, Shekhar Chandra, Bevan Koopman, Ja-
son Dowling, and Aaron Nicolson. 2025. AEHRC at
BioLaySumm 2025: Leveraging t5 for lay summari-
sation of radiology reports. InProceedings of the

24th Workshop on Biomedical Language Process-
ing (Shared Tasks), pages 171–178, Vienna, Austria.
Association for Computational Linguistics.
Yuhao Zhang, Yuhui Zhang, Peng Qi, Christopher D.
Manning, and Curtis P. Langlotz. 2021. Biomedical
and clinical english model packages for the stanza
python nlp library.Journal of the American Medical
Informatics Association, 28(9):1892–1899.
Kun Zhao, Chenghao Xiao, Sixing Yan, Haoteng Tang,
William K. Cheung, Noura Al Moubayed, Liang
Zhan, and Chenghua Lin. 2026. X-ray made simple:
Lay radiology report generation and robust evalua-
tion. InFindings of the Association for Computa-
tional Linguistics: ACL 2026, pages 34583–34598.
Association for Computational Linguistics.
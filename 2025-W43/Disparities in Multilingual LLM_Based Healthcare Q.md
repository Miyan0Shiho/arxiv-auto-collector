# Disparities in Multilingual LLM-Based Healthcare Q&A

**Authors**: Ipek Baris Schlicht, Burcu Sayin, Zhixue Zhao, Frederik M. Labonté, Cesare Barbera, Marco Viviani, Paolo Rosso, Lucie Flek

**Published**: 2025-10-20 12:19:08

**PDF URL**: [http://arxiv.org/pdf/2510.17476v1](http://arxiv.org/pdf/2510.17476v1)

## Abstract
Equitable access to reliable health information is vital when integrating AI
into healthcare. Yet, information quality varies across languages, raising
concerns about the reliability and consistency of multilingual Large Language
Models (LLMs). We systematically examine cross-lingual disparities in
pre-training source and factuality alignment in LLM answers for multilingual
healthcare Q&A across English, German, Turkish, Chinese (Mandarin), and
Italian. We (i) constructed Multilingual Wiki Health Care
(MultiWikiHealthCare), a multilingual dataset from Wikipedia; (ii) analyzed
cross-lingual healthcare coverage; (iii) assessed LLM response alignment with
these references; and (iv) conducted a case study on factual alignment through
the use of contextual information and Retrieval-Augmented Generation (RAG). Our
findings reveal substantial cross-lingual disparities in both Wikipedia
coverage and LLM factual alignment. Across LLMs, responses align more with
English Wikipedia, even when the prompts are non-English. Providing contextual
excerpts from non-English Wikipedia at inference time effectively shifts
factual alignment toward culturally relevant knowledge. These results highlight
practical pathways for building more equitable, multilingual AI systems for
healthcare.

## Full Text


<!-- PDF content starts -->

Disparities in Multilingual LLM-Based Healthcare Q&A
Ipek Baris Schlicht1, Burcu Sayin2, Zhixue Zhao3,
Frederik M. Labonté4,5,6,Cesare Barbera7,Marco Viviani8Paolo Rosso1,9,Lucie Flek4,5,6,
1Universitat Politècnica de València, Spain2University of Trento, Italy
3University of Sheffield, United Kingdom4Bonn-Aachen International Center for IT, Germany
5University of Bonn, Germany6Lamarr Institute for ML and AI, Germany
7University of Pisa, Italy8University of Milano-Bicocca, Italy
9ValgrAI Valencian Graduate School and Research Network of Artificial Intelligence, Spain
Abstract
Equitable access to reliable health informa-
tion is vital when integrating AI into health-
care. Yet, information quality varies across
languages, raising concerns about the reliabil-
ity and consistency of multilingual Large Lan-
guage Models (LLMs). We systematically ex-
amine cross-lingual disparities in pre-training
source and factuality alignment in LLM an-
swers for multilingual healthcare Q&A across
English, German, Turkish, Chinese (Mandarin),
and Italian. We (i)constructed Multilingual
Wiki Health Care (MultiWikiHealthCare), a
multilingual dataset from Wikipedia; (ii)ana-
lyzed cross-lingual healthcare coverage; (iii)
assessed LLM response alignment with these
references; and (iv) conducted a case study
on factual alignment through the use of con-
textual information and Retrieval-Augmented
Generation (RAG). Our findings reveal substan-
tial cross-lingual disparities in both Wikipedia
coverage and LLM factual alignment. Across
LLMs, responses align more with English
Wikipedia, even when the prompts are non-
English. Providing contextual excerpts from
non-English Wikipedia at inference time effec-
tively shifts factual alignment toward culturally
relevant knowledge. These results highlight
practical pathways for building more equitable,
multilingual AI systems for healthcare.
1 Introduction
LLMs are increasingly deployed across healthcare
applications, and people seeking health informa-
tion routinely turn to LLM-based systems for ad-
vice and guidance (Yagnik et al., 2024; Yu et al.,
2024). Since LLMs are trained primarily on large-
scale online data, their responses are shaped by the
availability and quality of online health informa-
tion (Nigatu et al., 2025). However, this informa-
tion varies markedly across languages, reflecting
disparities in health communication, infrastructure,
and cultural norms (Tierney et al., 2025; Yang and
Valdez, 2025).Several healthcare benchmarks probe both LLM
hallucination and disparity analysis (Agarwal et al.,
2024; Koopman and Zuccon, 2023; Kim et al.,
2025; Zhu et al., 2019; Samir et al., 2024), but
they remain largely English-centric or too coarse-
grained to diagnose performance gaps across lan-
guages. Moreover, although related, the two con-
cepts operate on different levels of analysis. While
hallucination detectionfocuses on identifying fac-
tually incorrect or fabricated content (Kim et al.,
2025; Zhang et al., 2025),disparity analysisexam-
ines differences in how information is represented
or prioritized across linguistic and contextual
boundaries, even when the facts themselves may
vary across cultural settings (Samir et al., 2024;
Ranathunga and de Silva, 2022). Prior work has
applied disparity analysis to Wikipedia articles on
people and cuisines (Samir et al., 2024; Wang et al.,
2025) and to LLM outputs in medicine (Gupta et al.,
2025; Restrepo et al., 2025). However, no study has
explicitly linked disparities between training data
(e.g., Wikipedia) and LLM-generated responses.
In this paper, we introduce a holistic framework
to assess how LLM-generated health answers align
with factuality and culture across languages. As
illustrated in Figure 1, (i)we begin by compar-
ing healthcare-related Wikipedia pages, which is a
pretraining corpus for many LLMs (Singhal et al.,
2023), across languages to characterize similari-
ties and discrepancies in coverage, phrasing, and
citation patterns. From this analysis, (ii)we con-
struct aligned cross-lingual fact sets and use them
to generate questions posed to several multilingual
LLMs: Llama3.3-70B (Dubey et al., 2024), Qwen3-
Next-80B-A3B-Instruct (Yang et al., 2025), and
Aya (Dang et al., 2024). Subsequently, (iii) we
evaluate the responses for quality and alignment
with both English Wikipedia and the correspond-
ing target-language pages. Finally, magenta (iv)
we present a case study testing whether provid-
ing non-English contextual excerpts at inference
1arXiv:2510.17476v1  [cs.CL]  20 Oct 2025

…Although 70 mg/dL is the lower limit of normal glucose,
symptoms of hypoglycemia usually do not occur until blood sugar
has fallen to 55 mg/dL or lower…The blood-glucose level at which
symptoms of hypoglycemia develop in someone with several prior
episodes of hypoglycemia may be even lower. If a person without
diabetes accidentally takes medications that are traditionally used
to treat diabetes, this may also cause…Turkish Wikipedia(1) Source-Level Comparison
…Ciddi hastalık düşük kanşekerine neden olabilir. Neredeyse tüm
ana organ sistemlerininşiddetli hastalığı, ikincil bir sorun olarak
hipoglisemiye neden olabilir. Hastanede yatan kişiler, özellikle
yoğun bakım üniteleri veya yemek yemeleri engellenenler, birincil
hastalıklarının bakımıyla ilgili çeşitli durumlardan hipoglisemi
geliştirebilirler…
…Serious illness can cause low blood sugar. Severe disease of
almost any major organ system can cause hypoglycemia as a
secondary problem. People who are hospitalized, especially those
in intensive care units or those who are prevented from eating,
can develop hypoglycemia from a variety of conditions related to
the care of their primary illness…Hipoglisemi genellikle
hangi koşullarda ortaya
çıkar?
Hypoglycemia, or low blood sugar, typically occurs under the
following conditions:
1. Diabetes Management: In people with diabetes, especially those
who take insulin or certain oral medications, hypoglycemia can
occur if:
- Too much insulin or medication is taken.
- Meals are skipped or delayed.
- Physical activity levels are higher than usual (without adjusting
medication or food intake).
- Alcohol is consumed…
English Wikipedia(2) Question Generation
Translation
What conditions does
hypoglycemia usually
occur?Hipoglisemi, kanşekeri düzeyinin normal aralığın altına
düşmesiyle ortaya çıkan bir durumdur ve genellikle aşağıdaki
koşullarda görülebilir:
1. Diyabet Tedavisi: En yaygın nedenlerden biri, tip 1 veya tip 2
diyabetli hastalarda insülin veya diğer kanşekeri düşürücü
ilaçların aşırıkullanımıdır.İlaç dozajının yanlışayarlanması, yemek
yeme alışkanlıklarındaki değişiklikler veya egzersiz miktarındaki
artış, kanşekeri seviyelerinin çok düşmesine neden olabilir…
Hypoglycemia is a condition that occurs when blood sugar levels
drop below the normal range and is typically seen in the following
conditions:
1. Diabetes Treatment: One of the most common causes is the
overuse of insulin or other blood sugar-lowering medications in
patients with type 1 or type 2 diabetes. Improper medication
dosage, changes in eating habits, or increased exercise can cause
blood sugar levels to drop too low…(3) Response-Level Comparison
LLM interaction
(4) Factuality Alignment
Search Engine
PubMedRAG Wiki
System PromptFigure 1: Analyzing source- and response-level disparity and factuality alignment: (1) comparison of Turkish and
English Wikipedia pages, (2) fact-based question generation, (3) response factuality evaluation and (4) contextual
alignment using Wiki pages and RAG. English translations are shown in blue.
time shifts factual alignment toward locally rele-
vant sources, through the use of contextual informa-
tion andRetrieval-Augmented Generation(RAG).
Accordingly, the main contributions of this work
can be summarized as follows:
•We introduceMultiWikiHealthCare, a mul-
tilingual dataset of trending healthcare top-
ics from Wikipedia in English, German, Ital-
ian, Turkish, and Chinese, enabling system-
atic cross-lingual comparisons at both source
and response levels1.
•Wikipedia analysis reveals major disparities:
Chinese pages show the least factual overlap
with English, and German pages more often
cite regional sources, while others rely on in-
ternational ones.
•Analysis of responses shows a pronounced
English-centric alignment: responses more
closely track English Wikipedia than same-
topic pages in other languages. When ques-
tions are posed in English, similarity to non-
English source pages drops further. While this
may be acceptable for universal medical facts
1Our dataset and code will be released publicly after re-
view.but problematic for culturally specific knowl-
edge where practices and guidelines vary.
•Our case study demonstrates that providing
non-English contextual excerpts can shift
models toward locally relevant information
at inference time.
2 Related Work
Several studies have examined how prompt design
affects the factual reliability of LLMs in medical
Q&A. Koopman and Zuccon (2023); Kim et al.
(2025) showed that prompt phrasing and cues sig-
nificantly influence factual accuracy, while Sayin
et al. (2024) found that prompt engineering en-
hances physician–LLM collaboration and error cor-
rection. Kaur et al. (2024) reported that LLMs
rarely contradict medical facts but often fail to
challenge false ones. Extending this to multilin-
gual settings, Jin et al. (2024) observed substantial
cross-lingual differences in accuracy, consistency,
and verifiability.
Agarwal et al. (2024) introduce the MedHalu
dataset and the accompanying MedHaluDetect
framework for detecting fine-grained factual hallu-
cinations in responses by LLMs. MedHalu consists
of healthcare-related questions in English from
HealthQA (Zhu et al., 2019), LiveQA (Abacha
2

et al., 2017) and MedicationQA (Abacha et al.,
2019). The authors utilized fine-grained hallu-
cination types proposed by Zhang et al. (2025),
namely, input-conflicting, context-conflicting and
fact-conflicting and then generated artificially hal-
lucinations on the answers based on this taxonomy
by using GPT-3.5 (Achiam et al., 2023). They
evaluated Llama (Touvron et al., 2023), GPT-3.5
and GPT-4 (Touvron et al., 2023) as evaluators and
compared them against layman and expert users.
They found that the LLMs underperformed relative
to expert and layman users in detecting hallucina-
tions.
For cross-lingual factual analysis, Samir et al.
(2024) proposed InfoGap which is a GPT-4 based
framework that decomposes and aligns multilin-
gual Wikipedia facts, applied to biographies from
the LGBTBio Corpus (Park et al., 2021). In the
medical domain, Gupta et al. (2025) analyzed LLM
consistency and empathy across translated mental
health Q&A, while Restrepo et al. (2025) devel-
oped a multilingual ophthalmology benchmark and
introduced CLARA, a RAG-based method for de-
biasing inference. Schlicht et al. (2025) compared
LLM outputs across languages, identifying discrep-
ancies in detail, numerical accuracy, and citation
reliability. Unlike prior healthcare benchmarks, we
investigate the correlation between factual knowl-
edge potentially acquired during pretraining and
the factual alignment of LLM-generated responses
across languages. Furthermore, we extend this anal-
ysis to culturally diverse languages.
3 Multilingual Wiki Health Care
Existing benchmark datasets largely focus on
monolingual or general-purpose tasks, lacking the
specificity needed to assess how pretraining sources
affect multilingual answer quality in specialized do-
mains such as healthcare. To address this gap, we
introduceMultiWikiHealthCare, a multilingual,
health-focused Q&A dataset. It is derived from
Wikipedia, pretraining source for LLMs. Figure 2
presents the pipeline for constructing the dataset.
Vaccination, Weight Loss, Diet, Obesity, Nutri-
tion, Flu, Cold, Influenza, Ebola, COVID, Al-
lergy, Smoking, Pain, Depression, Diabetes, Car-
diovascular Disease, Cancer
Table 1: List of the main health topics which we used
for searching trending sub-topics from Google Trends.
1. Searching 
health topics 
2. Scraping 
Wiki pages 3. Fact extraction 
and alignment 4. Filtering 
(Relevancy) English 
Italian German 
Turkish Chinese 
GPT-4o mini 
GPT-4o mini 
InfoGap 
1. Question 
Generation Fact
GPT-4o mini 
Paragraph 
(from Wiki) Prompt Wiki Title 
4. Answer 
Translation 
Aligned Fact Extraction 
Q&A Generation 
5. Answer 
Generation 
Llama 
Aya
ChatGPT 5 
2. Question 
Quality Check DeepSeek 
Qwen 
Figure 2: MultiWikiHealthCare - Pipeline for Q&A
construction
3.1 Construction of Aligned facts
3.1.1 Healthcare Topics
To construct MultiWikiHealthCare, we first curated
a list of trending and controversial health topics2
by using Google Trends,3and Wikipedia’s list of
controversial issues in science, biology, health,4
and related surveys (Schlicht et al., 2024). These
topics formed the basis of content collection. Ta-
ble 1 presents the main health topics in the dataset.
We used Google Trends to identify subtopics from
related entities trending between 2004 and 2025
across global and country-level search data (U.S.,
U.K., Turkey, Germany, Italy, and China). Across
all languages, we identified 1193 unique entities.
Figure 3 shows a word cloud of these subtopics.
Symptoms, causes, and diseases are common en-
tities across languages while some entities are not
specific to the healthcare domain.
Figure 3: Word cloud of trending subtopics derived
from Google Trends (2004–2025) across global and six
countries. The size of each blob corresponds to the
number of languages in which the subtopic appears.
3.1.2 Scraping Wikipedia Pages
We used Llama 3.3-70B, prompted as in Figure 4
and served through the vLLM inference frame-
2Search was done in 2025
3https://trends.google.com/trends/explore
4http://bit.ly/4m59RTa
3

You are an intelligent assistant that identifies whether an entity is re-
lated to healthcare. When given an entity (e.g., a term, organization,
person, product, etc.), first determine whether it is related to health-
care. If it is related to healthcare, return the URL of its Wikipedia
page (e.g., https://en.wikipedia.org/wiki/ENTITY_NAME). If it is
not related to healthcare, return False. Consider healthcare to include:
medicine, medical devices, hospitals, health policy, public health,
pharmaceuticals, biotechnology, mental health, nutrition, diet, food
safety, and functional foods.
Figure 4: Prompt for finding Wikipedia pages
work (Kwon et al., 2023), to (i) filter out entities
not related to healthcare and (ii) link the remain-
ing entities to their corresponding Wikipedia pages.
After removing duplicates, we retained 918 unique
Wikipedia pages.
Wikipedia page titles often differ across lan-
guages, especially when scripts differ or the title
is a common noun rather than a proper name (e.g.,
English/Italian “Nausea” vs. German “Übelkeit”).
Therefore, we retrieved interlanguage titles using
the open-source Wikipedia API5. Pages available
only in English were excluded as they are unsuit-
able for comparative analysis, yielding 815 titles
present in English and at least one additional lan-
guage. A subsequent manual review showed that
some pages referred to ‘films’ or ‘people’. We col-
lected category metadata from English Wikipedia
and removed pages whose categories included ‘peo-
ple’, ‘film’, or ‘doctoral degree’. The final English-
language subset comprises 799 pages.
3.1.3 Fact Extraction and Alignment
Language Pair F1-Macro Random
en↔tr 0.841 0.574
en↔zh 0.724 0.535
en↔de 0.684 0.479
en↔it 0.638 0.538
Table 2: Results comparison between human annotators
and InfoGap predictions and random guess
From the Wikipedia articles, we extracted fac-
tual facts together with their supporting paragraphs
as evidence. We categorized these facts into
two groups:(1) Cross-lingual overlapping facts,
where the same factual content appears in both En-
glish and non-English version; and(2) Language-
specific facts, which are unique to the non-English
article without an English counterpart. The first
5https://github.com/martin-majlis/
Wikipedia-APIgroup is used to construct the Q&A dataset and the
second is part of Wikipedia analysis in Section 4.1.
Fact extraction and alignment were performed
with InfoGap (Samir et al., 2024; Wang et al.,
2025), a state-of-the-art framework for cross-
lingual fact extraction and alignment on Wikipedia.
The framework combines OpenAI LLMs with a
hubness-based correction technique to improve
cross-language alignment accuracy. Because a sin-
gle Wikipedia page can contain hundreds of atomic
facts, running OpenAI models at corpus scale is
costly (Samir et al., 2024). While the latest Info-
Gap (Wang et al., 2025) employs GPT-4o (Achiam
et al., 2023), we adopted GPT-4o-mini (Hurst et al.,
2024) as the backbone model, which is approxi-
mately 10% less expensive than GPT-4o. To assess
reliability on our corpus, we sampled 50 facts per
language following the InfoGap evaluation proto-
col. Annotations were conducted by volunteer na-
tive speakers using the official InfoGap guidelines.
Our results (see Table 2) are comparable to those
reported by Samir et al. (2024), which outperform
a random prediction6.
Example Relevancy
Vitamin C exhibits low/acute toxicity.Relevant
Non-radioactive iodine can be taken in the
form of potassium iodide tablets.Relevant
If a dog was sick, they would get better food. Irrelevant
Routine chickenpox vaccination was intro-
duced in the United States.Irrelevant
Table 3: Examples of relevant and irrelevant facts.
3.1.4 Selecting Relevant facts
You are given a factual statement from Wikipedia. Decide whether
it would be useful for someone seeking health-related information.
Relevantmeans it provides useful health-related information such
as symptoms, causes, risk factors, treatments, prevention, prognosis,
lifestyle advice, or other patient-centered context.
Not relevantmeans it is historical, administrative, technical, or
other information that is not directly useful for someone with health-
related queries.
Task:Respond with only “Relevant” or “Not relevant”.
Input:[Single factual statement]
Response:Relevant / Not relevant
Figure 5: Prompt for selecting relevant facts
After cross-lingual extraction and alignment, we
retained only bidirectionally matched facts: the
intersection of the two directions for each language
6In (Samir et al., 2024), random guessing outperformed
Natural Language Inference transformers.
4

Language LLM Mono TF Cross TF
en 82.69 86.4988.44
tr 77.89 83.4684.20
zh 68.4777.0272.11
de 76.67 70.8881.32
it 71.96 72.0083.33
Table 4: Comparison of transformers (TF): cross-lingual
and mono-lingual against GPT-4o-mini in terms of F1-
macro.
pair (e.g. en<->tr). Although the facts are atomic,
InfoGap returns the aligned source sentences in
both languages. We then located the paragraphs
containing these sentences and matched each fact
with its corresponding bilingual evidence.
We observed that not all Wikipedia-derived
facts are relevant to health information seekers
(e.g., some facts are historical or highly techni-
cal). The examples of relevant/irrelevant facts are
given in Table 3. To remove such content, we
implemented a relevance filter. As the corpus is
large, running GPT-4o-mini over all facts would
be time-consuming and costly. To lower inference
cost while preserving quality, we distilled GPT-4o-
mini’s judgments (and DeepSeek-R1 (Guo et al.,
2025)’s for Chinese) into a smaller model.
To calibrate a shared understanding of the rele-
vance task and design the prompt for labeling, we
performed annotations. The annotators first labeled
at least 25 English samples, with two annotators per
sample. We held regular discussions to refine the
annotation guidelines. Inter-annotator agreement,
measured by Krippendorff’s alpha (Ford, 2004) on
a subset of 52 samples, was α= 0.72 . Each an-
notator then labeled 50 samples in their respective
language. The annotations form the test set for
comparing the LLMs with transformer-based mod-
els across languages. Furthermore, we finalized
the labeling prompt in English (given in Figure 5).
The native speakers translated it into the other lan-
guages.
We annotated 2,000 Turkish and English, 4,000
German, 3,000 Italian, and 3,000 Chinese samples
with LLMs. GPT-4o-mini was used for all except
Chinese, which was labeled with DeepSeek-R1 due
to poor alignment. Each dataset was randomly
split into 70% training and 30% development sets.
We fine-tuned language specific transformers on
their monolingual dataset (Roberta-base (Liu et al.,
2019) for English, BertTurk (Schweter, 2020) for
Turkish, German BERT (Devlin et al., 2019) forGerman) and Chinese BERT (Devlin et al., 2019)
for the Chinese dataset. Additionally, for each non-
English dataset was combined with the English
samples, and then XLM-Roberta7was fine-tuned.
All models were fine-tuned with a learning rate of
2e-5, for 3 epochs, and a batch size of 16. The
models with the best F1-macro scores (see Table 4)
were selected. Except for Chinese, cross-lingual
models performed best and were used as relevancy
classifiers. Finally, we labeled the corpora with
the fine-tuned transformers (Wolf et al., 2020) and
discarded unrelated samples.
3.2 Question and Answer Generation
You are a person seeking health-related information. Task: Using
the provided Fact from the ENTITY Wikipedia page and the accom-
panying Paragraph for context, generate only one natural question.
The question must be directly related to the given Fact.
The answer to the question must be fully answered by the informa-
tion in the Paragraph (do not add outside knowledge).
The question should sound like something a person curious about
health might naturally ask.
Keep the question concise and easy to understand.
Fact:
{fact}
Paragraph:
{paragraph}
Generated Question:
Figure 6: Prompt for health-care question generation
I want to analyze whether the LLM has generated a sensible
question based on the prompt I provided. Here is the prompt:
“‘You are someone searching for health information that can help
you make decisions about your own or someone else’s health.
Task: Using the information provided on the entity Wikipedia page
and the related paragraph as context, generate only one natural
question.
- The question must be directly related to the given information.
- The question must be answerable solely based on the information
in the paragraph (do not add knowledge from external sources).
- The question must be something a person curious about
health-related information would naturally ask.
- The question must be clear and easy to understand.
Input:
Fact: [A single factual/informational sentence]
Paragraph: [Paragraph]
Output: [Generated question]”’
Now, I will give you a CSV file containing a set of examples, each
row representing one instance. The columns in the file are as
follows:
"fact":the information
"evidence":the paragraph from which the fact is derived
"llm_output":the question generated by the LLM according to the
{LANG} prompt
Please carefully examine the CSV file and evaluate the quality of
the generated question for each example. Use only the quality
criteria defined in the prompt above; do not introduce any additional
criteria. Then, provide me a detailed analysis and a technical report
in English.
Figure 7: Prompt for quality check with GPT-5
7FacebookAI/xlm-roberta-base
5

Question Generation.We generated synthetic
health-related questions using a prompt that in-
structed GPT-4o-mini to act as a health information
seeker. The prompt (Figure 6) took as input a Wiki
page name, a fact, and its paragraph. From the
dataset mentioned in Section 3.1.4, we sampled
1,100 samples per language to generate.
Formally, let each data instance be represented
asd= (f x, fen, px, pen), where fxis a fact in
Language X, fen={f1
en, . . . , fn
en}(n≥1 ) is
its aligned English fact(s), pxis the paragraph in
Language X containing fx, and penis the aligned
English paragraph(s). Given (fx, px, pen), we used
GPT-4o-mini to generate a question qxin Language
X and then translated it to English with Google
Translate to get the question pair.
We evaluated the quality of the generated ques-
tions based on the LLM-as-a-Judge technique (Gu
et al., 2025) using ChatGPT-5 (Extended Think-
ing) (OpenAI, 2025) with the prompt provided in
Figure 7. This prompt instructed ChatGPT-5 to
execute a deterministic Python-based evaluation
pipeline within its data-analysis sandbox (OpenAI,
2025). The model applied four binary criteria to
each question: (1) relevance to both the input fact
and the source paragraph, (2) answerability based
solely on the paragraph, (3) alignment with natural
health-seeker intent, and (4) clarity of expression.
To assess alignment with human judgments, we
sampled 20 items per language and compared
ChatGPT-5’s decisions to native-speaker annota-
tions. The agreement ranged from 44% to 76%
(highest for Turkish, lowest for German). ChatGPT-
5 consistently accepted fewer questions than human
annotators, indicating a more conservative crite-
rion. Based on this preliminary evidence, we used
ChatGPT-5 as a pre-filter for question quality.
Answer Generation.To generate answers, we
deliberately use multilingual, open-weight LLMs
from distinct organizations to mitigate model-
family bias in our evaluation. Specifically, we
include Llama 3.3–70B (Meta; an upgraded re-
lease of Llama 3 (Grattafiori et al., 2024)), Qwen
(Qwen3-Next-80B-A3B-Instruct) (Yang et al.,
2025) from Alibaba Cloud and Aya (Expanse-32b)
from Cohere (Dang et al., 2024). We discarded
DeepSeek-R1 from the analysis due to its size and
cost. We run Aya locally on a GPU, while the
other models are accessed via APIs through Hug-
ging Face Inference8. For all LLMs, we set the
8huggingface.co/docs/huggingface_hub/en/Language #Sample
Turkish 854
German 997
Italian 502
Chinese 548
Table 5: Final number of the samples per language in
the dataset for analysis of LLM responses
temperature to 1 and the maximum token length to
4096.
4 Experiments and Results
Main research question in this paper is how dispari-
ties across languages in pre-training data contribute
to inconsistencies in LLM answers for multilingual
healthcare Q&A. We first characterize healthcare-
related Wikipedia pages; we then evaluate each
model’s answers for cross-lingual factual alignment
against evidence extracted from the corresponding
pages.
4.1 Comparison of Wikipedia Pages
Aligned facts (%)
Language Paragraphs Facts en→Target Target→en
en 26,752 20,5468 NA NA
tr 8,554 47,711 %33.79 %91.82
zh 728 6,155 %6.98 %59.46
de 18,284 11,8268 %23.36 %98.44
it 14,846 96,342 %26.73 %54.23
Table 6: Statistics of paragraphs, facts and aligned facts
on 502 Wiki pages of MultiWikiHealthCare. NA is Not
Applicable. Many English facts don’t exist in other
language editions. Chinese Wiki has the lowest aligned
facts with the English wiki pages.
Sections Paragraphs Facts Links
tr-en1.51×10−962.88×10−871.72×10−1066.24×10−68
de-en2.48×10−223.39×10−281.99×10−487.14×10−79
zh-en1.44×10−771.39×10−1411.59×10−1336.55×10−60
it-en1.35×10−501.80×10−509.11×10−704.75×10−74
Table 7: The English (en) Wiki pages contain statisti-
cally more information than their other editions (Paired
t-test p value).
We examine the amount of information pre-
sented in Wikipedia articles across languages. To
this extent, we compare number of sections, para-
graphs, facts and external links that the articles
cited.
As articles are typically organized into sections
that aid navigation and reflect the logical structure
package_reference/inference_client
6

of the content, we use the number of sections as
a simple, language-agnostic proxy. Accordingly,
we measure and compare section counts across lan-
guages. We use Beautiful Soup9to parse each
article’s HTML. Next, we count the paragraphs
and facts obtained through InfoGap, as described
in Section 3.1.3, with the results summarized in
Table 6. Lastly, we analyze the references cited by
Wikipedia editors to examine how reference prefer-
ences vary across languages. References serve as
important indicators of information diversity and
reliability. We restrict our analysis to entries with
articles in all target languages. We begin by com-
paring the number of references per article across
languages and then compare the sources cited by
each edition. Additionally, we extract the sources
of the external links by usingtldextract10.
As shown in Table 7, English Wikipedia con-
tains significantly more information than other lan-
guages (paired t-test (Ross and Willson, 2017)).
It has the most paragraphs and extracted facts,
many without counterparts elsewhere. Chinese
Wikipedia shows the fewest English-aligned facts,
likely because English Wikipedia benefits from
a broader, more global editor base. Across lan-
guages, domains associated with the National In-
stitutes of Health and the Centers for Disease
Control and Prevention are common. Other high-
frequency domains are news outlets, scholarly jour-
nals, and file archives. Distinctively, the German
edition also cites national sources (e.g., rki.de ,
aerzteblatt.de ). In summary, citation coverage
and practices vary substantially across language
editions, reflecting differences in local information
ecosystems and editorial norms.
4.2 Analyzing Answers
As our tools are English-based, all non-English an-
swers and evidence were translated into English
for scoring. We begin the analysis with the re-
sponse length. Across non-English prompts, Qwen
consistently produced longer answers, reflecting a
tendency toward more elaborated outputs. For Chi-
nese questions, most models gave brief answers,
except Qwen, likely reflecting its stronger Chinese-
language training.
We evaluate English and non-English answers
against their Wikipedia evidence using Align-
Score (Zha et al., 2023), which measures sentence-
level factual consistency via a RoBERTa alignment
9https://beautiful-soup-4.readthedocs.io
10https://github.com/john-kurkowski/tldextractQuery Llama Aya Qwen
non-en en non-en en non-en en
tr 27.44 30.89 17.78 22.62 14.45 18.31
en 16.96 21.64 15.03 20.02 14.04 18.28
de 18.13 18.56 16.26 17.92 13.64 16.59
en 16.85 18.68 14.69 17.37 13.38 17.02
zh 22.39 28.13 20.22 26.78 17.45 22.71
en 20.09 26.23 16.90 23.81 15.58 21.72
it 20.16 23.51 16.40 20.94 14.12 18.41
en 16.65 21.64 15.04 19.97 13.54 17.97
Table 8: Factual alignment between answers and Wiki
excerpts in the source language (non-en) and English
(en), measured by AlignScore. Non-English content
was translated into English. Overall, answers align more
closely with English pages, while English questions
show lower similarity to source-language references.
model (Liu et al., 2019). For questions with multi-
ple aligned English passages, we report the passage
with the highest AlignScore.
Answers from conversational LLMs frequently
include background and discursive material beyond
the minimal facts needed to address a question (Xu
et al., 2022), whereas our evidence snippets are
concise and fact-dense. Consequently, absolute
AlignScores tend to be modest; we interpret them
as a relative proxy for factual alignment rather than
a comprehensive quality metric. Hence, we expect
higher scores when a response is factually similar
to the source-language reference, and lower scores
otherwise.
Table 8 shows that the generated answers are
factually close to English references in most cases.
However, when questions are asked in English, fac-
tual alignment decreases for all languages except
German. In particular, factual alignment is higher
when evaluated against source-language evidence
than against evidence from the English Wikipedia
pages. This finding is consistent with prior work
showing that responses to equivalent prompts
can diverge across English and non-English set-
tings (Jin et al., 2024; Schlicht et al., 2025).
Lastly, we used the Spearman correlation coeffi-
cient (Wissler, 1905) ( ρ) to assess the relationship
between Wikipedia page quality and LLM answer
quality across languages. For this analysis, we ad-
ditionally measured the relevancy of the answers to
the questions by using ragas (Es et al., 2024) with
GPT-4o-mini. Correlations were generally weak
to moderate ( ρ= 0.01–0.34). For Italian, Turkish,
and German, the relationships were negligible or
weak ( ρ< 0.20), suggesting that the number of
sections, paragraphs, or facts in the corresponding
7

Wikipedia entries had limited predictive power for
the factuality, relevancy, or length of model out-
puts. In contrast, the correlations to the factuality
scores for Chinese were higher, reaching moder-
ate strength for Aya ( ρ≈ 0.34) and Llama ( ρ≈
0.30), and weak-to-moderate for Qwen ( ρ≈0.27).
Wikipedia content quality in Chinese Wikipedia
might have positive effect on factuality of the re-
sponses.
4.3 Case Study: Factuality Alignment
Target MethodLlama Aya Qwen
non-en en non-en en non-en en
trBase 17.28 22.46 15.34 20.91 14.24 18.88
Wiki 78.67 59.21 41.52 35.60 78.53 58.48
RAG 23.51 33.71 15.55 21.45 15.91 27.10
deBase 16.89 18.70 14.71 17.41 13.41 17.02
Wiki 80.68 27.62 34.73 18.43 72.72 20.86
RAG 23.75 34.69 15.89 19.84 15.65 30.21
zhBase 22.44 28.13 16.89 23.87 15.47 21.74
Wiki 83.14 51.39 43.67 34.40 77.99 46.95
RAG 29.83 43.65 17.98 25.62 21.78 36.49
itBase 16.84 21.84 14.37 16.70 13.56 18.17
Wiki 76.78 41.03 37.10 26.79 81.55 40.05
RAG 24.56 40.05 15.31 21.06 15.83 20.29
Table 9: When the excerpt from non-English (non-
en) Wiki pages is translated into English, the answers
aligned more to the source context. With RAG, that is
opposite.
Response alignment with high-resource knowl-
edge sources such as English Wikipedia is often
desirable, as health information in low-resource
languages is often limited or lower quality (Weis-
senberger et al., 2004; Lawrentschuk et al.; Davaris
et al., 2017). In such cases, knowledge from
high-resource languages can effectively comple-
ment existing gaps. However, certain scenarios
demand more localized or domain-specific infor-
mation, where English-centric knowledge may not
be sufficiently reliable or contextually appropriate
for target users (Yang and Valdez, 2025). To ex-
plore this, we evaluate: (1) providing contextual
information into prompt and (2) RAG (Lewis et al.,
2020).
We adopt a RAG prompt from HuggingFace11
for both setups. For the first method, we in-
corporate translated excerpts from non-English
Wikipedia pages that are semantically aligned with
the given question. For RAG, we scrape PubMed
articles12using Paperscraper13, querying by entity
11https://huggingface.co/blog/ngxson/
make-your-own-rag
12https://pubmed.ncbi.nlm.nih.gov/
13https://github.com/jannisborn/paperscraperand nationality keywords (e.g., allergy + Turkish)
to obtain culturally specific information. We dis-
card entities with fewer than 50 retrieved PubMed
articles and perform analysis on the rest. The RAG
model is built using the BM25-Sparse retriever (Lù,
2024). We incorporate top 10 articles that returned
from the retriever into the context.
We compare the approaches against the base-
line where the LLM responds to questions posed
in English (Section 4.2). Both approaches im-
prove reference alignment by producing more fac-
tual and concise answers than the baseline. As
shown in Figure 9, LLMs incorporating excerpts
from Wikipedia produce responses that align more
closely with non-English references, while with
RAG they align more with English references.
RAG results contain usually noisy context, leading
to cases where LLMs are uncertain about their an-
swers. Additionally, due to the increased prompt
length, Aya was unable to generate responses for a
few examples. Explicit, high-quality contextual in-
formation might be crucial for effective alignment.
In future work, we plan to explore more advanced
information retrieval and RAG methods to improve
contextual relevance.
5 Conclusion and Future Work
We introduced MultiWikiHealthCare, a multilin-
gual benchmark for studying disparities across lan-
guages in healthcare question answering. By pair-
ing popular user queries with language-specific
Wikipedia evidence, we quantified how differences
in encyclopedic coverage (e.g., structure, citations,
and fact availability) manifest in model behavior.
Our results show substantial differences across lan-
guages: LLMs often privilege English-centric ev-
idence, while conditioning generation on source-
language excerpts shifts grounding toward locally
relevant knowledge. Taken together, these findings
highlight a practical path for improving equity in
multilingual healthcare Q&A, explicitly anchor an-
swers in the user’s language of reference rather
than defaulting to English.
Future work will extend sources beyond
Wikipedia to clinical and public health materials,
adopt multilingual factuality metrics with native-
speaker review, and scale to more data, languages,
and multi-turn dialogue.
8

Ethics Statement and Limitations
This study examines informational disparities in
LLMs applied to healthcare contexts across multi-
ple languages. All analyses were conducted using
publicly available data from Wikipedia. There-
fore, no private, sensitive or personally identifiable
health information was accessed or processed. All
evaluations were performed solely for research pur-
poses and none of the LLMs analyzed should be
solely used to provide medical device. We acknowl-
edge that the disparities observed across languages
and Wikipedia may reflect broader inequities in
global health communication and data representa-
tion. Through this analysis, we aim to promote
more equitable and fair multilingual health-care
applications.
Furthermore, our work has several limitations:
(i) AlignScore is English-centric, so we trans-
late non-English evidence and answers into En-
glish, which may introduce artifacts; (ii) Wikipedia
is used as the proxy reference for constructing
questions and evidence, and neither Wikipedia ex-
cerpts nor model outputs were independently fact-
checked; (iii) budget constraints (paid APIs for
most models) and limited availability of native
speakers restricted us to a relatively small num-
ber of Q&A pairs and single-turn interactions; and
(iv) because LLMs are trained in heterogeneous
sources, their knowledge does not need to align
with Wikipedia, so our findings reflect alignment
relative to Wikipediarather than clinical correct-
ness.
References
Asma Ben Abacha, Eugene Agichtein, Yuval Pinter, and
Dina Demner-Fushman. 2017. Overview of the med-
ical question answering task at TREC 2017 liveqa.
InTREC, volume 500-324 ofNIST Special Publica-
tion. National Institute of Standards and Technology
(NIST).
Asma Ben Abacha, Yassine Mrabet, Mark Sharp,
Travis R. Goodwin, Sonya E. Shooshan, and Dina
Demner-Fushman. 2019. Bridging the gap between
consumers’ medication questions and trusted an-
swers. InMedInfo, volume 264 ofStudies in Health
Technology and Informatics, pages 25–29. IOS Press.
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, and 1 others. 2023. Gpt-4 techni-
cal report.arXiv preprint arXiv:2303.08774.Vibhor Agarwal, Yiqiao Jin, Mohit Chandra, Munmun
De Choudhury, Srijan Kumar, and Nishanth Sas-
try. 2024. Medhalu: Hallucinations in responses to
healthcare queries by large language models.arXiv
preprint arXiv:2409.19492.
John Dang, Shivalika Singh, Daniel D’souza, Arash
Ahmadian, Alejandro Salamanca, Madeline Smith,
Aidan Peppin, Sungjin Hong, Manoj Govindassamy,
Terrence Zhao, and 1 others. 2024. Aya expanse:
Combining research breakthroughs for a new multi-
lingual frontier.arXiv preprint arXiv:2412.04261.
Myles Davaris, Stephen Barnett, Robert Abouassaly,
Nathan Lawrentschuk, and 1 others. 2017. Thoracic
surgery information on the internet: a multilingual
quality assessment.Interactive Journal of Medical
Research, 6(1):e6732.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. BERT: Pre-training of
deep bidirectional transformers for language under-
standing. InProceedings of the 2019 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies, Volume 1 (Long and Short Papers), pages
4171–4186, Minneapolis, Minnesota. Association for
Computational Linguistics.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, and 1 others. 2024. The llama 3 herd of models.
arXiv preprint arXiv:2407.21783.
Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. RAGAs: Automated evalu-
ation of retrieval augmented generation. InProceed-
ings of the 18th Conference of the European Chap-
ter of the Association for Computational Linguistics:
System Demonstrations, pages 150–158, St. Julians,
Malta. Association for Computational Linguistics.
John M Ford. 2004. Content analysis: An introduc-
tion to its methodology.Personnel psychology,
57(4):1110.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The Llama 3 Herd
of Models.arXiv preprint arXiv:2407.21783.
Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan,
Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen,
Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun
Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni, and
Jian Guo. 2025. A survey on llm-as-a-judge.arXiv,
abs/2411.15594.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang,
Shirong Ma, Xiao Bi, and 1 others. 2025. Deepseek-
r1 incentivizes reasoning in llms through reinforce-
ment learning.Nature, 645(8081):633–638.
9

Ashim Gupta, Maitrey Mehta, Zhichao Xu, and Vivek
Srikumar. 2025. Found in translation: Measuring
multilingual llm consistency as simple as translate
then evaluate.arXiv preprint arXiv:2505.21999.
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam
Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow,
Akila Welihinda, Alan Hayes, Alec Radford, and 1
others. 2024. GPT-4o System Card.arXiv preprint
arXiv:2410.21276.
Yiqiao Jin, Mohit Chandra, Gaurav Verma, Yibo Hu,
Munmun De Choudhury, and Srijan Kumar. 2024.
Better to ask in english: Cross-lingual evaluation
of large language models for healthcare queries. In
Proceedings of the ACM Web Conference 2024, page
2627–2638, New York, NY , USA. Association for
Computing Machinery.
Navreet Kaur, Monojit Choudhury, and Danish Pruthi.
2024. Evaluating large language models for health-
related queries with presuppositions. InFindings of
the Association for Computational Linguistics: ACL
2024, pages 14308–14331, Bangkok, Thailand. As-
sociation for Computational Linguistics.
Yubin Kim, Hyewon Jeong, Shan Chen, Shuyue Stella
Li, Mingyu Lu, Kumail Alhamoud, Jimin Mun,
Cristina Grau, Minseok Jung, Rodrigo Gameiro, and
1 others. 2025. Medical hallucinations in founda-
tion models and their impact on healthcare.arXiv
preprint arXiv:2503.05777.
Bevan Koopman and Guido Zuccon. 2023. Dr ChatGPT
tell me what I want to hear: How different prompts
impact health answer correctness. InProceedings
of the 2023 Conference on Empirical Methods in
Natural Language Processing, pages 15012–15022,
Singapore. Association for Computational Linguis-
tics.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.
Gonzalez, Hao Zhang, and Ion Stoica. 2023. Effi-
cient memory management for large language model
serving with pagedattention. InProceedings of the
ACM SIGOPS 29th Symposium on Operating Systems
Principles.
N Lawrentschuk, R Abouassaly, E Hewitt, A Mulc-
ahy, DM Bolton, and T Jobling. Health information
quality on the internet in gynecological oncology:
a multilingual evaluation.Eur. J. Gynaecol. Oncol,
37(4):2016.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InProceedings of the 34th Inter-
national Conference on Neural Information Process-
ing Systems, NIPS ’20, Red Hook, NY , USA. Curran
Associates Inc.Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Man-
dar Joshi, Danqi Chen, Omer Levy, Mike Lewis,
Luke Zettlemoyer, and Veselin Stoyanov. 2019.
Roberta: A robustly optimized bert pretraining ap-
proach.arXiv preprint arXiv:1907.11692.
Xing Han Lù. 2024. Bm25s: Orders of magnitude faster
lexical search via eager sparse scoring.
Hellina Hailu Nigatu, Nuredin Ali Abdelkadir, Fiker
Tewelde, Stevie Chancellor, and Daricia Wilkinson.
2025. Into the Void: Understanding Online Health
Information in Low-Web Data Languages.arXiv
preprint arXiv:2509.20245.
OpenAI. 2025. Introducing gpt-5. https://openai.
com/index/introducing-gpt-5/ . Accessed 2025-
09-29.
Chan Young Park, Xinru Yan, Anjalie Field, and Yulia
Tsvetkov. 2021. Multilingual Contextual Affective
Analysis of LGBT People Portrayals in Wikipedia.
InProceedings of the International AAAI Conference
on Web and Social Media, volume 15, pages 479–
490.
Surangika Ranathunga and Nisansa de Silva. 2022.
Some languages are more equal than others: Prob-
ing deeper into the linguistic disparity in the NLP
world. InProceedings of the 2nd Conference of the
Asia-Pacific Chapter of the Association for Compu-
tational Linguistics and the 12th International Joint
Conference on Natural Language Processing (Vol-
ume 1: Long Papers), pages 823–848, Online only.
Association for Computational Linguistics.
David Restrepo, Chenwei Wu, Zhengxu Tang, Zi-
tao Shuai, Thao Nguyen Minh Phan, Jun-En Ding,
Cong-Tinh Dao, Jack Gallifant, Robyn Gayle Dy-
chiao, Jose Carlo Artiaga, André Hiroshi Bando, Car-
olina Pelegrini Barbosa Gracitelli, Vincenz Ferrer,
Leo Anthony Celi, Danielle Bitterman, Michael G
Morley, and Luis Filipe Nakayama. 2025. Multi-
ophthalingua: A multilingual benchmark for assess-
ing and debiasing llm ophthalmological qa in lmics.
Proceedings of the AAAI Conference on Artificial
Intelligence, 39(27):28321–28330.
Amanda Ross and Victor L Willson. 2017. Paired sam-
ples t-test. InBasic and advanced statistical tests:
Writing results sections and creating tables and fig-
ures, pages 17–19. Springer.
Farhan Samir, Chan Young Park, Anjalie Field, Vered
Shwartz, and Yulia Tsvetkov. 2024. Locating in-
formation gaps and narrative inconsistencies across
languages: A case study of LGBT people portrayals
on Wikipedia. InProceedings of the 2024 Confer-
ence on Empirical Methods in Natural Language
Processing, pages 6747–6762, Miami, Florida, USA.
Association for Computational Linguistics.
Burcu Sayin, Pasquale Minervini, Jacopo Staiano, and
Andrea Passerini. 2024. Can LLMs correct physi-
cians, yet? investigating effective interaction meth-
ods in the medical domain. InProceedings of the
10

6th Clinical Natural Language Processing Workshop,
pages 218–237, Mexico City, Mexico. Association
for Computational Linguistics.
Ipek Baris Schlicht, Eugenia Fernandez, Berta Chulvi,
and Paolo Rosso. 2024. Automatic detection of
health misinformation: a systematic review.Journal
of Ambient Intelligence and Humanized Computing,
15(3):2009–2021.
Ipek Baris Schlicht, Zhixue Zhao, Burcu Sayin, Lu-
cie Flek, and Paolo Rosso. 2025. Do llms provide
consistent answers to health-related questions across
languages? InAdvances in Information Retrieval
(ECIR 2025), pages 314–322, Cham. Springer Na-
ture Switzerland.
Stefan Schweter. 2020. BERTurk - BERT models
for Turkish. https://doi.org/10.5281/zenodo.
3770924.
Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara
Mahdavi, Jason Wei, Hyung Won Chung, Nathan
Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen
Pfohl, Perry Payne, Martin Seneviratne, Paul Gam-
ble, Chris Kelly, Abubakr Babiker, Nathanael Schärli,
Aakanksha Chowdhery, Philip Mansfield, Dina
Demner-Fushman, and 13 others. 2023. Large lan-
guage models encode clinical knowledge.Nature,
620(7972):172–180.
Aaron A Tierney, Mary E Reed, Richard W Grant, Flo-
rence X Doo, Denise D Payán, and Vincent X Liu.
2025. Health Equity in the Era of Large Language?
Models.American Journal of Managed Care, 31(3).
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, Dan Bikel, Lukas Blecher, Cristian Can-
ton Ferrer, Moya Chen, Guillem Cucurull, David
Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, and
49 others. 2023. Llama 2: Open Foundation and
Fine-Tuned Chat Models.arXiv, abs/2307.09288.
Zining Wang, Yuxuan Zhang, Dongwook Yoon,
Nicholas Vincent, Farhan Samir, and Vered Shwartz.
2025. Wikigap: Promoting epistemic equity by sur-
facing knowledge gaps between english wikipedia
and other language editions.arXiv preprint
arXiv:2505.24195.
Christian Weissenberger, S Jonassen, J Beranek-Chiu,
M Neumann, D Müller, S Bartelt, S Schulz, JS Mönt-
ing, K Henne, G Gitsch, and 1 others. 2004. Breast
cancer: patient information needs reflected in English
and German web sites.British Journal of Cancer,
91(8):1482–1487.
Clark Wissler. 1905. The spearman correlation formula.
Science, 22(558):309–311.
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien
Chaumond, Clement Delangue, Anthony Moi, Pier-
ric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz,
Joe Davison, Sam Shleifer, Patrick von Platen, ClaraMa, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le
Scao, Sylvain Gugger, and 3 others. 2020. Trans-
formers: State-of-the-art natural language processing.
InEMNLP (Demos), pages 38–45. Association for
Computational Linguistics.
Fangyuan Xu, Junyi Jessy Li, and Eunsol Choi. 2022.
How do we answer complex questions: Discourse
structure of long-form answers. InProceedings of the
60th Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
3556–3572, Dublin, Ireland. Association for Compu-
tational Linguistics.
Niraj Yagnik, Jay Jhaveri, Vivek Sharma, and Gabriel
Pila. 2024. Medlm: Exploring language models for
medical question answering systems.arXiv preprint
arXiv:2401.11389.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
Tian Yang and Susana Valdez. 2025. How machine
translation is used in healthcare.Digital Translation.
Haoran Yu, Chang Yu, Zihan Wang, Dongxian Zou, and
Hao Qin. 2024. Enhancing Healthcare through Large
Language Models: A Study on Medical Question
Answering. In2024 IEEE 6th International Confer-
ence on Power, Intelligent Computing and Systems
(ICPICS), pages 895–900. IEEE.
Yuheng Zha, Yichi Yang, Ruichen Li, and Zhiting Hu.
2023. AlignScore: Evaluating factual consistency
with a unified alignment function. InProceedings
of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 11328–11348, Toronto, Canada. Association
for Computational Linguistics.
Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu,
Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang,
Yulong Chen, and 1 others. 2025. Siren’s song in the
ai ocean: A survey on hallucination in large language
models.Computational Linguistics, pages 1–46.
Ming Zhu, Aman Ahuja, Wei Wei, and Chandan K.
Reddy. 2019. A hierarchical attention retrieval model
for healthcare question answering. InThe World
Wide Web Conference, WWW ’19, page 2472–2482,
New York, NY , USA. Association for Computing
Machinery.
11
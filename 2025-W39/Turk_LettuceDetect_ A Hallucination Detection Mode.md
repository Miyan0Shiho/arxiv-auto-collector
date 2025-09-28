# Turk-LettuceDetect: A Hallucination Detection Models for Turkish RAG Applications

**Authors**: Selva Taş, Mahmut El Huseyni, Özay Ezerceli, Reyhan Bayraktar, Fatma Betül Terzioğlu

**Published**: 2025-09-22 12:14:11

**PDF URL**: [http://arxiv.org/pdf/2509.17671v1](http://arxiv.org/pdf/2509.17671v1)

## Abstract
The widespread adoption of Large Language Models (LLMs) has been hindered by
their tendency to hallucinate, generating plausible but factually incorrect
information. While Retrieval-Augmented Generation (RAG) systems attempt to
address this issue by grounding responses in external knowledge, hallucination
remains a persistent challenge, particularly for morphologically complex,
low-resource languages like Turkish. This paper introduces Turk-LettuceDetect,
the first suite of hallucination detection models specifically designed for
Turkish RAG applications. Building on the LettuceDetect framework, we formulate
hallucination detection as a token-level classification task and fine-tune
three distinct encoder architectures: a Turkish-specific ModernBERT,
TurkEmbed4STS, and multilingual EuroBERT. These models were trained on a
machine-translated version of the RAGTruth benchmark dataset containing 17,790
instances across question answering, data-to-text generation, and summarization
tasks. Our experimental results show that the ModernBERT-based model achieves
an F1-score of 0.7266 on the complete test set, with particularly strong
performance on structured tasks. The models maintain computational efficiency
while supporting long contexts up to 8,192 tokens, making them suitable for
real-time deployment. Comparative analysis reveals that while state-of-the-art
LLMs demonstrate high recall, they suffer from low precision due to
over-generation of hallucinated content, underscoring the necessity of
specialized detection mechanisms. By releasing our models and translated
dataset, this work addresses a critical gap in multilingual NLP and establishes
a foundation for developing more reliable and trustworthy AI applications for
Turkish and other languages.

## Full Text


<!-- PDF content starts -->

Turk-LettuceDetect: A Hallucination Detection
Models for Turkish RAG Applications
Selva Tas ¸, Mahmut El Huseyni, ¨Ozay Ezerceli, Reyhan Bayraktar, Fatma Bet ¨ul Terzio ˘glu
Newmind AI
Istanbul, T ¨urkiye
{stas,mehussieni,oezerceli,rbayraktar,fbterzioglu@newmind.ai}@newmind.ai
Abstract—The widespread adoption of Large Language Models
(LLMs) has been hindered by their tendency to hallucinate,
generating plausible but factually incorrect information. While
Retrieval-Augmented Generation (RAG) systems attempt to ad-
dress this issue by grounding responses in external knowledge,
hallucination remains a persistent challenge, particularly for
morphologically complex, low-resource languages like Turkish.
This paper introduces Turk-LettuceDetect, the first suite of
hallucination detection models specifically designed for Turkish
RAG applications. Building on the LettuceDetect framework, we
formulate hallucination detection as a token-level classification
task and fine-tune three distinct encoder architectures: a Turkish-
specific ModernBERT, TurkEmbed4STS, and multilingual Eu-
roBERT. These models were trained on a machine-translated
version of the RAGTruth benchmark dataset containing 17,790
instances across question answering, data-to-text generation, and
summarization tasks. Our experimental results show that the
ModernBERT-based model achieves an F1-score of 0.7266 on
the complete test set, with particularly strong performance on
structured tasks. The models maintain computational efficiency
while supporting long contexts up to 8,192 tokens, making them
suitable for real-time deployment. Comparative analysis reveals
that while state-of-the-art LLMs demonstrate high recall, they
suffer from low precision due to over-generation of hallucinated
content, underscoring the necessity of specialized detection mech-
anisms. By releasing our models and translated dataset, this work
addresses a critical gap in multilingual NLP and establishes
a foundation for developing more reliable and trustworthy AI
applications for Turkish and other languages.
Index Terms—turkish language detection, retrieval-augmented
generation, hallucination detection, large language models, token
classification
I. INTRODUCTION
The rapid advancement of Large Language Models (LLMs)
has revolutionized natural language processing (NLP) appli-
cations, demonstrating remarkable capabilities in text genera-
tion, question answering, and reasoning tasks [30]. However,
their propensity to generate plausible-sounding but factually
incorrect information, commonly referred to as hallucinations,
remains a critical challenge that limits their deployment in
real-world applications where accuracy and reliability are
paramount [4], [5].
Retrieval-Augmented Generation (RAG) has emerged as
a promising paradigm to mitigate hallucination issues by
grounding LLM responses in external knowledge sources [11],
[17]. By retrieving relevant documents and incorporating them
as context during generation, RAG systems aim to anchor
model outputs in factual information rather than relying solelyon parametric knowledge. Despite these architectural im-
provements, empirical studies have shown that RAG systems
still exhibit significant hallucination rates, particularly when
dealing with complex queries or when the retrieved context is
incomplete or contradictory [6], [9].
The challenge of hallucination detection becomes even more
pronounced in low-resource languages where training data
is scarce and evaluation benchmarks are limited. Turkish,
despite being spoken by over 80 million people worldwide,
represents one such language where robust hallucination de-
tection approachs are critically needed but underexplored [31].
The linguistic complexity of Turkish, characterized by its
agglutinative morphology and rich inflectional system, poses
additional challenges for accurate hallucination detection com-
pared to morphologically simpler languages like English [15].
Existing hallucination detection approaches can be broadly
categorized into two paradigms: prompt-based methods that
leverage large language models as judges [16], [18], and fine-
tuned encoder-based models that perform token-level classifi-
cation [10], [19]. While prompt-based methods offer flexibility
and can achieve high accuracy, they suffer from computational
inefficiency and high inference costs due to their reliance
on large models. Conversely, traditional encoder-based ap-
proaches, while computationally efficient, are constrained by
limited context windows that prevent them from processing
long documents typical in RAG applications.
Recent advances in encoder architectures have begun to ad-
dress these limitations. ModernBERT [12], building upon the
foundational BERT architecture [20], incorporates several key
innovations including rotary positional embeddings (RoPE)
[26] and local-global attention mechanisms that enable pro-
cessing of sequences up to 8,192 tokens. These architectural
improvements make ModernBERT particularly well-suited for
hallucination detection in RAG systems, where long context
understanding is essential for accurate verification of generated
content against source documents.
Drawing from the recently developed LettuceDetect frame-
work [1], we introduce Turk-LettuceDetect, a tailored vari-
ant optimized for Turkish RAG systems. The original Let-
tuceDetect framework showed substantial advancements in
identifying hallucinations within English content and was
subsequently expanded to accommodate various languages
such as German, French, Spanish, Italian, Polish, and Chinese
through EuroBERT integration. It achieved a 79.22% F1arXiv:2509.17671v1  [cs.CL]  22 Sep 2025

score on the RAGTruth benchmark, which marks a 14.8%
enhancement compared to earlier encoder-based methods in
English. Nevertheless, a significant void persists in cross-
linguistic hallucination detection, especially for languages with
complex morphology and limited resources such as Turkish.
Our approach extends the core LettuceDetect methodology
by adapting three different model base such as ModernBERT,
gte-multilingual based TurkEmbed4STS and EuroBERT on to-
ken classification to handle the unique linguistic characteristics
of Turkish while maintaining the computational efficiency of
the original framework. We fine-tuned Turkish-specific models
on machine translated versions of the RAGTruth dataset [6],
formulating the task as a binary token classification problem
where each token in the generated answer is labeled as either
supported or hallucinated based on the given context and
question.
The main contributions of this paper are threefold:
1) We presentTurk-LettuceDetect, the first adaptation of
the LettuceDetect for Turkish RAG applications with
three open-source models, addressing a significant gap
in multilingual hallucination detection research.
2) We demonstrate that the ModernBERT-based archi-
tecture can maintain competitive performance when
adapted to Turkish text, preserving the computational
efficiency advantages of the original framework while
handling the morphological complexity of Turkish.
3) We provide extensive experimental validation across
multiple task types including question answering, data-
to-text generation, and summarization, showing that our
Turkish-specific models achieve consistent performance
improvements over baseline multilingual approaches.
4) The translated Turkish-RAGTruth dataset and the fine-
tuned hallucination-detection models have been released
under an open-source license to support and accelerate
future research.
The multilingual extensions built on diverse model back-
bones demonstrate the approach’s adaptability to cross-lingual
contexts, thereby paving the way for broader multilingual
deployment of the LettuceDetect methodology.
The remainder of this paper is organized as follows: Sec-
tion II reviews related work; Section III describes the dataset
and methodology; Section IV presents experimental results
and comparisons; Section V concludes the paper with a
discussion of future directions.
II. RELATEDWORK
Hallucinations are prevalent in both general-purpose LLMs
and RAG systems [4], [5]. These phenomena have been exten-
sively studied in recent literature, including the comprehensive
survey by Ahadian and Guan [14], which identifies two major
causes: (1) training data biases and memorization effects,
and (2) decoding time errors such as sampling artifacts or
insufficient grounding.
In RAG systems, hallucinations can arise due to misinterpre-
tation of retrieved evidence, retrieval inaccuracies, or misalign-
ment between retrieved documents and the user’s query [6],[11]. Several mitigation strategies have been proposed, broadly
categorized intoprompt-basedandmodel-basedapproaches.
Prompt-based methods leverage LLMs as judges to evaluate
factual consistency using structured prompts or reasoning
chains [16], [18]. While flexible and easy to implement, they
suffer from high computational cost, lack of interpretability,
and inconsistent judgments across model versions [19].
Model-based approaches, on the other hand, fine-tune
smaller encoder-based models to detect hallucinated spans di-
rectly. These methods offer better efficiency and reproducibil-
ity, making them suitable for real-time deployment. Notable
examples includeLuna[10], which uses a classification head
over contextual embeddings to identify unsupported content,
andLettuceDetect[1], which extends this idea with token-
level classification and achieves significant improvements in
precision and recall over prior encoder-based baselines.
ModernBERT [12], building upon BERT’s foundational
design [20], incorporates several innovations such as rotary
positional embeddings (RoPE) [26] and local-global attention
mechanisms, enabling it to process sequences up to 8,192
tokens. This makes it especially well-suited for hallucination
detection in RAG systems, where long context understanding
is crucial for accurate verification against source documents.
The development of robust and innovative Turkish embed-
ding models for downstream tasks such as semantic textual
similarity (STS) and information retrieval necessitates the ex-
ploration and deployment of novel backbone architectures tai-
lored to the unique characteristics of the Turkish language. Re-
cent contributions, including TurkEmbed4STS and TurkEm-
bed4Retrieval [2], have demonstrated the critical importance of
domain-specific fine-tuning for Turkish retrieval applications,
while simultaneously highlighting the need for architectural
innovations to achieve optimal performance across diverse
downstream tasks. Building upon these foundational insights,
our work extends this paradigm by adapting state-of-the-art
encoder architectures specifically for hallucination detection in
Turkish RAG applications, thereby addressing both the inher-
ent morphological complexity of Turkish and the limitations
associated with multilingual transfer learning approaches.
Unlike traditional NLI-based approaches, our method is en-
tirely trained on the machine-translated version of RAGTruth
dataset [6], using ModernBERT, GTE-multilingual-base and
EuroBERT base architectures. We also incorporate multilin-
gual extensions of the dataset, translating it using Gemma3-
27b-it [27] for cross-lingual generalization.
Our results show that efficient encoder-based hallucination
detection can rival prompt-based methods while maintaining
inference speed and scalability, an important step toward
deploying RAG systems in low-resource, high-stakes environ-
ments.
III. METHODOLOGY
This section presents our methodology for hallucination
detection within RAG systems. We employed the LettuceDe-
tect framework to train hallucination detection models using
three distinct Turkish-supported encoder architectures. The

first model, ModernBERT-base-tr, was specifically fine-tuned
for this research on Turkish Natural Language Inference
(NLI) and STS downstream tasks, following the established
paradigms and methodologies demonstrated by the TurkEm-
bed approach. The second model is TurkEmbed4STS model
(gte-multilingual-based base) and the third model utilized was
euroBERT, an established pre-existing multilingual encoder.
This experimental design enabled a comprehensive evaluation
of hallucination detection capabilities across diverse encoder
backbones within the Turkish linguistic context.
A. Dataset
We utilized the RAGTruth dataset [6] for both training and
evaluation of our models. RAGTruth constitutes the first large-
scale benchmark specifically constructed to evaluate halluci-
nation phenomena in RAG settings. The dataset comprises
annotated 17.790 train, 2.700 test instances, covering three
distinct tasks: question answering, data-to-text generation, and
summarization.
For the question answering task, instances were sourced
from the MS MARCO dataset [24]. Each question was
matched with up to three context passages retrieved via
information retrieval techniques, and LLMs were prompted
to generate answers based on these contexts. In the data-to-
text generation task, models produced reviews for businesses
selected from the Yelp Open Dataset [25]. For the news sum-
marization task, documents were randomly sampled from the
training set of the CNN/Daily Mail corpus [28], and the LLMs
were asked to generate abstractive summaries accordingly.
A diverse set of LLMs was employed for response genera-
tion, including GPT-4-0613 [22], Mistral-7B-Instruct [23], and
several models from the LLaMA family, such as LLaMA2-
7B-Chat and LLaMA2-13B-Chat [3]. Each data point in the
dataset includes responses from six different models, allowing
for multi-model comparison at the example level. All samples
were manually annotated by human experts, who identified
hallucinated spans within the model outputs and provided
rationale for their decisions. The dataset further classifies
hallucinations into four distinct categories: Evident Conflict,
Subtle Conflict, Evident Introduction of Baseless Information,
and Subtle Introduction of Baseless Information. However, for
the purposes of our model training, we simplified this classi-
fication to a binary hallucination detection task, disregarding
the specific hallucination types.
An analysis of token lengths within the dataset revealed
a mean input length of 801 tokens, a median of 741, with
lengths ranging from 194 to 2,632 tokens. These findings
underscore the need for long-context language models which
has context window bigger than 4096 tokens to effectively
identify hallucinations, especially in lengthy and context-rich
inputs.
B. Multilingual Extension of RAGTruth Dataset
To address the monolingual limitation of the RAGTruth
dataset, we developed a multilingual extension by translating
its content into multiple target languages by following the waythet is translated to other european languages. This subsection
details the translation protocols, and a sample of the translated
dataset.
1) Translation Methodology:The translation pipeline uti-
lized thegoogle/gemma-3-27b-itmodel, executed via
vLLM on a single NVIDIA A100 GPU. This configuration
supported parallel processing of approximately 30 examples,
with a full translation pass for target language completed in
roughly 12 hours.
2) Translation Protocols:The translation protocols in this
study were designed to maintain structural integrity while
ensuring accurate cross-lingual transfer of both content and
metadata. Two distinct translation procedures were imple-
mented: one for answer content and another for prompt
instructions, each tailored to handle specific linguistic and
structural requirements.
Core Translation Prompt
Translate the following text from
{source_lang} to {target_lang}. If the
original text contains <HAL> tags, translate
the content inside <HAL> tags and ensure the
number of the <HAL> tags remain exactly the
same in the output. If the original text does
not contain <HAL> tags, just translate the
text. Do NOT add any <HAL> tags if they were
not in the original text. Do NOT remove any
<HAL> tags that were in the original text. Do
not include any additional sentences
summarizing or explaining the translation.
Your output should be just the translated
text, nothing else.
a) Answer Translation Protocol:The answer translation
protocol was specifically designed to handle hallucination-
annotated content while preserving the evaluation framework’s
structural requirements. When translating answers from source
language to target language, the following systematic approach
was employed:
•Tag Preservation:The exact number and positioning of
<HAL>tags from the source text were maintained, with
translation applied exclusively to the content within these
tags. This ensures that hallucination annotations remain
consistent across languages for comparative analysis.
•Content-Only Translation:For text segments without
<HAL>tags, direct translation was performed while
maintaining the original semantic meaning and contextual
nuances.
•Structural Integrity:No<HAL>tags were introduced
unless present in the source material, and no existing
tags were removed, ensuring the hallucination detection
framework remains intact across all translated versions.
•Output Specification:Only the translated text was pro-
duced, excluding any meta-commentary, explanations, or
translation notes that could interfere with subsequent
automated processing.
b) Prompt Translation Protocol:The prompt translation
protocol addressed the challenge of translating instruction
sets while maintaining their functional effectiveness across

different linguistic contexts. For translating prompts from
source language to target language, the following specialized
methodology was implemented:
•Comprehensive Content Translation:All prompt com-
ponents were translated, including both natural language
instructions and structured elements such as JSON ob-
jects, where both keys and values underwent linguistic
transformation to ensure cultural and linguistic appropri-
ateness.
•Functional Equivalence:Translation prioritized main-
taining the prompt’s intended function and directive clar-
ity rather than literal word-for-word conversion, ensuring
that the translated prompts elicit equivalent responses
from language models.
•Clean Output Generation:The translation process pro-
duced only the target language prompt without supple-
mentary explanatory text, facilitating direct integration
into the experimental pipeline.
These protocols collectively ensure that the cross-lingual
evaluation maintains both the semantic integrity of the original
content and the structural requirements necessary for consis-
tent hallucination detection across multiple language contexts.
3) Sample Translated Data:Table I provides an example
of translated RAGTruth data, including a question, reference
passage, response, and associated annotation, demonstrating
the application of the translation protocols.
C. Model Architecture
We propose a token-level hallucination detection pipeline
based on three transformer-based encoder architectures: Mod-
ernBERT [12], TurkEmbed4STS, and euroBERT. The frame-
work formulates hallucination detection as a binary token
classification task, where each token in the generated response
is classified as either supported or unsupported by the provided
context.
The classification head produces binary predictions for each
token in the input sequence. Training minimizes the cross-
entropy loss across all tokens in the sequence, with appropriate
masking for special tokens and padding.
Unlike previous encoder-based approaches that require NLI
pretraining or auxiliary task transfers, our architecture operates
directly on the base transformer representations. This design
choice eliminates the need for multi-stage training pipelines
and cross-task knowledge transfer, resulting in a more stream-
lined and reproducible system. The architecture’s simplicity
facilitates deployment in production environments while main-
taining competitive performance on hallucination detection
tasks. The token-level granularity enables fine-grained identifi-
cation of hallucinated content, providing interpretable outputs
for downstream applications requiring precise localization of
factual inconsistencies.
D. Training Configuration
We fine-tuned three transformer-based models
ModernBERT-base-tr, TurkEmbed4STS, and lettucedect-
210m-eurobert-tr-v1 as token-level classifiers on theTABLE I
EXAMPLE OFTRANSLATEDRAGTRUTHDATA
PROMPT Summarize the following news within 116 words:
Seventy years ago, Anne Frank died of typhus in a Nazi
concentration camp at the age of 15. Just two weeks after
her supposed death on March 31, 1945, the Bergen-Belsen
concentration camp where she had been imprisoned was
liberated – timing that showed how close the Jewish diarist
had been to surviving the Holocaust. But new research
released by the Anne Frank House shows that Anne and
her older sister, Margot Frank, died at least a month earlier
than previously thought. . . .
ANSWERNew research conducted by the Anne Frank House has
revealed that Anne Frank and her sister Margot likely
died in the Bergen-Belsen concentration camp at least a
month earlier than previously believed. The researchers
examined archives of the Red Cross, the International
Training Service, and the Bergen-Belsen Memorial, as
well as testimonies from survivors. . . .
ANNOTATION labels:{start: 636, end: 653, text: “February 7, 2022.”,
type: Evident Conflict},{start: 871, end: 969, text: “has
prompted...”, type: Evident Baseless Info},{start: 607,
end: 646, text: “believed to have died before February 7”,
type: Evident Conflict}
Type:Summary
split:train
PROMPT As ¸a ˘gıdaki haberi 116 kelimeyle ¨ozetleyin:
Yetmis ¸ yıl ¨once Anne Frank, Nazi toplama kampında
15 yas ¸ında tifo nedeniyle ¨old¨u. 31 Mart 1945’te ¨old¨u˘g¨u
varsayılan tarihten sadece iki hafta sonra, tutuklu oldu ˘gu
Bergen-Belsen toplama kampı kurtarıldı; bu zamanlama,
Yahudi g ¨unl¨uk yazarının Holokost’tan kurtulmaya ne
kadar yakın oldu ˘gunu g ¨osterdi. Ancak Anne Frank Evi
tarafından yayınlanan yeni aras ¸tırmalar, Anne ve ablası
Margot Frank’ın daha ¨once d ¨us ¸¨un¨uld¨u˘g¨unden en az bir
ay¨once ¨old¨u˘g¨un¨u g¨osteriyor. . . .
ANSWERAnne Frank Evi tarafından yapılan yeni bir aras ¸tırma,
Anne Frank ve kız kardes ¸i Margot’un daha ¨once
inanıldı ˘gından en az bir ay ¨once Bergen-Belsen
toplama kampında ¨olm¨us ¸ olabilece ˘gini ortaya koydu.
Aras ¸tırmacılar, Kızıl Hac ¸, Uluslararası E ˘gitim Hizmeti ve
Bergen-Belsen Anıtı ars ¸ivlerinin yanı sıra kurtulanların
ifadelerini inceledi. . . .
ANNOTATION labels:{“start”: 545, “end”: 596, “label”: “Evident Con-
flict”},{“start”: 824, “end”: 906, “label”: “Evident Base-
less Info”}
Type:Summary
split:train
language:tr
RAGTruth dataset. To focus on answer tokens, context
and question tokens were masked (label = -100), while
answer tokens were assigned labels of 0 (supported) or 1
(hallucinated) based on human annotations. Each model
underwent independent fine-tuning with identical supervision
to enable a fair comparison of their token-level hallucination
detection performance under consistent data and labeling
conditions.
Training was conducted over 6 epochs with a learning rate
of1×e−5and a batch size 4. All training was performed on
an NVIDIA A100 40GB GPU. Each epoch took around 20
minutes and whole training took 2 hours for each model.
IV. EVALUATION
This section presents a comprehensive evaluation of our
hallucination detection models across multiple tasks and set-

tings. We assess model performance on the RAGTruth dataset
test split and analyze the hallucination behavior of various
decoder-only models and encoders to demonstrate the effec-
tiveness of our approach.
A. Experimental Setup
We conducted a comprehensive evaluation of our models
using the test split of the RAGTruth dataset, encompass-
ing three core task types: question answering (QA), data-
to-text generation, and summarization. To ensure a nuanced
assessment, model performance was analyzed at both the
example level and the token level, which provides a fine-
grained perspective on model behavior within each generated
text.
a) Evaluation Metrics:To rigorously assess hallucina-
tion detection capabilities, we employed a multi-metric eval-
uation framework:
•Precision: The proportion of predicted hallucinated (or
supported) instances that are actually correct, reflecting
the model’s ability to avoid false positives.
•Recall: The proportion of actual hallucinated (or sup-
ported) instances that are correctly identified by the
model, capturing sensitivity to true cases.
•Macro F1-Score: Calculated separately for hallucinated
and supported instances, this metric provides a balanced
measure of precision and recall, ensuring fair assessment
across both prediction types.
•Area Under the Receiver Operating Characteristic Curve
(AUROC): Summarizes model performance across all
classification thresholds, offering an aggregate measure
of discriminative ability.
This multi-level, multi-metric approach enables a robust and
multifaceted evaluation of model performance across diverse
aspects of hallucination detection in natural language gener-
ation tasks. All experiments were conducted using the latest
stable versions of the lettucedetect and datasets libraries, and
computations were performed on an NVIDIA A100 GPU with
40GB of memory, ensuring both methodological transparency
and reproducibility.
B. Comparative Model Performance and LLM Hallucination
Behavior
We evaluated three encoder-based models for hallucination
detection in Turkish RAGTruth:
•modernbert-base-tr-uncased-stsb-HD: A Turkish-
specific ModernBERT variant fine-tuned on hallucination
detection.
•TurkEmbed4STS-HallucinationDetection: A Turkish
embedding model optimized for semantic similarity and
adapted to hallucination detection.
•lettucedect-210m-eurobert-tr-v1: A multilingual Eu-
roBERT variant fine-tuned on hallucination detection.
The results, summarized in Figure 1, reveal distinct per-
formance patterns across models and tasks. Among encoder-
based models,TurkEmbed4STS-HallucinationDetectionshowsTABLE II
PERFORMANCE OFTOKEN-LEVELHALLUCINATIONDETECTIONACROSS
MODELS
Model Task Type Precision Recall F1-Score AUROC
ModernBERT-base-tr
Summary0.6935 0.5705 0.6007 0.5705
Data2txt 0.7652 0.7182 0.7391 0.7182
QA0.7642 0.7536 0.7588 0.7536
Whole Dataset0.7583 0.7024 0.7266 0.7024
TurkEmbed4STS
Summary 0.6325 0.5656 0.5862 0.5656
Data2txt 0.73970.73330.73650.7333
QA 0.7378 0.7382 0.7380 0.7382
Whole Dataset 0.7268 0.7014 0.7132 0.7014
lettucedect-210m-eurobert-tr
Summary 0.6465 0.5546 0.5771 0.5546
Data2txt0.78660.72180.74960.7218
QA 0.7388 0.7262 0.7323 0.7262
Whole Dataset 0.7511 0.6908 0.7163 0.6908
the most consistent behavior, with relatively balanced pre-
cision and recall across all tasks.LettuceDetect-210m-
EuroBERT-Tr-v1achieves the highest AUROC in data-to-
text generation (0.8966), indicating strong discriminative
power between supported and hallucinated content. Mean-
while,modernbert-base-tr-uncased-stsb-HDperforms best in
QA (AUROC = 0.8833), suggesting that Turkish-specific pre-
training improves performance in structured tasks.
However, all encoder-based models show reduced effective-
ness in summarization, particularly in detecting hallucinated
tokens, with F1 scores below 0.65. This highlights the need
for improved handling of abstractive generation in future
hallucination detection frameworks.
When analyzing LLM hallucination behavior, we observe
that GPT-4.1 and Mistral models achieve high recall (up
to 0.9938), indicating a strong tendency to generate content
flagged as hallucinated. However, their precision remains
low, suggesting over-generation or systematic hallucination
patterns.
In contrast, Qwen3-14B demonstrates the best overall bal-
ance, achieving the highest F1-score (0.7429) on the full
dataset. It performs particularly well in data-to-text generation
(precision = 0.8255), while QwenQ-32B excels in QA with the
highest AUROC (0.7267).
These findings underscore the critical necessity of imple-
menting specialized hallucination detection frameworks, such
as Lettuce Detect, in conjunction with fine-tuned language
models particularly within production-level Turkish RAG ap-
plications where maintaining factual accuracy and compu-
tational efficiency represents paramount operational require-
ments.
C. Token-Level Hallucination Detection
Table II presents a detailed breakdown of hallucination
detection performance at the token level across different
models and task types. The results reveal several important

Summary Data2txt QA Whole Dataset
Task0.00.20.40.60.81.0PrecisionPrecision Performance Comparison
Summary Data2txt QA Whole Dataset
Task0.00.20.40.60.81.0RecallRecall Performance Comparison
Summary Data2txt QA Whole Dataset
Task0.00.20.40.60.81.0F1-ScoreF1-Score Performance Comparison
Summary Data2txt QA Whole Dataset
Task0.00.20.40.60.81.0AUROCAUROC Performance ComparisonModernBERT-base-tr
TurkEmbed4STSlettucedect-210m-...
GPT-4.1Qwen3-14B
QwenQ-32BMistral Small 3.2
Mistral Small 3.1Fig. 1. Performance of example-level hallucination detection across models
patterns in how well models distinguish between supported
and hallucinated content within generated responses.
First, all models demonstrate relatively strong performance
on supported tokens (Class 0), with precision values above
0.63 and recall above 0.72. This indicates that the models are
generally effective at identifying content that is grounded in
the provided context. However, there is a noticeable drop in
performance when detecting hallucinated tokens (Class 1), par-
ticularly in summarization tasks. For example,modernbert-
base-tr-uncased-stsb-HDachieves only 0.6935 precision in
summarization, suggesting that hallucinated content is more
challenging to identify in this domain.
Second, among the evaluated models,modernbert-base-
tr-uncased-stsb-HDconsistently delivers the most balanced
performance across both classes. In question answering, it
achieves a precision of 0.7642 and a recall of 0.7536, showing
its ability to detect hallucinations without sacrificing support
accuracy. This supports our earlier findings that multilingual
pre-training and fine-tuning enhances robustness and general-
ization across RAG tasks.
Third, whilemodernbert-base-tr-uncased-stsb-HDper-
forms best in summary, QA and whole dataset it shows weaker
hallucination detection in data2txt, where best precision andF1-score achieved bylettucedect-210m-eurobert-trand best
recall and AUROC achieved byTurkEmbed4STSmodels.
This suggests that Turkish-specific pretraining helps in struc-
tured tasks but may require further adaptation for abstractive
generation settings.
Interestingly,TurkEmbed4STSexhibits the most consistent
behavior across all tasks, with precision and recall values
staying relatively close. Although not always top-performing,
it avoids extreme imbalances between supported and hallu-
cinated class detection, making it a promising candidate for
applications requiring stability over peak performance.
Finally, the overall token-level scores demonstrate that
encoder-based models can reliably detect hallucinations in
RAG outputs. More importantly, the high AUROC values
confirm that these models possess strong discriminative power
between supported and hallucinated content, even in the pres-
ence of class imbalance.
V. CONCLUSION
In this work, we introducedTurk-LettuceDetect, a hallucina-
tion detection models specifically designed for Turkish RAG
applications. Our approach adapts the token-level classification
methodology of LettuceDetect to the linguistic characteris-
tics of Turkish by leveraging three novel achitectured base

models fine-tuned on the RAGTruth dataset [6]. Experimental
results show that our results achieves strong performance
across multiple tasks including question answering, data-to-
text writing, and summarization—while maintaining compu-
tational efficiency and supporting long-context inputs up to
8,192 tokens [12].
Our evaluation reveals critical insights with important
implications for multilingual hallucination detection. The
modernbert-base-tr-uncased-stsb-HDmodel demonstrates su-
perior performance in summary, qa and whole datasets while
lettucedect-210m-eurobert-tr-v1model excel in data2txt. Task-
dependent behavior patterns show summarization as the most
challenging domain, highlighting the need for task-specific de-
tection strategies. Multilingual transfer learning proves highly
effective, with EuroBERT-based models achieving robust
cross-lingual generalization without full in-language retrain-
ing. LLMs (GPT-4.1, Mistral) exhibit high recall but consis-
tently low precision, reinforcing the necessity of dedicated
detection mechanisms. Our encoder-based fine-tunel models
offers a favorable efficiency-accuracy trade-off with vary range
of model sizes such as 135M, 210M and 305M, enabling
token-level detection that improves interpretability for real-
world applications.
These results establishTurk-LettuceDetectmodels as a foun-
dational step toward robust hallucination detection in Turkish
RAG systems. Our work fills a critical gap in multilingual
RAG evaluation and supports future research in low-resource,
morphologically complex languages. We plan to publicly
release our models to encourage reproducibility and broader
adoption in real-time, trustworthy RAG pipelines.
ACKNOWLEDGMENT
This study is supported byGSI Attorney Partnership.
The authors would also like to express their gratitude for the
valuable insights and support provided throughout the research
process.
REFERENCES
[1] ´A. Kov ´acs, B. ´Acs, D. Kov ´acs, S. Szendi, Z. Kadlecik, and S. D ´avid,
“LettuceDetect: a hallucination detection framework for RAG applica-
tions,” arXiv preprint arXiv:2502.17125, 2025..
[2] ¨O. Ezerceli, G. G ¨um¨us ¸c ¸ekicci, T. Erkoc ¸, and B. ¨Ozenc ¸, “TurkEm-
bed4Retrieval: Turkish Embedding Model for Retrieval Task,” preprint.
[3] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-
Dahle, A. Letman, A. Mathur, A. Schelten, and T. Zhang, “Llama 3:
The next generation in open-source language models,” arXiv preprint
arXiv:2407.21783, 2024.
[4] T. Zhang, V . Kishore, F. Wu, K. Q. Weinberger, and Y . Artzi,
“BERTScore: evaluating text generation through BERT,” inProc. Int.
Conf. Learning Representations (ICLR), 2023.
[5] Z. Jiang, F. F. Xu, J. Araki, and G. Neubig, “Measuring factual accuracy
in generative question answering,” inProc. 61st Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers),
Toronto, Canada, 2023, pp. 1121–1135.
[6] C. Niu, Y . Wu, J. Zhu, S. Xu, K. Shum, R. Zhong, J. Song, and T.
Zhang, “RAGTruth: a hallucination corpus for developing trustworthy
retrieval-augmented language models,” inProc. 62nd Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers),
Bangkok, Thailand, 2024, pp. 10862–10878.
[7] X. Wang, A. Chowdhery, N. Kulkarni, S. Singh, S. Chaudhury, and M.
Naik, “Factored information retrieval for knowledge-grounded dialogue
evaluation,” arXiv preprint arXiv:2203.13245, 2022.[8] S. Lin, J. Hilton, and P. W. Koh, “TruthfulQA: Measuring how models
mimic human false beliefs,” arXiv preprint arXiv:2109.08654, 2022.
[9] K. Shuster, S. Frazier, A. Szlam, S. Sukhbaatar, and J. Weston, “Retrieval
is all you need: unifying retrieval and pre-trained models for knowledge-
intensive NLP tasks,” unpublished.
[10] M. Belyi, R. Friel, S. Shao, and A. Sanyal, “Luna: a lightweight evalua-
tion model to catch ‘language model hallucinations’ with high accuracy
and low cost,” inProc. 31st Int. Conf. Computational Linguistics:
Industry Track, Abu Dhabi, UAE, 2025, pp. 398–409.
[11] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, M.
Wang, and H. Wang, “Retrieval-augmented generation for large language
models: a survey,” arXiv preprint arXiv:2312.10997, 2024.
[12] B. Warneret al., “Smarter, better, faster, longer: a modern bidirectional
encoder for fast, memory efficient, and long context finetuning and
inference,” arXiv preprint arXiv:2412.13663, 2024.
[13] Z. Nussbaum, J. X. Morris, B. Duderstadt, and A. Mulyar, “Nomic
embed: Training a reproducible long context text embedder,” arXiv
preprint arXiv:2402.01613, 2025.
[14] P. Ahadian and Q. Guan, “A survey on hallucination in large language
and foundation models,” arXiv preprint arXiv:2407.21783, 2024.
[15] K. Oflazer, “Error-tolerant finite-state recognition and analysis of mor-
phologically complex languages with applications to Turkish,” inProc.
Fifth Int. Workshop on Finite State Methods in Natural Language
Processing, 2003, pp. 1–12.
[16] S. Min, X. Lyu, R. Shin, M. Li, H. Rashkin, S. Singh, L. Zettlemoyer,
and T. Lei, “Rationale-augmented ensembles in chain-of-thought har-
ness,” arXiv preprint arXiv:2306.09227, 2023.
[17] P. Lewiset al., “Retrieval-augmented generation for knowledge-intensive
NLP tasks,” inProc. 37th Int. Conf. Machine Learning (ICML), 2020,
pp. 6484–6494.
[18] M. Chern, S. Chen, D. Fried, D. Klein, and A. Ratner, “Hallucina-
tion detection using LLM judges with structured reasoning traces,” in
Proc. 2023 Conf. Empirical Methods in Natural Language Processing
(EMNLP), Singapore, 2023, pp. 1121–1135.
[19] M. Azaria and Y . Belinkov, “Hallucination detection in language models
via contextual verification,” inProc. 61st Annual Meeting of the Associ-
ation for Computational Linguistics (Volume 1: Long Papers), Toronto,
Canada, 2023, pp. 1234–1250.
[20] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-
training of deep bidirectional transformers for language understanding,”
inProc. 2019 Conf. North American Chapter of the Association for
Computational Linguistics: Human Language Technologies, Volume 1
(Long and Short Papers), Minneapolis, Minnesota, 2019, pp. 4171–4186.
[21] A. Vaswaniet al., “Attention is all you need,” inAdvances in Neural
Information Processing Systems, vol. 30, 2017.
[22] OpenAI Team, “GPT-4 technical report,” arXiv preprint
arXiv:2303.08774, 2024.
[23] A. Q. Jianget al., “Mistral 7b,” arXiv preprint arXiv:2310.06825, 2023.
[24] P. Bajajet al., “Ms marco: A human generated machine reading
comprehension dataset,” arXiv preprint arXiv:1611.09268, 2018.
[25] Yelp, “Yelp open dataset,” 2021. [Online]. Available:
https://www.yelp.com/dataset. [Accessed: Nov. 3, 2023].
[26] J. Su, M. Ahmed, Y . Lu, S. Pan, W. Bo, and Y . Liu, “Roformer: En-
hanced transformer with rotary position embedding,”Neurocomputing,
vol. 568, pp. 127063, 2024.
[27] Gemma Team, “Gemma 2: Improving open language models at a
practical size,” arXiv preprint arXiv:2408.00118, 2024.
[28] A. See, P. J. Liu, and C. D. Manning, “Get to the point: Summarization
with pointer-generator networks,” inProc. 55th Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers),
Vancouver, Canada, 2017, pp. 1073–1083.
[29] T. Wolfet al., “HuggingFace’s Transformers: State-of-the-art natural
language processing,” arXiv preprint arXiv:1910.03771, 2020.
[30] Qin, L., Chen, Q., Feng, X., Wu, Y ., Zhang, Y ., Li, Y ., Yu, P. S.
(2024). Large language models meet nlp: A survey. arXiv preprint
arXiv:2405.12819.
[31] Benkirane, K., Gongas, L., Pelles, S., Fuchs, N., Darmon, J., Stenetorp,
P., S ´anchez, E. (2024). Machine translation hallucination detection for
low and high resource languages using large language models. arXiv
preprint arXiv:2407.16470.
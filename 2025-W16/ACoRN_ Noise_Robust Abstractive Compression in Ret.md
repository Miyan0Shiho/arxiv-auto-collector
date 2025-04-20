# ACoRN: Noise-Robust Abstractive Compression in Retrieval-Augmented Language Models

**Authors**: Singon Kim, Gunho Jung, Seong-Whan Lee

**Published**: 2025-04-17 06:05:35

**PDF URL**: [http://arxiv.org/pdf/2504.12673v1](http://arxiv.org/pdf/2504.12673v1)

## Abstract
Abstractive compression utilizes smaller langauge models to condense
query-relevant context, reducing computational costs in retrieval-augmented
generation (RAG). However,retrieved documents often include information that is
either irrelevant to answering the query or misleading due to factual incorrect
content, despite having high relevance scores. This behavior indicates that
abstractive compressors are more likely to omit important information essential
for the correct answer, especially in long contexts where attention dispersion
occurs. To address this issue, we categorize retrieved documents in a more
fine-grained manner and propose Abstractive Compression Robust against Noise
(ACoRN), which introduces two novel training steps. First, we use offline data
augmentation on the training dataset to enhance compressor robustness against
two distinct types of retrieval noise. Second, since the language modelbased
compressor cannot fully utilize information from multiple retrieved documents
and exhibits positional bias, we perform finetuning to generate summaries
centered around key information that directly supports the correct answer. Our
experiments demonstrate that T5-large, trained with ACoRN as a compressor,
improves EM and F1 scores while preserving the answer string, which could serve
as direct evidence. ACoRN excels on datasets with many accuracy-reducing
documents, making it highly useful in real-world scenarios.

## Full Text


<!-- PDF content starts -->

ACoRN: Noise-Robust Abstractive Compression
in Retrieval-Augmented Language Models
Singon Kim
Department of Artificial Intelligence
Korea University
Seoul, Republic of Korea
singon kim@korea.ac.krGunho Jung
Department of Artificial Intelligence
Korea University
Seoul, Republic of Korea
ghjung@korea.ac.krSeong-Whan Lee*
Department of Artificial Intelligence
Korea University
Seoul, Republic of Korea
sw.lee@korea.ac.kr
Abstract —Abstractive compression utilizes smaller langauge
models to condense query-relevant context, reducing computa-
tional costs in retrieval-augmented generation (RAG). However,
retrieved documents often include information that is either
irrelevant to answering the query or misleading due to factual
incorrect content, despite having high relevance scores. This
behavior indicates that abstractive compressors are more likely
to omit important information essential for the correct answer,
especially in long contexts where attention dispersion occurs.
To address this issue, we categorize retrieved documents in a
more fine-grained manner and propose Abstractive Compression
Robust against Noise ( ACoRN ), which introduces two novel
training steps. First, we use offline data augmentation on the
training dataset to enhance compressor robustness against two
distinct types of retrieval noise. Second, since the language model-
based compressor cannot fully utilize information from multiple
retrieved documents and exhibits positional bias, we perform fine-
tuning to generate summaries centered around key information
that directly supports the correct answer. Our experiments
demonstrate that T5-large, trained with ACoRN as a compressor,
improves EM and F1 scores while preserving the answer string,
which could serve as direct evidence. ACoRN excels on datasets
with many accuracy-reducing documents, making it highly useful
in real-world scenarios.
Index Terms —Noise robustness, Abstractive compression,
Retrieval-augmented language models, Large language models
I. I NTRODUCTION
Retrieval-augmented language models (RALMs) [1], [2]
have strong capabilities in both academic research and indus-
trial applications within the area of natural language process-
ing (NLP). Retrieval-augmented generation (RAG) process in
RALMs was designed to improve the performance of large
language models (LLMs) [17], [36] by integrating external
knowledge. It can be achieved by simply appending supporting
documents to the input, without the need to update the LLMs.
While this approach helps bridge knowledge gaps in LLMs,
it significantly increases computational costs, particularly for
This research was supported by the Institute of Information & Communica-
tions Technology Planning & Evaluation (IITP) grant, funded by the Korea
government (MSIT) (No. RS-2019-II190079 (Artificial Intelligence Graduate
School Program (Korea University)), No. RS-2024-00436857 (Information
Technology Research Center (ITRC)), No. RS-2024-00457882 (AI Re- search
Hub Project), and No. RS-2024-00336673 (AI Technology for Interactive
Communication of Language Impaired Individuals).
* Seong-Whan Lee is the corresponding author.
. . . (complex content) From the 
interior to the outlet at Chicago , water 
flows from Superior to Huron and 
Michigan, southward to Erie, 
(complex content) . . .
Retrieved Document A
The Great Lakes (), also called the 
Laurentian Great Lakes . . . (complex 
content) which connect to the 
Atlantic Ocean through the Saint 
Lawrence River .  (context content) . . .
Retrieved Document B
... (complex content) The Great 
Lakes began to form at the end of 
the last glacial period around 14,000 
years ago . . . (complex content) ...Retrieved Document C
The Great Lakes (), also called the 
Laurentian Great Lakes . . . (complex 
content) which connect to the 
Atlantic Ocean through the Saint 
Lawrence River .  (context content) . . .
. . . (complex content) From the 
interior to the outlet at Chicago , water 
flows from Superior to Huron and 
Michigan, southward to Erie, 
(complex content) . . .
Question: where do the great lakes 
meet the ocean?The Great Lakes (), also called the 
Laurentian Great Lakes . . . (complex 
content) which connect to the 
Atlantic Ocean through the Saint 
Lawrence River .  (context content) . . .The Great Lakes (), also called the 
Laurentian Great Lakes . . . (complex 
content) which connect to the 
Atlantic Ocean through the Saint 
Lawrence River .  (context content) . . .
... (complex content) The Great 
Lakes began to form at the end of 
the last glacial period around 14,000 
years ago . . . (complex content) ...Question: where do the great lakes 
meet the ocean?
Question: where do the great lakes 
meet the ocean?
Compressor
The Great Lakes meet Atlantic Ocean through the Saint Lawrence River .The Great Lakes meetAtlantic Ocean at Chicago .There is no information
Fig. 1. An illustrative example of a challenge in retrieving and summarizing
information supporting to the correct answer from the documents. The
compressor performs well in summarizing content supported to the correct
answer when only the document including the correct answer is provided.
However, it generates incorrect information or misses the key information
when the retrieved documents contain inaccurate or irrelevant information.
transformer-based LLMs due to attention complexity [20],
[40]. With scaling, this burden becomes even more significant
[3]. To mitigate these issues without compromising critical
information, abstractive compression has been proposed [12].
Abstractive compression methods [12], [13] leverage the
query-focused summarization (QFS) capabilities of language
models [21] to reduce tokenization overhead effectively. How-
ever, since retrievers rank documents based on their relevance
to the the query [31], retrieved documents may contain in-
formation with high relevance scores that is either irrelevant
to generating the answer or is related but incorrect. Due to
limitations of language models, such as limited relevance
discrimination [8] and attention dispersion in long contexts
[5], existing abstractive compressors struggle with significant
information loss when retrieving multiple documents. Existing
abstractive compression methods do not address these issues,
leading to hesitation in their adoption for real-world appli-
cations. To investigate these issues step by step, we refer
to information that hinders the LLM from generating the
correct answer as noise and call the documents containing
such information noise documents . As shown in Fig. 1, variousarXiv:2504.12673v1  [cs.CL]  17 Apr 2025

types of retrieval noise exist in real-world scenarios, and
the responses generated by the compressor also vary due to
the interference of retrieval noise. We systematically explore
two types of retrieval noise: (i) retrieved documents that are
thematically related to the query but contain incorrect informa-
tion ( Factual error documents ), and (ii) retrieved documents
lacking sufficient information to answer the query ( Irrele-
vant documents ). Existing open-domain question answering
(ODQA) [19] training datasets do not consider the types of
noise documents, resulting in only partial robustness to noise
and causing a significant performance gap between the training
dataset and the test dataset.
In this work, we seek to mitigate noise influence effi-
ciently within the scope of the ODQA task. Then we propose
Abstractive Compression Robust against Noise ( ACoRN ), a
training method that addresses this problem with the following
two objectives: (i) mitigating distraction caused by two types
of retrieval noise. (ii) reducing the loss of information that
directly supports the correct answers in long contexts. Our
method, ACoRN, reconstructs the training dataset through
offline data augmentation to ensure robustness against two
types of retrieval noise, as described in Section III-C. To
reduce the loss of information that directly supports the correct
answer in long contexts, positional bias must be addressed.
This problem, known as the “lost in the middle” phenomenon,
often occurs in abstractive compression [5], [34]. To address
this issue, inspired by FILM [3], we propose fine-tuning that
targets information directly supporting the correct answer. To
create training dataset labels that capture key information
directly supporting the correct answer, we heuristically define
evidential documents as positive documents that contain the
answer string. Although the presence of the answer string
alone does not necessarily mean that evidential documents
contain information that directly supports the correct answer,
this heuristic definition still demonstrates highly effective
performance [27]. Then, we use data mining to extract and
provide only evidential documents, instead of giving all re-
trieved documents to the LLM. When training with a dataset
constructed using these labels, it preserves key information
while mitigating positional bias.
Through these training steps, ACoRN enhances robustness
against various types of retrieval noise and improves its ability
to summarize key information within evidential documents. To
demonstrate the effectiveness of our compression strategy for
retrieving and summarizing evidential documents, we analyze
the performance on ODQA benchmarks, including Natural
Questions (NQ) [14], TriviaQA [15], and PopQA [6]. We
also reconstruct benchmark test datasets by applying offline
data augmentation to introduce various types of noise into the
documents. Each test dataset is designed to demonstrate noise
robustness from different perspectives. Our method, ACoRN,
has shown improved performance over other compression
methods. We also show that distilling LLM summarization
abilities using evidential documents from top- kretrievals
better preserves the answer string compared to using all
top-kdocuments. Furthermore, in ACoRN we examine theimpact of offline data augmentation on noise robustness. Based
on the experimental results, ACoRN performs exceptionally
well on datasets with numerous accuracy-reducing documents,
demonstrating its high applicability in real-world scenarios.
The main contributions of our work can be summarized as
follows:
•A concise noise classification that distinguishes between
irrelevant information and misleading yet query-related
information, designed to enhance the robustness of ab-
stractive compression training against retrieval noise.
•We show that extracting only evidential documents from
the top- kretrieved results through data mining is an
effective approach for generating summarization labels,
compared to using the entire top- kresults.
•We propose ACoRN, an effective and efficient noise-
robust abstractive compression method. Validated on
three ODQA benchmarks, it outperforms other methods,
especially on datasets with a high noise-document ratio.
II. R ELATED WORK
A. Abstractive Compression
Query-aware context compression methods summarize con-
text differently depending on the question or task [23]. They
can be categorized into token pruning [22], [38], extractive
compression [12], [24], and abstractive compression [12], [13].
Unlike token pruning and extractive compression that simply
focus on retaining the necessary tokens and sentences, abstrac-
tive compression retrieves and integrates only the essential
information needed to answer the query and reformulates it
accordingly [33]. However, abstractive compression suffers
from significant information loss, which is essential for sup-
porting the correct answer to the query. This issue can be
mainly attributed to positional bias, known as the “lost in the
middle” phenomenon, where information in the middle of a
given context is more likely to be omitted [5], [34].
Previous abstractive compression methods [12], [13] have
been proposed to overcome this drawback and enhance the
effectiveness of compression. COMPACT [13] segments large
document sets into passages and iteratively retrieves key in-
formation for summarization. While this mitigates information
loss by summarizing across multiple steps, it does not solve
the core issue of abstractive compression loss and incurs
high computational and time costs. RECOMP [12] enhances
compressed passages by generating multiple summaries and
selecting low-perplexity samples that yield correct answers.
However, relying on API-based LLMs is costly, and when
key information is missing or buried, even numerous samples
may lack quality. Unlike RECOMP, our method focuses on
training the model to generate trustworthy labels curated for
noise robustness without producing multiple samples.
B. Noise-Robust Retrieval-Augmented Language Models
Retrieval-Augmented Language Models (RALMs) [1], [2]
refer to language models trained to generate answers based on
documents retrieved from an external source. This approach

Question:Who was the producer of Happy Gilmore?
Top- Retrieval
(complex content) Happy Madison takes its name from 
the films “Happy Gilmore” and “Billy Madison”, two box 
office successes starring Sandler himself, both produced 
by Robert Simonds . (complex content). . .
Retrieve only documents 
including answer string
"Happy Gilmore" was produced by Robert Simonds 
and directed by Dennis Dugan, featuring Adam 
Sandler as the title character, a struggling ice hockey 
player who excels in golf.
Compressor
Happy Gilmore is a 1996 American sports comedy film 
produced by Robert Simonds , starring Adam Sandler 
as the title character, an unsuccessful ice hockey 
player who discovers a newfound talent for golf.Compressor’s Summarization(complex content) Happy Madison takes its name from 
the films “Happy Gilmore” and “Billy Madison”, two box 
office successes starring Sandler himself, both produced 
by Robert Simonds . (complex content). . . Evidential
... (complex content) happy Gilmore is a 1996 American 
sports comedy film directed by dennis dugan with music 
by mark Mothersbaugh and produced by id. (complex
content) ... Factual Error
(complex content) A distraught Shooter attempts to 
steal the winner’s gold jacket, but he is tracked down, and 
beaten up by Mr. Larson, Gilmore’s imposing ex-boss, 
and a mob of fans. (complex content). . . Irrelevant
Generation Loss: 
Summarization label
Feedback
Teacher Model
Evidential Documents
Top- Retrieval Documents
Question:Who was the producer of Happy Gilmore? Question:Who was the producer of Happy Gilmore?
Fig. 2. Overview of Abstractive Compression Robust against Noise (ACoRN). We fine-tune a compressor on our curated training dataset to make it robust
against noisy documents and to retrieve evidential documents, focusing on summarizing the query based on the content of the evidential documents.
has been demonstrated to improve model performance across
a wide range of NLP tasks, including language modeling
[35] and ODQA [19]. However, due to the limitations of the
retriever’s capabilities, retrieval-augmented systems inevitably
have to deal with documents irrelevant or partially relevant
to the task [10], [39]. Prior studies [32], [37] have shown
that when the noise ratio in the retrieval context increases, the
performance of RALMs noticeably declines [9].
One of previous noise robustness studies, RAAT [25], has
made two important observations. First, language models
with fewer parameters are more easily distracted by noise
documents. Second, in real-world scenarios there are various
types of retrieval noise. It can be classified into three types:
relevant retrieval noise, irrelevant retrieval noise, and counter
factual retrieval noise. This demonstrates the need for training
models against various types of retrieval noise. Additionally,
Noise RAG Benchmark [18] defines as many as seven distinct
noise types from a linguistic perspective. However, training
robustness against various types of retrieval noise requires
substantial computational cost, so such a fine-grained division
can be challenging to implement. Thus, in contrast to previous
approaches, we aim to propose a concise noise classification
tailored specifically for abstractive compression training.
III. M ETHODOLOGY
We introduce Abstractive Compression Robust against
Noise (ACoRN), shown in Fig. 2. ACoRN is a simple yet
effective training approach for enhancing robustness to noise
induced by irrelevant and factual error retrieval noise. The
details of ACoRN are described in the following subsections.A. Problem Setup
In standard RALMs, when a query qis given, the retriever
is designed to retrieve top- kdocuments Dk={d1, d2, ...d k}
from an external database. A pre-trained language model M
predicts the correct answer yconditioned on the query q, top-k
documents Dk, and an instruction Ii. The instruction Iiacts as
a cue to guide Min generating the correct answer as follows:
M(Ii, Dk, q) =y. (1)
To reduce the computational cost of Mcaused by process-
ing top- kretrieved documents, a compressor is introduced to
summarize Dk. Building on this approach, the goal can be
formulated as follows:
arg max
πPM(y|Ii, Sπ, q), (2)
Sπ=π(Ic, Dk, q)with l(sπ)≪l(Dk), (3)
where πis a function that compresses documents Dkinto a
shorter context Sπbased on the query q, then lrepresents
the number of tokens and Icis the instruction to guide the
compressor in summarizing the documents.
We divide Dkinto two subsets: DeandDnoisy .Deis the set
of evidential documents, and Dnoisy is the set of noise docu-
ments. If a retrieved document dcontains the correct answer y
about qwe can denote d∈De. However, if ddoesn’t contain
y, we denote d∈Dnoisy . Let Sbe the compressed context
within the documents consisting of information that directly
supports y. We aim to fine-tune an abstractive compressor, π′,
that not only performs the mapping π′:{Ic, De, q} →S, but

TABLE I
THE STATISTICS OF THE THREE ODQA TEST DATASETS . #F ULL
REPRESENTS THE TOTAL NUMBER OF TEST DATA ,WHILE #SUBSET REFERS
TO THE REMAINING NUMBER OF TEST DATA WHEN CONTROLLED TO
EVALUATE PERFORMANCE VARIATIONS BASED ON NOISE TYPE
DatasetsTest
#Full #Subset Percentage (%)
NQ [14] 3,610 1,417 39.25
TriviaQA [15] 11,313 2,966 26.21
PopQA [6] 1,399 413 29.52
also effectively summarizes important information supporting
the correct answers, even in the presence of additional noise
documents Dnoisy . Formally, the function can be written as
π′:{Ic, De, Dnoisy, q} →S.
B. Classifying Noise Documents
Existing studies [18], [25] on retrieval noise robustness
classify types of noise commonly found in the real world.
However, it is burdensome to consider all of them when
training a compressor. Moreover, when using a detailed clas-
sification, some noise documents belong to multiple retrieval
noise groups, further complicating the process. Therefore, we
define the noise documents Dnoisy to closely mimic real-world
conditions but with minimal classification as follows:
Dnoisy =Dirr∪Df. (4)
Here, Dirrrepresents the set of irrelevant documents. For
anyd∈Dirr, this set encompasses contexts with high
relevance to the q, but where dlacks the information that
directly supports y.Dfis the set of factual error documents.
For any d∈Df, this set includes contexts that are thematically
related to qbut contain incorrect or misleading information,
such as incorrect historical facts or inaccurate numerical data.
To examine the influence of these distinct types of docu-
ments on compressors, we establish benchmarks for assessing
retrieval noise robustness. Specifically, we established bench-
marks for NQ [14], TriviaQA [15], and PopQA [6] individ-
ually, as detailed in Table I. The details of the construction
of this benchmark can be found in Section IV-B. Leverag-
ing these benchmarks, we evaluate performance variations
when the model is exposed only to evidential documents,
compared to when irrelevant documents and factual error
documents are additionally incorporated. As shown in Fig. 3,
we conduct experiments on Flan-T5-large [29], analyzing the
varying impacts of these three types of retrieved documents.
The inclusion of evidential documents improves performance,
whereas adding irrelevant or factual error documents results
in a decline ranging from 5.68% to 15.92%. Through a
comparative analysis of the effects of the two types of noise,
we observe that the presence of irrelevant documents has
minor impact on compressors with substantial capabilities.
C. Noise Documents Construction
Previous studies [32], [37] have explored solutions to im-
prove language models’ robustness to noise by embedding re-
trieved noise documents within the fine-tuning data. Precisely
51.45 51.59
46.86
43.26
1020304050607080
75.32 76.06
71.04
67.7
1020304050607080One Evidential Doc Two Evidential Docs
One Evidential Doc & Irrelevant Doc One Evidential Doc & Factual Error Doc
Natural Questions Trivia QA65.6272.64
59.0857.38
1020304050607080
PopQAFig. 3. Exact Match (EM) scores for different types of noise documents,
including irrelevant documents and factual error documents. Flan-T5-large
[29] compresses documents using Query-Focused Summarization (QFS),
compressed passages are then passed to LLaMA-3.1-8B-Instruct [36] to
generate answers to the queries.
calibrating both the kind and magnitude of noise is necessary
to optimize the model’s performance [25]. To effectively
incorporate the two types of retrieval noise in constructing
our training dataset, we employ offline data augmentation
Da. The main objective is to retrieve and summarize the
information supporting yfound in the evidential documents
Da
eeven when noise documents are present. According to
Fig. 3, factual error documents play a more critical role in
model performance than irrelevant documents. For this reason,
additional consideration is required for noise-robust training
to address factual errors in documents. We aim to train the
compressor to recognize and summarize evidential documents
when they provide conflicting information with factual er-
ror documents. To ensure consistency, we do not augment
factual error documents if evidential documents are absent.
However, since evidential documents are not always present
alongside factual error documents, we also intend to account
for cases where evidential documents are present without any
factual error documents during training. Hence, when there
are retrieved Nevidential documents, the probability of each
evidential document becoming a factual error document is
1
N+1. Additionally, the probability of none of the evidential
documents becoming factual error documents is also1
N+1.
Formally, the set of evidential documents before offline data
augmentation is De={e1, e2, ...eN}andDa
efrepresents the
collection of evidential documents that have been transformed
into factual error documents through data augmentation. For
an arbitrary number m∈ {1,2, ...N}given,
P(em∈Da
ef) =1
N+ 1,
with P
N[
j=1ej∈Da
ef
=NX
j=1P(ej∈Da
ef).(5)
We structured the input in the training dataset used for the
compressor accordingly.

D. Evidential Pseudo Labels for Summarization
We train an abstractive compressor, taking an input se-
quence x={Ic, q}along with a concatenation of top- k
retrieved documents Dkand producing a summary Sπ. Since
we desire the compressor to be substantially smaller than the
large-scale model M, we employ knowledge distillation [16]
to build a lightweight abstractive compressor. For generating
pseudo summarization labels Sbased on evidential documents,
we use a pre-trained LLM as a teacher model. The teacher
model is provided with a query and only the evidential
documents to perform QFS via an instruction. In other words,
the teacher model generates the pseudo summarization labels
Sby focusing on the evidential documents after offline data
augmentation Da
e. Formally, the objective πtof generating
ground truth summarization labels is as follows:
S=πt(Ic, Da
e, q). (6)
IfDa
e=∅, we don’t pass any retrieved documents to
M. Then Mgenerates the answer without any supporting
information as follows:
M(Ii, q) =y. (7)
E. Abstractive Compression Training
Using the training dataset constructed with Sbased on
evidential documents, the compressor is distilled [16]. At this
stage, the input differs from that of the teacher model as it
includes noise documents Da
noisy such as Da
irrandDa
f. To ef-
fectively train the compressor to summarize information based
on evidential documents, a function is defined as follows:
Sπ=π′(Ic, Da, q). (8)
The loss function is designed to facilitate summarization
training by enforcing a strong alignment between the sum-
marization labels Sand the generated summaries Sπ. In
this setup, Ndenotes the sequence length in each sample.
θdenotes the parameters of the compressor. Then the loss
function is expressed as:
Lgen(θ, x, S ) =−NX
i=1logPθ(Si|x, S<i). (9)
IV. E XPERIMENTS
A. Implementation Details
We evaluate our approach in language models using ODQA
benchmarks, specifically the NQ [14], TriviaQA [15] and
PopQA [6] datasets. NQ comprises queries qpaired with short
answers containing no more than five tokens. TriviaQA is
constructed by extracting spans from Wikipedia articles that
contain correct answers to each given query. PopQA is created
by transforming Wikidata knowledge triples, consisting of a
subject, a relationship, and an object into natural language
query using manually written templates for 16 diverse rela-
tionship types. For all experiments, documents are retrieved
from Wikipedia using the adversarial Dense Passage Retriever(DPR) [31], which finds five documents per query. For in-
ference, we use Llama-3.1-8B-Instruct [36] as the language
model Mand perform our experiments in a zero-shot manner.
As the compressor, Flan-T5-large [29] and T5-large [30] are
fine-tuned for the task. We use two NVIDIA RTX A6000
GPUs for fine-tuning and a single NVIDIA RTX A5000 GPU
for inference.
B. Datasets Construction
To make the compressor resistant to retrieval noise, we
create a training dataset and two test benchmarks using offline
data augmentation. In the cases where two or more evidential
documents are present in the top- 5retrieved documents, we
randomly select one or none of them, mask the answer entity
and use RoBERTa-large [28] to replace it with an incorrect
entity, resulting in a document with factual errors.
The training dataset T={q, Da, S}includes each query q,
offline-augmented documents Da, and summarization labels
S. To create more accurate pseudo labels S, we use only
the set of evidential documents Da
e⊆Daas supporting
documents. As with a previous method [12], we utilize the
QFS capabilities of GPT-3.5-turbo. It is provided with {q,Da
e}
to generate high quality QFS, which is then used as S.
We design benchmarks to evaluate two aspects of noise ro-
bustness: (i) assessing whether the top- 5retrieved documents,
despite retrieval noise, can detect and compress the evidence
directly supporting the correct answer, as discussed in Section
V-C. (ii) evaluating the impact of incorporating various types
of retrieved noise documents along with the evidential ones on
compression quality, as described in Section V-D. To evaluate
(i), we apply filtering to the queries, ensuring that each query
in the filtered subset contains at least one evidential document
from the offline data augmented test dataset. To assess (ii),
we create a new dataset by extracting samples that include all
types of retrieved documents: evidential documents, irrelevant
documents, and factual error documents. Performance is then
compared under these scenarios: (a) an evidential document
only, (b) an evidential document combined with an irrelevant
document, (c) an evidential document combined with a factual
error document. The statistics of this newly formulated bench-
mark compared to the original datasets is shown in Table I.
C. Training
We train the compressor using the training dataset T, where
Daandqare provided as inputs with a compression instruction
Ic. The training objective is formally defined by the function
π:{Ic, Da, q} → S, where Srepresents the summarization
labels produced by GPT-3.5-turbo, and the compressor is
trained to replicate these labels, effectively distilling GPT-
3.5-turbo’s summarization capability. We conduct experiments
with two models: Flan-T5-large and T5-large. We also provide
instructions Iconly to the Flan-T5-large. The batch size per
device is set to 2, with gradient accumulation step of 2.
Evaluation is performed every 1000 steps.

TABLE II
QUANTITATIVE EVALUATION OF ACORN ON THE ODQA TASKS USING LLAMA-3.1-8B-I NSTRUCT
MethodNQ TriviaQA PopQA
EM ( ↑) F1( ↑) CR( ↓) Inference Time( ↓) EM ( ↑) F1( ↑) CR( ↓) Inference Time( ↓) EM ( ↑) F1( ↑) CR( ↓) Inference Time( ↓)
No Retrieval 19.94 31.46 - 0.249 49.65 59.59 - 0.162 20.51 23.68 - 0.132
Top-1 document 27.87 42.12 - 0.241 48.21 61.67 - 0.216 17.87 21.01 - 0.154
Top-5 documents 31.83 47.40 - 0.444 57.90 69.38 - 0.246 41.46 51.02 - 0.321
LongLLMLingua [22] 26.84 42.72 0.562 0.396 54.45 66.59 0.562 0.297 38.24 47.87 0.597 0.238
Quito [38] 29.94 44.04 0.484 0.316 56.61 67.61 0.484 0.258 41.89 50.36 0.523 0.241
RECOMP [12] 32.58 45.42 0.052 0.225 58.64 68.40 0.049 0.174 - - - -
ACoRN (T5-large) 34.97 47.70 0.064 0.205 58.20 68.33 0.049 0.174 45.60 52.38 0.059 0.137
Flan-T5-large [29] 31.97 45.15 0.133 0.227 57.34 67.72 0.085 0.183 42.89 48.80 0.071 0.142
ACoRN (Flan-T5-large) 35.56 48.48 0.065 0.208 58.33 68.58 0.056 0.174 45.75 52.82 0.062 0.139
D. Evaluation Metrics
We assess the performance of our method by measuring
the Exact Match (EM) score and F1 score, and we also
evaluate efficiency using the compression ratio (CR) [7] and
inference time [11]. Specifically, EM measures how precisely
the system’s answer matches the reference answer at the char-
acter level, while the F1 score balances precision and recall,
evaluating the accuracy of identified answers and minimiz-
ing omissions. CR evaluates how efficiently the compressor
summarizes the information essential to answer the query.
Inference time refers to the time taken by the language model
Mto process and generate a response for input data during
inference. Inference time is critical for practical applications
requiring real-time user query responses or large-scale data
processing. The unit of inference time is seconds.
V. R ESULTS
A. Main Results
Table II presents our main results and illustrates the effec-
tiveness of our method ACoRN, compared to the baselines in
terms of EM, F1, CR, and inference time. First, we compared
the results of our method to those from top- 5retrieval without
compression. Our experiments demonstrate that our method,
ACoRN, minimizes the loss of critical information, effectively
conveys it to the language model M, and reduces inference
time, resulting in higher EM scores across all datasets. This
indicates that through ACoRN irrelevant documents and fac-
tual error information are eliminated, allowing Mto confi-
dently generate accurate answers. Second, we compared our
method’s results to those of other compression methods. Our
approach outperforms existing token pruning methods such as
LongLLMLingua [22] and Quito [38] across all metrics. Our
method shows slightly lower metrics on TriviaQA compared to
RECOMP [12], which is attributed to the inherent differences
of language model M. As explained in Section II-A, RECOMP
uses perplexity during label creation to selectively augment
data, optimizing summarization to improve the model’s ability
to answer correctly. Combining our method with RECOMP’s
selective data augmentation could yield even better results.
Third, the summarization generated by ACoRN significantly
reduces the latency of answer generation by the language
020406080
N=0 N=1 N=2 N=3 N=4NQ
020406080
N=0 N=1 N=2 N=3 N=4TQAEM EMOnly Evidential Documents
Evidential Documents w. Noise Documents
Only Evidential Documents regardless of 
Using all documents from the top-5 documents regardless of 
020406080
N=0 N=1 N=2 N=3 N=4PopQAEMFig. 4. Comparison of GPT-3.5-turbo QFS performance when only evidential
documents are included in the prompt versus when all top-5 documents
are included, based on random sampling of 100 cases for each evidential
document count Nin top-5 retrieval. When N=0 with retrieved only evidential
documents means using only internal knowledge. The compressed output is
passed to the inference model’s prompt, with the language model Mbeing
LLaMA-3.1-8B-Instruct. The dotted line represents the performance when
summarization is done by randomly sampling 100 instances, regardless of N.
TABLE III
EVALUATION OF PRESERVING ANSWER STRING RATIO (PAR) AND
COMPRESSION RATIO (CR) IN OFFLINE AUGMENTED TEST DATASET
MethodNQ TriviaQA PopQA
CR(↓) PAR( ↑) CR( ↓) PAR( ↑) CR( ↓) PAR( ↑)
Token pruning
LongLLMLingua 0.5607 0.6320 0.5625 0.7044 0.5963 0.7760
Abstractive compression
RECOMP 0.0557 0.4966 0.0522 0.6283 - -
ACoRN (T5-large) 0.0637 0.6425 0.0522 0.6380 0.0675 0.7521
ACoRN (Flan-T5-large) 0.0663 0.6687 0.0579 0.6686 0.0674 0.7635
model M, as demonstrated by inference time. With analysis
of EM, F1 scores, CR and inference time, we observe that
by summarizing key information critical for answer genera-
tion, Mwas able to easily retrieve relevant information and
generate accurate answers.
B. Reliability of Summarization Labels
As shown in Fig. 4, we compared the reliability of sum-
marization labels generated using all documents versus using
only evidential documents. We split the test dataset based
on the number of evidential documents Nfrom the top- 5
retrieved documents. We aim to examine how the quality of
summarization labels changes with the proportion of presented
evidential documents and how the performance gap between
the two settings varies accordingly. We randomly sampled 100
instances for each Nto ensure fair evaluation. We observe

TABLE IV
QUANTITATIVE EVALUATION OF ACORN PERFORMANCE CHANGES WITH INCREASED RETRIEVAL NOISE THROUGH DATA AUGMENTATION IN ODQA
TEST DATASET RETRIEVED DOCUMENTS ,COMPARED TO THE METHOD WITHOUT NOISE DOCUMENTS AUGMENTATION IN THE TRAINING DATASET .
MethodNQ TriviaQA PopQA
EM ( ↑) F1 ( ↑) EM ( ↑) F1 ( ↑) EM ( ↑) F1 ( ↑)
ACoRN (T5-large)34.79→32.68
(-2.11)47.70→45.59
(-2.11)58.20→57.31
(-0.89)68.33→67.35
(-0.98)45.60→44.39
(-1.21)52.38→50.67
(-1.71)
w/o noise documents augmentation35.15→32.54
(-2.61)47.65→44.88
(-2.77)58.57→57.38
(-1.19)68.54→67.23
(-1.31)44.10→41.82
(-2.28)50.37→47.88
(-2.49)
TABLE V
PERFORMANCE COMPARISON OF TRAINING A COMPRESSOR ,BUILT ON
T5- LARGE ,USING FACTUAL ERROR DOCUMENTS CONSTRUCTED
THROUGH OFFLINE DATA AUGMENTATION VERSUS USING ORIGINAL
RETRIEVAL DOCUMENTS ON LLAMA-3.1-8B-I NSTRUCT
MethodOne Evidential Doc One Evidential Doc
w. Irrelevant DocOne Eividential Doc
w. Factual Error Doc
EM ( ↑) F1( ↑) EM ( ↑) F1( ↑) EM ( ↑) F1( ↑)
Natural Questions
ACoRN 51.94 66.55 50.11 64.39 49.12 62.96
w/o noise documents augmentation 51.66 66.70 49.82 64.54 47.71 62.13
TriviaQA
ACoRN 74.68 82.98 72.96 81.62 70.20 78.73
w/o noise documents augmentation 74.88 83.14 71.94 80.27 68.84 77.27
PopQA
ACoRN 66.34 72.50 58.84 64.29 54.00 60.33
w/o noise documents augmentation 64.41 70.74 58.60 63.62 53.75 59.55
the summarization performance of the teacher model in two
scenarios: when only Nevidential documents are provided,
and when Nevidential documents are combined with 5−N
noise documents. When the fixed value of Nis small, using
only the evidential documents for summarization is more
effective compared to using all retrieved documents. When
we increase N, there are cases where utilizing all retrieved
documents is more effective. This is because the retrieval noise
from documents with low semantic relevance to the query
can help better differentiate evidential documents from noise,
leading to a more effective summarization [18]. However, the
effect is minimal when compared to using only evidential
documents. Results show that using only evidential documents
for summarization labels is more effective.
C. Evaluation of Answer String Preservation
When at least one evidential document exists for a query,
we evaluate the preservation ratio of the answer string (PAR)
in the compression output to assess how well the model
summarizes content that directly supports the correct answer
within the retrieved documents. As shown in Table III, our
method demonstrates a high compression ratio while main-
taining a remarkably high PAR. Notably, when using T5-large,
we increase PAR on NQ from 0.4966 to0.6425 compared to
RECOMP. We also achieve a PAR similar to LongLLMLingua,
which uses the token pruning method. Since token pruning
preserves key information based on a binary classification, it
is inherently easier to preserve tokens compared to abstrac-
tive compression. This demonstrates that ACoRN successfully
achieves its goal of recognizing and compressing evidential
information from the retrieved documents.
D. Impact of Training with Noise Documents Augmentation
To gain an understanding of the individual contribution
of each retrieval noise type within ACoRN to the overallperformance, we conduct an ablation study by comparing
the process with and without noise documents augmentation
to train a compressor. First, we evaluated the performance
of the two processes on three existing ODQA test datasets.
Next, we reconstructed the ODQA test datasets by retrieved
documents augmentation and compared the performance on
these reconstructed datasets. Then, we analyzed the degree of
performance degradation between the two test datasets. The
noise documents augmentation process for training reduced
the extent of performance degradation as the level of noise
increased. The results are shown in Table IV. Second, we com-
pared the changes in EM and F1 scores when each retrieval
noise type was incorporated to the evidential documents. When
only evidential documents were present, the noise documents
augmentation process for training had little impact. However,
as shown in Table V, when noise documents were provided
alongside evidential documents, this process improved the
noise robustness of the compressor.
VI. C ONCLUSION
In this paper, we introduce a simple training approach,
ACoRN, aimed at improving noise robustness during the com-
pression of retrieved documents. We improved the compressor
by focusing on two key aspects to ensure practical usability.
First, to be applicable in real-world scenarios, the compressor
must be noise-robust not only against irrelevant or insuffi-
cient documents but also against factual error documents that
provide incorrect information. Second, to address positional
bias in language models, we heuristically defined evidential
documents containing the answer string. These documents
enabled optimal summarization of key information while
mitigating positional bias through their distribution across
various positions. Additionally, we establish two benchmarks
to demonstrate the noise robustness of the compressor. The
first benchmark measures its ability to preserve the answer
string ratio and compression ratio during summarization. The
second benchmark assesses performance changes when noisy
documents are incorporated into evidential documents. We
verify that ACoRN notably increases the preserving answer
string ratio and improves noise robustness.
VII. L IMITATIONS
We analyze the limitations of our work and explore potential
improvements for future research. Two principal limitations
have been identified. First, we observed that the impact of
training with evidential documents diminishes when their num-
ber is high. Second, applying multi-hop reasoning with our

approach requires compressing evidential documents around
supporting facts rather than answer strings, which is challeng-
ing when such facts are not explicitly available in training data.
Moving forward, we will address these limitations and develop
efficient training methods to enhance noise robustness.
REFERENCES
[1] Y . B. Lee, U. Park, A. K. Jain, and S. W. Lee, “Pill-ID: Matching and
retrieval of drug pill images,” Pattern Recognition Letters, vol. 33, no.
7, pp. 904–910, 2012.
[2] P. Lewis, E. Perez, A. Piktus, F. Petroni, and V . Karpukhin et al.,
“Retrieval-augmented generation for knowledge-intensive NLP tasks,”
In Advances in Neural Information Processing Systems (NeurIPS), pp.
9459–9474, 2020.
[3] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child,
S. Gray, A. Radford, J. Wu, and D. Amodei, “Scaling laws for neural
language models,” arXiv preprint arXiv:2001.08361, 2020.
[4] S. An, Z. Ma, Z. Lin, N. Zheng, and J. G. Lou, “Make Your LLM Fully
Utilize the Context,” In Advances in Neural Information Processing
Systems (NeurIPS), 2024.
[5] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, and M. Bevilacqua et al., “Lost
in the middle: How language models use long contexts,” Transactions
of the Association for Computational Linguistics (TACL), vol.12, pp.
157–173, 2024.
[6] A. Mallen, A. Asai, V . Zhong, R. Das, and D. Khashabi et al., “When
not to trust language models: Investigating effectiveness of parametric
and non-parametric memories,” In Proceedings of the Annual Meeting of
the Association for Computational Linguistics (ACL), pp. 9802-–9822,
2023.
[7] Y . Li, B. Dong, F. Guerin, and C. Lin, “Compressing context to enhance
inference efficiency of large language models,” In Proceedings of the
Conference on Empirical Methods in Natural Language Processing
(EMNLP), pp. 6342-–6353, 2023.
[8] F. Shi, X. Chen, K. Misra, N. Scales, and D. Dohan et al., “Large
language models can be easily distracted by irrelevant context,” In
Proceedings of the International Conference on Machine Learning
(ICML), pp. 31210-–31227, 2023.
[9] J. Chen, H. Lin, X. Han, and L. Sun, “Benchmarking large language
models in retrieval-augmented generation,” In Proceedings of the As-
sociation for the Advancement of Artificial Intelligence (AAAI), pp.
17754–17762, 2024.
[10] X. Yin, B. Huang, and X. Wan, “ALCUNA: Large language models
meet new knowledge,” In Proceedings of the Conference on Empirical
Methods in Natural Language Processing (EMNLP), pp. 1397–1414,
2023.
[11] X. Zhu, J. Li, Y . Liu, C. Ma, and W. Wang, “A survey on model
compression for large language models,” Transactions of the Association
for Computational Linguistics (TACL), vol. 12, pp. 1556–1577, 2024.
[12] F. Xu, W. Shi, and E. Choi, “RECOMP: Improving retrieval-augmented
lms with context compression and selective augmentation,” In Interna-
tional Conference on Learning Representations (ICLR), 2024.
[13] C. Yoon, T. Lee, H. Hwang, M. Jeong, and J. Kang et al., “Compact:
Compressing retrieved documents actively for question answering,”
arXiv preprint arXiv:2407.09014, 2024.
[14] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, and A. Parikh et
al., “Natural questions: a benchmark for question answering research,”
Transactions of the Association for Computational Linguistics (TACL),
vol. 7, pp. 453-–466, 2019.
[15] M. Joshi, E. Choi, D. Weld, and L. Zettlemoyer, “TriviaQA: A Large
Scale Distantly Supervised Challenge Dataset for Reading Comprehen-
sion,” In Proceedings of the Annual Meeting of the Association for
Computational Linguistics (ACL), pp. 1601—1611, 2017.
[16] G. Hinton, O. Vinalys, and J. Dean, “Distilling the Knowledge in a
Neural Network,” arXiv preprint arXiv:1503.02531, 2015.
[17] T. Brown, B. Mann, N. Ryder, M. Subbiah, and J. D. Kaplan et
al., “Language models are few-shot learners,” In Advances in Neural
Information Processing Systems (NeurIPS), pp. 1877—1901, 2020.
[18] J. Wu, F. Che, C. Zhang, J. Tao, and S. Zhang et al., “Pandora’s Box or
Aladdin’s Lamp: A Comprehensive Analysis Revealing the Role of RAG
Noise in Large Language Models,” arXiv preprint arXiv:2408.13533,
2024.[19] D. Chen, A. Fisch, J. Weston, and A. Bordes. “Reading Wikipedia to
answer open-domain questions,” In Proceedings of the Annual Meeting
of the Association for Computational Linguistics (ACL) pp. 1870—
1879, 2017.
[20] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, and L. Jones et
al., “Attention is all you need,” In Advances in Neural Information
Processing Systems (NeurIPS), pp. 5998–6008, 2017.
[21] Y . Xu and M. Lapata, “Coarse-to-fine query focused multi-document
summarization,” In Proceedings of the Conference on Empirical Meth-
ods in Natural Language Processing (EMNLP), pp. 3632-–3645, 2020.
[22] H. Jiang, Q. Wu, X. Luo, D. Li, C. and Y . Lin et al., “Longllmlingua:
Accelerating and enhancing llms in long context scenarios via prompt
compression,” In Proceedings of the Annual Meeting of the Association
for Computational Linguistics (ACL), pp. 1658-–1677, 2024.
[23] S. Jha, L. E. Erdogan, S. Kim, K. Keutzer, and A. Gholami. “Charac-
terizing prompt compression methods for long context inference,” arXiv
preprint arXiv:2407.08892, 2024.
[24] Z. Wang, J. Araki, Z. Jiang, M. R. Parvez, and G. Neubig, “Learning
to filter context for retrieval-augmented generation,” arXiv preprint
arXiv:2311.08377, 2023.
[25] F. Fang, Y . Bai, S. Ni, M. Yang, and X. Chen et al., “Enhancing Noise
Robustness of Retrieval-Augmented Language Models with Adaptive
Adversarial Training,” In Proceedings of the Annual Meeting of the
Association for Computational Linguistics (ACL), pp. 93—102, 2024.
[26] F. Wang, X. Wan , R. Sun , J. Chen and S. ¨O. Arık, “Astute RAG: Over-
coming Imperfect Retrieval Augmentation and Knowledge Conflicts for
Large Language Models,” arXiv preprint arXiv:2410.07176, 2024.
[27] A. Asai, M. Gardner, and H. Hajishirzi, “Evidentiality-guided generation
for knowledge-intensive NLP tasks,” In Proceedings of the Conference
of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies (NAACL), pp. 2226-–2243,
2022.
[28] L. Zhuang, L. Wayne, S. Ya, and Z. Jun, “A robustly optimized
BERT pre-training approach with post-training,” In Chinese National
Conference on Computational Linguistics (CCL), pp. 1218-–1227, 2021.
[29] H. W. Chung, L. Hou, S. Longpre, B. Zoph, and Y . Tay et
al., “Scaling instruction-finetuned language models,” arXiv preprint
arXiv:2210.11416, 2022.
[30] C. Raffel, N. Shazeer, A. Roberts, K. Lee, and S. Narang et al.,
“Exploring the limits of transfer learning with a unified text-to-text
transformer,” Journal of Machine Learning Research (JMLR), vol. 21,
pp. 1-–67, 2020.
[31] V . Karpukhin, B. Oguz, S. Min, P. Lewis, and L. Wu et al., “Dense
passage retrieval for open-domain question answering,” arXiv preprint
arXiv:2004.04906, 2020.
[32] O. Yoran, T. Wolfson, O. Ram, and J. Berant, “Making retrieval-
augmented language models robust to irrelevvant context,” In Interna-
tional Conference on Learning Representations (ICLR), 2024.
[33] J. Kim, J. Schultz, T. Rohe, C. Wallraven, and S. W. Lee et al.,
“Abstract Representations of Associated Emotions in the Human Brain,”
Neuroscience, vol. 35, no. 14, pp. 5655—5663, 2015.
[34] M. Ravaut, A. Sun, N. Chen, and S. Joty. “On Context Utilization
in Summarization with Large Language Models,” In Proceedings of
the Annual Meeting of the Association for Computational Linguistics
(ACL), pp. 2764—2781, 2024.
[35] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang, “Retrieval aug-
mented language model pre-training,” In Proceedings of the International
Conference on Machine Learning (ICML), pp. 3929–3938, 2020.
[36] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, and A. Al-Dahle et al., “The
llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.
[37] W. Yu, H. Zhang, X. Pan, P. Cao, and K. Ma et al., “Chain-of-
note: Enhancing robustness in retrieval-augmented language models,” In
Proceedings of Conference on Empirical Methods in Natural Language
Processing (EMNLP), pp. 14672—14685, 2024.
[38] W. Wang, Y . Wang, Y . Fan, H. Liao, and J. Guo, “QUITO: Accelerating
Long-Context Reasoning through Query-Guided Context Compression,”
arXiv preprint arXiv:2408.00274, 2024.
[39] Y . K. Lim, S. H. Choi, and S. W. Lee, “Text extraction in MPEG
compressed video for content-based indexing,” In Preceedings of the
International Conference on Pattern Recognition (ICPR), vol. 4, pp. 409–
412, 2000.
[40] S. W. Lee, and H. H. Song, “A new recurrent neural-network architecture
for visual pattern recognition,” IEEE transactions on neural networks,
vol. 8, no. 2, pp. 331–340, 1997.
# HalluMix: A Task-Agnostic, Multi-Domain Benchmark for Real-World Hallucination Detection

**Authors**: Deanna Emery, Michael Goitia, Freddie Vargus, Iulia Neagu

**Published**: 2025-05-01 13:22:45

**PDF URL**: [http://arxiv.org/pdf/2505.00506v1](http://arxiv.org/pdf/2505.00506v1)

## Abstract
As large language models (LLMs) are increasingly deployed in high-stakes
domains, detecting hallucinated content$\unicode{x2013}$text that is not
grounded in supporting evidence$\unicode{x2013}$has become a critical
challenge. Existing benchmarks for hallucination detection are often
synthetically generated, narrowly focused on extractive question answering, and
fail to capture the complexity of real-world scenarios involving multi-document
contexts and full-sentence outputs. We introduce the HalluMix Benchmark, a
diverse, task-agnostic dataset that includes examples from a range of domains
and formats. Using this benchmark, we evaluate seven hallucination detection
systems$\unicode{x2013}$both open and closed
source$\unicode{x2013}$highlighting differences in performance across tasks,
document lengths, and input representations. Our analysis highlights
substantial performance disparities between short and long contexts, with
critical implications for real-world Retrieval Augmented Generation (RAG)
implementations. Quotient Detections achieves the best overall performance,
with an accuracy of 0.82 and an F1 score of 0.84.

## Full Text


<!-- PDF content starts -->

HalluMix: A Task-Agnostic, Multi-Domain Benchmark for Real-World
Hallucination Detection
Deanna Emery,
Michael Goitia ,Freddie Vargus ,Iulia Neagu
Quotient AI
{deanna, mike, freddie, julia}@quotientai.co
Abstract
As large language models (LLMs) are increas-
ingly deployed in high-stakes domains, de-
tecting hallucinated content—text that is not
grounded in supporting evidence—has become
a critical challenge. Existing benchmarks for
hallucination detection are often synthetically
generated, narrowly focused on extractive ques-
tion answering, and fail to capture the com-
plexity of real-world scenarios involving multi-
document contexts and full-sentence outputs.
We introduce the HalluMix Benchmark, a di-
verse, task-agnostic dataset that includes ex-
amples from a range of domains and formats.
Using this benchmark, we evaluate seven hal-
lucination detection systems—both open and
closed source—highlighting differences in per-
formance across tasks, document lengths, and
input representations. Our analysis highlights
substantial performance disparities between
short and long contexts, with critical implica-
tions for real-world Retrieval Augmented Gen-
eration (RAG) implementations. Quotient De-
tections achieves the best overall performance,
with an accuracy of 0.82 and an F1 score of
0.84.
1 Introduction
As large language models (LLMs) continue to gain
prominence across domains, ensuring the factual
correctness of their outputs has become a central
concern. A critical issue in this context is hallu-
cination , where the model generates content not
supported by, or contradictory to, a given source.
In high-stakes fields such as law, medicine, and
finance, such hallucinations can undermine trust
and lead to harmful consequences (Huang et al.,
2025).
While detecting hallucinations remains an active
area of research, progress has been hindered by
the lack of representative benchmarks. Most ex-
isting evaluation datasets are task-specific—often
focused on open-book question answering—andrely heavily on synthetic examples or narrow con-
text formats (Huang et al., 2025; Ravi et al., 2024;
Li et al., 2023; Niu et al., 2024; Yang et al., 2018).
This limits their generalizability to real-world set-
tings, where LLM outputs are typically multi-
sentence or paragraph responses grounded in multi-
document contexts.
We introduce the HalluMix Benchmark, a large-
scale, domain-diverse dataset specifically designed
to evaluate hallucination detection in realistic gen-
eration scenarios. Our dataset includes examples
drawn from multiple tasks—including summariza-
tion, question answering, and natural language in-
ference—and spans a wide array of domains such
as healthcare, law, science, and news. Each in-
stance consists of a multi-document context and a
response, with binary hallucination labels indicat-
ing whether the response is faithful to the provided
documents.
We further use this benchmark to systematically
evaluate seven state-of-the-art hallucination detec-
tion systems, including both open source and com-
mercial tools.
Our contributions are threefold:
•We propose a unified benchmark for hallu-
cination detection, constructed from high-
quality human-curated datasets spanning mul-
tiple tasks and domains.
•We introduce a consistent evaluation frame-
work that decouples hallucination detection
from task-specific assumptions (e.g. the pres-
ence of a question), reflecting more diverse
LLM use-cases.
•We conduct a comparative evaluation of exist-
ing hallucination detection methods, provid-
ing insights into their strengths, weaknesses,
and suitability for different real-world appli-
cations.
The following sections detail our benchmark
construction methodology, present our compara-arXiv:2505.00506v1  [cs.CL]  1 May 2025

tive evaluation of leading hallucination detection
systems, and discuss the implications of our find-
ings for both academic and industry applications.
2 The HalluMix Benchmark
Existing hallucination benchmark datasets are of-
ten generated by LLMs and are heavily focused
on question-answering tasks. In many cases, the
reference answers in these datasets are limited to
single-word spans extracted directly from a single
context, limiting their applicability to more com-
plex forms of generation. (Ravi et al., 2024; Li
et al., 2023; Niu et al., 2024; Yang et al., 2018).
However, hallucinations are not confined to
question-answering alone; they frequently arise in
other tasks such as summarization, dialogue, and
open-ended generation. Furthermore, real-world
LLM deployments typically use listsof documents,
often extracted via Retrieval Augmented Genera-
tion (RAG), to produce full-sentence outputs rather
than short extractive spans. To address these limi-
tations, we constructed a new benchmark that de-
couples hallucination detection from the question-
answering format. Each example consists of a
context (split into a list of text segments) and a
response, enabling evaluation based solely on the
factual consistency between the two.
To evaluate the performance of hallucination de-
tectors, we constructed the HalluMix Benchmark,
a diverse dataset that integrates samples from mul-
tiple human-curated sources. These include tasks
from summarization, natural language inference
(NLI), and question-answering (QA), covering a
wide array of domains such as news, science, law,
healthcare, dialogues, and long-form narratives.
Each example is labeled as either faithful orhallu-
cinated .
We selected datasets that are predominantly
human-labeled or human-curated. As shown in
the data process diagram in Figure 1, we applied a
variety of dataset-specific transformations to con-
struct hallucinated examples, while using original
annotations for faithful samples wherever possible.
2.1 Data Transformations
2.1.1 Natural Language Inference Datasets
NLI datasets were repurposed for hallucination de-
tection by reinterpreting their label schema. Each
example consists of a premise and a hypothesis ,
along with a label indicating their relationship. We
used the following mapping to convert the NLI
Figure 1: Overview of the HalluMix construction
pipeline, showing datasets and transformation strate-
gies.
labels to hallucination labels:
•Faithful: Hypotheses labeled as entailment .
•Hallucinated: Hypotheses labeled as neutral
orcontradiction .
In datasets with binary NLI labels ( entailment
vs.non-entailment ), we applied a similar map-
ping, treating non-entailment as hallucinated.
The following NLI datasets were used:
•sentence-transformers/all-nli
(Williams et al., 2018; Bowman et al.,
2015)
•stanfordnlp/snli (Bowman et al., 2015)
•snli-hard (Gururangan et al., 2018)
•glue: mnli, rte, and wnli (Wang et al.,
2018)
2.1.2 Summarization Datasets
Summarization datasets consist of long-form doc-
uments paired with human-written summaries.
Since summaries are designed to be faithful to the
original documents, we label these as faithful by de-
fault. To create hallucinated examples, we apply a
permutation-based transformation: summaries are
randomly mismatched with unrelated documents.

We include the following summarization
datasets:
•sentence-transformers/altex (Hidey and
McKeown, 2016)
•CNN/DailyMail (See et al., 2017)
•DialogSum (Chen et al., 2021)
•XSum (Narayan et al., 2018)
•arXiv summarization (Cohan et al., 2018a)
•GovReport summarization (Huang et al.,
2021)
•PubMed summarization (Cohan et al.,
2018b)
2.2 Question Answering Datasets
QA datasets contain a question, a context passage,
and a corresponding answer. By design, the an-
swers in these datasets are faithful . Hallucinated
variants are generated via multiple, dataset-specific
strategies. In some datasets, answers consisted of
single words; we used an LLM to expand these
into complete declarative sentences (e.g., Q: What
color is the car? A: Red →The car is red. ) to
ensure alignment with real-world use cases and
to separate dependence on questions to determine
hallucinations.
We used the following QA datasets:
•SQuAD-v2 (Rajpurkar et al., 2016, 2018):
Unanswerable questions (with blank answers)
were paired with LLM-generated answers
based solely on the question (without context),
labeled as hallucinated.
•DROP (Dua et al., 2019): Each context includes
multiple questions with typed answers (nu-
meric, date, string). Hallucinated examples
were created by replacing the correct answer
with another plausible answer of the same type
from within the same context.
•Databricks-Dolly-15K (Conover et al.,
2023) and PubMedQA (Jin et al., 2019): Hal-
lucinated examples were generated by mis-
matching answers and contexts.
•NarrativeQA (Koˇciský et al., 2018): This
dataset contains book-length texts, summaries,
questions, and answers. For tractability, we
primarily used the document summaries as
contexts. To preserve long-context evaluation,
we retained a small sample of shorter full texts.
Hallucinated examples were generated by mis-
matching answers with unrelated summaries
or passages.2.3 Final Dataset Structure
HalluMix is structured to support robust and flexi-
ble hallucination detection evaluation.
To better reflect real-world information retrieval
scenarios (i.e., RAG), each context was split into
even-sized chunks consisting of complete sen-
tences, ensuring that no chunk contained par-
tial or fragmented sentences. This approach pre-
served grammatical integrity while maintaining
manageable chunk lengths and preventing over-
segmentation. Additionally, we randomly shuf-
fled the document chunks to remove any ordering
advantages, as real-world retrieval systems often
return documents without preserving the original
narrative sequence.
To simulate realistic retrieval noise, we aug-
mented faithful examples with ten randomly se-
lected, irrelevant document chunks from unrelated
documents within the benchmark. The added con-
tent increases the challenge of identifying relevant
information without altering the evidence avail-
able for grounding the hypothesis. By applying
this augmentation exclusively to faithful examples,
we avoided inadvertently introducing supporting
evidence into hallucinated cases. This approach
creates an evaluation environment that mirrors
real-world conditions where hallucination detec-
tion systems must succeed despite noisy document
retrieval.
Each example in the final dataset includes:
•Adocuments field: the context represented as
a list of text chunks (e.g., tokenized sentences
or paragraph blocks),
•Ananswer : the hypothesis to be evaluated,
such as a summary sentence, answer, or claim,
•A binary hallucination label : where 0de-
notes faithful and1denotes hallucinated ,
•Asource identifier : to indicate the original
dataset for provenance tracking.
Representative examples of hallucinated and faith-
ful data points are provided in Tables 5 and 6 in the
Appendix.
Faithful examples ( label = 0 ) come directly
from human-labeled or human-curated datasets.
Hallucinated examples ( label = 1 ) were con-
structed, in some cases, through controlled transfor-
mations such as summary mismatches, QA context
permutations, or NLI relabeling. Due to these trans-
formations—including chunking, shuffling, distrac-
tor insertion, and label reassignment—each data

point in HalluMix has been substantially modified
and should not be considered equivalent to its orig-
inal source, even when the source identifier is pre-
served for tracking purposes.
The final dataset was de-duplicated and a strat-
ified random sample of 6.5k data points was col-
lected to achieve a balanced dataset with equal rep-
resentation across hallucination labels, data types
and sources. Each source has roughly equal rep-
resentation within each data type (NLI, QA, Sum-
marization), and each data type has roughly equal
representation across the benchmark dataset.
The resulting dataset offers a unified and exten-
sible benchmark for hallucination detection across
multiple domains, formats, and task settings.
3 Benchmarking Methodology
Using the HalluMix Benchmark, we compared the
performance of different hallucination detectors,
selecting methods based on practical deployment
considerations including model size ( ≤8B parame-
ters), inference cost, and latency requirements. Our
evaluation includes both open source and closed
source approaches:
•Llama-3-Patronus-Lynx-8B-Instruct-v1.1
(Ravi et al., 2024) - A Llama-3.1 model,
fine-tuned on hallucination detection datasets.
The output is a binary score.
•Ragas Faithfulness (Es et al., 2024) - A two-
step approach that uses an LLM to identify
the distinct claims within the model response,
then uses an LLM-as-a-Judge to determine
whether each claim is faithful to the source
documents. The output is the fraction of
claims that are faithful to the documents (i.e.
a value less than 1 indicates presence of hallu-
cination).
•Azure Groundedness (Azure AI Content
Safety, 2024) - A closed source API-based
hallucinations detector. The output is a binary
score.
•Vectara HHEM-2.1-Open (Forrest Bao and
Mendelevitch, 2024) - An open-weights ver-
sion of the HHEM-2.1 model. The output is
a likelihood (between 0-1) of faithfulness. In
this paper, we set a threshold such that values
less than 0.5 are predicted as hallucinated .
•Vertex AI Grounding (Google Vertex AI, 2025)
- A closed source API-based hallucinations
detector. The output is a likelihood (between
0-1) of faithfulness. In this paper, we set athreshold such that values less than 0.5 are
predicted as hallucinated .
•Bespoke-Minicheck-7B (Tang et al., 2024; Be-
spoke Labs, 2024) - A fine-tuned 7B param-
eter model that accepts a sentence and docu-
ment pair for evaluation. Multi-sentence re-
sponses are first tokenized into sentences be-
fore evaluation. The model returns a binary
score for each sentence. In this paper, if any
sentence is predicted to be hallucination, we
set the overall prediction to hallucinated .
•Quotient Detections - An LLM-as-a-Judge
that uses a sentence-based approach to iden-
tify hallucinations. The output is a binary
score indicating that at least one sentence con-
tains a hallucination.
The required inputs and input formats for each
of these hallucination detectors are different. Some
require a single context, while others accept a list
of documents; some require a question as an input,
while others only need context and response. Table
1 lists the input requirements for each hallucination
detection method.
Because our benchmark dataset is question-
agnostic, not all examples have an applicable ques-
tion; in these cases, we input the question as None
when required by the detector. For instances where
the detector did not accept a list of documents, we
instead join the documents with two new-line sep-
arators between each document. Both Patronus
Lynx 8B and Vectara HHEM-2.1-Open required a
single context input.
4 Results
We evaluated seven hallucination detection systems
on the HalluMix Benchmark, with performance
metrics shown in Table 2.
Quotient Detections achieved the highest overall
performance, leading in both Accuracy (0.82) and
F1 score (0.84), while maintaining a strong balance
between Precision (0.76) and Recall (0.93). While
Quotient Detections didn’t achieve the highest indi-
vidual precision or recall scores, the methods that
did excel in one metric showed substantial decline
in the other. For instance, Azure Groundedness1
demonstrates high precision (0.78) but achieves
1Azure Groundedness performance may be overestimated
due to its limits on the length of input documents. We were
unable to get Azure Groundedness evaluations on 304 of the
longest context examples. Generally, these long context exam-
ples are more challenging.

Single Context List of Documents Question Response
Quotient Detections - ✓ - ✓
Patronus Lynx 8B ✓ - ✓ ✓
Ragas Faithfulness - ✓ ✓ ✓
Azure Groundedness - ✓ Optional ✓
Vectara HHEM-2.1-Open ✓ - - ✓
Vertex AI Grounding - ✓ - ✓
Bespoke-Minicheck-7B - ✓ - ✓
Table 1: Input requirements and formats for each hallucination detection method. A checkmark indicates that the
field is required. A dash indicates that the field is not accepted. For Azure Groundedness, there are separate API
request formats for QA and summarization tasks, enabling hallucination detection both with and without a question.
Accuracy F1 Precision Recall
Quotient Detections 0.821 0.840 0.764 0.932
Bespoke Minicheck 7B 0.808 0.832 0.744 0.944
Patronus Lynx 8B 0.808 0.828 0.754 0.919
Ragas Faithfulness 0.787 0.818 0.719 0.950
Azure Groundedness* 0.784 0.788 0.781 0.795
Vectara HHEM-2.1-Open 0.749 0.771 0.715 0.836
Vertex AI Grounding 0.727 0.772 0.668 0.915
Table 2: Hallucination detection performance across methods evaluated on the full benchmark dataset. Quotient
Detections achieves the highest overall accuracy and F1 score, demonstrating balanced precision and recall. Azure
Groundedness1attains the highest precision but with low recall, whereas Ragas Faithfulness achieves the highest
recall at the expense of precision.
lower recall (0.79). Conversely, Ragas Faithfulness
shows high recall (0.95) but at the cost of precision
(0.72).
When examining performance across different
data sources (Table 3), we observe substantial vari-
ance both between methods and across datasets.
The most striking pattern emerges in Summa-
rization tasks, where performance diverges dra-
matically. Patronus Lynx 8B consistently outper-
forms other approaches on long-form summariza-
tion tasks. For example, on PubMed summa-
rization, it achieves 0.91 accuracy, compared to
0.63 for Quotient Detections and 0.58 for Bespoke-
Minicheck-7B .
Table 3 shows the accuracy scores of the halluci-
nation detection methods on each of the sub-data
sources within HalluMix. The high variance in
scores both across methods and across datasets
indicates that each detection method likely has
different strengths and weaknesses. Of note, the
methods that perform generally best in the NLI
and Question-Answering subsets tend to perform
more poorly on the Summarization subsets and vice
versa.
Table 4 illustrates the substantial differences incontent length across data types. NLI examples
are concise (averaging 11 tokens for responses and
88 for documents), while Summarization examples
involve much longer text (averaging 174 tokens for
responses and 439 for documents). These length
differences correlate strongly with detection perfor-
mance, suggesting that the quality hallucination de-
tection methods is dependent on the content length.
Figure 2 further demonstrates how content type
affects relative performance. When evaluating only
on shorter-context examples (NLI and QA sub-
sets, panel b), Patronus Lynx 8B drops from third
place to fifth in accuracy, while Quotient Detec-
tions maintains its lead. This shift underscores
how benchmark composition significantly influ-
ences performance rankings.
5 Discussion
Our comprehensive evaluation reveals several key
insights about the current state of hallucination de-
tection systems. While the best-performing models
achieve respectable accuracy overall, their effec-
tiveness varies depending on task type, content
length, and input format. These variations reflect
both the diversity of our benchmark dataset and

Quotient Bespoke Patronus Ragas Azure Vectara Vertex AI
Data Type Data Source Detections Minicheck 7B Lynx 8B Faithfulness Groundedness1HHEM-2.1-Open Grounding
NLIau123/snli-hard 0.856 0.921 0.686 0.813 0.878 0.789 0.694
nyu-mll/glue/mnli 0.847 0.847 0.662 0.759 0.912 0.802 0.751
nyu-mll/glue/rte 0.900 0.966 0.746 0.870 0.852 0.678 0.818
nyu-mll/glue/wnli 0.850 0.902 0.758 0.766 0.568 0.488 0.502
sentence-transformers/all-nli 0.901 0.908 0.673 0.821 0.936 0.821 0.755
stanfordnlp/snli 0.885 0.880 0.699 0.858 0.869 0.833 0.749
Question
AnsweringPubMedQA 0.624 0.572 0.928 0.586 0.596 0.670 0.542
databricks-dolly-15k 0.880 0.766 0.826 0.848 0.842 0.864 0.855
DROP 0.806 0.766 0.878 0.736 0.708 0.478 0.586
narrativeqa 0.886 0.858 0.916 0.864 0.748 0.832 0.758
squad_v2 0.920 0.912 0.890 0.892 0.912 0.818 0.890
Summarizationsentence-transformers/altlex 0.883 0.838 0.730 0.820 0.932 0.865 0.869
arxiv_summarization 0.614 0.591 0.926 0.702 0.568 0.926 0.633
cnn_dailymail 0.753 0.813 0.822 0.808 0.817 0.881 0.936
dialogsum 0.876 0.814 0.814 0.850 0.920 0.690 0.606
govreport_summarization 0.597 0.509 0.943 0.703 0.500 0.915 0.882
pubmed_summarization 0.629 0.582 0.911 0.695 0.615 0.892 0.803
xsum 0.715 0.720 0.725 0.617 0.798 0.606 0.601
Table 3: Accuracy scores of hallucination detection methods across data sources within the benchmark. Bold values
indicate the highest accuracy for each data source, while red values indicate the lowest. The substantial variation
across datasets suggests method-specific strengths, with some detectors excelling on specific data types while
underperforming on others—highlighting potential specialization or overfitting concerns in current hallucination
detection approaches.
(a) Performance on the full HalluMix benchmark
(including summarization data).
(b) Performance on the HalluMix benchmark
excluding summarization data.
Figure 2: Comparison of hallucination detection performance metrics (Accuracy, F1, Precision, Recall) across
all evaluated methods. Panel (a) shows performance on the complete benchmark dataset, while panel (b) shows
performance excluding summarization examples. Quotient Detections achieves highest accuracy and F1 in both
scenarios.
Avg Response Avg Document
Data Type Token Count Token Count
NLI 11 88
QA 32 167
Summ. 174 439
Table 4: Average token counts for each data type (NLI,
Question-Answering, and Summarization) in HalluMix.the design decisions embedded in each detection
method.
5.1 Evidence of Sub-Source Overfitting
Table 3 shows that some detection systems perform
exceptionally well on specific datasets while under-
performing on others. This pattern suggests that
certain hallucination detection methods may have
been trained on or heavily influenced by particular
sub-datasets, especially within the NLI and QA cat-
egories. For instance, high accuracy on well-known

datasets such as SNLI or SQuAD could indicate
exposure during pretraining or fine-tuning. While
this may not invalidate the performance, it does
raise questions about generalizability—particularly
in less conventional or domain-specific generation
scenarios.
An intriguing finding from our analysis is
that specialized, fine-tuned models like Patronus
Lynx 8B ,Vectara HHEM-2.1-Open , and Bespoke-
Minicheck-7B —despite being explicitly optimized
for hallucination detection—do not generally out-
perform methods that leverage general-purpose lan-
guage models with appropriate prompting strate-
gies, such as Ragas Faithfulness andQuotient De-
tections .
5.2 Content Length and Context
Representation Challenges
As shown in Table 4, the Summarization subset in-
volves substantially longer contexts and responses
than NLI or QA. The performance drop across most
models on summarization examples suggests that
long-form generation introduces additional chal-
lenges for hallucination detection, such as tracking
referents, maintaining discourse coherence, and
grounding claims across large textual spans.
Figure 3 reveals an important pattern in how
different architectural approaches handle content
length. Vectara HHEM-2.1-Open andPatronus
Lynx 8B —both fine-tuned models that process con-
tinuous rather than chunked context—consistently
demonstrate superior performance on longer con-
tent but struggle with shorter examples. In contrast,
sentence-based approaches like Quotient Detec-
tions andBespoke-Minicheck-7B excel with shorter
content ( ∼200 tokens) typical of NLI and QA tasks,
but show degraded performance on long-form sum-
marization examples.
This divergence in performance highlights funda-
mental trade-offs in context representation for hal-
lucination detection. Continuous-context methods
may better preserve document coherence, maintain-
ing critical discourse signals and cross-sentence
dependencies that support accurate faithfulness
assessment in longer texts. However, sentence-
based approaches offer greater precision for granu-
lar claim verification in shorter contexts but suffer
from information loss when processing longer doc-
uments.
Both sentence-based methods achieve recall
values approaching 1.0 on summarization exam-
ples, indicating they tend to over-predict hallucina-tions when evaluating long-form content—likely
because sentence isolation disrupts coreference
chains and other cross-sentence contextual signals
essential for accurate assessment.
These findings suggest several potential improve-
ments for sentence-based detectors. Incorporating
sliding window contexts that include neighboring
sentences during evaluation could help preserve
local coherence. Alternatively, a hierarchical veri-
fication approach might first evaluate individual
sentences and then perform a second-pass veri-
fication using full paragraph context. Such ap-
proaches could maintain the granularity advantages
of sentence-level detection while addressing the
context fragmentation issue.
5.3 Toward Robust Hallucination Detection
Future work for Quotient Detections will focus
on improving performance in long-context scenar-
ios. This includes exploring hybrid approaches that
maintain sentence-level granularity while leverag-
ing global document context. The goal is to develop
a detection system that remains effective across the
full spectrum of generation lengths, from brief fac-
tual claims to multi-paragraph summaries.
Overall, our findings emphasize the need for
hallucination detection systems that can operate re-
liably across varied input types, document lengths,
and generation formats. HalluMix provides a ro-
bust foundation for this line of research, and our
evaluation surfaces key directions for future im-
provement in both model architecture and input
handling strategies.
6 Conclusion
In this work, we introduced HalluMix , a large-
scale, task-diverse, and domain-spanning bench-
mark dataset for evaluating hallucination detection
in realistic language generation settings. Unlike
prior benchmarks, our dataset reflects the chal-
lenges of multi-document grounding and long-form
responses common in modern LLM deployments.
We systematically evaluated seven detection meth-
ods and revealed significant variation in their ef-
fectiveness based on input format, content length,
and underlying task. Overall, Quotient Detections
achieves the best performance in accuracy and F1.
Our analysis surfaced several key findings. Some
detection models appear to overfit to known
datasets, raising concerns about generalization.
Sentence-level approaches excel in shorter-content

(a) Accuracy by average document token count for each halluci-
nation detection method.
(b) Accuracy by answer token count for each hallucination de-
tection method.
Figure 3: Performance of hallucination detection methods as a function of content length. Panel (a) shows accuracy
versus average document token count, revealing how different methods handle increasingly complex contexts. Panel
(b) shows accuracy versus answer token count, demonstrating performance on longer responses. Both plots show
distinct performance patterns: some methods maintain consistent accuracy across lengths while others show clear
degradation with longer content.
detection but struggle with longer contexts, likely
due to loss of inter-sentential coherence. Mean-
while, models evaluated on full continuous text
may benefit from preserved context, highlight-
ing the limitations of segmented inputs typical
in retrieval-augmented generation pipelines. This
finding highlights critical trade-offs between gran-
ular claim verification and document-level coher-
ence assessment.
These findings have important implications for
LLM deployment in production systems. Organi-
zations implementing RAG-based solutions should
carefully consider the limitations of existing hallu-
cination detectors, particularly when working with
domain-specific content or long-form outputs.
Future directions include improving hallucina-
tion detection robustness across content lengths,
better modeling of discourse-level dependencies,
and adapting detectors to handle real-world, chun-
ked input more effectively. We make our bench-
mark publicly available to facilitate continued re-
search in this critical area of LLM safety and relia-
bility.
References
Azure AI Content Safety. 2024. Groundedness detec-
tion.Bespoke Labs. 2024. Bespoke-minicheck-7b.
Samuel R. Bowman, Gabor Angeli, Christopher Potts,
and Christopher D. Manning. 2015. A large anno-
tated corpus for learning natural language inference.
InProceedings of the 2015 Conference on Empiri-
cal Methods in Natural Language Processing , pages
632–642, Lisbon, Portugal. Association for Compu-
tational Linguistics.
Yulong Chen, Yang Liu, Liang Chen, and Yue Zhang.
2021. DialogSum: A real-life scenario dialogue sum-
marization dataset. In Findings of the Association
for Computational Linguistics: ACL-IJCNLP 2021 ,
pages 5062–5074, Online. Association for Computa-
tional Linguistics.
Arman Cohan, Franck Dernoncourt, Doo Soon Kim,
Trung Bui, Seokhwan Kim, Walter Chang, and Na-
zli Goharian. 2018a. A discourse-aware attention
model for abstractive summarization of long docu-
ments. In Proceedings of the 2018 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies, Volume 2 (Short Papers) , pages 615–621,
New Orleans, Louisiana. Association for Computa-
tional Linguistics.
Arman Cohan, Franck Dernoncourt, Doo Soon Kim,
Trung Bui, Seokhwan Kim, Walter Chang, and Na-
zli Goharian. 2018b. A discourse-aware attention
model for abstractive summarization of long docu-
ments. In Proceedings of the 2018 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies, Volume 2 (Short Papers) , pages 615–621,

New Orleans, Louisiana. Association for Computa-
tional Linguistics.
Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie,
Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell,
Matei Zaharia, and Reynold Xin. 2023. Free dolly:
Introducing the world’s first truly open instruction-
tuned llm.
Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel
Stanovsky, Sameer Singh, and Matt Gardner. 2019.
DROP: A reading comprehension benchmark requir-
ing discrete reasoning over paragraphs. In Proceed-
ings of the 2019 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies, Volume 1
(Long and Short Papers) , pages 2368–2378, Min-
neapolis, Minnesota. Association for Computational
Linguistics.
Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. RAGAs: Automated evalu-
ation of retrieval augmented generation. In Proceed-
ings of the 18th Conference of the European Chap-
ter of the Association for Computational Linguistics:
System Demonstrations , pages 150–158, St. Julians,
Malta. Association for Computational Linguistics.
Rogger Luo Forrest Bao, Miaoran Li and Ofer Mendele-
vitch. 2024. HHEM-2.1-Open.
Google Vertex AI. 2025. Check grounding with rag.
Suchin Gururangan, Swabha Swayamdipta, Omer Levy,
Roy Schwartz, Samuel Bowman, and Noah A. Smith.
2018. Annotation artifacts in natural language infer-
ence data. In Proceedings of the 2018 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies, Volume 2 (Short Papers) , pages 107–112,
New Orleans, Louisiana. Association for Computa-
tional Linguistics.
Christopher Hidey and Kathy McKeown. 2016. Identi-
fying causal relations using parallel Wikipedia arti-
cles. In Proceedings of the 54th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 1424–1433, Berlin, Ger-
many. Association for Computational Linguistics.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
Liu. 2025. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions. ACM Trans. Inf. Syst. , 43(2).
Luyang Huang, Shuyang Cao, Nikolaus Parulian,
Heng Ji, and Lu Wang. 2021. Efficient atten-
tions for long document summarization. Preprint ,
arXiv:2104.02112.
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W.
Cohen, and Xinghua Lu. 2019. Pubmedqa: A
dataset for biomedical research question answering.
Preprint , arXiv:1909.06146.Tomáš Ko ˇciský, Jonathan Schwarz, Phil Blunsom, Chris
Dyer, Karl Moritz Hermann, Gábor Melis, and Ed-
ward Grefenstette. 2018. The NarrativeQA reading
comprehension challenge. Transactions of the Asso-
ciation for Computational Linguistics , 6:317–328.
Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun
Nie, and Ji-Rong Wen. 2023. Halueval: A large-
scale hallucination evaluation benchmark for large
language models. Preprint , arXiv:2305.11747.
Shashi Narayan, Shay B. Cohen, and Mirella Lapata.
2018. Don‘t give me the details, just the summary!
topic-aware convolutional neural networks for ex-
treme summarization. In Proceedings of the 2018
Conference on Empirical Methods in Natural Lan-
guage Processing , pages 1797–1807, Brussels, Bel-
gium. Association for Computational Linguistics.
Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun
Shum, Randy Zhong, Juntong Song, and Tong Zhang.
2024. Ragtruth: A hallucination corpus for develop-
ing trustworthy retrieval-augmented language models.
Preprint , arXiv:2401.00396.
Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018.
Know what you don’t know: Unanswerable ques-
tions for SQuAD. In Proceedings of the 56th Annual
Meeting of the Association for Computational Lin-
guistics (Volume 2: Short Papers) , pages 784–789,
Melbourne, Australia. Association for Computational
Linguistics.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. SQuAD: 100,000+ questions for
machine comprehension of text. In Proceedings of
the 2016 Conference on Empirical Methods in Natu-
ral Language Processing , pages 2383–2392, Austin,
Texas. Association for Computational Linguistics.
Selvan Sunitha Ravi, Bartosz Mielczarek, Anand Kan-
nappan, Douwe Kiela, and Rebecca Qian. 2024.
Lynx: An open source hallucination evaluation
model. Preprint , arXiv:2407.08488.
Abigail See, Peter J. Liu, and Christopher D. Manning.
2017. Get to the point: Summarization with pointer-
generator networks. In Proceedings of the 55th An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers) , pages 1073–
1083, Vancouver, Canada. Association for Computa-
tional Linguistics.
Liyan Tang, Philippe Laban, and Greg Durrett. 2024.
Minicheck: Efficient fact-checking of llms on ground-
ing documents. In Proceedings of the 2024 Confer-
ence on Empirical Methods in Natural Language
Processing . Association for Computational Linguis-
tics.
Alex Wang, Amanpreet Singh, Julian Michael, Felix
Hill, Omer Levy, and Samuel Bowman. 2018. GLUE:
A multi-task benchmark and analysis platform for nat-
ural language understanding. In Proceedings of the
2018 EMNLP Workshop BlackboxNLP: Analyzing
and Interpreting Neural Networks for NLP , pages

353–355, Brussels, Belgium. Association for Com-
putational Linguistics.
Adina Williams, Nikita Nangia, and Samuel Bowman.
2018. A broad-coverage challenge corpus for sen-
tence understanding through inference. In Proceed-
ings of the 2018 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies, Volume 1
(Long Papers) , pages 1112–1122. Association for
Computational Linguistics.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing.Preprint , arXiv:1809.09600.

A Appendix
Documents
•Due to the Steelers’ loss to the Ravens the previous day, the Bengals
entered the game as the AFC North champions. The Bengals rushed
out to a 14-0 lead in the first half on a McCarron touchdown pass
and a Mohamed Sanu rush, but Denver cut the deficit to 11 points as
Brandon McManus nailed a short 23-yard field goal with just 18
seconds remaining before halftime. In the second half, momentum
shifted mightily after a missed field goal by Mike Nugent in the third.
Emmanuel Sanders hauled in an 8-yard pass from Brock Osweiler to
cut the deficit to 14-10, and Denver claimed the lead for the first time
in the game on a 39-yard touchdown run by C.J. Anderson with
11:17 remaining in the 4th Quarter. The Bengals marched down the
field to tie the game on Mike Nugent’s season-long 52-yard field
goal, making the score 17-17 at the end of regulation. The tired
Bengals failed to put any points on the board in the extra period,
allowing a 37-yard McManus field goal to make the score 20-17
Denver. A botched snap on the ensuing Bengals drive was recovered
by the Broncos, ending the game and Cincinnati’s hopes for a
first-round bye in the playoffs. With the loss, the Bengals fell to 11-4
on the season. The loss was also the 10th straight in Denver for the
Bengals, dating back to 1975.
Response The first field goal was by the Ravens.
Label Hallucinated
Table 5: An example of a hallucinated datapoint in the HalluMix Benchmark.

Documents
• Final Fantasy is a Japanese science fantasy anthology media
franchise created by Hironobu Sakaguchi and developed and owned
by Square Enix (formerly Square).
•Peter Wright, a law supervisor for the DNR, told WLUC-TV that the
officer was just doing his job. He said the officer believed it was a
feral pig, since it had no identifying marks to distinguish him as a
pet. ’I want to make it very clear that it’s never ever, ever the
department’s position that we want to shoot people’s pets,’ said
Wright. ’If he had any inkling it was a pet, he absolutely wouldn’t
have shot it.’ Upsetting: The family are now trying to get Caesar’s
body in order to bury him, but have been told they can only take
possession of his ashes . Brandy Savelle and Tony Gervasi are now
trying to get Caesar’s body back. However they have been told they
can only take possession of ashes. Ms Savelle is demanding that
some sort of recourse comes out of the situation. ’If it was that big of
a mistake then we would like to see better training,’ she said. ’Let’s
learn to identify not just pigs, but all pets.’
• God Hates Us All is the eighth studio album by American thrash
metal band Slayer .
• that’s right that’s exactly right so but a lot of more women are
starting their own businesses i’ve noticed than
• The franchise centers on a series of fantasy and science fantasy
role-playing video games. The first game in the series was released
in 1987, with 15 numbered main entries having been released to date.
• Shortly after 3600 BC Egyptian society began to grow and advance
rapidly toward refined civilization .
• boy pushing wagon with two pumpkins in it
Response Final Fantasy was created by Hironobu Sakaguchi
Label Faithful
Table 6: An example of a faithful datapoint in the HalluMix Benchmark.
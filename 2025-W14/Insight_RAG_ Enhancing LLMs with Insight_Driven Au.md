# Insight-RAG: Enhancing LLMs with Insight-Driven Augmentation

**Authors**: Pouya Pezeshkpour, Estevam Hruschka

**Published**: 2025-03-31 19:50:27

**PDF URL**: [http://arxiv.org/pdf/2504.00187v1](http://arxiv.org/pdf/2504.00187v1)

## Abstract
Retrieval Augmented Generation (RAG) frameworks have shown significant
promise in leveraging external knowledge to enhance the performance of large
language models (LLMs). However, conventional RAG methods often retrieve
documents based solely on surface-level relevance, leading to many issues: they
may overlook deeply buried information within individual documents, miss
relevant insights spanning multiple sources, and are not well-suited for tasks
beyond traditional question answering. In this paper, we propose Insight-RAG, a
novel framework designed to address these issues. In the initial stage of
Insight-RAG, instead of using traditional retrieval methods, we employ an LLM
to analyze the input query and task, extracting the underlying informational
requirements. In the subsequent stage, a specialized LLM -- trained on the
document database -- is queried to mine content that directly addresses these
identified insights. Finally, by integrating the original query with the
retrieved insights, similar to conventional RAG approaches, we employ a final
LLM to generate a contextually enriched and accurate response. Using two
scientific paper datasets, we created evaluation benchmarks targeting each of
the mentioned issues and assessed Insight-RAG against traditional RAG pipeline.
Our results demonstrate that the Insight-RAG pipeline successfully addresses
these challenges, outperforming existing methods by a significant margin in
most cases. These findings suggest that integrating insight-driven retrieval
within the RAG framework not only enhances performance but also broadens the
applicability of RAG to tasks beyond conventional question answering.

## Full Text


<!-- PDF content starts -->

Insight-RAG: Enhancing LLMs with Insight-Driven Augmentation
Pouya Pezeshkpour
Megagon Labs
pouya@megagon.aiEstevam Hruschka
Megagon Labs
estevam@megagon.ai
Abstract
Retrieval Augmented Generation (RAG) frame-
works have shown significant promise in lever-
aging external knowledge to enhance the per-
formance of large language models (LLMs).
However, conventional RAG methods often re-
trieve documents based solely on surface-level
relevance, leading to many issues: they may
overlook deeply buried information within indi-
vidual documents, miss relevant insights span-
ning multiple sources, and are not well-suited
for tasks beyond traditional question answering.
In this paper, we propose Insight-RAG , a novel
framework designed to address these issues. In
the initial stage of Insight-RAG, instead of us-
ing traditional retrieval methods, we employ
an LLM to analyze the input query and task,
extracting the underlying informational require-
ments. In the subsequent stage, a specialized
LLM—trained on the document database—is
queried to mine content that directly addresses
these identified insights. Finally, by integrating
the original query with the retrieved insights,
similar to conventional RAG approaches, we
employ a final LLM to generate a contextually
enriched and accurate response. Using two sci-
entific paper datasets, we created evaluation
benchmarks targeting each of the mentioned
issues and assessed Insight-RAG against tradi-
tional RAG pipeline. Our results demonstrate
that the Insight-RAG pipeline successfully ad-
dresses these challenges, outperforming exist-
ing methods by a significant margin in most
cases. These findings suggest that integrating
insight-driven retrieval within the RAG frame-
work not only enhances performance but also
broadens the applicability of RAG to tasks be-
yond conventional question answering. We re-
leased our dataset and code1.
1 Introduction
Recent advancements in large language models
(LLMs) have spurred renewed interest in Retrieval
1https://github.com/megagonlabs/Insight-RAG
Figure 1: In conventional RAG, using a retriever model,
we first retrieve relevant documents to answer a question.
In contrast, in Insight-RAG , we first identify necessary
insights to solve the task (e.g., answering a question),
and then feed the identified insights to an LLM con-
tinually pre-trained over the documents to extract the
necessary insights before feeding them to the final LLM
to solve the task.
Augmented Generation (RAG) frameworks (Gao
et al., 2023; Fan et al., 2024). RAG has emerged
as a powerful solution for mitigating inherent chal-
lenges in LLMs—such as hallucination and the
lack of recent information—by integrating exter-
nal document repositories with retrieval models
to produce contextually enriched responses. How-
ever, conventional RAG pipelines typically rely on
surface-level relevance metrics for document re-
trieval, which can result in several limitations: they
may overlook deeply buried information within
individual documents and miss relevant insights
distributed across multiple sources. Beyond these
retrieval challenges, traditional RAG frameworks
lack well-defined solutions for tasks that extend
beyond standard question answering.
Traditional retrieval mechanisms often fail to
capture the nuanced insights required for complex
tasks (Barnett et al., 2024; Agrawal et al., 2024;
Wang et al., 2024). For example, they may over-arXiv:2504.00187v1  [cs.CL]  31 Mar 2025

look deeply buried details within a single docu-
ment—such as subtle contractual clauses in a le-
gal agreement or hidden trends in a business re-
port—and may neglect relevant insights dispersed
across multiple sources, like complementary per-
spectives from various news articles or customer
reviews. Moreover, these methods are not well-
equipped for tasks beyond straightforward question
answering, such as identifying the best candidate
for a job by leveraging insights from a database
of resumes or extracting actionable recommenda-
tions for business strategy from qualitative feed-
back gathered from surveys and online reviews.
In this paper, we propose Insight-RAG—a novel
framework that refines the retrieval process by in-
corporating an intermediary insight extraction step
(See Figure 1). In the first stage, an LLM ana-
lyzes the input query and extracts the essential in-
formational requirements, effectively acting as an
intelligent filter that isolates critical insights from
the query context. This targeted extraction enables
the system to focus on deeper, task-specific con-
text. Subsequently, a specialized LLM continually
pre-trained (Ke et al., 2023) with LoRA (Hu et al.,
2021; Zhao et al., 2024a; Biderman et al., 2024)
(CPT-LoRA) on the target domain-specific corpus
leverages these identified insights to retrieve highly
relevant information from the document database.
Finally, the original input—now augmented with
these carefully retrieved insights—is processed by
a final LLM to generate a context-aware response.
To evaluate Insight-RAG, we use two scientific
paper datasets—AAN (Radev et al., 2013) and
OC (Bhagavatula et al., 2018)—and create tailored
datasets to address each RAG aforementioned chal-
lenge. We sample 5,000 papers from each dataset
using a Breadth-First Search strategy and extract
triples with GPT-4o mini (Hurst et al., 2024), fol-
lowed by manual/rule-based filtering and normal-
ization. For the deeply buried information chal-
lenge, we focus on subject-relation pairs that yield
a single object, selecting only those triples where
both the subject and object appear only once in
each document. For the multi-source challenge,
we choose subject-relation pairs that yield multiple
objects from different documents. We then, manu-
ally filter the samples after translating each triple
into a question using GPT-4o mini. Finally, for the
non-QA task challenge, we use the matching labels
between papers, capturing the citation recommen-
dation task, provided by Zhou et al. (2020).
By integrating multiple LLMs to compareInsight-RAG with the conventional RAG approach,
we observe that Insight-RAG can achieve up to 60
percentage points improvement in accuracy with
much less contextual information, for both deeply
buried and multi-source questions. Moreover, we
observe that for non-QA tasks such as paper match-
ing, Insight-RAG consistently helps improve per-
formance by up to 5.4 percentage points in accu-
racy, while using RAG shows mixed results, some-
times increasing and sometimes decreasing the per-
formance. Through various ablation studies, we
then connect models behavior to the performance
of different components in the pipelines, paving
the way for future applications of Insight-RAG.
2 Insight-RAG
In this section, we detail our proposed Insight-RAG
framework, which consists of three key units de-
signed to overcome the limitations of conventional
RAG approaches (see Figure 1). By incorporating
an intermediary insight extraction stage, our frame-
work captures nuanced, task-specific information
that traditional methods often miss. The pipeline
comprises the following units:
Insight Identifier: The Insight Identifier unit pro-
cesses the input to extract its essential informa-
tional requirements. Serving as an intelligent filter,
it isolates critical insights from both the input and
the task context, ensuring that subsequent stages
concentrate on deeper, necessary content. To fa-
cilitate this process, we employ LLMs guided by
a carefully designed prompt (provided in the Ap-
pendix).
Insight Miner: Inspired by previous work
(Pezeshkpour and Hruschka, 2025), the insight
miner unit leverages a specialized LLM to fetch
content for the insights identified earlier. We
adopt Llama-3.2 3B (Grattafiori et al., 2024) as
our insight-miner, continually pre-training it with
LoRA (Zhao et al., 2024a; Biderman et al., 2024)
over our scientific paper datasets. In line with
the previous work on insight mining (Pezeshkpour
and Hruschka, 2025), we continually pre-train the
model on both the original papers and the extracted
triples from them (see Section 3). This continual
pre-training enables the insight-miner to retrieve
highly relevant information to the task.
Response Generator: The final unit, response
generator, integrates the original query with the re-
trieved insights and employs a final LLM to gener-

Figure 2: We create our benchmark in several steps: 1) extracting triples from domain-specific documents using
GPT-4o mini and then manually normalizing/filtering them, 2) filtering the triples for each different type of issue, 3)
using GPT-4o mini to translate the sampled triples to question format, asking about the object of the triple.
ate a comprehensive, context-aware response. Fol-
lowing the conventional RAG approach, this aug-
mented input allows the model to produce outputs
that are both accurate and enriched by the addi-
tional insights. The prompt used for this stage is
provided in the Appendix.
3 Benchmarking
To evaluate the performance of our Insight-RAG
framework, we employ two scientific paper’s ab-
stract datasets—AAN and OC (provided by Zhou
et al. (2020))—to create tailored evaluation bench-
marks that address specific challenges encountered
in conventional RAG pipelines. Figure 2 provides
an overview of our process for creating the bench-
marks. Below, we detail our benchmarking pro-
cess for each identified issue. We provide the data
statistics of created benchmarks in Table 1 and the
prompts used in creating the benchmark in the Ap-
pendix.
Deeply Buried Insight: In this issue, our focus
is on the challenge of capturing deeply buried in-
formation within individual documents. We begin
by sampling 5,000 papers from each dataset using
a Breadth-First Search (BFS) strategy. From these
papers, following previous works (Papaluca et al.,
2023; Wadhwa et al., 2023), we use GPT-4o mini to
extract triples (we used the same prompt provided
in Pezeshkpour and Hruschka (2025)), followed by
manual/rule-based filtering and normalizing the re-
lations. Then, we select subject-relation pairs thatANN OC
# Docs 5,000 5,000
# Triples 21,526 23,662
# Deep-Insight Samples 318 403
# Multi-Source Samples 173 90
# Matching Samples 500 500
Table 1: Data statistics of the created benchmark.
yield a single object and ensure that both the sub-
ject and the object appear only once in the paper’s
abstract. This constraint guarantees that the ex-
tracted information is deeply buried and not overly
prominent, thereby testing the framework’s ability
to capture subtle details. We then convert the cu-
rated triples into question formats using GPT-4o
mini—which generates questions about the object
based on the subject-relation pair—and manually
filtered them for quality.
Multi-Source Insight: To assess the capability
of Insight-RAG in synthesizing information from
multiple sources, we incorporate the extracted
triples from the papers. More specifically, we fo-
cus on subject-relation pairs that yield multiple
objects drawn from different papers, thereby sim-
ulating scenarios where relevant insights are dis-
tributed across various sources. Once the multi-
source triples are curated, we convert them into
question formats using GPT-4o mini. Acknowl-
edging that some extracted triples may be noisy or
vague (e.g., constructs like "<we, show, x>"), we
manually filter the questions to ensure quality.

Non-QA Task: The third benchmark addresses
tasks beyond traditional question answering, specif-
ically evaluating the framework’s applicability for
citation recommendation. For this benchmark, we
leverage the matching labels between papers pro-
vided by Zhou et al. (2020), which capture the
citation recommendation task. Our goal is to de-
termine if the insights extracted from a document
database can effectively support solving arbitrary
tasks on inputs that share similarities with the doc-
uments, thereby extending the RAG framework’s
utility to a variety of real-world applications.
4 Experimental Details
We employ several state-of-the-art LLMs as in-
tegral components of the Insight-RAG pipeline:
GPT-4o (Hurst et al., 2024), GPT-4o mini, o3-
mini (OpenAI, 2025), Llama3.3 70B (Grattafiori
et al., 2024), and DeepSeek-R1 (Guo et al., 2025).
For the Insight Miner unit, we adopt Llama-3.2
3B as our insight-miner, continually pre-trained
with LoRA on domain-specific scientific papers
and extracted triples. We hyperparameter-tuned
the Llama-3.2 3B model based on loss, with addi-
tional training and datasets details provided in the
Appendix. Moreover, in the Insight-RAG pipeline,
we use the same LLM for both the Insight Iden-
tifier and Response Generator. For RAG Base-
lines, we used LlamaIndex (Liu, 2022) and the
embedding model gte-Qwen2-7B-instruct (Li et al.,
2023), which is the open-sourced state-of-the-art
model based on the MTEB leaderboard (Muen-
nighoff et al., 2022). Finally, for fair comparison,
we limit the insight miner’s maximum generated
token length to 100 tokens for both datasets, which
is less than the average document token length of
134.6 and 226.4 for AAN and OC, respectively. We
observe that further increasing the maximum gen-
erated token length does not significantly change
the performance. We evaluate LLM performance
using accuracy, exact match accuracy (calculated
by determining if the gold response exactly appears
in the generated response), and F1 Score (standard
QA metrics). We also employ Recall@K, which
measures the proportion of correct predictions in
the top-k results.
5 Experiments
In this section, our goal is to investigate the impact
of Insight-RAG in addressing the aforementioned
challenges—namely, deeply buried insights, multi-source information, and non-QA tasks. We begin
by evaluating the considered LLMs using our cre-
ated benchmarks. By analyzing the models’ be-
havior, we then explore the reasoning behind their
performance, examining each component of the
Insight-RAG pipeline and assessing the quality of
the identified insights.
5.1 Answering Questions using Deeply Buried
Insights
Figure 3 presents the exact match accuracy of
Insight-RAG versus conventional RAG using vari-
ous LLMs for answering questions based on deeply
buried information. First, the zero-shot perfor-
mance of all LLMs—i.e., without any context or
documents—is very low. This is primarily due to
the domain-specific nature of the questions, which
leaves the LLMs without the necessary informa-
tion to solve the task. Additionally, the questions
themselves may be ambiguous or even erroneous
when isolated; however, providing the associated
document context alleviates these issues.
As observed, Insight-RAG, even with only one
generated insight from the insight miner, achieves
significantly higher performance compared to the
conventional RAG approach. Although increas-
ing the number of retrieved documents improves
the performance of RAG, it still falls considerably
short of Insight-RAG. We suspect that the short-
comings of the RAG-based solution are due to
retrieval errors (as confirmed in Section 5.4) and
discrepancies in phrasing between the generated
questions and the text, which negatively impact per-
formance (Modarressi et al., 2025). DeepSeek-R1
performs best, followed by Llama-3.3, both out-
performing the OpenAI models. In contrast, o3
mini demonstrates the worst performance, primar-
ily because it tends to overthink the task, which is
reflected in its insight identifier performance (Sec-
tion 5.4).
We also report F1 performance of models in the
Appendix. Surprisingly, we observe that despite the
superior performance of DeepSeek in Exact Match,
its performance drops significantly in F1. Upon
further investigation, we observe that this is mostly
due to DeepSeek’s tendency to generate unneces-
sary content and occasional hallucinations, espe-
cially when the right document is not retrieved (we
removed the thinking part of DeepSeek-generated
answers to calculate the F1). Other models show
similar behavior as in Exact Match, with Llama-3.3
70B emerging as the best-performing model.

0 10 20 30 40 50
# Documents010203040506070Accuracy (%)
(a) AAN
0 10 20 30 40 50
# Documents010203040506070Accuracy (%)
GPT-4o mini + RAG
GPT-4o mini + Insight-RAG (1 generated insight)
GPT-4o + RAG
GPT-4o + Insight-RAG (1 generated insight)
o3-mini + RAG
o3-mini + Insight-RAG (1 generated insight)
Llama-3.3 70B + RAG
Llama-3.3 70B + Insight-RAG (1 generated insight)
DeepSeek-R1 + RAG
DeepSeek-R1 + Insight-RAG (1 generated insight) (b) OC
Figure 3: The performance comparison of RAG versus Insight-RAG across the AAN and OC datasets in answering
question based on deeply buried information. As demonstrated, DeepSeek-R1 performed the best, followed by
Llama-3.3 70B. Moreover, we observe that Insight-RAG, even with only one generated insight, outperforms RAG-
based solutions by a considerable margin. Additionally, while retrieving more documents reduces this performance
gap, Insight-RAG maintains a significant advantage.
Finally, focusing on DeepSeek-R1 because of its
superior performance, we report its RAG-based per-
formance when, instead of retrieving documents,
we retrieve triples from the set of all extracted
triples for each dataset (see the Appendix). We
observe that the model shows similar behavior to
document-based RAG, but with much less context—
since a triple is much shorter than a document—and
still falls significantly short compared to Insight-
RAG performance. This further highlights the
shortcomings of conventional retrieval approaches
and the complexity of resolving them.
5.2 Aggregating Information from Multiple
Sources
We present the averaged exact match accuracy (cal-
culated over gold answers for each sample) of
Insight-RAG versus conventional RAG using vari-
ous LLMs for answering questions based on infor-
mation from multiple sources in Figure 4. While
using the same number of retrieved documents and
generated insights, Insight-RAG consistently out-
performs the conventional RAG approach. More-
over, Insight-RAG performance increases rapidly
with only a few generated insights, and then its rate
of improvement slows down as more generated
insights are added. Although increasing the num-
ber of retrieved documents improves RAG’s per-
formance, it still falls short of Insight-RAG, even
though the performance gap narrows. Overall per-
formance in the multi-source scenario is lower com-
pared to the deeply buried information evaluation,
yet the same pattern of Insight-RAG’s superior-
ity is evident. Finally, DeepSeek-R1 remains thetop performer, followed by Llama, both of which
surpass the OpenAI models. We also report the
average F1 scores and triple-based RAG perfor-
mance for DeepSeek-R1 in the Appendix. Notably,
the performance trends mirror those observed in
the F1 metrics for questions on deeply buried in-
formation. For triple-based RAG, we observe a
degradation in performance—it yields results simi-
lar to document-based RAG but when using similar
number of tokens in the context.
5.3 RAG in Non-QA Tasks
In this section, we evaluate RAG-based solutions
on a non-question answering task—specifically, a
matching task for citation recommendation. For
the RAG baseline, we retrieve only one document
because the matching task is not well-defined for
traditional RAG approaches, and our experiments
did not show any improvement when retrieving
additional documents.
Our results, presented in Table 2, indicate that
Insight-RAG consistently outperforms the conven-
tional RAG baseline. This improvement is more
pronounced on the OC dataset, likely due to the
lower zero-shot performance of the LLMs on that
dataset. The subjective nature of the matching
task (particularly in the AAN dataset) constrains
the potential for improvement, resulting in a mod-
est performance gain. Furthermore, the RAG
baseline demonstrates mixed impacts—yielding
both positive and negative effects on model per-
formance across different configurations. Notably,
the o3 mini achieves the best overall performance,
whereas DeepSeek-R1 performs the worst. Upon

0 10 20 30 40 50
# Documents/Generated Insights0102030405060AVG-Accuracy (%)
(a) AAN
0 10 20 30 40 50
# Documents/Generated Insights1020304050AVG-Accuracy (%)
GPT-4o mini + RAG
GPT-4o mini + Insight-RAG
GPT-4o + RAG
GPT-4o + Insight-RAG
o3-mini + RAG
o3-mini + Insight-RAG
Llama-3.3 70B + RAG
Llama-3.3 70B + Insight-RAG
DeepSeek-R1 + RAG
DeepSeek-R1 + Insight-RAG (b) OC
Figure 4: The performance comparison of RAG versus Insight-RAG across the AAN and OC datasets in answering
questions requiring information from multiple sources. As demonstrated, DeepSeek-R1 performed the best, followed
by Llama-3.3 70B. Moreover, we observe that Insight-RAG with only a few generated insights achieves a much
higher performance, with the performance continuing to improve at a reduced rate as more insights are added.
ModelANN OC
Vanilla RAG (1 doc) Insight-RAG Vanilla RAG (1 doc) Insight-RAG
GPT-4o mini 80.8 81.6 (+0.8) 82.8 (+2.0) 74.4 70.0 (-4.4) 78.0 (+3.6)
GPT-4o 84.0 80.4 (-3.6) 84.0 (0.0) 71.6 73.6 (+2.0) 74.0 (+2.4)
o3 mini 85.4 85.6 (+0.2) 85.6 (+0.2) 77.0 74.2 (-2.8) 82.0 (+5.0)
Llama 3.3 70B 83.8 79.2 (-4.6) 84.4 (+0.6) 79.0 77.8 (-1.2) 81.4 (+2.4)
DeepSeek-R1 70.4 74.0 (+3.6) 73.8 (+3.4) 66.6 71.4 (+4.8) 72.0 (+5.4)
Table 2: The performance comparison of RAG versus Insight-RAG across the AAN and OC datasets in the paper
matching task. As demonstrated, o3 mini performs the best while DeepSeek-R1 shows the lowest performance.
Moreover, we observe that Insight-RAG consistently improves performance across all models, while RAG-based
solutions show mixed impacts on model performance.
further investigation, we found that DeepSeek-R1
tends to unnecessarily overthink the task, which
negatively impacts its performance. These findings
underscore the effectiveness of the insight-driven
approach in extending RAG to tasks beyond ques-
tion answering and highlight the need for tailored
retrieval strategies in non-QA contexts.
5.4 Components Analysis
In this section, we analyze the performance of the
two key components of the Insight-RAG frame-
work—Insight Identifier and Insight Miner—in ad-
dition to the retriever performance of RAG base-
lines, and discuss how their individual contribu-
tions drive the overall success of the systems.
Insight Identifier: The Insight Identifier plays
a crucial role by processing the input query and
distilling the essential informational requirements.
To measure the accuracy of the Insight Identifier
for deeply buried and multi-source questions, we
compare the identified insights with the gold in-
sights (which are concatenations of the subject and
relation used to generate the questions). We ask
AAN-Deep AAN-MultiOC-Deep OC-MultiGPT-4o mini
GPT-4o
o3 mini
Llama-3.3 70B
DeepSeek-R197.7 91.8 96.4 93.1
97.0 90.6 96.5 87.3
95.9 91.2 95.4 88.5
96.6 94.7 95.7 95.9
97.0 95.0 97.8 93.6
8890929496Figure 5: Insight Identifier performance: We ask GPT-
4o mini to score the identified insights compared to the
gold insights using a three-point scale: 0 (not similar),
0.5 (partially similar), and 1 (completely similar).
GPT-4o mini to score their similarity using a three-
point scale: 0 (not similar), 0.5 (partially similar),
and 1 (completely similar). We provide the prompt
in the Appendix.
As shown in Figure 5, all models demonstrate
high performance in identifying insights given sim-
ple questions. The o3 mini shows the lowest per-

formance, which aligns with its overall accuracy in
answering the questions. We speculate that this is
mostly due to this model’s tendency to overthink
the task. Moreover, all models show lower per-
formance in multi-source questions compared to
deeply buried questions, which is due to the fact
that when GPT-4o mini translates triples into ques-
tion format, it tends to add more unnecessary words
in multi-source questions (to capture the fact that
there is more than one answer).
Insight Miner: We calculate the accuracy of the
Insight Miner in predicting the object given the
concatenation of subject and relation used to create
questions in both deeply buried and multi-source
questions. Table 3 summarizes the Insight Miner’s
performance based on exact match accuracy for
deeply buried questions and recall@10 for multi-
source questions, respectively.
Our results indicate that continual pre-training of
Llama3.2 3B using LoRA on both the original pa-
pers and the extracted triples leads to a reasonably
well-performing Insight Miner, with higher perfor-
mance on deeply buried questions versus multi-
source questions. This difference is probably due
to the fact that it is easier for the model to learn
information about the pair of subject and relation
with one object compared to cases when there are
multiple objects for a given subject-relation pair.
Retriever: Given our knowledge of each ques-
tion’s source paper, we can evaluate the retriever
model’s accuracy in fetching relevant documents
for both deeply buried and multi-source questions.
Table 4 presents the retriever performance using
Hits@50 and MRR metrics, along with their aver-
aged values for multi-source questions. As shown,
retriever performance is consistently low across all
settings, which explains the poor performance of
the RAG-based baselines. We attribute this low per-
formance to two primary factors: first, embedding-
based representations struggle to capture deeply
buried concepts within documents; second, our
question generation method produces phrasing that
differs from the original text, making it more chal-
lenging for the retriever to retrieve the correct doc-
ument. Additionally, we observe similar perfor-
mance levels in both deeply buried information and
multi-source settings.
5.5 Identified Insights in Non-QA Tasks
To better understand the identified insights and their
impact on the matching task, we first extract theTask Type ANN OC
Deep-Insight 92.1 96.5
Multi-Source 72.1 74.8
Table 3: The Insight Miner performance: We report
exact match for deeply buried questions and Recall@10
for multiple source questions.
DataDeep-Insight Multi-Source
Hits@50 MRR A-Hits@50 A-MRR
AAN 39.3 0.13 46.8 0.16
OC 56.1 0.24 49.5 20.3
Table 4: The retriever performance: We report
Hits@50 and MRR for deeply buried questions and
the averaged Hits@50 and MRR for multiple source
questions.
insights generated by the Insight Identifier module
for each model and dataset. We then assign a binary
label (0 or 1) to each sample, indicating whether
augmenting the sample with these insights changes
the model’s prediction from correct to incorrect or
vice versa, respectively. Next, we identify words
with positive or negative impact by calculating the
Z-score—a metric introduced to detect artifacts in
textual datasets by measuring the correlation be-
tween the occurrence of each word and the corre-
sponding sample label (Gardner et al., 2021).
The Z-score results for the LLMs are shown
in Figure 6. Despite the fact that in the prompt
we clearly asked the models to identify insights
independent of the input identifiers (i.e., Paper A
and Paper B), we observe that "paper" appears as
an influential token in insights identified by GPT-
4o mini and o3 mini, mostly as a negative factor
except for o3 mini in the OC dataset.
Overall, OpenAI models appear to benefit from
relation words that indicate direct application or de-
scription (e.g., “used”, “based”, and “describes”),
while they are hindered by more discursive or pre-
dictive terms (e.g., “presents”, “discuss”, “relates”,
and “predict”). In contrast, open LLMs perform
better when relations emphasize analytical or con-
nective processes (e.g., “analyzed”, “connected”,
“enhance”, and “involve”), with generic or usage-
based terms impairing their performance (e.g., “in-
clude”, “based”, “used”, and “applied”). This in-
dicates that the same relation word can affect dif-
ferent models in opposite ways, highlighting the
significant role of model architecture and training
history in interpreting relational cues. Finally, we
observe that for GPT-4o, most identified insights

translationmachineautomaticbased
statisticalschemenovel
distinctionpresentspapers02Z_ScoreANN
used
algorithmsbasedphysical
techniquesaffectsassessesgene
discuss papers1
01Z_ScoreOC(a) GPT-4o mini
programlevel
monolingualdescribescorporamachinelinear label
statisticalmodels1
01Z_ScoreANN
used
addressesaffinity
algorithmsappliedscenariosmallsolvers spreadsuitable1.0
0.5
0.0Z_ScoreOC (b) GPT-4o
machinetranslationword
decisionmaximum expressionreferringrelateslevelpaper1
012Z_ScoreANN
paper gene
expressioncognitiveenough predict feature
extraction moleculesmortality1
01Z_ScoreOC (c) o3 mini
speechpart
tagging entropy transfer patternsdocumentincludesense
disambiguation02Z_ScoreANN
dynamics analyzedconnected regressionpsvmdisease natural cancer basedinclude1
01Z_ScoreOC
(d) Llama-3.3 70B
taggingenhance
informationmethodbayesianappliedsystemsused
techniquestext2
02Z_ScoreANN
impact appliedtools
requiresinvolve
structuralstreamssolutions improveimage2
0Z_ScoreOC (e) DeepSeek-R1
Figure 6: The quality of identified insights in the matching task: We identified the top-5 most positively and
negatively influential words in the identified insights using Z-score metrics for each LLM.
did not result in changes to model predictions, sug-
gesting that the Z-scores for this model may not be
very trustworthy.
6 Related Works
RAG has emerged as a prominent strategy for en-
hancing LLMs by grounding their responses in
external document repositories. Early works in this
area focused on integrating retrieval mechanisms
with LLMs to improve accuracy and contextual
relevance in tasks such as open-domain question
answering and summarization (Lewis et al., 2020;
Karpukhin et al., 2020; Guu et al., 2020). However,
conventional RAG approaches typically rely on
surface-level matching techniques, which can miss
deeper, context-specific information and fail to cap-
ture nuanced insights embedded within texts. More
complex versions of RAG, such as Iter-RetGen
(Shao et al., 2023) and self-RAG (Asai et al., 2023),
have also been proposed to handle decomposable
and multi-step reasoning tasks (Zhao et al., 2024b).
However, these methods were not applicable to our
experiments since we focus on atomic questions
that are not decomposable. Notably, these sophisti-
cated approaches could potentially be integrated on
top of Insight-RAG to further enhance performance
in scenarios requiring iterative refinement.
Parallel to these developments, research on in-sight extraction has demonstrated the value of iden-
tifying critical, often overlooked details within
documents. For example, transformer-based ap-
proaches such as OpenIE6 (Kolluru et al., 2020)
have advanced Open Information Extraction by
leveraging pretraining to capture nuanced relational
data from unstructured text. LLMs have emerged
as powerful tools for keyphrase extraction (Muham-
mad et al., 2024), and in recent years, they have
been increasingly adopted to mine insights from
documents across various domains (Ma et al., 2023;
Zhang et al., 2023; Schilling-Wilhelmi et al., 2024).
7 Conclusion and Future Work
We introduced Insight-RAG, a novel framework
that enhances traditional RAG by incorporating
an intermediary insight extraction process. Our
approach specifically addresses key challenges
in conventional RAG pipelines—namely, captur-
ing deeply buried information, aggregating multi-
source insights, and extending beyond standard
question answering tasks. By evaluating Insight-
RAG on our developed targeted benchmarks across
two scientific paper datasets (AAN and OC), we
have demonstrated that leveraging insight-driven
retrieval consistently improves performance. More-
over, through detailed component analysis, we iden-
tified both the reasoning behind Insight-RAG’s su-

perior performance and the factors contributing to
the traditional RAG pipeline’s poor performance.
Looking forward, Insight-RAG offers promising
directions for future research in various ways: (1)
Moving beyond the citation recommendation and
matching tasks explored in this study, the frame-
work can be extended to various domains, includ-
ing legal analysis, medical research, business intel-
ligence, and creative content generation. (2) Future
work could develop hierarchical insight extraction
methods that categorize insights by importance,
abstraction level, and relevance, enabling more nu-
anced retrieval strategies. (3) Extending the frame-
work to handle multimodal data would allow for
insight extraction from images, audio, and video
alongside text, creating a more comprehensive un-
derstanding of complex information ecosystems.
(4) Incorporating expert feedback loops would al-
low domain specialists to guide the insight ex-
traction process, particularly in highly specialized
fields where nuanced understanding is critical. (5)
Investigating the transferability of insights across
domains could reduce the need for domain-specific
training while maintaining high performance.
8 Limitations
While Insight-RAG offers significant improve-
ments over conventional RAG methods, several
limitations must be acknowledged. First, to capture
new knowledge and remain current with evolving
information, the Insight Miner requires periodic
re-training—a process that conventional RAG sys-
tems can avoid by directly retrieving documents
from an up-to-date corpus. This re-training require-
ment increases both maintenance complexity and
computational overhead. More details are provided
in the Appendix.
Additionally, the multi-stage design of Insight-
RAG introduces increased computational complex-
ity and potential latency, which may hinder its ap-
plicability in real-time or resource-constrained en-
vironments. The framework’s reliance on carefully
crafted prompts for the Insight Identifier also rep-
resents a limitation; minor deviations in prompt
design can lead to inconsistencies in the extraction
of critical insights, affecting downstream perfor-
mance.
Error propagation across the pipeline is another
concern. Inaccuracies in insight identification may
lead to misdirected retrieval efforts, ultimately im-
pacting the overall quality of the generated re-sponse. Finally, our evaluation has been primar-
ily conducted on scientific paper datasets, which
raises questions about the generalizability of the
approach to other domains or more unstructured
data sources. Future work should explore broader
applications and optimize the framework to address
these challenges.
References
Garima Agrawal, Tharindu Kumarage, Zeyad Alghamdi,
and Huan Liu. 2024. Mindful-rag: A study of points
of failure in retrieval augmented generation. In 2024
2nd International Conference on Foundation and
Large Language Models (FLLM) , pages 607–611.
IEEE.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
arXiv preprint arXiv:2310.11511 .
Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu,
Zach Brannelly, and Mohamed Abdelrazek. 2024.
Seven failure points when engineering a retrieval
augmented generation system. In Proceedings of
the IEEE/ACM 3rd International Conference on AI
Engineering-Software Engineering for AI , pages 194–
199.
Chandra Bhagavatula, Sergey Feldman, Russell Power,
and Waleed Ammar. 2018. Content-based citation
recommendation. arXiv preprint arXiv:1802.08301 .
Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz,
Mansheej Paul, Philip Greengard, Connor Jennings,
Daniel King, Sam Havens, Vitaliy Chiley, Jonathan
Frankle, and 1 others. 2024. Lora learns less and
forgets less. arXiv preprint arXiv:2405.09673 .
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. In Pro-
ceedings of the 30th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining , pages 6491–
6501.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen
Wang. 2023. Retrieval-augmented generation for
large language models: A survey. arXiv preprint
arXiv:2312.10997 .
Matt Gardner, William Merrill, Jesse Dodge, Matthew E
Peters, Alexis Ross, Sameer Singh, and Noah Smith.
2021. Competency problems: On finding and re-
moving artifacts in language data. arXiv preprint
arXiv:2104.08646 .
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,

Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models. arXiv preprint arXiv:2407.21783 .
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao
Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shi-
rong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025.
Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning. arXiv preprint
arXiv:2501.12948 .
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. In International confer-
ence on machine learning , pages 3929–3938. PMLR.
Steven CH Hoi, Doyen Sahoo, Jing Lu, and Peilin Zhao.
2021. Online learning: A comprehensive survey.
Neurocomputing , 459:249–289.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
and Weizhu Chen. 2021. Lora: Low-rank adap-
tation of large language models. arXiv preprint
arXiv:2106.09685 .
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam
Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow,
Akila Welihinda, Alan Hayes, Alec Radford, and 1
others. 2024. Gpt-4o system card. arXiv preprint
arXiv:2410.21276 .
Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. 2020. Dense passage re-
trieval for open-domain question answering. In
EMNLP (1) , pages 6769–6781.
Zixuan Ke, Yijia Shao, Haowei Lin, Tatsuya Kon-
ishi, Gyuhak Kim, and Bing Liu. 2023. Contin-
ual pre-training of language models. arXiv preprint
arXiv:2302.03241 .
Keshav Kolluru, Vaibhav Adlakha, Samarth Aggar-
wal, Soumen Chakrabarti, and 1 others. 2020. Ope-
nie6: Iterative grid labeling and coordination analy-
sis for open information extraction. arXiv preprint
arXiv:2010.03147 .
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented
generation for knowledge-intensive nlp tasks. Ad-
vances in Neural Information Processing Systems ,
33:9459–9474.
Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long,
Pengjun Xie, and Meishan Zhang. 2023. Towards
general text embeddings with multi-stage contrastive
learning. arXiv preprint arXiv:2308.03281 .
Juhao Liang, Ziwei Wang, Zhuoheng Ma, Jianquan Li,
Zhiyi Zhang, Xiangbo Wu, and Benyou Wang. 2024.
Online training of large language models: Learn
while chatting. arXiv preprint arXiv:2403.04790 .Jerry Liu. 2022. LlamaIndex.
Pingchuan Ma, Rui Ding, Shuai Wang, Shi Han, and
Dongmei Zhang. 2023. Demonstration of insightpi-
lot: An llm-empowered automated data exploration
system. arXiv preprint arXiv:2304.00477 .
Ali Modarressi, Hanieh Deilamsalehy, Franck Dernon-
court, Trung Bui, Ryan A Rossi, Seunghyun Yoon,
and Hinrich Schütze. 2025. Nolima: Long-context
evaluation beyond literal matching. arXiv preprint
arXiv:2502.05167 .
Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and
Nils Reimers. 2022. Mteb: Massive text embedding
benchmark. arXiv preprint arXiv:2210.07316 .
Umair Muhammad, Tangina Sultana, and Young Koo
Lee. 2024. Pre-trained language models for
keyphrase prediction: A review. ICT Express .
OpenAI. 2025. Openai o3-mini system card.
Andrea Papaluca, Daniel Krefl, Sergio Mendez Ro-
driguez, Artem Lensky, and Hanna Suominen. 2023.
Zero-and few-shots knowledge graph triplet extrac-
tion with large language models. arXiv preprint
arXiv:2312.01954 .
Pouya Pezeshkpour and Estevam Hruschka. 2025.
Learning beyond the surface: How far can continual
pre-training with lora enhance llms’ domain-specific
insight learning? arXiv preprint arXiv:2501.17840 .
Dragomir R Radev, Pradeep Muthukrishnan, Vahed
Qazvinian, and Amjad Abu-Jbara. 2013. The acl
anthology network corpus. Language Resources and
Evaluation , 47:919–944.
Mara Schilling-Wilhelmi, Martiño Ríos-García, Sher-
jeel Shabih, María Victoria Gil, Santiago Miret,
Christoph T Koch, José A Márquez, and Kevin Maik
Jablonka. 2024. From text to insight: large language
models for materials science data extraction. arXiv
preprint arXiv:2407.16867 .
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie
Huang, Nan Duan, and Weizhu Chen. 2023. Enhanc-
ing retrieval-augmented large language models with
iterative retrieval-generation synergy. arXiv preprint
arXiv:2305.15294 .
Somin Wadhwa, Silvio Amir, and Byron C Wallace.
2023. Revisiting relation extraction in the era of large
language models. In Proceedings of the conference.
Association for Computational Linguistics. Meeting ,
volume 2023, page 15566. NIH Public Access.
Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen,
and Sercan Ö Arık. 2024. Astute rag: Overcom-
ing imperfect retrieval augmentation and knowledge
conflicts for large language models. arXiv preprint
arXiv:2410.07176 .

Yunkai Zhang, Yawen Zhang, Ming Zheng, Kezhen
Chen, Chongyang Gao, Ruian Ge, Siyuan Teng,
Amine Jelloul, Jinmeng Rao, Xiaoyuan Guo, and
1 others. 2023. Insight miner: A large-scale multi-
modal model for insight mining from time series. In
NeurIPS 2023 AI for Science Workshop .
Justin Zhao, Timothy Wang, Wael Abid, Geoffrey An-
gus, Arnav Garg, Jeffery Kinnison, Alex Sherstin-
sky, Piero Molino, Travis Addair, and Devvret Rishi.
2024a. Lora land: 310 fine-tuned llms that rival gpt-4,
a technical report. arXiv preprint arXiv:2405.00732 .
Siyun Zhao, Yuqing Yang, Zilong Wang, Zhiyuan He,
Luna K Qiu, and Lili Qiu. 2024b. Retrieval aug-
mented generation (rag) and beyond: A comprehen-
sive survey on how to make your llms use external
data more wisely. arXiv preprint arXiv:2409.14924 .
Xuhui Zhou, Nikolaos Pappas, and Noah A Smith. 2020.
Multilevel text alignment with cross-document atten-
tion. arXiv preprint arXiv:2010.01263 .
A Prompts
The prompts used for the Insight Identifier, ques-
tion answering with and without augmentation,
matching with and without augmentation, and
evaluating the identified insights are provided in
prompts A.1, A.2, A.3, A.4, A.5, and A.6, respec-
tively.
Insight Identifier
You are given a question or task along with
its required input. Your goal is to extract
the necessary insight that will allow another
autoregressive LLM—pretrained on a dataset
of scientific papers—to complete the answer.
The insight must be expressed as a sentence
fragment (i.e., a sentence that is meant to be
completed).
Instructions:
Extract the Insight:
Identify the key information needed from
the dataset to solve the task or answer the
question.
Format the insight as a sentence fragment that
can be completed by the LLM trained on the
dataset.
For example, if the task is to find the
birthplace of Person X, your insight should be:
"Person X was born in".
Determine Answer Multiplicity:
Determine whether the answer should be singular
or plural based solely on the plurality of
the nouns in the question. Do not use common
sense or external context—rely exclusively on
grammatical cues in the question.
For instance, if the question uses plural nouns
(e.g., "What are the cities in California?"),
set Multi-answer to True. Conversely, if the
question uses singular nouns (e.g., "What doespizza contain?"), set it to False.
Relevance Check:
Only include insights that are directly
answerable from the dataset.
If an insight does not relate to the available
dataset, ignore it.
Output Format:
Return the result as a list of dictionaries.
Each dictionary must have two keys:
"Insight": The sentence fragment containing
the key insight.
"Multi-answer": A Boolean (True or False)
indicating whether multiple answers are
required.
Example Output for follwing questions, Where
was Person X born in? what does pizza contain?
What are the Cities in California?:
[
{"Insight": "Person X was born in",
"Multi-answer": false},
{"Insight": "Pizza contains", "Multi-answer":
false},
{"Insight": "The cities in California are",
"Multi-answer": true}
]
Please provide your final answer in this
JSON-like list-of-dictionaries format with no
additional commentary.
Also, make sure to NOT add any extra word to
the insights other than the word present in
the input.
Remove all unnecessary words and provide the
insight in its simplest form. For example,
if the query asks "what are the components
that X uses?", the insight should be "X uses".
Similarly, if the query asks "what are all the
components/techniques/features/applications
included in Z?", the insight should be "Z
include".
If a non-question task is given, possible
insights might involve asking about how two
concepts are connected or a definition of a
concept. Only identify the insight you believe
will help solve the task, and provide it as a
short sentence fragment to be completed. Do
not add any unnecessary content or summaries
of the input.
Additionally, for non-question tasks, the
insight should NOT refer to the specific input
or include any input-specific identifiers.
Instead, it should be a STAND-ALONE statement
focusing on the underlying concepts, entities,
and their relationships from the inputs. If
you cannot find any such insights, return a
list of EMPTY dictionary.
Task:
{}
QA
Answer the question. Do not include any extra
explanation.
Question: {}

Augmented QA
Answer the question using the context. Do not
include any extra explanation.
Question: {}
Context: {}
Matching
You are provided with two research papers,
Paper-A and Paper-B. Your task is to determine
if the papers are relevant enough to be cited
by the other. Your response must be provided
in a JSON format with two keys:
"explanation": A detailed explanation of your
reasoning and analysis.
"answer": The final determination ("Yes" or
"No").
Paper-A:
{}
Paper-B:
{}
Augmented Matching
You are provided with two research papers,
Paper-A and Paper-B, and some useful insights.
Your task is to determine if the papers are
relevant enough to be cited by the other. You
may use the insights to better predict whether
the papers are relevant or not. The insights
should only serve as supportive evidence; do
not rely on them blindly.
Your response must be provided in a JSON format
with two keys:
"explanation": A detailed explanation of your
reasoning and analysis.
"answer": The final determination ("Yes" or
"No").
Paper-A:
{}
Paper-B:
{}
Useful insights:
{}
Identified Insights Evaluation
You are given two incomplete sentences: a
target sentence and a generated sentence. Your
task is to evaluate how similar these two
incomplete sentences are in terms of meaning
and content. Please follow these instructions:
Similarity Criteria:
0: The sentences are not similar at all.
0.5: The sentences share some elements or
meaning, but are only partially similar.
1: The sentences are very similar or essentially
equivalent in meaning.Output Requirement:
Provide only the similarity score (0, 0.5, or
1) as your output.
Do not include any additional text or
explanation. The output format should be as
follownig:
Score: <0, 0.5, or 1)>
Target Sentence: {}
Generated Sentence: {}
B Experimental Details
Benchmarking: We use the processed abstracts
from the AAN dataset (Radev et al., 2013) and the
OC dataset (Bhagavatula et al., 2018), as provided
by Zhou et al. (2020). This curated set includes
approximately 13,000 paper abstracts from AAN
and 567,000 abstracts from OC, offering a rich and
diverse corpus of academic content. Specifically,
the AAN dataset comprises computational linguis-
tics papers published in the ACL Anthology from
2001 to 2014, along with their associated metadata,
while the OC dataset encompasses approximately
7.1 million papers covering topics in computer sci-
ence and neuroscience.
Modeling: For Insight Miner, we perform con-
tinual pre-training on LLaMA-3.2 3B with LoRA
and optimize hyperparameters through grid search
based on training loss. Specifically, following
Pezeshkpour and Hruschka (2025), we tuned learn-
ing rate α= [3×10−3,10−3,3×10−4,10−4,3×
10−5,10−5]; the LoRA rank r= [4,8,16]; the
LoRA-alpha ∈ {8,16,32}; and the LoRA-dropout
∈ {0.05,0.1}. We trained the LLaMA model for
30 epochs.
Cost and Complexity Considerations: Contin-
ual pre-training of the Insight Miner using LoRA
on 8 NVIDIA A100 SXM GPUs for 30 epochs per
dataset takes approximately 7 hours. Regarding
prompting costs, although Insight-RAG includes an
additional Insight Identifier component compared
to conventional RAG, its ability to achieve much
higher performance with a much shorter context
length results in lower API costs overall. Addition-
ally, while the Insight Miner unit requires periodic
retraining to incorporate new information, in many
settings this update can be performed infrequently.
For environments where new information arrives
regularly, an online learning-based solution (Hoi
et al., 2021; Liang et al., 2024) can be adopted to

0 10 20 30 40 50
# Documents0.00.10.20.30.40.50.6F1
(a) AAN
0 10 20 30 40 50
# Documents0.00.10.20.30.40.5F1
GPT-4o mini + RAG
GPT-4o mini + Insight-RAG (1 generated insight)
GPT-4o + RAG
GPT-4o + Insight-RAG (1 generated insight)
o3-mini + RAG
o3-mini + Insight-RAG (1 generated insight)
Llama-3.3 70B + RAG
Llama-3.3 70B + Insight-RAG (1 generated insight)
DeepSeek-R1 + RAG
DeepSeek-R1 + Insight-RAG (1 generated insight) (b) OC
Figure 7: The performance comparison of RAG versus Insight-RAG across the AAN and OC datasets based on
F1 metric for deeply buried information. As demonstrated, Llama-3.3 performed the best, while DeepSeek-R1
performed the worst.
0 10 20 30 40 50
# Documents/Generated Insights0.000.050.100.150.200.250.300.35F1
(a) AAN
0 10 20 30 40 50
# Documents/Generated Insights0.000.050.100.150.200.25F1
GPT-4o mini + RAG
GPT-4o mini + Insight-RAG
GPT-4o + RAG
GPT-4o + Insight-RAG
o3-mini + RAG
o3-mini + Insight-RAG
Llama-3.3 70B + RAG
Llama-3.3 70B + Insight-RAG
DeepSeek-R1 + RAG
DeepSeek-R1 + Insight-RAG (b) OC
Figure 8: The performance comparison of RAG versus Insight-RAG across the AAN and OC datasets based on
averaged F1 metric for multi-source questions. As demonstrated, Llama-3.3 performed the best, while DeepSeek-R1
performed the worst.
update the model incrementally without necessitat-
ing a full retraining cycle.
C Experimnets
We report F1 and averaged F1 performance for all
models for deeply buried and multi-source ques-
tions in Figure 7 and 8, respectively. Interestingly,
despite DeepSeek’s superior performance in Ex-
act Match metrics, its F1 scores show a significant
decline. Upon closer examination, we discovered
this discrepancy stems primarily from DeepSeek’s
tendency to generate excessive content and occa-
sional hallucinations, particularly when the cor-
rect document isn’t retrieved. This poor F1 perfor-
mances occur despite our removal of DeepSeek’s
“thinking” sections when calculating F1 scores. The
other evaluated models demonstrate performance
patterns similar to their Exact Match results, with
Llama-3.3 70B consistently emerging as the top-
performing model across both setting. Moreover,Table 5 presents the F1 scores for the paper match-
ing task. While these results follow similar trends
as the accuracy metric, the F1 scores reveal that
both the positive and negative impacts of conven-
tional RAG as well as the benefits of Insight-RAG,
are even more amplified compared to accuracy.
Finally, focusing on DeepSeek-R1 due to its su-
perior performance, we report its RAG-based re-
sults when, instead of retrieving documents, we
retrieve triples from the set of all extracted triples
for each dataset. Table 6 provides the exact match
accuracy for the deeply buried information setting,
along with the averaged exact match accuracy for
the multi-source setting. We observe that while the
model shows similar behavior to document-based
RAG, using much less context—since a triple is
much shorter than a document—it still falls sig-
nificantly short compared to Insight-RAG perfor-
mance. The overall gap between triple-based RAG
and Insight-RAG underscores the shortcomings of

ModelANN OC
Vanilla RAG (1 doc) Insight-RAG Vanilla RAG (1 doc) Insight-RAG
GPT-4o mini 78.8 79.9 (+1.1) 82.2 (+3.4) 66.0 57.9 (-8.1) 72.5 (+6.5)
GPT-4o 82.4 77.6 (-4.8) 82.8 (+0.4) 61.2 66.3 (+5.1) 65.6 (+4.4)
o3 mini 85.0 85.1 (+0.1) 85.4 (+0.4) 70.4 65.4 (-5.0) 78.9 (+8.5)
Llama 3.3 70B 83.8 80.0 (-3.8) 84.8 (+1.0) 73.8 71.8 (-2.0) 77.8 (+4.0)
DeepSeek-R1 59.3 66.7 (+7.4) 68.6 (+9.3) 50.4 60.6 (+10.2) 62.2 (+11.8)
Table 5: The F1 performance comparison of RAG versus Insight-RAG across the AAN and OC datasets in the paper
matching task. As demonstrated, o3 mini performs the best while DeepSeek-R1 shows the lowest performance.
Moreover, we observe that Insight-RAG consistently improves performance across all models, while RAG-based
solutions show mixed impacts on model performance.
ModelANN OC
1 triple 3 triples 10 triples 50 triples 1 triple 3 triples 10 triples 50 triples
DeepSeek-R1 (Deep) 13.8 18.9 25.8 35.2 20.1 27.0 33.0 42.2
DeepSeek-R1 (Multi) 12.1 14.0 14.7 25.2 10.6 13.9 17.9 22.7
Table 6: RAG-based exact match and averaged exact match accuracy of DeepSeek-R1 for deeply buried and
multi-source questions. Instead of retrieving documents, we retrieve triples—using the set of extracted triples.
conventional retrieval approaches and the complex-
ity of resolving them.
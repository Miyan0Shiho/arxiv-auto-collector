# Never Come Up Empty: Adaptive HyDE Retrieval for Improving LLM Developer Support

**Authors**: Fangjian Lei, Mariam El Mezouar, Shayan Noei, Ying Zou

**Published**: 2025-07-22 16:46:00

**PDF URL**: [http://arxiv.org/pdf/2507.16754v1](http://arxiv.org/pdf/2507.16754v1)

## Abstract
Large Language Models (LLMs) have shown promise in assisting developers with
code-related questions; however, LLMs carry the risk of generating unreliable
answers. To address this, Retrieval-Augmented Generation (RAG) has been
proposed to reduce the unreliability (i.e., hallucinations) of LLMs. However,
designing effective pipelines remains challenging due to numerous design
choices. In this paper, we construct a retrieval corpus of over 3 million Java
and Python related Stack Overflow posts with accepted answers, and explore
various RAG pipeline designs to answer developer questions, evaluating their
effectiveness in generating accurate and reliable responses. More specifically,
we (1) design and evaluate 7 different RAG pipelines and 63 pipeline variants
to answer questions that have historically similar matches, and (2) address new
questions without any close prior matches by automatically lowering the
similarity threshold during retrieval, thereby increasing the chance of finding
partially relevant context and improving coverage for unseen cases. We find
that implementing a RAG pipeline combining hypothetical-documentation-embedding
(HyDE) with the full-answer context performs best in retrieving and answering
similarcontent for Stack Overflow questions. Finally, we apply our optimal RAG
pipeline to 4 open-source LLMs and compare the results to their zero-shot
performance. Our findings show that RAG with our optimal RAG pipeline
consistently outperforms zero-shot baselines across models, achieving higher
scores for helpfulness, correctness, and detail with LLM-as-a-judge. These
findings demonstrate that our optimal RAG pipelines robustly enhance answer
quality for a wide range of developer queries including both previously seen
and novel questions across different LLMs

## Full Text


<!-- PDF content starts -->

Never Come Up Empty: Adaptive HyDE Retrieval for Improving
LLM Developer Support
Fangjian Lei∗
fangjian.lei@queensu.ca
Queen’s University
Kingston, Ontario, CanadaMariam El Mezouar
mariam.el-mezouar@rmc.ca
Royal Military College of Canada
Kingston, Ontario, Canada
Shayan Noei
s.noei@queensu.ca
Queen’s University
Kingston, Ontario, CanadaYing Zou
ying.zou@queensu.ca
Queen’s University
Kingston, Ontario, Canada
Abstract
Large Language Models (LLMs) have shown promise in assisting
developers with code-related questions; however, LLMs carry the
risk of generating unreliable answers. To address this, Retrieval-
Augmented Generation (RAG) has been proposed to reduce the unre-
liability (i.e., hallucinations) of LLMs. However, designing effective
pipelines remains challenging due to numerous design choices. In
this paper, we construct a retrieval corpus of over 3 million Java and
Python related Stack Overflow posts with accepted answers, and ex-
plore various RAG pipeline designs to answer developer questions,
evaluating their effectiveness in generating accurate and reliable
responses. More specifically, we (1) design and evaluate 7 different
RAG pipelines and 63 pipeline variants to answer questions that
have historically similar matches, and (2) address new questions
without any close prior matches by automatically lowering the
similarity threshold during retrieval, thereby increasing the chance
of finding partially relevant context and improving coverage for
unseen cases. We find that implementing a RAG pipeline combin-
ing hypothetical-documentation-embedding (HyDE) with the full-
answer context performs best in retrieving and answering similar
content for Stack Overflow questions. Finally, we apply our optimal
RAG pipeline to 4 open-source LLMs and compare the results to
their zero-shot performance. Our findings show that RAG with our
optimal RAG pipeline consistently outperforms zero-shot baselines
across models, achieving higher scores for helpfulness, correctness,
and detail with LLM-as-a-judge. These findings demonstrate that
our optimal RAG pipelines robustly enhance answer quality for a
wide range of developer queries—including both previously seen
and novel questions—across different LLMs.
Keywords
Stack overflow, RAG, Large Language Models
1 Introduction
Programmers often rely on online resources for a wide range of
development tasks, such as API usage, bug fixing, and understand-
ing of code or programming concepts [ 29,32,38]. A significant
portion of these help-seeking activities involves regular interac-
tion with community-driven Q&A platforms like Stack Overflow
(SO) [ 28,35,38]. Recently, the emergence of Large Language Models
(LLMs) has begun to reshape how developers search for assistancein programming activities that developers increasingly prefer using
conversational LLMs over traditional search methods like forums or
search engines for programming assistance[ 31]. Open-source LLMs
such as LLaMA family [ 34], have shown strong performance in
code understanding and generation tasks, gaining increasing atten-
tion among software practitioners and researchers. These models
offer the potential to serve as an alternative to traditional search on
Q&A platforms, enabling more conversational and context-aware
support during programming tasks.
Despite the rising popularity of Large Language Models (LLMs)
used for information seeking, there are growing concerns about the
reliability and correctness of generated content commonly referred
to as hallucination. Previous studies have shown that LLMs can
learn incorrect information during training and later reproduce
or even amplify these errors in the generated outputs [ 5,10,13].
LLMs are also capable of producing fabricated content that mimics
truthful responses, which can be difficult to detect, especially for
users without domain expertise [6, 7].
To mitigate hallucination, Retrieval-Augmented Generation (RAG)
has emerged as a promising solution and has shown strong poten-
tial in improving the quality of responses generated by LLMs when
a knowledge base with similar context is available for reference.
RAG enhances LLMs by incorporating external knowledge retrieved
from a document corpus into the generation process [ 19]. However,
the effectiveness of RAG is highly dependent on the retriever’s
ability to identify relevant information. When the input question
is novel or falls outside the scope of the retrieval corpus, existing
RAG retrievers often struggle to extract useful content [ 8,14]. Since
existing RAG systems rely solely on the input question, they may
fail to retrieve semantically relevant documents in such cases. The
final answer depends largely on the LLM’s pre-trained knowledge
and its ability to generalize. This limitation highlights the need
for methods that generate informative answers even when rele-
vant content cannot be retrieved, ensuring consistent performance
across diverse questions.
In this paper, we aim to address the limitations of existing RAG
approaches, which struggle with vague questions and often fail on
novel questions. More specifically, we explore two implementations
of Retrieval-Augmented Generation (RAG): (1) a question-based ap-
proach that searches the knowledge base using the original question,
and (2) the Hypothetical Document Embedding (HyDE) approach
[11], which first generates a hypothetical answer to improve thearXiv:2507.16754v1  [cs.SE]  22 Jul 2025

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
24 million
postsDataset
Construction
Pipeline Search
(RQ1)Adaptive
Thresholding (RQ2)Cross-Model
Generalization (RQ3)
RAG KB
Synthetic
Question Set
Unseen
Question Set63 RAG V ariants
(7 pipelines x 9 sim.
thresholds)
 Generate on
Synthetic Question
set
LLM-as-a-Judge
Scoring and
Coverage evaluation Generate
on Unseen
Question  setStart @ Strict
Similarity
Threshold
Iteratively Lower
Threshold
LLM-as-a-Judge
Scoring Optimal PipelineOptimal Pipeline
Run using 4 LLMs
with zero shot +
RAG
 Generate on
Synthetic Question
set
LLM-as-a-Judge
Scoring and
Significance T ests
Figure 1: Experimental Workflow. RAG KB = Stack Overflow Knowledge Base (3.4 M accepted-answer documents). Synthetic
Question Set : 385 questions auto-generated from the KB (seen). Unseen Question Set : 5,510 new Stack Overflow questions posted
after the KB snapshot (unseen).
relevance of retrieved content. The two RAG implementations are
further characterized by three key design dimensions: the first di-
mension, retrieval target , determines whether content is retrieved
directly from accepted answers or indirectly via similar questions.
The second, content granularity , specifies whether the system
retrieves full answers for broader context or individual sentences
for more precise and relevant information. The third, similarity
threshold , sets the semantic similarity score between the input
and retrieved content, directly influencing the amount and quality
of context for generation. These three design dimensions directly
affect the amount and quality of context extracted from RAG knowl-
edge base. Therefore, we conduct the experiments by systematically
varying the dimensions to assess how different pipelines affect the
quality of generated answers using LLMs based on RAG and iden-
tify the best-performing RAG pipeline that can extract relevant
content for enhanced answer quality.
In this paper, we aim to answer the following research questions:
RQ1: Which retrieval approach configurations yield the
highest response quality in LLM-generated answers? To deter-
mine how different design dimensions impact the effectiveness of
RAG, we systematically evaluate 7 RAG pipelines and 63 pipeline
variants that vary in retrieval target, content granularity, and simi-
larity threshold. We assess each pipeline in terms of both answer
quality and retrieval coverage on the Synthetic Question Set. Our
results show that the hypothetical-answer-based pipeline (HB1),
which retrieves from full answers in the knowledge base, consis-
tently achieves the best trade-off between high response quality
and broad coverage. This pipeline is selected as the optimal pipeline
for further research questions.
RQ2: How well does adaptive HyDE retrieval perform on
novel questions outside the training corpus? Developers often
pose novel questions that lack closely related content in the knowl-
edge base, which limits the effectiveness of standard RAG methods.
To address this, we extend our approach by dynamically decreasingthe similarity threshold for each question until relevant context
is retrieved. Evaluated on an unseen question set, dynamically de-
creasing the similarity threshold enables full coverage, ensuring
every question receives relevant contextual from RAG. Results show
that our method significantly improves answer quality over original
Stack Overflow answers, with statistical analysis confirming the
effectiveness of dynamic thresholding for unseen cases.
RQ3: How does our proposed RAG pipeline perform across
different LLMs? Given the diversity of available LLMs, it is im-
portant to understand whether our optimal RAG pipeline offers
consistent benefits across models. We apply the pipeline to several
open-source LLMs and compare its performance to standard zero-
shot prompting. Our findings reveal that our optimal RAG pipeline
robustly improves or matches answer quality across different mod-
els, demonstrating strong generalization and practical value for a
variety of LLM-based applications.
Our contributions are as follows:
•We present RAG frameworks for answering Java and Python
questions, using Stack Overflow as a retrieval base and open-
source LLMs for generating answers.
•We evaluate RAG implementations and propose a HyDE approach
to improve answer retrieval performance on both seen and unseen
questions.
•We provide an evaluation of multiple LLMs performance on devel-
oper questions in both matched (similar) and unmatched (unseen)
scenarios.
•We release our dataset and pipeline to support future research
in RAG-based methods for software engineering questions. Our
replication package is available at [4].
The remainder of this paper is organized as follows. Section
2 details the proposed approach. Section 3 presents the research
questions. Section 4 is the discussion section. Section 5 addresses

Never Come Up Empty: Adaptive HyDE Retrieval for Improving LLM Developer Support Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
the threats to validity. Section 6 presents previous related work.
Finally, Section 7 concludes the paper and outlines future work.
2 Methodology
Figure 1 outlines our experimental workflow. We first construct a
RAG Knowledge Base of 3.4 million Java and Python Stack Over-
flow posts with accepted answers. Then we create two evaluation
sets: (1) Synthetic Question Set of 385 generated questions to
represent "seen" cases, and (2) Unseen Question Set of 5,510 posts
made after May 2023 to simulate novel queries. Our evaluation pro-
ceeds in three stages: (RQ1) identifying the optimal RAG pipeline
by varying design choices and assessing both answer quality using
the LLM-as-a-Judge framework and retrieval coverage on the Syn-
thetic Question Set ; (RQ2) introducing adaptive thresholding to
improve retrieval for novel questions and evaluating its impact on
theUnseen Question Set ; and (RQ3) testing the generalizability
of our optimal pipeline across multiple open-source LLMs using
Synthetic Question Set .
2.1 Dataset Construction
To support domain-specific retrieval, we build a RAG Knowledge
Base from the official Stack Overflow data dump,1covering posts
from January 2008 to December 2024. We focus on posts tagged
with[java] or[python] because of the sustained relevance in both
educational and industrial contexts [ 9]. The dataset construction
process involves the following steps:
Filtering: We retain only non-duplicate questions with an accepted
answer to ensure reliable question–answer pairs for retrieval and
evaluation. After filtering the raw Stack Overflow data, we obtain
a total of 3,428,217 posts tagged with either [java] or[python]
that include an accepted answer, which we use to construct our
RAG knowledge base.
Cleaning: We remove HTML tags and markdown syntax to pro-
duce clean plain-text content suitable for indexing and retrieval. In
addition, we split the accepted answers into individual sentences
to support sentence-level retrieval. We also retain the complete ac-
cepted answers to support both sentence- and answer-level content
granularity.
Splitting: We reserve posts from May 2023 to December 2024 as
the evaluation set to simulate future or unseen queries, while using
earlier posts for retrieval to mitigate data leakage. To evaluate
the performance of the RAG pipelines, we construct two distinct
question sets: Synthetic Question Set andUnseen Question Set .
To evaluate the performance of various RAG pipelines, we sample
and synthesize 385 questions from the knowledge base to construct
Synthetic Question Set . Each question is created by sampling a
question from the Stack Overflow dataset, then prompting GPT-4o2
to generate a similar question in order to reduce the data leakage
issue. The selected sample size ensures a 95% confidence level with
a 5% margin of error [ 17]. This synthetic set is used as the main
evaluation for assessing answer quality in RQ1 and RQ3. To test
our approach on new and previously unseen questions (RQ2), we
construct an Unseen Question Set containing 5,510 questions
posted between May 2023 and December 2024. All questions in
1https://archive.org/details/stackexchange
2https://platform.openai.com/docs/modelsthis set are posted after the release of LLaMA-3.1-8B-Instruct [2],
the model used for response generation. We begin by extracting
all Stack Overflow posts tagged with either [java] or[python]
from this time period and remove any posts whose question titles
also appear in the RAG Knowledge Base. This yields a total of
11,771 questions—2,755 related to Java and 9,016 related to Python.
To balance the evaluation set, we randomly sample 2,755 Python
questions and combine them with all 2,755 Java questions, resulting
in a final set of 5,510 testing cases. None of these posts appear
in the RAG Knowledge Base, ensuring a true unseen scenario for
evaluation.
2.2 LLM-as-a-Judge Scoring Framework
We adopt the LLM-as-a-Judge mechanism to score every generated
answer based on helpfulness (whether the response addresses
the user query), technical correctness (technical accuracy), and
level of detail (completeness and depth). We feed the LLM-as-a-
Judge prompts to GPT-4o, to produce a single composite score per
answer on a 1–10 scale (based on the three criteria). We show the
specific prompts used throughout the rest of the paper. To verify
that automatic scores reasonably align with human judgment, two
independent annotators manually rate a sample of 54answer pairs
(chosen for a 90% confidence level and a 10% margin of error). Both
annotators are graduate students in computer science and each
has over 10 years of experience with Python, Java, and software
development. For comparison purposes, we map the LLM scores to
binary labels ( 2–5→0,6–9→1)3. Cohen’s 𝜅indicates moderate
agreement between the human annotators and LLM labels, which
we consider acceptable for this evaluation.
2.3 Identifying Optimal RAG Pipelines
RAG is used to enhance LLM responses by incorporating relevant
content retrieved from a Stack Overflow Knowledge Base consisting
ofJava andPython posts with accepted answers. We introduce two
main RAG implementations as shown in Figure 2. Based on the two
main RAG implementations, we outline multiple pipeline variants
with different component combinations.
2.3.1 Two Base Implementations. .
Question-Based RAG: This implementation directly uses the orig-
inal input question to initiate retrieval. The question is first embed-
ded using a sentence embedding model and then matched against a
pre-processed Stack Overflow corpus to retrieve the most relevant
content. The retrieved content is subsequently provided as context
to a LLM, which in turn generates the final response.
HyDE-Based RAG: This implementation addresses the possible
limitations of directly using the original question for retrieval. For
example, although embedding models capture semantic similarity,
short or vague questions are difficult to retrieve the most relevant
answers, whereas a pseudo-answer provides a richer and more
aligned representation for matching. To mitigate this, HyDE first
generates a hypothetical answer using an LLM. This pseudo-answer
tends to be more informative and semantically aligned with the
expected answer format, making it a more effective query [ 11].
The pseudo-answer is then embedded and used to retrieve relevant
3No answers in the sample received a raw score of 0, 1, or 10.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Input
QuestionEmbeddingGenerate
hypothetical
answer
Applicable to HyDE-
Based RAG Pipeline 
onlyRetrieval T arget: where to search?
Indirect
Retrieval
Direct RetrievalFind similar questions
in RAG KB then fetch
their answers
Search answers
directly in RAG KBRAG Design Dimensions
Content Granularity: how much to
pull from an answer?
Full Answer
Sentence LevelUse entire
accepted answer
Select individual
sentencesThreshold: how similar
to the input Question?
Iterate 
0.1  →  0.9 Retrieve and
Generate 
HyDE-Based RAG Pipeline 
uses only Direct RetrievalGPT-4o
Response
Figure 2: An Overview of Question-Based and HyDE-Based RAG pipeline. HyDE specific design elements shown in green.
Table 1: Summary of RAG pipeline variants. All pipelines share the same similarity-threshold range (0.1–0.9).
ID Pipeline Retrieval TargetContent
GranularitySimilarity
ThresholdNotes
QB1 Question-Based 1 Direct Sentence
0.1–0.9Basic sentence-level retrieval
QB2 Question-Based 2 Direct Full answer Basic full-answer retrieval
QB3 Question-Based 3 Indirect Sentence Indirect retrieval via similar questions
QB4 Question-Based 4 Indirect Full answer Indirect full-answer retrieval via similar questions
HB1 HyDE-Based 1 Direct Full answer Basic HyDE
HB2 HyDE-Based 2 Direct Sentence Basic HyDE, sentence level
HYB QB + HyDE-Based Indirect Full answer Full hybrid pipeline
content from the Stack Overflow corpus, which is passed to another
LLM to generate the final response.
2.3.2 Design Dimensions. To improve RAG performance, we de-
sign multiple pipeline variants by systematically varying key design
dimensions. As described below, we explore changes in content
granularity, retrieval target. In Figure 2, the dot-dashed boxes indi-
cate the pipeline elements that can be adjusted or selected for each
design dimension.
Content Granularity: Retrieved content can be either full accepted
answers orindividual sentences . Full-answer retrieval preserves co-
herence and broader context, but often includes irrelevant or redun-
dant information [ 26]. In contrast, sentence-level retrieval enables
more precise selection of relevant content by filtering out unrelated
or redundant text. Since not every sentence in an accepted answer
contributes directly to answering the question, selecting at the sen-
tence level allows us to incorporate multiple useful sentences from
different answers. This increases coverage by capturing relevant
information even from answers that do not directly or fully address
the question.
Retrieval Target: This design dimension determines the initial
source of retrieval within the Stack Overflow corpus. We explore
two primary strategies: (1) indirect retrieval, which identifies ques-
tions that are similar to the input question, and then extracts their
associated accepted answers or answer sentences; and (2) direct
retrieval, by querying the answer corpus directly—either at the sen-
tence or full-answer level. The first strategy benefits from the fact
that semantically similar questions often have relevant answers,
allowing it to retrieve useful content even if the input question is
worded differently from questions in the Stack Overflow corpus.Similarity Threshold: A key configuration in both implemen-
tations is the choice of similarity threshold, which controls the
trade-off between retrieving highly relevant but fewer results (us-
ing a high threshold) and achieving broader coverage at the cost
of including less relevant content (using a lower threshold). In
Question-Based RAG , the input question is first embedded into
a vector representation. To support embedding computation, we
employ the all-mpnet-base-v2 [30] model to embed input questions,
accepted answers, and hypothetical content. all-mpnet-base-v2 is
chosen for its strong performance on semantic similarity and re-
trieval tasks across diverse domains [ 16]. The system then computes
the cosine similarity [ 15] between the input question embedding
and the embeddings of contents in the RAG Knowledge Base ,
such as question titles or answer sentences depending on the differ-
ent pipeline dimensions design. Only candidates with a similarity
score above the specified threshold are selected as relevant context
for the next stage of generation. In HyDE-Based RAG , instead
of using the original question, we first generate a hypothetical an-
swer based on the question by using the GPT-4o. The hypothetical
answer is then embedded, and its vector is compared to the embed-
dings of potential content (either full answers or individual answer
sentences depending on the different pipeline dimensions design)
in the RAG Knowledge Base . And, only those candidates whose
similarity score exceeds the similarity threshold are extracted as
relevant context.
2.3.3 Pipeline Combinations. Each pipeline is defined by a specific
combination of Retrieval Target andContent Granularity . By
varying the similarity threshold from 0.1 to 0.9, we produce nine
distinct variants for each pipeline. In total, Table 1 provides an
overview of all 7 RAG pipelines and their 63 (7x9) unique pipeline
variants, along with the design dimension descriptions.

Never Come Up Empty: Adaptive HyDE Retrieval for Improving LLM Developer Support Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 2: LLM-as-a-Judge mean scores by pipeline and simi-
larity threshold. Bold marks the best score in each row. An
en-dash (–) indicates that the pipeline retrieved no context
at the specified threshold.
Thres-
holdQuestion-Based (QB)HyDE-Based
(HB)Hybrid
QB1 QB2 QB3 QB4 HB1 HB2
0.1 5.63 5.59 5.33 5.40 6.00 5.43 5.40
0.2 5.55 4.85 5.26 5.19 5.92 5.41 5.21
0.3 5.59 4.00 5.37 2.50 5.94 5.39 5.41
0.4 5.51 – 5.33 – 5.82 5.28 5.48
0.5 5.58 – 5.36 – 5.95 5.38 5.47
0.6 5.66 – 5.69 – 5.94 5.49 5.43
0.7 5.89 – 5.86 – 6.05 5.61 5.59
0.8 6.01 – 5.63 – 6.39 5.80 5.25
0.9 – – – – 7.00 6.00 –
Mean 5.68 4.81 5.48 4.36 6.11 5.53 5.41
2.4 Introducing the Adaptive Thresholding
Strategy
When developer questions do not have close matches in the knowl-
edge base, retrieval-augmented generation (RAG) pipelines often
fail to provide sufficient context, resulting in lower answer quality
[11]. To improve support for highly novel or out-of-scope questions,
we introduce an adaptive thresholding strategy. In this approach,
if the initial retrieval does not yield any relevant content above
the set similarity threshold, the system automatically relaxes the
threshold in discrete steps (-0.1 at a time) until some content is
retrieved or a minimum threshold is reached. This iterative process
helps maximize the chances of finding relevant content for novel
or previously unseen questions.
We evaluate the adaptive thresholding strategy using the Un-
seen Question Set , which consists of questions that do not appear
in the RAG Knowledge Base and are posted after the release
date of the generation model. This setup is intended to simulate
real-world scenarios where developers encounter entirely new or
previously unseen questions that are not covered by the existing
knowledge base.
2.5 Applying the Approach across Different
LLMs
To assess the generalizability and practical impact of our optimal
RAG pipeline, we apply it to multiple open-source LLMs to deter-
mine whether it consistently improves answer quality over stan-
dard zero-shot prompting. Specifically, we assess the pipeline using
Granite-3.1-8B-Instruct [24],Mistral-7B-Instruct-v0.3 [3],Qwen3-
8B[39], and LLaMA-3.1-8B-Instruct . All selected models have been
extensively evaluated in recent studies and have demonstrated
strong abilities in code generation, software comprehension, and
developer-focused question answering tasks [ 18,23,39]. For each
model, we generate responses under two settings: (1) using our opti-
mal RAG configuration and (2) using a baseline zero-shot prompting
approach without any retrieved context. The Synthetic Question
Setserves as the evaluation benchmark for this comparison./cgsFinal Response Generation
You are a knowledgeable and helpful assistant. The user is asking a question on Stack Overflow.
Use the provided context to craft an accurate, concise, and highly relevant response.
Present your answer in a clear and well-structured paragraph format, avoiding the use of bullet
points or lists.
Input
### Question:
input question
### Context:
extracted RAG contents
Please provide your best answer below:
Figure 3: The prompt is used to generate the final response
to input question by combining the extracted content from
RAG pipeline. Tags in red are placeholders.
We score every answer with the LLM-as-a-Judge model as out-
lined in Section 2.2. In addition, we conduct statistical significance
testing to measure the differences between RAG-augmented and
Stack Overflow accepted answers or zero-shot responses. We run
theall-mpnet-base-v2 embedding, LLaMA-3.1-8B-Instruct ,Granite-
3.1-8B-Instruct andMistral-7B-Instruct-v0.3 , and Qwen3-8B models
locally on an (NVIDIA) A100 GPU with 80 GB of memory.
3 Research Questions
In this section, we provide the motivation, approach, and results of
our research questions.
RQ1: Which retrieval approach configurations yield the high-
est response quality in LLM-generated answers?
Motivation .Although RAG is widely adopted to improve the out-
puts of LLMs, its effectiveness is highly dependent on the design of
its components [ 21,33]. In this RQ, we investigate how different
RAG design decisions affect the quality of the generated responses.
Our goal is to determine the most effective RAG pipeline. In addition,
subsequent experiments (e.g., RQ2) apply dynamic thresholding
that progressively lowers the threshold to ensure every question
receives context, it is important to evaluate how each pipeline per-
forms across the full range of retrieval thresholds. This ensures
that the selected configuration remains robust when coupled with
dynamic thresholding in subsequent experiments.
Approach .To evaluate the effectiveness of different RAG pipeline
configurations, we assess 63 pipeline variants (summarized in Ta-
ble 1) using the Synthetic Question Set, which includes 385 Stack
Overflow questions whose corresponding answers are present in
the RAG knowledge base (referred to as Seen Cases ). Each pipeline
is evaluated along two key dimensions:
(1) Answer quality. We assess the quality of the generated
responses based on three criteria: helpfulness (whether the response
addresses the user query), correctness (technical accuracy), and level
of detail (completeness and depth). Each response is scored from 1
to 10 using an LLM-as-a-Judge prompt (Figure 4). To examine how
similarity threshold affects quality, we conduct a sensitivity analysis
by varying the cosine similarity threshold from 0.1 to 0.9 across all
RAG pipelines. This allows us to identify the most effective pipeline
and optimal threshold setting.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Seen data Judge Prompt
System . We would like your feedback on several approaches to the question in [QUESTION
TITLE] . Rate the helpfulness, accuracy, and level of detail of each response. If a response
contains an incomplete code snippet, deduct from the score. Give each approach a score from
1–10; higher is better.
Output format . One line with the scores, separated by commas.
Input
[QUESTION TITLE]
question title
[THE START OF APPROACH 1 ANSWERS]
response 1
[THE END OF APPROACH 1 ANSWERS]
[THE START OF APPROACH 2 ANSWERS]
response 2
[THE END OF APPROACH 2 ANSWERS]
...
[THE START OF APPROACH n ANSWERS]
response n
[THE END OF APPROACH n ANSWERS]
Figure 4: Prompt used by the LLM-as-Judge to evaluate gen-
erated answers for helpfulness, accuracy, and detail across
all RAG pipelines at a single threshold. Tags in red are place-
holders replaced at evaluation time.
(2) Retrieval Coverage. We analyze how often each pipeline
retrieves relevant content across the different similarity thresholds.
This is computed using the same Synthetic Question Set, which
is limited to the Seen Cases . An effective pipeline should retrieve
relevant content for as many Seen Cases as possible, as higher
coverage reflects better stability and broader applicability. Retrieval
Coverage is calculated as the proportion of questions for which
relevant content is successfully extracted out of the total number
of cases (385). However, coverage is not perfect—some questions
still fail to retrieve relevant content even with the optimal pipeline
and threshold. To address these cases, we explore the dynamic
thresholding strategy in RQ2.
We compute LLM-as-a-Judge scores only for the Seen Cases
where the pipeline successfully retrieves relevant content, exclud-
ing questions without retrieval from scoring in this RQ. Table 2
summarizes the answer quality scores across all pipeline variants,
while Figure 5 shows the percentage of Seen Cases with relevant
content retrieved at each similarity threshold. Both answer quality
and retrieval coverage are considered when selecting the optimal
pipeline.
Results: HB1, which uses direct retrieval over full answers
with HyDE-generated queries, is the most effective pipeline,
consistently outperforming others in both answer quality
and retrieval coverage. As shown in Table 2, HB1 achieves the
highest average LLM-as-a-Judge score across all similarity thresh-
olds (mean = 6.05), with a maximum score of 7 out of 10 at threshold
0.9. It also maintains strong retrieval coverage, retrieving relevant
content for 80% of Seen Cases at threshold 0.7. While HB2, which
applies HyDE-based queries over sentence-level content, achieves
similar coverage, its average quality score is lower (mean = 5.53),
suggesting that answer-level granularity contributes to HB1’s bet-
ter performance. In contrast, QB2, which performs direct retrieval
over full answers using the original question as the query, shows
Figure 5: Percentage of retrieval coverage with retrievable
context across varying similarity thresholds. Each line rep-
resents a different RAG pipeline. As the threshold increases,
retrieval becomes more selective, reducing the number of
questions with matching context.
weaker performance on both metrics, with significantly lower cov-
erage and a mean score of only 4.81. These results confirm that HB1
offers the most robust balance between answer quality and retrieval
coverage, making it the optimal configuration for downstream use
in dynamic thresholding scenarios (RQ2).
Higher thresholds generally improve generation quality
but reduce the percentage of Seen Cases .For example, HB1
achieves the highest score (7.00) at threshold 0.9, but retrieves rele-
vant content for fewer than 1% of Seen Cases, limiting its practical
value. In contrast, HB1 at threshold 0.7 provides the best balance. It
reaches a strong mean score of 6.05 while maintaining high retrieval
coverage ofSeen Cases percentage. This configuration offers the best
trade-off between quality of answer andretrieval coverage among
the pipelines. The results also reveal that even within Seen Cases,
no pipeline achieves perfect coverage, motivating our subsequent
experiment.
RQ2: How well does adaptive HyDE retrieval perform on
novel questions outside the training corpus?
Motivation .In practice, developers often ask questions that do
not closely match any entry in a RAG knowledge base. Since RAG
pipelines depend on retrieving relevant context [ 11], the absence
of such content limits the effectiveness, which leads the model
to rely on training knowledge. In RQ1, even when evaluating on
a synthetic question set sampled from the knowledge base, our
optimal pipeline (HB1) retrieved relevant content for about 80%
of questions at threshold 0.7. This highlights the limitation that
high-threshold retrieval cannot guarantee full coverage. In this
RQ, we explore whether it is possible to improve coverage and
answer quality on a more realistic testing set consisting of Stack
Overflow questions posted after both the knowledge base cutoff
and the release of the generation model.
Approach .To address RQ2, we evaluate our method on the Un-
seen Question Set . Each question in the Unseen Question Set
is paired with its original accepted answer from Stack Overflow,

Never Come Up Empty: Adaptive HyDE Retrieval for Improving LLM Developer Support Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Unseen data Judge Prompt
System . We would like your feedback on several approaches to the question in [QUESTION
TITLE] . Rate the helpfulness, accuracy, and level of detail of each response. If a response
contains an incomplete code snippet, deduct from the score. Give each approach a score from
1–10; higher is better.
Output format . One line with the scores, separated by commas.
Input
[QUESTION TITLE]
question title
[THE START OF APPROACH 1 ANSWERS]
response 1
[THE END OF APPROACH 1 ANSWERS]
[THE START OF APPROACH 2 ANSWERS]
response 2
[THE END OF APPROACH 2 ANSWERS]
Figure 6: Prompt used by the LLM-as-Judge to score generated
answers and the accepted answer on helpfulness, accuracy,
and detail. Tags in red are placeholders replaced at evaluation
time.
allowing us to compare the performance between the generated
responses and the SO accepted answers.
Then, we apply the optimal pipeline identified in RQ1 (HB1),
to retrieve context and generate an answer. The retrieval process
starts with the highest similarity threshold. If no relevant content
is retrieved, we apply the adaptive thresholding strategy. Each an-
swer is then evaluated using the LLM-as-a-Judge prompt (Figure 6),
which generates a score based on helpfulness, correctness, and
detail. These scores are used to compare the quality of generated
responses against accepted Stack Overflow answers. To statistically
compare the quality of our method against accepted answers, we
use the Mann–Whitney U test ( 𝑝-value) and report effect size using
Cliff’s delta.
Results .Table 3 summarizes the performance of the adaptive
thresholding strategy across similarity thresholds ranging from
0.9 to 0.5.
We observe a clear trade-off between quality and coverage.
At the highest threshold (0.9), the pipeline retrieves context for only
0.7% of questions but achieves the highest response quality (mean
= 6.44, median = 7.0), outperforming the corresponding accepted
answers (mean = 5.03, median = 5.0). As the threshold is lowered,
coverage increases substantially, reaching 31.9% at 0.8 and 76.5%
at 0.7. Up to this threshold (0.7), generated responses consistently
outperform accepted answers, with statistically significant differ-
ences and medium to small effect sizes (Cliff’s delta ranging from
0.41 to 0.28, 𝑝<0.001).
While adaptive thresholding achieves full coverage by 0.5,
it comes at the cost of reduced but reasonable answer quality.
At thresholds 0.6 and 0.5, generated responses are outperformed
by accepted answers, with Cliff’s delta values of −0.29and−0.44,
indicating small to medium negative effect sizes. It is important
to note that the accepted answer scores at each threshold are not
affected by the threshold itself; they simply reflect the scores for the
subset of questions retrieved at that level. Despite the decline, the
generated answers remain within the range of scores observed for
accepted answers across the dataset (means between 4.69 and 6.34),Query
What methods can be used
to transform a ResultSet
into an actionable integer
value?
Accepted answer (LLM-
Judge = 5)
A ResultSet has a series
of getXYZ(int) methods
to retrieve
columns from it by their
relative index and
corresponding
getXYZ(String) methods
to retrieve those
columns by their alias.
In your case, using the
index variant getInt(
int) would be
the easiest:
recordNumber = resultSet.
getInt(1);Adaptive HB1 (LLM-Judge = 7)
To transform a ResultSet into an
actionable integer value,
you can use one of the following
methods:
- getInt(int column): [Basic
explanation omitted.]
- getObject(int column) followed by
casting to Integer:
[Details about usage and casting
omitted.]
- getLong(int column) or getShort(int
column):
[Explanation of method differences
omitted.]
Here 's an example of how to use these
methods:
[Sample code omitted.]
Note that getInt(int column) is
generally the recommended
approach when working with ResultSets
in Java.
[Further explanation omitted.]
Figure 7: Qualitative example from the Unseen Question Set .
HB1 scores 7/10, outperforming the accepted answer (5/10).
with median scores equal to 6.0. This indicates that even when the
retrieved context is only weakly related, the generated responses
maintain a reasonable level of quality.
When considering the entire Unseen Question Set, the
adaptive thresholding strategy leads to a statistically sig-
nificant overall improvement. The weighted mean score for
generated answers is 5.76 (median = 6.0), compared to 5.04 (median
= 5.0) for accepted answers. The difference is statistically signif-
icant (Mann- Whitney U, 𝑝<10−82) with a small positive effect
size (Cliff’s delta = 0.21). Overall, these results demonstrate that
dynamic thresholding reliably improves or matches the quality of
accepted answers, while guaranteeing full retrieval coverage for
previously unseen questions.
Qualitative insights. Figure 7 illustrates a typical unseen ques-
tion outcome. The accepted answer receives an LLM-Judge score of
5/10 because it only lists the basic getInt(int) call and provides
minimal justification. On the other hand, the HB1 response scores
7/10 because it lists three alterbative APIs ( getInt ,getObject ,
getLong/Short ) and offers additional guidance on when each is
appropriate. This example illustrates the pattern observed in our
quantitative results: HB1 provides more context and usage advice
than the accepted answer on SO.
RQ3: How does our proposed RAG pipeline perform across
different LLMs?
Motivation .While our proposed RAG pipeline has demonstrated
effectiveness with one language model, it is unclear whether this
performance (i) transfers to models with different architectures and
pre-training, and (ii) still outperforms a strong zero-shot baseline for
each model. By evaluating HB1 on multiple LLMs and comparing its
output with the models’ own zero-shot responses, we can measure

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Table 3: Adaptive–threshold performance on the Unseen Question Set (5 510 questions). For each threshold we report answer
quality (generated vs. accepted), significance tests, and retrieval coverage. Cumulative coverage reaches 100% once the threshold
reaches 0.5.
Threshold # QuestionsMean Median Statistical testCoverage (%)
Generated Accepted Generated Accepted U p 𝚫 ES
0.9 39 6.44 5.03 7.0 5.0 1 071.5 0.0015 0.41 med. 0.7 (+0.7)
0.8 1 720 6.20 4.91 7.0 5.0 2 037 269 <0.001 0.38 med. 31.9 (+31.2)
0.7 2 456 5.60 4.69 6.0 4.0 3 874 331 <0.001 0.28 small 76.5 (+44.6)
0.6 1 181 5.50 6.10 6.0 7.0 494 031 <0.001 –0.29 small 97.9 (+21.4)
0.5 108 5.13 6.34 6.0 7.0 3 260 <0.001 –0.44 med. 100.0 (+2.1)
U = Mann–Whitney statistic; 𝑝= two-sided p-value; Δ= Cliff’s delta; ES = effect-size label.
both robustness (does the pipeline generalize?) and added value
(does retrieval still help when the underlying model changes?).
Approach .To evaluate the generalizability and practical impact
of our optimal RAG pipeline, we apply it to multiple LLMs to test
whether it improves answer quality over standard zero-shot prompt-
ing. We select several widely used models for this comparison, in-
cluding Granite-3.1-8B-Instruct ,Mistral-7B-Instruct-v0.3 ,Qwen3-8B
andLLaMA-3.1-8B-Instruct used in previous RQs.
For each model, we generate answers to the 385 questions in
theSynthetic Question Set using two configurations: (1) our
optimized RAG pipeline (HB1) with dynamic thresholding from
0.9 to 0.1, and (2) a standard zero-shot prompting baseline without
retrieval. All responses are assessed using the same LLM-as-a-Judge
framework as in Figure4, with a focus on helpfulness, correctness,
and level of detail. To statistically compare the effectiveness of
each approach, we use the Wilcoxon signed-rank test [ 37], which
is well-suited for evaluating paired scores from HB1 and zero-shot
responses for each LLM. This non-parametric test is appropriate
because it directly compares paired samples without assuming the
score distributions are normal.
Results .Figure 8 presents the performance differences between
zero-shot prompting and HB1 across all tested LLMs. Our optimal
RAG pipeline (HB1) consistently improves or matches an-
swer quality across three evaluated LLMs. For LLaMA-3.1-8B-
Instruct, HB1 achieves a mean score of 5.95 (median 6.0), surpassing
the zero-shot baseline (mean 5.31, median 5.0) with statistical signif-
icance ( Wilcoxon p-value: 1×10−2). Granite-3.1-8B-Instruct shows
a similar pattern: HB1 reaches a mean of 6.31 (median 7.0), com-
pared to 6.14 (median 6.0) for zero-shot ( p-value: 3×10−3). For
Mistral-7B-Instruct-v0.3, HB1’s mean is 5.96 (median 7.0), again
outperforming zero-shot (mean 5.83, median 6.0) with a highly
significant difference ( p-value: 8.8×10−6).
For Qwen3-8B, HB1 does not improve over zero-shot prompt-
ing, showing that better-trained models benefit less from
retrieval augmentation. In contrast, for Qwen3-8B, zero-shot
prompting yields a slightly higher mean score (6.15) than HB1
(5.97), though both achieve a median of 6.0 and no statistically
significant difference is observed.
While mean scores offer a high-level comparison, the dis-
tribution reveals how HB1 improves not just typical answer
quality but also output consistency , as seen in Figure 8. First, formodels such as Mistral, LLaMA, and Granite, HB1 reduces the fre-
quency of low-quality outputs (scores <3) and narrows the overall
distribution. This indicates not only higher average scores, but also
increased answer consistency. For Mistral and LLaMA, we note an
observable upward shift in the distribution. Second, in the case of
Qwen3-8B, both HB1 and zero-shot exhibit somewhat similar wide
distributions centered around a median of 6. This suggests that
stronger models with broader pretraining may benefit less from
retrieval, as they already encode much of the required knowledge.
Overall, these results confirm that our optimal RAG pipeline -
HB1 not only generalizes well across open source LLMs but also
enhances answer quality and reliability in most cases, supporting
its practical value for technical Q&A.
Figure 8: Comparison of answer score distributions between
zero-shot prompting and the HB1 RAG pipeline across mul-
tiple LLMs
4 Discussion
4.1 Qualitative Insights
Results from the research questions show that our optimal RAG
pipeline consistently improves or matches the accepted answers
on Stack Overflow dataset and the zero-shot prompting across

Never Come Up Empty: Adaptive HyDE Retrieval for Improving LLM Developer Support Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
all evaluated LLMs. However, numeric gains alone do not fully
capture the practical value of the proposed RAG pipeline. To better
understand the nature of these improvements, we conduct a focused
qualitative review of the model outputs.
We randomly sample 20 questions (5 per model) from the KB-
Synthetic Q-Set where HB1. For each question, we compare the
answers and highlight common improvements, such as increased
detail, clearer structure, or usefulness. We then consolidate our
observations to identify common qualitative themes that illustrate
how HB1 enhances the response value beyond the observed few
point margin in the LLM-As-A-Judge score.
Cases where HB1 improves response quality: Among the ex-
amples analyzed, HB1 outperformed zero-shot prompting in 75%
of the cases. Score differences range from +2 to +6, with a mean of
2.3 and a median of 3.0. The qualitative review reveals consistent
improvements in the utility and presentation of the content.
Specifically, HB1 answers more frequently includes:
•Best-practice use of APIs. HB1 answers more frequently use
concise, recommended solutions that align with best practices
in a given programming ecosystem. For example, in a question
about assigning group indices in a Pandas DataFrame, HB1 use the
best-practice one-liner solution df[’id’] = df.groupby(’col’)
.cumcount() + 1 , while the zero-shot answer suggests manually
looping through rows, an approach that is less efficient and harder
to maintain.
•Richer contextualization. Compared to zero-shot answers, HB1
more often includes brief explanations of why a solution works
or when it should be used. For instance, in a question about
plt.axis(‘equal’) in Matplotlib, HB1 explains that this ensures
equal scaling on both axes (useful for plotting circles or squares
accurately) and shows how to adjust the figure layout to avoid
distortion, which are details missing in the zero-shot answer.
•Edge-case handling. HB1 is also more likely to mention failure
scenarios or constraints. For example, when answering a question
about limiting checkbox selection to three options using jQuery,
the HB1 response includes logic to disable unchecked boxes once
the user reaches the limit. This type of input validation is rarely
included in zero-shot answers but is critical for correct user inter-
action in production environments.
Table 4 lists five representative examples where HB1 adds value
compared to the baseline.
Cases where zero-shot performs better: In 25% of the cases from
the random sample, zero-shot prompting outperformed HB1. These
cases are more likely questions that require conceptual clarity rather
than code synthesis. We observe that retrieval either introduces
off-topic content or leads the model to address a broader variant of
the task, missing the specific user intent. Table 5 shows examples of
the questions and the observed issues. In two AngularJS questions,
HB1 is misled by retrieved content and generates off-topic answers
about service patterns. In the third case, HB1 overgeneralizes a
question about removing duplicate permutations, solving a more
complex variant than intended. In the three examples, zero-shot
responses were better aligned with the user’s intent.
To conclude, this analysis shows that HB1 outperforms zero-
shot prompting in 75% of sampled cases, offering code that is more
aligned with best practices, contextual explanations, and edge-casehandling. In the remaining 25%, zero-shot responses are stronger,
typically for concept-focused questions where retrieval led HB1 to
go off-topic or to overgeneralize.
4.2 Implications
Implications for practitioners. Our results demonstrate that com-
bining HyDE-based retrieval with full-answer granularity and dy-
namic thresholding improves both the retrieval coverage and qual-
ity of generated answers. The HB1 pipeline improves the LLM-as-
a-Judge scores while maintaining 100% retrieval coverage on unseen
questions. These improvements are most common in implementation-
oriented questions, where HB1 produces code more in line with
best practices, contextual detail, and input validation logic. Two
configuration choices were particularly effective: (1) initializing re-
trieval at a high similarity threshold (e.g., 0.7) and lowering it only
when no candidates are found, and (2) retrieving complete answers
rather than individual sentences, as full answers more often include
supporting rationale and edge-case handling. For concept-focused
questions, retrieval was slightly less effective. As such, RAG based
systems may benefit from a lightweight classifier to decide when
retrieval should be skipped in favor of zero-shot prompting.
Implications for researchers. The study raises two research direc-
tions. First, we observe model-dependent variation when measuring
the benefit of retrieval: Qwen-3-8B showed minimal gains from
HB1, likely due to broader pretraining, while other models show
important improvement. Future work should investigate when re-
trieval is beneficial based on model characteristics or question type.
Second, most of the improvements observed in HB1, such as the
best-practice use of APIs and secure coding practices, are unlikely
to be captured by standard metrics commonly used in prior work
(e.g., BLEU, ROUGE), which rely on lexical overlap. Our use of LLM-
as-a-Judge enables an evaluation of aspects like completeness and
practical utility. This highlights the need for evaluation methods
that better reflect developer-relevant quality criteria, and motivates
the development of metrics that assess answer usefulness beyond
surface-level similarity.
5 Threats to validity
In this section, we discuss the threats to the validity of our study.
Threats to construct validity Our evaluation relies on LLM-
as-a-Judge, which used and evaluated in recent studies, though it
may still introduce bias, inconsistency, or hallucination. To mitigate
these risks, we anonymize responses and conduct manual evalua-
tions to assess the alignment between LLM-as-a-Judge scores and
human judgments (see Section 2.2).
Threats to internal validity Data leakage is a significant con-
cern in LLM-based evaluation, as test questions may overlap with
the training data or retrieval corpus. To mitigate this risk, we gen-
erate synthetic questions for "seen" cases and construct an unseen
test set using posts published after the LLM training cutoff.
Threats to external validity We focus on questions tagged
with Java and Python from Stack Overflow, which may limit gener-
alization to other domains, languages, or Q&A platforms. However,
as Java and Python are among the most popular and widely used
programming languages, our findings are likely relevant to a broad
range of developer scenarios.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Table 4: Examples where HB1 significantly outperforms zero-shot.
Question (shortened) Model Zero-shot focus HB1 focus Why HB1 wins
Counting series in Pandas Mistral Uses a manual row loop to as-
sign group indices.Applies the concise
groupby().cumcount()
+ 1idiom.HB1 provides the correct and ef-
ficient one-liner standard in real-
world Pandas use.
jQuery execution order LLaMA Mentions only .each() as an
option.Lists five sequencing patterns
(e.g.,$().ready() , promises,
setTimeout ).HB1 offers broader and more prac-
tical coverage for managing execu-
tion flow.
Zero-padding integers in
JavaLLaMA Shows String.format()
without explanation.Includes both
String.format() and
manual padding with width
logic.HB1 is more complete and explains
when to use each approach.
plt.axis(’equal’)
behaviour Qwen Callsplt.axis(’equal’)
but omits layout implica-
tions.Addsset_aspect(’equal’,
adjustable=’box’) and ex-
plains distortion issues.HB1 provides a more robust solu-
tion with better layout control.
Vuev-model with props Mistral Uses basic prop binding but
omits two-way support.Implements the full com-
puted getter/setter pattern
with@input .HB1 shows the correct two-way
binding pattern for props.
Table 5: Examples where zero-shot significantly outperforms HB1.
Question (shortened) Model Zero-shot focus HB1 focus Why Zero-shot wins
Remove duplicate per-
mutations of digit ar-
raysMistral Set-based one-liner that cor-
rectly deduplicates permuta-
tions.Stringify() + Settechnique suit-
able for flat arrays but not
forarrays-of-arrays . Problem is
mis-interpreted.Zero-shot matches the stated task.
HB1 over-generalizes and solves the
wrong problem.
Curly-brace syntax in
AngularJSMistral Accurate explanation of {{ }}
interpolation, security consid-
erations, and ngBind .Discusses factory vs.service
patterns resulting in a topic
drift.Retriever fetches a semantically re-
lated but topically different answer
which leads HB1 off-topic.
Curly-brace syntax in
AngularJSLLaMA Same concise and on-point tu-
torial as MistralRepeats factory/service discus-
sion. Still off-topic.Same retrieval mismatch. Zero-shot
is aligned with the actual question.
6 Related Work
Stack Overflow remains a central resource for developer knowledge,
and its content has been widely leveraged to build datasets and
systems for tasks such as answer retrieval, code snippet recommen-
dation, and question summarization [ 40,41]. Many studies have
mined accepted answers or high-quality responses from Stack Over-
flow to serve as gold standards for training and evaluating machine
learning models [ 12,27]. In recent years, Retrieval-Augmented Gen-
eration (RAG) has emerged as a way to enhance large language
models (LLMs) with domain knowledge—including API usage, trou-
bleshooting, and community explanations—to improve developer
support [ 25]. While vanilla RAG setups rely on the input query
for retrieval [ 20], this approach can struggle with vague or novel
questions. To address these challenges, recent works have pro-
posed enhancements such as Hypothetical Document Embedding
(HyDE), which generates a pseudo-answer to improve retrieval
alignment [ 36], and filtering mechanisms to remove irrelevant con-
tent before generation [ 25]. Recent works such as RAGFix [ 22]
and StackRAG [ 1] have demonstrated the potential of combining
Stack Overflow knowledge with RAG pipelines for tasks like code
repair and developer Q&A. However, the most existing approaches
focus on direct query-based retrieval, which often fails on vagueor novel queries and do not explicitly address pipeline general-
ization. In contrast, our work systematically evaluates a range of
retrieval strategies—including hypothetical document embedding
(HyDE)—as well as content granularity and adaptive threshold-
ing, using large-scale Stack Overflow data. Through this analysis,
we identify pipeline configurations that effectively handle novel
questions and generalize across different open-source LLMs.
7 Conclusion
In this paper, we investigate adopting RAG to improve the LLMs’
capability to answer developers’ questions by constructing RAG
using Stack Overflow posts with accepted answers as a knowl-
edge base. By constructing a large-scale dataset of Java and Python
Q&A pairs and evaluating seven different RAG pipeline designs,
we identify HyDE-Based 1 (HB1) as the most effective. HB1 gen-
erates a hypothetical answer for each question and retrieves full
accepted answers based on this pseudo-answer for context, achiev-
ing a strong balance between answer quality and retrieval coverage
across a range of similarity thresholds. To address questions with
no close matches in the knowledge base, we introduce adaptive
thresholding, which dynamically lowers the retrieval threshold
to improve coverage for novel questions. To simulate real-world

Never Come Up Empty: Adaptive HyDE Retrieval for Improving LLM Developer Support Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
scenarios, we evaluate our approach on 5,510 Stack Overflow ques-
tions sampled from posts after the LLM’s training cutoff, ensuring
they were unseen during retrieval. At high similarity thresholds,
our approach often exceeds the quality of the original accepted
answers.
We further test our pipeline across multiple LLMs, finding that
our adaptive RAG approach robustly enhances answer quality or
matches strong zero-shot baselines for most models. Overall, our
findings demonstrate that the proposed optimal RAG pipeline, com-
bined with adaptive thresholding, provides a practical and effective
solution for delivering reliable, high-quality developer assistance
on both familiar and novel questions. In future work, we plan to
evaluate our approach on additional datasets and with proprietary
models such as the GPT family from OpenAI.
References
[1]Davit Abrahamyan and Foutse Khomh Fard. 2024. StackRAG Agent: Improving
Developer Answers with Retrieval-Augmented Generation. In IEEE International
Conference on Software Maintenance and Evolution (ICSME) . https://ieeexplore.
ieee.org/document/10795043
[2]Meta AI. 2024. LLaMA 3.1–8B-Instruct. https://huggingface.co/meta-llama/Meta-
Llama-3-8B-Instruct.
[3]Mistral AI. 2024. Mistral-7B-Instruct-v0.3. https://huggingface.co/mistralai/
Mistral-7B-Instruct-v0.3.
[4]Anonymous Authors. 2025. ICSE-C2-Stack-overflow: Retrieval-Augmented Gen-
eration for Developer Questions. https://anonymous.4open.science/r/ICSE-C2-
Stack-overflow-16E8/README.md. Accessed: July 2025.
[5]Emily M Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret
Shmitchell. 2021. On the dangers of stochastic parrots: Can language models
be too big?. In Proceedings of the 2021 ACM conference on fairness, accountability,
and transparency . 610–623.
[6]Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora,
Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma
Brunskill, et al .2021. On the opportunities and risks of foundation models. arXiv
preprint arXiv:2108.07258 (2021).
[7]Boxi Cao, Hongyu Lin, Xianpei Han, Le Sun, Lingyong Yan, Meng Liao, Tong
Xue, and Jin Xu. 2021. Knowledgeable or educated guess? revisiting language
models as knowledge bases. arXiv preprint arXiv:2106.09231 (2021).
[8]Mingyue Cheng, Yucong Luo, Jie Ouyang, Qi Liu, Huijie Liu, Li Li, Shuo Yu,
Bohou Zhang, Jiawei Cao, Jie Ma, et al .2025. A survey on knowledge-oriented
retrieval-augmented generation. arXiv preprint arXiv:2503.10677 (2025).
[9]Vineesh Cutting and Nehemiah Stephen. 2021. Comparative review of java and
python. International Journal of Research and Development in Applied Science and
Engineering (IJRDASE) 21, 1 (2021).
[10] Dilrukshi Gamage, Piyush Ghasiya, Vamshi Bonagiri, Mark E Whiting, and
Kazutoshi Sasahara. 2022. Are deepfakes concerning? analyzing conversations
of deepfakes on reddit and exploring societal implications. In Proceedings of the
2022 CHI conference on human factors in computing systems . 1–19.
[11] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise zero-
shot dense retrieval without relevance labels. In Proceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) .
1762–1777.
[12] Ziyu Gao, Xin Xia, David Lo, John Grundy, and Tian Zhang. 2023. I Know What
You Are Searching For: Code Snippet Recommendation from Stack Overflow
Posts. ACM Transactions on Software Engineering and Methodology (TOSEM)
(2023). https://dl.acm.org/doi/10.1145/3550150
[13] Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A Smith.
2020. Realtoxicityprompts: Evaluating neural toxic degeneration in language
models. arXiv preprint arXiv:2009.11462 (2020).
[14] Aidan Gilson, Xuguang Ai, Thilaka Arunachalam, Ziyou Chen, Ki Xiong Cheong,
Amisha Dave, Cameron Duic, Mercy Kibe, Annette Kaminaka, Minali Prasad,
et al.2024. Enhancing Large Language Models with Domain-specific Retrieval
Augment Generation: A Case Study on Long-form Consumer Health Question
Answering in Ophthalmology. arXiv preprint arXiv:2409.13902 (2024).
[15] Wael H Gomaa, Aly A Fahmy, et al .2013. A survey of text similarity approaches.
international journal of Computer Applications 68, 13 (2013), 13–18.
[16] Sai Muralidhar Jayanthi, Varsha Embar, and Karthik Raghunathan. 2021. Evalu-
ating pretrained transformer models for entity linking in task-oriented dialog.
arXiv preprint arXiv:2112.08327 (2021).
[17] Thomas Junk. 1999. Confidence level computation for combining searches with
small statistics. Nuclear Instruments and Methods in Physics Research Section A:
Accelerators, Spectrometers, Detectors and Associated Equipment 434, 2-3 (1999),435–443.
[18] S. Kumar and N. Patel. 2025. CodeCapBench: Benchmarking LLMs for Capability-
Oriented Software Engineering Tasks. In Proceedings of ICSE 2025 .
[19] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Kulkarni, Xiang Cheng, Angela Fan, Vishrav Chaudhary,
and et al. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP
Tasks. In Advances in Neural Information Processing Systems (NeurIPS) .
[20] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Kulkarni, Xiang Cheng, Angela Fan, Vishrav Chaudhary,
and et al. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP
Tasks. In Advances in Neural Information Processing Systems .
[21] Siran Li, Linus Stenzel, Carsten Eickhoff, and Seyed Ali Bahrainian. 2025. Enhanc-
ing Retrieval-Augmented Generation: A Study of Best Practices. arXiv preprint
arXiv:2501.07391 (2025).
[22] Elijah Mansur, Johnson Chen, Muhammad Anas Raza, and Mohammad Wardat.
2024. RAGFix: Enhancing LLM Code Repair Using RAG and Stack Overflow Posts.
In2024 IEEE International Conference on Big Data (BigData) . IEEE, 7491–7496.
[23] P. Mirza, L. Weber, and F. Küch. 2025. Stratified Selective Sampling for Instruction
Tuning with Dedicated Scoring Strategy. arXiv preprint arXiv:2505.22157 (2025).
[24] Mayank Mishra, Matt Stallone, Gaoyuan Zhang, Yikang Shen, Aditya Prasad,
Adriana Meza Soria, Michele Merler, Parameswaran Selvam, Saptha Surendran,
Shivdeep Singh, et al .2024. Granite code models: A family of open foundation
models for code intelligence. arXiv preprint arXiv:2405.04324 (2024).
[25] Mononito Mukherjee and Veselin J. Hellendoorn. 2025. SOSecure: Safer Code Gen-
eration with RAG and StackOverflow Discussions. arXiv preprint arXiv:2503.13654
(2025). https://arxiv.org/abs/2503.13654
[26] Sarah Nadi and Christoph Treude. 2020. Essential sentences for navigating stack
overflow answers. In 2020 IEEE 27th International Conference on Software Analysis,
Evolution and Reengineering (SANER) . IEEE, 229–239.
[27] Md Rabiul Parvez, Wasi Uddin Ahmad, Saikat Chakraborty, and Baishakhi Ray.
2021. Retrieval Augmented Code Generation and Summarization. arXiv preprint
arXiv:2108.11601 (2021). https://arxiv.org/abs/2108.11601
[28] Md Masudur Rahman, Jed Barson, Sydney Paul, Joshua Kayani, Federico Andrés
Lois, Sebastián Fernandez Quezada, Christopher Parnin, Kathryn T Stolee, and
Baishakhi Ray. 2018. Evaluating how developers use general-purpose web-search
for code retrieval. In Proceedings of the 15th International Conference on Mining
Software Repositories . 465–475.
[29] Nikitha Rao, Chetan Bansal, Thomas Zimmermann, Ahmed Hassan Awadallah,
and Nachiappan Nagappan. 2020. Analyzing web search behavior for software
engineering tasks. In 2020 IEEE International Conference on Big Data (Big Data) .
IEEE, 768–777.
[30] Nils Reimers and Iryna Gurevych. 2020. Making monolingual sentence embed-
dings multilingual using knowledge distillation. arXiv preprint arXiv:2004.09813
(2020).
[31] Steven I Ross, Fernando Martinez, Stephanie Houde, Michael Muller, and Justin D
Weisz. 2023. The programmer’s assistant: Conversational interaction with a large
language model for software development. In Proceedings of the 28th International
Conference on Intelligent User Interfaces . 491–514.
[32] James Skripchuk, Neil Bennett, Jeffrey Zhang, Eric Li, and Thomas Price. 2023.
Analysis of novices’ web-based help-seeking behavior while programming. In
Proceedings of the 54th ACM Technical Symposium on Computer Science Education
V. 1. 945–951.
[33] Jinyan Su, Jin Peng Zhou, Zhengxin Zhang, Preslav Nakov, and Claire Cardie.
2024. Towards More Robust Retrieval-Augmented Generation: Evaluating RAG
Under Adversarial Poisoning Attacks. arXiv preprint arXiv:2412.16708 (2024).
[34] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne
Lachaux, Timothée Lacroix, et al .2023. LLaMA: Open and Efficient Foundation
Language Models. arXiv preprint arXiv:2302.13971 (2023).
[35] Bogdan Vasilescu, Vladimir Filkov, and Alexander Serebrenik. 2013. Stackover-
flow and github: Associations between software development and crowdsourced
knowledge. In 2013 International conference on social computing . IEEE, 188–195.
[36] Zizhao Wang, Akari Asai, Xinyi Yu, Frank F Xu, Yizhou Xie, and Graham Neubig.
2024. Coderag-bench: Can retrieval augment code generation? arXiv preprint
arXiv:2406.14497 (2024). https://arxiv.org/abs/2406.14497
[37] Frank Wilcoxon. 1945. Individual comparisons by ranking methods. Biometrics
bulletin 1, 6 (1945), 80–83.
[38] Xin Xia, Lingfeng Bao, David Lo, Pavneet Singh Kochhar, Ahmed E Hassan, and
Zhenchang Xing. 2017. What do developers search for on the web? Empirical
Software Engineering 22 (2017), 3149–3185.
[39] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al .2025. Qwen3 technical
report. arXiv preprint arXiv:2505.09388 (2025).
[40] Chen Yang, Baowen Xu, Ferdian Thung, Yuxin Shi, and Tiancheng Zhang. 2022.
Answer summarization for technical queries: Benchmark and new approach.
Proceedings of the 37th IEEE/ACM International Conference on Automated Software
Engineering (ASE) (2022). https://dl.acm.org/doi/10.1145/3551349.3560421
[41] Tianyi Zhang, Ting Zhang, Yanjun Di, Minghui Chen, and Tao Zhang. 2022.
SOSum: A Dataset of Stack Overflow Post Summaries. Proceedings of the 19th

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
International Conference on Mining Software Repositories (MSR) (2022). https: //dl.acm.org/doi/10.1145/3524842.3528487
# FreshStack: Building Realistic Benchmarks for Evaluating Retrieval on Technical Documents

**Authors**: Nandan Thakur, Jimmy Lin, Sam Havens, Michael Carbin, Omar Khattab, Andrew Drozdov

**Published**: 2025-04-17 17:44:06

**PDF URL**: [http://arxiv.org/pdf/2504.13128v1](http://arxiv.org/pdf/2504.13128v1)

## Abstract
We introduce FreshStack, a reusable framework for automatically building
information retrieval (IR) evaluation benchmarks from community-asked questions
and answers. FreshStack conducts the following steps: (1) automatic corpus
collection from code and technical documentation, (2) nugget generation from
community-asked questions and answers, and (3) nugget-level support, retrieving
documents using a fusion of retrieval techniques and hybrid architectures. We
use FreshStack to build five datasets on fast-growing, recent, and niche topics
to ensure the tasks are sufficiently challenging. On FreshStack, existing
retrieval models, when applied out-of-the-box, significantly underperform
oracle approaches on all five topics, denoting plenty of headroom to improve IR
quality. In addition, we identify cases where rerankers do not clearly improve
first-stage retrieval accuracy (two out of five topics). We hope that
FreshStack will facilitate future work toward constructing realistic, scalable,
and uncontaminated IR and RAG evaluation benchmarks. FreshStack datasets are
available at: https://fresh-stack.github.io.

## Full Text


<!-- PDF content starts -->

FreshStack: Building Realistic Benchmarks for Evaluating
Retrieval on Technical Documents
Nandan Thakur∗
University of Waterloo
Ontario, CanadaJimmy Lin
University of Waterloo
Ontario, CanadaSam Havens
Databricks
San Francisco, USA
Michael Carbin
Databricks
San Francisco, USAOmar Khattab
Databricks
San Francisco, USAAndrew Drozdov
Databricks
San Francisco, USA
Abstract
We introduce FreshStack, a reusable framework for automatically
building information retrieval (IR) evaluation benchmarks from
community-asked questions and answers. FreshStack conducts the
following steps: (1) automatic corpus collection from code and tech-
nical documentation, (2) nugget generation from community-asked
questions and answers, and (3) nugget-level support, retrieving
documents using a fusion of retrieval techniques and hybrid archi-
tectures. We use FreshStack to build five datasets on fast-growing,
recent, and niche topics to ensure the tasks are sufficiently chal-
lenging. On FreshStack, existing retrieval models, when applied
out-of-the-box, significantly underperform oracle approaches on all
five topics, denoting plenty of headroom to improve IR quality. In
addition, we identify cases where rerankers do not clearly improve
first-stage retrieval accuracy (two out of five topics). We hope that
FreshStack will facilitate future work toward constructing realistic,
scalable, and uncontaminated IR and RAG evaluation benchmarks.
FreshStack datasets are available at: https://fresh-stack.github.io.
Keywords
Information Retrieval, General Framework, Evaluation Benchmark
1 Introduction
Retrieval-augmented generation (RAG) is a popular technique to
enhance traditional information retrieval (IR) capabilities with lan-
guage model generation. RAG systems use large language models
(LLMs) to generate long-form responses [ 5,20,23,33], grounded in
the information available from retrieved documents [ 18,31,33,40].
Despite its wide usage, evaluating RAG remains incredibly chal-
lenging. Existing IR and RAG benchmarks are not well-suited for
evaluation, as these are outdated and highly limited. In particular,
we observe three major issues in existing benchmarks:
•Lack of realistic, open-ended questions : Existing datasets con-
tain purely extractive short answers (e.g., Natural Questions [ 32],
TriviaQA [ 28]) or crowd-sourced questions (e.g., HotPotQA [ 77]).
A limited number of datasets capture “natural” human-asked
questions, i.e., MS MARCO [ 45] or Natural Questions [ 32], but
unfortunately, brief and straightforward questions are inserted
into a search box, failing to represent the complex questions that
real users might pose to modern RAG systems.
•Artificially easy : RAG represents an approach rather than a prob-
lem. Real users require systems capable of grounded question
∗Work done during Nandan’s internship at Databricks.IR / RAG BenchmarksNiche Complex Dynamic Challenge
Domains Questions Updates Level
CQADupstack [21] No No No Easy
CodeSearchNet [22] No No No Easy
COIR [35] Limited Yes No Moderate
Stack Overflow-QA [35] No Yes No Moderate
CodeRAG-Bench [72] Limited No No Moderate
Neural Code Search [34] No No No Moderate
SWE-Bench [27] No Yes Yes High
FreshStack (ours) Yes Yes Yes High
Table 1: A comparison of existing IR/RAG evaluation bench-
marks with FreshStack.
answering, i.e., responding to specialized questions by referenc-
ing knowledge from a document corpus. Consequently, datasets
constructed by design to be solvable via retrieval often fail to
encode challenges faced in RAG applications.
•Static and unspecialized : After sourcing questions and an-
swers, a benchmark becomes at the risk of (1) contamination ,
if current LLMs are trained on the same set of documents or
questions, (2) overfitting , when systems inevitably saturate by
repeated internal or external leaderboarding (e.g., BEIR [ 63]), and
(3)staleness , when questions or answers are not refreshed and
become outdated.
A realistic benchmark should measure model generalization on
niche domains, and continue to update. Additionally, it must capture
the complexity of human-generated queries—such as multi-hop rea-
soning [ 77], code understanding [ 27], or specialized terminology—
rather than relying on artificially easy questions. This drives the
robustness of systems in answering questions in evolving public
libraries or private code bases [ 27,80], a company’s internal fo-
rum [52], or technical troubleshooting [7, 50, 60].
In our work, we introduce FreshStack, a holistic framework for
constructing realistic datasets on niche and challenging domains,
seeking to avoid contamination due to (perpetual) recency. Using
FreshStack, we construct an evaluation benchmark on five niche
topics sourced from community-asked questions and answers on
Stack Overflow and a corpus containing code snippets and tech-
nical documents from public GitHub repositories. The FreshStack
framework contains three major steps:
(1)Automatic corpus collection (Section 3.2) with technical docu-
ments chunked and sourced from several GitHub repositories.
(2)Nugget generation (Section 3.3) with GPT-4o using community-
asked questions and answers in Stack Overflow.
1arXiv:2504.13128v1  [cs.IR]  17 Apr 2025

Thakur et al.
Chromadb from_documents  function giving error  
The following function was working till a few days ago but now  [ ... ] I slightly modify your code, using HuggingFaceEmbeddings
instead of SentenceT ransformerEmbeddingsThe error is due to a mismatch in the
function signature expected by
`Chroma.from_documents` when using
SentenceT ransformerEmbeddings.
 Initialize `HuggingFaceEmbeddings` with
the model name "sentence-
transformers/all-MiniLM-L6-v2"
Pass the initialized
`HuggingFaceEmbeddings` to 
the `Chroma.from_documents` function. Community-asked QueryAccepted Answer
GPT-4o
GPT-4o generated NuggetsRetrieved Document Chunks
ID: 78256389
langchain/[ ... ]/sentence_transformers.ipynb
GPT-4o
GPT-4o
GPT-4o
Figure 1: A data instance from LangChain generated with FreshStack. The question and answer pair is sourced from Stack
Overflow and provided as input to generate nuggets with GPT-4o (highlighted in blue). Next, code snippets and technical
documents from multiple GitHub repositories (e.g., Jupyter Notebook) are chunked, processed, and pooled for each question.
Finally, each pooled document chunk is judged with GPT-4o for binary relevance (either yes or no) at a nugget-level, i.e.,
whether the document factually supports the information present in each nugget.
(3)Nugget-level support (Section 3.5) with GPT-4o on document
chunks, retrieved from a fusion of retrieval techniques and
hybrid architectures.
FreshStack has several advantages over existing benchmarks (as
highlighted in Table 1). First, the framework utilizes user-asked
questions and curated answers, making the evaluation challenging.
We are not crafting artificial (LLM-generated) questions or sam-
pling questions myopically. Second, all answers are supported in
real time by information from technical documentation in GitHub
repositories. Third, the framework is designed to be general and
scalable without modification. Finally, the framework is focused
on niche domains and recent topics, taking careful measures to
mitigate risks with data contamination introduced by LLMs.
We provide a complete overview of our framework and inves-
tigate three research questions (RQs) to provide insights into the
design choices used in FreshStack:
RQ1 How to construct realistic and challenging IR evaluation
datasets from user-asked questions?
RQ2 How do LLMs act as an assessor for (1) nugget generation
with community-asked questions & answers and (2) nugget-
level support with GitHub-sourced documents?
RQ3 How do out-of-the-box retrieval and reranker systems per-
form on benchmark generated with FreshStack?
We investigate the framework’s quality by constructing five datasets
on niche topics, e.g., LangChain or Godot. We calibrate the auto-
matic stage with GPT-4o using a machine learning (ML) expert,
assessing the quality of nugget generation and nugget-level sup-
port for one of the topics (LangChain). Our results show that LLM-
generated nuggets capture crucial information required to answer
the question, and GPT-4o precisely labels support at a nugget level.
For the judgment pool construction, we assess pooling results by
comparing the oracle (having access to the answer) and inference
(relying only on the question) settings, showing that ensemble fu-
sion in question decomposition (in the inference setting) and nugget
generation (in the oracle setting) outperforms other techniques,
respectively. Beyond pool construction, we explore the document
retrieval setting on FreshStack, evaluating retrieval and rerankers
out-of-the-box, using the question alone. We find that retrieval
models drastically underperform oracle systems on all five topics,
showing a high headroom for improvement on specialized topics.In addition, ensemble fusion outperforms individual models, indi-
cating that diversity in models enhances retrieval, and rerankers
provide clear benefits in some but not all topics.
FreshStack is a general framework and can be applied to any
domain of a similar structure. The datasets will be made publicly
available to the research community. Overall, we hope FreshStack
can serve as a testbed for future work to develop similar challenging
benchmarks for evaluating IR and RAG systems.
2 Related Work
Retrieval-augmented generation. RAG has been widely used to
avoid “hallucinations” [ 81] seen by LLMs when handling knowledge-
intensive tasks [ 29]. RAG reduces the generation of factually incor-
rect content, leading to adoption in various commercial systems,
e.g., Bing Search or Google AI Overviews. Existing IR and RAG
benchmarks are stale, focused on text, evaluating on academic ques-
tion answering datasets [ 18,53,55], or are not challenging, being
constructed for RAG [ 8,36,46,76]. A limited number of datasets
refresh over time to avoid LLM decontamination [ 30,59,69,74],
however, these contain easy and unrealistic questions for evaluation.
In contrast, FreshStack generates niche and challenging datasets,
which are not constructed for RAG, and can refresh over time.
Code-based benchmarks. Neural code generation [ 42] requires
LLMs to generate code from scratch for generic programming ques-
tions. One popular benchmark is SWE-Bench [ 27], which evaluates
whether LLMs can generate code changes for GitHub pull requests
(PRs) in popular public repositories. Similarly, CodeSearchNet [ 22],
COIR [ 35], LiveCodeBench [ 25], and CodeRAG-Bench [ 72] focus
on the evaluation of high-level programming problems on popular
public repositories. In contrast, in FreshStack, we focus on assisting
developers, from a novice to a domain expert, by providing real-
time answers on fast-growing and recent topics such as LangChain
(introduced in 2023) by referencing technical documentation in
GitHub repositories.
Stack Overflow datasets. FreshStack is notthe first dataset to
use Stack Overflow for retrieval. However, the evaluation setting
of retrieving canonical documents from GitHub repositories re-
mains under-explored. Existing datasets such as CQADupstack
[21], LoTTE [ 57], and Stack Overflow-QA [ 35] follow a different
task, to retrieve the relevant answer snippet given the question
asked by a real user on Stack Overflow. The closest setting similar
2

FreshStack: Building Realistic Benchmarks for Evaluating Retrieval on Technical Documents
Top-k doc chunksLangchain: text splitter behavior Community-asked Query
According to the split_text  funcion in
RecursiveCharacterTextSplitter  
....
This will merge items if the cusum is less
than chunk size in your example is 10Expert Answer (Accepted)
(ii) Automatic Corpus Collection
(iii) Nugget   Generation
(i) Sourcing Question & AnswersDocument
Corpus
(iv) Ensemble Retrieval
(v) Binary Nugget-level SupportNugget 1: text or code
Nugget 2: text or code
Nugget 3: text or codeGPT-4oParsing &   
Chunking
GPT-4oNugget 1
Nugget 2
Nugget 3Doc 1 Doc kDocument
Corpus
Doc 1: 1, Doc k: 1
Doc 1: 0, Doc k: 1
Doc 1: 0, Doc k: 0
BM25
E5-Mistral-7BBGE
Voyage-large-2
Inference : Query,
Oracle : Answer,
NuggetsDecomp, answer
Figure 2: The FreshStack framework: (1) Stack Overflow ques-
tions and answers are sourced for recent and niche topics.
(2) GitHub repository documents are collected and chunked
to form the corpus (for each topic). (3) Nuggets or atomic
facts within the question and answer are generated with
GPT-4o. (4) Ensemble techniques and models retrieve docu-
ments, which construct our document judgment pools. (5)
GPT-4o evaluates support for every document-nugget pair
as a binary judgment.
to FreshStack is found in Neural Code Search [ 34], which incorpo-
rates public documentation and code snippet examples from GitHub
as the corpus to answer questions asked by real users on popular
programming topics such as Android.
3 The FreshStack Framework
The FreshStack framework involves five stages to construct an end-
to-end evaluation dataset (highlighted in Figure 2). The framework
includes three major design choices:
(1)Inclusion of recent and niche topics actively discussed in com-
puter programmer communities such as Stack Overflow.
(2)General and automatic framework that can be extended to
different domains (potentially even across languages) without
much manual effort.
(3)Sourcing community-asked questions and answers, to make our
evaluation challenging, requiring domain expert knowledge to
answer them correctly.
Stack Overflow is an online question answering platform for com-
puter programmers. It functions as a collaborative knowledge base,
utilizing a voting system to curate content quality. Users ask ques-
tions about a particular topic and provide a description (often a code
snippet with the error message) with a Stack Overflow tag. Anyone
in the community can answer the question, however, higher-voted
answers are prioritized, reflecting community consensus on the ac-
curacy and relevance of the answer. Questions and answers are also
tagged by topics, allowing for easy retrieval of topic-wise questions.
3.1 Stack Overflow Topic and Question Selection
For topic selection, we target niche and recent topics introduced
on Stack Overflow from 2023 onward, containing a minimum of
50 posts. We sort all topics using the overall number of posts and
curate five topics starting from the highest (LangChain) to thelowest frequency (Yolo v7 & v8) covering different domains and
sufficiently different from each other.
Questions & Answers. We extract relevant posts and answers
from the Stack Overflow XML data archive (dated October 2024).1
We scan the archive to pick all the relevant posts as questions con-
taining the required tag and to filter answer posts to the questions.
At the end, we filter and keep questions with an accepted answer ,
prioritizing precision over quantity in retaining questions with
high-quality answers.
3.2 Automatic Corpus Collection with GitHub
Answering a higher percentage of questions requires a robust set
of corpora from multiple sources . For instance, addressing issues
in LangChain may require ChromaDB GitHub documentation to
resolve errors related to its usage. In our work, we build a different
corpus set per topic by combining multiple GitHub repositories as
sources (we list the GitHub repositories per topic in Table 2).
Stack Overflow Tags. We analyze which GitHub repository to
select for the document corpus by analyzing tag frequency from
Stack Overflow posts. This involves identifying top-k co-occurring
tags, where k is the threshold balancing question coverage with
indexing costs. Some tags are generic, such as Python , whereas
others are specific, such as LangChainJS . Filtering the tags to keep
only a subset of repositories does not degrade the FreshStack dataset
quality. We manually verify GitHub repositories for each tag, with
plans to automate this procedure in the future. We could have
constructed the corpus directly, however, the top-k co-occurring
tags provide a useful signal on relevant GitHub repositories.
Chunking & Indexing. We clone the latest branch of a GitHub
repository in a local workspace and parse all files as a tree structure.
Each file (either a text document or code snippet) is chunked into
small sections of up to a maximum of 4096 tokens, skipping non-
text formats.2The GitHub filepath serves as the document identifier,
with additional chunk details encoded in the identifier. Finally, we
combine all chunks into a single corpus, prefixing all document
identifiers with the repository name to identify the common files
in each repository separately (e.g., LICENSE or requirements.txt).
3.3 Nuggetization or Nugget Generation
A nugget is a core concept or atomic fact essential in a system’s
response. The term nugget was informally referred to as SCU (sum-
mary content units) as clauses appearing in model summarization
[44] and later formalized as “information nugget” for evaluating
long-form answers [ 37,38,48,68].Nuggetization refers to con-
structing or generating nuggets from information-dense text. The
procedure decomposes a verbose answer into key atomic facts or
essential components, aiding evaluation. More recently, with the
onset of RAG, nugget-based evaluation has renewed interest with
LLMs for factual accuracy assessment in long answers [ 3,16,43,49].
Nuggetization. We automatically generate nuggets from Stack
Overflow question-answer pairs using GPT-4o [ 47], avoiding the
1The Stack Overflow XML data archive (CC BY-SA license) is updated once every
quarter: https://meta.stackexchange.com/questions/401324/announcing-a-change-to-
the-data-dump-process
2We skip indexing images, videos, .bin, .csv, and audio files or unrecognized file formats.
3

Thakur et al.
Topic GitHub Repositories
LangChain langchain-ai/langchain, langchain-ai/langchainjs, langchain-ai/langchain-nextjs-template, chroma-core/chroma, openai/openai-
cookbook, openai/openai-python, run-llama/llama_index, Azure-Samples/openai, Azure-Samples/azure-search-openai-demo, hug-
gingface/transformers
Yolo v7 & v8 ultralytics/ultralytics, ultralytics/docs, pytorch/pytorch, WongKinYiu/yolov7, opencv/opencv
Laravel 10 & 11 laravel/framework, laravel/laravel, laravel/laravel.com, laravel/docs, laravel/breeze, livewire/livewire, php/php-src, php/doc-en,
php/web-php
Angular 16, 17 & 18 angular/angular, angular/components, angular/angular-cli, microsoft/TypeScript
Godot4 godotengine/godot, godotengine/godot-demo-projects, godotengine/godot-docs, godotengine/godot-website, GDQuest/learn-
gdscript, dotnet/csharplang
Table 2: GitHub repositories used for constructing the document collection for every topic in FreshStack.
cumbersome procedure of manual nugget construction [ 49]. LLM-
based nugget generation is explored in the TREC 2024 RAG track3
[49] and in multiple works [ 14,16]. Separately, we experimented
with prompting techniques and found that grading notes style
prompts [ 41] provided parseable and high-quality nuggets in our
experiments.
3.4 Retrieval: Oracle & Inference Setting
A RAG evaluation dataset requires questions, answers, and a corpus
with documents, which helps support facts in the answer. In this
stage, we retrieve a list of highly relevant unjudged documents from
the corpus and construct judgment pools. Since, we are constructing
anevaluation dataset and we have curated answers for questions, we
retrieve a list of top-k documents using two methods: (1) Inference,
relying only on the question and automatic approaches, and (2)
Oracle, relying on the gold answer or list of nuggets, to pool diverse
documents for relevance judgment in the next stage.
Retrieval Settings. We experiment with multiple settings to in-
crease diversity in our judgment pools. First, we experiment with
two techniques in the inference setting :
•GPT-4o Sub-Questions. Decomposing the original question, we
generate a few synthetic sub-questions using GPT-4o, similar to
Rosset et al. [56], concatenated together to retrieve documents.
•GPT-4o Closed Book Answer. We generate a closed-book an-
swer for the original question with GPT-4o, similar to HyDE [ 17],
and use the closed-book answer to retrieve documents.
Next, in the Oracle setting, we also experiment with two techniques:
•Stack Overflow Answer: We use the curated Stack Overflow
answer as the question to retrieve documents.
•Stack Overflow Nuggets: We use the list of GPT-4o generated
nuggets (Section 3.3), concatenated as the question to retrieve
documents.
Retrieval Models. We experiment with five different code and
text-aware retrieval models:
(1)BM25 , a strong lexical baseline in BEIR [ 63]. We utilized the
default BM25 implementation available in Pyserini [39].
3TREC 2024 RAG track: https://trec-rag.github.io/(2)BGE (Gemma-2) [8] a dense retriever model4fine-tuned on the
backbone architecture of Gemma-2 (9B) [ 54] with an embedding
size of 3584 and 8K context length.
(3)E5 Mistral (7B) [70] is a dense retriever model5based fine-
tuned on the backbone of Mistral 7B [ 26] with 32 layers and
embedding size of 4096.
(4)Voyage-large-26is a proprietary and general-purpose embed-
ding model optimized for retrieval quality, with a context length
of 16K tokens and embedding size of 1536.
(5)Fusion , a hybrid retrieval strategy with the four individual
models, by normalizing and summing up the top 100 documents
and their scores from each model.
3.5 Nugget-Level Support Assessment
This is the final stage in the framework. Traditionally, relevance
judgments are conducted on selected pools of retrieved documents,
i.e., a human assessor judges the relevance of the question and
each provided document pair. Due to computational costs, recent
studies experiment with an LLM-as-a-judge (instead of a human
assessor) for conducting relevance judgments in information re-
trieval [ 15,51,64–66]. Questions in existing IR datasets are tradition-
ally short, making it easier to assess question-document relevance.
In contrast, questions in the FreshStack dataset are long and elab-
orate (between 350–500 tokens in length), containing a mixture
of text, code snippets, or outputs, making it challenging to judge
question-document relevance directly [ 13]. For instance, retrieved
documents may answer a major problem presented in the question,
address only part of the question, or contain relevant references
and examples, and we need to translate this into a relevance score.
Nugget-level Support. Instead of relying on traditional relevance
assessments, we simplify the judgment procedure for the LLM and
evaluate whether a document supports information (or contains)
provided by a nugget. A reminder that a nugget highlights an
essential fact of the Stack Overflow question or answer. Judging
document relevance at a nugget level is effective as nuggets are
factual and short information snippets, reducing the ambiguity
often seen during traditional relevance judgments. As we have 𝑛
4BGE Gemma-2: https://huggingface.co/BAAI/bge-multilingual-gemma2
5E5 Mistral 7B: https://huggingface.co/intfloat/e5-mistral-7b-instruct
6Voyage-large-2: https://docs.voyageai.com/docs/embeddings
4

FreshStack: Building Realistic Benchmarks for Evaluating Retrieval on Technical Documents
Topic DomainDataset Count Avg. Length % Containing Code Relevance Judgments
#Queries #Docs #GitHub Avg. N/Q Query Answer Query Answer Rel. Docs/N Rel. Docs/Q
LangChain Machine Learning (ML) 203 49,514 10 3.1 473.4 233.8 83.3% 62.1% 5.7 10.9
Yolo v7 & v8 Computer Vision (CV) 57 27,207 5 3.5 497.1 191.7 70.2% 71.9% 3.9 7.4
Laravel 10 & 11 Backend Development 184 52,351 9 3.0 474.4 155.5 43.5% 51.1% 3.2 6.0
Angular 16, 17 & 18 Front-end Development 129 117,288 4 3.2 463.3 215.1 69.8% 57.4% 4.4 8.7
Godot4 Game Development 99 25,482 6 3.3 350.4 263.0 52.5% 52.5% 2.9 5.9
Table 3: FreshStack dataset statistics; Dataset count measures the number of queries, documents, GitHub repositories, and
average nuggets per query; Avg. length measures the average text lengths (in tokens); % containing code measures the fraction
of queries and answers with code snippets; Relevance judgments measure relevant documents per nugget and query.
nuggets per question, repeating the assessment of 𝑘documents for
every nugget is prohibitive ( 𝑛×𝑘). To reduce computational costs,
we evaluate top-k documents (a maximum 𝑘=20) together with
the list of all nuggets for a question in a single inference call ( 𝑛+𝑘).
We evaluate support judgment with GPT-4o using chain-of-thought
prompting [73].
4 Dataset Statistics & Evaluation
Upon completion of previous stages, we employ two post-processing
steps to ensure high-quality question and answer pairs remain in the
dataset, reducing the overall dataset size. In the first post-processing
stage, we remove unsupported questions, i.e., questions that do not
contain even a single relevant document; this removes, on average,
11.8% of the total questions.7In the next stage, we aggressively filter
by removing questions containing at least one unsupported nugget,
i.e., a nugget that is not supported by any documents, reducing on
average 34.2% of the total questions.
4.1 Dataset Statistics
FreshStack datasets covers five commonly used topics for program-
mers: machine learning, computer vision, backend, front-end, and
game development, all listed in Table 3. Stack Overflow topics such
as LangChain were introduced in 2023, whereas others, like Laravel
or Angular, have questions about the latest versions (e.g., Laravel
10, Laravel 11). Each topic has at least 50 questions, and the corpus
has at least 25K documents sourced from 4–10 GitHub reposito-
ries. The questions are long, containing 350–500 tokens (computed
using GPT-4o tokenizer), and at least 50% of the questions and an-
swers contain code snippets. GPT-4o generates around 3–4 nuggets
for each question. Each nugget is matched to at least 3 relevant
documents with support judgment from GPT-4o, resulting in 5–6
relevant documents per question, across all topics.
4.2 Instance Description
Each FreshStack dataset instance contains the following four com-
ponents, as shown in Figure 1:
•Question & Answer : The title and body (description) of the
Stack Overflow post as the question, with the accepted answer.
The title is a short sentence, and the body contains the detailed
issue with code snippets and/or outputs.
7Future work may include these questions as they are potentially valuable to answer,
and better retrieval systems may be able to find relevant documents.•Nuggets : The list of atomic facts highlighting the essential in-
formation in the Stack Overflow question and answer.
•Document Corpus : The exhaustive list of chunked source doc-
uments (code snippets, text documentation, etc.) compiled from
GitHub repositories.
•Relevance Judgments : Unlike traditional IR benchmarks, such
as BEIR [ 63], which contain question and document-level rele-
vance judgments, FreshStack datasets contain nugget-level rele-
vance judgments for document chunks.
4.3 Retrieval Evaluation Metrics
Information retrieval (IR) evaluation follows the Cranfield para-
digm [ 67], focusing on individual document relevance, independent
of other documents. This is used to construct standard test col-
lections, such as BEIR [ 63] and TREC datasets such as the Deep
Learning (DL) track [ 10–12]. However, diversity in search [ 6,58,75]
penalizes information redundancy within retrieved documents to
enrich information content and improve efficiency. Therefore, we
evaluate retrieval systems using three diversity-focused metrics:
𝛼-nDCG@10 for diversity and relevance, Coverage@20 for nugget
coverage, and Recall@50 for traditional relevancy.
𝛼-nDCG@k. Introduced by Clarke et al . [9], this variant of Nor-
malized Discounted Cumulative Gain (nDCG) measures search
diversification. The 𝛼parameter is a geometric penalization for
redundant documents, i.e., each redundant document achieves a
penalization of×(1−𝛼). Despite the metric being used for different
user intents, we utilize it to ensure document rankings reference
diverse nuggets in the answer.
Coverage@k. The metric introduced in our work measures the
average proportion of the nuggets covered by the top-k retrieved
documents. The mathematical formula is calculated as:
Coverage@k =1
|𝑄|𝑄∑︁
𝑞=1Ð𝑘
𝑖=1Nuggets(𝑑𝑞𝑖)
|Nuggets(𝑞)|(1)
where𝑄contains all questions, Nuggets(𝑑𝑞𝑖)are nuggets supported
by document 𝑑𝑞𝑖and Nuggets(𝑞)are nuggets for question 𝑞.
Recall@k. The standard relevance metric measures the proportion
of relevant documents retrieved within the top-k results, out of all
relevant documents for a given question. A document is judged
relevant if it supports at least one nugget.
5

Thakur et al.
Method ModelLangChain (ML) Yolo v7 & v8 (Vision) Laravel 10 & 11 (Backend) Angular 16, 17 & 18 (Front-end) Godot4 (Game Dev)
𝛼-N@10 Cov@20 Rec@50 𝛼-N@10 Cov@20 Rec@50 𝛼-N@10 Cov@20 Rec@50 𝛼-N@10 Cov@20 Rec@50 𝛼-N@10 Cov@20 Rec@50
Inference Setting : Using a variant of the Stack Overflow question for retrieval of documents within the corpus
GPT-4o
Sub -
QuestionsBM25 0.228 0.495 0.249 0.150 0.427 0.328 0.349 0.656 0.464 0.307 0.666 0.378 0.154 0.326 0.211
BGE (Gemma-2) 0.220 0.561 0.324 0.220 0.554 0.367 0.407 0.727 0.585 0.360 0.707 0.459 0.240 0.532 0.382
E5 Mistral (7B) 0.262 0.613 0.362 0.266 0.593 0.484 0.306 0.643 0.528 0.305 0.617 0.397 0.220 0.461 0.349
Voyage-large-2 0.270 0.563 0.329 0.213 0.526 0.370 0.366 0.687 0.552 0.344 0.69 0.449 0.260 0.594 0.473
Fusion (4 models) 0.322 0.708 0.475 0.305 0.665 0.489 0.478 0.763 0.662 0.428 0.817 0.584 0.290 0.598 0.526
GPT-4o
Closed
Book
AnswerBM25 0.256 0.520 0.273 0.286 0.554 0.431 0.376 0.655 0.495 0.293 0.542 0.332 0.241 0.473 0.349
BGE (Gemma-2) 0.181 0.467 0.263 0.271 0.599 0.473 0.36 0.694 0.539 0.242 0.525 0.338 0.187 0.454 0.358
E5 Mistral (7B) 0.198 0.471 0.277 0.239 0.511 0.364 0.188 0.458 0.384 0.179 0.430 0.267 0.151 0.318 0.237
Voyage-large-2 0.220 0.500 0.301 0.247 0.557 0.495 0.317 0.658 0.524 0.227 0.461 0.338 0.253 0.510 0.454
Fusion (4 models) 0.275 0.630 0.432 0.356 0.686 0.578 0.420 0.738 0.641 0.290 0.582 0.470 0.288 0.538 0.492
Oracle Setting : Using the Stack Overflow answer directly or its variants for retrieval of documents within the corpus
Stack
Overflow
AnswerBM25 0.461 0.726 0.428 0.481 0.756 0.574 0.511 0.774 0.588 0.469 0.751 0.521 0.325 0.565 0.397
BGE (Gemma-2) 0.290 0.625 0.367 0.390 0.815 0.604 0.472 0.814 0.675 0.346 0.690 0.481 0.341 0.718 0.561
E5 Mistral (7B) 0.331 0.671 0.430 0.315 0.683 0.509 0.26 0.634 0.488 0.291 0.570 0.412 0.277 0.546 0.434
Voyage-large-2 0.385 0.700 0.432 0.405 0.703 0.589 0.439 0.791 0.641 0.371 0.680 0.477 0.371 0.626 0.541
Fusion (4 models) 0.484 0.821 0.619 0.546 0.854 0.788 0.564 0.892 0.820 0.470 0.805 0.695 0.449 0.741 0.683
Stack
Overflow
NuggetsBM25 0.467 0.739 0.445 0.519 0.796 0.657 0.540 0.840 0.654 0.485 0.787 0.536 0.428 0.680 0.489
BGE (Gemma-2) 0.308 0.667 0.405 0.461 0.784 0.572 0.448 0.806 0.666 0.393 0.756 0.536 0.335 0.664 0.555
E5 Mistral (7B) 0.323 0.684 0.432 0.437 0.737 0.554 0.287 0.631 0.533 0.346 0.670 0.470 0.292 0.596 0.494
Voyage-large-2 0.419 0.763 0.508 0.430 0.845 0.675 0.409 0.791 0.624 0.406 0.733 0.533 0.353 0.715 0.590
Fusion (4 models) 0.519 0.881 0.655 0.601 0.876 0.825 0.566 0.888 0.818 0.544 0.881 0.756 0.476 0.814 0.719
Table 4: Pooling results achieved by retrieval baselines (including fusion) in inference or oracle settings for constructing
the FreshStack dataset. 𝛼-N@10 denotes 𝛼-nDCG@10, Cov@20 denotes Coverage@20 and Rec@50 denotes Recall@50. Stack
Overflow Answer & Answer Nuggets methods rely on the gold answer for retrieval (oracle setting), whereas, other methods do
not rely on the gold answer for retrieval. Overall, we highlight the best result in bold for each setting.
5 Pooling & Qualitative Evaluation
Using FreshStack, we construct five IR datasets focused on docu-
ment retrieval evaluation with a subset of question-answer pairs.8
In this section, we attempt to answer RQ2 by evaluating techniques
that contribute to judgment pools. Next, we calibrate the quality
of GPT-4o against an expert human annotator. We first evaluate
the retrieval baselines during nugget-level support judgment (or
sampling pools) with inference and oracle settings.
In FreshStack, we are constructing a test evaluation dataset . There-
fore, we can use the Stack Overflow answer or its variants in con-
structing judgment pools, as discussed previously in Section 3.4.
We pool and sample documents from different settings and tech-
niques, similar to how existing question answering datasets are
constructed, such as Natural Questions [ 32] or XOR-TyDI [ 4], which
assess relevance by calculating the answer overlap in the document.
Experimental Settings. We perform retrieval with four tech-
niques and baselines (explained in Section 3.4) and an ensemble
fusion of the top 100 documents, with each model score normalized
and summed up. Evaluation metrics include 𝛼-nDCG@10, Cover-
age@20, and Recall@50. We use GPT-4o with a temperature setting
of 0.19for both the automatic stages. Nugget generation uses a
grading notes prompt with the question and answer, and support
assessment uses chain-of-thought prompting [ 73], judging up to a
maximum of 20 documents simultaneously with a list of nuggets
generated for each question. Finally, we sample and judge the top
20 fusion documents from each technique and setting, including
the question to avoid sampling holes, highlighting the importance
of document diversity in our judgment pools.
8For each question in the FreshStack dataset, the curated answer is present, allowing
us to extend answer evaluation in RAG as potential future work.
9Separately, we tested temperatures of 0.1 and 0.7, observing an identical downstream
retrieval accuracy during FreshStack construction.5.1 Pooling Results
We outline the results achieved on document judgment pools dur-
ing FreshStack dataset construction with four different sampling
techniques and models, including inference and oracle settings.
Table 4 shows results for all five topics. Key takeaways and findings
are discussed below:
Overall Highlights. Table 4 reveals two key findings: (1) Tech-
niques in the Oracle setting significantly outperform techniques
from the inference setting. We observe that both the Stack Over-
flow answer and nuggets techniques help pool documents relevant
to the question, and (2) Fusion outperforms all individual models,
highlighting the value of diversity in model choice, aiding in our
judgment pools.
Inference Setting. GPT-4o Sub-Questions achieves the best pool-
ing results for four topics (except Yolo v7 & v8), showing that
decomposing a Stack Overflow question into smaller sub-questions
helps pool documents relevant to the question. GPT-4o closed-book
answer excels only on Yolo v7 & v8, which we suspect due to pos-
sible data contamination. These overall confirm that FreshStack
contains non-contaminated data, as GPT-4o closed-book answer
(For example, the ChatGPT generated answer) cannot retrieve rele-
vant documents for answering niche questions in the FreshStack
dataset on most topics.
Oracle Setting. Stack Overflow Nuggets achieves the best re-
sults on all topics (except Laravel 10 & 11), showing that GPT-
4o-generated nuggets help pool more documents relevant to the
question, instead of using the Stack Overflow answer. Amongst
the individual models, BM25 achieves the best 𝛼-nDCG@10 on all
topics in both oracle methods (except Godot4), which asserts the
importance of lexical approaches required during the construction
of judgment pools in FreshStack.
6

FreshStack: Building Realistic Benchmarks for Evaluating Retrieval on Technical Documents
Nugget Quality Judgment Quality
Precision 90.1 % Relevant 71.7 %
Recall 96.6 % Partially Relevant 11.7 %
Groundedness 96.4 % Non-Relevant 16.6 %
Table 5: Expert evaluation of GPT-4o nugget quality and
nugget-document relevance judgments on LangChain.
5.2 Qualitative Analysis
In our work, a crucial component is the automatic construction of
nuggets and nugget-level document judgments with GPT-4o. To
assess the LLM quality, we calibrate using an expert annotator (ML
researcher) conducting a qualitative analysis on a small subset of
LangChain, evaluating the quality of generated nuggets and nugget-
document support labels for 60 randomly sampled questions.
5.2.1 Nugget Quality Evaluation. For nugget quality evaluation,
we ask the annotator to answer the following questions (A, B, and
C) after reading the Stack Overflow question, answer, and list of
nuggets. (1) 𝐴- does the nugget produce hallucinated content?
requiring a boolean response (2) 𝐵- is the information provided in
the nugget minor or redundant? also requiring a boolean response.
After annotating all nuggets, we ask (3) 𝐶- how many additional
nuggets are required to cover all key ideas, requiring an integer in
the response.
Evaluation Metrics. After answering all the above questions, we
measure quality by calculating three metrics: (1) precision : whether
nuggets generated are accurate, (2) recall : whether nuggets cover
the key aspects of the answer and (3) groundedness : whether nuggets
produce non-hallucinated content, i.e., outside the scope of the
answer. More formally, we define them as follows:
Precision =|𝑁𝑢𝑔𝑔𝑒𝑡𝑠|−sum(𝐵)
|𝑁𝑢𝑔𝑔𝑒𝑡𝑠|(2)
Recall =|𝑁𝑢𝑔𝑔𝑒𝑡𝑠|−sum(𝐵)
|𝑁𝑢𝑔𝑔𝑒𝑡𝑠|−sum(𝐵)+𝐶(3)
Groundedness =|𝑁𝑢𝑔𝑔𝑒𝑡𝑠|−sum(𝐴)
|𝑁𝑢𝑔𝑔𝑒𝑡𝑠|, (4)
where|𝑁𝑢𝑔𝑔𝑒𝑡𝑠|denotes the count of nuggets for a given question.
Experimental Results. As shown in Table 5, nuggets in the topic
of LangChain achieve above 90% in precision and 96% in recall
and groundedness, indicating GPT-4o can generate high-quality
nuggets required in the FreshStack framework. Most nuggets are
well-grounded, i.e., do not produce hallucinated content (3.6% error),
and cover the key aspects of the answer in terms of recall (3.4%
error). Precision errors are higher (9.9% error), showing nuggets
may contain either minor or repeated information. Within these
errors, the last positioned nugget is not informative in almost 50%
of the time, and either the first or second positioned nugget in the
rest of the error cases.
5.2.2 Relevance Judgment Quality Evaluation. We assess the rel-
evance between nuggets and documents in nugget-level support.
Since judging all documents (including negatives) for each nugget
is cumbersome with a limited budget, we qualitatively evaluate
the relevant pairs. We sample one positive document per question,totaling 60 randomly sampled nugget-document pairs. The anno-
tator labels the relevance on a three-level scale: relevant, partially
relevant, and non-relevant.
Experimental Results. As shown in Table 5, 71.7% of the judged
nuggets and documents are relevant, including an additional 11.7%
which are labeled partially relevant, indicating a high precision
in GPT-4o support judgment. On the other hand, GPT-4o makes
a mistake in judgment for 16.6% of the total questions. This dis-
crepancy arises from several factors: some documents are relevant
to only part of the nugget’s information, leading to mislabeling;
ambiguity within the nugget content can cause misjudgments; and
occasionally, literal grounding of a document in the nugget does
not translate to semantic relevance in answering the question.
6 Main Experiments
In this section, we evaluate leading neural retrievers and rerankers
for the document retrieval setting on the FreshStack dataset, ad-
dressing RQ3 posed in our introduction. All models are evaluated
out-of-the-box, i.e., using only the Stack Overflow question to re-
trieve documents, and do not include any information about the
Stack Overflow answer or nuggets, ensuring a fair assessment.
Experimental Settings. We evaluate the retrieval models used as
baselines during pooling in FreshStack: BM25, BGE (Gemma-2), E5
Mistral 7B, Voyage-large-2, and the ensemble fusion. In addition,
we evaluate the Voyage AI rerank-2 [ 1] as the reranker with a 16K
context length, reranking the top 50 documents retrieved from the
first-stage retrieval system. Metrics used for evaluation are defined
in Section 4.3: 𝛼-nDCG@10, Coverage@20, and Recall@50.
6.1 Model Evaluation Results
Figure 3 shows the results for each retrieval baseline without and
with VoyageAI rerank-2 reranking queries on each topic. Each
model category is color-coded. We plot the 𝛼-nDCG@10, Cover-
age@20, and Recall@50 scores from left to right. Key takeaways
and findings are discussed below:
Accuracy gap between Oracle indicates plenty of headroom.
Techniques from the Oracle setting (using Stack Overflow answers
or nuggets) achieve a substantially higher 𝛼-nDCG@10, Cover-
age@20, and Recall@50 in contrast to all models, including ensem-
ble fusion and reranking with VoyageAI rerank-2. This highlights
the complexity of answering FreshStack questions and demon-
strates the headroom for improvement in existing code-based re-
trieval models to decrease the gap. Further research should explore
techniques to improve models on specialized topics, containing
challenging and long questions in the FreshStack dataset, instead of
overfitting on existing academic IR benchmarks such as BEIR [ 63].
Ensemble fusion outperforms individual models. Individual
retrieval models demonstrate limited success on the FreshStack
dataset; whereas, the ensemble fusion of four retrieval models out-
performs each retrieval model across all metrics ( 𝛼-nDCG@10, Cov-
erage@20, and Recall@50) and all five topics, except 𝛼-nDCG@10
on Godot4. This highlights a crucial point in retrieval systems:
a compound retrieval system [ 78], developed as an ensemble of
retrieval models or something similar, is required to retrieve docu-
ments for niche and challenging topics, as an individual retriever
7

Thakur et al.
BM25
BM25 + Rerank BGE (Gemma-2)BGE + Rerank E5 Mistral (7B)E5 + Rerank
Voyage-large-2Voyage + Rerank Fusion (4 models)Fusion + Rerank0.00.10.20.30.40.5LangChain
-NDCG@10
0.230
0.322
0.216
0.349
0.304
0.385
0.246
0.345
0.337
0.397Best Answer (oracle retrieval): 0.484Best Nuggets (oracle retrieval): 0.519
BM25
BM25 + Rerank BGE (Gemma-2)BGE + Rerank E5 Mistral (7B)E5 + Rerank
Voyage-large-2Voyage + Rerank Fusion (4 models)Fusion + Rerank0.00.20.40.60.8Coverage@20
0.475
0.587
0.548
0.662
0.654
0.701
0.528
0.648
0.700
0.729Best Answer (oracle retrieval): 0.821Best Nuggets (oracle retrieval): 0.881
BM25
BM25 + Rerank BGE (Gemma-2)BGE + Rerank E5 Mistral (7B)E5 + Rerank
Voyage-large-2Voyage + Rerank Fusion (4 models)Fusion + Rerank0.00.10.20.30.40.50.6Recall@50
0.261
0.294
0.337
0.387
0.393
0.439
0.309
0.355
0.477
0.501Best Answer (oracle retrieval): 0.619Best Nuggets (oracle retrieval): 0.655
BM25
BM25 + Rerank BGE (Gemma-2)BGE + Rerank E5 Mistral (7B)E5 + Rerank
Voyage-large-2Voyage + Rerank Fusion (4 models)Fusion + Rerank0.00.10.20.30.40.50.6Yolo v7&v8
-NDCG@10
0.137
0.337
0.258
0.388
0.243
0.364
0.270
0.418
0.304
0.416Best Answer (oracle retrieval): 0.546Best Nuggets (oracle retrieval): 0.601
BM25
BM25 + Rerank BGE (Gemma-2)BGE + Rerank E5 Mistral (7B)E5 + Rerank
Voyage-large-2Voyage + Rerank Fusion (4 models)Fusion + Rerank0.00.20.40.60.8Coverage@20
0.342
0.590
0.547
0.666
0.552
0.628
0.570
0.670
0.627
0.733Best Answer (oracle retrieval): 0.854Best Nuggets (oracle retrieval): 0.876
BM25
BM25 + Rerank BGE (Gemma-2)BGE + Rerank E5 Mistral (7B)E5 + Rerank
Voyage-large-2Voyage + Rerank Fusion (4 models)Fusion + Rerank0.00.20.40.60.8Recall@50
0.337
0.424
0.430
0.459
0.394
0.468
0.453
0.514
0.534
0.592Best Answer (oracle retrieval): 0.788Best Nuggets (oracle retrieval): 0.825
BM25
BM25 + Rerank BGE (Gemma-2)BGE + Rerank E5 Mistral (7B)E5 + Rerank
Voyage-large-2Voyage + Rerank Fusion (4 models)Fusion + Rerank0.00.10.20.30.40.5Laravel 10&11
-NDCG@10
0.319
0.414
0.348
0.306
0.250
0.305
0.345
0.302
0.426
0.319Best Answer (oracle retrieval): 0.564Best Nuggets (oracle retrieval): 0.566
BM25
BM25 + Rerank BGE (Gemma-2)BGE + Rerank E5 Mistral (7B)E5 + Rerank
Voyage-large-2Voyage + Rerank Fusion (4 models)Fusion + Rerank0.00.20.40.60.8Coverage@20
0.602
0.729
0.699
0.646
0.565
0.613
0.701
0.653
0.748
0.671Best Answer (oracle retrieval): 0.892Best Nuggets (oracle retrieval): 0.888
BM25
BM25 + Rerank BGE (Gemma-2)BGE + Rerank E5 Mistral (7B)E5 + Rerank
Voyage-large-2Voyage + Rerank Fusion (4 models)Fusion + Rerank0.00.10.20.30.40.50.60.70.8Recall@50
0.441
0.509
0.574
0.571
0.470
0.510
0.543
0.529
0.646
0.614Best Answer (oracle retrieval): 0.82Best Nuggets (oracle retrieval): 0.818
BM25
BM25 + Rerank BGE (Gemma-2)BGE + Rerank E5 Mistral (7B)E5 + Rerank
Voyage-large-2Voyage + Rerank Fusion (4 models)Fusion + Rerank0.00.10.20.30.40.5Angular 16,17&18
-NDCG@10
0.259
0.346
0.323
0.296
0.262
0.306
0.304
0.300
0.385
0.318Best Answer (oracle retrieval): 0.47Best Nuggets (oracle retrieval): 0.544
BM25
BM25 + Rerank BGE (Gemma-2)BGE + Rerank E5 Mistral (7B)E5 + Rerank
Voyage-large-2Voyage + Rerank Fusion (4 models)Fusion + Rerank0.00.20.40.60.8Coverage@20
0.551
0.647
0.571
0.595
0.548
0.601
0.625
0.600
0.719
0.641Best Answer (oracle retrieval): 0.805Best Nuggets (oracle retrieval): 0.881
BM25
BM25 + Rerank BGE (Gemma-2)BGE + Rerank E5 Mistral (7B)E5 + Rerank
Voyage-large-2Voyage + Rerank Fusion (4 models)Fusion + Rerank0.00.10.20.30.40.50.60.7Recall@50
0.340
0.385
0.378
0.387
0.368
0.375
0.427
0.414
0.532
0.488Best Answer (oracle retrieval): 0.695Best Nuggets (oracle retrieval): 0.756
BM25
BM25 + Rerank BGE (Gemma-2)BGE + Rerank E5 Mistral (7B)E5 + Rerank
Voyage-large-2Voyage + Rerank Fusion (4 models)Fusion + Rerank0.00.10.20.30.4Godot4
-NDCG@10
0.144
0.251
0.199
0.324
0.217
0.315
0.282
0.342
0.265
0.340Best Answer (oracle retrieval): 0.449Best Nuggets (oracle retrieval): 0.476
BM25
BM25 + Rerank BGE (Gemma-2)BGE + Rerank E5 Mistral (7B)E5 + Rerank
Voyage-large-2Voyage + Rerank Fusion (4 models)Fusion + Rerank0.00.10.20.30.40.50.60.70.8Coverage@20
0.268
0.407
0.479
0.576
0.444
0.566
0.522
0.598
0.550
0.627Best Answer (oracle retrieval): 0.741Best Nuggets (oracle retrieval): 0.814
BM25
BM25 + Rerank BGE (Gemma-2)BGE + Rerank E5 Mistral (7B)E5 + Rerank
Voyage-large-2Voyage + Rerank Fusion (4 models)Fusion + Rerank0.00.10.20.30.40.50.60.7Recall@50
0.200
0.244
0.419
0.471
0.359
0.426
0.458
0.511
0.505
0.545Best Answer (oracle retrieval): 0.683Best Nuggets (oracle retrieval): 0.719
Figure 3: Main experiment results with retrieval and reranker baselines (including fusion) on five topics. Best scores in the
Oracle setting are taken from Table 4 (Stack Overflow Nuggets in blue, Stack Overflow Answer in red). The reranker is the
Voyage AI rerank-2 model [1]. Plots show 𝛼-nDCG-10, Coverage20, and Recall@50 scores for each topic (top to bottom).
is not sufficient, at present. While the ensemble fusion of retrieval
systems achieves the best scores on the FreshStack dataset, their
inefficiency at inference time adds up due to individual model in-
ference, requiring alternatives.
Opportunities to improve reranking. When using weak first-
stage retrieval, neural rerankers typically improve document rank-
ing [63], although it has been recently shown that this is not always
the case when strong first-stage retrieval is used [ 24,79]. Consis-
tent with these recent observations, reranking provides benefits
over BM25 for all topics in the FreshStack dataset. However, for our
dense retrievers, reranking provides a clear benefit on some but not
all datasets. Specifically, while the reranker enhances 𝛼-nDCG@10,Coverage@20, and Recall@50 for LangChain, Yolo v7 & v8, and
Godot4, it reduces those metrics on Laravel 10 & 11 and Angular
16, 17 & 18 for the dense retrievers. We suspect the reranker is
better in certain programming languages such as Python, and we
keep it as future work to understand the limitations of the neural
reranker [ 24]. Separately, we also tried GPT-4o-mini [ 47] naively
as a zero-shot listwise reranker following Sun et al . [61] and found
preliminary results underperformed heavily. We suspect this was
due to formatting challenges when concatenating multiple long
documents, and we leave the exploration of LLMs as rerankers
(perhaps with fine-tuning) to future work.
8

FreshStack: Building Realistic Benchmarks for Evaluating Retrieval on Technical Documents
7 Discussion
FreshStack is a general framework for building challenging IR
datasets. We apply the framework to community-sourced ques-
tions (with curated answers) and documents sourced from GitHub
repositories. The framework is adaptable to other domains like
Stack Exchange or internal forums. While we benchmark a few
key retrieval models, future work would continue to benchmark
code-focused retrieval models like voyage-3 [ 2], Code-T5+ [ 71],
CodeRankEmbed [ 62], and Jina-Code-v2 [ 19], and rerankers such
as CodeRankLLM [62].
Answer Evaluation. We focused on the evaluation of the retrieval
setting primarily due to two reasons: (1) Existing RAG datasets
evaluate retrieval using relevance criteria only, however, we eval-
uated models based on both diversity and relevance criteria, and
(2) a crucial step in FreshStack is sourcing and building a docu-
ment corpus and developing a general framework for high-quality
pools and automatic judgments, which we can evaluate better in
the retrieval setting. As we have curated Stack Overflow answers
available, future work can extend on evaluating the quality of LLM-
based answer generation with metrics such as nugget-based recall,
which calculates how many nuggets are supported within a sys-
tem’s response. Given the high difficulty of retrieval, validated with
results observed in Section 6, we anticipate answer generation by
LLM to be similarly challenging.
Benchmark Contamination. The FreshStack dataset builds on
Stack Overflow data and is susceptible to future data contamination.
To address this, FreshStack is intentionally designed to be generic
and extensible to new domains and languages. To mitigate data
contamination, the framework can add newer questions in existing
topics, retire old and contaminated topics, and add newer topics that
develop in the future. The relevance of FreshStack in the community
relies on a continued commitment to keeping it updated in the
upcoming years.
8 Conclusion
The emergence of RAG has improved modern retrieval systems by
allowing real-time data incorporation into LLMs. However, existing
IR and RAG benchmarks that measure retrieval quality are outdated.
In this work, we introduce a holistic framework, FreshStack, to con-
struct challenging datasets to evaluate retrieval systems realistically.
We source real user questions and answers from Stack Overflow and
build a document corpus using technical documents from public
GitHub repositories. Using FreshStack, we construct datasets on
five niche topics and evaluate four frontier retrieval models and
a reranker model in the document retrieval setting. The accuracy
gap observed between the retrieval models and approaches in the
Oracle setting indicates plenty of headroom for improvement, and
we identify cases that may motivate future research in reranking.
We hope FreshStack will encourage the community to build more
challenging and realistic IR and RAG datasets in the future.
Acknowledgments
We thank Sean Kulinski and Alexander Trott for helping us set up
the grading notes prompt required in nugget generation. We also
thank Jacob Portes, Max Marion, Matei Zaharia, and others from
Databricks who provided feedback at the early stages of the project.References
[1] Voyage AI. 2024. rerank-2 & rerank-2-lite: the next generation of Voyage multilin-
gual rerankers . https://blog.voyageai.com/2024/09/30/rerank-2/
[2]Voyage AI. 2024. voyage-3 & voyage-3-lite: A new generation of small yet
mighty general-purpose embedding models . https://blog.voyageai.com/2024/
09/18/voyage-3/
[3]Negar Arabzadeh and Charles L. A. Clarke. 2024. A Comparison of Methods
for Evaluating Generative IR. CoRR abs/2404.04044 (2024). doi:10.48550/ARXIV.
2404.04044 arXiv:2404.04044
[4]Akari Asai, Jungo Kasai, Jonathan Clark, Kenton Lee, Eunsol Choi, and Han-
naneh Hajishirzi. 2021. XOR QA: Cross-lingual Open-Retrieval Question An-
swering. In Proceedings of the 2021 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language Technolo-
gies, Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-
Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and
Yichao Zhou (Eds.). Association for Computational Linguistics, Online, 547–
564. doi:10.18653/v1/2021.naacl-main.46
[5] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Ruther-
ford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan
Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman
Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer,
Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero,
Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent Sifre. 2022. Improving
Language Models by Retrieving from Trillions of Tokens. In International Con-
ference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland,
USA (Proceedings of Machine Learning Research, Vol. 162) , Kamalika Chaudhuri,
Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato (Eds.).
PMLR, 2206–2240. https://proceedings.mlr.press/v162/borgeaud22a.html
[6]Jaime Carbonell and Jade Goldstein. 1998. The use of MMR, diversity-based
reranking for reordering documents and producing summaries. In Proceedings
of the 21st Annual International ACM SIGIR Conference on Research and Develop-
ment in Information Retrieval (Melbourne, Australia) (SIGIR ’98) . Association for
Computing Machinery, New York, NY, USA, 335–336. doi:10.1145/290941.291025
[7]Jingyi Chen, Songqiang Chen, Jialun Cao, Jiasi Shen, and Shing-Chi Cheung.
2025. When LLMs Meet API Documentation: Can Retrieval Augmentation Aid
Code Generation Just as It Helps Developers? CoRR abs/2503.15231 (2025).
doi:10.48550/ARXIV.2503.15231 arXiv:2503.15231
[8] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024.
BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text
Embeddings Through Self-Knowledge Distillation. CoRR abs/2402.03216 (2024).
doi:10.48550/ARXIV.2402.03216 arXiv:2402.03216
[9] Charles L.A. Clarke, Maheedhar Kolla, Gordon V. Cormack, Olga Vechtomova,
Azin Ashkan, Stefan Büttcher, and Ian MacKinnon. 2008. Novelty and diversity
in information retrieval evaluation. In Proceedings of the 31st Annual International
ACM SIGIR Conference on Research and Development in Information Retrieval
(Singapore, Singapore) (SIGIR ’08) . Association for Computing Machinery, New
York, NY, USA, 659–666. doi:10.1145/1390334.1390446
[10] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Jimmy Lin.
2022. Overview of the TREC 2021 deep learning track. In Text REtrieval Conference
(TREC) . NIST, TREC. https://www.microsoft.com/en-us/research/publication/
overview-of-the-trec-2021-deep-learning-track/
[11] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, Jimmy Lin, Ellen M.
Voorhees, and Ian Soboroff. 2023. Overview of the TREC 2022 deep learning track.
InText REtrieval Conference (TREC) . NIST, TREC. https://www.microsoft.com/en-
us/research/publication/overview-of-the-trec-2022-deep-learning-track/
[12] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Hossein A. Rahmani, Daniel Cam-
pos, Jimmy Lin, Ellen M. Voorhees, and Ian Soboroff. 2024. Overview of the
TREC 2023 Deep Learning Track. In Text REtrieval Conference (TREC) . NIST,
TREC. https://www.microsoft.com/en-us/research/publication/overview-of-
the-trec-2023-deep-learning-track/
[13] Tadele T. Damessie, Falk Scholer, and J. Shane Culpepper. 2016. The Influence
of Topic Difficulty, Relevance Level, and Document Ordering on Relevance
Judging. In Proceedings of the 21st Australasian Document Computing Symposium
(Caulfield, VIC, Australia) (ADCS ’16) . Association for Computing Machinery,
New York, NY, USA, 41–48. doi:10.1145/3015022.3015033
[14] Laura Dietz. 2024. A Workbench for Autograding Retrieve/Generate Systems.
InProceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval, SIGIR 2024, Washington DC, USA, July
14-18, 2024 , Grace Hui Yang, Hongning Wang, Sam Han, Claudia Hauff, Guido
Zuccon, and Yi Zhang (Eds.). ACM, 1963–1972. doi:10.1145/3626772.3657871
[15] Guglielmo Faggioli, Laura Dietz, Charles L. A. Clarke, Gianluca Demartini,
Matthias Hagen, Claudia Hauff, Noriko Kando, Evangelos Kanoulas, Martin
Potthast, Benno Stein, and Henning Wachsmuth. 2023. Perspectives on Large
Language Models for Relevance Judgment. In Proceedings of the 2023 ACM SI-
GIR International Conference on Theory of Information Retrieval (Taipei, Taiwan)
9

Thakur et al.
(ICTIR ’23) . Association for Computing Machinery, New York, NY, USA, 39–50.
doi:10.1145/3578337.3605136
[16] Naghmeh Farzi and Laura Dietz. 2024. Pencils Down! Automatic Rubric-based
Evaluation of Retrieve/Generate Systems. In Proceedings of the 2024 ACM SIGIR
International Conference on Theory of Information Retrieval, ICTIR 2024, Washing-
ton, DC, USA, 13 July 2024 , Harrie Oosterhuis, Hannah Bast, and Chenyan Xiong
(Eds.). ACM, 175–184. doi:10.1145/3664190.3672511
[17] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise Zero-Shot
Dense Retrieval without Relevance Labels. In Proceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
ACL 2023, Toronto, Canada, July 9-14, 2023 , Anna Rogers, Jordan L. Boyd-Graber,
and Naoaki Okazaki (Eds.). Association for Computational Linguistics, 1762–1777.
doi:10.18653/V1/2023.ACL-LONG.99
[18] Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023. Enabling Large
Language Models to Generate Text with Citations. In Proceedings of the 2023
Conference on Empirical Methods in Natural Language Processing, EMNLP 2023,
Singapore, December 6-10, 2023 , Houda Bouamor, Juan Pino, and Kalika Bali (Eds.).
Association for Computational Linguistics, 6465–6488. doi:10.18653/V1/2023.
EMNLP-MAIN.398
[19] Michael Günther, Jackmin Ong, Isabelle Mohr, Alaeddine Abdessalem, Tanguy
Abel, Mohammad Kalim Akram, Susana Guzman, Georgios Mastrapas, Saba
Sturua, Bo Wang, Maximilian Werk, Nan Wang, and Han Xiao. 2023. Jina Em-
beddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents.
CoRR abs/2310.19923 (2023). doi:10.48550/ARXIV.2310.19923 arXiv:2310.19923
[20] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang.
2020. Retrieval Augmented Language Model Pre-Training. In Proceedings of
the 37th International Conference on Machine Learning, ICML 2020, 13-18 July
2020, Virtual Event (Proceedings of Machine Learning Research, Vol. 119) . PMLR,
3929–3938. http://proceedings.mlr.press/v119/guu20a.html
[21] Doris Hoogeveen, Karin M. Verspoor, and Timothy Baldwin. 2015. CQADup-
Stack: A Benchmark Data Set for Community Question-Answering Research.
InProceedings of the 20th Australasian Document Computing Symposium (Parra-
matta, NSW, Australia) (ADCS ’15) . Association for Computing Machinery, New
York, NY, USA, Article 3, 8 pages. doi:10.1145/2838931.2838934
[22] Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc
Brockschmidt. 2019. CodeSearchNet Challenge: Evaluating the State of Semantic
Code Search. CoRR abs/1909.09436 (2019). arXiv:1909.09436 http://arxiv.org/
abs/1909.09436
[23] Gautier Izacard and Edouard Grave. 2021. Leveraging Passage Retrieval with
Generative Models for Open Domain Question Answering. In Proceedings of the
16th Conference of the European Chapter of the Association for Computational
Linguistics: Main Volume, EACL 2021, Online, April 19 - 23, 2021 , Paola Merlo, Jörg
Tiedemann, and Reut Tsarfaty (Eds.). Association for Computational Linguistics,
874–880. doi:10.18653/V1/2021.EACL-MAIN.74
[24] Mathew Jacob, Erik Lindgren, Matei Zaharia, Michael Carbin, Omar Khattab,
and Andrew Drozdov. 2024. Drowning in Documents: Consequences of Scaling
Reranker Inference. CoRR abs/2411.11767 (2024). doi:10.48550/ARXIV.2411.11767
arXiv:2411.11767
[25] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang,
Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. 2024. Live-
CodeBench: Holistic and Contamination Free Evaluation of Large Language
Models for Code. CoRR abs/2403.07974 (2024). doi:10.48550/ARXIV.2403.07974
arXiv:2403.07974
[26] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, De-
vendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel,
Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux,
Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix,
and William El Sayed. 2023. Mistral 7B. CoRR abs/2310.06825 (2023). doi:10.
48550/ARXIV.2310.06825 arXiv:2310.06825
[27] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir
Press, and Karthik R. Narasimhan. 2024. SWE-bench: Can Language Models
Resolve Real-world Github Issues?. In The Twelfth International Conference on
Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024 . OpenRe-
view.net. https://openreview.net/forum?id=VTF8yNQM66
[28] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA:
A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehen-
sion. In Proceedings of the 55th Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers) , Regina Barzilay and Min-Yen Kan
(Eds.). Association for Computational Linguistics, Vancouver, Canada, 1601–1611.
doi:10.18653/v1/P17-1147
[29] Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel.
2023. Large language models struggle to learn long-tail knowledge. In Proceedings
of the 40th International Conference on Machine Learning (Honolulu, Hawaii, USA)
(ICML’23) . JMLR.org, Article 641, 12 pages.
[30] Jungo Kasai, Keisuke Sakaguchi, yoichi takahashi, Ronan Le Bras, Akari Asai,
Xinyan Velocity Yu, Dragomir Radev, Noah A. Smith, Yejin Choi, and Kentaro
Inui. 2023. RealTime QA: What’s the Answer Right Now?. In Thirty-seventhConference on Neural Information Processing Systems Datasets and Benchmarks
Track . https://openreview.net/forum?id=HfKOIPCvsv
[31] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike
Lewis. 2020. Generalization through Memorization: Nearest Neighbor Language
Models. In 8th International Conference on Learning Representations, ICLR 2020,
Addis Ababa, Ethiopia, April 26-30, 2020 . OpenReview.net. https://openreview.
net/forum?id=HklBjCEKvH
[32] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins,
Ankur P. Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob De-
vlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei
Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural
Questions: a Benchmark for Question Answering Research. Trans. Assoc. Comput.
Linguistics 7 (2019), 452–466. doi:10.1162/TACL_A_00276
[33] Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural In-
formation Processing Systems 33: Annual Conference on Neural Information
Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual , Hugo
Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and
Hsuan-Tien Lin (Eds.). https://proceedings.neurips.cc/paper/2020/hash/
6b493230205f780e1bc26945df7481e5-Abstract.html
[34] Hongyu Li, Seohyun Kim, and Satish Chandra. 2019. Neural Code Search
Evaluation Dataset. CoRR abs/1908.09804 (2019). arXiv:1908.09804 http:
//arxiv.org/abs/1908.09804
[35] Xiangyang Li, Kuicai Dong, Yi Quan Lee, Wei Xia, Yichun Yin, Hao Zhang,
Yong Liu, Yasheng Wang, and Ruiming Tang. 2024. CoIR: A Comprehensive
Benchmark for Code Information Retrieval Models. CoRR abs/2407.02883 (2024).
doi:10.48550/ARXIV.2407.02883 arXiv:2407.02883
[36] Yifei Li, Xiang Yue, Zeyi Liao, and Huan Sun. 2024. AttributionBench: How
Hard is Automatic Attribution Evaluation?. In Findings of the Association for
Computational Linguistics: ACL 2024 , Lun-Wei Ku, Andre Martins, and Vivek
Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand,
14919–14935. doi:10.18653/v1/2024.findings-acl.886
[37] Jimmy Lin and Dina Demner-Fushman. 2005. Automatically Evaluating Answers
to Definition Questions. In Proceedings of Human Language Technology Conference
and Conference on Empirical Methods in Natural Language Processing , Raymond
Mooney, Chris Brew, Lee-Feng Chien, and Katrin Kirchhoff (Eds.). Association
for Computational Linguistics, Vancouver, British Columbia, Canada, 931–938.
https://aclanthology.org/H05-1117
[38] Jimmy Lin and Dina Demner-Fushman. 2006. Methods for Automatically Evalu-
ating Answers to Complex Questions. Information Retrieval 9, 5 (2006).
[39] Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong Yang, Ronak Pradeep,
and Rodrigo Frassetto Nogueira. 2021. Pyserini: A Python Toolkit for Repro-
ducible Information Retrieval Research with Sparse and Dense Representations.
InSIGIR ’21: The 44th International ACM SIGIR Conference on Research and Devel-
opment in Information Retrieval, Virtual Event, Canada, July 11-15, 2021 , Fernando
Diaz, Chirag Shah, Torsten Suel, Pablo Castells, Rosie Jones, and Tetsuya Sakai
(Eds.). ACM, 2356–2362. doi:10.1145/3404835.3463238
[40] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
Fabio Petroni, and Percy Liang. 2024. Lost in the Middle: How Language Models
Use Long Contexts. Transactions of the Association for Computational Linguistics
12 (2024), 157–173. doi:10.1162/tacl_a_00638
[41] Yi Liu, Matei Zaharia, and Ritendra Datta. 2024. Enhancing LLM-as-a-Judge with
Grading Notes . https://www.databricks.com/blog/enhancing-llm-as-a-judge-
with-grading-notes
[42] Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio
Blanco, Colin Clement, Dawn Drain, Daxin Jiang, Duyu Tang, Ge Li, Lidong Zhou,
Linjun Shou, Long Zhou, Michele Tufano, MING GONG, Ming Zhou, Nan Duan,
Neel Sundaresan, Shao Kun Deng, Shengyu Fu, and Shujie LIU. 2021. CodeXGLUE:
A Machine Learning Benchmark Dataset for Code Understanding and Generation.
InThirty-fifth Conference on Neural Information Processing Systems Datasets and
Benchmarks Track (Round 1) . https://openreview.net/forum?id=6lE4dQXaUcb
[43] James Mayfield, Eugene Yang, Dawn J. Lawrie, Sean MacAvaney, Paul McNamee,
Douglas W. Oard, Luca Soldaini, Ian Soboroff, Orion Weller, Efsun Selin Kayi, Kate
Sanders, Marc Mason, and Noah Hibbler. 2024. On the Evaluation of Machine-
Generated Reports. In Proceedings of the 47th International ACM SIGIR Conference
on Research and Development in Information Retrieval, SIGIR 2024, Washington
DC, USA, July 14-18, 2024 , Grace Hui Yang, Hongning Wang, Sam Han, Claudia
Hauff, Guido Zuccon, and Yi Zhang (Eds.). ACM, 1904–1915. doi:10.1145/3626772.
3657846
[44] Ani Nenkova and Rebecca Passonneau. 2004. Evaluating Content Selection in
Summarization: The Pyramid Method. In Proceedings of the Human Language
Technology Conference of the North American Chapter of the Association for Compu-
tational Linguistics: HLT-NAACL 2004 . Association for Computational Linguistics,
Boston, Massachusetts, USA, 145–152. https://aclanthology.org/N04-1019
[45] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan
Majumder, and Li Deng. 2016. MS MARCO: A Human Generated MAchine
10

FreshStack: Building Realistic Benchmarks for Evaluating Retrieval on Technical Documents
Reading COmprehension Dataset. In Proceedings of the Workshop on Cogni-
tive Computation: Integrating neural and symbolic approaches 2016 co-located
with the 30th Annual Conference on Neural Information Processing Systems (NIPS
2016), Barcelona, Spain, December 9, 2016 (CEUR Workshop Proceedings, Vol. 1773) ,
Tarek Richard Besold, Antoine Bordes, Artur S. d’Avila Garcez, and Greg Wayne
(Eds.). CEUR-WS.org. https://ceur-ws.org/Vol-1773/CoCoNIPS_2016_paper9.pdf
[46] Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, KaShun Shum, Randy Zhong,
Juntong Song, and Tong Zhang. 2024. RAGTruth: A Hallucination Corpus for
Developing Trustworthy Retrieval-Augmented Language Models. In Proceedings
of the 62nd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers) , Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.).
Association for Computational Linguistics, Bangkok, Thailand, 10862–10878.
doi:10.18653/v1/2024.acl-long.585
[47] OpenAI. 2024. Hello GPT-4o . https://openai.com/index/hello-gpt-4o/
[48] Virgil Pavlu, Shahzad Rajput, Peter B. Golbus, and Javed A. Aslam. 2012. IR
system evaluation using nugget-based test collections. In Proceedings of the
Fifth ACM International Conference on Web Search and Data Mining (Seattle,
Washington, USA) (WSDM ’12) . Association for Computing Machinery, New
York, NY, USA, 393–402. doi:10.1145/2124295.2124343
[49] Ronak Pradeep, Nandan Thakur, Sahel Sharifymoghaddam, Eric Zhang, Ryan
Nguyen, Daniel Campos, Nick Craswell, and Jimmy Lin. 2024. Ragnarök: A
Reusable RAG Framework and Baselines for TREC 2024 Retrieval-Augmented
Generation Track. CoRR abs/2406.16828 (2024). doi:10.48550/ARXIV.2406.16828
arXiv:2406.16828
[50] Yuan Pu, Zhuolun He, Tairu Qiu, Haoyuan Wu, and Bei Yu. 2025. Customized Re-
trieval Augmented Generation and Benchmarking for EDA Tool Documentation
QA. In Proceedings of the 43rd IEEE/ACM International Conference on Computer-
Aided Design (Newark Liberty International Airport Marriott, New York, NY,
USA) (ICCAD ’24) . Association for Computing Machinery, New York, NY, USA,
Article 116, 9 pages. doi:10.1145/3676536.3676730
[51] Hossein A. Rahmani, Emine Yilmaz, Nick Craswell, Bhaskar Mitra, Paul Thomas,
Charles L. A. Clarke, Mohammad Aliannejadi, Clemencia Siro, and Guglielmo
Faggioli. 2024. LLMJudge: LLMs for Relevance Judgments. In Proceedings of
The First Workshop on Large Language Models for Evaluation in Information
Retrieval (LLM4Eval 2024) co-located with 10th International Conference on Online
Publishing (SIGIR 2024), Washington D.C., USA, July 18, 2024 (CEUR Workshop
Proceedings, Vol. 3752) , Clemencia Siro, Mohammad Aliannejadi, Hossein A.
Rahmani, Nick Craswell, Charles L. A. Clarke, Guglielmo Faggioli, Bhaskar
Mitra, Paul Thomas, and Emine Yilmaz (Eds.). CEUR-WS.org, 1–3. https://ceur-
ws.org/Vol-3752/paper8.pdf
[52] Vatsal Raina and Mark Gales. 2024. Question-Based Retrieval using Atomic
Units for Enterprise RAG. In Proceedings of the Seventh Fact Extraction and VERi-
fication Workshop (FEVER) , Michael Schlichtkrull, Yulong Chen, Chenxi White-
house, Zhenyun Deng, Mubashara Akhtar, Rami Aly, Zhijiang Guo, Christos
Christodoulopoulos, Oana Cocarascu, Arpit Mittal, James Thorne, and Andreas
Vlachos (Eds.). Association for Computational Linguistics, Miami, Florida, USA,
219–233. doi:10.18653/v1/2024.fever-1.25
[53] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin
Leyton-Brown, and Yoav Shoham. 2023. In-Context Retrieval-Augmented Lan-
guage Models. Transactions of the Association for Computational Linguistics 11
(2023), 1316–1331. doi:10.1162/tacl_a_00605
[54] Morgane Rivière, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya
Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre
Ramé, Johan Ferret, Peter Liu, Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela
Ramos, Ravin Kumar, Charline Le Lan, Sammy Jerome, Anton Tsitsulin, Nino
Vieillard, Piotr Stanczyk, Sertan Girgin, Nikola Momchev, Matt Hoffman, Shan-
tanu Thakoor, Jean-Bastien Grill, Behnam Neyshabur, Olivier Bachem, Alanna
Walton, Aliaksei Severyn, Alicia Parrish, Aliya Ahmad, Allen Hutchison, Alvin
Abdagic, Amanda Carl, Amy Shen, Andy Brock, Andy Coenen, Anthony Laforge,
Antonia Paterson, Ben Bastian, Bilal Piot, Bo Wu, Brandon Royal, Charlie Chen,
Chintu Kumar, Chris Perry, Chris Welty, Christopher A. Choquette-Choo, Danila
Sinopalnikov, David Weinberger, Dimple Vijaykumar, Dominika Rogozinska,
Dustin Herbison, Elisa Bandy, Emma Wang, Eric Noland, Erica Moreira, Evan
Senter, Evgenii Eltyshev, Francesco Visin, Gabriel Rasskin, Gary Wei, Glenn
Cameron, Gus Martins, Hadi Hashemi, Hanna Klimczak-Plucinska, Harleen Ba-
tra, Harsh Dhand, Ivan Nardini, Jacinda Mein, Jack Zhou, James Svensson, Jeff
Stanway, Jetha Chan, Jin Peng Zhou, Joana Carrasqueira, Joana Iljazi, Jocelyn
Becker, Joe Fernandez, Joost van Amersfoort, Josh Gordon, Josh Lipschultz, Josh
Newlan, Ju-yeong Ji, Kareem Mohamed, Kartikeya Badola, Kat Black, Katie Milli-
can, Keelin McDonell, Kelvin Nguyen, Kiranbir Sodhia, Kish Greene, Lars Lowe
Sjösund, Lauren Usui, Laurent Sifre, Lena Heuermann, Leticia Lago, and Lilly
McNealus. 2024. Gemma 2: Improving Open Language Models at a Practical Size.
CoRR abs/2408.00118 (2024). doi:10.48550/ARXIV.2408.00118 arXiv:2408.00118
[55] Sara Rosenthal, Avirup Sil, Radu Florian, and Salim Roukos. 2025. CLAPnq:
Cohesive Long-form Answers from Passages in Natural Questions for RAG
systems. Transactions of the Association for Computational Linguistics 13 (2025),
53–72. doi:10.1162/tacl_a_00729
[56] Corby Rosset, Ho-Lam Chung, Guanghui Qin, Ethan C. Chau, Zhuo Feng, Ahmed
Awadallah, Jennifer Neville, and Nikhil Rao. 2024. Researchy Questions: ADataset of Multi-Perspective, Decompositional Questions for LLM Web Agents.
CoRR abs/2402.17896 (2024). doi:10.48550/ARXIV.2402.17896 arXiv:2402.17896
[57] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei
Zaharia. 2022. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late
Interaction. In Proceedings of the 2022 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language Technologies,
NAACL 2022, Seattle, WA, United States, July 10-15, 2022 , Marine Carpuat, Marie-
Catherine de Marneffe, and Iván Vladimir Meza Ruíz (Eds.). Association for
Computational Linguistics, 3715–3734. doi:10.18653/V1/2022.NAACL-MAIN.272
[58] Rodrygo L. T. Santos, Craig Macdonald, and Iadh Ounis. 2015. Search Result
Diversification. Foundations and Trends ®in Information Retrieval 9, 1 (2015),
1–90. doi:10.1561/1500000040
[59] Yijia Shao, Yucheng Jiang, Theodore Kanell, Peter Xu, Omar Khattab, and Mon-
ica Lam. 2024. Assisting in Writing Wikipedia-like Articles From Scratch with
Large Language Models. In Proceedings of the 2024 Conference of the North Amer-
ican Chapter of the Association for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers) , Kevin Duh, Helena Gomez, and Steven
Bethard (Eds.). Association for Computational Linguistics, Mexico City, Mexico,
6252–6278. doi:10.18653/v1/2024.naacl-long.347
[60] Sumit Soman and Sujoy Roychowdhury. 2024. Observations on Building RAG
Systems for Technical Documents. In The Second Tiny Papers Track at ICLR
2024, Tiny Papers @ ICLR 2024, Vienna, Austria, May 11, 2024 . OpenReview.net.
https://openreview.net/forum?id=RFujq4HoV4
[61] Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin
Chen, Dawei Yin, and Zhaochun Ren. 2023. Is ChatGPT Good at Search? Inves-
tigating Large Language Models as Re-Ranking Agents. In Proceedings of the
2023 Conference on Empirical Methods in Natural Language Processing , Houda
Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Lin-
guistics, Singapore, 14918–14937. doi:10.18653/v1/2023.emnlp-main.923
[62] Tarun Suresh, Revanth Gangi Reddy, Yifei Xu, Zach Nussbaum, Andriy Mulyar,
Brandon Duderstadt, and Heng Ji. 2024. CoRNStack: High-Quality Contrastive
Data for Better Code Ranking. CoRR abs/2412.01007 (2024). doi:10.48550/ARXIV.
2412.01007 arXiv:2412.01007
[63] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna
Gurevych. 2021. BEIR: A Heterogeneous Benchmark for Zero-shot Evalua-
tion of Information Retrieval Models. In Proceedings of the Neural Information
Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and
Benchmarks 2021, December 2021, virtual , Joaquin Vanschoren and Sai-Kit Yeung
(Eds.). https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/
65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html
[64] Paul Thomas, Seth Spielman, Nick Craswell, and Bhaskar Mitra. 2024. Large
Language Models can Accurately Predict Searcher Preferences. In Proceedings
of the 47th International ACM SIGIR Conference on Research and Development in
Information Retrieval, SIGIR 2024, Washington DC, USA, July 14-18, 2024 , Grace Hui
Yang, Hongning Wang, Sam Han, Claudia Hauff, Guido Zuccon, and Yi Zhang
(Eds.). ACM, 1930–1940. doi:10.1145/3626772.3657707
[65] Shivani Upadhyay, Ronak Pradeep, Nandan Thakur, Daniel Campos, Nick
Craswell, Ian Soboroff, Hoa Trang Dang, and Jimmy Lin. 2024. A Large-Scale
Study of Relevance Assessments with Large Language Models: An Initial Look.
CoRR abs/2411.08275 (2024). doi:10.48550/ARXIV.2411.08275 arXiv:2411.08275
[66] Shivani Upadhyay, Ronak Pradeep, Nandan Thakur, Nick Craswell, and Jimmy
Lin. 2024. UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing
RELevance Assessor. CoRR abs/2406.06519 (2024). doi:10.48550/ARXIV.2406.
06519 arXiv:2406.06519
[67] Ellen Voorhees. 2009. I Come Not To Bury Cranfield, but to Praise It. HCIR 2009:
Bridging Human Computer Interaction and Information Retrieval, Washington
DC, -1.
[68] Ellen M. Voorhees. 2003. Overview of the TREC 2003 Question Answering Track.
InProceedings of the Twelfth Text REtrieval Conference (TREC 2003) . Gaithersburg,
Maryland.
[69] Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar,
Yun-Hsuan Sung, Denny Zhou, Quoc Le, and Thang Luong. 2024. FreshLLMs: Re-
freshing Large Language Models with Search Engine Augmentation. In Findings
of the Association for Computational Linguistics: ACL 2024 , Lun-Wei Ku, Andre
Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics,
Bangkok, Thailand, 13697–13720. doi:10.18653/v1/2024.findings-acl.813
[70] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and
Furu Wei. 2024. Improving Text Embeddings with Large Language Models. In
Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16,
2024, Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for
Computational Linguistics, 11897–11916. doi:10.18653/V1/2024.ACL-LONG.642
[71] Yue Wang, Hung Le, Akhilesh Gotmare, Nghi D. Q. Bui, Junnan Li, and Steven
C. H. Hoi. 2023. CodeT5+: Open Code Large Language Models for Code Un-
derstanding and Generation. In Proceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-
10, 2023 , Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for
11

Thakur et al.
Computational Linguistics, 1069–1088. doi:10.18653/V1/2023.EMNLP-MAIN.68
[72] Zora Zhiruo Wang, Akari Asai, Xinyan Velocity Yu, Frank F. Xu, Yiqing Xie, Gra-
ham Neubig, and Daniel Fried. 2024. CodeRAG-Bench: Can Retrieval Augment
Code Generation? CoRR abs/2406.14497 (2024). doi:10.48550/ARXIV.2406.14497
arXiv:2406.14497
[73] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei
Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-Thought prompting
elicits reasoning in large language models. In Proceedings of the 36th International
Conference on Neural Information Processing Systems (New Orleans, LA, USA)
(NIPS ’22) . Curran Associates Inc., Red Hook, NY, USA, Article 1800, 14 pages.
[74] Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Benjamin Feuer, Sid-
dhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah, Sreemanti Dey,
Shubh-Agrawal, Sandeep Singh Sandha, Siddartha Venkat Naidu, Chinmay
Hegde, Yann LeCun, Tom Goldstein, Willie Neiswanger, and Micah Gold-
blum. 2025. LiveBench: A Challenging, Contamination-Limited LLM Bench-
mark. In The Thirteenth International Conference on Learning Representations .
https://openreview.net/forum?id=sKYHBTAxVa
[75] Haolun Wu, Yansen Zhang, Chen Ma, Fuyuan Lyu, Bowei He, Bhaskar Mitra,
and Xue Liu. 2024. Result Diversification in Search and Recommendation: A
Survey. IEEE Trans. on Knowl. and Data Eng. 36, 10 (April 2024), 5354–5373.
doi:10.1109/TKDE.2024.3382262
[76] Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal
Choudhary, Rongze Gui, Ziran Jiang, Ziyu JIANG, Lingkun Kong, Brian Moran,
Jiaqi Wang, Yifan Ethan Xu, An Yan, Chenyu Yang, Eting Yuan, Hanwen Zha,
Nan Tang, Lei Chen, Nicolas SCHEFFER, Yue Liu, Nirav Shah, Rakesh Wanga,
Anuj Kumar, Wen tau Yih, and Xin Luna Dong. 2024. CRAG - Comprehensive
RAG Benchmark. In The Thirty-eight Conference on Neural Information Processing
Systems Datasets and Benchmarks Track . https://openreview.net/forum?id=Q7lAqY41HH
[77] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. In Proceedings of the 2018
Conference on Empirical Methods in Natural Language Processing , Ellen Riloff,
David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (Eds.). Association for
Computational Linguistics, Brussels, Belgium, 2369–2380. doi:10.18653/v1/D18-
1259
[78] Matei Zaharia, Omar Khattab, Lingjiao Chen, Jared Quincy Davis, Heather Miller,
Chris Potts, James Zou, Michael Carbin, Jonathan Frankle, Naveen Rao, and
Ali Ghodsi. 2024. The Shift from Models to Compound AI Systems. https:
//bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/.
[79] Hamed Zamani, Michael Bendersky, Donald Metzler, Honglei Zhuang, and Xuan-
hui Wang. 2022. Stochastic Retrieval-Conditioned Reranking. In Proceedings of
the 2022 ACM SIGIR International Conference on Theory of Information Retrieval
(Madrid, Spain) (ICTIR ’22) . Association for Computing Machinery, New York,
NY, USA, 81–91. doi:10.1145/3539813.3545141
[80] Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi
Mao, Jian-Guang Lou, and Weizhu Chen. 2023. RepoCoder: Repository-Level
Code Completion Through Iterative Retrieval and Generation. In Proceedings
of the 2023 Conference on Empirical Methods in Natural Language Processing ,
Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational
Linguistics, Singapore, 2471–2484. doi:10.18653/v1/2023.emnlp-main.151
[81] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting
Huang, Enbo Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu,
Wei Bi, Freda Shi, and Shuming Shi. 2023. Siren’s Song in the AI Ocean: A
Survey on Hallucination in Large Language Models. CoRR abs/2309.01219 (2023).
doi:10.48550/ARXIV.2309.01219 arXiv:2309.01219
12
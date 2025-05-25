# Think Before You Attribute: Improving the Performance of LLMs Attribution Systems

**Authors**: João Eduardo Batista, Emil Vatai, Mohamed Wahib

**Published**: 2025-05-19 02:08:20

**PDF URL**: [http://arxiv.org/pdf/2505.12621v1](http://arxiv.org/pdf/2505.12621v1)

## Abstract
Large Language Models (LLMs) are increasingly applied in various science
domains, yet their broader adoption remains constrained by a critical
challenge: the lack of trustworthy, verifiable outputs. Current LLMs often
generate answers without reliable source attribution, or worse, with incorrect
attributions, posing a barrier to their use in scientific and high-stakes
settings, where traceability and accountability are non-negotiable. To be
reliable, attribution systems need high accuracy and retrieve data with short
lengths, i.e., attribute to a sentence within a document rather than a whole
document. We propose a sentence-level pre-attribution step for
Retrieve-Augmented Generation (RAG) systems that classify sentences into three
categories: not attributable, attributable to a single quote, and attributable
to multiple quotes. By separating sentences before attribution, a proper
attribution method can be selected for the type of sentence, or the attribution
can be skipped altogether. Our results indicate that classifiers are
well-suited for this task. In this work, we propose a pre-attribution step to
reduce the computational complexity of attribution, provide a clean version of
the HAGRID dataset, and provide an end-to-end attribution system that works out
of the box.

## Full Text


<!-- PDF content starts -->

arXiv:2505.12621v1  [cs.CL]  19 May 2025Think Before You Attribute: Improving the
Performance of LLMs Attribution Systems
João E. Batista
RIKEN-CCS
Kobe, Japan
joao.batista@riken.jpEmil Vatai
RIKEN-CCS
Kobe, JapanMohamed Wahib
RIKEN-CCS
Kobe, Japan
Abstract
Large Language Models (LLMs) are increasingly applied in various science do-
mains, yet their broader adoption remains constrained by a critical challenge: the
lack of trustworthy, verifiable outputs. Current LLMs often generate answers with-
out reliable source attribution, or worse, with incorrect attributions, posing a barrier
to their use in scientific and high-stakes settings, where traceability and account-
ability are non-negotiable. To be reliable, attribution systems need high accuracy
and retrieve data with short lengths, i.e., attribute to a sentence within a document
rather than a whole document. We propose a sentence-level pre-attribution step
for Retrieve-Augmented Generation (RAG) systems that classify sentences into
three categories: not attributable, attributable to a single quote, and attributable to
multiple quotes. By separating sentences before attribution, a proper attribution
method can be selected for the type of sentence, or the attribution can be skipped
altogether. Our results indicate that classifiers are well-suited for this task. In this
work, we propose a pre-attribution step to reduce the computational complexity
of attribution, provide a clean version of the HAGRID dataset1, and provide an
end-to-end attribution system that works out of the box2.
1 Introduction
Recent advances in the field of machine learning have enabled the creation of deep learning models
that are not limited to specific tasks but are general-purpose, capable of performing a wide range
of functions at near-human performance levels. Large Language Models (LLMs) [ 7] are widely
accessible and often provide accurate answers to general queries. However, they are also prone to
generating incorrect statements [ 11,16,26] (see supplementary material), which can be risky in
systems where accuracy is essential. Despite their shortcomings [ 43], LLMs are driving progress
across science and industry. Yet, users in domains where correctness and accuracy are of high
importance, e.g. medical, law, and academic scholarly, are hindered by the possibility of the
generated content being incorrect. Improving LLMs trustworthiness and verifiability becomes a
critical challenge, especially when it is difficult to judge the correctness of the answer [39].
Attribution can enhance the trustworthiness of LLMs by identifying the sources that influence their
outputs [ 38,46], improving transparency and enabling fact-checking [ 9]. In this work, we examine
two key issues with LLM-generated responses in the context of retrieval-augmented generation
(RAG) systems: (1) answers that contain false statements, and (2) references that are missing,
unrelated, or hallucinated [ 47]. Research shows that users are more likely to trust systems that
provide explanations – even if they are incorrect – compared to those that provide no explanation at
1https://anonymous.4open.science/r/HAGRID-clean-4223
2https://anonymous.4open.science/r/AttributionPipeline
Preprint. Under review.

End-to-end Attribution System
Query
AnswerText Generation Pre-attribution Attribution Training data:
   Hagrid
   WebGLM-QA
Documents
used for
RAGLanguage Model,
Sentiment Analyzer ,
Text Processer
Classification
AlgorithmClassifier
 Our proposal
One-to-one
AttributionOne-to-Many
AttributionNo
Attribution1 referenceNo references 2+ referencesAttribution
Method
Retrieved
sentences Tabular dataset:
   X = Information extracted
from each sentence
   Y = {"0 ref", "1 ref", "2+ ref"}
(training data labels)Decompose text into sentences 
and respective list of references
(training data)RAG
System
Text,
decomposed
into sentences
and respective
embeddingsFigure 1: Overview of our proposed end-to-end attribution system. We assume sentence-level
attribution in RAG systems typically requires a one-to-many algorithm, which is computationally
expensive. Our pre-attribution step predicts whether a sentence requires 0, 1, or multiple references,
allowing the system to apply more efficient attribution methods when appropriate. Green arrows
represent the model training process; yellow arrows indicate usage during inference.
all [5,12,23]. This highlights the risk of users placing undue trust in attribution systems that generate
inaccurate references, reinforcing the need for reliable attribution. Moreover, popular LLM interfaces
often cite entire documents as sources (e.g., Perplexity [ 35]), requiring users to read the full text to
verify claims. This makes fact-checking in such systems impractical [31, 46].
The lack of trust in LLMs results in their application being limited in high-risk applications, such
as dispatching emergency services, law enforcement, and border control, among other applications
identified by agencies, such as the US Congress [ 1], the European Comission [ 13], and the Japanese
AI Safety Institute [ 19]. This leads to scrutiny that aims to improve the reliability of LLMs. Examples
of unreliability of LLMs can be seen in the supplementary material section.
To address current attribution limitations, we propose a pre-attribution step (see Figure 1) for
sentence-level attribution in RAG systems [ 24]. Compared to document-level methods, sentence-level
attribution enables faster, more efficient fact-checking. Our model predicts whether each sentence
requires zero, one, or multiple references, allowing the system to apply attribution methods selectively
and reduce computational cost. Experiments show good results on the original datasets, with further
gains on our manually cleaned version of the HAGRID dataset [ 20]. To support future research, we
release both the cleaned HAGRID dataset and our implementation code on GitHub1,2.
2

2 Background and Related Work
This work focuses on sentence-level attribution in Retrieval Augmented Generation (RAG) [ 24]
systems. As such, the background is connected to the research fields of attribution in LLMs and
sentence-level Information Retrieval (IR). While our attribution method focuses on semantic simi-
larity with embedding, we also intend to explore using dense passage retrieval and keyword search
approaches.
2.1 Attribution in LLMs
Attribution can be broadly split into two categories based on when the source information is accessed:
at training or inference time. While training-time attribution has been explored [ 22,17,8], this work
is focused only on inference-time attribution, i.e., RAG. The increasing need for reliable attribution
has made "Attribution in LLMs" an increasingly popular research area, with several methods surveyed
by Li et al. [25]. Following this survey, we broadly categorize attribution into three categories:
Model-driven attribution: The model uses its internal knowledge to answer questions and provide
sources. Since the sources are generated, they are often hallucinations themselves [ 2], making this
approach not reliable;
RAG: Using RAG [ 4,6,24,41,37], before generating an answer, the algorithm uses the input query
to retrieve information from an external source and integrates it into the query. With this step, the
answer will be based in both the model’s internal knowledge and the retrieved information. While
theoretically speaking, the answer can be attributed back to the source, the internal knowledge of
the LLM may conflict with the external knowledge, resulting in a hallucinated answer [ 45]. As a
consequence, the answer does not match the referenced documents;
Post-generation Attribution: The answer is generated using the model’s internal knowledge or
using RAG. After generation, it uses an information retrieval algorithm to fetch external references
that back the generated answer [32, 36];
2.2 Sentence-level Information Retrieval
On a high level, the objective of this work is to, given a sentence and a document, figure out how to
obtain a quote from the document that matches the sentence. This task is related to the work done in
the information retrieval field. By providing direct quotations from the source documents, we can
guarantee that the explanation is true and reduce the time spent on fact-checking, from reading a
whole document to a single sentence or paragraph. Sentence retrieval has been studied for many
decades [ 3,29,21], including methods from the most naive (e.g., exact or fuzzy string matching) to
modern approaches that employ LLMs for semantic similarity using embeddings, keyword search, or
dense passage retrieval:
String Matching: Both exact and fuzzy string matching [ 33] aim to find an exact or similar matching
of the input sentence in the document. Although simple and computationally efficient, it is not reliable
to use this approach for attribution;
Semantic Similarity using Embeddings: Calculating the embeddings of a sentence using an
LLM [ 10,28,40,44], reduces the sentence to a point in an n-dimensional space. In this space,
sentences with similar meanings are placed closer to each other, allowing the association of sentences
that are similar in meaning, even when the sentences are written in different ways. While this approach
allows us to capture semantic meaning, calculating embeddings is computationally expensive;
Dense Passage Retrieval (DPR): Similarly to semantic similarity with embeddings, DPR [ 21] splits
documents into sections and ranks them based on similarity with the input. The main difference is
the size of the sections retrieved, which, due to their larger size, require LLMs that can handle larger
context lengths, requiring additional computational resources;
Keyword Search (KS): KS [ 29] is a middle point in the amount of information extracted and the
computational cost. By extracting keywords from sentences, two sentences are considered similar
if the same set of keywords is extracted from them. However, this approach may be susceptible to
similar (but not equal) keywords being extracted from both sentences and them not being matches.
3

Table 1: Details on the WebGLM-QA, HAGRID, and HAGRID-Clean datasets
No.Samples Longest No. Longest Total No. references
(No. Queries) Quote Quotes Sentence No. Sentences in each sentence
0 1 2+
WebGLM-QA 43979 2960 char 3-5 1433 char 186027 31.3k 118.0k 36.6k
HAGRID 2638 5862 char 1-12 1435 char 7702 714 5455 1533
HAGRID-Clean 2638 5862 char 1-12 1435 char 7308 403 6140 765
Table 2: Example of a sample within the HAGRID dataset.
Query What does it mean to be an evergreen tree?
Quotes 1- Trees are either evergreen, having foliage that persists and remains green...
2- In botany, an evergreen is a plant that has leaves throughout the year, always...
Answer #1 To be an evergreen tree means to have foliage that persists and remains green...
Answer #2 It is a plant that has leaves throughout the year and never completely loses... [1,2]
Most conifers, including pine and fir trees, are evergreens, while deciduous... [1]
In this work, we pick up previous work on information retrieval using semantic similarity and extend
it to the field of attribution in LLMs, providing a sentence-level attribution system that verifies the
truthfulness of a generated sentence by searching for equivalent sentences in reliable sources of data.
3 Methodology
This work introduces an end-to-end sentence-level attribution system that, given a generated text and
the documents used as source (RAG systems), attributes each sentence in the text to quotes from
the documents, allowing for quick fact-checking (see Figure 1). Since sentences can be attributed to
zero, one, or multiple quotes, the challenge lies in determining how many references are needed and
identifying the optimal set. To address this, we introduce a pre-attribution system that categorizes
sentences into the number of required references, allowing the system to efficiently choose an
attribution method, optimized for the reference count, significantly reducing computational overhead.
Our pre-attribution step relies on attribution datasets containing a list of sentences and target refer-
ences, text processing tools to extract relevant information regarding the complexity of each sentence,
and a classification algorithm to associate each sentence with the expected number of citations. This
section describes the datasets used, how we preprocessed the data, and the classifiers used.
Following pre-attribution, the pipeline selects the most appropriate attribution strategy for each
sentence. Below, we briefly summarize the benefits of each attribution type:
No attribution: Identifying sentences that do not require attribution avoids unnecessary computation,
reducing system costs.
One-to-one attribution: Lightweight attribution methods can be used for simple sentences, offering
computational efficiency. These also help signal when fact-checking is appropriate.
One-to-many attribution: Sentences requiring multiple references are flagged for fact-checking,
and more complex attribution methods are applied. While more computationally costly, this remains
the default fallback.
3.1 WebGLM-QA, HAGRID and HAGRID-Clean Datasets
While other attribution datasets exist, such as TabCite [ 30], ASQA [ 42], ELI5 [ 14], EquinorQA [ 15],
we focus on HAGRID [ 20] and WebGLM-QA [ 27], as both are ready-to-use sentence-level attribution
datasets. Both datasets contain a similar structure; thus, while we primarily describe HAGRID below,
the discussion also applies to WebGLM-QA. See Table 1 for additional dataset details.
The dataset samples comprise three elements: a query, one or two answers, and a list of quotes that
can be attributed to each sentence in the answers, as seen in Table 2. In this work, we are only
concerned with the answers (sentences to attribute) and references; the queries are not used. Each
answer comes split into a list of sentences, and each sentence is associated with a list of references (if
4

Table 3: Sentences within HAGRID. The classifier used for pre-attribution makes a prediction that
does not match the labeled number of references, but we consider the prediction correct.
Type of Issue Sentence
Over referencing The Earth’s atmosphere today contains 21% oxygen [1][2][3][4].
Over referencing Scotland’s national dish is haggis [1][2][4].
Under referencing The atomic number of mercury is 80.
Under referencing The first home model of microwave oven was introduced by Tappan, which
was licensed by Raytheon in 1955, but it was too large and expensive for
general home use [1].
Invalid sentence “Cairns Airport,” Wikipedia, Aug. 14, 2021.
https://en.wikipedia.org/wiki/Cairns_Airport [2]
any). Each sample contains up to 12 attributable quotes, and the sentences contain up to 9 references.
The displayed sample contains two attributable quotes and two answers. The first answer comprises a
single sentence, with no references. The second answer contains two sentences: the first is attributed
to both quotes, and the second is attributed to only the first quote.
3.1.1 Processing of the HAGRID Dataset
During our preliminary experiments, we encountered several challenges with both the HAGRID
and WebGLM-QA datasets that affected model accuracy. These included noisy input data and
inconsistencies between the content of sentences and the number of listed references. While WebGLM-
QA is too large for manual cleaning, we manually curated HAGRID to produce a cleaned version,
which we refer to as HAGRID-Clean.
Cleaning Labels:
We found that using the raw number of references per sentence to determine whether a sentence should
have zero, one, or multiple references introduced three main problems. First, some sentences were
over-referenced. Despite being simple, they included unnecessarily long lists of references, which
can mislead models into interpreting them as highly information-rich. Second, we observed under-
referenced sentences, where complex or factual statements were supported by too few references.
Third, some sentences were invalid, meaning they contained no attributable content and should not
be included in the attribution task at all. These cases are illustrated in Table 3.
To address these issues, we manually reviewed each sentence in HAGRID and assigned a cleaned
label indicating whether it should have zero, one, or multiple references or be marked as invalid. For
training purposes, invalid sentences were grouped with the "zero references" class to increase class
diversity.
In addition to label noise, we observed inconsistent referencing styles across samples3, which made
it non-trivial to extract citations accurately. These inconsistencies affected both HAGRID and
WebGLM-QA.
Cleaning Samples:
Cleaning the samples involved standardizing the reference format and removing duplicated sentences.
As previously seen in Table 2, the sentences in the datasets include the reference list. We went through
all the samples, separating each sentence into a tuple containing the sentence itself and the list of
references (now written consistently among all samples). Additionally, in multiple cases, the samples
contained duplicated sentences, i.e., sentences are shown multiple times, with different references.
Removing them reduced the total number of sentences within the dataset from 7702 to 7308.
3.2 Preprocessing the Datasets
3.2.1 Input Variables
As shown in Figure 1, the first step in our pipeline involves splitting the HAGRID and WebGLM-QA
datasets into individual sentences to extract input variables for the classification model. From each
3Referencing styles found: [1], [1][2], [1,2], [1, 2], [1,2,], [1 and 2], [1-2], (1), and [context 1], among others.
5

sentence, we compute 24 numerical features (detailed in Table 7, supplementary material). These
features capture various textual properties, such as sentence length, reading ease, and the number of
named entities, among others. Using these features as input and the reference count category (zero,
one, or multiple) as the label, we train a classifier to predict the expected number of references each
sentence should have.
Although each dataset sample includes one or two answers, each consisting of a list of sentences and
their associated references, the classification task operates at the sentence level. So, while Table 1
indicates that HAGRID contains 2638 samples, we derive 7702 training and test instances by treating
each sentence as a separate data point. Since many sentences are duplicates, HAGRID-Clean, despite
having the same number of queries, includes only 7308 unique sentences after deduplication.
3.2.2 Target Values for Pre-attribution
The goal of the pre-attribution step is to classify each sentence into one of three categories, based on
the number of references it ideally requires: zero, one, or multiple. This label later guides attribution,
helping optimize computational efficiency and accuracy. In the original HAGRID and WebGLM-QA
datasets, this labeling process was automatic. We assigned labels based on the number of citations
already linked to each sentence. However, during the manual cleaning of HAGRID, we introduced
an additional layer of supervision by assigning labels based on human judgment. Instead of relying
solely on citation count, we assessed each sentence’s informational content to determine how many
references it should have, correcting for over- or under-referencing in the raw data.
This task can also be interpreted as measuring the semantic complexity of a sentence. Sentences
labeled as requiring multiple references typically integrate more information and therefore demand
broader evidence. This interpretation aligns with prior work such as Hu et al. [18], who distinguish
among three types of complex sentence reasoning: (1) Union , where multiple independent sources
support different parts of an answer; (2) Intersection , where the answer arises from commonalities
across sources; and (3) Concatenation , where reasoning unfolds across a citation chain. While these
distinctions are important, the datasets used in this study only allow for labeling based on citation
quantity. Thus, we treat all complex cases as belonging to the same "multiple references" class.
3.2.3 Target Values for Attribution
Each sentence in all three datasets is accompanied by a list of references. To evaluate attribution
quality, we define a sentence as correctly attributed under the following criteria: if it has zero
references, it must be identified as requiring none; if it has one reference, that specific reference
must be correctly retrieved; and if it has multiple references, at least two correct references must
be identified, with no incorrect references included. For example, if a sentence is associated with
references [1, 2, 3], predictions like [1, 2] or [1, 2, 3] are considered valid, but [1, 2, 4] is not.
3.3 Classification Algorithms for Pre-attribution
To perform the pre-attribution step, we experimented with three classifiers to predict how many
references a sentence should have: Random Forest (RF), Histogram Gradient Boosting (HGB), and
Linear Support Vector Classifier (LSVC). We used the implementations provided by the sklearn
Python library [ 34]. Since the dataset is highly unbalanced, we changed the class _weight parameter
of all classifiers to balanced . Additionally, for RF and HGB, we limited the maximum tree depth to
14 in both HAGRID datasets and 22 in WebGLM-QA – the lowest depth value required for the RF to
memorize the training data – while keeping all other hyperparameters at their default settings.
For evaluation, we conducted 30 independent runs per classifier-dataset pair. In each run, we
combined the original training and test splits provided by the dataset authors. We then perform a
new random split, using 70% of the samples for training and 30% for testing, while maintaining the
original ratio between classes.
3.4 Attribution Algorithms
As discussed in the related work section, attribution algorithms generally follow one of three ap-
proaches: (1) leveraging LLM-generated sentence embeddings for similarity search, (2) applying
deep passage retrieval techniques, or (3) using keyword-based search.
6

WebGLM-QA HAGRID HAGRID Clean
Training 99.24 98.38 99.61
IQR 99.19–99.27 98.22–98.59 99.57–99.67
Test 63.94 70.62 95.67
IQR 63.82–63.99 69.93–71.14 95.44–95.85
Table 4: Median accuracy values (%) obtained in each dataset over 30 runs using the RF classifier,
and respective interquartile values.
We focus this work on embedding-based attribution due to its simplicity and strong baseline perfor-
mance. While there is room for improvement, our initial results were satisfactory. The pipeline is
designed to be fully modular, allowing users to implement and use other attribution algorithms. We
evaluate two embedding-based attribution strategies:
Matching with the closest embedding: This method attributes each sentence to the single most
similar reference based on embedding distance. It is computationally efficient and yields strong
results, but cannot assign multiple references or detect sentences that should not be attributed.
Matching with the closest two embeddings: This method computes the distances between the
sentence embedding and all individual quotes and quote pairs (using the average of two embeddings).
The sentence is then attributed to the closest quote or quote pair. Although more computationally
intensive, this method benefits from the pre-attribution step and achieves better results.
3.5 Real-World Application
Our ultimate goal is to provide a system that experts can use on their own devices for quick
fact-checking of the information generated by LLM systems. As such, we also include in this
work two real-world application examples using ChatGPT (GPT-4o), Perplexity (GPT-3.5), and
DeepSeek (DeepSeek-V3). To test this system, we use LLMs for two tasks:
Make a summary of scientific papers: We provide one of the papers referenced in this work [ 25]
and the query "Make a summary of the paper attached to this message using simple sentences." Then,
the answer and paper, in PDF format, are given as input to our system, which attempts to assign each
sentence in the answer to a quote from the paper;
Query about medicine consumption: We ask GPT about whether it is safe to increase the dosage of
Paracetamol, and then try to match the answer with leaflets written in both Portuguese and English.
4 Results
In this section, we discuss the accuracy of the pre-attribution classifiers in each of the three datasets,
followed by the accuracy of the whole pipeline for the attribution task. Additionally, we test the
pipeline on a real-world application use case, and comment on the current challenges and limitations,
and how we intend to improve the pipeline.
4.1 Accuracy on pre-attribution
This section will focus on the results obtained using the RF classifier during the pre-attribution step
since its results outperformed both HGB and LSVC classifiers, making it our recommended classifier
for this task. We performed 30 independent runs using different training and test splits of each dataset,
using a 70/30 split for training and test. The median accuracy results and respective interquartile
values are displayed in Table 4, and the confusion matrix plots can be seen in Figure 2. The confusion
matrices are normalized in each row, highlighting the precision in each class.
These results show that, while the RF classifiers were able to memorize the whole datasets, their
accuracy was severely reduced in the unclean datasets (WebGLM-QA and HAGRID), while showing
very satisfactory results in the cleaned dataset. In the clean dataset, the class for sentences with
multiple references has the lower precision, and yet, they are correctly identified 89% of the time.
Meanwhile, in the unclean datasets, sentences with no references and sentences with multiple
references are often mistaken for sentences containing a single reference.
7

Zero One Multiple
Predicted labelZero One MultipleTrue label98.39 1.36 0.26
0.39 99.39 0.23
0.22 0.34 99.44Training -- WebGLM-QA
020406080100
Zero One Multiple
Predicted labelZero One MultipleTrue label96.17 3.41 0.42
0.21 99.23 0.57
0.12 3.37 96.51Training -- HAGRID
020406080100
Zero One Multiple
Predicted labelZero One MultipleTrue label99.85 0.15 0.00
0.00 99.93 0.07
0.01 2.86 97.13Training -- HAGRID Clean
020406080100
Zero One Multiple
Predicted labelZero One MultipleTrue label48.90 35.53 15.57
13.88 66.52 19.60
13.19 54.35 32.46T est -- WebGLM-QA
020406080100
Zero One Multiple
Predicted labelZero One MultipleTrue label52.17 41.78 6.05
8.20 73.76 18.03
4.44 51.49 44.07T est -- HAGRID
020406080100
Zero One Multiple
Predicted labelZero One MultipleTrue label93.55 6.45 0.00
1.57 96.48 1.94
0.15 10.69 89.16T est -- HAGRID Clean
020406080100
Figure 2: Average confusion matrix obtained over 30 runs in each dataset using RF for pre-attribution.
The confusion matrices are normalized in each row, highlighting the precision (%) in each class.
Attribution Method WebGLM-QA HAGRID HAGRID Clean
Attribute to Closest Quote
No pre-attribution
Training (IQR) 54.61 ( 54.6 – 54.7) 54.39 ( 54.2 – 54.6) 70.93 ( 70.8 – 71.2)
Test (IQR) 54.65 ( 54.5 – 54.7) 54.24 ( 53.7 – 54.7) 70.91 ( 70.3 – 71.2)
Using pre-attribution
Training (IQR) 71.00 ( 71.0 – 71.1) 63.15 ( 62.9 – 63.4) 76.48 ( 76.3 – 76.6)
Test (IQR) 56.34 ( 56.3 – 56.5) 54.82 ( 54.4 – 55.6) 74.87 ( 74.3 – 75.3)
Attribute to Closest Two Quotes
No pre-attribution
Training (IQR) 34.06 ( 34.0 – 34.1) 37.51 ( 37.2 – 37.7) 38.99 ( 38.6 – 39.2)
Test (IQR) 34.08 ( 34.0 – 34.2) 37.34 ( 37.0 – 38.0) 38.51 ( 38.0 – 39.4)
Using pre-attribution
Training (IQR) 78.30 ( 78.3 – 78.4) 66.19 ( 65.9 – 66.6) 77.87 ( 77.7 – 78.0)
Test (IQR) 56.25 ( 56.2 – 56.4) 54.44 ( 54.1 – 55.3) 76.06 ( 75.6 – 76.4)
Table 5: Median accuracy values (%), and respective interquartile values, obtained in each dataset
over 30 runs using the RF classifier for pre-attribution and two attribution methods. All six text cases
have a statistically significant improvement in test accuracy, using a p-value of 0.01.
4.2 Accuracy on Attribution
Table 5 presents the median results of applying two attribution methods across three datasets, averaged
over 30 runs with different training and testing splits. Each attribution method was evaluated both
with and without the pre-attribution step. Across all six test cases, the use of pre-attribution resulted
in statistically significant improvements ( p-value of 0.01).
Attributing to the Closest Quote: When assigning a sentence to the closest quote, using pre-
attribution allows for filtering sentences that are supposed not to be attributed, increasing the accuracy
by detecting those sentences.
Attributing to the Two Closest Quote: When assigning a sentence to the closest quote or pair of
quotes, this method shows low accuracy without using pre-attribution. This is caused by its tendency
to pick quote pairs, while most dataset sentences correspond to a single reference. Using pre-
attribution, we can detect which sentences should be attributed to a single quote and use the previous
attributor instead. This results in drastic improvements when compared to not using pre-attribution.
8

4.3 Real-World Application
We evaluated our end-to-end attribution system in two tasks and provide the chat transcripts and
attribution logs in the supplementary materials:
Summarizing a scientific paper: We provided a paper [ 25], query for a summary to three LLM
systems (ChatGPT, Perplexity, and DeepSeek), and attribute each generated sentence back to the paper.
We observed that the pre-attribution model rarely identifies sentences as not requiring attribution.
Instead, it recommends searching for a single quote. This might be due to the training data consisting
mostly of single-reference sentences. In the attribution step, the system gives satisfactory results in
the text generated by Perplexity and DeepSeek, correctly attributing more than half of the generated
text. Although we detected issues attributing ChatGPT’s answer, we later dismissed them since, after
manually searching for attributable sentences in the paper, we found that part of the answer was
hallucinated. Since the system assigns each sentence to the most similar quote, hallucinated content
is often incorrectly attributed to seemingly random sections text, leading to unexpected results. In
future work, we plan to explore attribution algorithms that discard quotes unrelated to the input.
Query about medicine consumption: Without providing documents, we query GPT about whether
it is safe to increase the dosage of Paracetamol. Then, we provide the answer to our system and two
leaflets, written in Portuguese and English. In the Portuguese leaflet, our system provides a table
containing the daily maximum dosage values per age and weight; and in the English leaflet the system
provides a quote with the same information as GPT. This indicates that our system may be capable of
matching information to sources, even when they are written in different languages.
5 Conclusions and Limitations
In this work, we provide an end-to-end sentence-level attribution system that contains our proposed
pre-attribution step. Using this pre-attribution step, we induce a classifier to categorize sentences into
needing zero, one, or multiple references. With this, the attribution system can allocate a sentence to
computationally cheaper attribution algorithms if necessary and completely avoid attribution costs
when a sentence does not require attribution. The classifier is trained to separate sentences based on
their features such as: number of words, readability ease, sentiment analysis, among others.
During our preliminary results, we noticed a lack of reliable attribution datasets. To address this issue,
we also release a manually cleaned version of the HAGRID dataset. Our results in the pre-attribution
step indicate that, while the classifier often mistakes sentences as containing a single sentence in the
WebGLM-QA and HAGRID datasets, they show good robustness in the cleaned HAGRID dataset.
After predicting whether a sentence should have zero, one, or multiple references, we proceed to
attribution. Our results indicate that pre-attribution brings a statistically significant improvement to
the robustness (test accuracy) of the attribution methods in all six test cases.
We tested the pipeline in two real-world scenarios by: asking ChatGPT, Perplexity, and DeepSeek to
summarize a paper, and then attempting to attribute the answer back to paper. The system correctly
attributed the sentences obtained from Perplexity and DeepSeek more than half the time. The text
generated by ChatGPT contained hallucinated sentences that were attributed to seemingly random
quotes, motivating our need to detect when there are no related sentences in the document; and asking
ChatGPT about daily maximum dosages of medicine and attributing the answer to leaflets in two
different languages (Portuguese and English).
Limitations: While the attribution system provides a quick output for small documents (e.g., research
papers), we intend to use deep passage retrieval techniques to simplify the search space before
applying sentence-level attributions on larger documents. Currently, the attribution method does not
detect when there are no quotes that back a statement. We intend to explore other attribution methods,
such as keyword search, to detect hallucinations. Lastly, the datasets used for pre-attribution suffer
from class imbalance, resulting in low robustness in the pre-attribution model. We intend to explore
semi-supervised learning techniques to increase the training data, compensating for this issue.
Software and Data
The HAGRID [ 20] and WebGLM-QA [ 27] datasets can be accessed through their respective papers.
The code used in this work and our clean version of the HAGRID dataset are available online1,2.
9

References
[1]117th United States Congress (2021-2022). H.R.6580 - Algorithmic Accountability Act of
2022 . URLhttps://www.congress.gov/bill/117th-congress/house-bill/6580/text .
Accessed: November 01, 2024.
[2]Ayush Agrawal, Mirac Suzgun, Lester Mackey, and Adam Tauman Kalai. Do language models
know when they’re hallucinating references?, 2024. URL https://arxiv.org/abs/2305.
18248 .
[3]Alfred V . Aho and Margaret J. Corasick. Efficient string matching: an aid to bibliographic search.
Commun. ACM , 18(6):333–340, June 1975. ISSN 0001-0782. doi: 10.1145/360825.360855.
URLhttps://doi.org/10.1145/360825.360855 .
[4]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning
to retrieve, generate, and critique through self-reflection, 2023. URL https://arxiv.org/
abs/2310.11511 .
[5]Gagan Bansal, Tongshuang Wu, Joyce Zhou, Raymond Fok, Besmira Nushi, Ece Kamar,
Marco Tulio Ribeiro, and Daniel S. Weld. Does the whole exceed its parts? the effect of ai
explanations on complementary team performance, 2021. URL https://arxiv.org/abs/
2006.14779 .
[6]Bojana Bašaragin, Adela Ljaji ´c, Darija Medvecki, Lorenzo Cassano, Miloš Košprdi ´c, and Nikola
Miloševi ´c. How do you know that? teaching generative language models to reference answers
to biomedical questions. In Dina Demner-Fushman, Sophia Ananiadou, Makoto Miwa, Kirk
Roberts, and Junichi Tsujii, editors, Proceedings of the 23rd Workshop on Biomedical Natural
Language Processing , pages 536–547, Bangkok, Thailand, August 2024. Association for
Computational Linguistics. doi: 10.18653/v1/2024.bionlp-1.44. URL https://aclanthology.
org/2024.bionlp-1.44 .
[7]Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel
Herbert-V oss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M.
Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz
Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec
Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020.
URLhttps://arxiv.org/abs/2005.14165 .
[8]Tyler A. Chang, Dheeraj Rajagopal, Tolga Bolukbasi, Lucas Dixon, and Ian Tenney. Scalable
influence and fact tracing for large language model pretraining, 2024. URL https://arxiv.
org/abs/2410.17413 .
[9]Jifan Chen, Grace Kim, Aniruddh Sriram, Greg Durrett, and Eunsol Choi. Complex claim
verification with evidence retrieved in the wild, 2024. URL https://arxiv.org/abs/2305.
11859 .
[10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of
deep bidirectional transformers for language understanding. CoRR , abs/1810.04805, 2018. URL
http://arxiv.org/abs/1810.04805 .
[11] Nouha Dziri, Sivan Milton, Mo Yu, Osmar Zaiane, and Siva Reddy. On the origin of hal-
lucinations in conversational models: Is it the datasets or the models? In Marine Carpuat,
Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz, editors, Proceedings of the 2022
Conference of the North American Chapter of the Association for Computational Linguis-
tics: Human Language Technologies , pages 5271–5285, Seattle, United States, July 2022.
Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.387. URL
https://aclanthology.org/2022.naacl-main.387/ .
[12] Malin Eiband, Daniel Buschek, Alexander Kremer, and Heinrich Hussmann. The impact
of placebic explanations on trust in intelligent systems. In Extended Abstracts of the 2019
CHI Conference on Human Factors in Computing Systems , CHI EA ’19, page 1–6, New
York, NY , USA, 2019. Association for Computing Machinery. ISBN 9781450359719. doi:
10.1145/3290607.3312787. URL https://doi.org/10.1145/3290607.3312787 .
10

[13] European Commission. Regulation of the European Parliament and of the Council Laying
Down Harmonised Rules on Artificial Intelligence (Artificial Intelligence Act) and Amending
Certain Union Legislative Acts , 2021. URL https://eur-lex.europa.eu/legal-content/
EN/TXT/HTML/?uri=CELEX:52021PC0206 . Accessed: November 01, 2024.
[14] Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. ELI5:
Long form question answering. In Anna Korhonen, David Traum, and Lluís Màrquez, editors,
Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics ,
pages 3558–3567, Florence, Italy, July 2019. Association for Computational Linguistics. doi:
10.18653/v1/P19-1346. URL https://aclanthology.org/P19-1346/ .
[15] Darío Garigliotti, Bjarte Johansen, Jakob Vigerust Kallestad, Seong-Eun Cho, and Cèsar
Ferri. EquinorQA: Large Language Models for Question Answering Over Proprietary Data .
IOS Press, October 2024. ISBN 9781643685489. doi: 10.3233/faia241049. URL http:
//dx.doi.org/10.3233/FAIA241049 .
[16] Jocelyn Gravel, Madeleine D’Amours-Gravel, and Esli Osmanlliu. Learning to fake it: Limited
responses and fabricated references provided by chatgpt for medical questions. Mayo Clinic
Proceedings: Digital Health , 1(3):226–234, 2023. ISSN 2949-7612. doi: https://doi.org/10.
1016/j.mcpdig.2023.05.004. URL https://www.sciencedirect.com/science/article/
pii/S2949761223000366 .
[17] Roger Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini,
Benoit Steiner, Dustin Li, Esin Durmus, Ethan Perez, Evan Hubinger, Kamil ˙e Lukoši ¯ut˙e,
Karina Nguyen, Nicholas Joseph, Sam McCandlish, Jared Kaplan, and Samuel R. Bowman.
Studying large language model generalization with influence functions, 2023. URL https:
//arxiv.org/abs/2308.03296 .
[18] Nan Hu, Jiaoyan Chen, Yike Wu, Guilin Qi, Sheng Bi, Tongtong Wu, and Jeff Z. Pan. Bench-
marking large language models in complex question answering attribution using knowledge
graphs, 2024. URL https://arxiv.org/abs/2401.14640 .
[19] Japanese AI Safety Institute (AISI). Japanese AISI’s Webpage . URLhttps://aisi.go.jp/
about/index.html . Accessed: November 01, 2024.
[20] Ehsan Kamalloo, Aref Jafari, Xinyu Zhang, Nandan Thakur, and Jimmy Lin. Hagrid: A
human-llm collaborative dataset for generative information-seeking with attribution, 2023. URL
https://arxiv.org/abs/2307.16883 .
[21] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov,
Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering.
In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu, editors, Proceedings of the 2020
Conference on Empirical Methods in Natural Language Processing (EMNLP) , pages 6769–
6781, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/
2020.emnlp-main.550. URL https://aclanthology.org/2020.emnlp-main.550/ .
[22] Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions,
2017.
[23] Vivian Lai and Chenhao Tan. On human predictions with explanations and predictions of
machine learning models: A case study on deception detection. In Proceedings of the Conference
on Fairness, Accountability, and Transparency , FAT* ’19, page 29–38. ACM, January 2019.
doi: 10.1145/3287560.3287590. URL http://dx.doi.org/10.1145/3287560.3287590 .
[24] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and
Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. In Proceed-
ings of the 34th International Conference on Neural Information Processing Systems , NIPS ’20,
Red Hook, NY , USA, 2020. Curran Associates Inc. ISBN 9781713829546.
[25] Dongfang Li, Zetian Sun, Xinshuo Hu, Zhenyu Liu, Ziyang Chen, Baotian Hu, Aiguo Wu, and
Min Zhang. A survey of large language models attribution, 2023. URL https://arxiv.org/
abs/2311.03731 .
11

[26] Xinze Li, Yixin Cao, Liangming Pan, Yubo Ma, and Aixin Sun. Towards verifiable generation:
A benchmark for knowledge-aware language model attribution, 2024. URL https://arxiv.
org/abs/2310.05634 .
[27] Xiao Liu, Hanyu Lai, Hao Yu, Yifan Xu, Aohan Zeng, Zhengxiao Du, Peng Zhang, Yuxiao
Dong, and Jie Tang. Webglm: Towards an efficient web-enhanced question answering system
with human preferences, 2023. URL https://arxiv.org/abs/2306.07906 .
[28] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy,
Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized BERT
pretraining approach. CoRR , abs/1907.11692, 2019. URL http://arxiv.org/abs/1907.
11692 .
[29] Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze. Introduction to Information
Retrieval . Cambridge University Press, 2008.
[30] Puneet Mathur, Alexa Siu, Nedim Lipka, and Tong Sun. MATSA: Multi-agent table struc-
ture attribution. In Delia Irazu Hernandez Farias, Tom Hope, and Manling Li, editors,
Proceedings of the 2024 Conference on Empirical Methods in Natural Language Process-
ing: System Demonstrations , pages 250–258, Miami, Florida, USA, November 2024. As-
sociation for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-demo.26. URL
https://aclanthology.org/2024.emnlp-demo.26/ .
[31] Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen tau Yih, Pang Wei Koh, Mohit
Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. Factscore: Fine-grained atomic evaluation
of factual precision in long form text generation, 2023. URL https://arxiv.org/abs/2305.
14251 .
[32] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christo-
pher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna
Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, and John
Schulman. Webgpt: Browser-assisted question-answering with human feedback, 2022. URL
https://arxiv.org/abs/2112.09332 .
[33] Gonzalo Navarro. A guided tour to approximate string matching. ACM Comput. Surv. , 33(1):
31–88, March 2001. ISSN 0360-0300. doi: 10.1145/375360.375365. URL https://doi.org/
10.1145/375360.375365 .
[34] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion,
Olivier Grisel, Mathieu Blondel, et al. Scikit-learn: Machine learning in python. Journal of
machine learning research , 12(Oct):2825–2830, 2011.
[35] Perplexity. Perplexity.ai (AI Chatbot) , 2023. URL https://www.perplexity.ai/ . Accessed:
November 01, 2024.
[36] Yujia Qin, Zihan Cai, Dian Jin, Lan Yan, Shihao Liang, Kunlun Zhu, Yankai Lin, Xu Han,
Ning Ding, Huadong Wang, Ruobing Xie, Fanchao Qi, Zhiyuan Liu, Maosong Sun, and Jie
Zhou. WebCPM: Interactive web search for Chinese long-form question answering. In Anna
Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages
8968–8988, Toronto, Canada, July 2023. Association for Computational Linguistics. doi:
10.18653/v1/2023.acl-long.499. URL https://aclanthology.org/2023.acl-long.499 .
[37] Prashanth Radhakrishnan, Jennifer Chen, Bo Xu, Prem Ramaswami, Hannah Pho, Adriana
Olmos, James Manyika, and R. V . Guha. Knowing when to ask – bridging large language
models and data, 2024. URL https://arxiv.org/abs/2409.13741 .
[38] Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Lora Aroyo, Michael Collins, Dipanjan
Das, Slav Petrov, Gaurav Singh Tomar, Iulia Turc, and David Reitter. Measuring attribution in
natural language generation models, 2022. URL https://arxiv.org/abs/2112.12870 .
12

[39] Mersedeh Sadeghi, Daniel Pöttgen, Patrick Ebel, and Andreas V ogelsang. Explaining the
unexplainable: The impact of misleading explanations on trust in unreliable predictions
for hardly assessable tasks. page 36 – 46, 2024. doi: 10.1145/3627043.3659573. URL
https://www.scopus.com/inward/record.uri?eid=2-s2.0-85197860661&doi=10.
1145%2f3627043.3659573&partnerID=40&md5=b64ac26f3d944dce90a47ecd2709bb99 .
Cited by: 0; All Open Access, Bronze Open Access.
[40] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version
of bert: smaller, faster, cheaper and lighter. ArXiv , abs/1910.01108, 2019.
[41] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettle-
moyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach
themselves to use tools, 2023. URL https://arxiv.org/abs/2302.04761 .
[42] Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-Wei Chang. ASQA: Factoid ques-
tions meet long-form answers. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang,
editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language
Processing , pages 8273–8288, Abu Dhabi, United Arab Emirates, December 2022. As-
sociation for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.566. URL
https://aclanthology.org/2022.emnlp-main.566/ .
[43] Ne¸ set Özkan Tan, Niket Tandon, David Wadden, Oyvind Tafjord, Mark Gahegan, and Michael
Witbrock. Faithful reasoning over scientific claims. Proceedings of the AAAI Symposium
Series , 3(1):263–272, May 2024. ISSN 2994-4317. doi: 10.1609/aaaiss.v3i1.31209. URL
http://dx.doi.org/10.1609/aaaiss.v3i1.31209 .
[44] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. Minilm: Deep
self-attention distillation for task-agnostic compression of pre-trained transformers, 2020.
[45] Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. Adaptive chameleon or stubborn
sloth: Revealing the behavior of large language models in knowledge conflicts, 2024. URL
https://arxiv.org/abs/2305.13300 .
[46] Xiang Yue, Boshi Wang, Ziru Chen, Kai Zhang, Yu Su, and Huan Sun. Automatic evaluation of
attribution by large language models, 2023. URL https://arxiv.org/abs/2305.06311 .
[47] Guido Zuccon, Bevan Koopman, and Razia Shaik. Chatgpt hallucinates when attributing
answers. In Proceedings of the Annual International ACM SIGIR Conference on Research and
Development in Information Retrieval in the Asia Pacific Region , SIGIR-AP ’23, page 46–51,
New York, NY , USA, 2023. Association for Computing Machinery. ISBN 9798400704086. doi:
10.1145/3624918.3625329. URL https://doi.org/10.1145/3624918.3625329 .
13

Supplementary Material
Ben-u-ron
pamphletDo I
Attribute?
YesBody weight: Up to 50kg
Max. daily dosage: 2000mg
Body weight: Over 50kg
Max. daily dosage: 3000mgNo
May exceed
daily dosageStays within the
daily dosageXX ◡
Figure 3: ChatGPT use case for medicine. The user is told not to exceed 4000mg of paracetamol in
24 hours. We checked different sources that say the daily maximum dosage is: (1) 3000mg if you
are over 50kg1, (2) 2000mg2, (3) the minimum between 4000mg and 60mg/kg3, or (4) 4000mg4.
Following GPT’s instructions may lead to an overdose. The remainder of the answer is irrelevant for
this example. Websites from the USA seem to provide higher maximum dosage values (4000mg)
than European websites (from 2000mg to 4000mg). This might explain why, although this brand of
paracetamol has a 3000mg limit, GPT-4o recommends a 4000mg limit.
> INPUT
Do not exceed 4000 mg (4 grams) in 24 hours.
< OUTPUT (Ben-u-ron’s Leaflet1)
A posologia habitual para o paracetamol é: Peso corporal Idade Dose única por toma
Dose máxima d iária Até 50 Kg Adolescentes entre os 12 e 15 anos 1 cápsula Até 4 cápsulas
(equivalente a 2000 mg de paracetamol) Acima de 50 Kg Adolescentes entre os 16 e 18
anos e Adultos 1-2 cápsulas Até 6 cápsulas (equivalente a 3000 mg de paracetamol) Dose
máxima diária A dose máxima diária de Paracetamol não deve exceder 3 g/dia.
Translation Note: The system is providing a Table describing the daily dosage limits
per age and weight. Here, we can read "Adults should take up to 6 pills (equivalent to
3000mg of paracetamol)", which goes against GPT’s answer .
< OUTPUT (NHS Fife’s Leaflet4)
A maximum of 8 tablets (4000mg or 4g) should be taken in a 24 hour period.
Table 6: Application of our system to verify GPT-4o’s output using both the Ben-u-ron (Portuguese)
and NHS Fife (English) leaflets. The sentence is assigned to a quote that negates GPT’s answer
(Ben-u-ron) and a quote that verifies the answer (NHS Fife).
1https://ben-u-ron.pt/wp-content/uploads/2023/08/ben-u-ron-500mg-capsulas_FI.pdf
2https://www.medicines.org.uk/emc/product/5164/pil
3https://eliveragroup.com/products/ben-u-ron-1-000-mg-paracetamol-tablets
4https://www.nhsfife.org/media/ucgk29u1/paracetamol-_leaflet_pt_a5_v2.pdf
14

Table 7: Features extracted using the spacy ,textstat ,textblob , and nltkPython libraries.
ID Description
0 Fraction of unique words (lexical diversity).
1 Named entity density: fraction of tokens that are named entities.
2 Syntactic parse tree depth.
3 Flesch reading ease score (higher = easier to read).
4 Shannon entropy of character or word distribution.
5 Average number of WordNet synsets per word (semantic ambiguity).
6 Ratio of nouns to verbs.
7 Proportion of stopwords in the sentence.
8 Ratio of punctuation marks to total words or characters.
9 Average number of characters per word.
10 Total number of syllables in the sentence.
11 Total number of words.
12 Number of unique (distinct) words.
13 Average bigram probability (lower = less expected).
14 Average trigram probability (lower = less expected).
15 Ratio of pronouns to total words.
16 Ratio of verbs in passive voice.
17 Binary indicator if the sentence is a named entity.
18 SMOG index (grade level).
19 Coleman–Liau index (readability score).
20 Automated Readability Index.
21 Dale–Chall readability score.
22 Linsear Write readability formula.
23 Gunning Fog index.
15

Figure 4: Real-world application using ChatGPT 4o (part 1 of 2): ChatGPT’s webapp and answer.
16

> INPUT Sentence:
The paper studies how well large language models (LLMs) can solve problems that
require multiple steps of reasoning.
< OUTPUT Quotes:
Towards reliable and fluent large language models: Incorporating feedback learning
loops in qa systems.
> INPUT Sentence:
It focuses on a task called “iterated reasoning,” where a model has to think about
what others might do or think, and respond accordingly.
< OUTPUT Quotes:
This kind of attribution is essential for interpreting the model, providing insights
into whether the model is capturing task-relevant patterns or merely memorizing data.
> INPUT Sentence:
A common example of this is in games where players must guess what others will choose,
like the “beauty contest” game.
< OUTPUT Quotes:
To train the model in searching and using evidence, a preference model is used based
on human judgments.
> INPUT Sentence:
The authors test different LLMs, including GPT-4 and Claude 2, on such reasoning
tasks.
< OUTPUT Quotes:
These operations significantly enhance the attribution accuracy of LLMs in complex
knowledge-intensive tasks, improving their reasoning ability and knowledge utilization.
> INPUT Sentence:
They compare the models to humans to see if they behave similarly.
< OUTPUT Quotes:
While human evaluations provide in-depth insights, their costly and time-consuming
nature emphasizes the growing appeal for automated methods.
> INPUT Sentence:
Results show that LLMs can sometimes do this kind of reasoning, but they don’t always
follow human-like patterns.
< OUTPUT Quotes:
The internal state of an LLM knows when its lying.
> INPUT Sentence:
The paper also looks at how changing the way questions are asked affects model
performance.
< OUTPUT Quotes:
This performance hints at the ability of models to either memorize or reason through
patterns present in the data.
> INPUT Sentence:
In some cases, using chain-of-thought (explaining steps) or role-playing helps the
models reason better.
< OUTPUT Quotes:
Additionally, it plays a role in interpretability, aiding users in understanding how
the model arrives at certain conclusions by revealing the sources of information.‘ This
alignment with interpretability objectives helps in making the model’s decision-making
process more transparent and comprehensible.
...
Table 8: Real-world application using ChatGPT 4o (part 2 of 2): Attribution using our system. Unlike
the test cases using Perplexity and DeepSeek, our system seems to have issues matching the output
from GPT to the paper. Upon further inspection, the authors do not mention the word "Claude" nor
talk about games, leading us to believe these two sentences are hallucinations.
17

Figure 5: Real-world application using Perplexity (part 1 of 3). Perplexity’s webapp and answer.
18

Figure 6: Real-world application using Perplexity (part 2 of 3). Perplexity’s webapp and answer.
19

...
> INPUT Sentence:
Attribution is important because it helps users check if the model’s answers are
accurate and trustworthy
< OUTPUT Quotes:
However, the primary purposes of attribution include enabling users to validate the
claims made by the model, promoting the generation of text that closely aligns with
the cited sources to enhance accuracy and reduce misinformation or hallucination, and
establishing a structured framework for evaluating the completeness and relevance of
the supporting evidence in relation to the presented claims.
> INPUT Sentence:
The paper explains that LLMs can sometimes make up facts ("hallucinations") or provide
incorrect or unreliable references, especially if their training data contains errors
or biases
< OUTPUT Quotes:
LLMs might link content to irrelevant or incorrect sources.
...
> INPUT Sentence:
Direct model-driven attribution: The model itself tries to provide references for
its answers, but these are often incomplete or inaccurate.
< OUTPUT Quotes:
In direct model-driven attribution way, the reference document is derived from model
itself and is used to cite generated answer.
> INPUT Sentence:
Post-retrieval answering: The model retrieves information from external sources and
then answers, but it can be hard to tell if the answer really matches the sources.
< OUTPUT Quotes:
2.Post-retrieval answering: This approach is rooted in the idea of explicitly
retrieving information and then letting the model answer based on this retrieved data.
> INPUT Sentence:
Post-generation attribution: The model first generates an answer, then searches for
references to support it, and edits the answer if needed.
< OUTPUT Quotes:
3.Post-generation attribution: The system first provides an answer, then conducts a
search using both the question and answer for attribution.
...
> INPUT Sentence:
Comprehensive: Every claim should have a reference (high recall).
< OUTPUT Quotes:
2.Sufficiency attribution or citation (high precision): Every reference should
directly support its associated claim or statement.
> INPUT Sentence:
Sufficient: Every reference should directly support the claim (high precision)
< OUTPUT Quotes:
2.Sufficiency attribution or citation (high precision): Every reference should
directly support its associated claim or statement.
...
Table 9: Real-world application using Perplexity (part 3 of 3): Attribution using our system. Our
system seems to work well on the answer provided by Perplexity. Around half of the attribution log
was removed due to size constraints.
20

Figure 7: Real-world application using DeepSeek (part 1 of 2): DeepSeek’s webapp and answer.
21

...
> INPUT Sentence:
Direct model-driven attribution: The model itself tries to provide sources, but this
can be unreliable.
< OUTPUT Quotes:
With these requirements in mind, we can break down the main ways models handle
attribution into three types (see examples in Figure 3): 1.Direct model-driven
attribution: The large model itself provides the attribution for its answer.
> INPUT Sentence:
Post-retrieval answering: The model fetches information first, then answers based on
it, but this can mix up internal and external knowledge.
< OUTPUT Quotes:
2.Post-retrieval answering: This approach is rooted in the idea of explicitly
retrieving information and then letting the model answer based on this retrieved data.
> INPUT Sentence:
Post-generation attribution: The model gives an answer first, then finds sources to
support it, but this can lead to incomplete or wrong citations.
< OUTPUT Quotes:
In post-generation attribution way, an answer is first generated then citation and
attribution are purposed.
...
> INPUT Sentence:
Errors in attribution, such as citing the wrong sources or missing key details.
< OUTPUT Quotes:
This form of attribution is particularly lacking in high-risk professional fields
such as medicine and law, with research revealing a significant number of incomplete
attributions (35% and 31%, respectively); moreover, many attributions were derived from
unreliable sources, with 51% of them being assessed as unreliable by experts (Malaviya
et al., 2023).
...
> INPUT Sentence:
The paper includes examples and tables to show different types of errors and compares
datasets and methods for evaluating attribution.
< OUTPUT Quotes:
Table 4: List of different attribution errors types and example instance.
> INPUT Sentence:
The goal is to help researchers make LLMs more trustworthy and easier to understand.
< OUTPUT Quotes:
The aim is to make sure LLMs acknowledge sources without hindering their creative
potential.
> INPUT Sentence:
The authors note that this field is still new, and they share a GitHub repository to
track ongoing research.
< OUTPUT Quotes:
We believe that this field is still in its early stages; hence, we maintain
a repository to keep track of ongoing studies at https://github.com/HITsz-TMG/
awesome-llm-attributions .
Table 10: Real-world application using DeepSeek (part 2 of 2): Attribution using our system. Our
system seems to work well on the answer provided by DeepSeek, correctly attributing the types of
attribution and the conclusions. Part of the answer was removed for the Table to fit on a single page.
22
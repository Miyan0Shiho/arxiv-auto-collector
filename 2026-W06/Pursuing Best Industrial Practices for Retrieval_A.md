# Pursuing Best Industrial Practices for Retrieval-Augmented Generation in the Medical Domain

**Authors**: Wei Zhu

**Published**: 2026-02-03 10:37:42

**PDF URL**: [https://arxiv.org/pdf/2602.03368v1](https://arxiv.org/pdf/2602.03368v1)

## Abstract
While retrieval augmented generation (RAG) has been swiftly adopted in industrial applications based on large language models (LLMs), there is no consensus on what are the best practices for building a RAG system in terms of what are the components, how to organize these components and how to implement each component for the industrial applications, especially in the medical domain. In this work, we first carefully analyze each component of the RAG system and propose practical alternatives for each component. Then, we conduct systematic evaluations on three types of tasks, revealing the best practices for improving the RAG system and how LLM-based RAG systems make trade-offs between performance and efficiency.

## Full Text


<!-- PDF content starts -->

Pursuing Best Industrial Practices for Retrieval-Augmented Generation in
the Medical Domain
Wei Zhu1,∗
1University of Hong Kong, Hong Kong, HK, China
Abstract
While retrieval augmented generation (RAG)
has been swiftly adopted in industrial applica-
tions based on large language models (LLMs),
there is no consensus on what are the best
practices for building a RAG system in terms
of what are the components, how to organize
these components and how to implement each
component for the industrial applications, es-
pecially in the medical domain. In this work,
we first carefully analyze each component of
the RAG system and propose practical alterna-
tives for each component. Then, we conduct
systematic evaluations on three types of tasks,
revealing the best practices for improving the
RAG system and how LLM-based RAG sys-
tems make trade-offs between performance and
efficiency.
1 Introduction
Large Language Models (LLMs) have trans-
formed the way people access information online,
shifting from conventional web searches to direct
interactions with chatbots. Recently, there has been
a rapid development of commercial LLMs by in-
dustry players, demonstrating state-of-the-art per-
formance in question answering (QA) across both
general and medical domains (Achiam et al., 2023;
Anil et al., 2023; Singhal et al., 2023a,b; Nori et al.,
2023; Cui et al., 2023; Wang et al., 2024; Wen-
jing Yue and Wang, 2023; Zhang et al., 2023e;
Zhao et al., 2023; Xu et al., 2023; Ding et al., 2022;
Xin et al., 2024; Qin et al., 2023; Zhu et al., 2023;
Zhu et al., 2023a,b, 2021a; Li et al., 2023a; Zhu
et al., 2023c; Zhang et al., 2023b; Zhu et al., 2023e;
Guo et al., 2021; Zhu et al., 2021b; Zheng et al.,
2023; Sun et al., 2020; Zhang et al., 2023c,d; Wang
et al., 2023a; Zhu et al., 2019a; Zhu, 2021a; Zhang
et al., 2021; Wang et al., 2020; Li et al., 2025;
Leong et al., 2025; Zhang et al., 2025; Yin et al.,
∗Corresponding author. For any inquiries, please contact:
michaelwzhu91@gmail.com.2024; Zhu, 2026a,b). Despite these achievements,
a notable issue is that LLMs can sometimes pro-
duce responses that, while plausible, are factually
inaccurate, a problem referred to as hallucination
(Ji et al., 2023; Rawte et al., 2023). Additionally,
the training data for these models may not include
the most up-to-date information, such as recent
news or the latest scientific and medical research.
These shortcomings present significant challenges
and potential risks, particularly in critical areas
like finance, personal assistance, bio-medicine and
healthcare (Tian et al., 2024b; Hersh, 2024; Tian
et al., 2024b; Hersh, 2024; Zhu et al., 2024; Zhu
and Tan, 2023; Liu et al., 2022; Xie et al., 2024;
Cui et al., 2023; Zheng et al., 2024; Zhu et al.,
2023d; Gao et al., 2023a; Zuo et al., 2022; Zhang
et al., 2022; Sun et al., 2022; Zhu et al., 2021c;
Zhu, 2021b; Li et al., 2019; Zhu et al., 2019c,b;
Zhou et al., 2019; Zhang et al., 2025; Wang et al.,
2025; Liu et al., 2025; Yi-Ge et al., 2024; Tian
et al., 2024a).
To tackle this issue, retrieval-augmented genera-
tion (RAG) utilizes current and trustworthy docu-
ment collections to boost the capabilities of Large
Language Models (LLMs), potentially overcom-
ing various challenges in the field (Lewis et al.,
2020; Gao et al., 2023b; Zhao et al., 2024). By
anchoring LLMs’ reasoning in these retrieved doc-
uments, RAG can also improve their explainabil-
ity and transparency. As illustrated in Figure 1, a
standard RAG system for the open-domain or med-
ical domain question answering typically involves
several key steps: (a) query classification, which
assesses whether a given input requires retrieval;
(b) construction of the retrieval corpus, encompass-
ing processes like chunking and indexing; (c) the
retrieval process, which identifies the most rele-
vant information based on the search input; and (d)
response generation, where prompting strategies
are used to guide the LLMs. The complexity and
challenge lie in the variability of approaches forarXiv:2602.03368v1  [cs.CL]  3 Feb 2026

each step. For instance, when retrieving pertinent
documents for an input, multiple techniques such
as query rewriting (Ma et al., 2023) and pseudo-
response generation (Gao et al., 2022) can be ap-
plied to enhance the original query for more effec-
tive searching. Thus, one central research question
raises:
RQ1.What is the best practice for building a RAG
system, especially for the bio-medical tasks?
This study is designed to search for the best prac-
tices of RAG systems through comprehensive ex-
perimentation on both open-domain and medical
domain tasks. Given the impracticality of evalu-
ating every possible combination of methods, we
employ a three-step strategy to pinpoint the most
effective RAG practices. Initially, we examine rep-
resentative methods for each step or module within
the RAG process. Then, we optimize one module at
a time while keeping the others constant, iterating
through all modules to establish an optimal config-
uration for the RAG system. Lastly, we showcase
the experimental outcomes of this optimal RAG
setup and also present variations by altering one
module at a time, thus generating a series of RAG
configurations. Based on these results, we propose
several strategies for deploying RAG that effec-
tively balance performance and efficiency.
In summary, our contributions are two-fold:
•We thoroughly investigate existing approaches
for different modules of the RAG system.
•We have conducted extensive experiments to
investigate many combinations for the RAG
settings and identify the optimal RAG prac-
tices.
2 Related work
2.1 Retrieval-augmented generation
Retrieval-Augmented Generation (RAG) was
proposed by Lewis et al. (2020) to enhance the gen-
eration performance on knowledge-intensive tasks
by integrating retrieved relevant information. In the
LLM era led by OpenAI’s ChatGPT and GPT-4,
RAG not only mitigates the problem of halluci-
nations as LLMs are grounded on given contexts
but can also provide up-to-date knowledge that the
LLMs might not encode (Gao et al., 2023b; Zhao
et al., 2024). Many recent studies have been de-
voted to improving upon the vanilla RAG workflow
by either designing novel retrieval and generationmechanisms (Borgeaud et al., 2022; Zhang et al.,
2023a; Ram et al., 2023; Jiang et al., 2023), or
incorporating pre-training and fine-tuning for im-
proving LLMs’ capabilities in RAG (Zhang et al.,
2024; Siriwardhana et al., 2023; Xue et al., 2024).
In specific domains like bio-medicine, current
systematic evaluations of LLMs typically focus on
the vanilla LLMs without RAG (Zhu et al., 2023;
Singhal et al., 2023a,b; Nori et al., 2023; Chen et al.,
2023; Saab et al., 2024). There has been a series
of works on how RAG can help to improve LLMs’
capabilities in tasks like clinical decision-making,
scientific literature analysis, and information ex-
traction (Frisoni et al., 2022; Naik et al., 2021;
Xiong et al., 2024; Lála et al., 2023; Jin et al., 2024;
Zakka et al., 2024; Jeong et al., 2024; Wang et al.,
2023b). However, (a) comprehensive evaluation
that contains a variety of tasks is lacking, and (b)
systematic investigations on what are the best prac-
tices to build a RAG system in these domains, such
as the prompt strategies, are lacking, hindering its
further industrial applications. Our work comple-
ments the existing literature by investigating best
practices for the LLM-based RAG system.
3 RAG system
In this section, we detail the components of the
RAG system, and the approaches available. Figure
1 presents the RAG system with different methods
for each component.
3.1 Query classification
In an RAG system, not all queries necessitate re-
trieval augmentation and can be effectively handled
by the inherent capabilities of LLMs. Although
RAG improves information accuracy and mitigates
hallucinations, it also introduces increased latency
and computational complexity. Thus, our approach
begins with classifying queries to ascertain whether
retrieval is needed. Queries that require additional
information go through the RAG modules, while
those that do not are directly processed and an-
swered by the LLMs.
Therefore, we propose the classifying task of
determining if a query needs retrieval, and we train
a classifier to automate this decision-making pro-
cess. To build the dataset for this task, we label the
query in the dataset asneed RAGif its answer’s log-
likelihood function can increase when conditioned
on retrieved documents. Otherwise, the query will
be labelednot need RAG.

Figure 1: The workflow of retrieval-augmented generation considered in this study. We investigate the contribution
of each module and provide insights into optimal RAG practices through extensive experiments. The methods
selected for the best practice are underlined .
3.2 Chunking
It is essential to properly chunk the documents
into smaller segments to enhance retrieval preci-
sion and avoid length issues in LLMs. In this study,
we use the sentence-level chunking strategy (Gao
et al., 2023b), which is to break documents into
sentences and fill the sentences into chunks. Dur-
ing chunking, an important aspect to consider is the
chunk size. Chunk size significantly impacts per-
formance. Larger chunks provide more context but
could introduce irrelevant information and increase
latency. Smaller chunks improve retrieval recall
and efficiency but may lack sufficient context. This
study considers the chunk size Lc= 256 by default.
Moreover, we must consider the following aspect
of chunking, which determines how neighboring
chunks are connected.
Chunking techniquesWe consider the fol-
lowing three chunking techniques: (a) Vanilla
chunking, which chunks the documents into non-
overlapping segments of length Lc. (b) Small2big
chunking, which uses a small segment from vanilla
chunking for retrieval, but a longer segment con-
taining the small one will be returned. We set the
length of the smaller chunk to Ls
c=L c/2, and
the larger chunk to Ll
c=L c. (c) Sliding-window
chunking, which chunks the documents into over-
lapping segments. The overlap length is set to
Lo=L c/4.
Intuitively, the latter two techniques improve
retrieval quality over the vanilla chunking method
by organizing chunks’ relationships.3.3 Document indexing
After long documents are chunked into smaller
segments, we save them in a database with an index
built for efficient retrieval. Based on how the docu-
ments are indexed, the document indexing methods
are: (a) Sparse indexing, which utilizes the tech-
nique of inverted index. It first tokenizes the docu-
ment segments into words or tokens and then builds
an inverted index where each key term points to a
list of documents that contain that term. (b) Dense
indexing, which assumes the document segments
are transformed to semantic vectors via an embed-
ding model. Approximate nearest neighbor (ANN)
index (Abbasifard et al., 2014) like HNSW or IVF
is the possible method for indexing such a vector
database. (c) Hybrid indexing, which is to build
both indexes on the corpus.
Embedding modelsSince we consider dense
indexing, a high-quality embedding model is cen-
tral to the retrieval performance. In this work, we
consider a series of embedding models, both open-
domain and in-domain, with base sizes: (a) BGE-
base (Xiao et al., 2023). (b) GTE-base (Li et al.,
2023b). (c) MedCPT (Jin et al., 2023), a medi-
cal document representation model initialized from
PubMedBERT (Gu et al., 2021), and further pre-
trained on the retrieval tasks curated with PubMed.
3.4 Retrieval
Given a user’s query, the retrieval module selects
the top kmost relevant documents from a pre-built
corpus, based on the similarity between the query
and the documents. The generation model subse-

quently uses these selected documents to formulate
an appropriate response to the query. How the cor-
pus is indexed directly determines the retrieval al-
gorithm: (a) Best Matching 25 (BM25) (Robertson
et al., 2009) algorithm is used for sparse index-
ing. (b) ANN search is used for dense retrieval.
(c) hybrid search is used if the corpus uses hybrid
indexing. Sophistic toolkits are available to imple-
ment the first two. To implement a hybrid search,
one first uses the former two methods and then
combines the search results with a 3:1 weight ra-
tio for similarity scores to rerank the two lists of
retrieved documents.
Query augmentation strategy for searchIn a
standard RAG system, the user query is input into
the search engine to retrieve relevant documents.
However, the original queries often have poor ex-
pression and lack semantic information (Gao et al.,
2022), negatively impacting the retrieval results.
Thus, we evaluate four approaches to search inputs:
(a) Vanilla input, that is, directly utilizing the user
query as search input. (b) Query rewriting, that
is, refining queries and transforming the queries to
sub-questions so that they can better match relevant
documents. (c) Pseudo-response generation, that
is, the model first generates a response, which will
be concatenated to the user query.
3.5 Response generation
prompting strategyWhen the retrieval proce-
dure returns a list of referential documents, an LLM
will concatenate these contents at the front of the
prompt and generate the final response. Other than
relevant documents, the prompt contains instruc-
tions that reflect the prompting strategies: (a) Di-
rect answer (DA): Given the question, the prompt
asks the LLM to output the answer directly. (b)
Chain-of-thought (COT) (Wei et al., 2022) explic-
itly asks the LLM to think step by step and demon-
strate the intermediate outputs. (c) COT-Refine.
Building on COT and Self-Refine(Madaan et al.,
2024), this strategy assumes a response without
RAG has been generated, and this prompt asks the
model to reflect on this response, utilize the re-
trieved documents to make necessary corrections,
and finally draft a new response.
4 Experiments
We systematically evaluate the RAG workflow
on a series of knowledge-intensive tasks, provid-
ing a multidimensional analysis of different com-ponents in RAG and paving the way for optimal
practices for implementing RAG.
4.1 Experimental settings
Evaluation datasetsWe evaluate the RAG sys-
tems on three benchmarks, investigating how RAG
helps the LLMs in open-domain or in-domain
question-answering tasks and information extrac-
tion tasks: (a) MMLU (Hendrycks et al., 2020). (b)
PubMedQA (Jin et al., 2019). (c) PromptNER,
a mixture of samples from multiple named en-
tity recognition tasks, including CoNLL-03 (Sang
and De Meulder, 2003), OntoNotes 5.01and
BioNLP2004 (Collier and Kim, 2004). Introduc-
tions and dataset statistics are in Appendix A.
Now we elaborate on the steps we take to con-
struct the dataset for query classification:
(i) Query collection. We collects: (a) 10k samples
from the dev set and test set of the original MMLU
datasets, with no overlapping with the test set for
RAG evaluation. (b) 10k samples from the auto-
matic labeled set of PubMedQA. (c) 7.9k dev set
from PromptNER. Thus, we have a collection of
27.9k samples with query-response pairs.
(ii) Query labeling. We conduct automatic labeling
of the above dataset using the LlaMA-2 13B model
and log-likelihood function. For a sample with
a query-response pair, we first calculates the log-
likelihood of the response text conditioned only
on the query, that is, l0=LLL(response|query) .
Here, LLL() is the log-likelihood function cal-
culated with the given LlaMA-2 13B backbone.
We search the corpus with hybrid search (as in
Section 3.4), and retrieve k= 8 documents,
denoted as docs. Conditioned on the docs and
query , the response’s log-likelihood becomes l1=
LLL(response|query,docs) . The query is labeled
"need RAG" (or label 1) if l1−l0>0, and "not
need RAG" (label 0) otherwise. Using the above
procedure, there are 17.9% queries with the "need
RAG" label (denoted as label 1).
(iii) Dataset split. The automatically labeled
queries are split into a 24k:2k:1.9k train/dev/test
split.
Evaluation metricsFor the MMLU and Pub-
MedQA tasks, we will directly consider the correct-
ness of the final answers. Thus, we report accuracy
(denoted as acc).
For the PromptNER task, the output response
text will first be parsed and transformed to a json
1https://catalog.ldc.upenn.edu/LDC2013T19

Model backbone Acc F1
BERT-base 0.586 0.341
RoBERTa-base 0.638 0.382
DeBERTa-base 0.637 0.354
Table 1: The performance of query classification.
instance. If the response can not be parsed to json,
then we consider the prediction as a null list. We
adopt the instance-level strict micro-F1 following
Zhu et al. (2023), that is, the model predicts an
entity correctly if and only if it correctly predicts
all its components.
Other than the performance matrices on the eval-
uation datasets, we also measure the efficiency of
the RAG systems by the average latency (in sec-
onds (s)) for completing the response for a test
sample.
Settings for query classificationWe use the
query classification dataset introduced above to
build a query classifier. The pre-trained backbones
we consider are: (a) BERT-base (Devlin et al.,
2019), (b) RoBERTa-base (Liu et al., 2019), and (c)
DeBERTa-base (He et al., 2020). For fine-tuning
these models, we utilize the HuggingFace Trans-
formers package (Wolf et al., 2019), with batch
size 64, learning rate 2e-5, warm-up steps 100,
AdamW optimizer (Loshchilov and Hutter, 2019).
The other hyper-parameters are kept the same with
Transformers. For every 200 optimization steps, we
run evaluations on the dev set and save the model
checkpoint. The checkpoint with the best dev accu-
racy is used as the final fine-tuned model to make
predictions on the test set. And this checkpoint is
used as a part of the RAG system.
Retrieval corpusThe retrieval corpus consists
of the following sources: (a) Wikipedia (English)
corpus2, a large-scale open-source encyclopedia
containing 6.5 million documents for world knowl-
edge. (b) PubMed3is the most widely used litera-
ture resource, containing 23.9 million biomedical
articles’ titles and abstracts. (c) a proprietary cor-
pus containing 1.3 million books or documents
from science, education, and medicine. When
processing the retrieval corpus, we set chunk size
Lc= 256.
Retrieval settingsFor sparse indexing, we uti-
lize the ElasticSearch toolkit4with the BM25 algo-
2https://en.wikipedia.org/wiki/Wikipedia:
Database_download
3https://pubmed.ncbi.nlm.nih.gov/
4https://www.elastic.co/rithm for search. For dense indexing, we utilize the
Faiss5toolkit to index the document vectors and
implement efficient vector search. The top k= 8
document segments are retrieved and concatenated
to the input prompt (if using RAG for response
generation) for each query.
LLM backboneOur experiments uses the most
recent open-sourced LLM, LlaMA-2-chat 13B re-
leased by Meta (Touvron et al., 2023). After receiv-
ing a prompt or instruction, all the predictions are
generated using the model’s pretrained language
modeling head (LM head). For decoding during
inference, we use beam search with beam size 3.
Strategy to determine the optimal practiceTo
begin with, we consider the following settings for
the RAG system: sliding-window chunking as the
chunking strategy, semantic vector indexing for
indexing the retrieval corpus, BGE-base as embed-
ding model, pseudo-response generation for query
augmentation, COT for prompting. Following the
framework depicted in Figure 1, we optimize indi-
vidual modules step-by-step, and select the most
effective option among the possible choices. This
iterative process continued until we could not im-
prove the average task score.
Settings for the RAG systemWith the help
of the above strategy, we have locked down the
optimal practice for the RAG system: small2big
chunking as the chunking strategy, hybrid index-
ing for indexing the retrieval corpus, BGE-base as
the embedding model, using query classification,
pseudo-response generation for query augmenta-
tion, COT-Refine for prompting. We denote this
setting as BP-RAG . We also consider the following
settings to demonstrate the superiority of BP-RAG :
(a) No RAG, which is not to use RAG at all. This
setting is presented as a sanity check. (b) RAG_1,
which substitute small2big chunking in BP-RAG
to the vanilla chunking. (c) RAG_2, which substi-
tute small2big chunking in BP-RAG to the sliding-
window chunking. (d) RAG_3, which substitute
hybrid indexing in BP-RAG to the sparse indexing.
As a result, RAG_3 will use BM25 for search and
not use an embedding model. (e) RAG_4, which
substitute hybrid indexing in BP-RAG to the dense
indexing. (f) RAG_5, which substitute BGE-base
inBP-RAG to MedCPT, an in-domain embedding
model. It is interesting to see whether MedCPT
still performs well on open-domain retrieval. (g)
RAG_6, which substitute BGE-base in BP-RAG to
5https://github.com/facebookresearch/faiss

Method RAG Setting MMLU PubMedQA PromptNER Avg score Avg latency
BP-RAG BP-RAG59.756.925.8 47.514.3
No RAG - 49.3 43.4 20.6 37.8 10.8
RAG_1 + vanilla chunking 59.3 55.9 25.2 46.7 14.1
RAG_2 + sliding-window chunking 59.7 56.1 25.4 47.1 14.2
RAG_3 + sparse indexing 53.1 47.3 22.5 40.9 14.2
RAG_4 + dense indexing 58.9 55.7 25.3 46.6 14.3
RAG_5 + MedCPT 55.657.124.1 45.6 14.3
RAG_6 + GTE-base 59.3 56.2 25.7 47.1 14.2
RAG_7 - query classification 58.5 55.8 25.1 46.5 20.7
RAG_8 + query rewriting 57.4 54.5 23.8 45.2 11.4
RAG_9 + vanilla query 56.2 51.6 22.6 43.5 11.0
RAG_10 + COT 58.2 55.8 24.4 46.1 14.2
RAG_11 + direct answering 54.9 51.7 21.9 42.8 3.7
Table 2: Results of different RAG settings. The average score is calculated by averaging the scores of all tasks,
while the average latency is measured in seconds per query.
GTE-base. (h) RAG_7, which does not employ the
query classification module. That is, it will retrieve
documents for any given query. (i) RAG_8, which
substitute pseudo-response generation in BP-RAG
to query rewriting. (j) RAG_9, which substitute
pseudo-response generation in BP-RAG to vanilla
query. (k) RAG_10, which substitute COT-Refine
inBP-RAG to COT. (l) RAG_11, which substitute
COT-Refine in BP-RAG to direct answering.
4.2 Results
The results of the RAG settings are presented
in Table 2, while the results for the query classi-
fication tasks are reported in Table 1. Based on
the experimental results presented in Table 2, the
following observations can be made:
(a) The benefit of using RAG. Compared with
not using RAG and making responses directly, BP-
RAG significantly improves the performance by
25.6% on average. However, one can not ignore
that the latency of BP-RAG is 32.4% higher than
No RAG.
(b) Query classification module. Table 1 reports
the query classification performance on the query
classification dataset we developed. DeBERTa-
base (He et al., 2020) outperforms BERT-base (De-
vlin et al., 2019) and RoBERTa-base (Liu et al.,
2019) by achieving an accuracy of 65.3%. Accord-
ing to Table 2, this module is beneficial for both
the effectiveness and efficiency of the RAG system,
leading to an improvement in the average score
from 46.5% to 47.5% and a reduction in latency
time from 20.7 to 14.3 seconds per query. The op-
erations in this module do not significantly increase
the overall latency of the system since classifyingone query with DeBERTa-base can be done in less
than 20 ms.
(c) Chunking strategy. Table 2 demonstrates
that the small2big strategy slightly outperforms the
other two chunking strategies, demonstrating the
benefit of retaining contextual information in the
retrieved documents.
(d) Indexing Module. The experimental results
show that the hybrid indexing strategy attains the
highest scores. Hybrid indexing combines the
search results from sparse and dense indices, mak-
ing the retrieved documents more informative.
(e) Embedding models. Among the three base-
sized embedding models, the BGE-base model
works the best with the RAG. The MedCPT model
is further pre-trained on the PubMed corpus, mak-
ing it especially beneficial for the PubMedQA task,
but it is unsuitable for the open-domain MMLU
and PromptNER tasks.
(f) Query augmentation module. In this mod-
ule, query rewriting and pseudo-response genera-
tion significantly increase the latency. The former
increases the latency by 3.6%, while the latter in-
creases the latency by 30.1%. However, pseudo-
response generation benefits the RAG system by
helping retrieve more relevant and informative doc-
uments.
(g) Prompting module. Direct answering is
the most efficient in this module, but it signifi-
cantly underperforms compared to its two com-
petitors. From Table 2, COT-Refine outperforms
COT, demonstrating that explicitly asking the LLM
to reflect on the past response helps improve its
performance.
The experimental results demonstrate that each

module contributes uniquely to the overall perfor-
mance of the RAG system. Note that boosting
performance requires the RAG system to increase
latency in some of the modules. Thus, in certain
time-sensitive applications, industrial practitioners
can select settings different from BP-RAG, making
an informed trade-off between performance and
efficiency.
5 Conclusion
This study investigates the best practices for
implementing a retrieval-augmented generation
(RAG) system. First, we identify potential solu-
tions for each module within the RAG framework.
Second, we conduct extensive experiments to sys-
tematically assess these solutions and recommend
the most effective approach for each module. Dur-
ing this process, we demonstrate how different so-
lutions affect the RAG system’s performance and
latency. Our findings contribute to a deeper un-
derstanding of RAG systems and also provide key
insights for industrial applications.
Limitations
Despite the fact that we provide extensive ex-
periments for medical RAG on a wide collection
of tasks, the following limitations remain: (a) We
focus on open-sourced LLMs. Powerful language
models like GPT-4o, Gemini (Reid et al., 2024),
Claude-36, Grok7are not evaluated due to resource
limitation. (b) There are literature in RAG that
adopt more complicated workflow than our RAG
system (in Figure 1), such as iterative retrieval
(Zhang et al., 2023a; Jiang et al., 2023). These
methods require more LLM inference times. These
more advanced RAG strategies have not been eval-
uated in our current version, but we will address
this aspect in our updated version.
Ethical statement
Our work’s investigations on the best practices
of Retrieval-Augmented Generation (RAG) sys-
tems presents significant societal benefits alongside
critical considerations. By integrating Retrieval-
Augmented Generation (RAG) with Large Lan-
guage Models (LLMs), the RAG framework en-
hances access to reliable medical information, sup-
porting clinical decision-making and improving pa-
6https://www.anthropic.com/news/
claude-3-family.
7https://github.com/xai-org/grok-1tient outcomes through evidence-based responses.
The system’s transparency—enabled by source at-
tribution to retrieved documents—helps build trust
in AI-assisted medical applications while mitigat-
ing "black box" concerns. Our investigations and
experimental results provide useful messages to the
RAG systems in the industry.
However, these advancements also present po-
tential risks requiring proactive mitigation. Over-
reliance on AI systems could inadvertently erode
human clinical judgment, necessitating balanced
implementation where RAG serves as an assistive
tool rather than a decision-maker. Workforce impli-
cations and job displacement risks call for parallel
investments in healthcare worker retraining pro-
grams. We aim to advance LLM technologies that
ethically augment medical expertise while preserv-
ing human oversight.
References
Mohammad Reza Abbasifard, Bijan Ghahremani, and
Hassan Naderi. 2014. A survey on nearest neighbor
search methods.International Journal of Computer
Applications, 95(25).
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, et al. 2023. Gpt-4 technical report.
arXiv preprint arXiv:2303.08774.
Rohan Anil, Andrew M Dai, Orhan Firat, Melvin John-
son, Dmitry Lepikhin, Alexandre Passos, Siamak
Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng
Chen, et al. 2023. Palm 2 technical report.arXiv
preprint arXiv:2305.10403.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoff-
mann, Trevor Cai, Eliza Rutherford, Katie Milli-
can, George Bm Van Den Driessche, Jean-Baptiste
Lespiau, Bogdan Damoc, Aidan Clark, et al. 2022.
Improving language models by retrieving from tril-
lions of tokens. InInternational conference on ma-
chine learning, pages 2206–2240. PMLR.
Qingyu Chen, Jingcheng Du, Yan Hu, Vipina Kuttichi
Keloth, Xueqing Peng, Kalpana Raja, Rui Zhang,
Zhiyong Lu, and Hua Xu. 2023. Large language
models in biomedical natural language processing:
benchmarks, baselines, and recommendations.arXiv
preprint arXiv:2305.16326.
Nigel Collier and Jin-Dong Kim. 2004. Introduction to
the bio-entity recognition task at JNLPBA. InPro-
ceedings of the International Joint Workshop on Nat-
ural Language Processing in Biomedicine and its Ap-
plications (NLPBA/BioNLP), pages 73–78, Geneva,
Switzerland. COLING.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao,
Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and
Maosong Sun. 2023. Ultrafeedback: Boosting lan-
guage models with high-quality feedback.ArXiv,
abs/2310.01377.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. BERT: Pre-training of
deep bidirectional transformers for language under-
standing. InProceedings of the 2019 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies, Volume 1 (Long and Short Papers), pages
4171–4186, Minneapolis, Minnesota. Association for
Computational Linguistics.
Ning Ding, Yujia Qin, Guang Yang, Fu Wei, Zong-
han Yang, Yusheng Su, Shengding Hu, Yulin Chen,
Chi-Min Chan, Weize Chen, Jing Yi, Weilin Zhao,
Xiaozhi Wang, Zhiyuan Liu, Haitao Zheng, Jianfei
Chen, Yang Liu, Jie Tang, Juan Li, and Maosong
Sun. 2022. Delta tuning: A comprehensive study of
parameter efficient methods for pre-trained language
models.ArXiv, abs/2203.06904.
Giacomo Frisoni, Miki Mizutani, Gianluca Moro, and
Lorenzo Valgimigli. 2022. Bioreader: a retrieval-
enhanced text-to-text transformer for biomedical lit-
erature. InProceedings of the 2022 conference on
empirical methods in natural language processing,
pages 5770–5793.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2022. Precise zero-shot dense retrieval without rele-
vance labels.arXiv preprint arXiv:2212.10496.
Xiangxiang Gao, Wei Zhu, Jiasheng Gao, and Congrui
Yin. 2023a. F-pabee: Flexible-patience-based early
exiting for single-label and multi-label text classifica-
tion tasks. InICASSP 2023-2023 IEEE International
Conference on Acoustics, Speech and Signal Process-
ing (ICASSP), pages 1–5. IEEE.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen
Wang. 2023b. Retrieval-augmented generation for
large language models: A survey.arXiv preprint
arXiv:2312.10997.
Yu Gu, Robert Tinn, Hao Cheng, Michael Lucas, Naoto
Usuyama, Xiaodong Liu, Tristan Naumann, Jianfeng
Gao, and Hoifung Poon. 2021. Domain-specific lan-
guage model pretraining for biomedical natural lan-
guage processing.ACM Transactions on Computing
for Healthcare (HEALTH), 3(1):1–23.
Zhao Guo, Yuan Ni, Keqiang Wang, Wei Zhu, and
Guotong Xie. 2021. Global attention decoder for
Chinese spelling error correction. InFindings of
the Association for Computational Linguistics: ACL-
IJCNLP 2021, pages 1419–1428, Online. Association
for Computational Linguistics.
Pengcheng He, Xiaodong Liu, Jianfeng Gao, and
Weizhu Chen. 2020. Deberta: Decoding-
enhanced bert with disentangled attention.ArXiv,
abs/2006.03654.Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,
Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
2020. Measuring massive multitask language under-
standing.arXiv preprint arXiv:2009.03300.
William Hersh. 2024. Search still matters: informa-
tion retrieval in the era of generative ai.Journal of
the American Medical Informatics Association, page
ocae014.
Minbyul Jeong, Jiwoong Sohn, Mujeen Sung, and Jae-
woo Kang. 2024. Improving medical reasoning
through retrieval and self-reflection with retrieval-
augmented large language models.arXiv preprint
arXiv:2401.15269.
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan
Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea
Madotto, and Pascale Fung. 2023. Survey of halluci-
nation in natural language generation.ACM Comput-
ing Surveys, 55(12):1–38.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing
Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. 2023. Ac-
tive retrieval augmented generation.arXiv preprint
arXiv:2305.06983.
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W
Cohen, and Xinghua Lu. 2019. Pubmedqa: A dataset
for biomedical research question answering.arXiv
preprint arXiv:1909.06146.
Qiao Jin, Won Kim, Qingyu Chen, Donald C Comeau,
Lana Yeganova, W John Wilbur, and Zhiyong Lu.
2023. Medcpt: Contrastive pre-trained transformers
with large-scale pubmed search logs for zero-shot
biomedical information retrieval.Bioinformatics,
39(11):btad651.
Qiao Jin, Zhizheng Wang, Yifan Yang, Qingqing Zhu,
Donald Wright, Thomas Huang, W John Wilbur,
Zhe He, Andrew Taylor, Qingyu Chen, et al. 2024.
Agentmd: Empowering language agents for risk pre-
diction with large-scale clinical tool learning.arXiv
preprint arXiv:2402.13225.
Jakub Lála, Odhran O’Donoghue, Aleksandar Shtedrit-
ski, Sam Cox, Samuel G Rodriques, and Andrew D
White. 2023. Paperqa: Retrieval-augmented gener-
ative agent for scientific research.arXiv preprint
arXiv:2312.07559.
Hui Yi Leong, Yuheng Li, Yuqing Wu, Wenwen Ouyang,
Wei Zhu, Jiechao Gao, and Wei Han. 2025. Amas:
Adaptively determining communication topology for
llm-based multi-agent system. InProceedings of the
2025 Conference on Empirical Methods in Natural
Language Processing: Industry Track, pages 2061–
2070.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks.Advances in Neu-
ral Information Processing Systems, 33:9459–9474.

Xiaonan Li, Kai Lv, Hang Yan, Tianya Lin, Wei Zhu,
Yuan Ni, Guo Tong Xie, Xiaoling Wang, and Xipeng
Qiu. 2023a. Unified demonstration retriever for in-
context learning.ArXiv, abs/2305.04320.
Xiepeng Li, Zhexi Zhang, Wei Zhu, Zheng Li, Yuan
Ni, Peng Gao, Junchi Yan, and Guotong Xie. 2019.
Pingan smart health and SJTU at COIN - shared task:
utilizing pre-trained language models and common-
sense knowledge in machine reading tasks. InPro-
ceedings of the First Workshop on Commonsense
Inference in Natural Language Processing, pages
93–98, Hong Kong, China. Association for Computa-
tional Linguistics.
Yuheng Li, Jiechao Gao, Wei Han, Wenwen Ouyang,
Wei Zhu, and Hui Yi Leong. 2025. Ft-mdt: Extract-
ing decision trees from medical texts via a novel
low-rank adaptation method. InProceedings of the
2025 Conference on Empirical Methods in Natural
Language Processing: Industry Track, pages 65–76.
Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long,
Pengjun Xie, and Meishan Zhang. 2023b. Towards
general text embeddings with multi-stage contrastive
learning.arXiv preprint arXiv:2308.03281.
Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mo-
hta, Tenghao Huang, Mohit Bansal, and Colin Raffel.
2022. Few-shot parameter-efficient fine-tuning is
better and cheaper than in-context learning.ArXiv,
abs/2205.05638.
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Man-
dar Joshi, Danqi Chen, Omer Levy, Mike Lewis,
Luke Zettlemoyer, and Veselin Stoyanov. 2019.
Roberta: A robustly optimized bert pretraining ap-
proach.arXiv preprint arXiv:1907.11692.
Zequan Liu, Yi Zhao, Ming Tan, Wei Zhu, and
Aaron Xuxiang Tian. 2025. Para: Parameter-efficient
fine-tuning with prompt aware representation adjust-
ment.arXiv preprint arXiv:2502.01033.
Ilya Loshchilov and Frank Hutter. 2019. Decoupled
weight decay regularization. InICLR.
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. 2023. Query rewriting for retrieval-
augmented large language models.arXiv preprint
arXiv:2305.14283.
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler
Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
Nouha Dziri, Shrimai Prabhumoye, Yiming Yang,
et al. 2024. Self-refine: Iterative refinement with
self-feedback.Advances in Neural Information Pro-
cessing Systems, 36.
Aakanksha Naik, Sravanthi Parasa, Sergey Feldman,
Lucy Lu Wang, and Tom Hope. 2021. Literature-
augmented clinical outcome prediction.arXiv
preprint arXiv:2111.08374.Harsha Nori, Nicholas King, Scott Mayer McKinney,
Dean Carignan, and Eric Horvitz. 2023. Capabili-
ties of gpt-4 on medical challenge problems.arXiv
preprint arXiv:2303.13375.
Chengwei Qin, Aston Zhang, Zhuosheng Zhang, Jiaao
Chen, Michihiro Yasunaga, and Diyi Yang. 2023. Is
chatgpt a general-purpose natural language process-
ing task solver?arXiv preprint arXiv:2302.06476.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models.Transactions of the Association for
Computational Linguistics, 11:1316–1331.
Vipula Rawte, Amit Sheth, and Amitava Das. 2023. A
survey of hallucination in large foundation models.
arXiv preprint arXiv:2309.05922.
Machel Reid, Nikolay Savinov, Denis Teplyashin,
Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste
Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Fi-
rat, Julian Schrittwieser, et al. 2024. Gemini 1.5: Un-
locking multimodal understanding across millions of
tokens of context.arXiv preprint arXiv:2403.05530.
Stephen Robertson, Hugo Zaragoza, et al. 2009. The
probabilistic relevance framework: Bm25 and be-
yond.Foundations and Trends® in Information Re-
trieval, 3(4):333–389.
Khaled Saab, Tao Tu, Wei-Hung Weng, Ryutaro Tanno,
David Stutz, Ellery Wulczyn, Fan Zhang, Tim
Strother, Chunjong Park, Elahe Vedadi, et al. 2024.
Capabilities of gemini models in medicine.arXiv
preprint arXiv:2404.18416.
Erik F Sang and Fien De Meulder. 2003. Introduction
to the conll-2003 shared task: Language-independent
named entity recognition.arXiv preprint cs/0306050.
Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mah-
davi, Jason Wei, Hyung Won Chung, Nathan Scales,
Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl,
et al. 2023a. Large language models encode clinical
knowledge.Nature, pages 1–9.
Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres,
Ellery Wulczyn, Le Hou, Kevin Clark, Stephen
Pfohl, Heather Cole-Lewis, Darlene Neal, et al.
2023b. Towards expert-level medical question an-
swering with large language models.arXiv preprint
arXiv:2305.09617.
Shamane Siriwardhana, Rivindu Weerasekera, Elliott
Wen, Tharindu Kaluarachchi, Rajib Rana, and
Suranga Nanayakkara. 2023. Improving the domain
adaptation of retrieval augmented generation (rag)
models for open domain question answering.Trans-
actions of the Association for Computational Linguis-
tics, 11:1–17.
Haixia Sun, Jin Xiao, Wei Zhu, Yilong He, Sheng
Zhang, Xiaowei Xu, Li Hou, Jiao Li, Yuan Ni, and
Guotong Xie. 2020. Medical knowledge graph to

enhance fraud, waste, and abuse detection on claim
data: Model development and performance evalua-
tion.JMIR Med Inform, 8(7):e17653.
Tianxiang Sun, Xiangyang Liu, Wei Zhu, Zhichao Geng,
Lingling Wu, Yilong He, Yuan Ni, Guotong Xie, Xu-
anjing Huang, and Xipeng Qiu. 2022. A simple
hash-based early exiting approach for language un-
derstanding and generation. InFindings of the As-
sociation for Computational Linguistics: ACL 2022,
pages 2409–2421, Dublin, Ireland. Association for
Computational Linguistics.
Aaron Tian, Yi Zhao, Congrui Yin, Wei Zhu, Xing Tian,
and Yi Ge. 2024a. Fanlora: Fantastic loras and where
to find them in large language model fine-tuning. In
Proceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing: Industry
Track, pages 515–528.
Shubo Tian, Qiao Jin, Lana Yeganova, Po-Ting Lai,
Qingqing Zhu, Xiuying Chen, Yifan Yang, Qingyu
Chen, Won Kim, Donald C Comeau, et al. 2024b.
Opportunities and challenges for chatgpt and large
language models in biomedicine and health.Brief-
ings in Bioinformatics, 25(1):bbad493.
Hugo Touvron, Louis Martin, Kevin R. Stone, Peter
Albert, Amjad Almahairi, Yasmine Babaei, Niko-
lay Bashlykov, Soumya Batra, Prajjwal Bhargava,
Shruti Bhosale, Daniel M. Bikel, Lukas Blecher, Cris-
tian Cantón Ferrer, Moya Chen, Guillem Cucurull,
David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin
Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami,
Naman Goyal, Anthony S. Hartshorn, Saghar Hos-
seini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor
Kerkez, Madian Khabsa, Isabel M. Kloumann, A. V .
Korenev, Punit Singh Koura, Marie-Anne Lachaux,
Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai
Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov,
Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew
Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan
Saladi, Alan Schelten, Ruan Silva, Eric Michael
Smith, R. Subramanian, Xia Tan, Binh Tang, Ross
Taylor, Adina Williams, Jian Xiang Kuan, Puxin
Xu, Zhengxu Yan, Iliyan Zarov, Yuchen Zhang, An-
gela Fan, Melanie Kambadur, Sharan Narang, Aure-
lien Rodriguez, Robert Stojnic, Sergey Edunov, and
Thomas Scialom. 2023. Llama 2: Open foundation
and fine-tuned chat models.ArXiv, abs/2307.09288.
Li Wang, Wei Zhu, Sihang Jiang, Sheng Zhang, Ke-
qiang Wang, Yuan Ni, Guo Tong Xie, and Yanghua
Xiao. 2020. Mining infrequent high-quality phrases
from domain-specific corpora.Proceedings of the
29th ACM International Conference on Information
& Knowledge Management.
Pengfei Wang, Huanran Zheng, Silong Dai, Wenjing
Yue, Wei Zhu, and Xiaoling Wang. 2024. Ts-tcd:
Triplet-level cross-modal distillation for time-series
forecasting using large language models.arXiv
preprint arXiv:2409.14978.Pengfei Wang, Huanran Zheng, Qi’ao Xu, Silong Dai,
Yiqiao Wang, Wenjing Yue, Wei Zhu, Tianwen Qian,
and Liang Zhao. 2025. Ts-htfa: Advancing time-
series forecasting via hierarchical text-free alignment
with large language models.Symmetry, 17(3):401.
Xuwu Wang, Lihan Chen, Wei Zhu, Yuan Ni, Guo Tong
Xie, Deqing Yang, and Yanghua Xiao. 2023a. Multi-
task entity linking with supervision from a taxon-
omy.Knowledge and Information Systems, 65:4335
– 4358.
Yubo Wang, Xueguang Ma, and Wenhu Chen. 2023b.
Augmenting black-box llms with medical textbooks
for clinical question answering.arXiv preprint
arXiv:2309.02233.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Ed Huai hsin Chi, F. Xia, Quoc Le, and
Denny Zhou. 2022. Chain of thought prompting
elicits reasoning in large language models.ArXiv,
abs/2201.11903.
Wei Zhu Wenjing Yue and Xiaoling Wang. 2023.
Tcmeb: Performance evaluation of large language
models based on traditional chinese medicine
benchmarks. https://github.com/ywjawmw/
ShenNong-TCM-Evaluation-BenchMark.
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien
Chaumond, Clement Delangue, Anthony Moi, Pier-
ric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz,
and Jamie Brew. 2019. Huggingface’s transformers:
State-of-the-art natural language processing.ArXiv,
abs/1910.03771.
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas
Muennighof. 2023. C-pack: Packaged resources to
advance general chinese embedding.arXiv preprint
arXiv:2309.07597.
Tianfang Xie, Tianjing Li, Wei Zhu, Wei Han, and
Yi Zhao. 2024. Pedro: Parameter-efficient fine-
tuning with prompt dependent representation modifi-
cation.arXiv preprint arXiv:2409.17834.
Yi Xin, Siqi Luo, Haodi Zhou, Junlong Du, Xiao-
hong Liu, Yue Fan, Qing Li, and Yuntao Du. 2024.
Parameter-efficient fine-tuning for pre-trained vision
models: A survey.ArXiv, abs/2402.02242.
Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and
Aidong Zhang. 2024. Benchmarking retrieval-
augmented generation for medicine.arXiv preprint
arXiv:2402.13178.
Lingling Xu, Haoran Xie, Si-Zhao Joe Qin, Xiaohui
Tao, and Fu Lee Wang. 2023. Parameter-efficient
fine-tuning methods for pretrained language mod-
els: A critical review and assessment.ArXiv,
abs/2312.12148.
Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun
Chen, and Qian Lou. 2024. Badrag: Identifying vul-
nerabilities in retrieval augmented generation of large
language models.arXiv preprint arXiv:2406.00083.

Ellen Yi-Ge, Jiechao Gao, Wei Han, and Wei Zhu.
2024. Drum: Learning demonstration retriever
for large multi-modal models.arXiv preprint
arXiv:2412.07619.
Huiming Yin, Kun Wang, Ruyu Yang, Yanfang Tan,
Qiang Li, Wei Zhu, and Suzi Sung. 2024. A ma-
chine learning model for predicting acute exacerba-
tion of in-home chronic obstructive pulmonary dis-
ease patients.Computer Methods and Programs in
Biomedicine, 246:108005.
Cyril Zakka, Rohan Shad, Akash Chaurasia, Alex R
Dalal, Jennifer L Kim, Michael Moor, Robyn Fong,
Curran Phillips, Kevin Alexander, Euan Ashley,
et al. 2024. Almanac—retrieval-augmented lan-
guage models for clinical medicine.NEJM AI,
1(2):AIoa2300068.
Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin
Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and
Weizhu Chen. 2023a. Repocoder: Repository-level
code completion through iterative retrieval and gen-
eration.arXiv preprint arXiv:2303.12570.
Jingfang Zhang, Ming Tan, Pengyu Dai, and Wei-Guo
Zhu. 2023b. Leco: Improving early exiting via
learned exits and comparison-based exiting mech-
anism. InAnnual Meeting of the Association for
Computational Linguistics.
Juyuan Zhang, Jiechao Gao, Wenwen Ouyang, Wei Zhu,
and Hui Yi Leong. 2025. Time-llama: Adapting
large language models for time series modeling via
dynamic low-rank adaptation. InProceedings of the
63rd Annual Meeting of the Association for Com-
putational Linguistics (Volume 4: Student Research
Workshop), pages 1145–1157.
Tianjun Zhang, Shishir G Patil, Naman Jain, Sheng
Shen, Matei Zaharia, Ion Stoica, and Joseph E Gonza-
lez. 2024. Raft: Adapting language model to domain
specific rag.arXiv preprint arXiv:2403.10131.
Xinpeng Zhang, Ming Tan, Jingfan Zhang, and Wei
Zhu. 2023c. Nag-ner: a unified non-autoregressive
generation framework for various ner tasks. InAn-
nual Meeting of the Association for Computational
Linguistics.
Yuming Zhang, Xiangxiang Gao, Wei Zhu, and Xiaol-
ing Wang. 2023d. Fastner: Speeding up inferences
for named entity recognition tasks. InInternational
Conference on Advanced Data Mining and Applica-
tions.
Yuming Zhang, Peng Wang, Ming Tan, and Wei-Guo
Zhu. 2023e. Learned adapters are better than man-
ually designed adapters. InAnnual Meeting of the
Association for Computational Linguistics.
Zhen Zhang, Wei Zhu, Jinfan Zhang, Peng Wang, Rize
Jin, and Tae-Sun Chung. 2022. PCEE-BERT: Ac-
celerating BERT inference via patient and confident
early exiting. InFindings of the Association for Com-
putational Linguistics: NAACL 2022, pages 327–338,Seattle, United States. Association for Computational
Linguistics.
Zhexi Zhang, Wei Zhu, Junchi Yan, Peng Gao, and
Guowang Xie. 2021. Automatic student network
search for knowledge distillation.2020 25th Inter-
national Conference on Pattern Recognition (ICPR),
pages 2446–2453.
Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren
Wang, Yunteng Geng, Fangcheng Fu, Ling Yang,
Wentao Zhang, and Bin Cui. 2024. Retrieval-
augmented generation for ai-generated content: A
survey.arXiv preprint arXiv:2402.19473.
Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang,
Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen
Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen
Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang,
Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu,
Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023. A
Survey of Large Language Models.arXiv e-prints,
page arXiv:2303.18223.
Huanran Zheng, Wei Zhu, Pengfei Wang, and Xiaol-
ing Wang. 2023. Candidate soups: Fusing candi-
date results improves translation quality for non-
autoregressive translation.ArXiv, abs/2301.11503.
Huanran Zheng, Wei Zhu, and Xiaoling Wang. 2024.
Nat4at: Using non-autoregressive translation makes
autoregressive translation faster and better. InPro-
ceedings of the ACM on Web Conference 2024, pages
4181–4192.
Xiaofeng Zhou, Yuan Ni, Guotong Xie, Wei Zhu, Cai
Chen, Tianhao Wang, and Zhigang Pan. 2019. Anal-
ysis of the health information needs of diabetics in
china. InMEDINFO 2019: Health and Wellbeing
e-Networks for All, pages 487–491. IOS Press.
Wei Zhu. 2021a. Leebert: Learned early exit for bert
with cross-level optimization. InProceedings of the
59th Annual Meeting of the Association for Compu-
tational Linguistics and the 11th International Joint
Conference on Natural Language Processing (Vol-
ume 1: Long Papers), pages 2968–2980.
Wei Zhu. 2021b. Mvp-bert: Multi-vocab pre-training
for chinese bert. InAnnual Meeting of the Associa-
tion for Computational Linguistics.
Wei Zhu. 2026a. Mrag: Benchmarking retrieval-
augmented generation for bio-medicine.arXiv
preprint arXiv:2601.16503.
Wei Zhu. 2026b. Mrag: Benchmarking retrieval-
augmented generation for bio-medicine.arXiv
preprint arXiv:2601.21767.
Wei Zhu, Yilong He, Ling Chai, Yuanchun Fan, Yuan Ni,
Guo Tong Xie, and Xiaoling Wang. 2021a. paht_nlp
@ mediqa 2021: Multi-grained query focused multi-
answer summarization. InWorkshop on Biomedical
Natural Language Processing.

Wei Zhu, Wenfeng Li, Xiaoling Wang, Wendi Ji, Yuan-
bin Wu, Jin Chen, Liang Chen, and Buzhou Tang.
2023a. Extracting decision trees from medical texts:
An overview of the text2dt track in chip2022. In
Health Information Processing. Evaluation Track Pa-
pers, pages 89–102, Singapore. Springer Nature Sin-
gapore.
Wei Zhu, Wenfeng Li, Xiaoling Wang, Wendi Ji, Yuan-
bin Wu, Jin Chen, Liang Chen, and Buzhou Tang.
2023b. Extracting decision trees from medical texts:
An overview of the text2dt track in chip2022. In
Health Information Processing. Evaluation Track Pa-
pers, pages 89–102, Singapore. Springer Nature Sin-
gapore.
Wei Zhu, Yuan Ni, Xiaoling Wang, and Guotong Xie.
2021b. Discovering better model architectures for
medical query understanding. InProceedings of the
2021 Conference of the North American Chapter of
the Association for Computational Linguistics: Hu-
man Language Technologies: Industry Papers, pages
230–237, Online. Association for Computational Lin-
guistics.
Wei Zhu, Yuan Ni, Guo Tong Xie, Xiaofeng Zhou,
and Cai Chen. 2019a. The dr-kgqa system for au-
tomatically answering medication related questions
in chinese.2019 IEEE International Conference on
Healthcare Informatics (ICHI), pages 1–6.
Wei Zhu, Yuan Ni, Guotong Xie, Xiaofeng Zhou, and
Cai Chen. 2019b. The dr-kgqa system for automati-
cally answering medication related questions in chi-
nese. In2019 IEEE International Conference on
Healthcare Informatics (ICHI), pages 1–6. IEEE.
Wei Zhu and Ming Tan. 2023. SPT: Learning to se-
lectively insert prompts for better prompt tuning.
InProceedings of the 2023 Conference on Empir-
ical Methods in Natural Language Processing, pages
11862–11878, Singapore. Association for Computa-
tional Linguistics.
Wei Zhu, Aaron Xuxiang Tian, Congrui Yin, Yuan
Ni, Xiaoling Wang, and Guotong Xie. 2024. Iapt:
Instruction-aware prompt tuning for large language
models.arXiv preprint arXiv:2405.18203.
Wei Zhu, Peifeng Wang, Yuan Ni, Guo Tong Xie, and
Xiaoling Wang. 2023c. Badge: Speeding up bert
inference after deployment via block-wise bypasses
and divergence-based early exiting. InAnnual Meet-
ing of the Association for Computational Linguistics.
Wei Zhu, Peng Wang, Xiaoling Wang, Yuan Ni, and
Guotong Xie. 2023d. Acf: aligned contrastive fine-
tuning for language and vision tasks. InICASSP
2023-2023 IEEE International Conference on Acous-
tics, Speech and Signal Processing (ICASSP), pages
1–5. IEEE.
Wei Zhu, Xiaoling Wang, Mosha Chen, and Buzhou
Tang. 2023e. Overview of the promptcblue shared
task in chip2023.ArXiv, abs/2312.17522.Wei Zhu, Xiaoling Wang, Yuan Ni, and Guotong Xie.
2021c. GAML-BERT: Improving BERT early exit-
ing by gradient aligned mutual learning. InProceed-
ings of the 2021 Conference on Empirical Methods
in Natural Language Processing, pages 3033–3044,
Online and Punta Cana, Dominican Republic. Asso-
ciation for Computational Linguistics.
Wei Zhu, Xiaoling Wang, Huanran Zheng, Mosha Chen,
and Buzhou Tang. 2023. PromptCBLUE: A Chinese
Prompt Tuning Benchmark for the Medical Domain.
arXiv e-prints, page arXiv:2310.14151.
Wei Zhu, Xiaofeng Zhou, Keqiang Wang, Xun Luo,
Xiepeng Li, Yuan Ni, and Guotong Xie. 2019c. Panlp
at mediqa 2019: Pre-trained language models, trans-
fer learning and knowledge distillation. InProceed-
ings of the 18th BioNLP Workshop and Shared Task,
pages 380–388.
Yuhui Zuo, Wei Zhu, and Guoyong GUET Cai. 2022.
Continually detection, rapidly react: Unseen rumors
detection based on continual prompt-tuning. In
Proceedings of the 29th International Conference
on Computational Linguistics, pages 3029–3041,
Gyeongju, Republic of Korea. International Com-
mittee on Computational Linguistics.
A Appendix: Datasets
The datasets we experiment on are as follows:
•MMLUThe Massive Multitask Lan-
guage Understanding (MMLU) benchmark
(Hendrycks et al., 2020) has been introduced
to assess the knowledge gained by large lan-
guage models during pretraining, specifically
in zero-shot and few-shot scenarios. This ap-
proach aims to make the evaluation process
more rigorous and akin to human assessment
standards. The MMLU covers 57 diverse top-
ics, spanning STEM, humanities, social sci-
ences, and many others, with questions rang-
ing from elementary school to advanced pro-
fessional levels. It evaluates not only factual
knowledge but also problem-solving skills,
covering a wide array of fields from conven-
tional subjects like mathematics and history
to more specialized domains such as law and
ethics. The extensive range and detailed cover-
age of these subjects make the MMLU partic-
ularly effective for pinpointing areas where a
model may struggle. Initially, the dataset com-
prises a development set of 1,500 samples and
a test set of 14.1k samples. For our purposes,
we have selected 50 test samples from each
of the 57 categories, creating a subset of 2.8k
test samples.

•PubMedQAPubMedQA (Jin et al., 2019)
is a biomedical research QA dataset. The
dataset is released under the CC BY 4.0
(Creative Commons Attribution 4.0 Inter-
national) license and released in https://
pubmedqa.github.io/ . It has 1k manually an-
notated questions constructed from PubMed
abstracts. To test the capability of RAG sys-
tems to find related documents and answer
the question accordingly, we discard the rel-
evant context for each question originally in-
cluded in the dataset. The possible answer to
a PubMedQA question can be yes/no/maybe,
reflecting the authenticity of the question state-
ment based on biomedical literature. This task
has a set of 1.0k manually labeled PubMedQA
samples, which will be considered the test set
for evaluating the RAG system. And the set of
211k automatic labeled samples will be used
to construct the dataset for query classifica-
tion.
•PromptNERThis dataset is a mixture of
samples from multiple named entity recogni-
tion tasks:
–CoNLL-03 (Sang and De Meulder,
2003). The CoNLL 2003 Named Entity
Recognition (NER) dataset, introduced
as part of the Conference on Natural Lan-
guage Learning (CoNLL) shared task in
2003, is a cornerstone benchmark for
assessing named entity recognition sys-
tems. This dataset features a train/de-
v/test split of 14,041:3,250:3,453, mak-
ing it a valuable resource for NLP re-
searchers and practitioners.
–OntoNotes 5.08. A comprehensive re-
source in Natural Language Processing
(NLP), OntoNotes 5.0 extends its prede-
cessors with rich annotations covering
part-of-speech tagging, syntactic parsing,
semantic role labeling, and named entity
recognition across multiple languages,
including English, Chinese, Arabic, and
more. For NER, it categorizes entities
into predefined classes such as Person,
Location, Organization, and Misc (for
miscellaneous entities). The dataset’s di-
versity, encompassing various text genres
like newswire, web content, and conver-
8https://catalog.ldc.upenn.edu/LDC2013T19sational transcripts, makes it an excel-
lent choice for training robust machine
learning models capable of handling a
wide range of contexts. The English por-
tion of the NER dataset is divided into a
59,924/8,528/8,262 train/dev/test split.
–BioNLP2004 (Collier and Kim, 2004).
Designed specifically for the bioinfor-
matics and biomedical domains, the
BioNLP2004 NER dataset was launched
as part of the BioNLP Shared Task 2004
to evaluate systems that automatically
identify named entities within biomedi-
cal texts. Comprising abstracts from the
PubMed database, this dataset is metic-
ulously annotated with gene and pro-
tein names, serving as a critical tool
for developing and testing NER mod-
els in biological literature. It offers a
16,619/1,927/3,856 train/dev/test distri-
bution, facilitating the training and eval-
uation phases of NER models in this spe-
cialized field.
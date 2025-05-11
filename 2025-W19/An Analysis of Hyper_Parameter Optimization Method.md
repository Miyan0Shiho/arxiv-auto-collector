# An Analysis of Hyper-Parameter Optimization Methods for Retrieval Augmented Generation

**Authors**: Matan Orbach, Ohad Eytan, Benjamin Sznajder, Ariel Gera, Odellia Boni, Yoav Kantor, Gal Bloch, Omri Levy, Hadas Abraham, Nitzan Barzilay, Eyal Shnarch, Michael E. Factor, Shila Ofek-Koifman, Paula Ta-Shma, Assaf Toledo

**Published**: 2025-05-06 11:47:52

**PDF URL**: [http://arxiv.org/pdf/2505.03452v1](http://arxiv.org/pdf/2505.03452v1)

## Abstract
Finding the optimal Retrieval-Augmented Generation (RAG) configuration for a
given use case can be complex and expensive. Motivated by this challenge,
frameworks for RAG hyper-parameter optimization (HPO) have recently emerged,
yet their effectiveness has not been rigorously benchmarked. To address this
gap, we present a comprehensive study involving 5 HPO algorithms over 5
datasets from diverse domains, including a new one collected for this work on
real-world product documentation. Our study explores the largest HPO search
space considered to date, with two optimized evaluation metrics. Analysis of
the results shows that RAG HPO can be done efficiently, either greedily or with
iterative random search, and that it significantly boosts RAG performance for
all datasets. For greedy HPO approaches, we show that optimizing models first
is preferable to the prevalent practice of optimizing sequentially according to
the RAG pipeline order.

## Full Text


<!-- PDF content starts -->

An Analysis of Hyper-Parameter Optimization Methods for Retrieval
Augmented Generation
Matan Orbach, Ohad Eytan, Benjamin Sznajder, Ariel Gera, Odellia Boni,
Yoav Kantor, Gal Bloch, Omri Levy, Hadas Abraham, Nitzan Barzilay,
Eyal Shnarch, Michael E. Factor, Shila Ofek-Koifman, Paula Ta-Shma, Assaf Toledo
IBM Research
matano@il.ibm.com
Abstract
Finding the optimal Retrieval-Augmented Gen-
eration (RAG) configuration for a given use
case can be complex and expensive. Moti-
vated by this challenge, frameworks for RAG
hyper-parameter optimization (HPO) have re-
cently emerged, yet their effectiveness has not
been rigorously benchmarked. To address this
gap, we present a comprehensive study involv-
ing 5 HPO algorithms over 5 datasets from
diverse domains, including a new one collected
for this work on real-world product documen-
tation. Our study explores the largest HPO
search space considered to date, with two op-
timized evaluation metrics. Analysis of the
results shows that RAG HPO can be done effi-
ciently, either greedily or with iterative random
search, and that it significantly boosts RAG
performance for all datasets. For greedy HPO
approaches, we show that optimizing models
first is preferable to the prevalent practice of
optimizing sequentially according to the RAG
pipeline order.
1 Introduction
Despite ongoing advances in applying Large Lan-
guage Models (LLMs) to real-world use cases, en-
suring that LLMs have access to required knowl-
edge to resolve user queries is a challenge. This
has led to the Retrieval-Augmented Generation
(RAG) paradigm, where a retrieval system provides
the generative LLM with a context of grounded in-
formation (Lewis et al., 2020; Huang and Huang,
2024; Gao et al., 2024; Wang et al., 2024b).
The popularity of RAG is largely thanks to its
modular design. By relying on a dedicated re-
trieval component, RAG solutions focus LLMs on
grounded data, reducing the likelihood of depen-
dence on irrelevant preexisting knowledge. These
solutions also allow full control over which data
sources to pull information from for each user ques-
tion. The data sources can be refreshed regularly,
thus keeping the system up-to-date.The modularity of RAG also means that prac-
titioners are faced with a wide array of decisions
when designing their RAG pipelines. One such
choice is which generative LLM to use; other
choices pertain to parameters of the retrieval sys-
tem, such as how many items to retrieve per input
question, how to rank those items, and so forth.
Furthermore, evaluating even a single RAG con-
figuration is costly in terms of time and funds:
the embedding step as part of corpus indexing
is compute-intensive; generating answers using
LLMs is also a demanding task, especially for large
benchmarks; and evaluation with LLMaaJ adds an-
other costly round of inference.
Therefore, exploring and evaluating the full ex-
ponential search space of retrieval and generation
parameters is prohibitively expensive; and at the
same time, suboptimal parameter choices may sig-
nificantly harm performance. Moreover, assuming
that one finds a set of RAG parameters that work
well on a particular dataset, there is no guarantee
that the same parameters will perform well on other
datasets or domains, or even on the same dataset as
the source documents or user queries undergo data
drift over time, and as the available technologies
(e.g., generative and embedding models) evolve.
One approach for addressing these issues is
to use automated hyper-parameter optimization
(HPO) for RAG (Fu et al., 2024a; Kim et al., 2024).
The general idea of such approaches is to conduct
experiments on a limited set of RAG configura-
tions, and recommend a configuration based on the
results. Experiments can be performed in various
ways, ranging from established HPO algorithms
to a naive random sampling of configurations to
test. Importantly, although HPO frameworks are
already gaining ground in RAG use-cases, their
effectiveness has not been rigorously tested.
In this work we study the question of how well
HPO algorithms work in practice for RAG use
cases. To this end, we conduct a comprehen-
1arXiv:2505.03452v1  [cs.CL]  6 May 2025

sive study involving 5HPO algorithms: Tree-
Structured Parzen Estimators (TPE) (Watanabe,
2023), 3greedy optimizations, and random search.
Our experiments explore a large search space of
162 possible RAG configurations, formed from
5parameters of retrieval and generation. To our
knowledge, this is more than twice the size of the
largest RAG search space previously explored for
HPO.1Our study also takes into account both tra-
ditional lexical metrics and more recent LLMaaJ
approaches.
Our experiments span a diverse set of 5RAG
question-answering datasets from different do-
mains: arXiv articles on ML (Eibich et al., 2024),
Biomedical (Krithara et al., 2023), Wikipedia
(Rosenthal et al., 2024; Smith et al., 2008) and
a new product documentation dataset we collect
and release as open-source.
We emphasize that in all our experiments, we
select RAG configurations based on their perfor-
mance on a development set, but report their evalu-
ation on held out test data. This accurately models
the real-world scenario where a dataset is provided
up front for experimentation but unseen user ques-
tions arrive after deployment.
The main contributions of this work are: (i)
benchmark results over the largest RAG search
space yet to be explored (see above), showing that
RAG HPO can be done efficiently, either greedily
or with simple random search, and that it signif-
icantly boosts RAG performance for all datasets;
(ii) a detailed analysis of the results, exploring the
connections between the optimized parameters, the
dataset and the optimization objective. For greedy
HPO approaches, we show that optimizing models
first is preferable to the prevalent practice of op-
timizing sequentially according to pipeline order;
(iii) a new open-source RAG dataset over enterprise
product documentation.2
2 Related Work
Prior work on HPO for RAG includes AutoRAG-
HP (Fu et al., 2024b), which explores 75RAG con-
figurations induced from 3parameters, of which
our work includes two. In comparison, our search
space spans 162 configurations, and adds the
choice of a generative model, which we show to be
the most dominant RAG parameter (see §6). Also,
1Fu et al. (2024b) explored a search space of 75possible
configurations formed from 3parameters. See §2.
2https://huggingface.co/datasets/ibm-research/
watsonxDocsQAFu et al. (2024b) report results on the same set they
optimize on, hiding potential over-fitting, while our
reports are on a held-out test set.
Other work focuses on manual tuning of RAG
hyper-parameters, for example CRUD-RAG (Lyu
et al., 2024) and Wang et al. (2024c). While a man-
ual recipe for finding a good RAG configuration is
advantageous, automation is essential to general-
ize to new use cases and domains, and to handle
continually evolving datasets and models.
More recently, RAGEval (Zhu et al., 2024) uses
synthetic benchmark generation techniques to gen-
erate the DragonBall “omni” benchmark, which en-
compasses data from finance, law and medical texts
in English and Chinese. They found that certain
hyper-parameters are strongly affected by dataset,
such as the optimal chunk size and the number of
retrieved chunks. This provides further motivation
for automated RAG HPO.
Within the open-source community, multiple
tools offer out-of-the-box HPO algorithms for
RAG. One is the AutoRAG framework (Kim et al.,
2024), which takes a greedy approach, optimizing
one RAG parameter at a time based on a sequen-
tial pipeline order.3Another is RAGBuilder which
uses TPE.4RAG-centric libraries such as LlamaIn-
dex5also support HPO by integrating with similar
open source frameworks like RayTune (Liaw et al.,
2018), optuna (Akiba et al., 2019) and hyperopt
(Bergstra et al., 2013).
Missing from all the above works is a systematic
study that shows how well these algorithms work
in practice, and compares them with each other
over multiple datasets. That is the main focus of
our work. The greedy algorithms we explore in §4
are similar in spirit to the greedy algorithm used by
Kim et al. (2024), and the TPE algorithm we evalu-
ate is the same one employed by RAGBuilder.
3 Experimental Setup
The architecture of different RAG pipelines is di-
verse, with some components common to most
architectures. We focus on these common parts,
which we represent by the following linear flow.
Processing starts with chunking the input docu-
ments into smaller chunks, based on two prede-
fined parameters: the size of each chunk in to-
kens, and the overlap between consecutive chunks.
3The AutoRAG repository.
4The RAGBuilder repository.
5The LlamaIndex repository.
2

Figure 1: Our RAG pipeline with hyper-parameters.
This is followed by representing each chunk with
a dense vector created by an embedding model
– our third parameter. The vectors are stored in
a vector-database,6alongside their original text.
Upon receiving a query, it is embedded too, and
the Top-K ( Kbeing our forth parameter) relevant
chunks are retrieved ( retrieval ). Lastly, a prompt
containing the query and the retrieved chunks is
passed to a generative model (our fifth parameter)
to create an answer ( generation ).7The flow of our
RAG pipeline together with the hyper parameters
we used (see below) is depicted in Fig 1.
3.1 Search space
The specific values considered for each of our 5
hyper-parameters are described in Table 1. In to-
tal, the search space has 3∗2∗3∗3∗3 = 162
possible RAG configurations . Among them, 18
(3∗2∗2) relate to data indexing (chunking and
embedding). The values chosen for the chunk size,
overlap, and top-k reflect common practices. Pop-
ular open source models were selected as options
for the embedding and generative models.
Our experiments involve performing a full grid
search – i.e., evaluating all possible configurations
– in order to infer an upper bound for the optimiza-
tion strategies. Hence, due to computational con-
straints we avoid using an even larger search space.
3.2 Datasets
Each RAG dataset is comprised of a corpus of doc-
uments and a benchmark of QA pairs, with most
also annotating the document(s) with the correct
answer. Below are the RAG datasets we used:
AIArxiv This dataset was derived from the ARA-
GOG benchmark (Eibich et al., 2024) of technical
QA pairs over a corpus of machine learning papers
from ArXiv.8As gold documents are not annotated
in ARAGOG dataset, we added such labels where
6Milvus (Wang et al., 2021) with default settings; index
type is HNSW (Malkov and Yashunin, 2018).
7Generation details are in Appendix F.
8AIArxiv is available here.they could be found, obtaining 71answerable QA
pairs out of 107in the original benchmark.
BioASQ (Krithara et al., 2023) A subset of the
BioASQ Challenge train set.9Its corpus contains
40200 passages extracted from clinical case reports.
The corresponding benchmark of 4200 QA pairs
includes multiple gold documents per question.
MiniWiki A benchmark of 918QA pairs over
Wikipedia derived from Smith et al. (2008).10The
contents are mostly factoid questions with short
answers. This dataset has no gold document labels.
ClapNQ (Rosenthal et al., 2024) A subset of
the Natural Questions (NQ) dataset (Kwiatkowski
et al., 2019) on Wikipedia pages, of questions
that have long answers. The original benchmark
contains both answerable and unanswerable ques-
tions. For our analysis we consider only the former.
ClapNQ dataset consists of 178890 sub-documents
generated from 4293 pages. These sub-documents
constitute the input to the pipeline.
WatsonxQA (ProductDocs) A new open source
dataset and benchmark based on enterprise product
documentation. It consists of 5534 sub-documents
generated from 1144 pages of HTML product doc-
umentation. As with the ClapNQ dataset, the sub-
documents serve as input for the pipeline. The
benchmark includes 75QA pairs and gold docu-
ment labels, of which 25were generated by two
subject matter experts, and the rest were gener-
ated synthetically and then manually filtered and
reviewed by two of the authors.
Overall, these datasets exhibit variability in
many aspects. They represent diverse domains –
research papers, biomedical documents, wikipedia
pages and enterprise data. They also vary in ques-
tion and answer lengths; for example, MiniWiki
has relatively short answers, while ClapNQ was
purposely built with long gold answers. Corpus
sizes also vary, representing real-world retrieval
scenarios over small or large sets of documents.
Every benchmark was split into development and
test sets. To keep computations tractable, the num-
ber of questions in the large benchmarks (BioASQ
and ClapNQ) was limited to 1000 for development
and 150 for test. Table 2 lists the corpora and
benchmark sizes and domains. Question examples
are in Appendix A.
9BioASQ is available here.
10MiniWiki is available here.
3

Parameter RAG step Values
Chunk Size (# Tokens) Chunking 256, 384, 512
Chunk Overlap (% Tokens) Chunking 0%, 25%
Embedding Model Embedding multilingual-e5-large (Wang et al., 2024a)
bge-large-en-v1.5 (Xiao et al., 2023)
granite-embedding-125M-english (IBM, 2024)
Top-k (# Chunks to retrieve) Retrieval 3, 5, 10
Generative Model Generation Llama-3.1-8B-Instruct (AI, 2024)
Mistral-Nemo-Instruct-2407 (AI and NVIDIA, 2024)
Granite-3.1-8B-instruct (Granite Team, 2024)
Table 1: Hyper-Parameters in Explored RAG Configurations
Dataset Domain #Doc #Dev #Test
AIArxiv Papers 2673 41 30
BioASQ Biomedical 40181 1000 150
MiniWiki Wikipedia 3200 663 150
ClapNQ Wikipedia 178890 1000 150
WatsonxQA Business 5534 45 30
Table 2: Properties of the RAG Datasets in our experi-
ments, including the number of documents within the
corpus and QA pairs within the #Dev and#Test sets.
3.3 Evaluation
Retrieval Quality is evaluated using context cor-
rectness with Mean Reciprocal Rank (V oorhees
and Tice, 2000).11
Answer correctness compares generated and
gold answers, assessing overall pipeline quality ,
and is measured in two ways: First, a fast, lexical,
implementation based on token recall ( Lexical-AC ),
which provides a good balance between speed and
quality (Adlakha et al., 2024). Second, a standard
LLM-as-a-Judge answer correctness ( LLMaaJ-AC )
implementation from the RAGAS library (Es et al.,
2023), with GPT4o-mini (OpenAI, 2024) as its
backbone. This implementation performs 3calls
to the LLM on every invocation, making it much
slower and more expensive than the lexical variant.
While iterating over a benchmark, context cor-
rectness is computed per question after retrieval,
and answer correctness is computed after answer
generation. Averaging these scores over all bench-
mark instances determines the global scores of the
RAG configuration on the dataset.
11Note that as labeling is at the document level, any chunk
from the gold document is considered as a correct prediction,
even if it does not include the answer to the question.4 HPO Algorithms
An HPO algorithm takes as input a RAG search
space, a dataset (corpus + benchmark), and one or
more evaluation metrics, of which one is defined
as the optimization objective. Its goal is to find the
best-performing RAG configuration on the dataset
in terms of the objective metric.
The HPO algorithms in our experiments are it-
erative. At each iteration, the algorithm looks at
all previously explored RAG configurations and
their scores, and chooses one unexplored configu-
ration to try next. Simulating a fixed exploration
budget, after a fixed number of iterations the algo-
rithm stops and the best performing configuration
is returned. A good HPO algorithm identifies a top-
performing configuration with minimal iterations.
We consider two types of HPO algorithms. The
first are standard HPO algorithms not specifically
tailored to a certain type of RAG pipeline. The
other type of algorithms make specific assumptions
on the optimized pipeline and its components. All
optimize an answer correctness metric (see §3.3)
unless explicitly stated otherwise below.
Our first standard HPO algorithm is the Tree-
Structured Parzen Estimators ( TPE) algorithm
(Watanabe, 2023), implemented in the hyperopt
library (Bergstra et al., 2013). We configure it to
start with 5random bootstrapping iterations and
otherwise use the default settings. The second stan-
dard algorithm ( Random ) simply samples a new
random RAG configuration, without replacement,
at each iteration, while ignoring the scores of pre-
viously explored configurations.
The other algorithms are greedy, and require a
list of search space parameters as input, ordered by
their presumed impact on the RAG pipeline. They
iterate the list, optimizing one parameter at a time,
4

assuming that optimizing more impactful ones first
leads to quick identification of a good configura-
tion. When optimizing a list parameter p, assuming
values for those before on the list were already de-
termined, the algorithm samples random values for
all following list parameters. The parameter pis
then evaluated with all possible values. The value
yielding the best objective score is picked, and the
algorithm continues to the next hyper-parameter.
The greedy algorithms differ in the order in
which the hyper-parameters are optimized. One
option ( Greedy-M ) is to order the generative and
embedding models first, assuming they are more
important: Generative Model, Embedding Model,
Chunk Size, Chunk Overlap, Top-k. A second
prevalent one ( Greedy-R ) is to optimize by the or-
der of the RAG pipeline itself, first the retrieval
part (still with the model first), followed similarly
by the generation part: Embedding Model, Chunk
Size, Chunk Overlap, Generative Model and Top-k.
A third option (also prevalent) ( Greedy-R-CC ) uses
the same order, yet optimizes the first 3retrieval-
related parameters according to the score of a con-
text correctness metric evaluated on the produced
index, thus avoiding any LLM-based inference un-
til all retrieval parameters are chosen. After that,
the algorithm optimizes the remaining parameters
with answer correctness.
5 Results
Experiments Description The experiments con-
sider the search space of §3.1 and all described
datasets in §3.2. The optimization objective
is answer correctness, implemented with either
LLMaaJ-AC or Lexical-AC (§3.3).
Every HPO algorithm ran on each development
set for 10iterations. After each iteration, the best
configuration so far was applied to the test set. This
allows a per-iteration progress analysis (see below).
Since all algorithms involve a random element,
each run was repeated with 10different seeds.
Our setting was chosen to simulate a realistic
use case: an HPO algorithm is executed over a
benchmark (the development set) and is used to
select a RAG configuration to deploy; the deployed
configuration is expected to generalize to a set of
unseen questions – simulated here via a test set.
To establish upper bounds on the expected per-
formance, we also run a full grid search over the
development and test sets. These upper bounds
allow us to assess to what extent the different al-gorithms approach the optimal result. This entails
evaluating all possible 162configurations, includ-
ing18different indexes, on all 5datasets.
Main Results Figure 2 depicts the per-iteration
performance of the algorithms on the test sets. Also
shown (black dashed line) is the best achievable per-
formance on test, obtained by a full grid search on
that set. For all algorithms, it is enough to explore
around 10configurations to obtain performance
that is comparable to exploring all 162available
configurations with a naive grid search (solid red
lines in Figure 2). This strong result holds for both
evaluation metrics.
In some cases there is a substantial gap between
performance of the best achievable configuration
on the test test (dashed black lines), and the perfor-
mance obtained when taking the best configuration
on the development set, and applying it to the test
(red lines). This is an inherent limitation of any
generalization scenario. All algorithms, being ex-
posed only to the development set, can only aspire
to performance close to the red lines on the test,
and indeed the results all show they converge to
roughly that performance.
Also evident in the results is the improvement
at iteration 10over iteration 1. The latter, being a
single random choice, may be thought of as a proxy
for a naive unexperienced user choosing parameter
values in a manner that is similar to random. This
demonstrates the benefit of HPO for practitioners
making their first steps in RAG.
Focusing on the greedy algorithms, the results
imply that the order in which parameters are op-
timized is of great importance. Specifically, the
methods that start by optimizing retrieval-related
parameters (Greedy-R and Greedy-R-CC) require
more iterations to identify good configurations.
The naive option of random sampling also finds
a good RAG configuration after a small number
of iterations. The complex TPE algorithm does
not appear to add much value, choosing a config-
uration which is roughly equivalent in quality to
random choice. This may be a result of the size of
our search space, and also because 10 iterations is
relatively few for TPE which in any case begins
with several random iterations. Exploring larger
search spaces, possibly giving up on a full grid
search comparison, is a topic for further work.
Development Set Sampling The above results
were produced with each RAG pattern evaluated
on an entire development set. While simple, this
5

1 2 3 4 5 6 7 8 9 10
# Iterations0.50.6RAGAS ACAIArxiv
1 2 3 4 5 6 7 8 9 10
# Iterations0.5000.5250.5500.575RAGAS ACBioASQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.5000.5250.5500.5750.600RAGAS ACClapNQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.450.500.55RAGAS ACMiniWiki
1 2 3 4 5 6 7 8 9 10
# Iterations0.600.650.700.75RAGAS ACProductDocs
Grid
Random
TPEGreedy-M
Greedy-R
Greedy-R-CC
1 2 3 4 5 6 7 8 9 10
# Iterations0.50.6RAGAS ACAIArxiv
1 2 3 4 5 6 7 8 9 10
# Iterations0.5000.5250.5500.575RAGAS ACBioASQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.5000.5250.5500.5750.600RAGAS ACClapNQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.450.500.55RAGAS ACMiniWiki
1 2 3 4 5 6 7 8 9 10
# Iterations0.600.650.700.75RAGAS ACProductDocs
Grid
Random
TPEGreedy-M
Greedy-R
Greedy-R-CC
1 2 3 4 5 6 7 8 9 10
# Iterations0.50.6RAGAS ACAIArxiv
1 2 3 4 5 6 7 8 9 10
# Iterations0.5000.5250.5500.575RAGAS ACBioASQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.5000.5250.5500.5750.600RAGAS ACClapNQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.450.500.55RAGAS ACMiniWiki
1 2 3 4 5 6 7 8 9 10
# Iterations0.600.650.700.75RAGAS ACProductDocs
Grid
Random
TPEGreedy-M
Greedy-R
Greedy-R-CC
1 2 3 4 5 6 7 8 9 10
# Iterations0.50.6RAGAS ACAIArxiv
1 2 3 4 5 6 7 8 9 10
# Iterations0.5000.5250.5500.575RAGAS ACBioASQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.5000.5250.5500.5750.600RAGAS ACClapNQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.450.500.55RAGAS ACMiniWiki
1 2 3 4 5 6 7 8 9 10
# Iterations0.600.650.700.75RAGAS ACProductDocs
Grid
Random
TPEGreedy-M
Greedy-R
Greedy-R-CC
1 2 3 4 5 6 7 8 9 10
# Iterations0.50.6RAGAS ACAIArxiv
1 2 3 4 5 6 7 8 9 10
# Iterations0.5000.5250.5500.575RAGAS ACBioASQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.5000.5250.5500.5750.600RAGAS ACClapNQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.450.500.55RAGAS ACMiniWiki
1 2 3 4 5 6 7 8 9 10
# Iterations0.600.650.700.75RAGAS ACProductDocs
Grid
Random
TPEGreedy-M
Greedy-R
Greedy-R-CC(a) LLMaaJ-AC
1 2 3 4 5 6 7 8 9 10
# Iterations0.5500.5750.6000.625Lexical ACAIArxiv
1 2 3 4 5 6 7 8 9 10
# Iterations0.600.620.640.660.68Lexical ACBioASQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.500.550.60Lexical ACClapNQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.800.850.90Lexical ACMiniWiki
1 2 3 4 5 6 7 8 9 10
# Iterations0.7500.7750.8000.8250.850Lexical ACProductDocs
Grid
Random
TPEGreedy-M
Greedy-R
Greedy-R-CC
1 2 3 4 5 6 7 8 9 10
# Iterations0.5500.5750.6000.625Lexical ACAIArxiv
1 2 3 4 5 6 7 8 9 10
# Iterations0.600.620.640.660.68Lexical ACBioASQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.500.550.60Lexical ACClapNQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.800.850.90Lexical ACMiniWiki
1 2 3 4 5 6 7 8 9 10
# Iterations0.7500.7750.8000.8250.850Lexical ACProductDocs
Grid
Random
TPEGreedy-M
Greedy-R
Greedy-R-CC
1 2 3 4 5 6 7 8 9 10
# Iterations0.5500.5750.6000.625Lexical ACAIArxiv
1 2 3 4 5 6 7 8 9 10
# Iterations0.600.620.640.660.68Lexical ACBioASQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.500.550.60Lexical ACClapNQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.800.850.90Lexical ACMiniWiki
1 2 3 4 5 6 7 8 9 10
# Iterations0.7500.7750.8000.8250.850Lexical ACProductDocs
Grid
Random
TPEGreedy-M
Greedy-R
Greedy-R-CC
1 2 3 4 5 6 7 8 9 10
# Iterations0.5500.5750.6000.625Lexical ACAIArxiv
1 2 3 4 5 6 7 8 9 10
# Iterations0.600.620.640.660.68Lexical ACBioASQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.500.550.60Lexical ACClapNQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.800.850.90Lexical ACMiniWiki
1 2 3 4 5 6 7 8 9 10
# Iterations0.7500.7750.8000.8250.850Lexical ACProductDocs
Grid
Random
TPEGreedy-M
Greedy-R
Greedy-R-CC
1 2 3 4 5 6 7 8 9 10
# Iterations0.5500.5750.6000.625Lexical ACAIArxiv
1 2 3 4 5 6 7 8 9 10
# Iterations0.600.620.640.660.68Lexical ACBioASQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.500.550.60Lexical ACClapNQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.800.850.90Lexical ACMiniWiki
1 2 3 4 5 6 7 8 9 10
# Iterations0.7500.7750.8000.8250.850Lexical ACProductDocs
Grid
Random
TPEGreedy-M
Greedy-R
Greedy-R-CC (b) Lexical-AC
Figure 2: Per-iteration performance of all HPO algorithms on the test sets of five datasets, optimizing answer
correctness computed with an LLMaaJ metric (a) and a lexical metric (b). The dashed black lines denote the best
achievable performance on each test set.
option is costly for large datasets. Research in
other domains has suggested that limiting evalu-
ation benchmarks through random sampling im-
proves efficiency without compromising evaluation
quality (Perlitz et al., 2024; Polo et al., 2024). Thus,
we experiment with HPO that uses a sample of the
development set. To our knowledge, this is the first
study of sampling in the context of RAG HPO.
To match the RAG setting, we sample both the
benchmark and the underlying corpus, to reduce
both inference and embedding costs. Specifically,
focusing on the larger datasets of BioASQ and
ClapNQ, 10% of the development set QA pairs
were sampled along with their corresponding gold
documents – those containing the answers to the
sampled questions. To simulate realistic retrieval
scenarios we add “noise” – documents not contain-ing an answer to any of the sampled questions –
at a ratio of 9such documents per gold document.
This process results in 100sampled benchmark
questions per dataset. The sampled corpora com-
prise 1K (i.e. 1000 ) documents for ClapNQ (out of
178K), and 10K for BioASQ (out of 40K).12
Following sampling, we repeat the experiments
using the best HPO methods: Random, TPE, and
Greedy-M. Figure 3 depicts the results when opti-
mizing for LLMaaJ-AC, comparing all three meth-
ods when applied to the full (solid lines) or sam-
pled (dotted) development sets (similar Lexical-AC
results are in Appendix B). As can be seen, for
BioASQ results remain about the same, while for
12BioASQ has multiple gold documents per question, which
yields more sampled documents.
6

1 2 3 4 5 6 7 8 9 10
# Iterations0.500.520.540.560.58RAGAS ACBioASQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.500.520.540.560.580.60RAGAS ACClapNQ
Grid/Full
Grid/Sample
Random/FullRandom/Sample
TPE/Full
TPE/SampleGreedy-M/Full
Greedy-M/SampleFigure 3: Per-iteration LLMaaJ-AC performance on the
test sets of two largest datasets, for HPO algorithms
optimized using the fulldevelopment data (solid lines)
or asample of it (dotted). The dashed black lines show
the best achievable test performance.
ClapNQ sampling leads to identifying a suboptimal
configuration after all 10iterations, most notably
for TPE and Random.
In terms of cost, for both datasets, performing
inference for a given configuration is 10x cheaper
(as10% of the questions are used). Indexing is also
far cheaper, since the corpus is decreased by 4x for
BioASQ and 178x for ClapNQ.
Overall, while sampling the development set is
computationally promising, our mixed results sug-
gest it remains an open challenge – i.e., an appropri-
ate sampling approach must be used or else quality
is degraded. We leave this for further work.
6 Analysis
We analyze three aspects of our experiments: the
impact of the chosen evaluation metric, the impact
of retrieval quality on answer quality, and cost.
Impact of Metric Choice According to the re-
sults of Greedy-M in Figure 2, performance is
boosted significantly when optimizing the gener-
ative model parameter first, suggesting its impor-
tance. We therefore further examine the best per-
forming model in each experiment.Figure 4 depicts per-dataset maximal answer cor-
rectness scores for each generative model (maxi-
mum of all 162/3 = 54 configurations the model
participates in). Interestingly, the best RAG con-
figuration for all datasets involves Llama when op-
timizing by LLMaaJ-AC and Granite or Mistral
when optimizing by Lexical-AC. This difference
stems from the metrics, as Lexical-AC is recall-
oriented while LLMaaJ-AC accounts for both pre-
cision and recall. This difference highlights the
importance of selecting an objective metric that
correctly reflects use case preferences, as optimal
configuration may vary for different objective met-
rics.
Impact of Retrieval Quality Figure 5 depicts the
relationship between context correctness and an-
swer correctness. The two measures are correlated,
and specifically, when retrieval quality is low, so is
answer quality. Also evident is the great influence
of generative model choice on answer quality.
The impact of context correctness on answer
correctness diminishes as context correctness in-
creases, as implied by the line slopes in Figure 5
being steeper at low context correctness values.
This behavior can be attributed to the fact that sev-
eral retrieved chunks are utilized during generation.
When context correctness is low, there is a high
probability that none of the chunks are relevant,
and the generated answer will therefore be incor-
rect. Conversely, when at least one relevant chunk
is retrieved, the quality of the answer will largely
rely on generation hyper-parameters.
Cost Considerations We also track the cost of
each HPO algorithm, in terms of number of em-
bedded tokens and number of input and output
tokens during generation. These counts are accu-
mulated per-algorithm over all explored configura-
tions, where indices that are reused in multiple ex-
plored configurations are accounted for only once.
The number of different indices created by each
algorithm is the most significant factor in embed-
ding cost. TPE and Random behave the same, as
they have no preference to reuse existing indices
nor vice versa. Greedy-R and Greedy-R-CC be-
have the same: at start, they explore only differ-
ent indices, hence they are more expensive, but
once the index configuration is fixed, they become
cheaper. Greedy-M starts with one index and iter-
ates first over generation models, so it is initally
less costly, but over the next iterations it behaves
similarly to TPE and Random overtaking Greedy-R
7

(a) LLMaaJ-AC
 (b) Lexical-AC
Figure 4: The effect of chosen optimization metric on the generative model within the best RAG configuration.
0.60 0.63 0.65 0.68 0.70 0.73 0.75 0.78
Context Correctness0.440.460.480.500.520.540.56Answer Correctness (LLMaaJ)
BioASQ
Generative Model
Llama-3-1-8B-Instruct
Mistral_Nemo_Instruct
Granite-3.1-8B-Instruct
0.25 0.30 0.35 0.40 0.45 0.50
Context Correctness0.460.480.500.520.540.56Answer Correctness (LLMaaJ)
ClapNQ
Generative Model
Llama-3-1-8B-Instruct
Mistral_Nemo_Instruct
Granite-3.1-8B-Instruct
Figure 5: The effect of retrieval quality (context correctness, x-axis) on generation quality (answer correctness,
y-axis). Each point denotes one of the 162RAG configurations in the search space.
and Greedy-R-CC. These behaviors are consistent
across all datasets. We do not find a significant dif-
ference in generation token cost across algorithms.
Results for both metrics are in Appendix C.
7 Conclusion
Our work presented a comprehensive study of mul-
tiple RAG HPO algorithms, exploring their perfor-
mance over datasets from various domains, using
different metrics. The results can be summarized
into important guidelines for practitioners.
We demonstrated the importance of RAG HPO,
showing that it can boost performance significantly
compared to an arbitrary starting point (by as much
as20%). As this holds for a simple RAG pipeline,
the effect on a more complex (or agentic) RAG
pipelines with many more hyper-parameters is
likely to be larger.
We emphasize the need for an automated ap-
proach. For example, the impact of the optimiza-
tion metric on the optimal RAG configuration (see
Figure 4) dictates that when changing metrics eventhe same RAG pipeline must be re-adjusted to the
new metric with HPO. Similarly, changes in avail-
able models or evolving datasets require revising
the chosen RAG configuration.
We show that RAG HPO can be performed
efficiently, exploring only a fraction of all pos-
sible RAG configurations induced by the hyper-
parameters. Even without prior knowledge of the
RAG pipeline architecture, iterative random sam-
pling bounded by a fixed number of iterations can
suffice. We also show that a greedy HPO approach
that optimizes models first is preferable to the
prevalent practice of optimizing sequentially ac-
cording to pipeline order.
We take initial steps in improving efficiency of
RAG HPO, evaluating development set sampling,
and plan to further explore this in future work. We
also contribute a new RAG dataset, of enterprise
product documentation, for use by the community.
8

8 Limitations
We limited the size of our search space to en-
able a full grid search on all datasets in a reason-
able time and cost. Further work could explore a
larger search space without comparing to a full grid
search.
Our experiments optimized a simple linear RAG
workflow. Additional RAG approaches may intro-
duce additional hyper-parameters to the mix, such
as filters, rerankers, answer reviewers, HyDE etc.
Moreover, our prompts were fixed per generation
model and we did not explore prompt optimization.
Our analysis considers only English textual
datasets with answerable questions. The compari-
son of performance for multi-modal, multi-lingual
datasets that include also unanswerable questions
is left for future work.
It should be noted that our method for retrieval
quality evaluation is not accurate: any chunk from
the gold document is considered a correct predic-
tion even if it does not include the answer to the
question.
References
Vaibhav Adlakha, Parishad BehnamGhader, Xing Han
Lu, Nicholas Meade, and Siva Reddy. 2024. Eval-
uating correctness and faithfulness of instruction-
following models for question answering. Trans-
actions of the Association for Computational Linguis-
tics, 12:681–699.
Meta AI. 2024. Introducing llama 3.1: Our most capa-
ble models to date.
Mistral AI and NVIDIA. 2024. Mistral nemo: A state-
of-the-art 12b model with 128k context length.
Takuya Akiba, Satoshi Sano, Tatsuya Koyama, Yuji
Matsumoto, and Masanori Ohta. 2019. Optuna: A
next-generation hyperparameter optimization frame-
work. arXiv preprint arXiv:1907.10902 .
J. Bergstra, D. Yamins, and D. D. Cox. 2013. Making a
science of model search: Hyperparameter optimiza-
tion in hundreds of dimensions for vision architec-
tures. In Proceedings of the 30th International Con-
ference on Machine Learning (ICML 2013) , pages
I–115–I–123.
Matouš Eibich, Shivay Nagpal, and Alexander Fred-
Ojala. 2024. Aragog: Advanced rag output grading.
arXiv preprint arXiv:2404.01037 .
Shahul Es, Jithin James, Luis Espinosa-Anke, and
Steven Schockaert. 2023. Ragas: Automated eval-
uation of retrieval augmented generation. Preprint ,
arXiv:2309.15217.Jia Fu, Xiaoting Qin, Fangkai Yang, Lu Wang,
Jue Zhang, Qingwei Lin, Yubo Chen, Dongmei
Zhang, Saravan Rajmohan, and Qi Zhang. 2024a.
Autorag-hp: Automatic online hyper-parameter tun-
ing for retrieval-augmented generation. Preprint ,
arXiv:2406.19251.
Jia Fu, Xiaoting Qin, Fangkai Yang, Lu Wang, Jue
Zhang, Qingwei Lin, Yubo Chen, Dongmei Zhang,
Saravan Rajmohan, and Qi Zhang. 2024b. AutoRAG-
HP: Automatic online hyper-parameter tuning for
retrieval-augmented generation. In Findings of the
Association for Computational Linguistics: EMNLP
2024 , pages 3875–3891, Miami, Florida, USA. Asso-
ciation for Computational Linguistics.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gener-
ation for large language models: A survey. Preprint ,
arXiv:2312.10997.
IBM Granite Team. 2024. Granite 3.0 language models.
Accessed: 2025-02-14.
Yizheng Huang and Jimmy Huang. 2024. A survey
on retrieval-augmented text generation for large lan-
guage models. Preprint , arXiv:2404.10981.
IBM. 2024. Granite-embedding-125m-english model
card.
Dongkyu Kim, Byoungwook Kim, Donggeon Han, and
Matouš Eibich. 2024. Autorag: Automated frame-
work for optimization of retrieval augmented genera-
tion pipeline. arXiv preprint arXiv:2410.20878 .
Anastasia Krithara, Anastasios Nentidis, Konstantinos
Bougiatiotis, and Georgios Paliouras. 2023. Bioasq-
qa: A manually curated corpus for biomedical ques-
tion answering. Scientific Data , 10(1):170.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, et al. 2019. Natural questions: a benchmark
for question answering research. Transactions of the
Association for Computational Linguistics , 7:453–
466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. In Proceedings of the 34th Inter-
national Conference on Neural Information Process-
ing Systems , NIPS ’20, Red Hook, NY , USA. Curran
Associates Inc.
Richard Liaw, Eric Liang, Robert Nishihara, Philipp
Moritz, Joseph E Gonzalez, and Ion Stoica.
2018. Tune: A research platform for distributed
model selection and training. arXiv preprint
arXiv:1807.05118 .
9

Yuanjie Lyu, Zhiyu Li, Simin Niu, Feiyu Xiong,
Bo Tang, Wenjin Wang, Hao Wu, Huanyong Liu,
Tong Xu, and Enhong Chen. 2024. Crud-rag:
A comprehensive chinese benchmark for retrieval-
augmented generation of large language models.
arXiv:2401.17043 [cs.CL] . Accessed: 2025-02-14.
Yu. A. Malkov and D. A. Yashunin. 2018. Efficient and
robust approximate nearest neighbor search using
hierarchical navigable small world graphs. Preprint ,
arXiv:1603.09320.
OpenAI. 2024. Gpt-4o mini: Advancing cost-efficient
intelligence.
Yotam Perlitz, Elron Bandel, Ariel Gera, Ofir Arviv,
Liat Ein-Dor, Eyal Shnarch, Noam Slonim, Michal
Shmueli-Scheuer, and Leshem Choshen. 2024. Ef-
ficient benchmarking (of language models). In Pro-
ceedings of the 2024 Conference of the North Amer-
ican Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume
1: Long Papers) , pages 2519–2536, Mexico City,
Mexico. Association for Computational Linguistics.
Felipe Maia Polo, Lucas Weber, Leshem Choshen,
Yuekai Sun, Gongjun Xu, and Mikhail Yurochkin.
2024. tinybenchmarks: evaluating llms with fewer
examples. In Proceedings of the 41st Interna-
tional Conference on Machine Learning , ICML’24.
JMLR.org.
Sara Rosenthal, Avirup Sil, Radu Florian, and Salim
Roukos. 2024. Clapnq: Cohesive long-form answers
from passages in natural questions for rag systems.
Preprint , arXiv:2404.02103.
Noah A Smith, Michael Heilman, and Rebecca Hwa.
2008. Question generation as a competitive under-
graduate course project. In Proceedings of the NSF
Workshop on the Question Generation Shared Task
and Evaluation Challenge , volume 9.
Ellen M. V oorhees and Dawn M. Tice. 2000. The TREC-
8 question answering track. In Proceedings of the
Second International Conference on Language Re-
sources and Evaluation (LREC‘00) , Athens, Greece.
European Language Resources Association (ELRA).
Jianguo Wang, Xiaomeng Yi, Rentong Guo, Hai Jin,
Peng Xu, Shengjun Li, Xiangyu Wang, Xiangzhou
Guo, Chengming Li, Xiaohai Xu, Kun Yu, Yux-
ing Yuan, Yinghao Zou, Jiquan Long, Yudong Cai,
Zhenxiang Li, Zhifeng Zhang, Yihua Mo, Jun Gu,
Ruiyi Jiang, Yi Wei, and Charles Xie. 2021. Mil-
vus: A purpose-built vector data management sys-
tem. In Proceedings of the 2021 International Con-
ference on Management of Data , SIGMOD ’21, page
2614–2627, New York, NY , USA. Association for
Computing Machinery.
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,
and Rangan Majumderand Furu Wei. 2024a. Mul-
tilingual e5 text embeddings: A technical report.
Preprint , arXiv:2402.05672.Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran
Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi,
Zhengyuan Wang, Shizheng Li, Qi Qian, Ruicheng
Yin, Changze Lv, Xiaoqing Zheng, and Xuan-
jing Huang. 2024b. Searching for best prac-
tices in retrieval-augmented generation. Preprint ,
arXiv:2407.01219.
Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran
Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi,
Zhengyuan Wang, Shizheng Li, Qi Qian, et al. 2024c.
Searching for best practices in retrieval-augmented
generation. In Proceedings of the 2024 Conference
on Empirical Methods in Natural Language Process-
ing, pages 17716–17736.
Shuhei Watanabe. 2023. Tree-structured parzen esti-
mator: Understanding its algorithm components and
their roles for better empirical performance. Preprint ,
arXiv:2304.11127.
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas
Muennighoff. 2023. C-pack: Packaged resources
to advance general chinese embedding. Preprint ,
arXiv:2309.07597.
Kunlun Zhu, Yifan Luo, Dingling Xu, Ruobing Wang,
Shi Yu, Shuo Wang, Yukun Yan, Zhenghao Liu,
Xu Han, Zhiyuan Liu, and Maosong Sun. 2024.
Rageval: Scenario specific rag evaluation dataset gen-
eration framework. Preprint , arXiv:2408.01262.
A Datasets
Example questions per dataset are listed in Table 3.
B Additional results
Figure 6 details development sampling results for
the Lexical-AC metric.
C Embedding and Generation Costs
Figure 7 details accumulated numbers of (a) em-
bedded tokens and (b) number of tokens used in
generation part for HPO algorithms overall tested
configurations.
D Hardware and Costs
All used embedding and generation models are
open source models. An internal in-house infras-
tructure containing V100 and A100 GPUs was used
to run embedding computations and generative in-
ference. Specifically, embeddings were computed
using one V100 GPU, and inference was done on
one A100 GPU (i.e. no multi-GPU inference was
required).
10

Dataset Example Query
AIArxiv What significant improvements does BERT bring to the SQuAD v1.1,v2.0 and v13.5
tasks compared to prior models?
BioASQ What is the implication of histone lysine methylation in medulloblastoma?
MiniWiki Was Abraham Lincoln the sixteenth President of the United States?
ClapNQ Who is given credit for inventing the printing press?
WatsonxQA What are the natural language processing tasks supported in Product X?
Table 3: One question example from each experimental dataset.
1 2 3 4 5 6 7 8 9 10
# Iterations0.600.620.640.660.68Lexical ACBioASQ
1 2 3 4 5 6 7 8 9 10
# Iterations0.500.520.540.560.580.600.62Lexical ACClapNQ
Grid/Full
Grid/Sample
Random/FullRandom/Sample
TPE/Full
TPE/SampleGreedy-M/Full
Greedy-M/Sample
Figure 6: Per-iteration Lexical-AC performance on the
test sets of two largest datasets, for HPO algorithms
optimized using the fulldevelopment data (solid lines)
or asample of it (dotted). The dashed black lines show
the best achievable test performance.
The evaluation of the LLMaaJ-AC metric was
done with GPT4o-mini (OpenAI, 2024) as its back-
bone LLM. That model was used through Mi-
crosoft Azure. The overall cost was ∼500$.
E Use Of AI Assistants
AI Assistants were only used in writing for minor
edits and rephrases. They were also used to aid in
obtaining the correct LateX syntax for the various
figures.F Generation Details
Greedy decoding was used throughout all experi-
ments. The prompts were fixed to RAG prompts
tailored to each model: Figure 8 for Granite, Fig-
ure 9 for Llama and Figure 10 for Mistral. In each
prompt the {question} placeholder indicates where
the user question was placed, and {retrieved doc-
uments} the location of the retrieved chunks. For
Granite, each retrieved chunk was prefixed with
‘[Document] ’ and suffixed by ‘ [End] ’. Similarly,
for Llama each retrieved chunk was prefixed with
‘[document]: ’.
11

(a) Total number of tokens for chunks sent to the embedding models.
(b) Total number of tokens for prompts sent to the generation models.
Figure 7: Cost estimation for each algorithm after each iteration.
<|system|>
You are Granite Chat, an AI language model developed by IBM. You are a cautious assistant. You carefully follow
instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.
<|user|>
You are a AI language model designed to function as a specialized Retrieval Augmented Generation (RAG) assistant.
When generating responses, prioritize correctness, i.e., ensure that your response is grounded in context and user query.
Always make sure that your response is relevant to the question.
Answer Length: detailed
[Document]
{retrieved documents}
[End]
{question}
<|assistant|>
Figure 8: The prompt used for Granite.
12

<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful, respectful and honest assistant.
Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not
correct.
If you don’t know the answer to a question, please don’t share false information.
<|eot_id|><|start_header_id|>user<|end_header_id|>
[document]: {retrieved documents}
[conversation]: {question}. Answer with no more than 150 words. If you cannot base your answer on the given
document, please state that you do not have an answer.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Figure 9: The prompt used for Llama.
<s>[INST] «SYS»
You are a helpful, respectful and honest assistant.
Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not
correct.
If you don’t know the answer to a question, please don’t share false information.
«/SYS»
Generate the next agent response by answering the question. You are provided several documents with titles.
If the answer comes from different documents please mention all possibilities and use the titles of documents to
separate between topics or domains.
If you cannot base your answer on the given documents, please state that you do not have an answer.
{retrieved documents}
{question} [/INST]
Figure 10: The prompt used for Mistral.
13
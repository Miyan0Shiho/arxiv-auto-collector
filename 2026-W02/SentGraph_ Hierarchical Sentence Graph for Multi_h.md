# SentGraph: Hierarchical Sentence Graph for Multi-hop Retrieval-Augmented Question Answering

**Authors**: Junli Liang, Pengfei Zhou, Wangqiu Zhou, Wenjie Qing, Qi Zhao, Ziwen Wang, Qi Song, Xiangyang Li

**Published**: 2026-01-06 13:39:51

**PDF URL**: [https://arxiv.org/pdf/2601.03014v1](https://arxiv.org/pdf/2601.03014v1)

## Abstract
Traditional Retrieval-Augmented Generation (RAG) effectively supports single-hop question answering with large language models but faces significant limitations in multi-hop question answering tasks, which require combining evidence from multiple documents. Existing chunk-based retrieval often provides irrelevant and logically incoherent context, leading to incomplete evidence chains and incorrect reasoning during answer generation. To address these challenges, we propose SentGraph, a sentence-level graph-based RAG framework that explicitly models fine-grained logical relationships between sentences for multi-hop question answering. Specifically, we construct a hierarchical sentence graph offline by first adapting Rhetorical Structure Theory to distinguish nucleus and satellite sentences, and then organizing them into topic-level subgraphs with cross-document entity bridges. During online retrieval, SentGraph performs graph-guided evidence selection and path expansion to retrieve fine-grained sentence-level evidence. Extensive experiments on four multi-hop question answering benchmarks demonstrate the effectiveness of SentGraph, validating the importance of explicitly modeling sentence-level logical dependencies for multi-hop reasoning.

## Full Text


<!-- PDF content starts -->

SentGraph: Hierarchical Sentence Graph for Multi-hop
Retrieval-Augmented Question Answering
Junli Liang1, Pengfei Zhou1, Wangqiu Zhou2, Wenjie Qing1,
Qi Zhao1,Ziwen Wang1,Qi Song1*,Xiangyang Li1,
1University of Science and Technology of China,2Hefei University of Technology
{jlliang,pengfeizhou,qingwenjie,wangziwen}@mail.ustc.edu.cn;
rafazwq@hfut.edu.cn; zq2020email@163.com; {qisong09,xiangyangli}@ustc.edu.cn
Abstract
Traditional Retrieval-Augmented Generation
(RAG) effectively supports single-hop ques-
tion answering with large language models but
faces significant limitations in multi-hop ques-
tion answering tasks, which require combin-
ing evidence from multiple documents. Exist-
ing chunk-based retrieval often provides irrele-
vant and logically incoherent context, leading
to incomplete evidence chains and incorrect
reasoning during answer generation. To ad-
dress these challenges, we propose SentGraph,
a sentence-level graph-based RAG framework
that explicitly models fine-grained logical re-
lationships between sentences for multi-hop
question answering. Specifically, we construct
a hierarchical sentence graph offline by first
adapting Rhetorical Structure Theory to distin-
guish nucleus and satellite sentences, and then
organizing them into topic-level subgraphs with
cross-document entity bridges. During online
retrieval, SentGraph performs graph-guided ev-
idence selection and path expansion to retrieve
fine-grained sentence-level evidence. Exten-
sive experiments on four multi-hop question an-
swering benchmarks demonstrate the effective-
ness of SentGraph, validating the importance
of explicitly modeling sentence-level logical
dependencies for multi-hop reasoning.
1 Introduction
Large Language Models (LLMs) have demon-
strated strong capabilities in semantic understand-
ing and text generation, showing broad potential
in document reading comprehension tasks (Xiao
et al., 2023; Kumar, 2024; Ke et al., 2025). How-
ever, LLMs remain constrained by their internal
knowledge boundaries and are prone to hallucina-
tion (Bang et al., 2023; Huang et al., 2025), partic-
ularly in scenarios requiring strict factual accuracy.
To address these issues, Retrieval-Augmented
Generation (RAG) incorporates external knowl-
edge to support LLM generation (Lewis et al.,
*Corresponding authors
(a) Chunk-Level Graph
(b) Sentence-Level Graph
Sentence Unit Chunk Unit
S1
S2
S3Evidence Sentences
…
…
…
Chunk_3
Chunk_1
Chunk_2
Chunk_4
…
 S2
S3S1
…
……
Topic B
……
Topic A
…
… …
Topic N
S1
S3S2Figure 1: Comparison of the traditional chunk-level and
our adopted sentence-level graph construction methods.
2020; Gao et al., 2023). Traditional RAG methods
typically adopt a “chunk-index-retrieval” paradigm,
retrieving fixed-length text chunks based on se-
mantic similarity (Karpukhin et al., 2020; Gupta
et al., 2024). While effective for single-hop ques-
tion answering (QA), such methods struggle with
multi-hop question answering, which requires ag-
gregating evidence across multiple documents, of-
ten failing to capture complete evidence chains and
leading to incorrect or incomplete answers (Yang
et al., 2018; Ho et al., 2020; Shao et al., 2023).
To effectively tackle multi-hop question answer-
ing tasks, researchers have explored several solu-
tions. Post-retrieval optimization methods aim to
refine the retrieval results to improve their qual-
ity (Li et al., 2024; Xu et al., 2024). However, this
approach heavily depends on the accuracy of the
initial retrieval and often ignores the relationships
between evidence across different documents, mak-
ing it difficult to recover missing key evidence (Lee
et al., 2025). To address the limited recall of single-
step retrieval, iterative retrieval methods have been
1arXiv:2601.03014v1  [cs.CL]  6 Jan 2026

proposed to gradually construct an evidence chain
through multiple rounds of retrieval (Trivedi et al.,
2023; Sarthi et al., 2024). While iterative retrieval
can expand the coverage of retrieved evidence, the
repeated process introduces significant computa-
tional overhead and latency (Fang et al., 2025),
limiting its applicability in real-time scenarios.
In recent years, some prominent approaches have
begun to explicitly model cross-document relation-
ships by constructing graph-based knowledge struc-
tures offline, thereby reducing online inference la-
tency (Wang et al., 2024; Edge et al., 2024; Guo
et al., 2025). By organizing textual information into
structured graphs, these methods facilitate multi-
hop evidence capture during retrieval. Neverthe-
less, as shown in Figure 1(a), existing graph-based
methods often rely on similarity-based connections
at the chunk level, making it difficult to capture
fine-grained semantic and logical relationships be-
tween core sentences. Moreover, during online
retrieval, the returned chunks frequently contain
many sentences irrelevant to the query. Such redun-
dant information not only consumes context space
but also interferes with the reasoning process of
LLMs, increasing the risk of hallucination (Yoran
et al., 2024). Furthermore, weakly relevant yet cru-
cial evidence sentences may be overlooked because
they are not directly similar to the query, which can
ultimately lead to incorrect answers.
Inspired by these prior efforts and to overcome
their shortcomings, our basic idea is to propose
a sentence graph-based retrieval-enhanced gener-
ation method. Specifically, we attempt to reduce
the retrieval granularity from chunks to individ-
ual sentences and to explicitly model the semantic
and logical relationships between sentences using
a graph structure. However, constructing sentence-
level graphs faces three technical challenges:(1)
Context loss: Sentences containing anaphoric ex-
pressions such as pronouns and deictic terms lose
their specific referents when isolated from chunk
context, which can lead to ambiguity in understand-
ing what entities or concepts they refer to.(2) Re-
lationship modeling complexity: Different from
chunk-level connections that can rely on surface-
level similarity, sentence-level relationships are far
more diverse and complex, encompassing various
logical types such as causality, condition, and con-
trast. This diversity poses challenges in both deter-
mining which relationship types to represent and
accurately identifying them between sentences.(3)
High computational overhead: The number of sen-tences in documents is significantly larger than the
number of text chunks, making global sentence
graph construction computationally expensive and
potentially impractical for large-scale applications.
To overcome these challenges, we propose an
offline hierarchical sentence graph construction
framework, as shown in Figure 1(b). To address
context loss and relationship modeling complex-
ity, we employ a refined set of logical relationships
based on Rhetorical Structure Theory (RST) (Mann
and Thompson, 1988). Specifically, we adapt RST
by consolidating frequently occurring relations and
removing rare ones, resulting in a practical relation-
ship taxonomy that naturally distinguishes between
nucleus sentences and satellite sentences. To tackle
high computational overhead, we design a three-
layer graph structure with topic, core sentence, and
supplementary sentence layers. Instead of building
a dense global sentence graph, we first construct
topic-level subgraphs within individual documents,
then establish cross-document connections through
entity-concept bridges at the topic layer. In the on-
line retrieval stage, we introduce a sentence graph-
based retrieval-enhanced generation strategy that
enables fine-grained evidence selection, thereby re-
ducing irrelevant context and token consumption
during LLM generation.
We summarize our contributions as follows:
•We are the first to apply Rhetorical Structure
Theory to sentence-level graph construction
for retrieval-augmented generation, providing
a principled approach to model fine-grained
logical relationships between sentences.
•We propose an offline hierarchical sentence
graph construction method with dual logical
relationship modeling, along with an online
sentence-level retrieval strategy that leverages
the graph structure to retrieve key evidence
sentences with their logical context for multi-
hop question answering tasks.
•Extensive experiments on multi-hop question
answering benchmarks demonstrate the supe-
rior performance of our approach and validate
the effectiveness of our framework.
2 Related Work
2.1 Retrieval-Augmented Generation
Retrieval-Augmented Generation has shown strong
performance in question answering tasks (Lewis
2

et al., 2020; Karpukhin et al., 2020). However,
although standard RAG models perform well on
single-hop questions, they face clear challenges on
multi-hop QA tasks (Yang et al., 2018).
To improve retrieval quality, several extensions
have been proposed. Refiner (Li et al., 2024) ex-
tracts query-relevant content and reorganizes it in
a structured form, which helps LLMs better align
with the original context. (Lee et al., 2025) shifts re-
trieval from ranking individual passages to optimiz-
ing the overall quality of a passage set. However,
the performance of these methods still depends
heavily on the quality of the initial retrieval results.
To overcome the limited coverage of single-
round retrieval, (Trivedi et al., 2023) alternates be-
tween chain-of-thought reasoning and knowledge
retrieval. The reasoning process guides retrieval,
while retrieved evidence is used to refine reasoning.
This design targets knowledge-intensive multi-hop
question answering. KiRAG (Fang et al., 2025) fur-
ther introduces iterative retrieval based on knowl-
edge triples. It explicitly integrates reasoning into
the retrieval process to adapt to changing informa-
tion needs. While these approaches improve evi-
dence coverage, multi-round retrieval introduces
unavoidable computational overhead.
2.2 Graph-Based RAG Methods
Graph-based representations offer a promising al-
ternative. They encode document relations offline,
reducing the reliance on repeated online retrieval.
KGP (Wang et al., 2024) constructs a knowledge
graph over multiple documents and designs an
LLM-based graph traversal agent. This agent sup-
ports cross-document retrieval and question answer-
ing. GraphRAG (Edge et al., 2024) introduces
a graph-based approach that leverages LLMs to
extract entities and relationships from documents,
constructing a knowledge graph that captures se-
mantic structure. Through community detection, it
generates hierarchical summaries at multiple lev-
els, enabling both local and global reasoning for
complex queries. LightRAG (Guo et al., 2025)
proposes a graph-enhanced retrieval framework.
Instead of relying on community detection and hi-
erarchical summarization, it adopts a dual-level
retrieval strategy that enables both low-level and
high-level knowledge discovery.
However, most existing graph-based methods op-
erate at the chunk level. They treat multi-sentence
text chunks as graph nodes, which limits their abil-
ity to capture fine-grained semantic relations. Inaddition, retrieved chunks often contain irrelevant
sentences, introducing noise and increasing the risk
of missing weakly related but critical evidence due
to coarse chunk-level similarity.
3 The Proposed SentGraph Method
SentGraph is a sentence-level logic graph construc-
tion and retrieval-augmented generation framework
for multi-hop question answering. It departs from
the traditional chunk-based retrieval and model-
ing paradigm by explicitly capturing sentence-level
logical dependencies within and across documents.
This design enables finer-grained knowledge orga-
nization and reasoning path modeling. As shown
in Figure 2, the framework consists of two stages:
offline sentence logic graph construction and on-
line graph-based retrieval and answer generation.
In the offline stage, we construct a hierarchical
sentence logic graph by modeling sentence-level
logical structures. In the online stage, we perform
evidence retrieval and answer generation based on
the graph, thereby improving LLMs’ effectiveness
for cross-document reasoning on complex ques-
tions.
3.1 Offline Stage: Hierarchical Sentence
Logic Graph Construction
Given a document collection, denoted as D=
{d1, d2, . . . , d n}, we first decompose each docu-
ment into sentence-level semantic units. Compared
to traditional chunk-level modeling, sentence-level
units provide finer-grained evidence granularity,
but also introduce several challenges, including
context loss, complex relation modeling, and high
computational overhead when directly construct-
ing a global sentence graph. To address these chal-
lenges, we propose a hierarchical graph construc-
tion method based on refined Rhetorical Structure
Theory (RST). This method alleviates context loss
and relation complexity through RST-based logical
relation modeling, while reducing computational
overhead through hierarchical architecture design.
Considering that RST defines many relation
types, some of which have limited discriminative
power for reasoning and QA, we simplify the re-
lation set from the perspective of reasoning func-
tion and evidence organization, retaining only rela-
tion types that substantially impact cross-document
reasoning. Specifically, based on the functional
roles of sentences in discourse, we define two
categories of relations:Nucleus-Nucleus (N-N)
3

Intra-Document
…
1
2
3…
 k
1
2
3…
 k
1
2
3…
 kSplitTopic A
Topic B…
Topic NDocuments Sentences
 Link Bridge
Topic A
… …
…
…
Topic B
…… …
Topic N
… …
Cross-Document
… …
Topic N
Topic B
…… …
Topic A
… …
…
Offline Sentence Logic Graph Construction
Anchor Selection & Refinement Path Expansion User Query Answer
Query Embedding
Q: When did the country 
that has the same co-
official language as 
that of the movie about 
the city where Petar 
Trifunović died send an 
independent team to the 
Olympics?
Online Graph-Based Retrieval and Answer Generation
A: 2016 
… …
Topic N
Topic B
…… …
Topic A
… …
…
… …
Topic N
Topic B
…… …
Topic A
… …
…
Evidence 
SentencesS1: Dr. Petar Trifunović (31 
August 1910, Dubrovnik – 8...
S2: ... It is the official 
language of Serbia ... 
S3: ... national team at the 
2016 Summer Olympics.
Anchor Selection & Refinement Path Expansion User Query AnswerFigure 2: An overview of the SentGraph framework. The offline stage constructs a hierarchical sentence logic graph,
and the online stage performs graph-based retrieval and answer generation for multi-hop question answering.
relationsmodel logical connections between sen-
tences of equal importance that jointly convey core
document semantics, including conjunction, con-
trast, disjunction, multinuclear restatement, and se-
quence relations, whereasNucleus-Satellite (N-S)
relationsmodel asymmetric dependencies between
core sentences and their supporting sentences, in-
cluding cause, result, opposition, elaboration, cir-
cumstance, evaluation, and solutionhood relations.
Based on the above relation modeling, we design
a three-layer graph structure that balances expres-
sive power and computational complexity through
hierarchical organization, avoiding the computa-
tional overhead of directly constructing a global
sentence graph. The graph is formally defined as:
G= (V, E)
where the node set is:
V=V t∪V c∪V s
Here, topic nodes Vtrepresent document-level
semantic summaries for cross-document bridging,
core sentence nodes Vccorrespond to sentences
carrying key facts and reasoning support, and sup-
plementary sentence nodes Vsrepresent subordi-
nate sentences that elaborate on or conditionally
supplement core sentences.
The edge set is formally defined as:
E=E tt∪E tc∪E cc∪E csHere, inter-topic bridging edges Ettestablish
cross-document semantic connections, topic-to-
core edges Etcassociate topics with their subor-
dinate core sentences, core-to-core edges Eccrep-
resent N-N relations such as parallel, contrast, and
sequence, and core-to-supplementary edges Ecs
represent N-S dependencies such as cause, back-
ground, and evaluation.
The construction process consists of two
steps: intra-document logic modeling and cross-
document semantic bridging.
In the intra-document logic modeling stage, we
first employ LLMs guided by refined RST to iden-
tify core sentences carrying main facts and deter-
mine N-N logical relations between them. We then
cluster non-core sentences and assign them to cor-
responding core sentences based on semantic simi-
larity and contextual distance, before using LLMs
to establish N-S subordinate relations. This process
captures the hierarchical logical structure within
each document. Detailed prompt templates are pro-
vided in Appendix A.1 and Appendix A.2.
In the cross-document semantic bridging stage,
we leverage the background knowledge of LLMs to
identify commonsense relations between topic enti-
ties across different documents, and establish inter-
topic bridging edges Ettbetween topic nodes. This
enables the reasoning process to integrate key evi-
dence across documents and form cross-document
4

reasoning chains. The detailed prompt template is
provided in Appendix A.3.
3.2 Online Stage: Graph-Based Retrieval and
Answer Generation
Given a user query, the online process consists of
three modules: anchor selection and refinement,
adaptive path expansion, and answer generation.
Anchor selection and refinement.We adopt a
coarse-to-fine two-stage strategy. First, a retriever
computes similarity scores between the query and
all graph nodes, and the Top- Khighest-scoring
nodes are selected as candidate anchors. These can-
didates are then refined by the LLM, which filters
out loosely related nodes and evaluates whether
the remaining anchors contain sufficient evidence.
If the evidence is sufficient, the process proceeds
directly to answer generation. Otherwise, the pro-
cess triggers path expansion to retrieve additional
evidence. The detailed prompt template used for
anchor refinement is provided in Appendix A.4.
Adaptive path expansion.We explore reason-
ing paths starting from each anchor using a breadth-
first strategy. For each anchor, we maintain a
path queue and expand paths along graph edges
by selecting neighboring nodes based on similarity.
Newly selected nodes are appended to the current
path until a predefined maximum path length or
expansion limit is reached.
Answer generation.We extract all sentence
nodes from the retained paths to form the final ev-
idence set. Then, we provide this evidence along
with the query to the LLM, instructing it to gen-
erate a final answer based on the given context.
The LLM performs multi-hop reasoning over the
evidence and generates the final answer.
4 Experiment
4.1 Experiment Setting
Datasets.We evaluate our approach on four com-
plex multi-hop question answering datasets. Hot-
potQA (Yang et al., 2018) contains 113k question-
answer pairs that require cross-document reason-
ing and provide sentence-level supporting facts.
2WikiMultiHopQA (2Wiki) (Ho et al., 2020) is a
multi-hop question answering dataset constructed
from Wikipedia, requiring reasoning across mul-
tiple documents. MuSiQue (Trivedi et al., 2022)
contains approximately 25k questions and enforces
multi-step reasoning with 2–4 hops to avoid short-
cut solutions. MultiHopRAG(MultiHop) (Tang andYang, 2024) is a benchmark designed for RAG sys-
tems and includes 2,556 questions whose answers
must be synthesized from 2–4 news articles.
Source Models.We evaluate our method using
two retrieval models and multiple LLMs. For re-
trieval, we use BM25 (Robertson et al., 2009), a
traditional unsupervised method, and bge-large-
en-v1.5(BGE) (Xiao et al., 2024), a supervised
dense retriever. For generation, we use Llama-
3.1-8B-Instruct (Grattafiori et al., 2024) as the
primary LLM. To evaluate generalization across
model scales, we additionally test Qwen2.5-7B-
Instruct, Qwen2.5-14B-Instruct, and Qwen2.5-
32B-Instruct (Yang et al., 2024; Team, 2024).
Baselines.(1)Retrieval Only(RO)serves as a ba-
sic baseline and follows the standard in-context
RAG paradigm (Ram et al., 2023). We evalu-
ate both passage-level and sentence-level retrieval
units, where the sentence-level baseline uses simple
document sentence splitting. (2)RankLlama(Ma
et al., 2024) andSetR-CoT & IRI(Lee et al., 2025)
are post-retrieval optimization methods. (3)Ki-
RAG(Fang et al., 2025) represents iterative re-
trieval strategies. (4)LightRAG(Guo et al., 2025)
andKGP(Wang et al., 2024) are graph-enhanced
RAG methods that incorporate explicit graph struc-
tures for multi-hop QA.
For fair comparison, we restrict all baseline
methods to a single retrieval step, focusing the com-
parison on retrieval granularity and structural mod-
eling. Additionally, in terms of retrieval granularity,
all baseline methods follow prior work and operate
on passage-level retrieval units, unless otherwise
specified. In our experimental setting, passage-
level retrieval constitutes a concrete instantiation
of chunk-level retrieval. In contrast, our method
operates on sentence-level retrieval units.
Metrics.Following common practice in prior RAG
evaluations, we uniformly sample 500 questions
from each dataset for all methods and use the
same evaluation subsets to ensure fair comparison.
We use Exact Match(EM %) and F1 score( %) as
the primary evaluation metrics. For the MultiHo-
pRAG dataset, we report Accuracy, following its
previous evaluation protocol. All experiments are
conducted on a Linux server equipped with four
NVIDIA A800 80GB GPUs, dual Intel Xeon CPUs
(2.9 GHz), and 512 GB of memory. All models are
evaluated with deterministic decoding, with the
temperature set to 0.
5

Retrieval ModelRetrieval
UnitsAvg
# UnitsHotpotQA 2Wiki MuSiQue MultiHop
EM F1 EM F1 EM F1 Accuracy
BM25Retrieval Only Passage 3.00 9.20 15.20 3.40 9.65 1.40 4.27 37.20
Retrieval Only Sentence 3.00 31.20 42.44 23.60 29.29 9.60 16.86 61.60
RankLlama Passage 5.00 29.48 27.82 30.30 21.91 6.04 9.26 42.09
SetR-CoT & IRI Passage 2.63 32.20 30.57 32.17 24.22 6.62 10.57 44.13
KiRAG Passage 3.00 17.40 23.40 24.00 32.70 9.00 18.20 26.50
KGP Passage 3.00 36.82 49.94 22.80 31.21 9.31 18.66 62.00
LightRAG Passage 3.00 28.66 39.06 17.78 27.43 9.83 18.71 26.65
SentGraph(Ours) Sentence 2.57 43.80 57.13 32.20 39.48 16.00 26.83 63.40
BGERetrieval Only Passage 3.00 11.00 18.09 3.80 10.34 2.40 6.86 44.40
Retrieval Only Sentence 3.00 38.60 50.88 30.80 40.01 17.00 28.57 60.20
RankLlama Passage 5.00 31.88 32.95 32.24 25.78 7.61 11.77 43.51
SetR-CoT & IRI Passage 2.91 36.62 38.11 35.44 30.35 10.79 15.43 47.14
KiRAG Passage 3.00 18.20 24.50 26.40 35.00 6.20 17.00 22.50
KGP Passage 3.00 44.00 58.73 36.80 48.20 21.20 34.72 63.40
LightRAG Passage 3.00 27.17 37.75 17.40 26.99 8.60 17.71 20.44
SentGraph(Ours) Sentence 2.70 48.80 62.92 42.00 52.26 26.80 40.36 65.60
Table 1: Performance (%) comparison on four multi-hop question answering datasets (passage-level as chunk-level
instantiation). Bold and underlined indicate the best and second best performance, respectively.
4.2 Main Results
We compare SentGraph with baselines on four
multi-hop question answering datasets under both
sparse (BM25) and dense (BGE) retrieval settings
in Table 1. SentGraph consistently achieves the
best performance across all datasets and retrieval
settings. It is important to note that SentGraph op-
erates on sentence-level retrieval units while most
baselines use passage-level units, demonstrating
that fine-grained evidence modeling with structured
logical dependencies is more effective than coarse-
grained retrieval for multi-hop QA. Next, we sum-
marize key observations as follows:
Fine-grained retrieval is necessary but insuffi-
cient.Sentence-level retrieval significantly outper-
forms passage-level retrieval in the retrieval-only
setting. For instance, under BM25, sentence-level
units achieve 31.20 EM on HotpotQA compared to
only 9.20 EM for passage-level units, confirming
the importance of fine-grained retrieval for multi-
hop reasoning. However, sentence-level retrieval
alone still substantially underperforms SentGraph,
with gaps of 12.6 EM points on HotpotQA and 10.2
points on 2Wiki under BM25. This indicates that
modeling logical dependencies between sentences
is critical beyond granularity alone.
Post-retrieval and iterative methods show lim-
ited gains.RankLlama and SetR-CoT & IRI im-
prove over retrieval-only baselines through result
refinement, but their effectiveness remains con-
strained by initial retrieval quality. KiRAG at-
tempts to discover missing evidence through iter-
ative retrieval, but achieves limited improvements
when the number of iterations is restricted, obtain-ing only 17.40 and 18.20 EM on HotpotQA under
BM25 and BGE respectively.
Graph-enhanced methods benefit from struc-
ture but lack fine-grained modeling.LightRAG
and KGP outperform retrieval-only baselines by
introducing explicit structural connections. Among
all baselines, KGP achieves the strongest perfor-
mance, reaching 44.00 EM on HotpotQA and
36.80 EM on 2WikiMultiHopQA under BGE re-
trieval. However, these methods typically con-
struct graphs at the passage level, which limits
their ability to capture fine-grained logical rela-
tions between sentences. In contrast, SentGraph
models sentence-level logical dependencies and
achieves superior performance. Under BGE setting,
SentGraph outperforms KGP by 4.8 EM points on
HotpotQA, 5.2 points on 2WikiMultiHopQA, and
5.6 points on MuSiQue. This demonstrates that
SentGraph’s gains arise from fine-grained evidence
selection and structured reasoning paths rather than
increased context length.
4.3 Results across Different Base LLMs
Table 2 presents the performance of SentGraph
across multiple base LLMs with different model
sizes, ranging from 7B to 32B parameters. We
evaluate methods under two key dimensions: re-
trieval granularity (passage-level vs. sentence-
level) and structural modeling (retrieval-only vs.
graph-based). SentGraph consistently outperforms
all baselines across different model scales, demon-
strating the necessity of modeling logical depen-
dencies at the sentence level for multi-hop QA.
Specifically, across all LLMs, sentence-level
6

HotpotQA 2Wiki MuSiQue MultiHopRetrieval ModelRetrieval
UnitAvg
# Units EM F1 EM F1 EM F1 Accuracy
Base LLM: Qwen2.5-7B-Instruct
RO Passage 3.00 14.20 19.24 16.00 18.00 2.20 4.42 35.20
KGP Passage 3.00 43.66 56.14 32.00 38.14 13.60 22.13 67.40
RO Sentence 3.00 37.60 49.18 26.80 32.02 11.60 19.97 67.00BM25
SentGraph Sentence 2.89 48.80 61.98 44.40 52.53 25.00 35.09 68.80
RO Passage 3.00 13.60 18.26 10.00 10.94 3.20 4.60 39.40
KGP Passage 3.00 53.66 67.25 49.20 57.31 28.40 42.48 69.00
RO Sentence 3.00 43.80 56.52 40.80 49.71 22.40 34.53 66.00BGE
SentGraph Sentence 2.44 55.60 68.88 52.60 61.49 36.00 48.27 70.20
Base LLM: Llama-3.1-8B-Instruct
RO Passage 3.00 9.20 15.20 3.40 9.65 1.40 4.27 37.20
KGP Passage 3.00 36.82 49.94 22.80 31.21 9.31 18.66 62.00
RO Sentence 3.00 31.20 42.44 23.60 29.29 9.60 16.86 61.60BM25
SentGraph Sentence 2.57 43.80 57.13 32.20 39.48 16.00 26.83 63.40
RO Passage 3.00 11.00 18.09 3.80 10.34 2.40 6.86 44.40
KGP Passage 3.00 44.00 58.73 36.80 48.20 21.20 34.72 63.40
RO Sentence 3.00 38.60 50.88 30.80 40.01 17.00 28.57 60.20BGE
SentGraph Sentence 2.70 48.80 62.92 42.00 52.26 26.80 40.36 65.60
Base LLM: Qwen2.5-14B-Instruct
RO Passage 3.00 14.00 18.91 16.20 18.26 2.20 2.42 35.60
KGP Passage 3.00 43.66 56.06 32.20 38.37 11.60 19.89 67.20
RO Sentence 3.00 37.20 48.94 27.00 32.08 13.20 22.10 66.80BM25
SentGraph Sentence 2.89 49.00 62.36 43.80 51.84 24.00 33.93 69.20
RO Passage 3.00 13.60 18.25 10.00 10.88 3.00 4.51 39.20
KGP Passage 3.00 53.66 67.06 48.20 56.60 28.40 42.37 69.00
RO Sentence 3.00 43.00 56.14 40.80 49.60 22.60 34.90 66.20BGE
SentGraph Sentence 2.61 55.40 68.74 54.20 63.05 36.80 49.30 70.00
Base LLM: Qwen2.5-32B-Instruct
RO Passage 3.00 20.80 27.78 24.00 26.88 3.40 7.75 44.80
KGP Passage 3.00 46.08 58.29 34.40 40.20 14.60 25.25 66.40
RO Sentence 3.00 40.60 51.70 31.80 37.22 12.40 21.27 67.20BM25
SentGraph Sentence 2.54 51.40 64.03 45.80 53.69 25.20 36.78 72.40
RO Passage 3.00 21.00 28.52 23.40 26.24 5.00 9.16 51.20
KGP Passage 3.00 55.28 68.89 51.80 60.53 28.20 41.41 67.80
RO Sentence 3.00 43.20 57.01 45.00 52.76 25.00 36.89 68.80BGE
SentGraph Sentence 2.62 57.60 70.64 55.40 64.73 38.80 52.01 73.00
Table 2: Performance (%) comparison across four multi-hop question answering datasets at different LLM scales
(passage-level as chunk-level instantiation). “RO” denotes retrieval-only. Bold and underlined values indicate the
best and second-best results, respectively.
retrieval-only consistently outperforms passage-
level retrieval-only by substantial margins. For
example, with Qwen2.5-32B under BGE retrieval,
sentence-level retrieval achieves 43.20 EM on Hot-
potQA and 45.00 EM on 2WikiMultiHopQA, com-
pared to only 21.00 and 23.40 EM for passage-
level retrieval. This once again confirms that fine-
grained evidence units are critical for capturing
precise multi-hop reasoning paths. Moreover, we
observe that more powerful LLMs generally im-
prove the performance of all methods. However,
the improvement achieved by retrieval-only is lim-
ited. This further indicates that granularity alone is
insufficient and that proper evidence organization
remains crucial even with more capable LLMs.
The graph-based method KGP improves over
passage-level retrieval-only across all LLM scales.
With Qwen2.5-32B under BGE retrieval, KGPreaches 55.28 EM on HotpotQA and 51.80 EM
on 2WikiMultiHopQA, representing gains of 34.28
and 28.40 EM points over passage-level retrieval-
only. This confirms the benefit of explicit struc-
tural modeling for multi-hop reasoning. However,
coarse-grained passage units inherently mix rele-
vant and irrelevant sentences, limiting the effec-
tiveness of graph-based reasoning. In contrast,
SentGraph constructs a hierarchical sentence-level
graph that explicitly models logical dependencies
between sentences, enabling fine-grained evidence
selection and structured reasoning paths. With
the same LLM, SentGraph achieves 57.60 EM on
HotpotQA, 55.40 EM on 2WikiMultiHopQA, and
38.80 EM on MuSiQue, outperforming sentence-
level retrieval-only by 14.4, 10.4, and 13.8 EM
points, and surpassing KGP by 2.32, 3.6, and
10.6 EM points respectively. Notably, these im-
7

Components HotpotQA 2Wiki MuSiQue
AS AER GPE EM F1 EM F1 EM F1
✓× × 37.60 50.62 28.00 35.52 16.00 27.67
✓ ✓× 44.80 59.37 37.40 46.96 25.40 37.57
✓ ✓ ✓ 48.80 62.92 42.00 52.26 26.80 40.36
Table 3: Ablation study on core components.
provements remain consistent across different LLM
scales, demonstrating robust and scalable gains.
4.4 Ablation Study
Table 3 reports the ablation results of SentGraph by
progressively enabling its core components. AS de-
notes anchor selection, AER denotes adaptive evi-
dence refinement, and GPE denotes guided path ex-
pansion. AS alone provides baseline performance
by identifying locally relevant sentences, but re-
mains insufficient for complex multi-hop reason-
ing. Introducing AER leads to substantial improve-
ments, with EM gains of 7.2 points on HotpotQA,
9.4 points on 2WikiMultiHopQA, and 9.4 points
on MuSiQue, highlighting the importance of fil-
tering irrelevant anchors and assessing evidence
sufficiency. Further adding GPE yields additional
gains of 4.0, 4.6, and 1.4 EM points, respectively.
These results demonstrate that SentGraph benefits
from the complementary roles of its components,
where anchor selection provides initial candidates,
evidence refinement filters noise and assesses suf-
ficiency, and path expansion broadens evidence
coverage to support multi-hop reasoning.
Figure 3 presents the impact of the number of
anchors on SentGraph performance. As the num-
ber of anchors increases from 5 to 25, performance
consistently improves, with gains of 6.2 EM points
on HotpotQA, 8.8 points on 2WikiMultiHopQA,
and 8.4 points on MuSiQue. Beyond 20 anchors,
the performance improvement becomes smaller.
This indicates that an adequate pool of candidate
anchors is crucial for capturing diverse reasoning
paths in multi-hop questions, but excessively large
anchor sets yield diminishing returns.
4.5 Performance and Efficiency Analysis
Figure 4 compares the token usage of SentGraph
and KGP. SentGraph achieves consistent reductions
in both input and output token consumption while
maintaining superior performance. For input to-
kens, SentGraph reduces context length by 29.99%
on HotpotQA, 45.26% on 2WikiMultiHopQA, and
30.38% on MuSiQue compared to KGP. These re-
ductions stem from sentence-level retrieval granu-HotpotQA 2WikiMultiHopQA MuSiQue
51015202530152025303540455055
Anchor NumberEM (%)
(a) EM scores across differ-
ent anchor numbers.51015202530303540455055606570
Anchor NumberF1 (%)
(b) F1 scores across differ-
ent anchor numbers.
Figure 3: Performance across multi-hop question an-
swering datasets with varying anchor numbers.
KGP SentGraph
HotpotQA 2Wiki MuSiQue0100200300400Input Tokens
(a) Average input token
usage per query.HotpotQA 2Wiki MuSiQue0102030Output Tokens
(b) Average output token
usage per query.
Figure 4: Efficiency analysis on average token usage
per query across multi-hop question answering datasets.
larity, which enables more fine-grained evidence
selection and helps reduce irrelevant context that
is often included in passage-level retrieval. Out-
put token savings are even more pronounced, with
reductions of 69.00% on HotpotQA, 18.56% on
2WikiMultiHopQA, and 9.22% on MuSiQue. This
indicates that cleaner input evidence leads to more
concise and focused generation. Combined with
the performance improvements shown in Table 1,
these results demonstrate that SentGraph achieves
better accuracy with lower computational cost.
5 Conclusion
We propose SentGraph, a sentence-level graph-
based RAG framework for multi-hop QA that con-
structs hierarchical sentence graphs with explicit
logical dependencies by adapting RST. SentGraph
further employs a graph-guided retrieval strategy
to enable fine-grained evidence selection at the sen-
tence level. Extensive experiments show that Sent-
Graph achieves consistent performance improve-
ments with lower token consumption, highlighting
the importance of fine-grained logical dependency
modeling for effective multi-hop QA.
8

Limitations
Despite its effectiveness, SentGraph has several
limitations. First, the construction of the hierar-
chical sentence graph relies on LLMs to identify
N-N and N-S relations, as well as cross-document
semantic bridges. While LLMs provide strong gen-
eralization capabilities, their predictions may in-
troduce noise, which could affect the quality of
the constructed graph. We also observe that the
quality of relation annotation is sensitive to the ca-
pacity of the underlying LLMs, with larger models
tending to produce more reliable structures. How-
ever, we do not explicitly quantify the impact of
annotation errors on downstream reasoning perfor-
mance. Future work could investigate more robust
graph construction strategies or human-in-the-loop
validation mechanisms.
Second, our adaptation of Rhetorical Structure
Theory focuses on a refined set of relation types tai-
lored specifically to multi-hop question answering.
While effective for this task, it may not fully cap-
ture all discourse phenomena, and its applicability
to other downstream tasks remains to be explored.
Finally, SentGraph emphasizes online inference
efficiency by shifting most computation to an of-
fline stage. However, the offline graph construc-
tion process introduces additional computational
costs, especially when applied to large-scale cor-
pora. Future work could explore more efficient or
incremental graph construction strategies.
Ethics Statement
This work focuses on improving multi-hop retrieval
and reasoning through structured sentence-level
representations. The proposed method is evalu-
ated on publicly available benchmark datasets and
does not involve the collection or use of personal,
sensitive, or proprietary data. SentGraph does not
train or modify LLMs. It operates as a retrieval
framework that provides structured context to ex-
isting LLMs for answer generation. Like other
RAG frameworks, ethical considerations depend
on downstream application contexts and require
appropriate safeguards during deployment.
References
Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wen-
liang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei
Ji, Tiezheng Yu, Willy Chung, Quyet V . Do, Yan
Xu, and Pascale Fung. 2023. A multitask, multilin-
gual, multimodal evaluation of chatgpt on reasoning,hallucination, and interactivity. InProceedings of
the 13th International Joint Conference on Natural
Language Processing and the 3rd Conference of the
Asia-Pacific Chapter of the Association for Compu-
tational Linguistics, IJCNLP 2023 -Volume 1: Long
Papers, Nusa Dua, Bali, November 1 - 4, 2023, pages
675–718. Association for Computational Linguistics.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130.
Jinyuan Fang, Zaiqiao Meng, and Craig MacDonald.
2025. Kirag: Knowledge-driven iterative retriever
for enhancing retrieval-augmented generation. In
Proceedings of the 63rd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), ACL 2025, Vienna, Austria, July 27 -
August 1, 2025, pages 18969–18985. Association for
Computational Linguistics.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen
Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.
arXiv preprint arXiv:2312.10997, 2(1).
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models.arXiv preprint arXiv:2407.21783.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2025. LightRAG: Simple and fast retrieval-
augmented generation. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2025,
pages 10746–10761, Suzhou, China. Association for
Computational Linguistics.
Shailja Gupta, Rajesh Ranjan, and Surya Narayan
Singh. 2024. A comprehensive survey of retrieval-
augmented generation (rag): Evolution, current
landscape and future directions.arXiv preprint
arXiv:2410.12837.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing A multi-hop
QA dataset for comprehensive evaluation of reason-
ing steps. InProceedings of the 28th International
Conference on Computational Linguistics, COLING
2020, Barcelona, Spain (Online), December 8-13,
2020, pages 6609–6625. International Committee on
Computational Linguistics.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
Liu. 2025. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions.ACM Trans. Inf. Syst., 43(2):42:1–
42:55.
9

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing, EMNLP 2020, Online, Novem-
ber 16-20, 2020, pages 6769–6781. Association for
Computational Linguistics.
Wenjun Ke, Yifan Zheng, Yining Li, Hengyuan Xu,
Dong Nie, Peng Wang, and Yao He. 2025. Large
language models in document intelligence: A com-
prehensive survey, recent advances, challenges, and
future trends.ACM Transactions on Information Sys-
tems, 44(1):1–64.
Pranjal Kumar. 2024. Large language models (llms):
survey, technical frameworks, and future challenges.
Artif. Intell. Rev., 57(9):260.
Dahyun Lee, Yongrae Jo, Haeju Park, and Moontae
Lee. 2025. Shifting from ranking to set selection
for retrieval augmented generation. InProceedings
of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
ACL 2025, Vienna, Austria, July 27 - August 1, 2025,
pages 17606–17619. Association for Computational
Linguistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural In-
formation Processing Systems 33: Annual Confer-
ence on Neural Information Processing Systems 2020,
NeurIPS 2020, December 6-12, 2020, virtual.
Zhonghao Li, Xuming Hu, Aiwei Liu, Kening Zheng,
Sirui Huang, and Hui Xiong. 2024. Refiner: Re-
structure retrieval content efficiently to advance
question-answering capabilities.arXiv preprint
arXiv:2406.11357.
Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and
Jimmy Lin. 2024. Fine-tuning llama for multi-stage
text retrieval. InProceedings of the 47th Inter-
national ACM SIGIR Conference on Research and
Development in Information Retrieval, pages 2421–
2425.
William C Mann and Sandra A Thompson. 1988.
Rhetorical structure theory: Toward a functional the-
ory of text organization.Text-interdisciplinary Jour-
nal for the Study of Discourse, 8(3):243–281.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models.Transactions of the Association for
Computational Linguistics, 11:1316–1331.
Stephen Robertson, Hugo Zaragoza, and 1 others. 2009.
The probabilistic relevance framework: Bm25 and
beyond.Foundations and Trends® in Information
Retrieval, 3(4):333–389.Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D. Manning.
2024. RAPTOR: recursive abstractive processing for
tree-organized retrieval. InThe Twelfth International
Conference on Learning Representations, ICLR 2024,
Vienna, Austria, May 7-11, 2024. OpenReview.net.
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie
Huang, Nan Duan, and Weizhu Chen. 2023. En-
hancing retrieval-augmented large language models
with iterative retrieval-generation synergy. InFind-
ings of the Association for Computational Linguis-
tics: EMNLP 2023, Singapore, December 6-10, 2023,
pages 9248–9274. Association for Computational
Linguistics.
Yixuan Tang and Yi Yang. 2024. Multihop-rag: Bench-
marking retrieval-augmented generation for multi-
hop queries.arXiv preprint arXiv:2401.15391.
Qwen Team. 2024. Qwen2.5: A party of foundation
models.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. Musique: Multi-
hop questions via single-hop question composition.
Trans. Assoc. Comput. Linguistics, 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st annual meeting of the association for com-
putational linguistics (volume 1: long papers), pages
10014–10037.
Yu Wang, Nedim Lipka, Ryan A. Rossi, Alexa F. Siu,
Ruiyi Zhang, and Tyler Derr. 2024. Knowledge
graph prompting for multi-document question an-
swering. InThirty-Eighth AAAI Conference on Artifi-
cial Intelligence, AAAI 2024, Thirty-Sixth Conference
on Innovative Applications of Artificial Intelligence,
IAAI 2024, Fourteenth Symposium on Educational
Advances in Artificial Intelligence, EAAI 2014, Febru-
ary 20-27, 2024, Vancouver, Canada, pages 19206–
19214. AAAI Press.
Changrong Xiao, Sean Xin Xu, Kunpeng Zhang, Yufang
Wang, and Lei Xia. 2023. Evaluating reading com-
prehension exercises generated by llms: A showcase
of chatgpt in education applications. InProceedings
of the 18th Workshop on Innovative Use of NLP for
Building Educational Applications, BEA@ACL 2023,
Toronto, Canada, 13 July 2023, pages 610–625. As-
sociation for Computational Linguistics.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muen-
nighoff, Defu Lian, and Jian-Yun Nie. 2024. C-pack:
Packed resources for general chinese embeddings. In
Proceedings of the 47th international ACM SIGIR
conference on research and development in informa-
tion retrieval, pages 641–649.
Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2024. RE-
COMP: improving retrieval-augmented lms with con-
text compression and selective augmentation. InThe
10

Twelfth International Conference on Learning Rep-
resentations, ICLR 2024, Vienna, Austria, May 7-11,
2024. OpenReview.net.
An Yang, Baosong Yang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan
Li, Dayiheng Liu, Fei Huang, Guanting Dong, Hao-
ran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian
Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, and
40 others. 2024. Qwen2 technical report.arXiv
preprint arXiv:2407.10671.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing. InProceedings of the 2018 Conference on Em-
pirical Methods in Natural Language Processing,
Brussels, Belgium, October 31 - November 4, 2018,
pages 2369–2380. Association for Computational
Linguistics.
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Be-
rant. 2024. Making retrieval-augmented language
models robust to irrelevant context. InThe Twelfth
International Conference on Learning Representa-
tions, ICLR 2024, Vienna, Austria, May 7-11, 2024.
OpenReview.net.
A Prompt
A.1 Prompt for N-N Relations Recognition
In the intra-document logic modeling stage, we
first identify core sentences and their N–N (Nu-
cleus–Nucleus) logical relations. To accomplish
this, we design a prompt that instructs the LLM
to recognize five types of N–N relations between
sentences. The detailed prompt template is shown
in Figure 5.
A.2 Prompt for N-S Relations Recognition
After identifying core sentences, we cluster
non-core sentences and establish N-S (Nucleus-
Satellite) relations between them. We design a
prompt that instructs the LLM to recognize seven
types of N-S relations. The detailed prompt tem-
plate is shown in Figure 6.
A.3 Prompt for Cross-document Semantic
Bridging
To connect information across different documents,
we identify semantic relations between topic enti-
ties. We design a prompt that instructs the LLM to
extract entity relationships based on its background
knowledge. The detailed prompt template is shown
in Figure 7.
Prompt Template
Instruction:
You are an expert in rhetorical structure theory (RST) and 
discourse analysis. Your task is to analyze the logical and 
semantic relationship between sentences sharing the 
same topic entity, focusing on Nucleus-Nucleus (N-N) 
relations.
N-N Relation Types:
- Conjunction: Sentences A and B are connected and 
contribute equally to the overall meaning.
- Contrast: Sentences A and B present balanced 
opposing or contrasting information.
- Disjunction: Sentences A and B present alternative 
options or possibilities.
- Multinuclear Restatement: Sentences A and B restate 
the same content with equal emphasis.
- Sequence: Sentences A and B describe events in 
temporal or logical succession.
Input:
Topic Entity: {topic_entity}
Sentence List: {sentence_list}Figure 5: Prompt Template for N-N Relations Recogni-
tion.
Prompt Template
Instruction:
You are an expert in rhetorical structure theory (RST) and 
discourse analysis. Your task is to analyze the logical and 
semantic relationship between sentences sharing the 
same topic entity, focusing on Nucleus-Satellite (N-S) 
hierarchical relations.
N-S Relation Types:
- Cause: Sentence B describes a cause that leads to the 
situation stated in Sentence A.
- Result: Sentence B describes a result that follows from 
the situation stated in Sentence A.
- Opposition: Sentence B provides information that 
contradicts, limits, or challenges the situation or claim 
stated in Sentence A, such as expressing an opposing 
view, introducing a blocking condition, or conceding a 
contrary point.
- Elaboration: Sentence B adds further details, 
explanations, examples, justifications, motivations, 
methods, purposes, or other supportive information that 
helps the reader better understand or act upon the 
situation stated in Sentence A.
- Circumstance: Sentence B sets the background, 
context, time, place, or conditions under which Sentence 
A should be understood or interpreted.
- Evaluation: Sentence B gives an evaluative, interpretive, 
or summarizing statement about the content of Sentence 
A, such as expressing a value judgment, commentary, or 
concise restatement.
- Solutionhood: Sentence B proposes a solution to a 
problem, question, or challenge described in Sentence A.
Input:
Topic Entity: {topic_entity}
Sentence List: {sentence_list}
Figure 6: Prompt Template for N-S Relations Recogni-
tion.
11

Prompt Template
Instruction:
You are an expert in knowledge graph construction and 
entity relationship extraction. Your task is to analyze a 
given list of entities and identify meaningful known 
relationships between entity pairs.
Relation Extraction Guidelines:
- For each pair of entities, determine if there is a known 
relationship
- Use concise relation phrases (e.g., "directed", "acted in", 
"occurred in")
- Output triples in the format (entity1, relation, entity2)
- Do NOT output both directions for the same pair
- Skip pairs with no meaningful relationships
Input:
Entity List: {entity_list}Figure 7: Prompt Template for Cross-document Seman-
tic Bridging.
A.4 Prompt for Anchor Refinement
In the online reasoning stage, we refine the initially
retrieved anchor nodes. We design a prompt that
instructs the LLM to evaluate which evidence paths
are most relevant for answering the given query.
The detailed prompt template is shown in Figure 8.
Prompt Template
Instruction:
You are a multi-hop QA reasoning assistant. Given a 
question and candidate evidence sentences, your task is 
to select the most useful sentences for answering the 
question while retaining sufficient evidence for multi-hop 
reasoning.
Selection Guidelines:
- Keep paths unless they are clearly irrelevant.
- Output "action": "answer" if current evidence is sufficient 
to answer the question.
- Output "action": "expand" if more evidence is needed.
Input:
Question: {question}
Evidence Paths: {indexed_paths}
Figure 8: Prompt Template for Anchor Refinement.
12
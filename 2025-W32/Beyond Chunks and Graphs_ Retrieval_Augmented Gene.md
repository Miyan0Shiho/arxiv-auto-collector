# Beyond Chunks and Graphs: Retrieval-Augmented Generation through Triplet-Driven Thinking

**Authors**: Shengbo Gong, Xianfeng Tang, Carl Yang, Wei jin

**Published**: 2025-08-04 13:50:44

**PDF URL**: [http://arxiv.org/pdf/2508.02435v1](http://arxiv.org/pdf/2508.02435v1)

## Abstract
Retrieval-augmented generation (RAG) is critical for reducing hallucinations
and incorporating external knowledge into Large Language Models (LLMs).
However, advanced RAG systems face a trade-off between performance and
efficiency. Multi-round RAG approaches achieve strong reasoning but incur
excessive LLM calls and token costs, while Graph RAG methods suffer from
computationally expensive, error-prone graph construction and retrieval
redundancy. To address these challenges, we propose T$^2$RAG, a novel framework
that operates on a simple, graph-free knowledge base of atomic triplets.
T$^2$RAG leverages an LLM to decompose questions into searchable triplets with
placeholders, which it then iteratively resolves by retrieving evidence from
the triplet database. Empirical results show that T$^2$RAG significantly
outperforms state-of-the-art multi-round and Graph RAG methods, achieving an
average performance gain of up to 11\% across six datasets while reducing
retrieval costs by up to 45\%. Our code is available at
https://github.com/rockcor/T2RAG

## Full Text


<!-- PDF content starts -->

Beyond Chunks and Graphs: Retrieval-Augmented Generation through
Triplet-Driven Thinking
Shengbo Gong1, Xianfeng Tang2, Carl Yang1and Wei jin1
1Emory University,2Amazon
{shengbo.gong, j.carlyang, wei.jin}@emory.edu, xianft@amazon.com
Abstract
Retrieval-augmented generation (RAG) is criti-
cal for reducing hallucinations and incorporat-
ing external knowledge into Large Language
Models (LLMs). However, advanced RAG
systems face a trade-off between performance
and efficiency. Multi-round RAG approaches
achieve strong reasoning but incur excessive
LLM calls and token costs, while Graph RAG
methods suffer from computationally expen-
sive, error-prone graph construction and re-
trieval redundancy. To address these chal-
lenges, we propose T2RAG, a novel framework
that operates on a simple, graph-free knowl-
edge base of atomic triplets. T2RAG leverages
an LLM to decompose questions into search-
able triplets with placeholders, which it then
iteratively resolves by retrieving evidence from
the triplet database. Empirical results show
that T2RAG significantly outperforms state-of-
the-art multi-round and Graph RAG methods,
achieving an average performance gain of up to
11% across six datasets while reducing retrieval
costs by up to 45%. Our code is available at
https://github.com/rockcor/T2RAG .
1 Introduction
Large Language Models (LLMs) have become
central to open-domain question answering (QA)
systems, owing to their vast stores of parametric
knowledge and remarkable instruction-following
capabilities (Yue, 2025; Gu et al., 2024b). How-
ever, their effectiveness is often undermined by
critical challenges such as catastrophic forgetting
and hallucination, particularly when addressing
questions that require access to evolving, real-
world knowledge (Gu et al., 2024a; Huang et al.,
2025; Zhong et al., 2023). Consequently, Retrieval-
Augmented Generation (RAG) has emerged as a
robust paradigm to mitigate these issues (Lewis
et al., 2020; Gao et al., 2023) by retrieving relevant
documents from an external knowledge corpus.
However, standard RAG systems, which rank
document chunks by query similarity (Karpukhin
et al., 2020; Sawarkar et al., 2024; Khattab andZaharia, 2020), are effective for simple questions
but fail on complex ones that require multi-hop
reasoning (Tang and Yang, 2024). This failure
occurs because queries often lack the necessary
intermediate entities to connect information across
different chunks (Shen et al., 2024), and important
details can be lost in the compression loss of long
chunk embeddings (Zhang et al., 2024b).
To address these issues, two primary research
directions have emerged, each with its own chal-
lenges. Multi-Round RAG leverages the LLM’s
reasoning abilities by decomposing complex ques-
tions into sequential sub-queries. While effective
at traversing multi-hop knowledge paths, it is time
andtoken-consuming , often requiring numerous
(3-6) LLM calls in each round (Trivedi et al., 2023;
Xu et al., 2025; Shen et al., 2024), and up to around
8 rounds in total (Trivedi et al., 2023). Addition-
ally, it also faces the challenge of compression loss .
On the other hand, Graph RAG (Edge et al., 2024;
Han et al., 2024; Peng et al., 2024) structures the
corpus into a knowledge graph to retrieve logically
connected information. However, this approach is
hindered by an expensive and error-prone graph
construction process due to entity ambiguity is-
sue (Hoffart et al., 2014), redundancy in retrieval
from high-degree nodes (Peng et al., 2024), and
the difficulty LLMs face when understanding the
graph structures (Chai et al., 2023).
To circumvent these inherent inefficiencies
and architectural limitations of existing RAG
paradigms, we propose T2RAG ( Triplet-driven
Thinking for Retrieval- Augmented Generation), a
novel framework that fundamentally re-architects
the RAG pipeline and moves beyond traditional
chunk-based or graph-based retrieval by operat-
ing directly on atomic knowledge triplets. Un-
like Graph RAG , it completely sidesteps the costly,
time-consuming, and error-prone process of of-
fline knowledge graph construction. Instead of
building an explicit graph, T2RAG operates on
a graph-free knowledge base of atomic proposi-
tions, thus avoiding the high indexing costs andarXiv:2508.02435v1  [cs.IR]  4 Aug 2025

(a) Multi -round RAG
(b) Graph RAG
(c) Triplet -driven Thinking RAG (Ours)Question: Which film has the director born earlier, God’s Gift To Women or Aldri Annet Enn Brak ? 
I need to find the birth years of 
the directors of both films: 
Michael Curtiz for "God's Gift 
to Women " and Edith Carlmar
for "Aldri annet ennbråk....LLM Reasoning LLM Evaluation
NoIf the information is 
enough to answer?Yes + 
Answer
Director, Birthday, Born, 
God’s Gift To Women, 
Aldri Annet Enn BrakQuery Entity/Relation 
extractionAldri annet enn 
bråk\nAldri annet 
enn bråk is a 
1954 film...Corpus Entity/Relation extraction KG Construction
God’s Gift To Women ,is directed by, ?DirectorA
Aldri Annet Enn Brak ,is directed by, ?DirectorB
?DirectorA , was born in, ? birthYearA
?DirectorB , was born in, ? birthYearBLLM ReasoningAldri annet enn 
bråk\nAldri annet 
enn bråk is a 
1954 film...Corpus Entity/Relation extraction
Chunk 
Vector
DB
Triplet 
Vector 
DBPageRank/Shortest 
path/neighbor expansion…
Target subgraphAnswer
Chunk
Vector
DBCombine raw chunks
LLM ResolvingMichael Curtiz , was born in, ? birthYearA
Edith Carlmar , was born in, ? birthYearB…Answer
Stop until all 
triplets solved
…
Entity
Vector 
DB
Relation
Vector 
DB
Retrieve from DB
Control flowEntity AmbiguityCompression Loss
Token Consuming
RedundancyFigure 1: A comparison of three RAG paradigms, with their primary challenges highlighted in red. (a) Multi-round
RAG employs an iterative loop to retrieve large text chunks, but is hampered by compression loss from vector
embeddings and high token consumption during reasoning. (b) Graph RAG constructs a knowledge graph to
retrieve answers, but is vulnerable to entity ambiguity during creation and retrieval redundancy from high-degree
nodes. (c) T2RAG decomposes a query into triplets with “?” placeholders and iteratively resolves them by
retrieving evidence from a triplet database (DB) until all of them are resolved.
potential for retrieval errors caused by inaccurate
graph links. Simultaneously , it tackles the exces-
sive token consumption and latency that plagues
Multi-round RAG systems. Rather than generat-
ing verbose, natural language reasoning chains at
each step, T2RAG leverages the LLM to think in a
more structured, efficient manner. It expands com-
plex questions into “searchable triplets” containing
specific placeholders for unknown entities. The
system then iteratively retrieves context to resolve
these triplets. This design maintains a lean, struc-
tured state transition between iterations, passing
only compact triplets instead of verbose text. This
triplet-centric design ensures a tight coupling be-
tween retrieval and reasoning, retaining powerful
multi-hop capabilities while dramatically reducing
token overhead and enhancing performance. Our
main contributions are as follows:
•We introduce a novel RAG framework that lever-
ages triplets as the fundamental unit for indexing,
retrieval, and reasoning, moving beyond the lim-
itations of chunk-based and explicit graph-based
approaches.
•We demonstrate that our method achieves state-
of-the-art performance on various types of QA
benchmarks, outperforming leading models in
both the Multi-Round RAG and Graph RAG.
•We also significantly improve the efficiency. Ourmethod reduces inference time and token con-
sumption by up to 45% compared to other multi-
round methods and even achieves an efficiency
comparable to that of single-round approaches.
2 Preliminaries
The task of open-domain question answering
(ODQA) was formally introduced in the 1999 Text
REtrieval Conference (TREC) QA track (V oorhees
and Tice, 2000). Initially, it was defined as a fac-
toid QA task: Given a large corpus of unstruc-
tured documents, the goal was to extract a small
text snippet containing the correct answer to a
factual question. While the scope of ODQA has
since expanded to include summarization and open-
ended (Reja et al., 2003) tasks (Edge et al., 2024;
Xiao et al., 2025), factoid QA remains a signif-
icant challenge, evidenced by poor performance
(below 50%) on complex, multi-hop datasets like
MusiQue (Trivedi et al., 2023). Consequently, this
paper focuses on advancing the state-of-the-art in
factoid QA.
Factoid QA Task. Assume our collection contains
Ddocuments d1, d2, . . . , d D. We split each docu-
ment into passages of equal token length or apply-
ing expert split if it exists, yielding Mtotal chunks
C={c1, c2, . . . , c M}, where each chunk cican be
viewed as a token sequence (w(i)
1, w(i)
2, . . . , w(i)
|ci|).

Given a question q, the goal is to find a combi-
nation of tokens (w(j)
cm, . . . , w(j)
cm+k)drawn from
multiple chunks that collectively contain the infor-
mation necessary to answer qwhile minimizing
irrelevant noise to avoid hallucination. The answer
must be exact one entity in our setting, such as
persons, organizations, or locations or yes/no. Typ-
ically, a retriever R: (q,C)→ C Fis a function
that takes a question qand the corpus Cas input
and returns a much smaller set of chunks CF⊂ C,
where |CF|=k≪ |C| . For a fixed k, a retriever
can be evaluated in isolation using top- kretrieval
accuracy with respect to labeled golden chunks.
Retrieval Granularity. The preceding formula-
tion assumes the retrieval unit is the chunk, which
is a common setting (Karpukhin et al., 2020). How-
ever, recent works especially Guo et al. (2024);
Fan et al. (2025) argue that chunks often contain a
mix of relevant and irrelevant details, and a finer
granularity is needed for complex queries (Zhang
et al., 2024b). Inspired by work in Knowledge
Graphs (KGs) (Ji et al., 2021), the fundamental
unit of retrieval can be refined to more atomic ele-
ments:
•Entities (e(i)
1, e(i)
2, . . . , e(i)
|ci|): Named entities
such as persons, organizations, or locations.
•Triplets (t(i)
1, t(i)
2, . . . , t(i)
|ci|): Structured facts
represented as a (subject,predicate,object) tuple.
•Propositions (p(i)
1, p(i)
2, . . . , p(i)
|ci|): Atomic state-
ments or facts, often by converting triplets into
natural language sentences.
Propositions, which encapsulate a complete fact in
a single sentence, are often considered to have
greater semantic utility for modern embedding
models compared to isolated entities or structured
triplets (Zhang et al., 2024b). Our work explores
leveraging this fine-grained units for improved re-
trieval and reasoning.
3 Related Work
We group recent RAG efforts into multi-round , and
graph-enhanced RAG, each adding more interac-
tion or structured reasoning and paving the way for
the fine-grained design of T2RAG.
Multi-round RAG. Due to missing intermediate
entities problem we mentioned in Section 1 more
and more works follow a multi-round paradigm,
which enables the LLMs infer the intermediate
information thus better retrieve the final answer.
Some works focus on the query side. Khot
et al. (2023) decompose multi-hop questions into
single-hop sub-queries that are solved sequentially.
Yao et al. (2023) propose ReAct, interleavingchain-of-thought (CoT) (Wei et al., 2022) steps
with search actions issued by the LLM. Similarly,
Query2Doc (Wang et al., 2023b) expanding queries
into concise triplets to cut token usage while pre-
serving recall. Another line of works relies on
the generated intermediate results for next itera-
tion. Beam Retrieval (Zhang et al., 2024a) jointly
training an encoder and classifiers to keep multiple
passage hypotheses across hops. FLARE (Jiang
et al., 2023) forecasts upcoming sentences to de-
cide when fresh retrieval is needed during long-
form generation. IRCoT (Trivedi et al., 2023) and
ITER-RETGEN (Shao et al., 2023), alternately
expanding a CoT and fetching new evidence to an-
swer multi-step questions. Adaptive QA (Xie et al.,
2023) create an adaptive framework that picks the
simplest effective retrieval strategy according to
query complexity. Despite these advances, few
efforts explicitly aim to reduce token costs or num-
ber of llm calls during multi-round RAG. Previous
methods expand query or generates CoT with long
sentences in each round. In contrast, our work
minimizes token consumption by formulating query
expansions as triplets and simplifying reasoning
steps as triplets resolving.
Graph RAG. One major line of research addresses
complex QA by structuring knowledge into graphs.
Originating in Knowledge Graph QA (KGQA),
early methods focused on decomposing queries
or performing multi-round, LLM-evaluated traver-
sals from seed nodes (Luo et al., 2024; Sun et al.,
2024; Cheng et al., 2024; Mavromatis and Karypis,
2022). The application of this paradigm to gen-
eral ODQA was popularized by systems named
GraphRAG (Edge et al., 2024) that construct a
knowledge graph entirely with LLMs and use com-
munity detection for retrieval. Subsequent work
has aimed to make this process more efficient. For
instance, LightRAG (Guo et al., 2024) introduces a
dual-level retrieval system combining graph struc-
tures with vector search to improve knowledge dis-
covery. Targeting resource-constrained scenarios,
MiniRAG (Fan et al., 2025) builds a heterogeneous
graph of text chunks and named entities, enabling
lightweight retrieval suitable for Small Language
Models. To tackle the common challenge of en-
tity merging, HippoRAG (Gutiérrez et al., 2025a)
and HippoRAG2 (Gutiérrez et al., 2025b) create
synonym links between similar entity nodes and
employs a PageRank (Haveliwala, 1999) algorithm
for final node selection. Despite these advances,
a central challenge for Graph RAG remains the
costly and error-prone nature of graph construc-

tion from unstructured text.
Our method, T2RAG, skips the costly and error-
prone graph construction required by Graph RAG
while retains the multi-hop reasoning power by
Multi-round RAG. It also dramatically reduces to-
ken overhead by constraining both query expansion
and intermediate generation. Besides, some works
in ODQA such as GEAR (Shen et al., 2024) also
employ a triplet search component. These methods
typically rely on neighbor expansion, which in-
volves retrieving all other triplets that share a head
or tail entity. A key drawback of this approach is
that accurately identifying and linking the same
entity across different contexts is often inaccurate
and computationally expensive.
4 Methodology
4.1 Overview
Our proposed method, T2RAG (Triplet-driven
Thinking RAG ), is a novel paradigm for resolving
complex, multi-hop, factoid QA tasks. Unlike con-
ventional RAG systems that operate on coarser
document chunks or complex graph structures,
T2RAG is designed to operate directly on atomic
knowledge propositions derived from triplets, fos-
tering an intrinsic alignment between knowledge
representation and LLM reasoning. This frame-
work operates in two stages: an offline indexing fo-
cused on systematic knowledge distillation, and an
online retrieval characterized by iterative, adaptive
triplet resolution. This principled design ensures
both fine-grained retrieval for accuracy and a lean,
efficient reasoning process.
4.2 Offline Indexing: Constructing a
Graph-Free Knowledge Base
The goal of the offline stage is to transform a raw
text corpus Cinto a efficiently searchable knowl-
edge base of atomic propositions. The motivation
for adopting proposition level granularity is two
fold: 1) Compared to the entity level, each propo-
sition encodes an entire, unambiguous fact. 2)
Compared to the chunk level, it also avoids the
compression loss hindering the retrieval of details.
Canonical Triplet Generation. For each doc-
ument chunk ci∈ C, we employ an informa-
tion extraction model, LLM IE(·), to identify
key facts. This model performs Open Informa-
tion Extraction (OpenIE) (Martinez-Rodriguez
et al., 2018) to extract a set of knowledge
triplets Ti={t(i)
1, t(i)
2, . . .}. Each triplet t(i)
j
is formalized as a canonical knowledge triplet
(subject, predicate, object )that represents a sin-gle factual statement. All extracted triplets are then
aggregated into a global set for the entire corpus
Ttotal=SM
i=1Ti, where Mis the total number of
extracted triplets.
Triplet Embedding. To render these canonical
triplets semantically actionable for dense retrieval,
we are inspired by verbalization techniques (Oguz
et al., 2020; Baek et al., 2023) to convert each
triplet t∈ Ttotalinto a natural language sentence,
termed a proposition p, simply by concatenating its
components (e.g., “subject predicate object”). This
seemingly straightforward verbalization is a de-
liberate design choice: it maximizes the semantic
utility for embedding models, facilitating effective
and contextually rich retrieval compared to isolated
entities.
Triplet Vector DB Construction. The resulting
flat list of propositions Ptotal={p1, p2, . . . , p M}
is then encoded into dense vector representations
using a high-performance embedding model E(·).
For efficient real-time access, these vectors can be
subsequently indexed using a highly optimized vec-
tor search library (FAISS) (Douze et al., 2024), cre-
ating an index Ithat enables rapid similarity search
across all propositions in the corpus. This vector
DB is still called Triplet Vector DB as it keeps
original text of triplets. We also save the mapping
from those propositions to their source chunks be-
cause the original text is proved necessary in most
of Graph RAG works (Guo et al., 2024; Fan et al.,
2025). This pre-computation creates a fine-grained,
semantically enriched knowledge index without
the overhead of explicit graph structures .
The constructed proposition index, while offer-
ing significant advantages in terms of cost and
construction fidelity, introduces a critical chal-
lenge: how to effectively navigate complex, multi-
hop questions that typically rely on graph traver-
sals? In the subsequent subsection, we introduce
our novel online retrieval stage, where the LLM’s
triplet-driven thinking and adaptive iterative resolu-
tion strategically compensate for the graph traver-
sals and the path-based reasoning.
4.3 Online Retrieval: Iterative Triplets
Resolution
The online retrieval stage is an iterative process that
dynamically builds the context containing both the
triplets and chunks needed to answer user queries.
The overall retrieval process is shown in Figure 2.
Step 1: Structured Query Decomposition. Given
an initial query q, we first use an LLM to perform a
structured decomposition where the LLM identifies
the specific, atomic knowledge Triplets (denoted

Question
Resolved Triplets 
(with no“?”)Step 1: Structured Query
Decomposition
Step 2: Multi -round 
Triplet Resolution 
with Triplet RetrievalSearchable Triplets 
(with one“?”)Fuzzy Triplets
(with two or more “?”)
Total Resolved Triplets AnswerStep 3: Final Answering
Triplet 
Vector 
DB
Question+Searchable TripletsResolve
Resolved Triplets ORResolved Triplets Resolve
Triplet 
Vector 
DBFigure 2: Online retrieval stage of T2RAG.
asTq) that must be answered to address the overall
query. Critically, these derived triplets contain
explicit placeholders (‘?’) for unknown entities.
Based on the precise number of these placeholders,
we categorize these initial triplets into three types:
•Resolved Triplets (Tresolved ): Triplets with zero
placeholders, representing fully known facts that
require no further search.
•Searchable Triplets (Tsearchable ): Triplets with
exactly oneplaceholder. This specificity, with
two known elements, facilitates focused and ac-
curate searches.
•Fuzzy Triplets (Tfuzzy): Triplets with two or
more placeholders. These are inherently too am-
biguous for search with the at most one element.
It requires resolution in subsequent iterations to
upgrade to searchable orresolved .
This explicit categorization ensures that later re-
trieval efforts are always focused and efficient.
Step 2: Multi-Round Triplet Resolution with
Triplet Retrieval. In this step, we will resolve the
query triplets, i.e., try to eliminate all "?" place-
holders step by step by RAG. Considering different
complexity of queries and their triplets, we adopt
an adaptive retrieval strategy instead of a fixed top-
k. We also observed most of multi-hop questions
cannot be specifically retrieved by the query itself
as illustrated in Figure 1, which necessitate the
multi-round paradigm.
Step 2.1: Triplet-Based Adaptive Retrieval.
The current set of searchable triplets T(l)
searchableare
first converted into query propositions by simply
concatenating the elements without the placeholder.
These propositions are then embedded, using the
same embedding model E(·)in the indexing stage,
and used to query the proposition index I. Unlike
prior methods that retrieve a fixed top- kof propo-
sitions or triplets (Baek et al., 2023; Guo et al.,
2024), our retrieval process is critically adaptive
in two synergistic ways to ensure both relevanceand informational diversity: First , our method re-
trieves with the triplets while constrain the process
by chunks. More specifically, the retrieval dynami-
cally continues until context from kunique source
chunks of triplets has been retrieved. Second , we
aggregate retrieval candidates from all query propo-
sitions into a unified pool, ranking them globally
by similarity scores, rather than allocating separate
budgets to each proposition. These adaptive strate-
gies ensure robustness to varying query complexity,
allowing difficult questions to naturally draw from
a wider range of propositions. Finally, the retrieval
process returns the set of retrieved propositions
P(l)
retrievedand their corresponding source chunks
C(l)
retrieved. The necessity of reading original chunks
to complete details missing from triplets is widely
acknowledged in the field (Fan et al., 2025; Guo
et al., 2024).
Step 2.2: Resolving Triplets with Retrieved
Context. This step leverages the retrieved con-
tent to advance the query’s resolution. We prompt
the LLM to populate the placeholders within
these triplets using the provided context. The
retrieved propositions (P(l)
retrieved)and and their
source chunks (C(l)
retrieved)serve as context for an
LLM call. This is designed to either upgrade a
searchable triplet to a fully resolved one by filling
in its single placeholder, or to transform a fuzzy
triplet into a searchable or directly to a resolved one
by filling in one or more of its multiple placehold-
ers. This resolution process reduces the ambiguity
of existing triplets and makes it suitable for subse-
quent targeted retrieval. The process is shown in
Figure 2 and a detailed example is in Appendix D.
Step 2.3: State Update and Ending Condi-
tion. Following the triplet resolution step, the
system’s state is updated for the next iteration,
l+ 1. The set of resolved triplets is monotoni-
cally augmented with any newly resolved ones:
T(l+1)
resolved=T(l)
resolved∪ T(new)
resolved. Crucially, only the
newly searchable triplets are used for the subse-
quent retrieval step: T(l+1)
searchable=T(new)
searchable. Any
fuzzy triplets that remain unsolved are carried over
to the next round’s prompt. This set is updated
by removing any triplets that were just resolved or
became searchable: T(l+1)
fuzzy=T(l)
fuzzy\(T(new)
resolved∪
T(new)
searchable). At the end of each iteration, similar
to IRCoT (Trivedi et al., 2023), we check for an
early stopping condition. Instead of using an LLM
call, our method simply terminates if there are
no unresolved triplets left. Formally, the iteration
continues as long as there are any searchable or

fuzzy triplets remaining or maximum iterations N
reaches: |T(l+1)
searchable∪ T(l+1)
fuzzy|>0. This highly
structured state transition is key to our method’s ef-
ficiency. By passing compact triplets between iter-
ations, rather than the verbose CoT reasoning used
by approaches like IRCoT, we dramatically reduce
token overhead. Furthermore, this triplet-centric
design creates a powerful synergy: the LLM gener-
ates reasoning gaps in the same format,i.e., triplets,
ensuring strong semantic alignment between the
resolution and retrieval stages.
Step 3: Synthesizing the Final Answer. Once
the iterative loop terminates after Krounds, all
fully resolved triplets are aggregated into a final
set,Ttotal_solved =T(K)
resolved. A final LLM call is
then made to generate the answer, conditioned on
how the process ended:
(a) Successful Resolution : If the loop terminated
because all triplets were resolved, the LLM is
prompted with the original query ( q) and this pre-
cise set of structured knowledge to generate a con-
cise answer a:a=LLM Answer (q,Ttotal_solved ).
(b) Maximum Iterations Reached : If the loop
stopped because it reached the maximum number
of iterations, any remaining searchable triplets are
included with the resolved facts to form the best
possible context: a=LLM Answer (q,Ttotal_solved ∪
T(K)
searchable). By providing the LLM primarily with
the verified facts in Ttotal_solved instead of raw re-
trieved chunks, this method minimizes token costs
and reduces the risk of hallucination.
5 Experiments
5.1 Datasets
To ensure a comprehensive evaluation, we select
representative datasets for three distinct Open-
Domain Question Answering (ODQA) categories:
Simple QA, Multi-hop QA, and Domain-specific
QA. For the first two categories, we follow the
experimental setup from HippoRAG2 (Gutiérrez
et al., 2025b). We use PopQA (Mallen et al.,
2023) for simple questions. For multi-hop ques-
tions, we use 2Wiki-MultihopQA (2Wiki) (Ho
et al., 2020), MuSiQue (Trivedi et al., 2022), and
HotpotQA (Yang et al., 2018). For each of these
datasets, we use the same sample of 1,000 ques-
tions as the prior work. For domain-specific evalu-
ation, we adapt two datasets from the GraphRAG-
Bench (Xiao et al., 2025). We isolate the factoid
questions from the two datasets, Story and Medi-
cal, and use an LLM to shorten the ground-truth an-
swers, enabling more precise evaluation. Detailed
statistics for all datasets are provided in Table 3.5.2 Baselines and Implementation Details
To evaluate our approach, we select three strong
baselines representing state-of-the-art methods
across major RAG categories. For Graph RAG,
we choose HippoRAG2 (Gutiérrez et al., 2025b)
for its recognized efficiency and effectiveness. For
summarization-based RAG, we use Raptor (Sarthi
et al., 2024), a pioneering method that outperforms
most Graph RAG approaches in recent bench-
marks (Zhou et al., 2025). Lastly, for Multi-Round
RAG, we include the prominent IRCoT (Trivedi
et al., 2023) method. NOR method means the non-
retrieval method that directly answers the question.
Standard RAG retrieves chunks with an embed-
ding model and uses them to generate an answer.
To ensure a fair comparison, all methods are con-
figured with the same foundational models: NV-
Embed-v2 (Lee et al., 2024) for embeddings and ei-
ther Gemini-2.5-flash or GPT-4o-mini as the LLM
for all offline indexing and online retrieval stages.
For datasets lacking expert annotations, we employ
a standard chunking strategy of 1200 tokens with a
100-token overlap. For the top- kof chunk retrieval,
we set k= 5for all methods. For the multi-round
methods (T2RAG and IRCoT), we set a maximum
ofN= 3iterations and keeps the k= 5in each
iteration. Following standard practices (Trivedi
et al., 2023), we evaluate end-to-end QA perfor-
mance using Exact Match (EM) and F1 scores. We
focus specifically on these end-to-end QA metrics,
as retrieval performance is difficult to compare di-
rectly when the number of retrieved passages is
adaptive. Except for the performance comparisons,
all results presented in the subsequent sections are
obtained using GPT-4o-mini. Further experimental
details are available in Appendix B.
5.3 Results
We unfold our analysis of experimental results by
answering Research Questions (RQ) below.
RQ1: How does T2RAG perform against base-
lines? As shown Table 1, T2RAG achieves state-
of-the-art performance, stems from several key ad-
vantages. First , our method achieves state-of-the-
art overall performance, leading in both average
EM and F1 scores across the two LLM backbones,
except for the second place in F1 by GPT-4o-mini.
Notably, its advantage in EM is particularly pro-
nounced, a strength we attribute to the precision of
our triplet-based retrieval, which excels at identify-
ing the exact entities required for factoid QA. This
adaptability is further demonstrated by its consis-
tently strong results on domain-specific datasets,

Table 1: Main performance comparison on various types of QA datasets, showing Exact Match / F1 scores ×100.
The best result in each column is in bold , and the second best is underlined .
Simple QA Multi-Hop QA Domain-Specific QA Average
Method PopQA 2Wiki MuSiQue HotpotQA Story Medical EM F1
Gemini-2.5-flash
NOR 32.4 / 35.7 48.1 / 55.6 16.3 / 26.5 40.5 / 52.3 10.3 / 17.1 23.1 / 46.0 28.4 38.9
BM25 50.2 / 55.6 28.2 / 30.7 7.9 / 10.7 40.8 / 49.3 26.2 / 35.3 22.2 / 37.8 29.3 36.6
Standard 51.8 / 59.5 33.1 / 39.0 28.1 / 36.2 52.1 / 63.1 31.0 / 42.2 19.4 / 41.5 35.9 46.9
HippoRAG2 52.1 / 60.1 44.3 / 51.2 29.1 / 38.3 52.1 / 64.1 33.1 / 44.1 27.8 / 58.2 39.8 52.7
RAPTOR 52.3 / 56.8 36.3 / 41.1 31.8 / 39.7 60.9 / 72.7 46.2 / 59.0 34.2 / 58.1 43.6 54.6
IRCoT 51.2 / 58.7 61.6 / 71.7 39.7 / 49.8 61.2 /77.3 40.3 / 57.3 26.1 / 56.1 46.7 61.8
T2RAG 56.6 /62.4 69.3 /77.5 39.1 /49.1 62.3 / 73.2 46.7 /59.5 36.0 /61.4 51.7 63.9
GPT-4o-mini
NOR 28.7 / 31.4 28.0 / 34.1 10.2 / 20.3 28.8 / 38.6 11.5 / 18.9 19.3 / 44.2 21.1 31.3
BM25 47.6 / 54.8 42.9 / 48.2 15.3 / 21.1 47.2 / 57.6 29.0 / 38.5 25.9 / 43.6 34.7 44.0
Standard 51.9 / 60.0 53.1 / 60.2 31.2 / 44.3 58.0 / 71.1 27.3 / 60.1 27.0 / 59.9 41.4 59.3
HippoRAG2 52.2 / 60.2 59.6 / 69.3 34.1 / 48.1 58.1 / 71.1 41.2 / 58.3 28.1 / 59.4 45.6 61.1
RAPTOR 54.6 / 60.1 38.2 / 49.0 28.6 / 40.8 57.9 / 71.4 44.8 / 59.6 36.7 /63.7 43.5 57.4
IRCoT 45.3 / 54.7 60.7 /74.3 34.1 / 47.6 55.7 / 71.2 36.1 / 51.8 25.1 / 52.9 42.8 58.8
T2RAG 55.8 /63.2 66.7 / 74.4 34.3 / 45.6 54.2 / 67.3 38.7 / 50.1 33.5 / 60.4 47.2 60.2
underscoring the universality of the underlying rea-
soning framework. Second , its superiority is most
pronounced on Multi-hop QA datasets like 2Wiki.
It not only surpasses all single-round baselines
by a large margin but also outperforms the multi-
round baseline, IRCoT, by over 7.7% and 5.4%
in EM with Gemini-2.5-flash and GPT-4o-mini,
respectively. This highlights the effectiveness of
its triplet-driven mechanism for complex reason-
ing.Finally , the method demonstrates a powerful
synergy with reasoning LLMs. Its performance is
significantly higher when paired with Gemini-2.5-
flash compared to GPT-4o-mini. This suggests that
its structured process of query decomposition and
resolution can uniquely leverage the advanced rea-
soning capabilities of such models through its step-
by-step guidance. Conversely, certain methods
such as HippoRAG2 exhibit a decrease in perfor-
mance when employing reasoning LLMs. We hy-
pothesize this occurs because relegating the LLM
to a simple filtering task does not fully harness its
sophisticated reasoning capabilities.
RQ2: What is the impact of the triplet resolu-
tion module? To validate the effectiveness of our
core "triplet-driven thinking" design, we analyze
the final performance based on whether a query’s
underlying triplets are fully resolved. Figure 3 re-
veals a significant performance delta between these
two outcomes. Across all three datasets, there is a
strong correlation between successful triplet reso-
lution and high performance. For instance, on the
2Wiki dataset, the F1 score for unresolved ques-
tions drops to 53% from 76%, with a similar sharp
decline observed in EM scores. This result con-
firms that resolving all triplets is the key to success.
PopQA 2Wiki MuSiQue0.00.10.20.30.40.50.60.7Performance0.560.69
0.350.620.76
0.47
0.41 0.42
0.200.450.53
0.32EM (Resolved)
F1 (Resolved)
EM (Unresolved)
F1 (Unresolved)Figure 3: Performance vs. final resolution status.
Table 2: Ablation results
PopQA 2Wiki MuSiQue
Method EM F1 EM F1 EM F1
T2RAG 56.0 63.0 66.0 74.0 33.0 45.0
- single round54.8 60.5 51.0 59.0 15.0 24.0
↓2.1% ↓4.0% ↓22.7% ↓20.3% ↓54.5% ↓46.7%
- w/o chunk41.1 44.7 62.0 68.0 21.6 29.9
↓26.6% ↓29.0% ↓6.1% ↓8.1% ↓34.5% ↓33.6%
RQ3: Which components of T2RAG are impor-
tant? We conducted an ablation study to quantify
the contribution of its two key components. The re-
sults in Table 2 reveal that both the iterative process
and the use of chunks are important. The iterative
reasoning module proves to be a critical compo-
nent. Removing it (- single round) causes a sig-
nificant performance degradation, particularly on
multi-hop QA. For instance, F1 score on MuSiQue
drops by a remarkable 54.5%. This demonstrates
that the multi-round retrieval and resolution is es-
sential for decomposing and solving complex prob-
lems. Similarly, removing the raw chunk text dur-
ing the iteration, i.e, (- w/o chunk), is also substan-
tially harms performance, confirming that the raw
text complement missing details of triplets. This
observation is aligned with Fan et al. (2025).

T2RAG
HippoRAG2RAPTORIRCoT
LightRAG*GraphRAG*0123Token Consumption×108 HotpotQA
T2RAG
HippoRAG2RAPTORIRCoT0123×107 Medical
T2RAG
HippoRAG2RAPTORIRCoT
LightRAG*GraphRAG*0.000.250.500.751.001.25Time (s)×105
T2RAG
HippoRAG2RAPTORIRCoT02468×103
Indexing Stage Retrieval StageFigure 4: Comparison of token consumption and time.
Token consumption is calculated by (input + 4 ×output).
Results of LightRAG and GraphRAG are from a bench-
mark (Zhou et al., 2025).
RQ4: How does T2RAG compare in terms of
computational efficiency? This analysis com-
pares the computational cost of T2RAG with base-
lines during both the one-time offline indexing and
online retrieval phases. To better visualize the on-
line costs, the token and time values for the re-
trieval stage in Figure 4 are aggregated over 1,000
queries, assuming they are processed sequentially.
Figure 4 illustrates a strategic trade-off. During
indexing stage , T2RAG’s token consumption ap-
pears high because it processes the entire corpus
into triplets. However, this processing is merely
the first step for many advanced Graph RAG meth-
ods (Edge et al., 2024; Guo et al., 2024; Fan et al.,
2025).methods (Edge et al., 2024; Guo et al., 2024;
Fan et al., 2025). Their subsequent graph construc-
tion steps are far more costly. For example, Ligh-
tRAG and GraphRAG require around 6 ×and 10×
the token consumption of the initial triplet extrac-
tion phase, respectively (Gutiérrez et al., 2025b).
T2RAG’s indexing overhead remains highly com-
petitive within this category. At the retrieval stage ,
T2RAG is remarkably more efficient in both to-
kens and latency than the multi-round baseline,
IRCoT. More notably, its efficiency is even com-
parable to single-round methods. This is because
HippoRAG2 also invokes multiple LLM calls for
filtering, while Raptor retrieves longer summaries
than chunks. T2RAG’s efficiency stems from its de-
sign, which focuses on targeted search for triplets
rather than processing large, noisy text chunks. In
summary, T2RAG accepts a standard indexing cost
to deliver a highly efficient online system.
RQ5: How does performance scale with the
amount of retrieved context? To investigate
how T2RAG’s performance scales with context
2 3 4 5 6 7 8 9 10
Effective Top-k0.500.550.600.650.700.75Averaged EM & F1
Standard
HippoRAG2
T2RAG
IRCoTFigure 5: Performance vs. top- k. Multi-round methods
are calibrated by k×average number of iterations.
size, we compare it against other multi-round meth-
ods while varying the number of retrieved docu-
ments (top- k). Traditional RAG methods often
rely on retrieving more context to find the correct
answer, which can be inefficient. The trend in
Figure 5 shows T2RAG’s performance is consis-
tently high and robust to the value of top- k. It
achieves the plateau faster than other methods. In
contrast, baselines like IRCoT and HippoRAG2
exhibit a strong dependence on a larger context
window. This observation demonstrates its effec-
tiveness does not rely on scaling up the volume
of retrieved text but a more precise and specific
triplet-based retrieval.
6 Conclusion
In this work, we proposed the Triplet-driven Think-
ing RAG (T2RAG), a novel framework that em-
beds reasoning directly into the retrieval process.
By decomposing complex queries into atomic
triplets and resolving them step-by-step against
a triplet knowledge base, our method consistently
outperforms more complexly designed RAG sys-
tems. Our extensive experiments demonstrate that
T2RAG establishes a new state-of-the-art in factoid
QA tasks, particularly on challenging multi-hop
QA. This superior performance is achieved with
remarkable online efficiency; the retrieval stage
has significantly lower time and token consump-
tion compared to other multi-round methods and
maintains a comparable overhead to even single-
round approaches. Furthermore, our results reveal
a powerful synergy between T2RAG’s structured
thinking process and the capabilities of advanced
reasoning LLMs, highlighting a new path to un-
lock their full potential in this area. Looking for-
ward, T2RAG paves the way for more accurate and
efficient RAG systems by shifting the paradigm
from retrieving and generating unstructured con-
texts towards a more deliberate, reasoning-driven
synthesis of atomic facts.

7 Limitations
Although our method achieves state-of-the-art per-
formance with a simple design, it is not without
limitations. Experimentally , we limited our multi-
round methods to 3 iterations to match the com-
plexity of the datasets and ensure a fair efficiency
comparison; we also did not have the resources to
test on other embedding models especially LLM-
based ones, re-rankers or large external knowledge
graphs (e.g., Wikipedia KG (Hertling and Paul-
heim, 2018)). Our evaluation is also limited to
the black-box and end-to-end one which may lack
explanability without the recall score of chunks.
Methodologically , our approach is highly depen-
dent on the quality of the triplet extraction. While
higher-quality sources can be used, simple triplets
may not adequately represent complex knowledge
like many-to-many relationships, a challenge that
could be addressed with hypergraph modeling (Luo
et al., 2025) in future work. Besides, the efficiency
of triplet extraction can be further improved be-
yond the classic OpenIE pipeline. Developing
these methods needs efforts from information ex-
traction (Grishman, 2015) area. Finally , regarding
scalability, building the index from a very large
corpus is token-intensive. However, our method
is very efficient when using a pre-existing triplet
database. This design also makes it inherently suit-
able for evolving knowledge bases, as new triplets
are independent to previous ones thus they can be
added incrementally, offering a significant advan-
tage over static Graph RAG approaches (Zhang
et al., 2025).
References
Jinheon Baek, Alham Fikri Aji, Jens Lehmann, and
Sung Ju Hwang. 2023. Direct Fact Retrieval
from Knowledge Graphs without Entity Linking.
ArXiv:2305.12416 [cs].
Ziwei Chai, Tianjie Zhang, Liang Wu, Kaiqiao Han,
Xiaohai Hu, Xuanwen Huang, and Yang Yang. 2023.
Graphllm: Boosting graph reasoning ability of large
language model. arXiv preprint arXiv:2310.05845 .
Sitao Cheng, Ziyuan Zhuang, Yong Xu, Fangkai Yang,
Chaoyun Zhang, Xiaoting Qin, Xiang Huang, Ling
Chen, Qingwei Lin, Dongmei Zhang, Saravan Ra-
jmohan, and Qi Zhang. 2024. Call me when nec-
essary: LLMs can efficiently and faithfully reason
over structured environments. In Findings of the As-
sociation for Computational Linguistics: ACL 2024 ,
pages 4275–4295, Bangkok, Thailand. Association
for Computational Linguistics.
Matthijs Douze, Alexandr Guzhva, Chengqi Deng,
Jeff Johnson, Gergely Szilvasy, Pierre-EmmanuelMazaré, Maria Lomeli, Lucas Hosseini, and Hervé
Jégou. 2024. The faiss library. arXiv preprint
arXiv:2401.08281 .
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024. From Local to Global:
A Graph RAG Approach to Query-Focused Summa-
rization. ArXiv:2404.16130.
Tianyu Fan, Jingyuan Wang, Xubin Ren, and Chao
Huang. 2025. MiniRAG: Towards Extremely Simple
Retrieval-Augmented Generation.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang
Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun,
Haofen Wang, and Haofen Wang. 2023. Retrieval-
augmented generation for large language models: A
survey. arXiv preprint arXiv:2312.10997 , 2(1).
Ralph Grishman. 2015. Information extraction. IEEE
Intelligent Systems , 30(5):8–15.
Jia-Chen Gu, Hao-Xiang Xu, Jun-Yu Ma, Pan Lu, Zhen-
Hua Ling, Kai-Wei Chang, and Nanyun Peng. 2024a.
Model editing harms general abilities of large lan-
guage models: Regularization to the rescue. In Pro-
ceedings of the 2024 Conference on Empirical Meth-
ods in Natural Language Processing , pages 16801–
16819, Miami, Florida, USA. Association for Com-
putational Linguistics.
Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan,
Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan
Shen, Shengjie Ma, Honghao Liu, and 1 others.
2024b. A survey on llm-as-a-judge. arXiv preprint
arXiv:2411.15594 .
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2024. LightRAG: Simple and Fast Retrieval-
Augmented Generation. ArXiv:2410.05779.
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2025a. HippoRAG:
Neurobiologically Inspired Long-Term Memory for
Large Language Models.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025b. From RAG to Mem-
ory: Non-Parametric Continual Learning for Large
Language Models. ArXiv:2502.14802 [cs].
Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan
Ding, Yongjia Lei, Mahantesh Halappanavar, Ryan A
Rossi, Subhabrata Mukherjee, Xianfeng Tang, and 1
others. 2024. Retrieval-augmented generation with
graphs (graphrag). arXiv preprint arXiv:2501.00309 .
Taher Haveliwala. 1999. Efficient computation of pager-
ank. Technical report, Stanford.
Sven Hertling and Heiko Paulheim. 2018. Dbkwik:
A consolidated knowledge graph from thousands of
wikis. In 2018 IEEE International Conference on
Big Knowledge (ICBK) , pages 17–24. IEEE.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning
steps. arXiv preprint arXiv:2011.01060 .

Johannes Hoffart, Yasemin Altun, and Gerhard Weikum.
2014. Discovering emerging entities with ambiguous
names. In Proceedings of the 23rd international
conference on World wide web , pages 385–396.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and 1 oth-
ers. 2025. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions. ACM Transactions on Information
Systems , 43(2):1–55.
Shaoxiong Ji, Shirui Pan, Erik Cambria, Pekka Martti-
nen, and Philip S Yu. 2021. A survey on knowledge
graphs: Representation, acquisition, and applications.
IEEE transactions on neural networks and learning
systems , 33(2):494–514.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. In Proceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing , pages 7969–7992.
Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. 2020. Dense passage re-
trieval for open-domain question answering. In
EMNLP (1) , pages 6769–6781.
Omar Khattab and Matei Zaharia. 2020. Colbert: Effi-
cient and effective passage search via contextualized
late interaction over bert. In Proceedings of the 43rd
International ACM SIGIR conference on research
and development in Information Retrieval , pages 39–
48.
Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao
Fu, Kyle Richardson, Peter Clark, and Ashish Sab-
harwal. 2023. Decomposed prompting: A modular
approach for solving complex tasks. In The Eleventh
International Conference on Learning Representa-
tions .
Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan
Raiman, Mohammad Shoeybi, Bryan Catanzaro, and
Wei Ping. 2024. Nv-embed: Improved techniques
for training llms as generalist embedding models.
arXiv preprint arXiv:2405.17428 .
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented
generation for knowledge-intensive nlp tasks. Ad-
vances in neural information processing systems ,
33:9459–9474.
Haoran Luo, Guanting Chen, Yandan Zheng, Xi-
aobao Wu, Yikai Guo, Qika Lin, Yu Feng, Zemin
Kuang, Meina Song, Yifan Zhu, and 1 others. 2025.
Hypergraphrag: Retrieval-augmented generation
via hypergraph-structured knowledge representation.
arXiv preprint arXiv:2503.21322 .Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and
Shirui Pan. 2024. Reasoning on Graphs: Faithful
and Interpretable Large Language Model Reasoning.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. In Proceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 9802–9822, Toronto,
Canada. Association for Computational Linguistics.
Jose L Martinez-Rodriguez, Ivan López-Arévalo, and
Ana B Rios-Alvarado. 2018. Openie-based approach
for knowledge graph construction from text. Expert
Systems with Applications , 113:339–355.
Costas Mavromatis and George Karypis. 2022. ReaRev:
Adaptive Reasoning for Question Answering over
Knowledge Graphs.
Yixin Nie, Songhe Wang, and Mohit Bansal. 2019.
Revealing the importance of semantic retrieval for
machine reading at scale. In Proceedings of the
2019 Conference on Empirical Methods in Natu-
ral Language Processing and the 9th International
Joint Conference on Natural Language Processing
(EMNLP-IJCNLP) , pages 2553–2566.
Barlas Oguz, Xilun Chen, Vladimir Karpukhin, Stan
Peshterliev, Dmytro Okhonko, Michael Schlichtkrull,
Sonal Gupta, Yashar Mehdad, and Scott Yih. 2020.
Unik-qa: Unified representations of structured and
unstructured knowledge for open-domain question
answering. arXiv preprint arXiv:2012.14610 .
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. 2024. Graph retrieval-augmented generation:
A survey. arXiv preprint arXiv:2408.08921 .
Urša Reja, Katja Lozar Manfreda, Valentina Hlebec,
and Vasja Vehovar. 2003. Open-ended vs. close-
ended questions in web questionnaires. Develop-
ments in applied statistics , 19(1):159–177.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D Manning.
2024. Raptor: Recursive abstractive processing for
tree-organized retrieval. In The Twelfth International
Conference on Learning Representations .
Kunal Sawarkar, Abhilasha Mangal, and Shivam Raj
Solanki. 2024. Blended rag: Improving rag
(retriever-augmented generation) accuracy with se-
mantic search and hybrid query-based retrievers. In
2024 IEEE 7th international conference on multi-
media information processing and retrieval (MIPR) ,
pages 155–161. IEEE.
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie
Huang, Nan Duan, and Weizhu Chen. 2023. Enhanc-
ing retrieval-augmented large language models with
iterative retrieval-generation synergy. In Findings
of the Association for Computational Linguistics:
EMNLP 2023 , pages 9248–9274.

Zhili Shen, Chenxin Diao, Pavlos V ougiouklis, Pascual
Merita, Shriram Piramanayagam, Damien Graux,
Dandan Tu, Zeren Jiang, Ruofei Lai, Yang Ren,
and 1 others. 2024. Gear: Graph-enhanced agent
for retrieval-augmented generation. arXiv preprint
arXiv:2412.18431 .
Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo
Wang, Chen Lin, Yeyun Gong, Lionel M. Ni, Heung-
Yeung Shum, and Jian Guo. 2024. Think-on-Graph:
Deep and Responsible Reasoning of Large Language
Model on Knowledge Graph.
Yixuan Tang and Yi Yang. 2024. MultiHop-RAG:
Benchmarking Retrieval-Augmented Gener- ation
for Multi-Hop Queries.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. Musique: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics , 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving Retrieval
with Chain-of-Thought Reasoning for Knowledge-
Intensive Multi-Step Questions. In Proceedings
of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) ,
pages 10014–10037, Toronto, Canada. Association
for Computational Linguistics.
Ellen M. V oorhees and Dawn M. Tice. 2000. The
TREC-8 question answering track. In Proceed-
ings of the Second International Conference on Lan-
guage Resources and Evaluation (LREC’00) , Athens,
Greece. European Language Resources Association
(ELRA).
Liang Wang, Ivano Lauriola, and Alessandro Moschitti.
2023a. Accurate training of web-based question
answering systems with feedback from ranked users.
InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume
5: Industry Track) , pages 660–667.
Liang Wang, Nan Yang, and Furu Wei. 2023b.
Query2doc: Query Expansion with Large Language
Models.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
and 1 others. 2022. Chain-of-thought prompting
elicits reasoning in large language models. Advances
in neural information processing systems , 35:24824–
24837.
Yike Wu, Nan Hu, Sheng Bi, Guilin Qi, Jie Ren, An-
huan Xie, and Wei Song. 2023. Retrieve-Rewrite-
Answer: A KG-to-Text Enhanced LLMs Framework
for Knowledge Graph Question Answering.
Yilin Xiao, Junnan Dong, Chuang Zhou, Su Dong, Qian-
wen Zhang, Di Yin, Xing Sun, and Xiao Huang.
2025. GraphRAG-Bench: Challenging Domain-
Specific Reasoning for Evaluating Graph Retrieval-
Augmented Generation.Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and
Yu Su. 2023. Adaptive chameleon or stubborn sloth:
Revealing the behavior of large language models in
knowledge conflicts. In The Twelfth International
Conference on Learning Representations .
Derong Xu, Xinhang Li, Ziheng Zhang, Zhenxi Lin,
Zhihong Zhu, Zhi Zheng, Xian Wu, Xiangyu Zhao,
Tong Xu, and Enhong Chen. 2025. Harnessing
Large Language Models for Knowledge Graph Ques-
tion Answering via Adaptive Multi-Aspect Retrieval-
Augmentation.
Wei Yang, Yuqing Xie, Aileen Lin, Xingyu Li, Luchen
Tan, Kun Xiong, Ming Li, and Jimmy Lin. 2019.
End-to-end open-domain question answering with
bertserini. In Proceedings of the 2019 Conference of
the North American Chapter of the Association for
Computational Linguistics (Demonstrations) , pages
72–77.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing , pages
2369–2380.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
React: Synergizing reasoning and acting in language
models. In International Conference on Learning
Representations (ICLR) .
Murong Yue. 2025. A survey of large language
model agents for question answering. arXiv preprint
arXiv:2503.19213 .
Fangyuan Zhang, Zhengjun Huang, Yingli Zhou, Qin-
tian Guo, Zhixun Li, Wensheng Luo, Di Jiang, Yixi-
ang Fang, and Xiaofang Zhou. 2025. EraRAG: Effi-
cient and Incremental Retrieval Augmented Genera-
tion for Growing Corpora.
Jiahao Zhang, Haiyang Zhang, Dongmei Zhang, Yong
Liu, and Shen Huang. 2024a. End-to-End Beam
Retrieval for Multi-Hop Question Answering.
Nan Zhang, Prafulla Kumar Choubey, Alexander Fab-
bri, Gabriel Bernadett-Shapiro, Rui Zhang, Prasenjit
Mitra, Caiming Xiong, and Chien-Sheng Wu. 2024b.
SiReRAG: Indexing Similar and Related Information
for Multihop Reasoning.
Zexuan Zhong, Zhengxuan Wu, Christopher Man-
ning, Christopher Potts, and Danqi Chen. 2023.
MQuAKE: Assessing knowledge editing in language
models via multi-hop questions. In Proceedings of
the 2023 Conference on Empirical Methods in Natu-
ral Language Processing , pages 15686–15702, Sin-
gapore. Association for Computational Linguistics.
Yingli Zhou, Yaodong Su, Youran Sun, Shu Wang, Tao-
tao Wang, Runyuan He, Yongwei Zhang, Sicong
Liang, Xilin Liu, Yuchi Ma, and 1 others. 2025. In-
depth analysis of graph-based rag in a unified frame-
work. arXiv preprint arXiv:2503.04338 .

A Methodology
As the T2RAG consists of several steps with
clear control flow, we illustrate it by the following
pseudo algorithm.
B Experiments
B.1 Detailed Implementations
For all experiments, we set the Large Language
Model (LLM) temperature to 0 to ensure determin-
istic and reproducible outputs. Local embedding
generation was performed on a single NVIDIA
L40S GPU.
A key aspect of our benchmark is the standard-
ization of the final answer format. We modified the
prompt for all methods to include a specific format
template, which yielded a significant performance
boost compared to baseline implementations in
other studies (Gutiérrez et al., 2025a; Xiao et al.,
2025). In those works, methods such as RAP-
TOR and IRCOT consistently performed about
10% lower than graph-based RAG approaches. Fur-
thermore, in our implementation of the RAPTOR,
we replaced the original Gaussian Mixture Model
(GMM) for clustering with K-Means. This de-
cision was based on the superior computational
efficiency of K-Means, which has been demon-
strated to produce results of similar quality for this
type of task (Zhou et al., 2025). The cluster size is
set to 10 and level is set to 3 following the bench-
mark (Zhou et al., 2025). For HippoRAG2, we
simply run their program and follow all the hyper-
parameters. The prompts and procedure of IRCoT
are all from the code of Zhang et al. (2024b). One
of the advantages of T2RAG is it free of hyper-
paremeter tunning compared to Raptor, which has
clustering parameters or HippoRAG2, which has
PageRank parameters and synonym link threshold.
B.2 More Efficiency Results
This section provides a detailed analysis of the
time and token consumption of various Retrieval-
Augmented Generation (RAG) methods, as illus-
trated in Figure 6 and Figure 7. The primary goal
is to evaluate the computational efficiency of our
proposed method, T2RAG, against other estab-
lished baselines across different stages of the RAG
pipeline. The y-axis represents the wall-clock time
in seconds required for the indexing and retrieval
stages. The retrieval stage time has been scaled by
a factor of 1000 to ensure visibility on the chart
alongside the much larger indexing times. The y-
axis represents the total number of LLM tokensconsumed. This is a weighted sum calculated us-
ing the formula: Token Consumption = (#input
tokens) + 4 ×(#output tokens) . This weighting
reflects the common pricing models of LLM APIs,
where generation (output) is typically priced sig-
nificantly higher (by a factor of 4) than processing
(input). As with the time consumption chart, the
retrieval stage consumption is scaled by 1000. The
x-axis in both figures shows the performance of
four methods (T2RAG, HippoRAG2, RAPTOR,
and IRCoT) across six distinct datasets.
B.2.1 Indexing Stage Analysis
The indexing stage is a one-time, offline process,
but its cost can be substantial and even prohibitive
for very large corpora. As seen in Figure 6 and Fig-
ure 7, datasets like PopQA, 2Wiki, and MuSiQue
demand a considerable amount of time and token
resources for indexing across all methods. The
consumption patterns reveal that indexing costs
are not simply proportional to the raw size of
the document corpus. For instance, the token con-
sumption for RAPTOR’s summarization and the
triplet extraction for T2RAG and HippoRAG2 do
not scale linearly with the number of documents.
This variability likely stems from the informative-
ness and density of the source documents. A
document rich with distinct facts will lead to more
triplets or more detailed summaries, increasing the
computational load, whereas a sparse document
will be processed more quickly. This makes the ex-
act indexing cost unpredictable without analyzing
the content itself.
B.2.2 Retrieval Stage Analysis
The retrieval stage is an online process that occurs
for every query, making its efficiency critical for
user-facing applications. Our analysis shows that
T2RAG is as efficient as HippoRAG2 during
the retrieval stage. Both methods exhibit simi-
lar time and token consumption profiles across all
datasets. This is expected, as their retrieval mecha-
nisms are conceptually similar, operating over the
graph structures built during indexing.
More importantly, T2RAG demonstrates a sub-
stantial efficiency gain over multi-round RAG
methods like IRCoT. As seen in Figure 7, T2RAG
consistently consumes fewer tokens during re-
trieval than IRCoT across all tested datasets. In
some cases, such as the Medical and Story datasets,
the reduction in token consumption is over 45%.
This efficiency stems from T2RAG’s ability to syn-
thesize a direct answer from the retrieved triplets
in a single round, avoiding the compounding token

Algorithm 1 T2RAG: Online Iterative Triplet Resolution (Main Process)
Input: Query q, Triplet DB Index I, LLM, Max Iterations K, Target unique chunks k, Triplet-to-
Chunk-Map Mchunk
Output: Final answer a
1: ▷Step 1: Structured Query Decomposition
2:Tresolved ,Tsearchable ,Tfuzzy←LLM Decompose (q)
3: ▷Step 2: Multi-Round Triplet Resolving Loop
4:forl= 1→Kdo
5: if|Tsearchable ∪ T fuzzy|= 0then
6: break
7: end if
8: ▷Step 2.1: Call the Adaptive Retrieval (see Algorithm 2)
9:Pretrieved ,Cretrieved ←ADAPTIVERETRIEVE (Tsearchable ,I, k,Mchunk)
10: ▷Step 2.2: LLM-based Triplets Resolution
11: T(new)
resolved,T(new)
searchable←LLM Resolve (Tsearchable ,Tfuzzy,Pretrieved ,Cretrieved )
12: ▷Step 2.3: State Update
13: Tresolved ← T resolved ∪ T(new)
resolved;Tsearchable ← T(new)
searchable;Tfuzzy← T fuzzy\(T(new)
resolved∪ T(new)
searchable)
14:end for
15: ▷Step 3: Final Answering
16:if|Tsearchable ∪ T fuzzy|= 0then
17: Tcontext← T resolved
18:else
19: Tcontext← T resolved ∪ T searchable
20:end if
21:a←LLM Answer (q,Tcontext)
22:return a
Table 3: Dataset Statistics
Dataset PopQA 2Wiki Musique HotpotQA Story Medical
# questions 1000 1000 1000 1000 794 564
# chunks 33,595 6119 11 ,656 9811 1266 268
# tokens 2,768,270 454 ,715 964 ,203 914 ,956 915 ,484 189 ,271
# extracted triplets 398,924 65 ,028 127 ,640 124 ,722 22 ,812 5256
costs associated with the iterative query refinement
process in multi-round architectures.
Remarkably, T2RAG often achieves lower, or
at least comparable, token consumption than
even single-round methods like RAPTOR. This
is particularly evident in datasets like PopQA, Med-
ical, and Story. We attribute this advantage to the
nature of the final answer generation. T2RAG gen-
erates a concise answer directly from the structured
triplets, which minimizes the number of output to-kens. Since output tokens are heavily weighted
in our consumption metric (multiplied by 4), this
concise, triplet-formulated output provides a sig-
nificant efficiency advantage, leading to an overall
reduction in computational cost.
B.3 More Iteration Results
This analysis examines the average number of re-
trieval iterations required by T ²RAG and IRCoT to
answer a query on the 2Wiki dataset, varying the
number of retrieved chunks (top- k) per iteration.

T2RAG
HippoRAG2RAPTORIRCoT024×104 PopQA
T2RAG
HippoRAG2RAPTORIRCoT0.00.51.0×104 2Wiki
T2RAG
HippoRAG2RAPTORIRCoT0.00.51.0×104 MuSiQue
T2RAG
HippoRAG2RAPTORIRCoT0.00.51.0×104 HotpotQA
T2RAG
HippoRAG2RAPTORIRCoT0.02.55.07.5×103 Medical
T2RAG
HippoRAG2RAPTORIRCoT0.02.55.07.5×103 StoryTime (s)
Indexing Stage Retrieval Stage (x1000)Figure 6: Time consumption at indexing and retrieval stages across all datasets.
T2RAG
HippoRAG2RAPTORIRCoT0.00.51.0×108 PopQA
T2RAG
HippoRAG2RAPTORIRCoT0.00.51.01.5×107 2Wiki
T2RAG
HippoRAG2RAPTORIRCoT012×107 MuSiQue
T2RAG
HippoRAG2RAPTORIRCoT0123×107 HotpotQA
T2RAG
HippoRAG2RAPTORIRCoT0123×107 Medical
T2RAG
HippoRAG2RAPTORIRCoT0123×107 StoryToken Consumption (#input + 4 * #output)
Indexing Stage Retrieval Stage (x1000)
Figure 7: Token consumption at indexing and retrieval stages across all datasets.

Algorithm 2 Adaptive Triplet Retrieval
Require: Searchable triplets Tsearchable , Index I, Target chunks k, MapMchunk
Ensure: Retrieved propositions Pretrieved , Retrieved chunks Cretrieved
1:function ADAPTIVE RETRIEVE (Tsearchable ,I, k,Mchunk)
2: Pcandidates ← ∅
3: fort∈ T searchable do
4: query_prop ←Concatenate (t)
5: query_vec ←E(query_prop )
6: Pcandidates ←Pcandidates ∪Search (I,query_vec , N)
7: end for
8: SortPcandidates globally by similarity score
9:Pretrieved ← ∅; unique_chunk_ids ← ∅
10: forp∈sorted Pcandidates do
11: if|unique_chunk_ids | ≥kchunks then
12: break
13: end if
14: Pretrieved ← P retrieved ∪ {p}
15: chunk_id ← M chunk[p]
16: unique_chunk_ids ←unique_chunk_ids ∪ {chunk_id }
17: end for
18: Cretrieved ←GetChunksFromIDs (unique_chunk_ids )
19: return Pretrieved ,Cretrieved
20:end function
Table 4: Average Number of Retrieval Iterations vs.
top-kon the 2Wiki Dataset.
topk T²RAG IRCoT
2 1.54 1.85
3 1.56 1.83
4 1.73 1.70
5 1.70 1.46
6 1.56 1.40
A key observation from the data is that T²RAG
consistently saves on the number of retrieval
iterations compared to IRCoT , particularly when
retrieving fewer documents per step ( k= 2 or 3).
For instance, with k= 2, T ²RAG requires an aver-
age of only 1.54 iterations, whereas IRCoT needs
1.85 iterations—a reduction of approximately 17%.
This suggests that T ²RAG’s method of decom-
posing a query into structured triplets allows for
a more direct and efficient path to resolving the
query, requiring fewer rounds of retrieval to gather
the necessary context.
The results challenge the simple assumption that
retrieving fewer chunks per iteration (a smaller k)
would necessarily lead to a higher number of total
iterations. For T ²RAG, the number of iterationsremains relatively stable and low, fluctuating be-
tween 1.54 and 1.73 without a clear trend. For
IRCoT, the relationship is even more complex; as
kincreases from 4 to 6, the number of iterations
surprisingly decreases significantly. This indicates
that the effectiveness of the retrieved chunks is
more important than the sheer quantity. T ²RAG’s
focused retrieval, guided by placeholders in triplets,
appears to acquire high-quality context more reli-
ably, making it less dependent on the kvalue and
more efficient overall.
C Related Work
We group prior efforts into single-round ,multi-
round ,graph-enhanced RAG and summarization-
based RAG, each adding more interaction or struc-
tured reasoning and paving the way for the fine-
grained design of T2RAG.
Single-round RAG. Classical sparse retrievers
such as TF-IDF and BM25 paired with extrac-
tive readers perform strongly for open-domain QA
(Yang et al., 2019; Nie et al., 2019; Wang et al.,
2023a). Dense retrievers such as DPR (Karpukhin
et al., 2020) later replaced sparse vectors with
learned embeddings, retrieving a fixed top- kset in
one pass. However, answering multi-hop questions

often demands the intermediate results to further
retrieval, motivating the multi-round techniques
that follow.
Multi-round RAG. Due to the missing bridges
problem we mentioned in Section 1 more and
more works follow a multi-round, training-free
paradigm, which enables the LLMs infer the in-
termediate information thus better retrieve the fi-
nal answer. Some works focus on the query side.
Khot et al. (2023) decompose multi-hop questions
into single-hop sub-queries that are solved sequen-
tially. Yao et al. (2023) propose ReAct, interleav-
ing chain-of-thought (CoT) (Wei et al., 2022) steps
with search actions issued by the LLM. Similariy,
Query2Doc (Wang et al., 2023b) expanding queries
into concise triplets to cut token usage while pre-
serving recall. Another line of works relies on
the generated intermediate results for next itera-
tion. Beam Retrieval (Zhang et al., 2024a) jointly
training an encoder and classifiers to keep multiple
passage hypotheses across hops. FLARE (Jiang
et al., 2023) forecasts upcoming sentences to de-
cide when fresh retrieval is needed during long-
form generation. IRCoT (Trivedi et al., 2023) and
ITER-RETGEN (Shao et al., 2023), alternately
expanding a CoT and fetching new evidence to an-
swer multi-step questions. Adaptive QA (Xie et al.,
2023) create an adaptive framework that picks the
simplest effective retrieval strategy according to
query complexity. Despite these advances, few
efforts explicitly aim to reduce token costs or num-
ber of llm calls during multi-round RAG. Previous
methods expand query or generates CoT with long
sentences in each round. In contrast, our work
minimizes token consumption by formulating query
expansions as triplets and simplifying reasoning
steps as triplets resolving.
Graph RAG. One major line of research addresses
complex QA by structuring knowledge into graphs.
Originating in Knowledge Graph QA (KGQA),
early methods focused on decomposing queries
or performing multi-round, LLM-evaluated traver-
sals from seed nodes (Luo et al., 2024; Sun et al.,
2024; Cheng et al., 2024; Mavromatis and Karypis,
2022). The application of this paradigm to gen-
eral ODQA was popularized by systems that con-
struct a knowledge graph entirely with LLMs
and use community detection for retrieval (Edge
et al., 2024). Subsequent work has aimed to make
this process more efficient. For instance, Ligh-
tRAG (Guo et al., 2024) introduces a dual-level
retrieval system combining graph structures with
vector search to improve knowledge discovery.Targeting resource-constrained scenarios, Mini-
RAG (Fan et al., 2025) builds a heterogeneous
graph of text chunks and named entities, enabling
lightweight retrieval suitable for Small Language
Models. To tackle the common challenge of en-
tity merging, HippoRAG (Gutiérrez et al., 2025a)
and HippoRAG2 (Gutiérrez et al., 2025b) create
synonym links between similary entity nodes and
employs a PageRank (Haveliwala, 1999) algorithm
for final node selection. Despite these advances,
a central challenge for Graph RAG remains the
costly and error-prone nature of graph construc-
tion from unstructured text.
Summarization-based RAG. A distinct but re-
lated approach focuses on building hierarchical
summarization trees rather than explicit graphs.
These methods aim to capture information at vary-
ing levels of abstraction. For example, Rap-
tor (Sarthi et al., 2024) constructs a summary tree
by recursively clustering document chunks and
summarizing the content within each cluster to cre-
ate new, more abstract retrieval units (Wu et al.,
2023). Aiming to capture more detailed contextual
information, SireRAG (Zhang et al., 2024b) creates
a "relatedness tree" by summarizing fine-grained
propositions that share the same entities. How-
ever, these summarization-based methods often in-
cur high computational costs during the indexing
phase and risk losing the fine-grained, factual de-
tails that are essential for precise factoid QA.
D Case Study
We offer a full log of T2RAG during our experi-
ment running in Figure 8.
This case study showcases the effectiveness of
resolving the complex comparative query in 2 re-
trieval iterations. The system successfully decom-
posed the query into 4 necessary triplets (two di-
rectors, two birth years) and retrieved context only
by the searchable ones. By identifying both direc-
tors (Michael Curtiz, Edith Carlmar) and their birth
years (1886, 1911) from the triplet DB or initial set
of chunks, it bypassed the need for further retrieval
rounds. This immediate and complete information
acquisition demonstrates the power of T2RAG’s
query decomposition and high-quality triplet-based
retrieval.
E Prompts
We provide all prompt templates we used
at retrieval stage, namely structured query
decomposition, triplet resolving and fi-
nal answering. These are prompts used in

Question : Which film has the director born earlier, God’s Gift To Women orAldri Annet Enn Brak ?
Q: Think step by step about what information is needed to answer this question. Form triples in the format: subject | predicate | object. Use "?" as placeholder for unknown entities…
A: 
Searchable Triplets: God’s Gift To Women ,is directed by, ?DirectorA ; Aldri Annet Enn Brak ,is directed by, ?DirectorB ; 
Fuzzy Triplets: ?DirectorA , was born in, ? birthYearA ; ?DirectorB , was born in, ? birthYearB
Q: Use the context passages and propositions to solve any '?' placeholders with as much detail as possible, grounding your answe rs in the passage content. 
Retrieved Triplets : Aldri annet ennbrak was directed by Edith Carlmar , God's Gift to Women directed by Michael Curtiz, God's Gift to Women was completed as musical film, 
Aldri annet ennbrak is 1954 Norwegian comedy -drama film, God's Gift to Women starring Joan Blondell, Aldri annet ennbrak has English title Nothing but trouble, God's Gift to 
Women is 1931 American pre -Code romantic musical comedy film … (18 propositions in total)
Retrieved Chunks : Title: Altid ballade …  Title: God's Gift to Women … Title: Aldri annet enn brak ... Title : Edith Carlmar (Edith Mary Johanne Mathiesen) (15 November 1911 -
17 May 2003)  Title: Stanley Kwan
A: 
Newly Resolved triplets: God’s Gift To Women ,is directed by, Michael Curtiz ; Aldri Annet Enn Brak ,is directed by, Edith Carlmar ; Edith Carlmar , was born in, 1911
Newly searchable chunks : Michael Curtiz , was born in, ? birthYearA ;Query Decomposition
Iteration 1
Q: Use the context passages and propositions to solve any '?' placeholders with as much detail as possible, grounding your answe rs in the passage content. 
Retrieved Triplets : Michael Curtiz born as Mih1ly Kertsz , Michael Curtiz was Hungarian -born American film director, Michael Curtiz was born on December 24, 1886 , Michael 
Curtiz made stars of Bette Davis, The Vagabond King directed by Michael Curtiz, Bright Leaf directed by Michael Curtiz… (32 propositions in total )
Retrieved chunks : Title: Michael Curtiz Michael Curtiz \nMichael Curtiz (born Man Kaminer(1886 -1905) Mih1ly Kertz (1905); December 24, 1886 April 11, 1962) was a 
Hungarian -born American film director… Title: Bright Leaf… Title:  Altid ballade…Title: The Vagabond King… Title: JdJds…
A: 
Newly Resolved Triplets: Michael Curtiz , was born in, 1886
Newly Searchable Triplets : None.（all triplets resolved and the iteration is terminated)Iteration 2
Q: Based on the following triplets, please answer the following question. 
Total Resolved Triplets : God’s Gift To Women ,is directed by, Michael Curtiz ; Aldri Annet Enn Brak ,is directed by, Edith Carlmar ; Michael Curtiz , was born in, 1886 ; Edith 
Carlmar , was born in, 15 November 1911.
A: God’s Gift To WomenFinal Answering Figure 8: An example of T2RAG QA. To answer the question, we need intermediate facts about Michael Curtiz
(marked by yellow and Edith Carlmar (marked by red), which are not reflected in the question.
LLM Decompose ,LLM Resolve ,LLM Answer , respec-
tively. {·}represents the content needed to be
replaced by the original question, intermediate
generated triplets, or retrieved propositions and
chunks.

Structured Query Decomposition
You are tasked with reasoning about a question and extracting the necessary knowledge triples to
answer it.
Instructions :
1. Think step by step about what information is needed to answer this question
2. Form triples in the format: subject | predicate | object
3. Use "?" as placeholder for unknown entities
4. For comparative questions involving multiple entities, use distinct placeholders like ?entityA,
?directorA, ?directorB
5. Extract multiple triples if the question requires complex reasoning
Examples :
- Question: "What is the capital of France?" Reasoning: To answer this, I need to know what
France’s capital is. Triple: France | has capital | ?
- Question: "Who directed the movie that won Best Picture in 2020?" Reasoning: To answer this, I
need to know which movie won Best Picture in 2020, and who directed that movie. Triples: ? |
won Best Picture | 2020 ? | is directed by | ?
- Question: "Which film whose director was born first, MovieA or MovieB?" Reasoning: To
answer this, I need to know the director of each movie, and the birth year of each director to
compare them. Triples: MovieA | is directed by | ?directorA MovieB | is directed by | ?directorB
?directorA | was born in | ? ?directorB | was born in | ?
Now analyze this question:
Question : {query}
Provide your response in this format:
Reasoning : [Your step-by-step reasoning about what information is needed]
Triples : [List each triple on a new line in format: subject | predicate | object]
Triplets Resolving
Example: Context Propositions: {context propositions}
Fully Resolved Clue 1: Subject: Lothair II Predicate: has mother Object: Ermengarde of Tours
Newly Searchable Clue 1: Subject: Ermengarde of Tours Predicate: died on Object: ?
—
Now apply the same process to the following clues: Use the context passages and propositions
to resolve any ’?’ placeholders with as much detail as possible, grounding your answers in the
passage content. Instructions:
1. For searchable clues (one ’?’), replace ’?’ with the correct entity to fully resolve it, including
any relevant attributes.
2. For fuzzy clues (multiple ’?’), generate a Newly Searchable Clue by replacing one of the
placeholders with the correct entity, including any relevant context.
Original Query: {query}
Searchable Clues: {searchable clues text}
Fuzzy Clues: {fuzzy clues text}
Context Passages: {context passages}
Context Propositions: {context propositions}
Previous Resolved Clues: {resolved clues context}
Return two lists in this format:
Fully Resolved Clue 1: Subject: ... Predicate: ... Object: ...
Fully Resolved Clue 2: Subject: ... Predicate: ... Object: ...
Newly Searchable Clue 1: Subject: ... Predicate: ... Object: ...
Newly Searchable Clue 2: Subject: ... Predicate: ... Object: ...
(Continue numbering accordingly)

Final Answering
Based on the reasoning clues, please answer the following question.
Question: {query}
Key Reasoning Clues: {total resolved clues + remaining searchable clues}
Instructions:
1. Analyze the question step by step
2. Use the reasoning clues to understand what information is needed
3. Provide ONLY a concise answer
Answer format requirements:
- For WH questions (who/what/where/when): Provide the exact entity, date, full name, or full
place name only
- For yes/no questions: Answer only "yes" or "no"
- No explanations, reasoning, or additional text
- One entity or fact only
Answer:
# IndexRAG: Bridging Facts for Cross-Document Reasoning at Index Time

**Authors**: Zhenghua Bao, Yi Shi

**Published**: 2026-03-17 11:51:31

**PDF URL**: [https://arxiv.org/pdf/2603.16415v1](https://arxiv.org/pdf/2603.16415v1)

## Abstract
Multi-hop question answering (QA) requires reasoning across multiple documents, yet existing retrieval-augmented generation (RAG) approaches address this either through graph-based methods requiring additional online processing or iterative multi-step reasoning. We present IndexRAG, a novel approach that shifts cross-document reasoning from online inference to offline indexing. IndexRAG identifies bridge entities shared across documents and generates bridging facts as independently retrievable units, requiring no additional training or fine-tuning. Experiments on three widely-used multi-hop QA benchmarks (HotpotQA, 2WikiMultiHopQA, MuSiQue) show that IndexRAG improves F1 over Naive RAG by 4.6 points on average, while requiring only single-pass retrieval and a single LLM call at inference time. When combined with IRCoT, IndexRAG outperforms all graph-based baselines on average, including HippoRAG and FastGraphRAG, while relying solely on flat retrieval. Our code will be released upon acceptance.

## Full Text


<!-- PDF content starts -->

IndexRAG: Bridging Facts for Cross-Document Reasoning at Index Time
Zhenghua Bao
Continuum AI
Shanghai, China
zhenghua.bao@continuum-ai.devYi Shi
Continuum AI
San Francisco, CA, USA
yi.shi@continuum-ai.dev
Abstract
Multi-hop question answering (QA) requires
reasoning across multiple documents, yet ex-
isting retrieval-augmented generation (RAG)
approaches address this either through graph-
based methods requiring additional online pro-
cessing or iterative multi-step reasoning. We
present IndexRAG, a novel approach that shifts
cross-document reasoning from online infer-
ence to offline indexing. IndexRAG identifies
bridge entities shared across documents and
generates bridging facts as independently re-
trievable units, requiring no additional training
or fine-tuning. Experiments on three widely-
used multi-hop QA benchmarks (HotpotQA,
2WikiMultiHopQA, MuSiQue) show that In-
dexRAG improves F1 over Naive RAG by 4.6
points on average, while requiring only single-
pass retrieval and a single LLM call at inference
time. When combined with IRCoT, IndexRAG
outperforms all graph-based baselines on aver-
age, including HippoRAG and FastGraphRAG,
while relying solely on flat retrieval. Our code
will be released upon acceptance.
1 Introduction
Large language models (LLMs), built on the Trans-
former architecture (Vaswani et al., 2017), have
achieved remarkable performance across a wide
range of tasks, from language (Brown et al., 2020)
to vision (Dosovitskiy et al., 2020; Radford et al.,
2021). However, their reliance on static para-
metric knowledge limits their ability to incorpo-
rate domain-specific or up-to-date information, of-
ten leading to hallucinations (Huang et al., 2025).
Retrieval-Augmented Generation (RAG) mitigates
this by grounding generation in an external knowl-
edge base, allowing the model to retrieve relevant
context before responding (Lewis et al., 2020).
While effective for single-hop questions, con-
ventional RAG pipelines retrieve passages inde-
pendently and struggle when answering a question
LLMQuery
Where was the 
director of film 
Aylwin born?Result
Henry Edwards
Bridging fact
Henry Edwards, director of Aylwin, 
was born in Weston-super-Mare.Chunk A
Aylwin directed by H. Edwards 
LLMQuery
Where was the 
director of film 
Aylwin born?Naive RAG
IndexRAG
(ours)Chunk A
Aylwin directed by H. Edwards
Chunk B
Henry Edwards appeared in 30 films
Chunk C
Henry Edwards born in Weston-
super-Mare
Result
Weston-super-Mare
Chunk C
Henry Edwards born in Weston-
super-Mare
NOT RETRIEVEDDocs
ChunksDocs DocsNaive RAG
 flat retrieval at inferenceGraphRAG
graph traversal at inference IndexRAG (Ours)
flat retrieval at inference
Bridging factsReasoning
a) Offline IndexingEntity & Relation Fixed Chunking
Knowledge Graph AKUs
b) Online InferenceFigure 1: (a) Comparison of different RAG approaches
during offline indexing. IndexRAG shifts reasoning to
index time, generating bridging facts stored alongside
AKUs in a unified vector store. (b) Example from 2Wiki-
MultiHopQA (Ho et al., 2020). The query“Where was
the director of the film Aylwin born?”requires two-
hop reasoning. With naive RAG, Chunk C containing
the director’s birthplace is not retrieved, causing the
LLM to output the director’s name instead. With In-
dexRAG, a bridging fact connecting the film to the
director’s birthplace is directly retrieved, enabling the
correct answer. See Section 5.2.1 for detailed analysis.
requires synthesizing information across multiple
documents. Multi-hop QA requires reasoning over
multiple pieces of evidence spread across different
documents to reach the correct answer (Yang et al.,
2018; Ho et al., 2020; Trivedi et al., 2022). Graph-
based RAG systems (Edge et al., 2024; Gutiérrez
et al., 2024; You et al., 2025) address this by con-
structing knowledge graphs that explicitly repre-
sent inter-document relationships. However, these
methods often require multi-step online processing,
such as query-time entity extraction, graph traver-
sal, and multiple LLM calls, which increases infer-
ence cost and latency. Iterative approaches such as
IRCoT (Trivedi et al., 2023) decompose complex
queries through multiple rounds of retrieval and
generation, resulting in higher inference cost and
1arXiv:2603.16415v1  [cs.CL]  17 Mar 2026

Single-pass Cross-doc Single LLM Training- Index-time
retrieval reasoning call free reasoning
Naive RAG✓×✓ ✓×
HippoRAG ×✓×✓×
IRCoT ×✓×✓×
IndexRAG (Ours)✓ ✓ ✓ ✓ ✓
Table 1: Qualitative comparison of different RAG approaches. IndexRAG is the only method that achieves cross-
document reasoning with single-pass retrieval and a single LLM call at inference time.
slower response times.
To address these limitations, we propose In-
dexRAG, a novel approach that shifts cross-
document reasoning from online inference to of-
fline indexing. We observe that cross-document
reasoning patterns are largely query-independent,
as the connections between documents are deter-
mined by their content rather than any specific
query. This makes it possible to precompute these
reasoning connections at indexing time, transform-
ing per-query reasoning into an offline step. We
refer to this paradigm asindex-time reasoning. Dur-
ing indexing, we extract atomic knowledge units
(AKUs) and entities from each document, and
identify entities that appear across multiple docu-
ments. We then prompt an LLM to generatebridg-
ing factsthat capture cross-document reasoning
by linking related evidence from different sources.
Both AKUs and bridging facts are stored in a uni-
fied vector store. At inference time, a single re-
trieval pass and a single LLM call suffice, without
graph traversal, query decomposition, or iterative
retrieval-generation loops.
Our main contributions are as follows:
•We proposeIndexRAG, a novel approach that
shifts cross-document reasoning from online
inference to offline indexing.
•We introducebridging facts, a new type of
retrieval unit that encodes cross-document rea-
soning as independently retrievable entries in
a flat vector store, along with a balanced con-
text selection mechanism to control their pro-
portion at retrieval time.
•We propose a training-free framework that is
agnostic to the underlying retrieval strategy
and compatible with iterative methods such as
IRCoT, requiring no fine-tuning of the embed-
ding model or LLM.•Experiments on three multi-hop QA bench-
marks show that IndexRAG outperforms all
single-LLM-call baselines on average, and fur-
ther surpasses multi-call methods including
HippoRAG when combined with IRCoT.
2 Related Work
2.1 Retrieval-Augmented Generation
To mitigate hallucinations arising from the static
parametric knowledge of LLMs, RAG augments
generation with external retrieval (Lewis et al.,
2020). In a typical RAG pipeline, documents are
encoded into a vector store, and relevant passages
are retrieved based on embedding similarity at in-
ference time. Recent research has focused on im-
proving retrieval representations. Karpukhin et al.
(2020) replace sparse retrieval with learned dense
representations, while Gao et al. (2023) prompt
the LLM to generate a hypothetical answer and
use it as the retrieval query to better align with
target passages. Other work investigates how doc-
uments should be segmented for indexing. Chen
et al. (2024) show that finer-grained units such as
propositions yield stronger retrieval than fixed-size
chunks. While propositions decompose individual
passages into finer units, they do not capture rea-
soning across documents. RAPTOR (Sarthi et al.,
2024) recursively summarizes passages into a hier-
archical tree structure for retrieval.
These methods primarily focus on improving
how individual passages are represented or orga-
nized. In contrast, our work focuses on the implicit
reasoning connections across documents, generat-
ing new retrieval units that make cross-document
inferences explicitly retrievable.
2.2 Multi-hop Reasoning in RAG
Standard RAG retrieves passages independently,
struggling with questions that require reasoning
across multiple documents (Yang et al., 2018; Ho
2

LLMAKUs
(a1, a2, …, an)
Entities
(e1, e2, …, en)
Stage 2: Cross-Document Bridging Fact Generation
Flat RetrievalBalanced Context Selection (max kb)
…
Entity GroupBy
(freq ≥ 2)LLMBridging Facts
(b1, b2, …, bm)
Stage 1: AKU & Entity Extraction
a) Offline Indexing
LLM Answer a Query q 
b) Online InferenceAnswer aDocuments
(d1, d2, …, dn)Vector StoreFigure 2: Overview of IndexRAG. (a) Offline Indexing: In Stage 1, the LLM extracts AKUs and associated entities
from each document. In Stage 2, documents sharing the same entity are identified and used to generate bridging
facts. The resulting AKUs and bridging facts are encoded and stored in a unified vector store. (b) Online Inference:
the user query qis encoded and used to retrieve relevant context. After applying balanced context selection with kb,
the selected context (a mix of AKUs and bridging facts) is fed to the LLM to generate the answera.
et al., 2020; Trivedi et al., 2022). Two main direc-
tions have emerged to address this limitation.
The first incorporates graph structures into RAG.
GraphRAG (Edge et al., 2024) constructs an entity
graph from the corpus, applies community detec-
tion, and generates summaries for each community
to support global reasoning. FastGraphRAG1is
another graph-based approach that prioritizes in-
ference efficiency. HippoRAG (Gutiérrez et al.,
2024) draws inspiration from the human hippocam-
pal memory system, using a knowledge graph as
a long-term memory index to connect related pas-
sages, with Personalized PageRank for graph-based
retrieval. MS-RAG (You et al., 2025) maintains
separate vector stores for entities, relations, and
chunks, using vector search for entity extraction,
but still requires graph traversal at inference time.
Although these methods improve multi-hop per-
formance, they require additional inference-time
processing, such as entity extraction, graph traver-
sal, or reranking, beyond standard vector search.
The second direction uses iterative retrieval. IR-
CoT (Trivedi et al., 2023) interleaves chain-of-
thought reasoning with retrieval, using intermedi-
ate reasoning steps to formulate new queries across
multiple rounds. This improves recall of relevant
evidence but introduces repeated retrieval and gen-
eration cycles that increase inference latency.
In contrast, IndexRAG shifts cross-document
reasoning to indexing time, making implicit con-
nections retrievable through standard vector search
without additional inference-time overhead.
1https://github.com/circlemind-ai/
fast-graphrag3 Method
3.1 Overview
IndexRAG follows a two-phase pipeline: offline
indexing and online inference. During offline in-
dexing, we first extract atomic knowledge units
(AKUs) and entities from each document (Stage
1), then generate bridging facts that capture cross-
document reasoning by linking documents through
shared entities (Stage 2). Both AKUs and bridging
facts are encoded and stored in a unified vector
store, making cross-document reasoning directly
accessible through standard vector search. During
online inference, we perform single-pass retrieval
with balanced context selection and prompt the
LLM for answer generation. The overall pipeline
is illustrated in Figure 2 and detailed below.
3.2 Offline Indexing
3.2.1 Stage 1: AKU and Entity Extraction
Given a corpus of ndocuments D=
{d1, d2, . . . , d n}, we prompt an LLM to extract a
set of atomic facts, structured as question-answer
pairs, and associated entities from each document
di. We retain only the answers, merging them into
a single text unit per document as the minimal
retrievable unit. We refer to this unit as an
atomic knowledge unit(AKU), denoted ai. The
LLM simultaneously extracts a set of entities
Ei={e(i)
1, e(i)
2, . . .} . Each AKU aiis then
encoded by a dense embedding model fand stored
in a flat vector store V. Empirically, the choice of
extraction method in Stage 1 does not significantly
affect overall pipeline performance (Section 5.3.1).
3

3.2.2 Stage 2: Bridging Fact Generation
The key observation motivating this stage is that
documents sharing common entities often contain
complementary information. While each document
alone may only provide partial evidence, their com-
bination can yield reasoning conclusions not explic-
itly stated in any single source. By making these
implicit connections explicit and retrievable, Stage
2 enables cross-document reasoning without any
additional inference-time processing.
Bridge Entity Identification.We aggregate all
entities from Stage 1 and identifybridge entities
that appear across multiple documents:
Ebridge =(
e∈n[
i=1Ei2≤df(e)≤τ)
(1)
where df(e) =|{d i∈ D:e∈E i}|is the docu-
ment frequency of entity e, andτis an upper-bound
threshold. The lower bound ensures that bridge en-
tities connect at least two documents, while the
upper bound τexcludes overly generic entities
(e.g., common nouns or high-frequency terms) that
would produce trivial bridging facts.
Bridging Fact Generation.For each bridge en-
titye∈ E bridge , letDe={d i∈ D:e∈E i}be
the set of documents mentioning e. We collect the
subset of facts in each AKU aithat mention en-
titye, denoted ai[e], identified via string matching
against the entity list from Stage 1. We then prompt
the LLM to generate bridging facts:
Be=LLM(e,{a i[e] :d i∈ D e})(2)
where Beis the resulting set of bridging facts. We
limit the number of source documents to 5 and facts
per document to 8 to manage input length.
Unlike cross-document summaries, which com-
press existing content into shorter passages, bridg-
ing facts are constructed to directly answer implicit
cross-document questions. For example, given
Doc A stating “Aylwin is directed by Henry Ed-
wards” and Doc B stating “Henry Edwards was
born in Weston-super-Mare”, IndexRAG generates
the bridging fact “The director of the film Aylwin
was born in Weston-super-Mare”. Unlike either
source document, this fact directly encodes the two-
hop reasoning chain and is semantically aligned
with queries like “Where was the director of Ayl-
win born?”, making it retrievable via standard vec-
tor search even though neither source document
alone is sufficient to answer the query.All bridging facts B=S
e∈E bridgeBeare indi-
vidually encoded by fand stored alongside the
AKUs in V. When a new document dnewis added,
only Stage 1 for dnewand Stage 2 for affected
bridge entities need to be re-executed. This in-
cludes both existing bridge entities that appear in
dnew(Enew∩Ebridge ) and newly formed bridge enti-
ties whose document frequency reaches 2 after the
addition. The rest of the index remains unchanged.
3.3 Online Inference
Given a query q, we encode it with the same embed-
ding model fand retrieve the top- kentries from
Vby cosine similarity, where vj∈ V is either an
AKU or a bridging fact.
Balanced Context Selection.The retrieved top-
kset typically contains a mix of AKUs and bridg-
ing facts. However, because bridging facts are
considerably shorter than AKUs on average (166
vs. 634 characters in our experiments), they tend
to dominate the top- k, crowding out the longer,
information-dense AKUs.
To control this, let R={r 1, r2, . . .} be all en-
tries in Vsorted by descending similarity to q,
where each rjis either an AKU ( rj∈ A ) or a
bridging fact ( rj∈ B). We greedily build the con-
textCby iterating over Rin order. Each entry is
included if it is an AKU or if the number of bridg-
ing facts already in Cis below kb, until |C|=k .
The procedure is summarized in Appendix B.
The selected context Cis concatenated in sim-
ilarity order and provided to the LLM for answer
generation (see Figure 2b), without any additional
inference-time overhead.
4 Experimental Setup
4.1 Datasets
We evaluate on three widely-used multi-hop QA
datasets:HotpotQA(Yang et al., 2018),2Wiki-
MultiHopQA(Ho et al., 2020), andMuSiQue
(Trivedi et al., 2022).
HotpotQA contains 113k questions requiring rea-
soning over multiple Wikipedia passages. 2Wiki-
MultiHopQA provides multi-hop questions across
different types (comparison, bridge comparison,
compositional, and inference) with structured rea-
soning paths. MuSiQue consists of 25k 2-4 hop
questions composed from single-hop sub-questions,
and is generally considered the most challenging
among the three. Following prior work (Press et al.,
2023; Trivedi et al., 2023; Gutiérrez et al., 2024),
4

Dataset Questions Passages
HotpotQA 1,000 9,827
2WikiMultiHopQA 1,000 6,262
MuSiQue 1,000 9,723
Table 2: Dataset statistics for the 1,000 question subsets
used in our experiments.
we sample 1,000 questions from each validation set
and collect all associated passages (supporting and
distractor) as the retrieval corpus. Dataset statistics
are summarized in Table 2.
4.2 Implementation Details
We use GPT-4o-mini2as the LLM for all stages
and all methods, including baselines, to ensure a
fair comparison. All text is encoded using Ope-
nAI’s text-embedding-3-small3, and we use FAISS
(Douze et al., 2025) for vector indexing. Prompt
templates for all stages and answer generation are
provided in Appendix D.
Offline Indexing.In Stage 1, we pass each full
document to the LLM for AKU and entity extrac-
tion. In Stage 2, we set the entity frequency thresh-
oldτ= 10 , with a maximum of 5 source docu-
ments and 8 facts per document for each bridge en-
tity. The offline pipeline requires one LLM call per
document for Stage 1 and one LLM call per bridge
entity for Stage 2, with the total Stage 2 cost pro-
portional to |Ebridge|. Entities with no meaningful
cross-document connection are discarded, account-
ing for the non-empty rates in Table 3. For baseline
methods that require chunking (Naive RAG, RAP-
TOR, etc.), we split documents into passages of
approximately 100 words following (Chen et al.,
2024) with 80 characters of overlap.
Online Inference.All methods retrieve 20 can-
didates and select the top k= 10 for answer gener-
ation. For IndexRAG, balanced context selection
withkb= 3is additionally applied to control the
proportion of bridging facts in the selected context.
Unless otherwise specified, kb= 3 is used in all
IndexRAG experiments.
Since all evaluated datasets expect short-form
answers (typically a few words), we set the gener-
ation temperature to 0 and max tokens to 50. For
IRCoT experiments, we follow Trivedi et al. (2023)
2https://openai.com/index/
gpt-4o-mini-advancing-cost-efficient-intelligence/
3https://developers.openai.com/api/docs/
models/text-embedding-3-smallHotpotQA 2Wiki MuSiQue
Bridge entities 6,400 4,817 6,471
Bridging facts 8,000 6,821 8,010
Non-empty rate 80% 88% 79%
Table 3: Bridging fact generation statistics.
with 3 reasoning steps and top-20 retrieval per step.
When combined with IndexRAG, bridging facts
are included in the retrieval corpus and balanced
context selection is applied at the final answer gen-
eration step, withk b= 3.
4.3 Baselines
We compare IndexRAG against baselines span-
ning five categories: (1)Sparse retrieval:BM25
(Robertson and Walker, 1994), (2)Dense retrieval:
Naive RAG with fixed-size chunking using the
same embedding model, (3)Iterative Approach:
IRCoT (Trivedi et al., 2023), (4)Graph-based:
FastGraphRAG4and HippoRAG (Gutiérrez et al.,
2024), and (5)Hierarchical:RAPTOR (Sarthi
et al., 2024). We adopt FastGraphRAG as a repre-
sentative graph-based baseline due to its improved
efficiency over GraphRAG (Edge et al., 2024)
while maintaining competitive performance. Hip-
poRAG requires an additional LLM call for query-
time entity extraction and graph-based retrieval via
Personalized PageRank. Note that IRCoT was orig-
inally evaluated with BM25. To demonstrate the
generality of Stage 2, we also report IRCoT com-
bined with bridging facts.
4.4 Evaluation Metrics
We evaluate using three metrics: Exact Match
(EM), Accuracy (Acc), and F1 score, all computed
after answer normalization following Rajpurkar
et al. (2016). In addition, we report average re-
trieval latency and the number of LLM calls at
inference time to assess efficiency. Formal defini-
tions are provided in Appendix A.
5 Results
5.1 Quantitative Results
5.1.1 Multi-hop QA Performance
Table 4 presents the main results across the three
multi-hop QA benchmarks. We report F1 score as
the primary metric as it captures partial credit for
partially correct answers.
4https://github.com/circlemind-ai/
fast-graphrag
5

HotpotQA 2WikiMultiHopQA MuSiQue Average
EM Acc F1 EM Acc F1 EM Acc F1 EM Acc F1
BM25 47.2 51.2 60.3 31.6 32.5 35.9 9.8 10.7 19.2 29.5 31.5 38.5
Naive RAG 50.2 54.0 63.6 42.2 44.5 47.7 19.0 20.4 29.9 37.1 39.6 47.1
FastGraphRAG 50.0 53.8 63.549.5 54.8 57.417.5 18.8 27.2 39.0 42.5 49.4
RAPTOR 50.2 54.1 63.6 42.3 44.2 47.8 19.3 20.3 29.7 37.3 39.5 47.0
IndexRAG 54.1 59.3 68.944.8 50.3 51.722.4 24.4 34.4 40.3 44.7 51.7
HippoRAG56.5 60.7 70.550.2 55.7 57.2 23.8 25.5 34.7 43.5 47.3 54.1
IRCoT 49.8 54.9 62.5 39.7 41.3 43.1 11.1 12.3 19.3 33.5 36.2 41.6
IRCoT + IndexRAG 54.5 59.7 68.753.4 61.2 61.2 24.6 26.5 35.0 44.2 49.1 55.0
Table 4: Multi-hop QA performance (%) on three multi-hop QA benchmarks. IndexRAG uses kb= 3. Best
single-call results per dataset are inbold. Best multi-call results are inbold gray. Methods requiring multiple LLM
calls at inference time are greyed out.
Single-call methods.Among methods requir-
ing only a single LLM call at inference time, In-
dexRAG achieves the highest average scores across
all three metrics. IndexRAG obtains an average
F1 of 51.7, improving over Naive RAG (+4.6),
BM25 (+13.2), FastGraphRAG (+2.3), and RAP-
TOR (+4.7). On HotpotQA and MuSiQue, In-
dexRAG ranks first across all metrics, with par-
ticularly strong gains on MuSiQue (34.4 vs. 29.9
for Naive RAG), where all baselines achieve their
lowest scores across the three datasets. On 2Wiki-
MultiHopQA, FastGraphRAG achieves the best
single-call performance (57.4). This dataset is dom-
inated by comparison and bridge comparison ques-
tions (44%), which require parallel entity lookups
that naturally align with explicit graph structures.
IndexRAG’s bridging facts are generated along se-
quential reasoning paths and provide limited bene-
fit for these question types, as further analyzed in
Section 5.3.3. Despite this, IndexRAG achieves the
best average F1 among all single-call methods.
Multi-call methods.When combined with IR-
CoT, IndexRAG achieves the best results among
all methods including multi-call ones. IRCoT +
IndexRAG reaches an average of 55.0, surpassing
HippoRAG (54.1) and standalone IRCoT (41.6).
Adding bridging facts to IRCoT improves perfor-
mance by 13.4 points over IRCoT alone on aver-
age, with the largest gain on 2WikiMultiHopQA
(+18.1), suggesting that bridging facts provide
cross-document context that iterative reasoning
alone cannot capture. HippoRAG achieves the
highest single-dataset score on HotpotQA (70.5),
but requires an additional LLM call for query-time
entity extraction, resulting in significantly higher
retrieval latency, as detailed in Section 5.1.2.Method EM Time(s) Calls
Naive RAG 19.0 0.29 1.0
FastGraphRAG 17.5 2.55 1.0
RAPTOR 19.3 0.47 1.0
IndexRAG 22.40.30 1.0
HippoRAG 23.8 3.13 2.0
IRCoT + IndexRAG 24.6 1.08 3.2
Table 5: Quality-efficiency comparison (%) on
MuSiQue. Gray rows indicate multi-call methods.
5.1.2 Quality-Efficiency Comparison
Table 5 presents the quality-efficiency trade-off on
MuSiQue. Among single-call methods, IndexRAG
achieves the best EM (22.4) with a retrieval latency
of 0.30 seconds, nearly identical to Naive RAG
(0.29s) while improving EM by 3.4 points. Fast-
GraphRAG, despite requiring only a single LLM
call, requires 2.55 seconds per query, 8.5 ×slower
than IndexRAG while achieving lower EM (17.5).
Among multi-call methods, HippoRAG achieves a
slightly higher EM of 23.8 but at the cost of two
LLM calls and a retrieval latency of 3.13 seconds,
over 10 ×that of IndexRAG. IRCoT + IndexRAG
achieves the best EM of 24.6 with a retrieval latency
of 1.08 seconds, still 3 ×faster than HippoRAG de-
spite averaging 3.2 LLM calls per query.
These results show that IndexRAG effectively
shifts computational cost from inference to index-
ing, achieving strong performance with minimal
online overhead.
5.2 Qualitative Results
5.2.1 Case Study
To illustrate how bridging facts improve multi-hop
retrieval, we examine a concrete example from
2WikiMultiHopQA (Table 6).
The query “Where was the director of the film
Aylwin born?” requires two-hop reasoning: (1)
6

Query:Where was the director of the film Aylwin born?
Gold Answer:Weston-super-Mare
Naive RAG
Retrieved [1] Aylwin is a 1920 British silent drama film directed by Henry Edwards...
[2] Jim Wynorski (born August 14, 1950) is an American screenwriter...
[3] Frank Launder (28 January 1906 – 23 February 1997) was a British writer, film director...
Generated✗Henry Edwards
IndexRAG
Retrieved [1] Aylwin is a 1920 British silent drama film directed by Henry Edwards...(AKU)
[2] Henry Edwards directed both Aylwin and In the Soup, showcasing his career...(bridging fact)
[3]Henry Edwards, born on 18 September 1882 in Weston-super-Mare...(bridging fact)
Generated✓Weston-super-Mare
Table 6: Case study from 2WikiMultiHopQA. Naive RAG retrieves the correct film passage but fails to retrieve the
passage containing the director’s biographical information within its top- kresults, producing an incorrect answer
(“Henry Edwards”). IndexRAG retrieves two bridging facts within its top- kthat directly connect the film to the
director’s career and birthplace, producing the correct answer (“Weston-super-Mare”). The key bridging fact is
highlighted in green . For brevity, only the top-3 results are shown.
identifying the director of the film Aylwin as Henry
Edwards, and (2) determining his birthplace. The
relevant information is spread across two docu-
ments, requiring cross-document reasoning to con-
nect the film to the director’s biographical details.
For both methods, we present the top-3 retrieved
results. Naive RAG retrieves the correct film pas-
sage as its top-ranked result, which identifies Henry
Edwards as the director of Aylwin. However, the
remaining two passages contain unrelated directors.
The passage containing Edwards’ birthplace exists
in the vector store but falls outside the top- k, caus-
ing the LLM to output the director’s name rather
than his birthplace.
IndexRAG retrieves the same film passage as
its top-ranked result. However, the remaining two
slots are filled by bridging facts generated during
offline indexing: one connecting Edwards to his
directing career and another linking him to his
birthplace in Weston-super-Mare. With this cross-
document context directly available, the LLM cor-
rectly answers the query.
This example demonstrates that bridging facts
do not merely summarize existing content but cre-
ate new retrievable units that connect information
across document boundaries, making previously
unreachable evidence directly accessible through
standard vector search.
5.3 Ablation Study
5.3.1 Effect of Each Stage
Stage 1: Extraction Method.Table 7 compares
three Stage 1 strategies on HotpotQA without bridg-
ing facts. Naive chunking yields the lowest perfor-
mance with an F1 of 63.6. Summarization providesMethod EM Acc F1
Chunking 50.2 54.0 63.6
Summary 51.7 55.4 66.0
QA extraction 53.2 57.4 67.2
Table 7: Stage 1 extraction method comparison (%) on
HotpotQA without bridging facts.
Configuration EM Acc F1
Naive RAG 19.0 20.4 29.9
+ Stage 222.3 24.4 34.4
QA Extraction 18.1 20.1 30.1
+ Stage 222.4 24.4 34.4
Table 8: Stage 2 independence verification (%) on
MuSiQue.
a modest improvement (+2.4) over naive chunking
by compressing each document before indexing.
QA extraction achieves the best results across all
metrics, with an F1 improvement of 3.6 over chunk-
ing and 1.2 over summarization, suggesting that
structuring document content as question-answer
pairs produces denser and more query-aligned re-
trieval units, as each AKU directly encodes a piece
of answerable information rather than raw docu-
ment text. QA extraction is therefore adopted as
our Stage 1 strategy in all subsequent experiments.
Stage 2: Independence from Stage 1.Table 8
verifies that Stage 2 is independent of the Stage
1 extraction method. On MuSiQue, adding bridg-
ing facts to Naive RAG (EM 19.0 →22.3) yields
nearly identical gains to adding bridging facts to
QA extraction (EM 18.1 →22.4), confirming that
the bridging fact module is general-purpose and
7

Dataset Datastore Recall@10 EM
HotpotQAQA extraction 66.4 53.2
+ Bridging 57.554.1
2WikiQA extraction 50.5 41.7
+ Bridging 46.744.8
MuSiQueQA extraction 62.0 18.1
+ Bridging 52.822.4
Table 9: Recall@10 vs EM (%) with and without bridg-
ing facts.
Comp. Bridge. Composit. Infer.050100
78.2
63.3
17.220.679.6
63.5
22.9 24.3EM (%)w/o bridging facts
w/ bridging facts
Figure 3: Fine-grained EM (%) on 2WikiMultiHopQA
by question type. Results are shown with and without
bridging facts.
improves performance regardless of how the under-
lying documents are indexed. This independence
is practically important, as it allows Stage 2 to be
applied on top of any existing RAG system without
modifying the underlying indexing pipeline.
5.3.2 Recall vs. End-to-End Performance
Table 9 examines the relationship between passage
recall and QA performance when bridging facts
are added to the vector store. Across all three
datasets, adding bridging facts consistently reduces
Recall@10 yet improves EM, with the largest EM
gain on MuSiQue (+4.3), followed by 2WikiMul-
tiHopQA (+3.1) and HotpotQA (+0.9), suggest-
ing bridging facts are most beneficial for harder
datasets requiring more cross-document reasoning.
The recall drops because bridging facts compete
with original passages for retrieval slots. Since
bridging facts are newly generated by the LLM,
they do not correspond to any original passage and
therefore contribute no gain to passage-level re-
call. Despite this, the consistent EM improvement
across all three datasets suggests that the cross-
document reasoning encoded in bridging facts is
more valuable for answering multi-hop questions
than the original passages they replace.5.3.3 Performance Across Question Types
To further understand IndexRAG’s performance
on 2WikiMultiHopQA, we break down results by
question type. Figure 3 shows results across four
question types (comparison, bridge comparison,
compositional, and inference). The largest gains
from bridging facts appear on compositional (+5.7)
and inference (+3.7) questions, both of which re-
quire combining evidence from separate documents
to derive an answer. This aligns with the design
of bridging facts, which explicitly encode cross-
document connections.
Comparison questions also benefit from bridging
facts (+1.4), as they often require retrieving infor-
mation of two entities from different documents.
Bridge comparison questions show minimal im-
provement (+0.2). These questions require parallel
two-hop reasoning: each hop independently identi-
fies a piece of information (e.g., birthdate) of a dif-
ferent entity before comparing them. For instance,
“Which film has the director born later, X or Y?” re-
quires resolving the director and birthdate for both
X and Y separately. Since our bridging facts are
generated along sequential reasoning paths, they
tend to cover only one of the two parallel paths,
providing limited benefit for this question type.
These results suggest that bridging facts are most
effective for question types that require synthesiz-
ing evidence across documents, such as compo-
sitional and inference questions. The relatively
smaller gains on comparison and bridge compar-
ison questions, which together account for 44%
of the dataset, partially explain IndexRAG’s lower
overall performance on 2WikiMultiHopQA com-
pared to other benchmarks.
6 Conclusion
We presentIndexRAG, a novel approach that shifts
cross-document reasoning from online inference
to offline indexing. During indexing, IndexRAG
generates bridging facts that capture entity-level
connections across documents, stored alongside
original passages in a unified vector store. At infer-
ence time, a single retrieval pass and a single LLM
call suffice to produce the final answer. Extensive
experiments on three multi-hop QA benchmarks
show that IndexRAG achieves the best average F1
among single-LLM-call methods, outperforming
the strongest baseline by 2.3 F1 points. Future work
could explore question-type-aware bridging fact
generation to handle diverse reasoning patterns.
8

Limitations
While IndexRAG demonstrates strong perfor-
mance, several limitations remain. First, bridg-
ing fact quality depends on the LLM used during
offline indexing, and noisy or hallucinated facts
may introduce irrelevant retrieval units and hurt
performance. Second, bridge entities are currently
extracted by the LLM directly, which may miss
entities or introduce noise. Using a dedicated NER
model could potentially improve extraction preci-
sion. Third, evaluation is limited to English multi-
hop QA benchmarks, and generalization to other
languages or domains remains unexplored.
References
Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, and 1 others. 2020. Language models are
few-shot learners.Advances in neural information
processing systems, 33:1877–1901.
Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu,
Kaixin Ma, Xinran Zhao, Hongming Zhang, and
Dong Yu. 2024. Dense x retrieval: What retrieval
granularity should we use? InProceedings of the
2024 Conference on Empirical Methods in Natural
Language Processing, pages 15159–15177.
Alexey Dosovitskiy, Lucas Beyer, Alexander
Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
Thomas Unterthiner, Mostafa Dehghani, Matthias
Minderer, Georg Heigold, Sylvain Gelly, and 1
others. 2020. An image is worth 16x16 words:
Transformers for image recognition at scale.arXiv
preprint arXiv:2010.11929.
Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff
Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré,
Maria Lomeli, Lucas Hosseini, and Hervé Jégou.
2025. The faiss library.IEEE Transactions on Big
Data.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023. Precise zero-shot dense retrieval without rel-
evance labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 1762–1777.
Bernal J Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Ya-
sunaga, and Yu Su. 2024. Hipporag: Neurobiologi-
cally inspired long-term memory for large languagemodels.Advances in neural information processing
systems, 37:59532–59569.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning
steps. InProceedings of the 28th International Con-
ference on Computational Linguistics, pages 6609–
6625.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and 1 oth-
ers. 2025. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions.ACM Transactions on Information
Systems, 43(2):1–55.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 conference on empirical methods in natural
language processing (EMNLP), pages 6769–6781.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A Smith, and Mike Lewis. 2023. Measuring
and narrowing the compositionality gap in language
models. InFindings of the Association for Computa-
tional Linguistics: EMNLP 2023, pages 5687–5711.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sas-
try, Amanda Askell, Pamela Mishkin, Jack Clark, and
1 others. 2021. Learning transferable visual models
from natural language supervision. InInternational
conference on machine learning, pages 8748–8763.
PmLR.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. Squad: 100,000+ questions for
machine comprehension of text. InProceedings of
the 2016 conference on empirical methods in natural
language processing, pages 2383–2392.
Stephen E Robertson and Steve Walker. 1994. Some
simple effective approximations to the 2-poisson
model for probabilistic weighted retrieval. InSI-
GIR’94: Proceedings of the Seventeenth Annual In-
ternational ACM-SIGIR Conference on Research and
Development in Information Retrieval, organised by
Dublin City University, pages 232–241. Springer.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D Manning.
2024. Raptor: Recursive abstractive processing for
tree-organized retrieval. InThe Twelfth International
Conference on Learning Representations.
9

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. Musique: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics, 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st annual meeting of the association for com-
putational linguistics (volume 1: long papers), pages
10014–10037.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all
you need.Advances in neural information processing
systems, 30.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 conference on empiri-
cal methods in natural language processing, pages
2369–2380.
Xiaozhou You, Yahui Luo, and Lihong Gu. 2025. Ms-
rag: Simple and effective multi-semantic retrieval-
augmented generation. InProceedings of the 2025
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 22620–22636.A Evaluation Metrics
Letˆaanda∗denote the normalized prediction and
gold answer, and let T(·) denote the token set of
a string. Exact Match (EM) measures whether the
prediction is identical to the gold answer:
EM=(
1ifˆa=a∗
0otherwise(3)
Accuracy (Acc) measures whether the gold an-
swer appears as a substring of the prediction:
Acc=(
1ifa∗⊆ˆa
0otherwise(4)
F1 computes the token-level harmonic mean of
precision and recall:
P=|T(ˆa)∩T(a∗)|
|T(ˆa)|, R=|T(ˆa)∩T(a∗)|
|T(a∗)|
(5)
F1=2PR
P+R(6)
B Balanced Context Selection
Algorithm 1 presents the full procedure for bal-
anced context selection, which controls the propor-
tion of bridging facts in the retrieved context.
Algorithm 1Balanced Context Selection
1:C← ∅,n b←0
2:forr jinRdo
3:if|C|=kthen
4:break
5:end if
6:ifr j∈ Aorn b< k bthen
7:C←C∪ {r j}
8:ifr j∈ Bthen
9:n b←n b+ 1
10:end if
11:end if
12:end for
13:returnC
C Effect of Balanced Context Selection
The parameter kbcontrols the maximum number of
bridging facts included in the retrieved context. Ta-
ble 10 shows results for kb∈ {0,1,2,3,5} , where
kb= 0 corresponds to retrieval without bridging
facts. Performance peaks at kb= 2on HotpotQA
10

kb HotpotQA 2Wiki MuSiQue
0 (no bridging) 53.2 41.7 18.1
1 52.7 43.3 21.6
254.143.2 22.4
3 53.944.8 22.4
5 52.9 44.4 22.3
Table 10: Balanced retrieval ablation (EM, %).
and MuSiQue (with MuSiQue tying at kb= 2and
kb= 3), and at kb= 3on 2WikiMultiHopQA, then
declines as kbincreases further. This is because
bridging facts are shorter than AKUs on average
(166 vs. 634 characters), so including too many
bridging facts displaces original context, which
contains more informative content. We observe
that datasets requiring more cross-document rea-
soning, such as MuSiQue, tolerate a higherk b.
D Prompt Templates
We present all prompt templates used in our ex-
periments. Figure 4 shows the Stage 1 AKU
extraction prompt, which instructs the LLM to
extract factual information from a document as
atomic question-answer pairs. Figure 5 shows the
summary-based Stage 1 baseline. The LLM is
asked to produce a comprehensive summary of
each document. For Stage 2, Figure 6 provides
documents sharing a common entity and prompts
the LLM to generate cross-document bridging facts
in JSON format. Figure 7 is the answer generation
prompt shared across all methods, requiring the
LLM to output concise, exact answers without ex-
planation. Figure 8 presents the IRCoT reasoning
prompt, which elicits step-by-step chain-of-thought
reasoning and guides the LLM to suggest follow-
up search queries when additional information is
needed.
11

You are an expert information extractor specializing in converting unstructured documents into clear, atomic question-answer
pairs.
Extract ALL factual information from the following document as question-answer pairs. Each pair must answer exactly one
question, be self-contained, and be verifiable from the source content. Extract questions for facts, descriptions, properties,
relationships, and events. For each entity mentioned, also extract questions about its relationships to other entities.
Document:{text}
Return only a valid JSON object without any other text.
Figure 4:Prompt template for Stage 1 AKU extraction.
Given the following document, write a comprehensive summary that captures all key facts, entities, relationships, and
details. Be thorough and do not omit important information.
Document:{text}
Summary:
Figure 5:Prompt template for the Stage 1 summary strategy.
Given the following information about “{entity}” from multiple source documents, generate bridging facts that connect
information across these documents.
{doc_sections}
Requirements:
- Each bridging fact must combine information from 2+ documents
- Be factually accurate — only connect information that is logically related
- Each fact should be self-contained and understandable without context
- Do not generate speculative connections
- If documents share the entity name but are about unrelated topics, return empty
Return a JSON array of strings. If no meaningful connections exist, return [].
Figure 6:Prompt template for Stage 2 bridging fact generation.
You are a precise question answering assistant. Answer with ONLY the exact information requested, with no explanations
or extra words. If the answer is a name, give only the name. If the answer is a number, give only the number. If the answer
is yes/no, give only yes or no.
Context:{context}
Question:{question}
Figure 7:Prompt template for answer generation (used by all methods).
You are a reasoning assistant that helps answer multi-hop questions step by step.
Question:{question}
Retrieved Information:{context}
Reasoning so far:{cot_so_far}
Write ONE brief reasoning sentence that makes progress toward answering the question. If more information is needed,
suggest a specific search query.
Format your response as:
Reasoning:〈one sentence of reasoning〉
Search:〈next search query, or DONE if ready to answer〉
Figure 8:Prompt template for IRCoT chain-of-thought reasoning, following Trivedi et al. (2023).
12
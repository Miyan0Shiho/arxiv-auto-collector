# GLM-RAG: Graph Language Models for Graph-Based Retrieval-Augmented Generation

**Authors**: Maya Arseven, Anette Frank, Beni Egressy, Johann Higl, Moritz Plenz

**Published**: 2026-07-30 15:49:34

**PDF URL**: [https://arxiv.org/pdf/2607.28397v1](https://arxiv.org/pdf/2607.28397v1)

## Abstract
Retrieval-augmented generation (RAG) over knowledge graphs requires retrievers that can effectively capture both graph structure and semantic information. Recent approaches have explored graph neural network (GNN)-based retrievers to model graph topology in multi-hop reasoning tasks. In parallel, graph language models (GLMs) have emerged as a promising paradigm that integrates graph reasoning and the semantic capabilities of language models. In this work, we introduce a GLM-based retriever and investigate the comparative strengths of GLM-based, GNN-based, and traditional vector-search-based retrievers in single- and multi-hop RAG settings, and with a particular focus on transferability to unseen domains. Our findings suggest that finetuned GLM retrievers generalize better out of domain, achieving SOTA on two multi-hop benchmarks. On in-domain multi-hop QA datasets they remain comparable to prior work, with promising scaling as parameters and subgraph coverage increase. GNN-based retrievers achieve higher graph coverage with an efficient training setup, whereas the vector-search baseline excels at single-hop datasets.

## Full Text


<!-- PDF content starts -->

GLM-RAG: Graph Language Models for Graph-Based
Retrieval-Augmented Generation
Maya Arseven
 Anette Frank
 Beni Egressy⋆
Johann Higl⋆
Moritz Plenz⋆
Institute of Computational Linguistics, Heidelberg University
Aleph Alpha Research
{arseven, frank, plenz}@cl.uni-heidelberg.de
{maya.arseven, johann.higl, beni.egressy}@aleph-alpha-research.com
Abstract
Retrieval-augmented generation (RAG) over
knowledge graphs requires retrievers that can
effectively capture both graph structure and se-
mantic information. Recent approaches have
explored graph neural network (GNN)-based
retrievers to model graph topology in multi-
hop reasoning tasks. In parallel, graph lan-
guage models (GLMs) have emerged as a
promising paradigm that integrates graph rea-
soning and the semantic capabilities of lan-
guage models. In this work, we introduce a
GLM-based retriever and investigate the com-
parative strengths of GLM-based, GNN-based,
and traditional vector-search-based retrievers in
single- and multi-hop RAG settings, and with
a particular focus on transferability to unseen
domains. Our findings suggest that finetuned
GLM retrievers generalize better out of domain,
achieving SOTA on two multi-hop benchmarks.
On in-domain multi-hop QA datasets they re-
main comparable to prior work, with promising
scaling as parameters and subgraph coverage
increase. GNN-based retrievers achieve higher
graph coverage with an efficient training setup,
whereas the vector-search baseline excels at
single-hop datasets.1
1 Introduction
Language models (LMs) struggle with hallucina-
tion in many downstream applications (Zhang et al.,
2023) and lack domain-specific knowledge in spe-
cialized domains.
Retrieval-augmented generation (RAG) ad-
dresses these challenges by retrieving relevant doc-
uments and providing them as additional context
(Lewis et al., 2020). Thus, domain-specific knowl-
edge can be integrated into an LM’s reasoning
process, and hallucinations can be reduced by
grounding the LM’s answer in relevant facts. Al-
though RAG models show strong results in many
⋆Shared last authorship
1Code, models, and data will be released soon, contact us
for any questions in the meantime.
Figure 1: GLM-RAG and GFM-RAG knowledge graph
retrieval traces for the query “What is the three letter
abbreviation for the country, which maintains border
troops, and claims Bernd Baumgart as a citizen?” with
“GDR” being the golden answer. Example chosen from
the test set of MuSiQue.
knowledge-intensive tasks, they lag behind in tasks
requiring multi-hop reasoning across multiple doc-
uments (Tang and Yang, 2024).
Graphs can naturally capture relationships be-
tween documents.Unlike vector search, graph-
based retrieval can traverse multiple relation-
ships to incorporate multi-dimensional retrieval
chains, enabling stronger cross-document interac-
tions. Hence, graph RAG methods have been intro-
duced to combine the strengths of graph structures
and RAG models (Edge et al., 2024).
Current graph RAG methods (Gutiérrez et al.,
2024; He et al., 2024; Luo et al., 2025) typically
rely on shallow semantic integration, such as sen-
arXiv:2607.28397v1  [cs.AI]  30 Jul 2026

tence embeddings, and are unable to support end-to-
end training in a unified graph-text model. Figure 1
shows a knowledge graph and the retrieval chain of
graph RAG models for a complex query:“What is
the three letter abbreviation for the country, which
maintains border troops, and claims Bernd Baum-
gart as a citizen?”.
GFM-RAG (see details in §4.2.1) does not utilize
node labels. Consequently, it treats “East Germany
(GDR)” and “1976 Summer Olympics” as equally
relevant neighbors of “Bernd Baumgart”: both are
direct neighbors of “Bernd Baumgart” and con-
nected via similar relations (“competed for” and
“competed in”). By relying on these structural sig-
nals, GFM-RAG retrieves irrelevant entities such
as “Bruce Jenner” and “Kourtney Kardashian”.
Considering such shortcomings, we see great
potential in unleashing the semantic meaning of
text-attributed graphs, by using a model that has
a good understanding of the meaning of text
and can, at the same time, operate on graphs.
Following this motivation, we proposeGLM-
RAG:GraphLanguageModels for Graph-based
Retrieval-AugmentedGeneration. Building on
GFM-RAG (Luo et al., 2025), we replace their
GNN and sentence-embedding-based retriever with
an end-to-end trainable Graph Language Model
(GLM; Plenz and Frank, 2024). GLMs adapt pre-
trained language models into graph transformers
while preserving their pretrained parameters, and
hence language understanding capabilities (see §3).
This allows GLM-RAG to natively process text-
attributed graphs on the token level, simultaneously
leveraging both structural information and the se-
mantic meaning of entities, relations, and queries.
In the above example (cf. also Fig. 1) , this enables
GLM-RAG to make an informed decision between
the two similar neighbors and successfully retrieve
the semantically relevant neighborhood.
In addition to introducing GLM-RAG, we sys-
tematically assess when graph-enhanced retrieval
methods offer benefits. We conduct comprehensive
experiments across a range of domains, enabling a
rigorous comparison between graph-based methods
and a vanilla vector-search approach.
Zero-shot results on out-of-domain (OOD)
single-hop datasets reveal that vanilla RAG is suf-
ficient for datasets that require only single-step
reasoning. This is in line with recent findings by
Xiang et al. (2025). However, in multi-hop set-
tings, we see that the graph-based models outper-form vanilla RAG. Furthermore, results on OOD
datasets highlight GLM-RAG’s strong transferabil-
ity performance, outperforming GFM-RAG in all
settings and achieving new SOTA results on Med-
ical and computer science G-bench (Xiang et al.,
2025; Xiao et al., 2025).
Beyond these advantages, our ablations reveal
that GLM-RAG’s performance scales well with
model size, while GFM-RAG plateaus. Combined
with the strong OOD performance, this shows
GLM-RAG’s merit as a general graph RAG foun-
dation model.
Lastly, our analysis of the ranking results re-
veals that GLM-based retrievers show particular
strengths on questions requiring deeper seman-
tic understanding, whereas GNN-based retrievers
profit from covering larger parts of the knowledge
graph.
Our main contributions are:
1.We proposeGLM-RAG, a new graph RAG
framework powered by a Graph LM-based re-
triever for stronger integration of text features
2.GLM-RAG shows strongzero-shotgeneral-
ization capabilities, surpassing comparable
graph foundation models
3.Ourcomparative resultsshow that vanilla
RAG is sufficient for single-hop retrieval
tasks, while graph-based methods excel in
multi-hopsettings
4.Ablation studies show that GLM-RAGscales
betterwith increasing model size compared
to GNN-based retrievers
5.Extensiveanalysisreveals that GNN-based
retrievers achieve broader graph coverage but
lack the strongersemantic understandingof
GLM-based retrievers
2 Previous Work
Retrieval-Augmented Generation (RAG).
RAG typically relies on dense vector search,
where text corpora are split into isolated chunks
and indexed via sentence embeddings (Reimers
and Gurevych, 2019). Once retrieved, relevant
documents are included in an LM’s context to help
answer questions (Lewis et al., 2020; Karpukhin
et al., 2020). While offering a cheaper alternative
to fine-tuning models, RAG models struggle with
multi-hop reasoning and complex queries that span
across separate texts.

Graph RAG.Graph RAG methods aim to solve
more complex multi-document tasks by encoding
informational dependencies within a graph. Edge
et al. (2024) frame this as a query-focused summa-
rization task, rather than an explicit retrieval task.
However, later works use GNN-based retrievers
to leverage the graph topology of KGs in multi-
hop QA tasks (Mavromatis and Karypis, 2025).
Most relevant to our work is GFM-RAG (Luo et al.,
2025), a “graph foundational model” powered by a
GNN-based retriever, designed to be generalizable
to unseen domains without finetuning. GFM-RAG
achieved SOTA results on in-domain datasets and
showed promising transferability capabilities be-
yond the training data. In recent work, Luo et al.
(2026) extended GFM-RAG by including more di-
verse information in the constructed KG, thus sur-
passing the previous SOTA.
In this work, we build on GFM-RAG, with the
objective of improving its transferability by using
a Graph LM (GLM) for retrieval (Plenz and Frank,
2024). Graph LMs are based on pretrained LLMs
and therefore combine a strong understanding of
(unseen) texts with graph processing capabilities.
Combining our approach with Luo et al. (2026) is
left for future work.
LMs on graphs.A common approach to encode
KGs is to linearize and process them with a LM
(Schmitt et al., 2020; Ribeiro et al., 2021; Li et al.,
2021; Gao et al., 2024; Yamada and Sugiyama,
2026). While capturing the text features well, such
approaches make limited use of the graph structure.
Hence, GNNs in combination with often static se-
mantic embeddings are employed (Lin et al., 2019;
Malaviya et al., 2020; Yasunaga et al., 2022; Zhao
et al., 2023). By slightly adjusting a LM’s architec-
ture, while maintaining its pretrained parameters, a
deeper interaction between text features and graph
structure can be enabled (Plenz and Frank, 2024;
Egressy and Stühmer, 2025; Gong et al., 2026;
Vajda, 2026). We build on this line of work, to
improve graph reasoning in graph RAG.
3 Preliminary: Graph Language Models
Graph Language Models (GLMs) bridge language
models and graph transformers by converting a pre-
trained LM into a graph transformer (Plenz and
Frank, 2024). This design enables the model to
leverage pretrained language understanding from
the LM while incorporating structural graph rea-
soning through the architectural design.This design makes GLMs ideal for processing
text-attributed graphs, such as knowledge graphs.
The method involves two main steps. First, the
knowledge (sub)graph undergoes preprocessing:
edge and node labels are tokenized individually and
connected according to the original graph structure.
This transformation allows each triplet to be rep-
resented as a token sequence resembling natural
text, while still sharing node tokens across multiple
triplets, as in a standard graph structure. Second,
the LM’s self-attention mechanism is modified to
use relative positional encodings that capture dis-
tances between token pairs within the same triple,
thereby enabling the model toread triplets as se-
quential text– just as a language model would.
While these attention patterns, which are typical
for LMs, are applied to triplets, the GLM also uses
GNN-like attention patterns, which capture the
overall graph structure. Together, these adaptations
yield a model that is able to read a labeled graph
much like contiguous language, thereby achieving
strong (i) language understanding abilities and si-
multaneously, (ii) graph-based reasoning abilities
that are well-suited for graph RAG. We refer to
the original publications for more details on the
method as well as experimental validation.
4 GLM-RAG
Building on the ability of the GLM to jointly reason
over textandgraph-structured data, we propose
GLM-RAG, a RAG architecture that relies on a
GLM-based retriever.
We hypothesize that a GLM-based retriever has
better retrieval quality and subsequently QA per-
formance, due to its better understanding of text
features in the graph, compared to a GNN-based
retriever, which can only utilize the graph structure
with shallow text integration.
We build on the GFM-RAG framework of Luo
et al. (2025), adopting their KG and QA dataset
construction stages, as well as their document rank-
ing and answer generation mechanisms. Our con-
tribution is aGLM-based retriever(§4.2.2) that
replaces the original KG retrieval component. Cru-
cially, this setup yields a controlled environment
that allows us to assess a GLM-based versus a
GNN-based retriever. Please refer to App. B for
more implementation details.

[ 0][sent-emb] 
 Q: Where is EMNLP 2026 ? 
(EMNLP 2026) 
[Query Emb ] [ 0][ 0][ 0]
[ 0][ 0]
is-in 
[sent-emb] is-a 
[sent-emb] capital-of 
[sent-emb] neighbour-of 
[sent-emb] 
is-a 
[sent-emb] is-in 
[sent-emb] Romania Athens 
Graph LM 
Buda pest EMNLP2026 NLPConference Hungary ACL2027 
capital of isaisinneighbour ofWhere isEMNLP2026 ?GLM-based  Retriever GNN-based  Retriever 
isin isa
Buda pest
avg. , P(“ Budapest ”, Q) GNN 
, P(“ Budapest ”, Q) (Budapest) 
Seed node 
Visible nodes 
Invisible nodes  Query  Query Figure 2: Differences between how GNN- and GLM-based retrievers reason over the KG for the query “Where is
EMNLP 2026?”. GLM-based retriever (left) tokenizes the nodes, relations, and the query with the Graph LM’s
encoder. The tokenized input is passed through the graph LM to produce embeddings for each token. The token
embeddings corresponding to each node are aggregated to produce the final node representation, which is used for
computing the relevance scores of each entity in the extracted subgraph. GNN-based retriever (right) calculates the
query and relation embeddings via a sentence embedding model. Only the seed entit(ies) get initialized with the
query embedding and the rest of the nodes with the zero vector. The GNN iteratively propagates messages along
edges using message passing over the full graph, producing a score distribution over all nodes.
4.1 KG & QA Dataset Construction
The pipeline starts from a document corpus, and
constructs a corresponding knowledge graph on
entity-level, via an LM-based named entity recog-
nizer (NER), so that a document such as “EMNLP
2026 is taking place in Budapest, the capital
of Hungary.” is represented through the tuples
[(EMNLP_2026, is_in, Budapest),
(Budapest, capital_of, Hungary)]
within the KG.
The queries undergo a similar transformation,
so that a question “Where is EMNLP 2026?” is
represented through a seed entity “ EMNLP_2026 ”
as a starting point within the KG. There may be
multiple seed entities depending on the complexity
of the query.
4.2 KG Retrieval
4.2.1 GNN-based Retriever
The main building block of GFM-RAG is the query-
dependent GNN retriever, whose job it is to rank
the entities of the constructed KG by relevance to a
given input query. The message passing paradigm
(Gilmer et al., 2017) is used to propagate informa-
tion across the graph (see right side of Fig. 2).
While seed entities are initialized with the queryembeddings, all other nodes are initialized with
zero vectors. By contrast, relations are initialized
with their sentence embeddings. Consequently, the
resulting node representations from the GNN are
driven by their relative graph distance to the seed
entities rather than their semantic features. Node
texts are used exclusively to identify seed entities
and play no further role in the GNN retriever. Given
this shortcoming, we also explore a variant of GFM-
RAG, GFM-RAG+, where all nodes are initialized
with their sentence embeddings (see Sec. 5.1).
At the final layer, the per-node representations
are concatenated with the query embedding and
scored by an MLP to find the most relevant nodes.
Luo et al. (2025) pretrain the GNN retriever on
KG completion and finetune it on three Wikipedia-
based multi-hop QA datasets.
4.2.2 GLM-based Retriever
Our proposed GLM-based retriever aims for a
graph RAG approach that maximizes the potential
of text-attributed graphs.
The inner workings of the GLM retriever funda-
mentally differ from those of the GNN-based one.
Most notably, the GNN-based retriever does not
make use of the rich textual features in the graph
and instead relies on the query and the raw graph

structure, whereas our GLM retriever fully utilizes
the semantics in the graph, through the GLM’s text
encoding abilities (§3).
The GLM retriever extracts a local subgraph
around the seed entities, converts it into a sequence
of textual triplets, and encodes the individually to-
kenized edge and node representation through a
graph encoder with structure-aware relative posi-
tions (see left side of Fig. 2). Nodes and relations
are tokenized and embedded using the underlying
LM’s own tokenizer and token embedding layer.
The tokens are passed through the LM’s layers,
using attention masking to encode the graph struc-
ture. This stands in contrast to the GNN retriever,
where only the seed entities are initialized with
non-zero embeddings. As a result, the query-graph
interaction happens at a much earlier stage, namely
when the question text gets fused with the extracted
subgraph structure inside the GLM’s first atten-
tion layer. The resulting per-entity embeddings are
element-wise multiplied with a projected question
vector and passed through a scoring head.
We finetune the GLM-based retriever on three
Wikipedia-based multi-hop QA datasets (§5).
4.3 Ranking & Answer Generation
After the ranking is completed, the most relevant
entities are mapped back to their original docu-
ments, thereby converting the entity ranking to a
document ranking. Finally, an LM generates the
answer from the query and the top-kdocuments.
5 Experiments
In this section, we introduce our experimental setup
and report the results of the following experiments:
1.We finetune GLM- and GNN-based retriev-
ers on three Wikipedia-based multi-hop question-
answering (QA) datasets and evaluate retrieval
(§5.2) and QA performance (§5.3).
2.We test the finetuned GLM-based retrievers
transferability on 11 OOD datasets (§5.4).
Further ablation studies (§C) and a breakdown
of the retrieval performance (§D.1) per question
complexity can be found in the Appendix.
5.1 Experimental Setup
Models
We choose all-mpnet-base-v2 as the sen-
tence embedding model for GFM-RAG. Following
Plenz and Frank (2024), we initialize our GLM re-
triever with T5-large model weights, which we testin different sizes (as shown in Section 5.5). We use
gpt-4o-minias the LM for all text generation.
Datasets
We finetune the models on the train-splits of three
Wikipedia-based multi-hop QA datasets: Hot-
PotQA (Yang et al., 2018), 2WikiMultihopQA
(2Wiki) (Ho et al., 2020), and MuSiQue (Trivedi
et al., 2022).
Together, these splits yield 282k question-
document pairs for training2.
To evaluate the transferability capabilities of
the GLM retriever, we follow Luo et al. (2025),
who chose seven transferability datasets from var-
ious domains: TechQA (Castelli et al., 2020), Ex-
pertQA (Malaviya et al., 2024), eManual (Nandy
et al., 2021), DelucionQA (Sadat et al., 2023)
have a customer support focus, MS MARCO (Ba-
jaj et al., 2018) and HAGRID (Kamalloo et al.,
2023) contain general knowledge questions, while
PubmedQA (Jin et al., 2019) requires biomedical
knowledge. However, since these datasets do not
require multi-hop reasoning, they can be solved
effectively using vanilla RAG. Therefore, we also
include several multihop benchmarks:Multihop-
RAG(Tang and Yang, 2024), a dataset based on
English news articles, G-Bench Novel and Medical
(Xiang et al., 2025) and G-Bench Computer Sci-
ence (CS) (Xiao et al., 2025). Please refer to App.
A for dataset statistics.
Baselines
To contextualize the results of our proposed GLM-
RAG method, we compare to four baselines:
a.Avanilla RAGbaseline encodes the
documents as well as the queries with the
all-mpnet-base-v2 embedding model, and
retrieves the top- 5documents according to their
cosine similarity. This baseline tests the need for
graph RAG methods.
b.As a strong comparative baseline, we use
the latest release ofGFM-RAGon HuggingFace,
which uses an identical graph RAG pipeline.
c.Afinetuned-onlyversion of GFM-RAG that
we refer to asGFM-RAG*. We opt for a finetuned-
only version of GFM-RAG to test the effectiveness
of the pretraining stage. This is a more comparable
baseline to our GLM-RAG, considering we also
only perform finetuning.
2In contrast, Luo et al. (2025) use 60k Q-Doc pairs for
their results. However, in their follow-up work, Luo et al.
(2026) publish new results trained with 282k Q-Doc pairs.

d.A finetuned-only version of GFM-RAG
where nodes are initialized with text embeddings
rather than zero vectors. We refer to this asGFM-
RAG+. Compared to GLM-RAG, this approach
also utilizes node semantics, though only via static
sentence embeddings.
e.For QA performance we additionally test ano-
context baselinethat evaluates the LM’s capability
to answer questions without any retrieved context.
Metrics
We use Recall@2 at the document level to mea-
sure retrieval performance. We evaluate document-
level rather than entity-level performance, since
the correct retrieval of documents directly affects
the downstream QA performance. For QA perfor-
mance, we report exact match (EM). All metrics
are established in prior work, enabling a direct com-
parison. We also report Recall@5 in Section D and
F1 scores in Section E. For G-Bench, we follow
standard evaluation practice for each benchmark
and report, evidence recall for retrieval and accu-
racy (answer correctness and answer score) for QA
performance. We use gpt-4o-mini for all LLM-
as-a-judge calls. Following prior work we report
one run per experiment due to computational con-
straints and provide significance tests in Section H.
5.2 Retrieval Performance
Figure 3 shows retrieval performance on the test
sets of the three Wikipedia-based datasets. All
graph-enhanced methods outperform the RAG
baseline, highlighting the contribution of graph
RAG methods in multi-hop tasks. GFM-RAG and
GFM-RAG* (i.e., with and without pretraining)
show near-identical performance on most datasets,
calling into question the need for costly pretraining.
Hence, we also only finetune our GLM retriever.
Overall, GFM-RAG+ and GLM-RAG perform
the best, showing the advantage of utilizing node
semantics. On HotPotQA and 2Wiki, GFM-RAG+
slightly outperforms GLM-RAG, while GLM-RAG
achieves the best result on MuSiQue. This confirms
GLM-RAG’s competitive in-domain performance,
despite being restricted to smaller subgraphs.
Table 17 shows the performance of additional
graph RAG methods, including new SOTA results
by Luo et al. (2026) in concurrent work. Their main
innovation is to extend GFM-RAG by including
more information in the KG-indexing step. This
is orthogonal to our work and is compatible with
our proposed GLM-Retriever. Combining bothapproaches is beyond the scope of this paper and is
left for future work.
MuSiQue HotPotQA 2Wiki020406080100Recall@2 Score (%)33.259.4
48.5
43.973.179.0
43.772.779.0
45.676.979.6
46.474.878.6RAG Baseline
GFM-RAG*
GFM-RAG
GFM-RAG+
GLM-RAG (ours)
Figure 3: Retrieval performance (Recall@2) on in-
domain Wikipedia datasets.
5.3 QA Performance
A similar trend can be observed in downstream QA
performance in Figure 4. The no-context and RAG
baselines get the lowest scores, showing the value
of the retrieved context as well as graph-enhanced
retrievers. GFM-RAG and GFM-RAG* perform
comparably across all three datasets, with a maxi-
mum difference of 1.7 points on HotPotQA, con-
firming that the effect of pretraining is negligible.
The overall performance trends of GLM-RAG
and the baselines are comparable to the results in
retrieval. The only difference is that GLM-RAG
achieves the best EM scores on HotPotQA. QA
results of further methods are shown in Table 20.
MuSiQue HotPotQA 2Wiki020406080Exact Match Score (%)
10.825.824.1 23.943.7
40.9
29.053.467.9
28.755.167.8
31.655.369.0
32.155.767.5No Context
RAG Baseline
GFM-RAG*
GFM-RAG
GFM-RAG+
GLM-RAG (ours)
Figure 4: QA performance (exact match) on in-domain
Wikipedia datasets.
5.4 Transferability
While GLM-RAG’s in-domain performance is not
significantly better than GFM-RAG’s, transferabil-
ity experiments reveal a significantly better gener-
alization to OOD datasets.
In the following, we first train our models on
only one of the Wikipedia datasets, and examine
its transfer to the remaining Wikipedia datasets
(§5.4.1). Then we test the fully-trained models

T echQAPubMedQA MS MARCOHAGRID ExpertQA eManualDelucionQA020406080100Recall@2 Score (%)35.037.0
26.064.8
40.356.5
45.2
29.836.4
26.262.9
38.548.8
40.8
32.039.3
28.468.1
46.243.7 44.9
40.644.9
29.569.6
44.557.7
49.145.959.9
36.778.6
60.772.469.7RAG Baseline GFM-RAG* GFM-RAG GFM-RAG+ GLM-RAG (ours)Figure 5: Retrieval performance (Recall@2) on single-hop out-of-domain datasets.
on seven single-hop OOD datasets (§5.4), and a
multi-hop OOD dataset (§5.4.3).
Table 1: Generalization abilities of GLM and GFM-
RAG(*,+) when finetuned on one Wikipedia dataset.
Recall@2
Train Set Test Set GLM-RAG GFM-RAG* GFM-RAG+
MuSiQueHotpotQA 65.759.4 60.3
MuSiQue 44.640.8 41.7
2Wiki 74.071.3 72.6
HotpotQAHotpotQA 74.2 73.577.0
MuSiQue 45.442.6 43.6
2Wiki 76.475.6 75.7
2WikiHotpotQA 64.357.7 57.8
MuSiQue 43.339.8 39.1
2Wiki 77.7 78.979.6
5.4.1 Transferability on Wikipedia Datasets
We train GLM-RAG, GFM-RAG* and GFM-
RAG+ on each Wikipedia dataset and investigate
transferability to the other two Wikipedia datasets.
Table 1 shows that in all cases, GLM performs
better inout-of-domainsettings compared to the
GFM-RAG variants, even surpassing GFM-RAG+
(i.e. GFM-RAG with additional node embeddings).
For example, when trained on HotpotQA, GLM
outperforms GFM-RAG* by 2.8 points and GFM-
RAG+ by 1.8 points on MuSiQue.
On MuSiQue, GLM-RAG is consistently better
than both models, whereas GFM-RAG variants per-
form betterin-domainon 2Wiki and HotpotQA.
5.4.2 Transferability on Single-hop Datasets
Motivated by previous results, we test the mod-
els’ generalization abilities to unseen domains on 7
single-hop datasets.
Figure 5 shows that GLM-RAG consistently out-
performs GFM-RAG models, except for Recall@2
on ExpertQA. Considering that GFM-RAG is a
foundation model trained to generalize well, this
demonstrates the strong transferability capabilitiesof GLM-RAG. We hypothesize that this is largely
enabled through GLM’s deep integration of the
KG’s rich semantics.
Still, it is important to note that a vanillaRAG
baselineoutperforms all tested graph RAG models
in all datasets. While this may seem surprising,
it is due to the nature of the datasets: Single-hop
datasets host direct questions that do not require
multi-hop reasoning capabilities, so the RAG’s
“simpler” similarity metric is often sufficient to find
relevant documents. Graph RAG methods, by con-
trast, show their strengths in multi-hop datasets,
where retrieving documents that are similar to the
question is not sufficient to answer compositional
questions.
5.4.3 Transferability on Multi-hop Datasets
We challenge the models by testing their retrieval
and QA performance for zero-shot transferability
on MultihopRAG (Tang and Yang, 2024), G-Bench
Novel, G-Bench Medical (Xiang et al., 2025), and
G-Bench CS (Xiao et al., 2025), all requiring multi-
hop reasoning. Here, we also include the recently
introduced SOTA method, G-Reasoner (Luo et al.,
2026).
Table 2 shows the retrieval and QA results for
all methods. GLM-RAG’s retrieval performance
excels in this multi-hop zero-shot setting, surpass-
ing the RAG baseline, GFM-RAG variants and
G-Reasoner by 20 points in Recall@2 on Multi-
hopRAG. The same holds for the three G-Bench
benchmarks, where GLM-RAG outperforms all
baselines including G-Reasoner. As of now, GLM-
RAG achieves SOTA results for the Medical3and
CS4benchmarks, and ranks second on the Novel3
benchmark. AutoPrunedRetriever (Wang et al.,
3https://graphrag-bench.github.io/
4https://deep-polyu.github.io/RAG/

Table 2: Retrieval (Recall@2 and evidence recall) and QA (accuracy) performance on multi-hop out-of-domain
datasets.
MultihopRAG G-Bench (Novel) G-Bench (Medical) G-Bench (CS)
MethodR@2 Recall ACC Recall ACC ACC
RAG 32.5 55.9 47.9 75.1 61.0 71.7
GFM-RAG 34.1 75.9 58.6 82.2 72.2 72.1
GFM-RAG+ 39.0 86.1 60.6 93.3 75.7 76.1
G-Reasoner 34.9 87.7 58.9 93.8 73.3 73.9
GLM-RAG60.0 88.0 61.9 94.6 76.9 76.6
2026), a graph RAG method that persists the mini-
mal reasoning subgraph built for earlier questions
and incrementally extends it for later ones, is the
current SOTA method for the Novel benchmark.
Integrating this pruning idea to GLM-RAG is an
extension that we leave out for future work.
This challenging test case underlines GLM-
RAG’s merit as a graph foundation model with
strong capabilities for zero-shot multi-hop settings.
To conclude the main experiments, our find-
ings suggest that even though GLM-RAG outper-
forms other graph-enhanced methods on the trans-
fer datasets, a vanilla RAG is sufficient in single-
step reasoning. However, on a multi-hop OOD
dataset, GLM-RAG clearly outperforms both the
vanilla RAG baseline and all GFM-RAG variants.
5.5 Ablations
We aim to show that GLM-RAG’s transferability
gains stem from its deeper integration of textual
features rather than input filtering or model scale.
To test this, the following section ablates the two
primary differences between GLM-RAG and GFM-
RAG variants: graph input and model size. For
these ablations we train all models on a smaller
subset of the training data with 60k instances.
Effect of Subgraph Restriction
One of the key differences between GLM-RAG
and GFM-RAG is the restricted graph visibility of
GLM-RAG due to context size limitations. Here
we test whether this restriction acts as a information
bottleneck or a noise filter for retrieval performance.
We evaluate this by adjusting the number of visi-
ble triplets in the 2-hop neighborhood subgraph in
GLM-RAG as well as GFM-RAG*.
Table 3 shows that GFM-RAG*’s performance is
negatively affected by this restriction. Moreover, as
we tighten the restriction for both models, the recalldrops monotonically. This supports our claim that
GLM-RAG’s restricted visibilitylimitsrather than
aids its performance, and that relaxing it would
improve results.
Table 3: Retrieval results (Avg. Recall@2) with re-
stricted GFM-RAG* and GLM-RAG. Recall results are
averaged over HotpotQA, MuSiQue and 2Wiki test sets.
GFM-RAG* GLM-RAG
unrestricted63.50-
600 triplets 60.3663.3
500 triplets 59.52 62.8
400 triplets 59.46 61.8
300 triplets 58.61 60.8
Effect of Model Size
One might also argue that the gains of GLM-RAG
is due to its higher parameter count, considering
that the GFM* and GLM retrievers have 8M and
336M parameters, respectively. To rule out this
hypothesis, we compare the models in acapacity-
matchedsetting by increasing the GNN-based re-
trievers hidden dimensions. This results in three
capacity matched pairs, shown in Table 18.
Figure 6 shows the scaling trend of GNN- and
GLM-based retrievers with increasing parameter
count. While GFM-RAG* doesn’t show any perfor-
mance gains from having more parameters, GFM-
RAG+’s upward trend only holds for in-domain
datasets (see also Figure 12 and Table 19). In con-
trast, GLM-RAG shows an upward trend in both
settings, proving that its transferability advantage
holds even when controlling for model size.
This finding is in line with work related to
GNN’s overfitting and oversmoothing problems (Li
et al., 2018; Rong et al., 2020; Oono and Suzuki,
2020; Chen et al., 2022), indicating that scaling is
lacking behind transformer based models (Kaplan
et al., 2020; Dehghani et al., 2023).

30M 100M 500M
Model Size (Parameters)525354555657585960Avg. Recall@2
5 Datasets (3 ID + 2 OOD)
GFM-RAG*
GFM-RAG+
GLM-RAGFigure 6: Model scaling comparison of GLM-RAG vs
capacity-matched GFM-RAG* and GFM-RAG+.
To conclude, we showed that GLM-RAG’s trans-
ferability claims still hold under matched visibil-
ity and capacity, implying that the generalizability
gains stem from utilizing the existing large-scale
pretrained LMs. The scalability potential of GLMs
is an additional benefit that strongly indicates po-
tential for further improvements from switching to
larger and more recent LMs as the GLM’s base.
6 Analysis
After evaluating the in- and out-of-domain retrieval
and QA quality of the compared methods, we fur-
ther investigate the strengths and weaknesses of
each graph-based retriever model. Although these
properties are not a proxy for better retrieval, we
analyze them to better understand the characteris-
tics of each model’s retrieval strategy. Our analysis
shows that GLM uses more semantic knowledge
to retrieve the right document, whereas GNN re-
lies more on structural signals. Appendix G shows
calculations for these measures.
6.1 Similarity Measures
First of all, we want to confirm our hypothesis that
the GLM-Retriever utilizes semantic features more
effectively than the other models. For this analysis
we compute thesemantic similaritybetween the
question and the retrieved answer entities. Table 4
shows that GLM-RAG and GFM-RAG+ consis-
tently retrieves more semantically similar entities
across 11 datasets. GLM’s strength is especially
highlighted in the multi-hop datasets, having better
semantic understanding of the constructed graphs.Table 4: Comparison of cosine similarity between the
question and the retrieved entities.
Models GLM-RAG GFM-RAG* GFM-RAG GFM-RAG+
Multi-Hop Datasets
MuSiQue 0.309 0.283 0.286 0.297
HotPotQA 0.312 0.292 0.293 0.298
2Wiki 0.285 0.262 0.265 0.268
MultiHopRAG 0.245 0.209 0.212 0.241
Single-Hop Datasets
TechQA 0.255 0.234 0.228 0.245
PubMedQA 0.336 0.298 0.289 0.314
MS MARCO 0.294 0.280 0.280 0.314
HAGRID 0.345 0.329 0.323 0.355
ExpertQA 0.284 0.249 0.235 0.286
eManual 0.276 0.234 0.221 0.286
DelucionQA 0.294 0.287 0.286 0.313
6.2 Distance Measures
To compare thegraph coveragefor both models,
we find the shortest path from each retrieved en-
tity to any question entity, aggregated per-sample
and per-dataset. The results in Table 5 show that
the GNN-based retrievers reach further on average
compared to the GLM-based retriever. Although
this is expected due to their architectural differ-
ences, it shows that a strength of GNN-based re-
trievers lies in their wider coverage of the graph.
Table 5: Comparison of mean hop distance between the
seed entities and the retrieved entities.
Models GLM-RAG GFM-RAG* GFM-RAG GFM-RAG+
Multi-Hop Datasets
MuSiQue 1.09 1.22 1.29 1.34
HotpotQA 1.05 1.19 1.23 1.24
2Wiki 1.10 1.19 1.22 1.24
MultiHopRAG 0.88 1.02 0.99 1.24
Single-Hop Datasets
TechQA 1.06 1.05 1.17 1.10
PubMed 1.04 1.19 1.31 1.44
MS MARCO 1.25 1.26 1.39 1.49
HAGRID 1.32 1.32 1.42 1.56
ExpertQA 1.53 1.59 1.89 2.06
eManual 1.29 1.38 1.50 1.55
DelucionQA 1.31 1.35 1.56 1.49
7 Conclusion
In this paper, we present an alternative approach to
handle multi-hop questions in graph RAG settings
by training a GLM-based retriever. Our experi-
ments show that GLM-RAG is a more generaliz-
able method with better scalability, in line with
works on building a graph foundational model. By
integrating the semantic knowledge hosted in a text-
attributed graph, a GLM-based retriever can better
understand and select relevant documents, relying
more on assessing the similarity between document

entities and the question, whereas concurrent GNN-
based retrievers rely more on structural signals.
Our extensive comparison of vanilla RAG, GNN-
based and GLM-based retrievers reveal that: (i) for
single-hop questions, which do not require multi-
hop reasoning, a vanilla RAG baseline is sufficient,
(ii) for multi-hop questions a finetuned GFM-RAG
model initialized with node embeddings (GFM-
RAG+) shows competitive performance with GLM-
RAG, and (iii) for zero-shot multi-hop questions,
GLM-RAG shows consistently better transferabil-
ity capabilities compared to its GNN-based coun-
terparts.
Limitations
We build on GFM-RAG (Luo et al., 2025), which
was the SOTA at the time of our experiments. Re-
cently, Luo et al. (2026) extended GFM-RAG’s
KG indexing to capture more diverse information,
establishing a new SOTA. Although we do not com-
bine our GLM retriever with this indexing strategy,
we expect the two approaches to be complementary,
making their integration a promising direction for
future research.
We have tested our approach with LMs of up to
0.8B parameters, but based on our scaling experi-
ments, we see great potential in using much larger
base LMs, but this is left for future work.
Finally, being transformer-based, our GLM-
based retriever is computationally more demanding
than a GNN-based retriever. Consequently, we re-
strict the size of the processed subgraphs to enable
more efficient training. Future work could scale
up GLM-based retrievers to combine their strong
semantic capabilities with broader graph coverage.
Acknowledgments
We sincerely thank Fabien Benureau for all the
contributions he made to the project.
References
Aijun Bai, Rolf Jagerman, Zhen Qin, Le Yan, Pratyush
Kar, Bing-Rong Lin, Xuanhui Wang, Michael Ben-
dersky, and Marc Najork. 2023. Regression compat-
ible listwise objectives for calibrated ranking with
binary relevance.
Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng,
Jianfeng Gao, Xiaodong Liu, Rangan Majumder, An-
drew McNamara, Bhaskar Mitra, Tri Nguyen, Mir
Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary,
and Tong Wang. 2018. Ms marco: A human gener-
ated machine reading comprehension dataset.Vittorio Castelli, Rishav Chakravarti, Saswati Dana, An-
thony Ferritto, Radu Florian, Martin Franz, Dinesh
Garg, Dinesh Khandelwal, Scott McCarley, Michael
McCawley, Mohamed Nasr, Lin Pan, Cezar Pendus,
John Pitrelli, Saurabh Pujar, Salim Roukos, Andrzej
Sakrajda, Avi Sil, Rosario Uceda-Sosa, Todd Ward,
and Rong Zhang. 2020. The TechQA dataset. In
Proceedings of the 58th Annual Meeting of the Asso-
ciation for Computational Linguistics, pages 1269–
1278, Online. Association for Computational Linguis-
tics.
Tianlong Chen, Kaixiong Zhou, Keyu Duan, Wenqing
Zheng, Peihao Wang, Xia Hu, and Zhangyang Wang.
2022. Bag of tricks for training deeper graph neu-
ral networks: A comprehensive benchmark study.
IEEE Transactions on Pattern Analysis and Machine
Intelligence, 45(3):2769–2781.
Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Pi-
otr Padlewski, Jonathan Heek, Justin Gilmer, An-
dreas Peter Steiner, Mathilde Caron, Robert Geirhos,
Ibrahim Alabdulmohsin, Rodolphe Jenatton, Lucas
Beyer, Michael Tschannen, Anurag Arnab, Xiao
Wang, Carlos Riquelme Ruiz, Matthias Minderer,
Joan Puigcerver, Utku Evci, Manoj Kumar, Sjo-
erd Van Steenkiste, Gamaleldin Fathy Elsayed, Ar-
avindh Mahendran, Fisher Yu, Avital Oliver, Fantine
Huot, Jasmijn Bastings, Mark Collier, Alexey A. Grit-
senko, Vighnesh Birodkar, Cristina Nader Vasconce-
los, Yi Tay, Thomas Mensink, Alexander Kolesnikov,
Filip Pavetic, Dustin Tran, Thomas Kipf, Mario Lu-
cic, Xiaohua Zhai, Daniel Keysers, Jeremiah J. Harm-
sen, and Neil Houlsby. 2023. Scaling vision trans-
formers to 22 billion parameters. InProceedings of
the 40th International Conference on Machine Learn-
ing, volume 202 ofProceedings of Machine Learning
Research, pages 7480–7512. PMLR.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130.
Beni Egressy and Jan Stühmer. 2025. Set-llm: A
permutation-invariant llm.
Mikhail Galkin, Xinyu Yuan, Hesham Mostafa, Jian
Tang, and Zhaocheng Zhu. 2024. Towards founda-
tion models for knowledge graph reasoning. InThe
Twelfth International Conference on Learning Repre-
sentations.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gen-
eration for large language models: A survey.
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley,
Oriol Vinyals, and George E Dahl. 2017. Neural mes-
sage passing for quantum chemistry. InInternational
conference on machine learning, pages 1263–1272.
Pmlr.

Haisong Gong, Zhibo Liu, Qiang Liu, Shu Wu, and
Liang Wang. 2026. Nag: A unified native architec-
ture for encoder-free text-graph modeling in language
models.arXiv preprint arXiv:2601.22657.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2025. Lightrag: Simple and fast retrieval-
augmented generation.
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2024. Hipporag: Neu-
robiologically inspired long-term memory for large
language models. InThe Thirty-eighth Annual Con-
ference on Neural Information Processing Systems.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025. From rag to memory:
Non-parametric continual learning for large language
models.
Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh Chawla,
Thomas Laurent, Yann LeCun, Xavier Bresson, and
Bryan Hooi. 2024. G-retriever: Retrieval-augmented
generation for textual graph understanding and ques-
tion answering.Advances in Neural Information
Processing Systems, 37:132876–132907.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-
hop QA dataset for comprehensive evaluation of
reasoning steps. InProceedings of the 28th Inter-
national Conference on Computational Linguistics,
pages 6609–6625, Barcelona, Spain (Online). Inter-
national Committee on Computational Linguistics.
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William
Cohen, and Xinghua Lu. 2019. PubMedQA: A
dataset for biomedical research question answering.
InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the
9th International Joint Conference on Natural Lan-
guage Processing (EMNLP-IJCNLP), pages 2567–
2577, Hong Kong, China. Association for Computa-
tional Linguistics.
Ehsan Kamalloo, Aref Jafari, Xinyu Zhang, Nan-
dan Thakur, and Jimmy Lin. 2023. Hagrid:
A human-llm collaborative dataset for generative
information-seeking with attribution.arXiv preprint
arXiv:2307.16883.
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B
Brown, Benjamin Chess, Rewon Child, Scott Gray,
Alec Radford, Jeffrey Wu, and Dario Amodei. 2020.
Scaling laws for neural language models.arXiv
preprint arXiv:2001.08361.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781,
Online. Association for Computational Linguistics.Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InProceedings of the 34th Inter-
national Conference on Neural Information Process-
ing Systems, NIPS’20, Red Hook, NY , USA. Curran
Associates Inc.
Junyi Li, Tianyi Tang, Wayne Xin Zhao, Zhicheng Wei,
Nicholas Jing Yuan, and Ji-Rong Wen. 2021. Few-
shot Knowledge Graph-to-Text Generation with Pre-
trained Language Models. InACL Findings.
Mufei Li, Siqi Miao, and Pan Li. 2025. Simple is effec-
tive: The roles of graphs and large language models
in knowledge-graph-based retrieval-augmented gen-
eration. InThe Thirteenth International Conference
on Learning Representations.
Qimai Li, Zhichao Han, and Xiao-ming Wu. 2018.
Deeper insights into graph convolutional networks
for semi-supervised learning.Proceedings of the
AAAI Conference on Artificial Intelligence, 32(1).
Lei Liang, Zhongpu Bo, Zhengke Gui, Zhongshu Zhu,
Ling Zhong, Peilong Zhao, Mengshu Sun, Zhiqiang
Zhang, Jun Zhou, Wenguang Chen, Wen Zhang, and
Huajun Chen. 2025. Kag: Boosting llms in profes-
sional domains via knowledge augmented generation.
InCompanion Proceedings of the ACM on Web Con-
ference 2025, WWW ’25, page 334–343, New York,
NY , USA. Association for Computing Machinery.
Bill Yuchen Lin, Xinyue Chen, Jamin Chen, and Xiang
Ren. 2019. KagNet: Knowledge-aware graph net-
works for commonsense reasoning. InProceedings
of the 2019 Conference on Empirical Methods in Nat-
ural Language Processing and the 9th International
Joint Conference on Natural Language Processing
(EMNLP-IJCNLP), pages 2829–2839, Hong Kong,
China. Association for Computational Linguistics.
Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Dinh
Phung, Chen Gong, and Shirui Pan. 2025. GFM-
RAG: Graph foundation model for retrieval aug-
mented generation. InThe Thirty-ninth Annual Con-
ference on Neural Information Processing Systems.
Linhao Luo, Zicheng Zhao, Junnan Liu, Zhangchi Qiu,
Junnan Dong, Serge Panev, Chen Gong, Thuy-Trang
Vu, Gholamreza Haffari, Dinh Phung, Alan Wee-
Chung Liew, and Shirui Pan. 2026. G-reasoner:
Foundation models for unified reasoning over graph-
structured knowledge. InThe Fourteenth Interna-
tional Conference on Learning Representations.
Chaitanya Malaviya, Chandra Bhagavatula, Antoine
Bosselut, and Yejin Choi. 2020. Commonsense
knowledge base completion with structural and se-
mantic context.Proceedings of the 34th AAAI Con-
ference on Artificial Intelligence.

Chaitanya Malaviya, Subin Lee, Sihao Chen, Elizabeth
Sieber, Mark Yatskar, and Dan Roth. 2024. Ex-
pertQA: Expert-curated questions and attributed an-
swers. InProceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), pages 3025–3045,
Mexico City, Mexico. Association for Computational
Linguistics.
Costas Mavromatis and George Karypis. 2025. GNN-
RAG: Graph neural retrieval for efficient large lan-
guage model reasoning on knowledge graphs. In
Findings of the Association for Computational Lin-
guistics: ACL 2025, pages 16682–16699, Vienna,
Austria. Association for Computational Linguistics.
Abhilash Nandy, Soumya Sharma, Shubham Mad-
dhashiya, Kapil Sachdeva, Pawan Goyal, and NIloy
Ganguly. 2021. Question answering over electronic
devices: A new benchmark dataset and a multi-task
learning based QA framework. InFindings of the
Association for Computational Linguistics: EMNLP
2021, pages 4600–4609, Punta Cana, Dominican Re-
public. Association for Computational Linguistics.
Kenta Oono and Taiji Suzuki. 2020. Graph neural net-
works exponentially lose expressive power for node
classification. InInternational Conference on Learn-
ing Representations.
Moritz Plenz and Anette Frank. 2024. Graph language
models. InProceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 4477–4494, Bangkok,
Thailand. Association for Computational Linguistics.
Moritz Plenz, Juri Opitz, Philipp Heinisch, Philipp Cimi-
ano, and Anette Frank. 2023. Similarity-weighted
construction of contextualized commonsense knowl-
edge graphs for knowledge-intense argumentation
tasks. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 6130–6158, Toronto,
Canada. Association for Computational Linguistics.
Nils Reimers and Iryna Gurevych. 2019. Sentence-
BERT: Sentence embeddings using Siamese BERT-
networks. InProceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing
and the 9th International Joint Conference on Natu-
ral Language Processing (EMNLP-IJCNLP), pages
3982–3992, Hong Kong, China. Association for Com-
putational Linguistics.
Leonardo F. R. Ribeiro, Martin Schmitt, Hinrich
Schütze, and Iryna Gurevych. 2021. Investigating
pretrained language models for graph-to-text genera-
tion. InProceedings of the 3rd Workshop on Natural
Language Processing for Conversational AI, pages
211–227, Online. Association for Computational Lin-
guistics.
Stephen E Robertson and Steve Walker. 1994. Some
simple effective approximations to the 2-poissonmodel for probabilistic weighted retrieval. InSI-
GIR’94: Proceedings of the Seventeenth Annual In-
ternational ACM-SIGIR Conference on Research and
Development in Information Retrieval, organised by
Dublin City University, pages 232–241. Springer.
Yu Rong, Wenbing Huang, Tingyang Xu, and Junzhou
Huang. 2020. Dropedge: Towards deep graph convo-
lutional networks on node classification. InInterna-
tional Conference on Learning Representations.
Mobashir Sadat, Zhengyu Zhou, Lukas Lange, Jun
Araki, Arsalan Gundroo, Bingqing Wang, Rakesh
Menon, Md Parvez, and Zhe Feng. 2023. Delu-
cionQA: Detecting hallucinations in domain-specific
question answering. InFindings of the Association
for Computational Linguistics: EMNLP 2023, pages
822–835, Singapore. Association for Computational
Linguistics.
Keshav Santhanam, Omar Khattab, Jon Saad-Falcon,
Christopher Potts, and Matei Zaharia. 2022. Col-
BERTv2: Effective and efficient retrieval via
lightweight late interaction. InProceedings of the
2022 Conference of the North American Chapter of
the Association for Computational Linguistics: Hu-
man Language Technologies, pages 3715–3734, Seat-
tle, United States. Association for Computational
Linguistics.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D Manning.
2024. Raptor: Recursive abstractive processing for
tree-organized retrieval. InThe Twelfth International
Conference on Learning Representations.
Martin Schmitt, Sahand Sharifzadeh, V olker Tresp, and
Hinrich Schütze. 2020. An unsupervised joint sys-
tem for text generation from knowledge graphs and
semantic parsing. InProceedings of the 2020 Con-
ference on Empirical Methods in Natural Language
Processing (EMNLP), pages 7117–7130, Online. As-
sociation for Computational Linguistics.
Yixuan Tang and Yi Yang. 2024. Multihop-RAG:
Benchmarking retrieval-augmented generation for
multi-hop queries. InFirst Conference on Language
Modeling.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics, 10:539–554.
Dario Vajda. 2026. Teaching llms to see graphs: Unify-
ing text and structural reasoning.
Ning Wang, Kuanyan Zhu, Daniel Yuehwoon Yee, Yi-
tang Gao, Shiying Huang, Zirun Xu, and Sainyam
Galhotra. 2026. Pruning minimal reasoning graphs
for efficient retrieval-augmented generation.arXiv
preprint arXiv:2602.04926.

Zhishang Xiang, Chuanjie Wu, Qinggang Zhang,
Shengyuan Chen, Zijin Hong, Xiao Huang, and Jin-
song Su. 2025. When to use graphs in rag: A com-
prehensive analysis for graph retrieval-augmented
generation.arXiv preprint arXiv:2506.05690.
Yilin Xiao, Junnan Dong, Chuang Zhou, Su Dong, Qian
wen Zhang, Di Yin, Xing Sun, and Xiao Huang. 2025.
Graphrag-bench: Challenging domain-specific rea-
soning for evaluating graph retrieval-augmented gen-
eration.
Masatsugu Yamada and Mahito Sugiyama. 2026. When
graph language models go beyond memorization.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 conference on empiri-
cal methods in natural language processing, pages
2369–2380.
Michihiro Yasunaga, Antoine Bosselut, Hongyu Ren,
Xikun Zhang, Christopher D Manning, Percy Liang,
and Jure Leskovec. 2022. Deep bidirectional
language-knowledge graph pretraining. InAdvances
in Neural Information Processing Systems.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, et al. 2025. Qwen3
embedding: Advancing text embedding and rerank-
ing through foundation models.arXiv preprint
arXiv:2506.05176.
Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu,
Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang,
Yulong Chen, et al. 2023. Siren’s song in the ai ocean:
A survey on hallucination in large language models.
arXiv preprint arXiv:2309.01219.
Jianan Zhao, Meng Qu, Chaozhuo Li, Hao Yan, Qian
Liu, Rui Li, Xing Xie, and Jian Tang. 2023. Learning
on large-scale text-attributed graphs via variational
inference. InThe Eleventh International Conference
on Learning Representations.
Appendix
A Details on Datasets
Table 10 shows statistics of the training datasets and
provides a high-level overview of the constructed
KGs, including the total number of queries, docu-
ments, entities, relations, and triplets. The latest
version of GFM-RAG in (Luo et al., 2026) uses
the full training dataset. In this work, we use a
validation set for hyperparameter tuning and there-
fore train on 1k fewer queries per dataset. As
a result, we use a total of 273,830 queries and
2,208,920 documents for training. We also providestatistics for the test datasets in Table 11. The in-
domain test sets each contain 1k queries, whereas
the out-of-domain test sets contain varying num-
bers of queries, ranging from 132 (EManual) to
2,255 (MultiHopRAG).
Tables 12 and 13 provide more detailed statis-
tics about the constructed KGs using the following
metrics:
Theaverage degreeis the mean number of
edges incident to each node within the graph.
Thedensitymeasures the ratio of actual edges
to the total number of possible edges in a sim-
ple graph. Given the high values of |V|in these
datasets, density values near 0.0001 indicate highly
sparse graphs, which is characteristic of large-scale
knowledge bases where only specific and meaning-
ful relations exist.
Thenumber of components(# Components)
indicates the count of maximal subgraphs in which
any two vertices are connected to each other by
paths, but which are disconnected from the rest
of the graph. It reflects the level of fragmentation
within the dataset’s knowledge structure.
Thelargest connected component(CC in %) is
the ratio of nodes in the graph’s largest connected
subgraph to the total number of nodes. The high
percentages (approximately 99.5%) reported across
all datasets indicate that the vast majority of the
knowledge space is reachable through path-based
traversal.
BSupplementary Implementation Details
Table 14 shows the implementation and training set-
tings of the GFM-RAG, GFM-RAG*, GFM-RAG+,
and GLM-RAG models. To highlight the most im-
portant points once again:
•They share theKG-index constructionstage,
since we use pre-constructed KGs for all
datasets except MultiHopRAG. Only for this
dataset is the qwen-3-8b model used for
OpenIE.
•The graph is represented via relative position
encoding with additional buckets, depending
on the setting. The default setting is global
withFullyConnected enabled, resulting
in three additional buckets.
•After hyperparameter search, the loss weights
were set to 0.44 for the binary cross-entropy
(BCE) loss and 0.56 for the list cross-entropy
(ListCE) loss in the GLM retriever.

•The GLM retriever uses differential learning
rates: 5e-4 for the head ( entity_scorer ,
question_proj ) and 1e-4 for the T5 back-
bone.
•The GLM retriever is trained for 2 epochs
with a batch size of 2, whereas the GNN re-
triever is trained for 5 epochs with a batch
size of 4. However, the batch size was also
decreased to 2 for GFM-RAG+ due to the in-
creased memory usage caused by including
node embeddings.
B.1 GFM-RAG+ Details
We tested two ways to implementGFM-RAG+and
adopt the better-performing one as a stronger base-
line. Both initialize nodes with entity embeddings,
however they differ in how the query is injected.
The first, “only seed entities get query”, multi-
plies only the seed entities by the query embedding.
The second, “all entities get query”, initializes ev-
ery node as query_emb∗node_emb , contextual-
izing all entities with the query. This mirrors the
FullyConnected mode of the GLM retriever,
where entities also interact with the query.
In both variants we use the same embedding
model for entities, queries, and relations, and the
final node features combine the entity and query
embeddings. GFM-RAG+ is therefore a query-
dependent GNN, like GFM-RAG. As shown in
Table 6, contextualizing all entities outperforms
seeding alone on both recall metrics, so we use it
as our GFM-RAG+ baseline.
Table 6: Different implementations of GFM-RAG+
GFM-RAG+ Methods Recall@2 Recall@5
Only seed entities gets query 65.7 79.9
All entities gets query67.4 81.5
B.2 GLM-RAG Details
The differences between the models are largely ex-
plained in Section 4.2; however, we explain further
implementation details forGLM-RAG:
Subgraph Selection for Scalability:Due to its
transformer-based architecture, the GLM-retriever
exhibits higher computational overhead than the
GNN-retriever, making it infeasible to encode an
entire large-scale graph simultaneously. To address
this GPU bottleneck, the GLM-retriever employs asubgraph selection strategy that scores local neigh-
borhoods rather than the global graph.
We control the selection “budget” using
two hyperparameters: the number of hops
(max_hops ) and the maximum number of triplets
(max_triplets ). An analysis of the interaction
between these parameters is provided in Section C.
Based on these results, we set max_hops=2 and
max_triplets=600.
While this means that the GNN processes a
larger portion of the graph, we expect that the
GLM’s more sophisticated ranking mechanism
compensates for this restricted view.
Hop Prioritization:The seed entities are ex-
tracted from the queries during the QA dataset
construction process, and they serve as the starting
points in the graph, similar to Plenz et al. (2023). If
themax_hops neighborhoods around the seed en-
tities contain more triplets than max_triplets ,
we sub-sample the neighborhoods to satisfy the
budget constraint. We first prioritize triplets that di-
rectly connect pairs of seed entities. The remaining
quota is then filled using a randomized breadth-first
search starting from the seed entities.
Adjustment of the Losses:For GFM-RAG, Luo
et al. (2025) employ a weighted combination of
BCE and sigmoid-based ListCE (ranking) loss.
Since the GLM operates only on a selected sub-
graph rather than the full graph, the majority of
entities remain unscored with a logit of zero. In
the sigmoid-based ListCE formulation (Bai et al.,
2023), these unscored entities still contribute to
the partition function, where each zero logit con-
tributes σ(0) = 0.5 . For large entity sets (e.g.,
N≈42,000 ), this results in a large accumulated
constant contribution ( ≈21,000 ) in the denomi-
nator, which overwhelms the signal from positive
samples and substantially weakens the learning sig-
nal. To address this issue, we mask unscored en-
tities and restrict the normalization term only to
the selected subgraph, ensuring that the model op-
timizes over the ranking of observed entities.
LListCE =−log 
σ(spos)P
i∈Vsσ(si) +P
j∈Vuσ(sj)!
Computational Costs:GFM-RAG* is trained
on 8 NVIDIA A100 GPUs (80GB) with 7
hours of supervised fine-tuning, whereas GLM-
RAG [t5-large, max_triplets=600] is

trained on 16 GPUs with 20 hours of supervised
fine-tuning. The total estimated computational
cost for all experiments is approximately 180 GPU
hours.
We recognize that GLM-RAG has a higher infer-
ence latency and memory usage in comparison to
GNN based counterparts. Table 7 depicts average
latency and peak GPU memory usage during re-
trieval in 2Wiki dataset. While optimization could
potentially reduce the inference latency, we believe
that the latency is acceptable for complex RAG
tasks.
Table 7: Latency and memory usage comparison during
retrieval.
Latency in ms Allocated memory in GiB
GLM-RAG 692.3 6.8
GFM-RAG* 18.6 1.3
C Extensive Ablations
Figure 7 shows how both the GFM-RAG* and
GLM-RAG models scale with increasing amounts
of training data. Following this trend, and in order
to reduce computational costs, we use a smaller
subset of the data consisting of approximately 60k
queries and 700k documents, instead of approxi-
mately 277k queries and 2.9M documents, for the
following ablation analyses:
15k 30k 60k 277k
Number of Q-Doc Pairs0.580.600.620.640.66Avg. Recall@2
GFM-RAG*
GLM-RAG
Figure 7: Data Scaling: GFM-RAG* vs GLM-RAG
with increasing training data.
In the ablations, the default GLM was initialized
withT5-base, usedmeanembedding aggregation,
and extracted a subgraph from the2-hopneighbor-
hood containing up to600 triplets.
The query interaction was performed in two
ways: by enabling theFullyConnectedmode and
by preserving the later interaction with the query
embedding.Table 15 lists the ablation studies we conducted,
along with the Recall@2 and Recall@5 scores aver-
aged across the three training datasets: HotpotQA,
MusiQue, and 2Wiki.
Embedding Aggregation:We evaluate three
methods for obtaining node embeddings from the
GLM encoder’s output: (i) taking the mean of the
node’s token embeddings, (ii) using the embedding
of the first token, and (iii) using the embedding of
the last token. Our experiments reveal that choos-
ing one method over another does not lead to sub-
stantial performance differences. Therefore, we
usemeanembedding aggregation as the default
setting in order to represent all tokens equally.
Query Interaction:The node representations
interact with the query in two places: (i) the
query tokens are appended to the retrieved triplets
and passed together to the T5 encoder when
use_text is set to FullyConnected , and
(ii) the node representations are multiplied with
the precomputed query embedding from the sen-
tence encoder at a later stage. Disabling this
later interaction is partially compensated for when
FullyConnected remains enabled. Disabling
both interactions leads to the worst results, whereas
enabling both leads to the best performance; there-
fore, this configuration is used as the default set-
ting.
Number of Hops:The graph coverage of the
GLM retriever is bounded by the number of entity-
level hops it can traverse. Our experiments re-
veal that increasing the number of hops from 1
to 2 yields strong performance gains, whereas fur-
ther increases provide only marginal improvements.
Therefore, we set the retrieval scope of the model
to a 2-hop neighborhood.
Maximum Amount of Triplets:The extracted
subgraph is also constrained by the maximum num-
ber of triplets that can be included within its 2-
hop neighborhood. Here, we observe a clear trend
in which retrieval performance improves as more
triplets are included in the subgraph. To avoid ex-
hausting computational resources, we ultimately
setmax_triplets to 600; however, we hypoth-
esize that the performance could further improve
with larger triplet budgets.
T5-Size:Here, we also observe a scaling trend in
which retrieval performance improves with larger
encoder models. Considering that the parameter

sizes explored in this work are still relatively small
compared to modern LLMs, which often contain
tens to hundreds of billions of parameters, we hy-
pothesize that using larger models could further
benefit retrieval performance.
Graph-aware attention:We implemented a
baseline that linearizes the graph into a sequence of
triplets and passes it through T5’s encoder to rank
entities. This ablates the effect of thegraph-aware
attention. Note that this discards GLM’s graph-
aware relative-position encoding and sparsity mask,
the components that inject graph structure, leav-
ing only T5’s standard sequential positional bias.
Because of linearization, the token sequence be-
comes longer than the graph encoding. Thus our
default value of 600 triplets did not fit in memory.
Consequently we trained the linearized GLM re-
triever with max_triplets=300. To enable a direct
comparison, we also train GLM-RAG restricted to
300 triples. In line with our expectations, the de-
clining retrieval results show clearly that both the
stronger language understanding (GLM-RAG vs
GFM-RAG+), and the graph structure (GLM-RAG
vs T5) contribute to GLM-RAG’s success.
D Extensive Retrieval Results
Figures 9 to 11 show the Recall@2 and Recall@5
scores of each model. Additionally, Table 16 shows
a breakdown of the retrieval performance under the
low-data and full-data settings.
There were several issues with the originally re-
ported results of GFM-RAG. The results in the
initial paper (Luo et al., 2025) were affected by
a bug in the code. Later, the authors reported up-
dated results of GFM-RAG in their follow-up paper
(Luo et al., 2026). However, in this version they
evaluated HotpotQA on a different test set. To
avoid confusion arising from these inconsistencies,
we report the reproduced results of GFM-RAG in
our main paper, and Tables 17 and 20 compare
the originally reported and reproduced results of
GFM-RAG. As shown in Table 17, the only notable
difference appears, as expected, on HotpotQA.
Table 17 also shows retrieval comparisons with
previous works. The methods are categorized
as“non-structure methods”, meaning that they do
not utilize graph-based approaches, and“graph-
enhanced methods”. We would like to highlight
again that integrating our proposed GLM retriever
into the current SOTA method, G-Reasoner, is fea-
sible, since it currently employs a GNN-based re-triever, and leave this integration to future work.
D.1 Retrieval Performance by Question
Complexity
Furthermore, we wanted to analyze the retrieval
performance of each model across different lev-
els of question complexity. Thus, we examined
retrieval success at both grouping the questions’
complexity at their document and entity level.
Document Level
To evaluate retrieval success at the document level,
we grouped questions by the number of supporting
documents they require. Figure 15 shows the recall
degradation with increasing numbers of supporting
documents for the multi-hop datasets, while Fig-
ure 16 shows the same analysis for the single-hop
datasets.
Entity Level
Our entity-level measure of question complexity
captures how far the hardest-to-reach supporting
entity lies from the question entities in the KG.
As shown in Figure 8, there are almost always
multiple question (seed) entities as well as multiple
golden (supporting) entities for a question. There-
fore, we calculate the distances from all question
entities to all supporting entities and use the maxi-
mum of these distances for each question.
LetG= (V, E) be the undirected KG, S=
{s1, . . . , s m} ⊆V the set of question entities, and
T={t 1, . . . , t n} ⊆V\S the set of supporting
entities. For each supporting entity tj, we compute
its minimum shortest-path distance to any seed
entity:
d(tj, S) = min
s∈SdG(s, tj)
The entity-level distance is then defined as the
maximum over all reachable supporting entities:
depth(S, T) = max
j=1,...,n
d(tj,S)̸=∞d(tj, S)
This captures the minimum number of relational
hops the model must traverse to reach the most dis-
tant piece of evidence. Questions are grouped ac-
cording to this depth value to analyze how retrieval
performance degrades as reasoning complexity in-
creases.
It is important to note that the entity-level dis-
tances in the “single-hop datasets” do not imply

that these are “multi-hop datasets”, since these hops
do not cross document boundaries.
Figures 13 and 14 show the recall degradation
with increasing entity-level distance for both single-
hop and multi-hop datasets. All models follow a
similar trend, with performance decreasing as the
entity-level distance increases.
{"id": "cc7f68a9086c11ebbd61ac1f6bf848b6
",
"question": "Are The M Machine and
Signals Midwest from the same country?",
"answer": "yes",
"answer_aliases": [],
"supporting_facts":
["The M Machine",
"Signals Midwest"],
"question_entities":
["the m machine",
"signals midwest"],
"supporting_entities":
["ohio",
"cleveland",
"american punk rock band",
"signals midwest",
"2011",
"san francisco",
"united states",
"american electronic music duo",
"four eps",
"skrillex s label owsla",
"overall 1 slot on beatport",
"the m machine",
"ben swardlick",
"two remix collections",
"california",
"eric luttrell",
"a single"]
}
Figure 8: An example test question from 2Wiki.
D.2 Retrieval Performance by Subgraph
Budget
In order to analyze the subgraph budget behavior
on different neighborhoods, we separate questions
into three categories, namely sparse, medium and
dense, using the number of edges around the seed
entities. The density distribution of the queries
are more on the sparse side, however we account
for this by separating the queries in equal terciles.
Then we analyze whether the golden documents
are reachable with our subgraph selection method
under varying hop and triplet counts. This analyzes
thecoveragewith increasing subgraph budget.
Figure 17 shows that in dense neighborhoods,
1-hop alone can already exceed the budget, so in-
creasing max_hops without increasing the tripletbudget does not help. In sparse neighborhoods the
opposite holds; extra hops help, the triplet count
rarely binds. However, for instance on HotpotQA 2
hops and 200 triplets are enough to reach >96% of
the gold documents in all three density categories.
Following these findings, we have also analyzed
how theretrieval performanceis affected from
the density of the neighborhood. Figure 18 shows
degradation of performance with higher density
questions on 2Wiki and MuSiQue for both models,
however the trend is not so clear on HotPotQA.
Considering the visibility restriction of GLM-RAG
and despite the full visibility of GFM-RAG*, we
still see similar degrading trends in dense neighbor-
hoods, meaning that our restriction does not behave
disruptively worse in practice.
E Extensive QA Results
Figure 19 shows the Exact Match (EM) and F1
scores for the Wikipedia datasets, and Table 20
provides comparisons with additional baselines.
Figure 20 shows the prompt example provided
to the LLM.
Table 8 shows an additional baseline where the
model receives a subgraph instead of documents
in its context. The subgraph selection is done via
the same method in GLM-RAG, starting from the
question entities and selecting 600 triplets in a 2-
hop neighborhood.
In our RAG pipeline, the knowledge graph is
constructed by extracting entities from documents,
and the retriever acts on a subgraph to find the
relevant entities. The entities are then mapped back
to the documents so that the “relevant” documents
can be used for generation.
Although simply providing the entity-based
knowledge subgraph would not be an apples-to-
apples comparison and we expect it to underper-
form, we want to highlight the importance of giving
the whole document as context instead of just the
graphs itself since they contain more information.
Table 8 shows how thesubgraph baselinecom-
pares against other methods on MuSiQuE, with
gpt-4o-mini as the LLM. This resulted in a
weaker baseline that underperforms other graph-
based methods, showing the need for the full-
context with documents in question answering.
F Extensive Transferability Results
Table 21 shows the models’ F1 scores on the down-
stream QA task. There are two reasons why the

Table 8: QA results with a subgraph baseline. Just giv-
ing the subgraph underperforms the document baselines.
MuSiQuE EM F1
No-Context Baseline 10.8 20.6
Subgraph Baseline 22.9 30.7
RAG Baseline 23.8 33.3
GFM-RAG 28.7 40.8
GFM-RAG+ 31.6 43.3
GFM-RAG* 29.0 40.6
GLM-RAG32.1 43.8
high retrieval performance is not fully reflected in
the QA results:
Firstly, in comparison to the Wiki datasets, the
golden answers for these transfer datasets are full
sentences rather than single words or phrases. This
means that QA metrics such as EM and F1 do not
fully capture the quality of the generated answers.
Secondly, the strong performance of the
no-context baseline indicates that much of
this information is already contained within
gpt-4o-mini ’s parametric knowledge, meaning
that the model does not necessarily require the re-
trieved context to answer the questions correctly.
This potentially dilutes the impact a strong retrieval
can have.
Nonetheless, GLM-RAG achieves the best or
second-best F1 scores on 7 out of 8 datasets; how-
ever, the performance gaps remain relatively small
due to the issues described above.
The same problem arises in the QA evaluation
of MultihopRAG. Given that gpt-4o-mini can
already achieve quite high scores without any ad-
ditional context, shown on the no context baseline
on Table 9, we conclude that both the exact match
and the accuracy measures are not expressive of
the retrieval quality.
Table 9: Retrieval (Recall@2 and Evidence Recall from
G-Bench (Xiang et al., 2025)) and QA performance (EM
and ACC from G-Bench (Xiang et al., 2025)) on Multi-
hopRAG, answers generated withgpt-4o-mini.
R@2 RE EM ACC
No Context - - 72.8 75.7
RAG Baseline 32.5 55.879.4 78.1
GFM-RAG 34.1 54.5 71.1 73.5
GFM-RAG+ 39.0 55.8 72.7 74.2
GLM-RAG (ours)60.0 58.175.5 77.1G Extensive Analysis
In the following, we explain how the similarity and
distance measures (see Section 6) are computed.
Similarity Measure
Semantic similarity between the question and re-
trieved entities is computed as the cosine similarity
of their sentence-level embeddings. Specifically,
we encode the question text qand each retrieved
entity name eiusing a pretrained sentence embed-
ding model ( all-mpnet-base-v2 ), yielding
embedding vectors q= Enc(q) andei= Enc(e i).
Each embedding isℓ 2-normalized:
ˆq=q
∥q∥ 2,ˆei=ei
∥ei∥2
The semantic similarity score for a retrieved en-
tityeiwith respect to the question qis then defined
as:
sim(q, e i) =ˆq⊤ˆei
which is equivalent to the cosine similarity
cos(q,e i)∈[−1,1] . For each sample, the scores
are computed over the top- kretrieved entities and
averaged to produce a per-sample semantic rele-
vance score.
Distance Measure
Structural proximity between retrieved entities and
the question (seed) entities is measured as the
shortest-path distance in the knowledge graph. Let
G= (V, E) be the undirected knowledge graph
andS={s 1, . . . , s m} ⊆V the set of seed enti-
ties extracted from the question. For each retrieved
entity ei, we compute its minimum shortest-path
distance to any seed entity:
d(ei, S) = min
s∈SdG(s, e i)
where dG(s, e i)is the length of the shortest path
between sandeiinG, obtained via breadth-first
search with a cutoff of 10 hops. If no path exists
within the cutoff, the entity is considered unreach-
able. The per-sample structural proximity score is
defined as the average distance over all reachable
top-kretrieved entities:
¯d=1
|{i:d(e i, S)̸=∞}|kX
i=1
d(ei,S)̸=∞d(ei, S)

Lower values indicate that the retrieval model
favors entities that are structurally close to the ques-
tion entities in the KG.
H Error Bars and Significancy Tests
Table 22 shows significance tests with paired boot-
strap tests and 95% confidence intervals (CI) that
compare GLM-RAG against all other baselines.
Holm-Bonferroni correction is applied to account
for family-wide error rates. Significant results are
indicated with one or multiple stars(*)indicating
that p-values are; p: *** <.001 ** <.01 * <.05 ns
>=.05
The results confirm that although the differences
between GLM-RAG and GFM-RAG+ are not sig-
nificant in in-domain retrieval, they are significant
in out-of-domain retrieval.
I Declaration of AI usage
We use AI assistants for speeding up programming,
and to aid with reformulations. The content of this
work is our own, and not inspired by AI assistants.
Table 10: Statistics of the datasets and constructed KG-indexes used fortraining.
Dataset # Queries # Documents # Entities # Relations # Triplets
HotpotQA (train) 89,447 874,784 8,259,397 4,134,914 27,362,937
HotpotQA (valid) 1,000 9,742 92,651 46,290 314,944
MuSiQue (train) 18,938 378,724 1,374,033 801,704 4,315,300
MuSiQue (valid) 1,000 20,000 70,080 42,690 213,797
2WikiMultihopQA (train) 165,454 955,412 7,120,577 2,876,828 22,188,618
2WikiMultihopQA (valid) 1,000 5,761 42,489 17,228 130,397
Total Train 273,830 2,208,920 18,579,900 3,814,333 53,866,855
Table 11: Statistics of the datasets and constructed KG-indexes used fortesting.
Dataset Domain #Queries #Documents #Entities #Relations #Triplets
HotpotQA (test) Wikipedia 1,000 9,221 87,768 45,112 279,112
MuSiQue (test) Wikipedia 1,000 11,656 100,853 55,944 319,618
2Wiki (test) Wikipedia 1,000 6,119 48,779 20,748 160,950
MultiHopRAG News 2,255 609 16,147 9,416 24,180
PubMedQA Biomedical 2,450 5,932 42,389 20,952 149,782
DelucionQA Customer Support 184 235 2,669 2,298 6,183
TechQA Customer Support 314 769 10,221 4,606 57,613
ExpertQA Customer Support 203 808 11,079 6,810 16,541
EManual Customer Support 132 102 695 586 1,329
MS Marco General Knowledge 423 3,481 24,740 17,042 63,995
HAGRID General Knowledge 1,318 1,975 23,484 18,653 48,969
G-Bench (CS) Computer Science 1,018 15,011 187,217 79,578 991,630
G-Bench (Novel) Novels 2,010 1,881 39,870 37,344 68,483
G-Bench (Medical) Medical 2,062 390 8,676 6,814 32,386

Table 12: Graph statistics of the datasets and constructed KG-indexes used fortraining.
Dataset Avg Degree Density # Components Largest CC %
HotpotQA (train) 5.89 0.0001 138 99.66
HotpotQA (valid) 6.02 0.0001 125 99.69
MuSiQue (train) 5.59 0.0001 140 99.53
MuSiQue (valid) 5.48 0.0001 137 99.55
2Wiki (train) 5.47 0.0001 92 99.50
2Wiki (valid) 5.40 0.0001 75 99.59
Table 13: Graph statistics of the test datasets and constructed KG-indexes used fortesting.
Dataset Avg Degree Density # Components Largest CC %
HotpotQA (test) 5.68 0.0001 150 99.60
MuSiQue (test) 5.70 0.0001 201 99.56
2Wiki (test) 5.77 0.0001 120 99.41
MultiHopRAG 2.84 0.0002 378 93.63
PubMedQA 6.28 0.0001 129 99.27
DelucionQA 4.11 0.0015 31 97.45
TechQA 8.98 0.0009 98 97.36
ExpertQA 2.77 0.0002 449 89.99
EManual 3.42 0.0049 18 94.39
MS Marco 4.62 0.0002 124 98.82
HAGRID 3.81 0.0002 253 97.26
G-Bench (CS) 9.60 0.0001 558 99.29
G-Bench (Novel) 3.17 0.0020 34.25 95.63
G-Bench (Medical) 6.60 0.0008 39 98.80
MuSiQue HotPotQA 2Wiki020406080100Recall@2 Score (%)33.259.4
48.5
43.973.179.0
43.772.779.0
45.676.979.6
46.474.878.6RAG Baseline
GFM-RAG*
GFM-RAG
GFM-RAG+
GLM-RAG (ours)
MuSiQue HotPotQA 2Wiki020406080100Recall@5 Score (%)46.373.3
59.457.887.192.2
57.788.792.4
60.890.593.0
62.289.092.2RAG Baseline
GFM-RAG*
GFM-RAG
GFM-RAG+
GLM-RAG (ours)
Figure 9: Retrieval performance (Recall@2 and Recall@5) on in-domain Wikipedia datasets.

Table 14: The detailed implementation and training settings of GFM-RAG, GFM-RAG*, GFM-RAG+ and GLM.
SettingGFM-RAG (*, +) GLM-RAG
KG-index Construction
OpenIE GPT-4o-mini GPT-4o-mini
Entity resolution ColBERTv2 ColBERTv2
Models
Backbone GNN (based on Galkin et al. (2024)) GLM (based on T5-large)
# (Encoder) Layer 6 12
# Parameters 8,144,897 770M
Hidden dim ((d model)) 512 768
Attention heads - 12
Graph encoding - Levi graph + global RPE
Message DistMult -
Aggregation Sum -
Entity scorer 2-layer MLP 2-layer MLP
Sentence embedding model all-mpnet-base-v2 all-mpnet-base-v2
Doc ranker entities 20 20
Finetuning
α0.3 0.44
Optimizer AdamW AdamW
(Head) Learning rate 5e-4 5e-4
Backbone learning rate - 1e-4
Batch size 4 2
Training epochs 5 2
Max triplets - 600
# Hops - 2
T echQAPubMedQA MS MARCOHAGRID ExpertQA eManualDelucionQA020406080100Recall@2 Score (%)35.037.0
26.064.8
40.356.5
45.2
29.836.4
26.262.9
38.548.8
40.8
32.039.3
28.468.1
46.243.7 44.9
40.644.9
29.569.6
44.557.7
49.145.959.9
36.778.6
60.772.469.7RAG Baseline GFM-RAG* GFM-RAG GFM-RAG+ GLM-RAG (ours)
T echQAPubMedQA MS MARCOHAGRID ExpertQA eManualDelucionQA020406080100Recall@5 Score (%)48.861.0
56.983.7
59.267.275.0
47.659.356.581.5
58.165.972.1
48.161.960.488.2
67.078.076.2
57.569.1
63.088.9
67.784.381.5
58.990.3
73.093.8
83.994.1 95.1RAG Baseline GFM-RAG* GFM-RAG GFM-RAG+ GLM-RAG (ours)
Figure 10: Retrieval performance (Recall@2 and Recall@5) on 7 out-of-domain single-hop datasets.

Table 15: Ablation study results averaged over HotpotQA, MuSiQue, and 2Wiki test sets.
Ablation Method Recall@2 Recall@5
Embedding Aggregation Mean (default) 63.3 79.6
First 63.4 79.6
Last 63.3 79.2
Query Interaction
[yes/no and use_text=FullyConnected/no]Yes and FC (default) 63.3 79.6
No and FC 63.5 79.4
Yes and No FC 58.7 74.6
No and No FC 57.3 73.4
Number of Hops 1 Hop 49.4 66.4
2 Hops (default) 63.3 79.6
3 Hops 63.0 79.7
4 Hops 63.6 79.6
Max Triplets 300 60.8 76.9
400 61.8 78.4
500 62.8 79.1
600 (default) 63.3 79.6
T5 Encoder Size T5-Small (60M) 60.4 76.7
T5-Base (220M) (default) 63.3 79.6
T5-Large (770M) 64.7 80.6
Graph Aware Attentionmax_triplets = 300With 60.8 76.9
Without (linearized) 38.7 51.3
RAG BaselineGFM-RAG* GFM-RAG GFM-RAG+ G-ReasonerGLM-RAG (ours)020406080Recall@2 Score (%)32.534.6 34.139.034.960.0
RAG BaselineGFM-RAG* GFM-RAG GFM-RAG+ G-ReasonerGLM-RAG (ours)020406080Recall@5 Score (%)53.256.053.763.965.677.5
Figure 11: Retrieval performance (Recall@2 and Recall@5) on MultiHopRAG, an out-of-domain dataset with
multi-hop questions.

Table 16: Retrieval performance (Recall@2 and Recall@5) on in-domain Wikipedia datasets in low data and full
data settings of comparable graph methods. Best results areboldedand second best are underlined .
HotpotQA MuSiQue 2Wiki
Data Amount Method R@2 R@5 R@2 R@5 R@2 R@5
60k Q-Doc PairsRAG Baseline 48.5 46.3 33.2 59.4 73.3 59.4
GFM-RAG (reproduced) 69.5 85.9 42.8 57.1 77.4 91.5
GFM-RAG* 70.4 85.0 43.2 58.2 76.9 91.4
GFM-RAG+72.7 88.844.1 59.1 77.4 92.2
GLM-RAG (ours) 71.2 88.1 46.0 61.576.8 92.1
277k Q-Doc PairsGFM-RAG (reproduced) 72.7 88.7 43.7 57.7 79.0 92.4
GFM-RAG* 73.1 87.1 43.9 57.8 79.0 92.2
GFM-RAG+76.9 90.645.6 60.8 79.6 93.0
GLM-RAG (ours) 74.8 89.0 46.4 62.278.6 92.2
Table 17: Retrieval performance comparison (Recall@2 and Recall@5). Best results arebolded, second best are
underlined , and third best areitalized. Baseline scores are adapted from Luo et al. (2026).
HotpotQA MuSiQue 2Wiki
Method R@2 R@5 R@2 R@5 R@2 R@5
Non-structure Methods
BM25 (Robertson and Walker, 1994) 55.4 72.2 32.3 41.2 51.8 61.9
ColBERTv2 (Santhanam et al., 2022) 64.7 79.3 37.9 49.2 59.2 68.2
Qwen3-Emb (8B) (Zhang et al., 2025) 74.1 88.846.862.1 66.2 74.1
Graph-enhanced Methods
RAPTOR (Sarthi et al., 2024) 58.1 71.2 35.7 45.3 46.3 53.8
GraphRAG (MS) (Edge et al., 2024) 58.3 76.6 35.4 49.3 61.6 77.3
LightRAG (Guo et al., 2025) 38.8 54.7 24.8 34.7 45.1 59.1
KAG (Liang et al., 2025) 59.4 86.1 42.262.461.4 88.3
HippoRAG (Gutiérrez et al., 2024) 60.1 78.5 41.2 53.2 68.4 87.0
HippoRAG 2 (Gutiérrez et al., 2025) 80.5 95.7 53.5 74.2 80.5 95.7
SubgraphRAG (Li et al., 2025) 58.1 71.7 40.6 48.1 70.2 85.3
GFM-RAG (Luo et al., 2025) 75.6 89.6 43.5 57.6 79.1 92.4
GFM-RAG (reproduced) 72.7 88.7 43.7 57.7 79.0 92.4
GFM-RAG* 73.1 87.1 43.9 57.8 79.0 92.2
GFM-RAG+76.9 90.645.6 60.879.6 93.0
GLM-RAG (ours) 74.8 89.0 46.4 62.2 78.6 92.2
G-Reasoner (Luo et al., 2026)85.9 97.7 54.8 74.9 81.2 98.2

Table 18: GNN-based and GLM-based retriever models parameter comparison when scaling up the GFM-RAG* by
increasing its hidden dimensions from 512 to1024,2048and4096, following Luo et al. (2026). Since we only use
the encoder, the parameter count of GLMs is half the original parameter count of T5-models.
Models Parameter Count
GLM-RAG (t5-small) 36M
GFM-RAG(*, +) (hidden size 1024) 31M
GLM-RAG (t5-base) 110M
GFM-RAG(*, +) (hidden size 2048) 122M
GLM-RAG (t5-large) 336M
GFM-RAG(*, +) (hidden size 4096) 476M
30M 100M 500M
Model Size (Parameters)525354555657585960Avg. Recall@2
5 Datasets (3 ID + 2 OOD)
GFM-RAG*
GFM-RAG+
GLM-RAG
30M 100M 500M
Model Size (Parameters)6162636465
In-Domain
30M 100M 500M
Model Size (Parameters)35.037.540.042.545.047.550.052.5
Out-of-Domain
Figure 12: Comparision of GFM-RAG, GFM-RAG+ and GLM-RAG’s retrieval quality on the three in-domain
datasets (HotpotQA, MuSiQuE, 2Wiki), as well as one out-of-domain single-hop (PubMedQA) and one one out-of-
domain multi-hop (MultiHopRAG). Scaling up the GFM-RAG variants doesn’t bring the same contributions as
scaling up the GLM-RAG in out-of-domain settings.
Table 19: Retrieval performance (Recall@2 and Recall@5) of the scaling experiments on the 60k dataset. Best
results arebolded.
HotpotQA MuSiQue 2Wiki PubMedQA MultiHopRAG
Method R@2 R@5 R@2 R@5 R@2 R@5 R@2 R@5 R@2 R@5
GFM-RAG*(hidden_dim=512, 8M)70.4 85.0 43.2 58.2 76.9 91.4 35.8 58.8 29.7 50.1
GFM-RAG*(hidden_dim=1024, 31M)70.3 86.6 43.5 57.4 77.7 92.0 35.1 59.8 33.3 53.4
GFM-RAG*(hidden_dim=2048, 122M)71.3 87.0 42.3 57.5 77.9 91.3 35.0 57.4 34.8 57.6
GFM-RAG*(hidden_dim=4096, 476M)71.8 86.7 42.9 57.4 77.6 91.7 36.1 59.8 34.6 56.1
GFM-RAG+(hidden_dim=512, 8M)72.788.844.1 59.1 77.492.238.3 60.3 41.4 62.4
GFM-RAG+(hidden_dim=1024, 31M)70.4 87.4 42.2 56.9 76.4 91.0 37.7 60.3 38.0 59.6
GFM-RAG+(hidden_dim=2048, 122M)73.188.1 44.0 57.9 77.5 91.8 39.0 60.5 35.3 57.0
GFM-RAG+(hidden_dim=4096, 476M)73.187.2 44.8 58.278.191.4 37.0 59.8 40.1 62.8
GLM-RAG(t5-small, 36M)64.4 82.7 43.1 57.1 73.7 90.4 40.5 62.9 38.6 57.7
GLM-RAG(t5-base, 110M)68.8 85.9 44.8 61.1 76.1 91.9 43.6 66.6 48.2 66.6
GLM-RAG(t5-large, 336M)71.2 88.146.0 61.576.8 92.146.0 68.8 59.5 78.1

Table 20: QA reasoning performance comparison (Exact Match and F1). gpt-4o-mini is used as the LLM. Best
results arebolded, second best are underlined , and third best areitalized. Baseline scores are adapted from Luo
et al. (2026).
HotpotQA MuSiQue 2Wiki
Method EM F1 EM F1 EM F1
Non-structure Methods
BM25 (Robertson and Walker, 1994) 52.0 63.4 20.3 28.8 47.9 51.2
ColBERTv2 (Santhanam et al., 2022) 43.4 57.7 15.5 26.4 33.4 43.3
Qwen3-Emb (8B) (Zhang et al., 2025) 53.4 67.6 31.9 44.1 57.2 63.2
Graph-enhanced Methods
RAPTOR (Sarthi et al., 2024) 50.6 64.7 27.7 39.2 39.7 48.4
GraphRAG (MS) (Edge et al., 2024) 51.4 67.6 27.0 42.0 34.7 61.0
LightRAG (Guo et al., 2025) 9.9 20.2 2.0 9.3 2.5 12.1
KAG (Liang et al., 2025) 59.5 72.2 33.846.0 67.3 75.1
HippoRAG (Gutiérrez et al., 2024) 46.3 60.0 24.0 35.9 59.4 67.3
HippoRAG 2 (Gutiérrez et al., 2025)56.3 71.135.0 49.3 60.5 69.7
SubgraphRAG (Li et al., 2025) 44.5 57.0 25.1 35.7 62.7 69.0
G-retriever (He et al., 2024) 41.4 53.4 23.6 34.3 33.5 39.6
GFM-RAG (Luo et al., 2025) 56.2 69.5 30.249.269.8 77.7
GFM-RAG (reproduced) 55.1 70.7 28.7 40.8 67.8 75.7
GFM-RAG + 55.3 70.6 31.6 43.369.0 77.0
GFM-RAG* 53.4 68.3 29.0 40.6 67.9 75.9
GLM-RAG (ours) 55.7 70.3 32.1 43.8 67.5 75.5
G-Reasoner (Luo et al., 2026)61.4 76.0 38.5 52.5 74.9 82.1
2
# Supporting Documents020406080100Recall@2 (%)HotPotQA
RAG Baseline
GFM-RAG*
GFM-RAG
GFM-RAG+
GLM-RAG (ours)
2 4
# Supporting Documents
2WikiMQA
2 3 4
# Supporting Documents
MuSiQue
2 3 4
# Supporting Documents
MultiHopRAG
2
# Supporting Documents020406080100Recall@5 (%)
2 4
# Supporting Documents
2 3 4
# Supporting Documents
2 3 4
# Supporting Documents
Recall Degradation with Increasing Number of Supporting Documents (Multi-Hop Datasets)
Figure 13: Recall degradation with increasing number of supporting documents for multi-hop datasets.

1 2 3
# Supporting Documents020406080100Recall@2 (%)
DelucionQA
RAG Baseline
GFM-RAG*
GFM-RAG
GFM-RAG+
GLM-RAG (ours)
1 2 3
# Supporting Documents
eManual
1 2 3 4 5 6 7
# Supporting Documents
ExpertQA
1234567891011
# Supporting Documents
HAGRID
12345678910
# Supporting Documents
MS MARCO
1 2 3 4 5
# Supporting Documents
PubMedQA
1 2 3 4 5
# Supporting Documents
T echQA
1 2 3
# Supporting Documents020406080100Recall@5 (%)
1 2 3
# Supporting Documents
1 2 3 4 5 6 7
# Supporting Documents
1234567891011
# Supporting Documents
12345678910
# Supporting Documents
1 2 3 4 5
# Supporting Documents
1 2 3 4 5
# Supporting Documents
Recall Degradation with Increasing Number of Supporting Documents (Single-Hop Datasets)Figure 14: Recall degradation with increasing number of supporting documents for single-hop datasets.
1 2 3 4 5 6 7
Entity-Level Distance020406080100Recall@2 (%)
HotPotQA
GFM-RAG*
GFM-RAG
GFM-RAG+
GLM-RAG (ours)
1 2 3 4 5 6 7
Entity-Level Distance
2WikiMQA
2 3 4 5 6 7 8
Entity-Level Distance
MuSiQue
2 4 6 8 10 12
Entity-Level Distance
MultiHopRAG
1 2 3 4 5 6 7
Entity-Level Distance020406080100Recall@5 (%)
1 2 3 4 5 6 7
Entity-Level Distance
2 3 4 5 6 7 8
Entity-Level Distance
2 4 6 8 10 12
Entity-Level Distance
Recall Degradation with Increasing Entity-Level Distance (Multi-Hop Datasets)
Figure 15: Recall degradation with increasing entity-level distance between golden and retrieved entities for
multi-hop datasets.
2 3 4 5 6 7 8
Entity-Level Distance020406080100Recall@2 (%)
DelucionQA
GFM-RAG*
GFM-RAG
GFM-RAG+
GLM-RAG (ours)
2 3 4
Entity-Level Distance
eManual
345678910111213
Entity-Level Distance
ExpertQA
123456789101112
Entity-Level Distance
HAGRID
2 3 4 5 6 7 8 910
Entity-Level Distance
MS MARCO
1 2 3 4 5 6 7 8 9
Entity-Level Distance
PubMedQA
2 3 4 5 6 7 8
Entity-Level Distance
T echQA
2 3 4 5 6 7 8
Entity-Level Distance020406080100Recall@5 (%)
2 3 4
Entity-Level Distance
345678910111213
Entity-Level Distance
123456789101112
Entity-Level Distance
2 3 4 5 6 7 8 910
Entity-Level Distance
1 2 3 4 5 6 7 8 9
Entity-Level Distance
2 3 4 5 6 7 8
Entity-Level Distance
Recall Degradation with Increasing Entity-Level Distance (Single-Hop Datasets)
Figure 16: Recall degradation with increasing entity-level distance between golden and retrieved entities for
single-hop datasets.

200 300 400 500 6000.910.920.930.940.950.960.970.980.99HotPotQA
document reachability
max_hops = 1
200 300 400 500 600
max_hops = 2
200 300 400 500 600
max_hops = 3
density
sparse
medium
dense
200 300 400 500 6000.920.930.940.950.960.970.980.991.002Wiki
document reachability
200 300 400 500 600
 200 300 400 500 600
200 300 400 500 600
max_triplets0.800.820.840.860.880.900.920.940.96MuSiQue
document reachability
200 300 400 500 600
max_triplets
200 300 400 500 600
max_triplets
Document reachability vs budget by density stratumFigure 17: Golden document reachability of GLM-RAG split into 3 question categories regarding their neighborhood
density with increasing number of hops and triplets.
Table 21: QA performance (F1 score) on out-of-domain datasets, answers generated withgpt-4o-mini.
Method TechQA PubMedQA MS MARCO HAGRID ExpertQA eManual DelucionQA
No Context 19.1 20.8 29.9 21.5 16.7 19.6 19.1
RAG Baseline25.4 21.7 36.7 27.119.4 31.7 32.1
GFM-RAG* 23.6 21.0 35.7 27.1 19.6 31.2 33.6
GFM-RAG 23.5 21.1 35.3 26.9 19.7 31.534.3
GFM-RAG+ 21.3 20.3 33.0 26.8 17.7 27.7 30.8
GLM-RAG (ours) 25.2 21.3 35.9 27.1 19.9 32.533.2

sparse medium dense0.00.20.40.60.81.0HotPotQA
RecallRecall@2
sparse medium denseRecall@5
sparse medium dense0.00.20.40.60.81.02Wiki
Recall
sparse medium dense
sparse medium dense
neighbourhood density0.00.20.40.60.81.0MuSiQue
Recall
sparse medium dense
neighbourhood densityRanked recall by neighbourhood densityGLM-RAG GFM-RAG*Figure 18: Retrieval recall of GLM-RAG and GFM-RAG* split into 3 question categories regarding their neighbor-
hood density.
MuSiQue HotPotQA 2Wiki020406080Exact Match Score (%)
10.825.824.1 23.943.7
40.9
29.053.467.9
28.755.167.8
31.655.369.0
32.155.767.5No Context
RAG Baseline
GFM-RAG*
GFM-RAG
GFM-RAG+
GLM-RAG (ours)
MuSiQue HotPotQA 2Wiki020406080F1 Score (%)
20.638.2
33.335.858.4
46.4
40.668.375.9
40.870.775.7
43.370.677.0
43.870.375.5No Context
RAG Baseline
GFM-RAG*
GFM-RAG
GFM-RAG+
GLM-RAG (ours)
Figure 19: QA performance (Exact Match and F1) on in-domain Wikipedia datasets withgpt-4o-mini.

Question Answering Prompt
<system_prompt>
As an advanced reading comprehension assistant, your task is to analyze text passages and
corresponding questions meticulously. Your response start after "Thought: ", where you will
methodically break down the reasoning process, illustrating how you arrive at conclusions.
Conclude with "Answer: " to present a concise, definitive response, devoid of additional
elaborations.’
<examples>
input: |-
Wikipedia Title: Kurram Garhi
Kurram Garhi is a small village located near the city of Bannu, which is the part of Khyber
Pakhtunkhwa province of Pakistan. Its population is approximately 35000. Barren hills are near
this village. This village is on the border of Kurram Agency. Other nearby villages are Peppal,
Surwangi and Amandi Kala.
Wikipedia Title: 2001–02 UEFA Champions League second group stage
Eight winners and eight runners- up from the first group stage were drawn into four groups of four
teams, each containing two group winners and two runners- up. Teams from the same country or
from the same first round group could not be drawn together. The top two teams in each group
advanced to the quarter- finals.
Wikipedia Title: Satellite tournament
A satellite tournament is either a minor tournament or event on a competitive sporting tour or one
of a group of such tournaments that form a series played in the same country or region.
Wikipedia Title: Trojkrsti
Trojkrsti is a village in Municipality of Prilep, Republic of Macedonia.
Wikipedia Title: Telephone numbers in Ascension Island
Country Code:+ 247<br> International Call Prefix: 00 Ascension Island does not share the same
country code( +290) with the rest of St Helena.
Question:Are both Kurram Garhi and Trojkrsti located in the same country?
Thought:
response: |-
Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic
of Macedonia. Thus, they are not in the same country.
Answer:no.
<doc_prompt>
"Wikipedia Title: {title}
{content}"
<question>
"Question: {question}
Thought: "
Figure 20: The one-shot prompt template for 2Wiki. MuSiQue and HotpotQA have similar templates, adjusted to
their examples. For the no-context prompt template, only the<system_prompt>is given.

Table 22: Retrieval recall (mean ±half the 95% bootstrap CI width) on the question set shared by all systems within
each dataset. Markers give a paired bootstrap test of GLM vs each system, corrected across all 88 comparisons with
Holm-Bonferroni (FWER):∗/∗∗/∗∗∗= GLM significantly better (p < .05/.01/.001),†= significantly worse.
Recall@2 Recall@5
Dataset RAG GFM-RAG GNN GNN+GLM RAG GFM-RAG GNN GNN+GLM
2WikiMultihopQA.485 ±.017∗∗∗.790±.016 .790±.017 .796±.016 .785±.016 .594±.017∗∗∗.924±.011 .922±.011 .930±.011 .922±.011
DelucionQA.687 ±.077†††.423±.107 .473±.098 .460±.093 .500±.097 .946±.044 .717±.100 .747±.087 .770±.088 .813±.083
eManual.700 ±.121 .514±.148 .581±.145 .457±.143 .600±.140 .967±.038 .676±.140 .686±.133 .776±.119 .848±.098
ExpertQA.638 ±.052†††.385±.058∗.403±.056 .462±.055 .445±.056 .892±.037†††.581±.064∗∗∗.592±.060∗∗∗.670±.057 .677±.056
HAGRID.807 ±.017†††.629±.023∗∗∗.648±.022∗∗∗.681±.020 .696±.021 .961±.008†††.815±.019∗∗∗.837±.018∗∗∗.882±.015 .889±.015
MS MARCO.374 ±.022†††.262±.022∗.260±.022 .284±.021 .295±.021 .736±.021†††.565±.028∗∗∗.569±.029∗∗∗.604±.026 .630±.027
PubMedQA.603 ±.016†††.354±.020∗∗∗.364±.019∗∗∗.382±.019∗∗∗.437±.019 .914±.011†††.582±.021∗∗∗.601±.021∗∗∗.610±.021∗∗∗.684±.020
TechQA.561 ±.083 .290±.073∗∗∗.340±.078 .310±.081 .404±.080 .715±.073 .457±.079∗∗∗.482±.080 .480±.080 .571±.078
HotpotQA.592 ±.019∗∗∗.728±.020 .733±.021 .770±.019 .745±.019 .730±.019∗∗∗.887±.016 .871±.017 .906±.014 .890±.015
MultiHop-RAG.312 ±.013∗∗∗.341±.019∗∗∗.346±.020∗∗∗.390±.019∗∗∗.600±.019 .525±.015∗∗∗.537±.021∗∗∗.560±.020∗∗∗.639±.020∗∗∗.775±.017
MuSiQue.332 ±.016∗∗∗.437±.019∗∗∗.439±.019 .456±.019 .463±.018 .463±.017∗∗∗.577±.020∗∗∗.578±.020∗∗∗.608±.020 .620±.019
# Biomedical Question Answering via Multi-Level Summarization on a Local Knowledge Graph

**Authors**: Lingxiao Guan, Yuanhao Huang, Jie Liu

**Published**: 2025-04-02 02:40:19

**PDF URL**: [http://arxiv.org/pdf/2504.01309v1](http://arxiv.org/pdf/2504.01309v1)

## Abstract
In Question Answering (QA), Retrieval Augmented Generation (RAG) has
revolutionized performance in various domains. However, how to effectively
capture multi-document relationships, particularly critical for biomedical
tasks, remains an open question. In this work, we propose a novel method that
utilizes propositional claims to construct a local knowledge graph from
retrieved documents. Summaries are then derived via layerwise summarization
from the knowledge graph to contextualize a small language model to perform QA.
We achieved comparable or superior performance with our method over RAG
baselines on several biomedical QA benchmarks. We also evaluated each
individual step of our methodology over a targeted set of metrics,
demonstrating its effectiveness.

## Full Text


<!-- PDF content starts -->

Biomedical Question Answering via Multi-Level Summarization on a Local
Knowledge Graph
Lingxiao Guan
University of Michgian
lxguan@umich.eduYuanhao Huang
University of Michigan
hyhao@umich.eduJie Liu
University of Michigan
drjieliu@umich.edu
Abstract
In Question Answering (QA), Retrieval Aug-
mented Generation (RAG) has revolutionized
performance in various domains. However,
how to effectively capture multi-document re-
lationships, particularly critical for biomedical
tasks, remains an open question. In this work,
we propose a novel method that utilizes propo-
sitional claims to construct a local knowledge
graph from retrieved documents. Summaries
are then derived via layerwise summarization
from the knowledge graph to contextualize
a small language model to perform QA. We
achieved comparable or superior performance
with our method over RAG baselines on several
biomedical QA benchmarks. We also evaluated
each individual step of our methodology over a
targeted set of metrics, demonstrating its effec-
tiveness.
1 Introduction
Multi-Document Question Answering (MDQA)
via Large Language Models (LLMs) has improved
the experience and effectiveness of users by gen-
erating relevant, informed responses. A key ap-
proach in MDQA is Retrieval-Augmented Gener-
ation (RAG) (Lewis et al., 2020), in which LLMs
are augmented with documents retrieved from es-
tablished corpora providing up-to-date information.
RAG reduces hallucinations due to the retrieved
contexts grounding the answer, and providing ac-
cess to domain-specific knowledge in knowledge
bases. These benefits are even more important in
the domain of BioMedical Question Answering
(QA), as it has much higher requirements in terms
of what data is used and the outputs of models.
Many questions require the integration of informa-
tion from multiple documents, as healthcare profes-
sionals may use model outputs to inform decisions
such as medical diagnoses of rare and complex
conditions.However, extracting and leveraging multi-
document relationships remains an underexplored
challenge. Directly placing retrieved documents
into an LLM’s context window fails to capture
inter-document connections, thereby limiting the
model’s ability to aggregate evidence and leading
to inadequate reasoning. This issue is particularly
pronounced in Biomedical QA, where accurate an-
swers often require synthesizing multiple medical
concepts across diverse documents. While prior
work has introduced RAG-based techniques to mit-
igate the limitations of finite context windows and
noisy, inconsistent information in LLMs (Gao et al.,
2023b), these methods often overlook the deeper
relationships between retrieved documents. A re-
cent approach aimed at addressing this problem
in unstructured knowledge base involves hierarchi-
cal summarization of semantically related chunks
(Sarthi et al., 2024). However, because documents
can share topics while differing in semantic focus ,
this method risks missing critical connections and
producing non-diverse summaries. For example, a
document discussing breast cancer outcomes and
another on its genetic causes may both be relevant
to a query about breast cancer, but may not be
identified as related due to their differing seman-
tic focal points. Knowledge Graphs (KG) have
been proposed as an alternative for their represen-
tation of explicit relationships between concepts.
However, existing KG-based methods exhibit key
limitations. Some approaches require access to an
entire offline knowledge corpus (Edge et al., 2024),
limiting their applicability to dynamically retrieved
documents. Others construct graphs dynamically
but suffer from explicit information loss during
graph traversal for retrieval (Wang et al., 2024).
Therefore, there is a need for a method that effec-
tively extracts and utilizes relevant multi-document
relationships from dynamically updated knowledge
bases, enabling more comprehensive reasoning in
Biomedical QA and beyond.
1arXiv:2504.01309v1  [cs.CL]  2 Apr 2025

To remedy this, we propose utilizing the con-
struction of a knowledge graph to underlay lay-
erwise document summarization as an alterna-
tive. We utilize propositional claims to represent
information and facilitate handling conflicting and
noisy claims extracted from unstructured docu-
ments we retrieve. The knowledge graph structure
constructed from these propositional claims cap-
tures relationships beyond semantic similarity. Fi-
nally, our approach performs layerwise graph sum-
marization around several key claims of interest to
comprehensively capture multi-document relations
and fit them into a limited context window.
This method utilizes the properties of decontex-
tualized claims in the knowledge graph structure
and layerwise topological summarization to capture
explicit and implicit relationships between entities
in the documents, thus having a more comprehen-
sive context to provide to LLMs. We evaluate each
part of our methodology, and compare our method
to traditional RAG retrieval baselines on several
biomedical QA datasets, achieving comparable or
superior performance over all baselines.
Our approach makes three main contributions.
•We introduce a novel method of structuring
the information from retrieved documents
as propositional claims in local knowledge
graphs.
•We introduce a technique utilizing layerwise
topological graph summaries of key claims
in this local knowledge graph as context for
LLM QA tasks.
•We evaluate our approach on a comprehensive
set of benchmarks, testing both the properties
of the intermediate results of the method and
the final accuracy on several datasets.
2 Related Works
2.1 Retrieval Augmented Generation
Information Retrieval methods have long been used
for general question answering tasks, including
biomedical QA (Jin et al., 2022). RAG extends
these methods for use with LLMs, allowing for
the integration of large external corpora into pre-
trained Language Models. The naive RAG ap-
proach was first introduced in Lewis et al. (Lewis
et al., 2020), and has since been followed by many
follow-up refinements (Gao et al., 2023b). A num-
ber of works have been conducted on the applica-
tion of RAG in biomedical QA, such as MedRAGwhich retrieves documents from a variety of cor-
pora (Xiong et al., 2024), BioMedRAG which
trains the retriever for improved retrieval of medi-
cal documents (Li et al., 2024), and Self-BioRAG
which uses on-demand retrieval and reflection to-
kens to select the best evidence (Yu et al., 2023),
among many others (Liu et al., 2024; Zhou et al.,
2023), which tend to take the strategies used in gen-
eral domain RAG and adapt then to the biomedical
domain. While naive RAG provides benefits for
QA tasks, a number of follow up works extend it.
2.2 Summarization
Summarization of the input contexts is one method
by which the retrieved documents can be further
processed to better suit downstream tasks. Sum-
marization, by its nature, is capable of condensing
input documents into a format that retains relevant
information while using less input tokens. RAP-
TOR (Sarthi et al., 2024) attempts to use hierarchi-
cal summarization of input documents to capture
both locally relevant information and distant in-
terdependencies through multiple summarization
levels. However, its reliance on semantic similar-
ity means that it may miss explicit, non-semantic
connections. Long-context summarization meth-
ods like MemTree (Rezazadeh et al., 2024) follow a
similar vein of using embedding similarity to group
contextual information, and thus suffer from the
same problem of missing explicit connections.
2.3 RAG with Knowledge Graphs
More recently, there has been a line of work at-
tempting to perform community-based summariza-
tion on generated knowledge graphs. They partition
the knowledge graph into modular parts, either via
communities as with Graph Rag (Edge et al., 2024),
or into hierarchical tags as in MedGraphRAG (Wu
et al., 2024). While these methods are able to cap-
ture more multi-document relationships, they per-
form their method on the entire offline retrieval
corpus rather than dynamically retrieved online in-
put documents. This requires a high upfront cost
and a different level of granularity compared to our
method, while also requiring additional effort to
update their graph summaries with new informa-
tion.
Alternatively, retrieved documents can be turned
into a graph structure for additional processing.
Several works have opted for this method, with
many using semantic similarity of text chunks or a
combination of semantic similarity and structural
2

information (Wang et al., 2024; Guo et al., 2024)
to construct the graph. These methods that use
semantic similarity are unable to capture explicit
connections, and even with explicit connections
formed by structural relationships the retrieval uses
agents that can miss information outside of the
explicitly returned paths. Our method utilizes the
explicit connections formed from knowledge graph
RDF formats and does layerwise summarization
to capture these connections and relevant pieces of
information.
3 Methods
Approach Overview : Our approach handles the
problem of processing and connecting information
from multiple retrieved documents to solve biomed-
ical questions. At its core, our methodology takes
in a biomedical question, a set of retrieved docu-
ments, and possible multiple choice answers before
using a language model to process the documents
and determine the correct answer. More formally,
given an input biomedical question q, a set of an-
swer options A, and a corpus of dynamically up-
dated unstructured documents D, a language model
Lis used to select the correct answer a∈A. The
output should satisfy three requirements:
1.Comprehensively identify and connect multi-
document relations.
2.Efficiently use the limited context window of L.
3.Reduce noise while preserving relevant informa-
tion.
Our method seeks to improve the extraction and
presentation of relevant information and multi-
document relations from unstructured documents
by the addition of layerwise graph summarization,
and the process can be seen in Figure 1. It pro-
ceeds by first extracting decontextualized claims
from each d∈D(Section 3.1), using the entities in
these claims to build a graph (Section 3.2), before
summarizing the content in the graph topologically
into several key claims that are provided to Lto
solve the question (Section 3.3).
3.1 Relation Extraction
The relation extraction step transforms retrieved
unstructured documents into propositional claims
and associated RDF Triples. This allows us to
transform complex technical documents into
atomic pieces of information that can be reliably
connected and analyzed in later steps. It is subdi-
vided into three sections, including retrieving theunstructured documents from several knowledge
bases, extracting and decontextualizing proposi-
tional claims from these unstructured documents,
before extracting RDF Triples from each of these
claims to facilitate graph construction.
Retrieval : To accurately answer biomedical ques-
tions, our approach gathers relevant information
from several knowledge bases. For a given input
question q, the input question is first preprocessed
into a better suited retrieval query to retrieve rele-
vant documents d∈D. As mentioned in previous
work (Ma et al., 2023), question rewriting helps
to better match the input question to the model’s
understanding, improving the clarity and effective-
ness of the questions.
Our method follows with HyDE candidate an-
swer generation (Gao et al., 2023a). This approach
generates a candidate answer using the model L
given the input question qand associated answer
options. The candidate answer includes domain-
specific terminology and context from the model’s
internal parameters, which the retriever uses along-
side the input question and answer options to cap-
ture documents that share similar concepts and im-
prove the retrieval results.
The final query with the rewritten question,
answer options, and candidate answer is used
to retrieve text chunks d∈Dfrom four knowl-
edge bases: Statpearls documents, Simple
Wikipedia, Medical Textbooks, and PubMed
Abstracts/Fulltext articles. Further details on the
retrieval corpora and the retrieval process can be
found in Section 4.1.1.
Claim Extraction : To connect information across
documents, documents are broken down into con-
cise and independent pieces. From the retrieved
text chunks d∈D, we extract propositional claims
C={c1, c2, ..., c n}with the model L. These
propositional claims must be
•Atomic: includes only a single statement that
cannot be broken down, and
•Decontextualized: fully understandable on its
own with no unresolved entity references.
A previous work (Chen et al., 2024) has found that
using this chunking modality improves retriever
performance. The decontextualization process is
especially important for our task, because in later
reranking and summarization our approach needs
to not only understand the meaning of each claim
3

Figure 1: Overview of the proposed layerwise summarization method. Relation extraction: Load in documents
with a retriever, break documents into claims, break claims into triples. Graph creation: Build local graph with
triples and denoise. Graph summarization: Summarize the graph layerwise with the top re-ranked claims as the
roots. The final summaries are provided to a model as context for downstream QA tasks.
in isolation, but also its importance to the input
question without additional context.
Triple Extraction : Once each claim c∈Cis ex-
tracted, we need to prepare them for addition to
the local graph Gthat will be constructed. We
assume that the decontextualization and claim ex-
traction process has given us atomic propositional
claims, with each one only having one key relation.
Our method involves extracting a single RDF triple
(subj, pred, obj )from each claim c. This triple
format captures the relationship pred between the
two entities subj andobj.
3.2 Graph Creation
The graph creation step processes the RDF Triples
and claims from the relation extraction step into a
local graph structure that captures the relationships
between pieces of information. This is crucial for
identifying multi-document interactions that are
not apparent from individual claims. It can also be
subdivided into several key steps, including dedu-
plication of RDF triple entities, constructing the
local graph, and denoising irrelevant information
from the KG.
Deduplication : While our claim extraction phase
(Section 3.1) resolves coreferences to the same en-
tities, the entities in each RDF triple can still have
multiple possible representations. This means that
minor character differences can result in different
entity nodes for the RDF triples, breaking apart
connections.
Deduplication of entities in the RDF triples is
performed to ensure that all references to the same
concept point towards the same node in the graph.
Our approach uses the cosine similarity of eachentity’s embeddings rather than character-based
Levenshtein distance because medical entities
that have only minor character differences can
have entirely different meanings. Specifically,
embeddings are placed into the same cluster
with Unweighted Average Linkage Clustering
(UPGMA) (Sokal and Michener, 1958), using the
average embedding similarity of each entity in
the cluster. Using a similarity threshold of 0.8,
we map all entities within a cluster to an arbitrary
label in the cluster. This restores the connections
between claims that discuss similar concepts.
Graph Construction : After deduplication, the
processed RDF triples and claims are used to
construct the graph G. Each node in the graph is
an entity from the RDF triples (subj, pred, obj ),
one of the subj orobjentities. Each edge in the
graph represents the relationship between the two
entities and includes the representative claim cand
relevancy score s. Each of the relevancy scores are
calculated using a reranker Raccording to how
much the edge claim addresses the input question
q. All of the edges are treated as undirected in
further processing, and allow for multiple edges
between two entities.
Denoising : While nodes that represent the same
concept have been grouped together, there remains
the possibility that noise has been added to G.
Noise is a common problem in knowledge graph
creation, and it can mislead the LLM when an-
swering the input question or crowd out relevant
information from the model’s context window.
We consider a claim cto be noise if it is both
• Irrelevant to the input question directly, or
•Not useful in connecting claims that are impor-
4

tant to answering the question.
While using a reranker can determine claim rele-
vance, it is insufficient to determine whether the
claim helps in connecting important information.
Following (Boer et al., 2024), our approach evalu-
ates claim importance using its connections in G.
Specifically, our method provides each claim’s 1-
hop connected neighbors as context for a model
Lto determine the claim’s relevance to the input
question.
3.3 Graph Summarization
The final graph summarization stage of our
methodology condenses the content in Ginto
several key claims of interest to capture the most
relevant information for answering the input
question. This stage involves first obtaining
claims of interest as entry points into the graph,
performing layerwise summarization to capture
relevant connections between the claims, before
generating focused summaries around the claims
of interest to obtain a context that is provided to the
model for answering the input biomedical question.
Obtaining Claims of Interest : Due to the num-
ber of documents under consideration, we need to
focus on only the most relevant information. Our
method selects several key claims of interest K
fromG, which will provide a diverse set of entry
points into the graph. Our approach starts with the
top 10 ranked claims in the graph, because individ-
ually they were determined by the reranker to be
the most relevant for answering the question.
We want to determine each claim of interests’
potential to produce meaningful summaries for our
later layerwise summarization phase. Since claims
closer to the claims of interest will be given more
weight in the final summaries, each claim of in-
terest’s 1-hop neighboring claims are examined.
These neighboring claims are used as context to
generate test summaries , and the ranks of these test
summaries are used to again rerank the claims of
interest.
To ensure that there is sufficient diversity in per-
spectives, these reranked claims are filtered. As
adjacent claims should produce similar summaries,
we merge all claims that are a 1-hop neighbor of a
higher ranked claim into that higher ranked claim,
removing it from K. This returns a shorter, more
focused list of claims as our final set for K, improv-
ing the efficiency of the subsequent procedureswhile retaining coverage of relevant information.
Figure 2: Layerwise summarization method overview.
For a given claim of interest, the graph is organized into
layers based on the distance of each connected claim
from the claim of interest. The summarization process
begins from the furthest layer, moving inwards. For
each layer claims are summarized using the previously
generated summaries of their connected claims in lower
layers. This process ensures that path information and
multi-document relationships are preserved while filter-
ing out irrelevant information in the final summaries.
Layerwise Summarization : The process of lay-
erwise summarization for each claim of interest
involves organizing its connected component in G
into layers based on the distance each claim is from
it.
Definition 1 (Layer) .Given a claim of interest k
in graph G, theith layer consists of all claims that
are exactly i-hop away from kinG.
The summarization process starts from the out-
ermost layer and proceeds inwards. For each claim
in the current layer, our method considers the sum-
maries of connected claims one layer below. These
summaries from connected claims are again sum-
marized to create the current claim’s own summary.
Each claim is processed only once and only uses
summaries from already processed claims, ensur-
ing that there are no cycles. This occurs layer by
layer until the claim of interest is reached. An ex-
ample visualization of our method can be seen in
Figure 2.
This layerwise summarization is used because
it has three key benefits. First, it is capable of
capturing all the information in the local connected
component, including both the direct content and
path-based information. This is important for
understanding multi-document relations between
different medical concepts. Second, our layerwise
processing of claims will inherently filter out
irrelevant content, complementing our denoising
step. Finally, this method places emphasis on
claims closer in Gto the claims of interest,
which naturally prioritizes more topically relevant
information in the final summaries.
5

Summary Generation : The final summarization
for each claim of interest captures the information
from its entire connected component in G, but is fo-
cused on the perspective around the specific claim.
While these claims share common topics due to
their high relevance to the input question, each fi-
nal summary should differ in content due to them
emphasizing their local relationships. The final out-
put of this method is a concatenation of all of the
summaries in the order of their relevance rankings,
from the highest to lowest. This set of summaries
is provided as contexts for an LLM to perform QA
tasks.
4 Experiment Settings
4.1 Datasets
4.1.1 RAG Datasets
The retrieval corpora in this work are a combination
of medical and general corpora, including
• Simple Wikipedia (Foundation),
•Medical textbooks from MedQA (Jin et al.,
2020),
•PubMed abstracts and full text articles from
GLKB (Huang et al., 2024), and
• StatPearls articles1.
Simple Wikipedia provides general knowledge,
medical textbooks provide foundational concepts,
Statpearls documents provide detailed medical in-
formation, and PubMed abstracts/fulltext articles
provide research findings. This combination is
to improve the coverage of topics our method
can retrieve relevant information for, inspired by
MedRAG (Xiong et al., 2024).
4.1.2 Evaluation Datasets
Our evaluation datasets include the test sets of
PubMedQA (Jin et al., 2019), MedQA (Jin et al.,
2020), and the MMLU clinical topics datasets
(Hendrycks et al., 2021) (Anatomy, Clinical Knowl-
edge, College Biology, Professional Medicine, Col-
lege Medicine, and Medical Genetics). For val-
idation and ablation tests, we utilize a combina-
tion of the validation sets of the individual MMLU
datasets which we term MMLU Validation.
4.2 Model Settings
We use the Mistral-7B-Instruct-v0.1 model for both
construction and summarization of the graph for
all evaluations (Jiang et al., 2023). For experiments
1https://www.statpearls.com/that involved LLM-as-a-judge capabilities, we used
Mixtral-8x7B-Instruct-v0.1 (Jiang et al., 2024). For
Reranking, we used bge-reranker-v2-gemma, and
for embedding we used bge-large-en-v1.5 (Li et al.,
2023). We use the en_core_sci_scibert spacy model
(Neumann et al., 2019) due to its better perfor-
mance on scientific tasks compared to general do-
main spacy models, and the neural entity recogni-
tion pipeline to extract entities.
4.3 QA Baselines
We compared the QA accuracy of our Layerwise
method with four alternative measures.
•Baseline: The prompt only includes the input
question and answer options, relying on the
model’s parametric knowledge to answer the
questions.
•Rewrite: We use question rewriting to retrieve un-
structured documents from the knowledge bases,
before adding them with reranking to the model’s
context window until the context limit is reached.
•HyDE (Gao et al., 2023a): We use the HyDE
query generation method to use the question,
answer options, and candidate answer to re-
trieve unstructured documents from the various
knowledge bases. The retrieved documents are
reranked and added to the model’s context win-
dow up to the context limit.
•RAPTOR (Sarthi et al., 2024): We use the HyDE
query generation method to retrieve documents.
The RAPTOR process is used to produce a con-
text for each question for QA.
We evaluated these methods on the benchmarks
discussed in the Evaluation Datasets (Section 4.1.2)
Section.
4.4 Component Level Analysis
We evaluated the capabilities of each of the
individual components in our methodology to
determine their effectiveness and validate key
assumptions over our MMLU Validation dataset.
These included the modules of relation extraction,
graph creation, and graph summarization as can be
seen from Figure 1.
Relation Extraction : The goal of the relation
extraction phase is to turn the retrieved documents
into decontextualized claims with associated RDF
Triples. The desired properties of these claims and
triples are that each claim is self-contained and
the meaning of the source documents are retained.
6

Thus, for relation extraction, we evaluated the
methodology’s ability on three key criteria, namely:
•Decontextualization: fraction of explicit entity
references over all entity references extracted
with SpaCy from each claim.
•Preservation of semantic meaning: the semantic
similarity between the embedding of the input
document and the concatenated form of all of the
extracted claims.
•Key claim extraction: the fraction of key claims
extracted from the retrieved documents using a
judge LLM that are retained in the output sum-
maries.
To assess our method, we compare it with several
alternatives.
•Single stage (Our Method): Extracts the claims
from the documents and decontextualizes them
in a single prompt.
•Two stage: Performs the extraction and decontex-
tualization separately, could potentially improve
the performance of the decontextualization but
has a drop in efficiency.
•Direct triples: Extracts RDF triples instead of
claims, improves the efficiency of the overall
pipeline due to skipping the claim extraction step.
•Pairs relations: Extracts the entities first before
extracting the relations between entities, a more
traditional KG creation method.
Graph Creation : The goal of the graph creation
phase is to have the RDF Triples from the relation
extraction phase connect related claims. The com-
munities in the graph should make sense upon con-
sideration of their relevance to the input question.
Thus, for graph creation, we tested the methodol-
ogy’s ability to have high quality graph communi-
ties centered around key claims.
We compared the summaries produced from
subgraphs and semantic communities around the
claims of interests we obtained from the graph sum-
marization stage (Section 3.3).
•Subgraph communities: we consider all 1-hop
connections around the entities in the claims of
interests, using the claims on these connections
to produce summaries for each claim of interest.
•Semantic communities: we retrieve all claims
that have a similarity above the cosine similarity
threshold of 0.8 with the claims of interests, and
use these claims to produce summariesThe metric’s score for an index with either
method is calculated by obtaining the relevance
score relative to the input question of the concate-
nation of all produced summaries of that index. As
the actual relevance scores produced by rerankers
are only useful to compare the two methods, we
record which of the two methods had a higher
score for each index.
Graph Summarization : The goal of graph sum-
marization is to ensure that the summaries pro-
duced by the summarization method are useful for
the input question. The requirements for these
summaries are that the contents should be relevant ,
have little hallucinations , and have information
from various sources .
Thus, for graph summarization, we further test
three different metrics:
•Faithfulness (hallucination rate): fraction of
claims in the output summaries that are supported
by the input documents.
• Answer relevance: fraction of claims relevant to
the input question in the output summaries.
•Score diversity: fraction of input documents that
are included in the final summaries.
We compared our layerwise approach with the
summaries produced from the graph communities
formed from the 1-hop subgraphs around claims of
interests and those produced with semantic com-
munities around the claims of interests. These are
the same summaries we used in the Graph Creation
component analysis.
5 Results
5.1 QA Accuracy
The results of the evaluation of our methodology
compared with various RAG baselines can be seen
in Table 1. The largest average improvement is
over the Rewrite method and the smallest over
RAPTOR. Other than the PubMedQA dataset, our
method has comparable or improved performance
over the baselines on all datasets. For PubMedQA,
we believe that the slight drop in performance is
due to insufficient denoising in our created graph,
which we plan on addressing in future work. In
all, these results imply that our method has allowed
the model to more thoroughly analyze the provided
data, therefore more effectively synthesizing infor-
mation from the retrieved documents.
7

Approach MMLU-
V*MMLU-
AMMLU-
CBMMLU-
CMMMLU-
PMMMLU-
MGMMLU-
CKPMQA MedQA
Baseline 0.55 0.46 0.57 0.46 0.51 0.6 0.54 0.50 0.44
Rewrite 0.47 0.44 0.45 0.38 0.48 0.62 0.43 0.59 0.46
HyDE 0.55 0.47 0.47 0.45 0.57 0.65 0.46 0.60 0.50
RAPTOR 0.63 0.54 0.63 0.55 0.60 0.75 0.63 0.66 0.50
Layerwise 0.63 0.55 0.65 0.56 0.62 0.78 0.65 0.56 0.54
*MMLU prefixes denote: V-Validation, A-Anatomy, CB-College Biology, CM-College Medicine, PM-Professional Medicine,
MG-Medical Genetics, CK-Clinical Knowledge
Table 1: Comparison of accuracy scores across various BioMedical QA approaches. Results show the performance
on MMLU Clinical Topics, PubMedQA, and MedQA benchmarks. Our Layerwise method shows consistent
improvements over baseline methods, with comparable or superior performance across the non-validation datasets.
The MMLU prefixes denote different subject areas, as noted under the table.
Approach Ref Score Sem.
Sim.Claim
Ret.
single_stage 0.941 0.901 1.0
two_stage 0.946 0.903 1.0
direct_triples 0.971 0.865 1.0
pairs_relations 0.994 0.815 1.0
Table 2: Comparison of Relation Extraction methods
across three metrics. Ref. Score measures decontextu-
alization ability, Sem. Sim. measures preservation of
original meaning, and Claim Ret. measures preservation
of key information. Higher scores indicate improved
performance on the individual metrics, ranging from
0-1.0. Results demonstrate the trade-off between entity-
based and claim-based approaches, with our single stage
method achieving a balanced performance while main-
taining good computational efficiency.
5.2 Component Level Analyses Results
We obtained results for each of our relation ex-
traction, graph creation, and graph summariza-
tion components. Our relation extraction evalu-
ation compared four methods across three met-
rics: decontextualization quality (Ref Score), se-
mantic preservation of original documents’ mean-
ings (Sem. Similarity), and key claim retention
(Claim Ret.), with the results shown in Table 2.
The entity-based claim extraction approaches (di-
rect_triples and pairs_relations) achieved higher
reference tracking scores (0.994 and 0.971) com-
pared to claim-based methods (single_stage 0.941,
two_stage 0.946) due to their focus on extracting
explicit entities which naturally avoids leaving un-
resolved references. However, the claim-based
methods still achieved strong semantic preserva-
tion performance (0.901 and 0.903 vs 0.865 andApproach Summary Score Wins
Graph Communities 59.35%
Semantic Communities 40.65%
Table 3: Comparison of relevance scores between graph
and semantic-based summarization. Results show the
percentage of times each method produced summaries
with a higher relevance score, and demonstrate the graph
summary’s superior ability to capture relevant informa-
tion from the input documents.
Approach Faithfulness Relevancy Source
Diversity
Layerwise 0.9569 0.8414 0.9647
Semantic 0.9706 0.8604 0.9170
Subgraph 0.9453 0.7938 0.9356
Table 4: Evaluation of three summarization approaches
across faithfulness (hallucinations), relevancy (rele-
vance to input question), and source diversity (multi-
document relations) metrics. Scores range from 0-
1.0. Results demonstrate the Layerwise summarization
method’s ability to maintain a high faithfulness and
relevancy while achieving superior source diversity.
0.815). This advantage suggests that retaining the
sentence structure of the claims results in lower
information loss of semantic meaning. All of our
methods achieved a perfect key claim retention
score. These results support our usage of the single
stage approach with its comparable decontextual-
ization and superior semantic preservation scores
compared to the entity extraction approaches, and
it achieves almost identical performance to the two
stage approach at a fraction of the computational
cost.
For the Graph Creation component, the results
8

of these methods can be seen in Table 3. The sum-
maries produced by the graph communities had a
higher relevance score to the input question com-
pared to the summaries produced by the semantic
communities 59.35% of the time. While semantic
communities are limited to capturing relationships
based on pure textual similarity, our graph construc-
tion identifies connections that may be relevant
topically yet semantically dissimilar.
For the Graph Summarization component, the
results of these metrics can be seen in Table 4. Our
layerwise summarization method achieved compa-
rable faithfulness (0.9569) and relevancy scores
(0.8414) compared to the alternative approaches
while having superior source diversity (0.9647).
The slightly lower relevancy score of our layerwise
method (0.8414) compared to semantic clustering
(0.8604) stems from the inclusion of information
in the summaries that is not directly relevant to
the question but is useful for connecting relevant
statements. This design decision enables more
comprehensive answers but lowers the total num-
ber of claims that are directly relevant to the input
question in the summaries. The consistently high
faithfulness values (>0.94) for all three alternative
methods confirms that none of them suffer from
significant hallucinations. Our method achieving a
strong faithfulness (0.9569) balanced with superior
source diversity, means that it can integrate infor-
mation from many of the retrieved documents with
little hallucination in the produced summaries.
6 Conclusion
We introduce a novel method for retrieval based
BioMedical QA tasks that utilizes propositional
claims to construct a local knowledge graph from
retrieved documents, before constructing sum-
maries derived via layerwise summarization from
the graph to contextualize a small language model
to produce the final decisions. We achieved com-
parable or superior performance with our method
over RAG baselines on several biomedical bench-
marks, demonstrating its effectiveness. We per-
formed additional experiments covering the inter-
mediate stages of our pipeline, showcasing the ro-
bustness of our approach.
Moving forward, we plan to expand our ap-
proach in several directions. First, we plan to
enhance our graph creation step’s denoising ca-
pabilities through dynamic thresholding and im-
proved conflict resolution. Second, we intend tointegrate LLM reasoning methods and tool use into
our pipeline, potentially improving the QA perfor-
mance and expanding the number of applicable
downstream tasks. Third, we plan to evaluate our
approach’s generalizability with a wider selection
of models, datasets, and benchmarks. Finally, we
seek to improve the computational efficiency of our
method, focusing on reducing the number of LLM
calls for our graph construction and summarization.
References
Derian Boer, Fabian Koch, and Stefan Kramer. 2024.
Harnessing the power of semi-structured knowledge
and llms with triplet-based prefiltering for question
answering. CoRR , abs/2409.00861.
Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu,
Kaixin Ma, Xinran Zhao, Hongming Zhang, and
Dong Yu. 2024. Dense X retrieval: What retrieval
granularity should we use? In Proceedings of the
2024 Conference on Empirical Methods in Natural
Language Processing, EMNLP 2024, Miami, FL,
USA, November 12-16, 2024 , pages 15159–15177.
Association for Computational Linguistics.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024. From local to global: A
graph RAG approach to query-focused summariza-
tion. CoRR , abs/2404.16130.
Wikimedia Foundation. Wikimedia downloads.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023a. Precise zero-shot dense retrieval without rel-
evance labels. In Proceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers) , pages 1762–1777,
Toronto, Canada. Association for Computational Lin-
guistics.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo,
Meng Wang, and Haofen Wang. 2023b. Retrieval-
augmented generation for large language models: A
survey. CoRR , abs/2312.10997.
Tiezheng Guo, Chen Wang, Yanyi Liu, Jiawei Tang, Pan
Li, Sai Xu, Qingwen Yang, Xianlin Gao, Zhi Li, and
Yingyou Wen. 2024. Leveraging inter-chunk interac-
tions for enhanced retrieval in large language model-
based question answering. CoRR , abs/2408.02907.
Dan Hendrycks, Collin Burns, Steven Basart, Andy
Zou, Mantas Mazeika, Dawn Song, and Jacob Stein-
hardt. 2021. Measuring massive multitask language
understanding. In 9th International Conference on
Learning Representations, ICLR 2021, Virtual Event,
Austria, May 3-7, 2021 . OpenReview.net.
Yuanhao Huang, Zhaowei Han, Xin Luo, Xuteng Luo,
Yijia Gao, Meiqi Zhao, Feitong Tang, Yiqun Wang,
9

Jiyu Chen, Chengfan Li, et al. 2024. Building a liter-
ature knowledge base towards transparent biomedical
ai.bioRxiv , pages 2024–09.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de Las Casas, Florian Bressand, Gianna Lengyel,
Guillaume Lample, Lucile Saulnier, Lélio Re-
nard Lavaud, Marie-Anne Lachaux, Pierre Stock,
Teven Le Scao, Thibaut Lavril, Thomas Wang, Timo-
thée Lacroix, and William El Sayed. 2023. Mistral
7b.CoRR , abs/2310.06825.
Albert Q. Jiang, Alexandre Sablayrolles, Antoine
Roux, Arthur Mensch, Blanche Savary, Chris Bam-
ford, Devendra Singh Chaplot, Diego de Las Casas,
Emma Bou Hanna, Florian Bressand, Gianna
Lengyel, Guillaume Bour, Guillaume Lample,
Lélio Renard Lavaud, Lucile Saulnier, Marie-
Anne Lachaux, Pierre Stock, Sandeep Subramanian,
Sophia Yang, Szymon Antoniak, Teven Le Scao,
Théophile Gervet, Thibaut Lavril, Thomas Wang,
Timothée Lacroix, and William El Sayed. 2024. Mix-
tral of experts. CoRR , abs/2401.04088.
Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng,
Hanyi Fang, and Peter Szolovits. 2020. What dis-
ease does this patient have? a large-scale open do-
main question answering dataset from medical exams.
arXiv preprint arXiv:2009.13081 .
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William
Cohen, and Xinghua Lu. 2019. PubMedQA: A
dataset for biomedical research question answering.
InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the
9th International Joint Conference on Natural Lan-
guage Processing (EMNLP-IJCNLP) , pages 2567–
2577, Hong Kong, China. Association for Computa-
tional Linguistics.
Qiao Jin, Zheng Yuan, Guangzhi Xiong, Qianlan Yu,
Huaiyuan Ying, Chuanqi Tan, Mosha Chen, Song-
fang Huang, Xiaozhong Liu, and Sheng Yu. 2022.
Biomedical question answering: A survey of ap-
proaches and challenges. ACM Comput. Surv. , 55(2).
Patrick S. H. Lewis, Ethan Perez, Aleksandra Pik-
tus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih,
Tim Rocktäschel, Sebastian Riedel, and Douwe
Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. In Advances in Neu-
ral Information Processing Systems 33: Annual Con-
ference on Neural Information Processing Systems
2020, NeurIPS 2020, December 6-12, 2020, virtual .
Chaofan Li, Zheng Liu, Shitao Xiao, and Yingxia Shao.
2023. Making large language models a better founda-
tion for dense retrieval. Preprint , arXiv:2312.15503.
Mingchen Li, Halil Kilicoglu, Hua Xu, and Rui Zhang.
2024. Biomedrag: A retrieval augmented large
language model for biomedicine. arXiv preprint
arXiv:2405.00465 .Lei Liu, Xiaoyan Yang, Junchi Lei, Xiaoyang Liu, Yue
Shen, Zhiqiang Zhang, Peng Wei, Jinjie Gu, Zhixuan
Chu, Zhan Qin, and Kui Ren. 2024. A survey on
medical large language models: Technology, appli-
cation, trustworthiness, and future directions. ArXiv ,
abs/2406.03712.
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. 2023. Query rewriting for
retrieval-augmented large language models. CoRR ,
abs/2305.14283.
Mark Neumann, Daniel King, Iz Beltagy, and Waleed
Ammar. 2019. ScispaCy: Fast and Robust Models
for Biomedical Natural Language Processing. In Pro-
ceedings of the 18th BioNLP Workshop and Shared
Task, pages 319–327, Florence, Italy. Association for
Computational Linguistics.
Alireza Rezazadeh, Zichao Li, Wei Wei, and Yujia Bao.
2024. From isolated conversations to hierarchical
schemas: Dynamic tree memory representation for
llms. CoRR , abs/2410.14052.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D. Manning.
2024. RAPTOR: recursive abstractive processing for
tree-organized retrieval. In The Twelfth International
Conference on Learning Representations, ICLR 2024,
Vienna, Austria, May 7-11, 2024 . OpenReview.net.
Robert R Sokal and Charles D Michener. 1958. A statis-
tical method for evaluating systematic relationships.
Yu Wang, Nedim Lipka, Ryan A. Rossi, Alexa F. Siu,
Ruiyi Zhang, and Tyler Derr. 2024. Knowledge
graph prompting for multi-document question an-
swering. In Thirty-Eighth AAAI Conference on Artifi-
cial Intelligence, AAAI 2024, Thirty-Sixth Conference
on Innovative Applications of Artificial Intelligence,
IAAI 2024, Fourteenth Symposium on Educational
Advances in Artificial Intelligence, EAAI 2014, Febru-
ary 20-27, 2024, Vancouver, Canada , pages 19206–
19214. AAAI Press.
Junde Wu, Jiayuan Zhu, and Yunli Qi. 2024. Med-
ical graph RAG: towards safe medical large lan-
guage model via graph retrieval-augmented gener-
ation. CoRR , abs/2408.04187.
Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong
Zhang. 2024. Benchmarking retrieval-augmented
generation for medicine. In Findings of the Asso-
ciation for Computational Linguistics, ACL 2024,
Bangkok, Thailand and virtual meeting, August 11-
16, 2024 , pages 6233–6251. Association for Compu-
tational Linguistics.
Han Yu, Peikun Guo, and Akane Sano. 2023. Zero-
shot ecg diagnosis with large language models and
retrieval-augmented generation. In Proceedings of
the 3rd Machine Learning for Health Symposium ,
volume 225 of Proceedings of Machine Learning
Research , pages 650–663. PMLR.
10

Hongjian Zhou, Boyang Gu, Xinyu Zou, Yiru Li,
Sam S. Chen, Peilin Zhou, Junling Liu, Yining Hua,
Chengfeng Mao, Xian Wu, Zheng Li, and Fenglin
Liu. 2023. A survey of large language models
in medicine: Progress, application, and challenge.
ArXiv , abs/2311.05112.
11
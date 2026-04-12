# GROUNDEDKG-RAG: Grounded Knowledge Graph Index for Long-document Question Answering

**Authors**: Tianyi Zhang, Andreas Marfurt

**Published**: 2026-04-06 02:12:30

**PDF URL**: [https://arxiv.org/pdf/2604.04359v1](https://arxiv.org/pdf/2604.04359v1)

## Abstract
Retrieval-augmented generation (RAG) systems have been widely adopted in contemporary large language models (LLMs) due to their ability to improve generation quality while reducing the required input context length. In this work, we focus on RAG systems for long-document question answering. Current approaches suffer from a heavy reliance on LLM descriptions resulting in high resource consumption and latency, repetitive content across hierarchical levels, and hallucinations due to no or limited grounding in the source text. To improve both efficiency and factual accuracy through grounding, we propose GroundedKG-RAG, a RAG system in which the knowledge graph is explicitly extracted from and grounded in the source document. Specifically, we define nodes in GroundedKG as entities and actions, and edges as temporal or semantic relations, with each node and edge grounded in the original sentences. We construct GroundedKG from semantic role labeling (SRL) and abstract meaning representation (AMR) parses and then embed it for retrieval. During querying, we apply the same transformation to the query and retrieve the most relevant sentences from the grounded source text for question answering. We evaluate GroundedKG-RAG on examples from the NarrativeQA dataset and find that it performs on par with a state-of-the art proprietary long-context model at smaller cost and outperforms a competitive baseline. Additionally, our GroundedKG is interpretable and readable by humans, facilitating auditing of results and error analysis.

## Full Text


<!-- PDF content starts -->

GROUNDEDKG-RAG: Grounded Knowledge Graph Index for
Long-document Question Answering
Tianyi Zhang1Andreas Marfurt2
1getAbstract2Lucerne University of Applied Sciences and Arts
Switzerland
tianyiz0423@gmail.com, andreas.marfurt@hslu.ch
Abstract
Retrieval-augmented generation (RAG) systems have been widely adopted in contemporary large language models
(LLMs) due to their ability to improve generation quality while reducing the required input context length. In this work,
we focus on RAG systems for long-document question answering. Current approaches suffer from a heavy reliance
on LLM descriptions resulting in high resource consumption and latency, repetitive content across hierarchical levels,
and hallucinations due to no or limited grounding in the source text. To improve both efficiency and factual accuracy
through grounding, we proposeGroundedKG-RAG, a RAG system in which the knowledge graph is explicitly
extracted from and grounded in the source document. Specifically, we define nodes inGroundedKGas entities and
actions, and edges as temporal or semantic relations, with each node and edge grounded in the original sentences.
We constructGroundedKGfrom semantic role labeling (SRL) and abstract meaning representation (AMR) parses
and then embed it for retrieval. During querying, we apply the same transformation to the query and retrieve the most
relevant sentences from the grounded source text for question answering. We evaluateGroundedKG-RAGon
examples from the NarrativeQA dataset and find that it performs on par with a state-of-the art proprietary long-context
model at smaller cost and outperforms a competitive baseline. Additionally, ourGroundedKGis interpretable and
readable by humans, facilitating auditing of results and error analysis.
Keywords:knowledge graphs, retrieval-augmented generation, question answering, semantic role labeling,
abstract meaning representation
1. Introduction
Retrieval-Augmented Generation (RAG) is a tech-
nique that retrieves and incorporates additional in-
formation from an external knowledge source to
answer user queries more accurately. Its applica-
tion scenarios are diverse. For example, a chatbot
may retrieve up-to-date information released after
the model’s training cutoff to address users’ ques-
tions, or it may search over a confidential contract
provided by the user to verify specific terms and
conditions.
In this work, we focus on the second scenario,
wherealong-formdocument(bookswithpotentially
millions of words) is provided, and comprehensive,
accurate, and low-latency responses grounded in
the document are required.
Current approaches (Sarthi et al., 2024; Edge
et al., 2025) heavily rely on LLM generations to
describe book chunks, entities, relations. They
thencreateaheirarchicaltreestructure,recursively
summarizing clusters of lower-level entities. This
reliance on LLM outputs can backfire in two ways.
First, recursive summarization introduces a signifi-
cant risk of hallucination, as generated summaries
may not be strictly grounded in the original text and
can introduce content that does not exist in the
source document. Second, repeated information
across hierarchical levels leads to inefficiencies in
bothretrievalandpromptconstruction. Moreover,afixedtreestructurecannotadapttodifferentqueries
that may require aggregating different subsets or
perspectives of the original document.
To address these issues, we propose
GroundedKG-RAG1, a RAG system in which
every component of the constructed knowledge
graph is explicitly grounded in the original text,
and aggregation is query-adaptive rather than
fixed. We define aGroundedKGwhose nodes
represent entities and actions, and whose edges
encode semantic relations between entities and
actions defined by PropBank (Palmer et al., 2005),
as well as temporal relations between actions.
With this clear definition and construction, we only
need to embed the nodes while preserving the
explicit graph structure. Each node embedding
incorporates not only the node itself but also its
local graph context including the neighboring
nodes.
At query time, we retrieve nodes relevant to the
user query and use the graph structure to identify
semantically and structurally related nodes. The
corresponding grounded text spans are then re-
trieved and filtered using vector-based semantic
similarity and passed to a generative LLM for final
answer generation. The LLM used for answer gen-
eration is exchangeable as it does not depend on
our retrieval component. We use the same model
1The source code is available upon request.arXiv:2604.04359v1  [cs.CL]  6 Apr 2026

as our baseline for a fair comparison.
The proposedGroundedKG-RAGframework
avoids hallucinations introduced by recursive LLM
summarizationatmultiplehierarchicallevels. Every
step in the pipeline is human-readable and verifi-
able due to explicit grounding in the source text.
Furthermore, the framework enables efficient and
flexible aggregation of document content tailored
to different query requirements.
We evaluateGroundedKG-RAGon the Nar-
rativeQA benchmark using Exact Match, Se-
quence Match, BERTScore (Zhang et al., 2020),
and ROUGE-L F1 (Lin, 2004). Experimental
results show thatGroundedKG-RAGperforms
on par with a state-of-the art proprietary long-
context model at smaller cost, and outperforms
the GraphRAG (Edge et al., 2025) baseline across
all metrics.
2. Related Work
Several recent approaches have explored ques-
tionansweringoverlongdocumentsusingretrieval-
augmented generation (RAG). One line of work
aims to extract node–edge pairs to build a graph in-
dex for RAG systems. LightRAG (Guo et al., 2025),
HippoRAG (Gutiérrez et al., 2024) and HippoR-
AGv2 (Gutiérrez et al., 2025) create a knowledge
graph from entities and relations in the text. How-
ever,thesenodesandedgesarenotclearlydefined
and contain information at different levels, ranging
from words and phrases to paragraphs and pas-
sages, making the structure redundant and difficult
to organize. Moreover, during retrieval, these meth-
ods expand the found nodes with neighboring enti-
tiestoenablemulti-hopreasoning,acommonshort-
coming of retrieval-augmented generation models
(Li et al., 2024).
Another line of work relies on LLMs to gener-
ate descriptions of document chunks and orga-
nize them hierarchically. In RAPTOR (Sarthi et al.,
2024),leafnodescorrespondtosentencesorfixed-
size chunks (e.g., 100 tokens). These nodes are
recursively embedded and clustered using seman-
tic similarity, and parent-node summaries are gen-
erated by an LLM from their child nodes. These
summaries are then used for retrieval. This ap-
proach relies heavily on querying LLMs to build
summaries, which can lead to redundancy and hal-
lucinations. Another side effect is the use of a fixed
tree structure, which may be suboptimal when han-
dling diverse downstream questions.
GraphRAG (Edge et al., 2025) integrates knowl-
edge graph construction from the first line of work
with the hierarchical structures used in the sec-
ond approach. During the indexing stage, it con-
structs a hierarchical entity-level knowledge graph
by prompting large language models (LLMs) toextract entities from the text. These entities are
grouped into communities with the Leiden algo-
rithm (Traag et al., 2019). The communities are
then summarized by LLMs. This process of cluster-
ing and summarizing is repeated for a predefined
numberoflevels(1–4inthepaper). Inthequerying
stage, GraphRAG performs a vector search on the
embeddings of entities, relations, and communi-
ties. It can extend the results with related entities,
or even generate new sub questions (in theirdrift
search) to expand the search results.
GraphRAG is a compute-intensive and weakly
grounded2method that relies heavily on LLM gen-
erations for its search index. These shortcomings
motivated ourGroundedKG-RAG, which focuses
on grounding the knowledge graph in the source
text and is created in a resource-efficient manner.
3.GroundedKG-RAG
3.1.GroundedKGDefinition
To construct a reliable knowledge graph for reason-
ing that is readable for humans, we define a knowl-
edge graph Gas adirected graph, where nodes
represent entities and actions extracted from the
unstructured text, and edges represent semantic
relations between them.
Formally, we define the procedure ofGround-
edKGconstruction from unstructured text to struc-
tured representation asΦ : T → G. Given a doc-
ument represented as a sequence of sentences
S={s1, . . . , s N} ∈ T, we construct a grounded
knowledge graph
G= Φ(S) = (V, E),
where Vdenotes the set of nodes and E⊆V×
R ×Vdenotes the set of directed relational edges.
Each node is represented as a dictionary of at-
tributes:
v:{node_id,label,texts,node_type,
grounded_texts} −→Values.
Specifically, for each nodev∈V,
v(node_id)∈string,
v(label)∈string,
v(texts)∈list of strings,
v(node_type)∈ {entity,action},
v(grounded_texts)∈list of strings.
Here,labelis the core concept (e.g., "watch");
textscontains textual mentions (e.g., "digital
2Extracted entities, relationships, communities, their
descriptions, and importance scores are all generated
and evaluated by LLMs without verifiable grounding.

watch");grounded_textscontains sentences in the
document that the node is grounded to.
Similarly, each edge is represented as a dictio-
nary of attributes:
e:{source_node,target_node,edge_role,
edge_type,grounded_texts} −→Values.
Specifically, for each edgee∈E:
e(source_node)∈V,
e(target_node)∈V,
e(edge_role)∈String,
e(edge_type)∈ {action–entity,action–action},
e(grounded_texts)∈List[String].
wheresource_nodeandtarget_nodeare the
source and target node_ids of the edge;edge_role
is the semantic role of the relation, e.g. A0;
grounded_textscontains sentences in the docu-
ment that the node is grounded to.
3.2.GroundedKGConstruction
GroundedKGfromSRLparses.SemanticRole
Labeling (SRL) is a parsing task that captures the
“who did what to whom, when, where, and how”
structure of a sentence by identifying predicates
and assigning semantic roles to their arguments,
which describe how each argument participates in
an event.
For example, in the sentence“Peter’s mother
gave a dose of camomile tea to Peter.”SRL iden-
tifiesgaveas the predicate,Peter’s motheras the
agent (A0),a dose of camomile teaas the theme
(A1), andto Peteras the recipient (A2).
We leverage SRL to parse sentences and con-
struct ourGroundedKG. For each predicate, we
consideritsassociatedcomponents,includingcore
arguments(A0–A3)aswellastemporalandspatial
arguments when available. We first create the
graph nodes: each predicate is represented as an
action node, while all arguments are represented
asentity nodes. We then add edges by connecting
each predicate to its arguments viaentity–action
relations, for example: (A1, give, a dose
of camomile tea, entity–action, <ref-
erence to source text>) . Finally, for all
actions within a sentence, we introduce directed
action–action relationsto encode their temporal
order. For the previous example, Peter’s mother
first makes the tea before giving it to Peter, so the
actiongivefollowsmake: (next, make, give,
action–action, <reference to source
text>).
GroundedKGfrom AMR parses.Abstract
MeaningRepresentation(AMR)isasemanticpars-
ing task that maps a natural language sentence toa rooted, directed acyclic graph encoding its core
semanticmeaning,abstractingawayfromsyntactic
variation. In an AMR graph, nodes correspond to
concepts,includingactionsandentities,andedges
represent semantic relations between these con-
cepts.
For example, in the sentence“Peter’s mother
gave a dose of camomile tea to Peter.”AMR identi-
fiesgaveastherootaction( give-01 inPropBank),
with directed edges to:
•mother(A0) as a noun concept which is a per-
son with relations to another person concept
Peter,
•teaasanounconceptasroleA1withquantifier
doseand modifiercamomile,
•Peteras the person concept (A2).
Similar to SRL graph construction, we consider
all the concepts including actions and entities. We
first create nodes: each action is represented as
anaction node, while all other associated entities
are represented asentity nodes, and we add
modification words to the node text attribute, for
example: “(tea, [camomile tea, a dose
of camomile tea], tea, <reference to
source text>)” . We then link nodes with
edges by connecting each action to its associated
entities viaentity–action relations, for example:
“(A1, give-01, tea, entity–action,
<reference to source text>)” . Finally,
for all actions within a sentence, we introduce
directedaction–action relationsto encode their
temporal order, for example: “(next, make-01,
give-01, action–action, <reference
to source text>)”.
3.3. Embedding Techniques
AfterconstructingtheGroundedKG,wecreateour
search index by embedding each node into a vec-
tor representation that captures both its semantic
meaning and its context within the graph structure.
We explore three variants of node embedding:ba-
sicnodeembedding,averageneighborembedding,
andattention-based neighbor embedding.
Basic Node Embedding.We use a pretrained
embeddingmodel fembedtoencodethenodename
andnode textinto vector representations. The
embedding of a nodevis defined as:
Embed(v) =
α fembed (vname) + (1−α)1
ppX
i=1fembed (v(i)
text),
where vdenotes a node in theGround-
edKG, vname∈stris the node name, vtext=

{v(1)
text, . . . , v(p)
text}is the set of associated textual de-
scriptions, and α∈[0,1]is a weighting hyperpa-
rameter.
Average Neighbor Embedding.Beyond the ba-
sic node embedding that relies solely on a node’s
name and text, we incorporate the average em-
beddings of its neighboring nodes to capture local
graph context. The resulting node embedding is
computed as:
NeighborEmbed(v) =
βEmbed(v) + (1−β)1
qqX
j=1Embed(v j),
where {v1, . . . , v q}are the neighboring nodes of
v, and β∈ [0,1]is a weighting hyperparameter.
This formulation equally weights contextual infor-
mation from connected nodes, such as the actions
performed by an agent or the participants involved
in an event.
Attention-Based Neighbor Embedding.To ac-
count for varying importance among neighboring
nodes, we further introduce an attention-based
neighbor embedding. Instead of uniformly aver-
aging neighbor embeddings, we compute attention
weights based on cosine similarity between node
embeddings. The attention score between node v
and its neighborv jis defined as:
Attn(v, v j) = softmax j 
Embed(v)·Embed(v j)
where ·denotes the dot product, and vj∈
{v1, . . . , v q}. The final node embedding is then
computed as:
AttentionEmbed(v) =
βEmbed(v) + (1−β)qX
j=1Attn(v, v j) Embed(v j).
This attention-based neighbor embedding high-
lights the most relevant contextual information ac-
cording to semantic similarity, such as salient ac-
tions performed by an agent or the primary partici-
pants involved in an event.
3.4. Retrieval Techniques
In the querying stage, our goal is to retrieve a sub-
set of query-relevant content Sselected ⊆Swith high
recall. Specifically, we aim to select grounded texts
that are accurate, non-redundant, and that do not
omit relevant evidence required for answering thequery. To this end, we design three retrieval tech-
niques to extract and filter relevant content.
Forallretrievaltechniques,wefirstparsetheuser
queryusingthesamegraphconstructionprocedure
Φdefined above. Each query Qis converted into a
query graph
Gq= Φ(Q) = (V q, Eq).
Each query node is then embedded following the
respective node embedding method described in
Section 3.3.
Basic Node-Based Text Retrieval.For each
node u∈V qin the query graph Gq, we compute
its cosine similarity with all nodes v∈Vin the pre-
viously constructed knowledge graph GGroundedKG .
We then retrieve the Top- Kmost similar nodes
(K= 10in our experiments) and extract their
grounded texts.
Formally, we define the similarity function as
sim(v, u) = cos 
Embed(v)·Embed(u)
,
v∈V, u∈V q.
Foreachquerynode ui∈Vq,theTop- Kretrieval
is defined as
TopK(u i) = arg topK
v∈Vsim(v, u i).
The grounded texts associated with the retrieved
nodes are then collected as
Sselected (ui) =[
vij∈TopK(u i)vij.grounded_texts.
Finally, the overall selected text set is defined as
Sselected =[
ui∈VqSselected (ui), Sselected ⊆S.
Text Filter with Vector Similarity.In addition to
basic node-based retrieval, we further filter the se-
lected texts using vector similarity at the text level,
as is commonly done in contemporary RAG sys-
tems. Specifically,wecomputethecosinesimilarity
between the query representation qand each se-
lected text si∈Sselected, and retain the Top- Ktexts
with similarity scores higher than a thresholdτ:
VectorSim τ(q) ={s i∈Sselected |sim(q, s i)≥τ}.
TextFilterwithRetrievalCount.Weadditionally
introduce a salience-based filtering strategy that
measures the importance of each grounded text by
counting how frequently it is selected by different
retrieved nodes. Intuitively, if a grounded text sis
associated with multiple selected nodes, it is more
likely to be relevant to the query.

Formally, for a grounded text si∈Sselected, its
salience score is defined as
RetCount(s i) =X
v∈VselectedI
si∈v.grounded_texts
,
where Vselected =S
u∈V qTopK (u)and I[·]is the in-
dicator function.
4. Experiments
4.1. Dataset
NarrativeQA.NarrativeQA (Kočiský et al., 2017)
is a large reading comprehension benchmark de-
signedtoevaluatedeepunderstandingandreason-
ing over long texts. It consists of 1,572 full-length
stories from books and movie scripts, the split for
books are 548 train, 58 validation and 177 test.
Annotators wrote 46,765 question–answer pairs
(around 30 question-answer pairs for each book)
based on human-written summaries rather than on
the raw text, which encourages models to perform
global, integrative reasoning rather than superficial
span extraction. Each query requires synthesizing
information distributed throughout entire narratives,
necessitatingcomprehensionofcharacters,events,
and their relations across extended context spans.
Answers are generally short but often not found
as direct spans in the stories, reflecting the task’s
emphasis on full-content abstraction.
In this experiment, we select three books from
the NarrativeQA training split, covering a diverse
range of text lengths: The Tale of Peter Rab-
bit (∼5,000 words), The Phantom of the Opera
(∼90,000words),andRobinsonCrusoe( ∼120,000
words).
4.2. Metrics
We evaluate model performance using a combina-
tionofexact-matching,sequence-level,andseman-
tic similarity metrics, capturing both surface-form
accuracy and semantic equivalence between pre-
dicted and reference answers.
Exact Match (EM).EM measures whether the
predictions exactly contains the reference answers
after standard normalization (cleaning up whites-
pace) and lowercasing. This metric is strict and
assigns full credit only when the predicted answer
contains an identical segment of the ground truth.
Sequence Match (SM).SM evaluates whether
the predicted answer contains the reference se-
quence at the token level. Unlike Exact Match, it
assigns full credit when predicted answer (e.g. his
shoes and his jacket) contains correct reference to-
kensequences(e.g. hisshoesandjacket), thereby
providing a softer measure of correctness for an-
swers that are lexically similar but not identical.ROUGE-L.ROUGE-L (Lin, 2004) measures the
longest common subsequence (LCS) between the
predicted and reference answers. It captures both
precision and recall of overlapping subsequences
without requiring contiguous matches, making it
effective for evaluating content preservation in free-
form or abstractive generation tasks.
BERTScore.BERTScore (Zhang et al., 2020)
computes semantic similarity between predicted
and reference answers by aligning contextualized
token embeddings from a pretrained transformer
model (e.g., BERT). Unlike lexical overlap met-
rics, BERTScore accounts for paraphrasing and
semantic equivalence, providing a robust measure
of meaning-level similarity.
4.3. Experimental Settings
Optimal Settings.The best-performing variant of
GroundedKG-RAGusesgraphconstructionbased
on AMR parses, basic node embedding, K= 10
most similar nodes, and no additional text filters
from vector similarity or retrieval counts.
TextSegmentation.Documentsaresegmented
using the LangChain recursive character text split-
ter into chunks of 5,000 tokens with zero overlap,
following paragraph boundaries. Each chunk is fur-
ther segmented into sentences using spaCy with
theen_core_web_smmodel.
Coreference Resolution.Coreference reso-
lution is applied to each sentence using spaCy
for named entity detection and fastcoref for
coreference detection. Entities are grouped and
grounded to the original sentence. Each sentence
is stored as a tuple (sent_index, origi-
nal_sentence, normalized_sentence,
coref_modified_sentence)after processing.
Sentence Parsing.Semantic parsing is
performed on each coreference-resolved sen-
tence using either a semantic role labeling
(SRL) parser or an abstract meaning represen-
tation (AMR) parser. For SRL parsing, we use
spaCy for verb predicate identification and cu-
kairos/propbank_srl_seq2seq_t5_large
for argument extraction. For AMR parsing, we
use amrlib for AMR graph extraction together with
Penman for graph parsing.
Graph Construction.We employ networkx li-
brary forGroundedKGconstruction and pyvis for
storing the graph in JSON and html formats.
Embeddings.Nodes in the constructed graph
are embedded using the lightweight sentence em-
bedding model all-MiniLM-L6-v2 . For stability,
the vector is normalized after each step, includ-
ing node embedding, aggregated neighbor embed-
dings, combined node neighbor embeddings.
Retrieval-Augmented Generation.Queries
are processed using the same pipeline as docu-
ments. Relevant sentences are retrieved based on

graph matching and provided as context to OpenAI
modelsgpt-5-2025-08-07 andgpt-5-nano-
2025-08-07 for question answering. We did not
observe a big difference in the two model’s answer
qualities, and therefore selected gpt-5-nano for our
further experiments.
Hardware.All experiments are conducted using
asingleNVIDIAT4orL4GPUaccessedonGoogle
Colab, and the OpenAI API with batch processing
is used for answer generation.
5. Results
We evaluate ourGroundedKG-RAGby comparing
with different baselines and ablating our design
choices.
Baseline Comparison.Our main results are
shown in Table 1. We compare ourGroundedKG
with a closed-book LLM (without any context,no
context), an open-book LLM (offering the full book
content as context,full context), and GraphRAG.
Since the NarrativeQA questions are based on
copyright-free books from Project Gutenberg, we
assume that both the books and potentially also
the NarrativeQA question-answer pairs have been
part of the proprietary LLM’s training data. The
no contextbaseline establishes the performance
of the answer generation model when only retriev-
ing (book text or memorized NarrativeQA answers)
from the model weights.
Acrossthethreebooks,ourperformanceismuch
higher than theno contextbaseline. Forfull con-
text, we achieve three percent improvement in both
BERTScoreandRougeF1onPeterRabbit,from59
to 62 and 32 to 34. For Phantom of the Opera and
Robinson Crusoe,GroundedKG-RAG’s Rouge-L
F1 is relatively comparable to the full content with
only 1/3 of its total length. The BERTScore drops
from 62 to 56 and 55 to 46 compared to the full
context, showing that longer books are still more
challenging for the much bigger knowledge graph
we built for these books. Compared to GraphRAG
we achieve similar performance on BERTScore,
62 compared to 63, but better performance on
ROUGE-L, 34 compare to 8, indicating the differ-
ence in groundedness of the answers. Due to the
highnumberofLLMcallsperformedbyGraphRAG,
we limit our evaluation of the model to experiments
on Peter Rabbit.
Sentence Parsing.Comparing knowledge
graph construction from SRL parses with AMR
parses,weseeinTable2thatAMRparsesimprove
on SRL parses in both BERTScore and ROUGE-L
(52 vs. 58 and 30 vs. 34). This may be the result
of the finer granularity level allowing more accurate
node matches between query nodes andGround-
edKGnodes. For example, in SRL parsing nodes,
every token contributes equally to the node embed-ding, thusPeter’s fathermay have a higher cosine
similarity score withPeter’s motherthanthe four
little kids’ father. In AMR parsing, the core tokens
determine the node embedding, thus the embed-
ding offatherfor bothPeter’s fatherandthe four
little kid’s fatherwill lead to a match.
TopK.We test the choice of TopK. K= 5
receives lower BERTScore than K= 10. The
ROUGE-L is the same in Table 2. Although the
true grounded sentence has a higher probability
in the first several nodes, we assume selecting
more nodes allows more grounded sentences to
be incorporated in the context, leading to better
performance in the rare case that the gold answer
is a reflection of multiple sentences. For instance,
the nodePetercan be matched to a similar node
Peter’s jacket, which is not in the very top of the
matched nodes when compared toPeter,Peter’s
sisteretc., yet contains the correct answer to the
question.
Embedding.We ablate the performance of
our three embedding techniques on Peter Rab-
bit: basic node embedding, average neighbor em-
bedding, and attention-based neighbor embed-
ding. The three settings achieve similar scores
on BERTScore. The original node embedding
achievesthebestscoreinBERTScorewith62com-
pared to 60 for others. The attention-based neigh-
bor embedding gets the best score in ROUGE-L
with36. Theaverageneighborembeddinggetsthe
lowest scores. This is because the outliers in an
averageofembeddingscanhaveabiginfluenceon
the node embedding, modifying up to 50% of the
top-10 retrieval results. The attention-based neigh-
bor embedding is more stable compared to the
average neighbor embedding and only changes
10-20% of the retrieved results (often in the last
ranks). Since, the contextual embedding, which
considers the graph structure and the neighbors of
anodedoesnotexhibitanobviousgainonthetask,
we use the basic node embedding as our primary
setting.
Retrieval.We consider three different settings
in the retrieval stage: basic node grounded_text
retrieval, filtering with sentence vector cosine sim-
ilarity, and filtering with sentence retrieval counts.
For sentence cosine similarity, the peformance
drops 4 percent on BERTScore and 1 percent
on ROUGE-L compared with grounded_text re-
trieval forGroundedKG-RAG, and 1 percent in
BERTScore and 5 percent in ROUGE-L when us-
ing SRL parses and K= 5on Peter Rabbit (see
Table 2). For filtering based on sentence retrieval
counts, the BERTScore drops from 46 to 43 and
F1 drops from 26 to 23 on Robinson Crusoe (see
Table 4 in the appendix). We conjecture that filter-
ing can remove the gold grounded sentences for
some of the questions, leading to worse question

Model Exact match Sequence match Bertscore RougeL F1
Peter Rabbit: 300 nodes, 700 edges
No context1616 49 22
Full context 13 16 59 32
GraphRAG1620638
GroundedKG-RAG16 236234
The Phantom of the Opera: 25k nodes, 56k edges
No context 17 17 49 25
Full context34 37 62 36
GroundedKG-RAG34 375636
Robinson Crusoe: 33k nodes, 80k edges
No context 13 13 25 8
Full context 36 3655 31
GroundedKG-RAG40 4346 30
Table 1: Main result ofGroundedKGRAGcompared to baselines on our three selected books.
Model Exact match Sequence match Bertscore RougeL F1
No context1616 49 22
Full context 13 16 59 32
GroundedKG-RAGwith SRL andK= 510 16 52 30
GroundedKG-RAGwith SRL,
K= 5, and VectorSim (τ= 0.2) 10 13 51 25
GroundedKG-RAG16 23 6234
GroundedKG-RAGwithK= 513 20 58 34
GroundedKG-RAGwith VectorSim (τ= 0.2) 13 16 58 33
GroundedKG-RAG(basic Embed)16 23 6234
NeighborEmbed (β= 0.8)16 2360 33
AttentionEmbed (β= 0.5) 13 20 6036
AttentionEmbed (β= 0.8)16 236036
Table 2: Ablation of ourGroundedKG-RAGon Peter Rabbit.
answering performance. During error analysis, we
find that for the question“What did Peter do after
arriving home?”, the sentence“Peter arrives home
exhaustedly.”willhaveahighsimilarityscore,while
the true grounded text“Peter gets some food and
runstobedbecauseheistooexhausted.”receives
a similarity score below the threshold because of
the smaller number of matching tokens between
thesentenceandthequestion. Thesamehappens
for the retrieval count filter.
6. Error Analysis
We distinguish between (1) KG construction, (2)
retrieval and (3) answer generation errors. The
discussion of mapping“the four little kids’ father”
to the father concept is an example of an error of
the first kind (see Section 5). This works better
in graphs built from AMR parses than from SRL
parses. The second kind is a failure to retrieve
relevantnodesfromagivenquery(seeanexample
in Table 5 in the appendix). The third kind of erroroccurs when the answer generation model gets
distracted by seemingly relevant information in the
context, cannot handle the length of the context,
or ignores the context and therefore the instruction
(see third error type in Table 5).
7. Graph Case Study
We present a simple example to illustrate the cre-
ation of ourGroundedKGfrom SRL and AMR
parses in Table 3. We can clearly observe that
AMR-based graph grounded coreference-resolved
entities (“it” has been resolved to “some camomile
tea” in the second clause) into the same node, in
this casetea, while SRL created a duplicate node
for it. The same happens with the two mentions
of Peter (“Peter” and “to Peter”). The AMR graph
avoids duplicate node creation and improves node
matching during retrieval.

Input Text: Peter’s mother put Peter to bed and made some camomile tea; and Peter’s
mother gave a dose of some camomile tea to Peter!
AMR Graph: SRL Graph:
(a / and
:op1 (p / put-01
:ARG0 (p2 / person
:ARG0-of (h / have-rel-role-91
:ARG1 (p3 / person
:name (n / name
:op1 "Peter"))
:ARG2 (m / mother)))
:ARG1 p3
:ARG2 (b / bed))
:op2 (m2 / make-01
:ARG0 p2
:ARG1 (t / tea
:mod (c / camomile)
:quant (s / some)))
:op3 (d / dose-01
:ARG0 p2
:ARG1 p3
:ARG2 t))Predicate: put
ARG-0: Peter’s mother
ARG-1: Peter
ARG-2: to bed
Predicate: made
ARG-0: Peter’s mother
ARG-1: some camomile tea
Predicate: gave
ARG-0: Peter’s mother
ARG-1: a dose of some camomile tea
ARG-2: to Peter
GroundedKGwith AMR:GroundedKGwith SRL:
Table 3: Example comparing AMR and SRL graphs.
8. Conclusion
In this work, we propose theGroundedKG-
RAGframework, where each node and edge in
theGroundedKGis directly extracted from and
grounded in the source document at the sentence
level. The nodes represent the entities and actions,
and the edges represent the temporal or seman-
tic relation between them. For theGroundedKG
construction, we experimented with two different
document parsing methods: SRL and AMR. Due
to better semantic concept identification in AMR, it
achieves higher scores on NarrativeQA than SRL.
InourGroundedKG,weembedthenodestoincor-
porate the contexts. The graph’s edges serve to re-
latenodes’temporalandsemanticrelationship. We
evaluatethreetypesofembeddingtechniques,and
findthatincludingneighboringnodes’informationin
a node’s embedding does not significantly improve
retrieval results. We also experiment with filtering
retrieved grounded texts through either cosine sim-
ilarity with the query embedding or retrieval counts,but neither filtering improves evaluation scores. Er-
roranalysisshowsthatthesefiltersremovethegold
sentence in a few evaluation examples.
In future work, we plan to further improve our
GroundedKG-RAGin the following directions: (1)
there is still room for improvement in the node
matching, so different approaches to the graph
structure, node/edge composition, used relations,
and embedding techniques can be tried. (2) Even
though we were not successful with our experi-
ments in filtering out irrelevant source sentences,
webelievethattheanswergenerationmodelwould
profit from a less fragmented and sometimes con-
tradictingcontext,givinganopportunityforbetterfil-
tering techniques. (3)GroundedKG-RAG’s scores
improvingoverthefullcontextbaselinefortheshort
Tale of Peter Rabbit, but BertScore is lagging be-
hind on longer books shows that the large knowl-
edge graphs constructed for these books may ben-
efit from better graph retrieval techniques.

9. Acknowledgements
We thank getAbstract for the support and partial
funding of this project.
10. Bibliographical References
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness,
andJonathanLarson.2025. Fromlocaltoglobal:
A graph rag approach to query-focused summa-
rization.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and
Chao Huang. 2025. LightRAG: Simple and fast
retrieval-augmented generation. InFindings of
the Association for Computational Linguistics:
EMNLP 2025, pages 10746–10761, Suzhou,
China. Association for Computational Linguis-
tics.
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu,
Michihiro Yasunaga, and Yu Su. 2024. Hipporag:
neurobiologically inspired long-term memory for
large language models. InProceedings of the
38thInternationalConferenceonNeuralInforma-
tion Processing Systems, NIPS ’24, Red Hook,
NY, USA. Curran Associates Inc.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025. From RAG to
memory: Non-parametric continual learning for
large language models. InForty-second Interna-
tional Conference on Machine Learning.
Tomáš Kočiský, Jonathan Schwarz, Phil Blunsom,
Chris Dyer, Karl Moritz Hermann, Gábor Melis,
and Edward Grefenstette. 2017. The narrativeqa
reading comprehension challenge.
Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu
Mei, and Michael Bendersky. 2024. Retrieval
augmented generation or long-context LLMs? a
comprehensive study and hybrid approach. In
Proceedings of the 2024 Conference on Empir-
ical Methods in Natural Language Processing:
Industry Track, pages 881–893, Miami, Florida,
US. Association for Computational Linguistics.
Chin-Yew Lin. 2004. ROUGE: A package for auto-
matic evaluation of summaries. InText Summa-
rization Branches Out, pages 74–81, Barcelona,
Spain. Association for Computational Linguistics.
Martha Palmer, Daniel Gildea, and Paul Kingsbury.
2005. The Proposition Bank: An annotated cor-
pusofsemanticroles.ComputationalLinguistics,
31(1):71–106.Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D. Man-
ning. 2024. Raptor: Recursive abstractive pro-
cessing for tree-organized retrieval. InInterna-
tional Conference on Learning Representations
(ICLR).
V.A. Traag, L. Waltman, and N.J. van Eck. 2019.
From louvain to leiden: guaranteeing well-
connected communities. InSci Rep 9, 5233.
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q.
Weinberger, and Yoav Artzi. 2020. Bertscore:
Evaluating text generation with bert.
A. Question Answering Prompt
We used the following prompt format to answer
questions with and without context.
With content: Please answer the ques-
tion based on the following content:
Content:< content >
Question:< question >
Answer:
Without content: Please answer the
question based on your memory of the
book< book_name >:
Question:< question >
Answer:
B. Sentence Count Ablation
A small ablation on Robinson Crusoe for the text
filter with retrieval count is shown in Table 4.
C. Error Analysis
An error analysis with examples for the three types
of errors we encounter is shown in Table 5.

Model Exact match Sequence match Bertscore RougeL F1
Robinson Crusoe
No context 13 13 25 8
Full context 36 3655 31
GroundedKG-RAG40 4346 30
GroundedKG-RAGwith RetCount≥230 30 43 23
Table 4: Comparing AMR with sentence count of 2 or more on Robinson Crusoe.
Tyep I Error
Question What happened to the young rabbits’ father?
Matched Nodeto the young rabbits’ father [’with four little rabbits mother’, ’The Tale Of Peter Rabbit’, ’from
Peter cousin, little Benjamin Bunny’]
happen [’happen_112’, ’come_100’, ’come_113’]
Grounded Text text_0-3, text_0-6, text_0-20, text_0-21, text_0-27, text_0-37, text_0-40, text_0-42
Predicted Answer The text does not mention the father of the young rabbits.
Gold Answer Mrs. McGregor baked him into a pie.
Solution ImproveGroundedKGconstruction for node matching
Type II Error
Question What does Peter have for dinner after getting back home?
Matched NodePeter [’Peter’, ’Peter back’, ’First Peter’]
for dinner: [for supper, eat_26, eat_27]
have: [have_142, not have_66, get_131]
Peter: [Peter, Peter back, First Peter]
back home: [home, away, go back_109]
get: [get_131, not get_18, give_2]
Grounded Texttext_0-1, text_0-8, text_0-13, text_0-14, text_0-15, text_0-16, text_0-17, text_0-18, text_0-
20, text_0-21, text_0-23, text_0-24, text_0-25, text_0-27, text_0-28, text_0-29, text_0-30,
text_0-31, text_0-33, text_0-35, text_0-36, text_0-37, text_0-39, text_0-40, text_0-41, text_0-
42, text_0-43, text_0-44, text_1-0, text_1-1, text_1-2, text_1-4, text_1-5, text_1-7, text_1-9,
text_1-11
text_1-9 His mother put him to bed and made some camomile tea;; and she gave a dose of
some camomile tea to Peter!
text_1-11 But– Flopsy, Mopsy and Cottontail had bread and milk and blackberries for supper.
Predicted Answer Bread and milk and blackberries.
Gold Answer Chamomile tea
Solution Filter out distracting results and use a stronger question answering model to distinguish the
correct answer from misleading ones.
Type III Error
Question Where does Peter see his lost clothing?
Matched NodePeter [’Peter’, ’Peter back’, ’First Peter’]
see [’see_116’, ’look_30’, ’look_71’]
clothing [’his jacket’, ’the little jacket and the shoes’, ’underneath’]
lose [’lose_41’, ’lose_42’, ’lose_136’]
Grounded Texttext_0-13, text_0-14, text_0-15, text_0-16, text_0-17, text_0-18, text_0-19, text_0-20, text_0-
21, text_0-23, text_0-24, text_0-25, text_0-26, text_0-27, text_0-28, text_0-29, text_0-30,
text_0-31, text_0-33, text_0-35, text_0-36, text_0-37, text_0-39, text_0-40, text_0-41, text_0-
42, text_0-43, text_0-44, text_1-0, text_1-1, text_1-2, text_1-3, text_1-4, text_1-5, text_1-7,
text_1-9
text_1-3 Mr. McGregor hung up the little jacket and the shoes for a scare-crow to frighten the
blackbirds.
Predicted Answer In the tool-shed.
Gold Answer On McGregor’s scarecrow
Solution Better filtering and using a question answering model that can reason over long and similar
contexts.
Table 5: Error analysis with an example for each type of error.
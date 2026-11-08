# QueryFlow: Multi-Hop Graph Retrieval with Adaptive Query Evolution

**Authors**: Manzong Huang, Chenyang Bu, Yi He, Jiapu Wang, Xindong Wu

**Published**: 2026-11-03 10:14:30

**PDF URL**: [https://arxiv.org/pdf/2611.04625v1](https://arxiv.org/pdf/2611.04625v1)

## Abstract
Multi-hop retrieval-augmented generation (RAG) supports Web information seeking and fact verification by recovering complementary evidence across related pages within a limited retrieval budget. Scoring evidence independently against the original question can miss supporting units whose relevance emerges through intermediate facts. Iterative retrieval incorporates these facts into subsequent queries, while graph retrieval explicitly models relations among evidence units. Yet fixed-query graph retrievers update entity representations without adapting the query condition to accumulated relational context. We view multi-hop evidence recovery as the coupled evolution of evidence representations and query conditions, with accumulated context informing both what the retriever represents and how it searches for complementary evidence. QueryFlow realizes this perspective by recurrently integrating query-conditioned summaries of graph context into a shared query state, which controls boundary inputs and relation features for subsequent propagation. Across four representative multi-hop question answering (QA) datasets, QueryFlow achieves macro Recall@10 (R@10), Full Recall@10 (FR@10), and LLM-Acc gains of 4.98, 8.40, and 8.0 percentage points over representative baselines, respectively.

## Full Text


<!-- PDF content starts -->

QueryFlow: Multi-Hop Graph Retrieval with Adaptive Query
Evolution
Manzong Huang
Key Laboratory of Knowledge
Engineering with Big Data (the
Ministry of Education of China)
School of Computer Science and
Information Engineering
Hefei University of Technology
Hefei, China
manzonghuang@mail.hfut.edu.cnChenyang Bu∗
Key Laboratory of Knowledge
Engineering with Big Data (the
Ministry of Education of China)
School of Computer Science and
Information Engineering
Hefei University of Technology
Hefei, China
chenyangbu@hfut.edu.cnYi He
Department of Data Science
College of William and Mary
Williamsburg, Virginia, USA
yihe@wm.edu
Jiapu Wang
Nanjing University of Science and
Technology
Nanjing, China
jiapu.wang@njust.edu.cnXindong Wu∗
Key Laboratory of Knowledge
Engineering with Big Data (the
Ministry of Education of China)
School of Computer Science and
Information Engineering
Hefei University of Technology
Hefei, China
xwu@hfut.edu.cn
Abstract
Multi-hop retrieval-augmented generation (RAG) supports Web
information seeking and fact verification by recovering comple-
mentary evidence across related pages within a limited retrieval
budget. Scoring evidence independently against the original ques-
tion can miss supporting units whose relevance emerges through
intermediate facts. Iterative retrieval incorporates these facts into
subsequent queries, while graph retrieval explicitly models rela-
tions among evidence units. Yet fixed-query graph retrievers up-
date entity representations without adapting the query condition
to accumulated relational context. We view multi-hop evidence
recovery as the coupled evolution of evidence representations and
query conditions, with accumulated context informing both what
the retriever represents and how it searches for complementary
evidence. QueryFlow realizes this perspective by recurrently in-
tegrating query-conditioned summaries of graph context into a
shared query state, which controls boundary inputs and relation
features for subsequent propagation. Across four representative
multi-hop question answering (QA) datasets, QueryFlow achieves
macro Recall@10 (R@10), Full Recall@10 (FR@10), and LLM-Acc
gains of 4.98, 8.40, and 8.0 percentage points over representative
baselines, respectively.
CCS Concepts
•Information systems →Information retrieval;•Computing
methodologies→Knowledge representation and reasoning.
∗Corresponding authors: Chenyang Bu and Xindong Wu.
WWW 2027 submission, Dublin, Ireland
2027.Keywords
GraphRAG, graph neural networks, graph-based retrieval, multi-
hop question answering, web content mining
ACM Reference Format:
Manzong Huang, Chenyang Bu, Yi He, Jiapu Wang, and Xindong Wu. 2027.
QueryFlow: Multi-Hop Graph Retrieval with Adaptive Query Evolution.
InProceedings of The ACM Web Conference 2027 (WWW 2027 submission).
ACM, New York, NY, USA, 14 pages.
1 Introduction
Retrieval-augmented generation (RAG) grounds language-model
answers in external evidence [ 28]. In Web content mining, it sup-
ports complex information seeking and fact verification across
related pages [ 23,46]. Multi-hop questions make this evidence re-
quirement more demanding because their answers depend on linked
facts distributed across pages about related entities [ 16,47,58]. An-
swering these questions requires retrieving evidence units, such
as documents, passages, or chunks, that jointly support the nec-
essary reasoning steps. A collection of individually relevant units
can still omit a fact needed to complete the reasoning chain. Multi-
hop retrieval is fundamentally an evidence-set recovery problem,
with the goal of recovering the complete supporting set within a
given retrieval budget. By improving multi-hop question answer-
ing over interconnected Web content, our work aligns with the
Web Mining, Multimedia and Multilingual Content Analysis
track.
Recovering a complete supporting set requires more than match-
ing individual evidence units to the original question. Single-round
dense retrievers learn semantic matches between questions and
candidate evidence [ 25], but independent scoring can miss com-
plementary units whose relevance becomes apparent only after
arXiv:2611.04625v1  [cs.IR]  3 Nov 2026

WWW 2027 submission, 2027, Dublin, Ireland Huang et al.
Q: Where was the author of The Hobbit born ?
J.R.R. 
TolkienBirthplace?born inRound 1
Round 2
The HobbitJ.R.R. 
Tolkienwritten byContextJ.R.R. 
TolkienBirthplace?born inRequired
(1) Fixed query  state (2) Updated query state
Q1: Where was J.R.R. Tolkien born?chosen 
The 
HobbitJ.R.R. 
Tolkienwritten byFantasy 
novel
Middle-
earthBritish
writer
LiteratureThe Lord of 
the Rings
BloemfonteinNext-hopThe 
HobbitJ.R.R. 
Tolkienwritten byFantasy 
novel
Middle-
earthBritish
writer
LiteratureThe Lord of 
the Rings
BloemfonteinRemaining information need
The 
HobbitJ.R.R. 
Tolkienwritten byFantasy 
novel
Middle-
earthBritish
writer
LiteratureThe Lord of the 
Rings
BloemfonteinQ0: Where was the author of The Hobbit born? q0q0
q1
Related, but 
insufficientCompleted the 
evidence chain
Figure 1: A fixed query state continues to favor evidence di-
rectly related to the original question. Updating the state with
accumulated context guides retrieval toward complementary
evidence, helping complete the support set. 𝑄1illustrates the
updated state’s meaning.
intermediate facts are found. Iterative retrieval addresses this limi-
tation by conditioning subsequent searches on retrieved passages
or intermediate reasoning [ 48,56]. In these approaches, evidence
dependencies are conveyed through retrieval history, leaving en-
tity relations across documents implicit. Graph-based RAG makes
these connections explicit through entities and relations [ 7,61],
while neural graph retrievers learn to integrate connected evidence
through query-conditioned message passing [ 31,35]. A central
question in learned graph retrieval is how accumulated relational
context should guide subsequent evidence acquisition. Fixed-query
designs update entity representations while retaining the original
query condition [ 35]. As illustrated in Figure 1, identifying the au-
thor ofThe Hobbitshifts the remaining information need toward
that person’s birthplace, while a fixed query can continue to favor
book-related evidence. These designs accumulate relational con-
text in entity representations without an explicit feedback path for
that context to revise the query condition governing subsequent
propagation.
The query state should evolve with the evidence representations
it guides. Intermediate entities and relations can clarify what re-
mains to be retrieved, so accumulated graph context should inform
both the representation of evidence and the condition used to ex-
plore further evidence. Coupling these updates brings relational
propagation and query adaptation into the same retrieval process,
allowing newly established context to guide the recovery of missing
support.
We introduceQueryFlow, a graph retrieval framework that
couples recurrent query evolution with evidence propagation to
support multi-hop question answering. It maintains a shared query
state alongside entity representations that accumulate relational
context through message passing. Between successive propagation
layers, query-conditioned attention summarizes all entity represen-
tations, making context distributed across the graph available to the
query update. A gated recurrent transition integrates this summary
with the preceding query state to revise the retrieval condition.
Boundary feedback injects the updated state at question-linkedentities, while relation feedback modulates the features used to
transform neighboring messages. These interfaces change both the
context injected into the graph and how evidence propagates across
its relations, closing the feedback path from accumulated evidence
to subsequent retrieval. Finally, entity scores are aggregated into a
document ranking that supplies evidence for answer generation.
On four representative multi-hop QA datasets, QueryFlow
achieves macro R@10, FR@10, and LLM-Acc gains of 4.98, 8.40, and
8.0 percentage points over representative baselines, respectively.
Improvements in evidence coverage and complete recovery are
accompanied by higher downstream QA reasoning accuracy,
supporting the value of complete evidence retrieval for multi-hop
question answering.
Our contributions are threefold.
•We frame multi-hop graph retrieval as coupled evolution
of evidence representations and query conditions, treating
accumulated relational context as feedback for subsequent
evidence acquisition.
•We proposeQueryFlow, which couples recurrent query up-
dates with graph propagation through boundary and relation
feedback.
•Experiments on four multi-hop QA benchmarks show gains
over representative baselines in evidence recovery and an-
swer accuracy.
2 Related Work
Relevance-based retrieval.RAG integrates external evidence into
language-model training or generation [ 3,14,21,28]. Dense rep-
resentations and token interactions improve matching between
questions and evidence [ 19,25,27,39]. Precomputed passage repre-
sentations support efficient search, while token-level interactions
retain fine-grained matching signals within each candidate passage.
Better query representations, evidence fusion, and hierarchical in-
dexing further improve evidence access and use [ 10,20,40,42].
In single-round matching, retrieved evidence does not update the
scoring condition, so complementary support with weak direct rel-
evance can be missed. QueryFlow complements relevance scoring
with feedback from accumulated relational context.
Iterative and adaptive retrieval.Iterative retrievers use re-
trieved passages, condensed facts, or rewritten queries to guide
later searches [ 36,38,59]. MDR re-encodes the question with ear-
lier passages [ 56], while Baleen condenses retrieved passages into
facts for later hops [ 26]. Intermediate generation and reasoning
also supply feedback [ 41,48], while adaptive approaches control
when and how to retrieve based on generation state or question
difficulty [ 2,22,24,43]. In retrieval loops driven by passages or
reasoning, feedback passes through text history, leaving entity rela-
tions across documents implicit. QueryFlow brings this adaptation
into graph propagation, using intermediate entity context to update
the shared retrieval condition.
Graph-based retrieval.Knowledge graph research also stud-
ies graph completion, which predicts missing entities or relations
from graph structure. Completion methods learn multi-concept and
fine-grained entity–relation representations [ 52,53], or adapt how
entity, relation, and temporal information is combined to predict
missing facts [ 49,54,55]. Broader graph representation learning

QueryFlow: Multi-Hop Graph Retrieval with Adaptive Query Evolution WWW 2027 submission, 2027, Dublin, Ireland
combines local and global signals for node classification, while at-
tributed graph clustering uses collaborative graph augmentation
and contrastive learning [ 51,60]. These tasks learn graph repre-
sentations or predict missing facts, whereas graph retrieval ranks
source documents to recover evidence supporting a question. Graph
indices and diffusion associate evidence through explicit links [ 7,11–
13,61], while subgraph selection and path exploration construct
structured context [ 15,17,30,33,34,45]. Learned relational rea-
soning supports query-conditioned propagation [ 8,50,62,63,67],
and neural retrievers learn to aggregate or select connected evi-
dence [ 31,37,44]. Learned graph retrievers differ in how relational
context conditions retrieval. GFM-RAG propagates entity relevance
under a fixed query and maps it to documents [ 35], whereas D-
RAG updates instructions from graph context for knowledge graph
question answering [ 9,65,66]. QueryFlow studies this interaction
for complete document evidence recovery, using a shared evolving
query state to control boundary inputs and relation features.
3 Method
Overview.QueryFlow organizes multi-hop evidence retrieval as
a feedback process between entity representations and a shared
query state (Figure 2). Entity propagation aggregates relational
context, while recurrent query evolution converts that context
into the condition for subsequent propagation. The evolving query
state controls both boundary inputs and relation features, allowing
accumulated context to influence how further evidence is explored.
Final entity scores are aggregated into a document ranking for
answer generation.
3.1 Evidence Graph and Query Grounding
Retrieval task and graph interface.Given a question 𝑥and a
document corpus 𝐷, the objective is to recover the set of supporting
documents𝐷∗(𝑥)⊆𝐷 within a retrieval budget 𝑘. The first𝑘doc-
uments in the output ranking form b𝐷𝑘(𝑥). We represent the corpus
by an entity graph 𝐺=(𝑉,𝐸,R) [35], where𝑉contains entities
and𝐸contains edges labeled by relation types in R. An incidence
matrix𝐴∈{ 0,1}|𝑉|×|𝐷|records associations between entities and
documents, with 𝐴𝑖𝑗=1when entity 𝑖occurs in document 𝑗. Mes-
sage passing accumulates context at entity nodes; the incidence
matrix subsequently transfers entity relevance to documents.
Query and entity initialization.A frozen text encoder pro-
duces question and relation embeddings 𝑒𝑥and𝑒𝑟. Learned linear
projections𝜙𝑞and𝜙𝑟map them to a common 𝑑-dimensional space.
Let𝑚𝑥∈{0,1}|𝑉|identify the entities linked to 𝑥. The initial query,
relation features, and entity states are
𝑞0=𝜙𝑞(𝑒𝑥), 𝑅0(𝑟)=𝜙𝑟(𝑒𝑟),
𝐻0(𝑖)=𝑚𝑥(𝑖)𝑞 0.(1)
Thus entities linked to the question start from 𝑞0, while other en-
tities start from zero. At layer 𝑙, the shared query state 𝑞𝑙−1∈R𝑑
controls the update to entity representations 𝐻𝑙∈R|𝑉|×𝑑. The en-
tity representations store context specific to each entity, while the
query supplies a shared condition for propagation. Graph topology,
document associations, and the grounding mask remain fixed as
these representations evolve.3.2 Query-Conditioned Evidence Propagation
Each layer propagates entity context using boundary inputs and
relational messages [ 63]. The shared query state 𝑞𝑙−1controls both
the representations injected at question-linked entities and the
relation features that transform neighboring messages.
Boundary feedback.The boundary input at the original ground-
ing locations is
𝐵𝑙(𝑖)=𝑚𝑥(𝑖)𝑞𝑙−1.(2)
Its support is fixed by 𝑚𝑥, while its content changes with the query
state. At later layers, this input reintroduces the context encoded by
earlier query updates, maintaining a connection between question
grounding and evolving relational information.
Relation feedback.The same state generates a gate for each
feature of the base relation embeddings,
𝛾𝑙−1=𝜎(𝑊𝑟𝑞𝑙−1+𝑏𝑟),
¯𝑅𝑙(𝑟)=𝜓𝑙 𝑅0(𝑟)⊙𝛾𝑙−1.(3)
Here𝜎is the sigmoid function, ⊙denotes elementwise multiplica-
tion, and𝜓𝑙is a layer-specific two-layer MLP with an intervening
ReLU. The gate is shared across relation types, each retaining its
own base embedding 𝑅0(𝑟). It modulates dimensions of the rela-
tion features according to the current query condition before the
layer-specific transformation.
Relational message passing.Let N(𝑖) contain pairs of source
entities and relations (𝑗,𝑟) with an edge from 𝑗to𝑖. DistMult mes-
sages [ 57] and sum aggregation combine neighboring states with
the boundary to give
𝑢𝑙,𝑖=𝐵𝑙(𝑖)+∑︁
(𝑗,𝑟)∈N(𝑖)𝐻𝑙−1(𝑗)⊙ ¯𝑅𝑙(𝑟).(4)
The next entity representation is obtained by transforming the
concatenated previous state and aggregate, followed by a residual
connection:
𝑡𝑙,𝑖=𝑊𝑙[𝐻𝑙−1(𝑖);𝑢𝑙,𝑖]+𝑐𝑙,
𝐻𝑙(𝑖)=𝐻𝑙−1(𝑖)+ReLU(LN 𝑙(𝑡𝑙,𝑖)).(5)
Here[;]denotes concatenation, and LN𝑙is layer-specific normal-
ization. Boundary feedback changes the representation injected at
the starting entities, while relation feedback changes the features
used to carry messages along edges. For 𝑙<𝐿 ,𝐻𝑙supplies the next
layer’s node input and the context for query evolution.
3.3 Graph-Conditioned Query Evolution
Query evolution summarizes 𝐻𝑙to update the shared state 𝑞𝑙. For
𝑙<𝐿 , this state conditions the next propagation layer. The readout
ablation additionally applies the same update at 𝑙=𝐿 to(𝐻𝐿,𝑞𝐿−1),
obtaining the terminal state 𝑞𝐿without further graph propagation.
Query-conditioned graph summary.We use the preceding
query state to attend to the new entity representations with weights
𝑎𝑙,𝑖=exp
𝐻𝑙(𝑖)⊤𝑞𝑙−1/√
𝑑
Í
𝑗∈𝑉exp
𝐻𝑙(𝑗)⊤𝑞𝑙−1/√
𝑑.(6)
The normalized graph summary is
𝑧𝑙=LN sum ∑︁
𝑖∈𝑉𝑎𝑙,𝑖𝐻𝑙(𝑖)!
.(7)

WWW 2027 submission, 2027, Dublin, Ireland Huang et al.
Q: Where was the author 
of The Hobbit born ?
Frozen encoderq0
D1：The Hobbit  was written 
by J.R.R. Tolkien.D1：The Hobbit  was written 
by J.R.R. Tolkien.
D2：J.R.R. Tolkien  was 
born in Bloemfontein .D2：J.R.R. Tolkien  was 
born in Bloemfontein .
D1
D2Layer 2 Layer L Layer 3 Layer 1The HobbitJ.R.R. 
TolkienBloemfontein
......
...v...
authorborn in
qL q0 U q1 U U UEntity scoringq0 HL(v)
Entity
Relevance
GNN propagation with query evolutionq0...
Entity-document links A
Current message flow Other graph edgesH1 H2 H3 HLGraph ConstructionEntity graph with query -conditioned propagation
q2q2
U  Query updateQ
BloemfonteinLLM
Entity-to-document 
readout...SoftplusD1
D3
D2
softplus( )iivs=
j ij i i
iVd A w v
=
Weighted aggregationA
1/iiw df=
1/iiw df=
D1
D2
D3
D4Top-k
...
......
qL-1...
Figure 2: Overview of QueryFlow for multi-hop evidence retrieval from interconnected document collections. A question is
grounded in an entity graph whose nodes are linked to source documents. Graph propagation and recurrent query updates
alternate across layers; the updated query state conditions the boundary inputs and relation features of the next layer. Final
entity representations are scored together with the original query, and the scores of all entities are transformed by softplus and
aggregated through associations between entities and documents using reciprocal document frequency weights. The documents
with the highest scores provide evidence for answer generation. The terminal 𝑞𝐿update is used only for the readout ablation.
The softmax ranges over all entities in 𝑉. Consequently, 𝑧𝑙depends
jointly on the current query and the graph representations, empha-
sizing context compatible with the retrieval condition. The sum-
mary provides an input of fixed width to the controller regardless
of graph size.
Gated recurrent transition.A gated recurrent unit (GRU) [ 6]
integrates𝑧𝑙with the preceding query state. An additional gate
regulates the displacement in each feature dimension toward the
candidate:
e𝑞𝑙=GRUCell(𝑧 𝑙,𝑞𝑙−1),
𝑔𝑙=𝜎 𝑊𝑔[𝑞𝑙−1;𝑧𝑙]+𝑏𝑔,
Δ𝑙=𝑔𝑙⊙(e𝑞𝑙−𝑞𝑙−1).(8)
The candidate e𝑞𝑙combines previous query information with new
graph context, while 𝑔𝑙controls how much of their difference enters
the update. A scalar 𝛼∈[ 0,1]scales this increment before query
normalization:
𝑞𝑙=(
𝑞𝑙−1, 𝛼=0,
LN𝑞(𝑞𝑙−1+𝛼Δ𝑙),0<𝛼≤1.(9)
We use𝛼=1for recurrent propagation. Setting 𝛼=0retains𝑞𝑙=
𝑞0by bypassing the transition and its normalization, providing a
fixed-query intervention within the same model. The recurrent cell,
transition gate, and relation gate share parameters across layers;
LNsumandLN𝑞have separate parameters, each shared across layers.
For𝑙<𝐿 , the updated state controls the next boundary and relation
features through Eqs. (2)and(3), completing the feedback from
graph context to subsequent propagation.
3.4 Document Readout and Learning
Entity relevance scoring.After 𝐿propagation layers, a shared
two-layer MLP with an intervening ReLU scores each entity usingits final representation and the initial query:
𝑠𝑖=MLP([𝐻𝐿(𝑖);𝑞 0]).(10)
The default model uses 𝑞0in both training and retrieval to anchor
scores to the original question. The readout ablation replaces 𝑞0
with𝑞𝐿in Eq. (10), keeping𝐻𝐿unchanged. The entity-level objec-
tive trains the query controller through its influence on subsequent
graph propagation.
Entity-to-document readout.We convert each entity logit 𝑠𝑖
into a positive contribution weight using
𝑣𝑖=softplus(𝑠 𝑖)=log(1+exp(𝑠 𝑖)).(11)
Let𝑓𝑖=Í|𝐷|
𝑗=1𝐴𝑖𝑗denote the document frequency of entity 𝑖. We
assign reciprocal frequency weights [35],
𝑤doc
𝑖=(
𝑓−1
𝑖, 𝑓𝑖>0,
0, 𝑓𝑖=0,
𝑑𝑗=∑︁
𝑖∈𝑉𝐴𝑖𝑗𝑤doc
𝑖𝑣𝑖.(12)
The readout aggregates contributions from all entities associated
with each document. For entities that occur in many documents,
reciprocal frequency weighting reduces their contribution to each
document. We rank documents by 𝑑𝑗and return the top 𝑘asb𝐷𝑘(𝑥).
Entity-level learning.Let 𝑦𝑖∈{0,1}be an entity supporting
label and𝑃={𝑖 :𝑦𝑖=1}the positive set. With 𝑝𝑖=𝜎(𝑠𝑖), the
listwise loss that assigns normalized score mass to each supporting
entity is
𝜋𝑖=𝑝𝑖Í
𝑗∈𝑉𝑝𝑗+𝜀,
LListCE=−1
|𝑃|∑︁
𝑖∈𝑃log(𝜋𝑖+𝜀).(13)

QueryFlow: Multi-Hop Graph Retrieval with Adaptive Query Evolution WWW 2027 submission, 2027, Dublin, Ireland
Here𝜀is a numerical stabilizer. We combine the listwise objective
with a weighted binary cross-entropy term that classifies individual
entities and emphasizes negatives with high scores to obtain
L𝑥=0.3L BCE+0.7L ListCE.(14)
Gradients from entity scores jointly train graph propagation and
the recurrent controller. Aggregation of entity scores into document
scores and document ranking are applied at inference. Appendix B
details the training objective and retrieval algorithm.
4 Experiments
Our experiments connect answer quality to the recovery of sup-
porting evidence. We first compare generation and retrieval perfor-
mance, then examine query evolution and its feedback interfaces
through controlled interventions. Efficiency, retrieval budgets, and
paired recovery analyses characterize the costs and conditions un-
der which these gains occur.
4.1 Experimental Setup
Datasets.We evaluate on four multi-hop QA benchmarks, namely
2WikiMultiHopQA [ 16], HotpotQA [ 58], MuSiQue [ 47], and Con-
currentQA [ 1]. Their annotated supporting documents define the
target evidence sets. Dataset and evaluation details are provided in
Appendix A.1. Model and training settings appear in Appendix A.2.
Evaluation metrics.For answer generation, we report LLM-
Acc, the fraction of answers accepted by an LLM judge, and Contain,
the fraction of generated answers containing the normalized gold
answer string. For retrieval budget 𝑘, document recall and full recall
are
R@𝑘(𝑥)=|b𝐷𝑘(𝑥)∩𝐷∗(𝑥)|
|𝐷∗(𝑥)|,
FR@𝑘(𝑥)=I[𝐷∗(𝑥)⊆b𝐷𝑘(𝑥)],(15)
where Iis the indicator function. Both metrics are averaged over
questions with nonempty support sets; FR@ 𝑘measures complete
recovery of the supporting set. We report results for each dataset
and an unweighted macro average at 𝑘=10and𝑘=30(Tables 2
and 7).
Retrieval comparisons.We compare R@ 𝑘and FR@𝑘only
for methods whose final retrieval output is a ranked list of docu-
ments. Methods returning only graph structures (e.g., ToG [ 45]) or
text chunks without a document ranking are evaluated through
downstream answer quality instead.
Baselines.Our retrieval baselines cover hierarchical retrieval
with RAPTOR [ 40], query expansion with HyDE [ 10], and late-
interaction retrieval with ColBERTv2 [ 39]. The six graph retrieval
baselines are G-Retriever [ 15], LightRAG [ 11], GraphRAG [ 7],
DALK [ 29], HippoRAG 2 [ 13], and GFM-RAG [ 35]. Appendix A
provides retrieval results for each dataset.
Answer generation.We use Qwen3.5-9B for answer generation
across all methods. Following LinearRAG [ 64], we report LLM-Acc
and Contain using the definitions above. Table 1 compares the re-
trieval methods used to supply evidence, including RAPTOR, HyDE,
ColBERTv2, G-Retriever, LightRAG, GraphRAG, DALK, GFM-RAG,
LinearRAG, Dense RAG, FastToG [ 32], Think-on-Graph (ToG) [ 45],
PathRAG [ 5], and HippoRAG 2 [ 13]. The table groups these methods
by retrieval mechanism.4.2 Main Results
Table 1 compares downstream answer quality across query-based,
iterative and structure-aware, and graph-based retrieval methods
on four benchmarks. These results address the following research
question.
RQ1.How does QueryFlow compare with existing retrieval and rea-
soning methods on downstream answer generation?
Answer quality improves over representative baselines.
QueryFlow reaches macro LLM-Acc of 0.696 and Contain of 0.642.
Relative to LinearRAG, these scores increase by 8.0 and 6.9 per-
centage points, respectively. The gains over GFM-RAG are 11.6
and 5.6 points. Both comparisons show improvements in judged
correctness and inclusion of the gold answer.
The answer gains extend across all four benchmarks.Com-
pared with LinearRAG, LLM-Acc increases by 15.0 points on 2Wiki-
MultiHopQA, 3.4 on HotpotQA, and 7.0 on both MuSiQue and
ConcurrentQA. On MuSiQue, LLM-Acc rises from 0.369 to 0.439
while Contain improves from 0.332 to 0.375. QueryFlow improves
both metrics on every dataset relative to LinearRAG and GFM-RAG,
showing that the macro gains extend across all four collections.
Semantic correctness and answer inclusion improve to-
gether.Contain gains range from 4.3 to 12.5 points over LinearRAG
and from 3.8 to 7.5 points over GFM-RAG. For multi-hop questions,
intermediate facts identify the entity or relation to which the final
answer must refer. QueryFlow feeds this context into subsequent
propagation to guide retrieval of complementary support. The gains
in complete recovery examined in RQ2 are consistent with this ex-
planation of the answer gains.
To examine the evidence available for answer generation, we
compare coverage of supporting documents and complete recov-
ery of evidence sets at 𝑘= 10and𝑘= 30. Table 2 and Figure 3
summarize the results for the following question.
RQ2.How does QueryFlow compare with existing retrieval methods
on multi-hop evidence retrieval benchmarks?
Larger gains in complete recovery.QueryFlow improves
macro FR@10 and FR@30 over GFM-RAG by 8.40 and 9.63 per-
centage points, reaching 70.38% and 77.78%, respectively (Table 2).
Support coverage also improves, with R@10 of 81.93% and R@30 of
87.12%, gains of 4.98 and 5.70 points. The larger margins in full recall
locate the main benefit in recovering every document required by
a question. Retrieving the last missing supporting document turns
partial coverage into complete support, so this objective favors com-
plementary evidence. QueryFlow lets intermediate entity context
revise the condition for later propagation, providing a mechanism
for directing retrieval toward such evidence.
Consistent gains across datasets.Across datasets at 𝑘=10,
R@10 gains over GFM-RAG range from 4.57 to 5.28 points. The
FR@10 gains are 12.60 points on 2WikiMultiHopQA, 8.11 on Hot-
potQA, 5.99 on MuSiQue, and 6.90 on ConcurrentQA (Table 7). Each
dataset exhibits a larger gain in full recall than in average support
coverage. On MuSiQue, the margin in full recall grows from 5.99
points at𝑘=10to 8.88 at 𝑘=30, extending the improvement to
the larger retrieval budget.
Complete recovery distinguishes complementary support
from average coverage.On MuSiQue, HippoRAG 2 has higher
R@10 than QueryFlow (64.42% versus 61.60%), yet QueryFlow has

WWW 2027 submission, 2027, Dublin, Ireland Huang et al.
Table 1: Answer generation across four datasets. Scores range from 0 to 1. Baselines are grouped by retrieval mechanism and
ordered within each group by average LLM-Acc. Averages are unweighted. Best in bold, second underlined .
Category Method2WikiMultiHopQA HotpotQA MuSiQue ConcurrentQA Average
LLM-Acc Contain LLM-Acc Contain LLM-Acc Contain LLM-Acc Contain LLM-Acc Contain
Query-based
retrievalDense RAG 0.235 0.228 0.592 0.517 0.241 0.185 0.331 0.381 0.350 0.328
HyDE 0.273 0.266 0.654 0.568 0.288 0.231 0.374 0.415 0.397 0.370
ColBERTv2 0.313 0.306 0.624 0.538 0.237 0.187 0.464 0.486 0.410 0.379
Iterative and
structure-aware
retrievalFastToG 0.326 0.331 0.385 0.327 0.077 0.068 0.233 0.269 0.255 0.249
RAPTOR 0.264 0.269 0.393 0.346 0.102 0.076 0.308 0.364 0.267 0.264
ToG 0.418 0.419 0.317 0.282 0.111 0.083 0.221 0.196 0.267 0.245
PathRAG 0.443 0.426 0.465 0.390 0.163 0.125 0.203 0.185 0.319 0.282
Graph-based
retrievalG-Retriever 0.083 0.099 0.322 0.284 0.106 0.077 0.216 0.299 0.182 0.190
LightRAG 0.188 0.188 0.535 0.453 0.232 0.160 0.293 0.339 0.312 0.285
GraphRAG 0.358 0.357 0.484 0.437 0.140 0.111 0.357 0.381 0.335 0.322
DALK 0.384 0.363 0.463 0.356 0.213 0.140 0.338 0.305 0.350 0.291
HippoRAG 2 0.447 0.580 0.701 0.644 0.291 0.283 0.485 0.549 0.481 0.514
GFM-RAG 0.668 0.744 0.718 0.657 0.320 0.318 0.613 0.625 0.580 0.586
LinearRAG 0.679 0.694 0.752 0.647 0.369 0.332 0.662 0.617 0.616 0.573
Proposed QueryFlow0.829 0.819 0.786 0.695 0.439 0.375 0.732 0.678 0.696 0.642
Table 2: Macro retrieval scores (%), ordered by FR@30. We
compare R@ 𝑘and FR@𝑘for methods with document-level
outputs, obtained directly or by mapping retrieved graph
or hierarchical units to source documents. Results for each
dataset appear in Table 7. Best in bold, second underlined .
Method R@10 FR@10 R@30 FR@30
G-Retriever 36.22 16.72 38.58 17.75
DALK 57.09 29.38 60.63 33.09
RAPTOR 56.53 27.84 68.19 43.84
LightRAG 61.66 37.28 69.67 47.70
GraphRAG 65.72 41.58 74.65 54.27
HyDE 66.52 43.17 75.16 54.73
ColBERTv2 71.20 46.91 78.63 58.37
HippoRAG 2 74.29 51.38 80.95 61.59
GFM-RAG 76.95 61.98 81.41 68.15
QueryFlow81.93 70.38 87.12 77.78
higher FR@10 (37.82% versus 31.11%). The same ordering holds
at𝑘= 30. Average coverage rewards each recovered document,
whereas full recall requires all supporting documents to occur to-
gether. This contrast shows why coverage alone cannot characterize
evidence-set recovery and why the two metrics provide comple-
mentary views of the retrieval objective.
Closer agreement between coverage and completeness.Fig-
ure 3 shows that QueryFlow’s gap between support coverage and
complete recovery decreases from 11.55 points at 𝑘=10to 9.34
at𝑘=30. GFM-RAG records gaps of 14.97 and 13.26 points, and
HippoRAG 2 22.91 and 19.36 points, respectively. This difference
captures the average recall contributed by questions whose support
sets remain incomplete. Together with higher values of both met-
rics, these smaller gaps show that QueryFlow combines broader
R@10 FR@10 R@30 FR@30
QueryFlow
GFM-RAG
HippoRAG 2
ColBERTv2
HyDE
GraphRAG
LightRAG
DALK
RAPTOR
G-Retriever81.93 70.38 87.12 77.78
76.95 61.98 81.41 68.15
74.29 51.38 80.95 61.59
71.20 46.91 78.63 58.37
66.52 43.17 75.16 54.73
65.72 41.58 74.65 54.27
61.66 37.28 69.67 47.70
57.09 29.38 60.63 33.09
56.53 27.84 68.19 43.84
36.22 16.72 38.58 17.75aMacro scores (%)
0 300 3011.55
14.97
22.91
24.29
23.35
24.14
24.38
27.71
28.69
19.50@10
9.34
13.26
19.36
20.26
20.43
20.38
21.97
27.54
24.35
20.83@30bR−FR (pp)
0 50 100Figure 3: Macro retrieval comparison, ordered by R@10. (a)
Scores from Table 2; darker cells indicate higher values. (b)
Differences between the displayed R@ 𝑘and FR@𝑘at𝑘=10
and𝑘=30, with bars sharing a scale from 0 to 30 percentage
points.
support coverage with more frequent complete recovery at both
budgets.
4.3 Query Evolution Analysis
We vary the query update strength 𝛼∈{ 0,0.25,0.5,0.75,1}on all
four benchmarks while retaining the learned model parameters.
Figure 4 shows macro gains over propagation with a fixed query
and the effects of enabling query evolution on each dataset.
RQ3.How does adaptive query evolution affect retrieval performance?

QueryFlow: Multi-Hop Graph Retrieval with Adaptive Query Evolution WWW 2027 submission, 2027, Dublin, Ireland
0 0.25 0.5 0.75 1
Query evolution strength, α0369Gain over α=0 (pp)
+5.27+8.59
+3.76+6.45aQuery evolution
R@10
FR@10R@30
FR@30
2Wiki HotpotQA MuSiQue Concurrent
QAMacro051015Δ FR@10 (pp)+2.50+7.88 +8.48+15.51
+8.59bα=1 vs α=0 by dataset
Figure 4: Effect of query evolution. (a) Macro recall and full-
recall gains over 𝛼=0at𝑘=10and𝑘=30. (b) FR@10 gains
from𝛼=0to𝛼=1by dataset and their macro average with
equal weights.
Query evolution improves complete recovery across all
four benchmarks.At 𝛼=1, macro R@10 and FR@10 reach 81.93%
and 70.38%, gains of 5.27 and 8.59 percentage points over 𝛼= 0.
Figure 4(b) shows positive FR@10 gains on every dataset, ranging
from 2.50 points on 2WikiMultiHopQA to 15.51 on ConcurrentQA.
With the anchored 𝑞0scorer and document readout fixed, query
feedback affects document ranking through the propagated entity
representations. The larger gain in full recall connects this feedback
to recovering complementary support, beyond improving average
coverage alone.
Stronger updates consistently improve recovery across the
tested settings.From 𝛼=0.25to𝛼=1, macro R@10 increases
from 79.16% to 81.93%, while FR@10 rises from 65.60% to 70.38%
(Figure 4(a)). The same upward trend holds at 𝑘=30, where full
updates gain 3.76 points in recall and 6.45 in full recall over 𝛼=0.
Increasing𝛼gives the graph-derived update a larger role in the
query state that controls subsequent boundary inputs and rela-
tion features. Continued improvement up to 𝛼=1supports using
accumulated relational context throughout propagation.
4.4 Ablation Study
Table 3 examines the feedback interfaces and the choice between
initial-query 𝑞0and terminal-query 𝑞𝐿readout, retaining the fixed-
query variant from RQ3 for reference. Appendix C examines entity
aggregation alternatives.
RQ4.How do boundary feedback, relation feedback, and the readout
query contribute to retrieval performance?
Both interfaces contribute, with a larger effect from bound-
ary feedback.Removing boundary feedback lowers FR@10 andTable 3: Macro ablation scores (%). Best in bold.
Variant R@10 FR@10 R@30 FR@30
Full model 81.93 70.38 87.12 77.78
w/o query evolution (𝛼=0) 76.66 61.78 83.36 71.33
w/o boundary feedback 79.04 65.65 85.05 74.24
w/o relation feedback 79.83 66.94 85.62 75.20
Readout with𝑞 𝐿 80.65 68.56 86.13 76.76
FR@30 by 4.73 and 3.54 points, compared with 3.44 and 2.58 for rela-
tion feedback. The former refreshes query information at question-
linked entities; the latter controls neighboring-message transfor-
mations. Their complementary roles support using both interfaces.
Initial-query readout improves early ranking.The 𝑞0read-
out achieves R@10 and FR@10 of 81.93% and 70.38%, improving
over𝑞𝐿by 1.28 and 1.82 percentage points (Table 3). At 𝑘=30, the
corresponding scores are 87.12% and 77.78%, with gains of 0.99 and
1.02 points. The larger full-recall gain at 𝑘=10supports anchor-
ing final relevance scores to the original question while evolving
queries guide context accumulation.
4.5 Evidence Recovery Analysis
Figure 5 compares QueryFlow and GFM-RAG on 4,596 common
evaluable questions using the shared support targets defined in
Section 4.1. We report FR@10 by support set size and count paired
changes between incomplete and complete recovery.
RQ5.How do gains in complete recovery of evidence sets vary with
support set size?
1
(n=44)2
(n=3,842)3
(n=312)4
(n=398)04080FR@10 (%)+2.27+6.38
+9.62+18.09aSupport-set sizeGFM-RAG QueryFlow
Complete Incomplete
QueryFlowComplete
IncompleteGFM-RAG2,805  (61.0%) 86  (1.9%)
434  (9.4%) 1,271  (27.7%)bPaired outcomes · net +348
Figure 5: Evidence recovery at 𝑘=10. (a) FR@10 by support
set size; labels show gains (pp), and 𝑛is group size. (b) Paired
outcomes. The 434 gains minus 86 losses yield 348 net recov-
eries.
The largest absolute gain occurs for support sets with four
documents.Figure 5(a) shows that QueryFlow raises FR@10 from
32.16% to 50.25% for support sets with four documents, a gain of

WWW 2027 submission, 2027, Dublin, Ireland Huang et al.
18.09 percentage points. The gains for sets with two and three
documents are 6.38 and 9.62 points, respectively, extending the ben-
efit to questions requiring several supporting documents. Larger
support sets require more complementary documents to appear
together, and a single missing document prevents full recovery.
The larger margin in this group is consistent with feedback from
graph context helping connect evidence distributed across mul-
tiple entities and documents. Support set size describes evidence
requirements across the pooled benchmarks and also reflects their
different dataset composition.
Paired comparisons yield 348 additional complete recov-
eries.QueryFlow completes 434 support sets missed by GFM-RAG,
with 86 transitions in the reverse direction, for a net gain of 348
questions (Figure 5(b)). FR@10 weighted by question count rises
from 62.90% to 70.47%, a gain of 7.57 percentage points. Net gains
occur on every benchmark, comprising 121 questions on 2Wiki-
MultiHopQA, 75 on HotpotQA, 56 on MuSiQue, and 96 on Con-
currentQA. Each additional recovery makes an entire support set
available within the same document budget. The paired outcomes
thus identify additional questions with complete support across all
four collections.
Additional recoveries preserve most previously complete
support sets.Of the 1,705 questions incomplete under GFM-RAG,
QueryFlow completes 434, or 25.5%. It also retains complete recov-
ery for 2,805 of the 2,891 questions completed by GFM-RAG, or
97.0%. These rates pool questions across datasets; the main tables
weight datasets equally. These conditional rates separate the recov-
ery of missing support from the preservation of existing coverage.
Together, they show that the net gain combines additional complete
sets with substantial overlap in successful questions, rather than
merely exchanging which questions are recovered.
4.6 Retrieval Budget Analysis
Figure 6 compares complete evidence recovery as the retrieval bud-
get increases from 3 to 30 documents across the four benchmarks.
The curves address the following question.
RQ6.How does retrieval budget affect complete recovery of evidence
sets across methods?
QueryFlow reaches comparable recovery levels with fewer
returned documents.QueryFlow reaches 62.63% FR@5, compared
with GFM-RAG’s 61.98% FR@10; its 70.38% FR@10 also exceeds
GFM-RAG’s 68.15% FR@30. At matched cutoffs, the margin over
GFM-RAG grows from 4.37 points at 𝑘= 3to 8.40 at 𝑘= 10
and 9.63 at 𝑘= 30. QueryFlow also scores above HippoRAG 2,
ColBERTv2, HyDE, and LightRAG throughout the tested range.
These comparisons show that complete support sets appear earlier
in QueryFlow’s ranking. Aggregating relevance over connected
entities can promote complementary documents together, making
complete evidence available under smaller document budgets.
Larger budgets recover additional complete support sets.
QueryFlow’s macro FR@ 𝑘increases from 51.99% at 𝑘=3to 70.38%
at𝑘=10and 77.78% at 𝑘=30. Expanding the budget from 10 to 20
adds 4.87 percentage points, followed by another 2.54 points from
20 to 30. The gap between R@ 𝑘and FR@𝑘also narrows from 13.85
points at𝑘=5to 9.34 at𝑘=30. Under a fixed ranking, a question
becomes fully recovered when its last required document enters the
35 10 15 20 3020406080Macro FR@k (%)
aComplete recovery
62.6377.78
QueryFlow
GFM-RAGHippoRAG 2
ColBERTv2HyDE
LightRAG
3–5 5–10 10–15 15–20 20–30
Retrieval budget, k036pp / document5.32
1.55
0.59 0.38 0.25bMarginal gain · QueryFlowFigure 6: Retrieval budget analysis. (a) Macro FR@ 𝑘for meth-
ods that rank documents. (b) QueryFlow’s marginal FR gain,
ΔFR/Δ𝑘, in percentage points per added document.
returned prefix. Average gains per added document decline over the
tested intervals in Figure 6(b), characterizing the tradeoff between
retrieval budget and evidence completeness. The cutoff controls
evidence passed to the generator without changing the ranking
or repeating graph propagation. The paired and budget analyses
address complementary aspects of evidence recovery. Paired out-
comes show which questions gain or lose complete support at a
fixed cutoff, whereas budget curves show how early complete sup-
port appears in a ranking. Together, they connect the aggregate
retrieval improvements to the availability of a complete supporting
set for individual questions.
4.7 Efficiency Analysis
Table 4 compares QueryFlow with existing methods in indexing
cost, online time, and token consumption on 2WikiMultiHopQA.
Together with answer quality in Table 1, these results address the
following question.
RQ7.How does QueryFlow compare with existing methods in effi-
ciency, token cost, and answer quality?
Efficient online retrieval and answer generation.QueryFlow
records 0.580 ks for retrieval and answer generation, compared with
5.96 ks for GFM-RAG, 1.36 ks for HippoRAG 2, and 1.19 ks for ToG
(Table 4). The corresponding reductions in recorded time are 90.3%,
57.3%, and 51.2%, respectively. Query evolution refines the retrieval
condition through tensor operations on the existing graph, keeping
successive feedback steps within graph computation.
Query adaptation adds no API-token cost.QueryFlow uses
3.59M tokens for retrieval and reasoning, compared with GFM-
RAG’s 5.44M, a reduction of 34.1%. With comparable indexing costs

QueryFlow: Multi-Hop Graph Retrieval with Adaptive Query Evolution WWW 2027 submission, 2027, Dublin, Ireland
Table 4: Efficiency on 2WikiMultiHopQA. Online time and
tokens include query processing, retrieval, and answer gen-
eration.
MethodIndex/Build Retrieval/Reasoning
Time (ks) Tokens (M) Time (ks) Tokens (M)
GFM-RAG 1.80 11.2 5.96 5.44
HippoRAG 2 2.08 11.1 1.36 8.53
ToG 2.11 5.15 1.19 12.7
FastToG 0.128 6.44 0.00820 1.25
QueryFlow1.80 11.2 0.580 3.59
of about 11.2M tokens, QueryFlow’s estimated total is 14.8M versus
16.6M for GFM-RAG, a reduction of 10.8%. The graph retriever itself
consumes no API tokens, so updating the latent query state adds
no intermediate query-generation token cost. This allows adaptive
retrieval without repeated textual reasoning calls during graph
propagation.
Answer quality improves alongside resource efficiency.
Table 1 reports LLM-Acc of 0.829 for QueryFlow and 0.668 for GFM-
RAG on 2WikiMultiHopQA, a gain of 16.1 percentage points. The
macro gain over GFM-RAG is 11.6 points. These answer results
accompany lower recorded time and token costs relative to GFM-
RAG in Table 4. QueryFlow also achieves higher LLM-Acc than
FastToG (0.829 versus 0.326), with FastToG offering a lower recorded
online time and token cost.
5 Conclusion
We presented QueryFlow, a graph retrieval framework that sup-
ports multi-hop question answering through coupled evidence
propagation and query evolution. Accumulated graph context up-
dates a shared query state, which controls subsequent boundary
inputs and relation features. Across four multi-hop QA benchmarks,
QueryFlow improves macro R@10, FR@10, and LLM-Acc over GFM-
RAG by 4.98, 8.40, and 11.6 percentage points, respectively. At a
budget of five documents, it also exceeds this baseline’s full recall
at ten documents. Macro LLM-Acc also improves by 8.0 points over
LinearRAG. Query-state interventions and interface ablations sup-
port the contribution of feedback within QueryFlow. The results
connect evolving query conditions to more complete evidence sets
and stronger downstream answers.

WWW 2027 submission, 2027, Dublin, Ireland Huang et al.
References
[1]Simran Arora, Patrick Lewis, Angela Fan, Jacob Kahn, and Christopher Ré. 2023.
Reasoning over Public and Private Data in Retrieval-Based Systems.Transactions
of the Association for Computational Linguistics11 (2023), 902–921.
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
InThe Twelfth International Conference on Learning Representations.
[3]Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Ruther-
ford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan
Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman
Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer,
Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero,
Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent Sifre. 2022. Improv-
ing Language Models by Retrieving from Trillions of Tokens. InProceedings of
the 39th International Conference on Machine Learning (Proceedings of Machine
Learning Research, Vol. 162). PMLR, 2206–2240.
[4]Jaime Carbonell and Jade Goldstein. 1998. The Use of MMR, Diversity-Based
Reranking for Reordering Documents and Producing Summaries. InProceedings of
the 21st Annual International ACM SIGIR Conference on Research and Development
in Information Retrieval. Association for Computing Machinery, 335–336.
[5]Boyu Chen, Zirui Guo, Zidan Yang, Yuluo Chen, Junze Chen, Zhenghao Liu,
Chuan Shi, and Cheng Yang. 2026. PathRAG: Pruning Graph-based Retrieval
Augmented Generation with Relational Paths.Proceedings of the AAAI Conference
on Artificial Intelligence40, 36 (2026), 30183–30191.
[6]Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau,
Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning Phrase
Representations using RNN Encoder–Decoder for Statistical Machine Translation.
InProceedings of the 2014 Conference on Empirical Methods in Natural Language
Processing (EMNLP). Association for Computational Linguistics, 1724–1734.
[7]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2024. From Local to Global: A Graph RAG Approach to Query-Focused
Summarization. arXiv:2404.16130
[8]Mikhail Galkin, Xinyu Yuan, Hesham Mostafa, Jian Tang, and Zhaocheng Zhu.
2024. Towards Foundation Models for Knowledge Graph Reasoning. InThe
Twelfth International Conference on Learning Representations.
[9]Guangze Gao, Zixuan Li, Chunfeng Yuan, Jiawei Li, Jianzhuo Wu, Yuehao Zhang,
Xiaolong Jin, Bing Li, and Weiming Hu. 2025. D-RAG: Differentiable Retrieval-
Augmented Generation for Knowledge Graph Question Answering. InProceedings
of the 2025 Conference on Empirical Methods in Natural Language Processing.
Association for Computational Linguistics, 35398–35417.
[10] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise Zero-Shot
Dense Retrieval without Relevance Labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).
Association for Computational Linguistics, 1762–1777.
[11] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG :
Simple and Fast Retrieval-Augmented Generation. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2025. Association for Computational
Linguistics, 10746–10761.
[12] Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su.
2024. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large
Language Models. InAdvances in Neural Information Processing Systems, Vol. 37.
Curran Associates, Inc., 59532–59569.
[13] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025.
From RAG to Memory: Non-Parametric Continual Learning for Large Language
Models. InProceedings of the 42nd International Conference on Machine Learning
(Proceedings of Machine Learning Research, Vol. 267). PMLR, 21497–21515.
[14] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang.
2020. REALM: Retrieval-Augmented Language Model Pre-Training. InProceedings
of the 37th International Conference on Machine Learning (Proceedings of Machine
Learning Research, Vol. 119). PMLR, 3929–3938.
[15] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V. Chawla, Thomas Laurent, Yann
LeCun, Xavier Bresson, and Bryan Hooi. 2024. G-Retriever: Retrieval-Augmented
Generation for Textual Graph Understanding and Question Answering. InAd-
vances in Neural Information Processing Systems, Vol. 37. Curran Associates, Inc.,
132876–132907.
[16] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. InProceedings of the 28th International Conference on Computational
Linguistics. International Committee on Computational Linguistics, 6609–6625.
[17] Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao.
2025. GRAG: Graph Retrieval-Augmented Generation. InFindings of the Associa-
tion for Computational Linguistics: NAACL 2025. Association for Computational
Linguistics, 4145–4157.
[18] Manzong Huang, Chenyang Bu, Yi He, Xingrui Zhuo, and Xindong Wu. 2026.
Relink: Constructing Query-Driven Evidence Graph On-the-Fly for GraphRAG.Proceedings of the AAAI Conference on Artificial Intelligence40, 37 (2026), 31202–
31210.
[19] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bo-
janowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised Dense Infor-
mation Retrieval with Contrastive Learning.Transactions on Machine Learning
Research(2022).
[20] Gautier Izacard and Edouard Grave. 2021. Leveraging Passage Retrieval with
Generative Models for Open Domain Question Answering. InProceedings of the
16th Conference of the European Chapter of the Association for Computational
Linguistics: Main Volume. Association for Computational Linguistics, 874–880.
[21] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo
Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave.
2023. Atlas: Few-shot Learning with Retrieval Augmented Language Models.
Journal of Machine Learning Research24, 251 (2023), 1–43.
[22] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong Park.
2024. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
Models through Question Complexity. InProceedings of the 2024 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers). Association for Computational
Linguistics, 7036–7050.
[23] Yichen Jiang, Shikha Bordia, Zheng Zhong, Charles Dognin, Maneesh Singh, and
Mohit Bansal. 2020. HoVer: A Dataset for Many-Hop Fact Extraction And Claim
Verification. InFindings of the Association for Computational Linguistics: EMNLP
2020. Association for Computational Linguistics, 3441–3460.
[24] Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-
Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active Retrieval
Augmented Generation. InProceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing. Association for Computational Linguistics,
7969–7992.
[25] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP). Association for Computational
Linguistics, 6769–6781.
[26] Omar Khattab, Christopher Potts, and Matei Zaharia. 2021. Baleen: Robust
Multi-Hop Reasoning at Scale via Condensed Retrieval. InAdvances in Neural
Information Processing Systems, Vol. 34. Curran Associates, Inc., 27670–27682.
[27] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage
Search via Contextualized Late Interaction over BERT. InProceedings of the 43rd
International ACM SIGIR Conference on Research and Development in Information
Retrieval. Association for Computing Machinery, 39–48.
[28] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. InAdvances in Neural Information Processing
Systems, Vol. 33. Curran Associates, Inc., 9459–9474.
[29] Dawei Li, Shu Yang, Zhen Tan, Jae Young Baik, Sukwon Yun, Joseph Lee, Aaron
Chacko, Bojian Hou, Duy Duong-Tran, Ying Ding, Huan Liu, Li Shen, and Tian-
long Chen. 2024. DALK: Dynamic Co-Augmentation of LLMs and KG to answer
Alzheimer’s Disease Questions with Scientific Literature. InFindings of the Associ-
ation for Computational Linguistics: EMNLP 2024. Association for Computational
Linguistics, 2187–2205.
[30] Mufei Li, Siqi Miao, and Pan Li. 2025. Simple is Effective: The Roles of Graphs
and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented
Generation. InThe Thirteenth International Conference on Learning Representa-
tions.
[31] Zijian Li, Qingyan Guo, Jiawei Shao, Lei Song, Jiang Bian, Jun Zhang, and Rui
Wang. 2025. Graph Neural Network Enhanced Retrieval for Question Answering
of Large Language Models. InProceedings of the 2025 Conference of the Nations
of the Americas Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers). Association for Computational
Linguistics, 6612–6633.
[32] Xujian Liang and Zhaoquan Gu. 2025. Fast Think-on-Graph: Wider, Deeper and
Faster Reasoning of Large Language Model on Knowledge Graph.Proceedings of
the AAAI Conference on Artificial Intelligence39, 23 (2025), 24558–24566.
[33] Haoran Luo, Haihong E, Guanting Chen, Qika Lin, Yikai Guo, Fangzhi Xu, Zemin
Kuang, Meina Song, Xiaobao Wu, Yifan Zhu, and Anh Tuan Luu. 2026. Graph-R1:
Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learn-
ing. InProceedings of the 43rd International Conference on Machine Learning
(Proceedings of Machine Learning Research, Vol. 306). PMLR, 82220–82240.
[34] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. 2024. Reasoning
on Graphs: Faithful and Interpretable Large Language Model Reasoning. InThe
Twelfth International Conference on Learning Representations.
[35] Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Chen Gong, Dinh Phung, and
Shirui Pan. 2025. GFM-RAG: Graph Foundation Model for Retrieval Augmented
Generation. InAdvances in Neural Information Processing Systems, Vol. 38. Curran
Associates, Inc., 36371–36405.

QueryFlow: Multi-Hop Graph Retrieval with Adaptive Query Evolution WWW 2027 submission, 2027, Dublin, Ireland
[36] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. 2023. Query
Rewriting in Retrieval-Augmented Large Language Models. InProceedings of the
2023 Conference on Empirical Methods in Natural Language Processing. Association
for Computational Linguistics, 5303–5315.
[37] Costas Mavromatis and George Karypis. 2025. GNN-RAG: Graph Neural Re-
trieval for Efficient Large Language Model Reasoning on Knowledge Graphs. In
Findings of the Association for Computational Linguistics: ACL 2025. Association
for Computational Linguistics, 16682–16699.
[38] Peng Qi, Xiaowen Lin, Leo Mehr, Zijian Wang, and Christopher D. Manning.
2019. Answering Complex Open-domain Questions Through Iterative Query
Generation. InProceedings of the 2019 Conference on Empirical Methods in Natural
Language Processing and the 9th International Joint Conference on Natural Lan-
guage Processing (EMNLP-IJCNLP). Association for Computational Linguistics,
2590–2602.
[39] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei
Zaharia. 2022. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late
Interaction. InProceedings of the 2022 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language Technologies.
Association for Computational Linguistics, 3715–3734.
[40] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and
Christopher D. Manning. 2024. RAPTOR: Recursive Abstractive Processing for
Tree-Organized Retrieval. InThe Twelfth International Conference on Learning
Representations. OpenReview.net.
[41] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu
Chen. 2023. Enhancing Retrieval-Augmented Large Language Models with
Iterative Retrieval-Generation Synergy. InFindings of the Association for Com-
putational Linguistics: EMNLP 2023. Association for Computational Linguistics,
9248–9274.
[42] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike
Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2024. REPLUG: Retrieval-Augmented
Black-Box Language Models. InProceedings of the 2024 Conference of the North
American Chapter of the Association for Computational Linguistics: Human Lan-
guage Technologies (Volume 1: Long Papers). Association for Computational Lin-
guistics, 8371–8384.
[43] Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, and Yiqun Liu. 2024. DRAGIN:
Dynamic Retrieval Augmented Generation based on the Real-time Information
Needs of Large Language Models. InProceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers). Association
for Computational Linguistics, 12991–13013.
[44] Haitian Sun, Tania Bedrax-Weiss, and William W. Cohen. 2019. PullNet: Open
Domain Question Answering with Iterative Retrieval on Knowledge Bases and
Text. InProceedings of the 2019 Conference on Empirical Methods in Natural Lan-
guage Processing and the 9th International Joint Conference on Natural Language
Processing (EMNLP-IJCNLP). Association for Computational Linguistics, 2380–
2390.
[45] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun
Gong, Lionel M. Ni, Heung-Yeung Shum, and Jian Guo. 2024. Think-on-Graph:
Deep and Responsible Reasoning of Large Language Model on Knowledge Graph.
InThe Twelfth International Conference on Learning Representations.
[46] Alon Talmor and Jonathan Berant. 2018. The Web as a Knowledge-Base for
Answering Complex Questions. InProceedings of the 2018 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies, Volume 1 (Long Papers). Association for Computational
Linguistics, 641–651.
[47] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop Questions via Single-hop Question Composition.
Transactions of the Association for Computational Linguistics10 (2022), 539–554.
[48] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-
Intensive Multi-Step Questions. InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers). Association
for Computational Linguistics, 10014–10037.
[49] Jiapu Wang, Zheng Cui, Boyue Wang, Shirui Pan, Junbin Gao, Baocai Yin, and
Wen Gao. 2024. IME: Integrating Multi-curvature Shared and Specific Embedding
for Temporal Knowledge Graph Completion. InProceedings of the ACM Web
Conference 2024. Association for Computing Machinery, 1954–1962. doi:10.1145/
3589334.3645361
[50] Jiapu Wang, Kai Sun, Linhao Luo, Wei Wei, Yongli Hu, Alan Liew, Shirui Pan,
and Baocai Yin. 2024. Large Language Models-guided Dynamic Adaptation
for Temporal Knowledge Graph Reasoning. InAdvances in Neural Information
Processing Systems, Vol. 37. Neural Information Processing Systems Foundation,
8384–8410. doi:10.52202/079017-0269
[51] Jinlu Wang, Yanfeng Sun, Jiapu Wang, Junbin Gao, Shaofan Wang, Baocai Yin,
and Jipeng Guo. 2026. ComGRL: Comprehensive Graph Representation Learning
from Local to Global Bridged by Mixup.IEEE Transactions on Computational
Social Systems13, 3 (2026), 2839–2852. doi:10.1109/TCSS.2026.3665450
[52] Jiapu Wang, Boyue Wang, Junbin Gao, Simin Hu, Yongli Hu, and Baocai Yin.
2024. Multi-Level Interaction Based Knowledge Graph Completion.IEEE/ACMTransactions on Audio, Speech, and Language Processing32 (2024), 386–396. doi:10.
1109/TASLP.2023.3331121
[53] Jiapu Wang, Boyue Wang, Junbin Gao, Yongli Hu, and Baocai Yin. 2023. Multi-
Concept Representation Learning for Knowledge Graph Completion.ACM
Transactions on Knowledge Discovery from Data17, 1 (2023), 1–19. doi:10.1145/
3533017
[54] Jiapu Wang, Boyue Wang, Junbin Gao, Xiaoyan Li, Yongli Hu, and Baocai Yin.
2023. TDN: Triplet Distributor Network for Knowledge Graph Completion.IEEE
Transactions on Knowledge and Data Engineering35, 12 (2023), 13002–13014.
doi:10.1109/TKDE.2023.3272568
[55] Jiapu Wang, Boyue Wang, Junbin Gao, Xiaoyan Li, Yongli Hu, and Baocai Yin.
2024. QDN: A Quadruplet Distributor Network for Temporal Knowledge Graph
Completion.IEEE Transactions on Neural Networks and Learning Systems35, 10
(2024), 14018–14030. doi:10.1109/TNNLS.2023.3274230
[56] Wenhan Xiong, Xiang Lorraine Li, Srinivasan Iyer, Jingfei Du, Patrick Lewis,
William Wang, Yashar Mehdad, Wen-tau Yih, Sebastian Riedel, Douwe Kiela, and
Barlas Oğuz. 2021. Answering Complex Open-Domain Questions with Multi-Hop
Dense Retrieval. InInternational Conference on Learning Representations.
[57] Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. 2015. Em-
bedding Entities and Relations for Learning and Inference in Knowledge Bases.
InInternational Conference on Learning Representations.
[58] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. InProceedings of the 2018
Conference on Empirical Methods in Natural Language Processing. Association for
Computational Linguistics, 2369–2380.
[59] Jiahao Zhang, Haiyang Zhang, Dongmei Zhang, Yong Liu, and Shen Huang. 2024.
End-to-End Beam Retrieval for Multi-Hop Question Answering. InProceedings
of the 2024 Conference of the North American Chapter of the Association for Com-
putational Linguistics: Human Language Technologies (Volume 1: Long Papers).
Association for Computational Linguistics, 1718–1731.
[60] Tianxiang Zhao, Youqing Wang, Jinlu Wang, Jiapu Wang, Mingliang Cui, Junbin
Gao, and Jipeng Guo. 2025. Hybrid-Collaborative Augmentation and Contrastive
Sample Adaptive-Differential Awareness for Robust Attributed Graph Clustering.
InAdvances in Neural Information Processing Systems, Vol. 38. Neural Information
Processing Systems Foundation, 128567–128592. doi:10.52202/085713-3867
[61] Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, and Wei Hu. 2025. Knowledge
Graph-Guided Retrieval Augmented Generation. InProceedings of the 2025 Con-
ference of the Nations of the Americas Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Papers). Association
for Computational Linguistics, 8912–8924.
[62] Zhaocheng Zhu, Xinyu Yuan, Mikhail Galkin, Sophie Xhonneux, Ming Zhang,
Maxime Gazeau, and Jian Tang. 2023. A*Net: A Scalable Path-based Reasoning
Approach for Knowledge Graphs. InAdvances in Neural Information Processing
Systems, Vol. 36. Curran Associates, Inc., 59323–59336.
[63] Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal Xhonneux, and Jian Tang. 2021.
Neural Bellman-Ford Networks: A General Graph Neural Network Framework
for Link Prediction. InAdvances in Neural Information Processing Systems, Vol. 34.
Curran Associates, Inc., 29476–29490.
[64] Luyao Zhuang, Shengyuan Chen, Yilin Xiao, Huachi Zhou, Yujing Zhang, Hao
Chen, Qinggang Zhang, and Xiao Huang. 2026. LinearRAG: Linear Graph Re-
trieval Augmented Generation on Large-scale Corpora. InThe Fourteenth Inter-
national Conference on Learning Representations.
[65] Xingrui Zhuo, Shirui Pan, Jiapu Wang, Gongqing Wu, Zan Zhang, Rui Li, Zizhong
Wei, and Xindong Wu. 2025. Progressive Prefix-Memory Tuning for Complex
Logical Query Answering on Knowledge Graphs. InProceedings of the Thirty-
Fourth International Joint Conference on Artificial Intelligence. International Joint
Conferences on Artificial Intelligence Organization, 3716–3724. doi:10.24963/
ijcai.2025/413
[66] Xingrui Zhuo, Jiapu Wang, Gongqing Wu, Shirui Pan, and Xindong Wu. 2025.
Effective Instruction Parsing Plugin for Complex Logical Query Answering on
Knowledge Graphs. InProceedings of the ACM on Web Conference 2025. ACM,
4780–4792. doi:10.1145/3696410.3714794
[67] Xingrui Zhuo, Jiapu Wang, Gongqing Wu, Zhongyuan Wang, Jichen
Zhang, Shirui Pan, and Xindong Wu. 2026. Knowledge Reasoning
Language Model: Unifying Knowledge and Language for Inductive
Knowledge Graph Reasoning. InInternational Conference on Learning
Representations. https://proceedings.iclr.cc/paper_files/paper/2026/file/
c6e9f9d49cc052879efd19da0c3e1900-Paper-Conference.pdf

WWW 2027 submission, 2027, Dublin, Ireland Huang et al.
Table 5: Evaluation input records and retrieval corpus sizes.
Dataset Input records Documents
2WikiMultiHopQA 1,000 9,594
HotpotQA 1,000 9,811
MuSiQue 1,003 11,666
ConcurrentQA 1,600 2,840
A Additional Experimental Details
A.1 Experimental Setup
2WikiMultiHopQA.This benchmark combines Wikidata rela-
tions with Wikipedia text to construct questions with explicit multi-
hop reasoning paths [ 16]. These annotations specify the intermedi-
ate relations linking a question to its answer, providing a structured
basis for assessing evidence-set recovery.
HotpotQA.Questions draw on multiple Wikipedia documents
and include comparisons between entities [ 58]. Sentence-level sup-
porting facts identify the evidence needed for an answer. We map
these annotations to source documents to define the target evidence
sets.
MuSiQue.MuSiQue composes single-hop questions into con-
nected two- to four-hop questions [ 47]. Intermediate answers link
successive steps, reducing opportunities for disconnected reasoning
and making subsequent evidence needs depend on earlier discover-
ies. We use the answerable examples.
ConcurrentQA.ConcurrentQA requires evidence from distinct
public and private data distributions [ 1]. We use the supplied docu-
ments as a shared retrieval collection to evaluate evidence recovery
across heterogeneous sources with the same document-ranking
interface.
Evaluation subsets.Following GFM-RAG [ 35], LinearRAG [ 64],
and Relink [ 18], we use fixed evaluation subsets. Matching Lin-
earRAG’s reference question lists to source records yields 1,000
questions each for 2WikiMultiHopQA and HotpotQA, with Hot-
potQA drawn from distractor validation. For MuSiQue, we match
the reference list to records in the answerable validation split. Hot-
potQA and MuSiQue are shuffled with seed 42 before matching;
list membership determines selection. ConcurrentQA uses all 1,600
validation questions. Input sets are shared across methods. Retrieval
metrics are averaged over questions with nonempty support sets
(Section 4.1).
Retrieval corpora.Supporting and distractor documents from the
selected examples form each benchmark’s retrieval corpus. Table 5
reports selected input records and corpus sizes after preprocessing.
Each question is evaluated against this collection, with 𝑘limiting
the number of returned documents. Original support annotations
define the target evidence sets.
Training data and supervision.We use 20,000 examples and their
associated content from each of 2WikiMultiHopQA, HotpotQA,
and MuSiQue, totaling 60,000 training examples. Entity labels 𝑦𝑖
are derived from these training datasets. The graph interface isdefined in Section 3.1. Extraction, normalization, linking, ground-
ing, and document-to-entity label construction are detailed in the
supplementary code.
A.2 Implementation Details
Model architecture.The frozen all-mpnet-base-v2 encoder sup-
plies precomputed 768-dimensional question and relation embed-
dings, linearly projected to 𝑑=512. The propagation operator uses
the NBFNet formulation [ 63]. The retriever uses 𝐿=6layers with
512 hidden units, DistMult messages, sum aggregation, residual con-
nections, layer normalization, and ReLU. The query controller also
has 512-dimensional states. The full model enables both feedback
interfaces with 𝛼=1, attention temperature 1, and query-update
dropout 0.
Optimization and hardware.We train the retriever and query
controller from random initialization on a single NVIDIA A100
using AdamW, learning rate10−5, batch size one, gradient norm
clipping at 1.0, and seed 1024. Activation checkpointing reduces
propagation memory; the text encoder remains frozen.
Answer generation and judgment.All methods generate answers
with Qwen3.5-9B. Following LinearRAG [ 64], LLM-Acc records
LLM judge acceptance; Contain checks for the normalized gold
answer. Prompts, context handling, decoding, judge configuration,
and output parsing are detailed in the supplementary code.
A.3 Additional Results
Tables 6 and 7 report additional answer metrics and retrieval scores
by dataset. Macro averages weight datasets equally.
Table 6: Additional answer generation metrics for QueryFlow
(%). Macro is the unweighted mean across datasets.
Dataset EM F1 Precision Recall Contain LLM-Acc
2WikiMultiHopQA 65.3% 75.4% 73.9% 83.3% 81.9% 82.9%
HotpotQA 56.7% 69.6% 71.0% 77.0% 69.5% 78.6%
MuSiQue 24.5% 35.1% 35.3% 49.0% 37.5% 43.9%
ConcurrentQA 55.7% 63.5% 63.5% 72.5% 67.8% 73.2%
Macro 50.5% 60.9% 60.9% 70.4% 64.2% 69.6%
B Training Objective and Retrieval Algorithm
B.1 Document Evidence Readout
Default entity scoring uses 𝑞0during both training and retrieval.
Intermediate query updates influence document ranking through
the final entity representations𝐻 𝐿.
Aggregation over all entities.Softplus preserves entity ordering
and yields positive contributionsv. With 𝑓𝑖=Í
𝑗𝐴𝑖𝑗and𝑤doc
𝑖=
1/𝑓𝑖for𝑓𝑖>0(zero otherwise), the document scores are
d=𝐴⊤
wdoc⊙v
.(16)
Here𝐴∈{ 0,1}|𝑉|×|𝐷|andd∈R|𝐷|. All entities contribute; the top
𝑘scores define b𝐷𝑘(𝑥). Paired propagation interventions keep the
scorer and readout fixed.

QueryFlow: Multi-Hop Graph Retrieval with Adaptive Query Evolution WWW 2027 submission, 2027, Dublin, Ireland
Table 7: Retrieval scores for each dataset (%), ordered by macro FR@30. We compare R@ 𝑘and FR@𝑘for methods with
document-level outputs, obtained directly or by mapping retrieved graph or hierarchical units to source documents. Macro is
the unweighted mean across datasets. Best in bold, second underlined .
2WikiMultiHopQA HotpotQA MuSiQue ConcurrentQA Macro
Method R@10 FR@10 R@30 FR@30 R@10 FR@10 R@30 FR@30 R@10 FR@10 R@30 FR@30 R@10 FR@10 R@30 FR@30 R@10 FR@10 R@30 FR@30
G-Retriever 32.10 11.28 34.37 12.08 44.85 23.51 46.57 24.12 25.80 11.60 27.72 12.10 42.14 20.50 45.66 22.70 36.22 16.72 38.58 17.75
DALK 63.68 28.30 63.95 28.80 63.78 40.54 67.40 45.57 40.44 10.87 44.35 14.08 60.47 37.79 66.81 43.91 57.09 29.38 60.63 33.09
RAPTOR 59.22 27.10 80.30 61.70 65.80 40.70 65.80 40.70 40.87 8.87 66.43 38.29 60.22 34.69 60.22 34.69 56.53 27.84 68.19 43.84
LightRAG 52.20 24.50 59.75 31.80 82.35 67.60 88.90 79.30 57.40 24.23 69.42 39.38 54.69 32.81 60.59 40.31 61.66 37.28 69.67 47.70
GraphRAG 70.62 39.90 74.35 46.50 77.60 60.80 86.70 76.90 49.20 14.56 60.10 26.12 65.44 51.06 77.44 67.56 65.72 41.58 74.65 54.27
HyDE 58.52 30.80 66.22 38.80 86.05 73.80 91.75 84.60 63.17 31.0174.5346.46 58.34 37.06 68.16 49.06 66.52 43.17 75.16 54.73
ColBERTv2 70.50 42.20 74.98 49.10 86.75 74.30 92.70 85.50 56.60 22.33 67.95 37.19 70.97 48.81 78.91 61.69 71.20 46.91 78.63 58.37
HippoRAG 2 73.82 45.59 77.66 51.70 89.32 79.84 93.38 87.36 64.4231.11 74.51 45.56 69.60 48.97 78.27 61.74 74.29 51.38 80.95 61.59
GFM-RAG 88.52 75.00 89.72 77.50 86.19 76.68 89.69 82.38 57.04 31.84 65.32 42.32 76.06 64.39 80.91 70.40 76.95 61.98 81.41 68.15
QueryFlow93.81 87.60 95.36 90.70 90.99 84.78 94.74 90.9961.6037.8272.2751.20 81.33 71.29 86.09 78.24 81.93 70.38 87.12 77.78
B.2 Entity-Level Training Objective
Training masks 𝑦∈{ 0,1}|𝑉|define positives 𝑃={𝑖 :𝑦𝑖=1}and
negatives𝑁={𝑖 :𝑦𝑖=0}. Binary and listwise losses supervise the
graph and query controller.
Entity relevance objective.Let 𝑝𝑖=𝜎(𝑠𝑖), where𝑠𝑖is computed
using the anchored scorer in Eq. (10). The binary cross-entropy for
entity𝑖is
ℓ𝑖=−𝑦𝑖log𝑝𝑖−(1−𝑦𝑖)log(1−𝑝 𝑖).(17)
Positive weights are uniform; negative weights depend on current
logits,
𝜔𝑖= 
|𝑃|−1, 𝑖∈𝑃,
sgexp(𝑠𝑖/𝑇)Í
𝑗∈𝑁exp(𝑠𝑗/𝑇)
, 𝑖∈𝑁.(18)
Here𝑇=0.2, and sgstops gradients through the negative weights.
Negative entities with higher scores receive greater weight, con-
centrating the classification loss on negatives that the retriever
currently ranks highly. The weighted binary loss is
LBCE=Í
𝑖∈𝑉𝜔𝑖ℓ𝑖Í
𝑖∈𝑉𝜔𝑖.(19)
ListCE (Eq. (13),𝜀=10−5) normalizes sigmoid probabilities over
entities and averages negative log mass over all positives, exclud-
ing empty targets. The combined loss in Eq. (14)is averaged over
questions in each batch.
Aggregation of entity scores into document scores and document
ranking are performed at inference time, outside the entity-level
training objective. The query controller is trained through its in-
fluence on subsequent graph propagation and final entity scores;
the supporting-document sets define the downstream evaluation
targets.
B.3 Retrieval Procedure
Algorithm 1 applies 𝐿propagation layers with query updates be-
tween successive layers before ranking documents. The default
𝛼=1applies the gated increment before normalization; 𝛼=0fixes
the query without changing learned propagation parameters.At layer𝑙, propagation first computes 𝐻𝑙under𝑞𝑙−1. When𝑙<
𝐿, the updated entity states are summarized to form 𝑞𝑙, which
conditions the next layer. The default scorer combines 𝐻𝐿with
𝑞0. The𝑞𝐿variant adds the terminal update defined in Section 3.3
and substitutes 𝑞𝐿in the scorer. Graph structure, grounding, and
document associations remain fixed.
Algorithm 1QueryFlow with the default𝑞 0readout
Require:𝐺,𝐴,𝑚 𝑥,𝑒𝑥,{𝑒𝑟}; parameters;𝐿,𝛼,𝑘
Ensure:Ranked top-𝑘documents b𝐷𝑘(𝑥)
1:𝑞 0←𝜙𝑞(𝑒𝑥);𝑅0(𝑟)←𝜙𝑟(𝑒𝑟)for all𝑟
2:𝐻 0(𝑖)←𝑚𝑥(𝑖)𝑞 0for all𝑖∈𝑉
3:for𝑙=1,...,𝐿do
4:𝐵𝑙(𝑖)←𝑚𝑥(𝑖)𝑞𝑙−1
5:𝛾𝑙−1←𝜎(𝑊𝑟𝑞𝑙−1+𝑏𝑟)
6: ¯𝑅𝑙(𝑟)←𝜓𝑙(𝑅0(𝑟)⊙𝛾𝑙−1)
7:Compute𝐻 𝑙using Eqs. (4)–(5)
8:if𝑙<𝐿then
9:Compute𝑧 𝑙using Eqs. (6)–(7)
10:if𝛼=0then
11:𝑞 𝑙←𝑞𝑙−1
12:else
13:Forme𝑞 𝑙,𝑔𝑙,Δ𝑙using Eq. (8)
14:𝑞 𝑙←LN𝑞(𝑞𝑙−1+𝛼Δ𝑙)
15:end if
16:end if
17:end for
18:𝑠𝑖←MLP([𝐻 𝐿(𝑖);𝑞 0])for all𝑖∈𝑉
19:𝑣𝑖←softplus(𝑠 𝑖)for all𝑖∈𝑉
20:𝑓𝑖←Í|𝐷|
𝑗=1𝐴𝑖𝑗for all𝑖∈𝑉
21:𝑤doc
𝑖←1/𝑓𝑖if𝑓𝑖>0, otherwise0
22:𝑑𝑗←Í
𝑖∈𝑉𝐴𝑖𝑗𝑤doc
𝑖𝑣𝑖for all documents𝑗
23:returnthe𝑘highest-scoring documents

WWW 2027 submission, 2027, Dublin, Ireland Huang et al.
C Readout Analysis
We compare aggregation over all entities with two truncated entity
readouts, document-aware entity MMR [ 4] and relevance-based
Top-20 selection. All three use the same propagated entity repre-
sentations and reciprocal document frequency weighting.
Table 8: Document readout comparison. Macro retrieval
scores across four datasets (%).
Readout R@10 FR@10 R@30 FR@30
MMR Top-20 80.48 67.84 85.31 74.72
Top-20 (no MMR) 80.57 67.79 84.87 73.90
All entities (default)81.93 70.38 87.12 77.78
Aggregation over all entities improves complete evidence
recovery.The default readout exceeds both truncated alternatives
on all four metrics in Table 8. Relative to MMR Top-20, the gains are
2.54 and 3.07 percentage points in FR@10 and FR@30. Aggregating
every entity contribution lets documents receive relevance signals
distributed beyond a small selected subset, providing a mechanism
for retaining complementary evidence. Within the truncated alter-
natives, MMR improves R@30 and FR@30 by 0.44 and 0.82 points,
with mixed but small differences at 𝑘=10(−0.09R@10 and+0.05
FR@10).
Reciprocal weighting preserves each entity’s total contri-
bution.For 𝑓𝑖>0, summing 𝐴𝑖𝑗𝑣𝑖/𝑓𝑖over documents yields 𝑣𝑖.
Summing the complete document score vector gives
|𝐷|∑︁
𝑗=1𝑑𝑗=∑︁
𝑖∈𝑉𝑤doc
𝑖𝑣𝑖|𝐷|∑︁
𝑗=1𝐴𝑖𝑗=∑︁
𝑖∈𝑉:𝑓 𝑖>0𝑣𝑖.(20)
Document associations redistribute each entity’s score across its
source documents. An entity appearing in one document con-
tributes its full score there; one appearing in several documents
divides that contribution equally among them. This property con-
trols how graph relevance transfers to documents without selecting
a fixed number of entities. Each document can still accumulate
contributions from multiple entities, so its rank depends on their
combined relevance and document frequencies. The weighting acts
on entity frequency rather than normalizing by document length.
# Read the Route: Attention-Steered Hypergraph Retrieval for Multi-hop RAG

**Authors**: Jiate Liu, Liuyi Chen, Xiaofei Yuan, Yifu Tang, Zihan Yang, Xu Zhou, Zhengyi Yang

**Published**: 2026-10-26 13:24:28

**PDF URL**: [https://arxiv.org/pdf/2610.35412v1](https://arxiv.org/pdf/2610.35412v1)

## Abstract
Graph-based retrieval-augmented generation (GraphRAG) answers multi-hop questions by organizing a corpus into a graph and retrieving along its structure. Existing retrieval methods face two limitations. Diffusion- and similarity-based methods expand retrieval along fixed edge weights or by similarity to the question, without resolving its implicit entity references, allowing relevance errors to accumulate across hops. In contrast, exploration-based methods let an LLM reason about the next hop, but require a separate generation at each hop, leading to high inference costs. We observe that an LLM can resolve implicit entity references from retrieved passages and a multi-hop question before generating any tokens. Further analysis identifies a small set of attention heads, termed addressing heads, that capture this capability across multiple hops in a single forward pass. Building on this finding, we propose HyperAddress, which runs Personalized PageRank on a description-labelled hypergraph, using addressing heads to determine the restart distribution and transition weights, thereby guiding multi-hop retrieval without any generation at query time. On MuSiQue, 2WikiMultiHopQA, and HotpotQA, HyperAddress outperforms the strongest baseline by 3.5 points of Recall@5 and 7.4 points of All@5 on average, with gains that grow with the number of hops. It answers more than twice as fast and builds its index at half the cost.

## Full Text


<!-- PDF content starts -->

Read the Route: Attention-Steered Hypergraph Retrieval for
Multi-hop RAG
Jiate Liu
The University of New South Wales
Sydney, Australia
jiate.liu@unsw.edu.auLiuyi Chen
Hunan University
Changsha, China
liuyi.chen@hnu.edu.cnXiaofei Yuan
Euler AI
Australia
xiaofei.yuan@eulerai.au
Yifu Tang
Vecton AI
Australia
yves.tang@vectonai.comZihan Yang
Vecton AI
Australia
zihan.yang@vectonai.comXu Zhou
Hunan University
Changsha, China
zhxu@hnu.edu.cn
Zhengyi Yang∗
The University of Sydney
Sydney, Australia
zhengyi.yang@sydney.edu.au
Abstract
Graph-based retrieval-augmented generation (GraphRAG) answers
multi-hop questions by organizing a corpus into a graph and re-
trieving along its structure. Existing retrieval methods face two
limitations. Diffusion- and similarity-based methods expand re-
trieval along fixed edge weights or by similarity to the question,
without resolving its implicit entity references, allowing relevance
errors to accumulate across hops. In contrast, exploration-based
methods let an LLM reason about the next hop, but require a sep-
arate generation at each hop, leading to high inference costs. We
observe that an LLM can resolve implicit entity references from
retrieved passages and a multi-hop question before generating any
tokens. Further analysis identifies a small set of attention heads,
termed addressing heads, that capture this capability across mul-
tiple hops in a single forward pass. Building on this finding, we
proposeHyperAddress, which runs Personalized PageRank on
a description-labelled hypergraph, using addressing heads to de-
termine the restart distribution and transition weights, thereby
guiding multi-hop retrieval without any generation at query time.
On MuSiQue, 2WikiMultiHopQA, and HotpotQA,HyperAddress
outperforms the strongest baseline by 3.5 points of Recall@5 and 7.4
points of All@5 on average, with gains that grow with the number
of hops. It answers more than twice as fast and builds its index at
half the cost.
CCS Concepts
•Information systems →Retrieval models and ranking;•
Computing methodologies→Natural language processing.
Keywords
Retrieval-augmented generation, multi-hop retrieval, graph-based
retrieval, hypergraphs, attention heads, mechanistic interpretability
∗Corresponding author.1 Introduction
Retrieval-Augmented Generation (RAG) grounds Large Language
Models (LLMs) in external knowledge by retrieving passages from
a corpus and conditioning the answer on them [ 9,24]. Retrieval-
Augmented Generation (RAG) enables LLMs to leverage the vast
and continuously evolving knowledge available on the Web, sup-
porting more accurate and up-to-date question answering. Standard
RAG retrieves passages based on their relevance to the original
question, working well for single-hop questions but struggling
with multi-hop questions [ 20,34,42]. In multi-hop questions, some
essential evidence may not be directly relevant to the original ques-
tion but is instead connected through intermediate entities. We call
such an intermediate entity a bridge entity, and the question often
refers to it only implicitly. For example, to answerWhere did the
author of Princeps Pastorum die?, a system must find that the author
is Pope John XXIII and then retrieve his biography, although the
question never mentions him. A retriever that compares passages
with the question as a whole finds the passage about Princeps Pas-
torum but rarely the biography, so the second piece of evidence is
often missed [16, 35].
Graph-based RAG (GraphRAG) organizes passages into a graph
and retrieves evidence along its edges, enabling the discovery of
evidence passages connected to initially retrieved passages through
intermediate entities. While existing systems construct graphs us-
ing different representations [ 7,13,17,32,37,47], their multi-hop
retrieval strategies generally fall into two categories (Figure 1).
Diffusion- and similarity-based methods expand retrieval with a
fixed procedure: HippoRAG 2 [ 17] and LinearRAG [ 47] spread rel-
evance over the graph with Personalized PageRank (PPR) [ 3,18],
and PropRAG [ 37] searches paths of propositions by their similarity
to the question. Exploration-based methods such as IRCoT [ 35] and
Self-Ask [ 30] rely on LLM generation for two tasks: identifying in-
termediate entities to guide retrieval (semantic understanding) and
inferring and following relationships among entities (relational ex-
ploration). These methods iteratively retrieve passages and generate
arXiv:2610.35412v1  [cs.IR]  26 Oct 2026

Liu et al.
(a) Diffusion / similarity searchHippoRAG 2·LinearRAG·PropRAGQ: "Where did the author of Princeps Pastorum die?"John XXlll is referred to ,  but never named in the question.Follows qJohn XXlllwritten byCheapqSelect the startFixed weights or similarity miss what q refers to.Princeps Pastorumembeddin g
(fixed weights or similarity to q)biography(b) LLM explorationIRCoT·Self-Ask·Think-on-GraphThinkRetrieveObserveFollows qJohn XXlllCheapLLM1The author is John XXlll.12Generation at every hop adds cost.2John XXlll died in...Princeps Pastorumbiography(c) HYPERADDRESS (ours)L23-36 not run8 addressing headsL1-17the author of...John XXlllFollows qCheapOne forward pass. No generated tokens.Passages + qPrinceps PastorumFollow the question, without generating.
Read descriptions in context; let attention steer retrieval.documentsattentionbiographyretrieval
Figure 1: Three paradigms on the same graph: only
HyperAddressfollows the question without generating.
reasoning based on previously retrieved evidence, while exploring
entity relationships through textual contexts.
Both strategies have limitations. (i)Diffusion- and similarity-
based methods accumulate relevance errors across hops.At
each hop, these methods expand retrieval without identifying the
entity that the question refers to. For example, PPR propagates
relevance scores using fixed edge weights, regardless of which
connections are relevant to the question. As shown in Figure 1(a),
the question asks where the author of Princeps Pastorum died but
does not explicitly name John XXIII. While John XXIII is the rele-
vant bridge entity, PPR also propagates relevance scores to other
connected entities that are irrelevant to the question. These ir-
relevant entities can further propagate scores to their neighbors,
causing retrieval errors to accumulate across hops. Similarity-based
methods rank passages by their relevance to the original question,
often missing essential evidence at later hops that is connected
through intermediate entities. (ii)Exploration-based methods
incur high generation costs and underutilize structural in-
formation.First, semantic understanding requires step-by-step
LLM generation, where each hop relies on previously retrieved
passages and generated reasoning. As the retrieved evidence and
reasoning accumulate, the input context grows longer, increasing
the computational cost with each hop. Second, relational explo-
ration typically operates over flat textual contexts without explicitly
modeling structural relationships among entities, as in graphs. For
questions involving complex entity relationships, LLMs must re-
cover and combine relationships scattered across multiple passages.
Compared with explicit graph structures, this increases the risk of
missing important relationships and may require additional LLM
reasoning, further increasing computational costs.
These limitations raise a key question:Can we combine structural
information with question semantics for multi-hop retrieval at low
computational cost?Our strategy is to decouple semantic guidance from structural
exploration, which exploration-based methods typically handle
together through LLM generation. Instead of relying on LLMs to
generate reasoning at every hop, we use LLMs to provide question-
aware semantic signals and graph-based computation to explore
structural relationships.
Semantic guidance.The first challenge is whether we can
leverage LLMs’ semantic understanding capabilities without re-
lying on LLM generation. We revisit how LLMs resolve implicit
entity references. Interestingly, an LLM resolves implicit entity
references while reading the passages and before generating any
token [ 2,12,41]. This resolution is concentrated in a handful of
attention heads, which we calladdressing heads(Section 3). When
an LLM reads retrieved passages followed by a multi-hop question,
these heads move information from the mentions of each described
entity into the words that describe it. Forthe author of Princeps
Pastorum, this information comes from the mention of John XXIII
in the passage about Princeps Pastorum. One forward pass does this
for the bridges of several hops, each at the words of its own descrip-
tion. The heads can be identified without labels, and ablating them
halves the LLM’s accuracy on two-hop questions whose bridge is
described. In one forward pass, the addressing heads thus score
how strongly the question refers to each entity in the passages and
to each of its mentions.
Structural exploration.The next challenge is how to use these
semantic signals to explore relationships among entities without
relying on LLM-generated reasoning. To this end, we represent pas-
sages and their shared entities in a description-labelled hypergraph
and use the addressing scores to steer Personalized PageRank (PPR).
The hypergraph explicitly captures structural connections among
passages through shared entities, while the addressing scores guide
the walk toward entities and mentions relevant to the question. In
this way, multi-hop relational exploration is performed through
graph-based computation rather than step-by-step LLM generation.
HyperAddress, short for Hyper graph retrieval via in-context
Address ing, implements this strategy through two components:
an offline description-labelled hypergraph and online attention-
steered PageRank.
Our contributions are summarized as follows:
•We show that an LLM reading retrieved passages with a
multi-hop question identifies the entity that each descrip-
tion refers to through a handful of addressing heads. These
heads can be identified without labels, and the LLM depends
on them when a bridge is described. They cover several
hops in one forward pass.
•We proposeHyperAddress, a retriever that runs Personal-
ized PageRank on a description-labelled hypergraph and
takes both its restart distribution and its transition weights
from the addressing heads.
•On three multi-hop benchmarks,HyperAddressoutper-
forms the strongest of nine baselines, PropRAG, by 3.5
points of Recall@5 and 7.4 points of All@5 on average. Its
gains grow with the number of hops, to 6.3 points at four
hops on MuSiQue. It generates no token at query time, an-
swers more than twice as fast, and builds its index at half
the cost.

Read the Route: Attention-Steered Hypergraph Retrieval for Multi-hop RAG
2 Related Work
Graph-based RAG.Graph-based RAG systems differ in what their
index stores. Hierarchical indexes store summaries: RAPTOR [32]
recursively clusters and summarizes passages into a tree, and Mi-
crosoft’s GraphRAG [ 7] summarizes the communities of an LLM-
extracted entity graph. Knowledge-graph indexes store relations be-
tween entities. HippoRAG [ 16] and HippoRAG 2 [ 17] extract them
as open information extraction triples, and LightRAG [ 13] writes
LLM-generated descriptions on entities and edges. G-Retriever [ 19]
and GFM-RAG [ 26] run graph neural networks over such graphs.
Other indexes store propositions [ 37] or groups of related facts [ 43],
and HopRAG [ 25] connects passages with LLM-generated ques-
tions. Relation-free indexes record only which entities co-occur:
LinearRAG [ 47] links passages and sentences through the enti-
ties they mention, and HGRAG [ 36] treats entities as vertices and
passages as hyperedges.
Multi-hop traversal.Multi-hop retrieval methods also differ
in what decides each hop. In diffusion-based methods, relevance
spreads along the edges of the graph from the entities and passages
that match the question. HippoRAG, HippoRAG 2, and Linear-
RAG use Personalized PageRank [ 3,18], and HGRAG uses hyper-
graph diffusion. HippoRAG 2 filters the seed facts with an LLM,
and GFM-RAG learns query-dependent propagation from labelled
questions. PropRAG [ 37] searches proposition paths with a beam
search that scores each step by embedding similarity, without an
LLM. In exploration-based methods, an LLM decides each hop.
IRCoT [ 35] and Self-Ask [ 30] interleave retrieval with generated
reasoning or follow-up questions, and Think-on-Graph [ 33] and
HopRAG let the LLM choose the next node. ChainRAG [ 46] rewrites
each sub-question with the entity found at the previous hop, and
BridgeRAG [ 1] ranks later-hop candidates with an LLM judge con-
ditioned on the bridge evidence. EfficientRAG [ 48] trains small
models to replace the LLM in this loop.
Attention as a retrieval signal.Attention heads have been
used to retrieve information from the context. Retrieval heads copy
information from long inputs [ 38]. ICR [ 4], QRHead [ 44], and Block-
Rank [ 14] score the passages of a prompt by the attention that the
query pays to them. LOCOS [ 10] selects heads by their logit con-
tribution rather than their attention and finds heads that retrieve
non-literal matches. The attention of a question-answering reader
has also been distilled into a retriever [ 21]. Interpretability stud-
ies describe how models resolve references in context: through
binding IDs [ 8], through lookback mechanisms that retrieve infor-
mation bound at an earlier position [ 29], or through a mixture of
positional and lexical mechanisms [ 15]. Other studies trace how a
model resolves the bridge entity of a two-hop question [ 2,12,41].
3 In-Context Addressing in Multi-hop RAG
An LLM that reads retrieved passages together with a multi-hop
question identifies the entity that each description in the question
refers to before it generates anything. Exploration-based retrieval,
however, waits for the LLM to generate every hop. Since a Trans-
former moves information between positions only through atten-
tion, this identification relies on attention heads [ 8,15,29]. Before
introducing our method, we therefore analyze the attention of an
LLM while it reads the passages and the question. The analysis uses
did
the
rx
350
model
of
the
luxury
division
of
the
company
that
built
Daihatsu
boon
change
body
styleLexus RX 350
Lexus (hop 2)
Toyota Motor Corp. (hop 1)
Daihatsu Boonhop 2 hop 1 last word
02≥4
pointer scoreFigure 2: Pointer scores of the addressing heads at each word
of a three-hop MuSiQue question. Each bridge scores highest
at its own description and low at the last word.
Table 1: Reader accuracy (%) after mean ablation of a head set.
Probes: 400 held out, gold name not in context. MuSiQue: 304
two-hop test questions with two gold passages and four dis-
tractors. Random heads: change in points over three draws.
MuSiQue, bridge
ablated heads probes described named
none 76.5 40.1 62.5
8 addressing heads 48.5 19.7 53.6
at the description only — 27.0 —
at as many other tokens — 37.2 —
8 random, layers 18–22−2.8to−1.5−4.0to−2.0−3.0to−1.3
8 random, any layer−0.8to+0.8−1.0to+0.7−1.3to+1.0
Figure 2, a three-hop question from MuSiQue [ 34], as a running
example. Each observation is measured over hundreds of held-out
probes or test questions, and Appendix A.1 gives further results.
Analysis Setup.Our reader is Qwen3-4B [ 39]. It reads the
top passages of a first-stage retriever followed by the question
in one prompt and generates no token. The embedding model is
NV-Embed-v2 [ 23]. For a head ℎ, a question token 𝑡, and a candidate
entity𝑒whose mentions occupy the positions 𝑀𝑒, we measure the
share of the head’s output at𝑡that comes from these mentions:
𝑠ℎ(𝑡,𝑒)=Í
𝑗∈𝑀 𝑒𝛼ℎ(𝑡,𝑗)𝑊ℎ
𝑂𝑣ℎ(𝑗)
Í
𝑗𝛼ℎ(𝑡,𝑗)𝑊ℎ
𝑂𝑣ℎ(𝑗),(1)
where𝛼ℎ(𝑡,𝑗) is the attention weight, 𝑣ℎ(𝑗)the value vector, and
𝑊ℎ
𝑂the head’s slice of the output projection. Unlike an attention
weight, this share measures how much the mentions contribute
to what the head writes at 𝑡. Thepointer scoreof 𝑒ismax𝑡𝑠ℎ(𝑡,𝑒) ,
z-normalized over the candidate entities and averaged over a set of
heads𝐻. We select the heads without any benchmark label, from
probes generated from the index of the corpus. A probe consists of
a passage, its nine nearest passages, and a question built around one
description that the passage gives of an entity (Section 4.1); its target
is that entity. The eight of the 1,152 heads with the highest mean
reciprocal rank of the target on 600 probes are theaddressing heads.
Figure 2 shows their pointer scores at every word of a question.
Observation 1: Sparse Addressing Heads.Our first key ob-
servation is that a handful of mid-layer heads identify the entity
that a description refers to, and that the reader depends on them.
The eight addressing heads all lie in layers 18 to 22 of 36, and on
400 held-out probes they rank the target first for 59.0%. Table 1

Liu et al.
replaces their output by its mean. The reader’s accuracy then falls
from 76.5% to 48.5% on the probes and halves on two-hop MuSiQue
questions with a described bridge, from 40.1% to 19.7%. With the
bridge named, it falls by 8.9 points. The reader thus depends on
these heads most when it must identify a described entity (differ-
ence of drops 11.5, 95% CI [3.9,19.1]). Ablating the heads at the
words of the description alone costs 13.2 of the 20.4 points, and
at as many other tokens 3.0. Eight random heads cost at most 4.0
points from the same layers and 1.3 from any layer. This sparsity
motivates a reader that uses only the addressing heads and stops
its forward pass after layer 22, with heads that each corpus selects
from its own probes (Section 4.2).
Observation 2: Word-Level Multi-hop Addressing.Our sec-
ond key observation is that one forward pass identifies the bridges
of several hops, each at the words of its own description. Figure 2
shows a question that describes each bridge through the one before
it. A single embedding of this question mixes the hops, whereas the
pointer scores separate them by word. We place the gold passages
of each three- and four-hop MuSiQue question among ten passages
and rank the entities that the question does not name. All bridges of
a question are among the five highest-scoring candidates for 79.8%
and 75.6% of the questions, and each bridge alone for 85.0% to 89.8%.
These five candidates set the restart of the walk (Section 4.2). Each
bridge scores highest inside the words of its own description, the
part that no inner description covers, 2.2 to 2.7 times as often as the
length of that part predicts. At the later hops, the highest-scoring
entity at these words is the bridge that they describe about twice
as often as chance (64.0% against 34.0% at the second hop of three,
53.5% against 23.0% at the third of four).
By contrast, scores read only at the last question token, where
an answer would be generated and where BlockRank [ 14] reads
relevance, rank the later bridges much lower. For the second bridge
of three-hop questions, the top-five rate falls from 85.3% to 52.3%.
For the third bridge of four-hop questions, it falls from 85.0% to
36.2%, whereas at the first hop the rates fall by 9.7 and 11.8 points.
With the last token, all bridges are in the top five for 42.3% and
16.5% of the questions. Reading the pointer scores at every word
thus gives, in one forward pass, a score for each candidate entity
and for each of its mentions. This motivates a walk that takes both
its restart and its transitions from these scores (Section 4.2).
Observation 3: Context-Dependent Addressing.Our third
key observation is that this identification depends on reading the
passages in context. An index encodes each passage once, apart
from any question, which is what lets it scale. Table 2 tests whether
the identification survives such encoding. In context, the addressing
heads rank the target first for 59.0% of the probes and 55.3% of the
MuSiQue bridges, and the embeddings for at most 37.0% and 34.2%.
The paired gains in mean reciprocal rank are .174 and .198 on
the probes (95% CIs above 0). With either side encoded apart (the
question, or each passage, as in caches of precomputed passages [ 6,
27,40]), the same heads fall below both embeddings. With both
apart, their mean reciprocal rank falls by .403 and .383 (95% CIs
above 0), and they keep 29% of their margin over chance on both sets.
Position does not explain the loss. Every encoding keeps the token
positions of the joint prompt, and removing the rotary position
encoding from the heads changes their mean reciprocal rank by
less than .01 (Appendix A.1). End to end, scoring entities by theTable 2: Mean reciprocal rank (top-1 %) of the target among
the candidate entities. MuSiQue: bridges whose gold passage
is not in the walk’s top five.
probes (400) MuSiQue (304)
read in context
addressing heads .734 (59.0) .705 (55.3)
encoded apart
embedding, question and name .560 (37.0) .529 (34.2)
embedding, question and passages .537 (34.0) .502 (30.9)
addressing heads, question apart .482 .440
addressing heads, passages apart .512 .470
addressing heads, both apart .332 .323
chance .169 .160
similarity of the question to their stored descriptions recovers 1.8 of
the 8.4 points that the addressing heads add (Section 5.3). Because
the identification depends on context,HyperAddresscomputes it
at query time and stores only descriptions in the index (Section 4.1).
4 Attention-Steered Hypergraph Retrieval
Based on these observations,HyperAddressworks in two stages
(Figure 3). Offline, it organizes the corpus into adescription-labelled
hypergraph, which stores each relation on the incidence between
a passage and an entity (Section 4.1). Online, it retrieves with
Attention-Steered PageRank, a random walk on this hypergraph
whose restart and transition weights are set by the addressing
heads of the reader (Section 4.2).
Problem setting.Given a corpus Pof passages and a question
𝑞, a retriever ranks the passages so that the evidence for 𝑞appears
among the first 𝑘. For a multi-hop question, part of this evidence is
about entities to which𝑞refers only through descriptions.
4.1 Offline: Description-Labelled Hypergraph
Definition.A description-labelled hypergraph H=(P,E,𝑤,𝐷)
has the passages as its vertices and a set Eof hyperedges, one per
entity, each containing the passages that mention its entity. A pair
(𝑝,𝑒) with𝑝∈𝑒 is anincidence; it has a weight 𝑤(𝑝,𝑒)> 0and a
label𝐷(𝑝,𝑒) : the descriptions by which passage 𝑝identifies entity
𝑒without naming it. A relation is thus stored as the label of the
incidence between the passage that states it and the entity that it
concerns, rather than as an edge between two entities.
Construction.An LLM reads every passage once and writes
itsaddress book. For each entity that the passage mentions, an
entry gives the canonical name of the entity, its mentions in the
passage, and one to three descriptions. Each description identifies
the entity through a fact stated in the passage and is phrased the way
a question would refer to it. Canonical names and their aliases group
the entries of all passages into hyperedges. Each hyperedge also
contains every passage that names its entity, and the descriptions
label its incidences. An incidence has the weight 𝑤(𝑝,𝑒)=|𝑒|−1/2,
so that the walk moves less probability through an entity that
many passages mention. The LLM also writes the facts that the
descriptions leave out and a question that the passage answers.
Together with the passage’s own sentences and one card per entity,
which lists the entity’s name with its descriptions, these form the
content itemsof the passage.

Read the Route: Attention-Steered Hypergraph Retrieval for Multi-hop RAG
R = 2 roundsD e s c r i p t i o n - l a b e l l e d  h y p e r g r a p hπ = (1-d)r + dTw  πTOnline: attention-steered PageRankOffline: an LLM writes 
an address bookPrinceps Pastorumis 
is an encyclical of 
Pope John XXIll, 
issued in 1 959 ...e v ery passageWalk before reading1restart r0 (q): dense similarityJohn XXlll2Read [S; q] onceLM headweights w(p, e) = |e|- 1 / 2rank by π·top 3 on screen 1P r i n c e p s  P a s t o r u m234qWhere did the author of 
Princeps Pastorum die?L23-36 
not run
...S...encyclical of Pope John XXIll, issued...pointer scores ρq (e)John XXlll1 959...L1 8 -22addressing 
headsbiographyPrinceps PastorumbiographyWalk before reading3restart from q, top-n ρq , and q’John XXlllwq (p, e) = w(p, e) e x p( βαq (p, e))rank by π·top 3 retrie v ed 1biography2P r i n c e p s  P a s t o r u m3443Steerrestart+
incidence weightsL1-1 7
...LLMaddress boo k
John XXIl l
the author of Princeps Pastorum 
mention " Pope John XXIIl "  e v ery 
entity of e v ery passagelabels the incidencesreader· Q wen3-4 B36 layers × 32 headsbetween roundsRewriteOne forward pass per round .
N o generated tokens.ne x t roundq’ Where did John XXlll die?description → entity namePrinceps PastorumPPR massbiographygold passager e s t a r t  f r o m  q ,  q ’ρqRead in context. Steer retrieval without generation.
Descriptions label incidences; addressing heads guide the walk.passagesattentionretrie v al
Figure 3: Overview ofHyperAddress. Offline, an LLM labels each incidence with descriptions. Online, each of 𝑅rounds walks
(➀), reads the passages in one forward pass ( ➁), steers restart and transitions ( ➂), and rewrites one description into a name ( ➃).
Design rationale.A knowledge graph makes entities its nodes
and extracted relations its edges. Hinverts these roles: passages are
the vertices, entities the hyperedges, and what a passage says about
an entity stays on their incidence, in the words of the passage. This
placement serves both stages. Offline, a label is phrased the way a
question refers to an entity. The initial restart can therefore match
a description and bring the passage that states the relation into
the reader’s context. Online, the reader identifies the entity from
the text of that passage at query time (Observation 3). Since two
passages are connected only through an entity, the reader’s scores
re-weight every transition out of a passage that it has read. A new
passage is added with its own address book and joins the hyperedges
of the entities that it names. No other passage is processed again,
so the index supports non-parametric continual learning [17].
4.2 Online: Attention-Steered PageRank
HyperAddressranks the passages with Personalized PageRank on
Hand lets the reader set both of its inputs, the restart distribution
and the transition weights, over𝑅rounds (Algorithm 1).
Walk.The walk alternates between passages and hyperedges:
from a passage 𝑝it moves to a hyperedge 𝑒∋𝑝 with probabil-
ity proportional to 𝑤(𝑝,𝑒) , and from a hyperedge 𝑒to a passage
𝑝∈𝑒 with probability proportional to 𝑤(𝑝,𝑒) . This is a random
walk on a hypergraph with edge-dependent vertex weights [ 5,45].
Given a restart distribution 𝑟over passages and hyperedges and
the transition matrix𝑇 𝑤of the walk, the stationary scores satisfy
𝜋=(1−𝑑)𝑟+𝑑𝑇⊤
𝑤𝜋,(2)
with damping factor 𝑑. The initial restart 𝑟0(𝑞)is set before any
reading. It places probability on every passage in proportion to
its dense similarity to 𝑞, on the incidences whose labels are most
similar to𝑞, and on the passages of the content items most similar
to𝑞.Rank(𝜋) orders the passages by their stationary score, fused
with their dense similarity to the original question𝑞 0.Algorithm 1Attention-Steered PageRank
Require: hypergraphH=(P,E,𝑤,𝐷) , question𝑞, reader with address-
ing heads, rounds𝑅, passages per round𝑚
Ensure:a ranking ofP
1:𝑞 0←𝑞;𝑟←𝑟 0(𝑞);𝑤 𝑞←𝑤
2:𝜋←PPR(𝑟,𝑤 𝑞)⊲Eq. 2
3:for𝑡=1,...,𝑅do
4:𝑆←the first𝑚passages ofRank(𝜋)
5:(𝜌 𝑞,𝑎𝑞)←Read(𝑆,𝑞)⊲one forward pass
6:steer the restart𝑟with𝜌 𝑞 ⊲Eq. 3
7:steer the weights𝑤 𝑞of the incidences on𝑆with𝑎 𝑞 ⊲Eq. 4
8:if𝑡<𝑅then⊲between rounds
9:𝑞←Rewrite(𝑞); add𝑟 0(𝑞)to𝑟
10:𝜋←PPR(𝑟,𝑤 𝑞)
11:returnRank(𝜋)
In-context addressing.In each round, the reader processes the
set𝑆of the𝑚top-ranked passages, in their original text, followed
by the question. It does so in one forward pass that stops after the
last layer with an addressing head and generates no token. The
candidates are the entities that have an incidence with a passage in
𝑆and that𝑞does not name. For each candidate 𝑒, the heads give
its pointer score 𝜌𝑞(𝑒), computed from Equation (1)as in Section 3.
For each incidence (𝑝,𝑒) with𝑝∈𝑆 , the same score restricted to
the mentions of𝑒inside𝑝gives its incidence score𝑎 𝑞(𝑝,𝑒).
Steering the restart.The 𝑛candidates with the highest pointer
scores receive a share 𝜆of the restart probability, split between
their hyperedges and the passages in𝑆that mention them:
𝑟←(1−𝜆)𝑟+𝜆∑︁
𝑒∈top𝑛(𝜌𝑞)softmax(𝜌 𝑞)𝑒1
2 1𝑒+𝑢𝑒,(3)
where1𝑒is the indicator of hyperedge 𝑒and𝑢𝑒the uniform distri-
bution over the passages in 𝑆that mention 𝑒. The walk then restarts

Liu et al.
both from the passages similar to the question and from the entities
that the question refers to.
Steering the transitions.Every incidence that the reader has
read is re-weighted by its incidence score, in both directions:
𝑤𝑞(𝑝,𝑒)=𝑤(𝑝,𝑒)exp 𝛽𝑎𝑞(𝑝,𝑒),(4)
and𝑇𝑤𝑞replaces𝑇𝑤in Equation 2. A passage keeps the incidence
scores of the last round in which it was read. A passage in 𝑆thus
passes more of its probability to the entities that the question refers
to, instead of distributing it by the fixed weights.
Rounds and cost.Between rounds,HyperAddressreplaces
one description in the question with the name of the entity that it
most likely refers to, selected by the pointer scores and the labels
(Appendix A). The dense terms of the restart can then match this
name. The rewritten question adds its own restart and is the ques-
tion read in the next round. The reader sees the passages as plain
text. Appending the labels of their incidences does not change Re-
call@5 significantly (82.8 against 83.1), because each label is drawn
from a passage that the reader already reads.HyperAddressruns
two rounds. All generation happens offline, when the index is built.
At query time,HyperAddressencodes the question and, in each
round, runs one forward pass of the reader and one walk on the
hypergraph. Section 5.4 measures the cost of both stages.
5 Experiments
We evaluateHyperAddressalong five research questions.RQ1:
does it retrieve the evidence better than diffusion- and exploration-
based methods?RQ2: where does its gain come from?RQ3: what
does it cost to build and to query?RQ4: do better passages yield
better answers?RQ5: does it work with other readers?
5.1 Setup
Datasets and metrics.We follow the protocol of HippoRAG 2 [ 17]
on three multi-hop benchmarks: MuSiQue [ 34], whose questions
take two to four hops, 2WikiMultiHopQA (2Wiki) [ 20], and Hot-
potQA [ 42]. Each benchmark provides 1,000 questions and a cor-
pus of 11,656, 6,119, and 9,811 passages, respectively, which every
method indexes as a whole. The first 200 questions of each bench-
mark form its development set, and the remaining 800 its test set.
Every design choice and constant ofHyperAddresswas fixed on
the development questions of MuSiQue and kept for all bench-
marks. Baselines run with their released configurations. All results
are on the test questions; Appendix A.3 reports all 1,000 questions,
where our reproductions match the published numbers. Recall@ 𝑘
(𝑘= 2,5) is the fraction of gold passages among the top 𝑘, and
All@5 the fraction of questions whose gold passages all appear in
the top five. For question answering, Qwen3.5-27B answers every
question from the top five passages of each retriever, and we report
exact match (EM) and F1. Differences are tested over questions with
a paired bootstrap at the 95% level.
Baselines.We compareHyperAddresswith four families of
retrievers. (i) Dense retrieval: NV-Embed-v2 [ 23]. (ii) Graph-based
RAG: RAPTOR [ 32] retrieves over a tree of recursive summaries;
HippoRAG 2 [ 17] walks an LLM-extracted knowledge graph with
Personalized PageRank from LLM-filtered seeds; PropRAG [ 37]
searches paths of LLM-extracted propositions with an LLM-freebeam search; HGRAG [ 36] diffuses relevance over a hypergraph of
entities and passages; and LinearRAG [ 47] propagates relevance
over a relation-free graph of entities, sentences, and passages.
(iii) Exploration: IRCoT [ 35] interleaves retrieval with chain-of-
thought generation, with at most four steps on MuSiQue, two on the
other benchmarks, and ten passages per step, as in HippoRAG [ 16].
(iv) Attention-based re-ranking: ICR [ 4] and QRHead [ 44] re-rank
the top twenty passages of the dense retriever by the attention
that the question pays to them, both with Qwen3-4B, the reader of
HyperAddress. Every baseline uses NV-Embed-v2 and, wherever
it needs an LLM, Qwen3.5-27B [ 31], the LLM that writes the index
ofHyperAddress. All run with their official code, except HGRAG,
which releases none and which we reimplement from its paper
(Appendix A.5). We also run IRCoT with Qwen3-4B to compare
generation and reading on the same model.
Implementation.Qwen3.5-27B writes the index. The reader is
Qwen3-4B [ 39], whose eight addressing heads are selected per cor-
pus from 600 synthetic probes (Section 3). NV-Embed-v2 encodes
passages, descriptions, content items, and questions. All experi-
ments run on one NVIDIA RTX PRO 6000 GPU (96 GB).
5.2 RQ1: Retrieval Effectiveness
Table 3 reports the retrieval results.HyperAddressranks first on
every benchmark and every metric. Averaged over the three bench-
marks, it improves Recall@5 by 3.5 points and All@5 by 7.4 points
over the strongest baseline, PropRAG. Every improvement over
the strongest baseline of its column is significant (paired bootstrap,
95%).HyperAddressalso ranks first on the single-hop questions
of NQ and PopQA (Appendix A.4). We highlight three results.
Gains grow with question depth.The margin over PropRAG
is widest on MuSiQue, whose questions take up to four hops (5.3
points of Recall@5 and 8.7 of All@5), and narrowest on HotpotQA,
whose questions take two (2.1 and 4.2). Within MuSiQue, the margin
also grows with the number of hops (Section 5.3).
More questions get complete evidence.On every benchmark,
the gain in All@5 exceeds the gain in Recall@5:HyperAddress
retrieves not only more of the passages that a question needs, but
also more often all of them. On 2Wiki, 92.8% of the questions have
their complete evidence in the top five, against 86.1% with IRCoT
on the 27B model, the strongest baseline on this metric.
Steering outperforms reasoning and re-ranking.IRCoT and
HyperAddresscan run on the same 4B model. IRCoT generates
a reasoning step for every hop, whereasHyperAddressreads the
passages without generating. With this model,HyperAddressis
ahead by 5.9 points of Recall@5 and 10.4 of All@5 on average, and a
27B model does not close the gap. The same reader can re-rank the
dense passages by attention, as ICR and QRHead do.HyperAddress
is ahead of the better of them by about 9 points of Recall@5 and
17 of All@5, because re-ranking reorders the passages, whereas
steering reaches the passages that the dense retriever misses.
5.3 RQ2: Where the Gain Comes From
Figure 4 decomposes the gain ofHyperAddresson MuSiQue, the
benchmark with the deepest questions, by number of hops, and Ta-
ble 7 in Appendix A.2 lists every variant. The gain grows with depth

Read the Route: Attention-Steered Hypergraph Retrieval for Multi-hop RAG
Table 3: Retrieval results (%) on the test questions. All@5: questions with all gold passages in the top five. Gen.: generates text
at query time. Best in bold, second best underlined.
MuSiQue 2Wiki HotpotQA Average
Method Gen. R@2 R@5 All@5 R@2 R@5 All@5 R@2 R@5 All@5 R@5 All@5
Dense retrieval
NV-Embed-v2✗52.9 69.9 38.9 69.1 76.8 51.1 84.0 94.8 89.8 80.5 59.9
Graph-based RAG
RAPTOR✗41.7 59.4 35.4 60.8 67.8 61.4 72.3 83.6 78.5 70.3 58.4
HippoRAG 2✓56.1 75.3 48.5 77.3 93.2 81.5 80.4 95.8 92.5 88.1 74.2
PropRAG✗57.5 77.8 53.3 71.1 94.4 83.6 84.9 96.3 92.9 89.5 76.6
HGRAG✓53.1 72.9 48.5 71.2 91.4 79.4 82.3 95.7 91.6 86.7 73.2
LinearRAG✗54.7 73.8 48.2 75.8 92.2 85.3 81.1 96.0 92.2 87.3 75.2
Exploration-based retrieval
IRCoT (Qwen3-4B)✓56.1 76.4 47.8 77.0 90.8 82.8 84.3 94.0 90.1 87.1 73.6
IRCoT (Qwen3.5-27B)✓58.3 78.9 49.6 78.1 92.5 86.1 85.0 94.8 91.8 88.7 75.8
Attention-based re-ranking
ICR (Qwen3-4B)✗48.1 67.1 36.5 66.8 75.3 48.6 79.8 92.2 85.1 78.2 56.7
QRHead (Qwen3-4B)✗53.3 74.4 48.0 70.0 81.8 59.1 82.4 95.9 92.5 84.0 66.5
HyperAddress(Qwen3-4B) ✗ 61.7 83.1 62.0 80.6 97.5 92.8 87.3 98.4 97.1 93.0 84.0
H/y.sc/p.sc/e.sc/r.scA/d.sc/d.sc/r.sc/e.sc/s.sc/s.sc steering
onlyrewriting
only0510asteer or rewrite
H/y.sc/p.sc/e.sc/r.scA/d.sc/d.sc/r.sc/e.sc/s.sc/s.sc transitions
onlyrestart
onlybrestart or transitions
H/y.sc/p.sc/e.sc/r.scA/d.sc/d.sc/r.sc/e.sc/s.sc/s.sc name:
steer + rewritename:
rewrite0510cpoint or name
H/y.sc/p.sc/e.sc/r.scA/d.sc/d.sc/r.sc/e.sc/s.sc/s.sc last token
onlyembeddingsda/t_tention at every wordgain in Recall@5 (points)2 hops 3 hops 4 hops
Figure 4: Gain in Recall@5 over the walk without reading,
by number of hops (MuSiQue test). Indigo and lilac: the ad-
dressing heads set the transitions from every word.
when the addressing heads, read at every word of the question, set
the transitions of the walk.
Steering adds 8.4 points.Without reading, the walk on the
hypergraph reaches 74.7 points of Recall@5, close to HippoRAG 2
(75.3) and LinearRAG (73.8). A walk with fixed transitions thus
reaches a similar level whether its graph stores triples, co-occur-
rences, or descriptions. Steering the walk with the addressing heads
adds 8.4 points, and the gain grows with the number of hops: 7.6
points at two hops, 8.8 at three, and 10.1 at four. The margin over
IRCoT with the 27B model, the strongest baseline on MuSiQue,
grows from 3.6 points at two hops to 6.7 at four. Over PropRAG it
grows from 4.5 to 6.3, and over HippoRAG 2 from 6.0 to 11.6.
Most of the gain comes from steering.One forward pass
of the reader, with no generation and no rewrite, raises Recall@5
from 74.7 to 80.1 (Table 5), 5.4 of the 8.4 points. A second roundof steering reaches 81.7, and rewriting the question between the
two rounds reaches 83.1. Rewriting alone, without steering, reaches
77.3, and its gain does not grow with depth (Figure 4a). At four
hops it adds 1.8 points to the walk, whereas steering alone adds
8.5. A rewrite names one entity per round, as in ChainRAG [ 46].
One forward pass, in contrast, scores the bridges of several hops
(Observation 2), and steering passes all of these scores to the walk.
A third round adds 2.9 points at four hops, where a third bridge
remains, and less than one point at two and three hops.
Both inputs of the walk contribute.The steered restart re-
ceives 0.02 of the restart probability, on at most five entities and the
passages in𝑆that mention them. Without it ( 𝜆=0), Recall@5 drops
by 2.3 points, similarly at every depth (2.2, 2.6, and 2.0). Without
the steered transitions ( 𝛽=0), it drops by 2.4 points: 1.3 at two
hops, 3.0 at three, and 4.9 at four. With steered transitions alone,
the gain still grows with the number of hops; with a steered restart
alone, it shrinks (Figure 4b). The restart thus helps at every depth,
and the transitions help more with every additional hop.
Pointer scores outperform a generated name.Qwen3-4B can
instead generate the name of the bridge from the same passages, and
the name can replace the pointer scores (Appendix A). Rewriting the
question with the generated name reaches 78.1 points of Recall@5,
5.0 belowHyperAddressand 7.5 below at four hops. Using the
name for both rewriting and steering reaches 80.9, and the gap to
HyperAddressgrows with the number of hops (Figure 4c), to 53.5
against 57.9 at four. A name selects one entity, whereas one forward
pass scores every candidate entity and every incidence in 𝑆, and
steering uses all of these scores. Steering with the pointer scores
alone, without any rewrite, already reaches 81.7.
Steering outperforms re-ranking the walk.The reader can
also re-rank the passages that the walk finds. With the same Qwen3-
4B, QRHead re-ranks the walk’s top twenty passages to 80.0, 96.0,
and 97.1 points of Recall@5 on MuSiQue, 2Wiki, and HotpotQA,
below the 83.1, 97.5, and 98.4 ofHyperAddress. ICR falls below the
walk itself on all three. Re-ranking reorders the passages that the

Liu et al.
0 2 4
index: GPU hours607080Recall@5 costs more,
retrieves less+9.3
NV-Embed-v2
RAPTORHippoRAG 2PropRAG
HGRAGLinearRAGH/y.sc/p.sc/e.sc/r.scA/d.sc/d.sc/r.sc/e.sc/s.sc/s.sc
index tokens
2M 6M
0.1 1 10
query: seconds per question+9.3
RAPTORHippoRAG 2PropRAG
HGRAGLinearRAGH/y.sc/p.sc/e.sc/r.scA/d.sc/d.sc/r.sc/e.sc/s.sc/s.sc
NV-Embed-v2IRCoT
tokens per question
100 400
Figure 5: Recall@5 on MuSiQue against indexing time (left)
and query time (right, log scale) on one GPU. Marker area:
generated tokens. IRCoT retrieves from the dense index.
walk has already found, whereas steering changes which passages
the next walk reaches.
The scores need every word and the addressing heads.Scor-
ing only the last question token, where an answer would start,
loses 2.2 points, and 4.9 at four hops (Figure 4d). Later bridges score
highest at the words of their own descriptions, not at the last token
(Observation 2). Scoring the entities by the similarity between the
question and their descriptions, instead of by the addressing heads,
loses 6.6 of the 8.4 points, because the identification depends on
context (Observation 3). The choice of heads also matters. Heads of
the same reader selected as in QRHead, or by the logit contribution
of LOCOS [ 10], steer the same walk over the same candidates and
keep 2.4 and 1.2 of the 8.4 points. Both head sets still improve the
walk at two hops, where QRHead reaches 88.0 against 84.7. At four
hops, they fall to the level of the walk without reading (50.0 and
47.8, against 47.8), whereasHyperAddressreaches 57.9. Heads
selected for query relevance thus help on the first hop, but later
hops need heads that identify the entity each description refers to.
The labels set where the walk starts.The labels enter retrieval
through the initial restart. Removing them from the restart, with
the rest of the index unchanged, lowers Recall@5 by 3.2 points, and
the loss grows with depth, from 1.3 points at two hops to 9.1 at four.
The wording of the labels matters. Replacing each label with the
name of its entity, so that an incidence only records that a passage
mentions the entity, costs 2.2 points. The rewrite uses the labels:
selecting the span to rewrite without them costs 1.1 points.
The address book matters once the walk is steered.We
replace the address book with the entities that spaCy recognizes
in each passage, without descriptions or content items, and keep
the reader and its heads. Without reading, the two indexes perform
alike (74.2 with spaCy, 74.7 with the address book). After one round
of steering, the address book is ahead by 4.2 points (80.1 against
75.9, Table 5), and raises the gain of one round from 1.7 to 5.4 points.
5.4 RQ3: Indexing and Query Cost
Figure 5 plots the Recall@5 of each method on MuSiQue against its
cost, once to build its index and then for every question; Table 8 in
Appendix A.2 lists every cost.HyperAddressbuilds its index, head
selection included, with less than half the tokens that HippoRAG 2Table 4: Question answering (EM/F1, %) by Qwen3.5-27B from
the top five passages of each retriever.
Retriever MuSiQue 2Wiki HotpotQA Average
NV-Embed-v2 35.9/46.6 54.6/60.4 63.5/76.6 51.3/61.2
HippoRAG 2 41.1/53.2 64.2/72.4 64.4/78.1 56.6/67.9
PropRAG 42.7/53.5 65.0/71.4 64.9/78.3 57.5/67.7
LinearRAG 41.0/51.5 64.8/70.8 65.1/77.8 57.0/66.7
IRCoT (4B) 41.3/50.9 63.2/70.8 64.0/77.5 56.2/66.4
IRCoT (27B) 42.9/55.1 64.7/73.1 64.8/78.3 57.5/68.8
HyperAddress 46.5/59.1 66.8/75.1 67.1/80.1 60.1/71.4
Table 5: Recall@5 (%) on MuSiQue with other readers, each
with eight addressing heads selected without labels. Walk
without reading: 74.7. 1 round: steering once, no rewriting.
Reader heads (layers) 1 roundHyperAddress
Qwen3-1.7B 16–17 of 28 79.3 81.5
Qwen3-4B 18–22 of 36 80.1 83.1
Qwen3-8B 18–23 of 36 80.5 83.5
Llama-3.1-8B-Instruct 13–14 of 32 80.5 83.5
and PropRAG generate and in about half their GPU time. At query
time, it generates no token and answers a question in 0.51 seconds.
PropRAG takes more than twice as long, HippoRAG 2 five times as
long, and IRCoT with the 27B model 23 times as long. With the same
4B model, IRCoT takes 15 times as long and retrieves 6.7 points less
Recall@5. Every cheaper index or faster method scores at least 9
points of Recall@5 belowHyperAddress.
5.5 RQ4: Question Answering
Better retrieval gives better answers. In Table 4, Qwen3.5-27B an-
swers every question from the top five passages of each retriever.
With the passages ofHyperAddress, it gives the best answers on all
benchmarks. On average, it is 2.6 points of EM above PropRAG and
2.6 of F1 above IRCoT with the 27B model, the strongest baselines
on these metrics. On MuSiQue, whose questions are the deepest, it
is 3.6 and 4.0 points above IRCoT with the 27B model.
5.6 RQ5: Other Readers
We find addressing heads in every reader that we test. We se-
lect eight heads for Qwen3-1.7B, Qwen3-8B, and Llama-3.1-8B-
Instruct [ 11] with the same label-free probes and keep the rest of
HyperAddressunchanged (Table 5). Steering improves the walk
with every reader. With 1.7B parameters,HyperAddressreaches
81.5 points of Recall@5, 2.6 above IRCoT with a 27B model, the
strongest baseline. The gain over the walk grows with the reader,
from 6.8 points with 1.7B parameters to 8.8 with either 8B model.
In all four models, the selected heads lie in the middle layers.
6 Conclusion
We presentedHyperAddress, a GraphRAG framework in which
the attention of an LLM steers a graph walk without generation.
The LLM reads the top passages together with the question in
one forward pass, and its addressing heads set both the restart
and the transitions of Personalized PageRank on a description-
labelled hypergraph. Without generating any token at query time,

Read the Route: Attention-Steered Hypergraph Retrieval for Multi-hop RAG
HyperAddressoutperforms nine baselines on three multi-hop
benchmarks, with gains that grow with the number of hops. On
these benchmarks, steering a graph walk with the attention of an
LLM retrieves more evidence than letting the LLM generate every
hop, in a fraction of the time.
References
[1] Andre Bacellar. 2026. BridgeRAG: Training-Free Bridge-Conditioned Retrieval
for Multi-Hop Question Answering. arXiv:2604.03384 [cs.IR] https://arxiv.org/
abs/2604.03384
[2]Eden Biran, Daniela Gottesman, Sohee Yang, Mor Geva, and Amir Globerson.
2024. Hopping Too Late: Exploring the Limitations of Large Language Models on
Multi-Hop Queries. InProceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing (EMNLP). arXiv:2406.12775 [cs.CL] https://arxiv.
org/abs/2406.12775
[3] Sergey Brin and Lawrence Page. 1998. The anatomy of a large-scale hypertextual
Web search engine.Computer Networks and ISDN Systems30, 1-7 (1998), 107–117.
doi:10.1016/S0169-7552(98)00110-X
[4]Shijie Chen, Bernal Jiménez Gutiérrez, and Yu Su. 2025. Attention in Large
Language Models Yields Efficient Zero-Shot Re-Rankers. InInternational Con-
ference on Learning Representations (ICLR). arXiv:2410.02642 [cs.CL] https:
//arxiv.org/abs/2410.02642
[5] Uthsav Chitra and Benjamin J Raphael. 2019. Random Walks on Hypergraphs
with Edge-Dependent Vertex Weights. InProceedings of the 36th International
Conference on Machine Learning (ICML). arXiv:1905.08287 [cs.LG] https://arxiv.
org/abs/1905.08287
[6] Michiel de Jong, Yury Zemlyanskiy, Nicholas FitzGerald, Fei Sha, and William
Cohen. 2022. Mention Memory: incorporating textual knowledge into Transform-
ers through entity mention attention. InInternational Conference on Learning
Representations (ICLR). arXiv:2110.06176 [cs.CL] https://arxiv.org/abs/2110.06176
[7]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2025. From Local to Global: A Graph RAG Approach to Query-Focused
Summarization. arXiv:2404.16130 [cs.CL] https://arxiv.org/abs/2404.16130
[8]Jiahai Feng and Jacob Steinhardt. 2024. How do Language Models Bind Enti-
ties in Context?. InInternational Conference on Learning Representations (ICLR).
arXiv:2310.17191 [cs.LG] https://arxiv.org/abs/2310.17191
[9]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi
Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented
Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL]
https://arxiv.org/abs/2312.10997
[10] Aryo Pradipta Gema, Beatrice Alex, and Pasquale Minervini. 2026.
Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads.
arXiv:2607.01002 [cs.CL] https://arxiv.org/abs/2607.01002
[11] Aaron Grattafiori et al .2024. The Llama 3 Herd of Models.
arXiv:2407.21783 [cs.AI] https://arxiv.org/abs/2407.21783
[12] Tianyu Guo, Hanlin Zhu, Ruiqi Zhang, Jiantao Jiao, Song Mei, Michael I. Jordan,
and Stuart Russell. 2025. How Do LLMs Perform Two-Hop Reasoning in Context?
arXiv:2502.13913 [cs.CL] https://arxiv.org/abs/2502.13913
[13] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG:
Simple and Fast Retrieval-Augmented Generation. InFindings of the Association
for Computational Linguistics: EMNLP 2025. arXiv:2410.05779 [cs.IR] https:
//arxiv.org/abs/2410.05779
[14] Nilesh Gupta, Chong You, Srinadh Bhojanapalli, Sanjiv Kumar, Inderjit
Dhillon, and Felix Yu. 2025. Scalable In-context Ranking with Genera-
tive Models. InAdvances in Neural Information Processing Systems (NeurIPS).
arXiv:2510.05396 [cs.IR] https://arxiv.org/abs/2510.05396
[15] Yoav Gur-Arieh, Mor Geva, and Atticus Geiger. 2026. Mixing Mechanisms:
How Language Models Retrieve Bound Entities In-Context. InInternational
Conference on Learning Representations (ICLR). arXiv:2510.06182 [cs.CL] https:
//arxiv.org/abs/2510.06182
[16] Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu
Su. 2024. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large
Language Models. InAdvances in Neural Information Processing Systems (NeurIPS).
arXiv:2405.14831 [cs.CL] https://arxiv.org/abs/2405.14831
[17] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025.
From RAG to Memory: Non-Parametric Continual Learning for Large Language
Models. InProceedings of the 42nd International Conference on Machine Learning
(ICML). arXiv:2502.14802 [cs.CL] https://arxiv.org/abs/2502.14802
[18] Taher H. Haveliwala. 2002. Topic-sensitive PageRank. InProceedings of the 11th
International Conference on World Wide Web (WWW). ACM, 517–526. doi:10.
1145/511446.511513[19] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V. Chawla, Thomas Laurent, Yann
LeCun, Xavier Bresson, and Bryan Hooi. 2024. G-Retriever: Retrieval-
Augmented Generation for Textual Graph Understanding and Question An-
swering. InAdvances in Neural Information Processing Systems (NeurIPS).
arXiv:2402.07630 [cs.LG] https://arxiv.org/abs/2402.07630
[20] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. InProceedings of the 28th International Conference on Computational
Linguistics (COLING). arXiv:2011.01060 [cs.CL] https://arxiv.org/abs/2011.01060
[21] Gautier Izacard and Edouard Grave. 2021. Distilling Knowledge from Reader
to Retriever for Question Answering. InInternational Conference on Learning
Representations (ICLR). arXiv:2012.04584 [cs.CL] https://arxiv.org/abs/2012.04584
[22] Tom Kwiatkowski et al .2019. Natural Questions: A Benchmark for Question
Answering Research.Transactions of the Association for Computational Linguistics
7 (2019), 452–466. doi:10.1162/tacl_a_00276
[23] Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi,
Bryan Catanzaro, and Wei Ping. 2025. NV-Embed: Improved Techniques for
Training LLMs as Generalist Embedding Models. InInternational Conference on
Learning Representations (ICLR). arXiv:2405.17428 [cs.CL] https://arxiv.org/abs/
2405.17428
[24] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. InAdvances in Neural Informa-
tion Processing Systems (NeurIPS). arXiv:2005.11401 [cs.CL] https://arxiv.org/
abs/2005.11401
[25] Hao Liu, Zhengren Wang, Xi Chen, Zhiyu Li, Feiyu Xiong, Qinhan Yu, and
Wentao Zhang. 2025. HopRAG: Multi-Hop Reasoning for Logic-Aware Retrieval-
Augmented Generation. InFindings of the Association for Computational Linguis-
tics: ACL 2025. arXiv:2502.12442 [cs.IR] https://arxiv.org/abs/2502.12442
[26] Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Dinh Phung, Chen Gong, and
Shirui Pan. 2025. GFM-RAG: Graph Foundation Model for Retrieval Augmented
Generation. InAdvances in Neural Information Processing Systems (NeurIPS).
arXiv:2502.01113 [cs.IR] https://arxiv.org/abs/2502.01113
[27] Dongyang Ma, Yan Wang, and Lan Tian. 2025. Block-Attention for Effi-
cient Prefilling. InInternational Conference on Learning Representations (ICLR).
arXiv:2409.15355 [cs.LG] https://arxiv.org/abs/2409.15355
[28] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and
Hannaneh Hajishirzi. 2023. When Not to Trust Language Models: Investigating
Effectiveness of Parametric and Non-Parametric Memories. InProceedings of
the 61st Annual Meeting of the Association for Computational Linguistics (ACL).
arXiv:2212.10511 [cs.CL] https://arxiv.org/abs/2212.10511
[29] Nikhil Prakash, Natalie Shapira, Arnab Sen Sharma, Christoph Riedl, Yonatan
Belinkov, Tamar Rott Shaham, David Bau, and Atticus Geiger. 2026. Language
Models use Lookbacks to Track Beliefs. InInternational Conference on Learning
Representations (ICLR). arXiv:2505.14685 [cs.CL] https://arxiv.org/abs/2505.14685
[30] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike
Lewis. 2023. Measuring and Narrowing the Compositionality Gap in Language
Models. InFindings of the Association for Computational Linguistics: EMNLP 2023.
arXiv:2210.03350 [cs.CL] https://arxiv.org/abs/2210.03350
[31] Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents.
https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-
agents_602894 Blog post.
[32] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and
Christopher D. Manning. 2024. RAPTOR: Recursive Abstractive Processing for
Tree-Organized Retrieval. InInternational Conference on Learning Representations
(ICLR). arXiv:2401.18059 [cs.CL] https://arxiv.org/abs/2401.18059
[33] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun
Gong, Lionel M. Ni, Heung-Yeung Shum, and Jian Guo. 2024. Think-on-
Graph: Deep and Responsible Reasoning of Large Language Model on Knowl-
edge Graph. InInternational Conference on Learning Representations (ICLR).
arXiv:2307.07697 [cs.CL] https://arxiv.org/abs/2307.07697
[34] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabhar-
wal. 2022. MuSiQue: Multihop Questions via Single-hop Question Compo-
sition.Transactions of the Association for Computational Linguistics(2022).
arXiv:2108.00573 [cs.CL] https://arxiv.org/abs/2108.00573
[35] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-
Intensive Multi-Step Questions. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (ACL). arXiv:2212.10509 [cs.CL]
https://arxiv.org/abs/2212.10509
[36] Changjian Wang, Weihong Deng, Weili Guan, Quan Lu, and Ning Jiang. 2026.
Cross-Granularity Hypergraph Retrieval-Augmented Generation for Multi-hop
Question Answering. InProceedings of the AAAI Conference on Artificial Intelli-
gence (AAAI). arXiv:2508.11247 [cs.CL] https://arxiv.org/abs/2508.11247
[37] Jingjin Wang and Jiawei Han. 2025. PropRAG: Guiding Retrieval with Beam
Search over Proposition Paths. InProceedings of the 2025 Conference on Empirical
Methods in Natural Language Processing (EMNLP). arXiv:2504.18070 [cs.CL]

Liu et al.
doi:10.18653/v1/2025.emnlp-main.317
[38] Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. 2025.
Retrieval Head Mechanistically Explains Long-Context Factuality. InInterna-
tional Conference on Learning Representations (ICLR). arXiv:2404.15574 [cs.CL]
https://arxiv.org/abs/2404.15574
[39] An Yang et al .2025. Qwen3 Technical Report. arXiv:2505.09388 [cs.CL] https:
//arxiv.org/abs/2505.09388
[40] Hongkang Yang et al .2024. Memory3: Language Modeling with Explicit Memory.
arXiv:2407.01178 [cs.CL] doi:10.4208/jml.240708
[41] Sohee Yang, Elena Gribovskaya, Nora Kassner, Mor Geva, and Sebastian Riedel.
2024. Do Large Language Models Latently Perform Multi-Hop Reasoning?. In
Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (ACL). arXiv:2402.16837 [cs.CL] https://arxiv.org/abs/2402.16837
[42] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Rus-
lan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset
for Diverse, Explainable Multi-hop Question Answering. InProceedings of the
2018 Conference on Empirical Methods in Natural Language Processing (EMNLP).
arXiv:1809.09600 [cs.CL] https://arxiv.org/abs/1809.09600
[43] Nan Zhang, Prafulla Kumar Choubey, Alexander Fabbri, Gabriel Bernadett-
Shapiro, Rui Zhang, Prasenjit Mitra, Caiming Xiong, and Chien-Sheng Wu.
2025. SiReRAG: Indexing Similar and Related Information for Multihop
Reasoning. InInternational Conference on Learning Representations (ICLR).
arXiv:2412.06206 [cs.CL] https://arxiv.org/abs/2412.06206
[44] Wuwei Zhang, Fangcong Yin, Howard Yen, Danqi Chen, and Xi Ye. 2025. Query-
Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking. In
Proceedings of the 2025 Conference on Empirical Methods in Natural Language
Processing (EMNLP). arXiv:2506.09944 [cs.CL] https://arxiv.org/abs/2506.09944
[45] Dengyong Zhou, Jiayuan Huang, and Bernhard Schölkopf. 2006. Learning with
Hypergraphs: Clustering, Classification, and Embedding. InAdvances in Neural
Information Processing Systems (NIPS). 1601–1608.
[46] Rongzhi Zhu, Xiangyu Liu, Zequn Sun, Yiwei Wang, and Wei Hu. 2025. Mitigating
Lost-in-Retrieval Problems in Retrieval Augmented Multi-Hop Question Answer-
ing. InProceedings of the 63rd Annual Meeting of the Association for Computational
Linguistics (ACL). arXiv:2502.14245 [cs.CL] https://arxiv.org/abs/2502.14245
[47] Luyao Zhuang, Shengyuan Chen, Yilin Xiao, Huachi Zhou, Yujing Zhang, Hao
Chen, Qinggang Zhang, and Xiao Huang. 2026. LinearRAG: Linear Graph Re-
trieval Augmented Generation on Large-scale Corpora. InInternational Con-
ference on Learning Representations (ICLR). arXiv:2510.10114 [cs.CL] https:
//arxiv.org/abs/2510.10114
[48] Ziyuan Zhuang, Zhiyang Zhang, Sitao Cheng, Fangkai Yang, Jia Liu, Shujian
Huang, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, and Qi Zhang. 2024.
EfficientRAG: Efficient Retriever for Multi-Hop Question Answering. InProceed-
ings of the 2024 Conference on Empirical Methods in Natural Language Processing
(EMNLP). arXiv:2408.04259 [cs.CL] https://arxiv.org/abs/2408.04259
A Implementation Details
Constants.Table 6 lists the constants ofHyperAddress. They
were chosen on the development questions of MuSiQue and are
kept for all benchmarks and readers. NV-Embed-v2 encodes each
question twice: with its retrieval instruction against passages and
content items, and with an addressing instruction against labels.
Table 6: Constants ofHyperAddress.
component constant value
hypergraph incidence weight|𝑒|−1/2
walk damping𝑑; iterations 0.85; 30
restart𝑟 0 dense prior0.05×min–max similarity
top labels; top items 10; 30
ranking fusion𝑧(log𝜋)+0.5𝑧(dense)
reading passages per round𝑚; heads 10; 8
steer restart𝑛;𝜆; temperature 5; 0.02; 1
steer transitions𝛽1
rounds𝑅; span threshold; weight 2; 0.7; 2
Rewriting.We select the entity with the highest sum of its z-
normalized pointer score and the z-normalized similarity between
the question and its best label in 𝑆. The candidate spans of the
question are its word n-grams of at most 16 words that start at a
determiner or at a capitalized word. The span most similar to theselected entity’s best label is replaced by the entity’s name if their
cosine similarity is at least 0.7. The restart of the rewritten question
is added, with weight 2, to the restarts of the earlier versions of the
question.
Generated names.In the variants of Table 7 that generate the
bridge name, Qwen3-4B reads the same passages and question
asHyperAddress, followed by an instruction to name the entity
that the question describes without naming it. It decodes up to 16
tokens, greedily and without thinking mode, as the baselines do
(Appendix A.5). The name is mapped to the candidate in 𝑆whose
name is most similar to it under NV-Embed-v2, so that it selects an
entity in every round, as the pointer scores do. This entity replaces
the one selected for rewriting. To steer, it receives the whole share
𝜆of the restart, and each of its incidences in 𝑆receives the highest
incidence score of that round.
A.1 Addressing Heads: Additional Results
Binding.To test whether the addressing heads follow the text or
the name, we build 300 probes whose passage also mentions an
entity of the same type as the target, and we edit the passage. When
the passage binds the description to a made-up name, the made-up
name becomes the top candidate for 25% of the probes. When the
made-up name replaces the other entity, which the description does
not refer to, it becomes the top candidate for 3%. The heads thus
rank first the entity that the text binds to the description, whatever
its name.
Positional rotation.Removing the rotary position encoding
from the queries and keys of the addressing heads changes the
mean reciprocal rank of the target in Table 2 by less than .01 (.728
instead of .734 on probes, .702 instead of .705 on MuSiQue). The
match is therefore computed from content rather than position.
A.2 Decomposition and Cost
Table 7 lists the variants of Figure 4 and Section 5.3 with their
Recall@5, overall and by number of hops. Table 8 lists the costs of
Figure 5, including the tokens generated to build each index and to
answer each question.
A.3 Results on All 1,000 Questions
Table 9 scores the methods that we run on all 1,000 questions of each
benchmark, development questions included, next to the numbers
that their papers publish under the same protocol. The published
numbers use NV-Embed-v2 and LLMs of 70B parameters; our runs
use Qwen3.5-27B. Our reproductions match the published numbers:
NV-Embed-v2 within 0.4 points of Recall@5, and HippoRAG 2
within 0.6 points on MuSiQue and HotpotQA and 3.0 points above
on 2Wiki.
A.4 Single-Hop Questions
Following HippoRAG 2, Table 10 adds the single-hop questions of
NQ [ 22] and PopQA [ 28], all 1,000 of each. Our run of NV-Embed-v2
is within 0.4 points of the published 75.4 and 51.0.HyperAddress
ranks first on both datasets, 2.7 points of Recall@5 above Hip-
poRAG 2 on NQ and 12.5 above PropRAG on PopQA. On PopQA,
38.1% of the questions have complete evidence in the top five,
against 3.0% with dense retrieval.

Read the Route: Attention-Steered Hypergraph Retrieval for Multi-hop RAG
Table 7: Recall@5 (%) on MuSiQue test, overall and by number
of hops (415, 258, and 127 questions). Δ: change ofHyper-
Addressfrom the walk and of each indented row from
HyperAddress. All are significant (paired bootstrap, 95%)
except the appended labels.
by hops
Method R@5 2 3 4Δ
NV-Embed-v2 69.9 78.2 69.6 43.5
HippoRAG 2 75.3 86.3 71.8 46.3
PropRAG 77.8 87.8 74.5 51.6
IRCoT (Qwen3.5-27B) 78.9 88.7 76.7 51.2
hypergraph walk, no reading 74.7 84.7 71.8 47.8
HyperAddress 83.1 92.3 80.6 57.9+8.4
steering and rewriting
rewriting only, no steering 77.3 87.5 74.7 49.6−5.7
steering only, no rewriting 81.7 91.1 79.2 56.3−1.3
restart not steered (𝜆=0) 80.8 90.1 78.0 55.9−2.3
transitions not steered (𝛽=0) 80.6 91.0 77.6 53.0−2.4
a generated bridge name instead of the pointer scores
rewriting with the name 78.1 88.3 75.3 50.4−5.0
steering and rewriting with the name 80.9 91.2 77.6 53.5−2.2
reading
last question token only 80.8 91.7 77.1 53.0−2.2
re-ranking the walk instead of steering it 80.0 89.8 78.3 51.4−3.1
embeddings instead of attention 76.5 86.1 73.9 50.2−6.6
heads selected by QRHead 77.1 88.0 73.0 50.0−6.0
heads selected by LOCOS 75.9 86.9 72.1 47.8−7.2
three rounds instead of two 83.8 92.5 81.1 60.8+0.8
index
labels not in the restart 79.9 91.0 77.3 48.8−3.2
labels replaced by entity names 80.9 91.2 78.0 53.0−2.2
labels appended to the passages 82.8 92.4 80.7 55.5−0.3
Table 8: Cost on MuSiQue. Index: generated tokens (millions)
and GPU hours for 11,656 passages, including head selection
forHyperAddress. Per question: mean over the first 100
test questions. IRCoT retrieves from the dense index (–).
index per question
Method R@5 M tokens hours tokens seconds
NV-Embed-v2 69.9 0 0.07 0 0.07
RAPTOR 59.4 1.5 3.1 0 0.47
HippoRAG 2 75.3 6.1 4.2 61 2.80
PropRAG 77.8 7.0 4.5 0 1.20
HGRAG 72.9 4.3 1.3 34 1.5
LinearRAG 73.8 0 1.2 0 0.33
IRCoT (4B) 76.4 – – 301 7.90
IRCoT (27B) 78.9 – – 482 11.8
HyperAddress 83.1 2.9 2.3 0 0.51
A.5 Baselines
All baselines run on the corpora and questions of HippoRAG 2.
LLMs are served with vLLM, decode greedily, and run without
thinking mode; the reader ofHyperAddressruns in Transform-
ers. HippoRAG 2 uses the configuration of its released script, with
Qwen3.5-27B for open information extraction and for the filter of
the seed facts. Its NV-Embed-v2 is loaded in half precision, as in the
encoder ofHyperAddress. IRCoT runs the loop and prompts re-
leased with HippoRAG over its dense NV-Embed-v2 retriever. As in
the original IRCoT, it keeps only the first sentence of each generated
thought. RAPTOR, PropRAG, and LinearRAG run with their officialTable 9: Recall@5 (%) on all 1,000 questions: our runs and
published numbers under the same protocol.
our run published
Method MuSiQue 2Wiki Hotpot MuSiQue 2Wiki Hotpot
NV-Embed-v2 69.3 76.7 94.4 69.7 76.5 94.5
RAPTOR – – – 57.8 66.2 86.9
HippoRAG 2 74.5 93.4 95.7 74.7 90.4 96.3
HGRAG – – – 74.1 93.0 95.5
PropRAG – – – 78.3 94.1 97.4
IRCoT (4B) 75.9 90.9 93.8 – – –
IRCoT (27B) 78.4 92.6 94.7 – – –
HyperAddress 82.9 97.6 98.4 – – –
Table 10: Recall@5 and All@5 (%) on all 1,000 single-hop
questions of NQ and of PopQA. Published: reported by Hip-
poRAG 2 and PropRAG with Llama-3.3-70B (–: not reported).
NQ PopQA
Method R@5 All@5 R@5 All@5
Published
RAPTOR 68.3 – 48.7 –
HippoRAG 2 78.0 – 51.7 –
PropRAG 77.9 – 56.2 –
Our runs
NV-Embed-v2 75.1 42.5 51.4 3.0
hypergraph walk, no reading 79.4 49.1 59.2 19.0
HyperAddress 80.7 50.4 68.7 38.1
code, and HGRAG, which releases no code, is reimplemented follow-
ing its paper. All four use NV-Embed-v2 and, wherever they need
an LLM, Qwen3.5-27B. ICR and QRHead run their released scoring
code with Qwen3-4B on the top twenty passages of NV-Embed-v2.
ICR uses its default calibrated scoring and places the most relevant
passage last. QRHead uses sixteen heads that its released detection
procedure selects for Qwen3-4B on its NQ data. Both rebuild the
attention of the question tokens from their query states and the
cached keys, as their default code does.
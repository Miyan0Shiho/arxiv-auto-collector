# Learning Efficient and Generalizable Graph Retriever for Knowledge-Graph Question Answering

**Authors**: Tianjun Yao, Haoxuan Li, Zhiqiang Shen, Pan Li, Tongliang Liu, Kun Zhang

**Published**: 2025-06-11 12:03:52

**PDF URL**: [http://arxiv.org/pdf/2506.09645v1](http://arxiv.org/pdf/2506.09645v1)

## Abstract
Large Language Models (LLMs) have shown strong inductive reasoning ability
across various domains, but their reliability is hindered by the outdated
knowledge and hallucinations. Retrieval-Augmented Generation mitigates these
issues by grounding LLMs with external knowledge; however, most existing RAG
pipelines rely on unstructured text, limiting interpretability and structured
reasoning. Knowledge graphs, which represent facts as relational triples, offer
a more structured and compact alternative. Recent studies have explored
integrating knowledge graphs with LLMs for knowledge graph question answering
(KGQA), with a significant proportion adopting the retrieve-then-reasoning
paradigm. In this framework, graph-based retrievers have demonstrated strong
empirical performance, yet they still face challenges in generalization
ability. In this work, we propose RAPL, a novel framework for efficient and
effective graph retrieval in KGQA. RAPL addresses these limitations through
three aspects: (1) a two-stage labeling strategy that combines heuristic
signals with parametric models to provide causally grounded supervision; (2) a
model-agnostic graph transformation approach to capture both intra- and
inter-triple interactions, thereby enhancing representational capacity; and (3)
a path-based reasoning strategy that facilitates learning from the injected
rational knowledge, and supports downstream reasoner through structured inputs.
Empirically, RAPL outperforms state-of-the-art methods by $2.66\%-20.34\%$, and
significantly reduces the performance gap between smaller and more powerful
LLM-based reasoners, as well as the gap under cross-dataset settings,
highlighting its superior retrieval capability and generalizability. Codes are
available at: https://github.com/tianyao-aka/RAPL.

## Full Text


<!-- PDF content starts -->

arXiv:2506.09645v1  [cs.CL]  11 Jun 2025Learning Efficient and Generalizable Graph Retriever
for Knowledge-Graph Question Answering
Tianjun Yao1Haoxuan Li1,2Zhiqiang Shen1Pan Li3Tongliang Liu4,1Kun Zhang1,5
1Mohamed bin Zayed University of Artificial Intelligence
2Peking university3Georgia Institute of Technology
4The University of Sydney5Carnegie Mellon University
{tianjun.yao,haoxuan.li,zhiqiang.shen}@mbzuai.ac.ae
pan.li@gatech.edu ,tongliang.liu@sydney.edu.au ,kun.zhang@mbzuai.ac.ae
Abstract
Large Language Models (LLMs) have shown strong inductive reasoning ability
across various domains, but their reliability is hindered by the outdated knowledge
and hallucinations. Retrieval-Augmented Generation mitigates these issues by
grounding LLMs with external knowledge; however, most existing RAG pipelines
rely on unstructured text, limiting interpretability and structured reasoning. Knowl-
edge graphs, which represent facts as relational triples, offer a more structured and
compact alternative. Recent studies have explored integrating knowledge graphs
with LLMs for knowledge graph question answering (KGQA), with a significant
proportion adopting the retrieve-then-reasoning paradigm. In this framework,
graph-based retrievers have demonstrated strong empirical performance, yet they
still face challenges in generalization ability. In this work, we propose RAPL, a
novel framework for efficient and effective graph retrieval in KGQA. RAPLad-
dresses these limitations through three aspects: ❶a two-stage labeling strategy that
combines heuristic signals with parametric models to provide causally grounded
supervision; ❷a model-agnostic graph transformation approach to capture both
intra- and inter-triple interactions, thereby enhancing representational capacity;
and❸a path-based reasoning strategy that facilitates learning from the injected
rational knowledge, and supports downstream reasoner through structured inputs.
Empirically, RAPLoutperforms state-of-the-art methods by 2.66%−20.34%, and
significantly reduces the performance gap between smaller and more powerful
LLM-based reasoners, as well as the gap under cross-dataset settings, highlight-
ing its superior retrieval capability and generalizability. Codes are available at:
https://github.com/tianyao-aka/RAPL .
1 Introduction
Large Language Models (LLMs) [ 4,1,55] have demonstrated remarkable capabilities in complex
reasoning tasks across various domains [ 62,9,42], marking a significant step toward bridging the
gap between human cognition and artificial general intelligence (AGI) [ 20,60,66,5]. However, the
reliability of LLMs remains a pressing concern due to outdated knowledge [ 28] and hallucination [ 23,
21]. These issues severely undermine their trustworthiness in knowledge-intensive applications.
To mitigate these deficiencies, Retrieval-Augmented Generation (RAG) [ 14,33] has been introduced
to ground LLMs with external knowledge. While effective, most existing RAG pipelines rely on
unstructured text corpora, which are often noisy, redundant, and semantically diffuse [ 51,14]. In
contrast, Knowledge Graph (KG) [ 19] organizes information as structured triples (h, r, t ), providing
a compact and semantically rich representation of real-world facts [ 6,50]. As a result, incorporating
Preprint. Under review.

KGs into RAG frameworks (i.e., KG-based RAG) has emerged as a vibrant and evolving area for
achieving faithful and interpretable reasoning.
Building upon the KG-based RAG frameworks, recent studies have proposed methods that combine
KGs with LLMs for Knowledge Graph Question Answering (KGQA) [ 58,8,43,40,44,7,34]. A
prevalent approach among these methods is the retrieve-then-reasoning paradigm, where a retriever
first extracts relevant knowledge from the KG, and subsequently an LLM-based reasoner generates
answers based on the retrieved information. The retriever can be roughly categorized into LM-based
retriever and graph-based retriever. Notably, recent studies have demonstrated that graph neural
network (GNN [31, 17, 57, 64])-based graph retrievers can achieve superior performance in KGQA
tasks, even without fine-tuning the LLM reasoner [ 43,34]. The success can be mainly attributed to the
following factors: ❶Unlike LLM-based retrievers, GNN-based retrievers perform inference directly
on the KG, inherently mitigating hallucinations by grounding retrieval in the graph. ❷GNNs are
able to leverage the relational information within KGs, enabling the retrieval of contextually relevant
triples that are crucial for accurate reasoning. Despite these success, we argue that current graph-based
retrievers still face challenges that limit their effectiveness. Specifically, ❶high-quality supervision
is essential for machine learning models to generalize well. However, existing methods often rely on
heuristic-based labeling strategies, such as identifying the shortest path between entities [ 40,44,34].
While seemingly reasonable, this approach can introduce noise and irrationalities. Specifically, (i)
multiple shortest paths may exist for a given question, not all of which are rational, and (ii)some
reasoning paths may not be the shortest ones. We provide 6 examples for each case in Appendix J
to support our claim. ❷Although GNN-based retrievers inherently mitigate hallucinations, they
may struggle to generalize to unseen questions, as they are not explicitly tailored to the unique
characteristics of KGs and KGQA task, leading to limited generalization capacity . Motivated by
these challenges, we pose the following research question:
How to develop an efficient graph-based retriever that generalizes well for KGQA tasks?
To this end, we propose RAPL, a novel framework that enhances the generalization ability of graph
retrievers with Rationalized Annotator, Path-based reasoning, and Line graph transformation. Specif-
ically,❶instead of relying solely on heuristic-based labeling approaches such as shortest-path
heuristics, we propose a two-stage labeling strategy. First, a heuristic-based method is employed
to identify a candidate set of paths that are more likely to include rational reasoning paths. Then
we obtain the causally grounded reasoning paths from this candidate set by leveraging the inductive
reasoning ability of LLMs. ❷RAPLfurther improves the generalizability of the graph retriever via
line graph transformation. This approach is model-agnostic and enriches triple-level representations
by capturing both intra- and inter-triple interactions. Furthermore, it naturally supports path-based
reasoning due to its directionality-preserving nature. ❸We further introduce a path-based learning
and inference strategy that enables the model to absorb the injected rational knowledge, thereby
enhancing its generalizability. Additionally, the path-formatted outputs benefit the downstream
reasoner by providing structured and organized inputs. In practice, our method outperforms previous
state-of-the-art approaches by 2.66%−20.34% when paired with LLM reasoners of moderate param-
eter scale. Moreover, it narrows the performance gap between smaller LLMs (e.g., Llama3.1–8B) and
more powerful models (e.g., GPT -4o), as well as the gap under cross-dataset settings, highlighting its
strong retrieval capability and generalization performance.
2 Preliminary
triple τ.A triple represents a factual statement: τ=⟨e, r, e′⟩,where e, e′∈ Edenote the subject
and object entities, respectively, and r∈ R represents the relation linking these entities.
Reasoning Path p.A reasoning path p:=e0r1− →e1r2− → ···rk− →ekconnects a source entity to a
target entity through one or more intermediate entities. Moreover, we denote zp:={r1, r2,···rk}
as the relation path of p.
Problem setup. Given a natural language question qand a knowledge graph G, our goal in this study
is to learn a function fθthat takes as inputs the question entity eq, and a subgraph Gq⊂ G, to infer an
answer entity ea∈ Gq. Following previous practice, we assume that eqare correctly identified and
linked in Gq.
2

Question 𝒒:What is the mascot of the organization with the person named Eric F. Spina?
take_rolesHas_mascot
is_ais_ais_in_org
location
For the question, the retrieved reasoning paths are:Eric F. SpinaSyracuse UniversityOttoThe Orangetake_rolesEric F. SpinaSyracuse UniversitySyracusetake_roleslocation
<Path>From the triple:Eric F. SpinaSyracuse Universitytake_rolesWe establish that Eric F. Spinaholds a position at Syracuse University.Then, from the triple: Syracuse UniversityOttoThe Orangehas_mascotWe conclude that the mascot of the organization associated with Eric F. Spina is Otto the Orange.
Answer:
(1) Line GraphTransform(2) Label Selector(3) Graph Retriever
RetrieverReasoner
TrainableFrozenhas_mascot
Figure 1: Overall framework of RAPL. The generalization ability of RAPL arises from the label
rationalizer, line graph transformation, and the path-based reasoning paradigm.
3 Related Work
We discuss the relevant literature on the retrieval-then-reasoning paradigm and knowledge graph-
based agentic RAG in detail in Appendix B.
4 Empowering Graph Retrievers with Enhanced Generalization for KGQA
In this section, we discuss how to design an efficient and generalizable graph retriever from var-
ious aspects. The overall procedure is illustrated in Figure 1. Complexity analysis is deferred to
Appendix C.
4.1 Rationalized Label Supervision
High-quality label supervision is crucial for a machine learning model to generalize. In KGQA, we
define a labeling function h(·)such that eYq=h(eq, ea,Gq), where eYqrepresents the predicted paths
or triples serving as the supervision signal.
Previous studies often assume a heuristic-based labeling function hthat designates the shortest paths
between eqandeaas the teacher signal. However, as discussed in Section 1, a shortest path may
not be a rational path, employing these noisy signals can undermine the generalization ability of the
retrieval model. We show an example to demonstrate this point.
Example. For the question: What movie with film character named Woodson did Tupac
star in? Two candidate paths are: ❶Tupacfilm.actor− − − − − − − → m.0jz0c4film.performance− − − − − − − − − − − → Gridlock’d ,
and❷Tupacmusic.recording− − − − − − − − − − → m.0jz0c4recording.release− − − − − − − − − − − → Gridlock’d . The first path accurately
captures the actor-character-film reasoning chain, thereby offering a rational supervision signal.
In contrast, the second path primarily reflects musical associations. Despite correctly linking the
question entity to the answer entity, it fails to address the actual question intent. Learning from
rational paths enhances the model’s ability to generalize to similar question types. More broadly,
injecting causally relevant knowledge into the retrieval model contributes to improved generalization
across diverse reasoning scenarios.
To inject the rational knowledge into the graph retriever, we incorporate a LLM-based annotator in
the labeling function, reformulating it as:
eYq=hLM(q, eq, ea,Gq, γ), (1)
where γdenotes the LM-based reasoner. Specifically, our approach proceeds as follows. First, we
generate a set of candidate paths Pcand, with the path length constrained to the range [dmin, dmin+2],
where dminis the shortest path distance from eqtoea. Then, given the question qand the candidate
path set Pcand, we obtain the rational paths through eYq=γ(q,Pcand). The labeling function hLM(·)
3

injects rational knowledge and provides more causally grounded supervision for training the graph
retriever, thereby enhancing its generalization ability to unseen questions.
4.2 Model-Agnostic Graph Transformation
In this section, we propose to enhance the expressivity and generalizability of the graph retriever via
line graph transformation. We first introduce the definition of a directed line graph.
Definition 1. (Directed Line Graph ) Given a directed graph G= (V,E), where each edge e=
(u, v)∈ Ehas a direction from utov, the directed line graph l(G)is a graph where:
• Each node in l(G)corresponds to a directed edge in G.
•There is a directed edge from node e1= (u, v)to node e2= (v, w)inl(G)if and only if
the target of e1matches the source of e2(i.e.,vis shared and direction is preserved).
Transforming a directed graph GqintoG′
q=l(Gq)offers several advantages, as discussed below:
❶Capturing Intra- and Inter-triple Interactions. The line graph transformation enables more ex-
pressive modeling of higher-order interactions that are inherently limited in the original knowledge
graphs when using message-passing GNNs. In the main paper, we adopt GCN [ 31] as a motivating
example1, while further discussions on other widely-used GNN architectures are deferred to Ap-
pendix D. As shown in Eqn. 2, GCN updates entity (node) representations by aggregating messages
from neighboring entities and their associated relations. Although the node embedding h(k)
iat layer
kincorporates relational information through an affine transformation applied to h(k−1)
j andrij,
this formulation does not explicitly model the semantic composition of the full triple (ei, rij, ej).
Moreover, only the entity representations are iteratively updated, while the relation embedding
remains static throughout propagation. This also constrains the model’s representational capacity,
particularly in tasks that require fine-grained reasoning over relational structures.
h(k)
i=σ
X
j∈N(i)˜Aij·W(k−1)h
h(k−1)
j∥riji
, (2)
In contrast, the transformed line graph G′
qtreats each triple (ei, ri, e′
i)as a node, facilitating the
iterative refinement of triple-level representations. As illustrated in Eqn. 3, the message passing over
G′
qcaptures both: (i)the intra-triple interactions among entities and the associated relations, and
(ii)the inter-triple dependencies by aggregating contextual information from neighboring triples
in the line graph. Here, ˜A′
ijdenotes the normalized adjacency matrix in G′
q,Nℓ(i)denotes the
neighbors of node iin the line graph, and ϕ(·)is a aggregation function over triple components (e.g.,
concatenation).
h(k)
i=σ
X
j∈Nℓ(i)˜A′
ij·ϕ 
ej,rj,e′
j
W(k−1)
agg +ϕ(ei,ri,e′
i)W(k−1)
self
. (3)
Discussions. Prior studies in various domains have highlighted the importance of relation paths
for improving model generalization. For instance, in KGQA, RoG [ 40] retrieves relation paths
during the retrieval stage, while ToG [ 53] relies on relation exploration to search on the KG.
Beyond QA, recent advances in foundation models for KGs also leverage relational structures to
enhance generalizability [ 11,15,32,12,70]. However, how to effectively utilize relational structure
within graph retrieval models remains largely underexplored. While previous studies [ 34,11,12]
enhance model expressiveness through model-centric approaches such as labeling tricks [ 35,69],
our approach instead leverages model-agnostic graph transformation to explicitly model inter-triple
interactions, thereby addressing this gap. In particular, when only relation embeddings are learned
from the triples, the model degenerates into learning relational structures. By introducing inter-
triple message passing, RAPLcaptures richer contextual triple-level dependencies that go beyond
relation-level reasoning. Combined with causally grounded path-based supervision signals, the
generalization capability of RAPLis enhanced.
1Vanilla GCN doesn’t use edge features in the update, here we adopt the most widely-used form for
incorporating edge features.
4

❷Facilitating Path-based Reasoning. The line graph perspective offers two key advantages for
path-based learning and inference. (i)The line graph transformation preserves edge directionality ,
ensuring that any reasoning path pin the original graph Gquniquely corresponds to a path pℓin
the transformed graph G′
q. This property enables path-based learning and inference over G′
q.(ii)
In the original graph, path-based reasoning typically relies on entity embeddings. However, this
approach suffers when encountering unknown entities (e.g., m.48cu1), which are often represented
as zero vectors or randomly initialized embeddings. Such representations lack semantic grounding
and significantly impair the effectiveness of path-based inference. In contrast, the triple-level
representations adopted in G′
qencodes both entity and relation semantics, thereby enriching the
representational semantics.
4.3 Path-based Reasoning
In this section, we propose a path-based learning framework for RAPL, which leverages the causally
grounded rationales injected during the previous stage to enhance generalization. Moreover, the path-
formatted inputs provide a more structured and logically coherent representation, thereby facilitating
more effective reasoning by the downstream LLM-based reasoner (see Sec. 5.4).
Training . In the line graph perspective, each node virepresents a triple (ei, ri, e′
i). Consequently,
a reasoning path for a given question qis formulated as a sequence (vq(0), vq(1), . . . , v q(k)), where
vq(i)denotes the vertex selected at step i. Formally, the path reasoning objective is defined as:
max
θ,ϕPθ,ϕ
vq(i)vq(0), . . . , v q(i−1), q
. (4)
Here, the node representation for each viinG′
qis obtained by zi=fθ(vi;G′
q), where fθ(·)is
implemented as a GNN model. Furthermore, it is crucial to determine when to terminate reasoning.
To address this, we introduce a dedicated STOP node, whose representation at step iis computed as
zstop
q(i)=gϕ(AGG 
zq(0),zq(1), . . . ,zq(i−1)
, q), (5)
where gϕ(·)is instantiated as a Multi-layer Perceptron (MLP), and AGG denotes an aggregation func-
tion that integrates the node representations from the preceding steps, implemented as a summation
in our work. The loss objective Lpath for every step ican be shown as:
Lpath=ED"
−loge⟨zq,zq(i)⟩
P
j∈N(q(i−1))e⟨zq,zj⟩#
,s.t.i >0, (6)
where zqis the representation for question q.
Selecting question triple vq(0).Equation 6 applies only for i >0. At the initial step i= 0, the
model must learn to select the correct question triple vq(0)from all available candidate nodes. Since
candidates are restricted to those involving the question entity eq, the candidate space is defined as
Vcand:={vi|vi= (ei, ri, e′
i), ei=eq}. (7)
Although one can apply a softmax loss over Vcand, in practice this set often contains hundreds or
even thousands of nodes, while the ground-truth vq(0)is typically unique or limited. This imbalance
can lead to ineffective optimization and inference.
To mitigate this issue, we introduce positive sample augmentation . Specifically, given the question
representation zqand the set of relations present in Gq, a reasoning model (e.g., GPT-4o-mini) is
employed to select the most probable relation set for the question q, denoted as R∗. The augmented
positive vertex set is then defined as:
Vpos:={vq(0)} ∪ {vi|vi=⟨ei, ri, e′
i⟩, ei=eq, ri∈ R∗}, (8)
and the negative vertex set is given by Vneg:=Vcand\ Vpos. We adopt a negative sampling
objective [ 46] for optimizing the selection of the initial triple. The question triple selection loss Lqis
formulated as:
Lq=Eq∼D"
−1
|Vpos|X
v+∈Vposlogσ 
⟨zq,zv+⟩
−1
|Vneg|X
v−∈Vneglogσ 
−⟨zq,zv−⟩#
,(9)
5

where σ(x) = 1 /(1 +e−x)is the logistic sigmoid function.
Look-ahead embeddings. AsG′
qis an directed graph, the node representation zi, i∈ V can only
incorporate information from its predecessors vq(0), . . . , v q(i−1). This can be suboptimal, especially
for earlier nodes along the reasoning path, since they cannot utilize the information from subsequent
nodes to determine the next action. To address this issue, we introduce a look-ahead message-passing
mechanism by maintaining two sets of model parameters θ:={→
θ ,←
θ}, which acts on G′
qand its
edge-reversed counterpart←
G′qrespectively.
→zi=f→
θ 
vi;G′
q
,
←zi=f←
θ 
vi;←
G′
q
,
zi=MEAN (→zi,←zi).(10)
Inference. During the inference stage, we first sample the the question triple evq(0)forKtimes with
replacement. For each of these Kcandidates, we then sample 5 reasoning paths, we then choose
Mretrieved paths with highest probability score, followed by deduplication to eliminate repeated
paths. The resulting set of unique reasoning paths is passed to the reasoning model to facilitate
knowledge-grounded question answering.
5 Experiments
Table 1: Test performance on WebQSP and CWQ. The
best results are highlighted in bold, and the second-best
inunderline . We use red, blue, and green shading to
indicate the best-performing result within each retrieval
configuration. (X, Y)denotes the average number of
retrieved triples on WebQSP and CWQ respectively.
MethodWebQSP CWQ
Macro-F1 Hit Macro-F1 HitLLMQwen-7B [65] 35.5 50.8 21.6 25.3
Llama3.1-8B [45] 34.8 55.5 22.4 28.1
GPT-4o-mini [48] 40.5 63.8 40.5 63.8
ChatGPT [47] 43.5 59.3 30.2 34.7
ChatGPT+CoT [60] 38.5 73.5 31.0 47.5KG+LLMUniKGQA [26] 72.2 – 49.0 –
KD-CoT [58] 52.5 68.6 – 55.7
ToG (GPT-4) [53] – 82.6 – 67.6
StructGPT [24] – 74.6 – –
Retrieve-Rewrite-Answer [63] – 79.3 – –
G-Retriever [18] 53.4 73.4 – –
RoG [40] 70.2 86.6 54.6 61.9
EtD [38] – 82.5 – 62.0
GNN-RAG [44] 71.3 85.7 59.4 66.8
SubgraphRAG + Llama3.1-8B [34] 70.5 86.6 47.2 56.9
SubgraphRAG + GPT-4o-mini [34] 77.4 90.1 54.1 62.0
SubgraphRAG + GPT-4o [34] 76.4 89.8 59.1 66.6
Ours + Llama3.1-8B (24.87, 28.53) 74.8 87.8 48.6 57.6
Ours + GPT-4o-mini (24.87, 28.53) 79.8 92.0 56.6 68.0
Ours + GPT-4o (24.87, 28.53) 79.4 93.0 56.7 68.0
Ours + Llama3.1-8B (31.89, 38.76) 76.2 88.3 56.1 66.7
Ours + GPT-4o-mini (31.89, 38.76) 79.2 92.2 57.2 68.9
Ours + GPT-4o (31.89, 38.76) 79.2 92.3 58.3 68.8
Ours + Llama3.1-8B (41.51, 52.45) 77.3 88.8 56.8 67.2
Ours + GPT-4o-mini (41.51, 52.45) 80.4 92.5 58.1 69.3
Ours + GPT-4o (41.51, 52.45) 80.7 93.3 58.8 69.0This section evaluates our proposed
method by answering the following re-
search questions: RQ1: How does our
method compare to state-of-the-art base-
lines in overall performance? RQ2: How
do various design choices in RAPL influ-
ence performance? RQ3: How does our
method perform on questions with differ-
ent numbers of reasoning hops? RQ4: Can
the path-structured inputs enhance the per-
formance of downstream LLM-based rea-
soning modules? RQ5: How faithful and
generalizable is our method? Additionally,
we provide efficiency analysis and demon-
strations of retrieved reasoning paths in Ap-
pendix H.
5.1 Experiment Setup
Datasets. We conduct experiments on two
widely-used and challenging benchmarks
for KGQA: WebQSP [ 67] and CWQ [ 54],
Both datasets are constructed to test multi-
hop reasoning capabilities, with questions
requiring up to four hops of inference over
a large-scale knowledge graph. The un-
derlying knowledge base for both is Free-
base [ 3]. Detailed dataset statistics are pro-
vided in Appendix G.
Baselines. We compare RAPLwith 15 state-
of-the-art baseline methods, encompassing both general LLM without external KGs, and KG-
based RAG approaches that integrate KGs with LLM for KGQA. Among them, GNN-RAG and
SubgraphRAG utilize graph-based retrievers to extract relevant knowledge from the knowledge
graph. For SubgraphRAG, we adopt the same reasoning modules as used in our framework to ensure
6

fair comparisons. Since RAPL retrieves fewer than 100 triples in the experiments, we report the
performance of SubgraphRAG using the setting with 100 retrieved triples, as provided in its original
paper [34].
Evaluation. Following previous practice, we adopt Hits and Macro-F1 to assess the effectiveness
ofRAPL.Hits measures whether the correct answer appears among the predictions, while Macro-F1
computes the average of F1 scores across all test samples, providing a balanced measure of precision
and recall.
Setup. Following prior work, we employ gte-large-en-v1.5 [37] as the pretrained text encoder to
extract text embeddings to ensure fair comparisons. For reasoning path annotation, we use GPT-4o,
and conduct ablation studies on alternative annotators in Sec. 5.3. The graph retriever adopted is a
standard GCN without structural modifications. We evaluate three configurations for the retrieval
parameters (K, M ), which result in varying sizes of retrieved triple sets: {K= 60 , M= 80},
{K= 80, M= 120}, and{K= 120 , M= 200}. For the reasoning module, we consider GPT-4o,
GPT-4o-mini, and instruction-tuned Llama3.1-8B model without fine-tuning efforts.
5.2 Main Results
From Table 1, we can make several key observations: ❶General LLM methods consistently un-
derperform compared to KG-based RAG approaches, as they rely solely on internal parametric
knowledge for reasoning. ❷Among KG-based RAG methods, RAPL achieves superior perfor-
mance across most settings, significantly outperforming competitive baselines, particularly when
paired with moderately capable LLM reasoners such as GPT-4o-mini and LLaMa3.1-8B. ❸Com-
pared to SubgraphRAG, RAPL exhibits a notably smaller performance gap between strong and
weak reasoners. For instance, on the CWQ dataset, the Macro-F1 gap between GPT-4o and
LLaMA3.1-8B is reduced from 14.78% (for SubgraphRAG) to 2.22% with RAPL, indicating that
our retriever effectively retrieves critical information in a more compact and efficient manner.
Table 2: The impact of different label annotation
methods on WebQSP and CWQ. K= 60, M= 80
is used in the experiments.
Label AnnotatorWebQSP CWQ
Macro-F1 Hit Macro-F1 Hit
GPT-4oGPT-4o-mini 79.8 92.0 56.8 68.0
Llama3.1-8B 74.8 87.8 48.6 57.7
GPT-4o-miniGPT-4o-mini 78.6 91.6 54.6 64.2
Llama3.1-8B 73.7 87.8 46.0 55.4
ShortestPathGPT-4o-mini 78.6 91.7 54.7 63.8
Llama3.1-8B 74.3 88.7 46.3 55.2❹When pairing with Llama3.1-8B and GPT-
4o-mini, RAPL outperforms SubgraphRAG by
9.61%↑and3.91%↑in terms of Macro-
F1 in CWQ dataset, with 50%↓retrieved
triples. ❺For GPT-4o, a high-capacity reasoner,
RAPLslightly underperforms SubgraphRAG and
GNN-RAG. We hypothesize that this is due to
GPT-4o’s strong denoising ability, which allows
it to benefit from longer, noisier inputs. More-
over, although shortest-path-based labeling may
introduce noise, it ensures broader coverage of
information, which may benefit powerful rea-
soners. To test this hypothesis, we perform in-
ference with an extended budget up to 100 triples trained using shortest path labels as well as rational
labels, resulting in a Macro-F1 of 59.61% on CWQ. This suggests that labeling strategies should be
adapted based on the reasoning capacity of the downstream LLM.
5.3 Ablation Study
In this section, we investigate the impact of different label annotators and different architectural
designs in the graph retriever. We use K= 60 , M= 80 during inference time to conduct the
experiments on both datasets.
Impact of label annotators. While our main results are obtained using rational paths labeled by
GPT-4o, we further evaluate two alternative strategies: GPT-4o-mini and shortest path. As shown
in Table 2, Using GPT-4o as the label annotator yields the best overall performance across both
datasets. Its advantage is particularly evident on the more challenging CWQ dataset, which requires
multi-hop reasoning. Specifically, with GPT-4o-mini as the downstream reasoning module, GPT-
4o-labeled data achieves absolute gains of 2.16% and2.07% in Macro-F1, and 3.74% and4.15%
in Hit, compared to using GPT-4o-mini and shortest-path heuristics respectively. A similar trend is
observed when Llama3.1-8B is used as the reasoner. In contrast, GPT-4o-mini as a label annotator
consistently underperforms GPT-4o and yields results comparable to those derived from shortest-
7

Table 4: Breakdown of QA performance by reasoning hops.
WebQSP CWQ
1 2 1 2 ≥3
(65.8%) (34 .2%) (28 .0%) (65 .9%) (6 .1%)
Macro-F1 Hit Macro-F1 Hit Macro-F1 Hit Macro-F1 Hit Macro-F1 Hit
G-Retriever 56.4 78.2 45.7 65.4 - - - - - -
RoG 77.1 93.0 62.5 81.5 59.8 66.3 59.7 68.6 41.5 43.3
SubgraphRAG + Llama3.1-8B 75.5 91.4 65.9 83.6 51.5 63.1 57.5 68.9 41.9 47.4
SubgraphRAG + GPT-4o-mini 80.6 92.9 74.1 88.5 57.4 67.3 63.9 72.7 51.1 54.4
Ours + Llama3.1-8B 78.0 90.5 71.3 86.1 55.6 62.8 52.1 62.8 51.2 55.6
Ours + GPT-4o-mini 83.9 95.6 75.3 88.5 60.6 69.7 59.0 70.8 53.5 57.9
Ours + Llama3.1-8B 80.1 90.8 72.9 87.8 60.4 70.2 58.6 71.8 54.0 59.6
Ours + GPT-4o-mini 82.7 95.8 76.3 89.3 61.8 70.6 60.0 71.0 56.9 61.4
Ours + Llama3.1-8B 81.1 91.4 74.0 88.1 61.4 70.3 59.9 71.5 58.8 63.2
Ours + GPT-4o-mini 83.9 95.8 77.4 90.2 62.0 72.2 60.8 71.6 59.4 64.7
path supervision. These findings indicate that a language model with limited reasoning capacity
may struggle to extract causally grounded paths, leading to suboptimal downstream performance.
Table 3: Performance of different graph retriev-
ers. Best results are in bold .
Graph RetrieverWebQSP CWQ
Macro-F1 Hit Macro-F1 Hit
1-layer GCNGPT-4o-mini 74.0 90.4 51.4 62.5
Llama3.1-8B 67.7 84.1 44.2 53.1
2-layer GCN (w/o look-ahead)GPT-4o-mini 74.4 89.1 51.9 61.0
Llama3.1-8B 69.1 84.6 43.2 51.6
2-layer GCNGPT-4o-mini 79.8 92.0 56.8 68.0
Llama3.1-8B 74.8 87.9 48.6 57.7Impact of graph retriever architectures. We an-
alyze how different GNN architectures used in the
graph retriever affect KGQA performance. In the
experiment, the label annotator is fixed to GPT-4o,
and we evaluate performance using GPT-4o-mini
and Llama3.1-8B as the reasoning models. As
shown in Table 3, a 1-layer GCN performs sim-
ilarly as a 2-layer GCN without the look-ahead
mechanism. However, when the look-ahead mes-
sage passing is incorporated into the 2-layer GCN,
KGQA performance improves significantly across
both datasets and reasoning models. This highlights the importance of the look-ahead design, which
facilitates both reasoning paths selection and question triples selection by incorporating information
from subsequent nodes along the reasoning paths.
5.4 In-Depth Analysis
Performance analysis on varying reasoning hops. We evaluate the performance of RAPL on
test samples with varying numbers of reasoning hops under the three experimental settings in-
troduced earlier. We adopt Llama3.1-8B and GPT-4o-mini as the downstream reasoners, and
restrict the evaluation to samples whose answer entities are present in the KG. From Table 4,
we can make several key observations: ❶On the WebQSP dataset, RAPLachieves strong perfor-
mance even with only 24.87 retrieved triples, particularly for samples requiring two-hop reason-
ing. When increasing the retrieval to 41.51 triples, RAPL outperforms the second-best method
in terms of Macro-F1 by 12.39% with Llama3.1-8B and 4.44% with GPT-4o-mini. ❷On the
CWQ dataset, when the number of retrieved triples exceeds 38 ( K= 80, M= 120 ),RAPLcon-
sistently surpasses previous methods across all hops when using Llama3.1-8B as the reasoner.
Notably, for queries requiring ≥3hops, RAPL achieves substantial gains over SubgraphRAG,
improving Macro-F1 by 40.28% and18.97% with Llama3.1-8B and GPT-4o-mini respectively.
Figure 2: Impact of path-formatted in-
puts on reasoning performance.❸Finally, for 2-hop samples in the CWQ dataset, we ob-
serve that when using Llama3.1-8B as the reasoner, RAPL
outperforms SubgraphRAG. However, when paired with
GPT-4o-mini, RAPLunderperforms in terms of Macro-F1,
despite retrieving the same set of triples. This suggests
that GPT-4o-mini possesses stronger denoising capabil-
ities and is better able to reorganize unstructured triples
into coherent reasoning paths. Moreover, the relatively
small difference in Hit but large gap in Macro-F1 indi-
cates that the weakly supervised labeling strategy used
by SubgraphRAG may expose the reasoner to more di-
verse knowledge. As a result, the increased number of
retrieved triples may better facilitate GPT-4o-mini than
Llama3.1-8B.
8

WebQSP CWQ0.80.91.0Score
GPT-4o GPT-4o-mini Llama3.1-8B(a) Hallucination scores on WebQSP and
CWQ ( K=80,M=120 ).(b) Generalization performance of RAPL.↔denotes train on
one dataset while test on the other. We set K= 120 ,M = 180
andK= 800 ,M = 1200 respectively during inference time.
WebQSP CWQ
Macro-F1 Hit Macro-F1 Hit
SubgraphRAG (100) 77.5 90.1 54.1 62.0
SubgraphRAG ( ↔, 100) 73.8 ↓4.8% 88.1↓2.2% 44.7↓17.4% 54.2↓12.6%
SubgraphRAG ( ↔, 500) 76.2 ↓1.7% 91.2↑1.2% 50.3↓7.0% 60.8↓1.9%
Ours (41.5, 52.4) 80.4 92.5 58.1 69.3
Ours (↔, 52.8, 57.3) 79.2 ↓1.5% 92.6↑0.1% 56.3↓3.1% 67.7↓2.3%
Ours (↔, 88.7, 107.2) 80.1 ↓0.4% 93.1↑0.6% 56.1↓3.4% 67.9↓2.0%
Figure 3: Illustration of faithfulness and generalization performance.
Can Path-Formatted Inputs Facilitate LLM Reasoning? We investigate whether path-based inputs
can facilitate the reasoning efficacy of LLMs. We conduct experiments using retrieved texts with
K= 120 andM= 180 . For each sample, we convert the retrieved path into a set of triples and
then randomly shuffle these triples. This procedure preserves the overall information content but
removes structural coherence, requiring the model to reconstruct reasoning chains from disordered
facts, posing a greater challenge for multi-hop questions. We then evaluate on the CWQ dataset,
which contains more multi-hop reasoning samples than WebQSP. As shown in Figure 2, both GPT-4o
and GPT-4o-mini exhibit significant drops in Macro-F1 when using unstructured triple inputs. This
result indicates that path-formatted inputs consistently benefit LLM-based reasoning across models
with varying reasoning capacities, justifying the path-based reasoning paradigm in RAPL.
5.5 Empirical Analysis on Faithfulness and Generalization
Faithfulness. We further evaluate whether our framework relies on retrieved knowledge or intrinsic
parametric knowledge for question answering. To this end, we design a prompt to evaluate whether
the answer was generated based on retrieved information for each sample with Hit > 0 using GPT-
4o-mini. We define the average of the resulting binary outputs as the hallucination score , where for
a single sample a score of 1 indicates reliance on retrieved knowledge, and 0 otherwise. As shown
in Figure 3a, both GPT-4o and GPT-4o-mini achieve hallucination scores ≥0.9on both datasets,
indicating that the vast majority of predictions are grounded in retrieved information. On the other
hand, we observe that all models are more prone to hallucinations on the CWQ dataset, likely due to
its greater proportion of questions requiring more complex reasoning process.
Generalizability. In this section, we evaluate the generalization capability of our framework by
training the graph retriever on one dataset and testing it on another. As shown in Table 3b, the
performance gap between in-distribution and cross-dataset settings for RAPLis consistently smaller
than that of SubgraphRAG. Notably, on the CWQ dataset, the model trained on WebQSP using RAPL
experiences only a 3.1% drop in Macro-F1, compared to a 17.4% drop observed for SubgraphRAG.
These results demonstrate that our lightweight graph retriever, enhanced through design choices
specifically tailored to the structure of knowledge graphs, attains strong generalization performance
without requiring a (far) larger-scale LLM-based retriever.
6 Conclusion
We propose RAPL, a lightweight graph retriever specifically designed for knowledge graphs and the
KGQA task. By integrating several key components such as label rationalizer, model-agnostic graph
transformation, and bidirectional message passing, RAPLenhances representational capacity and
improves generalization performance. Empirically, RAPLachieves superior generalization ability,
outperforming prior state-of-the-art methods by 2.66%−20.34% when paired with GPT-4o-mini
and Llama3.1-8B. Notably, RAPLsignificantly reduces the performance gap between smaller and
more powerful LLM-based reasoners, as well as the gap under cross-dataset settings, highlighting its
superior retrieval capability and generalizability.
9

References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv
preprint arXiv:2303.08774 , 2023.
[2]Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on freebase from
question-answer pairs. In Proceedings of the 2013 conference on empirical methods in natural language
processing , pages 1533–1544, 2013.
[3]Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. Freebase: a collaboratively
created graph database for structuring human knowledge. In Proceedings of the 2008 ACM SIGMOD
international conference on Management of data , pages 1247–1250, 2008.
[4]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners.
Advances in Neural Information Processing Systems , 33:1877–1901, 2020.
[5]Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar,
Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro,
and Yi Zhang. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint
arXiv:2303.12712 , 2023.
[6]Michel Chein and Marie-Laure Mugnier. Graph-based knowledge representation: computational founda-
tions of conceptual graphs . Springer Science & Business Media, 2008.
[7]Liyi Chen, Panrong Tong, Zhongming Jin, Ying Sun, Jieping Ye, and Hui Xiong. Plan-on-graph:
Self-correcting adaptive planning of large language model on knowledge graphs. arXiv preprint
arXiv:2410.23875 , 2024.
[8]Mohammad Dehghan, Mohammad Alomrani, Sunyam Bagga, David Alfonso-Hermelo, Khalil Bibi, Abbas
Ghaddar, Yingxue Zhang, Xiaoguang Li, Jianye Hao, Qun Liu, Jimmy Lin, Boxing Chen, Prasanna
Parthasarathi, Mahdi Biparva, and Mehdi Rezagholizadeh. EWEK-QA : Enhanced web and efficient
knowledge graph retrieval for citation-based question answering systems. In Lun-Wei Ku, Andre Martins,
and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers) , pages 14169–14187, Bangkok, Thailand, August 2024. Association
for Computational Linguistics.
[9]Jingxuan Fan, Sarah Martinson, Erik Y Wang, Kaylie Hausknecht, Jonah Brenner, Danxian Liu, Nianli
Peng, Corey Wang, and Michael P Brenner. Hardmath: A benchmark dataset for challenging problems in
applied mathematics. arXiv preprint arXiv:2410.09988 , 2024.
[10] Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric. arXiv
preprint arXiv:1903.02428 , 2019.
[11] Mikhail Galkin, Xinyu Yuan, Hesham Mostafa, Jian Tang, and Zhaocheng Zhu. Towards foundation
models for knowledge graph reasoning. arXiv preprint arXiv:2310.04562 , 2023.
[12] Jianfei Gao, Yangze Zhou, and Bruno Ribeiro. Double permutation equivariance for knowledge graph
completion. arXiv preprint arXiv:2302.01313 , 2023.
[13] Yifu Gao, Linbo Qiao, Zhigang Kan, Zhihua Wen, Yongquan He, and Dongsheng Li. Two-stage gen-
erative question answering on temporal knowledge graph using large language models. arXiv preprint
arXiv:2402.16568 , 2024.
[14] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint
arXiv:2312.10997 , 2024.
[15] Yuxia Geng, Jiaoyan Chen, Jeff Z Pan, Mingyang Chen, Song Jiang, Wen Zhang, and Huajun Chen. Rela-
tional message passing for fully inductive knowledge graph completion. In 2023 IEEE 39th international
conference on data engineering (ICDE) , pages 1221–1233. IEEE, 2023.
[16] Aric Hagberg, Pieter J Swart, and Daniel A Schult. Exploring network structure, dynamics, and function
using networkx. Technical report, Los Alamos National Laboratory (LANL), Los Alamos, NM (United
States), 2008.
[17] Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs.
Advances in neural information processing systems , 30, 2017.
10

[18] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, and
Bryan Hooi. G-retriever: Retrieval-augmented generation for textual graph understanding and question
answering. Advances in Neural Information Processing Systems , 37:132876–132907, 2024.
[19] Aidan Hogan, Eva Blomqvist, Michael Cochez, Claudia d’Amato, Gerard De Melo, Claudio Gutierrez,
Sabrina Kirrane, José Emilio Labra Gayo, Roberto Navigli, Sebastian Neumaier, et al. Knowledge graphs.
ACM Computing Surveys (Csur) , 54(4):1–37, 2021.
[20] Jie Huang and Kevin Chen-Chuan Chang. Towards reasoning in large language models: A survey. In
Findings of the Association for Computational Linguistics: ACL 2023 , pages 1049–1065, 2023.
[21] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. A survey on hallucination in large language
models: Principles, taxonomy, challenges, and open questions. arXiv preprint arXiv:2311.05232 , 2023.
[22] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing
internal covariate shift. In International conference on machine learning , pages 448–456. pmlr, 2015.
[23] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea
Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Computing
Surveys , 55(12), 2023.
[24] Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, and Ji-Rong Wen. Structgpt: A
general framework for large language model to reason over structured data. In Proceedings of the 2023
Conference on Empirical Methods in Natural Language Processing , pages 9237–9251, 2023.
[25] Jinhao Jiang, Kun Zhou, Wayne Xin Zhao, Yang Song, Chen Zhu, Hengshu Zhu, and Ji-Rong Wen.
Kg-agent: An efficient autonomous agent framework for complex reasoning over knowledge graph. arXiv
preprint arXiv:2402.11163 , 2024.
[26] Jinhao Jiang, Kun Zhou, Xin Zhao, and Ji-Rong Wen. Unikgqa: Unified retrieval and reasoning for
solving multi-hop question answering over knowledge graph. In The Eleventh International Conference on
Learning Representations , 2022.
[27] Bowen Jin, Chulin Xie, Jiawei Zhang, Kashob Kumar Roy, Yu Zhang, Zheng Li, Ruirui Li, Xianfeng Tang,
Suhang Wang, Yu Meng, et al. Graph chain-of-thought: Augmenting large language models by reasoning
on graphs. arXiv preprint arXiv:2404.07103 , 2024.
[28] Jungo Kasai, Keisuke Sakaguchi, yoichi takahashi, Ronan Le Bras, Akari Asai, Xinyan Velocity Yu,
Dragomir Radev, Noah A. Smith, Yejin Choi, and Kentaro Inui. Realtime QA: What’s the answer right
now? In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks
Track , 2023.
[29] Jiho Kim, Yeonsu Kwon, Yohan Jo, and Edward Choi. KG-GPT: A general framework for reasoning on
knowledge graphs using large language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors,
Findings of the Association for Computational Linguistics: EMNLP 2023 , pages 9410–9421, Singapore,
December 2023. Association for Computational Linguistics.
[30] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint
arXiv:1412.6980 , 2014.
[31] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks.
arXiv preprint arXiv:1609.02907 , 2016.
[32] Jaejun Lee, Chanyoung Chung, and Joyce Jiyoung Whang. Ingram: Inductive knowledge graph embedding
via relation graphs. In International Conference on Machine Learning , pages 18796–18809. PMLR, 2023.
[33] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks. Advances in neural information processing systems , 33:9459–9474, 2020.
[34] Mufei Li, Siqi Miao, and Pan Li. Simple is effective: The roles of graphs and large language models in
knowledge-graph-based retrieval-augmented generation. arXiv preprint arXiv:2410.20724 , 2024.
[35] Pan Li, Yanbang Wang, Hongwei Wang, and Jure Leskovec. Distance encoding: Design provably more
powerful neural networks for graph representation learning. Advances in Neural Information Processing
Systems , 33:4465–4478, 2020.
11

[36] Shiyang Li, Yifan Gao, Haoming Jiang, Qingyu Yin, Zheng Li, Xifeng Yan, Chao Zhang, and Bing
Yin. Graph reasoning for question answering with triplet retrieval. In Findings of the Association for
Computational Linguistics: ACL 2023 , pages 3366–3375, 2023.
[37] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards general
text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281 , 2023.
[38] Guangyi Liu, Yongqi Zhang, Yong Li, and Quanming Yao. Explore then determine: A gnn-llm synergy
framework for reasoning over knowledge graph. arXiv preprint arXiv:2406.01145 , 2024.
[39] Guangyi Liu, Yongqi Zhang, Yong Li, and Quanming Yao. Dual reasoning: A gnn-llm collaborative
framework for knowledge graph question answering, 2025.
[40] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. Reasoning on graphs: Faithful and
interpretable large language model reasoning. In International Conference on Learning Representations ,
2024.
[41] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, and Jian Guo. Think-on-graph 2.0: Deep
and interpretable large language model reasoning with knowledge graph-guided retrieval. arXiv e-prints ,
pages arXiv–2407, 2024.
[42] Benjamin S Manning, Kehang Zhu, and John J Horton. Automated social science: Language models as
scientist and subjects. Technical report, National Bureau of Economic Research, 2024.
[43] Costas Mavromatis and George Karypis. Rearev: Adaptive reasoning for question answering over
knowledge graphs. In Findings of the Association for Computational Linguistics: EMNLP 2022 , pages
2447–2458, 2022.
[44] Costas Mavromatis and George Karypis. Gnn-rag: Graph neural retrieval for large language model
reasoning. arXiv preprint arXiv:2405.20139 , 2024.
[45] Meta. Build the future of ai with meta llama 3, 2024.
[46] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations
in vector space. arXiv preprint arXiv:1301.3781 , 2013.
[47] OpenAI. Introducing chatgpt, 2022.
[48] OpenAI. Hello gpt-4o, 2024.
[49] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang,
Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai,
and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library, 2019.
[50] Ian Robinson, Jim Webber, and Emil Eifrem. Graph databases: new opportunities for connected data . "
O’Reilly Media, Inc.", 2015.
[51] Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. Retrieval augmentation reduces
hallucination in conversation. In Findings of the Association for Computational Linguistics: EMNLP 2021 ,
pages 3784–3803, 2021.
[52] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout:
a simple way to prevent neural networks from overfitting. The journal of machine learning research ,
15(1):1929–1958, 2014.
[53] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel Ni, Heung-
Yeung Shum, and Jian Guo. Think-on-graph: Deep and responsible reasoning of large language model on
knowledge graph. In The Twelfth International Conference on Learning Representations , 2024.
[54] Alon Talmor and Jonathan Berant. The web as a knowledge-base for answering complex questions. In
Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies, Volume 1 (Long Papers) , pages 641–651, 2018.
[55] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and
fine-tuned chat models. arXiv preprint arXiv:2307.09288 , 2023.
[56] Petar Veli ˇckovi ´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio.
Graph attention networks. arXiv preprint arXiv:1710.10903 , 2017.
12

[57] Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, Yoshua Bengio, et al.
Graph attention networks. stat, 1050(20):10–48550, 2017.
[58] Keheng Wang, Feiyu Duan, Sirui Wang, Peiguang Li, Yunsen Xian, Chuantao Yin, Wenge Rong, and Zhang
Xiong. Knowledge-driven cot: Exploring faithful reasoning in llms for knowledge-intensive question
answering. arXiv preprint arXiv:2308.13259 , 2023.
[59] Yu Wang, Nedim Lipka, Ryan A Rossi, Alexa Siu, Ruiyi Zhang, and Tyler Derr. Knowledge graph
prompting for multi-document question answering. In Proceedings of the AAAI Conference on Artificial
Intelligence , volume 38, pages 19206–19214, 2024.
[60] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le,
and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In Advances in
Neural Information Processing Systems , 2022.
[61] Yilin Wen, Zifeng Wang, and Jimeng Sun. Mindmap: Knowledge graph prompting sparks graph of
thoughts in large language models. arXiv preprint arXiv:2308.09729 , 2023.
[62] Likang Wu, Zhi Zheng, Zhaopeng Qiu, Hao Wang, Hongchao Gu, Tingjia Shen, Chuan Qin, Chen Zhu,
Hengshu Zhu, Qi Liu, et al. A survey on large language models for recommendation. World Wide Web ,
27(5):60, 2024.
[63] Yike Wu, Nan Hu, Sheng Bi, Guilin Qi, Jie Ren, Anhuan Xie, and Wei Song. Retrieve-rewrite-answer: A
kg-to-text enhanced llms framework for knowledge graph question answering, 2023.
[64] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks?
InInternational Conference on Learning Representations , 2019.
[65] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan
Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian
Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin,
Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang,
Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao
Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei,
Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui,
Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. arXiv preprint arXiv:2407.10671 , 2024.
[66] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan.
Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information
Processing Systems , 36, 2024.
[67] Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and Jina Suh. The value of
semantic parse labeling for knowledge base question answering. In Proceedings of the 54th Annual Meeting
of the Association for Computational Linguistics (Volume 2: Short Papers) , pages 201–206, 2016.
[68] Jing Zhang, Xiaokang Zhang, Jifan Yu, Jian Tang, Jie Tang, Cuiping Li, and Hong Chen. Subgraph retrieval
enhanced model for multi-hop knowledge base question answering. In Proceedings of the 60th Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 5773–5784,
2022.
[69] Muhan Zhang, Pan Li, Yinglong Xia, Kai Wang, and Long Jin. Labeling trick: A theory of using graph
neural networks for multi-node representation learning. Advances in Neural Information Processing
Systems , 34:9061–9073, 2021.
[70] Jincheng Zhou, Beatrice Bevilacqua, and Bruno Ribeiro. An ood multi-task perspective for link prediction
with new relation types and nodes. arXiv preprint arXiv:2307.06046 , 23, 2023.
13

Appendix
Contents
A Broad Impact 15
B Related Work 15
C Complexity Analysis 15
D More Discussions on Line Graph Transformation 16
E Algorithmic Pseudocode 16
F Theoretical Properties of the Directed Line Graph 16
G Datasets 18
H More Details on Experimental Setup and Implementations 18
I Efficiency Analysis 18
J Motivating Examples on Rational Paths 19
K Demonstrations on Retrieved Reasoning Paths from RAPL 19
L Prompt Template 19
M Limitations of RAPL 20
N Software and Hardware 21
14

A Broad Impact
This work explores the integration of KGs with LLMs for the task of KGQA, with a focus on
improving retrieval quality and generalization through graph-based retrievers. By grounding LLMs
in structured relational knowledge and employing a more interpretable and generalizable retrieval
model, our proposed framework aims to significantly mitigate hallucination and improve factual
consistency.
The potential societal benefits of this research are considerable. More faithful and robust KG-
augmented language systems could enhance the reliability of AI assistants in high-stakes domains such
as scientific research, healthcare, and legal reasoning—where factual accuracy and interpretability
are critical. In particular, reducing hallucinations in LLM outputs can support safer deployment of AI
systems in real-world applications. Moreover, our method is efficient at inference stage, and does
not rely on large-scale supervised training or fine-tuning of LLMs, making it more accessible and
sustainable, especially in resource-constrained environments. Additionally, by supporting smaller
LLMs with strong retrieval capabilities, our work helps democratize access to high-performance
question answering systems without requiring access to proprietary or computationally expensive
language models for knowledge retrieval.
B Related Work
Retrieve-then-reasoning paradigm . In KG-based RAG, a substantial number of methods adopt
theretrieve-then-reasoning paradigm [ 36,29,39,63,61,44,34,40,43], wherein a retriever first
extracts relevant triples from the knowledge graph, followed by a (LLM-based) that generates the
final answer based on the retrieved information. The retriever can be broadly categorized into
LLM-based and graph-based approaches. LLM-based retrievers benefit from large model capacity
and strong semantic understanding. However, they also suffer from hallucination issues, and high
computational cost and latency. To address this, recent work has proposed lightweight GNN-based
graph retrievers [ 43,34,68,43] that operate directly on the KG structure. These methods have
achieved superior performance on KGQA benchmarks, benefiting from the strong reasoning and
denoising capabilities of downstream LLM-based reasoners. While graph-based retrievers offer
substantial computational advantages and mitigating hallucinations, they still face generalization
challenges [ 34]. In this work, we aim to retain the efficiency of graph-based retrievers while
enhancing their expressivity and generalization ability through a model-agnostic design tailored for
characteristics of knowledge graphs and KGQA task.
KG-based agentic RAG . Another line of research leverages LLMs as agents that iteratively explore
the knowledge graph to retrieve relevant information [ 13,59,25,53,7,41,27]. In this setting,
the agent integrates both retrieval and reasoning capabilities, enabling more adaptive knowledge
access. While this approach has demonstrated effectiveness in identifying relevant triples, the iterative
exploration process incurs substantial latency, as well as computational costs due to repeated LLM
calls. In contrast, our method adopts a lightweight graph-based retriever to balance efficiency and
effectiveness for KGQA.
C Complexity Analysis
Preprocessing. Given a graph G= (V,E)with|V|nodes and |E|edges, The time complexity of this
transformation is O(|E|dmax), where dmaxis the maximum node degree in G. The space complexity
isO(|E|+|E′|), where |E′|denotes the number of edges in the resulting line graph G′, typically on
the order of O(|E|davg), with davgas the average node degree.
Model training and inference. For both training and inference, with a K-layer GCN operating on
the line graph G′
q, the time complexity is O(K|E′|F), where Fis the dimensionality of node features.
The space complexity is O(|V′|F+|E′|), where |V′|is the number of triples (i.e., nodes in the line
graph). During model training and inference, it does not involve any LLM call for the retriever.
15

D More Discussions on Line Graph Transformation
In this section, we discuss the limitations of additional message-passing GNNs when applied to the
original knowledge graph, and elaborate on how these limitations can be addressed through line graph
transformation.
GraphSage [17] . The upadte function of GraphSage is shown in Eqn. 11.
m(k)
ij=W(k)
msg
h(k−1)
j∥rij
,h(k)
i=σ
W(k)
selfh(k−1)
i + MEAN
m(k)
ij	
. (11)
As seen, only the entity states h(k)
iare upadted, and relation embeddings rijremain static, so the
model cannot refine relation semantics nor capture cross-triple interactions. After converting to the
line graph G′
q, the propagation is performed between triples, enabling explicit inter-triple reasoning.
h(k)
u=σ
W(k)
selfϕ 
eu,ru,e′
u
+W(k)
nbr1
|Nℓ(u)|X
v∈Nℓ(u)ϕ 
ev,rv,e′
v
, (12)
Graph Attention Network (GAT) [ 56]. The upadte function of GAT is shown in Eqn. 13. Similarly
rijonly modulates one-hop attention, therefore the relational embedding is not updated, only the
entity embedding is learned.
α(k)
ij= softmax j
a⊤
Wh(k−1)
i∥Wh(k−1)
j∥Werij
,
h(k)
i=σX
j∈N(i)α(k)
ijV
h(k−1)
j∥rij
. (13)
After the line-graph transform, Attention is now computed between triples, so both intra-triple
composition and inter-triple dependencies influence edge-aware attention.
αij= softmax hj 
a⊤
Wϕ(ei,ri,e′
i)∥Wϕ(ej,rj,e′
j)
,h(k)
i=σX
hj∈Nℓ(hi)αijWϕ(ej,rj,e′
j)
.
(14)
Similarly, for all message-passing GNNs that follow the general message–aggregation–update
paradigm (Eq. 15), the relational structure in the original knowledge (sub)graph cannot be fully
exploited. In contrast, the line graph transformation offers a model-agnostic solution that enables
richer modeling of both intra- and inter-triple interactions, facilitating more expressive relational
reasoning. This transformation is broadly applicable across a wide range of GNN architectures.
m(k)
ij=M(k) 
h(k−1)
i,h(k−1)
j,rij
,∀j∈N(i),
a(k)
i=A(k)
m(k)
ij:j∈N(i)	
,
h(k)
i=U(k) 
h(k−1)
i,a(k)
i
,(15)
E Algorithmic Pseudocode
The overall preprocessing and training procedure is illustrated in Algorithm 1.
F Theoretical Properties of the Directed Line Graph
LetG= (V,E)be a finite directed graph. Its directed line graph is denoted by l(G) = (Vℓ,Eℓ), where
every node x∈ Vℓcorresponds to a directed edge ex= (u, v)∈ E, and there is an edge (x, y)∈ Eℓ
iff the head of exequals the tail of ey.
Proposition 1 (Bijective mapping of directed paths) .Let
P= 
v0e1− − →v1e2− − →. . .ek− − →vk
, k≥1,
16

Algorithm 1 Overall Procedure of RAPL
Require: Training set D={(q, eq, ea,Gq)}, epochs E, learning rate η, hyper -parameters λq, λpath
Ensure: Optimized retriever parameters θ={− →θ ,← −θ}andϕ
Preprocessing
1:for all (q, eq, ea,Gq)∈ D do
2: Construct line graph G′
q←LINEGRAPH (Gq)
3: Generate candidate paths Pcand←GENPATHS (Gq, dmin, dmin+2)
4: Rationalize labels Yq←γ(q,Pcand) ▷LLM call
5: Relation targeting R∗←γ(q,RELATIONS (Gq)) ▷LLM call
6:end for
Initialization
7:Initialize bi-directional GCN encoders f− →θ, f← −θand STOP MLP gϕ
Training
8:fore= 1toEdo
9: for all minibatch B ⊂ D do
10: for all (q,G′
q,Yq,R∗)∈ B do
11: Encode nodes zi←1
2 
f− →θ(vi) +f← −θ(vi)
12: BuildVpos,VnegusingR∗
13: Lq←NEGSAMPLE LOSS(θ, ϕ;Vpos,Vneg) ▷Eq. equation 9
14: Lpath←PATHLOSS(θ, ϕ;q,Yq) ▷Eq. equation 6
15: end for
16: L ← λqLq+λpathLpath ▷ λq,λpath default to 1.0
17: Update θ, ϕvia Adam with step size η
18: end for
19:end for
be a directed path of length kinG, where ei= (vi−1, vi)∈ E. Define
Φ(P) = 
e1− →e2− →. . .− →ek
,
regarding each eias its corresponding node in l(G). Then Φis a bijection between the set of directed
paths of length kinG, and the set of directed paths of length k−1inl(G).
Proof. Well-definedness. For consecutive edges ei= (vi−1, vi)andei+1= (vi, vi+1)the shared
endpoint is viwith consistent orientation, hence (ei, ei+1)∈ Eℓ. Thus Φ(P)is a valid path of length
k−1inl(G).
Injectivity. If two original paths P̸=P′differ at position j, then Φ(P)andΦ(P′)differ at node j,
soΦ(P)̸= Φ(P′).
Surjectivity. Take any path (x1→x2→ ··· → xk)inl(G)and let eibe the edge in Erepresented by
xi. Because (xi, xi+1)∈ Eℓ, the head of eiis the tail of ei+1. Hence (e1, . . . , e k)forms a length -k
path in Gwhose image under Φis the given path.
Proposition 2 (Path -length reduction) .Letu, v∈ V be connected in Gby a shortest directed path of
length d≥1,u=v0e1− →. . .ed− →v=vd.Letxuandxvbe the nodes of Vℓcorresponding to the first
edgee1and the last edge ed, respectively. Then the distance between xuandxvinl(G)is exactly
d−1, and no shorter path exists in l(G).
Proof. By Proposition 1, the path (e1, e2, . . . , e d)is a directed path of length d−1from xutoxv
inl(G). Assume for contradiction that there is a shorter path of length ℓ < d−1between xuand
xvinEℓ. The surjectivity of Φthen yields a directed path of length ℓ+ 1< d from utovinG,
contradicting the minimality of d. Hence the distance in l(G)equals d−1.
Propositions 1 confirms that there is a one-to-one mapping of reasoning paths between l(G)andG,
therefore we can perform path-based learning and inference in the line graph. Propositions 2 implies
that a K-layer GNN model in l(G)is equivalent in the receptive field as a (K+ 1) -layer GNN model
in graph G, therefore the receptive field of a GNN model increases via line graph transformation.
17

Table 5: Dataset statistics and distribution of answer set sizes.
DatasetDataset Size Distribution of Answer Set Size
#Train #Test #Ans= 1 2 ≤#Ans≤4 5≤#Ans≤9 # Ans≥10
WebQSP 2,826 1,628 51.2% 27.4% 8.3% 12.1%
CWQ 27,639 3,531 70.6% 19.4% 6.0% 4.0%
G Datasets
WebQSP is a benchmark dataset for KGQA, derived from the original WebQuestions dataset [ 2].
It comprises 4,737 natural language questions annotated with full semantic parses in the form of
SPARQL queries executable against Freebase. The dataset emphasizes single-hop questions, typically
involving a direct relation between the question and answer entities.
CWQ dataset extends the WebQSP dataset to address more challenging multi-hop question answering
scenarios. It contains 34,689 complex questions that require reasoning over multiple facts and
relations. Each question is paired with a SPARQL query and corresponding answers, facilitating
evaluation in both semantic parsing and information retrieval contexts. The datasets statistics can be
found in Table 5.
Following previous practice, we adopt the same training and test split, with the same subgraph
construction for each question-answer pair to ensure fairness [26, 40, 34, 44].
H More Details on Experimental Setup and Implementations
Setup. For model training, we employ two 2-layer GCNs to enable bidirectional message passing.
Each GCN has a hidden dimension of 512. We use the Adam optimizer [ 30] with a learning rate
of1×10−3, and a batch size of 10. Batch normalization [ 22] is not used, as we observe gradient
instability when it is applied. The graph retriever is trained for 15 epochs on both datasets, and model
selection is performed using cross-validation based on the validation loss. A dropout rate of 0.2 [ 52]
is applied for regularization. When multiple valid paths are available, we randomly sample one as the
ground-truth supervision signal at each training step.
For evaluating KGQA performance on the test sets, we first generate answers using the downstream
reasoner, followed by answer verification using GPT-4o-mini. To prevent potential information leak-
age, we decouple the answer generation and verification processes, avoiding inclusion of ground-truth
answers in the generation prompts. For answer verification, we adopt chain-of-thought prompting [ 60]
to ensure accurate estimation of Macro-F1 and Hit metrics.
Implementation. We utilize networkx [16] for performing line graph transformations and explore all
paths between question entities (source nodes) and answer entities (target nodes), and GPT-4o-mini
is used during preprocessing for relation targeting. Our remaining implementations are based on
PyTorch[49] and PyTorch Geometric [10].
I Efficiency Analysis
We evaluate the efficiency of the proposed method and baselines based on three metrics: average
runtime, average #LLM calls, and average #retrieved triples. As shown in Table 6, agentic RAG
methods (e.g., ToG) incur significantly higher latency and computational cost due to repeated LLM
invocations. Among KG-based RAG methods, approaches employing LLM-based retrievers generally
require more time than graph-based retrievers due to the LLM inference. Although GNN-RAG and
SubgraphRAG exhibit comparable runtime and LLM calls to RAPL, our method is more effective
thanks to its design choices specifically tailored for KGQA. Furthermore, RAPLretrieves no more
than 50 triples, benefiting from the path-based inference approach, which allows RAPLto learn when
to stop reasoning, thereby avoiding the need to recall a fixed number of triples as in SubgraphRAG.
As a result, RAPLenables more efficient downstream reasoning with reduced computes, balancing
effiency and effectiveness.
18

Table 6: Efficiency analysis of different methods on WebQSP dataset.
Methods Hit Avg. Runtime (s) Avg. # LLM Calls Avg. # Triples
RoG 85.6 8.65 2 49
ToG 75.1 19.03 13.2 410
GNN-RAG 85.7 1.82 1 27
SubgraphRAG 90.1 2.63 1 100
Ours 92.0 2.16 1 32
WebQTest-923_e3a2d3d50bac69d563de83a7f72eafc0
Question:
Which country with religious organization leadership Noddfa, Treorchy borders England?
Candidate shortest paths:
England →location.location.adjoin_s →m.04dgsfb →location.adjoining_relationship.adjoins →Wales
(rational)
England →law.court_jurisdiction_area.courts →National Industrial Relations
Court→law.court.jurisdiction →Wales
England →organization.organization_scope.organizations_with_this_scope →Police Federation of
England and Wales →organization.organization.geographic_scope →Wales
England →organization.organization_scope.organizations_with_this_scope →BES
Utilities →organization.organization.geographic_scope →Wales
. . .
Explanation:
The first path: It encodes a direct geographical-adjacency relation ( location.location.adjoin_s followed by
location.adjoining_relationship.adjoins ), so it correctly captures that Wales borders England.
The second path: It links England and Wales through a shared court system, reflecting legal jurisdiction rather than physical
contiguity; therefore it is not a rational answer.
The third path: It relies on an organisation (Police Federation of England and Wales) that operates in both regions. Operational
scope signals administrative overlap, not territorial borders.
The fourth path: Like the third, it uses an organisation’s geographic scope (BES Utilities) to connect the two regions, so it conveys
no information about adjacency and is likewise non-rational.
Figure 4: Motivating example on not all shortest paths are rational paths.
J Motivating Examples on Rational Paths
In this section, we provide 6 intuitive examples of the claim for each case: (i)Multiple shortest
paths may exist for a given question, not all of which are semantically meaningful, and (ii)Some
causally-grounded paths may not be the shortest. Figure 4-9 demonstrates supports for case 1, and
Figure 10-15 demonstrates supports for case 2.
K Demonstrations on Retrieved Reasoning Paths from RAPL
In this section, we provide 10 examples to demonstrate the qualitative results of RAPL, as shown in
Figure 16-25.
L Prompt Template
We provide the prompt template in this section, for rational paths identification, relation targeting,
and hallucination detection, as shown in Figure 26-28.
19

WebQTest-415_b6ad66a3f1f515d0688c346e16d202e6
Question:
What movie with film character named Mr. Woodson did Tupac star in?
Candidate shortest paths:
Tupac Shakur →film.actor.film →m.0jz0c4 →film.performance.film →Gridlock’d (rational)
Tupac Shakur →music.featured_artist.recordings →Out The Moon →music.recording.releases →Gridlock’d
Tupac Shakur →music.featured_artist.recordings →Wanted Dead or Alive
→music.recording.releases →Gridlock’d
Tupac Shakur →music.artist.track_contributions →m.0nj8wrw →music.track_contribution.track
→Out The Moon →music.recording.releases →Gridlock’d
Tupac Shakur →film.music_contributor.film →Def Jam’s How to Be a Player →film.film.produced_by
→Russell Simmons →film.producer.films_executive_produced →Gridlock’d
Explanation:
The first path: This path follows the relation film.actor.film from Tupac Shakur to a role entity, and then
film.performance.film to the film Gridlock’d. It accurately models the actor–character–film linkage, so it is a rational answer.
The second path: It connects Tupac Shakur to the film Gridlock’d via a featured music recording. This reflects musical involvement,
not acting or character presence, so it does not address the question.
The third path: Similar to the second, it identifies Tupac as a featured artist on a song associated with the film. However, musical
contributions do not imply he played a film character.
The fourth path: This path involves nested musical metadata, eventually reaching Gridlock’d via track and recording releases. It
does not establish that Tupac portrayed a character in the movie, hence it is not rational.
The fifth path: It connects Tupac to Gridlock’d via his contribution to another film (Def Jam’s How to Be a Player), which shares a
producer with Gridlock’d. This is an indirect production-based relation, not evidence of his acting in Gridlock’d.
Figure 5: Motivating example on not all shortest paths are rational paths.
WebQTrn-3696_b874dcb19fa3a6c4e6037dc13f1f3bc4
Question:
Which state senator from Georgia took this position at the earliest date?
Candidate shortest paths:
Georgia →government.political_district.representatives →m.030qq3n
→government.government_position_held.office_holder →Saxby Chambliss (rational)
Georgia →government.political_district.elections →United States Senate election in Georgia,
2008→common.topic.image →Saxby Chambliss
Explanation:
The first path: This path directly follows government.political_district.representatives from Georgia to a position
held by an entity, and then uses government.government_position_held.office_holder to identify the person (Saxby
Chambliss) who held the position. This correctly models a government role held by someone representing the district, making it a
rational and temporally grounded path for determining who assumed the position earliest.
The second path: This path connects Georgia to a 2008 Senate election, and then to Saxby Chambliss via an image relation.
Although the entity appears in the context of the election, the path does not encode any formal position-holding information or
temporal precedence. It is therefore unrelated to identifying the office-holder or the date of assuming the position, and is non-rational.
Figure 6: Motivating example on not all shortest paths are rational paths.
M Limitations of RAPL
While RAPLdemonstrates strong performance in KGQA tasks and effectively improves generaliza-
tion via structured graph retrieval, several limitations remain. First, our current approach focuses
on enhancing retrieval through structural and relational inductive bias, but does not leverage the
complementary strengths of LLM-based retrievers. Designing hybrid retrievers that combine the
efficiency of graph-based reasoning with the flexibility and expressiveness of LLMs remains an open
challenge. Such integration could potentially yield more robust and scalable KGQA systems. Second,
20

WebQTrn-3763_c707414f103503f2530fc654a85645fe
Question:
What country close to Russia has a religious organization named Ukrainian Greek Catholic Church?
Candidate shortest paths:
Ukrainian Greek Catholic Church →religion.religious_organization.leaders →m.05tnwqd
→religion.religious_organization_leadership.jurisdiction →Ukraine (rational)
Russia →location.location.partially_contains →Seym River →geography.river.basin_countries →Ukraine
Russia →olympics.olympic_participating_country.olympics_participated_in →2010 Winter Olympics
→olympics.olympic_games.participating_countries →Ukraine
Russia →organization.organization_founder.organizations_founded →Commonwealth of Independent States
→organization.organization.founders →Ukraine
Russia →location.location.adjoin_s →m.02wj9d3 →location.adjoining_relationship.adjoins →Ukraine
Explanation:
The first path: This path connects the Ukrainian Greek Catholic Church via religion.religious_organization.leaders to
a leadership entity, then via religion.religious_organization_leadership.jurisdiction to Ukraine. It correctly
encodes the organizational jurisdiction of the church and identifies the relevant country, making it a rational answer.
The second path: This connects Russia to the Seym River and then to Ukraine via river basin membership. It reflects geographic
proximity but does not capture any religious organizational structure. Hence, it is non-rational.
The third path: This path shows that both Russia and Ukraine participated in the same Olympic games. While it may imply
contemporaneity or international context, it says nothing about religious institutions or jurisdictions.
The fourth path: This connects Russia and Ukraine through the shared founding of the Commonwealth of Independent States.
Although it reflects political cooperation, it does not reveal any information about religious affiliation or structure.
The fifth path: This path shows Russia shares a border with Ukraine. It satisfies the "close to Russia" part of the question, but lacks
any information about the Ukrainian Greek Catholic Church, making it non-rational.
Figure 7: Motivating example on not all shortest paths are rational paths.
RAPLassumes the availability of well-formed KGs and does not address errors or missing entities in
the graph itself. Our future work may address the issue of the these aspects.
N Software and Hardware
We conduct all experiments using PyTorch [ 49] (v2.1.2) and PyTorch Geometric [ 10] on Linux
servers equipped with NVIDIA A100 GPUs (80GB) and CUDA 12.1.
21

WebQTrn-3548_c352f5de0efe2369ee74ef2a99973561
Question:
What city was the birthplace of Charlton Heston and a famous pro athlete who started their career in 2007?
Candidate shortest paths:
Charlton Heston →people.person.places_lived →m.0h28vy2 →people.place_lived.location →Los Angeles
(rational)
Charlton Heston →people.deceased_person.place_of_death →Beverly Hills
→base.biblioness.bibs_location.city →Los Angeles
Charlton Heston →people.person.children →Fraser Clarke Heston →people.person.place_of_birth
→Los Angeles
. . .
Explanation:
The first path: This path uses people.person.places_lived to access a lived-location node and then follows
people.place_lived.location to reach Los Angeles. Since place-of-birth often overlaps with early-life residence, this path is
rational in the absence of explicit birth data.
The second path: This connects Charlton Heston to Los Angeles via his place of death (Beverly Hills), which is a sub-location of
Los Angeles. However, death location is unrelated to birthplace, making this path non-rational.
The third path: This path identifies Charlton Heston’s child and retrieves that child’s birthplace (Los Angeles). While it shares a
location with the question’s answer, it provides no evidence of Charlton Heston’s own birthplace and is therefore non-rational.
Figure 8: Motivating example on not all shortest paths are rational paths.
WebQTrn-1399_64fc62dc06d16e612aafb00889d4ada1
Question:
What is the country close to Russia where Mikheil Saakashvili holds a government position?
Candidate shortest paths:
Mikheil Saakashvili →government.politician.government_positions_held →m.0j6t55g
→government.government_position_held.jurisdiction_of_office →Georgia (rational)
Mikheil Saakashvili →organization.organization_founder.organizations_founded
→United National Movement →organization.organization.geographic_scope →Georgia
Russia →location.location.partially_contains →Diklosmta
→location.location.partially_containedby →Georgia
Russia →location.country.languages_spoken →Osetin Language
→language.human_language.main_country →Georgia
Explanation:
The first path: This path follows government.politician.government_positions_held to retrieve a position node, and
then uses government.government_position_held.jurisdiction_of_office to reach Georgia. This explicitly identifies
the country where Mikheil Saakashvili held a government role, satisfying both the political and geographical parts of the question. It
is thus rational.
The second path: This path captures that Saakashvili founded an organization with activity in Georgia. While it indicates political
involvement, it does not assert that he held a formal government position in the country. Hence, non-rational.
The third path: This connects Russia to Georgia via a geographic relation involving Diklosmta. It reflects proximity, but does not
involve Saakashvili or government roles—so it is irrelevant to the question.
The fourth path: This path links Russia and Georgia through a shared spoken language (Osetin). While this indicates cultural or
linguistic ties, it provides no information about Saakashvili’s political role. It is non-rational.
Figure 9: Motivating example on not all shortest paths are rational paths.
22

WebQTrn-2946_93c6dae3d218dbe112d4120f45c93298
Question:
What team with mascot named Champ did Tyson Chandler play for?
Candidate shortest paths:
Tyson Chandler →sports.pro_athlete.teams →m.0j2jj7v →sports.sports_team_roster.team
→Dallas Mavericks (rational)
Tyson Chandler →sports.pro_athlete.teams →m.0110h779 →sports.sports_team_roster.team
→Dallas Mavericks
Champ→sports.mascot.team →Dallas Mavericks (shortest path)
Explanation:
The first path: This path uses sports.pro_athlete.teams to retrieve a team membership record, and
sports.sports_team_roster.team to reach the team (Dallas Mavericks). It directly connects Tyson Chandler to the team he
played for, making it a rational path.
The second path: This is structurally identical to the first but uses a different team-roster node. It also reaches Dallas Mavericks
through the correct relation pair, so it is equally valid and rational in form—though redundant if the goal is to find ateam Chandler
played for with mascot Champ.
The third path: This connects the mascot Champ to the Dallas Mavericks. It identifies the team correctly but does not involve Tyson
Chandler. Therefore, on its own, it does not answer the question. It is necessary context but not sufficient, and thus non-rational as a
standalone path.
Figure 10: Motivating example on shortest paths may not be rational paths.
WebQTrn-934_02aae167a8fa9f7d45daab265ac650cd
Question:
Who held their governmental position from 1786 and was the British General of the Revolutionary War?
Candidate shortest paths:
Kingdom of Great Britain →military.military_combatant.military_commanders →m.04fttv1
→military.military_command.military_commander →Charles Cornwallis, 1st Marquess Cornwallis
(rational)
American Revolutionary War →base.culturalevent.event.entity_involved
→Charles Cornwallis, 1st Marquess Cornwallis (shortest)
. . .
Explanation:
The first path: This path begins with the Kingdom of Great Britain, follows
military.military_combatant.military_commanders to a command structure, and then
military.military_command.military_commander to Charles Cornwallis. It directly encodes his role as a British military
commander, making it a rational path aligned with the question.
The second path: This connects Charles Cornwallis to the American Revolutionary War via an entity_involved relation. While
it establishes that he was involved in the war, it does not specify a command role nor relate to the governmental position, so it is
non-rational.
Figure 11: Motivating example on shortest paths may not be rational paths.
23

WebQTrn-2349_e831da3802943dad506eb1e3fb611847
Question:
What are the official bird and flower of the state whose capital is Lansing?
Candidate shortest paths:
Lansing →base.biblioness.bibs_location.state →Michigan
→government.governmental_jurisdiction.official_symbols →m.04st85s
→location.location_symbol_relationship.symbol →American robin (rational)
State bird →location.offical_symbol_variety.symbols_of_this_kind →m.04st83j
→location.location_symbol_relationship.symbol →American robin (shortest)
State flower →location.offical_symbol_variety.symbols_of_this_kind →m.0hz8zmz
→location.location_symbol_relationship.symbol →Apple Blossom (shortest)
. . .
Explanation:
The first path: Begin at Lansing, proceed through base.biblioness.bibs_location.state to reach Michigan, and then use
government.governmental_jurisdiction.official_symbols to retrieve the state’s official bird and flower via
location.location_symbol_relationship.symbol . It correctly model the semantic intent of the question and are rational.
The second path: This begins from the general category “State bird” and navigates to the American robin, but it lacks a connection
to Michigan or Lansing, so it cannot determine the relevant state’s identity. Thus, it is non-rational.
The third path: Similar to the third, this links the symbolic category “State flower” to Apple Blossom, but it does not connect this
symbol to any particular state. It is non-rational on its own.
Figure 12: Motivating example on shortest paths may not be rational paths.
WebQTrn-88_54f1262a1dbcb7b82f5b8ebd614401b9
Question:
In what city and state is the university that publishes the newspaper titled Santa Clara ?
Candidate shortest paths:
Santa Clara →education.school_newspaper.school →Santa Clara University →
location.location.containedby →Santa Clara (rational)
Santa Clara →education.school_newspaper.school →Santa Clara University →
location.location.containedby →California (rational)
Santa Clara →book.newspaper.circulation_areas →California (shortest)
Santa Clara →book.newspaper.headquarters →Santa Clara →
location.mailing_address.state_province_region →California
. . .
Explanation:
The first and second paths: These follow education.school_newspaper.school to reach Santa Clara University, then use
location.location.containedby to identify its city (Santa Clara) and state (California). These directly trace the geographic
location of the university that publishes the newspaper, making both paths rational.
The third path: This connects the newspaper Santa Clara to California via a general book.newspaper.circulation_areas
relation. While it suggests distribution within California, it does not ground the newspaper in a specific institution or location, so it is
non-rational.
The fourth path: This path uses the newspaper’s headquarters to reach a city and then derives the state from a mailing address field.
It circumvents the university relation required by the question and therefore does not directly answer it; it is non-rational.
Figure 13: Motivating example on shortest paths may not be rational paths.
24

WebQTrn-2591_9f8fc8341d7c53fe16d94a6a23638ec4
Question:
What country using the Malagasy Ariary currency is China’s trading partner?
Candidate shortest paths:
China→location.statistical_region.places_exported_to →m.04bfg2f →
location.imports_and_exports.exported_to →Madagascar (rational)
Malagasy ariary →finance.currency.countries_used →Madagascar (shortest)
China→travel.travel_destination.tour_operators →Bunnik Tours →
travel.tour_operator.travel_destinations →Madagascar
. . .
Explanation:
The first path: This path uses location.statistical_region.places_exported_to followed by
location.imports_and_exports.exported_to to connect China to Madagascar through a trade relationship. It correctly
captures the fact that Madagascar is one of China’s trading partners, making it rational.
The second path: This identifies Madagascar as a country that uses the Malagasy Ariary, but it does not involve China or any trade
relationship. It is relevant as supporting context for currency, but not sufficient to answer the question on its own.
The third path: This connects China to Madagascar via a tourism relationship involving a tour operator (Bunnik Tours). While it
shows interaction between the countries, it does not concern trade and is thus non-rational in the context of the question.
Figure 14: Motivating example on shortest paths may not be rational paths.
WebQTrn-2237_723ad981dc68e3cfe82e7134c8ca8fdb
Question:
Where are some places to stay at in the city where Gavin Newsom is a government officer?
Candidate shortest paths:
Gavin Newsom →government.political_appointer.appointees →m.03k0n_d →
government.government_position_held.jurisdiction_of_office →San Francisco →
travel.travel_destination.accommodation →W San Francisco (rational)
Gavin Newsom →people.person.place_of_birth →San Francisco →
travel.travel_destination.accommodation →W San Francisco (shortest)
Gavin Newsom →people.person.place_of_birth →San Francisco →
travel.travel_destination.accommodation →Hostelling International, City Center (shortest)
. . .
Explanation:
The first path: This path links Gavin Newsom to San Francisco through a government appointee role and then uses
government.government_position_held.jurisdiction_of_office to specify the city in which he held office. From there,
it identifies accommodations such as W San Francisco via travel.travel_destination.accommodation . It directly answers
the question and is rational.
The second and third paths: These use people.person.place_of_birth to link Gavin Newsom to San Francisco and then list
accommodations in that city. While San Francisco is both his birthplace and his jurisdiction of office, these paths do not establish that
the city is where he was a government officer. Thus, they are non-rational according to the expected relation chain.
Figure 15: Motivating example on shortest paths may not be rational paths.
25

WEBQSP-WebQTest-7
Question:
Where was George Washington Carver from?
Retrieved Paths:
George Washington Carver →people.person.nationality →United States of America
George Washington Carver →people.deceased_person.place_of_death →Tuskegee
George Washington Carver →people.person.places_lived →m.03prs0h
George Washington Carver →people.person.place_of_birth →Diamond
George Washington Carver →people.person.places_lived →m.03ppx0s
George Washington Carver →people.person.education →m.04hdfv4
George Washington Carver →people.person.education →m.04hdfv4 →
education.education.institution →Iowa State University
George Washington Carver →people.person.places_lived →m.03ppx0s →
people.place_lived.location →Tuskegee
George Washington Carver →people.person.places_lived →m.03prs0h →
people.place_lived.location →Joplin
George Washington Carver →people.person.place_of_birth →Diamond →
location.statistical_region.population →m.0hlfnly
Ground-truth:
Diamond
Figure 16: Example on the retrieved reasoning paths by RAPL
WEBQSP-WebQTest-928
Question:
What colleges did Harper Lee attend?
Retrieved Paths:
Harper Lee →people.person.education →m.0lwxmyl →
education.education.institution →Monroe County High School
Harper Lee →people.person.education →m.0lwxmy1 →
education.education.institution →Huntingdon College
Harper Lee →people.person.education →m.0lwxmy9 →
education.education.institution →University of Oxford
Harper Lee →people.person.education →m.0n1l46h →
education.education.institution →University of Alabama School of Law
Harper Lee →people.person.education →m.04hx138 →
education.education.institution →University of Alabama
Ground-truth:
University of Alabama, Huntingdon College, University of Oxford, University of Alabama School of Law
Figure 17: Example on the retrieved reasoning paths by RAPL
26

WEBQSP-WebQTest-1205
Question:
Who plays Harley Quinn?
Retrieved Paths:
Harley Quinn →tv.tv_character.appeared_in_tv_program →m.02wm17r →
tv.regular_tv_appearance.actor →Hynden Walch
Harley Quinn →cvg.game_character.games →m.09dycc_ →
cvg.game_performance.voice_actor →Arleen Sorkin
Harley Quinn →film.film_character.portrayed_in_films →m.0j6pcwz →
film.performance.actor →Chrissy Kiehl
Harley Quinn →tv.tv_character.appeared_in_tv_program →m.02wm18b →
tv.regular_tv_appearance.actor →Mia Sara
Harley Quinn →tv.tv_character.appeared_in_tv_program →m.0wz39vs →
tv.regular_tv_appearance.actor →Arleen Sorkin
Ground-truth:
Mia Sara, Hynden Walch, Arleen Sorkin
Figure 18: Example on the retrieved reasoning paths by RAPL
WEBQSP-WebQTest-1599
Question:
Where did Kim Jong-il die?
Retrieved Paths:
Kim Jong-il →people.deceased_person.place_of_burial →Kumsusan Palace of the Sun
Kim Jong-il →people.deceased_person.place_of_death →Pyongyang
Kim Jong-il →people.deceased_person.place_of_death →Pyongyang →
periodicals.newspaper_circulation_area.newspapers →Rodong Sinmun
Kim Jong-il →people.deceased_person.place_of_death →Pyongyang →
location.location.contains →Munsu Water Park
Kim Jong-il →people.deceased_person.place_of_death →Pyongyang →
location.location.contains →Pyongyang University of Science and Technology
Ground-truth:
Pyongyang
Figure 19: Example on the retrieved reasoning paths by RAPL
WEBQSP-WebQTest-1993
Question:
What language do they speak in Argentina?
Retrieved Paths:
Argentina →location.country.languages_spoken →Spanish Language
Argentina →location.country.languages_spoken →Yiddish Language
Argentina →location.country.languages_spoken →Guaraní language
Argentina →location.country.languages_spoken →Quechuan languages
Argentina →location.country.languages_spoken →Italian Language
Ground-truth:
Yiddish Language, Spanish Language, Quechuan languages, Italian Language, Guaraní language
Figure 20: Example on the retrieved reasoning paths by RAPL
27

CWQ-WebQTrn-962_f0c57985929ee8b823983f6e5f104971
Question:
What actor played a kid in the film with a character named Veteran at War Rally?
Retrieved Paths:
Forrest Gump →film.film_character.portrayed_in_films →m.0jycvw →
film.performance.actor →Tom Hanks
Forrest Gump →film.film_character.portrayed_in_films →m.02xgww5 →
film.performance.actor →Michael Connor Humphreys
Forrest Gump →common.topic.notable_for →g.1258qx91g
Veteran at War Rally →common.topic.notable_for →g.12z7tmqks →
film.performance.character →Veteran at War Rally
Veteran at War Rally →film.film_character.portrayed_in_films →m.0y55311 →
film.performance.actor →Jay Ross
Veteran at War Rally →film.film_character.portrayed_in_films →m.0y55311 →
film.performance.character →Veteran at War Rally
Forrest Gump →common.topic.notable_for →g.1258qx91g →
book.book_character.appears_in_book →Forrest Gump
Ground-truth:
Michael Connor Humphreys
Figure 21: Example on the retrieved reasoning paths by RAPL
CWQ-WebQTrn-2069_9a4491f5f6a880a03bd96b8180bace4c
Question:
If I were to visit the governmental jurisdiction where Ricardo Lagos holds an office, what languages do I need to learn to speak?
Retrieved Paths:
Ricardo Lagos →government.politician.government_positions_held →m.0nbbvk0 →
government.government_position_held.office_position_or_title →President of Chile
Ricardo Lagos →people.person.nationality →Chile→
location.country.official_language →Spanish Language
Ricardo Lagos →people.person.nationality →Chile→
base.mystery.cryptid_area_of_occurrence.cryptid_s_found_here →Giglioli’s Whale
Ricardo Lagos →people.person.place_of_birth →Santiago →
location.location.contains →Torre Santa Maria
Ricardo Lagos →people.person.nationality →Chile→
location.country.second_level_divisions →Choapa Province
Ricardo Lagos →people.person.place_of_birth →Santiago →
location.location.time_zones →Chile Time Zone
Ricardo Lagos →people.person.place_of_birth →Santiago →
location.location.contains →Santiago Metropolitan Park
Ground-truth:
Spanish Language, Mapudungun Language, Aymara language, Rapa Nui Language, Puquina Language
Figure 22: Example on the retrieved reasoning paths by RAPL
28

CWQ-WebQTrn-105_64489ea2de4b116070d33a0ebcfd4866
Question:
What currency is used in the country in which Atef Sedki held office in 2013?
Retrieved Paths:
Atef Sedki →government.politician.government_positions_held →m.0g9442f →
government.government_position_held.jurisdiction_of_office →Egypt
Atef Sedki →people.person.place_of_birth →Tanta→location.location.people_born_here →Atef Ebeid
Atef Sedki →people.person.place_of_birth →Tanta→
location.statistical_region.population →g.1jmcbhfn7
Atef Sedki →people.person.place_of_birth →Tanta→
location.location.people_born_here →Ramadan Abdel Rehim Mansour
Atef Sedki →people.person.nationality →Egypt→
organization.organization_scope.organizations_with_this_scope →Reform and Development Misruna Party
Atef Sedki →people.person.nationality →Egypt→
organization.organization_scope.organizations_with_this_scope →Islamist Bloc
Atef Sedki →people.person.nationality →Egypt→
location.location.events →Crusader invasions of Egypt
Atef Sedki →government.politician.government_positions_held →m.0g9442f →
government.government_position_held.jurisdiction_of_office →Egypt→
organization.organization_scope.organizations_with_this_scope →Free Egyptians Party
Ground-truth:
Egyptian pound
Figure 23: Example on the retrieved reasoning paths by RAPL
29

CWQ-WebQTest-213_cbbd86314870b15371b43439eb40587a
Question:
What celebrities did Scarlett Johansson have romantic relationships with that ended before 2006?
Retrieved Paths:
Scarlett Johansson →base.popstra.celebrity.dated →m.065q6ym →
base.popstra.dated.participant →Ryan Reynolds
Scarlett Johansson →base.popstra.celebrity.dated →m.065q9sh →
base.popstra.dated.participant →Josh Hartnett
Scarlett Johansson →base.popstra.celebrity.dated →m.064jrnt →
base.popstra.dated.participant →Josh Hartnett
Scarlett Johansson →base.popstra.celebrity.dated →m.064tt90 →
base.popstra.dated.participant →Justin Timberlake
Scarlett Johansson →base.popstra.celebrity.breakup →m.064ttdz →
base.popstra.breakup.participant →Justin Timberlake
Scarlett Johansson →base.popstra.celebrity.dated →m.065q1sp →
base.popstra.dated.participant →Jared Leto
Scarlett Johansson →base.popstra.celebrity.dated →m.065ppwb →
base.popstra.dated.participant →Patrick Wilson
Scarlett Johansson →base.popstra.celebrity.dated →m.065pwcr →
base.popstra.dated.participant →Topher Grace
Scarlett Johansson →base.popstra.celebrity.breakup →m.064fpc6 →
base.popstra.breakup.participant →nm1157013
Scarlett Johansson →base.popstra.celebrity.dated →m.064fp5j →
base.popstra.dated.participant →nm1157013
Scarlett Johansson →people.person.spouse_s →m.0ygrd3d →
people.marriage.spouse →Ryan Reynolds
Ground-truth:
Justin Timberlake, Jared Leto
Figure 24: Example on the retrieved reasoning paths by RAPL
CWQ-WebQTrn-2569_712922724a260d96fea082856cd21d6b
Question:
What sports facility is home to both the Houston Astros and Houston Hotshots?
Retrieved Paths:
Houston Rockets →sports.sports_team.arena_stadium →Toyota Center
Houston Hotshots →sports.sports_team.arena_stadium →NRG Arena
Houston Rockets →sports.sports_team.venue →m.0wz1znd →
sports.team_venue_relationship.venue →Toyota Center
Houston Hotshots →sports.sports_team.venue →m.0x2dzn8 →
sports.team_venue_relationship.venue →Lakewood Church Central Campus
Houston Rockets →sports.sports_team.venue →m.0wz8qf2 →
sports.team_venue_relationship.venue →Lakewood Church Central Campus
Houston Rockets →sports.sports_team.arena_stadium →Lakewood Church Central Campus
Houston Rockets →sports.sports_team.location →Houston →
travel.travel_destination.tourist_attractions →Toyota Center
Ground-truth:
Lakewood Church Central Campus
Figure 25: Example on the retrieved reasoning paths by RAPL
30

Prompt template for identifying rational paths
Example
Given a question < example question >, the reasoning paths are:
<reasoning paths >
The rational paths are:
<Rational Paths >
Explanation
<Explanation >
Task
Now given question < question >, the reasoning paths are:
<Candidate Paths >
Identify all the rational paths, and list below, with explanations:
<Rational Paths >
<Explanations >
Figure 26: Prompt template for retrieving rational reasoning paths.
Prompt template for relation targeting
Example
Given a question < example question >, the question entity is:< question entity >, the candidate relations for < question
entity > are: < relations >. The possible relations for this question are:
<relations >
Task
Now given question < question >, question entity: < question entity >, relations with the entity: < relations >. List the
possible relations for this question.
<relations >
Figure 27: Prompt template for potential relation targeting.
31

Prompt template for hallucination detection
Task
Given the question < question >, the retrieved reasoning paths are:
<reasoning paths >
Now answer this question, and indicate whether you used the information provided above to answer it.
<answers >
### LLM reasoner:
I have used the provided knowledge: < Yes|No >.
<Explanations >
Figure 28: Prompt template for hallucination detection.
32
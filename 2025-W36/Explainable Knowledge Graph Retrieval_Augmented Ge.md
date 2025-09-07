# Explainable Knowledge Graph Retrieval-Augmented Generation (KG-RAG) with KG-SMILE

**Authors**: Zahra Zehtabi Sabeti Moghaddam, Zeinab Dehghani, Maneeha Rani, Koorosh Aslansefat, Bhupesh Kumar Mishra, Rameez Raja Kureshi, Dhavalkumar Thakker

**Published**: 2025-09-03 18:29:30

**PDF URL**: [http://arxiv.org/pdf/2509.03626v1](http://arxiv.org/pdf/2509.03626v1)

## Abstract
Generative AI, such as Large Language Models (LLMs), has achieved impressive
progress but still produces hallucinations and unverifiable claims, limiting
reliability in sensitive domains. Retrieval-Augmented Generation (RAG) improves
accuracy by grounding outputs in external knowledge, especially in domains like
healthcare, where precision is vital. However, RAG remains opaque and
essentially a black box, heavily dependent on data quality. We developed a
method-agnostic, perturbation-based framework that provides token and
component-level interoperability for Graph RAG using SMILE and named it as
Knowledge-Graph (KG)-SMILE. By applying controlled perturbations, computing
similarities, and training weighted linear surrogates, KG-SMILE identifies the
graph entities and relations most influential to generated outputs, thereby
making RAG more transparent. We evaluate KG-SMILE using comprehensive
attribution metrics, including fidelity, faithfulness, consistency, stability,
and accuracy. Our findings show that KG-SMILE produces stable, human-aligned
explanations, demonstrating its capacity to balance model effectiveness with
interpretability and thereby fostering greater transparency and trust in
machine learning technologies.

## Full Text


<!-- PDF content starts -->

Explainable Knowledge Graph
Retrieval-Augmented Generation (KG-RAG) with
KG-SMILE
Zahra Zehtabi Sabeti Moghaddam1*, Zeinab Dehghani1,
Maneeha Rani1, Koorosh Aslansefat1, Bhupesh Kumar Mishra1,
Rameez Raja Kureshi1, Dhavalkumar Thakker1
1School of Digital and Physical Sciences, University of Hull, Hull, HU6
7RX, United Kingdom.
*Corresponding author(s). E-mail(s): zahra.zsm89@gmail.com;
Contributing authors: Z.Dehghani@hull.ac.uk; m.rani3-2022@hull.ac.uk;
k.aslansefat@hull.ac.uk; bhupesh.mishra@hull.ac.uk; r.kureshi@hull.ac.uk;
d.thakker@hull.ac.uk;
Abstract
Generative AI, such as Large Language Models (LLMs), has achieved impressive
progress but still produces hallucinations and unverifiable claims, limiting reli-
ability in sensitive domains. Retrieval-Augmented Generation (RAG) improves
accuracy by grounding outputs in external knowledge, especially in domains like
healthcare, where precision is vital. However, RAG remains opaque and essen-
tially a black box, heavily dependent on data quality. We developed a method-
agnostic, perturbation-based framework that provides token and component-level
interoperability for Graph RAG using SMILE and named it as Knowledge-Graph
(KG)-SMILE. By applying controlled perturbations, computing similarities, and
training weighted linear surrogates, KG-SMILE identifies the graph entities
and relations most influential to generated outputs, thereby making RAG more
transparent. We evaluate KG-SMILE using comprehensive attribution metrics,
including fidelity, faithfulness, consistency, stability, and accuracy. Our findings
show that KG-SMILE produces stable, human-aligned explanations, demonstrat-
ing its capacity to balance model effectiveness with interpretability and thereby
fostering greater transparency and trust in machine learning technologies.
Keywords: Retrieval-Augmented Generation (RAG), Explainability, Knowledge Graph
(KG), Large Language Models (LLMs)
1arXiv:2509.03626v1  [cs.AI]  3 Sep 2025

1 Introduction
Recent advances in Generative AI (GenAI), especially Large Language Models (LLMs),
have transformed how organisations approach content personalisation, decision-making,
and process optimization [ 1–4]. However, despite their remarkable capabilities, these
models still face critical limitations when applied in domains that demand precision and
accountability, such as healthcare. One of the most pressing concerns is the tendency
of large language models (LLMs) to produce hallucinations , outputs that are factually
incorrect or unverifiable [ 5–7]. Alongside this, ethical risks and embedded biases remain
significant barriers to adoption in sensitive settings [5, 8].
Retrieval-Augmented Generation (RAG) has emerged as one way to reduce halluci-
nations. By retrieving external knowledge and combining it with the model’s reasoning,
RAG helps ground outputs in verifiable facts [ 1,2,9]. While this improves accuracy
compared to standard LLMs, RAG is still essentially a black box; users cannot easily
tell which retrieved pieces of information shaped the final response. Moreover, we do
not fully understand how these models treat the retrieved knowledge or how they
make their final decisions. Knowledge Graphs (KGs) represent entities and their rela-
tionships in a structured manner, ensuring consistency and contextual relevance [ 10].
When integrated with RAG, this combination, often referred to as GraphRAG, grounds
model outputs in a richer and more organised knowledge base. However, despite this
structured foundation, GraphRAG still lacks mechanisms to explain how individual
graph components contribute to the generated responses.
This study builds on the principles of Explainable Artificial Intelligence (XAI),
which emphasise interpretability and accountability in AI outputs [ 8]. To achieve this,
we extend SMILE [11], amodel-agnostic , perturbation-based framework designed to
provide fine-grained interpretability for RAG, into KG-SMILE . KG-SMILE works
by applying controlled perturbations to the inputs and retrieved knowledge, then
measuring the resulting attribution shifts using Wasserstein distance , and finally
training lightweight weighted linear surrogate models to approximate how different KG
nodes and edges influence the output. In this way, KG-SMILE identifies the entities and
relations most influential to each generated response. This enables GraphRAG not only
to produce an answer but also to explain whythat answer was generated, an essential
feature for building trust in high-stakes domains such as healthcare and law [12].
The goal of this study is to improve the transparency and accountability of RAG-
based systems. To achieve this, we evaluate the KG-SMILE framework across multiple
datasets using attribution-focused metrics, including fidelity, faithfulness, consistency,
stability, and accuracy [13–15]. These measures assess the reliability and human-
alignment of explanations rather than raw model performance. Our findings show
that GraphRAG, equipped with KG-SMILE, produces interpretable, traceable, and
trustworthy reasoning traces, effectively balancing the generative strengths of LLMs
with the need for accountability.
1.1 Research Questions
This research addresses the following questions:
2

1.RQ1: How accurately can a model-agnostic, perturbation-based approach highlight
node and edge contributions within a KG for GraphRAG explainability?
2.RQ2: Which KG components (nodes, edges, relations) have the most significant
impact on response stability and attribution accuracy?
The notation and symbols used throughout the paper are summarized in Table 1.
Symbol Type Meaning
E Set Set of entities (KG nodes).
R Set Set of relations (KG edge labels).
G Graph Knowledge graph; G′⊆Gdenotes a subgraph, G∗is the
optimal retrieved subgraph.
A Set Set of candidate answers.
q Query Natural-language question.
a, a∗Answer An answer; a∗is the optimal answer to q.
T Triple A KG triple ( e, r, e′) with e, e′∈ E,r∈ R.
z Path schema Relation path z={r1, . . . , r l}.
wz Reasoning path Instantiation of z:e0r1− − →e1r2− − → ···rl− →el.
Tq, Aq Sets Entities mentioned in qand valid answers linked in G;Tq, Aq⊆
E.
P(G) Graph Perturbed graph produced by removing a set of triples from G.
Sprom Index set Indices of perturbation samples used for evaluation.
Rorg Vector Response vector (embedding) for the original (unperturbed)
output.
RpromiVector Response vector for the i-th perturbed output.
Emb (·) Function Text/graph embedding function (produces vectors).
δ(·) Function Mapping from text to embedding space (if distinguished from
Emb ).
Ci(·,·) Scalar Cosine similarity for the i-th perturbation.
WD(·,·) Scalar Wasserstein distance between vectors/distributions.
invWD(·,·) Scalar Inverse Wasserstein similarity derived from WD.
πi Scalar Similarity-based probability/weight for the i-th perturbation.
β0, . . . , β k Scalars Regression coefficients in the surrogate linear model.
ϵi Scalar Error term in the i-th regression observation.
θ, ϕ Params Learnable parameters of retriever and generator.
Table 1 : Symbols used throughout the paper.
2 Related Works
This section reviews prior research relevant to our study. We begin with an overview of
work in explainable artificial intelligence, highlighting intrinsic and post-hoc approaches.
We then examine LIME-based explainability methods, discussing their extensions and
adaptations for different domains. Next, we cover efforts in KG and LLM explainability,
with a focus on how structured knowledge can improve transparency. We further
explore strategies for enhancing explainability through KGs and approaches aimed at
3

addressing hallucinations in LLMs. In addition, we review advances in graph-based
interpretability for model decisions and conclude with studies on KG-aided question
answering, which directly inform our proposed framework.
2.1 Explainable Artificial Intelligence
The rapid advancement in complex machine learning models has heightened the need
for explainability to ensure transparency, fairness, and reliability [ 16,17]. Methods
can be broadly categorised as intrinsic or post-hoc. Intrinsic methods, such as linear
regression or decision trees [ 18], are inherently interpretable but often sacrifice predictive
power [ 19]. Post-hoc methods, such as LIME [ 20] and SHAP [ 21], explain the decisions of
complex ”black-box” models without altering their structure [ 22]. Post-hoc approaches
can be global, providing insights into overall model behaviour [ 23], or local, clarifying
individual predictions.
LIME (Local Interpretable Model-Agnostic Explanations) and its derivatives provide
case-specific interpretations through simple surrogate models, making outputs accessible
even to non-experts [ 8,20]. SHAP (SHapley Additive exPlanations), based on Shapley
values [ 24], offers broader model coverage but is computationally demanding [ 25,26].
Other approaches, such as MAPLE [ 27], combine interpretable and complex models
but introduce additional complexity [ 28]. Despite their impact, many post-hoc methods
still face challenges of stability, fidelity, and scalability. Addressing these limitations
has led to the development of advanced frameworks such as SMILE, which aim to
provide more reliable and consistent explanations.
2.2 LIME-based Explainability Methods
Building on the foundational principles of LIME [ 20], numerous methods have been
developed to overcome its limitations and extend its applicability across various domains.
These approaches target key aspects such as local fidelity, stability, computational
efficiency, and domain-specific interpretability [ 18,22]. LS-LIME (Local Surrogate
LIME) improves fidelity by sampling around regions most relevant to a prediction, albeit
with additional computational cost [ 29]. BayLIME incorporates Bayesian reasoning
and prior knowledge to address kernel sensitivity and inconsistency, thereby enhancing
robustness [ 30]. SLIME (Stabilised-LIME) applies a hypothesis-testing framework based
on the central limit theorem to determine the required number of perturbation points for
stable explanations [ 31]. S-LIME adapts explanations for AI-driven business processes
by accounting for feature constraints and task dependencies [ 32]. OptiLIME balances
stability and fidelity through optimised sampling [ 33], while ALIME (Autoencoder-
LIME) leverages autoencoders as weighting functions to better approximate decision
boundaries [34].
US-LIME improves consistency by generating samples closer to the decision bound-
ary [35]. DLIME replaces random perturbations with hierarchical clustering and KNN
to ensure determinism [ 36]. Anchor introduces rule-based ”anchors,” subsets of fea-
tures that reliably explain model decisions in critical applications [ 37]. Domain-specific
variants have further extended LIME. Sound-LIME [ 38] focuses on music data, while
4

G-LIME (Global-LIME) [ 39] combines global Bayesian regression with ElasticNet-
based refinements for deep models. GraphLIME [ 40] explains graph-structured data,
though at a high computational cost. Bootstrap-LIME (B-LIME) [ 41] adapts LIME
for temporally dependent signals such as ECGs through heartbeat segmentation and
bootstrapping, thereby improving fidelity and stability.
Recent work has also introduced task-oriented enhancements. DSEG-LIME [ 42]
supports image segmentation by perturbing object-level regions. SLICE [ 43] and
Stratified LIME [ 44] refine sampling strategies to improve diversity and subpopulation
fidelity. SS-LIME [ 45] integrates self-supervised learning to highlight semantically
meaningful regions in the input, offering richer explanations. These advances, along with
others [ 46], highlight the growing sophistication of post-hoc explainability approaches
tailored to deep learning. Finally, SMILE [ 11] addresses inconsistency and sensitivity
using statistical measures based on Empirical Cumulative Distribution Functions
(ECDF). It has been successfully applied across domains such as instruction-based
image editing [ 47], large language models [ 48], and point cloud networks [ 49]. Although
computationally more demanding, SMILE is more resilient to adversarial manipulation,
thereby enhancing the trustworthiness of explanations.
Despite these advances, many LIME-based methods still face challenges of scalability,
efficiency, and robustness. These limitations motivate our extension of SMILE into
KG-SMILE, which leverages structured knowledge graphs to deliver more transparent
and reliable explanations.
2.3 KG and LLM Explainability
The combination of KGs and LLMs has become a powerful way to improve the
transparency and explainability of AI systems. LLMs, while impressive in their reasoning
and generative abilities, often face challenges like factual inaccuracies, hallucinations
(where the model generates incorrect or made-up information), and a lack of clarity
in how decisions are made [ 50]. On the other hand, KGs store structured, fact-based
information, which makes them ideal for helping LLMs overcome these limitations. By
integrating KGs with LLMs, we can improve how these models retrieve and present
information, making them more reliable and easier to understand [ 51]. This combination
leads to more transparent, accurate, and traceable reasoning, addressing some of the
core issues that LLMs face on their own [ 51]. This integration helps to improve the
overall reliability of LLM-generated responses [ 52]. It also contributes to grounding
the responses in structured knowledge, enhancing the system’s interpretability [ 53].
Furthermore, the use of KGs can help reduce hallucinations in LLMs, making the
outputs more trustworthy [54].
2.3.1 Enhancing Explainability Through KGs
KGs play a crucial role in making LLMs more explainable by acting as external
repositories of structured knowledge. During inference (when the model is making
decisions), KGs can be queried to provide clear reasoning paths for the model’s
responses. Unlike LLMs, which rely on their internal, often opaque memory, KGs offer
a more transparent way of retrieving information, which enhances traceability and
5

interpretability [ 50]. This means that when an LLM generates an answer, KGs can help
back up that answer with specific, factual connections and reasoning [ 51]. There are
several methods that combine KGs with LLMs to improve reliability and transparency.
One popular approach is Graph Retrieval-Augmented Generation (GraphRAG), which
blends the power of vector-based retrieval with structured graph data [ 52]. For instance,
the HybridRAG framework combines KGs with vector retrieval to better ground the
generated text in context. This approach not only boosts explainability but also makes
AI systems more trustworthy by ensuring that responses are based on structured, factual
knowledge [ 53]. With this integration, LLMs can generate responses that are easier to
interpret and more reliable, significantly reducing the risk of producing hallucinations or
unverified information [ 55]. This approach also contributes to improving the traceability
and accuracy of the reasoning process [ 56]. Furthermore, it enhances the overall
transparency of LLMs by ensuring that generated responses are grounded in solid,
factual knowledge [ 57]. Finally, the use of GraphRAG techniques improves explainability
in a wide range of AI applications [58].
2.3.2 Addressing Hallucinations in LLMs
Hallucinations in LLMs, where models generate factually incorrect or misleading content,
remain a persistent challenge, particularly in domains requiring high factual accuracy [ 5].
KG-Enhanced Inference techniques mitigate this issue by incorporating structured
KG data during both pre-training and inference. One such approach, SubgraphRAG,
retrieves subgraphs of relevant expertise instead of relying solely on raw text-based
retrieval, reducing inconsistencies in generated responses [ 51]. Another technique,
Think-on-Graph (ToG), introduces an LLM–KG co-learning approach, where the LLM
acts as a reasoning agent, iteratively traversing a KG using beam search to identify the
most promising reasoning paths. This method improves knowledge traceability and
enhances the reliability of explanations [ 59]. Furthermore, tightly coupled LLM–KG
reasoning enables models to dynamically explore entities and relationships within a
KG, ensuring that outputs remain justifiable and interpretable [59].
Recent advancements in addressing hallucinations include a causal reasoning frame-
work, where a study introduced a method called causal-DAG construction and reasoning
(CDCR-SFT). This supervised fine-tuning framework trains LLMs to explicitly build
variable-level directed acyclic graphs (DAGs) and perform reasoning over them. As a
result, this approach improved the model’s ability to reason causally and reduced hal-
lucinations by 10% on the HaluEval benchmark [ 60]. Another important development
is the Rowen method, which enhances LLMs with a selective retrieval augmentation
process explicitly designed to address hallucinated outputs [61].
2.3.3 Graph-Based Interpretability for Model Decisions
Graph-based explainability methods enhance transparency by explicitly modelling
relationships between concepts. GraphLIME, for instance, adapts the LIME framework
to graph-structured data, using nonlinear feature selection techniques to improve the
interpretability of Graph Neural Networks (GNNs) [ 40]. GraphArena, a benchmarking
framework, evaluates LLMs on structured reasoning tasks, revealing the extent to
6

which these models can process and leverage graph-based knowledge [ 62]. LLMs can
be used as graph-based explainers in two primary ways:
•LLMs-as-Enhancers: LLMs refine textual node attributes in graphs, improving
downstream tasks such as node classification and relation prediction [63].
•LLMs-as-Predictors: LLMs directly infer graph structures and relationships,
reducing dependency on traditional graph models but introducing challenges in
accuracy and reliability [63].
Additionally, the GraphXAIN method introduces a novel approach to explain
Graph Neural Networks (GNNs) by generating narratives that highlight the underlying
reasoning paths of the models. This method makes the decision-making process of
GNNs more interpretable, providing transparent and human-readable explanations [ 64].
2.3.4 KG-Aided Question Answering
A key application of KG-enhanced explainability is Knowledge Graph-Augmented
Prompting (KAPING). This method embeds KG triples within LLM prompts, signif-
icantly improving zero-shot QA performance while reducing hallucinations [ 53]. By
dynamically integrating structured knowledge into LLM-generated responses, KAP-
ING ensures factual consistency and improves interpretability. In addition, LLM–KG
co-learning frameworks enable LLMs to contribute to automated KG construction,
embedding, and completion, allowing for continuous updates and dynamic improve-
ment of kGs over time [ 50]. The convergence of KGs and LLMs offers a promising
avenue for advancing explainable AI. By integrating RAG with structured KGs, graph-
based reasoning paradigms, and interactive knowledge retrieval, AI systems can achieve
greater transparency, reliability, and interpretability. However, key challenges remain
in scalability, computational efficiency, and generalisability, requiring further research
to fully harness the potential of KG–LLM hybrid approaches in real-world applications.
Recent research in this domain includes efforts to mitigate hallucinations in data
analytics. One study introduced and evaluated four targeted strategies: Structured
Output Generation, Strict Rules Enforcement, System Prompt Enhancements, and
Semantic Layer Integration, all aimed at reducing hallucinations in LLMs within data
analytics applications [ 65]. Another important development comes from Anthropic
researchers, who discovered ”persona vectors” , specific patterns of activity within an
AI’s neural network. These persona vectors influence the AI’s behaviour and character
traits, potentially explaining why AI responses can unpredictably shift, leading to
hallucinations [ 66]. The convergence of KGs and LLMs presents an exciting opportunity
to advance explainable AI. By integrating retrieval-augmented generation (RAG) with
structured KGs, along with graph-based reasoning paradigms and interactive knowledge
retrieval, AI systems can offer greater transparency, reliability, and interpretability.
However, challenges remain in areas like scalability, computational efficiency, and
generalizability, which require further research to unlock the full potential of KG–LLM
hybrid approaches in real-world applications.
7

3 Problem Definition
Despite significant advancements in GenAI [ 3,67] and RAG [ 1,68], current systems
still encounter significant challenges in high-stakes, domain-specific contexts. In fields
such as healthcare, precision, traceability, and reliability are essential [ 69]. However,
existing RAG models often struggle to trace the origins of their generated responses,
making it difficult to verify their reasoning [ 70,71]. Moreover, the inability to quantify
the contribution of individual data components within external knowledge sources
undermines both trust and usability [72].
The integration of KGs with RAG, forming what is commonly referred to as
GraphRAG [ 73–75], offers an opportunity to enhance contextual relevance and
traceability. Nonetheless, substantial gaps remain:
1.Component-Level Impact: The influence of specific entities and relationships
within a KG on attribution (ATT) fidelity, accuracy, and stability has not been
systematically studied [76].
2.Evaluation Metrics: Reliable approaches for assessing the performance and
explainability of GraphRAG systems, particularly under dynamic conditions, are
still under development [17, 28].
3.Explainability: Although GraphRAG systems theoretically improve traceability,
few practical methods exist for quantifying and visualising this explainability [18].
This research, therefore, addresses the following needs:
•Developing methodologies to systematically evaluate the impact of individual KG
components on AI-generated responses [58].
•Establishing robust metrics, such as cosine similarity and inverse Wasserstein distance,
to quantify ATT fidelity and stability [77].
•Enhancing explainability through perturbation-based analysis and regression
modelling, enabling users to interpret better and trust the system’s outputs [ 20,27].
By addressing these challenges, this study seeks to close critical gaps in the design
and evaluation of explainable AI systems, ensuring robustness, transparency, and
reliability for domain-specific applications such as legal research [8].
4 Proposed Method
GraphRAG is a framework that leverages external structured KGs to improve the
contextual understanding of LLMs and generate more informed responses [ 1]. The
goal of GraphRAG is to retrieve the most relevant knowledge from databases, thereby
enhancing the answers of downstream tasks. The process can be defined as Eq. (1)[78]:
a∗= arg max
a∈Ap(a|q, G), (1)
where a∗is the optimal answer to query qgiven graph G, and Ais the set of
possible responses. We then jointly model the target distribution p(a|q, G) with a graph
retriever pθ(G′|q, G) and an answer generator pϕ(a|q, G′), where θandϕare learnable
8

parameters. By applying the law of total probability, p(a|q, G) can be decomposed as
Eq. (2) [79]:
p(a|q, G) =X
G′⊆Gpϕ(a|q, G′)pθ(G′|q, G)≈pϕ(a|q, G∗)pθ(G∗|q, G),(2)
where G∗is the optimal subgraph. Because the number of candidate subgraphs can
grow exponentially with graph size, efficient approximation methods are required [ 80,
81]. In practice, a graph retriever is employed to extract the optimal subgraph G∗,
after which the generator produces the answer based on the retrieved subgraph.
The first step of this project entails generating responses to queries using both
original and perturbed KGs. The similarity between these responses is then calculated
using metrics such as inverse Wasserstein distance [ 14] and cosine similarity [ 82] to
assess how removed sections affect the system’s outputs. In the next step, a simplified
regression model is trained on the perturbation vectors, similarity scores, and assigned
weights to evaluate the importance of different KG components [ 20]. The coefficients
provided by the model indicate how each part of the graph influences the generation
process, thereby revealing which sections of the graph are most critical in response
generation [ 83]. This not only provides valuable insight into the way the model operates
but also enhances transparency and trust in GraphRAG’s outputs.
Finally, the KG is visualised for more straightforward interpretation. Different nodes
and edges are assigned varying colour intensities to indicate their relative importance:
the more intense the colour, the greater the contribution of that component. A colour
bar is included as a legend to clarify the meaning of these intensities. Such visualisation
highlights the internal mechanisms of the system and enhances interpretability by
illustrating how each section of the graph impacts the model’s responses [84, 85].
Step 1 – Constructing a KG
To construct the KG for this study, diabetes-related knowledge was extracted from
the test partition of the PrimeKGQA dataset, a comprehensive biomedical KGQA
resource derived from PrimeKG, the largest precision medicine-oriented KG containing
structured triples in subject–predicate–object format [ 86]. The test dataset, loaded
using Python’s json library, was filtered to isolate triples relevant to insulin-dependent
diabetes mellitus by applying a scoring function that prioritized core terms such as
”diabetes,” ”insulin,” and ”glucose,” alongside secondary terms like ”hyperglycemia”
and ”diabetic neuropathy.”
An undirected graph was constructed using the networkx library, with nodes
representing entities (e.g., ”Neonatal insulin-dependent diabetes mellitus,” ”diabetic
peripheral angiopathy”) and edges denoting predicates (e.g., ”associated with”). The
top 10 connected components were selected to focus on significant diabetes-related
subgraphs, which were partitioned into thematic parts (e.g., Part 1 for insulin-related
processes, Part 10 for complications), with each part assigned a range of triple indices.
This structured KG, tailored for integration with a RAG system, enables precise
querying and analysis of complex biomedical relationships. The resulting KG serves
as a foundational dataset for evaluating the stability and attribution (ATT) fidelity
9

of graph-based representations, which are central to the broader objectives of this
research [87, 88].
KGs contain abundant factual knowledge in the form of triples, as shown in
Eq. (3) [88]:
G={(e, r, e′)|e, e′∈ E, r∈ R} , (3)
where EandRdenote the set of entities and relations, respectively. Relation paths,
defined as sequences of relations, are given in Eq. (4) [89]:
z={r1, r2, . . . , r l}, (4)
where ri∈ R denotes the i-th relation in the path, and ldenotes the length of the
path. Reasoning paths are instances of a relation path zin KGs (see Eq. (5)) [90]:
wz=e0r1− →e1r2− → ···rl− →el, (5)
where ei∈ Edenotes the i-th entity and ri∈ R thei-th relation in the relation
path z.
Example 1: Eq. (6) gives a relation path:
z= Supervises − →Works on , (6)
And a reasoning path instance could be Eq. (7):
wz= Person 1Supervises− − − − − − → Person 2Works on− − − − − − → GraphRAG , (7)
Which denotes that ”Person 1” supervises ”Person 2” and ”Person 2” is working
on ”GraphRAG.”
KGQA is a typical reasoning task based on KGs. Given a natural language question
qand a KG G, the task aims to design a function fto predict answers a∈Aqbased
on knowledge from G, as in Eq. (8) [91]:
a=f(q, G). (8)
Following previous works [ 79,92], we assume the entities eq∈Tqmentioned in q
and the answers a∈Aqare labeled and linked to the corresponding entities in G, i.e.,
Tq, Aq⊆ E.
Alternatively, a predefined knowledge graph can be used. At this stage, accurately
defining both nodes and their relationships is essential. Once the knowledge graph is
constructed, the prompt is interpreted based on the graph’s structure and content.
This interpretation generates an answer, which we call the ”original answer,” serving
as a baseline for later improvements, as illustrated in Figs. 1 and 2.
After receiving a question, we first prompt LLMs to generate several relation paths
that are grounded in KGs as plans [ 79,91]. These plans are then used to retrieve
reasoning paths from KGs. Finally, faithful reasoning is conducted based on the retrieved
reasoning paths, and answers are generated with interpretable explanations [85].
Step 2 – Perturbation
The perturbation process systematically alters the KG to identify influential
entities and relationships [ 20,93]. This is achieved by removing specific triplets
10

Fig. 1 : Proposed framework for retrieval of structured and unstructured data with
system-formulated queries, showing inputs from structured sources and unstructured
documents, processing through a generative AI application, and generation of context-
aware and traceable answers.
Fig. 2 : GraphRAG explainability framework using perturbation-based masks, kernel
functions, and ECDF-based distances to train a surrogate model, with node, edge, and
feature masks highlighting influential components and illustrating extracted relation-
ships and semantic relevance in healthcare
11

(entity–relation–entity) individually or in groups, and observing the resulting impact
on model performance. Perturbing the graph enables us to clearly pinpoint which
components are most critical in generating the original answer.
The perturbation function P(G) operates by randomly removing sections of the
KGG, as defined in Eq. 9 [85]:
P(G) =G− {Ti1, Ti2, . . . , T in}, (9)
where Ti1, Ti2, . . . , T inare the triplets removed during perturbation. After pertur-
bation, the system computes similarity between the perturbed and original graphs
using cosine similarity between their embedding vectors Emb organdEmb prom i, as in
Eq. 10 [11]:
Ci(Emb org, Emb prom i) = Cosine( Emb (Torg), Emb (Tprom i)),∀i∈Sprom,(10)
Where Emb (·) denotes the embedding function for the respective triplets.
In addition to cosine similarity, Wasserstein Distance (WD) is used to measure the
difference between the original and perturbed response vectors, RorgandRprom i. This
helps assess how structural perturbations alter the output. The WD is computed as
Eq. 11 [11]:
Wi(Rorg,Rprom i) =
1
nnX
j=1δ(Emb (Tj
org))−δ(Emb (Tj
prom i))p
1/p
,∀i∈Sprom,
(11)
where Emb (·) is the embedding function and δ(·) transforms textual responses into
their corresponding embeddings.
By carefully evaluating different metrics, the inverse Wasserstein distance provided
the most reliable results for comparing text-based responses generated from pruned
graphs [ 94,95]. A higher similarity score indicates that the removal of specific graph
sections has minimal impact on the output. In contrast, a lower score highlights sections
that are more influential in shaping the final response.
After experimenting with various perturbation counts (10, 20, 30, 60, and 120), it
was determined that 20 perturbations strike the best balance between accuracy and
efficiency [ 96]. Each perturbed graph version P(G) generates a new response, which
is compared to the original response using Wasserstein distance, inverse Wasserstein
distance, and cosine similarity for the text output, and cosine similarity for the graph
structure. This dual evaluation allows us to measure how perturbations affect both the
graph’s internal structure and the generated outputs [85].
The calculated distances are used to assign weights to the removed graph compo-
nents, with a kernel function applied to adjust each component’s contribution. These
weights highlight the most influential sections of the KG, identifying the parts critical
for generating accurate responses [20].
Finally, a probability function πibased on cosine similarity is applied to analyse
the influence of perturbations, as in Eq. 12:
12

πi(Emb org, Embi
prom) = exp 
−(Ci(Emb org, Embi
prom))2
σ2!
,∀i∈Sprom,(12)
Where Cidenotes cosine similarity between the original and perturbed embeddings,
andσis a scaling parameter. By combining cosine similarity for the graph structure with
inverse Wasserstein distance for text outputs, we provide a comprehensive framework
to evaluate the impact of perturbations [11, 20].
Step 3 – Regression Model for Explainability
To enhance interpretability, we apply a linear regression model that relates the
applied perturbations and their weights to the similarity scores between responses from
the original and perturbed KGs [ 17,20]. Let the perturbations be Pi1, Pi2, . . . , P ik, with
corresponding weights Wi1, Wi2, . . . , W ik. The regression model is defined as Eq. 13:
Si=β0+β1Wi1Pi1+β2Wi2Pi2+···+βkWikPik+ϵi, (13)
Where:
•Pi1, Pi2, . . . , P ikare different perturbations applied to the KG,
•Wi1, Wi2, . . . , W ikare their corresponding weights, reflecting impact on responses,
•β1, β2, . . . , β kare regression coefficients indicating the influence of each weighted
perturbation on similarity score Si,
•ϵiis the error term for the i-th observation.
By mapping weighted perturbations to similarity scores, this regression model
quantifies the importance of each KG component. The regression coefficients thus serve
as interpretable indicators of how individual nodes and relationships contribute to
generating accurate responses, enabling both visualisation and deeper understanding
of the model’s decision process.
Loss Function
The regression model minimises the following loss function to determine the best-fit
parameters [97], as shown in Eq. 14:
Loss =1
nnX
i=1(Si−(β0+β1WiPi))2, (14)
which reduces the squared differences between the actual similarity scores Siand
the predicted scores from the regression model [98].
•Siis the similarity score between the original and perturbed graph responses.
•WiPidenotes the weighted impact of the perturbation Pion the responses.
•The regression model links perturbations and their weights to similarity scores,
enabling us to determine the relative significance of different graph components in
response generation.
Based on the regression coefficients, we visualise the most impactful nodes and
relationships within the knowledge graph [ 18]. Varying colour intensities correspond to
13

the relative significance of graph components: the more intense the colour, the greater
the contribution to response generation. A colour bar is included to provide a clear
legend for interpretation.
This visualisation helps users identify which parts of the knowledge graph contribute
most to the system’s responses. It also provides transparency by making the system’s
internal decision-making process more interpretable.
5 Numerical Results
Fig. 3 : In our investigation, we adopt a suite of evaluation metrics inspired by
foundational work provided by Google [ 99], emphasising the multifaceted nature of
assessing explainable models. This work highlights the significance of metrics such as
accuracy, ATT stability, ATT fidelity and consistency as essential tools for a rigorous
evaluation of model behaviour, particularly when comparing explainable models to
traditional black-box models. The adoption of these metrics provides a structured
methodology to dissect and understand model reliability in a more holistic manner [ 99].
We now present our experimental results regarding the system’s attribution (ATT)
fidelity, consistency, accuracy, and stability when perturbations are applied to the KG.
Weighted linear regression and Bayesian Ridge regression (inspired by BayLIME [ 100])
were employed for evaluation, along with similarity measures such as cosine similar-
ity [101], Wasserstein distance, inverse Wasserstein distance (inv WD) [ 95], and a
hybrid of both.
Our analysis focuses on the similarity of perturbed KGs compared with the original
KG structure and their generated responses. Specifically:
14

•Fidelity: the closeness of perturbed answers to the original answers.
•Faithfulness: The degree to which the explanation reflects the true internal workings
of the model.
•Consistency: the uniformity of model behaviour across multiple perturbations.
•Accuracy: whether the generated answer remains correct under perturbations.
•Stability: the robustness of outputs when small changes are introduced into the KG.
These evaluation criteria, summarised in Fig. 3, are essential for understanding system
behaviour under varying conditions. In the following sections, we provide detailed
results showing how well the perturbed KGs retain both structural integrity and
response reliability.
5.1 ATT fidelity Metrics
In evaluating ATT fidelity, which measures how well the responses generated from the
perturbed KG align with the original KG responses, we used cosine similarity [ 101],
Wasserstein distance (WD), inverse Wasserstein distance (inv WD) [ 95], and a com-
bination of them. These metrics were applied using two regression models: weighted
linear regression and Bayesian Ridge regression. Our goal was to determine the best
metric and model combination to ensure that perturbations to the KG do not compro-
mise the system’s output ATT fidelity [ 18]. ATT fidelity is quantified by the coefficient
of determination R2, which evaluates how much of the variance in the original response
is explained by the perturbed response [98]. The formula for R2is given as Eq.15:
R2= 1−PNp
i=1(f(Zi)−g(Zi))2
PNp
i=1(f(Zi)−f(Zi))2(15)
Where: - f(Zi) is the original response prediction, - g(Zi) is the perturbed response
prediction, - Npis the number of perturbations applied, - f(Zi)is the mean of the
original predictions.
In addition to R2, we used Mean L1 and L2 losses to measure the deviation between
the original and perturbed responses [ 49,97]. The Mean L1 loss calculates the average
absolute difference between the predictions in Eq.16:
L1=1
NpNpX
i=1|f(Zi)−g(Zi)| (16)
The Mean L1 loss ( L1) measures the average absolute deviation between the original
responses and the perturbed responses, providing a robust metric against outliers.
Similarly, the Mean L2 loss ( L2) evaluates the squared differences between
predictions, which penalizes larger deviations more heavily, as shown in Eq. 17:
L2=1
NpNpX
i=1(f(Zi)−g(Zi))2(17)
Weighted versions of these metrics, Lw
1andLw
2, incorporate kernel-based weights
wto emphasize the relative importance of specific perturbations. They are defined as:
15

Lw
1=1
NpNpX
i=1(|f(Zi)−g(Zi)| ·w) (18)
Lw
2=1
NpNpX
i=1 
(f(Zi)−g(Zi))2·w
(19)
The weighted coefficient of determination, R2
w, extends the classical R2to include
these weights and is defined as Eq. 20:
R2
w= 1−PNp
i=1(f(Zi)−g(Zi))2
PNp
i=1(f(Zi)−fw(Zi))2(20)
Since R2
wcan be biased upward in small samples, the weighted adjusted coefficient
of determination ˆR2
wcorrects for both sample size and the number of predictors, as in
Eq. 21:
ˆR2
w= 1−(1−R2
w)Np−1
Np−Ns−1
(21)
Finally, to capture the overall discrepancy in the mean values of predicted scores,
we compute the mean loss, Lm, as in Eq. 22:
Lm=PNp
i=1f(Zi)
Np−PNp
i=1g(Zi)
Np(22)
This metric quantifies the absolute difference between the average original and
perturbed predictions, offering an additional perspective on model stability. The results
are summarised in Table 2.
To thoroughly evaluate the ATT fidelity of responses generated from perturbed
KGs, we conducted a comprehensive analysis utilising a variety of metrics. These
included cosine similarity [ 101], Wasserstein distance (WD), inverse Wasserstein dis-
tance (inv WD) [ 95], and combinations. thereof, all assessed using Weighted Linear
Regression and Bayesian Ridge Regression [97, 100], as shown in Table 2.
Although some linear models, such as cosine, achieved the absolute lowest weighted
L2losses, we deliberately selected the Linear Regression model using inverse Wasserstein
Distance (inv WD) because the trade-off in numerical performance is negligible and the
benefits for explainability are substantial. As shown in Table 2, Linear cosine attains
a weighted L2of approximately 6 .65×10−33, while Linear inv WD has a weighted
L2of approximately 1 .57×10−31. Both are vanishingly small, and all linear models
have weighted R2values essentially equal to 1, indicating a near-perfect fit. Inv WD’s
interpretive advantages outweigh the marginal gap in raw loss. Unlike cosine, which
only measures angular similarity between embeddings, inverse Wasserstein distance
captures contrastive differences in the underlying distributions. This makes it more
sensitive to meaningful distributional shifts and asymmetries in kG structures. Such
sensitivity is critical when the objective is attribution fidelity : faithfully identifying
which perturbed nodes or relations drive output changes.
16

Metric CosineInv.
WDInv. WD
+ CosineWDWD
+ Cosine
BayLIME
Mean Loss (LM) 1.52E−062.71E−072.30E−072.71E−071.53E−06
Mean L1 Loss 1.17E−052.10E−061.78E−062.10E−061.18E−05
Mean L2 Loss 3.79E−101.21E−118.70E−121.21E−113.83E−10
Weighted L1 Loss 2.20E−063.93E−073.34E−073.93E−072.21E−06
Weighted L2 Loss 4.24E−111.35E−129.72E−131.35E−124.27E−11
R2ω 1.00E+001.00E+001.00E+001.00E+000.99999998
ˆR2ω 1.00E+001.00E+001.00E+001.00E+000.99999996
Linear Regression
Mean Loss (LM) 2.22E−167.77E−166.66E−165.55E−162.22E−16
Mean L1 Loss 2.05E−169.49E−169.05E−167.22E−162.16E−16
Mean L2 Loss 6.10E−321.18E−301.08E−307.57E−316.35E−32
Weighted L1 Loss 2.99E−171.88E−162.26E−161.75E−163.99E−17
Weighted L2 Loss 6.65E−331.57E−312.14E−311.63E−311.02E−32
R2ω 1.00E+001.00E+001.00E+001.00E+001.00E+00
ˆR2ω 1.00E+001.00E+001.00E+001.00E+001.00E+00
Table 2 : Fidelity comparison between perturbed and original KG responses using
BayLIME and Linear Regression across similarity metrics.
KGs are heterogeneous and often exhibit skewed distributions, with some dense
clusters and many sparse, rare entities. Metrics based purely on similarity can under-
represent rare but influential nodes. In contrast, inv WD is derived from the Wasserstein
family of distances. We used the inverse Wasserstein Distance in our weighted linear
regression model because the standard Wasserstein Distance tends to identify nodes
with the opposite influence on response generation. Using inv WD, we aimed to
highlight nodes with a direct impact on output. and is robust to differences in scale
and distributional shape. This makes explanations derived from Linear inv WD more
robust across both common and rare cases, which is particularly valuable in domains
such as healthcare, where long-tail examples often have outsized importance.
Moreover, by combining inv WD with a linear model, we retain complete model
transparency: coefficients directly indicate the contribution of each perturbation to ATT
fidelity changes. This aligns with the central aim of explainable AI research, moving
beyond numeric performance to produce explanations that are trustworthy, actionable,
and generalizable. Finally, inv WD helps avoid overfitting to mere embedding similarity
patterns. Similarity measures like cosine perform exceptionally well on in-sample data,
but may generalise poorly to novel or rare perturbations. Because inv WD explicitly
models opposing or divergent distributional changes, it provides richer attribution
signals that remain valid as the KG evolves.
In summary, our choice of Linear inv WD reflects a deliberate trade-off: we accept
a negligible increase in loss in exchange for markedly improved explanatory power,
robustness across heterogeneous data, and theoretical alignment with attribution
17

fidelity objectives. This makes Linear inv WD a more appropriate and principled choice
for our explainable KG reasoning experiments than simply selecting the numerically
top-ranked alternative. [98].
5.2 ATT Accuracy
Across the ten questions, the evaluation results offer insights into the explainability of
GraphRAG by examining how effectively the model highlights relevant parts of the
KG in response to user queries. The ground truth indicates the critical nodes in the
graph that should be highlighted (marked as “1”), while the model predictions under
different temperature settings show which nodes are actually emphasized. The analysis
reveals that temperature settings significantly affect the model’s ability to retrieve
and highlight pertinent information from the KG, influencing both explainability and
interpretability [ 102]. On average, the model achieves an AUC of 0.878 forT=0 and
0.744 forT=1 across all questions (see Appendix Table 7 for the detailed per-question
results).
At a temperature of 0, the retrieval process is deterministic, which means that
the model consistently selects the most relevant nodes based on the query. In most
instances, the predicted highlights align perfectly with the ground truth (AUC = 1),
indicating that the retrieval mechanism successfully identifies and selects the most
critical components of the knowledge graph. For example, in queries such as:
•What is insulin-like growth factor receptor binding associated with?
•What is pancreatic serous cystadenocarcinoma associated with?
•What is glucagon receptor activity associated with and how it interacts through ppi
with low-affinity glucose:proton symporter activity?
•What is glucose binding?
the model accurately highlights the intended nodes, as shown in Fig. 4 for temperature
0, demonstrating strong alignment with the ground truth. This level of consistency
suggests that deterministic retrieval ensures reliable knowledge attribution, making it
especially suitable for tasks where transparency and factual accuracy are essential.
In contrast, at a temperature of 1, the LLM generates responses, introducing
variability in how information is processed. Instead of strictly adhering to the retrieved
KG components, the LLM may shift its focus to different nodes, sometimes misaligning
with the expected ground truth highlights. This misalignment is evident in several cases:
•InWhat has a synergistic interaction with Flurandrenolide drug? , the LLM empha-
sizes incorrect node ’neurofibrillary tangles’ alongwith correct node ’insulin peglispro’,
resulting in an AUC of 0.89.
•InWhat is UDP-glucose:glycoprotein glucosyltransferase activity associated with and
how it interacts with serotonin:sodium symporter activity through ppi? , again picks
the least relevant nodes, leading to an AUC of 0.78.
•InWhat is Insulin tregopil and with which drug does it synergistic interaction? , the
LLM highlights equal correct and incorrect nodes, causing the AUC to drop to 0.56,
indicating a complete deviation from the expected explanation.
18

Fig. 4 : Accuracy Comparison for the Query “What is insulin-like growth factor receptor
binding associated with?”.
These variations suggest that at higher temperature settings, the model may
rely on a broader context rather than strictly following the retrieved knowledge,
potentially introducing hallucinations or misattributions [ 5]. However, not all queries
are equally affected. For instance, What is pancreatic serous cystadenocarcinoma
associated with? andWhat is glucagon receptor activity associated with and how it
interacts through ppi with low-affinity glucose:proton symporter activity? maintain
perfect alignment (AUC = 1) across both temperature settings, indicating that in
these cases, the LLM does not introduce additional noise and remains true to the
retrieved knowledge. Conversely, for queries like What is glucose binding? and”What
is glucose 1-phosphate phosphorylation and mitochondrial genome maintenance, how
do they interact through ppi? , the variability at temperature 1 results in a drop in AUC
to 0.89 and 0.78, respectively, highlighting the inconsistencies introduced by the LLM
during explanations.
The observations align with the broader role of temperature settings in LLMs,
where the temperature parameter controls the level of randomness in the generated
text. This parameter determines how likely the model is to select less probable words
instead of the most probable ones [ 103,104]. The impact of temperature settings on
text generation is well-documented:
•Temperature = 0: The model consistently picks the most probable next word,
resulting in deterministic output. This leads to consistent and factual responses,
making it ideal for applications that require high accuracy and predictability. However,
responses may sometimes be repetitive or lack creativity.
•Temperature = 1: The model introduces more randomness, allowing it to select
from a broader range of probable words. This produces more diverse, creative, and
19

human-like responses, which are suitable for applications like storytelling, brain-
storming, and creative writing. However, this increased diversity can sometimes
compromise coherence and factual accuracy.
These effects illustrate how temperature settings influence the explainability of
GraphRAG. A lower temperature ensures that the retrieval process is deterministic,
leading to consistent and explainable highlighting of relevant KG components. In
contrast, generating responses at a higher temperature introduces variability, which
can lead to mismatches between retrieved and highlighted knowledge. Although this
variability can enhance response diversity in some contexts, it can also compromise
explainability, making it more challenging to trace the generated answer back to the kG.
For applications where explainability and factual alignment are critical, such as
healthcare scenarios, scientific research, and knowledge-intensive question answering, a
lower temperature setting is preferable. This approach ensures that the model reliably
highlights the most relevant graph components [5, 102].
5.3 ATT Faithfulness
ATT faithfulness refers to the extent to which an explanation accurately reflects a
model’s reasoning process. A faithful attribution assigns high importance to inputs
that genuinely drive the model’s output, rather than highlighting spurious correlations
or irrelevant features. This property is widely recognised as essential for trustworthy
interpretability [105, 106].
In our study, we operationalise ATT faithfulness for GraphRAG by measuring
the correlation between attribution-based metrics produced by KG–SMILE (here,
ATT-AUC at different decoding temperatures: at T=0 retrieval is deterministic, while
atT=1 the LLM introduces variability in its responses) and the externally reported
benchmark accuracies of the same model. Concretely, for OpenAI’s GPT-3.5-turbo, for
each prompt t, we define
xt:= external benchmark accuracy of GPT-3.5-turbo on dataset/prompt t,
yt:= ATT-AUC score of KG-SMILE for GPT-3.5-turbo on dataset/prompt t.
We then compute the correlation between these values. Importantly, our results
show that whenever the model achieves higher benchmark accuracy, the corresponding
ATT-AUC scores are also higher, indicating that more faithful explanations accompany
better model performance. As summarised in Table 3, across n=10 prompts we observe a
strong positive correlation at T=0 (r=0.933) and a weak correlation at T=1 (r=0.070).
Formally, we compute the Pearson correlation coefficient:
r=Pn
i=1 
xi−¯x 
yi−¯y
pPn
i=1(xi−¯x)2pPn
i=1(yi−¯y)2, (23)
where xiare the externally reported benchmark accuracies, yiare the corresponding
KG-SMILE attribution scores (ATT-AUC at T=0 or T=1), and ¯x,¯ydenote their
20

means. A strong positive rindicates that attribution metrics faithfully track external
measures of model competence.
Temperature T n Pearson r
0 10 0.975524
1 10 0.845926
Table 3 : Pearson rbetween
ATT–AUC ( yi) and external
accuracy ( xi).
Thus, in our framework, ATT faithfulness is an externally validated property:
explanations are considered reliable to the extent that KG-SMILE ATT-AUC scores cor-
relate with established benchmarks, ensuring that interpretability provides meaningful
insights into knowledge-graph–augmented language models, particularly in high-stakes
applications [107].
5.4 ATT Stability
ATT stability evaluates how well a model maintains its accuracy and consistency in
the face of minor perturbations to the KG [ 108]. This property is a vital component of
an explainable model, as it ensures that small changes to the input do not significantly
affect the model’s predictions or the explanations it generates [20, 100].
In this study, ATT stability was assessed by perturbing the KG through slight
modifications to its structure, such as altering relationships between nodes, and then
measuring the similarity between the original and perturbed explanations. The Jaccard
index [49, 109], defined in Eq. 24, was used for this purpose:
Jaccard (A, B) =|A∩B|
|A∪B|(24)
Where Ais the set of explanations from the original graph and Bis the set of
explanations from the perturbed graph. The index ranges from 0 to 1 and quantifies
the similarity between the two sets. Higher values indicate greater ATT stability.
The results in Table 4. indicate that the ATT stability of explanations generated by
the knowledge graph (KG) model is significantly influenced by both structural changes
and temperature settings [100, 108].
As shown in Fig. 5, Analysis of temperature settings shows that at T= 0, where
the model operates deterministically, explanations remain stable despite minor mod-
ifications to the KG. Here, T denotes the temperature parameter controlling the
randomness of the model’s output. This is evidenced by consistently high Jaccard simi-
larity values (1.000 in multiple cases), indicating that the model retrieves explanations
that are nearly identical to those generated from the original KG. This suggests that,
in a low-temperature setting, the model prioritizes fixed reasoning paths, maintaining
robustness against small perturbations and ensuring consistency and reliability in its
explanations [20].
21

Perturbation triple added Jaccard ( T=1) Jaccard ( T=0)
(“insulin-like growth factor receptor binding”,
“associated with”, “type 1diabetes”)1.00 0.10
(“pancreatic serous cystadenocarcinoma”,
“treated with”, “chemotherapy”)0.90 1.00
(“glucagon receptor activity”, “inhibited by”,
“insulin”)0.80 1.00
(“glucose binding”, “expression present”,
“pancreas”)0.70 1.00
(“Flurandrenolide”, “used for”,
“diabetic skin conditions”)0.80 1.00
(“glucose 1-phosphate phosphorylation”,
“associated with”, “glycogen storage disease”)1.00 1.00
(“UDP-glucose:glycoprotein glucosyltransferase
activity”, “regulates”, “protein folding”)0.90 1.00
(“SFT2D2”, “expression present”,
“pancreatic islets”)1.00 1.00
(“diabetic peripheral angiopathy”, “caused by”,
“chronic hyperglycemia”)1.00 1.00
(“Insulin tregopil”, “used for”,
“type 2diabetes”)1.00 1.00
Table 4 : Perturbation triples and corresponding Jaccard similarities for T=1 and T=0.
Conversely, at T= 1, where the model introduces stochasticity in its response
generation, a slight decline in ATT stability is observed, with Jaccard similarity values
dropping a certain values in most of the cases. This indicates that higher temperature
settings lead to greater sensitivity to perturbations, prompting the model to explore
alternative reasoning pathways and retrieve explanations that deviate significantly
from the original [ 110]. The introduction of additional triples, even if unrelated to the
queried entity, alters the structure of the graph, causing a cascading effect that disrupts
the model’s ability to maintain consistency in its explanations [ 108], as illustrated in
Fig. 6, which depicts the ATT stability analysis.
This contrast between deterministic behavior at T= 0 and stochastic behavior
atT= 1 highlights the dual nature of temperature settings in LLM-driven KG
reasoning [ 102]. While a low-temperature setting ensures ATT stability, a higher-
temperature setting enhances adaptability, allowing the model to explore diverse
responses but at the cost of increased variability in explanations. The sensitivity
observed at T= 1 suggests that even minor structural modifications in the KG can
significantly alter the retrieved explanations, reducing the model’s reliability in dynamic
environments where perturbations are common. This raises concerns about the trade-
off between explainability and adaptability, as a more stochastic approach may enhance
response diversity but weaken robustness in explanation retrieval [100].
22

Fig. 5 : Impact of Adding the Triplet (”glucagon receptor activity”, ”inhibited by”,
”insulin”) at Temperature 0: Jaccard Similarity 1.0 in Knowledge Graphs
Fig. 6 : Impact of Adding the Triplet (”glucagon receptor activity”, ”inhibited by”,
”insulin”) at Temperature 1: Jaccard Similarity 0.8 in Knowledge Graphs
The findings emphasize the importance of temperature control in KG-based rea-
soning models, as temperature directly affects the balance between stability and
exploration. A key implication is that models designed for high-explainability tasks
should favor lower temperatures, where reasoning pathways remain consistent even
under minor modifications to the KG [20, 108].
5.5 Consistency
Consistency assesses how predictable the system is in its responses to small pertur-
bations in the KG. It evaluates how stable and reliable the model’s responses are
23

when minor modifications are made to the KG structure [ 20,110]. In our analysis,
we applied slight perturbations across different sections of the KG and evaluated the
consistency of the model’s responses. Across 50 runs, the responses varied minimally
across different partitions of the KG (parts 1 through 10). Part 1 exhibited the least
variability, and the remaining parts also showed only minor variations, indicating that
the system’s responses remained consistent in the face of small changes across all
sections [ 100]. The narrow range of variations suggests that the model’s responses
were stable and largely unaffected by minor perturbations. Despite the slight differ-
ences observed, the model demonstrated strong overall consistency by maintaining
reliable responses across all parts of the KG. This suggests that the system is highly
predictable when minor perturbations are introduced and can handle them without a
significant decline in performance. However, if more extensive or systematic alterations
are applied to critical nodes in the KG, greater variation could occur. Small, localized
changes, however, did not alter the system’s high degree of consistency [108].
The results from the 50 runs confirm strong consistency across most sections of
the KG and demonstrate that the model can produce stable and predictable responses
even when the graph is exposed to minor perturbations.
5.6 Computation Complexity
Method BayLIME Linear Regression
Cosine similarity 25.01s 25.60s
Inverse Wasserstein distance 20.02s 25.20s
Inverse Wasserstein + Cosine (hybrid) 23.32s 23.43s
Wasserstein distance 27.02s 25.68s
Wasserstein + Cosine (hybrid) 18.18s 17.17s
Table 5 : Wall time comparison of ATT fidelity configurations. The graph metric is
fixed to cosine similarity; row labels denote the text-side metric.
To systematically evaluate ATT fidelity across text and graph representations, we
analyzed multiple similarity metrics, including Cosine [101],Wasserstein + Cosine
(hybrid) [95,111],Inverse Wasserstein distance [94], and Inverse Wasserstein
+ Cosine (hybrid) [96]. Our experiments, which included 20 perturbations
per configuration [85], compared the performance of BayLIME [112] and Linear
Regression models [ 98], focusing on both computational efficiency and ATT
fidelity [18].
Table 5 illustrates the trade-off between computational efficiency and accuracy when
analyzing text and graph representations [ 20]. The fastest method, which combines
Hybrid Text Metrics (Wasserstein Distance + Cosine) with Graph Metrics (Cosine)
ATT fidelity Analysis, achieves the lowest processing time of 17.17 seconds . This
makes it the most efficient option for rapid evaluations while still maintaining reasonable
24

accuracy, as confirmed by linear regression analysis. In contrast, the BayLIME ATT
fidelity Analysis of Text-to-Text using Inverse Wasserstein Distance and Graph-to-
Graph using Cosine, takes 27.02 seconds . This method entails a significantly higher
computational cost due to its use of Inverse Wasserstein Distance, which improves
explainability by effectively capturing opposing nodes [11].
These results highlight the trade-off between computational efficiency and
ATT fidelity , emphasizing the importance of selecting appropriate similarity metrics
based on the specific requirements of the task [ 102]. Experiments were performed on
Google Colab’s standard runtime environment, which provides approximately 12GB of
RAM, demonstrating efficient time consumption compared to alternative approaches.
5.7 Chain of Thought
In complex QA tasks, particularly those involving GraphRAG, traditional LLMs often
struggle to produce meaningful responses due to their reliance on direct retrieval and
pattern matching [ 113,114]. Many real-world queries require multi-step reasoning,
where multiple entities and relationships within a KG must be linked together before
arriving at a well-supported conclusion. To address this challenge, we employ Chain-
of-Thought (CoT) reasoning [ 115], a structured approach that guides the GraphRAG
system through intermediate reasoning steps, ensuring that answers are generated in a
logical, explainable manner while leveraging the structured representation of the KG.
Example Question Requiring Chain-of-Thought Reasoning in GraphRAG Consider
the following complex question:
’Which disease is associated with glucose binding through its link to neurofibrillary tangles,
and how does this connect further to a condition related to calcium-release channel activity?’
This query requires GraphRAG to conduct multi-step reasoning, as it involves:
•Understanding glucose binding glucose binding through its link to neurofibrillary
tangles.
•Understanding diseases associated with diabetic binding.
•Analysing its connection to a condition related to calcium-release channel activity.
A simple LLM query often fails to adequately address such questions, as it lacks the
ability to connect these concepts explicitly within a structured KG. When simply run,
it may return: “I’m sorry, I don’t have enough information to provide a helpful answer.”
Instead, a CoT-based GraphRAG reasoning framework extracts relevant knowledge
from the KG, traces logical connections, and formulates an answer grounded in explicit
relationships rather than implicit language model heuristics [116, 117].
5.7.1 Extracting and Structuring Relevant KG Triples
The first step in this approach is extracting key entities and relations from the
input question using NLP. The function extract entities relations(question)
identifies relevant entities e.g. ’glucose’, ’disease’, ’diabetic’, ’neurofibrillary tan-
gles’ ) and relationships (e.g., bind’, ’relate’, ’associate’, ’interact’). The function
generate chain ofthought(kg, question) then iterates through subject-predicate-
object triples stored in the KG, constructing a reasoning chain such as:
25

glucose binding →[expression present] →Neurofibrillary tangles
Neurofibrillary tangles →[associated with] →Disease
Disease →[associated with] →calcium-release channel activity
By extracting and organizing these structured triples, the GraphRAG model
enhances reasoning interpretability, ensuring that each step in the answer is traceable
within the KG.
Complex questions such as the one above cannot be effectively answered by a direct
LLM query alone. Unlike simple retrieval-based approaches, which rely on match-
ing isolated facts, GraphRAG with CoT reasoning ensures a structured approach to
logical inference. Without CoT, the system may fail to correctly connect multi-hop
relationships, leading to fragmented or incomplete answers. The CoT framework decom-
poses queries into sequential reasoning steps, ensuring both accurate and explainable
responses [118].
Once the reasoning chain is established, the extracted triples are format-
ted into a structured prompt using the function format triples forprompt(
reasoning chain) . This process guides the LLM with explicit, factually grounded
information, reduces hallucinations, and ensures responses remain faithful to the
KG. By explicitly presenting the logical sequence of facts, the CoT approach enables
GraphRAG to generate structured and context-aware answers.
5.7.2 Explainability Through Perturbation Analysis
To assess the explainability of GraphRAG, we implemented KG-SMILE, a perturbation-
based approach that systematically altered the KG structure and evaluated the impact
on generated responses. In each iteration, we generated a perturbed version of the KG
by selectively removing or modifying nodes and edges. This perturbed KG was then
fed into the Chain-of-Thought (CoT) reasoning framework to generate an answer. By
comparing the perturbed responses with the original, we measured their alignment
using both cosine similarity [ 101] and inverse Wasserstein distance [ 95]. This iterative
process effectively identified the most influential nodes and relationships [ 20,85],
revealing their critical role in shaping the final generated response. As illustrated in
Fig. 7, the importance coefficients highlight key nodes which significantly contribute to
the reasoning. KG-SMILE provides a structured mechanism to interpret the decision-
making process of GraphRAG, ensuring greater transparency and explainability in
AI-driven question answering.
5.8 Pre-Prompt Mechanism
A common challenge in KGQA systems is their occasional inability to retrieve relevant
information when queried with a specific phrasing, often resulting in responses such as ”I
do not know.” This limitation arises due to the sensitivity of retrieval models to linguistic
variations and the inherent constraints of structured knowledge representations. To
address this issue, we employ a pre-prompting mechanism that generates multiple
26

Fig. 7 : Chain of Thought KG-SMILE Explainability
rephrased versions of the original query [ 119,120], increasing the likelihood of
alignment with stored knowledge. By broadening the phrasing spectrum, this approach
enhances retrieval efficacy and mitigates the risk of response failure.
Once the rephrased questions are generated, they are individually processed through
GraphQAChain to obtain possible answers. However, responses that still return ”I
do not know” or similar uncertainty indicators are systematically discarded [ 121]. A
filtering mechanism ensures that only valid, meaningful responses are retained for
further analysis. This selective approach significantly enhances response reliability
by eliminating uncertainty and ensuring that the final answer is constructed from
informative, contextually relevant outputs.
27

To determine the most accurate response, the filtered answers are aggregated
using embedding-based similarity techniques [ 13], which synthesize the most
semantically representative response. By leveraging multiple valid responses rather
than a single direct retrieval, this process improves answer robustness and completeness.
However, while this method enhances response accuracy, it introduces a trade-off in
explainability [ 107]. Because the final answer is an aggregation of multiple responses
rather than directly extracted from a single node, the system highlights more nodes in
the KG, similar to the way a simple LLM-based retrieval process does but with slightly
less intensity. While the correct knowledge component is still identified, the aggregation
process results in a broader distribution of emphasis across multiple nodes rather
than pinpointing a single definitive source. As illustrated in Fig. 8,by integrating pre-
prompting, selective response filtering, and answer aggregation, I significantly enhanced
the stability, accuracy, and interoperability of KG-based QA systems. This approach
ensures that even if an initial direct query results in ”I do not know,” alternative
formulations allow the system to retrieve a meaningful response. While explicit node
attribution may be more distributed, the modelstill correctly highlights the relevant
knowledge components, making this technique particularly valuable for high-stakes
applications such as healthcare research, institutional knowledge management, and
decision-support systems [122], where both precision and transparency are critical.
5.9 Cross-Model Evaluation
This section discusses cross-model evaluation to analyse the performance of responses
generated by the GraphRAG system against two specialized medical LLMs named
Mistral and MedAlpaca. The objective of performing cross-model evaluation is to
investigate how well the responses generated by this GraphRAG align with responses
generated by medical LLMs for the same 10 questions. This analysis provides insights
into the strengths and limitations of our approach. This system uses a constrained
dataset from PrimeKGQA. Mistral and MedAlpaca are often considered suitable
corpora for assessing ”ground truth” in biomedical querying. These two LLMs are
trained and fine-tuned on domain-specific medical corpora. For each of the 10 questions,
responses are generated independently using these models. To compare responses,
similarities are computed using a composite metric containing semantic and concept
similarity. Mean Composite Similarity is an average score combining semantic (meaning-
based), concept overlap (key ideas match), and content (word/structure match). The
±number represents the standard deviation, indicating how much scores vary, with
higher values indicating greater inconsistency. To assess semantic similarity, embedding
similarity is calculated using Sentence Transformers. Concept overlap and content
similarity measures are performed using key concept extraction and cosine similarity.
Mean Semantic Similarity represents just the meaning-based part by ignoring exact
words and focusing on ideas only. Classifications were assigned based on similarity
thresholds: very low ( <0.3), low (0.3–0.5), medium (0.5–0.7), high ( >0.7) A small
subset of the full PrimeKGQA is used to implement GraphRAG, which constrains
retrieved information. This subset focuses on specific entity-relation pairs relevant
to biomedical concepts, leading to potentially incomplete responses compared to the
comprehensive knowledge encoded in Mistral and MedAlpaca. At the temperature of
28

Fig. 8 : Preprompt KG-SMILE explainability
0, GraphRAG operates deterministically, strictly fetching triplets from the KG. At
temperature 1, the system allows partial integration of the LLM’s internal knowledge,
resulting in more elaborate responses. However, in our Graph-based RAG, outputs
are highly dependent on the supplied KG edges and nodes. Whereas medical LLMs
can generalize from trained patterns, they may not align precisely with PrimeKGQA’s
specific data. The medical LLMs possess broader clinical knowledge and alternative
interpretations that differ from our subset of KG. Additionally, potential hallucinations
in the medical LLMs could further contribute to mismatches, though their medical
fine-tuning mitigates this to some extent.
29

Comparison Mean Composite
SimilarityMean Semantic
SimilarityClassification
Distribution
Temperature 0
vs Mistral 0.319 ( ±0.214) 0.515 very low: 4
(40.0%)
low: 3 (30.0%)
medium: 3
(30.0%)
vs MedAlpaca 0.017 ( ±0.034) 0.017 very low: 10
(100.0%)
Temperature 1
vs Mistral 0.470 ( ±0.198) 0.716 very low: 3
(30.0%)
low: 3 (30.0%)
medium: 3
(30.0%)
high: 1 (10.0%)
vs MedAlpaca 0.020 ( ±0.024) 0.020 very low: 10
(100.0%)
Table 6 : Similarity statistics for GraphRAG outputs at different temperatures against
medical LLMs.
The results in Table 6 indicate low overall similarities to both Mistral and MedAl-
paca. Considering KG-based constraints: at a temperature of 0, it is highly precise
for KG triplets but lacks the depth of context-rich outputs of medical LLMs. For
instance, in Question 1, GraphRAG at temp 0 Mistral provides a broader physiological
explanation, whereas GraphRAG interprets relationships and entities strictly from the
literal KG subset. Temperature 1 exhibits modest improvements (e.g., 70% better than
Mistral), as it leverages the LLM’s internal knowledge to elaborate; however, it remains
KG-dependent and thus diverges from the medical models’ parametric generalizations.
Temp 1 performs slightly better than Temp 0 against Mistral, as it utilizes AI knowl-
edge, but has poor alignment with MedAlpaca. MedAlpaca results are poorer due to
its clinically detailed focus, which prioritizes ethical, patient-oriented phrasing that is
missing in our PrimeKGQA subset. The PrimeKGQA subset excludes comprehensive
relations, resulting in ”differing responses.” The KG’s sparsity restricts response depth,
unlike the medical LLMs’ broader knowledge. Despite this, the results are not entirely
poor; the moderate similarity to Mistral (0.47 at Temp 1) and consistency (0.617 vs.
MedAlpaca) suggest reliable outputs within the KG’s scope, potentially comparable
to human evaluation for fact-specific queries. The overall assessment results confirm
that the PrimeKGQA subset is accurate and that the KG-based RAG is precise, but it
needs a more comprehensive KG to broaden its coverage, marking a key direction for
future work. Additional improvements could involve iterative KG expansion to capture
emerging biomedical relations.
30

5.10 Future Prospects
GraphRAG technology faces several key challenges while also offering promising
research opportunities. One area of focus is the development of dynamic and adaptive
graphs instead of static databases [ 74,123–127] that can incorporate real-time updates.
Additionally, integrating multi-modal data such as images and audio would enhance
knowledge representation [128].
Addressing scalability is crucial for managing large-scale KGs, which requires
advanced retrieval mechanisms and efficient infrastructure. The use of graph foundation
models [42, 115]can improve the processing of graph-structured data, while lossless
compression techniques are necessary for effectively handling long contexts [123, 125]
Establishing standard benchmarks would create a framework for evaluating method-
ologies, promoting consistency and progress within the field. Expanding the applications
of GraphRAG to complex domains, such as healthcare [129], finance [130], legal com-
pliance [ 131], and the Internet of Things (IoT) [ 132], could significantly increase its
impact.
These challenges highlight the potential for GraphRAG to evolve into a robust and
versatile technology across a variety of domains.
6 Conclusion
In conclusion, this study presents KG-SMILE as a framework that brings explainability
and interpretability to GraphRAG systems. While standard RAG helps reduce halluci-
nations by grounding outputs in external knowledge, it still functions as a black box,
users cannot see which retrieved pieces of information shaped the final response. By
introducing perturbation-based attribution analysis, KG-SMILE makes this process
more transparent: it systematically alters graph components and observes the effect on
model outputs, thereby identifying the entities and relations most influential in gener-
ating a response. In this way, GraphRAG becomes not only a retrieval-and-generation
pipeline but also an explainable system that can clarify whyan answer was produced.
This capacity for explanation is especially important in sensitive domains such as
healthcare and law, where decision-making requires not only accurate outputs but
also accountability and trust in the reasoning process. Rather than claiming higher
performance, the emphasis here is on transparency and traceability, ensuring that users
can understand the link between retrieved knowledge and generated answers. Looking
forward, continued research into dynamic graph adaptation, multi-modal integration,
scalability, and standardized evaluation benchmarks will further shape the role of
explainable GraphRAG systems. With these directions, such frameworks can provide a
foundation for responsible AI applications across various sectors, including healthcare,
finance, IoT, and legal systems.
A Additional Results
To complement the results presented in the main text, we provide in Table 7 shows
the area under the ROC curve (AUC) scores for biomedical question–answer pairs
31

evaluated at two different decoding temperatures. These results illustrate the effect of
temperature variation on answer consistency across a range of biomedical tasks.
Question AUC ( T=0) AUC ( T=1)
What is insulin-like growth factor receptor binding
associated with?1.00 0.78
What is pancreatic serous cystadenocarcinoma associated
with?1.00 1.00
What is glucagon receptor activity associated with and
how does it interact through ppi with low-affinity
glucose:proton symporter activity?1.00 1.00
What is glucose binding? 1.00 0.89
Which lifestyle changes are most effective in preventing
neonatal insulin-dependent diabetes mellitus?0.50 0.44
What is glucose 1-phosphate phosphorylation and
mitochondrial genome maintenance, and how do they
interact through ppi?1.00 0.78
What is UDP-glucose:glycoprotein glucosyltransferase
activity associated with, and how does it interact with
serotonin:sodium symporter activity through ppi?0.89 0.78
Where is SFT2D2 expressed? 0.89 0.78
What is diabetic peripheral angiopathy associated with? 1.00 0.89
Which molecular interactions are associated with
neonatal insulin-dependent diabetes mellitus?0.50 0.10
Table 7 : AUC (area under the ROC curve) at decoding temperatures T=0 and T=1;
”ppi” denotes protein–protein interaction
32

B Declarations
B.1 Funding
No funding was received for conducting this study.
B.2 Competing Interests
The authors declare that they have no competing interests.
B.3 Ethical Considerations
his study did not involve human participants or animals; therefore, ethics approval
and informed consent were not required.
B.4 Data Availability
The datasets generated and analysed during the current study are available in
the PrimeKGQA repository, URL: https://zenodo.org/records/13829395, and DOI:
10.5281/zenodo . 13348626. https://github.com/koo-ec/XGRAG
B.5 Author Contributions
Zahra Zehtabi Sabeti Moghaddam: Conceptualization, methodology, analysis, and
writing – original draft.
Zeinab Dehghani: Analysis, writing, and contributions to coding.
Maneeha Rani: Contributions to coding.
Dr. Koorosh Aslansefat: Supervision, methodology, review, and editing.
Dr. Bupesh Mishra: Review and editing.
Dr. Rameez Raja Kureshi: Review.
Prof. Dhaval Thakker: Supervision, Review.
References
[1] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., K¨ uttler,
H., Lewis, M., Yih, W.-t., Rockt¨ aschel, T., et al. : Retrieval-augmented generation
for knowledge-intensive nlp tasks. Advances in neural information processing
systems 33, 9459–9474 (2020)
[2]B´ echard, P., Ayala, O.M.: Reducing hallucination in structured outputs via
retrieval-augmented generation. arXiv preprint arXiv:2404.08189 (2024)
[3]Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida,
D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report.
arXiv preprint arXiv:2303.08774 (2023)
[4]Liang, P., Bommasani, R., Lee, T., Tsipras, D., Soylu, D., Yasunaga, M., Zhang,
Y., Narayanan, D., Wu, Y., Kumar, A., et al.: Holistic evaluation of language
models. arXiv preprint arXiv:2211.09110 (2022)
33

[5]Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y.J., Madotto,
A., Fung, P.: Survey of hallucination in natural language generation. ACM
computing surveys 55(12), 1–38 (2023)
[6]Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., Chen, Q., Peng,
W., Feng, X., Qin, B., et al. : A survey on hallucination in large language models:
Principles, taxonomy, challenges, and open questions. ACM Transactions on
Information Systems 43(2), 1–55 (2025)
[7]Lin, S., Hilton, J., Evans, O.: Truthfulqa: Measuring how models mimic human
falsehoods. arXiv preprint arXiv:2109.07958 (2021)
[8]Arrieta, A.B., D´ ıaz-Rodr´ ıguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado,
A., Garc´ ıa, S., Gil-L´ opez, S., Molina, D., Benjamins, R., et al. : Explainable
artificial intelligence (xai): Concepts, taxonomies, opportunities and challenges
toward responsible ai. Information fusion 58, 82–115 (2020)
[9]Shuster, K., Komeili, M., Adolphs, L., Roller, S., Szlam, A., Weston, J.: Language
models that seek for knowledge: Modular search & generation for dialogue and
prompt completion. arXiv preprint arXiv:2203.13224 (2022)
[10]Zhong, L., Wu, J., Li, Q., Peng, H., Wu, X.: A comprehensive survey on automatic
knowledge graph construction. ACM Computing Surveys 56(4), 1–62 (2023)
[11]Aslansefat, K., Hashemian, M., Walker, M., Akram, M.N., Sorokos, I., Papadopou-
los, Y.: Explaining black boxes with a smile: Statistical model-agnostic
interpretability with local explanations. IEEE Software 41(1), 87–97 (2023)
[12]Chalkidis, I., Jana, A., Hartung, D., Bommarito, M., Androutsopoulos, I., Katz,
D.M., Aletras, N.: Lexglue: A benchmark dataset for legal language understanding
in english. arXiv preprint arXiv:2110.00976 (2021)
[13]Reimers, N., Gurevych, I.: Sentence-bert: Sentence embeddings using siamese
bert-networks. arXiv preprint arXiv:1908.10084 (2019)
[14]Kusner, M., Sun, Y., Kolkin, N., Weinberger, K.: From word embeddings to
document distances. In: International Conference on Machine Learning, pp.
957–966 (2015). PMLR
[15]Colombo, P., Staerman, G., Clavel, C., Piantanida, P.: Automatic text evaluation
through the lens of wasserstein barycenters. arXiv preprint arXiv:2108.12463
(2021)
[16]Adadi, A., Berrada, M.: Peeking inside the black-box: a survey on explainable
artificial intelligence (xai). IEEE access 6, 52138–52160 (2018)
[17]Doshi-Velez, F., Kim, B.: Towards a rigorous science of interpretable machine
34

learning. arXiv preprint arXiv:1702.08608 (2017)
[18]Christoph, M.: Interpretable machine learning: A guide for making black box
models explainable (2020)
[19]Rudin, C.: Stop explaining black box machine learning models for high stakes
decisions and use interpretable models instead. Nature machine intelligence 1(5),
206–215 (2019)
[20]Ribeiro, M.T., Singh, S., Guestrin, C.: ” why should i trust you?” explaining
the predictions of any classifier. In: Proceedings of the 22nd ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining, pp. 1135–
1144 (2016)
[21]Lundberg, S.M., Lee, S.-I.: A unified approach to interpreting model predictions.
Advances in neural information processing systems 30(2017)
[22]Guidotti, R., Monreale, A., Ruggieri, S., Turini, F., Giannotti, F., Pedreschi, D.:
A survey of methods for explaining black box models. ACM computing surveys
(CSUR) 51(5), 1–42 (2018)
[23]Du, M., Liu, N., Hu, X.: Techniques for interpretable machine learning.
Communications of the ACM 63(1), 68–77 (2019)
[24] Shapley, L.S., et al.: A value for n-person games (1953)
[25]Lundberg, S.M., Erion, G.G., Lee, S.-I.: Consistent individualized feature
attribution for tree ensembles. arXiv preprint arXiv:1802.03888 (2018)
[26]Sundararajan, M., Najmi, A.: The many shapley values for model explanation.
In: International Conference on Machine Learning, pp. 9269–9278 (2020). PMLR
[27]Plumb, G., Molitor, D., Talwalkar, A.S.: Model agnostic supervised local
explanations. Advances in neural information processing systems 31(2018)
[28]Hoffman, R.R., Mueller, S.T., Klein, G., Litman, J.: Metrics for explainable ai:
Challenges and prospects. arXiv preprint arXiv:1812.04608 (2018)
[29]Laugel, T., Renard, X., Lesot, M.-J., Marsala, C., Detyniecki, M.: Defining locality
for surrogates in post-hoc interpretablity. arXiv preprint arXiv:1806.07498 (2018)
[30]Zhao, X., Huang, W., Huang, X., Robu, V., Flynn, D.: Baylime: Bayesian local
interpretable model-agnostic explanations (supplementary material)
[31]Zhou, Z., Hooker, G., Wang, F.: S-lime: Stabilized-lime for model explanation.
In: Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery
& Data Mining, pp. 2429–2438 (2021)
35

[32]Upadhyay, S., Isahagian, V., Muthusamy, V., Rizk, Y.: Extending lime for business
process automation. arXiv preprint arXiv:2108.04371 (2021)
[33]Visani, G., Bagli, E., Chesani, F.: Optilime: Optimized lime explanations for
diagnostic computer algorithms. arXiv preprint arXiv:2006.05714 (2020)
[34]Shankaranarayana, S.M., Runje, D.: Alime: Autoencoder based approach for
local interpretability. In: Intelligent Data Engineering and Automated Learning–
IDEAL 2019: 20th International Conference, Manchester, UK, November 14–16,
2019, Proceedings, Part I 20, pp. 454–463 (2019). Springer
[35]Saadatfar, H., Kiani-Zadegan, Z., Ghahremani-Nezhad, B.: Us-lime: Increasing
fidelity in lime using uncertainty sampling on tabular data. Neurocomputing
597, 127969 (2024)
[36]Zafar, M.R., Khan, N.: Deterministic local interpretable model-agnostic expla-
nations for stable explainability. Machine Learning and Knowledge Extraction
3(3), 525–541 (2021)
[37]Garreau, D., Luxburg, U.: Explaining the explainer: A first theoretical analysis
of lime. In: International Conference on Artificial Intelligence and Statistics, pp.
1287–1296 (2020). PMLR
[38] Mishra, S., Sturm, B.L., Dixon, S.: Local interpretable model-agnostic explana-
tions for music content analysis. In: ISMIR, vol. 53, pp. 537–543 (2017)
[39]Li, X., Xiong, H., Li, X., Zhang, X., Liu, J., Jiang, H., Chen, Z., Dou, D.: G-lime:
Statistical learning for local interpretations of deep neural networks using global
priors. Artificial Intelligence 314, 103823 (2023)
[40]Huang, Q., Yamada, M., Tian, Y., Singh, D., Chang, Y.: Graphlime: Local
interpretable model explanations for graph neural networks. IEEE Transactions
on Knowledge and Data Engineering 35(7), 6968–6972 (2022)
[41]Abdullah, T.A., Zahid, M.S.M., Ali, W., Hassan, S.U.: B-lime: An improvement
of lime for interpretable deep learning classification of cardiac arrhythmia from
ecg signals. Processes 11(2), 595 (2023)
[42]Knab, P., Marton, S., Bartelt, C.: Beyond pixels: Enhancing lime with
hierarchical features and segmentation foundation models. arXiv preprint
arXiv:2403.07733 (2024). DSEG-LIME: Data -Driven Segmentation LIME :con-
tentReference[oaicite:1]index=1
[43]Bora, N., Others: SLICE: Structured sampling to improve lime image explanations.
In: Proceedings of [venue] (2024). DOI or arXiv not found; placeholder entry;
:contentReference[oaicite:2]index=2
36

[44]Rashid, M., Amparore, E.G., Ferrari, E., Verda, D.: Using stratified sampling
to improve lime image explanations. Proceedings of the AAAI Conference on
Artificial Intelligence 38(13), 14785–14792 (2024) https://doi.org/10.1609/aaai.
v38i13.29397 . Stratified LIME :contentReference[oaicite:3]index=3
[45]Lam, F., Others: Ss -lime: Self -supervised region detection for enhanced local
explanations. arXiv preprint arXiv:2501.xxxxx (2025). SS -LIME placeholder; not
found in search results
[46]Knab, P., Marton, S., Schlegel, U., Bartelt, C.: A Survey of LIME Extensions in
Deep Learning. Survey placeholder; :contentReference[oaicite:4]index=4 (2025)
[47]Dehghani, Z., Aslansefat, K., Khan, A., Rivera, A.R., George, F., Khalid, M.:
Mapping the Mind of an Instruction-based Image Editing using SMILE (2024).
https://arxiv.org/abs/2412.16277
[48]Dehghani, Z., Akram, M.N., Aslansefat, K., Khan, A., Papadopoulos, Y.: Explain-
ing Large Language Models with gSMILE (2025). https://arxiv.org/abs/2505.
21657
[49]Ahmadi, S.M., Aslansefat, K., Valcarce-Dineiro, R., Barnfather, J.: Explain-
ability of point cloud neural networks using smile: Statistical model-agnostic
interpretability with local explanations. arXiv preprint arXiv:2410.15374 (2024)
[50]Pan, S., Luo, L., Wang, Y., Chen, C., Wang, J., Wu, X.: Unifying large language
models and knowledge graphs: A roadmap. IEEE Transactions on Knowledge
and Data Engineering (2024)
[51]Li, M., Miao, S., Li, P.: Simple is effective: The roles of graphs and large language
models in knowledge-graph-based retrieval-augmented generation. arXiv preprint
arXiv:2410.20724 (2024)
[52]Sarmah, B., Mehta, D., Hall, B., Rao, R., Patel, S., Pasquali, S.: Hybridrag:
Integrating knowledge graphs and vector retrieval augmented generation for
efficient information extraction. In: Proceedings of the 5th ACM International
Conference on AI in Finance, pp. 608–616 (2024)
[53]Baek, J., Aji, A.F., Saffari, A.: Knowledge-augmented language model
prompting for zero-shot knowledge graph question answering. arXiv preprint
arXiv:2306.04136 (2023)
[54]Li, Y., Zhang, X., Luo, L., Chang, H., Ren, Y., King, I., Li, J.: G-refer: Graph
retrieval-augmented large language model for explainable recommendation. In:
Proceedings of the ACM on Web Conference 2025, pp. 240–251 (2025)
[55]Balanos, G., Chasanis, E., Skianis, K., Pitoura, E.: Kgrag-ex: Explainable
retrieval-augmented generation with knowledge graph-based perturbations. arXiv
37

preprint arXiv:2507.08443 (2025)
[56]Wu, R., Cai, P., Mei, J., Wen, L., Hu, T., Yang, X., Fu, D., Shi, B.: Kg-traces:
Enhancing large language models with knowledge graph-constrained trajectory
reasoning and attribution supervision. arXiv preprint arXiv:2506.00783 (2025)
[57]Baghershahi, P., Fournier, G., Nyati, P., Medya, S.: From nodes to narratives:
Explaining graph neural networks with llms and graph context. arXiv preprint
arXiv:2508.07117 (2025)
[58]Peng, B., Zhu, Y., Liu, Y., Bo, X., Shi, H., Hong, C., Zhang, Y., Tang, S.:
Graph retrieval-augmented generation: A survey. arXiv preprint arXiv:2408.08921
(2024)
[59]Sun, J., Xu, C., Tang, L., Wang, S., Lin, C., Gong, Y., Shum, H.-Y., Guo, J.:
Think-on-graph: Deep and responsible reasoning of large language model with
knowledge graph. arXiv preprint arXiv:2307.07697 (2023)
[60]Li, Y., Shen, Y., Nian, Y., Gao, J., Wang, Z., Yu, C., Li, S., Wang, J., Hu, X.,
Zhao, Y.: Mitigating hallucinations in large language models via causal reasoning.
arXiv preprint arXiv:2508.12495 (2025)
[61]Ding, H., Pang, L., Wei, Z., Shen, H., Cheng, X.: Retrieve only when it needs:
Adaptive retrieval augmentation for hallucination mitigation in large language
models. arXiv preprint arXiv:2402.10612 (2024)
[62]Tang, J., Zhang, Q., Li, Y., Li, J.: Grapharena: Benchmarking large language
models on graph computational problems. arXiv preprint arXiv:2407.00379
(2024)
[63]Chen, Z., Mao, H., Li, H., Jin, W., Wen, H., Wei, X., Wang, S., Yin, D., Fan,
W., Liu, H., et al. : Exploring the potential of large language models (llms) in
learning on graphs. ACM SIGKDD Explorations Newsletter 25(2), 42–61 (2024)
[64]Cedro, M., Martens, D.: Graphxain: Narratives to explain graph neural networks.
arXiv preprint arXiv:2411.02540 (2024)
[65]Gumaan, E.: Theoretical foundations and mitigation of hallucination in large
language models. arXiv preprint arXiv:2507.22915 (2025)
[66]Chen, R., Arditi, A., Sleight, H., Evans, O., Lindsey, J.: Persona vectors: Mon-
itoring and controlling character traits in language models. arXiv preprint
arXiv:2507.21509 (2025)
[67]Brown, T.B., Mann, B., Ryder, N., et al. : Language models are few-shot learners.
Advances in Neural Information Processing Systems 33, 1877–1901 (2020)
38

[68]Izacard, G., Lewis, P., Lomeli, M., Hosseini, L., Petroni, F., Schick, T., Dwivedi-
Yu, J., Joulin, A., Riedel, S., Grave, E.: Few-shot learning with retrieval
augmented language models. arXiv preprint arXiv:2208.03299 1(2), 4 (2022)
[69]Surden, H.: Machine learning and law: An overview. Research handbook on big
data law, 171–184 (2021)
[70]Nakano, R., Hilton, J., Balaji, S., Wu, J., Ouyang, L., Kim, C., Hesse, C., Jain, S.,
Kosaraju, V., Saunders, W., et al.: Webgpt: Browser-assisted question-answering
with human feedback. arXiv preprint arXiv:2112.09332 (2021)
[71]Manakul, P., Liusie, A., Gales, M.J.: Selfcheckgpt: Zero-resource black-box
hallucination detection for generative large language models. arXiv preprint
arXiv:2303.08896 (2023)
[72]Bommasani, R., Hudson, D.A., Adeli, E., Altman, R., Arora, S., Arx, S., Bernstein,
M.S., Bohg, J., Bosselut, A., Brunskill, E., et al.: On the opportunities and risks
of foundation models. arXiv preprint arXiv:2108.07258 (2021)
[73]Wang, M.Y.: Deep graph library: Towards efficient and scalable deep learning
on graphs. In: ICLR Workshop on Representation Learning on Graphs and
Manifolds (2019)
[74]Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S.,
Metropolitansky, D., Ness, R.O., Larson, J.: From local to global: A graph
rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130
(2024)
[75]Barry, M., Caillaut, G., Halftermeyer, P., Qader, R., Mouayad, M., Cariolaro,
D., Le Deit, F., Gesnouin, J.: Graphrag: Leveraging graph-based efficiency to
minimize hallucinations in llm-driven rag for finance data. In: 31st International
Conference on Computational Linguistics Workshop Knowledge Graph & GenAI
(2025)
[76]Li, L., Gan, Z., Cheng, Y., Liu, J.: Relation-aware graph attention network
for visual question answering. In: Proceedings of the IEEE/CVF International
Conference on Computer Vision, pp. 10313–10322 (2019)
[77]Bhatt, U., Weller, A., Moura, J.M.: Evaluating and aggregating feature-based
model explanations. arXiv preprint arXiv:2005.00631 (2020)
[78]Li, Y., Li, W., Nie, L.: Dynamic graph reasoning for conversational open-domain
question answering. ACM Transactions on Information Systems (TOIS) 40(4),
1–24 (2022)
[79]Sun, H., Bedrax-Weiss, T., Cohen, W.W.: Pullnet: Open domain question
answering with iterative retrieval on knowledge bases and text. arXiv preprint
39

arXiv:1904.09537 (2019)
[80]Sudhahar, S., Roberts, I., Pierleoni, A.: Reasoning over paths via knowledge base
completion. arXiv preprint arXiv:1911.00492 (2019)
[81]Yasunaga, M., Ren, H., Bosselut, A., Liang, P., Leskovec, J.: Qa-gnn: Reasoning
with language models and knowledge graphs for question answering. arXiv
preprint arXiv:2104.06378 (2021)
[82]Mikolov, T., Chen, K., Corrado, G., Dean, J.: Efficient estimation of word
representations in vector space. arXiv preprint arXiv:1301.3781 (2013)
[83]Serrano, S., Smith, N.A.: Is attention interpretable? arXiv preprint
arXiv:1906.03731 (2019)
[84]Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.: Grad-
cam: Visual explanations from deep networks via gradient-based localization.
In: Proceedings of the IEEE International Conference on Computer Vision, pp.
618–626 (2017)
[85] Ying, Z., Bourgeois, D., You, J., Zitnik, M., Leskovec, J.: Gnnexplainer: Gener-
ating explanations for graph neural networks. Advances in neural information
processing systems 32(2019)
[86]Yan, X.: PrimeKGQA: A comprehensive biomedical knowledge graph question
answering dataset. Zenodo (2024). https://doi.org/10.5281/zenodo.13829395 .
https://doi.org/10.5281/zenodo.13829395
[87]Paulheim, H.: Knowledge graph refinement: A survey of approaches and evaluation
methods. Semantic web 8(3), 489–508 (2016)
[88]Nickel, M., Murphy, K., Tresp, V., Gabrilovich, E.: A review of relational machine
learning for knowledge graphs. Proceedings of the IEEE 104(1), 11–33 (2015)
[89]Lin, Y., Liu, Z., Sun, M., Liu, Y., Zhu, X.: Learning entity and relation embeddings
for knowledge graph completion. In: Proceedings of the AAAI Conference on
Artificial Intelligence, vol. 29 (2015)
[90]Xiong, W., Hoang, T., Wang, W.Y.: Deeppath: A reinforcement learning method
for knowledge graph reasoning. arXiv preprint arXiv:1707.06690 (2017)
[91]Saxena, A., Tripathi, A., Talukdar, P.: Improving multi-hop question answering
over knowledge graphs using knowledge base embeddings. In: Proceedings of
the 58th Annual Meeting of the Association for Computational Linguistics, pp.
4498–4507 (2020)
[92]Hu, M., Peng, Y., Huang, Z., Li, D.: Retrieve, read, rerank: Towards end-to-end
multi-document reading comprehension. arXiv preprint arXiv:1906.04618 (2019)
40

[93]Geiger, A., Ibeling, D., Zur, A., Chaudhary, M., Chauhan, S., Huang, J., Arora,
A., Wu, Z., Goodman, N., Potts, C., et al. : Causal abstraction: A theoretical
foundation for mechanistic interpretability. Journal of Machine Learning Research
26(83), 1–64 (2025)
[94]Zhang, Q., Chen, S., Bei, Y., Yuan, Z., Zhou, H., Hong, Z., Dong, J., Chen, H.,
Chang, Y., Huang, X.: A survey of graph retrieval-augmented generation for
customized large language models. arXiv preprint arXiv:2501.13958 (2025)
[95]Cuturi, M.: Sinkhorn distances: Lightspeed computation of optimal transport.
Advances in neural information processing systems 26(2013)
[96]Tan, J., Geng, S., Fu, Z., Ge, Y., Xu, S., Li, Y., Zhang, Y.: Learning and evaluating
graph neural network explanations based on counterfactual and factual reasoning.
In: Proceedings of the ACM Web Conference 2022, pp. 1018–1027 (2022)
[97]Hastie, T.: The elements of statistical learning: data mining, inference, and
prediction. Springer (2009)
[98]Freedman, D.A.: Statistical Models: Theory and Practice. cambridge university
press, ??? (2009)
[99]Sanchez-Lengeling, B., Wei, J., Lee, B., Reif, E., Wang, P., Qian, W., McCloskey,
K., Colwell, L., Wiltschko, A.: Evaluating attribution for graph neural networks.
Advances in neural information processing systems 33, 5898–5910 (2020)
[100] Slack, D., Hilgard, A., Singh, S., Lakkaraju, H.: Reliable post hoc explanations:
Modeling uncertainty in explainability. Advances in neural information processing
systems 34, 9391–9404 (2021)
[101] Huang, P.-S., He, X., Gao, J., Deng, L., Acero, A., Heck, L.: Learning deep
structured semantic models for web search using clickthrough data. In: Proceed-
ings of the 22nd ACM International Conference on Information & Knowledge
Management, pp. 2333–2338 (2013)
[102] Lin, S., Hilton, J., Evans, O.: Teaching models to express their uncertainty in
words. arXiv preprint arXiv:2205.14334 (2022)
[103] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. :
Language models are unsupervised multitask learners. OpenAI blog 1(8), 9
(2019)
[104] Holtzman, A., Buys, J., Du, L., Forbes, M., Choi, Y.: The curious case of neural
text degeneration. arXiv preprint arXiv:1904.09751 (2019)
[105] Lyu, Q., Apidianaki, M., Callison-Burch, C.: Towards faithful model explanation
in nlp: A survey. Computational Linguistics 50(2), 657–723 (2024)
41

[106] Agarwal, C., Tanneru, S.H., Lakkaraju, H.: Faithfulness vs. plausibility: On
the (un) reliability of explanations from large language models. arXiv preprint
arXiv:2402.04614 (2024)
[107] Jacovi, A., Goldberg, Y.: Towards faithfully interpretable nlp systems: How
should we define and evaluate faithfulness? arXiv preprint arXiv:2004.03685
(2020)
[108] Carvalho, D.V., Pereira, E.M., Cardoso, J.S.: Machine learning interpretability:
A survey on methods and metrics. Electronics 8(8), 832 (2019)
[109] Burger, C., Walter, C., Le, T., Chen, L.: Towards robust and accurate stability
estimation of local surrogate models in text-based explainable ai. arXiv preprint
arXiv:2501.02042 (2025)
[110] Jiang, X., Havaei, M., Varno, F., Chartrand, G., Chapados, N., Matwin, S.: Learn-
ing to learn with conditional class dependencies. In: International Conference on
Learning Representations (2019)
[111] Peyr´ e, G., Cuturi, M., et al. : Computational optimal transport: With applications
to data science. Foundations and Trends ®in Machine Learning 11(5-6), 355–607
(2019)
[112] Zhao, X., Huang, W., Huang, X., Robu, V., Flynn, D.: Baylime: Bayesian
local interpretable model-agnostic explanations. In: Uncertainty in Artificial
Intelligence, pp. 887–896 (2021). PMLR
[113] Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y., Narasimhan, K.: Tree
of thoughts: Deliberate problem solving with large language models. Advances
in neural information processing systems 36, 11809–11822 (2023)
[114] Zhou, D., Sch¨ arli, N., Hou, L., Wei, J., Scales, N., Wang, X., Schuurmans, D.,
Cui, C., Bousquet, O., Le, Q., et al.: Least-to-most prompting enables complex
reasoning in large language models. arXiv preprint arXiv:2205.10625 (2022)
[115] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou,
D.,et al. : Chain-of-thought prompting elicits reasoning in large language models.
Advances in neural information processing systems 35, 24824–24837 (2022)
[116] Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery,
A., Zhou, D.: Self-consistency improves chain of thought reasoning in language
models. arXiv preprint arXiv:2203.11171 (2022)
[117] Zelikman, E., Wu, Y., Mu, J., Goodman, N.: Star: Bootstrapping reasoning with
reasoning. Advances in Neural Information Processing Systems 35, 15476–15488
(2022)
42

[118] Press, O., Zhang, M., Min, S., Schmidt, L., Smith, N.A., Lewis, M.: Measuring
and narrowing the compositionality gap in language models. arXiv preprint
arXiv:2210.03350 (2022)
[119] Dong, L., Mallinson, J., Reddy, S., Lapata, M.: Learning to paraphrase for
question answering. arXiv preprint arXiv:1708.06022 (2017)
[120] Anantha, R., Vakulenko, S., Tu, Z., Longpre, S., Pulman, S., Chappidi, S.: Open-
domain question answering goes conversational via question rewriting. arXiv
preprint arXiv:2010.04898 (2020)
[121] Rajpurkar, P., Jia, R., Liang, P.: Know what you don’t know: Unanswerable
questions for squad. arXiv preprint arXiv:1806.03822 (2018)
[122] Liang, Z., Bethard, S., Surdeanu, M.: Explainable multi-hop verbal reasoning
through internal monologue. In: Proceedings of the 2021 Conference of the North
American Chapter of the Association for Computational Linguistics: Human
Language Technologies, pp. 1225–1250 (2021)
[123] Fu, B., Qiu, Y., Tang, C., Li, Y., Yu, H., Sun, J.: A survey on complex question
answering over knowledge base: Recent advances and challenges. arXiv preprint
arXiv:2007.13069 (2020)
[124] Lan, Y., He, G., Jiang, J., Jiang, J., Zhao, W.X., Wen, J.-R.: A survey on complex
knowledge base question answering: Methods, challenges and solutions. arXiv
preprint arXiv:2105.11644 (2021)
[125] Lan, Y., He, G., Jiang, J., Jiang, J., Zhao, W.X., Wen, J.-R.: Complex knowledge
base question answering: A survey. IEEE Transactions on Knowledge and Data
Engineering 35(11), 11196–11215 (2022)
[126] Luo, H., Tang, Z., Peng, S., Guo, Y., Zhang, W., Ma, C., Dong, G., Song, M., Lin,
W., Zhu, Y., et al.: Chatkbqa: A generate-then-retrieve framework for knowledge
base question answering with fine-tuned large language models. arXiv preprint
arXiv:2310.08975 (2023)
[127] Yani, M., Krisnadhi, A.A.: Challenges, techniques, and trends of simple knowledge
graph question answering: a survey. Information 12(7), 271 (2021)
[128] Wei, Y., Wang, X., Nie, L., He, X., Hong, R., Chua, T.-S.: Mmgcn: Multi-modal
graph convolution network for personalized recommendation of micro-video.
In: Proceedings of the 27th ACM International Conference on Multimedia, pp.
1437–1445 (2019)
[129] Kashyap, S., et al.: Knowledge graph assisted large language models (2024)
[130] Arslan, M., Cruz, C.: Business-rag: Information extraction for business insights.
43

In: 21st International Conference on Smart Business Technologies, pp. 88–94
(2024). SCITEPRESS-Science and Technology Publications
[131] Kim, J., Hur, M., Min, M.: From rag to qa-rag: Integrating generative ai for
pharmaceutical regulatory compliance process. In: Proceedings of the 40th
ACM/SIGAPP Symposium on Applied Computing, pp. 1293–1295 (2025)
[132] Srivastava, S., Jain, M.D., Jain, H., Jaroli, K., Patel, V.M., Khan, L.: Iot mon-
itoring bin for smart cities. In: 3rd Smart Cities Symposium (SCS 2020), vol.
2020, pp. 533–536 (2020). IET
44
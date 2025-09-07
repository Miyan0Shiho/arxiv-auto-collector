# Towards Open-World Retrieval-Augmented Generation on Knowledge Graph: A Multi-Agent Collaboration Framework

**Authors**: Jiasheng Xu, Mingda Li, Yongqiang Tang, Peijie Wang, Wensheng Zhang

**Published**: 2025-09-01 08:26:12

**PDF URL**: [http://arxiv.org/pdf/2509.01238v1](http://arxiv.org/pdf/2509.01238v1)

## Abstract
Large Language Models (LLMs) have demonstrated strong capabilities in
language understanding and reasoning. However, their dependence on static
training corpora makes them prone to factual errors and knowledge gaps.
Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating
external knowledge sources, especially structured Knowledge Graphs (KGs), which
provide explicit semantics and efficient retrieval. Existing KG-based RAG
approaches, however, generally assume that anchor entities are accessible to
initiate graph traversal, which limits their robustness in open world settings
where accurate linking between the query and the entity is unreliable. To
overcome this limitation, we propose AnchorRAG, a novel multi-agent
collaboration framework for open-world RAG without the predefined anchor
entities. Specifically, a predictor agent dynamically identifies candidate
anchor entities by aligning user query terms with KG nodes and initializes
independent retriever agents to conduct parallel multi-hop explorations from
each candidate. Then a supervisor agent formulates the iterative retrieval
strategy for these retriever agents and synthesizes the resulting knowledge
paths to generate the final answer. This multi-agent collaboration framework
improves retrieval robustness and mitigates the impact of ambiguous or
erroneous anchors. Extensive experiments on four public benchmarks demonstrate
that AnchorRAG significantly outperforms existing baselines and establishes new
state-of-the-art results on the real-world question answering tasks.

## Full Text


<!-- PDF content starts -->

Towards Open-World Retrieval-Augmented Generation on Knowledge Graph: A
Multi-Agent Collaboration Framework
Jiasheng Xu1*, Mingda Li1*, Yongqiang Tang1†, Peijie Wang1, Wensheng Zhang1
1Institute of Automation, Chinese Academy of Sciences
Abstract
Large Language Models (LLMs) have demonstrated strong
capabilities in language understanding and reasoning. How-
ever, their dependence on static training corpora makes
them prone to factual errors and knowledge gaps. Retrieval-
Augmented Generation (RAG) addresses this limitation by
incorporating external knowledge sources, especially struc-
tured Knowledge Graphs (KGs), which provide explicit se-
mantics and efficient retrieval. Existing KG-based RAG ap-
proaches, however, generally assume that anchor entities are
accessible to initiate graph traversal, which limits their ro-
bustness in open world settings where accurate linking be-
tween the query and the entity is unreliable. To overcome this
limitation, we propose AnchorRAG , a novel multi-agent col-
laboration framework for open-world RAG without the pre-
defined anchor entities. Specifically, a predictor agent dynam-
ically identifies candidate anchor entities by aligning user
query terms with KG nodes and initializes independent re-
triever agents to conduct parallel multi-hop explorations from
each candidate. Then a supervisor agent formulates the it-
erative retrieval strategy for these retriever agents and syn-
thesizes the resulting knowledge paths to generate the final
answer. This multi-agent collaboration framework improves
retrieval robustness and mitigates the impact of ambiguous
or erroneous anchors. Extensive experiments on four public
benchmarks demonstrate that AnchorRAG significantly out-
performs existing baselines and establishes new state-of-the-
art results on the real-world question answering tasks.
1 Introduction
Large Language Models (LLMs) (Achiam et al. 2023; Yang
et al. 2025; Liu et al. 2024) are typically defined as deep
learning models with a massive number of parameters,
trained on large-scale corpora in a self-supervised manner.
Their internal parameterization allows for an implicit repre-
sentation of external knowledge. During the inference stage,
the Chain-of-Thought (CoT) (Wei et al. 2022; Trivedi et al.
2022; Zhou et al. 2022) method guides the models to “think
step-by-step” through carefully designed prompts. It helps
the models better handle logically complex or multi-step
reasoning tasks, achieving a better performance in natural
language processing tasks such as question answering (Gu
*These authors contributed equally.
†Corresponding author
Figure 1: Illustration of the difference between the closed-
world and the open-world setting in RAG.
et al. 2021). Despite these advances, LLMs are fundamen-
tally limited by the incompleteness and inaccuracies in their
static training corpora. This often results in factual hal-
lucinations (Huang et al. 2025), especially in knowledge-
intensive tasks, significantly hindering their reliability and
deployment in real-world applications.
To address these limitations, recent studies (Jeong et al.
2024; Xia et al. 2025) have investigated augmenting LLMs
with external knowledge sources to improve factual con-
sistency and reasoning. While fine-tuning (Hu et al. 2022;
Li and Liang 2021; Li et al. 2023b), LLM with the ex-
ternal knowledge is computationally expensive and inflex-
ible, a promising solution is Retrieval-Augmented Genera-
tion (RAG) (Lewis et al. 2020; Jiang et al. 2023b; Sarthi
et al. 2024), which combines real-time retrieval from exter-
nal sources with generative modeling. Among exiting meth-
ods, RAG on the Knowledge Graph (KG) (Jiang et al. 2023a;
Ma et al. 2025; He et al. 2024; Chen et al. 2024; Guo et al.
2024) has become a research hotspot in the field, due to thearXiv:2509.01238v1  [cs.AI]  1 Sep 2025

fact that KGs can provide structured knowledge and explicit
retrieval paths which benefit to enhancing the reasoning ca-
pabilities of LLMs. Typically, these KG-based RAG meth-
ods (Wang et al. 2025; Sun et al. 2023; Ma et al. 2024) can
be divided in two steps: (1) identifying a query-aware anchor
entity to initiate retrieval; (2) performing iterative retrieval
to construct the reasoning paths. Through above steps, these
methods can effectively exploit the external knowledge re-
lated to the query and improve the performance in the ques-
tion answering task.
However, existing KG-based RAG methods almost follow
the closed-world setting where the anchor entity is accessi-
ble and exists in the given KG. As shown in the Fig. 1 (a), the
anchor entity “Mitt Romney” is pre-defined for the question
“What university did Romney graduated from?”. The RAG
methods can directly locate the target entity for the follow-
ing retrieval. Actually, user questions usually conform to the
open-world setting where the anchor entity is unavailable. In
that case, existing methods extract the keywords in the query
and identify the candidate anchor entity by semantic match-
ing. Due to name abbreviations and aliases, these methods
usually suffer from the issue caused by imprecise or partial
matching as shown in Fig. 1(b). Then the challenge of the
open-world RAG lies in how to accurately identify anchor
entities to retrieve the relevant information.
To tackle this issue, we propose AnchorRAG , a multi-
agent collaborative (MAC) framework that can perform
RAG without the predefined anchor entities, thereby en-
abling more effective knowledge retrieval to enhance the
LLM reasoning performance. Specifically, our framework
follows a pipeline-based MAC framework, and a predictor
agent first extracts the keywords from the given question
and applies a semantic match model to obtain the candidate
entities by their name description and structure neighbor-
hood. Then, multiple retriever agents initiate parallel graph
traversal with each candidate, conducting iterative retrieval
to obtain the pivotal knowledge for the following reasoning.
Finally, a supervisor agent will synthesize the retrieved ev-
idence and determine whether an answer can be generated
and formulate the retrieval process. The main contributions
of this paper are summarized as follows:
•General Aspect. We emphasize the importance of identi-
fying the accurate anchor entity for the open-world RAG.
By integrating multi-agent collaboration into the work-
flow of question answering, we can enhance the general-
ity of RAG methods and improve the performance in the
open-world setting.
•Methodologies. For anchor identifying, we design an en-
tity grounding strategy to locate the candidate anchors by
entity name description and structural neighborhood. For
knowledge retrieval, we propose a novel retrieval method
with rough pruning and fine filtering, which can capture
the resultful knowledge paths to boost the LLM reason-
ing.
•Experimental Findings. Extensive experiments demon-
strate that AnchorRAG consistently outperforms existing
baselines on four public QA datasets, establishing state-
of-the-art results on the real-world question answeringtasks.
2 Related Work
While advanced prompting methods like Chain-of-Thought
(Wei et al. 2022) enhance LLM reasoning, their reliance
on static internal knowledge can lead to factual errors and
hallucinations that accumulate over multi-step inference.
Retrieval-Augmented Generation (RAG) (Lewis et al. 2020;
Semnani et al. 2023; He, Zhang, and Roth 2022) miti-
gates this by leveraging external knowledge, significantly
improving performance on knowledge-intensive tasks like
Knowledge Graph Question Answering (KGQA) (Yih et al.
2016), where the knowledge graphs (KGs) (Bollacker et al.
2008; Auer et al. 2007; Vrande ˇci´c and Kr ¨otzsch 2014) pro-
vide large-scale structured facts and rich semantic rela-
tionships. These KG-based RAG methods can be divided
into two main groups: KG-augmented Fine-tuning andKG-
augmented In-context Learning .
2.1 KG-augmented Fine-tuning
KG-augmented fine-tuning directly integrates structured
knowledge into open-source LLMs during training, oper-
ating at the entity, path, or subgraph levels (Zhang et al.
2025; Pan et al. 2024). At the entity level, methods like
ChatDoctor (Li et al. 2023a) and PMC-LLaMA (Wu et al.
2024) employ instruction tuning (Wang et al. 2022) to
enhance domain-specific knowledge understanding before
fine-tuning on downstream tasks that contain vast entities
and text descriptions. For path-level fine-tuning, RoG (Luo
et al. 2023b) trains the LLM to generate valid relation paths,
thereby aligning the reasoning process with the graph’s ac-
tual structure. The sources of the knowledge paths is diverse,
ranging from the direct routes between question-related en-
tities to complex logical rules extracted by graph retrieval
models (Mavromatis and Karypis 2024) or heuristic strate-
gies (Tan et al. 2024). To capture more comprehensive struc-
tural information, subgraph-level methods fine-tune LLMs
on the entire local neighborhood, compressing them into em-
beddings (He et al. 2024) or transforming them into text se-
quences (Luo et al. 2023a).
2.2 KG-augmented In-context Learning
For state-of-the-art closed-source LLMs where fine-tuning
is infeasible (Zhang et al. 2025), KG-augmented In-context
Learning (ICL) is the primary strategy (Huang et al. 2023).
Early methods like ToG (Sun et al. 2023) treat the LLM as
an agent that executes sequential reasoning along the re-
trieved relational path. Given that this methods are prone
to error accumulation and inefficient search, subsequent re-
search has developed more advanced reasoning frameworks.
To enhance reliability, Plan-on-Graph (Chen et al. 2024)
introduced adaptive planning and self-correction mecha-
nism, while Debate-on-Graph (Ma et al. 2025) proposed
multi-path argumentation. Besides, recent methods, e.g.,
Fast Think-on-Graph (Liang and Gu 2025), focused on
optimizing search efficiency, they accelerates the reason-
ing process and improves accuracy by enabling LLMs

Figure 2: An overview of the AnchorRAG framework, including the collaborative workflows of multi-agent exploration for
question answering.
to “think community-by-community” within the knowl-
edge graph. To address knowledge graph incompleteness,
ReKnoS (Wang et al. 2025) simplify reasoning by uti-
lizing “super-relations”, while others, e.g., Generate-on-
Graph (Xu et al. 2024), dynamically generate facts when
external knowledge is unavailable. Nevertheless, almost
of the existing methods are implemented for the closed-
world question answering, which limits their performance
in the real-world settings where accurate linking between
the query and the entity is unreliable.
3 Method
3.1 Task Formulation
The fundamental unit of data in a knowledge graph Gis
the triples, i.e., G={(h, r, t )|h, t∈ E, r∈ R} , where
EandRrespectively represent the set of entities and re-
lations.The goal of Knowledge Graph Question Answering
(KGQA) is to answer natural language questions by lever-
aging the structured triples within a knowledge graph. An
unavoidable challenge for KGQA is how to design a syn-
ergy between the semantic understanding of LLMs and the
structured facts of a knowledge graph, enabling both precise
knowledge retrieval and powerful reasoning for the QA task.
3.2 Overview
As shown in the Fig. 2, our framework follows a pipeline-
based MAC framework: anchor identifying by the predic-
tor agent, knowledge searching by the retriever agents and
question answering by the supervisor agent. Specifically, the
proposed predictor agent first extracts the keywords from the
given question and applies a semantic match model to obtainthe candidate entities by their name description and struc-
tural neighborhood. Then, multiple retriever agents initiate
parallel graph traversal for each candidate, performing iter-
ative retrieval and obtaining the pivotal knowledge for the
following reasoning. Furthermore, at each retrieval round, a
supervisor agent aggregates the retrieved evidence and deter-
mines whether an answer can be generated and the retrieval
process should be continued.
3.3 Anchor Entity Identifying
In our framework, the process of accurately identifying an-
chor entities from natural language questions is handled by
the predictor agent. This process consists of the following
two steps:
Candidate Entity Generation: The predictor agent is as-
sociated with a LLM to recognize and extract keywords
from the input question q, forming a keyword set D=
{d1, d2, . . . , d n}. For example, given the question q: ”What
country bordering France contains an airport that serves
Nijmegen?”, the predictor locates the keyword set D, i.e.,
D={France ,Nijmegen }. In addition, considering there
may exist certain spelling errors in the open-world ques-
tions, then the predictor will proofread the given question
to filter out irrelevant distractors and correct potential errors,
ensuring that the extracted keywords are clean and meaning-
ful. Subsequently, it adopts a semantic match between these
keywords and the entities set Eof the external KG, generat-
ing an initial list of candidate anchors ˜Ethrough in-context
learning. The prompt and in-context examples can be found
in the Appendix.

Relation-Aware Entity Grounding: The initial candidate
set˜Eoften contains numerous entities with similar names
that are irrelevant to the question’s context. Taking all of
them into account will increase the overhead of subsequent
retrieval and inevitably introduce noise. In this section, we
argue that an entity is a more probable anchor if the semantic
implied in its neighborhood is aligned with the given ques-
tion. Therefore, the predictor agent presents a relation-aware
grounding strategy to precisely identify the most relevant
candidate entities. For each candidate entity e∈˜E, the agent
gathers its one-hop neighboring relations and obtains the set
R(e). Then it invokes a pre-trained model SBERT (Reimers
and Gurevych 2019) to encode the neighboring relations and
the question, obtaining the relevance score s(ri, q)between
each relation ri∈ R(e)and the given question qas follows:
s(ri, q) =⟨SBERT( ri),SBERT( q)⟩, (1)
where ⟨⟩denotes the inner product. Based on the above
equation, we define the final relevance score of the candi-
date entity by aggregating the semantic information of its
neighboring relations as follows:
Score( e, q) =1
kX
r∈N(e)s(r, q), (2)
where N(e)denotes the set of top- krelevant relations ac-
cording to the Eq.(1). By scoring and ranking all candidates
in the set ˜E, the predictor agent selects the top- mentities
with the higher relevance scores (e.g., France and Nijmegen)
as the final anchor candidate set, which will be inputted to
the following retriever agents for parallel searching.
3.4 Rough Pruning and Fine Filtering
Given the top- mcandidate anchor entities obtained from
predictor agent, we assign each entity to an independent re-
triever agent. Thus there exist mretrieval agents operating
in parallel, each conducting iterative retrieval and reason-
ing based on its assigned anchor entity. At each iteration,
these agents will retrieve the one-hop neighborhood of the
given entity, where the neighborhood includes the neighbor-
ing relations and entities. Then a rough pruning and fine fil-
tering method is applied to select more meaningful neigh-
borhood for the question. Finally, the neighboring entities
will be considered as the initial ones for the next retrieval
iteration if necessary.
Neighboring Relations Pruning: At the t-th retrieval it-
eration, each retriever agent explores candidate relations
from its current given entities et
i. This agent retrieves all
one-hop incoming and outgoing relations of the given en-
tity from the knowledge graph G. To reduce noise, the
agent will filter out trivial or schema-level relations (e.g.,
type.object.name ,common. *,sameAs ), as well as
previously visited relations. The remaining relations are
scored by the LLM according to their relevance score with
the given question. Specifically, each agent constructs a
prompt containing the question, current anchor entity, and
candidate relations, applying the LLM to assign the nor-
malized scores. The prompt and in-context examples can befound in the Appendix. With the neighboring relations prun-
ing, the top- brelations are selected.
Neighboring Entities Pruning: Given the selected rela-
tions, each agent retrieves the neighboring entities linked
via these relations. Considering that each relation will have
many neighboring entities, to bound the length of LLM in-
put, we limit the number of entities per relation. Specifically,
each candidate entity is ranked by the LLM with the prompt
including the question, current anchor entity and the selected
relation. The prompt and in-context examples can be found
in the Appendix. Then top- bentities are obtained for each
relation, forming the knowledge triple set T(et
i).
Candidate Triples Filtering: To further identify the
question-related facts implied in the knowledge graph, each
agent applies a fine filtering method on the candidate triples
T(et
i), improving the performance in the real-world ques-
tion answering. Specifically, the retrieval agent first inte-
grates all candidate triples in T(et
i)into a structured list
and prompts a LLM to select the related triples according
to the given question. The prompt and in-context examples
can be found in the Appendix. Then the filtered triples are
adopted as high-quality external knowledge which will be
inputted to the following supervisor agent for answer gener-
ation. Furthermore, if the filtered triple set is empty, it means
that the current exploration path can no longer yield related
information for the given question. In that case, this agent
will terminate its subsequent retrieval and provide no in-
formation for the supervisor at the current iteration. This
“early-stopping” mechanism ensures the agent can adap-
tively waive unpromising knowledge paths, reducing the
computational overhead for efficient retrieval-augmented
generation.
3.5 Answer Generating
At the end of each iteration, all active retrieval agents in-
put their resulting paths to a supervisor agent. The supervi-
sor aggregates these paths and prompts the LLM to assess
whether the current information is sufficient to answer the
question. If so, it stops the reasoning process and returns the
final answer. If not, only agents with valid paths are per-
mitted to perform the next retrieval round. The prompt and
in-context examples can be found in the Appendix. To avoid
excessive computation overhead, the supervisor adopts the
early termination if (1) no agents remain active, or (2) the
maximum reasoning depth Lis reached. In such cases, the
supervisor invokes LLM to directly answer the question by
applying CoT reasoning with its internal knowledge.
4 Experiments
4.1 Dataset and Evaluation
In the experiments, we adopt four benchmark datasets to
evaluate the performance of our framework, including three
multi-hop Knowledge Graph Question Answering (Multi-
Hop KGQA) datasets: WebQuestionSP (WebQSP) (Yih
et al. 2016), GrailQA (Gu et al. 2021) and Complex We-
bQuestions (CWQ) (Talmor and Berant 2018), and one for
Open-Domain Question Answering: WebQuestions (Berant

DatasetWebQSP GrailQA CWQ WebQuestions
Hit@1 acc Hit@1 acc Hit@1 acc Hit@1 acc
Method Qwen-Plus
IO 63.3 42.2 33.2 26.6 33.2 33.2 56.3 43.2
Chain-of-Thought 62.9 41.5 32.3 26.7 37.6 37.6 54.1 41.2
Self-Consistency 63.0 41.2 33.8 28.4 39.2 39.2 52.7 40.1
PoG 33.1 25.4 36.9 33.0 39.5 29.2 25.3 21.5
ToG 66.1 46.3 41.9 35.3 39.8 39.8 56.1 43.7
Ours 73.3 56.4 62.7 56.0 47.0 47.0 60.9 50.5
Method GPT-4o-mini
IO 65.8 46.1 35.6 27.9 35.7 35.7 57.7 45.4
Chain-of-Thought 62.5 41.6 31.0 25.8 34.5 34.5 53.6 41.2
Self-Consistency 59.2 40.1 34.0 27.1 36.1 36.1 50.7 39.3
PoG 55.3 36.4 39.0 34.2 36.1 36.1 41.8 33.3
ToG 71.4 49.9 43.7 36.3 40.7 40.7 61.4 47.6
Ours 74.1 57.4 63.4 56.8 44.8 44.8 60.0 50.0
Table 1: The results of different methods on various Datasets, using Qwen-Plus and GPT-4o-mini as the LLM. The best results
are highlighted in bold. The suboptimal results are underlined.
et al. 2013). Freebase (Bollacker et al. 2008) is utilized as
the external Knowledge Graph, which contains the complete
knowledge facts that are necessary to answer questions in
the above datasets. In addition, we apply the exact match
(i.e., Hit@1) and accuracy (i.e., acc) as evaluation metrics
for our method which are widely used in the field of ques-
tion answering with RAG.
4.2 Implementation Settings
We extract all entities from Freebase, encode their name de-
scription into vectors using SBERT model (all-MiniLM-L6-
v2, without fine-tuning) (Reimers and Gurevych 2019), and
subsequently index them in a vector database (Douze et al.
2024) for downstream entity querying and matching. The
knowledge graph derived from Freebase is deployed on a lo-
cally deployed Virtuoso (Erling and Mikhailov 2009) server
for our experiments. We selected GPT-4o-mini and Qwen-
Plus as the fundamental LLM in our framework. To prevent
redundant exploration by agents and reduce computational
overhead, we set both the maximum search depth Land the
expansion width bto 3. This configuration aligns with the
empirically optimal values identified in (Sun et al. 2023).
Additionally, we fix the number of neighbor relations kdur-
ing anchor entity grounding to 5, and the number of retrieval
agents mis set to 3 for parallel multi-hop reasoning. The
data and source can be found in appendix and will be avail-
able soon.
4.3 Baselines
Some widely used prompting strategies are selected as the
baselines, including vanilla IO prompting (Brown et al.2020), Chain-of-Thought (CoT) (Wei et al. 2022), and Self-
Consistency (Wang et al. 2022). Moreover, we also compare
the proposal with two representative RAG methods: Think-
on-graph (Sun et al. 2023) and Plan-on-graph (Chen et al.
2024). It is worth noting that the above two methods are
both implemented in the closed-world setting where the an-
chor entities are predefined for each question. Thus, to make
a fair comparison, we apply the fundamental LLM to select
the anchor entities instead of directly identifying them for
the open-world question answering. Other settings and pa-
rameters remain unchanged.
4.4 Main Results
The main experimental results, as shown in Table 1, indicate
that our proposed method achieves a significant performance
advantage over all baseline methods when deployed with
both the Qwen-Plus and GPT-4o-mini LLMs. Specifically,
when applying Qwen-Plus, AnchorRAG achieves an aver-
age improvement by 10% and 11.2% on the metrics Hit@1
and Accuracy respectively, compared to the strongest base-
line. Besides, with GPT-4o-mini model, our proposal also
achieves the increases of 6.3% and 8.6%. We can also find
that our method obtain an outstanding performance on the
complex datasets like GrailQA, where its Hit@1 and Accu-
racy scores are increased substantially by 20.8% and 20.7%,
respectively. The reason may be that the questions and the
kG entities are semantically inconsistent in this dataset, and
the anchor entities are more difficult to locate in this case.
AnchorRAG can handle this issue by the multi-agent collab-
oration, thereby obtaining the state-of-the-art results. In ad-
dition, the performance of the RAG methods, e.g., ToG, con-
sistently surpasses the methods solely relying on the internal

knowledge, e.g., CoT. This demonstrates the effectiveness of
the RAG methods in the question answering task, especially
when dealing with the complex reasoning such as GrailQA
and CWQ.
DatasetWebQSP GrailQA
normal open-world normal open-world
Method Qwen-Plus
ToG 66.1 58.5 ↓11.5%41.9 32.1 ↓23.4%
AnchorRAG 73.3 71.3 ↓2.7%62.7 55.4 ↓11.6%
Method GPT-4o-mini
ToG 71.4 67.1 ↓6.0%43.7 35.5 ↓18.8%
AnchorRAG 74.1 70.3 ↓5.1%63.4 55.4 ↓12.6%
Table 2: Performance comparisons (Hits@1) of different
methods on the normal and open-world versions of WebQSP
and GrailQA datasets, using Qwen-Plus and GPT-4o-mini as
the backbone LLMs.
Method WebQSP GrailQA
AnchorRAG 74.1 63.4
w/o Entity Grounding 67.8 42.8
w/o Parallel Retrieval 72.8 60.2
w/o Triples Fine Filtering 71.6 62.8
Table 3: Ablation study of AnchorRAG: Effect of different
components on the Hits@1 performance.
4.5 Robust Analyses
Given that existing public QA datasets primarily assume
that the external KG contains all the necessary informa-
tion for answering the given questions, which is not suitable
for the open-world question answering task. Thus, to fur-
ther evaluate the generality and robustness of our method,
we constructed the open-world versions of the WebQSP and
GrailQA datasets. Specifically, we first added the semantic
noise by introducing some typos into the original questions
to prevent exact matches with entities in the KG. Then we
incorporated additional questions into the datasets whose an-
swers could not be directly retrieved from the KG. The de-
tailed construction process is provided in the appendix. The
results on the self-constructed datasets are presented in Ta-
ble 2, we can observe that our method shows strong robust-
ness on this open-world scenario. For instance, on GrailQA,
the performance of ToG using Qwen-Plus drops by 23.4%,
whereas our method only drops by 11.6%. These results
demonstrate that AnchorRAG can effectively address the
open-world QA tasks and further enhance the generality of
the RAG methods.
Figure 3: The effect of the anchor identifying and knowledge
retrieval on WebQSP.
4.6 Ablation Studies
To validate the effectiveness of the core components within
our proposed AnchorRAG framework, we conducted the
following ablation studies on the WebQSP and GrailQA
datasets, with the results presented in Table 3. Note that ‘w/o
Entity Grounding’ removes the neighborhood information
during anchor entity identification, ‘w/o Parallel Retrieval’
eliminates the multi-agent retrieval mechanism, replacing it
with a single-path retrieval approach, and ‘w/o Triples Fine
Filtering’ refers to the absence of fine semantic filtering for
candidate triples during retrieval. The results show that each
component can contribute to the question answering perfor-
mance. Comparing with parallel retrieval and fine filtering,
the entity grounding module has a greater impact on the per-
formance, achieving the improvement by 6.3% and 20.6%
on WebQSP and GrailQA respectively, which emphasizes
the importance of precise anchor entities in the open-world
question answering tasks.
4.7 Effect of the Retrieval
To further validate the effectiveness of our method in an-
chor identifying and knowledge retrieval, we compare the
effect of retrieval of our method against two RAG baselines
(PoG and ToG) and a variant (AnchorRAG w/o Parallel Re-
trieval). The results are shown in Fig. 3. The figures in the
top row illustrate the accuracy of anchor identifying (marked
in green), where our method significantly outperforms the
baselines by grounding the entities with their textual and
neighborhood information. The bottom row presents the
proportion of the exact match for the answers, including the
answers obtained by retrieval (marked in red) and those in-
ferred by CoT reasoning (marked in orange). Notably, An-
chorRAG achieves the highest retrieval rate at 49.6%, indi-
cating that the multi-agent collaborative mechanism can ef-
fectively obtain the correct answer entities, which supports
the following reasoning process to improve the QA perfor-
mance.
4.8 Hyper-parameter Analyses
We conducted the parameter sensitivity analyses on Grail-
QA to investigate the optimal configuration for the number

Setting k= 3 k= 5 k= 7
m= 1 60.4 57.6 56.4
m= 2 63.6 63.3 61.0
m= 3 63.1 63.4 63.3
m= 4 62.6 63.1 62.4
Table 4: The results of AnchorRAG with different hyper-
parameters using GPT-4o-mini on GrailQA.
of neighborhood relations kand retrieval agents m, where
the results are shown in Table 4. We can find that retaining
excessive neighboring relations kfor each candidate entity
will lead to a performance loss, due to the fact that it may
introduce noise and neglect the key structural information.
Besides, a proper number of retriever agent mcan make
AnchorRAG achieve the optimal results, which can reflect
the effectiveness of the MAC framework. In addition, we
can also find that more retrieval agents won’t always obtain
the better results. Excessive agents will introduce uncertain
retrieval information into the collaborative process, thereby
impairing the performance.
4.9 Case Study
To intuitively demonstrate the superiority of our method in
knowledge graph retrieval, we conduct a case study that
compares our approach with the ToG baseline on a repre-
sentative complex question from the CWQ dataset: “Who
inspired F. Scott Fitzgerald, and who was the architect that
designed The Mount?”. As shown in Fig. 4, we can ob-
serve that ToG can’t locate both the anchor entities, resulting
in meaningless reasoning paths and generating an incorrect
answer. In contrast, our multi-agent framework can effec-
tively identify the anchor entities for the given questions by
the entity grounding mechanism. With assigning multi re-
triever agents to independently explore different reasoning
paths, AnchorRAG accurately locates the correct answer en-
tity “Edith Wharton”. Additionally, this case highlights the
robustness of our method. For example, although it initially
considers an incorrect entity (“The Weapon”) as an anchor
entity, it promptly detects the irrelevance of the following re-
trieval path with the rough pruning and fine filtering method,
and terminates the corresponding agent, which prevents un-
necessary computation overhead.
5 Conclusion
In this paper, we present AnchorRAG, a novel multi-agent
collaborative RAG framework for open-world Knowledge
Graph Question Answering. Unlike existing methods that
rely on predefined anchor entities, AnchorRAG overcomes
the bottleneck of unreliable entity linking by identifying
and grounding plausible anchors without prior assumptions.
Specifically, a predictor agent dynamically identifies can-
didate anchor entities by aligning the given questions with
the candidate entities and initializes independent retriever
agents to conduct parallel multi-hop explorations. Then a su-
Figure 4: Comparisons of retrieval process between a base-
line reasoning (ToG) and our method (AnchorRAG) on a
complex question. The entities marked in yellow are anchor
entities, marked in green are correct answer entities. The
ones marked in blue and striped are invalid entities.
pervisor agent formulates the iterative retrieval strategy for
these retriever agents and synthesizes the reasoning paths
to generate the final answer. Extensive experiments on four
public KGQA benchmarks demonstrate that AnchorRAG
significantly outperforms the state-of-the-art baselines, vali-
dating the effectiveness of our MAC framework, especially
when dealing with the complex multi-hop reasoning tasks.
References
Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.;
Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.;
Anadkat, S.; et al. 2023. Gpt-4 technical report. arXiv
preprint arXiv:2303.08774 .
Auer, S.; Bizer, C.; Kobilarov, G.; Lehmann, J.; Cyganiak,
R.; and Ives, Z. 2007. Dbpedia: A nucleus for a web of open
data. In international semantic web conference , 722–735.
Springer.
Berant, J.; Chou, A.; Frostig, R.; and Liang, P. 2013. Se-
mantic parsing on freebase from question-answer pairs. In
Proceedings of the 2013 conference on empirical methods in
natural language processing , 1533–1544.
Bollacker, K.; Evans, C.; Paritosh, P.; Sturge, T.; and Taylor,
J. 2008. Freebase: a collaboratively created graph database
for structuring human knowledge. In Proceedings of the
2008 ACM SIGMOD international conference on Manage-
ment of data , 1247–1250.
Brown, T.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J. D.;
Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell,
A.; et al. 2020. Language models are few-shot learners. Ad-
vances in neural information processing systems , 33: 1877–
1901.

Chen, L.; Tong, P.; Jin, Z.; Sun, Y .; Ye, J.; and Xiong, H.
2024. Plan-on-graph: Self-correcting adaptive planning of
large language model on knowledge graphs. Advances in
Neural Information Processing Systems , 37: 37665–37691.
Douze, M.; Guzhva, A.; Deng, C.; Johnson, J.; Szilvasy, G.;
Mazar ´e, P.-E.; Lomeli, M.; Hosseini, L.; and J ´egou, H. 2024.
The faiss library. arXiv preprint arXiv:2401.08281 .
Erling, O.; and Mikhailov, I. 2009. RDF Support in the Vir-
tuoso DBMS. In Networked Knowledge-Networked Media:
Integrating Knowledge Management, New Media Technolo-
gies and Semantic Systems , 7–24. Springer.
Gu, Y .; Kase, S.; Vanni, M.; Sadler, B.; Liang, P.; Yan, X.;
and Su, Y . 2021. Beyond iid: three levels of generalization
for question answering on knowledge bases. In Proceedings
of the web conference 2021 , 3477–3488.
Guo, Z.; Xia, L.; Yu, Y .; Ao, T.; and Huang, C. 2024. Ligh-
trag: Simple and fast retrieval-augmented generation. arXiv
preprint arXiv:2410.05779 .
He, H.; Zhang, H.; and Roth, D. 2022. Rethinking with
retrieval: Faithful large language model inference. arXiv
preprint arXiv:2301.00303 .
He, X.; Tian, Y .; Sun, Y .; Chawla, N.; Laurent, T.; LeCun,
Y .; Bresson, X.; and Hooi, B. 2024. G-retriever: Retrieval-
augmented generation for textual graph understanding and
question answering. Advances in Neural Information Pro-
cessing Systems , 37: 132876–132907.
Hu, E. J.; Shen, Y .; Wallis, P.; Allen-Zhu, Z.; Li, Y .; Wang,
S.; Wang, L.; Chen, W.; et al. 2022. Lora: Low-rank adapta-
tion of large language models. ICLR , 1(2): 3.
Huang, L.; Yu, W.; Ma, W.; Zhong, W.; Feng, Z.; Wang, H.;
Chen, Q.; Peng, W.; Feng, X.; Qin, B.; et al. 2025. A survey
on hallucination in large language models: Principles, tax-
onomy, challenges, and open questions. ACM Transactions
on Information Systems , 1–55.
Huang, Q.; Ren, H.; Chen, P.; Kr ˇzmanc, G.; Zeng, D.; Liang,
P. S.; and Leskovec, J. 2023. Prodigy: Enabling in-context
learning over graphs. Advances in Neural Information Pro-
cessing Systems , 36: 16302–16317.
Jeong, S.; Baek, J.; Cho, S.; Hwang, S. J.; and Park, J. C.
2024. Adaptive-rag: Learning to adapt retrieval-augmented
large language models through question complexity. arXiv
preprint arXiv:2403.14403 .
Jiang, J.; Zhou, K.; Dong, Z.; Ye, K.; Zhao, W. X.; and Wen,
J.-R. 2023a. Structgpt: A general framework for large lan-
guage model to reason over structured data. arXiv preprint
arXiv:2305.09645 .
Jiang, Z.; Xu, F. F.; Gao, L.; Sun, Z.; Liu, Q.; Dwivedi-Yu, J.;
Yang, Y .; Callan, J.; and Neubig, G. 2023b. Active retrieval
augmented generation. In Proceedings of the 2023 Confer-
ence on Empirical Methods in Natural Language Process-
ing, 7969–7992.
Joshi, M.; Choi, E.; Weld, D. S.; and Zettlemoyer, L.
2017. Triviaqa: A large scale distantly supervised chal-
lenge dataset for reading comprehension. arXiv preprint
arXiv:1705.03551 .Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,
T.; et al. 2020. Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in neural infor-
mation processing systems , 33: 9459–9474.
Li, X. L.; and Liang, P. 2021. Prefix-tuning: Optimiz-
ing continuous prompts for generation. arXiv preprint
arXiv:2101.00190 .
Li, Y .; Li, Z.; Zhang, K.; Dan, R.; Jiang, S.; and Zhang, Y .
2023a. Chatdoctor: A medical chat model fine-tuned on a
large language model meta-ai (llama) using medical domain
knowledge. Cureus .
Li, Y .; Yu, Y .; Liang, C.; He, P.; Karampatziakis, N.; Chen,
W.; and Zhao, T. 2023b. Loftq: Lora-fine-tuning-aware
quantization for large language models. arXiv preprint
arXiv:2310.08659 .
Liang, X.; and Gu, Z. 2025. Fast think-on-graph: Wider,
deeper and faster reasoning of large language model on
knowledge graph. In Proceedings of the AAAI Conference
on Artificial Intelligence , 24558–24566.
Liu, A.; Feng, B.; Xue, B.; Wang, B.; Wu, B.; Lu, C.; Zhao,
C.; Deng, C.; Zhang, C.; Ruan, C.; et al. 2024. Deepseek-v3
technical report. arXiv preprint arXiv:2412.19437 .
Luo, H.; Tang, Z.; Peng, S.; Guo, Y .; Zhang, W.; Ma,
C.; Dong, G.; Song, M.; Lin, W.; Zhu, Y .; et al. 2023a.
Chatkbqa: A generate-then-retrieve framework for knowl-
edge base question answering with fine-tuned large language
models. arXiv preprint arXiv:2310.08975 .
Luo, L.; Li, Y .-F.; Haffari, G.; and Pan, S. 2023b. Reasoning
on graphs: Faithful and interpretable large language model
reasoning. arXiv preprint arXiv:2310.01061 .
Ma, J.; Gao, Z.; Chai, Q.; Sun, W.; Wang, P.; Pei, H.; Tao, J.;
Song, L.; Liu, J.; Zhang, C.; et al. 2025. Debate on graph: a
flexible and reliable reasoning framework for large language
models. In Proceedings of the AAAI Conference on Artificial
Intelligence , 24768–24776.
Ma, S.; Xu, C.; Jiang, X.; Li, M.; Qu, H.; Yang, C.; Mao,
J.; and Guo, J. 2024. Think-on-graph 2.0: Deep and
faithful large language model reasoning with knowledge-
guided retrieval augmented generation. arXiv preprint
arXiv:2407.10805 .
Mavromatis, C.; and Karypis, G. 2024. Gnn-rag: Graph
neural retrieval for large language model reasoning. arXiv
preprint arXiv:2405.20139 .
Pan, S.; Luo, L.; Wang, Y .; Chen, C.; Wang, J.; and Wu,
X. 2024. Unifying large language models and knowledge
graphs: A roadmap. IEEE Transactions on Knowledge and
Data Engineering , 36(7): 3580–3599.
Reimers, N.; and Gurevych, I. 2019. Sentence-bert: Sen-
tence embeddings using siamese bert-networks. arXiv
preprint arXiv:1908.10084 .
Sarthi, P.; Abdullah, S.; Tuli, A.; Khanna, S.; Goldie, A.;
and Manning, C. D. 2024. Raptor: Recursive abstractive
processing for tree-organized retrieval. In The Twelfth In-
ternational Conference on Learning Representations .

Semnani, S. J.; Yao, V . Z.; Zhang, H. C.; and Lam, M. S.
2023. WikiChat: Stopping the hallucination of large lan-
guage model chatbots by few-shot grounding on Wikipedia.
arXiv preprint arXiv:2305.14292 .
Sun, J.; Xu, C.; Tang, L.; Wang, S.; Lin, C.; Gong, Y .; Ni,
L. M.; Shum, H.-Y .; and Guo, J. 2023. Think-on-graph:
Deep and responsible reasoning of large language model on
knowledge graph. arXiv preprint arXiv:2307.07697 .
Talmor, A.; and Berant, J. 2018. The web as a knowledge-
base for answering complex questions. arXiv preprint
arXiv:1803.06643 .
Tan, Y .; Lv, H.; Huang, X.; Zhang, J.; Wang, S.; and Yang,
C. 2024. Musegraph: Graph-oriented instruction tuning of
large language models for generic graph mining. arXiv
preprint arXiv:2403.04780 .
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A. 2022. Interleaving retrieval with chain-of-thought rea-
soning for knowledge-intensive multi-step questions. arXiv
preprint arXiv:2212.10509 .
Vrande ˇci´c, D.; and Kr ¨otzsch, M. 2014. Wikidata: a free
collaborative knowledgebase. Communications of the ACM ,
57(10): 78–85.
Wang, S.; Lin, J.; Guo, X.; Shun, J.; Li, J.; and Zhu, Y . 2025.
Reasoning of large language models over knowledge graphs
with super-relations. arXiv preprint arXiv:2503.22166 .
Wang, X.; Wei, J.; Schuurmans, D.; Le, Q.; Chi, E.; Narang,
S.; Chowdhery, A.; and Zhou, D. 2022. Self-consistency
improves chain of thought reasoning in language models.
arXiv preprint arXiv:2203.11171 .
Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Xia, F.;
Chi, E.; Le, Q. V .; Zhou, D.; et al. 2022. Chain-of-
thought prompting elicits reasoning in large language mod-
els.Advances in neural information processing systems , 35:
24824–24837.
Wu, C.; Lin, W.; Zhang, X.; Zhang, Y .; Xie, W.; and Wang,
Y . 2024. PMC-LLaMA: toward building open-source lan-
guage models for medicine. Journal of the American Medi-
cal Informatics Association , 31(9): 1833–1843.
Xia, Y .; Zhou, J.; Shi, Z.; Chen, J.; and Huang, H. 2025.
Improving retrieval augmented language model with self-
reasoning. In Proceedings of the AAAI conference on artifi-
cial intelligence , 25534–25542.
Xu, Y .; He, S.; Chen, J.; Wang, Z.; Song, Y .; Tong, H.; Liu,
G.; Liu, K.; and Zhao, J. 2024. Generate-on-graph: Treat
llm as both agent and kg in incomplete knowledge graph
question answering. arXiv preprint arXiv:2404.14741 .
Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.;
Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3
technical report. arXiv preprint arXiv:2505.09388 .
Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y .; Cohen, W. W.;
Salakhutdinov, R.; and Manning, C. D. 2018. HotpotQA:
A Dataset for Diverse, Explainable Multi-hop Question An-
swering. In Conference on Empirical Methods in Natural
Language Processing (EMNLP) .
Yih, W.-t.; Richardson, M.; Meek, C.; Chang, M.-W.; and
Suh, J. 2016. The value of semantic parse labeling forknowledge base question answering. In Proceedings of the
54th Annual Meeting of the Association for Computational
Linguistics (Volume 2: Short Papers) , 201–206.
Zhang, Q.; Chen, S.; Bei, Y .; Yuan, Z.; Zhou, H.; Hong, Z.;
Dong, J.; Chen, H.; Chang, Y .; and Huang, X. 2025. A sur-
vey of graph retrieval-augmented generation for customized
large language models. arXiv preprint arXiv:2501.13958 .
Zhou, D.; Sch ¨arli, N.; Hou, L.; Wei, J.; Scales, N.; Wang,
X.; Schuurmans, D.; Cui, C.; Bousquet, O.; Le, Q.; et al.
2022. Least-to-most prompting enables complex reasoning
in large language models. arXiv preprint arXiv:2205.10625 .

6 Appendix
6.1 Details of Experimental Datasets
We use four benchmark datasets: WebQuestionSP (We-
bQSP) (Yih et al. 2016), GrailQA (Gu et al. 2021) and Com-
plex WebQuestions (CWQ) (Talmor and Berant 2018), and
WebQuestions (WebQ) (Berant et al. 2013) to evaluate the
effectiveness of our method. These datasets include a train-
ing set, a validation set, and a test set. For fair comparison,
we follow the baselines (Sun et al. 2023; Chen et al. 2024)
and use only the test sets as evaluation benchmarks. In order
to save computational resources, following prior research
(Sun et al. 2023; Chen et al. 2024), we only used a sam-
ple of 1000 data points from the GrailQA dataset for eval-
uation. Noticing that, in order to verify the performance of
each method in open world settings, we introduce noise into
the WebQSP and GrailQA datasets, rewriting keywords in
the questions to make it impossible to directly match entities
in the knowledge graph through string matching. We also
sample a small portion of questions from two other datasets,
HotpotQA (Yang et al. 2018) and TriviaQA (Joshi et al.
2017), and merge them with the noisy data from WebQSP
and GrailQA to create two open world datasets (Open-World
WebQSP and Open-World GrailQA) to verify the robustness
of our method. The statistics of the datasets are shown in Ta-
ble 5, and the statistics of the reasoning hops are shown in
Table 6.
Datasets #Train #Validation #Test
WebQSP 3098 - 1639
GrailQA 44,337 6763 13231
CWQ 27734 3480 3475
Webquestions 3778 - 2032
WebQSP(Open-World) 3098 - 1803
GrailQA(Open-World) 44,337 6763 1100
Table 5: Statistics of datasets.
Datasets 1 hop 2 hop ≥3 hop
WebQSP 65.49% 34.51% 0.00%
GrailQA 68.92% 25.82% 5.26%
CWQ 40.91% 38.34% 20.75%
Table 6: statistics of the reasoning hops in WebQSP, CWQ
and GrailQA.
6.2 Experiments of additional Hyper-parameter
Fig. 5 shows the hyperparameter sensitivity analysis for re-
trieval depth and width. The experimental results strongly
validate the effectiveness of our default configuration, i.e.,
width=3 and depth=3. On both datasets, the Hit@1 perfor-
mance of our method peaks when these settings are used.
Figure 5: Performances of AnchorRAG with different search
depths and widths.
This non-linear performance trend is attributed to two rea-
sons. First, the performance improvement is limited by the
inherent complexity of the datasets, as most questions do
not require more than 3 hops of reasoning, as shown in ta-
ble 6. Therefore, a deeper search does not yield significant
benefits. Second, excessively expanding the search space in-
troduces a significant amount of noise from irrelevant en-
tities and relations, which have the negative impact on the
reasoning and lead to a poorer performance. This is shown
by the performance degradation on GrailQA when depth or
width is increased to 4. In particular, the WebQSP dataset
does not contain questions requiring more than 3 hops. Con-
sequently, increasing the retrieval depth beyond 3 on We-
bQSP offers no performance benefit, and we therefore did
not perform experiments with depth=4 on this dataset. To
sum up, our parameter selection presents an optimal trade-
off between maintaining adequate search space, mitigating
noise, and balancing computational overhead.
6.3 Prompts and Examples
The prompts used in our method are detailed in Table 7 and
Table 8.

Prompt for Candidate Entity Generation
Extract all topic entities from the given multi-hop question. Topic entities are proper nouns, named entities, or
specific concepts that are crucial for retrieving external knowledge. They may come from different sub-questions
that contribute to the final answer. If there are multiple topic entities, separate them with commas.
Examples:
Question: Who directed the movie in which Leonardo DiCaprio played Jordan Belfort?
Leonardo DiCaprio, Jordan Belfort
Question: Which team won the Champions League in the same year that Spain won the FIFA World Cup?
Champions League, FIFA World Cup, Spain
Question: Who was the US president when the Berlin Wall fell?
US president, Berlin Wall
Question: Which book was written by the author of ”Pride and Prejudice” and published after 1800?
Pride and Prejudice
Question: What is the capital of the country where Mount Everest is located?
Mount Everest
Question: Who won the Nobel Prize in Physics in the same year that Albert Einstein died?
Nobel Prize in Physics, Albert Einstein
Question: In which city was the founder of Tesla Motors born?
Tesla Motors
Question:
<Question>
Prompt for Relations Pruning
Please retrieve %s relations (separated by semicolon) that contribute to the question and rate their contribution on
a scale from 0 to 1 (the sum of the scores of %s relations is 1).
Examples:
Please retrieve %s relations (separated by semicolon) that contribute to the question and rate their contribution on
a scale from 0 to 1 (the sum of the scores of %s relations is 1).
Name the president of the country whose main spoken language was Brahui in 1980?
Topic Entity: Brahui Language
Relations: Relations
1.{language.human language.main country (Score: 0.5)) }: This relation is highly relevant as it directly relates to
the country whose president is being asked for, and the main country where Brahui language is spoken in 1980.
2.{language.human language.countries spoken in (Score: 0.3) }: This relation is also relevant as it provides in-
formation on the countries where Brahui language is spoken, which could help narrow down the search for the
president.
3.{base.rosetta.languoid.parent (Score: 0.2) }: This relation is less relevant but still provides some context on
the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural
background of the country in question.
Question:
<Question>
Topic Entity:
<Topic Entity>
Relations:
<Relations>
Table 7: Prompt for Candidate Entity Generation and Relations Pruning.

Prompt for Entities Pruning
Please score the entities’ contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is
1).
Examples:
Please score the entities’ contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is
1).
The movie featured Miley Cyrus and was produced by Tobin Armbrust?
Relation: film.producer.film
Entites: The Resident; So Undercover; Let Me In; Begin Again; The Quiet Ones; A Walk Among the Tombstones
Score: 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
The movie that matches the given criteria is ”So Undercover” with Miley Cyrus and produced by Tobin Armbrust.
Therefore, the score for ”So Undercover” would be 1, and the scores for all other entities would be 0.
Question:
<Question>
Relation:
<Relation>
Entities:
<Entities>
Prompt for Triples Filtering
Given the question and a list of knowledge triples, identify and return only the triples that are directly relevant to
answering the question. Do not change the format of the triples. Do not generate any explanations or extra text.
Return only the filtered triples as-is.
Question:
<Question>
Triples:
<Triples>
Prompt for Triples Evaluating
Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to
answer whether it’s sufficient for you to answer the question with these triplets and your knowledge (Yes or No).
Examples:
Q: Find the person who said ¨Taste cannot be controlled by law¨, what did this person die from?
Knowledge Triplets: Taste cannot be controlled by law., media common.quotation.author, Thomas Jefferson
{No}. Based on the given knowledge triplets, it’s not sufficient to answer the entire question. The triplets only
provide information about the person who said ”Taste cannot be controlled by law,” which is Thomas Jefferson. To
answer the second part of the question, it’s necessary to have additional knowledge about where Thomas Jefferson’s
dead.
Q: The artist nominated for The Long Winter lived where?
Knowledge Triplets: The Long Winter, book.written work.author, Laura Ingalls Wilder Laura Ingalls Wilder, peo-
ple.person.places lived, Unknown-Entity Unknown-Entity, people.place lived.location, De Smet
{Yes}. Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De
Smet. Therefore, the answer to the question is De Smet.
...
Question:
<Question>
Triples:
<Triples>
Table 8: Prompt for Entities Pruning, Triples Filtering and Evaluating.
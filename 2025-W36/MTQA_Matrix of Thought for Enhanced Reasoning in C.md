# MTQA:Matrix of Thought for Enhanced Reasoning in Complex Question Answering

**Authors**: Fengxiao Tang, Yufeng Li, Zongzong Wu, Ming Zhao

**Published**: 2025-09-04 06:13:28

**PDF URL**: [http://arxiv.org/pdf/2509.03918v1](http://arxiv.org/pdf/2509.03918v1)

## Abstract
Complex Question Answering (QA) is a fundamental and challenging task in NLP.
While large language models (LLMs) exhibit impressive performance in QA, they
suffer from significant performance degradation when facing complex and
abstract QA tasks due to insufficient reasoning capabilities. Works such as
Chain-of-Thought (CoT) and Tree-of-Thought (ToT) aim to enhance LLMs' reasoning
abilities, but they face issues such as in-layer redundancy in tree structures
and single paths in chain structures. Although some studies utilize
Retrieval-Augmented Generation (RAG) methods to assist LLMs in reasoning, the
challenge of effectively utilizing large amounts of information involving
multiple entities and hops remains critical. To address this, we propose the
Matrix of Thought (MoT), a novel and efficient LLM thought structure. MoT
explores the problem in both horizontal and vertical dimensions through the
"column-cell communication" mechanism, enabling LLMs to actively engage in
multi-strategy and deep-level thinking, reducing redundancy within the column
cells and enhancing reasoning capabilities. Furthermore, we develop a
fact-correction mechanism by constructing knowledge units from retrieved
knowledge graph triples and raw text to enhance the initial knowledge for LLM
reasoning and correct erroneous answers. This leads to the development of an
efficient and accurate QA framework (MTQA). Experimental results show that our
framework outperforms state-of-the-art methods on four widely-used datasets in
terms of F1 and EM scores, with reasoning time only 14.4\% of the baseline
methods, demonstrating both its efficiency and accuracy. The code for this
framework is available at https://github.com/lyfiter/mtqa.

## Full Text


<!-- PDF content starts -->

MTQA: Matrix of Thought for Enhanced Reasoning in Complex Question
Answering
Fengxiao Tang1, Yufeng Li1, Zongzong Wu1, Ming Zhao1
1School of Computer,Science and Engineering,Central South University, Changsha, China
tangfengxiao@csu.edu.cn, 107552404871@stu.xju.edu.cn, Wzy Yeah@csu.edu.cn, meanzhao@csu.edu.cn
Abstract
Complex Question Answering (QA) is a fundamental and
challenging task in NLP. While large language models
(LLMs) exhibit impressive performance in QA, they suffer
from significant performance degradation when facing com-
plex and abstract QA tasks due to insufficient reasoning capa-
bilities. Works such as Chain-of-Thought (CoT) and Tree-of-
Thought (ToT) aim to enhance LLMs’ reasoning abilities, but
they face issues such as in-layer redundancy in tree structures
and single paths in chain structures. Although some studies
utilize Retrieval-Augmented Generation (RAG) methods to
assist LLMs in reasoning, the challenge of effectively uti-
lizing large amounts of information involving multiple enti-
ties and hops remains critical. To address this, we propose
the Matrix of Thought (MoT), a novel and efficient LLM
thought structure. MoT explores the problem in both horizon-
tal and vertical dimensions through the ”column-cell commu-
nication” mechanism, enabling LLMs to actively engage in
multi-strategy and deep-level thinking, reducing redundancy
within the column cells and enhancing reasoning capabili-
ties. Furthermore, we develop a fact-correction mechanism
by constructing knowledge units from retrieved knowledge
graph triples and raw text to enhance the initial knowledge
for LLM reasoning and correct erroneous answers. This leads
to the development of an efficient and accurate QA frame-
work (MTQA). Experimental results show that our frame-
work outperforms state-of-the-art methods on four widely-
used datasets in terms of F1 and EM scores, with reasoning
time only 14.4% of the baseline methods, demonstrating both
its efficiency and accuracy. The code for this framework is
available at https://github.com/lyfiter/mtqa.
Introduction
The human transition from learning to practice generally en-
tails comprehending and reasoning over diverse information
sources, followed by making decisions that align with—or
even extend beyond—the original information. Recent stud-
ies have demonstrated that large language models (LLMs)
can replicate this learning-to-practice paradigm in complex
domains such as intelligence analysis (Wu et al. 2024) and
creative writing (Qin et al. 2024). Complex question answer-
ing (QA) is a foundational and critical downstream task for
LLMs (Zhuang et al. 2023); it demands that a QA system
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.command extensive knowledge and perform sophisticated
reasoning over that knowledge to deliver satisfactory an-
swers.
Recently, large language models (LLMs) have become
one of the most promising and efficient solutions in the
QA field, thanks to their massive pre-trained knowledge
and the reasoning abilities they autonomously generate dur-
ing training(Guo et al. 2025). However, some studies in-
dicate that LLMs exhibit significant shortcomings in com-
plex question-answering reasoning tasks. Various research
efforts have attempted to enhance LLMs’ reasoning perfor-
mance using thought structures. For example, the Chain-of-
Thought (CoT) framework(Wei et al. 2022) breaks down
the reasoning process of LLMs into a series of ordered
steps, thereby improving their reasoning capabilities. How-
ever, CoT suffers from a single reasoning path, limiting its
ability to explore multiple reasoning paths within the solu-
tion space, and may also result in incorrect ideas that cannot
be corrected in time, leading to deviations. Tree-of-Thought
(ToT)(Yao et al. 2023a) organizes the reasoning process into
a hierarchical tree structure, allowing the model to reason
across multiple layers and branches. However, its node se-
lection mechanism is inefficient and often produces redun-
dant nodes, leading to resource wastage and ineffective rea-
soning.
Furthermore, some studies have employed Retrieval-
Augmented Generation (RAG) to assist LLMs in answering
questions, enhancing them with external knowledge through
retrieval and generation, leading to more accurate and re-
liable answers(Cuconasu et al. 2024; Fan et al. 2024). Cur-
rently, mainstream reasoning methods incorporate RAG into
thought structures to provide LLMs with the ability to val-
idate facts. RAT(Wang et al. 2024) refines the step-by-step
reasoning process of CoT by adding a RAG correction at
each reasoning stage to ensure the reasoning trajectory re-
mains on track. RATT(Zhang et al. 2025) combines the
multi-branch, multi-level reasoning of tree structures with
RAG’s external knowledge to correct factual errors, thereby
strengthening LLMs’ reasoning capabilities. However, these
works use vanilla RAG as the fact-validation module, which
limits its ability to assist in reasoning and correct facts. In
complex question-answering tasks, especially in open-ended
tasks involving multiple entities and multi-hop information,
the retrieved knowledge often contains irrelevant or even er-arXiv:2509.03918v1  [cs.CL]  4 Sep 2025

PromptQuery
X
...
Column  1
...
Column 2
...
Column n
LLM
LLM
LLM
LLM
LLM
LLMRAG
RAG
RAG
RAG
RAG
RAGMatrix of Thought  (MoT)  
Where is the German manufacturer of Pirna 014 based?
{}
The Pirna 014 was designed by 
engineers from a German 
manufacturer based in Berlin. … However, the manufacturer 
is not based in Berlin, 
Germany. 
"GERMANY", "GEO", "Germany is a 
geographical entity ." ….… JFM is an aircraft and 
aircraft engine manufacturer.
"JFM", "Dessau , Germany", 
"Aircraft engine manufacturer ..."The Pirna 014 
was designed 
by engineers 
from JFM, 
but JFM is 
not based in 
Berlin.. .....German 
manufacturer 
of Pirna 014 in 
Dessau .
We are also able to find errors and 
find strategies within one layer .Chain of Thought
Query
Where is the German manufacturer 
of Pirna 014 based?
The Pirna 014 was designed by engineers from a 
German manufacturer based in Berlin.The initial answer is incorrect
… by a former East German state -owned manufacturer 
based in Pirna, Saxony. 
German manufacturer of 
Pirna 014 in Pirna, Saxony .
Unable to think from other perspectives.
Get the wrong answer
Thought of Tree
 (ToT)
Query
LLM
Where is the German manufacturer 
of Pirna 014 based?
{The Pirna 014 ... a German 
manufacturer based in Berlin.
The Pirna 014 ... German 
manufacturer  not based in Berlin.The Pirna 014 ... German 
manufacturer based in Berlin.Generate duplicate answers
 … However, the manufacturer is 
not based in Berlin, Germany.  … the manufacturer is based in 
Pirna, Saxony. 
 The Pirna 014 
was designed by 
engineers from 
a German 
manufacturer, 
but its location 
is not in Berlin. ..
...High cost and 
slow to find 
answers
...LLM
 … the manufacturer is based in 
Pirna, Saxony. And somtimes incorrect.PromptQuery
X
...
Column  1
...
Column 2
...
Column n
LLM
LLM
LLM
LLM
LLM
LLMRAG
RAG
RAG
RAG
RAG
RAGMatrix of Thought  (MoT)  
Where is the German manufacturer of Pirna 014 based?
{}
The Pirna 014 was designed by 
engineers from a German 
manufacturer based in Berlin. … However, the manufacturer 
is not based in Berlin, 
Germany. 
"GERMANY", "GEO", "Germany is a 
geographical entity ." ….… JFM is an aircraft and 
aircraft engine manufacturer.
"JFM", "Dessau , Germany", 
"Aircraft engine manufacturer ..."The Pirna 014 
was designed 
by engineers 
from JFM, 
but JFM is 
not based in 
Berlin.. .....German 
manufacturer 
of Pirna 014 in 
Dessau .
We are also able to find errors and 
find strategies within one layer .Chain of Thought
Query
Where is the German manufacturer 
of Pirna 014 based?
The Pirna 014 was designed by engineers from a 
German manufacturer based in Berlin.The initial answer is incorrect
… by a former East German state -owned manufacturer 
based in Pirna, Saxony. 
German manufacturer of 
Pirna 014 in Pirna, Saxony .
Unable to think from other perspectives.
Get the wrong answer
Thought of Tree
 (ToT)
Query
LLM
Where is the German manufacturer 
of Pirna 014 based?
{The Pirna 014 ... a German 
manufacturer based in Berlin.
The Pirna 014 ... German 
manufacturer  not based in Berlin.The Pirna 014 ... German 
manufacturer based in Berlin.Generate duplicate answers
 … However, the manufacturer is 
not based in Berlin, Germany.  … the manufacturer is based in 
Pirna, Saxony. 
 The Pirna 014 
was designed by 
engineers from 
a German 
manufacturer, 
but its location 
is not in Berlin. ..
...High cost and 
slow to find 
answers
...LLM
 … the manufacturer is based in 
Pirna, Saxony. And somtimes incorrect.Figure 1: Comparison of Chain of Thought, Thought of Tree, and MoT for Complex Question Answering.
roneous information, which misguides the LLMs’ reasoning
process. The quality of the factual verification knowledge
provided by RAG becomes one of the key factors in guiding
LLMs’ reasoning.
To address the aforementioned issues and improve the re-
liability and accuracy of LLMs in complex question answer-
ing tasks, we aim to develop a method that efficiently uti-
lizes existing knowledge and performs multi-branch reason-
ing with low redundancy to generate accurate answers. In
terms of efficiently utilizing existing knowledge, we focus
on exploring the representation of knowledge retrieved by
RAG to assist LLMs in answering questions. In the context
of multi-branch and low-redundancy reasoning, we aim for
the model to autonomously adopt different perspectives to
understand the existing knowledge and perform reasoning
that leads to an answer close to the correct one, all while
using the same initial knowledge.
To achieve these goals, we propose a novel Matrix
of Thought (MoT) approach to ensure that the reason-
ing process for complex question-answering tasks is multi-
branching and low-redundancy. As shown in Figure 1, in
MoT, our RAG retrieves entity-relation-entity triples that
match the question, along with their corresponding original
text segments, which are then constructed into knowledge
units to serve as auxiliary knowledge for initiating reason-
ing and fact verification. Additionally, we introduce a col-
umn cell communication mechanism that actively helps the
LLM explore alternative perspectives for solving the prob-
lem, thereby reducing the redundancy in the generated rea-
soning process within the column. Then, MoT combines the
various branching strategies with RAG’s auxiliary knowl-
edge for summarization and optimization, using this as the
reasoning initiation knowledge for the next column cell,
thereby approaching the most promising answer.
The contributions of this paper are summarized as fol-lows:
1. We propose an innovative large model reasoning
paradigm, MoT, which addresses the issues of information
redundancy within the same layer of tree structures and the
single decision path in chain structures. MoT enhances the
flexibility and adaptability of reasoning at a minimal cost.
2. We have developed a novel framework for com-
plex QA tasks, the Matrix of Thought Question Answer-
ing (MTQA) framework. This framework innovatively com-
bines KG triples with their corresponding original text into
knowledge units for knowledge enhancement and correc-
tion. By integrating the MoT reasoning paradigm, it achieves
a fully automated QA pipeline, enabling efficient reasoning
on the problem from both breadth and depth.
3. We conducted a series of experiments to validate the
efficiency and reliability of our method in complex question
answering across various domains. The results demonstrate
that our method holds a clear advantage over existing ap-
proaches.
Related Work
Thought Structures for LLMs
The thought structures proposed (Ding et al. 2023; Mi-
naee et al. 2024) is a mainstream prompt engineering ap-
proach that guides LLMs to engage in more logical thought
processes, thereby improving the quality of generated re-
sponses. Chain of Thought (Wei et al. 2022) is a representa-
tive work that guides LLMs to generate intermediate reason-
ing steps within the prompt, thereby steering the large model
through the reasoning process. Wang et al. developed the
Self-consistency with Chain of Thought (CoT-SC) method
(Wang et al. 2023), which extends CoT by solving a problem
using multiple independent and distinct chains of thought.
The most reliable answer is then selected as the final output

response. The Tree of Thoughts (ToT) (Yao et al. 2023a) ab-
stracts the reasoning process of a problem into a tree search.
By constructing multiple branches and evaluating their con-
tributions to solving the problem, ToT guides the LLM in
reasoning, enabling a broad and global exploration of the
solution space.
Current mainstream research focuses on CoT and ToT;
however, the inherent properties of their structures present
performance bottlenecks, leading to issues such as single
reasoning paths, generation of redundant information, and
the risk of deviating from the reasoning trajectory due to in-
correct ideas during inference. One of the key focuses of our
research is to develop a more efficient and reliable thought
structure paradigm, enabling LLMs to perform exception-
ally well in QA tasks and beyond within this new frame-
work.
Retrieval-Augmented Generation (RAG) for LLM
on QA
When LLMs solve extremely complex tasks, they suf-
fer from ”complete accuracy collapse”(Shojaee*† et al.
2025), significant performance degradation, and LLMs hal-
lucination problems due to insufficient reasoning capabili-
ties(Sriramanan et al. 2024). Retrieval-Augmented Gener-
ation (RAG)(Lewis et al. 2020) is a pivotal technique for
mitigating such hallucinations and enhancing the quality
of LLM outputs. Retrieval-Augmented Generation (RAG)
stores task-relevant knowledge in an external repository
and, when the LLM is invoked, retrieves pertinent in-
formation to serve as supplementary context. GraphRAG
(Edge et al. 2024) extends this paradigm by incorporat-
ing knowledge graphs (KGs): an LLM produces concise
community-level summaries that both abstract the input doc-
uments and provide graph indices; each community then
generates its response independently, and the individual
outputs are subsequently aggregated into a global answer.
Recent studies have begun to integrate explicit reasoning
paradigms into the RAG workflow. Retrieval-Augmented
Thought (RAT)(Wang et al. 2024) merges the RAG retrieval
step directly into the chain-of-thought process, allowing the
model to iteratively refine and correct its reasoning step by
step. Retrieval-Augmented Thought Tree (RATT) (Zhang
et al. 2025) adopts a tree-structured reasoning framework
in which each node invokes RAG for factual verification,
thereby enhancing the logical soundness and factual relia-
bility of LLM reasoning.
However, current research often treats the retrieved
knowledge as various fragmented KG triples or text, which
do not align well with the preferences of LLMs. Addition-
ally, there has been limited work on designing RAG sys-
tems specifically for assisting thought structure reasoning.
This paper focuses on studying the optimal representation of
the knowledge retrieved by RAG to aid in reasoning within
thought structures, enabling LLMs to efficiently utilize the
knowledge for reasoning and derive the best possible an-
swers.Methodology
This section presents the Matrix of thought Question An-
swering (MTQA) framework in detail. As illustrated in Fig-
ure 2, MTQA is constructed atop the novel MoT reasoning
paradigm to tackle complex QA tasks. The framework is
designed to make fuller use of existing knowledge and to
equip LLMs with both breadth and depth when exploring
the problem’s solution space. Our method improves the re-
liability and comprehensiveness of LLM-generated answers
from two aspects: adopting a superior knowledge represen-
tation to enhance the LLM’s comprehension. And employ-
ing a powerful thought structures to strengthen the LLM’s
reasoning capability. The framework consists of two parts:
(1) retrieval-augmentation, (2) Matrix of thought.
Problem Definition
A question-answering task comprises a given question Q
and a set of input documents D={d1, d2, . . . , d n}, where
ndenotes the number of input documents. Both the question
and the input documents are expressed in natural language.
Dis the auxiliary factual knowledge that supports the large
language model Lin answering Qand thus obtaining the an-
swerA. This factual knowledge can take the form of textual
passages, triples in a knowledge graph, or Internet web page
URLs. This paper focuses on treating text paragraphs as aux-
iliary factual knowledge. Using the retriever R, we retrieve
from Dthe factual knowledge set K={k1, k2, . . . , k m}
that matches the current input Xto be verified by L, where
mrepresents the top- mtext blocks ranked by their degree of
match with X.
In MTQA, our objective is to construct a matrix of
thought Mthat can carry out the widest and deepest possible
inference within the solution space of question Q, thereby
identifying the optimal answer A.
Retrieval-Augmented
Step 1: Knowledge Database construction. In MTQA, the
first step is to build a retrievable external knowledge base
and partition the given set of documents into chunks whose
size is suitable for processing by the LLM:
C={c1, c2, c3, . . . , c k}, k =ND
Clen
(1)
Where Cdenotes the set of blocks after the document
group Dis chunked, krepresents the number of blocks, ND
is the total number of words in the document group D, and
Clenis the length of the set text block.
Subsequently, we use LLM to recognize and extract en-
tities (e.g., dates, places, people, events, etc.) and the re-
lationships between them according to set cues (Detailed
prompt are presented in the Appendix), generating a knowl-
edge graph Gthat comprehensively encompasses the con-
tent of the document, which consists of the extracted entities
vwith their relationships with each other ε, and reacts to the
connections and focuses in the whole document set.
First, llm embeds the chunked text block Ciafter filling it
to the prompt template:

Source
DocsChunk 1
Chunk 2
Chunk n……
Knowledge Units
LLMQuery
KG database
Text database
 Query
Prompt
LLM
...
LLM...
...
LLMMatrix of Thought Split
Where is the  
German 
manufacturer of 
Pirna 014 based?
 "JFM", "Dessau, Germany"
"JFM is an Aircraft engine manufacturer …
The wing's 46th Test Group was a tenant unit at Holloman Air Force...Original TextRelationshipsEntities
Dessau  
Tn
1
m
Tn
1
3
Tn
1
2
Tn
1
1
Tn
n
1
Tn
n
2
Tn
n
3
Tn
n
mKnowledge 
Units
Sn1
Sn1…(n-1)
Column 1 Column nKnowledge  Database Construction 1Retrieval -Augmented
2Retrieval  knowledge 
through queries
3
4Column   cell 
communication mechanism5correction & Summary Reasoning  enhanceKnowledge 
UnitsKnowledge 
UnitsIndexFigure 2: Framework of MTQA. Given an input including a query and source documents, the LLM retrieves relevant knowledge
units from both knowledge graphs and text databases. These knowledge units are then integrated into the Matrix of Thought
(MoT), where multiple reasoning strategies are explored in parallel through column-cell communication. The reasoning process
is enhanced by retrieval-augmented knowledge, and the final answer is refined through correction and summarization.
Lci= embed 
template( Ci, x)
(2)
Where xrepresents the template for performing entity and
relationship extraction, Cirepresents the text block, and Lci
represents the generated embedding vectors subsequently
input to the LLM for comprehension. The LLM performs
recognition and extraction operations to extract the entity v
and the relation εfrom Lci.
v, ε=Extraction & identification kX
i=1Lci!
(3)
Subsequently, we develop guidance llm to construct a
key-value pair (k, v)for each recognized and extracted en-
tityvand the relation εbetween them. The keys are com-
posed of individual words or short phrases and are later
used as query terms to retrieve the corresponding entities
and relations. For entities, their names function as keys; for
relations, the LLM summarizes the text that describes the
link between two entities to generate multiple keys. The
associated values are the source passages in the external
knowledge base from which those entities and relations are
drawn. The key–value mechanism both accelerates retrieval
and preserves direct access to the source passages in the ex-
ternal knowledge base. During the generation of entities and
their corresponding relations, the LLM augments each itemwith an explanatory description, allowing subsequent LLM-
based reasoning modules to exploit this enriched knowledge
more effectively for refinement and enhancement.
ˆv,ˆε= Deduplication 
v, ε
(4)
After completing the extraction of entity vand relation ε,
we perform a de-redundancy operation to identify and merge
the duplicates of vandεto obtain the unique ˆvandˆεin the
whole knowledge graph, which improves the overhead of the
subsequent operations on the knowledge graph and ensures
that there is no redundancy in the retrieved knowledge.
G= 
ˆv,ˆε
(5)
After completing the above operations, the entities ˆvas
nodes and the relations ˆεas edges construct the knowl-
edge graph G. Among them, the source entity ˆvsand the
target entity ˆvtwith the relation ˆεbetween them form
the basic knowledge graph structure: the knowledge graph
triple (ˆvs,ˆε,ˆvt), which serves as the basic unit for construct-
ing the knowledge graph and retrieving the knowledge base.
After constructing the knowledge graph, we consider
the incremental update operations for the knowledge graph
when the document group is updated. For the newly added
document group D′, we follow the aforementioned steps
to extract entities and relations, obtaining a new subgraph

G′= (V′, ε′). Next, the entities and relations in GandG′are
merged by taking their union, thereby combining G′withG.
The key-value structure of the knowledge graph that we
construct as the external knowledge base for MTQA offers
two advantages:
(1) Compared to indexing the original text passages, the
knowledge graph stores structural information, providing
the MTQA framework with multi-hop reasoning capabili-
ties to integrate global information, enabling LLMs to un-
derstand knowledge across text blocks.
(2) The key-value structure strongly associates the knowl-
edge graph with its source texts, facilitating efficient re-
trieval while enabling the construction of knowledge units
that combine knowledge graph triples with source text-
enhanced knowledge.
Step 2: Retrieval knowledge through queries. After
constructing the external knowledge base, we build the re-
trieval function mechanism to match the knowledge that best
supports the query. We use a dual retrieval mechanism (Edge
et al. 2024) to extract relevant information from the numer-
ous and complex intertwined dependencies of entities in the
external knowledge base. This includes detailed queries and
summary queries. Detailed queries focus on specific details
and require precise identification of relevant entities or rela-
tions for factual judgment or providing key evidence. Sum-
mary queries, on the other hand, are broader in scope, sum-
marizing topics or abstracts. They typically involve search-
ing across multiple entity-relation triples or even text blocks,
and are used to answer open-ended or comprehensive ques-
tions. We perform retrieval by combining graph structure in-
formation with vector representations. The retrieval process
is mainly divided into three steps:
(1) Extract the keywords from the query Q, distinguishing
between detailed keywords k(d)and generalized keywords
k(g).
(2) Match the extracted detailed keywords with the entity
keys in the vector database, and match the generalized key-
words with the relation keys.
(3) Further retrieve the neighboring nodes (one-hop and
two-hop adjacent nodes) in the knowledge graph for the
matched entities and relations.
The retrieved knowledge consists of entities (nodes of
the knowledge graph), relations (edges of the knowledge
graph), and the original text (indexed by the key-value
pairs of nodes and edges), forming knowledge units, as
shown in Figure 2. The knowledge units simultaneously
contain structured information, supporting multi-hop rea-
soning and emphasizing key information through knowledge
graph triples(Saxena, Tripathi, and Talukdar 2020), as well
as undistorted, broader content original (Lewis et al. 2020)
(such as complex metaphors, emotional expressions, etc.),
which better aligns with the preferences of LLMs (Dong
et al. 2025) in original text. Experimental results demon-
strate that these knowledge units significantly enhance the
reasoning performance of LLMs.
Matrix of Thought
Step 3: Reasoning enhance. Next, we start building the col-
umn units in MoT. A given query Qis fed into the retrievalenhancement module to get the knowledge unit KU 1that
underpins the reasoning of LLMs:
KU 1=Retrieval-Augmented (S(Q)) (6)
Next, we fill the knowledge unit KU 1and the query Q
into the prompt template, guiding the LLMs to leverage the
information contained in the knowledge unit to enhance their
understanding of the question. A strategy from shallow to
deep is employed to guide the LLMs in generating the initial
thought nodes:
Tn1
1=LLM (template (Q, KU 1)) (7)
The initial nodes contain relatively shallow and indicative
views and solutions to the question. MTQA will iteratively
explore and refine these nodes in terms of breadth and depth,
gradually evolving them so that the generated thought nodes
converge towards the target of the question.
Step 4: Column cell communication mechanism. To
more efficiently explore the breadth of the problem and
implement a multi-branching strategy, we have developed
the column cell communication mechanism. Specifically, in
MoT, a portion of the paragraph that represents the strategy
and direction of the previous thought node in the same col-
umn is extracted (typically the last few paragraphs of the
thought node). This portion is then used as an example to
guide the LLMs in generating the next thought node, encour-
aging a completely different line of thought and strategy.
Tnn
m=LLM 
template 
Q, KU n, αTnn
m−1
(8)
Where αis the communication weight with range 0≤
α≤1. It denotes the proportion (rounded up) of paragraphs
taken from node Tnn
m−1. A value of 0means there is no
communication between the two thought nodes, and a value
of1means the entire content of Tnn
m−1is passed to Tnn
m.
This parameter determines the transparency of prior ideas to
the LLMs. In MoT, the column-cell communication mech-
anism is essential and distinguishes MoT from ToT and re-
lated reasoning structures. Throughout MoT, we use a ma-
trixAto represent the communication weights. Given an
m×nthought matrix, the weight matrix has dimension
(m−1)×(n−1). The optimal value of this parameter
is detailed in the Experiments section.
A=
α1
m−1···αn−1
m−1.........
α1
1··· αn−1
1
 (9)
Step 5: Refinement and Summarization. After generat-
ing the entire column of thought nodes, MTQA matches all
thought nodes within the current column cell to the knowl-
edge units KU′
nin the external knowledge base, and uses
the LLM together with KU′
nto refine and aggregate these
thought nodes, yielding the summary node Snn.
KU′
n=Retrieval-Augmented 
S 
Q,mX
i=1Tnn
i!!
(10)

Sn
n=LLM 
template (Q, KU′
n)
(11)
The summary node corrects any fact-inconsistent strate-
gies that appear among the thought nodes in the column,
while preserving as much strategy diversity as possible.
When generating the next column cell, the process builds
on the current column’s summary node to further develop
promising strategies. After a specified number of iterations,
the final summary node is taken as the output of MTQA.
Algorithm 1: MTQA Framework
Require: Question Q, Document set D =
{d1, d2, . . . , d n}, Retriever R, LLM L, Knowledge
graph G
Ensure: Final Answer A
1:Step 1: Knowledge Database Construction
2:C← {c1, c2, . . . , c k}
3:foreach chunk ci∈Cdo
4: Lci←embed(template( ci, x))
5: v, ε←Extraction & Identification (Lci)
6:end for
7:ˆv,ˆε←Deduplication( v, ε)
8:G←(ˆv,ˆε)
Step 2: Retrieval via Queries
9:k(d), k(g)←Extract keywords (Q)
10:K ← Retrieve( k(d), k(g), G)
11:KU← {K Triples +KSource Text }
Step 3: Initial Thought Node Generation
12:KU 1←Retrieval-Augmented (S(Q))
13:Tn1
1←L(template( Q, KU 1))
Step 4: Column Cell Communication
14:foreach column n, row mdo
15: Tnn
m←L(template( Q, KU n, αTnn
m−1))
16: A∈R(m−1)×(n−1)
17:end for
Step 5: Refinement and Summarization
18:KU′
n←Retrieval-Augmented (S(Q,Pm
i=1Tnn
i))
19:Sn
n←L(template( Q, KU′
n))
Step 6: Output
20:A←Snfinalnfinal
Discussion
In this chapter, we further elaborate on the objectives in-
troduced in the Introduction. While controlling the frame-
work’s runtime overhead, our method integrates external
knowledge and refines it into knowledge units to help LLMs
efficiently enhance reasoning and correct strategies. More-
over, by adopting a matrix structure and a column-cell
communication mechanism, it encourages LLMs to proac-
tively explore multiple dimensions of the solution space. The
MTQA framework for complex question answering is an in-
stance of the MoT paradigm. It is worth noting that the MoT
paradigm abstracts the LLM reasoning process as a matrix.
When the number of columns or rows of this matrix equals
1, MoT degenerates into a CoT structure with RAG-based
correction. When the column-cell communication weight is0, MoT degenerates into a ToT-like structure. This indicates
that CoT and ToT are special cases of MoT, which confers
greater flexibility and adaptability on MoT.
Experiment
Experimental Setup
In this chapter, we design two experiments to validate the
effectiveness of MTQA.
(1)Multi-metric performance evaluation: we conduct
comparative experiments between MTQA and existing ad-
vanced methods using two metrics, the F1 score and EM.
(2)Multidimensional win-rate evaluation: we prompt a
powerful LLM to perform pairwise, multi-dimensional as-
sessments between MTQA and each baseline model’s re-
sponses, generate an evaluation report, and select the supe-
rior answer to compute the win rate.
To implement the MTQA framework, we default to GPT-
4o-mini as the LLM L. To ensure consistency with prior
work (Trivedi et al. 2022), the dataset chunk size is set to
1200. We use the nano vector database for vector database
operations. The hyperparameter settings for MoT are de-
tailed in the Numerical Analysis subsection of this chapter.
All experiments are conducted on a Linux server equipped
with six NVIDIA A6000 GPUs.
Multi-metric performance evaluation
In this experiment, we use NaturalQuestions (Kwiatkowski
et al. 2019) for single-hop QA evaluation, and Hot-
potQA (Yang et al. 2018) together with 2WikiMultihopQA
(2WMQA) (Ho et al. 2020) for multi-hop QA evaluation.
To ensure consistency with prior work, we sample 500 in-
stances from each dataset. We use the F1 score and Ex-
act Match (EM) as evaluation metrics for all three datasets.
We compare MTQA with GPT-4o-mini under direct prompt-
ing, Zero-shot CoT(Kojima et al. 2022), Few-shot CoT(Wei
et al. 2022), and ToT, and against RAG-based baselines in-
cluding vanila RAG(Lewis et al. 2020), ReAct(Yao et al.
2023b), IRCoT(Trivedi et al. 2022), FLARE(Jiang et al.
2023), Self-Rag(Asai et al. 2024), SearChain(Xu et al.
2024), Rowen(Ding et al. 2024), SlimPLM(Tan et al. 2024),
LightRAG(Edge et al. 2024), CoA(Zhang et al. 2024),
RAT(Wang et al. 2024), and RATT(Zhang et al. 2025). For
RATT, we configure its internal RAG to be identical to that
of MTQA (the original work uses online web pages as the
external knowledge base, which we replace with the knowl-
edge units described above).
Results Analysis: We compare MTQA with the above
baselines, as shown in Table 1. Based on the evaluation
metrics, we draw the following conclusions: First, MTQA
outperforms all baselines. Compared with non-RAG base-
lines, our method increases the F1 score by 5.3%–9.1% and
EM by 6.2%–8.9%. Relative to RAG-based and reasoning-
structure methods, we still maintain at least a 3% lead
in both F1 and EM, which is especially pronounced on
multi-hop QA datasets such as HotpotQA and 2WMQA.
These results demonstrate the effectiveness of the proposed
MoT paradigm and knowledge units. Second, we observe
that several RAG methods do not consistently surpass the

Dataset
MetricNQ HotpotQA 2WMQA
F1 EM F1 EM F1 EM
Vanilla GPT 4o-mini 0.425 0.298 0.380 0.265 0.316 0.229
Zero-shot CoT 0.457 0.295 0.355 0.262 0.324 0.216
Few-shot CoT 0.447 0.298 0.376 0.255 0.363 0.227
ToT 0.456 0.305 0.368 0.267 0.345 0.237
Vanilla RAG 0.385 0.258 0.387 0.254 0.314 0.244
ReAct 0.335 0.212 0.390 0.270 0.305 0.204
IRCoT 0.344 0.216 0.361 0.232 0.318 0.202
FLARE 0.455 0.318 0.391 0.268 0.364 0.246
Self-Rag 0.387 0.270 0.357 0.220 0.311 0.210
SearChain 0.337 0.214 0.349 0.216 0.313 0.222
Rowen 0.452 0.286 0.382 0.240 0.307 0.212
SlimPLM 0.442 0.280 0.393 0.266 0.368 0.242
LightRAG 0.439 0.312 0.362 0.282 0.356 0.276
CoA 0.452 0.326 0.387 0.272 0.362 0.243
RAT 0.462 0.338 0.392 0.275 0.376 0.239
RATT 0.479 0.323 0.410 0.279 0.406 0.284
MTQA 0.510△3.1%0.368△3.0%0.452△4.2%0.318△3.6%0.454△4.8%0.326△4.2%
Table 1: Overall results on three datasets for multi-metric performance evaluation. The existing best results are underlined and
the best results are in bold.△indicates the improvement of our results over the state-of-the-art results.
non-RAG baselines. This may be due to inefficient re-
trieval mechanisms that introduce noise or erroneous ev-
idence, thereby misleading the LLM’s responses. Finally,
our method achieves substantial gains in both F1 and EM,
indicating that the column-cell communication mechanism
effectively mines multi-branch lines of reasoning and in-
creases response breadth, thereby improving answer quality.
In the above experimental setup, we use the F1 score
and EM as evaluation metrics. However, the gold answers
in these datasets are typically short, while LLMs tend to
produce logically structured paragraph-length responses (Li
et al. 2022), and answers may be semantically similar yet
lexically non-identical (Zhou et al. 2023). These factors ren-
der F1 and EM insufficiently rigorous and prevent them
from evaluating the quality of non-standard answers in
open-domain QA (e.g., describing the personality traits of
a novel’s protagonist).
Multidimensional win-rate evaluation
To address the issues in Multi-metric performance evalua-
tion, we build on the method of Guo et al. (Edge et al. 2024)
and design a simple yet effective protocol to evaluate, in a
multidimensional manner, the response quality of QA mod-
els for open-domain QA tasks. We select four datasets from
UltraDomain (Qian et al. 2025), which is compiled from 428
English-language university textbooks and spans 18 disci-
plines. Specifically, we use the Biology, Law, Computer Sci-
ence (CS), and Physics subsets, each containing millions of
tokens.
Implementation details: To evaluate MTQA’s effective-
ness on high-level sensemaking QA tasks, we adopt the
generation procedure summarized by Edge et al(Edge et al.2024), instructing the LLM to generate five users in the tar-
get domain, with each user providing five tasks. Each user is
assigned domain-specific knowledge and attributes to elicit
greater nuance in task and question formulation. For each
task, the LLM generates ten questions that require under-
standing of the entire corpus. In total, each dataset yields
250 domain-aligned questions. For the responses produced
by MTQA and all baselines, we instruct GPT-4o to conduct
pairwise evaluations (MTQA vs. each baseline) along four
dimensions, including:
Comprehensiveness : How much detail the answer pro-
vides to cover all aspects and details of the question.
Accuracy : How well the answer ensures factual correct-
ness, data reliability, and logical consistency across different
perspectives and insights.
Empowerment : How well the answer helps the reader
understand the topic and make informed judgments.
Overall : An aggregate assessment of the preceding three
indicators to determine the best overall answer.
Regarding the baselines, we select the graph-based RAG
system LightRAG, as well as CoT-style methods with RAG
correction (CoA and RAT) and a ToT-style method (RATT),
to compute win rates against MTQA. In RAT and RATT, the
RAG component is configured identically to that of MTQA.
An example of the evaluation process is shown in Figure 10
in the appendix.
Results Analysis: As shown in Table 2, we compare the
win rates between MTQA and the aforementioned base-
lines pairwise. It is evident that our method achieves a win
rate of over 80% across most dimensions and in the over-
all assessment, further validating the effectiveness of the
MoT approach. Among them, the non-reasoning-structure

MetricBiology CS Legal Physics
LightRAG MTQA LightRAG MTQA LightRAG MTQA LightRAG MTQA
Comprehensiveness 5.6% 94.4% 4.8% 95.2% 8.8% 91.2% 5.2% 94.8%
Accuracy 6.4% 93.6% 4.0% 96.0% 4.8% 95.2% 5.6% 94.4%
Empowerment 3.6% 96.4% 6.8% 93.2% 6.0% 94.0% 7.6% 92.4%
Overall 4.4% 95.6% 3.6% 96.4% 5.2% 94.8% 5.2% 94.8%
CoA MTQA CoA MTQA CoA MTQA CoA MTQA
Comprehensiveness 13.6% 86.4% 9.6% 90.6% 11.2% 88.8% 17.2% 80.8%
Accuracy 15.6% 84.4% 13.6% 86.4% 13.6% 86.4% 12.0% 88.0%
Empowerment 17.2% 82.8% 15.6% 84.8% 16.4% 83.6% 18.8% 81.2%
Overall 15.2% 84.8% 10.4% 89.6% 12.4% 87.6% 13.2% 86.8%
RAT MTQA RAT MTQA RAT MTQA RAT MTQA
Comprehensiveness 17.2% 82.8% 10.8% 89.2% 16.4% 83.6% 18.8% 81.2%
Accuracy 17.6% 82.4% 25.2% 74.8% 23.6% 76.4% 22.4% 77.6%
Empowerment 19.6% 80.4% 22.8% 77.2% 16.4% 83.6% 20.8% 79.2%
Overall 16.4% 83.6% 11.2% 88.8% 15.2% 84.8% 18.4% 81.6%
RATT MTQA RATT MTQA RATT MTQA RATT MTQA
Comprehensiveness 16.8% 83.2% 12.0% 88.0% 22.4% 77.6% 29.6% 70.4%
Accuracy 34.4% 65.6% 32.0% 68.0% 16.4% 83.6% 32.0% 68.0%
Empowerment 24.0% 76.0% 19.2% 80.8% 24.8% 75.2% 19.2% 80.8%
Overall 18.4% 81.6% 15.2% 84.8% 18.4% 81.6% 15.2% 84.8%
Table 2: The winning rate of MTQA and its baseline in four datasets and four evaluation dimensions in the Multidimensional
win-rate evaluation. The one with the higher winning rate is marked in red.
method (LightRAG) achieves a win rate of no more than
10% across the four datasets. This further supports our view-
point that, under similar knowledge enhancement, reasoning
structures play a critical role in the utilization of knowl-
edge and the logical reasoning ability, significantly influ-
encing the overall quality of the answers. The chain-based
reasoning structure methods (CoA, RAT) achieve win rates
ranging from 9.6% to 25.2%, with the win rate for compre-
hensiveness assessment consistently lower than that of other
dimensions. This indicates that while chain-based reason-
ing structures have a positive effect on answer generation,
their lack of multi-branch reasoning capability results in a
noticeable disadvantage in comprehensiveness compared to
tree-based structures (e.g., RATT) and matrix-based struc-
tures (e.g., MTQA). The win rates for the tree-based reason-
ing structure method (RATT) range from 12% to 34.4%. We
attribute this advantage to the column-cell communication
mechanism employed by MTQA, which better stimulates
the LLMs’ multi-branch reasoning strategies for answering
the questions, thereby yielding higher-quality answers.
Ablation Study
Subsequently, we conducted an ablation study and devel-
oped four variants of MTQA to demonstrate the effective-
ness of each mechanism within MTQA. The comparison
is performed using the single-hop QA dataset NQ and the
multi-hop QA dataset HotpotQA. The experimental results
are shown in Table 3.
No RAG correction : The -RAG variant removes the
RAG correction module from MTQA. It is evident that themodel’s performance is significantly impacted by the knowl-
edge boundaries and LLM hallucinations, resulting in a no-
table decline in performance on both datasets. Interestingly,
compared to Table 1, the -RAG variant outperforms non-
RAG methods, which demonstrates the superiority of the
thought matrix over other reasoning structures.
No origin text enhancement : The -origin variant re-
moves the use of the original text within the knowledge unit.
It is evident that there is a certain degree of performance
loss compared to MTQA. In the multi-hop dataset, the per-
formance of the -origin variant is significantly lower than
that of the -KG variant, highlighting the clear advantage of
knowledge graph (KG)-structured data in multi-hop prob-
lems compared to single-hop datasets.
No KG enhancement : The -KG variant removes the use
of knowledge graph triples within the knowledge unit. By
combining the data from the -origin variant, it is evident that
the absence of certain forms of knowledge leads to a signif-
icant performance loss. Furthermore, when handling multi-
hop problems, the original text, used as corrective knowl-
edge, has limited effectiveness. We attribute this to the lack
of the retrieval capability for structured multi-hop informa-
tion from the knowledge graph.
No communication : The -comm variant sets the column-
cell communication weight matrix in the MoT module to 0,
effectively degenerating the MoT structure into a tree-based
reasoning structure. From the results in Table 3, it is evident
that removing the column-cell communication mechanism
causes a significant performance loss for MTQA, even ex-
ceeding the performance losses seen in the -KG and -origin

variants. This highlights that, during reasoning, the thought
structure plays a dominant role compared to the enhanced
forms of knowledge.
Dataset
MetricNQ HotpotQA
F1 EM F1 EM
-RAG 0.467 0.318 0.384 0.278
-origin 0.489 0.334 0.423 0.294
-KG 0.483 0.329 0.407 0.273
-comm 0.474 0.327 0.409 0.275
MTQA 0.510 0.368 0.452 0.318
Table 3: Comparison of F1 and EM scores of MTQA and its
four variant models on two datasets.
Numerical Analysis
In this subsection, we investigate two crucial hyperparame-
ters in the MTQA framework: the column-cell communica-
tion weight matrix and the size of the thought matrix. We
explore their impact on both the performance and efficiency
of the QA task using the HotpotQA dataset.
Weight Matrix Analysis First, we discuss the impact of
the column-cell communication weight matrix settings on
the framework’s performance. In this experiment, we con-
duct a comparative study of 12 different weight matrix con-
figuration methods using a 3×4matrix as an example. The
configuration methods include Uniform, Gaussian, four dif-
ferent constant matrices (0.2, 0.5, 0.8, 1.0), as well as ver-
tically increasing matrices (Vert-0.1, Vert-0.2, incrementing
row by row along the y-axis), horizontally increasing ma-
trices (Hor-0.1, Hor-0.2, incrementing column by column
along the x-axis), and combined vertical and horizontal in-
creasing matrices (Vert&Hor-0.1, Vert&Hor-0.2, increment-
ing both row by row and column by column along the x- and
y-axes). The Gaussian distribution is set with a mean of 0
and a variance of 1, and the generated matrix is normalized
to ensure each weight is between 0 and 1. The evaluation
metrics for the experiment are the F1 score and Exact Match
(EM) score.
UniformGaussianConst-0.2Const-0.5Const-0.8Const-1.0Vert-0.1Vert-0.2Hor-0.1Hor-0.2
Vert&Hor-0.1Vert&Hor-0.20.200.250.300.350.400.450.50Scorce
Setup Method F1  EM
Figure 3: Comparison of F1 and EM scores for various
weight setting methods.The experimental results are shown in Figure 3. We ob-
serve that the Vert&Hor-0.1 weight setting method achieves
the best performance among all methods, with F1 and EM
scores of 0.452 and 0.310, respectively. However, the per-
formance of Vert&Hor-0.2 decreases instead. We infer the
following reason: In the early stages of reasoning, the model
tends to passively adopt the thought process from the previ-
ous thought node, resulting in strategies that are highly sim-
ilar to the previous ones. Therefore, it is necessary to sup-
press the influence of the previous thought node’s reasoning
in the early stages to further stimulate the LLMs’ reasoning
abilities and explore different strategies. The performance of
the constant matrix with a weight of 0.2 outperforms that of
matrices with weights of 0.5, 0.8, and 1.0, further support-
ing this viewpoint. In the middle and later stages of reason-
ing, MTQA has already conducted a deep exploration of the
solution space for the problem. However, it may forget the
reasoning process and conclusions from the earlier thought
nodes, potentially generating strategies that are repetitive.
Therefore, it is necessary to expose more conclusions and
thought processes, allowing the LLM to actively avoid gen-
erating strategies that have already been produced.
MoT Size Analysis We next discuss the impact of MoT
size on its performance and computational overhead. Specif-
ically, we examine how different matrix sizes affect MTQA’s
reasoning performance (F1 score and EM) and its time cost
(Time). The considered matrix sizes include six configura-
tions: 2×2,2×3,3×3,3×4,4×4, and 4×5. The weight
matrix for each configuration is set using the Vert&Hor-0.1
method described above.
2x2 2x3 3x3 3x4 4x4 4x50.20.30.40.5Scores F1  EM
1234567
Time (min)
Matrix size Time
Figure 4: Comparison of F1 and EM scores for different ma-
trix sizes.
As shown in Figure 4, the experimental results indicate
that as the matrix size increases, the model’s F1 and EM
scores exhibit a gradual upward trend. Specifically, the F1
score increases from 0.392 with a 2×2matrix to 0.452 with
a3×4matrix. However, larger matrices ( 4×4,4×5) sta-
bilize around 0.45 or even show a slight decline. The EM
score follows a similar trend. This suggests that, for typi-
cal multi-hop problems, smaller matrices may lack sufficient
depth and breadth in exploring solution strategies; thus, en-

larging the thought matrix is an effective way to improve
reasoning performance. Nevertheless, this approach is not
absolute—once the matrix reaches a certain size, the strate-
gies supported by the available knowledge may have been
fully explored. Forcing the LLM to continue exploration at
this point may lead to hallucinations and inaccurate answers.
Furthermore, as the matrix size increases, the reasoning
time also rises significantly—from 1.324 minutes for a 2×2
matrix to 5.483 minutes for a 4×5matrix. Therefore, con-
sidering both performance and computational cost, the 3×4
matrix achieves the best F1 and EM scores while keeping the
reasoning time relatively short, making it a favorable choice.
For extremely complex problems or those involving a large
volume of knowledge, a larger thought matrix may be con-
sidered.
1.274
1.736
2.539
3.186
4.512
5.6021.398
1.654
2.702
3.351
4.359
5.3851.186
1.821
2.617
3.207
4.601
5.5191.309
1.701
2.456
3.095
4.287
5.2941.451
1.913
2.798
3.399
4.475
5.6711.243
1.584
2.581
3.243
4.3945.411.367
1.772
2.663
3.124
4.251
5.3271.215
1.599
2.424
3.288
4.623
5.5541.332
1.895
2.735
3.1594.415.2831.289
1.668
2.509
3.374
4.338
5.6381.324
1.721
2.618
3.224
4.438
5.4831.415
1.7892.683.216
4.569
5.471
0.384
0.403
0.415
0.424
0.425
0.4230.387
0.4010.410.419
0.415
0.4230.405
0.412
0.423
0.431
0.433
0.4280.394
0.406
0.417
0.423
0.424
0.4230.390.402
0.4140.420.418
0.4220.376
0.394
0.402
0.413
0.415
0.4120.408
0.413
0.426
0.437
0.435
0.4330.391
0.407
0.421
0.432
0.4340.430.382
0.398
0.415
0.429
0.431
0.4280.383
0.392
0.409
0.423
0.421
0.4320.416
0.424
0.438
0.452
0.451
0.4530.4070.410.419
0.434
0.435
0.437
0.257
0.268
0.274
0.283
0.285
0.2910.262
0.271
0.276
0.282
0.281
0.2790.278
0.283
0.291
0.304
0.306
0.2980.269
0.277
0.286
0.294
0.292
0.2930.261
0.266
0.275
0.287
0.289
0.2830.254
0.261
0.268
0.279
0.281
0.2790.259
0.265
0.281
0.286
0.284
0.2850.256
0.266
0.274
0.283
0.285
0.2860.251
0.263
0.271
0.279
0.281
0.2770.243
0.257
0.268
0.274
0.275
0.2730.294
0.303
0.309
0.318
0.312
0.3150.279
0.284
0.287
0.293
0.287
0.289UniformGaussianConst-0.2Const-0.5Const-0.8Const-1.0Vert-0.1Vert-0.2Hor-0.1Hor-0.2Vert&Hor-0.1Vert&Hor-0.2 Method
1.1802.0802.9803.8804.780
Time(min)
UniformGaussianConst-0.2Const-0.5Const-0.8Const-1.0Vert-0.1Vert-0.2Hor-0.1Hor-0.2Vert&Hor-0.1Vert&Hor-0.2Method
0.3760.3910.4070.4220.438
F1 Scorce2x2
2x3
3x3
3x4
4x4
4x5UniformGaussianConst-0.2Const-0.5Const-0.8Const-1.0Vert-0.1Vert-0.2Hor-0.1Hor-0.2Vert&Hor-0.1Vert&Hor-0.2Method
Matrix Size0.2430.2580.2730.2880.303
EM
Figure 5: Comparison of MTQA performance under differ-
ent matrix sizes and different weight setting methods.Joint Analysis After obtaining preliminary conclusions
for the weight matrix and MoT size individually, we further
investigate the interaction between these two hyperparame-
ters within the same experimental framework. The experi-
mental results, shown in Figure 5, lead us to the following
conclusions:
Second, in terms of time overhead, regardless of the
weight configuration, the reasoning time increases almost
monotonically with the number of rows and columns in the
matrix. At the 2×2size, the time ranges from 1.0 to 1.4
minutes, while at 4×5it rises to approximately 5.4–6.0 min-
utes. The growth is nearly linear in the early stages, but once
the size exceeds 3×4, the time cost increases sharply. In
contrast, the F1 and EM scores show no notable improve-
ment, and some weight configurations even result in de-
clines. Therefore, a 3×4size is the optimal choice in most
scenarios.
Overall, the 3×4configuration combined with Vert&Hor-
0.1 is the optimal trade-off: on HotpotQA, it achieves an
F1 score of 0.452 and an EM score of 0.318 with an av-
erage reasoning time of only 3.2 minutes. This experiment
further confirms two key conclusions: (i) the careful design
of the weight matrix is more beneficial than blindly enlarg-
ing the thought matrix, with small-step vertical–horizontal
increments effectively activating diverse reasoning within a
column without excessively leaking historical information;
and (ii) the thought matrix size has a “sweet spot”—a suffi-
cient number of rows and columns ensures adequate breadth
and depth in strategy exploration, but beyond 3×4the
marginal gains diminish while time costs rise sharply. Based
on these findings, we adopt 3×4+ Vert&Hor-0.1 as the
default MTQA configuration in all comparative experiments
above.
Time Complexity Analysis We compare the time com-
plexity of MTQA with that of the best-performing baseline,
RATT, from two perspectives. First, we compare the reason-
ing time costs of MTQA and RATT on the UltraDomain-
Biology and HotpotQA datasets. Second, we provide a the-
oretical proof of their time costs under the same matrix size
configuration.
Method
DatasetMTQA RATT
Biology HotpotOA Biology HotpotOA
125 times 402.94 394.56 2804.25 2743.58
average 3.22 3.16 22.43 21.95
Table 4: Comparison of time costs between MTQA and
RATT.
As shown in Table 4, both methods were tested 125 times
on the Biology dataset. The average reasoning time for
MTQA was 3.22 minutes, while RATT required 22.43 min-
utes. Similarly, on the HotpotQA dataset, the average rea-
soning time for MTQA was 3.16 minutes, while RATT re-
quired 21.95 minutes. The reasoning time of MTQA is ap-
proximately one-seventh of that of RATT. This demonstrates
that the MTQA method significantly reduces computational

overhead and provides efficient reasoning capabilities when
handling complex question-answering tasks.
Next, we continue to analyze the time complexity compo-
sition of the two methods. Let the constructed MTQA have
mrows and ncolumns, and correspondingly, the RATT tree
hasnlayers, each with mnodes. Let kbe the average time
to call the LLM to get a response, and rbe the time to call
RAG. Then:
TRATT =O(n(2m+1)k)+O(mnr) =O(n((2m+1)k+mr))
(12)
TMTQA =O(mnk) +O(2nr) =O(n(mk+ 2r))(13)
In RATT, each layer involves calling the LLM mtimes
to generate thought nodes, followed by mcalls to the LLM
andmcalls to RAG to refine the thought nodes. Finally,
a single LLM call is made to generate the summary node.
In contrast, in MTQA, we simplify this process by calling
RAG once to construct the knowledge unit, followed by m
LLM calls to generate the thought nodes enhanced by the
knowledge unit. The process is then completed by one call
to RAG and one call to LLM for refinement and summariza-
tion. Both experimental and theoretical results demonstrate
the high efficiency and performance of our approach.
Conclusion
In this paper, we propose the Thinking Matrix Ques-
tion Answering framework (MTQA) for complex question-
answering tasks. The framework, centered around the
novel MoT reasoning paradigm, integrates knowledge graph
triples and raw text into knowledge units embedded within
a retrieval-augmented process. The column-cell communi-
cation mechanism triggers multi-branch thinking laterally
and accumulates deep strategies vertically, overcoming the
common limitations of the CoT’s single path and ToT’s
in-layer redundancy. Through extensive experiments across
four mainstream datasets, evaluated with multiple metrics
and dimensions, we demonstrate the advantages of MTQA
over existing methods. Additionally, we reduce the reason-
ing time to one-seventh of the best baseline, balancing both
accuracy and efficiency. This advantage is particularly evi-
dent in high-level contextual understanding and other open,
abstract tasks.
Limitations: Although MTQA and the core MoT
paradigm perform excellently on complex question-
answering tasks, there are still several limitations that re-
quire further exploration. These include the fact that the ef-
fectiveness of the reasoning process and the final responses
still depend on the quality of task prompts and the external
knowledge base, as well as the high computational demands
and the need for manually configured hyperparameters. For
future work, we plan to extend the MoT paradigm to various
LLM reasoning and alignment tasks, incorporating multi-
modal information and adaptive communication weight ma-
trices. We aim to validate its generalizability and robustness
in broader application scenarios and explore more suitable
RAG systems for reasoning structures to achieve higher-
quality reasoning enhancement and knowledge correction.References
Asai, A.; Wu, Z.; Wang, Y .; Sil, A.; and Hajishirzi, H. 2024.
Self-rag: Learning to retrieve, generate, and critique through
self-reflection.
Cuconasu, F.; Trappolini, G.; Siciliano, F.; Filice, S.; Cam-
pagnano, C.; Maarek, Y .; Tonellotto, N.; and Silvestri, F.
2024. The power of noise: Redefining retrieval for rag sys-
tems. In Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Information
Retrieval , 719–729.
Ding, H.; Pang, L.; Wei, Z.; Shen, H.; and Cheng, X. 2024.
Retrieve only when it needs: Adaptive retrieval augmenta-
tion for hallucination mitigation in large language models.
arXiv preprint arXiv:2402.10612 .
Ding, R.; Zhang, C.; Wang, L.; Xu, Y .; Ma, M.; Zhang, W.;
Qin, S.; Rajmohan, S.; Lin, Q.; and Zhang, D. 2023. Every-
thing of thoughts: Defying the law of penrose triangle for
thought generation. arXiv preprint arXiv:2311.04254 .
Dong, G.; Zhu, Y .; Zhang, C.; Wang, Z.; Wen, J.-R.; and
Dou, Z. 2025. Understand what LLM needs: Dual prefer-
ence alignment for retrieval-augmented generation. In Pro-
ceedings of the ACM on Web Conference 2025 , 4206–4225.
Edge, D.; Trinh, H.; Cheng, N.; Bradley, J.; Chao, A.;
Mody, A.; Truitt, S.; Metropolitansky, D.; Ness, R. O.; and
Larson, J. 2024. From local to global: A graph rag ap-
proach to query-focused summarization. arXiv preprint
arXiv:2404.16130 .
Fan, W.; Ding, Y .; Ning, L.; Wang, S.; Li, H.; Yin, D.; Chua,
T.-S.; and Li, Q. 2024. A survey on rag meeting llms: To-
wards retrieval-augmented large language models. In Pro-
ceedings of the 30th ACM SIGKDD conference on knowl-
edge discovery and data mining , 6491–6501.
Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang, R.; Xu, R.;
Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; et al. 2025. Deepseek-r1:
Incentivizing reasoning capability in llms via reinforcement
learning. arXiv preprint arXiv:2501.12948 .
Ho, X.; Nguyen, A.-K. D.; Sugawara, S.; and Aizawa,
A. 2020. Constructing a multi-hop qa dataset for com-
prehensive evaluation of reasoning steps. arXiv preprint
arXiv:2011.01060 .
Jiang, Z.; Xu, F. F.; Gao, L.; Sun, Z.; Liu, Q.; Dwivedi-Yu,
J.; Yang, Y .; Callan, J.; and Neubig, G. 2023. Active retrieval
augmented generation. In Proceedings of the 2023 Confer-
ence on Empirical Methods in Natural Language Process-
ing, 7969–7992.
Kojima, T.; Gu, S. S.; Reid, M.; Matsuo, Y .; and Iwasawa,
Y . 2022. Large language models are zero-shot reason-
ers.Advances in neural information processing systems , 35:
22199–22213.
Kwiatkowski, T.; Palomaki, J.; Redfield, O.; Collins, M.;
Parikh, A.; Alberti, C.; Epstein, D.; Polosukhin, I.; Devlin,
J.; Lee, K.; et al. 2019. Natural questions: a benchmark for
question answering research. Transactions of the Associa-
tion for Computational Linguistics , 7: 453–466.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,

T.; et al. 2020. Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in neural infor-
mation processing systems , 33: 9459–9474.
Li, D.; Rawat, A. S.; Zaheer, M.; Wang, X.; Lukasik, M.;
Veit, A.; Yu, F.; and Kumar, S. 2022. Large language
models with controllable working memory. arXiv preprint
arXiv:2211.05110 .
Minaee, S.; Mikolov, T.; Nikzad, N.; Chenaghlu, M.; Socher,
R.; Amatriain, X.; and Gao, J. 2024. Large language models:
A survey. arXiv preprint arXiv:2402.06196 .
Qian, H.; Liu, Z.; Zhang, P.; Mao, K.; Lian, D.; Dou, Z.; and
Huang, T. 2025. Memorag: Boosting long context process-
ing with global memory-enhanced retrieval augmentation.
InProceedings of the ACM on Web Conference 2025 , 2366–
2377.
Qin, H. X.; Jin, S.; Gao, Z.; Fan, M.; and Hui, P. 2024. Char-
acterMeet: Supporting creative writers’ entire story charac-
ter construction processes through conversation with LLM-
powered chatbot avatars. In Proceedings of the 2024 CHI
Conference on Human Factors in Computing Systems , 1–19.
Saxena, A.; Tripathi, A.; and Talukdar, P. 2020. Improving
multi-hop question answering over knowledge graphs using
knowledge base embeddings. In Proceedings of the 58th
annual meeting of the association for computational linguis-
tics, 4498–4507.
Shojaee*†, P.; Mirzadeh*, I.; Alizadeh, K.; Horton, M.; Ben-
gio, S.; and Farajtabar, M. 2025. The Illusion of Thinking:
Understanding the Strengths and Limitations of Reasoning
Models via the Lens of Problem Complexity.
Sriramanan, G.; Bharti, S.; Sadasivan, V . S.; Saha, S.; Kat-
takinda, P.; and Feizi, S. 2024. Llm-check: Investigat-
ing detection of hallucinations in large language models.
Advances in Neural Information Processing Systems , 37:
34188–34216.
Tan, J.; Dou, Z.; Zhu, Y .; Guo, P.; Fang, K.; and Wen, J.-R.
2024. Small models, big insights: Leveraging slim proxy
models to decide when and what to retrieve for llms. arXiv
preprint arXiv:2402.12052 .
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A. 2022. Interleaving retrieval with chain-of-thought rea-
soning for knowledge-intensive multi-step questions. arXiv
preprint arXiv:2212.10509 .
Wang, P.; Wang, Z.; Li, Z.; Gao, Y .; Yin, B.; and Ren, X.
2023. Scott: Self-consistent chain-of-thought distillation.
arXiv preprint arXiv:2305.01879 .
Wang, Z.; Liu, A.; Lin, H.; Li, J.; Ma, X.; and Liang, Y .
2024. Rat: Retrieval augmented thoughts elicit context-
aware reasoning in long-horizon generation. arXiv preprint
arXiv:2403.05313 .
Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Xia, F.;
Chi, E.; Le, Q. V .; Zhou, D.; et al. 2022. Chain-of-
thought prompting elicits reasoning in large language mod-
els.Advances in neural information processing systems , 35:
24824–24837.Wu, Z.; Tang, F.; Zhao, M.; and Li, Y . 2024. Kgv: Integrat-
ing large language models with knowledge graphs for cy-
ber threat intelligence credibility assessment. arXiv preprint
arXiv:2408.08088 .
Xu, S.; Pang, L.; Shen, H.; Cheng, X.; and Chua, T.-S. 2024.
Search-in-the-chain: Interactively enhancing large language
models with search for knowledge-intensive tasks. In Pro-
ceedings of the ACM Web Conference 2024 , 1362–1373.
Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y .; Cohen, W. W.;
Salakhutdinov, R.; and Manning, C. D. 2018. HotpotQA: A
dataset for diverse, explainable multi-hop question answer-
ing. arXiv preprint arXiv:1809.09600 .
Yao, S.; Yu, D.; Zhao, J.; Shafran, I.; Griffiths, T.; Cao, Y .;
and Narasimhan, K. 2023a. Tree of thoughts: Deliberate
problem solving with large language models. Advances in
neural information processing systems , 36: 11809–11822.
Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan,
K.; and Cao, Y . 2023b. React: Synergizing reasoning and
acting in language models. In International Conference on
Learning Representations (ICLR) .
Zhang, J.; Wang, X.; Ren, W.; Jiang, L.; Wang, D.; and Liu,
K. 2025. Ratt: A thought structure for coherent and correct
llm reasoning. In Proceedings of the AAAI Conference on
Artificial Intelligence , volume 39, 26733–26741.
Zhang, Y .; Sun, R.; Chen, Y .; Pfister, T.; Zhang, R.; and Arik,
S. 2024. Chain of agents: Large language models collaborat-
ing on long-context tasks. Advances in Neural Information
Processing Systems , 37: 132208–132237.
Zhou, W.; Zhang, S.; Poon, H.; and Chen, M. 2023. Context-
faithful prompting for large language models. arXiv preprint
arXiv:2303.11315 .
Zhuang, Y .; Yu, Y .; Wang, K.; Sun, H.; and Zhang, C. 2023.
Toolqa: A dataset for llm question answering with external
tools. Advances in Neural Information Processing Systems ,
36: 50117–50143.
Main prompt template
In this section, we present various LLM data processing
prompt templates mentioned in the Methodology section, in-
cluding Figures 6, 7, 9, and 10.
Prompt for graph construct
As shown in Figure 6, the graph construction prompt tem-
plate is used to build a knowledge graph from text. It con-
sists of several steps: first identifying entities and their de-
scriptions, then extracting relationships between entities,
followed by capturing high-level thematic keywords, and fi-
nally outputting everything in a structured list. It emphasizes
strict formatting without additional explanations or sum-
maries.
Prompt for extract key
The prompt template, as shown in Figure 7, is used to ex-
tract both high-level and low-level keywords from the user’s
query. High-level keywords focus on overarching concepts

Goal :
Steps  1:
Identify all entities.  
For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_descripti on>Given a list of text documents and entity types that may be relevant to this activity, identify all entities of those types f rom the text and all relationships 
between identified entities. and output in the specified format. Please do not provide any explanations, summaries, or other content.
Steps  2:
From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to ea ch 
other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high -level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather 
than specific details
Format each relationship as 
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}< relationship_description>{tuple_delimiter}<relationship_
keywords>{tuple_delimiter}<relationship_strength>)
Steps  3:
Identify high -level key words that summarize the main concepts, themes, or topics of the entire text. These should 
capture the overarching ideas present in the document.
Format the content -level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)
Steps  4:
Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use 
{record_deli miter} as the list delimiter.
Steps  5:
When finished, output {completion_delimiter}
Note: Please only output entities and relationships that conform to the format, and do not have any other text.
Real Data
Entity_types: {entity_types}
Text: {input_text }
Output:
Graph Construct 
PromptGoal :
Steps  1:
Identify all entities.  
For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_descripti on>Given a list of text documents and entity types that may be relevant to this activity, identify all entities of those types f rom the text and all relationships 
between identified entities. and output in the specified format. Please do not provide any explanations, summaries, or other content.
Steps  2:
From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to ea ch 
other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high -level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather 
than specific details
Format each relationship as 
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}< relationship_description>{tuple_delimiter}<relationship_
keywords>{tuple_delimiter}<relationship_strength>)
Steps  3:
Identify high -level key words that summarize the main concepts, themes, or topics of the entire text. These should 
capture the overarching ideas present in the document.
Format the content -level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)
Steps  4:
Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use 
{record_deli miter} as the list delimiter.
Steps  5:
When finished, output {completion_delimiter}
Note: Please only output entities and relationships that conform to the format, and do not have any other text.
Real Data
Entity_types: {entity_types}
Text: {input_text }
Output:
Graph Construct 
PromptFigure 6: Graph Construct Prompt.

Goal :Role:
You are a helpful assistant tasked with identifying both high -level and low -level keywords in the user's query.
Given the query, list both high -level and low -level keywords. High -level keywords focus on overarching concepts or themes, 
while low -level keywords focus on specific entities, details, or concrete terms.
Instructions:
- Output the keywords in JSON format.
- Only return results in JSON format, and do not contain other text.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.
Real Data
Query: {query}
Output:
Extract Keywords 
PromptGoal :Role:
You are a helpful assistant tasked with identifying both high -level and low -level keywords in the user's query.
Given the query, list both high -level and low -level keywords. High -level keywords focus on overarching concepts or themes, 
while low -level keywords focus on specific entities, details, or concrete terms.
Instructions:
- Output the keywords in JSON format.
- Only return results in JSON format, and do not contain other text.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.
Real Data
Query: {query}
Output:
Extract Keywords 
PromptFigure 7: Extract Key Prompt.
or themes, while low-level keywords focus on specific en-
tities, details, or concrete terms. The template requires re-
turning the results in JSON format, containing two keys:
“high level keywords” for overarching concepts or themes,
and “low level keywords” for specific entities or details. It
emphasizes returning only JSON results, without any addi-
tional text.
Prompt for initial thought node
The Initial Thought Node prompt template, as shown in Fig-
ure 8, is used to generate structured answers based on pre-
viously provided information and current instructions. The
template requires the answer to be split into multiple para-
graphs and directly respond to the given instructions, with-
out adding extra explanations or introductions. It emphasizes
using the previous context and communication information
to ensure the structure and consistency of the answer. The
output of the template will include context and communi-
cation content, following the direct response to the instruc-
tions.
Prompt for Summary Node
The Summary Node prompt template, as shown in Figure 8,
is used to generate more detailed and logically coherent an-
swers based on the original answer and an additional supple-
mentary answer. The template requires maintaining a struc-
tured format with multiple paragraphs. First, review the orig-
inal answer, and if any part is correct and does not need fur-
ther details, retain it unchanged. For parts that need improve-
ment, enhance them by incorporating information from the
supplementary answer to make the response more compre-
hensive and accurate. If any errors are found, they shouldbe corrected while ensuring that the revised content remains
logically coherent. Finally, the generated answer should use
language consistent with the context and avoid summarizing
or replacing words on your own.

IMPORTANT:
Try to answer this question/instruction with step -by-step thoughts and make the answer more structural.
Use `\n\n` to split the answer into several paragraphs.
Just respond to the instruction directly. DO NOT add additional explanations or introducement in the answer unless you 
are asked to.
Initial Thought Node / 
Communication PromptThe **context** is the information I obtained from the local knowledge base, and the ** communication** is the answer provided 
by a previous agent (which may contain errors). Please assess whether the ** communication** is correct based on the 
**context** (if history exists), and then provide an answer. Please note that the generated answer should use the same words and 
descriptions as the * * context * * to answer the question, avoiding summarizing or replacing words on your own.
**context**:
{context}
**communication** ：
{communication}
Output：IMPORTANT:
Try to answer this question/instruction with step -by-step thoughts and make the answer more structural.
Use `\n\n` to split the answer into several paragraphs.
Just respond to the instruction directly. DO NOT add additional explanations or introducement in the answer unless you 
are asked to.
Initial Thought Node / 
Communication PromptThe **context** is the information I obtained from the local knowledge base, and the ** communication** is the answer provided 
by a previous agent (which may contain errors). Please assess whether the ** communication** is correct based on the 
**context** (if history exists), and then provide an answer. Please note that the generated answer should use the same words and 
descriptions as the * * context * * to answer the question, avoiding summarizing or replacing words on your own.
**context**:
{context}
**communication** ：
{communication}
Output：Figure 8: Initial Thought Node.
IMPORTANT:
Ensure the revised answer maintains a structured format (multiple paragraphs with subtitles) for better clarity. Separate 
the paragraphs with `\n\n` characters. Output only the enhanced answer directly, without any extra explanations or 
announcements unless specifically requested.
Summary Node  PromptBased on the original answer and an additional supplementary answer, generate a response that is richer in detail and logical ly 
coherent. Review the original answer:
1. If any part of the answer is correct and requires no further details, retain that portion unchanged and output it directly  as it is.
2. For parts that may be improved or lack necessary details, enhance them by integrating information from the supplementary 
answer to make the response more comprehensive and accurate.
3. If you identify any errors within the answers, correct these errors while ensuring that the revised content remains logica lly 
coherent.
4. Note that the generated answer should answer the question using the same words and description as the context, avoiding 
summarizing or replacing words on your own.
question: {question}
Original Answer: {previous Summary Node}
Supplementary Answer: {Thought Node column}
context: {context}IMPORTANT:
Ensure the revised answer maintains a structured format (multiple paragraphs with subtitles) for better clarity. Separate 
the paragraphs with `\n\n` characters. Output only the enhanced answer directly, without any extra explanations or 
announcements unless specifically requested.
Summary Node  PromptBased on the original answer and an additional supplementary answer, generate a response that is richer in detail and logical ly 
coherent. Review the original answer:
1. If any part of the answer is correct and requires no further details, retain that portion unchanged and output it directly  as it is.
2. For parts that may be improved or lack necessary details, enhance them by integrating information from the supplementary 
answer to make the response more comprehensive and accurate.
3. If you identify any errors within the answers, correct these errors while ensuring that the revised content remains logica lly 
coherent.
4. Note that the generated answer should answer the question using the same words and description as the context, avoiding 
summarizing or replacing words on your own.
question: {question}
Original Answer: {previous Summary Node}
Supplementary Answer: {Thought Node column}
context: {context}
Figure 9: Initial Thought Node.

Question
How does human intervention impact the balance of biodiversity?
Answer 1:
Human activities have significantly impacted the balance of biodiversity...
Human activities significantly impact biodiversity,primarily through habitat loss,...Answer 2:
Comprehensiveness:
Winner: Answer 1
Explanation:Answer 1 provides a thorough analysis of the various human activities impacting 
biodiversity,covering key aspects such as habitat loss,climate change,overexploitation,invasive 
species,pollution,and species extinction in detail.Each topic is expanded upon with specific 
examples and suggested solutions,offering a nuanced understanding of how these factors 
collectively affect biodiversity.Answer 2,while also comprehensive,does not delve into the same 
level of detail for each aspect,nor does it provide specific examples for all points mentioned.
Accuracy:
Winner: Answer 1
Explanation:Answer 1 accurately presents facts and examples related to human impacts on 
biodiversity,maintaining logical consistency and reliable data across the discussion of various 
influences.The examples provided,such as the Auroch and the blue shark population decline,are 
appropriately contextualized and illustrate the points made.Answer 2 also presents accurate 
information,but it is less specific and lacks concrete examples in some areas,which reduces its 
overall factual reliability.
Empowerment:
Winner: Answer 1
Explanation:Answer 1 empowers the reader by not only highlighting the issues caused by 
human intervention but also by presenting clear solutions and actionable steps for conservation 
and sustainability.The structured format aids in understanding complex relationships in 
biodiversity impact.Answer 2 provides valuable insights as well but presents them in a more 
general manner that may not be as effective in motivating and guiding the reader towards 
informed actions.
Overall Winner:
Winner: Answer 1
Explana tion:Overall,Answer 1 excels in comprchensiveness,accuracy,and cmpowerment. It 
provides a delailed exploration of  how human interventions impact biodiversity with relevant 
examples,clear causc -and-effect relationships,and conerete steps for addressal.While Answer 2 
also refects important points about human harm and po tential benefits,  it lacks the depth and 
specificity required to match the clarity and instructive nalure of Answer 1.Question
How does human intervention impact the balance of biodiversity?
Answer 1:
Human activities have significantly impacted the balance of biodiversity...
Human activities significantly impact biodiversity,primarily through habitat loss,...Answer 2:
Comprehensiveness:
Winner: Answer 1
Explanation:Answer 1 provides a thorough analysis of the various human activities impacting 
biodiversity,covering key aspects such as habitat loss,climate change,overexploitation,invasive 
species,pollution,and species extinction in detail.Each topic is expanded upon with specific 
examples and suggested solutions,offering a nuanced understanding of how these factors 
collectively affect biodiversity.Answer 2,while also comprehensive,does not delve into the same 
level of detail for each aspect,nor does it provide specific examples for all points mentioned.
Accuracy:
Winner: Answer 1
Explanation:Answer 1 accurately presents facts and examples related to human impacts on 
biodiversity,maintaining logical consistency and reliable data across the discussion of various 
influences.The examples provided,such as the Auroch and the blue shark population decline,are 
appropriately contextualized and illustrate the points made.Answer 2 also presents accurate 
information,but it is less specific and lacks concrete examples in some areas,which reduces its 
overall factual reliability.
Empowerment:
Winner: Answer 1
Explanation:Answer 1 empowers the reader by not only highlighting the issues caused by 
human intervention but also by presenting clear solutions and actionable steps for conservation 
and sustainability.The structured format aids in understanding complex relationships in 
biodiversity impact.Answer 2 provides valuable insights as well but presents them in a more 
general manner that may not be as effective in motivating and guiding the reader towards 
informed actions.
Overall Winner:
Winner: Answer 1
Explana tion:Overall,Answer 1 excels in comprchensiveness,accuracy,and cmpowerment. It 
provides a delailed exploration of  how human interventions impact biodiversity with relevant 
examples,clear causc -and-effect relationships,and conerete steps for addressal.While Answer 2 
also refects important points about human harm and po tential benefits,  it lacks the depth and 
specificity required to match the clarity and instructive nalure of Answer 1.Figure 10: example.
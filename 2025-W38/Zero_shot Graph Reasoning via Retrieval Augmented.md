# Zero-shot Graph Reasoning via Retrieval Augmented Framework with LLMs

**Authors**: Hanqing Li, Kiran Sheena Jyothi, Henry Liang, Sharika Mahadevan, Diego Klabjan

**Published**: 2025-09-16 06:58:58

**PDF URL**: [http://arxiv.org/pdf/2509.12743v1](http://arxiv.org/pdf/2509.12743v1)

## Abstract
We propose a new, training-free method, Graph Reasoning via Retrieval
Augmented Framework (GRRAF), that harnesses retrieval-augmented generation
(RAG) alongside the code-generation capabilities of large language models
(LLMs) to address a wide range of graph reasoning tasks. In GRRAF, the target
graph is stored in a graph database, and the LLM is prompted to generate
executable code queries that retrieve the necessary information. This approach
circumvents the limitations of existing methods that require extensive
finetuning or depend on predefined algorithms, and it incorporates an error
feedback loop with a time-out mechanism to ensure both correctness and
efficiency. Experimental evaluations on the GraphInstruct dataset reveal that
GRRAF achieves 100% accuracy on most graph reasoning tasks, including cycle
detection, bipartite graph checks, shortest path computation, and maximum flow,
while maintaining consistent token costs regardless of graph sizes. Imperfect
but still very high performance is observed on subgraph matching. Notably,
GRRAF scales effectively to large graphs with up to 10,000 nodes.

## Full Text


<!-- PDF content starts -->

Zero-shot Graph Reasoning via Retrieval Augmented Framework with
LLMs
Hanqing Li1, Kiran Sheena Jyothi2,
Henry Liang3,Sharika Mahadevan4,Diego Klabjan1
1Northwestern University,2EXL Service,3Vail Systems,4Netflix
Correspondence:hanqingli2025@u.northwestern.edu
Abstract
We propose a new, training-free method,
Graph Reasoning via Retrieval Augmented
Framework (GRRAF), that harnesses retrieval-
augmented generation (RAG) alongside the
code-generation capabilities of large language
models (LLMs) to address a wide range of
graph reasoning tasks. In GRRAF, the target
graph is stored in a graph database, and the
LLM is prompted to generate executable code
queries that retrieve the necessary information.
This approach circumvents the limitations of
existing methods that require extensive finetun-
ing or depend on predefined algorithms, and
it incorporates an error feedback loop with a
time-out mechanism to ensure both correctness
and efficiency. Experimental evaluations on
the GraphInstruct dataset reveal that GRRAF
achieves 100% accuracy on most graph reason-
ing tasks, including cycle detection, bipartite
graph checks, shortest path computation, and
maximum flow, while maintaining consistent
token costs regardless of graph sizes. Imperfect
but still very high performance is observed on
subgraph matching. Notably, GRRAF scales
effectively to large graphs with up to 10,000
nodes.
1 Introduction
Graph reasoning plays a pivotal role in modeling
and understanding complex systems across numer-
ous domains (Wu et al., 2020). Graphs naturally
represent entities and their interrelations in areas
such as social networks, transportation systems,
biological networks, and communication infras-
tructures. Graph reasoning tasks like determin-
ing connectivity, detecting cycles, and finding the
shortest path are not only central to theoretical com-
puter science but also have practical implications
in network optimization, anomaly detection, deci-
sion support systems, etc (Scarselli et al., 2008).
However, addressing these tasks requires a deep un-
derstanding of graph topology combined with pre-
Figure 1: A schematic representation of the GRRAF
concept. When a user asks a graph reasoning question,
the LLM generates code to query the target graph stored
in a graph database, retrieves the answer, and presents
it as the response. An error feedback loop is integrated
into GRRAF to prompt the LLM to refine the code
whenever execution or time-out errors occur.
cise computational procedures, underscoring the
critical challenge of developing efficient graph rea-
soning methods in contemporary machine learning
research (Zhao et al., 2024).
Large language models (LLMs) have demon-
strated an impressive capacity for multi-step reason-
ing, which enables them to interpret complex graph-
related questions expressed in natural language and
generate human-readable responses (Guo et al.,
2023). Several recent studies have leveraged LLMs
to tackle graph reasoning problems by converting
graph structures into textual representations or la-
tent embeddings through graph neural networks
(GNNs), thereby exploiting the powerful natural
language reasoning capabilities of LLMs (Perozzi
et al., 2024; Guo et al., 2023; Zhang, 2023; Wang
et al., 2024a; Fatemi et al., 2024; Skianis et al.,
2024; Lin et al., 2024). However, even when ad-
vanced prompting techniques are employed, these
1arXiv:2509.12743v1  [cs.AI]  16 Sep 2025

methods tend to perform poorly on fundamental
graph reasoning tasks, such as evaluating connec-
tivity or identifying the shortest path, with average
accuracies ranging from 20% to 60%. Alternative
approaches that achieve higher accuracy typically
either require extensive finetuning—which results
in poor performance on out-of-domain questions
(Chen et al., 2024; Zhang, 2023)—or rely on pre-
defined algorithms as input, thereby limiting their
ability to address unseen tasks (Hu et al., 2024).
To address these limitations, we introduce a
training-free and zero-shot method,the Graph Rea-
soning via Retrieval Augmented Framework(GR-
RAF), that leverages retrieval-augmented gener-
ation (RAG) (Lewis et al., 2020) alongside the
code-writing capabilities of large language mod-
els. In GRRAF, the target graph is stored in a
graph database, and the LLM is prompted to gen-
erate appropriate queries, written as code, that ex-
tract the desired answer by retrieving relevant in-
formation from the database. This strategy har-
nesses the LLM’s robust reasoning ability and its
proficiency in generating executable code, thereby
achieving high accuracy on a range of graph reason-
ing tasks without requiring additional finetuning
or predefined algorithms. In addition, we incor-
porate an error feedback loop combined with a
time-out mechanism to ensure that the LLM pro-
duces correct queries in a time-efficient manner.
Furthermore, since accurate code reliably yields
the correct answer regardless of the graph’s size,
GRRAF can easily scale for polynomial problems
to accommodate larger graphs without a drop in
accuracy. In GRRAF, we use Neo4j, an interactive
graph database, and NetworkX, a Python library
for graphs. GRRAF accepts the target graph as
either plain text or data already stored in Neo4j,
specified in the prompt by the graph file name. In
the former case, the prompt must specify if Neo4j
or NetworkX is to be used. The LLM then must ei-
ther create code to insert the graph specified in the
prompt to Neo4j or to a NetworkX graph object.
GRRAF offers a fully automated, end-to-end
framework for handling graph-reasoning problems
written entirely in text. By leveraging the world
knowledge encoded in LLMs, it generates cor-
rect code and returns accurate answers automat-
ically for a wide range of graph-reasoning tasks
expressed as natural-language questions. In addi-
tion, GRRAF establishes a foundation for future
work on real-world structured relational-inference
problems—ranging from knowledge-graph com-pletion to molecular analysis—that are naturally
represented as graph-structured data. An LLM user
could potentially accomplish the same by directly
prompting the LLM to create Python or Neo4j
queries for the task on hand. Our approach of-
fers the benefits of graph reading and loading, the
execution of the code with the error-feedback loop,
and the fallback approach.
Experimental results demonstrate that GRRAF
achieves 100% accuracy on many graph reasoning
tasks, outperforming state-of-the-art benchmarks.
Moreover, GRRAF is applicable to large graphs
containing up to 10,000 nodes, maintaining 100%
accuracy with no increase in token cost. Although
GRRAF only achieves 86.5% accuracy on sub-
graph matching, it still outperforms other state-
of-the-art methods. Our contributions are listed
below.
•Novel Graph Reasoning Approach:This
work introduces a new method that leverages
RAG to address graph reasoning tasks, such
as connectivity analyses, cycle detection, and
shortest path computations. It represents the
first application of RAG in the domain of
graph reasoning.
•Error Feedback Loop Innovation:The pa-
per introduces the integration of a time out
mechanism within an error feedback loop,
along with the dynamic refreshing of a prompt
to guide the LLM to produce more efficient
code. This mechanism enhances robustness
and efficiency of the generated query by pre-
venting an infinite loop.
•Scalable State-of-the-Art Performance:
The proposed method achieves state-of-the-art
accuracy and demonstrates exceptional scal-
ability, being the first to handle large graphs
effectively without significant degradation in
accuracy or substantial cost increases.
All implementations and datasets are available
inhttps://github.com/hanklee97121/GRRAF/
tree/main.
2 Related Works
2.1 Graph RAG
There exist numerous prior works that employ
graph data within RAG frameworks to enhance
the capabilities of LLMs, a paradigm often referred
2

Figure 2: GRRAF workflow. The retrieval component represents the interaction with the graph database through
code, while the generation component involves prompting an LLM to produce the output.
to as GraphRAG (Peng et al., 2024). These ap-
proaches retrieve graph elements containing rela-
tional knowledge relevant to a given query from a
pre-constructed graph database (Edge et al., 2024).
Several studies have contributed to the develop-
ment of open-source knowledge graph datasets for
GraphRAG (Auer et al., 2007; Suchanek et al.,
2007; Vrande ˇci´c and Krötzsch, 2014; Sap et al.,
2019; Liu and Singh, 2004; Bollacker et al., 2008).
Building on these datasets, many methods opt to
convert graphs to other easily retrievable forms,
such as text (Li et al., 2023; Huang et al., 2023; Yu
et al., 2023; Edge et al., 2024; Dehghan et al., 2024)
or vectors (He et al., 2024; Sarmah et al., 2024),
to improve the efficiency of query operations on
graph databases. To further enhance the quality of
retrieved data, several approaches optimize the re-
trieval process within GraphRAG by refining the re-
triever component (Delile et al., 2024; Zhang et al.,
2022a; Kim et al., 2023; Wold et al., 2023; Jiang
et al., 2023; Mavromatis and Karypis, 2024), opti-
mizing the retrieval paradigm (Wang et al., 2024b;
Sun et al., 2024c; Lin et al., 2019), and editing a
user query or the retrieved information (Jin et al.,
2024; LUO et al., 2024; Ma et al., 2025; Sun et al.,
2024a; Taunk et al., 2023; Yasunaga et al., 2021).
Furthermore, many methods enhance the answer
generation process of GraphRAG to ensure that
the LLM fully utilizes the retrieved graph data to
generate the correct answer (Dong et al., 2023;
Mavromatis and Karypis, 2022; Jiang et al., 2024;
Sun et al., 2024b; Zhang et al., 2022b; Zhu et al.,
2024; Wen et al., 2024; Shu et al., 2022; Baek
et al., 2023). However, these methods focus exclu-
sively on knowledge graphs and cannot be directlyapplied to solve graph reasoning questions. In con-
trast, GRRAF is the first method to employ RAG
for addressing graph reasoning questions on pure
graphs.
2.2 Graph Reasoning
Recent work has explored the use of LLMs to ad-
dress graph reasoning problems. Several meth-
ods rely solely on prompt engineering techniques
to enhance LLM reasoning capabilities on graphs
(Liu and Wu, 2023; Guo et al., 2023; Wang et al.,
2024a; Zhang et al., 2024; Fatemi et al., 2024; Wu
et al., 2024; Tang et al., 2025; Skianis et al., 2024;
Lin et al., 2024). Building on them, Perozzi et al.
(2024) integrate a trained graph neural network
(Scarselli et al., 2008) with an LLM to improve
its performance on graph reasoning tasks by en-
coding each graph into a token provided as input
to the LLM. Meanwhile, Zhang (2023) and Chen
et al. (2024) finetune an LLM with instructions
tailored to graph reasoning tasks to boost perfor-
mance. In another approach, Hu et al. (2024) pro-
pose a multi-agent solution for graph reasoning
problems by assigning an LLM agent to each node
and enabling communication among agents based
on a predefined algorithm. In contrast, GRRAF em-
ploys RAG to address graph reasoning problems
without extensive prompt engineering. This ap-
proach is training-free and thus unsupervised and
does not depend on any predefined algorithm. Fur-
thermore, unlike previous methods, the LLM in
GRRAF does not receive the entire graph as input;
consequently, the token usage remains independent
of graph size, thereby enabling efficient scalability
to very large graphs.
3

Figure 3: An illustrative example demonstrating the application of GRRAF to solve a shortest path question by
using NetworkX. GraphGin text is stored as an NetworkX object by code.
Task Node Size # of
Test
Graphs
Cycle Detection [2, 100] 400
Connectivity [2, 100] 400
Bipartite Graph Check [2, 100] 400
Topological Sort [2, 50] 400
Shortest Path [2, 100] 400
Maximum Triangle Sum [2, 25] 400
Maximum Flow [2, 50] 400
Subgraph Matching [2, 30] 400
Indegree Calculation [2, 50] 400
Outdegree Calculation [2, 50] 400
Table 1: The detailed information of GraphInstruct
dataset and two additional tasks (indegree calculation
and outdegree calculation). The subgraph matching task
is to verify if there exists a subgraph in Gthat is isomor-
phic to a given graphG′.
3 Method
In this section, we explain how GRRAF integrates
RAG to address graph reasoning questions and re-
trieve accurate answers. The entire workflow of
GRRAF is demonstrated in Figure 2. A graph rea-
soning question, denoted as Q, consists of two com-
ponents: a graph Gand a user prompt P. The graph
Grepresents the target graph associated with Qand
is stored either in Neo4j or as a NetworkX graph
object (code written by an LLM and executed by
an agent). The prompt Pcontains a graph-specific
question regarding G(e.g., “Does node 2 connect
to node 5?” or “What is the shortest path from node
5 to node 8?”). To enhance code generation by thelanguage model, we initially input Pinto the LLM,
requesting it to refine the prompt, clarify the format,
and eliminate redundant information. The resulting
refined prompt is denoted as P′. Then, the LLM
is prompted to generate a generic code template
Cthat addresses P′without incorporating graph-
specific details. For example, if P′states “Find the
shortest path from node 3 to node 5,” the template
Cencapsulates a generic shortest path algorithm
that does not include the specific node identifiers.
Additionally, we extract the schema S(compris-
ing of node properties and edge properties) from
the graph database using a hard-coded procedure.
This schema ensures that the LLM-generated code
utilizes correct variable names.
Subsequently, we provide P′,C, and Sto the
LLM and instruct it to generate the final code C′
that produces an answer Acorresponding to P′.
An error feedback loop is incorporated into this
process. If an error arises during the execution of
C′, the error message, along with C′, is supplied
back to the LLM, prompting it to produce a revised
version of the code. To promote the generation of
time-efficient code, given that multiple algorithms
with varying time complexities may be applicable,
we integrate a time-out mechanism within the error
feedback loop. Specifically, a time limit tis im-
posed on the execution of C′. If the execution time
exceeds t, the process is halted, and the LLM is
asked to modify C′so that it runs faster. If the feed-
back loop iterates more than ntimes, the system
reverts to using the original question Qas a prompt
to directly obtain the answer Afrom the LLM. This
forced exit is designed to prevent perpetual itera-
4

tions when addressing computationally intractable
NP-hard problems (e.g., substructure matching on
large graphs), where no modification of C′can
reduce the execution time below the thresholdt.
In the final step, the answer Ais provided to
the LLM to generate a reader-friendly natural lan-
guage response A0that addresses the graph reason-
ing question Q. An example of solving a graph
reasoning question with GRRAF is demonstrated
in Figure 3.
4 Computational Assessment
4.1 Dataset and Benchmark
We conduct experiments on GraphInstruct (Chen
et al., 2024), a dataset that comprises of nine graph
reasoning tasks with varying complexities. Due to
its diversity in graph reasoning tasks and its prior
use in evaluating state-of-the-art methods (Chen
et al., 2024; Hu et al., 2024), we select this dataset
for our evaluation. However, the task of finding
a Hamilton path lacks publicly available ground
truth labels and generating such labels through
code is infeasible due to the NP-hard nature of
the problem; consequently, we exclude this task
from our experiments. Accordingly, we assess the
performance of GRRAF on the following eight
tasks: cycle detection, connectivity, bipartite graph
check, topological sort, shortest path, maximum
triangle sum, maximum flow, and subgraph match-
ing. Details of these tasks are provided in Table 1.
Moreover, to achieve a more robust performance
evaluation, we augment the test dataset with two
additional simple tasks—indegree calculation and
outdegree calculation (as shown in Table 1)—to fa-
cilitate a comprehensive evaluation of GRRAF and
the state-of-the-art benchmarks. Each task has 400
question–graph pairs, each with a single correct
answer. We measure a method’s performance on
one task by its accuracy—that is, the proportion of
questions answered correctly out of the total (400).
GRRAF, i.e., its LLM, generates code which is
either correct or not. This is the reason why most
accuracies are going to be 100%. For tasks with
less than 100% accuracy, GRRAF yields correct
code but the underlying problems are NP-hard and
for some test graphs the execution times out. One
can argue that the output code is correct and thus
appropriate credit should be given, but on the other
hand, a more efficient algorithm and code can be
potentially produced. Sometimes the generated
code does not handle edge cases correctly, yet othertimes the code or algorithms are incorrect (they
solve only some test graphs by coincidence).
We compare the performance of GRRAF against
two state-of-the-art benchmarks: GraphWiz (Chen
et al., 2024) and GAR (Hu et al., 2024). Graph-
Wiz is trained on 17,158 questions and 72,785
answers, complete with reasoning paths, from
the training set of GraphInstruct. Since no sin-
gle version of GraphWiz consistently outperforms
the others across all tasks, we include three ver-
sions in our comparisons: GraphWiz (Mistral-7B),
GraphWiz-DPO (LLaMA 2-7B), and GraphWiz-
DPO (LLaMA 2-13B). GAR is a training-free
multi-agent framework that relies on a predefined
library of distributed algorithms created by humans.
As a result, it is incapable of solving unseen graph
reasoning tasks that require algorithms not present
in its library. Therefore, some results from GAR
are missing in the subsequent comparisons because
of its limitation.
4.2 Experiments
We conduct experiments using GRRAF with a time
limit of t= 5 minutes and a maximum error
feedback loop iteration of n= 3 . The backbone
LLM is GPT-4o. These parameter choices are jus-
tified by the sensitivity analysis in Appendix 4.3.
For the graph querying code, we evaluate two ap-
proaches: Cypher, a query language for Neo4j, and
NetworkX, a Python library for graphs, which we
denote as GRRAF Cand GRRAF N, respectively.
We deal with graph plain text, and thus can be con-
verted into either Neo4j data or NetworkX objects.
Figure 4 demonstrates that GARRF Noutper-
forms all benchmark methods, achieving 100% ac-
curacy on most graph reasoning tasks. GARRF C
exhibits comparable or superior performance rela-
tive to other benchmarks on the majority of tasks,
except for topological sort and subgraph match-
ing. Although GraphWiz outperforms GARRF C
in topological sort and subgraph matching, its in-
adequate performance on indegree calculation and
outdegree calculation suggests that it struggles with
even simple out-of-domain graph reasoning tasks.
Furthermore, due to its inherent limitations, GAR
is inapplicable to out-of-domain tasks such as max-
imum flow, subgraph matching, indegree calcula-
tion, and outdegree calculation. Consequently, con-
sidering both performance and generalization abil-
ity, GARRF Cand GARRF Nare better for address-
ing graph reasoning tasks than the other benchmark
models. The example code generated by GARRF N
5

Figure 4: Performance of GRRAF and benchmark models across all ten graph reasoning tasks. Missing data are
indicated as “NA” in the plot. The available-case mean refers to the average accuracy of each method calculated
using only the tasks where complete data is available (excluding maximum flow, subgraph matching, indegree
calculation, and outdegree calculation). The all-case mean refers to the average accuracy across all tasks, treating
’NA’ as 0.
for each graph reasoning task is presented in Ap-
pendix A.
Subgraph matching is NP-complete, and the
code produced by GARRF Nhas exponential time
complexity. For graphs of 20 nodes, executing that
code can take over a day—exceeding the time limit
t. Based on Section 3, in such cases GARRD N
falls back to using the original question Qas a
prompt to obtain the answer Adirectly from the
LLM, which may yield incorrect results. GRRAF C
likewise falls short of 100% accuracy on cycle de-
tection and bipartite-graph checking, since Cypher
queries execute more slowly than NetworkX. For
the maximum-flow task, GRRAF Cproduces code
that overlooks certain edge cases. And for topo-
logical sorting and subgraph matching, it generates
code that only succeeds on some graphs by chance.
Across the ten tasks, solving a single graph rea-
soning question requires GRRAF Nto use an av-
erage of 767 input tokens and 124 output tokens,
while GRRAF Cuses 796 input tokens and 201 out-
put tokens. In comparison, GraphWiz (Mistral-7B)
consumes an average of 1,046 input tokens and
126 output tokens per question, whereas GraphWiz-
DPO (LLaMA 2-7B) requires 1,046 input tokens
and 290 output tokens on average, and GraphWiz-
DPO (LLaMA 2-13B) uses 1,046 input tokens and
301 output tokens per question. Notably, GAR
demands more resources, averaging 8,095 input to-
kens and 5,987 output tokens for each graph reason-
ing question. Thus, comparing to other benchmark
methods, GRRAF Nand GRRAF Cachieve high
Figure 5: Accuracy of each method on the shortest path
task across graphs of differenct sizes (number of nodes).
accuracy in graph reasoning tasks while utilizing
fewer token resources.
Since the largest graph in GraphInstruct (Chen
et al., 2024) comprises of only 100 nodes, which
remains insufficient for real-world graph reason-
ing scenarios (Hu et al., 2024), we further evaluate
the best-performing method, GRRAF N, on large-
scale graphs. Following the approach of Hu et al.
(2024), we assess GRRAF Non the shortest path
task using larger graphs with 20 test samples for
each graph size. Whereas their work scales graphs
to 1,000 nodes, we extend this evaluation by scal-
ing graphs to 10,000 nodes to thoroughly assess
the performance of GRRAF N. According to Fig-
ure 5, GRRAF Nachieves 100% accuracy across
all graph sizes, demonstrating its exceptional scal-
ability. GAR attains 100% accuracy on graphs
with 100, 200, and 500 nodes, but its accuracy de-
creases to 90% on graphs with 1,000 nodes. Due to
token limitations, GAR is unable to address ques-
6

Figure 6: Average token cost for solving a graph rea-
soning problem across graphs of varying sizes on the
shortest path task.
tions on graphs with 2,000 nodes or more. In con-
trast, all three versions of GraphWiz perform poorly
on large graphs, achieving only 5-10% accuracy
on graphs with 100 nodes and failing entirely on
graphs with 200 nodes. The token limits of their
base model prevent them from processing graphs
larger than 200 nodes.
We also record the variation in token cost re-
quired to solve a single graph reasoning question
as the graph size increases on the shortest path task.
As illustrated in Figure 6, the number of tokens
used by GRRAF Nremains constant regardless of
the graph size. As detailed in Section 3, GRRAF in-
teracts with the graph solely via the graph database
through code execution; thus, the graph description
(nodes, edges, weights) is not directly input to the
LLM, and the token cost remains unaffected by
increases in graph size. In contrast, the token cost
for GraphWiz increases linearly with graph size be-
cause it must pass the information of each node and
edge to the LLM. The token cost for GAR is con-
siderably higher than that for GRRAF Nand grows
nearly exponentially with graph size. This is due
to GAR’s design, where each node is assigned an
LLM agent, and each agent communicates with ev-
ery adjacent agent in each iteration (Hu et al., 2024).
As the number of nodes increases, so do the number
of agents, the number of adjacent agents per node
(i.e., edges), and the number of iterations required
to obtain an answer, all of which contribute to a
significant rise in token cost. Therefore, compared
to other benchmarks, GRRAF Ncan readily scale
to very large graphs (up to 10,000 nodes) without
compromising performance and increasing token
cost.
To evaluate the effectiveness of the error feed-Method Execution Error Time-out
GRRAF N 2.2% 5.4%
GRRAF C 4.9% 9.1%
Table 2: Percentage of graph reasoning questions over
10 tasks triggering error feedback loop due to execution
errors or time-outs for each method.
back loop, we quantify the total percentage of ques-
tions that activate this loop, as reported in Table 2.
In general, GRRAF Ctriggers the error feedback
loop more frequently than GRRAF N. For both vari-
ants, the loop is activated due to time-outs more
often than due to execution errors, underscoring
the importance of time efficiency in graph reason-
ing tasks. Overall, the backbone LLM generates
correct code queries in most instances, and the in-
tegration of an error feedback loop with a time-out
mechanism further enhances code accuracy and
efficiency.
4.3 Sensitivity Analysis
Figure 7: Average accuracy of GRRAF Nwith different
time limitt.
Figure 8: Average accuracy of GRRAF Nwith different
maximum error feedback loop iterationn.
We perform sensitivity analyses on GRRAF N
to assess the impact of the time limit t, the max-
imum number of error-feedback loop iterations
n, and the choice of backbone LLM. We report
the average accuracy across all ten graph reason-
ing tasks. As shown in Figure 7, the accuracy
increases with tup to five minutes, after which
no further gains are observed. Figure 8 indicates
that accuracy peaks at n= 3 and declines slightly
7

Figure 9: Average accuracy of GRRAF Nwith different
backbone LLM.
forn >3 . Finally, we evaluated GRRAF Nus-
ing three backbone LLMs—GPT-4o, Claude-3.5-
Sonnet, and Llama3.1-405b-Instruct—and found
that all three yield comparable results, with GPT-4o
achieving a slightly higher average accuracy than
the others (Figure 9).
5 Conclusion
In this work, we introduced GRRAF, a novel frame-
work that integrates RAG with the code-writing
prowess of LLMs to address graph reasoning ques-
tions. Our approach, which operates without addi-
tional training or reliance on predefined algorithms,
leverages a graph database to store target graphs
and employs an error feedback loop with a time-out
mechanism to ensure the generation of correct and
efficient code queries. Comprehensive experiments
on the GraphInstruct dataset and two extra tasks
(indegree and outdegree) demonstrate that GRRAF
outperforms existing state-of-the-art benchmarks,
achieving 100% accuracy on a majority of graph
reasoning tasks while effectively scaling to graphs
containing up to 10,000 nodes without incurring
extra token costs. These findings underscore the
potential of combining retrieval-based techniques
with LLM-driven code generation for solving com-
plex graph reasoning problems. Future work could
explore extending this framework to dynamic graph
scenarios and additional reasoning tasks, further
enhancing its applicability and robustness.
6 Limitations
Although GRRAF Nattains 100% accuracy on all
polynomial-time graph reasoning tasks, it nev-
ertheless struggles to solve NP-complete prob-
lems—such as subgraph matching—both accu-
rately and efficiently. Moreover, the inferior per-
formance of GRRAF Crelative to GRRAF Nindi-
cates that our framework currently generates lower-
quality Cypher queries than the equivalent Pythoncode. These two issues constitute the primary limi-
tations of our method.
References
Sören Auer, Christian Bizer, Georgi Kobilarov, Jens
Lehmann, Richard Cyganiak, and Zachary Ives. 2007.
Dbpedia: A nucleus for a web of open data. In
International Semantic Web Conference, pages 722–
735.
Jinheon Baek, Soyeong Jeong, Minki Kang, Jong Park,
and Sung Hwang. 2023. Knowledge-augmented lan-
guage model verification. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 1720–1736.
Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim
Sturge, and Jamie Taylor. 2008. Freebase: a collabo-
ratively created graph database for structuring human
knowledge. InProceedings of the 2008 ACM SIG-
MOD International Conference on Management of
Data, pages 1247–1250.
Nuo Chen, Yuhan Li, Jianheng Tang, and Jia Li. 2024.
Graphwiz: An instruction-following language model
for graph computational problems. InProceedings
of the 30th ACM SIGKDD Conference on Knowledge
Discovery and Data Mining, pages 353–364.
Mohammad Dehghan, Mohammad Alomrani, Sunyam
Bagga, David Alfonso-Hermelo, Khalil Bibi, Ab-
bas Ghaddar, Yingxue Zhang, Xiaoguang Li, Jianye
Hao, Qun Liu, Jimmy Lin, Boxing Chen, Prasanna
Parthasarathi, Mahdi Biparva, and Mehdi Reza-
gholizadeh. 2024. EWEK-QA : Enhanced web and
efficient knowledge graph retrieval for citation-based
question answering systems. InProceedings of the
62nd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
14169–14187.
Julien Delile, Srayanta Mukherjee, Anton Van Pamel,
and Leonid Zhukov. 2024. Graph-based retriever
captures the long tail of biomedical knowledge. In
ICML’24 Workshop ML for Life and Material Sci-
ence: From Theory to Industry Applications.
Junnan Dong, Qinggang Zhang, Xiao Huang, Keyu
Duan, Qiaoyu Tan, and Zhimeng Jiang. 2023.
Hierarchy-aware multi-hop question answering over
knowledge graphs. InProceedings of the ACM web
conference 2023, pages 2519–2527.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024. From local to global: A
graph RAG approach to query-focused summariza-
tion.arXiv preprint arXiv:2404.16130.
Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi.
2024. Talk like a graph: Encoding graphs for large
language models. InThe Twelfth International Con-
ference on Learning Representations.
8

Jiayan Guo, Lun Du, Hengyu Liu, Mengyu Zhou, Xinyi
He, and Shi Han. 2023. GPT4Graph: Can large
language models understand graph structured data?
an empirical evaluation and benchmarking. InarXiv
preprint arXiv:2305.15066.
Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla,
Thomas Laurent, Yann LeCun, Xavier Bresson, and
Bryan Hooi. 2024. G-retriever: Retrieval-augmented
generation for textual graph understanding and ques-
tion answering. InThe Thirty-eighth Annual Confer-
ence on Neural Information Processing Systems.
Yuwei Hu, Runlin Lei, Xinyi Huang, Zhewei Wei, and
Yongchao Liu. 2024. Scalable and accurate graph
reasoning with LLM-based multi-agents. InarXiv
preprint arXiv:2410.05130.
Yongfeng Huang, Yanyang Li, Yichong Xu, Lin Zhang,
Ruyi Gan, Jiaxing Zhang, and Liwei Wang. 2023.
MVP-Tuning: Multi-view knowledge retrieval with
prompt tuning for commonsense reasoning. InPro-
ceedings of the 61st Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers), pages 13417–13432.
Boran Jiang, Yuqi Wang, Yi Luo, Dawei He, Peng
Cheng, and Liangcai Gao. 2024. Reasoning on effi-
cient knowledge paths: knowledge graph guides large
language model for domain question answering. In
2024 IEEE International Conference on Knowledge
Graph, pages 142–149.
Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Xin
Zhao, and Ji-Rong Wen. 2023. StructGPT: A general
framework for large language model to reason over
structured data. InProceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing, pages 9237–9251.
Bowen Jin, Chulin Xie, Jiawei Zhang, Kashob Ku-
mar Roy, Yu Zhang, Zheng Li, Ruirui Li, Xianfeng
Tang, Suhang Wang, Yu Meng, and Jiawei Han. 2024.
Graph chain-of-thought: Augmenting large language
models by reasoning on graphs. InFindings of the As-
sociation for Computational Linguistics: ACL 2024,
pages 163–184.
Jiho Kim, Yeonsu Kwon, Yohan Jo, and Edward Choi.
2023. KG-GPT: A general framework for reasoning
on knowledge graphs using large language models.
InFindings of the Association for Computational
Linguistics: EMNLP 2023.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474.
Shiyang Li, Yifan Gao, Haoming Jiang, Qingyu Yin,
Zheng Li, Xifeng Yan, Chao Zhang, and Bing Yin.
2023. Graph reasoning for question answering withtriplet retrieval. InFindings of the Association for
Computational Linguistics: ACL 2023, pages 3366–
3375.
Bill Yuchen Lin, Xinyue Chen, Jamin Chen, and Xiang
Ren. 2019. KagNet: Knowledge-aware graph net-
works for commonsense reasoning. InProceedings
of the 2019 Conference on Empirical Methods in Nat-
ural Language Processing and the 9th International
Joint Conference on Natural Language Processing,
pages 2829–2839.
Tianqianjin Lin, Pengwei Yan, Kaisong Song, Zhuoren
Jiang, Yangyang Kang, Jun Lin, Weikang Yuan, Jun-
jie Cao, Changlong Sun, and Xiaozhong Liu. 2024.
LangGFM: A large language model alone can be a
powerful graph foundation model. InarXiv preprint
arXiv:2410.14961.
Chang Liu and Bo Wu. 2023. Evaluating large language
models on graphs: Performance insights and compar-
ative analysis. InarXiv preprint arXiv:2308.11224.
Hugo Liu and Push Singh. 2004. ConceptNet—a practi-
cal commonsense reasoning tool-kit. InBT Technol-
ogy Journal, volume 22, pages 211–226.
LINHAO LUO, Yuan-Fang Li, Reza Haf, and Shirui
Pan. 2024. Reasoning on graphs: Faithful and in-
terpretable large language model reasoning. InThe
Twelfth International Conference on Learning Repre-
sentations.
Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li,
Huaren Qu, Cehao Yang, Jiaxin Mao, and Jian Guo.
2025. Think-on-graph 2.0: Deep and faithful large
language model reasoning with knowledge-guided
retrieval augmented generation. InThe Thirteenth In-
ternational Conference on Learning Representations.
Costas Mavromatis and George Karypis. 2022. ReaRev:
Adaptive reasoning for question answering over
knowledge graphs. InFindings of the Association
for Computational Linguistics: EMNLP 2022, pages
2447–2458.
Costas Mavromatis and George Karypis. 2024.
GNN-RAG: Graph neural retrieval for large lan-
guage model reasoning. InarXiv preprint
arXiv:2405.20139.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. 2024. Graph retrieval-augmented generation:
A survey. InarXiv preprint arXiv:2408.08921.
Bryan Perozzi, Bahare Fatemi, Dustin Zelle, Anton Tsit-
sulin, Mehran Kazemi, Rami Al-Rfou, and Jonathan
Halcrow. 2024. Let your graph do the talking: En-
coding structured data for LLMs. InarXiv preprint
arXiv:2402.05862.
Maarten Sap, Ronan Le Bras, Emily Allaway, Chan-
dra Bhagavatula, Nicholas Lourie, Hannah Rashkin,
Brendan Roof, Noah A Smith, and Yejin Choi. 2019.
9

Atomic: An atlas of machine commonsense for if-
then reasoning. InProceedings of the AAAI Con-
ference on Artificial Intelligence, volume 33, pages
3027–3035.
Bhaskarjit Sarmah, Dhagash Mehta, Benika Hall, Ro-
han Rao, Sunil Patel, and Stefano Pasquali. 2024.
HybridRAG: Integrating knowledge graphs and vec-
tor retrieval augmented generation for efficient infor-
mation extraction. InProceedings of the 5th ACM
International Conference on AI in Finance, pages
608–616.
Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus
Hagenbuchner, and Gabriele Monfardini. 2008. The
graph neural network model. InIEEE Transactions
on Neural Networks, volume 20, pages 61–80.
Yiheng Shu, Zhiwei Yu, Yuhan Li, Börje Karlsson,
Tingting Ma, Yuzhong Qu, and Chin-Yew Lin. 2022.
TIARA: Multi-grained retrieval for robust question
answering over large knowledge base. InProceed-
ings of the 2022 Conference on Empirical Methods
in Natural Language Processing, pages 8108–8121.
Konstantinos Skianis, Giannis Nikolentzos, and
Michalis Vazirgiannis. 2024. Graph reasoning with
large language models via pseudo-code prompting.
InarXiv preprint arXiv:2409.17906.
Fabian M Suchanek, Gjergji Kasneci, and Gerhard
Weikum. 2007. Yago: a core of semantic knowledge.
InProceedings of the 16th International Conference
on World Wide Web, pages 697–706.
Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo
Wang, Chen Lin, Yeyun Gong, Lionel Ni, Heung-
Yeung Shum, and Jian Guo. 2024a. Think-on-Graph:
Deep and responsible reasoning of large language
model on knowledge graph. InThe Twelfth Interna-
tional Conference on Learning Representations.
Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo
Wang, Chen Lin, Yeyun Gong, Lionel Ni, Heung-
Yeung Shum, and Jian Guo. 2024b. Think-on-graph:
Deep and responsible reasoning of large language
model on knowledge graph. InThe Twelfth Interna-
tional Conference on Learning Representations.
Lei Sun, Zhengwei Tao, Youdi Li, and Hiroshi Arakawa.
2024c. ODA: Observation-driven agent for integrat-
ing LLMs and knowledge graphs. InFindings of
the Association for Computational Linguistics: ACL
2024, pages 7417–7431.
Jianheng Tang, Qifan Zhang, Yuhan Li, Nuo Chen, and
Jia Li. 2025. GraphArena: Evaluating and exploring
large language models on graph computation. In
The Thirteenth International Conference on Learning
Representations.
Dhaval Taunk, Lakshya Khanna, Siri Venkata Pavan Ku-
mar Kandru, Vasudeva Varma, Charu Sharma, and
Makarand Tapaswi. 2023. GrapeQA: Graph augmen-
tation and pruning to enhance question-answering. In
Companion Proceedings of the ACM Web Conference
2023, pages 1138–1144.Denny Vrande ˇci´c and Markus Krötzsch. 2014. Wiki-
data: a free collaborative knowledgebase. InCom-
munications of the ACM, volume 57, pages 78–85.
Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan
Tan, Xiaochuang Han, and Yulia Tsvetkov. 2024a.
Can language models solve graph problems in nat-
ural language? InAdvances in Neural Information
Processing Systems, volume 36.
Yu Wang, Nedim Lipka, Ryan A Rossi, Alexa Siu, Ruiyi
Zhang, and Tyler Derr. 2024b. Knowledge graph
prompting for multi-document question answering.
InProceedings of the AAAI Conference on Artificial
Intelligence, volume 38, pages 19206–19214.
Yilin Wen, Zifeng Wang, and Jimeng Sun. 2024.
MindMap: Knowledge graph prompting sparks graph
of thoughts in large language models. InProceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 10370–10388.
Sondre Wold, Lilja Øvrelid, and Erik Velldal. 2023.
Text-to-KG alignment: Comparing current methods
on classification tasks. InProceedings of the First
Workshop on Matching From Unstructured and Struc-
tured Data, pages 1–13.
Qiming Wu, Zichen Chen, Will Corcoran, Misha Sra,
and Ambuj K Singh. 2024. Grapheval2000: Bench-
marking and improving large language models on
graph datasets. InarXiv preprint arXiv:2406.16176.
Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong
Long, Chengqi Zhang, and S Yu Philip. 2020. A
comprehensive survey on graph neural networks. In
IEEE Transactions on Neural Networks and Learning
Systems, volume 32, pages 4–24.
Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut,
Percy Liang, and Jure Leskovec. 2021. QA-GNN:
Reasoning with language models and knowledge
graphs for question answering. InNorth American
Chapter of the Association for Computational Lin-
guistics.
Donghan Yu, Sheng Zhang, Patrick Ng, Henghui
Zhu, Alexander Hanbo Li, Jun Wang, Yiqun Hu,
William Yang Wang, Zhiguo Wang, and Bing Xiang.
2023. DecAF: Joint decoding of answers and log-
ical forms for question answering over knowledge
bases. InThe Eleventh International Conference on
Learning Representations.
Jiawei Zhang. 2023. Graph-toolformer: To em-
power LLMs with graph reasoning ability via
prompt augmented by chatgpt. InarXiv preprint
arXiv:2304.11116.
Jing Zhang, Xiaokang Zhang, Jifan Yu, Jian Tang, Jie
Tang, Cuiping Li, and Hong Chen. 2022a. Subgraph
retrieval enhanced model for multi-hop knowledge
base question answering. InProceedings of the 60th
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 5773–
5784.
10

Xikun Zhang, Antoine Bosselut, Michihiro Yasunaga,
Hongyu Ren, Percy Liang, Christopher D Manning,
and Jure Leskovec. 2022b. GreaseLM: Graph REA-
Soning enhanced language models. InInternational
Conference on Learning Representations.
Zeyang Zhang, Xin Wang, Ziwei Zhang, Haoyang Li,
Yijian Qin, and Wenwu Zhu. 2024. LLM4DyG: Can
large language models solve spatial-temporal prob-
lems on dynamic graphs? InProceedings of the 30th
ACM SIGKDD Conference on Knowledge Discovery
and Data Mining, page 4350–4361.
Jianan Zhao, Le Zhuo, Yikang Shen, Meng Qu, Kai
Liu, Michael M. Bronstein, Zhaocheng Zhu, and
Jian Tang. 2024. Graphtext: Graph reasoning in text
space. InAdaptive Foundation Models: Evolving AI
for Personalized and Efficient Learning.
Yun Zhu, Yaoke Wang, Haizhou Shi, and Siliang Tang.
2024. Efficient tuning and inference for large lan-
guage models on textual graphs. InProceedings of
the Thirty-Third International Joint Conference on
Artificial Intelligence, pages 5734–5742.
A Example Code
This section presents example code generated by
GRRAF Nfor each graph reasoning task in our
experiments: cycle detection (Figure 10), connec-
tivity (Figure 11), bipartite graph check (Figure 12),
topological sort (Figure 13), shortest path (Figure
14), maximum triangle sum (Figure 15), maximum
flow (Figure 16), subgraph matching (Figure 17),
indegree calculation (Figure 18), and outdegree cal-
culation (Figure 19). All these examples produce
correct answers.
We also include in Figure 20 an example Cypher
query generated by GRRAF Cfor the maximum-
flow task. Although this query attempts to imple-
ment the Ford–Fulkerson algorithm, it omits the
backward residual edges, preventing any rerouting
of earlier flows. Consequently, on certain edge
cases (e.g., the graph in Figure 21), it produces
incorrect results. Similarly, Figure 22 shows an
instance where GRRAF Cgenerates an incorrect
Cypher query for topological sorting. That query
builds a spanning tree rooted at a node of zero in-
degree to derive the ordering—a method that is
unsound and succeeds only by chance on some
graphs.
11

Figure 10: An example of the final codeC′generated for the cycle detection task.
Figure 11: An example of the final codeC′generated for the connectivity task.
Figure 12: An example of the final codeC′generated for the bipartite graph check task.
Figure 13: An example of the final codeC′generated for the topological sort task.
Figure 14: An example of the final codeC′generated for the shortest path task.
Figure 15: An example of the final codeC′generated for the maximum triangle sum task.
Figure 16: An example of the final codeC′generated for the maximum flow task.
12

Figure 17: An example of the final codeC′generated for the subgraph matching task.
Figure 18: An example of the final codeC′generated for the indegree calculation task.
Figure 19: An example of the final codeC′generated for the outdegree calculation task.
Figure 20: An example of the final codeC′in Cypher query by GARRF Cgenerated for the maximum flow task.
13

Figure 21: An example directed graph with edge weights. The correct maximum flow from node 2 to 6 is 3 but the
Cypher query in Figure 20 returns 4 as the answer.
Figure 22: An example of the final codeC′in Cypher query by GARRF Cgenerated for the topological sort task.
14
# GNN-ACLP: Graph Neural Networks based Analog Circuit Link Prediction

**Authors**: Guanyuan Pan, Tiansheng Zhou, Bingtao Ma, Yaqi Wang, Jianxiang Zhao, Shuai Wang

**Published**: 2025-04-14 14:02:09

**PDF URL**: [http://arxiv.org/pdf/2504.10240v1](http://arxiv.org/pdf/2504.10240v1)

## Abstract
Circuit link prediction identifying missing component connections from
incomplete netlists is crucial in automating analog circuit design. However,
existing methods face three main challenges: 1) Insufficient use of topological
patterns in circuit graphs reduces prediction accuracy; 2) Data scarcity due to
the complexity of annotations hinders model generalization; 3) Limited
adaptability to various netlist formats. We propose GNN-ACLP, a Graph Neural
Networks (GNNs) based framework featuring three innovations to tackle these
challenges. First, we introduce the SEAL (Subgraphs, Embeddings, and Attributes
for Link Prediction) framework and achieve port-level accuracy in circuit link
prediction. Second, we propose Netlist Babel Fish, a netlist format conversion
tool leveraging retrieval-augmented generation (RAG) with large language model
(LLM) to enhance the compatibility of netlist formats. Finally, we construct
SpiceNetlist, a comprehensive dataset that contains 775 annotated circuits
across 10 different classes of components. The experimental results demonstrate
an improvement of 15.05% on the SpiceNetlist dataset and 12.01% on the
Image2Net dataset over the existing approach.

## Full Text


<!-- PDF content starts -->

arXiv:2504.10240v1  [cs.AR]  14 Apr 2025GNN-ACLP: Graph Neural Networks based Analog
Circuit Link Prediction
Guanyuan Pana, Tiansheng Zhoub, Bingtao Mab, Yaqi Wangc, Jianxiang
Zhaob, Shuai Wangb,∗
aHDU-ITMO Joint Institute, Hangzhou Dianzi University, No. 1158, 2nd Avenue,
Xiasha Higher Education Zone, Jianggan District, Hangzhou, 310018, Zhejiang
Province, China
bIntelligent Information Processing Laboratory, Hangzhou Dianzi University, No. 1158,
2nd Avenue, Xiasha Higher Education Zone, Jianggan
District, Hangzhou, 310018, Zhejiang Province, China
cCollege of Media Engineering, Communication University of Zhejiang, No. 998,
Xueyuan Street, Xiasha Higher Education Zone, Jianggan
District, Hangzhou, 310018, Zhejiang Province, China
Abstract
Circuit link prediction identifying missing component connections from in-
complete netlists is crucial in automating analog circuit design. However,
existing methods face three main challenges: 1) Insufficient use of topological
patterns in circuit graphs reduces prediction accuracy; 2) Data scarcity due
to the complexity of annotations hinders model generalization; 3) Limited
adaptability to various netlist formats. We propose GNN-ACLP, a Graph
Neural Networks (GNNs) based framework featuring three innovations to
tackle these challenges. First, we introduce the SEAL (Subgraphs, Embed-
dings, and Attributes for Link Prediction) framework and achieve port-level
accuracy in circuit link prediction. Second, we propose Netlist Babel Fish,
a netlist format conversion tool leveraging retrieval-augmented generation
(RAG) with large language model (LLM) to enhance the compatibility of
netlist formats. Finally, we construct SpiceNetlist, a comprehensive dataset
∗Corresponding author
Email addresses: panguanyuan@hdu.edu.cn (Guanyuan Pan),
zhoutiansheng_2024@163.com (Tiansheng Zhou), mabingtao@hdu.edu.cn (Bingtao
Ma), wangyaqi@cuz.edu.cn (Yaqi Wang), 246270147@hdu.edu.cn (Jianxiang Zhao),
shuaiwang@hdu.edu.cn (Shuai Wang)
Preprint submitted to Knowledge-Based Systems April 18, 2025

that contains 775 annotated circuits across 10 different classes of compo-
nents. The experimental results demonstrate an improvement of 15.05% on
the SpiceNetlist dataset and 12.01% on the Image2Net dataset over the ex-
isting approach.
Keywords: Circuit Link Prediction, GNNs, LLM, RAG, EDA
1. Introduction
The design of analog circuits has mainly been performed manually by
experienced engineers. However, analog design is susceptible to layout geom-
etry [1], therefore even experienced engineers face difficulties when designing
analog circuits. Therefore, automating analog circuit design has gained sig-
nificant attention recently. One main task of analog design automation is
circuit link prediction, which involves inferring missing component intercon-
nections from incomplete netlists.
There are two conventional approaches for general link prediction: heuris-
tic methods and learning-based approaches. Heuristic approaches can be
divided into two categories: local similarity metrics, which include the Jac-
card index, common neighbors, preferential attachment, and resource allo-
cation, and global similarity metrics, including the Katz index and Sim-
Rank [2]. Although heuristic methods offer simplicity and interpretability
through predefined topological features, they have limited generalizability
due to their dependence on handcrafted structural assumptions about link
formation mechanisms [3]. Therefore, they are not commonly used in circuit
link prediction.
In the Electronic Design Automation (EDA) field, where circuit netlists
represent graph-structured data, machine learning (ML) approaches have
gained more prominence. Paul et al. proposed a novel method using brain-
inspired hyperdimensional computing (HDC) for encoding and recognizing
gate-level netlists [4]. However, the high-dimensional vector operations in-
volved in HDC led to significant computational and storage overhead, es-
pecially when dealing with large-scale graph data [5]. Luo et al. pro-
posed a novel neural model called Directional Equivariant Hypergraph Neural
Network (DE-HNN) for effective representation learning on directed hyper-
graphs, particularly for circuit netlists in chip design [6]. However, DE-HNN
performing hypergraph diffusion or tensor operations could exponentially
2

increase computational costs as the hyperedge order increases, making it
challenging to scale the approach for large chip design scenarios [7].
GNNs can learn from circuit netlists’ inherent graph structure, making
them well-suited for circuit link prediction. However, a critical challenge
lies in the lack of annotated training data caused by high human workload
and data sensitivity [8]. Furthermore, the diversity of netlist formats poses
significant challenges in analog design automation. Additionally, customized
netlist formats are often created by modifying standard formats. Conse-
quently, different datasets are often created in different formats, complicating
the training processes for machine learning methods.
To address these challenges, we propose GNN-ACLP , a GNNs-based
method for analog circuit link prediction, as illustrated in Figure 1, and
SpiceNetlist, a large-scale netlist dataset, whose statistic is demonstrated
in Table 1. 1)On the one hand, we formulate circuit link prediction as an
undirected graph link prediction problem, where nodes represent circuit com-
ponents and edges denote connections. To solve this task, we leverage the
SEAL framework, trained using DRNL one-hot encoding augmented with
original component node features for enhanced representation learning. 2)
On the other hand, we introduce Netlist Babel Fish , a framework featur-
ing RAG for LLM-based interpretation and enabling bidirectional netlist for-
mat conversion through domain-specific knowledge integration, to overcome
the challenge of differing dataset formats. 3)Finally, we develop a robust
preprocessing pipeline to detect and correct content errors and syntactic in-
consistencies across four SPICE netlist datasets. We then integrated these
datasets into into one unified large-scale dataset, SpiceNetlist, to support
the training and evaluation of circuit link prediction methods. SpiceNetlist
has 775 annotated circuits in both SPICE and JSON formats, featuring 10
kinds of component. Our accuracy shows an improvement of 15.05% on the
SpiceNetlist dataset and 12.01% on the Image2Net dataset compared to [9].
The contributions of our work are outlined below:
•GNN-ACLP , a port-level accurate circuit link prediction method
based on the SEAL framework using graph neural networks (GNNs).
To the best of our knowledge, this is the first work to address cir-
cuit link prediction at the port level, as opposed to the conventional
component-level approach.
•Netlist Babel Fish , a netlist format converter enabling port-level
3

netlist parsing across multiple formats, leveraging RAG for LLM-based
interpretation.
•SpiceNetlist , a novel circuit netlist dataset comprising 775 annotated
circuits in both SPICE and JSON formats, featuring 10 component
types to facilitate the training and evaluation of circuit link prediction
methods.
𝑮𝟏’𝑮𝒏’Extracting SubgraphsSEAL
VDD
VoutM1 VinInput
𝑮 
？？
Graph Neural NetworkLink 
Prediction
√Netlist 
Babel Fish
Figure 1: Illustration of GNN-ACLP.
Table 1: SpiceNetlist statistic.
Dataset Train Test Classes Avg. Nodes Avg. Edges
SpiceNetlist 620 155 10 13.78 15.60
2. Preliminaries and problem formulation
The circuit link prediction problem is a part of the circuit completion
problem. One can define the circuit completion problem as follows:
Problem 2.1 (circuit completion problem) Let v0andv1represent the
netlists of two circuit schematics, where v1is a partial copy of v0missing all
4

connections associated with a component xthat are present in v0. The circuit
completion problem involves predicting the type of the missing component x
and all of its missing connections, where x∈ {1,2,3, . . . , k }.
Here, netlists define the connections between circuit components, includ-
ing ports, each with a unique ID and type. [10, 11]
There are several challenges in solving circuit completion problems. First,
the problem implicitly assumes an explicit criterion for determining a circuit’s
validity, which is necessary for verifying a solution’s correctness. Although
some checks can be performed, no definitive criteria can be used to validate
a circuit netlist. Second, it may not be relevant to one’s interests even if a
circuit is valid. As long as both v0andv′
0are netlists of designs that are
considered ”interesting” (where ”interesting” is defined by the specific ap-
plication), it may be acceptable to predict either as a valid completion of
the partial netlist v1. However, v′
0may be a trivial completion irrelevant to
the particular application being studied [12, 13, 14]. To address these chal-
lenges, we abandon any auxiliary information from the netlists and convert
each netlist into an undirected graph, denoted as G= (V, E, ϕ ). In this
graph, the set of vertices Vrepresents the ports of all components in the
netlist, while the edges denote the connections between these ports in Ethat
link the corresponding vertices. Additionally, each vertex vhas an integer
type, denoted as ϕ(v), taken from the set {1,2,3, . . . , k }.
We then define the circuit link prediction problem:
Problem 2.2 (circuit link completion problem on graphs) Let G=
(V, E, ϕ ), be a graph of a netlist with ϕ(v)∈[k] where [ k] :={1,2,3, . . . , k }.
LetG′be a graph obtained by removing an arbitrary vertex u∈Vfrom G.
Given the value of ϕ(u) and G′, compute the set of neighbors of uinG.
Here, the set of neighbors of a node u is defined as the following set:
{v∈V: (u, v)∈E}. We propose data-driven solutions to address the
problem. Specifically, we learn a map, ˆξ:G,[k]→2V, that takes as input
a graph Gand a component type ϕ∈[k]. This mapping returns a subset
of vertices (the power set of vertices V, represented as 2V) that indicate the
connections for the ports of components within graph G. We provide a visual
illustration of this problem in Figure 2 and 3.
3. Link prediction through GNNs
In Problem 2.2, we are given a graph Gcontaining various node types
and a distinguished node v. The objective of the problem is to predict the
5

U1论文作图 -1
Vref
Vfb
Error
AmplifierVin
M1
R2
CoVout
Iload
？
Figure 2: Illustration of the circuit link prediction problem. a)A schematic netlist repre-
sentation, b)Ports of new components and possible links.
Graph Neural NetworkLink 
Prediction
𝑮
𝑮𝟏’𝑮𝒏’Extracting SubgraphsSEAL
VDD
VoutRD
Vin M1Input
Netlist 
Babel Fish
Figure 3: Architecture of our circuit link prediction architecture with SEAL.
neighborhood of vwithin G. This problem can be simplified by determining
whether an edge should exist between vand each neighboring node uinG.
The goal is to assess whether two nodes in the graph will likely be connected
by an edge based on the other nodes’ connectivity. Therefore, this connection
test can be framed as a link prediction problem.
We select the SEAL framework [15], which we illustrate in Figure 3. The
idea is to enclose subgraphs around the links in a graph representing positive
class instances to extract training data. For a pair of nodes ( x, y), an enclos-
ing subgraph is defined as a subgraph that includes the h-hop neighborhood
of both xandy. In practice, this method is effective for the link prediction
task. This work proposes a node labeling approach denoted as Double-Radius
6

Node Labeling (DRNL) to label nodes within the subgraphs. DRNL aims to
identify the different roles of nodes while preserving structural information.
We adopt the SEAL framework for link prediction in circuit graphs and train
it using DRNL one-hot encoding combined with the original component node
features.
It is important to note that our work may not address the circuit link
prediction problem on graphs since there can be multiple valid ways to link
ports in a partial circuit [16, 17, 18]. Therefore, the proposed method may
occasionally encounter failures. However, we show through extensive exper-
iments that graph machine-learning methods yield reliable results on real-
world datasets to support human experts in the design, synthesis, and eval-
uation processes.
4. Netlist Babel Fish: Netlist Format Converter
To address the challenge of differing dataset formats, we develop Netlist
Babel Fish, a framework that enables bidirectional conversion between netlist
formats through integration with domain-specific information. Netlist Babel
Fish integrates a LLM with a RAG system grounded in a structured SPICE
knowledge base. This knowledge base includes brief references of the SPICE
devices and statements [19], methodologies for the netlist process, supported
component libraries with port assignment specifications, and examples illus-
trating practical implementations. Here, we utilize We use DeepSeek V2.5[20]
as the LLM here. We show the workflow of Netlist Babel Fish converting a
SPICE netlist to a custom JSON format[21] in Figure 4. This workflow is
also reversible for conversion from the custom JSON format to SPICE.
While the current implementation specifically addresses SPICE and spe-
cific JSON format conversions, the modular architecture theoretically sup-
ports arbitrary netlist format transformations given corresponding domain
knowledge bases. The system’s extensibility derives from its decoupled knowl-
edge representation framework, where format-specific conversion rules are
explicitly encoded rather than implicitly learned.
5. SpiceNetlist: Datasets Rectification
For the circuit-link-prediction problem, there are very few datasets avail-
able, and the existing ones do not contain enough readily accessible netlists
for practical training and evaluation. Moreover, existing SPICE netlist datasets
7

Netlist Babel Fish
C device - Capacitor. 
C{name} {+node} { -node} 
[{model}] {value} [IC={initial}] 
Examples: 
CLOAD  15  0  20pF
CFDBK   3 33  CMOD 10pF 
IC=1.5v
…………
2. 网表解析指南：
- 识别节点：每个组件由其标识
符、节点和附加参数（如值或模型）
定义。
- 读取连接性：对于每一行，读
取组件类型，提取其节点连接，并
确定其属性（例如电阻器的阻值）。
……
SPICE device 
& statementsNetlist Processing 
Methodologies| 组件大类 | 细分组件小类 | 端口 |
……
| Voltage | Voltage | Pos, Neg |
| Current | Current | In, Out |
……
注意
严格遵循支持组件及端口分配规范，
输出的 component_type （对应组
件大类）不允许超出范围！
Supported Component
Libraries & Port Assignment 
SpecificationsSPICE:
'''
Q3 ( Vout  VB \-5V 0) npn
……
'''
Examples of SPICE 
Netlist & Json Netlistjson  dict:
'''
[{'component_type ': 'NPN',
            'port_connection ': {'Base': 'VB',
                                'Collector': ' Vout ',
                                'Emitter': ' \\-5V'}},
            …..]
'''KnowledgeResponseRelevant 
Circuit
InformationEmbedded
queryNetlist and relevant information
 Large 
Language 
ModelSPICE
Format
Netlist
Customized
JSON Format
Netlist
Response
Customized
JSON Format
Netlist
Figure 4: Workflow of Netlist Babel Fish.
often contain content errors[21] and syntactic inconsistencies, leading to their
incompatibility with Netlist Babel Fish for proper parsing.
To mitigate this issue, we employ PySpice[22] to automatically detect
and filter out SPICE netlists with syntactic inconsistencies. Subsequently,
we convert all datasets to JSON format with Netlist Babel Fish, and perform
manual verification to identify and rectify netlists with content errors in both
formats. We illustrate this preprocessing pipeline in Figure 5.
Following this pipeline, we integrate four datasets — LTSpice examples,
LTSpice demos, KiCad Github [9] and AMSNet [23] as one unified large-scale
dataset, SpiceNetlist. Table 1 shows SpiceNetlist’s statistic.
SPICE Netlist with 
content errors &
syntactic inconsistencies
PySpice
SPICE Netlist with 
content errorsParse error on:
M1 net1 FCLK
SPICE Netlist with 
content errorsNetlist Babel Fish
JSON Netlist with 
content errors
Manual 
Identification
Rectified 
JSON NetlistsJSON Netlist with 
content errors
Netlist Babel Fish
Rectified 
SPICE NetlistsResults
Figure 5: SPICE netlist datasets preprocessing pipeline.
8

6. Experiments
6.1. Dataset Processing
For our experiments, we use SpiceNetlist and Image2Net1. Table 2 shows
Image2Net’s statistic, and table 3 demonstrates all components featured by
both datasets.
Table 2: Image2Net statistic.
Dataset Train Test Classes Avg. Nodes Avg. Edges
Image2Net 1568 392 13 26.36 37.80
Table 3: Component types and their labels/ports in the datasets.
Component Label Ports
PMOS/NMOS PMOS/NMOS Drain, Source, Gate
PMOS/NMOS with a bulk (body) port PMOS bulk/NMOS bulk Drain, Source, Gate, Body
Voltage Source Voltage Pos, Neg
Current Source Current In, Out
BJT (NPN/PNP) NPN/NPN cross/PNP/PNP cross Base, Emitter, Collector
Diode Diode In, Out
DISO Amplifier Diso amp InN, InP, Out
SISO Amplifier Siso amp In, Out
DIDO Amplifier Dido amp InN, InP, OutN, OutP
Passive Components (Cap, Ind, Res) Cap, Ind, Res Pos, Neg
We opt for an approach similar to batching in graph learning domain,
where we stack the adjacency matrices in diagonal blocks to represent the
entire dataset as a single huge graph. Let Ai∈[0,1]Ni×Nibe the adjacency
matrix of graph Giwith Ninodes. Then, the resulting graph will have
N:=P
1≤j≤nNjnodes. The stacked adjacency matrix Awith dimensions,
N×N, and the corresponding vector with concatenated node features are
defined as:
A=
A1
...
An
, X =
X1
...
Xn

1This dataset is provided by Yiren Pan (panyiren@hdu.edu.cn), who is currently writing
a paper about his team’s findings, and originated from [24].
9

Link prediction methods typically require a single large graph as input
to learn the network structure [25, 26]. Therefore, stacking all graphs in the
training dataset allows us to utilize off-the-shelf GNN methods for prediction
with minimal additional computational overhead. The large adjacency ma-
trix can also be stored in a compressed format using sparse data structures.
6.2. Experimental Setup
We conduct two distinct sets of experiments: one using the conventional
dataset splitting approach (train, validation, and test splits), and another
employing the 5-fold cross-validation strategy, given the prevalent challenge
of limited sizes in existing datasets. Here, the traditional split serves as
an ablation study to assess the impact of 5-fold cross-validation, ensuring
robustness in performance evaluation. For both configurations, we maintain
a fixed data split ratio of 80% training, 20% test, and 10% validation. We
also conduct experiments with the approach of [9] as the baseline, which uses
the conventional dataset splitting approach.
We adopt the SEAL framework with the following Pytorch Geometric im-
plementation: batch size = 1, 2-hop neighborhoods and maximum 50 training
epochs with early stopping. The early stopping criterion only engages once
test accuracy shows improvement and exceeds 0 .5000. Thereafter, train-
ing terminates if no improvement ≥0.0001 is observed for three consecutive
epochs. We use the one-hot encoding of node labels generated through DRNL
and concatenate them with the one-hot encoding of the component types.
For the SpiceNetlist dataset, we set the learning rate to 1 e−6. For the
Image2Net dataset, we set the learning rate to 1 e−6 for the baseline and
6e−8 for our approach.
The training graph is constructed using the adjacency stacking method
from Section 6.1. During each experiment, we remove one random vertex
from each test graph and predict its connections. We repeat the experiment
10 times and report the average accuracy and ROC-AUC.
6.3. Results and Analysis
We compare the link prediction results of our work with [9] in Table 4-5
and Figure 6-7. Our method demonstrates significant improvements over the
baseline [9]: 1)on SpiceNetlist, we achieve accuracy gains of 9.84% (con-
ventional split) and 15.05% (5-fold cross-validation); 2)on Image2Net, we
10

achieve accuracy gains of 9.33% (conventional split) and 12.01% (5-fold cross-
validation). The lower performance on SpiceNetlist comparing to Image2Net
is likely due to its smaller training set and limited component variety.
We visualize part of the results in Figure 8. Our project obtained promis-
ing results, indicating that the proposed framework effectively learns netlist
structures and can be utilized for component link prediction tasks.
Table 4: Accuracy comparison across different approaches.
Dataset Baseline[9] Conv. Split 5-fold CV
SpiceNetlist 78.98% 88.82% 94.03%
Image2Net 85.29% 94.62% 97.30%
Table 5: ROC-AUC comparison across different approaches.
Dataset Baseline[9] Conv. Split 5-fold CV
SpiceNetlist 88.14% 96.67% 98.75%
Image2Net 93.79% 99.17% 99.78%
Baseline Conv. Split 5-fold CV
Figure 6: Accuracy variation tendency across different approaches.
7. Conclusion
We propose GNN-ACLP, a novel graph neural networks based framework
with SEAL framework integration and achieve port-level accuracy in analog
circuit link prediction. We also propose Netlist Babel Fish, a netlist format
conversion tool combining RAG with large language model. Additionally,
we create the SpiceNetlist dataset for training and evaluation of circuit link
prediction methods. Experiment results show that our approach achieves
state-of-the-art performance.
11

Baseline Conv. Split 5-fold CVFigure 7: ROC-AUC variation tendency across different approaches.
Figure 8: Visualization of a part of link prediction results.
CRediT authorship contribution statement
Guanyuan Pan : Investigation, Validation, Software, Writing – origi-
nal draft, Visualization. Tiansheng Zhou : Data curation, Visualization.
Bingtao Ma : Writing – review and editing. Yaqi Wang : Conceptual-
ization, Writing – review and editing. Jianxiang Zhao : Writing – review
12

and editing. Shuai Wang : Resources, Supervision, Writing – review and
editing.
Data Availability
Data will be made available on request.
References
[1] H. Chen, M. Liu, X. Tang, K. Zhu, N. Sun, D. Z. Pan, Challenges and
Opportunities toward Fully Automated Analog Layout Design, Journal
of Semiconductors 41 (11) (2020) 111407. doi:10.1088/1674-4926/
41/11/111407 .
URL https://dx.doi.org/10.1088/1674-4926/41/11/111407
[2] A. Samad, M. Qadir, I. Nawaz, M. A. Islam, M. Aleem, A comprehensive
survey of link prediction techniques for social network., EAI Endorsed
Trans. Ind. Networks Intell. Syst. 7 (23) (2020) e3.
[3] D. Liben-Nowell, J. Kleinberg, The link prediction problem for social
networks, in: Proceedings of the twelfth international conference on
Information and knowledge management, 2003, pp. 556–559.
[4] P. R. Genssler, L. Alrahis, O. Sinanoglu, H. Amrouch, HDCircuit:
Brain-Inspired HyperDimensional Computing for Circuit Recognition,
in: 2024 Design, Automation &amp; Test in Europe Conference &amp;
Exhibition (DATE), IEEE, Valencia, Spain, 2024, pp. 1–2. doi:10.
23919/DATE58400.2024.10546587 .
URL https://ieeexplore.ieee.org/document/10546587/
[5] L. Ge, K. K. Parhi, Classification using hyperdimensional computing:
A review, IEEE Circuits and Systems Magazine 20 (2) (2020) 30–47.
doi:10.1109/mcas.2020.2988388 .
URL http://dx.doi.org/10.1109/MCAS.2020.2988388
[6] Z. Luo, T. S. Hy, P. Tabaghi, M. Defferrard, E. Rezaei, R. M. Carey,
R. Davis, R. Jain, Y. Wang, De-hnn: An effective neural model for
circuit netlist representation, in: International Conference on Artificial
Intelligence and Statistics, PMLR, 2024, pp. 4258–4266.
13

[7] J. Kim, S. Oh, S. Cho, S. Hong, Equivariant hypergraph neural net-
works, in: European Conference on Computer Vision, Springer, 2022,
pp. 86–103.
[8] X. Jiang, Y. Zhao, Y. Lin, R. Wang, R. Huang, et al., Circuitnet 2.0: An
advanced dataset for promoting machine learning innovations in realistic
chip design environment, in: The Twelfth International Conference on
Learning Representations, 2023.
[9] A. Said, M. Shabbir, B. Broll, W. Abbas, P. V¨ olgyesi, X. Kout-
soukos, Circuit design completion using graph neural networks, Neu-
ral Computing and Applications 35 (16) (2023) 12145–12157. doi:
10.1007/s00521-023-08346-x .
URL https://link.springer.com/10.1007/s00521-023-08346-x
[10] D. Skouson, A. Keller, M. Wirthlin, Netlist analysis and transformations
using spydrnet, in: Proceedings of the Python in Science Conference,
2020.
[11] J. de Muijnck-Hughes, W. Vanderbauwhede, Wiring Circuits Is Easy
as{0,1, ω}, or Is It..., in: K. Ali, G. Salvaneschi (Eds.), 37th European
Conference on Object-Oriented Programming (ECOOP 2023), Vol. 263
of Leibniz International Proceedings in Informatics (LIPIcs), Schloss
Dagstuhl – Leibniz-Zentrum f¨ ur Informatik, Dagstuhl, Germany, 2023,
pp. 8:1–8:28. doi:10.4230/LIPIcs.ECOOP.2023.8 .
URL https://drops.dagstuhl.de/entities/document/10.4230/
LIPIcs.ECOOP.2023.8
[12] K. Datta, I. Sengupta, H. Rahaman, A post-synthesis optimization
technique for reversible circuits exploiting negative control lines, IEEE
Transactions on Computers 64 (4) (2015) 1208–1214. doi:10.1109/TC.
2014.2315641 .
[13] A. Beg, A. Elchouemi, R. Beg, A collaborative platform for facilitating
standard cell characterization, in: Proceedings of the 2013 IEEE 17th
International Conference on Computer Supported Cooperative Work
in Design (CSCWD), 2013, pp. 202–206. doi:10.1109/CSCWD.2013.
6580963 .
14

[14] M. Hutton, J. Rose, J. Grossman, D. Corneil, Characterization and pa-
rameterized generation of synthetic combinational benchmark circuits,
IEEE Transactions on Computer-Aided Design of Integrated Circuits
and Systems 17 (10) (1998) 985–996. doi:10.1109/43.728919 .
[15] M. Zhang, Y. Chen, Link prediction based on graph neural networks,
Advances in neural information processing systems 31 (2018).
[16] H. Sarhan, A. Arriordaz, Automated analog design
constraint checking, https://semiengineering.com/
automated-analog-design-constraint-checking/ (February 2019).
[17] J. I. Hibshman, T. Weninger, Inherent limits on topology-based link
prediction (2023). arXiv:2301.08792 .
URL https://arxiv.org/abs/2301.08792
[18] J. Chen, H. He, F. Wu, J. Wang, Topology-aware correlations between
relations for inductive link prediction in knowledge graphs, in: Proceed-
ings of the AAAI conference on artificial intelligence, Vol. 35, 2021, pp.
6271–6278.
[19] eCircuit Center, SPICE Summary — ecircuitcenter.com, https://www.
ecircuitcenter.com/SPICEsummary.htm , [Accessed 06-02-2025].
[20] DeepSeek-V2.5: A New Open-Source Model Combining General and
Coding Capabilities — DeepSeek API Docs — api-docs.deepseek.com,
https://api-docs.deepseek.com/news/news0905 , [Accessed 12-02-
2025].
[21] N. C. of Technology Innovation for EDA, 2024 China Postgrad-
uate IC Innovation Competition: EDA Elite Challenge Q10
Guide, https://edaoss.icisc.cn/file/cacheFile/2024/8/7/
905c4088385441fea110889b1fdeb30d.pdf (2024).
[22] F. Salvaire, Pyspice, accessed: 2025-02-12.
URL https://pyspice.fabrice-salvaire.fr/
[23] Z. Tao, Y. Shi, Y. Huo, R. Ye, Z. Li, L. Huang, C. Wu, N. Bai, Z. Yu,
T.-J. Lin, et al., Amsnet: Netlist dataset for ams circuits, in: 2024 IEEE
LLM Aided Design Workshop (LAD), IEEE, 2024, pp. 1–5.
15

[24] 2024 China Postgraduate IC Innovation Competition ·EDA Elite Chal-
lenge Contest (2024).
URL http://edachallenge.cn
[25] P. Veliˇ ckovi´ c, G. Cucurull, A. Casanova, A. Romero, P. Li` o, Y. Bengio,
Graph Attention Networks, International Conference on Learning Rep-
resentations (2018).
URL https://openreview.net/forum?id=rJXMpikCZ
[26] X. Jiang, R. Zhu, P. Ji, S. Li, Co-embedding of nodes and edges with
graph neural networks, IEEE Transactions on Pattern Analysis and Ma-
chine Intelligence 45 (6) (2020) 7075–7086.
16
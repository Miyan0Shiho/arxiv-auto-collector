# GRAIL:Learning to Interact with Large Knowledge Graphs for Retrieval Augmented Reasoning

**Authors**: Ge Chang, Jinbo Su, Jiacheng Liu, Pengfei Yang, Yuhao Shang, Huiwen Zheng, Hongli Ma, Yan Liang, Yuanchun Li, Yunxin Liu

**Published**: 2025-08-07 15:34:41

**PDF URL**: [http://arxiv.org/pdf/2508.05498v1](http://arxiv.org/pdf/2508.05498v1)

## Abstract
Large Language Models (LLMs) integrated with Retrieval-Augmented Generation
(RAG) techniques have exhibited remarkable performance across a wide range of
domains. However, existing RAG approaches primarily operate on unstructured
data and demonstrate limited capability in handling structured knowledge such
as knowledge graphs. Meanwhile, current graph retrieval methods fundamentally
struggle to capture holistic graph structures while simultaneously facing
precision control challenges that manifest as either critical information gaps
or excessive redundant connections, collectively undermining reasoning
performance. To address this challenge, we propose GRAIL: Graph-Retrieval
Augmented Interactive Learning, a framework designed to interact with
large-scale graphs for retrieval-augmented reasoning. Specifically, GRAIL
integrates LLM-guided random exploration with path filtering to establish a
data synthesis pipeline, where a fine-grained reasoning trajectory is
automatically generated for each task. Based on the synthesized data, we then
employ a two-stage training process to learn a policy that dynamically decides
the optimal actions at each reasoning step. The overall objective of
precision-conciseness balance in graph retrieval is decoupled into fine-grained
process-supervised rewards to enhance data efficiency and training stability.
In practical deployment, GRAIL adopts an interactive retrieval paradigm,
enabling the model to autonomously explore graph paths while dynamically
balancing retrieval breadth and precision. Extensive experiments have shown
that GRAIL achieves an average accuracy improvement of 21.01% and F1
improvement of 22.43% on three knowledge graph question-answering datasets. Our
source code and datasets is available at https://github.com/Changgeww/GRAIL.

## Full Text


<!-- PDF content starts -->

GRAIL: Learning to Interact with Large Knowledge Graphs for Retrieval
Augmented Reasoning
Ge Chang1*†, Jinbo Su1*, Jiacheng Liu1*, Pengfei Yang1, Yuhao Shang1,
Huiwen Zheng3, Hongli Ma3, Yan Liang3, Yuanchun Li1, 2‡, Yunxin Liu1
1Institute for AI Industry Research (AIR), Tsinghua University
2Beijing Academy of Artificial Intelligence (BAAI)
3GDS Holdings Limited
{changge.cg.0622, sinianqwq }@gmail.com,
{liyuanchun, liuyunxin }@air.tsinghua.edu.cn
Abstract
Large Language Models (LLMs) integrated with Retrieval-
Augmented Generation (RAG) techniques have exhibited re-
markable performance across a wide range of domains. How-
ever, existing RAG approaches primarily operate on unstruc-
tured data and demonstrate limited capability in handling
structured knowledge such as knowledge graphs. Meanwhile,
current graph retrieval methods fundamentally struggle to
capture holistic graph structures while simultaneously facing
precision control challenges that manifest as either critical in-
formation gaps or excessive redundant connections, collec-
tively undermining reasoning performance. To address this
challenge, we propose GRAIL: Graph- Retrieval Augmented
Interactive Learning, a framework designed to interact with
large-scale graphs for retrieval-augmented reasoning. The
key idea of GRAIL is to train an agent that autonomously
interact with the graph nodes and edges to retrieve the most
relevant information based on the task target. Specifically,
GRAIL integrates LLM-guided random exploration with path
filtering to establish a data synthesis pipeline, where a fine-
grained reasoning trajectory is automatically generated for
each task. Based on the synthesized data, we then employ
a two-stage training process to learn a policy that dynami-
cally decides the optimal actions at each reasoning step. The
overall objective of precision-conciseness balance in graph
retrieval is decoupled into fine-grained process-supervised
rewards to enhance data efficiency and training stability. In
practical deployment, GRAIL adopts an interactive retrieval
paradigm, enabling the model to autonomously explore graph
paths while dynamically balancing retrieval breadth and pre-
cision. Extensive experiments have shown that GRAIL signif-
icantly outperforms existing baselines. Specifically, GRAIL
achieves an average accuracy improvement of 21.01% and
F1improvement of 22.43% on three knowledge graph
question-answering datasets. Our source code and datasets is
available at https://github.com/Changgeww/GRAIL.
*These authors contributed equally.
†Ge Chang, Jinbo Su, Jiacheng Liu, Pengfei Yang, and Yuhao
Shang worked on this project during their internship at AIR, Ts-
inghua University.
‡Corresponding author.Introduction
Large Language Models (LLMs) have demonstrated re-
markable performance across a wide range of natural lan-
guage processing tasks, including text comprehension, con-
tent generation, and question answering (Chang et al. 2024;
Hadi et al. 2023; Naveed et al. 2023). To further improve
their reliability in knowledge-intensive scenarios, Retrieval-
Augmented Generation (RAG) has been proposed, which
enhances LLMs by incorporating external information into
the generation process (Zhang et al. 2023; Lewis et al. 2020;
Gao et al. 2023). RAG works by retrieving the most relevant
documents from a large corpus and feeding them into the
LLM context, thereby providing useful information for LLM
generation without crowding the limited context length.
However, most existing RAG pipelines are designed for
unstructured text corpora, limiting their applicability to
other types of data - a typical example among which is
graph . In fact, a substantial portion of real-world data inher-
ently exhibits a graph structure, such as knowledge graphs,
social networks, recommendation systems, and IoT device
interconnections. Normal textual content can also be parsed
into graphs for more compact and semantically structured
representation (Peng et al. 2024; Procko and Ochoa 2024;
Zhang et al. 2025a).
A growing body of work has investigated integrating
graphs with RAG for graph-based question answering (Jin
et al. 2024; Xu et al. 2024; Chai et al. 2023; Li et al.
2023; Wang et al. 2023; Dehghan et al. 2024). Existing re-
trievers can be roughly categorized into similarity-based,
graph-based, and LLM-based methods (Zhang et al. 2025a).
Specifically, similarity-based methods employ pre-trained
encoders to compute cosine similarity between graph and
text embeddings for retrieval. Graph-based methods typi-
cally leverage graph neural networks (GNNs) trained in spe-
cific graph structures. In contrast, LLM-based approaches
utilize LLMs to further extract and represent graph infor-
mation. Similarity-based retrievers often rely on embedding
or keyword matching while neglecting the graph’s structural
information (Zhu et al. 2024; He et al. 2024), leading toarXiv:2508.05498v1  [cs.AI]  7 Aug 2025

Figure 1: A case study comparing GRAIL with existing
graph retrieval methods.
incomplete coverage or redundant retrieval. Graph-based re-
trievers depend on task-specific training and high-quality la-
beled data (Mavromatis and Karypis 2024), limiting their
scalability. LLM-based retrievers, on the other hand, depend
heavily on the model’s inherent reasoning abilities but are
constrained by context window size (Edge et al. 2024). To
summarize, as illustrated in the Figure 1, existing methods
struggle to simultaneously encode graph structures while
maintaining query adaptability, balance domain generaliza-
tion with task-specific precision, and achieve comprehensive
retrieval without introducing excessive redundancy. These
limitations arise from the absence of interactive and adap-
tive mechanisms that bridge the retrieval and reasoning pro-
cesses.
To address these challenges, we propose GRAIL ,Graph-
Retrieval Augmented Interactive Learning. GRAIL ad-
dresses the challenges of retrieval precision by introducing
a reinforcement learning agent that dynamically navigates
knowledge graphs through iterative, structure-aware explo-
ration and adaptive pruning.
Specifically, to fully leverage the structural characteris-
tics of knowledge graphs and overcome the limitations of
static retrieval, we propose an interactive retrieval frame-
work that enables the retriever to engage in step-wise ex-
ploration of the graph environment. At each decision point,
the retriever selects actions based on the query intent, avail-
able action space, and local graph context, thus supporting
adaptive and structure-aware retrieval. In contrast to conven-
tional methods that retrieve subgraphs in a single pass, our
approach allows for dynamic adjustment of retrieval granu-
larity throughout the reasoning process.
Drawing inspiration from recent advances in reasoning-
enhanced LLMs (Guo et al. 2025; Jaech et al. 2024), we
propose adopting reinforcement learning to enhance the rea-
soning ability of the graph retrieval agent. Directly applying
LLM-based reinforcement learning algorithms (e.g. GRPO)
to our problem is challenging due to the lack of graph-
structured knowledge in LLM pretraining data (Zhang et al.
2025b). To address this, we propose a two-stage training
framework that enhances the model’s graph comprehension
and reasoning capabilities. We first employ LLMs to gen-erate diverse exploratory trajectories over the graph, which
are then filtered using heuristic rules and decomposed into
fine-grained decision sequences suitable for training. Lever-
aging the resulting dataset, the retriever is first trained via
supervised learning to acquire basic alignment capabilities
and then fine-tuned with reinforcement learning to optimize
its ability to perform reasoning and exploration over graph-
structured data. Crucially, this training paradigm enables
the model to learn graph exploration strategies, rather than
merely capturing surface-level linguistic features. In prac-
tice, our method achieves average improvements of 21.01%
in accuracy and 22.43% in F1score across the WebQSP,
CWQ, and MetaQA benchmarks.
In summary, the main contributions of this work are as
follows:
• We propose an interactive retrieval framework that en-
ables adaptive, structure-aware exploration over large-
scale knowledge graphs.
• We develop a scalable pipeline for automatically generat-
ing reinforcement learning data for LLM-based retrievers
in graph environments, without human annotation.
• We design a two-stage training paradigm that combines
supervised and reinforcement learning to enhance the re-
triever’s reasoning ability.
Related Works
Graph Retriever
Current graph retrieval methods can be primarily catego-
rized into three types: (i) LLM-based methods , where
LLMs are employed to process complex graph data into
textual information and generate corresponding embeddings
and labels. Representative work in this area includes meth-
ods such as GraphRAG (Edge et al. 2024), LightRAG (Guo
et al. 2024), and KG-Retriever (Chen et al. 2024b). These
approaches aim to leverage the powerful language under-
standing capabilities of LLMs to enhance graph retrieval.
(ii)Graph-based . This category of methods focuses on
training Graph Neural Networks (GNNs) to achieve em-
bedding alignment between the graph data and textual
queries. Representative works in this domain include GNN-
Ret (Li et al. 2025), SURGE (Kang et al. 2023), and GNN-
RAG (Dong et al. 2024). (iii) Similarity-based methods
This approach focuses on fusing graph structural embed-
dings from GNNs with semantic embeddings from LLMs.
Representative works employing this methodology include
LLAGA (Chen et al. 2024a), G-retriever (He et al. 2024)
and graphGPT (Tang et al. 2024). However, compared to
GRAIL, existing approaches neglect to integrate reasoning
capabilities into graph retrievers for more precise multi-hop
retrieval, while the redundant information induced by multi-
hop operations presents another significant challenge.
RL reasoning
Test-time scaling techniques, such as OpenAI o1 and
DeepSeek R1 (Guo et al. 2025), have brought a major shift
in how large language models (LLMs) are developed and
used. These methods allow models to perform longer chains

Figure 2: The system overview of GRAIL.
of thought (CoT) during inference, which helps them show
more advanced and organized reasoning skills. As a result,
they perform better on difficult tasks like math problems
and competitive programming (Liu et al. 2024; Chowdhery
et al. 2023; Brown et al. 2020; Achiam et al. 2023). The
main factor behind this progress is large-scale reinforce-
ment learning (RL), which plays a central role in building
these reasoning abilities. RL helps the model develop com-
plex reasoning behaviors, such as checking its own answers
and improving them step by step, making the overall rea-
soning process deeper and more effective (Yu et al. 2025).
While these advances have significantly advanced reason-
ing in unstructured domains like mathematics and program-
ming, their application to large graph retrieval and reasoning
remains fundamentally underexplored (Zhang et al. 2025a).
GRAIL tries to transfer RL-based reasoning paradigms to
large-scale graph retrieval.
Method
We present GRAIL, a novel framework designed to con-
struct a highly efficient graph retriever for large language
models through strategic data synthesis and multi-stage
training, seamlessly integrated with an interactive retrieval
strategy. As illustrated in Figure 2, GRAIL comprises three
pivotal stages. During the data processing stage, GRAIL uti-
lizes closed-source large language models (notably GPT-
4o) to generate synthetic yet high-quality graph reason-
ing dataset. These rigorously validated datasets underpin
a two-stage training pipeline: an initial supervised fine-
tuning (SFT) phase to adapt the model to graph-specific
reasoning tasks, followed by reinforcement learning (RL)
to further enhance its inference capabilities and generaliza-
tion over graph-structured data. For production deployment,the framework implements a dynamic interactive retrieval
mechanism to ensure exhaustive extraction of all query-
critical graph patterns.
Data Processing Stage
Through prior research (Zhang et al. 2025b) and our ex-
perimental validation, we have identified a critical limita-
tion: current LLMs lack substantial pretraining on graph-
structured data, which significantly impairs their perfor-
mance on graph-related tasks. This fundamental gap ne-
cessitates extensive fine-tuning with high-quality graph
datasets to cultivate essential graph comprehension capabili-
ties. However, current research (Choubey et al. 2024) reveals
a persistent scarcity of high-quality graph-structured data in
both academic and industrial settings, owing to its inher-
ently complex construction process and heavy reliance on
manual annotation by domain experts. To tackle these chal-
lenges, GRAIL pioneers an innovative pipeline for synthe-
sizing graph reasoning data, effectively addressing the criti-
cal shortage of high-quality graph datasets. Specifically, we
first define three types of actions on graphs. Then, we lever-
age a close-source LLM to interact with the graphs based on
these actions, and retain the correct responses as our synthe-
sized data.
Graph-based Action Design To enable more effective ex-
ploration of graph data for synthetic training data genera-
tion, GRAIL formally defines three fundamental types of
graph exploration operations. We formally define the three
core operations as follows:
Explore Entity : This operation facilitates the systematic
examination of relation-centric contexts surrounding a given
entity. Formally, the operation takes as input a target entity x

and returns the complete set of associated entities and their
corresponding relations, as specified in Equation 1.
Explore (x) ={(x, r, y )|(x, r, y )∈ G}
∪ {(y, r, x )|(y, r, x )∈ G} (1)
Where x is the target entity, r and y denote the connected
relations and neighboring entities respectively, and Grepre-
sents the knowledge graph. When this function is invoked,
the observed entities and relations are added to the percep-
tion window Gp.
Choose Relation : Although the preceding operations
have yielded a collection of entity-relationship sets Gp,
the perception window remains excessively expansive for
LLMs. Directly inputting such lengthy contexts inevitably
incurs performance degradation. Consequently, the objective
of this operation is to prune the Perception window, thereby
distilling a refined subgraph Gsubthat is ultimately fed into
LLMs. This process can be formally articulated as Equa-
tion 2.
Choose (q,Gp) =n
(x, r, y )(x,r,y )∈Gp
∧F(q,(x,r,y ))=1o
(2)
In this equation, qdenotes the user’s current query, and
(x,r,y) represents a triple in Gp. F is a binary classifier that
determines whether each relation is relevant to the query,
outputting 1 if relevant and 0 otherwise. Here, F is imple-
mented using a closed-source LLM.
Finish : When this operation occurs, it indicates that the
information in the current Gsubis sufficient to answer the
user’s query. The exploration process will terminate imme-
diately, and the current Gsubalong with the user query q will
be fed into the LLMs to obtain and return the final answer.
Formally, this operation can be expressed as Equation 3.
Finish (q,Gsub) =Answer (q,Gsub) (3)
Here, Answer(q, Gsub) denotes answering the question q
based on the knowledge graph G.
Graph Reasoning Data Synthesis With the three opera-
tions formally defined, we now proceed to the concrete im-
plementation of graph reasoning data synthesis. In data syn-
thesis, GRAIL employs the current state-of-the-art (SOTA)
closed-source LLM, GPT-4. In each iteration, GRAIL first
inputs a predefined instruction, the current graph state, the
user’s query, and an action description into the large model,
which then generates an action selection accompanied by a
reasoning process. Based on the action output by the LLM,
GRAIL invokes the corresponding graph API to obtain the
result, then updates both GsubandGp. This process iter-
ates until either the maximum step limit is reached or the
model outputs a stop action. Through such multi-turn inter-
actions with the LLM and data generation, we produce a
collection of graph exploration trajectories with reasoning
processes. Only data instances with correct answers are re-
tained to serve as training data for subsequent stages.Training Stage
To enhance the model’s graph comprehension and reason-
ing capabilities, we adopt a two-stage fine-tuning approach
in training GRAIL. In the first stage, we perform supervised
fine-tuning (SFT) using the synthesized reasoning data from
the previous phase, aiming to equip the model with funda-
mental instruction understanding and graph reasoning abil-
ities. Building on existing research showing that reinforce-
ment learning (RL) can enhance models’ reasoning capabil-
ities (Guo et al. 2025) and improve their efficiency in explor-
ing reasoning paths (Yue et al. 2025), we introduce GRPO in
the second training phase of GRAIL, along with a special-
ized data processing pipeline optimized for RL training.
Process Reword RL In practice, we utilize the widely
adopted Group Relative Policy Optimization (GRPO) (Shao
et al. 2024) algorithm for reinforcement learning. For reward
design, existing approaches predominantly rely on outcome-
based reward signals, which have demonstrated remarkable
effectiveness in domains such as mathematical reasoning
and code generation. However, prior studies (Wang et al.
2025; Choudhury 2025; Deng et al. 2024) have shown that
in relatively complex scenarios such as graph retrieval, con-
ventional outcome-based reward signals tend to be overly
sparse. This sparsity hampers effective credit assignment to
early-stage actions, ultimately resulting in inefficient learn-
ing over long action chains. This observation motivates
our adoption of process-level rewards in the training of
GRAIL. However, our experiments reveal that directly ap-
plying process-level rewards on synthetic data leads to a de-
cline in model performance. An analysis of typical failure
cases indicates that this is primarily due to the presence of
noise in the synthetic data, which ultimately results in re-
ward signal misalignment. To address this issue, we intro-
duce a further refinement process on the RL training data.
RL Data Refinement Prior research has demonstrated
that reinforcement learning (RL) is particularly effective at
guiding the selection of optimal exploration paths during
training (Yue et al. 2025). However, our initial data synthesis
pipeline filtered exploration trajectories solely based on an-
swer correctness, leading to substantial redundancy and in-
efficiency in the resulting trajectories. To overcome this lim-
itation, we introduced a Shortest Path Refinement procedure
as a post-processing step for RL training data. This method
aims to retain only the most concise and efficient reasoning
paths among the correct trajectories, thereby improving the
quality and utility of the data used for policy learning.
Specifically, for each problem instance in the RL train-
ing data, we generate the most efficient execution path. This
pruning mechanism removes redundant exploration steps
from the original trajectories. The resulting dataset more
closely adheres to the minimality principle essential for ef-
fective reinforcement learning. Let the exploration path be
defined as T={τi}N
i=1, where each trajectory τiis repre-
sented as τi= (s1, a1, s2, a2, . . . , s Ti). Each path τibegins
from an initial state s1, progresses through a sequence of
actions (at)and transitions to subsequent states, and termi-
nates at state sTi. Here, Tidenotes the length of trajectory τi.
As shown in Equation (4), we define a refinement operator

Rthat operates on the raw path set T, producing a refined
set.
T∗=R(T) (4)
For each refined trajectory τ∗
i∈ T∗, the final state re-
mains semantically equivalent to that of the original τi; Each
τ∗
iis the shortest among all correct trajectories leading to the
same answer.
Formally, the refinement for each τiis given by Equa-
tion (5)
τ∗
i= arg min
τ∈F i|τ| (5)
where Fidenotes the set of all feasible trajectories that
arrive at the same final answer as τi.
Fi=
τFinalState (τ) =FinalState (τi),
CorrectAnswer (τ) =CorrectAnswer (τi)
Thus, the refined dataset T∗retains only the most con-
cise and efficient reasoning paths, explicitly eliminating ex-
ploration redundancy, and better adheres to the principle of
minimality required for reinforcement learning.
Interactive Retriever
To address the inherent trade-off between retrieval depth and
information redundancy in conventional graph RAG sys-
tems, GRAIL introduces an innovative interactive retrieval
mechanism. The proposed model dynamically searches and
prunes the graph structure in an iterative manner, achiev-
ing an optimal balance between retrieval depth and context
length. Specifically, before making each step-wise decision,
the model first performs an extended Chain-of-Thought
(CoT) reasoning process to determine: (1) which action to
select for the current step, and (2) the corresponding pa-
rameters for the chosen action. The operations mainly fall
into two categories: the explore action that further inves-
tigates the graph and incorporates new nodes into the cur-
rent perception window, and the choose relation action that
prunes the observed subgraph to maintain information con-
ciseness. The model dynamically alternates between these
operations until it determines the retrieved graph informa-
tion sufficiently answers the user query, then terminates the
process via the stop action. Through this interactive explo-
ration process, GRAIL achieves efficient subgraph retrieval.
Experiments
Datasets
WebQSP and CWQ. WebQSP is constructed by collect-
ing real-world questions through the Google Suggest API
and annotating them with SPARQL queries against the Free-
base knowledge graph. CWQ extends the WebQSP dataset
by utilizing the same knowledge source while introducing
more complex questions requiring multi-hop reasoning.
MetaQA specializes in movie-domain knowledge-based
question answering, built upon a knowledge graph contain-
ing 135k triples, 43k entities, and 9 relations.Metrics
Following previous studies (Chen et al. 2024b), we employ
Accuracy (Acc) and F1score as evaluation metrics to as-
sess our model’s performance. The calculation of Accuracy
is defined as Equation 6.
Acc. =Σ[E(gti, pred i) == 1]
size(data)(6)
where E denotes a GPT-4-based evaluation function that
determines the semantic equivalence between two input an-
swers, with gtirepresenting the ground-truth answer for the
i-th data instance and pred icorresponding to the model’s
predicted answer for the same instance.
Baselines
To validate the effectiveness of our proposed GRAIL, we
conducted extensive comparative experiments against sev-
eral representative graph retrieval methods. We evaluate the
model’s inherent graph understanding capability through
two configurations: the ”no graph” setting where the model
processes no graph input, and the ”no retriever” setting
where the model receives the entire graph structure directly
as input. For conventional RAG systems, we implement a
text similarity-based multi-hop retrieval approach. Specifi-
cally, our method first retrieves the graph nodes most simi-
lar to the input query, then performs k-hop expansion from
these seed nodes, ultimately feeding the collected informa-
tion to the model. For LLM-GNN integration methods, we
select G-Retriever (He et al. 2024) as the baseline, while for
LLM-based approaches, we choose ToG and LightRAG as
comparison methods.
Implementation Details
We conduct our experiments using three benchmark datasets
(WebQSP, CWQ, and MetaQA) as the source material for
our proposed Data Generation Pipeline. After processing,
we obtain a total of 9035 instances for supervised fine-tuning
and 3504 instances for reinforcement learning. In the SFT
stage, we fine-tune the open-source large language model
Qwen3-8B with a learning rate of 1e−4and train it for 3
epochs. In the subsequent RL stage, we adopt the GRPO al-
gorithm to further optimize the model. We set the training
batch size to 512, the number of training epochs to 15, the
learning rate to 1e−5, the value clipping range (cliprange)
to 0.5, and the KL divergence coefficient to 0.001. The en-
tire RL training phase takes approximately 32 hours on 8
NVIDIA A100 80GB GPUs.
Main Results
The primary experimental results on WebQSP, CWQ, and
MetaQA benchmarks are presented in Table 1. From these
results, we can draw the following conclusions: (1) Our
GRAIL-Retriever framework achieves state-of-the-art per-
formance across all three benchmarks, demonstrating the ef-
fectiveness of our proposed multi-stage training and inter-
active retrieval approach. (2) Compared to the non-retrieval
approach that directly feeds the entire graph into the LLM,

Retriever + GeneratorWebQSP CWQ MetaQA 1-hop MetaQA 2-hop MetaQA 3-hop
Acc F1 Acc F1 Acc F1 Acc F1 Acc F1
No graph + Qwen3-8B 5.16 8.11 6.26 7.35 2.00 2.88 0.07 0.95 0.20 1.19
No retriever + Qwen3-8B 0.25 1.83 0.37 1.29 0.00 0.29 0.00 0.49 0.00 1.25
RAG/1hop + Qwen3-8B 27.89 38.57 12.55 16.18 75.93 86.36 0.77 1.74 4.13 11.99
RAG/2hop + Qwen3-8B 14.07 24.47 7.00 10.55 42.03 55.59 10.07 21.65 2.60 9.42
RAG/3hop + Qwen3-8B 1.54 7.94 0.99 3.12 0.37 3.03 0.13 2.21 0.13 3.45
ToG + Qwen3-8B 6.14 9.79 7.01 9.61 1.37 1.75 0.00 0.00 0.00 0.20
LightRAG + Qwen3-8B 18.39 31.67 16.20 23.09 1.13 1.76 0.00 0.19 0.07 0.40
G-retriever + Qwen3-8B 25.74 35.45 15.38 18.62 0.63 1.60 0.10 0.77 0.03 1.87
GRAIL + Qwen3-8B 36.24 47.88 17.87 23.29 81.50 90.22 53.73 65.60 12.73 29.49
No graph + Llama3.1-8B 8.97 15.69 9.40 11.58 12.20 17.60 1.27 7.77 1.23 8.86
No retriever + Llama3.1-8B 0.18 1.97 0.14 1.29 0.00 0.58 0.00 1.11 0.00 2.94
RAG/1hop + Llama3.1-8B 24.82 35.28 13.85 17.26 60.17 70.84 2.40 5.98 4.03 15.46
RAG/2hop + Llama3.1-8B 11.06 22.94 6.29 10.68 29.07 42.47 4.50 15.06 1.80 11.30
RAG/3hop + Llama3.1-8B 1.04 6.67 0.65 3.43 0.33 3.67 0.17 3.37 0.07 5.76
ToG + Llama3.1-8B 8.85 14.28 8.42 12.33 12.40 15.88 0.00 0.63 1.43 6.10
LightRAG + Llama3.1-8B 15.85 36.66 8.33 15.01 13.13 21.47 0.93 4.33 1.00 6.38
G-retriever + Llama3.1-8B 22.67 32.26 13.91 17.47 0.67 1.56 0.10 0.83 0.10 1.77
GRAIL + Llama3.1-8B 32.31 43.26 17.11 21.17 67.50 76.56 40.17 55.49 10.73 29.55
No graph + Finetuned-8B 9.21 14.88 10.17 12.31 1.63 2.49 0.43 2.64 0.33 4.60
No retriever + Finetuned-8B 0.37 2.26 0.68 1.76 0.00 0.29 0.00 0.34 0.00 1.05
RAG/1hop + Finetuned-8B 28.87 41.48 19.85 26.01 59.50 69.54 1.83 7.44 3.30 18.61
RAG/2hop + Finetuned-8B 14.93 27.76 8.89 14.56 35.83 52.03 7.70 21.21 3.07 14.04
RAG/3hop + Finetuned-8B 1.54 7.81 0.93 4.36 0.57 2.99 0.43 2.31 0.13 4.23
ToG + Finetuned-8B 5.04 9.43 7.54 9.79 2.23 3.44 0.00 0.12 0.10 2.80
LightRAG + Finetuned-8B 17.38 32.59 13.85 19.96 13.77 20.60 0.97 3.71 0.53 4.28
G-retriever + Finetuned-8B 30.34 43.49 22.68 28.38 8.93 11.51 2.33 4.80 0.40 3.31
GRAIL + Finetuned-8B 44.29 58.45 23.62 30.44 82.77 92.04 63.17 76.18 14.70 36.29
Table 1: Overall results of GRAIL and baselines on graph-based QA benchmarks. The best results are highlighted in bold and
the second performance results are indicated by an underscore.
all retrieval-based methods demonstrate superior perfor-
mance. This validates the fundamental challenge in graph-
augmented generation - the inherent complexity of graph
data exceeds the processing capacity of LLMs when han-
dled without selective retrieval. (3) Compared to directly
feeding k-hop information into the model, our approach
achieves significant performance improvements, particularly
on multi-hop reasoning benchmarks. This demonstrates that
our proposed interactive retrieval mechanism effectively fil-
ters meaningful subgraph information, ensuring the input to
the LLM remains both sufficient and concise for generat-
ing higher-quality answers. (4) Compared to training-free
baselines (e.g., ToG, LightRAG), GRAIL-Retriever achieves
substantial improvements, particularly in multi-hop reason-
ing tasks. This demonstrates the effectiveness of our pro-
posed two-stage training strategy for retrieval, proving that
enhancing the reasoning capability of the retrieval model can
further boost retrieval performance.
Analysis
Ablation Study
The GRAIL framework comprises four key components:
SFT, RL, interactive inference, and a shortest-path filter. To
assess the contribution of each module to the overall systemperformance, we conduct a comprehensive set of ablation
experiments. Specifically, we ablate one module at a time
from the system and observe the performance changes. The
experimental results are shown in Table 2.
Ablation of SFT. Our ablation study in the SFT phase re-
veals significant declines in Accuracy and F1scores in all
tasks, confirming our conclusion that the SFT phase equips
the model with fundamental graph comprehension capabili-
ties. This critical component effectively compensates for the
lack of graph understanding cultivation during upstream pre-
training while elevating the performance ceiling for subse-
quent RL optimization.
Ablation of RL. RL ablation experiments demonstrate
consistent declines in both Accuracy and F1scores, with
particularly pronounced degradation on the more challeng-
ing CWQ and MetaQA tasks compared to the relatively
modest performance drop on simpler WebQSP benchmarks.
These results confirm that the RL phase effectively enhances
the model’s graph reasoning capabilities, thereby expanding
its performance boundary on complex multi-hop tasks.
Ablation of interactive inference. Our ablation study on
interactive inference reveals the most significant perfor-
mance degradation in both Accuracy and F1scores for 2-

MethodsDataset
WebQSP CWQMetaQA
1hop 2hop 3hop
Acc F1 Acc F1 Acc F1 Acc F1 Acc F1
Ours 44.29 58.45 23.62 30.44 82.77 92.04 63.17 76.18 14.70 36.29
Ours w/o SFT 1.64 44.41 7.74 8.77 1.27 9.38 6.30 4.12 2.07 4.98
Ours w/o RL 41.77 3.02 13.39 15.97 71.97 80.09 35.93 45.25 .73 11.46
Ours w/o interactive 28.87 41.48 9.85 6.01 59.50 69.54 1.83 7.44 3.30 8.61
Ours w/o shortest path 16.46 19.24 4.12 4.87 39.47 41.06 4.01 6.10 1.34 1.80
Table 2: Abalation studies of GRAIL.
Retriever Train MethodDataset
WebQSP CWQMetaQA
1hop 2hop 3hop
Acc F1 Acc F1 Acc F1 Acc F1 Acc F1
GRAIL with ORM 41.83 53.87 13.47 16.40 72.63 81.12 34.97 45.14 6.43 11.34
GRAIL with PRM 44.29 58.45 23.62 30.44 82.77 92.04 63.17 76.18 14.70 36.29
Table 3: Performance comparison of PRM and ORM training methods.
hop and 3-hop tasks, where the performance approaches that
of direct multihop RAG. This finding demonstrates that in-
teractive inference enables the model to effectively mitigate
the impact of redundant information.
Ablation of shortest-path filter. Our ablation study on the
shortest-path filter reveals the most substantial performance
degradation across all metrics compared to other module ab-
lations. This finding demonstrates that synthetic data with-
out shortest-path filtering introduces significant noise in re-
ward signals, which ultimately leads to: (1) low-precision
RL signals and (2) high-path redundancy. These effects col-
lectively explain the observed performance degradation after
RL training. The results confirm that the shortest-path filter
is essential in our data synthesis pipeline, as it enforces the
minimality principle required for effective RL training.
PRM or ORM?
In conventional RL approaches, prior studies (Zhang et al.
2025a) typically employ Outcome-based Reward Models
(ORMs) due to challenges in obtaining precise per-step re-
ward signals. In contrast, GRAIL leverages advanced pro-
prietary LLM to provide ground truth annotations for each
action step, implementing Process-based Reward Models
(PRMs).
To evaluate the comparative effectiveness of these ap-
proaches in our task domain, we conduct additional exper-
iments, with quantitative results presented in Table 3. The
experimental results demonstrate that the PRM grounded
in proprietary LLMs achieves statistically significant per-
formance improvements over conventional ORMs across all
benchmark tasks. These results indicate that when integrated
with our shortest-path filter, the reward signals generated by
proprietary LLMs achieve sufficient precision for supervised
training in graph retrieval tasks. Furthermore, the more gran-
ular process-based rewards ultimately yield superior train-
015200035005000Retrieved Item Count CWQ0
212001800120000160000 MetaQA-1hopg
raph data averageR
AG/1hopR
AG/2hopR
AG/3hopG
-retrieverG
RAIL0420008000110000140000 MetaQA-2hopg
raph data averageR
AG/1hopR
AG/2hopR
AG/3hopG
-retrieverG
RAIL0620008000110000145000 MetaQA-3hopFigure 3: Number of retrieved knowledge triples in GRAIL
and baselines on correct answers.
ing outcomes. This demonstrates that our proposed PRM
approach is more suitable than mainstream ORM methods
for graph retrieval tasks.
Effective Information Quantification Analysis
To verify the importance of our method in ensuring concise
yet effective information retrieval, we evaluate the retrieval
efficiency of GRAIL compared to baseline approaches by
measuring the number of triples retrieved required to pro-
duce correct answers (see Figure 3). Unlike traditional
methods that often retrieve excessive redundant informa-
tion, GRAIL demonstrates a significant reduction in retrieval
while maintaining higher accuracy. Specifically, our experi-
ments show that GRAIL retrieves only 11.44% of the triples
required by G-Retriever on average while achieving higher

accuracy, demonstrating its ability to balance search depth
and precision for reduced redundancy. This efficiency gain
is critical in real-world applications.
Conclusion
This paper highlights that existing mainstream RAG meth-
ods lack effective adaptation to graph-structured data, while
current graph-aware RAG approaches suffer from severe
retrieval redundancy issues. To address these limitations,
we present GRAIL, a novel framework that combines
an innovative data synthesis pipeline, multi-stage training
strategy, and interactive inference mechanism to signifi-
cantly improve graph information retrieval. Experimental re-
sults demonstrate that GRAIL achieves SOTA performance
across standard graph retrieval benchmarks.
Acknowledgments
This work is supported by National Natural Science Foun-
dation of China (Grant No.62272261), Tsinghua University
(AIR)–AsiaInfo Technologies (China) Inc. Joint Research
Center, and Wuxi Research Institute of Applied Technolo-
gies, Tsinghua University.
References
Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.;
Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.;
Anadkat, S.; et al. 2023. Gpt-4 technical report. arXiv
preprint arXiv:2303.08774 .
Brown, T.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J. D.;
Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell,
A.; et al. 2020. Language models are few-shot learners. Ad-
vances in neural information processing systems , 33: 1877–
1901.
Chai, Z.; Zhang, T.; Wu, L.; Han, K.; Hu, X.; Huang, X.; and
Yang, Y . 2023. Graphllm: Boosting graph reasoning ability
of large language model. arXiv preprint arXiv:2310.05845 .
Chang, Y .; Wang, X.; Wang, J.; Wu, Y .; Yang, L.; Zhu, K.;
Chen, H.; Yi, X.; Wang, C.; Wang, Y .; et al. 2024. A survey
on evaluation of large language models. ACM transactions
on intelligent systems and technology , 15(3): 1–45.
Chen, R.; Zhao, T.; Jaiswal, A.; Shah, N.; and Wang, Z.
2024a. Llaga: Large language and graph assistant. arXiv
preprint arXiv:2402.08170 .
Chen, W.; Bai, T.; Su, J.; Luan, J.; Liu, W.; and Shi,
C. 2024b. Kg-retriever: Efficient knowledge indexing for
retrieval-augmented large language models. arXiv preprint
arXiv:2412.05547 .
Choubey, P. K.; Su, X.; Luo, M.; Peng, X.; Xiong, C.; Le, T.;
Rosenman, S.; Lal, V .; Mui, P.; Ho, R.; et al. 2024. Distill-
SynthKG: Distilling Knowledge Graph Synthesis Work-
flow for Improved Coverage and Efficiency. arXiv preprint
arXiv:2410.16597 .
Choudhury, S. 2025. Process reward models for llm
agents: Practical framework and directions. arXiv preprint
arXiv:2502.10325 .Chowdhery, A.; Narang, S.; Devlin, J.; Bosma, M.; Mishra,
G.; Roberts, A.; Barham, P.; Chung, H. W.; Sutton, C.;
Gehrmann, S.; et al. 2023. Palm: Scaling language model-
ing with pathways. Journal of Machine Learning Research ,
24(240): 1–113.
Dehghan, M.; Alomrani, M.; Bagga, S.; Alfonso-Hermelo,
D.; Bibi, K.; Ghaddar, A.; Zhang, Y .; Li, X.; Hao, J.; Liu, Q.;
Lin, J.; Chen, B.; Parthasarathi, P.; Biparva, M.; and Reza-
gholizadeh, M. 2024. EWEK-QA : Enhanced Web and Ef-
ficient Knowledge Graph Retrieval for Citation-based Ques-
tion Answering Systems. In Ku, L.-W.; Martins, A.; and
Srikumar, V ., eds., Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume 1:
Long Papers) , 14169–14187. Bangkok, Thailand: Associa-
tion for Computational Linguistics.
Deng, Z.; Dou, Z.; Zhu, Y .; Wen, J.-R.; Xiong, R.; Wang, M.;
and Chen, W. 2024. From novice to expert: Llm agent pol-
icy optimization via step-wise reinforcement learning. arXiv
preprint arXiv:2411.03817 .
Dong, Y .; Wang, S.; Zheng, H.; Chen, J.; Zhang, Z.; and
Wang, C. 2024. Advanced rag models with graph structures:
Optimizing complex knowledge reasoning and text genera-
tion. In 2024 5th International Symposium on Computer En-
gineering and Intelligent Communications (ISCEIC) , 626–
630. IEEE.
Edge, D.; Trinh, H.; Cheng, N.; Bradley, J.; Chao, A.;
Mody, A.; Truitt, S.; Metropolitansky, D.; Ness, R. O.; and
Larson, J. 2024. From local to global: A graph rag ap-
proach to query-focused summarization. arXiv preprint
arXiv:2404.16130 .
Gao, Y .; Xiong, Y .; Gao, X.; Jia, K.; Pan, J.; Bi, Y .; Dai, Y .;
Sun, J.; Wang, H.; and Wang, H. 2023. Retrieval-augmented
generation for large language models: A survey. arXiv
preprint arXiv:2312.10997 , 2(1).
Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang, R.; Xu, R.;
Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; et al. 2025. Deepseek-r1:
Incentivizing reasoning capability in llms via reinforcement
learning. arXiv preprint arXiv:2501.12948 .
Guo, Z.; Xia, L.; Yu, Y .; Ao, T.; and Huang, C. 2024. Ligh-
trag: Simple and fast retrieval-augmented generation. arXiv
preprint arXiv:2410.05779 .
Hadi, M. U.; Qureshi, R.; Shah, A.; Irfan, M.; Zafar, A.;
Shaikh, M. B.; Akhtar, N.; Wu, J.; Mirjalili, S.; et al. 2023. A
survey on large language models: Applications, challenges,
limitations, and practical usage. Authorea Preprints .
He, X.; Tian, Y .; Sun, Y .; Chawla, N.; Laurent, T.; LeCun,
Y .; Bresson, X.; and Hooi, B. 2024. G-retriever: Retrieval-
augmented generation for textual graph understanding and
question answering. Advances in Neural Information Pro-
cessing Systems , 37: 132876–132907.
Jaech, A.; Kalai, A.; Lerer, A.; Richardson, A.; El-Kishky,
A.; Low, A.; Helyar, A.; Madry, A.; Beutel, A.; Carney,
A.; et al. 2024. Openai o1 system card. arXiv preprint
arXiv:2412.16720 .
Jin, B.; Liu, G.; Han, C.; Jiang, M.; Ji, H.; and Han, J. 2024.
Large language models on graphs: A comprehensive survey.
IEEE Transactions on Knowledge and Data Engineering .

Kang, M.; Kwak, J. M.; Baek, J.; and Hwang, S. J.
2023. Knowledge graph-augmented language models for
knowledge-grounded dialogue generation. arXiv preprint
arXiv:2305.18846 .
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,
T.; et al. 2020. Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in neural infor-
mation processing systems , 33: 9459–9474.
Li, Y .; Li, Z.; Wang, P.; Li, J.; Sun, X.; Cheng, H.; and
Yu, J. X. 2023. A survey of graph meets large lan-
guage model: Progress and future directions. arXiv preprint
arXiv:2311.12399 .
Li, Z.; Guo, Q.; Shao, J.; Song, L.; Bian, J.; Zhang, J.; and
Wang, R. 2025. Graph neural network enhanced retrieval for
question answering of large language models. In Proceed-
ings of the 2025 Conference of the Nations of the Americas
Chapter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Papers) ,
6612–6633.
Liu, A.; Feng, B.; Xue, B.; Wang, B.; Wu, B.; Lu, C.; Zhao,
C.; Deng, C.; Zhang, C.; Ruan, C.; et al. 2024. Deepseek-v3
technical report. arXiv preprint arXiv:2412.19437 .
Mavromatis, C.; and Karypis, G. 2024. Gnn-rag: Graph
neural retrieval for large language model reasoning. arXiv
preprint arXiv:2405.20139 .
Naveed, H.; Khan, A. U.; Qiu, S.; Saqib, M.; Anwar, S.;
Usman, M.; Akhtar, N.; Barnes, N.; and Mian, A. 2023. A
comprehensive overview of large language models. ACM
Transactions on Intelligent Systems and Technology .
Peng, B.; Zhu, Y .; Liu, Y .; Bo, X.; Shi, H.; Hong, C.; Zhang,
Y .; and Tang, S. 2024. Graph retrieval-augmented genera-
tion: A survey. arXiv preprint arXiv:2408.08921 .
Procko, T. T.; and Ochoa, O. 2024. Graph retrieval-
augmented generation for large language models: A survey.
In2024 Conference on AI, Science, Engineering, and Tech-
nology (AIxSET) , 166–169. IEEE.
Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang,
H.; Zhang, M.; Li, Y .; Wu, Y .; et al. 2024. Deepseekmath:
Pushing the limits of mathematical reasoning in open lan-
guage models. arXiv preprint arXiv:2402.03300 .
Tang, J.; Yang, Y .; Wei, W.; Shi, L.; Su, L.; Cheng, S.; Yin,
D.; and Huang, C. 2024. Graphgpt: Graph instruction tun-
ing for large language models. In Proceedings of the 47th
International ACM SIGIR Conference on Research and De-
velopment in Information Retrieval , 491–500.
Wang, H.; Leong, C. T.; Wang, J.; Wang, J.; and Li, W. 2025.
SPA-RL: Reinforcing LLM Agents via Stepwise Progress
Attribution. arXiv preprint arXiv:2505.20732 .
Wang, K.; Duan, F.; Wang, S.; Li, P.; Xian, Y .; Yin, C.; Rong,
W.; and Xiong, Z. 2023. Knowledge-driven cot: Exploring
faithful reasoning in llms for knowledge-intensive question
answering. arXiv preprint arXiv:2308.13259 .
Xu, Y .; He, S.; Chen, J.; Wang, Z.; Song, Y .; Tong, H.;
Liu, G.; Zhao, J.; and Liu, K. 2024. Generate-on-Graph:Treat LLM as both Agent and KG for Incomplete Knowl-
edge Graph Question Answering. In Al-Onaizan, Y .; Bansal,
M.; and Chen, Y .-N., eds., Proceedings of the 2024 Confer-
ence on Empirical Methods in Natural Language Process-
ing, 18410–18430. Miami, Florida, USA: Association for
Computational Linguistics.
Yu, Q.; Zhang, Z.; Zhu, R.; Yuan, Y .; Zuo, X.; Yue, Y .; Dai,
W.; Fan, T.; Liu, G.; Liu, L.; et al. 2025. Dapo: An open-
source llm reinforcement learning system at scale. arXiv
preprint arXiv:2503.14476 .
Yue, Y .; Chen, Z.; Lu, R.; Zhao, A.; Wang, Z.; Song, S.;
and Huang, G. 2025. Does reinforcement learning really in-
centivize reasoning capacity in llms beyond the base model?
arXiv preprint arXiv:2504.13837 .
Zhang, Q.; Chen, S.; Bei, Y .; Yuan, Z.; Zhou, H.; Hong,
Z.; Dong, J.; Chen, H.; Chang, Y .; and Huang, X. 2025a.
A Survey of Graph Retrieval-Augmented Generation for
Customized Large Language Models. arXiv preprint
arXiv:2501.13958 .
Zhang, Y .; Li, Y .; Cui, L.; Cai, D.; Liu, L.; Fu, T.; Huang,
X.; Zhao, E.; Zhang, Y .; Chen, Y .; et al. 2023. Siren’s song
in the AI ocean: a survey on hallucination in large language
models. arXiv preprint arXiv:2309.01219 .
Zhang, Y .-F.; Lu, X.; Hu, X.; Fu, C.; Wen, B.; Zhang, T.;
Liu, C.; Jiang, K.; Chen, K.; Tang, K.; et al. 2025b. R1-
reward: Training multimodal reward model through stable
reinforcement learning. arXiv preprint arXiv:2505.02835 .
Zhu, X.; Guo, X.; Cao, S.; Li, S.; and Gong, J. 2024. Struc-
tuGraphRAG: Structured Document-Informed Knowledge
Graphs for Retrieval-Augmented Generation. In Proceed-
ings of the AAAI Symposium Series , volume 4, 242–251.
# Human Cognition Inspired RAG with Knowledge Graph for Complex Problem Solving

**Authors**: Yao Cheng, Yibo Zhao, Jiapeng Zhu, Yao Liu, Xing Sun, Xiang Li

**Published**: 2025-03-09 11:50:39

**PDF URL**: [http://arxiv.org/pdf/2503.06567v1](http://arxiv.org/pdf/2503.06567v1)

## Abstract
Large language models (LLMs) have demonstrated transformative potential
across various domains, yet they face significant challenges in knowledge
integration and complex problem reasoning, often leading to hallucinations and
unreliable outputs. Retrieval-Augmented Generation (RAG) has emerged as a
promising solution to enhance LLMs accuracy by incorporating external
knowledge. However, traditional RAG systems struggle with processing complex
relational information and multi-step reasoning, limiting their effectiveness
in advanced problem-solving tasks. To address these limitations, we propose
CogGRAG, a cognition inspired graph-based RAG framework, designed to improve
LLMs performance in Knowledge Graph Question Answering (KGQA). Inspired by the
human cognitive process of decomposing complex problems and performing
self-verification, our framework introduces a three-stage methodology:
decomposition, retrieval, and reasoning with self-verification. By integrating
these components, CogGRAG enhances the accuracy of LLMs in complex problem
solving. We conduct systematic experiments with three LLM backbones on four
benchmark datasets, where CogGRAG outperforms the baselines.

## Full Text


<!-- PDF content starts -->

Human Cognition Inspired RAG with Knowledge Graph for Complex
Problem Solving
Yao Cheng, Yibo Zhao, Jiapeng Zhu, Yao Liu, Xing Sun, Xiang Li
Abstract
Large language models (LLMs) have demon-
strated transformative potential across various
domains, yet they face significant challenges
in knowledge integration and complex problem
reasoning, often leading to hallucinations and
unreliable outputs. Retrieval-Augmented Gen-
eration (RAG) has emerged as a promising solu-
tion to enhance LLMs accuracy by incorporat-
ing external knowledge. However, traditional
RAG systems struggle with processing com-
plex relational information and multi-step rea-
soning, limiting their effectiveness in advanced
problem-solving tasks. To address these limi-
tations, we propose CogGRAG, a cognition in-
spired graph-based RAG framework, designed
to improve LLMs performance in Knowledge
Graph Question Answering (KGQA). Inspired
by the human cognitive process of decompos-
ing complex problems and performing self-
verification, our framework introduces a three-
stage methodology: decomposition, retrieval,
and reasoning with self-verification. By in-
tegrating these components, CogGRAG en-
hances the accuracy of LLMs in complex prob-
lem solving. We conduct systematic exper-
iments with three LLM backbones on four
benchmark datasets, where CogGRAG outper-
forms the baselines.
1 Introduction
As a foundational technology for artificial general
intelligence (AGI), large language models (LLMs)
have achieved remarkable success in practical ap-
plications, demonstrating transformative potential
across a wide range of domains (Touvron et al.,
2023; AI@Meta, 2024; Yang et al., 2024). Their
ability to process, generate, and reason with natural
language has enabled significant advancements in
areas such as machine translation (Zhu et al., 2023),
text summarization (Basyal and Sanghvi, 2023),
and question answering (Pan et al., 2023). De-
spite their impressive performance, LLMs still facesignificant limitations in knowledge integration be-
yond their pre-trained data boundaries. These lim-
itations often lead to the generation of plausible
but factually incorrect responses, a phenomenon
commonly referred to as hallucinations , which un-
dermines the reliability of LLMs in critical appli-
cations.
To mitigate hallucinations, Retrieval-Augmented
Generation (RAG) (Niu et al., 2023; Gao et al.,
2023; Edge et al., 2024) has emerged as a promis-
ing paradigm, significantly improving the accuracy
and reliability of LLM-generated contents through
the integration of external knowledge. However,
while RAG successfully mitigates certain aspects
of hallucination, it still exhibits inherent limita-
tions in processing complex relational information.
As shown in Figure 1 (a), the core limitation of
standard RAG systems lies in their reliance on
vector-based similarity matching, which processes
knowledge segments as isolated units without cap-
turing their contextual interdependencies or seman-
tic relationships (Jin et al., 2024b). Consequently,
traditional RAG implementations are inadequate
for supporting advanced reasoning capabilities in
LLMs, particularly in scenarios requiring complex
problem-solving, multi-step inference, or sophisti-
cated knowledge integration.
Recently, graph-based RAG (Edge et al., 2024;
Ma et al., 2024; Jin et al., 2024b; Mavromatis and
Karypis, 2024) has been proposed to address the
limitations of conventional RAG systems by incor-
porating deep structural information from external
knowledge sources. These approaches typically
utilize knowledge graphs (KGs) to model complex
relation patterns within external knowledge bases,
employing structured triple representations (<en-
tity, relation, entity>) to integrate fragmented infor-
mation across multiple document segments. While
graph-based RAG has shown promising results in
mitigating hallucination and improving factual ac-
curacy, several challenges remain unresolved:arXiv:2503.06567v1  [cs.LG]  9 Mar 2025

Manchester United……Jose Mourinho……
Question: “The football manager who recruited David Beckham managed Manchester United during what timeframe?”
LLM
IcannotansweryourquestionsinceIdonothaveenoughinformation.
LLM
From2016to2018.
Manchester United…...    David Beckham …...        Alex Ferguson…...    Similarity Score：0.8Similarity Score：0.8Similarity Score：0.05
External Knowledge Sources
David Beckham……David Beckham was recruited by Jose Mourinho.
Jose Mourinho managed Manchester United.Jose Mourinho managed Manchester United from 2016 to 2018.
Answer: “1986-2013.”
(a) Retrieval-Augmented Generation for Enhancing LLMs(b) Iterative Retrieval-Augmented Generation for Enhancing LLMsFigure 1: Representative workflow of two Retrieval-Augmented Generation paradigms for enhancing LLMs.
•Complex Problem Reasoning. Complex prob-
lems cannot be resolved through simple queries;
they typically require multi-step reasoning to de-
rive the final answer. While previous work has
sought to enhance reasoning capabilities through
methods such as chain-of-thought (CoT) reasoning
or multi-hop information retrieval, these search-
based and iterative approaches still face inherent
disadvantages (Jin et al., 2024b; Mavromatis and
Karypis, 2024). As shown in Figure 1 (b), each
step in the iterative process relies on the result of
the previous step, indicating that errors occurred at
previous steps can propagate to subsequent steps.
•Hallucinations. Despite the integration of ex-
ternal knowledge sources, large language models
(LLMs) remain prone to generating inaccurate or
fabricated responses when confronted with retrieval
errors or insufficient knowledge coverage. How-
ever, prior research has lacked mechanisms for self-
verification, which undermines the reliability of
graph-based RAG in real-world applications.
To address these challenges, we propose Cog-
GRAG, a Cognition inspired Graph RAG frame-
work, designed to enhance the complex problem-
solving capabilities of LLMs in Knowledge Graph
Question Answering (KGQA) tasks. The frame-
work introduces a three-stage methodology: (1) De-
composition , inspired by human problem-solvingstrategies, breaks down complex problems top-
down into smaller, simpler sub-problems and forms
a mind map. (2) Retrieval , built on all the decom-
posed sub-problems, this stage extracts multi-level
structured information from external knowledge
sources using knowledge graphs. This stage oper-
ates at both local and global levels: local retrieval
identifies relevant information for individual sub-
problems, while global retrieval establishes con-
nections between multiple sub-problems, ensuring
detailed and comprehensive knowledge integration.
(3)Reasoning with Self-Verification , mimicking
human self-reflection, this stage evaluates the rea-
soning process and verifies the accuracy of interme-
diate and final results. CogGRAG emulates human
cognitive processes in tackling complex problems
by employing a top-down decomposition strategy,
comprehensive knowledge retrieval, and bottom-
up reasoning with validation. By integrating these
components, CogGRAG enhances the accuracy and
reliability of LLM-generated content on KGQA
tasks.
2 Related Work
Reasoning with LLM Prompting. Recent ad-
vancements in prompt engineering have demon-
strated that state-of-the-art prompting techniques
can significantly enhance the reasoning capabil-

External Knowledge Sources
!!!
"#"#!$"###"#%#"#&#"#'#!(!!)!
Q!!"!#"!""
!"!!!!!#!!$!!%!
Question: “The football manager who recruited David Beckham managed Manchester United during what timeframe?”
Answer: “1986-2013.”
Decomposition“WhorecruitedDavidBeckham?”“WhorecruitedDavidBeckhamwhilemanagingManchesterUnited?”Mind Map
ExtractionKeys
RetrievalTriplesReasoningSelf-VerificationLLM
LLMLLM
LLM
Figure 2: The overall process of CogGRAG. Given a target question Q, CogGRAG first prompts the LLM
to decompose it into a hierarchy of sub-problems in a top-down manner, constructing a structured mind map.
Subsequently, CogGRAG prompts the LLM to extract both local and global key information from these sub-
problems. Finally, CogGRAG guides the LLM to perform bottom-up reasoning and verification based on the mind
map and the retrieved knowledge, until the final answer is derived.
ities of LLMs on complex problems (Wei et al.,
2022; Yao et al., 2024; Besta et al., 2024). Chain
of Thought (CoT (Wei et al., 2022) explores how
generating a chain of thought—a series of interme-
diate reasoning steps—significantly improves the
ability of large language models to perform com-
plex reasoning. Tree of Thoughts (ToT) (Yao et al.,
2024) introduces a new framework for language
model inference that generalizes the popular Chain
of Thought approach to prompting language mod-
els, enabling exploration of coherent units of text
(thoughts) as intermediate steps toward problem-
solving. The Graph of Thoughts (GoT) (Besta
et al., 2024) models the information generated by
a LLM as an arbitrary graph, enabling LLM rea-
soning to more closely resemble human thinking
or brain mechanisms. However, these methods re-
main constrained by the limitations of the model’s
pre-trained knowledge base and are unable to ad-
dress hallucination issues stemming from the lack
of access to external, up-to-date information.
Knowledge Graph Augmented LLM.
KGs (Vrande ˇci´c and Krötzsch, 2014) offer distinct
advantages in enhancing LLMs with structured
external knowledge. Early graph-based RAGmethods (Edge et al., 2024; He et al., 2024; Hu
et al., 2024; Wu et al., 2023; Jin et al., 2024a)
demonstrated this potential by retrieving and
integrating structured, relevant information from
external sources, enabling LLMs to achieve
superior performance on knowledge-intensive
tasks. However, these methods exhibit notable
limitations when applied to complex problems.
Recent advancements have introduced methods
like chain-of-thought prompting to enhance
LLMs’ reasoning on complex problems (Sun
et al., 2023; Ma et al., 2024; Jin et al., 2024b).
Think-on-Graph (Sun et al., 2023) introduces a
new approach that enables the LLM to iteratively
execute beam search on the KGs, discover the most
promising reasoning paths, and return the most
likely reasoning results. ToG-2 (Ma et al., 2024)
achieves deep and faithful reasoning in LLMs
through an iterative knowledge retrieval process
that involves collaboration between contexts and
the KGs. GRAPH-COT (Jin et al., 2024b) propose
a simple and effective framework to augment
LLMs with graphs by encouraging LLMs to
reason on the graph iteratively. However, iterative
reasoning processes are prone to error propagation,

as errors cannot be corrected once introduced.
3 Preliminaries
3.1 Problem Definition
In this paper, we focus on the KGQA task. The
input of the task is the question Q, and the out-
put is the answer A. Given the input question Q,
we aim to design a graph-based RAG framework
with an LLM backbone pθto enhance the com-
plex problem-solving capabilities and generate the
answer A.
3.2 Mind Map
A mind map can be conceptualized as a tree-like
structure that hierarchically preserves questions de-
composed in a top-down manner. Formally, a mind
mapMcan be defined as a set of nodes. Each node
m= (q, t, s )∈Mcontains a question q, the level
t∈[0, T]of the question in the mind map, and
the state s∈ {“End”, “Continue” }of the ques-
tion, where Tis the maximum level of the mind
map and state indicates whether the question re-
quires further decomposition. The target question
Qserves as the root node m0, i.e., q0=Q, t 0= 0
ands0=“Continue”.
4 Methods
CogGRAG implements the graph-based RAG
paradigm, inspired by human cognitive processes.
Specifically, it simulates human cognitive strategies
by decomposing complex questions into a struc-
tured mind map and incorporates self-verification
mechanisms during the reasoning process. The
framework of CogGRAG consists of three key
steps: decomposition, retrieval, and reasoning with
self-verification. The overall workflow of the
framework is illustrated in Figure 2.
4.1 Decomposition
When confronted with complex question, humans
typically decompose them into a series of simpler
sub-questions and address them incrementally. In-
spired by this cognitive strategy, in this paper, we
decompose the complex question Qin a top-down
manner, which are logically interconnected and
form a structured mind map. The resulting mind
map aids in identifying the necessary information
required to answer, even if such information is not
explicitly mentioned in the original question. For
instance, in the question “ The football managerwho recruited David Beckham managed Manch-
ester United during what timeframe? ”, the key en-
tity “Alex Ferguson” (the manager who recruited
David Beckham) might be overlooked by tradi-
tional matching-based RAG methods, failing to
derive the correct answer.
Concretely, we begin at the root level t= 0. For
each question qtat an intermediate level t, if its
statest=“Continue” , then CogGRAG prompts
LLMs to decompose the question into several sub-
questions {qt+1
j, j= 1, ..., N}, each paired with
a corresponding states st+1
jat the next level t+
1. Formally, this process can be represented as
follows:
{(qt+1
j, st+1
j), j= 1, ..., N}
=Decompose (qt, pθ, prompt 1),(1)
where Nis the number of sub-questions, which
is adaptively determined by LLMs based on the
logic of each sub-question. prompt 1describes the
question decomposition task in natural language,
as shown in Table 6. After decomposition, each
sub-question and its state are added as a new node
(qt+1
j, t+ 1, st+1
j)at level t+ 1. Decomposition is
terminated when all the states of the child nodes
are “End”, constructing a mind map M.
David Beckham(Manager,  manage,  Manchester United)[(Manager,  manage,  Manchester United), (Manager, recruited, David Beckham)]
Figure 3: The key information extraction with local and
global level.
4.2 Retrieval
At the retrieval step, CogGRAG leverages the
mind map generated from question decomposition
to retrieve relevant information from an external
KG. Firstly, to ensure detailed and comprehensive
knowledge integration, we extract key information
from all questions in the mind map from two dis-
tinct granularity levels to support subsequent re-
trieval processes. As shown in Figure 3, entities
and triples are extracted from each individual ques-
tion at the local level , while inter-question relation-
ships are captured at the global level and repre-
sented as subgraphs. This process can be formal-
ized as follows:
Keys =Extract (M, p θ, prompt 2), (2)

where Keys consists of the sets of entities, triples
and subgraphs extracted from the mind map M,
guided by the extraction workflow outlined in
prompt 2, as shown in Table 7 and Table 8.
Next, based on the extracted key information
Keys , CogGRAG aims to retrieve relevant struc-
tured knowledge (triples) from the KG. Specifically,
we expand each entity in the entity set of Keys to
its neighbors in the KG, forming triples that link
the entity to its neighboring entities via relation-
ships. These retrieved tuples are added to a candi-
date tuple set ˜T. Subsequently, to reduce redun-
dant and noisy information, we employ the triples
and the subgraphs in Keys to prune the triples in
˜Tthrough similarity matching. In practice, only
tuples in ˜Twith similarity scores exceeding a pre-
defined threshold εare retained, resulting in the
final triples set T. This process can be formalized
as follows:
T=n
x∈˜T | ∃y∈Triple ∪Subgraph,
sim(x, y)> εo
,(3)
where Triple ∈Keys represents the set of triples
extracted from individual questions, Subgraph ∈
Keys denotes the set of subgraphs extracted from
inter-question relationships, sim(·)is implemented
using cosine similarity, with εbeing a hyper-
parameter. This pruning ensures that the tuples
retrieved from the KG remain both concise and
relevant, enhancing the accuracy of downstream
reasoning tasks.
4.3 Reasoning with Self-Verification
During the inference phase, CogGRAG utilizes the
triples Tretrieved in the previous step to reason
and generate answers for the input question. Draw-
ing upon the self-verification theory (Frederick,
2005), which posits that humans engage in self-
verification to validate their reasoning outcomes
during the cognitive process, we design a dual-
process framework inspired by dual-process theo-
ries (Vaisey, 2009). To this end, we incorporate two
LLMs into this step: one dedicated to the reasoning
process and the other to the self-verification of the
reasoning results. This approach ensures a more
robust and reliable inference mechanism by mim-
icking the human cognitive process of reasoning
followed by critical evaluation.
Specifically, given the triples set Tand the
mind map M, we prompt the LLM to addressall sub-questions in the mind map in a bottom-
up manner until reaching the root node, which
corresponds to the target question Q. At each
level t, the reasoning model LLM resinfers
and generates an answer atfor the question qt,
where at=LLM res(T, pθ, qt,ˆM, prompt 3)and
ˆM=
(qT,ˆaT),(qT−1,ˆaT−1), ...,(qt+1,ˆat+1)	
comprises all the question-answer pairs that have
already been successfully reasoned through. Subse-
quently, the verification model LLM verreflects on
the current reasoning results at, aiming to identify
potential errors or inconsistencies in the reasoning
process based on the accumulated reasoning path.
If the verification model LLM verdetects an error
at the current step t, it will prompt the reasoning
model LLM resto re-think and generate a new an-
swer ˆat, else ˆat=at. Notably, during reasoning,
we explicitly prompt the LLM to respond with "I
don’t know" for questions it cannot answer, rather
than generating incorrect responses. This helps
mitigate the hallucination problem commonly ob-
served in LLMs during reasoning tasks. Ultimately,
CogGRAG infers the answer Ato the target ques-
tionQ, where A= ˆa0. In summary, reasoning
with self-verification employs a bottom-up reason-
ing process based on the mind map within a dual-
process framework. All prompt templates used in
CogGRAG can be found in the Appendix D.
Table 1: Datasets statistics.
Domain DatasetData Split
Train Dev Test
WikipediaHotpotQA 90564 7405 7405
WebQSP 3098 0 1639
CWQ 27,734 3480 3475
Academic GRBENCH-Academic 850
E-commerce GRBENCH-Amazon 200
Literature GRBENCH-Goodreads 240
Healthcare GRBENCH-Disease 270
5 Experiments
5.1 Experimental Settings
Datasets and evaluation metrics. In order to test
CogGRAG’s complex problem-solving capabili-
ties on KGQA tasks, we evaluate CogGRAG on
three widely used complex KGQA benchmarks: (1)
HotpotQA (Yang et al., 2018), (2) WebQSP (Yih
et al., 2016), (3) CWQ (Talmor and Berant, 2018).
Following previous work (Li et al., 2023), full
Wikidata (Vrande ˇci´c and Krötzsch, 2014) is used

Table 2: Overall results of our CogGRAG on three KBQA datasets. The best score on each dataset is highlighted.
Type MethodHotpotQA CWQ WebQSP
RL EM RL EM RL EM
Without external knowledge graph
LLM-onlyDirect 19.1% 17.3% 31.4% 28.8% 51.4% 47.9%
CoT 23.3% 20.8% 35.1% 32.7% 55.2% 51.6%
With external knowledge graph
LLM+KGDirect+KG 27.5% 23.7% 39.7% 35.1% 52.5% 49.3%
CoT+KG 28.7% 25.4% 42.2% 37.6% 52.8% 48.1%
Graph-based RAGToG 29.3% 26.4% 54.6% 49.1% 61.7% 57.4%
MindMap 27.9% 25.6% 46.7% 43.7% 56.6% 53.1%
Ours CogGRAG 34.4% 30.7% 56.3% 53.4% 59.8% 56.1%
as structured knowledge sources for all of these
datasets. Considering that Wikidata is commonly
used for training LLMs, there is a need for a
domain-specific QA dataset that is not exposed
during the pretraining process of LLMs in order to
better evaluate the performance. Thus, we also test
CogGRAG on a recently released domain-specific
dataset GRBENCH (Jin et al., 2024b). All meth-
ods need interact with domain-specific graphs con-
taining rich knowledge to solve the problem in this
dataset. The statistics and details of these datasets
can734 be found in Table 1. For all datasets, we
use two evaluation metrics: (1) Exact match(EM) :
measures whether the predicted answer or result
matches the target answer exactly. (2) Rouge-
L(RL) : measures the longest common subsequence
of words between the responses and the ground
truth answers.
Baselines. In our main results, we compare Cog-
GRAG with three types state-of-the-art methods:
(1)LLM-only methods without external knowl-
edge, including direct reasoning and CoT (Wei
et al., 2022) by LLM. (2) LLM+KG methods
integrate relevant knowledge retrieved from the
KG into the LLM to assist in reasoning, including
direct reasoning and CoT by LLM. (3) Graph-
based RAG methods allow KGs and LLMs to
work in tandem, complementing each other’s capa-
bilities at each step of graph reasoning, including
Mindmap (Wen et al., 2023), Think-on-graph (Sun
et al., 2023) and Graph-CoT (Jin et al., 2024b).
Experimental Setup. We conduct experiments
with three LLM backbones: LLaMA2-13B (Tou-
vron et al., 2023), LLaMA3-8B (AI@Meta, 2024)and Qwen2.5-7B (Yang et al., 2024). For all LLMs,
we load the checkpoints from huggingface1and
use the models directly without fine-tuning. We
implemented CogGRAG and conducted the exper-
iments with one A800 GPU2. Consistent with the
Think-on-graph settings, we set the temperature
parameter to 0.4 during exploration and 0 during
reasoning. The threshold ϵin the retrieval step is
set to 0.7.
Due to the space limitation, we move details on
datasets (Sec A), and baselines (Sec B) to appen-
dices.
5.2 Main Results
We perform experiments to verify the effectiveness
of our framework CogGRAG, and report the re-
sults in Table 2. We use Rouge-L (RL) and Exact
match (EM) as metric for all three datasets. The
backbone model for all the methods is LLaMA2-
13B. From the table, the following observations can
be derived: (1) CogGRAG demonstrates the best
results in most cases. (2) Compared to methods
that incorporate external knowledge, the LLM-only
approach demonstrates significantly inferior per-
formance. This performance gap arises from the
lack of necessary knowledge in LLMs for reason-
ing tasks, highlighting the critical role of external
knowledge integration in enhancing the reasoning
capabilities of LLMs. (3) Graph-based RAG meth-
ods demonstrate superior performance compared
to LLM+KG approaches. This performance advan-
1https://huggingface.co
2Our code is available at: https://anonymous.4open.
science/r/RAG-5883

Table 3: Overall results of our CogGRAG with different backbone models on three KBQA datasets. We highlight
the best score on each dataset in bold.
Type MethodHotpotQA CWQ WebQSP
RL EM RL EM RL EM
LLM-onlyQwen2.5-7B 15.3% 15.0% 25.4% 24.1% 46.7% 45.3%
LLaMA3-8B 17.5% 14.9% 30.3% 27.5% 50.4% 45.1%
LLaMA2-13B 19.1% 17.3% 31.4% 28.8% 51.4% 47.9%
LLM+KGQwen2.5-7B+KG 24.2% 15.6% 33.8% 32.1% 46.7% 45.3%
LLaMA3-8B+KG 25.9% 21.4% 40.6% 35.3% 53.6% 49.1%
LLaMA2-13B+KG 27.5% 23.7% 39.7% 35.1% 52.5% 49.3%
Graph-Based RAGCogGRAG w/ Qwen2.5-7B 28.4% 27.1% 50.5% 45.7% 53.2% 51.6%
CogGRAG w/ LLaMA3-8B 32.1% 27.2% 53.5% 48.4% 57.2% 55.3%
CogGRAG w/ LLaMA2-13B 34.4% 30.7% 56.3% 53.4% 59.8% 56.1%
Table 4: Overall results of our CogGRAG on GRBENCH dataset. We highlight the best score on each dataset in
bold.
MethodE-commerce Literature Academic Healthcare
RL EM RL EM RL EM RL EM
LLaMA2-13B 7.1% 6.8% 5.4% 5.1% 5.4% 4.7% 4.3% 3.1%
Graph-CoT 26.4% 24.0% 26.7% 23.3% 19.3% 14.8% 28.1% 25.2%
CogGRAG 30.2% 28.7% 32.4% 30.1% 23.6% 21.5% 27.4% 25.6%
tage is particularly evident in complex problems,
where not only external knowledge integration but
also involving “thinking procedure” is essential.
Methods ToG and MindMap exhibit enhanced per-
formance by implementing iterative retrieval and
reasoning processes on the KGs, thereby generat-
ing the most probable inference outcomes through
“thinking” on the KGs.
We attribute CogGRAG ’s outstanding effective-
ness in most cases primarily to its ability to decom-
pose complex problems and construct a structured
mind map prior to retrieval. This approach enables
the construction of a comprehensive reasoning path-
way, facilitating more precise and targeted retrieval
of relevant information. Furthermore, CogGRAG
incorporates a self-verification mechanism during
the reasoning phase, further enhancing the accu-
racy and reliability of the final results. Together,
these designs collectively enhance LLMs’ ability
to tackle complex problems.
5.3 Performance with different backbone
models
We evaluate how different backbone models affect
its performance on three datasets HotpotQA, CWQ
and WebQSP, and report the results in Table 3.
We conduct experiments with three LLM back-bones LLaMA2-13B, LLaMA3-8B and Qwen2.5-
7B. For all LLMs, we load the checkpoints from
huggingface and use the models directly without
fine-tuning. From the table, we can observe the
following key findings: (1) CogGRAG achieves
the best results across all backbone models com-
pared to the baseline approaches, demonstrating
the robustness and stability of our method. (2) The
performance of our method improves consistently
as the model scale increases, reflecting enhanced
reasoning capabilities. This trend suggests that
our approach has significant potential for further
exploration with larger-scale models, indicating
promising scalability and adaptability.
5.4 Performance on domain-specific KG
Given the risk of data leakage due to Wikidata
being used as pretraining corpora for LLMs, we
further evaluate the performance of CogGRAG on
a recently released domain-specific dataset GR-
BENCH (Jin et al., 2024b) and report the results
in Table 4. This dataset requires all questions
to interact with a domain-specific KG. We use
Rouge-L (RL) and Exact match (EM) as metrics
for this dataset. The backbone model for all the
methods is LLaMA2-13B (Touvron et al., 2023).
The table reveals the following observations: (1)

10305070HotpotQACWQWebQSPRouge-LCogGRAGCogGRAG-ndCogGRAG-ngCogGRAG-nvFigure 4: Ablation study on the main components of
CogGRAG.
Table 5: Overall results of our CogGRAG on GR-
BENCH dataset. We highlight the best score on each
dataset in bold.
Method Correct Missing Hallucination
LLaMA2-13B 19.1% 25.7% 55.2%
ToG 29.3% 20.2% 50.5%
MindMap 27.9% 22.4% 49.7%
CogGRAG 34.4% 40.6% 44.9%
CogGRAG continues to outperform in most cases.
This demonstrates that our method consistently
achieves stable and reliable results even on domain-
specific knowledge graphs. (2) Both CogGRAG
and Graph-CoT outperform LLaMA2-13B by more
than 20%, which can be ascribed to the fact that
LLMs are typically not trained on domain-specific
data. In contrast, graph-based RAG methods can
effectively supplement LLMs with external knowl-
edge, thereby enhancing their reasoning capabili-
ties. This result underscores the effectiveness of
the RAG approach in bridging knowledge gaps and
improving performance in specialized domains.
5.5 Ablation Study
The ablation study is conducted to understand the
importance of main components of CogGRAG. We
select HotpotQA, CWQ and WebQSP as three rep-
resentative datasets. First, we remove the problem
decomposition module, directly extracting infor-
mation for the target question, and referred to this
variant as CogGRAG-nd ( nodecomposition). Next,
we eliminate the global-level retrieval phase, nam-
ing this variant CogGRAG-ng ( noglobal level).
Finally, we remove the self-verification mechanism
during the reasoning stage, designating this variant
as CogGRAG-nv ( noverification). These experi-
ments aim to systematically assess the impact ofeach component on the overall performance of the
framework. We compare CogGRAG with these
three variants, and the results are presented in Fig-
ure 4. Our findings show that CogGRAG outper-
forms all the variants on the three datasets. Further-
more, the performance gap between CogGRAG
and CogGRAG-nd highlights the importance of
decomposition for complex problem reasoning in
KGQA tasks.
5.6 Hallucination and Missing Evaluation
In the reasoning and self-verification phase of Cog-
GRAG, we prompt the LLM to respond with “I
don’t know” when encountering questions with
insufficient or incomplete relevant information dur-
ing the reasoning process. This approach is de-
signed to mitigate the hallucination issue com-
monly observed in LLMs. To evaluate the effective-
ness of this strategy, we test the model on the Hot-
potQA dataset, with results reported in Table 5. We
categorize the responses into three types: “Correct”
for accurate answers, “Missing” for cases where
the model responds with “I don’t know,” and “Hal-
lucination” for incorrect answers. As shown in the
table results, our model demonstrates the ability to
refrain from answering questions with insufficient
information, significantly reducing the occurrence
of hallucinations. This capability highlights the
effectiveness of our approach in enhancing the reli-
ability and truthfulness of the model’s responses.
6 Conclusion
In this paper, we proposed a graph-based RAG
framework, CogGRAG, to enhance the complex
reasoning capabilities of LLMs in KGQA tasks. In-
spired by human cognitive processes when tackling
complex problems, CogGRAG simulates human
reasoning by decomposing complex questions into
sub-questions and constructing a structured mind
map. Furthermore, recognizing that humans of-
ten engage in self-verification during reasoning,
CogGRAG incorporates a self-verification mecha-
nism during the reasoning phase, leveraging a dual-
process to perform reasoning followed by verifica-
tion, thereby reducing the occurrence of hallucina-
tions. By integrating decomposition, retrieval, and
reasoning with self-verification, CogGRAG signifi-
cantly improves the accuracy of LLMs in handling
complex problems. Extensive experiments demon-
strate that CogGRAG outperforms state-of-the-art
baselines in the majority of cases.

Limitations
Although CogGRAG significantly enhances the
reasoning capabilities of LLMs in complex prob-
lems on KGQA tasks, several limitations remain.
First, while our method avoids iterative retrieval
and reasoning, the processes of problem decom-
position and self-verification inevitably introduce
additional computational overhead. Thus, there is
room for future improvements in optimizing rea-
soning efficiency for practical applications. Second,
CogGRAG is currently limited to graph-based ex-
ternal knowledge and cannot be directly applied
to document-based knowledge sources, which re-
stricts its generalizability in real-world scenarios.
In the future, we plan to explore methods that inte-
grate multiple types of data structures as external
knowledge to further enhance the reasoning capa-
bilities of LLMs. Additionally, research on improv-
ing retrieval and reasoning efficiency remains a key
focus for our ongoing work.
Ethical Considerations
Our research is fundamental in nature and is not di-
rectly linked to specific applications. Consequently,
the potential for misuse or negative societal impacts
depends on how others may apply our methodol-
ogy. Furthermore, our work does not involve any
stakeholders who may benefit or be disadvantaged,
nor does it involve vulnerable groups. All datasets
used in this study are publicly available and widely
utilized in the research community, ensuring no
privacy risks and alignment with their intended pur-
pose for scientific inquiry.
References
AI@Meta. 2024. Llama 3 model card.
Lochan Basyal and Mihir Sanghvi. 2023. Text
summarization using large language models: a
comparative study of mpt-7b-instruct, falcon-7b-
instruct, and openai chat-gpt models. arXiv preprint
arXiv:2310.10449 .
Maciej Besta, Nils Blach, Ales Kubicek, Robert Gersten-
berger, Michal Podstawski, Lukas Gianinazzi, Joanna
Gajda, Tomasz Lehmann, Hubert Niewiadomski, Pi-
otr Nyczyk, et al. 2024. Graph of thoughts: Solving
elaborate problems with large language models. In
Proceedings of the AAAI Conference on Artificial
Intelligence , volume 38, pages 17682–17690.
Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim
Sturge, and Jamie Taylor. 2008. Freebase: a collabo-
ratively created graph database for structuring humanknowledge. In Proceedings of the 2008 ACM SIG-
MOD international conference on Management of
data, pages 1247–1250.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130 .
Shane Frederick. 2005. Cognitive reflection and de-
cision making. Journal of Economic perspectives ,
19(4):25–42.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen
Wang. 2023. Retrieval-augmented generation for
large language models: A survey. arXiv preprint
arXiv:2312.10997 .
Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla,
Thomas Laurent, Yann LeCun, Xavier Bresson, and
Bryan Hooi. 2024. G-retriever: Retrieval-augmented
generation for textual graph understanding and ques-
tion answering. arXiv preprint arXiv:2402.07630 .
Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan,
Chen Ling, and Liang Zhao. 2024. Grag: Graph
retrieval-augmented generation. arXiv preprint
arXiv:2405.16506 .
Bowen Jin, Gang Liu, Chi Han, Meng Jiang, Heng Ji,
and Jiawei Han. 2024a. Large language models on
graphs: A comprehensive survey. IEEE Transactions
on Knowledge and Data Engineering .
Bowen Jin, Chulin Xie, Jiawei Zhang, Kashob Ku-
mar Roy, Yu Zhang, Zheng Li, Ruirui Li, Xian-
feng Tang, Suhang Wang, Yu Meng, et al. 2024b.
Graph chain-of-thought: Augmenting large language
models by reasoning on graphs. arXiv preprint
arXiv:2404.07103 .
Xingxuan Li, Ruochen Zhao, Yew Ken Chia, Bosheng
Ding, Shafiq Joty, Soujanya Poria, and Lidong
Bing. 2023. Chain-of-knowledge: Grounding large
language models via dynamic knowledge adapt-
ing over heterogeneous sources. arXiv preprint
arXiv:2305.13269 .
Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li,
Huaren Qu, Cehao Yang, Jiaxin Mao, and Jian
Guo. 2024. Think-on-graph 2.0: Deep and faith-
ful large language model reasoning with knowledge-
guided retrieval augmented generation. Preprint ,
arXiv:2407.10805.
Costas Mavromatis and George Karypis. 2024. Gnn-
rag: Graph neural retrieval for large language model
reasoning. arXiv preprint arXiv:2405.20139 .
Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun
Shum, Randy Zhong, Juntong Song, and Tong Zhang.
2023. Ragtruth: A hallucination corpus for develop-
ing trustworthy retrieval-augmented language models.
arXiv preprint arXiv:2401.00396 .

Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Ji-
apu Wang, and Xindong Wu. 2023. Unifying large
language models and knowledge graphs: A roadmap,
2023. arXiv preprint arXiv:2306.08302 .
Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo
Wang, Chen Lin, Yeyun Gong, Heung-Yeung Shum,
and Jian Guo. 2023. Think-on-graph: Deep and
responsible reasoning of large language model with
knowledge graph. arXiv preprint arXiv:2307.07697 .
Alon Talmor and Jonathan Berant. 2018. The web as
a knowledge-base for answering complex questions.
arXiv preprint arXiv:1803.06643 .
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, et al. 2023. Llama 2: Open founda-
tion and fine-tuned chat models. arXiv preprint
arXiv:2307.09288 .
Stephen Vaisey. 2009. Motivation and justification: A
dual-process model of culture in action. American
journal of sociology , 114(6):1675–1715.
Denny Vrande ˇci´c and Markus Krötzsch. 2014. Wiki-
data: a free collaborative knowledgebase. Communi-
cations of the ACM , 57(10):78–85.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
et al. 2022. Chain-of-thought prompting elicits rea-
soning in large language models. Advances in neural
information processing systems , 35:24824–24837.
Yilin Wen, Zifeng Wang, and Jimeng Sun. 2023.
Mindmap: Knowledge graph prompting sparks graph
of thoughts in large language models. arXiv preprint
arXiv:2308.09729 .
Yike Wu, Nan Hu, Sheng Bi, Guilin Qi, Jie Ren, An-
huan Xie, and Wei Song. 2023. Retrieve-rewrite-
answer: A kg-to-text enhanced llms framework for
knowledge graph question answering. arXiv preprint
arXiv:2309.11206 .
An Yang, Baosong Yang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan
Li, Dayiheng Liu, Fei Huang, Guanting Dong, Hao-
ran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian
Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin
Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang
Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang,
Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng
Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin,
Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu,
Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng,
Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin
Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang
Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu
Cui, Zhenru Zhang, and Zhihao Fan. 2024. Qwen2
technical report. arXiv preprint arXiv:2407.10671 .
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W Cohen, Ruslan Salakhutdinov, andChristopher D Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing. arXiv preprint arXiv:1809.09600 .
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran,
Tom Griffiths, Yuan Cao, and Karthik Narasimhan.
2024. Tree of thoughts: Deliberate problem solving
with large language models. Advances in Neural
Information Processing Systems , 36.
Wen-tau Yih, Matthew Richardson, Christopher Meek,
Ming-Wei Chang, and Jina Suh. 2016. The value of
semantic parse labeling for knowledge base question
answering. In Proceedings of the 54th Annual Meet-
ing of the Association for Computational Linguistics
(Volume 2: Short Papers) , pages 201–206.
Wenhao Zhu, Hongyi Liu, Qingxiu Dong, Jingjing Xu,
Shujian Huang, Lingpeng Kong, Jiajun Chen, and
Lei Li. 2023. Multilingual machine translation with
large language models: Empirical results and analy-
sis.arXiv preprint arXiv:2304.04675 .

A Datastes
Here, we introduce the four datasets used in our
experiments in detail. For HotpotQA and CWQ
datasets, we evaluated the performance of all meth-
ods on the dev set. For WebQSP dataset, we eval-
uated the performance of all methods on the train
set. The statistics and details of these datasets can
be found in Table 1, and we describe each dataset
in detail below:
HotpotQA is a large-scale question answering
dataset aimed at facilitating the development of QA
systems capable of performing explainable, multi-
hop reasoning over diverse natural language. It
contains 113k question-answer pairs that were col-
lected through crowdsourcing based on Wikipedia
articles.
WebQSP is a dataset designed specifically for ques-
tion answering and information retrieval tasks, aim-
ing to promote research on multi-hop reasoning
and web-based question answering techniques. It
contains 4,737 natural language questions that are
answerable using a subset Freebase KG (Bollacker
et al., 2008).
CWQ is a dataset specially designed to evaluate
the performance of models in complex question an-
swering tasks. It is generated from WebQSP by ex-
tending the question entities or adding constraints
to answers, in order to construct more complex
multi-hop questions.
GRBENCH is a dataset to evaluate how effectively
LLMs can interact with domain-specific graphs
containing rich knowledge to solve the desired
problem. GRBENCH contains 10 graphs from
5 general domains (academia, e-commerce, litera-
ture, healthcare, and legal). Each data sample in
GRBENCH is a question-answer pair.
B Baselines
In this subsection, we introduce the baselines
used in our experiments, including LLaMA2-13B,
LLaMA3-8B, Qwen2.5-7B, Chain-of-Thought
(CoT) prompting, Think-on-graph (ToG),
MindMap and Graph-CoT.
LLaMA2-13B (Touvron et al., 2023) is a member
of the LLM series developed by Meta and is the
second generation version of the LLaMA (Large
Language Model Meta AI) model. LLaMA2-13B
contains 13 billion parameters and is a medium-
sized large language model.
LLaMA3-8B (AI@Meta, 2024) is an efficient and
lightweight LLM with 8 billion parameters, de-signed to provide high-performance natural lan-
guage processing capabilities while reducing com-
puting resource requirements.
Qwen2.5-7B (Yang et al., 2024) is an efficient and
lightweight LLM launched by Alibaba with 7 bil-
lion parameters. It aims at making it more efficient,
specialized, and optimized for particular applica-
tions.
Chain-of-Thought (Wei et al., 2022) is a method
designed to enhance a model’s reasoning ability,
particularly for complex tasks that require multiple
reasoning steps. It works by prompting the model
to generate intermediate reasoning steps rather than
just providing the final answer, thereby improving
the model’s ability to reason through complex prob-
lems.
Think-on-graph (Sun et al., 2023) proposed a new
LLM-KG integration paradigm, “LLM ⊗KG”,
which treats the LLM as an agent to interactively
explore relevant entities and relationships on the
KG and perform reasoning based on the retrieved
knowledge. It further implements this paradigm by
introducing a novel approach called “Graph-based
Thinking”, in which the LLM agent iteratively per-
forms beam search on the KG, discovers the most
promising reasoning paths, and returns the most
likely reasoning result.
MindMap (Wen et al., 2023) propose a novel
prompting pipeline, that leverages KGs to enhance
LLMs’ inference and transparency. It enables
LLMs to comprehend KG inputs and infer with
a combination of implicit and external knowledge.
Moreover, it elicits the mind map of LLMs, which
reveals their reasoning pathways based on the on-
tology of knowledge.
Graph-CoT (Jin et al., 2024b) propose a simple and
effective framework called Graph Chain-of-thought
to augment LLMs with graphs by encouraging
LLMs to reason on the graph iteratively. Moreover,
it manually construct a Graph Reasoning Bench-
mark dataset called GRBENCH, containing 1,740
questions that can be answered with the knowledge
from 10 domain graphs.
C Case Studies
In this section, we present a case analysis of the
HotpotQA dataset to evaluate the performance of
CogGRAG. As illustrated in Figure 5, CogGRAG
decomposes the input question into two logically
related sub-questions. Specifically, the complex
question is broken down into “In which league cup”

Decomposition
{“Sub-question”:“WhatleaguecupistheWiganAthleticF.C.competinginduringthe2017-18season?”,“State”:“End.”}{“Sub-question”:“Whatisthenameoftheleaguecupfromthe2017-18WiganAthleticF.C.season?”,“State”:“End.”}Question:“The2017–18WiganAthleticF.C.seasonwillbeayearinwhichtheteamcompetesintheleaguecupknownaswhatforsponsorshipreasons?”
{"Question":"InwhichleaguecupdidWiganAthleticF.C.competeduringthe2017–18season?","Answer":"WiganAthleticF.C.competedintheEFLCupduringthe2017–18season."}{"Question":"Whatwasthesponsorednameoftheleaguecupidentifiedinsub-question#1duringthe2017–18season?","Answer":"Thesponsorednameoftheleaguecupduringthe2017–18seasonwastheCarabaoCup."}[right][right][{ "Question": "The 2017–18 Wigan Athletic F.C. season will be a year in which the team competes in the league cup known as what for sponsorship reasons?","Answer": "Carabao Cup."}]Reasoning and Self-VerificationEntity:[“WiganAthleticF.C.”,“2017–18season”,“leaguecup”,"sponsorshipreasons"]Triples:[("WiganAthleticF.C.","competesin","leaguecup"),("leaguecup","sponsorshipname","unknown"),("2017–18season","associatedwith","WiganAthleticF.C."),("leaguecup","associatedwith","2017–18season")]Subgraph:[(“WiganAthleticF.C.”,“competesin”,“leaguecup”),(“WiganAthleticF.C.”,“competesin”,“2017-18season”),(“WiganAthleticF.C.”,“has”,“leaguecupname”)]Retrieval
Retrieval and PruneTriples: [("Wigan Athletic F.C.", "is a", "football club"),("Wigan Athletic F.C.", "based in", "Wigan, England"),（"Wigan Athletic F.C.", "founded in", "1932"),("Wigan Athletic F.C.", "competes in", "EFL Championship"), ("2017–18 season", "start date", "August 2017"), ("2017–18 season", "end date", "May 2018"), ("league cup", "official name", "EFL Cup"), ("league cup", "sponsored by", "Carabao"), ("league cup", "involves", "Wigan Athletic F.C."), ("league cup", "associated with", "EFL Championship"), ("league cup", "sponsorship name", "Carabao Cup")……]ExtractionQuestion: “The 2017–18 Wigan Athletic F.C. season will be a year in which the team competes in the league cup known as what for sponsorship reasons?”
Answer: “Carabao Cup.”Figure 5: Case of CogGRAG.
and “What was the sponsored name”, allowing the
system to first identify the league cup and then de-
termine its sponsored name based on the identified
league. These sub-questions form a mind map, cap-
turing the relationships between different levels of
the problem. Next, CogGRAG extracts key infor-
mation from all sub-questions, including both local
level information within individual sub-questions
and global level information across different sub-
questions. A subgraph is constructed to represent
the interconnected triples within the subgraph, en-
abling a global perspective to model the relation-
ships between different questions. The retrieved
triples are pruned based on similarity metrics. All
information retrieved from the KG is represented
in the form of triples. Using this knowledge, Cog-
GRAG prompts the LLM to perform bottom-up
reasoning and self-verification based on the con-
structed mind map. Through this process, the sys-
tem ultimately derives the answer to the target ques-
tion: “Carabao Cup.” This case demonstrates the
effectiveness of CogGRAG in handling complex,
multi-step reasoning tasks by leveraging hierarchi-
cal decomposition, structured knowledge retrieval,and self-verification mechanisms.
Additionally, Figure 6, Figure 8 and Figure 7
illustrate the process of prompting the large lan-
guage model (LLM) to perform reasoning and self-
verification, which provides a detailed breakdown
of how the model generates and validates interme-
diate reasoning steps, ensuring the accuracy and
reliability of the final output.
D Prompts in CogGRAG
In this section, we show all the prompts that need
to be used in the main experiments as shown in
Table 6, Table 7, Table 8, Table 9, Table 10 and
Table 11.

Input：Yourtaskistoanswerthequestionswiththeprovidedcompletedreasoningandinputknowledge.Pleasenotethattheresponsemustbeincludedinsquarebrackets[xxx].Thecompletedreasoning:[]Theknowledge:Triples:[("WiganAthleticF.C.","isa","footballclub"),("WiganAthleticF.C.","basedin","Wigan,England"),（"WiganAthleticF.C.","foundedin","1932"),("WiganAthleticF.C.","competesin","EFLChampionship"),("2017–18season","startdate","August2017"),("2017–18season","enddate","May2018"),("leaguecup","officialname","EFLCup"),("leaguecup","sponsoredby","Carabao"),("leaguecup","involves","WiganAthleticF.C."),("leaguecup","associatedwith","EFLChampionship"),("leaguecup","sponsorshipname","CarabaoCup")]TheInput:["InwhichleaguecupdidWiganAthleticF.C.competeduringthe2017–18season?","Whatwasthesponsorednameoftheleaguecupidentifiedinsub-question#1duringthe2017–18season?"]Output：Response：[{ "Question": "In which league cup did Wigan Athletic F.C. compete during the 2017–18 season?","Answer": "Wigan Athletic F.C. competed in the EFL Cup during the 2017–18 season."},{ "Question": "What was the sponsored name of the league cup identified in sub-question #1 during the 2017–18 season?","Answer": "The sponsored name of the league cup during the 2017–18 season was the Carabao Cup."}]Figure 6: The prompt case of reasoning.

Input：Your task is to answer the questions with the provided completed reasoning and input knowledge. Please note that the response must be included in square brackets [xxx]. The completed reasoning: [ { "Question": "In which league cup did Wigan Athletic F.C. compete during the 2017–18 season?", "Answer": "Wigan Athletic F.C. competed in the EFL Cup during the 2017–18 season." }, { "Question": "What was the sponsored name of the league cup identified in sub-question #1 during the 2017–18 season?", "Answer": "The sponsored name of the league cup during the 2017–18 season was the Carabao Cup." }] The knowledge: Triples: [("Wigan Athletic F.C.", "is a", "football club"), ("Wigan Athletic F.C.", "based in", "Wigan, England"), （"Wigan Athletic F.C.", "founded in", "1932"), ("Wigan Athletic F.C.", "competes in", "EFL Championship"), ("2017–18 season", "start date", "August 2017"), ("2017–18 season", "end date", "May 2018"), ("league cup", "official name", "EFL Cup"), ("league cup", "sponsored by", "Carabao"), ("league cup", "involves", "Wigan Athletic F.C."), ("league cup", "associated with", "EFL Championship"), ("league cup", "sponsorship name", "Carabao Cup")] The Input: ["The 2017–18 Wigan Athletic F.C. season will be a year in which the team competes in the league cup known as what for sponsorship reasons?"] Output: Response：[ { "Question": "The 2017–18 Wigan Athletic F.C. season will be a year in which the team competes in the league cup known as what for sponsorship reasons?","Answer": "During the 2017–18 season, Wigan Athletic F.C. competed in the league cup known as the Carabao Cup for sponsorship reasons."} ]Figure 7: The prompt case of reasoning with the completed reasoning.

Input：Youarealogicalverificationassistant.Yourtaskistocheckwhethertheanswertoagivenquestionislogicallyconsistentwiththeprovidedcompletedreasoningandinputknowledge.Iftheanswerisconsistent,respondwith“right”.Iftheanswerisinconsistent,respondwith“wrong”.Pleasenotethattheresponsemustbeincludedinsquarebrackets[xxx].Thecompletedreasoning:[]Theknowledge:Triples:[("WiganAthleticF.C.","isa","footballclub"),("WiganAthleticF.C.","basedin","Wigan,England"),（"WiganAthleticF.C.","foundedin","1932"),("WiganAthleticF.C.","competesin","EFLChampionship"),("2017–18season","startdate","August2017"),("2017–18season","enddate","May2018"),("leaguecup","officialname","EFLCup"),("leaguecup","sponsoredby","Carabao"),("leaguecup","involves","WiganAthleticF.C."),("leaguecup","associatedwith","EFLChampionship"),("leaguecup","sponsorshipname","CarabaoCup")]TheInput:["InwhichleaguecupdidWiganAthleticF.C.competeduringthe2017–18season?”]Theanswer:[{"Question":"InwhichleaguecupdidWiganAthleticF.C.competeduringthe2017–18season?","Answer":"WiganAthleticF.C.competedintheEFLCupduringthe2017–18season.”}]Output:Response：[right]The answers are logically consistent with the provided knowledge. The triples confirm that Wigan Athletic F.C. competed in the EFL Cup (league cup) during the 2017–18 season, and the sponsored name of the league cup was the Carabao Cup. Therefore, the answers are correct.Figure 8: The prompt case of self-verification.

Table 6: The prompt template for decomposition.
Decomposition prompt
Prompt head: “Your task is to decompose the given question Q into sub-questions.
You should based on the specific logic of the question to determine the number of sub-questions
and output them sequentially. ”
Instruction: “Please only output the decomposed sub-questions as a string in list format, where each
element represents the text of a sub-question, in the form of ’[\"subq1\", \"subq2\", \"subq3\"]’.
For each sub-questions, if you consider the sub-question to be sufficiently simple and no
further decomposition is needed, then output \"End.\", otherwise, output \"Continue.\".
Please strictly follow the format of the example below when answering the question.
Here are some examples: ”
“ Input: “What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger
as a former New York Police detective?”
Output: [ { “Sub-question”: “What movie starring Arnold Schwarzenegger as a former New York Police
detective is being referred to?”,
“State”: “Continue.” },
{ “Sub-question”: “In what year did Guns N Roses perform a promo for the movie mentioned
in sub-question #1?”,
“State”: “End.” } ]
Input: “What is the name of the fight song of the university whose main campus is in Lawrence, Kansas
and whose branch campuses are in the Kansas City metropolitan area?”
Output: [ { “Sub-question”: “Which university has its main campus in Lawrence, Kansas and branch
campuses in the Kansas City metropolitan area?”,
“State”: “End.”
},
{ “Sub-question”: “What is the name of the fight song of the university identified in sub-question #1?”,
“State”: “End.” } ]
Input: “Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?”
Output: [ { “Sub-question”: “Where is the Laleli Mosque located?”,
“State”: “End.”
},
{ “Sub-question”: “Where is the Esma Sultan Mansion located?”,
“State”: “End.” },
{ “Sub-question”: “Are the locations of the Laleli Mosque and the Esma Sultan Mansion in the same
neighborhood?”,
“State”: “End.” } ] ”
“Input: Question Q”
“Output: ”

Table 7: The prompt template for extraction on local level.
Extraction prompt
Prompt head: “Your task is to extract the entities (such as people, places, organizations, etc.)
and relations (representing behaviors or properties between entities, such as verbs,
attributes, or categories, etc.) involved in the input questions. These entities and relations can
help answer the input questions.”,
Instruction: “Please extract entities and relations in one of the following forms: entity, tuples,
or triples from the given input List. Entity means that only an entity, i.e. <entity>. Tuples means that
an entity and a relation, i.e. <entity-relation>. Triples means that complete triples, i.e. <entity-relation-entity>.
Please strictly follow the format of the example below when answering the question. ”,
“Input: [The mind map M]”
“Output: ”
Table 8: The prompt template for extraction on global level.
Extraction prompt
Prompt head: “Your task is to extract the subgraphs involved in a set of input questions.”,
Instruction: “Please extract and organize information from a set of input questions into structured
subgraphs. Each subgraph represents a group of triples (subject, relation, object) that share common
entities and capture the logical relationships between the questions. Here are some examples:”
“ Input: [“What is the capital of France?”, “Who is the president of France?”,
“What is the population of Paris?”]
Output: [(“France”, “capital”, “Paris”), (“France, “president”, “Current President”),
(“Paris”, “population”, “Population Number”) ] ”
“Input: [The mind map M]”
“Output: ”
Table 9: The prompt template for reasoning.
Reasoning prompt
Prompt head: “Your task is to answer the questions with the provided completed reasoning and input
knowledge.”,
Instruction: “Please note that the response must be included in square brackets [xxx].”
“The completed reasoning: [The reasoning process ˆM]”
“The knowledge graph: [Knowledge T]”
“Input: [Subquestion qt
j] ”
“Output: ”

Table 10: The prompt template for self-verification.
Self-verification prompt
Prompt head: “You are a logical verification assistant. Your task is to check whether the answer to a
given question is logically consistent with the provided completed reasoning and input knowledge.
If the answer is consistent, respond with “right”. If the answer is inconsistent, respond with “wrong”.”
Instruction: “Please note that the response must be included in square brackets [xxx].”
“The completed reasoning: [The reasoning process ˆM]”
“The knowledge graph: [Knowledge T]”
“Input: [Subquestion qt
j] ”
“Answer: [Answer at
j] ”
“Output: ”
Table 11: The prompt template for re-thinking.
Re-thinking prompt
Prompt head: “You are a reasoning and knowledge integration assistant. Your task is to re-think
a question that was previously answered incorrectly by the self-verification model. Use the provided
completed reasoning and input knowledge to generate a new answer.”
Instruction: “Please note, if the knowledge is insufficient to answer the question, respond
with “Insufficient information, I don’t know”.The response must be included in square brackets [xxx].”
“The completed reasoning: [The reasoning process ˆM]”
“The knowledge graph: [Knowledge T]”
“Input: [Subquestion qt
j] ”
“Output: ”
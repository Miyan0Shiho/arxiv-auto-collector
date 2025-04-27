# Credible plan-driven RAG method for Multi-hop Question Answering

**Authors**: Ningning Zhang, Chi Zhang, Zhizhong Tan, Xingxing Yang, Weiping Deng, Wenyong Wang

**Published**: 2025-04-23 15:03:17

**PDF URL**: [http://arxiv.org/pdf/2504.16787v1](http://arxiv.org/pdf/2504.16787v1)

## Abstract
Multi-hop question answering (QA) presents a considerable challenge for
Retrieval-Augmented Generation (RAG), requiring the structured decomposition of
complex queries into logical reasoning paths and the generation of dependable
intermediate results. However, deviations in reasoning paths or errors in
intermediate results, which are common in current RAG methods, may propagate
and accumulate throughout the reasoning process, diminishing the accuracy of
the answer to complex queries. To address this challenge, we propose the
Plan-then-Act-and-Review (PAR RAG) framework, which is organized into three key
stages: planning, act, and review, and aims to offer an interpretable and
incremental reasoning paradigm for accurate and reliable multi-hop question
answering by mitigating error propagation.PAR RAG initially applies a top-down
problem decomposition strategy, formulating a comprehensive plan that
integrates multiple executable steps from a holistic viewpoint. This approach
avoids the pitfalls of local optima common in traditional RAG methods, ensuring
the accuracy of the entire reasoning path. Subsequently, PAR RAG incorporates a
plan execution mechanism based on multi-granularity verification. By utilizing
both coarse-grained similarity information and fine-grained relevant data, the
framework thoroughly checks and adjusts intermediate results, ensuring process
accuracy while effectively managing error propagation and amplification.
Experimental results on multi-hop QA datasets demonstrate that the PAR RAG
framework substantially outperforms existing state-of-the-art methods in key
metrics, including EM and F1 scores.

## Full Text


<!-- PDF content starts -->

CREDIBLE PLAN -DRIVEN RAG METHOD FOR MULTI -HOP
QUESTION ANSWERING
Ningning Zhang1, Chi Zhang1, Zhizhong Tan1, Xingxing Yang1, Weiping Deng1, and Wenyong Wang∗1
1Macau University of Science and Technology
April 24, 2025
ABSTRACT
Multi-hop question answering (QA) presents a considerable challenge for Retrieval-Augmented
Generation (RAG), requiring the structured decomposition of complex queries into logical reasoning
paths and the generation of dependable intermediate results. However, deviations in reasoning paths
or errors in intermediate results, which are common in current RAG methods, may propagate and
accumulate throughout the reasoning process, diminishing the accuracy of the system’s responses
to complex queries. To address this challenge, we propose the Plan-then- Act-and- Review (PAR
RAG) framework, which is organized into three key stages—planning, act, and review—and aims
to offer an interpretable and incremental reasoning paradigm for accurate and reliable multi-hop
question answering by mitigating error propagation. PAR RAG initially applies a top-down problem
decomposition strategy, formulating a comprehensive plan that integrates multiple executable steps
from a holistic viewpoint. This approach avoids the pitfalls of local optima common in traditional RAG
methods, ensuring the accuracy of the entire reasoning path. Subsequently, PAR RAG incorporates a
plan execution mechanism based on multi-granularity verification. By utilizing both coarse-grained
similarity information and fine-grained relevant data, the framework thoroughly checks and adjusts
intermediate results, ensuring process accuracy while effectively managing error propagation and
amplification. Experimental results on multi-hop QA datasets demonstrate that the PAR RAG
framework substantially outperforms existing state-of-the-art methods in key metrics, including EM
and F1 scores.
Keywords RAG·Open-domain question answering ·Multi-hop ·Large large model
1 Introduction
Multi-hop question answering (QA) [ 1] requires the integration of multiple information sources and multi-step reasoning
to derive an answer and has wide-ranging applications in fields such as complex decision-making and deep analysis
[2]. In recent years, Retrieval-Augmented Generation (RAG) [ 3] has made significant progress in multi-hop QA tasks
[4], achieving significant advancements [ 5,6]. However, it still faces several key challenges, particularly deviations in
reasoning paths and errors in intermediate results. First, existing research highlights that prevalent failures in multi-hop
QA tasks are often caused by deviations in reasoning paths, typically stemming from errors in executing the first hop or
from inconsistencies in the reasoning paths [ 7,8,9,10], underscoring the critical importance of problem decomposition.
If the problem decomposition leads to an erroneous reasoning path, subsequent steps become challenging to correct,
ultimately resulting in failures. Second, during the subproblem-solving process, the accuracy of intermediate results is
vulnerable to issues such as interference from irrelevant information, position bias, or knowledge conflict [ 11,12,13,14].
Moreover, deviations in the reasoning path and errors in intermediate steps are propagated and amplified at each stage
of the multi-step reasoning process, eventually triggering a butterfly effect [ 15] that substantially reduces the accuracy
of answers to complex queries.
∗corresponding authorarXiv:2504.16787v1  [cs.CL]  23 Apr 2025

APREPRINT - APRIL 24, 2025
Humans demonstrate distinct cognitive superiority over artificial intelligence (AI) when addressing complex multi-step
reasoning tasks. Professionally trained individuals typically employ a structured cognitive framework, such as the
PDCA cycle [ 16], to develop a comprehensive plan based on a clear understanding of the problem’s intent, using a
top-down approach. They then execute and verify each step in accordance with the plan, always with the goal of
generating a final answer. The human cognitive process for complex problem-solving consists of the following critical
phases:
1.Global Planning Phase : From a holistic perspective, complex problems are decomposed into logically
interconnected yet independently solvable subproblems. This structural decomposition constructs coherent
reasoning pathways, effectively preventing deviations in the reasoning path.
2.Plan Execution and Multi-granularity Verification Phase : During plan execution, when encountering
insufficient local information, coarse-grained data with semantic similarity to the current subproblem are first
retrieved to generate a preliminary answer, thereby enhancing the understanding of the problem. Subsequently,
guided by the interim answer, fine-grained data with rich semantic correlations are further retrieved. A
cross-verification mechanism then facilitates the refinement of provisional solutions, constructing an error-
constrained reasoning trajectory. This granular retrieval-verification mechanism significantly mitigates the risk
of error accumulation in intermediate results.
3.Synthesis and Reasoning Phase : Upon resolving the subproblems, humans synthesize the complex problem
with the entire reasoning trajectory to derive the final answer.
The PDCA cycle, as a systematic improvement method, is grounded in the core principle of continuously enhancing
result quality through an iterative closed-loop process comprising the Plan-Do-Check-Act stages. This method aligns
with the requirements of multi-hop question answering, where the complexity of multi-hop QA necessitates a system
capable of solving problems in distinct, verifiable stages. Specifically, the planning phase corresponds to problem
decomposition, the execution phase involves evidence retrieval and reasoning, the checking phase entails result
verification, and the action phase focuses on refining results based on the verification outcomes.
Inspired by this, we propose an innovative RAG framework, PAR RAG, which integrates the Plan-then- Act-and- Review
methodology from the PDCA (Plan-Do-Check-Act) model with the reading comprehension and reasoning capabilities
of large language models (LLMs) [ 17,18,19,20,21,22]. In the planning phase, PAR RAG systematically decomposes
complex problems, ensuring a comprehensive understanding of the query’s intent. This approach converts the problem
into a structured plan with multiple reasoning steps, effectively mitigating the tendency of iterative reasoning to
converge on local optima, which can lead to deviations in the reasoning path. In the act (plan execution) phase, the
framework retrieves coarse-grained data to generate an initial answer based on the query. Subsequently, the interim
answer is used to retrieve finer-grained, relevant data, while a verification mechanism employs both the original question
and the retrieved information to refine and adjust the answer. This process aligns evidence and results to ensure
coherence [ 23,24,25], thereby reducing errors induced by hallucinations in LLMs [ 26,27], and ensuring the accuracy
and reliability of intermediate results. By implementing a top-down planning and execution process with iterative
verification and refinement, PAR RAG effectively mitigates error propagation and amplification caused by reasoning
path deviations and process errors in existing RAG methods for multi-hop reasoning tasks.
Extensive testing was carried out on the PAR RAG system across multiple multi-hop datasets. Experimental results
demonstrate that, in comparison to existing state-of-the-art methods, PAR RAG achieves superior results in terms of EM
and F1 scores. Moreover, experiments on single-hop datasets further confirm that PAR RAG exhibits strong adaptability,
making it applicable beyond multi-hop question-answering tasks.
Our contributions can be summarized as follows:
•We propose a trustworthy, plan-driven RAG framework, PAR RAG. This framework begins with the formulation
of a top-down plan and, during plan execution, verifies and refines the results of each step using a multi-
granularity verification mechanism. It offers an interpretable, incremental reasoning paradigm for solving
multi-hop question-answering tasks, effectively mitigating error propagation through comprehensive reasoning
path planning and multi-granularity verification.
•Through extensive experimentation across multiple multi-hop datasets, we demonstrate that the PAR RAG
method significantly improves the accuracy of multi-hop question answering, surpassing current state-of-the-art
methods. These results highlight the considerable potential of the integrated planning and verification approach
in tackling complex reasoning challenges.
•We identify that the language understanding and reasoning capabilities of large models represent a critical
bottleneck in advancing RAG for multi-hop question answering. The inability of large models to accurately
interpret the intent behind complex questions leads to deviations in reasoning paths, while their failure
2

APREPRINT - APRIL 24, 2025
to effectively leverage relevant retrieved information results in errors in intermediate steps. Both issues
substantially undermine the accuracy of answers in multi-hop question-answering tasks.
The structure of the paper is as follows: Section 2 provides a discussion of RAG technologies pertinent to multi-hop
question answering, establishing the research background. Section 3 details the overall architecture of our method and
explains the functioning mechanisms of its core components. In Section 4, we present the experimental setup, including
the configurations and parameters used. Section 5 offers a comprehensive analysis of the experimental results and their
implications. Finally, Section 6 concludes the paper.
2 Related Work
Multi-hop question answering, as opposed to single-hop question answering, requires a comprehensive understanding
of complex queries and the integration of multiple relevant information sources to generate an accurate answer. The
RAG methods used in multi-hop question answering can be broadly classified into three categories: data-enhanced
RAG, reasoning-enhanced RAG, and credibility-enhanced RAG.
2.1 Data-enhanced RAG
Compared to single-step question-answering tasks, multi-hop question answering typically requires retrieving relevant
information from multiple documents and synthesizing it through reasoning [ 28]. To enhance the efficiency of complex
information retrieval and reasoning, hierarchical data indexing mechanisms have emerged as a recent research focus.
The core of this approach lies in optimizing the retrieval path for multi-hop reasoning through the structured organization
of data. In related research, RAPTOR [ 29] implements a tree-based retrieval framework that recursively generates
document summaries and constructs a hierarchical index, with each node representing the semantic aggregation of its
child nodes. This structure effectively narrows the search space, thereby enhancing retrieval accuracy for multi-hop
reasoning. Similarly, GraphRAG [ 30] enhances RAG by leveraging knowledge graphs for improved information retrieval
and answer generation. It constructs knowledge graphs from unstructured text, extracting entities and relationships, and
employs hierarchical clustering (e.g., Leiden algorithm) to create community summaries. Supporting both global and
local search, it provides structured, context-rich inputs to LLMs, improving accuracy in complex queries. Additionally,
HippoRAG [ 31], inspired by human memory mechanisms, integrates LLMs, knowledge graphs, and the Personalized
PageRank algorithm [ 32] to construct a biomimetic memory index architecture, optimizing the efficiency of long-range
information retrieval. Moreover, StructRAG [ 33] introduces a dynamic data structure selection strategy, adaptively
transforming document representations (e.g., tree, graph) during the reasoning phase to accommodate various complex
tasks, such as fact verification and question answering, thereby demonstrating superior information integration. Notably,
SiReRAG [ 34] underscores the limitations of traditional multi-hop reasoning, which relies exclusively on similarity-
based retrieval. To address these limitations, it proposes a dual-tree index architecture that constructs both similarity
and relevance trees, improving semantic association modeling through recursive summarization, thereby achieving
superior performance in complex reasoning tasks.
Existing methods primarily enhance performance through two strategies: increasing information density (e.g., RAPTOR,
SiReRAG) or improving the precision of information retrieval (e.g., GraphRAG, HippoRAG). While these approaches
demonstrate exceptional performance in specific tasks, they still exhibit significant limitations when applied to complex
multi-hop question-answering tasks. Specifically, existing methods lack a comprehensive decomposition strategy for the
complex problem, which makes it difficult to construct precise retrieval conditions. When retrieval conditions deviate
from the essence of the query, this results in reduced data relevance, thereby compromising the reliability of the final
answer.
In contrast to current approaches, PAR RAG intentionally plans the challenging multi-hop question-answering assign-
ment by leveraging LLMs’ language comprehension and reasoning skills. An executable series of steps that break down
the problem in a methodical manner are developed as part of this planning process. Each stage in the plan is carefully
matched with the problem to be solved, which makes it possible to precisely retrieve pertinent data and successfully get
around the drawbacks of the previously discussed approaches.
2.2 Reasoning-enhanced RAG
In multi-hop question-answering tasks, directly utilizing the original query to retrieve relevant information often fails
to yield satisfactory results. Consequently, existing research commonly integrates the language understanding and
reasoning capabilities of large models, adopting a strategy of gradually decomposing complex problems and iteratively
retrieving information. This approach systematically narrows the search space over multiple rounds of retrieval, thereby
enabling the precise identification of the evidence chain essential for multi-hop reasoning. For example, IRCoT [ 35],
3

APREPRINT - APRIL 24, 2025
based on the Chain of Thought (CoT) framework [ 36], establishes an iterative retrieval framework that progressively
generates intermediate reasoning steps throughout multiple retrieval rounds to enhance retrieval accuracy. Iter-RetGen
[37] utilizes an iterative retrieval-generation collaborative mechanism, improving the large language model’s ability to
capture information from intricate queries. Adaptive RAG [ 38], extending IRCoT, introduces a complexity classifier that
dynamically adapts the retrieval strategy based on problem difficulty, thus enhancing question-answering performance.
Similarly, planning-based approaches enhance multi-hop question-answering tasks by generating intermediate reasoning
steps or plans. For example, RAP [ 39] establishes a dual-role framework in which a large language model (LLM)
functions as a world model and a reasoning agent. During the reasoning process, the LLM, acting as the agent,
incrementally constructs a reasoning tree under the guidance of the LLM, which serves as the world model. It employs
Monte Carlo tree search algorithms to identify the optimal reasoning path. Furthermore, [ 40] extracts multi-step plan
data from knowledge graphs and converts it into question-answer pairs in natural language. These pairs are subsequently
used to fine-tune the LLM, significantly improving its performance in multi-source information integration tasks.
However, while iterative retrieval methods can reduce problem complexity, they often lack global planning, making
them vulnerable to local optima. This misalignment in the reasoning path during the iterative process ultimately
compromises the accuracy of the final answer. In contrast, both planning-based methods and PAR RAG adopt a similar
approach, fully leveraging the language understanding and reasoning capabilities of large models. They decompose
complex problems into executable plans that encompass multiple reasoning steps, thereby alleviating the difficulty of
handling complex tasks and mitigating the risk of reasoning path deviation.
The key distinction between PAR RAG and existing planning methods lies in the incorporation of a multi-granularity
verification mechanism. PAR RAG combines coarse-grained semantic similarity data with fine-grained semantic
association data to dynamically verify and refine the results of each reasoning step. This approach reduces errors during
the execution of the entire plan, effectively minimizing the backward propagation and amplification of errors caused by
incorrect intermediate results.
2.3 Credibility-enhanced RAG
Recent research has proposed various methods to enhance answer credibility in multi-hop question-answering tasks,
thereby increasing the accuracy of the final answer by improving the credibility of intermediate results. These
approaches primarily include citation-based validation and self-reflection mechanisms. Citation-based methods increase
the credibility and transparency of answers by explicitly referencing evidence extracted from multiple knowledge
sources to support the response [ 41,42]. This strategy not only improves the interpretability of the answers, allowing
users to trace the reasoning process transparently, but also mitigates bias and uncertainty that may arise from relying on
a single information source by incorporating evidence from diverse references [43, 44, 45, 46].
On the other hand, the self-reflection mechanism guides the large model to review their reasoning process and output,
then identify and correct mistakes, theoretically enhancing the reliability of RAG systems [ 47]. However, empirical
studies have shown that LLMs relying solely on prompt instructions often struggle to achieve the desired outcomes
through self-reflection unless combined with external feedback or fine-tuned specifically for this purpose [ 48,49]. As
a result, current research tends to treat self-reflection as an auxiliary mechanism rather than an independent solution
[50, 51, 52, 53].
Citation-based methods are pivotal in enhancing both the accuracy and credibility of answers. Within the PAR RAG
framework, we utilize citation-based strategies to optimize the reasoning process. Specifically, during the plan execution
phase, the system applies a citation mechanism to filter retrieval results, selecting only the information most relevant
to the current reasoning step. This minimizes the influence of irrelevant data, thereby improving the accuracy and
credibility of intermediate results, reducing the risk of error propagation, and significantly increasing the reliability of
the final answer.
3 Method
3.1 Problem Formulation
Given a multi-hop question answering task Q, which can be decomposed into a set of sub-questions: Q=
(q1, q2, . . . , q m), and a document collection D= (D1, D2, . . . , D k), the goal of a plan-driven RAG method is to
find a plan Pthat facilitates sequential retrieval of information from the document collection Dand generates and
combines intermediate answers A1, A2, . . . , A n, such that
Afinal =fcombine (A1, A2, . . . , A n)
4

APREPRINT - APRIL 24, 2025
, where fcombine is a LLM or a specific method that integrates multi-hop information to produce the final answer Afinal .
Each intermediate step in the plan can be formally expressed as
Pi=fanswer (fretrieve (Qi))
, where fretrieve is the information retrieval method or LLM responsible for retrieving relevant information, and
fanswer is the LLM tasked with generating the answer for the intermediate step.
3.2 Framework of PAR RAG
Question
Plan Module
Plan
Action Module
Review Module
Question
TrajectoriesTrajectories
Trajectory 1
Trajectory N……Plan Phase Action Phase Review Phase
Execute the steps in the plan in order
Verify and revise the answer to the current stepUse the answer to the current step to 
refine the question of the next step
Generate the final answer
Figure 1: Overview of PAR RAG. This framework incorporates three key phases: planning, act (plan execution), and
review. It begins by converting the complex query into a plan consisting of logically interconnected yet independently
solvable steps. During the execution of each step, multi-granularity data are used to verify and revise the interim answer,
then the query of the next step is refined. After each step, an execution trajectory is generated and added to the trajectory
chain. Once all steps in the plan are completed, the final answer is generated by the language model (LM), incorporating
both the multi-hop question and the trustworthy trajectory chain collected during the plan’s execution.
This section provides a detailed description of the PAR RAG method proposed in this paper, as illustrated in Figure
1. The PAR RAG method consists of three core components: the Plan Module, the Action Module, and the Review
Module. First, the Plan Module (Section 3.4) breaks down the multi-hop question into a structured plan containing
multiple reasoning steps. Then, the Action Module (Section 3.5) sequentially executes the steps in the plan. During the
execution of each step, the Review Module (Section 3.6) verifies and revises the preliminary answer for the current step,
thereby enhancing the credibility of the results.
As described in Algorithm 1, the specific workflow of PAR RAG consists of the following procedures. First, the
Plan Module comprehensively understands the intent of question Qand formulates a plan Pwith multiple steps in a
top-down manner, thereby avoiding reasoning path deviation. Then, when executing the plan P, the Action Module
sequentially processes each reasoning step in the plan. For each step, it first retrieves coarse-grained semantic similarity
data using the question of the current step as the retrieval condition and generates the preliminary answer for that
step. During answer generation, the LM uses citation-based evidence selection to ensure the relevance of the evidence
employed. The interim answer is then used to retrieve fine-grained semantic association data. The Review Module
uses this fine-grained data to verify and revise the preliminary answer, identifying and correcting potential errors to
prevent them from propagating and amplifying in subsequent reasoning steps, thereby maximizing the credibility of the
answer. Next, the Action Module updates the question for the next step based on the current answer, replacing any
ambiguous information and generating a new question for the next reasoning step. Furthermore, after each step in the
5

APREPRINT - APRIL 24, 2025
Algorithm 1 Credible Plan-driven RAG
Input: Question Q, Plan Module PM , Large Language Model LM, Review Module RM, Coarse-Grained Information
Retrieval Method: fcoarse −grained −retrieve , Fine-Grained Information Retrieval Method: ffine−grained −retrieve ,
Trajectories Construction Method fconcat
Output: Afinal
i←1
P=P1, P2, ..., P n←PM (Q) ▷Generate zero-shot initial step-by-step plan P
repeat
Qi←Pi ▷Get the question of the current step
Ri←fcoarse −grained −retrieve (Qi) ▷Retrieve semantic similar information Rifrom corpus
Ai←LM (Qi, Ri) ▷Generate preliminary answer Aifor the current step
Ri*←ffine−grained −retrieve (Ai) ▷Retrieve fine-grained relevant information Ri*from corpus
Ai*←RM (Qi, Ai, Ri*) ▷Verify and revise the answer according to the original question and retrieved
fine-grained information
Qnext←Pi+1 ▷Get question of the next step
Qnext*←LM (Ai*, Qnext) ▷Refine the question of the next step according to current answer
T←fconcat (Qi, Ai*) ▷Append the current step into the trajectories
i←i+ 1 ▷Begin the next round
until i > n ▷Repeat until all steps in the plan were completed
Afinal←LM (Q, T ) ▷Generate the last answer
plan is completed, a trajectory record is generated and added to the trajectory chain. Each trajectory record consists of
the current question, its corresponding answer, and the relevant data cited by the answer. Finally, after completing all
the steps in the plan, we provide the question Qand the trajectory chain Tas input to the LM, prompting it to generate
the final answer.
3.3 Multi-granularity data indexing and retrieval mechanism
In multi-hop question-answering tasks, each step involves retrieving information and generating an answer, which helps
identify key missing details as we work toward the final answer. At each step, since the question remains somewhat
undefined until answered, information retrieval based on the current question can only provide coarse-grained semantic
similarity data. However, once a preliminary answer is generated, it holds richer and more specific semantic information.
Therefore, conducting a subsequent retrieval using the preliminary answer allows us to obtain fine-grained, more
semantically relevant data, which helps to identify and correct any errors in the initial response. Consequently, we
believe that using multi-granularity data retrieval and verification is essential for improving the quality of intermediate
answers in multi-hop question-answering tasks.
For establishing a multi-granularity data index, our data indexing process consists of two main parts. On the one hand,
we employ a retrieval encoder Eto encode each document in document collection Dand build a coarse-grained vector
index V. On the other hand, following the data indexing mechanism of HippoRAG [ 31], we use a knowledge graph
to extract and store fine-grained data (entities, relationships between entities) from the documents in the document
collection D. Specifically, we employ a prompt-driven LLM to extract entities Nas nodes and the relationships between
entities as edges Efrom each document in D, constructing a fine-grained knowledge graph KG according to the
OpenIE [54] standards.
When retrieving data for the question Q, for the coarse-grained retrieval, we use the same Retriever Encoder to encode
Qand then apply a vector similarity algorithm to search for the top-k most similar documents in the vector index V. On
the other side, for the fine-grained retrieval, based on the coarse-grained retrieval, we leverage a prompt-driven LLM
to extract a set of entities Nqueryfrom Q. Then, we search for the top-k most similar entities Mfrom the previously
defined entity set Nbased on Nquery. Subsequently, we use the Personalized PageRank (PPR) algorithm [ 32] to retrieve
relevant relational data from the knowledge graph KG based on M. Finally, the data from both parts are merged and
sorted, and the combined result is returned as the final retrieval output.
3.4 Plan Module
The Plan Module harnesses the language understanding and reasoning capabilities of LLMs to convert complex
problems into executable plans consisting of multiple reasoning steps. We posit that this global, top-down approach
to problem decomposition is more effective than the iterative method, as it enables a more precise understanding
6

APREPRINT - APRIL 24, 2025
of the problem’s intent and mitigates risks, such as local optima, that may lead to deviations in the reasoning path.
Additionally, it adaptively adjusts the steps of the plan based on the question’s complexity, rather than relying on the
fixed iteration counts employed by current CoT-based RAG methods.
Specifically, the Plan Module utilizes a prompt to guide the LLM in thoroughly analyzing the complex multi-hop
question and breaking it down into an actionable plan with multiple steps, denoted as P= (step 1, ..., step n). For each
step in the plan, we define step i= (thought i, question i), where thought irepresents the LLM’s reasoning process
for the current step, and question irefers to the sub-question derived from the decomposition of the complex problem.
The specific prompt used by the Plan Module is detailed in Appendix A.4.1.
3.5 Action Module
The Action Module sequentially executes the steps outlined in the plan. First, it uses the current step’s question qito
retrieve coarse-grained similar data through the retrieval encoder E, where di=E(qi). The LLM then generates the
preliminary answer for the current step ai=LM (qi, di)by referencing relevant data from the query results via citation.
Subsequently, the Action Module identifies the cited data sources, denoted as si=fcite(ai, di),fciteis the function
used to select the cited information. Next, the LLM utilizes the interim answer aito revise the question of the next step
qi+1new=LM (qi+1, ai).
Upon completing the processing of the current step, a trustworthy trajectory is formed, ti= (qi, ai, si), and added
to the trajectory chain T= (t1, ..., t i). Once all steps in the plan are executed, the initial question and the trajectory
chain are submitted to the LLM with the corresponding prompt to generate the final answer Afinal =LM (Q, T ). The
prompt used by the Action Module is detailed in Appendix A.4.2.
3.6 Review Module
The design intention of the Review Module is as follows: once a preliminary answer for the current step is obtained,
the information becomes more defined. We argue that performing a subsequent retrieval based on the interim answer
will yield more semantically relevant, fine-grained data. By validating and refining the preliminary answer with these
fine-grained data, the quality of the process results can be significantly improved, reducing the risk of error propagation
and amplification in subsequent steps.
Specifically, the Review Module first uses the current step’s preliminary answer aias a query to retrieve fine-grained
related data, dinew=E(ai) +KG(ai), from the vector database and knowledge graph. A prompt is then used to
instruct the LLM to check whether the provisional answer aligns with the intent of the question. The LLM validates the
preliminary answer by incorporating the question of the current step and fine-grained related data: if the interim answer
satisfies the requirements of the question, the LLM returns it directly; otherwise, the LLM generates a revised version
of the preliminary answer, ainew=RM (ai, dinew), based on the question and related data. Appendix A.4.3 provided
the prompt used by the Review Module.
4 Experiments
4.1 Datasets
To comprehensively evaluate the performance of PAR RAG on multi-hop question answering, we selected three
multi-hop QA datasets commonly used by other RAG methods:
1.2WikiMultiHopQA [55]. It is built upon Wikipedia content and knowledge graph paths, specifically designed
to evaluate models’ ability to perform two-hop reasoning tasks.
2.HotpotQA [2]. It challenges models with questions that bridge multiple Wikipedia articles, necessitating the
integration of information from different sources.
3.MuSiQue [56]. It introduces complexity by combining multiple single-hop queries into multi-hop questions,
requiring models to navigate through 2-4 logical steps.
We selected 500 instances from each dataset as test entities.
Additionally, to validate whether PAR RAG performs similarly on single-hop question-answering tasks, we selected the
following three single-hop datasets as evaluation sources, with 500 instances chosen from each dataset:
1.SQuAD [57]. It is a foundational dataset where human annotators craft questions directly from Wikipedia
articles, focusing on precise answer extraction within text spans.
7

APREPRINT - APRIL 24, 2025
2.Natural Questions . [58]. It leverages real-world search queries from Google users, pairing them with relevant
Wikipedia content to simulate practical question-answering scenarios.
3.TriviaQA [59]. It draws upon trivia questions from diverse online quiz platforms, offering a challenging
testbed for open-domain question-answering systems.
4.2 Baselines
We compared multiple RAG methods with PAR RAG:
1.Vanilla RAG [60]: Vanilla RAG uses a dense retrieval encoder. It directly retrieves relevant evidence based on
the question, selecting the top-k retrieved results as context data. These results, along with the question, are
passed to the LLM through a prompt (see Appendix A.1) to generate the answer.
2.RAPTOR [29]: It recursively organizes document fragments into hierarchical summaries, constructing a tree
where each node represents the summary of its child nodes. During retrieval, RAPTOR evaluates the relevance
of each node in the tree and identifies the most relevant nodes, with the number of nodes limited by the top-k
criterion. Appendix A.2 provided the prompt used by RAPTOR.
3.IRCoT+HippoRAG [31]: HippoRAG simulates the long-term memory mechanism of the human brain by
extracting entities and their relationships from documents to build a knowledge graph. It establishes an
index and efficiently retrieves relevant data using the Personalized PageRank (PPR) algorithm [ 32]. It can be
combined with IRCoT to enhance QA capabilities, offering a unique advantage in complex reasoning tasks.
Appendix A.3 provided the prompt used by the current method.
4.3 Evaluation Metrics
We used the following metrics to evaluate the performance of the methods:
1.EM (Exact Match) [ 61]: It measures whether the predicted answer precisely matches the expected result,
focusing on exact matches.
2.F1[62]: It measures the overlap between the predicted answer and the expected result, focusing on partial
matches.
3.RTPQ (Response Time Per Query): This metric gauges the response speed of each method by measuring the
average time it takes to respond to a query.
4.CTPQ (Consumed Token Per Query): This metric measures the usage cost of each method by evaluating the
average number of tokens consumed per query.
4.4 Implementation Details
As shown in Table 1, we select ColBertV2 [ 63] as the retrieval encoder for Vanilla RAG. RAPTOR uses GPT-4o-mini
as the language model to generate summaries during the index construction process. Both HippoRAG and PAR RAG
utilize GPT-4o-mini to extract entities and their relationships during the construction of the knowledge graph, and
ColBertV2 as the retrieval encoder.
Qwen-Plus [ 64] is used as the default large language model(LLM) for all baselines during the multi-hop reasoning task
tests, with the TOP-K value set to 10.
Method Name Value
Vanilla RAG Retrieval Encoder ColBertV2
RAPTOREmbedding Model sentence-transformers/multi-qa-mpnet-base-cos-v1
Summary Model GPT-4o-mini
IRCoT+HippoRAG
& PAR RAGInformation Extraction Model GPT-4o-mini
Retrieval Encoder ColBertV2
Table 1: The primary experiment settings
8

APREPRINT - APRIL 24, 2025
5 Results and Analysis
5.1 Overall Results
The experimental results presented in Table 2 demonstrate that PAR RAG outperforms other methods in terms of both
EM and F1 metrics across the three multi-hop datasets. Specifically, on the HotpotQA and MuSiQue datasets, compared
to the second-best values of the baselines, PAR RAG achieved a 31.57% and 37.93% increase in EM, and a 10.25% and
31.78% increase in F1, respectively. On the 2Wiki dataset, PAR RAG also achieved a 15.87% and 7.43% improvement
over the second-best values of the baselines.
We attribute these improvements to two main factors. First, PAR RAG decomposes complex multi-hop questions
into a plan consisting of executable steps from top to bottom, reducing the risk of reasoning path deviation that
often arises from iterative decomposition or direct solving, thereby facilitating more accurate answers to multi-hop
questions. Secondly, during the execution of the plan, PAR RAG employs a multi-granularity verification mechanism to
validate and revise each step’s answer, thereby reducing the likelihood of process errors and mitigating the risk of error
propagation and amplification.
Name2Wiki HotpotQA MuSiQue Average
EM F1 EM F1 EM F1 EM F1
Vanilla RAG 0.45 0.516 0.42 0.564 0.15 0.206 0.34 0.429
RAPTOR 0.29 0.397 0.28 0.424 0.14 0.239 0.237 0.353
IRCoT+HippoRAG 0.63 0.726 0.57 0.722 0.29 0.387 0.497 0.612
PAR RAG 0.73 0.78 0.75 0.796 0.4 0.51 0.627 0.695
Table 2: Comparison results across multi-hop datasets for various RAG methods. Bold and underline indicate the best
and the second-best results.
5.2 Ablation Study
2Wiki HotpotQA MuSiQue0.00.20.40.60.81.0EM (exact match)0.61
0.51
0.260.64
0.52
0.30.730.75
0.4PAR RAG (w/o Plan Module) PAR RAG (w/o Review Module) PAR RAG
2Wiki HotpotQA MuSiQue0.00.20.40.60.81.0F10.6910.67
0.3770.708
0.677
0.4180.780.796
0.51PAR RAG (w/o Plan Module) PAR RAG (w/o Review Module) PAR RAG
Figure 2: QA performance on multi-hop datasets for ablation study. The left side represents EM metric, and the right
side represents F1 metric.
In this work, we emphasize that deviations in reasoning paths and process errors may trigger a butterfly effect, wherein
tiny errors propagate and amplify throughout the reasoning process. To mitigate these issues, PAR RAG integrates the
Plan Module, which employs a plan-driven approach to problem decomposition that minimizes the risk of deviations
in reasoning paths. Moreover, PAR RAG employs the Review Module to validate the execution outcomes at each
step, thereby reducing process errors. To evaluate the contributions of these key modules, we conduct the ablation
experiments on both modules.
In this experiment, PAR RAG (w/o Plan Module) uses the original question as a single-step plan, with all other
components remaining unchanged. On the other hand, PAR RAG (w/o Review Module) disables the Review Module,
so the preliminary answers generated for each step are applied without validation or revision.
The experimental results, as presented in Figure 2, reveal that both PAR RAG (w/o Plan Module) and PAR RAG
(w/o Review Module) exhibit significant performance degradation across all metrics in each dataset relative to the
full version of PAR RAG. These findings strongly suggest that, for complex problems, it is essential to leverage the
language understanding and reasoning capabilities of existing LLMs, adopting a top-down approach that decomposes the
9

APREPRINT - APRIL 24, 2025
problem into multiple executable reasoning steps. This process is crucial for enhancing the accuracy of the final answer.
Moreover, it demonstrates that the multi-granularity validation mechanism substantially improves the correctness and
credibility of intermediate results, thereby mitigating the propagation of process errors.
5.3 Analysis
5.3.1 The impact of language models on performance
2Wiki HotpotQA MuSiQue0.10.20.30.40.50.60.7EM(exact match)Qwen-Plus
GPT-4o-mini
Llama3-8B
Llama3-70B
2Wiki HotpotQA MuSiQue0.20.30.40.50.60.70.8F1Qwen-Plus
GPT-4o-mini
Llama3-8B
Llama3-70B
Figure 3: The analysis for the impact of language models(LMs) on 2Wiki,HotpotQA and MuSiQue. The left side
represents EM metric, while the right side represents F1 metric.
For complex multi-hop question-answering tasks, it is essential to decompose the problem comprehensively and
generate a multi-step plan that does not deviate from the reasoning path, provided that the problem’s intent is well
understood. The language understanding and reasoning abilities of LLMs, which play this key role, are a critical
bottleneck that influences the quality of the problem decomposition.
For verifying this idea, we compared several large language models and pre-trained language models (PLMs). The
experimental results, as shown in Figure 3, indicate that models with smaller parameters, such as GPT-4o-mini or
quantized versions of PLMs, exhibit significant performance degradation when compared to Qwen-Plus. It proves that
choosing a model with stronger reasoning capabilities is essential for solving complex problems effectively.
However, we also observed that even the best-performing model, Qwen-Plus, still exhibits issues like hallucination and
shortcomings in language understanding or reasoning[ 65,66]. Therefore, to ensure the correctness and credibility of
the result, it is crucial to validate and revise the intermediate results of the multi-step reasoning process using relevant
evidence. This approach helps mitigate errors that may arise during reasoning and increases the reliability of the
answers generated.
5.3.2 Applicability
NameNatural Questions SQuAD TriviaQA Average
EM F1 EM F1 EM F1 EM F1
Vanilla RAG 0.43 0.572 0.48 0.65 0.50 0.550 0.47 0.591
RAPTOR 0.27 0.422 0.34 0.508 0.44 0.549 0.35 0.493
IRCoT+HippoRAG 0.35 0.517 0.37 0.528 0.46 0.535 0.393 0.527
PAR RAG 0.5 0.615 0.49 0.586 0.54 0.628 0.51 0.61
Table 3: Comparison of applicability against single-hop datasets.
Although we primarily design PAR RAG for multi-hop question-answering tasks, it also demonstrates significant
performance in single-hop question-answering tasks. As shown in Table 3, PAR RAG outperforms other methods in
almost all metrics across the three datasets, except for the F1 score on the SQuAD dataset. It indicates that PAR RAG
has strong applicability and can be seamlessly applied to single-hop question-answering tasks as well.
10

APREPRINT - APRIL 24, 2025
5.3.3 Efficiency and Cost
Name2Wiki HotpotQA MuSiQue Average
RTPQ CTPQ RTPQ CTPQ RTPQ CTPQ RTPQ CTPQ
Vanilla RAG 4.05 854.50 2.28 775.38 3.88 832.70 3.40 820.86
RAPTOR 1.08 1073.48 1.17 1033.98 0.96 994.30 1.07 1033.92
IRCoT+HippoRAG 4.46 1773.77 5.23 2226.32 5.31 2070.50 5.00 2023.53
PAR RAG 23.32 10132.81 23.8 8902.45 33.01 11206.45 26.71 10080.57
Table 4: Comparison of efficiency on multi-hop datasets for all baselines.
NameNatural Questions SQuAD TriviaQA Average
RTPQ CTPQ RTPQ CTPQ RTPQ CTPQ RTPQ CTPQ
Vanilla RAG 1.60 853.10 1.18 836.99 1.11 861.86 1.297 850.65
RAPTOR 1.45 1011.23 1.16 988.94 0.97 1020.28 1.193 1006.82
IRCoT+HippoRAG 5.12 1483.16 4.12 1642.37 5.18 1521.10 4.807 1548.88
PAR RAG 26.05 7934.35 29.19 8884.82 26.06 7825.10 27.10 8214.76
Table 5: Comparison of cost on single-hop datasets for all baselines.
For both multi-hop and single-hop question-answering tasks, while PAR RAG demonstrates strong performance in
terms of EM and F1 scores, the efficiency and cost aspects, as shown in Tables 4 and 5, highlight significant challenges
that PAR RAG faces: slow response times and high usage costs. The tests on average response time per query (RTPQ)
and average consumed tokens per query (CTPQ) indicate that PAR RAG has longer response times and higher usage
costs compared to other methods.
We attribute these elevated metrics to the following factors: to mitigate the potential butterfly effects resulting from
process errors propagating through the system, PAR RAG must verify and revise each step of the plan. Consequently,
this triggers multiple calls to the LLM during the process, leading to an increase in both response time and usage costs.
The results of this experiment demonstrate that PAR RAG, utilizing its plan-driven problem decomposition approach
and multi-granularity verification mechanism, achieves strong performance. However, it also presents significant
challenges in response time and usage costs. Therefore, when applying PAR RAG, it is essential to strike a balance
according to specific needs, which highlights an important area for future improvements.
6 Conclusion and Future Work
This study introduces an innovative RAG method, PAR RAG, which constructs a systematic multi-step reasoning solution
by simulating human cognitive patterns to address complex problems. Specifically, PAR RAG employs a top-down
problem decomposition strategy. It first performs a global analysis of the problem’s intent and generates a structured
multi-step execution plan, effectively mitigating the common issue of reasoning path deviations observed in traditional
methods. During the execution phase, it incorporates multi-granularity semantic similarity and relevant information to
dynamically validate and refine intermediate results, significantly minimizing the occurrence of procedural errors. This
approach effectively mitigates the butterfly effect resulting from error propagation caused by deviation in reasoning
paths and errors in the intermediate results. Experiments on several multi-hop datasets demonstrate that the PAR RAG
framework outperforms existing state-of-the-art methods in terms of answer accuracy.
Although PAR RAG significantly improves the quality and credibility of multi-hop question-answering tasks through
comprehensive path planning and multi-granularity verification. However, the repeated invocation of large language
models during plan execution increases both response time and usage costs. Future research will focus on enhancing
efficiency by developing methods to optimize response speed and reduce costs without compromising answer quality.
Declaration of competing interest
The authors declare that they have no known competing financial interests or personal relationships that could have
appeared to influence the work reported in this paper.
11

APREPRINT - APRIL 24, 2025
References
[1]V . Mavi, A. Jangra, and A. Jatowt, “Multi-hop question answering,” 2024. [Online]. Available:
https://arxiv.org/abs/2204.09140
[2]Z. Yang, P. Qi, S. Zhang, Y . Bengio, W. Cohen, R. Salakhutdinov, and C. D. Manning, “Hotpotqa: A dataset
for diverse, explainable multi-hop question answering,” in Proceedings of the 2018 Conference on Empirical
Methods in Natural Language Processing . Association for Computational Linguistics, 2018. [Online]. Available:
https://aclanthology.org/D18-1259/
[3]P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel
et al. , “Retrieval-augmented generation for knowledge-intensive nlp tasks,” Advances in neural information
processing systems , vol. 33, pp. 9459–9474, 2020. [Online]. Available: https://arxiv.org/abs/2005.11401
[4]Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, H. Wang, and H. Wang, “Retrieval-augmented
generation for large language models: A survey,” arXiv preprint arXiv:2312.10997 , vol. 2, 2023. [Online].
Available: https://arxiv.org/abs/2312.10997
[5]S. Siriwardhana, R. Weerasekera, E. Wen, T. Kaluarachchi, R. Rana, and S. Nanayakkara, “Improving the
domain adaptation of retrieval augmented generation (rag) models for open domain question answering,”
Transactions of the Association for Computational Linguistics , vol. 11, 2023. [Online]. Available:
https://aclanthology.org/2023.tacl-1.1/
[6]Z. Jiang, M. Sun, L. Liang, and Z. Zhang, “Retrieve, summarize, plan: Advancing multi-hop question
answering with an iterative approach,” arXiv preprint arXiv:2407.13101 , 2024. [Online]. Available:
https://arxiv.org/abs/2407.13101
[7]S. Yang, E. Gribovskaya, N. Kassner, M. Geva, and S. Riedel, “Do large language models latently perform
multi-hop reasoning?” in Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers) . Association for Computational Linguistics, 2024, p. 10210–10229.
[Online]. Available: https://aclanthology.org/2024.acl-long.550/
[8]Z. Li, G. Jiang, H. Xie, L. Song, D. Lian, and Y . Wei, “Understanding and patching compositional reasoning in
llms,” in Findings of the Association for Computational Linguistics ACL 2024 . Association for Computational
Linguistics, 2024, p. 9668–9688. [Online]. Available: https://aclanthology.org/2024.findings-acl.576/
[9]M. Mitchell and D. C. Krakauer, “The debate over understanding in ai’s large language models,” Proceedings
of the National Academy of Sciences , vol. 120, no. 13, p. e2215907120, 2023. [Online]. Available:
https://www.pnas.org/doi/abs/10.1073/pnas.2215907120
[10] X. Zhang, C. Du, T. Pang, Q. Liu, W. Gao, and M. Lin, “Chain of preference optimization: Improving
chain-of-thought reasoning in llms,” in Advances in Neural Information Processing Systems , A. Globerson,
L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, Eds., vol. 37. Curran Associates,
Inc., 2024, pp. 333–356. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2024/file/
00d80722b756de0166523a87805dd00f-Paper-Conference.pdf
[11] O. Yoran, T. Wolfson, O. Ram, and J. Berant, “Making retrieval-augmented language models robust to irrelevant
context,” arXiv preprint arXiv:2310.01558 , 2023. [Online]. Available: https://arxiv.org/abs/2310.01558
[12] T. Li, G. Zhang, Q. D. Do, X. Yue, and W. Chen, “Long-context llms struggle with long in-context learning,”
2024. [Online]. Available: https://arxiv.org/abs/2404.02060
[13] F. Shi, X. Chen, K. Misra, N. Scales, D. Dohan, E. H. Chi, N. Schärli, and D. Zhou, “Large language models can
be easily distracted by irrelevant context,” in Proceedings of the 40th International Conference on Machine
Learning , ser. Proceedings of Machine Learning Research, A. Krause, E. Brunskill, K. Cho, B. Engelhardt,
S. Sabato, and J. Scarlett, Eds., vol. 202. PMLR, 23–29 Jul 2023, pp. 31 210–31 227. [Online]. Available:
https://proceedings.mlr.press/v202/shi23a.html
[14] R. Xu, Z. Qi, Z. Guo, C. Wang, H. Wang, Y . Zhang, and W. Xu, “Knowledge conflicts for llms: A survey,” 2024.
[Online]. Available: https://arxiv.org/abs/2403.08319
[15] E. Lorenz, “Predictability: Does the flap of a butterfly’s wing in brazil set off a tornado in texas?” 1972. [Online].
Available: https://www.ias.ac.in/article/fulltext/reso/020/03/0260-0263
[16] W. E. Deming, Out of the Crisis, reissue . MIT press, 2018.
[17] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman,
S. Anadkat et al. , “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774 , 2023. [Online]. Available:
https://arxiv.org/abs/2303.08774
12

APREPRINT - APRIL 24, 2025
[18] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y . Babaei, N. Bashlykov, S. Batra, P. Bhargava,
S. Bhosale et al. , “Llama 2: Open foundation and fine-tuned chat models,” arXiv preprint arXiv:2307.09288 ,
2023. [Online]. Available: https://arxiv.org/abs/2307.09288
[19] T. Kojima, S. S. Gu, M. Reid, Y . Matsuo, and Y . Iwasawa, “Large language models are zero-shot reasoners,”
inAdvances in Neural Information Processing Systems , S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave,
K. Cho, and A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp. 22 199–22 213. [Online]. Available: https://
proceedings.neurips.cc/paper_files/paper/2022/file/8bb0d291acd4acf06ef112099c16f326-Paper-Conference.pdf
[20] D. Zhou, N. Schärli, L. Hou, J. Wei, N. Scales, X. Wang, D. Schuurmans, C. Cui, O. Bousquet, Q. Le, and E. Chi,
“Least-to-most prompting enables complex reasoning in large language models,” 2023. [Online]. Available:
https://arxiv.org/abs/2205.10625
[21] S. S. Raman, V . Cohen, E. Rosen, I. Idrees, D. Paulius, and S. Tellex, “Planning with large language models via
corrective re-prompting,” in NeurIPS 2022 Foundation Models for Decision Making Workshop , 2022. [Online].
Available: https://openreview.net/forum?id=cMDMRBe1TKs
[22] Z. Wang, S. Cai, G. Chen, A. Liu, X. Ma, and Y . Liang, “Describe, explain, plan and select: Interactive
planning with large language models enables open-world multi-task agents,” 2024. [Online]. Available:
https://arxiv.org/abs/2302.01560
[23] L. Gao, Z. Dai, P. Pasupat, A. Chen, A. T. Chaganty, Y . Fan, V . Y . Zhao, N. Lao, H. Lee, D.-C. Juan, and K. Guu,
“Rarr: Researching and revising what language models say, using language models,” 2023. [Online]. Available:
https://arxiv.org/abs/2210.08726
[24] B. Wang, S. Chern, E. Chern, and P. Liu, “Halu-j: Critique-based hallucination judge,” 2024. [Online]. Available:
https://arxiv.org/abs/2407.12943
[25] S. Cao and L. Wang, “Verifiable generation with subsentence-level fine-grained citations,” 2024. [Online].
Available: https://arxiv.org/abs/2406.06125
[26] J. Kaddour, J. Harris, M. Mozes, H. Bradley, R. Raileanu, and R. McHardy, “Challenges and applications of large
language models,” 2023. [Online]. Available: https://arxiv.org/abs/2307.10169
[27] V . Rawte, A. Sheth, and A. Das, “A survey of hallucination in large foundation models,” 2023. [Online]. Available:
https://arxiv.org/abs/2309.05922
[28] S. Zhao, Y . Yang, Z. Wang, Z. He, L. K. Qiu, and L. Qiu, “Retrieval augmented generation (rag) and beyond: A
comprehensive survey on how to make your llms use external data more wisely,” arXiv preprint arXiv:2409.14924 ,
2024. [Online]. Available: https://arxiv.org/abs/2409.14924
[29] P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and C. D. Manning, “Raptor: Recursive abstractive
processing for tree-organized retrieval,” in The Twelfth International Conference on Learning Representations ,
2024. [Online]. Available: https://arxiv.org/abs/2401.18059
[30] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt, D. Metropolitansky, R. O. Ness, and J. Larson,
“From local to global: A graph rag approach to query-focused summarization,” arXiv preprint arXiv:2404.16130 ,
2024. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0014292123001599/pdfft?md5=
068734ea3858b2159ec11cb37d879849&pid=1-s2.0-S0014292123001599-main.pdf
[31] B. Jimenez Gutierrez, Y . Shu, Y . Gu, M. Yasunaga, and Y . Su, “Hipporag: Neurobiologically inspired
long-term memory for large language models,” Advances in Neural Information Processing Systems , vol. 37, pp.
59 532–59 569, 2024. [Online]. Available: https://arxiv.org/abs/2405.14831
[32] T. H. Haveliwala, “Topic-sensitive pagerank,” in Proceedings of the 11th international conference on World Wide
Web, ser. WWW02. ACM, May 2002. [Online]. Available: https://dl.acm.org/doi/abs/10.1145/511446.511513
[33] Z. Li, X. Chen, H. Yu, H. Lin, Y . Lu, Q. Tang, F. Huang, X. Han, L. Sun, and Y . Li, “Structrag: Boosting
knowledge intensive reasoning of llms via inference-time hybrid information structurization,” arXiv preprint
arXiv:2410.08815 , 2024. [Online]. Available: https://arxiv.org/abs/2410.08815
[34] N. Zhang, P. K. Choubey, A. Fabbri, G. Bernadett-Shapiro, R. Zhang, P. Mitra, C. Xiong, and C.-S. Wu, “Sirerag:
Indexing similar and related information for multihop reasoning,” arXiv preprint arXiv:2412.06206 , 2024.
[Online]. Available: https://arxiv.org/abs/2412.06206
[35] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, “Interleaving retrieval with chain-of-thought
reasoning for knowledge-intensive multi-step questions,” in Proceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers) , 2023, pp. 10 014–10 037. [Online]. Available:
https://aclanthology.org/2023.acl-long.557/
13

APREPRINT - APRIL 24, 2025
[36] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V . Le, D. Zhou et al. , “Chain-of-thought
prompting elicits reasoning in large language models,” Advances in neural information processing systems ,
vol. 35, pp. 24 824–24 837, 2022. [Online]. Available: https://arxiv.org/abs/2201.11903
[37] Z. Shao, Y . Gong, Y . Shen, M. Huang, N. Duan, and W. Chen, “Enhancing retrieval-augmented large language
models with iterative retrieval-generation synergy,” in Findings of the Association for Computational Linguistics:
EMNLP 2023 , 2023, pp. 9248–9274. [Online]. Available: https://aclanthology.org/2023.findings-emnlp.620/
[38] S. Jeong, J. Baek, S. Cho, S. J. Hwang, and J. C. Park, “Adaptive-rag: Learning to adapt retrieval-augmented large
language models through question complexity,” in Proceedings of the 2024 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long
Papers) , 2024, pp. 7029–7043. [Online]. Available: https://aclanthology.org/2024.naacl-long.389/
[39] S. Hao, Y . Gu, H. Ma, J. Hong, Z. Wang, D. Wang, and Z. Hu, “Reasoning with language model is planning with
world model,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing ,
2023, pp. 8154–8173. [Online]. Available: https://aclanthology.org/2023.emnlp-main.507/
[40] J. Wang, M. Chen, B. Hu, D. Yang, Z. Liu, Y . Shen, P. Wei, Z. Zhang, J. Gu, J. Zhou et al. ,
“Learning to plan for retrieval-augmented large language models from knowledge graphs,” in Findings of
the Association for Computational Linguistics: EMNLP 2024 , 2024, pp. 7813–7835. [Online]. Available:
https://aclanthology.org/2024.findings-emnlp.459/
[41] T. Gao, H. Yen, J. Yu, and D. Chen, “Enabling large language models to generate text with citations,” in
Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , 2023, pp.
6465–6488. [Online]. Available: https://aclanthology.org/2023.emnlp-main.398/
[42] Y . Li, S. Liang, M. Lyu, and L. Wang, “Making long-context language models better multi-hop reasoners,” in
Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers) , 2024, pp. 2462–2475. [Online]. Available: https://aclanthology.org/2024.acl-long.135/
[43] Z. Shi, S. Zhang, W. Sun, S. Gao, P. Ren, Z. Chen, and Z. Ren, “Generate-then-ground in retrieval-augmented
generation for multi-hop question answering,” in Proceedings of the 62nd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers) , 2024, pp. 7339–7353. [Online]. Available:
https://aclanthology.org/2024.acl-long.397/
[44] L. Ranaldi, M. Valentino, and A. Freitas, “Eliciting critical reasoning in retrieval-augmented language
models via contrastive explanations,” arXiv preprint arXiv:2410.22874 , 2024. [Online]. Available:
https://arxiv.org/abs/2410.22874
[45] B. Ji, H. Liu, M. Du, and S.-K. Ng, “Chain-of-thought improves text generation with citations in large
language models,” in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 38, no. 16, 2024, pp.
18 345–18 353. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/29794/31374
[46] M. Song, S. H. Sim, R. Bhardwaj, H. L. Chieu, N. Majumder, and S. Poria, “Measuring and enhancing
trustworthiness of llms in rag through grounded attributions and learning to refuse,” in The Thirteenth International
Conference on Learning Representations , 2025. [Online]. Available: https://arxiv.org/abs/2409.11242
[47] Z. Wu, Q. Zeng, Z. Zhang, Z. Tan, C. Shen, and M. Jiang, “Large language models can self-correct with key
condition verification,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language
Processing . Miami, Florida, USA: Association for Computational Linguistics, 2024, pp. 12 846–12 867.
[Online]. Available: https://aclanthology.org/2024.emnlp-main.714/
[48] R. Kamoi, Y . Zhang, N. Zhang, J. Han, and R. Zhang, “When can llms actually correct their own mistakes? a
critical survey of self-correction of llms,” Transactions of the Association for Computational Linguistics , vol. 12,
pp. 1417–1440, 2024. [Online]. Available: https://aclanthology.org/2024.tacl-1.78/
[49] J. Huang, X. Chen, S. Mishra, H. S. Zheng, A. W. Yu, X. Song, and D. Zhou, “Large language
models cannot self-correct reasoning yet,” arXiv preprint arXiv:2310.01798 , 2023. [Online]. Available:
https://arxiv.org/abs/2310.01798
[50] Z. Jiang, F. F. Xu, L. Gao, Z. Sun, Q. Liu, J. Dwivedi-Yu, Y . Yang, J. Callan, and G. Neubig, “Active retrieval
augmented generation,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language
Processing , 2023, pp. 7969–7992. [Online]. Available: https://aclanthology.org/2023.emnlp-main.495/
[51] R. Zhao, X. Li, S. Joty, C. Qin, and L. Bing, “Verify-and-edit: A knowledge-enhanced chain-of-thought
framework,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers) , 2023, pp. 5823–5840. [Online]. Available: https://aclanthology.org/2023.acl-long.320/
14

APREPRINT - APRIL 24, 2025
[52] A. Asai, Z. Wu, Y . Wang, A. Sil, and H. Hajishirzi, “Self-rag: Learning to retrieve, generate, and critique through
self-reflection,” in The Twelfth International Conference on Learning Representations , 2024. [Online]. Available:
https://arxiv.org/abs/2310.11511
[53] S.-Q. Yan, J.-C. Gu, Y . Zhu, and Z.-H. Ling, “Corrective retrieval augmented generation,” arXiv preprint
arXiv:2401.15884 , 2024. [Online]. Available: https://arxiv.org/abs/2401.15884
[54] S. Zhou, B. Yu, A. Sun, C. Long, J. Li, H. Yu, J. Sun, and Y . Li, “A survey on neural open information
extraction: Current status and future directions,” arXiv preprint arXiv:2205.11725 , 2022. [Online]. Available:
https://www.ijcai.org/proceedings/2022/0793.pdf
[55] X. Ho, A.-K. D. Nguyen, S. Sugawara, and A. Aizawa, “Constructing a multi-hop QA dataset for comprehensive
evaluation of reasoning steps,” in Proceedings of the 28th International Conference on Computational Linguistics .
Barcelona, Spain (Online): International Committee on Computational Linguistics, Dec. 2020, pp. 6609–6625.
[Online]. Available: https://aclanthology.org/2020.coling-main.580/
[56] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, “Musique: Multihop questions via single-hop
question composition,” Transactions of the Association for Computational Linguistics , vol. 10, pp. 539–554, 2022.
[Online]. Available: https://aclanthology.org/2022.tacl-1.31/
[57] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, “Squad: 100,000+ questions for machine comprehension of
text,” in Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing , 2016, pp.
2383–2392. [Online]. Available: https://aclanthology.org/D16-1264/
[58] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh, C. Alberti, D. Epstein, I. Polosukhin, J. Devlin,
K. Lee et al. , “Natural questions: A benchmark for question answering research,” Transactions of the Association
for Computational Linguistics , vol. 7, pp. 452–466, 2019. [Online]. Available: https://aclanthology.org/Q19-1026/
[59] M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer, “Triviaqa: A large scale distantly supervised
challenge dataset for reading comprehension,” in Proceedings of the 55th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers) , 2017, pp. 1601–1611. [Online]. Available:
https://aclanthology.org/P17-1147/
[60] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, M. Wang, and H. Wang, “Retrieval-augmented
generation for large language models: A survey,” 2024. [Online]. Available: https://arxiv.org/abs/2312.10997
[61] A. Mallen, A. Asai, V . Zhong, R. Das, D. Khashabi, and H. Hajishirzi, “When not to trust language models:
Investigating effectiveness of parametric and non-parametric memories,” in Proceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , 2023, pp. 9802–9822.
[Online]. Available: https://aclanthology.org/2023.acl-long.546/
[62] J. Baek, S. Jeong, M. Kang, J. C. Park, and S. Hwang, “Knowledge-augmented language model verification,”
inProceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , 2023, pp.
1720–1736. [Online]. Available: https://aclanthology.org/2023.emnlp-main.107/
[63] K. Santhanam, O. Khattab, J. Saad-Falcon, C. Potts, and M. Zaharia, “Colbertv2: Effective and efficient retrieval
via lightweight late interaction,” in Proceedings of the 2022 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies , 2022, pp. 3715–3734. [Online].
Available: https://aclanthology.org/2022.naacl-main.272/
[64] S. Pan, K. Liu, W. Chen, and B. He, “Performance analysis of chinese large language models in solving math
word problems,” in 2024 International Conference on Intelligent Education and Intelligent Research (IEIR) , 2024,
pp. 1–8.
[65] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, “Lost in the middle: How
language models use long contexts,” Transactions of the Association for Computational Linguistics , vol. 12, pp.
157–173, 2024. [Online]. Available: https://aclanthology.org/2024.tacl-1.9/
[66] C. Zheng, L. Li, Q. Dong, Y . Fan, Z. Wu, J. Xu, and B. Chang, “Can we edit factual knowledge by in-context
learning?” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , 2023,
pp. 4862–4876. [Online]. Available: https://aclanthology.org/2023.emnlp-main.296/
A Appendix. Prompts
A.1 Vanilla RAG
Given the following context,
15

APREPRINT - APRIL 24, 2025
answer the question in the format:
"So the answer is: [ANSWER]"
If the answer is not found in the context, respond with "I don’t know."
Context:
{context}
Question:
{question}
A.2 RAPTOR
Given the following context: {context}
Give the best full answer amongst the option to question: {question}.
If you don’t know the answer, just return "I don’t know".
Return only the answers without additional explanations or irrelevant information.
A.3 IRCoT+HippoRAG
You serve as an intelligent assistant,
adept at facilitating users through complex,
multi-hop reasoning across multiple documents.
This task is illustrated through demonstrations,
each consisting of a document set paired with a relevant question
and its multi-hop reasoning thoughts.
Your task is to generate one thought for current step,
Respond only, do not explain yourself or output anything else.
If you reach what you believe to be the final step, start with "So the answer is:".
{samples}
Question: {query}
Thought: {thought}
A.4 PAR RAG
A.4.1 Plan Module
You are very good at solving complex problems by breaking them down.
Think carefully, build a step-by-step plan for this [question]:
{current_question} using JSON format.
For each step in the plan, generate a sub-question.
# Format instructions
Use the following Strict format,
only choose an action from the list:[Retrieve, Answer]:
[
{
"Thought":"[your thought about the current step]"
"Question":"[the question you generated for the current step]"
"Action": "[the action you chose]"
}
]
Return only your answer,
do not explain it and do not output anything that is not relevant to the answer.
A.4.2 Action Module
The prompt used to answer the question of each step
Please provide an answer based solely on the provided sources.
16

APREPRINT - APRIL 24, 2025
When referencing information from a source,
cite the appropriate source(s) using their corresponding numbers.
Every answer should include at least one source citation.
Only cite a source when you are explicitly referencing it.
Directly respond your answer,
do not explain your answer or distract from irrelevant information in the source,
and do not output anything that is not relevant to the question.
# Examples begin
Source 1:
The sky is red in the evening and blue in the morning.
Source 2:
Water is wet when the sky is red.
Source 3:
Wolves and dogs belong to the species, Canis lupus.
Query: When is water wet?
Answer: In the evening. Water is wet when the sky is red[2],
which occurs in the evening [1].
# Examples end
Now it’s your turn. Below are several numbered sources of information:
{context}
Query: {question}
Answer:
The prompt used to refine the question of the next step
Refine the current question:{next_question} based on the information provided: {answer}.
Remember: the question you optimize must aim to answer the original question: {question}.
Respond with the refined question only, do not explain yourself or output anything else.
The prompt used to generate the final answer
Answer this question: {question}, referring to the following reasoning trajectories:
# Trajectories start
{trajectories}
# Trajectories end
If you don’t know the answer, just return "I don’t know".
Return only your answer,
do not explain it and do not output anything that is not relevant to the answer.
A.4.3 Review Module
Given the question: {question},
and the answer: {answer},
and the corresponding background:
{context}
You need to check whether the answer is correct,
your judgment must be based on the question asked
and the answer must be supported by clear background knowledge.
If you think the answer correctly answers the question
and does not need to make any corrections,
just output the result in JSON format as the following:
{"Status": "PASS", "Answer": "The original answer"}
If you find some errors in the answer
and the errors coule be rectified using the
background knowledge,
revise the answer to make it better,
when referencing information from the background knowledge,
cite the appropriate source(s) using their corresponding numbers,
every answer should include at least one source citation,
only cite a source when you are explicitly referencing it,
17

APREPRINT - APRIL 24, 2025
then output the result in JSON format as the following:
{"Status": "REVISED", "Answer": "The revised answer"}
If you find some necessary details are ignored,
generate a new question to make the answer more plausible according to the related text,
then output the result in JSON format as the following:
{"Status": "UNCONFIDENT", "Question": "The new question you generated"}
Respond only, do not explain yourself or output anything else.
18
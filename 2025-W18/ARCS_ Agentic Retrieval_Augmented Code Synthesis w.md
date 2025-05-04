# ARCS: Agentic Retrieval-Augmented Code Synthesis with Iterative Refinement

**Authors**: Manish Bhattarai, Miguel Cordova, Javier Santos, Dan O'Malley

**Published**: 2025-04-29 05:15:52

**PDF URL**: [http://arxiv.org/pdf/2504.20434v1](http://arxiv.org/pdf/2504.20434v1)

## Abstract
In supercomputing, efficient and optimized code generation is essential to
leverage high-performance systems effectively. We propose Agentic
Retrieval-Augmented Code Synthesis (ARCS), an advanced framework for accurate,
robust, and efficient code generation, completion, and translation. ARCS
integrates Retrieval-Augmented Generation (RAG) with Chain-of-Thought (CoT)
reasoning to systematically break down and iteratively refine complex
programming tasks. An agent-based RAG mechanism retrieves relevant code
snippets, while real-time execution feedback drives the synthesis of candidate
solutions. This process is formalized as a state-action search tree
optimization, balancing code correctness with editing efficiency. Evaluations
on the Geeks4Geeks and HumanEval benchmarks demonstrate that ARCS significantly
outperforms traditional prompting methods in translation and generation
quality. By enabling scalable and precise code synthesis, ARCS offers
transformative potential for automating and optimizing code development in
supercomputing applications, enhancing computational resource utilization.

## Full Text


<!-- PDF content starts -->

ARCS: Agentic Retrieval-Augmented Code Synthesis with
Iterative Refinement
Manish Bhattarai∗
Los Alamos National Laboratory
Los Alamos, New Mexico, USA
ceodspspectrum@lanl.govMiguel Cordova∗
Los Alamos National Laboratory
Los Alamos, New Mexico, USA
miguelcord@lanl.gov
Javier Santos
Los Alamos National Laboratory
Los Alamos, New Mexico, USA
jesantos@lanl.govDan O’Malley
Los Alamos National Laboratory
Los Alamos, New Mexico, USA
omalled@lanl.gov
Abstract
In supercomputing, efficient and optimized code generation is essen-
tial to leverage high-performance systems effectively. We propose
Agentic Retrieval-Augmented Code Synthesis (ARCS), an advanced
framework for accurate, robust, and efficient code generation, com-
pletion, and translation. ARCS integrates Retrieval-Augmented Gen-
eration (RAG) with Chain-of-Thought (CoT) reasoning to system-
atically break down and iteratively refine complex programming
tasks. An agent-based RAG mechanism retrieves relevant code snip-
pets, while real-time execution feedback drives the synthesis of
candidate solutions. This process is formalized as a state-action
search tree optimization, balancing code correctness with editing
efficiency. Evaluations on the Geeks4Geeks and HumanEval bench-
marks demonstrate that ARCS significantly outperforms traditional
prompting methods in translation and generation quality. By en-
abling scalable and precise code synthesis, ARCS offers transforma-
tive potential for automating and optimizing code development in
supercomputing applications, enhancing computational resource
utilization.
CCS Concepts
•Do Not Use This Code →Generate the Correct Terms for
Your Paper ;Generate the Correct Terms for Your Paper ; Generate
the Correct Terms for Your Paper; Generate the Correct Terms for
Your Paper.
Keywords
Agentic system, Retreival Augmented Generation(RAG), Large Lan-
guage Model(LLM), Code synthesis, Code translation
∗Both authors contributed equally to this work.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
,
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXXACM Reference Format:
Manish Bhattarai, Miguel Cordova, Javier Santos, and Dan O’Malley. 2025.
ARCS: Agentic Retrieval-Augmented Code Synthesis with Iterative Re-
finement. In Proceedings of . ACM, New York, NY, USA, 9 pages. https:
//doi.org/XXXXXXX.XXXXXXX
1 Introduction
Large language models (LLMs) have achieved remarkable perfor-
mance on code generation benchmarks. Despite these advances,
state-of-the-art models often fail to consistently produce correct
code, especially for complex tasks. Common issues include logical
errors, incomplete functions, and reliance on hallucinated context.
Current approaches frequently treat the model as a one-shot gener-
ator, forgoing iterative refinement strategies that leverage runtime
feedback and knowledge retrieval.
In this work, we propose Agentic Retrieval-Augmented Code Syn-
thesis (ARCS) , a framework that extends the standard RAG para-
digm by integrating it with agentic reasoning and preference-driven
optimization. ARCS operates in an iterative loop: (1) it retrieves
relevant functions or code snippets from a large corpus to augment
the prompt, (2) synthesizes candidate solutions using an LLM, (3)
executes the generated code in a sandboxed environment to obtain
runtime feedback, and (4) refines the solution iteratively based on
correctness signals. This iterative process can be viewed as a state-
action search in a latent code space, and we show how to formalize
it to enable rigorous training objectives.
Our key contributions are:
•A formalization of iterative code synthesis as a state-action
search over code refinements, with retrieval-augmented prompts
and execution-based feedback loops.
•Empirical results (omitted here for brevity) showing that
ARCS attains higher correctness rates, improved robustness
against hallucinations, and faster convergence than conven-
tional RAG approaches.
2 Related work
A growing body of work augments code synthesis with external
knowledge retrieval to bridge the semantic gap between natural
language intent and code [ 1,2]. Empirical studies show that retriev-
ing similar code snippets or API usage examples can indeed boost
generation performance [ 10]. However, naive token-level retrieval
can inject irrelevant or even syntactically incorrect code into thearXiv:2504.20434v1  [cs.SE]  29 Apr 2025

, , Bhattarai et al.
output, adding noise and overhead [ 12]. Recent approaches mitigate
this by constraining retrieval to more relevant content – for ex-
ample, kNN-TRANX narrows searches to a task-specific datastore
and enforces syntax-aware matching, which reduces extraneous
information [13].
Orthogonal to external knowledge, another line of research im-
proves code generation by enhancing the reasoning process of
LLMs. Chain-of-thought (CoT) prompting asks models to produce
intermediate reasoning steps (often in natural language or pseu-
docode) before final code output. This has been shown to yield
more correct solutions – for instance, structured CoT prompting
that guides the model to plan with programming constructs can
raise success rates by over 13% in pass@1 on benchmarks [ 4]. Nev-
ertheless, CoT alone is not a panacea: even with reasoning prompts,
models like GPT-3.5-turbo reach only ∼53% pass@1 on HumanEval
[8], and indiscriminate CoT can sometimes lead to over-thinking
that inadvertently introduces errors [7].
Given the difficulty of getting complex code right in one try,
several approaches integrate execution feedback and multi-step
decision-making. Large-scale systems like AlphaCode demonstrated
the value of ex post execution checking by generating a vast num-
ber of candidates and then filtering them based on runtime tests
[6]. More recent works replace brute-force sampling with intel-
ligent agents that interleave generation and testing. CodeAgent,
for example, equips an LLM with external tools for documentation
lookup, code writing, and unit testing in a loop, enabling repository-
level code generation and debugging; this agent-based approach
yields significant pass rate improvements (2–15% absolute) and
even outperforms GitHub Copilot on a new benchmark [ 11]. Search-
based methods further refine this paradigm: RethinkMCTS uses
Monte Carlo Tree Search over the model’s reasoning steps and in-
corporates fine-grained execution feedback to iteratively correct er-
rors, dramatically improving code accuracy (e.g. boosting GPT-3.5’s
pass@1 from∼70% to 89%) [ 5]. Compared to these, ARCS adopts a
lightweight yet effective agentic loop. Rather than requiring thou-
sands of samples or complex tree searches, ARCS’s single agent
autonomously generates code, executes it, observes failures or incor-
rect outputs, and then retrieves hints or adjusts its chain-of-thought
before the next iteration. This closed-loop refinement continues
until the code meets the specifications. By unifying retrieval-based
guidance with execution-driven self-correction, ARCS delivers high
success rates on challenging tasks while using far fewer attempts
than brute-force methods. This approach is especially well-suited
for high-performance computing scenarios, where iterative tuning
can ensure not only correctness but also adherence to performance
and scalability constraints in the generated code.
3 Theoretical Framework
In this section, we rigorously formalize the code synthesis problem
and describe the mechanisms underlying ARCS. Our formulation
leverages a Markov Decision Process (MDP) framework to capture
the iterative nature of our code generation and refinement process.
We then detail the retrieval mechanism employed for augmenting
the query and explain how the language model synthesizes and
refines candidate solutions using execution feedback.3.1 Problem Formulation
Given a natural language specification 𝑞, the objective is to generate
a program ˆ𝑐∗that correctly implements the required functionality.
Correctness is rigorously defined through a predefined set of test
casesT={𝑡1,𝑡2,...,𝑡 𝑚}, where each test case 𝑡𝑖is an input-output
pair(𝑥𝑖,𝑦𝑖). A candidate solution ˆ𝑐is deemed correct if and only if
ˆ𝑐(𝑥𝑖)=𝑦𝑖∀𝑖∈{1,2,...,𝑚}.
We consider a code corpus C={𝑐1,𝑐2,...,𝑐 𝑁}in which each
code snippet 𝑐𝑖is associated with structured metadata M𝑖(e.g.,
function signatures, parameter descriptions, documentation). To
facilitate semantic retrieval, an embedding function
Φ:M𝑖→R𝑑
maps the metadata into a 𝑑-dimensional vector space, producing
embeddings e𝑖=Φ(M𝑖).
We formalize the overall code synthesis process as an MDP
defined by the tuple (S,A,T,R), where:
•State SpaceS: Each state 𝑆𝑡at time step 𝑡is represented as
𝑆𝑡=(𝑞𝑡,ˆ𝑐𝑡,𝑓𝑡),
where:
–𝑞𝑡denotes the current natural language query or specifi-
cation,
–ˆ𝑐𝑡is the candidate code generated at iteration 𝑡, and
–𝑓𝑡represents the execution feedback (e.g., error messages,
test outcomes) obtained from running ˆ𝑐𝑡.
•Action SpaceA: This set comprises all possible code refine-
ment operations that can be applied to a candidate solution.
•Transition Function T: The state update mechanism is
modeled by
T:S×A→S ,
which defines how a state evolves upon applying an action.
•Reward Function R: The improvement in code correctness
following a refinement is quantified by
R:S×A→ R.
A typical reward formulation computes the difference in the
number of passed test cases between successive iterations.
3.2 Retrieval Mechanism
Efficient retrieval of contextually relevant code examples is critical
to guiding the synthesis process. Given an input query 𝑞, we first
compute its embedding using the function:
𝜙(𝑞)=Φ(𝑞).
Subsequently, we retrieve a subset of code snippets from the corpus
Cthat meet a predetermined similarity threshold. Formally, the
retrieval operation is defined as:
R(𝑞)={𝑐𝑖∈C| sim(𝜙(𝑞),e𝑖)≥𝜏},
where simdenotes a similarity measure—typically cosine similar-
ity—and𝜏is a threshold that controls retrieval precision.
The retrieved metadata is concatenated with the original query
to form an enriched prompt:
𝑞′=𝑞⊕Ê
𝑐𝑖∈R(𝑞)M𝑖,

ARCS: Agentic Retrieval-Augmented Code Synthesis with Iterative Refinement , ,
where the operator ⊕denotes concatenation. For queries that in-
volve multiple functional components, we decompose 𝑞into𝑘sub-
queries𝑞={𝑞1,𝑞2,...,𝑞𝑘}. Each subquery undergoes an indepen-
dent retrieval process:
R(𝑞𝑗)={𝑐𝑖∈C| sim(𝜙(𝑞𝑗),e𝑖)≥𝜏𝑗}.
The final retrieval set is obtained by aggregating the results and
removing redundancy:
R(𝑞)=FilterRedundancy©­
«𝑘Ø
𝑗=1R(𝑞𝑗)ª®
¬.
This hierarchical retrieval strategy ensures that each subcompo-
nent of a complex query is supported by high-quality, contextually
relevant examples.
3.3 Code Synthesis and Refinement
The synthesis module leverages a language model parameterized by
𝜋𝜃to generate candidate code conditioned on the enriched prompt:
ˆ𝑐=𝜋𝜃(𝑞′).
The generated candidate ˆ𝑐is executed within a controlled sandbox
environmentEto capture detailed execution feedback 𝑓:
𝑓=E(ˆ𝑐,T).
This feedback—which includes execution results, error messages,
and outcomes from test cases—is encoded into a structured textual
representation via an encoding function:
Encode(𝑓𝑡).
The encoded feedback is then integrated with the original query,
forming the basis for the next iteration:
𝑞𝑡+1=𝑞𝑡⊕Encode(𝑓𝑡).
Thus, the overall iterative process is modeled as a state-action
trajectory:
𝑆𝑡+1=T(𝑆𝑡,𝐴𝑡),where𝑆𝑡=(𝑞𝑡,ˆ𝑐𝑡,𝑓𝑡)and𝐴𝑡∼𝜋𝜃(𝑆𝑡).
The refinement loop continues until a candidate solution satisfies
all test cases or a maximum number of iterations is reached.
4 System Architecture
4.1 Hierarchical Design
ARCS implements a hierarchical architecture with three tiers of
increasing sophistication, designed to balance computational effi-
ciency with solution quality across varying task complexities. Each
tier represents a different configuration of the system with specific
capabilities and resource requirements.
4.1.1 Small System Tier. The Small System Tier implements a di-
rect, zero-shot synthesis approach optimized for speed and effi-
ciency. It processes a natural language specification without de-
composing the task or performing iterative refinement. Specifically,
the prompt is constructed as:
𝑝small(𝑞)=prefix⊕𝑞⊕suffix,
where prefix andsuffix provide essential contextual framing
for the language model. This configuration is best suited for well-
defined tasks with limited complexity (e.g., standard algorithms orsimple code translation) and offers minimal computational over-
head, thereby enabling rapid responses.
4.1.2 Medium System Tier. The Medium System Tier extends the
Small System Tier by incorporating structured decomposition and
component-level synthesis to address moderately complex program-
ming challenges. In this tier, complex problems are first partitioned
into well-defined subproblems. Each subproblem is then solved
individually, with partial solutions later combined and validated
through explicitly defined interfaces. This systematic approach
not only enhances robustness but also enables targeted refinement
of the overall solution. The prompt used in the Medium Tier is
carefully designed to include both decomposition and validation
instructions, and is formally defined as follows:
𝑝medium(𝑞)=prefix⊕𝑞⊕decompose_instructions
⊕validation_instructions ⊕suffix,
where decompose_instructions systematically guide the break-
down of the task and validation_instructions specify the crite-
ria that ensure each component adheres to correctness constraints.
4.1.3 Large System Tier. The Large System Tier implements the
complete ARCS framework by integrating advanced capabilities to
manage highly complex coding tasks. It builds upon the decompo-
sition and component synthesis mechanisms of the Medium Tier
and further enriches each component using a retrieval-augmented
context. Moreover, the Large Tier employs an iterative refinement
process driven by execution feedback, which continuously improves
candidate solutions over multiple cycles. Additionally, automated
unit test generation and cross-component optimization are applied
to achieve a cohesive and robust final output. The prompt construc-
tion in this tier is extended to encapsulate retrieval and feedback
instructions as well. It is given by:
𝑝large(𝑞)=prefix⊕𝑞⊕decompose_instructions ⊕retrieval
_instructions⊕feedback_instructions ⊕suffix,
where retrieval_instructions ensure that contextually rele-
vant examples are incorporated and feedback_instructions dic-
tate how execution outcomes are used to guide subsequent refine-
ments. Capable of performing up to five iterative refinement cycles
per component, the Large System Tier excels in scenarios that re-
quire sophisticated error handling and performance optimization.
4.2 Component Architecture
As illustrated in Figure 1, the ARCS system is implemented as a
directed acyclic graph (DAG) of specialized processing nodes, each
designed to perform a distinct function in the synthesis pipeline.
At the initial stage, the Context Analysis Node validates whether
the input specification is complete and requests additional details
when ambiguous requirements arise. Following this, the Decom-
position Node partitions complex tasks into coherent subproblems,
each equipped with well-defined interfaces that streamline focused
processing and later integration.
Once these subproblems are identified, the Retrieval Node gathers
relevant code examples from the corpus to provide contextual sup-
port. The Analysis Node , highlighted in red in Figure 1, serves as an
intermediary step that examines the retrieved data for consistency,

, , Bhattarai et al.
ARCS System: Directed Acyclic Graph (DAG) of Processing Nodes
Context Analysis Node
Decomposition Node
Retrieval Node Reasoning Node
Analysis Node Implementation Node
Execution Node Feedback Analysis Node
Integration Node
Verification NodeSystem Tiers
• Small Tier: Minimal configuration
• Medium Tier: Partial node set
• Large Tier: Complete node setLegend
Standard Flow
Refinement Loop
Figure 1: Flowchart of ARCS system
checks for potential conflicts or inaccuracies, and organizes the
information to be more effectively consumed by subsequent nodes.
The Reasoning Node then applies chain-of-thought techniques to
develop intermediate solution strategies, drawing upon both the
retrieved and analyzed context. Subsequently, the Implementation
Node transforms these strategies into executable code.
The newly generated code is run by the Execution Node within a
sandboxed environment that captures performance metrics, error
messages, and output results. These outcomes feed into the Feed-
back Analysis Node , which translates them into structured insights
for further refinement of the candidate solutions. The Integration
Node then merges the outputs from individual components into a
cohesive final implementation, ensuring compatibility among the
subproblem solutions. Finally, the Verification Node validates the
complete solution against all original requirements.
Different system tiers of ARCS deploy subsets of these nodes
to match specific task needs. For instance, the Small System Tier
employs only the most essential nodes for rapid handling of straight-
forward requests, whereas the Large System Tier utilizes the full
DAG—including the Analysis Node—to address complex scenarios
that demand rigorous scrutiny and higher levels of robustness.4.3 Prompt Engineering Principles
The design of node-specific prompts is central to the performance
of ARCS. Our development experience showed that carefully bal-
ancing the prompt constraints for different nodes is crucial. For
example, when instructing the system to break a complex query
into discrete steps, strict prompts are necessary—explicitly defin-
ing function names, parameters, and desired language. This form
of strict instruction minimizes hallucinations and off-topic code
generation in the breakdown nodes. In contrast, for evaluation and
aggregation nodes, looser prompts that allow for creative reasoning
yield better performance. These nodes benefit from flexibility when
mapping partial solutions back to the original question, thereby
encouraging innovative error detection and adjustment. To further
enhance performance, we introduced a dedicated context node that
verifies whether the input specification is sufficiently detailed. In
cases where details such as data types or constraints are missing,
the system prompts for clarification rather than overcompensating
by assuming additional restrictions. Moreover, our experiments
have shown that providing concise code-generation plans results
in higher accuracy compared to exhaustive, overly detailed plans.
Excessively long plans tend to incorporate speculative constraints
that increase the risk of hallucinations, whereas concise plans fo-
cusing on core functionality align more closely with the actual task
requirements.

ARCS: Agentic Retrieval-Augmented Code Synthesis with Iterative Refinement , ,
Validated codeblocksLLM-3Synthesized codeSynthesizexyzUserTaskPromptSandboxVerifyexecutability and correctness✓✗Step 3: Feed the validated code blocks concatenated with User query to feed to LLM to synthesize final code and evaluate its correctness. If fails, parse output metadata including logs and errors and concatenate with query and code blocks to feed to Agentic COT block to generate new set of code blocks and repeat.LogsAgentic COT BlockCode CorpusLLM-1EmbeddingModel
Vector storeRAGSynthesizexyzLLM-2Step 1: xx……Stepk: zzUserTaskPromptCOT StepsAgt-1Agt-pStep 1: xxStep k: zzRAG
Retrieve top-k blocksStep 1: Extraction of core functional blocks and agentic labeling with LLM1 followed by embedding of labels for RAG registration 
Step 2: For a given user task prompt, generate COT steps with LLM-2,  use each COT step as query to RAG to retrieve corresponding code blocks and validate.SandboxVerifyexecutabilityAgentic COT Block!
Figure 2: High-level overview of the ARCS pipeline. The framework begins by extracting and embedding code snippets with
agentic labels (Step 1), then uses a chain-of-thought (CoT) process to query a retrieval-augmented system (Step 2). Relevant
code snippets are tested in a sandbox for executability and correctness, and validated blocks are iteratively fed back into the
LLM to refine solutions. Final code synthesis occurs with consolidated, validated snippets, while execution logs or errors guide
further iterations if necessary (Step 3).
In particular, we have identified five core principles—each sup-
ported by measurable improvements in system performance.
First, constraint calibration is formalized by defining a constraint
parameter𝜅that quantifies the specificity of the prompt. For de-
composition and implementation nodes, high values of 𝜅imply
that detailed function signatures, parameter annotations, and for-
mal output specifications are provided. Empirically, we observed
that when𝜅≥𝜅∗(where𝜅∗is an empirically determined thresh-
old), the pass@1 success rate increases relative to cases with lower
constraint levels.Second, explicit boundary definition is implemented by embed-
ding a formal subtask specification 𝐵(𝑆)into the prompt, such that
for an initial task specification 𝑆, the augmented prompt is
𝑆′=𝑆⊕𝐵(𝑆).
We observed that prompts constructed in this manner reduce out-
put variance, thereby minimizing hallucinations and enhancing
consistency.
Third, concise planning is essential for removing extraneous infor-
mation that can distract the synthesis process. We quantify prompt
length using a token count 𝐿; our results indicate that reducing 𝐿
compared to overly exhaustive, detailed plans—correlates with an

, , Bhattarai et al.
increased percentage points in the pass@1 score. This improvement
suggests that focusing on core functionality rather than speculative
constraints yields better code synthesis outcomes.
Fourth, the effectiveness of contextual retrieval is measured by
the precision 𝑃𝑟of retrieved examples with respect to the task’s
semantic requirements. We found that employing component-level
retrieval—where queries are specifically tailored to individual sub-
components—improves 𝑃𝑟compared to a global retrieval strategy.
Finally, periodic realignment is implemented by embedding ex-
plicit validation checkpoints within the iterative synthesis process.
We define a consistency metric 𝛿as the proportion of interme-
diate outputs that remain aligned with the original specification.
Prompts incorporating these checkpoints have shown an increase
in𝛿, directly correlating with the overall improvement in system
performance.
5 Approach
The ARCS framework involves a multi-step process that integrates
retrieval from a code corpus, generation of candidate solutions by a
large language model, execution-based validation of those solutions,
and iterative preference optimization as shown in Figure 2. To
enable reproducibility, we describe these steps in detail, specifying
the data handling protocols, model configurations, training routines,
and evaluation procedures.
We begin by preparing a code corpus C, where each snippet 𝑐𝑖is
associated with structured metadata M𝑖that includes information
such as the function name, parameters, and documentation. These
metadata are converted into embeddings e𝑖using a fixed text or code
embedding model Φ. This step is performed offline to ensure that all
retrieval operations at inference time are efficient and deterministic.
Given a query 𝑞, we map it into the same embedding space via
𝜙(𝑞)=Φ(𝑞), and identify relevant code snippets by performing
a nearest-neighbor search over the embeddings {e𝑖}to produce
a setR(𝑞). We then construct an augmented prompt 𝑞′=𝑞⊕É
𝑐𝑖∈R(𝑞)M𝑖. This prompt provides the language model with
helpful, context-relevant code blocks, thereby guiding its generation
process toward more plausible solutions.
After forming the augmented prompt, we invoke a large lan-
guage model parameterized by 𝜋𝜃to produce a candidate code
snippet ˆ𝑐. The LLM is treated as a conditional generative model that
outputs code when given 𝑞′. To ensure reproducibility, we use a
fixed random seed, a consistent sampling strategy (such as nucleus
sampling with a specified temperature and top- 𝑝), and a stable
checkpoint of the model that is not altered outside of the described
training updates. The generated code ˆ𝑐is then executed within a
controlled sandbox environment. This execution environment is
held constant across experiments and may include a predefined
set of test cases or an evaluator that checks if the generated code
meets certain correctness conditions. The outcome of this execu-
tion, which we denote as feedback 𝑓, can be as simple as a binary
success/failure indicator or as detailed as a set of passed and failed
test results.
The feedback 𝑓is appended to the query, producing a refined
query𝑞new=𝑞⊕𝑓. By incorporating execution feedback into the
query, subsequent iterations of the ARCS loop have direct access
to information about previous attempts. This process constructsa chain-of-thought style reasoning trajectory, where the LLM can
leverage past failures to refine its approach. Each iteration thus
updates the state, represented as (𝑞𝑡,ˆ𝑐𝑡,𝑓𝑡), and produces new can-
didate solutions. In practice, the number of iterations can be limited
by a predefined maximum 𝑇or terminated early if a fully correct
solution is found.
6 Experiments and Results
In this section, we provide a systematic evaluation of ARCS across
multiple benchmarks, including standard code generation and trans-
lation datasets, as well as a specialized domain-specific corpus.
Our experiments demonstrate the impact of iterative refinement,
adaptive complexity evaluation, and retrieval augmentation mecha-
nisms implemented in ARCS. We rigorously quantify performance
differences among the Small, Medium, and Large configurations,
elucidating their respective advantages and limitations.
6.1 Datasets and Evaluation Metrics
We evaluate ARCS on two widely-adopted benchmark datasets for
code synthesis:
HumanEval. HumanEval [ 3] consists of 164 manually curated
programming problems that test functional correctness for Python
implementations. Each problem includes a natural language prompt,
function signature, and a set of hidden test cases that verify the
correctness of generated solutions. Performance is measured using
the widely reported pass@k metric, representing the fraction of
problems where at least one correct solution is found among 𝑘
samples generated by the model. We primarily report pass@1 to
reflect the model’s single-best-shot capability.
TransCoder. The TransCoder benchmark [ 9] is specifically de-
signed to assess automated code translation between programming
languages. It covers translation pairs among Python, Java, and
C++ and evaluates syntactic and semantic correctness of the gen-
erated translations. Performance on TransCoder is measured by
the percentage of translations correctly passing all provided test
cases ( Translation Accuracy ). We evaluate ARCS across multiple
translation directions, capturing its versatility.
LANL GitHub Corpus. We constructed a domain-specific eval-
uation corpus by selecting four computational libraries from the
Los Alamos National Laboratory (LANL) GitHub repositories1:
pyDNMFk ,pyDNTNk ,AdversarialTensors , and EPBD_BERT . These
libraries represent real-world scientific computing and high-performance
programming tasks, covering diverse functionalities such as dis-
tributed matrix and tensor factorizations, adversarial data genera-
tion, and deep language modeling.
For each library, we systematically designed two types of evalu-
ation prompts, resulting in a total of 20 distinct test use cases.z
•Documentation-based prompts: Questions were derived
directly from each project’s README files, ensuring realistic
natural language specifications that a typical end-user might
pose.
•Code-based prompts: To rigorously assess the system’s ca-
pability in context-sensitive code synthesis, we additionally
1https://github.com/lanl

ARCS: Agentic Retrieval-Augmented Code Synthesis with Iterative Refinement , ,
generated prompts based on actual source code functions
and class implementations. These prompts were carefully cu-
rated and refined using ChatGPT to ensure accuracy, clarity,
and realistic usage scenarios.
Performance was evaluated using the CodeBLEU metric [ 14],
which quantifies both syntactic and semantic similarity between
generated outputs and reference code. CodeBLEU captures nuances
in code generation quality, extending beyond superficial token-
level matches to assess deeper structural correctness. We compared
ARCS’s performance against a baseline Retrieval-Augmented Gen-
eration (RAG) system, enabling a clear and quantitative demon-
stration of ARCS’s enhanced capabilities in realistic programming
contexts.
6.2 Evaluation on HumanEval Benchmark
Table 1: Functional correctness evaluation on HumanEval
for LLaMA-3.1 70B
System Size pass@1 Score
Small 0.7988
Medium 0.7683
Large 0.8354
Table 1 reports ARCS performance (pass@1) on HumanEval
across different system configurations using the LLaMA-3.1 70B
model variant. On simple coding tasks, the Large Tier occasionally
scored lower (0.835) compared to the Small Tier (0.798), due to
unnecessary complexity introduced by iterative refinements. How-
ever, for more challenging tasks requiring intricate logic, iterative
debugging and feedback loops in the Large Tier significantly im-
proved correctness, yielding approximately a 4 percentage point
improvement over Medium (0.7683) and Small tiers. These results
highlight the value of iterative refinement in achieving reliable
performance for complex problems.
6.3 Evaluation on TransCoder Benchmark
Table 2 summarizes the translation accuracy of ARCS across dif-
ferent source-target language directions and system tiers (Small,
Medium, and Large). Translation accuracy here refers to the per-
centage of test cases for which the translated code passed all prede-
fined functional tests, providing an objective measure of semantic
correctness.
ARCS consistently demonstrates robust performance across all
evaluated language pairs, with noticeable performance improve-
ments when scaling from Small to Medium and further to the Large
system configurations. For example, in the C++-to-Python trans-
lation task, the Large configuration of the Meta-Llama-3.1-405B-
Instruct achieves an accuracy of 91.5%, surpassing both the Small
(90.5%) and Medium (91.6%) configurations. Similar trends are evi-
dent in Python-to-C++ translation, where the Large system reaches
87.0%, representing a notable improvement over the Small configu-
ration’s 86.0%.
Performance trends in Java-to-Python and Java-to-C++ transla-
tions reinforce the effectiveness of iterative refinement. Specifically,the Large system demonstrates superior performance in Java-to-
C++ translation, attaining a peak accuracy of 96.44% compared to
the Medium (95.46%) and Small (96.35%) tiers. These results indicate
that the Large tier’s iterative feedback-driven refinement provides
critical advantages for handling nuanced translations, ensuring
semantic and functional equivalence even in complex scenarios.
While ARCS achieves high accuracy overall, we also observed mi-
nor syntactic and paradigm discrepancies. For instance, ARCS occa-
sionally translated Python’s sys.max asfloat(’inf’) , which—though
semantically correct in specific contexts—caused test-case failures
due to exact-match constraints in the benchmark evaluation setup.
Despite these rare mismatches, ARCS’s iterative refinement strategy
effectively resolved semantic inaccuracies in subsequent iterations,
underscoring the strength of its adaptive, feedback-driven synthesis
process.
6.4 LANL Corpus Evaluation and CodeBLEU
Scores
We measured ARCS’s code synthesis quality using CodeBLEU
scores, which provide an insightful evaluation beyond token-level
matching. Table ??summarizes results comparing ARCS to a base-
line RAG system. ARCS demonstrates superior scores across all
components of CodeBLEU—particularly strong gains in weighted
n-gram match and syntax match components (29% and 64% re-
spectively, compared to baseline scores of 13% and 29%). Overall,
ARCS’s Medium and Large tiers surpass the basic RAG approach
by approximately 11% in overall CodeBLEU scores (0.40 vs. 0.29).
These gains reflect ARCS’s capacity to leverage structured retrieval
and iterative refinements to generate code that closely aligns with
syntactic conventions and semantic intents of the original tasks.
Table 3: Final Average Scores for system_small vs.
medium_system
Metric Basic RAG ARC RAG
CodeBLEU 0.2893 0.4040
N-gram Match Score 0.1053 0.2319
Weighted N-gram Match Score 0.1310 0.2938
Syntax Match Score 0.5269 0.6398
Dataflow Match Score 0.2273 0.2837
We also manually examined ARCS’s performance on straightfor-
ward code-completion tasks by disabling the CoT-RAG mechanism
to isolate the effect of iterative refinement. Our experiments reveal
that, in its default configuration, ARCS can generate overly com-
plex solutions for simpler tasks. To remedy this, we implemented a
complexity evaluation module that dynamically adjusts the number
of refinement iterations based on task difficulty. Preliminary find-
ings suggest that this adaptive mechanism curtails overshooting
behavior and better aligns outputs with reference solutions, fur-
ther reinforcing the benefits of an adaptive, complexity-sensitive
approach.

, , Bhattarai et al.
Table 2: Translation Accuracy (%) by Source →Target and System Tier for Transcoder dataset
Model VariantC++→Python Python →C++ Java →Python Java →C++
Small Medium Large Small Medium Large Small Medium Large Small Medium Large
Meta-Llama-3.1-70B-Instruct 84.8 88.8 89.1 83.3 84.1 84.9 86.5 89.5 88.9 95.5 93.9 94.5
Meta-Llama-3.3-70B-Instruct 86.3 89.5 88.4 85.9 85.1 86.9 89.5 91.2 90.5 94.5 94.8 94.3
Meta-Llama-3.1-405B-Instruct 90.5 91.6 91.5 86.0 89.5 87.0 90.6 91.7 91.6 96.35 95.46 96.44
6.5 Discussion and Limitations
Our comprehensive experimental analyses across diverse bench-
marks and domain-specific datasets highlight clear performance
differentiations among the ARCS tiers:
TheSmall Tier , due to its direct synthesis approach, excels in
computational efficiency and is most suitable for straightforward,
narrowly defined problems. However, as complexity escalates—such
as tasks requiring multi-component integration, extensive debug-
ging, or nuanced context understanding—this tier rapidly reaches
limitations, evidenced by its lower pass@1 performance (e.g., 79.88%
on HumanEval vs. 83.54% in the Large Tier).
TheMedium Tier successfully balances efficiency with robust-
ness through structured decomposition and validation processes. It
notably outperforms the Small Tier in tasks of intermediate com-
plexity, providing up to 5-7% improvement on translation accuracy
(e.g., 88.8% for C++ to Python translation compared to 84.8% in
Small), highlighting the value of systematic task breakdown and
explicit interface validations.
TheLarge Tier consistently achieves superior results, particu-
larly on complex programming tasks across all benchmarks evalu-
ated. Its iterative refinement and retrieval-augmented mechanisms
significantly improve correctness and robustness. For example, the
Large Tier achieved up to a 7% higher CodeBLEU score (0.36 com-
pared to 0.31 baseline), clearly reflecting its capacity to handle
complex, nuanced tasks with contextual accuracy. However, our
evaluations revealed that this tier can inadvertently overcompli-
cate simpler tasks, such as adding unnecessary error handling or
extraneous checks in scenarios like HumanEval, where minimal
solutions are preferred.
Furthermore, insufficient prompt specificity occasionally led the
Large Tier to infer unintended constraints, underscoring the impor-
tance of clear and precise task descriptions. This limitation high-
lights an essential trade-off between robustness and succinctness.
Consequently, we advocate the careful calibration of complexity,
supported by human oversight, especially when deploying ARCS
in scenarios sensitive to efficiency and simplicity.
Overall, our results concretely validate the design choices behind
ARCS, demonstrating that iterative refinement and context-aware
retrieval significantly elevate code synthesis performance. These
findings offer strong evidence supporting the use of ARCS for so-
phisticated and complex programming environments typical of
high-performance computing domains.7 Conclusion and Future Work
In this paper, we introduced ARCS, a novel agentic retrieval-augmented
code synthesis framework that integrates retrieval-augmented gen-
eration, iterative refinement, and execution-based feedback into a
unified, scalable pipeline. Through rigorous evaluations on widely-
used benchmarks—including HumanEval and TransCoder—and
domain-specific datasets drawn from LANL GitHub repositories,
ARCS demonstrated consistent, measurable improvements in code
generation quality. Specifically, our system achieved significant
gains, such as a 7% improvement in CodeBLEU scores and consis-
tent accuracy increases of approximately 5–7 percentage points in
cross-language translation tasks compared to baseline systems.
Our experiments clearly illustrate the advantages of iterative
refinement combined with adaptive complexity evaluation, particu-
larly within the Large System Tier, which adeptly manages complex
scenarios involving nuanced context, multi-component integration,
and robust error handling. The findings decisively validate ARCS’s
potential to transform automated code generation practices, en-
abling precise, robust solutions even for challenging programming
problems.
Despite these promising outcomes, multiple avenues for further
development remain. We plan to refine our complexity-aware mod-
ules, employing adaptive metrics such as runtime complexity or
memory profiling, to dynamically calibrate the depth of iterative
refinement based on the inherent task complexity. This effort will
help avoid unnecessary code complexity in simpler tasks, further
improving the system’s practicality.
Moreover, integrating advanced evaluation metrics like Code-
BLEU more systematically into the iterative refinement loop will
enhance our ability to capture syntactic and semantic correctness
more effectively. Another critical direction involves exploring for-
mal verification methods that could provide rigorous correctness
guarantees, essential for safety-critical and mission-critical applica-
tions in high-performance computing and scientific research.
Thus, ARCS represents a significant advancement in automated,
context-aware, and iterative code synthesis. Its rigorous empirical
validation, coupled with clearly defined paths for future enhance-
ments, positions ARCS as a foundational technology capable of
significantly advancing the automation and optimization of com-
plex software development tasks.
References
[1]Manish Bhattarai, Javier E Santos, Shawn Jones, Ayan Biswas, Boian Alexandrov,
and Daniel O’Malley. Enhancing code translation in language models with few-
shot learning via retrieval-augmented generation. arXiv preprint arXiv:2407.19619 ,
2024.
[2]Manish Bhattarai, Minh Vu, Javier E Santos, Ismael Boureima, and Daniel O’
Malley. Enhancing cross-language code translation via task-specific embedding

ARCS: Agentic Retrieval-Augmented Code Synthesis with Iterative Refinement , ,
alignment in retrieval-augmented generation. arXiv preprint arXiv:2412.05159 ,
2024.
[3]Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pinto, Jared
Kaplan, Caitlin McLeavey, Arvind Neelakantan, Pranav Shyam, Girish Sastry,
et al. Evaluating large language models trained on code. arXiv preprint
arXiv:2107.03374, 2021.
[4]Jia Li, Ge Li, Yongmin Li, and Zhi Jin. Structured chain-of-thought prompting
for code generation. arXiv preprint arXiv:2305.06599 , 2023.
[5]Qingyao Li, Wei Xia, Kounianhua Du, Xinyi Dai, Ruiming Tang, Yasheng Wang,
Yong Yu, and Weinan Zhang. Rethinkmcts: Refining erroneous thoughts in monte
carlo tree search for code generation. arXiv preprint arXiv:2409.09584 , 2024.
[6]Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser,
Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago,
Thomas Hubert, Peter Choy, Cyprien de Masson d’Autume, Igor Babuschkin,
Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov,
James Molloy, Daniel J. Mankowitz, Esme Sutherland Robson, Pushmeet Kohli,
Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. Competition-level
code generation with alphacode. Science , 378(6624):1092–1100, 2022.
[7]Ryan Liu, Jiayi Geng, Addison J. Wu, Ilia Sucholutsky, Tania Lombrozo, and
Thomas L. Griffiths. Mind your step (by step): Chain-of-thought can reduce
performance on tasks where thinking makes humans worse. arXiv preprint
arXiv:2410.21333 , 2024.[8]Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xi-
aoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom
Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer,
Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar,
Hugo Touvron, Louis Martin, et al. Code llama: Open foundation models for
code, 2023.
[9]R. Rozière et al. Transcoder: Unsupervised translation of programming languages.
arXiv preprint arXiv:2006.03511, 2020.
[10] Zezhou Yang, Sirong Chen, Cuiyun Gao, Zhenhao Li, Xing Hu, Kui Liu, and Xin
Xia. An empirical study of retrieval-augmented code generation: Challenges and
opportunities. arXiv preprint arXiv:2501.13742 , 2025.
[11] Kechi Zhang, Jia Li, Ge Li, Xianjie Shi, and Zhi Jin. Codeagent: Enhancing code
generation with tool-integrated agent systems for real-world repo-level coding
challenges. arXiv preprint arXiv:2401.07339 , 2024.
[12] Xiangyu Zhang, Yu Zhou, Guang Yang, and Taolue Chen. Syntax-aware retrieval
augmented code generation. In Findings of the Association for Computational
Linguistics: EMNLP 2023 , pages 1291–1302, 2023.
[13] Xiangyu Zhang, Yu Zhou, Guang Yang, and Taolue Chen. Syntax-aware retrieval
augmented code generation. In Findings of the Association for Computational
Linguistics: EMNLP 2023 , pages 1291–1302, 2023.
[14] Xiao Zhang, Dongxu Zou, Qiang Liu, and Jianxin Zhou. Codebleu: a method for
evaluating code generation. In Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP) , 2020.
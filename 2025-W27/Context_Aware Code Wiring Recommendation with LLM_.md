# Context-Aware Code Wiring Recommendation with LLM-based Agent

**Authors**: Taiming Wang, Yanjie Jiang, Chunhao Dong, Yuxia Zhang, Hui Liu

**Published**: 2025-07-02 03:00:23

**PDF URL**: [http://arxiv.org/pdf/2507.01315v1](http://arxiv.org/pdf/2507.01315v1)

## Abstract
Copy-paste-modify is a widespread and pragmatic practice in software
development, where developers adapt reused code snippets, sourced from
platforms such as Stack Overflow, GitHub, or LLM outputs, into their local
codebase. A critical yet underexplored aspect of this adaptation is code
wiring, which involves substituting unresolved variables in the pasted code
with suitable ones from the surrounding context. Existing solutions either rely
on heuristic rules or historical templates, often failing to effectively
utilize contextual information, despite studies showing that over half of
adaptation cases are context-dependent. In this paper, we introduce WIRL, an
LLM-based agent for code wiring framed as a Retrieval-Augmented Generation
(RAG) infilling task. WIRL combines an LLM, a customized toolkit, and an
orchestration module to identify unresolved variables, retrieve context, and
perform context-aware substitutions. To balance efficiency and autonomy, the
agent adopts a mixed strategy: deterministic rule-based steps for common
patterns, and a state-machine-guided decision process for intelligent
exploration. We evaluate WIRL on a carefully curated, high-quality dataset
consisting of real-world code adaptation scenarios. Our approach achieves an
exact match precision of 91.7% and a recall of 90.0%, outperforming advanced
LLMs by 22.6 and 13.7 percentage points in precision and recall, respectively,
and surpassing IntelliJ IDEA by 54.3 and 49.9 percentage points. These results
underscore its practical utility, particularly in contexts with complex
variable dependencies or multiple unresolved variables. We believe WIRL paves
the way for more intelligent and context-aware developer assistance in modern
IDEs.

## Full Text


<!-- PDF content starts -->

arXiv:2507.01315v1  [cs.SE]  2 Jul 20251
Context-Aware Code Wiring Recommendation with
LLM-based Agent
Taiming Wang∗, Yanjie Jiang†, Chunhao Dong∗, Yuxia Zhang∗, and Hui Liu∗
∗School of Computer Science & Technology, Beijing Institute of Technology, Beijing, China, Email:
wangtaiming@bit.edu.cn, dongchunhao22@bit.edu.cn, yuxiazh@bit.edu.cn, liuhui08@bit.edu.cn
†School of Computer Science & Technology, Tianjin University, Tianjin, China, Email: 2990094974@qq.com
Abstract —Copy-paste-modify is a widespread and pragmatic
practice in software development, where developers adapt reused
code snippets—sourced from platforms such as Stack Overflow,
GitHub, or LLM outputs—into their local codebase. A critical
yet underexplored aspect of this adaptation is code wiring , which
involves substituting unresolved variables in the pasted code with
suitable ones from the surrounding context. Existing solutions
either rely on heuristic rules or historical templates, often
failing to effectively utilize contextual information, despite studies
showing that over half of adaptation cases are context-dependent.
In this paper, we introduce WIRL , an LLM-based agent for
code wiring framed as a Retrieval-Augmented Generation (RAG)
infilling task. WIRL combines an LLM, a customized toolkit, and
an orchestration module to identify unresolved variables, retrieve
context, and perform context-aware substitutions. To balance
efficiency and autonomy, the agent adopts a mixed strategy:
deterministic rule-based steps for common patterns, and a state-
machine-guided decision process for intelligent exploration. We
evaluate WIRL on a carefully curated, high-quality dataset
consisting of real-world code adaptation scenarios. Our approach
achieves an exact match precision of 91.7% and a recall of
90.0%, outperforming advanced LLMs by 22.6 and 13.7 percent-
age points in precision and recall, respectively, and surpassing
IntelliJ IDEA by 54.3 and 49.9 percentage points. These results
underscore its practical utility, particularly in contexts with
complex variable dependencies or multiple unresolved variables.
We believe WIRL paves the way for more intelligent and context-
aware developer assistance in modern IDEs.
Index Terms —Copy-paste-modify Practices, Code Reuse, Code
Adaptation, Large Language Models, Agent.
I. I NTRODUCTION
Copy-paste-modify is a common and inevitable practice dur-
ing development. The developers frequently reuse code snip-
pets from online programming Q&A communities, e.g., Stack
Overflow [16], open-source repositories, e.g., GitHub [15],
or code generation of LLMs. However, in most cases the
reused code snippets need additional adaptations to integrate
to the local codebase. As reported, more than 85% code
snippets from Stack Overflow need to be adapted (modified)
before integration into the local code [46], [47]. Variable
identifiers are usually under adaptation since undeclared or
conflict identifiers would lead to compilation errors or po-
tential bugs. Consequently, code wiring, replacing unresolved
variables with existing ones from the local context, is one of
the most prevalent forms of adaptation. This term aligns with
the analogy of electrical wiring, where components (variables
* Hui Liu is the corresponding author.in code) are intentionally connected to establish functionality.
An example of code wiring from Stack overflow [9] to
GitHub [10] is presented in Fig. 1. Automatic code wiring can
help developers get rid of the repetitive and trivial processes
during the adaptation and concentrate on the complex busi-
ness logic. However, existing approaches are designed based
on either simple heuristic rules [6] or templates extracted
from historical modifications [12], leaving the context not
effectively leveraged although 56.1% of the adaptation cases
are dependent on the surrounding context as reported by
Zhang et al. [46]. Full-parameter LLMs have demonstrated
promising performance; however, their high latency renders
them impractical for real-world applications. On the other
hand, distilled LLMs with smaller parameter sizes tend to
under perform in terms of accuracy and robustness.
To this end, we present WIRL , an LLM-based agent for
code Wiring through Infilling with RAG and LLMs. WIRL
consists of three core components: an LLM, a customized
toolkit, and an agent pilot. The customized toolkit offers three
primary functionalities: (1) identifying and locating unresolved
elements, (2) analyzing and collecting contextual information,
and (3) infilling and recommending suitable substitutions.
The agent pilot is responsible not only for initializing the
prompt but also for coordinating the interaction between the
LLM agent and the toolkit. It parses the LLM’s output,
updates the prompt dynamically, and invokes the appropriate
tools as needed. With a dynamically updated prompt, WIRL
incrementally gathers relevant context by invoking appropriate
tools until the agent determines that sufficient information
has been collected to make a final recommendation. To
fully leverage the capabilities of LLMs, we reformulate the
adaptation task as a retrieval-augmented generation (RAG)-
based infilling task for the unresolved elements—framing it
as a code completion problem, which aligns more naturally
with the strengths of LLMs. Empirical evidence from Zhang
et al. [30] supports this reformulation, showing that LLM
performance on code snippet adaptation tasks lags behind
their performance on code completion and generation tasks.
Moreover, by adopting the RAG-based infilling strategy, even
distilled LLMs with smaller parameter sizes can outperform
full-parameter LLMs, thereby meeting both the latency and
accuracy requirements of real-world software development
scenarios. To enhance the efficiency of WIRL , we adopt a
hybrid execution mode for the agent, wherein essential steps
are extracted and executed initially without invoking the LLM.

2
In addition to this optimization, we introduce a state machine
to guide the agent’s exploration process, thereby reducing
unnecessary or ineffective attempts and improving overall
execution efficiency.
To assess the efficiency of WIRL , we manually curated
a high-quality code wiring dataset derived from a publicly
available code adaptation dataset [14]. Evaluated on this
dataset, WIRL achieves an exact match precision of 91.7%
and a recall of 90.0%. Notably, it outperforms advanced LLM
baselines by 22.6 and 13.7 percentage points in precision and
recall, respectively, and exceeds the performance of IntelliJ
IDEA by 54.3 and 49.9 percentage points. These results
highlight the effectiveness and practical advantage of WIRL .
We also analyzed time efficiency and token consumption,
demonstrating that WIRL meets real-time requirements with
affordable computational cost.
The contributions of this paper are as follows:
•We propose WIRL , an LLM-based agent specifically de-
signed for context-aware code wiring recommendations.
•We develop a customized toolkit that provides essential
functionalities, enabling the agent to perform precise and
efficient code adaptation.
•We construct and release a high-quality dataset for code
wiring evaluation, curated and validated through careful
manual inspection.
The rest of this paper is structured as follows. Section II
motivates this study. Section III illustrates the design of WIRL .
Section IV presents experimental details and results. Section V
discusses the threats and limitations of our study. Section VI
discusses the related works, and Section VII concludes this
paper.
II. M OTIVATING EXAMPLE
To provide the intuition about how WIRL works and to
define the key terminologies, we begin with a motivating
example of a code wiring instance. Fig. 1 presents an example
of copy-paste-modify (specifically code wiring) practice where
a developer reused a piece of code snippet from Stack Over-
flow [9] and adapted it to the GitHub local repository [10].
The developer copied and pasted the line 6 and line 7 and then
modified them into line 8 and line 9. For the sake of clarity, in
this paper we call the code snippets in line 6 and line 7 isolated
code since it is isolated before integration. The code snippets
in line 8 and line 9 are called integrated code , i.e., code after
the integration. Specifically, in this example, local variable
“list” (highlighted in red) is undefined and leads to syntax
errors. The developers substituted it with the predefined field
variable “mTags” (highlighted in yellow and green) to resolve
the syntax errors. We call “list” anunresolved element . It
is worth noting that variables are not the only code entity
that need to be modified, literal values or a method invocation
expression could also be the target code entity. Consequently,
we use “element” instead of only “variable” .“mTags” is
called the context element , i.e., the code elements in the
context. Similarly, a context element could also be either a
variable (including local variables, parameters, and fields) or
an expression (e.g., method invocation).
private final SortedSet <Tag>    mTags ;
// http://stackoverflow.com/a/669165/1036813 March 17 2015 blainel
public String getTagsAsString() {
final StringBuilder sb = new StringBuilder();
String delim = "";
delim = ",";
}
return sb.toString();
}for (Item i :    List) {
sb.append(delim).append(i);--
for (Tag tag :    mTags) {
sb.append(delim).append(tag. getName ());++list
mTagsmTags 1
345672
89
10111213Useful ContextFig. 1. Motivating Example
Before illustrating how WIRL operates, we first present the
problem formulation, which serves as an important foundation
of our approach. Assume the set of unresolved elements is
denoted by U=uiand the set of context elements byC=ci.
The code wiring task can be formally defined as identifying an
injective mapping M:U → C , where each ui∈ U is mapped
to a distinct ci∈ C. In this work, however, we reformulate
code wiring as a retrieval-augmented generation (RAG)-based
infilling task. Specifically, each ui∈ U is replaced with a
placeholder token “ <Infill> ” in the isolated code . The
objective then becomes to infill these placeholders using the
retrieved context information, including the variable names
themselves, which is motivated by the observation that literal
similarity often provides valuable contextual cues. This refor-
mulation is grounded in the insight that code completion is
more naturally aligned with the capabilities of large language
models (LLMs) than explicit code adaptation. This design
choice is further supported by recent empirical findings by
Zhang et al. [30], which highlight the superior performance of
LLMs in code completion tasks relative to direct adaptation
scenarios.
In this example, WIRL first identifies the unresolved variable
“list” , and then proceeds to iteratively collect relevant con-
textual information such as available variables in the current
scope, semantic hints indicating that “list” should refer to a
collection object, and method names suggesting that “Tags” is
the intended target. Then it assesses the suitability of candidate
variables by invoking a set of external tools tailored for context
analysis. Once the context is deemed sufficient, WIRL conduct
infilling and “mTags” is filled as the appropriate variable
here. Eventually, WIRL automatically applies the modification
within the IDE for developers.
III. A PPROACH
A. Overview
The overview of WIRL is illustrated in Fig. 2. WIRL consists
of three core components: an LLM , a customized toolkit
containing three types of tools, i.e., locator ,collector , and
completer , and an agent pilot , which orchestrates communi-
cation between the LLM and the toolkit by parsing outputs,
updating prompts, and executing tool invocations. Given a
piece of isolated code , the agent pilot begins by initializing
the prompt ( Step 1 ). This prompt includes system roles, task

3
Customized Toolkit Agent Pilot
LLM
Locator
Collector
Completer②Prompt
Dynamically
③Action
&Args④Execute
Action
⑤Raw Tool
OutputParse Output
Isolated 
code
Integrated 
code
Update PromptExecute Tools①Initialization
Fig. 2. Overview of WIRL
descriptions, and instructions on how the task should be
addressed using the available tools. It also embeds the location
ofunresolved elements and results from preliminary analyses.
Using this prompt, the agent pilot issues a request to the
LLM ( Step 2 ). The LLM responds by specifying a tool to
invoke from the customized toolkit ( Step 3 ). The agent pilot
then parses the response and executes the corresponding tool
with the appropriate arguments ( Step 4 ). The tool’s output is
subsequently parsed and integrated into the prompt ( Step 5 ) for
the next iteration. This loop continues iteratively until either
thecompleter tool is executed, indicating task completion, or
a predefined computational budget is reached.
B. Dynamic Prompting
WIRL adopts a reasoning-and-acting framework (i.e., Re-
Act[45]), iterating until either the goal is achieved or the
computational budget is exhausted. In each iteration, the agent
pilot parses the output of the invoked tool and incorporates the
results into the prompt, which then serves as the input for the
subsequent iteration. The static and dynamic sections of the
prompt are described in detail below:
1) Role (Static): The static section defines the agent’s role
as a Senior Java Development Engineer and clearly articulates
the task objective: to perform variable-level code integration
based on the provided code context. Notably, this section
explicitly instructs the agent to make decisions autonomously,
without relying on user intervention at any point during the
process.
2) Goals (Static): We define three primary goals for the
agent to accomplish:
•Analyze context: Evaluate the current code context to
determine whether it provides sufficient information to
suggest a valid substitution.
•Collect more context: If the existing context is inad-
equate, invoke the appropriate tools from the provided
toolkit to gather additional relevant information.
•Suggest a substitution: Once adequate context is ob-
tained, leverage the agent’s completion tool to recom-
mend a suitable substitution for the unresolved element.
3) Guidelines (Static): We define a set of guidelines to
which the agent should adhere.•Limited budget: We emphasize that the budget is limited
and it is important to be efficient to select the next steps.
•Type should match: We inform the agent that the basic
principle is to make sure the retrieved context element is
type-compatible with unresolved element .
•Be careful to stretch: More context typically leads to
higher latency. Consequently, the agent is instructed to
complete its recommendation using the minimal neces-
sary context and to extend the context scope cautiously.
•Extend scope: We instruct the agent that in some cases,
no suitable substitution candidates may be found within
the immediate context. In such scenarios, the agent should
expand its search scope beyond the current class, as the
context element is more likely to be a method invocation
expression rather than a simple variable.
4) State Description (Dynamic): To guide the agent in
utilizing the customized toolkit in a meaningful and efficient
manner, we define a state machine comprising three states:
initial state ,insufficient context state , and sufficient context
state . This mechanism is introduced because, in our prelimi-
nary experiments, we observed that the agent often engaged in
aimless exploration when operating without explicit guidance.
Each state is associated with a set of available tools, as
described in Section III-C, and the state description section of
the prompt informs the agent of its current status. Specifically,
the agent begins in the initial state , where it identifies unre-
solved elements and performs a preliminary analysis of the
context. Based on this analysis and subsequent tool executions,
the agent transitions into the insufficient context state . In this
state, it iteratively invokes relevant tools to gather additional
contextual information until either the context is deemed
sufficient or the predefined resource budget is exhausted. Once
the agent reaches the sufficient context state , it invokes the
completion module (also implemented as an LLM invocation,
detailed in Section III-C) to make the final recommendation.
It is important to note that while the state machine offers
structured guidance, it does not enforce a fixed sequence of
tool invocations. The selection and ordering of tools are left
entirely to the agent’s discretion.
5) Available Tools (Dynamic): This section illustrates the
available tools (refer Section III-C) agent can call in each state.

4
6) Gathered Information (Dynamic): A key capability of
WIRL lies in its ability to gather contextual information
through tool execution, which forms the foundation for gener-
ating accurate recommendations. To ensure the agent remains
aware of previously executed tools and the context collected,
this section of the prompt records the reasoning thoughts,
the name of tool, its corresponding arguments (parsed from
the LLM’s output), along with the results returned by tool
invocations (parsed from the tools’ output).
7) Output Format (Static): To enable the agent pilot to
effectively parse the output, we require that all responses from
the LLM be structured in JSON format. Each JSON object
must contain three mandatory fields: “thought” ,“action” ,
and“action input” . The “thought” field captures the agent’s
reasoning and decision-making process based on the current
context. The “action” field specifies the name of the tool to
be invoked next, while the “action input” field provides the
necessary arguments for that tool, also formatted as a JSON
object.
C. Customized Toolkit
One of the key novelties of WIRL lies in its ability to
autonomously decide which tools to invoke based on the
current state. The toolkit provided for WIRL (as shown in
Table I) is carefully customized to facilitate the code wiring
task.
1) Identifying unresolved elements: A prerequisite for ef-
fective code wiring is the accurate identification of unresolved
elements . The locator tool ( identify unresolved elements )
leverages compiler information to detect unresolved variables
and their references, and it is executed as the initial step.
Since identifying and locating these elements is fundamental
to addressing the task, the agent pilot invokes the locator
automatically, without requiring input from the LLM.
2) Collecting Context Information: To minimize unnec-
essary LLM invocations and reduce iteration overhead,
we provide two tools, i.e., getavailable variables and
getunused variables , to help the agent efficiently construct
a list of candidate variables. The getavailable variables
tool parses the AST of the current Java file and collects
all accessible local variables, method parameters, and class
fields. To reflect realistic development scenarios, only local
variables declared before the unresolved element are consid-
ered valid candidates. Once available variables are gathered,
thegetunused variables tool performs data-flow analysis to
examine the usage of each variable. Variables with no detected
references are marked as unused. To determine the syntactic
role of an unresolved element , the tools isargument and
isreceiver inspect its AST context to identify whether it is
enclosed within a method invocation. If the element appears as
a method argument, its expected type can be inferred from the
corresponding formal parameter, and the parameter name itself
provides useful semantic cues. If the element is the receiver
of a method call, the tool returns the invoked method member
to support further analysis. The reserve type compatible ones
tool filters candidate variables by retaining only those whose
types are compatible with that of the unresolved element .When the unresolved element is identified as a method
receiver, the retrieve identical function call tool takes the
invoked method member as input and searches the current
file for instances of the same member invocation, which
may reveal contextually relevant patterns or variable usages.
Finally, if no valid candidates are found in the current scope,
thegetmethod names tool is invoked. Given a class name, it
returns the method members of that class that match the type
of the unresolved element , providing additional opportunities
for accurate substitution. The sort byliteral similarity tool is
invoked when the agent determines that lexical similarity is a
relevant factor. This tool computes the Levenshtein distance
between the unresolved element and each candidate variable,
ranking the candidates based on their literal similarity.
3) Infilling Isolated Code: Once the agent has col-
lected sufficient contextual information, it invokes the exe-
cution completion tool to infill the placeholders using the
gathered data. This tool is implemented via an LLM call,
where the prompt corresponds to the final version of the
dynamically updated prompt that incorporates all previously
collected context information.
D. Agent Pilot
The agent pilot orchestrating the communication between
LLM andcustomized toolkit , playing a essential role in WIRL .
Its responsibilities are outlined as follows:
1) Initializing Prompt: The agent pilot begins by
initializing the prompt with static components such as the
role definitions and task objectives. It then incorporates
the input code (as described in Section IV-E1) into
the prompt. Subsequently, the prompt is dynamically
updated with the locations of unresolved elements and
relevant contextual information, obtained by invoking
the Locator (identify_unresolved_elements )
and Collector tools ( get_available_variables ,
get_unused_variables , is_argument , and
is_receiver ). This initialization process substantially
reduces unnecessary LLM invocations and accelerates context
analysis, thereby enhancing the overall efficiency of WIRL .
2) Parsing Output: Since the outputs of LLMs are returned
in JSON format, they can not be directly used for tool execu-
tion. Accordingly, the second responsibility of the agent pilot
is to validate and process these outputs. If an LLM response
deviates from the expected structure, e.g., due to hallucinations
or formatting errors, the agent pilot must attempt to correct the
output or handle exceptions gracefully. In addition to output
parsing and error handling, the agent pilot also maintains a
memory of previously executed tools and their corresponding
arguments to prevent redundant operations.
3) Executing Tools: With the parsed tool names and cor-
responding arguments, the agent pilot invokes the designated
tools and returns their outputs. Notably, all tool executions are
performed within an isolated environment to ensure they do
not interfere with the host system or the internal functioning
ofWIRL .
4) Updating Prompt: Once the tool output is available, the
agent pilot updates all dynamic sections of the prompt in

5
TABLE I
AVAILABLE TOOLS FOR THE AGENT
Tools Type Applicable State Description
identify unresolved elements Locator Initialization Analyze compiler information to identify the unresolved elements .
getavailable variables Collector Initialization Conduct code analysis and get the available variables in the current context scope.
getunused variables Collector Initialization Conduct data flow analysis and get the unused variables in the current context.
isargument Collector Initialization Judge whether the unresolved element plays an argument in a method invocation.
isreceiver Collector Initialization Judge whether the unresolved element plays a receiver in a method invocation.
retrieve identical function call Collector Insufficient Context Retrieve variables that invoke the identical function calls with unresolved element .
reserve type compatible ones Collector Insufficient Context Only reserve the variables that are type-compatible as the unresolved element .
sort byliteral similarity Collector Insufficient Context Sort the available variables by their similarity with the unresolved element .
getmethod names Collector Insufficient Context Collect method members with same type as unresolved element in the given class.
execute completion Completer Sufficient Context Invoke LLM to complete the code snippets with the collected context information.
preparation for the next iteration. Specifically, it modifies the
current state and the list of available tools, and appends the
“thought” ,“action” ,“action input” , and “observation” (i.e.,
the tool’s output) to the accumulated context information.
IV. E VALUATION
A. Research Questions
•RQ1: How well does WIRL perform compared with
baselines?
•RQ2: How well does WIRL perform regarding time cost?
•RQ3: How well does WIRL perform regarding token
consumption and monetary cost?
RQ1 investigates the effectiveness of WIRL in comparison
to selected baseline approaches for automatic code wiring.
By addressing this question, we aim to understand how well
WIRL performs in real-world development scenarios, as the
evaluation leverages real data under realistic experimental
conditions. Additionally, this analysis helps identify specific
cases where WIRL outperforms the baselines. Gaining insights
into the strengths of WIRL can guide future improvements and
optimizations of the approach.
RQ2 examines the time efficiency of WIRL relative to
baseline methods, with a particular focus on whether WIRL
can meet the responsiveness requirements of integrated devel-
opment environments (IDEs) and developers. To answer this
question, we use the average time taken to complete a code
wiring task as the primary metric. This allows us to assess
whether WIRL delivers timely recommendations that support
fluid and productive software development workflows.
RQ3 assesses the token consumption and associated mone-
tary cost of WIRL . Specifically, we evaluate input/output token
usage and the total cost incurred per code wiring instance.
By comparing these metrics against those of the baseline
approaches, we aim to determine the cost-efficiency of WIRL
in delivering practical code wiring support at a reasonable
computational expense.
B. Dataset
To the best of our knowledge, there are no existing datasets
specifically collected for the evaluation of automatic code
wiring approaches. Recently, Zhang et al. [46] investigated
context-based code snippet adaptation and constructed a high-
quality dataset through a combination of automated processingand meticulous manual curation. The resulting dataset com-
prises 3,628 real-world code reuse cases, where code snippets
were copied from Stack Overflow and subsequently adapted
for use in GitHub projects. Their dataset was initially derived
from the latest version of SOTorrent [41] (version 2020-12-
31), available via Zenodo [17], and includes only those reuse
cases in which GitHub files contain explicit references to
Stack Overflow posts. Given the quality and relevance of this
dataset, we adopted it as the foundation for our evaluation
and performed additional filtering and manual curation. The
construction of the evaluation dataset used in this paper
involves the following steps:
•To prevent data leakage and potential overfitting, we first
excluded the 300 sampled instances used by Zhang et
al. [46], as the design of WIRL was partially inspired by
their empirical observations.
•From the remaining 3,328 reuse cases, we performed
a deduplication step to eliminate redundancy caused by
forked repositories. Subsequently, we constructed a map-
ping between GitHub repositories and the corresponding
Stack Overflow posts.
•AsWIRL depends on compiler feedback and AST binding
analysis, it requires the full project to be resolvable.
Therefore, we conducted a project-wise manual inspec-
tion of the remaining dataset. The mappings were sorted
in descending order based on the frequency of references
to each repository, prioritizing more frequently reused
code.
•Following this order, two authors independently reviewed
each code reuse case to determine whether it qualifies
as a code wiring instance. In cases of disagreement,
discussions were held until a consensus was reached.
Ultimately, we identified 100 code wiring cases involved
with 221 pairs of unresolved elements and context el-
ements from Stack Overflow to GitHub. For ease of
reference, we refer to this curated evaluation dataset as
CWEvaluation .
C. Selected Baselines
1) SOTA Approaches: We selected ExampleStack [28] as a
baseline because it represents the state-of-the-art in automatic
code wiring. ExampleStack is a Google Chrome extension
designed to assist developers in adapting and integrating online

6
code examples into their own code repositories. The authors
provide comprehensive reproduction instructions, which al-
lowed us to successfully set up and run ExampleStack in our
evaluation.
Besides that, we selected the widely adopted industrial IDE,
IntelliJ IDEA [6], as one of our baseline approaches. The
rename recommendation support in IDEA is both convenient
and practical, and it is also suitable to address the code wiring
problem.
2) LLMs: We select the following LLMs as our baselines:
Distilled tiny models, including GPT-4o-mini [2],Qwen2.5-
Coder-14B [3], and Qwen2.5-Coder-32B [4]. Full-parameter
models, including DeepSeek-V3 [1] and Qwen-max [5].
D. Metrics
We adopted the evaluation metrics used by Wang et al. [44]
to assess the performance of WIRL and the baselines. The
definitions are introduced in the following:
•#Total Cases: the number of cases involved in the eval-
uation, i.e., number of unresolved elements .
•#Recommendation (Rec): the number of cases where the
evaluated approaches make a recommendation for the
developers. Note that the number of recommendations
(#recommendation) may not always match the total num-
ber of cases (#total cases).
•#Exact Match (EM): the number of cases where the
recommended code elements are identical to the ground
truth (i.e., the existing names in context).
•EM Precision :the number of exact matches divided by
the number of recommendations, i.e.,
EM Precision =#Exact Match
#Recommendation(1)
•EM Recall :the number of exact matches divided by the
number of cases involved in the evaluation.
EM Recall =#Exact Match
#Total Cases(2)
E. Experimental Setup
1) Input and Output Format: To better simulate real-world
development scenarios, the input code format is configured as
follows:
•For the LLM baselines, the entire class containing the
isolated code is provided as input. The isolated code ,
representing the segment requiring adaptation, is explic-
itly marked using a pair of control tokens, < start >
and< end > , to delineate the adaptation region. The
complete prompt format used for the LLM baselines is
available in our public repository [8].
•To enhance efficiency, the input to WIRL consists solely
of the method declaration containing the isolated code .
WIRL is designed to operate with the minimal necessary
context and selectively expand the context scope when
needed.TABLE II
COMPARISON WITH BASELINES
#EM #Rec EMPrecision EMRecall
IDEA 79 189 41.8% 35.7%
ExampleStack 4 10 40.0% 1.8%
GPT-4o-mini 101 145 69.7% 45.7%
Qwen2.5-Coder-14B 110 157 70.1% 49.8%
Qwen2.5-Coder-32B 149 197 75.6% 67.4%
Qwen-max 124 159 78.0% 56.1%
DeepSeek-V3 129 171 75.4% 58.4%
WIRL - Locator 110 157 70.1% 49.8%
WIRL - Collector 169 190 88.9% 76.5%
WIRL - Completer 146 189 77.2% 66.1%
WIRL 199 217 91.7% 90.0%
•Furthermore, we assume that any code following the
isolated code is unavailable, reflecting the common top-
down coding pattern observed during software develop-
ment. In this scenario, only the code preceding the adap-
tation region, which within the same method declaration,
is accessible as context.
For the sake of clarity, we constrained the LLM baselines
to output only the unresolved elements along with their
corresponding context elements . To promote self-consistency
and mitigate output variability, we set the temperature to 0.
Furthermore, each evaluated approach was executed five times,
and the most frequently generated output was selected as the
final result, following the protocol outlined by Chen et al. [18].
As both IDEA and ExampleStack produce ranked lists of
candidate substitutions, we only considered their top-ranked
recommendation as the final answer to ensure a fair and
consistent comparison across all evaluated approaches.
2) Implementations: To implement WIRL , we built a plugin
for IntelliJ IDEA using the IntelliJ PlatForm Plugin SDK [7]
and the Langchain4J [13] framework to interface with LLM
APIs. While WIRL is designed to be compatible with any
advanced LLMs, in this study we use Qwen2.5-Coder-14B
as the backend model for two key reasons. First, if WIRL
demonstrates superior performance over state-of-the-art full-
parameter LLMs while relying on a smaller model, it further
highlights the strength of our design. Second, models with
fewer parameters typically respond faster [42], [43], making
them more suitable for real-time usage in development envi-
ronments.
To support code analysis and contextual understanding, we
leverage the Program Structure Interface (PSI) [11], a power-
ful abstraction for syntactic and semantic analysis, including
data flow inspection and code structure extraction.
In addition, WIRL employs an iterative strategy to perform
context-aware code wiring. To balance latency, computational
cost, and effectiveness, we empirically set the maximum
number of iterations at two.
F . RQ1: Outperform the SOTA
In Table II, the first column lists the evaluated approaches.
The second and third columns indicate, respectively, the
number of cases in which the recommended code elements

7
exactly match the ground truth and the number of cases
in which a recommendation is made by the approach. The
final two columns, EM Precision andEM Recall , report how
often the recommendations are correct and how many of the
correct substitutions are successfully identified, respectively
(see Section IV-D for detailed definitions). From Table II, we
make the following observations:
•WIRL outperforms all selected baselines by a significant
margin in both EM Recall andEM Precision . It produces
the highest number of recommendations (217) while also
achieving the highest number of exact matches (199). The
resulting EM Recall andEM Precision scores of 90.0%
and 91.7%, respectively, reflect its strong and reliable
performance in real-world development scenarios.
•Compared to IDEA ,WIRL achieves improvements of
54.3 percentage points in EM Recall and 49.9 percent-
age points in EM Precision . Against ExampleStack , it
shows even greater gains: 88.2 and 51.7 percentage
points in EM Recall andEM Precision , respectively. When
benchmarked against the advanced LLM model Qwen2.5-
Coder-32B ,WIRL still delivers notable improvements of
22.6 and 13.7 percentage points, further demonstrating its
superior effectiveness.
•WIRL also substantially improves upon the performance
of directly using a raw LLM ( Qwen2.5-Coder-14B ).
Without the academic design described in Section III, the
baseline LLM achieves only 49.8% EM Recall and 70.1%
EM Precision . With the integration of the external toolkit
and structured prompt design, WIRL improves EM Recall
andEM Precision by 40.2 and 21.6 percentage points,
respectively, highlighting the value of its architectural
innovations.
•The three types of tools in WIRL contribute differently
to its performance: locator tools have the greatest im-
pact, followed by completer tools, with collector tools
contributing the least. Locator tools contribute the most
because the collector and completer tools rely on the
location information provided by them. Removing the
locator tools will reduce the system to relying solely on
raw LLMs.
We further analyzed the reasons behind WIRL ’s superior per-
formance compared to baseline approaches, with a particular
focus on IDEA . During the evaluation of IDEA , we recorded
not only the top-ranked recommendation but also the complete
list of suggestions. Our quantitative and qualitative analysis
revealed that in 84.1% of the cases (159 out of 189), IDEA did
include the correct recommendation, but it was not ranked first.
Statistically, the correct answer appeared in the top position in
only 49.7% of those 159 cases. Notably, in some instances, the
recommendation list contained more than 100 items, requiring
developers to scroll extensively before locating the correct
option. These findings indicate that while IDEA ’s underlying
recommendation algorithm, originally designed for renaming
suggestions, can potentially address the code wiring problem,
it lacks an effective ranking mechanism to prioritize relevant
suggestions. This inefficiency increases the cognitive load and
selection time for developers. In contrast, WIRL generates asingle high-confidence recommendation for each unresolved
element, achieving a high EM Precision of 91.7%. This direct
substitution approach not only reduces the time and effort
required from developers but also allows them to focus more
on adapting business logic rather than sifting through lengthy
suggestion lists. Furthermore, IDEA failed to generate any
recommendation in 14.5% of the cases (32 out of 221).
These failures occurred when the unresolved elements were
expressions rather than variables or literals, i.e., cases that
exceed the intended scope of IDEA ’s renaming functionality.
In contrast, WIRL successfully handled such cases through its
use of static code analysis, showcasing its broader applicability
and robustness.
ForExampleStack , we conducted a manual analysis of the
input data and identified the key limitation behind its unsatis-
fied performance. The core rationale of ExampleStack depends
on a pre-collected database of historical modifications of code
snippets from Stack Overflow to GitHub. Consequently, its
effectiveness is heavily constrained by the presence or absence
of identical or similar historical modifications in the database.
This reliance significantly hampers its applicability to previ-
ously unseen cases, which are common in real-world devel-
opment scenarios. In fact, within the CWEvaluation , only 6
code wiring instances involving 18 unresolved elements had
similar modifications present in the original database provided
byExampleStack [12]. Across all test cases, ExampleStack
made 10 recommendations, only 4 of which were correct. In
contrast, WIRL leverages the reasoning and generation capabil-
ities of large language models (LLMs) to recommend context
elements dynamically and from scratch. This design enables
WIRL to operate independently of historical reuse data, thereby
overcoming the fundamental limitation of ExampleStack and
improving its generalization and effectiveness across diverse
code wiring scenarios.
Regarding the performance of raw LLMs, their relatively
lowEM Recall can be attributed to two main factors. First,
the length of the input class significantly impacts the ability
of LLMs to make accurate recommendations. When the class
size exceeds the model’s maximum token limit, the LLM
cannot process the full context, resulting in failure to generate
any meaningful output. For instance, the maximum token
limit for DeepSeek-V3 is 64K tokens. If a class exceeds
this size, DeepSeek-V3 becomes incapable of handling the
instance entirely. Second, raw LLMs sometimes fail to identify
all unresolved elements within the code. As illustrated in
Fig. 3, there are two unresolved elements, i.e., “list” and
“target” , with their corresponding context elements being
“listView” and “mCommentListPosition” . While Qwen2.5-
Coder-14B successfully identified “list” and recommended
“listView” , it failed to detect “target” , thus missing a key
aspect of the task. In contrast, WIRL , by incorporating com-
piler information, accurately identified both unresolved ele-
ments (including all six references to “list” ) and successfully
recommended the appropriate context elements, “listView”
and “mCommentListPosition” , by leveraging the reasoning
capability of LLMs.
The relatively low EM Precision of raw LLMs can be at-
tributed to their difficulty in analyzing long and complex con-

8
public void refreshBlocksForCommentStatus( CommentStatus) 
newStatus) {
}}}
Qwen2.5- Coder -14B: list->listView
CCWHelper : list->listView list ->listView target ->mCommentListPosition
list->listView list ->listView list ->listView list ->listViewint start = list.getFirstVisiblePosition();
for(int i=start, j= list.getLastVisiblePosition(); i<=j;i ++)
if(target== list.getItemAtPosition( i)){
View view = list.getChildAt (i-start);
list.getAdapter ().getView (i, view, list);-----
int firstPosition = listView.getFirstVisiblePosition();
int endPosition = listView.getLastVisiblePosition();
for(int i= firstPosition; i< endPosition; i++) {
if(mCommentListPosition == i) {
View view = listView.getChildAt (i-firstPosition);
listView.getAdapter ().getView (i, view, listView);++++++
Fig. 3. Example of Identification Failure for Raw LLM
import java.nio.charset.Charset ;
public static String[] readFileIntoStringArr (final String path) 
throws IOException {
……
}
Qwen2.5- Coder -14B: encoding ->lines
CCWHelper : encoding ->Charset.defaultCharset()List<String> lines = Files.readAllLines (Paths.get(path), encoding); -
final List<String> lines = Files.readAllLines (Paths.get(path), 
Charset.defaultCharset());+
+
Fig. 4. Example of Recommendation Failure for Raw LLM
texts to correctly identify mapping relationships, particularly
when the appropriate context element is not explicitly present
in the immediate code. As illustrated in Fig. 4, the unresolved
element is encoding , and its corresponding context element
is“Charset.defaultCharset()” . It is challenging for raw LLMs
to recommend “Charset.defaultCharset()” without the context
information that the “Charset” class provides a static method
“defaultCharset()” that returns an instance of “Charset” . In
contrast, WIRL addresses this limitation by leveraging its
toolkit. The initial analysis of the context ( isargument tool)
determines the type of encoding , which is “Charset” . Then,
it uses the getmethod names tool to retrieve methods in
the“Charset” class that return the same type. This process
successfully identifies “defaultCharset()” , enabling WIRL to
include it in the prompt for the final completion module,
which then makes the correct recommendation. Additionally,
raw LLMs occasionally generate substitutions beyond the un-
resolved elements, introducing unexpected recommendations
and increasing the number of false positives, which further
impacts EM Precision .
G. RQ2: Time Efficiency
In this research question, we examine the time efficiency
ofWIRL compared to other baseline approaches on the
CWEvaluation . The comparison results are summarized
in Table III, where the second column reports the average
execution time per code wiring instance. It is worth noting
that the reported time costs are averaged over five independent
runs to ensure robustness and consistency. From Table III, we
make the following observations:
•WIRL required only marginally more time (approximately
0.6 seconds) to generate a recommendation compared to
GPT-4o-mini  Qwen2.5-Coder-14B  Qwen2.5-Coder-32B  Qwen-max  DeepSeek-V3   WIRL
ApproachesTime Cost (ms)
0   10000   20000     30000Fig. 5. Time Cost Distribution
Qwen2.5-Coder-14B , and it demonstrated greater time
efficiency than all other evaluated LLMs except GPT-
4o-mini . This efficiency can be attributed to its use of a
relatively compact LLM (14B parameters), whereas other
models incur longer response times due to their signif-
icantly larger parameter sizes. For example, DeepSeek-
V3, with 671B parameters, exhibited the highest latency,
averaging 11.5 seconds per recommendation.
•Although IDEA andExampleStack achieved superior per-
formance in terms of time cost, they fall short in ensuring
recommendation accuracy. We argue that WIRL offers a
more practical trade-off between recommendation quality
and acceptable latency, making it well-suited for real-
world development workflows.
To gain deeper insights into the time cost distribution
ofWIRL and the baseline approaches, we visualized the
results using a violin plot, as shown in Fig. 5. The plot
reveals that WIRL is capable of handling the majority of
code wiring instances in under 5 seconds. Its median time
cost is 4,201.5ms, which is comparable to that of Qwen2.5-
Coder-14B (3,977.5ms), and substantially lower than that of
DeepSeek-V3 (7,983.5ms). This demonstrates WIRL ’s ability
to deliver high-quality recommendations with competitive la-
tency.
H. RQ3: Token Consumption and Monetary Cost
In this research question, we evaluate the token usage
and monetary cost of WIRL compared to other LLM-based
baselines on the CWEvaluation . The comparison results are
summarized in Table III. Columns three to five report the
average number of input tokens, output tokens, and total tokens
per code wiring instance, respectively, while the final column
shows the corresponding average monetary cost. All figures
are averaged over five independent experimental runs.
As shown in Table III, WIRL consumes slightly more
tokens than Qwen2.5-Coder-14B , with an average increase of
237.3 input tokens and 80.2 output tokens. This results in
a marginally higher monetary cost, i.e., approximately 0.001
CNY more per instance. Surprisingly, although GPT-4o-mini
incurs the lowest monetary cost, its performance is relatively
poor, as shown in Table II. The increase in token usage is
attributed to WIRL ’s iterative design, where it progressively
collects and integrates contextual information into the prompt
before generating a final recommendation. As the output from
one iteration is passed as input to the next, the cumulative

9
TABLE III
TIME, TOKEN CONSUMPTION ,AND MONETARY COST
Avg. Time Cost (ms) Avg. #Input Tokens Avg. #Output Tokens Avg. #Total Tokens Avg. Monetary Cost (CNY)
IDEA 3.7 - - - -
ExampleStack 729.1 - - - -
GPT-4o-mini 3,930.4 2,769.4 27.4 2,796.8 0.0031
Qwen2.5-Coder-14B 4,624.5 2,769.4 28.3 2,797.7 0.0057
Qwen2.5-Coder-32B 6,799.4 2,769.4 43.8 2,813.2 0.0058
Qwen-max 5,393.6 2,769.4 24.9 2,794.3 0.0069
DeepSeek-V3 11,489.9 2,769.4 22.0 2,791.4 0.0057
WIRL 5,255.6 3,006.7 108.5 3,115.2 0.0067
0 5000 15000 20000
GPT-4o-mini Qwen2.5-Coder-14B  Qwen2.5-Coder-32B Qwen-max DeepSeek-V3 WIRL
ApproachesToken Number  
10000
Fig. 6. Token Number Distribution
token count naturally exceeds that of the single-pass LLM
baselines.
To further examine the distribution of token consumption
forWIRL and the LLM baselines, we present a violin plot in
Fig. 6. The results indicate that WIRL handles the majority
of cases using fewer than 2,500 tokens, with a median token
count of 2,306.5 per code wiring instance, which is compara-
ble to that of Qwen2.5-Coder-14B having a median of 1,962.5
tokens. Additionally, both WIRL andQwen-max exhibit stable
token usage, with minimal outlier points. Notably, Qwen-
max shows consistent performance in both time and token
consumption, largely because it omits cases that exceed its
maximum token limit.
V. D ISCUSSION
A. Threats to Validity
A potential threat to external validity lies in the limited size
and diversity of the dataset used for evaluation. To mitigate this
risk, we leveraged a publicly available dataset and performed a
thorough manual inspection to ensure its quality. Additionally,
we have open-sourced both the dataset and the evaluation
results [8] to facilitate replication and further validation by
the research community.
A threat to construct validity arises from possible bias in the
manual identification of real-world code wiring instances. To
address this, both the first and second authors independently
inspected the data. The inter-rater agreement was high, with
a Cohen’s kappa coefficient of 0.81, indicating strong consis-
tency. Discrepancies were resolved through discussion until
consensus was reached.B. Limitations
One limitation of WIRL is that it recommends substitutions
forunresolved elements using only predefined variables avail-
able in the surrounding context. However, in some cases, the
necessary variables may not exist in the current context, and
declaring new variables becomes necessary. Since WIRL is
specifically designed to support context-aware code adaptation,
handling variable declarations falls outside its current scope.
We plan to extend WIRL to address such scenarios in future
work.
Another limitation is that WIRL currently supports only the
Java programming language and has been evaluated solely on
Java-based code. This is partly due to the limited availability of
cross-language datasets involving real-world code reuse, such
as from Stack Overflow to GitHub. In future research, it would
be valuable to explore how to generalize and adapt WIRL
to support copy-paste-modify practices in other programming
languages.
VI. R ELATED WORK
A. Code Snippet Adaptation
Numerous empirical studies and tools have been proposed
to tackle code snippet adaptation. Wu et al. [48] conducted
the first empirical investigation into how developers reuse
Stack Overflow code, identifying five reuse patterns: exact
copy, cosmetic modification, non-cosmetic modification (e.g.,
adding/removing statements or altering structure), idea conver-
sion (writing from scratch), and information use. Extending
this work, Chen et al. [47] examined how Stack Overflow
answers are reused in GitHub projects. Using a hybrid of
clone detection, keyword search, and manual inspection, they
identified genuine reuse cases and proposed 11 sub-types of
non-cosmetic modifications. They found that most developers
reused at least some code, frequently applying statement-level
or structural edits.
Zhang et al. [46] explored context-based adaptations
through interviews with 21 developers, highlighting their
prevalence and significance. They proposed four adaptation
patterns and later introduced an interactive prompt engineering
approach [30], involving executor and evaluator roles. Though
iterative, this approach is not agent-based, as it cannot dynam-
ically interact with surrounding context. Key differences from
WIRL include method-level (vs. block-level) adaptation and
distinct strategies for context acquisition.

10
Initial efforts to automate variable resolution include
CSNIPPEX [29] and NLP2Testable [27], which declare
default-valued variables to resolve missing references. While
these address immediate issues, they lack the capacity for dy-
namic context adaptation, making them unsuitable for complex
reuse scenarios. As Zhang et al. [46] noted, 56.1% of reuse
cases require context-based adaptation, underscoring the need
for more robust solutions like WIRL .
Some approaches leverage code clones and templates.
Wightman et al.’s SniptMatch [24], an Eclipse plugin, maps
natural language queries to snippets with variable placehold-
ers, which users must manually fill. However, its focus is on
preventing syntax errors, not enabling active adaptation. Zhang
et al. [28] proposed ExampleStack , a Chrome extension that
integrates online examples by leveraging past adaptations from
open-source projects. It highlights frequently modified regions
while preserving unchanged parts, but its dependency on
existing adaptations limits its generalization to unseen cases.
Lin et al. introduced CCDemon [25], which recommends edits
to pasted code based on local clones, contextual elements,
and naming conventions. They later proposed MICoDe [22],
an Eclipse plugin for summarizing reusable module-level
patterns. Similarly, EUKLAS [19] for JavaScript automatically
imports missing variables and functions from the local project.
While effective, these tools are constrained to local contexts
and struggle with adaptation across different projects or unseen
scenarios.
Another line of work treats adaptation as a deobfuscation
task [20], [21], masking variables and recovering them via
context-aware models pre-trained with dataflow anonymization
objectives. Though conceptually aligned, these methods are
computationally inefficient for scenarios involving only a few
unresolved variables and are typically limited to single-file
contexts, making them inadequate for broader project-level
adaptation.
Other approaches support adaptation in alternative scenar-
ios.JigSaw [26] evaluates the compatibility of inserted code
via AST analysis to assist in integrating missing functionality.
APIzation [23] helps convert code snippets into well-defined
method declarations by statically analyzing parameters and
return values. While related, these methods target different
application goals than WIRL .
In contrast, WIRL enables automatic snippet adaptation
with a particular emphasis on resolving unresolved variables
through dynamic context analysis and LLM-powered recom-
mendations.
B. LLM-based Agents for Software Engineering
LLM-based agents have become transformative tools in au-
tomating software engineering tasks through iterative reason-
ing and code synthesis [34], [37]. Recent work highlights their
versatility: Huang et al. [31] introduced AgentCoder , a multi-
agent system for code generation via dynamic testing; Lin et
al. [33] modeled software processes (e.g., Scrum) by assigning
LLMs to development roles (e.g., architect, tester); Zhang et al.
[39] and Yang et al. [38] developed agents for test generation
using code navigation toolchains. In code maintenance, Batoleet al.’s LocalizeAgent [35] recommends refactorings, while
Bouzenia et al.’s RepairAgent [40] performs program repair
via dynamic prompt engineering. Tao et al. [36] automated
GitHub issue resolution with role-specialized agents, and Ke
et al. [32] applied LLM agents to flaky test repair.
Despite these advancements, no prior work addresses
context-aware code wiring, specifically, automatically resolv-
ing unresolved variables by substituting them with existing
ones. WIRL fills this gap as the first framework to integrate
agent-based planning, customized toolkit, and dynamically
updated prompts for automated identifier harmonization.
VII. C ONCLUSIONS AND FUTURE WORK
In this paper, we introduced WIRL , a novel LLM-based
agent designed to facilitate context-aware code wiring, an
essential but underexplored aspect of the copy-paste-modify
paradigm in software development. Unlike prior approaches
that rely on static heuristics or historical templates, WIRL
keeps a dynamically updated prompt to collect and record
useful context information through a hybrid architecture that
combines structured tool invocation with autonomous reason-
ing capabilities of LLMs. By formulating the adaptation task
as an RAG infilling problem, WIRL aligns closely with the
natural strengths of LLMs in code completion. Our evalua-
tion on a carefully curated benchmark dataset demonstrates
thatWIRL achieves state-of-the-art performance, significantly
outperforming both commercial IDEs and advanced LLMs in
exact match precision and recall. These results underscore
the practical value of intelligent, context-sensitive tooling for
reducing developer effort and improving the reliability of
reused code.
Looking ahead, we envision WIRL as a stepping stone
toward more general-purpose adaptation agents that can sup-
port a broader range of integration and transformation tasks.
We believe this direction holds strong potential for reshaping
how developers interact with reused and AI-generated code in
modern software engineering workflows.
VIII. D ATA AVAILABILITY
The replication package, including tools and data, is pub-
licly available in [8].

11
REFERENCES
[1] “Deepseek-v3,” 2025. [Online]. Available: https://bailian.console.aliyun.
com/?tab=model#/model-market/detail/deepseek-v3
[2] “Gpt-4o-mini,” 2025. [Online]. Available: https://platform.openai.com/
docs/models/gpt-4o-mini
[3] “Qwen2.5-coder-14b,” 2025. [Online]. Available:
https://bailian.console.aliyun.com/?tab=model#/model-market/detail/
qwen2.5-coder-14b-instruct
[4] “Qwen2.5-coder-32b,” 2025. [Online]. Available:
https://bailian.console.aliyun.com/?tab=model#/model-market/detail/
qwen2.5-coder-32b-instruct
[5] “Qwen-max,” 2025. [Online]. Available: https://bailian.console.
aliyun.com/?tab=model#/model-market/detail/qwen-max?modelGroup=
qwen-max
[6] “Intellij idea,” 2025. [Online]. Available: https://www.jetbrains.com/
idea/
[7] “Intellij platform plugin sdk,” 2025. [Online]. Available: https:
//plugins.jetbrains.com/docs/intellij/welcome.html
[8] “Wirl,” 2025. [Online]. Available: https://github.com/
AnonymousAccount4SEConference/WIRL
[9] “Stack overflow url for motivating example,” 2025.
[Online]. Available: https://stackoverflow.com/questions/668952/
the-simplest-way-to-comma-delimit-a-list/669165#669165
[10] “Github url for motivating example,” 2025. [Online]. Avail-
able: https://github.com/CMPUT301W15T10/301-Project/blob/master/
src/com/cmput301/cs/project/models/Claim.java
[11] “Program structure interface,” 2025. [Online]. Available: https:
//plugins.jetbrains.com/docs/intellij/psi.html
[12] “Examplestack-artifact,” 2025. [Online]. Available: https://github.com/
tianyi-zhang/ExampleStack-ICSE-Artifact
[13] “Langchain4j,” 2025. [Online]. Available: https://docs.langchain4j.dev/
intro/
[14] “Code adaptation dataset,” 2025. [Online]. Available: https://figshare.
com/s/741503454b274cd85094
[15] “Github,” 2025. [Online]. Available: https://github.com
[16] “Stackoverflow,” 2025. [Online]. Available: https://stackoverflow.com
[17] “Zenodo,” 2025. [Online]. Available: https://zenodo.org/
[18] D. Z. Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H.
Chi, Sharan Narang, Aakanksha Chowdhery, “SELF-CONSISTENCY
IMPROVES CHAIN OF THOUGHT REASONING IN LANGUAGE
MODELS,” ICLR , vol. 98004, pp. 1–18, 2023.
[19] C. D ¨orner, A. R. Faulring, and B. A. Myers, “EUKLAS: Supporting
copy-and-paste strategies for integrating example code,” in PLATEAU
2014 - Proceedings of the 2014 ACM SIGPLAN Workshop on Evaluation
and Usability of Programming Languages and Tools, Part of SPLASH
2014 , 2014, pp. 13–20.
[20] M. Allamanis and M. Brockschmidt, “Smartpaste: Learning to adapt
source code,” 2017. [Online]. Available: https://arxiv.org/abs/1705.07867
[21] X. Liu, J. Jang, N. Sundaresan, M. Allamanis, and A. Svyatkovskiy,
“AdaptivePaste: Intelligent Copy-Paste in IDE,” 2023, pp. 1844–1854.
[22] Y . Lin, G. Meng, Y . Xue, Z. Xing, J. Sun, X. Peng, Y . Liu, W. Zhao,
and J. Dong, “Mining implicit design templates for actionable code
reuse,” in ASE 2017 - Proceedings of the 32nd IEEE/ACM International
Conference on Automated Software Engineering , 2017, pp. 394–404.
[23] V . Terragni and P. Salza, “APIzation: Generating Reusable APIs from
StackOverflow Code Snippets,” in Proceedings - 2021 36th IEEE/ACM
International Conference on Automated Software Engineering, ASE
2021 . IEEE, 2021, pp. 542–554.
[24] D. Wightman, Z. Ye, J. Brandt, and R. Vertegaal, “Snipmatch: using
source code context to enhance snippet retrieval and parameterization,”
inProceedings of the 25th Annual ACM Symposium on User Interface
Software and Technology , ser. UIST ’12. New York, NY , USA:
Association for Computing Machinery, 2012, p. 219–228. [Online].
Available: https://doi.org/10.1145/2380116.2380145
[25] Y . Lin, X. Peng, Z. Xing, D. Zheng, and W. Zhao, “Clone-based and
interactive recommendation for modifying pasted code,” in 2015 10th
Joint Meeting of the European Software Engineering Conference and the
ACM SIGSOFT Symposium on the Foundations of Software Engineering,
ESEC/FSE 2015 - Proceedings , 2015, pp. 520–531.
[26] R. Cottrell, R. J. Walker, and J. Denzinger, “Jig saw: A tool for the small-
scale reuse of source code,” in Proceedings - International Conference
on Software Engineering , 2008, pp. 933–934.
[27] B. Reid, C. Treude, and M. Wagner, “Optimising the fit of stack
overflow code snippets into existing code,” GECCO 2020 Companion
- Proceedings of the 2020 Genetic and Evolutionary Computation
Conference Companion , no. 1, pp. 1945–1953, 2020.[28] T. Zhang, D. Yang, C. Lopes, and M. Kim, “Analyzing and Supporting
Adaptation of Online Code Examples,” in Proceedings - International
Conference on Software Engineering , vol. 2019-May. IEEE, 2019, pp.
316–327.
[29] V . Terragni, Y . Liu, and S. C. Cheung, “CSNIPPEX: Automated syn-
thesis of compilable code snippets from Q&A sites,” in ISSTA 2016 -
Proceedings of the 25th International Symposium on Software Testing
and Analysis , 2016, pp. 118–129.
[30] T. Zhang, Y . Yu, X. Mao, S. Wang, K. Yang, Y . Lu, Z. Zhang, and
Y . Zhao, “Instruct or interact? exploring and eliciting llms’ capability in
code snippet adaptation through prompt engineering,” 2024. [Online].
Available: https://arxiv.org/abs/2411.15501
[31] D. Huang, J. M. Zhang, M. Luck, Q. Bu, Y . Qing, and H. Cui,
“Agentcoder: Multi-agent-based code generation with iterative testing
and optimisation,” 2024. [Online]. Available: https://arxiv.org/abs/2312.
13010
[32] K. Ke, “ NIODebugger: A Novel Approach to Repair Non-Idempotent-
Outcome Tests with LLM-Based Agent ,” in 2025 IEEE/ACM 47th
International Conference on Software Engineering (ICSE) . Los
Alamitos, CA, USA: IEEE Computer Society, May 2025, pp. 762–
762. [Online]. Available: https://doi.ieeecomputersociety.org/10.1109/
ICSE55347.2025.00226
[33] F. Lin, D. J. Kim, Tse-Husn, and Chen, “Soen-101: Code generation
by emulating software process models using large language model
agents,” 2024. [Online]. Available: https://arxiv.org/abs/2403.15852
[34] H. Jin, L. Huang, H. Cai, J. Yan, B. Li, and H. Chen, “From llms to llm-
based agents for software engineering: A survey of current, challenges
and future,” 2024. [Online]. Available: https://arxiv.org/abs/2408.02479
[35] F. Batole, D. OBrien, T. N. Nguyen, R. Dyer, and H. Rajan, “An llm-
based agent-oriented approach for automated code design issue local-
ization,” in ICSE2025: The 47th International Conference on Software
Engineering , April 27-May 3 2025.
[36] W. Tao, Y . Zhou, Y . Wang, W. Zhang, H. Zhang, and Y . Cheng,
“Magis: Llm-based multi-agent framework for github issue resolution,”
inAdvances in Neural Information Processing Systems , A. Globerson,
L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang,
Eds., vol. 37. Curran Associates, Inc., 2024, pp. 51 963–51 993.
[37] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen,
J. Tang, X. Chen, Y . Lin et al. , “A survey on large language model based
autonomous agents,” Frontiers of Computer Science , vol. 18, no. 6, p.
186345, 2024.
[38] J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan,
and O. Press, “Swe-agent: Agent-computer interfaces enable automated
software engineering,” in Advances in Neural Information Processing
Systems , A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet,
J. Tomczak, and C. Zhang, Eds., vol. 37. Curran Associates, Inc.,
2024, pp. 50 528–50 652.
[39] Y . Zhang, H. Ruan, Z. Fan, and A. Roychoudhury, “Autocoderover:
Autonomous program improvement,” in Proceedings of the 33rd
ACM SIGSOFT International Symposium on Software Testing and
Analysis , ser. ISSTA 2024. New York, NY , USA: Association
for Computing Machinery, 2024, p. 1592–1604. [Online]. Available:
https://doi.org/10.1145/3650212.3680384
[40] I. Bouzenia, P. Devanbu, and M. Pradel, “RepairAgent: An Autonomous,
LLM-Based Agent for Program Repair,” 2024. [Online]. Available:
http://arxiv.org/abs/2403.17134
[41] S. Baltes, L. Dumani, C. Treude, and S. Diehl, “SOTorrent: Reconstruct-
ing and analyzing the evolution of stack overflow posts,” in Proceedings
of the 15th International Conference on Mining Software Repositories
(MSR 2018) , 2018, pp. 319–330.
[42] X. Zhu, J. Li, Y . Liu, C. Ma, and W. Wang, “A survey on
model compression for large language models,” Trans. Assoc. Comput.
Linguistics , vol. 12, pp. 1556–1577, 2024. [Online]. Available:
https://doi.org/10.1162/tacl a00704
[43] J. Li, J. Xu, S. Huang, Y . Chen, W. Li, J. Liu, Y . Lian, J. Pan, L. Ding,
H. Zhou, Y . Wang, and G. Dai, “Large language model inference
acceleration: A comprehensive hardware perspective,” 2025. [Online].
Available: https://arxiv.org/abs/2410.04466
[44] T. Wang, H. Liu, Y . Zhang, and Y . Jiang, “Recommending variable
names for extract local variable refactorings,” ACM Trans. Softw.
Eng. Methodol. , Jan. 2025, just Accepted. [Online]. Available:
https://doi.org/10.1145/3712191
[45] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y . Cao,
“ReAct: Synergizing Reasoning and Acting in Language Models,” pp.
1–33, 2023. [Online]. Available: http://arxiv.org/abs/2210.03629
[46] T. Zhang, Y . Lu, Y . Yu, X. Mao, Y . Zhang, and Y . Zhao, “How do
Developers Adapt Code Snippets to Their Contexts? An Empirical Study

12
of Context-Based Code Snippet Adaptations,” IEEE Transactions on
Software Engineering , vol. 50, no. 11, pp. 2712–2731, 2024.
[47] J. Chen, V . Tech, and V . Tech, “How Do Developers Reuse StackOver-
flow Answers in Their GitHub Projects ?” 2024 39th IEEE/ACM In-
ternational Conference on Automated Software Engineering Workshops
(ASEW) , pp. 146–155, 2024.
[48] Y . Wu, S. Wang, C. P. Bezemer, and K. Inoue, “How do developers uti-
lize source code from stack overflow?” Empirical Software Engineering ,
vol. 24, no. 2, pp. 637–673, 2019.
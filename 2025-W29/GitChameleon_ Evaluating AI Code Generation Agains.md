# GitChameleon: Evaluating AI Code Generation Against Python Library Version Incompatibilities

**Authors**: Diganta Misra, Nizar Islah, Victor May, Brice Rauby, Zihan Wang, Justine Gehring, Antonio Orvieto, Muawiz Chaudhary, Eilif B. Muller, Irina Rish, Samira Ebrahimi Kahou, Massimo Caccia

**Published**: 2025-07-16 16:10:42

**PDF URL**: [http://arxiv.org/pdf/2507.12367v1](http://arxiv.org/pdf/2507.12367v1)

## Abstract
The rapid evolution of software libraries poses a considerable hurdle for
code generation, necessitating continuous adaptation to frequent version
updates while preserving backward compatibility. While existing code evolution
benchmarks provide valuable insights, they typically lack execution-based
evaluation for generating code compliant with specific library versions. To
address this, we introduce GitChameleon, a novel, meticulously curated dataset
comprising 328 Python code completion problems, each conditioned on specific
library versions and accompanied by executable unit tests. GitChameleon
rigorously evaluates the capacity of contemporary large language models (LLMs),
LLM-powered agents, code assistants, and RAG systems to perform
version-conditioned code generation that demonstrates functional accuracy
through execution. Our extensive evaluations indicate that state-of-the-art
systems encounter significant challenges with this task; enterprise models
achieving baseline success rates in the 48-51\% range, underscoring the
intricacy of the problem. By offering an execution-based benchmark emphasizing
the dynamic nature of code libraries, GitChameleon enables a clearer
understanding of this challenge and helps guide the development of more
adaptable and dependable AI code generation methods. We make the dataset and
evaluation code publicly available at
https://github.com/mrcabbage972/GitChameleonBenchmark.

## Full Text


<!-- PDF content starts -->

GitChameleon: Evaluating AI Code Generation Against Python Library
Version Incompatibilities
Diganta Misra1,2*, Nizar Islah3,10*, Victor May4, Brice Rauby3, 5,
Zihan Wang6,Justine Gehring3,7,8,Antonio Orvieto1,2,9,Muawiz Chaudhary3,
Eilif B. Muller3,10,Irina Rish3,10,Samira Ebrahimi Kahou3,Massimo Caccia11
Team Leads, Data and Core Contributors, Senior Advisors
1ELLIS Institute Tübingen,2MPI-IS Tübingen,3Mila Quebec AI Institute,4Google,
5Polytechnique Montréal,6McGill University, Montréal,7Moderne,8Gologic,
9Tübingen AI Center,10Université de Montréal,11ServiceNow Research
Correspondence: diganta.misra@tue.ellis.eu, nizar.islah@mila.quebec
Abstract
The rapid evolution of software libraries poses
a considerable hurdle for code generation, ne-
cessitating continuous adaptation to frequent
version updates while preserving backward
compatibility. While existing code evolution
benchmarks provide valuable insights, they
typically lack execution-based evaluation for
generating code compliant with specific li-
brary versions. To address this, we introduce
GitChameleon , a novel, meticulously curated
dataset comprising 328 Python code comple-
tion problems, each conditioned on specific li-
brary versions and accompanied by executable
unit tests. GitChameleon rigorously evaluates
the capacity of contemporary large language
models (LLMs), LLM-powered agents, code as-
sistants, and RAG systems to perform version-
conditioned code generation that demonstrates
functional accuracy through execution. Our ex-
tensive evaluations indicate that state-of-the-art
systems encounter significant challenges with
this task; enterprise models achieving base-
line success rates in the 48-51% range, un-
derscoring the intricacy of the problem. By
offering an execution-based benchmark empha-
sizing the dynamic nature of code libraries,
GitChameleon enables a clearer understand-
ing of this challenge and helps guide the devel-
opment of more adaptable and dependable AI
code generation methods. We make the dataset
and evaluation code publicly available1.
1 Introduction
Large language models (LLMs) are increasingly
integral to software development, being adopted
for tasks like code generation and review (Council,
2024; Lambiase et al., 2025).
Despite LLM advancements like larger context
windows (Su et al., 2023), faster inference (Dao
et al., 2022), and high performance on general
*Equal Contribution
1https://github.com/mrcabbage972/GitChameleon
Benchmark
Problem Statement 
Instruction: Write a custom_violinplot  function that 
visualizes x and y from a Pandas DataFrame; scales the 
bandwidth to 1.5. Use the library Seaborn  version 0.13.0. 
import  seaborn as sns 
from  matplotlib.axes import  axes 
def custom_violinpolot( data : pd.DataFrame) 
-> Axes: 
    return 
Attempted Solution 
Model : gpt-4o-mini 
Solution : 
sns.violinplot( x='x', y='y', data =data, 
bw=1.5) 
Validation Result: 
AssertionError : bw parameter should not be 
used. Use bw_method and bw_adjust instead. Figure 1: In this GitChameleon problem, the
gpt-4o-mini model produced an incorrect solution due
forseaborn.violinplot by using the deprecated bw
parameter, instead of the appropriate bw_method and
bw_adjust required by the specified library version.
coding benchmarks (Hendrycks et al., 2021; Chen
et al., 2021), a critical capability remains under-
evaluated: generating code that is compliant with
a specific library version. This task of version-
switching, which is essential for robust develop-
ment in environments with fixed or legacy de-
pendencies, is not well-verified in contemporary
LLMs.
Existing benchmarks, while valuable, often fo-
cus on migrating codebases to newer versions
(i.e., code evolution) or use non-executable eval-
uation methods. They do not fully address the
challenge of generating new, functionally cor-
rect code for a static version constraint. For in-
stance, PyMigBench (Islam et al., 2023) provides
comprehensive datasets of real-world, inter-library
migrations, rather than focusing on executable,
intra-library tasks conditioned on specific versions.
1arXiv:2507.12367v1  [cs.SE]  16 Jul 2025

Training
Versioned data1.6, 1.7,
1.8, 1.9
4.19,....,
4.27
0.0.240,...,
0.0.283
Code Evolution
OOD versions/ libraries2.0, 2.1, 2.2
New
VCG
ID versions/ libraries1.7, 1.8
4.25
Inference
GitChameleonRustEvo2, 
CodeUpdateEvalLLMFigure 2: An illustration of two evaluation paradigms
for code generation models .Code Evolution (right) as-
sesses model capabilities on out-of-distribution (OOD)
data, using library versions or new libraries not encoun-
tered during training. In contrast, Version-Conditioned
Generation (VCG) (left) focuses on the practical ability
to generate code for specific, in-distribution (ID) library
versions that the model has seen before.
CodeUpdateArena (Liu et al., 2025) valuably as-
sesses LLM knowledge editing using synthetically
generated API updates for functions in popular
libraries, a different approach from using docu-
mented historical breaking changes. Other relevant
studies, such as Wang et al. (2024b), investigate
the propensity of LLMs to generate code with dep-
recated APIs, which does not entirely cover the
broader capability of generating software that ad-
heres to precise, user-specified library versions in-
volving various types of API changes.
Code Evolution vs. Version Conditioned Gener-
ation (VCG). Existing code evaluation bench-
marks often focus on assessing the code evolu-
tion or migration capabilities of LLMs, where
changes occur only in the forward direction and
typically involve unseen library versions or entirely
new libraries. This framing inherently makes the
task out-of-distribution (OOD), as illustrated in
Figure 2. In contrast, version-conditioned gen-
eration (VCG)—the ability of LLMs to produce
code aligned with specific, previously seen library
versions—is critical for practical deployment. It
enables models to function reliably in real-world
production environments or constrained settings
where the libraries in use may not be the latest sta-
ble versions. To better evaluate this capability, a
benchmark must pose problems that are strictly in-
distribution (ID) with respect to the relevant library
version(s) required to solve them.
To bridge this gap, our work introduces
GitChameleon , an executable benchmark de-signed to assess the capability of LLMs and AI
agents in generating version-aware Python code.
GitChameleon features problems centered on doc-
umented breaking changes from popular libraries,
requiring models to produce solutions for explic-
itly specified versions (an illustrative example is
shown in Figure 1). The development of such a
benchmark faces challenges in meticulously curat-
ing version-specific breaking changes from library
changelogs and crafting corresponding testable sce-
narios. Our comprehensive evaluation of diverse
LLM-based tools on GitChameleon reveals criti-
cal limitations in existing systems’ ability to handle
library versioning.
In summary, our contributions are highlighted as
follows:
•We introduce a novel code completion
benchmark GitChameleon consisting of 328
Python-based version-conditioned problems,
including visible tests for self-debugging
and documentation references for Retrieval-
Augmented Generation (RAG).
•We present a comprehensive empirical study
onGitChameleon , evaluating the capabilities
of a diverse range of contemporary AI code
generation systems, including AI agents, IDE-
integrated and CLI-based coding assistants,
and RAG-based LLM pipelines.
•We reveal critical limitations in the ability
of current AI systems to adhere to specific
versioning constraints and highlight factors
impacting their performance, thereby provid-
ing insights to steer the development of more
adaptable and dependable AI code generation
methods.
2 GitChameleon Benchmark
We introduce GitChameleon , a manually au-
thored benchmark that comprises 328 Python-
based version-conditioned problems focused on
popular code libraries. To evaluate performance
onGitChameleon , each problem is accompanied
by a suite of assertion-based unit tests, enabling
a thorough execution-based assessment of poten-
tial solutions. In the following sections, we detail
the dataset structure, dataset statistics, evaluation
metrics, and sample verification process.
2

20 40 60
SWE-Bench ( )
40455055GitChameleon ( )
40 60 80
LiveCodeBench ( )
Models
GPT-4o
GPT-4o-Mini
GPT-4.1
GPT-4.1 MiniGPT-4.5
Gemini 1.5-Pro
Gemini 2.5-Pro
Gemini 2.5-FlashClaude 3.7 Sonnet
Claude 3.5 Sonnet
LLama 3.1
LLama 3.3Figure 3: Can you predict GitChameleon performance
from other code generation benchmarks? Here we
present the Spearman ( ρ) and Pearson ( r) correla-
tions between GitChameleon , SWE-Bench (Jimenez
et al., 2024), and LiveCodeBench (Jain et al., 2024).
GitChameleon exhibits a moderate correlation with
SWE-Bench, with ρof 0.550 and rof 0.675; and a
weak correlation with LiveCodeBench, with ρof 0.214
androf 0.130.
2.1 Dataset Structure
Each dataset sample includes a problem related to
a breaking change in a Python library.
To validate a candidate solution, we provide
a suite of tests, consisting of a comprehensive
suite of Hidden Tests to be used for model per-
formance evaluation and ranking and a concise
Visible Test to provide execution feedback for Self-
Debugging (Chen et al., 2023) experiments.
The detailed structure of dataset samples is pre-
sented in Table 5. For a schematic of the work-
flow for evaluating a method against a sample from
GitChameleon , see Figure 5.
2.2 Evaluation Metrics
The benchmark metric is the success rate on hidden
tests, which directly penalizes version mismatches
that cause runtime errors during our execution-
based validation. As a secondary metric, we use the
API Hit Rate (Wang et al., 2024a): the percentage
of generated solutions that correctly call all APIs
specified in the ground-truth solution. Note that
this hit rate can be lower than the success rate, as
functionally correct alternative solutions may use
different APIs.2.3 Statistics
GitChameleon consists of 328 Python-based ver-
sion conditioned problems based on 26 libraries
spanning scientific computing, data science and
web development. The samples were collected
from version releases over a period from the year
2014 to 2023 and exclude legacy and yanked ver-
sion releases.
0 10 20 30 40 50 60 70 802014201520162017201820192020202120222023(a) Number of samples by version release year
0 20 40 60 80 100ArgumentFunction
 NameSemanticsNew 
 FeatureOther(b) Number of samples by change category
Figure 4: (a) Most versions in GitChameleon were
released between 2021–2023, with a few in earlier years.
(b) The most common type of change between versions
was an argument or attribute change, while semantic or
functional changes were least common.
As demonstrated in Fig. 4(a), most of the sam-
ples in GitChameleon are from versions of li-
braries released in the years 2021-2023. We in-
tentionally use versions that fall within the training
window of most evaluated models. The challenge
is therefore not one of data contamination, but of
control and disambiguation : when a model has
been exposed to multiple library versions, can it
correctly generate code for the specific version re-
quired by the prompt?
The dataset was constructed through careful
manual effort, with over 350 hours invested in iden-
tifying historical breaking changes, crafting prob-
lem statements, and validating unit tests. Further
details about the benchmark and its construction
process are presented in Appendix A.
3 Empirical Study
We evaluate GitChameleon in a comprehensive
selection of settings, including Greedy Decod-
3

Problem 
Statement 
Starter Code 
Dependency 
Info
Candidate 
Solution Hidden Tests 
Visible Tests Validation Inputs 
Benchmark 
Success 
Self-Debug Figure 5: An illustration of the workflow for a single
example within GitChameleon . The inputs, comprising
the Problem Statement, Starter Code, and Dependency
Info, are processed by an LLM or an AI agent to gen-
erate a Candidate Solution. This candidate solution
then undergoes validation using the Hidden Tests to de-
termine success on the benchmark. Results from the
Visible Tests can be fed back into the solution method
for self-debugging.
ing, Chain-of-Thought (Wei et al., 2023), Self-
Debugging (Chen et al., 2023), RAG (Lewis et al.,
2020), Multi-Step Agents (Yao et al., 2023) and
enterprise Coding Assistant software products, to
assess their ability to generate version-specific exe-
cutable code.
This section first presents the experimental setup,
then reports the experiment results in each setting,
and finally shows a breakdown of the observed
results along a few key dimensions.
3.1 Experimental Setup
In this section, we present the experimental setup
used for each of our settings. To ensure version
compliance, we use a dual control mechanism: the
target version is explicitly included in the model’s
prompt, and the validation environment is config-
ured with that exact library version. All prompts
are shown in Appendix I. For prompt optimization,
we used the Anthropic Prompt Improver2. Fur-
ther automated prompt optimization efforts did not
make a significant change, as described in Table 11.
3.1.1 Greedy Decoding
We configured the generation parameters with a
sampling temperature of 0 and a top_p value of
0.95. We had specified a structured output schema
that specifies the fields Answer andExplanation ,
where both are of type string .
2https://docs.anthropic.com/en/docs/build-wit
h-claude/prompt-engineering/prompt-improver3.1.2 Zero-Shot Chain-Of-Thought (CoT)
We had used the same generation parameters as
for Greedy Decoding and an output schema that
specifies the fields Answer andSteps , where the
former is a of type string and the latter is a list of
string .
3.1.3 Self-Debugging
On examples that failed with Greedy Decoding, we
employed the method described in (Chen et al.,
2023) to feed the visible test error trace along with
the model’s explanation of its output back to the
model.
3.1.4 Retrieval-Augmented Generation
We designed a RAG (Lewis et al., 2020) pipeline
where we first constructed a vectorized database
(VectorDB) by embedding each sample’s rel-
evant API documentation with the OpenAI
text-embedding-3 large model3. The corpus
used for constructing the VectorDB included 536
documents, with 140 samples having 1 associated
document, 168 having 2 associated documents and
20 having 3 documents.
Subsequently, we used DocPrompting (Zhou
et al., 2022) to query the VectorDB to generate
solutions.
3.1.5 Multi-Step Agent
We conducted experiments with a tool-calling
agent, as implemented by the smolagents (Roucher
et al., 2025)4framework. This agent implemen-
tation mostly follows the ReAct (Yao et al., 2023)
method, but, it alternates between acting and plan-
ning (Li, 2024) steps.
Following the Agentic RAG approach (Singh
et al., 2025), we had equipped the agent with a
grounding tool in order to assess its capability to
independently fetch relevant info for solving the
benchmark problems. To this end, we had experi-
mented with the following grounding tools: Duck-
DuckGo Search (DuckDuckGo, 2025), Perplex-
ity (Perplexity AI, 2024), and Gemini with Ground-
ing (Google, 2025).
Additionally, we examined agentic multi-step
self-debugging (Jin et al., 2024) by including or
omitting a code execution sandbox tool (Rabin
et al., 2025), which provides the needed depen-
dencies for each example. The sandbox takes a
3https://openai.com/index/new-embedding-model
s-and-api-updates/
4https://huggingface.co/learn/agents-course/
en/unit2/smolagents/tool_calling_agents
4

Python program as input and outputs the standard
output from the program.
3.1.6 AI Coding Assistants
In addition to evaluating a generic agentic frame-
work endowed with basic tools, we also analyze
the performance of specialized AI coding assistant
software.
For this setting, we examine both Command-
Line Interface (CLI), such as Claude Code5cod-
ing assistants and Integrated Development Environ-
ment (IDE) coding assistants, such as Cline6.
Specifically, in this evaluation we aimed to eval-
uate the code completion functionality of code as-
sistants in an IDE or terminal environment wherein
the goal was to complete the starter code of each
GitChameleon problem with the generated solu-
tion.
The input to the assistants is given as a Python
file which consists of the required library, version
and extra dependencies as in-line comments and
subsequently the starter code. NOTE:All assistants
had internet and terminal commands execution ac-
cess.
We had furthermore ablated this setting versus
giving the full problem statement as input.
3.2 Experiment Results
This section presents the benchmark results in each
setting, as described in the Experimental Setup
section (3.1). Table 1 contains the results for
Greedy Decoding, Self-Debug and Zero-Shot CoT.
3.2.1 Greedy Decoding
We observe that the largest Enterprise-grade mod-
els, including Claude 3.7 Sonnet ,Gemini 2.5
Pro,GPT-4.1 ,GPT-4o , and o1, exhibit comparable
hidden success rates, generally falling within the
48–51% range. Among these o1(51.2% hidden)
achieves the highest hidden success rate.
The open-weight Llama models are notably be-
hind, even the recently released Llama 4 Maverick
FP8(40.8% hidden success rate).
Model size clearly impacts performance: for
instance, Gemini 2.5 Flash trails its Procoun-
terpart by nearly 12% on hidden tests (38.1%
vs. 50.0%). Similarly, the mini and nano se-
ries within the GPT family (e.g., GPT-4.1-mini ,
GPT-4.1-nano ,GPT-4o-mini ) consistently show
5https://docs.anthropic.com/en/docs/claude-c
ode/overview
6https://cline.bot/lower performance than their larger full-size sib-
lings, with differences on hidden tests ranging from
approximately 4 to 15 points.
3.2.2 Zero-Shot Chain-Of-Thought
This approach does not uniformly improve LLM
performance across all models. While some mod-
els demonstrate significant gains in hidden suc-
cess rates, a substantial number of enterprise-grade
models and their smaller variants experience per-
formance degradation.
For instance, notable improvements in hidden
success rates are observed in models such as Llama
3.1 Instruct Turbo (from 30.2% to 36.6%, a
+6.4 point increase) and o3-mini (from 45.1% to
50.9%, a +5.8 point increase).
Conversely, several models exhibit a decrease
in performance with CoT. Prominent examples in-
clude Gemini 2.0 Flash (from 44.2% to 36.0%)
and even the top-performing o1(from 51.2% to
41.2%).
3.2.3 LLM Self-Debugging
Hidden Success Rate : Across models, Self-
Debugging significantly improves the hidden suc-
cess rates. Observed gains range from approxi-
mately 10% to 20%. For instance, Llama 3.1 ’s
hidden success rate increases from 30% to 52.1%,
andGPT-4.1-mini shows an improvement from
44% to 68%. This demonstrates the strong capa-
bility of modern LLMs to diagnose failures and
generate corrected code.
Visible Success Rate : As expected, the im-
provement is even more pronounced on visible
tests, ranging from 13 to 37 points. For instance,
GPT-4.1 ’s success rate improves from 49% to 69%,
Claude 3.7 Sonnet ’s success rate improves from
56% to 83% and Gemini 2.0 Flash improves
from 50% to 75%.
Visible-Hidden Gap Analysis : In Figure 6, we
present the effect of self-debugging on the size of
the gap between the success rate on visible tests
and the success rate on hidden tests.
3.2.4 Multi-Step Agent
We report the performance of Multi-Step Agents on
GitChameleon in Table 2. A clear and significant
trend is the substantial increase in success rates for
all models and grounding methods when giving
the agent a sandbox tool. Overall, Claude Sonnet
3.5demonstrated the highest success rates with
a sandbox, across all grounding methods, while
5

ModelGreedy Decoding Greedy with Self-Debug Zero-shot CoT
Success
Rate (%)API
Hit
Rate (%)Success
Rate (%)API
Hit
Rate (%)Success
Rate (%)API
Hit
Rate (%)Hidden Visible Hidden Visible Hidden
Open-Weights Models
Llama 3.1 Instruct Turbo 30.2±2.5 38.1±2.7 39.7±2.7 52.1±2.8 69.2±2.5 41.5±2.7 36.6±2.7 35.3±2.6
Llama 3.3 Instruct Turbo 70B 36.3±2.7 43.3±2.7 36.4±2.7 53.0±2.8 70.1±2.5 37.4±2.7 37.5±2.7 37.2±2.7
Llama 4 Maverick 400B 40.8±2.7 46.6±2.8 49.5±2.8 58.5±2.7 72.3±2.5 46.8±2.8 46.6±2.8 41.3±2.7
Qwen 2.5-VL Instruct 72B 48.2±2.8 55.5±2.7 43.8±2.7 64.6±2.6 77.4±2.3 45.3±2.7 45.1±2.7 43.0±2.7
Enterprise Models
Claude 3.7 Sonnet 48.8±2.8 55.8±2.7 46.0±2.8 65.9±2.6 75.9±2.4 47.6±2.8 45.1±2.7 43.4±2.7
Gemini 1.5 Pro 45.1±2.7 51.5±2.8 46.8±2.7 62.5±2.8 72.6±2.4 48.6±2.7 43.3±2.7 44.6±2.8
Gemini 2.0 Flash 44.2±2.7 50.6±2.8 43.8±2.7 70.4±2.7 79.0±2.4 49.4±2.7 36.0±2.6 41.8±2.7
Gemini 2.5 Pro 50.0±2.8 61.0±2.8 47.7±2.7 61.3±2.8 73.8±2.2 49.2±2.7 49.4±2.8 49.1±2.8
Gemini 2.5 Flash 38.1±2.6 41.8±2.7 45.4±2.7 65.9±2.8 73.2±2.4 45.8±2.7 30.8±2.5 49.8±2.8
GPT-4.1 48.5±2.8 49.1±2.8 46.8±2.7 63.4±2.8 76.8±2.1 48.3±2.7 47.9±2.8 44.5±2.7
GPT-4.1-mini 44.2±2.7 50.0±2.8 44.5±2.7 68.0±2.8 79.3±2.3 46.3±2.7 24.1±1.8 41.3±2.7
GPT-4.1-nano 33.8±2.6 35.1±2.6 43.1±2.7 67.7±2.7 74.4±2.6 45.8±2.7 11.9±1.8 32.1±2.5
GPT-4o 49.1±2.8 54.0±2.8 46.5±2.7 64.9±2.8 72.3±2.5 48.0±2.7 50.3±2.8 42.5±2.7
GPT-4o-mini 37.2±2.6 46.3±2.7 38.4±2.6 60.4±2.7 71.6±2.6 40.6±2.7 36.0±2.6 37.3±2.6
GPT-4.5 40.8±2.7 46.0±2.7 52.8±2.8 66.2±2.8 74.4±2.4 54.4±2.7 39.9±2.6 48.8±2.8
Grok 3 48.2±2.8 53.7±2.8 44.8±2.7 67.1±2.8 77.1±2.3 46.3±2.8 49.4±2.8 44.2±2.7
Mistral Medium 3 43.6±2.7 49.1±2.8 44.2±2.7 61.3±2.8 71.3±2.5 45.4±2.7 44.2±2.7 44.1±2.7
Table 1: Success rate on visible and hidden tests and API hit rate under the Greedy, Self-Debug, and Zero-shot
CoT settings, grouped by OSS vs. Enterprise models. Model ranking on the benchmark is determined by Hidden
Success Rate . Visible Success Rate figures are for context on Self-Debugging. The best result in each column is in
bold. For full model details and citations, please refer to Appendix J.
0 10 20
Success Rate Gap With Self-Debug (%)0510152025Success Rate Gap Without Self-Debug (%)Models
Claude 3.7 Sonnet
Gemini 1.5 Pro
Gemini 2.0 Flash
Gemini 2.5 Pro
Gemini 2.5 Flash
GPT-4.1
GPT-4.1-mini
GPT-4.1-nano
GPT-4o
GPT-4o-mini
GPT-4.5
Grok 3
Mistral Medium 3
Figure 6: Analysis of the Visible-Hidden Gap Be-
fore and After Self-Debugging . We analyze how self-
debugging affects the gap between the success rate on
visible and hidden tests. We can see that for all models,
the gap increases after self-debugging. This shows that
self-debugging on visible tests has a limited ability to
improve on the hidden tests.
Gemini 1.5 Pro demonstrated the best results
without a sandbox.
3.2.5 AI Coding Assistants
Table 3 presents the success rates of various CLI
and IDE assistants on the visible and hidden tests
inGitChameleon . When the problem statement isModelGrounding
MethodSuccess
Rate (%)API Hit
Rate (%)
No Sandbox Sandbox No Sandbox Sandbox
Claude
Sonnet
3.5DuckDuckGo 41.7 ±2.7 55.3±2.7 42.2±2.7 48.9±2.8
Perplexity 44.1 ±2.7 51.4±2.8 41.8±2.7 46.0±2.8
Grounded Gemini 40.0 ±2.7 53.7±2.8 41.0±2.7 45.2±2.7
Gemini
1.5 ProDuckDuckGo 46.0 ±2.8 49.8±2.8 47.4±2.8 50.3±2.8
Perplexity 46.5±2.8 44.4±2.7 47.2±2.8 46.6±2.8
Grounded Gemini 44.1 ±2.7 49.2±2.8 49.7±2.8 51.2±2.8
GPT-4oDuckDuckGo 23.9 ±2.4 33.2±2.6 44.2±2.7 48.1±2.8
Perplexity 33.5 ±2.6 41.5±2.7 43.2±2.7 44.7±2.7
Grounded Gemini 25.4 ±2.4 50.0±2.8 46.5±2.8 44.2±2.7
Table 2: Multi-Step Agent performance with different
models, grounding methods, and sandbox states. The
best result in each column is in bold.
not given, Cline with GPT-4.1 achieves the best
result, with a success rate of 38.4%. All assistants
besides for Goose on GPT-4o demonstrate signif-
icant gains, ranging from 12 to 35 points, from
including the problem statement.
3.2.6 Retrieval-Augmented Generation
Table 4 presents the performance of various models
with RAG. Many models exhibit a significant (up to
10%) boost in success rate with RAG compared to
greedy decoding alone. Notably, GPT-4.1 , the best
7This version of the model is not FP8-quantized, unlike
the one presented in Table 1
6

Name ModelSuccess Rate
(%)API Hit Rate
(%)
No-prob Prob No-prob Prob
CLI Assistants
Claude Code Claude 3.7 Sonnet 32.0 ±2.6 48.8±2.8 44.2±2.7 45.5±2.7
GooseGPT-4o 36.3±2.7 36.9±2.7 43.9±2.7 54.5±2.7
GPT-4.1 19.2 ±2.2 55.5±2.7 41.7±2.7 53.0±2.8
IDE Assistants
ClineClaude 3.7 Sonnet 32.9 ±2.6 44.8±2.7 40.5±2.7 50.2±2.8
GPT-4.1 38.4 ±2.7 54.6±2.7 42.4±2.7 48.8±2.8
GPT-4.1-mini 27.1 ±2.5 42.1±2.7 32.9±2.6 52.4±2.8
GPT-4.1-nano 38.1 ±2.7 54.6±2.7 42.4±2.7 48.8±2.8
GPT-4o 41.5±2.7 – 42.7 ±2.7 –
Kilocode Claude 3.7 Sonnet 30.2 ±2.5 – 43.3±2.7 –
Roocode Claude 3.5 Sonnet 12.5 ±1.8 – 41.2 ±2.7 –
Table 3: Success and API-hit rates for CLI and IDE
coding assistants, under the setting where the problem
statement is given ( Prob ) and where it is not ( No-prob ),
in which case we evaluate a scenario akin to tab code-
completion. The results show that including the problem
statement improves success rate by double-digit margins
for 4 out of 5 cases evaluated.
ModelSuccess
Rate (%)API Hit
Rate (%)Precision
(%)Recall
(%) MRR
Open-Weights Models
Deepseek V3 48.9±2.8 48.5±2.8 41.6±2.2 50.4±2.80.62±0.03
Llama 4 Maverick745.1±2.7 50.5±2.8 41.2±2.2 49.8±2.80.61±0.03
Qwen3 41.8 ±2.7 39.6±2.7 36.3±2.0 46.9±2.80.56±0.03
Jamba 1.6 Large 41.8 ±2.7 47.1±2.8 41.9±2.2 50.7±2.80.62±0.03
Enterprise Models
Claude 3.7 Sonnet 56.1 ±2.7 53.0±2.8 41.9±2.2 50.7±2.80.62±0.03
Claude 4 Sonnet 59.4±2.8 55.8±2.8 41.9±2.2 50.7±2.80.62±0.03
Gemini 2.5 Pro 56.7 ±2.7 51.1±2.8 41.9±2.2 50.7±2.80.62±0.03
GPT-4.1 58.5 ±2.7 51.8±2.8 41.2±2.2 50.1±2.80.61±0.03
Grok3 54.3 ±2.7 55.2±2.8 41.6±2.2 50.4±2.80.62±0.03
Mistral Medium 3 52.4 ±2.7 51.2±2.8 41.6±2.2 50.4±2.80.62±0.03
Devstral Small 43.3 ±2.7 45.1±2.8 41.6±2.2 50.4±2.80.62±0.03
Nova Pro 44.2 ±2.7 42.4±2.7 40.7±2.2 49.6±2.80.60±0.03
Table 4: RAG performance for a subset of models when
retrieving k= 3 most relevant documents. The best
success rate and API hit rate results for each model
group are in bold. An extended version of the RAG
experiment results is presented in Appendix C.
performing model achieves a success rate of 58.5%,
up from 48.5% with greedy decoding. These results
demonstrate that the benchmark is still challenging
even with access to the library documentation, with
over 40% of the problems remaining unsolved in
the best case.
3.3 In-Depth Analysis of Findings
This section provides a detailed analysis of the ex-
perimental results, focusing on model performance
across several key dimensions. These dimensions
include the impact of different API change types,
a comparison between success rate and API hit
rate, and the effectiveness of self-debugging acrossvarious error types.
Comparison of Success Rate and API Hit Rate
API hit rate shows a moderate positive Pearson
correlation with hidden-test success under Greedy
Decoding with the Pearson correlation coefficient
(r= 0.392,p= 0.097,N= 19 ), indicating
that models which invoke the ground truth APIs
more often tend to perform better on hidden tests
in the Greedy setting, but falls just short of statis-
tical significance at 5% level. Under Zero-Shot
CoT, the correlation remains similar in magni-
tude ( r= 0.483) and is statistically significant
(p= 0.036,N= 19 ). In the Self-Debug regime,
however, the association becomes both stronger
and highly significant ( r= 0.615,p= 0.011,
N= 16 ), demonstrating that when models can
iteratively refine their outputs, invoking ground
truth APIs becomes an especially reliable predictor
of hidden-test performance.
Analysis of Performance by Type of API Change
Figure 7 illustrates the performance of mod-
els across various API change types within the
GitChameleon benchmark, revealing notable vari-
ations in success rates. Semantic changes were the
most tractable, with success rates ranging from
60–80% with Self-Debug and 55–65% without.
New-feature additions proved to be the most chal-
lenging, with success rates between 25–50% for
Greedy Decoding and 50–65% for Self-Debug. No-
tably, the Code Assistant Goose exhibited a sub-
stantial discrepancy in its performance on semantic
and function-name changes compared to argument
changes and new features. This suggests a height-
ened sensitivity to change category for Goose , a
characteristic not observed in the enterprise models
or the Claude -powered tool-calling agent.
Self-Debug Error Categorization Figure 8
shows that self-debugging consistently lowers the
rate of every class of traceback error, both in abso-
lute numbers and relative terms:
(a) Raw Counts: We observe that for
all error categories—from the most common
(AssertionError and TypeError ) down to the
rarest ( RuntimeError )—applying Self-Debugging
significantly lowers the total number of failures.
(b) Percentage Reduction: When nor-
malized by the Greedy Decoding baseline,
reductions span roughly 50% up to about
90%. The biggest relative improvements ap-
pear in the infrequent categories—such as
RuntimeError andSyntaxError —while the com-
7

0.2 0.4 0.6 0.8
Success RateClaude 3.7
Sonnet
Gemini 1.5
Pro
Gemini 2.5
Pro
GPT-4.1
O-1
DDG SB
Claude 3.5
GooseArgument Change
0.2 0.4 0.6 0.8
Success RateFunction Name
0.2 0.4 0.6 0.8
Success RateSemantics
0.2 0.4 0.6 0.8
Success RateNew Feature
Vanilla Self-DebugFigure 7: Success Rate Breakdown by Type of Change : We analyze success rates with and without self-debugging,
grouped by the type of change. Light shaded bars represent values obtained from self-debugging. Standard error is
drawn as a black line. We include DDG-SB , a Multi-Step Agent variant where DuckDuckGo is used for grounding
and access to a sandbox is enabled. and the Coding Assistant Goose . Self-Debug results for these are omitted.
AssertionErrorTypeError
AttributeErrorValueError IndexError
RuntimeErrorKeyError
NameErrorSyntaxErrorHTTPError
Error Categories02000400060008000T otal Errors
49.7%
59.0%
66.3%
64.1%
36.1%
55.1%
46.5%
76.6%
77.1%
68.1%Greedy Decoding
Self-Debug
Figure 8: Total error count for each category un-
der Greedy decoding versus Self-Debug . Self-Debug
yields substantial decreases all types of errors.
mon AssertionError andTypeError still see de-
crease in the range of 60-70%.
4 Related Work
The continuous evolution of software libraries
presents significant challenges for AI-driven code
generation. This section reviews existing bench-
marks designed to evaluate model performance in
this context. Specialized frameworks developed to
address the challenge are presented in appendix D.2
The challenge of evaluating large language mod-
els (LLMs) in the context of evolving software
libraries and their versions has been approached
by several benchmarks. These benchmarks, while
valuable, often differ in scope, methodology, or
evaluation techniques compared to GitChameleon .
PyMigBench Focusing on Python library mi-gration, this benchmark uses 321 real-world in-
stances, evaluating both individual code transfor-
mations and the functional correctness of entire mi-
grated segments via unit tests (Islam et al., 2023).
PyMigBench revealed that LLMs often handle in-
dividual changes well but struggle with achieving
full functional correctness, especially for complex
argument transformations.
VersiCode (Wu et al., 2024) and the dataset by
Wang et al. (Wang et al., 2024b) address library
evolution but primarily depend on string matching
for evaluation.
CodeUpdateArena (Liu et al., 2025) investi-
gates model adaptation to synthetically generated
API updates for functions in popular libraries.
GitChameleon distinguishes itself by focusing
on the real-world scenario where developers are
often constrained to specific library versions due
to technical debt. Unlike CodeUpdateArena’s syn-
thetic changes, GitChameleon evaluates LLMs
on their ability to generate code for actual, docu-
mented historical breaking changes within library
versions they were likely exposed to during training.
Furthermore, diverging from the string-matching
evaluations of VersiCode and Wang et al. (Wang
et al., 2024b), GitChameleon is based on ex-
ecutable tests. This provides a more practical
and rigorous assessment of functional accuracy in
version-specific code generation. For an extended
discussion of how GitChameleon is differentiated
from existing work, please see Appendix D.2.
8

5 Conclusion
The rapid evolution of software libraries presents
a critical challenge for LLM-powered AI sys-
tems in generating functionally correct, version-
conditioned code. To address this, we introduce
GitChameleon , a novel Python-based benchmark
meticulously curated with version-conditioned
problems and executable tests. Our extensive eval-
uation reveals that state-of-the-art LLMs, agents
and code assistants currently struggle significantly
with this task, achieving modest success rates.
By shedding light on current limitations
and facilitating execution-based evaluation,
GitChameleon aims to foster the development of
more robust and adaptable code generation models
for evolving software environments.
Acknowledgements
The authors thank the International Max Planck
Research School for Intelligent Systems (IMPRS-
IS) for supporting Diganta Misra. This work was
partially enabled by compute resources provided
by Mila8and was funded by the Max Planck &
Amazon Science Hub.
Limitations
While we aim to provide a comprehensive and
holistic evaluation of LLMs on the task of version-
conditioned generation, our benchmark is currently
limited to Python and a small set of libraries. More-
over, we focus solely on code generation from
natural language instructions, and do not evalu-
ate version-to-version translation—i.e., converting
code from one library version to another—even
when both versions are in-distribution relative to
the model’s training. For instance, if a model has
been trained on PyTorch versions 1.7, 1.8, and 1.9,
it would be valuable to assess whether it performs
better when given a solution in 1.8 and asked to
upgrade to 1.9 or downgrade to 1.7. Finally, we
do not include human evaluations, which could
provide a baseline for estimating average human
performance on this task.
8https://mila.quebec
9

References
Abubakar Abid, Ali Abdalla, Ali Abid, Dawood Khan,
Abdulrahman Alfozan, and James Zou. 2019. Gradio:
Hassle-free sharing and testing of ML models in the
wild. Preprint , arXiv:1906.02569.
Meta AI. 2025. Everything we announced at our first-
ever LlamaCon. https://ai.meta.com/blog/lla
macon-llama-news/ . Discusses Llama 3.3 Instruct
Turbo and Llama 4 Maverick.
Mohannad Alhanahnah, Yazan Boshmaf, and Benoit
Baudry. 2024. DepsRAG: Towards managing soft-
ware dependencies using large language models.
arXiv preprint arXiv:2405.20455v2 .
Anthropic. 2025. Claude 3.7 Sonnet and Claude Code.
https://www.anthropic.com/news/claude-3-7
-sonnet .
Arcee. Model Selection | Arcee AI Documentation —
docs.arcee.ai. https://docs.arcee.ai/arcee-c
onductor/arcee-small-language-models/mode
l-selection#caller-large-tool-use-and-fun
ction-call . [Accessed 15-07-2025].
Farnaz Behrang, Zhizhou Zhang, Georgian-Vlad Saioc,
Peng Liu, and Milind Chabbi. 2025. Dr.fix: Auto-
matically fixing data races at industry scale. Preprint ,
arXiv:2504.15637.
Lars Buitinck, Gilles Louppe, Mathieu Blondel, Fabian
Pedregosa, Andreas Mueller, Olivier Grisel, Vlad
Niculae, Peter Prettenhofer, Alexandre Gramfort,
Jaques Grobler, Robert Layton, Jake VanderPlas, An-
dreas Joly, Bertrand Druillette, Gael Varoquaux, and
Marion Gramfort. 2013. API design for machine
learning software: experiences from the scikit-learn
project. arXiv preprint arXiv:1309.0238 .
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan,
Henrique Ponde de Oliveira Pinto, Jared Kaplan,
Harri Edwards, Yuri Burda, Nicholas Joseph, Greg
Brockman, Alex Ray, Raul Puri, Gretchen Krueger,
Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela
Mishkin, Brooke Chan, Scott Gray, and 39 others.
2021. Evaluating large language models trained on
code. ArXiv .
Xinyun Chen, Maxwell Lin, Nathanael Schärli, and
Denny Zhou. 2023. Teaching large language models
to self-debug. Preprint , arXiv:2304.05128.
Keyuan Cheng, Xudong Shen, Yihao Yang, Tengyue
Wang, Yang Cao, Muhammad Asif Ali, Hanbin
Wang, Lijie Hu, and Di Wang. 2025. Codemenv:
Benchmarking large language models on code migra-
tion. Preprint , arXiv:2506.00894.
Matteo Ciniselli, Alberto Martin-Lopez, and Gabriele
Bavota. 2024. On the generalizability of deep
learning-based code completion across programming
language versions. Preprint , arXiv:2403.15149.Google Cloud. 2025. Gemini 2.5 on Vertex AI: Pro,
Flash & Model Optimizer Live. https://cloud.
google.com/blog/products/ai-machine-lea
rning/gemini-2-5-pro-flash-on-vertex-ai .
Discusses Gemini 2.5 Pro and Gemini 2.5 Flash.
Team Cohere, :, Aakanksha, Arash Ahmadian, Marwan
Ahmed, Jay Alammar, Milad Alizadeh, Yazeed Al-
numay, Sophia Althammer, Arkady Arkhangorodsky,
Viraat Aryabumi, Dennis Aumiller, Raphaël Avalos,
Zahara Aviv, Sammie Bae, Saurabh Baji, Alexandre
Barbet, Max Bartolo, Björn Bebensee, and 211 oth-
ers. 2025. Command a: An enterprise-ready large
language model. Preprint , arXiv:2504.00698.
Forbes Technology Council. 2024. Revolutionizing
software development with large language models.
https://www.forbes.com/councils/forbeste
chcouncil/2024/03/20/revolutionizing-sof
tware-development-with-large-language-mod
els/ .
Tri Dao, Daniel Y . Fu, Stefano Ermon, Atri Rudra,
and Christopher Ré. 2022. FlashAttention: Fast and
memory-efficient exact attention with IO-awareness.
InAdvances in Neural Information Processing Sys-
tems (NeurIPS) .
DeepSeek-AI. 2025. Deepseek-r1: Incentivizing rea-
soning capability in llms via reinforcement learning.
Preprint , arXiv:2501.12948.
DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingx-
uan Wang, Bochao Wu, Chengda Lu, Chenggang
Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan,
Damai Dai, Daya Guo, Dejian Yang, Deli Chen,
Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai,
and 181 others. 2025. Deepseek-v3 technical report.
Preprint , arXiv:2412.19437.
DuckDuckGo. 2025. DuckDuckGo: Privacy, simplified.
https://duckduckgo.com/ .
Lishui Fan, Mouxiang Chen, and Zhongxin Liu.
2024. Self-explained keywords empower large
language models for code generation. Preprint ,
arXiv:2410.15966.
Google. 2025. Grounding with Google Search | Gemini
API. https://ai.google.dev/gemini-api/doc
s/grounding .
Aric A Hagberg, Daniel A Schult, and Pieter J Swart.
2008. Exploring network structure, dynamics, and
function using NetworkX. In Proceedings of the 7th
Python in Science Conference , pages 11–15.
Charles R Harris, K Jarrod Millman, Stéfan J van der
Walt, Ralf Gommers, Pauli Virtanen, David Cour-
napeau, Eric Wieser, Julian Taylor, Sebastian Berg,
Nathaniel J Smith, Robert Kern, Matti Picus,
Changqing Hoyer, Marten H van Kerkwijk, Alex
Brett, Andrew Wen, Pete Zhang, Joe Igoe, Keith
Featherstone, and Travis E Oliphant. 2020. Array
programming with NumPy. Nature , 585(7825):357–
362.
10

Dan Hendrycks, Steven Basart, Saurav Kadavath, Man-
tas Mazeika, Akul Arora, Ethan Guo, Collin Burns,
Samir Puranik, Horace He, Dawn Song, and Jacob
Steinhardt. 2021. Measuring coding challenge com-
petence with apps. NeurIPS .
J. D. Hunter. 2007. Matplotlib: A 2D graphics en-
vironment. Computing in Science & Engineering ,
9(3):90–95.
Amazon Artificial General Intelligence. 2024. The ama-
zon nova family of models: Technical report and
model card. Amazon Technical Reports .
Mohayeminul Islam, Ajay Kumar Jha, Sarah Nadi, and
Ildar Akhmetov. 2023. Pymigbench: A benchmark
for python library migration. In 2023 IEEE/ACM
20th International Conference on Mining Software
Repositories (MSR) , pages 511–515.
Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia
Yan, Tianjun Zhang, Sida Wang, Armando Solar-
Lezama, Koushik Sen, and Ion Stoica. 2024. Live-
codebench: Holistic and contamination free eval-
uation of large language models for code. arXiv
preprint arXiv:2403.07974 .
Carlos E Jimenez, John Yang, Alexander Wettig,
Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R
Narasimhan. 2024. SWE-bench: Can language mod-
els resolve real-world github issues? In The Twelfth
International Conference on Learning Representa-
tions .
Haolin Jin, Zechao Sun, and Huaming Chen. 2024. Rgd:
Multi-llm based agent debugger via refinement and
generation guidance. Preprint , arXiv:2410.01242.
Kelsey Jordahl, Joris Van den Bossche, Martin Fleis-
chmann, Jacob Wasserman, James McBride, Jeffrey
Gerard, Jeff Tratner, Matthew Perry, Adrian Garcia
Badaracco, Carson Farmer, Geir Arne Hjelle, Alan D.
Snow, Micah Cochran, Sean Gillies, Lucas Culbert-
son, Matt Bartos, Nick Eubank, maxalbert, Aleksey
Bilogur, and 11 others. 2020. geopandas/geopandas:
v0.8.1.
Kat Kampf. 2025. Create and edit images with Gemini
2.0 in preview. https://developers.googleblo
g.com/en/generate-images-gemini-2-0-flash
-preview/ . Discusses Gemini 2.0 Flash.
Paul Kassianik, Baturay Saglam, Alexander Chen,
Blaine Nelson, Anu Vellore, Massimo Aufiero, Fraser
Burch, Dhruv Kedia, Avi Zohary, Sajana Weer-
awardhena, Aman Priyanshu, Adam Swanda, Amy
Chang, Hyrum Anderson, Kojin Oshiba, Omar San-
tos, Yaron Singer, and Amin Karbasi. 2025. Llama-
3.1-FoundationAI-SecurityLLM-Base-8B Technical
Report. arXiv preprint arXiv:2504.21039 . Cited for
Llama 3.1 Instruct Turbo.
Sachit Kuhar, Wasi Uddin Ahmad, Zijian Wang, Ni-
hal Jain, Haifeng Qian, Baishakhi Ray, Murali Kr-
ishna Ramanathan, Xiaofei Ma, and Anoop Deo-
ras. 2024. Libevolutioneval: A benchmark andstudy for version-specific code generation. Preprint ,
arXiv:2412.04478.
Stefano Lambiase, Gemma Catolino, Fabio Palomba,
Filomena Ferrucci, and Daniel Russo. 2025. Ex-
ploring individual factors in the adoption of llms
for specific software engineering tasks. Preprint ,
arXiv:2504.02553.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented Generation for Knowledge-
Intensive NLP Tasks. In Advances in Neural Infor-
mation Processing Systems , volume 33.
James Li. 2024. ReAct vs Plan-and-Execute: A Prac-
tical Comparison of LLM Agent Patterns. https:
//dev.to/jamesli .
Linxi Liang, Jing Gong, Mingwei Liu, Chong Wang,
Guangsheng Ou, Yanlin Wang, Xin Peng, and Zibin
Zheng. 2025. Rustevo: An evolving benchmark
for api evolution in llm-based rust code generation.
Preprint , arXiv:2503.16922.
Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen,
Jhonathan Osin, Itay Dalmedigos, Erez Safahi,
Shaked Meirom, Yonatan Belinkov, Shai Shalev-
Shwartz, Omri Abend, Raz Alon, Tomer Asida,
Amir Bergman, Roman Glozman, Michael Gokhman,
Avashalom Manevich, Nir Ratner, Noam Rozen, and
3 others. 2024. Jamba: A hybrid transformer-mamba
language model. Preprint , arXiv:2403.19887.
Yue Liu, Chakkrit Tantithamthavorn, Yonghui Liu,
Patanamon Thongtanunam, and Li Li. 2024. Au-
tomatically recommend code updates: Are we there
yet? Preprint , arXiv:2209.07048.
Zeyu Leo Liu, Shrey Pandit, Xi Ye, Eunsol Choi, and
Greg Durrett. 2025. Codeupdatearena: Benchmark-
ing knowledge editing on API updates.
Edward Loper and Steven Bird. 2002. NLTK: The natu-
ral language toolkit. In Proceedings of the ACL-02
Workshop on Effective Tools and Methodologies for
Teaching Natural Language Processing and Com-
putational Linguistics , pages 63–70, Philadelphia,
Pennsylvania, USA. Association for Computational
Linguistics.
Stephan Lukasczyk and Gordon Fraser. 2022. Pyn-
guin: Automated unit test generation for python. In
Proceedings of the ACM/IEEE 44th International
Conference on Software Engineering: Companion
Proceedings , pages 168–172.
Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xi-
ubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma,
Qingwei Lin, and Daxin Jiang. 2023. WizardCoder:
Empowering code large language models with Evol-
Instruct.
11

Wes McKinney. 2010. Data Structures for Statistical
Computing in Python. In Proceedings of the 9th
Python in Science Conference , pages 51–56.
Mistral AI. 2025. Medium is the new large: Introducing
mistral medium 3. https://mistral.ai/news/mi
stral-medium-3 . Accessed: 2025-05-17.
OpenAI. 2024a. GPT-4o System Card. arXiv preprint
arXiv:2410.21276 . Cited for GPT-4o.
OpenAI. 2024b. OpenAI o1 System Card. https:
//openai.com/index/openai-o1-system-car
d/. Discusses the o1 model series, including o1 and
mentioning o3-mini.
OpenAI. 2025a. Introducing GPT-4.1 in the API. ht
tps://openai.com/index/gpt-4-1/ . Discusses
GPT-4.1, GPT-4.1 mini, and GPT-4.1 nano.
OpenAI. 2025b. Introducing GPT-4.5. https://open
ai.com/index/introducing-gpt-4-5/ .
Adam Paszke, Sam Gross, Francisco Massa, Adam
Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca
Antiga, Alban Desmaison, Andreas Köpf, Edward
Yang, Zach DeVito, Martin Raison, Alykhan Te-
jani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang,
and 2 others. 2019. PyTorch: An imperative style,
high-performance deep learning library. Preprint ,
arXiv:1912.01703.
Perplexity AI. 2024. Getting started with Perplexity.
https://www.perplexity.ai/hub/blog/getti
ng-started-with-perplexity .
Qwen, :, An Yang, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan
Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Yang, Jiaxi Yang, Jingren Zhou, and 25 oth-
ers. 2025. Qwen2.5 technical report. Preprint ,
arXiv:2412.15115.
Rafiqul Rabin, Jesse Hostetler, Sean McGregor, Brett
Weir, and Nick Judd. 2025. Sandboxeval: To-
wards securing test environment for untrusted code.
Preprint , arXiv:2504.00018.
Reka. RekaAI/reka-flash-3 ·Hugging Face — hugging-
face.co. https://huggingface.co/RekaAI/reka
-flash-3 . [Accessed 15-07-2025].
Aymeric Roucher, Albert Villanova del Moral, Thomas
Wolf, Leandro von Werra, and Erik Kaunismäki.
2025. ‘smolagents‘: a smol library to build great
agentic systems. https://github.com/huggingfa
ce/smolagents .
Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Ta-
laei Khoei. 2025. Agentic retrieval-augmented
generation: A survey on agentic rag. Preprint ,
arXiv:2501.09136.Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha,
Bo Wen, and Yunfeng Liu. 2023. Roformer: En-
hanced transformer with rotary position embedding.
Preprint , arXiv:2104.09864.
Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan
Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer,
Damien Vincent, Zhufeng Pan, Shibo Wang, Soroosh
Mariooryad, Yifan Ding, Xinyang Geng, Fred Al-
cober, Roy Frostig, Mark Omernick, Lexi Walker,
Cosmin Paduraru, Christina Sorokin, and 1118 oth-
ers. 2024. Gemini 1.5: Unlocking multimodal un-
derstanding across millions of tokens of context.
Preprint , arXiv:2403.05530.
The pandas development team. 2020. pandas-
dev/pandas: Pandas.
Pauli Virtanen, Ralf Gommers, Travis E Oliphant, Matt
Haberland, Tyler Reddy, David Cournapeau, Ev-
geni Burovski, Pearu Peterson, Warren Weckesser,
Jonathan Bright, and 1 others. 2020. Scipy 1.0:
fundamental algorithms for scientific computing in
python. Nature methods , 17(3):261–272.
Chaozheng Wang, Shuzheng Gao, Cuiyun Gao, Wenx-
uan Wang, Chun Yong Chong, Shan Gao, and
Michael R. Lyu. 2024a. A systematic evaluation
of large code models in api suggestion: When, which,
and how. Preprint , arXiv:2409.13178.
Chong Wang, Kaifeng Huang, Jian Zhang, Yebo Feng,
Lyuye Zhang, Yang Liu, and Xin Peng. 2024b.
How and Why LLMs Use Deprecated APIs in Code
Completion? an Empirical Study. arXiv preprint
arXiv:2312.14617 .
Chong Wang, Kaifeng Huang, Jian Zhang, Yebo Feng,
Lyuye Zhang, Yang Liu, and Xin Peng. 2025a.
LLMs Meet Library Evolution: Evaluating Depre-
cated API Usage in LLM-based Code Completion .
In2025 IEEE/ACM 47th International Conference on
Software Engineering (ICSE) , pages 781–781, Los
Alamitos, CA, USA. IEEE Computer Society.
Chong Wang, Kaifeng Huang, Jian Zhang, Yebo Feng,
Lyuye Zhang, Yang Liu, and Xin Peng. 2025b.
Llms meet library evolution: Evaluating deprecated
api usage in llm-based code completion. Preprint ,
arXiv:2406.09834.
Xingyao Wang. 2025. Introducing openhands lm 32b
– a strong, open coding agent model. All Hands AI
Blog .
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and
Denny Zhou. 2023. Chain-of-thought prompting elic-
its reasoning in large language models. Preprint ,
arXiv:2201.11903.
Tongtong Wu, Weigang Wu, Xingyu Wang, Kang Xu,
Suyu Ma, Bo Jiang, Ping Yang, Zhenchang Xing,
Yuan-Fang Li, and Gholamreza Haffari. 2024. Versi-
Code: Towards version-controllable code generation.
12

xAI. 2025. Grok-3. Official xAI announcement. Ac-
cessed May 17, 2025.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, Chujie Zheng, Day-
iheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao
Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41
others. 2025. Qwen3 technical report. Preprint ,
arXiv:2505.09388.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik R Narasimhan, and Yuan Cao. 2023.
React: Synergizing reasoning and acting in language
models. In The Eleventh International Conference
on Learning Representations .
Sixiang Ye, Zeyu Sun, Guoqing Wang, Liwei
Guo, Qingyuan Liang, Zheng Li, and Yong Liu.
2025. Prompt alchemy: Automatic prompt re-
finement for enhancing code generation. Preprint ,
arXiv:2503.11085.
Shuyan Zhou, Uri Alon, Frank F Xu, Zhiruo
Wang, Zhengbao Jiang, and Graham Neubig. 2022.
DocPrompting: Generating code by retrieving the
docs.
13

A Benchmark Details
This appendix provides additional details on the
GitChameleon benchmark. We provide details on
the dataset construction process, the structure of
the dataset samples, on the processes for validating
the examples and constructing the hidden tests, and
finally present additional statistics regarding the
dataset.
A.1 Dataset Construction Process
The examples were created by the authors, which
took roughly 350 human hours. To construct that
dataset, we compiled a list of popular Python li-
braries, focusing on those that had more than 1000
stars on Github as well as detailed documentation
of changes between versions. For each library,
we reviewed the change logs to identify breaking
changes: deprecated functions, argument changes,
alterations in behavior, and newly introduced func-
tions.
For each identified change, we wrote a concise
problem statement, starter code, expected solution
and a suite of tests, consisting of a comprehen-
sive suite of hidden tests to be used for model
performance evaluation and ranking and a man-
ually written concise visible test to be used for self-
debugging experiments. We also added a ground-
truth set of relevant documents for RAG experi-
ments.
NOTE:Low-level changes—such as backend
optimizations that do not alter the surface-level
API—are not considered valid changes for our
benchmark. For example, if between Torch 1.7
andTorch 1.8 thetorch.nn.Softmax() func-
tion received a CUDA-based numerical stability im-
provement, this does not modify the API usage of
Softmax() and is therefore not labeled as a change
in our benchmark. Since most changes in mature
libraries primarily impact backend functionality,
collecting 328 valid samples required significant
effort.
A.2 Structure of Dataset Samples
The main fields of each sample are given in Table 5.
Additionally, each problem in GitChameleon is
associated with metadata to assist in the analysis of
the results, as described in Table 6. Each problem
is classified with a type of API evolution change
among the categories defined in Table 7.Library The software library under test.
Library Version The exact version of that library.
Task Description A problem centered on a particular library
change.
Initial Code The Python snippet provided as a starting
point.
Extra Dependencies Any additional packages required to solve the
task.
Hidden Tests Comprehensive unit tests designed to maxi-
mize coverage. The success rate on these is
the benchmark metric.
Visible Test A concise test that validates the specific tar-
get behavior, intended to be used for Self-
Debugging experiments.
Reference Solution A correct, ground-truth implementation.
Reference Documents A set of version-specific reference documents,
to be used for RAG experiments.
Table 5: Problem column definitions for the
GitChameleon dataset.
Change Category The type of library-evolution changes, as de-
fined in table 7.
Target Entity The specific function or class under test.
Solution Style “Functional” if only a function body is ex-
pected, or “Full” for a general code comple-
tion.
Web Framework Task “Yes” if the problem exercises a web-
development framework, otherwise “No.”
Table 6: Metadata column definitions.
A.3 Dataset Validation
To ensure the validity of the dataset examples, we
followed the following process: First, we created
a clean Docker container for each problem and
installed the required dependencies into it. Then,
we executed the visible and hidden validation tests
to ensure that all are successful.
A.4 Hidden Test Construction
This section presents how we generated the hid-
den tests for each dataset example. These tests
were generated by instructing the Zencoder AI
Coding Agent9to create test files for each ex-
ample, incorporating the appropriate dependency
versions. The Zencoder agent, built on the GPT-4.1
base model, operated with internet search enabled
and was granted execution access, allowing it to
self-correct outputs that initially failed during run-
time. Further errors encountered during verifica-
tion were resolved by supplying error traces back
to Zencoder or through an isolated instance of GPT-
4o, supplemented with manual intervention where
necessary. This process enabled us to construct
a robust and comprehensive test suite, achieving
a coverage of 96.5% . The decision to use ZEN-
CODER was motivated by limitations observed in
9https://zencoder.ai
14

Change Category Description
Argument or Attribute
changeThe API call to a function, method, or class
has a change in arguments (e.g. name, order,
new, deprecated argument) between versions.
Function Name change The name of the API call has changed be-
tween versions (e.g. pandas.append to
pandas.concat ).
Semantics or Function
Behavior changeThe semantic / runtime behavior of the API
call changed between versions (e.g. returning
a different type).
New feature or addi-
tional dependency-based
changeA feature was introduced in a specific ver-
sion; therefore, to execute the same function-
ality, a model using an older version should
make use of an additional dependency (e.g.
torch.special was introduced in TORCH
1.10 , previously one could use NUMPY for
the same).
Table 7: Categories of API Evolution Changes
alternative unit test generation approaches. Rule-
based generators such as Pynguin (Lukasczyk and
Fraser, 2022) fail to account for version differences
among samples that share the same or similar prob-
lem statements. Meanwhile, AI-based unit test gen-
erators like Claude Code and EarlyAI10were not
suitable: the former typically generated test classes
where each sub-function was populated only with
pass() statements, while the latter was restricted
to functional-style problems and could not handle
the more complex, class-based structures prevalent
inGitChameleon .
A.5 Additional Dataset Statistics
Figure 9 presents the number of unique versions
per library and the number of samples per library.
B Extra Methodologies: Reasoning,
Sampling and Prompting
This section presents results from additional exper-
imental methodologies:
•Temperature Sampling: Results are shown
in Table 9. We evaluate sampling at temper-
ature T= 0.8across 10 seeds using both
the OpenAI and Gemini model suites. The
performance difference compared to greedy
decoding is minimal.
•Reasoning Models: Performance results for
the OpenAI o-series reasoning models are pro-
vided in Table 8.
•Self-Explained Keywords (SEK) Prompt-
ing: We evaluate the SEK prompting method
proposed by Fan et al. (2024), applied to both
OpenAI and Gemini models. SEK involves
10https://www.startearly.ai/
scipytorch
networkxgradiosympyplotly
matplotlibpandaslibrosa
geopandasnltk
scikit-learnflask
tornadonumpylightgbmdjangojinja2
seaborn
mitmproxypytestfalconpillowtqdmspacykymatio
Library01234567Number of Unique Versions(a) Number of unique versions per library.
librosascipysympyflaskfalcontorchnumpy
scikit-learnpandasdjango
mitmproxynetworkxlightgbmpytestplotly
geopandasseaborntornadogradionltk
matplotlibpillowjinja2spacytqdm
kymatio
Library0510152025303540Number of Samples
(b) Number of samples per library.
Figure 9: Dataset library statistics. (a) The count of
distinct versions identified for each library, presented in
decreasing order of uniqueness. (b) The total frequency
of samples containing each library, ordered by their
occurrence count.
a two-stage process: (1) Keyword Extraction ,
where the model generates relevant keywords
for the coding task, and (2) Keyword Cate-
gorization , where keywords are ranked and
classified into (a) Function, (b) General, and
(c) Abstract categories. TF-IDF ranking is per-
formed using a 50,000-document subset of the
EVOL -CODEALPACA -V1corpus (Luo et al.,
2023). As shown in our empirical analysis,
SEK does not yield significant improvements
over greedy sampling, and in several cases
underperforms relative to it. NOTE:Temper-
ature T= 0 is used in both stages of SEK
prompting.
C Extended Experiment Results and
Analysis
This section contains the following additional ex-
perimental results:
•An experiment on Automatic Prompt Opti-
mization of the system prompt for Greedy De-
15

ModelVanilla Decoding Vanilla with Self-Debug Zero-shot CoT
Success
Rate (%)API
Hit
Rate (%)Success
Rate (%)API
Hit
Rate (%)Success
Rate (%)API
Hit
Rate (%)Hidden Visible Hidden Visible Hidden
o1 51.2±2.8 60.1±2.7 42.1±2.7 57.6±2.7 68.6±2.6 49.2±2.8 41.2±2.7 41.3±2.7
o3-mini 44.5±2.7 52.7±2.8 40.6±2.7 66.8±2.6 76.5±2.3 45.7±2.8 50.9±2.8 40.7±2.7
o4-mini 48.2±2.8 57.0±2.7 48.3±2.8 63.1±2.7 75.0±2.4 45.4±2.7 – –
codex-mini 48.5±2.8 58.2±2.7 47.5±2.8 – – – 32.0±2.6 37.9±2.7
Table 8: Success rate on visible and hidden tests and API hit rate under the Vanilla, Self-Debug, and Zero-shot CoT
settings, for the OpenAI o-series models. Model ranking on the benchmark is determined by Hidden Success Rate .
Visible Success Rate figures are for context on Self-Debugging. The best result in each column is in bold. For full
model details and citations, please refer to Appendix J.
coding is described in Table 11.
•An experiment on static analysis based gener-
ated solutions fixing to ensure model failures
are not attributed to confounding factors like
indentation problems and unused imports or
variable declarations. Refer to Table 13 for
further details.
•Table 12 contains an extended set of RAG
results, including both additional models and
the setting where only a single document is
retrieved.
We also present the following additional analy-
ses:
•A comparison of success rates between Self-
Debug and Greedy Decoding, when broken
down by version release year (Figure 10) and
by library (Figure 11).
•A comparison of success rates between RAG
and Greedy Decoding by library is shown in
Figure 12.
•Figure 13 analyzes the intra-model sample
agreement rates in the Greedy Decoding, Zero-
Shot CoT and RAG settings.ModelHidden Success
Rate (%)API Hit
Rate (%)
o1 50.5±0.8 44.0±0.8
o3-mini 46.4 ±1.6 42.5±0.6
GPT-4.1 48.9 ±1.4 48.1±1.0
GPT-4.1-mini 45.9 ±1.3 46.9±0.6
GPT-4.1-nano 33.8 ±1.1 43.8±0.8
GPT-4o 47.2 ±1.2 45.1±0.9
GPT-4o-mini 40.2 ±1.2 41.0±1.1
Gemini 1.5 Pro 45.4 ±1.2 45.5±0.7
Gemini 2.5 Pro 41.0 ±3.4 48.3±1.7
Gemini 2.0 Flash 43.4 ±3.1 42.5±0.9
Gemini 2.5 Flash 46.4 ±0.8 46.8±1.2
Table 9: Hidden Success Rate using temperature sam-
pling ( T= 0.8), averaged over 10 seeds. A compari-
son to the greedy decoding baseline in Table 1 reveals
that the changes in performance between greedy de-
coding and temperature sampling are mixed. For most
models, the differences are small, but for a few specific
models, the changes are big and noteworthy. For the
majority of models evaluated (8 out of 11), the perfor-
mance change is minor, typically within +/- 2 percentage
points. For example, Gemini-2.5-pro , shows a notable
decrease in success rate (-9.0 points).
16

0.2 0.4 0.6 0.8 1.0
Success RateClaude 3.7
Sonnet
Gemini 1.5
Pro
Gemini 2.5
Pro
GPT-4.1
O-1
DDG SB
Claude 3.5
Goose2021
0.0 0.2 0.4 0.6 0.8 1.0
Success Rate2022
0.0 0.2 0.4 0.6 0.8 1.0
Success Rate2023
Vanilla Self-DebugFigure 10: Success Rate Breakdown by Version Release Year . Lighter and darker shaded bars represent values
obtained with and without Self-Debugging, respectively. Standard error is drawn as a black line. This plot shows
that the release year does not significantly impact the results for most evaluated settings.
ModelHidden Success
Rate (%)API
Hit Rate (%)
GPT-4o 29.6 ±2.5 43.6±2.7
GPT-4o-mini 27.7 ±2.5 40.3±2.7
GPT-4.1 43.6 ±2.7 49.4±2.8
GPT-4.1-mini 41.2 ±2.7 44.0±2.7
GPT-4.1-nano 32.9 ±2.6 43.8±2.7
GPT-4.5 33.8 ±2.6 58.0±2.7
Gemini 1.5 Pro 44.5 ±2.7 45.7±2.8
Gemini 2.0 Flash 41.2 ±2.7 43.4±2.7
Gemini 2.5 Pro 47.3 ±2.8 50.0±2.8
Gemini 2.5 Flash 48.2±2.8 43.4±2.7
Table 10: Success and API hit rates under the SEK
setting. While SEK, being a two-round prompting
scheme, is expected to outperform greedy decoding,
we observe that it does not yield significant improve-
ments. For example, with GPT-4.1, the success rate
actually drops by 4.9% when using SEK compared to
greedy decoding.Model Best Round Success Rate (%) ∆(%)
GPT-4.1-mini 1 42.1 ±2.7 –2.1
GPT-4.1-nano 3 37.5 ±2.7 +3.7
GPT-4.1 1 50.0±2.8 +1.5
GPT-4o 0 49.1 ±2.8 0.0
Table 11: Automatic System Prompt Optimization
results . The prompt was optimized for at most 5 rounds
using the method described in (Ye et al., 2025), with
early stopping if the improvement over previous round
is less than 1.5%. We used GPT-4.1 as the mutation
model and a random fixed 20% subset of the dataset for
the optimization process. For the initial prompt, we use
the same system prompt that we had used for our Greedy
Decoding experiments, as given in Figure 17. We report
the delta of the hidden test success rate, in comparison to
the Greedy Decoding baseline. The results demonstrate
the limited utility of further optimizing the prompts we
had used in our experiments.
17

Model k= 1 k= 3
Success
Rate (%)API Hit
Rate (%)Success
Rate (%)API Hit
Rate (%)Precision
(%)Recall
(%) MRR
Open-Weights Models
CommandA 43.6 ±2.7 43.9±2.7 48.2±2.8 45.4±2.7 41.9±2.7 50.7±2.80.63±0.03
CommandR 7B 23.2 ±2.3 36.3±2.7 23.2±2.3 35.6±2.6 41.6±2.7 50.4±2.80.62±0.03
Deepseek R1 50.9±2.8 44.8±2.7 51.2±2.8 47.9±2.8 41.5±2.7 50.1±2.80.62±0.03
Reka Flash-3 8.5 ±1.5 34.5±2.6 11.6±1.8 31.9±2.6 29.9±2.5 39.6±2.80.47±0.03
Jamba 1.6 Mini 18.0 ±2.1 35.4±2.6 29.3±2.5 40.4±2.7 41.6±2.7 50.1±2.80.62±0.03
OpenHands LM 32B v0.1 34.8 ±2.6 41.0±2.7 28.9±2.5 36.5±2.7 25.9±2.4 33.7±2.70.42±0.03
Llama 4 Scout 38.7 ±2.7 45.1±2.7 39.3±2.7 43.6±2.7 41.3±2.7 50.4±2.80.62±0.03
Enterprise Models
Arcee CoderL 46.3 ±2.8 47.3±2.8 36.6±2.7 40.4±2.7 31.1±2.6 41.0±2.80.49±0.03
Claude 3.5 Haiku 43.6 ±2.7 47.9±2.8 43.0±2.7 47.5±2.8 41.9±2.7 50.7±2.80.62±0.03
Claude 3.5 Sonnet 8.5 ±1.5 18.6±2.1 49.4±2.8 51.5±2.8 41.9±2.7 50.7±2.80.62±0.03
Codestral 44.2 ±2.7 47.3±2.8 46.0±2.8 48.5±2.8 41.9±2.7 50.7±2.80.62±0.03
CommandR+ 32.0 ±2.6 43.0±2.7 36.6±2.7 41.9±2.7 41.6±2.7 50.4±2.80.62±0.03
Gemini 2.5 Flash 54.3±2.8 50.5±2.8 55.2±2.8 51.2±2.8 41.9±2.7 50.7±2.80.62±0.03
GPT-4.1-mini 46.9 ±2.8 50.0±2.8 48.8±2.8 50.0±2.8 41.3±2.7 50.4±2.80.62±0.03
GPT-4.1-nano 38.1 ±2.7 45.1±2.7 37.8±2.7 45.0±2.7 41.3±2.7 50.4±2.80.62±0.03
GPT-4o-mini 41.5 ±2.8 45.4±2.7 43.3±2.8 46.8±2.8 41.0±2.7 50.1±2.80.62±0.03
GPT-4o 48.2 ±2.8 47.0±2.7 52.1±2.8 49.4±2.8 40.6±2.7 49.5±2.80.61±0.03
Inflection 3 Productivity 24.7 ±2.8 42.0±2.6 21.9±2.7 44.2±2.7 41.9±2.7 50.7±2.80.62±0.03
LFM 40B MoE 30.8 ±2.7 38.3±2.7 20.7±2.7 34.0±2.7 33.8±2.7 44.8±2.80.53±0.03
Table 12: RAG performance of additional models when retrieving k= 1 andk= 3 most relevant documents.
Precision is shown only for k= 3as it is equivalent to Recall in the k= 1case. This table shows that retrieving
three documents is better in almost all cases than retrieving a single document, despite the incurred false positives
that arise due to most of the examples having less than three relevant documents.
18

0.0 0.2 0.4 0.6 0.8 1.0
Success RateClaude 3.7 Sonnet
Gemini 1.5 Pro
Gemini 2.5 Pro
GPT-4.1
O-1
DDG SB Claude 3.5
GooseTorch
0.0 0.2 0.4 0.6 0.8 1.0
Success RateNumPy
0.0 0.2 0.4 0.6 0.8 1.0
Success RateSymPy
0.0 0.2 0.4 0.6 0.8 1.0
Success RateClaude 3.7 Sonnet
Gemini 1.5 Pro
Gemini 2.5 Pro
GPT-4.1
O-1
DDG SB Claude 3.5
GooseSciPy
0.0 0.2 0.4 0.6 0.8 1.0
Success RateDjango
0.0 0.2 0.4 0.6 0.8 1.0
Success RateFlask
Vanilla Self-DebugFigure 11: Success Rate Breakdown by Library . This figure shows the differences in success rate between the
libraries included in GitChameleon . All evaluated settings do very well on NumPy, which is to be expected given
the popularity of the library and the subsequent abundance of code that uses it. The success rates on the web
development frameworks are notably lower than on the scientific computing libraries, perhaps due to having more
complex abstractions.
19

10%
 5%
 0% 5% 10% 15% 20%
Success Rate Liftpandasnumpytorchscipyflaskdjangolibrosasympyscikit-learnfalcon(a)GPT-4.1
20%
 10%
 0% 10% 20% 30% 40%
Success Rate Liftflasknumpytorchscipypandasfalconlibrosasympydjangoscikit-learn (b)GPT-4.1-mini
20%
 10%
 0% 10% 20% 30% 40% 50%
Success Rate Liftdjangoflasktorchpandasscipyfalconnumpylibrosasympyscikit-learn
(c)GPT-4.1-nano
Figure 12: ∆Success Rate of RAG over Greedy Decoding, per library . The 10 most frequent libraries in
GitChameleon are shown here. The plots demonstrate a trend where smaller models are less effective at using
RAG, with the full-size GPT-4.1 improving on 7 libraries, the mini version improving on 5 and the nano version
improving only on 3.
20

Claude 3.7 SonnetGemini 1.5 ProGemini 2.0 FlashGemini 2.5 ProGPT 4.1
GPT 4.1-miniGPT 4.1-nanoGPT 4.5 GPT-4o
GPT-4o-mini
Llama4 MaverickClaude 3.7 Sonnet
Gemini 1.5 Pro
Gemini 2.0 Flash
Gemini 2.5 Pro
GPT 4.1
GPT 4.1-mini
GPT 4.1-nano
GPT 4.5
GPT-4o
GPT-4o-mini
Llama4 Maverick0.75 0.78 0.76 0.80 0.78 0.75 0.71 0.81 0.77 0.71
0.73 0.75 0.78 0.72 0.74 0.70 0.77 0.75 0.76
0.74 0.74 0.75 0.74 0.76 0.76 0.77 0.70
0.76 0.75 0.70 0.70 0.73 0.70 0.72
0.80 0.75 0.72 0.84 0.78 0.73
0.77 0.73 0.77 0.75 0.72
0.71 0.76 0.78 0.69
0.73 0.73 0.71
0.80 0.72
0.76
0.700.720.740.760.780.800.82(a)Greedy Decoding
Gemini 1.5 ProGemini 2.0 FlashGemini 2.5 ProGPT 4.1
GPT 4.1-mini GPT 4.1-nanoGPT 4.5 GPT-4o
GPT-4o-mini
Llama4 MaverickGemini 1.5 Pro
Gemini 2.0 Flash
Gemini 2.5 Pro
GPT 4.1
GPT 4.1-mini
GPT 4.1-nano
GPT 4.5
GPT-4o
GPT-4o-mini
Llama4 Maverick0.75 0.77 0.79 0.69 0.64 0.75 0.79 0.77 0.81
0.70 0.74 0.64 0.70 0.69 0.72 0.77 0.70
0.79 0.62 0.58 0.74 0.78 0.70 0.77
0.66 0.60 0.77 0.84 0.76 0.79
0.74 0.69 0.66 0.69 0.66
0.65 0.56 0.70 0.57
0.76 0.74 0.75
0.76 0.80
0.71
0.600.650.700.750.80 (b)Zero-Shot Chain-Of-Thought
Claude 3.7 SonnetGemini 2.5 FlashGemini 2.5 ProGPT 4.1
GPT 4.1-mini GPT 4.1-nanoGPT-4o-miniGrok 3
Llama4 MaverickClaude 3.7 Sonnet
Gemini 2.5 Flash
Gemini 2.5 Pro
GPT 4.1
GPT 4.1-mini
GPT 4.1-nano
GPT-4o-mini
Grok 3
Llama4 Maverick0.81 0.81 0.79 0.80 0.74 0.77 0.79 0.77
0.86 0.81 0.79 0.73 0.80 0.80 0.79
0.83 0.76 0.71 0.75 0.76 0.74
0.83 0.73 0.79 0.82 0.78
0.78 0.82 0.80 0.80
0.82 0.73 0.78
0.81 0.79
0.78
0.720.740.760.780.800.820.840.86
(c)RAG (k=3)
Figure 13: Intra model sample agreement rates . These plots show the rate of samples that have the same pass/fail
result among all pairs of models, under the Greedy Decoding, Zero-Shot CoT and RAG settings. Each cell in these
plots represents the agreement rate of a pair of models, with the rate also being color-coded. The high agreement
rates in all three subfigures show that ensembling different models would have a limited effect on the success rates.
21

Assistant Model LinterPylint
Score↑Success
Rate (%)
Cline (IDE) GPT-4.1N/A 1.06 54.6 ±2.8
Black + Isort 1.69 54.6 ±2.8
Ruff 2.64 54.6±2.8
Goose (CLI) GPT-4oN/A 0.53 36.3 ±2.7
Black + Isort 1.82 36.3 ±2.7
Ruff 2.92 36.3±2.7
Claude
Code (CLI)Claude
3.7 SonnetN/A 0.00 48.8 ±2.8
Black + Isort 1.92 48.8 ±2.8
Ruff 2.60 48.8±2.8
Table 13: Static Analysis and Auto-
linting/Formatting. Pylint11scores are averaged across
code samples and are scored out of 10. The success rate
numbers presented are the same as in Table 3 wherein
Goose has no access to problem statement while Cline
and Claude are provided with the same. We observe that
the original generated solutions via coding assistants
do not meet minimum quality standard requirements,
however when improved via auto-linters like Black12,
ISort13and Ruff14, their code quality improves but
with no impact to the success rate. This demonstrates
that there are no confounding errors like indentation
issues, unused imports and other formatting issues
influencing our evaluation results observed. NOTE: For
Ruff formatting, we used the already formatted/ linted
solutions via Black and ISort.
D Related Work
D.1 Code Evolution Datasets
While the main text provides a high-level overview
of the most similar benchmarks, this section
offers a more detailed differentiation between
GitChameleon and other relevant works. We cat-
egorize these benchmarks based on several key
dimensions, including their evaluation method
(execution-based vs. non-executable) and, most
importantly, their core task format (instruction-
based generation vs. completion- or repair-
based tasks) . This distinction is critical as it tests
different capabilities of language models.
11https://pylint.pycqa.org/en/latest/index.html
12https://black.readthedocs.io/en/stable/
13https://pycqa.github.io/isort/
14https://docs.astral.sh/ruff/D.1.1 Task Format: Instruction-Based
Generation
GitChameleon is fundamentally an instruction-
based benchmark. For each problem, the model is
given a natural language "Problem Statement" and
starter code. The core challenge is to comprehend
the user’s intent and generate a new, functionally
correct solution that adheres to specific version
constraints. This tests a model’s ability to translate
human requirements into code.
D.1.2 Task Format: Code Update, Repair, and
Completion
In contrast, many other benchmarks focus on tasks
where the primary input is existing code, not a
natural language instruction. The model’s goal is
to modify, repair, or complete a given code snippet.
Code Update and Repair Benchmarks A sig-
nificant body of work evaluates a model’s ability to
modify or repair existing code.
•CodeUpdateEval (Liu et al., 2024) and
JavaVersionGenBench (Ciniselli et al., 2024)
are code modification benchmarks for Python
and Java, respectively. They provide a model
with a working piece of code and require it to
be updated to a newer library version.
•RustEvo2 (Liang et al., 2025) is a code re-
pair benchmark for Rust. It provides a model
with code that is broken due to a dependency
update and asks it to generate a fix based on
compiler errors.
These tasks are distinct from GitChameleon ’s, as
they test a reactive, corrective capability rather than
the proactive generation of new code from a speci-
fication.
Completion-Based and Non-Executable Bench-
marks Another category of benchmarks uses
non-executable metrics or focuses on code com-
pletion.
•LibEvolutionEval (Kuhar et al., 2024) is
a non-executable benchmark structured as a
"fill-in-the-middle" completion-based task .
Its evaluation is based on textual similarity
metrics (e.g., F1 score), not the functional cor-
rectness of the code.
•LLM-Deprecated-APl (Wang et al., 2025b),
which we note in our introduction, focuses on
replacing deprecated APIs. This is a specific
22

Benchmark Language Evaluation Method Core Task Source of Changes Key Differentiator from
GitChameleon
GitChameleon Python Execution-Based Generation for a static ver-
sion: Writes new code for a
specific, often older, library
version.Real, documented histori-
cal breaking changes.(Baseline for comparison)
CodeUpdateEval Python Execution-Based Code Updating : Modifies ex-
isting code to work with a
newer library version.Real-world software up-
date commits.Focuses on migrating code for-
ward to a newer version, not gen-
erating for a static one.
JavaVersionGenBench Java Execution-Based Code Updating : Modifies ex-
isting Java code to handle ver-
sion updates.Real-world Java projects. Focuses on the Java ecosystem
and its specific language/tooling
challenges.
LLM-Deprecated-APl Python Non-Executable Deprecation Fixing : Identi-
fies and replaces specific dep-
recated API calls.A curated list of depre-
cated APIs.Uses a non-executable evalua-
tion method and has a narrow
scope focused only on API dep-
recation.
LibEvolutionEval Python Non-Executable Code Completion : Fills in a
missing part of a code snippet
based on context.API documentation and re-
lease notes.Is acompletion-based task that
does not test functional correct-
ness through execution.
RustEvo2 Rust Execution-Based Code Repair : Fixes existing
code that fails to compile after
a dependency update.Real breaking changes
from Rust libraries
("crates").Focuses on the Rust ecosystem
and a reactive, compiler-error-
driven repair task.
CODEMENV Python Execution-Based Environment Compatibility:
Generates code that is com-
patible with a complex envi-
ronment specification.A broad set of environ-
ment configurations.Has a broader focus on overall
environment compatibility, not
specifically on historical break-
ing changes.
Table 14: Detailed comparison of GitChameleon with related benchmarks across several key dimensions, highlight-
ing differences in evaluation methodology, core task, and primary programming language.
type of repair task that is evaluated using non-
executable string matching.
•CODEMENV (Cheng et al., 2025) evaluates
a model’s ability to generate code compat-
ible with a complex environment specifica-
tion. While execution-based, its task is primar-
ily driven by satisfying technical constraints
rather than implementing a distinct, high-level
natural language instruction.
For a detailed breakdown, Table 14 contrasts
GitChameleon with these related benchmarks
across several key methodological dimensions.
D.2 Specialized Frameworks and Repair
Techniques
Recognizing the unique challenges of library evo-
lution, researchers and practitioners are develop-
ing specialized frameworks and automated repair
techniques that often combine LLMs with other
methods.
D.2.1 DepsRAG
This framework utilizes a multi-agent system built
around RAG and Knowledge Graphs specifically
for reasoning about software dependencies (Al-
hanahnah et al., 2024). It employs distinct agents
managed by an LLM: one to construct and querythe dependency KG, another for web searches, and
a critic agent to review and refine the generated
responses, aiming for higher accuracy in complex
dependency analysis tasks.
D.2.2 Dr.Fix
This tool represents a family of approaches using
LLMs, often combined with program analysis and
RAG, for automated program repair. It focuses on
fixing API misuse in LLM-generated code based on
the taxonomy of misuse types. It employs a detect-
reason-fix pipeline and demonstrates substantial
improvements in repair accuracy metrics such as
BLEU and Exact Match (Behrang et al., 2025).
D.2.3 ReplaceAPI / InsertPrompt
These are lightweight, targeted techniques de-
signed specifically to mitigate the use of depre-
cated APIs in LLM-based code completion. Re-
placeAPI performs a direct, post-generation sub-
stitution of known deprecated API calls with their
replacements, achieving high fix rates in evalua-
tions (Wang et al., 2025b). InsertPrompt modifies
the input prompt to discourage the generation of
deprecated APIs in the first place. They serve as
valuable baseline approaches for this specific prob-
lem (Wang et al., 2025a).
23

D.2.4 Conclusion
These works indicate a trend towards hybrid and
agentic systems, moving beyond single LLM calls
to more sophisticated architectures that integrate
LLMs with other methods for handling library evo-
lution. GitChameleon serves as an essential re-
source for evaluating such systems.
In the subsequent sections we present qualitative
sample analyses and model generation differences.
E Case Study: Code Assistant Failure
With Search
Here we present an example where a coding as-
sistant (Zencoder) fails at solving a problem in
our benchmark even when it can search the in-
ternet for the correct API call. The task re-
quired using the torch.special module or equiv-
alent scipy.special to calculate the the modi-
fied Bessel function of the first kind of order 0.
This can be computed using torch.special.i0
orscipy.special.i0 if the version pre-dates the
introduction of torch.special .
E.1 Inputs
Problem Statement: Calculate the modified
Bessel function of the first kind, order 0 using py-
torch’s special functions if available in this version,
otherwise you may use another library.
Environment
• Python Version: 3.7
• Pytorch Version: 1.10
Starting Code
import torch
def bessel _i0( input _ tensor : torch . Tensor
) -> torch . Tensor :
Listing 1: Initial Python function signature provided.
E.2 Model Attempt and Failure
The model’s attempt involved manually implement-
ing the modified Bessel function, handling small
and large inputs as separate cases.
Assistant Trace
[ truncated ]
** Web search **
Done
Let me check ifthe torch . special module
with Bessel functions is
specifically available inPyTorch
1.10.0.** Web search **
Done
The search results dont specifically
confirm iftorch . special .i0 is
available inPyTorch 1.10.0. Let me
search more specifically .
** Web search **
Done
Since Im not getting clear information
about the availability of torch .
special .i0 inPyTorch 1.10.0 , Ill
implement the function manually
using the mathematical definition of
the modified Bessel function of the
first kind of order 0.
Now , lets implement the solution :
Listing 2: The assistant’s trace while designing the
solution.
Proposed Solution
def bessel _i0( input _ tensor : torch . Tensor
) -> torch . Tensor :
result = torch . zeros _ like ( input _
tensor )
abs_x = torch . abs(input _ tensor )
# For small values (|x| < 15) , use
the series expansion
small _ mask = abs_x < 15.0
small _x = abs_x[ small _ mask ]
ifsmall _x. numel () > 0:
small _ result = torch . ones _ like (
small _x)
term = torch . ones _ like ( small _x)
x_ squared _ over _4 = ( small _x /
2.0) ** 2
# Sum the series until
convergence or max
iterations
for kin range (1, 30) : # 30
terms should be enough for
good precision
term = term * x_ squared _ over
_4 / (k ** 2)
small _ result = small _ result
+ term
# Check for convergence
iftorch . all( term < 1e -10 *
small _ result ):
break
result [ small _ mask ] = small _
result
# For large values (|x| $\ geq$ 15) ,
use the asymptotic approximation
large _ mask = ~ small _ mask
large _x = abs_x[ large _ mask ]
iflarge _x. numel () > 0:
large _ result = torch . exp ( large _x
) / torch . sqrt (2 * math .pi *
24

large _x)
result [ large _ mask ] = large _
result
return result }
Listing 3: The solution proposed by the assistant.
Error Traceback:
TestBesselI0 . test _ large _ values _________
_______________
def test _ large _ values ( self ):
""" Test bessel_i0 with large
positive and negative values . """
input _ tensor = torch . tensor ([10.0 ,
-10.0 , 20.0 , -20.0] , dtype = torch
. float32 )
result = bessel _i0( input _ tensor )
expected = torch . from _ numpy ( scipy _i0
(input _ tensor . numpy ()))
> torch . testing . assert _ close (
result , expected )
E AssertionError : Tensor - likes are
not close !
Listing 4: Error trace for the implementation of
bessel_i0 .
Explanation: The failure in the manual imple-
mentation arises because the asymptotic approxi-
mation
I0(x)≈ex
√
2πx
has only O(1/x)accuracy and omits higher-order
correction terms, so for x≳20its relative error ex-
ceeds the test tolerance of 10−5. To fix this, either
use the built-in torch .special .i0(x)or include
at least the first correction term in the expansion:
I0(x)≈ex
√
2πx
1−1
8x
,
which reduces the large- xerror below the required
threshold.
FCase Study: Self-Debugging in Batched
Matrix Exponential Computation
This appendix details an instance where self-
debugging helped rectify an initially incorrect so-
lution from GPT-4.5 for computing the matrix ex-
ponential of batched matrices. The task required
leveraging the scipy.linalg.expm function while
ensuring compatibility with batched inputs.
F.1 Inputs
Problem Statement: Complete the function that
computes the matrix exponential of batched matri-
ces, non specified parameters should use the default
valueEnvironment:
• Python Version: 3.10
• SciPy Version: 1.8.1
Starting Code
from scipy import linalg
import numpy as np
def compute _ matrix _ exponential (A: np.
ndarray ) -> np. ndarray :
# Model needs to complete this
function
pass
Listing 5: Initial Python function signature provided.
F.2 First Model Attempt and Failure
The model’s first attempt involved reshaping the
input array and applying scipy.linalg.expm di-
rectly.
Initial Model Output
from scipy import llinalg
import numpy as np
def compute _ matrix _ exponential (A: np.
ndarray ) -> np. ndarray :
return linalg . expm (A)
Listing 6: First attempt by the model to define
compute_matrix_exponential .
Model’s Explanation (Initial): The function di-
rectly returns linalg.expm(A) because SciPy’s expm
computes the matrix exponential for a given matrix.
Given that A is a batch of matrices, SciPy should
naturally handle this batch input and apply the ex-
ponential operation to each matrix independently,
producing the correct batched result.
Result: Failure
Error Traceback Failure: Calling
linalg.expm(A) on a 3D array A∈Rb×n×n
makes SciPy treat it as one “matrix,” and raises
ValueError: expected a square matrix
because it only supports 2D inputs.
Traceback ( most recent call last ):
File "/ app / repo / eval _ venvs / gcham _ venv _ 111/ lib /
python3 . 10/ site - packages / scipy / sparse / linalg /
_ matfuncs .py", line 601 , in _ expm
raise ValueError ( 'expected a square matrix ')
ValueError : expected a square matrix
Listing 7: Error traceback from the initial model output.
25

F.3 Self-Debugging Process and Correction
The self-debugging process recognized that sim-
ply calling linalg.expm was insufficient. Instead,
linalg.expm must be applied individually to each
2D matrix within the batch. Additionally, a crucial
pre-condition is that each matrix within the batch
must be square.
Second Model Attempt (Post-Debugging) The
model regenerated the solution, incorporating input
validation and iterating over the individual matrices
in the batch.
Regenerated Model Output
from scipy import linalg
import numpy as np
def compute _ matrix _ exponential (A: np.
ndarray ) -> np. ndarray :
return np. stack ([ linalg . expm (A[i])
for iin range (A. shape [0]) ifA[
i]. shape [0] == A[i]. shape [1]] ,
axis =0)
Listing 8: Corrected compute_matrix_exponential
after self-debugging.
Model’s Explanation (Regenerated): The cor-
rected solution applies linalg.expm separately to
each 2D square matrix in the batch using a list
comprehension, then re-stacks the results into a 3D
array. This ensures each call to expm receives a
valid input, resolving the error.
Result: Success
F.4 Analysis of the Correction
The regenerated output effectively addressed the
shortcomings of the initial attempt:
•Fix: Apply expm to each 2D slice Ai∈Rn×n
and re-stack, ensuring every call gets a proper
square matrix.
This revised approach correctly computes the ma-
trix exponential for each matrix in the batch and
handles the data dimensions appropriately, thus re-
solving the ValueError and fulfilling the problem
requirements.
G Qualitative Analysis
This section presents additional qualitative studies
of attempted solutions in the Greedy Decoding and
the Chain-of-Thought settings.
G.1 Greedy Decoding
G.1.1 Example 1: (PyTorch)
Problem Statement: Calculate the logarithm of
the cumulative distribution function of the standardnormal distribution using available functions. If
not available in PyTorch, use another library.
Environment:
• Python Version: 3.7
• Torch Version: 1.9.0
•Extra Dependencies: scipy==1.7.3,
numpy==1.21.6
Starting Code
import torch
def log _ ndtr ( input _ tensor : torch . Tensor )
-> torch . Tensor :
pass
Listing 9: Greedy Example 1 Starting Code
Solution
import numpy as np
from scipy . stats import norm
output = torch . from _ numpy ( norm .
logcdf ( input _ tensor . numpy ()))
return output
Listing 10: Greedy Example 1 Solution
Claude 3.7 Failure Output
import torch
import math
from torch . distributions import Normal
def log _ ndtr ( input _ tensor : torch . Tensor )
-> torch . Tensor :
normal = Normal (0, 1)
return normal . cdf ( input _ tensor ). log
()
Listing 11: Claude 3.7 Output
Gemini 2.5 Pro Failure Output
import torch
from torch . distributions import Normal
def log _ ndtr ( input _ tensor : torch . Tensor )
-> torch . Tensor :
loc = torch . tensor (0.0 , device = input _
tensor . device , dtype = input _ tensor .
dtype )
scale = torch . tensor (1.0 , device = input
_ tensor . device , dtype = input _ tensor
. dtype )
standard _ normal = Normal ( loc =loc ,
scale = scale )
log _ cdf _ values = standard _ normal . log _
cdf ( input _ tensor )
return log _ cdf _ values
Listing 12: Gemini 2.5 Pro Output
26

Analysis of Model Outputs
Both Claude 3.7 and Gemini 2.5 Pro incor-
rectly attempt to compute the log-CDF of the
standard normal distribution using PyTorch’s
Normal distribution. Claude 3.7 computes
normal.cdf(input_tensor).log() , which is
numerically unstable for large negative inputs since
the CDF approaches zero and its logarithm can un-
derflow to −∞, leading to invalid results. Gemini
2.5 Pro mistakenly calls a non-existent log_cdf
method on the PyTorch Normal object, causing an
immediate AttributeError in PyTorch 1.9. This
reflects a confusion between PyTorch and other li-
braries (such as SciPy or TensorFlow Probability)
that do provide this method. In summary, neither
model provides a robust or correct implementation;
the only reliable approach in this environment is to
usescipy.stats.norm.logcdf as shown in the
reference solution, or to implement a numerically
stable approximation in PyTorch when SciPy is
unavailable.
G.1.2 Greedy Example 2 (SciPy)
Problem Statement: complete the following func-
tion that check if all the batch of matrices are in-
vertible, using numpy 1.25.1.
Environment:
• Python Version: 3.10
• Scipy Version: 1.11.1
Starting Code
import warnings
from scipy . linalg import det
import numpy as np
warnings . filterwarnings ( 'error ')
def check _ invertibility ( matrices : np.
ndarray ) -> np. bool _:
pass
Listing 13: Greedy Example 3 Starting Code
Solution
return np.all( det ( matrices ))
Listing 14: Greedy Example 3 Solution
Gemini 2.5 Correct Output
import warnings
from scipy . linalg import det
import numpy as np
warnings . filterwarnings ( 'error ')
def check _ invertibility ( matrices : np.
ndarray ) -> np. bool _:
try:# Check if matrices is a batch
of matrices
ifmatrices . ndim < 3:
# If it 's a single matrix ,
add a batch dimension
matrices = np. expand _ dims (
matrices , axis =0)
# Check if all matrices are
square
batch _size , rows , cols =
matrices . shape
ifrows != cols :
return np.bool _( False )
# Calculate determinants for all
matrices in the batch
determinants = np. array ([ det (
matrix ) for matrix in
matrices ])
# A matrix is invertible if its
determinant is not zero
# Check if all determinants are
non - zero
return np.bool _(np. all(
determinants != 0))
except Exception :
# If any error occurs during
computation , return False
return np.bool _( False )
Listing 15: Gemini 2.5 Pro Output
GPT 4.1 Failure Output
import warnings
from scipy . linalg import det
import numpy as np
warnings . filterwarnings ( 'error ')
def check _ invertibility ( matrices : np.
ndarray ) -> np. bool _:
for matrix inmatrices :
try:
d = det ( matrix )
ifnp. isclose (d, 0):
return np.bool _( False )
except Warning :
return np.bool _( False )
return np.bool _( True )
Listing 16: GPT 4.1 Output
Analysis of Model Outputs
Gemini 2.5 Pro produces a robust solution that
handles both single matrices and batches, verifies
that matrices are square, computes the determinant
for each matrix, and correctly checks if all deter-
minants are non-zero to determine invertibility, re-
turning False on any exception to ensure safe fail-
ure. In contrast, GPT 4.1’s output contains a subtle
but critical error in exception handling: the code
attempts to catch Warning as an exception, but
Python treats warnings and exceptions differently,
27

so this approach is ineffective—especially since
warnings.filterwarnings(’error’) escalates
warnings to exceptions that should be caught as
Exception . As a result, GPT 4.1’s function may
fail or produce unpredictable results in the presence
of numerical issues or shape mismatches, despite
using the correct logic for checking determinant
values.
G.1.3 Greedy Example 3 (SymPy)
Problem Statement: Write a custom jacobi sym-
bols function that compute the Jacobi symbol.
Environment:
• Python Version: 3.9
• Sympy Version: 1.13
Starting Code
import sympy
def custom _ jacobi _ symbols (a: int, n: int
) -> int:
pass
Listing 17: Greedy Example 4 Starting Code
Solution
return sympy . jacobi _ symbol (a, n)
Listing 18: Greedy Example 4 Solution
GPT 4.1 Correct Output
import sympy
def custom _ jacobi _ symbols (a: int, n: int
) -> int:
return sympy . ntheory . jacobi _ symbol (a
, n)
Listing 19: GPT 4.1 Output
O1 Failure Output
import sympy
def custom _ jacobi _ symbols (a: int, n: int
) -> int:
return sympy . ntheory . residues . jacobi
_ symbol (a, n)
Listing 20: O1 Output
Analysis of Model Outputs
GPT-4.1 provides the correct solution by
calling sympy.ntheory.jacobi_symbol ,
which is the proper and documented API
for computing the Jacobi symbol in SymPy
1.13. In contrast, O1 incorrectly uses
sympy.ntheory.residues.jacobi_symbol ,assuming that the function is nested in the
residues submodule; however, in this version
of SymPy, jacobi_symbol is not found there,
resulting in an AttributeError on all tests.
This error highlights a lack of up-to-date API
knowledge or reliance on outdated documentation.
G.1.4 Greedy Example 4 (Flask)
Problem Statement: Complete the app set-up so
that, when given a batch of matrix, the json en-
coding compute the determinants of each matrix,
before flattening and converting the result to a list,
we are using scipy 1.11.1. Do not run the app in
your code.
Environment:
• Python Version: 3.10
• Flask Version: 3.0.0
• Extra Dependencoes: scipy==1.11.1
Starting Code
import flask
import numpy as np
from scipy import linalg
app = flask . Flask ( 'test1 ')
@app . route ( '/ data ')
def data ( num _ list ):
return flask . jsonify ({ 'numbers ': num
_list })
def eval _ app (app , data _fn , num _ arr ):
with app . test _ request _ context ():
response = data _fn( num _ arr )
return response . get _ data (as_ text
= True )
class MyCustomJSONHandler ( flask . json .
provider . DefaultJSONProvider ):
def default (self , obj ):
if isinstance (obj , np. ndarray )
and len ( obj . shape ) ==3 and
obj . shape [ -1]== obj . shape [ -2]
:
pass
Listing 21: Greedy Example 5 Starting Code
Solution
res = linalg . det ( obj )
return res . tolist ()
return super (). default ( obj )
app . json _ provider _ class =
MyCustomJSONHandler
app . json = app . json _ provider _ class (
app )
Listing 22: Greedy Example 5 Solution
Gemini 2.5 Pro Correct Output
28

import flask
import numpy as np
from scipy import linalg
app = flask . Flask ( 'test1 ')
@app . route ( '/ data ')
def data ( num _ list ):
return flask . jsonify ({ 'numbers ': num
_list })
def eval _ app (app , data _fn , num _ arr ):
with app . test _ request _ context ():
response = data _fn( num _ arr )
return response . get _ data (as_ text
= True )
class MyCustomJSONHandler ( flask . json .
provider . DefaultJSONProvider ):
def default (self , obj ):
if isinstance (obj , np. ndarray )
and len ( obj . shape ) ==3 and
obj . shape [ -1]== obj . shape [ -2]
:
# Calculate determinants for
each matrix in the
batch
determinants = linalg . det (
obj )
# Convert the resulting 1D
array of determinants to
a list
return determinants . tolist ()
# Fallback to the default
serialization for other
types
return super (). default ( obj )
# Apply the custom JSON provider to the
app
app . json = MyCustomJSONHandler ( app )
Listing 23: Gemini 2.5 Pro Output
Claude 3.7 Failure Output
import flask
import json
app = flask . Flask ( 'test ')
@app . route ( '/ data ')
def data ( num _ set):
return flask . jsonify ({ 'numbers ': num
_set})
def eval (app , data _fn , num _ set):
with app . test _ request _ context ():
response = data _fn( num _ set)
return response . get _ data (as_ text
= False )
def app _ set_up( app : flask . Flask ) -> None
:
def set _ encoder ( obj ):
if isinstance (obj , set):
return sorted (list ( obj ))
raise TypeError (f" Object of type
{ type ( obj )} is not JSON
serializable ")
app . json . encoder = lambda : json .
JSONEncoder ( default = set_ encoder )Listing 24: Claude 3.7 Output
Analysis of Model Outputs
Gemini 2.5 Pro provides a so-
lution that correctly subclasses
flask.json.provider.DefaultJSONProvider
and overrides the default method to check
for batches of square matrices, compute their
determinants using scipy.linalg.det , and
serialize the result as a list, fully aligning with
both the requirements and Flask 3.0.0’s JSON
provider API. In contrast, Claude 3.7’s output
incorrectly tries to customize JSON serialization
by modifying app.json.encoder —a property
that is not available in Flask 3.0.0—and focuses
on encoding sets rather than NumPy arrays,
neglecting the determinant computation altogether.
Consequently, Claude’s approach fails with an
AttributeError and does not satisfy the task
requirements even if the encoder logic were
correct, while Gemini’s implementation is robust
and functional for the specified environment.
G.2 Zero-Shot Chain-Of-Thought
G.2.1 CoT Example 1 (Torch)
Problem Statement: Calculate the modified
Bessel function of the first kind, order 1 using Py-
Torch’s special functions if available in this version,
otherwise you may use another library.
Environment:
• Python Version: 3.7
• Torch Version: 1.9.0
•Extra Dependencies: scipy==1.7.3,
numpy==1.21.6
Starting Code
import torch
def bessel _i1( input _ tensor : torch . Tensor
) -> torch . Tensor :
pass
Listing 25: CoT Example 1 Starting Code
Solution
import numpy as np
from scipy . special import i1 as
scipy _i1
output = torch . from _ numpy ( scipy _i1(
input _ tensor . numpy ()))
return output
Listing 26: CoT Example 1 Solution
29

Claude 3.7 Correct Output
import torch
from scipy import special
def bessel _i1( input _ tensor : torch . Tensor
) -> torch . Tensor :
# Convert PyTorch tensor to NumPy
array
input _ numpy = input _ tensor . cpu ().
numpy ()
# Apply the modified Bessel function
of the first kind , order 1
result _ numpy = special .i1( input _
numpy )
# Convert back to PyTorch tensor
result _ tensor = torch . from _ numpy (
result _ numpy ).to( input _ tensor .
device )
# Ensure the output has the same
dtype as the input
result _ tensor = result _ tensor .to(
dtype = input _ tensor . dtype )
return result _ tensor
Listing 27: Claude 3.7 Output
Gemini 2.5 Pro Failure Output
import torch
def bessel _i1( input _ tensor : torch . Tensor
) -> torch . Tensor :
""" Calculates the modified Bessel
function of the first kind ,
order 1.
Args :
input_tensor : The input tensor .
Returns :
A tensor containing the element -
wise modified Bessel
function of the first kind ,
order 1.
"""
# Check if torch . special .i1 is
available (it is in torch 1.9.0)
if hasattr ( torch . special , 'i1'):
return torch . special .i1( input _
tensor )
else :
# Fallback or error handling if
needed for other versions
# For torch 1.9.0 , this else
block should not be reached .
raise NotImplementedError (" torch
. special .i1 is not available
in this PyTorch version .")
Listing 28: Gemini 2.5 Pro Output
Analysis of Model Outputs
Claude 3.7 provides a robust solution by cor-
rectly identifying that torch.special.i1 is not
available in PyTorch 1.9.0 and instead leverag-
ingscipy.special.i1 to compute the modifiedBessel function, converting the result back to a
PyTorch tensor and ensuring proper device and
dtype handling; this approach is fully compatible
with the given environment and passes all tests.
In contrast, Gemini 2.5 Pro mistakenly assumes
that torch.special.i1 exists in PyTorch 1.9.0,
checks for its presence using hasattr , and raises
aNotImplementedError when unavailable; how-
ever, since this function is absent in the specified
version, the code fails every test due to the raised
exception.
G.2.2 CoT Example 2 (Scikit-learn)
Problem Statement: Adapt the use of ‘manhattan
distances‘ to obtain a pairwise distance matrix.
Environment:
• Python Version: 3.10
• Scikit-learn Version: 1.1
• Extra Dependencies: numpy==1.23.5
Starting Code
from sklearn . metrics . pairwise import
manhattan _ distances
import numpy as np
def get _ pairwise _ dist (X: np. ndarray ,Y:
np. ndarray ) -> np. ndarray :
distances = manhattan _ distances (X, Y
,sum_ over _ features = False )
pass
Listing 29: CoT Example 2 Starting Code
Solution
return np.sum( distances , axis =1)
Listing 30: CoT Example 2 Solution
GPT 4.1 Correct Output
from sklearn . metrics . pairwise import
manhattan _ distances
import numpy as np
def get _ pairwise _ dist (X: np. ndarray , Y:
np. ndarray ) -> np. ndarray :
distances = manhattan _ distances (X, Y
)
return distances
Listing 31: GPT 4,1 Output
Claude 3.7 Failure Output
from sklearn . metrics . pairwise import
manhattan _ distances
import numpy as np
def get _ pairwise _ dist (X: np. ndarray ,Y:
np. ndarray ) -> np. ndarray :
distances = manhattan _ distances (X, Y
,sum_ over _ features = False )
return np.sum( distances , axis =2)
Listing 32: Claude 3.7 Pro Output
30

Analysis of Model Outputs
In this example, GPT 4.1 provides
a correct implementation by calling
manhattan_distances(X, Y) without the
sum_over_features argument, which by default
returns the desired pairwise distance matrix of
shape (nsamplesX, nsamplesY). In contrast, Claude
3.7 incorrectly sets sum_over_features=False ,
causing the output to be a 3D array of shape,
and then applies np.sum(distances, axis=2) ,
which does not match the problem’s intent and
produces incorrect results if further summed in-
correctly. The misuse of the sum_over_features
parameter and the erroneous axis for reduction
demonstrates a misunderstanding of the scikit-
learn API and leads to a failure, while GPT 4.1’s
output is concise and correct for all valid input
shapes.
G.2.3 CoT Example 3 (Falcon)
Problem Statement: Create a class named Cus-
tomRouter to manage your application’s routes.
The class should maintain an internal dictionary
named routes for storing the mapping between
URI templates and their associated resources. Im-
plement an add_route method that accepts three
arguments: a URI template, a resource, and
additional keyword arguments. This method
should use Falcon’s routing utility—specifically,
the map_http_methods function from the fal-
con.routing module—to generate a mapping of
HTTP methods to resource handlers. If a ’fallback’
parameter is provided in the keyword arguments,
it should be passed to map_http_methods. The
method should then store a tuple consisting of the
resource and the generated method mapping in the
routes dictionary, using the URI template as the key,
and finally return the generated method mapping.
You must not modify the provided starting code,
which includes the empty CustomRouter class. In-
stead, complete the solution by dynamically adding
the required method in the solution() function.
Environment:
• Python Version: 3.10
• Falcon Version: 2.0
Starting Code
class CustomRouter :
def __ init __( self ):
self . routes = {}def solution () -> None :
def add _ route (
Listing 33: CoT Example 3 Starting Code
Solution
self , uri _ template , resource , ** kwargs ):
from falcon . routing import map _
http _ methods
method _ map =map_ http _ methods (
resource , kwargs . get ( '
fallback ', None ))
self . routes [ uri _ template ] = (
resource , method _ map)
return method _ map
CustomRouter . add _ route = add _ route
Listing 34: CoT Example 3 Solution
O1 Correct Output
import falcon
from falcon . routing import map _ http _
methods
class CustomRouter :
def __ init __( self ):
self . routes = {}
def solution () -> None :
def add _ route (self , uri _ template ,
resource , ** kwargs ):
fallback = kwargs . pop ( 'fallback '
, None )
iffallback is not None :
method _ map =map_ http _
methods ( resource ,
fallback = fallback )
else :
method _ map =map_ http _
methods ( resource )
self . routes [ uri _ template ] = (
resource , method _ map)
return method _ map
CustomRouter . add _ route = add _ route
Listing 35: O1 Output
GPT 4.1 Failure Output
from falcon . routing import map _ http _
methods
def solution () -> None :
def add _ route (self , uri _ template ,
resource , ** kwargs ):
fallback = kwargs . pop ( 'fallback '
, None )
iffallback is not None :
method _ map =map_ http _
methods ( resource ,
fallback = fallback )
else :
method _ map =map_ http _
methods ( resource )
self . routes [ uri _ template ] = (
resource , method _ map)
31

return method _ map
CustomRouter . add _ route = add _ route
Listing 36: GPT 4.1 Output
Analysis of Model Outputs
In this case, the O1 output provides a fully
correct solution by dynamically attaching an
add_route method to the CustomRouter class,
properly handling the optional fallback argu-
ment, and updating the routes dictionary accord-
ing to the problem statement. The method is in-
serted via the solution() function without mod-
ifying the provided class definition, and it uses
Falcon’s map_http_methods utility to construct
the method mapping as required. In contrast, the
GPT 4.1 output omits the explicit definition of the
CustomRouter class in its solution, violating the
requirement to use the existing starting code. Al-
though the logic within the solution() function
is correct, the absence of a CustomRouter defi-
nition in the completed module would lead to a
NameError or otherwise prevent the expected dy-
namic method attachment. The critical distinction
is that O1 respects all constraints including not
modifying the class definition directly, while GPT
4.1 provides an incomplete module, failing to meet
the initialization requirements set by the problem.
H Logic vs. Knowledge Retention
The goal of our proposed benchmark,
GitChameleon , is to evaluate a model’s ability to
retain version-specific knowledge—specifically,
whether it can recall the functionalities associated
with particular library versions it has been trained
on. Notably, this capability is distinct from the
ability to generate logically correct code. While
we do not explicitly disentangle whether model
failures on our evaluation suite stem from incorrect
logic generation or incorrect API version usage,
our benchmark is intentionally designed so that
most problems primarily test knowledge retention
rather than complex logic reasoning. For each
problem in our dataset, we compute the number
of logic-related nodes in the Abstract Syntax Tree
(AST) of the ground-truth solution and present
their distribution in Figure 14. As shown, most
ground-truth solutions contain fewer than five
logic-related AST nodes. This supports our claim
that the benchmark is primarily designed to assess
version-specific knowledge retention rather than
complex logic-based code generation.Table 15: Criteria for classifying AST nodes as logic-
related.
Condition Classification
Calling a user-defined function ✓
Calling built-in Python operators (e.g., +) ✓
Calling a math or utility function with non-
obvious purpose✓
Calling a library method (e.g.,
torch.from_numpy )✗
Composing multiple calls together ✓
The criteria for classifying AST nodes as logic-
related are provided in Table 15, and we include
visualizations of the ASTs for two example ground-
truth solutions for further illustration in Figures 15
and 16 respectively.
1. Sample ID: 0, Logic Nodes: 3
import torch
def log _ ndtr ( input _ tensor : torch .
Tensor ) -> torch . Tensor :
import numpy as np
from scipy . stats import norm
output = torch . from _ numpy ( norm .
logcdf ( input _ tensor . numpy ())
)
return output
Listing 37: Sample 0 Ground Truth Solution
2. Sample ID: 329, Logic Nodes: 0
import matplotlib . pyplot as plt
def use _ seaborn () -> None :
plt . style . use (" seaborn ")
Listing 38: Sample 329 Ground Truth Solution
0 20 40 60 80 100
Number of Logic Nodes050100150200250FrequencyLogic Nodes Distribution
Figure 14: Logic Nodes Distribution over samples’
ground truth solutions’ ASTs. Most ground truth
solutions have less than fivelogic nodes.
32

Figure 15: AST visualization for the ground-truth solution of Sample ID 0. The three color-coded call nodes (in
grey and green) represent the logic-related components, classified under the “composing multiple calls together”
category. The corresponding ground-truth code is shown in Code block 37 for reference.
Figure 16: AST visualization for the ground-truth solution of Sample ID 329. No logic nodes are present, as the
only call node corresponds to the “calling a library method” category. The ground-truth solution is provided for
reference in Code block 38.
33

I Prompt Templates
This appendix contains all the prompts we had used
for our experiments:
•The prompts for greedy sampling are given in
Figure 17.
•The prompts for self-debugging are given in
Figure 18.
•The prompt for the multi-step agent is given
in Figure 19.
• The prompt for RAG is given in Figure 20.
•The prompt and file format for Coding Assis-
tants are given in Figure 21.
•The prompt for SEK is given in Figure 22 (for
keywords generation) and Figure 23 (for code
generation).
J Artifacts and Model Details
This appendix provides citations for various arti-
facts and models mentioned in the paper.
J.1 Libraries
This is the full list of libraries included in
GitChameleon .
•PyTorch (Paszke et al., 2019)
•Geopandas (Jordahl et al., 2020)
•NLTK (Loper and Bird, 2002)
•NetworkX (Hagberg et al., 2008)
•GeoPy15
•Gradio (Abid et al., 2019)
•Scikit-Learn (Buitinck et al., 2013)
•Matplotlib (Hunter, 2007)
•PyCaret16
•Pandas (The pandas development team, 2020;
McKinney, 2010)
•NumPy (Harris et al., 2020)
•LightGBM17
15https://pypi.org/project/geopy/
16https://pycaret.org/
17https://lightgbm.readthedocs.io/•spaCy18
•Django19
•SciPy (Virtanen et al., 2020)
•Flask20
•Jinja221
•SymPy22
•Seaborn23
•mitmproxy24 25
•pytest26
•Falcon web framework27
•Tornado web server28
•Plotly29
•Librosa30
•Pillow31
•tqdm32
•Kymatio33
J.2 Models
Open-Weights Models
The following open-weights models were evalu-
ated:
•Llama 3.1 Instruct Turbo : (Kassianik
et al., 2025)
•Llama 3.3 Instruct Turbo 70B : (AI, 2025)
•Llama 4 Maverick 400B : (AI, 2025)
18https://spacy.io/
19https://www.djangoproject.com/
20https://flask.palletsprojects.com/
21https://jinja.palletsprojects.com/
22https://www.sympy.org/en/index.html
23https://seaborn.pydata.org/
24https://mitmproxy.org/
25https://mitmproxy.org/
26https://pytest.org/
27https://falconframework.org/
28https://www.tornadoweb.org/
29https://plotly.com/python/
30https://librosa.org/doc/latest/index.html
31https://python-pillow.org/
32https://github.com/tqdm/tqdm
33https://librosa.org/doc/latest/index.html
34

Figure 17: Prompts for Greedy Sampling
(a) System Prompt for Zero-Shot Prompting
You are a skilled Python programmer
tasked with solving a coding
problem . Your goal is to provide a
clear , efficient , and correct
solution that meets all the
specified requirements .
Please provide your solution
following these guidelines :
1. Use the required library in your
solution .
2. Incorporate the provided starter
code correctly .
3. Write your solution in Python .
4. Format your solution within a
markdown code block .
5. Ensure your code is clean ,
efficient , and well - commented .
6. Output only the code block and
nothing else .
Example output format :
```python
# [ Your code here , incorporating the
starter code ]
# [ Additional code and comments as
needed ]
```
After writing your solution , please
review it to ensure all
requirements are met and the code
is correct and efficient .
Here are the key elements for this
task :(b) System Prompt for Chain-Of-Thought Prompting
You are a skilled Python programmer
tasked with solving a coding
problem . Your goal is to provide a
clear , efficient , and correct
solution that meets all the
specified requirements .
First , let 's think step -by - step . Then
, please provide your solution
following these guidelines :
1. Use the required library in your
solution .
2. Incorporate the provided starter
code correctly .
3. Write your solution in Python .
4. Format your solution within a
markdown code block .
5. Ensure your code is clean ,
efficient , and well - commented .
6. Output nothing else after the code
block .
Example output format :
[Step -by - step thinking ]
```python
# [ Your code here , incorporating the
starter code ]
# [ Additional code and comments as
needed ]
```
After writing your solution , please
review it to ensure all
requirements are met and the code
is correct and efficient .
Here are the key elements for this
task :
(c) User Prompt
1. Required Library :
<library >
{{ library }}
</ library >
2. Python version :
<python >
{{ python_version }}
</ python >
2. Coding Problem :
<coding_problem >
{{ coding_problem }}
</ coding_problem >
3. Starter Code :
<starter_code >
{{ starter_code }}
</ starter_code >
35

•Qwen 2.5-VL Instruct 72B : (Qwen et al.,
2025)
•Qwen 3 235B :(Yang et al., 2025)
•Command A 111B : (Cohere et al., 2025)
•DeepSeek R1 685B : (DeepSeek-AI, 2025)
•DeepSeek v3 : (DeepSeek-AI et al., 2025)
•Openhands LM 32B v0.1 : (Wang, 2025)
•Reka Flash-3 : (Reka)
•Jamba 1.6 Mini, Large : (Lieber et al.,
2024)
Enterprise Models
The following enterprise models were evaluated:
•Arcee CoderL : (Arcee)
•Claude 3.5 Haiku34
•Claude 3.5 Sonnet35
•Claude 3.7 Sonnet : (Anthropic, 2025)
•Claude 4 Sonnet36
•CommandR+37
•Gemini 1.5 Pro : (Team et al., 2024)
•Gemini 2.0 Flash : (Kampf, 2025)
•Gemini 2.5 Pro : (Cloud, 2025)
•Gemini 2.5 Flash : (Cloud, 2025)
•GPT-4.1 : (OpenAI, 2025a)
•GPT-4.1-mini : (OpenAI, 2025a)
•GPT-4.1-nano : (OpenAI, 2025a)
•GPT-4o : (OpenAI, 2024a)
•GPT-4o-mini : (OpenAI, 2024a)
•GPT-4.5 : (OpenAI, 2025b)
•o1: (OpenAI, 2024b)
•o3-mini : (OpenAI, 2024b)
34https://www.anthropic.com/claude/haiku
35https://www.anthropic.com/news/claude-3-5-sonnet
36https://www.anthropic.com/claude/sonnet
37https://cohere.com/blog/command-r-plus-micro
soft-azure•codex-mini38
•Grok 3 : (xAI, 2025)
•Mistral Medium 3 : (Mistral AI, 2025)
•Devstral Small39
•Inflection 3 Productivity40
•Liquid LFM 40B MoE41
•Nova Pro :(Intelligence, 2024)
J.3 Coding Assistants (CLI/IDE)
The following coding assistants were studied as
part of the experimentation pipeline:
•Claude Code42(CLI)
•Goose43(CLI)
•Cline44(IDE-VSCode)
•RooCode45(IDE-VSCode)
•KiloCode46(IDE-VSCode)
38https://platform.openai.com/docs/models/code
x-mini-latest
39https://mistral.ai/news/devstral
40https://openrouter.ai/inflection/inflectio
n-3-productivity
41https://www.liquid.ai/blog/liquid-foundatio
n-models-our-first-series-of-generative-ai-mod
els
42https://docs.anthropic.com/en/docs/claude-c
ode/overview
43https://block.github.io/goose/
44https://cline.bot/
45https://roocode.com/
46https://kilocode.ai/
36

Figure 18: Prompts for Self-Debugging
(a) System Prompt
You are an expert programming assistant .
Your task is to fix issues in a
generated Python solution for a
given programming problem . You are
provided with :
- A problem statement
- Starter code
- A previously generated incorrect
solution
- A top - level execution trace or error
message
- Dependencies information ( versions ,
libraries ).
Please generate a corrected Python
solution by following these strict
guidelines :
1. Use the required libraries explicitly
in your code .
2. Correctly incorporate the provided
starter code - do not remove or
alter its structure .
3. Write in standard Python syntax .
4. Wrap your entire solution within a
single Markdown code block .
5. Do not include any text outside the
code block - no explanations ,
comments , docstrings , or usage
examples .
6. Ensure the code is clean , efficient ,
and syntactically valid .
7. Avoid interactive , stateful , or
environment - dependent constructs (e.
g., Django projects , web servers ).
8. Your output must be executable in a
non - interactive environment (e.g., a
test harness or script runner ).
Example output format :
```python
# [ Your corrected code here ]
```
Before submitting , carefully review your
code for correctness , completeness ,
and adherence to all constraints .(b) User Prompt
<Problem >
{ problem }
</ Problem >
<Python Version >
{ python_version }
</ Python Version >
<Library >
{ library }
</ Library >
<Version >
{ version }
</ Version >
<Extra Dependencies >
{ additional_dependencies }
</ Extra Dependencies >
<Starting Code >
{ starting_code }
</ Starting Code >
<Generated Solution >
{ solution }
</ Generated Solution >
<Trace >
{ top_level_trace }
</Trace >
37

Figure 19: Tool-Calling Agent Prompt
You are to solve a coding problem in
Python .
# Instructions :
* The coding problem requires using the
library { library }=={ version }. Try
using the problem with only this
library and the standard Python
libraries .
* Do a thorough research on the web
about how to solve the coding
problem for the given library
version . Repeat multiple times if
needed .
* BEFORE FINISHING YOUR WORK , YOU MUST
check your solution to the coding
problem by running the `
docker_problem_sandbox `tool .
* Use the `final_answer `tool to return
a self - contained Python script that
solves the problem . DO NOT INCLUDE
ANY TEXT BESIDES FOR THE CODE IN THE
FINAL ANSWER .
* The solution needs to be in a markdown
code block .
* The solution needs to start with the
starter code provided below .
# Coding Problem :
{ problem }
# Starter Code :
```python
{ starting_code }
```Figure 20: RAG Prompt
You are an AI assistant specialized in
solving Python programming problems
using information derived from
documentation .
Each query may specify particular
libraries and version constraints .
Your task is to generate a correct ,
efficient , and minimal Python
solution that adheres strictly to
these requirements .
Please follow these rules when crafting
your response :
1. Use only the specified libraries and
respect the given version
constraints .
2. Incorporate any provided starter code
as required .
3. Write only Python code - no in - line
comments or usage examples . Do not
provide anything in the response but
the code .
4. Ensure the code is clean , minimal ,
and adheres to best practices .
5. The code must be executable in a non -
interactive environment (e.g., avoid
frameworks like Django or code
requiring a web server ). Context :
{ context }
Based on the above , respond to the user
query below .
Query : { query }
Here, {context} refers to the context of the top- kre-
trieved documents from the vectorized database for that
query and {query} is the same as the User Prompt given
in Figure 17(c).
38

Figure 21: Prompt and File Format for Coding Assistants
(a) Prompt
Solve each sample_ {i}. py in this folder
then subsequently save your
solutions as py files with the same
name in a separate subfolder
called "{ assistant name }" that just
completes the starting code
provided in the sample and uses the
instructions written in the
comments at the start of each file .(b) Input File Format
# Complete using the following libraries
and /or extra dependencies and their
versions :
# problem statement : { problem }
# library : { library }
# version : { version }
# extra_dependencies : {
extra_dependencies }
{ starting_code }
(a) presents the prompt template we had used for our Coding Assistant experiments. (b) shows the format of the
example files referenced in the prompt.
Figure 22: Prompts for SEK (Keyword Generation Stage)
(a) System Prompt
You are a seasoned Python developer at a
Fortune 500 company who excels at
analyzing complex code . Analyze the
given code problem from the problem
statement and starter code provided .
Try to extract the keywords from
the code problem . For each
identified keyword :
1. Provide the keyword .
2. Give a formalized explanation of the
keyword using technical languages .
Provided Format :
Keywords :[ Keywords ]
Explainations :[ Formalized explanations ]
Guidelines :
- Prioritize keywords that are crucial
to understanding the input
parameters , return content or
supplementary information .
- Use precise languages in explanations
and provide formalized definitions
where appropriate .
- Ensure explanations are consistent
with the behaviors expected based on
the problem description .
- Limit to the top 1-3 important
keywords to focus on core concepts .
- You are supposed to output a
structured JSON output containing
the extracted keywords and their
corresponding formalized
explanations in individual lists of
strings . The keys for this JSON must
be Keywords and Explainations .
- Strictly adhere to the provided format
, do not output anything else .(b) User Prompt
<Problem Statement >
{ problem }
</ Problem Statement >
<Starting Code >
{ starting_code }
</ Starting Code >
39

Figure 23: Prompts for SEK (Code Generation Stage)
(a) System Prompt
You are a skilled Python programmer
tasked with solving a coding problem
. Your goal is to provide a clear ,
efficient , and correct solution that
meets all the specified
requirements .
Please provide your solution following
these guidelines :
1. Use the required library in your
solution .
2. Incorporate the provided starter code
correctly .
3. Write your solution in Python .
4. Format your solution within a
markdown code block .
5. Ensure your code is clean and
efficient .
6. Output only the code block and
nothing else . Do not add any in - line
comments , documentations ,
references or usage examples .
7. Make sure your code is executable in
a non - interactive environment . For
example , do not write code which
requires building a Django project
or deploying a web - app .
Example output format :
```python
# [ Your code here , incorporating the
starter code ]
```
After writing your solution , please
review it to ensure all requirements
are met and the code is correct and
efficient .
Here are the key elements for this task :(b) User Prompt
<Python Version >
{ python_version }
</ Python Version >
<Library >
{ library }
</ Library >
<Version >
{ version }
</ Version >
<Extra Dependencies >
{ extra_dependencies }
</ Extra Dependencies >
<Problem Statement >
{ problem }
</ Problem Statement >
<Keywords >
Analyze the following key terms and
their relationships within the
problem context :
{ General_Keywords }
{ Abstract_Keywords }
</ Keywords >
<Starting Code >
{ starting_code }
</ Starting Code >
40
# Meta-RAG on Large Codebases Using Code Summarization

**Authors**: Vali Tawosi, Salwa Alamir, Xiaomo Liu, Manuela Veloso

**Published**: 2025-08-04 17:01:10

**PDF URL**: [http://arxiv.org/pdf/2508.02611v1](http://arxiv.org/pdf/2508.02611v1)

## Abstract
Large Language Model (LLM) systems have been at the forefront of applied
Artificial Intelligence (AI) research in a multitude of domains. One such
domain is software development, where researchers have pushed the automation of
a number of code tasks through LLM agents. Software development is a complex
ecosystem, that stretches far beyond code implementation and well into the
realm of code maintenance. In this paper, we propose a multi-agent system to
localize bugs in large pre-existing codebases using information retrieval and
LLMs. Our system introduces a novel Retrieval Augmented Generation (RAG)
approach, Meta-RAG, where we utilize summaries to condense codebases by an
average of 79.8\%, into a compact, structured, natural language representation.
We then use an LLM agent to determine which parts of the codebase are critical
for bug resolution, i.e. bug localization. We demonstrate the usefulness of
Meta-RAG through evaluation with the SWE-bench Lite dataset. Meta-RAG scores
84.67 % and 53.0 % for file-level and function-level correct localization
rates, respectively, achieving state-of-the-art performance.

## Full Text


<!-- PDF content starts -->

Meta-RAG on Large Codebases Using Code
Summarization
Vali Tawosia,*, Salwa Alamira, Xiaomo Liuband Manuela Velosob
aJP Morgan AI Research, UK
bJP Morgan AI Research, US
ORCID (Vali Tawosi): https://orcid.org/0000-0001-5052-672X, ORCID (Salwa Alamir):
https://orcid.org/0009-0006-6650-7041, ORCID (Xiaomo Liu): https://orcid.org/0000-0003-4184-4202, ORCID
(Manuela Veloso): https://orcid.org/0000-0001-6738-238X
Abstract. Large Language Model (LLM) systems have been at the
forefront of applied Artificial Intelligence (AI) research in a multi-
tude of domains. One such domain is software development, where
researchers have pushed the automation of a number of code tasks
through LLM agents. Software development is a complex ecosystem,
that stretches far beyond code implementation and well into the realm
of code maintenance. In this paper, we propose a multi-agent system
to localise bugs in large pre-existing codebases using information
retrieval and LLMs. Our system introduces a novel Retrieval Aug-
mented Generation (RAG) approach, Meta-RAG, where we utilise
summaries to condense codebases by an average of 79.8%, into a
compact, structured, natural language representation. We then use an
LLM agent to determine which parts of the codebase are critical for
bug resolution, i.e. bug localisation. We demonstrate the usefulness
of Meta-RAG through evaluation with the SWE-bench Lite dataset.
Meta-RAG scores 84.67% and 53.0% for file-level and function-level
correct localisation rates, respectively, achieving state-of-the-art per-
formance.
1 Introduction
Research into the automation of software development has been at
the core of the intersection of AI and software engineering (SE). Prior
work has focused on traditional probabilistic models [17] as well as
neural-network models [49]. More recently, code language models
[15, 12, 55] as well as LLMs [28, 45] have been utilised in this do-
main, particularly for the tasks of code generation, code completion,
and bug resolution [31].
After the introduction of LLMs, due to their superior ability in
generating meaningful output with respect to previous AI models,
there has been an influx of research on LLM-based software engi-
neering tasks [14]. As such, LLM multi-agent systems have become
one of the standard techniques to implement effective SE automation
[41]. Multi-agent systems enable researchers and industry practition-
ers to maximise the utility of LLMs by designing specific roles and
prompts that achieve modular goals. Such systems have already been
introduced in the software development lifecycle and in AI commu-
nities in the form of code generation [59], computer control [39], and
web navigation [62], to name a few.
∗Corresponding Author. Email: vail.tawosi@jpmorgan.comNevertheless, the software development life cycle is strongly af-
fected by bugs. Therefore, the bug discovery, localisation, and reso-
lution account for a large proportion of software development costs
[22]. Even within this, bug localisation is especially costly and te-
dious, accounting for approximately 70% of developers’ time spent
resolving a bug [36, 35]. This has enticed software engineering re-
searchers and practitioners to develop a number of methods for au-
tomating the localisation and repair of software defects, with the lat-
est works utilising LLM agents.
In this paper, we propose Meta-RAG; a Retrieval Augmented Gen-
eration (RAG) approach to aid in bug localisation in large codebases.
Rather than retrieve the code itself as previous methods have done,
we retrieve code meta-data. In order to generate this meta data, we
also introduce a novel agent that constructs a compact, natural lan-
guage representation of a codebase; a codebase summary. Our sum-
maries dramatically reduce the size of codebases by approximately
80% on average.
It is vital to measure the performance of our solution using realistic
software development scenarios where developers are required to up-
date an existing codebase by adding new features or fixing bugs. Un-
fortunately, many of the existing code generation efforts with LLMs
utilise benchmarking datasets which are comprised of simple cod-
ing examples and are typically contained to one function [19]. Nev-
ertheless, developers are customarily required to update an existing
codebase by adding new features or fixing bugs. Therefore, assessing
code implementation systems requires a dataset that requires several
sub-tasks (e.g., bug localisation, code generation, code completion,
etc.) to be successfully executed. Consequently, we test our solution
against the SWE-bench Lite dataset; a collection of real-world issues
on Github from popular open-source repositories. As these are large,
complex code repositories, condensing the codebases into the form
of summaries also allows us to overcome two known limitations of
LLMs and general AI systems for code implementation: (1) context
window length constraints and (2) the diminishing effect of the at-
tention mechanism for long prompts.
For instance, current GPT models support a 128K context win-
dow length (GPT-4o and o1), including input and output tokens
[38]. Other LLMs such as Cloud 3-Opus have a 200K token win-
dow, and Gemini 1.5 pro has a 128K standard token window. [29].
Nonetheless, a sample code repository in our benchmark, such asarXiv:2508.02611v1  [cs.SE]  4 Aug 2025

SymPY , contains 6,360,381 tokens1; clearly exceeding the input con-
text length of the current common LLMs. For the second challenge
of diminishing attention, the LLM requires some form of interven-
tion, such as chain-of-thought or planning, in order to know where
and what changes need to be implemented to resolve the issue with-
out risk of hallucination [10, 1, 20], particularly when dependencies
are involved.
Our solution overcomes these challenges through two key contri-
butions; summaries and Meta-RAG. We find that our system achieves
the highest successful bug localisation rate at both file and function
level amongst state-of-the-art methods presented on the SWE-bench
Lite leader board.
2 Related Work
Prior to the introduction of LLMs, many researchers have investi-
gated the use of AI for many software engineering tasks, including
requirement engineering, software design, code completion, library
upgrades, automated testing, bug localisation, automated program re-
pair, and code review [56, 7]. After the introduction of LLMs, due to
their superior ability in generating meaningful output with respect to
previous AI models, there has been an influx of research on LLM-
based software engineering tasks [14]. These applications cover a
wide range of tasks such as code summarisation and understanding
[55, 37, 4, 48], code translation [40, 13], code review [6, 18], pro-
gram synthesis [42, 12], program repair [57, 23, 11], test generation
[46, 30], and planning [53].
It is clear that most of these applications, however, are focused
on the software implementation phase of the software development
lifecycle (SDLC), i.e. coding. Nevertheless, research has shown that
only 15% to 35% of human effort is spent on the implementation
phase of the SDLC [34]. In this paper we focus on specifically on
bug localisation, which remains a challenging task, particularly in
large-scale software systems. Bug localisation is the process of de-
termining the specific location of source code that needs to be mod-
ified in order to resolve a specific issue or bug [52]. Thereby, many
previous works have used traditional information retrieval methods
in order to aid in tackling this challenge.
BugLocator performs file-level localisation by ranking files based
on similarity of the text between the bug report and the source code,
taking into consideration information related to historical bugs that
have been resolved and using a revised Vector Space Model (rVSM)
[61]. BLUiR can identify bugs at the function-level by utilising
source code and bug reports. By structuring and indexing the data,
they are able to apply information retrieval methods (combining TF-
IDF with BM25) to code constructs, to achieve a higher accuracy for
bug localisation [47].
Goyal et. al show that the trend of techniques applied for bug triag-
ing has shifted from machine learning based approaches towards in-
formation retrieval based approaches [16]. Later works began to mix
both approaches by incorporating deep learning. One such example
is HyLoc, which uses text similarity between bug and source files
from rVSM in combination with a Deep Neural Network (DNN) to
learn the relationship between the terms in the reports and different
code tokens from the source files [25]. DNNLOC [26] works in a
similar way with the introduction of the project’s bug-fixing history
in order to achieve a higher accuracy.
The emergence of LLMs has increased present research in this
space. With this, many works have began implementing Retrieval
1The metric reported here is from a random code commit in this projects
history.Augmented Generation approaches (RAG). For instance, RAGFix
searches Stack Overflow posts via a Question and Answer Knowl-
edge Graph (KGQA) for bug localisation and program repair [33].
Nevertheless, one particular benchmarking dataset has seen a rise in
popularity to aid in the assessment of the LLM outputs for bug res-
olution; SWE-bench. SWE-Bench contains 2,294 instances of real-
world code issues collected from 12 prominent python repositories
on GitHub; the SWE-bench benchmarking dataset [21]. Each in-
stance of this dataset contains the issue description in natural lan-
guage, the gold patch which is produced by programmers to resolve
the issue, and two sets of unit tests: pass _to_pass (unit tests that
should pass before and after the change to confirm that the new edit
has not broken any previous functionality), and fail_to_pass (unit
tests that fail due to the reported issue and should pass if the intro-
duced edit resolves the issue successfully). SWE-bench also provide
subsets of the original dataset referred to as "Lite" (300 tasks, used
in this paper) and "Verified" (500 tasks).
A number of other benchmarking datasets have also been widely
used to evaluate software engineering tasks. The HumanEval Dataset
(introduced in 2021 by OpenAI), is a curated collection, com-
prised of 164 handcrafted programming challenges that include lan-
guage comprehension, basic mathematics, and algorithmic problem-
solving [12]. MBPP is comprised of 1,000 Python programming
challenges that are collected from a diverse set of contributors and
are tailored for software engineers at an introductory programming
level [9]. However, both HumanEval and MBPP are comprised of
single, isolated functions, and as such, do not provide a realistic sim-
ulation of the setting that is required to assess our system for bug
resolution.
The SWE-Bench paper not only provides the dataset, but an intro-
duces an approach to bug resolution. This method retrieves code to
provide additional context to a prompt through RAG. To achieve this,
they utilised a sparse retrieval method, which relies on BM25 and an
“Oracle” retrieval that returns the files edited by the gold-standard
patch referenced in Github [21]. As an engineer would not have pre-
vious knowledge where a new feature should be included, or where
a bug is localised, the “Oracle” approach is less realistic.
Another study utilising this dataset provides the LLM with a tex-
tual representation of hierarchy of the files in the codebase [58]. With
this approach, the LLM is required to decide which files to retrieve
based solely on the file name. It narrows down to classes and func-
tions, and line numbers, in iterations, until the LLM localises the
lines needed to edit. The performance of this method depends on
proper naming of the files in a project, which can vary from project
to project. Also, there are cases in which a file hosts multiple classes
and functions, and not all functionality included in the file is repre-
sented in the file name.
Agent-based approaches have also been adopted; SWE-Agent
[59] proposed an Agent-Computer Interface (ACI), where pre-
programmed tools are provided to an agent. This allows the agent
to ideate, search, and retrieve pieces of a codebase in a loop until it
finds relevant information to use. Although this approach succeeds
in many cases, it may still fail to find relevant information after tens
of tries. Our solution builds on the above by utilising an agent-based
architecture to create a novel retrieval method that returns relevant
code based on summaries.

Summary to Code Structural Matching
+ AST Root
imports
function f1
Class C1
function C1 -f1
function C1 -f2
mainAbstract Syntax Tree (AST) of code file# File sample.py Summary: 
<File Summary>
# Function Summary: function_f1
# Type declaration: function_f1() -> None
# Summary: < function_f1 summary>
# Class Summary: class_C1
# Summary: <class_C1 summary>
 # Function Summary : function_C1_f1
    # Type declaration: def function_C1_f1(self) -> None
    # Summary:<function_C1_f1 summary >
 # Function Summary : function_C1_f2
    # Type declaration: def function_C1_f2(self) -> None
    # Summary:<function_C1_f2 summary>
# __MAIN__ Summary: 
The main function is called if this file is run as a script.File Name: sample.py
New Function
Summary  AgentFigure 1 : Example of a summary template (right) and a summary update using the AST.
3 Methodology
3.1 Summaries
Bug localisation in a large codebase requires an overall knowledge
of the whereabouts of files, classes, and functions, in addition to their
connections. To provide our LLM agents with such information, we
transform the codebase into a compact yet familiar form of represen-
tation for our LLM agents: natural language summaries.
In this section, we introduce the Summary Agent. The Summary
Agent performs two main functions: the generation of summaries
and the update of summaries after each task’s code is generated by
an agent we refer to as the Code Agent. The Code Agent can be any
LLM agent (or agent system) that takes a task and outputs a generated
code solution.
Generating summaries for an existing codebase is an offline one-
off exercise. We generate and store one summary file per code file. It
includes a short summary of the file content (i.e., functionality sup-
ported by the code within the file), a short summary of each class or
file level function in the file (in the same order and indentation they
appear in the file), and the same per any inner class or function in a
recursive manner. Each summary item also includes important infor-
mation about the item: A file summary includes the name and path to
the file; a class summary includes the class name and its attributes;
and a function summary includes the function name and signature.
There may still exist some code that is not a class or function; this
is then summarised into "__MAIN__". Note that imports are not in-
cluded in the summary as they are present in the code that will later
be augmented with the LLM prompt. An example of a summary tem-
plate is shown in the grey box of Figure 1.
The code file’s content is used in combination with proper instruc-
tions in a prompt to ask the LLM to generate a summary. The gen-
erated summaries are stored in a database along with their code files
and are structurally matched to their corresponding file structures byparsing into an abstract syntax tree (AST), which is a common data
structure used to represent the structure of a program. This is car-
ried out by parsing the generated summary and identifying the code
elements, with their order and hierarchy in the file. We traverse the
AST node-by-node, marking the matching summary section to each
of the main AST nodes (i.e. functions and classes). We match then
those with the AST of the code file, to make sure: a) all code ele-
ments (classes and functions) have corresponding summaries, and b)
they are in the correct order and hierarchy. If any misalignment is
identified, the issue is fixed by reordering the summary elements or
updating the summary if needed.
After this matching procedure, the summaries and code content
are stored in a data structure similar to that of an AST, with each
code element stored alongside its summary. This helps with eas-
ier code and summary traversal and retrieval, especially when we
want to return the summary for a given code element or a code ele-
ment for a given summary. Figure 1 better depicts this mapping. By
compressing the codebase into this new representation, we observe
a 79.8% ( std. 9.1%) reduction on average for the SWE-bench Lite
code repositories.
Summary agent is also responsible for keeping the code sum-
maries up to date with changes to the codebase. Once a new func-
tion is introduced or edited by the Code Agent, we generate a new
summary for the new element using the Summary Agent, and update
the summary file by injecting the summary into the corresponding
location, taking the structural matching into consideration. This is
depicted again in Figure 1 where function C1-f2 is the new function
created for a task, and the summary agent is able to inject the sum-
mary for this function into the correct location in the database.
3.2 Meta-RAG
Once the summaries are generated and stored, LLM agents will work
with the summaries instead of code to localise the change. This will

Control Agent Read -list: Files and functions required to be 
aware of to complete task. Not to be edited
Write -list: Existing files and functions that must 
be edited to complete task
New -list: Files and functions that need to be 
created to complete taskParse 
InstructionsFetch code
Retrieve 
Summaries
Meta -RAG approach to find 
relevant context:
→ Use LLM  as a code 
component retriever
→ Consider what needs to be 
edited  and what is needed for 
relevant context→ Use fine-grained  summaries 
for RAGMeta -RAG Prompt Generator
You are provided …
….
Read: {read -list_code }
….
Edit: {write -list_code } 
…
 {new -list_todo } # TODO
Code Summaries
Codebase
//Input Task
LLM
Summary  Agent
Code  AgentFigure 2 : Meta-RAG architecture diagram.
help overcome the second challenge for working with large code-
bases: the diminishing attention effect. On that account, we introduce
Meta-RAG; a novel retrieval method that uses the summaries of the
code units from Summary Agent at different hierarchical levels, and
prompts the LLM to identify the relevant pieces of code required
from the codebase to complete the task at hand. The advantage of
this method, in combination with the summaries, is that there are dif-
ferent levels of granularity that the LLM can decide to retrieve (file,
class, function), typically starting at a higher level (file) and honing
in on the location at a lower level (function). The LLMs are able
to more efficiently traverse through the codebase, as the summaries
convey more information in fewer tokens. When code augmentation
or bug resolution is intended, we first need to find the location within
the codebase that needs to be changed or extended to support a bug
fix or a new feature. In the case of the SWE-bench dataset, the tasks
are predominantly bugs; thus this component aids particularly in bug
localisation.
Working with a large codebase means there are many interdepen-
dencies within the code in the form of API calls. Thus the new code
should attempt to reuse available code via these API calls, rather than
re-implementation. This is possible with Meta-RAG. Via prompt-
ing, an agent referred to as Control Agent, asks the LLM to find the
code units within the summaries that are relevant to the task at hand,
and return them in three lists: Read-list ,Write-list , and New-list . The
agent instructs the LLM to include into the Read-list , the code units
that provide context or are useful for implementing the change. The
Write-list contains the code units that need to be amended to re-
flect the new change. And the New-list may include new functions,
classes, or files that the LLM needs to author for a new feature. In the
case of a new project where there are no summaries, all code would
be in the New-list . This approach encourages the LLM to reuse pre-
vious code in the codebase and helps with a natural integration of
the new code into the existing codebase [54]. Furthermore, previous
RAG approaches rely solely on identifying the optimal Write-list . We
propose that a Read-list is provided to supply additional context to
the LLM and aid in downstream resolution of the task instances.
The aforementioned retrieval instructions are then sent to a Code
Agent. The Code Agent should retrieve the specified pieces of codefrom the existing codebase and augment these into a prompt, in com-
bination with the task, in order to generate a code patch. The LLM
output is parsed to extract the code snippets with regular expression
and rule based parsing to determine the filename and function name
to assess localisation. An overview of the system architecture is ex-
hibited in Figure 2.
There still exist repositories large enough that their summaries do
not fit within the context window of the LLM. Consequently, Meta-
RAG starts with file-level summaries (which are one-liner file sum-
maries) and the Control Agent requires the LLM to short-list the files
that are relevant to the current task. Once the files are selected, the
Control Agent is provided with the full summary of the selected files
and asked to select the relevant code parts (classes and functions).
This process can be broken down into many more rounds of retrieval
if any of the context gets larger than the context-window, or a pre-set
limit. Again, this remains possible due to availability of summaries
in different hierarchical levels.
4 Experimental Setup
We evaluate our approach using the SWE-bench Lite dataset [51].
This dataset is a sub-sample of the full SWE-bench dataset contain-
ing 300 instances across the 12 repositories. SWE-bench Lite is sam-
pled by the authors of the original dataset to be more self-contained,
with a focus on evaluating functional bug fixes.
To reduce the cost of summarisation, we only summarise the ear-
liest instance for each repository (based on their creation date) and
update the summaries for instances present after that. We do this by
getting a git diff between the consecutive instances and updating the
summaries for the changed parts in the code. Creating a summary of
the entire codebase for each task instance, rather than updating sum-
maries would require over 605 million code tokens to be provided to
the LLM (See Table 1).
With regards to the LLM, for all of the experiments, we use GPT-
4o. This LLM model has a 128K context length. When compar-
ing our approach with other solutions, we control for the LLM by
comparing our model to six top approaches on the SWE-bench Lite

Table 1 : The Total Code Tokens and Summary Tokens required to be generated for instances in each repository within the SWE-bench Lite
dataset is presented with the Codebase Size Reduction rate achieved via summarisation. Columns on the right half of the table show the
reduction rate we achieved with summary update method: New Commit Tokens shows the total tokens of code change in the consecutive
instances of a repo, Updated Summary is the number of the summary tokens generated for the code changes, and Saved by Summary Update
shows the rate of codebase size reduction using summary update. Also Saved on Code Submission shows the reduction rate in changed code
tokens submitted for summarization compared to re-summarising the codebase for each task instance.
Total Code Total Summary Codebase Size New Commit Updated Summary Saved on Saved by
Repo # Instances Tokens Tokens Reduction % Tokens Tokens Code Submission % Summary Update %
astropy 6 9,259,099 1,632,048 82.4% 4,020,211 675,685 58.6% 83.2%
django 114 119,282,807 46,566,608 61.0% 25,200,201 8,624,366 81.5% 65.8%
flask 3 208,232 48,165 76.9% 113,880 24,428 49.3% 78.5%
matplotlib 23 30,621,968 4,059,709 86.7% 9,722,932 1,058,121 73.9% 89.1%
pylint 6 2,091,552 529,337 74.7% 1,025,400 232,386 56.1% 77.3%
pytest 17 3,332,021 1,086,032 67.4% 2,192,246 689,091 36.5% 68.6%
requests 6 2,238,204 193,130 91.4% 375,410 98,343 49.1% 73.8%
scikit-learn 23 26,031,761 3,580,341 86.3% 8,297,394 1,116,840 68.8% 86.5%
seaborn 4 1,022,276 137,697 86.5% 397,902 63,110 54.2% 84.1%
sphinx 16 10,037,071 2,398,412 76.1% 4,437,583 1,130,412 52.9% 74.5%
sympy 77 399,640,756 45,324,399 88.7% 81,509,674 10,373,951 77.1% 87.3%
xarray 5 1,859,380 389,706 79.0% 1,327,671 245,474 37.0% 81.5%
Total 300 605,625,127 105,945,584 79.8% (Mean) 138,620,504 243,32,207 57.9% (Mean) 79.2% (Mean)
leaderboard, filtered only to open source projects using GPT-4o2.
We assess and report the localisation rate, which shows the pro-
portion of instances where our tool was able to identify the location
of the fix for the problematic units of code successfully. We com-
pute this metric by comparing the agent-identified locations to fix,
with those edited by the “gold” patch at both the file level and the
function level (See Equation 2). We rely on the lines changed by the
developer who has fixed the bug as the “gold” standard. Parsing the
available git patch in the benchmark, we extract the edited lines and
use the code AST to find the corresponding edited function from the
source file. To compare our localisation metric to that of previous
works, we calculate the same metric on their outputs reported in the
SWE-bench git repository [50].
g(t) =(
1if change(Meta-RAG Write_Listt)= change(gold-patcht)
0otherwise
(1)
%CorrectLocalisation =100
|dataset |X
task∈datasetg(task )(2)
As a baseline, we also compare our approach to BM25-based re-
trieval methods. Our baselines include BM25 from Lucene index
search3and BM25-Plus [32]4. BM25 is a popular ranking function
used in information retrieval to estimate the relevance of documents
to a search query. It works by calculating a score for each document
based on the appearance of the query terms in it. It uses both term fre-
quency and inverse document frequency for this calculation. Equa-
tion 3 shows how the BM25 score is calculated:
sq=X
t∈qlogN−d ft+ 0.5
(d ft+ 0.5)(k1+ 1).tftd
k1.
(1−b) +b.(Ld
Lavg)
+tftd
(3)
where the retrieval score ( s) for a given multi-term query q, is cal-
culated as the sum of individual term ( t) scores. In this equation, N
is the number of documents in the collections, d ftis the number of
2These metrics reflect the leaderboard as of 5 May, 2025.
3We used pyserini python implementation for Lucene index search with
BM25.
4We used rank-bm25 python implementation for BM25-Plus.documents containing the term (the document frequency), and tftd
is the number of times term toccurs in document d.Ldis the length
of the document (in terms) and Lavgis the mean of the document
lengths. The logterm calculates the inverse document frequency for
termt. There are two tuning parameters, b, andk1, set to their default
values 0.75 and 1.5, respectively.
BM25-Plus improves on the original BM25 algorithm by lower-
bounding the contribution of a single term occurrence by adding a δ
term to the fraction in Equation 3, to avoid penalisation by long doc-
uments. This helps a single occurrence of a search term to contribute
at least a constant amount to the retrieval score value, regardless of
document length. The δis set to 1.0 by default.
The BM25-Plus also uses Robertson-Walker IDF [27] (i.e.,
log(N+1
d ft)), which tends to zero as d fttends to N, instead of a com-
mon Robertson-Sparck Jones IDF [24] used in Equation 3, which for
a 1-term query containing a term that occurs in more than half the
documents, ranks all documents not containing that term higher than
those that do. This change makes BM25-Plus always consider doc-
uments containing the term to be more relevant than those that do
not.
We use BM25-Plus retrieval in two modes. First, we use it to re-
trieve full code files (i.e., each code file is treated as a document)
based on their relevance to the bug report, until we fill a pre-set con-
text limit. Specifically, we use BM25-Plus to rank all code files in the
codebase by their relevance to the bug report, and include the most
relevant files in the context until we reach to the pre-set context limit
(i.e., 13K, 27K, 50K, and 80K tokens). Then, providing only these
files, we use an LLM (GPT-4o) to find the file and function where the
bug fix should apply. This is a similar approach used by SWE-bench
RAG mechanism [21]. Furthermore, we also use BM25 directly on
code to retrieve bug locations. To do this, using code AST, we divide
code files into separate functions and index them using BM25 as sep-
arate documents. For each code file, we also include any code outside
a function or class in a single document called “MAIN”. Then we use
BM25 to retrieve the function (or MAIN part) which is relevant to
the bug report. We run this baseline using both BM25 using Lucene
search and BM25-Plus.
5 Results
The results of percentage correct localisation at file and function level
are presented in Table 2. We provide a comparison with the open-

Table 2 : Correct Localisation rate (%) on SWE-bench Lite benchmark. Listed are Information Retrieval (IR) and Software Engineering (SE)
approaches.
Approach (IR) LLM% Correct LocalisationApproach (SE) LLM% Correct Localisation
File Function File Function
BM25-Lucene - 33.67% 13.00% Aider [5] GPT-4o 65.33% 32.67%
BM25-Plus - 19.19% 6.40% Agentless 1.5 [2] GPT-4o 69.67% 37.00%
BM25-Plus (13K) GPT-4o 38.00% 13.33% Agentless RepoGraph [3] GPT-4o 71.00% 36.00%
BM25-Plus (27K) GPT-4o 54.18% 24.08% AppMap Navie v2 [8] GPT-4o 56.67% 26.00%
BM25-Plus (50K) GPT-4o 60.00% 25.33% AutoCodeRover [60] GPT-4o 65.00% 33.00%
BM25-Plus (80K) GPT-4o 57.73% 25.77% ReproducedRG [44] GPT-4o 71.67% 39.33%
Meta-RAG GPT-4o 84.67 % 53.00 %
source models that achieve state-of-the-art performance on the SWE-
bench Lite dataset, using GPT-4o as the LLM. The top six models
from the leaderboard are presented, and we can see that our correct
localisation rate is the highest among all methods at both file-level
(with 84.67%) and function level (with 53.0%).
The results of the baseline study show that BM25 is able to identify
the correct files with a rate of 60.0%, in the best case, with 50K
context window, and the correct functions with a rate of 25.77% with
80k context window. Using BM25 independent of an LLM leads to a
lower score (33.67% and 19.19% for Lucene-based and BM25-Plus,
respectively, for file-level and 13% and 6.4% for function-level).
Furthermore, we have observed an average code reduction of
79.8%, ranging between 91% (requests repository) and 61% (django
repository) among the 12 SWE-bench repositories. Table 1 shows the
total number of code token alongside the number of summary tokens
generated for a sample instance from each of the 12 repositories in
the SWE-bench dataset.
Table 3 : The average Time taken, tokens used, and cost –based on Au-
gust 2024 OpenAI pricing list– to resolve instance of each repository
(Repo) from SWE-bench Lite.
Time Taken Tokens Cost
Repo (seconds) Used (USD)
astropy 51.69 21,429.63 1.29 $
django 62.32 33,408.65 2.00 $
flask 25.60 5,320.67 0.32 $
matplotlib 52.15 16,140.09 0.97 $
pylint 36.58 13,292.83 0.80 $
pytest 62.94 15,033.59 0.90 $
requests 45.97 8,718.83 0.52 $
scikit-learn 48.36 14,626.73 0.88 $
seaborn 38.80 8,177.00 0.49 $
sphinx 61.68 12,398.44 0.74 $
sympy 137.29 32,026.42 1.92 $
xarray 78.95 13,284.80 0.80 $
Mean 58.53 16,154.81 0.97 $
6 Discussion
Our results in Table 1 show that we are able to compress the reposito-
ries to achieve a reduction of 79.8% on average. The maximum com-
pression was the requests library at 91.4% (a smaller code repository)
and the minimum was django at 61% (one of the larger repositories).
Initial analysis showed that this is a result of repositories with morespecific domain knowledge requiring more tokens even after sum-
marisation.
Furthermore, by implementing the summary update approach onto
the SWE-bench dataset, we observe a 57.9% reduction in tokens
compared to re-summarising the codebase for each task instance,
making this approach both a computationally efficient and cost-
effective approach. The real-time compression per updated instance
is 79.2%; comparable to the one-off compression of the entire code-
base.
The process of generating summaries for an existing codebase may
initially introduce a one-time overhead cost. However, we show that
the strategic use of these summaries can lead to substantial reductions
in the ongoing costs of utilising LLMs over time. By leveraging sum-
maries, developers can streamline the integration of LLMs into their
workflow, ultimately enhancing efficiency and minimising resource
expenditure in the long run. This shows how summary updates could
be efficient for a code base in the long run.
Employing codebase summaries within a prompt, rather than sub-
mitting the actual code, can offer an additional valuable advantage
in safeguarding the security of proprietary code. This approach cir-
cumvents the need to submit source code to a large language model
(LLM) for various software development tasks, thereby protecting
sensitive information. This security measure is particularly effective
when the summaries are generated and regularly updated by an in-
ternal LLM, ensuring that the proprietary code remains confidential
and secure.
Additionally, these generated summaries serve a dual purpose:
they can enhance existing code documentation, providing additional
context and clarity, and they can assist developers in gaining a deeper
understanding of the codebase. By offering concise and informative
insights, summaries facilitate a more efficient and secure develop-
ment process, empowering developers to work with greater confi-
dence and comprehension alongside, especially alongside AI agents.
The results then display the results of the assessment of localisa-
tion at the file level and the function level. We show that the Meta-
RAG approach achieved the highest correct file-level (84.67%) and
function-level (53.0%) localisation rate amongst the top performing
GPT-4o approaches. The BM25 retrieval baselines were able to lo-
calise the bugs to a reasonable file level (60%) and function level
(25.77%) accuracy, closing in on the more advanced approaches. We
observe that in the case of file level, as we increase the context length,
the accuracy increases. However, once the context length increased
from 50k to 80k, the localisation correctness drops. This confirms the
diminishing attention effect that our approach aims to overcome.
Our approach also reduces the cost of using LLMs for software de-
velopment, particularly within a real-world continuous development
environment. The price in dollars for running one task through our
system ranges from 0.32$ to 2.00$, with the mean price at 0.97$.

This depends not only on repository size, but file sizes as they would
require more iterations of traversal through summary hierarchies for
localisation. We also record the time taken to pass one task through
the system with the Meta-RAG framework. This ranges from 25.60
seconds to 137.29 seconds, with the mean being 58.53 seconds. As
expected, the longest time corresponds to sympy, the largest reposi-
tory, containing approximately 6 million tokens.
Overall, the file-level and function-level correctness in addition to
the cost benefits is a remarkable outcome as bug localisation is a
time-consuming task [43]. Identifying the correct file can save devel-
opers’ time in bug fixing, particularly in cases where the codebase
is large or the developer is inexperienced with it. Thus achieving a
high performance on this metric goes a long way towards a practical
solution in industry. This promising result for localisation can aid in
improving the bug resolution rates for LLM agents, something we
plan to investigate in future work. Finally, achieving a high locali-
sation capability for bug resolution is a proxy to understanding the
usefulness of code summaries and our Meta-RAG approach in help-
ing LLMs code independently in the future.
7 Conclusion and Future Work
In this paper, we introduce a multi-agent LLM-based framework de-
signed to utilise information retrieval methods for bug localisation.
We evaluated the system’s capabilities on the popular SWE-bench
Lite benchmark, achieving state-of-the art file-level and function-
level correct bug localisation rates. This is as a result of the use of
code summaries for context retrieval and Meta-RAG, both introduced
in this paper. We also showed that summarisation of the codebase
reduces its size in tokens by approximately 80% in average. This re-
duces the cost of using LLM-based agents for software development
down-stream tasks that require code retrieval in the long run.
The configuration of the agents was set up to tackle bug localisa-
tion, in addition to limited context window length, and diminishing
attention challenges. In the future, we plan to evaluate and build on
this work for the downstream task of bug resolution and even new
feature implementation.
Disclaimer
This paper was prepared for informational purposes by the Artifi-
cial Intelligence Research group of JPMorgan Chase & Co. and its
affiliates ("JP Morgan”) and is not a product of the Research Depart-
ment of JP Morgan. JP Morgan makes no representation and war-
ranty whatsoever and disclaims all liability, for the completeness,
accuracy or reliability of the information contained herein. This doc-
ument is not intended as investment research or investment advice,
or a recommendation, offer or solicitation for the purchase or sale of
any security, financial instrument, financial product or service, or to
be used in any way for evaluating the merits of participating in any
transaction, and shall not constitute a solicitation under any jurisdic-
tion or to any person, if such solicitation under such jurisdiction or
to such person would be unlawful.
References
[1] V . Agarwal, Y . Pei, S. Alamir, and X. Liu. Codemirage: Halluci-
nations in code generated by large language models. arXiv preprint
arXiv:2408.08333 , 2024.
[2] Agnetless 1.5. Agnetless 1.5 swe-bench lite evaluation. URL
https://github.com/swe-bench/experiments/tree/main/evaluation/lite/
20241028_agentless-1.5_gpt4o.[3] Agnetless RepoGraph. Agnetless repograph swe-bench lite eval-
uation. URL https://github.com/swe-bench/experiments/tree/main/
evaluation/lite/20240808_RepoGraph_gpt4o.
[4] T. Ahmed and P. Devanbu. Few-shot training llms for project-specific
code-summarization. In Proceedings of the 37th IEEE/ACM Inter-
national Conference on Automated Software Engineering , pages 1–5,
2022.
[5] Aider 2024. Aider, 2024. URL https://github.com/swe-bench/
experiments/tree/main/evaluation/lite/20240523_aider.
[6] A. Alami, V . V . Jensen, and N. A. Ernst. Accountability in code review:
The role of intrinsic drivers and the impact of llms. ACM Transactions
on Software Engineering and Methodology , 2025.
[7] S. Alamir, P. Babkin, N. Navarro, and S. Shah. Ai for automated
code updates. In Proceedings of the 44th International Conference on
Software Engineering: Software Engineering in Practice , pages 25–26,
2022.
[8] AppMap Navie v2. Appmap navie v2 swe-bench lite evaluation.
URL https://github.com/swe-bench/experiments/tree/main/evaluation/
lite/20240615_appmap-navie_gpt4o.
[9] J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan,
E. Jiang, C. Cai, M. Terry, Q. Le, et al. Program synthesis with large
language models. arXiv preprint arXiv:2108.07732 , 2021.
[10] R. Bairi, A. Sonwane, A. Kanade, A. Iyer, S. Parthasarathy, S. Raja-
mani, B. Ashok, and S. Shet. Codeplan: Repository-level coding using
llms and planning. Proceedings of the ACM on Software Engineering ,
1(FSE):675–698, 2024.
[11] I. Bouzenia, P. Devanbu, and M. Pradel. Repairagent: An autonomous,
llm-based agent for program repair. arXiv preprint arXiv:2403.17134 ,
2024.
[12] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Ka-
plan, H. Edwards, Y . Burda, N. Joseph, G. Brockman, A. Ray, R. Puri,
G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan,
S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian,
C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis,
E. Barnes, A. Herbert-V oss, W. H. Guss, A. Nichol, A. Paino, N. Tezak,
J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse,
A. N. Carr, J. Leike, J. Achiam, V . Misra, E. Morikawa, A. Radford,
M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. Mc-
Grew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba. Eval-
uating large language models trained on code, 2021.
[13] H. F. Eniser, H. Zhang, C. David, M. Wang, B. Paulsen, J. Dodds, and
D. Kroening. Towards translating real-world code with llms: A study of
translating to rust. arXiv preprint arXiv:2405.11514 , 2024.
[14] A. Fan, B. Gokkaya, M. Harman, M. Lyubarskiy, S. Sengupta, S. Yoo,
and J. M. Zhang. Large language models for software engineering:
Survey and open problems. In 2023 IEEE/ACM International Confer-
ence on Software Engineering: Future of Software Engineering (ICSE-
FoSE) , pages 31–53. IEEE, 2023.
[15] Z. Feng, D. Guo, D. Tang, N. Duan, X. Feng, M. Gong, L. Shou, B. Qin,
T. Liu, D. Jiang, and M. Zhou. Codebert: A pre-trained model for pro-
gramming and natural languages, 2020.
[16] A. Goyal and N. Sardana. Machine learning or information retrieval
techniques for bug triaging: Which is better? E-Informatica Software
Engineering Journal , 11:117–141, 01 2017. doi: 10.5277/e-Inf170106.
[17] A. Hindle, E. T. Barr, Z. Su, M. Gabel, and P. Devanbu. On the natu-
ralness of software. In 2012 34th International Conference on Software
Engineering (ICSE) , pages 837–847, 2012. doi: 10.1109/ICSE.2012.
6227135.
[18] R. I. T. Jensen, V . Tawosi, and S. Alamir. Software vulnerability and
functionality assessment using llms. In 2024 IEEE/ACM International
Workshop on Natural Language-Based Software Engineering (NLBSE) ,
pages 25–28. IEEE, 2024.
[19] J. Jiang, F. Wang, J. Shen, S. Kim, and S. Kim. A survey on large
language models for code generation. arXiv preprint arXiv:2406.00515 ,
2024.
[20] X. Jiang, Y . Dong, L. Wang, F. Zheng, Q. Shang, G. Li, Z. Jin, and
W. Jiao. Self-planning code generation with large language models.
ACM Transactions on Software Engineering and Methodology , 2023.
[21] C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and
K. Narasimhan. Swe-bench: Can language models resolve real-world
github issues? CoRR , abs/2310.06770, 2023. doi: 10.48550/ARXIV .
2310.06770. URL https://doi.org/10.48550/arXiv.2310.06770.
[22] M. Jin, S. Shahriar, M. Tufano, X. Shi, S. Lu, N. Sundaresan, and
A. Svyatkovskiy. Inferfix: End-to-end program repair with llms. In Pro-
ceedings of the 31st ACM Joint European Software Engineering Con-
ference and Symposium on the Foundations of Software Engineering ,
ESEC/FSE 2023, page 1646–1656, New York, NY , USA, 2023. Asso-
ciation for Computing Machinery. ISBN 9798400703270. doi: 10.1145/

3611643.3613892. URL https://doi.org/10.1145/3611643.3613892.
[23] M. Jin, S. Shahriar, M. Tufano, X. Shi, S. Lu, N. Sundaresan, and
A. Svyatkovskiy. Inferfix: End-to-end program repair with llms. In Pro-
ceedings of the 31st ACM Joint European Software Engineering Con-
ference and Symposium on the Foundations of Software Engineering ,
pages 1646–1656, 2023.
[24] K. S. Jones, S. Walker, and S. E. Robertson. A probabilistic model of
information retrieval: development and comparative experiments: Part
2.Information processing & management , 36(6):809–840, 2000.
[25] A. N. Lam, A. T. Nguyen, H. A. Nguyen, and T. N. Nguyen. Com-
bining deep learning with information retrieval to localize buggy files
for bug reports (n). In 2015 30th IEEE/ACM International Conference
on Automated Software Engineering (ASE) , pages 476–481, 2015. doi:
10.1109/ASE.2015.73.
[26] A. N. Lam, A. T. Nguyen, H. A. Nguyen, and T. N. Nguyen. Bug local-
ization with combination of deep learning and information retrieval. In
2017 IEEE/ACM 25th International Conference on Program Compre-
hension (ICPC) , pages 218–229, 2017. doi: 10.1109/ICPC.2017.24.
[27] L. Lee. Idf revisited: a simple new derivation within the robertson-
spärck jones probabilistic model. In Proceedings of the 30th annual
international ACM SIGIR conference on Research and development in
information retrieval , pages 751–752, 2007.
[28] R. Li, L. B. Allal, Y . Zi, N. Muennighoff, D. Kocetkov, C. Mou,
M. Marone, C. Akiki, J. Li, J. Chim, Q. Liu, E. Zheltonozhskii, T. Y .
Zhuo, T. Wang, O. Dehaene, M. Davaadorj, J. Lamy-Poirier, J. Mon-
teiro, O. Shliazhko, N. Gontier, N. Meade, A. Zebaze, M.-H. Yee,
L. K. Umapathi, J. Zhu, B. Lipkin, M. Oblokulov, Z. Wang, R. Murthy,
J. Stillerman, S. S. Patel, D. Abulkhanov, M. Zocca, M. Dey, Z. Zhang,
N. Fahmy, U. Bhattacharyya, W. Yu, S. Singh, S. Luccioni, P. Ville-
gas, M. Kunakov, F. Zhdanov, M. Romero, T. Lee, N. Timor, J. Ding,
C. Schlesinger, H. Schoelkopf, J. Ebert, T. Dao, M. Mishra, A. Gu,
J. Robinson, C. J. Anderson, B. Dolan-Gavitt, D. Contractor, S. Reddy,
D. Fried, D. Bahdanau, Y . Jernite, C. M. Ferrandis, S. Hughes, T. Wolf,
A. Guha, L. von Werra, and H. de Vries. Starcoder: may the source be
with you!, 2023.
[29] T. Li, G. Zhang, Q. D. Do, X. Yue, and W. Chen. Long-
context llms struggle with long in-context learning. arXiv preprint
arXiv:2404.02060 , 2024.
[30] K. Liu, Y . Liu, Z. Chen, J. M. Zhang, Y . Han, Y . Ma, G. Li, and
G. Huang. Llm-powered test case generation for detecting tricky bugs.
arXiv preprint arXiv:2404.10304 , 2024.
[31] A. Lozhkov, R. Li, L. B. Allal, F. Cassano, J. Lamy-Poirier, N. Tazi,
A. Tang, D. Pykhtar, J. Liu, Y . Wei, et al. Starcoder 2 and the stack v2:
The next generation. arXiv preprint arXiv:2402.19173 , 2024.
[32] Y . Lv and C. Zhai. Lower-bounding term frequency normalization. In
Proceedings of the 20th ACM international conference on Information
and knowledge management , pages 7–16, 2011.
[33] E. Mansur, J. Chen, M. A. Raza, and M. Wardat. Ragfix: Enhancing llm
code repair using rag and stack overflow posts. In 2024 IEEE Interna-
tional Conference on Big Data (BigData) , pages 7491–7496, 2024. doi:
10.1109/BigData62323.2024.10825785.
[34] A. N. Meyer, E. T. Barr, C. Bird, and T. Zimmermann. Today was a
good day: The daily life of software developers. IEEE Transactions on
Software Engineering , 47(05):863–880, may 2021. ISSN 1939-3520.
doi: 10.1109/TSE.2019.2904957.
[35] A. M. Mohsen, H. Hassan, R. Moawad, and S. H. Makady. A review
on software bug localization techniques using a motivational example.
International Journal of Advanced Computer Science and Applications ,
2022. URL https://api.semanticscholar.org/CorpusID:247211009.
[36] S. Muvva, A. E. Rao, and S. Chimalakonda. Bugl – a cross-language
dataset for bug localization, 2020. URL https://arxiv.org/abs/2004.
08846.
[37] D. Nam, A. Macvean, V . Hellendoorn, B. Vasilescu, and B. Myers. Us-
ing an llm to help with code understanding. In Proceedings of the
IEEE/ACM 46th International Conference on Software Engineering ,
pages 1–13, 2024.
[38] OpenAI. OpenAI Models - GPT-4o, 2024. URL https://platform.openai.
com/docs/models/gpt-4o.
[39] C. Packer, V . Fang, S. G. Patil, K. Lin, S. Wooders, and J. E. Gon-
zalez. Memgpt: Towards llms as operating systems. arXiv preprint
arXiv:2310.08560 , 2023.
[40] R. Pan, A. R. Ibrahimzada, R. Krishna, D. Sankar, L. P. Wassi, M. Mer-
ler, B. Sobolev, R. Pavuluri, S. Sinha, and R. Jabbarvand. Lost in
translation: A study of bugs introduced by large language models while
translating code. In Proceedings of the IEEE/ACM 46th International
Conference on Software Engineering , pages 1–13, 2024.
[41] J. S. Park, J. O’Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S.
Bernstein. Generative agents: Interactive simulacra of human behavior.InProceedings of the 36th annual acm symposium on user interface
software and technology , pages 1–22, 2023.
[42] N. Patton, K. Rahmani, M. Missula, J. Biswas, and I. Dillig.
Programming-by-demonstration for long-horizon robot tasks. Proceed-
ings of the ACM on Programming Languages , 8(POPL):512–545, 2024.
[43] S. Polisetty, A. Miranskyy, and A. Ba¸ sar. On usefulness of the deep-
learning-based bug localization models to practitioners. In Proceed-
ings of the Fifteenth International Conference on Predictive Models and
Data Analytics in Software Engineering , pages 16–25, 2019.
[44] ReproducedRG. Reproducedrg swe-bench lite evaluation. URL
https://github.com/swe-bench/experiments/tree/main/evaluation/lite/
20241117_reproducedRG_gpt4o.
[45] B. Rozière, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. E. Tan, Y . Adi,
J. Liu, R. Sauvestre, T. Remez, J. Rapin, A. Kozhevnikov, I. Evtimov,
J. Bitton, M. Bhatt, C. C. Ferrer, A. Grattafiori, W. Xiong, A. Défossez,
J. Copet, F. Azhar, H. Touvron, L. Martin, N. Usunier, T. Scialom, and
G. Synnaeve. Code llama: Open foundation models for code, 2024.
[46] G. Ryan, S. Jain, M. Shang, S. Wang, X. Ma, M. K. Ramanathan, and
B. Ray. Code-aware prompting: A study of coverage-guided test gen-
eration in regression setting using llm. Proceedings of the ACM on
Software Engineering , 1(FSE):951–971, 2024.
[47] R. K. Saha, M. Lease, S. Khurshid, and D. E. Perry. Improving
bug localization using structured information retrieval. In 2013 28th
IEEE/ACM International Conference on Automated Software Engineer-
ing (ASE) , pages 345–355, 2013. doi: 10.1109/ASE.2013.6693093.
[48] W. Sun, Y . Miao, Y . Li, H. Zhang, C. Fang, Y . Liu, G. Deng, Y . Liu,
and Z. Chen. Source code summarization in the era of large language
models. arXiv preprint arXiv:2407.07959 , 2024.
[49] A. Svyatkovskiy, Y . Zhao, S. Fu, and N. Sundaresan. Pythia: Ai-assisted
code completion system. In Proceedings of the 25th ACM SIGKDD
international conference on knowledge discovery & data mining , pages
2727–2735, 2019.
[50] SWE-bench Experiments. Swe-bench experiments. URL https://github.
com/swe-bench/experiments.
[51] SWE-bench Lite. Swe-bench lite. URL https://www.swebench.com/
lite.html.
[52] A. Takahashi, N. Sae-Lim, S. Hayashi, and M. Saeki. An extensive
study on smell-aware bug localization. Journal of Systems and Soft-
ware , 178:110986, Aug. 2021. ISSN 0164-1212. doi: 10.1016/j.jss.
2021.110986. URL http://dx.doi.org/10.1016/j.jss.2021.110986.
[53] V . Tawosi, S. Alamir, and X. Liu. Search-based optimisation of llm
learning shots for story point estimation. In International Symposium
on Search Based Software Engineering , pages 123–129. Springer, 2023.
[54] C. Wang, K. Huang, J. Zhang, Y . Feng, L. Zhang, Y . Liu, and X. Peng.
How and why llms use deprecated apis in code completion? an empiri-
cal study. arXiv preprint arXiv:2406.09834 , 2024.
[55] Y . Wang, W. Wang, S. Joty, and S. C. H. Hoi. Codet5: Identifier-aware
unified pre-trained encoder-decoder models for code understanding and
generation, 2021.
[56] C. Watson, N. Cooper, D. N. Palacio, K. Moran, and D. Poshyvanyk.
A systematic literature review on the use of deep learning in software
engineering research. ACM Transactions on Software Engineering and
Methodology (TOSEM) , 31(2):1–58, 2022.
[57] C. S. Xia, Y . Wei, and L. Zhang. Automated program repair in the era
of large pre-trained language models. In 2023 IEEE/ACM 45th Interna-
tional Conference on Software Engineering (ICSE) , pages 1482–1494.
IEEE, 2023.
[58] C. S. Xia, Y . Deng, S. Dunn, and L. Zhang. Agentless: De-
mystifying llm-based software engineering agents. arXiv preprint
arXiv:2407.01489 , 2024.
[59] J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan,
and O. Press. Swe-agent: Agent-computer interfaces enable automated
software engineering. arXiv preprint arXiv:2405.15793 , 2024.
[60] Y . Zhang, H. Ruan, Z. Fan, and A. Roychoudhury. Autocoderover:
Autonomous program improvement. arXiv preprint arXiv:2404.05427 ,
2024.
[61] J. Zhou, H. Zhang, and D. Lo. Where should the bugs be fixed? more
accurate information retrieval-based bug localization based on bug re-
ports. In 2012 34th International Conference on Software Engineering
(ICSE) , pages 14–24, 2012. doi: 10.1109/ICSE.2012.6227210.
[62] S. Zhou, F. F. Xu, H. Zhu, X. Zhou, R. Lo, A. Sridhar, X. Cheng, T. Ou,
Y . Bisk, D. Fried, et al. Webarena: A realistic web environment for
building autonomous agents. arXiv preprint arXiv:2307.13854 , 2023.
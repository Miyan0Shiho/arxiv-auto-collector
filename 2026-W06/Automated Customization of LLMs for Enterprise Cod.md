# Automated Customization of LLMs for Enterprise Code Repositories Using Semantic Scopes

**Authors**: Ulrich Finkler, Irene Manotas, Wei Zhang, Geert Janssen, Octavian Popescu, Shyam Ramji

**Published**: 2026-02-05 15:38:54

**PDF URL**: [https://arxiv.org/pdf/2602.05780v1](https://arxiv.org/pdf/2602.05780v1)

## Abstract
Code completion (CC) is a task frequently used by developers when working in collaboration with LLM-based programming assistants. Despite the increased performance of LLMs on public benchmarks, out of the box LLMs still have a hard time generating code that aligns with a private code repository not previously seen by the model's training data. Customizing code LLMs to a private repository provides a way to improve the model performance. In this paper we present our approach for automated LLM customization based on semantic scopes in the code. We evaluate LLMs on real industry cases with two private enterprise code repositories with two customization strategies: Retrieval-Augmented Generation (RAG) and supervised Fine-Tuning (FT). Our mechanism for ingesting the repository's data and formulating the training data pairs with semantic scopes helps models to learn the underlying patterns specific to the repository, providing more precise code to developers and helping to boost their productivity. The code completions of moderately sized customized models can be significantly better than those of uncustomized models of much larger capacity. We also include an analysis of customization on two public benchmarks and present opportunities for future work.

## Full Text


<!-- PDF content starts -->

Automated Customization of LLMs for Enterprise Code Reposi tories
Using Semantic Scopes
Ulrich Finkler
email uﬁnkler@us.ibm.comIrene Manotas
email irene.manotas@ibm.comWei Zhang
email weiz@us.ibm.com
Geert Janssen
email geert@us.ibm.comOctavian Popescu
email o.popescu@us.ibm.comShyam Ramji
email ramji@us.ibm.com
IBM Research
Yorktown Heights
New York, USA
February 6, 2026
Abstract
Code completion (CC) is a task frequently used
by developers when working in collaboration with
LLM-based programming assistants. Despite the
increased performance of LLMs on public bench-
marks, out of the box LLMs still have a hard time
generating code that aligns with a private code
repository not previously seen by the model’s train-
ing data. Customizing code LLMs to a private
repository provides a way to improve the model
performance. In this paper we present our ap-
proach for automated LLM customization based on
semantic scopes in the code. We evaluate LLMs
on real industry cases with two private enterprise
code repositories with two customization strategies:
Retrieval-Augmented Generation (RAG) andsuper-
vised Fine-Tuning (FT). Our mechanism for ingest-
ing the repository’s data and formulating the train-
ing data pairs with semantic scopes helps models to
learn the underlying patterns speciﬁc to the reposi-
tory, providing more precise code to developers and
helping to boost their productivity. The code com-
pletions of moderately sized customized models can
be signiﬁcantly better than those of uncustomized
models of much larger capacity. We also include an
analysis of customization on two public benchmarks
and present opportunities for future work.1 Introduction
Generating code snippets is still considered a sig-
niﬁcant challenge in code intelligence [17, 15]. Us-
ing LLMs for large and well established propri-
etary code repositories poses additional diﬃculties.
The primary issue is that the code of a proprietary
repository has not been seen by even the most ad-
vanced LLMs during training. Traditional bench-
marks evaluating code completion or code genera-
tion as isolated tasks (i.e., not in the context of a
repository) can present inﬂated and misleading re-
sults compared to repository-level code tasks [18].
Enterprise code often has a speciﬁc style, best
practice set and customized functionality, even
for common tasks as error handling and logging.
We worked closely with developers from enterprise
repositoriestogatherkeyaspectsrelatedtothecode
completion task. Developers identiﬁed the ’eﬀort to
value’ ratio as a key measurement. Having to write
a prompt even half the size of the desired code was
pointed out as undesirable. ’Near perfection and
conciseness’ of the prediction was another aspect of
high importance. Experienced developers consid-
ered ﬁxing a poor prediction a similar amount of
work than writing from scratch, just as having to
read and trim excessively long predictions. Last,
but not least, the latency of the prediction was in-
dicated as highly relevant. Having to wait more
1arXiv:2602.05780v1  [cs.SE]  5 Feb 2026

Automated Customization of LLMs for Enterprise Code Reposi tories Using Semantic Scopes
than a couple of seconds for a prediction was rated
poorly.
Although LLMs have shown great progress and
performance results for a variety of tasks, more re-
cently Small Language Models (SLMs) have also
shown to be suﬃciently powerful, inherently more
suitable, and necessarily more economical for many
invocations [1]. Diﬀerent to what previous work
have presented when comparing Retrieval Aug-
mented Generation (RAG) and ﬁne tuning (FT) for
code completion [14], we show that ﬁne tuning lan-
guage models on semantic scopes for code comple-
tion is better than RAG when using identical infor-
mation.
Based on the developers’ input, we focused on a
’minimal eﬀort’ scenario to obtain small but high
quality predictions as for example ﬁlling in the ar-
guments of a function call or adding error han-
dling following the project speciﬁc style. The ’min-
imal eﬀort’ consisted of pressing a ’hot-key’ with no
prompt to generate a prediction that completed a
’code scope’ at the current cursor position. Larger
predictions can be assembled by triggering the ’hot-
key’ more than once. The following example illus-
trates thegaps of even highly rated and large ’out of
the box’ models when applied for proprietary code.
The task is to ﬁll out the arguments for a repo-
speciﬁc logging call. The desired result (ground
truth) is the following:
PD_DEV | PD_STACK,
SQLT_rocmPLFSetPaths,
0, 0, rc, logPoint,
PD_LEVEL_SEV,
PD_3TUPLES_BEGIN,
PD_STRING( pLeftPath ),
PD_STRING( pRightPath ),
PD_UINT( bufLens ),
PD_STRING( &( dirPath[ 0 ] ) ),
PD_3TUPLES_END )
which includes a tuple of variable arguments that
contains symbols from the code prior to the call.
Given the entire start of the ﬁle up to the desired
completion, GPT-OSS-120B with a speciﬁc prompt
to complete the function call yields the following
result after manually extracting the code snippet
from the actual prediction text:
SQLT_rocmPLFSetPaths, rc, logPoint,
PD_STRING( dirPath) )
which correctly predicts 3 of the requiredarguments
and one of the optional arguments, but without thenecessary ’begin/end’ frame. Analogously, GPT-
OSS-20B predicts
logPoint, rc, "rocmPLFSetPaths" )
which swaps the argument order of the 3 predicted
required arguments. Furthermore, this result pro-
vides an incorrect location indicator and does not
cover the variable arguments. Importantly, even
with a situation speciﬁc prompt and manual ex-
traction of the code out of the ’wordy’ prediction
the GPT-OSS models were not able to get close to
the style and ’dialect’ due to the lack of knowledge
about the proprietary repository.
Even with explicit instructions to only produce
the completion code, both models generated large
amounts of output surrounding the desired snip-
pet and both models required 30 seconds or more
(GPT-OSS-120B up to 100 seconds) on a dedicated
H100 GPU with a warmed up model, depending on
the speciﬁc query. In comparison, our customized
granite-8b-code-instruct model produced in about 1
second the following result:
PD_EXT,
SQLT_rocmPLFSetPaths,
0, 0, rc, logPoint,
PD_LEVEL_ERR,
PD_3TUPLES_BEGIN,
PD_STRING( pLeftPath ),
PD_STRING( pRightPath ),
PD_UINT( bufLens ),
PD_3TUPLES_END )
predicting usable values for all required arguments
and 3 out of the 4 desired optional arguments in the
correct style. Importantly, this highlights a general
challenge of code completion. Models cannot reli-
ably predict discretionary choices, e.g. the severity
level and optional information, unless they are spec-
iﬁed in detail in the prompt (which would severely
degrade the eﬀort to value ratio). Hence, ’guessing’
good choices for discretionary values based on the
’style’ of a large code base is challenging.
In this paper, we present our method for auto-
mated preparation of training data from a large
repository such that a ﬁne tuned model generates
correct and concise code snippets using the pro-
prietary deﬁnitions, naming conventions and cod-
ing style of the repository without requiring a user
prompt. Our method is based on a novel approach
to create training data pairs based on semantic
2

Automated Customization of LLMs for Enterprise Code Reposi tories Using Semantic Scopes
scopes (see section 3) in the code without human
labor.
To support on-prem adaptation of models, we
customized a language model for two large propri-
etary repositories containing thousands of ﬁles. Our
evaluation on large proprietary repositories shows
that customizing the models via supervised Fine-
Tuning (FT) based on semantic scopes achieves the
best performance compared to both oﬀ-the-shelf
(pretrained) models and Retrieval-Augmented Gen-
eration (RAG). This work makes the following con-
tributions:
1) Introducing an automatic repository data inges-
tion pipeline to prepare training data based on se-
mantic scopes.
2) Creating a prototype for customization of LLMs
on repository-level data using these training data.
3) Presenting our evaluation of the so customized
language models and comparing RAG and FT and
out-of-the box language models.
This paper is structured as follows. In Section
2 we present related work. In Section 3 we intro-
duceourapproachforrepository-level andsemantic-
scope based data preparation. In Section 4, we de-
scribe our customization of LLMs for code tasks.
In Section 5, we outline the evaluation of our ap-
proach on both proprietary enterprise repositories
and public benchmarks. Finally, we close by outlin-
ing opportunities for future work.
2 Background
2.1 Repository-level Context for Coding
Tasks
Several benchmarks and previous work have an-
alyzed and proposed techniques to address the
LLM code generation task from a natural language
prompt without considering the repository context.
Insuchcases, anaturallanguagequerydescribesthe
purposeof the code to be generated [6, 5]. However,
many of these benchmarks (such as HumanEval)
have been identiﬁed as not adequately reﬂecting
practical development scenarios [7].
In contrast, repository-level coding tasks refer to
real-world development scenarios in the presence of
an existing code base that can provide contextual
information and also serve as a knowledge base.2.2 LLM Code Completion and Post
Training Strategies
Several approaches have explored how to leverage
Retrieval Augmented Generation (RAG) [3, 15],
prompting strategies [7, 14] and post-training (ﬁne-
tuning) for the code completion [13] and generation
tasks. DRACO [3], a dataﬂow-guided retrieval aug-
mentation approach improves pre-trained LLMs for
the code completion task inside Python reposito-
ries. CodeRAG-Bench [15] explored RAG for code
generation with diﬀerent retrievers and generators,
and found that retrievers often struggle to fetch
useful contexts, and generators face limitations in
using those contexts eﬀectively. We explore RAG
and Fine Tuning (FT) for customizing models on
the code completion task based on semantic scopes
from the code. We evaluate these two customiza-
tion strategies on two large proprietary repositories
covering Java and C/C++ code.
Post-training LLMs using a curriculum dataset
approach is proposed in [13] by extracting hard-to-
complete patterns from code repositories and gen-
erating context examples using semantic and static
analysis tools. The work shows that all ﬁne-tuned
models improve for code completion. Performance
gains are more pronounced for smaller parameter
models. Wang et. al. [14] conduct training at
the ﬁle level using source code for customizing code
LLMs forthecodecompletion task [14]. Diﬀerent to
previouswork, westudyandshowhowpost-training
of LLMs for code completion at the repository-level,
on two large and proprietary enterprise reposito-
ries for two diﬀerent languages (Java and C++),
perform and compare to a RAG customization ap-
proach.
2.3 Benchmarks
Severalrepository-level benchmarksforcodingtasks
have emerged in the last couple of years [11,
4, 9, 18]. RepoBench-C[11] is a benchmark
speciﬁcally designed for evaluating repository-level
code auto-completion systems with evaluation for
Python and Java; CrossCodeEval (CCEval) [4] is
a benchmark for code completion based on a di-
verse set of real-world, open-sourced, permissively-
licensed repositories, including a Java dataset and
three additional datasets for Python, TypeScript,
3

Automated Customization of LLMs for Enterprise Code Reposi tories Using Semantic Scopes
Figure 1: Customization Pipeline Based on Semantic Scopes f or Repository-level Code Completion.
and C#. HumanEvo [18] is an evolution-aware
repository-level code generation dataset and an au-
tomated execution-based evaluation tool with 400
samples split evenly between Python and Java.
EvoCodeBench [9] is another benchmark with an
automatic collection pipeline, which constructs new
versions from the latest version of the considered
repositories. The size of the dataset is 275 samples
collected from 25 repositories.
Although these benchmarks can provide insights
about how a pre-trained LLM performs for diﬀer-
ent coding tasks, they do not provide a training
set to investigate how a tuned model compares to
other non-tuning strategies. Therefore, we leverage
repository-level code benchmarks having not only
test samples but also data for post-training, such as
RepoBench [11] and CrossCodeEval (CCEval) [4],
to evaluate the tuning of LLMs for code completion
and code generation.
3 Automated Data Preparation
and Model Customization
Large industry software projects often have their
own coding style and ’dialect’, i.e., there is a lower
level collection of proprietary software artifacts.
Higher level software artifacts are built using those
lower level artifacts. When working with AI tools,
two questions arise:
(1) How does one automatically extract training
data from a large software repository (often many
GB) to teach a model the coding style and dialect?
(2) How to teach a model to ‘guess’ how much code
completion should a given LLM generate without
requiring a prompt?
Figure 1 shows the general automated customiza-
tion pipeline. We ingest the repository data andcreate sample pairs for code completion (and poten-
tially other coding tasks) based on semantic scopes
of a program. In this section we describe how our
language model customization approach works for
the code completion task.
3.1 Automated Data Preparation
In formal semantics, which focuses on the ’mean-
ing’ of text elements and their relationships, a se-
mantic scope is a collection of related text elements,
for example parts of text to which a semantic oper-
ator like negation may apply. Analogously, code
has semantic scopes that are independent of the
syntax or programming language. Since semantic
scopes are based on meaning, they remain intact
even when compiling for example C++ code to as-
sembler, which ﬂattens the syntactic structure to a
large degree. Figure 2 shows two examples of the
semantic scope identiﬁcation from a C/C++ code
snippet.
Semantic scopes provide language independent
units to generate training data. Pairing a code pre-
ﬁx with a code snippet that forms a semantic scope
teaches themodel to generate codecompletion snip-
pets with a ’meaning’ and with proper end-of-text
tags. In this way, the generated code snippets com-
plete the current semantic scope to achieve mean-
ingful conciseness without requiring a speciﬁcation
from the developer.
Extracting semantic scopes can be challenging
and may require data and control ﬂow analysis.
But in modern programming languages with func-
tions, classes and diﬀerent types of clauses, seman-
tic scopes often coincide with syntactic scopes. The
enterprise repositories we considered used Java and
C++. For these languages, the ’bodies’ between
matching brackets and parentheses are usable scope
4

Automated Customization of LLMs for Enterprise Code Reposi tories Using Semantic Scopes
Figure 2: Semantic Scope based Repository Data Ingestion. L eft side shows the process ﬂow; Right side
shows two examples of semantic scopes (shaded boxes).
candidates as are syntactic constructs found in the
syntax tree, e.g. function and method deﬁnitions or
the bodies of loops and conditionals.
Semantic scopes are recursive, i.e., a larger se-
mantic scope as ’code to load data and compute X’
contains a scope loading data and another to ’com-
pute X’. Hence, a repository yields scope candidates
of widely varying sizes and compositions that need
to be ﬁltered. Filter options for semantic scopes can
be thesizeof the scope, the depth(e.g. how many
levels of nested brackets are inside a scope), edit
time stamps, and keywords (e.g. deprecated sym-
bols). For a given repository and language, a search
for good ﬁlter parameters can yield useful improve-
ments in model performance.
Inourexperiments scopesbetween at least 50and
at most 1 ,000 bytes yielded good results. Another
useful ﬁlter option is the amount of code available
that precedes a scope. A minimum of 200 bytes has
proven useful in our experiments.
Our goal is to teach the model to complete the
currently open semantic scope. For example if the
input code ends with the opening parenthesis of a
method invocation , the model should complete the
argument list . If the inputcode ends with a position
inside a method’s deﬁnition, the model should com-
plete the method’s deﬁnition. To ingest and pro-
cess data from a repository for customization our
pipeline includes the processes described below.
Data Selection: In this step the pipeline collects all
code ﬁles from the repository and performs an iso-lation process where the ﬁles are selected by their
programming language. Data from each ﬁle in the
repository isstored inaJSONobject withmetadata
including the path to the ﬁle, the date, etc..
Scope Identiﬁcation: After selecting the code ﬁles
in a repository, we process each ﬁle to extract scope
candidates and apply the chosen ﬁlters.
Generating Completion Pairs: The generation of
code completion pairs begins with the continuous
code sections containing the ‘preﬁx’ and the ‘scope’
identiﬁed as described above. These sequences are
partitioned into pairs of a ‘query’ and a ‘label’. A
primary partitioning location is the start byte of the
scope, yielding ‘primary pairs’ . Additional pairs,
withastart at arandomnumberofbytesbehindthe
primary location for each candidate, can increase
the robustness of the ﬁnetuned model. Each ‘label’
is terminated with an ‘end-of-text’ token.
3.2 LLM Customization
To improve the performance of LLMs for a given
repository, we explore two strategies: Retrieval
Augmented Generation (RAG) and Fine Tuning.
3.2.1 RAG
The primary pairs are the basis for the RAG knowl-
edge base. The queriesare embedded into a high
dimensional vector space with a suitable model, for
example from the sentence transformers collection.
The embedding of the query forms the search key.
5

Automated Customization of LLMs for Enterprise Code Reposi tories Using Semantic Scopes
Thecodeof thecorrespondinglabel formsthevalue.
These key-value pairs are loaded into a suitable vec-
tor database. An inference query is embedded and
its top-N nearest neighbors are identiﬁed. The val-
ues of these nearest neighbors are used to augment
the inference query.
RAG introduces sourcesof errors. Thevector em-
bedding is an approximation for the correlation be-
tween queries, i.e. code preﬁxes. Hence the top-N
nearest neighbors may not include the best solution
candidates. Furthermore, even similar preﬁxes may
use diﬀerent symbols and thus lead to the selection
of symbols not occurring in the query preﬁx when
generating the code snippet. Last, but not least,
RAG has very limited opportunity to teach a model
to stop well by only being able to provide a number
of examples.
3.2.2 Fine Tuning
The primary pairs are also the basis of ﬁne tuning.
Since the‘preﬁxes’ can belarger than the ‘label’, for
example we allow up to 3kB of code as a preﬁx, the
masking of the query in the loss calculation as well
as the addition of an end-of-text token is essential
to obtain concise models. The ‘random start’ data
pairs can be used to augment the primary data for
training. It increases the robustness of the predic-
tion if the cursor is not at the beginning of a scope.
4 Methodology
In this section we outline our methodology to con-
duct experiments on enterprise repository data and
public benchmarks.
4.1 Models
Smaller models are gaining attention due to their
ease of deployment and training [2] and low la-
tency, amongst other reasons. This is particularly
true for on-prem deployment. Hence, we selected
two small models for customization, Llama-3.1-
8B-instruct (Llama) and Granite-8B-code-instruct
(Granite). To compare with bigger models, we in-
clude as a baseline model Qwen2.5-72B.Table 1: CC Datasets (number of sample pairs)
Lang. Splits Total
Train Test
DataB (prop) C/C++ 200,000 173200,173
STM (prop) Java 316,609 156316,765
RepoBench (pub) Java 6,956 1,740 8,696
CCEval (pub) Java 1,711 428 2,139
4.2 Datasets
We evaluated the performance of our customization
approach on two large enterprise code repositories:
DataB and STM. The summary of the datasets in-
gested from the private enterprise repositories are
shown in Table 1. Here we present the general in-
formation of these repositories.
DataB: A large Database management applica-
tion repository with around 27K ﬁles covering 13
million lines of C/C++ code.
STM: A large enterprise cloud-based platform
for asset and facilities lifecycle management with
around 13K ﬁles covering 1 .3 million lines of Java
code.
Table 2 summarizes the test case numbers by test
group that were manually curated from each enter-
prise repository. Test data were created manually
in close collaboration with the developers from a
set of hold out ﬁles, simulating the scenario of a
developer employing code completion while writing
new code. Developers directed the types of scenar-
ios and checked the quality. Additionally, a search
was performed to ensure that the desired prediction
was not present in the training data.
Excluding the entire ﬁle from which a test sample
was drawn proved essential to achieve decent corre-
lation between performance measurements and user
experience. The ﬁles in the repositories tended to
be relatively large and the similarities within a sin-
gle ﬁle tended to exceed those across ﬁles. Hence,
including parts of ﬁles from which tests were drawn
in the training data yielded overly positive measure-
ments compared to the experience of a real user re-
questing completions in new code.
Besides the above large enterprise repositories,
we selected two public repository-level benchmarks
to evaluate the performance of our customization
6

Automated Customization of LLMs for Enterprise Code Reposi tories Using Semantic Scopes
Table 2: Test Cases by Semantic Scope Categories
for STM and DataB Enterprise Repositories Sum-
mary
Category # Test Cases
DataB STM
elsebody 13 20
forbody 14 16
funcbody 16 38
ifbody 14 30
logging 36 19
funccall 80 33
approach: CCEval and RepoBench.
CrossCodeEval (CCEval) [4]. A Code Comple-
tion (CC) benchmark for single line completion.
Thegoal is to evaluate in-depthcross-ﬁle contextual
understanding to complete the code accurately.
RepoBench [11]. Samples functions from real-
world projects and evaluates how LLMs generate
these functions according to the target function
description and project context.
Comparingthestatisticsintable1showsthelarge
size diﬀerence between enterprise and public repos-
itories.
4.3 Evaluation Metrics
TheLevenshtein distance [8] is a well established
distance metric for text comparison and is directly
correlated to the eﬀort to correct a prediction into
the desired solution. Levenshtein distance also sat-
isﬁes the requirements of a mathematical metric as
for example the Euclidian distance. The proper-
ties of a mathematical metric are the foundation for
reasoning with measurements in most ﬁelds of en-
gineering. To compare the generated code with the
reference label, a.k.a. ground truth, we considered
two types of Levenshtein distances: full and opti-
mal.
The full Levenshtein distance ( Full) between a
prediction and the desired solution helps to under-
stand how close a part of the prediction to the de-
sired answer and how concise the prediction is. The
Fullmetric is higher if a prediction is very diﬀer-
ent from the desired ground truth, and it is higher
when the prediction is excessively long relative tothe desired solution.
The optimal prediction preﬁx ( Opt) is the be-
ginning of the prediction, varying the length, that
achieves the lowest Levenshtein distance relative to
the desired ground truth. This measurement is low
if the beginning of the prediction is close to the
desired solution, but is not degraded by additional
content.
The diﬀerence between FullandOptreﬂects an
approximation of how much undesired additional
content is predicted. For a conciseprediction, the
diﬀerence between FullandOptis close to zero. For
a nearly exact match, the Optdistance is also close
to zero.
Some popular metrics, e.g. BLEU score or exe-
cutability, havesigniﬁcantdisadvantages inthecode
completion scenario. In code, order is highly rele-
vant. Swapping two arguments in a function call
may render it already incorrect. The BLEU score
(and many other scores for text) do not take or-
der suﬃciently into account [10]. Scores limited to
a range, e.g. 0 to 1, in general do not satisfy the
properties of a mathematical metric.
Measurements that count exact match or require
parsing or compiling may assign the same result,
fail, to a random string and a prediction that re-
quires only the alteration of a single character to
match the desired outcome. The measurement is
far from proportional to developer eﬀort to check
and ﬁx the solution. Furthermore, the code in ’code
completion’ is by deﬁnition incomplete and may not
parse even for a perfect prediction.
4.4 Model Customization
Given a set Sof ’preﬁx-scope’ pairs (200 ,000 for
DataB and 260000 for STM), we evaluated two cus-
tomization strategies, RAG and supervisedﬁnetun-
ing (FT) utilizing Sfor the knowledge base and
training data, respectively.
4.4.1 RAG
To embed code sections, we used
all-MiniLM-L6-v2 from the sentence transformer
python package. To avoid errors due to approxima-
tion algorithms in top-N nearest-neighbor searches
we used a linear cost exact search based on cosine
7

Automated Customization of LLMs for Enterprise Code Reposi tories Using Semantic Scopes
Figure 3: Performance (optimal and full Levenshtein Distan ce) of Baseline Models shown in logarithmic
scale. (Left) Results on DataB repository’s test set. (Righ t) Results on STM repository’s test set. Shorter
bar is better.
distance. A small number of neighbors (3 or 5) as
query augmentation preceding the code yielded the
best results.
4.4.2 FT
Our FT training pipeline used the Huggingface
Trainer class and infrastructure with an Adam op-
timizer. One of the major hurdles was numerical
instability. Relatively smallchanges inhyperparam-
eters lead to unexpectedly large variations in model
performance. A change in version of the underly-
ing model lead in some cases to signiﬁcant degrada-
tion, despite preserving the number of parameters,
the underlying transformer stack and overall archi-
tecture to a large degree and despite of reaching a
similar ﬁnal training loss.
Careful investigation of the progression of train-
ing loss, gradient norm and test evaluation metrics
of intermediate checkpoints revealed that numeri-
cal instability was a major contributor. Literature
suggests that the Adam Optimizer can be a source
of numerical instability [12][16]. Indeed, oﬄoading
the optimizer to CPU, where it is executed in 32 bit
ﬂoatingpointarithmetic, reducedthefrequencyand
height of gradient norm spikes. Avoiding padding
and packing also improves the stability of conver-
gence.
Once gradient norm spikes are controlled, test
metrics tend to show a ’saturation level’ that canbe reached via diﬀerent training settings and some-
timesevenviadiﬀerentbasemodels, suggestingthat
the information in the training data dominates the
outcome.
In addition to the Levenshtein metrics, we gath-
ered developer feedback that tested the completion
capability in their work ﬂows.
5 Results
The most impoartant evaluation of our customiza-
tions was the feedback of developers that tested it
in their normal work environment. Here are couple
of testimonies:
“I would be interested in using [custom model] be-
cause the code suggestion there were more accu-
rate. Suggestions from [custom model] were concise
and easy to modify if required. Results for [custom
model] were consistent”
“Yes, the custom model, although absolutely not
tailored towards my use cases outside of DataB
code, performed surprisingly well and much better
than the [uncustomized] version - it outmatched the
other in every single test or was at least on par in
general tinkering outside of provided samples.”
The uncustomized version used in the user study
employed theGranite-8B model. Thecustom model
was an FT-customized version (as described above)
of that Granite-8B model. In general, users ex-
8

Automated Customization of LLMs for Enterprise Code Reposi tories Using Semantic Scopes
pressed a preference for the customized model for
code completion with respect to conciseness and
quality.
5.1 Measurements
Additionally, we measured the performance of base-
line models and customized models for both propri-
etary repositories and for the public benchmarks in
terms of the metrics described in Section 4.3.
The GPT-OSS models are reasoning chat models
and thus not suited for the code completion task
without a situation speciﬁc prompt, making a fair
comparison in the ’minimum eﬀort’ scenario infea-
sible. Figure 3 shows results for Qwen, Llama and
Granite, which were able to generate an at least
decent start of a prediction without a situation spe-
ciﬁc prompt. All the ’out of the box’ models gener-
ated excessive amounts of code without a situation
speciﬁc prompt. For the ’out of the box’ models,
the smaller Llama-8B and Granite-8B proved to be
competitive for the code completion task on enter-
prise data across the test categories, even relative
to Qwen2.5-72B. Hence, Llama-8b and Granite-8B
will be our references when investigating the beneﬁt
of customization.
Tables 3 and 4 show the improvements of RAG
and FT over the baseline Llama-8b and Granite-
8B models. The measurement that correlated best
with human evaluations was the full Levenshtein
distance, which vastly improved with ﬁne tuning in
all cases. For the ’opt’ distance, the improvements
vary depending on test category and repository.
Note that the metrics are distance measurements
(lower is better). Due to the ’discretionary options’
in predictions, a perfect match (distance zero) is ex-
tremely unlikely and hence there is a ’lower bound’
for what the models can realistically achieve. Hu-
man inspection reveals that a change in the ’opt’
Levenshtein distance by 10 edits is signiﬁcant for
measurements in the vicinity of 100, for example
from predicting the correct number of required ar-
guments with correct types to having one or two ar-
guments out of order, with incorrect type or missing
when ﬁlling out an argument list as in the example
in the introduction.Table 3: Baseline, RAG, and FT for DataB.
DataB Granite-8b Granite-8b Granite-8b
Tests Base RAG FT
Opt FullOpt FullOpt Full
elsebody 18610,260 17115,138 189 189
forbody 28410,942 30110,627 238 815
funcbody 4048,631 3389,359 3211,266
ifbody 18810,174 19410,897 229 239
logging 3479,783 17010,188 129 144
funccall 849,834 829,560 42 45
DataB Llama-8b Llama-8b Llama-8b
Tests Base RAG FT
Opt FullOpt FullOpt Full
elsebody 20613,741 22213,605 194 203
forbody 32211,716 30911,672 315 326
funcbody 45212,462 45012,024 363 368
ifbody 1,070 12,135 24810,760 256 280
logging 47012,954 40012,309 157 172
funccall 10811,592 10111,083 59 70
Table 4: Baseline, RAG, and FT for STM.
STM Granite-8b Granite-8b Granite-8b
Tests Base RAG FT
Opt FullOpt FullOpt Full
elsebody 607,296 587,868 78157
forbody 142 7,808 1467,374 153159
funcbody 3435,210 2266,152 224266
ifbody 13210,173 2266,152 144197
logging 408,046 397,656 2930
funccall 407,288 377,912 2121
STM Llama-8b Llama-8b Llama-8b
Tests Base RAG FT
Opt FullOpt FullOpt Full
elsebody 11811,174 9212,010 42393
forbody 98711,779 16912,918 163186
funcbody 1,210 11,295 26412,300 195225
ifbody 43710,678 14811,438 133183
logging 3913,316 6112,395 44 44
funccall 5610,914 5010,825 4352
The repositories consistency of style and the ap-
plication of best practices within a category has
a strong inﬂuence on how well a model can guess
successfully without requiring a prompt. For the
vast majority ofthe24combinations thecustomized
models show improvements in opt, which are for
some combinations very large. RAG customized
models failed consistently to achieve conciseness.
Categories with more consistent patterns as func-
tion body, logging and function calls enabled the
models to adapt better than categories with more
variability. For the full Levenshtein distance the
9

Automated Customization of LLMs for Enterprise Code Reposi tories Using Semantic Scopes
FT customized models produced the best result in
all combinations.
Table 5: Baseline and FT Model Results for CCE-
Val and RepoBench benchmarks.
Granite
Benchmark Base FT 3ep FT 10ep
Opt FullOptFullOptFull
CCEeval 293,125 32342930
RBench 204,138 3036 99
Llama
Benchmark Base FT 3ep FT 10ep
CCEeval 518,445 41424041
RBench 749,510 25252121
Table5showsresultsforFTcustomization forthe
public benchmarks, conﬁrming the potential for im-
provement across diﬀerent data sets and base mod-
els. The progression of the distance measurements
withtrainingillustrates theimportanceofgoodcon-
vergence and suﬃcient training epochs for the ’opt’
distance.
6 Conclusions
We presented our repository-level automated data
preparation and model customization methodology
on enterprise repository data for code completion.
Our results demonstrate how the strategy of ingest-
ing and creating training and testing samples based
on semantic scopes can helpamodel understandthe
implicit usage patterns from a repository and pre-
dict moreaccurate andconcise codethat aligns with
the ground truth without requiring human labor to
generate training or RAG data. Furthermore we
discussedmethodology aspectsaspropertiesofmea-
surements for thecode completion scenario, numeri-
cal instabilities and inaccuracies in nearest neighbor
searches and options to reduce their impact on con-
vergence and prediction quality.
User testimony and quantitative evaluation on
large private repositories with two base models and
multiple categories of developer guided tests shows
that customizing the models via supervised FT
achieves the best performance compared to oﬀ-the-
shelf models as well as oﬀ-the-shelf models com-
bined with RAG. It also establishes that customiza-
tion of smaller models, with much lower responselatencies, can provide a superior user experience
even compared to out-of-the box models that are 10
times larger and hence have correspondingly higher
response latencies. After eﬀort-to-value ratio and
quality and conciseness, latency was an important
criterion for user experience.
7 Future Work
Infutureworkwewillexpandtheimplementation of
the semantic scope-based data ingestion and model
customization to other enterprise repositories, in-
cluding other coding tasks. Also consider post-
training via Reinforcement Learning from Human
Feedback (RLHF), and analyze the eﬀect of cus-
tomized models in Agentic frameworks.
Acknowledgements
We thankXuanLiuandNicholas Fuller forsupport-
ing our work. We thank Aslam Nomani and team
for their technical insight and feedback.
References
[1] Peter Belcak, Greg Heinrich, Shizhe Diao,
Yonggan Fu, Xin Dong, Saurav Muralidha-
ran, YingyanCelineLin, andPavlo Molchanov.
Small language models arethe futureof agentic
ai, 2025.
[2] Yujia Chen, Yang Ye, Zhongqi Li, Yuchi Ma,
and Cuiyun Gao. Smaller but better: Self-
paced knowledge distillation for lightweight yet
eﬀective lcms. Proc. ACM Softw. Eng. ,2(FSE),
June 2025.
[3] Wei Cheng, Yuhan Wu, and Wei Hu. Dataﬂow-
guided retrieval augmentation for repository-
level code completion. In Proceedings of the
62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long
Papers). Association for Computational Lin-
guistics, August 2024.
[4] Yangruibo Ding, Zijian Wang, Wasi Uddin Ah-
mad, Hantian Ding, MingTan, Nihal Jain, Mu-
rali Krishna Ramanathan, Ramesh Nallapati,
10

Automated Customization of LLMs for Enterprise Code Reposi tories Using Semantic Scopes
Parminder Bhatia, Dan Roth, and Bing Xi-
ang. Crosscodeeval: a diverse and multilingual
benchmark for cross-ﬁle code completion. In
Proceedings of the Thirty-Seventh Annual Con-
ference on Neural Information Processing Sys-
tems, NIPS ’23, Red Hook, NY, USA, 2023.
Curran Associates Inc.
[5] Xinyi He, Jiaru Zou, Yun Lin, Mengyu Zhou,
Shi Han, Zejian Yuan, and Dongmei Zhang.
CoCoST: Automatic complex code generation
with online searching and correctness testing.
In Yaser Al-Onaizan, Mohit Bansal, and Yun-
Nung Chen, editors, Proceedings of the 2024
Conference on Empirical Methods in Natural
Language Processing , Miami, Florida, USA,
November2024. AssociationforComputational
Linguistics.
[6] Md. Ashraful Islam, Mohammed Eunus Ali,
and Md Rizwan Parvez. MapCoder: Multi-
agent code generation for competitive prob-
lem solving. In Lun-Wei Ku, Andre Martins,
andVivek Srikumar, editors, Proceedings of the
62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long
Papers), pages 4912–4944, Bangkok, Thailand,
August 2024. Association for Computational
Linguistics.
[7] Juyong Jiang, Fan Wang, Jiasi Shen, Sungju
Kim, and SunghunKim. A survey on large lan-
guage models for code generation. ACM Trans.
Softw. Eng. Methodol. , July 2025.
[8] Vladimir I Levenshtein. Binary codes capable
of correcting deletions, insertions, and rever-
sals.Soviet Physics Doklady. 10 (8) , 1966.
[9] Jia Li, Ge Li, Xuanming Zhang, Yunfei Zhao,
Yihong Dong, Zhi Jin, Binhua Li, Fei Huang,
and Yongbin Li. Evocodebench: an evolv-
ing code generation benchmark with domain-
speciﬁc evaluations. In The Annual Confer-
ence on Neural Information Processing Sys-
tems, NIPS ’24, Red Hook, NY, USA, 2025.
Curran Associates Inc.
[10] Chin-Yew Lin and Franz Josef Och. Auto-
matic evaluation of machine translation qualityusing longest common subsequence and skip-
bigram statistics. In Proceedings of the 42nd
Annual Meeting of the Association for Compu-
tational Linguistics (ACL-04) , pages 605–612,
Barcelona, Spain, July 2004.
[11] Tianyang Liu, Canwen Xu, and Julian
McAuley. Repobench: Benchmarking
repository-level code auto-completion sys-
tems, 2024.
[12] Igor Molybog, Peter Albert, Moya Chen,
Zachary DeVito, David Esiobu, Naman Goyal,
Punit Singh Koura, Sharan Narang, An-
drew Poulton, Ruan Silva, Binh Tang, Diana
Liskovich, Puxin Xu, Yuchen Zhang, Melanie
Kambadur, Stephen Roller, and Susan Zhang.
A theory on adam instability in large-scale ma-
chine learning, 2023.
[13] HiteshSagtani, RishabhMehrotra, andBeyang
Liu. Improving ﬁm code completions via con-
text & curriculum based learning. In Pro-
ceedings of the Eighteenth ACM International
Conference on Web Search and Data Mining ,
WSDM ’25, New York, NY, USA, 2025. Asso-
ciation for Computing Machinery.
[14] Chaozheng Wang, Zezhou Yang, Shuzheng
Gao, Cuiyun Gao, Ting Peng, Hailiang Huang,
Yuetang Deng, and Michael Lyu. Rag or ﬁne-
tuning? a comparative study on lcms-based
code completion in industry, 2025.
[15] Zora Zhiruo Wang, Akari Asai, Xinyan Ve-
locity Yu, Frank F. Xu, Yiqing Xie, Graham
Neubig, and Daniel Fried. CodeRAG-bench:
Can retrieval augment code generation? In
Luis Chiruzzo, Alan Ritter, and Lu Wang, edi-
tors,Findings of the Association for Computa-
tional Linguistics: NAACL 2025 , pages 3199–
3214. Association for Computational Linguis-
tics, April 2025.
[16] Juyoung Yun. Stabilizing backpropagation in
16-bit neural training with modiﬁed adam op-
timizer, 2025.
[17] Daoguang Zan, Bei Chen, Fengji Zhang, Dian-
jie Lu, Bingchao Wu, Bei Guan, Wang Yongji,
11

Automated Customization of LLMs for Enterprise Code Reposi tories Using Semantic Scopes
and Jian-Guang Lou. Large language models
meet NL2Code: A survey. In Anna Rogers,
Jordan Boyd-Graber, and Naoaki Okazaki, ed-
itors,Proceedings of the 61st Annual Meeting of
the Association for Computational Linguistics
(Volume 1: Long Papers) , Toronto, Canada,
July 2023. Association for Computational Lin-
guistics.
[18] Dewu Zheng, Yanlin Wang, Ensheng Shi,
Ruikai Zhang, Yuchi Ma, Hongyu Zhang, and
Zibin Zheng. HumanEvo: An Evolution-aware
Benchmark for More Realistic Evaluation of
Repository-level Code Generation . In 2025
IEEE/ACM 47th International Conference on
Software Engineering (ICSE) , Los Alamitos,
CA, USA, 2025. IEEE Computer Society.
12
# DynaCI: Generating GitHub Actions Workflows via Dynamic Analysis

**Authors**: Javid Ditty, Rahul Ghanshyam Chavan, Nimmi Weeraddana, Foyzul Hassan

**Published**: 2026-10-21 15:32:56

**PDF URL**: [https://arxiv.org/pdf/2610.29890v1](https://arxiv.org/pdf/2610.29890v1)

## Abstract
Continuous Integration (CI) is an essential practice that automatically validates code changes through builds and tests. However, configuring CI workflows remains challenging, as modern software often features complex architectures, numerous dependencies, and support for diverse execution environments. Most CI platforms also require workflows to be written in their particular domain-specific languages. As a result, CI workflows are often error-prone and hard to maintain, especially for those unfamiliar with CI semantics. Prior work on automated CI workflow generation require developers to create specifications that describe their intended workflows using natural language, partial workflows, etc. However, in practice, developers may struggle to articulate requirements and may be unaware of available CI features. To address these limitations, we propose DynaCI, a novel approach that uses runtime information from developers' local environments to generate CI workflows. DynaCI (i) traces the developers' local environment to collect runtime information, (ii) constructs an initial workflow by applying runtime-informed heuristics, and (iii) iteratively refines the workflow via LLM-based Retrieval Augmented Generation (RAG). We evaluate DynaCI on 50 popular Python projects that use GitHub Actions, a popular CI platform, and compare it against the state-of-the-art CI workflow generation approaches: GH-WCOM and LLM4GW. DynaCI outperforms these existing approaches by more than 21.13% in BLEU score and 20.96% in F1 score on average. It also generates workflows without missing functionality in 56% of cases, while requiring fewer manual modifications to repair such exceptions. These results demonstrate that leveraging runtime information substantially improves LLM-based CI workflow generation, and offer a promising direction for broader DevOps configuration automation.

## Full Text


<!-- PDF content starts -->

DynaCI: Generating GitHub Actions Workflows
via Dynamic Analysis
1stJavid Ditty
University of Michigan-Dearborn
Dearborn, Michigan, USA
jditty@umich.edu2ndRahul Ghanshyam Chavan
University of Michigan-Dearborn
Dearborn, Michigan, USA
rchavan@umich.edu
3rdNimmi Weeraddana
University of Calgary
Calgary, Alberta, Canada
nimmi.weeraddana@ucalgary.ca4thFoyzul Hassan
University of Michigan-Dearborn
Dearborn, Michigan, USA
foyzul@umich.edu
Abstract—Continuous Integration (CI) is an essential practice
that automatically validates code changes through builds and
tests. However, configuring CI workflows remains challenging, as
modern software often features complex architectures, numerous
dependencies, and support for diverse execution environments.
Most CI platforms also require workflows to be written in their
particular domain-specific languages. As a result, CI workflows
are often error-prone and hard to maintain, especially for those
unfamiliar with CI semantics. Prior work on automated CI
workflow generation require developers to create specifications
that describe their intended workflows using natural language,
partial workflows, etc. However, in practice, developers may
struggle to articulate requirements and may be unaware of
available CI features.
To address these limitations, we proposeDynaCI, a novel
approach that uses runtime information from developers’ local
environments to generate CI workflows.DynaCI(i) traces the
developers’ local environment to collect runtime information,
(ii) constructs an initial workflow by applying runtime-informed
heuristics, and (iii) iteratively refines the workflow via LLM-
based Retrieval Augmented Generation (RAG). We evaluate
DynaCIon 50 popular Python projects that use GitHub Actions,
a popular CI platform, and compare it against the state-of-the-
art CI workflow generation approaches:GH-WCOMandLLM4GW.
DynaCIoutperforms these existing approaches by more than
21.13% in BLEU score and 20.96% in F1 score on average. It
also generates workflows without missing functionality in 56% of
cases, while requiring fewer manual modifications to repair such
exceptions. These results demonstrate that leveraging runtime
information substantially improves LLM-based CI workflow
generation, and offer a promising direction for broader DevOps
configuration automation.
Index Terms—Continuous integration, Large language models
I. INTRODUCTION
Continuous Integration (CI) is a popular software engineer-
ing practice that automates the code integration process in
software repositories [1]. Prior work shows that CI enables
faster pull request acceptance and frequent releases [2], in-
creases the number of change merges [3], and boosts developer
productivity, efficiency, and confidence [4].While there are many CI platforms [5], [6], [7], GitHub
Actions (GHA) remains the most widely used [2], [8], with at
least 71.8% of popular GitHub repositories containing awork-
flowsince 2023 [9], [10]. Since December 2025, developers
can define a GHA workflow in a YAML using 84 different
components, such as event triggers, jobs, steps, actions, run-
ners, matrices, containers, secrets, and conditional rules [11].
With such a capacity for extensive customization, configuring
CI workflows is inherently challenging for developers. Saroar
et al. [12] surveyed 90 developers and found that 60.87% often
encounter difficulties writing workflows. Among these, the
most common challenges were related to confusing and error-
prone syntax. Zhang et al. [13] studied 6,590 Stack Overflow
questions and 315 GitHub issues, and found that workflow
configuration issues were discussed more than those related
to dependencies, security, integration, deployment, etc. Again,
the most common problem among these issues was workflows’
”nightmare” syntax [13]. Valenzuela-Toledo et al. [14] has
argued that this trend is due, in part, to inadequate tooling
support for writing GHA workflows.
Previous works have attempted to provide this tooling sup-
port. For example, Mastropaolo et al. [15] proposeGH-WCOM,
a transformer-based approach that provides code completion
suggestions for GHA workflows. Zhang et al. [16] propose
LLM4GW, a set of prompting techniques that optimize the
performance of Large Language Models (LLMs) on GHA
workflow generation, syntactic error identification, code in-
jection vulnerability detection, syntactic error fixing, and code
injection vulnerability repair. However, despite their intended
benefits, these tools may fall short in practice, as they still
require inputs that demand substantial domain knowledge of
GHA to produce acceptable configurations.GH-WCOM[15],
for example, still requires a developer to write the majority
of their workflows beforehand so that it can fill in any gaps.
LLM4GW[16] accepts natural language instructions instead,
but developers must provide detailed workflow-, job-, and step-
level specifications for functional correctness.
To address these concerns, we proposeDynaCI, the first
arXiv:2610.29890v1  [cs.SE]  21 Oct 2026

tool, to the best of our knowledge, that combines runtime
analysis with LLMs to synthesize GHA workflows. Our in-
tuition is that a developer’s local environment can serve as
a starting point for extracting the information required for
generating CI workflows. As the developer runs test suites
in their local environment,DynaCIcollects runtime traces
and extracts information about the executed commands, de-
pendencies, and execution order. It then applies heuristics to
select candidate GHA settings (i.e., components), assembles
an initial workflow, and iteratively refines it using LLM-
based Retrieval Augmented Generation (RAG) [17], [18]. We
evaluateDynaCIagainstGH-WCOMandLLM4GW, the state-
of-the-art in GHA workflow synthesis tools, on 50 popular
Python projects to answer the following research questions:
RQ1How effective isDynaCIat generating CI work-
flows?To assess whetherDynaCIoutperforms the state-of-
the-art, we compare the tools’ workflows against their targets
across several popular workflow similarity metrics. We found
thatDynaCIhas a BLEU score [19] of 52.81% and a F1
score of 62.06%, outperforming the state-of-the-art by more
than 21.13% and 20.96%, respectively.
RQ2How time and cost efficient isDynaCI?Since
DynaCIuses an LLM to iteratively refine workflows, we
measure its overhead in terms of elapsed time and LLM token
usage per workflow. We found thatDynaCIincurs a median
overhead of 5.55 minutes and 653,755 tokens per workflow.
RQ3How much modification is needed for CI work-
flows generated byDynaCI?Like other workflow synthesis
tools,DynaCImay generate workflows that lack intended
functionality. In our manual analysis, we found that only
44% ofDynaCIworkflows lacked intended functionality,
whereaseveryworkflow from state-of-the-art tools needed
modification.
This paper makes the following contributions:
●It presentsDynaCI, the first approach, to the best of our
knowledge, that synthesizes GHA workflows using runtime
information and LLMs;
●It conducts an empirical analysis ofDynaCIon 50 popular,
Python-based GitHub projects, showing that it is signifi-
cantly more effective than existing tools with little overhead
and need for modification;
●It conducts an ablation study, showing thatDynaCI’s
hybrid approach plays a substantial role beyond what can
be attributed to its heuristics or the LLM alone.
II. BACKGROUND ANDMOTIVATION
A. GitHub Actions Workflows
GHA workflows automate the building, testing, and deploy-
ment activities for a GitHub repository and are configured
through YAML files. Figure 1 (A) shows an example con-
figuration file that illustrates the main components of a GHA
workflow. A workflow is triggered in response to an event a,
which executes its jobs. A job bis a sequence of steps that
run on a shared machine cto accomplish a set of tasks. It can
initialize environmental variables e, set a maximum number
name : Example
on: push
jobs:
  quality-assurance :
    runs-on : ubuntu-latest
    timeout-minutes : 10
    env:
      TEST_SUITE_ID : 1
    fail-fast : true
    strategy :
      matrix :
        python-version : 
          - ' 3.10'
          - ' 3.11'
          - ' 3.12'
    steps :
      - uses : actions/checkout@v6
      - uses : actions/setup-python@v6
        with:
          python-version : 
           ↳ ${{ matrix.python-version }}
  cache : 'pip' 
      - uses : astral-sh/setup-uv@v6
        with:
          python-version : 
           ↳ ${{ matrix.python-version }}
      - name: Check Lint Rules
run: |
           pylint
      - name: Run Test Suite
    run : |
 pytesta
b
c
d
h
kjife
g
l(A) W orkflow Example (B) Motivating Example
TracersStartsWritesWritesWrites
Developer          <15 Lines Omitted>
steps:
  - <TO_BE_PREDICTED>
  - <FOR-LATER-USE>
  - name: Check Lint Rules
    <FOR-LATER-USE>
  - name: Run Test Suite
    <FOR-LATER-USE>
apt: [uv], pip: [pytest], env: {TEST_ID: 1 }Generate a GitHub W orkflow named `Example` ...
The workflow has one job.
The job id of the 1st job is `quality-assurance`.
This job will run on ubuntu-latest runner .
The job uses a matrix strategy to automatically ...
The variable `python-version` has 3 values: ...
The job sets an environment variable to use: ...
The maximum number of minutes to run the job is ...
< 6 Lines Omitted >
Traced By
Local 
EnvironmentModel
GENERA TE
LLM
GENERA TE
GENERA TE
Heuristics Initial 
WorkflowLLM
REFINE1 2
3
4
SCENARIOS
1= No Tools 2= GH-WCOM
3= LLM4GW 4= DynaCIGH-WCOM
WorkflowManual
Workflow
LLM4GW
Workflow
DynaCI
WorkflowFig. 1:Comparing GHA Workflow Synthesis Across Four Scenarios
of minutes to run d, use a matrix that repeats the job with
different configurations g, and stop queued configurations
if any fails f. A step is a group of shell commands that
accomplish a single task, such as korl, which performs
software quality assurance. An action is a reusable step whose
contents are hosted in an external repository. For example, h
checks out a repository into the runner so that the workflow
can access it, isets up a particular Python version in the
execution environment, and kdoes the same withuv, a
popular Python package manager. These, alongside services,
containers, and 84 additional settings [20], make workflows
difficult to write without supporting tools [12], [13], [14].
B. Motivating Example
Figure 1 (B) presents four scenarios in which a developer
tries to create a workflow. In scenario 1, which is the most
common, the developer attempts to write the entire workflow
without using tools. This requires them to (i) articulate their
needs and (ii) leverage substantial domain knowledge about
GHA. Since most developers do not meet either condition
[12], scenario 2introducesGH-WCOM[15], a tool that auto-
completes any gaps that a developer leaves behind in a GHA
workflow file. While this mitigates the knowledge requirement,
the developer must still be familiar enough with GHA to
write a partial workflow. For those who are not familiar with
writing GHA configuration files,LLM4GW[16] can be used,
as shown in scenario 3, where the developer writes natural
language instructions instead of a (partial) workflow. However,
the instructions required for this scenario can still be complex
and require substantial domain knowledge about GHA, such
that developers may expend effort comparable to that required
in the first scenario. To fill the gaps in the previous scenarios,
we introduceDynaCI—scenario 4in Figure 1 (B).DynaCI
traces the developer’s local environment during execution
and uses the runtime information to generate and refine the
workflow. Besides performing their regular development and
testing tasks and attachingDynaCIto their environment, the
developer is not required to produce additional inputs, such as

A. Execute Commands
Commands
pip install uv~=0.5.30
uv pip install --system .[minimum-jaxlib] -r build/test-requirements.txt
uv pip install --system --pre tensorflow==2.19.0
echo "JAX_NUM_GENERA TED_CASES=$JAX_NUM_GENERA TED_CASES"
echo "JAX_ENABLE_X64=$JAX_ENABLE_X64"
echo "JAX_ENABLE_CHECKS=$JAX_ENABLE_CHECKS"
echo "JAX_SKIP_SLOW_TESTS=$JAX_SKIP_SLOW_TESTS"
pytest -n auto --tb=short --maxfail=20 jax/experimental/jax2tf/tests/jax2tf_test.pyExample From jax-ml/jax
Environment V ariables
JAX_NUM_GENERA TED_CASES: 10
JAX_ENABLE_X64: 0
JAX_ENABLE_CHECKS: true
JAX_SKIP_SLOW_TESTS: true
•••Files Accessed
tests/ann_test.py
tests/aot_test.py
tests/api_test.py
tests/api_util_test.py
•••Packages
APT
python3.1 1
python3.1 1-pip
python3.1 1-venv
•••PIP
hypothesis
pillow
pytest-xdist
•••Developer
Durations
Command 1, 10 seconds
Command 2, 30 seconds
•••
C. Apply Heuristics
Non-Parametric
HeuristicsParametric
Heuristics D. Refine with LLMsLLM
Workflow
LinterRAG Final
Workflow
X
Error
MessagesRuntime
InformationInitial 
WorkflowB. Trace CommandsFile
ProbePackage
ProbeCommand
ProbePython Environ
ProbeShell Environ
ProbeDuration
ProbeFig. 2:Overview ofDynaCI
HACKING.md
Guidelines for Code Modification
Always run make test  before submitting a new PR. pylint ,
flake8 , pytest , doctest  and shellcheck  need to be
installed. (If you don't change any shell scripts you can run 
make test_py  and you don't need the shellcheck
dependency but it's an awesome tool, so check it out : )requirements.txt
flake8
pylint
pytest
setuptoolsDeveloper Commands
python -m pip install --upgrade pip
pip install -r requirements.txt
make test_py Example from
ranger/rangerImplicit
Explicit
Fig. 3:Example of Manually Identifying Developer Commands
workflow components (e.g., 1and 2) or natural language
instructions (e.g., 3).
III. APPROACH
This section presents a detailed overview ofDynaCI, and
Figure 2 shows its main steps: (A) Execute Commands, (B)
Trace Commands, (C) Apply Heuristics, and (D) Refine with
LLMs. Together, these steps produce a GitHub workflow that
reproduces the developer’s local environment. We designed
DynaCIto handle Python environments, since it is a popular
language in projects that use workflows [9], but our approach
could be generalized to other languages, such as Java.
A. Execute Commands
Software testing is crucial for improving code quality and
reliability [21]. Before contributing code changes to shared
repositories, developers typically validate their changes in
local development environments, where test executions haveaccess to project-specific resources such as test files, software
packages, and environment variables [22]. To perform these
validation activities, developers execute build and test com-
mands that exercise the project’s functionality. This funda-
mental software engineering practice helps detect faults before
code changes are pushed to shared repositories, avoiding
unnecessary CI wait times [23].
Developers usually record their build and test
commands in project documentation (e.g.,README.md,
CONTRIBUTING.md,HACKING.md, and webpages) and
configurations (e.g.,pyproject.toml,setup.py) to
help new code contributors implement the practice in their
own local environments. Figure 3 provides an example of
this in theranger/rangerproject. In this example, the
HACKING.mdfile providesexplicitguidance on how to build
and test the project, e.g., runmake test_pyto only test
Python scripts. However, these stated command(s)implicitly
depend on Python dependencies that must be installed (via
pip commands) from therequirements.txtfile.
B. Trace Commands
Once a developer identifies their project’s building and
testing commands (via its documentation and configurations),
they can run those commands in their local environment. To
start generating a CI workflow, the developer provides their
set of commands, as an input file, toDynaCI, which will
execute them in the same environmenttwiceto trace their
runtime information. In the first execution (Section III-B1),
it captures runtime traces, such as durations of commands,
file accesses in system calls, and dependency installations in
package managers, usingstrace[24]. In the second execu-
tion (Section III-B2),DynaCIcaptures environment variable
accesses usingltrace[25]. This separation is necessary
since multipleptracetools, such asstraceandltrace,
cannot attach to the same process at the same time [26]. Below,
we discuss these two execution runs in detail.
1)First Execution:In this execution run,DynaCIattaches
four probes to the developer’s local environment to partition
commands and trace their durations, file accesses, and package
dependencies withstrace[24], which we detail below.
●Command probe.Developers typically execute a sequence
of commands to prepare, build, test, and validate a project in
their local environment. While these commands collectively
accomplish multiple tasks, their boundaries are not explic-
itly demarcated. This poses a challenge when synthesizing
CI workflows, where commands must be organized into se-
mantically meaningful workflow steps [27], [28]. To address
this problem,DynaCIintroduces a Command probe that
analyzes developer command sequences and partitions them
into executable units that can later be mapped to workflow
steps. Each unit is a command or expression node that is
defined at the top-level of a command sequence’s parse
tree [29].
●Duration probe.We develop a Duration probe that cap-
tures the durations of commands identified by a Command

TABLE I:Example Runtime Traces Extracted by the Probes During
the First Execution
Probe Captured Field Example Extracted Data
Command ProbeCommand make ci
LoC 1
Duration ProbeStart Time 2025-07-20 23:19:46
End Time 2025-07-20 23:21:10
Duration 84 seconds
File Probe Accessed Files /app/tests/test requests.py
Package Probeapt Packages python3==3.10.12
pip Packages pytest==2.8.0
probe, i.e., the datetime the execution completes minus the
datetime the execution starts.
●File probe.We develop a File probe that captures references
to files in the commands’ system calls by tracing system
calls that use file paths, such asopenandopenat[30].
●Package probe.If a workflow runs developer commands,
it must install their packages, or dependencies. We develop
a Package probe that identifies these packages by checking
if any of their associated file paths have been accessed.
By the end of the first execution, the four probes have
extracted runtime traces, describing the behavior of each exe-
cuted command. Table I shows an example of the information
extracted for thepsf/requestsproject.
2)Second Execution:In this execution,DynaCItraces
access to environment variables, since they can have sub-
stantial impacts on the behavior of software systems [31].
DynaCIuses the following probes implemented for the shell
and Python interpreter:
●Shell environ probe.This probe captures environment
variable (i.e., environ1) accesses by system library calls,
such asgetenv[33], withltrace[25].
●Python environ probe.This probe captures environment
variable accesses by a language-specific runtime interface.
Many runtimes, such as the Python interpreter and the Java
Virtual Machine, provide an abstraction over the underlying
OS environment. The probe instruments this abstraction
to record environment variables that are accessed during
execution. For Python, we implement the probe using a site-
specific customization script that leverages theenviron
interface in theosmodule to capture environment variables
during interpreter initialization [34]. This approach can be
readily extended to other languages; for example, in Java,
a similar probe can be implemented using agent-based
instrumentation viajava.lang.instrument[35].
C. Apply Heuristics
In this phase,DynaCIapplies heuristics to runtime in-
formation collected from the developer’s local environment
(e.g., commands, dependencies, and environment variables) to
construct an initial workflow that will be refined later.
1)Filter Candidate Workflow Components:As of De-
cember 2025, GHA supports 84 workflow components, each
1environ is a global variable provided by the C standard library that holds
the environment variables on Linux [32].representing a configurable element of a workflow [20]. How-
ever, some components are CI-specific and do not correspond
to configurations that exist in a developer’s local environ-
ment. SinceDynaCIaims to generate workflows from local
environments, it avoids using components that do not have
local counterparts unless their inclusion is required to run
the workflow. We grouped the 84 workflow components into
eight categories using official GHA documentation [20]. For
three of the most important categories, below, we discuss
our inclusion and exclusion criteria for identifying candidate
workflow components.
Environment (27/84), i.e., the components that define key-
value pairs that can be accessed by other workflow compo-
nents: These pairs are either provided to the workflow as
secrets, i.e., confidential settings that are not available to
our research or the general public (21), or as hard-coded
values in the workflow’s file (6). The latter typically specify
environment variables required by the workflow, a job, or an
individual step. Since GitHub-provided inputs are not available
during local execution,DynaCIcan only add environment
variables that are observable in a local execution environment.
Execution (10/84), i.e., the components that interact with
the OS or shell context: These components either configure the
OS and shell environment (7) or execute commands within
these environments (3). We retain the command-execution
components because they directly correspond to commands
that developers run in their local environments. We exclude the
former components fromDynaCIworkflows since their usage
in real-world workflows is exceedingly rare. For example, the
former were only 0.48% of ourCI Workflow Dataset(Sec-
tion III-C3).DynaCI’s workflows thus only include Execution
components that directly execute commands.
Worker (8/84), i.e., the components that manage when jobs
and steps are started or stopped: These components, order job
executions (4), start jobs that meet a condition (2), and timeout
job or step executions (2). Because conditions that trigger jobs
are CI-specific,DynaCI’s workflows can only include Worker
components that order and timeout job and step executions.
2)Apply Non-Parametric Heuristics:Using the runtime
information extracted in Section III-B,DynaCIidentifies
the candidate components discussed in Section III-C1. It
then applies the followingnon-parametric heuristics—rule-
based decisions derived directly from the observed runtime
behavior—to generate a preliminary workflow file. These
heuristics rely solely on the runtime information collected
from the developer’s environment and do not depend on
tunable parameters.2
For each job, set the environment variables extracted
from the runtime traces:In GHA,jobs.<id>.envdefines
environment variables accessible to all steps in a job. Based
on runtime information,DynaCIinitializes this component
with variables accessed in the developer’s local environment.
2Note that some heuristics require parameters whose values must be derived
empirically; theseparametric heuristicsare described in Section III-C3.

For each job, set the configuration matrix
using the programming language version and OS
name extracted from runtime traces:In GHA,
jobs.<id>.strategy.matrixdefines a matrix of
configurations under which a job runs. Most matrices
cover a range of programming language versions, such as
Python 3.10, 3.11, and 3.12, and specify the machines on
which the workflow runs, such as GitHub-hosted Ubuntu,
Windows, and MacOS runners.DynaCIdefines the matrix
in each job using the OS and Python version observed
in runtime traces. However,DynaCIcurrently relies on
Linux-specific tools (e.g.,strace,ltrace) to capture
runtime information, so it only generates workflows for the
GHA Ubuntu runner, accounting for 64.6% of workflow jobs
in GHA in general [36].
For each job, set the execution machine us-
ing the OS defined in the configuration matrix:In
GHA, the machine on which a job runs is specified by
jobs.<id>.runs-oncomponent.DynaCIconfigures the
value of this component with the OS version mentioned in
jobs.<id>.strategy.matrix.
For each job, derive workflow steps from runtime
information:In GHA, the sequence of tasks that a job
performs is specified byjobs.<id>.steps[ *]. According
to runtime information,DynaCIadds a step to each job
that installs theaptandpippackages that are missing
in the developer’s local dependency configuration file (e.g.,
requirements.txt).DynaCIalso groups the developer’s
commands into separate steps. First, commands (from the
Command probe) are partitioned into candidate step groups,
preserving order and ensuring each step contains at least one
command. Second,DynaCIdetermines the total size, in LoC,
and duration of each group using the runtime information
extracted from the Command probe and the Duration probe.
Third, it uses Multi-Criteria Decision Making (MCDM) [37],
[38] to find the step group that minimizes the spread of
command durations across its steps while also maximizing the
total number of steps. This is based on our intuition that there
areprimarycommands, which perform the main activities and
thus run longer (e.g.,pytest), that are surrounded by shorter
secondarycommands, which provide support (e.g.,echo).
For each job, include reusable actions to
prepare the workflow environment:In GHA, actions
or reusable tasks are invoked as steps using the
jobs.<id>.steps[ *].usescomponent.DynaCI
always addsactions/checkout[39] so that the
workflow can access its repository. If the workflow
uses Python or theuvpackage manager,DynaCI
will also addactions/setup-python[40] or
astral-sh/setup-uv[41], respectively. These actions
set up program dependencies in the workflow environment.
3)Apply Parametric Heuristics:Some of GHA compo-
nents require parameter information that cannot be extracted
from runtime traces, and are CI- or project-specific (e.g.,
setting timeout and concurrency parameters). In our study,
we classify such parameters into two categories: thresholdparameters and multiplier parameters. Athreshold parameter
specifies the minimum execution duration a job must have
before a workflow component is included (e.g., enabling
concurrency and timeout settings). This allowsDynaCIto
add such GHA components into a workflow only for potential
longer-running jobs, where such components are more useful.
Amultiplier parameterdetermines the numeric value of a
workflow component by scaling the observed job duration
(e.g., setting timeout limits once the timeout component is
included). This allowsDynaCIto determine the numerical
value for setting certain components that are related to a
job’s runtime duration, i.e., the sum of the durations of
corresponding commands executed in the developer’s local
environment, as reported by the Duration probe.
We derived these parameters from Cardoen et al.’s dataset
of 2,595,399 GHA workflows [10]. We first filtered this
dataset by removing invalid workflows, old versions of valid
workflows, and workflows that are unrelated to Python-based
CI. We consider a workflow to be valid if it conforms to
a GHA workflow schema [42], and a workflow is Python-
based if its repository’s most popular language is Python
according to the GitHub REST API [43]. Doing so resulted
in 15,285 workflows, which we call theCI Workflow Dataset.
Setting aside 50 workflows with the highest star counts for our
Evaluation Dataset(Section IV), we used the GitHub REST
API to extract the last successful job runs, if any, from the
remaining 15,235 workflows. This left us with 7,157 job runs,
which we refer to as theHeuristic-Tuning Dataset. For the
workflow components below, we extracted the durations of job
runs that use them, removed their outliers, and treated their
averages as default threshold and multiplier values. Below,
we list these parameter values and their associated heuristics,
whichDynaCIuses to construct workflows.
Set a timeout limit for job execution
(Threshold = 319s, Multiplier = 13):In GHA,
jobs.<id>.timeout-minutesspecifies the maximum
number of minutes that a job can run before it is forced
to stop. It is included in the workflow if the corresponding
job duration observed in the developer’s local environment
exceeds 319 seconds; otherwise, it is excluded. When
included, its value is set to 13 times the corresponding job
duration observed in the developer’s local environment.
Set concurrency control to prevent long-running jobs
from executing simultaneously (Threshold = 138s, Multi-
plier = N/A):In GHA,jobs.<id>.concurrencypre-
vents jobs that are in the same concurrency group from
running at the same time. These groups are defined as strings
and often include context variables, or useful values that
GHA substitutes into workflows before running them [44].
If included,DynaCI, using contexts, sets its concurrency
group to the name of the workflow followed by the ref that
triggered it (i.e., ${{github.workflow}}-${{github.ref}}).
This prevents multiple, longer-running jobs from starting at
once.
Set fail-fast behavior to allow all jobs in a matrix to run
even if one fails (Threshold = 66s, Multiplier = N/A):In

GHA,jobs.<id>.strategy.fail-fast, if set to true,
will cancel all jobs in a configuration matrix if any job in that
matrix fails. It is included in the workflow and set to ”false”
if its job’s duration exceeds the threshold; otherwise, it is not
included, and the value would be ”true” by default.
By the end of this phase,DynaCIis expected to result in
an initial workflowfile that can then be refined via LLMs.
D. Refine with LLMs
To ensure that there are no missing components,Apply
Heuristicsstep is designed to add components as much as
possible to the workflow. However, this over-permissive design
can make the initial workflow generated byDynaCIoverly
complicated and error-prone. To mitigate this issue,Refine with
LLMsstep is designed to be more restrictive. It uses an LLM
to analyze runtime information, remove unnecessary and/or
problematic workflow components, and patch any syntax er-
rors that arise during refinement.
To support this process, we extract runtime information
about the GHA runner that the workflow is expected to run in,
providing it toDynaCIas CI context (Section III-D1). For
a given initial workflow,DynaCIseparates all the available
runtime information into chunks and encodes the chunks into
embeddings. (Section III-D2). Then,DynaCIretrieves the
most relevant chunks via Retrieval-Augmented Generation
(RAG) [17] to provide the LLM runtime context.DynaCI
finally refines the content of the workflow with the LLM on
two levels: the job-level (Section III-D3) and workflow-level
(Section III-D4).
1)Extract Runtime Environment from GitHub Actions:
In every initial workflow,DynaCIsets environment vari-
ables and installs packages found in the developer’s local
environment. However, in many cases, these features may
already be found on a GHA runner. To minimize redundancies
and potential conflicts,DynaCIprovides runtime information
about the target runner environment to the LLM as additional
context for refinement. We acquired this information manually
by runningapt,pip, andenvcommands on the target
runner.
2)Embed Runtime Information:The runtime information
available from local and CI environments can exceed1GB
in size, soDynaCIcannot pass it all to the LLM without
exploding costs and degrading performance [45], [46]. Instead,
DynaCIchunks and embeds the information in a database,
only sending the most relevant chunks to the LLM (i.e., RAG).
3)Refine at Job-Level:Because each job reflects a differ-
ent entry point in the developer’s local environment,DynaCI
traces them separately. Thus, each job has its own runtime
information. Instead of refining the entire workflow at once,
DynaCIuses adivide-and-conquerapproach; it extracts the
jobs from the initial workflow and improves them separately
(divide). For each job,DynaCIfirst retrieves runtime chunks
with cosine similarity≥60%of its run blocks. Next,DynaCI
sends a job prompt with these chunks to an LLM with in-
structions to remove unneeded components from the job (e.g.,
environment variables, package installations) and to substitutedirect references to OS and Python versions with the job’s
matrix variables, which is a best practice [11]. The prompt
includes information about the GHA environment that the job
will run in (Section III-D1) and warnings against making addi-
tional changes, as preliminary experiments showed a tendency
for the LLM to hallucinate beyond the intended scope.
4)Merge Refined Jobs and Refine at Workflow-Level:
AfterDynaCIreceives the refined job blocks, it instructs
the LLM to merge them into a single workflow (conquer).
To mitigate hallucinations,DynaCIvalidates the generated
workflow usingactionlint, a static analyzer for GHA
workflows [47]. Ifactionlintreturns an error,DynaCI
adds the error message to the workflow prompt and re-
generates the refined workflow. It repeats this process until
actionlintno longer produces an error, or five refinement
iterations have occurred, producing thefinal workflow.
IV. EXPERIMENTALSETUP
To evaluate the performance ofDynaCI, we required a
dataset that establishes a correspondence between CI work-
flows and their counterparts in local environments. We began
with ourEvaluation Datasetof 50 workflows with the highest
star counts that we keep aside during Section III-C3 to
ensure that there is no data leakage between ourHeuristic-
Tuning Datasetand the evaluation setup. For each project
in our sample, we then consulted the projects’ source code
and documentation to create representative local environments
(Section IV-A). Within each environment, we generate a
workflow file usingDynaCI(Section IV-B) as well as the
state-of-the-art workflow generation tools,GH-WCOM[15] and
LLM4GW[48] (Section IV-C). We finally compare these work-
flows against their targets on effectiveness, efficiency, and their
need for modification to answer RQ1–RQ3 (Section IV-D).
A.Local Environment Configuration
Two authors manually inspected each projects’ developer-
facing documentation (e.g.,CONTRIBUTING.md) and con-
figurations (e.g.,pyproject.toml) to identify the com-
mands that developers use to set up their local environments.
For example, inhttpie/cli, theCONTRIBUTING.md
file mentions test-related commands, such asmake test
andpython -m pytest. We found 90.8% of commands
in our sample involve similarly generic commands, such as
tox. To ensure that the reconstructed local environments
were executable and representative of developer machines,
we also added setup commands that were not always ex-
plicitly documented in the source code or project docu-
mentation. For instance, when needed, we installed project
dependencies using commands such aspip install -r
requirements.txtbefore executing the documented test
commands. Previous works used similar approaches to identify
build and test commands [49] as well as dependency installa-
tion commands [50] in documentation and configuration files.
To ensure the reproducibility of the aforementioned environ-
ments, we defined each environment in a Dockerfile that per-
forms the following key steps. First, the Dockerfile clones the

corresponding repository into the container. Second, it installs
DynaCIand its dependencies (i.e.,strace,ltrace, and
actionlint). Third, the Dockerfile installs the most recent
Python version that the project supports as well as the package
manager that the project uses, if any (e.g.,uv). Fourth, it
sets any environment variables with values mentioned in the
project’s source code and documentation. Lastly, the Dock-
erfile executes developer commands that we identified with
the aforementioned approach, and which a developer would
provide as an input file toDynaCI(Section III-B).
B.Workflow File Generation UsingDynaCI
Within each Dockerized environment,DynaCIis config-
ured to execute each of the developer’s commands. Before
their execution,DynaCIattaches its probes into environment
to capture its runtime information.DynaCItransforms this
runtime information into aninitial workflowfile, which then
would be refined via an LLM to create thefinal workflow
file. We treat this final workflow file as the output ofDynaCI
when comparing it with the baseline approaches.
C.Workflow Generation Using Baseline Approaches
To contextualizeDynaCI’s performance, we generate
workflows using several (state-of-the-art) approaches:
Commands-Only, i.e., using only the developer’s com-
mands to generate a workflow:For each project, we instruct
a LLM to produce a GHA workflowonlyusing the commands
that developers would be run in their Dockerized environment.
This baseline evaluates whether commands alone are enough
to produce an acceptable workflow with an LLM.
Heuristics-Only, i.e., usingDynaCI’s heuristics to gen-
erate an initial workflow without refinement:Specifically,
it followsDynaCI’sTrace Commandsstep (Section III-B)
andApply Heuristicsstep (Section III-C) to produce an initial
workflow, omitting the subsequent LLM-based refinement.
This baseline results in more permissive workflows that can
incorporate improbable components, which the refinement step
might have otherwise removed. We include this approach in
our paper asan ablation studyto check whether the refinement
step inDynaCImakes measurable improvements.
GH-WCOM[15], i.e., a transformer-based approach that
provides suggestions for auto-completing GHA workflows:
GH-WCOMis trained on two tasks: (i)Next Statement (NS),
which predicts the next step in a job given the preced-
ing workflow context, and (ii)Job Completion (JC), which
generates the remaining steps of a job given their names.
SinceGH-WCOMrequires incomplete workflows, we remove
runtime-based components, such as steps, from workflows
generated byHeuristics-Only. We then treat workflow
generation as an NS task, sequentially predicting each job’s
next steps until the original number of steps is restored.
LLM4GW[16], i.e., a set of prompt optimization tech-
niques for tasks related to GHA:LLM4GWintroduces five
prompt templates (P1–P5) for workflow generation tasks,
which include an increasing amount of information about
the workflow, jobs, and steps to be generated. However, ouruse case assumes that developers cannot articulate details
about their target workflows, so only prompt P1 ofLLM4GW,
which describes workflow-level information and job IDs, is
comparable toDynaCI. We use their tool to convert the input
inGH-WCOM[15] into a P1 prompt, providing it to the same
LLM thatDynaCIuses to generate a workflow.
D. Performance Metric Computation
1)RQ1. Effectiveness:We used textual similarity metrics
to evaluate how effectivelyDynaCIperforms against its
baselines. For each tool, we computed the average of the
following scores across our 50 projects:
●BLEU Scorecompares two texts on a token-level. It is
commonly used to evaluate program synthesis tools in
general [51], [52], [53], [54], and CI workflow generation
tools in particular [15], [16].
●F1 Scorecaptures how well a given tool balances cor-
rectness and completeness in including a component in
a workflow. Key-value matching metrics, such as the F1
score, are commonly used to evaluate YAML generation
tasks [55], [16], [56]. To compute this, we first record the
number of true-positive (TP), false-positive (FP), and false-
negative (FN) components in each generated workflow. For
example, a component is TP if it is found in both the
target workflow (i.e., developer-written workflow) and the
generated workflow and the values match. We then use these
values to calculate the precision and recall, of which F1
score is the harmonic mean.
2)RQ2. Efficiency:We recorded the time taken by each
tool to generate each workflow. If the tool used an LLM,
we also record the number of tokens it consumed. Together,
these metrics characterize the efficiency ofDynaCIand its
baselines.
3)RQ3. Modification:Similarity between generated and
target workflows (RQ1) does not guarantee similar execution.
To measure the latter, two authors (with two years of CI
research experience as of December 2025) manually executed
all generated workflows in forked projects, iteratively revising
them until their execution logs [57] matched the target work-
flows. The authors recorded each LoC changed in generated
workflows and, from these revisions, derived common repair
categories. We used this information to calculate the following
metrics forDynaCIand its baselines:
●Repair Ratemeasures the proportion of tool-generated
workflows that are missing functionality (e.g., builds, test
cases, etc.) in a generated workflow and thus require repairs.
●Delta LoCrefers to the total LoC inserted, modified, or
deleted to implement missing functionality in a workflow.
E. Hardware / Software Requirements
The Dockerized environments forDynaCIand
Heuristic-Onlywere hosted on a Ubuntu 24.04
machine with an Intel i5-10600K CPU @ 4.10GHz CPU,
a Radeon RX 5700 XT GPU, and 16 GB RAM. We used
OpenAI’s GPT-4o-mini [58] for generation and ADA-002 [59]

for embeddings, though neither model is intrinsic to our
approach and can be replaced with any suitable model.
V. RESULTS
A. RQ1. How effective isDynaCIat generating CI work-
flows?
Table II shows the textual similarity metrics forDynaCI
and our baselines on our evaluation set.
Observation 1:DynaCIdemonstrates improved perfor-
mance over the baselines across BLEU and F1-scores.
We observe thatDynaCIoutperforms the four baseline ap-
proaches in BLEU, achieving a +21.13% higher score on
average. We also found thatDynaCIhas the highest average
F1 score of 62.06%, outperforming all four baselines by
+13.89% and just the state-of-the-art by +20.96%.
The table reports TPs, FPs, and FNs to show howDynaCI’s
refinement step improves overHeuristics-Only. The
Heuristics-Onlyapproach, which corresponds to
DynaCI’sApply Heuristicsstep, is more permissive: it
includes as many runtime-observed components as possible,
but also introduces extraneous components, increasing FPs.
For example, inyt-dlp/yt-dlp,Heuristics-Only
added an environment variable, NO PROXY , from runtime
information to a workflow. However, this variable prevented
network traffic from routing through GHA proxy servers,
resulting in an error.DynaCImitigates this issue with its
more restrictiveRefine with LLMsstep, which excludes as
many extraneous components, such as NO PROXY , from
theApply Heuristicsstep as it can. While this added step
decreases FPs, it risks removing important components in
the process, decreasing TPs and increasing FNs. However,
DynaCIstill manages to balance TPs, FPs, and FNs with
these two steps, improving F1-scores overall. We used a
two-sample t-test to compare F1 scores betweenDynaCI
and the baselines; all differences in Table II are statistically
significant (p<0.05).
Table III reports average component-level F1 scores for
components where tools differ. Below are the main observa-
tions.
Observation 2:DynaCIandHeuristics-Onlygen-
erally perform better on components related to environ-
ment variables, error handling, and command execution.
For example, thejobs.<id>.steps[*].runcomponent runs the
TABLE II:Average Performance of Workflow Generation Tools
Tool BLEU F1 TP FP FN
DynaCI 52.81 62.06 14.22 5.00 13.50
Heuristics-Only 21.31 48.17 14.70 24.64 13.02
GH-WCOM[15] 31.68 41.10 8.04 5.88 19.68
LLM4GW[16] 11.99 18.65 3.00 6.34 24.72
Commands-Only 13.76 36.72 7.88 5.12 19.84
TABLE III:Breakdown of the Average F1 Scores
Component Type DynaCI Heuristics-Only GH-WCOM LLM4GW Commands-Only
jobs.<id>.env.* Environment 2.58 3.73 0.00 0.00 0.00
jobs.<id>.needs Worker 0.00 0.00 0.00 20.00 0.00
jobs.<id>.runs-on Execution 96.05 96.05 96.05 1.95 1.67
jobs.<id>.steps[*].run Execution 72.36 67.32 7.41 9.61 63.57
jobs.<id>.steps[*].uses Miscellaneous 96.80 96.80 91.70 90.51 66.02
jobs.<id>.steps[*].with.* Environment 71.65 71.65 71.65 1.00 0.00
jobs.<id>.strategy.fail-fast Error 32.89 32.89 0.00 0.00 0.00
jobs.<id>.strategy.matrix.* Environment 45.97 47.94 47.94 0.29 0.00TABLE IV:Median Duration of Tools (in Seconds)
Trace Commands Apply Heuristics Refine with LLMs
Tool Total Trace Parse Generate Embed Refine
DynaCI 333.66 147.92 39.52 0.01 133.01 13.20
Heuristics-Only 187.45 147.92 39.52 0.01 - -
GH-WCOM ˜0 - - ˜0 - -
LLM4GW ˜0 - - ˜0 - -
Commands-Only ˜0 - - ˜0 - -
developer’s commands within a shell environment.GH-WCOM,
LLM4GW, andCommands-Onlyoften exclude these devel-
oper’s commands (Section V-C) and thus have poor F1 scores.
Heuristics-Onlyaddsaptandpippackages that the
developer’s local environment accesses.DynaCIdoes the
same but removes packages that are already installed in the
GHA runner environment with an LLM. Since these package-
installation statements are absent from the target workflows,
they are counted as FPs, reducing both tools’ F1 scores.
Next,DynaCIandHeuristics-Onlyapproach outper-
forms the others with respect to thejobs.<id>.strategy.fail-
fastcomponents, which stops running a job’s matrix if a single
job is unsuccessful. While the other tools do not include this
component,DynaCIandHeuristics-Onlydo if the du-
ration of the developer’s commands exceeds a certain threshold
(Section III-C3). We also find that thejobs.<id>.env.*com-
ponent, which defines environmental variables across steps,
exhibits low overall performance. However,DynaCIand
Heuristics-Onlyoutperform the other tools by defining
all environmental variables from their runtime information,
including many that are FPs. SinceGH-WCOM,LLM4GW, and
Commands-Onlydo not define any environment variables,
their F1 scores are 0.
DynaCI,Heuristic-Only, andGH-WCOMconsistently
outperformLLM4GW[13] andCommands-Only, with re-
spect to components that define execution configurations,
such asjobs.<id>.runs-on,jobs.<id>.strategy.matrix., and
jobs.<id>. steps[*].with.*. This is because, unlike other tools,
LLM4GWandCommands-Onlydoes not define or use
strategy.matrixto parameterize configurations. As a
result, it fails to generate multiple configuration variants or
consistently propagate values to dependent components, such
asruns-onandsteps[ *].with. SinceLLM4GWand
Commands-Onlyrely on hard-coded job configurations, they
produce missing variants, inconsistencies, and FNs.
B. RQ2. How time and cost efficient isDynaCI?
Observation 3:DynaCIsynthesizes a workflow within
5 minutes and 33 seconds on average.Table IV reports the
average workflow-generation time forDynaCIand our base-
lines on our evaluation set.Commands-Only,GH-WCOM,
andLLM4GWhad negligible durations.Heuristics-Only
andDynaCItook longer, 187.45 and 333.66 seconds, respec-
tively, due to their additional steps. The longest step isTrace
Commands, which monitor the developer’s local environment
twice. Since designing a workflow from scratch typically
takes much longer than five minutes and 33 seconds for a
developer [13], [14], [12], all tools, includingDynaCI, are
time-efficient.

TABLE V:Median Token Cost per Workflow of Tools
Tool Total Embed Prompt
DynaCI 653,755 568,734 85,021
Heuristics-Only - - -
GH-WCOM - - -
LLM4GW 199 - 199
Commands-Only 868 - 868
TABLE VI:Median / Total Delta LoC by Change Type
Tools Total Changes Insert Delete Modify
DynaCI 4 / 52 2 / 28 1 / 11 1 / 13
Heuristics-Only 4 / 83 2 / 23 1 / 16 1 / 44
GH-WCOM 10 / 592 5 / 340 4 / 200 1 / 52
LLM4GW 10 / 592 5 / 340 4 / 200 1 / 52
Commands-Only 5 / 303 3 / 221 1 / 36 1 / 46
Observation 4: It generally costs 653,755 tokens to
synthesize a workflow withDynaCI.Table V reports the
average token cost per workflow.Commands-Onlyand
LLM4GWconsume only 868 and 199 tokens per workflow,
respectively, compared toDynaCI’s 653,755 tokens. This is
becauseDynaCIembeds and prompts using runtime informa-
tion chunks, which account for most of its token usage. How-
ever, this cost is modest because most repositories use only a
few CI workflows [9], and the performance gains ofDynaCI
outweigh this trade-off.GH-WCOMandHeuristics-Only
do not use LLMs and therefore incur no token cost.
C. RQ3. How much modification is needed for CI workflows
generated byDynaCI?
For theDynaCI-generated workflows that are missing
functionality from their target, Table VI shows the median and
total Delta LoC between the generated and target workflows.
Observation 5:DynaCI’sRefine with LLMsstep re-
moves many problematic components that itsApply
Heuristicsstep introduces.DynaCIrequires the fewest
total changes (52), while having the same median number
of changes (4) asHeuristics-Only. This difference in
total changes is due to the absence of theRefine with LLMs
step in theHeuristics-Onlyapproach. For example,
Heuristics-Onlyattempts to install allaptpackages in
the developer’s local environment. However, there were many
cases in which a package, such as a specific Python version,
could not be found due to a missing package repository in the
GHA environment. These unsupported installations often need
to be removed manually, whereasDynaCIremoves many of
them during LLM refinement.
Observation 6: Developers must change substantially
fewer LoC to repairDynaCIworkflows than baseline-
generated workflows.GH-WCOMandLLM4GW, i.e., our state-
of-the-art baselines, require the most changes (592), with
insertions dominating (340) due to two recurring issues.
First, the tools almost exclusively generate commands that
do not implement the functionality of their target. For exam-
ple,GH-WCOMgenerates a workflow forpallets/flask
that only installs dependencies, e.g.,pip install -r
requirements.txt, and does not lint the project’s source
code like the target workflow. Second, they default to com-
ponent configurations that are usually incompatible with their
projects. For example, the approaches default to Python 3.8 (orTABLE VII:Repair Rate by Repair Category
Repair Category DynaCI Heuristics-Only GH-WCOM LLM4GW Commands-Only
APT PACKAGE NOT RESOLVED 5 35 0 0 48
REFERENCE COMMAND NOT FOUND 8 5 50 50 12
SETUP UV NOT FOUND 1 0 10 10 9
SETUP PYTHON NOT FOUND 1 1 0 0 40
UNSUPPORTED PYTHON VERSION 2 5 37 37 0
BAD ENV V ARIABLE 1 3 0 0 0
ACROSS ALL CATEGORIES (Repaired / Total Projects) 22 / 50 39 / 50 50 / 50 50 / 50 50 / 50
the latest version [40]) when setting up the CI environment,
which many projects do not support.
Together, two authors inspected repairs performed with the
aforementioned modifications, reporting the repair rate for
each tool (Table VII) as well as their repair category.
Observation 7: Substantially fewerDynaCI-generated
workflows require repairs than baseline-generated work-
flows.The table shows that among the 50 workflows in
ourEvaluation Dataset,DynaCIandHeuristics-Only
workflows only require repairs in 22 and 39 cases, respectively.
DynaCIgenerates 28 out of 50 workflows (56%) without
requiring any manual repair. In contrast,GH-WCOM,LLM4GW,
andCommands-Onlyworkflows require repairs inevery
case to add missing functionality. We categorize these repairs
as follows:
APT PACKAGE NOT RESOLVED:When the name of
anaptpackage cannot be resolved, it cannot be installed. In
fact, there areaptrepositories, which host different Ubuntu
packages, that are registered in the developer’s local envi-
ronment but not in the GitHub Ubuntu runner. For example,
ppa:deadsnakes/ppa[60] provides newer Python ver-
sions that are unavailable in the defaultaptrepositories in a
developer’s local environment, but not necessarily in GitHub
Ubuntu runner. As a result,Heuristics-Onlymay install
locally observedaptpackages that are unnecessary/unsup-
ported in GHA.DynaCI’sRefine with LLMsstep mitigates
this issue, though developers may still need to remove prob-
lematic packages or configure the required repositories.
REFERENCE COMMAND NOT FOUND:When a
GHA workflow job does not run the commands that are
executed by the developer in their local environment, it
leads to missing functionality in the workflow. ForDynaCI
andCommands-Only, this occurs when the LLM removes
seemingly extraneous commands (e.g.,echo), modifies string
literals (e.g., regular expressions ingrep), or hallucinates.
ForGH-WCOMandLLM4GW, this arises, in part, because they
lack access to developer-executed commands (e.g.,pytest,
tox, etc.). To repair this problem in the generated workflows,
developers may manually add the missing commands.
SETUP UV NOT FOUND:When a step cannot find the
uv[61] package manager in its execution environment, its
uv-related commands will fail. In our evaluation set, ten
workflows useuv, implying thatGH-WCOMandLLM4GW,
which encountered this problem ten times, failed to install
it in each case. To repair this problem, developers may add
astral-sh/setup-uv[41] to failing jobs.
SETUP PYTHON NOT FOUND:Python commands
will fail if Python is not installed in their execution
environment.Commands-Onlywas the only tool
that frequently produced this problem by refusing to

include actions other thanactions/checkoutto
workflows. To repair this problem, developers may add
actions/setup-python[40] to failing jobs.
UNSUPPORTED PYTHON VERSION:If a job uses a
Python version that is incompatible with its commands, the
workflow may fail. This issue is common forGH-WCOMand
LLM4GW, which install only Python 3.8 or the latest Python
3.x version.Heuristics-OnlyandDynaCImay also en-
counter this issue when multiple Python versions are observed
in the developer’s local environment, since they use only the
latest version when generating the workflow. Developers may
repair this problem in the generated workflow by specifying a
compatible version inactions/setup-python[40].
BAD ENV V ARIABLE:When a job uses an environ-
ment variable that causes unexpected behavior, it can lead
to workflow failures. For example, in our local environment,
mkdocs/mkdocs[62] reads the SOURCE DATE EPOCH
variable, which forces builds to use a specified Unix timestamp
(e.g., 0 represents January 1, 1970) across their artifacts. While
this worked locally, it caused failures in GHA due to its own
builds not supporting timestamps before 1980. To repair this
problem, developers may manually remove such environment
variables that are causing the issues.
VI. THREATS TOVALIDITY
Internal Validity.To fix errors in the workflows, different
reviewers might choose different solutions or modifications,
potentially introducing subjectivity or inconsistency in quan-
tifying the required changes. To mitigate this, the first two
authors independently analyzed 25 projects and reached a con-
sensus on the number of lines required to make bothDynaCI-
generated and baseline-generated workflow files functional.
The first author then applied these criteria to all 50 projects.
External Validity.Our experiments use 50 Python projects
with GitHub Actions, which may limit generalizability to
other languages and CI platforms. However,DynaCIcan be
adapted to these contexts with modest extensions because it
derives workflow components from language-agnostic devel-
oper commands and their runtime behavior in local testing
environments. Besides, many CI platforms have similar fea-
tures [23]; only the components’ syntax would need to be
modified inDynaCI.DynaCI’s divide-and-conquer strategy
also enables future heuristics to be added for specific features
in other languages. WhileDynaCIcurrently only supports
Linux-based OSes, which is the most common runner in
GHA [36], in future, we hope to extendDynaCIto non-
Linux environments, including Windows. Additionally, to eval-
uateDynaCI, we needed projects with workflows whose
commands could run outside CI, and thus, our results may
not fully capture CI-specific features, such as (unofficial)
reusable actions. However, only 2.05% of the projects in our
CI Workflow Dataset, from which our experiment projects
were sampled, used such features. Our project filtering criteria
further ensured that these projects have substantial complexity;
thus, these commands are representative of real-world scenar-
ios and projects.Construct Validity.While BLEU and F1 scores capture
textual similarity (RQ1), they do not fully account for the
correctness of generated workflows in practice. We therefore
complement them with a manual analysis of whether generated
workflows require modifications to match target executions
and how many modifications are needed (RQ3). Together,
workflows are evaluated on a textual and a practical level, of-
fering a more comprehensive evaluation of their performance.
VII. RELATEDWORK
Prior studies have introduced CI workflow generation tools:
GHAST[63],CIGAR[64],PACGBI[65],GH-WCOM[15],
LLM4GW[16], andLLM4CI[48]. Among these tools, the clos-
est to our study areGH-WCOM[15] andLLM4GW[16]. In fact,
GH-WCOM[15] uses LLMs to complete gaps in a workflow
configuration whileDynaCIextends the use of LLMs to gen-
erate workflows from scratch. Then,LLM4GW[16] generates
workflows from scratch, but its prompts require domain knowl-
edge about GHA thatDynaCIdoes not. Overall,DynaCI
reduces developer effort by reusing runtime information from
existing local testing environments.
Another line of research focuses on leveraging runtime
information for source code generation and analysis [66], [67],
[68]. WhileDynaCIalso uses runtime information, it instead
targets CI workflow generation, i.e., configuration synthesis.
Lian et al. [69] show that LLMs are good at validating
configurations, such as HBase, without specifically focusing
on GHA. Those that focused on GHA, for example, Xu et
al. [70], leveraged CI/CD runtime logs to diagnose and repair
failures. However, these approaches do not generate workflows
from scratch. To the best of our knowledge,DynaCIis the
first runtime-informed approach to CI workflow synthesis.
VIII. CONCLUSION ANDIMPLICATIONS
This paper introducesDynaCI, the first system, to the
best of our knowledge, to combine runtime information with
LLMs to synthesize GHA workflows. Our evaluation on 50
Python projects shows thatDynaCIoutperforms state-of-
the-art baselines by generating more complete and precise
workflows. Below, we distill the implications of our results:
Developers may useDynaCIto translate local environ-
ments into effective CI workflows with little cost and mod-
ification.Prior work shows that writing CI workflows is chal-
lenging [12], [13], [14]. Existing tools, such asGH-WCOM[15]
andLLM4GW[16], still require substantial domain knowledge
of GHA. In contrast,DynaCIrelies only on the developer’s
local environment, thus reducing knowledge barriers and effort
requirements. Our Observation 1 shows thatDynaCIis able
to produce more effective workflows than the state-of-the-
art tools. Observations 6 and 7 show thatDynaCI-generated
workflows require fewer modifications, while Observation 4
shows that they can be produced at low cost.
Tool builders may develop recommendation systems
to assist in troubleshooting problems from CI workflow
generation tools, includingDynaCI, which produced work-
flows needing the least modification.CI workflow generation

tools can produce workflows that contain problems. While
Observations 6 and 7 show thatDynaCIworkflows require
substantially fewer modifications than the state-of-the-art, it
can still generate problems, such as missing functionality,
that developers would have to correct. Valenzuela-Toledo et
al. [14] emphasize the need for tools that could troubleshoot
problems in GHA workflows. Tool builders could leverage our
repair categories for GHA workflows (Section V-C) to build
a recommendation system for generated workflow repair.
Researchers may adoptDynaCI’s divide-and-conquer
approach to generate DevOps configurations beyond CI
workflows.Prior work treats CI workflow generation as a
single problem, producing an entire workflow at once [16],
[48].DynaCIbreaks down this problem intomultiplesub-
problems, each targeting an individual component in the
workflow; these components are then integrated through a
robust refinement process to produce a more effective an
efficient workflow that solves the initial problem and requires
little modification (Observation 5). Thus,DynaCIsets the
foundation for a family ofdivide-and-conquerapproaches to
CI workflow generation and beyond. We encourage researchers
to adopt our divide-and-conquer approach to automatically
generate other types of DevOps configurations as well (e.g.,
Docker and Terraform).REFERENCES
[1] B. Vasilescu, Y . Yu, H. Wang, P. Devanbu, and V . Filkov, “Quality
and productivity outcomes relating to continuous integration in github,”
inProceedings of the 2015 10th Joint Meeting on Foundations of
Software Engineering, ser. ESEC/FSE 2015. New York, NY , USA:
Association for Computing Machinery, 2015, p. 805–816. [Online].
Available: https://doi.org/10.1145/2786805.2786850
[2] M. Hilton, T. Tunnell, K. Huang, D. Marinov, and D. Dig, “Usage,
costs, and benefits of continuous integration in open-source projects,”
inProceedings of the 31st IEEE/ACM International Conference on
Automated Software Engineering, ser. ASE ’16. New York, NY , USA:
Association for Computing Machinery, 2016, p. 426–437. [Online].
Available: https://doi.org/10.1145/2970276.2970358
[3] Y . Zhao, A. Serebrenik, Y . Zhou, V . Filkov, and B. Vasilescu,
“The impact of continuous integration on other software development
practices: A large-scale empirical study,” in2017 32nd IEEE/ACM
International Conference on Automated Software Engineering (ASE),
2017, pp. 60–71. [Online]. Available: https://doi.org/10.1109/ASE.2017
.8115619
[4] E. Soares, G. Sizilio, J. Santos, D. A. da Costa, and U. Kulesza, “The
effects of continuous integration on software development: a systematic
literature review,”Empirical Softw. Engg., vol. 27, no. 3, May 2022.
[Online]. Available: https://doi.org/10.1007/s10664-021-10114-1
[5] I. Inc, “Travis ci,” 2025, Accessed June 30, 2026. [Online]. Available:
https://www.travis-ci.com/
[6] C. D. Foundation, “Jenkins,” 2025, Accessed June 30, 2026. [Online].
Available: https://www.jenkins.io/
[7] G. Inc, “Get started with gitlab ci/cd,” 2025, Accessed June 30, 2026.
[Online]. Available: https://docs.gitlab.com/ci/
[8] M. Golzadeh, A. Decan, and T. Mens, “On the rise and fall of ci
services in github,” in2022 IEEE International Conference on Software
Analysis, Evolution and Reengineering (SANER), 2022, pp. 662–672.
[Online]. Available: https://doi.org/10.1109/SANER53432.2022.00084
[9] A. Decan, T. Mens, P. R. Mazrae, and M. Golzadeh, “On
the use of github actions in software development repositories,”
in2022 IEEE International Conference on Software Maintenance
and Evolution (ICSME), 2022, pp. 235–245. [Online]. Available:
https://doi.org/10.1109/ICSME55016.2022.00029
[10] G. Cardoen, T. Mens, and A. Decan, “A dataset of github actions
workflow histories,” inProceedings of the 21st International Conference
on Mining Software Repositories, ser. MSR ’24. New York, NY , USA:
Association for Computing Machinery, 2024, p. 677–681. [Online].
Available: https://doi.org/10.1145/3643991.3644867
[11] GitHub, “Workflows,” 2025. [Online]. Available: https://docs.github.co
m/en/actions/concepts/workflows-and-actions/workflows
[12] S. G. Saroar and M. Nayebi, “Developers’ perception of github actions:
A survey analysis,” inProceedings of the 27th International Conference
on Evaluation and Assessment in Software Engineering, ser. EASE ’23.
New York, NY , USA: Association for Computing Machinery, 2023, p.
121–130. [Online]. Available: https://doi.org/10.1145/3593434.3593475
[13] Y . Zhang, Y . Wu, T. Chen, T. Wang, H. Liu, and H. Wang, “How do
developers talk about github actions? evidence from online software
development community,” inProceedings of the IEEE/ACM 46th
International Conference on Software Engineering, ser. ICSE ’24.
New York, NY , USA: Association for Computing Machinery, 2024.
[Online]. Available: https://doi.org/10.1145/3597503.3623327
[14] P. Valenzuela-Toledo and A. Bergel, “Evolution of github action
workflows,” in2022 IEEE International Conference on Software
Analysis, Evolution and Reengineering (SANER), 2022, pp. 123–127.
[Online]. Available: https://doi.org/10.1109/SANER53432.2022.00026
[15] A. Mastropaolo, F. Zampetti, G. Bavota, and M. Di Penta, “Toward
automatically completing github workflows,” inProceedings of the
IEEE/ACM 46th International Conference on Software Engineering, ser.
ICSE ’24. New York, NY , USA: Association for Computing Machinery,
2024. [Online]. Available: https://doi.org/10.1145/3597503.3623351
[16] X. Zhang, S. Muralee, S. Cherupattamoolayil, and A. Machiry, “On
the effectiveness of large language models for github workflows,”
inProceedings of the 19th International Conference on Availability,
Reliability and Security, ser. ARES ’24. New York, NY , USA:
Association for Computing Machinery, 2024. [Online]. Available:
https://doi.org/10.1145/3664476.3664497
[17] W. Fan, Y . Ding, L. Ning, S. Wang, H. Li, D. Yin, T.-S. Chua, and Q. Li,
“A survey on rag meeting llms: Towards retrieval-augmented large

language models,” inProceedings of the 30th ACM SIGKDD Conference
on Knowledge Discovery and Data Mining, ser. KDD ’24. New York,
NY , USA: Association for Computing Machinery, 2024, p. 6491–6501.
[Online]. Available: https://doi.org/10.1145/3637528.3671470
[18] M. Arslan, H. Ghanem, S. Munawar, and C. Cruz, “A survey on rag
with llms,”Procedia Computer Science, vol. 246, pp. 3781–3790, 2024,
28th International Conference on Knowledge Based and Intelligent
information and Engineering Systems (KES 2024). [Online]. Available:
https://www.sciencedirect.com/science/article/pii/S1877050924021860
[19] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “Bleu: a method for
automatic evaluation of machine translation,” inProceedings of the
40th Annual Meeting on Association for Computational Linguistics, ser.
ACL ’02. USA: Association for Computational Linguistics, 2002, p.
311–318. [Online]. Available: https://doi.org/10.3115/1073083.1073135
[20] GitHub, “Workflow syntax for github actions,” 2025, Accessed June
30, 2026. [Online]. Available: https://docs.github.com/en/actions/refer
ence/workflows-and-actions/workflow-syntax
[21] T.-H. P. Chen, S. Thomas, H. Hemmati, M. Nagappan, and A. E.
Hassan, “An empirical study on the effect of testing on code quality
using topic models: A case study on software development systems,”
IEEE Transactions on Reliability, vol. PP, pp. 1–19, 06 2017. [Online].
Available: https://doi.org/10.1109/TR.2017.2699938
[22] P. M. Duvall, S. Matyas, and A. Glover,Continuous Integration: Improv-
ing Software Quality and Reducing Risk. Addison-Wesley Professional,
2007, ISBN: 978-0-321-33638-5.
[23] P. Rostami Mazrae, T. Mens, M. Golzadeh, and A. Decan, “On the
usage, co-usage and migration of ci/cd tools: A qualitative analysis,”
Empirical Software Engineering, vol. 28, no. 2, p. 52, Mar 2023.
[Online]. Available: https://doi.org/10.1007/s10664-022-10285-5
[24] P. Kranenburg, 1991, Accessed June 30, 2026. [Online]. Available:
https://strace.io/
[25] J. Cespedes, Jul 2019, Accessed June 30, 2026. [Online]. Available:
https://ltrace.org/
[26] M. Kerrisk, “Linux manual page - ptrace(2) - errors,” Feb 2025,
Accessed June 30, 2026. [Online]. Available: https://man7.org/linux/m
an-pages/man2/ptrace.2.html#ERRORS
[27] GitHub, “Understanding github actions,” 2026, Accessed June 30,
2026. [Online]. Available: https://docs.github.com/en/actions/get-started
/understand-github-actions
[28] P. Rostami Mazrae, A. Decan, T. Mens, and M. Wessel, “An empirical
study of the evolution of github actions workflows,”Journal of
Systems and Software, vol. 236, p. 112824, 2026. [Online]. Available:
https://www.sciencedirect.com/science/article/pii/S0164121226000580
[29] Y . Dong, Z. Li, Y . Tian, C. Sun, M. W. Godfrey, and M. Nagappan,
“Bash in the wild: Language usage, code smells, and bugs,”ACM
Trans. Softw. Eng. Methodol., vol. 32, no. 1, Feb. 2023. [Online].
Available: https://doi.org/10.1145/3517193
[30] M. Kerrisk, “Linux manual page - open(2),” Feb 2025. [Online].
Available: https://man7.org/linux/man-pages/man2/open.2.html
[31] M. Rahman, S. Khatoonabadi, A. Abdellatif, H. Samaana, and E. Shihab,
“The impact of environment configurations on the stability of ai-enabled
systems,” inProceedings of the 29th International Conference on
Evaluation and Assessment in Software Engineering, ser. EASE ’25.
New York, NY , USA: Association for Computing Machinery, 2025, p.
512–523. [Online]. Available: https://doi.org/10.1145/3756681.3756953
[32] P. S. Foundation, “Os - miscellaneous operating system interfaces -
os.environ,” Aug 2025, Accessed June 30, 2026. [Online]. Available:
https://docs.python.org/3/library/os.html#os.environ
[33] M. Kerrisk, “Linux manual page - getenv(3),” Feb 2025. [Online].
Available: https://man7.org/linux/man-pages/man3/getenv.3.html
[34] P. S. Foundation, “site - site-specific configuration hook - sitecustomize,”
Aug 2025, Accessed June 30, 2026. [Online]. Available: https:
//docs.python.org/3/library/site.html#module-sitecustomize
[35] S. Peldszus, J. B ¨urger, and J. J ¨urjens, “Umlsecrt: Reactive security
monitoring of java applications with round-trip engineering,”IEEE
Transactions on Software Engineering, vol. 50, no. 1, pp. 16–47, 2024.
[Online]. Available: https://doi.org/10.1109/TSE.2023.3326366
[36] I. Bouzenia and M. Pradel, “Resource usage and optimization
opportunities in workflows of github actions,” inProceedings of the
IEEE/ACM 46th International Conference on Software Engineering, ser.
ICSE ’24. New York, NY , USA: Association for Computing Machinery,
2024. [Online]. Available: https://doi.org/10.1145/3597503.3623303[37] J. Blank and K. Deb, “Pymoo: Multi-objective optimization in python,”
IEEE Access, vol. 8, pp. 89 497–89 509, 2020. [Online]. Available:
https://doi.org/10.1109/ACCESS.2020.2990567
[38] A. Velji ´c, D. Viduka, L. Ili ´c, A. ˇSijan, and D. Karaba ˇsevi´c,
“Trends and networks in the application of mcdm methods in
computer science: Analysis of the web of science database,”
International Journal of Electrical and Computer Engineering Systems,
vol. 17, no. 3, pp. 241–255, Feb. 2026. [Online]. Available:
https://doi.org/10.32985/ijeces.17.3.6
[39] GitHub, “actions/checkout,” 2018, Accessed June 30, 2026. [Online].
Available: https://github.com/actions/checkout
[40] GitHub, “actions/setup-python,” Accessed July 30, 2026. [Online].
Available: https://github.com/actions/setup-python
[41] Astral, “astral-sh/setup-uv,” 2024, Accessed June 30, 2026. [Online].
Available: https://github.com/astral-sh/setup-uv
[42] SchemaStore, “Json schema for github workflows,” 2026, Accessed
June 30, 2026. [Online]. Available: https://www.schemastore.org/github
-workflow.json
[43] GitHub, “Rest api endpoints for repositories - list repository
languages,” 2026, Accessed June 30, 2026. [Online]. Available: https:
//docs.github.com/en/rest/repos/repos?apiVersion=2022-11-28&versionI
d=free-pro-team%40latest&productId=rest#list-repository-languages
[44] GitHub, “Contexts reference,” accessed: Jun. 30, 2026. [Online].
Available: https://docs.github.com/en/actions/reference/workflows-and
-actions/contexts
[45] P. Xu, W. Ping, X. Wu, L. McAfee, C. Zhu, Z. Liu, S. Subramanian,
E. Bakhturina, M. Shoeybi, and B. Catanzaro, “Retrieval meets
long context large language models,” inThe Twelfth International
Conference on Learning Representations, 2024. [Online]. Available:
https://openreview.net/forum?id=xw5nxFWMlo
[46] Y . Du, M. Tian, S. Ronanki, S. Rongali, S. B. Bodapati, A. Galstyan,
A. Wells, R. Schwartz, E. A. Huerta, and H. Peng, “Context
length alone hurts LLM performance despite perfect retrieval,”
inFindings of the Association for Computational Linguistics:
EMNLP 2025. Suzhou, China: Association for Computational
Linguistics, Nov. 2025, pp. 23 281–23 298. [Online]. Available:
https://aclanthology.org/2025.findings-emnlp.1264/
[47] Rhysd, “Rhysd/actionlint: Static checker for github actions workflow
files,” Jun 2021, Accessed June 30, 2026. [Online]. Available:
https://github.com/rhysd/actionlint
[48] T. A. Ghaleb and D. Rathnayake, “Can llms write ci? a study on
automatic generation of github actions configurations,” inProceedings
of the 41st IEEE International Conference on Software Maintenance
and Evolution (ICSME). IEEE, 2025. [Online]. Available: https:
//doi.org/10.1109/ICSME64153.2025.00077
[49] I. Bouzenia and M. Pradel, “You name it, i run it: An llm agent to
execute tests of arbitrary projects,”Proc. ACM Softw. Eng., vol. 2, no.
ISSTA, Jun. 2025. [Online]. Available: https://doi.org/10.1145/3728922
[50] Z. Yu, Y . Zhang, M. Wen, Y . Nie, W. Zhang, and M. Yang,
“Cxxcrafter: An llm-based agent for automated c/c++ open source
software building,”Proc. ACM Softw. Eng., vol. 2, no. FSE, Jun. 2025.
[Online]. Available: https://doi.org/10.1145/3729386
[51] Y . Wang, W. Wang, S. Joty, and S. C. Hoi, “CodeT5: Identifier-aware
unified pre-trained encoder-decoder models for code understanding
and generation,” inProceedings of the 2021 Conference on
Empirical Methods in Natural Language Processing. Online and
Punta Cana, Dominican Republic: Association for Computational
Linguistics, Nov. 2021, pp. 8696–8708. [Online]. Available: https:
//aclanthology.org/2021.emnlp-main.685/
[52] J. Li, Y . Li, G. Li, Z. Jin, Y . Hao, and X. Hu, “Skcoder: A
sketch-based approach for automatic code generation,” inProceedings
of the 45th International Conference on Software Engineering, ser.
ICSE ’23. IEEE Press, 2023, p. 2124–2135. [Online]. Available:
https://doi.org/10.1109/ICSE48619.2023.00179
[53] Y . Liu, C. Tantithamthavorn, Y . Liu, and L. Li, “On the reliability and
explainability of language models for program generation,”ACM Trans.
Softw. Eng. Methodol., vol. 33, no. 5, Jun. 2024. [Online]. Available:
https://doi.org/10.1145/3641540
[54] W. Xiong, Y . Guo, and H. Chen, “The program testing ability
of large language models for code,” inProceedings of the
2024 Conference on Empirical Methods in Natural Language
Processing: Industry Track. Miami, Florida, US: Association for
Computational Linguistics, Nov. 2024, pp. 23–34. [Online]. Available:
https://aclanthology.org/2024.emnlp-industry.3/

[55] Y . Xu, Y . Chen, X. Zhang, X. Lin, P. Hu, Y . Ma, S. Lu, W. Du,
Z. M. Mao, D. Cai, and E. Zhai, “Cloudeval-yaml: A practical
benchmark for cloud configuration generation,” inProceedings of
Machine Learning and Systems, vol. 6, 2024, pp. 173–195. [Online].
Available: https://proceedings.mlsys.org/paper files/paper/2024/file/55
4e056fe2b6d9fd27ffcd3367ae1267-Paper-Conference.pdf
[56] M. Santolucito, J. Zhang, E. Zhai, and R. Piskac, “Statically verifying
continuous integration configurations,” 2018, Accessed June 30, 2026.
[Online]. Available: https://doi.org/10.48550/arXiv.1805.04473
[57] GitHub, “Using workflow run logs,” 2026, Accessed June 30, 2026.
[Online]. Available: https://docs.github.com/en/actions/how-tos/monitor
-workflows/use-workflow-run-logs
[58] OpenAI, “Gpt-4o mini,” Jul 2024, Accessed June 30, 2026. [Online].
Available: https://developers.openai.com/api/docs/models/gpt-4o-mini
[59] OpenAI, “text-embedding-ada-002,” https://developers.openai.com/api/
docs/models/text-embedding-ada-002, 2022, accessed: Jun. 30, 2026.
[60] “deadsnakes” team, “New python versions,” 2026. [Online]. Available:
https://launchpad.net/ ∼deadsnakes/+archive/ubuntu/ppa
[61] Astral, “Astral-sh/uv: An extremely fast python package and project
manager, written in rust.” 2024, Accessed June 30, 2026. [Online].
Available: https://github.com/astral-sh/uv
[62] mkdocs, “mkdocs/mkdocs,” 2026, Accessed June 30, 2026. [Online].
Available: https://github.com/mkdocs/mkdocs
[63] G. Benedetti, L. Verderame, and A. Merlo, “Automatic security
assessment of github actions workflows,” inProceedings of the
2022 ACM Workshop on Software Supply Chain Offensive Research
and Ecosystem Defenses, ser. SCORED’22. New York, NY , USA:
Association for Computing Machinery, 2022, p. 37–45. [Online].
Available: https://doi.org/10.1145/3560835.3564554
[64] J. Huang and B. Lin, “Cigar: Contrastive learning for github action
recommendation,” in2023 IEEE 23rd International Working Conference
on Source Code Analysis and Manipulation (SCAM), 2023, pp. 61–71.
[Online]. Available: https://doi.org/10.1109/SCAM59687.2023.00017
[65] M. Sarschar, G. Zhang, and A. Nowak, “Pacgbi: A pipeline forautomated code generation from backlog items,” inProceedings of
the 39th IEEE/ACM International Conference on Automated Software
Engineering, ser. ASE ’24. New York, NY , USA: Association
for Computing Machinery, 2024, p. 2338–2341. [Online]. Available:
https://doi.org/10.1145/3691620.3695346
[66] Y . Peng, A. D. Gotmare, M. R. Lyu, C. Xiong, S. Savarese, and
D. Sahoo, “Perfcodegen: Improving performance of llm generated code
with execution feedback,” in2025 IEEE/ACM Second International
Conference on AI Foundation Models and Software Engineering
(Forge), 2025, pp. 1–13. [Online]. Available: https://doi.org/10.1109/Fo
rge66646.2025.00008
[67] T. Wu, R. Chen, W. Du, S. Ma, G. Qi, Z. Xing, S. Khadivi,
R. Periyathambi, and G. Haffari, “Environment-aware code generation:
How far are we?” inProceedings of the IEEE/ACM 48th International
Conference on Software Engineering, ser. ICSE ’26, New York, NY ,
USA, 2026. [Online]. Available: https://doi.org/10.48550/arXiv.2601.12
262
[68] J. Chen, Z. Pan, X. Hu, Z. Li, G. Li, and X. Xia, “Reasoning runtime
behavior of a program with llm: How far are we?” inProceedings of
the IEEE/ACM 47th International Conference on Software Engineering,
ser. ICSE ’25. IEEE Press, 2025, p. 1869–1881. [Online]. Available:
https://doi.org/10.1109/ICSE55347.2025.00012
[69] X. Lian, Y . Chen, R. Cheng, J. Huang, P. Thakkar, M. Zhang,
and T. Xu, “Large language models as configuration validators,”
inProceedings of the IEEE/ACM 47th International Conference on
Software Engineering, ser. ICSE ’25. IEEE Press, 2025, p. 1704–1716.
[Online]. Available: https://doi.org/10.1109/ICSE55347.2025.00017
[70] W. Xu, J. Luo, T. Huang, K. Sui, J. Geng, Q. Ma, I. Akasaka,
X. Shi, J. Tang, and P. Cai, “Logsage: An llm-based framework
for ci/cd failure detection and remediation with industrial validation,”
in2025 40th IEEE/ACM International Conference on Automated
Software Engineering (ASE), 2025, pp. 3742–3753. [Online]. Available:
https://doi.org/10.1109/ASE63991.2025.00310
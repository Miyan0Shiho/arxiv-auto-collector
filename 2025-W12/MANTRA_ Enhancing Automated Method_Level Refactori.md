# MANTRA: Enhancing Automated Method-Level Refactoring with Contextual RAG and Multi-Agent LLM Collaboration

**Authors**: Yisen Xu, Feng Lin, Jinqiu Yang, Tse-Hsun, Chen, Nikolaos Tsantalis

**Published**: 2025-03-18 15:16:51

**PDF URL**: [http://arxiv.org/pdf/2503.14340v1](http://arxiv.org/pdf/2503.14340v1)

## Abstract
Maintaining and scaling software systems relies heavily on effective code
refactoring, yet this process remains labor-intensive, requiring developers to
carefully analyze existing codebases and prevent the introduction of new
defects. Although recent advancements have leveraged Large Language Models
(LLMs) to automate refactoring tasks, current solutions are constrained in
scope and lack mechanisms to guarantee code compilability and successful test
execution. In this work, we introduce MANTRA, a comprehensive LLM agent-based
framework that automates method-level refactoring. MANTRA integrates
Context-Aware Retrieval-Augmented Generation, coordinated Multi-Agent
Collaboration, and Verbal Reinforcement Learning to emulate human
decision-making during refactoring while preserving code correctness and
readability. Our empirical study, conducted on 703 instances of "pure
refactorings" (i.e., code changes exclusively involving structural
improvements), drawn from 10 representative Java projects, covers the six most
prevalent refactoring operations. Experimental results demonstrate that MANTRA
substantially surpasses a baseline LLM model (RawGPT ), achieving an 82.8%
success rate (582/703) in producing code that compiles and passes all tests,
compared to just 8.7% (61/703) with RawGPT. Moreover, in comparison to
IntelliJ's LLM-powered refactoring tool (EM-Assist), MANTRA exhibits a 50%
improvement in generating Extract Method transformations. A usability study
involving 37 professional developers further shows that refactorings performed
by MANTRA are perceived to be as readable and reusable as human-written code,
and in certain cases, even more favorable. These results highlight the
practical advantages of MANTRA and emphasize the growing potential of LLM-based
systems in advancing the automation of software refactoring tasks.

## Full Text


<!-- PDF content starts -->

MANTRA: Enhancing Automated Method-Level Refactoring with
Contextual RAG and Multi-Agent LLM Collaboration
Yisen Xu
SPEAR Lab, Concordia University
Montreal, Canada
yisen.xu@mail.concordia.caFeng Lin
SPEAR Lab, Concordia University
Montreal, Canada
feng.lin@mail.concordia.caJinqiu Yang
O-RISA Lab, Concordia University
Montreal, Canada
jinqiu.yang@concordia.ca
Tse-Hsun (Peter) Chen
SPEAR Lab, Concordia University
Montreal, Canada
peterc@encs.concordia.caNikolaos Tsantalis
Department of Computer Science and
Software Engineering, Concordia
University
Montreal, Canada
nikolaos.tsantalis@concordia.ca
ABSTRACT
Maintaining and scaling software systems relies heavily on effective
code refactoring, yet this process remains labor-intensive, requir-
ing developers to carefully analyze existing codebases and prevent
the introduction of new defects. Although recent advancements
have leveraged Large Language Models (LLMs) to automate refac-
toring tasks, current solutions are constrained in scope and lack
mechanisms to guarantee code compilability and successful test
execution. In this work, we introduce MANTRA , a comprehensive
LLM agent-based framework that automates method-level refac-
toring. MANTRA integrates Context-Aware Retrieval-Augmented
Generation, coordinated Multi-Agent Collaboration, and Verbal
Reinforcement Learning to emulate human decision-making dur-
ing refactoring while preserving code correctness and readability.
Our empirical study, conducted on 703 instances of “pure refactor-
ings” (i.e., code changes exclusively involving structural improve-
ments), drawn from 10 representative Java projects, covers the
six most prevalent refactoring operations. Experimental results
demonstrate that MANTRA substantially surpasses a baseline LLM
model ( RawGPT ), achieving an 82.8% success rate (582/703) in pro-
ducing code that compiles and passes all tests, compared to just
8.7% (61/703) with RawGPT . Moreover, in comparison to IntelliJ’s
LLM-powered refactoring tool ( EM-Assist ),MANTRA exhibits a
50% improvement in generating Extract Method transformations. A
usability study involving 37 professional developers further shows
that refactorings performed by MANTRA are perceived to be as read-
able and reusable as human-written code, and in certain cases, even
more favorable. These results highlight the practical advantages
ofMANTRA and emphasize the growing potential of LLM-based
systems in advancing the automation of software refactoring tasks.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than ACM
must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,
to post on servers or to redistribute to lists, requires prior specific permission and/or a
fee. Request permissions from permissions@acm.org.
Conference’17, July 2017, Washington, DC, USA
©2025 Association for Computing Machinery.
ACM ISBN 978-x-xxxx-xxxx-x/YY/MM. . . $15.00
https://doi.org/10.1145/nnnnnnn.nnnnnnnACM Reference Format:
Yisen Xu, Feng Lin, Jinqiu Yang, Tse-Hsun (Peter) Chen, and Nikolaos
Tsantalis. 2025. MANTRA: Enhancing Automated Method-Level Refactoring
with Contextual RAG and Multi-Agent LLM Collaboration. In Proceedings
of ACM Conference (Conference’17). ACM, New York, NY, USA, 12 pages.
https://doi.org/10.1145/nnnnnnn.nnnnnnn
1 INTRODUCTION
Refactoring is the process of improving the overall design and
structure of the code without changing its overall behavior [ 16]. The
goal of refactoring is to improve maintainability and facilitate future
functionality extension [ 34], making it essential for adapting to
ever-evolving software requirements. However, despite its benefits,
many developers hesitate to refactor due to the time and effort
involved [ 45]. Developers often need to first analyze the possibility
of refactoring, then modify the code to refactor, and finally, ensure
that refactoring does not introduce new issues.
To assist developers with refactoring, researchers and Integrated
Development Environment (IDE) developers (e.g., Eclipse and In-
tellij IDEA) have proposed automated refactoring techniques. For
example, Tsantalis and Chatzigeorgiou [57,58]proposed JDeodor-
ant to detect code smells, such as Feature Envy andLong Method ,
and apply refactoring operations. WitchDoctor [ 15] makes refactor-
ing recommendations by monitoring whether code changes trigger
predefined specifications. One common characteristic of these tools
is that they are based on pre-defined rules or metrics. Although
useful, they lack a deep understanding of the project’s domain-
specific structure and cannot produce refactorings similar to those
written by developers, resulting in low acceptance in actual devel-
opment [38, 53].
Recent research on Large Language Models (LLMs) has shown
their great potential and capability in handling complex program-
ming tasks [ 30,62,66,68], making them a possible solution for
overcoming prior challenges, (i.e., generating high-quality refac-
tored code similar to human-written ones). Several studies [ 5,13,
34,41,51] have already explored the use of LLMs for refactoring,
demonstrating their strong ability to analyze refactoring opportu-
nities and applying code changes. However, existing techniques
primarily rely on simple prompt-based refactoring generation, fo-
cus on a limited set of refactoring types, and lack proper verificationarXiv:2503.14340v1  [cs.SE]  18 Mar 2025

Conference’17, July 2017, Washington, DC, USA Yisen Xu, Feng Lin, Jinqiu Yang, Tse-Hsun (Peter) Chen, and Nikolaos Tsantalis
through compilation checks and test execution. Moreover, these
approaches have not fully utilized the self-reflection [ 50] and self-
improvement capabilities of large language models, resulting in
limited effectiveness and performance that has yet to match human-
level proficiency in code refactoring.
In this paper, we propose MANTRA (Multi-AgeNTCode RefAactoring) ,
an end-to-end LLM agent-based solution for automated method-
level refactoring. Given a method to refactor and a specified refac-
toring type, MANTRA generates fully compilable, readable, and
test-passing refactored code. MANTRA includes three key compo-
nents: (1) Context-Aware Retrieval-Augmented Refactored Code
Generation , which constructs a searchable database to provide few-
shot examples for improving refactoring quality; (2) Multi-Agent
Refactored Code Generation , which employs a Developer Agent
and Reviewer Agent to simulate real-world refactoring processes
and produce high-quality refactoring; and (3) Self-Repair Using
Verbal Reinforcement Learning , which iteratively identifies and
corrects issues that cause compilation or test failures using a verbal
reinforcement learning framework [50].
We evaluated MANTRA using 10 Java projects used in prior refac-
toring studies [ 21,22,28]. These projects cover diverse domains,
have rich commit histories, and contain many tests. Since most
refactoring changes are accompanied by unrelated code changes
(e.g., bug fixes or feature additions) [ 53], we collected “pure refac-
toring changes” (i.e., no code changes other than refactoring) to
eliminate noise when evaluating MANTRA ’s refactoring ability. We
applied PurityChecker [36] to filter commits, ultimately obtaining
703 pure refactorings for our experiments, which cover six of the
most common refactoring activities [ 20,40,53]:Extract Method ,
Move Method ,Inline Method , along with related compound refactor-
ing activities: Extract and Move Method ,Move and Inline Method , and
Move and Rename Method . Using these refactorings, we compare
MANTRA with our LLM baseline ( RawGPT ), IntelliJ’s LLM-based
refactoring tool ( EM-Assist [42]), and human-written refactored
code. Furthermore, we conducted a user study to receive developer
feedback on MANTRA -generated code and an ablation study to
evaluate the contribution of each component within MANTRA .
Our results demonstrate that MANTRA outperforms RawGPT
in both functional correctness and human-likeness (how similar it
is compared to developer refactoring). MANTRA achieves a sig-
nificantly higher success rate of 82.8% (582/703) in generating
compilable and test-passing refactored code, compared to only
8.7% for RawGPT . Against EM-Assist ,MANTRA shows a 50% im-
provement in generating Extract Method refactorings. Compared
to human-written code, our user study (with 37 developers) found
that MANTRA and human refactorings share similar readability and
reusability scores. However, MANTRA performs better in Extract &
Move andMove & Rename refactorings due to its clear comments
and better code naming. In contrast, humans do better in Inline
Method refactoring by making additional improvements. Finally,
our ablation study highlights that removing any component from
MANTRA results in a noticeable performance drop (40.7% - 61.9%),
with the Reviewer Agent contributing the most to overall effective-
ness.
We summarize the main contributions as follows:•We proposed an end-to-end agent-based refactoring solu-
tion MANTRA , which considers compilation success and
functional correctness in the refactoring process. MANTRA
leverages Context-Aware Retrieval-Augmented Generation
to learn developer refactoring patterns, integrates multiple
LLM agents to simulate the developer’s refactoring process,
and adopts a verbal reinforcement learning framework to
improve the correctness of the refactored code.
•We conducted an extensive evaluation, and MANTRA suc-
cessfully generated 582/703 compilable and test-passing refac-
torings, significantly outperforming RawGPT , which only
produced 61 successful refactorings. Compared with EM-
Assist , a state-of-the-art LLM-based technique primarily fo-
cused on Extract Method refactoring, MANTRA achieved a
50% improvement.
•We conducted a user study to compare MANTRA ’s gener-
ated and human-written refactored code. The analysis of 37
responses shows that the refactored code generated by the
MANTRA is similar to the developer-written code in terms
of readability and reusability. Moreover, MANTRA ’s gener-
ated code has better advantages in method naming and code
commenting.
•We made the data and code publicly available [6].
Paper Organization . Section 2 discusses related work. Section
3 details the design and implementation of MANTRA . Section 4
presents the evaluation results of MANTRA . Section 5 discusses
the limitations and potential threats to validity. Finally, Section 6
summarizes our findings and outlines directions for future work.
2 RELATED WORK
Traditional Refactoring Approaches . Refactoring plays a criti-
cal role in software development and greatly influences software
quality. Traditional research in refactoring generally focuses on
two main aspects: identifying refactoring opportunities andrecom-
mending refactoring solutions . In terms of opportunity identification,
existing approaches have explored various methods, such as calcu-
lating distances between entities and classes for Move Method refac-
toring [ 57], assessing structural and semantic cohesion for Extract
Class opportunities [ 10], and utilizing logic meta-programming
techniques to uncover refactoring possibilities [ 56]. On the other
hand, solution recommendation often involves automated tech-
niques. [ 37] introduced a tool CODe-Imp, which uses search-based
techniques to perform refactoring. WitchDoctor [ 15] and BeneFac-
tor [19] monitor code changes and automatically suggest refactor-
ing operations. Nevertheless, these approaches are often limited by
their rule-based nature, which may restrict them to certain types
of refactoring or cause them to encounter unhandled issues during
the refactoring process.
LLM-Based Techniques for Generating Refactored Code . Re-
cent research on LLMs has demonstrated their remarkable ability
to handle complex tasks, making them promising solutions to over-
come the limitations of traditional refactoring approaches. Existing
studies utilize LLMs for various refactoring tasks: directly prompt-
ing GPT-4 for refactoring tasks [ 13,41], providing accurate iden-
tification of refactoring opportunities through carefully designed
prompts [ 31], and recommending specific refactoring types such

MANTRA: Enhancing Automated Method-Level Refactoring with Contextual RAG and Multi-Agent LLM Collaboration Conference’17, July 2017, Washington, DC, USA
asExtract Method [52]. Other studies further improve LLM-based
refactoring by emphasizing prompt clarity [ 5], using carefully se-
lected few-shot examples [ 51], and applying structured prompting
techniques [ 65]. Additionally, hybrid approaches combining rule-
based systems and LLMs have achieved superior outcomes com-
pared to single-method techniques [ 69]. Automated frameworks
and tools, such as the Context-Enhanced Framework for Automatic
Test Refactoring [ 17] and tools like EM-Assist [43], further illustrate
practical implementations of these techniques, highlighting the
value of integrating LLMs into comprehensive, feedback-driven
refactoring workflows. EM-Assist even outperforms all prior tools
onMove Method refactoring [43].
While these techniques leverage LLMs for automated refactor-
ing, they typically focus on only one or two types of refactoring,
neglecting compound or repository-level transformations (e.g., Ex-
tract and Move Method ) and failing to ensure that the refactored
code compiles and passes all tests. In contrast, MANTRA uses LLM
agents to emulate developers’ refactoring process and integrate tra-
ditional tools to provide feedback. MANTRA generates refactored
code for a broader range of refactoring activities and ensures the
generated code can compile and pass all tests.
LLM-Based Approaches for Code Quality Improvement. Pre-
vious studies have also explored using LLM to improve other aspects
of software quality, such as security, performance, and design. For
instance, Lin et al. [30] proposed leveraging Software Process Mod-
els to enhance the design quality in code generation tasks. Ye et al.
[68] introduced LLM4EFFI, conducting comprehensive research on
improving code efficiency. Wadhwa et al. [62] proposed CORE, an
approach utilizing instruction-following LLMs to assist developers
in addressing code quality issues through targeted revisions. Wu
et al. [66] presented iSMELL, which integrates LLMs for the detec-
tion and subsequent refactoring of code smells, thus systematically
enhancing software quality. Inspired by these studies, we designed
MANTRA to also consider code readability by integrating with code
style checkers (i.e., CheckStyle [11]) in the generation process.
3 METHODOLOGY
In this section, we introduce MANTRA (Multi-AgeNTCode RefAactoring) ,
an LLM-based, agent-driven solution for automated code refactor-
ing.MANTRA focuses on method-level refactorings because of their
wide adoption in practice [ 29,35]. In particular, we implement a
total of six refactoring activities, composing three of the most pop-
ular refactoring activities [ 20,40,53]:Extract Method ,Move Method ,
and Inline Method ; and three of their compound refactoring ac-
tivities: Extract And Move Method ,Move And Inline Method , and
Move And Rename Method . These refactoring activities consider
both straightforward and intricate refactoring scenarios.
MANTRA takes as input the code of a method to be refactored
and the specified refactoring type. It then automatically finds refac-
toring opportunities in the method and generates fully compil-
able and highly readable refactored code that can pass all the tests.
MANTRA consists of three key components: 1) RAG: Context-Aware
Retrieval-Augmented Refactored Code Generation, 2) Refactored
Code Generation : Multi-Agent Refactored Code Generation, and 3)
Repair : Self-Repair Using Verbal Reinforcement Learning. The RAG
component constructs a searchable database that contains priorrefactorings as few-shot examples to guide MANTRA . The Refac-
tored Code Generation component uses a multi-agent framework
that harnesses LLMs’ planning and reasoning abilities to generate
refactored code. Finally, the Self-Repair component implements
a verbal reinforcement learning framework [ 50] to automatically
identifies and corrects issues in the generated refactored code.
3.1 Context-Aware Retrieval Augmented
Refactored Code Generation
Figure 1 shows an overview of MANTRA ’s RAG construction pro-
cess. RAG provides LLM with relevant examples for few-shot learn-
ing, thus improving its ability to generate accurate and contextually
relevant output [ 23,51]. RAG combines information retrieval and
text generation by integrating external knowledge through a two-
stage process – retrieval and generation [ 18]. Below, we discuss the
details of MANTRA ’s RAG design.
Constructing a Database of Pure Refactoring Code Examples.
We aim to build a database containing only pure refactoring code
changes, which involve improving code structure without altering
functionality. However, in reality, refactoring is often accompanied
by unrelated code changes such as bug fixes or feature additions
[53]. These changes contain noise that makes such refactoring code
changes unusable as few-shot examples to guide LLM for generating
general refactored code.
We build the pure refactoring database using the Refactoring Ora-
cle Dataset [61], which contains 12,526 refactorings (mostly impure
refactorings) collected from 547 commits across 188 open-source
Java projects (2011 to 2021). We selected this dataset because it
includes various projects and refactoring types and was used as a
benchmark to evaluate the accuracy of refactoring detection tools
(e.g., RefDiff [54] and RefactoringMiner [60,61]), making it highly
suitable for our purpose. We incorporated PurityChecker [36], an
extension of RefactoringMiner that specializes in assessing the pu-
rity of method-level refactorings, excluding those associated with
functional feature modifications. We choose to use Refactoring-
Miner and PurityChecker because they are well-maintained and
known for their high detection accuracy. RefactoringMiner has an
average precision and recall of 99% and 94%, respectively [ 61] and
PurityChecker has an average precision and recall of 95% and 88%,
respectively [ 36] on the Refactoring Oracle Dataset . At the end of
this phase, we extracted 905 pure refactorings along with their
associated metadata (e.g., Class Name ,Method Signature ,File Path ,
andCall Graph ) from the GitHub repositories.
Incorporating Code Description and Caller-Callee Relation-
ships for Context-Aware RAG Retrieval. Using only source
code to construct a RAG database presents several challenges. First,
code with similar structures does not necessarily share the same
functionality or logic, both of which influence refactoring strategies.
For instance, in Move Method refactoring, a test method should only
be moved to a test class, not to production code. If the context does
not clearly indicate the class type, relying solely on source code
structure for retrieval may result in incorrect matches. Second, code
dependencies play a crucial role in refactoring, as refactorings like
Extract Method andMove Method are often driven by code depen-
dencies [ 59]. Without capturing these dependencies, the retrieved
examples may fail to align with the intended refactoring process.

Conference’17, July 2017, Washington, DC, USA Yisen Xu, Feng Lin, Jinqiu Yang, Tse-Hsun (Peter) Chen, and Nikolaos Tsantalis
LLMsClass InfoMethod Code before refactoring1. Extract Pure Refactoringswith structure information2. Incorporating caller-callee relationship to generate Code DescriptionCodedescriptionCodeRepositories
3. Construct a Database of Pure Refactoring ExamplesPure RefactoringDatabase for RAGPure RefactoringCaller and callee code 
Figure 1: An overview of how MANTRA constructs a database containing only pure-refactoring for RAG.
Therefore, in MANTRA , we incorporate 1) a natural language
description of the refactored code and 2) all direct callers and callees
of the refactored method as code-specific context to enhance RAG’s
retrieval capabilities. To generate the natural language description,
we follow a recent study in the NLP community [ 7]. We use LLMs to
generate a contextual description for every refactoring and concate-
nate this description with the corresponding code to construct the
contextual database. To guide the LLM in generating these descrip-
tions, we use a simple prompt: “{Code}{Caller/Callee}{Class
Info} Please give a short, succinct description to
situate this code within the context to improve search
retrieval of the code.” , where{Code} refers to the code before
refactoring, {Caller/Callee} denotes the direct callers/callees,
and{Class Info} presents the structure information for the Class
containing the code to be refactored, such as Package Name ,Class
Name ,Class Signature . The prompt includes the direct callers and
callees of method to be refactored to assist in description genera-
tion, as dependent methods may influence refactoring decisions.
Specifically, the prompt contains the method signatures and bodies
of all direct callers and callees, enabling the retrieval mechanism to
account for such dependencies.
Retrieving the Most Similar Code As Few-Short Examples. In
the retrieval stage of RAG, two main retrieval methods are com-
monly combined and fused for the best outcome [ 23], namely sparse
retrieval and dense trieval. Sparse retrieval uses textual similarity
to efficiently retrieve relevant documents, and dense retrieval re-
lies on semantic similarity. MANTRA leverages sparse and dense
retrieval separately for producing two similarity-ranked lists, then
combines the lists to create a unified ranking list. For sparse re-
trieval, MANTRA leverages BM25 [ 48], a robust ranking technique
to obtain a ranked list based on textual similarity. For dense re-
trieval, MANTRA employs all-MiniLM-L6-v2 , a pre-trained model
from Sentence Transformers [46] known for its speed and quality,
for embedding generation. MANTRA then computes the cosine
similarity between embeddings of the input refactoring request
and stored refactoring examples to obtain an additional ranked list
based on semantic similarity.Finally, MANTRA uses the Reciprocal
Rank Fusion (RRF) algorithm [ 12,49] to combine the sorted lists
and re-rank the results.
3.2 Multi-Agent Refactored Code Generation
MANTRA emulates how real-world code refactoring happens through
a multi-agent collaboration among the Developer Agent, the Re-
viewer Agent, and the Repair Agent. As shown in Figure 2, MANTRA
adapts a mixture-of-agents architecture [ 63] to organize the agent
communication in two layers. In the first layer, the Developer Agentis responsible for generating and improving the code, while the
Reviewer Agent reviews the code and provides feedback or sugges-
tions to the Developer Agent. If the refactored code fails to compile
or pass tests, the generated code enters the second layer of the
agent architecture, where the Repair Agent tries to repair any com-
pilation or test failures based on LLM-based reasoning and verbal
reinforcement learning [50].
3.2.1 Developer Agent. Given a method to refactor and a specified
refactoring type, the Developer Agent first autonomously extracts
the necessary information (e.g., repository structure, class informa-
tion) based on the observation (i.e., the refactoring type and the
provided inputs) by invoking our custom static analysis. It then
retrieves similar refactorings using our contextual RAG approach
(Section 3.1) as few-shot examples to enhance code generation.
Finally, it generates the refactored code using the extracted infor-
mation and the retrieved examples.
Dev-Agent-1: Using static code analysis to extract repository
and source code structures. The Developer Agent has access
to our custom static analysis tools to analyze the repository and
extract code and project structural information. The code structural
information includes the class hierarchies, inheritance relationship,
method signatures and their implementation in a class, and inter-
procedural method call graph. The project structural information
includes the project directory structure and the specific Java file
content. Among these, method signatures and their implementation
are mandatory for all refactoring types, whereas other information
is only necessary for specific types of refactoring.
To reduce static analysis overhead and unneeded information
to the LLM, the Developer Agent autonomously decides which
analyses to perform based on the given refactoring type and the
target code. For example, for the Move Method refactoring, the
Developer Agent first calls get_project_structure to retrieve the
overall project structure. Based on this information, it determines
the relevant file directories to inspect. It then calls get_file_content
to retrieve the source code files from the directory and assesses
whether the method should be moved to the target class/file. The
Developer Agent then uses the analysis results in the next phase to
guide the refactoring process and generate the refactored code.
Dev-Agent-2: Automated refactored code generation via RAG
and chain-of-thought. Given the static analysis result from the
previous step, the Developer Agent 1) retrieves similar refactorings
as few-shot examples using RAG and 2) generates the refactored
code using chain-of-thought reasoning [ 64]. The Developer Agent
provides the code to be refactored, the generated code description,
and direct callers/callees as input to the RAG database to retrieve

MANTRA: Enhancing Automated Method-Level Refactoring with Contextual RAG and Multi-Agent LLM Collaboration Conference’17, July 2017, Washington, DC, USA
RefactoringTypeTarget MethodCodeRepositoriesDeveloper AgentDev-Agent-1:Using static code analysis to extract repository and source code structures. Caller and callee code Project structureRetrieval similar refactoring examples fromRAGfor few-shotSimilarrefactoringsDev-Agent-2: Automated refactored code generation via RAG and chain-of-thought.RefactoredCodeReviewer Agent
Assemble class code RefactoringverificationCode style verificationCompilation&TestverificationFeedbackreportNoYesYesRefactoredCodeError LogRepair AgentNoYesNoGenerate feedback report
Class refactoredCodeError logGenerate repairedcodeCompilation & TestverificationNoYesClass Info
AutonomouslyextractTarget Method 
Java file content
Final RefactoredCodeMulti-Agent Refactored Code Generation
Self-Repair Using Verbal Reinforcement Learning
Figure 2: An overview of MANTRA .
similar refactorings for few-shot learning. Then, the Developer
Agent follows a structured chain-of-thought reasoning approach,
analyzing the provided information sequentially. Below, we provide
asimplified prompt example to show the code generation process.
###Task: Code Refactoring Based on a Specified Refactoring Type
###Instructions: Please follow the Step-by-Step Analysis:
Step 1: Code Analysis. Analyze the specific code segment that needs to be
refactored. And output a concise summary of the code to be refactored.
Step 2: Refactoring Method Reference. Search and retrieve up to three
similar refactoring examples from the RAG system.
Step 3: Structure Information Extraction. Based on the refactoring type
and code summary, use the provided tools to collect any structural information
you need. This may include code structure information as well as project
structure information.
Step 4: Refactoring Execution. Using the extracted structural information
and retrieved examples to generate the refactored code.
### Input: ‘{Code to be Refactored, Refactoring Type}’
### Response: ‘{Refactored Code}’
As shown in the example, the Developer Agent analyzes the
method source code and the entire class in which it resides. Then,
the agent autonomously decides whether to further analyze broader
contextual information, including direct callers/callees, the inheri-
tance graph, and the repository structure. After collecting source
code and other structural information, the agent then generates a
list of potential refactoring opportunities for the given refactoringtype, such as which parts of a method can be extracted (e.g., for Ex-
tract Method ) or possible classes to move a method to (e.g., for Move
Method ). Finally, the agent generates the refactored code based on
the retrieved few-shot examples and the list of potential refactoring
opportunities. Finally, the agent selects the most probable refac-
toring opportunity from the list and generates the refactored code
based on the retrieved few-shot examples.
3.2.2 Reviewer Agent. The Reviewer Agent is responsible for evalu-
ating the refactored code generated by the Developer Agent, ensur-
ing its correctness in two key aspects: refactoring verification and
code style consistency to ensure readability. The Reviewer Agent
first verifies whether the code has undergone the specified type of
refactoring by leveraging RefactoringMiner to detect refactoring ac-
tivities in the code. If the code fails this check, the Reviewer Agent
immediately generates a feedback report containing the refactor-
ing verification results and returns it to the Developer Agent for
correction. If the refactoring verification is successful, the Reviewer
Agent proceeds with code style consistency analysis, which is con-
ducted using the static analysis tool CheckStyle [ 11]. If any code
style issues are detected, such as non-standard variable naming or
formatting inconsistencies, the Reviewer Agent generates a feed-
back report highlighting these problems and sends it back to the
Developer Agent for correction. Once the refactored code passes
the refactoring verification and code style consistency check, the
Reviewer Agent will compile and test it. If there is no failure and
all tests pass, the generation process is done, and MANTRA returns
the final refactored code. If there is any failure, the generated code
and error log will be forwarded to the next phase.

Conference’17, July 2017, Washington, DC, USA Yisen Xu, Feng Lin, Jinqiu Yang, Tse-Hsun (Peter) Chen, and Nikolaos Tsantalis
3.2.3 Communication between the Developer and Reviewer Agents.
InMANTRA , the Developer Agent and the Reviewer Agent work
together in an iterative process, simulating human team collab-
oration. After generating refactored code, the Developer Agent
submits it for review. The Reviewer Agent then verifies both the
refactoring and code style, providing structured feedback. If any
issues are found, the Developer Agent refines and resubmits the
code, repeating the cycle until all required standards are met.
After verifying the refactoring activity and code style, the Re-
viewer Agent triggers the compilation and testing phase. The agent
integrates the generated refactored code into the project and creates
compilation commands based on the project’s build system (Maven
or Gradle). It then executes the commands to compile the project
and run the tests. If there are any compilation issues or test failures,
the issues are escalated to the Repair Agent for fixing.
3.3 Self-Repair Using Verbal Reinforcement
Learning
The Repair Agent iteratively fixes refactored code that fails to com-
pile and pass tests by leveraging verbal reinforcement learning. To
enhance its self-repairing capability, we integrate and adapt the
Reflexion framework [ 50], which systematically guides the repair
process through four distinct phases: initial analysis, self-reflection,
planning, and acting. Reflexion is designed to enable systems to
self-improve by incorporating iterative feedback.
The Repair Agent starts an (1) initial analysis , where the Repair
Agent examines the refactored code, the entire class where the
code resides, and associated error logs. Then, it generates an initial
patch based on the problematic code segments and corresponding
error descriptions. Subsequently, the agent applies the patch, re-
compiles the code, and re-runs the tests. If there are any issues, the
agent enters the (2) self-reflection phase, critically self-reviewing the
compilation and testing result. The agent generates error reasoning
by comparing the code and the associated error messages before
and after applying the patch. For instance, if the previous patch fails
to resolve a null pointer exception, the agent reflects on the absence
or inadequacy of necessary null checks by explicitly referencing
the corresponding lines in the stack trace.
Based on the self-reflection result, the Repair Agent enters the
(3) planning phase, where it generates a refined repair strategy,
specifying concrete code modifications required to resolve identi-
fied issues (e.g., adding necessary null checks, correcting variable
declarations, or revising incorrect method calls). Finally, in the (4)
acting phase, the Repair Agent applies the planned patch, followed
by compilation and test execution. This is an iterative process that
continues the code successfully compiles and passes all tests or
reaches a predefined maximum number of repair attempts (i.e., 20).
To ensure the code semantic remains unchanged, during the repair
process, we specify in the prompt that the agent should not modify
the code’s functionality and only focus on repair.
4 EVALUATION
In this section, we first present the studied dataset and evalua-
tion metrics. Then, we present the answers to the four research
questions.Table 1: The studied Java projects for evaluating MANTRA .
Project # Star # Commits # Pure Refactoring
checkstyle 8,462 14,606 91
pmd 4,988 29,117 125
commons-lang 2,776 8,404 59
hibernate-search 512 15,716 89
junit4 8,529 2,513 18
commons-io 1,020 5,455 93
javaparser 5,682 9,607 56
junit5 6,523 8,990 105
hbernate-orm 6,091 20,638 63
mockito 15,032 6,236 4
Total 59,615 121,282 703
Studied Dataset. Table 1 shows the Java projects that we use to
evaluate MANTRA . We select these 10 Java projects used in prior
change history tracking studies [ 21,22,28] based on three key
considerations. First, the projects cover diverse application domains,
providing a broad representation of software development practices.
Second, each project has a substantial commit history, with over
2,000 commits, indicating a richer development history and a higher
likelihood of identifying commits that involve refactoring activities.
Third, we chose projects where we could manually resolve the
compilation issues and successfully execute the tests to verify the
quality of the generated refactoring.
We evaluate and compare the refactored code generated by
MANTRA with that produced by human developers. To reduce
noise from unrelated changes, such as bug fixes, we analyze only
“pure refactoring changes” (i.e., no code changes other than refac-
toring) from these projects. Similar to Section 3.1, we apply Purity-
Checker [36] to select only the commits containing pure refactoring.
Since we want to evaluate the functional correctness of the gen-
erated refactoring by running the tests, we compile every pure
refactoring commit and its parent commit (to ensure there were no
compilation or test failures before refactoring), selecting only the
commits that could be successfully compiled and pass the tests.
We analyze test coverage to verify whether the tests cover the
refactored code (i.e., it is testing the refactored code’s functional be-
havior). We execute the test cases using Jacoco [26] to collect code
coverage information and filter out the commits where the refac-
toring changes have no coverage. Finally, we verify the existence
of target classes for the Move Method ,Extract and Move Method ,
andMove and Inline Method refactorings. This step was necessary
because the Move Method operation may move a method to newly
created classes, and it is difficult for MANTRA to predict the newly
created classes. After applying all the above data selection steps,
we identified 703 pure refactorings across the 10 Java projects.
Evaluation Metrics . We evaluate the refactored code along two
dimensions: functional correctness and human-likeness. For func-
tional correctness, we assess the code using 1) compilation success,
2) test pass, and 3) RefactoringMiner verification (i.e., Refactoring-
Miner detects that the refactoring activity indeed happened). Specif-
ically, we integrate the generated refactored code into the project.
We then compile it and execute the tests to verify if the builds are
successful. Because of LLMs’ hallucination issues [ 24], the gener-
ated code may pass the test cases but do not accurately perform
the intended refactoring. Hence, we verify whether the generated
code contains the target refactoring using RefactoringMiner [60].

MANTRA: Enhancing Automated Method-Level Refactoring with Contextual RAG and Multi-Agent LLM Collaboration Conference’17, July 2017, Washington, DC, USA
Even though we give a target method as input to MANTRA , it
still needs to find the specific part of the code that can be refactored.
Hence, we evaluate the human-likeness of MANTRA ’s generated
code to compare its refactoring decisions with that of developers.
We employ the CodeBLEU metric [ 47] and Abstract Syntax Tree
(AST) Diff Precision and Recall [ 3] to measure the difference. Code-
BLEU evaluates the grammatical and semantic consistency between
human-written refactored code and MANTRA -generated code. AST
Diff represents a set of mappings that capture code changes be-
tween the original and refactored code. Each mapping consists
of a pair of matched AST nodes from the diff between the origi-
nal and refactored versions. These mappings are obtained using
RefactoringMiner . To evaluate structural similarity, we compare the
mappings produced by MANTRA ’s refactored code with those of
the developer-written code. The number of MANTRA ’s mappings
that match the developer-written mappings is treated as true posi-
tives (TP). Precision is calculated as the ratio of TP to all mappings
produced by MANTRA , indicating how accurate MANTRA ’s map-
pings are. Recall is the ratio of TP to all developer-written mappings,
reflecting how much of the developer’s refactoring was successfully
captured. The values for CodeBLEU and AST Precision/Recall range
from 0 to 1, where 1 means a perfect match.
Environment . We selected OpenAI’s ChatGPT model [ 2] for our
experiment due to its popularity and ease of integration through the
OpenAI-API. Specifically, we utilized the gpt-4o-mini-2024-07-18
version, as it offers a balance of affordability and strong perfor-
mance. We implemented MANTRA using version 0.2.22 of Lang-
Graph [25] and various static analysis tools that we implemented
using a combination of APIs from RefactoringMiner , a modified
version of RefactoringMiner , and Eclipse JDT [ 14]. On average, one
complete refactored code generation (from querying the database,
code generation, and test execution to fixing compilation and test
failures) takes less than a minute on a Linux machine (Intel ®Core™
i9-9900K CPU @ 3.60GHz, 64GB Memory), costing less than $0.10.
RQ1: How effective is MANTRA in refactoring
code?
Motivation . In this RQ, we evaluate MANTRA ’s generated refac-
tored code along two dimensions: functional correctness and human-
likeness. We also analyze MANTRA ’s performance across differ-
ent refactoring types. This RQ offers insights into how effective
MANTRA is at performing refactoring tasks and the specific types
of refactoring tasks where MANTRA is most effective.
Approach . We evaluate MANTRA using the 703 pure refactoring
commits collected from the 10 studied Java projects. First, we use
git checkout on the commit before each pure refactoring commit,
allowing us to extract the original code before refactoring opera-
tions. The code repository is then fed into MANTRA to generate
the refactored code. For comparison, we include RawGPT as the
baselines (it uses the same LLM as MANTRA ).RawGPT directly
sends a simple prompt to the LLM to perform code refactoring. We
input RawGPT with basic code information, i.e., the same infor-
mation generated by the static-analysis component of MANTRA ’s
Developer Agent (e.g., class content and project structure). RawGPT
does not have the multi-agent component and is not prompted withfew-shot examples retrieved from RAG. The complete prompt can
be found online [6].
Result .MANTRA successfully generated 582/703 (82.8%) of the
refactored code that is compilable, passed all the tests, and ver-
ified by RefactoringMiner, while RawGPT could only generate
61/703 (8.7%) successfully. As shown in Table 2, MANTRA can
generate significantly more refactored code than RawGPT . Of the
703 refactorings, 636 generated by MANTRA successfully compiled
and passed the test cases, and 604 refactorings were further verified
byRefactoringMiner as true refactoring operations. In contrast, only
100 refactorings generated by RawGPT can compile and pass the
test cases, yet only 61 were verified by RefactoringMiner . Note that
there can be some generated refactored code that is verified by
RefactoringMiner but does not pass compilation/tests, or vice versa,
so the total successful refactoring is 582.
RawGPT has difficulties in generating Move Method ,Extract and
Move Method , and Move and Rename Method , where it cannot gener-
ate any refactoring. When doing these refactorings, RawGPT always
ignores the project structure information in the prompt and cannot
move the method to the correct class. In contrast, MANTRA has the
Reviewer Agent that gives feedback on the refactoring verification
to guide the Developer Agent to perform the Move operation. Nev-
ertheless, even for refactorings that do not require repository or
class structures (i.e., Extract Method andInline Method ),MANTRA
achieves a much higher success rate (317 v.s. 47 and 22 v.s. 8, re-
spectively).
MANTRA outperforms RawGPT in code similarity, producing
refactored code more similar to humans. Across all success-
ful refactorings, MANTRA achieved a CodeBLEU score of 0.640,
compared to RawGPT ’s 0.517, showcasing MANTRA ’s ability to
generate code that closely aligns with human-written refactorings.
Regarding structural accuracy, MANTRA achieved an AST Diff pre-
cision of 0.781, surpassing RawGPT ’s 0.773, while its AST Diff
recall reached 0.635, notably higher than RawGPT ’s 0.574. These
results indicate that MANTRA ’s generated code is more similar
and aligns better with the structural transformation of developer-
written refactoring.
MANTRA’s results are closer to developers’ decisions, where
18% (105/582) of MANTRA’s generated refactored code is iden-
tical to developer’s refactoring, compared to RawGPT’s 13.1%
(8/61). We further analyze the distribution of refactored code iden-
tical to the developer’s implementation across different refactoring
types. MANTRA correctly generated 84 Move Method refactorings,
whereas RawGPT failed to produce any valid refactorings for this
category. Additionally, MANTRA applied 11 Extract Method and
eight Inline Method that were the same as developers’ refactor-
ing, while RawGPT only managed four for both types. MANTRA
was also able to generate two composite refactorings ( Extract and
Move Method ) that were identical to those of developers. In short,
MANTRA ’s results match closer to developers’ refactoring decisions,
likely due to its retrieval-augmented generation (RAG) component,
which provides similar past refactorings as few-shot examples.

Conference’17, July 2017, Washington, DC, USA Yisen Xu, Feng Lin, Jinqiu Yang, Tse-Hsun (Peter) Chen, and Nikolaos Tsantalis
Table 2: Refactoring results of RawGPT andMANTRA . The table presents the number of refactorings to perform, compile-and-
test success rates, refactoring verification ( RM Verification ), and code similarity metrics with human-written refactorings ( Code
BLEU andAST Precision/Recall ). Successful Refactoring refers to the number of refactorings that compile, pass tests, and are
verified by RefactoringMiner . We compute the average for Code BLEU and AST Precision/Recall, and total for all other fields.
Approach Project# Pure Compile&Test RM Code AST AST Successful Extract Inline Move Extract And Move And Move And
Refactoring Success Verification BLEU Precision Recall Refactoring Method Method Method Move Method Rename Method Inline Method
RawGPTcheckstyle 91 14 9 0.667 0.603 0.233 6 4 2 0 0 0 0
pmd 125 35 30 0.502 0.791 0.338 19 17 2 0 0 0 0
commons-lang 59 2 13 0.640 0.67 0.363 2 2 0 0 0 0 0
hibernate-search 89 14 27 0.368 0.792 0.632 11 5 1 0 0 0 5
junit4 18 10 10 0.486 0.856 0.859 9 8 1 0 0 0 0
commons-io 93 8 22 0.773 0.804 0.95 6 5 1 0 0 0 0
javaparser 56 11 11 0.441 0.777 0.96 4 2 1 0 0 0 1
junit5 105 2 5 0 0 0 0 0 0 0 0 0 0
hibernate-orm 63 17 2 0.263 0.756 0.386 2 2 0 0 0 0 0
mockito 4 2 2 0.678 0.659 0.746 2 2 0 0 0 0 0
Total/Average 703 100 146 0.517 0.773 0.574 61 (8.7%) 47 8 0 0 0 6
MANTRAcheckstyle 91 90 86 0.624 0.514 0.501 85 31 4 9 39 1 0
pmd 125 119 108 0.676 0.725 0.766 106 49 2 28 24 3 0
commons-lang 59 56 46 0.567 0.815 0.257 46 42 1 0 3 0 0
hibernate-search 89 81 82 0.538 0.929 0.734 74 35 10 6 13 9 1
junit4 18 13 15 0.61 0.879 0.676 12 9 1 2 0 0 0
commons-io 93 87 81 0.623 0.873 0.662 80 62 2 9 7 0 0
javaparser 56 51 51 0.645 0.859 0.681 46 31 1 7 6 0 1
junit5 105 79 76 0.852 0.814 0.787 74 9 0 48 15 2 0
hbernate-orm 63 56 55 0.524 0.805 0.474 55 46 1 0 7 0 1
mockito 4 4 4 0.754 0.861 0.736 4 3 0 0 1 0 0
Total/Average 703 636 604 0.64 0.781 0.635 582 (82.8%) 317 22 109 115 15 3
Table 3: The number of generated refactored code (compilable
and pass all test cases) that is identical to that of developer’s.
Refactoring Type RawGPT MANTRA
Extract Method 4 11
Inline Method 4 8
Move Method 0 84
Extract And Move Method 0 2
Move And Rename Method 0 0
Move And Inline Method 0 0
Total 8 105
Table 4: The number of successful Extract Method refactor-
ings by EM-Assist andMANTRA -3.5-turbo.
Project # Extract Method EM-Assist MANTRA -3.5-turbo
checkstyle 34 14 34
pmd 55 24 48
commons-lang 54 18 41
hibernate-search 28 25 27
junit4 11 3 9
commons-io 68 33 49
javaparser 35 22 29
junit5 9 6 6
hibernate-orm 52 39 40
mockito 3 1 1
Total 359 185 277
MANTRA outperforms RawGPT in both functional correctness
and human-likeness, successfully generating 582/703 (82.8%)
compilable and test-passing refactorings, compared to 61/703
(8.7%) for RawGPT .MANTRA also has a higher CodeBLEU
score, AST Diff precision and recall, with 18% (105/582) of its
refactorings identical to developers’, versus 13.1% (8/61) for
RawGPT .RQ2: How does MANTRA compare to IntelliJ’s
LLM-based refactoring tool?
Motivation .We compare the code generated by MANTRA to the
code produced by IntelliJ’s LLM-based refactoring tool. IntelliJ IDEA
[27] provides a plugin, called EM-Assist [43,44], which uses LLM
for one specific type of refactoring, i.e., Extract Method .EM-Assist
utilizes in-context learning, providing all necessary instructions
within the prompt, including the task definition and relevant con-
textual information. The input for EM-Assist is the method to be
refactored, and it outputs a list of suggestions that include the start
and end lines to be extracted, along with the new method’s name.
To ensure the suggestions are valid, EM-Assist uses IntelliJ’s static
analysis abilities to filter out suggestions that would cause compila-
tion errors. Once valid suggestions are identified, EM-Assist applies
the refactorings via the IntelliJ IDEA API based on the AST. Given
its superior performance compared to tools like JDeodorant [ 32]
and GEMS [ 67], we selected EM-Assist as our comparison baseline.
Approach . We use the latest version of EM-Assist 0.7.5 [ 42] to per-
form Extract Method refactoring. This version of EM-Assist utilizes
gpt-3.5-turbo-0125 for refactoring. Since EM-Assist 0.7.5 is fully
integrated into IntelliJ IDEA and lacks an interface to change the
LLM version, we also used the same version of GPT model (i.e., gpt-
3.5-turbo-0125) to run MANTRA on all Extract Method refactorings.
Result .MANTRA (3.5-turbo) is able to refactor 77.1% (277/359)
of the Extract Method refactoring, while EM-Assist can only
refactor 51.5% (185/359), providing almost 50% improvement.
Table 4 shows the result of Extract Method Refactoring of EM-
Assist andMANTRA .EM-Assist refactors 185 out of 359 methods
successfully. In comparison, MANTRA outperforms EM-Assist by
successfully refactoring 277 methods. Among the refactorings per-
formed, 142 methods were successfully refactored by both tools.

MANTRA: Enhancing Automated Method-Level Refactoring with Contextual RAG and Multi-Agent LLM Collaboration Conference’17, July 2017, Washington, DC, USA
The finding shows that MANTRA is also complementary to EM-
Assist , as it successfully handled a significantly different subset of
refactoring cases and demonstrates the potential to combine both
techniques. We also see a decrease when changing MANTRA ’s un-
derlying LLM from GPT-4o-mini to 3.5-turbo, where the number of
successfully refactored Extract Method decreased from 317 (Table 2)
to 277 (12.6% decrease). This performance drop shows the impact
of the underlying LLM, but the overall result is still promising.
MANTRA (3.5-turbo) significantly outperforms EM-Assist in
Extract Method refactoring, achieving a 50% improvement by
successfully refactoring 277 methods compared to EM-Assist ’s
185.
RQ3: How does MUARF-generated refactored
code compare to human-written code?
Motivation . In prior RQs, we show that MANTRA can successfully
generate many refactorings. However, it is also important that the
refactored code is understandable and aligned with human coding
practices. Hence, in this RQ, we conduct a user study to compare
MANTRA ’s generated and human-written refactored code.
Approach . We randomly select 12 refactorings from the studied
projects used, composing two refactorings for each of the six refac-
toring types. For each sampled refactoring, we prepare the 1) code
before refactoring, 2) refactored code generated by MANTRA , and
3) developer’s refactored code. To increase the sample richness and
reduce the developer’s evaluation time, we divide these 12 samples
into two separate survey questionnaires [ 8,9], each questionnaire
covers samples from the six refactoring types. For each sample,
the participants compare two code snippets ( MANTRA -generated
and developer-written). We follow prior studies [ 1,4,33,55] and
ask the participants to assess the code’s readability and reusability
(on a scale from one to five, where five means highly readable or
reusable) and selecting the one they find more intuitive and well-
structured. For readability, we ask the participants how readable
is the refactored code. For reusability, we ask the participants how
easy it is to reuse or extend the refactored code. To avoid biases, we
do not specify which one is written by humans or generated by
LLM until the participants complete the survey . We then ask
for their opinion in free-form text after revealing this information.
Result . We shared the questionnaires through social media, and
in total, we collected 37 responses for the two questionnaires (20
and 17, respectively). We combine the results from the two ques-
tionnaires and present the results below. Overall, the participants
have programming experience ranging from 1 year (5.4%) to over
5 years (54.1%), and over 45% of the participants use Java as their
primary programming language.
On average, the participants find that MANTRA-generated
code has similar readability and reusability compared to human-
written code. As shown in Figure 3, LLM-generated code achieves
average readability and reusability scores of 4.15 and 4.13, respec-
tively, compared to human-written code, which scored 4.02 and
3.97. We further applied a student’s t-test, and we did not find
a statistically significant difference. This finding indicates, when
considering all refactoring types, the participants find MANTRA ’sgenerated code has similar readability and reusability compared to
human-written code.
However, for specific types of refactoring (i.e., Extract & Move
and Move & Rename), MANTRA-generated code shows roughly
a 20% improvement in readability and reusability, and it is
preferred 185% more than human-written code , with statisti-
cally significant difference (p-value <0.001). In contrast, human
developers achieve higher scores for Inline Method , an average of
8.5% higher in readability and 14.5% in reusability, and is 439% more
preferred than MANTRA -generated refactored code (statistically sig-
nificant with p-value <0.05). Although human developers achieve
slightly higher average readability and reusability scores for Move
and Inline Method , the difference is not statistically significant.
Figure 4 shows an example where the participants prefer the
MANTRA -generated code (readability and reusability score of 4.50
and 4.35, compared to 3.65 and 3.75 for human-refactored code,
respectively). Both MANTRA and a human developer performed
Extract and Move refactoring by extracting the same code snip-
pet into the same superclass. However, they differ in the method
names, comments, and parameters type. One participant says that
“the [LLM-generated code] is clearly easier to understand. From the
comments and names, I can guess the functionality of the code...
[Human-written code] is likely a very generic skeleton code. ”.
Overall, we find a trend based on the participants’ scores and
responses: MANTRA-generated code typically includes detailed
comments and refactoring that more closely aligns with its
intended purpose. For instance, MANTRA tends to use more de-
scriptive method names that improve clarity. For instance, one
participant mentioned that “ [LLM-generated code] is more struc-
tured and clear, and it seems more detailed and easier to understand. ”.
In comparison, developer-refactored code sometimes improves code
readability when doing specific refactoring types, especially during
Inline Method . For example, in one of the Inline Method code snip-
pets in the questionnaire, the developer improved the code during
refactoring, while MANTRA ’s generated refactored code simply
moved the code directly. A participant mentioned: “ [LLM-generated
code] added two lines of code instead of one, making it harder to
maintain in the future. ” The study shows that participants gener-
ally prefer MANTRA -generated code for its clarity and detailed
comments. However, in some cases, such as Inline Method , human
developers tend to write more readable and maintainable code by
making additional improvements beyond direct refactoring.
The participants find that MANTRA -generated code has similar
readability and reusability compared to human-written code.
MANTRA tends to perform better in Extract & Move andMove &
Rename due to its clear comments and better naming. For Inline
Method , human developers often write more readable code by
making additional improvements beyond direct refactoring.
RQ4: What is the contribution of each
component in MANTRA ?
Motivation . In this RQ, we conduct an ablation study to examine
the contribution of each component to the overall effectiveness
ofMANTRA . The results will highlight the importance of each

Conference’17, July 2017, Washington, DC, USA Yisen Xu, Feng Lin, Jinqiu Yang, Tse-Hsun (Peter) Chen, and Nikolaos Tsantalis
Extract MethodMove Method Inline MethodExtract & MoveMove & InlineMove & Rename12345Evaluation ScoreReadability & Reusability
criterion_model
readability-MANTRA
readability-Human
reusability-MANTRA
reusability-Human
Extract MethodMove Method Inline MethodExtract & MoveMove & InlineMove & Rename020406080100PercentagePreference Distribution Across Methods
Preference
MANTRA
Human
Same Good
Same Bad
Figure 3: Left panel: Boxplots depicting the readability and reusability scores from the questionnaire, comparing MANTRA -
generated code with human-written code. White markers indicate the mean score for each refactoring category. Right panel: A
visualization of participants’ preferences regarding which code they favor.
1:publicclassOrFileFilterextendsAbstractFileFilter{2:    /* ... */3:    @Override4:    publicStringtoString() {5:        finalStringBuilderbuffer = newStringBuilder();6:        buffer.append(super.toString());7:buffer.append("(");8:        if(fileFilters!= null) {9:            for(inti= 0; i< fileFilters.size(); i++) {10:if(i> 0) {11:                    buffer.append(",");12:                }13:    buffer.append(fileFilters.get(i));14:            }15:        }16:        buffer.append(")");17:        returnbuffer.toString();18:    }19:}publicabstractclassAbstractFileFilter{voidappend(finalList<?> list, finalStringBuilderbuffer) {for(inti= 0; i< list.size(); i++) {/* ... */}}}                                      Human-Refactored CodepublicabstractclassAbstractFileFilter{/** Appends the string representation of the file... **/protectedvoidappendFileFilters(StringBuilderbuffer,       List<AbstractFileFilter> fileFilters) {if(fileFilters!= null) {for(inti= 0; i< fileFilters.size(); i++) {/* ... */}}}}                                      MANTRA-Refactored CodeExtract a new method and Move to the superclass.
Figure 4: An example illustrating how MANTRA and human
developers implemented the Extract & Move refactoring.
6364053763595823452872220100200300400500600700
MANTRAw/o RAGw/o Repairw/o ReviewerNumber of RefactoringModelCompile&Test SuccessSuccessful refactorings
Figure 5: Contribution of each component in MANTRA .Com-
pile&Test Success shows the number of generated code that
compiles and passes all tests. Successful refacotorings means
the number of verified code that contains the specific refac-
toring.
component, inspiring future research on adapting them for related
tasks.
Approach . Our ablation study examines three key components:
RAG, the Reviewer Agent, and the Repair Agent. We define three
configurable models to evaluate the impact of key components in
MANTRA :MANTRA w/o RAG ,MANTRA w/o Reviewer , and MANTRA
w/o Repair . Each of the configure removes the corresponding key
component, which allows us to assess the contribution of each
component to MANTRA ’s overall performance.
Result .Removing a component reduces the number of success-
ful refactoring from 582 to 222–345 (40.7%–61.9% decrease),highlighting their contribution to MANTRA’s ability. Figure 5
shows the contribution of each component. Removing the RAG com-
ponent alone reduces the number of successfully compiled/tested
refactored code and the number of successful refactorings to 405
and 345 (36.3% and 40.7% decrease), respectively. The findings show
that a high-quality database for RAG has a non-negligible impact on
the generated refactored code. Similarly, removing the Repair Agent
significantly reduces the number of successfully compiled/tested
refactored code from 636 to 376 (40.9% decrease), as the Repair
Agent is responsible for fixing compilation issues and test failures.
Removing the Repair Agent also reduces the number of successful
refactoring from 582 to 287 (50.7% decrease), as the final-stage repair
process plays a crucial role in finalizing the refactoring changes.
Among the three components, removing the Reviewer Agent has
the most impact on the number of generated refactored that pass
compilation/test (decrease from 636 to 359) and the number of suc-
cessful refactoring (decrease from 582 to 222). The Reviewer Agent
leverages traditional tools to provide feedback in the refactoring
process. Our result highlights that without feedback from external
tools such as RefactoringMiner ,MANTRA encounters challenges in
generating valid refactored code. Our finding also shows a promis-
ing direction in combining traditional software engineering tools
to guide LLMs in producing better results.
Each component of MANTRA plays a crucial role in ensuring
successful refactorings, with the Reviewer Agent having the
most significant impact by leveraging external tools to validate
and refine refactored code. The findings also highlight the im-
portance of integrating traditional software engineering tools
with LLM-based approaches, as they provide essential feedback
that improves LLM’s results.
5 THREATS TO VALIDITY
Internal validity . Due to the generative nature of LLMs, their
responses may vary across different runs and model versions. In our
experiments, we set the temperature value to 0 to reduce variability
in the result. We used LLMs from OpenAI (i.e., 4o-mini and 3.5-
turbo) for our experiment. Future studies are needed to study the
impact of LLMs on generating the refactored code.
External validity . We focused on method-level refactorings due
to their popularity [ 29,35]. Although we included both straightfor-
ward and compound refactorings, the results may not generalize to

MANTRA: Enhancing Automated Method-Level Refactoring with Contextual RAG and Multi-Agent LLM Collaboration Conference’17, July 2017, Washington, DC, USA
other types of refactoring, such as class-level. Such refactorings are
less common and often involve other code changes (e.g., bug fixes)
[39], making data collection difficult. Further research is needed
to assess MANTRA ’s effectiveness in broader refactoring scenarios.
We focused on Java since it has extensive literature on refactoring-
related research. Future work should evaluate MANTRA across
multiple languages.
Construct validity . We use CodeBLEU and AST Precision/Recall to
evaluate the similarity between MANTRA -generated and developer-
written refactoring. However, although informative, these metrics
may still miss some differences in the code. Therefore, we conducted
a user study to compare MANTRA -generated and developer-written
code. While we gathered feedback from 37 developers with different
experience levels, some findings can be subjective. To avoid biases,
we do not tell the participants which code is refactored by MANTRA
or human developers until they finish the questionnaire.
6 CONCLUSION
In this paper, we introduced MANTRA , an end-to-end LLM agent-
based solution for automated method-level code refactoring. By
leveraging Context-Aware Retrieval-Augmented Generation, Multi-
Agent Collaboration, and Verbal Reinforcement Learning, MANTRA
generates human-like refactored code while ensuring correctness
and readability. Our evaluation on 703 real-world refactorings
across 10 diverse Java projects demonstrates that MANTRA signifi-
cantly outperforms LLM-based refactoring baseline by achieving
an 82.8% success rate in generating compilable and test-passing
code—far surpassing. It also has a 50% improvement over IntelliJ’s
LLM-based tool ( EM-Assist ). Furthermore, our user study with 37
developers reveals that MANTRA -refactored code is as readable and
reusable as human-written code, with better code for some specific
refactoring types. In short, our findings highlight the potential of
LLM-based refactoring tools in automating software maintenance.
REFERENCES
[1]Chaima Abid, Vahid Alizadeh, Marouane Kessentini, Thiago do Nascimento
Ferreira, and Danny Dig. 30 years of software refactoring research: A systematic
literature review. arXiv preprint arXiv:2007.02194 , 2020.
[2]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-
cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal
Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 , 2023.
[3]Pouria Alikhanifard and Nikolaos Tsantalis. A novel refactoring and semantic
aware abstract syntax tree differencing tool and a benchmark for evaluating the
accuracy of diff tools. ACM Transactions on Software Engineering and Methodology ,
sep 2024. ISSN 1049-331X. doi: 10.1145/3696002. URL https://doi.org/10.1145/
3696002. Just Accepted.
[4]Eman Abdullah AlOmar, Philip T Rodriguez, Jordan Bowman, Tianjia Wang, Ben-
jamin Adepoju, Kevin Lopez, Christian Newman, Ali Ouni, and Mohamed Wiem
Mkaouer. How do developers refactor code to improve code reusability? In
Reuse in Emerging Software Engineering Practices: 19th International Conference
on Software and Systems Reuse, ICSR 2020, Hammamet, Tunisia, December 2–4,
2020, Proceedings 19 , pages 261–276. Springer, 2020.
[5]Eman Abdullah AlOmar, Anushkrishna Venkatakrishnan, Mohamed Wiem
Mkaouer, Christian Newman, and Ali Ouni. How to refactor this code? an
exploratory study on developer-chatgpt refactoring conversations. In Proceed-
ings of the 21st International Conference on Mining Software Repositories , pages
202–206, 2024.
[6]Anonymous. Data and code of muarf, 2025. URL https://anonymous.4open.
science/r/MUARF-B49B. Accessed: March 13, 2025.
[7]Anthropic. Introducing contextual retrieval, 2024. URL https://www.anthropic.
com/news/contextual-retrieval. Accessed: Sep. 19, 2024.
[8]Paper Authors. Refactoring survey 1, 2024. URL https://prettyform.addxt.com/a/
form/vf/1FAIpQLSdJt97mC3NVAPPvvgdtlvFNqW9Xi3p6SwmexXNdhZ1WH-
_M0w. Accessed: Jun. 23, 2024.
[9]Paper Authors. Refactoring survey 2, 2024. URL https://
prettyform.addxt.com/a/form/vf/1FAIpQLSeVf1qD2oUIo6RQMWGKf_5wH2CzJpRAu0TkNGhib5IQxvtHLQ. Accessed: Jun. 23, 2024.
[10] Gabriele Bavota, Andrea De Lucia, and Rocco Oliveto. Identifying extract class
refactoring opportunities using structural and semantic cohesion measures. J.
Syst. Softw. , 84(3):397–414, 2011. doi: 10.1016/J.JSS.2010.11.918. URL https://doi.
org/10.1016/j.jss.2010.11.918.
[11] Checkstyle Team. Checkstyle, 2024. URL https://checkstyle.org/index.html.
Accessed: 2024-11-20.
[12] Gordon V. Cormack, Charles L. A. Clarke, and Stefan Büttcher. Reciprocal rank
fusion outperforms condorcet and individual rank learning methods. In James
Allan, Javed A. Aslam, Mark Sanderson, ChengXiang Zhai, and Justin Zobel,
editors, Proceedings of the 32nd Annual International ACM SIGIR Conference on
Research and Development in Information Retrieval, SIGIR 2009, Boston, MA, USA,
July 19-23, 2009 , pages 758–759. ACM, 2009. doi: 10.1145/1571941.1572114. URL
https://doi.org/10.1145/1571941.1572114.
[13] Kayla DePalma, Izabel Miminoshvili, Chiara Henselder, Kate Moss, and Eman Ab-
dullah AlOmar. Exploring chatgpt’s code refactoring capabilities: An empirical
study. Expert Systems with Applications , 249:123602, 2024.
[14] Eclipse Foundation. Eclipse jdt (java development tools), 2024. URL https:
//github.com/eclipse-jdt/. Accessed: March 13, 2025.
[15] Stephen R Foster, William G Griswold, and Sorin Lerner. Witchdoctor: Ide
support for real-time auto-completion of refactorings. In 2012 34th international
conference on software engineering (ICSE) , pages 222–232. IEEE, 2012.
[16] Martin Fowler. Refactoring - Improving the Design of Existing Code . Addison
Wesley object technology series. Addison-Wesley, 1999. ISBN 978-0-201-48567-7.
URL http://martinfowler.com/books/refactoring.html.
[17] Yi Gao, Xing Hu, Xiaohu Yang, and Xin Xia. Context-enhanced llm-based frame-
work for automatic test refactoring. arXiv preprint arXiv:2409.16739 , 2024.
[18] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai,
Jiawei Sun, Qianyu Guo, Meng Wang, and Haofen Wang. Retrieval-augmented
generation for large language models: A survey. CoRR , abs/2312.10997, 2023. doi:
10.48550/ARXIV.2312.10997. URL https://doi.org/10.48550/arXiv.2312.10997.
[19] Xi Ge, Quinton L DuBose, and Emerson Murphy-Hill. Reconciling manual
and automatic refactoring. In 2012 34th International Conference on Software
Engineering (ICSE) , pages 211–221. IEEE, 2012.
[20] Yaroslav Golubev, Zarina Kurbatova, Eman Abdullah AlOmar, Timofey Bryksin,
and Mohamed Wiem Mkaouer. One thousand and one stories: A large-scale
survey of software refactoring, 2021. URL https://arxiv.org/abs/2107.07357.
[21] Felix Grund, Shaiful Alam Chowdhury, Nick C. Bradley, Braxton Hall, and Reid
Holmes. Codeshovel: Constructing method-level source code histories. In 43rd
IEEE/ACM International Conference on Software Engineering, ICSE 2021, Madrid,
Spain, 22-30 May 2021 , pages 1510–1522. IEEE, 2021. doi: 10.1109/ICSE43902.2021.
00135. URL https://doi.org/10.1109/ICSE43902.2021.00135.
[22] Mohammed Tayeeb Hasan, Nikolaos Tsantalis, and Pouria Alikhanifard.
Refactoring-aware block tracking in commit history. IEEE Transactions on Soft-
ware Engineering , 50(12):3330–3350, 2024. doi: 10.1109/TSE.2024.3484586.
[23] Pengfei He, Shaowei Wang, Shaiful Chowdhury, and Tse-Hsun Chen. Explor-
ing demonstration retrievers in RAG for coding tasks: Yeas and nays! CoRR ,
abs/2410.09662, 2024. doi: 10.48550/ARXIV.2410.09662. URL https://doi.org/10.
48550/arXiv.2410.09662.
[24] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian
Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
Liu. A survey on hallucination in large language models: Principles, taxonomy,
challenges, and open questions. CoRR , abs/2311.05232, 2023. doi: 10.48550/ARXIV.
2311.05232. URL https://doi.org/10.48550/arXiv.2311.05232.
[25] LangChain Inc. Langgraph, 2024. URL https://langchain-ai.github.io/langgraph/.
Accessed: 2024-12-02.
[26] Jacoco. Jacoco, 2009. URL https://www.jacoco.org/jacoco/trunk/index.html.
Accessed: Jun. 1, 2009.
[27] JetBrains. Intellij idea, 2024. URL https://www.jetbrains.com/idea/. Accessed:
Jun. 10, 2024.
[28] Mehran Jodavi and Nikolaos Tsantalis. Accurate method and variable tracking
in commit history. In Proceedings of the 30th ACM Joint European Software Engi-
neering Conference and Symposium on the Foundations of Software Engineering ,
ESEC/FSE 2022, page 183–195, New York, NY, USA, 2022. Association for Com-
puting Machinery. ISBN 9781450394130. doi: 10.1145/3540250.3549079. URL
https://doi.org/10.1145/3540250.3549079.
[29] Miryung Kim, Thomas Zimmermann, and Nachiappan Nagappan. An empirical
study of refactoringchallenges and benefits at microsoft. IEEE Transactions on
Software Engineering , 40(7):633–649, 2014.
[30] Feng Lin, Dong Jae Kim, Tse-Husn, and Chen. Soen-101: Code generation by
emulating software process models using large language model agents, 2024.
URL https://arxiv.org/abs/2403.15852.
[31] Bo Liu, Yanjie Jiang, Yuxia Zhang, Nan Niu, Guangjie Li, and Hui Liu. An
empirical study on the potential of llms in automated software refactoring. arXiv
preprint arXiv:2411.04444 , 2024.
[32] Davood Mazinanian, Nikolaos Tsantalis, Raphael Stein, and Zackary Valenta.
Jdeodorant: clone refactoring. In Laura K. Dillon, Willem Visser, and Laurie A.

Conference’17, July 2017, Washington, DC, USA Yisen Xu, Feng Lin, Jinqiu Yang, Tse-Hsun (Peter) Chen, and Nikolaos Tsantalis
Williams, editors, Proceedings of the 38th International Conference on Software
Engineering, ICSE 2016, Austin, TX, USA, May 14-22, 2016 - Companion Volume ,
pages 613–616. ACM, 2016. doi: 10.1145/2889160.2889168. URL https://doi.org/
10.1145/2889160.2889168.
[33] Raimund Moser, Alberto Sillitti, Pekka Abrahamsson, and Giancarlo Succi. Does
refactoring improve reusability? In International conference on software reuse ,
pages 287–297. Springer, 2006.
[34] Emerson Murphy-Hill, Chris Parnin, and Andrew P Black. How we refactor, and
how we know it. IEEE Transactions on Software Engineering , 38(1):5–18, 2011.
[35] Stas Negara, Nicholas Chen, Mohsen Vakilian, Ralph E. Johnson, and Danny
Dig. A comparative study of manual and automated refactorings. In 27th
European Conference on Object-Oriented Programming , ECOOP’13, pages 552–
576, Berlin, Heidelberg, 2013. Springer-Verlag. ISBN 978-3-642-39037-1. doi:
10.1007/978-3-642-39038-8_23.
[36] Pedram Nouri. PurityChecker: A Tool for Detecting Purity of Method-level
Refactoring Operations . PhD thesis, Concordia University, 2023. URL https:
//spectrum.library.concordia.ca/id/eprint/993129/.
[37] Mark Kent O’Keeffe and Mel Ó Cinnéide. Search-based refactoring for software
maintenance. J. Syst. Softw. , 81(4):502–516, 2008. doi: 10.1016/J.JSS.2007.06.003.
URL https://doi.org/10.1016/j.jss.2007.06.003.
[38] Jevgenija Pantiuchina, Bin Lin, Fiorella Zampetti, Massimiliano Di Penta, Michele
Lanza, and Gabriele Bavota. Why do developers reject refactorings in open-source
projects? ACM Trans. Softw. Eng. Methodol. , 31(2), December 2021. ISSN 1049-
331X. doi: 10.1145/3487062. URL https://doi.org/10.1145/3487062.
[39] Massimiliano Di Penta, Gabriele Bavota, and Fiorella Zampetti. On the relation-
ship between refactoring actions and bugs: A differentiated replication, 2020.
URL https://arxiv.org/abs/2009.11685.
[40] Anthony Peruma, Steven Simmons, Eman Abdullah AlOmar, Christian D. New-
man, Mohamed Wiem Mkaouer, and Ali Ouni. How do i refactor this? an empirical
study on refactoring trends and topics in stack overflow. Empirical Software En-
gineering , 27(1), October 2021. ISSN 1573-7616. doi: 10.1007/s10664-021-10045-x.
URL http://dx.doi.org/10.1007/s10664-021-10045-x.
[41] Russell A Poldrack, Thomas Lu, and Gašper Beguš. Ai-assisted coding: Experi-
ments with gpt-4. arXiv preprint arXiv:2304.13187 , 2023.
[42] Dorin Pomian. Llm-powered extract method, 2024. URL https://plugins.jetbrains.
com/plugin/23403-llm-powered-extract-method. Accessed: Jun. 23, 2024.
[43] Dorin Pomian, Abhiram Bellur, Malinda Dilhara, Zarina Kurbatova, Egor Bogo-
molov, Timofey Bryksin, and Danny Dig. Next-generation refactoring: Combin-
ing LLM insights and IDE capabilities for extract method. In IEEE International
Conference on Software Maintenance and Evolution, ICSME 2024, Flagstaff, AZ,
USA, October 6-11, 2024 , pages 275–287. IEEE, 2024. doi: 10.1109/ICSME58944.
2024.00034. URL https://doi.org/10.1109/ICSME58944.2024.00034.
[44] Dorin Pomian, Abhiram Bellur, Malinda Dilhara, Zarina Kurbatova, Egor Bo-
gomolov, Andrey Sokolov, Timofey Bryksin, and Danny Dig. Em-assist: Safe
automated extractmethod refactoring with llms. In Marcelo d’Amorim, edi-
tor,Companion Proceedings of the 32nd ACM International Conference on the
Foundations of Software Engineering, FSE 2024, Porto de Galinhas, Brazil, July
15-19, 2024 , pages 582–586. ACM, 2024. doi: 10.1145/3663529.3663803. URL
https://doi.org/10.1145/3663529.3663803.
[45] Soumaya Rebai, Marouane Kessentini, Vahid Alizadeh, Oussama Ben Sghaier,
and Rick Kazman. Recommending refactorings via commit message analysis.
Inf. Softw. Technol. , 126:106332, 2020. doi: 10.1016/J.INFSOF.2020.106332. URL
https://doi.org/10.1016/j.infsof.2020.106332.
[46] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using
siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods
in Natural Language Processing . Association for Computational Linguistics, 11
2019. URL https://arxiv.org/abs/1908.10084.
[47] Shuo Ren, Daya Guo, Shuai Lu, Long Zhou, Shujie Liu, Duyu Tang, Neel Sun-
daresan, Ming Zhou, Ambrosio Blanco, and Shuai Ma. Codebleu: a method for
automatic evaluation of code synthesis. arXiv preprint arXiv:2009.10297 , 2020.
[48] Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance framework:
Bm25 and beyond. Foundations and Trends ®in Information Retrieval , 3(4):333–389,
2009.
[49] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei
Zaharia. Colbertv2: Effective and efficient retrieval via lightweight late interaction.
In Marine Carpuat, Marie-Catherine de Marneffe, and Iván Vladimir Meza Ruíz,
editors, Proceedings of the 2022 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies, NAACL
2022, Seattle, WA, United States, July 10-15, 2022 , pages 3715–3734. Association
for Computational Linguistics, 2022. doi: 10.18653/V1/2022.NAACL-MAIN.272.
URL https://doi.org/10.18653/v1/2022.naacl-main.272.
[50] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan,
and Shunyu Yao. Reflexion: language agents with verbal reinforcement
learning. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko,Moritz Hardt, and Sergey Levine, editors, Advances in Neural Informa-
tion Processing Systems 36: Annual Conference on Neural Information Pro-
cessing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10
- 16, 2023 , 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/
1b44b878bb782e6954cd888628510e90-Abstract-Conference.html.
[51] Atsushi Shirafuji, Yusuke Oda, Jun Suzuki, Makoto Morishita, and Yutaka
Watanobe. Refactoring programs using large language models with few-shot
examples. In 2023 30th Asia-Pacific Software Engineering Conference (APSEC) ,
pages 151–160. IEEE, 2023.
[52] Danilo Silva, Ricardo Terra, and Marco Tulio Valente. Recommending automated
extract method refactorings. In Proceedings of the 22nd international conference
on program comprehension , pages 146–156, 2014.
[53] Danilo Silva, Nikolaos Tsantalis, and Marco Túlio Valente. Why we refactor?
confessions of github contributors. In Thomas Zimmermann, Jane Cleland-Huang,
and Zhendong Su, editors, Proceedings of the 24th ACM SIGSOFT International
Symposium on Foundations of Software Engineering, FSE 2016, Seattle, WA, USA,
November 13-18, 2016 , pages 858–870. ACM, 2016. doi: 10.1145/2950290.2950305.
URL https://doi.org/10.1145/2950290.2950305.
[54] Danilo Silva, João Paulo da Silva, Gustavo Santos, Ricardo Terra, and Marco Tulio
Valente. Refdiff 2.0: A multi-language refactoring detection tool. IEEE Transactions
on Software Engineering , 47(12):2786–2802, 2021. doi: 10.1109/TSE.2020.2968072.
[55] Yahya Tashtoush, Zeinab Odat, Izzat M Alsmadi, and Maryan Yatim. Impact of
programming features on code readability. 2013.
[56] Tom Tourwé and Tom Mens. Identifying refactoring opportunities using logic
meta programmin. In 7th European Conference on Software Maintenance and
Reengineering (CSMR 2003), 26-28 March 2003, Benevento, Italy, Proceedings , pages
91–100. IEEE Computer Society, 2003. doi: 10.1109/CSMR.2003.1192416. URL
https://doi.org/10.1109/CSMR.2003.1192416.
[57] Nikolaos Tsantalis and Alexander Chatzigeorgiou. Identification of move method
refactoring opportunities. IEEE Trans. Software Eng. , 35(3):347–367, 2009. doi:
10.1109/TSE.2009.1. URL https://doi.org/10.1109/TSE.2009.1.
[58] Nikolaos Tsantalis and Alexander Chatzigeorgiou. Identification of extract
method refactoring opportunities for the decomposition of methods. J. Syst.
Softw. , 84(10):1757–1782, 2011. doi: 10.1016/J.JSS.2011.05.016. URL https://doi.
org/10.1016/j.jss.2011.05.016.
[59] Nikolaos Tsantalis, Victor Guana, Eleni Stroulia, and Abram Hindle. A multidi-
mensional empirical study on refactoring activity. In James R. Cordy, Krzystof
Czarnecki, and Sang-Ah Han, editors, Center for Advanced Studies on Collabo-
rative Research, CASCON ’13, Toronto, ON, Canada, November 18-20, 2013 , pages
132–146. IBM / ACM, 2013. URL http://dl.acm.org/citation.cfm?id=2555539.
[60] Nikolaos Tsantalis, Matin Mansouri, Laleh M Eshkevari, Davood Mazinanian,
and Danny Dig. Accurate and efficient refactoring detection in commit history.
InProceedings of the 40th international conference on software engineering , pages
483–494, 2018.
[61] Nikolaos Tsantalis, Ameya Ketkar, and Danny Dig. Refactoringminer 2.0. IEEE
Transactions on Software Engineering , 48(3):930–950, 2020.
[62] Nalin Wadhwa, Jui Pradhan, Atharv Sonwane, Surya Prakash Sahu, Nagarajan
Natarajan, Aditya Kanade, Suresh Parthasarathy, and Sriram K. Rajamani. CORE:
resolving code quality issues using llms. Proc. ACM Softw. Eng. , 1(FSE):789–811,
2024. doi: 10.1145/3643762. URL https://doi.org/10.1145/3643762.
[63] Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, and James Zou. Mixture-
of-agents enhances large language model capabilities. CoRR , abs/2406.04692, 2024.
doi: 10.48550/ARXIV.2406.04692. URL https://doi.org/10.48550/arXiv.2406.04692.
[64] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei
Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits
reasoning in large language models, 2023. URL https://arxiv.org/abs/2201.11903.
[65] Jules White, Sam Hays, Quchen Fu, Jesse Spencer-Smith, and Douglas C Schmidt.
Chatgpt prompt patterns for improving code quality, refactoring, requirements
elicitation, and software design. In Generative AI for Effective Software Develop-
ment , pages 71–108. Springer, 2024.
[66] Di Wu, Fangwen Mu, Lin Shi, Zhaoqiang Guo, Kui Liu, Weiguang Zhuang, Yuqi
Zhong, and Li Zhang. ismell: Assembling llms with expert toolsets for code
smell detection and refactoring. In Vladimir Filkov, Baishakhi Ray, and Minghui
Zhou, editors, Proceedings of the 39th IEEE/ACM International Conference on
Automated Software Engineering, ASE 2024, Sacramento, CA, USA, October 27 -
November 1, 2024 , pages 1345–1357. ACM, 2024. doi: 10.1145/3691620.3695508.
URL https://doi.org/10.1145/3691620.3695508.
[67] Sihan Xu, Aishwarya Sivaraman, Siau-Cheng Khoo, and Jing Xu. Gems: An
extract method refactoring recommender. In 2017 IEEE 28th International Sympo-
sium on Software Reliability Engineering (ISSRE) , pages 24–34, 2017.
[68] Tong Ye, Weigang Huang, Xuhong Zhang, Tengfei Ma, Peiyu Liu, Jianwei Yin,
and Wenhai Wang. Llm4effi: Leveraging large language models to enhance code
efficiency and correctness, 2025. URL https://arxiv.org/abs/2502.18489.
[69] Zejun Zhang, Zhenchang Xing, Xiaoxue Ren, Qinghua Lu, and Xiwei Xu. Refac-
toring to pythonic idioms: A hybrid knowledge-driven approach leveraging
large language models. Proceedings of the ACM on Software Engineering , 1(FSE):
1107–1128, 2024.
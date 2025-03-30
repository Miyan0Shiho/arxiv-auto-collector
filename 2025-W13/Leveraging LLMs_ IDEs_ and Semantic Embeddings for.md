# Leveraging LLMs, IDEs, and Semantic Embeddings for Automated Move Method Refactoring

**Authors**: Fraol Batole, Abhiram Bellur, Malinda Dilhara, Mohammed Raihan Ullah, Yaroslav Zharov, Timofey Bryksin, Kai Ishikawa, Haifeng Chen, Masaharu Morimoto, Shota Motoura, Takeo Hosomi, Tien N. Nguyen, Hridesh Rajan, Nikolaos Tsantalis, Danny Dig

**Published**: 2025-03-26 19:05:20

**PDF URL**: [http://arxiv.org/pdf/2503.20934v1](http://arxiv.org/pdf/2503.20934v1)

## Abstract
MOVEMETHOD is a hallmark refactoring. Despite a plethora of research tools
that recommend which methods to move and where, these recommendations do not
align with how expert developers perform MOVEMETHOD. Given the extensive
training of Large Language Models and their reliance upon naturalness of code,
they should expertly recommend which methods are misplaced in a given class and
which classes are better hosts. Our formative study of 2016 LLM recommendations
revealed that LLMs give expert suggestions, yet they are unreliable: up to 80%
of the suggestions are hallucinations. We introduce the first LLM fully powered
assistant for MOVEMETHOD refactoring that automates its whole end-to-end
lifecycle, from recommendation to execution. We designed novel solutions that
automatically filter LLM hallucinations using static analysis from IDEs and a
novel workflow that requires LLMs to be self-consistent, critique, and rank
refactoring suggestions. As MOVEMETHOD refactoring requires global,
projectlevel reasoning, we solved the limited context size of LLMs by employing
refactoring-aware retrieval augment generation (RAG). Our approach, MM-assist,
synergistically combines the strengths of the LLM, IDE, static analysis, and
semantic relevance. In our thorough, multi-methodology empirical evaluation, we
compare MM-assist with the previous state-of-the-art approaches. MM-assist
significantly outperforms them: (i) on a benchmark widely used by other
researchers, our Recall@1 and Recall@3 show a 1.7x improvement; (ii) on a
corpus of 210 recent refactorings from Open-source software, our Recall rates
improve by at least 2.4x. Lastly, we conducted a user study with 30 experienced
participants who used MM-assist to refactor their own code for one week. They
rated 82.8% of MM-assist recommendations positively. This shows that MM-assist
is both effective and useful.

## Full Text


<!-- PDF content starts -->

Leveraging LLMs, IDEs, and Semantic Embeddings
for Automated Move Method Refactoring
Fraol Batole*
Tulane University
fbatole@tulane.eduAbhiram Bellur*
University of Colorado Boulder
abhiram.bellur@colorado.eduMalinda Dilhara
Amazon Web Services
malwala@amazon.comMohammed Raihan Ullah
University of Colorado Boulder
raihan.ullah@colorado.edu
Yaroslav Zharov
JetBrains Research
yaroslav.zharov@jetbrains.comTimofey Bryksin
JetBrains Research
timofey.bryksin@jetbrains.comKai Ishikawa
NEC Corporation
k-ishikawa@nec.comHaifeng Chen
NEC Laboratories America
haifeng@nec-labs.com
Masaharu Morimoto
NEC Corporation
m-morimoto@nec.comShota Motoura
NEC Corporation
motoura@nec.comTakeo Hosomi
NEC Corporation
takeo.hosomi@nec.comTien N. Nguyen
University of Texas at Dallas
tien.n.nguyen@utdallas.eduHridesh Rajan
Tulane University
hrajan@tulane.edu
Nikolaos Tsantalis
Concordia University
nikolaos.tsantalis@concordia.caDanny Dig
JetBrains Research, University of Colorado Boulder
danny.dig@jetbrains.com
Abstract —M OVEMETHOD is a hallmark refactoring. Despite
a plethora of research tools that recommend which methods to
move and where, these recommendations do not align with how
expert developers perform M OVEMETHOD . Given the extensive
training of Large Language Models and their reliance upon nat-
uralness of code , they should expertly recommend which methods
aremisplaced in a given class and which classes are better hosts.
Our formative study of 2016 LLM recommendations revealed
that LLMs give expert suggestions, yet they are unreliable: up
to 80% of the suggestions are hallucinations.
We introduce the first LLM fully powered assistant for
MOVEMETHOD refactoring that automates its whole end-to-end
lifecycle, from recommendation to execution. We designed novel
solutions that automatically filter LLM hallucinations using static
analysis from IDEs and a novel workflow that requires LLMs
to be self-consistent, critique, and rank refactoring suggestions.
As M OVEMETHOD refactoring requires global, project-level
reasoning, we solved the limited context size of LLMs by employ-
ingrefactoring-aware retrieval augment generation (RAG). Our
approach, MM- ASSIST , synergistically combines the strengths of
the LLM, IDE, static analysis, and semantic relevance. In our
thorough, multi-methodology empirical evaluation, we compare
MM- ASSIST with the previous state-of-the-art approaches. MM-
ASSIST significantly outperforms them: (i) on a benchmark
widely used by other researchers, our Recall@1 and Recall@3
show a 1.7x improvement; (ii) on a corpus of 210 recent
refactorings from Open-source software, our Recall rates improve
by at least 2.4x. Lastly, we conducted a user study with 30
experienced participants who used MM- ASSIST to refactor their
own code for one week. They rated 82.8% of MM- ASSIST
recommendations positively. This shows that MM- ASSIST is both
effective and useful.
I. I NTRODUCTION
MOVEMETHOD is a key refactoring [1] that relocates a mis-
placed method to a more suitable class. A method is misplacedif it interacts more with another class’ state than its own.
MOVEMETHOD improves modularity by aligning methods
with relevant data, enhances cohesion, and reduces coupling.
It removes code smells like FeatureEnvy [2], GodClass [3],
DuplicatedCode [4], and MessageChain [1], reducing technical
debt. It ranks among the top-5 most common refactorings [5–
7], both manually and automatically performed.
The M OVEMETHOD lifecycle has four phases: (i) identi-
fying a misplaced method min its host class H, (ii) finding a
suitable target class T, (iii) ensuring refactoring pre- and post-
conditions to preserve behavior, and (iv) executing the trans-
formation. Each phase is challenging—identifying candidates
requires understanding design principles and the codebase,
while checking preconditions [2, 8] demands complex static
analysis. The mechanics involve relocating m, updating call
sites, and adjusting accesses. Due to these complexities, exist-
ing solutions are incomplete; IDEs handle preconditions and
mechanics, while research tools aim to identify opportunities.
The research community has proposed various ap-
proaches [2, 9–16] for identifying misplaced methods or rec-
ommending new target classes, typically optimizing software
quality metrics like cohesion and coupling. These approaches
fall into three categories: (i) static analysis-based [2, 9, 10], (ii)
machine learning classifiers [11–13], and (iii) deep learning-
based [14–16]. However, static analysis relies on expert-
defined thresholds, and ML/DL methods require continual
retraining as coding standards evolve, often diverging from
real-world development practices.
We hypothesize that achieving good software design that
is easy to understand and resilient to future changes is a
balancing act between science (e.g., metrics, design principles)arXiv:2503.20934v1  [cs.SE]  26 Mar 2025

andart(e.g., experience, expertise, and intuition about what
constitute good abstractions). This can explain why refactor-
ings that optimize software quality metrics are not always
accepted in practice [17–26].
In this paper, we introduce the first approach to automate
the entire M OVEMETHOD refactoring lifecycle using Large
Language Models (LLMs). We hypothesize that, due to their
extensive pre-training on billions of methods and their reliance
on the naturalness of code , LLMs can generate an abundance
of M OVEMETHOD recommendations. We also expect LLM
recommendations to better align with expert practices. In our
formative study, we found LLMs (GPT-4o, in particular) are
prolific in generating suggestions, averaging 6 recommen-
dations per class. However, two major challenges must be
addressed to make this approach practical.
First, LLMs produce hallucinations , i.e., recommendations
that seem plausible but are flawed. In our formative study
of 2016 LLM recommendations, we identified three types of
hallucinations (e.g., recommendations where the target class
does not exist), and found that up to 80% of LLM recom-
mendations are hallucinations. We discovered novel ways to
automatically eliminate LLM hallucinations, by complement-
ing LLM reasoning (i.e., the creative, non-deterministic, and
artistic part akin to human naturalness) with static analy-
sis embedded in the IDE (i.e., the rigorous, deterministic,
scientific part ). We utilized code-trained vector embeddings
from AI models to identify misplaced methods, and used
refactoring preconditions [8] in existing IDEs (IntelliJ IDEA)
to effectively remove allLLM hallucinations. We present these
techniques in Section III-B.
Second, M OVEMETHOD refactoring requires global,
project-level reasoning to determine the best target classes
where to relocate a misplaced method. However, passing an
entire project in the prompt is beyond the limited window size
of current LLMs [27]. Even with larger window sizes, passing
the whole project as context introduces noise and redundancy,
as not all classes are relevant; instead this further distracts the
LLM [27, 28]. We address the limited context size of LLMs
by using retrieval augmented generation (RAG) to enhance
the LLM’s input. Our two-step retrieval process combines
mechanical feasibility (IDE-based static analysis) with
semantic relevance (V oyageAI [29]), enabling our approach
to make informed decisions and perform global project-level
reasoning. We coin this approach refactoring-aware retrieval
augmented generation , which addresses LLM hallucinations
and context limitations while fulfilling the specific needs of
MOVEMETHOD refactoring (see Section III-C).
We designed, implemented, and evaluated these novel solu-
tions as an IntelliJ IDEA plugin for Java, MM- ASSIST (Move
Method Professional). It synergistically combines the strengths
of the LLM, IDE, static analysis, and semantic relevance.
MM- ASSIST generates candidates, filters LLM hallucinations,
validates and ranks recommendations, and then finally exe-
cutes the correct refactoring based on user approval using the
IDE.
We designed a comprehensive, multi-methodology evalua-tion of MM- ASSIST to corroborate, complement, and expand
research findings: formative study, comparative study, repli-
cation of real-world refactorings, repository mining, user/case
study, and questionnaire surveys. We compare MM- ASSIST
with the previous best in class approaches in their respective
categories: JM OVE [10] – uses static analysis, FETRUTH [16]
– uses Deep Learning, and HM OVE [30] – uses graph neural
network to suggest moves and LLM to verify refactoring pre-
conditions. These have been shown previously to outperform
all previous M OVEMETHOD recommendation tools. Using a
synthetic corpus widely used by previous approaches, we
found that MM- ASSIST significantly outperforms them: for
class instance methods, our Recall@1 and Recall@3 are 67%
and 75%, respectively, which is an almost 1.75x improvement
over previous state-of-the-art approaches (40% and 42%).
Moreover, we extend the corpus used by previous researchers
with 210 actual refactorings that we mined from OSS reposi-
tories in 2024 (thus avoiding LLM data contamination), con-
taining both instance and static methods. We compared against
JMOVE, HM OVE and FETRUTH on this real-world oracle, and
found that MM- ASSIST significantly outperforms them. Our
Recall@3 is 80%, compared to 33% for the previous best
tool, HM OVE– a 2.4x improvement. This shows that MM-
ASSIST ’s recommendations better align with human developer
best practices.
Whereas existing tools often require several hours to analyze
a project and overwhelm developers with up to 57 recommen-
dations to analyze per class, MM- ASSIST needs only about 30
seconds—even for tens of thousands of classes—and provides
no more than 3 high-quality recommendations per class.
In a study where 30 experienced participants used MM-
ASSIST on their own code for a week, 82.8% rated its recom-
mendations positively and preferred our LLM-based approach
over classic IDE workflows. One participant remarked, “I am
fairly skeptical when it comes to AI in my workflow, but still
excited at the opportunity to delegate grunt work to them. ”
This paper makes the following contributions:
•Approach. We present the first end-to-end LLM-powered
assistant for M OVEMETHOD . Our approach advances key
aspects: (i) recommendations are feasible and executed
correctly, (ii) it requires no user-specified thresholds or
model (re)-training, making it future-proof as LLMs
evolve, and (iii) it handles both instance and static meth-
ods (avoided by others due to large search space).
•Best Practices. We discovered a new set of best practices
to overcome the LLM limitations when it comes to refac-
torings that require global reasoning. We automatically
filter LLM hallucinations and conquer the LLM’s limited
context size using refactoring-aware RAG.
•Implementation. We designed, implemented, and evalu-
ated these ideas in an IntelliJ plugin, MM- ASSIST , that
works on Java code. It addresses practical considerations
for tools used in the daily workflow of developers.
•Evaluation. We thoroughly evaluated MM- ASSIST , and
it outperforms previous best-in-class approaches. We also
created an oracle replicating actual refactorings done by

public  class  EsqlSession  {
  private  PolicyResolver  policyResolver ;
  ... 
  public  void  execute (EsqlQueryRequest  request , ...){ 
     LOGGER .debug ("ESQL query: \n{}", request .query ()); ...}
  private  LogicalPlan  parse (String  query , ...) {...} 
  public  void  analyzedPlan (...) {...} 
  public  void  optimizedPlan (...) {...} 
  
  private  void  preAnalyze (...) {
  ... 
    resolvePolicy (groupedListener, policyNames, resolution); 
  }
  ... 
}
2
3 4
policyResolver. resolvePolicy (...); /* Resolves a set of policies and adds them to 
a given resolution.*/ 
private  void  resolvePolicy (
  ActionListener  groupedListener ,
  Set policyNames , 
  Resolution  resolution ) {
    ...
    for  (policyName  : policyNames) { 
      policyResolver .resolvePolicy (
          policyName, 
     resolution .resolvedPolicies ()::add) 
 );
    } 
}21
34
PolicyResolver Fig. 1: A real-world example demonstrating a M OVEMETHOD onresolvePolicy performed by developers in the Elasticsearch
project, commit 876e7015
OSS developers, where MM- ASSIST showed even more
improvements. Our user study confirms that MM- ASSIST
aligns with and replicates real-world expert logic.
II. M OTIVATING EXAMPLE
We illustrate the challenges of recommending
MOVEMETHOD using a real-world refactoring that occurred
in the Elasticsearch project – a distributed open-source
search and analytics engine. We illustrate the refactoring
in Figure 1, and the full commit can be seen in [31]. The
resolvePolicy method (See 4in Figure 1), originally part
of the EsqlSession class, is misplaced. While EsqlSession
handles parsing and executing queries, resolvePolicy is
responsible for resolving and updating policies. Specifically,
resolvePolicy accesses the field policyResolver (See 1)
and parameters like groupedListener ,policyNames , and
resolution . Recognizing this misalignment, the developers
refactored the code by moving resolvePolicy to the
PolicyResolver class (not shown in the figure due to space
constraints), updating the method body accordingly, and
modifying the call sites (See 3). After the refactoring, both
EsqlSession andPolicyResolver became more cohesive.
Automatically identifying such refactoring opportunities
is essential for maintaining software quality, but it poses
significant challenges for existing tools. We first applied
HM OVE [30], the state-of-the-art M OVEMETHOD technique.
HM OVE, a classification tool, takes in <method, targetClass >
pairs, and gives a probability score indicating whether to move
the method. After computing all 158 pairs of inputs, HM OVE
executed for 1.5 hours and generated 36 M OVEMETHOD rec-
ommendations. Its first and second highest recommendations
to move are subfields andexecute . Its third recommendation
to move is resolvePolicy . However, HM OVE recommended
moving resolvePolicy to the class FunctionRegistry , which
is not an appropriate fit, as the method does not interact
with it. This illustrates major shortcomings of classification-
based tools like HM OVE: they need to be triggered on lots(avg. 145) of <method, targetClass >pairs, which means long
runtime. Furthermore, they overwhelm the users with so many
recommendations to analyze, testing the developer’s patience
and endurance. Moreover, they don’t provide actionable steps.
Next, we ran JM OVE [10], a state-of-the-art
MOVEMETHOD recommendation tool that solely relies on
static analysis. To analyze the whole project JM OVE requires
12+ hours. To speed up JM OVE, we ran it on a sub-project
of Elasticsearch containing EsqlSession . After it finished
running for 30 minutes on a sub-project of Elasticsearch,
JMOVE did not produce any recommendations for the
EsqlSession class. This highlights a major shortcoming of
existing static-analysis based tools like JM OVE: they need
to analyze the entire project-structure and compute program
dependencies – thus they do not scale to medium to large-size
projects like Elasticsearch (800K LOC).
Next, we explored the potential of Large Language Mod-
els (LLMs) to recommend M OVEMETHOD refactoring. We
used GPT-4o, a state-of-the-art LLM developed by Ope-
nAI [32], and prompted it with the content of the EsqlSession
class, asking: “Identify methods that should move out of the
EsqlSession class and where?” . Our result highlighted both
the strengths and limitations of LLMs for this task. In order of
priority, the LLM identified 5 methods for relocation (see 2),
including execute ,parse ,optimizedPlan , and analysePlan ,
all of which rightly belong in Esqlsession and were never
moved by developers. Notably, the LLM did successfully
identify resolvePolicy as a candidate for refactoring, show-
ing its ability to detect semantically misplaced methods.
Despite success, the LLM recommended other methods before
resolvePolicy . A developer would need to filter out many
irrelevant suggestions before arriving at a useful one.
After identifying that the method resolvePolicy is mis-
placed, a tool must find a suitable target class to move the
method into. While the LLM was able to recommend the
correct target class, it also responded with (i) two target classes
(i.e., Resolution ,ActionListener ), which are plausible target

classes, but are not the best-fit semantically for the method;
(ii) two hallucinations, i.e., classes that do not exist (i.e.,
PolicyResolutionService ,PolicyUtils as the LLM lacks
project-wide context. A naive approach to address the LLM’s
lack of project-wide understanding is to prompt it with the
entire codebase. However, this is currently impractical due to
the LLM’s context size limitations and inability to efficiently
handle long contexts [33] (even though context limits con-
tinuously increase). Even state-of-the-art LLMs can’t process
large projects like Elasticsearch in a single prompt without
truncating crucial information. Moreover, the processing cost
of large inputs with commercial LLM APIs is prohibitive.
These experiments reveal both the strengths and limitations
of LLMs for M OVEMETHOD refactoring. On the positive side,
LLMs show proficiency in generating multiple suggestions
and demonstrate an ability to identify methods that are se-
mantically misplaced. However, they also exhibit significant
limitations, including difficulty in suggesting appropriate target
classes, and a high rate of irrelevant or infeasible suggestions.
These limitations underscore the need for caution when relying
on LLM-generated refactoring recommendations, and the need
for a tedious manual analysis. Developers need to manually
collect and re-analyze the suggestions, verify the suitability
of each method for relocation, prompt the LLM again for
suitable target classes, and meticulously identify and filter out
hallucinations such as non-existent classes and methods that
are impossible to move. In the example (Figure 1), a developer
needs to sift through 5 candidate methods and, for each
method, understand if any of the 5 or more proposed target
classes are adequate. The developer analyzes 20+ <method,
target class >pairs before finding one they agree with.
This example motivates our approach, MM- ASSIST , which
significantly streamlines the refactoring process by (1) uti-
lizing semantic relevance to find candidate methods that are
the least compatible with the host class, (2) employing static
analysis to validate and filter suggestions, and (3) leveraging
LLMs to prioritize only valid recommendations. For the ex-
ample above, MM- ASSIST expertly recommends as the top
candidate moving resolvePolicy toPolicyResolver . MM-
ASSIST liberates developers so they can focus on the creative
part. Rather than sifting through many invalid recommenda-
tions, developers use their expertise to examine a few high-
quality recommendations.
III. A PPROACH
Figure 2 shows the architecture and the steps performed
by MM- ASSIST . First, MM- ASSIST applies a set of pre-
conditions that filter out the methods that cannot be safely
moved, such as constructors ( 1in Figure 2). It then leverages
vector embeddings from Language Models to identify methods
that are the least cohesive with their host class ( 2in Figure 2).
In Figure 1, by comparing the embeddings of resolvePolicy
andEsqlSession using cosine similarity, MM- ASSIST detects
that this method might be misplaced ( §III-B). Next, MM-
ASSIST passes the remaining candidates to the LLM (i.e.,
the method signature and the class body), which analyzestheir relationships with the host class to prioritize the most
promising M OVEMETHOD recommendations ( 3in Figure 2).
Once MM- ASSIST identifies candidate methods, it system-
atically evaluates potential target classes from the project
codebase. For the resolvePolicy method, which utilizes
the enrichPolicyResolver field ( 1in Figure 1), MM-
ASSIST initially identifies several candidate classes, including
EnrichPolicyResolver andPolicyManager .
To narrow down the target classes, MM- ASSIST calculates
relevance scores between the candidate method and each
potential target class – establishing a ranking ( 4in Figure 2).
We label this process as “refactoring-aware RAG”. Then, we
feed the LLM with the narrowed-down list of target classes,
and ask it to pick the best one ( 5in Figure 2). In this case,
it correctly selects EnrichPolicyResolver as the appropriate
destination for resolvePolicy , aligning with the developers’
actual refactoring decision (see 3in Figure 1). Finally,
refactoring suggestions ( 6in Figure 2) are presented to the
user, and MM- ASSIST leverages the IDE’s refactoring APIs
to safely execute the chosen one automatically.
Next, we discuss each of these steps and concepts in detail.
A. Important Concepts
Definition III.1. (MOVEMETHOD Refactoring) A
MOVEMETHOD refactoring moves a method mfrom a
host class H(where it currently resides) to a target class T.
We define a MOVEMETHOD "ω" as a triplet (m,H,T).
Definition III.2. (MOVEMETHOD Recommendations) A list
ofMOVEMETHOD refactoring candidates, ordered by priority
(most important at the beginning). We denote this with ℜ.
Definition III.3. (Valid Refactoring Recommendations) These
are recommendations that do not break the code. They are
mechanically feasible: syntactically correct and successfully
pass the preconditions as checked by the IDE. We differentiate
between moving an instance and a static method:
1) An Instance Method can be moved to a type in the
host class’ fields, or a type among the method’s param-
eters. Several preconditions ensure the validity of the
MOVEMETHOD recommendation, including:
•Method Movability: Is the method a part of the class
hierarchy? If not, the method can be safely moved.
•Access to references: Does the moved method loses
access to the references it needs for computation?
2) A Static Method can be moved to almost any class in
the project. A valid static method move is one where the
method can still access its references (e.g., fields, methods
calls) from the new location.
Definition III.4. (Invalid M OVEMETHOD Recommendations)
We classify the LLM’s invalid M OVEMETHOD suggestions
as hallucinations and categorize them as follows:
1)Target class does not exist (H1): The LLM comes up
with an imaginary class.

Java 
class 
Display Valid 
Suggestions Apply 
Refactoring 
Improved 
Java Class Prioritize 
methods 
to move? 
Critique 
Identify Valid 
Target Classes 
Critique Which 
class to 
move? Ranked 
Move-Method 
suggestions {}
{}
{}
Filter Valid 
Methods in 
class 
LM-based 
similarity 
Compute 
Similarity 1 2 3
5 4 6Least 
Compatible 
Methods {}
{}
{}
Prompt 
Prompt Fig. 2: Architecture of MM- ASSIST .
2)Mechanically infeasible to move (H2): The target class
exists, but a refactoring suggestion is invalid according
to definitions in the previous subsection III.3.
3)Invalid Methods (H3) : The method is a part of the
software design, and moving it requires multiple other
refactoring changes. For example, moving a getter/setter
needs to be accompanied by moving the appropriate field.
B. Identifying Which Method To Move
While we believe LLMs have great potential for suggesting
MOVEMETHOD refactorings, directly using LLMs to identify
potential methods that may be misplaced within a class is risky,
as it results in invalid M OVEMETHOD recommendations.
Filter Invalid Candidates via Sanity Checks. Following
established refactoring practices [10, 16, 30], we filter methods
that are likely already in the correct class. First, MM- ASSIST
filters out getter and setter methods, as they cannot be moved
without also relocating the associated fields. Next, it excludes
methods involved in inheritance chains that can be overridden
in subclasses, since moving these would require additional
structural changes. It also removes test methods and those with
irrelevant content, such as empty bodies or methods containing
only comments.
Identify Least Compatible Methods via Embedding-
Based Analysis. To further refine candidate methods, we
use an embedding-based analysis. An embedding is a vector
representation that captures the semantic features of an entity
(methods and classes) based on their content and relationships.
We leverage V oyageAI embeddings [29], specifically trained
on code, as they more effectively capture the semantic rel-
evance of programming constructs. We use V oyageAI [29]
due to its state-of-the-art performance in code-related tasks.
Vectors are generated for two inputs: one for the method
body and another for the host class, excluding the method
body. Excluding the method ensures that the class embedding
remains unbiased by the method itself. We then calculate the
cosine similarity between these vectors to assess how well
each method semantically aligns with its host class.
Prioritize Methods with LLM Guidance. To make the
analysis computationally tractable and fit the LLM context
size, we further narrow candidate methods based on LLM’s
understanding of class level design. We use the LLM to rankthe existing methods in our suggestion pool. Using Chain-of-
Thought (CoT) reasoning, we prompt the LLM to perform a
structured analysis: evaluate each method’s purpose, cohesion
and dependencies, summarize the host class’s responsibilities,
and assess overall alignment of the method & class.
C. Recommending Suitable Target Classes
After identifying potential methods for relocation, the sub-
sequent task is to determine the most appropriate target
classes for these methods. However, this presents a substantial
challenge, requiring a comprehensive analysis of the entire
codebase. LLMs struggle with such tasks due to their limited
context windows. To address this, we employ Retrieval Aug-
mented Generation (RAG) [34]. RAG is a systematic approach
designed to retrieve and refine relevant contextual information,
thereby augmenting the input to the LLM. In MM- ASSIST ,
we efficiently retrieve and augment the model with the most
relevant target classes. Instance methods can be moved to
a limited number of feasible classes. Static methods can be
moved to almost any class in the project. Thus, we compare
the structure and semantics to find the most suitable target
classes (described below). In both cases, we rank target classes
based on semantic relevance and finally provide a suitable list
of target classes to the LLM to allow it to choose the best fit.
Since we designed the retrieval process to enhance refactoring,
we call this “refactoring-aware RAG”, and we explain it below.
Target Class Retrieval - Instance Methods. As instance
methods can be moved to a few suitable classes - we select
the method’s parameter types and the host class’ field types as
potential destinations. Then, we utilize the IDE’s preconditions
to retain only valid M OVEMETHOD suggestions.
Target Class Retrieval - Static Methods. We identify
potential target classes within the project based on two key
aspects: package proximity and utility class identification.
Package proximity quantifies structural closeness in the pack-
age hierarchy by computing shared package segments (e.g., for
org.example.core and org.example.utils, "org" and "example"
are shared) normalized by the host package depth, providing
an initial structural filter. Utility classes, identified through
conventional naming patterns (containing "util" or "utility"),
are prioritized as common targets for static methods. These
heuristics are efficient filters to narrow down the search space

of potential target classes. We rank potential target classes
based on the above heuristics, with greater importance given
to proximity. Our ranking function is:
RankingScore (T) = 2·proximity (T,H) +isUtility (T)(1)
where:
•proximity (T,H)evaluates the package proximity between
class Tand the host class H
•isUtility (T)is a boolean function that returns 1 if Tis a
utility class, and 0 otherwise
Finally, we utilize static analysis from the IDE to validate
whether the method can be moved to the potential target class.
Semantic Relevance-Based Target Class Ranking. While
static analysis offers foundational understanding of valid refac-
toring opportunities, it often yields a broad set of potential
target classes, as it lacks the ability to capture deeper semantic
relationships. To augment the results of static analysis, we
incorporate a semantic relevance analysis, which compares
both the content and intent of the candidate method and
target classes. To do this, we utilize V oyageAI’s vector em-
beddings [29] to compute the cosine similarity between the
method body and potential target classes. We sort the target
classes by their cosine similarity scores in descending order to
select the most semantically relevant candidates for the LLM
to analyze. To stay within the LLM’s context window, we
limit the candidates to those fitting within a 7K token budget
– typically accommodating 10-12 class summaries with their
signatures.
Ranking Target Classes Using LLM. In the final phase,
MM- ASSIST asks the LLM for the best-suited target class,
utilizing its vast training knowledge. To avoid context over-
flow, we create a concise representation of each target class,
including its name, field declarations, DocString, and method
signatures. The LLM then takes as input the method to be
moved along with these summarized target class representa-
tions, returning a prioritized list of target classes.
D. Applying Refactoring Changes
After compiling a list of M OVEMETHOD recommendations,
MM- ASSIST presents the method-class pairs to developers
through an interactive interface, accompanied by a rationale
explaining each suggestion. Upon developer selection of a spe-
cific recommendation, MM- ASSIST encapsulates the approved
method-target class pair into a refactoring command object. It
then executes the command automatically through the IDE’s
refactoring APIs, ensuring safe code transformation (moving
the method, and changing all call sites and references).
IV. E MPIRICAL EVALUATION
To evaluate MM- ASSIST ’s effectiveness and usefulness,
we designed a comprehensive, multi-methodology evaluation
to corroborate, complement, and expand our findings. This
includes a formative study, comparative study, replication of
real-world refactorings, repository mining, user study, and
questionnaire surveys. These methods combine qualitative and
quantitative data, and together answer four research questions.RQ1. How effective are LLMs at suggesting
MOVEMETHOD refactoring opportunities? This question
assesses Vanilla LLMs’ ability to recommend M OVEMETHOD
through a formative study examining the quality of LLM
suggestions.
RQ2. How effective is MM- ASSIST at suggesting
MOVEMETHOD refactoring opportunities? We evaluate
the performance of MM- ASSIST against the state-of-the-art
tools, FETRUTH [16] and HM OVE [30] (representatives for
DL approaches), and JM OVE [10] (representative for static
analysis approaches). We used both a synthetic corpus used
by other researchers and a new dataset of real refactorings
from open-source developers.
RQ3. What is MM- ASSIST ’s runtime performance? This
helps us understand MM- ASSIST ’s scalability and suitability
for integration into developers’ workflows.
RQ4. How useful is our approach for developers? We focus
on the utility of MM- ASSIST from a developer’s perspective.
We conduct a user study with 30 participants with industry
experience who used our tool on their own code for a week.
A. Subject Systems
To evaluate LLMs’ capability when suggesting
MOVEMETHOD refactoring opportunities, we employed
two distinct datasets: a synthetic corpus widely used by
previous researchers [10, 13, 30, 35] and a new corpus that
contains real-world refactorings that open-source developers
performed. Each corpus comes along with a “gold set” Gof
MOVEMETHOD refactorings that a recommendation tool must
attempt to match. We define Gas a set of M OVEMETHOD
refactorings (see Definition III.1) - each containing a triplet
of method-to-move, host class, and target class (m,H,T).
Synthetic corpus. The synthetic corpus was created by
Terra et al. [10] moving different methods mout of their
original/host class Hto a random destination class T. The
researchers then created the gold set as tuples (m,H,T), i.e.,
methods mthat a tool should now move from Hback to its
original class T. This dataset moves only instance methods ;
it does not move static methods . This corpus consists of 10
open-source projects.
Real-world corpus. As refactorings in the real-world are
often complex and messy [36], we decided to complement
the synthetic dataset with a corpus of actual MOVEMETHOD
refactorings that open-source developers performed on their
projects. This dataset allows us to determine whether various
tools can match the rationale of expert developers in real-
world situations. We construct this oracle using Refactoring-
Miner [37], the state-of-the-art for detecting refactorings [38].
We took extra precautions to prevent LLM data contam-
ination , ensuring that the LLMs used by MM- ASSIST had
no prior exposure to the data we tested and could not rely
on previously memorized results. With GPT-4’s knowledge
cutoff in October 2023, we focused our analysis on repository
commits from January 2024 onward. We ran RefactoringMiner
on the 25 most actively maintained Java repositories on
GitHub, ranked by commit history, and with over 1000 stars.

However, many of the M OVEMETHOD reported by Refac-
toringMiner are false positives, often resulting from residual
effects of other refactorings such as M OVECLASS (where an
entire class is relocated to another package) or E XTRACT
CLASS (where a class is split into two, creating a new class
along with the original). To filter out these false positives, we
employed several techniques: first, we verified that both the
source and target classes existed in both versions of the code
(i.e., at the commit head and its previous head). We removed
test methods, getters, setters, and overriden methods (they
violate preconditions for a M OVEMETHOD ). For instance
methods, we then checked if the method was moved to a field
in the source class, to a parameter type, or if the signature of
the moved method contained a reference to the source class.
Starting from the instances detected by RefactoringMiner, we
curated a dataset of 210 verified M OVEMETHOD , with 102
static methods and 108 instance methods – on 12 popular
open-source projects. On average, each project contains 8743
classes and 66306 methods spanning 1032344 LOCs. This
oracle enables an evaluation of MM- ASSIST on authentic
refactorings made by experienced developers.
B. Effectiveness of LLMs (RQ1)
Evaluation Metrics. Using these datasets, we evaluated the
recommendations made by the vanilla LLM and identified the
hallucinations, as defined in Definition III.4: H1 – target class
does not exist in the project; H2 – moving the method to
the target class is mechanically infeasible; and H3 – violating
preconditions in Section III-B.
Experimental Setup. We use the vanilla implementa-
tion of GPT-4o, a state-of-the-art LLM from OpenAI [32].
While MM- ASSIST is model agnostic (i.e., we can simply
swap different models), we chose GPT-4o because other re-
searchers [39, 40] show that it outperforms other LLMs when
used for refactoring tasks. GPT-4o is also widely adopted in
many software-engineering tools [41–44]. We designed our
experimental setup to assess the model’s inherent capabilities
in understanding and recommending M OVEMETHOD without
additional context or task-specific tuning. We formulated a
prompt where we provided the source code in a given host
class and asked the LLM which methods to move and where.
We set the LLM temperature parameter to 0 to obtain deter-
ministic results.
For each host class in our Gold sets (synthetic and real-
world), we submitted prompts to the LLM and collected its
recommendations.
TABLE I: Different kinds of hallucinations from Vanilla LLM
Corpus # R # H1 # H2 # H3
Synthetic (235) 723 362 168 51
Real-world (210) 1293 431 275 320
R: Recommendations, H1: Hall-class, H2: Hall-Mech, H3: Invalid Method.
Results. Table I illustrates the distribution of valid sugges-
tions and different types of hallucinations produced by the
vanilla LLM for both synthetic and real-world datasets. Weobserved a prevalence in all three types of hallucinations:
H1, H2, and H3 (as defined in Definition III.4) Crucially,
actuating any of these hallucinations would lead to broken
code, compilation errors, or degraded software design.
In both the synthetic and real-world dataset, a mere 20%
(142/723 and 267/1293 respectively) were valid. The over-
whelming 80% were hallucinations. These findings underscore
the impracticality of using vanilla LLM recommendations for
MOVEMETHOD without extensive filtering and validation. For
every valid recommendation, a developer would need to sift
through and discard 3-4 invalid ones – which may introduce
critical errors if implemented. This undermines the potential
time-saving benefits of automated refactoring and introduces
significant risks of introducing bugs or degrading code quality.
C. Effectiveness of MM- ASSIST (RQ2)
To evaluate MM- ASSIST ’s effectiveness, we compared it
against state-of-the-art M OVEMETHOD recommendation tools.
Baseline Tools. We directly compare with the best-in-class
tools: JM OVE [10] is a state-of-the-art static analysis tool,
FETRUTH [16] is the former best in class tool that uses
ML/DL, and HM OVE [30] is a recently introduced state-
of-the-art tool that uses graph neural networks to generate
suggestions and LLM to check refactoring preconditions.
HM OVE has been shown to outperform all previous tools.
We also compare with the Vanilla-LLM (GPT 4o), which
represents the standard LLM solution (without using MM-
ASSIST ’s enhancements). We went the extra mile to ensure the
most fair comparison: we consulted with HM OVE,FETRUTH ,
and JM OVE authors to ensure the optimal tool’s settings and
clarified with the authors when their tools did not produce the
expected results. We are grateful for their assistance.
Evaluation Metrics. For evaluation, we employ recall-
based metrics, following an approach similar to that used
in the evaluation of JMove [10], a well-established tool in
this domain. In our setting, recall is a more suitable metric
(against precision) because it measures how many relevant rec-
ommendations are retrieved, avoiding the need for subjective
judgments about whether a recommendations is a false posi-
tive. Furthermore, to provide actionable recommendations and
avoid overwhelming the developer, we use recall@k, which
returns a small number of recommendations (k). Furthermore,
recall@k is similar to precision by evaluating the quality of a
limited set of recommendations.
We present recall for each phase of suggesting the move-
method refactoring: first, identifying that a method is mis-
placed, no matter the recommened target class ( Recall M);
second, identifying a target class for the misplaced method
(Recall C); third, identifying the entire chain of refactoring: se-
lecting the right method and the right target class ( Recall MC).
For a recommendation list ℜ(see III.2), we define Recall M,
Recall CandRecall MCas follows:
Recall M=|ℜM|
|G|,Recall C=|ℜ∩G|
|ℜM|,Recall MC =|ℜ∩G|
|G|

TABLE II: Recall rates of MM- ASSIST on the synthetic corpus of 235 refactorings [10] that moved instance methods. Recall M
= identify the method, Recall C= identify the target class for a method, Recall MC= identify the method&target class pair
ApproachRecall M Recall C Recall MC
@1 @2 @3 @1 @2 @3 @1 @2 @3
JMOVE 41% 43% 43% 97% 97% 97% 40% 42% 42%
FETRUTH 2% 3% 3% 100% 100% 100% 2% 3% 3%
HM OVE 31% 37% 40% 32% 37% 39% 21% 24% 26%
Vanilla-LLM 71±2% 75±1% 79±2% 70±1% 70±1% 70±1% 53±2% 55±2% 57±2%
MM- ASSIST 72±1% 79±0% 80±0% 91±1% 97±0% 98±1% 67±0% 73±0% 75±0%
Where ℜMis the subset of ℜcontaining refactorings whose
method components match those in the ground truth set G.
Formally, we define ℜMas follows:
ℜM={g|g∈G∧ ∃(gm, gc,∗)∈ ℜ}
For each recall metric, we calculate Recall@k for the top
krecommendations, where k ∈{1, 2, 3}.
Experimental Setup. We trigger all tools on each host class
from the gold set (both synthetic and the real-world dataset).
To account for the inherent non-determinism in LLMs, we
ran the vanilla LLM with multiple temperature values (0,
0.5, 1) for five runs each. Additionally, we ran MM- ASSIST
three times. We report the average and standard deviation for
both vanilla LLM and MM- ASSIST . Considering the number
of entries in the datasets, given that JM OVE can take a
long time to run (12+ hours on a large project), we cutoff
its execution after 1 hour. HM OVE takes both the method
and candidate target class as input, and returns a probability
score indicating whether to move the method. As a result, it
becomes impractical to recommend moving static methods,
because there can be thousands of (method, target-class) pairs
– each taking a minute on average to classify. For example, to
recommend moving a single static-method in the Elasticsearch
project with 21615 target classes would require ≈14days to
process. Thus, we limit HM OVE to instance move method
recommendations.
All the tools generate a ranked list of M OVEMETHOD
suggestions. We compared these recommendations against
the ground truth to calculate recall for each tool using the
evaluation metrics presented earlier – Recall M,Recall C, and
Recall MC.
Results on the Synthetic Corpus. Table II shows the
effectiveness of MM- ASSIST and baseline tools on the syn-
thetic dataset. MM- ASSIST demonstrates superior perfor-
mance across many of the recall metrics compared to other
tools, especially Recall MC@1(1.7x to 30x improvement)
– the most comprehensive measure assessing both correct
method and target class identification. While JM OVE exhibits
high accuracy in target class identification ( Recall C@1=
97%), it shows limitations in method identification. HM OVE
demonstrates comparable performance as JM OVE in iden-
tifying misplaced methods (i.e., Recall M) but does not
match JM OVE’s ability to recommend the correct target class
(i.e., Recall C). Consequently, its combined Recall MC is
lower than JM OVE. Interestingly, FETRUTH achieves perfect
Recall Cbut extremely low Recall M(2% – 3%) despite
being very prolific in recommending as many as 67 methodsto be moved from a class. When it does correctly identify
a method, it accurately suggests the target class. However,
because it rarely identifies the misplaced methods themselves,
the overall Recall MCremains low. We confirmed this paradox
with FETRUTH authors. Interestingly, the Vanilla-LLM shows
comparable performance to MM- ASSIST in method identi-
fication ( Recall M). This could be because the LLMs have
been pre-trained on the synthetic dataset, and memorize their
responses as a result.
Results on the Real-world Corpus. With the real-world
dataset performed by open-source developers, we found a
wider difference in performance. First, we distinguish between
cases when the M OVEMETHOD target was an instance or
static method – shedding light on the effectiveness of MM-
ASSIST in different usage scenarios. Second, we distinguish
between small and large classes based on their method count.
Our analysis of the real-world oracle reveals a heavy-tail
distribution – we label classes with fewer than 15 methods
(90th percentile across all projects) as Small Classes (avg.
6 methods/class) and the rest as Large Classes (avg. 48
methods/class).
Tables III and IV summarize our results for instance
and static method moves, respectively. Since JM OVE could
not finish running sometimes and HM OVE sometimes failed
with a syntax error due to unsupported Java language features,
we note the number of completed entries in parenthesis. For
instance methods in Small Classes, MM- ASSIST achieved 2.4x
to 4x ( Recall MC@3) higher recall compared to baseline tools.
For Small Classes, we noticed that our performance was com-
parable to the synthetic dataset, while for other tools dropped
significantly. Notably, our Recall C@3 was 89%, which can
be attributed to the performance of the LLM in picking the
suitable target classes. However, we observed a performance
degradation in all tools when identifying M OVEMETHOD
opportunities in large classes. This happens because large
classes are more prone to significant technical debt, and there
are many candidate methods that can be moved – thus it is
harder to pick the proverbial “needle from the haystack”.
However, the differences are more nuanced when we evalu-
ate MM- ASSIST on static methods: we find that our Recall C
drops significantly. This is because the scope of moving static
methods is massive - they can be moved to (almost) any class
in the project. For large projects like Elasticsearch, this means
picking the right target class among 21615 candidates. The
real-world oracle contains on average 8743 classes per project.
This shows that recommending which static methods to move

TABLE III: Recall rates on 108 instance methods moved by OSS developers in 2024. First column shows the number of small
or large classes in the oracle. Recall M= identify the method, Recall C= identify the target class for a previously identified
method to be moved, Recall MC= identify the method&target class pair.
Oracle Size ApproachRecall M Recall C Recall MC
@1 @2 @3 @1 @2 @3 @1 @2 @3
SmallClasses (38)JMOVE (19) 5% 5% 5% 0% 0% 0% 0% 0% 0%
FETRUTH 20% 20% 20% 100% 100% 100% 20% 20% 20%
HM OVE (30) 23% 37% 40% 37% 43% 47% 17% 30% 33%
Vanilla-LLM 52%±3 71%±3 83%±3 67%±4 67%±4 67%±4 58%±3 54%±2 58%±3
MM- ASSIST 75%±1 92%±0 95%±0 85%±1 89%±0 89%±0 68%±2 80%±1 80%±1
LargeClasses (70)JMOVE (24) 8% 8% 8% 100% 100% 100% 8% 8% 8%
FETRUTH 2% 8% 12% 76% 76% 76% 2% 6% 9%
HM OVE (55) 9% 15% 20% 15% 20% 24% 4% 5% 5%
Vanilla-LLM 15%±2 23%±2 24%±2 44%±8 44%±8 44%±8 8%±1 8%±1 8%±1
MM- ASSIST 36%±1 38%±1 47%±1 75%±1 80%±1 80±2 30%±1 31%±2 36%±2
TABLE IV: Recall rates on 102 static methods moved by OSS developers in 2024. Recall M= identify the method, Recall C
= identify the target class for a given method, Recall MC= identify the method&target class pair.
Oracle Size ApproachRecall M Recall C Recall MC
@1 @2 @3 @1 @2 @3 @1 @2 @3
SmallClasses (40)FETRUTH 7% 15% 15% 14% 14% 14% 1% 2% 2%
Vanilla-LLM 42%±1 55%±2 64%±1 8%±1 8%±1 8%±1 4%±1 4%±1 4%±1
MM- ASSIST 54%±1 62%±3 69%±1 22%±4 26%±4 26%±4 12%±2 16%±2 18%±2
LargeClasses (62)FETRUTH 6% 11% 15% 6% 6% 6% 0.4% 1% 1%
Vanilla-LLM 14%±1 22%±1 28%±6 3%±5 3%±5 3%±5 0.3%±1 1%±1 1%±2
MM- ASSIST 17%±4 28%±4 32%±4 43%±5 43%±5 47%±5 7%±1 12%±1 15%±1
is a much harder problem than recommending instance meth-
ods, as a tool should analyze thousands of classes to find the
right one. This could explain why prior M OVEMETHOD tools
do not give recommendations for moving static methods. As
we are the first ones to make strides in this harder problem, we
hope that by contributing this dataset of static M OVEMETHOD
to the research community, we stimulate growth in this area.
D. Runtime performance of MM- ASSIST (RQ3)
Experimental Setup. We used both the synthetic and real-
world corpus employed in other RQs to measure the time-
taken for each tool to produce recommendations. To under-
stand what components of MM- ASSIST take the most time,
we also measured the amount of time it took to generate
responses from the LLM, and the time it took to process
suggestions. To ensure real-world applicability, we conducted
these measurements using the MM- ASSIST plugin for IntelliJ
IDEA, mirroring the actual usage scenario for developers. We
conducted all experiments on a commodity laptop, an M1
MacBook Air with 16GB of RAM.
Results. Our empirical evaluation demonstrates that MM-
ASSIST achieves an average runtime of 27.5 seconds for gener-
ating suggestions. The primary computational overhead stems
from the LLM API interactions consuming approximately 9
seconds. In our experience with JM OVE, on the larger projects
in our real-world dataset, JM OVE takes several hours (up to
24 hours) to complete running, thus we imposed the 1-hour
cutoff time. Similarly, HM OVE also takes an average of 80min
to execute on a single entry in our dataset – it needs to be
triggered on all possible <method, target class >pairs (avg.
145 pairs per class). In extreme cases where the host class
was large (>10K LOC), HM OVE took 4 whole days to executeon a single entry in our dataset. Out of the box, FETRUTH is
also slow and can take 12+ hours to run on large projects.
With the help of FETRUTH authors, we were able to run it
on a single class at a time – this takes an average 6 minutes
per class. Thus, compared with the baselines, MM- ASSIST is
two, two, and one orders of magnitude faster than JM OVE,
HM OVE, and FETRUTH , respectively. Thus, it is practical.
E. Usefulness of MM- ASSIST (RQ4)
We designed a user study to assess the practical utility of
MM- ASSIST from a developer’s perspective.
Dataset. We made the deliberate choice to have participants
use projects with which they were familiar. This decision
was grounded in several key considerations. First, familiarity
with their codebases enables them to make more informed
judgments about the appropriateness and potential impact of
the suggested refactorings. Second, using personal projects
enhances the validity of our study, as it closely mimics real-
world scenarios where developers refactor code they have
either authored or maintained extensively. Third, this approach
allows us to capture a diverse range of project types, sizes, and
domains, potentially uncovering insights that might be missed
in a more constrained, standardized dataset.
Experimental Setup. 30 students (25 Master’s and 5 Ph.D.
students) volunteered to participate in our study. Based on
demographic information provided by the participants, 73%
have industrial experience. All participants, with the exception
of two, have experience with the Java programming language.
Finally, the majority of participants (24 out of 30) have prior
experience with refactoring.
We instructed the participants to use MM- ASSIST for a
week and run it on at least ten different Java classes from

projects they work on. For each class they selected, MM-
ASSIST provided up to three M OVEMETHOD recommenda-
tions. We chose to present three recommendations to strike a
balance between variety and practicality. Afterward, they sent
us the fine-grained telemetry data from the plugin usage. For
confidentiality reasons, we anonymize the data by stripping
away any sensitive information about their code. We collected
usage statistics from each invocation of the plugin on each
class. In particular, we collected this information: how the
users rated each individual recommendation and whether they
finally changed their code based on the recommendation.
Participants rated each recommendation on a 6-point Likert
scaled ranging from (1) Very unhelpful to (6) Very helpful. We
chose this 6-point Likert scale to force a non-neutral stance,
encouraging participants to lean towards either a positive or
negative assessment. We asked the participants to rate the
MM- ASSIST ’s recommendations while they were fresh in their
minds, right after they analyzed each recommendation.
After participants sent their usage telemetry, we asked them
to fill out an anonymous survey asking about their experience
using MM- ASSIST . We asked participants to compare MM-
ASSIST ’s workflow against the IDE, and asked for open-ended
feedback about their experience.
Results. 30 participants applied MM- ASSIST on 350
classes. We found that, in 290 classes the participants posi-
tively rated one of the recommendations (82.8% of the time).
Moreover, the users accepted and applied a total of 216
refactoring recommendations on their code, i.e. 7 refactorings
per user, on average. This shows that our tool is effective at
generating useful recommendations that developers, who are
familiar with their code, accept.
The participants also provided feedback in free-form text.
Of the 30 participants, 80% of them rated the plugin’s ex-
perience highly, when comparing against the workflow in the
IDE. In praise of MM- ASSIST , the participants said that MM-
ASSIST gave them a sense of control, allowing them to apply
refactorings that they agreed with.
V. D ISCUSSION
Internal Validity: Dataset bias poses a potential threat to
the effectiveness of MM- ASSIST . To mitigate this, we employ
both a synthetic dataset (widely used by others), offering a
controlled environment, and a real-world dataset comprising
refactorings performed by open-source developers.
External Validity: This concerns the generalization of
our results. Because we rely on a specific LLM (GPT-4o),
it may impact the broader applicability of our findings. We
anticipate that advancements in LLM technology will im-
prove overall performance, though this needs to be verified
empirically. Second, MM- ASSIST currently focuses on Java
code. Although our approach is conceptually language- and
refactoring-agnostic, extending to additional refactoring types
and languages requires adapting three key components: (1)
static analysis for validating refactoring preconditions, (2)
semantic analysis of code relationships, and (3) refactoring ex-
ecution mechanics. Using protocols like the Language ServerProtocol (LSP) [45] can simplify handling language-specific
features, facilitating broader applicability. Future work will
explore the effectiveness of our tool across various languages
and refactorings.
Tool implementation. MM- ASSIST ’s implementation fol-
lows a modular architecture, separating language-specific con-
cerns from the core refactoring workflow. Components such
as the LLM service, embedding model, and IDE integration
communicate via well-defined interfaces, facilitating extensi-
bility and integration across various environments and lan-
guages. To address the non-deterministic nature of LLMs, we
experimented with various temperature settings and found the
variability in the LLM’s outputs to be consistently low, as
evidenced by the small standard deviations reported in our
evaluation. For IDE developers, MM- ASSIST shows the safe
integration of AI-powered suggestions with existing IDEs.
VI. R ELATED WORK
We organize the related work into: (i) research on
MOVEMETHOD , and (ii) usage of LLMs for refactoring.
MOVEMETHOD refactoring. Many researchers focus on
identifying and recommending M OVEMETHOD refactorings.
JMOVE [10], JDeodorant [46], and MethodBook [9] sug-
gest refactorings based on software metrics derived from
static analysis. Additionally, JM OVE introduced a widely-used
synthetically created dataset of M OVEMETHOD refactorings.
HM OVE [30], a recently introduced tool, uses graph neural
networks to classify a M OVEMETHOD suggestion as go/no-
go. Then, HM OVE only uses LLM as a judge to filter
suggestions that don’t meet certain preconditions. Similarly,
FETRUTH [16], RMove [13], and PathMove [35], utilize DL
techniques to identify M OVEMETHOD opportunities.
Most importantly, MM- ASSIST attacks the problem in a
drastically different way. Previous tools compute whole project
dependencies (which is computationally expensive and doesn’t
scale) and then produce a confidence score for each method
in the project. Thus, they treat this as a classification , not a
recommendation problem: they produce 57 recommendations
on average to move out a given class (many of which are
unuseful), which puts tremendous analysis burden on the
programmer. In contrast, MM- ASSIST offers up to 3 rec-
ommendations per class, aligned with how expert developers
refactor code.
Refactoring in the age of LLMs. A recent systematic
study [47] analyzing 395 research papers demonstrates that
LLMs are being employed to solve various software engineer-
ing tasks. While code generation has been the predominant
application, recently LLMs like ChatGPT have been applied
to automate code refactoring [30, 48–51] and detect code
smells [52]. Cui et al. [53] leverage intra-class dependency
hypergraphs with LLMs to perform extract class refactoring,
while iSMELL [54] uses LLMs to detect code smells and
suggest corresponding refactorings. However, LLMs are prone
to hallucinate, which can introduce incorrect or broken code,
posing challenges for automated refactoring systems. Unlike
other approaches, MM- ASSIST addresses this limitation by

validating and ranking LLM-generated outputs, ensuring that
developers can safely execute refactoring recommendations.
The prevalence of hallucinations in LLM-based refactoring
is widely studied. Pomian et al. [39, 55] investigated hallucina-
tions in E XTRACT METHOD refactoring, while [40] analyzed
hallucinations in Python code modifications. These studies
consistently show that LLMs can hallucinate during refactor-
ing tasks, substantiating our findings, where LLMs halluci-
nated in 80% of the cases when suggesting M OVEMETHOD .
This highlights the necessity of robust validation mechanisms,
which are integral to our MM- ASSIST , ensuring the reliability
and safety of the suggestions generated by LLMs.
VII. C ONCLUSION
Despite years of research in M OVEMETHOD refactoring,
progress has been incremental. The rise of LLMs has revital-
ized the field. Our approach and tool, MM- ASSIST , signifi-
cantly outperforms previous best-in-class tools and provides
recommendations that better align with the practices of expert
developers. When replicating refactorings from recent open-
source projects, MM- ASSIST achieves 4x higher recall while
running 10x–100x faster. Additionally, in a one-week case
study, 30 experienced developers rated 82.8% of MM- ASSIST
’s recommendations positively.
The key to unleashing these breakthroughs is combining
static and semantic analysis to (i) eliminate LLM hallucina-
tions and (ii) focus its laser. MM- ASSIST checks refactoring
preconditions automatically which cuts down the LLM hal-
lucinations. By leveraging semantic embedding into a RAG
approach, MM- ASSIST narrows down the context for the LLM
so that it can focus its laser on a small number of high-
quality prospects. This was instrumental in picking the right
candidate from industrial large scale projects. We hope that
these techniques inspire others to solve many other refactoring
recommendation domains such as splitting large classes or
packages.
ACKNOWLEDGEMENTS
We are grateful for the constructive feedback from the
members of the AI Agents team at JetBrains Research. This
research was partially funded through the NSF grants CNS-
1941898, CNS-2213763, 2512857, 2512858, the Industry-
University Cooperative Research Center on Pervasive Person-
alized Intelligence, and a gift grant from NEC.
REFERENCES
[1] M. Fowler, Refactoring: Improving the Design of Existing
Code , 1999.
[2] N. Tsantalis and A. Chatzigeorgiou, “Identification of
Move Method Refactoring Opportunities,” TSE, 2009.
[3] G. Bavota, A. De Lucia, and R. Oliveto, “Identifying
Extract Class refactoring opportunities using structural
and semantic cohesion measures,” JSS, 2011.
[4] Y . Lin, X. Peng, Y . Cai, D. Dig, D. Zheng, and W. Zhao,
“Interactive and guided architectural refactoring with
search-based recommendation,” in FSE, 2016.[5] S. Negara, N. Chen, M. Vakilian, R. E. Johnson, and
D. Dig, “A Comparative Study of Manual and Automated
Refactorings,” in ECOOP , 2013.
[6] E. Murphy-Hill, C. Parnin, and A. P. Black, “How We
Refactor, and How We Know It,” TSE, 2012.
[7] N. Tsantalis, A. Ketkar, and D. Dig, “RefactoringMiner
2.0,” TSE, 2022.
[8] W. F. Opdyke, “Refactoring: A Program Restructuring
Aid in Designing Object-Oriented Application Frame-
works,” Ph.D. dissertation, University of Illinois at
Urbana-Champaign, 1992.
[9] G. Bavota, R. Oliveto, M. Gethers, D. Poshyvanyk,
and A. De Lucia, “Methodbook: Recommending Move
Method Refactorings via Relational Topic Models,” TSE,
2014.
[10] R. Terra, M. T. Valente, S. Miranda, and V . Sales,
“JMove: A novel heuristic and tool to detect move
method refactoring opportunities,” JSS, 2018.
[11] T. Bryksin, E. Novozhilov, and A. Shpilman, “Automatic
recommendation of move method refactorings using clus-
tering ensembles,” in IWoR , 2018.
[12] Z. Kurbatova, I. Veselov, Y . Golubev, and T. Bryksin,
“Recommendation of Move Method Refactoring Using
Path-Based Representation of Code,” in ICSEW , 2020.
[13] D. Cui, S. Wang, Y . Luo, X. Li, J. Dai, L. Wang, and
Q. Li, “RMove: Recommending Move Method Refactor-
ing Opportunities using Structural and Semantic Repre-
sentations of Code,” in ICSME , 2022.
[14] H. Liu, Z. Xu, and Y . Zou, “Deep learning based feature
envy detection,” in ASE, 2018.
[15] H. Liu, J. Jin, Z. Xu, Y . Zou, Y . Bu, and L. Zhang, “Deep
Learning Based Code Smell Detection,” TSE, 2021.
[16] B. Liu, H. Liu, G. Li, N. Niu, Z. Xu, Y . Wang, Y . Xia,
Y . Zhang, and Y . Jiang, “Deep Learning Based Feature
Envy Detection Boosted by Real-World Examples,” in
ESEC/FSE , 2023.
[17] J. Ivers, A. Ghammam, K. Gaaloul, I. Ozkaya,
M. Kessentini, and W. Aljedaani, “Mind the Gap: The
Disconnect Between Refactoring Criteria Used in Indus-
try and Refactoring Recommendation Tools,” in ICSME ,
2024.
[18] A. C. Bibiano, W. K. G. Assunção, D. Coutinho,
K. Santos, V . Soares, R. Gheyi, A. Garcia, B. Fonseca,
M. Ribeiro, D. Oliveira, C. Barbosa, J. L. Marques,
and A. Oliveira, “Look Ahead! Revealing Complete
Composite Refactorings and their Smelliness Effects,” in
ICSME , 2021.
[19] L. P. Antonio Mastropaolo, Emad Aghajani and
G. Bavota, “Automated variable renaming: are we there
yet?” in EMSE , 2023.
[20] J. Pantiuchina, B. Lin, F. Zampetti, M. Di Penta,
M. Lanza, and G. Bavota, “Why Do Developers Reject
Refactorings in Open-Source Projects?” TOSEM , 2021.
[21] D. Silva, N. Tsantalis, and M. T. Valente, “Why we
refactor? confessions of GitHub contributors,” in FSE,
2016.

[22] M. Dilhara, A. Ketkar, and D. Dig, “Understanding
Software-2.0: A Study of Machine Learning Library
Usage and Evolution,” TOSEM , 2021.
[23] M. Dilhara, A. Ketkar, N. Sannidhi, and D. Dig, “Dis-
covering repetitive code changes in Python ML systems,”
inICSE , 2022.
[24] M. Dilhara, D. Dig, and A. Ketkar, “PYEVOLVE: Au-
tomating Frequent Code Changes in Python ML Sys-
tems,” in ICSE , 2023.
[25] S. Fakhoury, D. Roy, A. Hassan, and V . Arnaoudova,
“Improving source code readability: Theory and prac-
tice,” in ICPC , 2019.
[26] S. Scalabrino, G. Bavota, C. Vendome, M. Linares-
Vasquez, D. Poshyvanyk, and R. Oliveto, “Automatically
assessing code understandability,” TSE, 2019.
[27] S. Tworkowski, K. Staniszewski, M. a. Pacek, Y . Wu,
H. Michalewski, and P. Mił o ´s, “Focused Transformer:
Contrastive Training for Context Scaling,” in NeurIPS ,
2023.
[28] “Needle In A Haystack - Pressure Testing LLMs,” https:
//github.com/gkamradt/LLMTest_NeedleInAHaystack.
[29] V oyageAI, “V oyage AI Embeddings,” https://docs.
voyageai.com/docs/embeddings.
[30] D. Cui, J. Wang, Q. Wang, P. Ji, M. Qiao, Y . Zhao, J. Hu,
L. Wang, and Q. Li, “Three Heads Are Better Than One:
Suggesting Move Method Refactoring Opportunities with
Inter-class Code Entity Dependency Enhanced Hybrid
Hypergraph Neural Network,” in ASE, 2024.
[31] Elasticsearch, “Move Method Refactoring in the Elastic-
search Project,” https://github.com/elastic/elasticsearch/
commit/876e70159c01ae306251281ae2fdbabca8732ed9.
[32] “OpenAI,” https://openai.com/.
[33] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua,
F. Petroni, and P. Liang, “Lost in the Middle: How
Language Models Use Long Contexts,” TACL , 2023.
[34] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin,
N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel
et al. , “Retrieval-augmented generation for knowledge-
intensive nlp tasks,” NeurIPS , 2020.
[35] Z. Kurbatova, I. Veselov, Y . Golubev, and T. Bryksin,
“Recommendation of Move Method Refactoring Using
Path-Based Representation of Code,” in ICSEW , 2020.
[36] D. Oliveira, W. K. G. Assunção, A. Garcia, A. C. Bib-
iano, M. Ribeiro, R. Gheyi, and B. Fonseca, “The untold
story of code refactoring customizations in practice,” in
ICSE, 2023 .
[37] N. Tsantalis, M. Mansouri, L. M. Eshkevari, D. Mazi-
nanian, and D. Dig, “Accurate and Efficient Refactoring
Detection in Commit History,” in ICSE , 2018.
[38] O. Leandro, R. Gheyi, L. Teixeira, M. Ribeiro, and
A. Garcia, “A Technique to Test Refactoring Detection
Tools,” in SBES , 2022.
[39] D. Pomian, A. Bellur, M. Dilhara, Z. Kurbatova, E. Bo-
gomolov, T. Bryksin, and D. Dig, “Next-GenerationRefactoring: Combining LLM Insights and IDE Capa-
bilities for Extract Method,” in ICSME , 2024.
[40] M. Dilhara, A. Bellur, T. Bryksin, and D. Dig, “Unprece-
dented Code Change Automation: The Fusion of LLMs
and Transformation by Example,” in FSE, 2024.
[41] W.-L. Chiang, L. Zheng, Y . Sheng, A. N. Angelopoulos,
T. Li, D. Li, H. Zhang, B. Zhu, M. Jordan, J. E. Gonzalez,
and I. Stoica, “Chatbot arena: An open platform for
evaluating LLMs by human preference,” 2024.
[42] “Chatbot Arena Leaderboard,” 2024, https://huggingface.
co/spaces/lmsys/chatbot-arena-leaderboard.
[43] Cursor, “Cursor,” https://www.cursor.com/, accessed:
2024-10-07.
[44] Github, “Copilot,” https://github.com/features/copilot, ac-
cessed: 2024-10-07.
[45] “Language server protocol (lsp),” 2024, https://github.
com/python-rope/pylsp-rope.
[46] N. Tsantalis, T. Chaikalis, and A. Chatzigeorgiou, “Ten
years of JDeodorant: Lessons learned from the hunt for
smells,” in SANER , 2018.
[47] X. Hou, Y . Zhao, Y . Liu, Z. Yang, K. Wang, L. Li,
X. Luo, D. Lo, J. Grundy, and H. Wang, “Large Lan-
guage Models for Software Engineering: A Systematic
Literature Review,” TOSEM , 2024.
[48] A. Shirafuji, Y . Oda, J. Suzuki, M. Morishita, and
Y . Watanobe, “Refactoring programs using large lan-
guage models with few-shot examples,” in APSEC , 2023.
[49] K. DePalma, I. Miminoshvili, C. Henselder, K. Moss, and
E. A. AlOmar, “Exploring ChatGPT’s code refactoring
capabilities: An empirical study,” Expert Systems with
Applications , 2024.
[50] E. A. AlOmar, A. Venkatakrishnan, M. W. Mkaouer,
C. Newman, and A. Ouni, “How to refactor this code?
An exploratory study on developer-ChatGPT refactoring
conversations,” in MSR , 2024.
[51] B. Liu, Y . Jiang, Y . Zhang, N. Niu, G. Li, and H. Liu,
“Exploring the potential of general purpose LLMs in
automated software refactoring: an empirical study,”
ASE(J) , 2025.
[52] L. L. Silva, J. R. d. Silva, J. E. Montandon, M. Andrade,
and M. T. Valente, “Detecting Code Smells using Chat-
GPT: Initial Insights,” in ESEM , 2024.
[53] D. Cui, Q. Wang, Y . Zhao, J. Wang, M. Wei, J. Hu,
L. Wang, and Q. Li, “One-to-One or One-to-Many?
Suggesting Extract Class Refactoring Opportunities with
Intra-class Dependency Hypergraph Neural Network,” in
ISSTA , 2024.
[54] D. Wu, F. Mu, L. Shi, Z. Guo, K. Liu, W. Zhuang,
Y . Zhong, and L. Zhang, “iSMELL: Assembling LLMs
with Expert Toolsets for Code Smell Detection and
Refactoring,” in ASE, 2024.
[55] D. Pomian, A. Bellur, M. Dilhara, Z. Kurbatova, E. Bo-
gomolov, A. Sokolov, T. Bryksin, and D. Dig, “EM-
Assist: Safe Automated ExtractMethod Refactoring with
LLMs,” in FSE, 2024.
# Agent-Based Test Assertion Generation via Diverse Perspective Aggregation

**Authors**: Dong Wang, Qiaoyu Han, Lin Yang, Jianyi Zhou, Guangtai Liang, Junjie Chen

**Published**: 2026-08-06 09:51:43

**PDF URL**: [https://arxiv.org/pdf/2608.05822v1](https://arxiv.org/pdf/2608.05822v1)

## Abstract
Test assertions are critical elements of unit tests, serving as checkpoints to validate expected behavior and ensure software correctness. Numerous techniques have been proposed to automate assertion generation, with recent progress notably driven by large language models (LLMs). Despite the promise, existing approaches such as ChatAssert suffer from modest accuracy, heavy reliance on oversampling, and vulnerability to model randomness due to one-shot prompting. To address these limitations, we propose AssertMate, a novel agent-based assertion generation framework that enhances the quality and reliability of LLM-generated assertions through three key components: (1) actual value construction that identifies assertion targets via static analysis and type-aware heuristics; (2) multi-perspective expected value prediction using code generation, retrieval-augmented generation (RAG), and chain-of-thought (CoT) reasoning agents; and (3) an LLM-as-a-Judge collaboration mechanism to select the most appropriate assertion. Evaluation on the Defects4J benchmark demonstrates that AssertMate significantly outperforms state-of-the-art techniques in compilation success and pass rates, along with substantially higher bug detection capabilities. Integration with EvoSuite further validates AssertMate's practicality, yielding superior mutation coverage and kill counts. Ablation studies reveal that each of the three components makes a significant and complementary contribution to the overall performance. This work affirms the great potential of aggregating diverse perspectives to enhance the effectiveness of LLM-based assertion generation.

## Full Text


<!-- PDF content starts -->

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING 1
Agent-Based Test Assertion Generation via Diverse
Perspective Aggregation
Dong Wang, Qiaoyu Han, Lin Yang, Jianyi Zhou, Guangtai Liang, and Junjie Chen*
Abstract—Test assertions are critical elements of unit tests,
serving as checkpoints to validate expected behavior and ensure
software correctness. Numerous techniques have been proposed
to automate assertion generation, with recent progress notably
driven by large language models (LLMs). Despite the promise,
existing approaches such as ChatAssert suffer from modest
accuracy, heavy reliance on oversampling, and vulnerability to
model randomness due to one-shot prompting. To address these
limitations, we proposeAssertMate, a novel agent-based assertion
generation framework that enhances the quality and reliability
of LLM-generated assertions through three key components: (1)
actual value construction that identifies assertion targets via static
analysis and type-aware heuristics; (2) multi-perspective expected
value prediction using code generation, retrieval-augmented gen-
eration (RAG), and chain-of-thought (CoT) reasoning agents;
and (3) an LLM-as-a-Judge collaboration mechanism to select
the most appropriate assertion. Evaluation on the Defects4J
benchmark demonstrates thatAssertMatesignificantly outper-
forms state-of-the-art techniques in compilation success and pass
rates, along with substantially higher bug detection capabilities.
Integration with EvoSuite further validatesAssertMate’s prac-
ticality, yielding superior mutation coverage and kill counts.
Ablation studies reveal that each of the three components makes
a significant and complementary contribution to the overall
performance. This work affirms the great potential of aggregating
diverse perspectives to enhance the effectiveness of LLM-based
assertion generation.
Index Terms—Test Assertion Generation, Static Program Anal-
ysis, Large Language Model
I. INTRODUCTION
UNIT testing plays a pivotal role in software quality as-
surance by verifying functional correctness and ensuring
that implementations align with specified requirements [1–10].
A complete unit test consists of two essential components:
a test prefix, which constructs the execution context and
exercises the focal method, and one or more assertions, which
serve as test oracles by specifying the expected behavior.
While recent progress in automated LLM-based test generation
has substantially improved the generation of executable test
prefixes, constructing correct and effective assertions remains
a fundamental challenge. This is because assertions ultimately
determine whether the observed program behavior conforms
to its expected behavior; consequently, even tests with high
code coverage may fail to reveal defects when their asser-
tions are weak, incomplete, or semantically incorrect, further
Dong Wang, Qiaoyu Han, Lin Yang, and Junjie Chen are with the College
of Intelligence and Computing, Tianjin University, Tianjin, China.
E-mail:{dong w, hanqiaoyu, linyang, junjiechen}@tju.edu.cn
Jianyi Zhou and Guangtai Liang are with Huawei Cloud Computing
Technologies Co., Ltd., Beijing, China. E-mail:{zhoujianyi2, liangguang-
tai}@huawei.com
Corresponding author: Junjie Chen.limiting their fault-detection capability in practice. Therefore,
test assertion generation has become an important standalone
research problem, focusing on the construction of high-quality
test oracles regardless of how test prefixes are obtained.
To automate the assertion generation, prior research has
introduced various types of automated assertion generation
techniques, including traditional techniques such as Sym-
bolic Execution [11–16] and Specification Mining [17–19].
While being effective in certain contexts, these traditional
techniques suffer from notable limitations, particularly the
path explosion [20] and the lack of structured documentation
in many real-world projects [21]. Recent advancements in
deep learning (DL) and large language models (LLMs) have
significantly improved automated assertion generation [1, 4, 6–
8]. By leveraging their strong code comprehension capabilities,
these models often outperform traditional techniques. Typi-
cally, these approaches take as input a focal method (i.e., the
method under test) and a test prefix (i.e., a test body without
assertions), and use either DL models or LLMs to generate
assertions. Among them, ChatAssert [1] stands as the state-of-
the-art LLM-based assertion generation technique, utilizing the
power of ChatGPT. Nevertheless, while ChatAssert advances
beyond earlier techniques, it still faces critical limitations
and presents opportunities for further improvement. Its top-1
accuracy remains modest at 0.45, with only a slight increase
to 0.54 in top-10 accuracy, even after applying multi-round
post-processing to iteratively fix compilation and runtime
errors. This first highlights substantial uncertainty in gen-
erating accurate assertions on the initial attempt. Moreover,
ChatAssert’s heavy reliance on over-sampling (generating up
to 10 assertions to cover a likely correct one) exposes the
inefficiency and search space under-exploration of single-
prompt inference, which also increases the burden of manual
validation in practice. The 9-percentage-point gain further
reflects its susceptibility to LLM randomness, limiting its
reliability in contexts that demand deterministic and precise
test oracles.
These limitations align with findings from recent work [22],
which observed that one-shot LLM queries often yield over-
confident or erratic outputs, resulting in stubborn or inconsis-
tent feedback during self-evaluation. The study further sug-
gests that incorporating diverse problem-solving perspectives
provides a more robust foundation for iterative refinement
and can significantly improve output quality. Inspired by this
insight, we theorize that adopting a multi-perspective prompt-
ing strategy holds promise for enhancing the effectiveness
and reliability of LLM-based assertion generation. Meanwhile,
it brings several unique challenges in this context.First,
arXiv:2608.05822v1  [cs.SE]  6 Aug 2026

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING 2
due to the under-specified nature of unit tests, the method
for verifying the focal methods’ intended behavior is often
ambiguous and lacks explicit contextual cues. LLMs may ran-
domly interpret the test target differently per perspective-based
query, leading to inconsistent or inaccurate assertions.Second,
a multi-prompting strategy relies on selecting and combining
diverse reasoning approaches, with each agent potentially
emphasizing different aspects of the code. An inappropriate
choice or combination of strategies may introduce noise or
redundancy, undermining the complementarity from multiple
perspectives, thus reducing the quality of generated assertions.
Third, when multiple agents generate candidate assertions,
selecting the most accurate and semantically aligned one
becomes a non-trivial task.
In this paper, we proposeAssertMate, a novel agent-
based assertion generation framework that effectively lever-
ages LLMs for producing high-quality test assertions.Assert-
Mateis designed with three core components, each tailored
to address key limitations of prior work and challenges intro-
duced by multi-perspective prompting:
•Accurate Identification of Assertion Targets.Given a
focal method and its test prefix,AssertMatefirst performs
static analysis, assisted by LLMs, to identify a set of test
targets, such as return values and modified attributes. It then
constructs the corresponding actual values (assertion targets)
using a rule-based and type-aware strategy. This process
effectively clarifies “what to test” and narrows the task
to expected value prediction, thereby reducing ambiguity
and providing a solid foundation for the subsequent multi-
perspective assertion generation.
•Expected Value Generation from Diverse Perspectives.
To address the overconfidence and bias inherent in single-
prompt strategies,AssertMateadopts a multi-perspective
reasoning approach for expected-value prediction (i.e., infer-
ring the desired outcomes corresponding to the constructed
actual values). Here, a perspective refers to a distinct
reasoning strategy characterized by its information source
and inference mechanism. Specifically,AssertMateem-
ploys three complementary perspectives: code-generation-
based reasoning, which infers expected values directly from
program semantics; retrieval-augmented reasoning (RAG),
which grounds prediction on semantically similar examples;
and chain-of-thought (CoT) reasoning, which derives ex-
pected values through explicit intermediate reasoning.
To further improve reliability,AssertMateaggregates the
candidate assertions generated from these complementary
perspectives through a probability-based reranking mecha-
nism that prioritizes the most promising candidates, thereby
reducing the search space. The remaining candidates are
then passed to the subsequent collaboration stage for final
selection, effectively answering the question of “what to
expect.”
•“LLM-as-a-Judge” Collaboration Mechanism.To en-
hance decision-making capabilities for selecting high-
quality assertions,AssertMateintroduces a Judge agent
based on the “LLM-as-a-Judge” [23] paradigm. This agent
conducts an in-depth comparative analysis of candidateassertions, leveraging a CoT reasoning strategy to evaluate
their semantic correctness, coherence with the test context,
and alignment with the actual values, ultimately selecting
the most appropriate assertion.
To summarize,AssertMate’s novelty lies in a structured shift
from monolithic assertion generation to a decomposed and
software-engineered workflow, built upon two key design
contributions:
(I) Task Decoupling with Domain-Specific Optimization.
By decomposing assertion generation into structured subtasks,
such as actual value derivation through program analysis,As-
sertMatereduces LLM hallucinations and improves reliability.
(II) Synergistic Agent Design.AssertMatecombines
complementary perspectives, actual-value construction, and
expected-value prediction through modular components. It first
uses confidence-based reranking to select one candidate from
each perspective and then uses the Judge agent for cross-
perspective selection, addressing the “last-mile” challenge
without manual top-kinspection.
We evaluatedAssertMateon the Defects4J [24] benchmark
and compared its effectiveness against four widely adopted or
state-of-the-art baseline techniques across both DL-based and
language model-based approaches:EditAS,TOGLL,RetriGen,
andChatAssert∗(without its repair module). Across five inde-
pendent runs,AssertMateachieves mean Compilation Success
Rate (CSR) and Pass Rate (PR) values of 76.78% and 60.09%,
respectively; the corresponding sample standard deviations
(SDs) are 0.90 and 1.07 percentage points. These results
outperform the strongest baseline,ChatAssert∗, by 52.92%
and 50.94% in relative terms, respectively. Its mean Bug
Detection Rate (BDR) is 46.25%, and the corresponding SD is
1.75 percentage points. In comparison,ChatAssert∗achieves
a mean BDR of 37.32% with an SD of 1.82 percentage
points. After averaging the five per-bug outcomes, a paired
Wilcoxon signed-rank test indicates a significant improvement
overChatAssert∗(p= 0.0489,delta= 0.152).
Furthermore, we examined the practical utility of integrating
AssertMatewith Evosuite [9]. Results show that the asser-
tions generated byAssertMateconsistently yield higher CSRs
and achieve the highest number of covered and killed mu-
tants among all compared techniques. Its killed-mutant count
improves by 40.49%–492.76% overChatAssert∗,TOGLL,
andEditAS. Ablation studies further validate the substantial
contribution ofAssertMate’s three core components: actual
value constructor, multi-perspective expected value generation,
and the collaboration mechanism. Notably, the actual value
constructor not only mitigates hallucinations and guides LLM
reasoning during assertion generation but also enhances exist-
ing LLM-based techniques when incorporated.
Contributions.The key contributions of this paper are:
•We propose a novel agent-based assertion generation frame-
work to enhance assertion quality, with a full replication
package publicly released for reproducibility [25].
•Experimental results comprehensively demonstrate thatAs-
sertMateexcels at generating syntactically correct and se-
mantically rich assertions, consistently delivering superior
bug-detection performance compared to all baselines.

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING 3
(a) Focal Method(b) Test Prefix
Fig. 1: An illustrative example of focal method and test prefix
•Evaluation ofAssertMate’s integration with EvoSuite af-
firms its practicality and scalability in real-world testing
scenarios.
II. BACKGROUND ANDRELATEDWORK
In this section, we introduce the concept of fine-grained
assertion terminology and review related work on automated
test assertion generation techniques.
A. Fine-Grained Assertion Terminology
Generally, assertion generation techniques take the focal
method (FM) and a test prefix (TP) as input, and generate
one or more assertions for theFM−TPpair [7]. Figure 1(a)
and (b) illustrate the focal methodonClearAllPushedand
its corresponding test prefixtestOnClearAllPushed,
respectively. TheonClearAllPushedmethod defines the
behavior when the clear-all button is pressed: it clears all
the content in thescreenfield and updates thestateof
clearAllBtnto indicate activation (i.e.,false). In the
test prefix, thetestOnClearAllPushedmethod sets up
the testing scenario by instantiating aCalculatorobject
and invoking the focal method.
Since the focal method performs two actions (clearing
the screen and activating the button), comprehensive testing
requires two corresponding assertions: one to verify the up-
datedstateand another to confirm the clearedscreen.
Moreover, the form of assertions may vary depending on the
testing approach. For instance, to verify that the screen has
been cleared, one may useassertEqualsto compare the
screen content with a literal string value “0”, or alternatively,
useassertTrueto validate the equivalence relationship,
i.e.,content == "0". Existing work often treats assertion
generation as a unified task, which leads to an excessively large
search space and insufficient guidance during exploration,
particularly for advanced LLM-based approaches. As a conse-
quence, these methods become highly susceptible to the inher-
ent randomness of LLMs, ultimately undermining their effec-
tiveness. For example, the state-of-the-art method ChatAssert
prompts ChatGPT to generate a single assertion per input. In
doing so, the model must implicitly decide both what to test
and what to expect in one step. Given the wide range of possi-
ble test targets and the diverse ways to validate them, generat-
ing a correct assertion on the first attempt becomes extremely
difficult, especially when limited to one-shot inference. The
top-1 response generated by ChatAssert for Figure 1 is
assertFalse(calc.getClearAllBtn().getState
());which verifies the activation state of the clear-all buttonbut fails to test whether thescreencontent has been cleared,
leaving a key aspect of the behavior unverified.
To mitigate the challenge of undirected exploration in such a
vast space, we propose a formal decomposition of the assertion
into three distinct components:Assertion Function,Expected
Value, andActual Value. This fine-grained task partitioning
provides a more structured framework for guiding the assertion
generation process, enabling more targeted reasoning and
reducing the impact of stochastic variability. Figure 2 presents
an example assertion formulation using the JUnit framework.
Related assertion terminologies are defined below:
assertEquals (     ”0”    ,      calc     .getScreen().getContent());Assertion FunctionExpected ValueTest TargetResolution PathActual ValueassertTrue (     calc     .getScreen().getContent().isEmpty()  );Expected Value &Assertion FunctionTest TargetResolution PathActual Value(a) Junit Assertion with three components(b) Another Semantically Equivalent Assertion
Fig. 2: A fine-grained JUnit assertion for the illustration
Assertion Functionis the assertion method used in
the test case, which defines the expected relationship be-
tween the actual value and the expected value. For in-
stance,assertEqualsis suitable for verifying whether the
screenhas been correctly set to “0”.
Actual Valueis the target element being evaluated in
the assertion. It can take various forms, including variables,
method invocations, or combinations of both. To more pre-
cisely describe its composition, we define the actual value
as consisting of two components: theTest Targetand the
Resolution Path. The test target refers to a variable (e.g.,
calc) defined within the test prefix, or a field defined in the
test class. The resolution path represents a method invocation
or a sequence of invocations applied to the test target, such as
the.getScreen().getContent()method in Figure 2,
which specifies the extent to which the test target is inspected.
Expected Valuerepresents the desired outcome of the actual
value in an assertion. Unlike the actual value, which may have
a complex structure, we define expected values in an atomic
form. Importantly, the expected value is not always a concrete
data value; it can also involve selecting the appropriate asser-
tion function that best expresses the intended condition, such
asassertTruefor the equivalence relationship.
B. Test Assertion Generation Techniques
As test oracles remain one of the most critical challenges in
software testing, assertion generation has been widely studied
in recent years [1, 4, 6–8, 21]. Existing techniques can be
categorized into three groups: traditional, deep learning-based,
and large language model-based approaches. Traditional tech-
niques are typically divided into two major types: Symbolic
Execution [11–16] and Specification Mining [17–19]. The
former explores program paths to derive logical conditions
for assertions, while the latter extracts expected behaviors
from documentation to guide assertion construction. To ad-
dress coverage, scalability, and adaptability challenges faced
by traditional techniques, deep learning-based and language
model-based techniques have gained increasing attention.
Deep Learning-Based Techniques.ATLAS [6] is one of the
earliest studies to apply deep learning to assertion generation.

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING 4
It takes the focal method and test prefix as input and uses a
recurrent neural network to generate assertions. TOGA [21]
formulates assertion generation as a multi-class classification
problem. It first constructs a dictionary of common values
and types, then generates candidate assertions, and finally
uses an Assertion Oracle Ranker to select the most suitable
one. Integrating information retrieval (IR) algorithm with deep
learning models [4, 7] takes the assertions as documents, and
the inputs (i.e., focal method and test prefix) as queries, and
uses lexical similarity (e.g., Jaccard Similarity) as the metric
for searching. Yu et al. [7] were among the first to propose
combining IR with deep learning for this task. They used IR to
retrieve assertions similar to those required for the input and
then modified the retrieved assertions to better match the target
context. EditAS [4] further advanced this idea by explicitly
modeling the differences between the retrieved and actual
input pairs. It learns both edit operation sequences and token-
level transformations to enhance the quality and accuracy of
generated assertions.
DL models have significantly advanced assertion generation
by leveraging their capabilities in code understanding and
generation. However, these techniques require large amounts
of labeled data for training and are prone to overfitting.
Additionally, IR-based approaches are constrained by the
similarity between retrieved and target assertions, limiting their
adaptability to diverse code contexts.
Language Model-Based Techniques.Language models,
trained on large-scale code corpora, have demonstrated strong
generalizability across a wide range of code-related tasks [2,
5, 26, 27]. TOGLL [8] extends TOGA by addressing gener-
alizability challenges through fine-tuning a larger code model
and demonstrates strong performance, particularly in mutation
killing. [28] proposed RetriGen, a hybrid approach that
combines assertion retrieval with a pre-trained language model
(PLM)-based assertion generator to improve assertion quality.
While LLMs have proven effective in generating human-
readable unit tests [2], their potential in assertion generation
remains relatively underexplored. Hayet et al. [1] introduced
ChatAssert, which guides LLMs using code summaries and
one-shot examples to generate test assertions, followed by an
“execute-and-fix” mechanism to correct invalid outputs.
Although ChatAssert demonstrates the potential of LLM-
based approaches over prior work, it relies on a single prompt-
ing strategy and is heavily dependent on runtime execution for
post-hoc correction, which limits its reliability and scalability
in practical settings. In contrast, this study proposes an agent-
based framework that integrates multiple prompting strategies
in a structured and verifiable manner. By decoupling key tasks
(such as actual value construction and assertion formulation),
and avoiding reliance on execution-based repair, our approach
aims to improve both the accuracy and robustness of assertion
generation in unit testing scenarios.
C. LLM-Based Unit Test Generation
Recent advances in LLMs have significantly accelerated
end-to-end unit test generation, where LLMs generate com-
plete test cases including test setup, method invocations, andassertions. Representative approaches such as ChatTester [2]
and ChatUniTest [3] have demonstrated impressive capabilities
in automatically constructing executable unit tests with high
coverage. These studies highlight the potential of LLMs to
automate broader software testing workflows by jointly syn-
thesizing test inputs and test oracles. Nevertheless, complete
unit test generation and test assertion generation address
different challenges. End-to-end unit test generation primarily
focuses on constructing executable tests that effectively exer-
cise program behaviors, whereas assertion generation focuses
on constructing precise test oracles that correctly validate
those behaviors. High code coverage alone does not guar-
antee effective fault detection if the generated assertions are
weak, redundant, or semantically incorrect. Consequently, the
quality of assertions often becomes the bottleneck in the
effectiveness of automatically generated tests. Therefore, we
view these two research directions as complementary rather
than competing. Assertion generation techniques can naturally
serve as an oracle enhancement module for tests produced
by traditional automated testing tools (e.g., EvoSuite and
Randoop) as well as modern LLM-based unit test generation
frameworks. Following this complementary view, AssertMate
focuses specifically on oracle construction, with the goal of
improving the correctness and fault-detection capability of
automatically generated unit tests regardless of how the test
prefixes are obtained.
III. METHODOLOGY
Figure 3 presents an overview ofAssertMate. As shown,
AssertMateis composed of two main modules: theActual
Value Constructor (ActVCon) and theMulti-Perspective
Expected Value Generator (EVGen). Given a focal method
and its test prefix, ActVCon first identifies a set of actual
values and corresponding assertion functions, using a hybrid
rule-based approach that combines static program analysis
with LLM assistance. Subsequently, EVGen first generates
the matching expected values for each actual value identified
by ActVCon though multi-perspective inference. Then, EV-
Gen elects the best-fitting output assertion through a multi-
agent collaboration mechanism. Each assertion is formed by
combining the assertion function, actual value, and predicted
expected value. Each generated assertion can be independently
appended to the test prefix to produce a complete test case. To
make the workflow ofAssertMatemore intuitive, we use the
illustrative example shown in Figure 1 as a running example
throughout the remainder of this section.
A. Actual Value Constructor
To avoid overly-large search space when adopting multi-
perspective inference, and also to mitigate the impact brought
by randomness, it is necessary to first specify concrete and
valid actual values as the target of multi-perspective reasoning.
Therefore, ActVCon is designed to identify a set of valid
actual values as a pre-processing step before multi-perspective
reasoning. Specifically, ActVCon implements a hybrid rule-
based solution that combines static program analysis with
LLM assistant to systematically identify the actual values that

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING 5
Focal 
Method
Project Analysis
 Test Target
Actual
Value
Assertion
Function
Test
Prefix
Judge
 Assertion
Expected
Value
Actual Value Constructor
Multi-Perspective 
Expected Value Generator
Rerank
Actual Value Generate
Consensus ？
CGA
CoTA
RAGA
Collaborative Judge 
Agent
Fig. 3: Overview of the agent-based test assertion generation
frameworkAssertMate
should be tested, thereby addressing the fundamental challenge
of determining “what to test.”
ActVCon begins by selecting appropriate test targets for
assertions through static program analysis. Specifically, it
identifies two basic test targets as a starter for the iterative
traversing approach: (1)Return value of the focal method,
which is verified to ensure the method produces the expected
output when it returns an object or value; and (2)Attributes
defined in the focal class that have public getter methods, since
these attributes reflect side effects of the method’s execution
and are verified to confirm that the intended modifications have
been correctly applied.
Next, ActVCon constructs actual values through a type-
aware iterative traversal process, supplemented by LLM as-
sistance to handle external types. Given a set of candidate
actual values, ActVCon applies a rule-based iterative analysis
to generate the final actual values. The specific rules are
designed based on the types of candidate actual values, en-
abling accurate and context-sensitive construction of assertion
expressions. We define the following three rules:
•Primitive data typesrefer to a set of Java’s basic built-in
types (e.g.,boolean) as well as commonly used wrapper
classes such asStringandInteger, since their val-
ues can be directly compared. For these types, ActVCon
includes the actual value in the final set without further
decomposition or traversal.
•In-Project typesdenote types defined within the project,
such asButtonandScreenin the motivating example.
For these types, ActVCon expands the candidate actual
value set by including attributes that have parameterless
getter methods. Specifically, a getter method invocation is
attached to the end of the resolution path and forms a new
candidate value.
•External typesrefer to types that are not defined within the
project, such as those from third-party libraries or the Java
Collections Framework. Due to the lack of sufficient code
context (e.g., missing class definitions), ActVCon leverages
an LLM to assist in generating final actual values for these
types. Specifically, ActVCon queries the LLM to determine
how to properly test a given actual value, especially for the
resolution path.
Based on the types of final actual values, ActVConfurther determines the assertion functions to be used. In
this study, we focus on five common types of test as-
sertions:assertEquals,assertTrue,assertFalse,
assertNull, andassertNotNull. These five types ac-
count for up to 86% of all assertions, according to the empir-
ical study conducted by Yu et al. [7], based on data collected
from open-source repositories. Specifically, for boolean values,
ActVCon selectsassertTrueorassertFalseas the
appropriate assertion functions. For other primitive data types,
it usesassertEquals. If a test target may benull, for ex-
ample, when the focal method may returnnullunder certain
conditions, ActVCon includes an additionalassertNullor
assertNotNullassertion accordingly. For types classified
as external types, ActVCon consults the LLM to determine the
appropriate assertion function based on the type and context
of the actual value.
Illustrative Example.Using the focal method and
test prefix from Figure 1 as an example, ActVCon
first identifies two attributes (i.e.,screenand
clearAllBtn) defined in theCalculatorclass as
valid targets, and addscalculator.getScreen()
andcalculator.getClearAllBtn()to the initial
candidate actual value set. Since types (Screenand
Button) are defined within the same project, ActVCon
classifies them as In-Project types, inspects their class
definitions, and further identifiesgetContent()and
getState()as valid getter methods with a type of
String and boolean, respectively. These are both considered
primitive or directly comparable types, so the iteration
terminates. Finally, ActVCon constructs two actual values:
calculator.getScreen().getContent()and
calculator.getClearAllBtn().getState(), with
typesStringandboolean, respectively. The output of
ActVCon for this case is:
assertEquals(<exp>,calculator.getScreen().getContent
());
assert<exp>(calculator.getClearAllBtn().getState());
where<exp>is the placeholder for expected value.
B. Multi-Perspective Expected Value Generator
AssertMateemploys a Multi-Perspective Expected Value
Generator (EVGen) to address “what to expect” and generate
the corresponding expected values for test assertions. EVGen
first introduces multiple prompting strategies, each acting
as a distinct agent, to improve the comprehensiveness of
expected value prediction. These agents independently gen-
erate candidate expected values based on the given context.
Finally, EVGen adopts a multi-agent collaboration mechanism
to collectively select the most appropriate expected value.
Multi-Perspective Agent Design.AssertMateleverages a
multi-agent design to provide diverse perspectives for expected
value generation, thereby mitigating the overconfidence and
bias associated with relying on a single prompt. To achieve
this, we surveyed prompting strategies used in recent work on
assertion and unit test generation [1–3, 5], and identified three
major categories of prompt design. Based on this analysis, we
implemented three specialized assertion generation agents: the

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING 6
Code Generation Agent, the Retrieval-Augmented Generation
(RAG) Agent, and the Chain-of-Thought (CoT) Agent.
•Code Generation Agent (CGA).Given that assertion
generation is a sub-task of code generation and LLMs
have demonstrated strong capabilities in this domain, the
Code Generation Agent (CGA) leverages this strength by
prompting the model to complete the test case with an
appropriate assertion for a given actual value. This strategy
offers a valuable perspective grounded in code structure and
semantic consistency.
•RAG Agent (RAGA).IR-based assertion generation has
shown promising results [4, 7], and Retrieval-Augmented
Generation (RAG) is a well-established in-context learning
technique in the LLM literature. Building on the Jaccard
similarity-based retrieval algorithm used by EditAS [4],
we design a RAG Agent (RAGA) for assertion generation.
Specifically, the RAGA first retrieves the test case with the
highest Jaccard similarity to the input test prefix, and then
uses the retrieved case as a one-shot example to guide the
LLM in generating an assertion for the given actual value.
This approach enables RAGA to effectively incorporate
external knowledge and provides a unique perspective by
supplying contextual reference for assertion generation.
•CoT Agent (CoTA). Instructing LLMs to reason through
the functionality of a given focal method has proven ef-
fective in prior work [1, 2]. Building on this idea, we
design a CoT Agent (CoTA) that applies the Chain-of-
Thought (CoT) prompting strategy to break down complex
inference tasks and support assertion generation through
step-by-step reasoning. Specifically, CoTA operates in three
main stages: 1) focal method understanding, 2) test scenario
understanding, and 3) expected value prediction. Finally,
CoTA synthesizes a test assertion by querying the LLM
to produce a statement consistent with the reasoning chain
accumulated throughout the session. CoTA offers a unique
perspective by summarizing the intended behavior of the
focal method and the test design in the test prefix, thereby
potentially improving assertion quality through deeper con-
textual understanding.
Prompt Prefill.In addition to designing effective agents,
another major challenge lies in analyzing and constraining the
responses of LLMs. More recently, Yang et al. [5] introduced
a prefill mode for unit test generation. By prepending a
predefined header to the LLM’s response, this mode reduces
the likelihood of unexpected outputs and imposes structural
constraints, such as restricting accessible classes, which helps
improve the compilability of the generated code. Inspired
by this,AssertMatealso incorporates a prefill mechanism
in each agent. Specifically, when ActVCon determines a
specific assertion function (e.g.,assertEquals),Assert-
Matepre-attaches the corresponding function signature (e.g.,
assertEquals) at the beginning of the LLM’s response
to guide generation. In cases where ActVCon identifies mul-
tiple candidate assertion functions (e.g.,assertTrueor
assertFalse),AssertMateattaches a more generic prefix
such asassertto leave room for model inference while
maintaining structural control. For example, Figure 4 presents
Token6,FieldUtil…Avg.Prob0.970.930.930.95…0.93Fig. 4: Example of LLMs’ response tokens and probabilities
an illustrative case of an LLM-generated response. In this
example, the prefilled content “I think the assertion should
be:\n‘‘‘java\nassertEquals(” is prepended to the
LLM’s response, and each agent in EVGen begins generating
from the position immediately following the left parenthesis.
Probability-Based Response Re-Rank. To enhance the
comprehensiveness of assertion generation, EVGen adopts an
oversampling strategy. Following ChatAssert [1], each agent
generates five responses per input (i.e., FM-TP pair and actual
value), resulting in 15 total candidates (3 components × 5
responses). Our choice of five candidates aligns with prior
work [1, 6, 29], where performance gains plateau beyond this
point. Empirical results show notable improvements from top-
1 to top-5, with diminishing returns from top-5 to top-10,
balancing cost-effectiveness and effectiveness. However, this
strategy introduces the challenge of confidently selecting the
best candidate from a larger pool.
To address this,AssertMatedevises a probability-based re-
ranking algorithm to prioritize responses with higher model
confidence for each agent. In LLMs, each token in the re-
sponse is sampled from the vocabulary, and its associated
probability reflects the model’s confidence in appending that
token to the input. Since LLM-generated responses often
contain both the expected output (i.e., the assertion) and an
accompanying explanation, incorporating the probabilities of
explanatory tokens could bias the confidence estimation. To
mitigate this,AssertMateevaluates only the probabilities of
tokens within the assertion portion of the response. Figure 4
illustrates an example of an LLM-generated response along
with token probabilities. As shown, the response contains
two parts, the assertion itself (i.e.,assertEquals(6,
FieldUtils.safeMultiply(-2L,-3));) and a short
explanation“The assertion is testing the method... ”. In this
case,AssertMatecomputes the average probability (i.e., 0.93
in the “Avg.”) and uses this value as the confidence score for
the response. For each agent, the response with the highest
confidence score is selected and forwarded to the subsequent
collaboration stage.
C. Collaborative Judge Agent
After obtaining the top-ranked response from each agent,
the next task is to determine which response best aligns with
the given test prefix. To accomplish this,AssertMateadopts a
multi-agent collaboration strategy to select the final expected
value. When the multi-perspective expected value generators
do not reach a consensus on how the assertion should be writ-
ten,AssertMatedelegates the decision to a designated agent,
referred to asJudge, which conducts an in-depth analysis of
the candidate responses and selects the best-fitting one.Judge

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING 7
employs a CoT reasoning strategy, as recommended in [23],
to enhance its decision-making capabilities. Additionally, it
utilizes the prefill mode to ensure structured and consistent
evaluation of agent responses. As an illustrative example,
AssertMatefinally generates two assertions for the motivating
case in Figure 1 by predicting expected values based on actual
values provided by ActVCon:
assertEquals( "0",calculator.getScreen().getContent()
);
assert False(calculator.getClearAllBtn().getState());
All prompt designs used inAssertMateare publicly avail-
able as part of our replication package [25] to facilitate
reproducibility and future research.
IV. EXPERIMENTALSETUP
To evaluate the effectiveness ofAssertMate, we formulate
the following three research questions to guide our study:
•RQ1:How doesAssertMateperform compared to exist-
ing approaches?
•RQ2:How doesAssertMateperform when incorporated
with Evosuite?
•RQ3:How much contribution do ActVCon and EVGen
owe to the effectiveness ofAssertMate?
A. Dataset Preparation
In this study, we utilize the Defects4J benchmark
(v2.1.0) [24] for evaluation. Although other datasets are
available for assertion generation [6, 8, 21, 30, 31], none
of them, unlike Defects4J, support evaluation of bug detec-
tion capability, which is essential for assessing the practical
effectiveness and quality of generated assertions. Defects4J
includes 834 real-world bugs and has been widely adopted in
prior studies [5, 8, 32, 33]. Each bug is accompanied by a
buggy version, a fixed version, and a set of bug-triggering test
cases, enabling comprehensive and reproducible evaluation.
Additionally, Defects4J provides several off-the-shelf tools,
such as EvoSuite and Randoop, for automated unit test genera-
tion, further supporting the experimental process. Specifically,
we construct the following two sets of inputs, i.e., pairs of
focal methods (FM) and test prefixes (TP), based on the bugs
in Defects4J to evaluate the performance ofAssertMate.
Human-Written FM-TP pairs.To evaluate the correctness
of generated assertions, we first utilized the human-written test
cases provided in Defects4J. These test cases offer a reliable
foundation for assessing both pass rates and bug detection
capabilities.
We first select bugs whose triggering tests fail at assertion
statements. This criterion is based on the failure location
during test execution, rather than the underlying fault type;
therefore, we do not filter bugs based on whether they involve
exceptions or other specific fault categories. Exception-related
bugs are retained if the corresponding behavior is verified
through an assertion and the triggering test successfully
reaches that assertion. We exclude only cases where execution
terminates before reaching the target assertion (e.g., due to an
uncaught exception), because a generated assertion cannot be
evaluated without an executable assertion context. Followingthis criterion, we obtain 112 eligible bugs for constructing
FM–TP pairs.
Next, we identify the focal methods associated with the
selected bugs by combining patch-level and test-level evi-
dence. For each bug, we extract the methods modified by
the bug-fixing patch and the methods invoked by the trig-
gering test. When these two sets overlap, the corresponding
method is selected as the focal method for constructing an
FM–TP pair. Cases without such a reliable correspondence
are excluded because both AssertMate and existing assertion-
generation baselines require a focal method as input. This
matching strategy avoids unreliable focal-method inference
from complex test cases while leveraging the explicit root-
cause information provided by bug-fixing patches. To ensure
practical relevance, we statistically analyzed the complexity
of the selected assertion-triggering bugs. The results show
that, on average, the selected bugs span 35 lines of code,
involve 68 method calls, and have an average nesting depth
of 9. In cases where a single test method contains multiple
assertions, we extract all assertion statements via program
analysis and construct a distinct test case for each one. Each
synthesized test case contains all preceding code statements
(including prior assertions) up to the current assertion, en-
suring that necessary variable states are preserved. Therefore,
our construction does not assume that the oracle corresponds
only to the final assertion of a test case; assertions appearing
at intermediate positions are also included as independent
targets. The final assertion in each synthesized test case is then
determined as the target test assertion, forming an FM–TP pair
for training and evaluation. As a result, we obtained a total of
667 FM–TP pairs.
Evosuite-Generated FM-TP pairs.Following prior work [8,
21], we integratedAssertMatewith EvoSuite to demonstrate
its practical applicability in our evaluation. Specifically, we se-
lected focal methods based on the bug-fixing patches for each
bug in Defects4J and used EvoSuite to generate corresponding
test cases. The test generation process followed EvoSuite’s
default configuration and relied on its built-in coverage criteria
as the primary guidance. Following previous research [21, 34],
we addressed EvoSuite randomness by allocating 10 minutes
per target and using coverage-guided generation to maximize
test case quality. To construct valid test prefixes for focal
methods, we removed all assertions from the generated test
cases. In total, we obtained 2,203 Evosuite-generated FM-TP
pairs for evaluation.
B. Baselines
We selected four representative types of state-of-the-art
assertion generation techniques for comparison, covering DL-
based and language model-based approaches.
EditASis an information retrieval (IR)-based assertion
generation technique. It retrieves a similar FM-TP pair and
learns from the edit sequence between the retrieved pair
and the input, modifying the retrieved assertion accordingly.
EditASleverages an attention-based LSTM encoder-decoder
architecture [35] to perform the assertion generation process.
TOGLLis a pre-trained language (PLM)-based asser-
tion generation technique. It fine-tunes the CodeGen-350M

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING 8
model [36] on the SF110 dataset to perform assertion gen-
eration.
RetriGenis an enhanced retrieval-augmented assertion gen-
eration technique that uses a hybrid assertion retriever, which
considers both lexical and semantic similarity to identify
the most relevant test-assert pair from external codebases.
This hybrid retriever combines a token-based retriever with
an embedding-based retriever. The retrieved assertion is then
combined with the original focal test and used as input to fine-
tune a PLM-based assertion generator, implemented on top of
the CodeT5 model.
ChatAssertis an LLM-based assertion generation technique.
It first prompts ChatGPT to generate a detailed code summa-
rization of the test prefix. In addition, it retrieves a one-shot
example based on semantic similarity using UniXCoder [37]
to enhance in-context learning. Finally, it asks ChatGPT to
generate an assertion for the given test prefix and applies
an “execute-and-fix” post-processing step to resolve potential
issues (e.g., compilation errors). In this study, we focus
specifically on assertion generation in the context of unit
testing, where generating assertions without assuming function
correctness is essential. To ensure fairness, we disabled the
repair module in the original baseline, as its reliance on “bug-
free functions” fundamentally contradicts the assumptions of
unit testing. The adapted version is referred to asChatAssert∗.
C. Metrics
Previous research mainly relied on metrics such as Exact
Match, BLEU, and CodeBLEU to evaluate the effectiveness
of assertion generation [1, 4, 6–8, 21]. However, we argue that
these metrics are insufficient for fully capturing the correctness
and practical utility. On the one hand, Exact Match fails to
account for semantically equivalent assertions that differ syn-
tactically from the ground truth but exhibit identical behavior.
On the other hand, minor textual differences may have a
limited impact on BLEU or CodeBLEU scores, yet could lead
to serious issues such as invalid syntax or compilation failures.
To address these limitations, our evaluation primarily adopts
six comprehensive metrics,Compilation Success Rate, Pass
Rate, Bug Detection Rate, Number of Killed Mutants,Token
Overhead, andAverage Time.
Our evaluation incorporates two complementary metrics to
assess assertion quality from both syntactic and semantic per-
spectives. First, since producing compilable code constitutes
the fundamental capability of any code generation system,
we propose theCompilation Success Rate (CSR)metric to
quantify syntactic correctness. This aligns with prior studies
that employed the Exact Match metric as a proxy for syntax-
level validity, which inherently depends on correct syntax. For
semantic validation, we introduce thePass Rate (PR)metric to
evaluate functional correctness while mitigating false negatives
arising from semantically equivalent assertions. To avoid con-
flating assertion quality with assertion quantity, we evaluate
CSR and PR at the assertion-candidate level rather than at the
test-case level. For each FM-TP pair, ifAssertMateconstructs
Nactual values and therefore generatesNassertions, we give
each baseline the same assertion budget by duplicating thesame FM-TP inputNtimes and asking the baseline to generate
Nassertions. Each generated assertion is then independently
appended to the corresponding test prefix and wrapped as a
separate single-assertion test case. In other words, an FM-TP
pair with an assertion budget ofNproducesNindependent
test cases for each technique, each containing the same test
prefix and one generated assertion. The metrics are formally
defined as:CSR=Ncompiled
Ntotal,PR=Npassed
Ntotal, whereN compiled
denotes the number of test cases that compile successfully,
Npassed represents the number of test cases that execute
successfully, andN total is the total number of independently
constructed single-assertion test cases under the normalized
assertion budget.
We employed a suite of metrics to systematically evaluate
the bug-revealing capability of the generated assertions. The
primary metric,Bug Detection Rate (BDR), quantifies the
effectiveness in uncovering real-world bugs from the De-
fects4J benchmark. It is defined as the ratio of detected bugs
(Bdetected ) to the total number of applicable bugs (B total), using
the combination of generated assertions and bug-triggering
test prefixes. To assess capability against synthetic defects,
we adopted mutation testing metrics following established
practice [8]:#Covered Mutantscounts mutants executed
by synthesized test cases,#Killed Mutantstallies mutants
detected through assertion failures, andKill Rate(Killed
Covered)
measures the detection efficiency. This triangulated evaluation
strategy provides a comprehensive understanding of assertion
quality across both real-world and artificially injected bugs.
When comparing LLM-based approaches, we measured
computational efficiency through two key dimensions as pro-
posed in [38].Token Overheadreflects resource consumption
via the average token count (including both prompt and
response) per query, whileAverage Timecaptures temporal
cost as the mean processing duration per FM-TP pair. These
metrics collectively assess practical deployment feasibility in
contemporary AI-powered testing pipelines.
D. Implementation and Environment
Studied LLM.ForAssertMate, we usedDeepSeek-Coder-
7B-Instruct (DC-7B)[39] as the backbone model. This
choice was motivated by two main considerations: (1) prior
research[5] has shown that DC-7B offers a favorable trade-
off between cost and performance in unit test generation;
and (2) we aim to demonstrate thatAssertMatecan more
effectively leverage the capabilities of LLMs for assertion
generation, even when using a smaller-scale model. Regarding
the LLM inference configuration, we set the temperature to
1.0 to encourage response diversity, which aligns with our
oversampling strategy for generating complementary candidate
assertions. We set top-p to 1.0 to avoid additional probability
truncation and set the maximum token limit to 4,000 to
accommodate sufficiently long generations.
Baseline Replication.For the baseline techniques, we either
replicated them using their publicly available implementations
or re-implemented them strictly following the descriptions
provided in their papers. Specifically, forTOGLL, we followed
the original design using CodeGen-350M [36] with the P5

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING 9
TABLE I: Performance of Studied Assertion Generation Tech-
niques on Human-Written test prefixes (RQ1)
TechniqueCSR (%) PR (%) BDR (%) Token Overhead Avg. Time (s)
Mean SD Mean SD Mean SD Mean SD Mean SD
EditAS18.06 0.24 17.71 0.27 0.00 0.00 – – 0.37 0.01
TOGLL12.05 0.66 11.83 0.73 0.00 0.00 – – 0.81 0.03
RetriGen21.59 0.23 6.43 0.25 0.00 0.00 – – 0.23 0.01
ChatAssert∗50.21 1.83 39.81 2.04 37.32 1.82 1,609.67 103.72 13.27 0.43
AssertMate76.780.9060.091.0746.251.75 5,284.70 117.36 15.31 0.83
input format, i.e., (TP+ [sep] +FM) forTOGLL. To ensure
a fair comparison, we used the original training and testing
datasets provided by the authors of bothEditASandTOGLL.
We first reproduced their reported results to validate our setup
and then retained their original hyperparameter configurations
for evaluation. ForRetriGen, we used the publicly available
fine-tuned model parameters and replicated the experimental
setup from the original paper. In this setup, the retrieved
assertion is selected from the training set and combined with
the focal method and test prefix to form the augmented input
for generation [28]. ForChatAssert∗, we used the official
implementation released by the authors [40], but excluded the
post-hoc repair component to align with the focus of this study.
Environment.We implemented our pipeline using tree-
sitter [41] for static program analysis, using PyTorch 2.1.0 [42]
and Transformers 4.44 [43] to build the LLM runtime en-
vironment. Furthermore, we used the VLLM library [44] to
accelerate model inference. All experiments were conducted
on Ubuntu 20.04 LTS, equipped with an Intel Xeon Gold
6248R CPU, 512 GB RAM, and four NVIDIA A100 GPUs.
V. RESULTS
In this section, we present our three research questions along
with their corresponding analysis processes and results.
A. RQ1: Effectiveness comparison among studied techniques
Process:To answer RQ1, which compares the effectiveness
ofAssertMatewith baseline techniques, we used theHuman-
Written FM-TP pairs(described in Section IV-A) as inputs
and applied each studied technique to generate corresponding
assertions. The generated assertions were then appended to the
test prefixes to form complete test cases. We evaluated these
test cases using the Defects4J benchmark and reported three
key effectiveness metrics:Compilation Success Rate (CSR),
Pass Rate (PR), andBug Detection Rate (BDR). Additionally,
to assess computational efficiency, we measured and reported
theToken OverheadandAverage Timefor each technique.
To mitigate stochasticity in LLM operations and improve
result reliability, we independently ranAssertMateand every
baseline five times with the same configuration and reported
the per-technique mean and sample standard deviation (SD) for
each metric. To assess the robustness of the observed improve-
ments, we used Wilcoxon signed-rank tests [45] on paired
outcomes after averaging the five repetitions. We additionally
reported Cliff’s delta (δ) [46] as a measure of effect size. The
evaluation results are summarized in Table I.
Results:Across five independent runs,AssertMateachieves
mean CSR, PR, and BDR values of 76.78%, 60.09%, andTABLE II: Performance of studied assertion generation tech-
niques on Evosuite-generated test prefixes (RQ2)
Technique CSR (%)# Covered
Mutants# Killed
MutantsKill
Rate (%)
EditAS65.15 17,883 2,071 11.58
TOGLL70.95 29,224 5,374 18.39
TOGLL DN*1.21 138 9 6.52
RetriGen35.54 973 19620.14
ChatAssert∗79.08 63,665 8,738 13.72
EvoSuite100.0056,661 8,184 14.44
AssertMate83.7276,840 12,27615.98
*TOGLL DN refers to the variant ofTOGLLretrained on the unseen dataset.
46.25%, respectively, with corresponding SDs of 0.90, 1.07,
and 1.75 percentage points, demonstrating stable performance
across runs. Overall,AssertMateconsistently outperforms
the studied deep-learning-based and LLM-based approaches.
First, in terms of syntactic correctness,AssertMateachieves a
52.92% relative improvement in CSR over the best-performing
baseline,ChatAssert∗(50.21%). This substantial gain high-
lights the effectiveness of combining static program anal-
ysis with LLMs via ActVCon. Second, for semantic cor-
rectness,AssertMatedelivers a 50.94% relative gain in PR
overChatAssert∗(39.81%) likely due to the multi-agent
collaboration strategy that integrates diverse prompting styles
for improved semantic reasoning. Compared with the other
approaches,AssertMateachieves 3.39×, 5.08×, and 9.35×
higher PR ofEditAS(17.71%),TOGLL(11.83%), andRetri-
Gen(6.43%), respectively. Third, in bug detection,AssertMate
attains the highest BDR on the Defects4J benchmark, com-
pared with 37.32% forChatAssert∗, whileEditAS,TOGLL,
andRetriGenfailed to detect any bugs, yielding a BDR of
0%.
The statistical testing results further support the effective-
ness ofAssertMateover all baselines. Using the Wilcoxon
signed-rank test with Cliff’s delta effect size analysis, we
observe statistically significant improvements for all pair-
wise comparisons in terms of CSR, PR, and BDR. For
the comparison betweenAssertMateand the best-performing
ChatAssert∗,AssertMateachieves a significant improvement
in CSR (p= 2.974×10−29,δ= 0.377, medium effect),
PR (p= 9.383×10−22,δ= 0.302, small effect) and
BDR (p= 0.0489,δ= 0.152, small effect). These results
demonstrate that the performance gains ofAssertMateare
statistically significant across the evaluated baselines.
Efficiency analysis of computational costs reveals nuanced
insights intoAssertMate’s operational trade-offs. While our
approach requires 3.28× more tokens per generation than
ChatAssert∗(5,284.70 vs. 1,609.67 tokens), primarily due to
multi-prompt integration and quality-oriented oversampling,
this is offset by three key advantages: (1) The open-source
DC-7B model outperforms ChatGPT in CSR (76.78% vs.
50.21%), showing that targeted architecture can enable small-
scale models to exceed larger ones;
(2) Running 7B models is more cost-effective and practical
than proprietary models like ChatGPT; (3) Multi-agent paral-
lelism keeps generation time comparable to ChatAssert (15.31s
vs. 13.27s per generation), especially considering AssertMate’s
superior effectiveness. This balanced cost-performance profile

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING 10
makesAssertMatea compelling option for assertion generation
in safety-critical or resource-constrained environments.
RQ1 Summary:AssertMatedemonstrates strong as-
sertion generation performance, achieving mean CSR,
PR, and BDR values of 76.78%, 60.09%, and 46.25%,
respectively. The improvements over the baselines are
statistically significant, corresponding to relative gains
of52.92%–537.18%in CSR and50.94%–834.53%
in PR. While consuming 3.28×more tokens than
the state-of-the-art LLM-based approachChatAssert∗,
AssertMatemaintains a practical trade-off with a con-
trolled time and efficient performance using a small-
scale open-source model.
B. RQ2: Integration with Evosuite
Process:Following prior work [8], we integrated the studied
assertion generation techniques with EvoSuite to evaluate
their practical utility. Specifically, we used theEvoSuite-
generated FM-TP pairs(described in Section IV-A) as inputs
and followed the same procedure as in RQ1 to construct
complete test suites. We then employed the mutation testing
toolkit provided by Defects4J to assess the effectiveness of
each technique in killing mutants. Our evaluation focused on
Compilation Success Rate (CSR), the number of covered and
killed mutants, and the kill rate. We did not report Pass Rate
or Bug Detection Rate due to the lack of valid ground-truth
expected outputs in the generated test cases, which makes
distinguishing false positives infeasible in the unit testing
scenario. The results are presented in Table II.
Results:Evaluation results demonstrate thatAssertMaterep-
resents an advancement in assertion generation integrated
with EvoSuite, substantially outperforming baselines across
both syntactic and semantic dimensions. In terms of syntactic
validity,AssertMateachieves an 83.72% CSR, surpassingEdi-
tAS,TOGLL,RetriGen, andChatAssert∗by 28.50%, 18.00%,
135.57%, and 5.87%, respectively. Semantically,AssertMate
exhibits outstanding assertion quality, killing 12,276 mutants,
a 492.76%, 128.43%, and 40.49% improvement overEdi-
tAS,TOGLL, andChatAssert∗, respectively. WhileRetriGen
achieves the highest kill rate, it covers the fewest mutants and
kills far fewer mutants than the other compared techniques.
These results together underscoreAssertMate’s ability to gen-
erate precise expected values, which are essential for detecting
subtle faults in complex codebases.
When compared to the classic generation tool EvoSuite,
its extremely high CSR (100%) is expected, as EvoSuite’s
evolutionary process inherently ensures that the generated
tests compile successfully on the fixed program versions. In
contrast, language model-based generators, including Assert-
Mate, naturally face risks of syntactic hallucination, which
can potentially reduce CSR. Nonetheless, AssertMate achieves
the highest CSR among the studied assertion-generation tech-
niques, effectively suppressing hallucinations relative to other
baselines such as ChatAssert*. More importantly, AssertMate
achieves substantially higher mutant coverage (76,840 vs.TABLE III: Performance of studied variants (RQ3)
MetricSingle Agent Collaboration ActVConAssertMateCGA RAGA CoTA Debate V oting AssertMate−ChatAssert∗+
CSR 52.01 61.46 70.74 73.53 75.32 12.46 51.24 76.78
PR 43.96 41.79 46.59 53.25 44.73 10.27 52.17 60.09
BDR 38.39 35.71 41.07 45.53 37.50 0.00 48.21 46.25
56,661), kill counts (12,276 vs. 8,184), and kill rate (15.98%
vs. 14.44%), demonstrating stronger fault-detection capability
than EvoSuite.
Despite these advantages, our evaluation also reveals areas
for refinement. WhileAssertMateleads in the absolute number
of killed mutants, its kill rate (15.98%) suggests room for
improving the precision of generated assertions, especially
in capturing subtle behavioral variations. We first note that
overall, language model-based assertion generation approaches
outperform the DL-based one (i.e., 11.58% achieved byEd-
itAS). Among the language model-based baselines,RetriGen
achieved the highest kill rate at 20.14%; however, it covered
the fewest mutants, with only 973 detected. On the other
hand,TOGLLshows a kill rate (18.39%), which may also
superficially suggest superior assertion quality compared with
AssertMate. However, our follow-up investigation uncovered a
potential leakage issue:TOGLLwas originally trained on the
SF110 dataset [47], which includes EvoSuite-generated test
cases. This overlap with the evaluation setup likely inflated
its performance. To test this hypothesis, we retrainedTOGLL
usingData newdataset collected by Yu et al [7] and observed a
sharp decline in performance, with the kill rate dropping from
18.39% to 6.52% (as shown in Table II, lineTOGLL DN). This
confirms the susceptibility of data-driven PLM-based methods
to training–test distribution alignment and highlights a key
limitation where real-world assertion generation tasks rarely
have access to aligned test data during training. Together,
these findings indicate that existing approaches relying on fine-
tuning exhibit limited generalizability.
RQ2 Summary:AssertMatedemonstrates strong prac-
ticality and superiority when integrated with EvoSuite,
achieving substantially higher Compilation Success
Rate (CSR) as well as greater numbers of covered
and killed mutants. For example,AssertMateattains
an assertion-level CSR of 83.72%, outperforming the
studied assertion-generation approaches by margins
ranging from 5.87% to 135.57%. Meanwhile, it high-
lights the chance to improve assertion specificity and
generalizability.
C. RQ3: Ablation Study
Process:In this RQ, we further conducted an ablation study
to investigate the extent to which each major component
contributes to the effectiveness ofAssertMate. Three sets of
variants are designed:
1ActVCon Variants:We assessed the impact of the actual
value constructor (ActVCon) through two complementary
experiments. First, we removed ActVCon fromAssert-
Mate, denoted asAssertMate−, to evaluate its direct

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING 11
contribution toAssertMate’s overall effectiveness. Sec-
ond, we integrated ActVCon intoChatAssert∗, referred
to asChatAssert∗+, to demonstrate the generalizability
of ActVCon.
2EVGen Variants:One key strength ofAssertMatelies in
its multi-perspective expected value generator (EVGen),
which integrates diverse prompting strategies for expected
value generation. To evaluate the contribution of this
design, we compared the effectiveness of each individ-
ual agent, Code Generation Agent (CGA), Retrieval-
Augmented Generation Agent (RAGA), and Chain-of-
Thought Agent (CoTA), against the fullAssertMatever-
sion.
3Collaboration Strategies:In this set of variants, inspired
by prior research [48, 49], we investigated the effective-
ness of another two representative agent collaboration
strategies:DebateandVoting. Debate involves iterative
rounds of interaction where agents revise their responses
based on peer input until a consensus is reached. V oting
is one of the most commonly used and fundamental
decision-making strategies in multi-agent discussion sce-
narios and in this study, we used it as the naive strategy
for comparison.
We usedHuman-Written FM-TP pairsas inputs and eval-
uated each variant’s performance using Compilation Success
Rate (CSR), Pass Rate (PR), and Bug Detection Rate (BDR).
The results are presented in Table III.
Results:The dual capabilities of ActVCon (reasoning guid-
ance and hallucination mitigation) are key to its effectiveness
in promoting LLM-based assertion generation. By leveraging
static program analysis to construct accurate actual values,
ActVCon reduces hallucinated outputs, as evidenced by the
64.32-percentage-point drop in CSR and complete loss of
bug detection when ActVCon is removed (AssertMate−) com-
pared to the fullAssertMate. Additionally, ActVCon enhances
reasoning quality by anchoring assertion generation to con-
crete program states, leading to a PR improvement from
10.27% inAssertMate−to 60.09% inAssertMate. Its adapt-
ability is further confirmed by integrating it intoChatAssert∗:
ChatAssert∗+outperforms the originalChatAssert∗with rel-
ative gains of 2.05%, 31.05%, and 29.18% in CSR, PR, and
BDR, respectively. These results position ActVCon as both an
effective hallucination safeguard and a transferable reasoning
scaffold for LLM-based assertion generation techniques.
The EVGen framework outperforms individual prompting
agents (CGA, RAGA, and CoTA), as demonstrated byAs-
sertMate’s consistent superiority across all evaluation metrics.
While each prompting strategy captures distinct facets of
assertion generation, their isolated use limits holistic coverage.
By enabling collaboration among specialized agents, EVGen
integrates complementary insights and facilitates mutual vali-
dation. This coordinated design leads to relative improvements
of 8.54% in CSR, 28.98% in PR, and 12.61% in BDR over the
best-performing standalone prompting strategy, demonstrat-
ing that structured multi-agent collaboration more effectively
aligns generated assertions with both syntactic correctness and
semantic intent.The “LLM-as-a-Judge” serves as an effective collaboration
mechanism, striking a balance between efficiency and effec-
tiveness. In contrast, the Debate mechanism, despite yielding
a moderate bug detection rate (45.53% BDR), suffers from
the cascading effect of hallucinations across multi-round in-
teractions (73.53% CSR) [49]. These extended dialogues also
incur prohibitive computational costs, consuming 3.7× more
tokens per assertion thanAssertMate. While voting achieves
reasonable syntactic correctness (75.32% CSR), manual in-
spection revealed that it inevitably discards 12.5% of bugs
detectable only by individual agents using specific prompting
strategies (4 by CG, 3 by RAG, and 7 by CoT). This inherent
limitation of V oting accounts for its 8.75-percentage-point
drop in BDR compared toAssertMate.AssertMateovercomes
these limitations through single-round expert mediation, where
aJudgeagent preserves valuable minority perspectives while
mitigating hallucination risks. By combining consensus valida-
tion with expert appraisal,AssertMateachieves superior bug
detection (46.25% BDR) without sacrificing implementation
stability.
Beyond comparisons with individual agents, Debate, and
V oting, we further evaluate the selection effectiveness of
theJudgethrough a paired override analysis against CoTA,
the strongest individual agent in Table III. Specifically, we
compare the outcomes of the assertions selected by theJudge
and CoTA for the same input instance. Acorrect override
occurs when the CoTA-generated assertion fails but theJudge-
selected assertion succeeds, whereas anincorrect override
occurs when the opposite happens. TheJudgeachieves 103
correct overrides and 44 incorrect overrides for CSR, and
129 correct overrides and 40 incorrect overrides for PR; both
differences are statistically significant according to McNemar’s
test (p <0.001) [50]. For BDR, theJudgeachieves 13 correct
overrides and 4 incorrect overrides, also showing a statistically
significant difference (p <0.05). These results demonstrate
that theJudgemore frequently improves upon the strongest
individual agent than incorrectly replacing a better candidate.
RQ3 Summary: Both ActVCon and EVGen are criti-
cal toAssertMate’s effectiveness. ActVCon not only
mitigates hallucinations and guides LLM reasoning
during assertion generation, but also enhances existing
LLM-based techniques when integrated. Additionally,
the “LLM-as-a-Judge” strategy proves to be a more
effective collaboration mechanism compared to debate
and voting, striking a balance between effectiveness
and efficiency.
VI. DISCUSSION
In this section, we qualitatively demonstrate the effective-
ness ofAssertMatethrough a representative case, examine its
performance with more advanced LLMs, discuss the com-
plementarity and boundaries of its multi-agent design, and
highlight the threats to validity of our study.

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING 12
A. Case Study of Test Assertion Quality
Our evaluation (RQ1 and RQ2) confirms the effectiveness of
the proposedAssertMatein comparison to existing baselines,
particularly the second-best performing LLM-based approach
ChatAssert∗. To gain deeper insights into their behavioral
differences, we conducted an in-depth case study on a rep-
resentative test instance from our dataset, as illustrated in
Figure 5, focusing on a scenario involving bound parameter
estimation within an optimization problem.
(a) Focal Method and Test Prefix
(b) Test Assertions generated byAssertMateandChatAssert∗
Fig. 5: A representative case study of bound parameter esti-
mation
Case Description.As shown in the figure, the method under
test is a Gauss–Newton estimator for nonlinear least squares
optimization, and the test constructs an estimation problem
with two free parameters and one bound parameter. The
human-written ground truth is:
Assert.assertTrue(estimator.getRMS(problem) < 1.0e
-10);
which verifies that the optimizer achieves a sufficiently small
residual, indicating overall convergence of the estimation
process.
Outputs ofChatAssert∗andAssertMate.To ensure an
equal assertion budget, we invokedChatAssert∗three times
for this case, matching the three assertions generated by
AssertMate. However, all three generations yielded the same
assertion shown in Figure 5, which verifies that the bound
parameter remains fixed at its initial value (2.0), capturing
constraint enforcement but not overall convergence. On the
other hand,AssertMatenot only reproduced the parameter-
invariance assertion but also generated additional semantically
relevant ones. This is because ActVCon detected additional
internal attributes of the estimator class as potential actual
values through static program analysis, enabling richer asser-
tion generation. Moreover, if we manually add public utility
methods (such as getRMS()) of the target class into the input,
our approach is also able to synthesize assertions like:
Assert.assertTrue(estimator.getRMS(problem) < 1.0e
-3);
which checks the same convergence property as the human-
written ground truth, although with a looser threshold.
This case study highlights the key advantages ofAssert-
Mate: By identifying a set of valid actual values for assertions,TABLE IV: Performance of using different base model for
AssertMate
Model CSR (%) PR (%) BDR (%)
DR-32B 59.44 55.10 45.54
QC-32B 74.3068.73 58.04
DC-7B76.7860.09 46.25
AssertMateis able to target more meaningful verification
points beyond the values of local variables, thereby enhancing
the thoroughness of test validations. At the same time, the
example also uncovers opportunities for future improvement.
WhileAssertMateleverages program structure to guide the
selection of meaningful targets, it currently lacks the ability
to automatically recognize and incorporate public utility meth-
ods (such as getRMS()), which often encapsulate essential
correctness conditions. This limitation suggests a promising
research direction is optimizing the automated identification
and inclusion of relevant public methods from the target
class. Such methods usually serve as ideal actual values, and
incorporating them could significantly improve the quality and
relevance of generated assertions.
B. AssertMate with More Advanced LLMs
Considering experimental costs and the practical usage, we
initially adopted DC-7B as the base model forAssertMate.
However, with the rapid evolution of LLMs, more advanced
models have demonstrated notably stronger capabilities in
code understanding and generation.
In particular, recent reasoning-enhanced models such as
DeepSeek-R1 have garnered substantial attention for their su-
perior performance in complex tasks. To explore the potential
benefits of integrating more advanced models into our frame-
work, we conducted an additional experiment evaluating the
effectiveness ofAssertMatewhen paired with larger and more
advanced LLMs. Specifically, we selected two open-source
models:Qwen2.5-Coder-32B (QC-32B), representing state-of-
the-art code-centric LLMs, andDeepSeek-R1-Distill-Qwen-
32B (DR-32B), exemplifying reasoning-enhanced LLMs. All
experiments were conducted under the same setup as RQ1,
and the corresponding results are presented in Table IV.
Code-centric advanced LLMs tend to improve assertion
generation effectiveness.Comparing the effectiveness of QC-
32B and DC-7B, we observed that advanced code-centric
LLMs significantly enhance assertion generation. Although
QC-32B incurred a slight drop in CSR of 2.48 percentage
points compared with DC-7B, it delivered gains of 8.64
and 11.79 percentage points in PR and BDR, respectively,
highlighting its strength in generating semantically accurate
and bug-revealing assertions. These results suggest that, when
resources permit, adopting larger and more advanced code
LLMs can be a valuable choice for boosting assertion quality.
While reasoning capabilities enhance LLM performance,
they are not the sole determinant of assertion generation
effectiveness.Although DR-32B offers advanced reasoning
capabilities, it does not outperform DC-7B under the unified
assertion-generation protocol adopted in this work, with CSR,
PR, and BDR lower by 17.34, 4.99, and 0.71 percentage

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING 13
points, respectively. One possible explanation is that the par-
tially prefilled assertion template, while effective for ensuring
executable and consistent outputs, may not be optimal for
reasoning-oriented models, whose capabilities are better ex-
ploited through a more flexible think-then-answer generation
process. Nevertheless, DR-32B achieves the best bug detection
performance on the Math project, suggesting that reasoning-
oriented models remain advantageous for reasoning-intensive
scenarios. This observation motivates future work on adaptive
prompting strategies that decouple semantic reasoning from
structured assertion generation, allowing reasoning-oriented
models to first reason about expected behaviors before produc-
ing executable assertions. More broadly, future agent collab-
oration strategies could dynamically route or weight different
model families according to task characteristics, enabling
reasoning-oriented and code-centric models to complement
each other.
C. Complementarity and Boundaries of Multi-Agent Collabo-
ration
The effectiveness of multi-perspective collaboration does
not originate from the number of agents, but from the diversity
of evidence sources and reasoning pathways they provide.
AssertMate adopts CGA, RAGA, and CoTA because they cap-
ture complementary aspects of assertion reasoning. CGA de-
rives expected values primarily from local program semantics
and executable code structure; RAGA leverages structurally
analogous tests to transfer assertion patterns and expected
behaviors; and CoTA performs explicit multi-step reasoning by
composing semantic relations from the focal method and test
prefix. These heterogeneous perspectives reduce reliance on a
single reasoning strategy and provide opportunities to identify
valid assertions that may be missed by individual agents.
The RQ3 results provide empirical support for this comple-
mentarity. The agent ablation analysis demonstrates that dif-
ferent perspectives likely capture different aspects of assertion
generation, while the collaboration-strategy comparison shows
that effective aggregation can further improve the reliability
of generated assertions. These results suggest that the benefit
of multi-agent collaboration comes from combining comple-
mentary reasoning signals rather than simply increasing the
number of agents. However, multi-perspective collaboration
is not universally beneficial. Its effectiveness depends on
several conditions: (1) at least one perspective produces a
correct candidate assertion, (2) different perspectives provide
complementary rather than redundant evidence, and (3) the
aggregation mechanism can identify and preserve valuable
candidates. Collaboration may provide limited benefits when
all agents lack necessary contextual information, share similar
failure modes, or when the aggregation strategy eliminates
useful minority candidates.
The current evaluation explores a representative but
bounded collaboration design space. Although the EVGen
component of AssertMate can support different collaboration
architectures, this study focuses on three representative mech-
anisms (i.e., majority voting, debate, and LLM-as-a-Judge)
under the same perspective setting. These mechanisms rep-
resent direct consensus, iterative interaction, and expert-basedselection, respectively, but they do not imply that the selected
architecture is universally optimal. Future work will investigate
adaptive perspective selection, dynamic agent allocation, and
more advanced agentic collaboration strategies.
D. Extensibility of Actual-Value Construction
The current ActVCon rules (i.e., the return values and
public getters) are designed as a practical approximation
rather than an exhaustive semantic model of all possible
observation points. In our Defects4J benchmark, these rules
cover 86.5% of the assertion targets appearing in developer-
written tests. This result suggests that return values and
object states exposed through public getters represent common
and practically useful sources of test oracles. More impor-
tantly, these observation points can be reliably identified and
reconstructed, enabling the generation of executable asser-
tions while avoiding the uncertainty introduced by arbitrary
method invocation. The remaining 13.5% of assertion targets
mainly involve parameterized method invocations used directly
as actual values, such asassertEquals(expected,
obj.computeSomething(param)). Supporting such
cases requires more than identifying observable values; it in-
troduces a parameter-synthesis problem, where the framework
must infer valid argument values, construct corresponding
declarations and assignments, and ensure that the generated
invocation is both type-correct and semantically meaningful
in the test context. Therefore, we intentionally exclude these
cases from the current design rather than introducing broad
invocation rules that may generate uncompilable or unreliable
assertions.
Nevertheless, extending ActVCon to support richer observa-
tion points is feasible. Potential directions include combining
static analysis, constant propagation, and symbolic execution
to infer valid arguments; mining project-specific test pat-
terns to identify common invocation strategies; and leveraging
LLM-based argument generation guided by method signatures
and test context. These extensions are complementary to the
current framework rather than fundamental changes to its core
design. The existing rule-guided observation-point construc-
tion already provides broad practical coverage, while future
extensions can further improve coverage for more complex
assertion scenarios.
E. Threats to Validity
External Threats. Our evaluation is based on Defects4J, a
widely adopted benchmark comprising real-world Java defects.
However, its limited dataset size and Java-specific character-
istics may limit generalizability to other programming lan-
guages. To mitigate this, we selected 2,203 EvoSuite-generated
FM–TP pairs when validating the integration ofAssertMate
with EvoSuite, substantially expanding the experimental scale.
Nevertheless, AssertMate is not inherently restricted to Java.
The language-dependent components are mainly confined to
ActVCon, which requires adapting program analysis and
assertion mapping rules to different languages and testing
frameworks, while the core LLM-based components remain
language-independent. For example, extending AssertMate to

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING 14
Python or C++ only requires replacing Java-specific analysis
with corresponding mechanisms for attribute/member analy-
sis and adapting assertion templates to frameworks such as
pytest or GoogleTest. In future work, we plan to extend
our evaluation to larger, multi-language datasets to validate
generalizability further.
Internal Threats.One internal threat stems from the inherent
randomness of LLM outputs. Although we fixed the temper-
ature setting, stochastic generation may still affect assertion
reproducibility. To address this, we generated five responses
per query and applied a probability-based re-ranking strategy
to select the most reliable output, thereby mitigating variabil-
ity. Another potential threat is contamination in the backbone
LLM. DeepSeek-Coder is trained on large-scale open-source
code, while its pretraining corpus is not publicly available;
therefore, we cannot fully verify whether Defects4J-related
projects are included in its training data or perform strict de-
contamination. However,AssertMatedoes not directly prompt
the LLM to reproduce existing developer-written assertions.
Instead, ActVCon first derives assertion targets through static
analysis and type-aware rules, and the LLM subsequently pre-
dicts expected values for these constructed actual values. The
generated assertions are further validated through compilation
and execution. Therefore, even if related Defects4J code was
encountered during pretraining, successful assertion generation
requires context-specific oracle inference rather than simple
memorization of existing assertions. While this design reduces
the risk of contamination-driven performance gains, we cannot
completely eliminate this threat.
Construction Threats.One major construct validity threat lies
in the evaluation of the pass rate and bug detection rate. Since
we focus on the unit test scenario, the correctness of the focal
method is often unknown, making it difficult to distinguish
between true assertion failures and false alarms. To address
this challenge, we adopted two mitigation strategies: (1) for
evaluatingAssertMate’s effectiveness, we used human-written
bug-triggering test cases from Defects4J, which are expected
to pass on the fixed version and fail on the buggy version, thus
serving as reliable ground truth; and (2) in RQ2, we used the
number of killed mutants as an alternative evaluation metric,
offering a more objective measure of assertion effectiveness
in fault detection.
VII. CONCLUSION
We proposedAssertMate, a novel agent-based assertion
generation framework to facilitate generating effective as-
sertions for automated testing.AssertMateoffers three key
advantages: (1) it integrates static program analysis to generate
actual values, effectively mitigating LLM hallucinations; (2)
it leverages complementary prompting strategies for richer
context understanding; and (3) it employs a tailored multi-
agent collaboration mechanism to determine accurate expected
values. Evaluation results demonstrate thatAssertMatesig-
nificantly outperforms existing state-of-the-art techniques and
validate the effectiveness of its core components.
This work opens several promising avenues for future
research. One direction is to improve the efficiency ofAssert-
Mateand extend its applicability beyond Java, broadening itsimpact across languages. Another is to explore more advanced
collaboration mechanisms, such as dynamic task allocation, to
better leverage the reasoning capabilities of large language
models. We also plan to address hallucinations in expected
value generation and component coordination by incorporating
techniques like knowledge distillation to improve consistency
and reliability. Furthermore, evaluatingAssertMatein multi-
assertion generation scenarios presents another valuable direc-
tion. While AssertMate already supports multi-assertion gener-
ation through ActVCon’s ability to synthesize multiple actual
values for the same test prefix, the challenge of determining
“where to assert”, especially when multiple test scenarios are
embedded within a single test case, remains underexplored
across existing approaches, including ours. Addressing this
challenge is critical for scaling assertion generation from
isolated unit behaviors to more realistic test cases.
ACKNOWLEDGMENT
This work was supported by the National Key Re-
search and Development Program of China (Grant No.
2024YFB4506300), the National Natural Science Foundation
of China (Grant Nos. 62322208, 12411530122, 62232001).
REFERENCES
[1] I. Hayet, A. Scott, and M. d’Amorim, “Chatassert: Llm-
based test oracle generation with external tools assis-
tance,”IEEE Trans. Software Eng., vol. 51, no. 1, pp.
305–319, 2025.
[2] Z. Yuan, M. Liu, S. Ding, K. Wang, Y . Chen, X. Peng,
and Y . Lou, “Evaluating and improving chatgpt for unit
test generation,”Proc. ACM Softw. Eng., vol. 1, no. FSE,
pp. 1703–1726, 2024.
[3] Y . Chen, Z. Hu, C. Zhi, J. Han, S. Deng, and J. Yin,
“Chatunitest: A framework for llm-based test genera-
tion,” inSIGSOFT FSE Companion, 2024, pp. 572–576.
[4] W. Sun, H. Li, M. Yan, Y . Lei, and H. Zhang, “Revis-
iting and improving retrieval-augmented deep assertion
generation,” inASE, 2023, pp. 1123–1135.
[5] L. Yang, C. Yang, S. Gao, W. Wang, B. Wang, Q. Zhu,
X. Chu, J. Zhou, G. Liang, Q. Wang, and J. Chen,
“On the evaluation of large language models in unit test
generation,” inASE, 2024, pp. 1607–1619.
[6] C. Watson, M. Tufano, K. Moran, G. Bavota, and
D. Poshyvanyk, “On learning meaningful assert state-
ments for unit test cases,” inICSE, 2020, pp. 1398–1409.
[7] H. Yu, Y . Lou, K. Sun, D. Ran, T. Xie, D. Hao, Y . Li,
G. Li, and Q. Wang, “Automated assertion generation
via information retrieval and its integration with deep
learning,” inICSE, 2022, pp. 163–174.
[8] S. B. Hossain and M. B. Dwyer, “TOGLL: correct and
strong test oracle generation with llms,”CoRR, vol.
abs/2405.03786, 2024.
[9] A. Arcuri and G. Fraser, “Java enterprise edition support
in search-based junit test generation,” inSSBSE, vol.
9962, 2016, pp. 3–17.
[10] C. Pacheco and M. D. Ernst, “Randoop: feedback-
directed random testing for java,” inOOPSLA Compan-
ion, 2007, pp. 815–816.

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING 15
[11] R. S. Boyer, B. Elspas, and K. N. Levitt, “SELECT -
a formal system for testing and debugging programs by
symbolic execution,” inReliable Software. ACM, 1975.
[12] W. E. Howden, “Symbolic testing and the DISSECT
symbolic evaluation system,”IEEE Trans. Software Eng.,
vol. 3, no. 4, pp. 266–278, 1977.
[13] J. C. King, “A new approach to program testing,” inPro-
gramming Methodology, ser. Lecture Notes in Computer
Science, vol. 23. Springer, 1974, pp. 278–290.
[14] S. Khurshid, C. S. Pasareanu, and W. Visser, “General-
ized symbolic execution for model checking and testing,”
inTACAS, ser. Lecture Notes in Computer Science, vol.
2619. Springer, 2003, pp. 553–568.
[15] C. Cadar, D. Dunbar, and D. R. Engler, “KLEE: unas-
sisted and automatic generation of high-coverage tests
for complex systems programs,” inOSDI. USENIX
Association, 2008, pp. 209–224.
[16] C. S. Pasareanu and N. Rungta, “Symbolic pathfinder:
symbolic execution of java bytecode,” inASE. ACM,
2010, pp. 179–180.
[17] W. Zheng, H. Ma, M. R. Lyu, T. Xie, and I. King,
“Mining test oracles of web search engines,” inASE,
2011, pp. 408–411.
[18] S. H. Tan, D. Marinov, L. Tan, and G. T. Leav-
ens, “@tcomment: Testing javadoc comments to detect
comment-code inconsistencies,” inICST. IEEE Com-
puter Society, 2012, pp. 260–269.
[19] A. Goffi, A. Gorla, M. D. Ernst, and M. Pezz `e, “Auto-
matic generation of oracles for exceptional behaviors,”
inISSTA. ACM, 2016, pp. 213–224.
[20] R. Baldoni, E. Coppa, D. C. D’Elia, C. Demetrescu, and
I. Finocchi, “A survey of symbolic execution techniques,”
ACM Comput. Surv., vol. 51, no. 3, pp. 50:1–50:39, 2018.
[21] E. Dinella, G. Ryan, T. Mytkowicz, and S. K. Lahiri,
“TOGA: A neural method for test oracle generation,” in
ICSE. ACM, 2022, pp. 2130–2141.
[22] W. Zhang, Y . Shen, L. Wu, Q. Peng, J. Wang, Y . Zhuang,
and W. Lu, “Self-contrast: Better reflection through in-
consistent solving perspectives,” inACL (1). Association
for Computational Linguistics, 2024, pp. 3602–3622.
[23] L. Zheng, W. Chiang, Y . Sheng, S. Zhuang, Z. Wu,
Y . Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang,
J. E. Gonzalez, and I. Stoica, “Judging llm-as-a-judge
with mt-bench and chatbot arena,” inNeurIPS, 2023.
[24] R. Just, D. Jalali, and M. D. Ernst, “Defects4j: a database
of existing faults to enable controlled testing studies for
java programs,” inISSTA. ACM, 2014, pp. 437–440.
[25] “Homepage,” Accessed: 2025, https://github.com/
LeonYang95/AssertMate.
[26] Z. Tian, H. Shu, D. Wang, X. Cao, Y . Kamei, and J. Chen,
“Large language models for equivalent mutant detection:
How far are we?” inISSTA, 2024, pp. 1733–1745.
[27] C. Yang, L. Yang, Z. Wang, D. Wang, J. Zhou, and
J. Chen, “Clarifying semantics of in-context examples
for unit test generation,”ASE, 2025.
[28] Q. Zhang, C. Fang, Y . Zheng, Y . Zhang, Y . Zhao,
R. Huang, J. Zhou, Y . Yang, T. Zheng, and Z. Chen,
“Improving deep assertion generation via fine-tuningretrieval-augmented pre-trained language models,”ACM
Trans. Softw. Eng. Methodol., 2025.
[29] S. Primbs, B. Fein, and G. Fraser, “Assert5: Test assertion
generation using a fine-tuned code language model,”
arXiv preprint arXiv:2502.02708, 2025.
[30] P. Nie, R. Banerjee, J. J. Li, R. J. Mooney, and M. Glig-
oric, “Learning deep semantics for test completion,” in
ICSE. IEEE, 2023, pp. 2111–2123.
[31] M. Tufano, D. Drain, A. Svyatkovskiy, S. K. Deng,
and N. Sundaresan, “Unit test case generation with
transformers,”CoRR, vol. abs/2009.05617, 2020.
[32] F. Li, J. Jiang, J. Sun, and H. Zhang, “Hybrid automated
program repair by combining large language models and
program analysis,”ACM TOSEM, 2024.
[33] J. Jiang, Y . Wang, J. Chen, D. Lv, and M. Liu, “Variable-
based fault localization via enhanced decision tree,”ACM
Trans. Softw. Eng. Methodol., pp. 1–32, 2023.
[34] S. Shamshiri, R. Just, J. M. Rojas, G. Fraser, P. McMinn,
and A. Arcuri, “Do automatically generated unit tests
find real faults? an empirical study of effectiveness and
challenges (t),” inASE, 2015, pp. 201–211.
[35] S. Hochreiter and J. Schmidhuber, “Long short-term
memory,”Neural Comput., pp. 1735–1780, 1997.
[36] E. Nijkamp, B. Pang, H. Hayashi, L. Tu, H. Wang,
Y . Zhou, S. Savarese, and C. Xiong, “Codegen: An open
large language model for code with multi-turn program
synthesis,” inICLR. OpenReview.net, 2023.
[37] D. Guo, S. Lu, N. Duan, Y . Wang, M. Zhou, and J. Yin,
“Unixcoder: Unified cross-modal pre-training for code
representation,” inACL (1), 2022, pp. 7212–7225.
[38] Z. Tian, J. Chen, and X. Zhang, “Fixing large language
models’ specification misunderstanding for better code
generation,” inICSE. IEEE, 2025, pp. 1514–1526.
[39] D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong, W. Zhang,
G. Chen, X. Bi, Y . Wu, Y . K. Li, F. Luo, Y . Xiong, and
W. Liang, “Deepseek-coder: When the large language
model meets programming - the rise of code intelli-
gence,”CoRR, vol. abs/2401.14196, 2024.
[40] “Chatassert homepage,” Accessed: 2025, https://github.
com/ncsu-swat/chatassert.
[41] “Tree sitter: An incremental parsing system for program-
ming tools,” Accessed: 2025, https://tree-sitter.github.io/
tree-sitter/.
[42] “Pytorch,” Accessed: 2025, https://pytorch.org.
[43] “Huggingface transformers library,” Accessed: 2025,
https://huggingface.co/docs/transformers/index.
[44] “vllm library,” Accessed: 2025, https://github.com/
vllm-project/vllm.
[45] R. F. Woolson, “Wilcoxon signed-rank test,” inWiley
Encyclopedia of Clinical Trials. John Wiley & Sons,
2007, pp. 1–3.
[46] N. Cliff, “Dominance statistics: Ordinal analyses to an-
swer ordinal questions,”Psychological Bulletin, vol. 114,
no. 3, pp. 494–509, 1993.
[47] G. Fraser and A. Arcuri, “A large-scale evaluation of
automated unit test generation using evosuite,”ACM
Trans. Softw. Eng. Methodol., pp. 8:1–8:42, 2014.
[48] K. Xiong, X. Ding, Y . Cao, T. Liu, and B. Qin, “Ex-

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING 16
amining inter-consistency of large language models col-
laboration: An in-depth analysis via debate,” inEMNLP
(Findings), 2023, pp. 7572–7590.
[49] T. Guo, X. Chen, Y . Wang, R. Chang, S. Pei, N. V .
Chawla, O. Wiest, and X. Zhang, “Large language model
based multi-agents: A survey of progress and challenges,”
inIJCAI. ijcai.org, 2024, pp. 8048–8057.
[50] E. McCrum-Gardner, “Which is the correct statistical
test to use?”British Journal of Oral and Maxillofacial
Surgery, vol. 46, no. 1, pp. 38–41, 2008.
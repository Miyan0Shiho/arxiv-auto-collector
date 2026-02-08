# A Guide to Large Language Models in Modeling and Simulation: From Core Techniques to Critical Challenges

**Authors**: Philippe J. Giabbanelli

**Published**: 2026-02-05 17:00:07

**PDF URL**: [https://arxiv.org/pdf/2602.05883v1](https://arxiv.org/pdf/2602.05883v1)

## Abstract
Large language models (LLMs) have rapidly become familiar tools to researchers and practitioners. Concepts such as prompting, temperature, or few-shot examples are now widely recognized, and LLMs are increasingly used in Modeling & Simulation (M&S) workflows. However, practices that appear straightforward may introduce subtle issues, unnecessary complexity, or may even lead to inferior results. Adding more data can backfire (e.g., deteriorating performance through model collapse or inadvertently wiping out existing guardrails), spending time on fine-tuning a model can be unnecessary without a prior assessment of what it already knows, setting the temperature to 0 is not sufficient to make LLMs deterministic, providing a large volume of M&S data as input can be excessive (LLMs cannot attend to everything) but naive simplifications can lose information. We aim to provide comprehensive and practical guidance on how to use LLMs, with an emphasis on M&S applications. We discuss common sources of confusion, including non-determinism, knowledge augmentation (including RAG and LoRA), decomposition of M&S data, and hyper-parameter settings. We emphasize principled design choices, diagnostic strategies, and empirical evaluation, with the goal of helping modelers make informed decisions about when, how, and whether to rely on LLMs.

## Full Text


<!-- PDF content starts -->

A Guide to Large Language Models in Modeling
and Simulation: From Core Techniques to
Critical Challenges
Philippe J. Giabbanelli
AbstractLarge language models (LLMs) have rapidly become familiar tools to
researchers and practitioners. Concepts such as prompting, temperature, or few-shot
examples are now widely recognized, and LLMs are increasingly used in Modeling
& Simulation (M&S) workflows. However, practices that appear straightforward
may introduce subtle issues, unnecessary complexity, or may even lead to inferior
results. Adding more data can backfire (e.g., deteriorating performance through
model collapse or inadvertently wiping out existing guardrails), spending time on
fine-tuning a model can be unnecessary without a prior assessment of what it already
knows, setting the temperature to 0 is not sufficient to make LLMs deterministic,
providing a large volume of M&S data as input can be excessive (LLMs cannot attend
to everything) but naive simplifications can lose information. We aim to provide
comprehensive and practical guidance on how to use LLMs, with an emphasis
on M&S applications. We discuss common sources of confusion, including non-
determinism, knowledge augmentation (including RAG and LoRA), decomposition
of M&S data, and hyper-parameter settings. We emphasize principled design choices,
diagnostic strategies, and empirical evaluation, with the goal of helping modelers
make informed decisions about when, how, and whether to rely on LLMs.
Key words:Generative AI, Knowledge Augmentation, Non-Determinism, Model-
ing Workflows, Prompt Engineering
1 Introduction
Generative AI (GenAI) is often associated with Large Language Models (LLMs),
although the field also covers image or video generation. In the early 2020s, there
was a sense of marvel (or disappointment) depending on whether an LLM could
Philippe J. Giabbanelli
VMASC, Old Dominion University, USA, e-mail: pgiabban@odu.edu
1arXiv:2602.05883v1  [cs.AI]  5 Feb 2026

2 Philippe J. Giabbanelli
produce the right textual output solely based on prompts and a series of hand-curated
examples, using either simple notebooks in Python or a web-portal. Prompting could
already be challenging as studies showed that users relied on trial-and-error [137, 75],
with researchers noting that “finding the right prompt or model has become an
industry unto itself” [3]. The LLM ecosystem has only gotten more complex, within
a short amount of time that makes it challenging to keep up with technological
change. We now design systems with LLMs as part ofpipelinesthat operate in
multiple stages involving evaluation, deployment and monitoring with respect to
governance and risk controls (Figure 1) – each stage needing its own libraries. As
illustrated by other books [56], there are now over 100 ‘techniques’ to optimize the use
of LLMs, many ‘strategies’ to handle biases or privacy, and hundreds of benchmarks.
This proliferation makes it difficult for users to know the right procedures when using
LLMs in their application context. This affects modeling and simulation experts as
well as model commissioners and simulation end-users, who may harbor beliefs on
LLMs that are disconnected from technical feasibility, reducing them to a magic
wand that can ‘quickly’ be used [132, 114]. Our previous observation on infusing
Modeling & Simulation (M&S) with machine learning can thus be updated as
follows:
“when a new approach with high potential is identified, there can be a temptation for end-
users to aim at becoming experts in the approach. This is certainly attractive from a logistical
standpoint, as end-users can then take care of their own needs, at their pace. However,
acquiring another expertise can be challenging for simulationists. [...] Simulationists should
be exposed to what [an LLM] can do for them, and what it needs (e.g., data requirements,
computational costs). This would position simulationists as informed end-users who can
identify reasonable questions in a given context.” [37]
We aim to inform users of LLMs rather than AI/ML experts about the tools
and approaches. We draw extensively on our years of experience in using LLMs
for M&S along with the mistakes that we have (unintentionally) made, so that our
journey can be of benefit to others. In line with other tutorials and case studies on
LLMs for M&S [1, 31, 122], Section 2 begins by covering fundamental compo-
nents: prompting, hyper-parameters like temperature, and extending the knowledge
of LLMs through approaches such as retrieval-augmented generation. Rather than
abstract best practices, we examine each of these components as engineering deci-
sions where choices have real consequences for the performance, replicability, or
transparency of downstream modeling and simulation tasks. We thus pay particu-
lar attention to failures and lack of optimization through reliance on defaults, so
that simulation practitioners can reason critically about when and how LLM-based
techniques should be employed.
The next sections cover common mistakes and flawed beliefs. Section 3 begins
with the important matter of non-determinism: we clarify that there are different
sources for non-determinism (so it does not just go away by setting the temperature to
0) and we explain how the impact can first be evaluated and, when needed, mitigated.
Section 4 covers the temptation of using LLMs because they are a convenient one-
stop shop for everything, at the risk of reinventing the wheel or coming up with
solutions that appear satisfactory, but are suboptimal. Section 5 discusses emerging

Large Language Models for Modeling & Simulation 3
Fig. 1: Rather than a single prompt–response interaction, LLM-based systems are
organized aspipelines: we need to know what version of the data is used (e.g.,
viadata version control), automate the experiments (e.g., throughMLflow),
ensure that personally identifiable information is detected and removed (e.g., using
Presidio), evaluate with respect to fairness, etc.Orchestration(e.g., based on
LangChain) is necessary to coordinate these many aspects.
notions, such as forgetting in LLMs so that we can identify and remove errors
instead of adding patches (i.e., shifting from designing LLM pipelines by addition
to designing by subtraction), or the potential to use images alongside text as input.
Finally, we provide a series of exercises and practical exploration questions that cover
decomposing a problem so that an LLM can handle it (instead of simply feeding
massive inputs or haphazardly chopping them and losing information), assessing
what an LLM already knows, and augmenting its knowledge when needed.
There are many other relevant topics for LLMs, either in general or applied to
M&S in particular, such as comparing with state-of-the-art-solutions, or systemat-
ically removing components of the system to understand each part’s contribution
to overall performance (i.e., performing anablation study). These are common and
well motivated requests when reviewing a research paper on LLMs.

4 Philippe J. Giabbanelli
2 Core elements of LLMs: prompts, parameters, augmentation
2.1 Prompt engineering
The prompt is the textual input provided to an LLM. It is the central element
of the LLM-system interface through which users specify tasks, constraints, and
evaluation criteria. Since there are several surveys, frameworks [120], catalogs of
templates [105, 102], and taxonomies on prompt engineering (c.f. Figure 2 in [98]),
we briefly summarize best principles and then focus on prompt construction as it
pertains to modeling and simulation. The highly cited study “Why Johnny Can’t
Prompt” showed that non-experts struggled with generating prompts and evaluating
their effectiveness [137]. These issues partly stem from defining tasks implicitly, be-
ing overconfident about the results, evaluating results with just one metric, debugging
by adding more data. In contrast, best practices followed by experts includedefining
tasks explicitly, cautiouslyverifying the results,systematically debugging, andmea-
suring performances through multiple indicators. Empirical studies have also shown
that increasing prompt length or combining multiple prompt engineering techniques
does not always improve performance: it may even degrade output quality [80]. For
example, the notion ofover-promptingshows that performance can decrease as we
add too many examples [117] – which runs counter to common practices observed
in early applications of LLMs. Early studies suggested that the degradation of per-
formances from longer prompts was a problem of retrieval as LLMs may be unable
to find information in longer prompts, but newer studies show thatperformance
degrades with increasing input length even when retrieval is perfect[25]. Effective
prompt construction therefore requiresselectivityrather than including all informa-
tion. Overly complex prompts may introduce ambiguity, for example by blurring the
distinction between instructions, examples, and counter-examples.
In our experience using LLMs for modeling and simulation, six aspects have been
particularly helpful. First,the task should be decomposed. For example, extracting
a conceptual model for a corpus is not a specific granular task but a high-level
goal. Decomposing it into a series of prompts1would include finding concepts from
the text, then identifying which concepts are related, and finally characterizing the
relationship [41]. Second, each prompt to perform a task should be followed by a
validation prompt. Prompts 1 to 3 exemplify the division of a goal into smaller
tasks and the use of a validation prompt [104]. The overarching goal of creating
1Agentic AIorLLM multi-agent systemare sometimes used as an umbrella term for workflows with
multiple steps [136], such as observe/reason/decide [131]. Each prompt is then called an ‘agent’,
so our sample prompts 1 to 3 could be rebranded as agents 1 to 3. We do not use this terminology
here, as we consider thatsequential promptingisn’t agentic AI because simply decomposing a task
lacks the essential property ofautonomytypically associated with agents, while a basic chain of
coordination does not qualify as a multi-agent system. Note that there are architectures in which the
term ‘agent’ is used without referring to agentic AI, such as thechain-of-agentsapproach in which
a long input is segmented into units that are processed sequentially by different LLMs [141]. For
a complementary discussion on the terminological drift about ‘agents’ in LLMs and the minimal
criteria that should be met, we refer the reader to section 3 in [13].

Large Language Models for Modeling & Simulation 5
typed relationships between concepts is broken into two steps (finding relationships,
characterizing them) and each step is followed by a validation. Also note that each
prompt includes a clear task statement, an example, and the expected output format.
Prompt 1. Finding relationships.
List all causal relationships between a list of concepts with each relationship as a
pair, ensuring each concept is involved in at least one relationship. For example,
given [‘coffee consumption’, ‘sleep’, ‘energy level’], return [(‘coffee consumption’,
‘sleep’), (‘sleep’, ’energy level’), (‘coffee consumption’, ‘energy level’)] with no
other text.<<CONCEPTS GO HERE>>
Prompt 2. Validating relationships.
Return whether a causal relationship exists between the source and target concepts
for each pair in a list. For example, given [(‘smoking’, ‘cancer’), (‘ice cream sales’,
‘shark attacks’)], return [‘Y’, ‘N’] with no other text.<<CONCEPTS GO HERE>>
Prompt 3. Characterizing relationships.
Given a list of causal relationships, return whether each is positive or negative. For
example, given [(‘coffee consumption’, ‘sleep’), (‘sleep’, ‘energy level’), (‘coffee
consumption’, ‘energy level’)], return [‘-’, ‘+’, ‘+’] with no other text.<<CAUSAL
RELATIONSHIPS GO HERE>>
Third, as an interdisciplinary area, M&S can be applied to social systems. This
can involve asking LLMs to structure a conceptual model for a social problem or
tasking LLMs with reacting as if they were people with defined socio-demographic
attributes [36]. Such settings are prone totriggering guardrails, as developers set
rules to prevent an LLM for acting ‘like a real person’ or engaging in certain topics.
If a prompt starts by telling an LLM “you area 40 year old white male so what do
you think about...”, then the LLM may reply that no, it isn’t a real person. But if the
prompt instead begins by telling the LLM “imagine that you are” or “assume that
you are”, then the LLM may be willing to entertain the idea and actually answer the
question from the perspective of the target demographics2[145]. A related challenge
is the refusal of an LLM to engage in charged social topics, such as politics or self-
harm. We encountered this case when creating conceptual models and performing
simulations for suicide prevention, as this is a sensitive topic for LLMs. While we
appreciate limitations on topics such as self-harm given the problematic uses of
LLMs in mental health [127], there is a difference between (the acceptable practice
of) engaging in an academic discussion on structuring the causes of suicide in the
2This is known asrole promptingand it is not exactly equivalent to ajailbreak. Jailbreaking is
the practice of bypassing explicit safety policies (i.e., modifying the prompt to override the LLM’s
refusal) in order to induce disallowed content or extract restricted information – for examples,
see [126]. A related term is ared team attack, that is, a deliberate, adversarial testing process in
which researchers or practitioners actively try to break, misuse, or bypass the LLM’s safeguards to
identify its weaknesses before real users do [35]. In contrast, role prompting respects the model’s
constraint of refusing to be a person by shifting from identity claims (‘you are’) and instead treats
people as hypothetical constructs [90] for reasoning (‘imagine that you are’).

6 Philippe J. Giabbanelli
population compared to (engaging in harmful behavior by) providing an individual
with a plan for attempting. For modeling and simulation purposes, we note that an
LLM’s refusals are sometimes triggered by surface-level lexical cues (using words
that trigger a banned topic) rather than by the semantic intent of the task (how the topic
will be used). In this situation, we use a prompt rewrite strategy: one ‘permissive’
LLM rewrites the prompt by describing the topic without using triggering words,
then the prompt is sent to the LLM that we want to use [34].
Fourth,the format of the input matters. Theoretically, whether we describe a
conceptual model (i.e., a graph) as a list of nodes and edges or as an adjacency
matrix should be inconsequential because it specifies the same mathematical object.
Yet, Fatemi and colleagues showed that the model representation, the task, and even
the structure of the model itself can affect the LLM’s performance. For instance, a
graph can be specified as a list of edges (a,c), (b,c), (c,a) or as a list of neighbors
per nodes: a connects to c, b connects to c, c connects to a. To find if two nodes
are connected, the latter representation resulted in 53.8% accuracy and the former
yielded 19.8% accuracy on Google’s mid-size LLM, PaLM 62B3. The authors posit
that the more the LLM needs to make inferences to find the information needed for
a task, the less accurate it becomes [27]. Still, the choice of representation is not
obvious for all tasks, and different LLMs can sometimes have surprising results.
We thus recommend considering different representations of the model’s structure
and/or simulation results and empirically evaluating how they affect performances4.
Fifth,transfer learningis an efficient way to tell the LLM to do a task ‘in the
same style’ as something else than it knows but would be hard to define otherwise.
Intuitively, it would be a bad idea to tell an LLM (or a person) to ‘write me a beautiful
piece of text’ because it has to figure out what is ‘beauty’, so a better version could be
“write me a text in the style of Ernest Hemingway”. The idea of leveraging an LLM’s
understanding of ‘style’ (“explain a model in the style of a consultant”) was discussed
in details by Peter and Riemer [91]. When using an LLM to explain the simulated
journey of agents from agent-based models, we compareddirectlyprompting for an
empathetic narrative withindirectlygenerating an empathetic narrative through style
transfer from popular figures who are known for empathy. Our results showed that
3This model was introduced in 2022, and results may differ with more recent LLMs. This illustrates a
broader methodological limitation of the field: empirical findings can depend strongly on the specific
model version used. Readers are therefore encouraged to consider whether reported conclusions
may warrant re-evaluation as models evolve, for instance by checking publication dates or the
release timeline of the LLMs involved. In our views, simply re-running an existing experimental
setup with a newer LLM does not, in itself, constitute a novel research contribution, although such
replications may be valuable when conductedsystematicallyacrossmultiplestudies.
4Identifying the right representation can become part of the experimental design. When the
experimental design relies onbinaryfactors (c.f. the factorial design discussed in section 3.2),
adding a single factor (choice of model representation) with more than two values can become
problematic. This can be addressed to conveniently incorporate different representations into an
existing experimental design as two binary factors: whether it is an array or list, and whether it is
adjacency or tag-based. This creates four combinations: adjacency matrix, adjacency list, list of
tags (in the standard RDF notation), and array of tags (which describes the content of a matrix in an
XML notation). If the original experimental design examines 2𝑘combinations for𝑘factors, then
folding the choice of representation into the design results in 2𝑘+2combinations.

Large Language Models for Modeling & Simulation 7
leaving the LLM to decide what is ‘empathetic’ resulted in melodramatic descriptions
with a hint of 19th century romanticism, which may have had its qualities but did not
fit our needs of explaining agents to decision-makers. When we transferred empathy
from known figures, the narrative was more relatable and without excess [39].
Finally, dealing with prompts does not stop at building the prompt: we should
also be prepared toextract the answersfrom the LLM. Even when an LLM is
supposed to simply state an option from a list, it may not do it exactly as required.
For instance, it may state “I choose option A” or “The best option is A” instead of
just stating “Option A”. This may be solved through simple parsing options such as
using regular expressions, but we have encountered cases where neither prompts that
specified and exemplified the output format nor regular expressions could guarantee
that the answer was ready for analysis. If modelers note that some outputs cannot be
processed, then a last recourse is to use a simpler LLM for the extraction task. We
suggest using it only as a last recourse since regular expressions provide a consistent
answer for free, whereas LLMs may have a cost or variability in parsing. Some
providers also have the option of astructured output format(e.g., using JSON with
Mistral’s API), which simplifies the organization of the output as we can request
multiple keys such as ‘reasoning’ as a string, ‘score’ as a float, and ‘option’ as a
character.
Although our focus is on how to optimize the structure of prompts, it is also
becoming important tocommunicatethese prompts to other scientists or end-users.
As an analogy, consider the problem of communicating algorithms: conveying them
purely as narratives can become cumbersome as they grow in sophistication, so we
use pseudocode that have a clear sense of structure, or we provide code in a target
language. At present, many papers simply include prompts in a box (like prompts
1–3) or refer the readers to the code. A more systematic approach in communicat-
ing prompts could improve reproducibility, modularity (e.g., instead of copy and
pasting very large prompts without knowing why they were structured that way),
and interpretability (so that readers know which part of a prompt were designed for
what purpose). Among recent works,Prompt Decorators[46] propose a declarative
and composable syntax to express prompts through a set of decorators5that can be
parameterized when applicable, such as+++OutputFormat(format=JSON). An
alternative is to express the prompt through a formal notation such as EBNF, so that
the structure is clear and the formatting makes the parameters apparent, as shown in
Listing 1 from [116] and exemplified below.
⟨Prompt⟩::=⟨Context⟩⟨Examples⟩?⟨SelectionCriteria⟩?⟨Instructions⟩⟨Task⟩
⟨Context⟩::= ’You are assisting with the construction of an agent-based model for
urban mobility. The goal is to extract entities and interactions from policy
documents.’
⟨Examples⟩::=⟨ExampleHeader⟩⟨Example⟩+ (⟨ExampleHeader⟩⟨Example⟩+) ?
5For example,+++StepByStepto execute the task in small increments before synthesizing the
answer,+++Critiqueto provide structured feedback on strengths, weaknesses, and improvements,
or+++OutputFormatto enforce the syntax produced by the LLM.

8 Philippe J. Giabbanelli
⟨ExampleHeader⟩::= ’I give ({𝑁+}|{𝑁−}) examples with{𝐹𝐸𝐴𝑇𝑈𝑅𝐸}that
should be (included|excluded).’
⟨SelectionCriteria⟩::= ’Include only entities that represent actors, resources, or state
variables. Exclude purely descriptive or rhetorical concepts.’
⟨Instructions⟩::= ’Identify entities and directed relationships. Use concise names.
Do not infer entities not grounded in the text.’
⟨Task⟩::= ’Given the following document excerpt, extract a list of entities and
relationships.’
While this section focused on techniques that we found useful when using LLMs
for modeling and simulation, it would bemisleading to view all prompts as manually
crafted, human-readable, and based on a set catalog of design techniques. The idea
that generative AI would create a new job that mostly involves writing manually
crafted prompts (i.e., prompt engineers [59]) is being at least partly challenged
by progress inoptimization techniques for automated prompt design(c.f. dspy.ai).
As shown in the taxonomy from Cui and colleagues (c.f. Figure 1 in [21]), there
is a wide range of optimization methods and several tools are available (half of
which are open source). This does noteliminatethe need for human interventions
in interacting with LLMs, since tasks do have to be specified. Rather, automation
changes the nature of human involvement from manually crafting the entire prompt
to minimal involvement in the prompt and a greater emphasis on design choices such
as setting the right objectives for the optimization. There are also opportunities for
automation throughprompt compression techniques. We should be mindful that we
writeprompts to be executed by an LLM, not by humans. For example, we could cut
on politeness (‘hello, could you please’)6and it may be unaffected by removing stop
words (a, an, the...). However, in striving for concision, we may forget that repeated
instructions can be helpful. Techniques for prompt compression thus examine how
to reduce costs while maintaining (or even improving) LLM performances [142, 68].
Finally, newer LLMs may have different features from the previous generation (e.g.,
reasoning LLMs such as OpenAI’s o-series, DeepSeek-R1, or Qwen3), which calls
for new approaches to prompting. For instance, the classic prompt element “think
step-by-step” made sense for prior generations of LLMs, but new LLMs already
engage in chain-of-thought reasoning. Our experiments in using reasoning LLMs
as part of simulation models suggested that prompts that called for reasoning could
worsen performances [146].
6In the context of concision, the prompts include neither politeness markers nor rud content, so they
areneutralin tone. Prior studies have examined the impact of tonal variations in prompts, ranging
from polite to neutral and rude. For example, Dobariya and Kumar evaluated prompt templates
spanning from ‘Can you kindly consider the following problem and provide your answer’ to ‘I know
you are not smart, but try this.’ [24] The influence of such tonal choices depends on both the LLM
and the application. A systematic evaluation found that Gemini was largely insensitive to prompt
tone, while other models exhibited statistically significant effects only for specific humanities tasks.
When results are aggregated across domains, differences in performance attributable to prompt
tone tend to become negligible [15].

Large Language Models for Modeling & Simulation 9
Prompt
Tokenization
Neural network
(fixed weights)
Probability distribution
over next token
Decoding hyper-parameters
Token selection
Append token to contextThe textual input specifying the task, constraints, and con-
text provided to the LLM.
Conversion of text into discrete symbols (tokens) according
to a fixed vocabulary and encoding scheme.
A pretrained transformer that computes scores over possible
next tokens. Weights are fixed at inference time.
Normalized likelihoods assigned by the model to each can-
didate next token, conditioned on the current context.
Inference-time controls (e.g., temperature, top-𝑘, top-𝑝, rep-
etition penalties) to reshape/truncate the prob. distribution.
Select a token from the modified distribution, either deter-
ministically (e.g., argmax) or stochastically (sampling).
The selected token is appended to the existing token se-
quence before the next prediction step.
Fig. 2: Inference-time generation pipeline for LLMs. Gray boxes indicate stages that
are deterministic given a fixed prompt, tokenization scheme, and model weights.
Blue boxes indicate stages where stochasticity may be introduced through decoding
hyper-parameters and sampling. Generation proceeds autoregressively by repeatedly
appending selected tokens to the context.
2.2 Hyper-parameters
To appreciate the effect of hyper-parameters such as the well-knowntemperature,
it is useful to see the flow from the prompt (explained in the previous section) to
the generation of tokens. Inference in LLMs involvesdecoding hyper-parameters
(Figure 2). These should not be confused for the hyper-parameters that were used
when training a neural network (e.g., learning rate, batch size), which are not a
concern when using LLMs off-the-shelf. In other words, training hyper-parameters
shape the model’s weights (which comes pre-packaged when using e.g. GPT) while
decoding hyper-parameters shape the behavior of the model by specifying how
probabilities are turned into text.
The level of control that we can exert over an LLM at decoding time depends on
the level of access that we have [138, 107]. With open-source LLMs (e.g., LLaMA,
Mistral, Falcon), we can control the decodingstrategyand that decides which set of
hyper-parameters we can use. A deterministic strategy could be a greedy decoding
(which has no hyper-parameter) or beam search [49, 138] (which has a single hyper-
parameter𝑤), a contrastive strategy [66] that could be adaptive (no hyper-parameter)
or a search with two hyper-parameters (𝛼, k), or the sampling-based strategies used
by LLMs such as GPT with hyper-parameters including the temperature. LLMs
accessed through an API (e.g., GPT, Claude, Gemini, DeepSeek) overwhelmingly
rely on sampling-based decoding,not because it is the best for a user’s specific task,

10 Philippe J. Giabbanelli
but because it works well-enough across the diverse tasks that users may want and
it is relatively easy to control when enforcing policies. Other strategies may lead
to better results in some tasks. For instance, deterministic strategies are strong for
closed-ended or classification tasks. However, they may struggle with open-ended
generations7, lead to ‘degeneration’ [49], or interact poorly with safety mechanisms8
(e.g., small changes can affect the output massively). Alternatives such as contrastive
search strategies may need the hyper-parameters to be configured very carefully (e.g.,
for a contrastive search). Besides, sampling-based decoding parallelizes well and has
a clear cost per token, while other strategies may have a higher or less predictable
per-token cost [83] – which would be undesirable when running operations at a very
large scale. In short, API providers must optimize for safety compliance, cost, and
average satisfaction over a broad array of tasks, whereas researchers on decoding or
those willing to optimize a system for theirspecific task can maximize accuracy[4].
The modeling and simulation community has a (very) long way to go in terms of
optimizing decoding with LLMs. There is currently a lack of studies using diverse
decoding strategies, as most rely on OpenAI and thus have to use its sampling-
based decoding regime. Even so, there are several hyper-parameters that could be
optimized, in particular the temperature. Nonetheless, we find that most studies
leave the temperature to its default setting, either by stating it explicitly [111] or
by not reporting the use of any specific value. This non-reporting (and presumably
default value) happens regardless of whether the study seeks to match conceptual
models [8, 45, 94], assess a model [144, 43], or generate a model [57, 85, 55]. On
occasions when a different temperature is used [47], researchers “set the temperature
of the LLM to zero in order to ensure full replicability and thereby increase the
scientific rigor of the results” [11] – which is debatable as shown in section 3.1. This
practice is problematic for two reasons. First,a ‘default’ temperature is not a universal
constant: Mistral used 0.7 as default9, DeepSeek uses 1.010, and Llama chose 0.811.
This non-reporting of temperature settings undermines the reproducibility of studies,
which is already an important concern in modeling and simulation [118] as it affects
7For instance, for creative or more human-like responses, we seekdiversityin the responses. If the
answers are too similar, then we may get the feeling of conversing with a robot, and we cannot use
the LLM for creative tasks. The lack of diversity of techniques such as beam search can be addressed
by using lexical variations [123], which is satisfactory if the goal is to produce a large amount of
labels (e.g., for captioning a dataset that will later be used to train models). LLMs such as Mistral
have a penalty parameter to avoid the repetition of words or phrases. However, using different words
while adhering to the same structure or arguments may still come across as a ‘superficial’ level of
diversity in human-facing applications. New solutions continue to be proposed to improve decoding
processes to ensure that answers are semantically diverse [109].
8The output probabilities of an LLM can be adjustedat decoding time, e.g. by increasing the
probability of desired tokens and reducing the probability for undesirable ones with respect to a
safety property. See [125] and references therein for emerging research in this area.
9https://github.com/mistralai/mistral-inference/blob/
db6b4223c9f6c0c3c0182166760f5e0b8813a723/src/mistral inference/main.py#L105
10https://web.archive.org/web/20251206082455/https://api-docs.deepseek.com/
quick start/parameter settings
11https://github.com/abetlen/llama-cpp-python/blob/main/llama cpp/server/types.py#L25

Large Language Models for Modeling & Simulation 11
the trust that we can place in a model [44]. Second, relying on default settings is
a missed opportunity for optimization. This can be surprising in a modeling and
simulation context: for example, significant efforts have been devoted to optimizing
agent-based models [87], yet when they include LLMs, they use default values. This
situation may be partly explained by the high computational cost of simulations using
LLMs: an extensive hyperparameter optimization can be prohibitively expensive for
many research groups. Indeed, a review of 35 agent-based models using LLMs noted
that “a concerning expression of these costs is the widespread failure to conduct
robustness checks or sensitivity analyses”, which could have included varying the
temperature setting [61].
When optimizing the temperature setting in our modeling and simulation studies,
we noted important differences. We optimized the readability of a text that explains
a conceptual model based on the size of the context window provided to the LLM
(i.e., how many sentences we passed within one prompt) and the temperature [33].
Our optimization across three versions of GPT shows that the text was generally least
readable at a temperature of 1 and the optimal value depends on the other parameter
(the size of the context window and the specific LLM used). For example, on GPT
3.5 Turbo with 5 sentences, we obtained the highest readability at a temperature of
0.75, but if we increase to 10 sentences then the optimal temperature shifts to 0.25.
In another case, we sought to build simulation models and we optimized them with
respect to sparsity and accuracy by varying different temperature settings across
LLMs (Gemini, Claude, GPT) and applications. We found that different LLMs had
very different reactions to changes in temperature [104]. For example, Claude-3-
Opus barely reacted to the temperature settings, but the error could double when
using Gemini at the ‘wrong’ temperature setting. The best temperature was heavily
application-dependent: in one setting, a temperature of 0.25 produces the lowest
error while 0.75 produces the highest one, and in another setting this relationship is
flipped. If the simulation results are primarily optimized with respect to temperature,
then the study could include a line chart (varying the response𝑦as a function of
temperature𝑥). If the response is optimized as a function of the temperature and
another variable, then a heatmap can be used. And if the optimization is done over
several LLMs, then several heatmaps can be provided (Figure 3).
While temperature is a well-known hyper-parameter, sampling-based decoding
has other hyper-parameters (e.g.,𝑡𝑜𝑝 𝑝,𝑡𝑜𝑝𝑘). However, different APIs will expose
different hyper-parameters12and users should not be changing all of these hyper-
parameters together (e.g., Mistral recommends optimizing temperature or𝑡𝑜𝑝 𝑝but
not both). Beyond decoding-related hyper-parameters, there are also several control-
lable aspects that could be of particular interest as they relate to cost or latency. For
example, Mistral provides a ‘batch inference’ mode, while OpenAI experimented
with a ‘flex processing’ mode in which some models would be cheaper (e.g., half-
priced for GPT-o3) if users were willing to accept longer response time. When
evaluating rather than deploying LLMs, this would often be tolerable and could
reduce the budget for a study. Finally, we note that the hyper-parameters do change
12https://web.archive.org/web/20251102073739/https://docs.cloud.google.com/
vertex-ai/generative-ai/docs/learn/prompts/adjust-parameter-values

12 Philippe J. Giabbanelli
TemperatureSimulation
Parameter
0.1 0.4 0.7 0.91234(a) LLM 1
TemperatureSimulation
Parameter
0.1 0.4 0.7 0.91234(b) LLM 2
TemperatureSimulation
Parameter
0.1 0.4 0.7 0.91234(c) LLM 3
Fig. 3: Illustrative response surfaces showing that the optimal temperature depends
on the LLM and interacts with simulation parameters.
over time as new LLMs can work differently. A prominent example is the rollout
of reasoning models, which can expose a hyper-parameter to control their depth of
‘thinking’13(e.g., by limiting their internal token generation).
2.3 Augmenting knowledge: RAG and LoRA
An LLM contains knowledge: we can ask questions and it will provide answers
derived from (but not necessarily identical to) its training. Since this knowledge is
produced through the LLM’s parameters14, it is calledparametric knowledge. At
inference time, an LLM may also benefit from faithfully retrieving additional knowl-
edge, which is calledcontextual knowledgeor non-parametric [64]. This is partic-
ularly relevant in a modeling and simulation context, where knowledge is stored
externally in forms that include graphs or ontologies for the conceptual model, texts
that dictate the scenarios for which a model is built and the results that we should
expect (e.g., policy documents, published studies), and structured simulation out-
puts. Intuitively, we may expect that more knowledge would be better: if the LLM
has access to more information, then surely it would help its reasoning mechanisms
to arrive at the right conclusion. The reality is more nuanced:performances may
13Our experiments suggest that reasoning level may influence an LLM’s sensitivity to targeted
content, thus reasoning may interact with guardrails. For example, at minimal reasoning levels,
GPT-5 rarely refused to portray an agent, whereas at medium reasoning levels refusals increased
substantially and made it difficult to use this LLM to power virtual agents.
14When considering the LLM’s knowledge, we should not think of a ‘fact’ as being stored in a
‘neuron’. There is no retrieval of a complete fact from one specific place in the model’s weights.
Rather,parametric knowledge is distributed among the weights and accessed indirectly through
computations, which can result in different facts from those used during training.

Large Language Models for Modeling & Simulation 13
actually degrade. For instance, Martinez and colleagues consideredNetLogo(often
used to code agent-based models) as a low-resource language where an LLM could
benefit from contextual knowledge to generate better code, but their evaluations
showed that the use of contextual knowledge made the code worse [78]. Under-
standing when RAG improves vs. degrades performance is a complex question. The
authors offer several potential explanations, such as having insufficient contextual
knowledge (about 74 pages), a noisy retrieval process, or an integration issue. The
latter is of particular interest, as a common misunderstanding would be to see an
LLM as relying on its knowledge when it’s good or retrieving facts when they are
better. Contextual knowledge does not just ‘add facts’: it enters the model as tokens
and is propagated through a routing system instead of being merged with parametric
knowledge [143]. In other words, contextual knowledge does not override parametric
knowledge with ‘better facts’: they coexist [143]. Empirical studies show that this
coexistence may be inefficient, as the output of LLMs may lean heavily in favor of
contextual knowledge even when it is conflicting or irrelevant with their parametric
knowledge [19]. This may call for different prompting techniques and a careful in-
tegration of contextual knowledge. For instance, to reduce the risk of a premature
over-reliance on contextual knowledge, we can emphasize ordering and separating
(the prompt can ask the LLM for its own knowledge andthenfor retrieval from
contextual knowledge) and then an evaluation step (to reason about the facts).
Contextual augmentationcan be achieved by expanding a prompt, for instance by
adding many more examples. This does not need a selection or retrieval mechanism,
but long-context prompts have their own challenges and they may increase latency
or cost. Contextual augmentation can also be achieved throughRetrieval-Augmented
Generation(RAG), in which an external retrieval component selects relevant in-
formation that is then incorporated into the LLM’s prompt. RAG adds a retrieval
stage between the user prompt and the standard generation pipeline (Figure 4)15.
Before discussing the mechanics of retrieval by RAG, recall that similarity-based
15Figure 4 shows a typical application-level architecture for RAG systems, in which embedding,
retrieval, and prompt-augmentation logic are tightly coupled within a single codebase. Recent
tooling efforts instead propose standardized interfaces, such as the open standard often referred
to asModel Context Protocols(MCPs), introduced by Anthropic in November 2024, which
decouple context provision from the application logic of the LLM. Under this design, external
systems (e.g., vector databases, document repositories, simulation outputs, or APIs) expose
contextual information through a well-defined protocol, allowing the LLM or its orchestrator to
request relevant knowledgewithout being directly wired to a specific retrieval implementation.
This shifts RAG from an application-specific approach with custom glue code to a modular
context service that can be reused, composed, and audited across systems. MCPs are particularly
aligned withMACH architectures(Microservices, API-first, Cloud-native, and Headless), which
emphasize composability, loose coupling, and the ability to rapidly integrate or replace third-party
services. MCPs primarily address practical concerns of interoperability, maintainability, and
governance, so they are more commonly discussed in practitioner-oriented tooling documen-
tation [58] than in research papers. Research studies have covered MCPs from a cybersecurity
perspective [32] or via innovations in architectures [73]. For further discussions, see the archived
MCP at https://web.archive.org/web/20260111203209/https://modelcontextprotocol.info/ or
https://web.archive.org/web/20251218100607/https://cloud.google.com/discover/what-is-model-
context-protocol for a contrast with classic RAG implementations.

14 Philippe J. Giabbanelli
retrieval requires converting text into numerical representations. In practice, doc-
uments and queries are mapped to high-dimensional vectors (embeddings) using a
pretrained embedding model, such that semantically related texts are close to one
another in this vector space. Retrieval then amounts to a nearest-neighbor search
over these vectors, rather than a symbolic or keyword-based lookup. In a typical
RAG workflow, the prompt (or a transformation thereof) is embedded and used to
query an external knowledge source (e.g., document corpus, database). The retrieved
items are then assembled into an augmented context, which is passed to the LLM
as additional tokens before tokenization and generation proceed as usual. From the
perspective of the LLM, retrieved knowledge is no different from the rest of the
prompt: it enters the model as contextual tokens and is processed through the same
attention and feed-forward mechanisms as any other input, while the model’s weights
remain fixed [64]. As illustrated in Figure 4, the RAG pipeline introduces several
additional design choices: how queries are formed, how documents are selected, and
how retrieved content is structured and ordered. Interfaces such as the web-portal for
GPT make it simple to use a RAG: we only have to drag-and-drop documents. But
from a design viewpoint, a RAG isn’t a simple connection of an LLM to a database:
it requires a careful combination of retrieval16, context construction, and prompt
integration strategies17.
The review by Sharma provides a taxonomy of RAG systems (c.f. Figure 1
in [106]) and a list of optimizations along with their strengths, limitations, and
sample applications (c.f. Table 1 in [106]). Choosing among the right options de-
pends on the task. Simply put, static, high-precision domains favor conservative
retrieval and tightly controlled context construction, whereas exploratory, multi-
step, or simulation-driven tasks benefit from adaptive retrieval, structured evidence,
and explicit evaluation stages. Once the task narrows down the potential options, an
experimental design can account for the interplay of RAGandLLM choices can
optimize the architecture. For instance, Singh and colleagues optimized the retrieval
approach (e.g., keyword-based BM25, semantic search query) and its parameters
(e.g., number of most relevant documents retrieved or𝑡𝑜𝑝−𝑘) along with the choice
of LLM with respect to accuracy, latency, and cost [110]. In contrast, the use of RAGs
is relatively new in the modeling and simulation community so just like the temper-
ature (section 2.2), its reporting and optimization varies across studies. Nonetheless,
when examining a sample of seven recent M&S studies using RAGs (Table 1), we
can make two observations.
First, RAGs have not been used primarily to correct erroneous outputs: rather,
RAGs serve to ground the model and simulation within a context, which usually
16A common design performs a single retrieval step prior to generation. If the LLM has to
answer a complex question that requires connecting multiple pieces of information (i.e.,multi-hop
reasoning), then this simple design may not be sufficient. Instead, an iterative or adaptive retrieval
can be used so that intermediate steps serve to refine queries or trigger additional retrieval [134, 5].
This process can also mitigate the potential effects of a poorly specified initial query. However, this
is a more complex design to articulate, and it would increase latency.
17Retrieved items may be concatenated, summarized, reranked, or labeled and separated from the
user query. Ordering and framing of retrieved evidence influences whether the model relies on
contextual knowledge or defaults to parametric priors, particularly in long-context settings [70, 19].

Large Language Models for Modeling & Simulation 15
User prompt
Query embedding
Document retrieval
Context construction
Tokenization
Neural network
(fixed weights)
Probability distribution
over next token
Decoding hyper-parameters
Token selection
Append token to contextThe textual input specifying the task and any constraints. In
RAG, this prompt is also used to form a retrieval query.
Transform the prompt (or a subset of it) into a vector repre-
sentation used to query the external knowledge.
Select relevant documents or data items from an external
corpus based on similarity, metadata, or hybrid criteria.
Assemble retrieved items into an augmented context, in-
cluding ordering, truncation, and formatting decisions.
Convert the augmented text into discrete tokens according
to a fixed vocabulary and encoding scheme.
A pretrained transformer that processes both parametric and
contextual information but whose weights remain fixed.
Normalized likelihoods over candidate next tokens, condi-
tioned on the augmented context.
Inference-time controls (e.g., temperature, top-𝑘, top-𝑝)
applied after contextual augmentation.
Selection of a token from the modified distribution, either
deterministically or stochastically.
The selected token is appended to the context; generation
then proceeds autoregressively.
Fig. 4: Retrieval-augmented generation (RAG) pipeline. Gray boxes denote stages
that are deterministic given fixed embeddings, indices, and model weights, while
blue boxes denote stages where stochasticity may be introduced through decoding
and sampling. Retrieved contextual knowledge is incorporated by augmenting the
prompt prior to tokenization and generation, not by modifying model parameters.
comes from an external corpus but can include simulation-generated artifacts to
ensure consistency18. In mission engineering and industrial modeling, the RAG
ensures that downstream reasoning remains consistent with upstream assumptions
or formal representations [113, 108]. In social simulations, the RAG conditions
the agents’ actions to prior interactions, thus providing a sense of context without
requiring global memory [28, 77]. Second, there is someawarenessthat using RAG
is not a binary switch: several aspects must be carefully chosen and prepared. For
example, Marigliano and Carley show how to curate the corpus that goes into the
RAG by taking three key steps, summarized in Box 1 [77]. However, the practice
so far is more a matter of ‘we chose a retrieval method and used it’ rather than ‘our
criteria led us to the following solutions and we optimized them as follows’. There is
18A wide range of sources has been used, including authoritative textual sources (e.g., doctrine
and mission artifacts), formalized domain knowledge (e.g., industrial semantics, physical process
descriptions), curated datasets (e.g., traffic trajectories), or dynamically evolving simulation state
(e.g., prior agent interactions).

16 Philippe J. Giabbanelli
thus still a need to examine the impact of a RAG through parameters such as chunk
size and overlap (and why that granularity matches the task), embedding model
choice and dimensionality, sensitivity to𝑘when choosing the𝑡𝑜𝑝−𝑘, similarity
metric choice, and so on. While we currently measure the impact of using a RAG
in terms of improving the overall LLM pipeline (e.g., comparing LLM only vs.
LLM+RAG [84]), an optimization would also benefit from isolating where the
improvements come from better evidence [28], more relevant evidence, or changing
the prompt to use a RAG.
Box 1. Preparing a corpus for RAG.
As part of a RAG pipeline, several steps can be taken to prepare the corpus.Chunk-
ingrefers to splitting documents into smaller, semantically coherent text segments
prior to embedding, rather than indexing entire documents as single units. This im-
proves retrieval precision by allowing the retriever toreturn only the locally relevant
portions of a source, reduces embedding dilution from multi-topic documents, and
mitigates context-window constraints during prompt construction.De-duplication
removes identical or near-identical text segments from the corpus, preventing redun-
dant passages from dominating similarity-based retrieval and biasing the generation
toward repeated narratives or overrepresented sources. This step is needed when
assembling a corpus from different reports that may overlap.Metadata enrichment
augments each text chunk with structured descriptors that can be used to constrain,
filter, or re-rank retrieval results. The descriptors depend on the application. Metadata
enables semantically targeted queries and improves interpretability and traceability
of retrieved evidence.
Using a RAG is not the only feasible way to augment knowledge without retraining
a full LLM.Parametric knowledgecan instead be adapted throughparameter-efficient
fine-tuning(PEFT) methods [53, 76], among whichLow-Rank Adaptation(LoRA)
has become particularly influential19. Rather than updating any of the pretrained
model weights, LoRA keeps the base LLM entirely frozen and injects additional
trainable low-rank matrices into selected linear transformations. These low-rank
parameters produce additive updates during the forward pass, enabling task- or
domain-specific adaptation while preserving the original model and dramatically
reducing the number of trainable parameters. The difference between LoRA and RAG
is where the knowledge comes in: RAG injects information at inference time through
additional context tokens, whereas LoRA augments the model’s parametric pathway
by adding persistent, trainable low-rank components toselected20transformations.
19LoRA-style adaptations are widely used in image and video generation models (which are not
LLMs) to capture characters (e.g., a LoRA trained on a specific pet), visual styles (e.g., a linocut or
watercolor aesthetic), or abstract concepts (e.g., laughing like a maniac). Multiple LoRA modules
can be composed, for example, to generate a linocut-style image of a laughing cat. Because each
LoRA corresponds to a separate file containing only the additional low-rank weights, practitioners
commonly refer to ‘a LoRA’ or ‘using LoRAs’, which implies modular and reusable artifacts.
20The weights introduced by a LoRA are always present during inference, but their influence on
generation is typicallyconditionalon the prompt. In practice, LoRAs are trained such that their
effects are activated by specific lexical or semantic cues. For example, if a LoRA is trained for

Large Language Models for Modeling & Simulation 17 Ref Simulation task LLM role Knowledge
sourceRetrieved content How retrieval is used Impact on simulation
[113] Mission engineer-
ing problem fram-
ingText synthesis and
reasoningMission reports,
doctrine, lessons
learnedUnstructured textual
documentsRetrieved passages are inserted into the prompt
to ground early problem statements, assumptions,
and requirements, with explicit traceability to
sourcesReduces hallucinated assumptions, im-
proves completeness of mission defini-
tions, and supports stakeholder alignment
[28] Social media sim-
ulation of politi-
cal expression in
a Twitter-like net-
workEach LLM agent
reasons about
whether to post,
reshare, or remain
inactive, and
generates contentSimulation-
internal interac-
tion history from
agent-generated
posts and run-
time actions (no
external corpus)Previously generated
agent posts, reshared
content, and interac-
tion records stored
in a continuously up-
dated vector databaseSemantic retrieval selects a limited, contextually
relevant subset of past agent-generated content to
expose each agent at every timestep, functioning as
a recommendation mechanism (preference-based
or random) while keeping prompt length boundedRetrieval strategy directly shapes interac-
tion patterns and emergent network struc-
ture: preference-based retrieval increases
engagement, homophily, and echo cham-
ber formation, while random retrieval re-
duces polarization but lowers interaction
intensity
[10] Interaction with
closed-source
simulation soft-
wareNatural-language
interface and
guidanceSoftware manu-
als and usage
logsDocumentation ex-
cerpts and examplesRetrieved documentation contextualizes user
queries, enabling the LLM to explain commands,
parameters, and workflows without modifying the
simulatorImproves usability of simulation tools and
reduces trial-and-error during model setup
[77] Social media opin-
ion and belief dy-
namics at popula-
tion scaleSynthetic commu-
nity and persona
generation; opin-
ion and emotion
modelingCountry-specific
discourse
corpora, de-
mographic
and contextual
sourcesDiscourse frag-
ments encoding
national narratives,
community-relevant
framings, and topic-
specific salience cuesRetrieved passages ground a staged generation
pipeline: first constraining community profiles
(demographics, ideology, topic salience), then
conditioning persona construction and opinion
sampling; salience values modulate downstream
variance rather than overwriting model knowledgeImproves demographic plausibility, ide-
ological coherence, and psychologically
grounded diversity; reduces free-form hal-
lucination and produces structured opin-
ion variance aligned with social-science
expectations
[108] Industrial system
modeling and in-
teroperabilitySemantic interpre-
tation and model
constructionAsset Admin-
istration Shell
repositoriesStructured entities, at-
tributes, and relationsRetrieved AAS elements preserve formal seman-
tics and are summarized into prompts that guide
consistent model generationImproves semantic correctness and inter-
operability of generated industrial models
[84] Wildfire agent-
based spatial
simulationCode generation
and behavioral
rule designWildfire model-
ing literatureDescriptions of phys-
ical processes and in-
fluence factorsRetrieved domain knowledge informs how the
LLM formulates agent behavior rules and param-
eters during simulation code generationProduces spatial spread patterns closer to
physics-based simulators and real wildfire
data
[23] Traffic scenario
generationScenario synthesis Traffic scenario
datasetsEncoded trajectories
and interactionsRetrieved scenarios condition the generative pro-
cess, guiding the construction of new scenarios
with controlled density and interactionsIncreases diversity, realism, and controlla-
bility of generated simulation scenarios
Table 1: Use of retrieval-augmented generation in simulation-related studies.

18 Philippe J. Giabbanelli
LoRA is not intended as a general-purpose mechanism to ‘inject facts’: rather, it
specializes the parametric behavior of an LLM for a stable context. For instance, if
modelers repeatedly request that an LLM translates natural language into the same
type of conceptual model (e.g., causal loop diagram), or critique models from the
same paradigm (e.g., agent-based model), then a LoRA can be trained from curated
examples (e.g., learning a mapping given pairs of texts and associated causal loop
diagrams) to use the appropriate terminology or organize the output according to
accepted conventions. Alternatively, a LoRA could encode domain regularities: the
concepts, constraints, and interactions that characterize an application area such as
wildfire spread, flooding, or epidemics. It would not provide authoritative references
or scenario-specific facts (that would need RAG21), but it would ensure the use
of domain-appropriate abstractions. LoRAs can be combined22, for instance by
combining a LoRA from a paradigm (e.g., cellular automata) with a LoRA for the
application domain (e.g., wildfire spread). The low-rank update of a LoRA is typically
scaled by a tunable coefficient at inference time, called ‘weight’. That weight can
be larger than 1 to make the LoRA’s specialization matter more than it did during
training, for instance if it was trained conservatively (e.g., limited data, low rank)
or we want to force an effect even when the prompt is ambiguous. When multiple
LoRAs are active, adjusting these weights can bias the generation toward one set of
learned adaptations over another. Note that a weight larger than 1 does not make a
LoRA more correct, it only makes it moreforcefulby becoming (over)sensitive to
trigger tokens or suppression legitimate alternatives.
agent-based modeling but the prompt concerns system dynamics, the LoRA’s influence may remain
effectively dormant. Rather than repeatedly changing the architecture by dynamically enabling or
disabling LoRAs, practitioners commonly rely ontrigger wordsto activate certain LoRAs. For
instance, the LoRA on agent-based modeling can be included all the time, but only the phrase
‘agent-based model’ in the prompt would be likely to activate the learned adaptations. The trigger
words tend to be short and specialized, rather than a full sentence such as “why don’t you pass
the time by playing solitaire?”. In short, LoRA-encoded knowledge is persistent in the model
parameters, but only expressed when the prompt triggers the corresponding learned patterns.
21The nuance can be subtle, particularly when training a LoRA to frame the what-if questions
that go into a model. A LoRA could help to articulate the scenarios in a consistent format. If a
simulation workflow repeatedly needs to perform sensitivity analyses, then the LoRA could encode
their style. This is not about supplying facts or evidence (which would be the RAG’s role), but about
enforcing a consistent structure for how scenarios are articulated.
22When combining LoRAs, we should avoid triggering multiple adaptations that encode incom-
patible assumptions or conventions. This issue is particularly salient in hybrid modeling, where
multiple paradigms (e.g., agent-based modeling and cellular automata) are used together [82]. If
separate LoRAs have been trained on paradigm-specific notions such as ‘validation’ or ‘state up-
date’ and then both are activated, then it is difficult to know whether one LoRA would dominate,
whether their effects would blend into an uncontrolled hybrid, or whether the resulting behavior
would depend sensitively on prompt phrasing and training artifacts. The dominance of a LoRA may
be lowered by decreasing its weight, but this is a coarse approach that redistributes influence without
fundamentally resolving ambiguity. This is not only a technical limitation of LoRA composition,
but also reflects a deeper challenge in aspects of M&S that may lack normalized expectations about
how concepts should be reconciled across paradigms. As a result, conflicting LoRAs may echo the
ambiguity in the field rather than resolve it.

Large Language Models for Modeling & Simulation 19Aspect RAG LoRA Adapters Selection-based
Primary pur-
poseInject external knowl-
edge at inference timeSpecialize model behavior via
low-rank parametric adaptationSpecialize model behavior
via new trainable layersRoute computation through
existing model capacity
Where changes
occurInput/context pathway
(tokens)Parametric pathway (additive
weight updates)Architectural pathway (in-
serted layers)Control pathway (gating,
masking, routing)
Adds trainable
parametersNo Yes (low-rank matrices) Yes (bottleneck layers) Typically no (sometimes
small gating params)
Modifies archi-
tectureNo No Yes No
Knowledge per-
sistenceTransient (per prompt) Persistent (stored in parameters) Persistent (stored in parame-
ters)Persistent but implicit (in
routing policy)
Dependency on
prompt contentHigh (retrieval query de-
termines knowledge)Medium–high (trigger tokens
modulate effect)Medium (activated when
adapter is enabled)High (selection often condi-
tioned on input/task)
Control mecha-
nismAt token/content level Use scaling weights Enable/disable Discrete or soft via routing
decisions
Typical knowl-
edge encodedFacts, documents, evi-
dence, scenario-specific
dataConventions, styles, domain reg-
ularitiesTask-specific transformations Task regimes, behavioral
modes
Suitability for
changing sce-
nariosHigh Low–medium Low–medium Medium
Suitability for
stable domainsMedium High High Medium
Interpretability
for M&S usersHigh (retrieved sources
inspectable)Medium (effects distributed) High (explicit modules) Low–medium (routing often
opaque)
Risk profile Missing/irrelevant re-
trieval; context overloadOver-specialization; interference
between LoRAsArchitectural complexity;
stacking effectsUnintended routing; brittle
gating
Potential use in
M&SSupplying scenario facts,
references, policiesEnforcing modeling paradigms or
domain abstractionsTask-specific transformation
pipelinesSwitching between known
simulation regimes
Table 2: RAG changes what the modelseesat inference time; LoRA and adapters
change how the modelthinksby shaping its parametric behavior; selection-based
methods change which parts of the model areused. These approaches can be used to
provide exposure to scenario-specific information (RAG), encode stable modeling
conventions (LoRA/adapters), or switch between model configurations.

20 Philippe J. Giabbanelli
A complete inventory of methods to augment knowledge is beyond the scope of
this article, particularly as only a few have been used (or considered for application)
in M&S. Table 2 provides a summary based on our experience, but the suitability of
these methods across M&S tasks should be re-assessed regularly as this is a quickly
changing landscape. Two other well-defined categories of methods that we have
not covered as much currently, but whose prominence could change in the future,
include adapter and selection methods. LoRA reshapes existing transformations by
additive updatesto pretrained weight matrices (i.e.,𝑊 𝑒𝑓𝑓𝑒𝑐𝑡𝑖𝑣𝑒=𝑊𝑏𝑎𝑠𝑒+Δ𝑊𝐿𝑜𝑅𝐴 ),
so they can be merged back into the base model23. In contrast, adapters introduce
auxiliary parameterized mechanisms that modulate or reroute computation while
leaving the original weights untouched and unmerged24.Selectionmethods do not
introduce new transformations of the signal (like adapters) or augment existing
weight matrices (like LoRAs): they choose what to activate or freeze among existing
parameters. In other words, they make a selective use of the existing model capacity
by routing computations through different subsets.
3 Ignoring the impact of non-determinism
3.1 Why is there non-determinism in LLMs?
Non-determinism means that even if we use the same prompt and (what appears to
be) the same LLM environment, there is apossibilitythat the result is different. This
does not mean that the resultwillbe different each time, so simply running the prompt
a few times and seeing the same result does not provide an argument to conclude
that the process is deterministic. In fact, there are many mechanisms that produce
non-determinism with LLMs. We suggest that framing LLMs as mere ‘stochastic
parrots’ and attributing all non-determinism to the model is neither constructive
nor entirely accurate. Without precisely understanding sources of non-determinism,
we may just brush off some outputs as ‘errors’ while the general population may
take them as evidence of the LLM being ‘autonomous’ [93]. To be precise, we
should avoid conflating non-determinism, which can broadly be categorized into
inference non-determinism(e.g., implementation optimization) anddistributional
bias(what sequences the model assigns high probability to). Some of these sources
of non-determinism can be fixed, but it certainly takes more work than just setting
a temperature parameter to 0 – as several empirical studies show that high variation
in the output still happens [88, 6]. Even when non-determinism cannot be tamed,
inspecting its mechanisms may reveal more patterns than just ‘randomness’, thus
23This is common practice in text-to-image and text-to-video models, in which several LoRAs are
released individually and preferred ones are eventually packaged into a new version of the model.
24Adapters were introduced as ‘inserting new trainable layers’, but the field gradually shifted to a
broader definition that does not require explicitly inserted layers. For example, the Structured MOd-
ulation Adapter (SMoA) modulates existing transformations through learned, high-rank structured
parameters, yet remains an adapter because its effects cannot be absorbed into the base model [71].

Large Language Models for Modeling & Simulation 21
providing valuable insight into an LLM’s behavior [20]. For example, experiments
support the (distributional bias) hypothesis that “LLMs are sensitive to the probability
of the sequences they must produce”: they are more likely to give the right answer
if it has a high probability of occurrence, even if the task is deterministic [79]. In
short, correctness correlates with sequence likelihood, not task determinism.
It is well-known that LLMs model conditional token distributions [92, 50]. Again,
this is not to say that they are merely ‘stochastic parrots’ that excel at memorizing
vast amounts of training data and regurgitating them with a bit of randomness: the
ability of newer LLMs to plan and perform reasoning goes beyond just repeating
patterns from their training set [72]. The many reasons for which non-determinism
happens during inference are perhaps less well-known, or even unexpected for some
readers, thus we focus on these cases.
When combining LLMs with M&S, we rarely use ChatGPT and a web browser.
Rather, we automate some of the operations (e.g., using the LLM to generate a
conceptual model from a corpus) through code such as a Jupyter Notebook operating
in Python. If we use a single LLM such as OpenAI’s GPT, then the entry point to
access the model is the OpenAI API. However, in practice, we often need to use
several LLMs, either for optimization (which one is best for the task? or provides a
good tradeoff between task accuracy and cost?) and/or for evaluation (to which extent
is the conceptual model produced by the LLM shaped by the choice of model?). As
a result, it is common to use a centralized platform that offers access to many LLMs
across providers through the same API. While this may at first seem like a point of
detail that would be found in an implementation footnote (or even just omitted), it
can have a surprising effect on non-determinism. For a given model (e.g., DeepSeek-
R1-Distill-Llama-70B), a centralized platform such asOpenRouterprovides access
to several providers. They have different prices for inputs and outputs, different
latencies25, and may impose different limits on the maximum size of the output.
However, (i) these providers may run the models differently and (ii) OpenRouter
routes the request toanyof the available providers that serves the same base model
and satisfies a user’s needs for prompt size and parameters. So when we send several
prompts, some may be executed differently than others.
A difference in execution does not necessarily mean a difference in results: for
instance, speed-up ‘tricks’ such as predicting multiple tokens in parallel using a
smaller/faster model (i.e., speculative decoding) can provide a lossless acceleration of
LLMs during inference [130, 115], so theoretically they would only affect efficiency
rather not correctness. But there are also three differences that affect the output. First,
LLM quantizationrepresents a model’s weights and activations with lower-precision
numerical formats (e.g., int8, fp8, bf16 instead of fp32) to reduce memory use and
speed up inference, which introduces ‘small’ numerical approximation errors. In
our example of DeepSeek R1 Distill Llama 70B, some providers use fp8, some use
bf16, and some do not say. Just with quantization alone, even using the same prompt
25For instance, the report on Qwen3-Omni details a set of techniques to achieve ultra low-
latency [133]. Such models can be used for ‘real-time’ audio/video tasks such as transcription.

22 Philippe J. Giabbanelli
and seed can lead the token sampling to diverge26. Second, providers may perform
different optimizations to speed up computations or reduce memory consumption for
attention, which is the core mechanism that lets the LLM account for the previous
tokens when generating the next token27. This is not a change of precision (as
in quantization) but a potentially different numerical execution order: sums are
accumulated in different sequences, thus rounding happens at different points. In
other words, quantization changes the numbers that we compute with, but optimizing
the attention changes how we compute the numbers. Third, LLMs recompute the
same keys and values for previous tokens at each time step, so they have to manage
thekey-value cache(KV cache). The management strategy can make a difference
in long-context prompts: a simple sliding window drops the oldest cached key/value
pair when memory gets full, while a more advanced strategy would evict content
based on its expected importance to preserve long-range dependencies and thus
affect the overall inference quality [124, 60].
An additional and perhaps unexpected situation is that asking for the same LLM
does not mean that we get exactly the same weights or behavior. There are two
reasons for this situation. First, there is (silent)forwarding: LLMs eventually become
deprecated so providers such asDeepInfraapply their model deprecation policy by
which older models are removed and (to avoid breaking a user’s code) calls are then
forwarded to another model. Modelers may see that their code just works ‘as usual’
without realizing that it is now being serviced by another LLM. Second, a name is just
analias: asking for ‘GPT-5’ does not refer to one set of weights and policies forever,
but rather to whatever was in place at the time that the call was made. Providers may
update policies, add moderation, or route prompts differently based on account tiers
and geography (which can also determine which policies are applicable). In a highly
cited study, Chen and colleagues reported the opacity in determining when and how
some LLMs were updated, with clear differences in performances for the same LLM
by name (e.g., GPT-4) over several months. For instance, GPT-4 answered sensitive
survey questions more in March than in June [18].
3.2 Evaluating the impact of non-determinism
The goal is not necessarily to reduce non-determinism, because other considerations
can be more important for practical deployment. For example, costs28or latency may
26Since classical methods (e.g., GPTQ, AWQ, QuIP), new methods (e.g., QEP, GPTAQ, LoaQ)
have been developed to control error propagation across layers and reduce the accumulation of
quantization errors [54, 67].
27As an example of a highly cited optimization,FlashAttentionis a memory- and compute-
efficient implementation of transformer attention [22]. It changes the computation of matrices to
minimize intermediate memory writes.
28Cost can relate to variability across runs, particularly when usingcachingto retrieve the (same)
answers to queries that have already been computed (i.e., ‘replaying’ prior outputs). For example,
providers such as DeepSeek reduce its costs by orders of magnitudewhen users are willingto use
the cacheand there is a cache hitas the prefix of a new prompt sufficiently matches the prefix

Large Language Models for Modeling & Simulation 23
be the main objectives for a large-scale solution that serves a large number of users
in low-risk environments. But even if a certain application context is ‘willing’ to
live with some non-determinism in the outputs, this should be an informed decision:
to what extent are key performance measures affected by stochasticity arising from
sampling, inference, or system-level variability? Anablation studyis a common ap-
proach to decompose the performance of a system based on whether certain parts are
included (e.g., whether a RAG is used, whether additional examples are provided in
the prompt). However, that does not apply here since stochasticity is neither a binary
design choice nor fully under user control (unlike e.g. whether or not to use a RAG),
and its effects are inherently variable across runs. Alternatively, a simple statistical
approach is to report the average and confidence intervals for each performance
measurement across runs (also known as ’repeats’). However, it is well-known in
the modeling and simulation literature that the number of runs cannot be arbitrarily
set e.g. to 10, 100, or 500: we need to determine how many runs are statistically
sufficient otherwise the confidence intervals are not meaningful [95][pp. 182–193].
This has motivated the development of other statistical approaches for LLMs that
approximate the distribution of a model’s performance using bootstrapping [30].
Nonetheless, the implicit assumption that performance varies smoothly around a
central value is questionable in practice since empirical results show that perfor-
mance distributions can be multimodal and heavy-tailed, where best-case runs give
the illusion of high performances and hide the massive issues caused by worst-case
runs [112, 7]. For example, an LLM used as part of a logistics simulation may meet
its average resolution targets, yet generate some irrecoverable failures that violate
service-level guarantees. This motivates both the development of other metrics (see
Box 2) and the careful use of Design of Experiments (DoE), which have been a
staple of the modeling and simulation literature for decades.
Since Sanchez and colleagues have provided several tutorials on the different
types of DoE used in modeling and simulation [100, 101], we briefly coverhow
an experimental design applies to the assessment of non-determinism in LLMs for
simulations. A model is composed of different parameters: for example, an agent-
based model may have a measure of diversity in the population, or an influence
threshold after which an agent’s opinion would start to align on its peers. The LLM
component is also made up of several parts, such as whether to use a RAG or to add
examples to the prompt. A DoE can decompose variations in the response variable
(e.g., TAR@N or WorstAcc@N per Box 2) onto the individualand interacting
effects of these parameters, design elements, and stochastic effects. In afull factorial
experiment, we consider all combinations: if there are three levels of diversity and
four values for the influence threshold, then we cover all 3×4=12 cases. In a
of a previously computed prompt. However, caching can work very differently across providers.
For OpenAI, prompt caching is automatic since GPT-4o. In contrast, Anthropic’s Claude exposes
explicit caching controls via a dedicated field and allows users to extend the default caching window
for a fee, thus providing fine-grained control over cost and variability. Note that caching can give
theillusionof determinism by repeatedly returning identical outputs over a limited time window,
so caching should bedisabled(either explicitly or by waiting between repeated prompts) when
assessing a system’s variability.

24 Philippe J. Giabbanelli
simplified and commonly used design, we limit each aspect to two values. So if
we have 4 factors (agents’ diversity, influence threshold, RAG, examples) set to 2
levels each (high/low diversity, high/low threshold, presence/absence of RAG and
examples), then we need to study 2×2×2×2=24combinations. Generalizing this
example, a 2𝑘𝑟factorial designwith replications can study the𝑘binary factors across
𝑟runs for each combination. Non-determinism is captured via replication (Table 3).
Variability in the output measure would be decomposed on the individual factors and
their interactions in pairs (e.g., diversityandrandomness, diversity and threshold,
RAG and randomness), groups of three, and groups of four29. If a 2𝑘𝑟exceeds either
the available budget to repeatedly query LLMs or the time devoted to gathering
experimental results, then an alternativefractional factorial designcan reduce the
number of combinations at the expense of lesser precision in the analysis30.
As a practical case study, we used factorial designs to measure the amount of
variability when employing LLMs for the conceptual modeling task of combining
two models. That is, given several conceptual models, the LLM had to find how they
could be merged by accounting for semantic variability (different words referring to
the same concept). We found that the extent to which non-determinism drives the
results depended onhowthe problem was tackledandon the LLM. Using a simple
method that tries to find whether each concept of one model is directly equivalent
to concepts of another model (e.g., given the context, can stress be merged with
cortisol?) produced negligible amounts of randomness. Performances were primarily
driven by how the conceptual model was represented in the prompt (e.g., as an array
or list) and by the instructions (providing explanations, examples, counter-examples).
In contrast, a more sophisticated approach in which synonyms and antonyms were
gradually derived (e.g., stress relates to anxiety, which is measured by cortisol
levels, so cortisol is a match) had a massive amount of randomness, as most of the
variability in the results (from 53% with GPT to 95% with DeepSeek) were attributed
to non-determinism.Ignoring the presence of randomnessby forgetting to perform
repeats or analyze them would have erroneously suggested that design aspects such
as providing counter-examples had a high impact onto the performance, whereas
appropriately measuring the effect of randomness revealed that modelers had very
little control on performances.
29For examples of a factorial design in a modeling and simulation study, see [65, 74]. For code that
allows to run these experiments in parallel and supports the scalability of the analysis, see [62].
30Higher-order interactions (i.e., the synergistic effect attributable to agroupof several factors)
tend to have less of a contribution than lower-level interactions. Typically, single factors and pairs
explain most of the variance, as it is relatively unusual to observe effects that areonlyobtained
for specific combinations in the values of three or more parameters. A fractional factorial design
wouldconfound(i.e., ‘mix’) the variance that can be attributed to low- and high-order interactions.
For example, a single number accounts for the effects of both𝐴𝐷or the larger combination of
four parameters𝐴𝐵𝐶𝐷, but we expect that it mostly represents𝐴𝐷. A good fractional factorial
design would avoid confounding groups of similar sizes (e.g., two and three factors). The quality
of a design is evaluated by itsresolution, that is, the distance between the groups of factors that are
confounded. The notation indicates how many parametersshouldhave been used, how many were
fixed to cut experimental costs, and the quality of the design (denoted in roman numerals). For
instance, 23−1
𝐼𝐼𝐼is fractional factorial design that should have had 3 parameters but fixed one (which
reduces the number of experiments by half) and has a design resolution of III.

Large Language Models for Modeling & Simulation 25
Box 2. Characterizing systems: beyond means and confidence intervals
Consider two LLMs (𝐿𝐿𝑀 𝐴and𝐿𝐿𝑀𝐵) used to set the behaviors of agents in a
simulation. They are used to predict the next action for each of the 100 agents and
each simulation is performed 20 times. Accuracy is measured as the percentage of
correct actions across runs based on a ground truth dataset. For𝐿𝐿𝑀 𝐴, it gets the
same80 agents right each time (e.g., it knows exactly what to do for a subset of
agents based on environmental state or role). In contrast,𝐿𝐿𝑀 𝐵has an 80% chance
of being correct for the behavior of an agent, thus which ones are accurate vary
randomly (correctness is not patterned by the agents’ attributes). They have the same
average performance, yet𝐿𝐿𝑀 𝐴yields fully reproducible results while𝐿𝐿𝑀 𝐵is
very unstable.
Another measure is to compare the best and the worst accuracy of each LLM across
the runs. Here,WorstAcc@20(must always be correct for each agent across runs)
is 80% for𝐿𝐿𝑀 𝐴and it tends towards 0% for𝐿𝐿𝑀 𝐵. In contrast, theBestAcc@20
(must be correct at least once for each agent across runs) is 80% for𝐿𝐿𝑀 𝐴but 100%
for𝐿𝐿𝑀𝐵. The differentialΔ𝐴𝑐𝑐=𝐵𝑒𝑠𝑡𝐴𝑐𝑐@20−𝑊𝑜𝑟𝑠𝑡𝐴𝑐𝑐@20=100% for
𝐿𝐿𝑀𝐵indicates its high variability [7].
Alternatively, we can compute the Total agreement rate@N (TAR@N), defined as
the percentage of answers across N runs that are identical, regardless of whether
they are correct [7]. This may be further divided into whether the answers are
identical(string matching or ‘surface determinism’) orequivalent(this ‘decision
determinism’ can further be divided based on simple parsing or semantic distance).
Here, we would see a TAR@20 of 100% for𝐿𝐿𝑀 𝐴and about 0% for𝐿𝐿𝑀 𝐵.
3.3 Mitigating the impact of non-determinism
Ifthe analysis in section 3.2 reveals that non-determinism plays a larger role in deter-
mining the performances that is desirable for the modelers or model commissioners,
thenactions can be taken. Otherwise, there is no need to address a non-existing
problem. Actions do not necessarily mean rushing into active mitigation strategies:
further analyses could be used to better locate the problem. For instance, if we find
that 60% of the variance in performance comes from non-determinism, that could
be due to several phenomena: is it inference non-determinism (e.g., the routing ser-
vice uses slightly different LLMs unknowingly to the modeler) or distributional bias
(e.g., the LLM itself is the issue)? To isolate the source of the error, the experimental
design can be expanded, for instance by adding another binary factor to represent
whether we let the routing service use any version of the LLM (which reduces costs
and latency) or whether we should use the same one consistently. Once we have
gathered information to identify where we need to intervene, then several mitigation
strategies can be taken as follows.
If the issue comes from the dynamic routing to different versions of an LLM, it
cannot be addressed by setting specifications and requesting that only a perfect match

26 Philippe J. Giabbanelli
Table 3: The standard way to conduct a factorial analysis is to create a table with
the factors (e.g., A=agents’ diversity, B=influence threshold), their interactions (we
omit groups of 3 and the group of 4 for brevity), and the response variable𝑦across
runs. The effect of non-determinism is not a factor by itself; rather, it is computed
by examining the variation in the response variable. The coded levels -1 and 1 are
used to decompose the effect (by cross-product of the columns then sums) and they
correspond to the low and high level of each factor (e.g., low agents’ diversity is -1
and high diversity is 1). The factors are set as a sign table then other columns (𝐴𝐵,
𝐴𝐶, etc) are populated by multiplying (e.g., the content of𝐴𝐵is𝐴×𝐵).
Run ABCDABACADBCBDCD𝑦1𝑦2𝑦3
1 -1-1-1-1+1+1+1+1+1+10.72 0.70 0.71
2 +1-1-1-1-1-1-1+1+1+10.75 0.77 0.74
3 -1+1-1-1-1+1+1-1-1+10.68 0.66 0.67
4 +1+1-1-1+1-1-1-1-1+10.79 0.80 0.78
5 -1-1+1-1+1-1+1-1+1-10.73 0.71 0.72
6 +1-1+1-1-1+1-1-1+1-10.81 0.82 0.80
7 -1+1+1-1-1-1+1+1-1-10.69 0.68 0.70
8 +1+1+1-1+1+1-1+1-1-10.84 0.83 0.85
9 -1-1-1+1+1+1-1+1-1-10.70 0.69 0.71
10 +1-1-1+1-1-1+1+1-1-10.76 0.78 0.77
11 -1+1-1+1-1+1-1-1+1-10.67 0.66 0.65
12 +1+1-1+1+1-1+1-1+1-10.82 0.83 0.81
13 -1-1+1+1+1-1-1-1-1+10.72 0.73 0.71
14 +1-1+1+1-1+1+1-1-1+10.86 0.87 0.85
15 -1+1+1+1-1-1-1+1+1+10.71 0.70 0.72
16 +1+1+1+1+1+1+1+1+1+10.90 0.91 0.89
be used. Indeed, not all providers expose all specifications of their implementations
(e.g., we do not always know whether an LLM is in fp8 or fp32 form). Rather, the
solution is toexplicitly set one provider instead of dynamically changing between
providers. If the issue comes from the LLM being changed (e.g., automatic forward-
ing of deprecated models, model updates) then the solution includes self-hosting
an open-weight model (which is limited by computing power and excludes popular
options such as the latest GPT) or using a fully versioned ID thatguaranteesthe
same model. Long, immutable version IDs are not offered by all providers (e.g., it
is not the practice of OpenAI). Anthropic, which builds Claude, provides explicit,
dated model IDs (e.g., claude-3-opus-20240229) and currently guarantees that if
users refer to an ID then they get the same weights and behaviors.

Large Language Models for Modeling & Simulation 27
4 ‘Working’ is not enough: avoiding inferior science with LLMs
4.1 Risks of LLMs as a one-stop shop
When studying on a scientific topic, it is expected that we engage directly with
the literature. Reading papers is a process through which we encounter competing
methods, contradictory findings, and unresolved debates, and synthesize the evidence
base to understand not only what supports an argument but also what challenges it.
Against this backdrop, a visible (and sometimes normalized) misuse of LLMs is
their deployment to supply references. For example, a premier research conference
such as NeurIPS had over 50 accepted papers containing AI-fabricated citations in
2025 [42]. Fundamentally, using LLMs to add references or delegating reading tasks
to LLMs means that authors are not exposed to potential evidence that contradicts
them. Reframing scholarship as a search for confirmatory citations via LLMs creates
a form ofconfirmatory biasat scale. While it may seem that nudging LLMs to
produce real and relevant references (e.g., by using different prompts and automated
or manual verification [69]) solves the problem, this only satisfies a minimal notion
of what it means to ‘work’ rather than addressing the standards of scientific inquiry.
Even if an LLM appears to ‘work’ with real references, investigating their use has
revealed that they do not necessarily support (and sometimes even contradict) the
arguments for which they are used [129]. And even if LLMs fetch real papers and use
them well, this is a sub-optimal process that replaces expert judgment grounded in
critical synthesis of the literature with automated, argument-serving retrieval. This
pattern extends well beyond citations: in many scientific tasks, an LLM may appear
to ‘work’ at a superficial level while producing outcomes that are epistemically or
methodologically sub-optimal. For example, instead of writing a Python script or
using a statistical package, it is possible to ask an LLM to analyze data. As with
references, it may seem to ‘work’ by returning the correct resulttoday, but due to
non-determinism (see section 3), it is possible that it returns a different result next
time. We may fix this problem by asking the LLM to produce the code so that we
can execute it ourself, thus guaranteeing that the same result would be produced
each time. While the code may seem to ‘work’, there is growing evidence that AI-
generated code can be correct yet of poor quality and with security flaws [97, 119].
Overall, treating surface-level correctness as sufficient evidence that an LLM ‘works’
poses a risk of sub-optimal scientific processes.
M&S is not immune to problematic uses of LLMs, which may be driven by
a broader pressure to increase productivity along with the appeal of LLMs as a
one-stop shop. Recent work on LLM-generated simulation models illustrates how
difficult it is to define and automatically enforce what it means for a model to ‘work’.
For example, M ¨oltner and colleagues showed that a conjecture-based evaluation
pipeline failed when trying to have an LLM conjecture the expected properties of
a model (from a text description) and evaluate code generation accordingly [81].
The reasons for this failure highlight an interesting difference between an LLM
approach to generating and validating simulation code compared to the approach

28 Philippe J. Giabbanelli
taken by a (human) modeler. A human modeler would read a textual description of
a problem, derive a conceptual model, implement that model, and then evaluate the
implementation with respect to that same interpretation. In contrast, when LLMs are
used, the conjecture and the simulation code are produced by separate LLM calls that
independently interpret the same text, with no guarantee that they converge on the
same conceptual model. As a result, validation can fail even when the implementation
is reasonable, because it became a test of interpretiveconsistency(which is not an
LLM’s strength) rather than modeladequacy(which is the actual goal)31.
4.2 Translating rather than delegating with LLMs
A more subtle risk lies in using LLMs for tasks that are already well supported by
specialized tools. Consider the effort required to guide an LLM to compare a model’s
implementation with a specification in order to provide some form of certification.
That could be a research article showing that certain prompting patterns or RAGs
improve the results, and then other researchers may reuse this system. But there are
already solutions for this problem: model checkers or test generators [12]. While we
may all agree in principles that weshouldalways use the best tool for a task, we
do not necessarily knowhowto use such tools as we cannot be experts in every-
thing. However, the wrong conclusion would be to rely on inferior solutions simply
because LLMs are easier to use. Rather than attempting to reimplement specialized
capabilities through LLMs, we posit that LLMs are better positioned as translators:
mapping informal requirements into the formal languages required by specialized
tools, and translating those tools’ outputs back into forms that are accessible to
modelers [38]. For instance, a modeler may ask, “does my implementation behave
reasonably?”, yet this is not the kind of question that model-checking tools take
as input. Instead, such tools verify formally specified properties (e.g., reachability
of some states, invariants) expressed in languages such as linear temporal logic or
modal logics. In this context, LLMs are more appropriately used to map natural
language requirements into candidate formal properties than to attempt to perform
model checking directly. This is illustrated by theGIVUPtool [86]: the authors use
the LLM to extract structured information about model processes and properties into
31As an example, consider the following text specification: “Agents are located on a network
and hold binary opinions. At each time step, agents interact with their neighbors and may update
their opinion based on these interactions. There is no global coordination or central authority.”
Calling the LLM to generate a conjecture could result in considering that agents interact with
all neighbors at each step and align their opinion with the local majority. The simulations should
thus show rapid convergence toward consensus within connected components. In a separate call,
an LLM may implement the same textual description differently, by having each agent interact
with one randomly selected neighbor per time step. Neither interpretation is ‘invalid’: they both
could produce simulation outputs that match real-world trends over different time scales. But
the interpretations assume different conceptual models and this incompatibility results in a false
negative: the simulated model would be categorized as wrong because it converges more slowly.

Large Language Models for Modeling & Simulation 29
temporal logic, then they perform model checking through an external specialized
tool by verifying that the model satisfies a given temporal logic property.
Figure 5 illustrates the role of LLMs as translators that connect informal mod-
eling requirements to specialized formal tools. Starting from natural-language re-
quirements, auxiliary documentation, and potentially several models (finished or
in-progress), the LLM must first infer which formal representations and tools are
appropriate for a given task. This involves translating informal descriptions into
the specific input languages expected by those tools (Figure 5a). In practice, this
translation step is often nontrivial: many of these representations can be considered
low-resource languagesfor LLMs, and initial attempts may fail due to syntactic or
semantic mismatches. Rather than treating such failures as definitive and giving up
on using a specialized tool or requesting more training data, we suggest to leverage
feedback from the tools when available (Figure 5b). If tools provide parser errors
or violated constraints, then LLMs could use this feedback to iteratively refine the
translation. In this case, the LLM plays an orchestration role by dispatching candidate
representations to the appropriate tools, interpreting error messages, and adjusting
subsequent translations through round-tripping until a valid formal input is obtained
(Figure 5c). Once the specialized tools produce results, the LLM can then translate
these outputs back into forms that are accessible and meaningful to the modeler. In
this approach, LLMs are not re-implementing methods: they are mediating between
heterogeneous representations and tools.
Acting as a translator does not mean that the LLM just converts the syntax from a
modeler’s description onto the target representation of a specialized tool. We argue
that a key asset of LLMs in this situation is theirknowledge model: they can bridge
the gap that is likely to occur between a modeler’s description, which is sufficient
within the application context, and a tool’s description, which must be complete. For
example, some (low-level) rules may seem obvious to a modeler in a given application
context and they would not be stated, but the specialized tool cannot make inferences
without knowing them, so the LLM can suggest such missing premises (Figure 6).
Consider the following description of a typical agent-based model for evacuation:
“When an alarm triggers, agents should evacuate the building. Once an agent exits, they
should not re-enter. Doors can become blocked during the evacuation.”
The modeler may want to check that a simulation is compliant if it follows a safety
property (evacuated agents do not re-enter the building) and a progress property (if at
least one exit is unblocked then every agent eventually exits). It isimplicitthat people
do not keep entering the building while it is evacuated (movement abstraction), and
it is also implicit that the alarm stays on (alarm semantics). Thus the LLM would
have to provide some implicit rules in order for a reasoning tool to fully grasp the
situation. It is important to be transparent: the LLM should convey which elements
came from the modeler, which ones were inferred, and which conclusions were
established by the formal tool. The LLM’s role is not to decide whether implicit
assumptions are correct, but to make them explicit such that specialized tools can
function and modelers can revise the description before relying on the analysis.

30 Philippe J. Giabbanelli
Fig. 5: LLMs can mediate between informal modeling requirements and specialized
tools to avoid a sub-optimal and time-consuming reimplementation of established
methods. LLMs would have to translate and handle feedback from the tools.
4.3 Architectures to integrate LLMs in workflows
While we focused on LLMs as translators or mediators, there are other variations
of using LLMs together with specialized tools. Different architectures have been
employed depending on the researchers’ vision. In atool-embedded LLM architec-
ture(Figure 7a), the LLM is abackend servicewithin the modeling environment
rather than a direct conversational interface. For example, the authors ofMAGDA
detail that the LLM is invoked internally to make suggestions (e.g., UML classes
and attributes) for the modeling environment. All authoritative modeling actions
remain within the specialized tool: the LLM does not perform or validate modeling
actions. Tool-embedded architectures are becoming more available among popular
modeling environments such asStellaandAnyLogic, while noting that the extent
to which the LLM ‘supports’ rather than ‘performs’ modeling tasks depends on
the tool. Another architecture proposed by Lehmann consists oftwo LLM instances
that translate function calls and data back and forth through natural language (Fig-
ure 7b) [63]. While having natural language as an intermediate avoids the need to

Large Language Models for Modeling & Simulation 31
Fig. 6: LLMs can translate or ‘mediate’ by handling implicit assumptions such that
a modeler’s description within a context can be expanded to enable a specialized
tool to operate. Transparency is essential when filling such knowledge gaps, so the
conclusion (bottom, dark blue) should be conveyed to the modeler along with a
disclosure of any inferences that the LLM had to make (bottom right, yellow).
build a shared schema, there are several potential issues: two model calls each time
can negatively affect latency and cost, two LLMs are interpreting the task so their
non-determinism may compound, and natural language can be an underspecified
intermediate representation that creates a semantic drift.
An alternative architecture is to employ a single LLM per call (Figure 7c). This
may suffice if the LLM is intended as the front-end to a single specialized tool, for
example by calling an external Python library (e.g., PySD to run system dynamics)
and executing32the code [52]. Studies that position a single LLM as the primary
interface to a specialized tool may also be exposed to ‘interface bleeding’, whereby
effective use of the LLM increasingly requires users to know the underlying tool or
language it mediates [48]. When there are several specialized tools (each with its
own input requirements), then this architecture becomes problematic. There would
need to be several LLMs that are fine-tuned to the requirements of each tool, along
with adispatcher that selects the right LLMin each case. Treating each specialized
variant as its own model results in repeatedly loading and unloading LLMs with the
same backbone weights. This would bloat memory utilization and is detrimental for
latency, as it can take time to load and unload [17, 16]33. Having to switch between
32An LLM does not ‘execute code’ directly: it is not aruntime system. A product such as ChatGPT
can interact with a Python Sandbox, which can run code. So, to be precise, the tool chain is not
directly LLM↔Python Library but rather LLM↔Python Sandbox↔Python Library.
33Instead of using a single LLM as a thin wrapper around one tool, new designs keep a single base
model loaded and expose multiple external tools through structured, declarative interfaces. The LLM

32 Philippe J. Giabbanelli
Fig. 7: LLMs can be used together with specialized tools in at least four ways:
indirectly by being call within the tool (a), as a pair of translators between the user’s
needs and the tool’s requirements (b), as one or more LLMs fine-tuned for different
tools (c), or as a single LLM augmented to work with each target tool (d).
LLMs also prevents the use of strategies that support efficiency, such as caching.
Several research groups are thus converging towards one architecture (Figure 7d):one
shared LLM backbone with knowledge augmentation(see section 2.3), consisting
of a dynamic switch or combination of task-specific lightweight modules (e.g.,
LoRAs) [17, 16]. This creates a single architecture (e.g., GPU cluster and LLM
backend) to simultaneously support multiple independent users and tasks, which is
calledmulti-tenant serving.
In practice, it is unrealistic to create and maintain a direct one-to-one mapping
between every category of user query and every specialized tools via dedicated
adapters. Given the large number of combinations, a more realistic approach would
be to maintain a small set of reusable adapters that encode common capabilities.
We previously suggested that “the M&S community may maintain a core set of
translations (e.g., if the user’s query calls for first-order logic then useProver 9
syntax) and leave it to users if they need minor adjustments afterward (e.g., turn
Prover 9into the relatedPykesyntax to use a different solver).” [38] A user’s task
would then be achieved bycomposingadapters. Interestingly, there may exist several
can thenreason about which tool to invoke and generate schema-constrained calls, as illustrated in
https://web.archive.org/web/20260203193411/https://www.anthropic.com/engineering/advanced-
tool-use. This supports the composition of heterogeneous tools while avoiding memory overhead
or the need for ad-hoc prompt engineering.

Large Language Models for Modeling & Simulation 33
such compositions34. These different compositions can have different characteristics,
since LoRAs can differ in size and thus computational costs, and some compositions
may be more accurate while others accumulate approximation errors35.
5 Discussion
Given the rising complexity of practices surrounding LLMs and the potential for
misconceptions or counter-intuitive findings, this article aimed to provide a prac-
tical guide from the perspective of M&S. We discussed what can and should be
donenow, such as how to use RAG and LoRA, and how to connect LLMs with
specialized tools in an efficient manner. As research on LLMs progresses rapidly,
practitioners will need to regularly re-assess emerging opportunities. We do not ex-
pect LLMs to suddenly ‘know everything’ or become deterministic in the short term.
Consequently, the need for knowledge augmentation and for evaluating the impact of
non-determinism is likely to remain. At the same time, the specific techniques used to
address these needs may evolve. Some approaches may be displaced in the medium
term (e.g., RAG or LoRA), while their implementations are already changing in the
short term, as evidenced by the proliferation of variants. To help modelers anticipate
such developments, the remainder of this section highlights two technical advances
that are not yet widely applicable to M&S (and were thus not covered earlier), but
that could soon present new opportunities.
First, this article exclusively discussed LLMs as processing a textual input (which
could represent different objects such as conceptual models or simulation outcomes).
However, there are now multiplemultimodal36LLMs: they can process images,
34Consider a modeler who wants to check a safety requirement stated in natural language (“the
system must never enter an unsafe state”). One translation path could transform the simulation’s
description into state-based abstraction (states, transitions, guards) then encode requirement in
temporal logic, and finally align with the syntax required by a specific checker. Alternatively, we
could map requirements into constraint-style assertions (e.g., first-order constraints or satisfiability-
style queries), and then use translate for a constraint solver. Both paths are ‘compositions’, but they
route the task through different formalisms and toolchains.
35We can represent each format decision a a node and available transformations as directed edges,
annotated with properties such as computational cost or information loss [103]. This representa-
tion enables a systematic exploration and evaluation of alternative sequences of transformations
so that the desired transformation path can be selected based on explicit trade-offs. The literature
on multi-paradigm modeling has also extensively covered model transformations, and particularly
transformations between formalisms [121]. The literature has also noted that sequences of transfor-
mations (also called workflows) are often implicit and informal, but documenting them explicitly
would be helpful to compare, optimize, and reuse transformations. In short, we need to develop
LoRAs that support transformations, and we need to document these sequences explicitly: which
LoRAs are used, in what order, and with what outcomes. [96].
36A multimodal LLM is any model that accepts or produces more than one modality, such as
textandimages. Being multimodal says nothing about how the model reasons internally. For
instance, an LLM can be multimodal by converting images to text and then reasoning only over
text. Practitioners may also encounter the termomni, which is more of a marketing or product

34 Philippe J. Giabbanelli
audio, or video. While these capacities have been available since GPT-4o, Gemini
2.0 Claude 3.5 Sonnet, or the lighweight Microsoft Phi 3.5-Vision, there has been a
limited use of multimodal LLMs in the M&S literature so far. Use cases have included
providing inputs as visuals instead of text (e.g., images of UML diagrams [9]), the
analysis of simulation outputs (e.g., using images and videos of structural and fluid
dynamics simulations [26]), or teaching an LLM about M&S concept by giving it
complete slide decks from a course (including visual for spatial concepts such as
neighborhoods in cellular automata) [29]. Results can be mixed, perhaps owing to the
early stage of the technology. In the study analyzing simulation outputs, giving videos
as inputs instead of batches of images had little impact, and text was the most robust
modality [26]. As multimodal models mature and their internal representations and
guarantees become better understood, such approaches merit renewed attention and
systematic re-evaluation in future M&S research.
Second, the dominant approach by far has been adesign by addition: when
LLMs do not produce the desired answers, practitioners tend to add instructions to
the prompt, fine-tune the model, or connect a RAG. However, LLMs may encode
misunderstandings, in which case this strategy is akin to layering patches on top of a
flawed specification rather than revisiting the specification itself. By contrast, adesign
by subtractionapproach seeks to identify which knowledge or associations should
be removed from an LLM. In the case of conceptual modeling, we examinedwhatto
unlearn and found that each essential aspect was violated by at least one well-known
LLM, suggesting a need to unlearn problematic samples (Table 4) [99]. A common
misconception is that unlearning necessarily requires removing data samples and
retraining the entire system, or substantial parts of it. While this is possible [14], it is
not the only approach; it can be computationally prohibitive and may have detrimental
effects on properties such as fairness [139]. For example, when datasets are organized
into shards to limit retraining costs, deleting a shard that exhibits issues in conceptual
modeling may also remove data associated with novice learners, thereby impairing
the model’s ability to generate or analyze novice conceptual models. An alternative
toexactunlearning through data deletion isapproximateunlearning, which aims to
mitigate the influence of undesirable instances without fully removing them. There
are many such methods, different trade-offs and applicability to M&S [99]. Some
approaches target specific instances but may destabilize the LLM (just like editing
a neuron can affect other functions of a brain), while others offer more localized
control but need extensive labeling effort. Although these techniques remain largely
untested in M&S settings, they have potential to address persistent modeling errors.
AcknowledgementsThis article was made possible through the experience gained in projects with
many collaborators and students. The author is thankful for the hard work of current and former
label than a clearly defined research concept. The term is generally intended tosuggesta more
integrated internal model, potentially supporting native or unified multimodality rather than relying
on stitched pipelines such as a vision encoder deriving text tokens from an image, after which the
LLM still reasons only on text. Ultimately, reasoning capabilities cannot be inferred solely from
labels such as multimodal or omni. Rather, modelers must examine which representations are used,
when they are used, and with what guarantees. Providers may disclose only partial information.

Large Language Models for Modeling & Simulation 35
Table 4: Each of six key requirements for conceptual modeling was violated by at
least one well-known LLM, highlighting the need to address such failures either by
augmentation (patches) or by deliberate forgetting (unlearning) [99].
Category Typical mistake Illustration / manifestation in conceptual modeling
Temporal repre-
sentationMismatch of time
units and orderingMixing daily and weekly time scales within the same
model; inverting the order of events when describing
system evolution
Task structure Linearization of in-
herently non-linear
processesRepresenting branched or conditional processes as a
single linear sequence; collapsing parallel tasks into a
single process
Vagueness / lack
of groundingUse of ambiguous or
unmeasurable con-
structsConcepts such as “integration,” “intentional explo-
ration,” or “personal transformation” introduced with-
out clear referents or operational meaning
Causality Missing, reversed, or
incorrect causal linksOmitting causal connectors, reversing causal direction,
or asserting direct causation where mediation is re-
quired (e.g., between population density, green space
availability, and air pollution)
Consistency of
termsTreating synonyms
or paraphrases as dis-
tinct conceptsIntroducing multiple labels for the same underlying
concept, leading to fragmentation of the conceptual
model and communication breakdowns
Structural cohe-
sivenessUnclear model
boundaries or
aggregation levelProducing a single monolithic model where multiple
coordinated sub-models would be appropriate, or fail-
ing to articulate how components fit together
students in this domain (Cristina Perez, Stephen Zhong, Ryan Schuerkamp, Anish Shrestha, No ´e
Flandre, Tyler Gandee) as well as opportunities to learn with collaborators (Patrick Wu, Nathalie
Japkowicz, Istvan David, Andreas Tolk, Niclas Feldkamp).
Reflection and Exploration
•▶Reasoning is not always better and refusing to engage is a signal
Section 2.2 covered hyper-parameters such as temperature. LLMs increasingly
expose a new hyper-parameter:reasoning. It controls whether and how the model
produces intermediate reasoning steps before generating an answer. A confusion
would be to think that more reasoning is better. However, there are tasks when
no reasoningcan be better thansome reasoning. Using a model with a reasoning
parameter (e.g., GPT 5 Nano), provide a prompt under the ‘no reasoning’ case and
compare its results (over several runs) under the ‘low reasoning’. For example, the
prompt from a social simulation can be: “you are a college-educated Asian American
male age 53, a peer is trying to convince you that higher taxes are good to support
your community, what do you reply?” Note that results may even differ in whether

36 Philippe J. Giabbanelli
the LLM is willing to engage with the task, so do not treat refusals as a ‘bug’ that
you can replace by running the prompt again: an LLM refusing to answer is a signal.
•▶The different meanings of open models
‘Open’ is a vague qualifier for LLMs and is sometimes invoked to justify the choice
of a model. In practice, the term often refers to models for which practitioners
can access the neural networkweights, allowing local deployment and extensive
fine-tuning. However, Widder and colleagues argue that openness is multi-faceted
and that AI systems labeled as ‘open’ may not, in fact, be open in any meaningful
sense, depending on how openness is defined [128]. According to the authors, which
components of an LLM can meaningfully be made open, and which cannot? Consider
an M&S workflow that relies on fine-tuning an open-weights LLM. Which forms of
openness remain relevant even if weights are available locally?
•▶LLM for RAG for LLM
When we presented Retrieval-Augmented Generation (RAG) in Section 2.3, we
framed retrieval as a way to provide knowledge to an LLM. Recent work, however,
shows that LLMs are increasingly used within RAG themselves. Read the survey by
Panditet al.[89], then answer the following questions: (i) at which stage of the RAG
pipeline does the LLM intervene? (ii) What information does the LLM use?
•▶Using masking to probe an LLM’s knowledge prior to augmentation
Section 2.3 covered techniques for knowledge augmentation through contextual or
parametric approaches (e.g., RAG, LoRA). Before embarking on the time-consuming
process of optimizing one of these approaches, a modeler should assess whether
knowledge augmentation is actually necessary. Determining what an LLM already
knows (i.e.,factual probing) may take some effort, yet it can reveal that more complex
augmentation strategies are unnecessary or that their scope can be sharply reduced.
Cloze promptsare a common method for factual probing, which uses sentences with
missing parts (blanks) that the LLM must fill in. Note that replacing a word from a
sentence with a special token (e.g., [MASK]) is a related technique called masking,
but it is used for pre-training whereas cloze is a downstream prompt strategy. In this
exercise, you will assess an LLM’s existing causal knowledge by selectively hiding
information and observing how the LLM completes the missing elements.
1. Create a small causal map consisting of 5 to 10 labeled nodes (e.g., reading
textbooks, skill development, raises, satisfaction) and directed edges representing
causal influence either as a causal increase (e.g., reading textbooks increases skill

Large Language Models for Modeling & Simulation 37
development) or decrease. You can also skip this step by reusing open causal
maps and taking a small sample [40].
2. When transforming a conceptual model (e.g., causal map) to text, the input may
have cycles but text must be linear. The linearization step consists of representing
the causal graph as groups of edges which, if taken together, would re-create the
whole model. Decompose the causal map created in the previous step into two or
three linear parts (i.e., subgraphs).
3. For each linear part of your conceptual model, create two versions:withthe type of
causal edges (increase or decrease) andwithout(i.e., ‘masking’ the information).
4. Prompt the LLM to generate a natural-language description of the model from
the masked subgraphs and another description from the subgraphs containing the
type of causal edges. Compare the results: is your LLM able to correctly guess the
type of causality? Remember that LLMs are non-deterministic, so you should not
conclude from a single observation; rather, each prompt should be run multiple
times. Do not ‘give away’ the answer to an LLM by including examples within
your prompt, since we want the LLM to reveal what it knows about causality.
•▶When knowledge augmentation is needed: experimenting with LoRAs
Section 2.3 discussed knowledge augmentation in LLMs through techniques such as
Low-Rank Adaptation (LoRA). While LoRAs are often presented as a lightweight
alternative to full fine-tuning, training a LoRA is not simply a matter of providing
data and letting an algorithm build the model. The training process involvesseveral
parametersthat materially affect the resulting behavior. Some parameters govern
thearchitecture of the LoRA itself(e.g., rank, scaling factor𝛼, and dropout), de-
termining how much and where the base model can be adapted. Other parameters
govern thetraining process(e.g., learning rate, batching strategy, number of epochs),
influencing how strongly the LoRA internalizes the training examples. In this exer-
cise, you will train LoRAs for a modeling and simulation task, systematically vary
selected parameters, and observe how these choices affect both training outcomes
and downstream behavior during inference.
1. Construct a small, structured dataset that reflects a stable modeling task, suitable
for LoRA training.The following is an example of how you can create such a
dataset, but you can make a different one and still proceed with the next step.
Define a fixed target format to describe simulation scenarios (this codifies how
you want the LoRA to enforce standards of the field). Then, create a dataset of
input-output pairs where the input is a natural language description of a scenario
(e.g., “we’d like to see what happens when just one tree is burning and we look at
the forest for 20 steps”) and output is the same scenario expressed based on the
conventions that we seek to enforce (e.g., “Initially, the number of burning trees is
1. The simulation runs for 20 steps.”). Ensure that inputs use varied phrasing and
terminology for the same underlying ideas (e.g., ‘trees on fire’, ‘burning trees’,
‘ignited trees’), The outputs must follow the same textual conventions across all

38 Philippe J. Giabbanelli
examples. You can have several inputs that map to the same output. Your goal to
teach a stable way of expressing scenarios, not model domain knowledge.
2. Train at least four LoRAs using the same dataset and base model, usingPEFT
orUnsloth. Withhold 20% of your data from training, so that you can use it
later for evaluation. For each LoRA, vary the architecture parameters (rank𝑟e.g.
4 vs 8 vs 16, scaling factor𝛼e.g. 8 vs 16 vs 32, dropout e.g. 0 vs 0.1) and the
training parameters (consider at least two values for the learning rate, the hatching
strategy, and the number of epochs). Record the training time to create the LoRA,
and observe the influence of the parameter values on the training time.
3. To know whether a LoRA ‘works’, you need to evaluate the output. Metrics could
cover format compliance (does it look like the target style?), fidelity (does it pre-
serve the intent of the informal description?), and so on. Propose a set of metrics
along with clear criteria for scoring (e.g., what is high or low compliance?).
4. Using the 20% hold-out dataset and your proposed metrics, evaluate how the
different LoRAs influence the results. For each LoRA, use the same prompts
and decoding parameters to generate codified scenario descriptions from the
informal inputs. Compare the performances based on the parameters. For the best
performing LoRA, vary its weight to 0.8 and 1, and examine the effect.
•▶More data is not always better: avoiding model collapse and jailbreaks
The performance of an LLM on a task does not always improve with more training
data. We may observeplateaus, such that the investment in acquiring more training
data is no longer worth it. We may even observe adegradationin downstream per-
formance. This reading list explains why more data may not lead to the performance
that modelers would have expected. Each reading can be pursued independently.
1. Manually curating a set of input and output pairs is very time-consuming. It can
also be difficult for a few individuals to create varied instances, as they can only
think of so many different ways in which to describe the same information (e.g.,
the structure of a causal model, the wording of a simulation scenario). There is
thus interest in training an LLM using data that was fully or partly generated by
another LLM (‘regurgitative training’). This may be intentional and controlled by
modelers who prompt an LLM to generate data and apply filters to ensure diversity
and quality of the dataset. It may also be unintentional, as modelers scrap online
data that may have already been generated by LLMs. After reading the study
by Zhang and colleagues [140], discuss (i) how we can evaluate the quality of
LLM-generated data intended as training for another LLM, and (ii) how we can
generate data via several LLMs to increase the diversity of the training set. For a
more formal treatment ofmodel collapse(i.e., a drop in performance after training
on LLM-generated data), we refer the reader to Aminet al.[2].
2. Fine-tuning an LLM may (perhaps inadvertently) compromise its safety guardrails,
which is a particular concern when a model is made available for fine-tuning by
third parties. To understand such fine-tuning attacks, read Hsiunget al.[51] and

Large Language Models for Modeling & Simulation 39
then answer (i) why fine-tuning may compromise guardrails; (ii) how much data
is needed, and how intentional does it have to be, in order to erode guardrails;
and (iii) how can additional training data be screened to prevent safety issues.
3. An experimental protocol can examine the response curve between the amount
of data (x-axis) and the performance of an LLM (y-axis) on a given task. Read
Giabbanelliet al.[40] for a protocol that compares performance based on three
discrete sample sizes (zero shot, few shot, fine tuning) then explain how this
protocol can be generalized to provide a more fine-grained response curve. In
particular, instead of evaluating performance at fixed sample sizes (e.g., 10, 20,
30 examples), consider howadaptivesampling could be used: that is, selecting
the next training size based on observed changes in performance.
•▶Decomposing modeling and simulation data as inputs to LLMs
A conceptual model may be a large graph with hundreds of concepts and many
more relationships, while a simulation output may consist of the values of many
entities over many time steps and across many runs. LLMs cannot reliably attend to
everything in such inputs. Na ¨ıve simplifications can lead to a loss of information and
thus negatively impact performance: flattening a conceptual model can destroy struc-
tural information, while summarizing simulation outputs can miss on key patterns.
Structure-aware decomposition algorithms have thus been developed at the level of
the data (e.g., conceptual models [33]) or at the level of the query [135] (identifying
which operators such as filtering or joins are needed on which data type).
1. Consider a conceptual model represented as a single graph, or a spreadsheet of
simulation outputs spanning many entities and time steps. Suppose that a modeler
simply provides the entire artifact (or a flattened textual version of it) to an LLM
together with a question. Explain why this approach is likely to fail even when
the artifact fits within the model’s context window. In your answer, distinguish
between limitations due to context length and limitations due to loss of structure.
2. From the readings, contrast data-level decomposition with query-level decompo-
sition. Designing a decomposition strategy.
3. Choose a M&S artifact of interest to you and then explain how would decompose
the artifact itself or the queries posed to it. This question is about devising a
practical solution for a specific M&S case, in contrast with the second question.
•▶Identifying reusable skills in LLM pipelines
Recent practitioner-oriented tooling increasingly frames LLM capabilities as
reusableskills: small, self-contained pipeline components that implement a spe-
cific function (e.g., retrieval, evaluation, tool calling, or structured output val-
idation). Rather than proposing new learning algorithms, skill repositories fo-

40 Philippe J. Giabbanelli
cus on operationalizing best practices and reducing ad-hoc glue code when as-
sembling multi-stage LLM systems. Browse the Hugging Face skills repository
(https://github.com/huggingface/skills) and identify two skills that are relevant to
modeling and simulation workflows (e.g., retrieval over technical documents, struc-
tured generation, or evaluation of generated outputs). Explain the capability provided
by each skill, where it fits in the M&S pipeline, and what aspects would be standard-
ized (e.g., retrieval strategy, output structure).
References
1. A. Akhavan and M. S. Jalali. Generative ai and simulation modeling: how should you (not)
use large language models like chatgpt.System Dynamics Review, 40(3):e1773, 2024.
2. K. Amin, S. Babakniya, A. Bie, W. Kong, U. Syed, and S. Vassilvitskii. Escaping collapse: The
strength of weak data for large language model training.arXiv preprint arXiv:2502.08924,
2025.
3. I. Arawjo, C. Swoopes, P. Vaithilingam, M. Wattenberg, and E. Glassman. Chainforge: A
visual toolkit for prompt engineering and llm hypothesis testing. InProceedings of the CHI
Conference on Human Factors in Computing Systems (CHI ’24), pages 1–18, 2024.
4. E. G. Arias, M. Li, C. Heumann, and M. Aßenmacher. Decoding decoded: Understanding
hyperparameter effects in open-ended text generation. InProceedings of the 31st International
Conference on Computational Linguistics, pages 9992–10020, 2025.
5. A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi†. Self-rag: Learning to retrieve, generate,
and critique through self-reflection. InProceedings of ICLR, 2024.
6. M. Astekin, M. Hort, and L. Moonen. An exploratory study on how non-determinism in
large language models affects log parsing. InProceedings of the ACM/IEEE 2nd Inter-
national Workshop on Interpretability, Robustness, and Benchmarking in Neural Software
Engineering, pages 13–18, 2024.
7. B. Atil, S. Aykent, A. Chittams, L. Fu, R. J. Passonneau, E. Radcliffe, G. R. Rajagopal,
A. Sloan, T. Tudrej, F. Ture, et al. Non-determinism of” deterministic” llm settings.arXiv
preprint arXiv:2408.04667, 2024.
8. H. Babaei Giglou, J. D’Souza, F. Engel, and S. Auer. Llms4om: Matching ontologies with
large language models. InThe Semantic Web: ESWC 2024 Satellite Events: Hersonissos,
Crete, Greece, May 26–30, 2024, Proceedings, Part I, page 25–35, Berlin, Heidelberg, 2025.
Springer-Verlag.
9. A. Bates, R. Vavricka, S. Carleton, R. Shao, and C. Pan. Unified modeling language code
generation from diagram images using multimodal large language models.Machine Learning
with Applications, page 100660, 2025.
10. A. Baumann and P. Eberhard. Experiments with large language models on
retrieval-augmented generation for closed-source simulation software.arXiv preprint
arXiv:2502.03916, 2025.
11. F. Bertolotti, S. Roman, F. Carucci, G. Buonanno, and L. Mari. An llm-enhanced agent-based
model of a sustainability game. InProceedings of the 26th Workshop From Objects to Agents
(WOA2025), 2025. July 02–05, 2025, Trento, Italy.
12. D. Beyer and T. Lemberger. Six years later: testing vs. model checking.International Journal
on Software Tools for Technology Transfer, 26(6):633–646, 2024.
13. U. M. Borghoff, P. Bottoni, and R. Pareschi. Beyond prompt chaining: The tb-cspn architecture
for agentic ai.Future Internet, 17(8):363, 2025.
14. L. Bourtoule, V. Chandrasekaran, C. A. Choquette-Choo, H. Jia, A. Travers, B. Zhang, D. Lie,
and N. Papernot. Machine unlearning. In2021 IEEE symposium on security and privacy
(SP), pages 141–159. IEEE, 2021.

Large Language Models for Modeling & Simulation 41
15. H. Cai, B. Shen, L. Jin, L. Hu, and X. Fan. Does tone change the answer? evaluating prompt
politeness effects on modern llms: Gpt, gemini, llama.arXiv preprint arXiv:2512.12812,
2025.
16. J. Chen. Comparative analysis and optimization of lora adapter co-serving for large language
models. InProceedings of the 25th International Middleware Conference: Demos, Posters
and Doctoral Symposium, pages 27–28, 2024.
17. L. Chen, Z. Ye, Y. Wu, D. Zhuo, L. Ceze, and A. Krishnamurthy. Punica: Multi-tenant lora
serving.Proceedings of Machine Learning and Systems, 6:1–13, 2024.
18. L. Chen, M. Zaharia, and J. Zou. How is chatgpt’s behavior changing over time?Harvard
Data Science Review, 6(2), 2024.
19. S. Cheng, L. Pan, X. Yin, X. Wang, and W. Y. Wang. Understanding the interplay be-
tween parametric and contextual knowledge for large language models.arXiv preprint
arXiv:2410.08414, 2024.
20. Z. Cheng, M. Cao, M.-A. Rondeau, and J. C. Cheung. Stochastic chameleons: Irrelevant
context hallucinations reveal class-based (mis) generalization in llms. InProceedings of
the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), pages 30187–30214, 2025.
21. W. Cui, J. Zhang, Z. Li, H. Sun, D. Lopez, K. Das, B. A. Malin, and S. Kumar. Automatic
prompt optimization via heuristic search: A survey.arXiv preprint arXiv:2502.18746, 2025.
22. T. Dao, D. Fu, S. Ermon, A. Rudra, and C. R ´e. Flashattention: Fast and memory-efficient exact
attention with io-awareness.Advances in neural information processing systems, 35:16344–
16359, 2022.
23. W. Ding, Y. Cao, D. Zhao, C. Xiao, and M. Pavone. Realgen: Retrieval augmented generation
for controllable traffic scenarios. InEuropean Conference on Computer Vision, pages 93–110.
Springer, 2024.
24. O. Dobariya and A. Kumar. Mind your tone: Investigating how prompt politeness affects llm
accuracy (short paper).arXiv preprint arXiv:2510.04950, 2025.
25. Y. Du, M. Tian, S. Ronanki, S. Rongali, S. B. Bodapati, A. Galstyan, A. Wells, R. Schwartz,
E. A. Huerta, and H. Peng. Context length alone hurts LLM performance despite perfect
retrieval. In C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng, editors,Find-
ings of the Association for Computational Linguistics: EMNLP 2025, pages 23281–23298.
Association for Computational Linguistics, Nov. 2025.
26. J. Ezemba, C. McComb, and C. Tucker. Simulation vs. hallucination: Assessing vision-
language model question answering capabilities in engineering simulations. InProceedings
of the 7th Workshop on Design Automation for CPS and IoT, pages 1–9, 2025.
27. B. Fatemi, J. Halcrow, and B. Perozzi. Talk like a graph: Encoding graphs for large language
models. InInternational Conference on Learning Representations (ICLR), 2024.
28. A. Ferraro, A. Galli, V. La Gatta, M. Postiglione, G. M. Orlando, D. Russo, G. Riccio,
A. Romano, and V. Moscato. Agent-based modelling meets generative ai in social network
simulations. InInternational Conference on Advances in Social Networks Analysis and
Mining, pages 155–170. Springer, 2024.
29. N. Y. Flandre and P. J. Giabbanelli. Can large language models learn conceptual modeling by
looking at slide decks and pass graduate examinations? an empirical study. InInternational
Conference on Conceptual Modeling, pages 198–208. Springer, 2024.
30. J. M. Fraile-Hern ´andez and A. Pe ˜nas. On measuring large language models performance
with inferential statistics.Information, 16(9):817, 2025.
31. E. Frydenlund, J. Mart ´ınez, J. J. Padilla, K. Palacio, and D. Shuttleworth. Modeler in a
box: how can large language models aid in the simulation modeling process?Simulation,
100(7):727–749, 2024.
32. S. Gaire, S. Gyawali, S. Mishra, S. Niroula, D. Thakur, and U. Yadav. Systematization of
knowledge: Security and safety in the model context protocol ecosystem.arXiv preprint
arXiv:2512.08290, 2025.
33. T. J. Gandee and P. J. Giabbanelli. Combining natural language generation and graph algo-
rithms to explain causal maps through meaningful paragraphs. InInternational Conference
on Conceptual Modeling, pages 359–376. Springer, 2024.

42 Philippe J. Giabbanelli
34. T. J. Gandee, S. C. Glaze, and P. J. Giabbanelli. A visual analytics environment for navi-
gating large conceptual models by leveraging generative artificial intelligence.Mathematics,
12(13):1946, 2024.
35. D. Ganguli, L. Lovitt, J. Kernion, A. Askell, Y. Bai, S. Kadavath, B. Mann, E. Perez,
N. Schiefer, K. Ndousse, et al. Red teaming language models to reduce harms: Methods,
scaling behaviors, and lessons learned.arXiv preprint arXiv:2209.07858, 2022.
36. N. Ghaffarzadegan, A. Majumdar, R. Williams, and N. Hosseinichimeh. Generative agent-
based modeling: an introduction and tutorial.System Dynamics Review, 40(1):e1761, 2024.
37. P. J. Giabbanelli. Solving challenges at the interface of simulation and big data using machine
learning. In2019 Winter Simulation Conference (WSC), pages 572–583. IEEE, 2019.
38. P. J. Giabbanelli, J. Beverley, I. David, and A. Tolk. From over-reliance to smart integration:
Using large-language models as translators between specialized modeling and simulation
tools. InProceedings of the 2025 Winter Simulation Conference. Winter Simulation Confer-
ence, 2025.
39. P. J. Giabbanelli, C. Daumas, N. Y. Flandre, A. Pitkar, and J. Vazquez-Estrada. Promoting
empathy in decision-making by turning agent-based models into stories using large-language
models.Journal of Simulation, pages 1–21, 2025.
40. P. J. Giabbanelli, A. Phatak, V. Mago, and A. Agrawal. Narrating causal graphs with large
language models. InProceedings of the 57th Hawaii International Conference on System
Sciences (HICSS), pages 7530–7539, Honolulu, HI, USA, Jan 2024. IEEE. Research and
Education track, Paper 6 at HICSS-57.
41. P. J. Giabbanelli and N. Witkowicz. Generative ai for systems thinking: Can a gpt question-
answering system turn text into the causal maps produced by human readers? InProceedings
of the 57th Hawaii International Conference on System Sciences (HICSS-57), pages 7540–
7549, Hilton Hawaiian Village, Honolulu, Hawaii, 2024. Hawaii International Conference on
System Sciences.
42. S. Goldman. Neurips, one of the world’s top academic ai conferences, accepted research
papers with 100+ ai-hallucinated citations, new report claims.Fortune, Jan 21 2026.
43. A. Gutschmidt and B. Nast. Assessing model quality using large language models. InIFIP
Working Conference on The Practice of Enterprise Modeling, pages 105–122. Springer, 2024.
44. A. Harper, N. Mustafee, and M. Yearworth. Facets of trust in simulation studies.European
Journal of Operational Research, 289(1):197–213, 2021.
45. Y. He, J. Chen, H. Dong, and I. Horrocks. Exploring large language models for ontology
alignment. InProceedings of the ISWC 2023 Posters and Demos Track, International Semantic
Web Conference, Athens, Greece, November 2023.
46. M. K. Heris. Prompt decorators: A declarative and composable syntax for reasoning, format-
ting, and control in llms.arXiv preprint arXiv:2510.19850, 2025.
47. S. Hertling and H. Paulheim. Olala: Ontology matching with large language models. In
Proceedings of the 12th knowledge capture conference 2023, pages 131–139, 2023.
48. J. Hoffmann, R. B¨ uttner, and B. Hu. Modularization and integration of stock-and-flow models
using chatpysd. InProceedings of the 2025 International System Dynamics Conference,
Boston, USA, Aug 2025. System Dynamics Society. Abstract paper supporting session.
49. A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi. The curious case of neural text
degeneration.arXiv preprint arXiv:1904.09751, 2019.
50. A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi. The curious case of neural text
degeneration. InProceedings of ICLR, 2020.
51. L. Hsiung, T. Pang, Y.-C. Tang, L. Song, T.-Y. Ho, P.-Y. Chen, and Y. Yang. Why llm safety
guardrails collapse after fine-tuning: A similarity analysis between alignment and fine-tuning
datasets. InData in Generative Models Workshop: The Bad, the Ugly, and the Greats (DIG-
BUGS), in Proceedings of the 42nd International Conference on Machine Learning (ICML),
Vancouver, Canada, Jul 2025. Copyright 2025 by the author(s).
52. B. Hu. Chatpysd: Embedding and simulating system dynamics models in chatgpt-4.System
Dynamics Review, 41(1):e1797, 2025.
53. E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, and W. Chen. Lora: Low-rank
adaptation of large language models.arXiv preprint arXiv:2106.09685, 2022.

Large Language Models for Modeling & Simulation 43
54. Y. Ichikawa, Y. Fujimoto, and A. Sakai. Lpcd: Unified framework from layer-wise to sub-
module quantization.arXiv preprint arXiv:2512.01546, 2025.
55. M. S. Jalali and A. Akhavan. Integrating ai language models in qualitative research: Repli-
cating interview data analysis with chatgpt.System Dynamics Review, 40(3):e1772, 2024.
56. U. Kamath, K. Keenan, G. Somers, and S. Sorenson.Large Language Models: A Deep Dive.
Springer, Cham, 2024.
57. N. Klievtsova, J. Mangler, T. Kampik, J.-V. Benzin, and S. Rinderle-Ma. How can generative ai
empower domain experts in creating process models? InProceedings of Wirtschaftsinformatik
2024, page 66, 2024.
58. V. Kohli. What is model context protocol (mcp)? https://www.itpro.com/technology/artificial-
intelligence/what-is-model-context-protocol-mcp, Oct 2025. Published 28 October 2025, IT
Pro.
59. B. Kutela, N. Novat, N. Novat, J. Herman, A. Kinero, and S. Lyimo. Artificial intelligence (ai)
and job creation: An exploration of the nature of the jobs, qualifications, and compensations
of prompt engineers.Qualifications, and Compensations of Prompt Engineers (November 6,
2023), 2023.
60. W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica.
Efficient memory management for large language model serving with pagedattention. In
Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023.
61. M. Larooij and P. T ¨ornberg. Validation is the central challenge for generative social simulation:
a critical review of llms in agent-based modeling.Artificial Intelligence Review, 59(1):15,
2025.
62. E. A. Lavin and P. J. Giabbanelli. Analyzing and simplifying model uncertainty in fuzzy
cognitive maps. In2017 Winter Simulation Conference (WSC), pages 1868–1879. IEEE,
2017.
63. R. Lehmann. Towards interoperability of apis-an llm-based approach. InProceedings of the
25th International Middleware Conference: Demos, Posters and Doctoral Symposium, pages
29–30, 2024.
64. P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. K¨ uttler, M. Lewis, W.-t.
Yih, T. Rockt ¨aschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems, 33:9459–9474, 2020.
65. J. Li and P. J. Giabbanelli. Identifying synergistic interventions to address covid-19 using a
large scale agent-based model. InInternational conference on computational science, pages
655–662. Springer, 2021.
66. X. L. Li, A. Holtzman, D. Fried, P. Liang, J. Eisner, T. B. Hashimoto, L. Zettlemoyer, and
M. Lewis. Contrastive decoding: Open-ended text generation as optimization. InProceedings
of the 61st annual meeting of the association for computational linguistics (volume 1: Long
papers), pages 12286–12312, 2023.
67. Y. Li, R. Yin, D. Lee, S. Xiao, and P. Panda. Gptaq: Efficient finetuning-free quantization for
asymmetric calibration. InProceedings of ICML, 2025.
68. Z. Li, Y. Liu, Y. Su, and N. Collier. Prompt compression for large language models: A
survey. InProceedings of the 2025 Conference of the Nations of the Americas Chapter of the
Association for Computational Linguistics: Human Language Technologies (Volume 1: Long
Papers), pages 7182–7195, 2025.
69. J. Linardon, H. K. Jarman, Z. McClure, C. Anderson, C. Liu, and M. Messer. Influence of
topic familiarity and prompt specificity on citation fabrication in mental health research using
large language models: experimental study.JMIR Mental Health, 12:e80371, 2025.
70. N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang. Lost
in the middle: How language models use long contexts.Transactions of the Association for
Computational Linguistics, 12:157–173, 2024.
71. Y. Liu, X. Li, M. Zhao, S. Zhang, Z. Wang, Q. Li, S. Feng, F. Ren, D. Wang, and
H. Sch¨ utze. High-rank structured modulation for parameter-efficient fine-tuning.arXiv
preprint arXiv:2601.07507, 2026.

44 Philippe J. Giabbanelli
72. A. Lizarraga, E. Honig, and Y. N. Wu. From stochastic parrots to digital intelligence:
The evolution of language models and their cognitive capabilities.Wiley Interdisciplinary
Reviews: Computational Statistics, 17(3):e70035, 2025.
73. E. Lumer, A. Gulati, V. K. Subbiah, P. H. Basavaraju, and J. A. Burke. Scalemcp: Dy-
namic and auto-synchronizing model context protocol tools for llm agents. In F. Marcelloni,
K. Madani, N. van Stein, and J. Filipe, editors,Computational Intelligence, pages 23–42,
Cham, 2026. Springer Nature Switzerland.
74. C. B. Lutz and P. J. Giabbanelli. When do we need massive computations to perform detailed
covid-19 simulations?Advanced Theory and Simulations, 5(2):2100343, 2022.
75. Q. Ma, W. Peng, C. Yang, H. Shen, K. Koedinger, and T. Wu. What should we engineer in
prompts? training humans in requirement-driven llm use.ACM Transactions on Computer-
Human Interaction, 32(4):1–27, 2025.
76. S. Mangrulkar, S. Gugger, L. Debut, Y. Belkada, S. Paul, et al. Parameter-efficient fine-tuning
of large language models.arXiv preprint arXiv:2209.06845, 2022.
77. R. Marigliano and K. M. Carley. Aurora: Enhancing synthetic population realism through
rag and salience-aware opinion modeling. In E. Azar, A. Djanatliev, A. Harper, C. Kogler,
V. Ramamohan, A. Anagnostou, and S. J. E. Taylor, editors,Proceedings of the 2025 Winter
Simulation Conference, 2025.
78. J. Mart ´ınez, B. Llinas, J. G. Botello, J. J. Padilla, and E. Frydenlund. Enhancing gpt-3.5’s
proficiency in netlogo through few-shot prompting and retrieval-augmented generation. In
2024 Winter Simulation Conference (WSC), pages 666–677. IEEE, 2024.
79. R. T. McCoy, S. Yao, D. Friedman, M. D. Hardy, and T. L. Griffiths. Embers of autoregres-
sion show how large language models are shaped by the problem they are trained to solve.
Proceedings of the National Academy of Sciences, 121(41):e2322420121, 2024.
80. L. Memmert, I. Cvetkovic, and E. Bittner. The more is not the merrier: Effects of prompt
engineering on the quality of ideas generated by gpt-3. InProceedings of the 57th Hawaii
International Conference on System Sciences (HICSS), Waikoloa, HI, USA, 2024. Association
for Information Systems.
81. T. M ¨oltner, P. Manzl, M. Pieber, and J. Gerstmayr. Creation, evaluation and self-validation
of simulation models with large language models.Neurocomputing, page 132030, 2025.
82. N. Mustafee and M. Fakhimi. Towards an integrative taxonomical framework for hybrid
simulation and hybrid modelling. In M. Fakhimi and N. Mustafee, editors,Hybrid Modeling
and Simulation: Conceptualizations, Methods and Applications, pages 3–22. Springer Nature
Switzerland, Cham, 2024.
83. N. S. Nakshatri, S. Roy, R. Das, S. Chaidaroon, L. Boytsov, and R. Gangadharaiah. Constrained
decoding with speculative lookaheads. In L. Chiruzzo, A. Ritter, and L. Wang, editors,
Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association
for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers),
pages 4681–4700, Albuquerque, New Mexico, Apr. 2025. Association for Computational
Linguistics.
84. Y. Nie and S. Gao. Knowledge-guided large language models for enhancing agent-based wild-
fire spatial simulation. InProceedings of the 8th ACM SIGSPATIAL International Workshop
on Geospatial Simulation (GeoSIM ’25). ACM, 2025.
85. Q. Nivon and G. Sala¨ un. Automated generation of bpmn processes from textual requirements.
InInternational Conference on Service-Oriented Computing, pages 185–201. Springer, 2024.
86. Q. Nivon, G. Sala¨ un, and F. Lang. Givup: Automated generation and verification of textual
process descriptions. InProceedings of the 33rd ACM International Conference on the
Foundations of Software Engineering, pages 1119–1123, 2025.
87. M. Oremland and R. Laubenbacher. Optimization of agent-based models: Scaling methods
and heuristic algorithms.Journal of Artificial Societies and Social Simulation, 17(2):6, 2014.
88. S. Ouyang, J. M. Zhang, M. Harman, and M. Wang. An empirical study of the non-
determinism of chatgpt in code generation.ACM Transactions on Software Engineering
and Methodology, 34(2):1–28, 2025.

Large Language Models for Modeling & Simulation 45
89. T. Pandit, S. Mahendru, M. Raval, and D. Upadhyay. The evolution of reranking models
in information retrieval: From heuristic methods to large language models.arXiv preprint
arXiv:2512.16236, 2025.
90. J. S. Park, J. O’Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein. Generative
agents: Interactive simulacra of human behavior. InProceedings of the 36th annual acm
symposium on user interface software and technology, pages 1–22, 2023.
91. S. Peter and K. Riemer. Creative assistants with style: Making sense of generative ai as “style
engines”. InProceedings of the 57th Hawaii International Conference on System Sciences
(HICSS), pages 3980–3989, Hilton Hawaiian Village, Honolulu, Hawaii, USA, 2024. Hawaii
International Conference on System Sciences.
92. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al. Language models are
unsupervised multitask learners.OpenAI blog, 1(8):9, 2019.
93. A. Rapp, C. Di Lodovico, and L. Di Caro. How do people react to chatgpt’s unpredictable
behavior? anthropomorphism, uncanniness, and fear of ai: A qualitative study on individuals’
perceptions and understandings of llms’ nonsensical hallucinations.International Journal of
Human-Computer Studies, 198:103471, 2025.
94. B. Reitemeyer and H.-G. Fill. Applying large language models in knowledge graph-based
enterprise modeling: Challenges and opportunities.arXiv preprint arXiv:2501.03566, 2025.
95. S. Robinson.Simulation: the practice of model development and use (2nd ed.). Palgrave
Macmillan, 2014.
96. A. Ry ´s, L. Lima, J. Exelmans, D. Janssens, and H. Vangheluwe. Model management to
support systems engineering workflows using ontology-based knowledge graphs.Journal of
Industrial Information Integration, 42:100720, 2024.
97. A. Sabra, O. Schmitt, and J. Tyler. Assessing the quality and security of ai-generated code:
A quantitative analysis.arXiv preprint arXiv:2508.14727, 2025.
98. P. Sahoo, A. K. Singh, S. Saha, V. Jain, S. Mondal, and A. Chadha. A systematic survey of
prompt engineering in large language models: Techniques and applications.arXiv preprint
arXiv:2402.07927, 2024.
99. S. K. Sakib, S. W. Liddle, C. J. Lynch, A. Agrawal, and P. J. Giabbanelli. Rethinking
learning: The role of unlearning in generative ai-based conceptual modeling. InInternational
Conference on Conceptual Modeling, pages 24–44. Springer, 2025.
100. S. M. Sanchez, P. J. Sanchez, and H. Wan. Work smarter, not harder: A tutorial on designing
and conducting simulation experiments. In2020 Winter Simulation Conference (WSC), pages
1128–1142. IEEE, 2020.
101. S. M. Sanchez and H. Wan. Better than a petaflop: The power of efficient experimental design.
InProceedings of the 2009 Winter Simulation Conference (WSC), pages 60–74. IEEE, 2009.
102. D. C. Schmidt, J. Spencer-Smith, Q. Fu, and J. White. Towards a catalog of prompt patterns
to enhance the discipline of prompt engineering.ACM SIGAda Ada Letters, 43(2):43–51,
2024.
103. R. Schuerkamp and P. J. Giabbanelli. Extensions of fuzzy cognitive maps: a systematic
review.ACM Computing Surveys, 56(2):1–36, 2023.
104. R. Schuerkamp and P. J. Giabbanelli. Guiding evolutionary algorithms with large language
models to learn fuzzy cognitive maps.Neural Computing and Applications, 37(18):11891–
11908, 2025.
105. S. Schulhoff, M. Ilie, N. Balepur, K. Kahadze, A. Liu, C. Si, Y. Li, A. Gupta, H. Han,
S. Schulhoff, et al. The prompt report: a systematic survey of prompt engineering techniques.
arXiv preprint arXiv:2406.06608, 2024.
106. C. Sharma. Retrieval-augmented generation: A comprehensive survey of architectures, en-
hancements, and robustness frontiers.arXiv preprint arXiv:2506.00054, 2025.
107. C. Shi, H. Yang, D. Cai, Z. Zhang, Y. Wang, Y. Yang, and W. Lam. A thorough examination
of decoding methods in the era of llms.arXiv preprint arXiv:2402.06925, 2024.
108. D. Shi, J. Li, O. Meyer, and T. Bauernhansl. Enhancing retrieval-augmented generation
for interoperable industrial knowledge representation and inference toward cognitive digital
twins.Computers in Industry, 171:104330, 2025.

46 Philippe J. Giabbanelli
109. W. Shi, Y. Cui, Y. Wu, J. Fang, S. Zhang, M. Li, S. Han, J. Zhu, J. Xu, and X. Zhou. Semantic-
guided diverse decoding for large language model.arXiv preprint arXiv:2506.23601, 2025.
110. R. Singh, A. Hamilton, A. White, M. Wise, I. Yousif, A. Carvalho, Z. Shan, R. A. Baf,
M. Mayyas, L. A. Cavuoto, et al. A multimodal manufacturing safety chatbot: Knowledge
base design, benchmark development, and evaluation of multiple rag approaches.arXiv
preprint arXiv:2511.11847, 2025.
111. L. L. Snijder, Q. T. Smit, and M. H. de Boer. Advancing ontology alignment in the labor
market: Combining large language models with domain knowledge. InProceedings of the
AAAI Symposium Series, volume 3, pages 253–262, 2024.
112. Y. Song, G. Wang, S. Li, and B. Y. Lin. The good, the bad, and the greedy: Evaluation of llms
should not ignore non-determinism. InProceedings of the 2025 Conference of the Nations
of the Americas Chapter of the Association for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers), pages 4195–4206, 2025.
113. R. Soule and B. C. Ezell. A simulation-enabled framework for mission engineering problem
definition: Integrating ai-driven knowledge retrieval with human-centered design. In E. Azar,
A. Djanatliev, A. Harper, C. Kogler, V. Ramamohan, A. Anagnostou, and S. J. E. Taylor,
editors,Proceedings of the 2025 Winter Simulation Conference, 2025.
114. M. Steyvers, H. Tejeda, A. Kumar, C. Belem, S. Karny, X. Hu, L. W. Mayer, and P. Smyth.
What large language models know and what people think they know.Nature Machine
Intelligence, 7(2):221–231, 2025.
115. Z. Sun, U. Mendlovic, Y. Leviathan, A. Aharoni, J. H. Ro, A. Beirami, and A. T. Suresh.
Block verification accelerates speculative decoding.arXiv preprint arXiv:2403.10444, 2024.
116. E. Syriani, I. David, and G. Kumar. Screening articles for systematic reviews with chatgpt.
Journal of Computer Languages, 80:101287, 2024.
117. Y. Tang, D. Tuncel, C. Koerner, and T. Runkler. The few-shot dilemma: Over-prompting large
language models.arXiv preprint arXiv:2509.13196, 2025.
118. S. J. Taylor, T. Eldabi, T. Monks, M. Rabe, and A. M. Uhrmacher. Crisis, what crisis–does
reproducibility in modeling & simulation really matter? In2018 winter simulation conference
(WSC), pages 749–762. IEEE, 2018.
119. N. Tihanyi, T. Bisztray, M. A. Ferrag, R. Jain, and L. C. Cordeiro. How secure is ai-generated
code: A large-scale comparison of large language models.Empirical Software Engineering,
30(2):47, 2025.
120. M. S. Torkestani, A. Alameer, S. Palaiahnakote, and T. Manosuri. Inclusive prompt engineer-
ing for large language models: a modular framework for ethical, structured, and adaptive ai.
Artificial Intelligence Review, 58(11):348, 2025.
121. H. Vangheluwe and J. de Lara. Computer automated multi-paradigm modelling: Meta-
modelling and graph transformation. In S. Chick, P. J. S ´anchez, D. Ferrin, and D. J. Morrice,
editors,Proceedings of the 2003 Winter Simulation Conference, pages 595–603, New Orleans,
Louisiana, USA, 2003. IEEE Computer Society Press.
122. L. Vanh ´ee, M. Borit, P.-O. Siebers, R. Cremades, C. Frantz, ¨O. G¨ urcan, F. Kalvas, D. R. Kera,
V. Nallur, K. Narasimhan, et al. Large language models for agent-based modelling: Current
and possible uses across the modelling cycle.arXiv preprint arXiv:2507.05723, 2025.
123. A. Vijayakumar, M. Cogswell, R. Selvaraju, Q. Sun, S. Lee, D. Crandall, and D. Batra.
Diverse beam search for improved description of complex scenes. InProceedings of the
AAAI Conference on Artificial Intelligence, volume 32, 2018.
124. D. Wang, Z. Liu, S. Wang, Y. Ren, J. Deng, J. Hu, T. Chen, and H. Yang. Fier: Fine-grained
and efficient kv cache retrieval for long-context llm inference. InFindings of the Association
for Computational Linguistics: EMNLP 2025, pages 9702–9713, 2025.
125. X. Wang, S. Zhu, and X. Cheng. Speculative safety-aware decoding. InProceedings of
the 2025 Conference on Empirical Methods in Natural Language Processing, pages 12838–
12852, 2025.
126. A. Wei, N. Haghtalab, and J. Steinhardt. Jailbroken: How does llm safety training fail?
Advances in Neural Information Processing Systems, 36:80079–80110, 2023.

Large Language Models for Modeling & Simulation 47
127. L. Weidinger, J. Uesato, M. Rauh, C. Griffin, P.-S. Huang, J. Mellor, A. Glaese, M. Cheng,
B. Balle, A. Kasirzadeh, et al. Taxonomy of risks posed by language models. InProceedings
of the 2022 ACM conference on fairness, accountability, and transparency, pages 214–229,
2022.
128. D. G. Widder, M. Whittaker, and S. M. West. Why ‘open’ai systems are actually closed, and
why this matters.Nature, 635(8040):827–833, 2024.
129. K. Wu, E. Wu, K. Wei, A. Zhang, A. Casasola, T. Nguyen, S. Riantawan, P. Shi, D. Ho, and
J. Zou. An automated framework for assessing how well llms cite relevant medical references.
Nature Communications, 16(1):3615, 2025.
130. H. Xia, Z. Yang, Q. Dong, P. Wang, Y. Li, T. Ge, T. Liu, W. Li, and Z. Sui. Unlocking efficiency
in large language model inference: A comprehensive survey of speculative decoding. In L.-
W. Ku, A. Martins, and V. Srikumar, editors,Findings of the Association for Computational
Linguistics: ACL 2024, pages 7655–7671, Bangkok, Thailand, Aug. 2024. Association for
Computational Linguistics.
131. Y. Xia, D. Dittler, N. Jazdi, H. Chen, and M. Weyrich. Llm experiments with simulation:
Large language model multi-agent system for simulation model parametrization in digital
twins. In2024 IEEE 29th International Conference on Emerging Technologies and Factory
Automation (ETFA), pages 1–4. IEEE, 2024.
132. C. Xu, B. Wen, B. Han, R. Wolfe, L. L. Wang, and B. Howe. Do language models mirror
human confidence? exploring psychological insights to address overconfidence in LLMs. In
W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, editors,Findings of the Association
for Computational Linguistics: ACL 2025, pages 25655–25672, Vienna, Austria, July 2025.
Association for Computational Linguistics.
133. J. Xu, Z. Guo, H. Hu, Y. Chu, X. Wang, J. He, Y. Wang, X. Shi, T. He, X. Zhu, Y. Lv, Y. Wang,
D. Guo, H. Wang, L. Ma, P. Zhang, X. Zhang, H. Hao, Z. Guo, B. Yang, B. Zhang, Z. Ma,
X. Wei, S. Bai, K. Chen, X. Liu, P. Wang, M. Yang, D. Liu, X. Ren, B. Zheng, R. Men,
F. Zhou, B. Yu, J. Yang, L. Yu, J. Zhou, and J. Lin. Qwen3-omni technical report, 2025.
134. S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao. React: Synergizing
reasoning and acting in language models. InThe eleventh international conference on learning
representations, 2022.
135. X. Yu, P. Jian, and C. Chen. Tablerag: A retrieval augmented generation framework for
heterogeneous document reasoning. In C. Christodoulopoulos, T. Chakraborty, C. Rose,
and V. Peng, editors,Proceedings of the 2025 Conference on Empirical Methods in Natu-
ral Language Processing, pages 14063–14082, Suzhou, China, Nov. 2025. Association for
Computational Linguistics.
136. K. A. Yuksel, T. C. Ferreira, M. Al-Badrashiny, and H. Sawaf. A multi-ai agent system
for autonomous optimization of agentic ai solutions via iterative refinement and llm-driven
feedback loops. InProceedings of the 1st Workshop for Research on Agent Language Models
(REALM 2025), pages 52–62, 2025.
137. J. D. Zamfirescu-Pereira, R. Y. Wong, B. Hartmann, and Q. Yang. Why johnny can’t prompt:
how non-ai experts try (and fail) to design llm prompts. InProceedings of the 2023 CHI
conference on human factors in computing systems, pages 1–21, 2023.
138. S. Zarrieß, H. Voigt, and S. Sch¨ uz. Decoding methods in neural language generation: A
survey.Information, 12(9):355, 2021.
139. D. Zhang, S. Pan, T. Hoang, Z. Xing, M. Staples, X. Xu, L. Yao, Q. Lu, and L. Zhu. To be
forgotten or to be fair: Unveiling fairness implications of machine unlearning methods.AI
and Ethics, 4(1):83–93, 2024.
140. J. Zhang, D. Qiao, M. Yang, and Q. Wei. Regurgitative training: The value of real data in
training large language models.arXiv preprint arXiv:2407.12835, 2024.
141. Y. Zhang, R. Sun, Y. Chen, T. Pfister, R. Zhang, and S. Arik. Chain of agents: Large language
models collaborating on long-context tasks.Advances in Neural Information Processing
Systems, 37:132208–132237, 2024.
142. Z. Zhang, J. Li, Y. Lan, X. Wang, and H. Wang. An empirical study on prompt compression
for large language models.arXiv preprint arXiv:2505.00019, 2025.

48 Philippe J. Giabbanelli
143. J. Zhao, Y. Yang, X. Hu, J. Tong, Y. Lu, W. Wu, T. Gui, Q. Zhang, and X. Huang. Under-
standing parametric and contextual knowledge reconciliation within large language models.
InThe Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.
144. Y. Zhao, N. Vetter, and K. Aryan. Using large language models for ontoclean-based ontology
refinement.arXiv preprint arXiv:2403.15864, 2024.
145. S. Zhong, N. Japkowicz, F. Amblard, and P. J. Giabbanelli. A parameter-free model for
the online spread of far-right messages: Combining agent-based models with large-language
models. InComputational Science – ICCS 2025 Workshops: 25th International Conference,
Singapore, Singapore, July 7–9, 2025, Proceedings, Part II, page 208–223, Berlin, Heidelberg,
2025. Springer-Verlag.
146. S. Zhong, N. Japkowicz, and P. Giabbanelli. Do we Still Need People? Comparing Human and
LLM Personas in Political Modeling and Simulation . In2025 ACM/IEEE 28th International
Conference on Model Driven Engineering Languages and Systems Companion (MODELS-C),
pages 512–521, Los Alamitos, CA, USA, Oct. 2025. IEEE Computer Society.

Index
Ablation study, 3
Agentic AI, 4
Hyper-parameters
Decoding strategy, 9
Reasoning, 12, 35
Temperature, 9–11
top-p, 11
Jailbreaking, 5, 38
Knowledge augmentation
Contextual, 12–16
Parametric, 12–20
LLM evaluation
Adequacy, 28
Consistency, 28
LLM execution
Alias, 22–26
Attention, 22
Caching, 22
distributional bias, 20
Flex processing, 11
Forwarding, 22
inference non-determinism, 20
non-determinism, 20–26
Routing, 21–26
Speculative decoding, 21
LLM pipelines, 2–33
Low-Rank Adaptation (LoRA),
16–20, 32, 37MACH architectures, 13
Model Context Protocols (MCPs), 13
Multimodal LLM, 33
Open weights, 36
Orchestration, 3
parameter-efficient fine-tuning, 16
Prompt engineering
Cloze prompts, 36
Data decomposition, 39
Data representation, 6
Optimization, 8
Over-prompting, 4
Parsing outputs, 7
Prompt compression, 8
Showing prompts, 7
Step-by-step, 8
Style tranfer, 6
Task decomposition, 4
Validation prompt, 4
Retrieval-Augmented Generation
(RAG), 13–16, 36
Chunking, 16
De-duplication, 16
Metadata, 16
Reranking, 14
skills, 39
Unlearning, 34
49
# Key-Augmented Neural Triggers for Knowledge Sharing

**Authors**: Alex Wolf, Marco Edoardo Palma, Pooja Rani, Harald C. Gall

**Published**: 2025-08-05 11:40:56

**PDF URL**: [http://arxiv.org/pdf/2508.03340v1](http://arxiv.org/pdf/2508.03340v1)

## Abstract
Repository-level code comprehension and knowledge sharing remain core
challenges in software engineering. Large language models (LLMs) have shown
promise by generating explanations of program structure and logic. However,
these approaches still face limitations: First, relevant knowledge is
distributed across multiple files within a repository, aka semantic
fragmentation. Second, retrieval inefficiency and attention saturation degrade
performance in RAG pipelines, where long, unaligned contexts overwhelm
attention. Third, repository specific training data is scarce and often
outdated. Finally, proprietary LLMs hinder industrial adoption due to privacy
and deployment constraints. To address these issues, we propose Key-Augmented
Neural Triggers (KANT), a novel approach that embeds knowledge anchors into
both training and inference. Unlike prior methods, KANT enables internal access
to repository specific knowledge, reducing fragmentation and grounding
inference in localized context. Moreover, we synthesize specialized data
directly from code. At inference, knowledge anchors replace verbose context,
reducing token overhead and latency while supporting efficient, on premise
deployment. We evaluate KANT via: a qualitative human evaluation of the
synthesized dataset's intent coverage and quality across five dimensions;
compare against SOTA baselines across five qualitative dimensions and inference
speed; and replication across different LLMs to assess generalizability.
Results show that the synthetic training data aligned with information-seeking
needs. KANT achieved over 60% preference from human annotators and a LocalStack
expert (preferring 79% of cases). Also, KANT reduced inference latency by up to
85% across all models. Overall, it is well-suited for scalable, low-latency,
on-premise deployments, providing a strong foundation for code comprehension.

## Full Text


<!-- PDF content starts -->

Key-Augmented Neural Triggers for Knowledge Sharing
Alex Wolfa, Marco Edoardo Palmaa, Pooja Rania, Harald C. Galla
aDepartment of Informatics, University of Zurich, Zurich, Switzerland
Abstract
Repository-level code comprehension and knowledge sharing remain core challenges in software engineering. Large
language models (LLMs) have shown promise by generating explanations of program structure and logic. Retrieval-
Augmented Generation (RAG), the state-of-the-art (SOTA), improves relevance by injecting context at inference time.
However, these approaches still face limitations: First, semantic fragmentation across structural boundaries impairs
comprehension, as relevant knowledge is distributed across multiple files within a repository. Second, retrieval inef-
ficiency and attention saturation degrade performance in RAG pipelines, where long, weakly aligned contexts over-
whelm model attention. Third, repository specific training data is scarce, often outdated, incomplete or misaligned.
Finally, proprietary LLMs hinder industrial adoption due to privacy and deployment constraints. To address these
issues, we propose Key-Augmented Neural Triggers (KANT), a novel approach that embeds knowledge anchors,
symbolic cues linking code regions to semantic roles, into both training and inference. Unlike prior methods, KANT
enables internal access to repository specific knowledge, reducing fragmentation and grounding inference in local-
ized, semantically structured memory. Moreover, we synthesize specialized instruction tuning data directly from
code, eliminating reliance on noisy or outdated documentation and comments. At inference, knowledge anchors re-
place verbose context, reducing token overhead and latency while supporting e fficient, on premise deployment. We
evaluate KANT via: a qualitative human evaluation of the synthesized dataset’s intent coverage and quality across
five dimensions; compare against SOTA baselines across five qualitative dimensions and inference speed; and repli-
cation across di fferent LLMs to assess generalizability. Results show that the synthetic training data aligned with
information-seeking needs: over 90% of questions and answers were rated relevant and understandable; 77.34%,
69.53%, and 64.58% of answers were considered useful, accurate, and complete, respectively. KANT achieved over
60% preference from human annotators and a LocalStack expert over the baselines (e.g., 21% RAG) and notably the
expert preferred KANT in over 79% of cases. Also, KANT reduced inference latency by up to 85% across all models.
Overall, KANT demonstrated its e ffectiveness across all evaluated areas, implying that it is well-suited for scalable,
low-latency, on-premise deployments, providing a strong foundation for repository-level code comprehension.
Keywords: large language models, deep learning, repository-level, code comprehension, information-seeking
1. Introduction
Code comprehension remains one of the most fun-
damental, time-consuming, and cognitively demanding
tasks in software engineering [1, 2, 3]. It requires devel-
opers to acquire knowledge about a software system by
navigating software artifacts, reading source code, and
consulting related documentation [1]. Studies show that
developers spend nearly 58% of their time on such ac-
tivities, often interrupted by frequent context switches
Email addresses: wolf@ifi.uzh.ch (Alex Wolf ),
marcoepalma@ifi.uzh.ch (Marco Edoardo Palma ),
rani@ifi.uzh.ch (Pooja Rani ),gall@ifi.uzh.ch (Harald C.
Gall )between IDEs, browsers, and documentation platforms,
which disrupts cognitive flow and amplify e ffort [1].
Moreover, a significant portion of developer questions
focuses on seeking information to understand code be-
havior (37%), usage examples (28%), and locating def-
initions or instantiations (18%) [4].
For instance, a question like “Can you explain the
purpose of the ExecutionFailedEventDetails class
and how it’s used in the codebase?” requires archi-
tectural awareness, dependency tracking, and an under-
standing of relationships across code elements. Yet, an-
swering such questions is hard because relevant knowl-
edge is fragmented across the project structure, control
flow, and type hierarchies, each o ffering only a partial
Preprint submitted to Journal of Systems and Software August 6, 2025arXiv:2508.03340v1  [cs.SE]  5 Aug 2025

view of the codebase [5, 6, 7]. These challenges are
further compounded by outdated, incomplete, and in-
consistent resources, including forums, documentation,
community chats, and unreliable code comments, and
by the lack of memory cues that help surface relevant
information [8, 9, 10]. Overall, these challenges point
to a core limitation in comprehension workflows: se-
mantic fragmentation across structural boundaries and
a lack of an internal memory mechanism that connects
questions to the relevant regions of the codebase.
Recent work has explored the use of intelligent, in-
tegrated assistance through Large Language Models
(LLMs) in supporting code comprehension by interpret-
ing program structure, logic, and dependencies through
natural language interaction [11]. They can generate
human-readable explanations across diverse program-
ming languages [12, 13, 14], helping developers navi-
gate and reason about unfamiliar or complex code re-
gions. Retrieval Augmented Generation (RAG), State-
of-the-Art (SOTA) pipelines extend this capability by
incorporating additional context, such as external docu-
mentation or code snippets retrieved at inference time,
enabling more accurate and grounded answers.
However, despite these advances, fundamental limi-
tations remain. These models often hallucinate or mis-
interpret code behavior [15, 16], overlook key informa-
tion in long contexts due to the “lost in the middle” ef-
fect [17], and are sensitive to ambiguous or incomplete
context [18, 16]. Therefore, retrieved content can be
noisy, irrelevant, or poorly aligned with the user intent
[18, 16]. Specifically, RAG increases prompt length
and token consumption due to context fragmentation
and attention saturation in the retrieved context, often
degrading model performance [19]. These challenges
highlight a second core limitation of such pipelines:
retrieval ine fficiency and attention saturation , where
verbose and loosely aligned context overwhelms the
model’s attention window, fragments reasoning, and de-
grades answer quality [19, 16, 17].
Fine-tuning LLMs for specific domains or reposito-
ries o ffers a promising alternative to overcome these
limitations. However, it requires representative, high-
quality datasets that capture the structural and seman-
tic nuances of real-world codebases, which in practice
are scarce. Public corpora are often incomplete, unrep-
resentative of real codebases, outdated, or poorly doc-
umented [8, 20, 21, 22], while industrial repositories
are typically proprietary and inaccessible. Manually
curating up-to-date instruction tuning data at scale is
costly and labor-intensive. While RAG pipelines and
task-specific fine-tuning help mitigate context limita-
tions, they lack built-in mechanisms for context reten-tion without relying on external dependencies. This
leads to a third core limitation: repository specific data
scarcity and misalignment , the limited availability of
training corpora aligned with the underlying code se-
mantics specific to, and representative of, individual
repositories [8, 20, 22]. This scarcity limits fine-tuning
and long-term adaptability of comprehension models.
Collectively, these three limitations hinder the adapt-
ability, precision, and e fficiency of comprehension sys-
tems. To address them, we propose Key Augmented
Neural Trigger (KANT), a novel approach that embeds
internal knowledge anchors , symbolic cues that cap-
ture semantic associations between code regions and
their roles in the system. Unlike traditional approaches
that rely on context-based retrieval or documentation,
KANT enables models to retrieve repository-specific
knowledge internally, improving contextual grounding
and token e fficiency. Rooted in cognitive psychol-
ogy [23, 24], KANT reshapes model representations
during training to support more stable and distinctive
memory segments aligned with a repository’s structure.
KANT tackles repository-level comprehension by
embedding knowledge anchors directly into the model.
To mitigate repository-specific data scarcity, KANT
generates a synthetic instruction-tuning dataset directly
from source code, enabling scalable, up-to-date train-
ing without external documentation, manual annota-
tion, or external supervision. The dataset comprises
question-answer (QA) pairs used in a two-stage fine-
tuning pipeline. During Fill-in-the-Middle (FiM) train-
ing, code regions are augmented with knowledge an-
chors , allowing the model to associate structural con-
text with anchor keys. Subsequently, instruction tun-
ing reuses the same anchors alongside the QA pairs.
Together, these stages guide the model to form stable
associations between semantically related code regions,
addressing semantic fragmentation. At inference time,
the model is prompted using only the relevant knowl-
edge anchors, without injecting large in-prompt con-
text, thus enabling compact, memory-e fficient inputs.
This approach addresses retrieval ine fficiency and at-
tention saturation by replacing the construction of ver-
bose prompts and external context retrieval with anchor-
based memory access.
The approach is validated through a series of qual-
itative evaluations: assessing the quality and coverage
of the generated instruction dataset, comparing compre-
hension and inference performance against baselines,
and evaluating generalizability across model families.
In summary, the primary contributions of this paper are
the following (associated material is provided in the
replication package [25]):
2

•A validated automatic pipeline for generating
instruction-tuning data points directly from source
code, serving as a generalizable foundation for
fine-tuning downstream models for repository-
level code comprehension and knowledge sharing.
•The introduction of KANT, a reusable training ap-
proach that distills repository-specific knowledge
into internal symbolic anchors mined from code,
enabling LLMs to form resilient, context-rich
memory representations, reduce inference-time to-
ken consumption, and support fast, deployment-
friendly, more accurate, and context-related code
comprehension.
•An in-depth qualitative human evaluation of the
synthetic instruction tuning dataset (QA pairs) gen-
erated directly from source code, demonstrating
their semantic fidelity, diversity, and relevance to
real-world developer information needs.
•A comprehensive qualitative benchmark compar-
ing KANT to fine-tuned RAG-based baselines,
showing consistent gains in answer accuracy, com-
pleteness, inference latency, and other dimensions
across models.
The remainder of the paper is organized as follows:
Section 2 describes the KANT approach and training
process. Section 3 outlines our experimental design and
threats to validity. Section 4 presents evaluation results.
We discuss related work in Section 5 and future work in
Section 6. Finally, we conclude in Section 8.
2. Approach
This section introduces the KANT approach for
unblocking repository-specific code comprehension in
LLMs through the use of symbolic memory keys, re-
ferred to as knowledge anchors . The KANT approach is
designed to address the three core limitations outlined
in Section 1: (i) semantic fragmentation across struc-
tural boundaries and a lack of internal memory associ-
ations [4, 6, 16, 26], (ii) retrieval ine fficiency and atten-
tion saturation [15, 16, 18, 19, 17], and (iii) reposito-
ry-specific data scarcity and misalignment [8, 20, 22].
Embedding knowledge anchors during training, al-
lows KANT to equip models with structured internal
representations of code regions, enabling repository-
specific structural reasoning and symbolically grounded
memory access, thereby bypassing the need for verbose
retrieval, brittle documentation, or prompt-heavy con-
text injection. This design shifts comprehension fromexternal retrieval to e fficient internal memory access, re-
ducing dependence on misaligned auxiliary inputs.
KANT links each code region with a knowledge an-
chor that functions as a stable anchor during both train-
ing and inference. These anchors are embedded into
the model alongside source code and QA pairs, guid-
ing the model to form localized addressable memory
regions. Associating each code context with a distinct
key would help guide the model to condition its next-
token predictions on the relevant code region, increasing
the probability that subsequently generated tokens align
with the associated region. This structured separation
could help the model distinguish between semantically
distinct areas within a repository and retrieve relevant
information more e ffectively by referencing the appro-
priate key during inference. The knowledge anchoring
enables the model to distinguish between structurally
distinct, yet semantically similar regions, thereby reduc-
ing fragmentation across file boundaries and promoting
location-aware grounding of code semantics.
KANT builds on recent advances in code-pretrained
LLMs [27] and supports multiple LLMs families, in-
cluding CodeLlama [27] and Granite Code [28]. It
comprises of three components: a) synthetic training
data generation, b) a knowledge anchoring mechanism,
c) and a two-stage key-augmented training process.
The remainder of this section outlines the compo-
nents along with the training and inference protocols.
2.1. Synthetic training data generation
To address the scarcity of aligned, repository-specific
training data, KANT synthesizes QA pairs directly
from the relevant source code. Unlike existing data
generation approaches, such as CS1QA [29], and
ProCQA [30], which derive questions from code com-
ments or community platforms (e.g., GitHub, Stack
Overflow), KANT reduces reliance on external sources
that are frequently outdated, missing, or poorly aligned
with code semantics [8, 20, 22]. This enables the gen-
eration of semantically grounded and repository- and
task-specific QA pairs, with questions such as “Where is
the retry logic implemented?” or “What does this func-
tion return?”, tailored to comprehension needs [5, 4],
and suitable for reuse in downstream tasks like summa-
rization and search, where linking developer queries to
specific code regions is essential.
The process begins by partitioning the codebase into
token constrained segments, filtering out non-code arti-
facts (e.g., binaries, media). This improves over stan-
dard fine-tuning pipelines by tightly coupling training
data with the actual code structure, directly addressing
limitation iii regarding dataset alignment and scarcity.
3

Each code chunk ciis processed using structured
prompts to generate questions qiand corresponding an-
swers aifrom the same context, as shown in Figure 1.
This guarantees high alignment between training data
and the code itself. We formalize this as a genera-
tion functionG(ci,pi)→(qi,ai), where pidenotes the
chunk’s position within the repository and contributes
positional context to the generation process, resulting in
a dataset of tuples ( ci,qi,ai,pi), outlined in Table 1, that
encode both semantic and structural cues for training.
This pipeline reflects practical developer inquiries (e.g.,
“Where is the retry logic implemented?” or “What does
this function return?”) [5, 4], and eliminates the brittle-
ness of comment-based corpora by grounding question
generation directly in code semantics.
Table 1: Dataset statistics
Dataset Component Train Test
FiM dataset 2 ,596 -
Generated Q&A pairs 164 ,508 -
Deduplicated Q&A dataset 139 ,611 15,512
2.2. Key-augmented fine-tuning
KANT introduces knowledge anchors ka∈ K into
the model as memory keys tied to specific code re-
gions. These anchors function as cognitively inspired
cues [23, 24], conditioning the model’s behavior such
that predictions become more aligned with the seman-
tics of the anchored code region. Each training example
is expressed as a triple ( ka,x,y), where kais a knowl-
edge anchor, xis the input sequence (e.g., code or ques-
tion), and yis the prediction target. During training, the
anchor conditions the model’s internal state such that
the probability distribution over the next tokens shifts
toward completions that reflect the semantic and struc-
tural characteristics associated with the knowledge an-
chor. This mechanism enables the model to internalize
repository specific behaviors and produce responses that
remain consistent with localized semantics, even in the
absence of full in-context retrieval.
This mechanism mitigates the following: limitation
(i) by guiding the model to di fferentiate and relate over-
lapping or distributed code regions through knowledge
anchor base associations, and limitation (ii) by reduc-
ing the need for token intensive retrieval and reducing
context length at inference time
Training proceeds in two stages, illustrated in Fig-
ure 2: structural training using FiM objectives aug-
mented with knowledge anchors, and instruction tuning
using knowledge anchor-aware QA data.Fill-in-the-Middle training with knowledge anchors
In the first stage, the model is trained using a FiM
objective where each file is segmented into prefix cpre,
masked span cmid, and su ffixcpost. A knowledge anchor
derived from the file path (e.g., src/utils/math.py )
is prepended to each sample:
x=[KEY] ka[CTX] cpre<FILL>cpost, y=cmid.
Anchors are designed to be: (i) unique per artifact
within the repository, (ii) hierarchically meaningful, re-
flecting structural boundaries, and (iii) stable under mi-
nor changes. This trains the model to learn contextual
distinctions even across syntactically similar code and
enforces location-specific knowledge anchoring.
Instruction tuning with knowledge anchor aware QA
The second stage perform QA-style instruction tun-
ing, where each synthetic question-answer pair ( qi,ai)
is tagged with the same anchor kafrom the FiM stage:
x=[KEY] ka[Q] q, y=[A] a.
This conditioning aligns QA behavior with structural
memory representations established earlier. By train-
ing the model to answer questions in the presence of
knowledge anchors, we reduce hallucinations and pro-
mote consistency across questions that refer to similar
or overlapping content. Both stages use QLoRA [31]
for efficient low-rank adaptation, allowing the approach
to scale to large repositories and models under memory
constraints.
2.3. Inference protocol
At inference time, a user query q∗is paired with k
relevant knowledge anchors ka∗retrieved via seman-
tic similarity using a fine-tuned RAG mechanism. The
model then generates a response conditioned on this
lightweight input:
[KEY] ka∗[Q] q∗→ˆa
This inference structure allows the model to retrieve
previously learned internal knowledge tied to knowl-
edge anchors without relying on in-prompt source code
or external documentation. It e ffectively reduces con-
text size, and thus reduces token consumption result-
ing in lower latency and mitigating e ffects of the “lost-
in-the-middle” phenomenon ( i.e.,second limitation de-
scribed in section 1). Moreover, it decreases dependen-
cies on external resources, e.g., documentation, and mit-
igating issues related to ambiguous, incomplete, or mis-
leading context (limitation ii).
4

Answer generation
Input dataPrompt
Question generation
Code chunks RepositoryQuestion
datasetQ&A
datasetPrompt
You're a LocalStack developer. Only provide
Questions. No Answers.
                 {A|B}
Number each question starting with 1.
File location: {path}
Code: {context}
Questions:Write 20 different
questions that a
developer might ask
the codebase owner
based on the code
below.Write 20 different
search questions (such
as where can I find or
where is) about the code
below using the file
location.You're a LocalStack developer. Answer in a
concise and correct way. Always use Python if
asked for examples.
{question}
File location : {path}
Use the code below to answer.
Code: {context}
Evaluate
 Evaluate
TestTrainFigure 1: Training data generation and evaluation pipeline (for RQ1)
Ultimately, KANT o ffers a scalable and token-
efficient alternative for code comprehension in large and
evolving repositories, grounded in knowledge anchor-
ing and self-generated training data.
3. Experiments
This section presents the experimental design used
to evaluate the proposed KANT approach. We address
three research questions that assess the quality of syn-
thetically generated training data across multiple dimen-
sions, the inference performance of KANT compared to
SOTA baselines using qualitative criteria and inference
latency, and the generalization across model families.
RQ 1To what extent can pretrained LLMs generate
QA pairs directly from source code for the pur-
pose of code comprehension fine-tuning, and
what types of comprehension-relevant questions
do they produce?
To train LLMs for software engineering QA tasks,
prior datasets such as CodeQA [32], CS1QA [29], and
ProCQA [30] derive questions from external sources
such as code comments or community forums (e.g.,
StackOverflow). These sources are often outdated, in-
complete, or misaligned with code semantics, and re-
quire manual annotation or human supervision, which
limits scalability and adoption [8]. In this research ques-
tion, we investigate whether instruction-tuned LLMscan instead synthetically generate up-to-date and repos-
itory specific QA pairs directly from source code, by-
passing the need for external sources. Demonstrating
such a capability would represent a step toward au-
tomating repository-level comprehension pipelines by
enabling the systematic construction of training data
without manual annotation or supervision.
The focus lies on evaluating the syntactic and seman-
tic integrity of these synthetic QA pairs, along with
their practical utility for downstream comprehension
tasks. To address this, we generated a dataset of 164 ,508
QA pairs from the LocalStack repository1using prompt
based LLMs. LocalStack is a popular open-source tool
for offline AWS development and testing, maintained by
over 570 contributors, with ∼60,000 GitHub stars and a
codebase of∼3,000 mostly Python files.
To evaluate the quality and diversity of the generated
data, we conducted a structured human evaluation on
the QA pairs. We manually analyzed a representative
sample of 384 QA pairs with 95% confidence and a 5%
margin of error [33]. To ensure question uniqueness and
diversity, we applied pairwise clustering (textual simi-
larity threshold >0.8) using the NLTK library2prior to
sampling, resulting in 48 ,112 clusters.
Each QA pair was randomly and exclusively assigned
to one of the three evaluators ( ≈126 pairs each) to en-
1LocalStack: https://github.com/localstack/localstack/
commit/85b16b2 v3.7
2https://www.nltk.org/
5

Key Augmented Neural Triggers
Instruction fine-tuning
Input+Fill in the Middle
Input+
Repository
Q&A
datasetSource
codeInference
evaluation 1
Test set
Questions
 Compare & evaluate
approachesKANT2
Answerinternal
experts
RAG
BASE LLMInstruction
fine-tuningQ&A
datasetFigure 2: Two stage fine-tuning and evaluation pipeline (for RQ2&RQ3)
sure balanced workload, broad coverage, and reliable
quality assessment. If a pair was unclear or di fficult to
evaluate, a second evaluator independently reviewed it.
In case of disagreement, a third evaluator reviewed and
reached the decision using majority voting mechanism.
Each pair was evaluated along two complementary
dimensions: (i) Intent classification: To analyze the
purpose behind each question, we annotated its under-
lying intent using the taxonomy proposed by Liu and
Wan [32], which builds on prior work categorizing de-
veloper questions by functionality, purpose, and imple-
mentation, to support comprehension tasks [29, 34, 32].
This schema captures a broad range of developer infor-
mation needs. We extended it where needed to reflect
question types specific to our dataset, enabling struc-
tured analysis of intent diversity in the generated QA
pairs. (ii) QA quality assessment: Each QA pair was in-
dependently rated using binary criteria, three for ques-
tions and five for answers, where 1 indicates the cri-
terion was satisfied and 0 otherwise: relatedness (the
question is relevant to the code and the answer addresses
the question), usefulness (the question /answer provides
meaningful insights for comprehension), understand-
ability (the phrasing of question /answer is clear and
unambiguous), accuracy (the answer is factually correct
with respect to the source code), and completeness (the
answer contains all necessary details).
This evaluation provides empirically show the feasi-
bility and e ffectiveness of generating synthetic QA pairs
from source code and validates the resulting dataset forsupporting the KANT approach.
RQ 2How does KANT compare to a fine-tuned RAG
system (SOTA) in answering repository-specific
comprehension questions, with respect to an-
swer quality and inference e fficiency?
This question investigates whether KANT, by em-
bedding knowledge anchors, improves a model’s abil-
ity to answer repository-level comprehension questions.
LLMs, when used for software repositories, often strug-
gle to reason over fragmented information (semantic
fragmentation across structural boundaries) [17, 18, 15,
16]. While RAG remains SOTA in many tasks, it re-
lies on token-intensive retrieval methods that introduce
latency and reduce scalability. To this end, we evalu-
ate whether KANT o ffers measurable gains in answer
quality and inference e fficiency compared to (i) a SOTA
RAG pipeline and (ii) a base LLM without adaptation.
To answer this, we held out 10% of the generated
dataset as a test set and used the remaining 90% for
training. We then used the test set questions to com-
pare the performance of three configurations based on
the CodeLlama-7B [27] model: the KANT model fine-
tuned with knowledge anchors as described in Sec-
tion 2, a SOTA RAG approach using fine-tuned AoE
embeddings [35] and ChromaDB for top- ksemantic
retrieval ( k=5), and the original base model with-
out any adjustments. To ensure a fair comparison, we
adopted the original hyperparameter settings from prior
work [27, 35] wherever applicable. However, we re-
duced the batch size to 16 across all configurations to
accommodate GPU memory constraints. All models
were trained until convergence or for a maximum of 30
epochs; further details are provided in Table 2.
We evaluated the generated answers from each sys-
6

Table 2: Hyperparameter settings used in model comparisons
Parameter Values
FiM epochs 30
Instruction epochs 5
FiM-rate 0.5
top-k 5
context chunk size 3000
context window size 4000
batch size 16
tem on the held-out test set using a structured human
evaluation protocol. A statistically representative sam-
ple of 100 model outputs was selected based on a power
analysis for paired two-tailed t-tests ( α=0.05, e ffect
size=0.3, power =0.8) [33]. Each generated answer
was independently rated by one of three human evalua-
tors, randomly assigned to ensure unbiased distribution
and without access to the evaluations of their peers. In
parallel, a domain expert from the LocalStack team per-
formed an independent assessment of the same set of
generated answers, also blinded to the evaluators scores
to avoid confirmation bias. Each response was evaluated
on a 5-point Likert scale (1 =Low, 5 =High) across six
dimensions, similar to the quality assessment used to
judge the questions: preference (among KANT, RAG,
and Base), relatedness, understandability, usefulness,
accuracy, and completeness. This evaluation design en-
ables a rigorous comparison of model performance by
capturing the quality of the generated answers, their
consistency across similar questions, and the e ffective-
ness of each system in addressing repository-level com-
prehension tasks.
RQ 3To what extent does KANT generalize across
different underlying LLMs, and how does its
performance vary?
This question investigates the extent to which the ben-
efits of the KANT approach generalize across model
families of LLMs, which di ffer in training data scale,
pretraining recency, and hyperparameter configurations.
Although, prior work has demonstrated that adapting
LLMs to code-specific tasks can yield performance
gains [14, 11, 28], these gains are often sensitive to a
model’s pretraining distribution, tokenization scheme,
and context window size, raising concerns about robust-
ness and generalizability [36, 37, 38]. Accordingly, we
assess whether the improvements in answer quality and
inference e fficiency observed in RQ 2hold when applied
to a newer and more capable model. Demonstratingsuch generalization would provide evidence for the ro-
bustness of KANT, reinforcing the generalizability of
the knowledge anchoring approach, and highlighting its
potential for adaptation across future model families.
To this end, we replicated the experimental setup
from RQ 2, replacing CodeLlama with Granite Code-
8B [28], a LLM trained on a larger and more recent
corpus comprising approximately 4.5 trillion tokens of
code and technical content. For evaluation compara-
bility, prompt lengths were standardized across models.
All other aspects including training hyperparameters for
both KANT and RAG, inference protocols, and human
evaluation procedures were kept fixed to enable a con-
trolled and isolated comparison of outcomes.
4. Results
This section presents the results obtained from the ex-
periments conducted in Section 3. For each research
question, we analyze our approach’s output and evaluate
it using appropriate validation methods. Our evaluation
focuses on three key aspects: ( RQ 1) synthetic training
data quality, ( RQ 2) the e ffectiveness and e fficiency of
the KANT approach in comparison to RAG, and ( RQ 3)
its generalization across di fferent model families.
4.1. RQ 1: Quality of synthetic training data
We evaluated 384 randomly sampled QA pairs from
our generated data and assessed them along five binary
criteria: relatedness, understandability, usefulness, ac-
curacy, and completeness. Each pair was independently
rated by a human evaluator.
The generated questions demonstrated consistently
strong performance across all evaluation dimensions.
Specifically, we found that 97.66% of questions were
related to the associated code context, 95.57% were
understandable, and 88.02% were useful for compre-
hension or development purposes. These results indi-
cate that the LLM-generated questions generally target
meaningful aspects of the code and are phrased in a way
that aligns with typical developer questions. For exam-
ple, the question “How does the ‘Gateway‘ class handle
SSL certificates?” was rated positively across all dimen-
sions. It addresses common developer concerns around
security configuration, implementation specific details,
and operational behavior, and was judged as relevant,
understandable, and useful.
Compared to the questions, the quality of the gener-
ated answers showed greater variance. We found that
97.66% of answers were related to their correspond-
ing questions and 90.36% were understandable, only
7

77.34% were considered useful. Accuracy and com-
pleteness were rated lower, at 69.53% and 64.58% re-
spectively. These findings suggest that although the an-
swers were generally on topic, they often lacked depth
or failed to capture relevant details, particularly when
the question required a broader understanding of the
codebase. For example, in response to the previous
question about SSL certificate handling, the answer
misattributed responsibility, fabricated non-existent API
calls, and omitted critical details such as fallback mech-
anisms and caching behavior. Consequently, the re-
sponse was judged incomplete and only partially accu-
rate, illustrating the model’s di fficulty in synthesizing
information that spans multiple interacting components.
To understand the purpose behind the synthetic QA
pairs, we annotated each question using the intent tax-
onomy described in Section 3. As shown in Table 3,
most questions focused on code functionality, followed
by explanation and purpose. This distribution closely
aligns with findings from prior empirical studies [4, 5],
which report that developers most frequently ask ques-
tions about what code does, how it works, and why
it was written a certain way. We observed a simi-
lar trend in interrogative forms. Most questions began
with “how”, followed by “what”, while forms such as
“why”, “where”, and “who” appeared only occasion-
ally. This pattern is consistent with the nature of source
code, which more readily supports procedural and fac-
tual queries than those requiring design rationale or au-
thorship history. Prior studies have also shown that
“how” questions are among the most common in soft-
ware development settings [34, 39, 40, 5, 4]. The preva-
lence of functionality-oriented questions, such as “How
is retry logic implemented in this module?”, reflects
common developer concerns and confirms that the syn-
thetic data captures information seeking questions ob-
served on platforms such as Stack Overflow [34, 40, 39].
These results confirm that LLMs can generate high-
quality code-specific questions that align well with in-
formation seeking questions also observed on commu-
nity platforms like Stack Overflow [34, 40, 39]. Al-
though the answers are often relevant, accurate, and
understandable, their usefulness and completeness of-
ten su ffer due to restricted code visibility. That is,
the base model operates over isolated code segments
that may omit critical dependencies, related definitions,
or higher-level usage contexts necessary to synthesize
complete and precise answers. For example, answering
questions about architectural roles, control flow behav-
ior, or configuration logic may require reasoning across
multiple files or modules, which are not visible to the
model during generation. Despite these limitations inTable 3: Distribution of question types and intents in the evaluated
dataset
Category Count Type Count
functionality 146 how 248
explanation 93 what 90
purpose 91 explain 42
property 65 why 8
workflow 58 where 7
example usage 34 describe 4
programming concepts 18 which 4
relationship 13 when 4
referencing 10 whom 1
reasoning 6 who 1
best practices 4
knowledge recall 4
other 11
answer completeness, the overall quality of the syn-
thetic QA pairs demonstrates their feasibility as a source
of grounded, code-specific training data. These find-
ings underscore the need for improved generation mech-
anisms that account for broader structural and semantic
context, such as knowledge anchors to link questions
and answers to specific code regions and improve inter-
nal recall of repository-level semantics.
RQ 1– In summary: The evaluation demonstrates that
the synthetic questions are consistently relevant, well-
phrased, and aligned with typical developer concerns ob-
served in prior work [4, 5]. However, answer quality
varies more substantially, with notable deficiencies in
completeness and usefulness due to restricted code vis-
ibility. These findings establish the viability of LLM-
generated QA pairs directly from code for fine-tuning,
while also underscoring the need for more context-aware
methods such as KANT to improve coverage and depth.
4.2. RQ 2: KANT vs. RAG: Answer quality and infer-
ence efficiency
To evaluate the impact of knowledge anchors,
we compared three configurations based on the
CodeLlama-7B model: a base pretrained model with-
out fine-tuning, a SOTA RAG model fine-tuned on the
same QA data, and the KANT model fine-tuned using
the procedure described in Section 2. All systems were
evaluated on the same held-out test set.
Human evaluators consistently preferred KANT over
competing approaches. Across all responses, 62% of
answers generated by KANT were favored, compared
to 21% for RAG. The remaining 17% were either rated
8

in favor of the base model or marked as undecided. Un-
decided ratings occurred in two primary scenarios: ei-
ther all responses were equally unhelpful, such as in
response to speculative or poorly posed questions (for
example, “How does the Execution class’s Id property
relate to the Id property in the State class?”, where no
such relation exists in the code), or all answers were
judged equally correct. This second case was rare and
typically occurred with simpler questions that could be
answered from a single class or file. One such example
is, “What is the name of the table for single metrics in
your database?”, where all approaches gave satisfactory
answers. Table 4 reports the full breakdown, including
ratings from a LocalStack domain expert. The expert
preferred KANT in 87% of cases, compared to 10% for
RAG, with the remainder assigned to the base model or
marked undecided.
Table 4: Answer preferences by approaches: Evaluators vs. Expert
PreferenceCodeLlama Granite Code
Eval. Exp. Eval. Exp.
KANT 62% 87% 62% 79%
RAG 21% 10% 17% 16%
Base 5% 3% 7% 2%
Undecided 12% - 14% 3%
In addition to preference ratings, each answer was
scored on a 5-point Likert scale across five evaluation
dimensions: relatedness, understandability, usefulness,
accuracy, and completeness. As shown in Figure 3,
KANT outperformed the RAG baselines across all cri-
teria. In particular, knowledge anchoring led to gains
in completeness. The median completeness rating in-
creased from 2 .0±1.057 in the RAG configuration to
3.5±1.128 for KANT. Similarly, accuracy improved,
with the median rising from 2 .5±1.166 to 3.0±1.061,
and usefulness from 3 .0±1.139 to 4.0±1.078. These
results suggest that KANT helps the model generate
more precise and contextually complete responses. In
the Likert scale assessment, the expert rated KANT’s
answers with a median completeness of 4 .0±0.803, ac-
curacy of 4.0±0.838 and usefulness of 4 .0±0.792, com-
pared to RAG’s scores of 3 .0±0.893, 3.0±0.978, and
3.0±0.961, respectively.
Regarding the inference speed, KANT achieved
faster inference, where the average response time per
batch of 16 queries was 37.2 seconds, compared to
248.8 seconds for RAG. This represents an approx-
imately 85% reduction in latency. This speedup is
primarily due to KANT’s memory-based conditioning,
Relatedness
UnderstandabilityUsefulness AccuracyCompleteness12345Likert Score (1=Low, 5=High)
Evaluator
Relatedness
UnderstandabilityUsefulness AccuracyCompleteness
Expert
Model (Source)
RAG KANTFigure 3: Likert scale ratings for CodeLlama-based models.
which avoids the runtime overhead of full-context re-
trieval, as shown in Figure 4.
150200250
KANT RAG
Approach343638404244
Inference time (seconds / batch of 16)
Figure 4: Average inference time using CodeLlama. Lower is better.
RQ 2– In summary: KANT consistently outperforms
both the RAG and base configurations in answer qual-
ity, as confirmed by human and expert preference judg-
ments and Likert scale ratings across all dimensions. In-
ference latency was also significantly lower, with an 85%
reduction compared to RAG. These results demonstrate
that KANT is both an e ffective and e fficient approach for
repository-level code comprehension.
4.3. RQ 3: Generalizability across model families
To evaluate whether KANT’s benefits generalize be-
yond a single model family, we repeated the RQ 2exper-
iments using Granite Code-8B as the underlying model.
Both KANT and RAG were retrained on the same syn-
thetic QA dataset and evaluated on the same test set.
Results remained consistent with those observed for
CodeLlama. Human evaluators again preferred KANT
in 62% of cases, compared to 17% for RAG, with the
rest favoring the base model or indicating no prefer-
ence. The LocalStack expert showed a similar trend,
9

preferring KANT in 79% of cases and RAG in 16%.
This suggests that knowledge anchoring remains e ffec-
tive across model variants with varied training setups.
Likert scale ratings further support this finding. As
shown in Figure 5, KANT achieved higher or compara-
ble scores across all evaluation dimensions. Notably,
median completeness improved from 2 .0±0.955 for
RAG to 3.0±1.087 for KANT, while usefulness in-
creased from 3 .0±1.029 to 4.0±1.078. Accuracy
scores remained stable between models, both with a me-
dian of 3.0, although KANT showed slightly greater
variance (±1.175 vs.±1.152), indicating a broader
range of output specificity. Similarly, the LocalStack
expert assigned KANT a median completeness score of
4.0±0.905, accuracy of 4 .0±0.863, and usefulness score
of 4.0±0.89, compared to RAG’s scores of 3 .0±0.787,
3.0±0.935, and 3.0±0.902, respectively.
Relatedness
UnderstandabilityUsefulness AccuracyCompleteness12345Likert Score (1=Low, 5=High)
Evaluator
Relatedness
UnderstandabilityUsefulness AccuracyCompleteness
Expert
Model (Source)
RAG KANT
Figure 5: Likert scale ratings for Granite Code-based models.
The relative speedup of ≈80.9% was comparable to
the one observed with CodeLlama, reinforcing the gen-
eralizability of its e fficiency benefits.
KANT also maintained its inference time advantage
when deployed with Granite Code, as visualized in Fig-
ure 6. The average response time per batch of 16 queries
was 80.9% lower than that of the RAG configuration.
This result aligns with the latency reduction observed in
the CodeLlamaexperiments and reflects the e fficiency of
knowledge anchoring, which removes the need for dy-
namic context retrieval and prompt expansion.
These results suggest that the design principles under-
lying KANT, especially the use of knowledge anchor-
based memory for grounding, are robust to changes in
model architecture and remain e ffective with larger or
differently structured models.
100150200250300
KANT RAG
Approach404244464850
Inference time (seconds / batch of 16)Figure 6: Average inference time using Granite Code. Lower is better.
RQ 3– In summary: KANT generalizes e ffectively
across model families. When evaluated with Granite
Code-8B, it maintained its performance advantages in
both answer quality and inference latency. Improve-
ments in completeness and usefulness were particularly
pronounced, and evaluator preferences mirrored those
observed with CodeLlama. These results confirm that
knowledge anchoring provides a robust and transferable
mechanism for enhancing code comprehension across
modern LLMs.
5. Related work
Developer information needs and code search. Prior
work has extensively examined the types of questions
developers ask during software development, particu-
larly in the context of debugging, maintenance, and
comprehension tasks [41, 42, 43, 7, 6]. These stud-
ies show that developers often struggle to answer com-
plex, context-dependent questions about code, espe-
cially those requiring reasoning across files or compo-
nents. For instance, Ko et al. [41] highlight the signif-
icant e ffort developers invest in understanding unfamil-
iar code, while Sillito et al. [7] catalog a diverse ques-
tions, many involving relationships among distributed
elements. de Alwis and Murphy [6] describe such in-
quiries as “conceptual queries” that are inherently di ffi-
cult to resolve. Recent findings confirm that even with
access to LLM-based tools, developers continue to ask
explanation-seeking questions about code behavior, in-
tent, and design that current models fail to answer ef-
fectively [44, 45, 4]. Although sometimes treated as a
standalone task, code search is increasingly seen as cen-
tral to code comprehension, as shown by studies that
examine how developers use search to acquire knowl-
edge about software systems [13, 1, 4]. It supports ac-
10

tivities such as code reuse, bug localization, and explo-
ration [5, 46], and must operate over language-specific
syntax, semantic variation, and abstract program repre-
sentations [47]. Field studies further show that devel-
opers continue to struggle with identifying code usage,
architectural relationships, and rationale [4].
KANT addresses these gaps by jointly training on
comprehension and search-style questions anchored to
specific code regions, enabling models to reason over
repositories more e ffectively and capture relationships
spanning multiple components and files.
Question answering datasets for comprehension. To
support automated code comprehension and QA tasks,
benchmark datasets have emerged to evaluate both re-
trieval and question answering capabilities, often cate-
gorizing questions by intent, ( e.g., functionality or pur-
pose [34, 32, 29]). CodeSearchNet [47], focuses on
text-to-code retrieval, while datasets like CodeQA [32],
CS1QA [29], ProCQA [30], and CoReQA [48] derive
questions from code comments, GitHub issues, or com-
munity platforms like Stack Overflow. While these
datasets are valuable for training and evaluation, they
suffer from two key limitations. First, they rely heav-
ily on external artifacts such as comments, documenta-
tion, GitHub issues, or forum posts, which are often out-
dated, incomplete, or misaligned with the actual source
code [8, 20, 21, 22, 9]. Second, the resulting QA pairs
tend to be highly localized, typically scoped to a single
function or small code segment [48] and thus fail to cap-
ture broader semantic and structural reasoning across
files or modules, which is essential for addressing the
diverse set of developer information needs.
The KANT approach fundamentally di ffers from
prior works. Instead of mining documentation or ex-
ternal sources, KANT generates synthetic QA pairs di-
rectly from the code itself using structured prompt tem-
plates for comprehension and search. This enables
up-to-date, internally consistent training data that re-
flects divers developer information needs. Furthermore,
KANT uses these QA pairs not just for evaluation, but
to fine-tune LLMs using embedded knowledge anchors
that link questions to specific code elements.
Use of LLMs in code comprehension LLMs have been
applied to software tasks such as code explanation [45],
program repair [49], and interactive development [50].
Several studies focus on their use in QA settings, par-
ticularly using RAG to combine retrieved information
with generative models [44, 51]. However, RAG per-
formance depend on retrieval quality, which may de-
grade in the absence of comprehensive documentation
or when applied to proprietary code. Moreover, they
often inflate context windows with long, noisy inputs,amplifying the lost-in-the-middle effect [17, 16].
In contrast, KANT integrates knowledge anchors , in-
ternal associations between question and code context,
directly into the LLM’s parameters. This design reduces
token usage at inference, improves precision, and avoids
external dependency. Similar to Alassan et al. [52],
who demonstrate the viability of open-source LLMs
for on-premise deployment in privacy-sensitive settings,
KANT supports fully local fine-tuning and inference to
ensure data confidentiality and deployment autonomy.
Parameter-e fficient fine-tuning. To minimize the
computational cost of repository-specific adaptation,
we fine-tune all models using QLoRA, a quantization-
aware variant of Low-Rank Adaptation (LoRA) [53]
that enables e fficient fine-tuning on 4-bit quantized
weights with minimal memory overhead [54]. QLoRA
combines trainable low-rank updates with paged op-
timizers to preserve performance while operating un-
der strict hardware constraints, making it well-suited
for large models in limited-resource environments. We
use QLoRA for fine-tuning all models to maximize e ffi-
ciency without compromising performance.
6. Future work
This section discusses implications and several
promising directions that remain for future research.
An improvement direction is refining the granularity
of knowledge anchors beyond the current file level. An-
choring at finer levels, such as methods, class or type
declarations, or control-flow blocks could improve re-
trieval and reasoning over semantically localized code
regions. This is especially useful for answering devel-
oper questions tied to detailed implementation seman-
tics, including best practices, edge-cases, or parameter-
specific logic. Prior work has shown the prevalence of
such questions and their reliance on small but critical
code elements, including method preconditions, control
flow paths, and usage constraints [42, 7, 43]. Finer-
grained anchors could thus improve both retrieval accu-
racy and response specificity by enabling more precise
access to structurally rich scoped code elements.
Another promising direction enriching knowledge
anchors with contextual metadata from auxiliary arti-
facts, such as version control, issue trackers, or own-
ership graphs. This would enable reasoning about au-
thorship, design intent, and maintenance activity, as-
pects not captured in source code alone. Such ques-
tions have been observed in prior work [43], yet remain
difficult to address in KANT’s current setup, as it re-
lies purely on source code. Augmenting anchors with
this metadata could help the model answer historically
11

and organizationally grounded queries otherwise inac-
cessible. A further avenue involves adapting KANT
to reasoning-centric /chain-of-thought (CoT) models.
Integrating knowledge anchors with CoT prompting or
structured memory access could support deeper, multi-
hop reasoning over complex repositories, improving co-
herence across answers and promoting more consistent
retrieval of interconnected knowledge.
The generalizability of KANT across programming
languages also warrants further investigation. While
the current evaluation focused on a Python-based repos-
itory, extending the approach to languages such as
JavaScript, C ++, and Java will help assess whether
knowledge anchors remain stable and precise across
different syntactic and semantic structures. Language-
specific benchmarks will be essential for evaluating
KANT’s robustness in more structurally diverse and
syntactically varied environments.
Supporting evolving codebases is another key con-
cern. To maintain model accuracy over time, we plan
to explore incremental fine-tuning and memory up-
date strategies that allow KANT to adapt to reposi-
tory changes without requiring full retraining, allow-
ing CI /CD integration. Lastly, a deployment toolkit is
planned to facilitate ease of adoption.
Collectively, these directions will further improve the
utility of KANT, enabling more intelligent, scalable,
and context-aware developer tools grounded in LLMs.
7. Threats to Validity
We outline the potential threats to the validity of our
findings and describe the steps taken to mitigate them.
Construct validity. Construct validity concerns
whether the evaluation metrics and procedures accu-
rately capture the intended phenomena, in this case, QA
quality and comprehension e fficacy. Since human eval-
uations inherently involve subjective interpretation, we
mitigated this risk by providing detailed scoring rubrics
and assigning each answer to one of three independent
human evaluators. A LocalStack domain expert inde-
pendently rated the same set of responses without ac-
cess to reviewer scores, providing an additional calibra-
tion point. For RQ 1andRQ 2, QA pairs were randomly
assigned to evaluators to minimize bias introduced by
evaluator familiarity or systematic ordering e ffects.
Internal validity. The quality of generated QA data
and the downstream performance of fine-tuned models
are influenced by the design of the prompts used in gen-
eration. To minimize this threat, we conducted a prelim-
inary prompt optimization process through trial and er-
ror. For instance, we empirically found that splitting thequestion generation task into two targeted subprompts,
one for general understanding and one for search related
comprehension tasks, improved specificity and coher-
ence. Although this iterative refinement was not for-
malized as a chain-of-thought search, it followed a pro-
gressive, outcome-guided tuning process. Nevertheless,
some residual bias in prompt formulation may persist.
External validity. Our experiments focus exclu-
sively on the LocalStack codebase, a large-scale, open-
source Python project representative of modern cloud
infrastructure systems. While this provides a realis-
tic and high-impact evaluation context, generalization
to other ecosystems, such as statically typed languages
(e.g., Java, C ++) or front-end focused languages (e.g.,
JavaScript), remains untested. We leave cross-language
replication and multi-domain benchmarking as impor-
tant directions for future work.
Conclusion validity. Although we followed standard
statistical practices in determining sample sizes (e.g.,
95% confidence with 5% margin of error), practical
constraints such as annotator availability limited the ab-
solute scale of human evaluation. We addressed this by
combining reviewer ratings with independent expert as-
sessments, and by using a structured Likert scale frame-
work to collect detailed judgments across dimensions.
We deliberately excluded automated QA metrics (e.g.,
BLEU, ROUGE) due to their known misalignment with
human judgment in code-oriented tasks [55, 51].
Tooling limitations. The RAG baseline depends on
both the embedding model and the retrieval backend.
While AoE embeddings [35] o ffer a strong, lightweight
baseline, performance may vary with di fferent embed-
ding methods or di fferent chunking strategies.
Data generation bias. All QA data was synthe-
sized using Llama-2-Chat 7B. Although the model pro-
vides high-quality generation capabilities, it reflects
limitations of its training data and decoding strategies.
Employing more recent or specialized code-generation
models could yield better synthetic data and potentially
stronger downstream performance. Exploring this axis
remains a compelling avenue for future work.
Model compatibility. The KANT approach relies on
FiM capabilities, which limits compatibility to mod-
els such as CodeLlama and Granite Code. Adapting
KANT to models without FiM support would require ar-
chitectural or pretraining modifications [56], which may
impact ease of adoption in practice.
8. Conclusion
This work introduces KANT, a novel, lightweight,
and e fficient approach for repository-specific code com-
12

prehension in LLMs. KANT advances automated
knowledge sharing in software repositories through:
(i) a validated and scalable pipeline that synthesizes
instruction-tuning datasets directly from source code.
These datasets, composed of QA pairs, eliminate re-
liance on outdated, misaligned, or labor-intensive man-
ually constructed datasets, reducing the need for ex-
tensive human supervision, and (ii) a fine-tuning strat-
egy that embeds symbolic knowledge anchors into both
training and inference, enabling models to retrieve se-
mantically grounded, repository-internal context with-
out dependence on external documentation, long in–
context prompts, or excessive token overhead.
Evaluations with CodeLlama and Granite Code,
showed the e ffectiveness of this approach. Most no-
tably, the LocalStack expert preferred KANT-generated
answers in over 79% of cases, highlighting strong repos-
itory specific alignment. Human annotators likewise
favored KANT in over 60% of comparisons, confirm-
ing its e ffectiveness in repository level code compre-
hension. These preferences were further supported by
Likert scale evaluations from both the expert and anno-
tators, indicating consistent gains in completeness, use-
fulness, and accuracy. Compared to the SOTA base-
line, KANT consistently produced more coherent and
semantically grounded responses. In addition, KANT
reduced inference latency by up to 85%. These findings
indicate a suitability for low latency, on premise envi-
ronments.
KANT guides the model to develop localized mem-
ory regions that align with semantically meaningful
code units, thereby increasing the likelihood that sub-
sequent token predictions remain anchored to relevant
knowledge. This enables repository-specific compre-
hension without the token overhead associated with
long in-prompt contexts.
By mitigating attention saturation and reducing sus-
ceptibility to hallucinations, this strategy provides a
lightweight yet e ffective mechanism for enabling pre-
cise and e fficient inference in LLMs.
In summary, KANT presents a practical and e fficient
LLM-based pipeline for code comprehension. By em-
bedding knowledge anchors directly into the model, it
improves answer informativeness and semantic preci-
sion while reducing inference costs. These contribu-
tions establish KANT as a practical and scalable foun-
dation for precise and context-aware developer tools.
Acknowledgments
The research leading to these results has been sup-
ported by the Swiss National Science Foundation(SNSF) through Grant SNSF204632.
References
[1] X. Xia, L. Bao, D. Lo, Z. Xing, A. E. Hassan, S. Li, Measuring
program comprehension: A large-scale field study with profes-
sionals, IEEE Transactions on Software Engineering 44 (2018)
951–976. doi: 10.1109/TSE.2017.2734091 .
[2] M. Hansen, R. L. Goldstone, A. Lumsdaine, What makes code
hard to understand?, arXiv preprint arXiv:1304.5257 (2013).
[3] Y . Zi, L. Li, A. Guha, C. J. Anderson, M. Q. Feldman, ” i would
have written my code di fferently”: Beginners struggle to un-
derstand llm-generated code, arXiv preprint arXiv:2504.19037
(2025).
[4] K. T. Stolee, T. Welp, C. Sadowski, S. Elbaum, 10 years
later: Revisiting how developers search for code, Proc.
ACM Softw. Eng. 2 (2025). URL: https://doi.org/10.1145/
3715774 . doi: 10.1145/3715774 .
[5] C. Sadowski, K. T. Stolee, S. Elbaum, How developers search
for code: a case study, in: Proceedings of the 2015 10th joint
meeting on foundations of software engineering, 2015, pp. 191–
201.
[6] B. de Alwis, G. C. Murphy, Answering conceptual queries
with ferret, in: Proceedings of the 30th International
Conference on Software Engineering, ICSE ’08, Association
for Computing Machinery, New York, NY , USA, 2008, p.
21–30. URL: https://doi.org/10.1145/1368088.1368092 .
doi:10.1145/1368088.1368092 .
[7] J. Sillito, G. C. Murphy, K. De V older, Questions program-
mers ask during software evolution tasks, in: Proceedings of
the 14th ACM SIGSOFT International Symposium on Founda-
tions of Software Engineering, SIGSOFT ’06 /FSE-14, Associa-
tion for Computing Machinery, New York, NY , USA, 2006, p.
23–34. URL: https://doi.org/10.1145/1181775.1181779 .
doi:10.1145/1181775.1181779 .
[8] P. Rani, A. Blasi, N. Stulova, S. Panichella, A. Gorla, O. Nier-
strasz, A decade of code comment quality assessment: A sys-
tematic literature review, Journal of Systems and Software
195 (2023) 111515. URL: https://www.sciencedirect.com/
science/article/pii/S0164121222001911 . doi: https://
doi.org/10.1016/j.jss.2022.111515 .
[9] Z. Gu, J. Wu, J. Liu, M. Zhou, M. Gu, An empirical
study on API-misuse bugs in open-source C programs, in:
2019 IEEE 43rd Annual Computer Software and Applica-
tions Conference (COMPSAC), volume 1, 2019, pp. 11–20.
URL: https://ieeexplore.ieee.org/document/8754426?
utmsource=chatgpt.com . doi: 10.1109/COMPSAC.2019.
00012 .
[10] F. Fronchetti, D. C. Shepherd, I. Wiese, C. Treude, M. A.
Gerosa, I. Steinmacher, Do contributing files provide in-
formation about oss newcomers’ onboarding barriers?, in:
Proceedings of the 31st ACM Joint European Software En-
gineering Conference and Symposium on the Foundations
of Software Engineering, ESEC /FSE 2023, Association for
Computing Machinery, New York, NY , USA, 2023, p.
16–28. URL: https://doi.org/10.1145/3611643.3616288 .
doi:10.1145/3611643.3616288 .
[11] X. Hou, Y . Zhao, Y . Liu, Z. Yang, K. Wang, L. Li, X. Luo,
D. Lo, J. C. Grundy, H. Wang, Large language models for
software engineering: A systematic literature review, ArXiv
abs/2308.10620 (2023). URL: https://api.semanticscholar.
org/CorpusID:261048648 .
[12] C.-Y . Su, A. Bansal, Y . Huang, T. J.-J. Li, C. McMillan,
13

Context-aware code summary generation, arXiv preprint
arXiv:2408.09006 (2024).
[13] M. Allamanis, E. T. Barr, P. Devanbu, C. Sutton, A survey of
machine learning for big code and naturalness, ACM Comput.
Surv. 51 (2018). URL: https://doi.org/10.1145/3212695 .
doi:10.1145/3212695 .
[14] W. Ahmad, S. Chakraborty, B. Ray, K.-W. Chang, Unified pre-
training for program understanding and generation, in: Pro-
ceedings of the 2021 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Lan-
guage Technologies, Association for Computational Linguis-
tics, 2021, pp. 2655–2668. URL: https://www.aclweb.org/
anthology/2021.naacl-main.211 .
[15] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y . Xu, E. Ishii, Y . J. Bang,
A. Madotto, P. Fung, Survey of hallucination in natural language
generation, ACM Comput. Surv. 55 (2023). URL: https://doi.
org/10.1145/3571730 . doi: 10.1145/3571730 .
[16] B. He, N. Chen, X. He, L. Yan, Z. Wei, J. Luo, Z.-H.
Ling, Retrieving, rethinking and revising: The chain-of-
verification can improve retrieval augmented generation, in:
Y . Al-Onaizan, M. Bansal, Y .-N. Chen (Eds.), Findings of
the Association for Computational Linguistics: EMNLP 2024,
Association for Computational Linguistics, Miami, Florida,
USA, 2024, pp. 10371–10393. URL: https://aclanthology.
org/2024.findings-emnlp.607/ . doi: 10.18653/v1/2024.
findings-emnlp.607 .
[17] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua,
F. Petroni, P. Liang, Lost in the middle: How language mod-
els use long contexts, Transactions of the Association for
Computational Linguistics 12 (2024) 157–173. URL: https:
//aclanthology.org/2024.tacl-1.9/ . doi: 10.1162/tacl_a_
00638 .
[18] A. Mallen, A. Asai, V . Zhong, R. Das, D. Khashabi, H. Ha-
jishirzi, When not to trust language models: Investigating ef-
fectiveness of parametric and non-parametric memories, in:
A. Rogers, J. Boyd-Graber, N. Okazaki (Eds.), Proceedings
of the 61st Annual Meeting of the Association for Computa-
tional Linguistics (V olume 1: Long Papers), Association for
Computational Linguistics, Toronto, Canada, 2023, pp. 9802–
9822. URL: https://aclanthology.org/2023.acl-long.546/ .
doi:10.18653/v1/2023.acl-long.546 .
[19] M. Levy, A. Jacoby, Y . Goldberg, Same task, more to-
kens: the impact of input length on the reasoning perfor-
mance of large language models, in: L.-W. Ku, A. Martins,
V . Srikumar (Eds.), Proceedings of the 62nd Annual Meet-
ing of the Association for Computational Linguistics (V olume
1: Long Papers), Association for Computational Linguistics,
Bangkok, Thailand, 2024, pp. 15339–15353. URL: https://
aclanthology.org/2024.acl-long.818/ . doi: 10.18653/v1/
2024.acl-long.818 .
[20] M. Guizani, A. Chatterjee, B. Trinkenreich, M. E. May,
G. J. Noa-Guevara, L. J. Russell, G. G. Cuevas Zambrano,
D. Izquierdo-Cortazar, I. Steinmacher, M. A. Gerosa, A. Sarma,
The long road ahead: Ongoing challenges in contributing to
large OSS organizations and what to do 5 (2021) 407:1–407:30.
URL: https://dl.acm.org/doi/10.1145/3479551 . doi: 10.
1145/3479551 .
[21] E. Larios Vargas, M. Aniche, C. Treude, M. Bruntink,
G. Gousios, Selecting third-party libraries: the practition-
ers’ perspective, in: Proceedings of the 28th ACM Joint
Meeting on European Software Engineering Conference and
Symposium on the Foundations of Software Engineering, ES-
EC/FSE 2020, Association for Computing Machinery, 2020, pp.
245–256. URL: https://dl.acm.org/doi/10.1145/3368089.
3409711 . doi: 10.1145/3368089.3409711 .[22] W. S. Tan, M. Wagner, C. Treude, Wait, wasn’t that code
here before? detecting outdated software documentation, in:
2023 IEEE International Conference on Software Maintenance
and Evolution (ICSME), 2023, pp. 553–557. URL: https:
//ieeexplore.ieee.org/abstract/document/10336349 .
doi:10.1109/ICSME58846.2023.00071 .
[23] A. W. Melton, Implications of short-term memory for a gen-
eral theory of memory, Journal of Verbal Learning and Verbal
Behavior 2 (1963) 1–21. URL: https://api.semanticscholar.
org/CorpusID:28308229 .
[24] R. Reed Hunt, Two contributions of distinctive processing to
accurate memory, Journal of Memory and Language 48 (2003)
811–825. URL: https://www.sciencedirect.com/science/
article/pii/S0749596X03000184 . doi: https://doi.org/
10.1016/S0749-596X(03)00018-4 .
[25] A. Wolf, M. E. Palma, P. Rani, H. C. Gall, Replication Package,
2025. URL: https://doi.org/10.5281/zenodo.16566076 .
[26] F. L. de la Mora, S. Nadi, An empirical study of metric-
based comparisons of software libraries, in: Proceedings of the
14th International Conference on Predictive Models and Data
Analytics in Software Engineering, PROMISE’18, Association
for Computing Machinery, 2018, pp. 22–31. URL: https://dl.
acm.org/doi/10.1145/3273934.3273937 . doi: 10.1145/
3273934.3273937 .
[27] B. Roziere, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. E. Tan,
Y . Adi, J. Liu, R. Sauvestre, T. Remez, et al., Code llama: Open
foundation models for code, arXiv preprint arXiv:2308.12950
(2023).
[28] M. Mishra, M. Stallone, G. Zhang, Y . Shen, A. Prasad, A. M.
Soria, M. Merler, P. Selvam, S. Surendran, S. Singh, et al., Gran-
ite code models: A family of open foundation models for code
intelligence, arXiv preprint arXiv:2405.04324 (2024).
[29] C. Lee, Y . Seonwoo, A. Oh, Cs1qa: A dataset for assisting
code-based question answering in an introductory programming
course, arXiv preprint arXiv:2210.14494 (2022).
[30] Z. Li, J. Zhang, C. Yin, Y . Ouyang, W. Rong, Procqa: A
large-scale community-based programming question answer-
ing dataset for code search, arXiv preprint arXiv:2403.16702
(2024).
[31] T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer, Qlora:
Efficient finetuning of quantized llms, Advances in Neural In-
formation Processing Systems 36 (2024).
[32] C. Liu, X. Wan, CodeQA: A question answering dataset
for source code comprehension, in: M.-F. Moens,
X. Huang, L. Specia, S. W.-t. Yih (Eds.), Findings of
the Association for Computational Linguistics: EMNLP
2021, Association for Computational Linguistics, Punta
Cana, Dominican Republic, 2021, pp. 2618–2632. URL:
https://aclanthology.org/2021.findings-emnlp.223 .
doi:10.18653/v1/2021.findings-emnlp.223 .
[33] S. L. Lohr, Sampling: Design and Analysis, 3rd ed., CRC Press,
2021.
[34] M. Allamanis, C. Sutton, Why, when, and what: analyzing stack
overflow questions by topic, type, and code, in: 2013 10th
Working conference on mining software repositories (MSR),
IEEE, 2013, pp. 53–56.
[35] X. Li, J. Li, AoE: Angle-optimized embeddings for seman-
tic textual similarity, in: L.-W. Ku, A. Martins, V . Srikumar
(Eds.), Proceedings of the 62nd Annual Meeting of the Associ-
ation for Computational Linguistics (V olume 1: Long Papers),
Association for Computational Linguistics, Bangkok, Thailand,
2024, pp. 1825–1839. URL: https://aclanthology.org/2024.
acl-long.101/ . doi: 10.18653/v1/2024.acl-long.101 .
[36] V . Aryabumi, Y . Su, R. Ma, A. Morisot, I. Zhang, A. Lo-
catelli, M. Fadaee, A. ”Ust ” un, S. Hooker, To code, or not to
14

code? exploring impact of code in pre-training, arXiv preprint
arXiv:2408.10914 (2024).
[37] G. Dagan, G. Synnaeve, B. Rozi‘ere, Getting the most out
of your tokenizer for pre-training and domain adaptation, in:
Proceedings of the 41st International Conference on Machine
Learning, ICML’24, JMLR.org, 2024.
[38] Y . Ding, L. L. Zhang, C. Zhang, Y . Xu, N. Shang, J. Xu, F. Yang,
M. Yang, Longrope: extending llm context window beyond 2
million tokens, in: Proceedings of the 41st International Con-
ference on Machine Learning, ICML’24, JMLR.org, 2024.
[39] P. Rani, M. Birrer, S. Panichella, M. Ghafari, O. Nierstrasz,
What do developers discuss about code comments?, in: 2021
IEEE 21st International Working Conference on Source Code
Analysis and Manipulation (SCAM), IEEE, 2021, pp. 153–164.
[40] S. Beyer, C. Macho, M. Di Penta, M. Pinzger, What kind of
questions do developers ask on stack overflow? a comparison of
automated approaches to classify posts into question categories,
Empirical Software Engineering 25 (2020) 2258–2301.
[41] A. J. Ko, R. DeLine, G. Venolia, Information needs in collocated
software development teams, in: 29th International Conference
on Software Engineering (ICSE’07), IEEE, 2007, pp. 344–353.
[42] T. D. LaToza, B. A. Myers, Hard-to-answer questions about
code, in: Evaluation and usability of programming languages
and tools, 2010, pp. 1–6.
[43] T. Fritz, G. C. Murphy, Using information fragments to an-
swer the questions developers ask, in: Proceedings of the 32nd
ACM /IEEE International Conference on Software Engineering-
V olume 1, 2010, pp. 175–184.
[44] E. A. Haque, C. Brown, T. D. LaToza, B. Johnson, Information
seeking using ai assistants, arXiv preprint arXiv:2408.04032
(2024).
[45] J. Richards, M. Wessel, What you need is what you get: Theory
of mind for an llm-based code understanding assistant, arXiv
preprint arXiv:2408.04477 (2024).
[46] L. Di Grazia, M. Pradel, Code search: A survey of techniques
for finding code, ACM Computing Surveys 55 (2023) 1–31.
[47] H. Husain, H.-H. Wu, T. Gazit, M. Allamanis, M. Brockschmidt,
Codesearchnet challenge: Evaluating the state of semantic code
search, arXiv preprint arXiv:1909.09436 (2019).
[48] J. Chen, K. Zhao, J. Liu, C. Peng, J. Liu, H. Zhu, P. Gao, P. Yang,
S. Deng, Coreqa: Uncovering potentials of language models in
code repository question answering, 2025. URL: https://arxiv.
org/abs/2501.03447 .arXiv:2501.03447 .
[49] N. Nashid, M. Sintaha, A. Mesbah, Retrieval-based prompt se-
lection for code-related few-shot learning, in: 2023 IEEE /ACM
45th International Conference on Software Engineering (ICSE),
IEEE, 2023, pp. 2450–2462.
[50] D. Nam, A. Macvean, V . Hellendoorn, B. Vasilescu, B. Myers,
Using an llm to help with code understanding, in: Proceedings
of the IEEE /ACM 46th International Conference on Software
Engineering, 2024, pp. 1–13.
[51] Y . Hicke, A. Agarwal, Q. Ma, P. Denny, Chata: Towards an
intelligent question-answer teaching assistant using open-source
llms, arXiv preprint arXiv:2311.02775 (2023).
[52] M. S. Y . Alassan, J. L. Espejel, M. Bouhandi, W. Dahhane,
E. H. Ettifouri, Benchmarking open-source language models
for efficient question answering in industrial applications, arXiv
preprint arXiv:2406.13713 (2024).
[53] E. J. Hu, Y . Shen, P. Wallis, Z. Allen-Zhu, Y . Li, S. Wang,
L. Wang, W. Chen, et al., Lora: Low-rank adaptation of large
language models., ICLR 1 (2022) 3.
[54] T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer, Qlora:
efficient finetuning of quantized llms, in: Proceedings of the
37th International Conference on Neural Information Process-
ing Systems, NIPS ’23, Curran Associates Inc., Red Hook, NY ,USA, 2023.
[55] E. Shi, F. Zhang, Y . Wang, B. Chen, L. Du, H. Zhang, S. Han,
D. Zhang, H. Sun, Sotana: The open-source software develop-
ment assistant, arXiv preprint arXiv:2308.13416 (2023).
[56] M. Bavarian, H. Jun, N. Tezak, J. Schulman, C. McLeavey,
J. Tworek, M. Chen, E fficient training of language models to
fill in the middle, 2022. URL: https://arxiv.org/abs/2207.
14255 .arXiv:2207.14255 .
15
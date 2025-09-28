# One Agent to Serve All: a Lite-Adaptive Stylized AI Assistant for Millions of Multi-Style Official Accounts

**Authors**: Xingyu Fan, Feifei Li, Wenhui Que, Hailong Li

**Published**: 2025-09-22 13:49:37

**PDF URL**: [http://arxiv.org/pdf/2509.17788v1](http://arxiv.org/pdf/2509.17788v1)

## Abstract
Conversational agents deployed in industrial-scale official account platforms
must generate responses that are both contextually grounded and stylistically
aligned-requirements that existing methods struggle to meet. Chain-of-thought
(CoT) prompting induces significant latency due to multi-turn reasoning;
per-account fine-tuning is computationally prohibitive; and long prompt-based
methods degrade the model's ability to grasp injected context and style. In
this paper, we propose WeStar, a lite-adaptive framework for stylized
contextual question answering that scales to millions of official accounts.
WeStar combines context-grounded generation via RAG with style-aware generation
using Parametric RAG (PRAG), where LoRA modules are dynamically activated per
style cluster. Our contributions are fourfold: (1) We introduce WeStar, a
unified framework capable of serving large volumes of official accounts with
minimal overhead. (2) We propose a multi-dimensional, cluster-based parameter
sharing scheme that enables compact style representation while preserving
stylistic diversity. (3) We develop a style-enhanced Direct Preference
Optimization (SeDPO) method to optimize each style cluster's parameters for
improved generation quality. (4) Experiments on a large-scale industrial
dataset validate the effectiveness and efficiency of WeStar, underscoring its
pracitical value in real-world deployment.

## Full Text


<!-- PDF content starts -->

One Agent to Serve All: a Lite-Adaptive Stylized AI Assistant for Millions of
Multi-Style Official Accounts
Xingyu Fan, Feifei Li, Wenhui Que*, Hailong Li
WeChat, Tencent Inc., Beijing, China
{fanxfan, niyali, victorque, aloneli}@tencent.com
Abstract
Conversational agents deployed in industrial-scale official ac-
count platforms must generate responses that are both con-
textually grounded and stylistically aligned—requirements
that existing methods struggle to meet. Chain-of-thought
(CoT) prompting induces significant latency due to multi-turn
reasoning; per-account fine-tuning is computationally pro-
hibitive; and long prompt-based methods degrade the model’s
ability to grasp injected context and style. In this paper, we
propose WeStar, a lite-adaptive framework for stylized con-
textual question answering that scales to millions of offi-
cial accounts. WeStar combines context-grounded generation
via RAG with style-aware generation using Parametric RAG
(PRAG), where LoRA modules are dynamically activated per
style cluster. Our contributions are fourfold: (1) We intro-
duce WeStar, a unified framework capable of serving large
volumes of official accounts with minimal overhead. (2) We
propose a multi-dimensional, cluster-based parameter sharing
scheme that enables compact style representation while pre-
serving stylistic diversity. (3) We develop a style-enhanced
Direct Preference Optimization (SeDPO) method to optimize
each style cluster’s parameters for improved generation qual-
ity. (4) Experiments on a large-scale industrial dataset vali-
date the effectiveness and efficiency of WeStar, underscoring
its pracitical value in real-world deployment.
Introduction
Inspired by the outstanding capabilities of large lan-
guage models in question-answering tasks, the industry has
adopted conversational agents that dialogue with users in
many tasks, such as AI in games (van Stegeren and Mys-
liwiec 2021), voice assistants (Arakawa, Lehman, and Goel
2024), and official account assistants. Our Industrial Offi-
cial Accounts serve as a powerful communication channel
for individuals, media outlets, enterprises, government bod-
ies, and other organizations, enabling them to disseminate
information in article form within our ecosystem. Users can
interact with these articles by leaving comments, which are
often replied to directly by the author. Additionally, users
may pose questions to the Official Account’s intelligent as-
sistant via the chat interface, expecting responses that are
both context-grounded and style-aware which is grounded
in the author’s published articles and reflective of the au-
thor’s personal communication style. While the articles pro-
vide rich, factual content, they are typically formal and donot capture the author’s conversational tone. In contrast, the
author’s replies to user comments offer a more authentic and
fine-grained reflection of their stylistic preferences in inter-
active settings. Motivated by this observation, we treat arti-
cles as the source of question-specific knowledge and lever-
age the author’s historical comment replies as the basis for
style-specific knowledge to address theStylized Contextual
Question Answeringproblem.
To tackle the problem, existing approaches can be broadly
categorized into three paradigms:fine-tuning-basedmeth-
ods, which directly adapt the model on style-specific data;
CoT-basedmethods, which employ multi-step prompting
to decompose and solve the task; andprompt-basedmeth-
ods, which inject both knowledge-specific and style-specific
knowledge into a single prompt. However, each of these
comes with limitations in scalability, efficiency, or effective-
ness when deployed at industrial scale.
In recent years, fine-tuning-based approaches have shown
strong performance in the domain of stylized text genera-
tion (Liu, Diddee, and Ippolito 2024). A widely adopted
strategy involves supervised fine-tuning (SFT) on cus-
tomized style-specific corpora, enabling large language
models (LLMs) to adjust their output distributions by up-
dating model parameters accordingly. However, fine-tuning
remains a significant bottleneck in many real-world appli-
cations. For instance, in our public account assistant tasks,
a distinct model must be fine-tuned and maintained for
each individual author to ensure stylistic consistency. This
process incurs substantial time and computational costs,
severely limiting scalability.
Chain-of-thought (CoT)-based methods offer a promising
direction for mitigating the computational and deployment
burden associated with stylized text generation. A straight-
forward solution is to decompose the stylized contextual
question answering task into two sequential sub-tasks: (1)
generating a contextually relevant answer, followed by (2)
applying a text style transfer model to adapt the response to
the target style. While conceptually simple, this two-stage
pipeline introduces practical limitations. Specifically, invok-
ing a large language model (LLM) twice not only increases
computational overhead but also introduces latency, which
can significantly degrade the user experience.
Therefore, prompt-based methods presents a more prac-
tical solution to the stylized contextual question answer-arXiv:2509.17788v1  [cs.CL]  22 Sep 2025

ing task. By incorporating retrieved question-specific arti-
cles and customized style-specific corpora into the input
of an end-to-end LLM, the system injects external stylis-
tic and semantic knowledge directly into the model’s con-
text window, enabling dynamic adaptation without param-
eter updates. However, injecting knowledge from multiple
sources via input prompts inevitably leads to increased con-
text length. This extended input not only introduces addi-
tional computational overhead and latency during inference,
but also degrades the LLM’s ability to effectively understand
and utilize the injected information—particularly in scenar-
ios requiring complex reasoning (Levy, Jacoby, and Gold-
berg 2024).
To address these challenges, we proposeWeStar, a novel
framework to build one lite-adaptive style-ware and context-
grounded agent capable of serving millions of multi-style
official accounts. Before online inference, WeStar first per-
forms fine-grained style labeling over each public account
author’s corpus across multiple stylistic dimensions. Au-
thors with similar style are then grouped into clusters, and
each cluster is associated with a shared set of stylized model
parameters trained via style-enhanced Direct Preference Op-
timization (SeDPO). This cluster-based parameter sharing
enables compact storage of stylistic knowledge and sup-
ports scalable deployment across millions of public account
authors. During inference, WeStar incorporates question-
specific knowledge (i.e., retrieved articles) into the input
prompt to enrich the model’s domain understanding and
enhance its question-answering capabilities. In parallel, in-
stead of relying exclusively on prompt-based knowledge in-
jection, WeStar adopts a parameter injection approach in-
spired by PRAG (Su et al. 2025), where style-specific
knowledge is embedded directly into the model parameters.
This dual-channel design not only enhances stylistic consis-
tency but also substantially reduces prompt length, thereby
mitigating context overflow and improving inference effi-
ciency.
To evaluate the effectiveness of WeStar in real-world in-
dustrial scenarios, we conducted experiments on a large-
scale public account dataset. WeStar achieves state-of-the-
art performance across four customized evaluation dimen-
sions, including contextual alignment, question relevance,
stylistic strength and fluency. These results demonstrate the
practical applicability of our framework in stylized contex-
tual generation tasks.
Our main contributions are summarized as follows:
• We propose WeStar, a novel framework to build one lite-
adaptive stylized contextual question-answering agent
capable of serving millions of official accounts.
• We introduce a multi-dimensional style-specific cluster-
based parameter sharing approach that enables compact
parameter storage while preserving a broad spectrum
of stylistic knowledge, facilitating scalable deployment
across diverse official account authors.
• We leverage a style-enhanced Direct Preference Opti-
mization (SeDPO) strategy to train the parameter repre-
sentations for each style cluster, thereby improving the
model’s capacity for style-aware generation.• We conduct evaluations on a large-scale industrial dataset
and validate the proposed method across four key met-
rics, demonstrating its practical value and profitability in
real-world applications.
Related Work
Text Style Transfer
Text style transfer (TST) is a different but related task com-
pared to our work, which aims to alter the stylistic attributes
of a given text while preserving its original content. Early
research focused on rule-based and statistical methods. With
the rise of deep learning, neural methods became dominant,
particularly unsupervised frameworks leveraging content-
style disentanglement or reinforcement learning (Liu et al.
2022; Luo et al. 2019). More recent work investigates the
potential of contrastive learning and pattern mining (Han
et al. 2023). LLMs also bring new paradigms to TST. Reif et
al.(Reif et al. 2022) and Mukherjee et al.(Mukherjee, Ojha,
and Dusek 2024) explored zero-shot and few-shot style
transfer via prompt engineering, revealing LLMs’ surpris-
ing generalization ability. Ostheimer et al. (Ostheimer et al.
2024) further investigated LLM-based evaluation, demon-
strating strong correlation with human judgments.
Stylized Answer Generation
Several studies tackle stylized answer generation using fine-
tuned large language models or style-controlled decoding
strategies (Zheng et al. 2021; Gao et al. 2019). To bet-
ter ground stylized generation in external knowledge, Sun
et al.(Sun et al. 2022) introduce disentangled template
rewriting in knowledge-grounded scenarios. Additionally,
Feature-Guided Knowledge Augmentation(Li et al. 2023)
retrieves stylistic sentences to guide content planning and
uses contrastive learning to enhance fluency and style con-
trol. Despite these advances, most prior work operates on
relatively small-scale or synthetic style corpora, which limits
their applicability in real-world, large-scale scenarios. Re-
cent efforts have begun to address scalability (Li et al. 2024;
Jing et al. 2024), but none match the scale and complexity
of the deployment setting considered in our work.
Method
Given a user-issued questionQ, a retrieved context-specific
knowledgeCrelevant toQ, and a retrieved style-specific
knowledgeSrepresenting a target response style, the objec-
tive is to generate a style-aware and context-grounded an-
swerAsuch that
•Acorrectly addresses the information need expressed in
Q, grounded in the content ofC, and
•Aadheres to the stylistic characteristics exemplified by
S.
To address the challenges of stylized contextual question
answering, WeStar adopts a dual-injection approach: it in-
jects question-specific knowledge into the prompt and style-
specific knowledge into the model parameters. In this sec-
tion, we first describe how WeStar clusters authors with
similar style and trains shared stylize-specific parameters

for each style cluster. We then present the online infer-
ence mechanism of WeStar, which dynamically composes
responses by combining the retrieved content and the appro-
priate style-specific parameters.
CQA Construction
To construct high-quality CQA (Context, Question, Answer)
triplets for training, we adopt two complementary strategies:
a forward-thinking method that prompts a large language
modelMto generate questions and answers based on given
article segments, and a bottom-up method that promptsMto
simulate realistic user roles and queries based on the domain
of each public account, followed by context retrieval and an-
swer generation withM. The forward-thinking approach of-
fers high scalability, while the bottom-up strategy introduces
domain-relevant and user-intent-aligned queries, ensuring a
better match to real-world QA workflows. Together, these
methods yield a CQA dataset that balances diversity, diffi-
culty, and contextual grounding. Full prompt templates and
examples are provided in theAppendix.
Style Labeling
We design twelve style classification standards, organized
across four stylistic dimensions:
•Semantic level:intention type, degree of authority
•Grammatical level:omitted features, use of inversion,
use of passive voice
•Syntactic level:sentence complexity, rhetorical features,
cohesion mechanisms
•Lexical level:lexical complexity, emotional polarity, fre-
quency of emojis, and degree of formality
For each official account author, we perform fine-grained
labeling of their style corpus using LLMM, based on the
twelve predefined classification standards. Specifically, for
every QA pair in the corpus,Mgenerates candidate labels
for each stylistic dimension. We then aggregate these anno-
tations by taking the majority label for each standard across
all QA pairs within the author’s corpus. These aggregated
labels serve as the author’s stylistic profile and provide the
foundation for subsequent style tree building. The detailed
prompt templates are provided in theAppendix.
Style Tree Building
Following the style labeling stage, we construct a hierar-
chical style clustering tree to group the authors with sim-
ilar style into the same cluster. The construction process,
outlined in Algorithm 1, proceeds by traversing the prede-
fined set of style labeling standardsSin a specified hier-
archical order. For each leaf node in the current tree, if the
associated style corpus can be partitioned by a label stan-
dards∈ S—and each resulting subset contains more than
kexamples—then the node is expanded into multiple child
nodes, each corresponding to a distinct stylistic subgroup
defined bys. Upon completion of the algorithm, each leaf
node in the resulting style clustering tree represents a styl-
ized cluster of the original corpusC. The path from a givenAlgorithm 1: style tree building
Input:C– a set of style corpora
Input:S– a set of style classification standards
Output:T– a hierarchical style tree
1Initialize treeTwith a root node containing
all corporaC
2Initialize queueQ← {root node}
3Set corpus size thresholdk
4 For eachstyle standards∈ S:
5 WhileQis not empty:
6n←Q.front()
7Q.pop()
8 Ifnodencan be split into child nodes{n 1, ..., n m}
bysandeach childn isatisfies|C ni|> k,then:
9Expand nodeninTinto children{n 1, ..., n m}
10Push all new leaf nodes inTinto queueQ
11ReturnT
leaf node to the root captures its cumulative stylistic charac-
teristics.
In this way, authors with similar stylistic characteristics
are grouped into the same clusters. This hierarchical orga-
nization facilitates style-specific parameter training by en-
abling parameter sharing among authors within the same
cluster, thus reducing both training costs and storage over-
head. Moreover, for authors with limited stylistic data, this
clustering mechanism allows them to benefit from the shared
parameters of stylistically similar authors, thereby enhanc-
ing generalization and mitigating the data scarcity issue.
CQSA Construction
Many recent studies have demonstrated the strong capabili-
ties of LLMs in text style transfer(TST) (Mukherjee, Ojha,
and Dusek 2024). Building on this foundation, we lever-
age LLMs to transform standard CQA instances into CQSA
(Context, Question, Stylized Answer)instances that align
with the target style for each cluster.
For each corpus within the style cluster derived from the
style tree, we prompt LLMMto rewrite the original an-
swer—while preserving the factual correctness of the re-
sponse—into a form that conforms to the style of the target
corpus. The prompt is composed of the following elements:
1. the input context and question from the original CQA
pair,
2. the original answer to be rewritten,
3. the full set of twelve style classification standards and
labels associated with the target cluster,
4.min-context examples selected from the same cluster,
each consisting of a user comment and the corresponding
author reply.
To ensure stylistic consistency, the in-context examples are
sampled uniformly at random from each author in the same
cluster. The detailed prompts are presented in theAppendix.
Data Selection
Inspired by metric-based RLHF (Wang et al. 2025), we ap-
ply metric-based constraints during data construction to bet-

Figure 1: Overview of WeStar.
ter align the model’s outputs with the target stylistic expec-
tations, while simultaneously reducing the likelihood of hal-
lucinations. Following previous works (Wang et al. 2025;
Ostheimer et al. 2024), we conduct automated evaluation
on each CQSA instance using LLMMacross four key
dimensions: Contextual Alignment (C–A), Question Rele-
vance (Q–A), Stylistic Strength (S–A) andFluency, which
is presented in theAppendix. We will further discuss these
metrics in SectionMetrics. We aggregate the scores across
these four metrics and select the top 10,000 CQSA instances
as high-quality samples for subsequent style-specific param-
eter training.
Style-Enhanced DPO
Prior to this stage, we have obtained high-quality CQSA in-
stances for each style cluster to support style-specific pa-
rameter training. In this step, we adopt style-enhanced DPO
(SeDPO) to train the parameters. Specifically, for a given
style cluster, the top 10,000 CQSA instances are used as
chosen samples. We construct the corresponding rejected
sample for each chosen sample by retrieving answers to the
same question with high stylistic similarity—such as sibling
nodes in the style tree—while ensuring that these answers
differ in a certain style label. This construction aligns with
the principle of controlled variable experimentation: when
the negative sample shares most contextual and semantic
features with the positive sample, the model is encouraged to
focus more on the fine-grained stylistic distinctions. Conse-
quently, this training setup facilitates more effective learning
of style-specific behaviors within each stylistic cluster.
We adopt the same parameterization and injection
paradigm as PRAG (Su et al. 2025), employing LoRA (Xu
et al. 2024) as our fine-tuning and parameter storage strat-
egy. This design enables each style cluster to be associated
with an independently trained set of low-rank adaptation pa-
rameters, which allows the model to encode style-specific
behaviors in a parameter-efficient manner, enabling scalable
and flexible deployment across a wide range of stylistic clus-
ters without the need to train or deploy the full base model.Online Inference
After training a set of style-specific parameters for each style
cluster, WeStar applies them during online inference to de-
liver style-aware, context-grounded responses at scale.
WeStar performs inference by jointly injecting question-
specific and style-specific knowledge into the generation
process. Specifically, question-specific article segments are
inserted into the input prompt, providing contextual ground-
ing. Simultaneously, style-specific LoRA parameters cor-
responding to the author’s style cluster are retrieved via a
PRAG (Parametric Retrieval-Augmented Generation) man-
ner and injected into the model’s parameter space. This dual-
injection strategy enables WeStar to generate responses that
are both contextually relevant and stylistically aligned, while
maintaining scalability across a large number of public ac-
count authors.
Experiments
Experimental Settings
DatasetTo the best of our knowledge, there exists no
public dataset that simultaneously provides articles, user
queries, and large-scale stylized replies from millions of au-
thors. Therefore, we directly evaluate our method using pro-
prietary data from a widely-used, real-world industrial offi-
cial accounts platform. We deployed the WeStar framework
in this environment and conducted evaluations across ten
representative style clusters, constructed using the method-
ology detailed in SectionStyle Tree Building. Each clus-
ter contains real user comments and the corresponding au-
thor replies collected before July 2025, serving as style ref-
erence for stylized question answering. The contextual re-
trieval corpus consists of all historical articles published
by the authors prior to the same date. To perform evalua-
tion, we constructed a test set of 2,000 instances using the
same methodology described in SectionCQA Construc-
tionand SectionCQSA Construction. To better simulate
real-world deployment scenarios, we further supplemented
the test set with 3,000 user-generated information-seeking
questions collected from live interactions. In total, the eval-

uation dataset comprises 5,000 queries, covering both con-
trolled and in-the-wild user intents.
MetricsTo ensure consistency and comparability, we em-
ploy the same four evaluation metrics as described in Sec-
tionData Selectionto assess model performance. These
metrics have been widely adopted in previous studies (Os-
theimer et al. 2024; Wang et al. 2025):
•Contextual Alignment (C–A): the degree of semantic
consistency between the generated answer and the re-
trieved context;
•Question Relevance (Q–A): the extent to which the an-
swer accurately addresses the core intent of the input
question;
•Stylistic Strength (S–A): the degree to which the answer
adheres to the target stylistic attributes;
•Fluency: the grammaticality and naturalness of the gen-
erated response.
To obtain consistent and scalable evaluations, we em-
ployed DeepSeek-R1 (DeepSeek-AI et al. 2025) to rate each
output along the four dimensions, using standardized evalu-
ation prompts. The exact prompt formulations are provided
in theAppendix.
BaselinesAs discussed in theIntroduction, stylized con-
textual question answering in industrial settings—such as
official account platforms—requires a fully end-to-end LLM
pipeline. This constraint renders multi-step or sequential
prompting approaches infeasible due to latency and system
complexity. To the best of our knowledge, there exists no
prior method capable of supporting scalable stylized contex-
tual question answering across millions of official account
authors with distinct stylistic preferences.
To evaluate the effectiveness of WeStar under this chal-
lenging setting, we compare it against five baseline meth-
ods, covering diverse methodological paradigms: (1) two
prompting-based approaches, (2) two SFT variants from
WeStar, and (3) one variant of DPO:
•R1-Prompt. We adopt DeepSeek-R1 (DeepSeek-AI
et al. 2025), a recently released open-source large lan-
guage model that has demonstrated strong performance
across a wide range of reasoning, instruction-following,
and language understanding benchmarks, making it one
of the most representative open-source LLMs available
to date. In this prompt-based baseline, we use DeepSeek-
R1 as the base model and construct the input prompt by
injecting four key elements: (1) the user’s question, (2)
a set of retrieved articles from the corresponding official
account, (3) recent high-quality author replies that reflect
the account’s stylistic preferences, and (4) a system-level
instruction aligned with the intelligent assistant scenario.
The model then generates an answer conditioned on both
contextual knowledge and stylistic cues. We consider
R1-Promp to be a strong representative of theprompt-
based paradigm, as it leverages a state-of-the-art base
model and simulates realistic deployment constraints us-
ing only prompt engineering without any task-specific
fine-tuning.•SFT-Prompt. While DeepSeek-R1 exhibits strong per-
formance, its large parameter size incurs substantial in-
ference latency, making it less practical for real-time
applications. To address this issue, we construct a sec-
ond baseline, SFT-Prompt, based on the Qwen3-32B
model (Yang et al. 2025), which offers a better trade-off
between performance and latency in industrial deploy-
ment scenarios. Specifically, we first fine-tune Qwen3-
32B using 10,000 CQA instances to enhance its ability
to perform contextual reasoning (Liu et al. 2024). After
fine-tuning, we adopt the same prompt injection strategy
as in R1-Prompt, incorporating the user query, retrieved
articles, representative stylistic replies, and system-level
task instructions. This setting represents theSFT-then-
prompting paradigmunder similar model scale con-
straints and serves as a strong baseline for evaluating
methods that enhance LLMs via lightweight supervised
adaptation followed by prompting.
•LoRA-SFT. This baseline adopts the same online-
inference paradigm as proposed in WeStar. For each style
corpus, we apply supervised fine-tuning using LoRA to
train a set of style-specific parameters. At inference time,
these parameters are dynamically injected into the base
model following the PRAG parameter loading mecha-
nism. The training set consists of 10,000 randomly se-
lected CQSA instances sampled prior to the data selec-
tion step described in SectionData Selection.
•LoRA-SFT-S. This variant builds on LoRA-SFT, with
the only difference being the training data quality. Instead
of random sampling, we train the stylized parameters us-
ing the top 10,000 high-quality CQSA instances selected
through the metric-based data selection process intro-
duced in SectionData Selection. This setting enables us
to assess the impact of curated, high-quality training data
on the effectiveness of style-specific parameter learning.
•WeStar MDPO . This baseline shares the same online in-
ference setup and training paradigm as WeStar, but dif-
fers in the construction of rejected samples used for DPO.
Instead of using style-aware, high-quality CQSA data
selected via the procedures outlined in SectionsCQSA
ConstructionandData Selection, the rejected responses
are distilled from the base LLM by prompting it with cor-
responding(C, Q)pairs, without further quality filtering.
To enhance learning signals, we apply a metric-guided
DPO strategy inspired by MDPO (Wang et al. 2025),
where we select rejected samples that exhibit the great-
est deviation from the chosen instances across the four
evaluation metrics.
Implementation DetailsWe adopt Qwen3-32B as the
base language model. Additionally DeepSeek-R1 is chosen
as the the auxiliary LLMMthroughout our framework. Dur-
ing theCQA Constructionphase, we promptMto generate
3representative user roles for each official account domain,
and subsequently generate3domain-relevant questions per
role to construct a diverse question set. In theStyle Tree
Buildingstage, we set the corpus size thresholdk=100to
ensure sufficient stylistic representation in each node before
further partitioning. ForStyle-enhanced DPO, we employ

Figure 2: Results of WeStar vs. Prompt-Based Methods
LoRA with a rank of 16 and train for one epoch, enabling
efficient parameter adaptation for each style corpus while
maintaining training scalability.
Main Results
WeStar vs. Prompt-Based MethodsFigure 2 illustrates
the comparative performance between WeStar and prompt-
based methods across all four evaluation metrics mentioned
above. Specifically, the top-left, top-right, bottom-left, and
bottom-right subfigures respectively show the per-cluster
comparison results between WeStar, R1-Prompt and SFT-
Prompt along each metric. The x-axis represents the identi-
fiers of different style corpus clusters, while the y-axis cor-
responds to the metric scores achieved by each method on
those clusters. More detailed numerical results are presented
in Table 1. The first column lists the style corpus clusters and
the associated evaluation metrics. Columns 2, 3, and 7 re-
port the performance of R1-Prompt, SFT-Prompt, and WeS-
tar, respectively.
From the results, we observe that WeStar consistently out-
performs both prompting-based approaches on Q–A, C–A ,
and Fluency metrics on average. This improvement stems
from the limitations of prompt-based methods when dealing
with long input sequences. These approaches inject both the
retrieved articles and the full style corpus into the prompt,
leading to significant context length expansion, which in
turn challenges the LLM’s attention window. As a result,
the model’s ability to accurately comprehend and utilize the
injected information is weakened. For the Stylistic Strength
(S–A) metric, only R1-Prompt achieves performance com-
parable to WeStar on average. We attribute this to two main
reasons: (1) Prompt-based methods can flexibly incorpo-
rate the original author’s entire style corpus directly into
the prompt, whereas WeStar utilizes corpora from style-
similar authors identified via the style tree; (2) The base
model used in R1-Prompt, DeepSeek-R1, contains signifi-
cantly more parameters than our base model, Qwen3-32B,
which may contribute to its stronger stylistic generation per-
formance. Consequently, WeStar matches but does not sur-
pass R1-Prompt on the S–A dimension.
Figure 3: Results of WeStar vs. Variants of WeStar
WeStar vs. Variants of WeStarFigure 3 presents a com-
parative analysis of WeStar and its three training variants:
LoRA-SFT, LoRA-SFT-S, and WeStar MDPO —across all
four evaluation metrics. The detailed results can be found
in Table 1, where the 4th to 7th columns correspond to
the respective performances of LoRA-SFT, LoRA-SFT-S,
WeStar MDPO , and WeStar.
Among these, LoRA-SFT underperforms the other three
methods across all metrics on average. This is primarily due
to the lack of metric-based data filtering in its training phase.
In contrast, the remaining three approaches benefit from the
metric-based data selection pipeline, which ensures higher-
quality CQSA instances and leads to more accurate style-
aware and context-grounded responses.
Overall, WeStar achieves the best average performance,
outperforming both LoRA-SFT-S and WeStar MDPO , except
on the Q–A metric, where WeStar MDPO leads by a small
margin of 0.01. Across all four metrics, the performance
differences among these three approaches remain relatively
small (average gap is smaller than 0.06), indicating that data
quality plays a major role, while training objectives provide
more fine-grained benefits.
Notably, WeStar achieves the highest score on the S–A
(Stylistic Strength) metric, validating the effectiveness of
using style-specific rejected samples during DPO training.
In contrast, WeStar MDPO shows slightly inferior perfor-
mance on S–A. A likely explanation is that its rejected sam-
ples differ significantly from the chosen ones across all four
dimensions, allowing the model to easily distinguish them
and optimize logits without focusing specifically on stylis-
tic nuances. This diluted objective may have weakened the
model’s ability to capture fine-grained stylistic preferences.
Time Cost Analysis
Among our baselines, SFT-prompt employs a smaller back-
bone model compared to R1-prompt, leading to shorter in-
ference time. Therefore, we focus our runtime comparison
on SFT-prompt and WeStar. We measure the average infer-
ence time per sample on the test set: WeStar takes 2.08 sec-
onds, while SFT-prompt takes 2.47 seconds, resulting in a
1.19x speedup. This improvement stems from the fact that
in SFT-prompt, injecting style-related tokens significantly

Table 1: Main Results
Prompt-Based Methods SFT-Based Methods DPO-Based Methods
Dataset & Metrics R1-Prompt SFT-Prompt LoRA-SFT LoRA-SFT-S WeStar MDPO WeStar
cluster 0 Q-A 4.49 4.20 4.34 4.44 4.50 4.56
cluster 0 C-A 4.53 4.28 4.37 4.51 4.62 4.63
cluster 0 S-A 4.61 4.23 4.40 4.73 4.694.74
cluster 0 Fluency 4.87 4.78 4.91 4.91 4.964.92
cluster 1 Q-A 4.48 4.28 4.42 4.44 4.584.54
cluster 1 C-A 4.53 4.28 4.48 4.50 4.704.66
cluster 1 S-A 4.69 3.23 4.684.78 4.76 4.77
cluster 1 Fluency 4.87 4.65 4.924.90 4.90 4.89
cluster 2 Q-A 4.27 4.21 4.15 4.33 4.374.32
cluster 2 C-A 4.42 4.24 4.29 4.41 4.484.46
cluster 2 S-A 3.914.21 4.02 4.05 4.11 4.20
cluster 2 Fluency 4.634.82 4.77 4.71 4.75 4.78
cluster 3 Q-A 4.31 4.29 4.424.38 4.40 4.40
cluster 3 C-A 4.43 4.32 4.51 4.46 4.494.57
cluster 3 S-A 3.853.80 3.62 3.75 3.72 3.79
cluster 3 Fluency 4.64 4.60 4.58 4.59 4.564.67
Due to the space limit, we put the detailed results of other six clusters in theAppendix.
average Q-A 4.38 4.26 4.35 4.41 4.444.43
average C-A 4.45 4.30 4.43 4.49 4.52 4.55
average S-A 4.253.73 3.92 4.22 4.204.25
average Fluency 4.75 4.70 4.734.77 4.76 4.77
Boldindicates the best result; Underline indicates the second-best.
Figure 4: Cases of WeStar vs. Prompt-Based Methods
increases the input length, which imposes substantial over-
head during decoding. In contrast, WeStar injects stylistic
knowledge via LoRA modules, which are lightweight and
efficiently loaded. This comparison highlights the efficiency
and scalability of parameterized style injection over prompt-
based alternatives in latency-sensitive applications.
Case Study
We present a case study in Figure 4, comparing responses
generated by WeStar and two prompt-based methods for the
same input question. The question is designed to reference
three article snippets from an Official Account. To make the
comparison intuitive and reader-friendly, we choose a stylis-
tic exemplar from Chinese literature—the fictional character
Huang Rong (wikipedia 2025) from Jin Yong’s novels—as
the target persona. Due to space limits, we omit the ref-
erenced article snippets and only display the opening por-
tion of each generated response. Phrases, words, or sentencefragments that align with Huang’s style are underlined. As
shown, WeStar produces responses that more consistently
reflect the target style. Notably, the last sentence generated
by WeStar showed in Figure 4 omits the subject, which mir-
rors a grammatical-level stylistic trait of Huang’s speech pat-
terns. In contrast, both prompt-based methods underperform
in stylistic consistency. This is likely due to the injection of
lengthy article segments into the prompt, which increases
the context length and challenges the model’s attention win-
dow. This case highlights the effectiveness of WeStar’s pa-
rameterized style representation: by encoding style-specific
knowledge directly into the model’s parameter space, WeS-
tar avoids the limitations imposed by prompt size and better
preserves target style during generation.
Conclusion
In this work, we tackle the underexplored yet practical
task of stylized contextual question answering for official
accounts, where responses must be both style-aware and
context-grounded at scale. Existing fine-tuning, CoT, and
prompt-based methods struggle with efficiency or scalabil-
ity in industrial settings. We propose WeStar, a lite-adaptive
AI assistant designed for millions of multi-style Official Ac-
counts. WeStar leverages RAG for retrieving context and
PRAG for injecting style-specific LoRA modules. It builds
high-quality training data through hierarchical style clus-
tering, LLM-based stylized rewriting, and metric-driven se-
lection, and further enhances stylistic alignment via Style-
enhanced DPO. Experiments on large-scale industrial data
show that WeStar outperforms strong baselines in contextual
relevance, style fidelity, and fluency—offering an effective,
scalable solution for real-world deployments.

References
Arakawa, R.; Lehman, J. F.; and Goel, M. 2024. PrISM-
Q&A: Step-Aware V oice Assistant on a Smartwatch En-
abled by Multimodal Procedure Tracking and Large Lan-
guage Models.Proc. ACM Interact. Mob. Wearable Ubiqui-
tous Technol., 8(4): 180:1–180:26.
DeepSeek-AI; Guo, D.; Yang, D.; Zhang, H.; Song, J.;
Zhang, R.; Xu, R.; Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; Zhang,
X.; Yu, X.; Wu, Y .; Wu, Z. F.; Gou, Z.; Shao, Z.; Li, Z.; Gao,
Z.; Liu, A.; Xue, B.; Wang, B.; Wu, B.; Feng, B.; Lu, C.;
Zhao, C.; Deng, C.; Zhang, C.; Ruan, C.; Dai, D.; Chen, D.;
Ji, D.; Li, E.; Lin, F.; Dai, F.; Luo, F.; Hao, G.; Chen, G.; Li,
G.; Zhang, H.; Bao, H.; Xu, H.; Wang, H.; Ding, H.; Xin,
H.; Gao, H.; Qu, H.; Li, H.; Guo, J.; Li, J.; Wang, J.; Chen,
J.; Yuan, J.; Qiu, J.; Li, J.; Cai, J. L.; Ni, J.; Liang, J.; Chen,
J.; Dong, K.; Hu, K.; Gao, K.; Guan, K.; Huang, K.; Yu, K.;
Wang, L.; Zhang, L.; Zhao, L.; Wang, L.; Zhang, L.; Xu,
L.; Xia, L.; Zhang, M.; Zhang, M.; Tang, M.; Li, M.; Wang,
M.; Li, M.; Tian, N.; Huang, P.; Zhang, P.; Wang, Q.; Chen,
Q.; Du, Q.; Ge, R.; Zhang, R.; Pan, R.; Wang, R.; Chen,
R. J.; Jin, R. L.; Chen, R.; Lu, S.; Zhou, S.; Chen, S.; Ye,
S.; Wang, S.; Yu, S.; Zhou, S.; Pan, S.; and Li, S. S. 2025.
DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
via Reinforcement Learning.CoRR, abs/2501.12948.
Gao, X.; Zhang, Y .; Lee, S.; Galley, M.; Brockett, C.; Gao,
J.; and Dolan, B. 2019. Structuring Latent Spaces for Styl-
ized Response Generation. In Inui, K.; Jiang, J.; Ng, V .; and
Wan, X., eds.,Proceedings of the 2019 Conference on Em-
pirical Methods in Natural Language Processing and the 9th
International Joint Conference on Natural Language Pro-
cessing, EMNLP-IJCNLP 2019, Hong Kong, China, Novem-
ber 3-7, 2019, 1814–1823. Association for Computational
Linguistics.
Han, J.; Wang, Q.; Zhang, L.; Chen, W.; Song, Y .; and
Mao, Z. 2023. Text Style Transfer with Contrastive Trans-
fer Pattern Mining. In Rogers, A.; Boyd-Graber, J. L.; and
Okazaki, N., eds.,Proceedings of the 61st Annual Meeting
of the Association for Computational Linguistics (Volume 1:
Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023,
7914–7927. Association for Computational Linguistics.
Jing, L.; Song, X.; Lin, X.; Zhao, Z.; Zhou, W.; and Nie, L.
2024. Stylized Data-to-text Generation: A Case Study in the
E-Commerce Domain.ACM Trans. Inf. Syst., 42(1): 25:1–
25:24.
Levy, M.; Jacoby, A.; and Goldberg, Y . 2024. Same Task,
More Tokens: the Impact of Input Length on the Reasoning
Performance of Large Language Models. In Ku, L.; Martins,
A.; and Srikumar, V ., eds.,Proceedings of the 62nd Annual
Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), ACL 2024, Bangkok, Thailand,
August 11-16, 2024, 15339–15353. Association for Compu-
tational Linguistics.
Li, J.; Zhang, Z.; Chen, X.; Zhao, D.; and Yan, R. 2023.
Stylized Dialogue Generation with Feature-Guided Knowl-
edge Augmentation. In Bouamor, H.; Pino, J.; and Bali, K.,
eds.,Findings of the Association for Computational Linguis-
tics: EMNLP 2023, Singapore, December 6-10, 2023, 7144–
7157. Association for Computational Linguistics.Li, J.; Zhang, Z.; Tu, Q.; Cheng, X.; Zhao, D.; and Yan,
R. 2024. StyleChat: Learning Recitation-Augmented Mem-
ory in LLMs for Stylized Dialogue Generation.CoRR,
abs/2403.11439.
Liu, R.; Gao, C.; Jia, C.; Xu, G.; and V osoughi, S. 2022.
Non-Parallel Text Style Transfer with Self-Parallel Super-
vision. InThe Tenth International Conference on Learn-
ing Representations, ICLR 2022, Virtual Event, April 25-29,
2022. OpenReview.net.
Liu, X.; Diddee, H.; and Ippolito, D. 2024. Customizing
Large Language Model Generation Style using Parameter-
Efficient Finetuning. In Mahamood, S.; Nguyen, M. L.; and
Ippolito, D., eds.,Proceedings of the 17th International Nat-
ural Language Generation Conference, INLG 2024, Tokyo,
Japan, September 23 - 27, 2024, 412–426. Association for
Computational Linguistics.
Liu, Z.; Ping, W.; Roy, R.; Xu, P.; Lee, C.; Shoeybi, M.;
and Catanzaro, B. 2024. ChatQA: Surpassing GPT-4 on
Conversational QA and RAG. In Globersons, A.; Mackey,
L.; Belgrave, D.; Fan, A.; Paquet, U.; Tomczak, J. M.; and
Zhang, C., eds.,Advances in Neural Information Process-
ing Systems 38: Annual Conference on Neural Information
Processing Systems 2024, NeurIPS 2024, Vancouver, BC,
Canada, December 10 - 15, 2024.
Luo, F.; Li, P.; Zhou, J.; Yang, P.; Chang, B.; Sun, X.; and
Sui, Z. 2019. A Dual Reinforcement Learning Framework
for Unsupervised Text Style Transfer. In Kraus, S., ed.,Pro-
ceedings of the Twenty-Eighth International Joint Confer-
ence on Artificial Intelligence, IJCAI 2019, Macao, China,
August 10-16, 2019, 5116–5122. ijcai.org.
Mukherjee, S.; Ojha, A. K.; and Dusek, O. 2024. Are Large
Language Models Actually Good at Text Style Transfer? In
Mahamood, S.; Nguyen, M. L.; and Ippolito, D., eds.,Pro-
ceedings of the 17th International Natural Language Gener-
ation Conference, INLG 2024, Tokyo, Japan, September 23 -
27, 2024, 523–539. Association for Computational Linguis-
tics.
Ostheimer, P. S.; Nagda, M. K.; Kloft, M.; and Fellenz, S.
2024. Text Style Transfer Evaluation Using Large Lan-
guage Models. In Calzolari, N.; Kan, M.; Hoste, V .; Lenci,
A.; Sakti, S.; and Xue, N., eds.,Proceedings of the 2024
Joint International Conference on Computational Linguis-
tics, Language Resources and Evaluation, LREC/COLING
2024, 20-25 May, 2024, Torino, Italy, 15802–15822. ELRA
and ICCL.
Reif, E.; Ippolito, D.; Yuan, A.; Coenen, A.; Callison-Burch,
C.; and Wei, J. 2022. A Recipe for Arbitrary Text Style
Transfer with Large Language Models. In Muresan, S.;
Nakov, P.; and Villavicencio, A., eds.,Proceedings of the
60th Annual Meeting of the Association for Computational
Linguistics (Volume 2: Short Papers), ACL 2022, Dublin,
Ireland, May 22-27, 2022, 837–848. Association for Com-
putational Linguistics.
Su, W.; Tang, Y .; Ai, Q.; Yan, J.; Wang, C.; Wang, H.; Ye, Z.;
Zhou, Y .; and Liu, Y . 2025. Parametric Retrieval Augmented
Generation. In Ferro, N.; Maistro, M.; Pasi, G.; Alonso,
O.; Trotman, A.; and Verberne, S., eds.,Proceedings of the

48th International ACM SIGIR Conference on Research and
Development in Information Retrieval, SIGIR 2025, Padua,
Italy, July 13-18, 2025, 1240–1250. ACM.
Sun, Q.; Xu, C.; Hu, H.; Wang, Y .; Miao, J.; Geng, X.;
Chen, Y .; Xu, F.; and Jiang, D. 2022. Stylized Knowledge-
Grounded Dialogue Generation via Disentangled Template
Rewriting. In Carpuat, M.; de Marneffe, M.; and Ru ´ız,
I. V . M., eds.,Proceedings of the 2022 Conference of the
North American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies, NAACL
2022, Seattle, WA, United States, July 10-15, 2022, 3304–
3318. Association for Computational Linguistics.
van Stegeren, J.; and Mysliwiec, J. 2021. Fine-tuning GPT-2
on annotated RPG quests for NPC dialogue generation. In
Fowler, A.; Pirker, J.; Canossa, A.; Arya, A.; and Harteveld,
C., eds.,FDG’21: The 16th International Conference on the
Foundations of Digital Games 2021, Montreal, QC, Canada,
August 3-6, 2021, 2:1–2:8. ACM.
Wang, Y .; Tian, B.; Su, Y .; Fan, Y .; and Guo, J. 2025.
MDPO: Customized Direct Preference Optimization with a
Metric-based Sampler for Question and Answer Generation.
In Rambow, O.; Wanner, L.; Apidianaki, M.; Al-Khalifa, H.;
Eugenio, B. D.; and Schockaert, S., eds.,Proceedings of the
31st International Conference on Computational Linguis-
tics, COLING 2025, Abu Dhabi, UAE, January 19-24, 2025,
10660–10671. Association for Computational Linguistics.
wikipedia. 2025. Huang Rong.
Xu, Y .; Xie, L.; Gu, X.; Chen, X.; Chang, H.; Zhang,
H.; Chen, Z.; Zhang, X.; and Tian, Q. 2024. QA-LoRA:
Quantization-Aware Low-Rank Adaptation of Large Lan-
guage Models. InThe Twelfth International Conference on
Learning Representations, ICLR 2024, Vienna, Austria, May
7-11, 2024. OpenReview.net.
Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.;
Yu, B.; Gao, C.; Huang, C.; Lv, C.; Zheng, C.; Liu, D.; Zhou,
F.; Huang, F.; Hu, F.; Ge, H.; Wei, H.; Lin, H.; Tang, J.;
Yang, J.; Tu, J.; Zhang, J.; Yang, J.; Yang, J.; Zhou, J.; Zhou,
J.; Lin, J.; Dang, K.; Bao, K.; Yang, K.; Yu, L.; Deng, L.;
Li, M.; Xue, M.; Li, M.; Zhang, P.; Wang, P.; Zhu, Q.; Men,
R.; Gao, R.; Liu, S.; Luo, S.; Li, T.; Tang, T.; Yin, W.; Ren,
X.; Wang, X.; Zhang, X.; Ren, X.; Fan, Y .; Su, Y .; Zhang,
Y .; Zhang, Y .; Wan, Y .; Liu, Y .; Wang, Z.; Cui, Z.; Zhang,
Z.; Zhou, Z.; and Qiu, Z. 2025. Qwen3 Technical Report.
CoRR, abs/2505.09388.
Zheng, Y .; Chen, Z.; Zhang, R.; Huang, S.; Mao, X.; and
Huang, M. 2021. Stylized Dialogue Response Generation
Using Stylized Unpaired Texts. InThirty-Fifth AAAI Con-
ference on Artificial Intelligence, AAAI 2021, Thirty-Third
Conference on Innovative Applications of Artificial Intel-
ligence, IAAI 2021, The Eleventh Symposium on Educa-
tional Advances in Artificial Intelligence, EAAI 2021, Vir-
tual Event, February 2-9, 2021, 14558–14567. AAAI Press.
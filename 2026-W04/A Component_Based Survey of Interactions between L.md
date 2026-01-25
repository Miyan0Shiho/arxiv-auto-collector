# A Component-Based Survey of Interactions between Large Language Models and Multi-Armed Bandits

**Authors**: Miao Xie, Siguang Chen, Chunli Lv

**Published**: 2026-01-19 10:53:57

**PDF URL**: [https://arxiv.org/pdf/2601.12945v2](https://arxiv.org/pdf/2601.12945v2)

## Abstract
Large language models (LLMs) have become powerful and widely used systems for language understanding and generation, while multi-armed bandit (MAB) algorithms provide a principled framework for adaptive decision-making under uncertainty. This survey explores the potential at the intersection of these two fields. As we know, it is the first survey to systematically review the bidirectional interaction between large language models and multi-armed bandits at the component level. We highlight the bidirectional benefits: MAB algorithms address critical LLM challenges, spanning from pre-training to retrieval-augmented generation (RAG) and personalization. Conversely, LLMs enhance MAB systems by redefining core components such as arm definition and environment modeling, thereby improving decision-making in sequential tasks. We analyze existing LLM-enhanced bandit systems and bandit-enhanced LLM systems, providing insights into their design, methodologies, and performance. Key challenges and representative findings are identified to help guide future research. An accompanying GitHub repository that indexes relevant literature is available at https://github.com/bucky1119/Awesome-LLM-Bandit-Interaction.

## Full Text


<!-- PDF content starts -->

A COMPONENT-BASEDSURVEY OFINTERACTIONS BETWEEN
LARGELANGUAGEMODELS ANDMULTI-ARMEDBANDITS
Siguang Chen1,Chunli Lv1,2, andMiao Xie*1,2
1College of Information and Electrical Engineering, China Agricultural University, Beijing 100083, China
2Key Laboratory of Agricultural Machinery Monitoring and Big Data Application, Ministry of Agriculture and Rural
Affairs, Beijing 100083, China
*Corresponding author:xiemiao@cau.edu.cn
Abstract
Large language models (LLMs) have become powerful and widely used systems for language understanding
and generation, while multi-armed bandit (MAB) algorithms provide a principled framework for adaptive
decision-making under uncertainty. This survey explores the potential at the intersection of these two fields.
As we know, it is the first survey to systematically review the bidirectional interaction between large language
models and multi-armed bandits at the component level. We highlight the bidirectional benefits: MAB algorithms
address critical LLM challenges, spanning from pre-training to retrieval-augmented generation (RAG) and
personalization. Conversely, LLMs enhance MAB systems by redefining core components such as arm definition
and environment modeling, thereby improving decision-making in sequential tasks. We analyze existing LLM-
enhanced bandit systems and bandit-enhanced LLM systems, providing insights into their design, methodologies,
and performance. Key challenges and representative findings are identified to help guide future research.
An accompanying GitHub repository that indexes relevant literature is available at https://github.com/
bucky1119/Awesome-LLM-Bandit-Interaction.
KeywordsSystematic Review, Large Language Models, Multi-Armed Bandits, Reinforcement Learning, Sequential
Decision-Making
1 Introduction
Large language models (LLMs) have recently emerged as powerful general-purpose systems for language understanding,
reasoning, and generation. Built upon transformer architectures and enabled by large-scale pretraining and post-training
techniques, modern LLMs are no longer standalone natural language processing (NLP) tools but have evolved into
modular components that can be invoked, combined, and scheduled within larger AI systems. This paradigm shift
has led to their widespread adoption in applications such as code generation [ 1], intelligent assistants [ 2], scientific
discovery [3], and decision support [4], fundamentally reshaping the design of contemporary intelligent systems [5].
Reinforcement learning (RL) studies how agents learn to make sequential decisions through interaction with an
environment. Among its many formulations,multi-armed bandits(MABs) represent a classical and efficient setting that
focuses on online decision-making under uncertainty through theexploration(trying new actions) andexploitation
(choosing known rewarding actions) trade-off. See [ 6] for an introduction to various RL methods and [ 7] for a
more in-depth look into bandit algorithms. Compared to full Markov decision processes (MDPs), bandit models rely
on immediate feedback rather than long-horizon state transitions, leading to simpler algorithmic structures, lower
computational overhead, and stronger system controllability. Due to these properties, bandit frameworks have been
widely adopted in practical systems such as recommendation [ 8], online advertising [ 9], adaptive experimentation [ 10,
11].
Recent advances in LLMs have revealed a strong structural complementarity between LLMs and bandit frameworks.
On the one hand, LLMs can enhance bandit algorithms by providing rich contextual representations [ 12], prior
knowledge [ 13], and reward predictions [ 14]. On the other hand, bandit methods offer LLM-based systems a principledarXiv:2601.12945v2  [cs.CL]  21 Jan 2026

mechanism for adaptive online decision-making [ 15], enabling tasks such as model selection [ 16], tool invocation [ 17],
generation control [ 18], and inference strategy selection [ 19]. This synergy has driven increasing interest in combining
LLMs and bandits across a wide range of interactive and decision-critical applications.
Despite this growing body of cross-domain research, a systematic and modular understanding of the LLM–bandit
intersection remains lacking. Prior surveys, such as Mui et al. [ 20] on adaptive learning and Bouneffouf et al. [ 11] on
bandit applications, primarily focus on general-purpose domains like recommendation or ad placement, and do not
consider the emerging structure of LLM systems. Likewise, surveys on reinforcement learning in NLP [ 21,22] and
sequence models [ 4] do not focus on the specific intersection between bandit algorithms and large language models.
Concurrent with our work, Bouneffouf et al. [ 23] also surveyed the interplay between bandit algorithms and large
language models. While their work provides a conceptual overview of mutual enhancement between the two paradigms,
our review focuses on a system-level component perspective that systematically maps interactions between LLM
modules and bandit algorithmic components.
In this survey, we adopt acomponent-based perspectiveon the interactions between large language models and
multi-armed bandits. We organize existing work into two complementary categories:bandit-enhanced LLM systems,
where bandit algorithms improve the online decision-making and control of LLM pipelines, andLLM-enhanced bandit
frameworks, where LLMs augment key components of traditional bandits. This perspective enables a unified technical
mapping of existing methods, clarifies common design patterns, and highlights open challenges at the intersection of
large language models and multi-armed bandits.
Our main contributions are summarized as follows.
•Systematic Component-Level Survey of the Bandit–LLM Intersection:To the best of our knowledge,
this survey represents the first systematic review that analyzes the interaction between multi-armed bandit
algorithms and large language models from a component-based and algorithmic perspective.
•Unified Component-Based Taxonomic Framework:We introduce a structured component-based taxonomy
for both LLM systems and bandit systems, enabling a unified technical lens to examine how bandit mechanisms
enhance LLM pipelines and how LLMs reshape core bandit components.
•Identification of Challenges and Research Opportunities:This survey critically examines existing cross-
domain methods, synthesizes key technical and conceptual challenges, and outlines promising research
directions that can guide future work at the intersection of bandit algorithms and large language models.
The organization of this article is as follows. Section 2 introduces the background information about large language
models and multi-armed bandits. Section 3 describes the survey methodology adopted in this work. A component-based
taxonomic framework for both LLM systems and bandit systems is presented in Section 4. We then review representative
studies on the bidirectional integration between LLMs and bandit algorithms in Sections 5 and 6. Finally, we discuss
challenges and future opportunities in Section 7 and conclude in Section 8.
2 Background
In this section, we present the relevant preliminary background to facilitate readers’ comprehension of large language
models (LLMs) and bandit algorithms. For in-depth information, we recommend consulting the original works.
2.1 Large Language Models
Large language models (LLMs) are a class of neural language models designed to learn the statistical structure of
natural language at scale. At their core, LLMs model the probability distribution of token sequences, enabling them
to predict and generate text conditioned on a given context. By learning from large corpora of human-written text,
LLMs acquire rich representations of linguistic patterns, factual knowledge, and semantic relationships. Comprehensive
introductions to language modeling can be found in existing surveys such as [24, 25, 5].
Modern LLMs are predominantly built upon the transformer architecture [ 26], which employs self-attention mechanisms
to capture long-range dependencies in text. A common training paradigm consists of large-scale self-supervised pre-
training on unlabeled text data, followed by task-specific fine-tuning or alignment procedures to adapt the model to
downstream applications [ 5]. Depending on architectural design, pretrained language models can be categorized into
encoder-only [ 27], decoder-only [ 28], and encoder–decoder models [ 29], each suited to different types of language
understanding and generation tasks.
LLMs can be viewed as scaled-up extensions of earlier pretrained language models, characterized by substantially
increased model capacity and training data volume. Representative examples include GPT-3 [ 30], PaLM [ 31], and
2

LLaMA [ 32]. Empirical studies on neural scaling laws suggest that increasing model size, data scale, and computational
budget can lead to consistent performance improvements across a wide range of tasks [ 33,34]. Moreover, LLMs have
been observed to exhibit emergent capabilities—such as in-context learning, instruction following, and multi-step
reasoning—that are not explicitly programmed but arise as a result of large-scale training [35].
Owing to these capabilities, LLMs have demonstrated strong performance not only in traditional natural language
processing tasks, but also in a growing set of applications including information retrieval [ 36], code generation [ 1],
science discovery [ 3], personalized digital assistants [ 2], education [ 37], finance [ 38], and healthcare [ 39]. Their ability
to reason over unstructured text, integrate external knowledge, and interact with tools has further motivated research into
system-level integration strategies, where LLMs are deployed as components within larger decision-making pipelines.
In this survey, rather than focusing on model architectures or training techniques, we adopt a system-level perspective
that decomposes LLM-based systems into functional components. This abstraction enables a clearer analysis of how
decision-making mechanisms—such as bandit algorithms—can be integrated with LLMs to improve learning efficiency,
adaptability, and robustness. Detailed component definitions and taxonomy are presented in Section 4.1.
2.2 Multi-Armed Bandit
A multi-armed bandit (MAB) problem models a sequential decision-making setting in which an agent repeatedly selects
an action (arm) from a predefined set and observes stochastic feedback. See the literature [ 40,7] for a detailed paradigm
of bandit algorithms and a more comprehensive overview of various algorithms. At each interaction, the agent balances
exploration of uncertain actions and exploitation of actions with high expected reward, with the goal of optimizing
long-term performance, typically measured by cumulative reward or regret.
Research in MAB has evolved fromstochastic bandits, where reward distributions are fixed but unknown, tocontextual
bandits, where actions depend on contextual information [ 41,42]. Further advancements includesemi-parametric
bandits[ 43,44], which integrate parametric models to estimate rewards, andneural bandits, which leverage deep
learning for breaking linear assumptions in large-scale contexts [ 45,46]. Additionally,non-stationary banditsaddress
environments where reward distributions change over time [ 47], whileadversarial banditsaccount for adversarial
changes in the reward structure [ 48]. Recent innovations have introduced the concept ofmeta-bandits, exemplified
by systems likeAutoBandit, which dynamically adapt and optimize bandit algorithms in real-time decision-making
environments [ 49]. These advancements have made MABs critical in applications such asonline recommendations,
advertisement placement,adaptive learning systems, and evenmachine learning algorithms[ 11]. Despite their
differences, these variants primarily differ in how they model rewards, represent actions, incorporate contextual
information, and handle uncertainty over time.
As a result, MABs have been widely applied to practical decision-making problems such as online recommendation,
advertisement placement, and adaptive learning systems [ 11], where decision pipelines often involve multiple interacting
components. Motivated by this diversity of modeling assumptions and application requirements, bandit formulations
vary depending on assumptions about the reward generation process, the availability of contextual information, and the
structure of the action space. Accordingly, we adopt a system-level perspective and defer a detailed component-wise
decomposition of bandit systems to the taxonomy presented in Section 4.2.
3 Survey Methodology
To ensure a comprehensive and objective overview, this study follows a systematic review methodology [ 50,51].
We conducted an extensive search across major scholarly databases using approximately 30key terms spanning
the intersection of Multi-Armed Bandits (MABs) and Large Language Models (LLMs). Our initial search yielded
over 300candidate papers. Through a rigorous manual screening process—focusing on the technical integration
of bandit mechanisms within LLM workflows—we refined the selection to over 100core papers. To facilitate
community research and ensure reproducibility, we have curated an open-source repository that indexes these works
according to our proposed taxonomy. The complete list of hierarchical search queries and the detailed selected
papers are documented in the repository to support further systematic updates. This resource is available at https:
//github.com/bucky1119/Awesome-LLM-Bandit-Interaction and will be continuously maintained to reflect
emerging advancements.
3

4 Taxonomic Framework
4.1 Component-Based Taxonomy of LLM Systems
Following recent surveys on large language models (LLMs) [ 24,25,5], the lifecycle of LLMs can be systematically
divided into two major stages: thebuilding stageand theaugment stage. While the building and augmentation stages
describe thelifecycleof LLM development and deployment, our taxonomy does not categorize methods by stages
themselves, but rather by thefunctional system componentsthat operate within or across these stages.
Building Stage.The goal of the building stage is to train a general-purpose foundation model that encodes broad world
knowledge and strong language understanding capabilities. This stage typically consists of two sub-stages: pre-training
and post-training [24].
During the pre-training stage, LLMs are trained on large-scale corpora using self-supervised objectives. This process
involves data collection and cleaning, tokenization, model architecture design (e.g., transformer-based architectures),
positional encoding schemes, and large-scale optimization [ 5]. Through pre-training, the model acquires general
linguistic representations and extensive world knowledge.
The post-training stage further shapes the model’s capabilities and behavior to better align with downstream usage
requirements. This stage commonly includes fine-tuning techniques such as instruction tuning, as well as alignment
methods (e.g., reinforcement learning from human feedback, RLHF) that aim to align the model’s outputs with
human preferences, values, and safety constraints [ 52]. Together, the pre-training and post-training stages yield a
general-purpose LLM that can serve as the backbone for a wide range of applications.
Table 1: Component-Based Taxonomy of LLM Systems
LLM Component Description
Pre-training Large-scale self-supervised training on broad corpora to learn general linguistic representations
and parametric world knowledge.
Fine-tuning Supervised or instruction-based post-training that adapts a foundation model to downstream
tasks or domains.
Alignment Post-training procedures that steer model behavior toward human preferences and safety
constraints using feedback signals (e.g., preference data or reward models).
Prompt Design and Se-
lectionDesigning and selecting prompts (including templates and exemplars) to specify tasks and
control behavior at inference time without changing model parameters.
Tool and Function Call-
ingMechanisms that enable LLMs to decide when and how to invoke external tools, APIs, or
functions and integrate their outputs into generation.
Context Understanding Structuring, filtering, and interpreting provided context (e.g., user intent, dialogue state, long
context) to support coherent and relevant conditional generation.
Retrieval-Augmented
Generation (RAG)Retrieving external information and conditioning generation on retrieved evidence to comple-
ment parametric memory and improve factual grounding.
Inference Optimization System-level strategies that reduce latency and cost (e.g., caching, batching, routing, speculative
decoding) while maintaining output quality during deployment.
Decoding Strategies Token-level generation procedures (e.g., sampling, beam search, reranking) that trade off
diversity, stability, and quality.
Adaptation and Person-
alizationMechanisms that tailor model behavior to users, tasks, or environments over time, enabling
adaptive and personalized interactions.
Augmentation Stage.Once an LLM has been trained, it can be directly applied to various tasks via prompting. However, in practical
deployments, LLMs often exhibit notable limitations, including hallucinations, lack of persistent memory, large computational
overhead, inherently stochastic generation behavior, and outdated or incomplete knowledge [ 25]. To address these issues and fully
exploit the potential of LLMs, researchers increasingly rely on external, system-level enhancement mechanisms [ 53], many of which
operate at inference time without (or with minimal) modification of model parameters. We refer to this phase as theaugmentation
stage[ 25]. The augment stage encompasses a variety of techniques and system components designed to improve usability, robustness,
and task performance. Representative examples include prompt design and selection, tool and function calling, retrieval-augmented
generation (RAG), adaptation and personalization mechanisms [54], as well as inference optimization strategies [55].
From asystematic component-based breakdownperspective, we abstract LLM systems as compositions of decision-relevant
functional modules that may reside at different stages of the LLM lifecycle. Considering that bandit algorithms—viewed as a
4

minimal modeling paradigm within reinforcement learning for sequential decision-making under uncertainty—are naturally suited
for characterizing and optimizing such decision-related modules (e.g., prompt selection, tool invocation, and context management),
we adopt this perspective to structure our taxonomy.
As illustrated in table 1, we identify a set of core components that together constitute an LLM system, includingpre-training,
fine-tuning,alignment,prompt design and selection,tool and function calling,context understanding,retrieval-augmented
generation (RAG),inference optimization,decoding strategies, andadaptation and personalization. While these components
are instantiated at different stages of the LLM lifecycle, they are grouped based on their functional roles in the overall system rather
than their temporal order.
For conceptual clarity, these components can be viewed as belonging to several high-level functional categories: training-related
components (pre-training, fine-tuning, and alignment), input control mechanisms (prompt design and selection, and tool and function
calling), context augmentation modules (context understanding and retrieval-augmented generation), inference and generation
processes (inference optimization and decoding strategies), and output adaptation strategies (adaptation and personalization).
However, to maintain consistency with the bandit taxonomy and to facilitate component-wise analysis in subsequent sections,
we treat individual components as the primary units of organization. See Section 5 for detailed component-level discussions and
representative bandit-based enhancement methods.
4.2 Component-Based Taxonomy of Bandit Systems
A bandit algorithm addresses a sequential decision-making problem in which an agent repeatedly selects actions under uncertainty
and updates its strategy based on observed feedback, with the objective of optimizing long-term performance such as cumulative
reward or regret [ 7,56]. Despite the diversity of existing bandit formulations and algorithms [ 40,7], their core decision processes
can be characterized by a shared system structure.
From a system perspective, a bandit algorithm can be viewed as a composition of functional components that together define how the
decision problem is specified and solved. Specifically, these components include the optimization objective that guides learning, the
definition of the action (arm) space, assumptions about the environment dynamics, the formulation of the reward signal, and the
exploration-driven decision mechanism used to select actions over time. Each component captures a distinct aspect of the bandit
decision pipeline, from problem modeling to action execution.
Based on this perspective, we adopt a component-based taxonomy to organize bandit systems according to their constituent modules
rather than their specific algorithmic forms.
Table 2: Component-Based Taxonomy of Bandit Systems
Bandit Component Description
Regret Minimization Ob-
jectiveDefines the learning goal as minimizing cumulative regret, i.e., the performance gap between
the learner’s chosen actions and an oracle benchmark (typically the best fixed arm, or the
best policy in contextual settings).
Arm Definition Specifies what constitutes an action (arm) and its structure, ranging from a finite set of
discrete arms to structured/continuous action spaces, including context-dependent actions in
contextual bandits.
Environment States assumptions on how rewards are generated over time, such as stochastic and station-
ary rewards, non-stationary or drifting rewards, or adversarial reward sequences, which
determine what guarantees are achievable.
Reward Formulation Defines the feedback signal and its type, such as scalar reward, binary click, cost-sensitive
reward, or vector/multi-objective reward, as well as whether feedback is full-information,
bandit (partial), delayed, or corrupted/noisy.
Sampling Strategy Describes how the algorithm balances exploration and exploitation when collecting data, e.g.,
optimism (UCB-style indices), randomized exploration (e.g., ϵ-greedy), posterior sampling
(Thompson sampling), or elimination-based schemes.
Action Decision Specifies the concrete decision rule executed each round given current estimates/posteriors
and constraints, e.g., selecting arg max of an index, sampling an arm from a learned
distribution, or choosing a constrained/safe action in combinatorial or risk-aware settings.
Concretely, within this taxonomy, each component corresponds to a distinct modeling or decision-making function in a bandit system.
Theregret minimization objectivespecifies the performance criterion that guides algorithm design and theoretical analysis. The
arm definitiondetermines the action space over which decisions are made, ranging from simple discrete arms to structured or
contextual action representations. Theenvironmentcomponent captures assumptions about reward generation dynamics, such as
stationarity, non-stationarity, or adversarial behavior. Thereward formulationdefines how feedback is observed, including issues of
noise, delay, sparsity, or partial observability. Thesampling strategydefines how uncertainty is modeled and exploited to construct
5

action selection policies, such as optimism-based methods or posterior sampling. Theaction decisioncomponent corresponds to the
execution step that selects a concrete action at each interaction based on the chosen strategy.
Table 2 summarizes the key components considered in this taxonomy. Detailed discussions of each bandit component and their
corresponding LLM-based enhancement mechanisms are provided in Section 6.
Having established this component-based decomposition, we are able to systematically map how bandit techniques enhance specific
components of LLM systems and, conversely, how LLMs can improve traditional bandit algorithms. By dissecting both domains at the
component level, this framework helps uncover previously hidden synergies, identify shared challenges, and provide clearer insights
into their integration. Moreover, it enables evaluation across multiple dimensions, including research problems, methodologies,
validation protocols, and datasets, thereby providing a structured foundation for understanding and advancing the intersection of
MAB algorithms and LLMs.
5 Bandit-Based Enhancements for LLM Systems
This section provides a component-level review of how bandit algorithms are integrated into large language models to improve
training, inference, and adaptation. Table 3 offers a consolidated overview of selected works discussed in this section.
Table 3: Bandit-Based Enhancements for LLM Systems
LLM Component Research Problem Bandit-Enhanced Solutions References
Pre-trainingStatic masking strategy selection Bandit-based dynamic masking pattern selection [57]
Static task sampling in multi-task pre-training Structured multi-task bandit formulation [58]
Suboptimal trade-off between data quality and diversity Online domain-level sampling ratio optimization [59, 60, 61]
Fine-tuningReward overfitting in RL-based fine-tuning Adaptive training data reweighting and selection [62, 63, 64, 65]
Inefficient preference data collection Bandit-based preference and context selection [66, 67, 68]
Static budget allocation under non-stationarity Bandit-based adaptive dataset mixture optimization [69, 70, 71]
AlignmentCostly and inefficient preference feedback collection Adaptive bandit-based preference query selection [72, 73, 74]
Non-stationary and myopic alignment dynamics Bandit-style exploration in online preference learning [75, 76, 77]
Structured decision stages in alignment pipelines Bandit-guided structured alignment pipelines [66]
Prompt Design
and SelectionBudget-limited prompt evaluation Bandit-based best prompt identification [78, 79, 80]
Implicit and costly prompt feedback Preference-based and dueling bandit optimization. [81, 82]
Suboptimal prompt composition Bandit-based example and trajectory selection [83, 84, 85]
Large-scale and privacy-constrained prompt spaces Offline, neural, and federated bandit optimization [86,87,88,89,
90]
Tool and
Function CallingDelayed and noisy feedback in multi-step tool use Execution-feedback-driven bandit tool selection [91, 92]
Large and heterogeneous tool spaces Semantic-aware bandit-based tool generalization [93, 17]
Delayed feedback and credit assignment Bandit-informed agentic tool optimization [94, 95]
Context
UnderstandingExploration and adaptation under sparse feedback Bandit-based in-context exploration [96, 97, 98, 99]
Inefficient long-context utilization Adaptive bandit-guided context selection [66, 100]
Non-stationary inference-time decisions Contextual bandit-based online strategy selection. [101,102,103,
104]
Retrieval-Augmented
Generation (RAG)Adaptivity requirements in retrieval control across queries
and budgetsBandit-based retrieval strategy selection [105, 106, 107]
Heterogeneity and context dependence of retrieved evi-
denceBandit-driven knowledge source selection [108, 100]
Multi-objective and hierarchical RAG optimization. Hierarchical and multi-objective bandit optimization [109, 110]
Inference
OptimizationStatic and inefficient inference model selection Adaptive bandit-based LLM routing [111,112,113,
114]
High inference cost and resource waste Bandit-driven cache and resource optimization [115]
Decoding
and RerankingLimited generalization of static decoding and reranking
heuristicsAdaptive decoding configuration selection [116, 117]
Rich verification signals in speculative decoding Verification-augmented online decoding optimization [118]
Irreversible dependencies in token-level decoding Token-level bandit modeling [119]
Adaptation
and PersonalizationHeterogeneity and non-stationarity of user preferences Online user behavior exploration–exploitation [120,121,122,
123]
Performance–cost trade-offs in multi-task personalization Bandit-based personalized model selection and routing [16, 111, 114]
Inference-time personalization without retraining Bandit-driven inference-time preference adaptation [119, 124]
6

5.1 Pre-training
Pre-training is the stage where an LLM acquires general-purpose language representations from massive corpora, and it is dominated
by decisions about which data to sample and how to allocate computation. Traditional heuristics for data mixture and curriculum
design are often static and manually tuned, which makes it hard to adapt to shifting data quality, domain distributions, and
task requirements. Therefore, integrating MAB into pre-training offers a principled way to treat data selection, masking, and
task scheduling as sequential decision problems under uncertainty, with feedback defined directly by pre-training progress and
downstream performance [60, 59, 57, 58].
Recent work explores several complementary ways to couple MAB with LLM pre-training.
One line of methods uses bandit-based policies to adapt training dynamics, for example by treating masking patterns as arms and
updating their selection based on reward signals such as loss reduction; Mukherjee et al. [ 57] follow this idea and show that a
bandit-driven dynamic masking scheme can reduce the number of training iterations and the need for extensive hyper-parameter
search.
A second line formulates multi-task pre-training as a structured bandit problem, where each task or task cluster is an arm and
the bandit exploits correlations among them; Urteaga et al. [ 58] implement this view through a decision transformer that predicts
task-specific rewards and leverages task-level structure to improve data efficiency and generalization.
A third line focuses on online data mixture and selection. Albalak et al. [ 59] propose an online data mixing (ODM) algorithm that
treats domain-level sampling ratios as bandit arms and adjusts them to maximize information gain, Zhang et al. [ 60] introduce
Quad, which uses MAB to balance data quality and diversity in a scalable selection pipeline, and Ma et al. [ 61] extend this idea
with AC-ODM, which models mixture optimization as an actor–critic process and uses gradient-alignment rewards to adapt domain
sampling for better cross-domain generalization.
Despite these advances, existing bandit-based pre-training methods still face limitations in reward design and granularity. Most
approaches rely on short-horizon or proxy-based rewards (e.g., loss or perplexity) and operate at coarse domain or task levels, leaving
open challenges in aligning bandit feedback with long-term generalization and in extending adaptive control to finer-grained data or
training decisions.
5.2 Fine-tuning
Fine-tuning aims to adapt LLMs to specific domains and align them with human feedback, but RL-based fine-tuning introduces
domain-specific challenges such as reward overfitting, unstable multi-step updates, and brittle generalization in low-resource settings,
as observed in Myanmar dialect translation [ 125]. Traditional supervised or RL pipelines rely on static datasets and manually
tuned exploration schedules, complicating the balance between efficiency and robustness while risking myopic optimization toward
short-term, noisy rewards [ 64]. Therefore, recent work increasingly formulates fine-tuning as a sequential decision problem, where
MAB-style allocation of data, rollouts, and compute can be used to stabilize learning and improve the feedback efficiency of LLM
alignment.
Recent methods can be broadly grouped into three lines.
First, curriculum and distribution-shaping approaches control the training signal by dynamically reweighting data and regularizing
policy updates. Dynamic data mixing adjusts sampling weights according to dataset redundancy and correlation to optimize
instruction tuning of Mixture-of-Experts models [ 62], while Iterative Data Smoothing progressively refines soft labels to alleviate
reward overfitting and improve reward convergence [ 63]. Complementary work proposes a dual-model online RL framework in
which a policy model and a reflection model cooperate to refine decisions from human feedback [ 64], and analyzes KL-regularization
as a mechanism that can reduce sample complexity and improve policy efficiency in RLHF [ 65]. Although not always framed
explicitly as MAB, these approaches implicitly tackle an exploration–exploitation trade-off over data and rewards by adapting the
effective training distribution.
Second, bandit-based preference and data selection methods explicitly treat samples or context fragments as arms and optimize
which feedback to collect. Duan et al. [ 66] introduce a chunk-sampling framework where a MAB mechanism selects context
segments based on reward feedback and multi-round rollouts, using the resulting diverse responses to construct preference data
and strengthen DPO training for long-context reasoning. Liu et al. [ 67] model alignment as a contextual dueling bandit problem,
using Thompson Sampling and an uncertainty-aware reward model to actively choose comparison pairs, thereby increasing the
utility of each preference query and providing a unified banditized optimization view of online alignment. Tajwar et al. [ 68] analyze
online sampling and negative gradients in preference fine-tuning, and integrate bandit strategies into both data sampling and reward
modeling to better exploit limited preference datasets.
Third, bandit-driven model and dataset selection methods cast different LLMs or data sources as arms and dynamically allocate
fine-tuning budgets in non-stationary environments. DynamixSFT formulates dataset mixing as a MAB problem and adaptively
tunes the mixture during fine-tuning to improve the use of heterogeneous corpora [ 69]. Xia et al. [ 70] model online model selection
as a non-stationary bandit, proposing TI-UCB to exploit the empirical pattern that reward first increases and then saturates with
more fine-tuning steps, and thus to distribute limited budget across candidate LLMs more effectively. Building on this idea, Xia et
al. [71] predict performance trends over training steps and combine UCB with change-detection mechanisms to decide which model
to continue fine-tuning so as to maximize long-term return under strict resource constraints.
7

However, these approaches also have notable limitations. Dynamic data mixing may face scalability issues as data grow and can
introduce additional noise into the training signal [ 62], while Iterative Data Smoothing has not yet been extensively validated in more
complex or highly non-stationary environments [ 63]. Future work should broaden the applicability of bandit-enhanced fine-tuning
to multilingual and multimodal settings, develop richer arm and reward abstractions that better capture long-horizon alignment
objectives, and provide stronger theoretical guarantees for non-stationary and hierarchical bandit formulations that arise in large-scale
LLM training.
5.3 Alignment
In LLM systems, the alignment component aims to steer model behavior toward human preferences, safety constraints, and task-
specific objectives [ 126,127]. Beyond static supervised fine-tuning, alignment is increasingly framed as a sequential and interactive
process, where models are repeatedly evaluated and adjusted based on feedback signals such as human judgments, preference
comparisons, or implicit user satisfaction [ 128]. These feedback signals are often noisy, sparse, delayed, and costly to obtain,
which poses significant challenges for efficiently selecting alignment actions and updating policies [ 119]. From a decision-making
perspective, alignment can be viewed as a process of repeatedly choosing alignment actions—such as preference queries, reward
model updates, or policy adjustments—under uncertainty about their long-term effects [ 129,130]. By framing alignment as a bandit
problem, LLM systems can adaptively balance these trade-offs and improve alignment outcomes under limited feedback budgets.
Recent work leverages multi-armed bandit (MAB) algorithms to improve alignment efficiency and robustness along several
complementary lines.
A first line of work focuses onsample-efficient preference query and data selection. In this setting, candidate preference comparisons
or training examples are treated as bandit arms, and the learner adaptively selects which queries to present to human annotators
under a constrained annotation budget. Several studies formulate preference annotation as an exploration–exploitation problem and
use uncertainty-aware or confidence-based criteria to prioritize informative comparisons, significantly reducing the cost of human
feedback while preserving alignment quality [ 72,73]. More recent approaches further provide theoretically grounded selection
rules for non-linear preference models. In particular, ActiveDPO derives uncertainty measures inspired by neural dueling bandits
and actively selects preference data based on the gradients of the current LLM, enabling data selection strategies that are explicitly
tailored to the model being aligned [74].
A second line of work views alignment as anonline and iterative preference optimization process, emphasizing the sequential
and potentially non-stationary nature of human feedback. From this perspective, alignment is modeled as repeated interaction
with a preference environment, where policies are continuously updated based on newly collected comparisons generated by the
current model. Count-based and optimism-driven exploration strategies instantiate bandit-style decision rules in online preference
optimization, encouraging systematic exploration of the prompt–response space during alignment [ 75]. Related formulations
establish connections between online preference optimization, self-play, and game-theoretic learning dynamics, showing that stable
aligned policies can be interpreted as equilibria of repeated preference learning processes [76, 77].
Beyond direct policy updates, a third line of work exploresstructured alignment decisionswithin the alignment pipeline using bandit
abstractions. Instead of treating alignment solely as parameter optimization, these methods apply MAB algorithms to intermediate
decision points that influence preference data construction. For example, recent work treats long-context chunks as bandit arms
and uses MAB-guided sampling to adaptively select informative subsets of context for response generation, which are then used to
construct high-quality preference pairs for DPO training [ 66]. This perspective highlights the flexibility of bandit formulations in
capturing modular and structured alignment decisions beyond standard policy optimization.
Overall, framing alignment as a bandit problem provides a unified abstraction for handling uncertainty, feedback scarcity, and
adaptivity in human-in-the-loop optimization. Nevertheless, existing bandit-based alignment methods face several challenges. Many
formulations rely on simplified assumptions about preference feedback or implicit reward structures, while empirical studies show
that alignment performance can be sensitive to reward scaling, regularization, and data distribution mismatch [ 72,131]. In addition,
online alignment settings reveal a strong coupling between exploration strategies and policy optimization dynamics: although
adaptive data generation improves coverage, it may also destabilize training if exploration is not properly regularized [76, 77].
These limitations also point to promising opportunities. Recent work suggests that bandit reasoning can be extended beyond isolated
decisions to coordinate multiple alignment components, such as preference querying and structured input selection [ 66]. Moreover,
advances in online preference optimization indicate that richer bandit formulations capable of handling non-stationary feedback may
further improve alignment robustness [75, 74].
5.4 Prompt Design and Selection
Prompt design and Selection aims to select instructions and in-context examples that elicit high-quality outputs from an LLM under
strict query and computation budgets. The core challenge is that LLM performance is highly sensitive to prompt choices, while the
model behaves as a black box and each query is costly. Traditional heuristic or manually designed prompts do not scale to large
candidate spaces, struggle with complex example relationships and privacy constraints, and provide little principled support for
trading off exploration and exploitation. Therefore, integrating MAB formulations provides a natural way to model prompt variants
as arms, optimize under bandit feedback, and formalize budget-aware prompt search [132].
Recent work leverages MAB algorithms to structure prompt optimization along several complementary lines.
8

A first line casts prompt selection as best-arm identification from a fixed candidate pool, seeking the most effective prompt under
tight evaluation budgets; here, prompts or even prompt-optimization strategies are arms, and bandit procedures control which variants
to evaluate, as in best-arm formulations for prompt pools and efficient variants that exploit prompt embeddings and clustering to
scale to large candidate sets [78, 79, 80].
A second line adopts preference-based and dueling bandits, where feedback is given via pairwise comparisons or implicit preferences
instead of numerical scores; dueling frameworks use human feedback to adapt prompt choice and, more recently, dueling MAB with
double Thompson sampling and prompt mutation to optimize prompts without relying on labeled data [81, 82].
A third line focuses on bandit-based selection and ordering of in-context examples or trajectory prompts, treating example subsets or
trajectories as arms and optimizing them for in-context learning or multi-task reinforcement learning; representative studies optimize
the composition and order of examples for better in-context learning, select trajectory-level prompts for decision transformers, and
use meta-bandit schemes such as EXPO and EXPO-ES to tune task descriptions, meta-instructions, and example sets for sequential
decision tasks [83, 84, 85].
A fourth line develops offline, federated, and neural bandit approaches that scale to large prompt spaces and privacy-sensitive settings:
logged bandit data and clustering are used to reduce variance in gradient-based prompt tuning, off-policy bandit optimization (DSO)
improves prompt policies for personalized recommendation tasks in large candidate spaces, neural bandits are combined with
transformers or soft prompts to optimize black-box LLMs under query limits, and federated bandit algorithms such as FedPOB
optimize prompts by sharing model parameters rather than raw data across parties [86, 87, 88, 89, 90].
Despite these advances, existing approaches often assume a pre-generated prompt or strategy pool, which constrains exploration
of genuinely novel prompts and limits adaptation to evolving tasks. Several methods also face practical issues such as high
computation cost in large-scale sequential decision tasks, substantial communication overhead in federated prompt optimization,
and misidentification risks in best-arm procedures over very large candidate sets [ 85,90,79]. Future research should therefore
develop more adaptive and generative bandit formulations that jointly expand and evaluate the prompt space, improve efficiency and
robustness for large-scale deployments, and reduce reliance on dense human feedback and heavily engineered candidate pools.
5.5 Tool and Function calling
Tool and function calling enables LLM systems to interact with external APIs, databases, and software tools, extending their
capabilities beyond pure text generation [ 133]. At inference time, models must decide whether and how to invoke tools under
partial or ambiguous context, where incorrect usage can incur nontrivial costs such as latency, execution failures, or degraded user
experience [ 134,135]. These decisions are inherently sequential and uncertain, as the utility of a tool invocation depends on task
context, user intent, and downstream outcomes [ 17]. Bandit-based formulations therefore offer a principled abstraction for modeling
tool selection as an exploration–exploitation problem, enabling adaptive and cost-aware tool usage through interaction.
Recent work improves tool and function calling in LLM systems along several complementary lines.
A first line models tool invocation as a multi-turn, error-aware reasoning process, showing that learning from execution feedback and
assigning credit at finer temporal granularity can substantially improve robustness in multi-step tool-use scenarios [91, 92].
A second line addresses scalability by incorporating semantic tool descriptions, contextual representations, or evolving abstractions,
allowing tool selection policies to generalize across large and dynamic tool spaces [93, 17].
A third line adopts a system-level view, framing tool use as a long-horizon learning problem in agentic settings, where reinforce-
ment learning optimizes tool-calling policies through interaction while exposing challenges such as delayed feedback and credit
assignment [94, 95].
Despite these advances, tool calling remains challenging due to limited generalization across heterogeneous tools, difficulty in credit
assignment under long-horizon interactions, and increasing uncertainty and cost in large tool ecosystems. These challenges also
point to promising opportunities: integrating bandit-based decision-making with agentic reinforcement learning and rich contextual
representations offers a scalable and adaptive framework for robust tool orchestration, positioning tool and function calling as a core
sequential decision component in LLM-based systems.
5.6 Context Understanding
Contextual understanding in LLM-based decision systems aims to interpret rich, evolving input streams and translate them into
adaptive exploration–exploitation behavior. Traditional supervised and reinforcement learning methods often struggle to scale in
high-context regimes and to track non-stationary user preferences, especially when feedback is sparse or costly to obtain. Integrating
MAB with LLMs is therefore valuable, as MAB provides principled uncertainty-aware exploration and feedback-efficient learning,
while LLMs contribute semantic reasoning and in-context adaptation over complex textual or structured environments [96, 98, 99].
Recent work can be grouped into three main directions.
First, a line of research studies bandit-driven in-context exploration and feedback-efficient learning. Krishnamurthy et al. [ 96] use
MAB benchmarks to probe how well LLMs perform in-context exploration, revealing when prompt-based adaptation suffices and
when explicit bandit structure is needed. Building on this idea, Rahn et al. [ 97] manipulate internal activations to steer exploratory
behavior, effectively coupling bandit-style exploration objectives with mechanistic control of LLM representations. To reduce the
9

cost of supervision, Dwaracherla et al. [ 98] propose a double Thompson sampling scheme that maintains competitive performance
while minimizing feedback queries, and Tang et al. [ 99] cast code repair as a bandit problem where Thompson sampling over
candidate patches guides LLM-based program edits.
Second, several methods use bandit algorithms to select and attribute context in long or structured documents. Duan et al. [ 66] treat
document chunks as arms in a LongMab-PO framework and use UCB-style selection to focus the LLM on the most informative
combinations of text segments for long-context reasoning. Pan et al. [ 100] similarly model each passage as an arm and apply adaptive
sampling under a query budget to identify context most critical to answering, thereby making the LLM’s context dependencies
explicit and more efficiently exploited.
Third, a growing body of work uses contextual bandits to model user preferences, select among multiple LLMs, or guide inference-
time decisions, including adversarial ones. Qin et al. [ 101] introduce the MoRE framework, which combines explicit and implicit
preferences in a multi-view reflection system to improve sequence modeling in recommendation tasks. Nie et al. [102] incorporate
bandit algorithms at inference time so that the LLM can adaptively choose reasoning strategies based on contextual signals. Poon et
al. [103] formulate online model selection as a contextual bandit, using a greedy LinUCB policy to dynamically pick the most suitable
LLM per step rather than modeling future context evolution. In a different direction, Ramesh et al. [ 104] design a MAB-based
context-switching query procedure that incrementally elicits harmful outputs, demonstrating how bandit-guided querying can probe
and circumvent LLM safety mechanisms.
Despite these advances, current approaches still face important limitations. Many exploration schemes are evaluated in simplified
settings and may fail to generalize to highly dynamic, safety-critical environments with complex feedback channels. Bandit-guided
context selection, activation control, and multi-LLM routing can introduce non-trivial computational overhead, especially when
managing high-entropy policies or large search trees. Future work should therefore prioritize scalable exploration algorithms that
remain robust under distribution shift, while designing more adaptive, feedback-efficient bandit frameworks that can operate with
noisy, delayed, or partially observable signals [98, 101].
5.7 Retrieval-Augmented Generation (RAG)
Retrieval-Augmented Generation (RAG) integrates external knowledge sources into LLM inference to improve factual consistency
and interpretability. However, deciding when to retrieve, how many documents to fetch, and which strategy to use under varying
query complexity, context requirements, and resource constraints remains a central challenge. Therefore, combining LLMs with
MAB provides a principled way to cast retrieval control as online decision-making under uncertainty, balancing utility, latency, and
computational cost.
A first line of work formulates retrieval strategy selection as a bandit problem to adapt retrieval intensity and prompt configuration
to query difficulty and deployment conditions. MAB-RAG models candidate retrieval strategies as arms and uses online feedback
to adjust retrieval behavior for queries with different complexity, enabling dynamic control over retrieval depth and pattern [ 105].
AdaRAG frames the tuning of retrieval ratio and prompt length as bandit convex optimization in edge settings, so that the system can
automatically trade off generation quality and end-to-end delay under limited resources [ 106]. M-RAG further couples multi-agent
reinforcement learning with a bandit mechanism that treats database partitions as arms, allowing the system to adaptively select
knowledge partitions that best support each query while maintaining scalable retrieval [107].
A second line of work focuses on bandit-driven knowledge and evidence selection. KnowGPT employs a bandit controller that,
conditioned on query context, chooses among candidate knowledge sources and prompt formats, thereby jointly adapting what to
retrieve and how to present it to the LLM [ 108]. In a related direction, Pan et al. use bandit exploration over document subsets to
identify which retrieved passages genuinely support the model output, reducing exhaustive subset evaluation while preserving key
supporting evidence in RAG pipelines [100].
A third direction leverages bandits for multi-objective and hierarchical RAG optimization. For knowledge-graph-driven RAG in
non-stationary environments, Tang et al. design a deep multi-objective contextual bandit and employ the Generalized Gini Index to
balance accuracy, coverage, and latency, enabling explicit control over competing objectives [ 109]. From a systems perspective,
AutoRAG-HP adopts a hierarchical MAB framework in which a high-level module selects pipeline variants and a low-level search
tunes associated hyperparameters online, providing an automatic mechanism for RAG hyperparameter optimization [110].
Despite these advances, existing bandit-based RAG systems often rely on fixed bandit formulations and manually specified reward
proxies, which may be brittle when facing new task types, domain shifts, or highly skewed query distributions [ 105,109]. Moreover,
frequent parameter updates and the evaluation of multiple retrieval or configuration strategies can introduce substantial computational
overhead, limiting scalability in large-scale or real-time applications. Future work should pursue more adaptive and sample-efficient
bandit architectures, richer reward designs grounded in long-term user-centric signals, and tighter coupling between bandit controllers
and LLM reasoning to enable robust retrieval–generation coordination across diverse and evolving environments.
5.8 Inference Optimization
Inference optimization aims to improve the efficiency and cost-effectiveness of LLM deployment at inference time, encompassing
latency reduction, resource allocation, and model or strategy selection under computational constraints [ 136]. In practice, LLM
systems must dynamically balance output quality against monetary cost and system load, with these trade-offs varying across queries
due to differences in input complexity and user requirements [137].
10

These challenges can be naturally framed as sequential decision-making problems, where the system repeatedly selects inference
strategies—such as choosing among model variants or allocating computation budgets—and observes feedback only for the executed
choice. Bandit algorithms provide a principled abstraction for this setting by enabling adaptive exploration–exploitation trade-offs,
allowing inference policies to be learned online under uncertainty.
A first line of work formulates LLM routing and model selection as a bandit problem, treating candidate models or inference
backends as arms. Li et al. propose a cost-efficient LLM inference framework that adaptively routes queries based on observed
performance–cost trade-offs [ 111]. Panda et al. study adaptive LLM routing under explicit budget constraints, framing inference-
time model selection as a contextual bandit problem [ 112]. Tongay et al. further demonstrate that dynamic model selection via
sequential decision-making can effectively respond to changing workloads and deployment conditions [ 113]. Wei et al. introduce a
preference-conditioned contextual bandit framework that learns routing policies directly from bandit feedback, enabling fine-grained
control over performance–cost trade-offs without full-information supervision [114].
A second line of work focuses on system-level cost-aware inference control, particularly through caching and reuse of LLM outputs.
Yang et al. model LLM cache selection as a combinatorial bandit with heterogeneous query sizes, showing that adaptive cache
management can substantially reduce inference cost while preserving performance guarantees [115].
Despite these advances, existing bandit-based inference optimization methods often rely on manually designed reward proxies
that may be brittle under shifting user preferences or evolving deployment objectives [ 111,114]. Moreover, exploration over
inference strategies can introduce nontrivial overhead in large-scale or latency-sensitive systems. Future work should investigate
more sample-efficient and preference-aware bandit formulations, tighter integration between bandit controllers and LLM reasoning,
and extensions to multi-turn and conversational inference settings.
5.9 Decoding Strategies
Decoding and reranking components govern how candidate outputs are generated and selected during LLM inference. Modern
LLMs often produce multiple candidate responses through stochastic decoding strategies, which are subsequently filtered or reranked
based on criteria such as relevance, safety, or user preferences. However, the effectiveness of decoding or reranking strategies can
vary significantly across tasks, contexts, and users, and static heuristics often fail to generalize. From a sequential decision-making
viewpoint, decoding configuration and candidate selection can be modeled as a bandit problem, enabling adaptive trade-offs between
output diversity and quality via online feedback.
Recent work can be grouped into three main lines.
A first line of work uses bandits for training-free adaptive control of speculative decoding configurations. Hou et al. model alternative
speculative setups as arms and apply regret-driven online selection to approach near-oracle configuration choice under changing
prefixes and workloads [ 116]. In a more targeted instantiation, Liu et al. cast the choice of draft length as a small-arm bandit and use
Thompson sampling to dynamically balance acceleration and verification success [117].
A second line shows that some speculative settings provide richer-than-bandit feedback. By leveraging verification structure, Liu et al.
demonstrate that one can efficiently score multiple drafters per query, upgrading partial-feedback bandits to (near) full-information
online learning and yielding faster identification of strong drafters than standard bandit exploration [118].
A third direction provides theoretical foundations at finer granularity. Shin et al. formalize decoding as irreversible token-by-token
selection, establish hardness in general, and identify structural conditions under which sublinear regret becomes achievable, offering
a principled lens on when simple decoding rules (e.g., greedy) can be effective [119].
Overall, existing bandit-based decoding approaches are most effective for speed- or acceptance-oriented objectives, but they often rely
on hand-designed reward proxies that imperfectly capture user-perceived quality, safety, or long-term utility [ 116,117]. In addition,
maintaining online controllers and evaluating multiple decoding configurations can introduce non-negligible control overhead,
partially offsetting the gains from acceleration [118].
Future work should therefore move toward contextual and multi-objective bandit formulations that explicitly incorporate user-centric
quality and safety signals, potentially leveraging richer or counterfactual feedback when verification or logging permits [ 118,119].
Moreover, the bandit framing should be extended beyond decoding acceleration to reranking-specific decisions, such as adaptive
best-of- Nsizing and reranker selection under delayed or preference-based feedback, which remain largely underexplored in current
systems.
5.10 Adaptation and Personalization
In adaptation and personalization, the key problem is to align LLM behaviour with user-specific and time-varying preferences while
keeping inference and update costs acceptable. Traditional personalization pipelines rely on static fine-tuning or offline clustering,
which are slow to adapt, expensive to retrain, and often brittle under heterogeneous feedback across tasks and modalities. Integrating
MAB with LLMs offers a principled way to balance exploration of new behaviours and exploitation of learned preferences under
limited interaction budgets, especially in conversational agents, recommendation, and summarization systems. [123, 121, 120]
Recent work can be grouped into three main lines.
11

The first line uses neural bandits to directly personalize soft prompts or representations based on user feedback. Chen et al. [ 120]
and Monea et al. [ 122] model user interactions as contextual bandits, updating soft prompt embeddings to improve the relevance
of generated content while preserving a light-weight adaptation mechanism for each user. Moerchen et al. [ 123] employ a bandit
framework in a speech-based NLU system, using implicit feedback to adjust music recommendation policies without retraining the
full model. In parallel, Gönç et al. [ 121] propose an online learning framework that repeatedly fine-tunes intent classification models
with reinforcement-style updates, enabling faster adaptation to novel user inputs in dialog settings and complementing bandit-based
approaches.
A second line focuses on cost-efficient model selection and routing for multi-task personalization. Dai et al. [ 16] design a
compositional MAB that selects among multiple LLMs across tasks, trading off performance against computation by treating model
choices as combinatorial arms. Li et al. [ 111] introduce user-controllable preference vectors that encode the relative importance of
accuracy and cost into the bandit reward, and combine them with model identity embeddings to learn routing policies that generalize
as user needs and available models change. Wei et al. [ 114] extend this idea to multi-objective bandits, using low-dimensional
preference vectors over performance and cost so that model routing can be adjusted online according to user-specified trade-offs at
inference time.
A third line addresses inference-time alignment and adaptive reasoning for personalization. Shin et al. [ 119] reformulate alignment
and personalization as a linear bandit problem over tokenized decoding trajectories, combining structured sequence hypotheses with
online updates so that the model learns user preferences during inference without further training. Zhang et al. [ 124] compare LLMs
and humans in non-stationary environments and show that adjusting reasoning patterns toward more directed exploration improves
the adaptability of LLMs, thereby informing bandit-style strategies for personalized decision-making under preference drift.
However, most existing LLM–MAB personalization methods still assume relatively stable or predictable feedback, which may
not hold in highly dynamic or sparse interaction regimes. Many approaches rely on soft-prompt adaptation or low-dimensional,
manually specified preference vectors, which can miss richer, multi-faceted user objectives and face cold-start issues when only
a few generic test items are available. [ 120,122,111,114] Future work should explore multi-modal and long-horizon feedback,
more expressive preference representations that cover latency, robustness, and risk, and bandit algorithms that can generalize across
complex, unstructured environments while updating user models online at inference time.
6 LLM-Based Enhancements for Bandit Systems
This section systematically examines how large language models (LLMs) enhance various components of multi-armed bandit
algorithms. Table 4 presents a consolidated overview of representative papers corresponding to each component.
6.1 Regret Minimization Objective
In MAB problems, the regret minimization objective is to maximize cumulative reward while keeping regret small. Classical
algorithms typically assume stationary reward distributions and optimize regret through fixed exploration schedules, which leads to
slow convergence and high computational cost in large or rapidly changing environments. Therefore, recent work explores how
LLMs can provide prior knowledge, richer contextual understanding, and adaptive feedback so that regret-minimization objectives
can be optimized more efficiently in complex decision-making tasks.
Existing studies cluster into several lines of work that reshape the optimization objective by coupling LLM reasoning with bandit
exploration–exploitation.
A first line uses LLMs to guide information-theoretic exploration, explicitly trading off information gain against regret: Shufaro et
al. [138] use LLM-derived insights to target promising regions of the action space, reducing redundant sampling while maintaining
exploration quality, which effectively refines the objective from uniform regret minimization to information-efficient regret control.
A second line leverages contextual priors and in-context learning to accelerate regret minimization. Xia et al. [ 139] show that LLM
agents can integrate contextual knowledge about tasks into the bandit objective, allowing the algorithm to down-weight expensive
exploration and achieve faster convergence in real-time decision scenarios.
A third line analyzes regret from the perspective of adaptive LLM agents. Park et al. [ 14] study how LLMs update their behavior
based on accumulated experience, providing a regret-centric view of iterative adaptation and suggesting optimization objectives that
explicitly encode learning dynamics over time.
A fourth line focuses on non-stationary environments, where the optimization objective must evolve with the underlying process.
De Curt‘o et al. [ 140] combine LLM-based real-time insights with bandit strategies to dynamically adjust the effective objective,
reducing exploration cost and preserving cumulative reward under shifting reward distributions.
These LLM-enhanced objectives remain constrained by several limitations. Many approaches rely heavily on priors encoded in
the LLM that may not generalize across domains, which raises robustness and safety concerns for regret guarantees. Moreover,
integrating LLM calls into the optimization loop can introduce substantial computational overhead and latency, limiting scalability in
strict real-time settings. Future work should design lightweight surrogates or hybrid architectures in which LLMs shape high-level
objectives and priors while core bandit updates remain computationally efficient, and should develop principled methods to balance
static LLM priors with online adaptation so that regret remains controlled under abrupt or persistent environment shifts.
12

Table 4: LLM-Based Enhancements for Bandit Systems
Bandit Component Research Problem LLM-Enhanced Solutions References
Regret Minimization
ObjectiveSlow convergence under uniform regret
minimizationLLM-guided information–regret trade-off
optimization[138]
Costly exploration in real-time decision sce-
nariosContextual-prior–driven regret minimiza-
tion[139]
Static regret objectives under learning dy-
namicsRegret-centric adaptive agent optimization [14]
Regret degradation under shifting rewards Dynamic regret objective adjustment [140]
Arm
DefinitionRedundant exploration in high-dimensional
action spacesLLM-based semantic arm compression [141, 142]
Static arms under preference drift Dynamically redefining or personalizing
arms through contextual reasoning[139, 143]
Limited expressiveness of discrete arms LLM-generated structured arm manifolds [144]
EnvironmentStationary or low-dimensional assumptions
fail in non-stationary or semantic-rich set-
tingsInterpreting evolving contexts to guide
adaptation and detect regime shifts.[140,145,13,
146]
Realistic environments are difficult to simu-
late for bandit trainingGenerating structured synthetic environ-
ments with diverse semantic dynamics[144, 147]
Unstructured and latent environment states Language-to-context environment mapping [148, 149]
Reward
FormulationNoisy and ambiguous feedback Context-aware dynamic reward shaping [150, 14]
Sparse or ethically constrained rewards LLM-based reward prior and regression [151, 149]
Misalignment between numeric rewards
and human goalsNatural-language-to-reward translation [152, 145]
Long-horizon credit assignment difficulty LLM-mediated preference and algorithmic
rewards[153, 154]
Sampling
StrategyNaive exploration is costly in high-
dimensional textual action spacesLLM-guided informative sampling [138,155,15,
13, 16, 151]
Classical sampling rules generalize poorly
across heterogeneous tasksLearning meta-level exploration policies
from textual interaction histories[153, 156]
Explicit posteriors are hard to define for
language-based rewardsUsing natural language as an implicit poste-
rior to guide exploration[157, 158,
148]
Action
DecisionPoor action ranking under complex context LLM-enhanced value and preference esti-
mation[12,159,155,
160,161,151]
Rigid numerical decision heuristics LLM-as-policy action selection [153, 157,
158]
Inconsistent confidence-based decisions LLM-aligned posterior-driven action choice [156, 148]
6.2 Arm Definition
In MAB problems, arms define the set of actions available to the algorithm, and their design is crucial for balancing exploration and
exploitation in complex, high-dimensional environments. Traditional methods usually treat arms as discrete indices with limited
structure, which makes it difficult to encode heterogeneous contextual variables and semantic relationships, leading to suboptimal
action partitions and redundant exploration. Integrating LLMs into arm definition introduces a semantic layer that can jointly capture
context, user preferences, and latent structure, thereby enabling more adaptive and interpretable action spaces.
Existing work explores three main lines for LLM-enhanced arm definition.
First, semantic arm construction and compression uses LLMs to map high-dimensional options and feedback into compact,
semantically coherent arm sets. Verma et al. [ 141] propose a neural dueling bandit framework in which an LLM defines arms and
their relations based on semantic similarity, so that the bandit operates over meaningfully clustered actions, improving decision
accuracy while reducing the effective action space. Liu et al. [ 142] show that, in a reinforcement learning setting, a pretrained
language model can organize actions into prior-informed categories before interaction, thereby making subsequent online decisions
more sample-efficient.
Second, dynamic and personalized arm redefinition employs LLM reasoning to maintain relevant action sets under non-stationary
preferences. Xia et al. [ 139] use an LLM within a dueling bandit to continually refine arm definitions according to current contextual
13

cues, so that the compared actions remain aligned with the most informative semantic distinctions in real time. Song et al. [ 143] study
a health incentive task where an LLM generates personalized natural language for each intervention type selected by a contextual
MAB, effectively expanding a small set of discrete interventions into a rich manifold of language-level arm realizations tailored to
individual users.
Third, generative arm augmentation combines LLM-based attribute extraction with data generation to define arms as semantically
structured data subsets. Kerim et al. [ 144] use an LLM to automatically extract domain-specific attributes and construct structured
prompts for Stable Diffusion, and then treat the resulting synthetic data clusters, partitioned by semantic attributes, as distinct arms
for downstream sample selection.
These approaches demonstrate that LLM-guided arm definition can substantially improve the adaptability and robustness of MAB
algorithms in high-dimensional, context-sensitive environments. However, they also introduce additional computational overhead and
latency, which can limit deployment in strict real-time settings and raise concerns about the stability of LLM-generated semantics.
Future research should focus on more efficient pipelines for LLM-based arm construction, scalable personalization that balances
individual feedback with generalizable arm structures, and theoretical analyses that clarify how semantic arm design affects regret
and robustness in large-scale bandit systems.
6.3 Environment
In MAB settings, the environment determines how contexts, rewards, and latent dynamics evolve, so non-stationarity and adversarial
shifts can rapidly invalidate fixed exploration–exploitation strategies. Previously proposed reinforcement learning (RL) and Bandit
algorithms [ 6,7] are mainly suited for stationary environments. Classical methods usually assume stationary reward distributions or
simple drift models and rely on low-dimensional statistics, which makes them brittle under high-dimensional semantic feedback,
delayed effects, or abrupt regime changes. Reference [ 162] summarizes the solutions of RL algorithms in both finite and infinite
horizons for non-stationary environments. Integrating LLMs into the environment component is therefore valuable, as their contextual
and sequence reasoning enables richer state representations, change detection, and forecasting of future conditions in which bandit
policies will operate.
Existing work can be grouped into three main lines.
First, several studies use LLMs to guide adaptation in non-stationary or adversarial environments. These approaches let the LLM
parse evolving contexts and histories to adjust bandit beliefs, anticipate regime shifts, or regularize decisions against adversarial
manipulation. For example, De Curtò et al. [ 140] design an LLM-informed policy for non-stationary settings that conditions
arm selection on changing contextual patterns; Verma et al. [ 145] design a social-choice–guided framework in which LLMs
generate candidate reward functions and an adjudicator selects those best aligned with multi-group preferences in structured RMAB
environments; Alamdari et al. [ 13] use LLM-generated priors to warm-start policies in new environments and reduce exploration
time; and Salewski et al. [ 146] show that LLMs can track dynamic contexts in real time, identify systematic biases, and refine policy
updates accordingly.
Second, a complementary line uses LLMs to construct more structured environments for training bandit agents. Kerim et al. [ 144]
employ LLMs to analyse the image domain and define an organized attribute space in which diffusion models generate diverse
yet semantically coherent samples, enabling the bandit to explore a structured synthetic dataset and select the most informative
image subsets. Similarly, Zala et al. [ 147] use LLMs to generate and adapt synthetic environments that mimic complex real-world
dynamics, thereby improving agent training and long-term adaptability.
Third, several works exploit LLMs to transform unstructured narratives into bandit-ready contextual representations, effectively
enriching the observable environment. Arumugam et al. [ 148] summarize interaction trajectories in natural language and use these
summaries to update posterior beliefs over rewards and transitions in a Bayesian fashion without explicit parametric modeling; and
Nessari et al. [ 149] map unstructured clinical notes into structured patient features that serve as high-dimensional, interpretable
contexts for online treatment recommendation in real-world medical environments.
Despite these advances, most LLM-enhanced environment models still rely on simplified assumptions about how drift, adversaries,
and feedback loops arise, which limits their robustness in truly open-world settings. Future work should more tightly couple
LLM-generated priors with on-line adaptive learning so that policies remain flexible in dynamic environments without overfitting to
handcrafted descriptions or pre-specified knowledge. It is also important to develop models that generalize across heterogeneous tasks
and environment types, proactively predict harmful shifts, improve robustness in adversarial regimes, and leverage LLM-generated
synthetic environments to expose agents to the diversity and hidden biases of real-world conditions while maintaining fairness in
downstream decisions.
6.4 Reward Formulation
Reward formulation is a central but challenging component in MAB, because the agent only observes partial, noisy, and often
delayed feedback while needing to optimize long-term performance. Traditional designs rely on fixed numerical rewards or manually
engineered transformations, which struggle with non-stationary preferences, multi-objective trade-offs, and ambiguous or language-
based feedback. LLMs provide a way to translate rich contextual and linguistic signals into structured reward objects, thereby
aligning MAB objectives more closely with high-level human goals and complex environments.
Existing work can be grouped into four main lines.
14

First, a set of studies uses LLMs for dynamic and context-aware reward shaping in non-stationary or multi-agent settings. Felicioni
et al. [ 150], and Park et al. [ 14] further show how LLMs can detect changes in reward patterns, disambiguate noisy or underspecified
feedback, and adapt online reward structures to improve cumulative performance.
Second, some work treats the LLM as a reward surrogate or prior model. Sun et al. [ 151] propose TS-LLM and RO-LLM frameworks
in which the LLM performs in-context regression from contextual descriptions to expected arm returns, replacing classical parametric
regressors within Thompson sampling and robust optimization to better handle nonlinear reward surfaces. Nessari et al. [ 149]
use LLM-generated structured features to train counterfactual outcome models in healthcare, and then use the predicted potential
treatment effects as prior rewards for the bandit, which mitigates sparse and ethically constrained reward observations.
Third, LLMs are used to construct reward functions directly from natural language specifications of objectives and preferences.
Behari et al. [ 152] let an LLM convert high-level public-health guidelines into explicit RMAB reward functions and iteratively
refine them, so that the decision process tracks evolving human policy priorities. Verma et al. [ 145] extend this idea by generating
multi-objective reward candidates from prompts and applying a social-choice-style evaluation step to select reward structures that
best reconcile conflicting subgroup priorities.
Fourth, several studies integrate algorithmic and preference-based signals into the LLM’s reward modeling. Chen et al. [ 153]
design strategic rewards based on instantaneous regret and consistency with UCB-style decisions, which are then processed by
an LLM policy optimized via RL to improve exploration and alleviate long-horizon credit assignment. Xia et al. [ 154] propose a
dueling-bandit setting where the LLM transforms pairwise preference feedback expressed in natural language into decision signals,
effectively replacing explicit numerical rewards and improving decision efficiency in preference-driven tasks.
Despite these advances, LLM-based reward formulation still faces several limitations. Current methods often depend on costly
model queries and may inherit biases or inconsistencies present in language feedback, which can distort long-run bandit behavior.
Future work should develop more sample-efficient, interpretable, and theoretically grounded reward-learning pipelines that combine
LLM priors with off-policy bandit data, and systematically study robustness under distribution shift, multi-objective trade-offs, and
misaligned or strategic human feedback.
6.5 Sampling Strategy
In MAB problems, the sampling strategy defines how actions are selected based on current uncertainty estimates, governing the
balance between exploration and exploitation throughout the learning process. LLM-enhanced sampling strategies aim to address
this classical dilemma when actions, context, and feedback are expressed in high-dimensional text. In such settings, naive exploration
is costly and hand-crafted rules based only on numeric statistics struggle to capture semantic structure, shifting user preferences, or
heterogeneous costs. Integrating LLMs into the sampling module allows the algorithm to use textual histories and prior knowledge to
propose more informative arms, potentially improving sample efficiency under tight interaction or computation budgets.
Existing studies can be broadly grouped into three methodological directions.
A first line uses LLMs to guide information-aware and cost-aware exploration on top of standard bandit statistics. Shufaro et al. [ 138]
formalize a trade-off between regret and information gain, and query an LLM to identify contextually informative arms so that
exploration reduces redundant sampling while controlling cumulative regret. Related work uses LLM reasoning to prioritize arms in
high-dimensional contextual spaces or to adapt exploration intensity over time, for example by ranking candidate arms using textual
cues [ 155], adjusting exploration strength based on observed feedback [ 15], or “jump-starting” learning with LLM proposals that
reduce early-stage exploration [ 13]. Dai et al. [ 16] study online selection among multiple LLMs, using LLM-provided contextual
insight to design cost-efficient sampling policies under resource constraints, while Sun et al. [ 151] tune the generation temperature of
an LLM to implement a Thompson-style randomized sampling mechanism, treating controlled output stochasticity as a proxy for
posterior uncertainty without explicitly maintaining a parametric posterior.
A second line lets LLMs directly learn and represent exploration policies that behave like meta-bandits. Chen et al. [ 153] train
LLMs via supervised fine-tuning or reinforcement learning to imitate classical strategies such as UCB, so that the model maps
textual reward histories into explicit next-arm choices and can generalize exploration behavior across different reward distributions.
Schmied et al. [ 156] further enhance self-generated chain-of-thought, combining it with mechanisms such as ϵ-greedy selection,
self-consistency, and reward-gain heuristics so that the LLM’s reasoning trace actively shapes when to explore or exploit.
A third direction leverages language as an implicit posterior over rewards or environments, using natural-language inference in place
of explicit probabilistic sampling. Hazime et al. [ 157] ask the LLM to decide in natural language when to switch arms based on
textual reward histories, thereby approximating exploration–exploitation trade-offs without explicit numeric confidence bounds.
Lim et al. [ 158] treat text-based success and failure feedback as updates to the LLM’s subjective reward estimates, effectively using
language as the sampling basis instead of formal posterior sampling. Extending this idea to reinforcement learning, Arumugam et
al. [148] prompt LLMs to generate language-level posterior hypotheses about the environment and then sample "plausible MDPs"
from these hypotheses, realizing a PSRL-like exploration mechanism where uncertainty is encoded and manipulated in natural
language before driving action selection.
Despite these advances, current LLM-based sampling strategies often rely on heuristic prompt designs and narrow experimental
settings, which limits theoretical understanding and robustness in large, noisy, or highly multi-armed environments. For example,
purely language-driven exploration can be sensitive to decoding parameters such as temperature, leading to unstable sampling
quality when rewards are noisy or the arm set is large [ 157]. Future work should therefore seek stronger theoretical guarantees for
15

language-mediated exploration, broader empirical validation across dynamic and non-stationary tasks, and more resource-efficient
designs that reduce LLM call overhead. Promising directions include post-learning schemes that feed back long-horizon bandit
traces into the LLM, and tighter integration with external memory, tool use, and adaptive prompting so that LLM-guided sampling
becomes more systematic, controllable, and interpretable.
6.6 Action Decision
The action decision component determines which arm is selected given current statistics and context, and is therefore central
to maximizing long-term reward in dynamic environments. Classical policies such as greedy rules, upper confidence bounds,
or probability matching rely on hand-crafted numerical heuristics that struggle with high-dimensional context, non-stationary
preferences, and semantically rich feedback. Integrating LLMs with MAB aims to inject prior knowledge and contextual reasoning
into this decision process, enabling more adaptive and interpretable action choices.
Recent work on LLM-enhanced action decision can be grouped into three main directions.
First, several studies use LLMs to refine value and preference estimates before applying a bandit rule. Baheri et al. [ 12] propose an
LLM-augmented contextual bandit framework in which the LLM encodes textual contexts into dense semantic representations to
improve action selection. Wang et al. [ 159] exploit LLM-based semantic analysis of context and interaction history to identify actions
with potentially higher reward in dynamic user-interest exploration. Guo et al. [ 155] combine LLM priors with high-dimensional
action spaces to guide arm selection, while Liu et al. [ 160] treat the LLM as an evolutionary optimizer that improves policy quality
with reduced exploration. Zhang et al. [ 161] let an LLM continuously tune hyperparameters in real-time tasks so that updated
configurations induce better downstream decisions, and Sun et al. [ 151] extend LLM prediction to pairwise preference comparison,
embedding estimated win probabilities into a dueling bandit to obtain more stable preference-based action rankings.
Second, another line of work treats the LLM itself as the decision rule that maps summaries of interaction data directly to arms.
Chen et al. [ 153] feed statistical aggregates such as empirical means or confidence summaries into an LLM, which learns decision
heuristics resembling greedy or UCB-style rules and assumes full responsibility for action choice. Hazime et al. [ 157] and Lim et
al. [158] instead prompt the LLM with natural-language interaction histories so that actions are selected through textual reasoning
rather than explicit bandit formulas, effectively replacing classical decision policies by language-based ones.
Third, some methods explicitly align LLM reasoning with posterior or confidence-based decision mechanisms. Schmied et al. [ 156]
use reinforcement-learning fine-tuning to enforce consistency between the LLM’s inferred UCB estimates and its chosen actions,
mitigating greedy bias and “knowing-but-not-doing” behavior, while Arumugam et al. [ 148] provide sampled hypothetical MDPs to
a decision LLM so that it reasons in language space about the optimal action under posterior samples, thereby realizing a PSRL-style
decision flow via LLM inference.
Despite these advances, current approaches rarely treat the bandit action decision module itself as an object of systematic optimization
with strong theoretical guarantees, and many works implicitly focus on improving exploration rather than jointly shaping exploration
and exploitation. Empirical studies also reveal instability in language-driven decision rules: for example, Hazime et al. [ 157] observe
random drift and sensitivity to complex reward structures, which can cause local fluctuations to be misinterpreted as long-term trends.
Future research may thus combine structured history summarization, constrained decision templates, and external evaluators with
chain-of-thought control, restrictive policies, or reward models, aiming to obtain more stable, theory-grounded LLM-based action
decision mechanisms that balance semantic reasoning with reliable bandit behavior.
6.7 Evaluation Issues
The reviewed papers apply LLM-enhanced bandit algorithms using both synthetic datasets, such as non-stationary and restless
bandit simulations, and real-world datasets like MovieLens and Yahoo! Front Page. Evaluation metrics focus on cumulative reward,
regret, and precision-based metrics like Precision@k (P@k) and click-through rate (CTR), commonly used in recommendation
systems. LLMs also contribute to reducing exploration costs, with metrics like time to convergence and computational efficiency
being critical in multi-task and real-time applications. Table 5 and 6 provides representative examples of datasets and evaluation
metrics commonly adopted in prior work, and is not intended to be exhaustive.
Table 5: Representative experimental datasets used in LLM-enhanced bandit studies
Dataset Type Dataset (Reference)
Synthetic Non-stationary bandit simulations [140]
Restless bandit simulations [163, 145]
Contextual bandit simulations [12]
Real-world MovieLens [159]
Yahoo! Front Page [86]
OpenAI Gym [160]
Amazon Product Data [155]
16

Table 6: Representative evaluation metrics adopted in LLM-enhanced bandit studies
Metric Representative Studies
Cumulative reward Shufaro et al. [138]; Sun et al. [151]
Regret Xia et al. [139, 154]; Park et al. [14]
Precision@k (P@k) Wang et al. [159]; Baheri et al. [12]
Click-through rate (CTR) Kiyohara et al. [86]
Time to convergence Alamdari et al. [13]
Exploration cost Guo et al. [155]; Chen et al. [15]
Computational efficiency Dai et al. [16]
Prompt–utility ratio Verma et al. [145]
Only a small number of papers in LLM-enhanced bandit research focus on providing strong theoretical guarantees for regret
minimization, particularly in non-stationary and adversarial settings. These works aim to demonstrate how LLMs can improve
exploration-exploitation trade-offs, with papers like Shufaro et al. (2024) [138] and Xia et al. (2024) [139] offering rigorous regret
upper bounds. For example, Shufaro et al. (2024) [ 138] analyze the trade-off between regret and information gain, and Xia et al.
(2024) [ 139] prove regret bounds in dueling bandit settings with LLM agents. However, the theoretical regret proofs often rely on
simplifying assumptions about the environment or the behavior of the LLMs themselves. For instance, Alamdari et al. (2024) [ 13]
leverage LLM-generated prior knowledge to reduce exploration cost, but their regret bounds depend heavily on the quality and
accuracy of the priors, which might not always hold in real-world applications.
A major limitation is the lack of formal guarantees for LLM performance in complex spaces, making regret proofs dependent
on strong assumptions like well-structured environments. Without such conditions, proving regret bounds for these algorithms
is challenging. Given these constraints, future research should prioritize practical effectiveness in real-world applications over
theoretical proofs. Researchers need to develop methods that perform well in dynamic environments, even without strong theoretical
guarantees, striking a balance between theoretical rigor and practical validation.
7 Challenges and Future Opportunities
Bandit algorithms and Large Language Models (LLMs) exhibit a mutually reinforcing relationship. On one hand, bandits offer
principled frameworks for optimizing LLM training, inference, and adaptation in dynamic environments. On the other hand, LLMs
provide rich contextual reasoning and prior knowledge that can significantly enhance the design of next-generation bandit algorithms.
This section outlines future opportunities along both directions, beginning withBandits for LLMsand followed byLLMs for Bandits.
Bandits for LLMs.Current research demonstrates that bandit algorithms are indispensable for enhancing the efficiency,
adaptability, and robustness of LLM systems. However, there are several challenges that need to be addressed in order to fully
leverage bandit algorithms in LLM systems:
•Exploration-Exploitation Balance in High-Dimensional Decision Spaces:LLMs generate outputs in complex, high-
dimensional decision spaces (e.g., word choice, sentence structure, context adaptation). Traditional bandit algorithms
often struggle to efficiently explore such spaces. Developing methods to balance exploration and exploitation in these
high-dimensional spaces remains a key challenge.
•Sparse and Noisy Feedback Signals:The feedback from LLMs is often sparse, noisy, and difficult to quantify (e.g., user
satisfaction, output relevance). Converting these unstructured feedback signals into actionable reward signals for bandit
algorithms is a major obstacle.
•Long-Term Reward Prediction and Adaptation:Many LLM tasks, such as multi-turn dialogue or multi-step text
generation, involve long-term dependencies. Bandit algorithms are typically optimized for short-term rewards, making
it difficult to account for long-term rewards and adapt to them. Developing methods to predict and optimize long-term
rewards remains a significant challenge.
•Non-Stationary Environments:LLMs, particularly in dialogue systems or multi-task learning, are highly sensitive to
contextual changes. The rewards in these environments are often non-stationary, and traditional bandit algorithms are not
designed to handle such variability effectively. Bandit algorithms need to be adapted to handle non-stationary feedback.
•Multi-Task and Multi-Objective Optimization:LLM systems often need to balance multiple tasks and objectives, such
as accuracy, diversity, and user satisfaction. Bandit algorithms must be extended to handle these complex trade-offs and
efficiently allocate resources across multiple tasks or objectives.
Several future directions can further bridge theory and practice in using bandit algorithms to enhance LLMs:
17

•Continual Learning Optimization:Use bandits to dynamically manage continual learning in LLMs, selecting the most
informative data for long-term performance. This would ensure that LLMs can efficiently adapt to new data without
catastrophic forgetting.
•Automatic Prompt Engineering:Develop bandit-driven systems that automate prompt design, optimizing LLM perfor-
mance across different tasks. This would enable adaptive prompting, improving the overall efficiency and accuracy of
LLM-based systems.
•Multi-Task and Multi-Objective Learning Bandits:Develop multi-task and multi-objective bandits that allocate
resources to optimize LLM performance across several tasks and objectives simultaneously, balancing the trade-offs
between different goals.
•LLM Compression and Efficiency:Apply bandits to intelligently explore model compression and pruning strategies,
improving computational efficiency. Bandit algorithms can help in selecting the most efficient model configurations while
maintaining high performance.
LLMs for Bandits.LLMs are increasingly powerful at understanding context, generating insightful priors, and making decisions
in dynamic environments. These capabilities offer significant potential for enhancing bandit algorithms, particularly in handling
complex environments with high-dimensional data. Several challenges and research opportunities exist in leveraging LLMs to
advance bandit algorithms:
•Adaptive Exploration with Continuous Learning:Bandit algorithms must continuously adapt to changes in the
environment. LLMs can help improve exploration-exploitation strategies by learning from feedback in real time and
adjusting exploration behaviors over time.
•LLM-Driven Multi-Modal Bandits:Traditional bandit algorithms often operate in single-modal contexts (e.g., text or
numerical data). LLMs, with their ability to process multiple types of data (e.g., text, images), can be integrated into
multi-modal bandit systems, enabling more sophisticated decision-making environments that require integrating different
data modalities.
•Human-in-the-Loop Bandits:Incorporating real-time human feedback into bandit algorithms can greatly enhance
their performance in adaptive settings. LLMs can act as mediators between the system and the human input, guiding
decision-making based on the user’s preferences or contextual changes.
•Regret-Aware Bandit Systems:LLMs can predict long-term outcomes and adjust bandit strategies accordingly, mini-
mizing regret in non-stationary and adversarial environments. This ability to incorporate sophisticated reasoning could
improve decision-making under uncertainty.
•Contextual Bandits for Dynamic Decision-Making:LLMs can leverage their contextual understanding to make more
informed decisions in dynamic, uncertain environments. Developing contextual bandits that use LLMs to adaptively select
actions based on evolving inputs (e.g., user interactions, environmental changes) will significantly enhance the flexibility
and robustness of decision-making systems.
As bandit algorithms become increasingly integrated with large language models, developing rigorous theoretical guarantees will
become substantially more challenging due to the high complexity, non-linearity, and implicit reasoning structures of LLMs.
Nevertheless, future research may shift toward evaluating such hybrid bandit–LLM systems primarily through their empirical
performance in real-world applications, rather than relying solely on traditional theoretical analyses. This shift could drive the
practical deployment of these systems in a variety of complex, real-world environments.
8 Conclusion
To the best of our knowledge, this work represents the first systematic review that analyzes the LLM–MAB intersection from a
component-based and algorithmic perspective. Through a systematic breakdown of key components, we demonstrate how LLMs
enhance the performance of bandit algorithms and how bandit frameworks, in turn, optimize various LLM tasks. This survey
establishes a solid foundation for future research, highlighting the importance of application-driven studies that strike a balance
between theoretical rigor and practical impact. We hope this work will inspire and guide future advancements in this rapidly evolving
field.
References
[1]Jia Li, Chongyang Tao, Jia Li, Ge Li, Zhi Jin, Huangzhao Zhang, Zheng Fang, and Fang Liu. Large language model-aware
in-context learning for code generation.ACM Transactions on Software Engineering and Methodology, 34(7):1–33, 2025.
[2]Xin Luna Dong, Seungwhan Moon, Yifan Ethan Xu, Kshitiz Malik, and Zhou Yu. Towards next-generation intelligent
assistants leveraging llm techniques. InProceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and
Data Mining, pages 5792–5793, 2023.
[3] A Jo. The promise and peril of generative ai.Nature, 614(1):214–216, 2023.
18

[4]Muning Wen, Runji Lin, Hanjing Wang, Yaodong Yang, Ying Wen, Luo Mai, Jun Wang, Haifeng Zhang, and Weinan Zhang.
Large sequence models for sequential decision-making: a survey.Frontiers of Computer Science, 17(6):176349, 2023.
[5]Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Naveed Akhtar, Nick
Barnes, and Ajmal Mian. A comprehensive overview of large language models.ACM Trans. Intell. Syst. Technol., 16(5),
August 2025.
[6]Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction 2nd ed.MIT press Cambridge, 1(2):25,
2018.
[7] Tor Lattimore and Csaba Szepesvári.Bandit algorithms. Cambridge University Press, 2020.
[8]Bo Liu, Ying Wei, Yu Zhang, Zhixian Yan, and Qiang Yang. Transferable contextual bandit for cross-domain recommendation.
InProceedings of the AAAI conference on artificial intelligence, volume 32, 2018.
[9]Tong Geng, Xiliang Lin, Harikesh S Nair, Jun Hao, Bin Xiang, and Shurui Fan. Comparison lift: Bandit-based experimentation
system for online advertising. InProceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 15117–
15126, 2021.
[10] Yao Zhao, Kwang-Sung Jun, Tanner Fiez, and Lalit Jain. Adaptive experimentation when you can’t experiment.Advances in
Neural Information Processing Systems, 37:121928–121991, 2024.
[11] Djallel Bouneffouf, Irina Rish, and Charu Aggarwal. Survey on applications of multi-armed and contextual bandits. In2020
IEEE Congress on Evolutionary Computation (CEC), pages 1–8. IEEE, 2020.
[12] Ali Baheri and Cecilia O Alm. Llms-augmented contextual bandit.arXiv preprint arXiv:2311.02268, 2023.
[13] Parand Alamdari, Yanshuai Cao, and Kevin Wilson. Jump starting bandits with llm-generated prior knowledge. InProceedings
of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 19821–19833, 2024.
[14] Chanwoo Park, Xiangyu Liu, Asuman Ozdaglar, and Kaiqing Zhang. Do llm agents have regret? a case study in online
learning and games.arXiv preprint arXiv:2403.16843, 2024.
[15] Dingyang Chen, Qi Zhang, and Yinglun Zhu. Efficient sequential decision making with large language models. InProceedings
of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 9157–9170, 2024.
[16] Xiangxiang Dai, Jin Li, Xutong Liu, Anqi Yu, and John Lui. Cost-effective online multi-llm selection with versatile reward
models.arXiv preprint arXiv:2405.16587, 2024.
[17] Guoxin Chen, Zhong Zhang, Xin Cong, Fangda Guo, Yesai Wu, Yankai Lin, Wenzheng Feng, and Yasheng Wang. Learning
evolving tools for large language models. InThe Thirteenth International Conference on Learning Representations, 2025.
[18] Shameem A Puthiya Parambath, Christos Anagnostopoulos, and Roderick Murray-Smith. Sequential query prediction
based on multi-armed bandits with ensemble of transformer experts and immediate feedback.Data Mining and Knowledge
Discovery, pages 1–25, 2024.
[19] Baran Atalar, Eddie Zhang, and Carlee Joe-Wong. Neural bandit based optimal llm selection for a pipeline of tasks.arXiv
preprint arXiv:2508.09958, 2025.
[20] John Mui, Fuhua Lin, and M Ali Akber Dewan. Multi-armed bandit algorithms for adaptive learning: a survey. InInternational
Conference on Artificial Intelligence in Education, pages 273–278. Springer, 2021.
[21] Baihan Lin. Reinforcement learning and bandits for speech and language processing: Tutorial, review and outlook.Expert
Systems with Applications, page 122254, 2023.
[22] Jelena Luketina, Nantas Nardelli, Gregory Farquhar, Jakob Foerster, Jacob Andreas, Edward Grefenstette, Shimon Whiteson,
and Tim Rocktäschel. A survey of reinforcement learning informed by natural language.arXiv preprint arXiv:1906.03926,
2019.
[23] Djallel Bouneffouf and Raphael Feraud. Survey: Multi-armed bandits meet large language models, 2025.
[24] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang,
Zican Dong, et al. A survey of large language models.arXiv preprint arXiv:2303.18223, 1(2), 2023.
[25] Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, and Jianfeng Gao.
Large language models: A survey.arXiv preprint arXiv:2402.06196, 2024.
[26] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia
Polosukhin. Attention is all you need.Advances in neural information processing systems, 30, 2017.
[27] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers
for language understanding. InProceedings of the 2019 conference of the North American chapter of the association for
computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019.
[28] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised
multitask learners.OpenAI blog, 1(8):9, 2019.
[29] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J
Liu. Exploring the limits of transfer learning with a unified text-to-text transformer.Journal of machine learning research,
21(140):1–67, 2020.
19

[30] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav
Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners.Advances in neural information
processing systems, 33:1877–1901, 2020.
[31] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham,
Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways.Journal of
Machine Learning Research, 24(240):1–113, 2023.
[32] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière,
Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models.arXiv preprint
arXiv:2302.13971, 2023.
[33] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford,
Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models.arXiv preprint arXiv:2001.08361, 2020.
[34] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas,
Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche,
Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Oriol Vinyals, Jack W. Rae, and Laurent Sifre.
Training compute-optimal large language models. InProceedings of the 36th International Conference on Neural Information
Processing Systems, NIPS ’22, Red Hook, NY , USA, 2022. Curran Associates Inc.
[35] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko
Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report.arXiv preprint arXiv:2303.08774, 2023.
[36] An Zhang, Yang Deng, Yankai Lin, Xu Chen, Ji-Rong Wen, and Tat-Seng Chua. Large language model powered agents for
information retrieval. InProceedings of the 47th International ACM SIGIR Conference on Research and Development in
Information Retrieval, pages 2989–2992, 2024.
[37] Sghaier Guizani, Tehseen Mazhar, Tariq Shahzad, Wasim Ahmad, Afsha Bibi, and Habib Hamam. A systematic literature
review to implement large language model in higher education: issues and solutions.Discover Education, 4(1):1–25, 2025.
[38] Yifei Dong, Fengyi Wu, Kunlin Zhang, Yilong Dai, Sanjian Zhang, Wanghao Ye, Sihan Chen, and Zhi-Qi Cheng. Large
language model agents in finance: A survey bridging research, practice, and real-world deployment. InFindings of the
Association for Computational Linguistics: EMNLP 2025, pages 17889–17907, 2025.
[39] Usman Iqbal, Afifa Tanweer, Annisa Ristya Rahmanti, David Greenfield, Leon Tsung-Ju Lee, and Yu-Chuan Jack Li. Impact
of large language model (chatgpt) in healthcare: an umbrella review and evidence synthesis.Journal of Biomedical Science,
32(1):45, 2025.
[40] Aleksandrs Slivkins et al. Introduction to multi-armed bandits.Foundations and Trends® in Machine Learning, 12(1-2):1–286,
2019.
[41] Lihong Li, Wei Chu, John Langford, and Robert E Schapire. A contextual-bandit approach to personalized news article
recommendation.Proceedings of the 19th international conference on World wide web, pages 661–670, 2010.
[42] Vianney Perchet and Philippe Rigollet. The multi-armed bandit problem with covariates.arXiv preprint arXiv:1302.6013,
2013.
[43] Yi Peng, Miao Xie, Jiahao Liu, Xuying Meng, Nan Li, Cheng Yang, Tao Yao, and Rong Jin. A practical semi-parametric
contextual bandit. InIJCAI, pages 3246–3252, 2019.
[44] Young-Geun Choi, Gi-Soo Kim, Seunghoon Paik, and Myunghee Cho Paik. Semi-parametric contextual bandits with
graph-laplacian regularization.Information Sciences, 645:119367, 2023.
[45] Zhengyuan Zhou, Haohan Zhou, and Lihong Li. Neural contextual bandits with ucb-based exploration.Proceedings of the
37th International Conference on Machine Learning, pages 11492–11502, 2020.
[46] Yang Chen, Miao Xie, Jiamou Liu, and Kaiqi Zhao. Interconnected neural linear contextual bandits with ucb exploration. In
Pacific-Asia Conference on Knowledge Discovery and Data Mining, pages 169–181. Springer, 2022.
[47] Omar Besbes, Yishay Gur, and Assaf Zeevi. Stochastic multi-armed-bandit problem with non-stationary rewards.Advances
in neural information processing systems, 27, 2014.
[48] Peter Auer, Nicolo Cesa-Bianchi, Yoav Freund, and Robert E Schapire. The nonstochastic multiarmed bandit problem.SIAM
journal on computing, 32(1):48–77, 2002.
[49] Miao Xie, Wotao Yin, and Huan Xu. Autobandit: A meta bandit online learning system. InIJCAI, pages 5028–5031, 2021.
[50] Edoardo Aromataris and Alan Pearson. The systematic review: an overview.AJN The American Journal of Nursing,
114(3):53–58, 2014.
[51] Alex Pollock and Eivind Berge. How to do a systematic review.International Journal of Stroke, 13(2):138–156, 2018.
[52] Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang,
Guoyin Wang, and Fei Wu. Instruction tuning for large language models: A survey.ACM Comput. Surv., November 2025.
Just Accepted.
[53] Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Hongyi Jin, Tianqi Chen, and Zhihao Jia. Towards efficient
generative large language model serving: A survey from algorithms to systems.ACM Comput. Surv., 58(1), September 2025.
20

[54] Jiahong Liu, Zexuan Qiu, Zhongyang Li, Quanyu Dai, Wenhao Yu, Jieming Zhu, Minda Hu, Menglin Yang, Tat-Seng
Chua, and Irwin King. A survey of personalized large language models: Progress and future directions.arXiv preprint
arXiv:2502.11528, 2025.
[55] Aske Plaat, Annie Wong, Suzan Verberne, Joost Broekens, Niki Van Stein, and Thomas Bäck. Multi-step reasoning with large
language models, a survey.ACM Comput. Surv., 58(6), December 2025.
[56] P Auer. Finite-time analysis of the multiarmed bandit problem, 2002.
[57] Subhojyoti Mukherjee, Josiah P Hanna, Qiaomin Xie, and Robert Nowak. Pretraining decision transformers with reward
prediction for in-context multi-task structured bandit learning.arXiv preprint arXiv:2406.05064, 2024.
[58] Iñigo Urteaga, Moulay-Zaïdane Draïdia, Tomer Lancewicki, and Shahram Khadivi. Multi-armed bandits for resource efficient,
online optimization of language model pre-training: the use case of dynamic masking.arXiv preprint arXiv:2203.13151,
2022.
[59] Alon Albalak, Liangming Pan, Colin Raffel, and William Yang Wang. Efficient online data mixing for language model
pre-training. InR0-FoMo: Robustness of Few-shot and Zero-shot Learning in Large Foundation Models, 2023.
[60] Chi Zhang, Huaping Zhong, Kuan Zhang, Chengliang Chai, Rui Wang, Xinlin Zhuang, Tianyi Bai, Jiantao Qiu, Lei Cao,
Ye Yuan, et al. Harnessing diversity for important data selection in pretraining large language models.arXiv preprint
arXiv:2409.16986, 2024.
[61] Jing Ma, Chenhao Dang, and Mingjie Liao. Actor-critic based online data mixing for language model pre-training, 2025.
[62] Tong Zhu, Daize Dong, Xiaoye Qu, Jiacheng Ruan, Wenliang Chen, and Yu Cheng. Dynamic data mixing maximizes
instruction tuning for mixture-of-experts.arXiv preprint arXiv:2406.11256, 2024.
[63] Banghua Zhu, Michael I Jordan, and Jiantao Jiao. Iterative data smoothing: Mitigating reward overfitting and overoptimization
in rlhf.arXiv preprint arXiv:2401.16335, 2024.
[64] Runlong Zhou, Simon S Du, and Beibin Li. Reflect-rl: Two-player online rl fine-tuning for lms.arXiv preprint
arXiv:2402.12621, 2024.
[65] Heyang Zhao, Chenlu Ye, Quanquan Gu, and Tong Zhang. Sharp analysis for kl-regularized contextual bandits and rlhf. In
NeurIPS 2024 Workshop on Fine-Tuning in Modern Machine Learning: Principles and Scalability, 2024.
[66] Shaohua Duan, Xinze Li, Zhenghao Liu, Xiaoyuan Yi, Yukun Yan, Shuo Wang, Yu Gu, Ge Yu, and Maosong Sun. Chunks as
arms: Multi-armed bandit-guided sampling for long-context llm preference optimization.arXiv preprint arXiv:2508.13993,
2025.
[67] Zichen Liu, Changyu Chen, Chao Du, Wee Sun Lee, and Min Lin. Sample-efficient alignment for llms.arXiv preprint
arXiv:2411.01493, 2024.
[68] Fahim Tajwar, Anikait Singh, Archit Sharma, Rafael Rafailov, Jeff Schneider, Tengyang Xie, Stefano Ermon, Chelsea Finn,
and Aviral Kumar. Preference fine-tuning of llms should leverage suboptimal, on-policy data. InProceedings of the 41st
International Conference on Machine Learning, pages 47441–47474, 2024.
[69] Haebin Shin, Lei Ji, Xiao Liu, Zhiwei Yu, Qi Chen, and Yeyun Gong. Dynamixsft: Dynamic mixture optimization of
instruction tuning collections.arXiv preprint arXiv:2508.12116, 2025.
[70] Yu Xia, Fang Kong, Tong Yu, Liya Guo, Ryan A Rossi, Sungchul Kim, and Shuai Li. Which llm to play? convergence-aware
online model selection with time-increasing bandits. InProceedings of the ACM on Web Conference 2024, pages 4059–4070,
2024.
[71] Yu Xia, Fang Kong, Tong Yu, Liya Guo, Ryan A Rossi, Sungchul Kim, and Shuai Li. Convergence-aware online model
selection with time-increasing bandits. InThe Web Conference 2024, 2024.
[72] Zichen Liu, Changyu Chen, Chao Du, Wee Sun Lee, and Min Lin. Sample-efficient alignment for LLMs. InLanguage
Gamification - NeurIPS 2024 Workshop, 2024.
[73] Viraj Mehta, Syrine Belakaria, Vikramjeet Das, Ojash Neopane, Yijia Dai, Ilija Bogunovic, Barbara E Engelhardt, Stefano
Ermon, Jeff Schneider, and Willie Neiswanger. Sample efficient preference alignment in LLMs via active exploration. In
Second Conference on Language Modeling, 2025.
[74] Xiaoqiang Lin, Arun Verma, Zhongxiang Dai, Daniela Rus, See-Kiong Ng, and Bryan Kian Hsiang Low. Activedpo: Active
direct preference optimization for sample-efficient alignment.arXiv preprint arXiv:2505.19241, 2025.
[75] Chenjia Bai, Yang Zhang, Shuang Qiu, Qiaosheng Zhang, Kang Xu, and Xuelong Li. Online preference alignment for
language models via count-based exploration. InThe Thirteenth International Conference on Learning Representations, 2025.
[76] Daniele Calandriello, Zhaohan Daniel Guo, Remi Munos, Mark Rowland, Yunhao Tang, Bernardo Avila Pires, Pierre Harvey
Richemond, Charline Le Lan, Michal Valko, Tianqi Liu, Rishabh Joshi, Zeyu Zheng, and Bilal Piot. Human alignment
of large language models through online preference optimisation. InProceedings of the 41st International Conference on
Machine Learning, ICML’24. JMLR.org, 2024.
[77] Rémi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Zhaohan Daniel Guo,
Yunhao Tang, Matthieu Geist, Thomas Mesnard, Côme Fiegel, et al. Nash learning from human feedback. InForty-first
International Conference on Machine Learning, 2024.
21

[78] Chengshuai Shi, Kun Yang, Jing Yang, and Cong Shen. Best arm identification for prompt learning under a limited budget.
arXiv preprint arXiv:2402.09723, 2024.
[79] Chengshuai Shi, Kun Yang, Zihan Chen, Jundong Li, Jing Yang, and Cong Shen. Efficient prompt optimization through the
lens of best arm identification.Advances in Neural Information Processing Systems, 37:99646–99685, 2024.
[80] Rin Ashizawa, Yoichi Hirose, Nozomu Yoshinari, Kento Uchida, and Shinichi Shirakawa. Bandit-based prompt design
strategy selection improves prompt optimizers.arXiv preprint arXiv:2503.01163, 2025.
[81] Xiaoqiang Lin, Zhongxiang Dai, Arun Verma, See-Kiong Ng, Patrick Jaillet, and Bryan Kian Hsiang Low. Prompt optimization
with human feedback.arXiv preprint arXiv:2405.17346, 2024.
[82] Yuanchen Wu, Saurabh Verma, Justin Lee, Fangzhou Xiong, Poppy Zhang, Amel Awadelkarim, Xu Chen, Yubai Yuan, and
Shawndra Hill. Llm prompt duel optimizer: Efficient label-free prompt optimization.arXiv preprint arXiv:2510.13907, 2025.
[83] Zhaoxuan Wu, Xiaoqiang Lin, Zhongxiang Dai, Wenyang Hu, Yao Shu, See-Kiong Ng, Patrick Jaillet, and Bryan Kian Hsiang
Low. Prompt optimization with ease? efficient ordering-aware automated selection of exemplars.Advances in Neural
Information Processing Systems, 37:122706–122740, 2024.
[84] Finn Rietz, Oleg Smirnov, Sara Karimi, and Lele Cao. Prompt tuning decision transformers with structured and scalable
bandits, 2025.
[85] Mingze Kong, Zhiyong Wang, Yao Shu, and Zhongxiang Dai. Meta-prompt optimization for llm-based sequential decision
making.arXiv preprint arXiv:2502.00728, 2025.
[86] Haruka Kiyohara, Yuta Saito, Daniel Yiming Cao, and Thorsten Joachims. Prompt optimization with logged bandit data. In
ICLR 2024 Workshop on Navigating and Addressing Data Problems for Foundation Models, 2024.
[87] Haruka Kiyohara, Daniel Yiming Cao, Yuta Saito, and Thorsten Joachims. Prompt optimization with logged bandit data.
arXiv preprint arXiv:2504.02646, 2025.
[88] Hanzhuo Tan, Qi Luo, Ling Jiang, Zizheng Zhan, Jing Li, Haotian Zhang, and Yuqun Zhang. Prompt-based code completion
via multi-retrieval augmented generation.arXiv preprint arXiv:2405.07530, 2024.
[89] Xiaoqiang Lin, Zhaoxuan Wu, Zhongxiang Dai, Wenyang Hu, Yao Shu, See-Kiong Ng, Patrick Jaillet, and Bryan Kian Hsiang
Low. Use your instinct: Instruction optimization for llms using neural bandits coupled with transformers. InInternational
Conference on Machine Learning, pages 30317–30345. PMLR, 2024.
[90] Pingchen Lu, Zhi Hong, Zhiwei Shang, Zhiyong Wang, Yikun Ban, Yao Shu, Min Zhang, Shuang Qiu, and Zhongxiang Dai.
Fedpob: Sample-efficient federated prompt optimization via bandits, 2025.
[91] Siliang Zeng, Quan Wei, William Brown, Oana Frunza, Yuriy Nevmyvaka, Yang Katie Zhao, and Mingyi Hong. Reinforcing
multi-turn reasoning in llm agents via turn-level credit assignment. InICML 2025 Workshop on Computer Use Agents, 2025.
[92] Sijia Chen, Yibo Wang, Yi-Feng Wu, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and Lijun Zhang. Advancing
tool-augmented large language models: Integrating insights from errors in inference trees. In A. Globerson, L. Mackey,
D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors,Advances in Neural Information Processing Systems,
volume 37, pages 106555–106581. Curran Associates, Inc., 2024.
[93] Robert Müller. Semantic context for tool orchestration.arXiv preprint arXiv:2507.10820, 2025.
[94] Dongfu Jiang, Yi Lu, Zhuofeng Li, Zhiheng Lyu, Ping Nie, Haozhe Wang, Alex Su, Hui Chen, Kai Zou, Chao Du, et al.
Verltool: Towards holistic agentic reinforcement learning with tool use.arXiv preprint arXiv:2509.01055, 2025.
[95] Yunpeng Zhai, Shuchang Tao, Cheng Chen, Anni Zou, Ziqian Chen, Qingxu Fu, Shinji Mai, Li Yu, Jiaji Deng, Zouying Cao,
et al. Agentevolver: Towards efficient self-evolving agent system.arXiv preprint arXiv:2511.10395, 2025.
[96] Akshay Krishnamurthy, Keegan Harris, Dylan J Foster, Cyril Zhang, and Aleksandrs Slivkins. Can large language models
explore in-context?arXiv preprint arXiv:2403.15371, 2024.
[97] Nate Rahn, Pierluca D’Oro, and Marc G Bellemare. Controlling large language model agents with entropic activation steering.
arXiv preprint arXiv:2406.00244, 2024.
[98] Vikranth Dwaracherla, Seyed Mohammad Asghari, Botao Hao, and Benjamin Van Roy. Efficient exploration for llms.arXiv
preprint arXiv:2402.00396, 2024.
[99] Hao Tang, Keya Hu, Jin Peng Zhou, Sicheng Zhong, Wei-Long Zheng, Xujie Si, and Kevin Ellis. Code repair with llms gives
an exploration-exploitation tradeoff.arXiv preprint arXiv:2405.17503, 2024.
[100] Deng Pan, Keerthiram Murugesan, Nuno Moniz, and Nitesh Chawla. Context attribution with multi-armed bandit optimization.
arXiv preprint arXiv:2506.19977, 2025.
[101] Weicong Qin, Yi Xu, Weijie Yu, Chenglei Shen, Xiao Zhang, Ming He, Jianping Fan, and Jun Xu. Enhancing sequential
recommendations through multi-perspective reflections and iteration.arXiv preprint arXiv:2409.06377, 2024.
[102] Allen Nie, Yi Su, Bo Chang, Jonathan Lee, Ed H. Chi, Quoc V Le, and Minmin Chen. Evolve: Evaluating and optimizing
llms for in-context exploration. InForty-second International Conference on Machine Learning, 2025.
[103] Manhin Poon, XiangXiang Dai, Xutong Liu, Fang Kong, John Lui, and Jinhang Zuo. Online multi-llm selection via contextual
bandits under unstructured context evolution.arXiv preprint arXiv:2506.17670, 2025.
22

[104] Aditya Ramesh, Shivam Bhardwaj, Aditya Saibewar, and Manohar Kaul. Efficient jailbreak attack sequences on large
language models via multi-armed bandit-based context switching. InThe Thirteenth International Conference on Learning
Representations, 2025.
[105] Xiaqiang Tang, Qiang Gao, Jian Li, Nan Du, Qi Li, and Sihong Xie. Mba-rag: a bandit approach for adaptive retrieval-
augmented generation through question complexity, 2025.
[106] Tao Ouyang, Guihang Hong, Kongyange Zhao, Zhi Zhou, Weigang Wu, Zhaobiao Lv, and Xu Chen. Adarag: Adaptive
optimization for retrieval augmented generation with multilevel retrievers at the edge. InIEEE INFOCOM 2025 - IEEE
Conference on Computer Communications, pages 1–10, 2025.
[107] Zheng Wang, Shu Xian Teo, Jieer Ouyang, Yongjun Xu, and Wei Shi. M-rag: Reinforcing large language model performance
through retrieval-augmented generation with multiple partitions, 2024.
[108] Qinggang Zhang, Junnan Dong, Hao Chen, Daochen Zha, Zailiang Yu, and Xiao Huang. Knowgpt: Knowledge graph based
prompting for large language models. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang,
editors,Advances in Neural Information Processing Systems, volume 37, pages 6052–6080. Curran Associates, Inc., 2024.
[109] Xiaqiang Tang, Jian Li, Nan Du, and Sihong Xie. Adapting to non-stationary environments: Multi-armed bandit enhanced
retrieval-augmented generation on knowledge graphs. InProceedings of the AAAI Conference on Artificial Intelligence,
volume 39, pages 12658–12666, 2025.
[110] Jia Fu, Xiaoting Qin, Fangkai Yang, Lu Wang, Jue Zhang, Qingwei Lin, Yubo Chen, Dongmei Zhang, Saravan Rajmohan, and
Qi Zhang. Autorag-hp: Automatic online hyper-parameter tuning for retrieval-augmented generation, 2024.
[111] Yang Li. Llm bandit: Cost-efficient llm generation via preference-conditioned dynamic routing, 2025.
[112] Pranoy Panda, Raghav Magazine, Chaitanya Devaguptapu, Sho Takemori, and Vishal Sharma. Adaptive llm routing under
budget constraints.arXiv preprint arXiv:2508.21141, 2025.
[113] Ninad Tongay.Dynamic and Cost-Efficient Deployment of Large Language Models Using Uplift Modeling and Multi Armed
Bandits. PhD thesis, 2025.
[114] Wang Wei, Tiankai Yang, Hongjie Chen, Yue Zhao, Franck Dernoncourt, Ryan A. Rossi, and Hoda Eldardiry. Learning to
route llms from bandit feedback: One policy, many trade-offs, 2025.
[115] Hantao Yang, Hong Xie, Defu Lian, and Enhong Chen. Llm cache bandit revisited: Addressing query heterogeneity for
cost-effective llm inference.arXiv preprint arXiv:2509.15515, 2025.
[116] Yunlong Hou, Fengzhuo Zhang, Cunxiao Du, Xuan Zhang, Jiachun Pan, Tianyu Pang, Chao Du, Vincent YF Tan, and Zhuoran
Yang. Banditspec: Adaptive speculative decoding via bandit algorithms.arXiv preprint arXiv:2505.15141, 2025.
[117] Jiahao Liu, Qifan Wang, Jingang Wang, and Xunliang Cai. Speculative decoding via early-exiting for faster llm inference
with thompson sampling control mechanism.arXiv preprint arXiv:2406.03853, 2024.
[118] Hongyi Liu, Jiaji Huang, Zhen Jia, Youngsuk Park, and Yu-Xiang Wang. Not-a-bandit: Provably no-regret drafter selection in
speculative decoding for llms.arXiv preprint arXiv:2510.20064, 2025.
[119] Suho Shin, Chenghao Yang, Haifeng Xu, and Mohammad T Hajiaghayi. Tokenized bandit for llm decoding and alignment.
arXiv preprint arXiv:2506.07276, 2025.
[120] Zekai Chen, Po-Yu Chen, and Francois Buet-Golfouse. Online personalizing white-box llms generation with neural bandits.
InProceedings of the 5th ACM International Conference on AI in Finance, pages 711–718, 2024.
[121] Kaan Gönç, Baturay Sa ˘glam, Onat Dalmaz, Tolga Çukur, Serdar Kozat, and Hamdi Dibeklioglu. User feedback-based online
learning for intent classification. InProceedings of the 25th International Conference on Multimodal Interaction, pages
613–621, 2023.
[122] Giovanni Monea, Antoine Bosselut, Kianté Brantley, and Yoav Artzi. Llms are in-context bandit reinforcement learners.
arXiv preprint arXiv:2410.05362, 2024.
[123] Fabian Moerchen, Patrick Ernst, and Giovanni Zappella. Personalizing natural language understanding using multi-armed
bandits and implicit feedback. InProceedings of the 29th ACM international conference on information & knowledge
management, pages 2661–2668, 2020.
[124] Ziyuan Zhang, Darcy Wang, Ningyuan Chen, Rodrigo Mansur, and Vahid Sarhangian. Comparing exploration-exploitation
strategies of llms and humans: Insights from standard multi-armed bandit tasks.arXiv preprint arXiv:2505.09901, 2025.
[125] Ye Kyaw Thu, Thazin Myint Oo, and Thepchai Supnithi. Rl-nmt: Reinforcement learning fine-tuning for improved neural
machine translation of burmese dialects. InProceedings of the 5th ACM International Conference on Multimedia in Asia
Workshops, pages 1–8, 2023.
[126] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep
Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback.
arXiv preprint arXiv:2204.05862, 2022.
[127] Yufei Wang, Wanjun Zhong, Liangyou Li, Fei Mi, Xingshan Zeng, Wenyong Huang, Lifeng Shang, Xin Jiang, and Qun Liu.
Aligning large language models with human: A survey.arXiv preprint arXiv:2307.12966, 2023.
23

[128] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal,
Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell,
Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human
feedback. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors,Advances in Neural Information
Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc., 2022.
[129] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference
optimization: Your language model is secretly a reward model. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and
S. Levine, editors,Advances in Neural Information Processing Systems, volume 36, pages 53728–53741. Curran Associates,
Inc., 2023.
[130] Remi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Zhaohan Daniel Guo,
Yunhao Tang, Matthieu Geist, Thomas Mesnard, Côme Fiegel, Andrea Michi, Marco Selvi, Sertan Girgin, Nikola Momchev,
Olivier Bachem, Daniel J Mankowitz, Doina Precup, and Bilal Piot. Nash learning from human feedback. In Ruslan
Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors,
Proceedings of the 41st International Conference on Machine Learning, volume 235 ofProceedings of Machine Learning
Research, pages 36743–36768. PMLR, 21–27 Jul 2024.
[131] Kexin Huang, Junkang Wu, Ziqian Chen, Xue Wang, Jinyang Gao, Bolin Ding, Jiancan Wu, Xiangnan He, and Xiang Wang.
Larger or smaller reward margins to select preferences for LLM alignment? InForty-second International Conference on
Machine Learning, 2025.
[132] Xiangling Yang. Multi-armed bandit algorithms for large language model optimization: A survey of theory and applications.
InITM Web of Conferences, volume 80, page 02006. EDP Sciences, 2025.
[133] Aske Plaat, Max van Duijn, Niki van Stein, Mike Preuss, Peter van der Putten, and Kees Joost Batenburg. Agentic large
language models, a survey.arXiv preprint arXiv:2503.23037, 2025.
[134] Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, and Ji-Rong Wen. Tool learning
with large language models: A survey.Frontiers of Computer Science, 19(8):198343, 2025.
[135] Duo Wu, Jinghe Wang, Yuan Meng, Yanning Zhang, Le Sun, and Zhi Wang. Catp-llm: Empowering large language models
for cost-aware tool planning. InProceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages
8699–8709, October 2025.
[136] Zixuan Zhou, Xuefei Ning, Ke Hong, Tianyu Fu, Jiaming Xu, Shiyao Li, Yuming Lou, Luning Wang, Zhihang Yuan, Xiuhong
Li, Shengen Yan, Guohao Dai, Xiao-Ping Zhang, Yuhan Dong, and Yu Wang. A survey on efficient inference for large
language models, 2024.
[137] Haojun Xia, Zhen Zheng, Yuchao Li, Donglin Zhuang, Zhongzhu Zhou, Xiafei Qiu, Yong Li, Wei Lin, and Shuaiwen Leon
Song. Flash-llm: Enabling cost-effective and highly-efficient large generative model inference with unstructured sparsity.
Proc. VLDB Endow., 17(2):211–224, October 2023.
[138] Itai Shufaro, Nadav Merlis, Nir Weinberger, and Shie Mannor. On bits and bandits: Quantifying the regret-information
trade-off.arXiv preprint arXiv:2405.16581, 2024.
[139] Fanzeng Xia, Hao Liu, Yisong Yue, and Tongxin Li. Beyond numeric awards: In-context dueling bandits with llm agents.
arXiv preprint arXiv:2407.01887, 2024.
[140] J De Curtò, Irene de Zarzà, Gemma Roig, Juan Carlos Cano, Pietro Manzoni, and Carlos T Calafate. Llm-informed
multi-armed bandit strategies for non-stationary environments.Electronics, 12(13):2814, 2023.
[141] Arun Verma, Zhongxiang Dai, Xiaoqiang Lin, Patrick Jaillet, and Bryan Kian Hsiang Low. Neural dueling bandits: Preference-
based optimization with human feedback. InThe Thirteenth International Conference on Learning Representations, 2025.
[142] Yuqing Du, Olivia Watkins, Zihan Wang, Cédric Colas, Trevor Darrell, Pieter Abbeel, Abhishek Gupta, and Jacob Andreas.
Guiding pretraining in reinforcement learning with large language models. InInternational Conference on Machine Learning,
pages 8657–8677. PMLR, 2023.
[143] Haochen Song, Dominik Hofer, Rania Islambouli, Laura Hawkins, Ananya Bhattacharjee, Meredith Franklin, and Joseph Jay
Williams. Investigating the relationship between physical activity and tailored behavior change messaging: Connecting
contextual bandit with large language models.arXiv preprint arXiv:2506.07275, 2025.
[144] Abdulrahman Kerim, Leandro Soriano Marcolino, Erickson R. Nascimento, and Richard Jiang. Multi-armed bandit approach
for optimizing training on synthetic data, 2024.
[145] Shresth Verma, Niclas Boehmer, Lingkai Kong, and Milind Tambe. Balancing act: prioritization strategies for llm-designed
restless bandit rewards. InInternational Conference on Game Theory and AI for Security, pages 376–394. Springer, 2025.
[146] Leonard Salewski, Stephan Alaniz, Isabel Rio-Torto, Eric Schulz, and Zeynep Akata. In-context impersonation reveals large
language models’ strengths and biases.Advances in Neural Information Processing Systems, 36, 2024.
[147] Abhay Zala, Jaemin Cho, Han Lin, Jaehong Yoon, and Mohit Bansal. Envgen: Generating and adapting environments via
llms for training embodied agents.arXiv preprint arXiv:2403.12014, 2024.
[148] Dilip Arumugam and Thomas L Griffiths. Toward efficient exploration by large language model agents.arXiv preprint
arXiv:2504.20997, 2025.
24

[149] Saman Nessari and Ali Bozorgi-Amiri. Prior-informed optimization of treatment recommendation via bandit algorithms
trained on large language model-processed historical records.arXiv preprint arXiv:2510.19014, 2025.
[150] Nicolò Felicioni, Lucas Maystre, Sina Ghiassian, and Kamil Ciosek. On the importance of uncertainty in decision-making
with large language models.arXiv preprint arXiv:2404.02649, 2024.
[151] Jiahang Sun, Zhiyong Wang, Runhan Yang, Chenjun Xiao, John C. S. Lui, and Zhongxiang Dai. Large language model-
enhanced multi-armed bandits, 2025.
[152] Nikhil Behari, Edwin Zhang, Yunfan Zhao, Aparna Taneja, Dheeraj Nagaraj, and Milind Tambe. A decision-language model
(dlm) for dynamic restless multi-armed bandit tasks in public health. InProceedings of the 38th International Conference on
Neural Information Processing Systems, pages 3964–4002, 2024.
[153] Sanxing Chen, Xiaoyin Chen, Yukun Huang, Roy Xie, and Bhuwan Dhingra. When greedy wins: Emergent exploitation bias
in meta-bandit llm training.arXiv preprint arXiv:2509.24923, 2025.
[154] Fanzeng Xia, Hao Liu, Yisong Yue, and Tongxin Li. Beyond numeric rewards: In-context dueling bandits with llm agents. In
Findings of the Association for Computational Linguistics: ACL 2025, pages 9959–9988, 2025.
[155] Pei-Fu Guo, Ying-Hsuan Chen, Yun-Da Tsai, and Shou-De Lin. Towards optimizing with large language models.arXiv
preprint arXiv:2310.05204, 2023.
[156] Thomas Schmied, Jörg Bornschein, Jordi Grau-Moya, Markus Wulfmeier, and Razvan Pascanu. Llms are greedy agents:
Effects of rl fine-tuning on decision-making abilities.arXiv preprint arXiv:2504.16078, 2025.
[157] Jawad Hazime and Junaid Farooq. Evaluation of llm powered agentic ai for solving multi-arm bandit problems. In2025 IEEE
International Conference on Omni-layer Intelligent Systems (COINS), pages 1–6. IEEE, 2025.
[158] Jimin Lim, Arjun Damerla, Arthur Jiang, and Nam Le. Textbandit: Evaluating probabilistic reasoning in llms through
language-only decision tasks.arXiv preprint arXiv:2510.13878, 2025.
[159] Jianling Wang, Haokai Lu, Yifan Liu, He Ma, Yueqi Wang, Yang Gu, Shuzhou Zhang, Shuchao Bi, Lexi Baugher, Ed Chi,
et al. Llms for user interest exploration: A hybrid approach.arXiv preprint arXiv:2405.16363, 2024.
[160] Shengcai Liu, Caishun Chen, Xinghua Qu, Ke Tang, and Yew-Soon Ong. Large language models as evolutionary optimizers.
In2024 IEEE Congress on Evolutionary Computation (CEC), pages 1–8. IEEE, 2024.
[161] Michael R Zhang, Nishkrit Desai, Juhan Bae, Jonathan Lorraine, and Jimmy Ba. Using large language models for hyperpa-
rameter optimization. InNeurIPS 2023 Foundation Models for Decision Making Workshop, 2023.
[162] Sindhu Padakandla. A survey of reinforcement learning algorithms for dynamically varying environments.ACM Comput.
Surv., 54(6), July 2021.
[163] Yunfan Zhao, Nikhil Behari, Edward Hughes, Edwin Zhang, Dheeraj Nagaraj, Karl Tuyls, Aparna Taneja, and Milind Tambe.
Towards a pretrained model for restless bandits via multi-arm generalization. InIJCAI. IJCAI, 2024.
25
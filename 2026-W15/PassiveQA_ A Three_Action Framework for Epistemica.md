# PassiveQA: A Three-Action Framework for Epistemically Calibrated Question Answering via Supervised Finetuning

**Authors**: Madhav S Baidya

**Published**: 2026-04-06 09:54:16

**PDF URL**: [https://arxiv.org/pdf/2604.04565v1](https://arxiv.org/pdf/2604.04565v1)

## Abstract
Large Language Models (LLMs) have achieved strong performance in question answering and retrieval-augmented generation (RAG), yet they implicitly assume that user queries are fully specified and answerable. In real-world settings, queries are often incomplete, ambiguous, or missing critical variables, leading models to produce overconfident or hallucinated responses.
  In this work, we study decision-aware query resolution under incomplete information, where a model must determine whether to Answer, Ask for clarification, or Abstain. We show that standard and enhanced RAG systems do not reliably exhibit such epistemic awareness, defaulting to answer generation even when information is insufficient.
  To address this, we propose PassiveQA, a three-action framework that aligns model behaviour with information sufficiency through supervised finetuning. Our approach integrates structured information-state representations, knowledge graph-grounded context, and a finetuned planner that explicitly models missing variables and decision reasoning.
  Experiments across multiple QA datasets show that the finetuned planner achieves significant improvements in macro F1 and abstention recall while reducing hallucination rates, under a compute-constrained training regime.
  These results provide strong empirical evidence that epistemic decision-making must be learned during training rather than imposed at inference time.

## Full Text


<!-- PDF content starts -->

Madhav S Baidya PassiveQA
PassiveQA: A Three-Action Framework for
Epistemically Calibrated
Question Answering via Supervised Finetuning
Madhav S Baidya
Indian Institute of Technology (BHU) Varanasi
madhavsukla.baidya.chy22@itbhu.ac.in
Abstract
Large Language Models (LLMs) have achieved strong performance in question answering
and retrieval-augmented generation (RAG), yet they implicitly assume that user queries are
fully specified and answerable. In real-world settings, queries are often incomplete, ambiguous,
or missing critical variables, leading models to produce overconfident or hallucinated responses.
Existing approaches lack mechanisms to decide whether a query should be answered, clarified,
or abstained from.
In this work, we studydecision-aware query resolution under incomplete information,
where a model must jointly infer the sufficiency of available information and select an
appropriate action:Answer,Ask, orAbstain. We show that standard and enhanced RAG
systems do not consistently exhibit such epistemic awareness, defaulting to answer generation
even when information is insufficient.
To address this, we proposePassiveQA, a three-action decision framework that learns
to align model behaviour with information sufficiency through supervised finetuning. Our
approach integrates structured state representations, knowledge graph-grounded context,
and a finetuned planner that explicitly models missing variables and decision reasoning.
Experiments across four multi-source QA datasets show that the finetuned planner
achieves 55.6% macro F1 — a 20.3 percentage point gain over the best inference-time
architecture — withAbstainrecall rising from 13.3% to 58.1% and hallucination rate falling
from 42.7% to 33.8%, while maintaining 100% structured output compliance across 5,218 test
samples. Our results highlight the limitations of inference-time interventions and emphasise
the need for training-time alignment of epistemic decision-making in language models.
Keywords:question answering, retrieval-augmented generation, epistemic calibration, knowl-
edge graphs, supervised finetuning, hallucination reduction.
Code Availability.The full codebase — including dataset construction pipelines, knowledge
graph builder, finetuning dataset generator, LoRA training scripts, and the three-agent inference
pipeline — is publicly available at https://github.com/MadsDoodle/PassiveQA . Questions,
bug reports, and suggestions are welcome via the repository’s issue tracker.
1 Introduction
Consider a legal QA system asked:“Am I entitled to redundancy pay?”The system retrieves
the correct statutory clause and responds:“Yes, employees are entitled to statutory redundancy
pay after two years of continuous service”— without ever establishing that the user has been
employed for fourteen months. The retrieved document is correct. The generated text is faithful
to it. The system fails anyway, because it committed to an answer without first determining
1arXiv:2604.04565v1  [cs.CL]  6 Apr 2026

Madhav S Baidya PassiveQA
whether the information available was sufficient to ground one. This is anepistemic failure,
structurally invisible to standard retrieval and generation metrics.
Large language models and retrieval-augmented generation systems share an assumption that
is rarely stated and almost never challenged: every query is answerable [Lewis et al., 2020, Gao
et al., 2023]. The system’s only task, in the standard framing, is to find the best answer. There is
no mechanism for deciding whether an answer should be attempted at all. This assumption fails
routinely in deployment — queries arrive incomplete, with implicit variables the user expects
the system to ask about, or with topics absent from the knowledge base entirely. In each case,
the standard RAG response is the same: retrieve the highest-scoring chunks and generate the
most fluent completion [Li et al., 2025, Ji et al., 2023]. Critically, improving retrieval quality
does not fix this. Better retrieval surfaces more plausible-looking evidence, which, without an
epistemic decision gate,increasesoverconfident answering [Li et al., 2025]. We demonstrate this
empirically: an enhanced RAG pipeline with semantic chunking, hybrid retrieval, cross-encoder
reranking, and self-reflection achieves a hallucination rate of 51.7% — higher than the 42.7%
baseline it was designed to improve.
The standard mitigation is confidence calibration [Geifman and El-Yaniv, 2017, Xin et
al., 2021] — assign a scalar to each response and abstain below a threshold. This framing is
insufficient for two reasons. First, it collapses two qualitatively distinct epistemic states: a
query unanswerable because its topic is absent from the KB (irrecoverable — honest refusal is
correct) versus a query unanswerable because a critical variable is missing butrecoverable through
dialogue(a targeted clarification question is correct). Conflating these discards information the
user needs [Aliannejadi et al., 2019, Saeidi et al., 2018]. Second, entropy-based confidence signals
are insufficient for safe selective prediction in LLMs [Phillips et al., 2026], and models frequently
fail to refuse even when their confidence is low [Yin et al., 2024].
We instead ground the routing decision in theinformation state S(q) = (Vknown , Vmissing , C)
and define three actions with non-overlapping epistemic semantics:
(i)Answer—V missing =∅and retrieved evidence provides a complete reasoning path.
(ii)Ask— Vmissing ̸=∅but missing variables arerecoverable: a targeted clarification resolves
the gap.
(iii)Abstain— Vmissing ̸=∅and missing information isirrecoverable: no user clarification will
help.
Hallucination risk scales monotonically with the incompleteness measure I(q) =|Vmissing|/(|Vknown|+
|Vmissing|), motivating variable-state tracking as the primary routing signal.
The central claim of this paper is that this epistemic calibrationcannot be achieved at
inference time. We establish this by exhausting the space of inference-time interventions
across three progressively stronger RAG architectures. All three plateau at 34–38% decision
accuracy and below 14%Abstainrecall, with a consistent failure mode: the model defaults to
Answerregardless of the information state, because that behaviour is encoded in its pretraining
distribution [Ouyang et al., 2022]. No inference-time signal can override a prior this strong
without modifying the parameters that encode it.
To address this, we proposePassiveQA, a framework for epistemically calibrated QA
through training-time alignment. PassiveQA constructs adecision-weighted knowledge graph
G2in which edge weights encode the three-action supervision signal —Answer-supporting
paths are reinforced,Abstain-associated paths penalised, and recoverable missing variables are
injected as explicit ?var placeholder nodes. A 34K-sample finetuning corpus of graph-grounded
structured reasoning chains then trains a LoRA-adapted Mistral-7B-Instruct planner [Jiang et
al., 2023, Hu et al., 2021] to produce explicit, structured decisions over the three-action space.
Results.The finetuned planner achieves55.6% macro F1on the held-out test split — a
20.3 pp gainover the best inference-time architecture (v3: 35.3%) and28.9 ppover the
2

Madhav S Baidya PassiveQA
standard RAG baseline (26.7%).Abstainrecall rises from 13.3% to58.1%, confirming that
the empty-graph signal is learnable through training in a way that inference-time thresholds
cannot replicate. Per-output structural compliance is 100% across 5,218 test samples. These
results hold under a severely compute-constrained regime — 2 epochs over 9,000 samples (26%
of the available 34K finetuning set) — establishing a conservative lower bound.
Positioning.PassiveQA sits at the intersection of three research threads that have largely
developed independently: selective prediction [Geifman and El-Yaniv, 2017, Xin et al., 2021],
conversational clarification [Saeidi et al., 2018, Aliannejadi et al., 2019], and knowledge graph-
augmented reasoning [Yasunaga et al., 2021, Sun et al., 2019]. Prior work in selective prediction
operates on classification models with calibrated outputs and treats abstention as a binary
post-hoc decision. Prior work on clarification focuses on question generation conditioned on
a fixed dialogue policy, without modelling the information state that determineswhetherto
ask. Prior work on KG-augmented QA uses graph structure to improve answer quality, but
not to route queries away from answering entirely. PassiveQA unifies these threads by treating
the three-action routing decision as a first-class learning objective, grounded simultaneously in
structured knowledge, explicit variable-state tracking, and training-time behavioural alignment.
The decision-weighted knowledge graph G2is the concrete artefact that makes this unification
possible: it is the only component in the pipeline that jointly encodes semantic grounding,
multi-hop reasoning signal, and epistemic supervision in a single structure that the planner can
query at inference time.
Contributions:
•A formal three-action decision framework grounded in decision theory and information-state
decomposition, including a joint answerability signal A(q) and multi-turn resolution rate ρt
(§3).
•Strong empirical evidence that inference-time interventions fail to consistently instil epistemic
passivity, established across three progressively stronger RAG architectures (§5).
•A multi-source dataset pipeline unifying four QA benchmarks with explicit variable-state
supervision (§4).
•A three-phase KG construction procedure whose central novelty is query-guided decision
reinforcement on edge weights and ?var placeholder injection for recoverable missing variables
(§6).
•A 34K-sample graph-grounded finetuning dataset and LoRA training procedure for instilling
three-action routing as a learned capability (§8).
•A three-agent execution architecture with strict routing/generation separation, making planner
and agent failures independently diagnosable (§9).
Paper organisation.Section 2 surveys related work across the six threads that PassiveQA
draws from. Section 3 develops the formal framework, including the information-state decomposi-
tion, utility-maximisation framing, and the full set of retrieval and graph scoring signals. Section 4
describes dataset construction, balancing, and variable population. Section 5 presents the three
RAG architectures and their evaluation, establishing the inference-time ceiling. Section 6 de-
tails the three-phase knowledge graph construction and post-processing pipeline. Section 7
describes the finetuning dataset construction and prompt schema. Section 8 covers the LoRA
training procedure. Section 9 presents the three-agent execution architecture. Section 10 reports
quantitative results, planner observations, and qualitative error analysis. Section 11 discusses
limitations and future directions, and Section 12 concludes.
3

Madhav S Baidya PassiveQA
Figure 1: Full PassiveQA pipeline.Left to right:the four source datasets are merged into a
unified 61K-sample schema with explicit variable-state fields (§4); a knowledge base of 105,420
chunks is constructed and indexed (§5); three progressive RAG architectures are evaluated on
the KB alone (§5); the KB is simultaneously processed through a three-phase KG construction
pipeline producing the decision-weighted graph G2(§6); G2and the unified dataset jointly
generate the 34K KG-grounded finetuning dataset (§7), which trains the LoRA planner (§8); at
inference the planner receives the query and KG context and routes to one of three specialised
agents (§9). The dashed feedback arrow models multi-turn state update (Eq. 20): a resolved
variable from theAskagent transitions fromV missing toV known before the next planner call.
2 Related Work
Retrieval-Augmented Generation.RAG systems augment language models with non-
parametric retrieval over an external knowledge base [Lewis et al., 2020, Gao et al., 2023].
Standard pipelines retrieve top- kchunks via dense passage retrieval [Karpukhin et al., 2020] or
retrieval-conditioned language modelling [Guu et al., 2020]. Subsequent work explored fusion-in-
decoder generation [Izacard and Grave, 2021] and retrieval from massive datastores [Borgeaud
et al., 2022]. Hybrid pipelines combining BM25 [Robertson and Zaragoza, 2009] with dense
encoders and late-interaction reranking [Santhanam et al., 2022] are now standard. Despite
4

Madhav S Baidya PassiveQA
these advances, retrieval improvements without explicit epistemic modelling tend to increase
overconfident answering — a finding we corroborate empirically in§5.
Hallucination and Faithfulness.Hallucination in LLMs is widely recognised as a critical
obstacle to deployment [Ji et al., 2023, Maynez et al., 2020]. Dedicated detection pipelines
with calibrated ensembles have recently been proposed for RAG settings [Goswami and Kurra,
2025], and selective abstention learning has been used as a training-time mechanism to suppress
hallucination at the source [Huang et al., 2025]. Our work differs from all of the above by
targeting thedecision to answer at all, rather than post-hoc verification of a generated answer [Li
et al., 2025].
Selective Prediction and Abstention.Selective prediction [Geifman and El-Yaniv, 2017]
studies the coverage–accuracy trade-off. Xin et al. [Xin et al., 2021] formalise abstention for
NLP via error regularisation; Varshney and Baral [Varshney and Baral, 2023] extend this to
open-domain QA by revisiting abstained instances post-hoc. Recent empirical work shows that
LLMs frequently do not know when to refuse [Yin et al., 2024], and that confidence entropy
alone is an insufficient gating signal [Phillips et al., 2026]. We extend selective prediction to a
three-way open-ended QA routing decision.
Conversational and Clarification-seeking QA.The ShARC benchmark [Saeidi et al.,
2018] introduced clarification-seeking before committing to a policy-grounded answer, directly
motivating ourAskaction. QuAC [Choi et al., 2018] models information-seeking dialogues with
explicit unanswerable annotations. The importance of proactively asking clarifying questions
was established by Aliannejadi et al. [Aliannejadi et al., 2019], and learning which questions to
generate was studied by Rao and Daum´ e [Rao and Daum´ e, 2018]. Recent work on conversational
machine reading formalises the decision to inquire [Zhou et al., 2023], but without a unified
three-action framework or graph-grounded supervision. The original ShARC system achieved
61.4% macro accuracy using task-specific rule-following models [Saeidi et al., 2018]; PassiveQA
does not target this benchmark directly, as it is designed as a general-purpose epistemic routing
framework rather than a policy-compliance system, but the ShARC component of our evaluation
inherits its annotation scheme and supervision signal.
Knowledge Graphs for QA.Graph-augmented retrieval supports multi-hop reasoning by
organising knowledge into structured representations [Jiang et al., 2025]. QA-GNN [Yasunaga et
al., 2021] jointly reasons over language models and knowledge graphs via graph neural networks;
Sun et al. [Sun et al., 2019] ground multi-hop QA directly over entity subgraphs; and KagNet [Lin
et al., 2019] integrates graph networks with commonsense knowledge. Our work extends this
paradigm by constructing adecision-reinforcedknowledge graph whose edge weights encode the
three-action supervision signal — a dimension absent from prior graph-augmented QA systems.
Parameter-Efficient Finetuning and Alignment.LoRA [Hu et al., 2021] and QLoRA [Dettmers
et al., 2023] enable efficient finetuning within single-GPU memory budgets. InstructGPT [Ouyang
et al., 2022] demonstrated that RLHF can align model behaviour with human preferences, mo-
tivating training-time alignment as the fundamental mechanism for behaviour change. Direct
Preference Optimisation [Rafailov et al., 2023] offers a stable supervised alternative. We apply
LoRA to instil three-action decision behaviour in Mistral-7B-Instruct-v0.3 [Jiang et al., 2023],
treating epistemic routing as a behaviour alignment problem rather than a prompt engineering
problem.
5

Madhav S Baidya PassiveQA
3 Formal Framework
3.1 Information State and Decision Policy
Letqdenote a user query. We define theinformation stateofqas:
S(q) = 
Vknown , Vmissing , C
(1)
where Vknown is the set of concrete entities or attributes explicitly present in the query and
retrieved context, Vmissing is the set of variables required to resolve qbut absent from S, and C
is the set of constraints and conversational context. Theincompleteness measureis:
I(q) =|Vmissing|
|Vknown|+|V missing|(2)
The plannerπ θis a probabilistic policy over actionsa∈ {Answer,Ask,Abstain}:
πθ(a|S(q)) = softmax 
fθ(S(q))
(3)
wheref θis a neural network mapping the information state to action logits.
The planner’s role is not to generate a response but to select the action that maximises
expected utility given S(q). This separation betweendecisionandgenerationis the architectural
commitment that distinguishes PassiveQA from standard RAG systems, which conflate the two
into a single generation step.
State initialisation.At the start of each dialogue, the state is initialised from the current
query and retrieved context:
S0(q) = 
ent(q)∪ent(D),req(q)\ent(D),∅
(4)
where ent(·) denotes named entities extracted from a text and req(q) denotes the set of variables
logically required to answer q. The constraint set Cis empty at turn zero and accumulates
resolved variables and dialogue history across subsequent turns.
Action space semantics.The three actions partition the response space into non-overlapping
epistemic commitments. Formally, let Rbe the space of all possible responses. The planner
defines a routing function:
ϕ:S(q)−→

Rans a=Answer
Rclar a=Ask
Rref a=Abstain(5)
where Ransis the space of KB-grounded factual responses, Rclaris the space of well-formed
clarification questions, and Rrefis the space of specific, honest refusals. The routing function
ϕis implemented by the finetuned planner at inference time and by the hard gate (Eq. 21) in
Architecture 3.
Relationship between incompleteness and action.The three actions correspond to three
regimes ofI(q):
a∗(q)≈

AnswerI(q)≈0
Ask0< I(q)<1∧V missing is recoverable
AbstainI(q)≈1∨V missing is irrecoverable(6)
6

Madhav S Baidya PassiveQA
The distinction betweenAskandAbstainwhen Vmissing ̸=∅depends on whether the missing
information isrecoverable— i.e. whether a user clarification could plausibly supply it. This
recoverability judgement is encoded in the knowledge graph via the presence or absence of ?var
nodes (§6): a missing variable that can be injected as a graph placeholder is recoverable; a query
whose topic has no graph nodes at all is not.
3.2 Decision as Utility Maximisation
Action selection is framed as expected utility maximisation:
a∗= arg max
aE
U(a|S(q))
(7)
The utility function encodes the epistemic trade-offs:
•U(Answer): high reward for a correct, grounded response; high penalty for hallucination.
•U(Ask): moderate cost (extra conversational turn) but reduces|V missing|.
•U(Abstain): safe floor; avoids harm but provides no useful information.
3.3 Hallucination Risk
We define hallucination risk as the conditional probability of generating an incorrect answer
when committing toAnswer:
H(q) =P 
incorrect answer|S(q), a=Answer
(8)
We hypothesise a structural relationship between hallucination risk and information incomplete-
ness:
H(q)∝I(q) (9)
Hallucination risk increases monotonically with information incompleteness.This motivates
the use of I(q) as a gating signal before answer generation. This relationship is motivated by
observed trends in our experiments, though we do not provide explicit statistical validation.
3.4 Retrieval Signals
Two complementary retrieval-derived signals are formalised.Confidencemeasures the best-
matching chunk in the document setD:
Conf(q) = max
d∈Dsim(q, d) (10)
Coveragemeasures lexical completeness of the query terms against the retrieved set:
Coverage(q) =|terms(q)∩terms(D)|
|terms(q)|(11)
3.5 Ambiguity Score
The ambiguity signal is a feature-based estimator overKheuristic signalsh i(q)∈[0,1]:
Amb(q) =1
KKX
i=1hi(q) (12)
The heuristics include: query length ( ≤4 words), presence of dangling pronouns, vague quantifiers,
absence of named entities, and comparative constructions without both comparison arguments.
7

Madhav S Baidya PassiveQA
3.6 Conflict Score
When multiple retrieved chunks address the same query from contradictory directions, committing
toAnswerrisks producing an internally inconsistent response. We define theconflict scoreas
the mean pairwise cosine dissimilarity among the top- kretrieved chunk embeddings {e1, . . . , e k}:
Conflict(q) = 1−2
k(k−1)k−1X
i=1kX
j=i+1sim(e i, ej) (13)
A high conflict score ( Conflict (q)> τ con) indicates that the retrieved evidence is internally
inconsistent — the query may be ambiguous across document boundaries, or the KB contains
contradictory entries. In either case, the appropriate response is to seek clarification rather than
to synthesise a potentially misleading answer. Conflict thus serves as the highest-priority signal
in the hard gating rule (Eq. 21).
3.7 Joint Answerability Signal
The four signals — Conf (q),Coverage (q),Amb(q), and Conflict (q) — are each necessary but
not individually sufficient for routing. We define a scalarjoint answerability signal A(q)∈[0,1]
that summarises evidential support:
A(q) = Conf(q)·Coverage(q)· 
1−Amb(q)
· 
1−Conflict(q)
(14)
Each factor lies in [0 ,1], so A(q) = 1 only when retrieval confidence is maximal, query terms
are fully covered, the query is unambiguous, and retrieved chunks are mutually consistent —
the ideal conditions forAnswer. When any factor approaches zero, A(q) collapses regardless
of the others, encoding the intuition that a single failure mode is sufficient to warrant caution.
The hard gate (Eq. 21) can equivalently be read as a threshold rule on Eq. 14 with per-factor
thresholds that allow individual signals to override the product when their deviation is extreme.
3.8 Variable Resolution and Information Gain
In multi-turn settings, theAskaction is only valuable if it reduces incompleteness. We define
theresolution rateat turntas:
ρt=|Vknown,t |
|Vknown,t |+|V missing,t |= 1−I t(q) (15)
A successfulAskat turn ttransitions one or more variables from Vmissing,t toVknown,t+1 per
Eq. 20, strictly increasing ρt+1> ρt. The expected information gain of issuing a clarification
questioncis:
IG(c|S t) =E[ρ t+1−ρt|St, at=Ask, c] (16)
A well-formed clarification question maximises IGby targeting the specific missing variable
that is both (a) necessary to resolve the query and (b) most likely to be known to the user.
The anchor-based question builder (§7) implements a heuristic approximation of this objective:
it selects the missing variable with the highest SBERT similarity to the known variable set,
targeting the gap most proximate to information the user has already provided.
3.9 Knowledge Graph Scoring
Each edge ein the knowledge graph receives a final weight combining semantic grounding
(Phase 2) and query-action reinforcement (Phase 3):
we=α·sem(e) + (1−α)·act(e) (17)
8

Madhav S Baidya PassiveQA
where sem(e) is the semantic similarity score between the triple and its source passage, and
act(e) is the accumulated action reinforcement signal. With α= 0.5, both components receive
equal weight.
The path score over a graph pathpin the set of all pathsP(q) is:
PathScore(q) = max
p∈P(q)Y
e∈pwe (18)
Strong paths ( PathScore≈ 1) supportAnswer; broken paths supportAsk; absent paths
supportAbstain.
3.10 Training Objective
The finetuning loss decomposes into a decision classification component and a generation
component:
L=L decision +λL generation (19)
where Ldecision =−P
(q,a,y) logP θ(a|S (q)) is the cross-entropy loss over action labels and
Lgeneration is the standard causal language modelling loss over the structured reasoning chain.
3.11 Multi-turn State Update
For multi-turn dialogues, the information state is updated after each turn:
St+1=St∪resolve(q t, at) (20)
where resolve transfers variables from Vmissing toVknown when the user’s response satisfies a
clarification request.
3.12 Decision Boundary
The hard gating rule in Architecture 3 implements the following decision boundary:
a=

Answerif Conf(q)> τ c∧I(q)≈0
AskifI(q)> τ i∨Amb(q)> τ a
Abstainotherwise(21)
whereτ c,τi,τaare tunable thresholds.
3.13 Relationship Between Formal Signals and Implementation
The equations in this section range from directly implemented to approximately realised. Table 1
summarises the mapping.
Equations A(q),PathScore (q), and IG(c|S t) are best understood as conceptual abstractions
of the decision process, with current implementations relying on heuristic approximations.
Grounding these signals in learnable parameters — for instance, learning αin Eq. 17 end-to-end,
or training a dedicated path scorer — is deferred to future work.
4 Dataset Construction
4.1 Dataset Selection
Four publicly available datasets were selected to provide complementary coverage of the three
target actions across diverse reasoning contexts.
9

Madhav S Baidya PassiveQA
Table 1: Mapping from formal signals to implementation components.
Equation Status Implementation
I(q) (Eq. 2) Direct Variable population via GPT-4o-
mini (§4); threshold τiin hard
gate (Eq. 21)
Conf(q) (Eq. 10) Direct Max sigmoid cross-encoder logit,
Architecture 3
Coverage(q) (Eq. 11) Direct Non-stopword term overlap, Ar-
chitecture 3
Amb(q) (Eq. 12) Direct Five heuristic signals, Architec-
ture 3
Conflict(q) (Eq. 13) Direct Pairwise chunk dissimilarity, Ar-
chitecture 3
A(q) (Eq. 14) Approximated The hard gate uses per-signal
thresholds rather than the prod-
uct form; A(q) provides the theo-
retical motivation for why all four
signals are jointly necessary
PathScore(q) (Eq. 18) Approximated Edge weights are used to rank
triples during KG context re-
trieval (§7); explicit path enumer-
ation is not computed at inference
time
IG(c|S t) (Eq. 16) Heuristic The anchor-based question
builder (§7) maximises SBERT
similarity between missing vari-
able and known variable set as a
proxy; no empirical validation is
conducted
L(Eq. 19) Direct Causal LM loss over assistant
turn tokens,λ= 1
ShARC [Saeidi et al., 2018].A multi-turn dialogue dataset grounded in government policy
documents. Each sample contains a regulatory snippet, an optional user scenario, a question,
and a gold label. The label space maps directly onto the three actions: Yes/No responses map
toAnswer(13,830 samples); Follow-on responses — where the system must request additional
information — map toAsk; and Irrelevant labels map toAbstain(1,256 samples). ShARC
provides explicit evidence chains (clarification questions) that serve as primary supervision for
theAskaction. Approximately 81.6% of samples include a user scenario, and dialogue history
depth ranges from 0 to 6 prior turns.
QuAC [Choi et al., 2018].A large-scale conversational reading comprehension dataset built
on Wikipedia passages, comprising 83,568 training QA pairs across 11,567 dialogues (avg. 7.2
turns). CANNOTANSWER responses (17.3% of training samples) are mapped toAbstain; turns
marked followup=y are mapped toAsk; all remaining answered turns map toAnswer. QuAC
is the primary source of multi-turnAnswersupervision.
HotpotQA [Yang et al., 2018].A multi-hop QA dataset with 90,447 training samples
requiring reasoning over two or more Wikipedia passages. All samples have gold answers; the
10

Madhav S Baidya PassiveQA
entire split maps toAnswer. The dataset contributes two reasoning types:bridgequestions
(72,991 samples) requiring chained inference across documents, andcomparisonquestions (17,456
samples) requiring attribute comparison.
ContractNLI [Koreeda and Manning, 2021].Legal contract review framed as natural
language inference over 423 NDA documents, with 7,191 clause-level annotations across 17 NDA
clause types. Entailment and Contradiction labels both map toAnswer(60.8%); NotMentioned
maps toAbstain(39.2%). This dataset provides a controlledAbstainsource where abstention
arises from genuine absence of relevant information rather than query-context mismatch.
4.2 Unified Schema
All four datasets are converted into a single JSON schema with fields for query ,context documents ,
action label ,response , a structured state object ( known variables ,missing variables ,
failure mode,completeness), andmetadata(source, multi-turn flag, turn ID, difficulty).
The ShARC evidence chain directly populates missing variables , providing explicitAsk
supervision. The final merged training set comprises 203,096 samples with the distribution:
Answer65.1%,Ask22.4%,Abstain12.5%.
4.3 Dataset Balancing
The raw merged dataset is heavily skewed towardAnswer(65.1%), which would bias the
planner toward reflexive answering — precisely the behaviour this work aims to counteract. A
balanced 61K subset was constructed with the target distribution:Answer33% ·Ask37% ·
Abstain30%.
Dialogues were treated as atomic units during sampling to preserve conversational coherence.
Turn depth was capped per dialogue length. Of 12,195 unique dialogues, 609 were capped; all
resulting sequences were verified for continuity with zero broken chains.
Source minimums were enforced: ContractNLI (7K — all available), HotpotQA (14K),
ShARC (14K), QuAC ( ∼26K). The achieved action distribution after trimming settled at
Answer49.9% /Ask28.9% /Abstain21.2%, shifted substantially toward passive actions.
The hardAnswerfloor from HotpotQA (which contributes noAsk/Abstainsignal) prevented
reaching the original targets exactly.
4.4 Variable Population via GPT-4o-mini
The unified schema includes known variables and missing variables fields. For ShARC,
missing variables were directly available from the evidence chain. For all other sources, these
fields were populated via GPT-4o-mini under controlled prompting. While this introduces
potential annotation noise, the variables serve as weak supervision signals, and the model
primarily learns aggregate decision patterns rather than relying on exact variable identity. The
extraction prompt instructed the model to identify: (a) known variables — concrete entities or
attributes explicitly present in the query (max 5); and (b) missing variables — information
required to resolve the query but absent, with the hard constraint thatAnswer-action samples
always receive an emptymissing variableslist.
Extraction was parallelised in batches of 50 with 5 concurrent workers and checkpoint-based
crash recovery. The resulting variable fields serve dual purposes: inputs to the knowledge graph
builder (known variables anchor graph traversal) and the core supervision signal for the planner’s
finetuning dataset.
Table 2 summarises the final dataset composition.
The contrast in context length between ShARC (45.5 words — compact policy snippets) and
ContractNLI (1,674 words — full NDA documents) motivates the source-aware chunking strategy
11

Madhav S Baidya PassiveQA
Table 2: Final merged dataset composition across sources and action labels.
Source Total Answer Ask Abstain
ShARC 14,000 – Primary 1,256
QuAC 26,000 Primary FollowupCANNOTANSWER
HotpotQA 14,000 All – –
ContractNLI 7,000 60.8% – 39.2%
Merged 61K49.9% 28.9% 21.2%
Table 3: Dataset characteristics across the four sources.
Property ShARC QuAC HotpotQA ContractNLI
Avg. question length (words) 7.5 6.5 17.8 18.2
Avg. context length (words) 45.5 400+ varies 1,674
Multi-turn Yes (up to 6) Yes (avg 7.2) No No
Asksignal source Evidence chainfollowup=yNone None
Abstainsignal source Irrelevant labelCANNOTANSWERNone NotMentioned
Reasoning type Policy Wikipedia Multi-hop NLI
Avg. turns per dialogue 2.3 7.2 1.0 1.0
% queries with named entities 61.2 74.8 92.1 88.4
described in§5: uniform chunking would either fragment ShARC snippets below minimum
coherence or leave ContractNLI documents unchunked and oversized for retrieval.
5 RAG Architectures
We evaluate three progressively more sophisticated RAG architectures on a balanced 900-sample
held-out set (300 per action) before introducing the graph-grounded planner.
5.1 Architecture 1 — Baseline RAG
The baseline follows a standard retrieve-then-generate design. Context documents are dedupli-
cated by MD5 hash and indexed with source-aware chunking: ContractNLI documents (dense
legal text, typically 1,600+ words) are chunked when exceeding 300 words; QuAC passages when
exceeding 400 words; ShARC and HotpotQA are kept whole. A fixed word-window strategy
with 50-word overlap is used.
All chunks are encoded with all-MiniLM-L6-v2 (384-dim) and indexed in a FAISS flat
inner-product index over L2-normalised vectors for exact cosine similarity search. At inference,
the top- kchunks are retrieved, concatenated with the query and conversation history, and
passed to Mistral-7B-Instruct-v0.3 (4-bit quantised, NF4) with a system prompt enforcing the
three-action output format. Decoding is greedy (temperature=1.0, do sample=False).
Results.Decision accuracy: 34%; hallucination rate: 42.7%; macro F1: 26.7%. Per-action ac-
curacy reveals severe skew:Answerpredicted correctly 81% of the time;Askaccuracy collapses
to 12%;Abstainto 9%. The confusion matrix shows the model defaulting overwhelmingly to
Answer: 80 of 100Abstaincases and 75 of 100Askcases are incorrectly classified asAnswer.
This provides strong support for the central thesis: a standard RAG system with no epistemic
passivity training is effectively incapable of distinguishing when to ask or abstain, reducing to a
near-unconditional answering system.
12

Madhav S Baidya PassiveQA
5.2 Architecture 2 — Enhanced RAG
The enhanced pipeline introduces five targeted modifications addressing specific failure modes
identified in the baseline.
Multi-Granularity Knowledge Base.The KB is indexed at two granularities: coarse (full
semantic chunks via sentence-boundary-aware chunking) and fine (individual sentences). Both
are embedded in a single FAISS index.
Hybrid Retrieval.A BM25 sparse index is added alongside the dense index. At query time,
scores from both are normalised to [0 ,1] and fused as a weighted sum with α= 0.5, addressing
vocabulary mismatch for domain-specific regulatory and legal text.
Query Understanding.Two pre-retrieval transformations are applied: (i) query rewriting
via Mistral to produce retrieval-friendly formulations of elliptical multi-turn queries; and (ii)
multi-hop detection that decomposes queries likely requiring multi-document evidence into 2–3
independent sub-queries.
Cross-Encoder Reranking and Context Compression.The top-20 hybrid results are
reranked using cross-encoder/ms-marco-MiniLM-L-6-v2 . The top-5 chunks are then com-
pressed to retain only query-relevant sentences via keyword overlap scoring.
Self-Reflection.A second prompt asks the model to verify its action decision. Reflection
changed the initial decision in 12.7% of cases .
Results.Despite these five enhancements, overall decision accuracy remains at 34% with macro
F1 of 26.7% — statistically identical to the baseline. Per-action accuracy is likewise unchanged:
Answer81%,Ask12%,Abstain9%. The confusion structure is preserved verbatim from
the baseline. Hallucination rate rose to 51.7%, demonstrating that better retrieval without
a decision gate canincreaseoverconfident answering by surfacing more plausible-looking but
insufficient evidence. This consistent negative result highlights a key limitation of the RAG
section: theAnswer-skew failure mode is unlikely to be solely a retrieval quality problem,
and not easily addressed through inference-time modifications alone. The root cause lies in the
model’s underlying generation priors — the LLM has no finetuned disposition toward epistemic
passivity.
5.3 Architecture 3 — Decision-Aware RAG (v3)
Architecture 3 abandons the single-prompt generate-and-decide paradigm. Instead, a dedicated
pre-generation pipeline of explicit, interpretable signals gates the LLM before any response is
generated.
Evidence Scoring.Three signals are computed directly from retrieved chunks before any
generation call.Confidenceuses the maximum sigmoid-normalised cross-encoder logit score.
Coveragemeasures the fraction of non-stopword query terms appearing across the retrieved chunk
set.Ambiguityis computed via Equation (12) over five heuristic signals.Conflictmeasures
pairwise cosine dissimilarity among the top-4 chunk embeddings.
Answerability Classifier.A dedicated lightweight LLM call is issued before answer genera-
tion, asking only whether the context is sufficient. The classifier outputs one of three labels:
ANSWERABLE ,NEEDS CLARIFICATION , orNOTANSWERABLE , decoupling epistemic judgement from
fluent answer generation.
13

Madhav S Baidya PassiveQA
Hard Gating.The hard gate combines all signals with explicit priority ordering:
1. Conflict>0.70⇒Ask
2. Conf(q)<0.35∧Coverage(q)<0.30⇒Abstain
3. Amb(q)>0.45⇒Ask
4. Classifier =NOT ANSWERABLE⇒Abstain
5. Classifier =NEEDS CLARIFICATION⇒Ask
6. None of the above⇒Answer
These thresholds are tunable without retraining, making this the only component where passive-
action bias is directly controllable.
Action-Specific Generation.Three separate task-specific prompts handle the three actions.
TheAskprompt receives the gate reason as explicit input and enforces a single well-formed
clarification question. TheAbstainprompt generates a specific statement about missing
information rather than a generic refusal.
Results.Architecture 3 is the only pipeline achieving measurable improvement: decision
accuracy 38%, macro F1 35.3% (+8.6 pp), hallucination rate 33.8% ( −9 pp), andAskrecall
40% (vs. 2% baseline). However,Abstainrecall remains low at 13.3%, and all architectures
expose the ceiling of inference-time interventions, motivating the finetuning approach.
5.4 Quantitative Comparison
Table 4: Quantitative comparison of three RAG architectures on the 900-sample evaluation set.
Metric Baseline Enhanced v3
Decision Accuracy 34.0% 34.0% 38.0%
Macro F1 26.7% 26.7% 35.3%
Hallucination Rate 42.7% 51.7% 33.8%
AskRecall 2.0% 12.0% 40.0%
AbstainRecall 26.0% 9.0% 13.3%
Coverage 67.3% 78.7% 54.0%
6 Knowledge Graph Construction
The knowledge graph is built through a three-phase pipeline that transforms flat KB chunks into
a structured, decision-weighted graph. The key novelty is that the graph is not a passive factual
store — it is actively shaped by the three-action supervision signal, so edge weights encode both
semantic coherence and epistemic utility.
6.1 Phase 1 — Entity and Relation Extraction (G 0)
All 105,420 KB chunks are parsed using spaCy ( encore websm). Two design constraints
distinguish this extraction from standard open information extraction.
Named-entity-only nodes.Only spaCy NER entities are admitted as graph nodes; noun
chunks are excluded to avoid high-degree hub nodes from generic phrases. Entities are normalised,
14

Madhav S Baidya PassiveQA
Query qHybrid
RetrievalEvidence
ScoringHard
GateAnswer
Agent
Ask
Agent
Abstain
AgentAnswer
Ask
AbstainFinetuned Planner
(Mistral-7B-
Instruct-v0.3
+ LoRA)
signalsdecision
Figure 2: Overview of the PassiveQA pipeline. After hybrid retrieval and evidence scoring, either
the hard gate (Architecture 3) or the finetuned planner routes the query to one of three specialised
agents. Dashed arrows indicate the planner path used in the full three-agent architecture.
filtered through a hard validator (rejecting pronouns, stopwords, pure numerics, strings under 3
characters), and mapped to seven semantic categories: Person, Organisation, Location, Attribute,
Work, Concept, Event.
Entity-constrained triples.Relations are extracted via syntactic dependency parsing. A triple
is admitted only if both subject and object map to named entities in the entity set, eliminating
pronoun-based spurious hub nodes. Weak verbs (copulae, auxiliaries) are filtered except in
HotpotQA, where relational verbs carry genuine multi-hop signal. Initial confidence scores: 0 .8
for direct SVO triples, 0.7 for prepositional-object chains.
G0after Phase 1: 27,189 nodes, 24,491 edges. Isolated nodes are immediately pruned.
6.2 Phase 2 — Semantic Validation (G 1)
Each triple is converted to a natural-language sentence via a template. Both KB chunk texts
and triple sentences are encoded with all-MiniLM-L6-v2 . For each edge, the cosine similarity
between the triple sentence embedding and its source chunk embeddings measures semantic
grounding. A frequency bonus log(1 +freq)×0.03 rewards cross-document corroboration. Edges
belowτ= 0.50 are removed.
G1after Phase 2: 15,468 nodes, 11,965 edges — 12,526 noisy edges removed (51.1% of G0
edges). Average edge weight rises slightly from 0.726 to 0.728, confirming that removed edges
were low-confidence extractions.
6.3 Phase 3 — Query-Guided Decision Reinforcement (G 2)
Phase 3 is the central novel contribution of the KG module. For each of the 273,809 (query,
KB, action) triples in the dataset, the associated graph edges are updated according to the gold
action label:
•Answer: edge weight +0.20 (context was sufficient to answer).
•Ask: edge weight +0.05 (context partially relevant but insufficient).
•Abstain: edge weight−0.10 (context despite surface relevance did not support answering).
For HotpotQA entries, the shortest path between pairs of KB-linked nodes is found and each
edge along that path is reinforced, propagating multi-hop signal through the graph.
Variable node injection.For everyAsk-action sample, query entities absent from the graph
are added as special ?var nodes connected to the nearest graph entity via a requires relation
15

Madhav S Baidya PassiveQA
with weight 0.9. These nodes serve as structural placeholders representing missing information.
A total of 4,295 variable nodes were injected (21.7% of total nodes).
To illustrate the injection mechanism concretely: ?var quac 000014 , anchored tothe United
States, encodes the missing entity in the query“What do we know about Cove Reber?”; while
?var quac 000051 , anchored tothe Air Force, encodes the unresolved gap in“Where was Bernie
born?”— each representing the precise information theAskagent should surface to the user.
The final edge weight combines both signals per Equation (17).
G2after Phase 3: 19,763 nodes (4,295 variable), 17,715 edges. Average edge weight rises to
0.819, reflecting reinforcement of frequently traversedAnswerpaths.
6.4 Graph Statistics
Table 5 summarises the knowledge graph at each construction phase.
Table 5: Knowledge graph statistics across construction phases.
Phase Nodes Edges Avg. Weight Notes
G0 27,189 24,491 0.726 Raw extraction
G1 15,468 11,965 0.728 Semantic validation
G2 19,763 17,715 0.819 Incl. 4,295 variable nodes
The entity distribution across the 15,468 real nodes: Person (31.2%), Organisation (26.8%),
Attribute (18.1%), Location (11.9%), Work (7.7%), Concept (2.4%), Event (1.9%). The graph
is deliberately sparse: average real node degree 1.55, organised into 3,907 weakly connected
components, the largest spanning 5,834 nodes. Edge weight distribution shows 47.1% of edges
in the top bucket (0.8–1.0), a direct consequence of Phase 3 reinforcement concentrating weight
onAnswer-supporting paths. Top relation types are semantically rich and source-appropriate:
legal relations from ContractNLI ( disclose to,transfer from), biographical relations from
QuAC (bear in,refer as), and event relations from HotpotQA (defeat,win,release on).
6.5 Graph Post-Processing
Following Phase 3, the graph undergoes a structured post-processing pass that repairs three
categories of quality issues without rerunning any construction phase.
Noise node removal.Despite the hard validator in Phase 1, a residual population of low-
quality nodes survives into G2: (i) mixed alphanumeric strings that passed the length filter
(e.g.the 1980s); (ii) short non-entity tokens marginally exceeding the three-character threshold;
(iii) generic hub nodes with degree exceeding three times the graph mean whose text spaCy does
not recognise as any named entity type; and (iv) definite descriptions beginning with“the”that
carry no NER label. Detection is performed by batch re-running spaCy NER over normalised
node name strings rather than source passages, providing a second-pass entity check independent
of extraction context. Noise nodes are removed with all incident edges.
Variable node re-anchoring.Following noise removal, a fraction of ?var anchors are
invalidated because their anchor node was itself classified as noise. For each invalidated variable
node, the originating query is encoded with SBERT and matched against all surviving real nodes;
the variable is re-anchored to the highest-similarity node above threshold 0.30. This threshold is
intentionally permissive — any plausible entity connection is preferable to a dangling variable
node. Variable nodes whose best candidate falls below 0.30 are removed entirely. In the G2
post-processing run: 609 bad anchors detected; 331 successfully re-anchored; 278 removed as
unresolvable.
16

Madhav S Baidya PassiveQA
Isolated node pruning and weight recomputation.Degree-0 nodes remaining after the
above steps are removed. Every surviving edge weight is recomputed as:
we= min 
0.95,0.5·sem(e) + 0.5·act(e)
(22)
The cap at 0.95 prevents any single edge from dominating path scoring. Thekb tonodesand
node tokbsindices are recomputed over the surviving node set. The post-processed graph
retains 15,468 real entity nodes and 11,965 edges, with every variable node guaranteed to be
anchored to a spaCy-verified named entity.
7 Planner Finetuning Dataset
7.1 KG-Grounded Context Extraction
For each of the 61K balanced samples, graph context is retrieved via two parallel search paths.
The query is encoded with SBERT and matched against all KG node embeddings (cosine
similarity threshold 0.55, top-5 nodes). Each known variable is similarly matched (top-2 per
variable). The union of matched nodes forms a seed set from which a 2-hop ego graph is extracted.
The top-12 triples by final weight are retained, subject to a secondary relevance check (threshold
0.35) between the triple sentence and the query.
ForAsk-action samples, ?var *nodes adjacent to any seed node are additionally included,
appearing as entity | requires | ?unknown Ntriples. These edges are the graph’s explicit
encoding of missing information.Answer-action samples have all requires /?unknown triples
filtered out.
For multi-turn dialogues, variable states accumulate across the conversation history: known
variables from all prior turns are merged;Ask-action turns propagate resolved variables into the
resolved set, preventing repeated clarification requests.
7.2 Prompt Schema
Each training sample is a three-message conversation in standard chat format:
•System prompt: defines decision rules (complete graph path ⇒Answer; missing linking
variables⇒Ask; no relevant nodes⇒Abstain).
•User turn: structured XML tags carrying <conversation history> ,<resolved variables> ,
<remaining unknowns> ,<query> ,<known variables> ,<graph context> ,<missing variables> .
•Assistant turn: four-step structured reasoning chain followed by <decision> ,<justification> ,
and (forAsk)<clarification question>tags.
The four reasoning steps are deterministically template-filled from real graph evidence and
variable states — not LLM-generated — ensuring supervision is always grounded in actual graph
content.
7.3 Quality Filtering
Six quality checks gate each sample before admission.Answersamples are rejected if the graph
context is empty or no graph triple contains a known entity.Asksamples are rejected if they
have no effective missing variable and no requires edges. A semantic specificity filter rejects
missing variable strings above cosine similarity 0.45 to a set of generic anchor phrases.
Of 60,995 input samples, 34,831 survive (57.1% yield). The 26,164 rejected samples fall
into three categories: answer empty graph (14,386 samples —Answercases where the KB
produced no graph context at all), asknomissing novarnodes (several thousandAsk
17

Madhav S Baidya PassiveQA
Figure 3: Three-phase construction of the PassiveQA knowledge graph.Phase 1( G0):
dependency parsing extracts SVO triples, including low-confidence edges (e.g., pronoun-
linked relations).Phase 2( G1): semantic filtering using SBERT cosine similarity removes
edges below a threshold ( τ= 0.50), pruning noisy connections.Phase 3( G2): graph
refinement via action-based reinforcement, whereAnswersignals strengthen edges and
Abstainsignals penalize them; additionally, variable nodes are introduced via requires
relations to capture missing information.
samples with no usable missing variable or requires edge — incoherent supervision), and
answer graph irrelevant (graph context was present but entirely unrelated to the query).
Table 6 shows the final finetuning dataset statistics.
The action distribution (Answer 23%, Ask 39%, Abstain 37%) is heavily passive-skewed
relative to the raw merged dataset, which is the central design goal. Splits are performed at the
dialogue level with zero dialogue-level contamination confirmed across all splits.
18

Madhav S Baidya PassiveQA
Table 6: Planner finetuning dataset statistics across splits.
Property Train Val Test
Total samples 24,456 5,157 5,218
Answer5,663 (23.2%) 1,213 (23.5%) 1,265 (24.2%)
Ask9,726 (39.8%) 1,987 (38.5%) 2,025 (38.8%)
Abstain9,067 (37.1%) 1,957 (37.9%) 1,928 (36.9%)
Multi-turn 59.4% 58.9% 58.5%
Avg. KG triples 2.31 2.35 2.33
Avg. matched nodes 4.46 4.48 4.42
Zero-triple samples 41.7% 41.4% 39.3%
Avg. known vars 3.39 3.45 3.25
Avg. missing vars 0.80 0.81 0.80
Avg. user turn (words) 108 109 106
Avg. assistant turn (words) 138 138 139
8 Planner Finetuning
8.1 Model and LoRA Configuration
The planner is initialised from mistralai/Mistral-7B-Instruct-v0.3 (7.24B parameters) [Jiang
et al., 2023], loaded in bfloat16 on a single GPU. Full parameter finetuning is computation-
ally prohibitive; Low-Rank Adaptation (LoRA) [Hu et al., 2021] is applied with rank r= 32,
α= 64 (effective scale α/r= 2.0), dropout 0.05. LoRA is applied to all seven projection ma-
trices: qproj, k proj, v proj, o proj, gate proj, up proj, down proj. Including the
MLP projections is important because the decision task requires routing based on structured
reasoning over graph content, a capability residing in the feedforward layers as much as the
attention layers. Trainable parameters:∼83M (1.15% of total).
The rank r= 32 is larger than typical LoRA setups ( r= 8 or 16), justified by task complexity:
the model must parse XML-tagged structured input, reason over graph triples in a four-step
format, output structured XML tags in correct order, and generate focused clarification questions.
The model is loaded in 4-bit NF4 quantisation following QLoRA [Dettmers et al., 2023] to fit
within single-GPU VRAM constraints.
8.2 Training Procedure
Training data is subsampled at the dialogue level to 9,000 training and 1,200 validation samples
targetingAnswer30% /Ask38% /Abstain32%. The Mistral chat template is applied:
<s>[INST]{system prompt} {user turn}[/INST]{assistant turn}</s>
Loss is computed only on {assistant turn}</s> tokens — the system prompt and user turn
are masked, so the model learns purely from the structured reasoning chain and decision tags.
Table 7 details the training hyperparameters.
Gradient checkpointing is enabled to reduce VRAM usage. The adapter is saved to persistent
storage after each epoch. The training objective instantiates Equation (19) with λ= 1, treating
decision classification and reasoning generation as equal components of the causal LM loss.
19

Madhav S Baidya PassiveQA
Table 7: Planner finetuning hyperparameters.
Hyperparameter Value
Base model Mistral-7B-Instruct-v0.3
LoRA rank (r) 32
LoRA alpha (α) 64
LoRA dropout 0.05
Trainable parameters∼83M (1.15%)
Sequence length 512 tokens
Precision bfloat16
Epochs 2
Per-device batch size 4
Gradient accumulation steps 8
Effective batch size 32
Learning rate 2×10−4
LR scheduler Cosine decay
Warmup ratio 5%
Optimiser AdamW
Max gradient norm 1.0
Weight decay 0.01
Best model criterion Minimum eval loss
9 Three-Agent Architecture
Knowledge
Graph G2Planner
(Mistral-7B-
Instruct-v0.3
+ LoRA)
Query + KG contextAnswer Agent
(Enhanced RAG)
Ask Agent
(Clarification)
Abstain Agent
(Honest Refusal)triples<decision>Answer
<decision>Ask
<decision>Abstain
Figure 4: Three-agent architecture. The finetuned planner parses the <decision> tag
and delegates execution to one of three specialised agents. TheAskagent reuses the
<clarification question>tag directly.
The trained planner serves as the routing controller for a three-agent system with a strict
separation of concerns: the planner is responsible solely for the decision, and the agents are
responsible solely for execution. This design makes failure modes transparent — a routing error
is a planner failure; a poor output given correct routing is an agent failure.
20

Madhav S Baidya PassiveQA
Answer Agent.Invoked on <decision>ANSWER</decision> . Operates using the enhanced
RAG retrieval pipeline (hybrid BM25 + dense retrieval, cross-encoder reranking, context
compression) followed by a context-grounded generation prompt. The decision burden is
deliberately removed from the generation step, reducing hallucination pressure.
Ask Agent.Invoked on <decision>ASK</decision> . The planner’s <clarification question>
tag already contains the specific question, constructed during finetuning dataset generation via
the anchor-based question builder: missing variable strings are matched against known variables
via SBERT similarity (threshold 0.20) to find anchor entities, and the final question follows the
template“Regarding [anchor]: could you specify [missing]?”. The Ask Agent’s role is primarily
structural: packaging the question into the conversational response format and appending it
to history with theAskaction label. In subsequent turns, the user’s response is assumed to
resolve the missing variable, transitioning it fromV missing toV known per Equation (20).
Abstain Agent.Invoked on <decision>ABSTAIN</decision> . Generates a specific, honest
statement explaining why the query cannot be answered, distinguishing between two causes:
(a) the query topic is entirely absent from the KB, and (b) the topic is present but the required
specific information is not recoverable through clarification. This distinction communicates
meaningfully to the user and avoids generic refusals.
10 Results and Discussion
10.1 Key Observations
Observation 1 — Retrieval quality improvements alone cannot instil epistemic
passivity.The Enhanced RAG introduced five non-trivial improvements yet produced identical
overall accuracy (34.0%) and identical macro F1 (26.7%) to the baseline. The enhanced pipeline’s
hallucination rate washigherthan baseline (51.7% vs. 42.7%), demonstrating that better retrieval
without a decision gate increases overconfident answering.
Observation 2 — Hard gating produces the only measurable improvement across all
three inference-time architectures.Architecture 3 is the only pipeline improving on any
metric relative to baseline: accuracy +4 pp, macro F1 +8.6 pp (26.7% →35.3%), hallucination
rate−9 pp, andAskrecall +38 pp (2% →40%). The macro F1 gain reflects a genuine
redistribution of predictions towardAskrather than a marginal shift in the dominant class.
Observation 3 — Ask is the most recoverable action without finetuning.Archi-
tecture 3 achieves 40%Askrecall vs. 2% baseline and 12% enhanced, demonstrating that
ambiguity heuristics and the answerability classifier together capture a meaningful fraction of
clarification-needing cases at query level alone.
Observation 4 — Abstain remains the hardest action.Abstainrecall peaks at 26% in
baseline and falls to 9% and 13.3% in enhanced and v3 respectively. The v3 confusion matrix
shows 87 of 150Abstaincases misclassified asAnswer— the highest single misclassification
cell. Recognising theabsenceof relevant information is structurally harder to gate on than
detecting ambiguity or conflict. Low confidence and coverage scores are necessary but not
sufficient signals for true irrelevance.
Observation 5 — Coverage and accuracy trade off by design.Baseline coverage 67.3%,
enhanced 78.7%, v3 54.0%. The hard gate’s conservative thresholds route more queries to
Ask/Abstain, reducing the fraction of queries the system attempts to answer. For the epistemic
21

Madhav S Baidya PassiveQA
passivity objective, a system that correctly refuses 46% of queries while answering the rest
accurately is preferable to one that attempts everything but hallucinates half of it.
Observation 6 — All three architectures expose the ceiling of inference-time in-
terventions.Even v3 achieves only 38% accuracy and 13.3%Abstainrecall, establishing a
clear upper bound on what can be achieved without model-level adaptation. This motivates
the knowledge graph and planner finetuning approach as the necessary next step: the decision
behaviour must be internalised during training, not imposed post-hoc at inference.
10.2 Planner Observations
Observation 7 — The planner surpasses all RAG baselines on macro F1 even under
compute-constrained training.The finetuned planner achieves 55.6% macro F1 on the
held-out test split, compared to 35.3% for the best inference-time architecture (v3). This 20.3
pp gain confirms the central thesis: decision behaviour must be internalised at training time.
Notably, this result holds despite the planner being trained on only 9,000 samples (26% of the
available 34K finetuning set) for 2 epochs — establishing a conservative lower bound rather than
a ceiling.
Observation 8 — Answer is the most reliably learned action; Ask remains the
hardest.Per-action F1:Answer71.2%,Abstain63.4%,Ask32.6%. TheAskdeficit is
partially explained by sequence length truncation: ASK-action samples have the longest user
turns (multi-turn history plus remaining unknowns), and the 512-token budget disproportionately
truncates the conversation context that distinguishes a clarification-needing query from one that
should be answered directly. TheAbstainresult is the most surprising: recall rises substantially
from 13.3% (v3) to 58.1%, confirming that the graph’s empty-node signal is learnable through
training in a way that inference-time thresholds cannot replicate.
Observation 9 — The single-turn vs. multi-turn performance gap directly impli-
cates the 512-token sequence budget.Single-turn accuracy: 78.4%. Multi-turn accuracy:
25.6%. This 52.8 pp gap is the largest structural finding of the finetuning analysis. Multi-turn
samples require the planner to parse <conversation history> ,<resolved variables> , and
<remaining unknowns> blocks in addition to the base query and graph context — in many cases
exceeding the 512-token limit, causing the history to be silently truncated by the tokeniser. The
planner then operates on a contextually amputated prompt, lacking the referents needed to
resolve elliptical queries. Extending the sequence length to 1,024 or 2,048 tokens is the single
highest-priority hyperparameter change for the full training run.
Observation 10 — Zero unparseable outputs confirms that structured format is
acquired independently of decision quality.Every planner output across the full evaluation
set (5,218 samples) contained well-formed <reasoning> ,<decision> , and <justification>
tags in the correct order. TheAsk-action <clarification question> tag was present and
non-empty in 100% of ASK-routed outputs. This confirms that structured XML output format
is a rapidly learnable surface pattern, separable from the harder semantic problem of choosing
the correct action — an encouraging result for practical deployment, as malformed outputs are
a common failure mode in structured generation.
10.3 Finetuning Analysis
The finetuning run is constrained by compute: 2 epochs on a 9K subsample of the full 34K
dataset. TheINSUFFICIENT VARIABLESfailure mode accounts for 76.6% of training
samples; 23.4% carry theCOMPLETEmode. The 41.3% zero-triple rate means a large fraction
22

Madhav S Baidya PassiveQA
of samples rely on node-presence/absence signal alone; the planner may generalise differently
across sparse versus rich graph contexts. Results should be treated as a proof of concept — the
full training run is the natural next step.
10.4 Qualitative Error Analysis
To ground quantitative results in concrete model behaviour, we examine six representative
predictions — two per action class, one correct and one incorrect — selected to illustrate the
dominant failure modes in the confusion matrix.
Answer cases
✓Answer — CorrectHotpotQA·single-turn
Query:“Which band released the album Silent Alarm in February 2005?”
bloc party|release|silent alarm
silent alarm|release in|february 2005
The planner identifies a complete two-node path from the known entity (Silent Alarm) to
the answer entity (Bloc Party) via a high-weight release relation and correctly routes to
Answer. This represents the ideal operating condition: a short factual query, a known
entity anchoring retrieval, and a direct high-confidence edge in the graph.
×Answer→Ask (misclassified)ContractNLI·single-turn
Query:“Does the contract allow the receiving party to share confidential information with
employees?”
receiving party|disclose to|representatives
receiving party|requires|?unknown 1
Ground truth:Answer(Entailment). The planner fixates on the requires | ?unknown 1
variable node, interpreting its presence as a signal that clarification is needed. This is agraph
context contaminationfailure: the ?var node was injected during Phase 3 from a different
training sample sharing the same KB chunk. Becausereceiving partyis a high-degree hub in
ContractNLI, its variable node adjacency reflects the aggregate of all co-located training
samples rather than the current query. The fix is a query-conditioned triple filter that
suppresses requires edges whose anchor entity similarity to the current query falls below a
threshold.
Ask cases
✓Ask — CorrectShARC·single-turn
Query:“Am I eligible for the pension plan?”
pension plan|require|employment type
pension plan|require|years of service
pension plan|requires|?unknown 1
Missing variables:[employment type,years of service]
The planner names both missing variables in Step 3 of its reasoning and produces the
23

Madhav S Baidya PassiveQA
clarification question“Regarding pension plan: could you specify employment type?”ShARC
evidence chains provide the clearestAsksupervision in the dataset; single-turn ShARC
queries with explicit multi-condition policy snippets represent the easiestAskcases for the
planner.
×Ask→Abstain (misclassified)QuAC·multi-turn·turn 5
Query:“Did she win any awards for that performance?”
priya nair|requires|?unknown 1 [history truncated | 4 prior turns exceed 512-token
budget]
Ground truth:Ask. This is the most common failure mode: 15 of 30Askcases are
misclassified asAbstain. Two factors compound here. First, at turn 5 with four prior
history turns, the full user turn exceeds the 512-token budget; the history block is truncated,
removing the turns that establishsheas Dr. Priya Nair andthat performanceas a 2019
keynote. Without this context, the dangling pronoun triggers the ambiguity heuristic and
biases the planner towardAbstain. Second, the graph contains only a single requires
edge with no relational context — a weak partial-path signal. This case directly motivates
extending the maximum sequence length to 1,024 tokens in the full training run.
Abstain cases
✓Abstain — CorrectContractNLI·single-turn
Query:“Does the contract prohibit the receiving party from reverse engineering any soft-
ware?”
[no relevant nodes found in knowledge graph]
The reverse engineering clause (nda-11) has an 85.8% NotMentioned rate — the highest
of all 17 ContractNLI clause types — and the specific NDA contains no relevant language.
The planner routes cleanly toAbstainwith justification:“graph has no resolvable path —
‘reverse engineering clause’ is entirely absent from the knowledge base.”This is the cleanest
Abstaincase: total topical absence from the KB.
×Abstain→Answer (misclassified)QuAC·single-turn
Query:“What is the GDP of Iceland?”
iceland|locate in|north atlantic
iceland|known as|land of fire and ice
Ground truth:Abstain. The graph contains real nodes for Iceland with non-trivial relational
content, producing a plausible confidence score and suppressing the empty-graphAbstain
trigger. The planner’s Step 2 reasoning identifiesIcelandas a matched node and incorrectly
infers evidential sufficiency — it has not learned that thetypeof information connected to a
node matters: geographical and cultural relations do not support an economic query. The
answer agent subsequently generates a fabricated GDP figure. This is thehallucination-
via-correct-routingfailure mode, where a planner routing error directly enables downstream
hallucination. Addressing this requires either richer node-type labelling in the graph or
24

Madhav S Baidya PassiveQA
negative training examples contrasting entity-present but domain-mismatched queries against
trueAnswercases.
11 Limitations and Future Work
The finetuning dataset builder produces plausible, graph-grounded supervision labels, but label
quality depends on KG extraction quality and GPT-4o-mini variable population accuracy. The
41.3% zero-triple rate means thin reasoning chains for a large fraction of training samples —
these samples can reference matched node names but not relational paths, and the trained
planner may therefore generalise differently on queries where the KG provides rich relational
context versus sparse entity-only context. A full training run over the complete 34K dataset
with more epochs, hyperparameter search, and larger LoRA rank would likely improve decision
accuracy substantially; the baseline finetuning results should therefore be treated as a proof of
concept for the pipeline architecture rather than a performance ceiling.
Future directions include:
•Scaling the finetuning run to the full 34K dataset with hyperparameter optimisation.
•Extending the framework to retrieval over live knowledge bases with dynamic graph updates.
•Investigating reinforcement learning from human feedback to alignAsk/Abstainbehaviour
with user preferences.
•Evaluating on domain-specific benchmarks (medical, legal, financial) where epistemic passivity
has the highest stakes.
•Studying the multi-turn convergence properties of the variable-tracking mechanism across
longer dialogue chains.
12 Conclusion
We have presented PassiveQA, a three-action decision framework for epistemically calibrated
question answering. Through systematic experiments across three RAG architectures, we
established that inference-time retrieval improvements are insufficient to instil epistemic passivity
— the decision behaviour must be internalised at training time. Our knowledge graph construction
procedure, which encodes both semantic validity and three-action behavioural supervision into
edge weights, provides a principled foundation for graph-grounded planner training. The
finetuned Mistral-7B-Instruct-v0.3 planner, trained on structured reasoning chains derived
from graph evidence, represents a proof-of-concept for training-time alignment of epistemic
decision-making. The three-agent execution architecture — in which the planner routes and
specialised agents execute, with failures independently diagnosable as planner errors or agent
errors — provides a clean separation of concerns that simplifies both debugging and targeted
improvement of individual components. We hope this work motivates further research into
decision-aware QA systems that know not just how to answer, but when to answer — and when
not to. We emphasise that the presented results should be interpreted as indicative of broader
trends rather than definitive benchmarks, and that further large-scale validation remains an
important direction for future work.
Reproducibility.All code, dataset construction scripts, knowledge graph artefacts, finetuning
data, and model training configurations are released at https://github.com/MadsDoodle/
PassiveQA to support reproduction and extension of the results reported here. Should any
ambiguity remain in the implementation details, readers are warmly encouraged to open an issue
on the repository — all issues will be responded to promptly.
25

Madhav S Baidya PassiveQA
Acknowledgements
The author thanks the open-source communities behind HuggingFace Transformers, FAISS,
spaCy, and the dataset creators of ShARC, QuAC, HotpotQA, and ContractNLI.
A System and Agent Prompts
This appendix reproduces the exact prompts used in the PassiveQA pipeline. All prompts are
passed as user-turn messages via the Mistral instruct chat template ([INST]...[/INST]) with
the appropriate role prefix. Placeholders enclosed in {braces} are filled at runtime from the
structured information stateS(q).
A.1 Planner System Prompt
The system prompt below is prepended to every planner call and defines the decision logic, input
tags, and required output format. It is the primary mechanism through which the three-action
routing behaviour is expressed to the model at inference time.
Planner — System Prompt (passed asrole: system)
You are a decision planner for a question-answering system.
Your task: given a user query, search the knowledge graph for relevant
nodes, evaluate what information is present and what is missing, then
decide the correct action.
Decision logic:
- Search the graph for nodes matching the query subject and known variables
- If the graph contains a complete path connecting known entities→ANSWER
- If the graph contains the topic but key linking variables are missing
→ASK (specify what is missing)
- If the graph has no relevant nodes or the topic is absent→ABSTAIN
You will receive:
<query> | the user’s question
<known_variables> | entities explicitly present in the query
<graph_context> | KG triples: subject | relation | object
<missing_variables> | variables required but not present
<conversation_history>| prior turns (multi-turn queries only)
Output format (strictly follow this):
<reasoning>
Step 1 | Query subject: identify what the query is asking about
Step 2 | Graph search: what nodes were found, what connections exist
Step 3 | Variable check: what is known, what is missing
Step 4 | Decision rationale: why this action is correct
</reasoning>
<decision>
ANSWER | ASK | ABSTAIN
</decision>
<justification>
One sentence grounded in the graph evidence.
</justification>
Rules:
- Reasoning must reference actual graph content, not generic statements
26

Madhav S Baidya PassiveQA
- Never say "unspecified variables" | name the specific missing variable
- If graph_context is empty, default to ABSTAIN unless context is clearly
partial (then ASK)
- Do not use prior world knowledge | only the graph context provided
A.2 Answer Agent Prompt
The Answer Agent is invoked after the planner routes toAnswer. Retrieved chunks (from
hybrid BM25 + dense retrieval, reranked by cross-encoder) are injected as {context block} .
Conversation history from the last 3 turns is injected as{history block}when present.
Answer Agent — User Prompt Template
You are a knowledgeable assistant. Answer the query using ONLY the
provided context. Be concise and factual.
{history_block}
Query: {query}
Context:
[Source 1 | {source} | {granularity}]
{chunk_text_1}
[Source 2 | {source} | {granularity}]
{chunk_text_2}
...
Answer:
The agent receives no system-level instruction to decidewhetherto answer — that decision has
already been made by the planner. Removing the decision burden from the generation step
reduces hallucination pressure and simplifies the generation objective to strict context-grounded
answering.
A.3 Ask Agent Prompt
The Ask Agent constructs a single, well-formed clarification question. The missing variable string
is first cleaned of meta-prefixes and matched to a known anchor entity via SBERT similarity;
the anchor populates {anchor} . Graph entities extracted from the top-5 KG triples populate
{known context}.
Ask Agent — User Prompt Template
Ask ONE focused clarification question to help answer the user’s query.
{history_block}
User query: {query}
Missing information: {missing_str}
Known context: {known_context}
Rules:
- Ask exactly ONE question ending with ?
- Be specific about what is missing
- Reference the query topic directly
Clarification question:
27

Madhav S Baidya PassiveQA
Post-processing enforces a trailing ?if absent, and a fallback template“Regarding {anchor }:
could you specify {missing variable }?”is used when the model output is under 10 characters.
100% ofAskoutputs in evaluation were properly formed questions.
A.4 Abstain Agent Prompt
The Abstain Agent generates a specific, honest refusal. The {reason} field is populated
deterministically from three branching conditions evaluated before the LLM call: (a) query too
vague with no known variables; (b) graph has content but no connection to the specific question;
(c) required variables are entirely absent from the KB. The {history note} field lists variables
already resolved in prior turns.
Abstain Agent — User Prompt Template
You cannot answer the following query from the available knowledge base.
Write a brief, honest refusal. Explain why you cannot answer.
Do NOT make up information.
{history_note}
Query: {query}
Reason: {reason}
Your response:
A minimum-length guard (20 characters) triggers a fallback:“I’m unable to answer this query.
{reason }You may want to consult a specialised source.”The Abstain Agent explicitly distin-
guishes between two causes of abstention: (a) the query topic is entirely absent from the KB, and
(b) the topic is present but the required information is not recoverable through user clarification,
communicating meaningfully to the user rather than issuing a generic refusal.
A.5 Annotated Training Example
The following is a representativeAsk-action multi-turn training sample from the finetuning
dataset, abbreviated for space. It illustrates the full prompt schema and the anchor-based
clarification question construction described in§7.
Finetuning Dataset — Annotated Ask Sample (QuAC, multi-turn)
[User turn]
<conversation_history>
Turn 1 | ASK | Q: "When was the album released?"
| A: [Clarification requested: ’release date’ is needed]
→resolved: ’release date’
</conversation_history>
<resolved_variables> release date </resolved_variables>
<remaining_unknowns> sales performance </remaining_unknowns>
<query> How did the album sell? </query>
<known_variables> album, release date </known_variables>
<graph_context>
best album of | requires | ?unknown_1
albums for the | requires | ?unknown_2
</graph_context>
<missing_variables> sales performance </missing_variables>
[Assistant turn | supervision label]
<reasoning>
Step 1 | Query subject: album, release date.
28

Madhav S Baidya PassiveQA
Query asks: ’How did the album sell?’
Step 2 | Graph search: matched KG nodes: ’album’.
Relations seen: requires. Variable placeholder nodes
indicate missing information; path cannot be completed.
Step 3 | Variable check: Known: album, release date.
Required but absent from graph: ’sales performance’.
Failure mode: INSUFFICIENT_VARIABLES.
Step 4 | Decision rationale: graph has partial connections but
cannot complete the reasoning path without: ’sales performance’.
</reasoning>
<decision> ASK </decision>
<justification>
Regarding album: could you specify sales performance?
</justification>
<clarification_question>
Regarding album: could you specify sales performance?
</clarification_question>
The clarification question is generated by matching the missing variable string (sales performance)
against known variables via SBERT similarity (threshold 0.20) to find the anchor entity (album),
then applying the template“Regarding [anchor]: could you specify [missing]?”
A.6 Unified Dataset JSON Schema
The schema below defines the canonical record format used for all four data sources after
merging. Every field is present for all sources; source-specific sub-fields inside source specific
are populated only where applicable and set tonullotherwise.
Unified Dataset Schema (JSON)
{
"id": "string",
"query": "string",
"context": {
"documents": [
{
"doc_id": "string",
"text": "string",
"url": "string",
"file_name": "string",
"chunk_idx": "int",
"total_chunks": "int",
"spans": ["[int, int]"]
}
]
},
"state": {
"known_variables": ["string"],
"missing_variables": ["string"],
"constraints": ["string"],
"failure_mode":
"COMPLETE | INSUFFICIENT_VARIABLES | MULTI_HOP_REQUIRED",
"difficulty": "easy | medium | hard | very_hard",
"completeness": "complete | partial | incomplete"
},
"action": "ANSWER | ASK | ABSTAIN",
"response": "string",
29

Madhav S Baidya PassiveQA
"metadata": {
"source": "quac | sharc | hotpotqa | contract_nli",
"multi_turn": "bool",
"turn_id": "int | null",
"dialogue_id": "string | null",
"requires_reasoning": "bool",
"num_missing_variables": "int",
"variable_types": ["string"],
"source_specific": {
// ShARC
"sharc_answer": "Yes | No | Follow-on | Irrelevant",
"evidence_depth": "int",
"history_depth": "int",
"utterance_id": "string",
// HotpotQA
"question_type": "bridge | comparison",
"level": "easy | medium | hard",
"num_supporting_facts": "int",
// ContractNLI
"nli_choice": "Entailment | Contradiction | NotMentioned",
"label_id": "string",
"num_spans": "int",
// QuAC
"yesno": "y | n | x",
"followup_flag":"y | n | m"
}
}
}
Key field semantics.The state object is the core contribution of the schema: known variables
andmissing variables drive both knowledge graph traversal and planner finetuning supervi-
sion. The failure mode field encodes why a query cannot be answered as-is — COMPLETE
samples always have an empty missing variables list and map exclusively toAnswer;
INSUFFICIENT VARIABLES samples cover allAskand mostAbstaincases; MULTI HOPREQUIRED
flags HotpotQA bridge queries where multi-document evidence is mandatory. The source specific
block preserves original dataset annotations so the unified schema remains lossless with respect
to each source’s native label space.
References
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., K¨ uttler, H., Lewis, M.,
Yih, W.-t., Rockt¨ aschel, T., Riedel, S., and Kiela, D. (2020). Retrieval-augmented generation
for knowledge-intensive NLP tasks. InAdvances in Neural Information Processing Systems
(NeurIPS), volume 33, pages 9459–9474.
Karpukhin, V., O˘ guz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., and Yih, W.-t.
(2020). Dense passage retrieval for open-domain question answering. InProceedings of EMNLP
2020, pages 6769–6781.
Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M.-W. (2020). REALM: Retrieval-augmented
language model pre-training. InProceedings of ICML 2020.
Izacard, G. and Grave, E. (2021). Leveraging passage retrieval with generative models for open
domain question answering. InProceedings of EACL 2021, pages 874–880.
30

Madhav S Baidya PassiveQA
Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., van den Driessche,
G., Lespiau, J.-B., Damoc, B., Clark, A., et al. (2022). Improving language models by
retrieving from trillions of tokens. InProceedings of ICML 2022, pages 2206–2240.
Robertson, S. and Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and
beyond.Foundations and Trends in Information Retrieval, 3(4):333–389.
Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., and Zaharia, M. (2022). ColBERTv2:
Effective and efficient retrieval via lightweight late interaction. InProceedings of NAACL
2022, pages 3715–3734.
Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M., and Wang, H.
(2023). Retrieval-augmented generation for large language models: A survey.arXiv preprint
arXiv:2312.10997.
Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y. J., Madotto, A., and Fung,
P. (2023). Survey of hallucination in natural language generation.ACM Computing Surveys,
55(12):1–38.
Maynez, J., Narayan, S., Bohnet, B., and McDonald, R. (2020). On faithfulness and factuality
in abstractive summarization. InProceedings of ACL 2020, pages 1906–1919.
Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., Chen, Q., Peng, W., Feng, X.,
Qin, B., and Liu, T. (2025). Alleviating hallucinations from knowledge misalignment in large
language models via selective abstention learning. InProceedings of ACL 2025 (Long Papers).
Goswami, S. and Kurra, S. (2025). HALT-RAG: A task-adaptable framework for hallucination
detection with calibrated NLI ensembles and abstention.arXiv preprint arXiv:2509.07475.
Li, Y., Fu, X., Verma, G., Buitelaar, P., and Liu, M. (2025). Mitigating hallucination in large
language models: An application-oriented survey on RAG, reasoning, and agentic systems.
arXiv preprint arXiv:2510.24476.
Geifman, Y. and El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks In
Advances in Neural Information Processing Systems (NeurIPS), pages 6327–6338.
Xin, J., Tang, R., Yu, Y., and Lin, J. (2021). The art of abstention: Selective prediction and
error regularization for natural language processing. InProceedings of ACL-IJCNLP 2021,
pages 1040–1051.
Varshney, N. and Baral, C. (2023). Post-abstention: Towards reliably re-attempting the abstained
instances in QA.arXiv preprint arXiv:2305.01812.
Yin, Z., Sun, Q., Guo, Q., Wu, J., Qiu, X., and Huang, X. (2024). Do large language models
know what they don’t know? InFindings of ACL 2024.
Phillips, E., Gustafsson, F. K., Wu, S., Thakur, A., and Clifton, D. A. (2026). Entropy alone is
insufficient for safe selective prediction in LLMs.arXiv preprint arXiv:2603.21172.
Feng, S., Shi, W., Wang, Y., Ding, W., Balachandran, V., and Tsvetkov, Y. (2024). Don’t
hallucinate, abstain: Identifying LLM knowledge gaps via multi-LLM collaboration. In
Proceedings of ACL 2024.arXiv preprint arXiv:2402.00367.
Saeidi, M., Bartolo, M., Lewis, P., Singh, S., Rockt¨ aschel, T., Sheldon, M., Bouchard, G., and
Riedel, S. (2018). Interpretation of natural language rules in conversational machine reading.
InProceedings of EMNLP 2018, pages 2087–2097.
31

Madhav S Baidya PassiveQA
Choi, E., He, H., Iyyer, M., Yatskar, M., Yih, W.-t., Choi, Y., Liang, P., and Zettlemoyer,
L. (2018). QuAC: Question answering in context. InProceedings of EMNLP 2018, pages
2174–2184.
Aliannejadi, M., Zamani, H., Crestani, F., and Croft, W. B. (2019). Asking clarifying questions
in open-domain information-seeking conversations. InProceedings of SIGIR 2019, pages
475–484.
Rao, S. and Daum´ e III, H. (2018). Learning to ask good questions: Ranking clarification
questions using neural expected value of perfect information. InProceedings of ACL 2018,
pages 2737–2746.
Zhou, S., Ouyang, S., Zhang, Z., and Zhao, H. (2023). Towards end-to-end open conversational
machine reading. InFindings of the Association for Computational Linguistics: EACL 2023,
pages 2064–2076, Dubrovnik, Croatia. Association for Computational Linguistics.
Yasunaga, M., Ren, H., Bosselut, A., Liang, P., and Leskovec, J. (2021). QA-GNN: Reasoning
with language models and knowledge graphs for question answering. InProceedings of
NAACL-HLT 2021, pages 535–546.
Sun, H., Dhingra, B., Zaheer, M., Mazaitis, K., Salakhutdinov, R., and Cohen, W. W. (2019).
Open domain question answering using early fusion of knowledge bases and text. InProceedings
of EMNLP-IJCNLP 2019, pages 4231–4242.
Lin, B. Y., Chen, X., Chen, J., and Ren, X. (2019). KagNet: Knowledge-aware graph networks
for commonsense reasoning. InProceedings of EMNLP-IJCNLP 2019, pages 2829–2839.
Jiang, P., Ouyang, S., Jiao, Y., Zhong, M., Tian, R., and Han, J. (2025). A survey on
retrieval and structuring augmented generation with large language models.arXiv preprint
arXiv:2509.10697.
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2021).
LoRA: Low-rank adaptation of large language models.arXiv preprint arXiv:2106.09685.
Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. (2023). QLoRA: Efficient
finetuning of quantized LLMs.arXiv preprint arXiv:2305.14314.
Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal,
S., Slama, K., Ray, A., et al. (2022). Training language models to follow instructions with
human feedback. InAdvances in Neural Information Processing Systems (NeurIPS), volume 35,
pages 27730–27744.
Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. (2023). Direct
preference optimization: Your language model is secretly a reward model. InAdvances in
Neural Information Processing Systems (NeurIPS), volume 36.
Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de las Casas, D.,
Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. (2023). Mistral 7B.arXiv preprint
arXiv:2310.06825.
Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., and Manning,
C. D. (2018). HotpotQA: A dataset for diverse, explainable multi-hop question answering. In
Proceedings of EMNLP 2018, pages 2369–2380.
Koreeda, Y. and Manning, C. D. (2021). ContractNLI: A dataset for document-level natural
language inference for contracts. InFindings of EMNLP 2021, pages 1907–1919.
32
# Goal-Oriented Reasoning for RAG-based Memory in Conversational Agentic LLM Systems

**Authors**: Jiazhou Liang, Armin Toroghi, Yifan Simon Liu, Faeze Moradi Kalarde, Liam Gallagher, Scott Sanner

**Published**: 2026-05-12 14:51:02

**PDF URL**: [https://arxiv.org/pdf/2605.12213v1](https://arxiv.org/pdf/2605.12213v1)

## Abstract
LLM-based conversational AI agents struggle to maintain coherent behavior over long horizons due to limited context. While RAG-based approaches are increasingly adopted to overcome this limitation by storing interactions in external memory modules and performing retrieval from them, their effectiveness in answering challenging questions (e.g., multi-hop, commonsense) ultimately depends on the agent's ability to reason over the retrieved information. However, existing methods typically retrieve memory based on semantic similarity to the raw user utterance, which lacks explicit reasoning about missing intermediate facts and often returns evidence that is irrelevant or insufficient for grounded reasoning. In this work, we introduce Goal-Mem, a goal-oriented reasoning framework for RAG-based agentic memory that performs explicit backward chaining from the user's utterance as a goal. Rather than progressively expanding from retrieved context, Goal-Mem decomposes each goal into atomic subgoals, performs targeted memory retrieval to satisfy each subgoal, and iteratively identifies what information from memory should be retrieved when intermediate goals cannot be resolved. We formalize this process in Natural Language Logic, a logical system that combines the verifiability of reasoning provided by FOL with the expressivity of natural language. Through extensive experiments on two datasets and comparing to nine strong memory baselines, we show that Goal-Mem consistently improves performance, particularly on tasks requiring multi-hop reasoning and implicit inference.

## Full Text


<!-- PDF content starts -->

Goal-Oriented Reasoning for RAG-based Memory in
Conversational Agentic LLM Systems
Jiazhou Liang∗
University of Toronto
joe.liang@mail.utoronto.caArmin Toroghi∗
University of Toronto
armin.toroghi@mail.utoronto.ca
Yifan Simon Liu
University of TorontoFaeze Moradi Kalarde
University of TorontoLiam Gallagher
University of Toronto
Scott Sanner
University of Toronto
Vector Institute for Artificial Intelligence
Abstract
LLM-based conversational AI agents struggle to maintain coherent behavior over
long horizons due to limited context. While RAG-based approaches are increasingly
adopted to overcome this limitation by storing interactions in external memory
modules and performing retrieval from them, their effectiveness in answering
challenging questions (e.g., multi-hop, commonsense, etc.) ultimately depends
on the agent’s ability to reason over the retrieved information. However, existing
methods typically retrieve memory by semantic similarity to the raw user utterance,
which lacks explicit reasoning over missing intermediate facts and often returns
evidence that is irrelevant or insufficient for grounded reasoning. In this work,
we introduceGOAL-MEM, a goal-oriented reasoning framework for RAG-based
agentic memory that performs explicit backward chaining from the user’s utterance
as a goal. Rather than progressively expanding from retrieved context, GOAL-MEM
decomposes each goal into atomic subgoals, performs targeted memory retrieval
to satisfy each subgoal, and iteratively identifies what information from memory
should be retrieved when intermediate goals cannot be resolved. We formalize this
process in Natural Language Logic, a logical system that combines the verifiability
of reasoning provided by FOL with the expressivity of natural language. Through
extensive experiments on two datasets, comparing to eight strong memory baselines,
we show that GOAL-MEMconsistently improves performance, particularly on tasks
requiring multi-hop reasoning and implicit inference.
1 Introduction
Large Language Models (LLMs) are increasingly deployed as conversational and agentic systems
for complex, long-horizon tasks [ 21,36,9,19,15]. Yet their ability to provide coherent responses
across long interaction horizons remains fundamentally constrained by finite context windows and
the difficulty of reliably using information across long contexts [ 3,16]. Recent work augments the
conversational agentic LLM systems with explicitexternal long-term memory: a persistent store that
encodes information from prior interactions and makes it available for later retrieval [ 37,32]. Within
this line of work,retrieval-augmented generation (RAG)-based memory stores detailed records of
∗Equal contribution.
40th Conference on Neural Information Processing Systems (NeurIPS 2026).arXiv:2605.12213v1  [cs.AI]  12 May 2026

Figure 1: Comparison of retrieval from external memory: utterance-based semantic retrieval (left),
forward reasoning (middle), and GOAL-MEM(right). Utterance-based semantic retrieval and forward
reasoning are easily misled by irrelevant memory traces. GOAL-MEMdecomposes the utterance into
subgoals and performs targeted retrieval for memory-grounded verifiable response generation.
past interactions as episodic memory traces in an external memory store, retrieves only a relevant
subset at inference time using the user’s utterance as a query, and conditions response generation on
the retrieved traces. These approaches support responses that remain coherent with past interactions
while keeping the input within a feasible context window.
Several works have focused on more effective encoding, organizing, and retrieving episodic traces,
such as Mem0 [ 4], MAGMA [ 10], A-MEM [ 33], and MemTree [ 24]. However, a key question remains
underexplored:which queries should the agent form to retrieve the right memory trace?. Existing
methods merely focus on the surface form of the utterance, relying on semantic similarity through
global dense retrieval over stored memory traces [ 27] or traversal heuristics over the predefined
structures [ 33,10]. However, to satisfy the goal of generating appropriate responses, conversational
agentic systems require substantial reasoning over the memory [ 1], and the information needed to
support such reasoning is not always semantically adjacent to the utterance. Thus, existing approaches
often fail on multi-hop questions that require chaining intermediate queries across memory entries.
Reasoning over external long-term memory poses several challenges. The agent must correctly
identify intermediate queries to recover the required memory traces across multi-hop steps [ 30,18]
and ground the final response in retrieved memory facts rather than relying on the model’s parametric
knowledge [ 22]. These challenges make many popular forward reasoning strategies, such as self-
reflection and query reformulation, brittle: early irrelevant retrievals or unsupported inferences can
propagate across steps, causing the agent to accumulate facts without converging on the correct set of
memory traces, as illustrated in Figure 1 (middle). Such approaches also lack an explicit verification
mechanism to ensure that the final answer is grounded in retrieved memory facts.
To address this, we propose GOAL-MEM:Goal-Oriented Reasoning for RAG-basedMemory, a
framework rooted in the classical principle of backward chaining [ 7], which can be applied to arbitrary
memory backbones. Instead of expanding outward from the utterance and the memory retrieved so
far, GOAL-MEMtreats the user’s question as the goal and decomposes a set of intermediate subgoals
whose satisfaction is necessary to establish the goal (cf. right of Figure 1), and retrieves memory
traces targeted at each subgoal. If the retrieved memory is insufficient to resolve any subgoals, the
agent then recursively derives their antecedents as queries, until the original goal can be answered. To
make this process verifiable, we represent both goals and memory traces in typedNatural Language
Logic(NL-Logic). NL-Logic retains a language-like expressiveness since predicates and types remain
NL phrases, while inheriting the structural guarantees of First-Order Logic (FOL) through typed
variables and explicit substitutions. As a key component of NL-Logic, we introduce averifiable
unificationfor memory where the subgoal is grounded only when the retrieved memory facts are
2

type-consistent and logically entailed via LLM-judged entailment over NL predicates, ruling out the
spurious pure semantic matches. In summary, our contributions are as follows:
•We proposeGOAL-MEM, a novel goal-oriented reasoning framework for RAG-based
memory with NL-Logic. GOAL-MEMperforms backward chaining from the user utterance
and retrieves only the goal-relevant facts with a verifiable response generation.
•We show that GOAL-MEMis structure-agnostic and can be seamlessly applied to popular
RAG-based memory backbones, yielding substantial empirical improvements.
•We compare GOAL-MEMagainst popular forward reasoning approaches using the same
memory backbones and show that GOAL-MEMconsistently yields larger gains.
2 Related Work
Retrieval-Augmented Generation (RAG) Based MemoryRAG-based memory retrieves a relevant
subset of stored conversational traces or structured records at inference time. This paradigm has been
widely studied in recent LLM-agent memory systems, where prior interactions are stored externally
and selectively recalled to support long-term or task-grounded responses [ 37]. We therefore focus on
representative retrieval-augmented memory backbones that cover the major design choices in this
line of work. Early work like MemoryBank introduced updatable long-term memory with forgetting
mechanisms [ 39]. More recent systems focus on advanced structural organization and routing
heuristics. For relational structures, Mem0 [ 4] and A-MEM [ 33] utilize dynamic graphs and linked
notes, concepts further advanced by agentic graph routing in AdaMem [ 34] and causal tool retrieval
in AMA-Agent [ 38]. To manage abstraction, memory is increasingly structured hierarchically;
MemTree [ 24] and RAPTOR [ 27] construct multi-level trees. Furthermore, methods like MAGMA
perform query-adaptive traversal across multi-view graphs [ 10]. However, GOAL-MEMis orthogonal
to all these works. Whereas prior methods primarily focus on memory structure and retrieval, the
problem is addressing what information should be queried for the utterance. GOAL-MEMserves as a
structure-agnostic reasoning layer that can be flexibly integrated with these memory backbones to
enforce goal-oriented verifiable reasoning.
LLM Reasoning via Backward ChainingBackward chaining is an established paradigm in logical
reasoning. Unlike forward reasoning approaches such as query reformulation [ 20], self-reflection
[35,28], iterative retrieval with LLM reranking [ 13,17], and memory-specific methods such as
MemGuide [ 5] that aim to derive the goal from observed facts progressively, backward chaining works
backward from the goal, avoiding exhaustive exploration of facts and thereby yielding substantial
efficiency gains [ 26,2]. This goal-directed approach has recently been used in LLM-based reasoning.
LAMBADA [ 12] pioneered this integration by prompting the LLM to emulate the backward-chaining
process by decomposing complex goals into sub-goals. Symba [ 14] leverages a symbolic solver
that undergoes backward chaining and calls the LLM to provide additional facts. LLM-TRes [ 29]
employs theory resolution to unify LLM-derived commonsense axioms into backward chaining.
Despite the collective success of these works in enhancing LLM reasoning efficiency, they remain
constrained by a reliance on static knowledge, and to our knowledge, no research has extended the
potential of backward LLM reasoning to the dynamic domain of memory-augmented agents, where
backward chaining must be tightly coupled with adaptive memory and continual context updates.
3 Preliminaries
Retrieval-Augmented Generation (RAG) Based MemoryWe consider a conversation as a
sequence of turns. At turn t, the user provides an utterance ut, and the agent generates a response at.
Let[(u1, a1), . . . ,(u t−1, at−1)]denote thedialogue historyprior to turn t. A RAG-based memory-
augmented LLM is additionally equipped with an external long-term memory Mthat stores episodic
traces distilled from dialogue history.
A RAG-based memory backbone Bconsists of three components: (1) a persistent external long-term
memory store M, instantiated as a flat vector database, graph-structured memory, or tree-structured
memory; (2) a retrieval function Bretrieve (qt,M) that returns memory facts from Mgiven a query
qt; and (3) an update mechanism to encode information in utintoM. Given the utterance ut,
the agent first constructs one or more queries qtthat encapsulate the information needs in ut.
3

Each query is passed to the memory backbone B, which retrieves relevant memory facts Mt:
Mt=B retrieve (qt,M). The agent then generates a response conditioned on both the current utterance
and the retrieved fact,a t=fLLM(ut,Mt). The backbone updatesMwith the new turn.
As noted in section 2, choices on what information in dialogue is encoded, how it is organized and
updated, and how queries are matched to traces in Mare made by Band fall outside the scope of our
work. We focus on constructing a sequence of queriesq tvia goal-oriented reasoning.
Natural Language Logic (NL-Logic)While conversation and memories are inherently expressed
in NL, using LLMs for reasoning over pure NL induces the risk of generating ungrounded or
hallucinated content. Moreover, NL-based reasoning is often difficult to verify, as the intermediate
steps leading to a conclusion are not explicitly structured. Formal logical systems such as first-order
logic (FOL) [ 26] provide a structured framework with explicit variables, substitutions, and entailment
relations, enabling transparent and verifiable reasoning. However, its reliance on predefined symbols
limits its expressiveness for memory traces, motivating the need for a hybrid logical NL representation.
To combine the strengths of both paradigms, we extend FOL into NL-Logic, where predicates,
objects, and relations are expressed directly in NL, preserving logical structure while maintaining
expressiveness. Let Edenote the constants appearing in the current utterance or retrieved memory
facts,Va set of variables, and Tthe set of NL type descriptions. A typed term is written as t:τ,
where t∈ E ∪ V andτ∈ T . Unlike conventional typed logical systems, we do not assume that Tis
a fixed, predefined type vocabulary. Instead, types are open-ended, on-demand NL phrases and type
checking is performed by the LLM through NL entailment. This allows more flexible definitions.
In NL-Logic, an atomic formula is R(t1, . . . , t n),where Ris an NL relational phrase and each
argument tiis explicitly typed (e.g., “(Alice:user)likes(x:flavor).” ). Type constraints are
enforced on variables: x:τ⇒x∈ E τmeaning that xbelongs to Eτ, the subset of constants in E
with type τ. Type specification for constants (e.g.,“Alice”,“Matcha”) is optional, as their names
implicitly encode their semantic type. Thus, an atomic formula in NL-Logic is represented as a
structured NL statement with typed arguments, rather than as a purely symbolic predicate application.
In this logical system, a knowledge base (e.g. Memory) can be represented as a set of ground atoms:
M={R(e 1, . . . , e n)|(e 1, . . . , e n)∈ En}(1)
4 Methodology
Given a user utterance utat turn t, the objective is to generate a grounded response. GOAL-MEMfirst
maps utto a formal NL-Logic goal ϕut(x1, . . . , x m), which captures both the semantic intent of the
utterance and the set of variables {x1, . . . , x m}that must be instantiated to satisfy the user’s request.
Running Example 1.
User utterance:u t=“Alice: Which drink did I order at cafe Momoco?”
NL-Logic goal:ϕ ut=“(x:drink) ordered by (Alice:human) at (Momoco:cafe).”
Answering the user requires resolving variable x, whose type isdrink. The response-generation
problem thus reduces to a logical inference: finding a substitution θ={x 1/e1, . . . , x m/em}, where
each variablex iis replaced by a type-consistent constante idrawn from the retrieved memoryM t.
Running Example 1 (continued).
Mt={“(Alice) ordered (Kyoto Latte) at cafe Momoco.”},θ={(x:drink)/(Kyoto Latte)}
This formulation grounds the response in facts in retrieved memory, mitigating hallucinations that
arise when existing approaches lack explicit verification on what fact should support the answer.
4.1 Goal-Oriented Backward Chaining
In practice, a direct substitution that satisfies ϕutusingMtrarely exists when ϕutrequires multi-hop
reasoning or open-domain knowledge. In these cases, GOAL-MEMapplies backward chaining to
decompose ϕutinto a series of subgoals, each of which can be independently grounded against Mt.
As illustrated in Figure 2, GOAL-MEMoperates through two nested iterative procedures. In each
breadth loop, GOAL-MEMdecomposes the user utterance utinto a set of atomic subgoals ψband
retrieves the corresponding memory from the backbone Bfor each subgoal separately, yielding the
4

Figure 2: Overview of GOAL-MEM. The framework starts from the user utterance and goal initial-
ization (top), decomposes the goal into NL-Logic subgoals for memory retrieval from a selected
backbone (middle), and checks whether the retrieved memory grounds all subgoals through unifica-
tion. If not, it enters the depth loop (middle), identifying new subgoals with targeted retrieval until
all variables have been substituted or the maximum depth (bottom left) is reached. The failed goal
decomposition would then be provided to future decompositions to prevent a duplicate approach.
aggregated retrieved memory Mt. The goal is satisfied when all ψbare grounded by Mtunder a
type-consistent non-conflicting substitution and all variables required by the goal are bound. If not,
GOAL-MEMrefines each unresolved subgoal by inferring an intermediate subgoal whose satisfaction
is necessary to establish the unresolved subgoal, and retrieves memory facts from B. Thisdepth loop
repeats until either every variable is successfully grounded or a maximum recursion depth is reached.
We initialize the backward chaining with an empty substitutionθ(0)=∅and subgoal setψ(0)=∅.
Goal Decomposition.At each breadth step b∈ {1, . . . , B max}, the agent maintains the current set
of subgoals ψ(b), the retrieved memory M(b)
t, and the current substitution θ(b). Given the original
goalϕ ut, GOAL-MEMdecomposes the goal into a set of candidate subgoals:
ψ(b)= ΨLLM
decomp
ϕut,{ψ(0), . . . , ψ(b−1)}
,(2)
where ΨLLM
decomp is an LLM-based decomposition function that avoids repeating earlier decompositions
(i.e.,{ψ(0), . . . , ψ(b−1)}) that led to failed subgoals and instead proposes a distinct way to satisfy the
goal. ΨLLM
decomp induces a rule in NL-Logic of the form: ψ1∧ ··· ∧ψ k⇒ϕ ut,where each ψi∈ψ(b)
is an atomic NL-Logic formula that must hold forϕ utto be satisfied.
Running Example 2.
ut=“Alice: I like the cafe I went to last week, what drink should I try this time?”
ϕut=“Recommend (x:drink) to (Alice) from the menu of the cafe she went to last week.”,
The LLM may generate the following rule:
“(Alice) likes (y:flavor)”∧“(x:drink) contains (y:flavor)”∧“(x:drink) served in (z:cafe visited last week)”
⇒“recommend (x:drink) to (Alice).”
Targeted Memory Retrieval.Each subgoal ψi∈ψ(b), an atomic NL-Logic formula, is issued as a
query to the memory backboneBto retrieve the corresponding facts:
M(b)
t=[
ψi∈ψ(b)Bretrieve (ψi,M).(3)
5

Running Example 2 (continued).
For the subgoalψ i=“(x:drink) contains (y:flavor)”,the memory backbone retrieves
Bretrieve(ψi,M) ={“(Kyoto Latte) with (Matcha Powder)”}.
4.2 Verifiable Unification
Given the subgoals ψ(b)and the aggregated retrieved memory M(b)
t, the agent grounds each subgoal
by assessing whether the retrieved memories provide sufficient facts to satisfy it. Specifically, for
each subgoal ψi∈ψ(b), the agent attempts to unify it with a retrieved memory fact mj∈ M(b)
t
conditioned on the current substitutionθ(b):
δi= UNIFY(ψ i, mj|θ(b))whereSUBST(θ(b)∪δi, ψi) =LLMSUBST(θ(b)∪δi, mj).(4)
Here, δidenotes the candidate substitution induced by mjforψi, and SUBST(δ i, p)denotes the
formula obtained by applying substitutionδ itop.
Running Example 2 (continued).Recall a previous subgoalψ iand a retrieved memorym j∈ M(b)
t:
ψi=“(x:drink) contains (y:flavor)”.
mj=“The cafe advertised a (Kyoto Latte) with (Matcha Powder:ingredient)”.
To determine whetherψ ican be grounded bym j, we computeUNIFY(ψ i, mj|θ(b)).
UNIFY(ψ i, mj|θ(b))proposes a variable substitution: δi={x /“Kyoto Latte”, y /“Matcha Powder”}.
Unification succeeds only if three conditions hold: (i) the proposed substitution satisfies type
consistency; (ii) the proposed substitution is equivalent to the existing substitution θ(b); and (iii)
the retrieved memory fact mjlogically entails the grounded subgoal. Because NL-Logic types and
relations are expressed in natural language, these checks are performed by an LLM.
Type Consistency.A variable (x:τ) in the subgoal ψican be substituted with a constant e′from a
retrieved memory fact mjif and only if their types are consistent. If e′is explicitly typed as (e′:τ′),
we requireτ′⊢LLMτ; otherwise, the entity must be an instance of the type, i.e.,τ LLM(e′).
Running Example 2 (continued).
Since “drink” LLM(“Kyoto Latte”)and “ingredient”⊢ LLM “flavor”, the substitutionδ iis type-compatible.
Equality Checks with Existing Substitutions.If the substitution assigns a variable already bound
inθ(b), the assigned constants must be equal in the given context. Formally, let dom(θ) denote the
set of variables bound by a substitution θ. For every variable x∈dom(δ i)∩dom(θ(b)), the two
bindings must agree under the LLM-judged equality relation= LLM:
δi(x) = LLMθ(b)(x),∀x∈dom(δ i)∩dom(θ(b)).(5)
Running Example 2 (continued).
Letθ(b)={y /“Matcha”} be an existing substitution. Since y∈dom(δ i)∩dom(θ(b))and
“Matcha Powder”= LLM “Matcha” in the given context,δ+
iis accepted under this condition.
In contrast, ifδ−
i={y /“Strawberry”}, then “Strawberry”̸= LLM “Matcha”, the substitution is rejected.
Logical Entailment.We apply the type-consistent and substitution-consistent θ(b)∪δito obtain
the grounded subgoal ψg
i=ψi[θ(b)∪δi]. The LLM then checks whether any retrieved memory fact
mj∈ M(b)
tentails the grounded subgoal:m j⊢LLMψg
i.
Running Example 2 (continued).
Given the type-consistent and non-contradictory variable substitution
δi={x /“Kyoto Latte”, y /“Matcha Powder”},
the subgoalψ iis grounded toψg
i=“(Kyoto Latte) contains (Matcha Powder).”Thus,m j⊢LLMψg
i.
If all three conditions hold, the agent accepts the substitution δiinferred from ψiandmj∈ M t, and
updates θ(b)←θ(b)∪δi. In practice, unification should be performed simultaneously across ψ(b)
givenM twithin a single LLM call to prevent bias in iteration order.
6

4.3 Subgoal Refinement via Backward Chaining
If any subgoals in ψ(b)fail to unify with Mt, GOAL-MEMenters the subgoal-refinement (depth)
loop. At each depth stepd∈ {1, . . . , D max}, GOAL-MEMrefines the unresolved subgoals as:
ψ(b,d)={ΨLLM
refine(ψi,M(b,d−1)
t ) :ψ i∈ U(b,d−1)}.(6)
where ψ(b,d−1)denotes the set of subgoals that failed to unify at the previous depth step. The operator
ΨLLM
refine is an LLM-based refinement function that generates a requisite antecedent whose grounding
is required to satisfy the unresolved subgoal. Given the refined subgoals ψ(b,d)
i= ΨLLM
refine(ψ(b,d−1)
i ),
the agent then queries Bto retrieve additional facts as M(b,d)
t=M(b,d−1)
t ∪B retrieve(ψ(b,d),M) . The
agent then re-applies the unification procedure over M(b,d)
t to ground the unresolved subgoals. This
process continues until all original and newly identified subgoals can be grounded by the accumulated
retrieved memoryM(b,d)
t, no new memory can be retrieved, or maximum depth is reached.
Running Example 2 (continued).
ψ(b,d−1)
i =“(x:drink)served in(z:cafe visited last week)”,
mj=“The(Momoco cafe)advertised a(Kyoto Latte)as its seasonal drink.”
Unification fails ( “Momoco cafe”⊬ LLM “cafe visited last week” ) since the cafe visited last week is
unknown.
ΨLLM
refine(ψ(b,d−1)
i ) =ψ(b,d)
i=“(Alice)visited(z:cafe)last week.”
If no new memory can be retrieved or the maximum depth is reached, this decomposition is considered
infeasible for the given MandB.. The system then generates a different decomposition ψ(b+1)of
the original goal. This process continues until the maximum number of breadth steps.
4.4 Answer Generation and Memory Updates
If verifiable unification grounds all active subgoals and binds every required answer variable in
the original goal, yielding a grounded goal ϕg
utsupported by retrieved memory facts, the goal is
considered satisfied. The agent then generates the final response by conditioning an LLM on the
user utterance ut, the grounded goal ϕg
ut, and the supporting facts Fused to ground the subgoals.
The new turn, or information distilled from it, is written to the persistent memory Mdirectly via
B, as defined in Section 3. GOAL-MEMtherefore inherits the storage and update semantics of any
compatible backbone without modification. We provide the complete algorithm in Appendix A.
Running Example 2 (continued).
fLLM(ut, ϕg
ut,F)=“I recommend the Kyoto Latte, a matcha-flavored drink from Cafe Momoco”
4.5 Complexity
GOAL-MEMrequires O(B maxDmax)LLM calls and O(B maxDmaxKmax)retrievals per turn. Kmax
be an upper bound on the number of subgoals at any depth step. In practice, the cost is significantly
lower since execution often terminates early (i.e., within B=1 ,D≤3 as we show in Appendix B), and
batched unification keeps LLM calls per depth step constant. As a result, its token cost is comparable
to forward-reasoning methods. (cf. Appendix B for additional details and token costs analysis).
5 Experiments
To evaluate the effectiveness of the proposed GOAL-MEMbackward chaining framework on real
agentic memory tasks, we consider the following research questions:
RQ1:How does GOAL-MEM, when paired with existing memory backbones, compare with each
memory backbone alone?
RQ2:How does GOAL-MEMcompare with forward-reasoning approaches (e.g., Query Reformula-
tion, Self-Reflection, MemGuide, and ReAct) under matched memory-backbone settings?
RQ3:How do different parameters, such as the maximum breadth and depth. of GOAL-MEM
reasoning, contribute to overall performance?
7

Dataset Category BackboneGemma-4-26B GPT-5.4-mini
Backbone Alone Backbone with GOAL-MEMBackbone Alone Backbone with GOAL-MEM
LLM Acc. (%) F1 LLM Acc. (%) ∆Acc. (%) F1 ∆F1 LLM Acc. (%) F1 LLM Acc. (%) ∆Acc. (%) F1 ∆F1
LoCoMoNo StructureRAG (Dense) 62.39(2.42)0.543(0.018) 79.44(2.02) ↑17.06 0.618(0.018) ↑0.075 63.10(2.41)0.519(0.019) 76.10(2.13) ↑13.00 0.584(0.019) ↑0.065
RAG (BM25) 60.51(2.44)0.530(0.019) 77.69(2.08) ↑17.19 0.610(0.018) ↑0.080 66.86(2.35)0.542(0.019) 74.58(2.17) ↑7.72 0.571(0.019) ↑0.029
Mem0 61.93(2.42)0.484(0.019) 68.22(2.32) ↑6.29 0.508(0.019) ↑0.024 63.42(2.40)0.505(0.019) 65.95(2.37) ↑2.53 0.501(0.019) ↓0.004
Tree-BasedMemTree 42.41(2.47)0.332(0.018) 54.60(2.49) ↑12.19 0.414(0.020) ↑0.082 44.62(2.48)0.388(0.019) 60.51(2.44) ↑15.89 0.468(0.020) ↑0.081
RAPTOR 51.10(2.50)0.441(0.019) 65.30(2.38) ↑14.20 0.519(0.019) ↑0.078 55.90(2.48)0.478(0.019) 63.23(2.41) ↑7.33 0.521(0.019) ↑0.042
Graph-BasedMem0-Graph 59.53(3.23)0.450(0.025) 63.36(3.17) ↑3.83 0.468(0.025) ↑0.018 61.93(2.42)0.469(0.019) 62.91(2.41) ↑0.97 0.463(0.019) ↓0.006
A-MEM 44.42(2.48)0.386(0.019) 59.08(2.45) ↑14.66 0.457(0.020) ↑0.071 41.76(2.46)0.378(0.019) 65.05(2.38) ↑23.28 0.523(0.020) ↑0.145
GraphRAG 55.51(2.48)0.456(0.019) 65.82(2.37) ↑10.31 0.510(0.019) ↑0.054 57.91(2.46)0.476(0.019) 64.50(2.28) ↑6.59 0.509(0.018) ↑0.033
LongMemEvalNo StructureRAG (Dense) 53.33(5.56)0.165(0.019) 63.00(5.46) ↑9.67 0.496(0.049) ↑0.331 58.89(4.19)0.205(0.030) 61.67(5.10) ↑2.78 0.505(0.064) ↑0.299
RAG (BM25) 55.33(5.63)0.160(0.020) 64.00(5.43) ↑8.67 0.503(0.049) ↑0.343 54.44(4.28)0.210(0.031) 57.78(5.22) ↑3.33 0.491(0.064) ↑0.280
Mem0 49.67(5.66)0.200(0.022) 58.67(5.57) ↑9.00 0.457(0.049) ↑0.257 54.17(4.92)0.202(0.032) 59.17(7.79) ↑5.00 0.484(0.079) ↑0.282
Tree-BasedMemTree 56.00(5.62)0.155(0.018) 66.33(5.35) ↑10.33 0.505(0.048) ↑0.350 51.67(5.27)0.208(0.026) 55.83(5.28) ↑4.16 0.436(0.055) ↑0.228
RAPTOR 51.11(5.30)0.204(0.031) 65.67(5.37) ↑14.56 0.512(0.049) ↑0.308 50.00(4.16)0.216(0.031) 56.67(5.24) ↑6.67 0.487(0.064) ↑0.271
Graph-BasedMem0-Graph 44.67(5.63)0.138(0.015) 51.00(5.66) ↑6.33 0.409(0.049) ↑0.271 48.33(4.94)0.194(0.039) 52.16(5.94) ↑3.83 0.410(0.078) ↑0.217
A-MEM 56.33(5.51)0.166(0.019) 64.00(5.43) ↑7.67 0.497(0.049) ↑0.331 52.50(5.84)0.204(0.036) 59.17(5.79) ↑6.67 0.493(0.080) ↑0.290
GraphRAG 53.00(5.65)0.160(0.019) 61.67(5.50) ↑8.67 0.472(0.050) ↑0.312 46.33(4.94)0.198(0.038) 50.83(5.94) ↑4.50 0.431(0.080) ↑0.234
Table 1: RQ1 cross-backbone comparison of standalone memory backbones and the same backbones
augmented with GOAL-MEM. ∆Acc. and ∆F1 report the absolute improvement over the correspond-
ing backbone-alone result. Values in parentheses are 95% confidence-interval half-widths .
Memory Backbones and Baselines.We instantiate GOAL-MEMwith eight retrieval-augmented
memory backbones: BM25-RAG [ 25], Dense-RAG [ 11], Mem0 and Mem0-Graph [ 4], MemTree [ 24],
RAPTOR [ 27], GraphRAG [ 6], and A-MEM [ 33]. These cover unstructured, tree-based, and
graph-based memory designs. Since public implementations differ in availability and interfaces,
we implement all methods in a unified evaluation pipeline following their original algorithmic
descriptions, while controlling for retrieval budgets, prompts, and backbone. We compare against
forward reasoning baselines: query reformulation [ 20], LLM-based self-reflection, MemGuide [ 5],
and ReAct [35] under the same memory backboneB. Please refer to Appendix F for details.
Datasets and Setup.We evaluate GOAL-MEMon two long-term conversational memory bench-
marks: LoCoMo [ 21] and LongMemEval-Small [ 31]. LoCoMo emphasizes long-range, multi-session
memory with many multi-hop questions over dispersed evidence, while LongMemEval evaluates
core assistant memory abilities, including extraction, multi-session and temporal reasoning, knowl-
edge updates, and abstention (cf. Appendix C for details). We use two LLM backbones: GPT-5.4
mini [23] and Gemma-4-26B-A4B [8]. We use identical hyperparameters in the standalone and
GOAL-MEM-augmented settings. The performance delta ( ∆) isolates the contribution of GOAL-
MEMfrom backbone-specific tuning. We maintain a consistent retrieval budget across methods: the
memory backbone alone retrieves up to k= 60 memories in a single call, while reasoning-based
approaches distribute the same retrieval cap across iterations. The maximum breadth is consistently
set to 3, and the maximum depth is set to 5. We evaluate the generated responses with reference
answer via LLM-as-judge using prompts provided by LoCoMo and LongMemEval, and token F1.
5.1 Experimental Results
RQ1.Table 1 reports performance with and without GOAL-MEMacross two datasets and a range of
memory backbones. GOAL-MEMdelivers consistent gains in both LLM-as-judge accuracy and token-
level F1, regardless of how the underlying memory is organized. The largest absolute improvements
appear on the simplest flat backbones, BM25-RAG and Dense-RAG, suggesting that goal-oriented
backward chaining can compensate for limited memory structure. Mem0 and Mem0-Graph show
smaller gains because their abstract memory facts omit fine-grained cues such as temporal details and
raw dialogue context, weakening intermediate subgoal retrieval, which can be mitigated through a
modification in structures, but altering the original backbone designs.
To localize the source of these gains, Figure 3 breaks down accuracy by question type on LoCoMo.
GOAL-MEMshows a substantial lift on multi-hop questions, even for backbones whose overall
improvements are smaller. This pattern supports the claim that semantic similarity alone is insufficient
for multi-hop questions because the intermediate queries needed for chaining are often missing.
GOAL-MEM’s goal-directed backward chaining directly addresses this problem.
RQ2.Table 2 compares GOAL-MEMwith representative forward reasoning approaches under
identical LLM and memory backbone settings. GOAL-MEMeither outperforms or achieves similar
8

Dense BM25 Mem0
MemTreeRAPTOR MAGMA Mem0-GA-MEM
GraphRAG0255075100LLM-as-judge accuracy (%)Multi-hop
mean lift +18.9
↑18.79↑22.70↑17.73
↑10.99↑28.01
↑28.01↑7.56
↑20.21↑15.96
Dense BM25 Mem0
MemTreeRAPTOR MAGMA Mem0-GA-MEM
GraphRAGSingle-hop
mean lift +18.5
↑35.20↑31.78
↑2.80↑32.09
↑9.66↑17.45
↑8.33 ↑22.74
↑6.23
Dense BM25 Mem0
MemTreeRAPTOR MAGMA Mem0-GA-MEM
GraphRAGTemporal
mean lift +22.2
↑26.04 ↑29.17↑8.33
↑21.88↑26.04↑38.54
↑7.55↑20.83↑21.88
Dense BM25 Mem0
MemTreeRAPTOR MAGMA Mem0-GA-MEM
GraphRAGOpen-domain
mean lift +7.9
↑8.56 ↑8.44
↑3.57
↑3.92↑9.99↑18.31
↑0.42
↑9.04↑8.68Gemma 26BAlone Goal-MemFigure 3: LLM accuracy by question type on LoCoMo with Gemma-4-26B . GOAL-MEMconsistently
improves multi-hop performance across all backbones. (cf. Appendix D for GPT-5.4 mini result).
Dataset Category BackboneGemma-4-26B(%)GPT-5.4-mini(%)
Query Reformulation Self-Reflection MemGuide ReAct GOAL-MEMQuery Reformulation Self-Reflection MemGuide ReAct GOAL-MEM
LoCoMoNo StructureRAG (Dense) 56.49(2.47)56.23(2.48)57.72(2.47)44.81(2.48) 79.44(2.02)66.60(2.35)67.32(2.34)67.96(2.33)41.05(2.60) 76.10(2.13)
RAG (BM25) 60.89(2.44)60.51(2.44)60.44(2.44)50.45(2.50) 77.69(2.08)71.85(2.24)70.69(2.27)69.58(2.30)45.46(2.49) 74.58(2.17)
Mem0 57.72(2.47)58.04(2.46)57.72(2.47)64.51(2.91) 68.22(2.32)62.52(2.42)66.02(2.36)63.88(2.40)63.04(2.41) 65.95(2.37)
Tree-BasedMemTree 47.73(2.49)46.43(2.49)44.81(2.48)51.59(3.04) 54.60(2.49)56.94(2.47)56.94(2.47)55.13(1.56)54.33(2.11) 60.51(2.44)
RAPTOR 54.73(2.48)52.33(2.49)52.79(2.49)51.98(3.04) 65.30(2.38)54.35(2.49)58.24(2.46)59.14(2.45)52.97(4.97) 63.23(2.41)
Graph-BasedMem0-Graph 55.36(3.27)55.69(3.27)52.56(3.04)56.80(3.01) 63.36(3.17)58.55(2.97)57.76(3.01)63.68(4.70)52.16(1.12) 62.91(2.41)
A-MEM 38.85(2.43)36.38(2.40)34.85(2.66)35.29(2.91) 59.08(2.45)48.64(2.49)56.42(2.47)52.59(2.49)46.16(2.68) 65.05(2.38)
GraphRAG 55.97(2.48)54.99(2.48)56.03(2.48)37.99(2.95) 65.82(2.37)60.57(2.44)60.05(2.44)53.31(2.49)46.54(2.16) 64.50(2.28)
LongMemEvalNo StructureRAG (Dense) 58.33(5.58)57.67(5.59)59.67(5.55)45.00(4.27) 63.00(5.46)58.33(4.20)58.89(4.19)62.22(7.08)35.56(6.99) 61.67(5.10)
RAG (BM25) 55.00(5.63)56.33(5.61)57.67(5.59)51.00(5.66) 64.00(5.43)52.78(4.29)54.44(4.28)55.00(7.27)40.56(4.17) 57.78(5.22)
Mem0 58.67(5.57)58.33(5.58)60.33(5.54)61.33(5.51) 58.67(5.57)55.83(5.89)51.67(5.94)56.67(5.87)56.67(5.87) 59.17(7.79)
Tree-BasedMemTree 57.33(5.60)57.67(5.59)59.00(5.57)43.33(5.61) 66.33(5.35)56.25(6.28)55.83(6.28)56.67(6.27)35.00(6.03) 55.83(5.28)
RAPTOR 59.33(5.56)57.67(5.59)60.56(4.14)44.67(5.63) 65.67(5.37)50.33(4.20)57.78(5.22)52.22(4.08)37.22(4.06) 56.67(5.24)
Graph-BasedMem0-Graph 43.33(5.61)43.00(5.60)43.67(5.61)42.67(5.65) 51.00(5.66)44.17(5.89)48.33(5.94)49.17(5.94)46.67(5.93) 52.16(5.94)
A-MEM 61.67(5.50)61.67(5.50)51.12(5.77)43.67(5.61) 64.00(5.43)59.17(5.79)63.33(5.62)60.00(5.77)37.50(5.66) 59.17(5.79)
GraphRAG 56.33(5.61)56.67(5.61)57.11(6.14)32.33(5.29) 61.67(5.50)50.83(5.94)45.83(5.92)51.67(5.94)22.50(4.47) 50.83(5.94)
Table 2: RQ2 comparison of LLM accuracy (%) across forward reasoning approaches layered on the
same memory backbone. Values in parentheses are 95% confidence-interval half-widths.
LLM accuracy the forward reasoning baseline across most backbones, including memory-tailored
forward methods such as MemGuide. Our qualitative analysis (cf. Appendix E) further shows that
GOAL-MEMachieves the large gains on multi-hop questions, where forward baselines often retrieve
relevant memory facts but fail to utilize them across hops to produce a valid response. GOAL-MEM
mitigates this by unifying and grounding subgoals with their corresponding memory facts prior to
response generation, providing more guided chaining across hops for correct retrieval utilization.
RQ3.We next ablate the two budgets that govern GOAL-MEM’s search: the maximum depth
(subgoal refinement) and the maximum breadth (alternative goal decompositions). Figure 4 shows
cumulative LLM-as-judge accuracy as each budget is increased, for two backbone LLMs. Increasing
depth yields accuracy improvements until max depth ≥3, and the gains are largest for backbones with
weaker baseline performance. When the first retrieved memory is incomplete or noisy, antecedent-
guided refinement lets agents navigate the memory space more systematically, recovering missing
context that weaker backbones are especially likely to miss. Increasing breadth produces little
additional gain. We attribute this to the strength of SOTA LLMs; at a single goal, decomposition is
typically sufficient, and the depth loop also refines potential failures. Allocating the search budget to
depth rather than breadth is therefore the more practical strategy.
6 Conclusion
We introduced GOAL-MEM, a framework that enhances the reasoning capabilities of memory-
augmented LLM agents through goal-oriented backward chaining. By converting user utterances into
NL-Logic goals and iteratively grounding subgoals via targeted retrieval and verifiable unification,
GOAL-MEMenables more effective use of external memory. In addition, its compatibility with diverse
memory backbones makes it broadly applicable across existing RAG-based systems. Experimental
results show consistent improvements across benchmarks, with particularly strong gains on tasks
requiring multi-step inference. Overall, GOAL-MEMprovides an effective approach for improving
memory-grounded reasoning in agentic LLM systems.
9

0 1 2 3 4 5
Max Depth55%60%65%70%75%LLM as Judge Accuracy
Depth (GPT-5.4-Mini)
0 1 2 3 4 5
Max Depth45%50%55%60%65%70%75%80%
Depth (Gemma-4-26B)
1 2 3
Max Breadth55%60%65%70%75%LLM as Judge Accuracy
Breadth (GPT-5.4-Mini)
1 2 3
Max Breadth45%50%55%60%65%70%75%80%
Breadth (Gemma-4-26B)Depth Ablation Breadth Ablation
RAG (Dense)
RAG (BM25)Mem0
MemTreeRAPTOR
MAGMAMem0-Graph A-MEM GraphRAGFigure 4: Accuracy vs. Dmax(left two) and Bmax(right two). Depth yields steady gains, particularly
on weaker backbones; breadth saturates after a single decomposition.
References
[1]Emre Can Acikgoz, Jeremiah Greer, Akul Datta, Ze Yang, William Zeng, Oussama Elachqar,
Emmanouil Koukoumidis, Dilek Hakkani-Tür, and Gokhan Tur. Can a single model mas-
ter both multi-turn conversations and tool use? CoALM: A unified conversational agen-
tic language model. InProceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages 12370–12390, Vienna, Austria,
2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.acl-long.605. URL
https://aclanthology.org/2025.acl-long.605/.
[2]Ajlan Al-Ajlan. The comparison between forward and backward chaining.International
Journal of Machine Learning and Computing, 5(2):106–113, 2015. doi: 10.7763/IJMLC.
2015.V5.492. URL https://www.ijml.org/index.php?a=show&c=index&catid=56&
id=554&m=content.
[3]Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao
Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. LongBench: A
bilingual, multitask benchmark for long context understanding. InProceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pages 3119–3137, Bangkok, Thailand, 2024. Association for Computational Linguistics. doi: 10.
18653/v1/2024.acl-long.172. URLhttps://aclanthology.org/2024.acl-long.172/.
[4]Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0:
Building production-ready AI agents with scalable long-term memory, 2025. URL https:
//arxiv.org/abs/2504.19413.
[5]Yiming Du, Bingbing Wang, Yang He, Bin Liang, Baojun Wang, Zhongyang Li, Lin Gui,
Jeff Z. Pan, Ruifeng Xu, and Kam-Fai Wong. MemGuide: Intent-driven memory selection for
goal-oriented multi-session LLM agents.Proceedings of the AAAI Conference on Artificial
Intelligence, 40(36):30584–30592, 2026. doi: 10.1609/aaai.v40i36.40313. URL https:
//ojs.aaai.org/index.php/AAAI/article/view/40313.
[6]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven
Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global:
A graph rag approach to query-focused summarization.arXiv preprint arXiv:2404.16130, 2024.
[7]Richard E. Fikes and Nils J. Nilsson. STRIPS: A new approach to the application of theorem
proving to problem solving.Artificial Intelligence, 2(3–4):189–208, 1971. doi: 10.1016/
0004-3702(71)90010-5.
[8]Google AI. Gemma 4 model overview, 2026. URL https://ai.google.dev/gemma/docs/
core. Accessed: 2026-05-07.
[9]David Guo, Minqi Sun, Yilun Jiang, Jiazhou Liang, and Scott Sanner. VOGUE: A multimodal
dataset for conversational recommendation in fashion, 2025. URL https://arxiv.org/abs/
2510.21151. Accepted as a full paper at ACM UMAP 2026.
[10] Dongming Jiang, Yi Li, Guanpeng Li, and Bingzhe Li. MAGMA: A multi-graph based agentic
memory architecture for AI agents, 2026. URL https://arxiv.org/abs/2601.03236 .
ACL 2026 Main.
10

[11] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov,
Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering.
InProceedings of the 2020 Conference on Empirical Methods in Natural Language Processing,
pages 6769–6781, Online, 2020. Association for Computational Linguistics. doi: 10.18653/v1/
2020.emnlp-main.550. URLhttps://aclanthology.org/2020.emnlp-main.550/.
[12] Mehran Kazemi, Najoung Kim, Deepti Bhatia, Xin Xu, and Deepak Ramachandran. LAM-
BADA: Backward chaining for automated reasoning in natural language. InProceedings of the
61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pages 6547–6568, Toronto, Canada, 2023. Association for Computational Linguistics. doi: 10.
18653/v1/2023.acl-long.361. URLhttps://aclanthology.org/2023.acl-long.361/.
[13] Junyoung Kim, Anton Korikov, Jiazhou Liang, Justin Cui, Yifan Simon Liu, Qianfeng Wen,
Mark Zhao, and Scott Sanner. Bayesian active learning with gaussian processes guided by llm
relevance scoring for dense passage retrieval.arXiv preprint arXiv:2604.17906, 2026.
[14] Jinu Lee and Wonseok Hwang. SymBa: Symbolic backward chaining for structured natural
language reasoning. InProceedings of the 2025 Conference of the Nations of the Amer-
icas Chapter of the Association for Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), pages 2468–2484, Albuquerque, New Mexico, 2025.
Association for Computational Linguistics. doi: 10.18653/v1/2025.naacl-long.124. URL
https://aclanthology.org/2025.naacl-long.124/.
[15] Jiazhou Liang, Yifan Simon Liu, David Guo, Minqi Sun, Yilun Jiang, and Scott Sanner.
Evaluating scene-based in-situ item labeling for immersive conversational recommendation.
arXiv preprint arXiv:2604.09698, 2026.
[16] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni,
and Percy Liang. Lost in the middle: How language models use long contexts.Transactions of
the Association for Computational Linguistics, 12:157–173, 2024. doi: 10.1162/tacl_a_00638.
URLhttps://aclanthology.org/2024.tacl-1.9/.
[17] Yifan Liu, Qianfeng Wen, Jiazhou Liang, Mark Zhao, Justin Cui, Anton Korikov, Armin
Toroghi, Junyoung Kim, and Scott Sanner. Multimodal item scoring for natural language
recommendation via gaussian process regression with llm relevance judgments.arXiv preprint
arXiv:2510.22023, 2025.
[18] Yifan Liu, Qianfeng Wen, Mark Zhao, Jiazhou Liang, and Scott Sanner. MA-DPR: Manifold-
aware distance metrics for dense passage retrieval. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing, pages 31085–31103, Suzhou, China, 2025.
Association for Computational Linguistics. doi: 10.18653/v1/2025.emnlp-main.1582. URL
https://aclanthology.org/2025.emnlp-main.1582/.
[19] Yifan Simon Liu, Ruifan Wu, Liam Gallagher, Jiazhou Liang, Armin Toroghi, and Scott
Sanner. Semantic XPath: Structured agentic memory access for conversational AI, 2026. URL
https://arxiv.org/abs/2603.01160.
[20] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. Query rewriting in retrieval-
augmented large language models. InProceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, pages 5303–5315, Singapore, 2023. Association
for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.322. URL https://
aclanthology.org/2023.emnlp-main.322/.
[21] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and
Yuwei Fang. Evaluating very long-term conversational memory of LLM agents. InProceedings
of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers), pages 13851–13870, Bangkok, Thailand, 2024. Association for Computational
Linguistics. doi: 10.18653/v1/2024.acl-long.747. URL https://aclanthology.org/2024.
acl-long.747/.
[22] Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, KaShun Shum, Randy Zhong, Juntong Song,
and Tong Zhang. RAGTruth: A hallucination corpus for developing trustworthy retrieval-
augmented language models. InProceedings of the 62nd Annual Meeting of the Association for
11

Computational Linguistics (Volume 1: Long Papers), pages 10862–10878, Bangkok, Thailand,
2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.585. URL
https://aclanthology.org/2024.acl-long.585/.
[23] OpenAI. Introducing GPT-5.4 mini and nano, mar 2026. URL https://openai.com/index/
introducing-gpt-5-4-mini-and-nano/. Accessed: 2026-05-07.
[24] Alireza Rezazadeh, Zichao Li, Wei Wei, and Yujia Bao. From isolated conversations to
hierarchical schemas: Dynamic tree memory representation for LLMs, 2024. URL https:
//arxiv.org/abs/2410.14052.
[25] Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and
beyond.Foundations and Trends in Information Retrieval, 3(4):333–389, 2009. doi: 10.1561/
1500000019. URLhttps://doi.org/10.1561/1500000019.
[26] Stuart J. Russell and Peter Norvig.Artificial Intelligence: A Modern Approach. Prentice Hall,
Englewood Cliffs, NJ, 1995. ISBN 0131038052.
[27] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D.
Manning. RAPTOR: Recursive abstractive processing for tree-organized retrieval, 2024. URL
https://arxiv.org/abs/2401.18059.
[28] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R. Narasimhan, and Shunyu Yao. Re-
flexion: Language agents with verbal reinforcement learning. InAdvances in Neural Information
Processing Systems, 2023. URLhttps://openreview.net/forum?id=vAElhFcKW6.
[29] Armin Toroghi, Willis Guo, Ali Pesaranghader, and Scott Sanner. Verifiable, debuggable, and
repairable commonsense logical reasoning via LLM-based theory resolution. InProceedings of
the 2024 Conference on Empirical Methods in Natural Language Processing, pages 6634–6652,
Miami, Florida, USA, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.
emnlp-main.379. URLhttps://aclanthology.org/2024.emnlp-main.379/.
[30] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Inter-
leaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step ques-
tions. InProceedings of the 61st Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 10014–10037, Toronto, Canada, 2023. Asso-
ciation for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.557. URL https:
//aclanthology.org/2023.acl-long.557/.
[31] Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. Long-
MemEval: Benchmarking chat assistants on long-term interactive memory. InThe Thirteenth
International Conference on Learning Representations, 2025. doi: 10.48550/arXiv.2410.10813.
URLhttps://openreview.net/forum?id=pZiyCaVuti.
[32] Yaxiong Wu, Sheng Liang, Chen Zhang, Yichao Wang, Yongyue Zhang, Huifeng Guo, Ruiming
Tang, and Yong Liu. From human memory to AI memory: A survey on memory mechanisms in
the era of LLMs, 2025. URLhttps://arxiv.org/abs/2504.15965.
[33] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-MEM: Agen-
tic memory for LLM agents, 2025. URL https://arxiv.org/abs/2502.12110 . NeurIPS
2025.
[34] Shannan Yan, Jingchen Ni, Leqi Zheng, Jiajun Zhang, Peixi Wu, Dacheng Yin, Jing Lyu, Chun
Yuan, and Fengyun Rao. AdaMem: Adaptive user-centric memory for long-horizon dialogue
agents, 2026. URLhttps://arxiv.org/abs/2603.16496.
[35] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and
Yuan Cao. ReAct: Synergizing reasoning and acting in language models. InThe Eleventh
International Conference on Learning Representations, 2023. doi: 10.48550/arXiv.2210.03629.
URLhttps://openreview.net/forum?id=WE_vluYUL-X.
12

[36] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik R. Narasimhan. τ-bench: A bench-
mark for tool-agent-user interaction in real-world domains. InThe Thirteenth International
Conference on Learning Representations, 2025. doi: 10.48550/arXiv.2406.12045. URL
https://openreview.net/forum?id=roNSXZpUDN.
[37] Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Quanyu Dai, Jieming Zhu, Zhenhua Dong,
and Ji-Rong Wen. A survey on the memory mechanism of large language model based agents,
2024. URLhttps://arxiv.org/abs/2404.13501.
[38] Yujie Zhao, Boqin Yuan, Junbo Huang, Haocheng Yuan, Zhongming Yu, Haozhou Xu, Lanxiang
Hu, Abhilash Shankarampeta, Zimeng Huang, Wentao Ni, Yuandong Tian, and Jishen Zhao.
AMA-Bench: Evaluating long-horizon memory for agentic applications, 2026. URL https:
//arxiv.org/abs/2602.22769.
[39] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. MemoryBank: Enhancing
large language models with long-term memory.Proceedings of the AAAI Conference on
Artificial Intelligence, 38(17):19724–19731, 2024. doi: 10.1609/aaai.v38i17.29946. URL
https://ojs.aaai.org/index.php/AAAI/article/view/29946.
13

A GOAL-MEMAlgorithm
The workflow of the GOAL-MEMreasoning framework is summarized in Algorithm 1. The reasoning
process begins by mapping the utterance utinto a formal goal ϕut, which specifies the variables that
must be instantiated to answer the query. An initially empty substitution θis maintained to store
variable bindings as they are discovered.
The algorithm proceeds through thebreadth loop. At each breadth step b, the goal ϕutis decomposed
into a set of subgoals ψ(b)using an LLM-based decomposition function, conditioned on previously
explored decompositions {ψ(0), . . . , ψ(b−1)}(with ψ(0)=∅). Each subgoal represents a necessary
condition for satisfying the original goal.
For each subgoal ψi∈ψ(b), the algorithm performstargeted retrievalby querying the memory
backbone. The retrieved facts are aggregated into a memory set M(b,0)
t. The algorithm then attempts
to ground the subgoals via verifiableunification, which matches subgoals against retrieved memory
while enforcing three constraints: (i) type consistency between variables and candidate values, (ii)
consistency with previously established substitutions, and (iii) logical entailment of the grounded
subgoal by the retrieved memory (as judged by the LLM).
This step returns an updated substitution θ(b,0), a set of unresolved subgoals U(b,0), and a set of
supporting memory facts Fthat successfully grounded subgoals. We also maintain the accumulated
subgoal setΨ(b,≤0)=ψ(b), and define the set of resolved subgoalsS(b,0)= Ψ(b,≤0)\ U(b,0).
If all subgoals are successfully grounded (i.e., U(b,0)=∅), the algorithm terminates and generates
the final response conditioned on the utterance, the substitution, and only the supporting memory F.
Otherwise, the algorithm enters adepth loop, which performs recursive refinement of unresolved
subgoals. At each depth step d, only the subgoals from the previous refinement frontier ψ(b,d−1)
(initialized as U(b,0)) are refined using an LLM-based refinement function, producing a new set ψ(b,d)
of intermediate subgoals. These refined subgoals are used for additional targeted retrieval, and the
newly retrieved facts are accumulated into the memoryM(b,d)
t.
The accumulated subgoal set is updated as Ψ(b,≤d)= Ψ(b,≤d−1)∪ψ(b,d). The unification procedure
is then re-applied over Ψ(b,≤d)andM(b,d)
t to attempt further grounding. To avoid redundant
computation while preserving correctness, subgoals that have already been successfully grounded
(tracked in S(b,d−1)) are skipped during unification, although they remain part of the accumulated
subgoal set for consistency. This step produces updated substitution θ(b,d), unresolved subgoals
U(b,d), and supporting factsF, and updates the resolved set asS(b,d)= Ψ(b,≤d)\ U(b,d).
This refinement process continues until either (i) all subgoals are grounded, (ii) no new memory can
be retrieved, or (iii) the maximum depth is reached. If grounding succeeds at any depth, the algorithm
generates the final response using the grounded substitution and the supporting memoryF.
If the current decomposition fails to yield a complete grounding after exhausting the depth budget,
the algorithm returns to the breadth loop and considers an alternative decomposition of the original
goal, enabling exploration of multiple reasoning strategies.
B Complexity
LetKmaxbe an upper bound on the number of subgoals at any depth step. We exclude memory
backbone LLM calls, since the latter depend on the implementation of Bretrieve . For each breadth
attempt, GOAL-MEMuses one LLM call for goal decomposition and one batched LLM call for
verifiable unification over all retrieved facts and active subgoals. At each refinement depth, it then uses
one LLM call to refine the unresolved subgoals and one batched LLM call to re-run unification after
the additional retrieval. Thus, the LLM calls is bounded by 1+B max 
2+2D max
orO(B maxDmax)
LLM calls. The memory retrieval is bounded by:C ret≤B max(Dmax+ 1)K max.
We conduct various empirical analysis in Figure 4 and next paragraph. In practice, the realized
cost is often lower because it frequently terminates before the algorithm reaches BmaxandDmax
(i.e., 95 percentiles of samples have finished within B= 1 andD= 3 across datasets), as soon as
all variables in ϕutare grounded. In addition, under batched unification, GOAL-MEMuses only a
14

Algorithm 1GOAL-MEM: Goal-Oriented Backward Reasoning
Require:Utteranceu t, memoryM, retrievalB retrieve , max breadthB max, max depthD max
Ensure:Responsea t
1:ϕut←PARSETONLLOGIC(u t)
2:θ(0,0)← ∅
3:ψ(0)← ∅
4:forb= 1toB maxdo
5:ψ(b)←Ψdecomp
LLM (ϕut,{ψ(0), . . . , ψ(b−1)})
6:M(b,0)
t← ∅
7:for allψ i∈ψ(b)do
8:M(b,0)
t← M(b,0)
t∪B retrieve (ψi, M)
9:end for
10:(θ(b,0),U(b,0),F)←UNIFY(ψ(b),M(b,0)
t, θ(0,0),∅)
11:Ψ(b,≤0)←ψ(b)
12:S(b,0)←Ψ(b,≤0)\ U(b,0)
13:ifU(b,0)=∅then
14:returnGENERATE(u t, θ(b,0),F)
15:end if
16:ψ(b,0)← U(b,0)
17:ford= 1toD maxdo
18:ψ(b,d)← ∅
19:for allψ i∈ψ(b,d−1)do
20:ψ′
i←Ψrefine
LLM(ψi,M(b,d−1)
t )
21:ψ(b,d)←ψ(b,d)∪ {ψ′
i}
22:end for
23:M(b,d)
t← M(b,d−1)
t
24:for allψ′
i∈ψ(b,d)do
25:M(b,d)
t← M(b,d)
t∪B retrieve (ψ′
i,M)
26:end for
27:Ψ(b,≤d)←Ψ(b,≤d−1)∪ψ(b,d)
28:(θ(b,d),U(b,d),F)←UNIFY(Ψ(b,≤d),M(b,d)
t, θ(b,d−1),S(b,d−1))
29:S(b,d)←Ψ(b,≤d)\ U(b,d)
30:ifU(b,d)=∅then
31:returnGENERATE(u t, θ(b,d),F)
32:end if
33:ifno new memory retrievedthen
34:break
35:end if
36:end for
37:end for
38:returnFAILUREFLAG
constant number of LLM calls per depth step. Thus, since BmaxandDmaxare small, GOAL-MEM
has a similar asymptotic per-turn LLM-call complexity as existing forward-reasoning approaches.
B.1 Distribution of Breadth, Depth, and Subgoals in Real Datasets
Figure 5 shows that the empirical search cost is concentrated near the smallest possible values.
The realized breadth distribution is dominated by breadth 1, indicating that the first generated goal
decomposition is sufficient for most examples. Only a relatively small fraction of cases require
additional breadth attempts, and the configured cap Bmax= 3is reached mainly by harder examples
for which the first decomposition does not unify with the retrieved memories.
The realized depth distribution is even more concentrated. Most examples terminate at depth 0,
meaning that GOAL-MEManswers after the initial decomposition, retrieval, and verifiable unification
step, without invoking the missing-information refinement loop. A smaller tail reaches depths 1
15

0 1 2 3 4 5
Maximum Breadth05000100001500020000FrequencyDistribution of Breadth Usage
0 1 2 3 4 5
Depth05000100001500020000FrequencyDistribution of Depth Usage
0 1 2 3 4 5 6 7
Number of Sugboals0500010000150002000025000FrequencyDistribution of Sugboal Count Per DecompositionFigure 5: Empirical distributions of realized search statistics in GOAL-MEM. The left panel shows
the realized breadth usage, the middle panel shows the realized refinement depth, and the right panel
shows the number of subgoals generated per decomposition. Although the search caps are Bmax= 3
andD max= 5, most examples terminate using substantially smaller realized breadth and depth.
through 5, showing that depth search is used selectively for under-specified or partially grounded
questions. Thus, while Dmax= 5provides a safety cap for difficult cases, the typical realized depth
is far below this maximum.
The subgoal-count distribution further supports the efficiency of the method. Most decompositions
contain only a small number of subgoals, with the mass concentrated around two subgoals and only a
small tail with more complex decompositions. Since memory retrieval scales with the number of
queried subgoals, this concentration keeps the realized retrieval fan-out small. Moreover, because
unification is batched over the active subgoals and retrieved facts, additional subgoals do not require
separate unification LLM calls within the same depth step.
Overall, these distributions show that the theoretical O(B maxDmax)LLM-call bound is a worst-case
guarantee rather than the typical operating regime. With Bmax= 3andDmax= 5, the search space
is tightly bounded, but the realized breadth and depth are usually much smaller. This supports the
claim that GOAL-MEMpreserves practical per-turn efficiency while still allowing additional breadth
and depth when the initial decomposition is insufficient.
B.2 Token Cost and Wall-Clock Time Across Reasoning Baselines
All experiments were run using the OpenAI API (GPT-5.4-mini) and DeepInfra API (Gemma-4-
26B-A4B). No local GPU resources were required for backbone LLM inference. Retrieval-based
backbones (BM25, Dense) were run on a single CPU machine; graph- and tree-based backbones
(Mem0, MAGMA, MemTree, RAPTOR, A-MEM, GraphRAG) were run on a single machine with
16 GB RAM.
Table 3 report the average token cost and wall-clock time across the evaluated forward-reasoning
methods using the RAG-BM25 backend. We use RAG-BM25 because it retrieves directly from raw
utterances and therefore provides the most consistent token accounting, without additional token
overhead introduced by different memory structures. We exclude fixed prompt tokens, since they
vary across datasets and use cases and are negligible relative to the retrieved memory facts. We also
exclude token usage from memory updating and encoding stages, as these costs depend on the chosen
memory backbone and are outside the scope of this work.
Backbone Alone and Query Reformulation incur minimal token cost because they do not require an
LLM agent to consume all retrieved memory facts during reasoning. In Query Reformulation, the
query is rewritten before the retrieval stage using only the raw user utterance, while Backbone Alone
directly retrieves memory facts without an iterative agent loop. In contrast, Self-Reflection, ReAct,
MemGuide, and GOAL-MEMare iterative memory-conditioned approaches: each step must consume
retrieved memory facts to perform reflection, action selection, reranking, or unification for subsequent
retrieval. Their token usage is therefore naturally higher than Backbone Alone. However, due to
batched unification, GOAL-MEMdoes not repeatedly call the LLM for every subgoal-condition check,
allowing it to maintain a competitive token cost compared with other iterative reasoning methods.
16

Table 3: Comparison of runtime and estimated token cost across different reasoning methods with
the bm25 backend.
Method Avg. End to End Runtime (s) Avg. Tokens
Backbone Alone 0.68 0
Query Reformulation 7.52 1827
Self-Reflection 15.18 12269
ReAct 16.01 22802
GOAL-MEM15.10 16977
MemGuide 14.90 14225
C Additional Information on the Evaluation Datasets
We evaluate GOAL-MEMon two long-term conversational memory benchmarks: LoCoMo [ 21] and
LongMemEval-Small [ 31]. These datasets are designed to assess the ability of LLM-based agents to
store, retrieve, and reason over information accumulated across extended interactions.
LoCoMo.LoCoMo is a benchmark for evaluating long-range conversational memory in multi-
session settings. It consists of dialogues that span multiple sessions, where relevant information
required to answer a query may be distributed across distant parts of the interaction history. The
dataset emphasizes questions that requiremulti-hop reasoning, where the model must retrieve and
combine multiple pieces of evidence to arrive at the correct answer. In addition to multi-hop questions,
LOCOMOincludes single-hop, temporal, and open-domain queries, enabling a fine-grained analysis
of model performance across different reasoning types. A key challenge in LOCOMOis that the
information necessary to answer a question is often not semantically similar to the query itself,
requiring the model to identify and retrieve intermediate facts. This makes it particularly suitable for
evaluating the effectiveness of goal-directed retrieval and backward chaining.
LONGMEMEVAL-SMALL.LONGMEMEVAL-SMALLis a benchmark focused on core memory
capabilities of conversational agents. It evaluates a range of skills, including information extraction
from dialogue, reasoning across multiple sessions, handling temporal dependencies, updating knowl-
edge over time, and abstaining when sufficient evidence is not available. Compared to LOCOMO,
this dataset places greater emphasis on the correctness and consistency of memory usage, as well
as the ability to track evolving user information. The dataset includes diverse query types that test
whether the model can accurately retrieve and ground responses in previously stored memory, rather
than relying on parametric knowledge.
Evaluation Protocol.For both datasets, we evaluate model outputs using an LLM-as-a-judge
framework, which assesses the correctness and grounding of generated responses with respect to
the reference answers. In addition, we report token-level F1 scores to provide a complementary
quantitative measure of answer overlap. These evaluation metrics allow us to capture both semantic
correctness and lexical fidelity, offering a comprehensive view of model performance on long-term
memory reasoning tasks.
D LoCoMo Performance by Question Type
Table 4 and Table 5 present detailed performance breakdowns on the LoCoMo dataset by question
type. Notably, we observe a consistent improvement on multi-hop questions across both LLM
backbones.
E Qualitative Analysis of Multi-hop Forward-Reasoning Failures
We select the failure examples in Table 6 from LoCoMo via RAG-BM25 memory backbone since it
has the the most comprehensive memory traces.
The forward baselines are strong retrieval-augmented methods, but they primarily decide what to
retrieve from the surface form of the question and the currently accumulated evidence. As a result,
17

LLM Category BackboneMulti-hop(%)Single-hop(%)Temporal(%)Open-domain(%)
Alone GOAL-MEM ∆Alone GOAL-MEM ∆Alone GOAL-MEM ∆Alone GOAL-MEM ∆
Gemma
26BNo StructureRAG (Dense) 44.33 (5.80) 63.12 (5.63) ↑18.7948.60 (5.47) 83.80 (4.03) ↑35.2022.92 (8.41) 48.96 (10.00) ↑26.0478.12 (2.79) 86.68 (2.30) ↑8.56
RAG (BM25) 37.59 (5.65) 60.28 (5.71) ↑22.7049.84 (5.47) 81.62 (4.24) ↑31.7819.79 (7.97) 48.96 (10.00) ↑29.1776.81 (2.85) 85.26 (2.40) ↑8.44
Mem0 49.65 (5.84) 67.38 (5.47) ↑17.7353.27 (5.46) 56.07 (5.43) ↑2.8042.71 (9.90) 51.04 (10.00) ↑8.3371.46 (3.05) 75.03 (2.93) ↑3.57
Tree-BasedMemTree 26.24 (5.13) 37.23 (5.64) ↑10.9932.40 (5.12) 64.49 (5.24) ↑32.0919.79 (7.97) 41.67 (9.86) ↑21.8854.10 (3.37) 58.03 (3.34) ↑3.92
RAPTOR 33.69 (5.52) 61.70 (5.67) ↑28.0132.40 (5.12) 42.06 (5.40) ↑9.6620.83 (8.12) 46.88 (9.98) ↑26.0467.42 (3.17) 77.41 (2.83) ↑9.99
Graph-BasedMAGMA 23.40 (4.94) 51.42 (5.83) ↑28.0158.88 (5.38) 76.32 (4.65) ↑17.4513.54 (6.84) 52.08 (9.99) ↑38.5463.73 (3.25) 82.05 (2.59) ↑18.31
Mem0-Graph 51.74 (7.47) 59.30 (7.34) ↑7.5653.33 (7.29) 61.67 (7.10) ↑8.3326.42 (11.87) 33.96 (12.75) ↑7.5568.12 (4.17) 68.54 (4.15) ↑0.42
A-MEM 27.66 (5.22) 47.87 (5.83) ↑20.2140.50 (5.37) 63.24 (5.27) ↑22.7418.75 (7.81) 39.58 (9.78) ↑20.8354.46 (3.37) 63.50 (3.25) ↑9.04
GraphRAG 40.43 (5.73) 56.38 (5.79) ↑15.9638.63 (5.33) 44.86 (5.44) ↑6.2321.88 (8.27) 43.75 (9.92) ↑21.8870.75 (3.07) 79.43 (2.73) ↑8.68
GPT-5.4
miniNo StructureRAG (Dense) 38.30 (5.67) 58.51 (5.75) ↑20.2164.49 (5.24) 81.31 (4.26) ↑16.8244.79 (9.95) 48.96 (10.00) ↑4.1772.89 (3.00) 83.08 (2.54) ↑10.19
RAG (BM25) 35.11 (5.57) 53.90 (5.82) ↑18.7973.83 (4.81) 78.19 (4.52) ↑4.3647.92 (9.99) 47.92 (9.99) ↑0.0076.93 (2.85) 83.12 (2.53) ↑6.18
Mem0 48.23 (5.83) 57.80 (5.76) ↑9.5756.70 (5.42) 55.14 (5.44) ↓1.5652.08 (9.99) 50.00 (10.00) ↓2.0872.29 (3.02) 74.55 (2.94) ↑2.26
Tree-BasedMemTree 24.47 (5.02) 37.59 (5.65) ↑13.1233.96 (5.18) 65.73 (5.19) ↑31.7832.29 (9.35) 40.62 (9.82) ↑8.3356.72 (3.35) 68.37 (3.14) ↑11.65
RAPTOR 38.30 (5.67) 53.90 (5.82) ↑15.6036.76 (5.27) 37.07 (5.28) ↑0.3142.71 (9.90) 36.46 (9.63) ↓6.2570.51 (3.08) 79.31 (2.74) ↑8.80
Graph-BasedMAGMA 26.60 (5.16) 48.94 (5.83) ↑22.3461.06 (5.33) 74.45 (4.77) ↑13.4041.67 (9.86) 46.88 (9.98) ↑5.2165.87 (3.20) 78.24 (2.79) ↑12.37
Mem0-Graph 49.65 (5.84) 57.09 (5.78) ↑7.4552.65 (5.46) 51.40 (5.47) ↓1.2553.12 (9.98) 51.04 (10.00) ↓2.0870.51 (3.08) 70.51 (3.08) ↑0.00
A-MEM 24.47 (5.02) 47.87 (5.83) ↑23.4046.73 (5.46) 75.08 (4.73) ↑28.3533.33 (9.43) 45.83 (9.97) ↑12.5046.61 (3.37) 69.08 (3.12) ↑22.47
GraphRAG 37.59 (5.65) 50.00 (5.53) ↑12.4141.12 (5.38) 44.69 (5.15) ↑3.5739.58 (9.78) 47.71 (9.38) ↑8.1273.25 (2.99) 79.14 (2.64) ↑5.90
Table 4: RQ1 breakdown of LLM-as-judge accuracy (%) by question type (LoCoMo). For each
type, we report the backbone-alone score, the score after augmenting the same backbone with GOAL-
MEM, and the absolute lift ( ∆). Adversarial questions are omitted. Values in parentheses are 95%
confidence-interval half-widths .
LLM Category BackboneMulti-hop Single-hop Temporal Open-domain
Alone GOAL-MEM ∆Alone GOAL-MEM ∆Alone GOAL-MEM ∆Alone GOAL-MEM ∆
Gemma
26BNo StructureRAG (Dense) 0.390 (0.038) 0.501 (0.039) ↑0.111 0.458 (0.030) 0.644 (0.037) ↑0.187 0.145 (0.059) 0.339 (0.084) ↑0.194 0.674 (0.024) 0.680 (0.024) ↑0.007
RAG (BM25) 0.341 (0.037) 0.472 (0.039) ↑0.131 0.451 (0.032) 0.637 (0.037) ↑0.186 0.120 (0.055) 0.323 (0.084) ↑0.203 0.671 (0.024) 0.681 (0.024) ↑0.010
Mem0 0.398 (0.038) 0.479 (0.039) ↑0.081 0.495 (0.037) 0.543 (0.038) ↑0.048 0.223 (0.065) 0.343 (0.085) ↑0.120 0.540 (0.027) 0.524 (0.026) ↓0.015
Tree-BasedMemTree 0.207 (0.032) 0.314 (0.039) ↑0.107 0.286 (0.034) 0.514 (0.043) ↑0.228 0.064 (0.029) 0.285 (0.082) ↑0.221 0.423 (0.027) 0.425 (0.028) ↑0.002
RAPTOR 0.329 (0.037) 0.482 (0.040) ↑0.153 0.287 (0.029) 0.410 (0.040) ↑0.123 0.122 (0.054) 0.284 (0.082) ↑0.162 0.576 (0.027) 0.601 (0.026) ↑0.026
Graph-BasedMAGMA 0.237 (0.034) 0.412 (0.037) ↑0.175 0.403 (0.037) 0.590 (0.037) ↑0.187 0.087 (0.036) 0.331 (0.083) ↑0.243 0.547 (0.027) 0.643 (0.025) ↑0.096
Mem0-Graph 0.385 (0.050) 0.450 (0.053) ↑0.065 0.454 (0.048) 0.546 (0.050) ↑0.092 0.160 (0.075) 0.239 (0.102) ↑0.079 0.506 (0.036) 0.473 (0.036) ↓0.033
A-MEM 0.265 (0.034) 0.355 (0.037) ↑0.091 0.363 (0.034) 0.553 (0.041) ↑0.190 0.080 (0.043) 0.269 (0.079) ↑0.189 0.471 (0.028) 0.476 (0.028) ↑0.005
GraphRAG 0.342 (0.038) 0.451 (0.038) ↑0.109 0.282 (0.032) 0.409 (0.040) ↑0.127 0.120 (0.049) 0.290 (0.082) ↑0.171 0.600 (0.026) 0.596 (0.026) ↓0.005
GPT-5.4
miniNo StructureRAG (Dense) 0.344 (0.038) 0.441 (0.039) ↑0.097 0.492 (0.035) 0.583 (0.037) ↑0.091 0.283 (0.077) 0.353 (0.085) ↑0.070 0.616 (0.026) 0.660 (0.025) ↑0.044
RAG (BM25) 0.304 (0.035) 0.423 (0.039) ↑0.120 0.520 (0.035) 0.573 (0.038) ↑0.053 0.273 (0.074) 0.345 (0.085) ↑0.072 0.663 (0.024) 0.648 (0.025) ↓0.015
Mem0 0.395 (0.038) 0.414 (0.039) ↑0.019 0.497 (0.034) 0.524 (0.036) ↑0.027 0.379 (0.087) 0.330 (0.082) ↓0.049 0.561 (0.026) 0.542 (0.026) ↓0.019
Tree-BasedMemTree 0.232 (0.034) 0.304 (0.039) ↑0.073 0.378 (0.033) 0.522 (0.039) ↑0.144 0.177 (0.065) 0.275 (0.079) ↑0.098 0.469 (0.028) 0.526 (0.028) ↑0.057
RAPTOR 0.346 (0.037) 0.409 (0.039) ↑0.063 0.358 (0.030) 0.371 (0.034) ↑0.013 0.284 (0.075) 0.232 (0.077) ↓0.052 0.591 (0.026) 0.649 (0.025) ↑0.058
Graph-BasedMAGMA 0.241 (0.034) 0.390 (0.037) ↑0.149 0.465 (0.035) 0.550 (0.035) ↑0.085 0.290 (0.078) 0.310 (0.082) ↑0.019 0.563 (0.026) 0.631 (0.026) ↑0.069
Mem0-Graph 0.356 (0.038) 0.392 (0.040) ↑0.036 0.465 (0.033) 0.459 (0.036) ↓0.005 0.383 (0.087) 0.335 (0.085) ↓0.048 0.520 (0.027) 0.504 (0.027) ↓0.016
A-MEM 0.247 (0.033) 0.384 (0.038) ↑0.137 0.472 (0.036) 0.580 (0.038) ↑0.108 0.222 (0.071) 0.341 (0.086) ↑0.119 0.404 (0.027) 0.570 (0.027) ↑0.166
GraphRAG 0.298 (0.036) 0.365 (0.036) ↑0.067 0.356 (0.031) 0.414 (0.032) ↑0.059 0.276 (0.079) 0.326 (0.078) ↑0.050 0.604 (0.026) 0.619 (0.025) ↑0.015
Table 5: RQ1 breakdown of token-level F1 by question type (LoCoMo). For each type, we report
the backbone-alone score, the score after augmenting the same backbone with GOAL-MEM, and
the absolute lift ( ∆). Adversarial questions are omitted. Values in parentheses are 95% confidence-
interval half-widths .
they often retrieve a salient memory and then answer with the most lexical or locally plausible span.
For example , the forward methods retrieve the shelter memory but return the city Stamford, even
though the question asks for the state. Or several baselines return the unresolved phrase “my old area,”
which is evidence that the flood memory was found but the required referring-expression grounding
was not completed.
GOAL-MEMinstead of treating retrieved memories as a flat context for generation, it decomposes the
question into checkable subgoals and then applies verifiable unification. This allows the model to keep
track of answer variables, requested answer types, and unresolved referring expressions. For example,
the goal is not merely to find any game-related memory, but to bind the variable corresponding to
the game played in James’s April online tournament. The first pass does not directly ground this
variable, so the missing-information refinement step asks for the specific game associated with that
tournament. The later evidence grounds the same event to Apex Legends, while unrelated games
such as Civilization VI are rejected because they do not satisfy the tournament-and-time constraint.
These examples also show depth is useful when the current decomposition is correct but one variable
remains ungrounded, as in the April tournament example. Breadth is useful when the first decomposi-
tion is too literal or retrieves only weak evidence. The system initially fails to find a direct phrasing
of “suggested outdoor activity.” A later breadth attempt reformulates the goal as evidence that Jolene
is interested in an activity and wants to do it with Deborah.
18

Question Goal-
MemForward baseline outputs Goal-Mem reasoning pattern
In which state is
the shelter from
which James
adopted the
puppy?Connecticut All forward baselines return
Stamford.Grounds the adoption memory to the shelter
location Stamford, keeps the requested an-
swer type as a state, and applies the location-
granularity bridge Stamford →Connecticut.
What game was
James playing in
the online gam-
ing tournament
in April 2022?Apex Leg-
endsQuery Reformulation, Self-
Reflection, and MemGuide
return “I don’t know”; Re-
Act returns Civilization VI.Grounds the April tournament event, refines
the missing game slot, and links the tempo-
rally aligned game mention to Apex Legends
rather than a different gaming memory.
What is the fa-
vorite game Jo-
lene plays with
her partner?It Takes
TwoQuery Reformulation, Self-
Reflection, and ReAct re-
turn Detroit; MemGuide re-
turns that it is hard to choose
one.Compares multiple partner-gaming memo-
ries and accepts the memory that explic-
itly entails favorite/shared enjoyment: Jolene
says she loves playing It Takes Two together.
What outdoor ac-
tivity did Jolene
suggest doing to-
gether with Deb-
orah?Surfing Query Reformulation re-
turns none; Self-Reflection
and MemGuide say no
outdoor activity was sug-
gested; ReAct returns “I
don’t know.”Uses a later breadth decomposition that
searches for an activity Jolene is learning
and wants to do with Deborah; unifies the
question with the surfing exchange and the
invitation to “come with me.”
What area was
hit by a flood?West
CountyQuery Reformulation, Self-
Reflection, and MemGuide
return “my old area”; ReAct
says the memory is unavail-
able.Treats “my old area” as an unresolved re-
ferring expression, then grounds it with an
earlier memory where John identifies his old
area as West County.
What fun ac-
tivity did Evan
mention doing
in July 2023?Skiing Query Reformulation and
Self-Reflection return
“drove somewhere fun”;
ReAct returns a mystery
novel; MemGuide returns
The Great Gatsby.Uses temporal alignment: a later August
memory says Evan drove somewhere fun
“last month,” and the follow-up memory iden-
tifies the activity as skiing, which places the
activity in July 2023.
Table 6: Qualitative examples where all four forward-reasoning baselines fail but GOAL-MEMreturns
the correct answer. The examples illustrate different multi-hop patterns: location-granularity bridging,
temporal event alignment, distractor disambiguation, breadth search over alternative decompositions,
and grounding of underspecified referring expressions.
The failure cases highlight that multi-hop memory QA is not only a retrieval-volume problem. The
relevant evidence is often present in the retrieved pool, but the final answer requires an additional
grounding step: resolving a location to the requested granularity, linking a relative time expression
to an event month, choosing the memory that entails the queried relation, or replacing a referring
expression with its named antecedent. GOAL-MEM’s decomposition and unification process makes
these intermediate requirements explicit, which helps prevent the model from returning a retrieved
but ungrounded span as the final answer.
F Forward Reasoning Baseline Implementations
This appendix describes the forward reasoning baselines used in RQ2: Query Reformulation, Self-
Reflection, ReAct, and MemGuide. All four methods are implemented as wrappers around the same
retrieval-augmented memory backbone Bused by GOAL-MEM. Thus, these baselines do not change
the memory storage, indexing, or update procedure of the underlying backbone; they only change
how queries are generated, how retrieved memory units are accumulated or filtered, and how the final
answer is produced.
19

Shared retrieval protocol.For a controlled comparison, all baselines use the same inference LLM,
decoding configuration, memory backbone B, and answer-generation prompt as the corresponding
GOAL-MEMrun. We use a shared retrieved-memory cap of
kret
max= 60,(7)
where kret
maxdenotes the maximum number of unique retrieved memory units that can be exposed to
the final answer stage for one question. This retrieval cap is distinct from Kmaxin the complexity
analysis, where Kmaxdenotes the maximum number of subgoals produced by a goal decomposition.
For iterative baselines, the same cap is distributed across retrieval actions: each retrieval action calls
Bretrieve with a bounded per-query fan-out, retrieved units are deduplicated, and the merged pool is
clipped to at mostkret
maxunits before final answer generation or filtering.
Query Reformulation.The Query Reformulation baseline follows the query-rewriting view of
retrieval-augmented generation, where an LLM adapts the user input before retrieval to reduce the
gap between the surface form of the question and the information needed by the reader [ 20]. Given
the user utterance ut, the LLM generates nqr= 5diverse retrieval queries. The original question is
also included as a retrieval query, giving
Qqr(ut) ={u t} ∪Rewrite LLM(ut;nqr= 3).(8)
The prompt asks the LLM to target different entities, events, temporal expressions, synonyms, and
background aspects that may appear in memory, while preserving important entities and events from
the original question. The retrieved results from all queries are merged, deduplicated, and clipped to
kret
max= 60 . The final answer is then generated using the standard answer prompt over the resulting
memory pool. This baseline is intentionally simple: it performs no symbolic decomposition, no
unification, no missing-slot refinement, and no post-retrieval reranking.
Self-Reflection.The Self-Reflection baseline adapts the reflective-agent principle of Reflexion to
the single-question memory retrieval setting. Reflexion uses verbal feedback and an episodic memory
buffer to improve later decisions; in our evaluation, we use the same forward-style reflection idea
without carrying reflections across evaluation questions, so that the baseline does not receive extra
persistent feedback unavailable to the other methods. The method first retrieves memories using
the original question. At each reflection turn, the LLM is shown the question, all previously issued
queries, and the accumulated retrieved memory pool. It outputs a JSON decision with three fields:
whether the pool is sufficient, one concise next query if more evidence is needed, and a short rationale.
If the pool is judged sufficient, the loop terminates; otherwise, the follow-up query is sent to Bretrieve ,
and the newly retrieved units are merged and deduplicated with the existing pool.
We set the maximum number of reflection turns to Tsr
max= 5. The method can therefore terminate
because the LLM judges the memory pool sufficient, because no new retrieval query is proposed,
because retrieval returns no new memory units, or because the turn limit is reached. The final answer
is generated from the accumulated memory pool after clipping to kret
max= 60 . Unlike GOAL-MEM,
Self-Reflection does not maintain typed variables or substitutions and does not verify that a retrieved
memory entails a particular subgoal; each additional query is generated forward from the current
question and evidence pool.
ReAct.The ReAct baseline follows the reasoning-and-acting paradigm in which the LLM alternates
between natural-language reasoning traces and actions that interact with an external environment [ 35].
In our memory-QA adaptation, the external environment is the shared memory backbone B. At each
step, the model emits exactly oneThoughtand oneAction. The action space is restricted to
Retrieve[query]orFinish[answer].(9)
When the model selects Retrieve[query] , the query is sent to Bretrieve , and the returned memories
are appended to the trajectory as an Observation . When the model selects Finish[answer] , the
answer is returned directly. This preserves the ReAct design choice of interleaving reasoning traces
with environment actions, while making the environment identical to the memory backend used by
GOAL-MEM.
We set the maximum number of ReAct steps to Sreact
max= 5. If the model output does not match the
required Thought /Action format, the implementation retries once with a stricter format reminder.
20

If the step limit is reached before a Finish action, the model is forced to produce a final Finish
answer using only the observations collected so far. Retrieved observations are deduplicated, and
the accumulated memory pool is clipped to kret
max= 60 . As with Self-Reflection, ReAct is a
forward reasoning baseline: it decides the next retrieval action from the current trajectory, rather than
decomposing the target answer into subgoals and verifying substitutions.
MemGuide.MemGuide is a memory-selection method for goal-oriented multi-session LLM agents.
The original method introduces a two-stage intent-driven framework: Intent-Aligned Retrieval
retrieves memory units that share the current task intent, and Missing-Slot Guided Filtering reranks
candidate memory units according to their marginal gain in filling missing task slots [ 5]. Our
implementation follows this two-stage structure while adapting it to the unified memory-backbone
interface used in our experiments.
First, an intent-capture prompt maps the question to a compact intent description and generates
nintent = 5intent-aligned retrieval queries. The original question is included as an additional seed
query. These queries retrieve an initial candidate pool from B. Second, a missing-slot reasoner
inspects the current candidate pool and decides whether the pool is sufficient. If not, it enumerates
concrete missing fields, such as a date, location, counterpart person, event outcome, or relation, and
proposes at most Fmg
max= 5 follow-up queries for the next refinement round. We allow at most
Rmg
max= 3missing-slot refinement rounds.
After retrieval, MemGuide applies slot-guided filtering. Candidate memory units are scored by an
LLM judger according to how directly they answer the question or fill the identified missing slots.
The implementation combines the LLM score with the retrieval prior,
s(f) =λs LLM(f) + (1−λ)s ret(f), λ= 0.7,(10)
where sret(f)favors earlier retrieved candidates. The final selected memory pool is sorted by this
score and clipped to kret
max= 60 before answer generation. The candidate pool is also clipped to the
same cap after retrieval and deduplication.
This adaptation is faithful to the algorithmic role of MemGuide as an intent- and slot-aware forward
memory selector, but it does not assume MemGuide’s original task-oriented-dialogue memory format
or its task-specific fine-tuned filtering model. Instead, all candidate units are retrieved from the
same memory backbone Bused by the other baselines, and the slot filter is implemented with the
shared inference LLM. This keeps the comparison controlled: MemGuide receives the benefit of
intent-aligned retrieval and missing-slot filtering, but it does not receive a method-specific memory
representation or extra supervised training signal unavailable to the other methods.
G Broader Impacts
Improving the coherence and factual grounding of long-horizon conversational agents has clear posi-
tive societal value in applications such as healthcare assistance, education, and personal productivity.
By anchoring responses in retrieved memory rather than parametric knowledge, GOAL-MEMreduces
the risk of hallucinated but plausible-sounding responses. However, as with all existing RAG-based
memory works and LLM-based systems, there may be concerns about the possibility of mistakes and
the risk of over-reliance on the system’s outputs.
H Limitations
As we observed in Section 5.1, GOAL-MEMconsistently improves the performance of existing
RAG-based agentic memory systems. At the same time, since it builds upon an underlying memory
and retrieval backbone and leverages an LLM for components such as goal decomposition and answer
generation, its effectiveness naturally depends on the quality of these components.
I Complete Prompts
In this appendix, we disclose the prompt we used in this experiment for GOAL-MEM.
21

I.1 Goal Decomposition
I.1.1 Initial decomposition
system: |-
system_commonsense: |-
You will be provided only with a question about a person.
Generate one concise commonsense goal decomposition that can guide later evidence retrieval and answer
verification.
Requirements:
- Do NOT use any specific retrieved facts or invented details.
- Use general human commonsense only.
- Keep premises/subgoals abstract and checkable from future conversation evidence.
- The decomposition should help identify what evidence would be sufficient to answer the question.
- Phrase each premise as the kind of EVENT, ACTION, ATTRIBUTE, or ATTITUDE a participant would naturally
mention in a casual personal conversation.
- DO NOT phrase premises as META-CLAIMS about the conversation itself. FORBIDDEN openings include: "the
conversation explicitly states ...", "the conversation gives ...", "the message names ...", "a fact
says ...", "the speaker explicitly mentions ...".
- Prefer concrete event/attribute language ("the person visits Stamford", "the person talks about a
wedding in December") over evidentiary framing ("the conversation contains a statement about a
wedding in December").
- Preserve exact entity anchors from the question and make answer variables explicit as (x:type), (y:
type), etc.
Goal decomposition:
Step 1 - Identify the central entities in the question: subject, object/topic/instrument/place/activity,
named person, time qualifier, and requested answer type.
Step 2 - Identify any variables that must be substituted to answer the question. Write variables as (x:
type), (y:type), etc.
Step 3 - Generate atomic subgoals. Each subgoal must be factual, verifiable, and phrased as the kind of
event, action, attribute, or attitude that could naturally appear in a personal conversation memory.
Step 4 - Preserve exact entity anchors from the question. Closely related but distinct entities are not
substitutes (e.g. guitar vs violin; Korean class vs Korea trip; current role vs previous role).
Step 5 - Use facts only as optional grounding context. Do NOT answer the question and do NOT invent
concrete substitutions.
Output format (exactly):
Goal: <one goal for the question, using typed variables where needed>
Rule: IF <subgoal 1> AND <subgoal 2> AND ... THEN <goal is satisfied>
Variables:
- (<variable:type>): <what this variable must be substituted with, or "none" if no answer variable is
needed>
Subgoals:
- <atomic subgoal 1>
- <atomic subgoal 2>
Premises:
- <same text as subgoal 1, repeated for backward compatibility with retrieval code>
- <same text as subgoal 2, repeated for backward compatibility with retrieval code>
input_template_commonsense: |-
Question: {{QUESTION}}
{% if BACKGROUND_SUMMARY %}
Background (per-speaker conversation summary):
{{BACKGROUND_SUMMARY}}
Note on the background: this is OPTIONAL context to help you ground the
speakers –- it is NOT exhaustive and NOT a fact source. Do NOT restrict your
rule to only what appears here, do NOT cite the summary as evidence, and do
NOT treat absence from the summary as absence from the conversation. General
commonsense reasoning should still be the primary driver of the rule; use the
background only as light supporting context.
{% endif %}
{% if ASSUMED_KNOWN_INFO %}
Known Info (assume true): {{ASSUMED_KNOWN_INFO|join(’\n’)}}
{% endif %}
few-shot:
- input:
QUESTION: |-
What career field is Caroline likely to pursue in her education?
output: |-
Goal: (Caroline) is likely to pursue (x:career_field) in her education.
Rule: IF Caroline is actively exploring (x:career_field) AND Caroline has expressed direct interest
in (x:career_field) AND Caroline finds work in (x:career_field) meaningful, THEN Caroline is likely
to pursue (x:career_field) in her education.
Variables:
- (x:career_field): the education or career field Caroline is likely to pursue
22

Subgoals:
- Caroline is actively exploring (x:career_field).
- Caroline has expressed direct interest in (x:career_field).
- Caroline finds work in (x:career_field) meaningful.
Premises:
- Caroline is actively exploring (x:career_field).
- Caroline has expressed direct interest in (x:career_field).
- Caroline finds work in (x:career_field) meaningful.
- input:
QUESTION: |-
Would Melanie be likely to attend an outdoor fitness event?
output: |-
Goal: (Melanie) is likely to attend an outdoor fitness event.
Rule: IF Melanie regularly engages in outdoor physical activity AND Melanie is training for a
competitive running event, THEN Melanie would likely be interested in attending an outdoor fitness
event.
Variables:
- none: no answer variable is needed because the question asks for a yes/no judgment
Subgoals:
- Melanie regularly engages in outdoor physical activity.
- Melanie is training for a competitive running event.
Premises:
- Melanie regularly engages in outdoor physical activity.
- Melanie is training for a competitive running event.
- input:
QUESTION: |-
Did Caroline use art as a way to explore personal identity?
output: |-
Goal: (Caroline) used art as a way to explore personal identity.
Rule: IF Caroline has used art as a medium to explore identity AND Caroline values art as a
meaningful practice, THEN Caroline used art as a way to explore personal identity.
Variables:
- none: no answer variable is needed because the question asks for a yes/no judgment
Subgoals:
- Caroline has used art as a medium to explore identity.
- Caroline values art as a meaningful practice.
Premises:
- Caroline has used art as a medium to explore identity.
- Caroline values art as a meaningful practice.
- input:
QUESTION: |-
What career field is Caroline likely to pursue in her education?
output: |-
Goal: (Caroline) is likely to pursue (x:career_field) in her education.
Rule: IF Caroline consistently expresses sustained interest in (x:career_field) AND Caroline takes
concrete steps toward training or education in (x:career_field), THEN Caroline is likely to pursue (x:
career_field) in her education.
Variables:
- (x:career_field): the education or career field Caroline is likely to pursue
Subgoals:
- Caroline consistently expresses sustained interest in (x:career_field).
- Caroline takes concrete steps toward training or education in (x:career_field).
Premises:
- Caroline consistently expresses sustained interest in (x:career_field).
- Caroline takes concrete steps toward training or education in (x:career_field).
- input:
QUESTION: |-
Would Melanie be likely to attend an outdoor fitness event?
output: |-
Goal: (Melanie) is likely to attend an outdoor fitness event.
Rule: IF Melanie regularly engages in outdoor exercise AND Melanie shows motivation for fitness-
related activities, THEN Melanie is likely to attend an outdoor fitness event.
Variables:
- none: no answer variable is needed because the question asks for a yes/no judgment
Subgoals:
- Melanie regularly engages in outdoor exercise.
- Melanie shows motivation for fitness-related activities.
Premises:
- Melanie regularly engages in outdoor exercise.
- Melanie shows motivation for fitness-related activities.
I.1.2 Next-hop decomposition
system: |-
You will be provided with a question about a person and previously generated inference rules.
Generate one NEW commonsense goal decomposition that is complementary, not redundant, with prior rules.
Requirements:
23

- Do NOT use specific retrieved facts or invented concrete details.
- Use general human commonsense only.
- Avoid repeating any prior rule verbatim or semantically.
- Keep premises/subgoals abstract and checkable from future conversation evidence.
- Phrase each premise as the kind of EVENT, ACTION, ATTRIBUTE, or ATTITUDE a participant would naturally
mention in a casual personal conversation.
- DO NOT phrase premises as META-CLAIMS about the conversation itself. FORBIDDEN openings include: "the
conversation explicitly states ...", "the conversation gives ...", "the message names ...", "a fact
says ...", "the speaker explicitly mentions ...".
- Prefer concrete event/attribute language over evidentiary framing.
- Preserve exact entity anchors and make answer variables explicit as (x:type), (y:type), etc.
Output format (exactly):
Goal: <one goal for the question, using typed variables where needed>
Rule: IF <subgoal 1> AND <subgoal 2> AND ... THEN <goal is satisfied>
Variables:
- (<variable:type>): <what this variable must be substituted with, or "none" if no answer variable is
needed>
Subgoals:
- <atomic subgoal 1>
- <atomic subgoal 2>
Premises:
- <same text as subgoal 1, repeated for backward compatibility with retrieval code>
- <same text as subgoal 2, repeated for backward compatibility with retrieval code>
Difference From Prior Decompositions: <one concise sentence explaining what is new>
input_template_commonsense: |-
Question: {{QUESTION}}
{% if BACKGROUND_SUMMARY %}
Background (per-speaker conversation summary):
{{BACKGROUND_SUMMARY}}
Note on the background: this is OPTIONAL context to help you ground the
speakers –- it is NOT exhaustive and NOT a fact source. Do NOT restrict your
rule to only what appears here, do NOT cite the summary as evidence, and do
NOT treat absence from the summary as absence from the conversation. General
commonsense reasoning should still be the primary driver of the rule; use the
background only as light supporting context.
{% endif %}
{% if ASSUMED_KNOWN_INFO %}
Known Info (assume true): {{ASSUMED_KNOWN_INFO|join(’\n’)}}
{% endif %}
Existing rules/decompositions that should NOT be repeated:
{{AXIOMS|join(’\n’)}}
New goal decomposition:
few-shot:
- input:
QUESTION: |-
What career field is Caroline likely to pursue in her education?
AXIOMS:
- IF Caroline consistently expresses sustained interest in (x:career_field) AND Caroline takes
concrete steps toward training or education in (x:career_field), THEN Caroline is likely to pursue (x:
career_field) in her education.
output: |-
Goal: (Caroline) is likely to pursue (x:career_field) in her education.
Rule: IF Caroline repeatedly describes (x:career_field) as meaningful or personally rewarding, THEN
evidence of that repeated valuation would be sufficient additional support for inferring Caroline may
pursue (x:career_field).
Variables:
- (x:career_field): the education or career field Caroline is likely to pursue
Subgoals:
- Caroline repeatedly describes (x:career_field) as meaningful or personally rewarding.
Premises:
- Caroline repeatedly describes (x:career_field) as meaningful or personally rewarding.
Difference From Prior Decompositions: The new decomposition uses repeated valuation rather than
training steps.
- input:
QUESTION: |-
Would Melanie be likely to attend an outdoor fitness event?
AXIOMS:
- IF Melanie regularly engages in outdoor exercise AND Melanie shows motivation for fitness-
related activities, THEN Melanie is likely to attend an outdoor fitness event.
output: |-
Goal: (Melanie) is likely to attend an outdoor fitness event.
Rule: IF Melanie voluntarily participates in social or organized fitness activities, THEN evidence
of that participation would be sufficient complementary support for judging likely attendance at a
future outdoor fitness event.
Variables:
- none: no answer variable is needed because the question asks for a yes/no judgment
Subgoals:
- Melanie voluntarily participates in social or organized fitness activities.
24

Premises:
- Melanie voluntarily participates in social or organized fitness activities.
Difference From Prior Decompositions: The new decomposition uses social or organized participation
rather than general outdoor exercise and motivation.
I.2 Unification
system: |-
You are answering questions about people based on facts extracted from their personal conversations.
memory: a self-contained sentence describing an event, action, state, or attribute related to the person.
For example, "Caroline is attending therapy sessions" or "Melanie enjoys running in the park".
This prompt implements the VERIFIABLE UNIFICATION stage of Goal-Mem.
Task:
- Given the question, optional goal, optional subgoals, general rules, current substitutions/known info,
and retrieved facts, decide whether the retrieved facts ground the goal.
- Compute any candidate variable substitutions theta = {x / e, ...} that are supported by retrieved
facts.
- Generate the final answer only if the goal is fully grounded by type-consistent, non-conflicting,
entailed substitutions.
- If grounding fails, report unresolved subgoals and known grounded information.
If no explicit Subgoals section is provided, treat the premises of the General Rule as the active
subgoals. If no explicit Goal is provided, infer the goal from the Question and General Rule.
ENTITY-ANCHOR CHECK (do this BEFORE selecting facts):
1. Identify the question’s central entities: its subject, the specific object/topic/instrument/place/
activity it asks about, and any qualifier ("previous", "first", named person, time window, etc.).
2. Use ONLY facts that mention those exact entities or facts that can be unified through an explicit
variable in a subgoal. Closely related but distinct entities (guitar vs violin; Korean class on
Wednesday vs trip to Korea; current role vs previous role; one party’s brownies vs another party’s
cake) are NOT substitutes.
3. If no fact mentions the question’s central entity and no subgoal variable can validly bridge to it,
the goal is not grounded. Do not pick a thematically similar fact as a fallback.
UNIFICATION PROCESS:
1. Apply any Current Substitution / Known Info to the active subgoals before evaluating new facts.
2. For each subgoal psi_i and candidate fact m_j, propose substitutions only for explicit variables
such as (x:drink) or (z:cafe).
3. Type consistency: accept x/e only if e is an instance of the variable type or has a type that
entails it in context. For example, Kyoto Latte may fill (x:drink); guitar may not fill (x:instrument
asked as violin) unless the subgoal variable is typed broadly as instrument and the question does
not require violin.
4. Equality with existing substitutions: if x is already bound in the current substitution, any new
binding for x must be the same entity in context. Reject conflicting bindings.
5. Logical entailment: after applying the candidate substitution, the retrieved fact must entail the
grounded subgoal. Topical similarity is not enough.
6. Simultaneous consistency: perform the check across all active subgoals and facts as a set. Do not
let the order of facts decide which conflicting substitution wins.
7. Conflict handling: if facts ground the same required variable with incompatible values and the
conflict cannot be resolved from the facts alone, answer "I don’t know".
ANSWER RULES:
- Your answer must be based on the provided facts and general rules/subgoals. State the used facts and
rules explicitly in your reasoning.
- Indicate the number of facts and general rules that you used to answer the question.
- If the question requires a specific value (date, name, number, location, object, etc.) and the value
is not grounded by a successful substitution or an explicit fact, answer "I don’t know".
- Partial or hedged answers ("around that time", "probably", "likely", "it seems") are FORBIDDEN unless
the question itself asks for likelihood/preference and the likelihood judgment is grounded by the
rule.
- If the current facts and rules are not sufficient to answer the question, Final Answer must be "I don’
t know".
- Do NOT make assumptions beyond the provided facts, rules, and current substitutions.
- Do NOT output Missing Info or Relation Type here. Use Unresolved Subgoals and Known Info only;
refinement is a separate prompt.
OUTPUT FORMAT –- you MUST use exactly this structure:
Unification Status: <satisfied|unsatisfied|conflict>
Substitution: <{x / entity, y / entity, ...} or {} or "none">
Grounded Subgoals:
- <subgoal> <= <supporting fact # or Known Info>; bindings: <bindings or none>
Unresolved Subgoals:
- <subgoal> -- <why it is not grounded, or "none">
Reasoning: <cite the specific facts and rules that support your answer, or explain why the facts are
insufficient; include Used N facts and M general rules>
25

Final Answer: <the shortest possible phrase directly supported by the facts/substitution, or "I don’t
know">
Known Info: <concise semicolon-separated statements already grounded, or "none">
input_template: |-
Question: {{QUESTION}}
{% if GOAL %}
Goal: {{GOAL}}
{% endif %}
{% if SUBGOALS %}
Subgoals:
{{ SUBGOALS|join(’\n’) }}
{% endif %}
{% if CURRENT_SUBSTITUTION %}
Current Substitution: {{CURRENT_SUBSTITUTION}}
{% endif %}
{% if ASSUMED_KNOWN_INFO %}
Known Info (assume true for this depth): {{ ASSUMED_KNOWN_INFO|join(’\n’) }}
{% endif %}
{% if HINTS %}
Hints: {{ HINTS|join(’\n’) }}
{% endif %}
{% if GENERAL_RULE %}
General Rule: {{GENERAL_RULE}}
{% endif %}
{% if FACTS %}
Facts: {{ FACTS|join(’\n’) }}
{% endif %}
few-shot:
- input:
QUESTION: |-
What career field is Caroline likely to pursue in her education?
Goal: (Caroline) is likely to pursue (x:career_field) in her education.
Subgoals:
- Caroline is actively exploring (x:career_field).
- Caroline has expressed direct interest in (x:career_field).
- Caroline finds work in (x:career_field) meaningful.
General Rule: IF Caroline is actively exploring (x:career_field) AND Caroline has expressed direct
interest in (x:career_field) AND Caroline finds work in (x:career_field) meaningful, THEN Caroline
is likely to pursue (x:career_field) in her education.
Facts: 1- (Caroline, exploring, counseling or mental health career options) [event point_in_time:
15 July, 2023] [spoken at: 1:51 pm on 15 July, 2023]
2- (Caroline, finds, counseling and mental health work tough but rewarding) [event point_in_time:
6 July, 2023] [spoken at: 8:18 pm on 6 July, 2023]
3- (Caroline, interested_in, counseling) [event point_in_time: 3 July, 2023] [spoken at: 1:36 pm
on 3 July, 2023]
output: |-
Unification Status: satisfied
Substitution: {x / counseling and mental health}
Grounded Subgoals:
- Caroline is actively exploring (x:career_field). <= Fact 1; bindings: x / counseling and mental
health
- Caroline has expressed direct interest in (x:career_field). <= Fact 3; bindings: x / counseling
- Caroline finds work in (x:career_field) meaningful. <= Fact 2; bindings: x / counseling and mental
health
Unresolved Subgoals:
- none
Reasoning: Fact 1 states Caroline is exploring counseling or mental health career options, Fact 3
states she is interested in counseling, and Fact 2 states she finds counseling and mental health work
tough but rewarding. These facts jointly satisfy the three subgoals under the consistent
substitution x / counseling and mental health. Used 3 facts and 1 general rule.
Final Answer: counseling and mental health
Known Info: Caroline is exploring counseling or mental health career options; Caroline is interested
in counseling; Caroline finds counseling and mental health work rewarding
- input:
QUESTION: |-
What is the name of Melanie’s dog?
Goal: (Melanie’s dog) has name (x:person_or_pet_name).
Subgoals:
- Melanie has a dog.
- Melanie’s dog has name (x:person_or_pet_name).
General Rule: IF Melanie has a dog AND Melanie’s dog has name (x:person_or_pet_name), THEN x is
the answer.
Facts: 1- (Melanie, has_pet, dog) [event point_in_time: 5 August, 2023] [spoken at: 11:20 am on 5
August, 2023]
2- (Melanie, enjoys, walking her dog in the park) [event point_in_time: 5 August, 2023] [spoken at:
11:20 am on 5 August, 2023]
output: |-
Unification Status: unsatisfied
Substitution: {}
26

Grounded Subgoals:
- Melanie has a dog. <= Fact 1; bindings: none
Unresolved Subgoals:
- Melanie’s dog has name (x:person_or_pet_name). -- no fact states the dog’s name
Reasoning: Facts 1 and 2 confirm that Melanie has a dog and enjoys walking it, but neither fact
states the dog’s name. The answer variable x is not grounded. Used 1 fact and 1 general rule.
Final Answer: I don’t know
Known Info: Melanie has a dog; Melanie enjoys walking her dog in the park
I.3 Refinement
system: |-
You are refining unresolved subgoals for Goal-Mem’s depth loop.
You will be provided with a question, the original goal, the current rule/subgoals, the current
substitution or known grounded information, retrieved facts, and the unresolved subgoals.
This prompt implements SUBGOAL REFINEMENT VIA BACKWARD CHAINING.
Objective:
- Do NOT answer the user question.
- Do NOT redo unification.
- For each unresolved subgoal, generate a more basic requisite antecedent subgoal whose grounding would
help satisfy the unresolved subgoal.
- Produce retrieval-ready Missing Info and Relation Type outputs for the next memory retrieval hop.
Refinement principles:
1. Target the exact unresolved subgoal. Preserve all central entities and qualifiers from the question
and from the unresolved subgoal.
2. Do not merely paraphrase the unresolved subgoal. Generate an antecedent that would make the
unresolved part checkable.
Example: unresolved "(x:drink) served in (z:cafe visited last week)" can refine to "Alice visited (z:
cafe) last week" if z is unknown.
3. Keep unresolved variables explicit as (x:type), (y:type), etc. Reuse the same variable names when the
refined subgoal is intended to ground the same variable.
4. Respect existing substitutions. If x is already bound, use the bound entity unless the unification
trace says the binding is conflicted.
5. Do not invent constants. Only use constants that appear in the question, Known Info, Current
Substitution, or retrieved facts.
6. Avoid repeated refinement. If Previously Retrieved Missing Info or Previously Refined Subgoals are
provided, the new refinement MUST take a different angle.
7. If no useful new refinement is possible, set Refinement Status to stop and explain why.
Relation Type selection:
- temporal: the missing evidence is about when something happened or the sequence of events.
- causal: the missing evidence is about why something happened, what caused it, or what resulted from it.
- semantic: the missing evidence is about the same specific topic, activity, object, name, preference,
property, or event.
- spatial: the missing evidence is about where something happened, a place identity, or what else
happened at the same location.
OUTPUT FORMAT –- you MUST use exactly this structure:
Refinement Status: <refine|stop>
Missing Info: <specific information needed for the next retrieval hop>
Relation Type: <temporal|causal|semantic|spatial>
Refined Subgoals:
- <atomic subgoal to retrieve next, or "none">
Retrieval Queries:
- <natural-language query corresponding to refined subgoal 1, or "none">
Rationale: <one concise sentence explaining why the refined subgoal is necessary>
input_template: |-
Question: {{QUESTION}}
{% if GOAL %}
Goal: {{GOAL}}
{% endif %}
{% if GENERAL_RULE %}
General Rule: {{GENERAL_RULE}}
{% endif %}
{% if SUBGOALS %}
Current Subgoals:
{{SUBGOALS|join(’\n’)}}
{% endif %}
{% if CURRENT_SUBSTITUTION %}
Current Substitution: {{CURRENT_SUBSTITUTION}}
{% endif %}
27

{% if KNOWN_INFO %}
Known Info:
{{KNOWN_INFO|join(’\n’)}}
{% endif %}
{% if UNIFICATION_TRACE %}
Unification Trace:
{{UNIFICATION_TRACE}}
{% endif %}
{% if UNRESOLVED_SUBGOALS %}
Unresolved Subgoals:
{{UNRESOLVED_SUBGOALS|join(’\n’)}}
{% endif %}
{% if PREVIOUSLY_RETRIEVED_MISSING_INFO %}
Previously Retrieved Missing Info (do NOT repeat or paraphrase):
{{PREVIOUSLY_RETRIEVED_MISSING_INFO|join(’\n’)}}
{% endif %}
{% if PREVIOUSLY_REFINED_SUBGOALS %}
Previously Refined Subgoals (do NOT repeat or paraphrase):
{{PREVIOUSLY_REFINED_SUBGOALS|join(’\n’)}}
{% endif %}
{% if FACTS %}
Retrieved Facts:
{{FACTS|join(’\n’)}}
{% endif %}
Refinement:
few-shot:
- input:
QUESTION: |-
Alice: I like the cafe I went to last week, what drink should I try this time?
GOAL: |-
Recommend (x:drink) to (Alice) from the menu of the cafe she went to last week.
GENERAL_RULE: |-
IF Alice likes (y:flavor) AND (x:drink) contains (y:flavor) AND (x:drink) is served in (z:cafe
visited last week), THEN recommend (x:drink) to Alice.
SUBGOALS:
- Alice likes (y:flavor).
- (x:drink) contains (y:flavor).
- (x:drink) is served in (z:cafe visited last week).
CURRENT_SUBSTITUTION: "{x / Kyoto Latte, y / Matcha Powder}"
KNOWN_INFO:
- Alice likes matcha.
- Kyoto Latte contains Matcha Powder.
UNRESOLVED_SUBGOALS:
- (x:drink) is served in (z:cafe visited last week). -- z is not grounded; no fact identifies the
cafe Alice visited last week
FACTS:
- "1- The Momoco cafe advertised a Kyoto Latte as its seasonal drink."
output: |-
Refinement Status: refine
Missing Info: Which cafe Alice visited last week, so z can be grounded for the served-in subgoal.
Relation Type: spatial
Refined Subgoals:
- Alice visited (z:cafe) last week.
Retrieval Queries:
- Alice visited cafe last week which cafe
Rationale: The unresolved served-in subgoal cannot be checked until the variable z:cafe visited last
week is grounded.
- input:
QUESTION: |-
What is the name of Melanie’s dog?
GOAL: |-
(Melanie’s dog) has name (x:person_or_pet_name).
GENERAL_RULE: |-
IF Melanie has a dog AND Melanie’s dog has name (x:person_or_pet_name), THEN x is the answer.
CURRENT_SUBSTITUTION: "{}"
KNOWN_INFO:
- Melanie has a dog.
UNRESOLVED_SUBGOALS:
- Melanie’s dog has name (x:person_or_pet_name). -- no fact states the dog’s name
FACTS:
- "1- (Melanie, has_pet, dog) [event point_in_time: 5 August, 2023] [spoken at: 11:20 am on 5
August, 2023]"
- "2- (Melanie, enjoys, walking her dog in the park) [event point_in_time: 5 August, 2023] [spoken
at: 11:20 am on 5 August, 2023]"
output: |-
Refinement Status: refine
Missing Info: The name Melanie uses for her dog.
Relation Type: semantic
Refined Subgoals:
- Melanie mentions her dog by name as (x:person_or_pet_name).
28

Retrieval Queries:
- Melanie dog name mentions her dog by name
Rationale: The dog is established, but the answer variable x remains unbound until a memory names
the dog.
29
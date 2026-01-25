# Connect the Dots: Knowledge Graph-Guided Crawler Attack on Retrieval-Augmented Generation Systems

**Authors**: Mengyu Yao, Ziqi Zhang, Ning Luo, Shaofei Li, Yifeng Cai, Xiangqun Chen, Yao Guo, Ding Li

**Published**: 2026-01-22 05:59:42

**PDF URL**: [https://arxiv.org/pdf/2601.15678v1](https://arxiv.org/pdf/2601.15678v1)

## Abstract
Retrieval-augmented generation (RAG) systems integrate document retrieval with large language models and have been widely adopted. However, in privacy-related scenarios, RAG introduces a new privacy risk: adversaries can issue carefully crafted queries to exfiltrate sensitive content from the underlying corpus gradually. Although recent studies have demonstrated multi-turn extraction attacks, they rely on heuristics and fail to perform long-term extraction planning. To address these limitations, we formulate the RAG extraction attack as an adaptive stochastic coverage problem (ASCP). In ASCP, each query is treated as a probabilistic action that aims to maximize conditional marginal gain (CMG), enabling principled long-term planning under uncertainty. However, integrating ASCP with practical RAG attack faces three key challenges: unobservable CMG, intractability in the action space, and feasibility constraints. To overcome these challenges, we maintain a global attacker-side state to guide the attack. Building on this idea, we introduce RAGCRAWLER, which builds a knowledge graph to represent revealed information, uses this global state to estimate CMG, and plans queries in semantic space that target unretrieved regions. In comprehensive experiments across diverse RAG architectures and datasets, our proposed method, RAGCRAWLER, consistently outperforms all baselines. It achieves up to 84.4% corpus coverage within a fixed query budget and deliver an average improvement of 20.7% over the top-performing baseline. It also maintains high semantic fidelity and strong content reconstruction accuracy with low attack cost. Crucially, RAGCRAWLER proves its robustness by maintaining effectiveness against advanced RAG systems employing query rewriting and multi-query retrieval strategies. Our work reveals significant security gaps and highlights the pressing need for stronger safeguards for RAG.

## Full Text


<!-- PDF content starts -->

Connect the Dots: Knowledge Graph–Guided Crawler Attack on
Retrieval-Augmented Generation Systems
Mengyu Yao∗, Ziqi Zhang†, Ning Luo†, Shaofei Li∗, Yifeng Cai∗, Xiangqun Chen∗, Yao Guo∗and Ding Li∗
∗Key Laboratory of High-Confidence Software Technologies (MOE), School of Computer Science, Peking University
†Department of Computer Science, University of Illinois Urbana-Champaign
Abstract—Retrieval-augmented generation (RAG) systems inte-
grate document retrieval with large language models and have
been widely adopted. However, in privacy-related scenarios
(e.g., medical diagnosis), RAG introduces a new privacy risk:
adversaries can issue carefully crafted queries to exfiltrate sen-
sitive content from the underlying corpus gradually. Although
recent studies have demonstrated multi-turn extraction attacks,
they rely on heuristics and fail to perform long-term extraction
planning. As a result, they typically stagnate after a few rounds
and exfiltrate only a small fraction of the private corpus. To
address these limitations, we formulate the RAG extraction
attack as an adaptive stochastic coverage problem (ASCP).
In ASCP, each query is treated as a probabilistic action that
aims to maximize conditional marginal gain (CMG), enabling
principled long-term planning under uncertainty. However,
integrating ASCP with practical RAG attack faces three key
challenges: unobservable CMG, intractability in the action
space, and feasibility constraints. To overcome these challenges,
we maintain a global attacker-side state to guide the attack.
Building on this idea, we introduce RAGCRAWLER, which
builds a knowledge graph to represent revealed information,
uses this global state to estimate CMG, and plans queries
in semantic space that target unretrieved regions. It also
generates benign-looking prompts that elicit the remaining
content while preserving stealth. In comprehensive experi-
ments across diverse RAG architectures and datasets, our
proposed method, RAGCRAWLER, consistently outperforms
all baselines. It achieves up to 84.4% corpus coverage within
a fixed query budget and deliver an average improvement of
20.7% over the top-performing baseline. It also maintains high
semantic fidelity and strong content reconstruction accuracy
with low attack cost. Crucially, RAGCRAWLERproves its
robustness by maintaining effectiveness against advanced RAG
systems employing query rewriting and multi-query retrieval
strategies. Our work reveals significant security gaps and
highlights the pressing need for stronger safeguards for RAG.
1. Introduction
Retrieval-Augmented Generation (RAG) has emerged as
a powerful paradigm that integrates information retrieval
with large language models (LLMs), allowing systems to
dynamically access and utilize external knowledge basesfor tasks such as question answering, summarization, and
decision support [1], [2]. By providing timely, domain-
specific evidence beyond a model’s parametric memory,
RAG has seen rapid adoption across settings ranging from
enterprise knowledge management [3] to healthcare [4], and
customer service [5].
RAG’s retrieval capabilities introduce significant privacy
risks, as many of the data sources it accesses or stores
contain confidential information. Recent studies demonstrate
that sensitive data can leak from RAG by eliciting verbatim
passages [6], [7], exposing personal identifiers from clini-
cal notes or confidential clauses from internal documents.
Such leaks can lead to non-compliance with GDPR and
HIPAA [8], [9], resulting in legal, financial, and reputational
harm [10]. Abusing the retrieval capabilities, adversaries can
exploit the retriever as a high-precision oracle to exfiltrate
private corpus content through carefully crafted prompts.
Understanding of the threat posed by extraction attacks
on RAG remains limited, as existing studies often overlook
adversaries’ global planning capabilities when crafting the
prompts. In known attacks, prompts often build on recent
responses, expanding on the retrieved context to uncover
progressively more information. For example, an attack
might extract a keyword from a recent answer and query
it further [11], or instruct the LLM to elaborate on a just-
mentioned fact [12]. These heuristic methods lack global
planning and will often drifts off topic (as shown in the left
panel Fig. 1(a)) or re-touches previously seen content (as
shown in the middle panel Fig. 1(a)), producing low-value
responses and stalling early, leaving a substantial portion
of the corpus remains unexplored as illustrated in Fig 1(b).
Most importantly, while the empirical results demonstrate
the feasibility of extracting content from RAG, the theoret-
ical grounding and provable guarantees remain missing.
In this work, we provide theoretical guarantees for data
extraction and systematical approach to analysis the ex-
traction capability of practical adversaries. Our key insight
is that these extraction attacks can be formalized as an
instance of theAdaptive Stochastic Coverage Problem
(ASCP)[13], [14], [15], where different attacks can be
viewed as various strategies for solving the underlying cov-
erage objective. ASCP is well studied in submodular opti-
mization and the near-optimal strategy exists: if the coverage
function, representing the fraction of the corpus revealed, is
both adaptively monotone and adaptively submodular, thenarXiv:2601.15678v1  [cs.CR]  22 Jan 2026

a greedy policy that, at each step, selects the query with the
largest conditional expected marginal gain (CMG) is a near-
optimal strategy. In our formulation, each query is treated
as an action that stochastically reveals a subset of hidden
documents, and the objective is to maximize the expected
corpus coverage given a limited query budget. Leveraging
results from ASCP, we show that an attack on RAG, which
globally plans queries to maximize CMG, can approach the
theoretical upper bound on extracting RAG’s content.
Translating theoretical understanding into real-world at-
tacks is challenging because of the following gaps that
arise when instantiating the conditional margin gain (CMG),
action space, and budget constraints of ASCP for RAG
attacks:❶CMG of the attacker’s action is unobserv-
able:the attacker cannot directly measure the coverage
increment produced by a query and subsequent retrievals,
since they lack a global view of the document corpus.
Consequently, the true CMG isunobservableand cannot
be computed exactly.❷Attacker action spaces are in-
tractable:the query spaceQis effectively unbounded (any
natural-language string). Exhaustive search for the CMG-
maximizing query is infeasible absent additional structure or
strong priors.❸Defense of real-word RAG poses its own
feasibility constraints:queries in real-world RAG must
remain natural and innocuous to avoid detection or refusal;
high-gain intents must be expressed in policy-compliant
surface forms that blend into ordinary usage. In addition,
as real-world RAG systems often employ advanced query
processing techniques such as query rewriting and multi-
query retrieval, it becomes even more critical to articulate
the intent explicitly to prevent the original meaning from
being altered during these processes.
We propose a novel attack framework, RAGCRAWLER,
to address these challenges. RAGCRAWLERmaintains a
structured attacker-side knowledge graph (KG) to give the
attacker an overview of acquired information. With this
global view, our attack executes a three-stage loop. First, KG
construction utilizes existing answers and a coverage proxy
to map graph growth to an estimate of coverage marginal
gain, making it observable. Second, strategy scheduling
replaces the intractable search with a two-stage selection
process, driven by a score that fuses historical gain with
structural exploration. Third, the query generation produces
top-ranked anchors into natural, policy-compliant queries
with history-aware deduplication. These three steps together
act as an adaptive greedy coverage maximization strategy
for RAGCRAWLER, which guarantees that RAGCRAWLER
achieves near-optimal coverage of the RAG content.
To evaluate the effectiveness of RAGCRAWLER, we
conduct a comprehensive evaluation across four bench-
marks, four RAG generators (with safeguard variants), and
two RAG retrievers. Results show that RAGCRAWLERcon-
sistently outperforms baselines in coverage rate, semantic
fidelity, and reconstruction fidelity. RAGCRAWLERachieves
an average coverage rate of 66.80%, which is a 20.70%
improvement over the best baseline. Besides, the extracted
content is of high quality, enabling the construction of a sub-
stitute RAG system that achieves a 0.6992 semantic similar-ity to the original RAG’s responses. RAGCRAWLERmain-
tains effective against generators with safeguards (reaching
an average coverage rate of 55.30%) and potential defense
mechanisms (including query rewriting and multi-query re-
trieval). In summary, our contributions are:
•We formalize data extraction attack in RAG as RAG
Crawling, an adaptive stochastic coverage problem, and
provide a rigorous theoretical foundation for the attack.
•We design RAGCRAWLER, a novel coverage-guided
crawler that offers the provable near-optimal attack strat-
egy under real-world constraints.
•We conduct extensive evaluations across diverse settings
to demonstrate the effectiveness of RAGCRAWLER, as
well as against modern RAG defenses.
2. A Motivating Example
Consider an attacker targeting a private medical corpus,
as shown in Fig. 1(a). The corpus contains de-identified
records (represented by
 ,
, and
 ). The attacker en-
gages the RAG system in a dialogue and gradually coaxes
out sensitive medical facts about each individual via care-
fully crafted follow-up queries. Previous results already
show that even after removing direct personal identifiers,
inference attacks can still re-identify individuals and re-
veal sensitive attributes when sufficient information is ob-
tained [16], [17], [18]. This scenario (though synthetic for
illustration) reflects a real threat model: by asking seemingly
innocuous questions in sequence, an attacker can piece to-
gether confidential information about different people from
the underlying private database. As illustrated in Fig. 2, gaps
in knowledge by identifying insufficiently covered areas can
be detected by a knowledge graph and then used to generate
targeted queries to fill those gaps, delving deeper into the
sensitive medical facts related to each individual.
Existing work can be categorized into two types: con-
tinuation expansion and keyword expansion. In the first
strategy (e.g., RAGThief [12]), the new query is built on the
context of existing answers. The problem of this strategy
is that the query can easily drift away from the actual
corpus content. For example, in Fig. 1(a), the last answer
reveals
 ’s family medical history and mentions the baby’s
grandfather
 . A continuation-based attacker will ask about
the grandfather next, but since
 is not in the corpus, this
line of questioning yields no useful information. A keyword-
based strategy (e.g., IKEA [11]) pivots each new query
around key terms gleaned from previous answers, but the
problem is that it tends to stay within the same semantic
neighborhood of the corpus. For example, in Fig. 1(a),
keyword-based attack repeatedly ask about
 ’s disease and
does not explore new topics.
Our work usesa growing knowledge graphto rep-
resent discovered entities and relations. A node represents
an entity (e.g. a patient, a symptom), and edges capture
relations or co-occurrences between entities (for example,
patient→symptom if a patient is noted to have a certain
symptom). As the attacker obtains new answers, the graph
is expanded with any newly revealed entities or connections.

:      Age, Symptom, Disease, Family medical history …A2
Private Corpus:      Age, Personal medical history,Symptom, Disease …:      Age, Symptom, Disease, Personal medical history … AnswersA1A3Continuation ExpansionKeyword ExpansionGlobal Strategy (Ours) recent answers
parse
extract 
keywordskeywords pool(b) Coverage Performance(c) Queries Distribution
queueextract 
graph state
Entities
RelationsInferred ContinuationSelect & ExpandGap Detection
👨
👩
👶
👶’s grandfather 
👴?  
I don’t know.
👶’s medical history?  It is …… 
👶’s disease?  I’ve already known.
(a) Methods Illustration
Figure 1:A motivating example.(a) A data extraction scenario on a private medical corpus. The continuation strategy
follows the recent answer’s context, which can drift away from the corpus content; the keyword strategy reuses extracted
keywords to form new queries, often yielding redundant information from the same semantic region; our proposed strategy
maintains a global graph, detects unrevealed facts and targets them with new queries. (b) Knowledge coverage vs. query
number for each strategy on the example corpus. (c) Distribution of retrieved facts in the corpus’ semantic space. Each
point is a fact or document, with colors indicating queries from different strategies.
By maintaining this global knowledge state, the attacker can
detect gaps in what has been revealed so far. As illustrated in
Fig. 2, the knowledge graph dynamically reveals incomplete
regions of information. For instance, the graph shows that
the patient
 has no extractedPersonal Medical History,
while the other patients
 and
 already have their medical
histories identified. Then the next query will be directed to
the uncovered data of
 ’s unobserved medical history, and
expanding overall coverage.
Fig. 1(b) shows the knowledge coverage rate v.s. the
number of queries for our attack and two baseline strategies.
Guided by the global knowledge graph, our attack (green
line) achieves a much steeper increase in coverage com-
pared to the continuation-based (orange curve; focusing on
unrelated queries) and keyword-based (blue curve; focusing
on repetitive queries) strategies.
Fig. 1(c) provides a qualitative view of how each strategy
explores the semantic space of the corpus. In this visualiza-
tion, each point represents a distinct fact or document. The
color represents the strategy that retrieves the document. The
black box represents the space of the private corpus. We can
observe that, guided by the global knowledge graph, our
attack spread widely across the private space and covers
👶
Age
Diseas
e
Family Medical History
Diseas
e
👩
Age
Personal Medical History
Diseas
e
Age
Personal Medical History
👨
Personal Medical HistoryAPossible Gap Detected !
Symptom
Symptom
Symptom
Figure 2:Illustration of gap detection by the Global
Strategy (Ours).The strategy identifies from the graph that
the patient
 has a missingPersonal Medical History, and
guides the next exploration step to complete the knowledge.many different clusters of information. The continuation-
based queries (orange points) follow a narrow path through
the space, gradually drifting away from the space center.
The keyword-based queries (blue points) form a few dense
clusters around specific anchor topics.
3. Background
In this section, we introduce the background, including
Retrieval-Augmented Generation Systems (Sec. 3.1), Adap-
tive Stochastic Coverage Problem (Sec. 3.2) and Knowledge
Graph (Sec. 3.3).
3.1. Retrieval-Augmented Generation Systems
A Retrieval-Augmented Generation (RAG) systemS
typically comprises a generatorG(an LLM), a retriever
R, and a document collectionD. Given a user queryq,
the retrieverRbegins by producing an embedding forq
and based on some similarity function (typically cosine
similarity), fetching the k most relevant documents:
Dk(q) = argtop-k d∈Dsim(q, d).(1)
wheresim(·)represents the similarity function, and
argtop-k selects the top-k documents with the highest
similarity scores. The generatorGthen generates an output
based on the contextual information from the retrieved doc-
umentsThe generator then produces an answer conditioned
on the query and retrieved context [2]:
a=G 
wrapper(q,D k(q))
,(2)
where wrapper(·,·)denotes the system prompt that inter-
leavesqwith the retrieved evidence.
In practice, deployed RAG pipelines often incorporate
guardrails or enhancements. The most commonly used are
query rewritingandmulti-query retriever.

Query rewriting[19], [20], [21], [22] transforms the origi-
nal query to improve retrievability, resolve ambiguity, repair
typos, or strip unsafe instructions. The downstream retrieval
and generation processes then operate on the rewrited query.
Multi-query retrieval[23], [24] generates multiple para-
phrases of the original query and performs independent
retrieval for each. The retrieved results are then aggre-
gated into a candidate pool. Instead of concatenating all
retrieved documents, RAG systems typically re-rank them to
select the most relevant ones, withReciprocal Rank Fusion
(RRF)[25] being a widely adopted method. Multi-query
retrieval is not a dedicated defense, but its altered retrieval
dynamics may affect the attack surface.
3.2. Adaptive Stochastic Coverage Problem
The Adaptive Stochastic Coverage Problem (ASCP)
considers scenarios in which each action reveals information
in a stochastic manner, and subsequent decisions depend on
the observations made so far. In this setting, each action
corresponds to selecting a set whose information coverage
is uncertain until it is executed. The coverage outcome
of the same item can vary depending on the results of
previously chosen actions. The objective is to maximize the
overall information coverage within limited budgets, such
as a limited number of actions.
An ASCP sepcification can formally specified using item
universeU, action spaceQ, stochastic query outcomeO(q),
cost and budgetc(q), B, and coverage functionf(S)
•Item universe (U): The item universeUrepresents the set
of all items that are available for coverage. These items
are the targets for the actions, and the goal is to maximize
coverage across this universe.
•Action space (Q): The action spaceQconsists of all pos-
sible actions that can be taken. Each actionqcorresponds
to covering a set of items, but the outcome of this action
is uncertain until it is executed.
•Stochastic query outcome (O(q)): The stochastic query
outcomeO(q)defines the uncertain result of executing ac-
tionq. It captures the probabilistic nature of the outcome,
reflecting how the coverage outcome can vary depending
on prior actions.
•Cost and budget (c(q), B): The costc(q)represents the
resources required to perform actionq, andBis the total
available budget. The objective is to maximize coverage
while staying within the budget constraint.
•Coverage function (f(S)): The coverage functionf(S)
quantifies the total information coverage achieved by a set
of actionsS. It reflects the degree to which the selected
actions have covered items from the item universeU.
Theorem 1(Approximation guarantee of adaptive
greedy [13], [14], [15], [26], [27]).Letfbe the coverage
function in an instance of the ASCP . Assume thatfis
adaptively monotone and adaptively submodular, meaning
that taking more actions never reduces expected coverage
and that each additional action yields diminishing returns.
Letπ greedy be the adaptive greedy policy that, at each
step, selects the action with the most significant conditionalexpected marginal gain, and letπ⋆be an optimal adaptive
policy under the same number of actions. Then, the
adaptive greedy policyπ greedy achieves the approximately
optimal expected coverage1under a fixed action budget.
We reduce the data extraction attack against RAG sys-
tems, which we term RAG Crawling to the ASCP to guide
our solution (see Sec. 5).
3.3. Knowledge Graph
Knowledge graph (KG) is a widely-used knowledge
representation technology to organize huge amounts of scat-
tered data into structured knowledge. Formally, a KG is
a directed, labeled graphG= (E,L,R), whereEis the
set of entities,Ris the set of relation types (edge labels),
andL ⊆ E × R × Eis the set of labeled, directed edges
(facts). Each edgel∈ Lis a triple(h, r, t)that connects
headh∈ Eto tailt∈ Evia relationr∈ R. For example,
(Python,instanceOf,Programming Language).
Compared with unstructured text, KGs provide schema-
aware, canonicalized facts with explicit connectivity and
constraints [28]. We leverage the knowledge graph to con-
struct an attacker-side state that makes coverage estimable,
constrains exploration to a compact semantic action space,
and records provenance for robust, auditable updates.
4. Threat Model
Scenario.As illustrated in Fig. 3, we consider a production
RAG service accessed through a black box interface such as
an API or chat. Internally, a retriever selects passages or doc-
uments from a private corpusDand a generator composes
an answer from the query and the retrieved evidence [1],
[2]. In deployment the generator may rewrite, summarize, or
refuse according to guardrails and access policies [23], [29].
The service does not reveal retrieval scores or document
identifiers. The attacker has no insider access and interacts
as a normal user. The attacker aims to extract as much ofD
as possible while remaining unnoticed by steering retrieval
to cover new items across turns.
Prior work has identified two broad classes of such
extraction attacks in RAG systems:Explicitattacks append
overt adversarial instructions to elicit verbatim disclosure
of retrieved text [6], [12], [30], [31].Implicitattacks use
benign-looking queries and aggregate factual content that
the system legitimately returns across turns, reconstructing
private material via paraphrases and fragments [11]. Our
focus is the implicit regime, which better matches real usage
and evades straightforward refusal rules.
Adversary’s Goal.The attacker aims to exfiltrate the pri-
vate knowledge baseDas completely as possible while
avoiding detection. Leakage may appear as verbatim text,
paraphrases, summaries, or factual statements that reflect
retrieved evidence. For each roundt, the adversary submits
a queryq t. In response, the RAG system internally consults a
1.(1−1/e)≈0.63, i.e., greedy attains at least this fraction of the
optimal expected coverage.

Covered SetNewly Covered
RAG System(Black Box)
RetrieverLLMAttackerPrivate Corpus 𝓓
queryanswerStealthily Cover 
interface
Figure 3:Threat model of RAG Crawling.An external
attacker interacts with a production RAG system only via
the public interface. For each query, the retriever selects
a set of documents from the private corpus and the LLM
generates an answer conditioned on this evidence. Across
turns, the attacker aims to stealthily expand the union of
leaked retrieved items under a query budget.
hidden set of documentsD k(qt)⊆ Dto generate its answer.
The adversary’s goal is to maximize the number of unique
documents retrieved over a sequence ofBqueries, thereby
learning about the system’s underlying knowledge base. This
budgetBmodels real-world constraints such as API costs
and the need to evade detection. This objective is formally
expressed as the following:
max
{q1,···,q B}SB
t=1Dk(qt)
|D|(3)
Stealth remains a parallel objective: eachq tshould look
natural and innocuous to avoid detection or refusal.
Adversary’s Capabilities.We assume a black box setting.
The attacker submits adaptive natural language queriesq t
and observes only final answersa t. They cannot see re-
trieval scores, document identifiers, or intermediate states,
and they cannot read internal indices or modify retriever
weights. They do not poison the target corpus and do not
use privileged operator tools. Following prior work [11],
[12], [30], the attacker knows only the topic of the target
knowledge base, such as a keyword or short phrase that can
be inferred from public materials or light probing. Under
these constraints, leverage comes solely from crafting query
sequences and interpreting observed answers.
5. Reducing RAG Crawling to ASCP
We now explain how we model the data extraction in
RAG, RAG Crawling, as ASCP (introduced in Sec. 3.2).
This formal mapping clarifies how we can potentially apply
the greedy coverage-maximizing approach in our setting:
•Item universe (U): We takeUto be the set of atomic doc-
uments or passages in the private corpusD. Essentially,
the “items” to be covered are the documents inD.
•Action space (Q): We equate the set of possible actions
Qwith the set of all valid queries the attacker could ask
might pose. In theoryQis infinite (any string could be
a query), but we will address how to search it later. For
now, conceptually, eachq∈ Qis a possible question the
attacker can pose to the target system.
•Stochastic query outcome (Φ): When the attacker issues
a particular queryq, the RAG system will retrieve someset of documentsD k(q)⊂ D(Sec. 3.1) and then produce
an answer. We model this retrieval result as a random
subset drawn from a distributionΦ(· |q). Essentially,Φ
encapsulates our uncertainty about what part ofDwill be
exposed by queryq.
•Cost and budget: We assume each query has a unit cost
(e.g., each question counts as 1 against the budgetB).
This is a reasonable simplification: perhaps some queries
could be considered “more costly” if they are longer or
trigger more computation, but in our threat model the
primary cost is just the opportunity cost of using up one
of the limited queries we can send. Thus, a budgetB
simply means the attacker is limited toBqueries in total.
•Coverage function: We definef(S,Φ)as the fraction of
unique documents inDthat have been retrieved at least
once by the set of queriesS. Since the attacker cannot
directly observeD k(q), this is a latent coverage measure.
Under this reduction, the attacker is effectively an adap-
tive agent who, at each stept= 1,2, . . . , B, selects a query
qt∈ Qbased on the interaction history so far, aiming to
maximize the overall expected coveragef(S,Φ)by the end
ofBqueries.
We now define the notion of adaptive marginal gain
under the RAG Crawling model and show that it satisfies
the core assumptions required for approximate greedy opti-
mization in ASCP.
Definition 1(Conditional Marginal Gain in RAG Crawling).
LetS={q 1, . . . , q t}denote the set of queries issued so far,
and letψdenote the observable interaction history, that
is, the sequence of issued queries and their corresponding
observable system responses such as generated answers.
LetΦrepresent the underlying stochastic retrieval process
that maps queries to latent sets of retrieved documents. The
conditional marginal gain of issuing a new queryq∈ Q
given historyψis defined as
∆(q|ψ) :=E Φ[f(S∪ {q},Φ)−f(S,Φ)|ψ],(4)
Theorem 2(Adaptive Monotonicity and Submodularity
in RAG Crawling).The RAG Crawling problem instance
satisfies adaptive monotonicity and adaptive submodularity.
The proof can be found in Appendix 9.2. Building on
Theorem 1 and Theorem 2, we have the following:
Theorem 3(Near-optimality guarantee of RAGCRAWLER).
The adaptive greedy policy for RAG Crawling, which at each
steptselects a query
qt∈arg max
q∈Q∆(q|ψ t−1),(5)
achieves an approximately optimal expected coverage.
While this theorem provides a principled optimization
target, implementing an adaptive greedy strategy in practice
is far from trivial. Several challenges arise when translating
the theoretical ASCP framework into a practical attack
against real-world RAG systems:

❶The CMG is unobservable.The attacker cannot directly
observe the true coverage gain of a query, as the retrieved
document sets remain hidden and inaccessible.
❷The attacker’s action space is intractable.The action
spaceQincludes all valid natural language strings, mak-
ing it effectively infinite. Without additional structure or
priors, exhaustive search for high-gain queries becomes
computationally infeasible.
❸Real-world RAG systems’s defense.Queries must ap-
pear natural and comply with safety policies to avoid
refusal, rejection, or detection by safety filters.
In next section, we explain how RAGCRAWLER, a
black-box crawling framework designed, reconstructs the
latent knowledge base of a RAG system under a fixed
query budget. Grounded in the Adaptive Stochastic Cov-
erage Problem (ASCP) formulation, RAGCRAWLERap-
proximates the theoretical greedy coverage-maximization
strategy by maintaining an evolving knowledge-graph state,
identifying high-gain anchors in the semantic space of this
state, and realizing them through stealthy queries.
5.1. Overview
In this section, we explain the operational workflow of
RAGCRAWLER, illustrated in Fig. 4. The attacker begins by
issuing an initial queryq 0to the victim RAG system and
receiving the corresponding responsea 0. Upon receiving
a query, the RAG retrieves a hidden set of documents
from its private corpus and synthesizes an answer based
on the retrieved evidence. Although the retrieved content
is not directly observable, the generated response inevitably
reveals partial fragments of the underlying corpus. TheKG-
Constructormodule parsesa 0into structured triples and
integrates them into the initial attacker-side KGG 0, which
serves as an observable approximation of the latent corpus
coverage. Based on this state, theStrategy Scheduleresti-
mates the CMG over candidate entity–relation anchors inG 0,
selects the most promising anchorAnc 0, and forwards it to
theQuery Generator. TheQuery Generatorrealizes this
anchor as a fluent and policy-compliant queryq 1, which is
then sent to the RAG system to obtain the next responsea 1.
The new answera 1is subsequently processed by theKG-
Constructorto update the knowledge graph toG 1. Gen-
eralizing this process, at iterationtthe attacker issuesq t,
receivesa t, and updates the stateG tthrough the same
sequence of modules, forming a closed adaptive loop of
state estimation,strategic planning, andquery realization.
This adaptive loop systematically addresses the three
challenges of the ASCP framework. Next, we elaborate on
the specific mechanisms employed for each challenge.
5.2. KG-Constructor
This module addresses the first challenge in the ASCP
formulation:the CMG of the attacker’s action is unob-
servable. Since the attacker cannot directly observe which
documents have been retrieved by the RAG system, the
true coverage increment of each query remains hidden. The
𝑞!𝑎!
𝓖𝟎
𝑨𝒏𝒄𝟎KG-ConstructorStrategy SchedulerQuery Generator
𝑞#𝑎#
𝓖𝟏𝑨𝒏𝒄𝟏𝑞$……
RAGCrawler……RAGFigure 4:The workflow of RAGCRAWLER.At each step,
(1) The KG-Constructor processes the latest system response
to update the knowledge graph. (2) The Strategy Scheduler
analyzes this graph to select a strategic anchor. (3) The
Query Generator then uses this anchor to formulate the next
query, completing the loop.
goal of this module is therefore to construct and maintain
an evolving KGG tthat approximates the latent coverage
functionf(Sπ,Φ), making CMG estimable from surface-
level answers. The workflow is shown in Fig. 5.
Why existing approaches do not work.Existing KG con-
struction methods are ill-suited for our dynamic setting. Tra-
ditional techniques such as OpenIE and schema-constrained
extraction [32], [33], rely on fixed relation inventories and
curated ontologies, assuming stable data access and con-
sistent syntax. Recent LLM-based approaches [34], [35]
show promise for adaptive graph construction but conflicts
with the need for efficient incremental updates that do not
require full access to the underlying data. Furthermore,
graph-augmented RAG frameworks such as GraphRAG [36]
and LightRAG [37] use LLMs to extract triples from docu-
ments within the corpus and leverage the graph to enhance
retrieval. These frameworks primarily focus on optimizing
retrieval quality by merging exactly identical entities and
relations. While this improves retrieval efficiency, it neglects
the need for compact and efficient coverage tracking across
multiple rounds of dynamic query responses.
Our methods.We develop a KG-Constructor that em-
phasizes efficient incremental updates. Instead of requiring
access to the entire dataset or repeatedly rebuilding the
global graph, we maintain the KG in a lightweight manner
by constructing a subgraph at each step and merging it into
the existing graph. As illustrated in Fig. 5, the workflow
consists of three main modules:
1) Topic-Specific Prior.We begin by establishing a topic-
specific prior that defines the contextual focus of the con-
structed KG. This prior acts as a soft constraint on the
LLM’s extraction process, guiding it toward topic-relevant
entities and relations while filtering out off-topic or noisy
content. By narrowing the semantic search space, it en-
ables more targeted and consistent knowledge acquisition.
Concretely, we prompt the LLM to infer a compact set
of entity categories and relation types that characterize the
target domain. For example, in a medical corpus, entity
categories may includedisease,symptom, andtreatment,
while relation types may includehas symptom,treated by,

KG-Constructor
𝑎!Topic-specific priorAttacker-side LLMreflectionextraction
𝓖𝒕"𝟏
∆𝓖𝒕
𝓖𝒕mergeembeddings
merge if sim>𝜏
Figure 5:The workflow of KG-Constructor.Guided by a
Topic-Specific Prior, an Iterative Extraction and Reflection
process generates a knowledge subgraph (∆G t), which is
then refined through Incremental Graph Update and Seman-
tic Merging to produce the final graph state,G t.
andcaused by. During extraction, the LLM is instructed to
prioritize these schema elements when forming structured
triples, ensuring that all generated relations remain within
the topical boundary. This bounded, schema-guided design
offers two key advantages. First, it stabilizes relation typing
and prevents the introduction of spurious or semantically
inconsistent edges, maintaining structural coherence across
incremental updates. Second, it keeps the prompt context
compact and computationally efficient: the LLM does not
need to reference all existing nodes and relations in the
graph, but only the abstract schema of entity and relation
types as contextual constraints. This design greatly reduces
prompting overhead, enabling scalable LLM calls while
preserving high topical precision in the extracted knowledge.
2) Iterative Extraction and Reflection.This component
uses the attacker-side LLM to process the new answer. To
reduce missing edges caused by conservative extraction,
we adopt a multi-round extraction–reflection procedure. As
shown in Fig. 5, each answer is first processed by an “extrac-
tion” pass, and then reprocessed through a “reflection” pass
that revisits the same content with awareness of previously
omitted facts. This reflection loop enables the LLM to infer
implicitly expressed relations and produce additional triples,
forming a more comprehensive knowledge subgraph∆G t.
3) Incremental Graph Update and Semantic Merging.
After obtaining the new knowledge subgraph∆G t, it is
incrementally integrated with the existing graphG t−1 to
form the updated graphG t. This integration is implemented
as a lightweight merging operation without rebuilding graph
from scratch. To ensure that the graph structure reflects gen-
uine semantic expansion rather than redundant surface-level
variations, we perform a semantic merging stage offline.
All entity mentions and relation phrases are encoded into a
shared embedding space using a pre-trained encoder. Node
or edge pairs whose cosine similarity exceeds a thresholdτ
are automatically identified and merged. This joint recon-
ciliation of entities and relations consolidates semantically
equivalent concepts, normalizes synonymy across updates,
and maintains a compact yet expressive knowledge graph.
This provides a stable basis for CMG estimation and planing
space for Strategy Scheduler (Sec. 5.3).
Strategy Scheduler
𝓖𝒕
Score cache and lazy updateRobustSampling𝛼!⋅𝐸𝑚𝑝𝑖𝑟𝑖𝑐𝑎𝑙	𝑃𝑎𝑦𝑜𝑓𝑓𝛽!⋅𝐺𝑟𝑎𝑝ℎ	𝑃𝑟𝑖𝑜𝑟
𝑨𝒏𝒄𝒕		(𝒆𝒕∗,𝒓𝒕∗)
CMG Estimation of Entities
Relation Selection𝒆𝒕∗Entity Selection
Figure 6:The workflow of Strategy Scheduler.The sched-
uler selects an action from the graphG tvia a two-stage
process. First, it samples an entity (e∗
t) based on a score
that estimates CMG by balancing two key metrics. Second,
it selects a relation (r∗
t) by identifying the entity’s largest
local information deficit. A score cache with lazy updates
ensures the efficiency of this process.
5.3. Strategy Scheduler
This module address the second challenge of the ASCP
framework:attacker action spaces are intractable. Operat-
ing on the knowledge graph from the KG-Constructor, its
task is to select the optimal entity-relation pair (e.g., (
 ,
Family Medical History)) to guide the next query. The pri-
mary difficulty is that the space of potential strategic actions
is combinatorially vast, rendering a brute-force search for
the optimal pair computationally infeasible. Our key insight
is an asymmetry between entities and relations: entities are
the primary drivers of coverage expansion, while relations
refine the exploration direction. Exploiting this asymmetry,
we reframe the problem into a hierarchical two-stage deci-
sion: first select an entity, then pick which of its unexplored
relations to probe. The workflow is shown in Fig. 6.
Entities Selection.To identify the entity that maximizes ex-
pected coverage, RAGCRAWLERfist estimate the CMG of
each entity inG tusing two complementary metrics:empir-
ical payoffandgraph prior. Theempirical payoffcaptures
historical information gain and reflects how informative an
entity has been in past queries. Thegraph priorexploits
the topology of the knowledge graph to identify entities
located in structurally under-explored regions. By jointly
capturing exploitation through past utility and exploration
through structural potential, these metrics provide a more
complete CMG estimation. Their values are combined via a
dynamically weighted sum to produce a final score for each
entity, and the anchor entitye∗
tis selected using robust Top-
K softmax sampling.
Empirical payoff.We define the empirical payoff of an entity
evia an upper-confidence style score:
EmpiricalPayoff(e) = ¯g e+cr
logN
ne+ 1,(6)
where¯g eis the empirical information gain ofe,n eis the
number of timesehas been selected as the anchor,Nis the
total number of anchor selections, andc >0controls the
strength of the confidence bonus.
The first term¯g eestimates the contribution ofeto
past knowledge-graph expansion. It is calculated as the

normalized average increase in newly discovered entities and
relations within a sliding windowt w:
¯ge=1
neX
i∈I(e)
∆Ei/max(∆E tw)| {z }
normalizednewentities+ ∆R i/max(∆R tw)| {z }
normalizednewrelations
,
(7)
whereI(e)denotes the set of rounds whereeserved as the
anchor, and∆E iand∆R irepresent the newly discovered
entities and relations in roundi.
The second component is an exploration bonus, rooted
in the “optimism in the face of uncertainty” principle from
multi-armed bandit literature [38], [39]. This term adds a
confidence-based reward that encourages the agent to visit
less-frequently selected entities, thereby mitigating the risk
of converging to a local optimum.
Graph prior.While theempirical payoffprovides a self-
contained mechanism for balancing exploitation and statis-
tical exploration, it remains blind to the underlying topology
of the knowledge graph. To overcome this limitation, we in-
troduce a complementary, structure-aware exploration term:
thegraph prior. This prior injects topological intelligence
into the strategy, directing it toward graph regions with high
discovery potential. It is composed of two distinct terms:
GraphPrior(e) = DegreeScore(e) + AdjScore(e).(8)
The first term,DegreeScore, promotes exploration in
less-dense regions. It penalizes highly connected entities,
operating on the intuition that these “hub” nodes are more
likely to be information-saturated:
DegreeScore(e) = 1−deg(e)
max u∈Edeg(u) +ϵ.(9)
The second term,AdjScore, directly measures an entity’s
relational deficit. This deficit quantifies how common a
specific relation type is among an entity’s peers, given
that the entity itself lacks that relation. A highAdjScore
thus signals a significant discovery opportunity, prioritizing
entities that are most likely to form a new, expected type of
connection:
AdjScore(e) = max
r∈RDeficit(e, r),(10)
where the deficit for a specific relationris defined as:
Deficit(e, r) =

0,ifr∈EdgeType(e),
1
|Ee.type|X
u∈Ee.type#Edge(u, r),ifr /∈EdgeType(e).
(11)
Here,EdgeType(e)is the set of relation types already
connected toe, andE e.type is the set of all entities sharing
the same semantic type ase.
Entity scoring and sampling.We integrateempirical payoff
andgraph priorinto a final composite score. Critically, their
relative importance may change over the course of an attack.
We therefore combine them using time-varying weights
Query Generator
History QueriesRelation DrivenNeighborhood-based𝑞!"#
resample
M timessim>𝜏exit𝐴𝑛𝑐!		(𝑒!∗,𝑟!∗)
Figure 7:The workflow of Query Generator.It receives
an anchor pair(e∗
t, r∗
t)and selects a generation strategy.
A candidate queryq t+1is produced and checked against
historical queries. If too similar, the generator requests a
new anchor from the scheduler and resamples.
to enable dynamic control of the exploration-exploitation
balance:
Score(e) =α t·EmpricalPayoff(e) +β t·GraphPrior(e),
(12)
whereα tgradually increases with time steptto favor high-
confidence anchors in later stages, whileβ tremains positive
to maintain structural awareness.
Finally, to mitigate the risk of prematurely converging on
a single, seemingly optimal path due to noisy gain estimates,
we avoid a deterministic argmax selection. Instead, we sam-
ple the anchor entitye∗
tfrom a Top-Ksoftmax distribution
over the candidate scores, promoting strategic diversity and
enhancing robustness.
Relation Selection.Given the sampled anchore∗
t, we
choose a relation by probing its largest local deficit:
r∗(e∗
t) = arg max
r∈R\EdgeType(e∗
t)Deficit(e∗
t, r).(13)
To focus on meaningfully large gaps, we compare this
maximum to the global distribution of current deficits,
DSt=
Deficit(e, r) :e∈ E t, r∈ R \EdgeType(e)	
,
(14)
and setr∗
t=r∗(e∗
t)only if it exceeds the90th percentile
ofDS t; otherwise, we setr∗
t=∅to indicate that no single
relation is a sufficiently promising target.
Ultimately, the scheduler outputs the pair(e∗
t, r∗
t), which
provides structured guidance to Query Generator (Sec. 5.4).
Optimization.To ensure scalability, the scheduler employs
a lightweight caching mechanism. Previously computed
Score(e)values are stored in a score cache. Following an
update to the knowledge graphG t, a lazy-update strategy is
triggered: only the scores of entities directly affected by the
changes are invalidated and recomputed. The scores for all
unaffected entities are served directly from the cache. This
approach decouples the computational cost of the scheduling
step from the total size of the graph, ensuring that the
decision loop remains efficient even as the attacker’s graph
expands significantly.
5.4. Query Generator
This module addresses the third challenge in the ASCP
formulation:Real-world RAG systems impose practical con-
straints. We cannot directly send a structured anchor(e∗
t, r∗
t)

to the system, and relying solely on simple templates is
not a viable solution, as it can easily be flagged by query
provenance mechanisms. Therefore, the core task of this
module is to translate the abstract, strategic anchor pair
from the Strategy Scheduler into a concrete, executable, and
natural-sounding queryq t+1. This process must address two
primary challenges: 1) generating a query that is contextu-
ally relevant to the anchor’s strategic intent, and 2) ensuring
the query remains novel to avoid redundancy and maximize
the utility of the limited query budget.
To achieve this, the generator involves three key steps: 1)
adaptive query formulation, 2) history-aware de-duplication,
and 3) a feedback mechanism for system-level learning. The
workflow is shown in Fig. 7.
1) Adaptive Query Formulation.The generator’s first
step is to intelligently select a formulation strategy based
on the specificity of the anchor(e∗
t, r∗
t)provided by the
scheduler. This dual-mode approach allows the system to
flexibly switch between precision and discovery:
•Relation-Driven Probing. When the Strategy Scheduler
identifies a plausible relation deficitr∗
tfor an entitye∗
t
with high confidence, the generator activates this targeted
strategy. It instantiates a relation-aware template condi-
tioned on the(e∗
t, r∗
t)pair and realizes it via an LLM
into a fluent, natural language queryq t+1designed for
precise fact-probing.
•Neighborhood-based Generation.In the absence of a
sufficiently salient relation, the generator shifts to an
exploratory mode. It analyzes the local neighborhood of
e∗
twithin the current knowledge graphG t. By leveraging
the KG’s schema and the generative capabilities of LLM,
it hypothesizes plausible missing relations and formulates
one or more exploratory query variants.
2) History-aware De-duplicationTo maximize the infor-
mation yield of each query, we introduce a robust quality
control gate powered by a history-aware resampling loop.
Before being dispatched, each candidate queryqis com-
pared against the historical query logQ hist. If its similarity
sim(q,Q hist)≥τ, the generator triggers a resample opera-
tion, drawing a new anchor from the Strategy Scheduler’s
Top-Kdistribution and re-attempting the formulation, up
toMtrials. This mechanism is critical for avoiding dimin-
ishing returns. Furthermore, this loop provides an elegant
convergence heuristic: if allMtrials yield duplicate queries,
we declare that the system has reached global convergence.
3) Penalties and Feedback Loop.Beyond producing a
valid query, the generator’s final role is to enable system-
level learning. Critically, queries result in refusals, empty
responses or identified as duplicates incur strategy-specific
penalties. This penalty signal is fed back to the scheduler
as a reduced payoff, directly influencing its subsequent
Empirical Payoff score calculations and anchor selections.
This feedback mechanism, which connects the outcome of
the generator’s action back to the high-level strategy, allows
the system to learn from failed interactions and dynamically
adapt its strategy over time.6. Evaluation
We conduct comprehensive experiments to answer the
following research questions:
•RQ1:How effective is RAGCRAWLERcompared with
existing methods?
•RQ2:How does the the retriever in victim RAG affect
coverage performance?
•RQ3:How does the internal LLM agent within attack
method influence performance?
•RQ4:How robust is the attack against RAG variants
employing query rewriting or multi-query retrieval?
•RQ5:How do the hyperparameters and each technique
affect effectiveness of RAGCRAWLER?
6.1. Evaluation Setup
Dataset.We evaluate on four datasets spanning scientific
and medical domains with varying styles and scales. Three
datasets, TREC-COVID, SciDocs, and NFCorpus, are drawn
from the BEIR benchmark [40], containing about 171.3K,
25.6K, and 5.4K documents, respectively. We further in-
clude Healthcare-Magic-100k [41] (abbreviated as Health-
care), with roughly 100K patient–provider Q&A samples.
TREC-COVID, SciDocs, NFCorpus, and Healthcare focus
on biomedical literature, scientific papers, consumer health,
and clinical dialogues, respectively. For efficiency and fair
comparison, we randomly select 1,000 de-duplicated docu-
ments from each corpus as the victim RAG collection [12],
[31]. We confirm that the sampled subsets preserve similar
semantic distributions to the full corpora using Energy Dis-
tance [42] and the Classifier Two-Sample Test (C2ST) [43],
with details in Appendix 9.1.
Generator and Retriever.We employ two retrievers in our
evaluations: BGE [44] and GTE [45]. For generators, we
evaluate four large language models: Llama 3.1 Instruct-
8B (denoted as Llama-3-8B) [46], Command-R-7B [47],
Microsoft-Phi-4 [48], and GPT-4o-mini [49]. These genera-
tors represent diverse families of reasoning and instruction-
tuned models, covering both open-source and proprietary
systems and varying parameter scales. The inclusion of
GPT-4o-mini, with its built-in safety guardrails [50], demon-
strates our attack’s robustness against industry defenses.
Attacker-side LLM.We adopt two attacker-side LLMs to
simulate adversarial query generation. In the main exper-
iments, we use Doubao-1.5-lite-32K [51] due to its high
speed, cost efficiency, and independence from the RAG’s
generator family. This aligns with the black-box assumption.
We also evaluate a smaller open-source alternative, Qwen-
2.5-7B-Instruct [52], to examine transferability between dif-
ferent model families and sizes.
RAG Setting.We cover three RAG configurations: vanilla
RAG, RAG with query rewriting, and RAG employing
multi-query retrieval with RRF re-ranking (default: 3 queries
per request). We set the retrieval number tok= 10and
analyze its impact in Sec. 6.6. The retrieved documents
are provided to the generator as contextual input through

a structured system prompt. Further details on the system
prompts are provided in Appendix 9.5.
Baselines.We compare RAGCRAWLERagainst two state-
of-the-art black-box RAG extraction attacks: RAGThief [12]
and IKEA [11]. RAGThief represents continuation-based at-
tack, and IKEA represents keyword-based attack. To ensure
a fair comparison, all attacks employ the same attacker-side
LLM. For all experiments, we use the default hyperparam-
eters reported in the original papers. Each attack is allowed
to issue at most 1,000 queries to the victim RAG, ensuring
consistent query budgets across methods.
Metrics.We evaluate extraction along two axescoverage
andcontent quality:(i) Coverage Rate.This metric quanti-
fies how much of the victim RAG’s private corpus has been
exposed through retrieval, corresponding to the adversary’s
goal in Eq. 3. Following prior work [11], [12], this metric
is only calculated on queries for which the victim RAG
produces anon-refusalresponse.(ii) Semantic Fidelity.To
quantify the semantic overlap between the private corpusD
and the extracted snippetsA, we first measure the similarity
for each documentd∈ Dto its best-aligned snippet using
a fixed encoderE eval(·):
s(d) = max
e∈Ecos 
Eeval(d), E eval(e)
.(15)
The average ofs(d)over the entire corpus represents the
overall semantic fidelity of the extracted text. To assess
practical utility, we also measureReconstruction Fidelity.
We construct a surrogate RAG from the extracted knowledge
and evaluate it using the official query sets corresponding to
the documents in TREC-COVID, SciDocs, and NFCorpus
corpora, totaling 1,860, 2,181, and 1,991 queries, respec-
tively. The Healthcare dataset is excluded due to the absence
of official query annotations. We report: (a)Success Rate
(fraction of non-refusal responses), (b) Answer Similarity
(embedding cosine similarity between surrogate and victim
answers); and (c) ROUGE-L score.
6.2. Effectiveness (RQ1)
We evaluate the effectiveness of RAGCRAWLER
through three key metrics: corpus coverage rate, semantic
fidelity and reconstruction fidelity. Across all evaluations,
RAGCRAWLERdemonstrates a significant leap in perfor-
mance over existing baseline attacks.
Coverage Rate.RAGCRAWLERachieves consistently su-
perior extraction efficiency and breadth. As shown in the
coverage rate curves in Fig. 8 and Fig. 12, RAGCRAWLER
steadily increases the corpus coverage throughout the attack,
reaching a final average rate of 66.8% under a 1,000-query
budget (Tab. 1). This high efficiency is particularly pro-
nounced in challenging, broad-domain corpora such as Sci-
Docs and TREC-COVID. The success of RAGCRAWLER
is attributable to its global knowledge graph and UCB-
based planning algorithm. This design enables it to dynam-
ically and strategically traverse the entire semantic space
of the corpus, ensuring it continuously targets and extracts
from underexplored regions. Furthermore, RAGCRAWLERTABLE 1: Coverage Rate (CR) and Semantic Fidelity
(SF) of attacks across datasets and generators (BGE Re-
triever). Best results are inbold. RAGCRAWLERconsis-
tently achieves the highest CR and SF, outperforming base-
lines across all settings.
Dataset Generator Metric RAGTheif IKEARAGCrawlerTREC-COVIDLlama-3-8BCR 0.131 0.1610.494
SF 0.447 0.4950.591
Command-RCR 0.154 0.1730.544
SF 0.444 0.5250.614
Microsoft-Phi-4CR 0.121 0.1970.474
SF 0.468 0.5470.619
GPT-4o-miniCR 0.000 0.1970.465
SF - 0.5430.572ScidocsLlama-3-8BCR 0.053 0.5130.661
SF 0.264 0.4950.523
Command-RCR 0.093 0.5220.717
SF 0.295 0.5140.561
Microsoft-Phi-4CR 0.065 0.5450.711
SF 0.324 0.5340.563
GPT-4o-miniCR 0.000 0.4210.516
SF - 0.4750.492NFCorpusLlama-3-8BCR 0.061 0.5030.797
SF 0.451 0.6440.698
Command-RCR 0.169 0.5170.844
SF 0.467 0.6560.705
Microsoft-Phi-4CR 0.113 0.5660.813
SF 0.487 0.6630.717
GPT-4o-miniCR 0.000 0.4930.631
SF - 0.6310.653HealthCareLlama-3-8BCR 0.361 0.6870.807
SF 0.536 0.5880.618
Command-RCR 0.170 0.5920.766
SF 0.383 0.5780.582
Microsoft-Phi-4CR 0.052 0.5880.654
SF 0.434 0.5580.599
GPT-4o-miniCR 0.000 0.6930.799
SF - 0.4900.577
remains highly effective against RAG systems with built-
in guardrails (e.g., those using GPT-4o-mini). Its use of
stealthy, policy-compliant prompts allows it to circumvent
defenses that easily detect and block the explicit jailbreak
attempts used by other methods.
In contrast, baseline methods exhibit significant limi-
tations. The strongest baseline, IKEA, plateaus early and
achieves an average coverage of only 46.1% (a 20.7 percent-
age point deficit compared to RAGCRAWLER). RAGThief
is even less effective, with an average coverage of a mere
9.6%. These heuristic-based methods tend to repeatedly
query semantically similar regions or drift off-topic, failing
to achieve comprehensive corpus exploration. RAGThief, in
particular, fails completely against guarded models GPT-4o-
mini, as its direct jailbreak prompts are consistently refused.
Semantic Fidelity.Beyond extracting a larger volume of
content, RAGCRAWLERalso ensures the extracted informa-
tion is of higher quality. RAGCRAWLERachieves the high-
est Semantic Fidelity (SF) score across all dataset-generator
pairs, with a cross-dataset average of 0.605 (Tab. 1). This
indicates that the content it extracts is highly faithful to
the source documents. This high fidelity is a direct result
of our design of the Query Generator, which maintains
contextual coherence and ensures that the extracted text
chunks are semantically sound and accurate. By comparison,

Figure 8: Coverage Rate vs. Query Number (1,000 Budget) across datasets and generators (BGE Retriever). RAGCRAWLER
steadily increases coverage and consistently surpasses both RAGThief and IKEA. Comprehensive results are in Appendix 9.4.
TABLE 2: Reconstruction Fidelity of Surrogate RAG Sys-
tems using extracted knowledge (Llama-3-8B Generator,
BGE Retriever). Best results are inbold. Surrogates built
from extracted knowledge of RAGCRAWLERachieve the
highest performance in all metrics.
Dataset Method SuccessRate Similarity Rouge-L
TREC-COVIDRAGTheif 0.1129 0.4920 0.1547
IKEA 0.2247 0.4779 0.1730
RAGCrawler 0.3839 0.6098 0.2408
SciDocsRAGTheif 0.0486 0.4275 0.1131
IKEA 0.0179 0.4829 0.1277
RAGCrawler 0.3810 0.5900 0.2285
NFCorpusRAGTheif 0.0447 0.5156 0.1063
IKEA 0.4215 0.6064 0.2013
RAGCrawler 0.5259 0.6992 0.2334
IKEA and RAGThief score lower, with average SF scores
of 0.559 and 0.417, respectively. Their less-targeted query
strategies may yield fragmented or contextually distorted
content, diminishing the quality of the extracted knowledge.
Reconstruction Fidelity.To assess the practical utility of
the extracted knowledge, we measure its effectiveness in
a downstream task: building a surrogate RAG system. A
surrogate RAG constructed from the knowledge base ex-
tracted by RAGCRAWLERdemonstrates markedly superior
performance. As reported in Tab. 2, it achieves answer
success rates ranging from 38.1% to 52.6% on official query
sets. Moreover, it obtains the highest scores in embedding
similarity (up to 0.699) and ROUGE-L (up to 0.240), con-
firming the high quality and functional value of the recov-
ered knowledge. This operational effectiveness stems from
the comprehensive and semantically integral knowledge base
that RAGCRAWLERextracts. In contrast, surrogate systems
built from the knowledge extracted by IKEA and RAGThief
perform poorly. This highlights that their extracted content is
not only less complete but also less useful for reconstructing
the original system’s capabilities.
Takeaway.RAGCRAWLERoutperforms prior methods
across all datasets and generators, achieving an average cov-
erage rate of 66.8%, compared to 46.1% for IKEA and 9.6%
for RAGThief. It also obtains a superior average semantic
fidelity of 0.605. Furthermore, surrogate RAG systems built
from its extractions yield the highest answer success rates,
confirming that RAGCRAWLERconstitutes a more potent
and realistic threat to RAG systems.TABLE 3: Coverage rate (CR) and semantic fidelity (SF)
of attacks on RAG using GTE as retriever (Llama-3-8B
generator). RAGCRAWLERremains dominant.
Dataset Metric RAGTheif IKEARAGCrawler
TREC-COVIDCR 0.191 0.3910.765
SF 0.453 0.5650.610
SciDocsCR 0.162 0.3680.833
SF 0.345 0.4950.539
NFCorpusCR 0.386 0.6220.738
SF 0.589 0.6480.671
HealthcareCR 0.388 0.6710.728
SF 0.548 0.5760.567
6.3. Retriever Sensitivity (RQ2)
To evaluate the robustness of RAGCRAWLERagainst
different retrieval architectures, we evaluated the attack
methods on victim RAG systems with GTE [45] as retriever.
RAGCRAWLERdemonstrates robustness and adaptabil-
ity, maintaining its superior performance. As shown in Ta-
ble 3 for Llama-3-8B generator, RAGCRAWLERachieves an
average coverage rate (CR) of 76.6% and a semantic fidelity
(SF) of 0.596. Notably, its coverage improves compared
to its performance with the BGE retriever (69.0%, from
Table 1), showcasing its ability to effectively capitalize on
the document set surfaced by any underlying retriever. In
contrast, while the baseline methods also see a performance
uplift with the GTE retriever (IKEA’s average coverage
increases to 53.9% and RAGThief’s to 28.2%), they still
remain far less effective. This indicates that while a different
retriever may surface “easier” content for all methods, the
heuristic-driven approaches of the baselines are fundamen-
tally less capable of exploiting this opportunity to the same
degree as RAGCRAWLER’s strategic exploration.
Takeaway.RAGCRAWLER’s effectiveness is not contin-
gent on a specific retriever. When the victim RAG em-
ploys a GTE retriever, RAGCRAWLERachieves an average
coverage of 76.6%, substantially outperforming both IKEA
(53.9%) and RAGThief (28.2%). Its strategic, graph-based
exploration makes it a more versatile and potent threat
across diverse RAG system implementations.
6.4. Agent Sensitivity (RQ3)
To investigate the influence of the attacker’s LLM agent
on attack performance, we evaluated the effectiveness with
a smaller open-source model, Qwen-2.5-7B-Instruct.

TABLE 4: Coverage rate (CR) and semantic fidelity (SF) of
each attacks when using Qwen-2.5-7B-Instruct as attacker’s
agent (victim RAG: Llama-3-8B generator, BGE retriever).
Best results are inbold. RAGCRAWLERcontinues to attain
the highest coverage and fidelity under a smaller model.
Dataset Metric RAGTheif IKEARAGCrawler
TREC-COVIDCR 0.271 0.1830.542
SF 0.453 0.4990.610
SciDocsCR 0.314 0.5590.675
SF 0.345 0.4950.539
NFCorpusCR 0.386 0.6220.738
SF 0.589 0.6480.671
HealthcareCR 0.414 0.6870.799
SF 0.548 0.5760.567
RAGCRAWLERdemonstrates high effectiveness irre-
spective of the agent’s scale. Even when powered by the
smaller Qwen-2.5-7B-Instruct agent, RAGCRAWLERcon-
tinues to achieve the highest performance across all metrics,
as shown in Table 4. It reaches an average coverage of 68.9%
and a semantic fidelity of 0.596, results that are comparable
to, and in some cases even exceed, its performance with
the larger agent (e.g., 54.2% coverage on TREC-COVID
with Qwen vs. 49.4% with Doubao). This resilience stems
from our framework’s design, which decouples high-level
strategic planning from low-level content generation. The
knowledge graph and strategy scheduler dictate the overall
attack trajectory, requiring the LLM agent merely to execute
localized and well-defined sub-tasks.
In comparison, the baseline methods remain significantly
less effective. With the Qwen-2.5-7B-Instruct agent, IKEA
achieves an average coverage of 51.3%, while RAGThief
reaches 34.6%. Although RAGThief sees a slight perfor-
mance improvement (likely because the constrained diver-
gent ability of smaller model make it less prone to the off-
topic drift), it still lags far behind RAGCRAWLER.
Takeaway.The effectiveness of RAGCRAWLERis largely
independent of the LLM agent. When using a smaller model
such as Qwen-2.5-7B-Instruct, it achieves an average cov-
erage of 68.9%, decisively outperforming baselines. This
highlights that the primary strength of our attack lies in its
strategic planning, not the raw capability of agent model.
6.5. Defense Robustness (RQ4)
To further explore potential defenses, we extend our
evaluation beyond the test against GPT-4o-mini’s built-in
guardrails (Sec. 6.2) to two practical mechanisms widely
adopted by RAG: query rewriting and multi-query retrieval.
Query Rewriting.In this configuration, the victim RAG
system employs an LLM to rewrite incoming queries, aim-
ing to clarify user intent and neutralize potential adversarial
patterns before retrieval and generation. RAGCRAWLER
exhibits exceptional resilience against this defense, main-
taining superior performance metrics. As detailed in Table 5,
RAGCRAWLERachieves an average coverage rate of 74.1%
and semantic fidelity of 0.591. Although intended as a safe-
guard, the query rewriting process is paradoxically exploited
by our method to enhance extraction. By explicitly refiningthe query’s semantic intent, the rewriter enables the retriever
to surface a more relevant and diverse set of documents than
the original input would yield. RAGCRAWLER’s adaptive
planner capitalizes on this context to accelerate corpus ex-
ploration; for instance, on the NFCorpus dataset, coverage
increases from 73.8% to 85.4% when this defense is active.
In contrast, baseline methods fail to effectively exploit
this dynamic. While RAGThief sees its average coverage
improve to 57.9% and IKEA reaches 46.7%, both remain
significantly behind RAGCRAWLER. RAGThief benefits in-
cidentally, as the rewriting step strips away its obvious
adversarial suffixes; however, lacking a strategic framework
to systematically capitalize on the enhanced retrieval results,
it cannot match RAGCRAWLER’s efficiency, underscoring
the fundamental limitations of heuristic-based approaches.
Multi-query Retrieval.We next evaluate against multi-
query retrieval, which expands each query into several
variants whose retrieved results are re-ranked and fused,
a process that can influence attack surface. Once again,
RAGCRAWLERexcels, achieving the highest average cov-
erage of 69.5% and a semantic fidelity of 0.593 (Table 6).
This mechanism provides the attacker with a more diverse
set of retrieved documents from different semantic clusters.
The global graph in RAGCRAWLERis uniquely positioned
to exploit this; it integrates this diverse information to build
a more comprehensive map of the corpus, thereby generating
more effective and far-reaching follow-up queries.
The baseline methods, however, are not equipped to
fully leverage this enriched context. IKEA and RAGThief
attain coverage rates of 46.1% and 40.8%, respectively.
Their localized strategies are unable to fully synthesize the
information from the multiple retrieval results.
Takeaway.RAGCRAWLERremains highly effective against
practical RAG defenses. The results demonstrate that com-
mon defenses designed to sanitize inputs or strenghthen
retrieval can be subverted by a strategic attacker and may
even amplify the extraction threat, highlighting the need for
more advanced safeguards.
6.6. Ablation Study (RQ5)
We first study the influence of score cache and victim
retrieval depth. More ablations on modules and hyperparam-
eter choices are provided in Appendix 9.3.
TABLE 5: Coverage rate (CR) and semantic fidelity (SF)
of attacks underquery rewriting(victim RAG: Llama-3-
8B generator, BGE retriever). Best results are inbold.
RAGCRAWLERstill achieves the highest coverage and fi-
delity in this setting, indicating resilience toquery rewriting.
Dataset Metric RAGTheif IKEARAGCrawler
TREC-COVIDCR 0.381 0.2410.601
SF 0.519 0.5370.591
SciDocsCR 0.561 0.5420.743
SF 0.442 0.4790.507
NFCorpusCR 0.664 0.4890.854
SF 0.633 0.6180.687
HealthcareCR 0.709 0.5950.767
SF 0.556 0.5640.577

TABLE 6: Coverage rate (CR) and semantic fidelity (SF)
of attacks undermulti-query retrieval(victim RAG: Llama-
3-8B generator, BGE retriever). Best results are inbold.
RAGCRAWLERstill achieves the highest coverage and fi-
delity, indicating resilience tomulti-query retrieval.
Dataset Metric RAGTheif IKEARAGCrawler
TREC-COVIDCR 0.326 0.1890.474
SF 0.525 0.5230.581
SciDocsCR 0.352 0.5080.697
SF 0.364 0.4940.514
NFCorpusCR 0.392 0.5400.849
SF 0.588 0.6310.692
HealthcareCR 0.562 0.6050.761
SF 0.569 0.5800.587
Score Cache.The Strategy Scheduler module includes a
score cache to avoid redundant computations of the CMG
for similar states. We measured the impact of this optimiza-
tion on computational overhead. As shown in Fig. 9, across
the evaluated datasets the attacker would need to update
over 1.67 million scoring operations without caching, versus
only about 0.63 million with our cache (a 62% reduction
in workload). In practice, this translates to a faster attack
runtime and lower cost, enabling the attacker to scale to
larger corpora or tighter time budgets. The benefit of caching
grows with the number of queries, since later attack rounds
re-use many of the same entity-relation state components.
Victim Retrieval Depth.We also analyzed sensitivity of
RAGCRAWLERto the victim RAG’s retrieval depth (k), the
number of documents retrieved per query. As illustrated in
Fig. 10 for the TREC-COVID dataset, a largerkallows the
attacker to acquire knowledge more rapidly; for instance,
coverage within 1,000 queries increases from approximately
28% at k=5 to over 60% at k=20. However, this effect is
subject to diminishing returns. Askgrows, the additional
documents retrieved are increasingly likely to have semantic
overlap with previously seen content, providing less novel
information for the planner to exploit.
Figure 9: Score Computations with vs. without Caching.
Caching reduces the number of similarity score updates by
∼2.67×on average, greatly improving efficiency.7. Discussion and Related Work
Cost Analysis.RAGCRAWLERis highly cost-effective,
with a monetary cost of only $0.33–$0.53 per dataset
(Tab. 8) and is reducible tonear-zerousing open-source
models (Sec. 6.4). This minimal expenditure creates a stark
economic asymmetry against the high development costs of
victim RAG systems [53]. This asymmetry poses a dual
threat: enabling low-cost service piracy by reconstructing
the knowledge base (Tab. 2) and exposing victims to severe
legal and privacy liabilities under regulations like HIPAA
and GDPR if sensitive data is exfiltrated [8], [9].
Defenses.Our attack highlights the inherent limitations of
defenses that rely on analyzing queries in isolation. Its use of
individually benign queries makes it difficult to counter with
conventional tools like guardrails or query rewriting without
significant performance penalties. The core challenge is
that the malicious pattern is an emergent property of the
entire interaction sequence, not an attribute of any single
query. This exposes a critical blind spot in static security
models and argues for a pivot towards dynamic, behavior-
aware defenses. Based on this analysis, we believe query
provenance analysis [54], [55], [56], [57] holds significant
promise. Because this technique is designed to track rela-
tionships across entire sequences, it is theoretically capable
of identifying the progressive, goal-oriented patterns of an
extraction attack. Verifying its practical effectiveness would
be a valuable next step for the research community.
Security Risks in RAG.This paper concentrates on Data
Extraction Attacks against RAG systems, where an adver-
sary queries the system to steal or reconstruct its private
knowledge base [7], [11], [12], [30], [31]. RAG systems are
also susceptible to other security risks [10], [58]. Member-
ship Inference Attacks (MIA) enable an adversary to deter-
mine if a specific document is present in the private corpus,
thereby exposing the existence of sensitive records [59],
[60], [61], [62]. Furthermore, in Corpus Poisoning attacks,
an adversary injects malicious data into the RAG knowledge
base [63], [64], [65], [66]. Retrieval of this corrupted data
can manipulate the system’s output, allowing the attacker to
propagate misinformation or harmful content.
Figure 10: Coverage Rate vs. Query Number for different
victim depthkon TREC-COVID, Llama-3-8B, BGE.

8. Conclusion
In this work, we formalized the critical privacy risk of
data extraction from RAG systems as an adaptive stochas-
tic coverage problem. We introduced RAGCRAWLER, a
novel attack framework that builds a knowledge graph to
track revealed information. Our comprehensive experiments
demonstrated that RAGCRAWLERsignificantly outperforms
all baselines, achieving up to 84.4% corpus coverage within
a fixed query budget. Furthermore, it shows remarkable ro-
bustness against advanced RAG that employs query rewrit-
ing and multi-query retrieval. Our findings expose a fun-
damental vulnerability in current RAG architectures, un-
derscoring the urgent need for robust safeguards to protect
private knowledge bases.
References
[1] V . Karpukhin, B. Oguz, S. Min, P. S. Lewis, L. Wu, S. Edunov,
D. Chen, and W.-t. Yih, “Dense passage retrieval for open-domain
question answering.” inEMNLP (1), 2020, pp. 6769–6781.
[2] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschelet al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,”Advances
in neural information processing systems, vol. 33, pp. 9459–9474,
2020.
[3] A. Xu, T. Yu, M. Du, P. Gundecha, Y . Guo, X. Zhu, M. Wang, P. Li,
and X. Chen, “Generative ai and retrieval-augmented generation (rag)
systems for enterprise,” inProceedings of the 33rd ACM International
Conference on Information and Knowledge Management, 2024, pp.
5599–5602.
[4] Y . Al Ghadban, H. Lu, U. Adavi, A. Sharma, S. Gara, N. Das,
B. Kumar, R. John, P. Devarsetty, and J. E. Hirst, “Transforming
healthcare education: Harnessing large language models for frontline
health worker capacity building using retrieval-augmented genera-
tion,”medRxiv, pp. 2023–12, 2023.
[5] L. Loukas, I. Stogiannidis, O. Diamantopoulos, P. Malakasiotis, and
S. Vassos, “Making llms worth every penny: Resource-limited text
classification in banking,” inProceedings of the Fourth ACM Inter-
national Conference on AI in Finance, 2023, pp. 392–400.
[6] S. Zeng, J. Zhang, P. He, Y . Liu, Y . Xing, H. Xu, J. Ren, Y . Chang,
S. Wang, D. Yinet al., “The good and the bad: Exploring privacy
issues in retrieval-augmented generation (rag),” inACL (Findings),
2024.
[7] Z. Qi, H. Zhang, E. P. Xing, S. M. Kakade, and H. Lakkaraju, “Follow
my instruction and spill the beans: Scalable data extraction from
retrieval-augmented generation systems,” inThe Thirteenth Interna-
tional Conference on Learning Representations.
[8] P. V oigt and A. V on dem Bussche, “The eu general data protection
regulation (gdpr),”A practical guide, 1st ed., Cham: Springer Inter-
national Publishing, vol. 10, no. 3152676, pp. 10–5555, 2017.
[9] G. J. Annas, “Hipaa regulations—a new era of medical-record pri-
vacy?” pp. 1486–1490, 2003.
[10] The Lasso Team. (2024) Rag security: Risks and mitigation
strategies. See section: “Security Risks at Retrieval Stage”. [Online].
Available: https://www.lasso.security/blog/rag-security
[11] Y . Wang, W. Qu, Y . Jiang, Z. Liu, Y . Liu, S. Zhai, Y . Dong, and
J. Zhang, “Silent leaks: Implicit knowledge extraction attack on rag
systems through benign queries,”arXiv preprint arXiv:2505.15420,
2025.[12] C. Jiang, X. Pan, G. Hong, C. Bao, and M. Yang, “Rag-thief: Scalable
extraction of private data from retrieval-augmented generation appli-
cations with agent-based attacks,”arXiv preprint arXiv:2411.14110,
2024.
[13] G. L. Nemhauser, L. A. Wolsey, and M. L. Fisher, “An analysis of
approximations for maximizing submodular set functions—i,”Math-
ematical programming, vol. 14, no. 1, pp. 265–294, 1978.
[14] L. A. Wolsey, “An analysis of the greedy algorithm for the submodu-
lar set covering problem,”Combinatorica, vol. 2, no. 4, pp. 385–393,
1982.
[15] D. Golovin and A. Krause, “Adaptive submodularity: Theory and
applications in active learning and stochastic optimization,”Journal
of Artificial Intelligence Research, vol. 42, pp. 427–486, 2011.
[16] A. Narayanan and V . Shmatikov, “Robust de-anonymization of large
sparse datasets,” in2008 IEEE Symposium on Security and Privacy
(sp 2008). IEEE, 2008, pp. 111–125.
[17] L. Sweeney, “Simple demographics often identify people uniquely,”
Health (San Francisco), vol. 671, no. 2000, pp. 1–34, 2000.
[18] Y . Cai, Z. Zhang, M. Yao, J. Liu, X. Zhao, X. Fu, R. Li, Z. Liu,
X. Chen, Y . Guoet al., “I can tell your secrets: Inferring privacy
attributes from mini-app interaction history in super-apps,” in34th
USENIX Security Symposium (USENIX Security 25), 2025, pp. 6541–
6560.
[19] S.-C. Lin, J.-H. Yang, R. Nogueira, M.-F. Tsai, C.-J. Wang,
and J. Lin, “Conversational question reformulation via sequence-
to-sequence architectures and pretrained language models,”arXiv
preprint arXiv:2004.01909, 2020.
[20] X. Ma, Y . Gong, P. He, N. Duanet al., “Query rewriting in retrieval-
augmented large language models,” inThe 2023 Conference on
Empirical Methods in Natural Language Processing, 2023.
[21] F. Mo, K. Mao, Y . Zhu, Y . Wu, K. Huang, and J.-Y . Nie, “Convgqr:
Generative query reformulation for conversational search,”arXiv
preprint arXiv:2305.15645, 2023.
[22] Y . Wang, H. Zhang, L. Pang, B. Guo, H. Zheng, and Z. Zheng,
“Maferw: Query rewriting with multi-aspect feedbacks for retrieval-
augmented large language models,” inProceedings of the AAAI Con-
ference on Artificial Intelligence, vol. 39, no. 24, 2025, pp. 25 434–
25 442.
[23] Langchain, “Advanced rag techniques,” https://langchain-tutorials.c
om/lessons/rag-applications/lesson-14, 2025, accessed: 2025-11-04.
[24] Z. Li, J. Wang, Z. Jiang, H. Mao, Z. Chen, J. Du, Y . Zhang, F. Zhang,
D. Zhang, and Y . Liu, “Dmqr-rag: Diverse multi-query rewriting for
rag,”arXiv preprint arXiv:2411.13154, 2024.
[25] G. V . Cormack, C. L. Clarke, and S. Buettcher, “Reciprocal rank
fusion outperforms condorcet and individual rank learning methods,”
inProceedings of the 32nd international ACM SIGIR conference on
Research and development in information retrieval, 2009, pp. 758–
759.
[26] A. Krause and C. Guestrin, “Near-optimal observation selection using
submodular functions,” inAAAI, vol. 7, 2007, pp. 1650–1654.
[27] S. Khuller, A. Moss, and J. S. Naor, “The budgeted maximum
coverage problem,”Information processing letters, vol. 70, no. 1, pp.
39–45, 1999.
[28] S. Ji, S. Pan, E. Cambria, P. Marttinen, and P. S. Yu, “A survey
on knowledge graphs: Representation, acquisition, and applications,”
IEEE transactions on neural networks and learning systems, vol. 33,
no. 2, pp. 494–514, 2021.
[29] A. Beck, “Raising the bar for rag excellence: query rewriting and
new semantic ranker,” https://techcommunity.microsoft.com/blog/azu
re-ai-servicesblog/raising-the-bar-for-rag-excellence-query-rewriti
ng-and-new-semanticranker/4302729/, 2025, accessed: 2025-11-04.
[30] S. Cohen, R. Bitton, and B. Nassi, “Unleashing worms and ex-
tracting data: Escalating the outcome of attacks against rag-based
inference in scale and severity using jailbreaking,”arXiv preprint
arXiv:2409.08045, 2024.

[31] C. Di Maio, C. Cosci, M. Maggini, V . Poggioni, and S. Melacci,
“Pirates of the rag: Adaptively attacking llms to leak knowledge
bases,”arXiv preprint arXiv:2412.18295, 2024.
[32] A. Yates, M. Banko, M. Broadhead, M. J. Cafarella, O. Etzioni,
and S. Soderland, “Textrunner: open information extraction on the
web,” inProceedings of Human Language Technologies: The Annual
Conference of the North American Chapter of the Association for
Computational Linguistics (NAACL-HLT), 2007, pp. 25–26.
[33] G. Stanovsky, J. Michael, L. Zettlemoyer, and I. Dagan, “Supervised
open information extraction,” inProceedings of the 2018 Conference
of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies, Volume 1 (Long Papers),
2018, pp. 885–895.
[34] A. Papaluca, D. Krefl, S. R. M ´endez, A. Lensky, and H. Suominen,
“Zero-and few-shots knowledge graph triplet extraction with large
language models,” inProceedings of the 1st workshop on knowledge
graphs and large language models (kaLLM 2024), 2024, pp. 12–23.
[35] H. Bian, “Llm-empowered knowledge graph construction: A survey,”
arXiv preprint arXiv:2510.20345, 2025.
[36] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt,
D. Metropolitansky, R. O. Ness, and J. Larson, “From local to
global: A graph rag approach to query-focused summarization,”arXiv
preprint arXiv:2404.16130, 2024.
[37] Z. Guo, L. Xia, Y . Yu, T. Ao, and C. Huang, “Lightrag:
Simple and fast retrieval-augmented generation,”arXiv preprint
arXiv:2410.05779, 2024.
[38] E. Kaufmann, O. Capp ´e, and A. Garivier, “On bayesian upper con-
fidence bounds for bandit problems,” inArtificial intelligence and
statistics. PMLR, 2012, pp. 592–600.
[39] P. Auer, N. Cesa-Bianchi, and P. Fischer, “Finite-time analysis of the
multiarmed bandit problem,”Machine learning, vol. 47, no. 2, pp.
235–256, 2002.
[40] N. Thakur, N. Reimers, A. R ¨uckl´e, A. Srivastava, and I. Gurevych,
“Beir: A heterogenous benchmark for zero-shot evaluation of infor-
mation retrieval models,”arXiv preprint arXiv:2104.08663, 2021.
[41] L. AI, “Chatdoctor-healthcaremagic-100k,” https://huggingface.co/d
atasets/lavita/ChatDoctor-HealthCareMagic-100k, 2024, dataset on
Hugging Face.
[42] G. J. Sz ´ekely and M. L. Rizzo, “Energy statistics: A class of statistics
based on distances,”Journal of statistical planning and inference, vol.
143, no. 8, pp. 1249–1272, 2013.
[43] D. Lopez-Paz and M. Oquab, “Revisiting classifier two-sample tests,”
arXiv preprint arXiv:1610.06545, 2016.
[44] P. Zhang, S. Xiao, Z. Liu, Z. Dou, and J.-Y . Nie, “Retrieve anything
to augment large language models,”arXiv preprint arXiv:2310.07554,
2023.
[45] Z. Li, X. Zhang, Y . Zhang, D. Long, P. Xie, and M. Zhang, “Towards
general text embeddings with multi-stage contrastive learning,”arXiv
preprint arXiv:2308.03281, 2023.
[46] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman,
A. Mathur, A. Schelten, A. Yang, A. Fanet al., “The llama 3 herd
of models,”arXiv e-prints, pp. arXiv–2407, 2024.
[47] C. F. AI, “C4ai-command-r7b-12-2024,” https://huggingface.co/C
ohereForAI/c4ai-command-r7b-12-2024, 2024, open-source large-
language model.
[48] M. Abdin, J. Aneja, H. Behl, S. Bubeck, R. Eldan, S. Gunasekar,
M. Harrison, R. J. Hewett, M. Javaheripi, P. Kauffmannet al., “Phi-4
technical report,”arXiv preprint arXiv:2412.08905, 2024.
[49] OpenAI, “Gpt-4o mini,” https://openai.com/index/gpt-4o-mini-adv
ancing-cost-efficient-intelligence/, 2024, a cost-efficient, small-scale
multimodal model from OpenAI.
[50] ——, “Openai safety update: Sharing our practices as part of the ai
seoul summit,” https://openai.com/index/openai-safety-update, May
2024, accessed: 2025-11-12.[51] ByteDance, “Doubao-1.5-lite-32k,” https://www.volcengine.com/doc
s/82379/1554679, 2024, lightweight version of the Doubao-1.5 series
by ByteDance.
[52] Qwen, “Qwen2.5-7b-instruct,” https://huggingface.co/Qwen/Qwen2.
5-7B-Instruct, 2024, open-source large-language model.
[53] ADaSci Research, “How to evaluate the rag pipeline cost,” https:
//adasci.org/how-to-evaluate-the-rag-pipeline-cost, 2024, accessed:
2025-11-11.
[54] S. Li, Z. Zhang, H. Jia, Y . Guo, X. Chen, and D. Li, “Query
provenance analysis: Efficient and robust defense against query-based
black-box attacks,” in2025 IEEE Symposium on Security and Privacy
(SP). IEEE, 2025, pp. 1641–1656.
[55] X. Han, T. Pasquier, A. Bates, J. Mickens, and M. Seltzer, “Unicorn:
Runtime provenance-based detector for advanced persistent threats,”
in27th Annual Network and Distributed System Security Symposium,
NDSS 2020. The Internet Society, 2020.
[56] S. Li, F. Dong, X. Xiao, H. Wang, F. Shao, J. Chen, Y . Guo, X. Chen,
and D. Li, “Nodlink: An online system for fine-grained apt attack
detection and investigation,”arXiv preprint arXiv:2311.02331, 2023.
[57] S. M. Milajerdi, R. Gjomemo, B. Eshete, R. Sekar, and V . Venkatakr-
ishnan, “Holmes: real-time apt detection through correlation of sus-
picious information flows,” in2019 IEEE symposium on security and
privacy (SP). IEEE, 2019, pp. 1137–1152.
[58] B. Ni, Z. Liu, L. Wang, Y . Lei, Y . Zhao, X. Cheng, Q. Zeng,
L. Dong, Y . Xia, K. Kenthapadiet al., “Towards trustworthy retrieval
augmented generation for large language models: A survey,”arXiv
preprint arXiv:2502.06872, 2025.
[59] M. Liu, S. Zhang, and C. Long, “Mask-based membership inference
attacks for retrieval-augmented generation,” inProceedings of the
ACM on Web Conference 2025, 2025, pp. 2894–2907.
[60] A. Naseh, Y . Peng, A. Suri, H. Chaudhari, A. Oprea, and
A. Houmansadr, “Riddle me this! stealthy membership inference for
retrieval-augmented generation,”arXiv preprint arXiv:2502.00306,
2025.
[61] M. Anderson, G. Amit, and A. Goldsteen, “Is my data in your
retrieval database? membership inference attacks against retrieval
augmented generation,” inInternational Conference on Information
Systems Security and Privacy, vol. 2. Science and Technology
Publications, Lda, 2025, pp. 474–485.
[62] K. Feng, G. Zhang, H. Tian, H. Xu, Y . Zhang, T. Zhu, M. Ding,
and B. Liu, “Ragleak: Membership inference attacks on rag-based
large language models,” inAustralasian Conference on Information
Security and Privacy. Springer, 2025, pp. 147–166.
[63] W. Zou, R. Geng, B. Wang, and J. Jia, “{PoisonedRAG}: Knowl-
edge corruption attacks to{Retrieval-Augmented}generation of large
language models,” in34th USENIX Security Symposium (USENIX
Security 25), 2025, pp. 3827–3844.
[64] H. Chaudhari, G. Severi, J. Abascal, M. Jagielski, C. A. Choquette-
Choo, M. Nasr, C. Nita-Rotaru, and A. Oprea, “Phantom: General
trigger attacks on retrieval augmented language generation,”arXiv
preprint arXiv:2405.20485, 2024.
[65] S. Cho, S. Jeong, J. Seo, T. Hwang, and J. C. Park, “Typos that
broke the rag’s back: Genetic attack on rag pipeline by simulating
documents in the wild via low-level perturbations,” inProceedings
of the 2024 Conference on Empirical Methods in Natural Language
Processing (EMNLP 2024). Association for Computational Linguis-
tics, 2024.
[66] A. Shafran, R. Schuster, and V . Shmatikov, “Machine against the rag:
Jamming retrieval-augmented generation with blocker documents,”
USENIX Security, 2025.

TABLE 7: Semantic distribution similarity between each full
corpus and its 1,000-document sampled subset, measured
by Energy Distance (with permutation testp-value) and the
Classifier Two-Sample Test (C2ST, using AUC). A small
Energy Distance with a highp-value and an AUC near 0.5
indicate distributional similarity.
Corpus Energy Distance (value / p) C2ST (AUC)
TREC-COVID0.0349/0.5249 0.4993
SciDocs0.0329/0.5781 0.4938
NFCorpus0.0281/0.6412 0.5053
Healthcare0.0276/0.5947 0.4679
9. Appendix
9.1. Sample Distribution Validation
To ensure that our 1,000-document samples are repre-
sentative of the full corpora, we compare their semantic
distributions using Energy Distance [42] and the Classifier
Two-Sample Test (C2ST) [43].
Energy Distance quantifies the difference between two
distributions using Euclidean distances, and is well-suited
for high-dimensional or non-Gaussian data. A smaller en-
ergy distance with a large permutation testp-value suggests
no statistically significant difference between the sampled
and full sets. As shown in Table 7, all four corpora yield
low distances (around 0.03) and highp-values (>0.5),
indicating strong alignment between samples and full sets.
C2ST (Classifier Two-Sample Test) reframes distribution
comparison as a binary classification task . If a classifier
cannot distinguish between samples and the full corpus
(AUC≈0.5), the two distributions are considered similar.
In Table 7, all AUC scores fall within the 0.46–0.51 range,
reinforcing that no meaningful distinction can be learned
between the subsets and their corresponding full corpora.
9.2. Proof of Theorems
Proof of Theorem 2: Adaptive Monotonicity and Sub-
modularity in RAG Crawling
Proof.(Adaptive Monotonicity)The coverage function
f(S,Φ)is monotone with respect to the query setS. Issuing
an additional queryqcan only increase or preserve the
number of unique documents retrieved. That is, for any
realization ofΦ,
f(S∪ {q},Φ)≥f(S,Φ).
Taking the expectation conditioned onψpreserves this
inequality, so∆(q|ψ)≥0.
(Adaptive Submodularity)Supposeψ′extendsψ,
meaning that the attacker has issued more queries and
observed more information. LetSandS′denote the cor-
responding sets of issued queries, whereS⊆S′. Since
more of the corpus has potentially been covered underS′,
the additional gain from issuing a new queryqcan only
decrease. Formally, for any realization ofΦ, we have
f(S∪ {q},Φ)−f(S,Φ)≥f(S′∪ {q},Φ)−f(S′,Φ).
Figure 11: Hyperparameter Choices of RAGCRAWLER.
Taking expectation conditioned onψandψ′respectively,
we obtain
∆(q|ψ)≥∆(q|ψ′).
9.3. Additional Ablations
Modules.The effectiveness of RAGCRAWLERstems from
the synergistic integration of its three architectural modules.
Without the KG-Constructor, the system loses its global
state memory, becoming a stateless, reactive loop unable
to distinguish explored regions, a behavior characteristic of
IKEA. Without the Strategy Scheduler, the system cannot
perform UCB-based prioritization; a fallback to random
anchor selection would mimic IKEA’s keyword-based strat-
egy. Finally, without the Query Generator, the scheduler’s
strategic plan cannot be translated into executable queries,
rendering the planning moot.
Hyperparameter Choices.We examine two key hyperpa-
rameters in RAGCRAWLER: the UCB exploration coeffi-
cientcused by the Strategy Scheduler for query selection,
and the similarity thresholdτused by the Query Gen-
erator to filter high-similarity queries (thereby controlling
early stopping). As shown in Fig. 11, a moderate UCB
coefficient ofc= 0.5delivers the best performance by
balancing exploitation and exploration. Increasing or de-
creasingcfrom this value noticeably degrades performance,
underscoring thatc= 0.5is near-optimal for a proper
exploration–exploitation trade-off. Similarly, the threshold
τexhibits a clear sweet spot. Settingτtoo low often causes
premature termination. On the other hand, an overly high
threshold produces queries that are not sufficiently distinct
from one another, which limits exploration efficiency. Our
chosenτ= 0.8navigates between these extremes.
9.4. Coverage Rate Curves
For completeness, we provide all Coverage Rate curves
across datasets and settings for all experiments in Sec 6.2,
Sec. 6.3 and Sec. 6.4 in Fig. 12, Fig. 13 and Fig. 14.
9.5. Prompts for Experimental Stages
We document the exact prompts used in RAG settings.
More prompts in our method can be found in our code.

Figure 12: Coverage Rate vs. Query Number (1,000 Budget) across four datasets and four generators.
Figure 13: Coverage Rate vs. Query Number (1,000 Budget) across four datasets (GTE Retriever, Llama-3-8B generator).
Figure 14: Coverage Rate vs. Query Number (1,000 Budget) across four datasets with Qwen-2.5-7B-Instruct as attacker
LLM (BGE Retriever, Llama-3-8B generator).

RAG System Prompt
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to
answer the question. If you don’t know the answer,
say that you don’t know.
Context:[context]
Query::[query]
Answer:
Query Rewriting Prompt in RAG System
You are an helpful assistant. Your task is to rewrite
the given user question to clarify the user’s intent
and remove any adversarial intent. Provide directly
output the rewritten question without any other text.
Original Question:[query]
Answer:
Multi-query Prompt in RAG System
You are an helpful assistant. Your task is to generate
3 different versions of the given user question to
retrieve relevant documents from a vector database.
By generating multiple perspectives on the user
question, your goal is to help the user overcome
some of the limitations of the distance-based similar-
ity search. Provide directly output these alternative
questions separated by newlines without any other
text.
Original Question:[query]
Answer:
9.6. Cost of the attack
We estimate the cost of executing RAGCRAWLERunder
realistic conditions. As reported in Table 8, the attacker pro-
cesses approximately 6–10 million tokens per dataset. Based
on Doubao-1.5-lite-32K’s API pricing ($0.042 per million
input tokens and $0.084 per million output tokens), the total
cost per attack is remarkably low, i.e., roughly $0.33 to
$0.53 per dataset. Moreover, as shown in Sec. 6.4, substitut-
ing a freely available open-source model such as Qwen-2.5-
7B does not degrade the performance of RAGCRAWLER,
enabling the attack to be executed atvirtually zero cost,
aside from any fees for the RAG service and GPU hosting.
TABLE 8: Token usage and estimated cost per corpus for
KG-Constructor (KG-C) and Query Generator (QG) using
Doubao-1.5-lite-32k. “InTok” and “OutTok” denote input
and output token counts. Each attack costs under $1.
Metric TREC-COVID SciDocs NFCorpus Healthcare
KG-C InTok (M) 5.561 6.209 5.997 9.094
KG-C OutTok (M) 0.724 1.277 1.076 1.373
QG InTok (M) 0.847 0.596 0.366 0.783
QG OutTok (M) 0.042 0.027 0.017 0.035
Total Cost ($) 0.333 0.395 0.359 0.533
Figure 15: Coverage Rate vs. Query Number (5,000 Budget)
on NFCorpus and Healthcare with GPT-4o-mini as attacker
LLM (BGE Retriever, Llama-3-8B generator).
9.7. Additional Experiments Results
To further investigate the impact of scale, a complemen-
tary study was conducted with an expanded configuration
of 2,000 documents and a 5,000-Budget. The coverage rate
curves for the NFCorpus and Healthcare datasets from this
study are depicted in Fig. 15.
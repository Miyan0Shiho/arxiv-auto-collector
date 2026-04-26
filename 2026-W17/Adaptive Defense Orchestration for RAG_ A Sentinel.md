# Adaptive Defense Orchestration for RAG: A Sentinel-Strategist Architecture against Multi-Vector Attacks

**Authors**: Pranav Pallerla, Wilson Naik Bhukya, Bharath Vemula, Charan Ramtej Kodi

**Published**: 2026-04-22 11:17:10

**PDF URL**: [https://arxiv.org/pdf/2604.20932v1](https://arxiv.org/pdf/2604.20932v1)

## Abstract
Retrieval-augmented generation (RAG) systems are increasingly deployed in sensitive domains such as healthcare and law, where they rely on private, domain-specific knowledge. This capability introduces significant security risks, including membership inference, data poisoning, and unintended content leakage. A straightforward mitigation is to enable all relevant defenses simultaneously, but doing so incurs a substantial utility cost. In our experiments, an always-on defense stack reduces contextual recall by more than 40%, indicating that retrieval degradation is the primary failure mode. To mitigate this trade-off in RAG systems, we propose the Sentinel-Strategist architecture, a context-aware framework for risk analysis and defense selection. A Sentinel detects anomalous retrieval behavior, after which a Strategist selectively deploys only the defenses warranted by the query context. Evaluated across three benchmark datasets and five orchestration models, ADO is shown to eliminate MBA-style membership inference leakage while substantially recovering retrieval utility relative to a fully static defense stack, approaching undefended baseline levels. Under data poisoning, the strongest ADO variants reduce attack success to near zero while restoring contextual recall to more than 75% of the undefended baseline, although robustness remains sensitive to model choice. Overall, these findings show that adaptive, query-aware defense can substantially reduce the security-utility trade-off in RAG systems.

## Full Text


<!-- PDF content starts -->

ADAPTIVEDEFENSEORCHESTRATION FORRAG: A
SENTINEL-STRATEGISTARCHITECTURE AGAINST
MULTI-VECTORATTACKS
Pranav Pallerla
School of Computer and Information Sciences
University of Hyderabad
Hyderabad, Telangana, India
pallerlapranavdec27@gmail.comWilson Naik Bhukya
School of Computer and Information Sciences
University of Hyderabad
Hyderabad, India
rathore@uohyd.ac.in
Bharath Vemula
Purdue University
West Lafayette, USA
vemula.sai8@gmail.comCharan Ramtej Kodi
School of Computer and Information Sciences
University of Hyderabad
Hyderabad, India
21mcpc10@uohyd.ac.in
April 24, 2026
ABSTRACT
Retrieval-augmented generation (RAG) systems are increasingly deployed in sensitive domains
such as healthcare and law, where they rely on private, domain-specific knowledge. This capability
introduces significant security risks, including membership inference, data poisoning, and unintended
content leakage. A straightforward mitigation is to enable all relevant defenses simultaneously, but
doing so incurs a substantial utility cost. In our experiments, an always-on defense stack reduces
contextual recall by more than 40%, indicating that retrieval degradation is the primary failure
mode. To mitigate this trade-off in RAG systems, we propose the Sentinel-Strategist architecture,
a context-aware framework for risk analysis and defense selection. A Sentinel detects anomalous
retrieval behavior, after which a Strategist selectively deploys only the defenses warranted by the
query context. Evaluated across three benchmark datasets and five orchestration models, ADO is
shown to eliminate MBA-style membership inference leakage while substantially recovering retrieval
utility relative to a fully static defense stack, approaching undefended baseline levels. Under data
poisoning, the strongest ADO variants reduce attack success to near zero while restoring contextual
recall to more than 75% of the undefended baseline, although robustness remains sensitive to model
choice. Overall, these findings show that adaptive, query-aware defense can substantially reduce the
security-utility trade-off in RAG systems.
KeywordsRetrieval-Augmented Generation (RAG), Large Language Models, AI Security, Dynamic Orchestration,
Security-Utility Trade-off, Data Poisoning, Membership Inference
1 Introduction
In the realm of Large Language Models (LLMs), significant advancements have been made across various tasks
such as automated code generation, medical diagnostics, and document summarization [ 1,2]. However, standalone
LLMs suffer from two well-documented limitations: they tend to generate plausible but factually incorrect content,
known as hallucinations, and they are limited by their training cutoff, unable to access information beyond it [ 3,4,5].
Retrieval-Augmented Generation (RAG) directly addresses these issues by retrieving relevant passages from an external
knowledge base before each generation step, thereby grounding responses in up-to-date, domain-specific evidence. AsarXiv:2604.20932v1  [cs.CR]  22 Apr 2026

APREPRINT- APRIL24, 2026
illustrated in Fig. 1, the standard RAG pipeline encodes a user query, retrieves the top- kmatching documents, and
conditions the generator on the retrieved context. Due to these capabilities, RAG is widely adopted in high-stakes
settings such as healthcare, legal analysis, and enterprise financial systems.
This integration of Large Language Models (LLMs) with a live, externally managed knowledge store in Retrieval-
Augmented Generation (RAG) systems significantly expands the attack surface compared to standalone LLMs. RAG
systems inherit vulnerabilities from standard LLMs while introducing new threats that target both the retrieval mecha-
nism and the underlying knowledge base [ 6]. These critical attack vectors pose significant concerns for deployed RAG
systems:
•Membership Inference Attacks (MIAs), in which an adversary determines whether a specific sensitive
document is present in the vector knowledge base by probing targeted queries [7, 8].
•Data Poisoning, where malicious actors inject crafted adversarial documents into the knowledge store to
make the retriever surface them in response to trigger queries and guide downstream generation towards
attacker-controlled outputs [9].
•Content Leakage, where the system reproduces verbatim or near-verbatim segments from private retrieved
documents, exposing confidential information directly [10].
The presence of these attack vectors poses a challenge to the core security properties of privacy, integrity, and
confidentiality that RAG deployments must uphold [6].
While recent work has begun to formalize these threats, the predominant approach is still to study each attack vector
in isolation. Recent surveys identify membership inference, content leakage, and data poisoning as core risks, yet
fail to empirically quantify their joint impact on end-to-end RAG utility [ 6]. Attack and benchmark-focused work
either targets a single class of adversary, such as membership inference against RAG [ 11,12], or concentrates on
knowledge-base corruption and prompt-injection style poisoning without modeling privacy leakage [ 9,13,14]. To
the best of our knowledge, we are not aware of prior empirical work that simultaneously (i) evaluates RAG under
concurrentmulti-vector threats, specifically membership inference and data poisoning in our empirical study, while
architecturally designing for content leakage, and (ii) measures the semantic utility cost of the defenses added to the
system to tackle these attacks.
Existing defenses address individual stages of the pipeline: differentially private retrieval (DP-RAG) [ 15] perturbs
query-document similarity scoresinside the retrieverafter encoding and before top- kselection to suppress membership
signals; TrustRAG-style clustering [ 16] filters semantic outliers from the retrieved set at the post-retrieval hook to
neutralize poisoned documents; and attention-variance filters [ 17] inspect attention concentration over retrieved passages
at the pre-generation hook to prune overly dominant context before decoding, thereby mitigating content leakage. A
natural engineering response is to activate all three mechanisms simultaneously, which we term astatic full-defense
stack, so that every incoming query passes through all three layers regardless of the actual threat level. However,
this indiscriminate combination imposes a cumulative overhead that disproportionately burdens the retrieval and
conditioning stages: DP-RAG noise disrupts high-precision nearest-neighbour search, TrustRAG filtering aggressively
prunes valid documents, and the attention-variance check conservatively prunes context that would otherwise ground
accurate answers.
This leads to what we call thesecurity-utility paradox: enabling the full defense stack reduces contextual recall by
41-46% across benchmark datasets. Faithfulness, by contrast, remains stable, confirming that the LLM generator itself
is unimpaired. The utility collapse originates largely in the retrieval phase: the generator is starved of sufficient context
and can no longer meet its intended task requirements. Our experimental results in Section 6 confirm this phenomenon
empirically for the evaluated poisoning and membership-inference settings.
To resolve this paradox, we proposeAdaptive Defense Orchestration (ADO), a modular framework that separates
risk assessment from defense enforcement. Rather than applying all defenses to every query, ADO activates only the
mechanisms warranted by the current threat level, preserving retrieval quality for benign workloads while strengthening
protection against the evaluated threats when adversarial signals are detected. We realize this policy through the
Sentinel-Strategist architecture, a two-stage orchestrator shown in Fig. 2. The split is intentional rather than cosmetic:
threat assessment and defense selection have different inputs, objectives, and failure modes. The Sentinel continuously
monitors lightweight pipeline signals, including lexical query overlap and vector-space dispersion among retrieved
documents, and compresses them into a structured per-query risk profile. The Strategist then maps that profile to
targeted defense configurations across four enforcement hooks in the RAG pipeline, dynamically enabling or tightening
individual defenses as warranted. Decoupling evidence fusion from action selection makes the control policy easier to
audit and recalibrate, allows the defense registry to evolve without rewriting the detector, and keeps the control prompts
small enough to run on compact controller models. This improves practical deployment viability, preserves retrieval
performance on benign traffic, and provides an adaptive security posture.
2

APREPRINT- APRIL24, 2026
OFFLINE 
/ 
INGESTION 
(Knowledge 
Base 
build)
ONLINE 
/ 
QUERY 
(Retrieval 
+ 
Generation)
Knowledge 
Base 
/ 
Vector 
Database
External 
Documents
Document 
Chunking
Embedding 
Model
User 
Query
Query 
Encoding 
(Embedding 
Model)
Retriever: 
top-k 
Similarity 
Search
Retrieved 
Context
Prompt 
Construction 
/ 
Augmented 
Prompt
Large 
Language 
Model 
(LLM) 
/ 
Generator
Final 
Output
Knowledge 
Base
Retrieval 
System
Generation 
System
top-k 
search
retrieved 
vectors
Figure 1: Architecture of the standard Retrieval-Augmented Generation (RAG) pipeline. The framework consists of
three main phases: offline ingestion (document chunking and embedding to build the knowledge base), online retrieval
(encoding the user query and performing a top- ksimilarity search), and augmentation (prompt construction utilizing
the retrieved context to condition the generator).
Conceptually, this design aligns with the Zero Trust Architecture (ZTA) model formalized by NIST SP 800-207 [ 18], in
which a centralized Policy Decision Point (PDP) continuously evaluates contextual risk and instructs distributed Policy
Enforcement Points (PEPs) that gate access to protected resources.
Our contributions are as follows:
•We provide an empirical study of RAG systems under two evaluated attack classes, membership inference and
data poisoning, and quantify the semantic utility cost of statically combining the corresponding defenses.
•We demonstrate that stacking the available defense modules as an always-on static policy causes a 41-46%
collapse in contextual recall, even when the underlying generator remains fully intact, identifying retrieval as
the primary bottleneck.
•We introduce ADO and the Sentinel-Strategist architecture, a unified multi-vector orchestration framework
that coordinates defenses for membership inference, data poisoning, and content leakage through separate
enforcement hooks. The Sentinel performs lightweight risk estimation, while the Strategist maps that risk
profile to hook-level actions; this separation makes the policy more auditable, easier to extend, and practical
to run with compact controller models. In this paper, the empirical evaluation centers on the first two attack
classes, while content leakage is incorporated architecturally through the A V-filter path.
•We provide code, attack configurations, Sentinel and Strategist prompt templates, and evaluation scripts
to support independent reproducibility; submission-time artifact access is described in the Open Science
appendix.
2 Related Work
2.1 Retrieval-Augmented Generation (RAG)
Retrieval-Augmented Generation (RAG) is designed to increase the capabilities of Large Language Models (LLMs) by
combining them with external knowledge sources [ 4]. This approach effectively handles the issues of hallucinations
and knowledge cutoff limitations found in standalone parametric models [ 19,20]. In contrast to traditional LLMs that
depend solely on fixed internal parameters, RAG systems actively fetch semantically relevant context from external
databases. As illustrated in Fig. 1, the pipeline consists of three phases: ingestion (document chunking and embedding),
retrieval (top- ksimilarity search), and augmentation (prompt construction for the generator) [ 4]. Recent developments
have progressed to include advanced techniques such as token-level retrieval methods [ 21], adaptive data chunking
3

APREPRINT- APRIL24, 2026
strategies [ 22], and graph-structured knowledge models that represent complex relational dependencies among entities
[23, 24].
2.2 Privacy & Security Risks in RAG Systems
Integrating external knowledge bases significantly improves LLMs, but it also changes the threat landscape by creating
additional attack surfaces. Therefore, we categorize these risks into three primary vectors: membership inference, data
poisoning, and content leakage.
2.2.1 Membership Inference Attacks (MIA)
Membership Inference Attacks (MIA) [ 25,26] pose a privacy risk by attempting to determine whether a specific
data entry was included in the training dataset or knowledge base of a machine learning model. For conventional
language models, MIAs have been thoroughly examined, with notable techniques such as the Loss Attack, which infers
membership based on the model’s loss [ 27], the Zlib Entropy Attack, which adjusts the loss by considering compression
size [ 28], and the Min- k% Probability Attack, which uses the least probable tokens in a sample to assess membership
[29]. RAG-MIA [ 7] determines membership by directly querying the system to identify whether a specific document
is included in the retrieved context, based on the model’s response. Extending this to a more targeted setting, the
Mask-Based Attack (MBA) [ 8] strategically masks tokens within a candidate document and queries the RAG system to
assess whether it reconstructs the masked content; successful reconstruction indicates that the document is present in
the retrieval knowledge base, making MBA particularly effective against systems that retrieve and expose verbatim
document segments.
2.2.2 Data Poisoning
Data poisoning attacks targeting Retrieval-Augmented Generation (RAG) systems [ 9,30] manipulate model outputs by
inserting adversarial documents into the retrieval knowledge base while leaving the model parameters unchanged. An
attacker adds specially crafted documents ( Dpoi) to the original corpus ( D), thereby creating a contaminated database
D⋆=D∪D poi. This ensures that the retriever yields these documents in response to particular trigger queries. Data
poisoning attacks typically fall into two categories: harmful generation and content promotion. In harmful generation,
the attacker aims to produce misleading, biased, or dangerous outputs. In content promotion, the goal is to steer the
system toward specific target phrases, entities, or brand mentions that may not be relevant to the user’s intent. A
poisoning attack is considered successful if the retriever yields at least one injected document in response to a trigger
queryq∗:
R(E(q∗),E(D⋆), k)∩D poi̸=∅.(1)
2.2.3 Content Leakage and Unintended Disclosure
In Retrieval-Augmented Generation (RAG) systems, content leakage refers to the unauthorized reconstruction of
sensitive information from the retrieval knowledge base. This risk is particularly critical in domains such as healthcare,
where the generator ( G) may produce segments that are similar to retrieved documents ( d), thereby exposing patient
information. Adversaries can induce such leakage using a combined query approach q=q i+qc. An anchor query
(qi) steers the retriever ( R) toward a specific group of sensitive documents, while a command prompt ( qc) instructs the
generator to replicate the retrieved content with minimal modifications. Leakage occurs when the similarity between the
generated output ( y) and a sensitive document ( di) exceeds a predefined threshold τ. Recent studies [ 6] systematically
investigate unauthorized content leakage in RAG systems and its privacy implications.
2.3 Defenses
Given the growing threat to RAG systems, specialized defense mechanisms are required. All these strategies target
specific stages of the RAG pipeline, like retrieval, augmentation, or generation, to reduce distinct attack vectors.
2.3.1 Defenses Against Membership Inference
In Membership Inference Attacks (MIAs), an attacker tries to find if a particular record exists within a system’s database.
Recent studies apply Differential Privacy (DP) [ 31] to the retrieval stage of Retrieval-Augmented Generation (RAG)
systems [ 15]. A common instantiation perturbs query-document similarity scores after query and document encoding
but before final top- kranking, thereby reducing the influence of any single document on the retrieved set and limiting
the adversary’s capacity to identify its presence.
4

APREPRINT- APRIL24, 2026
A randomized mechanism Mis said to fulfill (ε, δ) -differential privacy if, for every pair of neighboring datasets Dand
D−that vary by exactly one entry, as well as for any measurable output setO, the following condition is satisfied:
Pr[M(D)∈ O]≤eεPr[M(D−)∈ O] +δ.(2)
2.3.2 Defenses Against Data Poisoning
The main focus of Defenses against data poisoning is post-retrieval filtering and robust aggregation before it reaches the
generation phase. The goal is to recognize and eliminate malicious documents, such as those that insert misleading
statements or specific keywords, which the retriever pulls in along with valid evidence. TrustRAG [ 16] implements
a multi-stage validation process to enhance the reliability of the system, based on the idea that poisoned documents
frequently appear as semantic anomalies. ReliabilityRAG [ 13] enhances by prioritizing resilience for RAG systems.
In contrast to heuristic filtering, ReliabilityRAG offers statistical assurances that the system’s outputs will remain
consistent, even when there are a limited number of harmful documents among the top-k results.
2.3.3 Defenses Against Content Leakage & Unintended Disclosure
Measures to prevent content leakage concentrate on the safety barrier during the generation process. Their main goal
is to stop the Large Language Model (LLM) from reproducing exact or near-exact sections of gathered documents
that may contain confidential or sensitive information, such as personally identifiable information in medical records.
ControlNet [ 32] is presented as a firewall for retrieval-augmented generation (RAG) systems. Rather than depending
on keyword filtering, ControlNet observes the model’s internal activation patterns and hidden states throughout the
generation process. In the Attention-Variance Filter [ 17], the generator’s attention over the retrieved context is monitored
for anomalously concentrated weights. When a model is compelled to disclose certain content, often through malicious
prompting or poisoning, it assigns a disproportionately high attention weight to specific tokens within the passage to
override its internal biases.
3 System Model and Threat Landscape
In this section, we outline the retrieval-augmented generation (RAG) system model, the threat landscape, the security-
utility paradox, and the adversarial capabilities. We first define the underlying RAG pipeline, then describe the
adversary’s interaction model and objectives, and finally state the security-utility problem that motivates our defense
architecture.
3.1 Retrieval-Augmented Generation (RAG) System Model
As shown in Fig. 1, the RAG system comprises three main components: a knowledge base, a retrieval system, and a
generation system. A large language model (LLM) powers the generation component, while the knowledge base and
retriever work together to select and supply contextual information. This information conditions the model’s output on
relevant and factual data.
3.1.1 Offline Ingestion Phase
Prior to inference, the knowledge corpus must be prepared to enable efficient retrieval. The underlying document
collection is partitioned into a set of discrete passages D={d 1, . . . , d N}. The encoder Ecomputes dense vector
representations for all passages, yielding the encoded corpusE(D), which is stored in a vector index.
3.1.2 RAG Inference Pipeline
Throughout the manuscript, we use the retriever signature
Dq=R(E(q),E(D), k),
where R(·) denotes top- kretrieval over the encoded corpus. At inference time, the encoder embeds query q, the retriever
selects top-kdocumentsD q, and the generatorGproduces responseyconditioned on the augmented prompt:
y=G(Augment(q, R(E(q),E(D), k), s)).
wheresdenotes system-level instructions, andAugment(·)is a deterministic concatenation operation.
3.2 Adversarial Model
5

APREPRINT- APRIL24, 2026
3.2.1 Adversary Knowledge
We consider a black-box adversary Athat observes only query-response pairs. Formally, given the system S, the
adversary can choose a sequence of queries
q1, q2, . . . , q T
and observe the corresponding outputs
yt=G(Augment(q t, R(E(q t),E(D), k), s)),
t= 1, . . . , T.
The adversary hasnodirect access to the generator G’s model parameters θ, to its internal activations or gradients, or to
the raw documents in the corpus D. We also assume that the adversary does not control the embedding function Eor
the retrieval index implementation; the defender operates these.
3.2.2 Adversary Capabilities
Within this black-box setting, the adversary has two main capabilities.
Query AccessThe adversary can issue an unbounded (or practically high) number of queries to the system, possibly
at machine speed. We denote by Qthe space of admissible queries, and write Q={q t}T
t=1⊂ Q for the set of queries
submitted during an attack episode. This capability underlies both probing for membership information and crafting
prompts that induce leakage.
Data Injection.In addition, we allow the adversary to tamper with the knowledge base by injecting a set of crafted
documentsD poiinto the ingestion pipeline. The resulting corpus becomes
D⋆=D∪D poi,
and the retriever subsequently operates over the encoded corpus E(D⋆). This capability models a realistic compromise
of upstream data sources or ingestion jobs in RAG deployments.
We do not consider stronger capabilities such as direct modification of model weights, exfiltration of the entire corpus
D, or control over the embedding and retrieval infrastructure. These are out of scope for this work.
3.2.3 Threat Objectives
The adversary’s behavior has three distinct objectives, corresponding to standard security properties in information
forensics: privacy, integrity, and confidentiality. We formalize each of these objectives below.
Membership Inference (Privacy Violation).The adversary seeks to determine whether the knowledge base D
contains a specific sensitive documentd target. To make the non-member case explicit, we define the adjacent corpus
D−=D\ {d target},
and let Aissue a set of membership probes Qmia. In our setting, this objective is instantiated with the Mask-Based
Attack (MBA): each probe masks selected tokens from dtarget and queries the RAG system to reconstruct them. Let
r(q;D)∈ {0,1} denote whether probe q∈Q miaexactly reconstructs the masked target span when the system operates
over corpus D. A membership signal exists when exact reconstruction is more likely under the member corpus than
under the adjacent non-member corpus, i.e.,
Pr[r(q;D) = 1]>Pr
r(q;D−) = 1
.
This formulation keeps the underlying objective membership-based while matching the reconstruction-style signal
produced by MBA. Because our empirical study does not threshold that signal into a separate binary detector, we
operationalize MIA empirically using MBA exact mask-fill leakage rate:
Lmia=1
|Qmia|X
q∈Q miar(q;D).
That is, the reported MIA leakage rate is the percentage of evaluation probes for which the masked target content is
reconstructed exactly as the reference text; in Section 5, we report it either in aggregate or separately for member and
non-member probe sets.
6

APREPRINT- APRIL24, 2026
Data Poisoning (Integrity Violation).In the poisoning setting, the adversary injects a set of malicious documents
Dpoiinto the corpus and aims to steer the system towards a targeted adversarial output yadvfor a specific trigger query
qtrig. After poisoning, the effective corpus isD⋆=D∪D poi, and the retriever operates asR(E(q),E(D⋆), k).
The attacker’s objective is to maximize
Pr 
y=y advqtrig, R 
E(q trig),E(D⋆), k
.
Content Leakage (Confidentiality Violation).For content leakage, the adversary constructs a query qleakintended to
coerce the generator into revealing verbatim or near-verbatim content from private documents. Let Dpriv⊆D represent
a subset of sensitive documents that contains personal information, with dpriv∈D privsignifying a particular target
document.
The attack succeeds if the similarity between the generated outputyandd privexceeds a predefined thresholdτ:
sim 
y, d priv
> τ.
While we formalize content leakage here to provide a complete view of the RAG threat landscape, our empirical
evaluation in Section 5 restricts its attack benchmarks to membership inference and data poisoning, including their joint
evaluation on TriviaQA.
Concurrent Attacks.We allow the adversary to pursue these objectives concurrently over time. In other words, a
single adversarial client may interleave membership probes, poisoning triggers, and leakage prompts within the same
query stream. Consequently, any practical defense must operate under a unified orchestration framework that balances
robustness against all three threat vectors with the preservation of system utility, rather than tuning to a single attack in
isolation.
3.3 Security-Utility Problem Statement
3.3.1 Defense Policies
We model a defense configuration as a policy πthat, for each incoming query qand system state s, selects a setting for
the available defenses. Concretely, πcan enable or disable mechanisms such as differentially private retrieval (DP-RAG),
TrustRAG-style consistency filtering, and attention-variance-based leakage filters, and adjust their hyperparameters
(e.g., privacy budgetϵor similarity thresholds).
LetΠdenote the space of admissible policies. For a given policy π∈Π , query q, and underlying corpus D, the
defended system induces a distribution over outputs
y∼ Sπ(q, D),
3.3.2 Utility and Risk Measures
To quantify semantic utility, we employ an LLM-as-a-judge framework[ 33] to compute a vector of evaluation metrics.
The judge evaluates the system using combinations of four core parameters: the user query q, the retrieved context
Dq, the expected reference output y∗, and the actual generated output y. Let JLLMdenote the evaluation function.
Specifically, we compute four distinct metrics, each relying on a specific subset of these parameters:
•Contextual Recall:Evaluates if the retrieved context covers the expected answer, requiring(y∗, Dq)1.
•Contextual Relevancy:Measures the relevance of the retrieved context to the query, requiring(q, D q)2.
•Answer Relevancy:Assesses how directly the generated answer addresses the query, requiring(q, y)3.
•Faithfulness:Verifies that the generated answer is strictly grounded in the retrieved context, requiring (Dq, y)
4.
1https://deepeval.com/docs/metrics-contextual-recall#how-is-it-calculated
2https://deepeval.com/docs/metrics-contextual-relevancy#how-is-it-calculated
3https://deepeval.com/docs/metrics-answer-relevancy#how-is-it-calculated
4https://deepeval.com/docs/metrics-faithfulness#how-is-it-calculated
7

APREPRINT- APRIL24, 2026
Consequently, the overall utility vector can be formalized as:
U(q, y∗;π) =J LLM(q, D q, y∗, y)∈R4,
whereπdenotes the system policy that inducesD qandy.
In parallel, we define a risk vector representing vulnerabilities to specific attacks:
R(q;π) = 
Rmia(q;π), R poi(q;π), R leak(q;π)
,
where the components correspond to the risk of membership inference, data poisoning, and content leakage, respectively.
3.3.3 Security-Utility Optimization
At a high level, the defender seeks a policy πthat maximizes expected utility for benign workloads while keeping
expected risk under specified tolerances. This can be expressed as the constrained optimization problem
max
π∈ΠEq∼Q benign
u(U(q;π))
s.t. ¯R(π)⪯τ,
where u(·)is a scalar aggregation of the utility vector (e.g., a weighted sum of contextual recall and answer relevancy), τ
is a vector of acceptable risk thresholds for membership inference, poisoning, and leakage, and ⪯denotes element-wise
inequality.
In practice, we evaluate a small set of representative policies rather than solving this optimization exactly. Of particular
interest are:
We evaluate three representative policies: an undefended πbase, a fully-armed static stack πstatic, and the proposed
adaptiveπ ado, described in Section 4.
3.3.4 The Security-Utility Paradox
We define the security-utility paradox in RAG systems as the empirical phenomenon in which defensive policies such as
πstaticsatisfy the risk constraints R(π static)⪯τ , but drive the utility term U(π static)to low values. In our evaluation, this
results in a significant reduction in contextual recall on the order of 41−46% across datasets. However, faithfulness
remains unchanged, indicating that the underlying generator retains its reasoning ability but no longer receives adequate
context.
Our work addresses this problem along two complementary axes. First, we introduce a utility-aware evaluation
framework that jointly quantifies semantic utility and adversarial risk under different defense policies, making the
security-utility trade-off explicit. Second, we propose an Adaptive Defense Orchestration (ADO) policy πado, realized
via the Sentinel-Strategist architecture, which dynamically configures defenses per query in an attempt to preserve
utility while maintaining strong protection against the threats in Section 3.2. Section 4 details the design of ADO and
the underlying orchestration mechanisms, while Section 5 describes the experimental protocol used to instantiate and
evaluate the proposed framework.
4 Proposed Methodology: Adaptive Defense Orchestration
In this section, we present the Adaptive Defense Orchestration (ADO) framework, which operationalizes the security-
utility optimization objective defined in Section 3.3. Rather than treating individual attacks or defenses in isolation, ADO
provides both (i) a hook-based Data Plane that exposes measurable enforcement points for systematically quantifying the
trade-off between security and semantic utility, and (ii) a stateful Control Plane, realized through the Sentinel-Strategist
orchestrator, that dynamically configures defenses per query in response to observed risk.
4.1 Architectural Overview
ADO wraps the base RAG system S= (E, R, G, D) , with aControl Planeand aData Plane, following a strict
separation of policy decision from policy enforcement, a design that directly instantiates the Zero Trust Architecture
(ZTA) model of NIST SP 800-207 [ 18], in which a centralized Policy Decision Point (PDP) continuously evaluates
contextual risk and instructs distributed Policy Enforcement Points (PEPs) that gate access to protected resources.
TheControl Plane(PDP) comprises the Sentinel risk assessment stage, the Strategist defense configuration stage,
and the Persistence Layer maintaining the per-user Global Trust ScoreS trust. It is responsible exclusively for deciding
whichdefenses to activate andhowto parametrize them per query, without touching query or document content directly.
Section 4.3 details its internal operation.
8

APREPRINT- APRIL24, 2026
CONTROL 
PLANE
DATA 
PLANEq
y
Sentinel
(metrics 
/ 
signals)
Strategist
(defense 
config)
Global 
Trust 
Score 
(S_trust)
Hook 
A
Query 
sanitization 
/
pre-retrieval 
controls
Retriever
Vector 
DB
Hook 
B
Context 
filtering 
/
poisoning 
defense
Hook 
C
Prompt 
guardrails 
/
context 
pruning
Generator 
(LLM)
Hook 
D
Output 
auditing 
/
leakage 
checks
risk 
profile
retrieval 
lookup
Defense 
Manager
(metrics 
/ 
config 
dispatch)
metrics 
/ 
signals
defense 
config
metrics 
/ 
config
metrics 
/ 
config
metrics 
/ 
config
metrics 
/ 
config
retrieval 
lookup
Adversarial 
Evaluation 
Module
attack 
payloads 
/ 
evaluation 
inputs
Ingestion 
/ 
Embedding
documents 
/ 
embeddings
poisoning
Figure 2: The Sentinel-Strategist Architecture for Adaptive Defense Orchestration (ADO). The system is divided into a
Control Plane and a Data Plane. The Control Plane acts as a centralized Policy Decision Point: the Sentinel ingests
per-query signals and pipeline metrics to produce a structured risk profile, which is passed to the Strategist to generate a
defense configuration via the Defense Manager. The Data Plane applies those configurations across four enforcement
hooks in the RAG pipeline, Hook A (query sanitization and pre-retrieval controls), Hook B (context filtering and
poisoning defense), Hook C (prompt guardrails and context pruning), and Hook D (output auditing and leakage checks).
The Adversarial Evaluation Module and Ingestion pipeline interface with the query path and vector database to support
red-teaming, poisoning injection, and corpus updates.
TheData Plane(PEPs) comprises the Core RAG Engine and the four interception hooks pre-retrieval, post-retrieval,
pre-generation, and post-generation, that wrap it. It handles all live query traffic and executes the defense configurations
issued by the Control Plane. Section 4.2 describes these hook-level mechanisms and the Adversarial Evaluation Module
that enables controlled red-teaming against this layer.
4.2 Data Plane
To enable both systematic evaluation and dynamic control of defenses, we wrap the base RAG system S= (E, R, G, D)
with four configurable interception hooks. The Core RAG Engine forms the execution backbone of the framework. It
implements the end-to-end RAG pipeline, including data ingestion, document chunking, embedding into the vector
database, query encoding, retrieval, prompt augmentation, and response generation.
4.2.1 Interception Hooks
Around the Core RAG Engine, the Data Plane uses lightweight interceptor-style middleware that provides a unified
interface for enabling, disabling, or reconfiguring security mechanisms at query runtime. Through this wrapper, ADO
exposes four interception hooks aligned with key stages of the RAG pipeline:
•Pre-Retrieval Hook.Intercepts the raw query qprior to encoding. This hook supports query sanitization
(e.g., removal of personally identifiable information), benign transformations (e.g., query expansion), and the
injection of prompt-level guardrails before the query enters the embedding functionE.
•Post-Retrieval Hook.Intercepts the retrieved document set Dq=R(E(q),E(D), k) . At this stage, the
framework can apply defenses such as TrustRAG-style clustering and consistency checks to identify and
remove semantic outliers, or other post-retrieval filters designed to neutralize poisoning attempts. Algorithm 1
summarizes the TrustRAG filtering procedure used at this hook.
•Pre-Generation Hook.Wraps the augmented prompt q′that will be passed to the generator G. This hook is
used to apply system-level safety prompts, jailbreak- and prompt-injection defenses, and attention-variance-
based context pruning that removes disproportionately dominant passages before decoding. Algorithm 2
provides the corresponding attention-variance pruning routine.
9

APREPRINT- APRIL24, 2026
Algorithm 1TrustRAG: Clustering-Based Poisoning Filter
Require:Retrieved documentsD qwith embeddingsE, Similarity thresholdσ, ROUGE thresholdρ
Ensure:Filtered document setD f⊆D q
1:Normalise embeddings inE
2:(C 0, C1)←KMEANS(E, k=2)
3:Compute intra-cluster cosine means s0,s1forC 0, C1
4:ifmin( s0,s1)≥σthen
5:D f←D q ▷Both clusters are coherent; retain all
6:else
7:if s0≥s1then
8:D f←C 0
9:else
10:D f←C 1
11:end if
12:▷Retain cluster with higher coherence
13:end if
14:if|D f|>1then
15:for alld∈D fdo
16:c(d)←1
|Df|−1P
d′∈Df, d′̸=dcos(d, d′)
17:end for
18:for allunordered pairs(d i, dj)⊆D fdo
19:ifROUGE-L(d i, dj)> ρthen
20:Removearg min
d∈{d i,dj}c(d)fromD f ▷Prune near-duplicate with lower coherence
21:end if
22:end for
23:end if
24:returnD f
Algorithm 2Attention-Variance Filter: Context Pruning
Require:Context passagesC={c 1, . . . , c n}, questionq, maximum removalsR, variance thresholdϑ
Ensure:Pruned contextC′⊆C
1:C′←C
2:forr= 1toRdo
3:p←NORMALIZEDATTENTION(q, C′)
4:ifVar(p)≤ϑthen
5:break▷Attention is evenly distributed; no anomalous passage.
6:end if
7:j∗←arg max jpj
8:C′←C′\ {cj∗}▷Remove passage with disproportionately high attention
9:end for
10:returnC′
•Post-Generation Hook.Audits the final output yagainst both the query qand the retrieved context Dq. Here,
we attach output-side mechanisms such as hallucination detectors or policy-based response filters, which can
mask or regenerate tokens when the generated answer deviates from grounded evidence or violates disclosure
constraints.
For reproducibility, we include pseudocode for the two representative hook-level mechanisms discussed above.
We implement each concrete defense as a pluggable module. This includes differentially private retrieval (DP-RAG),
TrustRAG-style post-retrieval filtering, and attention-variance-based pre-generation context pruning. These modules
can be bound to one or more hooks with configurable parameters (e.g., privacy budget ϵ, clustering thresholds, similarity
cutoffs). This design allows the Sentinel-Strategist orchestrator to reconfigure the security posture by adjusting
hook-level parameters, without requiring changes to the Core RAG Engine.
10

APREPRINT- APRIL24, 2026
4.2.2 Adversarial Evaluation Module
To support rigorous, repeatable evaluation of security-utility trade-offs under different defense policies, we introduce an
Adversarial Evaluation Module that acts as an automated red-teaming environment. This module interacts with the
framework in an offline or batch setting and comprises three main components:
Poison Injector.The Poison Injector targets the ingestion layer by injecting a controlled set of adversarial documents
Dpoiinto the knowledge base. These documents are designed with particular trigger phrases or semantic similarities
so that, according to the retrieval function R, they show up among the top- kresults for specific trigger queries qtrig.
By varying the number and content of injected documents, we can systematically study the susceptibility of different
defense policies to data poisoning and its impact on retrieval accuracy and downstream outputs.
Adversarial Payload Generator and Probe Construction.The Adversarial Payload Generator constructs the
adversarial query sets used to probe privacy and confidentiality risks. It programmatically generates membership
inference probes, leakage-inducing prompts, and other attack payloads, together with the metadata needed for evaluation
such as identifiers of target documents, mask positions, or poisoning keywords. This component ensures that the same
adversarial workloads can be replayed across policiesπ base, πstatic, πadofor fair comparison.
Automated Evaluator and Metric Computation.The Automated Evaluator closes the loop by consuming paired
datasets of adversarial (and benign) payloads and corresponding system outputs. For each policy π, it computes the
security and utility metrics defined in Section 5, including attack success rate for poisoning, MBA exact mask-fill
leakage rates on member and non-member probe sets, leakage similarity statistics, and semantic utility scores such
as contextual recall, contextual relevancy, answer relevancy, and faithfulness. These measurements instantiate the
abstract utility U(π) and risk R(π) objectives from Section 3.3, enabling direct empirical comparison between static
full-defense configurations and the proposed adaptive orchestration.
4.3 Control Plane: Sentinel-Strategist Orchestration
The Control Plane is the Sentinel-Strategist orchestration logic, serving as a centralized Policy Decision Point (PDP). It
observes pipeline telemetry from the Data Plane and issues per-query defense configurations back to the enforcement
hooks without executing the core retrieval or generation path. In our implementation, the Control Plane does inspect
query-side information and derived retrieval metrics for policy decisions, but it does not consume raw document text
from the knowledge base.
The orchestration logic is implemented in three components: a persistence layer that maintains a per-user Global Trust
Score, the Sentinel risk assessment stage, and the Strategist defense configuration stage.
In our method, both the Sentinel and the Strategist are instantiated as large language model (LLM) agents, but with
deliberately different roles. The Sentinel takes as input a structured description of the current query, user state, and
pipeline metrics, and outputs a risk profile. The Strategist consumes only that profile together with the current trust
state and produces a defense plan for the four enforcement hooks. We separate these stages because risk estimation
and defense selection require different reasoning: the Sentinel aggregates weak signals and preserves calibration,
whereas the Strategist operates over a constrained action space that should remain stable even as the defense registry
evolves. This modularity also leaves open future deployment variants in which the Strategist could be partially static
or rule-based once the mapping from risk profiles to defense actions has been calibrated. A monolithic controller
would entangle detection with actuation, enlarge the prompt, and make failures harder to diagnose. The split instead
yields short, structured prompts that can be served by smaller control models, reducing per-query latency and cost.
Operationally, this yields a two-pass LLM pipeline per query: an early pass before retrieval to configure pre-retrieval
defenses, and a second pass after retrieval to refine the risk assessment using derived retrieval metrics and to configure
post-retrieval and post-generation defenses.
4.3.1 Persistence Layer and Global Trust Score
In order to address slow-burn attacks where attackers spread their probing or poisoning efforts across numerous queries,
the system maintains a Global Trust Score Strust∈[0,1] for each user. Conceptually, a user with a score of Strust= 1.0
has a record of safe, valuable interactions. A score of Strust= 0.0 signifies a history of suspicious behavior, such as
repeated jailbreak attempts or frequent activation of leakage filters. After each interaction, based on observed signals,
the Global Trust Score is updated. The observed signals include whether defenses were triggered, whether retrieved
context had to be pruned before generation, and whether outputs were masked or regenerated by output-side audits.
Strustserves as a prior in the orchestration logic. Lower trust values reduce the activation threshold for strict defenses,
11

APREPRINT- APRIL24, 2026
which makes them easier to trigger and thereby increases scrutiny. Higher trust values raise the threshold, resulting in a
lighter default security posture.
Formally, after each interaction, the Sentinel emits a score delta δ∈[−0.1,0.1] at both the pre-retrieval and post-
retrieval passes, based on whether the query exhibited adversarial patterns, whether defenses were triggered, whether
context pruning was required, and whether outputs were masked or regenerated by downstream audits. The Global
Trust Score is then updated as:
S(t+1)
trust =clip
S(t)
trust+δ pre+δ post,0,1
(3)
where δpreandδpostare the deltas emitted by the Sentinel at the pre- and post-retrieval passes respectively, and clip(·,0,1)
ensures Strustremains a valid probability. A benign interaction (no defense activations, no context pruning, no masked
output) yields a positive δ, gradually raising trust. A suspicious interaction yields a negative δ, lowering the activation
threshold for subsequent defenses. This design addresses slow-burn attacks in which adversaries spread probing
attempts across many queries to evade per-query detectors.
4.3.2 Stage 1: Sentinel Risk Assessment
To estimate the current threat level, the Sentinel acts as an intelligence fusion center that aggregates evidence from
multiple sources. For each incoming query q, it takes four categories of input: the Global Trust Score Strustfor the
requesting user, the raw query text q, a configurable window of recent session history (past queries and defense
activations), and a set of computationally inexpensive metrics computed over the query and retrieved results.
Before activating the generator G, the system calculates a set of heuristic indicators that anchor the Sentinel’s reasoning
in observable actions. These indicators fall into two groups: pre-retrieval metrics (query analysis) and retrieval metrics
(vector-space analysis). In the pre-retrieval set,Lexical Overlap MLEXmeasures Jaccard similarity between qand
recent queries from the same user; for membership inference attacks, high overlap (e.g., MLEX≥0.8 ) suggests
calibrated probing behavior.Complexity Score MCMPis a ratio of non-alphanumeric characters to total characters in q.
Intent Velocity MINTis the inverse of the time interval between consecutive queries. In the retrieval-metric set,Vector
Dispersion MDISmeasures the spread of retrieved document embeddings under cosine similarity, whileScore Drop-off
MDRPcompares the similarity scores of the top-1 and top- kretrieved documents. The Sentinel combines Strust, session
history, and these metrics to create arisk profile:
• An overall threat level
ℓ∈ {LOW,ELEV ATED,CRITICAL}.
•Estimated likelihoods for each attack method, such as pmia,ppoi, and pleak, which relate to membership
inference, data poisoning, and content leakage.
4.3.3 Stage 2: Strategist Defense Configuration
In the orchestration layer, the Strategist acts as a decision-making engine. It takes the Sentinel’s risk profile and
the current Strustvalue and maps them to a defense plan P. This plan determines how the four enforcement hooks
are configured. It keeps aDefense Registrythat includes differentially private retrieval (DP-RAG), TrustRAG-style
clustering, and attention-variance-based leakage filters. We represent the adaptive policy as:
πado: 
q, S trust,history, M LEX, M CMP, M INT,
MDIS, M DRP
7→ P,
wherePrepresents, for each hook, which defenses are enabled :
•High Membership Inference Risk: If pmiais high, MLEXexceeds a similarity threshold, and queries arrive at
high velocity, then the Strategist configures DP-RAG with a smaller ϵfor the upcoming retrieval stage. The
retriever then injects more noise into similarity scores before top- kselection, while poisoning defenses remain
at their nominal settings.
•High Poisoning Risk: The Strategist enables TrustRAG filtering at the post-retrieval hook when MDISand
related dispersion indicators exceed threshold values, discarding semantic outliers before generation.
•Low Risk and High Trust: The Strategist deactivates computationally expensive defenses when the overall
threat level is LOW andS trustis close to 1.0.
12

APREPRINT- APRIL24, 2026
Algorithm 3Adaptive Defense Orchestration: Per-Query Inference
Require:queryq, useru, retrieval budgetk
Ensure:answera
1:S trust←TRUSTMANAGER.GETSCORE(u)
2:M pre←COMPUTEPREMETRICS(q,history(u))
3:ρ pre←SENTINEL.PREANALYSE(q, S trust, M pre)
4:P pre←STRATEGIST(ρ pre,stage-1)
5:ACTIVATEDEFENSES(P pre)
6:q′, k′←DEFENSEMANAGER.PRERETRIEVAL(q, k)
7:D q←RETRIEVE(E(q′),E(D),top-k′,Ppre)
8:M post←COMPUTEPOSTMETRICS(D q)
9:ρ post←SENTINEL.POSTANALYSE(ρ pre, M post, Strust)
10:P post←STRATEGIST(ρ post,stage-2)
11:ACTIVATEDEFENSES(P post)
12:D f←DEFENSEMANAGER.POSTRETRIEVAL(D q, q)
13:(s sys, puser, C)←
DEFENSEMANAGER.PREGENERATION(q, D f,Ppost)
14:a←LLM.GENERATE(p user, C, s sys)
15:a←DEFENSEMANAGER.POSTGENERATION(a,P post)
16:TRUSTMANAGER.UPDATE(u,∆ =δ pre+δ post)
17:returna
4.3.4 Per-Query Orchestration Procedure
The Sentinel-Strategist orchestration executes two LLM passes per query, as formalized in Algorithm 3.
Pass 1 (Pre-Retrieval).Before retrieval, the Sentinel analyses the raw query q, the user’s Strust, and pre-retrieval
metrics Mpre(lexical overlap, complexity, intent velocity) to emit an initial risk profile ρpre. The Strategist maps ρpreto
a defense plan Pprethat sets any query-side transformations and configures retrieval-stage DP-RAG parameters for the
upcoming retrieval call.
Pass 2 (Post-Retrieval).After the top- kdocuments Dqare fetched, the Sentinel refines its assessment using post-
retrieval metrics Mpost(vector dispersion, score drop-off) to produce ρpost. The Strategist issues an updated plan
Ppostthat configures the post-retrieval hook (TrustRAG filtering), the pre-generation hook (prompt guardrails and
attention-variance context pruning), and the post-generation hook (output auditing). The Global Trust Score Strustis
then updated based on observed signals from the completed interaction.
5 Experiments
To evaluate our proposed Sentinel-Strategist architecture, we conducted a detailed analysis across multiple knowledge
domains. We designed our experiments to measure two key outcomes: (i) the degree of semantic utility degradation
caused by static, full-defense configurations, and (ii) the degree to which Adaptive Defense Orchestration (ADO)
restores utility while maintaining security assurances.
5.1 Datasets and Experimental Setup
We chose three standard benchmark datasets to evaluate our architecture, namely Natural Questions (NQ), PubMedQA,
and TriviaQA. For Natural Questions (NQ), we use the dpr-w100 split from ir_datasets to represent open-domain,
real-world user queries [ 34,35,36]. For PubMedQA, we adopt the pqa_labeled configuration to model medical question
answering, where accurate technical retrieval is needed [ 37]. For TriviaQA, we employ the rc (reading comprehension)
configuration [ 38]. Using a fixed random seed, we sample 50 benign queries from each dataset for the utility-oriented
evaluation of retrieval and generation quality. These benign queries are distinct from the attack-specific poisoning
and membership-inference query/probe sets described in Section 5.3. We execute each benign query against the
700-document ingested corpus. This sample size balances computational feasibility against the high per-query cost of
the LLM-as-a-Judge evaluation pipeline; it is intentionally scoped to demonstrate the mechanistic properties of ADO
rather than to characterize large-scale production performance (see Section 7 for a full discussion of scope limitations).
13

APREPRINT- APRIL24, 2026
5.2 Implementation Details
We implement a complete RAG pipeline and the ADO framework in a modular Python architecture. As shown in
Table 1, we useMeta-Llama-3.1-8B-Instruct5as the primary generation model, with a temperature of 0.0. Dense
embeddings were generated using the all-MiniLM-L6-v26model [ 39] and all vectors were indexed usingChromaDB7
for similarity search. We deliberately use smaller controller models for Sentinel and Strategist than one might reserve
for answer generation in a production assistant, because these components consume structured telemetry and emit
discrete hook configurations rather than long-form responses. In the default setup, Sentinel and Strategist use Llama-3-
8B-Instruct, served via Ollama8. We additionally evaluate the control plane with Gemma-3-4B, GPT-4o, Qwen-3-8B,
and Mistral-7B to study controller sensitivity. Most local controller variants therefore remain in the compact 4B-8B
range, which keeps the per-query orchestration loop more realistic for deployment by reducing incremental latency
and cost even though the control plane runs on every query. All models were accessed either through Ollama or via
the GPT-4o API, and all controller variants were prompted to operate over JSON-structured risk profiles and defense
configurations. The Global Trust Score Strustis initialized to 0.5for all users at the start of each evaluation episode,
reflecting a neutral prior, and updated per interaction via Eq. 3 throughout the attack sequence.
5.3 Threat Model and Defense Configuration
As shown in Table 2, our experimental evaluation focuses on two adversarial attacks under different defense config-
urations: data poisoning and membership inference. For poisoning, we instantiate thePoisonedRAGattack [ 9] by
injecting ten adversarial documents per target query to introduce conflicting contexts, and we evaluate it on a separate
set of 50 attack queries sampled with a fixed random seed. For membership inference, we instantiate the Mask-Based
Attack (MBA) [ 8] and evaluate it on a separate 50-probe set: 30 member probes (documents present in the corpus)
and 20 non-member probes (documents absent from the corpus). We use GPT-2-XL as a proxy reference model to
estimate the likelihood of masked tokens without bias from the target model. Consistent with the rest of the paper, we
score MBA using exact mask-fill accuracy rather than converting the reconstruction signal into a separate thresholded
inference-advantage statistic.
We tailor the attack assignments to the characteristics of each dataset. We do not evaluate membership inference on
Natural Questions. In PubMedQA, we focus on privacy-oriented evaluation and therefore do not run poisoning attacks.
TriviaQA is used for the joint setting in which both poisoning and membership inference are evaluated. Evaluating the
A V-filter path against a dedicated content-leakage benchmark, such as exact string matching or PII-style reconstruction,
is left to future work; in this paper, its primary role is to complete the Sentinel-Strategist orchestration logic.
Table 1: Core System and Retrieval Configuration
Parameter Value Description
Generator Model Llama-3.1-8B-Instruct RAG generator
Embedding Model all-MiniLM-L6-v2 Dense vectorizer
Corpus Size 700 Documents per dataset
Chunk Size 512 Tokens per chunk
Chunk Overlap 50 Sliding window overlap
Top-k5 Retrieved chunks
5.4 Evaluation Methodology
For our method evaluation, utility is evaluated using the DeepEval framework [ 40] with a Llama-3-base as a judge,
specifically over Answer Relevancy, Faithfulness, Contextual Recall, and Contextual Relevancy. Security is measured
via Attack Success Rate (ASR) for poisoning and MBA reconstruction leakage rate for membership inference. For
5https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
6https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
7https://www.trychroma.com/
8https://ollama.com/
14

APREPRINT- APRIL24, 2026
Table 2: Attack Configurations
Attack Parameter Value Notes
PoisonedRAG Poisoning Rate 10 Docs per query
Diversity Level True Semantic diversity
Mask-based MIA Masking Count 5 Tokens masked
Proxy Model GPT-2-XL Likelihood estimator
Table 3: Defense Hyperparameters
Defense Parameter Value Function
DP-RAGε3.0 Privacy budget
Candidate Multiplier 3 Oversampling
TrustRAG Similarity Threshold 0.88 Consistency filter
ROUGE Threshold 0.25 Content check
A V Filtering Threshold 50 Variance cutoff
Max Corruptions 3 Robustness limit
MIA, the reported leakage rate is the exact mask-fill accuracy of the MBA attack, i.e., the percentage of evaluation
probes for which the masked target content is reconstructed exactly as the reference text. Because MBA is instantiated
as a reconstruction attack in our setup, we do not report a separate inference-advantage statistic; instead, we report
leakage either as an aggregate rate (Table 6) or separately for member and non-member probe sets (Table 9). We used
two testing protocols: a static setup (without ADO), where the benign utility queries and the attack-specific query/probe
sets were evaluated sequentially, and an adaptive setup (with ADO), where inputs from those benign and adversarial
pools were interleaved and randomly shuffled. Rather than relying on predictable attack patterns, this mixed evaluation
ensures that the system is tested on its ability to dynamically adjust defenses per query.
6 Results and Observations
6.1 The Security-Utility Paradox
We begin by measuring theutility taxthat is introduced by static defense strategies. We compare the baseline setup
with no defenses against a Full Static Stack where DP-RAG, TrustRAG, and A V-Filtering are all enabled. Table 4
highlights how performance degrades across three benchmark datasets. Detailed metric-wise results and additional
defense combinations are presented in Table 8.
Table 4: The Security-Utility Paradox: Impact of static defenses on retrieval performance.
Context Recall Faithfulness Recall
DatasetBase Full Base FullDrop
Natural Questions 0.593 0.350 0.740 0.720-41.0%
PubMedQA 0.625 0.335 0.643 0.720-46.4%
TriviaQA 0.575 0.336 0.790 0.770-41.6%
Based on the results in Table 4, we observe a sharp drop in retrieval utility when all static defenses are enabled. For
PubMedQA, contextual recall falls by 46.4% (0.625 →0.335). This occurs because noise and filtering defenses
disrupt high-precision retrieval. Faithfulness remains stable (within ±2-3%) across datasets, indicating that the LLM’s
15

APREPRINT- APRIL24, 2026
reasoning and generation remain largely unaffected. As a result, the utility loss is driven primarily by impaired retrieval
rather than by degradation of the generator itself. Table 8 further shows that configurations containing DP-RAG exhibit
the largest recall drops, whereas TrustRAG-only and A V-Filtering-only remain much closer to the undefended baseline.
This pattern suggests that retrieval-side perturbation is the dominant contributor to utility loss in the evaluated settings.
6.2 Defense Effectiveness Against Targeted Threats
We measure the Attack Success Rate (ASR) for data poisoning attacks and the MBA reconstruction leakage rate, defined
as exact mask-fill accuracy, for membership-inference attacks.
6.2.1 Defense Against Data Poisoning
Table 5: Poisoning mitigation across datasets using TrustRAG (CR = Contextual Recall, F = Faithfulness).
Dataset Metric No Defense TrustRAG Change
TriviaQAASR (%) 56.00.0 -56.0%
CR 0.585 0.574 -1.9%
F 0.790 0.790 0.0%
Natural QuestionsASR (%) 35.00.0 -35.0%
CR 0.598 0.614 +2.7%
F 0.730 0.790 +8.2%
As shown in Table 5, TrustRAG is highly effective at mitigating data poisoning attacks. On both TriviaQA and Natural
Questions, the attack success rate (ASR) falls to zero in the evaluated setting. Relative to the full static stack in Table 4,
this targeted defense avoids the severe utility collapse caused by enabling every module simultaneously. We also
observe a slight improvement in faithfulness on Natural Questions, suggesting that the defense mechanism filters out
low-quality or poisoned retrieved passages.
6.2.2 Defense Against Membership Inference
Table 6: Privacy Preservation: DP-RAG reduces MBA reconstruction leakage rate (exact mask-fill accuracy).
Dataset No Defense DP-RAG Gain
TriviaQA 29.5% 16.8%+12.7 pp
PubMedQA 37.0% 19.3%+17.7 pp
Based on the results in Table 6, we observe that DP-RAG partially mitigates Membership Inference Attacks (MIAs).
Specifically, the MBA reconstruction leakage rate, measured as exact mask-fill accuracy, is reduced across datasets,
dropping from 29.5% to 16.8% for TriviaQA and from 37.0% to 19.3% for PubMedQA. These findings suggest that
injecting calibrated noise makes it harder for the attacker to reconstruct the masked target content exactly.
6.3 Adaptive Restoration: The ADO Advantage
ADO reconfigures defenses on a per-query basis through the Sentinel-Strategist pipeline described in Section 4.3. We
compare it against two static baselines: a Full Static Stack that enables all defenses for every query, and a Static Targeted
configuration that activates only the defense most relevant to the evaluated attack. Table 7 reports aggregate results, and
Table 9 provides the per-controller, per-dataset breakdown. In the ADO rows, the model label denotes the controller
used by the Sentinel-Strategist pair; the generator remains fixed. Poisoning metrics are macro-averaged over Natural
Questions and TriviaQA, while MIA metrics are macro-averaged over PubMedQA and TriviaQA.
MIA Defense: Uniform Suppression of MBA Reconstruction Leakage.As shown in Table 7, ADO drives the
aggregate MIA leakage rate to 0.0% across all five evaluated controller variants. In the per-dataset breakdown of Table 9,
all five variants also achieve 0.0% leakage on both member and non-member probe sets. This improves on the static
targeted DP-RAG baseline, which still leaves 18.05% aggregate leakage, while restoring contextual recall to 0.411 with
the Llama 3 controller and 0.497 with the Mistral (7B) controller.
Poisoning Defense and Inter-Model Variance.Poisoning results are more controller-sensitive. The Llama 3 and
Mistral controllers keep ASR at 0.0% and 4.0%, respectively, while recovering contextual recall to 0.429 and 0.470,
16

APREPRINT- APRIL24, 2026
Table 7: Comparison of ADO against static baselines on the evaluated poisoning and membership-inference settings.
N/A indicates that the security metric was not evaluated.
Attack Scenario Configuration Security Metric Contextual Recall Faithfulness
PoisoningFull Stack (Static) ASR: N/A 0.343 0.745
Static Targeted (TrustRAG) ASR: 0.0% 0.594 0.790
ADO (Ours, Llama 3) ASR: 0.0% 0.429 0.753
ADO (Ours, Gemma 3) ASR: 1.0% 0.297 0.760
ADO (Ours, GPT-4o) ASR: 35.0% 0.272 0.770
ADO (Ours, Qwen 3) ASR: 44.0% 0.439 0.730
ADO (Ours, Mistral) ASR: 4.0% 0.470 0.770
MIAFull Stack (Static) Leakage: N/A 0.336 0.745
Static Targeted (DP-RAG) Leakage: 18.05% 0.379 0.750
ADO (Ours, Llama 3) Leakage: 0.0% 0.411 0.737
ADO (Ours, Gemma 3) Leakage: 0.0% 0.282 0.745
ADO (Ours, GPT-4o) Leakage: 0.0% 0.290 0.760
ADO (Ours, Qwen 3) Leakage: 0.0% 0.420 0.720
ADO (Ours, Mistral) Leakage: 0.0% 0.497 0.698
Table 8: Utility scores for all defense permutations across datasets (CR = Contextual Recall, CP = Contextual Relevancy,
AR = Answer Relevancy, F = Faithfulness).
Natural Questions PubMedQA TriviaQA
CR CP AR F CR CP AR F CR CP AR F
Baseline (none) 0.593 0.489 0.930 0.740 0.625 0.424 0.833 0.643 0.575 0.514 0.700 0.790
DP-RAG only (D) 0.361 0.321 0.863 0.770 0.398 0.270 0.793 0.680 0.359 0.264 0.750 0.820
TrustRAG only (T) 0.586 0.470 0.860 0.720 0.584 0.418 0.803 0.712 0.567 0.509 0.720 0.810
A V-Filtering only (A) 0.592 0.449 0.780 0.770 0.591 0.390 0.760 0.666 0.606 0.513 0.680 0.830
D + T 0.380 0.344 0.853 0.670 0.383 0.267 0.803 0.660 0.339 0.266 0.790 0.750
D + A 0.364 0.286 0.740 0.580 0.354 0.224 0.563 0.700 0.354 0.222 0.740 0.760
T + A 0.571 0.387 0.750 0.690 0.520 0.326 0.477 0.770 0.590 0.494 0.713 0.790
Full Stack (D+T+A) 0.350 0.305 0.863 0.720 0.335 0.205 0.583 0.720 0.336 0.232 0.690 0.770
approaching the static targeted TrustRAG upper bound of 0.594. GPT-4o and Qwen 3 are less reliable, reaching
35.0%-44.0% aggregate ASR and up to 50.0% on Natural Questions, suggesting looser activation of TrustRAG under
borderline cases. Gemma 3 largely suppresses poisoning while restoring the least utility. Overall, ADO shows strong
MBA suppression in the evaluated membership-inference setting, whereas poisoning robustness remains controller-
dependent. Content leakage remains architecturally represented through the A V-filter path rather than benchmarked
separately.
7 Conclusion and Future Work
This work exposes a security-utility paradox in RAG systems: an always-on defense stack sharply reduces contextual
recall by 41-46% even when faithfulness remains stable, indicating that the main failure mode is impaired retrieval
rather than degraded generation. We address this with Adaptive Defense Orchestration (ADO), in which a Sentinel
forms per-query risk profiles and a separate Strategist maps them to hook-level defenses. This split keeps detection and
actuation auditable and allows the control plane to run on compact controller models. Across five controller-model
variants and three benchmark datasets with attack-specific pairings, ADO suppresses MBA reconstruction leakage
in the evaluated membership-inference setting and recovers retrieval utility relative to the static full stack, although
poisoning robustness remains model-sensitive. Content leakage is represented through the A V-filter path but is not
benchmarked separately here.
Limitations and Future WorkOur evaluation is intentionally scoped to feasibility: 50 benign queries and 50
adversarial queries per attack class per dataset, a 700-document corpus, single-run point estimates, no dedicated
content-leakage benchmark, and no systematic temporal or multi-turn attack patterns. The Sentinel-Strategist control
plane introduces additional orchestration overhead per query. We do not report a fixed end-to-end latency multiplier
because the observed cost varies with the inference backend, model serving configuration, and the defense intensity
17

APREPRINT- APRIL24, 2026
Table 9: Per-controller-model utility and security results. CR = Contextual Recall, CP = Contextual
Relevancy, AR = Answer Relevancy, F = Faithfulness; ASR = poisoning attack success rate; Leak. =
membership-inference leakage; N/A = not evaluated.
Controller Dataset AR F CR CP ASR Leak. (Mem.) Leak. (Non.)
Llama 3 (8B)PubMedQA 0.730 0.723 0.347 0.587 N/A 0.0% 0.0%
NaturalQ 0.897 0.755 0.382 0.523 0.0% N/A N/A
TriviaQA 0.810 0.750 0.475 0.539 0.0% 0.0% 0.0%
Gemma 3 (4B)PubMedQA 0.723 0.720 0.241 0.376 N/A 0.0% 0.0%
NaturalQ 0.770 0.750 0.271 0.365 2.0% N/A N/A
TriviaQA 0.770 0.770 0.322 0.345 0.0% 0.0% 0.0%
GPT-4oPubMedQA 0.763 0.760 0.290 0.393 N/A 0.0% 0.0%
NaturalQ 0.810 0.780 0.255 0.341 36.0% N/A N/A
TriviaQA 0.750 0.760 0.289 0.326 34.0% 0.0% 0.0%
Qwen 3 (8B)PubMedQA 0.850 0.750 0.364 0.469 N/A 0.0% 0.0%
NaturalQ 0.883 0.770 0.402 0.513 50.0% N/A N/A
TriviaQA 0.830 0.690 0.475 0.493 38.0% 0.0% 0.0%
Mistral (7B)PubMedQA 0.830 0.685 0.482 0.600 N/A 0.0% 0.0%
NaturalQ 0.930 0.830 0.427 0.564 4.0% N/A N/A
TriviaQA 0.850 0.710 0.512 0.555 4.0% 0.0% 0.0%
†0.0% leakage under ADO indicates that MBA did not exactly reconstruct the masked target span on the evaluated
probes.
selected for a given query. In practice, ADO activates defenses on most queries, but calibrates their aggressiveness to
the assessed threat level rather than applying the full static stack uniformly. Next steps include temporal and multi-turn
attack studies, model-specific prompt calibration, Sentinel distillation, and more efficient parallel defense activation.
References
[1]Hongbin Ye, Tong Liu, Aijia Zhang, Wei Hua, and Weiqiang Jia. Cognitive mirage: A review of hallucinations
in large language models. In Ningyu Zhang, Tianxing Wu, Meng Wang, Guilin Qi, Haofen Wang, and Huajun
Chen, editors,Proceedings of the First International OpenKG Workshop: Large Knowledge-Enhanced Models
co-located with The International Joint Conference on Artificial Intelligence (IJCAI 2024), volume 3818 ofCEUR
Workshop Proceedings, pages 14–36, Jeju Island, South Korea, August 2024. URL https://ceur-ws.org/
Vol3818/paper2.pdf.
[2]Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang,
Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, and Shuming Shi. Siren’s song in the ai ocean:
A survey on hallucination in large language models.Computational Linguistics, 51(4):1373–1418, 12 2025. ISSN
0891-2017. doi: 10.1162/COLI.a.16. URLhttps://doi.org/10.1162/COLI.a.16.
[3]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A
survey on rag meeting llms: Towards retrieval-augmented large language models. InProceedings of the 30th
ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’24, pages 6491–6501, New York,
NY , USA, 2024. Association for Computing Machinery. ISBN 9798400704901. doi: 10.1145/3637528.3671470.
URLhttps://doi.org/10.1145/3637528.3671470.
[4]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information processing systems, 33:9459–9474, 2020.
[5]Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm
Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. Improving language models by
retrieving from trillions of tokens. InInternational conference on machine learning, pages 2206–2240. PMLR,
2022.
[6]Atousa Arzanipour, Rouzbeh Behnia, Reza Ebrahimi, and Kaushik Dutta. Rag security and privacy: Formalizing
the threat model and attack surface, 2025. URLhttps://arxiv.org/abs/2509.20324.
[7]Maya Anderson, Guy Amit, and Abigail Goldsteen. Is my data in your retrieval database? membership inference
attacks against retrieval augmented generation. InProceedings of the 11th International Conference on Information
18

APREPRINT- APRIL24, 2026
Systems Security and Privacy, pages 474–485. SCITEPRESS - Science and Technology Publications, 2025. doi:
10.5220/0013108300003899. URLhttp://dx.doi.org/10.5220/0013108300003899.
[8]Mingrui Liu, Sixiao Zhang, and Cheng Long. Mask-based membership inference attacks for retrieval-augmented
generation. InProceedings of the ACM on Web Conference 2025, WWW ’25, pages 2894–2907, New York, NY ,
USA, 2025. Association for Computing Machinery. ISBN 9798400712746. doi: 10.1145/3696410.3714771. URL
https://doi.org/10.1145/3696410.3714771.
[9]Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia. {PoisonedRAG }: Knowledge corruption attacks to
{Retrieval-Augmented }generation of large language models. In34th USENIX Security Symposium (USENIX
Security 25), pages 3827–3844, 2025.
[10] Zhenting Qi, Hanlin Zhang, Eric P. Xing, Sham M. Kakade, and Himabindu Lakkaraju. Follow my instruction and
spill the beans: Scalable data extraction from retrieval-augmented generation systems. InICLR 2024 Workshop on
Navigating and Addressing Data Problems for Foundation Models, 2024. URL https://openreview.net/
forum?id=el5wbHYKeS.
[11] Kaiyue Feng, Guangsheng Zhang, Huan Tian, Heng Xu, Yanjun Zhang, Tianqing Zhu, Ming Ding, and Bo Liu.
Ragleak: Membership inference attacks on rag-based large language models. In Willy Susilo and Josef Pieprzyk,
editors,Information Security and Privacy, pages 147–166, Singapore, 2025. Springer Nature Singapore. ISBN
978-981-96-9101-2.
[12] Shixuan Sun, Siyuan Liang, Ruoyu Chen, Jianjie Huang, Jingzhi Li, and Xiaochun Cao. Sma: Who said that?
auditing membership leakage in semi-black-box rag controlling, 2025. URL https://arxiv.org/abs/2508.
09105.
[13] Zeyu Shen, Basileal Imana, Tong Wu, Chong Xiang, Prateek Mittal, and Aleksandra Korolova. ReliabilityRAG:
Effective and Provably Robust Defense for RAG-based Web-Search.arXiv e-prints, art. arXiv:2509.23519,
September 2025. doi: 10.48550/arXiv.2509.23519.
[14] Xun Liang, Simin Niu, Zhiyu Li, Sensen Zhang, Hanyu Wang, Feiyu Xiong, Zhaoxin Fan, Bo Tang, Jihao Zhao,
Jiawei Yang, Shichao Song, and Mengwei Wang. SafeRAG: Benchmarking security in retrieval-augmented
generation of large language model. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher
Pilehvar, editors,Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 4609–4631, Vienna, Austria, July 2025. Association for Computational Linguistics.
ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.230. URL https://aclanthology.org/2025.
acl-long.230/.
[15] Nicolas Grislain. Rag with differential privacy. In2025 IEEE Conference on Artificial Intelligence (CAI), pages
847–852, 2025. doi: 10.1109/CAI64502.2025.00150.
[16] Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, Yue Chen, Zhenhao Li, Zhaoyang Wang, Hamed Haddadi, and
Emine Yilmaz. Trustrag: Enhancing robustness and trustworthiness in retrieval-augmented generation, 2025. URL
https://arxiv.org/abs/2501.00879.
[17] Sarthak Choudhary, Nils Palumbo, Ashish Hooda, Krishnamurthy Dj Dvijotham, and Somesh Jha. Through the
stealth lens: Rethinking attacks and defenses in rag, 2025. URLhttps://arxiv.org/abs/2506.04390.
[18] Scott Rose, Oliver Borchert, Stu Mitchell, and Sean Connelly. Zero trust architecture. NIST Special Publication
800-207, National Institute of Standards and Technology, 2020. URL https://doi.org/10.6028/NIST.SP.
800-207.
[19] Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. Retrieval augmentation reduces
hallucination in conversation. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau
Yih, editors,Findings of the Association for Computational Linguistics: EMNLP 2021, pages 3784–3803, Punta
Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.
findings-emnlp.320. URLhttps://aclanthology.org/2021.findings-emnlp.320/.
[20] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang,
and Haofen Wang. Retrieval-augmented generation for large language models: A survey.arXiv preprint
arXiv:2312.10997, 2(1), 2023.
[21] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through
memorization: Nearest neighbor language models.arXiv preprint arXiv:1911.00172, 2019.
[22] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham.
In-context retrieval-augmented language models.Transactions of the Association for Computational Linguistics,
11:1316–1331, 2023. doi: 10.1162/tacl_a_00605. URLhttps://aclanthology.org/2023.tacl-1.75/.
19

APREPRINT- APRIL24, 2026
[23] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha
Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to
query-focused summarization, 2025. URLhttps://arxiv.org/abs/2404.16130.
[24] Shijie Wang, Wenqi Fan, Yue Feng, Lin Shanru, Xinyu Ma, Shuaiqiang Wang, and Dawei Yin. Knowledge graph
retrieval-augmented generation for LLM-based recommendation. In Wanxiang Che, Joyce Nabende, Ekaterina
Shutova, and Mohammad Taher Pilehvar, editors,Proceedings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pages 27152–27168, Vienna, Austria, July 2025.
Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.1317.
URLhttps://aclanthology.org/2025.acl-long.1317/.
[25] Hongsheng Hu, Zoran Salcic, Lichao Sun, Gillian Dobbie, Philip S. Yu, and Xuyun Zhang. Membership inference
attacks on machine learning: A survey.ACM Comput. Surv., 54(11s), September 2022. ISSN 0360-0300. doi:
10.1145/3523273. URLhttps://doi.org/10.1145/3523273.
[26] Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against
machine learning models. In2017 IEEE Symposium on Security and Privacy (SP), pages 3–18, 2017. doi:
10.1109/SP.2017.41.
[27] Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. Privacy risk in machine learning: Analyzing
the connection to overfitting. In2018 IEEE 31st computer security foundations symposium (CSF), pages 268–282.
IEEE, 2018.
[28] Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-V oss, Katherine Lee, Adam
Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training data from large language models. In
30th USENIX security symposium (USENIX Security 21), pages 2633–2650, 2021.
[29] Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke
Zettlemoyer. Detecting pretraining data from large language models.arXiv preprint arXiv:2310.16789, 2023.
[30] Baolei Zhang, Yuxi Chen, Zhuqing Liu, Lihai Nie, Tong Li, Zheli Liu, and Minghong Fang. Practical poisoning
attacks against retrieval-augmented generation.arXiv preprint arXiv:2504.03957, 2025.
[31] Cynthia Dwork. Differential privacy. In Michele Bugliesi, Bart Preneel, Vladimiro Sassone, and Ingo Wegener, ed-
itors,Automata, Languages and Programming, pages 1–12, Berlin, Heidelberg, 2006. Springer Berlin Heidelberg.
ISBN 978-3-540-35908-1.
[32] Hongwei Yao, Haoran Shi, Yidou Chen, Yixin Jiang, Cong Wang, and Zhan Qin. Controlnet: A firewall for
rag-based llm system, 2025. URLhttps://arxiv.org/abs/2504.09593.
[33] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan
Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena.Advances in neural
information processing systems, 36:46595–46623, 2023.
[34] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle
Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei
Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question
answering research.Transactions of the Association for Computational Linguistics, 7:452–466, 2019. doi:
10.1162/tacl_a_00276. URLhttps://aclanthology.org/Q19-1026/.
[35] Sean MacAvaney, Andrew Yates, Sergey Feldman, Doug Downey, Arman Cohan, and Nazli Goharian. Simplified
data wrangling with ir_datasets. InSIGIR, 2021.
[36] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. Dense passage retrieval for open-domain question answering. InProceedings of the 2020 Conference
on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781, Online, November 2020.
Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.550. URL https://www.
aclweb.org/anthology/2020.emnlp-main.550.
[37] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. Pubmedqa: A dataset for biomedical
research question answering. InProceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages
2567–2577, 2019.
[38] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised
challenge dataset for reading comprehension. In Regina Barzilay and Min-Yen Kan, editors,Proceedings of the
55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611,
Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1147. URL
https://aclanthology.org/P17-1147/.
20

APREPRINT- APRIL24, 2026
[39] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. Minilm: deep self-attention
distillation for task-agnostic compression of pre-trained transformers. InProceedings of the 34th International
Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY , USA, 2020. Curran Associates
Inc. ISBN 9781713829546.
[40] Confident AI. deepeval: The open-source llm evaluation framework. https://github.com/confident-ai/
deepeval, 2026. Version 3.7.8, accessed March 2026.
A Open Science
The core contributions of this paper depend on the following artifacts: (i) the RAG pipeline implementation, (ii) Sentinel
and Strategist prompts together with the control registry, (iii) attack scripts for PoisonedRAG and the Mask-Based
Attack (MBA), (iv) evaluation scripts, configuration files, and random seeds, and (v) instructions for reproducing the
reported tables and figures. For double-blind review, these artifacts are provided through the following anonymous
archive:
https://github.com/Pranavdec/Adaptive-RAG-Orchestrator
The anonymous archive maps the included files to the paper’s experiments and tables, documents the required software
dependencies, and explains any steps needed to reproduce the reported results. Third-party model weights, external
APIs, and access credentials are not redistributed in the archive; instead, we document the exact model identifiers,
configuration settings, and interfaces used in the experiments.
B Ethical Considerations
This paper studies membership-inference and data-poisoning attacks against retrieval-augmented generation systems in
order to evaluate defensive orchestration strategies. Because the work discusses attack mechanisms, releasing artifacts
may lower the barrier to misuse. We therefore evaluate all attacks in a controlled offline setting over benchmark datasets
and locally constructed corpora, and we do not test against third-party production systems or private user deployments.
Our experiments do not involve human-subject interaction, and the reported datasets are public benchmarks commonly
used for question answering research. The study does, however, concern privacy and integrity risks for systems that
may store sensitive documents. To reduce the chance of harm, we frame the attacks only as evaluation instruments
for measuring defense effectiveness, avoid including live targets or deployment-specific secrets, and separate reusable
configuration details from any sensitive credentials or infrastructure information. The goal of releasing the methodology
is to support reproducible defensive research rather than to operationalize attacks against real users.
21
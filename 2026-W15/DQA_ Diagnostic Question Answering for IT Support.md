# DQA: Diagnostic Question Answering for IT Support

**Authors**: Vishaal Kapoor, Mariam Dundua, Sarthak Ahuja, Neda Kordjazi, Evren Yortucboylu, Vaibhavi Padala, Derek Ho, Jennifer Whitted, Rebecca Steinert

**Published**: 2026-04-07 02:42:32

**PDF URL**: [https://arxiv.org/pdf/2604.05350v2](https://arxiv.org/pdf/2604.05350v2)

## Abstract
Enterprise IT support interactions are fundamentally diagnostic: effective resolution requires iterative evidence gathering from ambiguous user reports to identify an underlying root cause. While retrieval-augmented generation (RAG) provides grounding through historical cases, standard multi-turn RAG systems lack explicit diagnostic state and therefore struggle to accumulate evidence and resolve competing hypotheses across turns. We introduce DQA, a diagnostic question-answering framework that maintains persistent diagnostic state and aggregates retrieved cases at the level of root causes rather than individual documents. DQA combines conversational query rewriting, retrieval aggregation, and state-conditioned response generation to support systematic troubleshooting under enterprise latency and context constraints. We evaluate DQA on 150 anonymized enterprise IT support scenarios using a replay-based protocol. Averaged over three independent runs, DQA achieves a 78.7% success rate under a trajectory-level success criterion, compared to 41.3% for a multi-turn RAG baseline, while reducing average turns from 8.4 to 3.9.

## Full Text


<!-- PDF content starts -->

DQA: Diagnostic Question Answering for IT Support
Vishaal Kapoor Mariam Dundua Sarthak Ahuja Neda Kordjazi Evren Yortucboylu
Vaibhavi Padala Derek Ho Jennifer Whitted Rebecca Steinert
Amazon
{vishaalk,madundua,sarahuja,nedakord,yortuc,vapadala,derekjho,jenwhitt,rsteinrt}@amazon.com
Abstract
Enterprise IT support interactions are funda-
mentally diagnostic: effective resolution re-
quires iterative evidence gathering from am-
biguous user reports to identify an underlying
root cause. While retrieval-augmented genera-
tion (RAG) provides grounding through his-
torical cases, standard multi-turn RAG sys-
tems lack explicit diagnostic state and therefore
struggle to accumulate evidence and resolve
competing hypotheses across turns.
We introduce DQA, a diagnostic question-
answering framework that maintains persistent
diagnostic state and aggregates retrieved cases
at the level of root causes rather than indi-
vidual documents. DQA combines conversa-
tional query rewriting, retrieval aggregation,
and state-conditioned response generation to
support systematic troubleshooting under enter-
prise latency and context constraints.
We evaluate DQA on 150 anonymized enter-
prise IT support scenarios using a replay-based
protocol. Averaged over three independent
runs, DQA achieves a 78.7% success rate under
a trajectory-level success criterion, compared
to 41.3% for a multi-turn RAG baseline, while
reducing average turns from 8.4 to 3.9. This
improvement reflects the benefit of explicitly
representing competing explanations and ag-
gregating evidence across turns in unscripted
troubleshooting.
1 Introduction
Within enterprise IT support environments, interac-
tions are fundamentally diagnostic: users present
incomplete and ambiguous symptoms, and support
agents iteratively gather evidence to identify an un-
derlying root cause. Unlike single-turn fact-based
question answering, effective troubleshooting re-
quires tracking competing hypotheses, interpretingpartial signals, and deciding when to ask clarify-
ing questions versus proposing resolutions. These
properties make enterprise IT support a challenging
setting for retrieval-augmented language models.
Retrieval-augmented generation (RAG) is a stan-
dard approach for grounding language models in
external evidence. Extensions to multi-turn set-
tings incorporate conversational query rewriting to
improve retrieval under contextual references.
However, standard multi-turn RAG systems typ-
ically lack an explicit representation of diagnostic
state. Retrieved documents are consumed inde-
pendently at each turn, making it difficult to accu-
mulate evidence across the interaction, reconcile
conflicting signals, or maintain awareness of unre-
solved hypotheses. As a result, conversational co-
herence is often conflated with diagnostic progress.
This setting raises three challenges: limited con-
text, lack of diagnostic state, and principled action
selection under uncertainty.
We introduce DQA, a diagnostic question-
answering framework that explicitly maintains di-
agnostic state (relative support over competing
root-cause hypotheses and accumulated evidence)
across turns and aggregates retrieved evidence at
the level of root causes rather than individual doc-
uments. DQA is designed to support systematic
troubleshooting under strict latency and context
constraints typical of enterprise environments.
DQA aggregates retrieved cases into root-cause–
level signals, maintains a diagnostic belief state
tracking competing hypotheses, and uses this state
to guide questioning and resolution. This separa-
tion of evidence aggregation from decision-making
enables multi-turn diagnostic reasoning without
expanding the context window.
Our contribution is a system-level design thatarXiv:2604.05350v2  [cs.CL]  8 Apr 2026

integrates retrieval aggregation, persistent diagnos-
tic state, and state-conditioned action selection for
enterprise IT support. While evaluated in an IT sup-
port setting, the design principles underlying DQA
apply more broadly to retrieval-augmented systems
that must reason over large case repositories under
bounded context and latency.
2 Background and Related Work
Retrieval-augmented generation (RAG) conditions
language model outputs on retrieved external evi-
dence to improve factual grounding (Lewis et al.,
2020). In enterprise IT support, the retrieved cor-
pus often consists of internal documentation and
large ticket repositories. At this scale, raw ticket
retrieval yields redundant near-duplicate evidence:
many retrieved cases describe the same symptoms
and resolutions with superficial variation, wast-
ing context and latency budget. Case-based rea-
soning similarly frames troubleshooting as retriev-
ing similar resolved cases followed by adapta-
tion (Aamodt and Plaza, 1994); DQA follows this
spirit but targets aggregation—compressing large
retrieved neighborhoods into compact diagnostic
signals rather than adapting from a few exemplar
cases. Whereas simple deduplication removes re-
dundancy by collapsing similar cases, aggregation
preserves distributional information (e.g., cluster
prevalence) that guides downstream action selec-
tion.
Multi-turn conversational retrieval.Multi-
turn retrieval is difficult because user turns are
frequently non-standalone (e.g., anaphora, ellip-
sis, follow-ups), causing retrieval drift unless con-
text is resolved (Choi et al., 2018; Reddy et al.,
2019). Conversational query rewriting (CQR)
rewrites each user turn into a standalone query to
improve retrieval robustness under contextual refer-
ences (Yu et al., 2020; Vakulenko et al., 2021; Qian
and Dou, 2022). Benchmarks such as MTRAG and
CORAL show that conversational RAG quality de-
grades over longer interactions and that rewriting
can mitigate retrieval failures (Katsis et al., 2025;
Cheng et al., 2025). However, even with strong
rewriting, these systems do not explicitly represent
diagnostic progress or track which explanations
remain plausible as evidence accumulates.
Diagnostic dialogue.Diagnostic assistance
differs from fact-seeking QA: the assistant must
choose questions and actions that reduce uncer-
tainty over a latent cause rather than directly return-ing an answer. Prior diagnostic systems in technical
and medical domains often rely on decision trees or
manually engineered symptom hierarchies, which
can be brittle and costly to maintain as environ-
ments evolve (Aamodt and Plaza, 1994). Enter-
prise IT support amplifies these issues because user
descriptions are noisy, infrastructures are heteroge-
neous, and failure modes shift over time, making
static workflows difficult to sustain.
Gap addressed by DQA.Prior multi-turn
RAG systems improve retrieval robustness but do
not explicitly represent evolving diagnostic con-
text—which hypotheses remain plausible or how
evidence has shifted across turns. DQA introduces
persistent diagnostic state and evidence aggrega-
tion to enable systematic uncertainty reduction in
unscripted troubleshooting.
3 Diagnostic Question Answering (DQA)
We introduce Diagnostic Question Answering
(DQA), a framework for multi-turn enterprise IT
support that treats troubleshooting as progressive
uncertainty reduction over latent root causes. DQA
operates in unscripted diagnostic settings and main-
tains explicit diagnostic state across turns, enabling
systematic action selection under uncertainty.
3.1 Evidence Aggregation
DQA constructs data-driven priors over plausible
root causes using historical support tickets. Each
resolved ticket contains a free-text resolution field
extracted from the interaction transcript, which we
treat as a proxy for the underlying root cause.
Given an initial user description, the system re-
trieves similar tickets and clusters them by their
resolution embeddings. The clusters represented in
this neighborhood define the candidate hypotheses,
and their relative frequencies form an empirical
prior. This aggregation deduplicates near-identical
cases and combines evidence across many similar
incidents, producing a compact diagnostic signal.
3.2 Retrieval-Induced Diagnostic State
DQA maintains diagnostic state using a struc-
tured representation that summarizes support for
each candidate root cause. The state includes a
hypothesis-weight vector ht∈RK(where each
element corresponds to a root-cause cluster), along
with associated evidence such as representative
symptoms, KB articles, and canonical resolutions.
As new information becomes available, the sys-
tem re-retrieves and re-aggregates evidence, updat-

ing the diagnostic state without requiring explicit
symbolic belief tracking or hand-engineered proba-
bility models. Structured state fields persist across
turns, while retrieval-induced weights are recom-
puted from freshly aggregated evidence.
3.3 Action-Aware Diagnostic Policy
The diagnostic state guides action selection. Rather
than generating unconstrained free-form responses,
DQA frames troubleshooting as a policy over a
small set of diagnostic actions (cf. Yao et al., 2023).
This abstraction makes diagnostic progress explicit
by separating evidence formation from decisions
about what to do next.
Concretely, the policy operates over three ac-
tion types: clarifying questions, investigative steps,
and resolution proposals. These correspond to
gathering discriminative evidence, validating likely
causes, and proposing a fix once uncertainty is re-
duced.
As evidence accumulates and support concen-
trates on a few root causes, the policy shifts from
broad questioning toward targeted investigation and
resolution. Section 5.2 describes how this policy
is executed via state-conditioned response genera-
tion.
4 RAggG: Retrieval-Aggregated
Generation for Root-Cause Priors
We now describe RAggG (Retrieval-Aggregated
Generation) in detail. Standard RAG retrieves in-
dividual documents and conditions generation on
raw retrieved text. When applied to large reposi-
tories of historical support cases, this leads to re-
dundancy and poor signal-to-noise ratios: many
retrieved tickets describe near-identical symptoms
and resolutions with only superficial variation.
RAggG addresses this by aggregating retrieved
evidence along a task-relevant dimension—root
cause—prior to generation.
RAggG operates in three stages: (1) retrieval
selects a large candidate neighborhood, (2) aggre-
gation clusters candidates by root-cause descrip-
tions and computes per-cluster evidence counts and
representative cases, and (3) generation operates
on these aggregates rather than individual tickets.
Algorithm 1 summarizes the approach.
DQA operates as a multi-turn control loop. At
each turn, the current diagnostic state conditions
query rewriting, retrieval, and action selection.
RAggG is used as an aggregation subroutine toAlgorithm 1RAggG: Retrieval Neighborhood Ag-
gregation
1:Input:User queryx, ticket repositoryD
2: Encode query:z x←f(x)
3: Retrieve similar tickets:N(x)←TopK(D, z x, K)
4: Cluster retrieved tickets on root-cause:
5:{C j}J
j=1←Cluster(N(x))
6:foreach clusterC jdo
7: Count evidence mass: nj← |C j|{raw evidence
counts}
8: Select representative cases:R j←SelectRepr(C j)
9:end for
10:Output:Aggregated evidenceE={(n j, Rj)}J
j=1
Algorithm 2DQA: Diagnostic State Update
(Event-Driven)
1:Input:User query x, conversation history H, diagnostic
states, repositoryD
2: Rewrite query:˜x←Rewrite(x,H, s)
3: Aggregate evidence (Algorithm 1):
4:E ←RAGGG(˜x,D)
5: Update diagnostic state schema/content:
6:s←UpdateState(s,E,H)
7: Compute retrieval-induced weights (from counts inE):
8:foreach(n j,·)∈ Edo
9:h j←njP
(nj′,·)∈Enj′
10:end for
11:Store retrieval-conditioned hypothesis state in s:s.h←h
12: Generate response:
13:y←GenerateResponse(H ∪ {x}, s,E)
14: Update history:H ← H ∪ {(x, y)}
15:Output:Responsey
compress the retrieved neighborhood into updated
diagnostic evidence. We maintain a structured di-
agnostic state stacross turns, which includes a
retrieval-induced weight vector ht∈RKas one
component, along with additional fields such as ex-
tracted symptoms, exemplar tickets, and canonical
resolutions.
State updates are implemented via re-retrieval
and re-aggregation at each turn, rather than explic-
itly propagating probabilities over ht. This design
recomputes retrieval-induced weights from fresh
evidence while maintaining a persistent structured
diagnostic state.
4.1 Ticket Representation and Clustering
Each resolved ticket d∈ D is processed with
an encoder that extracts: (i) a short free-textroot
cause description(resolution field extracted from
the interaction transcript), (ii) a summary of user-
reported and engineer-confirmed symptoms, and
(iii) resolution steps. We encode the root cause
description with a sentence encoder f(·) (Reimers
and Gurevych, 2019) to obtain an embedding zd∈

Rh. We then cluster {zd}in this space using a scal-
able clustering algorithm (e.g., mini-batch k-means
or hierarchical agglomerative clustering), yielding
clusters{C 1, . . . , C K}.
Each cluster Ckis interpreted as a data-driven
approximation to a latent root cause rk, with an
empirical prior
p(rk) =|Ck|
|D|,(1)
which is reported for interpretability only and not
used online.
4.2 Query-Conditioned Priors and Scalability
Given a new contact with description x, we com-
pute a query embedding zx=f(x) and retrieve
the top- Knearest tickets in embedding space. Let
N(x) denote this neighborhood and let nk(x)be
the number of retrieved tickets that fall in cluster
Ck. At interaction time, DQA defines a retrieval-
conditioned hypothesis distribution over the clus-
ters present in the neighborhood:
hk=nk(x)P
k′∈K(x) nk′(x),(2)
where K(x) is the set of clusters that appear in the
retrieved neighborhood.
The aggregation step is crucial for scalability.
Rather than passing hundreds of individual tickets
to downstream processing, RAggG operates on a
small number of aggregate statistics (typically ≤5
clusters), enabling real-time performance even with
repositories containing 100K+ historical cases.
5 Diagnostic State and Action Selection
The diagnostic state is represented as a structured
state object that includes a retrieval-induced weight
vector h∈RK, whereP
khk= 1, with each ele-
ment corresponding to a root-cause cluster induced
from historical tickets. Each entry hkrepresents
the relative prevalence of cluster Ckwithin the
current retrieval neighborhood, computed by nor-
malizing evidence counts obtained via aggregation.
The weight vector is recomputed at each turn via
re-retrieval and aggregation rather than explicitly
propagating probabilities across turns.
5.1 Diagnostic State Schema
At each turn, the diagnostic state maintains four
components: (i) a current hypothesis string summa-
rizing the working diagnostic theory, (ii) a list ofuser-reported symptoms extracted from the conver-
sation, (iii) a ranked list of knowledge base (KB)
articles, dynamically filtered and re-ranked based
on the current diagnostic hypothesis state, and (iv) a
set of candidate root-cause clusters, each associated
with a retrieval-induced weight hk, a root-cause de-
scription, and representative resolved tickets.
State updates are hybrid. Structured, non-weight
fields of the diagnostic state—such as the hypothe-
sis string, symptom list, knowledge base references,
and the candidate set of root-cause clusters—are
updated incrementally as new information becomes
available. Retrieval-induced weights are then re-
computed from the current aggregated evidence
and stored in the state, rather than numerically prop-
agated from the previous turn.
This structured diagnostic state is serialized into
a prompt that the language model conditions on
when selecting the next diagnostic action.
5.2 Action Selection
DQA implements the policy from Section 3.3
through state-conditioned response generation. At
each turn, the language model is provided with the
current diagnostic state (including the leading hy-
potheses and accumulated evidence) together with
the conversation history and aggregated retrieval
summaries, and it generates the next response ac-
cordingly. A single generated response may in-
corporate elements of multiple action types (e.g.,
proposing a targeted investigative step while asking
a clarifying question).
Conditioning on explicit diagnostic state encour-
ages discriminative next steps rather than generic
information requests.
5.3 Query Rewriting for Conversational
Retrieval
Multi-turn enterprise support interactions fre-
quently contain anaphora, informal terminology,
and underspecified follow-ups (e.g., "still broken"),
which can cause retrieval drift if treated in isola-
tion. We apply lightweight conversational query
rewriting to normalize terminology and resolve ref-
erences.
The rewriter conditions on both dialogue his-
tory and the current diagnostic state. When users
provide vague follow-ups after a diagnostic hy-
pothesis has emerged, the rewriter incorporates the
dominant diagnostic context, maintaining focus on
plausible root causes and preventing retrieval from
reverting to generic failure modes.

6 Experimental Setup
We evaluate DQA using a replay-based protocol on
real enterprise IT support interactions. This section
describes the dataset, evaluation procedure, system
variants, and metrics.
6.1 Dataset and Replay Protocol
We evaluate on 150 anonymized enterprise IT sup-
port interactions, each consisting of a multi-turn
interaction, labeled with a final resolved root cause.
The evaluation set is intentionally heterogeneous,
spanning hardware failures, software configuration
issues, access management, authentication systems,
and enterprise applications, with a majority of cases
involving permission and access-related failures,
reflecting the distribution observed in real-world
enterprise helpdesk workloads.
To enable controlled comparison under realis-
tic diagnostic conditions, we adopt a replay-based
evaluation protocol. At each replay turn t, systems
are provided with the dialogue prefix up to that
point and generate a response, which may empha-
size information gathering, investigative steps, or
resolution proposals. The interaction ends when
the system proposes a resolution or after 15 turns.
User responses are generated by an LLM sim-
ulating the customer, conditioned on a scenario-
specific persona (communication style, knowledge
level, and priorities) and deterministic state transi-
tions that define outcomes of investigative actions
(e.g., "restart performed →same issue persists").
Trajectories diverge across systems: each system’s
responses produce different simulated user reac-
tions, enabling evaluation of diagnostic steering
rather than response quality to fixed inputs.
System responses are generated offline to pro-
duce complete interaction transcripts. A structured
post-hoc extraction pass identifies diagnostic facts,
resolution steps, and antipattern violations for eval-
uation. All systems share the same dialogue con-
text, evidence sources, and retrieval infrastructure;
differences arise from how evidence is formulated.
6.2 Systems Compared and Ablations
We compare DQA against increasingly capable
RAG-based systems that isolate the effects of in-
dividual components. RAG (no CQR) uses raw
user utterances for retrieval without conversational
rewriting. RAG (baseline) adds query rewriting to
produce standalone retrieval queries. RAG + Clus-
tering extends the baseline by aggregating retrievedtickets via root-cause clustering (RAggG), with-
out maintaining persistent diagnostic state. DQA
further adds a persistent diagnostic state to guide
action selection across turns. All systems share the
same retrieval infrastructure, encoders, language
model, and high-level prompting framework, oper-
ating over a combined corpus of historical tickets
and KB articles; aggregation and diagnostic hy-
pothesis induction are applied to tickets, while KB
articles are surfaced as supporting evidence. DQA
additionally conditions responses on explicit diag-
nostic state maintained across turns to guide action
selection.
6.3 Evaluation Metrics
We evaluate systems using a task-level success cri-
terion aligned with enterprise troubleshooting re-
quirements. Each scenario specifies a set of re-
quired diagnostic facts, required resolution steps,
and one antipattern constraint. All three compo-
nents are evaluated by an LLM judge (Zheng et al.,
2023) applied to the full interaction trajectory.
Diagnosis score measures the fraction of re-
quired diagnostic facts correctly identified (range
0–1). Resolution score measures the fraction of
required resolution steps provided (range 0–1). An-
tipattern compliance evaluates whether the agent
exhibited a scenario-specific problematic behav-
ior—such as redundant troubleshooting of steps the
user has already attempted, pursuing a diagnostic
path that contradicts available symptoms, or setting
false expectations about resolution timelines.
A test case is considered successful if, at any
point during the interaction, the system achieves
diagnosis score = 1.0, resolution score = 1.0, and
satisfies the antipattern constraint. Success does not
require the system to terminate on the turn at which
these criteria are met. Scores reported in Table
1 are averaged over all 150 scenarios, including
unsuccessful cases.
We additionally report average interaction length
(number of system turns until termination) as a
secondary efficiency metric.
7 Results
This section reports comparative performance be-
tween DQA and RAG-based baselines under the
replay-based evaluation protocol described in Sec-
tion 6. All reported results are averaged over three
independent executions of the replay protocol.

System Success Diag. Res. Turns↓
RAG (no CQR) 40.4% 0.86 0.73 8.17
RAG (baseline) 41.3% 0.86 0.73 8.43
RAG + Clustering 53.8% 0.90 0.80 6.53
DQA (ours)78.7% 0.99 0.94 3.93
∆vs baseline+37.4pp +0.13 +0.21−4.50
Table 1: Diagnostic performance averaged over three
runs on 150 enterprise IT support scenarios. DQA im-
proves both end-to-end success and convergence effi-
ciency, resolving cases in substantially fewer system
interaction turns than RAG-based baselines.
7.1 Overall Diagnostic Performance
Table 1 summarizes diagnostic performance across
system variants under the success criterion defined
in Section 6.3. The full DQA system achieves
a 78.7% success rate, outperforming all ablated
variants under the replay-based protocol.
Compared to the baseline RAG system (41.3%),
DQA improves success rate by 37.4 percentage
points, a 90.6% relative improvement. Improve-
ments are observed across diagnosis completeness,
resolution completeness, and antipattern avoidance.
These gains are consistent with the hypothesis
that persistent diagnostic state enables evidence
accumulation across turns. In addition to higher
diagnostic success, DQA converges substantially
faster. Averaged over three independent runs, DQA
resolves successful cases in 3.9 turns on average,
compared to 8.4 turns for the baseline RAG system.
7.2 Ablation Results
Ablation results in Table 2 isolate the contribu-
tions of individual system components. Conver-
sational query rewriting provides a modest im-
provement over raw retrieval, primarily by miti-
gating retrieval drift in context-dependent user ut-
terances. The relatively modest gains from con-
versational query rewriting likely reflect the eval-
uation setup: replayed user turns, including simu-
lated persona variations, are generally more well-
formed and information-dense than many live en-
terprise inputs, which are often underspecified or
noisy—conditions under which CQR is known to
provide larger benefits. Retrieval aggregation via
clustering improves success rate from 41.3% to
53.8%, indicating that compressing retrieved evi-
dence at the root-cause level improves diagnostic
signal quality.
Introducing persistent diagnostic state further in-
creases success to 78.7%, representing the largestComponent Introduced Success∆Relative Gain
Query Rewriting (CQR) +0.9pp +2.2%
Semantic Clustering +12.5pp +30.3%
Diagnostic State +24.9pp +46.3%
Table 2: Ablation results showing the effect of query
rewriting, retrieval clustering, and persistent diagnostic
state.
incremental gain among the evaluated components.
These results suggest that persistent diagnostic state
contributes more to end-to-end troubleshooting suc-
cess than retrieval improvements alone.
7.3 Latency and Scalability
DQA incurs one additional LLM call per turn for
query rewriting and hypothesis-conditioned state
updates, but converges in substantially fewer turns
(3.9 vs. 8.4), yielding comparable or lower total
interaction cost. By structuring interactions into
targeted diagnostic actions, DQA focuses computa-
tion on uncertainty reduction. Retrieval, clustering,
and aggregation use non-LLM preprocessing with
offline-cached embeddings, maintaining interactive
latency.
8 Conclusion
We presented DQA, a framework for multi-turn
enterprise IT support that treats troubleshooting as
an iterative diagnostic process. DQA integrates
RAggG for aggregating large retrieval neighbor-
hoods into root-cause–level evidence, a persistent
diagnostic state that organizes evidence across
turns, and state-conditioned action selection for
questioning, investigation, and resolution.
Rather than propagating probabilistic beliefs,
DQA recomputes diagnostic support at each turn
from freshly aggregated retrieval evidence while
maintaining structured diagnostic context.
More broadly, RAggG illustrates a general pat-
tern for retrieval-augmented systems: compressing
large, redundant retrieval results into compact, task-
relevant aggregates that can guide multi-turn deci-
sion making under latency and context constraints.
9 Limitations
This work has several limitations. First, our evalu-
ation uses simulated users rather than live deploy-
ment, which enables controlled comparison but
does not capture user experience factors such as
perceived helpfulness or trust. Second, poorly doc-
umented resolutions may yield noisy root-cause

structure. Third, we evaluate on a single enterprise
IT support domain; generalization to other diag-
nostic settings (e.g., medical diagnosis, hardware
troubleshooting) remains to be demonstrated.
Ethics Statement
This work uses anonymized enterprise IT support
logs with all personally identifiable information
removed.
References
Agnar Aamodt and Enric Plaza. 1994. Case-based rea-
soning: Foundational issues, methodological varia-
tions, and system approaches.AI Communications,
7(1):39–59.
Yiruo Cheng, Xinzhe Wu, Yiwei Zhang, Wenxuan Feng,
Minhao Chen, Qianqian Liu, and Ming Yin. 2025.
CORAL: Benchmarking multi-turn conversational
retrieval-augmented generation. InFindings of the
North American Chapter of the Association for Com-
putational Linguistics.
Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-
tau Yih, Yejin Choi, Percy Liang, and Luke Zettle-
moyer. 2018. QuAC: Question answering in context.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2174–2184.
Yannis Katsis, Sara Rosenthal, Kshitij Fadnis, Jian-
feng Chen, and Avirup Sil. 2025. MTRAG: A
multi-turn conversational benchmark for evaluat-
ing retrieval-augmented generation systems.arXiv
preprint arXiv:2501.03468.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474.
Hongjin Qian and Zhicheng Dou. 2022. Explicit query
rewriting for conversational dense retrieval. InFind-
ings of the Association for Computational Linguistics:
EMNLP 2022, pages 4054–4064.
Siva Reddy, Danqi Chen, and Christopher D. Manning.
2019. CoQA: A conversational question answering
challenge.Transactions of the Association for Com-
putational Linguistics, 7:249–266.
Nils Reimers and Iryna Gurevych. 2019. Sentence-
BERT: Sentence embeddings using siamese BERT-
networks. InProceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing,
pages 3982–3992.Svitlana Vakulenko, Shayne Longpre, Zhucheng Tu,
and Raviteja Anantha. 2021. A comparison of ques-
tion rewriting methods for conversational passage
retrieval. InAdvances in Information Retrieval: 43rd
European Conference on IR Research, pages 418–
424. Springer.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
ReAct: Synergizing reasoning and acting in language
models. InInternational Conference on Learning
Representations.
Shi Yu, Jiahua Liu, Jingqin Yang, Chenguang Shen,
Linjun Wang, Kun Liu, and Zhenglu Zhang. 2020.
Few-shot generative conversational query rewriting.
InProceedings of the 43rd International ACM SI-
GIR Conference on Research and Development in
Information Retrieval, pages 1933–1936.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging
LLM-as-a-judge with MT-bench and chatbot arena.
InAdvances in Neural Information Processing Sys-
tems, volume 36.
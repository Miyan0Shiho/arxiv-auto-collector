# FedMark-FM: Auditable, Risk-Adjusted Data Markets for Federated Foundation-Model Adaptation

**Authors**: Phat T. Tran-Truong, Xuan-Bach Le, Minh Nhat Nguyen

**Published**: 2026-07-08 15:23:30

**PDF URL**: [https://arxiv.org/pdf/2607.07529v1](https://arxiv.org/pdf/2607.07529v1)

## Abstract
Federated foundation-model adaptation increasingly relies on heterogeneous private artifacts (retrieval corpora, prompts and demonstrations, LoRA adapters, preference and safety data, and update sketches), yet existing federated-learning incentive mechanisms price clients as homogeneous data or update providers. This assumption poorly matches foundation-model pipelines, where contribution value is heterogeneous, non-IID, pipeline-dependent, privacy-constrained, and vulnerable to strategic behavior. We propose FedMark-FM, an auditable, risk-adjusted data-market framework that models clients as sellers of typed artifacts, estimates marginal contribution with S3Val, a stratified, uncertainty-aware Shapley estimator supporting pipeline-ordered valuation, and converts lower-confidence-bound values into budget-feasible payments penalizing duplication, sybil splitting, poisoned adapters, privacy-budget gaming, and cost inflation. We evaluate FedMark-FM-Bench across FEVER retrieval, held-out generator-backed RAG, and trained PEFT/LoRA tracks. Under a held-out prompt-injection poisoner, FedMark-FM improves downstream accuracy by 7.5-8.1 points over volume, leave-one-out, and FL-Shapley while selecting zero strategic clients. Split-conformal calibration reaches full lower-bound coverage at mean width 0.0141, versus 0.33 for naive intervals. We prove pipeline-ordered valuation is the unique credit rule respecting serving causality, and show it materially changes credit assignment (Spearman 0.76, selected-set overlap 0.67) while leaving held-out task quality unchanged; the market preserves rare specialists with audit-ready ledgers at 200-1000-client scale. FedMark-FM shows incentives for federated foundation models can be engineered as auditable data infrastructure coupling valuation, mechanism design, privacy interfaces, and pipeline-order semantics.

## Full Text


<!-- PDF content starts -->

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 1
FedMark-FM: Auditable, Risk-Adjusted Data
Markets for Federated Foundation-Model
Adaptation
Phat T. Tran-Truong1,2, *, Xuan-Bach Le1,2, †, and Minh Nhat Nguyen3
✦
Abstract—Federated foundation-model adaptation increasingly relies
on heterogeneous private artifacts—retrieval corpora, prompts and
demonstrations, LoRA adapters, preference and safety data, and update
sketches—yet existing federated-learning incentive mechanisms price
clients as homogeneous data or update providers. This assumption is
poorly matched to foundation-model pipelines, where contribution value
is heterogeneous, non-IID, pipeline-dependent, privacy-constrained,
and vulnerable to strategic behavior. We proposeFedMark-FM(a
FederatedMarket forFoundationModels), an auditable, risk-adjusted
data-market framework that models clients as sellers of typed arti-
facts, estimates marginal contribution withS3Val(SecureSurrogate
ShapleyValuation)—a stratified, uncertainty-aware Shapley estima-
tor that also supports pipeline-ordered valuation—and converts lower-
confidence-bound values into budget-feasible payments that penalize
duplication, sybil splitting, poisoned adapters, privacy-budget gaming,
and cost inflation. We evaluateFedMark-FM-Bench(the FedMark-FM
Benchmark) across FEVER retrieval, held-out generator-backed RAG,
and trained PEFT/LoRA tracks. Under a held-out prompt-injection poi-
soner, FedMark-FM improves downstream accuracy by 7.5–8.1 points
over volume, leave-one-out, and FL-Shapley while selecting zero strate-
gic clients. Split-conformal calibration reaches full lower-bound coverage
at mean width 0.0141, versus 0.33 for naive intervals. We prove that
pipeline-ordered valuation is the unique credit rule respecting serving
causality, and show it materially changes credit assignment (Spearman
0.76, selected-set overlap 0.67) while leaving held-out task quality un-
changed; the market also preserves rare specialists with audit-ready
ledgers at 200–1000-client scale. FedMark-FM shows that incentives
for federated foundation models can be engineered as auditable data
infrastructure that couples valuation, mechanism design, privacy inter-
faces, and pipeline-order semantics.
Index Terms—Federated foundation models, data markets, data val-
uation, incentive mechanisms, retrieval-augmented generation, LoRA
adapters, data-centric AI.
1 INTRODUCTION
Foundation models are increasingly deployed as data
pipelines rather than monolithic networks. A production
1Faculty of Computer Science and Engineering, Ho Chi Minh City University
of Technology (HCMUT), 268 Ly Thuong Kiet Street, Dien Hong Ward, Ho
Chi Minh City, Vietnam.
2Vietnam National University Ho Chi Minh City, Linh Xuan Ward, Ho Chi
Minh City, Vietnam.
3RMIT University, Ho Chi Minh City, Vietnam.
E-mail: phatttt@hcmut.edu.vn; lexuanbach@hcmut.edu.vn;
minh.nguyen244@rmit.edu.vn.
*First author.†Corresponding author.assistant, search stack, or enterprise copilot answers a query
by composing retrieved passages, prompt templates, in-
context demonstrations, parameter-efficient adapters, pref-
erence data, and safety probes, and these artifacts are re-
freshed continuously after deployment. In cross-silo settings
they are owned by different organizations—hospitals, data
vendors, safety labs—that cannot or will not centralize raw
data. Realizing federated foundation models (FedFMs) at
this scale is therefore as much an economic problem as an
algorithmic one: a shared pipeline improves only if the own-
ers of these private artifacts are paid for their contribution,
and they participate only if that contribution is measured
and rewarded fairly, at scale, and with evidence they can
dispute.
Two research lines bear on this problem. Federated-
learning (FL) incentive mechanisms allocate rewards for
participation and risk sharing, typically through Shapley-
based or auction-based schemes over datasets or model
updates [1], [2]. Data-valuation methods, led by Data Shap-
ley and its efficient approximations, quantify each source’s
marginal contribution to a trained model [3], [4]. Recent
FedFM surveys, in turn, raise incentives, game mechanisms,
privacy, and heterogeneity to first-order open problems [5].
However, these approaches price a client as a homo-
geneous provider of a dataset or gradient under a fixed
supervised objective, and stop at a scalar reward com-
puted after training. A deployable FedFM market needs
four properties that no existing method provides together:
heterogeneous artifacts must be traded astyped, separately
priced products; credit must respect theserving order—
retrieval precedes prompting, which precedes adaptation,
which precedes safety—rather than conflating upstream
and downstream contributions; valuation mustscale under
privacy constraintsthat forbid inspecting raw artifacts; and
payments must leave anauditable ledgerthat can be disputed
and revalued when the model, retriever, or policy changes.
In this paper we propose FedMark-FM, an auditable,
risk-adjusted data-market framework for FedFM adapta-
tion that supplies all four properties. A client contributes
typed artifacts—retrieval corpora, LoRA adapters, prompts,
demonstrations, preference data, safety data, or update
sketches—and the market evaluates coalitions through a
contract utility spanning task quality, safety, robustness, la-
arXiv:2607.07529v1  [cs.GT]  8 Jul 2026

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 2
tency, privacy budget, and cost. Its central conceptual move
is to stop treating FedFM artifacts as exchangeable players:
because retrieval changes the context seen by prompts and
adapters, FedMark-FM assigns credit along the contract-
visible serving order, and we prove that this pipeline-
ordered rule is theuniquecredit assignment consistent with
serving causality (Theorem 1).
Because exhaustive coalition evaluation is infeasible,
FedMark-FM estimates value with S3Val (Secure Surrogate
Shapley Valuation), a stratified, uncertainty-aware Shapley
estimator that combines contribution sketches, redundancy
clustering, pipeline-ordered sampling, and a learned utility
surrogate, and converts lower-confidence-bound values into
budget-feasible payments that penalize duplication, sybil
splitting, poisoned adapters, privacy-budget gaming, and
cost inflation. On real FEVER retrieval, held-out generator-
backed RAG, and trained PEFT/LoRA tracks, FedMark-
FM improves downstream accuracy by 7.5–8.1 points over
volume, leave-one-out, and FL-Shapley under a held-out
prompt-injection poisoner while selecting zero strategic
clients, and its split-conformal payments attain full lower-
bound coverage at mean interval width 0.0141.
To make the gap concrete, suppose a hospital contributes
a small but unique biomedical retrieval corpus, a web
vendor contributes a large but redundant one, a strategic
client copies the hospital corpus under another identity, and
a safety lab contributes probes that improve policy compli-
ance without improving ordinary retrieval. Volume-based
rewards overpay the vendor and the duplicate; a naive
validation-loss delta underpays the hospital when biomed-
ical queries are rare; and a standard Shapley estimator
becomes too expensive once each coalition requires adapter
routing or RAG evaluation. FedMark-FM instead values
typed artifacts under a contract utility, stratifies evaluations
by domain and artifact type, discounts duplicate clusters,
rewards scarce safety contributions, and audits high-risk
payments, while retaining enough evidence to contest a
valuation without exposing raw records. The framework is
domain-agnostic; a licensed clinical or enterprise case study
is future work rather than a claim of the present paper.
This work makes five contributions:
•Typed artifact market.A FedFM data market that
treats heterogeneous artifacts as first-class contribu-
tion units (Section 3).
•Pipeline-ordered valuation.A serving-aware credit
rule that we prove is theuniquevalue satisfying
standard axioms plus a serving-causal downstream-
realization axiom, routing each coalition’s comple-
mentarity dividend to its serving frontier (Theo-
rem 1; Section 3).
•Scalable valuation (S3Val).A scalable, privacy-
constrained valuation algorithm combining contri-
bution sketches, stratified coalition sampling, sur-
rogate modeling, and uncertainty-triggered audits
(Section 5).
•Risk-adjusted payments.A budget-feasible pay-
ment rule on lower-confidence-bound values with
penalties for cost, privacy, duplication, and manip-
ulation risk (Section 6).•Benchmark and system.1A deployed-style architec-
ture, benchmark, and stress-test suite covering sybil,
duplicate, poison, non-IID-suppression, privacy-
gaming, and cost-inflation attacks (Sections 4, 7, 8).
Pipeline-ordered credit.FedMark-FM supports both
unordered S3Val and an ordered variant that restricts
marginal-credit permutations to the contract-visible serving
precedence, so a retrieval client is credited for improving
downstream context while an adapter is credited condi-
tional on the retrieval and prompt state that precedes it
at serving time. Empirically, ordered valuation changes se-
lected sets on FEVER (Spearman 0.7554, selected-set overlap
0.6667; risk-adjusted utility 0.2410 unordered vs. 0.2425 or-
dered). A controlled held-out serving study on real FEVER
data (supplementary material,2Table S15) shows that this
redistribution isserving-neutral: it changes which clients
are credited so that payment respects serving causality,
while leaving held-out task accuracy statistically unchanged
(∆ = +0.004±0.014). Ordered valuation is therefore a
payment-fairness choice rather than an accuracy lever or an
implementation detail.
Significance.The contribution is fundamentally a data-
management and mechanism-design one that matters wher-
ever private contributions are priced under uncertainty.
A market round begins with a registry that binds each
private artifact to typed metadata, provenance commit-
ments, privacy limits, and evaluation endpoints. Coalition
evaluations then populate a value index with marginal-
value estimates, uncertainty, risk flags, and drift evidence.
Payments are written to a ledger with formula hashes,
budget scaling, and evidence pointers, after which audit
logs support disputes and revaluation when the model,
retriever, or policy changes. The mechanism-design layer is
therefore not isolated game theory; it is the pricing logic in-
side a deployment-style data system for registering, valuing,
paying, and auditing foundation-model adaptation artifacts.
Paper organization.Section 2 surveys related work. Sec-
tion 3 formalizes the FedFM market and pipeline-ordered
valuation, including its uniqueness characterization. Sec-
tion 4 presents the system architecture, Section 5 the S3Val
estimator, and Section 6 the risk-adjusted payment mech-
anism and its guarantees. Section 7 introduces FedMark-
FM-Bench, and Section 8 reports the real-data evaluation.
Section 9 discusses limitations and scope, and Section 10
concludes.
2 RELATEDWORK
2.1 Federated Learning Incentives
Federated learning enables collaborative model training
without centralizing raw data, but its practical adoption
depends on incentives for participation, computation, com-
munication, and risk sharing [1]. Shapley-based reward
allocation is widely studied because it satisfies fairness
1. Source code, benchmark harness, data, and reproduction scripts:
https://anonymous.4open.science/r/FedMark-FM-3A89.
2. The supplementary material—appendices covering registry and
ledger schemas, notation, extended cost analysis, differential-privacy
and attestation details, threat models, and full experimental protocols—
is available with this submission. Its sections, tables, and figures are
numbered with an “S” prefix.

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 3
axioms in cooperative games [6], [2]. Other work stud-
ies adaptive contribution scoring, seller selection, auctions,
Stackelberg games, and collaborative fairness in FL mar-
kets [7], [8], [9]. Classical VCG-style mechanisms and peer-
prediction mechanisms offer stronger truthfulness results
under restrictive observability and reporting assumptions
[10], [11], [12], [13]. These mechanisms assume observabil-
ity and reporting conditions that FedFM artifacts do not
meet: artifacts are privacy-constrained, non-verifiable in raw
form, and pipeline-dependent. FedMark-FM therefore uses
auditable approximate valuation with explicit uncertainty
and dispute logs, and treats VCG as a full-observability
reference point: VCG attains higher utility when artifacts
and utilities are fully observable, whereas FedMark-FM is
built for the privacy-constrained case where raw observ-
ability is unavailable. Closest in spirit are privacy-aware
FL auctions such asFL-Market, which trade models under
local differential privacy with optimal aggregation [14], and
sybil-poisoning defenses such asFoolsGold, which down-
weight low-diversity gradient updates [15]; both, however,
target gradient or dataset providers under a supervised
objective, whereas FedMark-FM pricestypedFM artifacts
under pipeline causality and can host a gradient-diversity
defense inside its update-sketch track rather than replace
it. These approaches provide important foundations, but
they typically treat each client as contributing a dataset or
update under a relatively fixed supervised-learning objec-
tive. FedFMs introduce richer artifact types and pipeline
dependencies.
2.2 Data Valuation
Data valuation estimates the contribution of examples,
sources, or clients to a model or task. Data Shapley values
allocate utility by averaging marginal contributions over
coalitions [3]. Efficient variants exploit nearest-neighbor
structure, gradient similarity, influence approximations, or
surrogate models [4], [16]. DVRL and noise-reduced Shap-
ley variants provide additional peer-reviewed baselines for
non-market valuation [17], [18]. Recent foundation-model
valuation work studies document credit in LLM sum-
maries and data auctions for RAG [19], [20]. In foundation-
model applications, valuation must also handle retrieval,
in-context learning, fine-tuning, preference optimization,
safety data, and unlearning audits. FedMark-FM builds on
Shapley-style marginal value but treats valuation as an
operational layer over heterogeneous artifacts and market
logs.
2.3 Federated Foundation Models
Federated foundation models combine the general ca-
pabilities of foundation models with privacy-preserving
multi-client collaboration. Recent surveys and challenge
papers identify private data use, non-IID heterogeneity,
bidirectional knowledge transfer, incentives, game mech-
anisms, privacy, security, watermarking, and efficiency as
open problems [5]. FM-enabled cross-silo incentive work
has begun to study knowledge hoarding, free-riding, and
Stackelberg-style compensation [9]. This paper targets the
data-engineering part of that agenda: how to price hetero-
geneous artifacts that improve a shared foundation-model
pipeline without centralizing raw private data.2.4 Foundation-Model Adaptation Artifacts
Modern foundation-model systems can be adapted through
retrieval-augmented generation (RAG), prompt engineer-
ing, in-context demonstrations, LoRA or other parameter-
efficient adapters, preference data, safety data, and tool
traces [21], [22], [23]. Federated PEFT and LoRA studies
show that adapter rank, aggregation, initialization, robust-
ness, privacy noise, and heterogeneity matter in collabo-
rative foundation-model tuning [24], [25], [26], [27], [28],
[29], [30]. Adapter merging and routing policies can change
coalition interference, so our framework includes proxy
comparisons for centroid merging, volume routing, risk-
aware routing, orthogonal-dropout-style merging, and hier-
archical merging. These works primarily improve aggrega-
tion or global-model accuracy. FedMark-FM is complemen-
tary: a secure evaluator can run any of these aggregation
rules, while the market layer values the submitted adapters,
charges for privacy/cost/risk, and records payment evi-
dence.
2.5 Collaborative RAG and Federated Seller Measure-
ment
Collaborative RAG studies show that shared passage stores
can improve low-resource clients, but that irrelevant or
hard-negative passages affect both performance and partic-
ipation incentives [31]. Privacy-preserving federated RAG
such asFedE4RAGlearns retriever embeddings under pri-
vacy constraints [32]; FedMark-FM is compatible with such
systems, treating a federated retriever as an upstream arti-
fact to be valued and paid. Decentralized data-market work
on federated data measurements ranks sellers with rele-
vance and diversity signals without training task-specific
models [33]. FedMark-FM treats these measurements as
useful priors rather than replacements for valuation: rele-
vance/diversity sketches initialize redundancy clusters and
sampling priorities, while secure coalition evaluation still
determines auditable payments under duplicate, poison,
and scarcity adjustments.
2.6 Structure-Aware Valuation and Secure Incentive
Systems
Recent structure-aware valuation work argues that clas-
sical Shapley symmetry can be inappropriate when data
sources enter ordered pipelines. Asymmetric Data Shapley
relaxes symmetry to respect group or temporal precedence
[34], [35], and precedence-constrained Winter values extend
constrained-permutation valuation to graph dependencies
[36]; our pipeline-ordered value is a serving-pipeline in-
stance of this broader family of constrained-permutation
valuations. This is directly relevant to foundation-model
systems, where retrieval corpora, prompts, adapters, and
preference/safety data may be evaluated in an ordered serv-
ing path. Efficient valuation work such as Owen-sampling-
based FL contribution estimation and data-free spectral-
entropy metrics provide low-cost alternatives or priors for
S3Val’s sampling and sketch layers [37], [38]. Secure system
work such as C-FedRAG focuses on confidential federated
retrieval, while FWeb3 focuses on incentive-aware FL set-
tlement with Web3 support services [39], [40]. FedMark-
FM is complementary: it supplies typed artifact valuation,

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 4
TABLE 1
FedFM market contribution types.
Type Example Evaluation interface
Retrieval
corpusPrivate passages Federated retriever
Adapter LoRA checkpoint Adapter rout-
ing/merge
Prompt Template Contract prompt
probes
Demonstration ICL examples Context selection
Preference data Pairwise labels Reward/alignment
probes
Safety data Red-team cases Hidden safety tests
Update sketch FL gradient/up-
dateSecure aggregation
FedFM
marketretrieval
corpusprompts
demosLoRA
adapter
preference
datasafety
casesupdate
sketchsecure
evaluationvalue, risk,
paymenttyped registry evidence
Fig. 1. Heterogeneous contribution taxonomy: RAG corpora, prompts,
demonstrations, adapters, preference data, safety cases, and update
sketches as typed market artifacts.
risk-adjusted payments, DP/enclave evidence, and dispute
ledgers that can sit above confidential retrieval or settlement
substrates.
Table S8 summarizes the gap. The closest prior lines
each solve part of the problem, but FedMark-FM treats the
market itself as a deployable data-management layer for
foundation-model artifacts.
3 PROBLEMFORMULATION
3.1 Market Participants
The market contains a set of clientsN={1, . . . , n}, a
market operator, downstream consumers, and optionally
an external auditor. Each clientiowns a private portfolio
Πi={z i1, . . . , z iki}. A contributionzis a typed artifact:
z= (type, payload, metadata, policy).
The payload may remain private. The operator may
observe signed hashes, provenance commitments, differen-
tially private summaries, adapter fingerprints, secure eval-
uation outputs, or local evaluation certificates. The operator
should not need raw private corpora, private test sets, or
sensitive user records.
3.2 Contribution Types
Table 1 summarizes the main contribution classes, and Fig-
ure 1 shows the corresponding heterogeneous contribution
taxonomy. A full deployment binds each artifact type to a
separate secure evaluation interface.
3.3 Contract Utility
For a coalitionC⊆N, letΠ(C) =∪ i∈CΠi. The market
utility is:U(C) =w qQ(C) +w sS(C) +w rR(C)
−wlL(C)−w pΦ(C)−w kK(C),(1)
whereQis task quality,Sis safety compliance,Ris
robustness,Lis latency or serving overhead,Φis privacy-
budget consumption or leakage risk, andKis financial/-
compute cost. The weights are contract parameters and
must be reported as part of the benchmark card.
3.4 Contribution Value
The ideal client value is the Shapley-style marginal contri-
bution:
ϕi=Eω[U(Pre ω(i)∪ {i})−U(Pre ω(i))],(2)
whereωis a random client ordering andPre ω(i)are
clients beforei. Artifact-level values are defined analogously
for eachz ij. In practice, direct computation is infeasible be-
cause it requires many RAG indexes, adapter compositions,
prompt evaluations, or secure aggregation rounds.
3.5 Pipeline-Ordered Contribution Value
The exchangeability implicit in Eq. (2) is often wrong for
foundation-model serving. A RAG passage is upstream of a
prompt, an adapter acts after retrieved context is selected,
and preference or safety data may only matter after the
answer policy is invoked. We therefore define a pipeline-
ordered value,ϕord
i, by restricting permutations to respect a
partial order over artifact groups:
G: retrieval≺prompt/demo
≺adapter≺preference/safety.
Letg(i)∈ Gbe the pipeline group of clienti. LetΩ Gbe the
set of client permutations in which all clients from earlier
groups appear before clients from later groups; clients inside
the same group remain randomly ordered. The ordered
value is
ϕord
i=Eω∼ΩGh
U(PreG
ω(i)∪ {i})−U(PreG
ω(i))i
.(3)
This formulation is an Asymmetric-Data-Shapley-style re-
laxation of symmetry for FedFM markets: it preserves
marginal-credit accounting but uses the serving path as
a contract-visible precedence constraint. The practical im-
plication is simple. A retrieval client is credited for im-
proving the context available to downstream prompts and
adapters, while an adapter is credited conditional on the
retrieval/prompt state that would actually precede it at
serving time. The contract card records whether a market
round uses unordered S3Val or pipeline-ordered S3Val, and
the ledger stores the group order so disputed payments can
be replayed.
3.6 Axiomatic Characterization of Ordered Valuation
Equation (3) is one way to respect serving order; we now
show it is theprincipledone. We characterizeϕordas the
unique credit rule satisfying four standard value axioms
plus one market axiom that encodes serving causality. This
turns pipeline-ordered valuation from a heuristic relaxation

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 5
into the Shapley value for the layered precedence structure
induced byG, in the sense of games under precedence
(permission) constraints [41], [42].
Apipeline gameis a tuple(N, U,G)withU(∅) = 0and
G= (G 1≺ ··· ≺G L)an ordered partition ofNinto serving
layers, whereg(i)is the layer index of clienti. Avalueψ
maps each pipeline game to a payoff vector(ψ i)i∈N. For
∅ ̸=T⊆N, the unanimity (pure-complementarity) game is
uT(S) =1[T⊆S], andℓ(T) = max{g(k) :k∈T}is the
most downstream layer thatTreaches. We impose, on the
class of pipeline games with fixedG:
(A1)Linearity.ψ(αU+βV) =αψ(U) +βψ(V).
(A2)Null player.IfU(S∪ {i}) =U(S)for allS, then
ψi(U) = 0.
(A3)Efficiency.P
i∈Nψi(U) =U(N).
(A4)Within-layer symmetry.Ifi, j∈G aandU(S∪
{i}) =U(S∪ {j})for allS⊆N\ {i, j}, then
ψi(U) =ψ j(U).
(A5)Downstream realization.For everyT, any member
i∈Twithg(i)< ℓ(T)receives none ofT’s pure
complementarity:ψ i(uT) = 0.
Axiom (A5) is the serving-causal content: when a coali-
tion’s value is realized only once all members are assembled
and served, the marginal is emitted at the serving frontier
Gℓ(T), so an upstream member that shares that synergy with
a strictly-downstream partner draws no credit from it. The
symmetric Shapley value violates (A5); it splits every com-
plementarity dividend equally across all ofT. We state (A5)
as a deliberate contract-leveldesignchoice about serving-
time attribution—the market elects to credit synergy where
it is realized—rather than a claim that upstream inputs are
normatively worthless; an operator who prefers to reward
necessary upstream inputs can simply run unordered S3Val,
and the contract card records which rule is in force.
Theorem 1(Serving-order characterization).On pipeline
games with layered precedenceG,ϕordis the unique value
satisfying (A1)–(A5). Moreover it routes each coalition’s Harsanyi
dividend to that coalition’s most downstream members, split
equally:
ϕord
i(uT) =(
1/|T∩G ℓ(T)|, i∈T∩G ℓ(T),
0,otherwise.
Proof:ϕordsatisfies the axioms.Marginal contributions
are linear inUand averaging overΩ Gpreserves linearity,
giving (A1). A null player has zero marginal in every order,
giving (A2). Each order telescopes,P
i[U(PreG
ω(i)∪ {i})−
U(PreG
ω(i))] =U(N)−U(∅) =U(N), and averaging
preserves the sum, giving (A3). Transposing two clients in
the same layer is a measure-preserving bijection ofΩ G, so
symmetric same-layer clients receive equal value, giving
(A4). Finally, inu Ta clientihas marginal1in orderωiff
i∈Tand every other member ofTprecedesi, i.e.iis
the last member ofTinω; underΩ Gthat last member lies
inG ℓ(T) and is uniform overT∩G ℓ(T), which yields the
displayed formula and in particular (A5).
Uniqueness.The unanimity games{u T}∅̸=T⊆N form a
basis of the space of games withU(∅) = 0; writeU=P
TcTuT. By (A1),ψ i(U) =P
TcTψi(uT), soψis fixed
once{ψ(u T)}is fixed. Take anyT. Clientsi /∈TareTABLE 2
Worked example: symmetric versus ordered credit for a
retrieval≺adapter game.
Client (layer) Symmetric Shapley Ordered value
r1(retrieval) 2.0 1.5
r2(retrieval) 2.0 1.5
a(adapter) 2.0 3.0
Total 6.0 6.0
null inu T, so (A2) givesψ i(uT) = 0; clientsi∈Twith
g(i)< ℓ(T)giveψ i(uT) = 0by (A5). The remaining clients
lie inT∩G ℓ(T), share the layerG ℓ(T), and are mutually
symmetric inu T, so (A4) equates their values to a common
s. Efficiency (A3) forces|T∩G ℓ(T)|s=u T(N) = 1, hence
s= 1/|T∩G ℓ(T)|. Thusψ(u T) =ϕord(uT)for everyT, and
by linearityψ=ϕord.
Replacing (A4)–(A5) by full symmetry recovers the clas-
sical Shapley value, which splits eachc Tequally over all
ofT; ordered valuation instead routesc Tto the serving
frontier. Two consequences matter for the market. First,
because Theorem 1 fixes the value on the unanimity basis,
contracts that agree onGyield identical ordered credit,
and the budget-feasible scaling of the payment rule in
Eq. (5) multiplies all positive values by a common factor,
preserving the within-layer order the theorem induces. Sec-
ond, combined with the held-out serving-neutrality result
(Table S15), ordered valuation rests on two pillars: it is the
uniqueserving-causal credit rule, and it costs no measurable
held-out task quality. It is therefore a principled payment-
fairness guarantee rather than an accuracy heuristic.
Worked example.Consider two retrieval clientsr 1, r2in
layerG 1and one adapter clientain layerG 2, withU(∅) = 0,
U({r 1}) =U({r 2}) = 2,U({a}) = 0,U({r 1, r2}) = 3,
U({r 1, a}) =U({r 2, a}) = 5, andU(N) = 6: the adapter
is useless without retrieved context but adds3once either
retrieval client is present. The nonzero Harsanyi dividends
arec{ri}= 2,c {r1,r2}=−1,c {ri,a}= 3, andc N=−3.
The symmetric Shapley value splits each dividend across
all of its members and pays(2,2,2). Ordered valuation in-
stead routes the two cross-layer complementarity dividends
c{ri,a}entirely to the serving frontiera, giving(1.5,1.5,3)
(Table 2): the adapter that realizes the retrieval–adapter
synergy at serving time is paid for it, while the retrieval
clients keep only their standalone and same-layer credit.
Both rules are efficient (P
iϕi= 6); they differ precisely
on who is paid for cross-layer complementarity. Averaging
the two serving-order permutations(r 1, r2, a)and(r 2, r1, a)
in Eq. (3) reproduces(1.5,1.5,3), confirming the dividend-
routing form of Theorem 1.
Beyond attribution, (A5) has a strategic consequence that
distinguishes ordered from symmetric credit.
Corollary 1(No upstream synergy capture).Underϕord,
clienti’s value draws only on coalitions whose deepest layer is
i’s own:ϕord
i(U) =P
T:i∈T, ℓ(T)=g(i) cT/|T∩G g(i)|. Hence
ireceives nothing from any coalition that contains a strictly
more downstream member; a positive cross-layer complementarity
dividendc Twithℓ(T)> g(i)paysia sharec T/|T|under the
symmetric Shapley value but0underϕord.

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 6
Proof:By linearity,ϕord
i(U) =P
TcTϕord
i(uT); The-
orem 1 makesϕord
i(uT)nonzero only wheng(i) =ℓ(T),
which gives the displayed sum and the vanishing of every
term withℓ(T)> g(i).
Two operational consequences follow. First, an up-
stream provider cannot raise its ordered payment by enter-
ing or fabricating cross-layer complementarity, so identity-
splitting or collusion aimed at harvesting downstream syn-
ergy is unprofitable under ordered credit even before the
duplicate penalty of Section 6 is applied. Second, credit—
and therefore the manipulation surface—concentrates on
the serving frontier, which tells the operator to prioritize
duplicate and poison audits on the most downstream layer
a disputed coalition reaches. A full incentive-compatibility
analysis of frontier-concentrated credit is left to future work.
3.7 Strategic Clients
Clients may split one portfolio across identities, duplicate
another client’s artifacts, submit poisoned adapters, overfit
public prompts, inflate costs, exaggerate privacy constraints,
or collude to create artificial complementarity. FedMark-
FM targets approximate manipulation resistance under a
bounded strategic model: common manipulations should
not produce higher expected profit than honest participa-
tion after duplicate penalties, uncertainty discounts, hidden
audits, and risk-adjusted payments. This is the appropriate
guarantee for privacy-constrained artifacts, whose raw form
cannot be verified to support exact dominant-strategy truth-
fulness.
4 THEFEDMARK-FM SYSTEM
Figure 2 shows the deployed-style architecture, and Fig-
ure 3 shows the dataflow within a single market round.
The design keeps registry entries, value estimates, pay-
ment decisions, and cryptographic evidence as explicit data-
engineering artifacts across client, operator, and auditor
trust boundaries. The contract card is committed before
submissions, and later disputes replay the recorded hashes,
utility weights, evaluator evidence, and payment formula.
4.1 Contribution Registry
The registry stores client credentials, artifact type, domain
tags, signed hashes, license constraints, privacy budget,
declared cost, and artifact endpoint. The registry is not
only a database; it defines what can be evaluated and what
evidence can be used in disputes.
4.2 Secure Evaluation Sandbox
The sandbox evaluates sampled coalitions. For RAG, it
builds temporary indexes or queries federated retriever
endpoints. For adapters, it loads or routes LoRA modules in
isolated workers. For private probes, it accepts signed client-
side score certificates. For update sketches, it uses secure
aggregation outputs. Each evaluation produces an evidence
hash and utility vector.4.3 Value Index and Ledger
The value index stores point estimates, confidence intervals,
redundancy clusters, complementarity links, risk flags, and
historical value drift. The market ledger stores the payment
formula version, utility weights, lower confidence bounds,
penalties, bonuses, final payment, and evidence hashes.
This is essential for dispute resolution and revaluation after
model, prompt, retriever, or policy changes.
5 S3VAL: SECURESURROGATESHAPLEYVALU-
ATION
5.1 Overview
S3Val estimates either the unordered value in Eq. (2) or the
pipeline-ordered value in Eq. (3). The same evidence path
is used in both cases; the ordered variant changes only the
admissible coalition permutations. The estimator has four
components:
1) contribution sketches;
2) redundancy and complementarity clustering;
3) stratified or pipeline-ordered coalition sampling;
4) surrogate utility modeling with uncertainty-
triggered direct audits.
5.2 Contribution Sketches
Each client submits typed metadata and privacy-preserving
summaries. A retrieval corpus may submit centroid embed-
dings, MinHash sketches, domain histograms, and prove-
nance hashes. An adapter may submit rank, target modules,
delta norms, calibration traces, and routing hints. A prompt
contribution may submit task tags and response statistics.
These sketches provide features for valuation while limiting
raw data exposure.
5.3 Coalition Sampling
Coalitions are sampled by artifact type, domain, redun-
dancy cluster, and risk stratum. The sampler oversamples
clients whose value confidence interval crosses zero, whose
payment would be large, or whose manipulation risk is
high. This matters because a market does not need uni-
formly precise values for every client; it needs reliable
values near payment decisions.
5.4 Utility Surrogate
The surrogateg θ(C, i)predicts:
∆i(C) =U(C∪ {i})−U(C).(4)
Features include coalition size, artifact mix, domain
gaps, duplicate overlap, adapter interference, safety risk,
privacy budget, cost, and scarcity. The prototype uses a
dependency-free linear surrogate to keep experiments re-
producible; the framework can replace it with calibrated
gradient-boosted trees, Gaussian processes, or neural mod-
els. Algorithm 1 summarizes the full S3Val procedure.
With fixed strata and bounded marginal variance, a
client receivingm idirect marginal observations has stan-
dard errorO(1/√mi). Thus the LCB width shrinks at the

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 7
Fig. 2. Architecture of FedMark-FM. Solid arrows are data/control flow; dashed arrows are audit/dispute flow.
Fig. 3. Single market-round dataflow, from contract-card commitment
through settlement and dispute replay.
same rate until the stopping rule is met or the call budget is
exhausted. The audit predicate is explicit:
A(i, C, H) =1{w i> τw∨ri> τr∨bi> τb∨ξi< hmin},
whereb iis the distance of clienti’s current scaled payment
from the budget threshold,ξ i∼Uniform(0,1), andh min
is the configured random direct-audit floor. The direct-
evaluation cost is2|D| ≤2Mcoalition utility calls, plus
sketch clustering and surrogate updates; forkstrata, a bal-
anced allocation givesO(M/k)direct samples per stratum.
6 PAYMENTMECHANISM
6.1 Lower-Bound Payment Rule
Letbϕibe the estimated value,σ iits calibrated uncertainty,d i
duplicate risk,r imanipulation risk,c iverified or declared
cost,ϵpriv
i privacy budget, ands iscarcity score. The raw
payment is the positive part of a single contract score:
epi=h
bϕi−λσ i−βc i−γϵpriv
i−ηmax(d i, ri) +ρs ii
+.
(5)Hereλis the uncertainty discount,βthe cost penalty,γthe
privacy-consumption penalty,ηthe risk penalty, andρthe
scarcity bonus weight.
Raw payments epiare scaled if their total exceeds the
market budget. Lower-confidence-bound payment is inten-
tional: a high-variance client should not receive a large
payment before audit evidence supports the value. Default
values areλ= 0.75,β= 0.28,γ= 0.20,η= 0.75, and
ρ= 0.25; Table S11 lists these defaults, and real-track scripts
sweep risk, duplicate, privacy, cost, and scarcity settings and
record the selected operating point in the contract card.
Operational scores.In the prototype,d iis computed
from provenance hash matches when available and other-
wise from token/trigram overlap for retrieval passages or
train-text signatures for adapter proxies. The manipulation
scorer iis an audit prior built from hidden-probe failure,
duplicate-cluster membership, provenance/timing anoma-
lies, declared-cost outliers, and privacy declarations that
reduce observability without improving validation utility.
The scarcity scores iis not a flat entitlement: it is positive
only when a domain or artifact type is underrepresented
on the contract validation card and the client improves that
slice. The privacy termϵpriv
i is the measured DP spend or
declared privacy-constrained evaluation budget. The coef-
ficientγis only the unit conversion from that spend to
a payment penalty; it is not itself a privacy budget. The
prototype sweepsγfor Laplace and zCDP-style Gaussian
releases to test payment stability under different operator
cost assumptions.
Costc ishould be verified when it affects payment.
The deployment interface accepts signed evaluator receipts:
wall-clock time, accelerator type, token count, retrieval calls,
adapter-train steps, DP releases, and enclave or worker
measurement. If receipts are unavailable, the prototype
treats cost as adversarially declared and applies reserve

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 8
Algorithm 1:S3Val
Input:ClientsN, utilityU, strataS, call budgetM,
width targetτ, stability windowT, audit
predicateA(i, C, H)
Output:Value ledger
1Collect contribution sketches and provenance
commitments
2Build redundancy and domain clusters
3Initialize direct setD← ∅, surrogateg θ, widths
wi← ∞
4while|D|< Mandmax iwi> τand top-kpayments
are not stable forTupdatesdo
5Choose stratums∈ Swith probability
proportional to1 + ¯w s+ ¯rs+¯bs
6Sample clientifromsproportional to
1 +w i+ri+bi, whereb iis payment-boundary
proximity
7Sample coalitionC⊆N\ {i}from
same/different domain and redundancy strata
8ifA(i, C, H) = 1, whereHis the current ledger
historythen
9Evaluate∆ i(C) =U(C∪ {i})−U(C);
append(x(C, i),∆ i(C))toD
10Updateg θby ridge regression onD; update
residual and marginal-variance widthsw i
11foreachclientido
12Estimate bϕifrom direct and surrogate marginals
13Compute uncertainty, lower confidence bound,
duplicate risk, and manipulation risk
14returnvalue ledger
checks plus a cost-gaming stress test. In that test, a client
jointly inflates cost by1×–10×and submits a duplicate;
the attacker is not selected under the current contract be-
cause higher declared cost lowers value density and dupli-
cate/risk penalties dominate.
6.2 Manipulation Defenses
Sybil splitting and duplicate submission are handled by
redundancy clusters and group-level marginal discounts.
Poisoned adapters are penalized through hidden safety
probes and manipulation-risk scores. Privacy-budget gam-
ing increases uncertainty and therefore lowers the lower
confidence bound. Cost inflation is limited by cost penalties
and reserve checks. Non-IID specialist suppression is ad-
dressed through stratified evaluation and scarcity bonuses.
6.3 Theoretical Properties
We state the mechanism’s guarantees informally here and
defer formal statements and proofs to the supplementary
material. S3Val admits a Hoeffding-style estimation-error
bound; under a value-gap margin condition, budgeted se-
lection recovers the downstream-oracle set with high prob-
ability; budget feasibility and payment monotonicity hold
by construction; and approximate incentive-compatibility,
sybil, poisoning, and individual-rationality bounds quantify
when common manipulations are unprofitable.7 FEDMARK-FM-BENCH
7.1 Tracks
FedMark-FM-Bench contains four tracks:
•Federated retrieval corpus market: clients contribute
private passages for RAG.
•Federated adapter market: clients contribute LoRA
or adapter-effect artifacts.
•Prompt and demonstration market: clients contribute
prompt templates or in-context examples.
•Preference and safety market: clients contribute pair-
wise labels and red-team cases.
7.2 Client Types
The benchmark includes high-quality specialists, redundant
generalists, noisy contributors, sybil splitters, duplicate sub-
mitters, poisoned adapter clients, privacy-sensitive clients,
cost inflaters, and rare-domain contributors.
7.3 Metrics
Valuation metrics include rank correlation with expensive
leave-one-client-out, top-khigh-value precision, harmful-
client AUROC, duplicate discount ratio, value sign stability,
and confidence interval coverage. Market metrics include
utility per dollar, budget violation rate, individual rational-
ity rate, diversity of selected domains, and regret against
oracle selection. Strategic metrics include sybil gain ratio,
duplicate overpayment, poisoning profit, prompt-gaming
gain, privacy-gaming gain, and cost-inflation gain.
7.4 Submission and Leaderboard Protocol
To make FedMark-FM-Bench more than a one-off re-
production script, a benchmark submission consists of
a signed client manifest, typed artifact descriptors, op-
tional sketches, and a score file. Each manifest records
round_id,client_id,artifact_type,base_model,
provenance_hash,privacy_budget,declared_cost,
endpoint_ref, andsignature. Raw private payloads
are not submitted to the leaderboard. Held-out evaluation
uses a contract card whose validation-slice hashes and audit
policy are committed before submissions; test labels, hidden
probes, and rare-slice membership are not exposed until the
round closes. A leaderboard row reports per-track raw util-
ity, risk-adjusted utility, strategic clients selected, rare clients
retained, audit calls, model calls, wall-clock time, budget
use, and dispute count. FedMark-FM-Bench includes JSON
schemas and toy clients for specialist, duplicate, sybil,
poison, privacy-gaming, and cost-inflation submissions so
outside researchers can test new market mechanisms against
the same interface. Figure 4 shows the resulting submission
flow.
A runnable, extensible leaderboard.To lower the bar-
rier to entry, FedMark-FM-Bench ships as a small library and
a one-command harness. An external method is ascorerthat
returns a contribution score per client, given a committed
contract card and a validation-utility oracle that exposes the
same secure-evaluation interface every baseline uses. The
harness selects a budget-feasible coalition from the returned
scores, serves it on a disjoint held-out test card, and appends

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 9
TABLE 3
FedMark-FM-Bench leaderboard on a frozen 100-client instance:
held-out accuracy and operator economics at a fixed budget.
Method Acc. Util/$ Rare kept Strat. sel. Calls
FedMark-FM 0.550 0.126 1.00 0.00 351
Equal 0.517 0.116 1.00 0.00 0
Shapley-UCB [7] 0.517 0.117 1.00 0.33 351
Retrieval similarity [33] 0.442 0.100 0.67 0.33 0
Leave-one-out [16] 0.108 0.025 0.33 10.67 117
Volume 0.092 0.020 0.00 12.00 0
FL-Shapley [3] 0.075 0.017 0.00 4.67 351
a standardized row; a JSON submission schema and the
frozen contract-card hash make results comparable and
commit-reveal-checkable. Table 3 shows the leaderboard
pre-populated with the seven baselines on a frozen 100-
client instance (contract72a9f6c9). Beyond task quality
it reports operator economics at a fixed budget: FedMark-
FM attains the best held-out accuracy and utility-per-dollar
while retaining the rare specialist and selecting no strategic
clients, whereas volume and leave-one-out spend the same
budget on high-volume poison (10–12 strategic clients) and
collapse. Because selection and serving are decoupled by the
contract card, a new valuation method is scored against the
identical instance in a single call.
8 REAL-DATAVALIDATION
8.1 Tracks and Baselines
We evaluate on three real-data tracks. The first is a FEVER
retrieval-corpus market: clients own private shards of ev-
idence passages and are evaluated by whether a coalition
retrieves the gold evidence for held-out claims. The second
is a de-circularized FEVER serving track: each method se-
lects a paid coalition using a contract-validation card, then
the selected coalition is actually served on a separate held-
out test card through retrieve–read scoring. The headline
metrics are downstream accuracy, macro-F1, evidence exact
match, and regret against a downstream oracle; none of
these metrics use attack labels or the payment formula.
The third is a trained low-rank adapter validation: each AG
News client trains a small PyTorch low-rank adapter, and
coalitions are evaluated by summing adapter logits. These
tracks exercise real text, real labels, strategic clients, and the
major requested baselines.
We also add one optional generator-backed RAG serving
track. It usesgoogle/flan-t5-smallas a real seq2seq
reader after coalition selection: the market selects clients
on a validation card, retrieves held-out evidence from the
paid coalition, and prompts the generator to answer the
FEVER claim as yes/no/unknown, which is mapped back
to SUPPORTS/REFUTES/NOT ENOUGH INFO. The attack
is deliberately potent: a high-volume poison corpus is useful
on the validation card but injects wrong-answer instructions
into held-out evidence. This is a realistic prompt-injection
threat for RAG systems; FedMark-FM does not use attack-
specific rules for it, only the same generic risk, duplicate,
uncertainty, and audit scoring used elsewhere. This track is
not a large-generator benchmark, but it tests whether the
market advantage survives a real generated answer rather
than only retrieval scoring.TABLE 4
Generator-backed held-out FEVER RAG using
google/flan-t5-small. Values are mean±95% CI.
Method Acc. Macro-F1 Poison Strategic
Volume 0.3187±0.0531 0.2362±0.0428 1.0 1.0
Leave-one-out 0.3125±0.0490 0.2269±0.0411 1.0 3.6
FL-Shapley 0.3125±0.0624 0.2102±0.0629 0.8 2.6
FedMark-FM 0.3937±0.0219 0.2718±0.0198 0.0 0.0
We compare equal payment, volume, leave-one-out,
sampled FL-Shapley, retrieval similarity, Shapley-UCB, and
FedMark-FM where applicable. Each rule ranks clients, then
selects a budget-feasible subset. Utility includes task per-
formance minus cost, privacy, redundancy, and risk terms.
Risk-adjusted utility additionally penalizes selected strate-
gic clients and rewards retained rare specialists, so we report
it as an operator composite rather than an independent ac-
curacy metric. Tables therefore keep raw utility, downstream
task metrics, strategic selections, rare retention, AUC, and
audit/call counts separate wherever possible; the alpha-
sweep stress test shows when the composite ordering flips.
8.2 De-Circularized Held-Out Serving
Table 4 is the primary serving experiment, and it de-
circularizes evaluation by construction: after selecting coali-
tions on a validation card, the paid coalition is served
on a disjoint test card with a real seq2seq reader. Un-
der a high-volume held-out prompt-injection poisoner,
FedMark-FM improves downstream accuracy by 7.5–
8.1 percentage points over Volume, leave-one-out, and
FL-Shapley, with paired 95% CIs that remain positive
(0.0812±0.0477vs FL-Shapley,0.0813±0.0372vs leave-one-
out, and0.0750±0.0372vs volume). The weaker retrieve–
read proxy, moved to Appendix Table S21, has the same
direction of effect but is near the three-class chance floor:
FedMark-FM has the highest mean accuracy and macro-F1,
but several CIs overlap.
8.3 Market Utility, Ordered Credit, and Adapter Evi-
dence
The FEVER RAG market illustrates the failure FedMark-
FM targets: Equal, volume, retrieval-similarity, leave-one-
out, FL-Shapley, and Shapley-UCB all select on average
two strategic clients under the configured budget, while
FedMark-FM selects none and reaches the honest-client
utility reference (harmful-client AUC is only a constructed-
attack diagnostic; the robustness case rests on held-out
serving, strategic counts, and ablations). The second load-
bearing result is that ordered S3Val changes rank and
selected-set evidence enough to be a contract choice, and
the controlled study of Table S15 shows the change is
serving-neutral—a payment-fairness choice, not an accuracy
lever. On the trained low-rank LoRA validation, FedMark-
FM obtains the best utility and risk-adjusted utility while
retaining the rare adapter and selecting no strategic clients.
8.4 De-Circularized Serving at Scale
To establish serving quality at scale independently of
the market’s own objective, we run a larger, fully de-

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 10
Fig. 4. FedMark-FM-Bench submission flow.
TABLE 5
Real-data market validation. R-utility is risk-adjusted utility. Spearman is rank correlation with leave-one-out values. AUC measures harmful-client
ranking.
Track Method Utility R-utility Spearman AUC Strategic
FEVER RAG Equal 0.0219±0.0308 −0.1881±0.0308 -0.5874 0.5000 2.0
FEVER RAG Volume −0.0386±0.0308 −0.2485±0.0308 -0.0489 0.7500 2.0
FEVER RAG Leave-one-out 0.0096±0.0308 −0.2004±0.0308 1.0000 1.0000 2.0
FEVER RAG FL-Shapley 0.0096±0.0308 −0.2004±0.0308 0.8881 1.0000 2.0
FEVER RAG Retrieval similarity 0.0158±0.0393 −0.1942±0.0393 0.4405 0.7031 2.0
FEVER RAG Shapley-UCB 0.0219±0.0308 −0.1881±0.0308 0.5944 0.6719 2.0
FEVER RAG FedMark-FM 0.1321±0.0308 0.1621±0.0308 0.8112 1.0000 0.0
Trained LoRA Equal 0.2284±0.0441 0.2284±0.0441 -0.1715 0.5000 0.0
Trained LoRA Volume 0.2284±0.0441 0.2284±0.0441 -0.1715 0.2500 0.0
Trained LoRA Leave-one-out 0.2284±0.0441 0.2284±0.0441 1.0000 0.9166 0.0
Trained LoRA FedMark-FM 0.2555±0.0063 0.2856±0.0063 0.8418 1.0000 0.0
circularized end-to-end test. We build FEVER retrieval mar-
kets at 50, 100, and 200 clients, each with a rare specialist and
roughly 12% strategic clients: a high-volume poison corpus
that is validation-useful but injects flipped-label copies of
the held-out gold evidence, plus duplicate and sybil iden-
tities. Every baseline selects a budget-feasible coalition on
a validation card; the selected coalition is then served on a
disjointheld-out test card and scored only on downstream
task accuracy, macro-F1, and evidence exact match—metrics
that use neither attack labels nor the risk-adjusted-utility
formula. Table 6 reports held-out accuracy. Volume and
leave-one-out buy the high-volume poison (6, 12, and 24
poison clients at the three scales) and collapse to0.07–
0.13accuracy; sampled FL-Shapley is similarly harmed.
FedMark-FM selects zero strategic clients at every scale and
is the best or statistically tied-for-best method—its interval
overlaps Equal at 50 clients and Shapley-UCB at 100, and
it is highest at 200—while beating the strongeststandard
valuationbaseline (Volume, leave-one-out, or FL-Shapley) by
+0.26,+0.44, and+0.50as the market grows. Because the
score is a held-out task metric, this is a non-circular demon-
stration that FedMark-FM’s risk-aware selection converts
strategic-robustness signals into serving quality, and that the
margin widens with market size. Two comparisons place
the margin in context: Shapley-UCB, which also discounts
uncertain sellers, is the closest competitor, and simple Equal-
weighting performs respectably because the cost model
already embeds a risk term. The 200-client markets run in
about27s in the prototype.TABLE 6
De-circularized held-out serving at scale. Test-card accuracy (mean±
95% CI) for coalitions selected on a separate validation card; no attack
labels or risk-adjusted utility enter the score.
Method 50 clients 100 clients 200 clients
Equal 0.475±0.060 0.500±0.074 0.517±0.188
Volume 0.070±0.018 0.090±0.033 0.100±0.075
Leave-one-out 0.085±0.040 0.100±0.041 0.133±0.091
FL-Shapley 0.205±0.185 0.095±0.042 0.108±0.071
Retrieval similarity 0.420±0.048 0.465±0.070 0.483±0.145
Shapley-UCB 0.455±0.059 0.545±0.059 0.592±0.114
FedMark-FM 0.465±0.091 0.540±0.045 0.633±0.114
9 DISCUSSION
9.1 Value Drift
Contribution values depend on the base model, prompt,
retrieval stack, adapter routing, evaluation distribution, and
safety rules, so the value index must support revaluation
triggers: a client valuable for one base model may be redun-
dant or harmful for another.
9.2 Limitations and Scope
FedMark-FM occupies a specific operating point. FL-
Shapley and Shapley-UCB can exceed it on raw utility
because they optimize marginal validation utility directly,
whereas FedMark-FM prices contribution under duplicate,
sybil, poison, privacy, and audit constraints, preserves
scarce clients when the validation slice supports them,
and produces a ledger that can be disputed and reval-
ued. Its objective is therefore a practical balance between
utility, robustness, participation fairness, and auditability
rather than immediate accuracy alone. Two scope notes

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 11
TABLE 7
Security and privacy primitives.
Primitive Status What fails if missing
Signed contract
cardMandatory Rare slices, weights, and probes
can be retrofitted.
Payload hashes Mandatory Duplicate and dispute evidence
is not replayable.
Bounded DP ag-
gregate releasesMandatory for
public aggre-
gatesPublished statistics can leak
sensitive values.
Per-round privacy
accountingMandatory (ϵ, δ)spend cannot be audited.
Hidden audit
probe separationMandatory Strategic clients can overfit
public probes.
Secure aggrega-
tionMandatory
for private
update
sketchesOperator can inspect individual
updates.
Hardware TEE at-
testationMandatory for
raw-payload
evaluationSigning key has no hardware
root of trust (evidence stays
third-party verifiable in soft-
ware).
Watermark checks Optional
defenseAdaptive mimicry becomes
easier.
follow. The adapter track uses a trained, cached Hugging-
Face/PEFT LoRA path; full multi-adapter PEFT valuation
is a scale-up experiment rather than the default. And be-
cause raw artifacts are privacy-constrained, FedMark-FM
provides approximate manipulation resistance—lowering
expected gains from common attacks, exposing uncertainty,
and making high-risk payments auditable—rather than a
dominant-strategy truthfulness guarantee.
9.3 Privacy Limits
The framework reduces raw data exposure but does not
eliminate privacy risk. The prototype provides bounded DP
aggregate releases with composition accounting and third-
party-verifiable (Ed25519) signed evidence whose schema
maps field-for-field to hardware TEE attestation documents
(Table S13); payment values are protected through secure
or attested evaluation and ledger access control rather than
per-value DP . Table 7 separates the mandatory primitives
(signed contract cards, payload hashes, bounded DP re-
leases, hidden-probe separation, immutable logs, per-round
accounting) from those needed for stronger confidentiality
(secure aggregation and hardware-backed attestation), and
states what fails without each. Without hardware attesta-
tion or secure aggregation, the prototype is a reproducible
evidence path, not an end-to-end privacy proof.
10 CONCLUSION
This paper introduced FedMark-FM, an auditable, risk-
adjusted data-market framework for federated foundation-
model adaptation. The core idea is to value and pay for
heterogeneous foundation-model adaptation artifacts un-
der privacy, non-IID heterogeneity, strategic behavior, and
system constraints. FedMark-FM combines a contribution
registry, secure evaluation sandbox, unordered or pipeline-
ordered S3Val valuation engine, uncertainty-aware payment
rule, market ledger, and dispute auditor. Experiments on
FEVER retrieval, held-out RAG serving, low-rank PyTorch
adapters, and cached HF/PEFT LoRA validation show thatFedMark-FM preserves held-out task quality under adver-
sarial participation—improving downstream accuracy by
7.5–8.1 points over standard valuation baselines while se-
lecting zero strategic clients—and provides a concrete mar-
ket layer for ordered, risk-adjusted, rare-client-preserving,
and audit-ready FedFM adaptation contribution valuation.
DATA ANDCODEAVAILABILITY
The source code, benchmark harness (FedMark-FM-Bench),
datasets, and scripts required to reproduce all experiments,
tables, and figures in this paper are available at https://
anonymous.4open.science/r/FedMark-FM-3A89.
REFERENCES
[1] P . Kairouz, H. B. McMahan, B. Avent, A. Bellet, M. Bennis, A. N.
Bhagoji, K. Bonawitz, Z. Charles, G. Cormode, R. Cummingset al.,
“Advances and open problems in federated learning,”Foundations
and Trends in Machine Learning, vol. 14, no. 1–2, pp. 1–210, 2021.
[2] X. Yang, S. Xiang, C. Penget al., “Federated learning incentive
mechanism design via shapley value and pareto optimality,”Ax-
ioms, vol. 12, no. 7, p. 636, 2023.
[3] A. Ghorbani and J. Zou, “Data shapley: Equitable valuation of
data for machine learning,” inProceedings of the 36th International
Conference on Machine Learning, 2019, pp. 2242–2251.
[4] R. Jia, D. Dao, B. Wang, F. A. Hubis, N. Hynes, N. M. G ¨urel,
B. Li, C. Zhang, D. Song, and C. J. Spanos, “Efficient task-specific
data valuation for nearest neighbor algorithms,”Proceedings of the
VLDB Endowment, vol. 12, no. 11, pp. 1610–1623, 2019.
[5] T. Fan, H. Gu, X. Caoet al., “Ten challenging problems in federated
foundation models,”IEEE Transactions on Knowledge and Data
Engineering, vol. 37, no. 7, pp. 4314–4337, 2025.
[6] L. S. Shapley, “A value for n-person games,” inContributions to the
Theory of Games II. Princeton University Press, 1953, pp. 307–317.
[7] K. Chen and Z. Xu, “Federated learning for data market:
Shapley-ucb for seller selection and incentives,”arXiv preprint
arXiv:2410.09107, 2024.
[8] Z. Wanget al., “Fedave: Adaptive data value evaluation frame-
work for collaborative fairness in federated learning,”Neurocom-
puting, vol. 574, p. 127227, 2024.
[9] N. Zhang, X. Xu, X. Liu, J. Wu, and H. Tang, “Incentive mecha-
nism of foundation model enabled cross-silo federated learning,”
Scientific Reports, vol. 15, p. 24181, 2025.
[10] W. Vickrey, “Counterspeculation, auctions, and competitive sealed
tenders,”Journal of Finance, vol. 16, no. 1, pp. 8–37, 1961.
[11] E. H. Clarke, “Multipart pricing of public goods,”Public Choice,
vol. 11, pp. 17–33, 1971.
[12] T. Groves, “Incentives in teams,”Econometrica, vol. 41, no. 4, pp.
617–631, 1973.
[13] N. Miller, P . Resnick, and R. Zeckhauser, “Eliciting informative
feedback: The peer-prediction method,” inManagement Science,
vol. 51, no. 9, 2005, pp. 1359–1373.
[14] Z. Jiang, Y. Cao, Y. Wang, H. Chen, and C. Xu, “FL-Market:
Trading private models in federated learning,”arXiv preprint
arXiv:2106.04384, 2021.
[15] C. Fung, C. J. M. Yoon, and I. Beschastnikh, “Mitigating sybils
in federated learning poisoning,”arXiv preprint arXiv:1808.04866,
2018.
[16] P . W. Koh and P . Liang, “Understanding black-box predictions
via influence functions,” inProceedings of the 34th International
Conference on Machine Learning, 2017, pp. 1885–1894.
[17] J. Yoon, S. O. Arik, and T. Pfister, “Data valuation using reinforce-
ment learning,” inProceedings of the 37th International Conference
on Machine Learning (ICML), ser. Proceedings of Machine Learning
Research, vol. 119, 2020, pp. 10 842–10 851.
[18] Y. Kwon and J. Zou, “Beta shapley: A unified and noise-reduced
data valuation framework for machine learning,” inInternational
Conference on Artificial Intelligence and Statistics, 2022, pp. 8780–
8802.
[19] Z. Ye and H. Yoganarasimhan, “Fair document valuation in llm
summaries via shapley values,”arXiv preprint arXiv:2505.23842,
2025.

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 12
[20] M. Han, S. A. Esmaeili, M. Albert, and H. Xu, “Data auctions for
retrieval augmented generation,”arXiv preprint arXiv:2508.16007,
2025.
[21] P . Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschelet al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,” in
Advances in Neural Information Processing Systems, 2020.
[22] E. J. Hu, Y. Shen, P . Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang,
and W. Chen, “Lora: Low-rank adaptation of large language mod-
els,” inInternational Conference on Learning Representations, 2022.
[23] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright,
P . Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Rayet al., “Training
language models to follow instructions with human feedback,” in
Advances in Neural Information Processing Systems, 2022.
[24] Y. J. Cho, L. Liu, Z. Xu, A. Fahrezi, and G. Joshi, “Heterogeneous
lora for federated fine-tuning of on-device foundation models,”
arXiv preprint arXiv:2401.06432, 2024.
[25] S. Chen, Y. Ju, H. Dalal, Z. Zhu, and A. Khisti, “Robust federated
finetuning of foundation models via alternating minimization of
lora,”arXiv preprint arXiv:2409.02346, 2024.
[26] R. Singhal, K. Ponkshe, and P . Vepakomma, “FedEx-LoRA: Exact
aggregation for federated and efficient fine-tuning of foundation
models,”arXiv preprint arXiv:2410.09432, 2025.
[27] J. Bian, L. Wang, L. Zhang, and J. Xu, “LoRA-FAIR: Federated
LoRA fine-tuning with aggregation and initialization refinement,”
arXiv preprint arXiv:2411.14961, 2024.
[28] Y. Wanget al., “FLoRA: Federated fine-tuning large language
models with heterogeneous low-rank adaptations,” inAdvances
in Neural Information Processing Systems (NeurIPS), 2024.
[29] J. Huang, X. Wu, T. He, and Q. Lao, “Stabilized fine-tuning with
lora in federated learning: Mitigating the side effect of client size
and rank via the scaling factor,”arXiv preprint arXiv:2603.08058,
2026.
[30] M. Kou, X. Xia, Z. Wang, I. Khalil, R. Luo, J. Zhou, and M. Xue,
“WinFLoRA: Incentivizing client-adaptive aggregation in feder-
ated LoRA under privacy heterogeneity,” inProceedings of the ACM
Web Conference 2026, 2026, pp. 5241–5252.
[31] A. Muhamed, M. Diab, and V . Smith, “CoRAG: Collaborative
retrieval-augmented generation,” inProceedings of the 2025 Con-
ference of the Nations of the Americas Chapter of the Association for
Computational Linguistics (NAACL): Short Papers, 2025, pp. 265–276.
[32] Q. Maoet al., “FedE4RAG: Privacy-preserving federated embed-
ding learning for retrieval-augmented generation,”arXiv preprint
arXiv:2504.19101, 2025.
[33] C. Lu, M. M. Amiri, and R. Raskar, “Data measurements for
decentralized data markets,”arXiv preprint arXiv:2406.04257, 2024.
[34] X. Zheng, X. Chang, R. Jia, and Y. Tan, “Towards data valuation via
asymmetric data shapley,”arXiv preprint arXiv:2411.00388, 2024.
[35] X. Zheng, Y. Huang, X. Chang, R. Jia, and Y. Tan, “Rethinking data
value: Asymmetric data shapley for structure-aware valuation
in data markets and machine learning pipelines,”arXiv preprint
arXiv:2511.12863, 2025.
[36] H. Chi, Z. Yang, L. Zeng, W. Fan, and Y. Ma, “Precedence-
constrained winter value for effective graph data valuation,”arXiv
preprint arXiv:2402.01943, 2024.
[37] H. KhademSohi, H. Hemmati, J. Zhou, and S. Drew, “Owen sam-
pling accelerates contribution estimation in federated learning,”
arXiv preprint arXiv:2508.21261, 2025.
[38] A. Ukaye, M. Abdu-Aguye, N. Tastan, and K. Nandakumar,
“Data-free contribution estimation in federated learning using
gradient von neumann entropy,”arXiv preprint arXiv:2604.22562,
2026.
[39] P . Addison, M.-T. H. Nguyen, T. Medan, J. Shah, M. T. Manzari,
B. McElrone, L. Lalwani, A. More, S. Sharma, H. R. Roth, I. Yang,
C. Chen, D. Xu, Y. Cheng, A. Feng, and Z. Xu, “C-FedRAG: A
confidential federated retrieval-augmented generation system,”
arXiv preprint arXiv:2412.13163, 2024.
[40] P . Yan, S. Liang, Y. Hua, L. Jiang, K. Yu, Y. Sun, Y. Zhang,
T. Song, N. Hu, X. Liang, B. He, and H. Guan, “Fweb3: A practical
incentive-aware federated learning framework,”arXiv preprint
arXiv:2603.00666, 2026.
[41] U. Faigle and W. Kern, “The Shapley value for cooperative games
under precedence constraints,”International Journal of Game Theory,
vol. 21, no. 3, pp. 249–266, 1992.
[42] G. Owen, “Values of games with a priori unions,” inMathematical
Economics and Game Theory. Springer, 1977, pp. 76–88.[43] M. Bun and T. Steinke, “Concentrated differential privacy: Simpli-
fications, extensions, and lower bounds,” inTheory of Cryptography
Conference, 2016, pp. 635–658.
[44] I. Mironov, “R ´enyi differential privacy,” in2017 IEEE 30th Com-
puter Security Foundations Symposium (CSF). IEEE, 2017, pp. 263–
275.

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 13
SUPPLEMENTARYMATERIAL
This supplement contains the appendices referenced from
the main paper; its sections, tables, figures, and equations
are numbered with an “S” prefix.
S1 DEFERREDPROOFS ANDADDITIONALRE-
SULTS
S1.1 Theoretical Properties
We establish the following properties under assumptions
that define their scope. Utilities are bounded in[a, b]. The
surrogate marginal prediction error is at mostϵ surin ex-
pectation. Privacy-preserving sketches introduce at most
ϵpriv utility distortion. Client metadata and provenance
commitments are verifiable and cannot be forged. Duplicate
clustering identifies a same-source split with recall at least
rdup. A hidden audit detects a poisoned contribution with
probabilityqand applies audit penaltyκ aud. A strategic
deviation can increase apparent value by at mostg ibefore
market penalties. For sybil analysis,ϵ split is the maximum
extra apparent value created by splitting a fixed portfolio,
andδis the duplicate penalty applied to each detected extra
identity in the same redundancy cluster.
Using the notation above, final payments under budget
Bare
pi=(
epi,P
jepj≤B,
Bepi/P
jepj,P
jepj> B.(S1)
Client utility isu i=p i−cact
i−ℓi, wherecact
iis actual
participation cost andℓ iis privacy or manipulation loss.
Proposition 1(S3Val estimation error).Let ¯∆ibe the empirical
mean ofm ibounded direct marginal evaluations and let bϕireplace
some direct calls with a surrogate and DP/sketch releases. If
utilities lie in[a, b], the surrogate error is at mostϵ surwith
probability1−δ sur, and privacy/sketch distortion is at most
ϵpriv with probability1−δ priv, then with probability at least
1−δ−δ sur−δpriv,
|bϕi−ϕi| ≤(b−a)s
log(2/δ)
2mi+ϵsur+ϵpriv.
Proof:Let eϕibe the empirical mean of true sampled
marginals. Hoeffding’s inequality gives the first term for
|eϕi−ϕi|. The estimator differs from eϕionly through surro-
gate replacement and privacy/sketch perturbation. A union
bound over the three events and the triangle inequality yield
the additive decomposition.
We calibrate the LCB intervals empirically. We use split
conformal calibration: sampled marginal estimates are com-
puted on a proper valuation split, absolute residuals are
computed on a calibration split, and the lower predictive
bound for a test client is bϕi−bq1−α, where bq1−α is the finite-
sample conformal residual quantile. Under exchangeability
of calibration and test residuals within a contract stratum,
this gives marginal1−αlower-bound coverage for the
target value used by that stratum. Exchangeability is only
approximate under non-IID coalitions, so we report cover-
age rather than assuming it. On small FEVER submarkets
with at most ten clients, we compute exact Shapley values
exhaustively; naive sampled intervals cover 0.3333 of exactvalues, while split conformal intervals cover 1.0 with mean
width 0.0141. On larger markets where exact Shapley is
infeasible, the same procedure is reported against leave-one-
client-out and sampled-Shapley surrogates as a diagnostic,
not as a formal proof.
Theorem 2(Selection correctness under value gaps).Fix a
budgetB, a deterministic budgeted selectorA(·), and a down-
stream test utilityU test that is independent of attack labels
and payment penalties. LetS⋆=A(ϕ)denote the budget-
feasible downstream-oracle set under true client valuesϕ i, and
letbS=A( bϕ)be the paid set obtained from estimated values.
SupposeS⋆has margin
∆A= min
i∈S⋆,j /∈S⋆ϕi
ci−ϕj
cj
>0
with respect to the value-density ordering used byA, and costs
satisfyc i≤cmax. If every client has
mi≥2(b−a)2
(∆Aci/2−ϵ sur−ϵpriv)2log2n
δ
direct-equivalent audits and∆ Aci/2> ϵ sur+ϵpriv, then
Pr[bS̸=S⋆]≤δ.
Consequently, the operator regret against the downstream oracle
satisfies
Utest(S⋆)−U test(bS) = 0
on the same high-probability event, and is at most the range of
Utestotherwise.
Proof:By Proposition 1 and a union bound overn
clients, all value estimates are within∆ Aci/2of their true
values with probability at least1−δ. On that event, every se-
lected client’s value density remains above every unselected
client’s value density, so the deterministic selector returns
S⋆. The regret statement follows because the selected set is
identical on the high-probability event; outside the event,
regret is trivially bounded by the utility range.
Remark 1 (budget feasibility).Budget feasibility is an arith-
metic invariant of the scaled payment rule. IfP
iepi≤B,
payments are unchanged; otherwisep i=Bepi/P
jepj, soP
ipi=B.
Remark 2 (payment monotonicity).For two clients with
equal cost, privacy, risk, uncertainty, and scarcity terms, a
largerLCB igives a weakly larger raw payment. Budget
scaling multiplies all positive raw payments by the same
nonnegative factor, preserving the order.
Proposition 2(Approximate incentive compatibility).Sup-
pose a deviation can increase clienti’s apparent payment by at
mostg i, but triggers expected audit penaltyqκ aud, risk discount
ηri, and manipulation lossℓman
i. Honest reporting isϵ i-dominant
with
ϵi= max(0, g i−qκ aud−ηr i−ℓman
i).
In particular, the deviation is unprofitable whenqκ aud+ηr i+
ℓman
i≥gi.
Proof:The deviation’s utility gain is at most its ap-
parent payment gain minus expected audit penalty, risk
discount, and manipulation loss. Taking the positive part
givesϵ i.

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 14
Proposition 3(Sybil splitting unprofitability).If a portfolio is
split intoksame-cluster identities and the aggregate split value
can exceed the honest unsplit value by at mostϵ split, then
Gain sybil≤ϵsplit−rdup(k−1)δ.
Sybil splitting is unprofitable wheneverr dup(k−1)δ≥ϵ split.
Proof:The split can add at mostϵ split apparent value.
With probability or recallr dup, each extra identity after
the first receives duplicate penaltyδ. Subtracting expected
penalties gives the bound.
Proposition 4(Audit-adjusted poisoning profitability).A
poisoned contribution with apparent gaing, audit detection prob-
abilityq, audit penaltyκ aud, and risk scorer ihas expected extra
profit at most
E[profit poison ]≤g−qκ aud−ηr i.
Poisoning is unprofitable wheneverqκ aud+ηr i≥g.
Proof:The apparent gain isg. The expected audit
penalty isqκ aud, and the deterministic market risk discount
isηr i. Summing the terms yields the inequality.
Proposition 5(Individual rationality).For an honest client
withℓ i= 0, ifp i≥cact
i, then participation is individually
rational. A sufficient pre-scaling condition is
LCB i+ρs i≥βc i+γϵpriv
i+cact
i.
Proof:The first statement follows fromu i=p i−
cact
i≥0. The sufficient condition makes the raw payment at
least actual cost when risk is zero; budget scaling preserves
non-negativity, and individual rationality can be enforced
by rejecting clients whose scaled payment falls below re-
serve cost.
S1.2 Extended Validation
We also run a larger FEVER validation with 28 clients,
including 24 honest clients and four strategic clients. Ta-
ble S1 shows that FL-Shapley obtains the best raw utility
in this larger setting, while FedMark-FM is close in raw
utility and does not force rare-client selection when the rare
slice is not utility-improving. Consistent with its design,
FedMark-FM optimizes a different operating point from a
pure accuracy maximizer: raw utility can favor FL-Shapley
or Shapley-UCB, while FedMark-FM targets risk-adjusted
and audit-ready market value. Figure S1 plots the risk-
adjusted utilities for the large FEVER RAG market and the
trained LoRA coalition track, where FedMark-FM reaches
0.65 versus 0.22–0.29 for the baselines.
Finally, we run actual HuggingFace/PEFT LoRA
experiments using the pretrained public encoder
distilbert-base-uncased, replacing the earlier
random tiny test fixture. A single-adapter smoke run
completes locally with 630,532 trainable LoRA parameters
out of 67,587,080 total parameters (0.9329%), takes 5.6732
seconds for training and 1.6443 seconds for inference,
and reaches 0.4688 AG News accuracy on a 64-example
smoke test. We also run a PEFT coalition market: six
client adapters are trained separately, and coalitions are
evaluated by averaging adapter logits. In this coalition
track, FedMark-FM selects no strategic adapters, retains the
Equal Volume
LeaveOneOutFL-ShapleyShapley-UCB FedMark-FM0.00.20.40.60.8Risk-adjusted utility Urisk
0.120.22
0.120.22
0.170.29
0.19 0.18 0.180.65
Large FEVER RAG
Trained LoRA (coalition)FedMark-FM (FEVER)
FedMark-FM (LoRA)Fig. S1. Risk-adjusted utilityU riskfor the large FEVER RAG market
and the trained LoRA coalition track; higher is better. FedMark-FM is
highlighted in each group.
rare adapter, and reaches 0.6462 risk-adjusted utility versus
0.2200 for equal and volume selection and 0.2859 for leave-
one-out. These results are not competitive classification
benchmarks; they verify a real pretrained-transformer PEFT
coalition path.
S1.3 Scalability
Table S2 reports the original RAG valuation runtime as
the number of clients and coalition samples increase. The
script counts two utility calls per sampled marginal. To
ground the data-systems claim in measurement rather
than projection, Table S3 adds measured FEVER markets
at approximately 200, 500, and 1000 clients with a non-
degenerate coverage utility. These larger rows use fewer
coalition samples because they validate wall-clock scale and
selection quality, not to replace the smaller high-fidelity
RAG evaluator. Client counts include four strategic stress
clients; Volume uses no coalition utility calls, while FL-
Shapley and FedMark-FM share sampled-marginal calls.
The table uses a tight selection budget with method-specific
runtime and call accounting, so Volume and FL-Shapley do
not coincide. Under this cheap coverage utility, FL-Shapley
is the strongest raw selector at scale, while FedMark-FM
remains above Volume but gives up raw coverage at 1000
clients because its risk/duplication discounts are not tuned
for the proxy. A formal cost decompositionT(n, m, k, h) =
Tinit(n)+m·(2h T ev(n)+(1−h)T su+Tfe+Tup)+L T dp+R T sg,
a coarse fit ofT step(n)≈2.0×10−3n1.5seconds forn≥16, a
per-component breakdown, and a directT ev(n)microbench-
mark are given in the appendix S3Val Computational Cost
subsection (Tables S16, S18, and S17). The analytic model
serves as an explanatory fit checked against measurements
rather than as primary evidence.
S1.4 Attack Tests
Table S4 reports stress tests on the real-data track. The values
are attack indicators: for duplicate and poison, lower is
better; for non-IID specialist, higher is better. FedMark-FM
passes all tested attacks on both tracks.
S1.5 Consolidated Robustness and Sensitivity Checks
We add targeted checks for the most fragile parts of the
market layer. Instead of presenting each auxiliary CSV as

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 15
TABLE S1
Extended validation. Large FEVER uses 28 clients. PEFT validates a trained, cached LoRA path.
Method/track Utility R-utility AUC
Large FEVER FL-Shapley 0.1869±0.0118 0.1869±0.0118 1.0000
Large FEVER Shapley-UCB 0.1837±0.0138 0.1837±0.0138 0.6573
Large FEVER FedMark-FM 0.1792±0.0137 0.1792±0.0137 1.0000
Large FEVER Leave-one-out 0.1712±0.0163 0.1712±0.0163 1.0000
Large FEVER Volume 0.0938±0.0098 0.1238±0.0098 0.7500
HF/PEFT LoRA smoke 0.4587 N/A N/A
HF/PEFT LoRA coalition 0.6162±0.1732 0.6462±0.1732 N/A
TABLE S2
Scalability of sampled valuation on the FEVER RAG track.
Clients Samples Runtime (s) Calls
8 6 0.6495 12
8 16 1.1540 32
16 6 0.8888 12
16 16 1.5915 32
32 6 1.8874 12
32 16 5.0247 32
64 6 6.2499 12
64 16 15.1439 32
TABLE S3
Measured FEVER market scale. Selection quality is the selected
coalition’s coverage utility divided by a greedy downstream oracle;
higher is better.
Clients Method Calls Runtime (s) Quality Gap
204 Volume 0 0.0001 0.6658 0.1154
204 FL-Shapley 816 0.0092 0.9260 0.0256
204 FedMark-FM 816 0.0092 0.9056 0.0326
504 Volume 0 0.0002 0.5968 0.1876
504 FL-Shapley 1008 0.0256 0.8290 0.0794
504 FedMark-FM 1008 0.0256 0.8170 0.0851
1004 Volume 0 0.0003 0.4517 0.2611
1004 FL-Shapley 2008 0.0819 0.6331 0.1748
1004 FedMark-FM 2008 0.0819 0.5260 0.2257
a separate appendix table, Fig. S2 groups roughly fifteen
checks into four panels, reported in two consolidated tables.
Table S5 covers calibration and defenses: naive sampled-
marginal intervals are overconfident while exact-Shapley
split-conformal calibration reaches full coverage on small
FEVER submarkets; no single surrogate family dominates
every artifact type; the hard-paraphrase duplicate threshold
gives zero false positives at recall 0.9575; and the incentive-
compatibility bound is zero under full defenses but positive
(attacker profit 0.035–0.075) under a deliberately weakened
profile, so the empirical bound tracks defense settings.
Table S6 covers sensitivity and system behavior: budget,
privacy-cost, DP-ϵ, and policy-drift sweeps, per-type LCB
fairness, wall-clock overhead, multi-round drift, and VCG,
procurement, and ordered-valuation baselines.
S1.6 Ablations
Table S7 ablates duplicate, risk, scarcity, and full market
penalties. The largest degradation comes from removing
risk penalties, which admits poisoned or sybil clients and
sharply reduces risk-adjusted utility. Removing scarcity
hurts rare-specialist selection in the adapter track. TheseTABLE S4
Attack summary for the real-data track.
Track Attack Mean value Pass rate
FEVER RAG Duplicate 0.0 1.0
FEVER RAG Poison 0.0 1.0
FEVER RAG Sybil 0.0 1.0
FEVER RAG Non-IID specialist 1.0 1.0
Fig. S2. Consolidated robustness and sensitivity evidence, compressing
roughly fifteen stress and sensitivity checks into four panels.
ablations support the claim that the payment mechanism is
not merely a Shapley scorer; the market-specific adjustments
drive strategic robustness.
S2 REGISTRY, LEDGER,ANDEXPERIMENT
SCHEMA
Table S9 gives the minimal registry fields used by the pro-
totype. A deployment can add organization-specific com-
pliance fields, but these fields are sufficient to reproduce
valuation, payment, and dispute checks.
The ledger stores the same round id, client id, estimated
value, uncertainty, LCB, penalties, scarcity bonus, raw/s-
caled payment, evidence hashes, and dispute state so a
payment can be replayed after model or policy changes.
Table S10 summarizes the configuration of each experiment
track.
S3 NOTATION ANDCONTRACTCARD
Table S11 collects the notation used across valuation, pay-
ment, and theory. This table is intended to make the mecha-
nism auditable: every payment field in the ledger maps to a
symbol in the paper.
A market round is defined by an immutable contract
card containing the round id, base model, allowed artifact

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 16
TABLE S5
Calibration, surrogate, and defense checks.
(a) Calibration and surrogates
Check Setting Error/coverage Rank/utility Reading
CI calibration FEVER, LOO diagnostic Naive 0.2500; calibrated
0.8875Spearman 0.8629 Raw intervals need calibra-
tion.
Exact-Shapley CI FEVERn≤10 Naive 0.3333; conformal
1.0000Width 0.0141 LCB coverage tied to exact
values.
Surrogate audit budget FEVER 0%→100% direct MAE 0.0371→0.0062 Spearman
0.4971→0.8349Direct audits matter near
payments.
Surrogate family FEVER RF, 50% audit MAE 0.0104 Spearman 0.6415; R-
util. 0.1447RF is strongest on retrieval.
Surrogate family AG News prior, 50% audit MAE 0.0399 Spearman 0.4549; R-
util. 0.6792Class-volume prior is com-
petitive.
Surrogate family HF/PEFT smoke, 50% audit MAE 0.038–0.084 R-util. 0.6462 vs 0.2200 Learned surrogate beats vol-
ume prior.
Generator RAG FLAN-T5 held-out reader Acc. 0.3937 Gap 0.075–0.081 Real generator shows robust
serving gap.
(b) Defenses
Check Setting Detection Market outcome Reading
Semantic clustering Threshold 0.55 P/R/F1
1.0000/0.9975/0.9987FPR 0.0000 Easy duplicates are caught.
Embedding duplicate Combined threshold 0.50 P/R/F1
1.0000/0.9889/0.9944FPR 0.0000 Hash embeddings improve
paraphrase recall.
Hard paraphrases Threshold 0.65 P/R 1.0000/0.9575 FPR 0.0000 Stricter threshold trades re-
call for precision.
Collusion audit Cap 0.10 vs none Colluders 0.0 vs 2.0 R-util.−0.0176vs
−0.1373Group cap suppresses artifi-
cial complementarity.
Higher-order collusion 3 clients, cap 0.08 Colluders 0.0 R-util.−0.0366 Three-client split is also sup-
pressed.
Collusion flags Threshold 0.25 Precision 0.0747; recall
1.0000Legit-FP 0.0 Flags trigger audit, not au-
tomatic penalty.
IC bounds Full defenses q= 0.09–1.0 ϵIC= 0; profit 0 Implemented attacks un-
profitable.
IC bounds No cluster,η= 0.10 q= 0,κ= 0 ϵIC, profit 0.035–0.075 Bound has discriminating
power.
Held-out attacks Adaptive keyword Poison 0.0; strategic 0.2 Acc. 0.3857; regret
0.0514Served test metric is not cir-
cular.
Cost gaming 1×–10×cost plus duplicate Attacker 0.0 Strategic 0.0 Cost inflation does not raise
profit.
Contract-card swap Weight/slice/probe swaps Detected 1.0 Payment preserved 1.0 Commit-reveal catches post-
hoc changes.
types, utility weights, validation slices, budget, valuation
budget, payment parameters, audit policy, and release pol-
icy.
S4 METRICDEFINITIONS
For a selected client setS, the reported utility is the same
contract utility used for market selection. Risk-adjusted
utility adds a market-level robustness adjustment:
Urisk(S) =U(S)−α strat|S∩S strategic |
+αrare1{S∩S rare̸=∅}.
The coefficients are fixed by the experiment card. Harmful-
client AUC treats poisoned, sybil, and duplicate clients as
positives and ranks clients by risk score. Duplicate discount
is the ratio by which a duplicate client’s payment is reduced
relative to its unsuppressed score. Utility ratio divides a
method’s utility by the oracle subset utility under the same
budget. Strategic selected is the number of selected clients
with attack labels. Rare selected indicates whether at least
one rare-domain specialist is retained.
S5 STRESS-TESTPROTOCOL
The stress tests instantiate sybil splitting, duplicate sub-
mission, poisoned contribution, privacy gaming, cost infla-
tion, non-IID specialists, and collusion. Each test recordsselected strategic clients, attack profit, duplicate discount,
rare retention, or reserve violation. The goal is executable
measurement of common market failures, not immunity to
arbitrary adaptive attacks.
S6 CONTRIBUTIONINTERFACEDETAILS
The registry separates private payloads from market-visible
metadata. The payload may remain at the client, inside a
secure enclave, or behind a federated endpoint. The market-
visible portion must be sufficient to schedule evaluations,
detect obvious duplicates, compute privacy and cost adjust-
ments, and bind later disputes to the same artifact. Table S12
details the required, optional, and private fields for each
artifact type.
The market operator should reject artifacts whose base-
model id, license, endpoint policy, or provenance commit-
ment is inconsistent with the round contract. This is a data-
quality and governance check before any mechanism-design
step is applied.
S6.1 Secure Evaluation Interfaces
For retrieval corpora, the sandbox creates a tempo-
rary view or queries signed client endpoints, then com-
putes hit rate, redundancy, latency, and risk without

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 17
TABLE S6
Sensitivity, operating-point, system, and policy-stability checks.
(a) Sensitivity and operating point
Check Setting Utility/stability Selection Reading
Contract weights FEVER supports slice R-util. 0.2463 Strategic 0.0; rare 1.0 Slice-aware cards matter.
Budget sweep FEVERB= 0.9→2.2 Audit 0.95–0.93 Selected 5–13; rare 0–0.67 More budget changes reten-
tion.
Privacy-cost sweep γ= 0→0.35 R-util. 0.1964–0.1965 Strategic 0.0; rare 1.0 Ordering stable in this grid.
DP epsilon ϵ= 0.5→16 Rank corr.−0.4637–
−0.317616 releases Per-value DP noise harms
payments.
Policy stability RAG top1/top5 vs top3 Spearman 0.8948/0.9594 Strategic 0.0; rare 1.0 Retrieval policy changes
values.
Drift stability Dense/slice policies Spearman 0.7995/0.5614 Strategic 0.0; rare 1.0 Slice drift is quantitatively
visible.
Adapter policies Hierarchical/orthogonal proxies R-util. 0.7169/0.6749 Strategic 0.0; rare 1.0 Adapter routing is a con-
tract dimension.
LCB fairness Adapter per-type LCB R-util. 0.6863 Unc. ratio 1.352 Per-type calibration helps
rare clients.
Alpha Pareto FEVERα s, αrgrid Rank 4.33 ifα r= 0; 1.67
ifαr>0Raw 0.1664 Composite ranking is trans-
parent.
(b) System and policy stability
Check Setting Runtime/calls Outcome Reading
System overhead 100 clients, 16 samples 3200 calls; 3.4146s 937.3 calls/s Smoke-shard overhead is
small.
Utility kernel Fixed shard, 100 clients 0.296 ms/call 24 utility calls Confirms fast 100-client row.
Utility kernel Scale-sweep shard, 8 clients 1.371 ms/call 24 utility calls Corpus growth explains
slower fit.
50+ client scale FEVER/AG News proxy 324/200 calls 1.50s/1.10s Larger proxy run remains
practical.
Measured scale FEVER 204/504/1004 clients 0–2008 calls 0.0001–0.0819s Real wall-clock rows con-
firm scale by measurement.
DP enclave overhead 20 releases 0.0045s 0.226 ms/release Software evidence overhead
is tiny.
Multi-round 6 rounds Drift 0.0000–0.0306 Strategic 0.0 New rounds handle policy
drift.
VCG toy Observable 6-client market VCG 0.0799; FedMark-
FM 0.0588VCG needs raw obs. VCG wins when observabil-
ity holds.
Procurement baseline FEVER OPT-style R-util. 0.2472 Rare 0.0 vs FedMark-FM
rare 1.0Budget efficiency differs
from auditability.
Ordered valuation FEVER ADS-style Spearman 0.7554 Overlap 0.6667 Pipeline order can change
payments.
TABLE S7
Ablation summary.
Track Variant R-utility Strategic
FEVER RAG Full 0.1843 0.0
FEVER RAG No risk penalty -0.4086 3.0
FEVER RAG No market penalties -0.3091 2.67
receiving the full corpus. For adapters, the contract
fixes the base model, target modules, composition pol-
icy, and validation slices; our implementation includes a
cacheddistilbert-base-uncasedPEFT path. Prompts,
demonstrations, preference data, and safety data are evalu-
ated by contract probes, paraphrase variants, reward/pref-
erence probes, or hidden policy probes; public outputs
are aggregate scores and evidence hashes unless a dispute
requires controlled disclosure.
S7 DIFFERENTIALPRIVACY ANDENCLAVEIMPLE-
MENTATION
Our implementation provides a concrete privacy layer. The
DP module supports bounded mean releases with clipping,
Laplace noise for pure DP , Gaussian noise for approximate
DP , post-processing to the declared range, and a composi-tion accountant. Gaussian releases are accounted with zCDP
and can be related to RDP-style accounting [43], [44]:
ρ=X
t∆2
t
2σ2
t, ϵ(δ) =ρ+ 2q
ρlog(1/δ).
Laplace releases compose through sequential epsilon addi-
tion. Each release recordsn, sensitivity, mechanism, noise
scale, per-release privacy spend, and cumulative privacy
spend.
The local secure-enclave backend enforces a policy be-
fore releasing any statistic. The policy specifies allowed
functions, maximum epsilon, maximum delta, whether DP
is required, and whether raw values may be released. The
default policy forbids raw-value release. Every enclave out-
put includes a manifest measurement, policy hash, input
hash, output hash, cumulative DP spend, backend type,
hardware-attestation flag, and a signature. The reproducible
backend signs evidence with an Ed25519 key, so any auditor,
buyer, or client can verify evidence with the public key
alone; an HMAC, by contrast, can be forged by any party
able to check it and is therefore not third-party verifiable.
The genuine evidence verifies while any tamper to the
released value or the freshness nonce invalidates the sig-
nature. The evidence schema mirrors the field layout of
real TEE attestation documents (Table S13), so a hardware
backend is swapped in by replacing only the signer and

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 18
TABLE S8
Positioning FedMark-FM against adjacent literature. Coverage in the middle columns is shown as Harvey balls (empty = none, quarter =
limited/indirect, half = partial, full = yes/addressed).
Line of work Contribution unit FM artifacts Incentives Strategic robustness Gap addressed by FedMark-FM
FL incentives [2], [7] Client data or update Does not model RAG, prompts, adapters, pref-
erence, and safety artifacts as traded products.
Shapley/data valuation [3],
[4], [17]Examples, sources, clients Estimates contribution, but usually lacks a
payment ledger, audits, and attack-aware mar-
ket controls.
Data markets and auc-
tions [14], [33]Datasets or sellers Prices data access, but does not bind valuation
to private FedFM evaluation pipelines.
Structure-aware Shap-
ley [34], [35], [36]Ordered data groups Respects pipeline order, but does not define
FedFM registry, payment, and audit machin-
ery.
FedFM systems [5], [9] Private model/data collaboration Identifies incentives as a challenge, but leaves
the concrete market mechanism unspecified.
Secure/incentive FL sys-
tems [39], [40]Updates or RAG context Provides confidentiality or settlement sub-
strate, but not typed FM artifact markets.
FM adaptation valua-
tion [19], [20]Prompts, RAG passages, adapters Values artifacts for model improvement with-
out federated sellers, payments, and dispute
evidence.
FedMark-FM Typed private artifacts Integrates scalable valuation, budgeted pay-
ments, risk controls, and auditable system
state.
TABLE S9
Contribution registry schema.
Field Meaning
client id Stable seller identity or verified organiza-
tion handle
artifact id Unique artifact identifier bound to prove-
nance hashes
artifact type Retrieval, adapter, prompt, demonstration,
preference, safety, or update sketch
domain tags Declared and inferred task/domain strata
for sampling
provenance hash Signed hash or commitment for duplicate
and dispute checks
privacy budget Declared evaluation budget or privacy pol-
icy handle
declared cost Claimed compute, curation, labeling, or
serving cost
risk flags Poison, duplicate, sybil, policy, or uncer-
tainty indicators
endpoint ref Federated retrieval, adapter, local evalua-
tor, or certificate endpoint
license policy Use constraints, retention policy, and pay-
ment eligibility
its certificate chain, leaving the market ledger unchanged.
The one property the software backend cannot provide is a
hardware root of trust for the signing key: on real hardware
thecertificate/cabundlefields carry a manufacturer-
endorsed chain (AWS Nitro, Intel SGX/DCAP , AMD SEV-
SNP), whereas the software backend self-signs them and
reportshardware_attested=false.
The prototype releases two DP aggregate statistics in-
side the software enclave: mean contribution quality and
mean privacy budget. The privacy, enclave, and attestation
components exercise bounded input, DP release, budget
check, output hash, Ed25519-signed evidence, third-party
verification, and ledger storage, writing signed evidence
and an attestation document to disk. Table S14 lists the DP
accounting settings used in these checks.
The DP epsilon sweep shows the expected privacy-
utility tension. With 16 bounded value releases on FEVER,
strong noise atϵ= 0.5sharply degrades rank correlation,
and evenϵ= 16remains noisy in this tiny per-client
release setting. We therefore do not recommend privatizing
each individual value release independently in production:composition over many per-client, per-round value releases
forces aggressive noise, which can invert payment ranks.
The intended use is DP aggregate reporting plus secure or
attested evaluation for payment-critical values. Figure S3 vi-
sualizes this privacy-utility tradeoff. Thus payments them-
selves are not claimed to be differentially private in the cur-
rent prototype; their confidentiality comes from the secure-
evaluation interface and evidence policy.
S8 S3VALIMPLEMENTATION ANDSENSITIVITY
NOTES
S3Val samples marginal pairs(C, i)by artifact type, valida-
tion slice, redundancy cluster, and risk stratum. The surro-
gate features include coalition size, artifact mix, slice cov-
erage gaps, redundancy withC, privacy/cost fields, drift,
risk, scarcity, and a direct-evaluation indicator; retrieval and
adapter tracks add sketch and adapter-fingerprint features.
Direct audits are prioritized when uncertainty is high, a
payment is near zero or the budget boundary, or a risk
flag is present. The evaluation runner reports the detailed
CSV rows; the compact reading is that direct audits reduce
FEVER MAE from 0.0371 at 0% audit to 0.0062 at 100%, and
that no single surrogate dominates every artifact type.
Budget, privacy, and policy sweeps are treated as cal-
ibration evidence rather than universal defaults. Increas-
ing FEVER budget from 0.9 to 2.2 grows selected clients
from 5 to 13 while preserving zero strategic selections
in the tested grid. The privacy-cost sweep overγ∈
{0,0.05,0.10,0.20,0.35}changes payment scores but not
strategic selection in the current setup because risk and
duplicate terms dominate. Operators should publish the
Pareto frontier over raw utility, strategic selections, rare
retention, audit rate, and dispute-candidate rate before set-
tling a market round. Figure S4 shows how the FedMark-FM
rank on FEVER depends on the operator-side risk-adjusted
utility weights: the rank improves when rare retention is
valued and weakens whenα rare= 0, making the operating-
point tradeoff visible rather than implicit.

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 19
TABLE S10
Experiment configuration card.
Track Seeds Clients Artifacts Utility and baselines
FEVER RAG 3 18 Claim evidence passages with dupli-
cate, rare, and poison clientsRetrieval accuracy minus risk; equal, volume,
leave-one-out, FL-Shapley, retrieval similarity,
Shapley-UCB, FedMark-FM
Trained low-rank adapter 3 12 Lightweight PyTorch low-rank adapters Held-out classification accuracy and robustness
checks
Large FEVER RAG 10 28 More clients and strategic variants Multiseed RAG utility, risk-adjusted utility,
AUC, and scalability counters
HF/PEFT LoRA smoke 1 1 Cached pretrained DistilBERT LoRA
pathTraining seconds, inference seconds, trainable
parameters, total parameters, and LoRA per-
centage
HF/PEFT LoRA coalition 2 4 Real PEFT client adapters and coalition
evaluationsCoalition utility with equal, volume, leave-one-
out, and FedMark-FM payment selections
TABLE S11
Notation summary.
Symbol Meaning
N Set of participating clients
Πi,Π(C) Clienti’s private portfolio and coalition portfolio
union
C Coalition of clients evaluated by the sandbox
U(C) Contract utility of coalitionC
ϕi Ideal Shapley-style value of clienti
bϕi Estimated client value returned by S3Val
σi Estimation uncertainty for clienti
LCB i Lower confidence bound bϕi−λσ i
λ Uncertainty discount; default value0.75
β Cost-penalty coefficient; default value0.28
γ Conversion from privacy/evaluation spend to pay-
ment penalty; default value0.20
η Manipulation-risk penalty; default value0.75
ρ Scarcity bonus weight; default value0.25
di, ri Duplicate risk and manipulation risk
Φ(C) Privacy-budget consumption or leakage term in util-
ity
κaud Audit penalty applied after verified manipulation
ci, ϵpriv
i, si Verified cost, privacy budget, and scarcity score
epi, pi Raw and budget-scaled payments
B Market budget for the round
Choosing the payment coefficients.The weights
(λ, β, γ, η, ρ)are operator policy rather than data-fitted
parameters, and we recommend selecting them from the
Pareto frontier above: fix the uncertainty discountλfrom the
desired LCB coverage, set the cost and privacy conversions
β, γfrom verified unit prices, and choose the risk and
scarcity weightsη, ρto hit a target strategic-selection and
rare-retention rate. Selection is robust to the remaining slack
because, as the sweeps show, the risk and duplicate terms
dominate the ordering onceηexceeds a small threshold, and
the privacy-cost sweep leaves strategic selection unchanged.
Duplicate and manipulation scores are audittriggersbacked
by provenance commitments, not guarantees: token/tri-
gram and embedding overlaps catch easy and paraphrased
copies but can be evaded by strong adaptive paraphrase,
so a deployment should add provenance attestation and
watermark checks for defense-in-depth, and update-sketch
tracks can plug a gradient-diversity sybil defense [15] in
behind the same evaluator interface.
Scarcity bonuses require pre-committed validation slices;
post-hoc seller-proposed micro-slices are dispute evidence,
not automatic bonuses. Duplicate defenses combine prove-
nance hashes, lexical/embedding similarity, adapter delta
fingerprints, target-module metadata, and hidden probes.
0.5 1 4 8 16
DP  per release
0.4
0.2
0.00.20.4mean metric
DP value-release tradeoff
rank correlation
payment overlap
risk-adjusted utilityFig. S3. DP value-release tradeoff on FEVER.
0.05 0.1 0.3 1
strategic penalty αstrat0
0.05
0.3
1rare bonus αrare4.3 4.3 4.3 4.3
1.7 1.7 1.7 1.7
1.7 1.7 1.7 1.7
1.7 1.7 1.7 1.7FedMark-FM rank across risk-utility weights
1.01.52.02.53.03.54.04.55.0
mean rank (lower is better)
Fig. S4. Sensitivity of FedMark-FM’s rank to the operator-side risk-
adjusted utility weights on FEVER.
The token/trigram tests are reproducible low-cost checks,
not a complete adaptive-paraphrase defense.
S8.1 Ordered Pipeline Valuation
S3Val can use unordered or ADS-style ordered sampling.
Ordered sampling restricts permutations to retrieval→

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 20
TABLE S12
Artifact-specific contribution interface. Required fields are visible to the market; private fields remain behind the endpoint or are exposed only
through secure evaluation.
Artifact Required market-visible fields Optional sketches/certificates Private payload
Retrieval corpus Domain tags, language, passage count,
provenance hash, endpoint reference, li-
cense policyMinHash, embedding centroids, BM25 term
sketch, coverage histogram, duplicate hashesPassage text, document metadata, private
access logs
LoRA adapter Base-model id, rank, target modules,
parameter count, adapter hash, calibra-
tion split idDelta norms, routing hints, public calibration
trace, safety certificateAdapter weights when served behind a
private loader
Prompt template Task tags, token length, prompt hash,
policy constraintsRobustness trace, paraphrase sensitivity,
public/private score gapFull prompt if proprietary
Demonstrations Task tags, count, format, label-space de-
scription, source hashEmbedding centroid, class/domain his-
togram, annotator reliabilityDemonstration text or labels
Preference data Pair count, preference dimensions, an-
notator policy, source hashDisagreement rate, category histogram,
reward-model certificatePairwise preference records
Safety data Risk taxonomy, policy version, source
hash, severity mixHidden-probe certificate, jailbreak family
tags, red-team provenanceFull safety prompts and expected policies
Update sketch Round id, model version, secure aggre-
gation handle, update norm rangeClipped norm certificate, DP parameter cer-
tificate, validation deltaRaw gradient/update
TABLE S13
Attestation-evidence fields map to real TEE attestation documents;
only the root of trust for the signing key differs in the software backend.
FedMark-FM evidence AWS Nitro Intel SGX/DCAP AMD SEV-SNP
measurement PCR0 MRENCLAVE MEASUREMENT
code_hash PCR8 MRSIGNER IDKEY DIGEST
public_key public_key REPORTDATA REPORT DATA
nonce nonce REPORTDATA REPORT DATA
output_hash user_data REPORTDATA REPORT DATA
certificate certificate PCK cert VCEK cert
signature COSE Sign1 ECDSA quote VCEK sig
TABLE S14
DP accounting settings used in our checks.
Check Mechanism Per rel.(ϵ, δ) Clip Releases
Enclave mean Gauss./Lap. (0.5,10−6) [0,1] 1–20
Value sweep Laplace (0.5–16,10−6)[−1,1] 16
Privacy grid Lap./zCDP
proxyγgrid track-
specific16–50
prompt/demonstration→adapter→preference/safety
groups, preserving marginal-credit accounting while match-
ing serving precedence. On FEVER, ordered S3Val has
Spearman 0.7554 and selected-set overlap 0.6667 versus
unordered S3Val. The contract card stores the group order
so disputed payments can be replayed.
Sensitivity to order misspecification.Because credit
depends on the declared group order, we test how much
a wrong order matters. Within a group the value is
permutation-invariant by within-layer symmetry, so only
the cross-group precedence is consequential. Reversing the
retrieval≺reader precedence on the real-FEVER pipeline
redistributes credit substantially (correct-versus-reversed
Spearman0.57) and modestly lowers held-out serving ac-
curacy (∆ =−0.04±0.04), confirming that the serving
order is a consequential contract parameter rather than
a free choice. The operator therefore commits the order
in the contract card and disputes replay payments under
the recorded order; when the correct order is genuinely
unknown, running unordered S3Val avoids imposing an
unwarranted precedence.
Is ordered valuation more correct, or only different?TABLE S15
Ordered valuation is serving-neutral. Paired ordered−unordered
held-out FEVER serving accuracy at strong precedence (g=1); every
CI covers zero. Credit is nonetheless redistributed (bottom row).
Budget ∆accuracy (95% CI) P(∆>0)
1.4 +0.011±0.021 0.47
1.7 −0.009±0.022 0.40
2.0 +0.004±0.014 0.42
Credit redistribution Spearman 0.725 overlap 0.690
Changed payments are not by themselves evidence that
ordered credit is the right credit. We therefore ran a con-
trolled held-out serving study on real FEVER data with
two artifact groups that have a genuine serving precedence:
retrieval clients provide context and reader clients supply
an answer policy that can only correct a label once the
retrieval group has surfaced the gold evidence. A prece-
dence knobg∈[0,1]tunes how strongly reader value
is gated on retrieval (g=0: no precedence;g=1: strong
precedence). Each rule scores clients on a validation card,
selects a budget-feasible coalition, and is then served on a
disjoint held-out test card; the served accuracy is the ground
truth, and the same market and budget are used for both
rules. Table S15 reports the outcome. Ordered valuation
redistributes credit (ordered-vs-unordered Spearman0.725,
selected-set overlap0.690), but the paired held-out accuracy
difference is statistically indistinguishable from zero across
budgets at strong precedence (∆∈[−0.009,+0.011], all
95% CIs covering zero, sign≈50/50). An exact-Shapley
small market agrees (−0.011±0.058) and confirms the
samplers recover their targets (sampled-ordered vs exact-
orderedρ=0.958; sampled-unordered vs exact-symmetric
ρ=0.935). We conclude that ordered valuation isserving-
neutral: it aligns payment with the causal serving order
without changing task quality, which is a payment-fairness
property rather than an accuracy improvement.
S8.2 Computational Cost
We decompose one market round into a one-time setup
phase and a per-marginal sampling phase. Letnbe the
number of clients,mthe number of sampled marginal pairs

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 21
TABLE S16
Complexity-model fit to Table S2. Fitted constants are rounded; the
scaling interpretation is intended forn≥16.
Clientsn Marginalsm Measured (s) Fitted (s) Rel. err.
8 6 0.6495 0.444 −31.6%
8 16 1.1540 0.834 −27.7%
16 6 0.8888 0.821 −7.6%
16 16 1.5915 1.891 +18.8%
32 6 1.8874 1.916 +1.5%
32 16 5.0247 4.855 −3.4%
64 6 6.2499 4.974 −20.4%
64 16 15.1439 13.048 −13.8%
TABLE S17
DirectT ev(n)microbenchmark. Fixed-shard rows keep per-client shard
size constant; scale-sweep rows use the same shard schedule as the
scalability fit.
Shard schedule Clients Docs/client ms/call
Fixed smoke 8 3 0.163
Fixed smoke 32 3 0.248
Fixed smoke 64 3 0.264
Fixed smoke 100 3 0.296
Scale sweep 8 22 1.371
Scale sweep 32 5 0.433
Scale sweep 64 3 0.265
Scale sweep 100 3 0.291
(C, i),kthe number of redundancy clusters,h∈[0,1]the
direct-audit fraction,Lthe number of differentially private
(DP) releases per round, andRthe number of evidence
records signed. The round cost is
T(n, m, k, h) =n T sk+n2Tcl+n T reg| {z }
Tinit(n)
+m· 2h T ev(n) + (1−h)T su+Tfe+Tup
+L T dp+R T sg,(S2)
whereT skis sketch construction per client (MinHash and
centroids),T clis pairwise redundancy clustering (linkage
over sketch similarity,O(n2)worst case),T regis registry
write plus provenance commit,T ev(n)is one coalition utility
call (the only term that scales withnthrough retrieval index
size or coalition payload),T suis one surrogate prediction,
Tfeis feature-vector construction,T upis the online surrogate
update (online ridge:O(d2)wheredis feature dimension),
Tdpis one DP release (clip, noise, accountant update), and
Tsgis one signed-evidence write. Utility evaluation domi-
nates whenn,m, orhis large; DP and signing scale with
round-policy constants. A coarse fit to Table S2 gives
Tstep(n)≈2.0×10−3n1.5s, T init(n)≤1 s,
forn≥16. Table S16 reports measured versus fitted
runtime; Table S17 separates fixed-shard and scale-sweep
utility-call costs; Table S18 shows that per-marginal evalua-
tion dominates oncemgrows.
Combining the fits, FedMark-FM valuation at fixed sur-
rogate audit ratehcosts
T(n, m) =O n1.5hm+O(n2) +O d2m+O(L+R),
where the first term (coalition utility evaluation) is domi-
nant wheneverhm≥Θ(n0.5)and where theO(n2)clus-
tering term is incurred once per round. Under the defaultTABLE S18
Projected runtime under the fitted model form∈ {16,64,256,1024}.
Calls counts assumeh=1(full direct audits, two utility calls per
marginal). Surrogate triage withh<1reduces total time proportionally.
Clientsn Marginalsm Calls Tpred (s) Tstep share
64 16 32 13.05 99.0%
64 64 128 51.80 99.7%
64 256 512 206.83 99.9%
64 1024 2048 826.92 99.98%
100 16 32 24.88 99.5%
100 64 128 99.16 99.9%
100 256 512 396.30 99.97%
100 1024 2048 1584.83 99.99%
direct-audit budgeth m≤O(nlogn), total cost is ˜O(n2.5)
per round, which is the relevant scaling for sizing the model-
call budget against an operator latency SLO.
S8.3 Calibration, Redundancy, and Drift
LCBs use calibrated intervals rather than raw sampled-
marginal variance. In the current FEVER run, naive inter-
vals cover 0.25 of leave-one-out values, while conformal
calibration reaches 0.8875 with mean width 0.0408. Retrieval
duplicate clustering uses provenance hashes plus
sdup(x, y) = 0.65s tok(x, y) + 0.35s tri(x, y),
with adapter deployments adding delta-norm and target-
module fingerprints. The hard-paraphrase recall at thresh-
old 0.65 is 0.9575, so these checks are audit triggers, not com-
plete guarantees. Other auxiliary results are consolidated
as follows: collusion group caps suppress two- and three-
client artificial complementarity in the tested suite; per-
type LCB normalization improves adapter rare-client utility
from 0.6625 to 0.6863; policy drift is visible when changing
retriever or adapter-composition rules; and the software
enclave adds only 0.226 ms per bounded signed release.
Payments should not be reused across model, retriever, or
policy changes.
S9 PAYMENT, BUDGETING,ANDDISPUTES
S9.1 Payment Decomposition
For each client, the ledger stores a decomposed payment:
Si=bϕi−λσ i−βc i−γϵpriv
i−ηmax(d i, ri) +ρs i,(S3)
pi=scale B([Si]+),(S4)
wherescale Bapplies the budget-feasible scaling rule. The
decomposition is important for review and dispute han-
dling. A client should be able to see whether low payment
is due to low estimated value, high uncertainty, duplicate
risk, privacy restrictions, verified cost, or manipulation risk.
The operator should be able to recompute the exact payment
from the contract card and ledger fields.
S9.2 Budget Scaling
IfP
iepi> B, the rule scales every positive raw payment
by the same factor. This preserves payment order among
clients with fixed adjustment terms and prevents the market
from exceeding the budget. A deployment may add reserve

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 22
1 2 3 4 5 6
market round0.0000.0250.0500.0750.1000.125score
slice shiftMulti-round market dynamics
payment drift
reputation penaltyrare retained
strategic selected0.00.20.40.60.81.0
selection count
Fig. S5. Six-round market dynamics with client arrivals and a slice
shift after round three, separating payment drift, reputation penalty, rare
retention, and strategic selections.
checks after scaling: if a client’s scaled payment falls below
its verified participation cost, the client can be excluded and
the remaining budget recomputed.
S9.3 Dispute Workflow
A dispute proceeds in five steps. First, the client submits the
disputed round id, artifact id, and reason. Second, the au-
ditor retrieves the contract card, registry entry, value ledger,
and evidence hashes. Third, the auditor reruns determin-
istic evaluations under the recorded seed where possible.
Fourth, the auditor can request additional hidden probes
or provenance review if duplicate, poison, or leakage flags
are contested. Fifth, the ledger records a signed decision:
unchanged, revalued, rejected, or escalated. The important
design point is that the dispute is about auditable data
artifacts, not an opaque payment number.
S9.4 Revaluation Policy
Model, retriever, prompt, and safety-policy changes can al-
terU(C). FedMark-FM treats a market round as immutable
once settled: the original contract card, model hash, policy
hash, utility weights, and evidence hashes define the value
of that round. Later model or policy changes create a new
round with a new contract card. The default policy is
forward-looking revaluation rather than clawback, because
clawbacks make participation risky and can punish honest
clients for operator-side changes. Clawbacks are reserved
for fraud cases where provenance, duplicate, or poison
evidence shows that the original submission violated the
signed contract. Figure S5 illustrates six-round market dy-
namics with a slice shift after round three, showing why
revaluation should be round-bound rather than silently
rewriting old payments.
S9.5 Commit-Reveal and Operator Threats
The client threat model is not sufficient by itself: an oper-
ator could collude with a seller by retrofitting a rare slice,
changing hidden probes, or reweighting utility after seeing
submissions. FedMark-FM therefore uses a commit-reveal
round protocol. Before the submission window opens, theoperator publishes and signs a contract-card hash contain-
ing utility weights, rare-slice definitions, validation-slice
commitments, audit policy, payment formula version, and
dispute window. Clients submit artifacts against that hash.
After the window closes, the operator reveals the card
and evaluation seeds; auditors verify that the revealed
card matches the pre-submission commitment. The multi-
round stress test includes a post-hoc slice-shift condition; the
ledger treats it as a new round, so prior payments remain
bound to the original committed card rather than being
silently recomputed.
S9.6 Operator and Auditor Adversaries
The strongest auditability guarantees require more than an
honest operator. If the operator selectively evaluates coali-
tions, the registry and ledger still expose which evaluation
calls were made, but value completeness requires public
sampling seeds, third-party reruns, or an external auditor.
If the operator swaps the contract card, signed pre-window
hashes and a public bulletin-board anchor detect the swap.
If the operator redacts ledger entries, append-only storage
and third-party hash anchoring reveal gaps but cannot
reconstruct missing private payloads without client-side
evidence. If the operator colludes with a seller, hidden-probe
commitments, provenance hashes, and dispute replay limit
post-hoc rare-slice and weight manipulation, but economic
fairness depends on an independent auditor or buyer-side
challenge process. If the auditor is compromised, signatures
and public anchors preserve tamper evidence, but dispute
decisions require auditor rotation, threshold signatures, or
multi-auditor review. If keys are lost or stolen, old evidence
remains verifiable only up to the last trusted key-rotation
checkpoint. Table S19 summarizes this operator/auditor
threat surface, the required mitigation, and the residual risk
in each case.
S10 THREATMODEL
The market assumes clients may be strategic but not om-
nipotent. They may split identities, copy or paraphrase
artifacts, poison data, tune to public probes, inflate costs,
exaggerate privacy constraints, or collude. The market as-
sumes that clients cannot forge provenance commitments,
cannot break cryptographic hashes, cannot observe hidden
audit probes before submitting, and cannot force the oper-
ator to evaluate raw private payloads outside the declared
interface. The operator is not assumed to be fully trusted
for auditability claims. The base experiments assume the
operator follows the published contract card, while the de-
ployed protocol relies on signed cards, append-only ledgers,
public hash anchors, attestation chains, and external dispute
review to detect selective evaluation, card swaps, ledger
redaction, and collusion. Table S20 summarizes the client-
side threats, their market mitigations, and the residual risk
of each.
S11 EXPERIMENTALDETAILS
S11.1 Baseline Definitions
Equal payment assigns all clients the same score before
budget selection. Volume ranks clients by artifact count or

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 23
TABLE S19
Operator/auditor threat surface.
Adversary action What survives Required mitigation Residual risk
Selective evaluation Ledger shows sampled coalitions and
missing callsPublic seeds, audit reruns, model-call counters Private endpoints may be
unavailable later
Contract-card swap Pre-window signed hash detects mis-
matchPublic bulletin board or third-party hash anchor If no anchor exists, clients
rely on operator logs
Ledger redaction Hash-chain gap is detectable Append-only ledger, external checkpoints Redacted payload
evidence may need
client replay
Seller/operator collusion Committed weights and rare slices con-
strain retrofitsHidden-probe commitments and buyer/auditor
challengeCollusion before card pub-
lication is governance risk
Auditor compromise Signatures and anchors preserve raw evi-
dence integrityAuditor rotation, threshold review, public dispute
summariesBad decisions can still de-
lay payment
Key failure Prior checkpoints remain verifiable Key rotation, HSM/TEE-backed signing, revoca-
tion logEvidence after compro-
mise must be re-attested
TABLE S20
Threat model and mitigation summary.
Threat Client action Market mitigation Residual risk
Sybil split Divide one portfolio across many identities Redundancy clusters, group marginal dis-
count, duplicate penaltiesSemantic splits may evade weak
clustering
Duplicate copy Copy another client’s passages or adapter Provenance hashes, sketch similarity, dupli-
cate discountParaphrases require stronger se-
mantic checks
Poisoning Improve public validation while harming
hidden behaviorHidden probes, risk score, audit penalty Adaptive poisoners can target
unknown policies
Prompt gaming Tune prompts to public probes Paraphrase probes, public-hidden gap, un-
certainty discountProbe leakage weakens defense
Privacy gaming Claim restrictive privacy to reduce evalua-
tion visibilityLower confidence bound, alternative secure
evaluation, reserve checksLegitimate privacy may also
widen intervals
Cost inflation Overstate collection or compute cost Verified cost, reserve prices, cost penalties Verification may be domain-
specific
Collusion Split complementary artifacts across identi-
tiesPairwise complementarity audit and group
capsTrue complementarity should not
be over-penalized
Benchmark leakage Submit data derived from validation an-
swersProvenance review and hidden evaluation
slicesPerfect leakage detection is im-
possible
corpus size. Retrieval similarity ranks retrieval clients by
similarity to validation queries or gold evidence sketches.
Leave-one-client-out estimatesU(N)−U(N\ {i}), which
is accurate but expensive and can behave differently from
Shapley values when interactions are strong. FL-Shapley
approximates marginal contribution over sampled client
orderings. Shapley-UCB uses uncertainty-aware seller selec-
tion inspired by bandit upper-confidence bounds. FedMark-
FM differs by ranking with lower-confidence-bound value
plus explicit market adjustments for cost, privacy, duplicate
risk, manipulation risk, and scarcity.
S11.2 Experiment Construction Details
FEVER RAG shards claim-evidence pairs into honest gener-
alists, rare specialists, duplicates, sybils, and poisoners. The
held-out serving tracks separate validation-card selection
from final test-card scoring and report task metrics without
using attack labels. The trained low-rank adapter track uses
lightweight PyTorch adapters, while the HF/PEFT path
usesdistilbert-base-uncasedto report real LoRA
parameters, runtime, and coalition behavior. The PEFT
track proves the interface and accounting path; large multi-
adapter foundation-model valuation remains future scale-
up work.
S11.3 Hyperparameters and Reporting Conventions
We use fixed random seeds for each experiment. Bud-
geted selection uses the same budget within each track forall methods. Risk-adjusted utility uses the same strategic
penalty and rare-client bonus within a track. Harmful-client
AUC is reported only when harmful labels exist. PEFT run-
time fields are reported only when the optional PEFT depen-
dencies and cacheddistilbert-base-uncasedmodel
are available.

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 24
TABLE S21
Retrieve–read held-out FEVER serving proxy. Accuracy, macro-F1, evidence exact match (EM), and regret are computed only on the final test
card and do not use attack labels. This table is supporting evidence; Table 4 is the primary served-pipeline result.
Attack family Method Acc. Macro-F1 EM Regret Poison Strategic
Known flip FL-Shapley 0.3685±0.0094 0.3136±0.0100 0.1343±0.0424 0.0686±0.0399 0.6 2.4
Known flip Leave-one-out 0.3428±0.0274 0.2978±0.0313 0.1257±0.0568 0.0943±0.0423 0.6 2.4
Known flip FedMark-FM 0.3914±0.0303 0.3490±0.0498 0.0914±0.0188 0.0457±0.0407 0.0 0.2
Hard paraphrase FL-Shapley 0.3600±0.0399 0.3158±0.0521 0.0972±0.0321 0.0800±0.0563 0.4 2.2
Hard paraphrase Leave-one-out 0.3457±0.0311 0.2999±0.0400 0.1171±0.0610 0.0914±0.0368 0.4 2.2
Hard paraphrase FedMark-FM 0.3857±0.0137 0.3463±0.0338 0.0828±0.0243 0.0543±0.0340 0.0 0.2
Adaptive keyword FL-Shapley 0.3714±0.0237 0.3252±0.0420 0.1029±0.0375 0.0657±0.0431 0.4 2.4
Adaptive keyword Leave-one-out 0.3514±0.0332 0.3029±0.0388 0.1457±0.0422 0.0857±0.0380 0.8 2.2
Adaptive keyword FedMark-FM 0.3857±0.0317 0.3437±0.0495 0.0886±0.0216 0.0514±0.0424 0.0 0.2
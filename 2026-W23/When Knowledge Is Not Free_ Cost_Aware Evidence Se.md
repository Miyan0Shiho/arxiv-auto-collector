# When Knowledge Is Not Free: Cost-Aware Evidence Selection in Retrieval-Augmented Generation

**Authors**: Mingyan Wu, Han Yang, Omer Ben-Porat, Yftah Ziser

**Published**: 2026-06-01 13:39:39

**PDF URL**: [https://arxiv.org/pdf/2606.02245v1](https://arxiv.org/pdf/2606.02245v1)

## Abstract
Retrieval-Augmented Generation (RAG) typically assumes that external knowledge is free, but many high-quality sources are paywalled, licensed, restricted, or otherwise costly to access. We introduce cost-aware RAG, a setting where retrieved evidence is assigned access-cost tiers and systems must answer under an explicit evidence-access budget. We instantiate this setting by augmenting MS MARCO v2.1 with access-friction tiers and evaluate budgeted evidence selection across general-domain and domain-specific QA benchmarks. Our results show that static selection is brittle: no fixed selector uniformly dominates, and larger budgets do not reliably improve answer quality, even when costly evidence is domain-matched. We then study agentic cost-aware RAG, where an LLM decides when to retrieve, which tier to access, and when to stop. Agents show strong promise as adaptive evidence-acquisition controllers, but their behavior remains highly model- and task-dependent. These findings suggest that cost-aware evidence acquisition is a central challenge for the next generation of RAG systems. All code and data are available at https://github.com/Mignonmy/Cost-Aware.

## Full Text


<!-- PDF content starts -->

When Knowledge Is Not Free: Cost-Aware Evidence Selection in
Retrieval-Augmented Generation
Mingyan Wu1*, Han Yang2,3*, Omer Ben-Porat4, Yftah Ziser5,6†
1Northeastern University,2Technical University of Munich,
3GESIS – Leibniz Institute for the Social Sciences,
4Technion–Israel Institute of Technology,5NVIDIA Research,6University of Groningen
Abstract
Retrieval-Augmented Generation (RAG) typi-
cally assumes that external knowledge is free,
but many high-quality sources are paywalled,
licensed, restricted, or otherwise costly to ac-
cess. We introduce cost-aware RAG, a setting
where retrieved evidence is assigned access-
cost tiers and systems must answer under an
explicit evidence-access budget. We instanti-
ate this setting by augmenting MS MARCO
v2.1 with access-friction tiers and evaluate
budgeted evidence selection across general-
domain and domain-specific QA benchmarks.
Our results show that static selection is brit-
tle: no fixed selector uniformly dominates,
and larger budgets do not reliably improve
answer quality, even when costly evidence
is domain-matched. We then study agen-
tic cost-aware RAG, where an LLM decides
when to retrieve, which tier to access, and
when to stop. Agents show strong promise
as adaptive evidence-acquisition controllers,
but their behavior remains highly model- and
task-dependent. These findings suggest that
cost-aware evidence acquisition is a central
challenge for the next generation of RAG sys-
tems. All code and data are available at https:
//github.com/Mignonmy/Cost-Aware.
1 Introduction
Retrieval-Augmented Generation (RAG) systems
are usually evaluated as if external knowledge were
freely and uniformly accessible. In practice, how-
ever, many high-quality sources are restricted, li-
censed, paywalled, or otherwise costly to access,
including academic journals, clinical references,
professional databases, licensed code repositories,
and proprietary business resources. This makes
retrieval not only a relevance-ranking problem, but
also a resource-allocation problem: a system must
decide which evidence is worth acquiring under a
*indicates equal contribution.
†indicates corresponding author.
Figure 1: Comparison between vanilla and cost-aware
RAG. (a) Vanilla RAG assumes that all retrieved evi-
dence is freely accessible. (b) Our setup assigns each
evidence item an access-cost tier and charges cost only
for evidence selected and provided to the generator un-
der budgetB.
limited access budget. This access cost is distinct
from commonly studied RAG constraints such as
computation time, context length, or token budgets
(Jeong et al., 2024; Yue et al., 2025; Lu et al., 2025).
Those constraints limit how much evidence can be
processed once it is available; access costs deter-
mine which evidence can be obtained in the first
place. Cost-aware RAG therefore requires estimat-
ing the value of evidence relative to its cost, rather
than optimizing retrieval relevance or generation
quality alone.
We study this setting by introducing a cost-
aware RAG evaluation framework that augmentsarXiv:2606.02245v1  [cs.CL]  1 Jun 2026

MS MARCO v2.1 with discrete access-cost tiers.
These tiers serve as coarse proxies for access fric-
tion, distinguishing freely accessible sources, gen-
eral open-web sources, and higher-friction profes-
sional or restricted sources. As shown in Figure 1,
each evidence item has an access cost, and the
system must select evidence under an explicit bud-
get before passing it to the generator. We evalu-
ate this framework on general-domain QA bench-
marks, including HotpotQA, Natural Questions
(NQ), and TriviaQA, and domain-specific medi-
cal QA benchmarks, including MedQA-US and
MMLU-Med. For the medical setting, we augment
the high-cost tier with domain-matched textbook
evidence, allowing us to test whether costly spe-
cialized sources help when relevant knowledge is
concentrated in higher-friction resources. We com-
pare static evidence-selection strategies, including
cost-agnostic top- k, relevance-only selection, cost-
aware greedy selection, knapsack-based optimiza-
tion, and redundancy-aware methods.
Our results show that cost-aware RAG does not
admit a single fixed selection rule. No static selec-
tor uniformly dominates across datasets, budgets,
and backbone models, and larger access budgets do
not reliably improve answer quality. Since the gen-
erator receives a bounded evidence context, increas-
ing the budget mainly changes the composition of
the selected evidence, allowing low-cost passages
to be replaced by higher-cost ones. Even in the
medical setting, where costly evidence is domain-
matched, a more expensive evidence mix is not nec-
essarily more useful. These findings motivate an
adaptive formulation. Rather than assigning every
query the same budget and applying a fixed selec-
tor, a cost-aware system should decide during in-
ference whether more evidence is needed, whether
higher-cost evidence is worth acquiring, and when
the current context is sufficient. We therefore study
an agentic formulation of cost-aware RAG, where
an LLM acts as an evidence-acquisition controller.
Agents can substantially improve the performance–
cost tradeoff with some backbones, but their behav-
ior remains model- and task-dependent, showing
that current LLMs are still imperfect cost-sensitive
evidence controllers.
Contributions.We make three contributions:
(i) We introduce a cost-aware RAG evaluation
setting by augmenting MS MARCO v2.1 with
discrete access-cost tiers that model relative ac-
cess friction. (ii) We benchmark static budget-Tier Access Level
0 Open community resources (e.g., Wikipedia, pub-
lic forums)
1 General open-web content (blogs, tutorials, news
sites)
2 Curated/official/professional sources with moder-
ate friction (official vendor docs & API refs, sup-
port KBs, standards/gov technical docs; may in-
clude free-login or soft gating)
3 High-friction restricted sources (paid paywalls, en-
terprise SSO portals, proprietary aggregators)
Table 1: Evidence cost tiers modeling relative access
friction.
constrained evidence-selection strategies across
general-domain and domain-specific QA bench-
marks, showing that no fixed selector uniformly
dominates and that larger access budgets do not reli-
ably improve answer quality. (iii) We study agentic
cost-aware evidence acquisition in per-query and
shared-budget settings, showing that agentic frame-
works are promising for cost-aware RAG while
revealing that current LLMs remain unreliable cost-
sensitive evidence controllers.
2 Constructing a Cost-Aware Retrieval
Corpus
2.1 Cost-annotated corpus.
We build on the MS MARCO v2.1 document col-
lection (Nguyen et al., 2016), which consists of
web-sourced passages with associated queries and
source URLs. We keep the original queries un-
changed, and augment each evidence item with an
access-cost label inferred from its originating do-
main. These costs are intended to model relative
access friction rather than literal monetary prices.
Specifically, each source domain is assigned to a
discrete cost tier, summarized in Table 1, rang-
ing from open community resources to general
open-web content and higher-friction curated, pro-
fessional, or restricted sources. This construction
enables budget-constrained retrieval and evalua-
tion while preserving the original MS MARCO
evidence and supervision.
2.2 Domain distribution analysis.
We first analyze the source-domain distribu-
tion of the MS MARCO v2.1 segment corpus.
The corpus contains 113,520,750 segments from
1,035,806 unique domains, and the distribution
is highly skewed. The most frequent domain,
en.wikipedia.org , contributes 7,006,110 seg-
ments, corresponding to 6.17% of the corpus, while

the top 30 domains together account for 19,178,928
segments, or 16.89% of all segments. This skew
motivates a domain-level annotation strategy: a
relatively small number of high-coverage domains
accounts for a substantial fraction of the corpus,
while most domains occur only in the long tail.
2.3 LLM-based domain classification.
Since manually labeling more than one million
domains is impractical, we annotate the highest-
coverage domains using an LLM-as-judge proce-
dure. We select the top 20,000 domains by seg-
ment count, which together cover 82,604,646 seg-
ments, or 72.77% of the full corpus. Given the
tier definitions in Table 1, GPT-4.1-mini assigns
each domain to an access-friction tier based on the
domain identity and its likely accessibility charac-
teristics. To avoid forcing arbitrary assignments,
the model is also allowed to abstain when the do-
main cannot be confidently classified; abstentions
are rare in practice (around 0.3%). The initial anno-
tation used a four-way rubric separating curated or
professional sources from high-friction restricted
sources. Under this rubric, the top 20,000 domains
were classified as follows: 1,904 domains in Tier
0 (9.52%), 14,350 in Tier 1 (71.75%), 2,715 in
Tier 2 (13.58%), and 1,031 in Tier 3 (5.16%). Be-
cause the two highest-friction classes are relatively
sparse, we merge the original Tiers 2 and 3 into the
final Tier 2 used in our experiments. This yields
a three-tier structure that preserves the distinction
between free, low-cost, and costly evidence while
making the highest-cost tier sufficiently populated
for evaluation.
2.4 Long-tail imputation and final tier
distribution.
The remaining domains form a large long tail with
low individual coverage. Rather than querying the
LLM for every rare domain, we impute their tiers
by sampling from the empirical tier proportions
observed among the annotated top 20,000 domains.
After propagating domain labels to all segments
and merging the two highest-friction tiers, the fi-
nal MS MARCO-only corpus contains 36,547,305
Tier 0 segments (32.19%), 62,632,643 Tier 1 seg-
ments (55.17%), and 14,340,802 Tier 2 segments
(12.64%), as shown in Table 2. In all budgeted re-
trieval experiments, these tiers are mapped to costs
of 0, 1, and 4, respectively.Tier # Segments Proportion
0 36,547,305 32.19%
1 62,632,643 55.17%
2 14,340,802 12.64%
Table 2: Final tier distribution for the cost-annotated MS
MARCO v2.1 segment corpus after long-tail imputation
and merging the two highest-friction tiers.
3 Experimental Setting
3.1 Models
We evaluate two open-weight language models:
Llama-3.1-8B-Instruct (Touvron et al., 2023) and
Qwen3-8B (Team, 2025). This allows us to
compare cost-aware retrieval behavior across two
widely used model families at a similar param-
eter scale. During retrieval, Qwen3-embedding-
0.6B (Zhang et al., 2025) is used to retrieve query-
related documents, and the number of retrieval doc-
uments is limited to the top 30.
3.2 Datasets
Retrieval corpusWe use the cost-annotated
MS MARCO v2.1 corpus as our main retrieval
source (Nguyen et al., 2016). Each passage in-
herits an access-cost label from its source domain,
capturing relative access friction rather than literal
monetary price. In our experiments, the final cost
tiers are mapped to costs 0,1, and 4, corresponding
to free, low-cost, and costly evidence, respectively.
Evaluation benchmarksWe evaluate on five
question-answering benchmarks that span both
open-domain and domain-specific settings. For
open-domain QA, we use HotpotQA (Yang et al.,
2018), Natural Questions (NQ) (Kwiatkowski et al.,
2019), and TriviaQA (Joshi et al., 2017). These
benchmarks require broad web knowledge and
cover different retrieval demands, from relatively
direct factoid questions to multi-hop reasoning over
multiple pieces of evidence. For domain-specific
QA, we use MedQA-US (Jin et al., 2020) and
MMLU-Med (Xiong et al., 2024).
To create a more realistic high-friction retrieval
scenario for the medical benchmarks, we aug-
ment the Tier-2 evidence pool for MedQA-US and
MMLU-Med with Textbooks (Jin et al., 2020),
the large medical textbook corpus released with
MedQA (Jin et al., 2020). We treat this corpus as
costly domain-specific evidence, simulating expert

medical sources such as scientific articles, text-
books, and clinical reference materials that are
more expensive to access but may provide higher-
quality information. This setup allows us to test
cost-aware evidence selection both when relevant
knowledge is broadly available on the web and
when it is concentrated in specialized, high-friction
sources. We provide further experimental details
in Appendix B.
3.3 Budgeted Retrieval Protocol
For each query q, we retrieve a candidate set of
passages and select a subset to provide to the gen-
erator under an evidence-access budget B. Each
candidate passage diis associated with a relevance
score vi, estimated by the similarity score between
the passage and the query, and an access cost ci,
determined by its source cost tier. Unless oth-
erwise stated, the generator receives at most 30
pieces of evidence per question. When the budget
is exhausted before reaching this evidence limit,
the remaining context slots can be filled with the
highest-ranked Tier-0 evidence, since these pas-
sages incur no additional access cost. We evaluate
answer quality as a function of the available budget,
measuring how different selection strategies trade
off relevance, cost, and redundancy.
3.4 Baselines and Evidence Selection Methods
BaselinesWe compare cost-aware selection
against two standard baselines. Thevanilla LLM
baseline answers without retrieved evidence. The
top-kbaseline provides the generator with the top-
kretrieved passages, ignoring access costs. These
baselines separate the effect of retrieval itself from
the effect of budgeted evidence selection.
Budgeted selectorsWe evaluate five budgeted
evidence-selection methods:
•Relevance-onlyranks passages by relevance
and selects the highest-scoring passages that
fit within the budget.
•Greedy cost-awareranks passages by cost-
normalized relevance, prioritizing high pre-
dicted utility per unit cost.
•Knapsackformulates selection as a 0/1 knap-
sack problem:
max
xX
ivixis.t.X
icixi≤B,(1)where xi∈ {0,1} indicates whether passage
diis selected. This directly maximizes pre-
dicted relevance under the budget constraint,
while assuming independent passage utilities.
•Redundancy-aware knapsackextends the
knapsack objective with a pairwise redun-
dancy penalty:
max
xX
ivixi−ηX
i<jsijxixjs.t.X
icixi≤B,
(2)
where sijdenotes the similarity between pas-
sages and ηcontrols the strength of the redun-
dancy penalty. This encourages the selector to
spend budget on complementary rather than
near-duplicate evidence. Theηis 0.2.
•Maximum Marginal Relevance (MMR)
with cost penaltyperforms sequential set-
aware selection. Given the selected set S, each
remaining passage is scored as
mi(S) =αr(q, d i)−(1−α) max
dj∈Ss(di, dj)−γc i,(3)
where r(q, d i)is the query–passage relevance
score, s(di, dj)is passage similarity, and γ
controls the cost penalty. αis 0.7 and γis 0.1.
This method balances relevance, diversity, and
access cost at each selection step.
4 Results
We first ask whether cost-aware evidence selection
can be solved by a fixed budgeted selector. The
results suggest not: across datasets, budgets, and
backbone models, no single rule consistently dom-
inates, and increasing the evidence-access budget
often fails to improve performance. These find-
ings motivate a more adaptive formulation, where
the system decides how much to spend and which
evidence tier to access on a per-query basis.
No fixed selector uniformly dominates.Fig-
ure 2 shows that the best selector depends strongly
on the dataset, budget, and backbone model. With
Qwen3-8B, Greedy is best on HotpotQA and NQ,
reaching 40.90 and 45.19 F1 at B= 15 , while
Knapsack is best on TriviaQA and Redundancy
on MedQA. On MMLU, several methods reach the
best score of 84.00 at different budgets. The pattern
changes with Llama3.1-8B-Instruct: Relevance is
best on HotpotQA and TriviaQA, Knapsack on NQ,
and Redundancy on MedQA. Thus, the relative

051015202530
30.032.535.037.540.042.5
HotpotQA
051015202530
34363840424446
NQ
051015202530
52.555.057.560.062.565.0
TriviaQA
051015202530
60626466687072
MedQA
051015202530
74767880828486
MMLUF1 (%)
BudgetRelevance Greedy Knapsack MMR Redundancy(a) Performance on Qwen3-8B.
051015202530
303540
HotpotQA
051015202530
303234363840
NQ
051015202530
54565860626466
TriviaQA
051015202530
45.047.550.052.555.057.560.0
MedQA
051015202530
50607080
MMLUF1 (%)
BudgetRelevance Greedy Knapsack MMR Redundancy
(b) Performance on Llama3.1-8B-instruct.
Figure 2: Overall Performance. For more detailed results, please refer to Appendix C.
050100NQRelevance Greedy Knapsack MMR Redundancy
051015202530
050100MedQA
051015202530
051015202530
051015202530
051015202530
Dataset / Tier Distribution (%)
BudgetT0 T1 T2 Wiki
Figure 3: Tier Distribution. Wiki belongs to T0. We conduct analysis experiments using the Qwen3-8B model. The
observation in Llama3.1-8B-instruct is the same, since the passages are selected by rule-based selectors. For more
detailed results, please refer to Appendix C.
ranking of selectors is unstable: the same objective
can be competitive in one setting and suboptimal
in another.
More budget is not necessarily better.Since
the generator receives a fixed evidence window, in-
creasing Bmainly changes the composition of the
selected evidence rather than the amount of context.
The relevant question is therefore whether a more
expensive evidence mix improves answer quality.
It does not reliably do so. Across the five selec-
tors and five datasets, none of the 25 Qwen3-8B
curves is monotonically non-decreasing over bud-
gets{0,5,10,15,20,25,30} , and the same holds
for Llama3.1-8B-Instruct. Moreover, only 6 out
of 25 Qwen3-8B curves and 4 out of 25 Llama3.1-
8B-Instruct curves achieve their best finite-budget
score at B= 30 . This remains true even inthe domain-specific benchmarks, where the high-
cost tier includes domain-matched medical evi-
dence. With Qwen3-8B on MMLU-Med, Rele-
vance reaches 84.00 F1 at B= 5 but drops to
79.00 at B= 30 , while Greedy reaches 84.00 at
B= 10 andB= 25 but drops to 78.00 at B= 30 .
On MedQA, Relevance and Knapsack also peak
before the largest budget. Thus, even when costly
evidence is plausibly useful, simply allowing more
of it into the context is not enough.
Budget changes evidence composition differ-
ently across domains.Figure 3 helps explain
the non-monotonic trends. In the open-domain
setting, represented by NQ, larger budgets do not
necessarily lead to substantially greater use of the
highest-cost tier; many methods continue to rely
on lower-cost evidence. This suggests that general-

knowledge questions can often be answered from
open or low-cost sources. In contrast, for MedQA,
where the costly tier contains domain-specialized
medical evidence, higher budgets often lead selec-
tors to use more high-cost evidence. The value
of expensive evidence is therefore task-dependent:
conservative policies may miss useful specialized
evidence, while aggressive policies may waste bud-
get when low-cost evidence is sufficient.
Reducing the number of evidence passages is not
sufficient.As shown in Table 12 and 13, the cost-
agnostic top- kbaseline controls for context size
by providing only the khighest-ranked passages,
without considering access cost. If the weakness
of fixed selectors were simply that they include too
many passages, then some fixed kshould perform
consistently well. The results do not support this
explanation. Top- ksometimes benefits from using
fewer than 30 passages, but no value of kis con-
sistently strong across datasets or models. With
Qwen3-8B, the best top- kscores on HotpotQA
and NQ are 37.67 and 40.09 F1, below the best
budgeted-selector scores of 40.90 and 45.19 F1.
With Llama3.1-8B-Instruct, top- kalso falls short
on HotpotQA and NQ. Thus, shorter contexts alone
do not explain the gains; the harder problem is de-
ciding which evidence is worth including and from
which cost tier.
Implication: cost-aware RAG requires adaptive
allocation.Taken together, these results show
that cost-aware evidence selection is difficult to
reduce to a single fixed rule. The best selector
varies across datasets, models, and budgets; larger
budgets do not reliably help; and the usefulness of
costly evidence depends on the domain and query.
This motivates a sequential formulation in which
the system assesses the evidence already acquired,
decides whether it is sufficient, and determines
whether additional or higher-cost evidence is worth
the expense. We therefore turn to an agentic formu-
lation of cost-aware RAG, where evidence acqui-
sition is treated as an online decision process: the
model must decide not only what to retrieve, but
also when to spend, where to spend, and when to
stop.
5 Agentic Cost-Aware Evidence
Acquisition
The results in Section 4 suggest that fixed budgeted
selectors are too brittle for cost-aware RAG. We
Figure 4: Agentic cost-aware evidence acquisition loop.
therefore study an LLM-based retrieval agent as
a testbed for adaptive evidence acquisition. At
inference time, the agent must decide whether to
retrieve, which cost tier to access, and when the
accumulated evidence is sufficient to answer. This
setting is more flexible than static selection, but
it also introduces new failure modes: the agent
may stop prematurely, spend too aggressively, or
retrieve from an inappropriate tier. We therefore
view agentic RAG not as a complete solution, but
as a probe of whether current LLMs can act as
cost-sensitive evidence controllers.
Design.We instantiate the agent with a ReAct-
style loop (Yao et al., 2023). At each step, the
agent observes the question, collected evidence, re-
maining budget, and available tiers with their costs.
It then chooses between two actions: RETRIEVE,
which specifies a tier and returns one passage from
that tier ranked by query similarity, or ANSWER,
which terminates the loop (see Figure 4). The agent
uses the same corpus, retriever, and cost tiers as
the fixed-selector experiments; only the acquisition
policy changes. The prompt instructs the agent to
obey the budget and answer only when the evidence
is sufficient or no useful retrieval is affordable. The
full prompt is shown in Table 4. Unless other-
wise stated, we use a per-query budget of B= 20
with the same two backbones as before, Qwen3-8B
and Llama3.1-8B-Instruct. The budget is an upper
bound; we report the average realized cost. We also

Figure 5: Agent Overall Performance. We compare agent performance with evidence selection method baselines.
(a) Average Cost
 (b) F1
Figure 6: Cost and F1 with Increasing Budget for Agent
on NQ and MedQA.
evaluate ashared-budgetvariant in which the agent
receives batches of five questions and allocates a
total budget of 100 before answering. This diag-
nostic setting tests whether the model can identify,
at the workload level, which questions are likely
to require costly evidence. Additional details are
provided in Appendix B.
Per-query agents can be efficient evidence con-
trollers.Figure 5 compares per-query agents
with fixed budgeted selectors. The main result is
that agentic acquisition can often match or exceed
fixed selectors while using far fewer passages and
much lower realized cost. This improvement is not
explained by shorter contexts alone: as shown in
Section 4, cost-agnostic top- kselection does not
consistently perform well. The clearest gains come
from Qwen3-8B. On NQ, the agent reaches 49.93
F1 with only 0.50 average cost and 2.47 retrieved
passages, outperforming all fixed selectors while
spending far less than the high-budget settings. On
TriviaQA, it reaches 66.73 F1 with 0.52 cost and
2.60 passages, and on MMLU-Med it reaches 85.00F1 with 2.90 cost and only 1.14 passages. On
MedQA, it obtains 68.00 F1 with 5.08 cost and
1.28 passages, competitive with the best fixed selec-
tors despite using a much smaller evidence context.
These results suggest that the agent is not simply
retrieving more evidence; rather, it often stops after
acquiring a small amount of useful evidence.
Agentic cost awareness is model- and task-
dependent.Agentic control is not automatically
cost-aware. HotpotQA remains difficult: with
Qwen3-8B and B= 20 , the agent reaches only
30.34 F1, below the best fixed selector. This sug-
gests that multi-hop questions may require more
than repeated retrieval with the original query, such
as query decomposition, sub-question planning, or
targeted query rewriting.Backbone choice also
matters. Llama3.1-8B-Instruct can achieve com-
petitive scores, but often by spending much more.
With B= 20 , it reaches 46.77 F1 on NQ using
13.47 average cost and 12.54 passages, and 68.40
F1 on TriviaQA using 11.53 cost and 11.53 pas-
sages. On MedQA, it reaches 60.00 F1 with 18.39
cost, and on MMLU-Med it reaches only 72.00 F1,
substantially below the best fixed selectors. In con-
trast to Qwen3-8B, the Llama agent often treats the
available budget as a resource to consume rather
than a constraint to manage. Thus, the key chal-
lenge is not only choosing the right tier, but also
estimating when the current evidence is sufficient.
Budget sensitivity reveals evidence-sufficiency
failures.A cost-aware agent should not spend
more merely because more budget is available. We
therefore vary the per-query budget on NQ and
MedQA, covering both open-domain and domain-

specific settings. Figure 6 shows that increasing the
budget has limited effect on final F1, but substan-
tially changes spending behaviour. Qwen3-8B ex-
hibits relatively stable stopping behaviour. Increas-
ing the budget from B= 20 toB= 30 changes
its average cost from 0.50 to 1.06 on NQ and from
5.08 to 4.72 on MedQA, while F1 changes from
49.93 to 45.20 and from 68.00 to 69.00. In con-
trast, Llama3.1-8B-Instruct spends much more as
the budget increases, without corresponding gains:
from B= 20 toB= 30 , its cost rises from
13.47 to 15.54 on NQ and from 18.39 to 27.01
on MedQA, while F1 drops from 46.77 to 42.89
and from 60.00 to 56.00. These trends indicate that
evidence-sufficiency estimation is a central bottle-
neck for agentic cost-aware RAG. Representative
trajectories are provided in Appendix C and a cor-
responding case study comparing this behaviour
across both models is presented in Appendix D.
Shared-budget allocation probes workload-level
planning.We next evaluate shared-budget allo-
cation as a diagnostic for workload-level planning.
Unlike the per-query setting, the agent receives
a batch of questions and must decide in advance
which ones deserve more budget. For Qwen3-
8B, this improves the performance–cost tradeoff
on most datasets: compared with the per-query
B= 20 agent, it reduces cost and passage usage on
HotpotQA, TriviaQA, MedQA, and MMLU-Med
while improving or maintaining F1. For example,
TriviaQA improves from 66.73 to 70.27 F1 while
cost drops from 0.52 to 0.33; MedQA improves
from 68.00 to 71.00 F1 while cost drops from
5.08 to 4.38; and MMLU-Med maintains 85.00 F1
while reducing cost from 2.90 to 2.56. However,
the benefit is not uniform: on NQ, Qwen spends
more while performing worse, and for Llama3.1-
8B-Instruct, lower cost and passage usage often
come with lower F1. Thus, shared-budget alloca-
tion can help when the model estimates question
difficulty well, but it also introduces a new failure
mode: the agent must predict before retrieval which
questions are worth spending on.
Takeaway.Agentic frameworks naturally fit cost-
aware RAG: each retrieval decision can be con-
ditioned on the question, the evidence collected
so far, and the remaining budget. This adaptivity
can improve the performance–cost tradeoff when
the model recognizes that a small amount of ev-
idence is sufficient. However, current agents re-
main unreliable cost-sensitive controllers: theirbehaviour varies with the backbone, task struc-
ture, and evidence-sufficiency judgments. Cost-
aware RAG therefore highlights a capability un-
derexplored in standard RAG: deciding whether
additional evidence is worth its access cost.
6 Related Work
RAG under resource constraints.Most RAG
work assumes that retrieved evidence is freely ac-
cessible (Lewis et al., 2020; Guu et al., 2020;
Izacard and Grave, 2021). Recent work stud-
ies resource-constrained RAG by limiting context
length, token usage, latency, inference-time com-
putation, or the number of retrieved chunks (Jeong
et al., 2024; Yue et al., 2025; Lu et al., 2025). Re-
lated budgeted-RAG work attaches cost to later
stages of the pipeline: CORAG selects among
already-retrieved chunks under a cost constraint
(Wang et al., 2024), while Chunk-as-a-Service stud-
ies pricing for chunks used to enrich prompts (Al-
Maliki et al., 2026). Our setting moves the cost ear-
lier: evidence may be costly to access depending
on its source, so the system must decide whether
a higher-cost source is worth acquiring before its
evidence can be used.
Adaptive and agentic RAG.Adaptive and agen-
tic RAG methods make retrieval conditional rather
than fixed: Adaptive-RAG routes queries by es-
timated complexity (Jeong et al., 2024), Active-
RAG/FLARE and Self-RAG trigger retrieval dur-
ing generation (Jiang et al., 2023; Asai et al., 2024),
and ReAct interleaves reasoning with actions and
observations (Yao et al., 2023). These methods
motivate our agentic formulation, but they do not
study evidence access costs. In our setting, the
model must decide not only whether more evidence
is needed, but whether it is worth acquiring from a
low- or high-cost source.
7 Conclusion
We introduced cost-aware RAG, where evidence
carries source-dependent access costs and retrieval
becomes a budgeted acquisition problem. Across
open-domain and domain-specific QA, fixed se-
lectors are brittle: no rule consistently dominates,
and higher budgets do not reliably improve answer
quality. Agentic RAG offers a promising alterna-
tive by deciding when to retrieve, which tier to
access, and when to stop, but current agents remain
highly model- and task-dependent.

8 Limitations
Simulated access costs.Our access-cost tiers are
proxies for relative access friction rather than direct
measurements of monetary cost, licensing restric-
tions, latency, or institutional access. We assign
costs at the domain level and map them to a small
number of discrete tiers, which enables large-scale
evaluation but cannot capture variation within the
same source.
Automatic tier annotation.Our cost annotations
rely on automatic domain classification and long-
tail imputation. The LLM judge may misclassify
some domains, and imputed labels for rare domains
may not reflect their true access properties. These
errors could affect the apparent value of high-cost
evidence and the behavior of cost-aware selectors.
Evaluation scope.Our evaluation is limited to
five QA benchmarks and sampled subsets. This al-
lows controlled comparisons across selectors, bud-
gets, and backbones, but leaves open how stable the
conclusions are on larger samples, additional do-
mains, and different answer formats. The medical
benchmarks should be viewed as domain-specific
QA testbeds rather than evidence of clinical readi-
ness.
Simplified retrieval environment.The agent re-
trieves from a fixed candidate pool using the orig-
inal question and receives one passage at a time
from a selected tier. It does not perform query
rewriting, sub-question decomposition, source ver-
ification, or multi-query search, which likely limits
performance on multi-hop questions.
Zero-shot agent control.Our agentic controller
is evaluated in a zero-shot prompting regime with
two open-weight backbones. The results show that
agentic cost-aware RAG can be efficient, but stop-
ping and spending behavior remains highly model-
dependent. We therefore view the agentic results as
a diagnostic of current LLMs as cost-sensitive evi-
dence controllers, rather than as a fully optimized
policy.
Acknowledgement
Han Yang received funding from the Deutsche
Forschungsgemeinschaft (DFG) under grant num-
ber: MA 3964/15-3 (SocioHub project). Han Yang
received additional funding from the European
Union under the Horizon Europe grant OMINO –Overcoming Multilevel INformation Overload1un-
der grant number 101086321 (Hołyst et al., 2024).
References
Shawqi Al-Maliki, Ammar Gharaibeh, Mohamed Ra-
houti, Mohammad Ruhul Amin, Mohamed M. Ab-
dallah, Junaid Qadir, and Ala I. Al-Fuqaha. 2026.
Budget-constrained online retrieval-augmented gen-
eration: The chunk-as-a-service model.CoRR,
abs/2604.26981.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations, ICLR 2024, Vienna, Austria, May
7-11, 2024. OpenReview.net.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat,
and Ming-Wei Chang. 2020. Retrieval augmented
language model pre-training. InProceedings of the
37th International Conference on Machine Learning,
ICML 2020, 13-18 July 2020, Virtual Event, Pro-
ceedings of Machine Learning Research, pages 3929–
3938. PMLR.
Janusz A Hołyst, Philipp Mayr, Michael Thelwall, Ingo
Frommholz, Shlomo Havlin, Alon Sela, Yoed N
Kenett, Denis Helic, Aljoša Rehar, Sebastijan R
Maˇcek, and 1 others. 2024. Protect our environ-
ment from information overload.Nature Human
Behaviour, 8(3):402–403.
Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open do-
main question answering. InProceedings of the 16th
Conference of the European Chapter of the Associ-
ation for Computational Linguistics: Main Volume,
EACL 2021, Online, April 19 - 23, 2021, pages 874–
880. Association for Computational Linguistics.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong Park. 2024. Adaptive-rag: Learn-
ing to adapt retrieval-augmented large language mod-
els through question complexity. InProceedings of
the 2024 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long
Papers), NAACL 2024, Mexico City, Mexico, June
16-21, 2024, pages 7036–7050. Association for Com-
putational Linguistics.
Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, EMNLP 2023, Singapore, Decem-
ber 6-10, 2023, pages 7969–7992. Association for
Computational Linguistics.
1https://ominoproject.eu/

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng,
Hanyi Fang, and Peter Szolovits. 2020. What dis-
ease does this patient have? A large-scale open do-
main question answering dataset from medical exams.
CoRR, abs/2009.13081.
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke
Zettlemoyer. 2017. TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. InProceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1601–1611, Vancouver,
Canada. Association for Computational Linguistics.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research.Transactions of the Association for Compu-
tational Linguistics, 7:452–466.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonza-
lez, Hao Zhang, and Ion Stoica. 2023. Efficient mem-
ory management for large language model serving
with pagedattention. InProceedings of the 29th Sym-
posium on Operating Systems Principles, SOSP 2023,
Koblenz, Germany, October 23-26, 2023, pages 611–
626. ACM.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural In-
formation Processing Systems 33: Annual Confer-
ence on Neural Information Processing Systems 2020,
NeurIPS 2020, December 6-12, 2020, virtual.
Songshuo Lu, Hua Wang, Yutian Rong, Zhi Chen,
and Yaohua Tang. 2025. TurboRAG: Accelerating
retrieval-augmented generation with precomputed
KV caches for chunked text. InProceedings of the
2025 Conference on Empirical Methods in Natural
Language Processing, pages 6588–6601, Suzhou,
China. Association for Computational Linguistics.
Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao,
Saurabh Tiwary, Rangan Majumder, and Li Deng.
2016. MS MARCO: A human generated machine
reading comprehension dataset. InProceedings of
the Workshop on Cognitive Computation: Integrat-
ing neural and symbolic approaches 2016 co-located
with the 30th Annual Conference on Neural Infor-
mation Processing Systems (NIPS 2016), Barcelona,
Spain, December 9, 2016, CEUR Workshop Proceed-
ings. CEUR-WS.org.
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick
Lewis, Majid Yazdani, Nicola De Cao, James Thorne,
Yacine Jernite, Vladimir Karpukhin, Jean Maillard,Vassilis Plachouras, Tim Rocktäschel, and Sebastian
Riedel. 2021. KILT: a benchmark for knowledge
intensive language tasks. InProceedings of the 2021
Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies, NAACL-HLT 2021, Online,
June 6-11, 2021, pages 2523–2544. Association for
Computational Linguistics.
Qwen Team. 2025. Qwen3 technical report.CoRR,
abs/2505.09388.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, Aurélien Rodriguez, Armand Joulin, Edouard
Grave, and Guillaume Lample. 2023. Llama: Open
and efficient foundation language models.CoRR,
abs/2302.13971.
Ruobing Wang, Qingfei Zhao, Yukun Yan, Daren Zha,
Yuxuan Chen, Shi Yu, Zhenghao Liu, Yixuan Wang,
Shuo Wang, Xu Han, Zhiyuan Liu, and Maosong
Sun. 2025. Deepnote: Note-centric deep retrieval-
augmented generation. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2025,
Suzhou, China, November 4-9, 2025, pages 19688–
19715. Association for Computational Linguistics.
Ziting Wang, Haitao Yuan, Wei Dong, Gao Cong, and
Feifei Li. 2024. CORAG: A cost-constrained re-
trieval optimization system for retrieval-augmented
generation.CoRR, abs/2411.00744.
Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong
Zhang. 2024. Benchmarking retrieval-augmented
generation for medicine. InFindings of the Asso-
ciation for Computational Linguistics, ACL 2024,
Bangkok, Thailand and virtual meeting, August 11-
16, 2024, Findings of ACL, pages 6233–6251. Asso-
ciation for Computational Linguistics.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing. InProceedings of the 2018 Conference on Em-
pirical Methods in Natural Language Processing,
Brussels, Belgium, October 31 - November 4, 2018,
pages 2369–2380. Association for Computational
Linguistics.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023.
React: Synergizing reasoning and acting in language
models. InThe Eleventh International Conference
on Learning Representations, ICLR 2023, Kigali,
Rwanda, May 1-5, 2023. OpenReview.net.
Zhenrui Yue, Honglei Zhuang, Aijun Bai, Kai Hui, Rolf
Jagerman, Hansi Zeng, Zhen Qin, Dong Wang, Xuan-
hui Wang, and Michael Bendersky. 2025. Inference
scaling for long-context retrieval augmented gener-
ation. InThe Thirteenth International Conference
on Learning Representations, ICLR 2025, Singapore,
April 24-28, 2025. OpenReview.net.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
CoRR, abs/2506.05176.

A License
This section summarizes the licenses of the datasets
used in our experiments.
All datasets used in this work permit academic
use under their respective licenses and agreements:
Natural Questions (CC-BY-SA-3.0 License); Hot-
potQA (CC-BY-4.0 License); TriviaQA (Apache
2.0 License); MedQA-US and MMLU-Med are un-
der the terms of the United States Copyright Act.
B Additional Experimental Details
Implementation Details.We conducted our ex-
periments on an NVIDIA H100 NVL GPU with 94
GB of memory. We used vLLM (Kwon et al., 2023)
as the inference engine to accelerate the inference
process.
Corpus Dataset.We use the MS MARCO v2.12
corpus as our main retrieval source (Nguyen et al.,
2016) for general QA tasks. For domain-specific
tasks we use Textbooks3(Jin et al., 2020) as knowl-
edge base. Textbooks is a collection of 18 widely
used English medical books, which were originally
collected in PDF format and converted into digital
text via OCR followed by clean-up processing. The
dataset contains 231,581 paragraphs with an aver-
age paragraph length of 55.0 and a maximum para-
graph length of 1,234. We select this corpus for our
domain-specific tasks because it has been shown
to effectively support MedQA and MMLU-Med,
according to (Xiong et al., 2024). We access the
Textbooks corpora in version of MIRAGE4(Xiong
et al., 2024).
Evaluation Benchmarks & Metrics.We eval-
uate on five question-answering benchmarks that
span both open-domain and domain-specific set-
tings. For open-domain QA, we use Hot-
potQA (Yang et al., 2018), Natural Ques-
tions (Kwiatkowski et al., 2019), and Trivi-
aQA (Joshi et al., 2017). For domain-specific QA,
we use MedQA-US (Jin et al., 2020) and MMLU-
Med (Xiong et al., 2024).
We have access to general-domain tasks via the
KILT Benchmarking5(Petroni et al., 2021), and
domain-specific tasks from their individual GitHub
2https://trec-rag.github.io/annoucements/
2024-corpus-finalization/
3https://github.com/jind11/MedQA
4https://github.com/gzxiong/MIRAGE
5https://trec-rag.github.io/annoucements/
2024-corpus-finalization/
Figure 7: Number of Passages and EM with Increasing
Budget for Agent on NQ and MedQA. Left: average
number of passages; Right: EM score.
repositories. Following Wang et al. (2025) and con-
sidering our computational resources and the diffi-
culty of conducting case-by-case investigation, we
randomly sample 100 questions from each bench-
mark. We evaluate performance using Token-Level
F1 (F1) and Exact Match (EM). While the F1 score
measures the trade-off between coverage and redun-
dancy by allowing partial matches, EM imposes
a stricter criterion, requiring the model’s output
to perfectly match the reference answer. If the
benchmark provides multiple reference answers,
we adopt the highest score. For each benchmark
and method, token-level F1 is first computed for
each question and then averaged over the 100 sam-
pled questions. We compute the 95% confidence
interval for the mean F1 using a normal approxima-
tion, where the standard error is estimated from the
sample variance of question-level F1 scores. Val-
ues in the tables are reported as the mean F1 ±the
corresponding 95% confidence-interval margin.
Hyperparameter Search for Budgeted Selectors.
We present the hyperparameter selection for the
Budget Selectors discussed in Section 3.4. In
Equation 2, ηcontrols the strength of the redun-
dancy penalty in the Redundancy-aware Knapsack
method. The hyperparameters αandγin Equa-
tion 3 balance query–passage relevance against
passage similarity and control the cost penalty, re-
spectively. Figure 8 reports the F1 scores of these
methods under different hyperparameter settings on
the NQ dataset. The results show that the best over-
all performance on both benchmarks is achieved
withη= 0.2 for Redundancy-aware Knapsack,
and withα= 0.7andγ= 0.1for MMR.
Prompt Templates in Agentic Framework.We
provide the full prompts used for the agent in Ta-
ble 4 and Table 5. Specifically, Table 4 presents the
prompt for the agent with a per-query budget, while

Model Budget Strategy Dataset Avg. Cost Avg. Psg Avg. Steps EM (%) F1 (%)
Qwen3-8BPer-query Budget,B= 20HotpotQA 1.38 6.35 7.41 20.0030.34 ±8.15
NQ 0.50 2.47 3.49 37.0049.93 ±8.65
TriviaQA 0.52 2.60 3.61 64.0066.73 ±9.05
MedQA 5.08 1.28 2.29 68.0068.00 ±9.19
MMLU 2.90 1.14 2.14 85.0085.00 ±7.03
Shared BudgetHotpotQA 1.08 5.66 6.70 25.0035.30 ±8.61
NQ 1.14 2.21 3.21 33.0045.52 ±8.72
TriviaQA 0.33 2.36 3.37 66.0070.27 ±8.63
MedQA 4.38 1.13 2.13 71.0071.00 ±8.94
MMLU 2.56 0.86 1.86 85.0085.00 ±7.03
Llama3.1-8B-InstructPer-query Budget,B= 20HotpotQA 13.97 13.16 14.56 15.0025.21 ±7.68
NQ 13.47 12.54 13.74 35.0046.77 ±8.64
TriviaQA 11.53 11.53 12.81 63.0068.40 ±8.70
MedQA 18.39 4.84 5.85 60.0060.00 ±9.65
MMLU 15.74 7.02 8.11 72.0072.00 ±8.84
Shared BudgetHotpotQA 10.97 10.12 11.40 15.0025.99 ±7.69
NQ 10.00 9.87 11.04 28.0040.41 ±8.59
TriviaQA 9.02 8.80 10.06 63.0066.87 ±8.94
MedQA 17.84 4.72 5.74 55.0055.00 ±9.80
MMLU 16.04 6.28 7.35 72.0072.00 ±8.84
Table 3: Overall Performance of Agent across Backbone Models and Budget Strategies. For per-query budget, the
budget (B) is 20; for shared budget, we set the budget as 100 per batch.
(a)
 (b)
 (c)
Figure 8: Hyperparameter search for Budgeted Selectors. Subfigure 8a shows the selection of ηfor the Redundancy-
aware Knapsack method. Subfigure 8b shows the selection of αfor MMR method. Subfigure 8c shows the selection
ofγfor MMR method.
Table 5 presents the prompt for budget allocation
in a shared budget setting.
Tier Annotation Consistency with Human.We
manually inspected 30 randomly sampled tier as-
signments produced by GPT-4.1-mini. Most labels
were considered reasonable, and disagreements
mainly occurred between adjacent tiers, suggest-
ing that the generated tier annotations are generally
consistent for our analysis purposes.
C Additional Experimental Results
Full Result of Evidence Selection Methods.
Figure 2 shows different evidence selection meth-
ods across datasets and backbone models. We pro-
vide more detailed overall performance Tables in
Table 12 and 13, containing Vanilla LLM and TopkRAG performance methods. In addition, Figure 3
presents the tier distribution of retrieved documents
on NQ and MedQA datasets. We provide the full
result of tier distribution across all datasets in Fig-
ure 9.
Full Result of Agentic framework.Figure 5
presents the agent performance compared with evi-
dence selection baselines. We report more detailed
results of the agent in Table 3, containing Avg.
Cost, Avg. Psg, and Avg. Steps.
In Section 5, we discuss the sensitivity of the
agent to increasing budgets. Further analyze the
impact on the average number of retrieved passages
and the EM score in Figure 7. We observe a similar
trend, that the Llama3.1-8B-Instruct model tends
to retrieve more passages with increasing budget,

while the Qwen3-8B model remains stable. Also,
the increasing budget does not lead to a higher EM
score.
Additional Experiment with Alternative Pricing
Setting.In our main experimental setting, we
employ the pricing setting where tier 0 is free, tier 1
costs 1, and tier 2 costs 4. We additionally conduct
experiments under an alternative pricing setting,
where tier 0 is free, tier 1 costs 1, and tier 2 costs
2. As shown in Table 14, the main conclusions are
aligned with the results observed in Section 4.
D Case Studies: Agents’ Cost-aware
Decision Making
We illustrate how the agent processes tasks. In the
first case, as shown in Table 6, the agent first iden-
tifies that the question requires external evidence
about a specific historical election and therefore
begins with tier 0, the free retrieval source. When
the first retrieved passage fails to provide informa-
tion about the target year, it continues retrieving
from tier 0, since the needed fact is likely available
in encyclopedic sources. The second retrieval pro-
vides sufficient evidence that Eliot Cutler finished
ahead of Libby Mitchell in Maine’s 2010 guberna-
torial election, allowing the agent to answer cor-
rectly. This example shows that the agent performs
tier-aware evidence acquisition. It keeps retrieving
when evidence is insufficient, avoids unnecessarily
expensive sources, and reaches the correct answer.
In the second case, as shown in Table 7, the agent
handles a question that requires specific and up-
to-date factual evidence about Mount Ruapehu’s
most recent eruption. The agent first retrieves from
tier 0, since the question asks for a factual answer
that may be available in free community resources.
However, the initial passages only provide histori-
cal information, rather than the exact date of the lat-
est eruption. After multiple unsuccessful attempts
from tier 0 and tier 1, the agent decides to explore
a higher-cost tier, since cheaper sources have not
yet provided the required evidence and the remain-
ing budget is still sufficient. Although the tier 2
retrieval also fails to directly answer the question,
the agent does not stop prematurely. Instead, it
continues retrieval and eventually finds a useful
tier 0 passage that explicitly states the last eruption
date as 25 September 2007. This case shows that
the agent can adaptively balance cost and evidence
quality. It incurs a total cost of 7 and switches
across all three tiers, but eventually retrieves thenecessary passage and stops retrieving once suffi-
cient evidence is found.
In the third case, as shown in Table 8, the agent
handles a domain-specific medical question and
correctly identifies tier 2 as the appropriate source
for professional medical knowledge. Since the re-
trieved passages do not fully specify the protocol
for this situation, the agent remains uncertain and
continues searching until the budget is exhausted.
With no remaining budget, the agent then combines
the incomplete evidence with its own parametric
knowledge and reasoning to make a correct final
decision. This case shows that the agent can iden-
tify the appropriate knowledge tier, understand the
trade-off between task difficulty and budget, and
continue evidence acquisition when confidence is
insufficient.
We further demonstrate how the agent performs
flexible budget allocation under the shared-budget
setting. As shown in Table 9, the agent first esti-
mates the domain, difficulty, and likely retrieval
tier of each question. It assigns smaller budgets to
questions that can likely be answered by public or
low-cost sources, while reserving larger budgets
for questions that require more specific historical
or scientific knowledge. The agent then iteratively
adjusts the allocation until the total budget exactly
matches the shared budget constraint. This case
shows that the agent can reason about the relative
difficulty of multiple questions and distribute the
available budget accordingly.
We further compare cost-aware behaviours
across different backbone models. As shown in Ta-
ble 10 and Table 11, both agents identify the ques-
tion as requiring domain-specific medical knowl-
edge and therefore select tier 2 for retrieval. How-
ever, they differ substantially in when to stop re-
trieving. The agent with Qwen3-8B retrieves only
one tier 2 passage, recognizes that the evidence
is sufficient, and stops with a total cost of 4. In
contrast, the agent with Llama3.1-8B-Instruct re-
peatedly retrieves from tier 2 despite already having
relevant evidence, and only answers after exhaust-
ing the full budget of 40. This comparison suggests
that cost-aware behaviour is strongly influenced by
the backbone model.

050100RelevanceHotpotQA NQ TriviaQA MedQA MMLU
050100Greedy
050100Knapsack
050100MMR
051015202530
050100Redundancy
051015202530
051015202530
051015202530
051015202530
Method / Tier Distribution (%)
BudgetT0 T1 T2 WikiFigure 9: Full Tier Distribution. Wiki belongs to T0. We conduct analysis experiments using the Qwen3-8B model.
The observation in Llama3.1-8B-instruct is the same, since the passages are selected by rule-based selectors.

Prompt
You are a budgeted retrieval-augmented QA
agent.
At each step, first decide whether the question
requires external knowledge beyond the current
context.
If external knowledge is needed and budget
allows, retrieve exactly ONE passage from the
most appropriate tier.
Actually, you can retrieve the passages from
the same tier many times.
Only answer when the current context is
sufficient, the question can be answered
confidently without retrieval,
or the remaining budget cannot support a useful
retrieval.
Do not answer merely because the current context
is empty; at the beginning, empty context is
normal.
Follow the budget strictly. Output MUST be a
single JSON object.
Budget: <budget> | Spent: <spent> | Remaining:
<remaining>
Question: <question>
Tier options: <tier_desc>
Trajectory so far: <traj_block>
Current context: <ctx_block>
Decision objective:
- First judge whether answering requires
outside knowledge.
- If outside knowledge is needed, retrieve one
passage from the best tier.
- Continue retrieving across steps until the
evidence is sufficient, or the budget is
exhausted.
- If the question can already be answered
confidently from the question itself or the
accumulated context, answer directly.
Decide the next action. Choose one:
1) answer now
2) retrieve one passage from a tier (0/1/2)
Return JSON in one line with schema:
ACTION:ANSWER |RETRIEVE,TIER: 0 |1|2|null,
REASON: string
Rules:
- If action=answer, set tier=null.
- If action=retrieve, set tier to 0/1/2.
- If Remaining < cost(tier), you MUST answer.
- Do not answer just because Current context is
empty.
- If the information is insufficient, retrieve
before answering.
- Prefer the cheapest tier likely to be
sufficient, but use higher tiers when the
question likely needs official, professional,
or restricted-domain knowledge.
Table 4: Prompt Template for Per-query Budget Agent.Prompt
You are a budget allocation controller for a
retrieval-augmented QA system.
Before retrieval starts, allocate integer
retrieval budgets across all questions in the
batch.
Each tier of retrieval has a known cost, and
the downstream agent will spend the allocated
budget on retrieval to gather evidence for
answering.
Questions that likely need multi-hop, official,
professional, or domain-specific evidence
should receive more budget.
Questions that can likely be answered from
the prompt or cheap evidence can receive less
budget.
Return only the final tagged TSV allocation
block.
Batch size: <batch_size>
Total shared budget: <total_budget>
Per-question bounds: min=<min_budget>,
max=<max_text>
Retrieval tier options:
<tier_desc>
Questions: <Questions>
Allocate budgets with these strict rules:
- Include every id exactly once.
- Each budget must be a non-negative integer
within the per-question bounds.
- The sum of all budgets must equal the total
shared budget.
- Budget is spent only by retrieval; unused
budget may remain after the downstream agent
answers.
Return the final allocation as plain text, not
JSON. Use exactly this TSV block format:
BEGIN_BUDGET_ALLOCATION
<question id> <integer budget> <short reason>
...
END_BUDGET_ALLOCATION
Do not put Markdown tables, bullets, or JSON
inside the final block.
Table 5: Prompt Template for Budget Allocation in
Shared Budget setting.

Case:HotpotQA (ID: 5a70f39c5542994082a3e429).
Question:What is the name of the independent candidate in Maine’s 2010 gubernatorial race who finished ahead
of Libby Mitchell?
Ground-Truth:Eliot Cutler
Round 1:Okay, let’s tackle this question. The question is about a specific historical election, which likely
requires external knowledge. . . .The budget is 20, and I haven’t spent anything yet. . . .Considering the tiers,
tier 0 is free and includes sources like Wikipedia. That’s the cheapest option. I should start by retrieving
from tier 0. If the information isn’t there, I can move to higher tiers.. . .
Action:RETRIEVETIER: 0 COST: 0
Round 2:The question is about 2010, and the context here is 2018. So I need to find out who the independent
candidate was in 2010. . . .Since the question is about a specific historical event, tier 0 (Wikipedia) is the
best starting point. The budget is 20. . . .Therefore, the agent should retrieve another passage from tier 0 to
find the 2010 race details.. . .The budget allows for more retrievals.. . .
Action:RETRIEVETIER: 0 COST: 0
Round 3:The Passage 2 says: . . .Therefore, Cutler would have more votes than Mitchell. Thus, the answer is
Eliot Cutler.
Action:ANSWEREliot Cutler
TOTALCOST: 0 TOTALNUMBER OFPASSAGES: 2
Table 6: Case Study 1: the Agent Retrieves and Answers withTier Selection, andCost Consideration.

Case:NQ (ID: -458159261891926107).
Question:when was the last time mount ruapehu erupted?
Ground-Truth:25September2007
Round 1:The question requires specific geological data about the volcano’s eruptive history. I need to retrieve
data from an external source. . . .The budget is 20, and the cost for tier 0 is 0.Tier 0 includes community
resources like Wikipedia. That’s the cheapest option, so I should start there. If Wikipedia doesn’t have the
information, maybe I can move to a higher tier.. . .
Action:RETRIEVETIER: 0 COST: 0
Round 2:The current context mentions that Ruapehu . . .erupted between 15,000–2,000 years ago. But the
question is about the last eruption, . . .which might be within the last few hundred years or even more recently.
. . .The next step should be to retrieve another passage.Since the budget is 20 and the first retrieval was from
tier 0 (cost 0), there’s still a lot left.Tier 0 is the cheapest, so using tier 0 again makes sense.The reason is to
check community resources like Wikipedia for the most recent eruption data.If that doesn’t have it, maybe
move to tier 1 or 2, but tier 0 is the best first step.
Action:RETRIEVETIER: 0 COST: 0
Round 3: . . .The information in the passages is historical but doesn’t cover the most recent activity.Looking
at the tiers, Tier 0 is community resources like Wikipedia, which we’ve already used. Tier 1 is general
open-web content like news sites, which might have more recent information. Tier 2 is official sources,
which could be reliable but might be more expensive. The budget is 20, and the remaining is 20. Since
we’ve already retrieved two passages from Tier 0, maybe we should try Tier 1 next. Let me check the cost:
Tier 1 is 1 per passage. So retrieving from Tier 1 would cost 1, leaving 19.Tier 1 is the next best option.
Action:RETRIEVETIER: 1 COST: 1
Round 4: . . .Passage 3 is tier 1, but it’s a general geology article and doesn’t provide specific dates. . . .The user
needs external knowledge here.The budget is 20, spent 1, remaining 19.Since tier 0 was used already . . .tier
1 might have more recent info.If I choose tier 1 again, cost 1, which is possible.. . .
Action:RETRIEVETIER: 1 COST: 1
Round 5: . . .Passage 4 talks about . . .not Ruapehu. I need to retrieve more information. Therefore, the next
action should be to retrieve another passage, possibly from tier 1 or maybe tier 2. . . .But given the budget, if I
retrieve from tier 1 again,it’s 1 cost, which is affordable. . . .However, if the information isn’t found there, then
tier 2 might be needed. The next tier to consider is tier 1 again, as it’s the next most appropriate after tier 0.. . .
Action:RETRIEVETIER: 1 COST: 1
Round 6: . . .Passages 3, 4, and 5 are from tier 1, but they talk about other volcanoes . . .Since the budget
allows for more retrievals (remaining is 17), I need to decide whether to retrieve another passage. Tier 0 is
cheaper, but maybe tier 1 has more recent news articles. However, the previous retrievals from tier 1 didn’t
provide the answer.Maybe I should check a higher tier, like tier 2, but that’s more expensive. Wait, the
user’s remaining budget is 17, and tier 2 costs 4 per passage. If I retrieve from tier 2, I could spend 4,
leaving 13.. . .The user’s previous steps have already tried tier 0 and 1 without success.
Since the budget allows, perhaps it’s worth trying tier 2.
Action:RETRIEVETIER: 2 COST: 4
Round 7: . . .None of these passages provide the needed information.The user’s trajectory shows that they’ve
already spent 7 out of 20 budget points, with 13 remaining. They’ve tried tiers 0, 1, and 2.Since the
current context doesn’t have the required information, the next step is to retrieve another passage. . . .Wait, the
user has already used tier 2 once. But maybe they need another source from tier 0 or 1.The remaining budget is
13, so even if they retrieve another tier 1 (cost 1), they can do that. . . .Since the user has already used tier 0
and 1, maybe tier 2 again?But the cost is 4, and remaining budget is 13. If they retrieve tier 2 again, they
can spend 4, leaving 9. . . .Since tier 0 is the cheapest, perhaps retrieve another tier 0 passage.But the
user has already retrieved two tier 0 passages. However, the system allows retrieving from the same tier multiple
times. So the next action is to retrieve from tier 0 again, hoping to find the answer.
Action:RETRIEVETIER: 0 COST: 0
Round 8: . . .Wait, actually, in Passage 7 (tier 0), there’s a direct answer: it last erupted on the 25th of September
2007. That seems specific. . . .Yes, in the current context, Passage 7 is listed. So the answer is September 25,
2007. So the correct action is to answer now.
Action:ANSWER2007
TOTALCOST: 7 TOTALNUMBER OFPASSAGES: 7
Table 7: Case Study 2: the Agent Retrieves and Answers withTier Selection, andCost Consideration.

Case:MMLU (ID: 1022).
Question:A 1-year-old male is brought to the emergency department with blistered fresh burns over his hands
and feet in a stocking and glove distribution. His mother reports that he was accidentally burned when he turned
on the hot water in the bathtub while she was answering the telephone in the bedroom. In addition to debridement
of the burns, pain management, a thorough history and physical examination, and baseline laboratory studies, the
most appropriate next step is to Options: A. administer a 5% albumin infusion B. administer intravenous steroids
C. contact child protective services D. discharge the patient home with his parents
Ground-Truth:C
Round 1:Maybe I need to check if there’s any standard protocol here. . . .So, the agent should retrieve from
tier 2, which includes medical sources.So the next action is to retrieve from tier 2 to get the appropriate
guidelines.
Action:RETRIEVETIER: 2 COST: 4
Round 2:The options include contacting child protective services (option C), which might be relevant because
the burn pattern is suspicious for abuse . . .. Maybe the answer is C, but the current context doesn’t support that.
Therefore, I need to retrieve another passage.
But the remaining budget is 16. The next step would be to retrieve another passage from tier 2, which costs
4, leaving 12.
Action:RETRIEVETIER: 2 COST: 4
Round 3: . . .Passage 2 says that . . .The passages provided don’t mention CPS or child abuse protocols. But the
answer might be C. But without retrieval, can’t be sure.The budget allows for more retrievals (remaining is
12), I should check if Tier 2 has information on this. If I need to retrieve another passage from Tier 2, that
would cost 4, leaving 8.
Action:RETRIEVETIER: 2 COST: 4
Round 4:The options include contacting CPS, which would be relevant if there’s suspicion of abuse. Passage
2 says that burns . . .So maybe the answer requires external knowledge.The user has already spent 12, so
remaining is 8. Tier 2 costs 4 per passage. So if I retrieve another tier 2 passage, that would cost 4, leaving
4
Action:RETRIEVETIER: 2 COST: 4
Round 5:In young children, such burns can also be signs of neglect or abuse. Passage 1 and 2 talk about . . ..
Passage 3 and 4 mention . . .. However, none of the passages directly address child protective services (CPS) or
suspicion of abuse.But the remaining budget is 4, and tier 2 costs 4. If I retrieve another passage from tier
2, that would use up the remaining budget. So, even though the budget is low, maybe I should retrieve one
more passage from tier 2 to get that info.Then, after that, the agent can answer based on the new passage.
Action:RETRIEVETIER: 2 COST: 4
Round 6:However, none of the passages specifically mentions child abuse suspicion or CPS protocols.The
passages retrieved are from tier 2, which are medical sources.Therefore, the answer might require external
knowledge beyond the given context.But the budget is already exhausted (remaining 0), so I can’t retrieve
more.However, if the context doesn’t provide that, but the model is supposed to answer based on existing
knowledge, then answer C
Action:ANSWERC
TOTALCOST: 20 TOTALNUMBER OFPASSAGES: 5
Table 8: Case Study 3: the Agent Retrieves and Answers withTier Selection, andCost Consideration.

Case:HotpotQA.
Q1:Did Lewis Allen or Danny DeVito play Louie De Palma?
Q2:Who wrote Tom Vaughan’s popular 2008 film?
Q3:Are Ruggero Deodato from Italy, and Mexican Alejandro Springall, both film directors?
Q4:Which actor and film director is from German Jewish descent, Kurt Gerron or Dell Henderson?
Q5:Which genus has more species, Monstera or Cercis?
Responce:
Question 1: . . .the information is likely to be found in public knowledge or general open-web content (tier
1).So maybe tier 1, budget allocation of 1 or 2. perhaps tier 1 with budget 1.Question 2: . . .this information
might be found in general open-web sources (tier 1).So again, tier 1, budget 1 or 2. Maybe 2.Question 3:
. . .This is likely to be found in tier 1 (general open-web content) or tier 2 if it’s more specific.maybe tier 1.
Budget 1.Question 4: . . .This is a specific historical fact, which might require more detailed sources. If
the answer isn’t readily available in general sources, maybe tier 2. But if it’s public knowledge, tier 1.
Let’s say tier 2, budget 4. Question 5: . . .It might require more specific scientific sources, which could be tier
2. So tier 2, budget 4. Now, let’s try to assign budgets. Let’s assume that questions 1, 2, 3 are tier 1 (budget 1
each), question 4 tier 2 (budget 4), question 5 tier 2 (budget 4).Total would be 1+1+1+4+4=11, which is way
under 100. So clearly, that’s not sufficient.
So I need to assign each question a budget such that the sum is 100.
Let me reassess each question again with the idea of what’s needed:But that gives 1+2+1+4+4=12. Still way too
low. Let me think again about the questions: . . .4), question 5 tier 2 (budget 4).Total is 20+20+10+40+20=110.
Over the limit. Alternatively, adjust: . . .4), question 5 tier 2 (budget 4).Total is 15+15+10+30+20=90. Need
10 more. So distribute the remaining 10 across the questions. Maybe increase tier 2 budgets: Question 4: 40,
question 5: 20.Total 15+15+10+40+20=100. That works.
But I need to check if the reasoning for tiers is correct. . . .Alternatively,maybe question 5 is in tier1 because
it’s a general botanical fact.But if I change question 5 to tier 1, then the total becomes:Question1:15, q2:15,
q3:10, q4:40, q5:20 (total 100). But if question 5 is tier1, then budget 20 (assuming tier1). But I need to check if
the other questions need higher tiers. Alternatively, maybe question 4 is tier2, and question 5 is tier2. If question5
is tier2, then budget for q5 is 20 (which is 5 passages at 4 per passage).Then total would be:q1:15, q2:15,
q3:10, q4:40, q5:20.Total 100. That works. But maybe other allocations are possible. Let me think if any of the
questions require tier0.
Let’s assume that some questions can be answered with tier0 (cost 0). For example: Question 3: . . .This is
factual information. If the answer is in public knowledge (Wikipedia), then tier 0. So allocate 0. But that
would save budget. So perhaps: Question3: tier0, budget 0. So total would be 4), question 5 tier 2 (budget
4).15+15+0+40+20=90. Need 10 more.Let’s add 10 to question 1: 25, q2:15, q3:0, q4:40, q5:20. Total
25+15+0+40+20=100.
Final Allocation:Q1 : 25, Q2 : 15, Q3 : 0, Q4 : 40, Q5 : 20.
Table 9: Case Study 4: Flexible Budget Allocation in Shared Budget SettingTier Selection, andCost Considera-
tion.

CaseMedQA ( ID: 563)
Question:A 47-year-old woman seeks evaluation at your office because she has had postcoital vaginal bleeding
for the past 8 months with occasional intermenstrual watery, blood-tinged vaginal discharge. Her family history
is negative for malignancies and inherited disorders. She is the result of a pregnancy complicated by numerous
miscarriages in the 1960s, for which her mother received diethylstilbestrol. During a pelvic examination, you
notice a polypoid mass on the anterior wall of the vagina. The bimanual examination is negative for adnexal
masses. You suspect the presence of carcinoma and, therefore, send tissue samples to pathology, which confirmed
the presence of malignant cells. Which of the following is the most likely malignant tumor in this patient?
Options: "A": "Clear cell adenocarcinoma", "B": "Melanoma", "C": "Botryoid sarcoma", "D": "Verrucous
carcinoma"
Ground-Truth:A
Round 1:
Since DES is a known risk factor for clear cell adenocarcinoma . . .Since I’m not 100% certain without checking,
maybe I should retrieve. But the budget allows for it.Tier 2 is for medical sources, which would be the best for
this. So, I should retrieve from tier 2 to confirm the association between DES and clear cell adenocarcinoma.
But the remaining budget is 40, and tier 2 costs 4 per passage.
Action:RETRIEVETIER: 2 COST: 4
Round 2:
. . .Since the current context already provides the necessary information . . .there’s no need for further retrieval.
The answer can be confidently given as A. The budget remaining is 36, which is sufficient, but since the info is
already there, answering now is appropriate.
Action:ANSWERA
TOTALCOST: 4 TOTALNUMBER OFPASSAGES: 1
Table 10: Case Study 5: the Agent (Qwen3-8B, Budget=40) Retrieves and Answers withTier Selection, andCost
Consideration.

CaseMedQA ( ID: 563)
Question:A 47-year-old woman seeks evaluation at your office because she has had postcoital vaginal bleeding
for the past 8 months with occasional intermenstrual watery, blood-tinged vaginal discharge. Her family history
is negative for malignancies and inherited disorders. She is the result of a pregnancy complicated by numerous
miscarriages in the 1960s, for which her mother received diethylstilbestrol. During a pelvic examination, you
notice a polypoid mass on the anterior wall of the vagina. The bimanual examination is negative for adnexal
masses. You suspect the presence of carcinoma and, therefore, send tissue samples to pathology, which confirmed
the presence of malignant cells. Which of the following is the most likely malignant tumor in this patient?
Options: "A": "Clear cell adenocarcinoma", "B": "Melanoma", "C": "Botryoid sarcoma", "D": "Verrucous
carcinoma"
Ground-Truth:A
Round 1 – 2:
Question requires medical knowledge, so a higher tier is preferred for accuracy.
Action:RETRIEVETIER: 2 COST: 4
Round 3:
Since the current context is not sufficient to confidently answer the question, andthe remaining budget is more
than the cost of a passage from tier 2, the next action should be toretrieve a passage from tier 2 to gather
more specific information about the possible malignant tumors in the patient.
The question asks for the most likely malignant tumor in the patient, suggesting that more specific information is
needed, and a higher tier is preferred for accuracy.
Action:RETRIEVETIER: 2 COST: 4
Round 4 – 8:
The question asks for the most likely malignant tumor in the patient, suggesting that more specific information is
needed, and a higher tier is preferred for accuracy.
Action:RETRIEVETIER: 2 COST: 4
Round 9:
Based on the current context and the question asked, it is clear that the question requires outside knowledge.
Since the remaining budget is 8, and the cost of tier 2 is 4, we can afford to retrieve one more passage.
Therefore, the next action is to retrieve one passage from tier 2.
Action:RETRIEVETIER: 2 COST: 4
Round 10:The question asks . . .suggesting that more specific information is needed, and a higher tier is
preferred for accuracy.
Action:RETRIEVETIER: 2 COST: 4
Round 11: Action:ANSWERA
TOTALCOST: 40 TOTALNUMBER OFPASSAGES: 10
Table 11: Case Study 6: the Agent (Llama3.1-8B-Instruct, Budget=40) Retrieves and Answers withTier Selection,
andCost Consideration.

Method Bgt/kOpen-Domain Datasets Domain-Specific Datasets
HotpotQA NQ TriviaQA MedQA MMLU
C F1 (%) C F1 (%) C F1 (%) C F1 (%) C F1 (%)
Vanilla LLM - N/A37.80 ±8.66 N/A19.81 ±6.54 N/A43.57 ±8.75 N/A45.00 ±9.80 N/A75.00 ±8.53
Top-k5 N/A31.90 ±8.40 N/A37.14 ±7.75 N/A56.74 ±8.85 N/A66.00 ±9.33 N/A83.00 ±7.40
10 N/A37.19 ±8.61 N/A36.87 ±7.88 N/A54.16 ±8.67 N/A72.00 ±8.84 N/A84.00 ±7.22
15 N/A34.45 ±8.71 N/A40.10 ±8.28 N/A56.48 ±8.43 N/A68.00 ±9.19 N/A82.00 ±7.57
20 N/A33.02 ±8.56 N/A39.06 ±8.07 N/A59.41 ±8.45 N/A62.00 ±9.56 N/A83.00 ±7.36
25 N/A37.70 ±8.72 N/A39.60 ±8.21 N/A60.30 ±8.52 N/A61.00 ±9.61 N/A80.00 ±7.88
30 N/A35.43 ±8.80 N/A38.25 ±8.07 N/A63.11 ±8.37 N/A64.00 ±9.46 N/A78.00 ±8.16
Relevance0 0.0034.44 ±8.55 0.0037.49 ±7.91 0.0055.37 ±8.49 0.0067.00 ±9.26 0.0080.00 ±7.88
5 4.8033.73 ±8.47 5.0042.54 ±8.03 4.9059.39 ±8.58 5.0067.00 ±9.26 5.0084.00 ±7.22
10 9.2035.73 ±8.75 9.6042.95 ±8.44 9.1058.28 ±8.65 10.065.00 ±9.40 9.9080.00 ±7.88
15 12.935.35 ±8.75 13.642.41 ±8.22 12.659.30 ±8.60 14.966.00 ±9.33 14.782.00 ±7.57
20 15.536.44 ±8.67 16.840.42 ±8.15 15.657.48 ±8.59 19.968.00 ±9.19 19.681.00 ±7.73
25 17.238.70 ±8.79 19.142.51 ±8.05 18.062.79 ±8.58 25.067.00 ±9.26 24.180.00 ±7.88
30 18.138.11 ±8.80 20.936.50 ±7.78 19.759.99 ±8.41 29.864.00 ±9.46 28.179.00 ±8.02
∞19.235.49 ±8.70 23.439.50 ±8.09 22.364.43 ±8.35 61.767.00 ±9.26 48.280.00 ±7.88
Greedy0 0.0034.98 ±8.39 0.0035.76 ±7.79 0.0055.44 ±8.49 0.0061.00 ±9.61 0.0080.00 ±7.88
5 4.7033.40 ±8.31 4.9041.79 ±8.14 4.8059.86 ±8.55 4.8065.00 ±9.40 4.8082.00 ±7.57
10 8.9039.64 ±8.85 9.4041.13 ±8.29 8.8057.96 ±8.76 9.2066.00 ±9.33 9.1084.00 ±7.22
15 12.640.90 ±9.00 12.945.19 ±8.46 12.059.90 ±8.49 13.764.00 ±9.46 13.779.00 ±8.02
20 15.036.54 ±8.77 16.140.28 ±8.20 14.959.97 ±8.59 18.562.00 ±9.56 18.282.00 ±7.57
25 16.935.55 ±8.62 18.740.08 ±8.16 17.659.83 ±8.45 23.566.00 ±9.33 23.084.00 ±7.22
30 18.035.26 ±8.80 20.543.09 ±7.94 19.461.79 ±8.43 28.669.00 ±9.11 27.178.00 ±8.16
∞19.234.57 ±8.69 23.440.66 ±8.14 22.360.78 ±8.48 61.764.00 ±9.46 48.279.00 ±8.02
Knapsack0 0.0036.31 ±8.77 0.0037.11 ±7.91 0.0055.57 ±8.50 0.0062.00 ±9.56 0.0081.00 ±7.73
5 4.8033.70 ±8.49 5.0043.25 ±8.28 4.8058.94 ±8.49 4.9066.00 ±9.33 4.9083.00 ±7.40
10 9.2040.33 ±8.84 9.5041.64 ±8.02 8.9060.68 ±8.52 9.6065.00 ±9.40 9.6082.00 ±7.57
15 12.635.02 ±8.76 13.240.67 ±8.29 12.360.11 ±8.56 14.367.00 ±9.26 14.182.00 ±7.57
20 15.337.09 ±8.74 16.441.41 ±8.10 15.557.52 ±8.65 19.263.00 ±9.51 19.077.00 ±8.29
25 17.134.97 ±8.72 18.940.84 ±8.12 17.861.24 ±8.45 24.365.00 ±9.40 23.679.00 ±8.02
30 18.036.93 ±8.79 20.743.32 ±8.03 19.662.80 ±8.28 29.364.00 ±9.46 27.778.00 ±8.16
∞19.235.18 ±8.58 23.439.34 ±7.92 22.362.87 ±8.32 61.761.00 ±9.61 48.276.00 ±8.41
MMR0 0.0034.49 ±8.49 0.0038.92 ±8.04 0.0053.71 ±8.50 0.0064.00 ±9.46 0.0081.00 ±7.73
5 4.7034.63 ±8.56 4.9042.55 ±8.17 4.7060.38 ±8.56 4.7065.00 ±9.40 4.6083.00 ±7.40
10 8.0037.17 ±8.76 8.8044.11 ±8.18 8.0062.89 ±8.13 7.6066.00 ±9.33 7.9080.00 ±7.88
15 9.8037.92 ±8.90 11.141.80 ±8.24 9.9058.26 ±8.61 8.8063.00 ±9.51 9.1084.00 ±7.22
20 10.640.77 ±9.00 12.043.67 ±8.29 10.958.87 ±8.57 9.2065.00 ±9.40 9.5079.00 ±8.02
25 10.939.59 ±8.91 12.543.81 ±8.22 11.259.94 ±8.49 9.3067.00 ±9.26 9.6084.00 ±7.22
30 10.938.35 ±8.77 12.541.34 ±8.11 11.259.71 ±8.50 9.4065.00 ±9.40 9.7078.00 ±8.16
∞10.937.71 ±8.95 12.542.24 ±8.33 11.259.30 ±8.57 9.4070.00 ±9.03 9.7080.00 ±7.88
Redundancy0 0.0032.51 ±8.25 0.0036.51 ±7.95 0.0055.28 ±8.64 0.0063.00 ±9.51 0.0083.00 ±7.40
5 0.2032.61 ±8.26 0.3037.40 ±8.02 0.1057.47 ±8.50 1.2069.00 ±9.11 0.6080.00 ±7.88
10 0.2033.34 ±8.41 0.3039.90 ±8.25 0.1058.64 ±8.45 1.7067.00 ±9.26 0.9079.00 ±8.02
15 0.2033.78 ±8.48 0.3036.86 ±7.75 0.1055.05 ±8.57 1.8067.00 ±9.26 0.9080.00 ±7.88
20 0.2038.66 ±8.72 0.3038.61 ±5.85 0.1058.46 ±8.34 1.8070.00 ±9.03 0.9079.00 ±8.02
25 0.2030.47 ±8.12 0.3036.39 ±7.82 0.1057.19 ±8.49 1.8066.00 ±9.33 0.9080.00 ±7.84
30 0.2034.21 ±8.63 0.3039.34 ±8.00 0.1059.00 ±8.47 1.8071.00 ±8.94 0.9077.00 ±8.29
∞0.2034.19 ±8.61 0.3040.85 ±8.09 0.1056.86 ±8.46 1.8070.00 ±9.03 0.9083.00 ±7.40
Table 12: Overall Performance. We use Qwen3-8B as our backbone model.CandF1denote the average
optimization cost ( avg_cost ) and F1 score, respectively. ReportedF1scores incorporate 95% confidence intervals
(CIs) formatted as Value±Margin . For Top-k,Cis marked N/A because this baseline ignores access costs during
selection.

Method Bgt/kOpen-Domain Datasets Domain-Specific Datasets
HotpotQA NQ TriviaQA MedQA MMLU
C F1 (%) C F1 (%) C F1 (%) C F1 (%) C F1 (%)
Vanilla LLM - N/A14.61 ±4.63 N/A12.52 ±3.98 N/A24.28 ±5.82 N/A55.00 ±9.80 N/A67.00 ±9.26
Top-k5 N/A15.36 ±5.55 N/A27.55 ±6.90 N/A47.41 ±8.70 N/A56.00 ±9.78 N/A64.00 ±9.46
10 N/A26.53 ±7.65 N/A33.55 ±7.66 N/A54.38 ±8.62 N/A62.00 ±9.56 N/A63.00 ±9.51
15 N/A30.24 ±7.98 N/A36.57 ±7.92 N/A59.13 ±8.37 N/A56.00 ±9.78 N/A72.00 ±8.84
20 N/A33.99 ±8.30 N/A34.52 ±7.61 N/A63.23 ±8.31 N/A52.00 ±9.84 N/A68.00 ±9.19
25 N/A32.98 ±8.24 N/A39.60 ±8.21 N/A59.45 ±8.41 N/A54.00 ±9.82 N/A67.00 ±9.26
30 N/A32.69 ±8.18 N/A33.33 ±7.53 N/A60.40 ±8.47 N/A59.00 ±9.69 N/A71.00 ±8.94
Relevance0 0.0033.45 ±8.20 0.0039.48 ±8.10 0.0061.19 ±8.40 0.0056.00 ±9.78 0.0061.00 ±9.61
5 4.8031.23 ±8.01 5.0033.90 ±7.79 4.9060.87 ±8.28 5.0057.00 ±9.75 5.0062.00 ±9.56
10 9.2034.47 ±8.52 9.7037.12 ±7.97 9.1059.18 ±8.48 10.053.00 ±9.83 9.9066.00 ±9.33
15 12.936.99 ±8.56 13.642.44 ±8.25 12.660.93 ±8.35 14.956.00 ±9.78 14.771.00 ±8.94
20 15.541.02 ±8.69 16.841.36 ±8.00 15.660.29 ±8.40 19.947.00 ±9.83 19.662.00 ±9.56
25 17.233.55 ±8.27 19.231.93 ±7.76 18.064.47 ±8.08 25.052.00 ±9.84 24.170.00 ±9.11
30 18.137.73 ±8.55 20.939.51 ±7.97 19.760.62 ±8.34 29.853.00 ±9.83 28.165.00 ±9.40
∞19.237.14 ±8.46 23.534.71 ±7.86 22.361.81 ±8.20 61.755.00 ±9.80 48.266.00 ±9.33
Greedy0 0.0030.48 ±8.10 0.0035.42 ±8.18 0.0058.03 ±8.23 0.0053.00 ±9.83 0.0067.00 ±9.26
5 4.7431.35 ±7.96 4.8934.25 ±7.81 4.8256.84 ±8.34 4.7851.00 ±9.85 4.8258.00 ±9.72
10 8.9334.21 ±8.33 9.4034.43 ±7.99 8.7859.36 ±8.41 9.2248.00 ±9.84 9.1367.00 ±9.26
15 12.5636.26 ±8.60 12.9436.34 ±8.17 12.0361.44 ±8.36 13.7357.00 ±9.75 13.6867.00 ±9.26
20 15.0836.19 ±8.44 16.1036.87 ±7.96 14.9462.12 ±8.31 18.5454.00 ±9.82 18.1670.00 ±9.11
25 16.9835.17 ±8.47 18.7234.17 ±7.94 17.6061.18 ±8.27 23.5050.00 ±9.80 22.9976.00 ±8.41
30 18.0037.80 ±8.49 20.5436.66 ±8.15 19.3863.71 ±8.32 28.5755.00 ±9.80 27.0871.00 ±8.94
∞19.2435.74 ±8.46 23.4633.60 ±7.83 22.3060.34 ±8.33 61.7350.00 ±9.80 48.2475.00 ±9.26
Knapsack0 0.0033.91 ±8.15 0.0032.21 ±7.89 0.0056.13 ±8.16 0.0051.00 ±9.85 0.0060.00 ±9.60
5 4.7733.56 ±8.15 4.9735.91 ±7.91 4.8258.87 ±8.41 4.8754.00 ±9.75 4.8864.00 ±9.46
10 9.1733.05 ±8.20 9.4634.52 ±7.80 8.8758.50 ±8.46 9.5853.00 ±9.83 9.5557.00 ±9.75
15 12.6536.62 ±8.56 13.2135.70 ±7.83 12.3064.00 ±8.32 14.3058.00 ±9.72 14.1069.00 ±9.11
20 15.3833.59 ±8.40 16.4936.88 ±7.90 15.4560.05 ±8.39 19.2051.00 ±9.85 18.9772.00 ±8.84
25 17.1036.65 ±8.53 18.9433.35 ±7.73 17.8161.69 ±8.24 24.3157.00 ±9.75 23.5969.00 ±9.11
30 18.0634.43 ±8.49 20.7237.65 ±8.06 19.5962.29 ±8.27 29.2652.00 ±9.84 27.6879.00 ±8.02
∞19.2437.00 ±8.54 23.4635.31 ±7.91 22.3062.57 ±8.29 61.7354.00 ±9.82 48.2472.00 ±8.84
MMR0 0.0029.69 ±7.69 0.0033.52 ±8.08 0.0056.99 ±8.41 0.0047.00 ±9.83 0.0055.00 ±9.80
5 4.6728.78 ±7.89 4.8633.25 ±7.88 4.7360.37 ±8.39 4.6652.00 ±9.84 4.6270.00 ±9.11
10 8.0434.52 ±8.23 8.7834.21 ±7.99 8.0062.89 ±8.13 7.5856.00 ±9.75 7.8569.00 ±9.11
15 9.8235.19 ±8.23 11.1234.70 ±8.16 9.8763.78 ±8.38 8.8153.00 ±9.83 9.1266.00 ±9.33
20 10.6133.51 ±8.27 12.0436.45 ±8.11 10.8661.92 ±8.32 9.1853.00 ±9.83 9.5263.00 ±9.51
25 10.8838.97 ±8.58 12.4633.31 ±7.66 11.1661.07 ±8.29 9.3458.00 ±9.75 9.6366.00 ±9.33
30 10.8834.68 ±8.33 12.5636.55 ±8.26 11.1863.55 ±8.38 9.3753.00 ±9.83 9.6865.00 ±9.40
∞10.8835.69 ±8.43 12.5638.33 ±8.25 11.1861.83 ±8.32 9.3758.00 ±9.72 9.6866.00 ±9.33
Redundancy0 0.0030.62 ±8.01 0.0032.14 ±7.91 0.0058.66 ±8.27 0.0047.00 ±9.83 0.0068.00 ±9.26
5 0.1530.36 ±7.96 0.2634.12 ±7.86 0.0956.26 ±8.46 1.1852.00 ±9.84 0.6459.00 ±9.69
10 0.1930.46 ±8.00 0.3035.01 ±8.06 0.0957.41 ±8.47 1.6859.00 ±9.11 0.9164.00 ±9.40
15 0.1930.71 ±8.11 0.3034.70 ±8.11 0.0959.05 ±8.52 1.7657.00 ±9.75 0.9166.00 ±9.33
20 0.1932.29 ±8.06 0.3036.23 ±8.06 0.0957.42 ±8.46 1.7654.00 ±9.82 0.9164.00 ±9.40
25 0.1928.53 ±7.73 0.3034.87 ±8.06 0.0957.60 ±8.49 1.7659.00 ±9.11 0.9165.00 ±9.40
30 0.1928.74 ±7.86 0.3036.23 ±8.15 0.0958.24 ±8.49 1.7651.00 ±9.84 0.9163.00 ±9.51
∞0.1933.32 ±8.43 0.3035.73 ±7.91 0.0957.66 ±8.41 1.7651.00 ±9.84 1.7664.00 ±9.40
Table 13: Overall Performance. We use Llama3.1-8B-instruct as our backbone model.CandF1denote the average
optimization cost ( avg_cost ) and F1 score, respectively. ReportedF1scores incorporate 95% confidence intervals
(CIs) formatted as Value±Margin . For Top-k,Cis marked N/A because this baseline ignores access costs during
selection.

Method Bgt/kOpen-Domain Datasets Domain-Specific Datasets
HotpotQA NQ TriviaQA MedQA MMLU
C F1 (%) C F1 (%) C F1 (%) C F1 (%) C F1 (%)
Vanilla LLM - N/A39.10 ±8.71 N/A21.89 ±6.74 N/A41.49 ±8.67 N/A44.00 ±9.78 N/A74.00 ±8.64
Top-k5 N/A34.14 ±8.59 N/A35.61 ±7.50 N/A56.85 ±8.75 N/A71.00 ±8.94 N/A81.00 ±7.69
10 N/A32.75 ±8.48 N/A37.91 ±7.72 N/A54.74 ±8.73 N/A71.00 ±8.94 N/A80.00 ±7.84
15 N/A36.07 ±8.70 N/A39.97 ±8.26 N/A58.97 ±8.49 N/A65.00 ±9.40 N/A79.00 ±7.98
20 N/A34.53 ±8.63 N/A38.55 ±8.23 N/A59.92 ±8.42 N/A61.00 ±9.56 N/A82.00 ±7.47
25 N/A32.98 ±8.24 N/A40.37 ±8.02 N/A58.88 ±8.61 N/A62.00 ±9.56 N/A82.00 ±7.47
30 N/A37.45 ±8.90 N/A35.12 ±7.64 N/A63.49 ±8.31 N/A61.00 ±9.56 N/A79.00 ±7.98
Relevance0 0.0036.91 ±8.73 0.0039.48 ±8.34 0.0055.38 ±8.52 0.0062.00 ±9.56 0.0077.00 ±8.25
5 4.8434.71 ±8.63 4.9844.19 ±8.36 4.8756.02 ±8.66 5.0066.00 ±9.33 4.9884.00 ±7.22
10 9.1037.89 ±8.91 9.5141.44 ±8.08 8.9757.51 ±8.78 9.9963.00 ±9.51 9.9580.00 ±8.00
15 12.3536.15 ±8.85 13.2942.44 ±8.34 12.2660.77 ±8.63 15.0065.00 ±9.40 14.6180.00 ±8.00
20 14.2935.79 ±8.53 15.8541.36 ±8.21 14.8560.14 ±8.55 19.8862.00 ±9.56 18.9680.00 ±8.00
25 15.0135.28 ±8.69 17.2440.21 ±8.13 16.2459.96 ±8.46 24.5763.00 ±9.51 22.6382.00 ±7.47
30 15.2637.79 ±8.97 17.8339.51 ±7.97 16.6260.62 ±8.34 28.7962.00 ±9.56 25.5080.00 ±7.84
∞15.4036.68 ±8.74 17.9838.63 ±8.13 16.8062.92 ±8.36 35.5568.00 ±9.19 28.9677.00 ±8.29
Greedy0 0.0031.09 ±8.16 0.0037.12 ±7.95 0.0055.64 ±8.48 0.0059.00 ±8.87 0.0071.00 ±8.94
5 4.8432.90 ±8.37 4.9540.77 ±8.31 4.8258.01 ±8.63 28.4955.00 ±9.80 25.2471.00 ±8.94
10 8.9940.66 ±8.86 9.4139.89 ±8.09 8.8458.93 ±8.71 9.3764.00 ±9.46 9.6880.00 ±7.84
15 12.2736.65 ±8.82 13.0638.46 ±8.24 12.1561.30 ±8.54 1.4966.00 ±9.33 0.7078.00 ±8.16
20 14.2235.91 ±8.69 15.6640.71 ±8.02 14.6860.32 ±8.48 28.7962.00 ±9.56 25.5080.00 ±7.84
25 15.0037.77 ±8.88 17.1840.43 ±8.11 16.2060.58 ±8.41 1.4968.00 ±9.11 0.7079.00 ±8.02
30 15.2437.80 ±8.49 17.8039.83 ±8.29 16.6063.71 ±8.41 28.7953.00 ±9.83 25.5080.00 ±7.84
∞15.4037.68 ±8.73 17.9838.36 ±7.83 16.8065.70 ±8.19 35.5565.00 ±9.40 28.9675.00 ±8.53
Knapsack0 0.0037.37 ±8.79 0.0038.78 ±8.10 0.0059.37 ±8.47 0.0065.00 ±9.40 0.0079.00 ±8.02
5 4.8436.85 ±8.67 4.9841.53 ±8.23 4.8759.41 ±8.56 29.2664.00 ±9.46 27.6879.00 ±7.98
10 9.1038.67 ±8.75 9.5045.44 ±8.34 8.9759.66 ±8.64 1.4966.00 ±9.33 0.7078.00 ±8.16
15 12.3437.74 ±8.78 13.2141.46 ±8.33 12.2662.29 ±8.33 35.5565.00 ±9.40 28.9677.00 ±8.20
20 14.2740.00 ±8.98 15.8038.80 ±7.89 14.8562.61 ±8.46 1.4964.00 ±9.46 0.7079.00 ±8.02
25 15.0136.94 ±8.78 17.2441.94 ±7.96 16.2461.67 ±8.34 35.5564.00 ±9.46 28.9672.00 ±8.84
30 15.2637.98 ±8.84 17.8337.86 ±7.86 16.6262.73 ±8.33 1.4965.00 ±9.40 0.7079.00 ±7.98
∞15.4037.66 ±8.91 23.4638.95 ±8.00 16.8062.57 ±8.29 35.5561.00 ±9.61 28.9679.00 ±7.98
MMR0 0.0035.10 ±8.40 0.0038.46 ±8.08 0.0057.09 ±8.51 0.0063.00 ±9.51 0.0082.00 ±7.47
5 4.6734.85 ±8.47 4.9243.58 ±8.10 4.8059.54 ±8.49 4.9364.00 ±9.46 4.9283.00 ±7.40
10 8.0437.83 ±8.77 9.3943.40 ±8.26 8.6062.89 ±8.33 9.7565.00 ±9.40 9.6582.00 ±7.47
15 9.8239.54 ±8.87 12.5642.35 ±8.16 11.1862.18 ±8.32 27.5964.00 ±9.46 23.0080.00 ±7.84
20 11.3838.49 ±8.55 12.8142.35 ±8.00 11.2862.18 ±8.30 28.7965.00 ±9.40 25.5080.00 ±7.84
25 11.7736.06 ±8.71 15.9740.64 ±7.98 13.7461.07 ±8.29 28.7967.00 ±9.26 25.5080.00 ±7.84
30 11.7737.68 ±8.84 16.3442.35 ±8.33 13.8663.55 ±8.38 28.7965.00 ±9.40 25.5080.00 ±7.84
∞11.7736.30 ±8.84 16.4243.86 ±8.25 13.9061.83 ±8.32 35.5562.00 ±9.56 28.9681.00 ±7.47
Redundancy0 0.0036.61 ±8.62 0.0036.97 ±7.91 0.0055.60 ±8.53 0.0062.00 ±9.56 0.0082.00 ±7.57
5 0.1534.16 ±8.66 0.2638.93 ±8.11 0.0957.39 ±8.47 1.4967.00 ±9.26 0.7080.00 ±7.88
10 0.1937.80 ±8.84 0.3037.86 ±7.86 0.0959.37 ±8.47 1.4965.00 ±9.40 0.7079.00 ±7.98
15 0.1937.98 ±8.84 0.3037.65 ±8.06 0.0962.29 ±8.27 1.4965.00 ±9.40 0.7079.00 ±7.98
20 0.1937.68 ±8.84 0.3039.17 ±8.06 0.0958.24 ±8.47 1.4966.00 ±9.33 0.7078.00 ±8.16
25 0.1934.16 ±8.66 0.3041.96 ±8.06 0.0959.96 ±8.46 1.4968.00 ±9.19 0.7080.00 ±7.84
30 0.1937.79 ±8.97 0.3039.51 ±7.97 0.0956.27 ±8.49 1.4966.00 ±9.33 0.7078.00 ±8.16
∞0.1933.32 ±8.49 0.3036.23 ±8.06 0.0957.66 ±8.41 1.4966.00 ±9.33 1.4979.00 ±8.02
Table 14: Overall Performance when Pricing Setting is 0, 1, and 2 for Tier 0, 1, and 2, respectively. We use Qwen3-
8B as our backbone model.CandF1denote average optimization cost ( avg_cost ) and F1 score, respectively.
ReportedF1scores incorporate 95% confidence intervals (CIs) formatted as Value±Margin . For Top-k,Cis marked
N/A because this baseline ignores access costs during selection.
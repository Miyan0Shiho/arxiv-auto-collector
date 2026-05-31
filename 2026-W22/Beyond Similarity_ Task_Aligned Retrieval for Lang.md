# Beyond Similarity: Task-Aligned Retrieval for Language Models

**Authors**: Zhixing Sun, Shenghe Xu, Tao Li

**Published**: 2026-05-27 04:45:57

**PDF URL**: [https://arxiv.org/pdf/2605.27951v1](https://arxiv.org/pdf/2605.27951v1)

## Abstract
Retrieval-augmented generation (RAG) ranks passages by semantic similarity to the input, implicitly assuming that semantic similarity is a reliable indication of applicability in downstream tasks. This assumption breaks down when task success depends not on topical relevance but on applying the correct rules, constraints, or procedural guidance. In such settings, the most useful context may be the rule triggered by the input rather than the most semantically similar passage. We propose Task-Aligned Retrieval (TAG), a retrieval framework that replaces similarity-based retrieval with applicability-based rule selection. TAG transforms source documents into traceable condition-action rules, identifies which rules apply to a given input through pairwise LLM judgments, and generates the output conditioned only on the selected actions. We empirically observe that across Wikipedia NPOV rewriting, HumanEval with PEP~8 compliance, and NBA transaction reasoning on RuleArena, TAG consistently outperforms standard RAG, with the largest gains in high-mismatch settings (up to 12.2\%) while reducing retrieved context by up to 93\%. These results suggest that, in rule- and instruction-governed tasks, retrieval should optimize for applicability rather than for semantic similarity alone.

## Full Text


<!-- PDF content starts -->

Beyond Similarity: Task-Aligned Retrieval for Language Models
Zhixing Sun
Beijing University of Posts
and Telecommunications
Beijing, China
3342349163@bupt.edu.cnShenghe Xu
City University of Hong Kong
Hong Kong SAR, China
shenghexu@gmail.comTao Li
City University of Hong Kong
Hong Kong SAR, China
li.tao@cityu.edu.hk
Abstract
Retrieval-augmented generation (RAG) ranks
passages by semantic similarity to the input,
implicitly assuming that semantic similarity is
a reliable indication of applicability in down-
stream tasks. This assumption breaks down
when task success depends not on topical rel-
evance but on applying the correct rules, con-
straints, or procedural guidance. In such set-
tings, the most useful context may be the rule
triggered by the input rather than the most se-
mantically similar passage. We propose Task-
Aligned Retrieval (TAG), a retrieval frame-
work that replaces similarity-based retrieval
with applicability-based rule selection. TAG
transforms source documents into traceable
condition-action rules, identifies which rules
apply to a given input through pairwise LLM
judgments, and generates the output condi-
tioned only on the selected actions. We em-
pirically observe that across Wikipedia NPOV
rewriting, HumanEval with PEP 8 compliance,
and NBA transaction reasoning on RuleArena,
TAG consistently outperforms standard RAG,
with the largest gains in high-mismatch settings
(up to 12.2%) while reducing retrieved context
by up to 93%. These results suggest that, in
rule- and instruction-governed tasks, retrieval
should optimize for applicability rather than for
semantic similarity alone.
1 Introduction
Retrieval-augmented generation (RAG) improves
language models by conditioning generation on re-
trieved external context rather than relying solely
on parametric memory (Lewis et al., 2020). In
standard RAG pipelines, retrieval is driven by se-
mantic similarity between the input and indexed
passages, resting on an implicit assumption that
the context most semantically similar to the input
is also the most useful for the downstream task.
This is a premise that works well in fact-seeking
settings, where topical relevance often correlates
with evidential usefulness.However, this assumption fails in tasks where
task completion pertains not to retrieving semanti-
cally related evidence, but to applying the correct
rules, constraints, or procedural guidance. In such
settings, the retrieval target is not the passage with
the greatest topical overlap, but the rule applica-
ble to the input. Revising a Wikipedia sentence,
checking code against a style guide, or evaluating
the legality of a transaction all require identifying
the governing rule rather than the most topically
similar text.
We formalize this failure mode asretrieval objec-
tive mismatch: a divergence between the criterion
for retrieving context and the criterion that deter-
mines the utility of retrieved context. In settings
governed by instructional documents, applicability
may depend on abstract conditions, decision bound-
aries, or dispersed procedural logic that semantic
similarity fails to capture. As a result, retrieving
the most semantically similar chunk can systemat-
ically miss the context that should actually drive
generation.
Figure 1 illustrates this mismatch, where the
LLM is tasked with revising a Wikipedia sentence
to comply with the Neutral Point of View (NPOV)
policy. Given an input sentence describing Hamas
as a “militant organization”, a similarity-based re-
triever retrieves politically related chunks due to
topical overlap. However, the rule in the policy that
actually applies to this sentence concerns attribu-
tion of disputed characterizations rather than how
political entities should be described in general. Be-
cause this applicable rule shares little lexical over-
lap with the input, it is ranked low by similarity-
based retrieval even though it is the context that
should determine the correct revision.
Our framework.To address the retrieval objec-
tive mismatch, we propose TAG, a task-aligned
retrieval framework that alters both the workflow
and the objective of retrieval. Instead of retriev-arXiv:2605.27951v1  [cs.IR]  27 May 2026

Figure 1: The retrieval objective mismatch. The top chunk retrieved by similarity discusses how to describe political
movements in general, whereas the applicable rule concerns the assertion of disputed characterisations in the
authoritative voice of Wikipedia. The applicable rule has low semantic similarity with the input.
ing free-form text chunks by semantic similarity,
TAG first transforms the source document into
atomic condition-action rules through an LLM-
driven extraction pipeline, with each rule linked to
a traceable source context. A pairwise applicability
matcher then determines, for each rule indepen-
dently, whether the task input satisfies its condition.
Finally, only the actions of the matched rules are
passed to a domain-neutral executor for genera-
tion. This decomposition makes retrieval decisions
explicit, executable, and traceable to the matched
rules that drive generation. This shift from similar-
ity to applicability is most helpful when the applica-
ble rule shares little semantic similarity (measured
by cosine similarity) with the input, as is often
the case in regulatory compliance, and semantic
similarity poorly predicts which rule applies (§E).
Contributions.Our contributions are threefold.
First, we identifyretrieval objective mismatch, a
structural failure mode of similarity-based RAG
in which semantic similarity is misaligned with
downstream utility. Second, we introduce TAG,
a task-aligned retrieval framework that retrieves
applicability-matched atomic rules rather than se-
mantically similar text chunks. Third, across
Wikipedia NPOV rewriting, PEP 8-constrained
code generation, and NBA transaction reasoning,
we characterize when task-aligned retrieval helps:
gains are largest under substantial retrieval mis-
match, smaller when similarity and applicability
already align, and limited when downstream rea-soning requires complex multi-step rule interac-
tion.
2 Related Work
Similarity-based and reranked RAG.Standard
RAG retrieves semantically similar passages from
the chunked source documents using dense re-
trieval or reranking (Lewis et al., 2020; Karpukhin
et al., 2020; Nogueira and Cho, 2019; Glass et al.,
2022). Subsequent work improves this pipeline
through query rewriting, adaptive retrieval, itera-
tive reasoning, and graph-based retrieval (Ma et al.,
2023; Jeong et al., 2024; Trivedi et al., 2023; Edge
et al., 2024; Jiang et al., 2023). These methods im-
prove retrieval quality while preserving semantic
similarity as the governing retrieval relation.
Structured and rule-guided RAG.Several
methods modify the retrieval unit rather than the
ranking relation. GraphRAG (Edge et al., 2024)
retrieves graph neighborhoods rather than flat pas-
sages, and other variants enrich chunks with higher-
level structure. RuleRAG uses externally provided
rules as in-context guidance (Chen et al., 2024),
while RuAG mines rules from training data (Zhang
et al., 2025). TAG differs by automatically ex-
tracting executable rules from the source document
itself, and retrieving them based on applicability
rather than similarity.
LLM-based reranking and alignment.TAG is
mechanically related to LLM-based rerankers and

relevance judges (Nogueira and Cho, 2019; Glass
et al., 2022; Wang et al., 2024), but differs in the
retrieval predicate: instead of asking whether a pas-
sage is broadly relevant, TAG evaluates whether
the input satisfies a candidate rule’s condition. Re-
lated systems such as Self-RAG (Asai et al., 2024),
AlignRAG (Wei et al., 2025), chain-of-retrieval
(Wang et al., 2025), and DynamicRAG (Sun et al.,
2025) refine or adapt retrieval during or after gen-
eration. TAG instead changes the retrieval relation
itself.
3 Methodology
3.1 Problem Formulation
LetDdenote a source document and Ta down-
stream task whose correct outputs are determined
by the rules and constraints in D. Standard RAG
(Lewis et al., 2020) partitions DintoNpassages
{c1, . . . , c N}and retrieves the top- kpassages most
similar to inputx:
CRAG(x) = Top-k ci∈Dsim 
emb(x),emb(c i)
(1)
where emb(·) is a dense encoder (Xiao et al., 2023)
andsim(·,·)is cosine similarity.
The retrieval objective mismatch.Eq. 1 opti-
mizes semantic similarity, whereas in tasks where
success depends on applying the correct rules or
constraints, the relevant retrieval relation is appli-
cability. Similarity measures representational prox-
imity, whereas applicability asks whether a rule’s
condition is satisfied by the task input. Formally,
a conditional rule rapplies to input xifcond(r)
holds for x, where cond(r) denotes the condition
ofr. The idealized retrieval target is therefore
R⋆(x;D) ={r∈ R(D)|cond(r)holds forx}.
(2)
Since exact applicability is not directly observable
at inference time, we approximate R⋆with a pair-
wise LLM matcherR matched (x)(Eq. 4).
3.2 Framework Overview
TAG follows three design principles, implemented
through a three-stage pipeline (Fig. 2): (P1) re-
trieval operates over atomic rules; (P2) selection
is based on applicability; and (P3) matcher and ex-
ecutor roles are separated, so the matcher selects
rules while the executor follows only their actions.
Stage 1 (offline) convertsDinto a rule setri= 
name i,condition i,action i,source i,tagsi
,
(3)
where condition ispecifies a judgeable condition,
action ia concrete operation, source ia verbatim
source span for traceability, and tagsidenotes
lightweight metadata for deduplication.
Stage 2 (online) evaluates each rule indepen-
dently through pairwise applicability judgment:
match(x, r i) =f judge(x, r i),(4)
where fjudge returns YES or NO. The matched set
Rmatched (x) ={r i: match(x, r i) = YES} re-
places similarity-based top-kretrieval.
Stage 3 prompts the executor with input xand
only the actions of matched rules.
3.3 Stage 1: Five-Phase Rule Extraction
Stage 1 converts DintoRthrough five LLM-
driven phases (Fig. 3), each targeting a distinct
extraction failure mode.
Phase 1: Source text span detection.The LLM
extracts instructional spans from Dwhile preserv-
ing source boundaries for traceability, yielding
S={s 1, . . . , s L}(prompt: App. F.1.1).
Phase 2: Atomic decomposition.Each sj
is decomposed into atomic units containing a
single condition–action pair, yielding A=
{a1, . . . , a M}.(App. F.1.2)
Phase 3: Operationalization.Each atomic unit
akis converted into the structured rule tuple in
Eq. 3. The prompt enforces judgeable conditions,
executable actions, traceable source spans, and
lightweight category tags used for downstream
comparison and deduplication.(App. F.1.3)
Phase 4: Deduplication and conflict resolution.
To avoid O(M2)comparison, we compare only
rules sharing at least one tag. An LLM classifies
each pair as duplicate (merged), overlap (linked
but retained), conflict (flagged), or independent,
yieldingM′≤Mrules.(App. F.1.4)
Phase 5: Verification.We apply three sanity
checks:Faithfulness(source spans remain trace-
able to D),Coverage(each source span in Sis
represented by at least one rule), andIndependence
(a duplicate-rate proxy based on rule-name unique-
ness). The full verification procedure is provided
in App. F.1.5.

Figure 2: Overview of TAG. Document D is processed offline into atomic rules (§3.3); at inference, each rule is
judged for applicability by a pairwise LLM matcher (§3.4), and matched actions condition execution (§3.5).
3.4 Stage 2: Pairwise Applicability Matching
Applicability prompt.Unlike standard LLM
rerankers, which judge whether a passage is
broadly relevant to the input, our matcher evaluates
a narrower question: whether the input satisfies a
candidate rule’s condition. Concretely, instead of
asking “Is this rule relevant to the input?”, we ask
“Does the input satisfy the condition of this rule?”
This reduces retrieval to a binary entailment-style
judgment over a structured condition.
Pairwise rather than batched.We evaluate each
pair(x, r i)in isolation rather than batching all
Mrules into a single prompt, as batched judg-
ments suffer from attention dilution (Liu et al.,
2024) and inter-rule interference. For each pair,
the judge produces a single YES/NO decision at
temperature 0, and rules judged YES are added to
Rmatched (x).(App. F.2)
3.5 Stage 3: Rule-Guided Execution
Stage 3 prompts the executor with input xand
the actions of matched rules. Following P3, only
action iis exposed to the executor; conditions,
source spans, and tags are withheld to preserve
the matcher–executor separation and prevent the
executor from implicitly re-ranking rules.
4 Experiments
4.1 Research Questions and Experiment
Design
Our experiments are organized around four re-
search questions. RQ1 asks whether retrieval objec-tive mismatch exists in practice: do applicability-
based methods yield larger gains in domains where
semantic similarity is poorly aligned with rule ap-
plicability? RQ2 asks whether replacing top- ksimi-
larity retrieval with pairwise applicability matching
improves downstream generation across diverse
tasks. RQ3 asks which design component drives
the gain: rule structuring (P1), applicability-based
selection (P2), or action-only execution (P3). This
question is addressed by the ablations in §4.7 and
§4.8. RQ4 asks where TAG fails, i.e., which task
characteristics limit its effectiveness.
4.2 Setup
Domains.We evaluate on three domains chosen
to vary in the degree of mismatch between semantic
similarity and rule applicability.Wikipedia NPOV
(text rewriting, 107cases from the NPOV Notice-
board 2020–2024) is a high-gap setting: given
a sentence flagged for a neutrality violation, the
model must rewrite it to remove the violation while
preserving factual content, and violations often
share little lexical overlap with the governing rules.
HumanEval ×PEP 8(Chen et al., 2021) is a code-
generation and style-compliance task built from
164HumanEval programming problems. Given a
function signature and docstring, the model must
generate a function body that both passes the offi-
cial unit tests and conforms to PEP 8.RuleArena-
NBA(Zhou et al., 2025) ( 156problems stratified
into L0/L1/L2 by transaction count, see Appendix
A for detail) is a high-conditionality multi-step set-
ting: each problem asks whether one or more pro-

posed transactions are permitted under the NBA
Collective Bargaining Agreement, and asks the
model to identify the offending transaction and re-
sponsible team. Dataset details are in Appendix A.
Metrics.For NPOV we report a binary Violation
Fix Rate (VFR) plus four 1–5 auxiliary scores (Re-
mediation, Preservation, Tone, Fluency). For code
we report Pass@1 and pylint (headline style metric).
For RuleArena-NBA we follow the benchmark’s
strict accuracy, which requires the True/False an-
swer, the offending transaction, and the responsible
team to all be correct. Scoring details and evaluator
prompts are in Appendix B.
Models and baselines.DeepSeek-V4 serves as
both the executor and the pairwise matcher; we
also report results with Qwen3-30B-A3B-Instruct-
2507 as executor. We compare againstM0(no re-
trieval),M1(all extracted rules in the prompt),M2
(Standard RAG over 500-character chunks with
100-character overlap, indexed with BGE (Xiao
et al., 2023), with top- kfork∈ {5,10,15,20} ).
M3 (TAG, Ours)performs single-shot pairwise
applicability judgment at temperature 0 for each
(input, rule) pair and conditions the executor on the
actions of matched rules.
4.3 Experiment 1: Wikipedia NPOV
Setup.We collected 107NPOV violation cases
from the Noticeboard, where editors discuss article-
level disputes about whether content satisfies the
Neutral Point of View policy. Each case pairs a
flagged sentence with an editor-provided descrip-
tion of the neutrality issue. We process the NPOV
policy ( ∼15K words) into M=119 atomic rules;
faithfulness, coverage, and independence all ex-
ceed 0.94. Full curation criteria, extraction statis-
tics, and the trivial-rewrite filter are in Appendix A.
Results and analysis.Table 1 reports the main
results. TAG achieves the highest Violation Fix
Rate (VFR) on both executors, reaching 68.2%
on DeepSeek-V4 and 64.5% on Qwen3-30B-A3B-
Instruct-2507, outperforming the strongest Stan-
dard RAG baseline by 7.4and5.6absolute points
respectively. Standard RAG shows no consistent
improvement as retrieval depth increases for both
LLMs, with the largest candidate pool yielding the
worst result for Qwen. This suggests that simply
enlarging a similarity-driven retrieval pool does not
reliably find the governing policy rules. The All-
Rules baseline isolates the effect of rule structuringwithout selection. Although it is the strongest non-
TAG baseline on both LLMs ( 63.5% and62.6% ),
TAG still improves upon it while reducing aver-
age context size from 131.0 rules to 21.0 and32.6,
corresponding to 6.2× and4.0× reductions. This
shows that the improvement cannot be explained by
rule structuring alone: selecting the applicable sub-
set is more effective than conditioning the executor
on the full rule set. On the auxiliary metrics, TAG
achieves the highest Remediation score in both set-
tings ( 3.55 and3.45), with modest decreases in
Preservation and Fluency, suggesting that TAG pri-
oritizes remediation when neutrality preservation
trades off against fluency.
We validate the LLM judge with human an-
notation on n= 40 cases by two independent
raters; inter-annotator agreement is substantial
(κ∈[0.68,0.75] ) and the ranking M3 >M1>M2
is preserved across all metrics. See Appendix G.
4.4 Experiment 2: HumanEval with PEP 8
Compliance
Setup.HumanEval (Chen et al., 2021) provides
164 function-completion tasks. PEP 8 ( ∼8K
words) is processed into 127atomic rules. Pass@1
is computed following the official HumanEval pro-
tocol; pylint configurations are detailed in Ap-
pendix B..
Results and analysis.Table 2 reports the results.
TAG achieves the highest Pylint score on both
LLMs, reaching 9.36 on DeepSeek-V4 and 9.27
on Qwen3-30B-A3B-Instruct-2507. Compared
to Wikipedia NPOV , the gains over the strongest
Standard RAG baseline are smaller ( +0.15 on
DeepSeek-V4 and +0.42 on Qwen3), indicating
less room for applicability-based retrieval to im-
prove over similarity-based retrieval in this set-
ting. Pass@1 provides a complementary robustness
check. On DeepSeek-V4, functional correctness re-
mains essentially unchanged across methods ( 97.6–
98.8% ), indicating that stronger style compliance
does not come at a meaningful cost to task comple-
tion. On Qwen3, No Retrieval achieves the highest
Pass@1 ( 93.3% ), while retrieval-augmented vari-
ants cluster near 90%, suggesting a modest func-
tionality trade-off in this setting. The All-Rules
baseline is also highly competitive, reaching Pylint
scores of 9.32 and9.23, close to TAG. This in-
dicates that structured rule conditioning already
captures much of the available gain in this domain.

Executor Method #Rules VFR (%) Rem Pres Tone Flu Avg
DeepSeek-V4No Retrieval (M0) 0.0 32.7 2.334.703.444.583.76
All Rules (M1) 131.0 63.5 3.51 4.243.964.21 3.98
Std RAG, top-5 5.0 55.1 3.34 4.36 3.78 4.25 3.93
Std RAG, top-10 10.0 59.8 3.34 4.38 3.76 4.08 3.89
Std RAG, top-15 15.0 59.8 3.39 4.40 3.79 4.30 3.97
Std RAG, top-20 20.0 60.8 3.48 4.44 3.84 4.294.01
TAG (Ours) 21.068.2 3.554.13 3.94 4.21 3.96
Qwen3-30B-A3B
-Instruct-2507No Retrieval (M0) 0 6.5 1.254.943.135.003.58
All Rules (M1) 131 62.6 3.33 3.99 3.92 4.333.89
Std RAG, top-5 5 57.0 3.28 3.763.974.35 3.84
Std RAG, top-10 10 55.1 3.21 3.79 3.92 4.30 3.81
Std RAG, top-15 15 58.9 3.31 3.79 3.96 4.35 3.85
Std RAG, top-20 20 53.3 3.13 3.80 3.83 4.27 3.76
TAG (Ours) 32.664.5 3.453.70 3.90 3.79 3.71
Table 1: Results on Wikipedia NPOV rewriting ( n= 107 ). VFR is the binary Violation Fix Rate; Rem/Pres/Tone/Flu
are 1–5 auxiliary scores; Avg is their mean. #Rules reports the average number of rules (or chunks, for Standard
RAG) provided to the executor.
Executor Method #Units Pass@1 (%) Pylint
DeepSeek-V4No Retrieval (M0) 098.89.12
All Rules (M1) 12798.89.32
Std RAG, top-5 5 97.6 9.13
Std RAG, top-10 10 98.4 9.21
Std RAG, top-15 15 98.2 9.20
Std RAG, top-20 2098.89.11
TAG (Ours) 22.0 98.19.36
Qwen3-30B-A3B
-Instruct-2507No Retrieval (M0) 093.38.26
All Rules (M1) 127 90.2 9.23
Std RAG, top-5 5 87.8 8.63
Std RAG, top-10 10 88.4 8.80
Std RAG, top-15 15 88.4 8.70
Std RAG, top-20 20 89.0 8.85
TAG (Ours) 36.1 90.29.27
Table 2: Results on HumanEval with PEP 8 compliance ( n= 164 ). Pylint serves as the headline style metric;
Radon and Bandit are reported as secondary indicators.
4.5 Experiment 3: RuleArena-NBA (High
Conditionality, Multi-Step)
Setup.We use the NBA subset of RuleArena
(Zhou et al., 2025) ( 156 problems across
L0/L1/L2). The NBA CBA ( ∼60K words) is pro-
cessed into 195atomic rules. See Appendix A for
the per-level breakdown and rule coverage.
Results and analysis.Table 3 shows that
RuleArena-NBA remains challenging under the
strict metric, with TAG achieving the highest
overall strict accuracy on both LLMs: 44.2% on
DeepSeek-V4 and 35.9% on Qwen3-30B-A3B-
Instruct-2507. The level-stratified breakdown helps
explain where these gains arise. On DeepSeek-V4,
TAG performs particularly strongly on the single-
transaction L0 slice, reaching 65.6% compared
with55.7% for the strongest similarity-based base-
line. On Qwen3, TAG achieves the best or tied-bestresult across all three levels. This pattern suggests
that applicability-based retrieval is especially effec-
tive when success depends primarily on identify-
ing the governing rules. Performance differences
are less pronounced on the more complex L1/L2
cases, where solving the task additionally requires
composing multiple interacting rules after retrieval.
This indicates that RuleArena-NBA combines two
distinct challenges: retrieving the relevant con-
straints and performing downstream multi-step rule
composition. TAG directly addresses the former,
while the latter remains LLM-dependent. TAG is
also substantially more context-efficient than the
All-Rules baseline, using only 20.8 and14.0 re-
trieved rules instead of 195, while still achieving
higher strict accuracy on both LLMs. Standard
RAG remains non-monotonic with retrieval depth
(e.g., DeepSeek-V4 drops from 43.6% at top-10 to

Executor Method #Units Strict (%) L0 (%) L1 (%) L2 (%)
DeepSeek-V4No Retrieval (M0) 0 42.9 54.1 43.5 15.4
All Rules (M1) 195 39.7 50.8 39.1 15.4
Std RAG, top-5 5 40.4 55.7 37.7 11.5
Std RAG, top-10 10 43.6 49.246.4 23.1
Std RAG, top-15 15 34.6 47.5 29.0 19.2
Std RAG, top-20 20 42.9 55.7 42.0 15.4
TAG (Ours) 20.844.2 65.636.2 15.4
Qwen3-30B-A3B
-Instruct-2507No Retrieval (M0) 0 30.1 37.7 29.0 15.4
All Rules (M1) 195 34.0 45.9 29.019.2
Std RAG, top-5 5 32.1 42.6 29.0 15.4
Std RAG, top-10 10 31.4 42.6 29.0 11.5
Std RAG, top-15 15 29.5 39.3 24.619.2
Std RAG, top-20 20 32.747.526.1 15.4
TAG (Ours) 14.035.9 47.5 31.9 19.2
Table 3: Results on RuleArena-NBA ( n= 156 ).#Unitsreports the average number of retrieval units supplied to the
executor: 500-character chunks for Standard RAG, and atomic rules for All Rules and TAG; M0 retrieves nothing.
Strict accuracy requires the True/False answer, the offending transaction, and the responsible team to all be correct.
L0/L1/L2 are complexity-stratified breakdowns by single, two, and three-or-more interacting transactions.
34.6% at top-15), indicating that larger similarity-
based candidate pools do not reliably improve rule
selection.
4.6 Cross-Domain Analysis
Taken together, the three domains reveal a con-
sistent pattern: TAG is most beneficial when se-
mantic similarity is a weak proxy for the context
that governs downstream success. In Wikipedia
NPOV , where the relevant policy rules often share
limited lexical overlap with the input, TAG de-
livers its largest gains over similarity-based re-
trieval. In HumanEval with PEP 8 compliance,
where many applicable rules are directly reflected
in code structure, similarity-based retrieval is al-
ready comparatively effective, leaving less room
for applicability-based selection to improve perfor-
mance. RuleArena-NBA reveals a different limita-
tion: while TAG achieves the strongest overall strict
accuracy, the level-stratified breakdown shows that
retrieval quality is only one component of task dif-
ficulty.
A second consistent finding is that better re-
trieval does not require more context. Across
all three domains, TAG matches or exceeds the
strongest baselines while conditioning the LLMs
on substantially fewer retrieved units than the All-
Rules baseline. This suggests that the gains arise
from more precise rule selection rather than simply
providing more structured context.
Overall, the results support the central claim of
this work: in rule and instruction governed tasks,
retrieval should be aligned with applicability ratherVariantM#picks Metric (%)
(a) NPOV (DeepSeek-V4)
Full(5 phases) 119 22.168.2
−Phase 1 (span detection) 178 39.4 65.4
−Phase 2 (decomposition) 84 19.6 64.5
−Phase 3 (operational.) 118 28.4 63.6
−Phase 4 (dedup.) 127 29.7 65.4
−Phase 5 (verification) 127 24.8 64.5
−Relevance ranker (§F.5) 119 34.3 63.6
(b) RuleArena-NBA (Qwen3-30B)
Full(5 phases) 195 14.035.9
−Phase 1 (span detection) 245 17.5 34.4
−Phase 2 (decomposition) 165 13.0 32.3
−Phase 3 (operational.) 196 34.0 29.2
−Phase 4 (dedup.) 230 22.0 31.3
−Phase 5 (verification) 205 14.5 30.2
Table 4: Ablation on the five-phase rule extraction
pipeline across two domains. Panel (a): NPOV
(n=107 ) with DeepSeek-V4. Panel (b): RuleArena-
NBA ( n=156 ) with Qwen3-30B. Mis the final rule-set
size; #picks is the average number of rules retrieved.
than defaulting to semantic similarity.
4.7 Ablation: Rule Extraction Phases
To assess the contribution of each phase in the rule
extraction pipeline (§3.3), we perform leave-one-
out ablations on two domains: Wikipedia NPOV
with DeepSeek-V4 as executor, and RuleArena-
NBA with Qwen3-30B-A3B-Instruct-2507. In
each variant, the matcher (M3-pair, temperature
0) and evaluation protocol are fixed; only the ex-
tracted rule set varies. Table 4 reports the results.
Table 4 shows that Phase 3 (operationalization) is
the largest contributor in both domains, reducing

performance by 4.6VFR points on NPOV and 6.7
strict-accuracy points on RuleArena-NBA when
removed. The remaining phases form a second
tier whose ordering shifts with document structure:
on NPOV , Phases 2 (atomic decomposition) and 5
(verification) contribute most ( −3.7 each), while
on NBA the second tier is led by Phase 5 ( −5.7 )
and Phase 4 (deduplication, −4.6 ). Phase 1 (span
detection) is never the dominant factor in either
setting.
The dominant effect of Phase 3 suggests that
extraction alone is insufficient: rules must also
be operationalized into concrete, judgeable condi-
tions and executable actions. Removing this phase
increases the number of matched rules in both
domains ( 22.1→28.4 on NPOV; 14.0→34.0
on NBA), indicating that vague conditions reduce
matcher selectivity while providing less actionable
guidance to the executor. The primacy of opera-
tionalization across a short policy document and a
substantially longer procedural rulebook suggests
that this effect reflects a structural property of the
pipeline rather than a domain-specific artifact. See
detailed per-phase analysis in Appendix C.
Table 4 also includes a relevance ranker result
(§F.5). A controlled relevance-reranking baseline
shows that TAG ’s gains do not come merely from
using an LLM judge: replacing applicability match-
ing with relevance matching drops performance
to 63.6% VFR, essentially identical to the All-
Rules baseline (63.5%) and 4.6 points below TAG
(68.2%). This suggests that the key improvement
comes from framing retrieval around applicabil-
ity rather than generic LLM-based relevance judg-
ment.
4.8 Ablation: Disentangling Retrieval Unit
from Retrieval Relation
TAG changes two aspects of standard RAG at once:
the retrievalunit(text chunk →atomic rule) and
the retrievalrelation(BGE similarity →LLM pair-
wise applicability). To disentangle these effects,
we run a 2×2 factorial ablation on Wikipedia
NPOV , introducing the two off-diagonal condi-
tions: RULES+SIMILARITY(BGE retrieval over
the same atomic-rule corpus used by TAG) and
CHUNKS+APPLICABILITY(pairwise applicability
matching over raw text chunks). All four settings
share the same Qwen3-30B-A3B-Instruct-2507 ex-
ecutor and evaluation protocol. Table 5 reports the
results. We use Wikipedia NPOV for this ablation
because its large mismatch between semantic simi-Unit Relation #picks VFR (%)
Chunks Similarity (BGE) 20.0 53.3
Chunks Applicability (pair) 21.4 59.8
Atomic rules Similarity (BGE) 21.0 54.2
Atomic rules Applicability (pair) 32.664.5
Table 5: 2×2 ablation disentangling the retrieval unit
(chunks vs. atomic rules) from the retrieval relation
(BGE similarity vs. LLM pairwise applicability) on
Wikipedia NPOV (n=107, executor Qwen3-30B).
larity and rule applicability makes component-level
differences easier to observe. Both factors con-
tribute, but the retrieval relation contributes more.
With the unit fixed at chunks, replacing similarity
with applicability raises VFR from 53.3 to59.8
(+6.5 points); with the unit fixed at atomic rules,
the same replacement raises VFR from 54.2 to64.5
(+10.3). By comparison, changing the retrieval unit
yields smaller gains: +0.9 points under similarity
and +4.7 under applicability. This rules out a sim-
ple cleaner-unit explanation: even when operating
over the same atomic-rule corpus, applicability-
based retrieval substantially outperforms similarity-
based retrieval ( 64.5 vs.54.2). Rule structuring
remains complementary, but the dominant gain
comes from aligning retrieval with the downstream
applicability relation rather than semantic similar-
ity. See detailed analysis in Appendix D.
5 Conclusion
We introducedretrieval objective mismatch, a fail-
ure mode in rule and instruction governed tasks
where semantic similarity is misaligned with down-
stream utility. To address this, we proposed TAG,
which transforms source documents into condition–
action rules and retrieves context through applica-
bility judgments rather than similarity ranking.
Across three domains, TAG consistently outper-
forms standard similarity-based RAG while using
substantially fewer retrieved units than all-rule con-
ditioning, indicating that the gains arise from more
precise rule selection rather than simply providing
more structured context.
More broadly, our findings suggest that retrieval
should be aligned with the structure of the down-
stream task rather than defaulting to semantic simi-
larity. When task success depends on conditional
rules, constraints, or procedural guidance, the most
useful context may not be the most similar passage,
but the rule applicable to the input.

Limitations
TAG assumes that source documents can be faith-
fully transformed into explicit condition–action
rules. Tasks whose governing knowledge is highly
implicit, ambiguous, or primarily factual rather
than procedural may be less suitable for this formu-
lation. Our current pairwise applicability matching
also introduces additional inference cost compared
with dense retrieval, although this is partly offset
by substantially reduced context passed to the ex-
ecutor. Our RuleArena-NBA results also suggest a
second boundary: improving retrieval helps when
rule selection is the primary bottleneck, but does
not by itself resolve tasks requiring complex multi-
step rule composition. Finally, while we evaluate
across three distinct domains, broader validation on
additional instruction- and policy-governed tasks
would strengthen the generality of our conclusions.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InInternational Conference on Learning Representa-
tions (ICLR).
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan,
Henrique Ponde de Oliveira Pinto, Jared Kaplan,
Harri Edwards, Yuri Burda, Nicholas Joseph, Greg
Brockman, and 1 others. 2021. Evaluating large
language models trained on code.arXiv preprint
arXiv:2107.03374.
Zhongwu Chen, Chengjin Xu, Dingmin Wang, Zhen
Huang, Yong Dou, Xuhui Jiang, and Jian Guo. 2024.
RuleRAG: Rule-guided retrieval-augmented gener-
ation with language models for question answering.
arXiv preprint arXiv:2410.22353.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024. From local to global: A
graph RAG approach to query-focused summariza-
tion.arXiv preprint arXiv:2404.16130.
Michael Glass, Gaetano Rossiello, Md Faisal Mahbub
Chowdhury, Ankita Naik, Pengshan Cai, and Alfio
Gliozzo. 2022. Re2G: Retrieve, rerank, generate.
InProceedings of the 2022 Conference of the North
American Chapter of the Association for Computa-
tional Linguistics (NAACL).
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong C. Park. 2024. Adaptive-RAG:
Learning to adapt retrieval-augmented large language
models through question complexity. InProceedings
of the 2024 Conference of the North American Chap-
ter of the Association for Computational Linguistics
(NAACL).Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing (EMNLP).
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP).
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Informa-
tion Processing Systems (NeurIPS).
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts.Transactions of the Association
for Computational Linguistics, 12.
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. 2023. Query rewriting in retrieval-
augmented large language models. InProceedings
of the 2023 Conference on Empirical Methods in
Natural Language Processing (EMNLP).
Rodrigo Nogueira and Kyunghyun Cho. 2019. Pas-
sage re-ranking with BERT.arXiv preprint
arXiv:1901.04085.
Jiashuo Sun, Xianrui Zhong, Sizhe Zhou, and Jiawei
Han. 2025. Dynamicrag: Leveraging outputs of large
language model as feedback for dynamic reranking
in retrieval-augmented generation.arXiv preprint
arXiv:2505.07233.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st Annual Meeting of the Association for Com-
putational Linguistics (ACL).
Liang Wang, Haonan Chen, Nan Yang, Xiaolong
Huang, Zhicheng Dou, and Furu Wei. 2025. Chain-
of-retrieval augmented generation.arXiv preprint
arXiv:2501.14342.
Yuhao Wang, Ruiyang Ren, Junyi Li, Wayne Xin
Zhao, Jing Liu, and Ji-Rong Wen. 2024. REAR:
A relevance-aware retrieval-augmented framework
for open-domain question answering.arXiv preprint
arXiv:2402.17497.
Jiaqi Wei, Hao Zhou, Xiang Zhang, Di Zhang, Zijie
Qiu, Wei Wei, Jinzhe Li, Wanli Ouyang, and Siqi
Sun. 2025. Alignrag: Leveraging critique learning
for evidence-sensitive retrieval-augmented reasoning.
arXiv preprint arXiv:2504.14858.

Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas
Muennighoff. 2023. C-Pack: Packed resources
for general Chinese embeddings.arXiv preprint
arXiv:2309.07597.
Yudi Zhang, Pei Xiao, Lu Wang, Chaoyun Zhang, Meng
Fang, Yali Du, Yevgeniy Puzyrev, Randolph Yao,
Si Qin, Qingwei Lin, Mykola Pechenizkiy, Dong-
mei Zhang, Saravan Rajmohan, and Qi Zhang. 2025.
RuAG: Learned-rule-augmented generation for large
language models. InInternational Conference on
Learning Representations (ICLR).
Ruiwen Zhou, Wenyue Hua, Liangming Pan, Sitao
Cheng, Xiaobao Wu, En Yu, and William Yang Wang.
2025. RuleArena: A benchmark for rule-guided rea-
soning with LLMs in real-world scenarios. InPro-
ceedings of the 63rd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers), pages 550–572, Vienna, Austria. Associa-
tion for Computational Linguistics.
A Setup Details
Wikipedia NPOV .Cases are drawn from the
Wikipedia NPOV Noticeboard between 2020 and
2024. Each case consists of a flagged sentence, a
violation description written by an editor, and a
link to the source article. We retain only cases for
which the violation is localised to a single sentence
and the editor’s description is specific enough to
anchor objective scoring, yielding 107cases. The
NPOV policy ( ∼15K words) is processed by the
five-phase pipeline into M=119 rules after verifi-
cation. To prevent reward hacking through trivial
pass-through, we automatically set VFR = 0 for
any rewrite whose character similarity to the input
exceeds98%.
HumanEval ×PEP 8.HumanEval (Chen et al.,
2021) provides 164function signatures and doc-
strings to be completed. PEP 8 ( ∼8K words) is
processed into 127atomic rules covering inden-
tation, naming, whitespace, line wrapping, im-
ports, expression formatting, and other style di-
mensions. For Pass@1, we prepend the prompt
as setup code before executing the body, follow-
ing the official HumanEval protocol (which han-
dles cases such as HumanEval/38, where helper
functions are defined in the prompt). The pylint
evaluation uses the JSON2 output format with the
default ruleset, with missing-final-newline and
trailing-newlinesdisabled.
RuleArena-NBA.The NBA subset comprises
156 problems stratified across three complex-
ity levels: L0 ( 61problems, single transaction),L1 (69problems, two interacting transactions),
and L2 ( 26problems, three or more interact-
ing transactions with sequential reasoning). The
NBA Collective Bargaining Agreement ( ∼60K
words) is processed into 195atomic rules cover-
ing salary caps, contract limits, exception types
(BAE/NTMLE/MLE/supermax), free-agency me-
chanics, and trade restrictions. For brevity, we
use the following CBA abbreviations throughout
the paper:MLE(Mid-Level Exception),NTMLE
(Non-Taxpayer Mid-Level Exception, the higher
MLE variant available to non-tax teams),BAE(Bi-
Annual Exception),TPE(Traded Player Excep-
tion), andsupermax(Designated Veteran Player
Extension).
B Evaluation Protocols
NPOV scoring.The four 1–5 auxiliary scores
are:Remediation(how completely the rewrite re-
solves the flagged issue),Preservation(how faith-
fully the rewrite preserves the original factual con-
tent),Tone(whether the rewrite reads as ency-
clopaedic prose), andFluency(grammatical and
stylistic naturalness). Both VFR and the auxiliary
scores are produced by an LLM evaluator; the full
evaluator prompt is in App. F.4.1.
Code metrics.Pass@1 is obtained from the of-
ficial HumanEval test suite. Pylint operates over
[−10,10] . Radon reports the average cyclomatic
complexity per function; Bandit reports the number
of security issues flagged. Radon and Bandit are
reported as secondary indicators only.
RuleArena-NBA strict accuracy.A problem is
strict-correct only if the True/False answer, the
identified offending transaction, and the identified
responsible team are all correct. Partial credit is
not awarded.
C Detailed Analysis for Rule Extraction
Phases
NPOV results.Panel (a) of Table 4 reports the
per-phase costs on NPOV . Phase 3 (operational-
ization) contributes the most: removing it costs
4.6VFR points ( 68.2→63.6 ), the largest drop
among all variants. Without this phase, vague trig-
gers such as “when appropriate” and aspirational
actions such as “improve the tone” remain in the
rule set, and the matched rules, while still topically
correct, do not give the executor operations that can
be carried out without further interpretation. The

Figure 3: Five-phase rule extraction applied to a passage from the NPOV policy.
drop suggests that downstream performance de-
pends not only on which rules are selected, but also
on whether the selected rules are concrete enough
to act on, and that operationalization is what turns
otherwise plausible extracted prose into executable
guidance. Phases 2 and 5 are also clearly impor-
tant, with costs of 3.7points each ( 68.2→64.5 ).
Removing Phase 5 allows unverified rules whose
source spans cannot be matched in the policy to
remain in the rule set; the matcher still selects a
similar number of rules per query ( 24.8 #picks),
but these unsupported rules may crowd out use-
ful ones in the matched set and can introduce in-
structions not grounded in the source document.
Removing Phase 2 (atomic decomposition) leaves
compound rules intact, forcing the executor to dis-
entangle multiple prescribed actions per item and
raising the chance that one action is applied while
another is dropped. We note that removing Phase 2
reduces#picks ( 22.1→19.6 ): without decompo-
sition the rule set contains fewer, coarser items,
so the matcher returns fewer matches, but each
bundles several actions and is harder to execute
faithfully, so the cost reflects degraded actionabil-
ity per matched rule rather than an inflated matched
set. These two phases target failure modes distinct
from Phase 3, namely hallucination and bundling
rather than vagueness, and their costs confirm
that all three are load-bearing components rather
than incremental refinements. Phases 4 (dedupli-cation) and 1 (span detection) have the smallest
and equal effect on NPOV , each costing 2.8points
(68.2→65.4 ). Without Phase 4, near-duplicate
rules inflate the matched set (#picks rises from
22.1 to29.7), but the additional rules carry little
new information, so the effect appears to be mainly
increased prompt load rather than a large change
in the available information. The small effect of
Phase 1 we attribute to the fact that the NPOV pol-
icy is densely normative, leaving relatively little
expository material for Phase 1 to filter.
RuleArena-NBA results.Panel (b) of Table 4 re-
ports the same ablation on a substantially different
setting: a longer, more procedural source document
and a smaller open-weight executor. The dominant
phase is shared with NPOV . Phase 3 (operational-
ization) is again the single largest contributor, with
a drop of 6.7strict-accuracy points ( 35.9→29.2 ).
A similar pattern appears in NBA: without oper-
ationalization, the rule set retains vague triggers
and under-specified actions, which on NBA trans-
late into conditions such as “when the contract is
sufficiently large” or actions such as “apply the rel-
evant exception”. These survive into the matched
set but cannot be turned into a concrete legality
judgment over specific transactions. #picks rises
sharply (from 14.0 to34.0), indicating that vague
conditions make the matcher less discriminative,
so the executor is simultaneously given more rules

and less actionable ones.
Phases 5, 4, and 2 form a second tier, with costs
of5.7,4.6, and 3.6points respectively. The inter-
nal ordering differs from NPOV: Phase 4 (dedu-
plication) is comparatively more costly on NBA,
and #picks rises from 14.0 to22.0 when it is re-
moved. We attribute this to the structure of the
CBA, which restates similar provisions across ex-
ception types (e.g., parallel rules for MLE, BAE,
and NTMLE), so near-duplicates are more preva-
lent than in NPOV and crowd the matched set more
aggressively when not collapsed. Phase 5 (verifi-
cation) is also more costly on NBA than on NPOV ,
consistent with a longer source document offering
more opportunity for ungrounded rules to survive
without verification. Phase 2 (atomic decomposi-
tion) is slightly less costly on NBA than on NPOV ,
likely because CBA provisions are already drafted
in a more itemized style, leaving fewer compound
rules to split. Phase 1 (span detection) is again
the smallest contributor ( −1.5 ), even though the
CBA contains more non-normative content than the
NPOV policy; we suspect the gain from filtering
is partially offset by the matcher’s own ability to
reject non-applicable rules at inference time, and
leave a finer-grained analysis to future work.
Cross-Domain Consistency of Phase Contribu-
tionsAcross both panels of Table 4, two contribu-
tions are consistent despite substantial differences
in source-document length, document style, and
executor type. Phase 3 (operationalization) is the
largest contributor in both settings, with drops of
4.6points on NPOV ( 68.2→63.6 ) and 6.7points
on NBA ( 35.9→29.2 ), and in both cases it leads
the next-most-costly phase by a clear margin. At
the other end, Phase 1 (span detection) is never
the dominant factor, costing 2.8points on NPOV
and1.5on NBA. The ranking of the middle phases,
however, isnotstable: on NPOV the second tier is
Phases 2 and 5 (tied at 3.7), with Phases 1 and 4
below them (tied at 2.8), whereas on NBA the or-
der is Phase 5 ( 5.7)>Phase 4 ( 4.6)>Phase 2
(3.6)>Phase 1 ( 1.5). The fact that the primacy
of operationalization holds under a 4×difference
in document length ( ∼15K vs. ∼60K words) and
across two distinct executors suggests that this con-
tribution reflects a structural property of the extrac-
tion pipeline, while the reordering of the middle
phases reflects properties of the source document
itself, as detailed below.Operationalization as the Dominant Bottleneck
Phase 3 separates itself from the other phases by a
clear margin on both domains. On NPOV , remov-
ing it incurs a 4.6-point drop, which is 0.9points
larger than the next-most-costly variants (Phases 2
and 5 at 3.7); on NBA, the same ablation incurs a
6.7-point drop, 1.0point larger than Phase 5 ( 5.7).
A consistent secondary signal accompanies this
drop: #picks rises when Phase 3 is removed, from
22.1 to28.4 on NPOV and, far more sharply, from
14.0 to34.0 on NBA. The conjunction of a larger
matched set and a lower downstream metric indi-
cates that the failure mode is not under-retrieval but
under-specification: vague conditions degrade the
matcher’s discriminativeness while simultaneously
failing to provide actionable guidance to the execu-
tor, so additional rules translate into prompt bloat
rather than useful instruction.
Domain-Sensitive Reordering Within the Sec-
ond TierApart from the consistently dominant
role of Phase 3 and the consistently minor role of
Phase 1, the relative costs of the middle phases
shift in interpretable ways between the two do-
mains. Phase 4 (deduplication) is comparatively
more costly on NBA ( 4.6points, with #picks rising
from 14.0 to22.0) than on NPOV ( 2.8points, with
#picks rising from 22.1 to29.7), consistent with the
parallel-provision structure of the CBA, where sim-
ilar clauses recur across exception types. Phase 5
(verification) is likewise more costly on NBA ( 5.7
vs.3.7points), consistent with a longer source doc-
ument leaving more room for ungrounded rules
to survive. Conversely, Phase 2 (atomic decom-
position) is less costly on NBA ( 3.6points) than
on NPOV ( 3.7points), consistent with the more
itemized drafting style of regulatory text, which
leaves fewer compound rules to split. These shifts
rearrange the middle phases but do not alter the
position of Phase 3 as the largest contributor or
Phase 1 as the smallest in either domain.
Bounded Absolute Gains on the Composition-
Limited DomainThe absolute spread of per-
phase costs is comparable across domains ( 4.6
points on NPOV , 6.7points on NBA), but the ceil-
ings differ substantially: the full pipeline reaches
68.2 VFR on NPOV against 35.9 strict accuracy
on NBA. As a fraction of the full-pipeline metric,
the cost of removing Phase 3 is therefore 6.7% on
NPOV but 18.7% on NBA, meaning the extraction
pipeline accounts for a substantially larger share of
attainable performance on NBA in relative terms,

even though the headroom in absolute points is
broadly similar. This pattern is consistent with the
broader finding in §4.5 that NBA is bottlenecked by
multi-step rule composition rather than rule selec-
tion alone: a clean, operationalized rule set is nec-
essary to recover the gains that are recoverable, but
the absolute level remains constrained by reasoning
demands that no extraction phase can address.
D Detailed Analysis for Disentangling
Retrieval Unit from Retrieval Relation
TAG changes two aspects of standard RAG at once:
the retrievalunit(text chunk →atomic rule) and
the retrievalrelation(BGE similarity →LLM pair-
wise applicability). The main results in Table 1
only compare the two corner cells (Chunks + Sim-
ilarity vs. Rules + Applicability), which conflates
these two design choices. To separate them, we
run a 2×2 factorial ablation on NPOV that fills in
the two off-diagonal cells. RULES+ SIMILARITY
applies BGE over the same 119 atomic rules used
by TAG (embedding the concatenation of name ,
condition , and action ), with top- kchosen to ap-
proximately match the average number of picks of
the applicability-based matcher. CHUNKS+ AP-
PLICABILITYapplies the same pairwise applica-
bility judge over the NPOV chunks. All four cells
share the same Qwen3-30B-A3B-Instruct-2507 ex-
ecutor and the same evaluation protocol.
Both axes contribute to the overall improvement,
but the relation contributes more. With the unit
fixed at chunks, switching from similarity to appli-
cability raises VFR by 6.5 points ( 53.3→59.8 );
with the unit fixed at rules, the same switch raises
VFR by a larger 10.3 points ( 54.2→64.5 ). With
the relation fixed at similarity, switching from
chunks to rules raises VFR by only 0.9 points
(53.3→54.2 ); with the relation fixed at appli-
cability, the same switch raises VFR by 4.7 points
(59.8→64.5 ). On both axes, therefore, the larger
of the two single-factor lifts is observed when the
otherfactor is already aligned with the downstream
task, indicating that the two design choices rein-
force rather than substitute for each other.
This is confirmed by an additivity check at the di-
agonal (RULES+ APPLICABILITY, 64.5): adding
the two single-axis lifts to the CHUNKS+ SIM-
ILARITYcell gives a linear prediction of 53.3 +
6.5 + 0.9 = 60.7 , which underestimates the ob-
served 64.5 by3.8points. The joint move from
CHUNKS+ SIMILARITYto RULES+ APPLICA-BILITYtherefore yields a modest super-additive
gain on top of the two main effects.
Two further patterns are worth noting. First,
CHUNKS+ APPLICABILITYalone ( 59.8) already
closes most of the gap between CHUNKS+ SIM-
ILARITY( 53.3) and the full method ( 64.5), sug-
gesting that on NPOV the applicability relation is
the dominant single factor and that rule structuring
contributes a smaller, complementary refinement
on top of it. Second, even when operating over the
same atomic-rule corpus, RULES+ SIMILARITY
trails RULES+ APPLICABILITYby 10.3 points
(54.2 vs.64.5). This rules out the possibility that
the gain is an artifact of feeding the executor a
cleaner retrieval unit, and confirms that the relation
itself is decisive. Together, these results support the
central claim of the paper: the benefit of TAG on
NPOV comes primarily from replacing similarity
with applicability, with rule structuring providing
a smaller complementary improvement that com-
pounds with the relation change.
E Discussion
E.1 When Does TAG Help?
The cross-domain results suggest three conditions
under which the framework is most beneficial. The
first is a high abstraction gap, where the inputs
are concrete, and the applicable rules are stated
abstractly, so that the two share little semantical
similarity. NPOV violations evaluated against the
NPOV policy are the clearest case. The second
is strong conditionality, where each rule carries
a sharp fire-when clause and applicability there-
fore becomes a well-defined binary question that
an LLM judge can answer reliably; the CBA is
structured in this way. The third is high distractor
density, where the document contains many rules
that look superficially similar to the input but are
not the applicable ones, such as regular max-salary
rules versus supermax extensions. Embedding sim-
ilarity is most likely to fail within such regions.
E.2 When Does It Not Help?
Three settings limit the benefit. When rules and
inputs share semantical similarity, as with PEP 8
style rules and Python code, a strong similarity re-
triever already retrieves the relevant context, and
applicability matching adds little. When the total
number of rules is small enough that an All Rules
baseline fits comfortably in the context, retrieval
ceases to be the bottleneck. And when the bot-

tleneck is multi-step rule composition rather than
rule selection, an improved retriever does not help;
the L2 slice of RuleArena-NBA is precisely such a
setting.
E.3 What Are We Actually Contributing?
We wish to be clear about what TAG does and does
not claim. The contribution is not a claim that
LLMs are superior to dense retrievers in general,
nor that pairwise judgment by an LLM is a uni-
versally better reranking step. The contribution is
narrower: for documents whose useful units are
conditional rules, retrieval should optimize condi-
tion satisfaction rather than semantical similarity,
and this reframing is most consequential precisely
where the two relations diverge. The experiments
are designed to support that narrow claim by show-
ing both where it holds (NPOV) and where it does
not (PEP 8).
=====================================================================
F Prompts
This appendix lists, in full, every prompt used by
TAG and by the LLM-based evaluators reported in
the main paper. All prompts are reproduced ver-
batim from our experimental code, with runtime-
substituted fields written as {slot_name} . Unless
otherwise stated, every LLM call uses tempera-
ture0, top-p= 1.0 , and a maximum output length
of4096 tokens. System and user messages are
shown separately wherever the underlying API dis-
tinguishes them.
We organise this appendix to mirror the three-
stage pipeline of §3. §F.1 covers offline rule extrac-
tion (Stage 1); §F.2 covers pairwise applicability
matching (Stage 2); §F.3 covers rule-guided ex-
ecution (Stage 3). §F.4 gives the LLM-as-judge
evaluation prompts, and §F.5 gives the controlled
relevance-reranker baseline used to isolate the con-
tribution of the applicability framing.
F.1 Stage 1: Rule Extraction Prompts
The five-phase extraction pipeline converts an
instructional document into a set of structured
condition–action rules. Phases 1–4 are imple-
mented as LLM calls with fixed JSON output
schemas;Phase 5 is implemented programmati-
cally as a verification step, not as an additional
LLM call.The phases address distinct extraction
failure modes, identified in §4.7: missed normative
content (Phase 1), bundled obligations (Phase 2),unjudgeable or inexecutable rules (Phase 3), redun-
dancy and conflict (Phase 4), and ungrounded or
uncovered rules (Phase 5).
F.1.1 Phase 1: Normative Span Detection
Purpose.Phase 1 identifies source spans that ex-
press prescriptive, proscriptive, permissive, or con-
ditional guidance. The objective is high recall over
normative content while preserving verbatim trace-
ability to the source document.
System message.
You are an expert annotator for instructional
and policy documents.,→
Your task is to identify normative spans:
text segments that prescribe,,→
prohibit, recommend, permit, or conditionally
require an action. A span,→
is normative if it tells an actor what should,
must, may, should not, or,→
must not be done, or if it defines a
condition under which such an,→
action applies.
Extract spans conservatively and preserve
traceability to the source,→
document. Do not paraphrase source text.
Domain:
{domain}
User message.
Task:
Extract all normative spans from the document
section below.,→
A normative span includes:
- requirements, obligations, or mandatory
instructions;,→
- prohibitions or discouraged practices;
- recommendations, preferences, or best
practices;,→
- permissions or exceptions that affect
whether a rule applies;,→
- conditional statements that specify when an
action should or should,→
not be taken.
Exclude:
- purely descriptive background;
- historical or motivational prose;
- examples that do not themselves state a
rule;,→
- definitions that do not constrain behavior,
unless the definition is,→
necessary to determine when a rule applies.
Document section:
"""
{doc}
"""

Output a JSON array. Each item must have the
following fields:,→
{
"span_id": "S-{start_id:03d}",
"text": "<verbatim span copied exactly from
the document>",,→
"normative_type": "requirement" |
"prohibition" | "recommendation",→
| "permission" | "exception"
| "conditional",,→
"context_summary": "<one concise sentence
describing the local topic>",→
}
Requirements:
1. The "text" field must be an exact
substring of the document section.,→
2. Do not rewrite, normalize, or merge
non-contiguous text.,→
3. Prefer the shortest span that preserves
the complete normative,→
meaning.
4. Include all normative spans, including
weak recommendations such as,→
"prefer", "avoid", "normally", "should",
and "recommended".,→
5. Return only the JSON array. Do not include
markdown fences or,→
explanations.
F.1.2 Phase 2: Atomic Decomposition
Purpose.Phase 2 decomposes bundled normative
spans into atomic units. An atomic unit contains
at most one primary subject, one applicability con-
text, and one prescribed action or constraint. This
addresses thebundlingfailure mode identified in
the ablation of §4.7: without decomposition, the
executor must disentangle multiple prescribed ac-
tions per matched item, increasing the chance that
one action is applied while another is dropped.
System message.
You are an expert at decomposing normative
language into atomic rule,→
units.
An atomic unit contains one primary subject,
one applicability context,,→
and one prescribed action or constraint. The
goal is to split bundled,→
spans into units that can later be converted
into independently,→
checkable condition-action rules.
Preserve source traceability. Do not
introduce new obligations that are,→
not entailed by the source span.
Domain:
{domain}
User message.Task:
Decompose the following normative spans into
atomic units.,→
Input spans:
{spans}
Decomposition rules:
1. Split a span when it contains multiple
independently enforceable,→
actions.
2. Split a span when it contains alternative
conditions, such as,→
"if A or B".
3. Split a span when different subjects are
governed by different,→
actions.
4. Do not split conjunctive conditions that
must hold together, such,→
as "if A and B".
5. Do not split exception clauses away from
the rule they qualify.,→
6. Do not create a unit unless it is entailed
by the original span.,→
7. Preserve the original wording as much as
possible. If minor,→
rewriting is needed to make the unit
self-contained, keep the,→
meaning faithful.
Output a JSON array. Each item must have the
following fields:,→
{
"atomic_id": "A-001",
"source_span_id": "S-XXX",
"text": "<self-contained atomic normative
unit>",,→
"original_text": "<verbatim source span
text>",,→
"was_split": true | false,
"split_rationale": "<short reason if
was_split is true, else null>",→
}
Requirements:
- Each atomic unit should express at most one
condition-action relation.,→
- The "original_text" field must be copied
from the corresponding input,→
span.
- Return only the JSON array. Do not include
markdown fences or,→
explanations.
F.1.3 Phase 3: Operationalisation
Purpose.Phase 3 converts each atomic unit into an
operational rule with ajudgeablecondition and an
executableaction. The ablation in Table 4 identifies
this phase as the largest single contributor to down-
stream performance, with removal costing −4.6
VFR on Wikipedia NPOV and −6.7 strict-accuracy
points on RuleArena-NBA. The dominance of this
phase reflects the fact that vague triggers and as-
pirational actions simultaneously degrade matcher
selectivity and reduce the actionability of the rules
passed to the executor.

System message.
You are an expert at converting atomic
normative statements into,→
structured, operational rules for
instruction-following systems.,→
Each rule must have:
1. a judgeable condition: a state of the task
input that can be checked,→
by inspecting the input;
2. an executable action: a concrete operation
that a downstream executor,→
can perform;
3. a faithful source: the original text
supporting the rule;,→
4. concise tags for grouping related rules.
Avoid vague, aspirational, or purely topical
rules. The output must be,→
suitable for pairwise applicability matching
and downstream execution.,→
Domain:
{domain}
User message.
Task:
Convert each atomic normative unit into an
operational rule.,→
Input atomic units:
{atomics}
For each unit, produce one structured rule
with the following fields:,→
{
"rule_id": "R-001",
"source_atomic_id": "A-XXX",
"rule_name": "<3-8 word distinctive rule
name>",,→
"condition": "<detectable condition under
which the rule applies>",,→
"action": "<concrete operation to perform
when the condition holds>",,→
"source_text": "<verbatim source text
supporting the rule>",,→
"category_tags": ["<tag1>", "<tag2>"]
}
Quality requirements:
1. The condition must be checkable from a
task input without external,→
knowledge unless the task itself provides
that knowledge.,→
2. The condition must not be merely topical.
It should specify when the,→
rule applies.
3. The action must specify what the executor
should do, not merely state,→
a goal.
4. Avoid vague actions such as "improve",
"ensure quality",,→
"be appropriate", or "make better" unless
they are rewritten into,→
concrete operations.5. The source_text must be copied from the
corresponding atomic unit's,→
original source text.
6. Use 1-3 concise category_tags that capture
the rule family.,→
7. Do not invent rules not supported by the
atomic unit.,→
Examples of operationalisation:
(Wikipedia NPOV)
- Weak condition: "The text is biased."
- Better condition: "The sentence states an
evaluative characterization,→
in the author's voice
without
attribution.",→
,→
- Weak action: "Make the text neutral."
- Better action: "Replace the evaluative
characterization with a,→
neutral descriptive term
or attribute it to a
named,→
,→
source if such a source
is present.",→
(PEP 8)
- Weak condition: "The code has import
problems.",→
- Better condition: "A wildcard import (from
<module> import *) is used.",→
- Weak action: "Improve the imports."
- Better action: "Replace the wildcard
import with explicit imports,→
that list each imported
name.",→
(NBA CBA)
- Weak condition: "The transaction involves
apron rules.",→
- Better condition: "A team attempts a
transaction listed in the,→
Transaction Restrictions
Table, and the
transaction,→
,→
has a defined Applicable
Apron Level.",→
- Weak action: "Check whether the
transaction is allowed.",→
- Better action: "Compute the team's salary
immediately after the,→
transaction and flag a
violation if it
exceeds the,→
,→
corresponding Applicable
Apron Level.",→
Return only the JSON array. Do not include
markdown fences or,→
explanations.
F.1.4 Phase 4: Deduplication and Conflict
Resolution
Purpose.Phase 4 compares rules that share at
least one category tag and identifies duplicate, sub-
suming, overlapping, or conflicting pairs. Tag-
restricted comparison avoids O(M2)scaling, while
the resulting labels support both automatic merging

(for duplicates) and downstream diagnostics (for
conflicts).
System message.
You are a rule-set quality auditor.
Your task is to compare candidate operational
rules and identify,→
redundancy, overlap, and conflicts. The goal
is to improve the rule set,→
while preserving coverage and traceability.
Classify only meaningful relationships. If
two rules are independent,,→
omit them from the output.
Domain:
{domain}
User message.
Task:
Classify the relationship between each pair
of rules below.,→
Rule pairs:
{rules}
Relationship labels:
- "duplicate": the two rules have
substantially the same condition and,→
prescribe substantially the same action.
They should be merged or one,→
should be removed.
- "subsumption": one rule is a more specific
case of the other. Both,→
may be kept if the narrower rule adds useful
boundary information.,→
- "overlap": the rules share part of their
applicability space but,→
neither fully subsumes the other.
- "conflict": the rules can apply to the same
input but prescribe,→
incompatible actions.
- "independent": the rules govern different
situations or actions.,→
Output only non-independent pairs.
For each non-independent pair, output:
{
"rule_i": "R-XXX",
"rule_j": "R-YYY",
"relationship": "duplicate" | "subsumption"
| "overlap" | "conflict",,→
"preferred_action": "merge" | "keep_both" |
"manual_review",,→
"explanation": "<one concise sentence>"
}
Decision guidelines:
1. Use "duplicate" only when both condition
and action are effectively,→
the same.
2. Use "subsumption" when one rule is clearly
more general and the other,→
is a special case.3. Use "overlap" when the rules intersect but
neither rule contains the,→
other.
4. Use "conflict" only when following both
rules would lead to,→
incompatible executor behavior.
5. Do not report weak topical similarity.
If all pairs are independent, return [].
Return only the JSON array. Do not include
markdown fences or,→
explanations.
F.1.5 Phase 5: Verification
Purpose.Phase 5 verifies the extracted rule set us-
ingprogrammaticchecks rather than an additional
LLM call. It reports three diagnostics – Faithful-
ness, Coverage, and Independence – and acts on
each according to the policy below. Faithfulness
failures result inrule removal; Coverage failures
trigger Phase 3 re-extractionover the uncovered
spans; Independence is reported as adiagnostic
and does not by itself cause rule removal.
Faithfulness.For each rule, we verify whether
itssource_text can be matched to the origi-
nal document. Exact substring matches are ac-
cepted directly. If no exact match is found,
we compute a character-level fuzzy match us-
ing difflib.SequenceMatcher(None, src,
w).ratio() over sliding windows wof the docu-
ment, with stride 50characters and window length
|src|+ 50 . A rule is counted as faithful when the
best match score exceeds τfaith= 0.85 . Rules
whose source cannot be matched above τfaithare
discarded.
Coverage.For each normative span sjextracted in
Phase 1, we check whether at least one surviving
operational rule has a source_text that overlaps
sjat the character level by at least 50%. Uncov-
ered spans are sent back through Phase 3 for re-
extraction; spans still uncovered after re-extraction
are logged and excluded from the final rule set.
Independence.We compute the ratio of unique
rule names to total rules after deduplication. The
ratio is reported as a diagnostic and is not used to
drop rules; rules that survive Phase 4 but share a
name with another surviving rule are flagged for
manual inspection only.
F.2 Stage 2: Pairwise Applicability Matching
Prompts
For methods using applicability-based retrieval, we
evaluate each input–rule pair independently. The

matcher is asked to decide whether the input sat-
isfies the rule’s condition, rather than whether the
rule is topically similar or generally useful.
Design note.In both prompts below, only the rule’s
rule_id ,rule_name ,condition , and tag-level
metadata are exposed to the matcher; the action
field is deliberately withheld. Exposing the ac-
tion biases the judge toward rules with appealing
actions even when the condition does not hold, con-
flating applicability with action desirability. This
withholding is the central distinction between the
applicability matcher used by TAG and the con-
trolled relevance reranker reported in §F.5.
A singlegeneralapplicability matcher is used
for Wikipedia NPOV and HumanEval + PEP 8
(§F.2.1). The RuleArena-NBA matcher uses a
domain-specific schema with explicit boundary-
condition handling (§F.2.2), reflecting the structure
of the CBA, where many rules differ only by ex-
ception type, transaction type, or contract class.
F.2.1 General Applicability Matcher
Purpose.For each pair (x, r i), the matcher decides
whether the input xsatisfies the condition of rule
ri. The decision is intentionally formulated as con-
dition satisfaction rather than topical relevance or
general usefulness.
Input.A task input xand one candidate rule ri.
Only the rule’s rule_id ,rule_name ,condition ,
andcategory_tags are shown to the matcher. The
rule’s action is withheld to prevent the judge from
selecting rules because their actions appear useful
even when their conditions do not hold.
Output.A JSON object with a binary verdict:
{"verdict": "YES"}or{"verdict": "NO"}.
System message.
You are a strict applicability judge for
instruction-following tasks.,→
Given one task input and one candidate rule
extracted from an,→
instruction document, decide whether the rule
is APPLICABLE to the,→
input. A rule is applicable if, and only if,
the observable condition,→
of the rule is satisfied by the input.
Your decision should be based on condition
satisfaction, not topical,→
similarity, general usefulness, or whether
the rule sounds helpful.,→
Be conservative: if the condition is
ambiguous, underspecified, or not,→
directly supported by the input, answer NO.Output ONLY a single JSON object. Do not
include reasoning,,→
explanations, markdown fences, or any other
text.,→
User message.
Task input:
"""
{query}
"""
Candidate rule:
- rule_id: {rule_id}
- rule_name: {rule_name}
- condition: {condition}
- category_tags: {tags}
Question:
Does the task input satisfy the candidate
rule's condition?,→
Decision criteria:
- Answer YES only if the condition clearly
holds for this specific,→
input.
- Answer NO if the rule is merely topically
related but its condition,→
is not satisfied.
- Answer NO if deciding applicability would
require information not,→
present in the input.
- Answer NO if applying the rule would
require inventing facts,,→
assumptions, sources, or hidden context.
- Borderline or uncertain cases default to NO.
Output format:
{"verdict": "YES"} or {"verdict": "NO"}
F.2.2 RuleArena-NBA Matcher
The NBA Collective Bargaining Agreement con-
tains many rules that differ only by boundary
conditions on a particular exception type, trans-
action type, or contract class. The matcher
prompt therefore exposes an expanded rule schema
(applies_when ,constraint ,violation_check ,
does_not_apply_to ) and adds explicit decision
criteria for distinguishing exception-specific from
generic regulatory rules.
System message.
You are a strict applicability judge for NBA
Collective Bargaining,→
Agreement (CBA) rules.
Given one transaction scenario and one
candidate CBA rule, decide,→
whether the rule is applicable to evaluating
the scenario. A rule is,→
applicable only if the scenario contains
observable facts satisfying,→

the rule's non-trivial preconditions. Do not
select a rule merely,→
because it is topically related to salary-cap
reasoning.,→
Be conservative. If the scenario does not
provide enough information,→
to verify the rule's preconditions, or if a
veto condition applies,,→
answer false.
Output only a single JSON object.
User message.
Scenario:
"""
{scenario}
"""
Candidate rule:
- rule_id: {rule_id}
- rule_name: {rule_name}
- primary_tag: {primary_tag}
- applies_when: {applies_when}
- constraint: {constraint}
- violation_check: {violation_check}
- does_not_apply_to:
{does_not_apply_to}
Question:
Is this candidate rule applicable to
evaluating the transaction,→
scenario?
Decision criteria:
1. Match the rule to the primary transaction
type in the scenario,,→
such as signing, trade, extension,
renegotiation, qualifying offer,,→
or sign-and-trade. Reject the rule if its
transaction type does,→
not match the scenario.
2. Answer true only when the scenario
contains at least one,→
substantive fact satisfying the rule's
non-trivial precondition.,→
3. Do not treat generic procedural phrases,
such as "evaluate",,→
"check", "compute", or "test", as
substantive preconditions.,→
4. If the rule concerns a specific exception,
contract type, or,→
mechanism, such as MLE, BAE, TPE, two-way
contract, 10-day,→
contract, rookie extension, or designated
veteran extension, the,→
scenario must mention that mechanism or
clearly instantiate it.,→
5. If the rule governs a general regulatory
constraint, such as cap,,→
tax, apron, maximum salary, salary growth,
or contract length, the,→
scenario does not need to name the legal
term explicitly, but it,→
must contain the facts needed to evaluate
the constraint.,→6. If any item in does_not_apply_to matches
the scenario, answer,→
false.
7. When several closely related rules differ
only by boundary,→
conditions, answer true only for the rule
whose boundary condition,→
is specifically supported by the scenario.
8. Borderline or uncertain cases default to
false.,→
Output exactly one JSON object:
{
"applicable": true | false,
"reason": "<one short sentence citing the
decisive scenario fact or veto>",→
}
F.3 Stage 3: Executor Prompts
The executor prompt is held fixed within each
retrieval-unit type: rule-mode prompts are used by
TAG and by the All-Rules baseline, while chunk-
mode prompts are used by Standard RAG. Methods
differ only in the reference material supplied to the
executor ( {reference_rules} in the templates be-
low), allowing the comparison in the main tables
to isolate the retrieval mechanism.
F.3.1 Wikipedia NPOV Rewriting Executor
Rule-mode system message.
You are a Wikipedia editor revising a
sentence for compliance with,→
Wikipedia's Neutral Point of View (NPOV)
policy.,→
You will receive one input sentence and a
list of candidate rewrite,→
actions extracted from the NPOV policy. Treat
the provided actions as,→
the only policy guidance available for this
task. Apply any action,→
whose required edit can be performed using
only information already,→
present in the input sentence.
Do not introduce new facts, sources,
citations, viewpoints, or external,→
context. If an action requires information
not present in the input,,→
do not apply that action. Preserve the
original topic and factual,→
content as much as possible while making the
sentence more neutral.,→
Output exactly three lines in the following
format:,→
Applied rules: <comma-separated rule IDs from
the provided list, or NONE>,→
Reasoning: <one concise sentence explaining
the concrete edit performed>,→
Rewrite: <the revised sentence on a single
line>,→

If no provided action can be applied without
adding external,→
information, write Applied rules: NONE and
copy the input sentence,→
unchanged in the Rewrite line.
Rule-mode user message.
Task:
Revise the input sentence to better comply
with Wikipedia's Neutral,→
Point of View policy by applying the provided
rewrite actions.,→
Candidate rewrite actions:
{reference_rules}
Input sentence:
"""
{query}
"""
Instructions:
1. Apply only actions whose required edit can
be performed using,→
information already present in the input
sentence.,→
2. Do not add new facts, sources, citations,
viewpoints, or external,→
context.
3. Preserve the original factual content and
topic as much as possible.,→
4. If multiple actions can be applied, apply
the subset that produces,→
the most direct and minimal neutral
rewrite.,→
5. If no provided action can be applied
without adding external,→
information, output NONE and leave the
sentence unchanged.,→
Output exactly three lines:
Applied rules: <comma-separated rule IDs from
the provided list, or NONE>,→
Reasoning: <one concise sentence explaining
the concrete edit performed>,→
Rewrite: <the revised sentence on a single
line>,→
Chunk-mode system message.
You are a Wikipedia editor revising a
sentence for compliance with,→
Wikipedia's Neutral Point of View (NPOV)
policy.,→
You will receive one input sentence and
several excerpts from the NPOV,→
policy. Treat the provided excerpts as the
only policy guidance,→
available for this task. Apply any concrete
guidance from the excerpts,→
whose required edit can be performed using
only information already,→
present in the input sentence.Do not introduce new facts, sources,
citations, viewpoints, or external,→
context. If the guidance requires information
not present in the input,,→
do not apply it. Preserve the original topic
and factual content as,→
much as possible while making the sentence
more neutral.,→
Output exactly three lines in the following
format:,→
Applied rules: POLICY or NONE
Reasoning: <one concise sentence explaining
the concrete guidance applied>,→
Rewrite: <the revised sentence on a single
line>,→
If no policy guidance can be applied without
adding external,→
information, write Applied rules: NONE and
copy the input sentence,→
unchanged in the Rewrite line.
Chunk-mode user message.
Task:
Revise the input sentence to better comply
with Wikipedia's Neutral,→
Point of View policy by using the provided
policy excerpts.,→
Policy excerpts:
{reference_rules}
Input sentence:
"""
{query}
"""
Instructions:
1. Apply only guidance from the excerpts
whose required edit can be,→
performed using information already
present in the input sentence.,→
2. Do not add new facts, sources, citations,
viewpoints, or external,→
context.
3. Preserve the original factual content and
topic as much as possible.,→
4. If multiple pieces of guidance can be
applied, apply the subset that,→
produces the most direct and minimal
neutral rewrite.,→
5. If no provided guidance can be applied
without adding external,→
information, output NONE and leave the
sentence unchanged.,→
Output exactly three lines:
Applied rules: POLICY or NONE
Reasoning: <one concise sentence explaining
the concrete guidance applied>,→
Rewrite: <the revised sentence on a single
line>,→
F.3.2 HumanEval + PEP 8 Executor
The executor receives the original HumanEval func-
tion signature and docstring as {prompt} , and a set

of style-guidance items as {reference_rules} .
The same prompt is used for both rule-mode
and chunk-mode retrieval; only the form of
{reference_rules}changes.
System message.
You are a Python code generation system.
Complete the given Python,→
function according to its signature and
docstring.,→
You may also receive coding-style rules or
reference guidance. Treat,→
the provided material as the only style
guidance available for this,→
task. Apply any rule whose condition is
relevant to the code you,→
write, but do not sacrifice functional
correctness for style.,→
Output only plain Python source code. Do not
use markdown fences. Do,→
not include explanations, tests, or extra
text.,→
User message.
Task:
Complete the following Python function.
Function to complete:
{prompt}
Coding-style rules or reference guidance:
{reference_rules}
Instructions:
1. Output the complete function, including
the original signature and,→
docstring.
2. The function name and signature must match
the prompt exactly.,→
3. Preserve the provided docstring as-is.
4. Functional correctness is the top priority.
5. Apply every provided coding-style rule
whose condition is relevant,→
to the code you produce.
6. Do not introduce unnecessary imports,
comments, dead code, unused,→
variables, or overly complex control flow.
7. Output only Python source code, with no
markdown fences or,→
explanations.
F.3.3 RuleArena-NBA Executor
System message.
You are an NBA salary-cap analyst. Determine
whether the proposed,→
transactions comply with the NBA Collective
Bargaining Agreement.,→
You will receive one transaction scenario and
a set of CBA rules or,→reference guidance selected by a retrieval
method. Use only the,→
provided reference material and the facts
stated in the scenario. Do,→
not invent missing contract terms, cap
figures, exceptions, or team,→
circumstances.
Return a JSON object only. Do not include
markdown fences,,→
explanations outside the JSON, or extra text.
User message.
Task:
Evaluate whether the proposed NBA transaction
scenario is legal under,→
the Collective Bargaining Agreement.
Scenario:
"""
{scenario}
"""
CBA rules or reference guidance:
{reference_rules}
Instructions:
1. Determine whether the transaction is legal
using the provided,→
reference material and the facts stated in
the scenario.,→
2. If the transaction is legal, set "answer"
to false and set both,→
"illegal_operation" and "problematic_team"
to null.,→
3. If the transaction is illegal, set
"answer" to true, identify the,→
specific illegal operation, and identify
the responsible team.,→
4. Do not rely on unstated facts, unstated
exceptions, or external,→
knowledge.
5. If the provided reference material is
insufficient to establish,→
illegality, prefer "answer": false.
Output exactly one JSON object:
{
"answer": true | false,
"illegal_operation": "<operation id or
short description, or null>",,→
"problematic_team": "<team name, or null>",
"rationale": "<one short sentence>"
}
F.4 Evaluation Prompts
All NPOV metrics are produced by an LLM judge;
HumanEval Pass@1 and PEP 8 pylint scores are
computed by their respective standard tools; and
RuleArena-NBA strict accuracy is computed pro-
grammatically from the executor’s JSON output.
The LLM-judged prompt is given below.

F.4.1 NPOV Violation Fix Rate and Auxiliary
Scores Judge
A single LLM judge produces both the binary Vio-
lation Fix Rate (VFR) label and the four 1–5 auxil-
iary scores (Remediation, Preservation, Tone, Flu-
ency) reported in Table 2. To prevent reward hack-
ing via pass-through rewrites, any candidate rewrite
whose character similarity to the original input ex-
ceeds 0.98 is automatically assigned VFR= 0
before this prompt is invoked.
System message.
You are a strict and fair Wikipedia Neutral
Point of View (NPOV),→
evaluation judge.
Your task is to evaluate whether a rewritten
sentence fixes the,→
specific NPOV violation described for the
original sentence, while,→
preserving the original factual content.
Judge only the rewrite,→
quality; do not reward or penalize the
retrieval method that produced,→
it.
Be strict about the target violation: a
rewrite receives VFR=true only,→
if the specific violation described in the
input is fully resolved.,→
Partial fixes, generic improvements,
unsupported additions, or,→
rewrites that create a new comparable
neutrality problem should,→
receive VFR=false.
Output only a single JSON object. Do not
include markdown fences,,→
comments, or any extra text.
User message.
Task:
Evaluate a single attempted rewrite of a
Wikipedia sentence.,→
Original sentence:
"""
{original}
"""
Specified NPOV violation:
"""
{violation}
"""
Rewrite to evaluate:
"""
{rewrite}
"""
Evaluation dimensions:
1. VFR (Violation Fix Rate), binary:- true: the rewrite fully resolves the
specified NPOV violation.,→
- false: the specified violation remains,
is only partially,→
addressed, or is replaced by a new
neutrality problem of,→
comparable severity.
2. Remediation, integer from 1 to 5:
- 1: the specified violation is not
addressed.,→
- 2: minimal or mostly ineffective attempt.
- 3: partial fix with a clear remaining
issue.,→
- 4: mostly addresses the violation with
minor weakness.,→
- 5: fully addresses the specified
violation.,→
3. Preservation, integer from 1 to 5:
- 1: key factual content is removed,
contradicted, or fabricated.,→
- 2: important facts are substantially
altered or omitted.,→
- 3: some factual content is changed or
lost, main point remains.,→
- 4: nearly all factual content preserved
with only minor,→
compression.
- 5: all factual content relevant to the
original is preserved.,→
Penalize rewrites that fix neutrality by
deleting essential facts.,→
4. Tone, integer from 1 to 5:
- 1: clearly biased, promotional,
judgmental, or inflammatory.,→
- 2: noticeably non-neutral.
- 3: mostly neutral but with some
evaluative wording.,→
- 4: neutral with minor stylistic weakness.
- 5: fully neutral and encyclopedic in
tone.,→
5. Fluency, integer from 1 to 5:
- 1: ungrammatical or difficult to
understand.,→
- 2: awkward or unclear.
- 3: readable but stilted.
- 4: fluent with minor issues.
- 5: polished, natural encyclopedic prose.
Additional judging rules:
- Do not give VFR=true merely because the
rewrite is shorter or more,→
generic.
- Do not give VFR=true if the rewrite removes
the problematic language,→
by dropping essential factual content.
- Do not reward unsupported attribution,
added sources, added,→
viewpoints, or factual claims not present
in the original sentence.,→
- A near-copy of the original should receive
VFR=false unless the,→
specified violation is actually removed.
- Scores should be integers only.
Output exactly one JSON object:
{

"VFR": true | false,
"Rem": <integer 1-5>,
"Pres": <integer 1-5>,
"Tone": <integer 1-5>,
"Flu": <integer 1-5>,
"reason": "<one concise sentence explaining
the main judgment>",→
}
F.5 Controlled Baseline: LLM Relevance
Reranker
To isolate the contribution of theapplicabilityfram-
ing specifically – as opposed to merely substituting
an LLM judge for a dense retriever – we addition-
ally evaluate a controlled baseline that uses the
same pairwise LLM-judgment setup as TAG, but
asks a standard relevance question. The baseline
differs from the applicability matcher of §F.2 in
exactly two respects: (i) the predicate isrelevance
rather thancondition satisfaction, and (ii) the rule’s
action field is exposed, matching the standard
LLM-reranker setup of Nogueira and Cho (2019)
and Glass et al. (2022). All other elements – the
rule set, the executor prompt, the model, and the
decoding settings – are held fixed across the two
conditions.
System message.
You are a strict relevance judge for
instruction-following tasks.,→
Given one task input and one candidate rule
extracted from an,→
instruction document, decide whether the rule
is RELEVANT to the,→
input. A rule is relevant if it is plausibly
useful for handling the,→
input, regardless of whether its specific
condition is satisfied.,→
Output ONLY a single JSON object. Do not
include reasoning,,→
explanations, markdown fences, or any other
text.,→
User message.
Task input:
"""
{query}
"""
Candidate rule:
- rule_id: {rule_id}
- rule_name: {rule_name}
- condition: {condition}
- action: {action}
- category_tags: {tags}
Question:Is this candidate rule relevant to the task
input?,→
Output format:
{"verdict": "YES"} or {"verdict": "NO"}
The contrast between this prompt and the appli-
cability matcher in §F.2 is the cleanest available
prompt-level test of whether the applicability fram-
ing – rather than LLM judgment alone – accounts
for TAG’s gains.
G Human Evaluation on Wikipedia
NPOV
To validate the LLM-as-a-judge scores used in Sec-
tion 4.3, we conduct a small-scale human evalua-
tion on a 40-query subset of the Wikipedia NPOV
evaluation set. Two annotators independently rate
anonymized outputs from three systems: All Rules
(M1), Standard RAG top-20 (M2), and TAG (M3,
ours). We use this study as a trend-level validation
rather than as a replacement for the full automatic
evaluation on all 107 cases.
G.1 Protocol
Each query is shown with the original sentence,
the editor-written violation description, and three
candidate rewrites. The rewrites are presented un-
der blinded labels A/B/C, and annotators are not
told which method produced which rewrite. Anno-
tators rate each rewrite on four dimensions: VFR
(binary), Remediation, Preservation, and Tone. The
three 1–5 scores follow the definitions used by the
LLM evaluator in Appendix F.4.1.
The subset consists of Q001–Q040 from the 107-
case NPOV evaluation set. Since this subset is not
a random sample, we report the LLM-judge scores
on the same 40 cases when comparing human and
automatic evaluation.
G.2 Annotator Recruitment and Consent
We recruited two university students as volunteer
annotators, who were compensated for their time.
Both annotators were given written instructions de-
scribing the Wikipedia NPOV rewriting task and
the rating criteria. For each item, they were shown
the original sentence, the violation description, and
the anonymized system rewrites, and were asked
to rate whether each rewrite fixed the violation and
to score remediation, preservation, and tone (the
rating dimensions detailed in Appendix G.1). An-
notators were informed that their ratings would be
used for research purposes in aggregate form. We

did not collect any personal or sensitive informa-
tion from the annotators beyond their ratings.
G.3 Human Scores
Table 6 reports the human evaluation results. TAG
obtains the highest human VFR and Remediation
score, and the overall ranking on the main metric
is M3>M1>M2.
Method VFR Rem. Pres. Tone
All Rules (M1) 85.0 4.28 4.85 4.37
Std RAG top-20 (M2) 65.0 3.41 4.83 3.51
TAG (M3, ours) 92.4 4.47 4.88 4.06
Table 6: Human evaluation on 40 Wikipedia NPOV
cases. VFR is reported as a percentage; Remediation,
Preservation, and Tone are 1–5 scores averaged over
two annotators.
G.4 Comparison with the LLM Judge
Table 7 compares human scores with the LLM-
judge scores on the same 40-query subset. Human
annotators assign higher absolute VFR scores than
the LLM judge, suggesting that the LLM judge is
more conservative on borderline rewrites. However,
the method ranking is preserved: TAG performs
best, followed by All Rules and Standard RAG
top-20.
Method LLM VFR Human VFR
All Rules (M1) 55.0 85.0
Std RAG top-20 (M2) 47.5 65.0
TAG (M3, ours) 70.0 92.4
Table 7: LLM-judge and human VFR on the same 40-
query subset. Human annotators are more permissive
in absolute scores, but preserve the same ranking as the
LLM judge.
G.5 Agreement and Significance
Inter-annotator agreement is substantial on the
main VFR metric (Cohen’s κ= 0.68 ). Agree-
ment is also substantial on Remediation and Tone
under quadratic-weighted κ, while Preservation has
lower chance-corrected agreement because most
rewrites receive high Preservation scores, leaving
little variance.
On the 40-query subset, TAG outperforms Stan-
dard RAG top-20 on human VFR by 27.5 percent-
age points. This difference is significant under
an exact McNemar test ( p= 0.003 ). TAG is notsignificantly different from All Rules, which is con-
sistent with the main-paper result that rule struc-
turing already captures a substantial portion of the
available gain, while applicability selection adds a
smaller improvement.
G.6 Subset Representativeness
The 40-query subset is not a random sample of
the full 107-case set. To avoid comparing differ-
ent sample sets, we compare human evaluation
only against the LLM-judge scores on the same 40
cases. Under the LLM judge, the subset scores are
55.0%, 47.5%, and 70.0% VFR for M1, M2, and
M3, respectively, preserving the same ranking as
the full-set results in Table 1. This suggests that the
human evaluation supports the same system-level
trend observed in the main experiment, while the
full-set automatic evaluation remains the primary
result.
G.7 Summary
The human evaluation provides a small-scale vali-
dation of the main NPOV trend. Although human
annotators are more permissive than the LLM judge
in absolute VFR, both evaluations rank the methods
in the same order: TAG >All Rules >Standard
RAG top-20. We therefore treat the human study
as supporting evidence that TAG’s improvement is
not solely an artifact of the LLM-based evaluator.
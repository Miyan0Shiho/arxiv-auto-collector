# Beyond Localization: Recoverable Headroom and Residual Frontier in Repository-Level RAG-APR

**Authors**: Pengtao Zhao, Boyang Yang, Bach Le, Feng Liu, Haoye Tian

**Published**: 2026-03-30 23:10:19

**PDF URL**: [https://arxiv.org/pdf/2603.29067v1](https://arxiv.org/pdf/2603.29067v1)

## Abstract
Repository-level automated program repair (APR) increasingly treats stronger localization as the main path to better repair. We ask a more targeted question: once localization is strengthened, which post-localization levers still provide recoverable gains, which are bounded within our protocol, and what residual frontier remains? We study this question on SWE-bench Lite with three representative repository-level RAG-APR paradigms, Agentless, KGCompass, and ExpeRepair. Our protocol combines Oracle Localization, within-pool Best-of-K, fixed-interface added context probes with per-condition same-token filler controls and same-repository hard negatives, and a common-wrapper oracle check. Oracle Localization improves all three systems, but Oracle success still stays below 50%. Extra candidate diversity still helps inside the sampled 10-patch pools, but that headroom saturates quickly. Under the two fixed interfaces, most informative added context conditions still outperform their own matched controls. The common-wrapper check shows different system responses: under a common wrapper, gains remain large for KGCompass and ExpeRepair, while Agentless changes more with builder choice. Prompt-level fusion still leaves a large residual frontier: the best fixed probe adds only 6 solved instances beyond the native three-system Solved@10 union. Overall, stronger localization, bounded search, evidence quality, and interface design all shape repository-level repair outcomes.

## Full Text


<!-- PDF content starts -->

Beyond Localization: Recoverable Headroom and Residual
Frontier in Repository-Level RAG-APR
Pengtao Zhao
School of Computing and Information
Systems, University of Melbourne
Australia
pengtaozhao@unimelb.edu.auBoyang Yang
School of Artificial Intelligence
(School of Software), Yanshan
University
China
yby@ieee.orgBach Le
School of Computing and Information
Systems, University of Melbourne
Australia
bach.le@unimelb.edu.au
Feng Liu
School of Computing and Information
Systems, University of Melbourne
Australia
feng.liu1@unimelb.edu.auHaoye Tian∗
Department of Computer Science,
Aalto University
Finland
haoye.tian@aalto.fi
Abstract
Repository-level automated program repair (APR) increasingly
treats stronger localization as the main path to better repair. We ask
a more targeted question: once localization is strengthened, which
post-localization levers still provide recoverable gains, which are
bounded within our protocol, and what residual frontier remains?
We study this question on SWE-bench Lite with three represen-
tative repository-level RAG-APR paradigms,Agentless,KGCom-
pass, andExpeRepair. Our protocol combines Oracle Localization,
within-pool Best-of- 𝐾, fixed-interface added context probes with
per-condition same-token filler controls and same-repository hard
negatives, and a common-wrapper oracle check. Oracle Localiza-
tion improves all three systems, but Oracle success still stays below
50%. Extra candidate diversity still helps inside the sampled 10-
patch pools, but that headroom saturates quickly. Under the two
fixed interfaces, most informative added context conditions still
outperform their own matched controls. The common-wrapper
check shows different system responses: under a common wrapper,
gains remain large forKGCompassandExpeRepair, whileAgent-
lesschanges more with builder choice. Prompt-level fusion still
leaves a large residual frontier: the best fixed probe adds only 6
solved instances beyond the native three-systemSolved@10 union.
Overall, stronger localization, bounded search, evidence quality,
and interface design all shape repository-level repair outcomes.
1 Introduction
Repository-level automated program repair (APR) on benchmarks
such as SWE-bench [ 13,41] requires large language models (LLMs)
to work over codebases that are much larger than one prompt.
Recent systems therefore rely on retrieval-augmented generation
(RAG) and vector indexes such as Faiss [ 8,19] to build bounded, task-
relevant repair contexts. We useretrieval-augmented generation
based automated program repair(RAG-APR) to refer to repository-
level APR systems that retrieve bounded task-relevant evidence
before patch generation. Across current repository-level repair
systems, this bounded-context RAG formulation has become a
mainstream practical route, including phase-structured pipelines,
∗Corresponding author.structure-aware retrieval, memory-augmented repair, and software-
agent style systems [31, 36, 46, 52, 54, 55, 62, 63].
This success has also encouraged a common optimization story:
localization first [ 52,55]. If retrieval can find the right files and
spans, repair should improve substantially, and recent localization-
oriented analyses support that intuition [ 15,21,39,44,51]. However,
current RAG-APR papers still mainly compare end-to-end solve
rates or new retrieval designs [ 63]. Those evaluations do not cleanly
separate gains from localization, context construction, patch gen-
eration, and validation [ 20,25,26]. As a result, they leave open
a more important question: after localization gets stronger, what
remains the main source of recoverable gain? This is not just an
evaluation-detail issue: without isolating the post-localization bot-
tlenecks, apparent gains can be misattributed to retrieval itself,
even when the limiting factor lies in how evidence is packaged,
consumed, or translated into valid patches [ 20,22,25]. That distinc-
tion matters for both system design and scientific interpretation,
because it determines whether further progress should come from
better localization, better interface construction, or stronger patch-
generation and validation strategies [26, 40].
There are already useful empirical studies and technical analyses
around repository-level repair. Benchmark audits examine leader-
board composition, model memory, solved-issue correctness, and
evaluation reliability, while context studies show that long context
is not automatically useful and that retrieved evidence can still be
used poorly [ 9,20,22,25,26,37,48,57]. Repository-level task stud-
ies also argue for finer-grained decomposition of issue resolution
[5,40]. These papers make the field better understood, but they still
stop short of one controlled post-localization study that measures
three things together: how much within-system search headroom
remains, whether added cross-paradigm evidence still helps under
controlled interfaces, and how much frontier remains jointly un-
solved after those gains are exhausted [ 20,39,40]. Without that
joint view, it is still hard to tell whether current repository-level
repair systems are primarily limited by finding the right evidence,
by using already-retrieved evidence effectively, or by a residual
frontier that is unlikely to disappear through prompt-level context
improvements alone [ 20,22,39]. This missing decomposition isarXiv:2603.29067v1  [cs.SE]  30 Mar 2026

, , Zhao et al.
important because these failure sources imply very different re-
search bets, yet they are often conflated in end-to-end comparisons
of repository-level APR systems [25, 40, 57].
In this paper, we study that post-localization gap on SWE-bench
Lite with three representative repository-level RAG-APR paradigms:
Agentless,KGCompass, andExpeRepair. We design a protocol
that isolates several post-localization factors, including Oracle Lo-
calization, within-pool Best-of- 𝐾sampling, controlled context aug-
mentation under fixed interfaces, and a unified oracle check [ 4,20,
39, 47]. This lets us ask a narrower but important question:
“Once localization gets stronger, what can still be re-
covered, which gains stay bounded, and what remains
jointly unsolved?”
Compared with prior work, our goal is not to propose another
repair system or another retrieval module. Instead, we connect
localization, search, evidence attribution, and residual-frontier anal-
ysis in one controlled empirical study. Our results show a consistent
pattern: stronger localization helps, but it does not remove most
of the gap; within-system search and informative added context
still recover cases under bounded settings; and prompt-level fusion
still leaves a broad jointly-unsolved frontier. The common-wrapper
check also shows that some oracle gains depend on how the final
repair interface is built, especially forAgentless.
The study is organized around four research questions that
separate oracle localization, within-pool search headroom, fixed-
interface added context, and the residual frontier; we state them
formally in Section 5.1.
This paper makes the following contributions:
•Controlled Post-Localization Study.We run a four-part con-
trolled study on SWE-bench Lite across three representative
repository-level RAG-APR paradigms.
•Bounded Post-Localization Headroom.We characterize how
much recoverable headroom remains after stronger localization,
including the bounded within-system search gains available in-
side sampled candidate pools.
•Control-Aware Evidence Attribution.We test fixed-interface
added context with per-condition same-token filler controls,
same-repository hard negatives, token-based budget audits, and
a common-wrapper oracle check.
•Residual Frontier Characterization.We show that prompt-
level gains recover only part of native complementarity and
still leave a broad jointly-unsolved frontier under the current
protocol.
2 Background
2.1 Repository-Level APR
Traditional APR often targeted isolated bugs within one function
or one file [ 10,17,18,24,27,29,30,33]. Repository-level APR is
harder because real issues can span multiple files, depend on project-
specific structure, and require the model to reason over a codebase
that is much larger than one prompt [ 13]. Even with longer context
windows, feeding the whole repository is still impractical and can
dilute useful signals [ 22]. RAG is therefore a practical way to build
a bounded repair context [ 19]. In this setting, the central designproblem is not only whether to retrieve evidence, but also what
evidence to retrieve and how to organize it before repair.
2.2 RAG-APR Decomposition
To bridge the gap between repository-scale code and bounded LLM
contexts, prior work has converged on a RAG-based formulation
[14,19,45]. In this paper, we useRAG-APRto refer to the following
three-stage paradigm: (1)Localization:narrow the search space to
candidate files, functions, or lines relevant to the issue. (2)Con-
text Construction:assemble compact evidence into a bounded
prompt. (3)Repair:generate candidate patches with the LLM
conditioned on the constructed context.
2.3 Representative RAG-APR Paradigms
Our study focuses onAgentless,KGCompass, andExpeRepair
because they share this bounded-context workflow but differ in
how they build repair evidence.Agentlessuses a staged workflow
with hierarchical localization, compressed repository views, and
test-based patch selection [ 52].KGCompassbuilds a repository-
aware knowledge graph, ranks function-level candidates with graph
proximity, and provides entity paths as structured evidence for
prompting [ 54].ExpeRepairuses dual memory to retrieve past
demonstrations and semantic insights, then injects them into itera-
tive test and patch generation before final validation [31].
These methods are therefore not orthogonal end-to-end: they
still share localization, bounded prompts, and downstream patch
generation logic. What differs most is the evidence carrier.Agent-
lessrelies on staged workflow control and compressed repository
structure,KGCompassrelies on graph-structured repository re-
lations, andExpeRepairrelies on experience-based repair mem-
ory. Recent repository-level repair systems span staged pipelines,
structure-aware retrieval, memory-augmented repair, and broader
agent workflows [ 31,36,43,46,52–55,62,63]. We therefore use
these three systems as a compact comparison set for evidence-
construction strategies in repository-level bounded-context repair.
3 Related Work
Recent repository-level APR work is largely built on the SWE-bench
family, which now includes multimodal, multilingual, long-horizon,
and live variants [ 7,13,56,58,61]. Most system papers still aim at
end-to-end gains through staged pipelines, agents, structure-aware
search, graph retrieval, or repair memory [ 31,52–55,63]. These
works define the main design space of repository-level RAG-APR,
but they do not directly study what remains hard after localization
is strengthened.
Recent empirical analyses ask what current benchmarks and
leaderboards actually measure. Prior audits study leaderboard ar-
chitectures, benchmark composition, evaluation rigor, solved-issue
correctness, memory effects, and benchmark mutation [ 9,25,26,
37,48,57]. RepoFixEval further decomposes repository-level repair
into issue discovering, localization, and fixing [ 40]. This line is clos-
est to our paper, but it still stops short of measuring how much
headroom remains after localization is strengthened.
A related line of work studies retrieval quality and context use
more directly [ 43]. RepoBugs and its RLCE study show that naive
repository context extraction can easily become redundant and

Beyond Localization: Recoverable Headroom and Residual Frontier in Repository-Level RAG-APR , ,
imprecise [ 5]. Prior retrieval-augmented repair systems explore
exemplars, static analysis, templates, and knowledge graphs as
evidence carriers [ 14,23,32,35,45]. RepoBench, ContextBench,
and SWE-ContextBench evaluate context retrieval and use more
directly, rather than relying only on final solve rate [ 20,60,64].
Lost in the Middle similarly shows that relevant evidence is not
reliably used just because it appears in a long prompt [ 22]. These
studies motivate our fixed-interface added context probes and our
same-token filler and hard-negative controls.
Localization remains a major focus because better fault evi-
dence can improve repair [ 1,15,21,50,51,59]. RGFL studies this
with an oracle-style upper-bound analysis [ 39]. Pass@k and self-
consistency motivate our analysis of within-system search head-
room [ 4,12,16,47]. Our question is narrower: after localization is
strengthened on SWE-bench Lite, how much headroom remains,
how fast does search saturate, and how much frontier stays jointly
unsolved? Thus, unlike prior system papers or single-factor analy-
ses, our paper contributes a controlled post-localization diagnosis
rather than another repair method.
4 Approach
4.1 Overview
This paper uses controlled interventions to study post-localization
headroom under a fixed benchmark split. We keep the benchmark
split and harness fixed, then vary localization, within-system search,
and added context while leaving the evaluation pipeline unchanged.
All primary comparisons use SWE-bench Lite.
Candidate Repair Blocks Repo Viewo1, .., oj
r1, .., rk
o1, .., oj
r1, .., rk
o1, .., oj
r1, .., rkIssue +
RepositoryAgentless
KGCompassOracle Locations (1..k)
❶ ❺ ❻❷❸
ExpeRepair❹
Candidate Repair Blocks Repo View
Candidate Repair Blocks Repo View
Memory ContextCode Evidence
Graph Evidence<..>
</>cont = ......
...<..>
</>cont = ......
...<..>
</>cont = ......
...
Patch
GenerationTest
Validation
Figure 1: High-level oracle intervention points in the three
native pipelines. Oracle injects gold-patch-derived pre-patch
file and line spans before each system builds its own final
repair prompt.
Figure 1 shows the shared study flow. The issue description is
the shared input, each system first builds its own candidate repair
blocks, Oracle can then inject gold-derived pre-patch spans, each
system still builds its own final repair prompt, the model generates
a patch, and the harness validates the result. The later interventions
change only selected steps in this flow. We read their effects through
outcome transitions, cross-system overlap, and residual-frontier
audits.4.2 Oracle Localization
Oracle Localization is the paper’s main counterfactual intervention.
At the oracle step in Figure 1, we parse the gold diff and extract
pre-patch fault spans 𝐺={(file,[start,end])} . Oracle injects repair
blocks derived from 𝐺into each system’s candidate set before that
system builds its own final repair prompt. Only pre-patch file and
line information is exposed; patch text and replacement code are
never revealed. This intervention asks how much end-to-end gap
stronger candidate localization removes when the downstream
prompt construction, patch generation, and validation logic remain
system-specific.
Ground-truth extraction.For each instance, we parse the gold
patch, extract its diff hunks, map them to pre-patch file paths and
line ranges, and obtain a set of fault spans 𝐺. The same extracted
spans are used for all systems, but they are merged into different
candidate sets and wrapped into different final prompts.
Interface-specific merge rules.The span extraction is shared,
but the merge rule is system-specific and fixed. This is a deliberate
design choice. Context construction is one of the main differences
among repository-level APR systems. If we instead standardized
all three pipelines with one shared post-oracle prompt builder, the
absolute scores might rise, but the comparison would collapse the
context-construction differences that this study is meant to pre-
serve. We therefore keep the oracle spans shared while preserving
each system’s own context-construction logic.Agentless.Or-
acle files are moved to the front of found files, and oracle line
blocks are prepended within found edit locations, so the native
repair interface still consumes the same data structure.KGCom-
pass.Oracle file-line spans are converted into method-like entries
and prepended to the existing final locations method list before the
native issue/method evidence block is rebuilt.ExpeRepair.The
parsed oracle spans inject the bug locations structure with a single
oracle group, while the downstream memory-stage interface and
final patch instruction remain unchanged.
The intervention is therefore shared at the span level but inter-
face specific at the merge stage. It combines shared oracle spans
with each system’s own prompt-construction logic. We later pair
the native-Oracle analysis with a common-wrapper oracle check
that replays Oracle under one common repair wrapper, and one
variant also uses one common oracle prompt builder.
4.3 Best-of-K and Ideal Selection
To estimate how much additional headroom remains after stronger
localization, we evaluate an idealized selector over 𝐾= 10inde-
pendently sampled patch candidates per instance under Oracle Lo-
calization, following multi-sample evaluation and self-consistency
ideas [ 4,47].Solved@K is 1 if any of the 𝐾samples resolves the
instance and 0 otherwise. This gives a within-pool upper bound
for within-system candidate diversity plus ideal selection, while
keeping each system’s repair and validation pipeline unchanged.
For intermediate 𝐾<10, we greedily reorder the sampled 10-patch
pools by marginal solved gain and derive within-poolSolved@K
from that order.

, , Zhao et al.
4.4 Fixed-Interface Added Context Probes and
Controls
Motivated by the complementarity exposed by the first two inter-
ventions, we run a follow-up added context attribution study on two
fixed repair interfaces, in the same spirit as process-level context
studies that separate retrieved context from downstream use [ 20].
Both blocks reuse native oracle prompt artifacts upstream, then in-
tegrate extra context immediately before the final patch-generation
stage.
Agentlessinterface.We reuse the nativeAgentlessOracle
repair prompt as a fixed interface. The pipeline first follows the
originalAgentlessoracle run and keeps that prompt skeleton un-
changed. We then insert extra context before the final output-format
instructions and continue with the same downstreamAgentless
repair call. On SWE-bench Lite with 𝐾= 10samples and the official
harness, we compare three informative added context conditions:
(i) aKG-augmented context, which adds onlyKGCompassoracle-
side issue and method evidence reconstructed from its oracle final
locations; (ii) anExpe-augmented context, which adds only the pre-
assistant conversational context extracted fromExpeRepair; and
(iii)UnionContext, which combines both evidence channels. For
each informative condition, we then build two own controls: (iv) an
exact filler control, which keeps the same wrapper skeleton and sec-
tion structure but replaces the added evidence with non-informative
filler matched to that condition’s total prompt input tokens; and (v)
anirrelevant hard negative, which keeps the same wrapper skeleton
and raw prompt style but swaps in same-repository evidence from
a different issue, again matched to that condition’s total prompt
input tokens. We also report the nativeAgentlessOracle Best-of- 𝐾
result for the same interface.
ExpeRepairfinal-step interface.We repeat the same idea on the
nativeExpeRepairfinal-step interface. This block also starts from
nativeExpeRepairoracle generation conversations and their final-
step prompt artifacts. We keep the native message-thread structure
and final patch-format instruction fixed, integrate only extra pre-
generation context immediately before the final patch-generation
stage, and continue with the same final-step patch call. This keeps
the native oracle artifacts fixed at the same final-step integration
point used in theAgentlessblock. We compare three informative
conditions: (i) anAgentless-augmented context, which inserts the raw
Agentlessoracle repair prompt; (ii) aKG-augmented context, which
inserts the raw or reconstructedKGCompassoracle prompt; and (iii)
UnionContext, which inserts both blocks. For each informative
condition, we then build the same two own controls as above:
(iv) anexact filler control, which keeps the same message-thread
structure but replaces the added evidence with non-informative
filler matched to that condition’s total prompt input tokens; and
(v) anirrelevant hard negative, which keeps the same message-
thread structure and raw prompt style but inserts same-repository
evidence from a different issue, again matched to that condition’s
total prompt input tokens. We also report the nativeExpeRepair
Oracle Best-of-𝐾result for the same interface.4.5 Diagnostic Measures
We track three diagnostic views after the interventions. First, we
use an outcome taxonomy with three states:resolved,completed-
unresolved, andno patch submitted. This display-level collapse maps
empty patch and not submitted status tono patch submitted. Under
the SWE-bench Lite harness, execution-failure cases are still sub-
mitted but unresolved outcomes, so we keep them insidecompleted-
unresolved. Second, we measure overlap across the three native
systems to quantify how much complementarity remains jointly un-
recovered after each stage. Third, for the residual jointly-unsolved
set, we run a heuristic qualitative failure audit onAgentlessoracle-
run logs, the most detailed logs available.
5 Evaluation
We organize the evaluation around four RQs: what Oracle Local-
ization removes (RQ1), how much extra headroom within-system
candidate diversity and ideal selection can recover (RQ2), whether
added context gains come from informative context rather than
prompt length (RQ3), and how much frontier remains after those
prompt-level gains (RQ4).
5.1 Research Questions
We organize the study around four research-question groups that
separate oracle localization, within-pool search headroom, fixed-
interface evidence attribution, and the residual frontier:
RQ1 (Oracle Gain and Failure Concentration):
•RQ1.1 (Oracle Effect):How much success does Oracle Local-
ization recover over the native Baseline?
•RQ1.2 (Wrapper Dependence):How much of that gain remains
under a shared wrapper and a shared oracle builder?
•RQ1.3 (Failure Concentration):After localization is strength-
ened, where does the remaining failure mass concentrate?
RQ2 (Selection Headroom and Saturation):
•RQ2.1 (Search Headroom):How much within-pool search
headroom remains after stronger localization?
•RQ2.2 (Selector Sensitivity):How sensitive is the remaining
headroom to the selector used to rank candidates?
RQ3 (Fixed-Interface Evidence Attribution):
•RQ3.1 (Context vs Controls):Do informative added context
conditions outperform their own same-token fillers and hard
negatives?
•RQ3.2 (Early Gains and Stability):Do stronger added context
conditions help earlier in the 𝐾prefix and increase stable solves?
•RQ3.3 (Fusion Limits and Wrong Evidence):Does simple
prompt fusion saturate the single-source gains, and what happens
when the added evidence is wrong?
RQ4 (Residual Cross-System Frontier):
•RQ4.1 (Fixed-Probe Recovery):How much new frontier can
the best fixed informative probes recover beyond the native
union?
•RQ4.2 (Recovery Ceiling):How much more frontier is recov-
erable under looser post hoc probe unions and oracle ceilings?
•RQ4.3 (Frontier Breadth and Failure Modes):How broad is
the residual frontier, and what failure patterns remain there?

Beyond Localization: Recoverable Headroom and Residual Frontier in Repository-Level RAG-APR , ,
5.2 Experimental Setup
5.2.1Benchmark Split and Harness.
Benchmark.We use the SWE-bench Lite [ 13,41] test split because
it provides a curated 300-instance subset that preserves the broad
distribution and difficulty spectrum of the original benchmark while
making controlled, full-factorial evaluation across multiple systems
and intervention settings computationally practical. Each instance
consists of a repository snapshot, an issue description, and a held-
out test suite for validation.
Task split.We evaluate all systems and intervention settings on the
full set of 300 instances without subsampling, so that comparisons
are made on exactly matched tasks rather than being affected by
sampling variance. Instances that fail to produce a patch or complete
execution are counted as failures.
Evaluation harness.We adopt the official SWE-bench Lite har-
ness to align with the benchmark-standard evaluation protocol and
ensure reproducible, directly comparable results across systems. An
instance is counted asresolvedonly if the submitted patch passes
all tests in the harness.
5.2.2Systems and Backbone.We compare three repository-
level APR systems that represent different RAG-APR design choices:
Agentless[ 52], a phase-structured localization–repair–validation
pipeline;KGCompass[ 54], a structure-aware system grounded in
repository knowledge graphs; andExpeRepair[ 31], a memory-
augmented repair system. Full commands, prompts, and run con-
figurations are provided in the accompanying anonymous artifact.
In the artifact, dense retrieval uses jinaai/jina-embeddings-v2-base-
code [11], and lexical retrieval uses BM25 [38] where applicable.
Our primary cross-system comparisons useDeepSeek-V3[ 6]
across all three systems. To check whether the Oracle effect is tied
to one backbone, we also runAgentlessandKGCompasswith
GPT-4.1 [34] under the same Baseline and Oracle settings.
Baseline.Each system first runs in its native form. Given the issue
description and repository snapshot, it performs its own localization
and context construction, then generates and validates a patch
with its default control flow. All interventions below are measured
against this baseline.
5.2.3Metrics and Statistics.
Primary metrics.Our main end-to-end metric isSuccess, the pro-
portion of resolved instances on the 300-instance SWE-bench Lite
split. For the multi-sample analysis in RQ2, we reportSolved@K
computed from one fixed sampled pool. For the direct localiza-
tion audit in RQ1, we report File hit, File Recall@1, and File Re-
call@5. For the fixed-interface added context study in RQ3, we
reportSolved@10 together with mean prompt input tokens as
the main prompt-size measure. We record character counts in the
artifact.
Secondary metrics.We record completion rate as a complemen-
tary view of how much failure mass stays outside harness-completed
runs. Here completion rate means harness-completed runs divided
by SWE-bench Lite, so empty patch cases stay outside its numerator.
For RQ4, we summarize recovery over the jointly-unsolved frontier
as recovered/frontier cases, with percentages reported when useful.Statistics.For proportions, we report Wilson 95% confidence in-
tervals [ 3,49]. For paired binary outcomes on the same bench-
mark split, we use the exact McNemar test [ 28] and report paired
risk differences with percentile bootstrap 95% confidence inter-
vals over instance resampling. For the small-N repository-level
frontier recovery rates in RQ4, we report exact binomial 95% confi-
dence intervals. For the greedy within-pool upper-bound curves
in RQ2, we draw percentile bootstrap 95% bands over the shared
instance pool. For RQ3, we apply Holm correction separately to
the six informative-versus-own-exact-filler comparisons and to the
six informative-versus-own-hard-negative comparisons. We also
report native-interface paired comparisons.
5.3 RQ1: Oracle Gain and Failure Concentration
5.3.1Experimental Design.RQ1 asks how much end-to-end
gap Oracle Localization removes once each system keeps its own
prompt-construction, patch-generation, and validation flow, and
where the remaining failures concentrate. We compare each sys-
tem’s Baseline run against Oracle Localization on SWE-bench Lite
under the native pipelines described in Section 4.2. Because native
Oracle keeps each system’s own merge rule and prompt builder,
RQ1 also includes a common-wrapper check on the shared 300-
instance pool. In that check, ‘Shared native’ and ‘Shared oracle’ hold
the repair wrapper fixed, and ‘Shared-builder oracle’ also standard-
izes the oracle builder. We report Success, completion rate, paired
wins and losses, and a direct localization audit with File hit, File
Recall@1, and File Recall@5.
5.3.2Experimental Results for RQ1.1 (Oracle Effect).
Oracle Localization improves all three systems, but success still
stays below 50%.Table 1 shows that Oracle Localization raises re-
solved instances from 84 to 121 forAgentless, from 88 to 129
forKGCompass, and from 98 to 117 forExpeRepair. Success here
means test-passing under the SWE-bench Lite harness. Completion
rate is completed/total, and confidence intervals are Wilson 95%.
Completion rate also increases for all three systems, reaching 99.0%,
98.7%, and 98.0% under Oracle. However, success under Oracle still
remains at only 40.3%, 43.0%, and 39.0%. Oracle raises completion
for all three systems, but 61.6%–79.3% of the total success gain is
associated with higher pass rates among completed runs rather
than completion alone. The remaining gap is therefore large even
after the ground-truth fault location is supplied, and most of it
remains after completion.
Table 1: Baseline and Oracle Localization on SWE-bench Lite.
Success is resolved/300, completion rate is completed/300,
and success among completed is resolved/completed.
System Setting Success (95% Wilson CI) Completion rate Success among completed
AgentlessBaseline 28.0% (23.2–33.3) 84.7% 33.1%
Oracle40.3%(34.9–46.0) 99.0% 40.7%
KGCompassBaseline 29.3% (24.5–34.7) 90.0% 32.6%
Oracle43.0%(37.5–48.7) 98.7% 43.6%
ExpeRepairBaseline 32.7% (27.6–38.2) 92.0% 35.5%
Oracle39.0%(33.7–44.6) 98.0% 39.8%
The paired gains are real, but regressions still occur.Table 2 reports
paired wins and losses on the same instances. In this paired com-
parison, wins denote instances where Oracle succeeds and Baseline

, , Zhao et al.
fails, whereas losses denote the reverse. All three systems show
more wins than losses, with strong evidence forAgentlessand
KGCompass. The paired risk-difference intervals also stay above
zero for all three systems. At the same time, losses are non-zero for
all three systems. Oracle Localization is therefore a strong inter-
vention, but it is not monotonic.
Table 2: Paired success changes from Baseline to Oracle Local-
ization on SWE-bench Lite. Wins mean Oracle-only successes,
and losses mean Baseline-only successes.
SystemPaired success
change (points, 95% CI) Wins / losses McNemar𝑝
Agentless+12.3 (7.7–17.0)46 / 94.34×10−7
KGCompass+13.7 (8.3–18.7)54 / 134.47×10−7
ExpeRepair+6.3 (1.0–11.7)43 / 242.71×10−2
GPT-4.1 comparison.Under GPT-4.1, Oracle again improves suc-
cess from 74/300 to 109/300 forAgentlessand from 55/300 to
126/300 forKGCompass, with paired wins/losses of 40/5 and 78/7,
which matches the main-backbone pattern of clear gains and more
wins than losses.
5.3.3Experimental Results for RQ1.2 (Wrapper Dependence).
A common-wrapper check shows different system responses under
a shared wrapper.Table 3 compares three interface-controlled vari-
ants on the shared 300-instance pool. The asymmetry is sharpest
onAgentless: under the shared wrapper, success changes only
from 35.7% to 37.0%, and the large jump appears only under shared-
builder oracle (51.0%). By contrast,KGCompassandExpeRepair
already retain most of their oracle gain under the shared wrap-
per, reaching 50.3% and 51.3% before builder standardization. This
comparison makes the contribution of wrapper and builder choices
visible alongside the shared oracle spans.
Table 3: Common-wrapper oracle check on the shared 300-
instance pool. Shared native keeps one shared repair wrapper,
shared oracle adds oracle spans under that wrapper, and
shared-builder oracle also shares the oracle prompt builder.
SystemShared
nativeShared
oracleShared-builder
oracleΔoracle
vs shared nativeΔbuilder
vs shared nativeΔoracle
vs shared builder
Agentless35.7% 37.0% 51.0% +1.3 +15.3-14.0
KGCompass17.0% 50.3% 51.0% +33.3 +34.0 -0.7
ExpeRepair40.3% 51.3% 51.0% +11.0 +10.7 +0.3
5.3.4Experimental Results for RQ1.3 (Failure Concentra-
tion).
Baseline localization already helps, but it is still far from enough.
Table 4 summarizes Baseline file-level hit rates, span-hit success,
and whether Oracle regressions already had a Baseline span hit.
Baseline localization is already substantial: file hit reaches 65.3%–
85.3%, file Recall@1 reaches 35.0%–74.3%, and file Recall@5 reaches
51.7%–84.3%. Yet even on span-hit instances, Baseline success is still
only 34.8%–45.5%. Most Oracle regressions also already had at least
one Baseline span hit: all losses forAgentlessandExpeRepair,
and 9/13 losses forKGCompass. Localization therefore helps, but it
does not by itself explain the remaining failures.Table 4: RQ1 localization audit from the Baseline runs. Gold
file and gold span mean overlap with the gold patch location.
SystemGold file
in setGold file
top-1Gold file
top-5Success with
gold spanSuccess with
no gold spanOracle losses
with top-1
gold spanOracle losses
with any
gold span
Agentless82.3% 71.7% 82.3%38.1%1.2% 9/9 9/9
KGCompass65.3% 35.0% 51.7%45.5%17.1% 4/13 9/13
ExpeRepair85.3% 74.3% 84.3%34.8%0.0% 24/24 24/24
The instance-level transition paths are still highly similar across
all three systems.Figure 2 tracks the same instances from Baseline
to Oracle Localization and then to Best-of- 𝐾. Gray denotesno patch
submitted, orange denotesunresolved, and green denotesresolved;
execution-failure cases stay insideunresolved. Across all three sys-
tems, Oracle mainly shrinks theno patchbucket, from 45/28/24
cases to 2/2/5, while execution failures remain rare at 1 →1, 2→2,
and 0→1. Table 5 makes the stage changes explicit: the first two
columns report counts before and after Oracle, and the last three
report transitions between failure buckets. ForAgentlessandKG-
Compass,no patchcorresponds to ‘empty_patch’; forExpeRepair,
it corresponds to ‘not_submitted’. Table 5 also shows that most
recovered no-patch cases first becomeunresolved, and that Oracle
upgrades many already-completed runs toresolved. Best-of- 𝐾then
acts mainly on the remaining Oracle-stageunresolvedpool, con-
verting 33/41/31 such cases toresolved. This stage view again points
to the same RQ1 conclusion: after localization improves, the main
remaining mass is submitted but unresolved repair attempts.
Table 5: RQ1 failure-subtype audit behind Figure 2. The table
counts how cases move between the three states in that figure.
SystemNo patch
(Baseline→Oracle)Execution error
(Baseline→Oracle)No patch
→UnresolvedNo patch
→ResolvedUnresolved
→Resolved
Agentless45→2 1→1 28 17 29
KGCompass28→2 2→2 16 11 43
ExpeRepair24→5 0→1 15 7 36
[RQ-1] Findings:Oracle Localization closes part of the gap, but
success still stays below 50%. Most recovered gain comes from
better pass rates among completed runs rather than completion
alone, and the common-wrapper check shows thatAgentless
is much more builder-mediated thanKGCompassandExpeRe-
pair.Insights:Stronger localization mainly reduces no-patch
failures; the main bottleneck then shifts to post-submission
repair.
5.4 RQ2: Selection Headroom and Saturation
5.4.1Experimental Design.RQ2 asks whether stronger localiza-
tion leaves substantial within-system post-localization headroom,
or only a bounded amount inside the sampled pools. Under Oracle
Localization, we sample 𝐾= 10candidate patches per instance and
evaluate two views on the same fixed pool. The mainSolved@K
upper-bound view greedily reorders the pool by marginal solved
gain before taking the top- 𝐾prefix. We then replay fixed prefix
orders on that same pool, including the raw sample-index order
and simple rerankers, to ask how much of that upper bound is
practically reachable. That raw sample-index order is the system’s
native generation order rather than a shared temperature schedule.

Beyond Localization: Recoverable Headroom and Residual Frontier in Repository-Level RAG-APR , ,
28%
57%
15%40%
59%51%
49%Resolved
Completed
unresolved
No patch
submitted
Baseline Oracle Best-of-K
(a)Agentless
29%
61%
9%43%
56%56%
43%Resolved
Completed
unresolved
No patch
submitted
Baseline Oracle Best-of-K (b)KGCompass
33%
59%
8%39%
59%49%
50%Resolved
Completed
unresolved
No patch
submitted
Baseline Oracle Best-of-K (c)ExpeRepair
Figure 2: Instance-level three-state outcome transition paths from Baseline to Oracle Localization and then to Best-of- 𝐾(𝐾= 10).
Each ribbon follows the same benchmark instances across stages, with green for resolved, orange for completed but unresolved,
and gray for no patch submitted.
5.4.2Experimental Results for RQ2.1 (Search Headroom).
Best-of-𝐾adds meaningful headroom over Oracle, but most of it
is already recovered by 𝐾=5.Table 6 compares Oracle Localization
success rate withSolved@5 andSolved@10. The Oracle Local-
ization rate is taken from Table 1.Solved@5 andSolved@10 are
computed from the same 10-sample pools after greedily reordering
samples by marginal solved gain. Best-of- 𝐾adds 9.7–13.3 points
over the Oracle single-run success rates, so candidate diversity and
ideal selection still matter. But the stronger saturation result is that
this gain is already almost exhausted by 𝐾=5. The gain from 𝐾=5
to𝐾=10is only 1.3–1.7 points, and the greedy 𝐾=5prefix already
captures 86.2%–87.5% of the total Oracle-to-Solved@10 headroom.
Even after greedy reordering,Solved@10 still remains below 60%
for all three native systems. The remaining search headroom is
therefore real but strongly front-loaded.
Table 6: Selection headroom under stronger localization. Ora-
cle is the single run oracle result, andSolved@5/Solved@10
are within-pool upper bounds from the same 10-sample
patch pools.
System OracleSolved@5Solved@10 Extra gain (5→10)
Agentless40.3% 49.7% 51.0%+1.3
KGCompass43.0% 54.7% 56.3%+1.7
ExpeRepair39.0% 47.3% 48.7%+1.3
The greedy upper-bound curves rise early and then flatten.Figure 3
keeps the greedy within-pool upper bound as the reference line
and adds fixed-pool selector replay on the same oracle patch pools.
Across the three panels, the red band marks the bootstrap 95%
interval for that greedy upper bound on the shared instance pool,
and the green band marks the span across the fixed random seeds.
The greedy upper-bound curves still rise early and flatten by around
𝐾=5, so the bounded-headroom conclusion remains unchanged.
Read these curves as an early-prefix diagnosis: most recoverable
gain is already present in the first few positions of the sampled
pool, so later samples add little extra headroom.
5.4.3Experimental Results for RQ2.2 (Selector Sensitivity).
Selector replay shows that the pool often has a solution, but the
early prefix still misses it.Figure 3 keeps the two most informative
non-required references: the raw sample order and cluster-diversity
reranking. Table 7 reports the omitted fixed-order and learned
rerankers at 𝐾=5. Those omitted rows do not change the mainpicture: reverse order helpsAgentlessandKGCompassbut hurts
ExpeRepair, LLM reranking is strongest only onKGCompass, and
no omitted selector beats the raw sample order onExpeRepair.
Native sample order therefore already carries system-specific search
signal. At the same time, it still trails the greedy 𝐾=5upper bound by
3.3 points onAgentless, 3.3 onKGCompass, and 1.3 onExpeRepair,
so system-native ordering is itself a clear improvement target. Even
if we pick the strongest available selector for each system at 𝐾=5,
a 1.3–2.0 point gap to the greedy upper bound still remains.
Table 7: Additional fixed-order and learned selector replay at
𝐾=5on the same oracle patch pools. These rows are omitted
from Figure 3 to keep the main plot focused on raw-order
signal and family coverage.
Selector (𝐾=5)Agentless KGCompass ExpeRepair
Reverse order 48.0% 52.7% 41.0%
LLM reranker 46.3% 52.7% 44.0%
Cross-encoder reranker 44.7% 48.3% 43.3%
Taken together, the selector curves point to a two-part bottleneck.By
𝐾=5, the greedy upper bound has already captured 86.2%–87.5% of
the total within-pool headroom, but the default practical reranker
still leaves a visible early-prefix gap, especially onKGCompass.
Patch-family clustering suggests why. The fixed 10-patch pools
are not very diverse to begin with: they contain only 1.68–1.83
approximate patch families on average, and only 10.3%–14.2% of
solvable instances contain more than one solved family. Yet the
default𝐾=5prefixes still cover only 1.35–1.43 unique families on
average, leaving about 3.5 duplicate slots inside the prefix. Cluster-
diversity reranking pushes family coverage close to full coverage,
but exact success still stays 3.4–4.4 points below the family-hit rate.
The bottleneck is therefore not just more samples or a universally
better scorer. Small prefixes must first reach the right patch family,
and then rank the right variant early inside that family.

, , Zhao et al.
1 2 3 4 5 6 7 8 9 10
K253035404550556065Solved@K (%)
 Random mean
Original order
Cluster-diversity reranker
Greedy upper bound
(a)Agentless
1 2 3 4 5 6 7 8 9 10
K253035404550556065Solved@K (%)
 (b)KGCompass
1 2 3 4 5 6 7 8 9 10
K253035404550556065Solved@K (%)
 (c)ExpeRepair
Figure 3: Greedy within-pool upper bounds and fixed-pool selector replay under Oracle Localization for the three native
systems. The panels keep the four main references: greedy upper bound, random mean, original order, and cluster-diversity
reranking. The red curve and band show the greedy upper bound with a bootstrap 95% interval, and the green band shows the
range across fixed random seeds. Table 7 reports the remaining selector rows at𝐾=5.
[RQ-2] Findings:Extra sampling still improves Oracle, but
the gain is bounded and strongly front-loaded: by 𝐾=5, greedy
selection already captures 86.2%–87.5% of the total available
within-pool headroom. The remaining headroom is selector-
sensitive, but no single selector closes it consistently, and the
default practical reranker still leaves a visible early-prefix gap.
Insights:The main bottleneck is not more samples alone. It is
reaching the right patch family early and then ranking the right
variant within that family.
5.5 RQ3: Fixed-Interface Evidence Attribution
5.5.1Experimental Design.Since RQ2 shows that the search
within the system still helps but saturates early, RQ3 asks whether
additional gains can instead be recovered by adding cross-paradigm
context blocks after each interface has already built its native oracle
prompt artifacts. On theAgentlessside, this is the native oracle
repair interface. On theExpeRepairside, this is the native oracle
conversation carried to the final-step patch-generation interface. In
both blocks, the added evidence is integrated only before the final
patch-generation stage. Each informative condition is evaluated
against its own same-token filler control and same-repository hard
negative, with prompt input tokens as the main budget unit and
Holm-corrected paired tests as described in Section 5.2.3.
5.5.2Experimental Results for RQ3.1 (Context vs Controls).
Informative evidence beats its controls, but more context is not
always better.Table 8 reports each informative condition alongside
its own same-token filler control and same-repository hard negative,
with prompt input tokens as the main budget unit. After Holm
correction, five of the six informative-versus-filler comparisons
and all six informative-versus-hard-negative comparisons remain
significant; in the native-interface paired comparisons, only the two
lighter single-source transfers are not significant. Budget matching
is tight: the mean prompt-token gap between each informative
row and its own filler or hard negative stays below one token
on average, although the hard negatives still require truncation on
37.7%–63.0% of instances. Tokens are not monotonic. OnAgentless,
KGCompass-augmented reaches 60.7% with 13.46k input tokens,
whereas the longerUnionContextrow reaches 60.0% with 15.29k.OnExpeRepair,UnionContextis still the top point estimate, but
it improves overKGCompass-augmented by only 1.3 points (59.0%
vs. 57.7%) while adding 1.69k more input tokens. Added evidence
therefore helps because it adds useful structure, not because more
context is always better.
Table 8: Fixed-interface informative probes with their own
per-condition controls under Oracle Localization with Best-
of-𝐾(𝐾= 10). Native, added context, and control columns
reportSolved@10; tokens are mean input tokens in thou-
sands, and the last column is the share of hard negatives that
were truncated to fit the budget.
Interface Added contextNative
promptAdded
contextToken-matched
fillerSame-repo
hard negativeInput
tokens (k)Hard negative
truncated
Expe-augmented
(ExpeRepair only)51.0% 54.7% 51.3% 47.3% 3.47 42.0%
AgentlessKGCompass-augmented
(KGCompassonly)51.0%60.7%51.0% 49.3% 13.46 37.7%
UnionContext
(KGCompass+ ExpeRepair)51.0%60.0%50.0% 21.0% 15.29 55.3%
Agentless-augmented
(Agentless only)48.7% 52.3% 44.7% 44.3% 3.50 50.7%
ExpeRepairKGCompass-augmented
(KGCompassonly)48.7%57.7%46.0% 43.3% 4.93 43.0%
UnionContext
(Agentless +KGCompass)48.7%59.0%45.0% 41.7% 6.62 63.0%
5.5.3Experimental Results for RQ3.2 (Early Gains and Sta-
bility).
The strongest added context rows help early, but their stability
effects differ by wrapper.Within the same sampled 10-patch pools,
these gains do not appear only at 𝐾=10. OnAgentless,KGCompass
reaches 48.3% at 𝐾=1and 54.0% at 𝐾=2, already above native 𝐾=3
and𝐾=5; by𝐾=2,UnionContexthas already realized about 89%
of its final gain at 𝐾=10. OnExpeRepair,UnionContextreaches
45.7% at𝐾=1and 51.7% at 𝐾=2, and by𝐾=2it has already realized
essentially all of its final gain at 𝐾=10. The strongest rows therefore
look less like late lucky hits and more like earlier routing into
stronger patch families. The stability pattern, however, is more
mixed. OnAgentless,KGCompass-augmented reduces all-miss
cases by 29 and raises the number of 10/10 stable solves from 63 to
86, whereasUnionContextreduces all-miss by 27 but reaches only
72 stable solves. OnExpeRepair,UnionContextreduces all-miss
slightly more thanKGCompass-augmented (31 vs. 27), but it still
ends with fewer 10/10 stable solves (81 vs. 86). The extra source

Beyond Localization: Recoverable Headroom and Residual Frontier in Repository-Level RAG-APR , ,
therefore changes the gain shape differently across wrappers: it
mostly dilutes stability onAgentless, but onExpeRepairit mainly
buys earlier coverage with only a small final-ceiling gain. The
lighter transfers behave differently:AgentlessExpe-only mainly
trims full misses but adds many low-frequency solves, whereas
ExpeRepair Agentless-only mostly turns already-solvable cases
into 10/10 solves without widening coverage much.
5.5.4Experimental Results for RQ3.3 (Fusion Limits and
Wrong Evidence).
The singleUnionContextprompt does not dominate the best
single-source row.On theAgentlessinterface, running the Expe-
only andKGCompass-only probes separately solves 196 unique
instances, whereas the singleUnionContextprompt solves 180.
There are 20 cases that the separate single-source probes recover
but the singleUnionContextprompt does not, and only 4 cases
in the reverse direction. The same pattern appears on theExpeRe-
pairinterface: the separate single-source probes solve 189 unique
instances, whereas the singleUnionContextprompt solves 177,
with a 20-case separate-only advantage versus 8 in the reverse
direction. This fusion limit also appears inside the stronger rows
themselves. OnAgentless, the longerUnionContextprompt
does not improve over the best single-source row, and it converts
26 cases that are 10/10 underKGCompass-augmented into only
1–9/10 solves. OnExpeRepair,UnionContextstill improves early
coverage overKGCompass-augmented, but it converts 21 of that
row’s 10/10 cases into 1–9/10 solves while raising finalSolved@10
by only 1.3 points. Simple prompt concatenation therefore does not
simply add source-specific gains together; it can also dilute strong
single-source trajectories.
Wrong evidence is not a neutral control.ForAgentless Union-
Context, the same-token hard negative drops from 180 to 63 re-
solved instances and raises pool-level empty-patch incidence from
41 to 257; forExpeRepair UnionContext, it drops from 177 to 125
and raises the same incidence from 20 to 133. This drop is too large
to read as prompt length alone. The hard negatives keep the same
wrapper skeleton and repository style, but replace task-relevant
evidence with the wrong structured constraints. Added context
therefore helps because it supplies the right structured evidence,
not because the prompt is merely longer.
[RQ-3] Findings:Most informative added context conditions
outperform their matched controls, and the largest gains come
fromKGCompass-based andUnionContextprompts. These
gains appear early in the 𝐾prefix, but the value of an extra
source is not monotonic: onAgentless, the best single-source
row already matches or exceeds the longer union prompt, while
onExpeRepairthe union prompt mainly improves earlier cov-
erage and only slightly raises the final ceiling.Insights:Added
context helps when it adds task-relevant structure, but simple
prompt fusion does not reliably add the gains from multiple
sources and can dilute stable single-source gains.
5.6 RQ4: Residual Cross-System Frontier
5.6.1Experimental Design.RQ3 shows that informative added
context still helps, RQ4 then asks whether these prompt-level gains
are enough to explain most of the remaining complementarity, orwhether a broader residual frontier still remains. We answer it with
system-overlap progression, a compact frontier-recovery ladder
from the native union to progressively looser post hoc ceilings, and
a qualitative log audit. This design uses the overlap and frontier
diagnostics introduced in Section 4 and reports frontier recovery
rates with exact binomial confidence intervals as described in Sec-
tion 5.2.3.
5.6.2Experimental Results for RQ4.1 (Fixed-Probe Recov-
ery).
Prompt-level gains are mostly consolidation gains rather than
frontier-opening gains.Figure 4 shows solved-instance overlap across
the three native systems under Baseline, Oracle Localization, and
Best-of-𝐾. Region labels in the figure report the exact overlap
counts, and across the three panels the jointly-unsolved set shrinks
from 161 to 130 to 101. Under Best-of- 𝐾, the native three-system
union reaches 199/300 solved instances, but 101 instances remain
jointly unsolved, so a large frontier still remains. Figure 5 then
shows only small frontier recovery beyond that native union: +6
for the best single probe, +10 for the best two-probe union, and +14
for the post hoc ceiling. Table 9 then shows why: most probe-local
wins stay inside the native union rather than opening new frontier
cases. Across all six probes, only 5.0%–15.8% of wrapper-local wins
open new frontier cases, whereas 84.2%–95.0% stay inside the native
union. Even the stronger rows follow the same pattern.KGCompass
context onExpeRepairandUnionContextonExpeRepairpost
38 and 44 wins over their own native wrapper, but only 6 of those
wins in each case expand the native union.UnionContexton
Agentlessis more concentrated still, with 34 of its 36 wins (94.4%)
staying inside the native union. Even the best fixed single probe
therefore recovers only 6 of the 101 native frontier cases (5.9%, exact
95% CI 2.2–12.5) and still leaves 95 jointly unsolved.
Table 9: Decomposing each informative probe’s wins over its
own native wrapper under Best-of- 𝐾(𝐾= 10). The last two
columns split those wins into cases already solved by another
native system and cases that newly expand the three-system
union.
Target wrapper Added contextWins vs
own nativeAlready solved
by native unionNew frontier
cases
AgentlessKGCompass37 32 5
ExpeRepair 21 18 3
UnionContext36 34 2
ExpeRepairKGCompass38 32 6
Agentless 20 19 1
UnionContext44 38 6
5.6.3Experimental Results for RQ4.2 (Recovery Ceiling).
Even looser post hoc ceilings remain small.The full ladder in
Figure 5 stays shallow: the best three-probe union reaches 212/300
(+13), the best four-probe union reaches 213/300 (+14), and the all-6
oracle stays at the same 213/300. Even under that favorable ceiling,
recovery rises only to 14 of the 101 frontier cases (13.9%, exact 95%
CI 7.8–22.2), leaving 87 residual cases. The last two probes therefore
add no extra frontier recovery beyond the best four-probe union.
This low ceiling is also not just a fixed-probe artifact: even allowing
repo-adaptive choice among single informative probes would still
recover only 10 of the 101 frontier cases, which is the same order
as the best two-probe union.

, , Zhao et al.
6
2114
2124
1340Agentless KGCompass
ExpeRepair
(a)Baseline. Jointly unsolved = 161; solved
by at least one system = 139/300.
9
2123
1814
1075Agentless KGCompass
ExpeRepair(b)Oracle Localization. Jointly unsolved =
130; solved by at least one system =
170/300.
102518
1010
11115AgentlessKGCompass
ExpeRepair(c)Best-of-𝐾(𝐾=10). Jointly unsolved =
101; solved by at least one system =
199/300.
Figure 4: Solved-instance overlap acrossAgentless,KGCompass, andExpeRepairunder Baseline, Oracle Localization, and
Best-of-𝐾. Each Venn region label is an exact instance count.
0 50 100 150 200 250 300
Solved instances on the 300-case poolNative
union
UnionContext
on ExpeRepair
+ KGCompass
on Agentless
+ KGCompass
on ExpeRepair
+ UnionContext
on Agentless
All 6
probes199/300
205/300 (+6)
209/300 (+10)
212/300 (+13)
213/300 (+14)
213/300 (+14)
Figure 5: Compact RQ4 frontier ladder on the shared 300-
instance pool. Blue is the native three-system union, green is
extra frontier recovered beyond that union, and gray is the
remaining frontier.
5.6.4Experimental Results for RQ4.3 (Frontier Breadth and
Failure Modes).
The residual frontier is concentrated, broad, and still hard at the
repository level.The native Best-of- 𝐾jointly-unsolved frontier spans
11 of the 12 repositories in SWE-bench Lite, but it is not uniformly
spread. sympy anddjango alone account for 65 of the 101 frontier
cases. Even there, recovery stays small: the best fixed single probe
recovers only 4 of those 65 cases, and even a repo-wise oracle over
single probes recovers only 5. At the same time, five frontier-bearing
repositories remain zero-gain even under that favorable repo-wise
oracle, covering 13 frontier cases in total. The residual frontier is
therefore both concentrated and broad: a few large repositories
contribute most of the raw frontier mass, but unrecovered cases
still span almost the whole benchmark.
Qualitative log-audit patterns in the residual frontier.Figure 6
summarizes a qualitative audit of the original 101 jointly-unsolved
native cases. The labels come from heuristic annotation ofAgent-
lessoracle-run evaluation logs. Within that scope, the most com-
mon labels are wrong API or runtime exceptions, incomplete fixes,
and missing imports or NameError s. These patterns are not only
common but also hard to recover under the current probe family:
wrong API, incomplete fix, and assertion mismatch together ac-
count for 60 of the 101 frontier cases, yet the all-6 oracle recovers
only 4 of them. The residual frontier is therefore not just broadacross repositories; it also remains stubborn within the dominant
post-localization failure modes.
0% 5% 10% 15% 20% 25% 30%
Share of the 101 frontier casesWrong API /
runtime exception
Incomplete fix
Missing import /
NameError
T est assertion
mismatch
Other / unclear
Flaky / external
dependency25 (24.8%)
21 (20.8%)
20 (19.8%)
14 (13.9%)
12 (11.9%)
9 (8.9%)
Figure 6: Qualitative log-audit labels for the 101 jointly-
unsolved native cases under Best-of- 𝐾. Bar-end labels show
count and share of the 101-case frontier.
[RQ-4] Findings:Prompt-level fusion recovers only a small
part of the remaining complementarity. Across all six infor-
mative probes, most wrapper-local wins stay inside the native
union rather than opening new frontier cases, and even the
favorable post hoc ceiling recovers only 14 of the 101 native
frontier cases. The residual frontier is both concentrated and
broad: a few large repositories contribute most of the raw fron-
tier mass, but unrecovered cases still span almost the whole
benchmark.Insights:Most added context gains reinforce ex-
isting strengths rather than unlock new frontier, and the re-
maining frontier persists both across repositories and within
the dominant post-localization failure modes.
6 Threats to Validity
Internal.The native Oracle intervention keeps each system’s
own wrapper and prompt builder, so the measured gain is not a
pure localization-only effect. We address this threat with common-
wrapper and common-builder checks that separate span-level shar-
ing from interface-level choices. We therefore interpret the Oracle
results as controlled diagnostics under the current protocol, rather
than as universal localization-only estimates.

Beyond Localization: Recoverable Headroom and Residual Frontier in Repository-Level RAG-APR , ,
Construct.Our main outcome is harness-defined test-passing suc-
cess on SWE-bench Lite. This is the benchmark’s standard outcome,
but it does not guarantee full semantic correctness. We therefore
complement it with completion rate, paired wins and losses, local-
ization audits, selector replay, and residual-frontier analysis, and
interpret our findings as evidence about benchmark-defined repair
behavior.
External.We evaluate on the full SWE-bench Lite split, a widely
used benchmark that covers diverse repositories and supports di-
rect horizontal comparison with prior work [ 13,41,42]. We also
compare three representative RAG-APR paradigms under the same
evaluation setup. However, we do not claim that the same effect
sizes will hold for all APR benchmarks, repair interfaces, or model
backbones.
7 Conclusion
In this paper, we study repository-level RAG-APR as a controlled
empirical analysis after stronger localization rather than as a new-
system competition. Across the four RQs, stronger localization
helps but does not remove most of the remaining gap. Within-
system search and informative added context still recover extra
cases under bounded settings, and informative content outperforms
matched controls more reliably than a prompt-length-only expla-
nation would predict. The common-wrapper check also shows
different system responses under a shared wrapper, especially for
Agentless. Prompt-level fusion still recovers only part of native
complementarity, and a large residual frontier remains under the
current protocol. These results argue for a broader post-localization
agenda that includes search, evidence quality, and interface design.
8 Data Availability Statement
All code and data for this study are available in an anonymized, DOI-
minted repository [ 2]. The repository contains the materials needed
to reproduce the reported results. To preserve double-anonymous
review, repository metadata are anonymized and will be restored
after acceptance.
References
[1]Rui Abreu, Peter Zoeteweij, Rob Golsteijn, and Arjan J. C. van Gemund. 2009.
A practical evaluation of spectrum-based fault localization.J. Syst. Softw.82, 11
(Nov. 2009), 1780–1792. doi:10.1016/j.jss.2009.06.035
[2]Anonymous Authors. 2026.Anonymous Artifact for ASE 2026 (Code and Data).
doi:10.5281/zenodo.19112800
[3]Lawrence D. Brown, T. Tony Cai, and Anirban DasGupta. 2001. Interval Es-
timation for a Binomial Proportion.Statist. Sci.16, 2 (2001), 101–117. http:
//www.jstor.org/stable/2676784
[4]Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de
Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg
Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf,
Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail
Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter,
Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fo-
tios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex
Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shan-
tanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike,
Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight,
Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario
Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluat-
ing Large Language Models Trained on Code. (2021). arXiv:2107.03374 [cs.LG]
https://arxiv.org/abs/2107.03374
[5]Yuxiao Chen, Jingzheng Wu, Xiang Ling, Changjiang Li, Zhiqing Rui, Tianyue
Luo, and Yanjun Wu. 2024. When Large Language Models Confront Repository-
Level Automatic Program Repair: How Well They Done? . In2024 IEEE/ACM46th International Conference on Software Engineering: Companion Proceedings
(ICSE-Companion). IEEE Computer Society, Los Alamitos, CA, USA, 459–471.
doi:10.1145/3639478.3647633
[6]DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu,
Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan,
Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun
Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang,
Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian
Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi
Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie
Qiu, Junlong Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin
Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong
Wang, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang,
Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang,
Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi
Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu Zhang, Ruyi
Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu,
Shengfeng Ye, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping
Yu, Shunfeng Zhou, Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W. L.
Xiao, Wangding Zeng, Wanjia Zhao, Wei An, Wen Liu, Wenfeng Liang, Wenjun
Gao, Wenqin Yu, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi,
Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaokang Zhang,
Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Cheng, Xin
Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xinnan Song, Xinxia Shan, Xinyi Zhou,
Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, Y. K. Li, Y. Q. Wang, Y. X.
Wei, Y. X. Zhu, Yang Zhang, Yanhong Xu, Yanhong Xu, Yanping Huang, Yao
Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Yu, Yi Zheng, Yichao
Zhang, Yifan Shi, Yiliang Xiong, Ying He, Ying Tang, Yishi Piao, Yisong Wang,
Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yu Wu, Yuan Ou, Yuchen
Zhu, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yukun Zha, Yunfan Xiong,
Yunxian Ma, Yuting Yan, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou,
Z. F. Wu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhen Huang, Zhen
Zhang, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma,
Zhigang Yan, Zhihong Shao, Zhipeng Xu, Zhiyu Wu, Zhongyu Zhang, Zhuoshu
Li, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, and
Zizheng Pan. 2025. DeepSeek-V3 Technical Report. arXiv:2412.19437 [cs.CL]
https://arxiv.org/abs/2412.19437
[7]Xiang Deng, Jeff Da, Edwin Pan, Yannis Yiming He, Charles Ide, Kanak Garg,
Niklas Lauffer, Andrew Park, Nitin Pasari, Chetan Rane, Karmini Sampath, Maya
Krishnan, Srivatsa Kundurthy, Sean Hendryx, Zifan Wang, Vijay Bharadwaj, Jeff
Holm, Raja Aluri, Chen Bo Calvin Zhang, Noah Jacobson, Bing Liu, and Brad
Kenstler. 2025. SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software
Engineering Tasks? arXiv:2509.16941 [cs.SE] https://arxiv.org/abs/2509.16941
[8]Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy,
Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2025.
The faiss library.IEEE Transactions on Big Data(2025).
[9]Spandan Garg, Benjamin Steenhoek, and Yufan Huang. 2026. Saving SWE-
Bench: A Benchmark Mutation Approach for Realistic Agent Evaluation.
arXiv:2510.08996 [cs.SE] https://arxiv.org/abs/2510.08996
[10] Luca Gazzola, Daniela Micucci, and Leonardo Mariani. 2019. Automatic Software
Repair: A Survey.IEEE Trans. Softw. Eng.45, 1 (Jan. 2019), 34–67. doi:10.1109/
TSE.2017.2755013
[11] Michael Günther, Jackmin Ong, Isabelle Mohr, Alaeddine Abdessalem, Tanguy
Abel, Mohammad Kalim Akram, Susana Guzman, Georgios Mastrapas, Saba
Sturua, Bo Wang, Maximilian Werk, Nan Wang, and Han Xiao. 2024. Jina Em-
beddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents.
arXiv:2310.19923 [cs.CL] https://arxiv.org/abs/2310.19923
[12] Carol Hanna, Aymeric Blot, and Justyna Petke. 2024. Reinforcement Learn-
ing for Mutation Operator Selection in Automated Program Repair. (2024).
arXiv:2306.05792 [cs.SE] https://arxiv.org/abs/2306.05792
[13] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press,
and Karthik Narasimhan. 2024. SWE-bench: Can Language Models Resolve Real-
World GitHub Issues? arXiv:2310.06770 [cs.CL] https://arxiv.org/abs/2310.06770
[14] Matthew Jin, Syed Shahriar, Michele Tufano, Xin Shi, Shuai Lu, Neel Sundaresan,
and Alexey Svyatkovskiy. 2023. InferFix: End-to-End Program Repair with LLMs.
InProceedings of the 31st ACM Joint European Software Engineering Conference
and Symposium on the Foundations of Software Engineering(San Francisco, CA,
USA)(ESEC/FSE 2023). Association for Computing Machinery, New York, NY,
USA, 1646–1656. doi:10.1145/3611643.3613892
[15] Sungmin Kang, Gabin An, and Shin Yoo. 2024. A Quantitative and Qualitative
Evaluation of LLM-Based Explainable Fault Localization.Proceedings of the ACM
on Software Engineering1, FSE (July 2024), 1424–1446. doi:10.1145/3660771
[16] Sungmin Kang and Shin Yoo. 2022. Language models can prioritize patches for
practical program patching. InProceedings of the Third International Workshop on
Automated Program Repair(Pittsburgh, Pennsylvania)(APR ’22). Association for
Computing Machinery, New York, NY, USA, 8–15. doi:10.1145/3524459.3527343
[17] Dongsun Kim, Jaechang Nam, Jaewoo Song, and Sunghun Kim. 2013. Automatic
patch generation learned from human-written patches. InProceedings of the 2013

, , Zhao et al.
International Conference on Software Engineering(San Francisco, CA, USA)(ICSE
’13). IEEE Press, 802–811.
[18] Claire Le Goues, ThanhVu Nguyen, Stephanie Forrest, and Westley Weimer. 2012.
GenProg: A Generic Method for Automatic Software Repair.IEEE Trans. Softw.
Eng.38, 1 (Jan. 2012), 54–72. doi:10.1109/TSE.2011.104
[19] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. InProceedings of the 34th International Conference
on Neural Information Processing Systems(Vancouver, BC, Canada)(NIPS ’20).
Curran Associates Inc., Red Hook, NY, USA, Article 793, 16 pages.
[20] Han Li, Letian Zhu, Bohan Zhang, Rili Feng, Jiaming Wang, Yue Pan, Earl T. Barr,
Federica Sarro, Zhaoyang Chu, and He Ye. 2026. ContextBench: A Benchmark
for Context Retrieval in Coding Agents. arXiv:2602.05892 [cs.LG] https://arxiv.
org/abs/2602.05892
[21] Kui Liu, Anil Koyuncu, Tegawendé F. Bissyandé, Dongsun Kim, Jacques Klein,
and Yves Le Traon. 2019. You Cannot Fix What You Cannot Find! An Investigation
of Fault Localization Bias in Benchmarking Automated Program Repair Systems.
In2019 12th IEEE Conference on Software Testing, Validation and Verification (ICST).
102–113. doi:10.1109/ICST.2019.00020
[22] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
Fabio Petroni, and Percy Liang. 2024. Lost in the Middle: How Language Models
Use Long Contexts.Transactions of the Association for Computational Linguistics
12 (2024), 157–173. doi:10.1162/tacl_a_00638
[23] Pei Liu, Bo Lin, Yihao Qin, Cheng Weng, and Liqian Chen. 2024. T-RAP: A
Template-guided Retrieval-Augmented Vulnerability Patch Generation Approach.
InProceedings of the 15th Asia-Pacific Symposium on Internetware(Macau, China)
(Internetware ’24). Association for Computing Machinery, New York, NY, USA,
105–114. doi:10.1145/3671016.3672506
[24] Matias Martinez, Thomas Durieux, Romain Sommerard, Jifeng Xuan, and Martin
Monperrus. 2017. Automatic repair of real bugs in java: a large-scale experiment
on the defects4j dataset.Empirical Softw. Engg.22, 4 (Aug. 2017), 1936–1964.
doi:10.1007/s10664-016-9470-4
[25] Matias Martinez and Xavier Franch. 2026. Dissecting the SWE-Bench Leader-
boards: Profiling Submitters and Architectures of LLM- and Agent-Based Repair
Systems. arXiv:2506.17208 [cs.SE] https://arxiv.org/abs/2506.17208
[26] Matias Martinez and Xavier Franch. 2026. What’s in a Benchmark? The Case
of SWE-Bench in Automated Program Repair. arXiv:2602.04449 [cs.SE] https:
//arxiv.org/abs/2602.04449
[27] Matias Martinez and Martin Monperrus. 2014. ASTOR: Evolutionary Automatic
Software Repair for Java. arXiv:1410.6651 [cs.SE] https://arxiv.org/abs/1410.6651
[28] Quinn McNemar. 1947. Note on the sampling error of the difference between
correlated proportions or percentages.Psychometrika12, 2 (1947), 153–157.
[29] Sergey Mechtaev, Jooyong Yi, and Abhik Roychoudhury. 2016. Angelix: scalable
multiline program patch synthesis via symbolic analysis. InProceedings of the
38th International Conference on Software Engineering(Austin, Texas)(ICSE ’16).
Association for Computing Machinery, New York, NY, USA, 691–701. doi:10.
1145/2884781.2884807
[30] Martin Monperrus. 2018. Automatic Software Repair: A Bibliography.ACM
Comput. Surv.51, 1, Article 17 (Jan. 2018), 24 pages. doi:10.1145/3105906
[31] Fangwen Mu, Junjie Wang, Lin Shi, Song Wang, Shoubin Li, and Qing Wang. 2025.
EXPEREPAIR: Dual-Memory Enhanced LLM-based Repository-Level Program
Repair. arXiv:2506.10484 [cs.SE] https://arxiv.org/abs/2506.10484
[32] Noor Nashid, Mifta Sintaha, and Ali Mesbah. 2023. Retrieval-Based Prompt Selec-
tion for Code-Related Few-Shot Learning. In2023 IEEE/ACM 45th International
Conference on Software Engineering (ICSE). 2450–2462. doi:10.1109/ICSE48619.
2023.00205
[33] Hoang Duong Thien Nguyen, Dawei Qi, Abhik Roychoudhury, and Satish Chan-
dra. 2013. SemFix: Program repair via semantic analysis. In2013 35th Interna-
tional Conference on Software Engineering (ICSE). 772–781. doi:10.1109/ICSE.2013.
6606623
[34] OpenAI. 2025. Introducing GPT-4.1 in the API. OpenAI product announcement.
https://openai.com/index/gpt-4-1/ Accessed 2026-03-12.
[35] Shuyin Ouyang, Jie M. Zhang, Zeyu Sun, and Albert Merono Penuela. 2025.
Knowledge-Enhanced Program Repair for Data Science Code . In2025 IEEE/ACM
47th International Conference on Software Engineering (ICSE). IEEE Computer
Society, Los Alamitos, CA, USA, 898–910. doi:10.1109/ICSE55347.2025.00246
[36] Zhongqiang Pan, Chuanyi Li, Wenkang Zhong, Yi Feng, Bin Luo, and Vincent
Ng. 2026. RepoRepair: Leveraging Code Documentation for Repository-Level
Automated Program Repair. arXiv:2603.01048 [cs.SE] https://arxiv.org/abs/2603.
01048
[37] Thanosan Prathifkumar, Noble Saji Mathews, and Meiyappan Nagappan.
2025. Does SWE-Bench-Verified Test Agent Ability or Model Memory?
arXiv:2512.10218 [cs.SE] https://arxiv.org/abs/2512.10218
[38] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Frame-
work: BM25 and Beyond.Found. Trends Inf. Retr.3, 4 (April 2009), 333–389.
doi:10.1561/1500000019[39] Melika Sepidband, Hamed Taherkhani, Hung Viet Pham, and Hadi Hemmati.
2026. RGFL: Reasoning Guided Fault Localization for Automated Program Repair
Using Large Language Models. arXiv:2601.18044 [cs.SE] https://arxiv.org/abs/
2601.18044
[40] Tao Sun, Yang Yang, Xianfu Cheng, Jian Yang, Yintong Huo, Zhuoren Ye, Rubing
Yang, Xiangyuan Guan, Wei Zhang, Hangyuan Ji, Changyu Ren, Mengdi Zhang,
Xunliang Cai, and Zhoujun Li. 2024. RepoFixEval: A Repository-Level Program
Repair Benchmark From Issue Discovering to Bug Fixing. https://openreview.
net/forum?id=LaNCeNmoHR
[41] SWE-bench Team. 2024. SWE-bench Lite. https://www.swebench.com/lite.html
Accessed 2026-02-18.
[42] SWE-bench Team. 2025. SWE-bench Official Leaderboards. Online leaderboard.
https://www.swebench.com/ Accessed 2026-02-18.
[43] Yicheng Tao, Yao Qin, and Yepang Liu. 2026. Retrieval-Augmented
Code Generation: A Survey with Focus on Repository-Level Approaches.
arXiv:2510.04905 [cs.SE] https://arxiv.org/abs/2510.04905
[44] Bo Wang, Ming Deng, Mingda Chen, Youfang Lin, Jianyi Zhou, and Jie M. Zhang.
2025. Assessing the effectiveness of recent closed-source large language models in
fault localization and automated program repair.Automated Software Engineering
33, 1 (2025), 26. doi:10.1007/s10515-025-00549-x
[45] Weishi Wang, Yue Wang, Shafiq Joty, and Steven C.H. Hoi. 2023. RAP-Gen:
Retrieval-Augmented Patch Generation with CodeT5 for Automatic Program
Repair. InProceedings of the 31st ACM Joint European Software Engineering Con-
ference and Symposium on the Foundations of Software Engineering(San Francisco,
CA, USA)(ESEC/FSE 2023). Association for Computing Machinery, New York,
NY, USA, 146–158. doi:10.1145/3611643.3616256
[46] Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen
Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang
Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff,
Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao Peng, Heng Ji, and
Graham Neubig. 2025. OpenHands: An Open Platform for AI Software Developers
as Generalist Agents. arXiv:2407.16741 [cs.SE] https://arxiv.org/abs/2407.16741
[47] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang,
Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain
of thought reasoning in language models.arXiv preprint arXiv:2203.11171(2022).
[48] You Wang, Michael Pradel, and Zhongxin Liu. 2025. Are "Solved Issues" in SWE-
bench Really Solved? arXiv:2503.15223 [cs.SE] https://arxiv.org/abs/2503.15223
[49] Edwin B. Wilson. 1927. Probable Inference, the Law of Succession, and Statistical
Inference.J. Amer. Statist. Assoc.22, 158 (1927), 209–212. http://www.jstor.org/
stable/2276774
[50] Franz Wotawa. 2002. On the relationship between model-based debugging and
program slicing.Artif. Intell.135, 1–2 (Feb. 2002), 125–143. doi:10.1016/S0004-
3702(01)00161-8
[51] Yonghao Wu, Zheng Li, Jie M. Zhang, Mike Papadakis, Mark Harman, and Yong
Liu. 2023. Large Language Models in Fault Localisation. arXiv:2308.15276 [cs.SE]
https://arxiv.org/abs/2308.15276
[52] Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. 2025. De-
mystifying LLM-Based Software Engineering Agents.Proc. ACM Softw. Eng.2,
FSE, Article FSE037 (June 2025), 24 pages. doi:10.1145/3715754
[53] Boyang Yang, Zijian Cai, Fengling Liu, Bach Le, Lingming Zhang, Tegawendé F.
Bissyandé, Yang Liu, and Haoye Tian. 2025. A Survey of LLM-based Au-
tomated Program Repair: Taxonomies, Design Paradigms, and Applications.
arXiv:2506.23749 [cs.SE] https://arxiv.org/abs/2506.23749
[54] Boyang Yang, Haoye Tian, Jiadong Ren, Shunfu Jin, Yang Liu, Feng Liu, and Bach
Le. 2025. Enhancing Repository-Level Software Repair via Repository-Aware
Knowledge Graphs. arXiv:2503.21710 [cs.SE] https://arxiv.org/abs/2503.21710
[55] John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao,
Karthik Narasimhan, and Ofir Press. 2024. SWE-agent: Agent-Computer In-
terfaces Enable Automated Software Engineering. InAdvances in Neural In-
formation Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan,
U. Paquet, J. Tomczak, and C. Zhang (Eds.), Vol. 37. Curran Associates,
Inc., 50528–50652. https://proceedings.neurips.cc/paper_files/paper/2024/file/
5a7c947568c1b1328ccc5230172e1e7c-Paper-Conference.pdf
[56] John Yang, Carlos E Jimenez, Alex Zhang, Kilian Lieret, Joyce Yang, Xindi Wu,
Ori Press, Niklas Muennighoff, Gabriel Synnaeve, Karthik Narasimhan, Diyi
Yang, Sida Wang, and Ofir Press. 2025. SWE-bench Multimodal: Do AI Sys-
tems Generalize to Visual Software Domains?. InInternational Conference on
Learning Representations, Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (Eds.),
Vol. 2025. 2794–2829. https://proceedings.iclr.cc/paper_files/paper/2025/file/
07d6332ae36730707fddddba736d7b6c-Paper-Conference.pdf
[57] Boxi Yu, Yuxuan Zhu, Pinjia He, and Daniel Kang. 2025. UTBoost: Rigorous
Evaluation of Coding Agents on SWE-Bench. arXiv:2506.09289 [cs.SE] https:
//arxiv.org/abs/2506.09289
[58] Daoguang Zan, Zhirong Huang, Wei Liu, Hanwu Chen, Shulin Xin, Linhao
Zhang, Qi Liu, Aoyan Li, Lu Chen, Xiaojian Zhong, Siyao Liu, Yongsheng Xiao,
Liangqiang Chen, Yuyu Zhang, Jing Su, Tianyu Liu, RUI LONG, Ming Ding,
and liang xiang. 2025. Multi-SWE-bench: A Multilingual Benchmark for Issue
Resolving. InThe Thirty-ninth Annual Conference on Neural Information Processing

Beyond Localization: Recoverable Headroom and Residual Frontier in Repository-Level RAG-APR , ,
Systems Datasets and Benchmarks Track. https://openreview.net/forum?id=
MhBZzkz4h9
[59] Andreas Zeller and Ralf Hildebrandt. 2002. Simplifying and Isolating Failure-
Inducing Input.IEEE Trans. Softw. Eng.28, 2 (Feb. 2002), 183–200. doi:10.1109/32.
988498
[60] Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi
Mao, Jian-Guang Lou, and Weizhu Chen. 2023. RepoCoder: Repository-Level
Code Completion Through Iterative Retrieval and Generation. InProceedings
of the 2023 Conference on Empirical Methods in Natural Language Processing,
Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational
Linguistics, Singapore, 2471–2484. doi:10.18653/v1/2023.emnlp-main.151
[61] Linghao Zhang, Shilin He, Chaoyun Zhang, Yu Kang, Bowen Li, Chengxing
Xie, Junhao Wang, Maoquan Wang, Yufan Huang, Shengyu Fu, Elsie Nalli-
pogu, Qingwei Lin, Yingnong Dang, Saravan Rajmohan, and Dongmei Zhang.2025. SWE-bench Goes Live!. InThe Thirty-ninth Annual Conference on Neu-
ral Information Processing Systems Datasets and Benchmarks Track. https:
//openreview.net/forum?id=OGWkr7gXka
[62] Quanjun Zhang, Chengyu Gao, Yu Han, Ye Shang, Chunrong Fang, Zhenyu
Chen, and Liang Xiao. 2026. SGAgent: Suggestion-Guided LLM-Based Multi-
Agent Framework for Repository-Level Software Repair. arXiv:2602.23647 [cs.SE]
https://arxiv.org/abs/2602.23647
[63] Yuntong Zhang, Haifeng Ruan, Zhiyu Fan, and Abhik Roychoudhury. 2024. Au-
toCodeRover: Autonomous Program Improvement. InProceedings of the 33rd
ACM SIGSOFT International Symposium on Software Testing and Analysis(Vienna,
Austria)(ISSTA 2024). Association for Computing Machinery, New York, NY, USA,
1592–1604. doi:10.1145/3650212.3680384
[64] Jared Zhu, Minhao Hu, and Junde Wu. 2026. SWE Context Bench: A Benchmark
for Context Learning in Coding. arXiv:2602.08316 [cs.SE] https://arxiv.org/abs/
2602.08316
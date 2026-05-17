# Does RAG Know When Retrieval Is Wrong? Diagnosing Context Compliance under Knowledge Conflict

**Authors**: Yihang Chen, Pin Qian, Su Wang, Sipeng Zhang, Huan Xu, Shuhuai Lin, Xinpeng Wei

**Published**: 2026-05-14 07:14:19

**PDF URL**: [https://arxiv.org/pdf/2605.14473v1](https://arxiv.org/pdf/2605.14473v1)

## Abstract
The Context-Compliance Regime in Retrieval-Augmented Generation (RAG) occurs when retrieved context dominates
  the final answer even when it conflicts with the model's parametric knowledge. Accuracy alone does not reveal
  how retrieved context causally shapes answers under such conflict. We introduce Context-Driven Decomposition
  (CDD), a belief-decomposition probe that operates at inference time and serves as an intervention mechanism for
  controlled retrieval conflict. Across Epi-Scale stress tests, TruthfulQA misconception injection, and cross-
  model reruns, CDD exposes three patterns. P1: context compliance is measurable in an upper-bound adversarial
  setting, where Standard RAG reaches 15.0% accuracy on TruthfulQA misconception injection (N=500). P2:
  adversarial accuracy gains transfer across model families: CDD improves accuracy on Gemini-2.5-Flash and on
  Claude Haiku/Sonnet/Opus, but rationale-answer causal coupling does not transfer. CDD reaches 64.1% mistake-
  injection causal sensitivity on Gemini-2.5-Flash, while sensitivities for all three Claude variants fall in the
  [-3%, +7%] range, suggesting that the Claude-side accuracy gains operate through a mechanism distinct from the
  explicit conflict-resolution trace. P3: explicit conflict decomposition improves robustness under temporal
  drift and noisy distractors, with CDD reaching 71.3% on temporal shifts and 69.9% on distractor evidence on the
  full Epi-Scale adversarial benchmark. These three patterns identify context-compliance as a structural axis
  along which standard RAG can be probed and intervened on, distinct from retrieval-quality or single-method
  robustness questions, and motivate releasing Epi-Scale for systematic study across model families and retrieval
  pipelines.

## Full Text


<!-- PDF content starts -->

Does RAG Know When Retrieval Is Wrong?
Diagnosing Context Compliance under Knowledge Conflict
Yihang Chen1,*Pin Qian2,*Su Wang2,*Sipeng Zhang3
Huan Xu1Shuhuai Lin2Xinpeng Wei1
1Georgia Institute of Technology
2Carnegie Mellon University
3University of California San Diego
ychen3726@gatech.edu, pqian@alumni.cmu.edu
suwang@alumni.cmu.edu, siz018@ucsd.edu, huan.xu71@gmail.com
shuhuail@andrew.cmu.edu, william.xp.wei@outlook.com
*Equal contribution.
Abstract
TheContext-Compliance Regimein Retrieval-
Augmented Generation (RAG) occurs when
retrieved context dominates the final answer even
when it conflicts with the model’s parametric
knowledge. Accuracy alone does not reveal how
retrieved context causally shapes answers under
such conflict. We introduceContext-Driven
Decomposition (CDD), a belief-decomposition
probe that operates at inference time and serves as
an intervention mechanism for controlled retrieval
conflict. Across Epi-Scale stress tests, Truth-
fulQA misconception injection, and cross-model
reruns, CDD exposes three patterns.P1:context
compliance is measurable in an upper-bound
adversarial setting, where Standard RAG reaches
15.0% accuracy on TruthfulQA misconception
injection (N=500).P2:adversarial accuracy
gains transfer across model families—CDD
improves accuracy on Gemini-2.5-Flash and
on Claude Haiku/Sonnet/Opus—but rationale-
answer causal coupling does not transfer. CDD
reaches 64.1% mistake-injection causal sensitiv-
ity on Gemini-2.5-Flash, while sensitivities for
all three Claude variants fall in the [-3%, +7%]
range, suggesting that the Claude-side accuracy
gains operate through a mechanism distinct from
the explicit conflict-resolution trace.P3:explicit
conflict decomposition improves robustness under
temporal drift and noisy distractors, with CDDreaching 71.3% on temporal shifts and 69.9%
on distractor evidence on the full Epi-Scale
adversarial benchmark. These three patterns
identify context-compliance as a structural axis
along which standard RAG can be probed and
intervened on, distinct from retrieval-quality or
single-method robustness questions, and motivate
releasing Epi-Scale for systematic study across
model families and retrieval pipelines.
1 Introduction
Standard Retrieval-Augmented Generation
(RAG) [ 6] can enter aContext-Compliance
Regime: retrieved context dominates the final
answer even when it conflicts with the model’s
parametric knowledge. Such conflicts arise in
benign retrieval mismatch, stale documents, and
deliberate adversarial injection [ 15,19]. For
example, when a widespread misconception such
as “cracking your knuckles causes arthritis” [ 7] is
supplied as retrieved context, a model may follow
the context rather than its internal scientific prior.
This behavior matters because RAG systems
often treat retrieved evidence as authoritative
without directly measuring whether the model
has detected and resolved an epistemic conflict.
Standard accuracy evaluation is insufficient
for this setting. A correct answer does not reveal
whether the model used the retrieved evidence,
ignored it, or followed an internally inconsistent
1arXiv:2605.14473v1  [cs.CL]  14 May 2026

rationale; an incorrect answer does not distinguish
ordinary knowledge failure from context-induced
compliance. Under conflict, we therefore need
diagnostics that expose the relationship between
contextual claims, parametric beliefs, rationales,
and final answers.
We introduceContext-Driven Decom-
position (CDD)as such a probe—a belief-
decomposition procedure that elicits separate
contextual and parametric answers, asks the
model to compare them, isolates conflicting
premises, and records the resolution trace. We
use this trace to measure when retrieved context
causally shapes the final answer through mistake-
injection and truncation interventions. CDD is
not presented as a production defense method and
is not claimed to improve average-case accuracy
universally; its role is to make otherwise implicit
conflict-resolution behavior observable.
We make two substantive contributions and
one resource contribution:
1.Diagnostic framework:We formalize
context compliance within a belief-revision
framework and measure it under con-
trolled synthetic conflict and worst-case
misconception injection.
2.Intervention mechanism:We show that
CDD improves robustness under controlled
retrieval conflict, with consistent gains
across perturbation families and positive
transfer across model families.
3.Resource:We introduceEpi-Scale, a
4,500-sample benchmark for probing com-
pliance, coupling, and robustness regimes
across retrieval settings; data and code will
be released upon publication.
Scope and non-claims.The scope of this pa-
per is diagnostic. We use CDD to expose when
conflict-resolution behavior is present, absent, or
model-specific; we do not claim that CDD is a
universal defense, an average-case state-of-the-art
RAG method, or a substitute for retrieval filtering.
The present evidence should be read as a con-
trolled study, not a deployment recommendation:
we do not establish that CDD solves hallucination
or that it protects against organic multi-document
misinformation retrieval. Those remain follow-up
evaluations.2 Related Work
Knowledge conflict in RAG.The tension be-
tween parametric memory and external evidence
is well documented. Prior work studies when
models should rely on parametric versus retrieved
knowledge [9], how entity substitutions alter QA
outputs [ 8], and how models respond to conflict-
ing retrieved evidence [ 15,3]. Unlike work that
primarily evaluates whether a system answers cor-
rectly under conflict, we use decomposition as a
probe for the model’s conflict-resolution behavior.
Recent 2024 work has refined this picture
along three axes. ClashEval [ 14] systematically
quantifies the “tug-of-war” between an LLM’s
parametric prior and external evidence as a
function of evidence quality, framing the conflict
as a measurable spectrum. ASTUTE RAG
[12] addresses imperfect retrieval and knowl-
edge conflict at the method level by combining
internal-knowledge elicitation with iterative
consolidation. Corrective Retrieval-Augmented
Generation (CRAG) [ 17] inserts a retrieval
evaluator that classifies retrieved evidence into
correct/ambiguous/incorrect bins before gen-
eration, modifying the retrieval pipeline rather
than the generation step. A recent survey of
knowledge conflicts in LLMs [ 16] organizes
failure modes into context-memory, inter-context,
and intra-memory conflicts. CDD targets the
context-memory conflict axis specifically and
complements these methods by treating the con-
flict as an inference-time observable rather than a
quantity to be filtered, mitigated, or consolidated
upstream of generation.
Robust and filtered RAG.Methods such
as Self-RAG [ 2] train models to generate reflec-
tion tokens, while NLI-filtered RAG removes un-
supported retrieved documents before generation
[18]. These methods attempt to improve genera-
tion quality or retrieval reliability. CDD differs in
objective: it is an inference-time diagnostic that
keeps the conflict visible, even when doing so
does not improve accuracy on a particular model.
Context-aware decoding and parametric-
contextual conflict.Context-Aware Decoding
contrasts contextual and parametric logits to re-
duce over-reliance on misleading context [ 10].
Our formulation is related, but we target closed
API models where logit-level access is unavail-

able. Instead of computing token-level divergence
directly, CDD elicits contextual and parametric
answers in natural language and treats the result-
ing trace as an observable diagnostic artifact.
Faithfulness of chain-of-thought.A critical
challenge in chain-of-thought (CoT) reasoning
is whether the generated rationale actually influ-
ences the final answer. Turpin et al. [ 11] show
that CoT explanations can be systematically mis-
leading. Lanham et al. [ 5] introduce causal inter-
ventions such as truncation and mistake injection.
We adapt these interventions to retrieval-induced
conflict, using them to measure whether CDD’s
resolution trace is causally coupled to the answer.
3A Belief-Revision Framework
for Context-Parametric Conflict
3.1 Conceptual Formulation of Conflict
Knowledge conflicts in LLMs can be conceptual-
ized under a belief-revision framework [ 1]. We
distinguish between the model’s parametric prior
Pθ(a|q) —internal world knowledge—and its
contextual posterior Pθ(a|q, c) conditioned on
retrieved evidence. TheCompliance Regime
arises when the posterior dominates despite di-
vergence from the prior; standard RAG can enter
this regime because generation is directly con-
ditioned on retrieved evidence. TheResolution
Regime, in contrast, requires detection and ar-
bitration of divergence. While the conceptual
axis between these regimes corresponds to a di-
vergence measure between the two distributions
(e.g., Jensen-Shannon Divergence), exact token-
level computation requires white-box logit access.
Our prompt-based instantiation makes this diver-
gence observable through explicit answer genera-
tion rather than logit comparison; we treat token-
level JSD instantiation on open-weight models as
a natural extension.
3.2 The CDD Framework
CDD makes the implicit conflict explicit via a
five-step reasoning trace (Figure 1). These five
steps function as a probe instrument: each step
exposes a specific belief-revision operation that
remains implicit under standard RAG.
Step 1: Contextual Extraction:Outputˆa ctx.Step 2: Parametric Extraction:Outputˆa param .
Step 3: Divergence Check:Compare ˆactxand
ˆaparam .
Step 4: Premise Isolation:If they conflict, extract
discrete premises fromc.
Step 5: Resolution:Evaluate premises against the
elicited parametric answer to output the fi-
nal answer.
3.3 Algorithmic Variant: CDD-α
We also reportCDD- α, a compute-aware rout-
ing variant used to study compute/accuracy
trade-offs. The context cis segmented into
sentences {si}. An NLI model scores entailment:
Score i=P NLI(Contradiction|s i,ˆaparam ). If
max(Score i)> τ , the sample is routed to the
deep CDD logic. Otherwise, it defaults to Stan-
dard RAG. This routing rule operationalizes a
selective-intervention setting in which only high-
conflict examples invoke the full decomposition
trace.
4 Experimental Setup
4.1 The Epi-Scale Benchmark
Epi-Scale contains 4,500 instances drawn evenly
fromHotpotQA(multi-hop),Natural Questions
(single-hop), andFEVER(fact verification).
Construction:50% of the data is cleanly re-
trieved context. The remaining 50% is passed
through an LLM-based perturbation engine pa-
rameterized for high semantic variance. We gener-
ated four mutually exclusive perturbation subsets:
Entity Swap, Temporal Shift, Logical Contradic-
tion, and Distractor Evidence. (Full generation
details in Appendix B).
Limitations of Synthetic Perturbations:
While Epi-Scale improves upon templated
datasets, LLM-generated adversarial texts often
exhibit lower perplexity and uniform lexical
diversity compared to organic human misinforma-
tion. We mitigate this by including a real-world
evaluation onTruthfulQA[7].

Query
+ Context
NLI
Gate
τContextual
ExtractionParametric
ExtractionDivergence
CheckPremise
IsolationResolution
Standard RAG
BypassFinal
Answer1 2 3 45
sNLI> τ
sNLI≤τFull CDD probe (Steps 1–5)
Figure 1: CDD pipeline with the CDD-αNLI-gated bypass. High-conflict samples enter the full decom-
position probe, while low-conflict samples follow the Standard RAG bypass before converging at the final
answer.
4.2 Evaluation Scope
We use three evaluation settings, each with a dif-
ferent diagnostic role. Epi-Scale synthetic con-
flict is a controlled perturbation stress test for
isolating specific conflict types. TruthfulQA mis-
conception injection is a worst-case upper-bound
compliance test, not an organic retrieval bench-
mark. Claude-family replication is a cross-model
diagnostic check for whether the observed cou-
pling and conflict-resolution signals are model-
specific. The main adversarial analysis reports the
full Epi-Scale adversarial split: 2,250 examples
evenly divided across the four perturbation types
(∼562–563 per type). Per-perturbation accuracies
in Table 1 are computed over all examples in each
group, and the macro average is the unweighted
arithmetic mean across the four perturbation cells.
4.3 Evaluation Metrics
We useNormalized Match(lowercase, punctua-
tion stripping, alias mapping) to prevent penaliz-
ing safe hedging.
•Accuracy and macro average:Per-
perturbation accuracy is computed indepen-
dently for each perturbation group. “Macro
Avg.” denotes the unweighted arithmetic
mean of the four displayed perturbation
accuracies.
•Confidence intervals:For Table 1,
each per-cell 95% CI is a normal-
approximation binomial interval of half-
width z0.975p
ˆp(1−ˆp)/n over the ∼562–
563 examples in that perturbation group.
The macro-average half-width is computed
asz0.975×(1/4)×pP
iˆpi(1−ˆp i)/ni,under the assumption that the four per-
turbation cells are independent samples.
We note that this propagation does not
capture potential correlations introduced
by the shared question pool from Hot-
potQA/NQ/FEVER.
•Causal Sensitivity:We quantify faithful-
ness using intervention tests (Truncation
and Mistake Injection) [ 5]. Sensitivity
is the relative accuracy drop, defined as
(Acc clean−Acc corrupted )/Acc clean.
4.4 Significance Reporting
In place of paired hypothesis tests, we report a
CI-based conservative significance check on the
largest method gaps. For the full adversarial split,
we report per-cell binomial 95% confidence inter-
vals; the non-overlap of CIs between CDD and the
strongest non-CDD baseline (Self-RAG) on En-
tity Swap (88.0% ±2.7 vs 69.5% ±3.8) and Logi-
cal Contradiction (83.2% ±3.1 vs 65.0% ±3.9) in-
stantiates this check on the largest observed gaps.
Mixed clean/adversarial comparisons use paired
bootstrap over the full 4,500-sample Epi-Scale
benchmark, with the harmonic mean of clean and
adversarial accuracy as the aggregate robustness
metric.
4.5 Model Setup
We use gemini-2.5-flash-001 for pri-
mary evaluations. To study cross-architecture
generalization, we extend evaluations to Claude
Haiku, Sonnet, and Opus; exact API model
identifiers are listed in the reproducibility state-
ment. Closed-API behavior may still drift on
non-pinned dependencies, which we treat as a
reproducibility limitation.

5 Results
5.1 Adversarial Stress Test and Baselines
We compare CDD against existing baselines not
to claim method superiority, but to verify that the
probe registers a non-trivial signal under adver-
sarial context—a prerequisite for the faithfulness
analysis in §5.5.
Figure 2 visualizes the perturbation-level pat-
tern, while Table 1 reports the exact values for
CDD, chain-of-thought prompting (CoT) [ 13],
Self-RAG [ 2], and NLI-filtered RAG [ 18] on the
full Epi-Scale adversarial split (N=2,250) using
gemini-2.5-flash-001.
Standard RAG reaches a 63.0% macro
average under targeted misinformation across
the full adversarial benchmark. CDD reaches
78.1%, with the largest observed differences
on explicit factual manipulations (Entity Swap:
88.0% and Logical Contradiction: 83.2%) and
clear gains on Temporal Shift (71.3%) and
Distractor Evidence (69.9%). This pattern
indicates that explicit conflict decomposition
improves robustness across perturbation families
ongemini-2.5-flash-001 ; it should be
read as a controlled robustness gain rather than a
universal deployment guarantee.
Average-case calibration.In a separate
mixed clean/adversarial calibration run, CDD and
Standard RAG are statistically tied (72.23% and
72.33% harmonic means, respectively; bootstrap
p=0.5311). We treat this as a conservative cali-
bration check on average-case behavior; the main
claim remains the controlled-conflict robustness
gains reported above.
5.2 Ablation Study
Table 3 details the component ablation. Remov-
ing explicit Premise Isolation (Step 4) causes the
reported adversarial aggregate to drop to 65.1%
(numerically close to Vanilla CoT in Table 1, with
no statistical difference test reported), suggesting
that isolating the specific contradictory sentence
is important for the observed diagnostic signal.
To rule out that the observed CDD signal de-
rives merely from increased generation length,
we evaluate a length-matched sham variant: a
5-step prompt structurally similar to CDD but
with semantically vacuous steps (e.g., “restatethe question”, “list entities”). The sham variant
obtains a 40.1% reported adversarial aggregate
(Table 3), below full CDD (78.1%). This supports
the narrower claim that the type of decomposition
matters; adding generic multi-step structure is not
sufficient and may worsen context-compliance
behavior.
5.3 Diagnostics & Compute Tradeoff
(CDD-α)
To analyze the compute tradeoff, we evaluated
CDD- αatτ= 0.7 . The threshold was selected
by inspecting NLI contradiction-score histograms
on a 100-sample held-out subset and choosing a
value that separated the visible high-conflict clus-
ter from the low-conflict tail; we did not perform
a full τsweep, which we note as a limitation in
the discussion of CDD- α. Atτ= 0.7 , CDD- α
routed 30.0% of samples to deep CDD reasoning,
bypassing the remaining 70% to Standard RAG.
This setting yielded 68.5% adversarial accuracy
(Figure 4), between Standard RAG and full CDD
while using fewer API tokens than full CDD.
5.4 Real-World Misinformation (Truth-
fulQA) — P1
Setup:We extracted 500 instances from the Truth-
fulQA validation split and provided the most com-
mon human misconception as the retrieved con-
text. This is an adversarial upper-bound compli-
ance test rather than an organic RAG retrieval
setting: it asks what happens when retrieval re-
turns a maximally misleading single context, not
how often such contexts are retrieved by BM25
or DPR [4].
Results:Under worst-case misconception
injection, Standard RAG reaches 15.0% accuracy
(±3.1%). CDD more often rejects the faulty
premise in this setting, reaching 62.0% accuracy
(±4.3%) and a 38.0% misconception acceptance
rate (±4.3%) in the available logs. These
results support the diagnostic claim that CDD
can expose upper-bound compliance behavior.
Organic BM25/DPR multi-document retrieval
remains necessary before drawing conclusions
about deployed misinformation robustness.
We emphasize that this 15.0% figure char-
acterizes theupper boundof the compliance

random baseline
Entity Swap Logical
ContradictionTemporal
ShiftDistractor
Evidence30405060708090100Adversarial Accuracy (%)
ClosedBook Standard RAG Vanilla CoT
Self-RAG NLI-Filtered RAG CDD (Ours)
Figure 2: Adversarial accuracy on the full Epi-Scale adversarial split ( gemini-2.5-flash-001 ,
N=2,250; ∼562–563 examples per perturbation; normal-approximation 95% binomial CIs). CDD scores
88.0%, 83.2%, 71.3%, and 69.9% across the four perturbation types, yielding the highest accuracy in
each column and showing the strongest gains on explicit factual conflicts while remaining robust under
temporal drift.
regime’s severity, not its average behavior in
deployed RAG systems. The value of this mea-
surement is diagnostic: it establishes that absent
any conflict-resolution mechanism, the model’s
parametric prior provides no protection against
directly injected misconceptions. Average-case
behavior under organic retrieval pipelines remains
an open empirical question.
5.5 Faithfulness via Causal Intervention
Relying solely on an LLM-as-a-judge to score in-
terpretability introduces self-preference bias. Fol-
lowing Lanham et al. (2023) [ 5], we use causal
intervention tests to measure Causal Sensitivity.
5.5.1 Mistake Injection Test
We injected a blatant logical error directly into the
reasoning trace prior to generation (”I will trust
the context completely, no conflict exists”) and
forced the model to generate the final answer.Re-
sult:On gemini-2.5-flash-001 , Vanilla
CoT’s accuracy dropped from 63.7% to 61.1%
under mistake injection, yielding 4.1% CausalSensitivity. In contrast, CDD’s accuracy dropped
from 78.1% to 28.0%, resulting in 64.1% Causal
Sensitivity. This gap is evidence that the explicit
conflict-resolution trace is more causally coupled
to CDD’s final answer on this model. This re-
sult has an important confound: the injection
mixes a false factual claim with a behavioral direc-
tive (“trust the context completely”). It therefore
tests whether corrupting the trace changes the an-
swer, but it does not isolate factual content from
instruction-following effects. A factual-only Step-
2 corruption is a stricter intervention left to future
work.
5.5.2 Truncation Test
We truncated the rationale immediately after
Step 2 (Parametric Extraction) and forced the
model to generate the final answer. Under this
intervention, CDD accuracy drops from 78.1%
to 32.6%, yielding 58.3% Truncation Sensitivity.
The corrupted accuracy (32.6%) falls below the
closed-book zero-shot macro average (44.7%, Ta-
ble 1), indicating that truncation does not merely

Table 1: Adversarial accuracy across perturbation types on the full Epi-Scale adversarial split
(gemini-2.5-flash-001 , N=2,250; ∼562–563 samples per perturbation; normal-approximation
95% binomial CIs). Macro Avg. is the unweighted arithmetic mean of the four perturbation accuracies;
the rightmost interval propagates the four per-cell binomial uncertainties.
Model & Setting Entity Swap Log. Contradict. Temp. Shift Distract. Evid. Macro Avg.
ClosedBook (Zero-shot) 43.7%(±4.1) 40.7%(±4.1) 50.0%(±4.1) 44.4%(±4.1) 44.7%(±2.0)
Standard RAG 58.4%(±4.1) 56.0%(±4.1) 68.8%(±3.8) 68.8%(±3.8) 63.0%(±2.0)
Vanilla CoT 62.0%(±4.0) 61.3%(±4.0) 63.2%(±4.0) 68.1%(±3.9) 63.7%(±2.0)
Self-RAG (Prompted) 69.5%(±3.8) 65.0%(±3.9) 66.0%(±3.9) 67.5%(±3.9) 67.0%(±1.9)
NLI-Filtered RAG 68.0%(±3.9) 64.5%(±4.0) 65.5%(±3.9) 67.0%(±3.9) 66.2%(±2.0)
CDD (Ours) 88.0%(±2.7) 83.2%(±3.1) 71.3%(±3.7) 69.9%(±3.8) 78.1%(±1.7)
Table 2: Cross-model replication on the full
Epi-Scale adversarial split (Claude-family mod-
els, N=2,250 adversarial examples). CDD im-
proves on all three Claude-family models, support-
ing cross-model generalization and architecture-
robust transfer of the decomposition mechanism.
Method Haiku Sonnet Opus
Standard RAG 79.0% 76.0% 79.4%
Vanilla CoT 73.6% 73.2% 75.4%
CDD (Ours) 81.2% 80.6% 82.0%
Table 3: Component-wise ablation on the
gemini-2.5-flash-001 ablation run. The
“Overall” column is the run-level ablation aggre-
gate; Entity Sw. and Log. Con. isolate the two
perturbations where ablation effects are largest.
The full-benchmark CDD macro average is re-
ported separately in Table 1.
Ablation Variant Entity Sw. Log. Con. Overall
Full CDD 88.0% 83.2% 78.1%
Length-Matched Sham CoT 42.4% 32.0% 40.1%
w/o Step 4 (Isolation) 75.2% 72.0% 65.1%
w/o Step 3 (Diverge) 77.0% 73.5% 66.0%
remove the resolution steps but also leaves the
model in a partially executed structured-output
protocol whose termination tokens (e.g., <final
answer> ) are no longer reachable through the
prompted procedure. We therefore interpret the
58.3% sensitivity as consistent with—but not
independent confirmation of—the 64.1% mistake-
injection result: both interventions disrupt the
resolution trace and yield substantial accuracy
drops, but neither cleanly isolates trace removal
from protocol or instruction-following confounds.
We also note that the mistake-injection corruptedaccuracy (28.0%, §5.5.1) is lower than the trun-
cation corrupted accuracy (32.6%), indicating
that the behavioral directive in mistake injection
has a stronger disruptive effect than removing
Steps 3–5 entirely. This is consistent with the
view that mistake-injection sensitivity (64.1%)
upper-bounds the trace’s causal contribution
by including instruction-following effects. The
mistake-injection result in §5.5.1, which pre-
serves the output protocol, remains our primary
causal-faithfulness measurement; stricter step-
specific corruption tests are deferred to future
work as discussed in Limitations.
5.5.3 Cross-Model Faithfulness
To explore cross-architecture consistency, we ap-
plied the Mistake Injection test (N=100) to the
Claude family.
Figure 3 shows a clear dissociation between
two signals. Mistake-injection causal sensitivity,
which tests whether corrupting the resolution
trace changes the answer, is large on Gemini-
2.5-Flash (64.1%) but vanishes on all three
Claude variants (in the [-3%, +7%] range). In
contrast, adversarial accuracy gains (Table 2)
transfer positively across the Claude family.
This dissociation matters: the Claude-family
accuracy improvements cannot be attributed to
the explicit conflict-resolution trace causally
driving the answer, because the coupling signal
that would test such causation is absent. Plau-
sible alternative mechanisms include increased
output length, format-induced calibration, or
alignment-training-specific responses to multi-
step prompts. Disentangling these requires
open-weight replication and is left to future work.

CDD coupling signal absent
shaded region: noise band [-3%, +7%]
Gemini-2.5-Flash Claude Haiku Claude Sonnet Claude Opus020406080
+4.1 +2.8+20.3
+1.3+64.1
+6.5
−2.7 −1.3Mistake-injection causal sensitivity (%)Vanilla CoT
CDD (ours)
Figure 3: Mistake-injection causal sensitivity across model families (N=100 per cell). On Gemini-2.5-
Flash, CDD’s resolution trace is strongly coupled to the final answer (64.1%); on all three Claude variants
the signal falls inside the shaded noise band, even though adversarial accuracy still improves under CDD
(Table 2). The Claude-Sonnet Vanilla-CoT bar (20.3%) is an outlier discussed in §6.3.
1 1.5 2 2.5 3607080
Standard RAG (63.0%)CDD-α,τ=0.7(68.5%)Full CDD (78.1%)
+9.6pp1.4×compute
Relative compute cost (×Standard RAG)Adversarial accuracy (%)
Figure 4: Compute–accuracy trade-off on thegemini-2.5-flash-001adversarial split. CDD-αat
τ= 0.7 routes 30% of samples through deep decomposition and reaches 68.5%; the remaining 9.6 pp gap
to Full CDD costs roughly 1.4× more compute. Relative compute is approximate; exact ratios depend on
token-level prompt and rationale lengths.
6Findings: Conflict-Aware Ro-
bustness
6.1 Error Analysis
The perturbation-level pattern suggests three re-
curring robustness modes. Entity Swap and Logi-
cal Contradiction benefit most from CDD because
the conflicting premise is localized and can be iso-
lated as a discrete factual claim. Temporal Shift
now also improves, which is consistent with ex-
plicit conflict decomposition helping the model
reconcile stale parametric priors with newer re-
trieved evidence rather than over-privileging one
source of information. Distractor Evidence re-mains harder than the other perturbations, but the
CDD gain is still positive, indicating that the de-
composition trace helps even when the retrieved
context is noisy or partially irrelevant.
6.2 Finding 1: Temporal Robustness Un-
der Conflict (P3)
As seen in Figure 2 and Table 1, CDD improves
on Temporal Shift (71.3% vs 68.8% for Standard
RAG), suggesting that explicit conflict decompo-
sition can mitigate stale-parametric-prior effects
rather than amplifying them. Vanilla CoT, which
shows lower mistake-injection sensitivity on Gem-
ini (4.1%), tracks the context more loosely, while

CDD provides a stronger and more stable conflict-
resolution signal under temporal drift.
6.3 Finding 2: Cross-Family Accuracy
Transfer (P2)
The dissociation between accuracy transfer (Ta-
ble 2) and causal-coupling transfer (Figure 3) sug-
gests a more nuanced principle: what transfers
across model families is the adversarial accuracy
benefit of explicit decomposition, not the underly-
ing causal coupling between the resolution trace
and the final answer. On Gemini-2.5-Flash, the
trace is causally entangled with the answer (64.1%
mistake-injection sensitivity); on the Claude fam-
ily, the same prompting intervention raises ad-
versarial accuracy without leaving a measurable
causal-coupling footprint. We therefore phrase
the finding conservatively: explicit decomposition
is a robust accuracy intervention across families
in our controlled conflict setting, but its mech-
anism appears family-specific. Identifying the
Claude-side mechanism—whether it is alignment-
training-specific behavior, output-length effects,
or format-induced calibration—is an open ques-
tion for follow-up work.
Limitations
We explicitly acknowledge the following limita-
tions:
•Cross-Family Generalization: The
strongest causal-sensitivity signal (64.1%
mistake-injection sensitivity) still comes
from Gemini-2.5-Flash, and the magnitude
of the effect varies by model family. That
said, the Claude-family reruns now show
positive adversarial accuracy gains as well,
so future work should isolate whether
the remaining variation comes from ar-
chitecture, alignment training, or prompt
sensitivity. Controlled studies on open-
weight models with matched parameter
scale and varied training recipes would be
needed to discriminate these hypotheses.
•CDD- αThreshold Tuning: The CDD- α
routing threshold τ= 0.7 was selected
from a 100-sample histogram inspection
rather than a systematic sweep. A full τ∈{0.3,0.5,0.7,0.9} ablation with multiple
NLI backbones is needed before the re-
ported Pareto point can be interpreted as
deployment-relevant.
•TruthfulQA Realism:Using the explicit
top misconception as context is a worst-
case scenario. Future evaluations must use
realistic BM25 retrieval pipelines to mea-
sure average-case degradation.
•Conceptual-Only Belief-Revision Fram-
ing: Our belief-revision framework oper-
ates at the conceptual level rather than com-
puting token-level divergence (e.g., JSD)
between prior and posterior distributions.
A white-box instantiation on open-weight
models would enable direct quantitative val-
idation of the compliance/resolution regime
distinction, which we leave to future work.
•Stricter Faithfulness Tests: Our mistake-
injection intervention combines factual con-
tent and behavioral directives. Stricter inter-
ventions should include factual-only Step-2
corruption, rationale swaps across exam-
ples, step-specific corruption, and answer-
hidden interventions that prevent the con-
text from leaking the correct answer. De-
signing these requires careful construction
of contexts that do not themselves reveal
the answer.
•Closed-API Drift: The primary models are
closed APIs. Exact behavior may change
as providers update model snapshots, so
future work should replicate the diagnostics
on version-pinned open-weight models.
Future Work
The most important next evaluations are organic
BM25/DPR retrieval for TruthfulQA; multi-
document retrieval with one false passage mixed
with ktrue passages; factual-only rationale
intervention; rationale-swap tests; step-specific
corruption tests; and open-weight replication with
logit-level prior/posterior divergence.

Reproducibility Statement
We preserve anonymity and will release code,
prompts, and Epi-Scale data upon publication.
The API identifiers used in the reported exper-
iments are:
Geminigemini-2.5-flash-001
Claude Haikuclaude-haiku-4-5-20251001
Claude Sonnetclaude-sonnet-4-6
Claude Opusclaude-opus-4-6
All Claude model IDs are version-pinned snap-
shots: Haiku 4.5 uses the dated format from
pre-4.6 generations, while Sonnet 4.6 and Opus
4.6 use the dateless pinned-snapshot format
introduced with the 4.6 generation. Experiments
were conducted between January and March
2026. Closed-API model behavior may drift
on non-pinned dependencies (e.g., moderation
layers), which we treat as a reproducibility limi-
tation. All runs use temperature 0.0 and greedy
decoding constraints. Dataset construction draws
evenly from HotpotQA, Natural Questions, and
FEVER; perturbations are generated along four
mutually exclusive axes and audited manually
on 50 examples per subgroup (92% valid con-
flict generation rate). Metrics use normalized
string matching with lowercase, punctuation
stripping, and alias mapping; FEVER labels are
mapped to Boolean support/refute classes in
the appendix. Confidence interval computation
follows the formulas detailed in §4.3. We release
per-example predictions, prompts, and seeds
alongside the code to support paired follow-up
analyses. Compute resources: All experiments
use closed-API inference at standard rate limits;
we did not perform local fine-tuning or training.
7 Conclusion
We used CDD as an inference-time probe to
study how standard RAG handles epistemic
conflict. Three structural observations emerge:
(P1) standard RAG can enter a measurable
context-compliance regime under worst-case
misconception injection; (P2) adversarial accu-
racy gains transfer across Gemini and Claude
families, but rationale-answer causal coupling
does not, suggesting thatwhat improvesandwhy
it improvescan decouple across architectures; and
(P3) explicit conflict decomposition improvestemporal robustness rather than amplifying
stale-prior effects, as seen in the Temporal Shift
results.
These findings position context-compliance
as a structural RAG behavior that benefits from
systematic diagnostics and conflict-aware inter-
vention. In this controlled setting, CDD exposes
where retrieved context changes the answer, and
it does so while improving robustness across
perturbation families and model families. We
release Epi-Scale to enable systematic study of
these regimes across model families and retrieval
pipelines.
References
[1]Carlos E. Alchourr ´on, Peter G ¨ardenfors, and
David Makinson. On the logic of theory
change: Partial meet contraction and revi-
sion functions.The Journal of Symbolic
Logic, 50(2):510–530, 1985.
[2]Akari Asai, Zeqiu Wu, Yizhong Wang,
Avirup Sil, and Hannaneh Hajishirzi. Self-
RAG: Learning to retrieve, generate, and
critique through self-reflection. InInterna-
tional Conference on Learning Representa-
tions, 2024.
[3]Wenhu Chen, Hongyi Huang, Xueguang
Wang, et al. Rich knowledge sources bring
complex knowledge conflicts: Recalibrating
models to reflect conflicting evidence. In
Proceedings of the 2022 Conference on Em-
pirical Methods in Natural Language Pro-
cessing, 2022.
[4]Vladimir Karpukhin, Barlas Oguz, Sewon
Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih.
Dense passage retrieval for open-domain
question answering. InProceedings of the
2020 Conference on Empirical Methods in
Natural Language Processing, 2020.
[5]Tamera Lanham, Anna Chen, Ansh Rad-
hakrishnan, Benoit Steiner, Carson Denison,
Danny Hernandez, et al. Measuring faithful-
ness in chain-of-thought reasoning.arXiv
preprint arXiv:2307.13702, 2023.

[6]Patrick Lewis, Ethan Perez, et al. Retrieval-
augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural
Information Processing Systems, 2020.
[7]Stephanie Lin, Jacob Hilton, and Owain
Evans. TruthfulQA: Measuring how models
mimic human falsehoods. InProceedings of
the 60th Annual Meeting of the Association
for Computational Linguistics, 2022.
[8]Shayne Longpre, Yi Lu, Nan Du, Diyi Yang,
and Mohit Iyyer. Entity-based knowledge
conflicts in question answering. InProceed-
ings of the 2021 Conference on Empirical
Methods in Natural Language Processing,
2021.
[9]Alex Mallen, Akari Asai, Victor Zhong, Ra-
jarshi Das, Daniel Khashabi, and Hannaneh
Hajishirzi. When not to trust language mod-
els: Investigating effectiveness of paramet-
ric and non-parametric memories. InPro-
ceedings of the 61st Annual Meeting of the
Association for Computational Linguistics,
2023.
[10] Freda Shi, Xinyun Chen, et al. Trusting
your evidence: Hallucinate less with context-
aware decoding. InProceedings of the 2023
Conference of the North American Chapter
of the Association for Computational Lin-
guistics, 2023.
[11] Miles Turpin, Julian Michael, Ethan Perez,
and Samuel Bowman. Language models
don’t always say what they think: Unfaithful
explanations in chain-of-thought prompting.
InAdvances in Neural Information Process-
ing Systems, 2023.
[12] Fei Wang, Xingchen Wan, Ruoxi Sun,
Jiefeng Chen, and Sercan ¨O. Arık. AS-
TUTE RAG: Overcoming imperfect re-
trieval augmentation and knowledge con-
flicts for large language models.arXiv
preprint arXiv:2410.07176, 2024.
[13] Jason Wei, Xuezhi Wang, Dale Schuur-
mans, Maarten Bosma, Brian Ichter, Fei Xia,
Ed Chi, Quoc Le, and Denny Zhou. Chain-
of-thought prompting elicits reasoning inlarge language models. InAdvances in Neu-
ral Information Processing Systems, 2022.
[14] Kevin Wu, Eric Wu, and James Zou. ClashE-
val: Quantifying the tug-of-war between an
LLM’s internal prior and external evidence.
arXiv preprint arXiv:2404.10198, 2024.
[15] Jian Xie et al. Adaptive chameleon or stub-
born sloth? unraveling the behavior of large
language models in knowledge conflicts. In
International Conference on Learning Rep-
resentations, 2023.
[16] Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunx-
iang Wang, Hongru Wang, Yue Zhang, and
Wei Xu. Knowledge conflicts for LLMs: A
survey. InFindings of EMNLP 2024, 2024.
[17] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and
Zhen-Hua Ling. Corrective retrieval
augmented generation.arXiv preprint
arXiv:2401.15884, 2024.
[18] Ori Yoran, Tomer Wolfson, Ori Ram,
and Jonathan Berant. Making retrieval-
augmented language models robust to irrel-
evant context. InInternational Conference
on Learning Representations, 2023.
[19] Wei Zou, Runpeng Wang, Xiaowei Yang,
et al. PoisonedRAG: Knowledge poisoning
attacks to retrieval-augmented generation
of large language models.arXiv preprint
arXiv:2402.07867, 2024.

AAppendix: Reproducibility De-
tails
All evaluations use temperature 0.0and greedy
decoding constraints. String normalization
mapping for FEVER: [supports, true, yes] →
True; [refutes, false, no, contradicts] →False.
Standard RAG Template:
Context:{context}
Question:{question}
Based on the context, answer
concisely.
Vanilla CoT Template:
Context:{context}
Question:{question}
Think step-by-step, then
answer the question. Wrap
final answer in
<final answer>.
Length-Matched Sham CoT Template:
Context:{context}
Question:{question}
Follow these five steps
before answering.
Step 1: Restate the question
in your own words.
Step 2: Restate the
retrieved context in your own
words.
Step 3: List the named
entities or key terms
mentioned in the question and
context.
Step 4: Identify the broad
question type (e.g., person,
place, date, fact
verification, or
explanation).
Step 5: Answer the question
concisely based on the
context.
Wrap your final concise
answer in <final answer>
tags.
Self-RAG (Prompted) Template:
Given the context, answer the question. You must include reflection tokens.
Context: {context}Question: {question}
Format your response as:
[Relevant: Yes/No]
[Supported: Fully/Partially/No]
[Contradicts Prior: Yes/No]
Final Answer: <your concise answer>
CDD Template:
You are resolving cognitive
conflicts in
retrieval-augmented
generation.
Context:{context}
Question:{question}
Follow these steps:
Step 1: Extract the answer
implied by the context.
Step 2: State your
parametric/internal knowledge
answer to the question.
Step 3: Compare the two
answers. Do they conflict?
Step 4: If they conflict,
isolate the premises causing
the conflict. Evaluate if
they violate established
history/science.
Step 5: Resolve the conflict
and provide the most
factually reliable final
answer.
Wrap your final concise
answer in <final answer>
tags.
B Appendix: Epi-Scale Details
Epi-Scale comprises 4,500 samples (1,500 each
from HotpotQA, NQ, and FEVER). The dataset
is split 50% clean, 50% adversarial. The adver-
sarial subset is divided uniformly among Entity
Swap, Temporal Shift, Logical Contradiction,
and Distractor Evidence. The perturbations were
generated using gemini-2.5-flash-001
with prompt templates constraining semantic
alterations to exactly one variable axis. To ensure
the LLM-based perturbation engine successfully
generated genuine conflicts without destroying
syntax, we manually audited a random sample of
50 instances per subgroup, finding a 92% valid
conflict generation rate.
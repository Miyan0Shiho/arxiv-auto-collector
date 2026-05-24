# Claim-Selective Certification for High-Risk Medical Retrieval-Augmented Generation

**Authors**: Shao Kan

**Published**: 2026-05-21 03:29:50

**PDF URL**: [https://arxiv.org/pdf/2605.21949v1](https://arxiv.org/pdf/2605.21949v1)

## Abstract
Medical RAG systems in high-risk QA settings are often evaluated through a single answer-or-abstain decision, but mixed evidence may support one claim, require conditions for another, and contradict a third. We study claim-selective certification: each response is decomposed into verifiable claims, scored against retrieved evidence, and mapped by an intent-aware selector to {full, partial, conflict, abstain}. On the primary weak-label certificate protocol, whose real-source-only dev/test rows cover the naturally occurring non-abstain actions, the full system records UCCR=0.0000, PAU=1.0000, PAU Precision=0.9901, and action accuracy=0.9204 on dev (n=314), and UCCR=0.0000, PAU=0.9967, PAU Precision=0.9739, and action accuracy=0.8997 on test (n=319). UCCR measures unsupported-claim risk within the certificate definition, and a source-missing counterfactual slice evaluates abstain under empty evidence. Shortcut controls quantify the action-label prior explained by source and intent metadata, while source/evidence-novel slices characterize transfer boundaries. The resulting interface separates action-label prediction from evidence-linked claim selection under mixed evidence.

## Full Text


<!-- PDF content starts -->

Claim-Selective Certification for High-Risk Medical
Retrieval-Augmented Generation
Shao Kan
Jinglue Technology Development (Nanjing) Co., Ltd.
Room 1215-13, 12th Floor, Building A2, Huizhi Science and Technology Park,
No. 8 Hengtai Road, Nanjing Economic and Technological Development Zone,
Nanjing, China
shaokan1991@gmail.com
https://orcid.org/0009-0003-4872-6193
May 22, 2026
Abstract
Medical RAG systems in high-risk QA settings are often evaluated through a single answer-or-abstain decision, but
mixed evidence may support one claim, require conditions for another, and contradict a third. We studyclaim-selective
certification: each response is decomposed into verifiable claims, scored against retrieved evidence, and mapped by an
intent-aware selector to {full, partial, conflict, abstain}. On the primary weak-label certificate protocol, whose real-
source-only dev/test rows cover the naturally occurring non-abstain actions, the full system records UCCR =0.0000 ,
PAU=1.0000 , PAU Precision =0.9901 , and action accuracy =0.9204 on dev ( n=314 ), and UCCR =0.0000 , PAU =
0.9967 , PAU Precision =0.9739 , and action accuracy =0.8997 on test ( n=319 ). UCCR measures unsupported-claim
risk within the certificate definition, and a source-missing counterfactual slice evaluates abstain under empty evidence.
Shortcut controls quantify the action-label prior explained by source and intent metadata, while source/evidence-novel
slices characterize transfer boundaries. The resulting interface separates action-label prediction from evidence-linked
claim selection under mixed evidence.
1 Introduction
Retrieval-augmented generation (RAG) grounds language models in external knowledge [ 1,2], but high-risk medical
QA has an asymmetric answer contract: unsupported safety, dosing, or contraindication claims can be harmful, while
blanket abstention can suppress usable evidence. The same evidence set may support a dosing constraint, leave
monitoring uncertain, and contradict a contraindication. We studyclaim-selective certification: decompose a question
into verifiable claim units, score each claim against evidence, and choose whether to state it directly, state it with
conditions, contest it, or withhold it.
The implementation has three stages: template-based claim decomposition, cue-based relation scoring for support,
conflict, and limitation signals, and anintent-aware risk-calibrated selector. The selector receives a question_intent :
pregnancy, lactation, monitoring, and special-population questions often require condition-limited wording, whereas
contraindication and interaction questions more often require certification or conflict handling.
We evaluate this interface on 2,223 medical QA samples, including 2,103 examples (94.6%) from publicly
downloaded real sources and 120 synthetic examples (5.4%). Primary results use a real-source-only split, with synthetic,
source-missing, and source/evidence-novel slices characterizing behavior under synthetic examples, empty evidence,
and source/evidence novelty. We compare threshold-only, binary-form, NLI-based, learned-relation, and full claim-
selective systems under the same weak-label certificate protocol. The study contributes a claim-level certification/action
formulation, an intent-aware fixed policy layer, and a certificate-producing evaluation framework that reports UCCR,
PAU, PAU Precision, action accuracy, risk–coverage, source-overlap, shortcut, and source/evidence-novel analyses for
high-risk medical RAG.
1arXiv:2605.21949v1  [cs.CL]  21 May 2026

2 Related Work
Retrieval-augmented generation grounds language models by retrieving external documents and conditioning generation
on them [ 1,3,2]. Much of the recent work improves evidence acquisition through dense retrieval [ 4], iterative
retrieval and self-reflection [ 5], query rewriting for RAG pipelines [ 6], or decomposition strategies for compositional
questions [ 7]. RAG methods determine which evidence is available to the model. Our setting starts from the next
decision point: once evidence is available, which claims should appear in the answer?
Selective prediction and abstention reduce risk by withholding low-confidence predictions [8, 9]. Related QA and
dialogue work studies time-sensitive answerability [ 10], calibrated belief-state distributions [ 11], and uncertainty over
meanings rather than surface strings [ 12]. These approaches are effective when an instance is globally answerable,
unanswerable, or uncertain. They are less expressive when a medical question contains several claims with different
evidence states. We keep the risk-control objective but move the decision unit from the whole response to individual
claims and action types.
Faithfulness and fact-checking methods verify whether generated or proposed claims are supported by sources [ 13–
17], while citation-grounded generation work studies how models present attributed evidence in the output itself [ 18].
More directly, long-form factuality evaluators decompose responses into atomic facts and check evidence sup-
port [ 19,20], while efficient grounded checkers such as MiniCheck train smaller models for document-grounded
fact verification [ 21]. These works make factual support measurable at the claim or fact level. Claim-selective RAG
uses the same kind of signal earlier in the pipeline, before the final response is formed, and maps each claim to an
explicit action and certificate state rather than only scoring a completed answer.
Diagnostic evaluation work shows that high aggregate accuracy can hide shortcut behavior, annotation artifacts,
and distribution-shift failures. Prior studies expose such behavior with hypothesis-only artifact analyses [ 22], con-
trolled heuristic challenge sets [ 23], behavioral test suites [ 24], data maps [ 25], and in-the-wild distribution-shift
benchmarks [ 26]. We bring the same perspective to claim-selective RAG: shortcut controls, source/evidence-novel
slices, and source-missing abstention evaluations separate certificate-producing behavior from label-space priors and
transfer failures.
Medical QA has been studied in retrieval, benchmark, and large-model settings [ 27–32]. Many datasets evaluate
multiple-choice, yes/no, or complete-answer behavior. Practical medical questions, however, often need partial support,
contraindication handling, or condition-specific wording. The present work targets that interface gap by evaluating
claim-level selection, explicit action calibration, and risk–utility metrics such as UCCR and PAU.
3 Problem Formulation
Letqdenote a medical question and let E={e 1,...,e k}be the retrieved evidence available to the system. A conventional
RAG policy emits one response r=LM(q,E) and is then evaluated as a whole. This is too coarse for mixed evidence:
the same evidence packet can support one claim, only conditionally support a second, contradict a third, and leave a
fourth unsupported. We instead treat the response as a decision over claims.
The interface first produces a small set of verifiable claim skeletons C={c 1,...,c n}and metadata mifor each
claim, including question intent and claim type. Each claim–evidence pair receives a relation vector
s(ci,ej) = 
ssup,sconf,slim
∈[0,1]3,(1)
where the coordinates summarize support, conflict, and limitation/conditionality. Scores are pooled across evidence
with max aggregation:
Sℓ(ci) =max
ej∈Esℓ(ci,ej), ℓ∈ {sup,conf,lim}.(2)
The selector maps(S sup,Sconf,Slim,mi)to a claim status
zi∈ {certified,condition-limited,conflicting,omitted}.(3)
Certified and condition-limited claims may be expressed with evidence links; conflicting claims trigger conflict behavior;
omitted claims are withheld.
The final action is a coarse summary of the selected claim set:
a∈ {full,partial,conflict,abstain}.(4)
2

Intuitively, full means the requested answer can be stated directly, partial means usable information exists but
requires caveats or omissions, conflict means material evidence disagreement or risk should be surfaced, and
abstain means no claim should be expressed. The action is therefore not a free-form answer label; it is a policy
decision induced by claim-level statuses and evidence certificates.
Evaluation follows the same claim interface. Let Pbe the expressed predicted claims and Gusethe construction-
derived gold-usable claims under the weak-label protocol. Unsupported Critical Claim Rate (UCCR) measures
certificate-level risk among expressed critical claims:
UCCR=|{c∈P:cis critical andc/∈G use}|
|{c∈P:cis critical}|,(5)
defined as zero when no critical claim is expressed. The target is zero unsupported expressed critical claims. Partial
Answer Utility (PAU) measures retained usable information:
PAU=|P∩G use|
|Guse|.(6)
We also report PAU Precision and F1 to distinguish retaining usable claims from over-expressing extra claims, and
Action Accuracy to evaluate the four-way action interface. These are protocol metrics over weak labels, not expert
clinical judgments.
This formulation makes the central tradeoff explicit. If the evidence supports Csup⊂C and does not support the
remaining claims, a document-level answer policy largely chooses between expressing everything, which can raise
UCCR, or abstaining, which lowers PAU. A claim-selective policy can instead express Csup, omit unsupported claims,
and return partial when appropriate. The experiments ask whether this interface can meet the certificate target while
preserving utility and whether the resulting actions remain meaningful beyond shortcut priors.
4 Method
The system follows a three-stage pipeline: claim decomposition, relation scoring, and intent-aware risk-calibrated
selection. We study the post-retrieval decision layer: given a fixed evidence set, the system decides which claims should
be certified, condition-limited, contested, or omitted. Figure 1 summarizes the pipeline. The system exposes a small set
of candidate claims, scores each claim against retrieved evidence, and applies an explicit action policy. This keeps the
main decisions inspectable under weak supervision and makes it possible to evaluate action calibration separately from
open-ended generation.
4.1 Design Rationale
The design centers the post-retrieval decision boundary: the claim interface fixes the decision unit, the relation scorer
exposes support, conflict, and limitation signals, and the selector maps those signals to an action. This structure lets us
study action calibration under a common weak-label protocol while keeping claim selection, certification, and audit tied
to explicit intermediate variables.
4.2 Claim Decomposition
Given a query q, the decomposition layer emits verifiable claim skeletons C={c 1,...,c n}. The templates produce
a small set of high-level critical claims rather than unrestricted fine-grained extractions. Each query also receives
aquestion_intent , covering indication, dosage, contraindication, interaction, pregnancy/lactation, monitoring,
missed-dose, and special-population questions. The resulting skeletons provide a stable interface for studying selective
inclusion, condition-limited wording, conflict handling, and omission under a shared weak-label protocol.
4.3 Relation Scoring
For each claim–evidence pair (ci,ej), the relation module estimates support, conflict, and limitation scores. The scorer
combines lexical overlap with a cue lexicon to identify whether evidence supports a claim, contradicts it, or supports it
3

Evaluated Object: Claim-Selective Certification Policy for High-Risk Medical RAG
1 2 3 4 5 Input
Medical question q
?
Intent cues
...
Retrieved evidence
E = {e1,...,ek}
e1 Passage from source A
e2 Passage from source B
ek Passage from source K...Claim interface
question_intent
dosage pregnancy interaction
Critical claim candidates C = {c1,...,cn}
c1 Dose recommendation is supported
c2 Pregnancy use is condition-limited
c3 Interaction risk is reported
cn Interaction evidence is limited...
stable weak-label unitRelation scoring
claim-evidence pairs
claimse1 e2 ... ekmax-pooled
score
c1
c2
...
cn...
... ...S_supS_conf
S_lim
support conflict
limitation - no evidenceIntent-aware selector
claim statuses
certified (supported)
condition-limited (limited)
conflicting (contradicted)
omitted (no coverage)
final action
{full, partial, conflict, abstain}
full partial conflict abstainOutput
Evidence-linked certificate
Claim-level certificate table
Claim Status Key evidence
c1 e1, e4
c2 e2
c3 e3
... ... ...
Post-retrieval decision layer: auditable claim-level actions from mixed evidence.
Evaluation Artifact: Protocol, Audits, and Boundary Slices
A B C D Primary protocol
- real-source-only dev/test
- non-abstain main protocol
UCCR PAU
PAU Precision F1
Action AccShortcut audits
intent-majority no-intent
source+intent intent shuffled
source+claim-type evidence shuffled
separates action-label priors
from evidence-linked certificationBoundary slices
- source/evidence-novel holdout
- source-missing abstention stress
- source-level diagnostics
OpenFDA FDA FAERS
PubMed Literature PubMedQA
tests transfer and interface completenessScope statement
Weak-label certificate protocol
Measures auditable claim selection,
not an independent
clinical-safety guarantee.
Claim status (per claim)
certified (supported)
condition-limited (limited)conflicting (contradicted)
omitted (no coverage)Final action (policy-level)
fullall required critical
claims certified
partialsome required claims certified;
limitations presentconflictmaterial conflicts present
abstainno expressible claimFigure 1: System architecture. The pipeline combines template-based claim decomposition with explicit question intent,
cue-based relation scoring, and an intent-aware risk-calibrated selector.
Table 1: Claim-template and selector-policy interface. The rows summarize the operating families used by the selector.
Question group Claim skeleton and policy family
Indication / effectiveness Drug is indicated or supported for a condition; certify only with support,
otherwise mark as condition-limited or omit.
Pregnancy, lactation, special population Use is appropriate under a population-specific condition; prefer condition-
limited wording when limitation cues are present.
Contraindication / interaction Evidence reports a prohibition, interaction, or material risk; disclose con-
flict/risk rather than directly certify.
Dosage, missed dose, monitoring A specific dosing or monitoring action is supported; require explicit instruction-
level support.
only with conditions. The cue lexicon contains English phrases from labels and abstracts. Scores are aggregated across
retrieved evidence with max-style pooling so that one strong passage can activate the relevant relation signal:
Ssupport (ci) =max
ej∈Essupport (ci,ej)(7)
Sconflict (ci) =max
ej∈Esconflict (ci,ej)(8)
Slimitation (ci) =max
ej∈Eslimitation (ci,ej).(9)
The scorer preserves the originating question_intent , so the selector can condition on both evidence relations and
question type.
4.4 Intent-Aware Risk-Calibrated Selection
The selector maps each scored claim to one of four statuses:certified, when evidence is strong enough for direct
inclusion;condition-limited, when the claim should be used only with caveats or partial wording;conflicting, when
4

the evidence is contradictory or too risky for direct certification; andomitted, when support is too weak. The final
response action is chosen from {full, partial, conflict, abstain} based on the selected claim set.
The mapping is intent-aware. A single global threshold is not adequate across medical question types: monitoring,
pregnancy, lactation, and special-population questions often require condition-limited answers even when some support
is present, whereas contraindication and interaction questions more often require certification or conflict handling. The
selector groups intents into coarse policy families—full-certify oriented, partial-oriented, conflict-oriented, mixed, and
dosage-specific cases—and applies different mappings from (Ssupport,Sconflict,Slimitation )to claim status. The ablation
compares this policy with a threshold-only selector that uses generic score thresholds without intent-specific rules.
The implementation also contains two scoped source-family priors. First, explicit dosage-instruction cues from
openFDA label evidence can rescue dosage support in the relation scorer. Second, PubMed Literature review-level evi-
dence can downgrade selected full answers to partial for specified claim families. These priors are fixed implementation
choices, not learned calibration parameters; the experiments therefore report source-conditioned majority controls and
source/evidence-novel slices to expose how much action behavior is recoverable from source and intent metadata.
5 Experiments
We evaluate certificate risk, retained utility, intent-aware action calibration, expressiveness beyond answer/abstain
baselines, and behavior under source/evidence novelty.
5.1 Data Provenance and Main Evaluation Split
We use a 2,223-item medical QA collection: 2,103 public-source examples (94.6%) and 120 synthetic examples
(5.4%). Main results use a real-source-only split with train n=1,470 , dev eval n=314 , and test eval n=319 ; the
synthetic subset contributes a separate n=20 stress slice. We also derive a source/evidence-novel holdout from primary
dev/test rows whose normalized source_url andevidence_text are both absent from train (dev n=82 , test n=78 ).
This slice evaluates source/evidence novelty, while the primary split remains the main protocol. A source-missing
counterfactual slice preserves questions and claim skeletons, removes evidence, and sets gold action to abstain ,
targeting deterministic retrieval-failure abstention under empty evidence. Tables 4, 5, and 15 summarize counts.
PubMedQA maps abstract-level yes/no/maybe research conclusions into full /conflict /partial , so it serves as
an abstract-style interface-transfer slice rather than a drug-label QA proxy.
5.2 Metrics
We report four metrics.UCCRmeasures the fraction of expressed critical claims that are unsupported under the
certificate definition.PAUis recall-style utility over retained gold-usable claims and is paired with PAU Precision, F1,
and risk–coverage curves to expose over-expression.F1uses sample-scoped claim matching because claim identifiers
are reused.Action Accuracyevaluates {full, partial, conflict, abstain} actions and is interpreted together with certificate
production and perturbation controls.
5.3 Weak-Label Certificate Protocol
Weak labels define a shared certificate protocol: each prediction records claim status, evidence identifiers, and relation
scores. UCCR and PAU are protocol measurements over expressed unsupported critical claims and retained usable
claims. Shortcut controls and risk–coverage curves are reported alongside them so that action behavior, certificate
production, and label-space priors are analyzed together. To calibrate this protocol against external review, we separately
re-audited a 100-item human-validation subset using official-source-first evidence review with preserved screenshots,
HTML snapshots, and text traces. The final audit labels contain 49 full_support , 49conditional_support , and
2conflict cases, agreeing with the weak labels on 73/100 items (0.7300; Cohen’s κ=0.5027 ). This subset is not
used for threshold tuning or benchmark replacement; it is reported only to bound how closely the weak-label certificate
protocol tracks pharmacist adjudication.
5

Table 2: Main ablation results on the primary real-source-only split. UCCR is defined by the weak-label certificate
protocol (Section 5.3), and PAU Prec is the fraction of expressed claims that are gold-usable.
Split Configuration UCCR PAU PAU Prec F1 Action Acc
Dev (n=314)Retrieval only 0.0860 1.0000 0.9522 0.9755 0.4968
Relation only 0.1943 1.0000 0.9522 0.9755 0.4968
Threshold-only selector 0.0000 0.9933 0.9581 0.9754 0.5223
Full risk-calibrated0.0000 1.0000 0.9901 0.9950 0.9204
Test (n=319)Retrieval only 0.0878 1.0000 0.9373 0.9676 0.5862
Relation only 0.2069 1.0000 0.9373 0.9676 0.5862
Threshold-only selector 0.0000 0.9732 0.9510 0.9620 0.5517
Full risk-calibrated0.0000 0.9967 0.9739 0.9851 0.8997
5.4 Evaluation Protocol and Label–Policy Separation
All numbers are computed against construction-derived weak labels. We report dev and test results on the pri-
mary real-source-only split; appendix material provides split details, commands, intervals, and risk–coverage anal-
ysis. The threshold-only baseline is tuned on dev_eval by a small grid search over support ,conflict , and
condition_limited ; among UCCR =0candidates, we maximize PAU, then Action Accuracy, then F1. The selected
support =0.35, conflict =0.55, and condition_limited =0.30 thresholds transfer unchanged to test. The full
selector is a fixed policy specification with intent-conditioned branches and global fallback thresholds, analyzed with
shortcut controls, threshold perturbations, and a policy-constant audit. The speech-act-guided proxy tunes one global
answer_support=0.34 gate on dev and transfers it unchanged to test.
At inference, the full selector predicts from the question, claim skeleton and intent, retrieved evidence, relation
scores, and documented source/context fields used by scoped source-family priors. Because source type, source-level
claim type, and intent can carry action priors, we report metadata-only majority controls fit on train; Table 9 lists
label-policy information access.
The analysis covers retrieval-only, threshold-only, shortcut, binary-form, NLI, learned-relation, and full claim-
selective rows. Binary-form baselines collapse the final action space to answer/abstain; NLI and learned-relation rows
keep the native {full, partial, conflict, abstain} interface, with the learned row swapping only the relation module.
6 Results
6.1 Main Ablation Results
We first report primary real-source-only results, then use controls and transfer slices to separate certificate production
from action-label priors. Table 2 isolates the transition from evidence access to relation scoring and then to action-level
policy.
Retrieval-only and relation-only rows express available claims without action selection, producing nonzero UCCR
(0.0860/0.0878 and 0.1943/0.2069 on dev/test) and low action accuracy. The dev-tuned threshold-only selector restores
UCCR to zero and reaches PAU 0.9933/0.9732 and F1 0.9754/0.9620, but action accuracy remains 0.5223/0.5517. The
full selector keeps UCCR =0.0000 while improving PAU, PAU Precision, F1, and action accuracy to 0.9204/0.8997.
Bootstrap intervals keep the full selector’s F1 and Action Accuracy gains over threshold-only positive on both splits;
risk–coverage analysis gives the same within-system pattern. The separate 100-item human re-audit shows that the
weak-label protocol is informative but not expert-equivalent: weak and audited labels agree on 73/100 items with
κ=0.5027 , and the disagreement mass is concentrated in full /conditional boundary cases rather than support-
versus-conflict reversals. Table 2 should therefore be read as a protocol-level certificate and action comparison, not as a
substitute for expert clinical adjudication.
Shortcut controls reveal a strong action-label prior. Intent-majority reaches 0.9013/0.8934 action accuracy without
evidence; source+intent and source+claim-type action-only rows reach 0.9299/0.9310 and 0.9331/0.9279, exceeding
the full selector on action accuracy alone. These rows do not select claims, assign evidence, expose relation scores,
or produce certificates, so UCCR and PAU are not applicable. Perturbations confirm that the full selector still uses
6

No intent
thresholdIntent
majoritySource+intent
majoritySource+claim
majorityEvidence
shuffledIntent
shuffledFull
selector0.00.20.40.60.81.0Action accuracy0.520.900.93 0.93
0.83
0.490.92
0.550.890.93 0.93
0.81
0.470.90Shortcut and Perturbation Controls
Selector Metadata only Perturbed CertificateFigure 2: Shortcut and perturbation controls on the primary split. Metadata-only majority rows are action-only controls
fit from training-set weak labels, and certificate metrics apply only to evidence-linked claim outputs.
UCCR PAU F1 AccVanilla
Binary
Verifier
Citation
SpeechAct Proxy
ClaimSel + NLI
ClaimSel + Learned
ClaimSel + Full0.21 0.79 0.98 0.50
0.00 0.78 0.85 0.41
0.00 0.77 0.85 0.39
0.21 0.79 0.98 0.50
0.00 0.97 0.97 0.48
0.00 0.86 0.92 0.75
0.00 0.98 0.98 0.88
0.00 1.00 1.00 0.92Dev
UCCR PAU F1 AccVanilla
Binary
Verifier
Citation
SpeechAct Proxy
ClaimSel + NLI
ClaimSel + Learned
ClaimSel + Full0.22 0.77 0.97 0.59
0.00 0.77 0.84 0.47
0.00 0.75 0.83 0.45
0.22 0.77 0.97 0.59
0.00 0.96 0.96 0.56
0.00 0.81 0.88 0.66
0.00 0.99 0.98 0.87
0.00 1.00 0.99 0.90T est
0.00.20.40.60.81.0
Metric valueBaseline Package Overview
Figure 3: Baseline operating map on the primary split. Binary-form baselines reduce unsupported-claim risk by
collapsing the action space, whereas claim-selective baselines retain the native action interface.
policy inputs: removing intent gives 0.5223/0.5517, shuffling intent gives 0.4936/0.4671, and shuffling evidence gives
0.8344/0.8088.
6.2 External-Form Baselines
The ablation table tests the within-system contribution chain. We add response-format baselines: direct answering,
binary answer/abstain behavior, a speech-act-guided answer/abstain proxy, a standard NLI-based semantic claim-
selective baseline, and a stronger learned-relation claim-selective variant. Full numeric rows are reported in Table 11.
Direct and citation-only answers incur large UCCR penalties. Binary answer/abstain and verifier-only rows drive
UCCR to zero but sacrifice PAU and action accuracy by collapsing partial andconflict . The speech-act-guided
proxy is stronger, reaching PAU 0.9732/0.9599 and F1 0.9652/0.9551 with UCCR zero, but action accuracy remains
0.4777/0.5611. The NLI-based claim-selective baseline improves action accuracy to 0.7484/0.6614. The strongest
external baseline swaps the relation module while retaining the same intent-aware selector and native claim-selective
action space; it reaches 0.8758/0.8652 action accuracy. A controlled relation-module comparison shows action-accuracy
changes of−0.0446 on dev and−0.0345 on test, placing the main contribution at the claim-selective action policy.
7

All Evidence-text
novelSource+evidence
novel0.00.20.40.60.81.0Action accuracy0.92
(n=314)
0.84
(n=140)0.78
(n=82)0.90
(n=319)0.84
(n=150)
0.77
(n=78)Source/Evidence-Novel Boundary
Dev
T estFigure 4: Source/evidence-novel boundary. The full selector keeps the certificate target as overlap constraints tighten,
but action accuracy drops on the strict source/evidence-novel slice.
6.3 Abstract-Style Transfer Behavior and Failure Modes
Source slices localize the remaining errors (Table 12). OpenFDA and FDA FAERS are stable; PubMed Literature is
mostly handled by the partial-answer policy. PubMedQA is the main abstract-style transfer slice: its yes/no/maybe
research judgments do not map cleanly ontofull/conflict/partial, and test action accuracy is 0.5161.
The primary split has no exact question overlap but substantial source reuse: train-to-dev/test source-URL over-
lap is 0.7389/0.7555 and evidence-text overlap is 0.5541/0.5298. We therefore materialize the strictest still-usable
source/evidence-novel holdout. The full selector keeps UCCR =0.0000 and PAU =1.0000 , but action accuracy drops
from 0.9204/0.8997 to 0.7805 on dev ( n=82 ) and 0.7692 on test ( n=78 ). It remains above unchanged threshold-only
(0.5854/0.5256) and learned-relation swap (0.6341/0.5769), while source-conditioned majority rows remain competitive
or stronger on action accuracy alone. Excluding PubMedQA, source/evidence-novel action accuracy is 0.9245 ( 49/53 )
on dev and 0.9362 ( 44/47 ) on test; the aggregate drop is concentrated in PubMedQA (0.5172/0.5161). A source-missing
counterfactual slice evaluates the abstain action under empty evidence. Both threshold-only and full selectors abstain
for all dev/test examples ( n=314/319 ), giving action accuracy 1.0000; retrieval-only has action accuracy 0.0000
and UCCR =1.0000 . The matching threshold-only result places this slice as action-space coverage evidence for
the fourth action. Claim-type and error slices in the appendix show the same pattern: remaining errors are mostly
full/partial/conflictboundary decisions under ambiguous evidence.
7 Discussion
The results support the claim-selective action interface. Under a shared certificate protocol, it calibrates primary
full ,partial , and conflict behavior while the source-missing counterfactual slice exercises abstain under empty
evidence. A dev-tuned threshold-only selector reaches the same UCCR target and abstains under empty evidence, but
remains much weaker on primary non-abstain action accuracy.
For high-risk medical QA, the design implication is that answer selection and claim selection should be evaluated
separately. A system may have enough evidence to state a dosing constraint, insufficient evidence for monitoring
advice, and contradictory evidence for a contraindication in the same retrieved packet. A document-level answer/abstain
interface collapses those states, whereas claim-selective certification records which claims are expressed, which evidence
supports them, and which response action follows from the resulting certificate.
Shortcut controls are part of the empirical result. Intent-majority reaches 0.9013/0.8934 action accuracy, and
source-conditioned action-only rows can exceed the full selector. The distinction is structural: metadata-only controls
predict an action label, while the proposed interface selects claims, assigns evidence, records relation scores, and
produces an auditable certificate. No-intent, intent-shuffled, and evidence-shuffled rows show sensitivity to policy
8

inputs.
This distinction matters for interpreting the primary numbers. Action accuracy alone would make the source-
conditioned majority rows a confound: they show that a large part of the weak-label action space is recoverable
from source and intent metadata. The controls expose that structure and keep the empirical claim at the level of
evidence-linked certification. The proposed interface is valuable because it attaches action decisions to claim-level
evidence states and makes the remaining dependence on metadata measurable.
The external baselines locate the contribution. Direct answering incurs unsupported-claim risk; binary abstention
reduces that risk by sacrificing utility; NLI and learned-relation claim selection are the strongest model-side comparisons.
PubMedQA and source/evidence-novel cases show that remaining errors are mainly full /partial /conflict decisions
under abstract-style evidence, rather than unsupported generation alone. The resulting picture is a measured same-
source-family protocol with explicit source/evidence-novel and abstract-style transfer boundaries.
The source/evidence-novel result separates certificate behavior from action generalization. The full selector keeps
UCCR at zero on the novelty slice, which means the expressed-claim certificate target remains satisfied under the weak-
label metric. At the same time, action accuracy drops to 0.7805/0.7692, and the drop is concentrated in PubMedQA.
The method preserves the certificate constraint on this derived boundary slice, while the intent/source policy still
struggles when abstract-level yes/no/maybe judgments must be mapped to the drug-QA action interface. The abstention
experiment completes action coverage: the source-missing counterfactual evaluates policy behavior after complete
evidence removal, selector-based systems deterministically enter the abstain branch, and retrieval-only behavior cannot.
The matching threshold-only result identifies the slice as coverage evidence for the fourth action. Overall, the empirical
pattern supports a methodological claim. Claim-selective certification gives a structured way to expose mixed-evidence
decisions, compare answer/abstain systems with native multi-action systems, and audit where source-family priors enter
the policy. The results characterize a same-source-family weak-label protocol with explicit failure-boundary slices;
deployment-oriented validation would require independently sampled source-disjoint data, expert adjudication, and
naturally occurring abstention cases.
8 Limitations
Several limitations follow from the evaluation design. Labels are construction-derived weak labels, and UCCR is
an expressed-claim certificate metric within this protocol; the 100-item human re-audit reaches 73/100 agreement
with κ=0.5027 , so the paper provides weak-label protocol evidence rather than clinical-safety validation. Shortcut
controls reveal source- and intent-conditioned action priors, and majority rows provide action-only comparisons
without certificate production. The primary split has no exact question duplicates but does reuse sources and evidence:
train-to-dev/test source-URL overlap is 0.7389/0.7555 and evidence-text overlap is 0.5541/0.5298. The derived
source/evidence-novel holdout drops action accuracy to 0.7805/0.7692, mainly from PubMedQA, and the source-
missing counterfactual tests only deterministic abstention under complete evidence removal. Finally, the selector is
a fixed policy with intent-conditioned constants and scoped source-family priors, and template decomposition plus
cue-based relation scoring still leave fine-grained multi-claim decomposition, implicit contraindications, semantic
disagreement, and PubMedQA-style abstract judgments as open extensions.
9 Conclusion
We study claim-selective certification as an auditable alternative to answer-or-abstain behavior for mixed-evidence med-
ical QA. On the real-source-only protocol, the full system records UCCR =0.0000 on dev/test, PAU =1.0000/0.9967 ,
PAU Precision =0.9901/0.9739 , and action accuracy =0.9204/0.8997 ; shortcut controls and novelty slices define both
the value and the boundary of this weak-label protocol. Rather than claiming source-disjoint clinical generalization, the
paper isolates a reproducible certification interface, the shortcut structure of its weak-label action space, and the transfer
boundary exposed by abstract-style evidence.
References
[1]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in Neural Information Processing Systems, 33:9459–9474, 2020.
9

[2]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang.
Retrieval-augmented generation for large language models: A survey.arXiv preprint arXiv:2312.10997, 2023.
[3]Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for open domain question
answering. InProceedings of the 16th Conference of the European Chapter of the Association for Computational
Linguistics: Main Volume, pages 874–880. Association for Computational Linguistics, 2021.
[4]Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-
tau Yih. Dense passage retrieval for open-domain question answering. InProceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing, pages 6769–6781. Association for Computational Linguistics,
2020.
[5]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve,
generate, and critique through self-reflection.arXiv preprint arXiv:2310.11511, 2023.
[6]Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. Query rewriting in retrieval-augmented large
language models. InProceedings of the 2023 Conference on Empirical Methods in Natural Language Processing,
pages 13679–13690. Association for Computational Linguistics, 2023.
[7]Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike Lewis. Measuring and narrowing
the compositionality gap in language models. InFindings of the Association for Computational Linguistics:
EMNLP 2023, pages 5664–5687. Association for Computational Linguistics, 2023.
[8]Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. InAdvances in neural
information processing systems, pages 4878–4887, 2017.
[9]Amita Kamath, Robin Jia, and Percy Liang. Selective question answering under domain shift. InProceedings of
the 58th Annual Meeting of the Association for Computational Linguistics, pages 5684–5696. Association for
Computational Linguistics, 2020.
[10] Wenhu Chen, Xinyi Wang, and William Yang Wang. A dataset for answering time-sensitive questions.arXiv
preprint arXiv:2108.06314, 2021.
[11] Carel van Niekerk, Michael Heck, Christian Geishauser, Hsien-chin Lin, Nurul Lubis, Marco Moresi, and Milica
Gasic. Knowing what you know: Calibrating dialogue belief state distributions via ensembles. InFindings of
the Association for Computational Linguistics: EMNLP 2020, pages 3096–3102. Association for Computational
Linguistics, 2020.
[12] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty
estimation in natural language generation. InInternational Conference on Learning Representations, 2023.
[13] Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. On faithfulness and factuality in abstractive
summarization. InProceedings of the 58th Annual Meeting of the Association for Computational Linguistics,
pages 1906–1919. Association for Computational Linguistics, 2020.
[14] Nouha Dziri, Ehsan Kamalloo, Sivan Milton, Osmar Zaiane, Mo Yu, Edoardo Maria Ponti, and Siva Reddy. Faith-
dial: A faithful benchmark for information-seeking dialogue.Transactions of the Association for Computational
Linguistics, 10:1473–1490, 2022.
[15] Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Lora Aroyo, Michael Collins, Dipanjan Das, Slav Petrov,
Gaurav Singh Tomar, Iulia Turc, and David Reitter. Measuring attribution in natural language generation models.
Computational Linguistics, pages 1–64, 2023.
[16] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. Fever: a large-scale dataset for
fact extraction and verification. InProceedings of the 2018 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies, pages 809–819, 2018.
[17] Tal Schuster, Adam Fisch, and Regina Barzilay. Get your vitamin c! robust fact verification with contrastive
evidence. InProceedings of the 2021 Conference of the North American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies, pages 624–640. Association for Computational Linguistics,
2021.
10

[18] Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language models to generate text with
citations. InProceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages
6465–6488. Association for Computational Linguistics, 2023.
[19] Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer,
and Hannaneh Hajishirzi. Factscore: Fine-grained atomic evaluation of factual precision in long form text
generation. InProceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages
12076–12100, 2023.
[20] Cosmo Du, Nathan Hu, Da Huang, Jie Huang, Quoc Le, Ruibo Liu, Yifeng Lu, Daiyi Peng, Xinying Song, Dustin
Tran, Jerry Wei, and Chengrun Yang. Long-form factuality in large language models. InAdvances in Neural
Information Processing Systems 37, pages 80756–80827, 2024.
[21] Liyan Tang, Philippe Laban, and Greg Durrett. Minicheck: Efficient fact-checking of llms on grounding documents.
InProceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 8818–8847,
2024.
[22] Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, Samuel R. Bowman, and Noah A. Smith.
Annotation artifacts in natural language inference data. InProceedings of the 2018 Conference of the North
American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages
107–112. Association for Computational Linguistics, 2018.
[23] R. Thomas McCoy, Ellie Pavlick, and Tal Linzen. Right for the wrong reasons: Diagnosing syntactic heuristics
in natural language inference. InProceedings of the 57th Annual Meeting of the Association for Computational
Linguistics, pages 3428–3448. Association for Computational Linguistics, 2019.
[24] Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. Beyond accuracy: Behavioral testing
of NLP models with CheckList. InProceedings of the 58th Annual Meeting of the Association for Computational
Linguistics, pages 4902–4912. Association for Computational Linguistics, 2020.
[25] Swabha Swayamdipta, Roy Schwartz, Nicholas Lourie, Yizhong Wang, Hannaneh Hajishirzi, Noah A. Smith, and
Yejin Choi. Dataset cartography: Mapping and diagnosing datasets with training dynamics. InProceedings of
the 2020 Conference on Empirical Methods in Natural Language Processing, pages 9275–9293. Association for
Computational Linguistics, 2020.
[26] Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani,
Weihua Hu, Michihiro Yasunaga, Richard L. Phillips, Irena Gao, Tony Lee, Etienne David, Ian Stavness, Wei
Guo, Brian Earnshaw, Imran Haque, Sara Beery, Jure Leskovec, Anshul Kundaje, Emma Pierson, Sergey Levine,
Chelsea Finn, and Percy Liang. WILDS: A benchmark of in-the-wild distribution shifts. InProceedings of the
38th International Conference on Machine Learning, pages 5637–5664. PMLR, 2021.
[27] Asma Ben Abacha, Chaitanya Shivade, and Dina Demner-Fushman. Overview of the mediqa 2019 shared task on
textual inference, question entailment and question answering. InProceedings of the 18th BioNLP Workshop and
Shared Task, pages 370–379, 2019.
[28] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. Pubmedqa: A dataset for biomedical
research question answering.arXiv preprint arXiv:1909.06146, 2019.
[29] Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does
this patient have? a large-scale open domain question answering dataset from medical exams.Applied Sciences,
11(14):6421, 2021.
[30] Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Medmcqa: A large-scale multi-subject
multi-choice dataset for medical domain question answering.arXiv preprint arXiv:2203.14371, 2022.
[31] Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay
Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. Large language models encode clinical knowledge.Nature,
620(7972):172–180, 2023.
11

[32] Harsha Nori, Nicholas King, Scott Mayer McKinney, Dean Carignan, and Eric Horvitz. Capabilities of gpt-4 on
medical challenge problems.arXiv preprint arXiv:2303.13375, 2023.
A Appendix Overview
This appendix reports the claim summary, additional diagnostics, uncertainty estimates, reproducibility commands, and
asset information used to support the main paper.
Algorithm 1Schematic claim-selective action interface. The procedure summarizes the fixed decision flow from
retrieved evidence to the final action.
1:Input:questionq, retrieved evidenceE
2:Emit claim skeletonsCand question intent.
3:forclaimc i∈Cdo
4:Score support, conflict, and limitation against eache j∈E.
5:Aggregate scores into(S sup,Sconf,Slim).
6:Map relation scores and intent to a status in {certified, condition-limited, conflicting, omitted}.
7:end for
8:Build an evidence-linked certificate for expressed claims.
9:ifno claim is expressiblethen
10:returnabstain
11:else ifmaterial conflict is selectedthen
12:returnconflict
13:else ifall required critical claims are directly certifiedthen
14:returnfull
15:else
16:returnpartial
17:end if
B Claim Summary
Table 3: Summary of the main claims and their evaluation level.
Claim Evaluation level
Claim-selective RAG defines an action interface over certified, condition-limited, conflicting, and
omitted claimsmethod formulation
Intent-aware selection improves action behavior over global thresholding controlled weak-label dev/test comparison
Binary answer/abstain baselines are less expressive for mixed evidence external-form baseline comparison
UCCR=0 is achieved by the full selector on the primary split certificate-level metric on expressed critical claims
Shortcut controls expose source- and intent-conditioned action priors metadata and perturbation diagnostics
C Primary Split and Protocol Tables
Table 4: Primary real-source-only split used for the main results.
Split Size Removed synthetic rows
Train 1,470 84
Dev eval 314 18
Test eval 319 18
12

Table 5: Gold action coverage and abstention scope. The primary dev/test protocol evaluates the non-abstain actions,
and source-missing rows exercise abstention under counterfactual retrieval failure.
Evaluation set n full partial conflict abstain Role
Primary dev eval 314 156 143 15 0 main protocol
Primary test eval 319 187 112 20 0 main protocol test split
Source-missing dev stress 314 0 0 0 314 abstain coverage
Source-missing test stress 319 0 0 0 319 abstain coverage
D Human Re-audit of Weak Labels
To estimate how closely the weak-label certificate protocol tracks pharmacist adjudication, we separately re-audited a
100-item human-validation subset using official-source-first evidence review. The audit proceeded through an initial
pass, a second-pass whole-set adjudication, and an item-by-item third-pass web re-audit of all remaining disagreement
cases, with screenshots, HTML snapshots, and text snapshots preserved under data/annotation/third_pass_web_
reaudit/ . The final audited labels were not used to tune thresholds or replace the primary benchmark; they are
reported only as a calibration layer for interpreting the weak-label protocol.
Table 6: Summary of the 100-item human re-audit. Agreement is measured against the original weak labels.
Quantity Value
Audited subset size 100
Finalfull_support49
Finalconditional_support49
Finalconflict2
Weak-label agreement 73 / 100 = 0.7300
Cohen’sκ0.5027
Table 7: Disagreement structure in the 100-item human re-audit. Rows count weak-label →audited-label transitions
among the 27 disagreement items.
Weak→audited transition Count
conditional_support→full_support11
full_support→conditional_support11
conflict→conditional_support3
conflict→full_support2
Most disagreements are boundary cases between full_support andconditional_support , not reversals
between support and conflict. The third-pass item-by-item web re-audit made no further changes to the final audited
set. We therefore use this subset to calibrate scope, not to relabel the main benchmark: it supports the claim that the
protocol carries nontrivial medical signal while still falling short of expert-adjudicated clinical gold data.
The source-missing counterfactual slice completes action-interface coverage. It preserves the original primary
dev/test questions and claim skeletons, sets evidence_pool=[] and clears evidence_text , converts all gold claims
toomitted , and sets the gold action to abstain . Table 8 reports the corresponding action behavior. Because every
stress item has empty evidence, these rows characterize deterministic abstention under retrieval failure; semantic
evidence insufficiency, natural abstention frequency, and expert-labeled correctness require separate evaluation data.
13

Table 8: Source-missing abstention results. Selector rows abstain under empty evidence, whereas retrieval/relation-only
rows express claims without evidence and fail the action check.
Split Configuration UCCR Action Acc Gold abstain
Dev (n=314)Retrieval only 1.0000 0.0000 314
Relation only 1.0000 0.0000 314
Threshold-only selector 0.0000 1.0000 314
Full risk-calibrated0.0000 1.0000314
Test (n=319)Retrieval only 1.0000 0.0000 319
Relation only 1.0000 0.0000 319
Threshold-only selector 0.0000 1.0000 319
Full risk-calibrated0.0000 1.0000319
Table 9: Information available to weak-label construction and model-side policies.
Component Weak labels Selector Note
Question text yes yes shared task input
Retrieved evidence yes yes shared RAG setting
Weak label / gold action yes no evaluation target only
Question intent claim skeleton yes tested by intent controls
Source type / root claim type label construction scoped priors tested by source-conditioned controls
Relation scores no yes tested by no-intent and shuffled controls
Dev labels tuning tuning controls transferred to test
Test labels evaluation no reporting only
Table 10: Shortcut controls on the primary split. Majority rows are action-only controls fit from training-set weak labels,
and certificate metrics apply only to rows with evidence-linked claim outputs.
Control Dev Action Acc Test Action Acc
No-intent threshold selector 0.5223 0.5517
Train intent-majority only 0.9013 0.8934
Train source+intent majority only 0.9299 0.9310
Train source+claim-type majority only 0.9331 0.9279
Evidence-shuffled full selector 0.8344 0.8088
Intent-shuffled full selector 0.4936 0.4671
Full intent-aware selector 0.9204 0.8997
14

Table 11: Baseline package on the primary split. Binary-form baselines use the mapping answer→full and
abstain→abstain for action accuracy, whereas claim-selective rows retain the native {full, partial, conflict, abstain}
action space.
Split Method UCCR PAU F1 Action Acc
Dev (n=314)vanilla answer 0.2070 0.7860 0.9755 0.4968
binary answer / abstain 0.0000 0.7793 0.8535 0.4076
verifier only 0.0000 0.7659 0.8450 0.3949
citation only 0.2070 0.7860 0.9755 0.4968
speech-act-guided proxy 0.0000 0.9732 0.9652 0.4777
claim-selective + NLI relation 0.0000 0.8595 0.9179 0.7484
claim-selective + learned relation 0.0000 0.9766 0.9815 0.8758
claim-selective full0.0000 1.0000 0.9950 0.9204
Test (n=319)vanilla answer 0.2194 0.7692 0.9676 0.5862
binary answer / abstain 0.0000 0.7659 0.8373 0.4671
verifier only 0.0000 0.7525 0.8287 0.4545
citation only 0.2194 0.7692 0.9676 0.5862
speech-act-guided proxy 0.0000 0.9599 0.9551 0.5611
claim-selective + NLI relation 0.0000 0.8094 0.8768 0.6614
claim-selective + learned relation 0.0000 0.9900 0.9801 0.8652
claim-selective full0.0000 0.9967 0.9851 0.8997
E Extended Diagnostics
This section expands the main source- and claim-type analyses for the full system. The tables and figures below report
diagnostic slices under the same weak-label protocol as the main results.
Table 12: Source-level diagnostics for the full system on the primary split.
SourceDev Test
n PAU Action Acc n PAU Action Acc
OpenFDA 208 1.0000 0.9663 209 0.9949 0.9330
FDA FAERS 40 1.0000 1.0000 41 1.0000 1.0000
PubMed Literature 37 1.0000 0.8919 38 1.0000 0.9211
PubMedQA 29 1.0000 0.5172 31 1.0000 0.5161
15

OpenFDA FDA FAERS PubMed Lit. PubMedQA0.00.20.40.60.81.01.00 1.00 1.00 1.00 0.99 1.00 1.00 1.00PAU by source
Dev
T est
OpenFDA FDA FAERS PubMed Lit. PubMedQA0.00.20.40.60.81.0 0.971.00
0.89
0.520.931.00
0.92
0.52Action Accuracy by sourceSource-Level DiagnosticsFigure 5: Extended source-level diagnostics for the full system. OpenFDA has the highest action accuracy, whereas
PubMedQA has the lowest.
Table 13: Hard claim-type diagnostics for the full system. Slices withn<20 are reported descriptively.
Claim typeDev Test
n PAU Action Acc n PAU Action Acc
Indication 29 1.0000 0.5172 31 1.0000 0.5161
Pregnancy 11 1.0000 0.7273 14 0.9231 0.5714
Lactation 5 1.0000 0.8000 7 1.0000 0.7143
Interaction 19 1.0000 0.8421 17 1.0000 0.7647
Dosage adjustment 11 1.0000 0.9091 11 1.0000 0.8182
Dosage 6 1.0000 1.0000 12 1.0000 0.9167
Special population 8 1.0000 0.7500 9 1.0000 1.0000
16

full
partial
conflict
abstainGold action0.94
(103)0.06
(6)0.00
(0)0.00
(0)
0.00
(0)1.00
(88)0.00
(0)0.00
(0)
0.00
(0)0.09
(1)0.91
(10)0.00
(0)
0.00
(0)0.00
(0)0.00
(0)0.00
(0)Dev: OpenFDA
0.47
(8)0.53
(9)0.00
(0)0.00
(0)
0.38
(3)0.62
(5)0.00
(0)0.00
(0)
0.50
(2)0.00
(0)0.50
(2)0.00
(0)
0.00
(0)0.00
(0)0.00
(0)0.00
(0)Dev: PubMedQA
fullpartial conflict abstain
Predicted actionfull
partial
conflict
abstainGold action0.92
(119)0.08
(10)0.01
(1)0.00
(0)
0.01
(1)0.99
(67)0.00
(0)0.00
(0)
0.00
(0)0.18
(2)0.82
(9)0.00
(0)
0.00
(0)0.00
(0)0.00
(0)0.00
(0)T est: OpenFDA
fullpartial conflict abstain
Predicted action0.60
(12)0.40
(8)0.00
(0)0.00
(0)
0.50
(1)0.50
(1)0.00
(0)0.00
(0)
0.11
(1)0.56
(5)0.33
(3)0.00
(0)
0.00
(0)0.00
(0)0.00
(0)0.00
(0)T est: PubMedQA
0.00.20.40.60.81.0
Row-normalized shareAction Confusion on Strongest and Weakest SourcesFigure 6: Action confusion on the strongest and weakest sources. PubMedQA errors are dominated by full /partial
boundary mistakes rather than unsupported generation.
F Source-Overlap Analysis
The primary split removes exact question duplicates and retains source-family overlap. We measure exact overlap over
question text, source URL, and evidence text. Table 14 reports the main analysis. Train-to-dev/test question overlap is
zero, while source-URL overlap is substantial: 232/314 dev examples and 241/319 test examples share a source URL
with training. Evidence-text overlap is also nontrivial at 174/314 and 169/319. The source pattern is uneven: OpenFDA
examples have 100% train source-URL overlap in both dev and test, FDA FAERS has partial overlap, and PubMed
Literature/PubMedQA have no exact train source-URL overlap.
Table 14: Source-overlap analysis for the primary split. Novelty-slice metrics use the unchanged full selector, and the
source/evidence-novel holdout requires both source URL and evidence text to be absent from train.
Diagnostic Split / key n Overlap Rate Action Acc
Exact overlap train→dev question 314 0 0.0000 –
Exact overlap train→dev source URL 314 232 0.7389 –
Exact overlap train→dev evidence text 314 174 0.5541 –
Exact overlap train→test question 319 0 0.0000 –
Exact overlap train→test source URL 319 241 0.7555 –
Exact overlap train→test evidence text 319 169 0.5298 –
Full selector dev all 314 – – 0.9204
Full selector dev source/evidence-novel 82 – – 0.7805
Full selector test all 319 – – 0.8997
Full selector test source/evidence-novel 78 – – 0.7692
17

The source/evidence-novel holdout is smaller and distribution-shifted. Its lower action accuracy characterizes the
primary split as a controlled same-source-family evaluation with a separate source/evidence-novel transfer slice. On
this boundary slice, the full selector records UCCR =0.0000 under the weak-label certificate metric, with PAU/F1 of
1.0000/0.9873 on dev and 1.0000/0.9583 on test. Tables 15 and 16 provide the reproducible slice definition and the
corresponding comparison rows.
Table 15: Source/evidence-novel holdout construction. Rows are selected from the primary dev/test evaluation files
when both normalizedsource_urlandevidence_textare absent from train.
Split Distribution n full partial conflict
Dev all source/evidence-novel 82 32 46 4
Dev PubMed Literature / PubMedQA / FDA FAERS 82 37 / 29 / 16
Test all source/evidence-novel 78 31 38 9
Test PubMed Literature / PubMedQA / FDA FAERS 78 38 / 31 / 9
Table 16: Source/evidence-novel holdout results. Selector and baseline rows use fixed primary-split settings with no
threshold retuning, and majority rows are action-only controls that do not produce certificates.
Split Configuration UCCR PAU F1 Action Acc Role
Dev (n=82)Threshold-only selector 0.0000 1.0000 0.9873 0.5854 fixed selector
Full risk-calibrated0.0000 1.0000 0.98730.7805 certificate
Learned relation + selector 0.0000 0.9103 0.9342 0.6341 module swap
NLI relation + selector 0.0000 0.7949 0.8671 0.5488 NLI baseline
Source+intent majority – – – 0.8293 action-only
Source+claim-type majority – – – 0.8415 action-only
Evidence-shuffled full 0.0000 1.0000 0.9750 0.7073 perturbation
Test (n=78)Threshold-only selector 0.0000 0.9420 0.9420 0.5256 fixed selector
Full risk-calibrated0.0000 1.0000 0.95830.7692 certificate
Learned relation + selector 0.0000 0.9565 0.9296 0.5769 module swap
NLI relation + selector 0.0000 0.8551 0.8613 0.4872 NLI baseline
Source+intent majority – – – 0.8333 action-only
Source+claim-type majority – – – 0.8205 action-only
Evidence-shuffled full 0.0000 0.9710 0.9241 0.6282 perturbation
Table 17: Source/evidence-novel full-selector accuracy by source. The non-PubMedQA aggregate shows that most of
the action drop is localized to the abstract-style PubMedQA transfer slice.
Source groupDev Test
n Action Acc n Action Acc
FDA FAERS 16 1.0000 9 1.0000
PubMed Literature 37 0.8919 38 0.9211
PubMedQA 29 0.5172 31 0.5161
Non-PubMedQA aggregate 53 0.9245 47 0.9362
On the source/evidence-novel holdout, the full selector makes 18 action errors on dev and 18 on test. The source-
specific pattern is concentrated: FDA FAERS remains at 1.0000 action accuracy on both slices, PubMed Literature
remains high (0.8919/0.9211), and PubMedQA remains low (0.5172/0.5161). Excluding PubMedQA, source/evidence-
novel action accuracy is 0.9245/0.9362. This supports interpreting the holdout as an abstract-style and source-shift
boundary rather than a broad failure of the certificate metric.
G Statistical Uncertainty
We computed nonparametric bootstrap intervals on the primary real-source-only dev and test splits. Table 18 reports
point estimates and 95% intervals for threshold-only selection and the full intent-aware selector.
18

The threshold-only comparison uses thedev-selectedoperating point described in Section 5. A grid search
over support ,conflict , and condition_limited thresholds selects the candidate with UCCR =0and then max-
imizes PAU, action accuracy, and F1 in that order. The selected setting is support =0.35, conflict =0.55, and
condition_limited =0.30, transferred unchanged to test. The speech-act-guided answer/abstain proxy is tuned
separately on dev by searching a small set of global answer_support gates while keeping its profile-specific retrieval
logic fixed; this selectsanswer_support=0.34.
Table 18: Bootstrap 95% intervals for the main selector comparison. Intervals are estimated from 1,000 sample-level
bootstrap resamples.
Split Method UCCR PAU F1 Action Acc
Dev threshold-only 0.0000 [0.0000, 0.0000] 0.9933 [0.9832, 1.0000] 0.9754 [0.9633, 0.9870] 0.5223 [0.4682, 0.5732]
Dev full selector 0.0000 [0.0000, 0.0000] 1.0000 [1.0000, 1.0000] 0.9950 [0.9884, 1.0000] 0.9204 [0.8854, 0.9490]
Test threshold-only 0.0000 [0.0000, 0.0000] 0.9732 [0.9529, 0.9900] 0.9620 [0.9456, 0.9756] 0.5517 [0.4953, 0.6050]
Test full selector 0.0000 [0.0000, 0.0000] 0.9967 [0.9870, 1.0000] 0.9851 [0.9749, 0.9935] 0.8997 [0.8652, 0.9310]
The paired bootstrap deltas localize the gain. On dev, the full selector improves PAU by 0.0067 with a 95% interval
of [0.0000, 0.0168], F1 by 0.0196 [0.0097, 0.0299], and action accuracy by 0.3981 [0.3408, 0.4522]. On test, the
corresponding deltas are 0.0234 [0.0068, 0.0435] for PAU, 0.0231 [0.0099, 0.0381] for F1, and 0.3480 [0.2915, 0.4075]
for action accuracy.
For UCCR, bootstrap intervals degenerate at zero because no unsupported expressed critical claim is observed
under the reported weak-label certificate metric on the primary split. Interpreting the dev/test expressed critical claims
as Bernoulli trials gives 0 observed events out of 302 expressed critical claims on dev and 0 out of 306 on test. The
corresponding upper ends of the two-sided 95% Wilson score intervals are 0.0126 and 0.0124.
H PAU Precision
PAU is a recall-style utility metric: it measures how many gold usable claims are retained. To make the over-expression
boundary explicit, we also compute a precision counterpart, |gold usable∩pred usable|/|pred usable| . This precision
counterpart distinguishes retained useful claims from over-expression.
Table 19: PAU precision on the primary split. Higher values indicate fewer extra usable predictions beyond the
weak-label usable set.
Method Dev Test
Retrieval only 0.9522 0.9373
Threshold-only selector 0.9581 0.9510
Full selector 0.9901 0.9739
Retrieval-only keeps PAU at 1.0000 because it expresses all gold usable claims, but its PAU precision is lower
because it also expresses claims outside the weak-label usable set. The full selector has the strongest PAU precision
among these rows, indicating that its high PAU is not obtained by broad over-expression.
I Selector Threshold Sensitivity
We run one-at-a-time perturbations around the reported selector operating points. For the tuned threshold-only selector,
we perturb support ,conflict ,condition_limited , and limitation by±0.05 from the dev-selected point. For
the full selector, we apply the same perturbation to its global fallback thresholds while leaving the intent-conditioned
policy branches fixed.
19

Table 20: Selector threshold sensitivity. All rows have UCCR =0.0000 under the reported weak-label certificate metric,
and the full selector is unchanged because the high-impact decisions are governed by intent-conditioned policy branches.
Split Selector variant PAU F1 Action Acc
Dev threshold-only base 0.9933 0.9754 0.5223
Dev threshold-only,condition_limited+0.05 0.8161 0.8777 0.4076
Dev full selector base 1.0000 0.9950 0.9204
Dev full selector, any one global threshold±0.05 1.0000 0.9950 0.9204
Test threshold-only base 0.9732 0.9620 0.5517
Test threshold-only,condition_limited+0.05 0.8094 0.8721 0.4671
Test full selector base 0.9967 0.9851 0.8997
Test full selector, any one global threshold±0.05 0.9967 0.9851 0.8997
The sensitivity analysis matches the main result table. Threshold-only behavior can keep UCCR at zero, but its
utility and action behavior are brittle to a modest increase in the condition-limited gate. The full selector is dominated
by intent-conditioned branches, so these global fallback-threshold perturbations leave its dev/test metrics unchanged.
The result localizes the reported decisions to the intent-conditioned policy branches rather than to the global fallback
gates.
J Selector Policy Specification
The full selector is a fixed policy specification with intent-conditioned branches and global fallback thresholds.
The repository includes a policy-constant audit at scripts/audit_selector_policy_constants.py forsrc/
selection/selector.py . The audit covers 445 lines across the fallback, intent-aware, and final classification
functions and finds 141 numeric constants in those functions. These constants define the implemented policy for
the current relation-score distribution; shortcut controls, source slices, and threshold perturbations characterize the
sensitivity of that policy.
We additionally run a branch-family sensitivity audit at scripts/analyze_selector_branch_sensitivity.
py. This audit perturbs the support, conflict, or limitation score entering one intent branch family at a time by ±0.05
and±0.10 to characterize robustness of the fixed policy. The largest overall action-accuracy drops are −0.0382 on dev
and−0.0408 on test, both under a −0.10 support perturbation to the indication branch. Slice-level changes are larger in
small or boundary-heavy families, including interaction on dev and dosage-adjustment and research-question slices on
test. In the largest-change rows, UCCR remains 0.0000 under the certificate metric. This audit localizes sensitivity to
specific intent-conditioned branches rather than to the global fallback gates.
K Selective-Prediction View
We also report a risk–coverage view at the critical-claim level. Each critical claim is treated as a selectable item.
Coverage is selected critical claims divided by total gold critical claims, and risk is selected critical claims not marked
supportable by the gold weak labels divided by selected critical claims. This view is supplementary because it ignores
conflict disclosure and certificate semantics.
For the threshold-only selector, we sweep the support threshold and tie the condition-limited threshold to support
- 0.10 while keeping the conflict thresholds fixed. We also plot the dev-tuned threshold-only operating point
(support =0.35, conflict =0.55, condition_limited =0.30). The full selector, learned claim-selective baseline,
and NLI baseline are reported as fixed operating points. On dev, the tuned threshold-only point reaches coverage
0.9873 and risk 0.0419, whereas the full selector reaches coverage 0.9618 and risk 0.0099. On test, the corresponding
comparison is 0.9592 / 0.0490 for the tuned threshold-only point versus 0.9592 / 0.0261 for the full selector. The
learned claim-selective baseline remains close but is slightly lower-coverage and higher-risk than the full selector on
both splits (0.9427/0.9561 coverage and 0.0135/0.0295 risk).
20

0.0 0.2 0.4 0.6 0.8 1.0
Coverage0.000.020.040.060.080.100.12Risk
Dev
0.0 0.2 0.4 0.6 0.8 1.0
Coverage0.000.020.040.060.080.100.12Risk
T estThreshold-only sweep Threshold default Threshold tuned Full Learned NLIFigure 7: Selective-prediction view on the primary split. The threshold-only selector traces a tunable risk–coverage
family, and the full selector keeps similar coverage but lower risk than the dev-tuned threshold-only operating point on
both splits.
L Compute Resources
Experiments were reproduced on a Windows workstation with an Intel Core Ultra 9 185H CPU (16 physical cores /
22 logical processors), approximately 33.95 GB of physical memory, Intel Arc integrated graphics, and an NVIDIA
GeForce RTX 4060 Laptop GPU. The main runtime costs come from relation-model loading and repeated baseline
sweeps rather than large-scale training.
Representative wall-clock times are:
• dev ablation rerun: 110.93 seconds
• baseline package rerun on the primary real-source-only split: 321.94 seconds
M Existing Assets and Terms
The reported evaluation chain depends on the following external assets.
•openFDA / FAERS APIs.The official openFDA terms state that, unless otherwise noted, content and data are
generally unrestricted and made available under a CC0 1.0 dedication, while also warning that some third-party
content may be separately marked and that API use is subject to service limits and terms.1
•DailyMed.DailyMed is provided by the U.S. National Library of Medicine as the official public source of FDA
label information, but the site also notes that NLM does not review SPL content before publication and that the
“in use” labeling may differ from the most recent FDA-approved labeling.2
•PubMed / NLM literature content.NLM’s own copyright guidance states that some NLM data are U.S.
government works while abstracts and other contributed materials may still be protected by copyright, leaving
downstream users responsible for respecting those restrictions when redistributing content.3
•PubMedQA.The official PubMedQA repository distributes the benchmark under the MIT license and specifies
the dataset download and evaluation process.4
•NLI baseline model.Thefacebook/bart-large-mnlimodel card lists the model under the MIT license.5
The evaluation distinguishes between public-domain or permissive API access, NLM-hosted content that may still
contain copyrighted abstracts, and benchmark/model assets distributed under repository-specific licenses.
1https://open.fda.gov/terms
2https://dailymed.nlm.nih.gov/dailymed/about-dailymed.cfm
3https://www.nlm.nih.gov/databases/download.html
4https://github.com/pubmedqa/pubmedqa
5https://huggingface.co/facebook/bart-large-mnli
21

N Reproducibility Note
The experiments use a single real-source-only split family: data/splits/primary_real_source/ , which contains
train.jsonl ,dev_eval.jsonl , and test_eval.jsonl . The main reported results use only the dev_eval.jsonl
andtest_eval.jsonlfiles from this real-source-only split.
Main ablation reruns.The main ablation table can be regenerated with:
python scripts/run_ablation_study.py \
data/splits/primary_real_source/dev_eval.jsonl \
outputs/regression_checks/<dev_run>
python scripts/run_ablation_study.py \
data/splits/primary_real_source/test_eval.jsonl \
outputs/regression_checks/<test_run>
Baseline package reruns.The external-form baseline package can be regenerated with:
python scripts/run_current_baseline_package.py \
--split-dir data/splits/primary_real_source \
--output-dir outputs/baselines/current_clean_mainline
The speech-act-guided proxy tuning run can be regenerated with:
python scripts/tune_pragaura_proxy_baseline.py
Slice diagnostics and consistency checks.Source-level and claim-type diagnostics are regenerated from the same
split family, and the repository includes diagnostic scripts for the data and gapfill state:
python scripts/analyze_eval_slices.py --input <split_jsonl> ...
python scripts/analyze_shortcut_controls.py
python scripts/audit_source_overlap.py
python scripts/audit_selector_policy_constants.py
python scripts/audit_data_experiment_state.py
python scripts/audit_gapfill_state.py
Reproducibility package.The commands above rerun the reported implementation inside the project repository
and define the split files, frozen outputs, and regeneration commands used by the experiments. The anonymous
supplemental package is a curated whitelist of the scripts, split files, stress slices, canonical outputs, and diagnostics
needed to inspect the paper’s claims; non-primary workflow holdouts, unfinished manual-audit preparation folders,
and historical diagnostic snapshots are intentionally excluded. The package also includes a Croissant metadata
file,artifact/croissant_metadata.json , with core dataset fields and Responsible AI notes for data collection,
weak-label annotation, known biases, limitations, and intended research use.
O Synthetic Stress Slice
The synthetic stress slice is reported separately from the main result table. On the 20-example synthetic stress set, the
full system obtains UCCR =0.0000 , PAU =0.0000 , F1=0.0000 , and action accuracy =0.6500 . The zero PAU and
F1 values indicate a label-interface mismatch, so the slice functions as a stress test while the real-source-only chain
remains the primary result source.
22
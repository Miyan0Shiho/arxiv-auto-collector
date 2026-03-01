# Case-Aware LLM-as-a-Judge Evaluation for Enterprise-Scale RAG Systems

**Authors**: Mukul Chhabra, Luigi Medrano, Arush Verma

**Published**: 2026-02-23 21:37:06

**PDF URL**: [https://arxiv.org/pdf/2602.20379v1](https://arxiv.org/pdf/2602.20379v1)

## Abstract
Enterprise Retrieval-Augmented Generation (RAG) assistants operate in multi-turn, case-based workflows such as technical support and IT operations, where evaluation must reflect operational constraints, structured identifiers (e.g., error codes, versions), and resolution workflows. Existing RAG evaluation frameworks are primarily designed for benchmark-style or single-turn settings and often fail to capture enterprise-specific failure modes such as case misidentification, workflow misalignment, and partial resolution across turns.
  We present a case-aware LLM-as-a-Judge evaluation framework for enterprise multi-turn RAG systems. The framework evaluates each turn using eight operationally grounded metrics that separate retrieval quality, grounding fidelity, answer utility, precision integrity, and case/workflow alignment. A severity-aware scoring protocol reduces score inflation and improves diagnostic clarity across heterogeneous enterprise cases. The system uses deterministic prompting with strict JSON outputs, enabling scalable batch evaluation, regression testing, and production monitoring.
  Through a comparative study of two instruction-tuned models across short and long workflows, we show that generic proxy metrics provide ambiguous signals, while the proposed framework exposes enterprise-critical tradeoffs that are actionable for system improvement.

## Full Text


<!-- PDF content starts -->

Case-Aware LLM-as-a-Judge Evaluation for Enterprise-Scale RAG
Systems
Mukul Chhabra
Dell Technologies
Austin, TX, USA
mukul.chhabra@dell.comLuigi Medrano
Dell Technologies
Austin, TX, USA
Luigi.Medrano@dell.comArush Verma
Dell Technologies
Austin, TX, USA
Arush.Verma@dell.com
Abstract
Enterprise Retrieval-Augmented Generation (RAG) assistants operate in multi-turn, case-based workflows
such as technical support and IT operations, where evaluation must reflect operational constraints, structured
identifiers (e.g., error codes, versions), and resolution workflows. Existing RAG evaluation frameworks are
primarily designed for benchmark-style or single-turn settings and often fail to capture enterprise-specific
failure modes such as case misidentification, workflow misalignment, and partial resolution across turns.
We present a case-aware LLM-as-a-Judge evaluation framework for enterprise multi-turn RAG systems.
The framework evaluates each turn using eight operationally grounded metrics that separate retrieval quality,
grounding fidelity, answer utility, precision integrity, and case/workflow alignment. A severity-aware
scoring protocol reduces score inflation and improves diagnostic clarity across heterogeneous enterprise
cases. The system uses deterministic prompting with strict JSON outputs, enabling scalable batch evaluation,
regression testing, and production monitoring.
Through a comparative study of two instruction-tuned models across short and long workflows, we show that
generic proxy metrics provide ambiguous signals, while the proposed framework exposes enterprise-critical
tradeoffs that are actionable for system improvement.
1 Introduction
Retrieval-Augmented Generation (RAG) is widely
used to deploy large language models (LLMs) in
enterprise environments by combining retrieval
over proprietary content with conditional gener-
ation. While effective in reducing hallucinations,
enterprise deployments differ substantially from
benchmark-style QA: support cases are multi-turn,
operationally constrained, and require precise han-
dling of structured identifiers (e.g., error codes,
versions) and workflow alignment.
In production systems, responses that appear rel-
evant may still fail to resolve a case, misinterpret
structured signals, or violate troubleshooting or-
der. Generic RAG evaluation metrics—such as
faithfulness and relevance—often conflate retrieval
accuracy, grounding, and resolution quality into
coarse signals (Es et al., 2023, 2024). As a result,
they provide limited diagnostic value for enterprise
iteration and deployment decisions.
We propose an enterprise-focused evaluationframework based on the LLM-as-a-Judge paradigm
(Zheng et al., 2023). Our key contribution iscase-
aware evaluation: the judge conditions on multi-
turn history, case metadata, and retrieved evidence
while enforcing structured scoring across eight
enterprise-aligned metrics. The resulting frame-
work exposes operational failure modes—such as
retrieval mismatch, hallucination, case misidentifi-
cation, and workflow misalignment—and provides
actionable signals for production monitoring and
system improvement.
1.1 Contributions
This paper makes the following contributions:
•We formalize evaluation requirements for enter-
prise multi-turn RAG systems and identify recur-
ring operational failure modes not captured by
generic metric suites.
•We introduce eight case-aware evaluation met-
rics that disentangle retrieval quality, groundingarXiv:2602.20379v1  [cs.CL]  23 Feb 2026

fidelity, answer utility, precision integrity, and
resolution alignment.
•We propose a severity-aware scoring protocol
that improves diagnostic clarity across heteroge-
neous enterprise cases.
•We design an audit-friendly LLM-as-a-Judge im-
plementation with deterministic prompting and
structured JSON outputs for scalable batch eval-
uation.
•We empirically demonstrate that the proposed
framework yields more actionable diagnostic in-
sights than generic proxy metrics.
2Background and Enterprise Evaluation
Requirements
Enterprise RAG deployments differ from
benchmark-style QA in several key respects:
queries are multi-turn, retrieval may be partially
correct, and answers must comply with operational
workflows and structured identifiers (e.g., error
codes, versions). In such settings, a response can
be relevant yet misaligned with case resolution, or
factually grounded yet operationally incorrect.
Generic RAG evaluation frameworks, such as
RAGAS (Es et al., 2023, 2024), decompose quality
into dimensions such as faithfulness and relevance.
While effective for reference-free evaluation, these
metrics do not explicitly capture enterprise-specific
failure modes such as workflow compliance, preci-
sion integrity, and correct case interpretation. Our
framework complements this line of work by intro-
ducing case-aware, operationally grounded evalua-
tion signals.
2.1 LLM-as-a-Judge for Multi-Turn
Evaluation
LLM-as-a-Judge methods provide scalable evalu-
ation without exhaustive human labeling (Zheng
et al., 2023). Prior work identifies potential biases
(e.g., verbosity and position bias) and proposes
structured rubric-based prompting to improve reli-
ability (Liu et al., 2023; Dubois et al., 2024). We
adopt this paradigm while constraining the judge
to evidence-only inputs and enforcing structured
JSON outputs to maximize auditability and action-
ability in enterprise settings.
3 Related Work
Automated RAG evaluation has progressed toward
decomposed and component-aware metrics. RA-GAS introduces reference-free faithfulness and rel-
evance metrics (Es et al., 2023, 2024). ARES trains
lightweight judges for retrieval and generation scor-
ing (Saad-Falcon et al., 2024), and RAGChecker
provides diagnostic benchmarking across retrieval-
generation interactions (Ru et al., 2024).
Complementary work studies grounding and at-
tribution. RARR proposes retrieval-driven revision
pipelines (Gao et al., 2023), AttributionBench eval-
uates attribution reliability (Li et al., 2024), and
FActScore decomposes long-form generations into
atomic factual units (Min et al., 2023).
Our framework builds on these directions but
targets enterprise RAG systems, introducing case-
aware, workflow-sensitive, and precision-focused
metrics designed for multi-turn operational support
environments.
4 Framework Overview
Figure 1 summarizes the batch evaluation pipeline.
Each row corresponds to one evaluated turn with
structured case fields, retrieved contexts, and the
model answer. For each row, the system normal-
izes inputs, constructs a single deterministic judge
prompt, invokes the LLM once, validates a strict
JSON schema, and aggregates per-metric scores
intoS final.
Inputs and scope.Let qbe the current query,
Hthe conversation history, csthe case subject,
cdthe case description, R={r 1, . . . , r n}the re-
trieved chunks, and athe model answer (optionally
a reference g). The judge is explicitly constrained
to{H, q, c s, cd, R, a, g} and must not use external
knowledge.
Outputs.The judge returns eight continuous
scores si∈[0,1] with brief justifications, plus
a weighted aggregate Sfinalfor monitoring and re-
gression tests (0=failure, 1=full compliance; inter-
mediate values reflect partial rubric satisfaction).
Case-aware evaluation.We call the evaluation
case-awarebecause scoring conditions on (i) case
metadata (cs, cd), (ii) multi-turn state Hcapturing
prior attempts and constraints, and (iii) retrieved
evidence R, while restricting judgment to these
inputs. This contrasts with common single-turn
RAG evaluation that ignores workflow constraints
and identifier-critical correctness.
Case-aware evaluation.We definecase-aware
LLM-judge evaluation as scoring that condi-

Inputs
H, q, c s, cd, R, a
(+ optionalg)Normalize
& ValidateBuild
Judge PromptLLM Judge
(single call)Parse JSON
& Sanity Check
8 Metric Scores
+ JustificationsAggregate
SfinalOutputs
eval table
+ compact table
Figure 1: End-to-end evaluation pipeline for case-aware LLM-as-a-Judge scoring of multi-turn RAG responses.
tions on (i) structured case metadata (e.g., sub-
ject/description), (ii) multi-turn history capturing
attempted steps and constraints, and (iii) retrieved
evidence, while restricting judgments to these in-
puts only. This differs from standard RAG evalua-
tion, which typically assumes single-turn indepen-
dence and does not model workflow constraints or
identifier-critical correctness.
4.1 Explainability and Actionability
Each metric is tied to an engineering lever:Re-
trieval CorrectnessandContext Sufficiencydiag-
nose retriever, chunking, and evidence coverage;
Identifier Integrityflags high-severity command/ID
corruption; and metric-level justifications make
regressions auditable and accelerate production
triage.
4.2 Metric Definitions
We score eight case-aware dimensions spanning
evidence quality, grounded response quality, and
workflow safety:
•Evidence quality:Retrieval Correctness(re-
trieved chunks contain needed facts) andCon-
text Sufficiency(retrieval covers all required evi-
dence).
•Grounded response quality:Hallucination /
Grounding Fidelity(claims supported by R),An-
swer Helpfulness(actionable and clear), andAn-
swer Type Fit(diagnose vs. instruct vs. clarify).
•Workflow safety:Identifier Integrity(no corrup-
tion of commands/IDs/paths),Case Issue Identi-
fication(correct issue given H, c s, cd), andRes-
olution Alignment(steps satisfy constraints and
likely resolve).
Formal definitions and scoring rubrics are pro-
vided in Appendix C.
5 Severity-Based Scoring and
Aggregation
5.1 Severity-Based Scoring
To reflect enterprise risk, composite scores are
mapped to four severity bands (Critical, Ma-jor, Moderate, Minor); definitions appear in Ap-
pendix B.
5.2 Aggregated Score
For monitoring and regression testing, we compute
a weighted sum:
Sfinal=8X
i=1wisi,(1)
where weights wireflect organizational risk toler-
ance, emphasizing grounding fidelity and retrieval
quality. We use: Hallucination 0.20, Retrieval Cor-
rectness 0.15, Context Sufficiency 0.10, Answer
Helpfulness 0.15, Answer Type Fit 0.10, Identi-
fier Integrity 0.10, Case Issue Identification 0.10,
Resolution Alignment 0.10.
6 Implementation
The framework is a deterministic batch pipeline:
each turn is serialized into a structured judge
prompt and evaluated with a single LLM call. Out-
puts must match a strict JSON schema; invalid
JSON triggers bounded retries and otherwise fails
closed for inspection.
Judge configuration: GPT-4 via Azure OpenAI,
temperature=0.0, top_p=1.0, max_tokens=1024.
We verify weight-robustness using (i) uniform
weights ( wi=0.125 ) and (ii) a retrieval-heavy pro-
file (Retrieval Correctness and Context Sufficiency
set to 0.20 each; others scaled proportionally). Con-
clusions are stable: the long-query ordering is un-
changed and composite scores shift minimally (typ-
ically<3%).
6.1 Cost Analysis
Evaluation cost is linear in turns (O(N)) with one
judge call per turn. Using OpenAI API standard
pricing for gpt-4.1-2025-04-14 ($3 / 1M input
tokens, $12 / 1M output tokens) (OpenAI, 2026),
the per-turn judge cost is ≈3×10−6Tin+ 12×
10−6Toutdollars. For a typical prompt of ∼3,000
input tokens and ∼400 output tokens, this is ≈
$0.014 per turn; evaluating 469 turns would cost

Metric GPT-OSS (Short) LLaMA (Short) GPT-OSS (Long) LLaMA (Long)
Hallucination (Grounding Fidelity) 0.6890 0.7431 0.7586 0.7132
Retrieval Correctness 0.7429 0.7647 0.7941 0.7578
Context Sufficiency 0.6526 0.6429 0.7411 0.6546
Answer Helpfulness 0.6943 0.6143 0.7724 0.6143
Answer Type Fit 0.7603 0.7274 0.8360 0.6825
Identifier Integrity 0.9421 0.9354 0.9559 0.9329
Case Issue Identification 0.7636 0.6956 0.8783 0.7052
Resolution Alignment 0.7004 0.6459 0.8208 0.6759
Weighted Aggregate 0.7353 0.7202 0.8099 0.7136
Table 1: Corrected retrieval-aware evaluation results. Scores reflect full judge evaluation with retrieved contexts
provided.
≈$6.6 (Azure pricing may differ, but the estimate
is directly token-scalable).
7 Evaluation Methodology
This section describes the experimental protocol
used to evaluate the proposed case-aware LLM-
as-a-Judge framework within an enterprise trou-
bleshooting setting.
7.1 Dataset and Experimental Setup
Evaluation was conducted on two disjoint subsets
of anonymized enterprise support cases:
•Short queries:237 cases
•Long queries:232 cases
Each row corresponds to a single evaluated turn
(one model response) with any available prior con-
versation history included in the case fields. The
judge is invoked once per row/turn. When reporting
conversation-level results, we aggregate turn-level
scores by a conversation identifier.
Cases were sampled from real-world enterprise
troubleshooting scenarios and normalized to re-
move personally identifiable information.
Sampling Procedure.Cases were sampled from
anonymized enterprise support logs across mul-
tiple product categories over a six-month period.
Sampling was stratified to include both short-form
queries (single-issue diagnostic requests) and long-
form diagnostic narratives containing prior trou-
bleshooting context. Duplicate cases and non-
technical requests were excluded.
7.2 Evaluation Paradigms
We compare three evaluation approaches:
•Generic proxy metrics:RAGAS-style signals
including faithfulness and answer relevancy.•Heuristic checks:keyword overlap and retrieval
scoring thresholds.
•Proposed framework:an eight-metric case-
aware judge with severity-weighted aggregation
into a final scoreS final .
The objective is to assess whether enterprise-
aligned metrics provide more diagnostic and opera-
tionally actionable signals than generic proxies.
7.3 Models Evaluated
To stress-test the evaluation framework across het-
erogeneous model behavior, we evaluated two
instruction-tuned models representing distinct ar-
chitectural trade-offs.
All models were tested using identical retrieval
pipelines, prompts, and conversation histories to
isolate model and evaluation effects from system
configuration differences.
•Llama-3.3-70B-Instruct
•gpt-oss-120b
The comparison is intentionally heterogeneous
to demonstrate that the framework remains discrim-
inative across model families.
8 Results and Analysis
This evaluation does not seek leaderboard ranking,
but rather tests whether case-aware metrics reveal
enterprise-relevant distinctions that generic proxy
signals overlook.
Table 1 reports mean enterprise metric scores
across evaluated cases.
8.1 Key Findings
Proxy metrics produce mixed signals. In con-
trast, the proposed case-aware framework exposes
workflow-critical differences including issue target-
ing, resolution alignment, and precision integrity.

Dataset n Mean (GPT) Mean (Llama)∆Wilcoxonp
Short Queries 70 0.7353 0.7202 +0.0151 0.6495
Long Queries 63 0.8099 0.7136 +0.0963 0.0011
Table 2: Paired Wilcoxon signed-rank test on per-case weighted aggregate scores.∆denotes GPT minus Llama.
•On short queries, Llama demonstrates stronger
grounding fidelity and identifier preservation.
•GPT-oss exhibits higher helpfulness and stronger
alignment with case resolution workflows.
•On long-form diagnostic queries, GPT-oss
achieves significantly higher weighted aggre-
gate scores, indicating greater robustness under
context-heavy and multi-step diagnostic condi-
tions.
These findings demonstrate that faithfulness and
relevance alone are insufficient for enterprise de-
ployment decisions.
8.2 Input Complexity
Short queries average 3.47 words (median = 3),
whereas long queries average 14.76 words (median
= 14), representing a 4.25 ×increase in input length.
The long-query subset therefore reflects substan-
tially higher descriptive complexity and retrieval
demands.
8.3 Statistical Significance
Unit of analysis.Tests are performed at thecon-
versation(case) level, not per turn. We group turns
by conversation identifier and compute each con-
versation’s mean weighted composite score. This
yields n=70 short-query conversations and n=63
long-query conversations, avoiding inflated signifi-
cance from within-conversation correlation.
Tests.We use paired, two-sided Wilcoxon signed-
rank tests on per-conversation weighted scores (Ta-
ble 2); paired t-tests and bootstrap confidence in-
tervals are reported as robustness checks.
For short queries, GPT-oss achieves a direction-
ally higher weighted aggregate score, but the dif-
ference is not statistically significant ( p= 0.6495 ).
The bootstrap 95% confidence interval for the mean
difference is [−0.0315,0.0623] , indicating overlap
around zero.
For long diagnostic queries, GPT-oss signifi-
cantly outperforms Llama under the case-aware
evaluation framework ( p= 0.0011 ). The boot-
strap 95% confidence interval for the mean differ-
ence is [0.0486,0.1469] , remaining strictly abovezero. This indicates that performance separation be-
comes pronounced under higher input complexity
and multi-step diagnostic reasoning.
8.4 Judge Robustness
We re-ran the full case-aware evaluation with
llama-3.3-70b-instruct as an alternative judge
using the same rubric, strict JSON schema, and ag-
gregation weights (Eq. 1). This second judge is not
treated as ground truth; it is used to probe stability
under evaluator variation.
Conclusions are stable at the system level: GPT-
oss scores higher than Llama under both judges
(Table 3). The long-query separation remains statis-
tically significant (GPT-4 judge: p=0.0011 ; Llama
judge: p=0.0016 ). Short-query effects are smaller
and judge-sensitive (GPT-4: p=0.6495 ; Llama:
p=0.0005 ). Inter-judge agreement on the weighted
aggregate is moderate-to-strong ( ρ∈[0.52,0.69] ),
with lower agreement on a few short-query metrics
(Appendix K.2).
8.5 Comparison with Generic Proxy Metrics
Proxy metrics suggested higher faithfulness for
Llama and stronger answer relevancy for GPT-oss
(Table 4), yet failed to indicate which system better
supported case resolution workflows.
Unlike proxy metrics, the proposed framework
decomposes performance into actionable enterprise
dimensions, reducing the risk of false confidence
during deployment decisions.
8.6 Enterprise Diagnosis and Operational Use
Metric-level outputs enable targeted engineering
intervention:
•Low retrieval correctness with low sufficiency in-
dicates retriever or chunk refinement weaknesses.
•Strong grounding but weak helpfulness suggests
response structuring improvements.
•Low alignment signals workflow violations re-
quiring improved case extraction or conversa-
tional memory.

Subset JudgenMean (GPT-oss) Mean (Llama)∆Wilcoxonp
Short GPT-4 70 0.7353 0.7202 +0.0151 0.6495
Short Llama-3.3-70B 70 0.6639 0.6060 +0.0578 0.0005
Long GPT-4 63 0.8099 0.7136 +0.0963 0.0011
Long Llama-3.3-70B 63 0.7143 0.6589 +0.0554 0.0016
Table 3: Judge-robustness check on conversation-level weighted aggregate scores ( Sfinal).∆denotes GPT-oss minus
Llama.
Metric Llama GPT-oss
Faithfulness (Short) 0.8439 0.5799
Answer Relevancy (Short) 0.7404 0.8192
Faithfulness (Long) 0.8123 0.6005
Answer Relevancy (Long) 0.7276 0.8330
Table 4: Representative proxy metrics produce mixed
signals and do not capture workflow alignment or issue
targeting.
Severity-weighted scoring prevents inflation
from partially correct answers, ensuring high-risk
failures meaningfully impact aggregate evaluation.
This enables:
• Release gating via thresholdedS final
• Regression testing across model updates
•Root-cause diagnosis through metric decomposi-
tion
• Continuous production monitoring
8.7 Human Alignment Validation
To assess alignment between the LLM judge and
human evaluators, we conducted a lightweight val-
idation on a stratified random sample of 60 evalu-
ated turns (30 short, 30 long). Two domain experts
independently reviewed each sampled turn using
the same case fields and retrieved contexts available
to the judge.
Humans provided binary judgments for three op-
erationally critical dimensions: Hallucination, Iden-
tifier Integrity, and Resolution Alignment. Agree-
ment between the LLM judge and the majority
human vote was:
• Hallucination: 88%
• Identifier Integrity: 91%
• Resolution Alignment: 84%
Disagreements occurred primarily in borderline
cases involving partially correct remediation steps
or implicitly referenced identifiers. Overall, results
suggest strong alignment for high-risk enterprise
failure modes.9 Limitations
Our framework relies on the availability of repre-
sentative evaluation conversations and high-quality
case fields (subject/description) to enable case-
aware judging. While severity-based scoring im-
proves stability for heterogeneous enterprise cases,
rubric calibration and metric weights may require
adjustment across different organizations, indus-
tries, and risk tolerances.
10 Conclusion
We present a practical, case-aware LLM-as-a-judge
evaluation framework for enterprise multi-turn
RAG systems. By grounding evaluation in oper-
ational failure modes including workflow align-
ment, issue targeting, and precision integrity, the
framework provides diagnostic signals that generic
proxy metrics often miss. The resulting metric suite
and batch implementation enable scalable audit-
ing, regression testing, and release gating decisions
rooted in production realities.
11 Reproducibility and Artifacts
To support replication, we document the full evalua-
tion protocol (judge rubric, prompt structure, JSON
schema, and aggregation procedure) and provide
code/configuration details sufficient to reproduce
the reported aggregates on any similarly format-
ted case set. Due to enterprise confidentiality, raw
case logs cannot be shared. Where permitted, we
will make available de-identified examples and/or
synthetic cases that preserve identifier and work-
flow structure to illustrate end-to-end scoring and
aggregation.
Available (where permitted):judge rubric and
prompt templates; strict JSON schema and val-
idation rules; aggregation weights/config pro-
files; analysis and significance-testing scripts; de-
identified and/or synthetic example cases with ex-
pected judge outputs.

12 Ethical and Safety Considerations
Enterprise support assistants can recommend op-
erational actions with real-world consequences.
Our evaluation explicitly treats unsafe or destruc-
tive guidance as severe failures via hallucination
(grounding fidelity), identifier integrity, and reso-
lution alignment. The framework is intended to
support release gating and regression testing to re-
duce deployment risk. We also note that LLM-
judge evaluation can encode biases; we mitigate
this via deterministic prompting, strict evidence-
only inputs, schema validation, and (when feasible)
cross-judge and human alignment checks.
References
Yann Dubois, Balázs Galambosi, Percy Liang, and Tat-
sunori B. Hashimoto. 2024. Length-controlled al-
pacaeval: A simple way to debias automatic evalua-
tors.arXiv preprint arXiv:2404.04475.
Shahul Es, Jithin James, Luis Espinosa-Anke, and
Steven Schockaert. 2023. RAGAS: Automated eval-
uation of retrieval-augmented generation.arXiv
preprint arXiv:2309.15217.
Shahul Es, Jithin James, Luis Espinosa-Anke, and
Steven Schockaert. 2024. RAGAs: Automated evalu-
ation of retrieval-augmented generation. InProceed-
ings of the 18th Conference of the European Chap-
ter of the Association for Computational Linguistics:
System Demonstrations.
Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony
Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent
Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, and
Kelvin Guu. 2023. RARR: Researching and revising
what language models say, using language models.
InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume 1:
Long Papers), pages 16477–16508, Toronto, Canada.
Association for Computational Linguistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and Sebastian Riedel. 2020. Retrieval-
augmented generation for knowledge-intensive NLP
tasks. InAdvances in Neural Information Processing
Systems.
Yifei Li, Xiang Yue, Zeyi Liao, and Huan Sun. 2024.
AttributionBench: How hard is automatic attribu-
tion evaluation? InFindings of the Association for
Computational Linguistics: ACL 2024, pages 14919–
14935, Bangkok, Thailand. Association for Compu-
tational Linguistics.
Percy Liang, Rishi Bommasani, Tony Lee, Dimitris
Tsipras, Dilara Soylu, Michihiro Yasunaga, YianZhang, Deepak Narayanan, Yuhuai Wu, Ananya Ku-
mar, Benjamin Newman, Binhang Yuan, Bobby Yan,
Ce Zhang, Christian Cosgrove, Christopher D. Man-
ning, Christopher Ré, Diana Acosta-Navas, Drew A.
Hudson, and 31 others. 2022. Holistic evaluation of
language models.arXiv preprint arXiv:2211.09110.
Stephanie Lin, Jacob Hilton, and Owain Evans. 2022.
TruthfulQA: Measuring how models mimic human
falsehoods. InProceedings of the 60th Annual Meet-
ing of the Association for Computational Linguistics.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang,
Ruochen Xu, and Chenguang Zhu. 2023. G-eval:
NLG evaluation using gpt-4 with better human align-
ment. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 2511–2522, Singapore. Association for Com-
putational Linguistics.
Potsawee Manakul, Adian Liusie, and Mark J. F. Gales.
2023. SelfCheckGPT: Zero-resource black-box hal-
lucination detection for generative large language
models.arXiv preprint arXiv:2303.08896.
Shikib Mehri and Maxine Eskenazi. 2020. USR: An
unsupervised and reference-free evaluation metric
for dialog generation. InProceedings of the 58th
Annual Meeting of the Association for Computational
Linguistics.
Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike
Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer,
Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023.
FActScore: Fine-grained atomic evaluation of factual
precision in long-form text generation. InProceed-
ings of the 2023 Conference on Empirical Methods
in Natural Language Processing.
Souvik Nath, Sumit Wadhwa, and Luis Perez.
2025. Domain-adaptive small language models
for structured tax code prediction.arXiv preprint
arXiv:2507.10880.
Aashutosh Nema, Samaksh Gulati, Evangelos Giak-
oumakis, and Bipana Thapaliya. 2025. Modp: Multi
objective directional prompting.arXiv preprint
arXiv:2504.18722.
OpenAI. 2026. Openai api pricing. https:
//developers.openai.com/api/docs/pricing/ .
Accessed: 2026-02-14.
Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin,
and Sameer Singh. 2020. Beyond accuracy: Be-
havioral testing of NLP models with CheckList. In
Proceedings of the 58th Annual Meeting of the Asso-
ciation for Computational Linguistics, pages 4902–
4912, Online. Association for Computational Lin-
guistics.
Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang,
Peng Shi, Shuaichen Chang, Cheng Jiayang, Cunxi-
ang Wang, Shichao Sun, Huanyu Li, Zizhao Zhang,
Binjie Wang, Jiarong Jiang, Tong He, Zhiguo Wang,
Pengfei Liu, Yue Zhang, and Zheng Zhang. 2024.

RAGChecker: A fine-grained framework for diagnos-
ing retrieval-augmented generation. InAdvances in
Neural Information Processing Systems 37 (NeurIPS
2024), Datasets and Benchmarks Track.
Jon Saad-Falcon, Omar Khattab, Christopher Potts, and
Matei Zaharia. 2024. ARES: An automated evalua-
tion framework for retrieval-augmented generation
systems. InProceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), pages 338–354,
Mexico City, Mexico. Association for Computational
Linguistics.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging
LLM-as-a-judge with MT-bench and chatbot arena.
arXiv preprint arXiv:2306.05685.

A Extended Experimental Analysis
This appendix provides a deeper analysis of the pro-
posed case-aware LLM-as-a-judge framework. The
main paper presents high-level results and key find-
ings; here we include diagnostic visualizations, sta-
tistical observations, and implementation specifics
that further validate robustness, interpretability, and
enterprise applicability.
A.1 Evaluation Dataset Characteristics
The evaluation dataset consists of multi-turn enter-
prise support conversations involving troubleshoot-
ing, configuration issues, and error-code-driven res-
olution workflows.
Each conversation turn was independently evalu-
ated across eight continuous metrics:
• Hallucination
• Retrieval Correctness
• Context Sufficiency
• Answer Helpfulness
• Answer Type Fit
• Identifier Integrity
• Case Issue Identification
• Resolution Alignment
Each metric is represented as a continuous scalar
in[0,1] . Scores near 0 indicate severe rubric viola-
tions, while values near 1 indicate strong compli-
ance. Intermediate values represent partial satisfac-
tion of the metric criteria.
A.2 Operational Evaluation Considerations
Enterprise evaluation systems require strict pars-
ing and validation, including JSON-only outputs,
bounded score ranges, and required metric keys.
Batch scoring may require extended runtime;
therefore progress tracking, retry mechanisms, and
output logging are necessary for reliability. All
prompts, raw judge outputs, and parsed metric
scores were stored for auditability.
For diagnostic workflows, two output formats
are recommended:
•Granular output:per-row metric scores, textual
justifications, and raw judge outputs.
•Compact output:case fields, response, retrieved
contexts, and aggregated Sfinalfor monitoring
dashboards.B Severity Band Definitions
B.1 Severity-Based Scoring
In enterprise settings, a single severe failure (e.g.,
hallucinated destructive action) should strongly im-
pact evaluation. We therefore recommend severity-
based scoring bands, where the judge first identifies
the most severe issue per metric and maps it to a
score region:
•Severefailure: 0.00–0.30 (unsafe, contradictory,
or clearly wrong)
•Moderatefailure: 0.31–0.60 (partly correct but
flawed / incomplete)
•Minorissues: 0.61–0.85 (mostly correct with
small gaps)
•No meaningful issues: 0.86–1.00 (excellent)
This reduces score inflation, improves stability
across heterogeneous cases, and makes metric out-
puts more interpretable by engineering teams.
C Enterprise Metric Suite
We define eightindependentandactionable
metrics designed to disentangle retrieval quality,
grounding fidelity, resolution utility, precision in-
tegrity, and workflow alignment. Each metric maps
directly to a distinct engineering intervention.
M1: Hallucination (Grounding Fidelity).Mea-
sures whether claims and prescribed steps are sup-
ported by retrieved context and case fields. Un-
supported or contradictory assertions reduce this
score.
M2: Retrieval Correctness.Evaluates whether
retrieved contexts Rare relevant and case-
appropriate, isolating retriever-layer failures such
as incorrect versions, environments, or stale docu-
mentation.
M3: Context Sufficiency.Assesses whether R
provides enough evidence for a safe response.
When evidence is incomplete, strong answers re-
quest clarification rather than speculate.
M4: Answer Helpfulness.Captures practical
utility: whether the response advances resolution
through prioritized actions, structured troubleshoot-
ing, or clarifying questions.

M5: Answer Type Fit.Determines whether re-
sponse specificity matches query intent (e.g., con-
crete commands for operational issues versus high-
level guidance for conceptual questions).
M6: Identifier Integrity.Evaluates correct inter-
pretation and preservation of structured identifiers
(e.g., error codes, versions, commands). Corrup-
tion or misapplication is penalized.
M7: Case Issue Identification.Measures
whether the response correctly identifies the un-
derlying issue rather than drifting into adjacent
topics.
M8: Resolution Alignment.Assesses compli-
ance with case constraints and workflow, including
respect for prior attempted steps, required sequenc-
ing, and prohibited actions.
D Dataset Composition Details
The evaluation dataset spans multiple enterprise
storage and infrastructure product families across
heterogeneous troubleshooting workflows.
Error-code-driven cases represent approximately
3.4% of short cases and 6.0% of long cases.
All cases were sampled from anonymized enter-
prise troubleshooting scenarios and normalized to
remove personally identifiable information.
D.1 Inter-Metric Correlation Analysis
To examine metric redundancy and interdepen-
dence, we compute Pearson correlation coefficients
across all eight metrics.
Observations.Several important patterns
emerge:
•Retrieval Correctness and Context Suffi-
ciencyshow moderate positive correlation, in-
dicating that correct retrieval often—but not
Figure 3: Correlation heatmap across evaluation met-
rics.
always—implies sufficient contextual ground-
ing.
•Case Issue Identification and Resolution
Alignmentdemonstrate strong correlation,
suggesting that accurate issue diagnosis fre-
quently leads to aligned resolution steps.
•Hallucinationremains comparatively orthog-
onal to resolution alignment, reinforcing that
factual grounding and actionable resolution
are distinct evaluation dimensions.
This validates the need for multi-dimensional
scoring rather than collapsing evaluation into a sin-
gle faithfulness metric.
D.2 Per-Metric Score Distributions
We analyze score distributions across conversation
turns.
Distributional Insights.
•Retrieval-related metrics exhibit higher vari-
ance, particularly in early conversation turns
involving ambiguous queries.
Halluc.Retr. Suff. Help.TypeFit Ident.IssueIDAlign.00.10.2WeightExample default aggregation weights (tune per domain/risk)
Figure 2: Example metric weights forS final. Rendered as a wide figure to avoid cut-off in two-column layout.

Figure 4: Score distributions per metric.
Figure 5: Word count distribution of LLM judge justifi-
cations per metric.
•Identifier integrity demonstrates bimodal be-
havior when structured identifiers (e.g., error
codes, versions, commands) are present ver-
sus absent.
•Answer Type Fit maintains consistently high
scores, suggesting structural compliance is
easier to satisfy than deep contextual correct-
ness.
These results illustrate that enterprise RAG sys-
tems fail more frequently in retrieval-context align-
ment than in structural formatting.
D.3 Justification Length as a Complexity
Proxy
Because the framework requires textual justifica-
tions per metric, we analyze justification length as
a proxy for reasoning complexity.
Key Findings.
•Hallucination detection produces the longest
justifications, consistent with the need for fac-
tual cross-verification.•Answer helpfulness judgments require ex-
tended reasoning chains to evaluate complete-
ness and actionability.
•Retrieval correctness explanations are compar-
atively shorter, reflecting binary factual match-
ing.
This suggests that certain evaluation dimen-
sions impose greater cognitive load on the judging
model.
D.4 Aggregate Metric Validity
We examine the relationship between the weighted
aggregate score (S final ) and answer helpfulness.
Construct Validity.Pearson correlation between
Sfinal and Answer Helpfulness was r= 0.82 (p <
0.001 ), indicating strong alignment between the
composite score and perceived answer utility. This
indicates:
•Improvements across structured enterprise
metrics translate into improved perceived use-
fulness.
•The evaluation framework aligns with down-
stream user experience metrics.
•Composite scoring captures meaningful sys-
tem quality signals.
This supports the framework’s construct validity
in real-world enterprise settings.
E Extended Metric Interpretation
E.1 Retrieval and Sufficiency Distributions
Mean retrieval correctness and context sufficiency
scores were lower relative to identifier integrity
and answer type fit. This reflects the strict rubric
applied during evaluation.
Retrieval correctness required alignment with
case-specific constraints such as product version,
deployment context, and previously attempted
steps. Partial matches or stale references were pe-
nalized. Context sufficiency required that retrieved
evidence fully support safe resolution without omis-
sion of critical procedural elements.
In contrast, identifier integrity evaluates preser-
vation and correct use of structured identifiers
within the generated response itself (e.g., error
codes, versions, commands), a narrower constraint
that is typically easier to satisfy when explicit iden-
tifiers are present.

These distributional differences therefore reflect
distinct failure modes rather than retriever collapse.
E.2 Enterprise Metric Profile Visualization
Figure 6: Radar visualization of normalized mean enter-
prise metric scores for long diagnostic queries. While
statistical significance is reported in Table 2, this figure
provides a qualitative view of performance divergence
across workflow-aligned dimensions.
FEnterprise Motivation and Operational
Context
Enterprise RAG systems operate under opera-
tional constraints that differ substantially from
benchmark-style QA evaluation. In customer sup-
port environments, partially helpful or structurally
plausible responses can still lead to escalation, in-
creased handling time, and reduced customer trust.
In production deployments, we observed cases
where standard evaluation metrics rated responses
as successful despite failures to:
•Correctly interpret structured identifiers (e.g.,
error codes, versions)
• Follow prescribed troubleshooting order
• Align with established escalation workflows
•Maintain issue continuity across multi-turn
exchanges
Such failures directly impact operational cost
and deployment risk. Evaluation therefore func-
tions not merely as an academic measurement tool,
but as a gating mechanism for deployment, rollback
decisions, and system iteration.
A key requirement in enterprise settings isdiag-
nostic granularity. When quality degrades, engi-
neers must determine whether the root cause arises
from:
• Retrieval mismatch
Figure 7: Weighted aggregate score ( Sfinal ) vs. answer
helpfulness.
• Insufficient contextual grounding
• Hallucination
• Case misinterpretation
• Resolution misalignment
Generic faithfulness and relevance metrics of-
ten conflate these dimensions, limiting their use-
fulness for production debugging. The proposed
case-aware evaluation framework is motivated by
repeated observations that precision-sensitive en-
terprise failure modes are underrepresented in stan-
dard RAG evaluation approaches.
G Extended Positioning Against Prior
Work
G.1 Limitations of Generic RAG Evaluation
Assumptions
Generic RAG evaluation often assumes: (i) context-
independent queries, (ii) complete and correct re-
trieved evidence, and (iii) externally verifiable
ground truth.
Enterprise cases routinely violate these assump-
tions. Retrieval may be partially correct while omit-
ting critical procedural steps. Evidence may mix
stale and current policy documentation. Queries
frequently depend on prior troubleshooting actions,
environment constraints (e.g., production vs. lab),
and organizational workflows.
In these settings:
•Answer relevance may be high while case
alignment is low.
•Faithfulness may be ambiguous when re-
trieved context is incomplete.

•Resolution quality depends on workflow com-
pliance, not only factual correctness.
Our framework complements generic proxy met-
rics by explicitly modeling workflow compliance,
precision integrity, and case resolution progress.
G.2 Comparison to RAGAS, ARES, and
RAGChecker
RAGAS decomposes RAG quality into faithfulness
and relevance signals, enabling reference-free eval-
uation (Es et al., 2023, 2024). ARES introduces
a supervised pipeline for lightweight evaluators
(Saad-Falcon et al., 2024). RAGChecker proposes
a fine-grained diagnostic benchmark spanning re-
trieval and generation failures (Ru et al., 2024).
These works demonstrate the value of decom-
posed evaluation. However, enterprise deploy-
ments frequently require additional dimensions:
•Structured identifier sensitivity (e.g., error
codes)
• Case issue identification accuracy
•Compliance with prescribed troubleshooting
order
• Multi-turn resolution alignment
These operational constraints motivate our case-
aware metric design.
G.3 Grounding and Attribution Literature
Prior work studies factual grounding and attribu-
tion reliability. RARR introduces retrieval-driven
revision pipelines (Gao et al., 2023). Attribution-
Bench highlights evaluation challenges in attribu-
tion quality (Li et al., 2024). FActScore measures
atomic factual precision in long-form text (Min
et al., 2023).
While these approaches focus on factual support,
enterprise RAG evaluation requires additional op-
erational criteria—specifically, whether the answer
advances resolution under case constraints.
G.4 LLM-as-a-Judge Reliability
Considerations
LLM-as-a-judge approaches scale evaluation with-
out exhaustive human labeling (Zheng et al., 2023).
However, studies document verbosity bias, position
bias, and length sensitivity (Dubois et al., 2024).
To mitigate such biases, our framework:
• Enforces continuous scoring per metric• Requires explicit justification
• Uses deterministic temperature settings
•Restricts inputs to retrieved evidence and case
context only
This design improves auditability and repro-
ducibility in enterprise evaluation settings.
H Evaluation Schema and
Implementation Details
H.1 Evaluation Record Schema
Each evaluated turn produces a structured JSON
record containing:
• real-valued scalar in[0,1]
•A short natural-language justification per met-
ric
• An aggregated final scoreS final
Table 5 shows the full evaluation schema used
in batch processing.
H.2 Continuous Scoring Interpretation
Each metric is represented as a real-valued scalar
in[0,1] . A score of 0 indicates complete failure for
that dimension, while 1 indicates full compliance
with the rubric. Intermediate values reflect partial
satisfaction of the metric criteria.
Continuous scoring enables finer-grained aggre-
gation, regression detection, and cross-case com-
parison in heterogeneous enterprise workflows. To
improve stability, scoring anchors are defined in the
rubric and deterministic prompting is used during
evaluation.
H.3 Batch Processing and Validation
The evaluation pipeline enforces:
• Deterministic prompting
• Strict JSON schema validation
• Automatic retries for malformed outputs
• Turn-level and case-level aggregation
I Detailed Limitations of Existing RAG
Evaluation Frameworks
When applied to enterprise multi-turn deployments,
existing RAG evaluation frameworks exhibit sev-
eral recurring limitations:

Metric Score field Justification field
M1 Hallucinationhallucination_score hallucination_justification
M2 Retrieval Correctnessretrieval_correctness_score retrieval_correctness_justification
M3 Context Sufficiencycontext_sufficiency_score context_sufficiency_justification
M4 Answer Helpfulnessanswer_helpfulness_score answer_helpfulness_justification
M5 Answer Type Fitanswer_type_fit_score answer_type_fit_justification
M6 Identifier Integrityidentifier_integrity_score identifier_integrity_justification
M7 Case Issue Identificationcase_issue_identification_score case_issue_identification_justification
M8 Resolution Alignmentcase_resolution_alignment_score case_resolution_alignment_justification
Table 5: Full evaluation schema used in batch processing.
I.1 Independent Metric Evaluation
Many frameworks score retrieval and generation
quality independently. While useful for bench-
marking, this separation can obscure compound
failure modes in production systems, where partial
retrieval mismatch combined with subtle reasoning
drift produces operationally significant errors.
I.2 Single-Turn Assumptions
Enterprise cases frequently span multiple turns
with clarification, hypothesis testing, and progres-
sive resolution steps. Single-turn evaluation fails to
capture degradation, recovery, and workflow conti-
nuity across turns.
I.3 Lack of Severity Awareness
Generic metrics often treat minor stylistic devia-
tions and critical misinformation equivalently. In
operational environments, severity matters: misin-
terpreting an error code or violating troubleshoot-
ing order carries substantially higher risk than mi-
nor phrasing issues.
I.4 Limited Actionability
Scores such as faithfulness or relevance provide
coarse quality estimates but offer limited guidance
for engineering decisions such as retriever adjust-
ments, prompt revisions, or model rollouts. En-
terprise teams require metrics that map directly to
identifiable failure modes.
These limitations motivated the development of
a case-aware, severity-sensitive evaluation frame-
work tailored to operational RAG deployments.
J Extended Metric Definitions and
Examples
This section provides illustrative examples and
edge-case clarifications for each metric.J.1 Hallucination vs. General Guidance
General diagnostic advice is permissible if clearly
framed as generic and not presented as case-
specific fact. Hallucination occurs when the model
invents case-specific assertions unsupported by re-
trieved evidence.
J.2 Retrieval Failure Examples
Common retrieval errors include incorrect prod-
uct version, environment mismatch (production vs.
lab), and stale policy references. These failures
are isolated from generation quality by the retrieval
correctness metric.
J.3 Illustrative Failure Case: Workflow
Violation Despite Correct Retrieval
Case Type:Long diagnostic query involving a
firmware update failure requiring prerequisite soft-
ware upgrade.
Retrieved Context (summarized):The re-
trieved knowledge article specifies that software
version 2.14 must be installed before applying the
firmware patch.
Model Response (example failure):The model
recommends applying the firmware patch directly
without verifying or updating the software version.
Expected Outcome:The correct workflow re-
quires verifying software version first and perform-
ing the prerequisite upgrade before the firmware
patch.
Metric Impact:Retrieval Correctness may re-
main high (context contains the correct prerequi-
site), while Resolution Alignment is penalized due
to violating documented sequencing. This illus-
trates why proxy metrics can miss operational fail-
ure modes even when retrieval appears correct.
J.4 Severity of Identifier Corruption
Enterprise environments frequently depend on pre-
cise identifiers (error codes, command flags, file
paths). Altered versions or corrupted commands

ConditionnSpearmanρonS final
Short (Llama responses) 70 0.515
Short (GPT-oss responses) 70 0.577
Long (Llama responses) 63 0.691
Long (GPT-oss responses) 63 0.644
Table 6: Inter-judge rank consistency between GPT-4
and Llama judges on weighted aggregate scores.
constitute high-severity failures and are explicitly
penalized.
J.5 Workflow Misalignment
Answers that repeat already attempted steps, skip
required diagnostic sequencing, or recommend pro-
hibited actions (e.g., unsafe production changes)
reduce the Resolution Alignment score.
K Statistical Robustness Considerations
K.1 Variance Across Conversation Turns
Evaluation variance is highest in:
• Short queries lacking explicit error codes
• Multi-turn clarification exchanges
•Retrieval boundary cases with partially rele-
vant documents
This highlights enterprise-specific RAG brittle-
ness.
K.2 Inter-Judge Consistency
We measure inter-judge rank consistency be-
tween GPT-4 and llama-3.3-70b-instruct on
conversation-level scores using Spearman correla-
tion.
L LLM-as-a-Judge Prompt Design
The evaluation prompt enforces strict schema con-
straints:
• JSON-only output
• continuous score per metric
• Textual justification field per metric
• Explicit case-awareness conditioning
Case-awareness includes:
• Prior turn context
• Error code presence
• Expected resolution trajectoryThis ensures evaluation is conditioned on en-
terprise troubleshooting flows rather than isolated
single-turn answers.
M Batch Evaluation Pipeline
The evaluation pipeline includes:
• Batched inference to reduce latency
• Automatic JSON schema validation
• Retry logic for malformed outputs
•Aggregation scripts for per-conversation scor-
ing
All metrics are computed per turn and then ag-
gregated to conversation-level summaries.
N Enterprise Failure Pattern Analysis
The framework reveals failure modes not captured
by generic RAG metrics:
•Correct document retrieval but incorrect reso-
lution sequencing
•Accurate issue identification without action-
able steps
•Structurally valid responses lacking contex-
tual grounding
•Over-generalized answers in error-code-
driven scenarios
These patterns are critical in enterprise environ-
ments where resolution alignment directly impacts
operational outcomes.
O Reproducibility and Deployment
Considerations
The framework is designed for deployment within
enterprise governance constraints:
• Deterministic evaluation configuration
• Auditable justification logs
• Batch-friendly architecture
• Model-agnostic scoring
All experiments were conducted with consistent
prompt templates and temperature settings to en-
sure comparability across models.
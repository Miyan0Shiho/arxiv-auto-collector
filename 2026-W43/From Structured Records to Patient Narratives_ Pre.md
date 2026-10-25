# From Structured Records to Patient Narratives: Preserving Clinical Alignment in LLM-Generated Health Summaries

**Authors**: Bayram Ayadi, Florian Geissler, Johanna Schmidhuber

**Published**: 2026-10-22 15:58:57

**PDF URL**: [https://arxiv.org/pdf/2610.31467v1](https://arxiv.org/pdf/2610.31467v1)

## Abstract
As wearable and digital health systems increasingly generate longitudinal physiological records, communicating these observations in patient-accessible language remains a significant challenge. Large Language Models (LLMs) offer a promising mechanism for translating structured clinical information into readable narratives, but this process may introduce semantic drift, where clinically precise meaning is distorted during simplification. We investigate evidence-grounded health summarization for menstrual health monitoring using a constrained generation framework that combines structured evidence conditioning with Retrieval-Augmented Generation (RAG) using a prompt-based constrained generation framework. We evaluate semantic alignment using Natural Language Inference (NLI), atomic factuality metrics, and readability measures. Across three LLMs and 4,500 generated summaries, explicit grounding constraints consistently improve clinical alignment, particularly by improving uncertainty preservation and, once faithfulness is evaluated against retrieved evidence, generally preserving or improving faithfulness, while effects on omission remain model-dependent. We also find that improved clinical alignment comes at the cost of increased reading difficulty and medical terminology. These findings highlight the need to jointly evaluate faithfulness, uncertainty preservation, and readability when developing patient-facing medical LLM systems.

## Full Text


<!-- PDF content starts -->

From Structured Records to Patient Narratives: Preserving Clinical
Alignment in LLM-Generated Health Summaries
Bayram Ayadi1,2Florian Geissler2Johanna Schmidhuber2
1University of Passau, Passau, Germany
2Fraunhofer Institute for Cognitive Systems IKS, Munich, Germany
ayadi02@ads.uni-passau.de,florian.geissler@iks.fraunhofer.de
Abstract
As wearable and digital health systems in-
creasingly generate longitudinal physiologi-
cal records, communicating these observations
in patient-accessible language remains a sig-
nificant challenge. Large Language Models
(LLMs) offer a promising mechanism for trans-
lating structured clinical information into read-
able narratives, but this process may introduce
semantic drift, where clinically precise mean-
ing is distorted during simplification.
We investigate evidence-grounded health sum-
marization for menstrual health monitoring us-
ing a constrained generation framework that
combines structured evidence conditioning
with Retrieval-Augmented Generation (RAG)
using a prompt-based constrained generation
framework. We evaluate semantic alignment us-
ing Natural Language Inference (NLI), atomic
factuality metrics, and readability measures.
Across three LLMs and 4,500 generated sum-
maries, explicit grounding constraints consis-
tently improve clinical alignment, particularly
by improving uncertainty preservation and,
once faithfulness is evaluated against retrieved
evidence, generally preserving or improving
faithfulness, while effects on omission remain
model-dependent. We also find that improved
clinical alignment comes at the cost of in-
creased reading difficulty and medical terminol-
ogy. These findings highlight the need to jointly
evaluate faithfulness, uncertainty preservation,
and readability when developing patient-facing
medical LLM systems.
1 Introduction
The growing integration of wearable and digital
health technologies into everyday life has created
an expanding stream of longitudinal physiologi-
cal data (spanning menstrual cycle phases, heart
rate variability (HRV), skin temperature, and sleep
quality) that is increasingly interpreted by patients
themselves rather than by clinicians. Communicat-
ing these observations in accessible lay language istherefore a task with direct implications for patient
safety and health literacy.
Large Language Models (LLMs) present a com-
pelling mechanism for this translation. Recent
work demonstrates that adapted LLMs can match
or exceed medical expert performance on stan-
dard clinical summarization benchmarks (Van Veen
et al., 2024), and the field has seen rapid advances
in factuality verification, constrained generation,
and longitudinal narrative synthesis. Semantic drift,
defined as the accumulation of semantic change as
medical concepts move from clinical to public dis-
course, has been shown to be greatest at precisely
this transition point (Karakkaparambil James et al.,
2026). Our work operationalizes this broader phe-
nomenon in the context of automated summariza-
tion, introducing a five-category clinical alignment
taxonomy and a constrained generation framework
to systematically measure and mitigate misalign-
ment across these dimensions.
Existing approaches to health text simplification
assume a semantically equivalent reference docu-
ment to simplify. Our setting is fundamentally dif-
ferent, generating patient-facing narratives directly
from structured physiological observations, where
faithfulness must be verified against raw evidence
rather than a reference text. This gap is particu-
larly consequential in menstrual health monitoring,
where cyclical signals, intrinsic uncertainty, and the
documented consequences of urgency miscalibra-
tion make evidence-grounded generation especially
critical.
This paper makes the following contributions:
•We formalize the structured-to-narrative sum-
marization task for longitudinal physiological
records and introduce an evaluation concept
using a taxonomy of five semantic drift con-
ditions:Faithfulness Violation,Epistemic
Misalignment,Diagnostic Inference Over-
reach,Omission Drift, andUrgency Mis-
1
arXiv:2610.31467v1  [cs.CL]  22 Oct 2026

alignment.
•We design and implement a framework for
the semi-synthetic generation of medically
grounded menstrual health data.
•We propose an LLM-based generation
pipeline combining explicit evidence-
grounding constraints, targeting each
alignment category in the taxonomy, with
RAG that supports simplification through
retrieved lay-term definitions.
•We systematically evaluate the clinical align-
ment of generated patient narratives along the
defined taxonomy and derive preliminary in-
sights on favorable configurations of such a
setup in menstrual health monitoring.
Taken together, this work positions trustworthy sim-
plification as a central and measurable challenge in
human-centered digital health.
2 Related Work
2.1 Clinical Summarization and
Lay-Language Simplification
Clinical summarization includes both expert-to-
expert tasks, such as radiology impressions and
discharge summaries, and expert-to-patient com-
munication, which requires lay-language simpli-
fication grounded in health literacy. (Van Veen
et al., 2024) show that adapted LLMs can match
or exceed human expert performance across mul-
tiple clinical summarization tasks, suggesting that
output fluency is less of a bottleneck than seman-
tic fidelity. In longitudinal settings, CLIN-SUMM
(D’souza et al., 2025) demonstrates the value of
temporally structured summarization, while (Fer-
razzi et al., 2026) show that supervised fine-tuning
and constrained decoding can allow small domain-
specific models to outperform much larger ones.
This motivates our use of explicit generation con-
straints, though we apply them at the prompting
level rather than through decoding-time control.
For patient-facing communication, simplifica-
tion is closely tied to health literacy and self-
management outcomes (Fitzpatrick, 2023). Re-
sources such as README (Yao et al., 2024) and
multi-stage simplification pipelines like S3 support
medical-to-lay translation, but prior work consis-
tently highlights a trade-off between readabilityand fidelity. Simplification can improve accessi-
bility while omitting or distorting clinically impor-
tant qualifiers, a pattern consistent with semantic
drift in the transition from evidence to narrative
(Karakkaparambil James et al., 2026). Recent work
addresses this tension through fidelity-readability
optimization (Verma et al., 2026), and reproductive
health further complicates simplification due to in-
direct or stigmatized language (Hua et al., 2026).
These findings underscore the need for evidence-
grounded generation.
However, existing benchmarks focus on summa-
rizing or simplifyingexistingclinical text. They
do not address generation directly from struc-
tured physiological observations, where faithful-
ness must be preserved against raw evidence rather
than a reference document.
2.2 NLI-Based Faithfulness Assessment and
Hallucination Verification
Natural Language Inference (NLI) (Bowman et al.,
2015; Williams et al., 2018) is widely used for
factuality assessment in summarization, especially
when combined with atomic fact decomposition
(Min et al., 2023; Kamoi et al., 2023). In medical
settings, this is particularly relevant because hal-
lucinations often reflect reasoning failures rather
than missing knowledge (Pal et al., 2023). Bench-
marks such as Med-HALT and MedHalu show that
instruction tuning can even worsen hallucination
behavior, motivating verification-based evaluation.
Clinical NLI was established by MedNLI (Ro-
manov and Shivade, 2018), but several limitations
remain. Clinical NLI datasets contain annotation
artifacts that can inflate performance (Herlihy and
Rudinger, 2021), NLI models degrade under do-
main shift (Jullien et al., 2024), and textual entail-
ment is not equivalent to factual support (Mor-Lan
and Levi, 2024). In addition, atomic-level verifi-
cation depends heavily on the consistency of fact
decomposition in clinical text (Munnangi et al.,
2025).
The most relevant paradigm is decompose-then-
verify. FActScore (Min et al., 2023) evaluates
generated text at the atomic-claim level against
trusted evidence, while AlignScore (Zha et al.,
2023) learns a broader alignment function across
NLI, QA, and IR tasks. Despite known limitations,
these claim-level methods are better suited than
surface metrics such as ROUGE for detecting the
subtle semantic drift central to our setting.
Overall, prior work has studied constrained gen-
2

eration, factuality verification, and longitudinal
summarization largely in isolation. To our knowl-
edge, no work has combined them for structured-
to-narrative, patient-facing summarization of wear-
able health data. Our paper addresses this gap.
3 Methodology
This work investigates whether evidence-grounded
and constrained language generation can improve
semantic alignment during the simplification of
longitudinal health observations into patient-facing
summaries. The framework targets factual consis-
tency, uncertainty calibration, and clinical faithful-
ness while maintaining accessibility for non-expert
users.
3.1 Task Definition
Given a structured set of physiological observations
derived from longitudinal health monitoring, the
task is to generate a patient-oriented natural lan-
guage summary that remains faithful to the under-
lying evidence, avoids unsupported medical claims,
preserves clinically relevant uncertainty, and is un-
derstandable to non-expert users. Formally, let
X={x 1, x2, . . . , x n}denote a structured physio-
logical record containing observed signals, symp-
tom annotations, and contextual metadata. The
objective is to generate a lay-language summary
Ysuch that the generated content remains seman-
tically aligned with Xwhile minimizing factual
distortion. The evaluation of this semantic align-
ment is detailed in Sec. 3.5.
3.2 Dataset Construction
Due to the limited availability of publicly accessi-
ble longitudinal menstrual-health datasets paired
with patient-oriented summaries, this study em-
ploys a semi-synthetic structured dataset gener-
ated via a causal directed acyclic graph (DAG)
that encodes known physiological dependencies
across the menstrual cycle. The generator produces
500 single-day observations, each drawn from a
stochastically determined cycle, following a bio-
logically motivated causal ordering: temporal struc-
ture (cycle and luteal-phase length, calibrated to
population data (Bull et al., 2019; Fehring et al.,
2006)) determines phase and cycle day; sleep qual-
ity and duration are conditioned on phase, reflect-
ing progesterone-mediated sleep disruption (Baker
and Driver, 2007); resting heart rate and HRV trend
are jointly conditioned on phase and sleep, captur-
ing both hormonal modulation (Tenan et al., 2014)and acute sympathetic override; SpO 2is sampled
independently as a homeostatically regulated sig-
nal; and symptoms are generated as downstream
consequences of these upstream states rather than
independent annotations.
To simulate real-world data sparsity, each sam-
ple is assigned one of five user profiles (mini-
mal_trackertoadvanced_wearable) that determine
which fields are observable, probabilistically avail-
able, or systematically absent, producing behav-
iorally mediated structured missingness. A confi-
dence score, derived from resulting data complete-
ness and phase clarity, provides uncertainty meta-
data for downstream generation. The full causal
record (pre-masking) is retained as ground truth
X, enabling evaluation of whether generated sum-
maries faithfully reflect available evidence without
hallucinating masked information.
Full details of the generative DAG and an exam-
ple record are provided in Appendix A.
3.3 System Overview
Our system takes each patient’s structured
menstrual-health record (fields such as cycle tim-
ing, symptoms, trends, and related clinical con-
text) and generates patient-facing summaries under
three generation regimes:Base,Constrained, and
RAG-Constrained. In the Base setting, the model
summarizes directly from the record with minimal
control; in the Constrained setting, the same input
is generated under explicit clinical communication
constraints, implemented as structured instructions
within the prompt (for example, factual ground-
ing, uncertainty handling, and nuance-preservation
rules, full prompts in Appendix B). Finally, in the
RAG-Constrained setting, those constraints are re-
tained while adding retrieved lay-term definitions
to support clearer, simplified explanations (see Ap-
pendix C for implementation details). Specifically,
the role of Retrieval-Augmented Generation (RAG)
in this setting is to facilitate text simplification us-
ing the README dataset, an extensive collection
of over 50,000 unique pairs of medical terms and
their lay definitions (Yao et al., 2024). Together,
these constraints enforce evidence-aware summa-
rization over open-ended medical interpretation.
3.4 Experimental Setup and Baselines
To systematically evaluate the impact of our
evidence-grounded generation framework, we
benchmarked the task across three distinct tiers
of Large Language Models. This selection allows
3

Figure 1: Overview of the pipeline and five-category semantic alignment evaluation framework.
us to observe how model scale affects clinical align-
ment and constraint adherence:
•GPT-4o: Representing a state-of-the-art,
large-scale proprietary model (OpenAI et al.,
2024).
•Qwen2.5:14B-Instruct: Representing a
highly capable, mid-sized open-weight model
(Qwen et al., 2025).
•Phi-4-mini-4B: Representing a lightweight,
efficient small language model (SLM) suitable
for low-latency or on-device health applica-
tions (Microsoft et al., 2025).
For our experiments, we sampled 500 records
from the semi-synthetic menstrual health dataset.
Each record was processed across the three mod-
els using three distinct generation configurations,
resulting in a total corpus of 4,500 patient-facing
summaries. The configurations evaluated include:
•Base(Vanilla LLM): The model directly
translates the structured physiological obser-
vations into a summary.
•Constrained: The model generates a sum-
mary utilizing explicit evidence-grounding
constraints (e.g., Uncertainty Preservation, Di-
agnostic Constraint) injected into the prompt,
but operates without Retrieval-Augmented
Generation (RAG).
•RAG-Constrained: The full pipeline combin-
ing non-parametric retrieval from a curated
medical knowledge base (medical term, lay
definition) with explicit prompt based clinical
constraints.3.5 Evaluation
We define all semantic alignment metrics so that
a score of zero indicates perfect alignment and
higher values indicate greater drift. Let Xdenote
the structured physiological record and Ythe gen-
erated patient-facing summary.
Faithfulness violation ( Mfaith).Each summary’s
precomputed atomic_facts are checked against
a textual premise formed by concatenating all
non-null fields of X(excluding sample_id ) into
field-value statements. Each claim ai∈A is tested
with pritamdeka/PubMedBERT-MNLI-MedNLI
(Deka et al., 2023) via the Hugging Face text-
classification pipeline; a claim is supported only if
labeledentailment:
For each atomic claim ai∈A , we query
pritamdeka/PubMedBERT-MNLI-MedNLI
via the Hugging Face text-classification
pipeline, using X as the premise and
aias the hypothesis. Let NLI(X, a i)∈
{entailment,neutral,contradiction} denote the
predicted label. We define the support indicator:
Isup(ai) =(
1ifNLI(X, a i) =entailment
0otherwise
(1)
Mfaith=1
|A||A|X
i=1(1−I sup(ai))(2)
Epistemic misalignment ( Mepist).A record
bears uncertainty if confidence_score<0.60
ormissing_fields is non-empty. Claims from
such records are uncertainty-sensitive if they con-
tain cycle- or interpretation-related keywords (e.g.,
4

Alignment Condi-
tionDefinition Example Metric Related Work
Faithfulness Viola-
tionUnsupported physiologi-
cal claims generated be-
yond the scope of the
structured evidenceStating elevated HRV when
only temperature data is
presentEq. 2 Faithfulness and factual-
ity(Maynez et al., 2020)
Epistemic mis-
alignmentAmplification of uncer-
tain observations into
definitive statements“Your cycle is irregular” in-
stead of “your cycle may be
irregular”Eq. 3 (Clark et al., 2025)
Diagnostic infer-
ence overreachImplication of pathology
unsupported by the under-
lying evidenceSuggesting endometriosis
from mild cramp annotationsEq. 4 (Zhu et al., 2026)
Omission Drift Removal of clinically rel-
evant nuance during sim-
plificationDropping confidence scores
or phase uncertainty from the
summaryEq. 5 (Oukelmoun et al., 2025)
Urgency Misalign-
mentUnnecessarily alarming
or over-reassuring lan-
guage“This is completely normal”
for an ambiguous HRV dropEq. 6
Table 1: Semantic alignment taxonomy covering five clinical alignment conditions, their definitions, illustrative
examples, formal metrics, and connections to related literature concepts.
phase,ovulation,trend,normal). A claim is defini-
tive if it contains no hedge markers (e.g.,may,
might,likely) and is asserted directly:
Mepist=1
|Aunc|X
a∈A uncI(IsDefinitive(a)).(3)
With Mepist= 0, no uncertainty-sensitive claims
exist.
Diagnostic inference overreach ( Mdiag).Unsup-
ported pathology mentions are detected via scis-
paCy en_ner_bc5cdr_md with medSpaCy nega-
tion handling; only non-negated DISEASE mentions
are retained as DY. A summary overreaches if any
extracted diagnosis is absent from the textualized
source record:
Mdiag=I(D Y̸⊆X).(4)
We also report a corrected version excluding
pregnancy andectopic pregnancy , which man-
ual review showed were typically explanatory
reproductive-context mentions rather than unsup-
ported diagnoses.
Omission drift ( Momit).For a fixed salient-field
set (cycle timing, phase, ovulation estimate, sleep,
resting heart rate, HRV trend, SpO 2, temperature
trend, symptoms), we generate a reference state-
ment per non-null field and test reversed entailment
with the same NLI model (summary as premise,
field statement as hypothesis). Unentailed fields
count as omitted:
Momit=#omitted salient fields
#non-null salient fields inX.(5)Urgency misalignment ( Murg).Unlike the claim-
level metrics above, urgency concerns perceived
tone rather than entailment. Each summary Yis
rated by an LLM judge (Claude-3.5-Haiku (An-
thropic, 2024) , temperature 0) on a continuous
scale from reassuring to alarming, independent of
whether that tone is clinically warranted:
Murg= Judge(Y)∈[0,1].(6)
Readability, lexical diversity, and jargon.We as-
sess the fidelity-accessibility trade-off along three
axes.
Grade-level readability: SMOG, FKGL, and ARI,
plus Dale-Chall (vocabulary familiarity, largely
length-independent); since the three grade-level
indices are highly correlated, FKGL is our pri-
mary endpoint, with SMOG and ARI as robustness
checks.
Lexical and syntactic complexity: MTLD (forward-
backward averaged, factor threshold 0.72) for lex-
ical variety, and a fixed spaCy parser for mean
dependency-tree depth and subordinate-clause ra-
tio, applied identically across conditions; word
counts are tracked to detect verbosity inflation.
Medical jargon: scispaCy with a UMLS linker iden-
tifies terms resolving to a documented UMLS con-
cept, reported as density (entities per 100 words),
raw entity count, and entities per sentence.
3.6 Statistical Procedure
For each metric we run paired Wilcoxon signed-
rank tests (Wilcoxon, 1992) , since the design is
5

fully matched and several metrics are not normally
distributed. We report the paired mean and median
difference, a paired Cohen’s d(mean difference
over the standard deviation of the paired differ-
ences), and the matched-pairs rank-biserial correla-
tion as a non-parametric effect size. Two families
of contrasts are tested separately: Within-model
comparisons across the three summary types, and
within-summary-type comparisons across the three
models. Because n= 500pairs per test, most con-
trasts reach statistical significance; we therefore
treat effect size, not the raw p-value, as the basis
for interpretation. p-values are corrected for mul-
tiple comparisons with the Benjamini-Hochberg
procedure, applied separately within each of the
two contrast families.
4 Results
We evaluated three models (GPT-4o, Phi-4-mini,
and Qwen2.5-14B) across three generation settings
(base ,constrained , and rag_constrained ) on
500 structured records, yielding 4,500 generated
summaries in total. We report the five semantic-
alignment metrics (Table 1), together with read-
ability, lexical-diversity, syntactic-complexity, and
jargon measures to characterize the fidelity-
accessibility trade-off. Because the grade-level
readability metrics are strongly intercorrelated, we
use FKGL as the primary grade-level endpoint and
treat SMOG and ARI as robustness checks; Dale-
Chall, MTLD, dependency depth, and subordinate-
clause ratio are interpreted as distinct constructs.
4.1 Clinical alignment
Faithfulness violation.Table 2 reports mean faith-
fulness violation rates across models and settings
(corrected for RAG-constrained as described in
§5.2). For all three models, Mfaithis lowest or
comparable in the RAG-constrained condition rela-
tive to constrained: GPT-4o improves further under
retrieval (0.186 →0.152), Qwen2.5-14B is essen-
tially unchanged (0.165 →0.162), and Phi-4-mini
shows a small increase (0.212 →0.220). All three
models remain well below their base levels. Pooled
across settings, mean Mfaithis 0.242 for base, 0.188
for constrained, and 0.178 for rag_constrained;
across models, Qwen2.5-14B performs best overall
(0.172), followed by GPT-4o (0.201) and Phi-4-
mini (0.235).
Epistemic misalignment.As shown in Table 2,
Mepist decreases monotonically from base toconstrained torag_constrained for all three
models, with the pooled mean falling from 0.297
to 0.263 to 0.248. GPT-4o achieves the lowest fi-
nal value (0.237); Phi-4-mini and Qwen2.5-14B
show the same monotonic pattern. Median Mepist
is 0.0 in every model-by-setting cell, so the mean
is driven by a smaller subset of problematic cases.
Notably, this improvement occurs despite the aver-
age number of uncertainty-sensitive claims rising
across settings (pooled: 1.17 →1.76→2.70), indi-
cating that explicit uncertainty-preservation instruc-
tions and retrieved explanatory support help models
state uncertain observations more cautiously rather
than more definitively.
Diagnostic inference overreach.The raw
diagnostic-overreach metric suggested substantial
monotonic increases under constraints and re-
trieval; manual inspection traced nearly all flagged
cases topregnancyorectopic pregnancymentions
used in fertility-window or ovulation explanations
rather than as unsupported diagnoses (discussed
further in §5.2). After excluding these two terms,
overreach collapses to near zero in every cell:
GPT-4o and Qwen2.5-14B fall to 0.000 in all
settings, while Phi-4-mini reaches only 0.004 in
baseand 0.006 inrag_constrained.
Omission drift.Omission drift shows more
heterogeneous behavior. GPT-4o improves
monotonically, from 0.219 ( base ) to 0.130
(rag_constrained ), with a median of 0.0 under
rag_constrained . Qwen2.5-14B performs best
under constrained generation (0.147) and rises
slightly under retrieval (0.167), remaining below its
base value (0.176). Phi-4-mini is the only model
that worsens monotonically, from 0.264 to 0.286.
Pooled across settings, omission drift declines only
modestly (0.220 →0.205→0.194), masking these
model-specific differences; we relate Phi-4-mini’s
reversal to its more discursive output style in §5.2.
Urgency misalignment.Mean Murgincreases
significantly from base toconstrained for all
three models (e.g., GPT-4o: 0.170 →0.227, d=
−0.61 ; allpFDR<10−5), then drops significantly
from constrained torag_constrained for every
model ( d= 0.37 –0.45, allpFDR<10−13), return-
ing GPT-4o and Phi-4-mini to base levels while
Qwen2.5-14B only partially recovers. Across set-
tings, GPT-4o is consistently the least alarming
model, though this gap narrows once retrieval is
added.
6

Model SettingM faith↓M epist↓M diag↓M omit↓M urg
GPT-4o Base 0.264 0.293 0.000 0.219 0.170
Constrained 0.186 0.243 0.000 0.195 0.227
RAG-Constrained 0.152 0.237 0.000 0.130 0.179
Qwen2.5-14B Base 0.190 0.299 0.000 0.176 0.247
Constrained 0.165 0.275 0.000 0.147 0.270
RAG-Constrained 0.162 0.251 0.000 0.167 0.199
Phi-4-mini Base 0.273 0.298 0.004 0.264 0.237
Constrained 0.212 0.270 0.000 0.273 0.263
RAG-Constrained 0.220 0.255 0.006 0.286 0.181
Table 2: Clinical alignment metrics by model and generation setting, ordered by model scale (large to small). Mfaith:
faithfulness violation rate (RAG-constrained: entailed by Xorretrieved context); Mepist: epistemic misalignment;
Mdiag: diagnostic inference overreach (corrected, excluding pregnancy/ectopic pregnancy mentions);M omit: omis-
sion drift;M urg: urgency misalignment. All metrics range[0,1]with lower values indicating better alignment.
4.2 Readability and stylistic trade-offs
Grade-level readability and syntactic complex-
ity.Across all three models, readability worsens
under added control: all three grade-level met-
rics increase monotonically or near-monotonically
from base toconstrained torag_constrained
(Table 6), with FKGL as the primary endpoint. For
GPT-4o, the largest shift occurs between base and
constrained (FKGL paired d= 1.39 ; ARI d=
1.22; Dale-Chall d= 1.23 ; allpFDR<10−70),
while the further increase from constrained to
rag_constrained is modest for FKGL ( d= 0.26 )
and negligible for ARI ( d=−0.08 , n.s. after
correction) and Dale-Chall ( d=−0.09 , signifi-
cant but trivial). Dependency-tree depth follows
the same pattern: a large increase from base to
constrained (d= 0.90 ) with essentially no fur-
ther change under retrieval ( d= 0.01 , n.s.). This
suggests explicit grounding constraints, rather than
retrieval itself, are the main driver of increased
reading difficulty. Table 6 shows the same quali-
tative pattern across models: Phi-4-mini produces
the highest grade-level outputs under all settings,
GPT-4o occupies the middle range, and Qwen2.5-
14B is typically lowest on FKGL and ARI once
constraints are applied.
Lexical diversity.Using MTLD (to assess lexi-
cal variety independent of length), GPT-4o shows
a clear decline from base torag_constrained ,
modest between base and constrained (d=
−0.26 ) and substantially larger between base and
rag_constrained (d=−0.63 ). Phi-4-mini de-
clines similarly, from a much higher starting point.
Qwen2.5-14B is comparatively stable, even ris-
ing slightly from base toconstrained before re-
turning near baseline under retrieval. The lexical-diversity cost of constrained generation is therefore
model-dependent rather than universal.
Medical jargon.Medical-jargon usage shows
a more nuanced trade-off than a single mea-
sure captures. As shown in Table 7, raw medi-
cal entity count rises consistently from base to
rag_constrained for every model, with large ef-
fects for GPT-4o ( constrained minus base :d=
1.89;rag_constrained minus base :d= 2.24 ).
By contrast, jargon density (entities per 100 words)
declines across settings for all models, since total
word count grows.
5 Discussion
5.1 Readability
Taken together, these results indicate that the
evidence-grounding mechanisms evaluated for
their effect on factual alignment in Section 3 carry a
measurable cost in accessibility. Grade-level read-
ability increases substantially once explicit con-
straints are applied, largely independent of whether
retrieval is also used, while lexical diversity de-
creases and domain-term mentions increase in ab-
solute terms even as their density per word declines.
This is consistent with the readability-fidelity ten-
sion noted in constrained clinical summarization
more broadly (Verma et al., 2026), and suggests
that the drift-reduction constraints introduced in
this work should be evaluated jointly against both
factual alignment and reading difficulty rather than
either in isolation. We treat this joint analysis, cor-
relating the per-sample drift metrics of Table 1
against the per-sample readability measures re-
ported here, as a direct extension of the present
results. One caveat applies to how we measure
readability: FKGL, ARI, and SMOG are based
7

on sentence length and syllable counts, not on
how conceptually difficult a word actually is. This
means a technical term like "ptosis" can score as
easier than an everyday word like "congratulations"
simply because it has fewer syllables. This matters
for our RAG results as FKGL increases modestly
from Constrained to RAG-Constrained (d=0.26)
despite stable sentence length, driven mainly by
near-doubled medical-term counts. This suggests
the extra difficulty these formulas detect comes
mainly from added vocabulary, not from harder
sentence structure, especially since RAG also sup-
plies a lay definition alongside each term. This
doesn’t remove the broader readability cost we re-
port above, but it suggests the part of that cost
caused specifically by RAG may be smaller than
the raw grade-level scores suggest.
5.2 Clinical Alignment
Table 2 reports all four clinical alignment metrics
across models and generation settings. Across
the four metrics, only epistemic misalignment im-
proves consistently across all three models as gen-
eration moves frombasetoConstrainedtoRAG-
Constrained. The other three metrics deviate for
distinct, explainable reasons.
Faithfulness under retrieval was underesti-
mated by the original formulation. Mfaith
verifies claims against the structured record X
alone, ignoring retrieved context. Because RAG-
constrained prompts instruct models to use re-
trieved lay-term definitions for explanations, valid
supporting content (e.g., a plain-language defini-
tion of “HRV”) is generated but flagged as un-
supported, inflating Mfaith without reflecting gen-
uine unfaithfulness. We correct this by treating
a claim as supported if entailed by Xorby the
retrieved context. Under this correction, mean
Mfaith forrag_constraineddrops significantly for
all three models relative to the original formula-
tion ( d= 0.76 –0.80, allp <10−40), and is now
significantlylowerthanconstrainedfor GPT-4o
(d= 0.284 ,p= 1.07×10−10) and statistically
indistinguishable for Phi-4-mini and Qwen2.5-14B
(p >0.15).
Diagnostic overreach is almost entirely a vocab-
ulary artifact. Its apparent rise under RAG is fully
explained by terms like "pregnancy" and "ectopic
pregnancy." Pregnancy is a physiological state and
not a true pathology that legitimately surfaces when
explaining fertility, which RAG actively supplies.
Removing these terms drops overreach to ≤0.6 per-cent for every model and condition, proving that
the models almost never suggest genuinely unsup-
ported pathologies. We recommend reporting these
corrected figures as the primary result and sepa-
rating reproductive contexts from the pathological
diagnostic vocabulary.
Omission drift is model-dependent: GPT-4o ben-
efits from the full pipeline, Qwen2.5-14B from
constraints alone, and Phi-4-mini uniquely wors-
ens with more grounding. Phi-4-mini’s regression
aligns with its readability profile; it adds discursive,
explanatory prose rather than restating specific field
values outright (e.g., "Resting HR: 71.3"). Because
the omission metric checks for direct statements of
salient fields, Phi-4-mini fails the strict entailment
check even when the underlying information is not
truly lost. This indicates the metric is sensitive to
surface realization and stylistic differences.
These results complicate a simple "more ground-
ing is better" narrative. Explicit constraints alone
account for most metric improvements. Retrieval’s
marginal contribution is genuinely positive for epis-
temic calibration and, once corrected, largely pos-
itive or neutral for faithfulness but its effect on
omission remains mixed, and its apparent cost for
diagnostic overreach is a vocabulary artifact rather
than a real effect.
6 Conclusion
We find that evidence-grounded constrained gen-
eration improves clinical alignment along specific
dimensions. The clearest and most consistent ef-
fect across all three models is reduced epistemic
misalignment. Retrieval further improves faithful-
ness for GPT-4o and is faithfulness-neutral for the
other two models, while its effects on omission and
urgency remain smaller and model-dependent. At
the same time, constrained and RAG-constrained
outputs become harder to read, often less lexically
diverse, and heavier in medical terminology.
These results suggest that safer health summa-
rization is not a single-metric problem. Systems
must be evaluated jointly for faithfulness, uncer-
tainty preservation, coverage, and readability. This
work provides practical guidance to tackle this chal-
lenge. More broadly, it contributes a structured-to-
narrative evaluation setting and a semantic align-
ment framework for studying how LLMs turn phys-
iological data into patient-readable explanations.
8

Limitations
Our study has several limitations that should inform
how these results are interpreted.
No human validation of automatic metrics.
All five clinical alignment metrics rely on auto-
matic tools: an NLI model (PubMedBERT-MNLI-
MedNLI) for faithfulness violation and omission
drift, a named entity recognizer with negation han-
dling for diagnostic inference overreach, and a sin-
gle LLM judge for urgency misalignment. While
these are established approaches in the factuality
literature, prior work has documented that clini-
cal NLI models contain annotation artifacts and
degrade under domain shift, and that textual entail-
ment does not always correspond to factual support.
We do not have a human-annotated subsample to
validate any of the five metrics against clinician
or expert judgment, and we treat this as the most
important direction for follow-up work.
Single LLM judge for urgency.Urgency mis-
alignment is scored by a single model (Claude 3.5
Haiku) at a fixed temperature, without inter-rater re-
liability checks, comparison to human raters, or an
ensemble of judges. Given that this metric concerns
tone rather than propositional content, it is more
subjective than the other four and may be more
sensitive to idiosyncrasies of the judge model.
Synthetic data without clinician review.Our
dataset is generated from a causal DAG that en-
codes documented physiological relationships be-
tween cycle phase, sleep, and autonomic measures,
and includes internal coherence checks (Appendix
A). The DAG structure and its parameters are our
own modeling choices, and neither the generated
records nor the associated ground truth have been
reviewed by a clinician for biological realism. We
note that this limitation is partially mitigated by
the nature of our evaluation: our alignment met-
rics measure whether a generated summary Yis
semantically faithful to its own source record X,
not whether Xitself is a biologically accurate de-
piction of the menstrual cycle. In this sense the task
is closer to a semantic consistency analysis than
a biological validity analysis, and the alignment
scores we report do not depend on the DAG being
an accurate model of physiology. Clinician review
of the DAG would therefore primarily affect the
realism of the input distribution rather than the va-
lidity of the alignment measurements themselves.
That said, this gap still matters for generalizationas real-world self-tracked data is likely to be nois-
ier, more idiosyncratic, and structured differently
than data produced by a hand-specified generative
model, and our results should be understood as
characterizing model behavior on this particular
semi-synthetic input distribution rather than on real
user data.
Narrow instantiation of retrieval.In our RAG-
Constrained setting, retrieval is limited to lay-term
definitions drawn from the README dataset and
is not used to supply additional clinical evidence
beyond the structured record. Our findings about
the marginal contribution of retrieval should there-
fore be understood as specific to this specific use
of RAG, and may not generalize to settings where
retrieval is used to bring in additional clinical con-
text.
No external baseline system.As discussed in
the main text, we are not aware of prior systems
that perform structured-to-narrative generation in
this domain, so we rely on internal ablations (Base,
Constrained, RAG-Constrained) rather than com-
parison to an external baseline. This limits our
ability to situate the absolute performance of our
pipeline relative to other possible approaches.
Domain and demographic scope.Our experi-
ments focus exclusively on menstrual health mon-
itoring using a semi-synthetic dataset. We do not
evaluate other physiological domains, and the gen-
erative model does not represent the full diversity
of menstrual cycle presentations, medical condi-
tions, or demographic variation present in real pop-
ulations. Findings may not transfer directly to
other health domains or to more clinically atyp-
ical records.
Ethical Considerations
Data and privacy.All experiments in this pa-
per use a semi-synthetic dataset generated from
a causal statistical model (Appendix A). No real
patient records, wearable device data, or person-
ally identifiable health information were collected,
accessed, or used at any stage of this work. This
eliminates direct privacy risk from our experiments,
though it also means our findings should not be
taken as validated on real-world data distributions.
Risk of misuse or overreliance.This work stud-
ies failure modes, including faithfulness violation,
epistemic misalignment, diagnostic inference over-
9

reach, omission drift, and urgency misalignment,
in LLM-generated health summaries. Our intent
is diagnostic and evaluative: to make these failure
modes measurable so that they can be mitigated,
not to present any of the three tested generation con-
figurations as ready for deployment in a real patient-
facing product. None of the metrics or pipelines
described here have been clinically validated, and
none should be used, as-is, to make or support real
health or diagnostic decisions. In particular, our
diagnostic inference overreach metric is intended
todetectunsupported pathology mentions for re-
search purposes; it is not a clinical safety filter and
should not be relied on as one.
Reproductive health context.Menstrual health
monitoring involves personal and sometimes stig-
matized health information. We chose this do-
main because cyclical signals, intrinsic physio-
logical uncertainty, and the documented conse-
quences of urgency miscalibration make it a partic-
ularly informative and consequential setting for
studying evidence-grounded generation, not be-
cause we treat it as a low-stakes proxy for health
tasks more broadly. We encourage readers to in-
terpret our results as evidence for the general diffi-
culty of structured-to-narrative health summariza-
tion, rather than as specific claims about menstrual
health applications being closer to solved than other
clinical domains.
Automatic and LLM-based evaluation.Sev-
eral of our metrics rely on automatic NLI models,
named entity recognition, and an LLM-as-judge
(for urgency misalignment). These are research
instruments for comparing generation configura-
tions at scale and have known limitations, includ-
ing domain shift sensitivity and lack of validated
correspondence to clinician judgment (see Limita-
tions). They should not be interpreted as clinically
validated safety metrics, and results derived from
them should not be used to certify any system as
safe for patient-facing use.
Downstream use.We release our dataset genera-
tion code (Appendix A) to support reproducibility
and further research into evidence-grounded health
summarization. We do not release, and did not
train, any model intended for deployment. We en-
courage future work building on this framework to
include clinician review and human evaluation of
generated summaries before considering any real-
world patient-facing application.Acknowledgments
We thank Reinhard Stolle and Annette Hautli-
Janisz for their valuable guidance, insightful dis-
cussions, and continuous support throughout this
project.
This work was supported by the Free State of
Bavaria within the DSgenAI project under Grant
No. RMF-SG20-3410-2-18-4.
References
Anthropic. 2024. Model card addendum: Claude 3.5
haiku and upgraded claude 3.5 sonnet. Technical
report, Anthropic.
Fiona C Baker and Helen S Driver. 2007. Circadian
rhythms, sleep, and the menstrual cycle.Sleep
medicine, 8(6):613–622.
Samuel R. Bowman, Gabor Angeli, Christopher Potts,
and Christopher D. Manning. 2015. A large anno-
tated corpus for learning natural language inference.
InProceedings of the 2015 Conference on Empiri-
cal Methods in Natural Language Processing, pages
632–642, Lisbon, Portugal. Association for Compu-
tational Linguistics.
Jonathan R Bull, Simon P Rowland, Elina Berglund
Scherwitzl, Raoul Scherwitzl, Kristina Gemzell
Danielsson, and Joyce Harper. 2019. Real-world
menstrual cycle characteristics of more than 600,000
menstrual cycles.NPJ digital medicine, 2(1):83.
Cecilia Castro-Diehl, Ana V Diez Roux, Susan Redline,
Teresa Seeman, Paula McKinley, Richard Sloan, and
Steven Shea. 2016. Sleep duration and quality in
relation to autonomic nervous system measures: The
multi-ethnic study of atherosclerosis (mesa).Sleep,
39(11):1927–1940.
Nicholas Clark, Hua Shen, Bill Howe, and Tanu Mitra.
2025. Epistemic alignment: A mediating framework
for user-LLM knowledge delivery. InSecond Con-
ference on Language Modeling.
Pritam Deka, Anna Jurek-Loughrey, and Deepak P.
2023. Multiple evidence combination for fact-
checking of health-related information. InThe 22nd
Workshop on Biomedical Natural Language Pro-
cessing and BioNLP Shared Tasks, pages 237–247,
Toronto, Canada. Association for Computational Lin-
guistics.
Valentina D’souza, Alaleh Azhir, Danielle F. Pace,
Samuel Friedman, Arash Nargesi, Tristan Naumann,
Christopher D. Anderson, Steven J. Atlas, Judy Hung,
and Mahnaz Maddah. 2025. Clin-summ: Tem-
poral summarization of longitudinal clinical notes.
medRxiv.
10

Richard J. Fehring, Mary Schneider, and Kathleen
Raviele. 2006. Variability in the phases of the men-
strual cycle.Journal of Obstetric, Gynecologic, &
Neonatal Nursing, 35(3):376–384.
Pietro Ferrazzi, Milica Cvjeticanin, Alessio Piraccini,
and Davide Giannuzzi. 2026. Is agentic rag worth
it? an experimental comparison of rag approaches.
Preprint, arXiv:2601.07711.
Patrick J Fitzpatrick. 2023. Improving health literacy
using the power of digital communications to achieve
better health outcomes for patients and practitioners.
Frontiers in digital health, 5:1264780.
Christine Herlihy and Rachel Rudinger. 2021. MedNLI
is not immune: Natural language inference artifacts
in the clinical domain. InProceedings of the 59th
Annual Meeting of the Association for Computational
Linguistics and the 11th International Joint Confer-
ence on Natural Language Processing (Volume 2:
Short Papers), pages 1020–1027, Online. Associa-
tion for Computational Linguistics.
Ashley Hua, Adya Daruka, Yang Hong, and Sharifa
Sultana. 2026. "openbloom": A question-based llm
tool to support stigma reduction in reproductive well-
being.Preprint, arXiv:2602.00243.
Mael Jullien, Marco Valentino, and André Freitas. 2024.
SemEval-2024 task 2: Safe biomedical natural lan-
guage inference for clinical trials. InProceedings of
the 18th International Workshop on Semantic Evalua-
tion (SemEval-2024), pages 1947–1962, Mexico City,
Mexico. Association for Computational Linguistics.
Ryo Kamoi, Tanya Goyal, Juan Diego Rodriguez, and
Greg Durrett. 2023. WiCE: Real-world entailment
for claims in Wikipedia. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 7561–7583, Singapore. As-
sociation for Computational Linguistics.
Charu Karakkaparambil James, Marcio Monteiro, and
Sophie Fellenz. 2026. Semantic echo pathways
(SEP): Tracing how medical language propagates
and transforms. InProceedings of the 1st Workshop
on Linguistic Analysis for Health (HeaLing 2026),
pages 55–66, Rabat, Morocco. Association for Com-
putational Linguistics.
Joshua Maynez, Shashi Narayan, Bernd Bohnet, and
Ryan McDonald. 2020. On faithfulness and factu-
ality in abstractive summarization. InProceedings
of the 58th Annual Meeting of the Association for
Computational Linguistics, pages 1906–1919, On-
line. Association for Computational Linguistics.
Microsoft, :, Abdelrahman Abouelenin, Atabak Ash-
faq, Adam Atkinson, Hany Awadalla, Nguyen Bach,
Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav
Chaudhary, Congcong Chen, Dong Chen, Dong-
dong Chen, Junkun Chen, Weizhu Chen, Yen-Chun
Chen, Yi ling Chen, Qi Dai, and 57 others. 2025.
Phi-4-mini technical report: Compact yet powerful
multimodal language models via mixture-of-loras.
Preprint, arXiv:2503.01743.Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis,
Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettle-
moyer, and Hannaneh Hajishirzi. 2023. FActScore:
Fine-grained atomic evaluation of factual precision
in long form text generation. InProceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing, pages 12076–12100, Singa-
pore. Association for Computational Linguistics.
Guy Mor-Lan and Effi Levi. 2024. Exploring factual
entailment with NLI: A news media study. InPro-
ceedings of the 13th Joint Conference on Lexical
and Computational Semantics (*SEM 2024), pages
190–199, Mexico City, Mexico. Association for Com-
putational Linguistics.
Monica Munnangi, Akshay Swaminathan, Jason Alan
Fries, Jenelle A Jindal, Sanjana Narayanan, Ivan
Lopez, Lucia Tu, Philip Chung, Jesutofunmi Omiye,
Mehr Kashyap, and Nigam Shah. 2025. FactEHR:
A dataset for evaluating factuality in clinical notes
using LLMs. InProceedings of the 10th Machine
Learning for Healthcare Conference, volume 298 of
Proceedings of Machine Learning Research. PMLR.
OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal,
Lama Ahmad, Ilge Akkaya, Florencia Leoni Ale-
man, Diogo Almeida, Janko Altenschmidt, Sam Alt-
man, Shyamal Anadkat, Red Avila, Igor Babuschkin,
Suchir Balaji, Valerie Balcom, Paul Baltescu, Haim-
ing Bao, Mohammad Bavarian, Jeff Belgum, and
262 others. 2024. Gpt-4 technical report.Preprint,
arXiv:2303.08774.
Achir Oukelmoun, Nasredine Semmar, Gaël de Chal-
endar, Clement Cormi, Mariame Oukelmoun, Eric
Vibert, and Marc-Antoine Allard. 2025. Detecting
omissions in LLM-generated medical summaries. In
Proceedings of the 2025 Conference on Empirical
Methods in Natural Language Processing: Industry
Track, pages 325–337, Suzhou (China). Association
for Computational Linguistics.
Ankit Pal, Logesh Kumar Umapathi, and Malaikannan
Sankarasubbu. 2023. Med-HALT: Medical domain
hallucination test for large language models. InPro-
ceedings of the 27th Conference on Computational
Natural Language Learning (CoNLL), pages 314–
334, Singapore. Association for Computational Lin-
guistics.
Qwen, :, An Yang, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan
Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Yang, Jiaxi Yang, Jingren Zhou, and 25 oth-
ers. 2025. Qwen2.5 technical report.Preprint,
arXiv:2412.15115.
Alexey Romanov and Chaitanya Shivade. 2018.
Lessons from natural language inference in the clini-
cal domain. InProceedings of the 2018 Conference
on Empirical Methods in Natural Language Process-
ing, pages 1586–1596, Brussels, Belgium. Associa-
tion for Computational Linguistics.
11

Mohaned Shilaih, Brianna M Goodale, Lisa Falco, Flo-
rian Kübler, Valerie De Clerck, and Brigitte Leeners.
2018. Modern fertility awareness methods: wrist
wearables capture the changes in temperature asso-
ciated with the menstrual cycle.Bioscience reports,
38(6):BSR20171279.
Matthew S. Tenan, R. Matthew Brothers, Andrew J.
Tweedell, Anthony C. Hackney, and Lisa Griffin.
2014. Changes in resting heart rate variability across
the menstrual cycle.Psychophysiology, 51(10):996–
1004.
Dave Van Veen, Cara {Van Uden}, Louis Blanke-
meier, Jean-Benoit Delbrouck, Asad Aali, Christian
Bluethgen, Anuj Pareek, Malgorzata Polacin, {Ed-
uardo Pontes} Reis, Anna Seehofnerová, Nidhi Ro-
hatgi, Poonam Hosamani, William Collins, Neera
Ahuja, {Curtis P} Langlotz, Jason Hom, Sergios Ga-
tidis, John Pauly, and {Akshay S} Chaudhari. 2024.
Adapted large language models can outperform med-
ical experts in clinical text summarization.Nature
Medicine, 30(4):1134–1142. © 2024. The Author(s),
under exclusive licence to Springer Nature America,
Inc.
Deepika Verma, Daison Darlan, and Rammohan
Mallipeddi. 2026. Multi-objective optimization of
large language model summaries for clinical text nar-
ratives.IEEE Access, 14:13597–13609.
Frank Wilcoxon. 1992. Individual comparisons by rank-
ing methods. InBreakthroughs in statistics: Method-
ology and distribution, pages 196–202. Springer.
Adina Williams, Nikita Nangia, and Samuel R. Bow-
man. 2018. A broad-coverage challenge corpus for
sentence understanding through inference. InPro-
ceedings of the 2018 Conference of the North Amer-
ican Chapter of the Association for Computational
Linguistics: Human Language Technologies, Volume
1 (Long Papers), pages 1112–1122, New Orleans,
Louisiana. Association for Computational Linguis-
tics.
Zonghai Yao, Nandyala Siddharth Kantu, Guanghao
Wei, Hieu Tran, Zhangqi Duan, Sunjae Kwon,
Zhichao Yang, and Hong Yu. 2024. README:
Bridging medical jargon and lay understanding for
patient education through data-centric NLP. InFind-
ings of the Association for Computational Linguistics:
EMNLP 2024, pages 12609–12629, Miami, Florida,
USA. Association for Computational Linguistics.
Yuheng Zha, Yichi Yang, Ruichen Li, and Zhiting Hu.
2023. AlignScore: Evaluating factual consistency
with a unified alignment function. InProceedings
of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 11328–11348, Toronto, Canada. Association
for Computational Linguistics.
Shanshan Zhu, Han Zhang, J. Doris Chi, Subigya Nepal,
and Koustuv Saha. 2026. Causal stories from sen-
sor traces: Auditing epistemic overreach in llm-generated personal sensing explanations.Preprint,
arXiv:2605.08590.
A Synthetic Data Generation: Causal
DAG Specification
This appendix details the generative model used to
produce the semi-synthetic menstrual health dataset
described in Section 3.2. The generator implements
a timeline-first causal DAG in which each variable
is sampled conditional only on its causal parents,
ensuring that inter-variable correlations arise from
shared physiological mechanisms rather than arbi-
trary co-occurrence.
A.1 Causal Graph Structure
The generation follows a strict topological order-
ing:
Cycle Length →Luteal Length →Phase Bound-
aries→Cycle Day→Phase
,→Sleep Quality→Sleep Duration
,→RHR, HRV(conditioned on Phase +
Sleep)
,→SpO2(independent)
,→Temperature Trend→Symptoms
A.2 Node 1: Cycle Length
Cycle length is drawn from a lognormal distribu-
tion, reflecting the right-skewed population dis-
tribution documented in Bull et al. (2019) ( N >
600,000cycles):
Lcycle∼Lognormal(µ= 3.355, σ= 0.155),
Lcycle∈[21,50]
This parameterization yields a median of approx-
imately 28.7 days, a mean of approximately 29.3
days, and standard deviation of approximately 4.4
days.
A.3 Node 2: Luteal Phase Length
The luteal phase is sampled from a tight normal
distribution, encoding the well-established finding
that inter-cycle variability is attributable almost
entirely to the follicular phase, with luteal length
remaining approximately constant (Fehring et al.,
2006):
Lluteal∼ N(µ= 14.0, σ= 1.5), L luteal∈[10,17]
12

A.4 Node 3: Phase Boundary Derivation
Phase boundaries are computed deterministically
rather than assigned via hard-coded day ranges:
dovulation =L cycle−L luteal
Menstrual: days1−5
Follicular: days6−(d ovulation −1)
Ovulatory: dayd ovulation
Luteal: days(d ovulation + 1)−L cycle
A cycle day is then sampled uniformly: d∼
Uniform{1, . . . , L cycle}.
A.5 Node 4–5: Sleep (Phase-Conditioned)
Sleep quality is drawn from a categorical distribu-
tion conditioned on the current phase, reflecting
progesterone-mediated sleep architecture disrup-
tion (Baker and Driver, 2007):
SleepQuality∼Categorical(P(quality|phase))
Table 3: Conditional probability table:
P(sleep quality|phase)
Phase Poor Moderate Good
Menstrual 0.35 0.40 0.25
Follicular 0.08 0.32 0.60
Ovulatory 0.12 0.40 0.48
Luteal 0.32 0.43 0.25
Sleep duration is then sampled conditional on
quality from a truncated normal:
Duration∼ N(µ q, σq),clipped to[min q,max q]
where the parameters (µq, σq,min q,max q)are
(4.5,0.6,3.5,5.5) for poor, (6.3,0.5,5.5,7.2) for
moderate, and(7.8,0.6,7.0,9.2)for good quality.
A.6 Nodes 6–7: Autonomic Biometrics (Phase
+ Sleep Override)
Resting heart rate is drawn from a phase-specific
baseline with an additive penalty for poor sleep:
RHR=N(µ phase, σphase)|{z }
hormonal baseline
+Uniform(4,8)·1[sleep=poor]| {z }
acute sympathetic penaltyThis encodes both chronic luteal-phase elevation
(Tenan et al., 2014) and acute sleep-deprivation
effects on sympathovagal balance (Castro-Diehl
et al., 2016).
Table 4: RHR baseline parameters by phase (bpm)
Phaseµ σMin Max
Menstrual 63.0 3.0 57 71
Follicular 60.0 2.5 55 66
Ovulatory 64.0 3.0 58 72
Luteal 67.0 3.5 61 78
HRV trend is sampled categorically from
phase-specific probabilities, with a hard override
when sleep quality is poor (forcing sympathetic-
dominant states):
P(HRV trend|poor sleep) =(
0.55decreasing
0.45erratic
A.7 Node 8: SpO 2(Independent)
Blood oxygen saturation is sampled independently
of all other nodes, reflecting tight chemoreceptor-
mediated homeostatic regulation in healthy individ-
uals:
SpO2∼ N(97.8,0.8),SpO2∈[95.0,100.0]
A.8 Node 9: Temperature Trend
Temperature trend is sampled categorically from
phase-specific probabilities, encoding the well-
documented biphasic thermal shift driven by post-
ovulatory progesterone thermogenesis (Shilaih
et al., 2018).
A.9 Node 10: Symptoms (Causal Placeholder)
Symptoms are generated as logical consequences
of upstream states rather than as independent draws.
The causal triggers include:
•Fatigue: triggered by poor sleep OR elevated
RHR (>70bpm)
•Headache: triggered by poor sleep AND de-
clining/erratic HRV
•Cramps, back pain: phase-gated (menstrual)
•Bloating, breast tenderness, mood changes:
phase-gated (luteal)
•Mood changes: also triggered by autonomic
stress (poor sleep + declining HRV)
13

This ensures that symptom co-occurrence pat-
terns reflect plausible physiological mechanisms
rather than arbitrary correlation.
A.10 User Profile Masking
After complete record generation, a masking layer
simulates five distinct user archetypes that deter-
mine field observability:
Table 5: User profile distribution and data access char-
acteristics
ProfileNMean Fields
Minimal tracker 75∼2–3
App logger 100∼4–5
Basic wearable 125∼6–8
Advanced wearable 125∼9–12
Inconsistent user 75∼3–5
Each profile specifies fields asalways present,
probabilistically available(with per-field inclusion
probabilities), ornever observed. This produces
structured missingness patterns that simulate the
heterogeneous data completeness characteristic of
real-world self-tracking behavior.
A.11 Confidence Score
A data-driven confidence score is computed from
the ratio of present diagnostic fields to total possi-
ble fields, modulated by phase clarity (menstrual
phase being most identifiable due to bleeding, fol-
licular least identifiable due to ambiguous signals):
C= 0.30 + 0.50·|present diagnostic fields|
|total diagnostic fields|
+ 0.20·clarity(phase) +ϵ
where ϵ∼ N(0,0.03) and clarity values are
{menstrual: 0.90, luteal: 0.80, ovulatory: 0.75,
follicular: 0.70}.
A.12 Causal Coherence Validation
To verify that the DAG produces internally consis-
tent data, we report two key checks on the gener-
ated dataset:
•Sleep→RHR: Mean RHR under poor sleep
is elevated by +6–8 bpm compared to good
sleep, consistent with the encoded sympa-
thetic penalty.
•Sleep→HRV override: When sleep is poor,
100% of HRV trends are either “decreasing”
or “erratic,” confirming the hard override
mechanism.A.13 Example Record
An example ground-truth record produced by the
generator, prior to user-profile masking, is shown
below.
{
"cycle_day": 22,
"phase": "luteal",
"temperature_trend": "elevated",
"hrv_trend": "decreasing",
"symptoms": ["fatigue", "cramps"],
"sleep_quality": "poor",
"resting_hr_bpm": 71.3,
"confidence_score": 0.81,
"missing_fields": ["spo2_pct",
"sleep_duration_hrs"]
}
BSystem Prompts and Hyperparameters
This appendix provides the exact system prompts
and generation configurations used across the three
experimental settings described in Section 3.4. For
all generation runs, the model temperature was set
toT= 0.4 to balance deterministic fidelity with
linguistic fluency.
B.1 Base (Vanilla LLM) Setting
System Prompt:
“You are a helpful health assistant. Summarize
the following user tracking data into a friendly,
patient-facing health summary using simple lan-
guage. ”
User Prompt Template:
USER DATA:
{user_data}
Return only the summary text.
B.2 Constrained Setting
System Prompt:
“You are a highly precise clinical summarization
assistant. Your task is to translate structured, lon-
gitudinal physiological data into an accessible,
lay-language patient narrative. You must strictly
follow the constraints below:
INPUT DATA:
- Structured Physiological Record ( X):
{user_data}
STAGE-GATED GENERATION CON-
STRAINTS:
1.Evidence Consistency: Only write claims
directly supported by the provided record. Do not
mention, assume, or infer missing metrics.
2.Uncertainty Preservation: Use hedged
language when confidence is low or evidence is
incomplete.
3.Diagnostic Restraint: Do not infer or suggest
any diagnosis, disease, or pathology.
4.Nuance Preservation: Keep important
14

qualifiers like phase context and explicit trends.
5.Tone Calibration: Keep tone supportive and
balanced; avoid panic or over-reassurance.
OUTPUT FORMAT:
Return only a patient-facing summary with:
- Your Current Status
- What your data is showing”
B.3 RAG-Constrained (Proposed) Setting
System Prompt:
“You are a highly precise clinical summarization
assistant. Your task is to translate structured, lon-
gitudinal physiological data into an accessible,
lay-language patient narrative. You must strictly
follow the constraints below:
INPUT DATA:
- Structured Physiological Record ( X):
{user_data}
RETRIEVED CONTEXT:
Use this context only as support for explanations.
Do not contradict the record.
{retrieved_context}
STAGE-GATED GENERATION CON-
STRAINTS:
1.Evidence Consistency: Only write claims
directly supported by record and context.
2.Uncertainty Preservation: Use hedged
language when confidence is low.
3.Diagnostic Restraint: Do not infer or suggest
any diagnosis or pathology.
4.Nuance Preservation: Keep important
qualifiers and explicit trends.
5.Tone Calibration: Keep tone supportive,
avoiding panic/over-reassurance.
OUTPUT FORMAT:
Return only a patient-facing summary with:
- Your Current Status
- What your data is showing
- Symptom Context”
C Retrieval-Augmented Generation
Details
This appendix specifies the retrieval component
used in the RAG-Constrained setting. Retrieval
plays a deliberately narrow role: it supplies lay-
term definitions that support simplification, rather
than injecting new clinical evidence. All grounding
definitions are drawn from the README dataset
(Yao et al., 2024), represented as(medical term,
lay definition)pairs, where the term field is in-
dexed for retrieval and the definition field is re-
turned as grounding context. Prior to indexing,
we drop rows with a missing term or definition
and remove exact duplicate pairs after case- and
whitespace-normalization, so that a single defini-
tion cannot occupy multiple retrieval slots; only the
deduplicated, non-null rows are indexed.We use a sparse TF–IDF retriever with cosine
similarity. At this corpus size, TF–IDF is fully lo-
cal (requiring no model download or additional
network dependency), fast to build and query,
and provides a transparent, inspectable match-
ing signal, while leaving a clear upgrade path to
dense retrieval without changing the surrounding
pipeline. The term index is built with an sklearn
TfidfVectorizer using English stop-word re-
moval and an n-gram range of (1,2) ; combining
unigram and bigram features allows multi-word
terms such as“menstrual cramps”to match on
shared unigrams while still retaining some phrase-
level signal. For a given query q, we compute
cosine similarity against all indexed terms, rank
candidates in descending order, and return at most
k= 3 definitions, retaining only matches with simi-
larity at least 0.12. This similarity floor treats weak
lexical overlap as “no match” rather than forcing
spurious grounding, and any duplicate definitions
surviving to the ranking stage are skipped so that
the injected context contains distinct entries. The
concatenated text forms the retrieval query, and the
top-kretrieved definitions are formatted as a bul-
leted RETRIEVEDCONTEXTblock and inserted
into the RAG-Constrained prompt (Appendix B).
The prompt further constrains how retrieved con-
tent may be used: definitions serve only as support
for explanations and wording, and must not contra-
dict or extend the structured recordX.
D Readability metrics
15

Model Metric Base Constrained RAG-constrained
GPT-4o FKGL 7.38 9.93 10.31
GPT-4o ARI 8.35 10.97 10.83
GPT-4o Dale-Chall 9.67 10.60 10.54
Phi-4-mini FKGL 10.63 12.18 12.32
Phi-4-mini ARI 12.63 14.13 14.29
Phi-4-mini Dale-Chall 9.84 10.44 10.38
Qwen2.5-14B FKGL 7.64 9.72 10.39
Qwen2.5-14B ARI 8.07 10.66 11.64
s Qwen2.5-14B Dale-Chall 10.07 11.01 10.92
Table 6: Mean grade-level readability scores by model and generation setting.
16

Appendix E: Medical Jargon-Related Metrics
Figure 2: Distribution of jargon density, raw medical entity counts, and entities per sentence across generation
conditions (base, constrained, rag_constrained) for all evaluated models.
Model Summary Type Jargon Med. Ent. Ent./Sent. Sent.
GPT-4o base 19.26 16.63 3.51 4.80
Phi-4 base 18.91 24.28 5.59 4.97
Qwen base 21.90 18.38 3.44 5.34
GPT-4o constrained 18.92 26.07 3.92 6.67
Phi-4 constrained 17.65 33.61 4.54 7.65
Qwen constrained 19.31 26.26 3.63 7.45
GPT-4o rag-constr. 18.07 32.29 3.78 8.65
Phi-4 rag-constr. 17.37 40.96 4.57 9.14
Qwen rag-constr. 18.37 32.44 3.63 9.04
Table 7: Summary of medical jargon-related metrics (mean values).
17

F Metric correlations
Figure 3: Pooled pairwise correlations ( r) across seven
readability, lexical, and syntactic-complexity metrics
computed over all 4,500 generated summaries. Grade-
level indices are strongly intercorrelated (r≥0.87).
18
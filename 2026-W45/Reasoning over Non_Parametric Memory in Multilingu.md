# Reasoning over Non-Parametric Memory in Multilingual LLMs: A Cognitive-Based Analysis of Relational Knowledge

**Authors**: Zohreh Abbasi Liasi, Batoul Hammoud, Anas Madkoor, Ehsaneddin Asgari

**Published**: 2026-11-04 16:11:26

**PDF URL**: [https://arxiv.org/pdf/2611.06715v1](https://arxiv.org/pdf/2611.06715v1)

## Abstract
Retrieval-augmented generation (RAG) assumes that language models can reason over relations stated in their context rather than fall back on associations memorized during pretraining. We study this evidence-conditioned relational reasoning in isolation, with the facts supplied directly in the prompt rather than retrieved. \textsc{Rel-Bloom} is a controlled multilingual diagnostic of 450 multiple-choice questions built from 30 Wikidata relations; the same source relations are instantiated at five Bloom-inspired levels (Remember, Understand, Apply, Analyze, Evaluate) and under three veracity conditions (Factual, Counterfactual, Neutralized). Across 43 models, six languages, and two multilingual settings (226,800 predictions), three findings stand out. First, accuracy drops sharply from Apply to Analyze in every evaluated open-weight model series, reaching 47.2 percentage points for the Large Granite bucket, but not for either Gemini model. Second, Evaluate items remain a bottleneck even for models near ceiling on the other task types: Gemini~3.1 Pro averages 99.5\% on the other four levels but 60.7\% on Evaluate. Third, item- and relation-level difficulty rankings are consistently preserved across languages, indicating a difficulty structure partly shared across languages within this benchmark. Across knowledge conditions, Counterfactual contexts are generally harder than Factual ones, while removing entity identities can improve higher-level performance in some model regimes. Greater model capability therefore does not by itself ensure reliable reasoning over supplied relational evidence. Relevant resources are available at: https://huggingface.co/spaces/QCRI/non-parametric-memory-relational-reasoning.

## Full Text


<!-- PDF content starts -->

Published as a conference paper at COLM 2026
Reasoning over Non-Parametric Memory in Multilingual
LLMs: A Cognitive-Based Analysis of Relational Knowledge
Zohreh Abbasi Liasi∗†Batoul Hammoud∗†‡Anas Madkoor†Ehsaneddin Asgari†§
†Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University
‡Universitat Pompeu Fabra
/envel⌢peeasgari@hbku.edu.qa
Abstract
Retrieval-augmented generation (RAG) assumes that language models
can reason over relations stated in their context rather than fall back
on associations memorized during pretraining. We study this evidence-
conditioned relational reasoning in isolation, with the facts supplied di-
rectly in the prompt rather than retrieved. REL-BLOOMis a controlled
multilingual diagnostic of 450 multiple-choice questions built from 30
Wikidata relations; the same source relations are instantiated at five
Bloom-inspired levels (Remember, Understand, Apply, Analyze, Evalu-
ate) and under three veracity conditions (Factual, Counterfactual, Neu-
tralized). Across 43 models, six languages, and two multilingual set-
tings (226,800 predictions), three findings stand out. First, accuracy
drops sharply from Apply to Analyze in every evaluated open-weight
model series, reaching 47.2 percentage points for the Large Granite
bucket, but not for either Gemini model. Second, Evaluate items re-
main a bottleneck even for models near ceiling on the other task types:
Gemini 3.1 Pro averages 99.5% on the other four levels but 60.7% on
Evaluate. Third, item- and relation-level difficulty rankings are consis-
tently preserved across languages, indicating a difficulty structure partly
shared across languages within this benchmark. Across knowledge con-
ditions, Counterfactual contexts are generally harder than Factual ones,
while removing entity identities can improve higher-level performance
in some model regimes. Greater model capability therefore does not by
itself ensure reliable reasoning over supplied relational evidence. Rel-
evant resources are available at:https://huggingface.co/spaces/QCRI/
non-parametric-memory-relational-reasoning.
1 Introduction
Large language models are increasingly deployed in knowledge-intensive settings, most
notably retrieval-augmented generation (RAG), where a model must reason over facts
supplied in its context rather than rely solely on information acquired during pretraining
(Lewis et al., 2020). A fundamental challenge in evaluating such systems is separating an-
swers grounded in the supplied evidence from answers driven by memorized associations:
a model may answer correctly by drawing on parametric knowledge about a named entity,
or answer incorrectly because that knowledge conflicts with the relation stated in context
(Longpre et al., 2021). Existing benchmarks rarely operationalize this distinction directly,
so high accuracy does not by itself show whether a model follows the supplied evidence.
End-to-end RAG evaluations further confound this question with retrieval quality. We
therefore remove retrieval and supply the relevant fact directly in the prompt, measuring
accuracy on evidence-conditioned relational questions as a diagnostic of one component
on which RAG systems depend.
*Equal contribution.
§Corresponding author.
1
arXiv:2611.06715v1  [cs.CL]  4 Nov 2026

Published as a conference paper at COLM 2026
Bloom’s revised taxonomy (Krathwohl, 2002) offers a principled framework for character-
izing the cognitive complexity of tasks, organizing cognitive processes into six hierarchical
levels: Remember, Understand, Apply, Analyze, Evaluate, and Create. Originally designed
to classify human learning objectives, the taxonomy has been adopted in NLP and LLM re-
search to evaluate models across varying depths of reasoning (Fei et al., 2024; Cao et al.,
2024; Abootorabi et al., 2026). A critical assumption underlying such work remains insuf-
ficiently tested, however: whether accuracy on Bloom-inspired task types, designed from
a taxonomy of human learning objectives, follows the ordering the taxonomy implies. In
particular, stronger models may not show a monotonic decline in accuracy across levels,
and different level transitions may be associated with different patterns of error.
Multilingual evaluation adds a further dimension to this challenge. Benchmarks such as
XTREME (Hu et al., 2020) and MEGA (Ahuja et al., 2023) have established cross-lingual
performance baselines, yet multilingual evaluations have largely focused on whether mod-
els can perform tasks across languages. Less attention has been paid to whether the un-
derlying structure of model behavior itself is stable across languages: whether the same
items and relations remain difficult, whether models exhibit similar sensitivity to knowl-
edge format, and whether cognitive difficulty profiles and interference from memorized
associations reflect shared model behavior or language-specific artifacts.
We address these gaps simultaneously with REL-BLOOM, a multilingual diagnostic bench-
mark for evaluating how language models reason over supplied relational knowledge.
REL-BLOOMcontains 450 multiple-choice questions grounded in 30 curated Wikidata rela-
tions (Wang et al., 2021). Each question is instantiated under three veracity conditions: Fac-
tual, in which real entities are paired with the correct relation; Counterfactual, in which one
entity is replaced with an incorrect real entity while preserving the relation; and Neutral-
ized, in which entity identities are replaced with typed abstract placeholders. This design
allows us to vary the degree to which parametric entity associations can assist or inter-
fere with reasoning over the supplied context. Questions span five levels adapted from
Bloom’s revised taxonomy: Remember, Understand, Apply, Analyze, and Evaluate, with
Create omitted because it is not naturally compatible with controlled multiple-choice eval-
uation.
We evaluate 43 models: 41 open-weight models from seven model families, analyzed as
eight model series (Qwen2.5 Base, Qwen2.5 Instruct, Qwen3, Qwen3.5, Gemma, LLaMA,
Granite, and DeepSeek, with the Qwen2.5 family split into Base and Instruct series), to-
gether with two closed-weight Gemini models. Evaluation covers six languages, Arabic,
Chinese, English, Hindi, Persian, and Spanish, and two multilingual settings, yielding
226,800 predictions. Our analysis considers model scale, Bloom cognitive level, knowledge
veracity, relation usage frequency, and language. Matched accuracy contrasts are evalu-
ated using relation-clustered inference over the 30 source relations, while cross-language
stability is assessed using dedicated correlation-based procedures.
Contributions.(i)Benchmark and experimental design.REL-BLOOMinstantiates the
same 30 usage-stratified Wikidata relations at five Bloom-inspired levels and under three
matched veracity conditions (450 English MCQs), translated into six languages under two
evaluation settings, with ten pre-specified hypotheses tested by relation-clustered infer-
ence. (ii)Findings on Bloom levels and knowledge conditions.Accuracy drops sharply
from Apply to Analyze in every open-weight model series, but not for Gemini, while Eval-
uate remains a bottleneck even for models near ceiling on the other levels; Counterfactual
contexts are generally the hardest condition, and in many open-weight regimes removing
entity identities helps more at Analyze and Evaluate than at Remember and Understand.
Scaling generally improves accuracy, with series-dependent trajectories. (iii)Multilingual
stability.Item- and relation-level difficulty rankings are consistently positively correlated
across the six languages. This suggests that REL-BLOOMcaptures a difficulty structure that
is partly shared across languages, despite differences in language-specific performance.
2

Published as a conference paper at COLM 2026
2 Related Work
Our work sits at the intersection of three active research threads: cognitive-complexity
benchmarking of LLMs, context faithfulness under knowledge conflicts, and the evaluation
of reasoning in retrieval-augmented and knowledge-graph settings.
2.1 Bloom’s Taxonomy as a Cognitive Complexity Framework for LLMs
Bloom’s revised taxonomy (Krathwohl, 2002) has become a widely adopted scaffold for
stratifying the cognitive demands placed on language models, moving evaluation beyond
aggregate accuracy toward a diagnosis of where competence breaks down. Early adoption
appeared in domain-specific benchmarks such as LawBench (Fei et al., 2024) and scien-
tific evaluation suites such as SciEval (Sun et al., 2024) and SciAssess (Cai et al., 2025).
Subsequent work extended this to law (Dong et al., 2025), medicine (Zhou et al., 2025),
software engineering (Ma et al., 2025), and biographical reasoning (Cekinmez et al., 2025),
consistently revealing that performance degrades at higher cognitive levels. StructEval
(Cao et al., 2024) generates test instances automatically for all six levels, while Khoram-
far et al. (2026) report accuracy drops of up to 70% as tasks ascend the hierarchy, echo-
ing findings from Todorovikj et al. (2025), who apply Bloom’s framework to characterize
tasks in knowledge-graph benchmarks. BloomWise (Zoumpoulidi et al., 2025) reveals a
non-monotonic pattern in mathematical reasoning: models score higher on Apply than
on Understand; we do not reproduce this ordering in the relational setting. Raimondi &
Gabbrielli (2026) further argue, via mechanistic probing, that Bloom’s levels correspond to
distinguishable internal representations.
Gap.None of these benchmarks jointly control (i) matched items across Bloom levels, (ii)
factual/counterfactual/entity-neutralized conditions of the same relation, and (iii) multi-
ple languages. REL-BLOOMbrings these dimensions together on a single, item-matched
design.
2.2 Context Faithfulness Under Knowledge Conflicts
A parallel line of work studies whether models ground responses in supplied context or
default to memorized knowledge. Longpre et al. (2021) formalize this as an entity-level
knowledge conflict, measuring models’ tendency to favor parametric answers over coun-
terfactual context. Li et al. (2023) demonstrate that large models exhibit poor controllability
in conflict scenarios and introduce Knowledge-Aware Fine-Tuning (KAFT), incorporating
counterfactual and irrelevant contexts to improve robustness. Basmov et al. (2024) show
that faithfulness further degrades for hypothetically or modally framed contexts, while
Meng et al. (2022) provide mechanistic evidence linking this behavior to localized factual
associations stored in mid-layer feed-forward modules. Liu et al. (2025) introduce a gener-
ative framework for relational reasoning that recreates classic cognitive-science paradigms,
such as transitive inference and the symbolic distance effect, and report that several human
reasoning biases reappear in LLMs.Gap.Prior work characterizes when parametric asso-
ciations override supplied context but rarely varies cognitive demand on thesamerelation.
REL-BLOOMadds a Neutralized condition, which removes entity-identity cues without
introducing a conflict, and traces how parametric interference scales with Bloom level.
2.3 RAG Reasoning and Knowledge-Graph QA
The original RAG framework (Lewis et al., 2020) demonstrated that combining parametric
memory with non-parametric retrieval improves factuality on knowledge-intensive tasks.
Pan et al. (2024) survey the landscape of LLM–KG integration, and GraphRAG variants
(Thompson et al., 2025) leverage entity-relation graphs for multi-hop traversal and asso-
ciative retrieval. Sarthi et al. (2024) organize documents into hierarchical summaries for
improved retrieval. Zhang et al. (2025) report that the average factual correctness rate
across 16 popular KGQA datasets is only 57%, and Lin et al. (2026) propose HybridRAG-
Bench to control for parametric memorization. Todorovikj et al. (2025) specifically examine
3

Published as a conference paper at COLM 2026
Benchmark construction
450 English MCQs
from 30 curated Wikidata relations
RememberUnderstandApplyAnalyzeEvaluate
5 Bloom levels
3 veracity conditions
Factual, Counterfactual,
Neutralized
Same 30 relations at every level
and condition:30×5×3=450Multilingual evaluation
6 languages
Arabic, Chinese, English,
Hindi, Persian, Spanish
2 evaluation settings
Hybrid cross-lingual,
Fully translated
226,800 predictions
43 models in fully translated
+ 41 (open-weight) in hybridModels & analysis
43 models
Open-weight
(7 families, 8 series)
Qwen2.5 (Base, Instruct), Qwen3,
Qwen3.5, Gemma, LLaMA,
Granite, DeepSeek
Closed-weight (Gemini)
Gemini 3 Flash, Gemini 3.1 Pro
Analysis dimensions
Scale, Bloom level, Veracity,
Relation usage frequency, Language
Figure 1: Overview of REL-BLOOM. The same 30 curated Wikidata source relations (10 per
relation usage frequency tier) are each instantiated at all five Bloom-inspired levels (shown
as a pyramid) and under all three veracity conditions, yielding 30×5×3=450 English
items; this full crossing allows level and condition effects to be compared on matched re-
lations. Items are then evaluated in six languages under two multilingual settings. The
43 models comprise 41 open-weight models from seven model families, analyzed as eight
model series because Qwen2.5 Base and Qwen2.5 Instruct are treated separately, plus two
Gemini models.
how Bloom’s taxonomy can characterize KG tasks, identifying gaps in coverage of higher-
order reasoning.Gap.Existing RAG/KGQA benchmarks measure whether the retrieved
context is used, but not how well the model understands the specific relation carried by that
context. REL-BLOOMtargets this by evaluating matched relations across factual, counter-
factual, and neutralized instantiations of the same triple.
3 Methods
3.1 Benchmark Overview
We introduce REL-BLOOM, a compact diagnostic benchmark for sentence-level under-
standing of Wikidata-style relations. Each item consists of a fact sentence, a multiple-choice
question with four answer options, and metadata describing the Bloom level, veracity con-
dition, and relation usage tier. REL-BLOOMspans five Bloom-inspired cognitive levels, Re-
member, Understand, Apply, Analyze, and Evaluate, with 90 questions per level, totalling
450 questions. We omit the sixth level (Create) because it requires open-ended generation
that is incompatible with a controlled multiple-choice format. Within each Bloom level,
questions are distributed equally across three veracity conditions (Factual, Counterfactual,
and Neutralized, 30 questions each), and each set of 30 is further balanced across three
relation usage tiers (High, Medium, and Low, 10 questions each).
Figure 1 summarizes the design.
3.2 Relation Selection and Usage Tiers
We draw candidate relations from Wikidata5M (Wang et al., 2021) and stratify them into
Low, Medium, and High usage tiers using MediaWiki property-frequency counts as a
proxy for relation prevalence. These counts approximate how often a relation is used in
Wikidata; they are not estimates of how often the relation appears in model pretraining
4

Published as a conference paper at COLM 2026
Table 1: Bloom-level operationalization in REL-BLOOM.
Bloom level Operationalization in REL-BLOOM
Remember Extract the value explicitly stated in the fact.
Understand Select a paraphrase preserving the relation semantics; distractors express different
relations or attributes.
Apply Apply an abstract relation schema to a new statement with the same subject–
relation–object structure and meaning.
Analyze Discriminate the target relation from near-miss alternatives (semantically related
or structurally similar but logically incorrect).
Evaluate Select the strongest justification grounded in the defining semantic criterion of the
relation; distractors are plausible but non-definitional.
data or of how strongly it is memorized, and we interpret relation usage frequency results
(H6) accordingly. We manually curate 10 relations per tier, yielding 30 source relations. The
complete sampling, frequency transformation, and stratification procedure is provided in
Appendix A.4.
3.3 Triple Extraction and Manual Curation
For each candidate relation, we extract one representative factual triple(h,p,t)and man-
ually curate a final set of 30 relations, with 10 from each usage tier. Curation prioritizes
semantic clarity, domain diversity, and non-overlapping entities, while excluding sensitive
or excessively technical relations. Each selected relation contributes one source triple that
is reused across Bloom levels and veracity conditions, prioritizing controlled comparison
over broad relation coverage. Relation-level results therefore describe these 30 relation in-
stances; effects attributed to a relation may partly reflect the particular entities and fact
chosen for it. Further extraction and curation details are provided in Appendix A.4.
3.4 Fact Verbalization and Question Generation
For each extracted triple(h,p,t), GPT-51verbalizes the fact and assists in generating level-
specific multiple-choice questions. Table 1 summarizes the operationalization of the five
Bloom levels.
The levels differ in the operation a question requires, and therefore also in how questions
and distractors are constructed. Analyze distractors are near-miss relations that can be re-
jected by checking them against the stated fact, whereas Evaluate distractors are plausible
statements about the relation that are not its defining criterion and cannot be rejected in
this way. Differences in question construction and distractor difficulty may therefore con-
tribute to accuracy differences between levels. We interpret level-wise gaps as differences
in accuracy on these operationalized task types, not as direct measurements of distinct
cognitive abilities.
All generated questions, answer options, and answer keys are manually reviewed for fac-
tual correctness, preservation of the intended relation, answer uniqueness, and distrac-
tor plausibility. As an additional validation pass, Gemini and Grok independently flag
disagreements with the provided answers; all flagged cases are manually inspected, and
ambiguous items are revised and re-evaluated. Additional quality-control details are pro-
vided in Appendix A.5.
3.5 Veracity Conditions
To separate the contributions of entity-specific prior knowledge, evidence-following, and
relational abstraction, we construct three matched veracity conditions for each Bloom level,
1We accessed GPT-5 in early 2026. All verbalized facts were subsequently verified manually by
the authors.
5

Published as a conference paper at COLM 2026
Table 2: Veracity variants in REL-BLOOM, generated with GPT-5 and manually verified.
Variant Construction rule What it tests
Factual Verbalize the extracted
triple(h,p,t)as a sentence-
level fact.Baseline relation under-
standing with real entity
surface forms.
Counterfactual Edit exactly one argument
of(h,p,t), preserving
coarse entity type and
relation context.Evidence-following under
conflict with parametric
priors.
Neutralized Replace entity mentions
with typed placehold-
ers (e.g.,[PERSON 1],
[SPORT 1]) while preserv-
ing the question template.Relation abstraction with
reduced entity-prior short-
cuts.
summarized in Table 2. The variants preserve the underlying relation and Bloom-level
question structure, enabling controlled comparisons across Factual, Counterfactual, and
Neutralized conditions. The Counterfactual condition is designed to conflict with knowl-
edge a model is likely to hold about well-known entities; we do not directly measure
whether each model has memorized each source fact, so Counterfactual effects indicate
sensitivity to likely prior-knowledge conflict rather than verified memorization.
3.6 Multilingual Evaluation Settings
REL-BLOOMis constructed in English and evaluated in Arabic, Chinese, English, Hindi,
Persian, and Spanish under two settings. In thehybrid cross-lingual setting, the fact and
question remain in English while the answer options are translated into the target lan-
guage, isolating cross-lingual answer selection from a shared English stimulus. In thefully
translated setting, the fact, question, and options are all presented in the target language
while preserving the relation, Bloom-level operation, veracity condition, answer key, and
option order. We analyze the two settings separately.
Reproducibility.Model checkpoints (Appendix A.1), benchmark construction (Ap-
pendix A.4), quality control (Appendix A.5), and hypothesis-specific test statis-
tics, permutation/cluster-bootstrap procedures, and the Holm correction family (Ap-
pendix A.2) are documented. The complete benchmark, translations, prompt templates,
decoding configurations, and per-model prediction logs will be released upon publication.
3.7 Hypothesis Testing Framework
Hypothesis catalog.We evaluate ten pre-specified (not publicly preregistered) hypothe-
ses spanning model scale, Bloom-level difficulty, veracity, relation usage frequency, and
cross-language stability, listed in Table 3. For scale-dependent analyses, open-weight mod-
els are grouped into Small (<7B), Medium (≥7B and<14B), and Large (≥14B) buckets;
Gemini models are reported individually because their parameter counts are undisclosed.
Checkpoints within a bucket and across buckets also differ in training data, recipe, and in
some cases architecture (Appendix A.1), so size-bucket comparisons are descriptive rather
than causal estimates of scaling effects.
Estimation and inference.For H1–H5, H7, and H8, matched contrasts are aggregated
within the 30 source relations and evaluated using one-sided relation-cluster sign-flip
tests with 95% relation-cluster bootstrap intervals. H6 is evaluated using an observed
Low/Medium/High spread below 5 pp together with a±5 pp equivalence margin for the
three pairwise usage contrasts; Medium and Large are the confirmatory buckets, while
Small is reported exploratorily. For cross-language stability, H9 summarizes the 15 pair-
wise item-level Spearman correlations and evaluates their mean using a one-sided sign-flip
6

Published as a conference paper at COLM 2026
Table 3: The ten hypotheses tested in REL-BLOOM, grouped by theme.
Theme ID Statement
ScaleH1Diminishing returns: Small→Medium gain exceeds
Medium→Large.
H2Monotonic scaling: Medium>SmallandLarge>Medium.
Bloom levelH3Apply>Analyze accuracy; a “cliff” is marked when the lower 95%
CI on the drop exceeds 10 pp.
H8Evaluate underperforms the mean of the other four Bloom levels.
VeracityH4Ordering: Neutralized>Factual>Counterfactual.
H5Factual–Counterfactual gap is larger at Analyze/Evaluate than at
Remember/Understand.
H7Neutralized-over-Factual advantage is larger at Analyze/Evaluate
than at Remember/Understand.
Usage H6Relation usage frequency has a practically negligible effect (ob-
served Low/Medium/High spread<5 pp; equivalence assessed
within a±5 pp margin).
LanguageH9Item-level difficulty rankings are preserved across the six languages
(mean of 15 pairwise Spearman correlations).
H10Relation-level difficulty rankings are preserved across languages
(relation-identity permutation test).
test, whereas H10 uses a relation-identity permutation test in which relation correspon-
dences are shuffled across languages and the full mean-correlation statistic is recomputed.
Holm correction is applied within hypothesis-specific families, typically the S/M/L tests
within each series; full definitions and correction families are given in Appendix A.2.
4 Results
Table 4 summarizes hypothesis support across the fully translated and hybrid cross-lingual
settings; the full per-series, per-bucket matrices are given in Appendix A.3 (Tables 6 and 7).
The most consistent findings are cross-language stability (H9 –H10 ), the Apply-to-Analyze
drop in open-weight models (H3 ), the Evaluate bottleneck (H8 ), and Small-to-Medium
scaling (H2 );H1 andH4 –H7 depend more on model series and size bucket.
4.1 Scaling Behavior
Accuracy generally increases with model size (H2 ). In the fully translated setting, Medium
significantly exceeds Small in all eight open-weight series, while Large significantly ex-
ceeds Medium in seven; DeepSeek is the only exception. The same broad pattern holds
in the hybrid cross-lingual setting. For example, Qwen2.5 Instruct gains 23.8 percentage
points from Small to Medium and 6.1 points from Medium to Large.
Diminishing returns (H1 ) are less universal, holding in five of eight series in the fully
translated setting and six of eight in the hybrid cross-lingual setting. LLaMA and Gran-
ite show larger upper-transition gains, while DeepSeek supports (H1 ) only in the hybrid
cross-lingual setting and slightly declines from Medium to Large when fully translated.
Qwen2.5 Base and Instruct show similar scaling trends.
4.2 Bloom-Level Patterns
Apply exceeds Analyze across all eight open-weight series in both settings (H3 ), but the
magnitude of the drop varies sharply with series and size bucket. A “cliff” denotes a sta-
tistically supported drop whose lower 95% relation-cluster bootstrap bound exceeds 10
percentage points. At Large, Granite exhibits a 47.2-point drop, whereas Qwen3.5 drops
7

Published as a conference paper at COLM 2026
Table 4: Summary of hypothesis support (Holm-corrected; full matrices in Tables 6–7).
Units differ between blocks and are stated in the block headers and cells. The Gemini
column counts supported models (fully translated setting only).
Open-weight series (of 8) Gemini (of 2)
ID Hypothesis Fully transl. Hybrid Fully transl.
Per-bucket tests: series significant in all 3 buckets (in≥1 bucket)
H9 Item-level cross-language stability 8 (8) 8 (8) 2
H10 Relation-level cross-language stability 8 (8) 8 (8) 2
H3 Apply>Analyze 7 (8) 6 (8) 0
H8 Evaluate bottleneck 6 (8) 5 (8) 2
H7 Neutralized gain larger at high levels 3 (8) 6 (8) 0
H5 F–CF gap larger at high levels 0 (1) 0 (2) 2
H4 Neutralized>Factual>Counterfactual 0 (1) 0 (5) 0
Other units
H2 Monotonic scaling (per transition) M>S: 8, L>M: 7 M>S: 8, L>M: 7 —
H1 Diminishing returns (series-level test) 5 series 6 series —
H6 Negligible usage effect (equivalence) 0 series 0 series 0
only 7.2 points. Neither Gemini model exhibits the Apply-to-Analyze accuracy drop ob-
served across the evaluated open-weight model series. Across open-weight series, the cliff
is neither scale-invariant nor monotonic with model size.
Evaluate presents a different pattern (H8 ). Gemini 3.1 Pro averages approximately 99.5%
across the other four levels but only 60.7% on Evaluate, a gap of about 39 percentage points
for a model that shows no Apply-to-Analyze drop at all. The Evaluate gap appears in
all eight open-weight series and both Gemini models (Table 4). The two task types thus
show different model-dependent accuracy patterns (§5); as noted in §3.4, these gaps are
differences between operationalized task types and may partly reflect item and distractor
construction.
4.3 Veracity Effects
The full Neutralized>Factual>Counterfactual ordering (H4 ) is common directionally
but statistically supported in only a few size buckets and in no series as a whole. Counter-
factual is generally the weakest condition, whereas Neutralized does not consistently ex-
ceed Factual and reverses in several high-capability models, including Qwen3.5, Gemma,
and both Gemini models.
The Factual–Counterfactual gap (H5 ) also depends on cognitive demand. Its expected
widening from lower to higher Bloom levels appears directionally in several open-weight
series but is statistically supported in only a few size buckets. The effect is strongest for
Gemini, where both models show significant widening despite near-ceiling performance
on the lower-complexity levels.
4.4 Neutralization and Cognitive Demand
H7compares the Neutralized-over-Factual advantage at Analyze and Evaluate with that at
Remember and Understand. In the fully translated setting, the high-complexity advantage
is significantly larger in the Small bucket for seven of eight open-weight series and remains
present in several Medium and Large regimes.
The effect weakens or reverses in some stronger series: it remains substantial for Qwen2.5
Base and Instruct, becomes smaller for Qwen3, and is absent or reversed for several of the
strongest models, including Gemini 3.1 Pro. Support forH7 is therefore recurrent but not
universal, varying across model series and size buckets.
8

Published as a conference paper at COLM 2026
4.5 Relation Usage Frequency
H6receives limited support. Although several Medium and Large buckets exhibit ob-
served Low/Medium/High spreads below five percentage points, no open-weight series
satisfies the stronger interval-based equivalence criterion in the fully translated setting.
Gemini 3.1 Pro also shows a small observed spread, whereas Gemini 3 Flash lies at the
five-point boundary under the rounded result. We therefore regard the influence of re-
lation usage frequency as inconclusive rather than absent. Because the tiers reflect Wiki-
data property counts, this result says nothing about pretraining exposure or memorization
strength.
4.6 Cross-Language Stability
Item-level difficulty rankings (H9 ) show consistently positive cross-language stability
across every model series and size bucket in both settings and for both Gemini models. Sig-
nificance alone does not convey strength: in the fully translated setting, the mean pairwise
correlation is approximatelyρ=0.54 for Gemini 3 Flash, a moderate level of agreement,
andρ=0.74 for Gemini 3.1 Pro, a strong one.
Relation-level difficulty rankings (H10 ) show the same pattern: all series, size buckets, and
both Gemini models exhibit significantly positive cross-language stability. Thus, differ-
ences in aggregate accuracy coexist with a difficulty structure that is partly shared across
languages. Because all language versions derive from the same English items and trans-
lated templates, shared item construction may contribute to these correlations, as may the
single triple per relation. The results establish stability for the evaluated relation instances,
not language-independent reasoning ability in general.
5 Analysis
5.1 How Analyze and Evaluate Items Differ
The two task types show different model-dependent accuracy patterns. The Apply-to-
Analyze drop varies with series and size and is absent for Gemini, whereas the Evaluate
gap persists for every series and both Gemini models. Their difficulty is therefore not
captured by a single smooth progression across levels, although this alone does not show
that they tap different underlying abilities: a harder task can remain hard after an easier
one saturates.
The operationalization in Table 1 offers one account that is consistent with this pattern,
though our experiments do not isolate it. Analyze distractors are near-miss relations that
are inconsistent with the stated fact, so an item can in principle be solved by checking each
option against the evidence. Evaluate distractors are consistent with the fact but are not
the defining criterion of the relation, so such checking does not eliminate them.
An equally important alternative is distractor plausibility: Evaluate distractors are by con-
struction more plausible, and the gap may reflect discrimination among harder alternatives
rather than a distinct operation. Separating these accounts requires items that vary distrac-
tor plausibility within a fixed level. “Definitional” criteria are also partly conventional,
so some Evaluate errors may reflect reasonable disagreement with the key despite man-
ual review and independent model cross-checks (Appendix A.5). We therefore report the
Analyze–Evaluate contrast as a difference between operationalized task types, not between
cognitive capabilities.
5.2 The Effect of Entity Cues Depends on Task Complexity
Averaged over Remember and Understand, Neutralized contexts usually underperform
Factual contexts across model-series and size-bucket combinations, suggesting that entity
names can provide useful cues for simpler tasks. At Analyze and Evaluate, the pattern
9

Published as a conference paper at COLM 2026
Table 5: Representative errors consistent with memory competition. In the Factual condi-
tion, the model selects a salient entity-associated answer rather than the relation required
by the question.
Entity Wrong answer Correct relational answer
Apollo 11 Returned safely to Earth Recovered by USS Hornet
Bohemian Rhapsody A six-minute song Tonality is B-flat major
Amanita muscaria Red cap with white spots Spore print color is white
often reverses, consistent with entity-specific associations becoming more disruptive as
relational demands increase.
This crossover underlies H7. One possibility is that familiar entities help locate the stated
fact at lower levels but supply competing associations when the task concerns the relation
itself. The Neutralized advantage shrinks in several higher-performing series, consistent
with stronger models being less affected by such competition. Because this comparison is
exploratory and rests on one source triple for each of 30 relations, we neither interpret it as
a scaling law nor claim it generalizes to other triples of the same relations.
5.3 Errors Consistent with Memory Competition
We inspect matched Neutralized–Factual pairs for which Qwen2.5-32B-Instruct succeeds
under Neutralized input but fails under Factual input. The Factual errors often select a
salient attribute or related fact associated with the named entity rather than the relation re-
quested by the question. Table 5 shows representative examples. Although these cases do
not establish an internal causal mechanism, they are consistent with entity-specific para-
metric associations competing with the relational evidence supplied in context.
6 Conclusions
We presented REL-BLOOM, a controlled multilingual diagnostic of evidence-conditioned
relational reasoning over matched relations, Bloom-inspired task types, and veracity con-
ditions. Its results identify limitations in evidence-conditioned relational reasoning that
may affect the reliability of downstream RAG systems.
They suggest three concrete practices for evaluating RAG systems. First, report accuracy
separately by the operation a question requires: a system that reliably checks retrieved re-
lations against the question (Analyze) may still fail to identify which retrieved property
justifies an answer (Evaluate), and aggregate accuracy hides this. Second, include coun-
terfactual and entity-neutralized variants of retrieved passages, since accuracy on factual
passages alone conflates evidence-following with prior knowledge about the entities in-
volved. Third, since difficulty rankings transfer across languages here, a translated item
set can serve as a first multilingual screen, though language-specific accuracy must still be
measured directly.
Limitations and future work.REL-BLOOMfavors controlled comparison over breadth.
With one source triple per relation, the 30 relation clusters give limited power for small
effects (notably H6) and for generalizing to unseen triples of the same relations. We do
not measure end-to-end RAG with noisy retrieval or open-ended generation. Level-wise
gaps may reflect item construction; memorization is not measured; size comparisons are
descriptive; shared templates may inflate cross-language correlations; and the error anal-
ysis uses one checkpoint. Future work can add triples per relation and vary distractor
plausibility within levels.
10

Published as a conference paper at COLM 2026
Ethics Statement
In accordance with COLM 2026 policy, we disclose thatClaude,GPT-5, andGeminiwere
used to assist in rewriting and improving manuscript drafts across revision rounds. All
scientific content, experimental design, and conclusions are the authors’ own responsibil-
ity.
References
Mohammad Mahdi Abootorabi, Omid Ghahroodi, Anas Madkoor, Marzia Nouri, Do-
ratossadat Dastgheib, and Ehsaneddin Asgari. Almieyar-oryx-BloomBench: A bilin-
gual multimodal benchmark for cognitively informed evaluation of vision-language
models. In Maria Liakata, Viviane P . Moreira, Jiajun Zhang, and David Jurgens (eds.),
Findings of the Association for Computational Linguistics: ACL 2026, pp. 28404–28436, San
Diego, California, United States, July 2026. Association for Computational Linguis-
tics. ISBN 979-8-89176-395-1. doi: 10.18653/v1/2026.findings-acl.1416. URLhttps:
//aclanthology.org/2026.findings-acl.1416/.
Kabir Ahuja, Harshita Diddee, Rishav Hada, Millicent Ochieng, Krithika Ramesh, Prachi
Jain, Akshay Nambi, Tanuja Ganu, Sameer Segal, Mohamed Ahmed, Kalika Bali, and
Sunayana Sitaram. MEGA: Multilingual evaluation of generative AI. InConference on
Empirical Methods in Natural Language Processing, pp. 4232–4267. Association for Compu-
tational Linguistics, 2023. doi: 10.18653/v1/2023.emnlp-main.258.
Victoria Basmov, Yoav Goldberg, and Reut Tsarfaty. LLMs’ reading comprehension is
affected by parametric knowledge and struggles with hypothetical statements.arXiv
preprint arXiv:2404.06283, 2024. doi: 10.48550/arXiv.2404.06283.
Hengxing Cai, Xiaochen Cai, Junhan Chang, Sihang Li, Lin Yao, Wang Changxin, Zhifeng
Gao, Hongshuai Wang, Li Yongge, Mujie Lin, et al. SciAssess: Benchmarking LLM profi-
ciency in scientific literature analysis. InFindings of the Association for Computational Lin-
guistics: NAACL 2025, pp. 2335–2357. Association for Computational Linguistics, 2025.
doi: 10.18653/v1/2025.findings-naacl.125.
Boxi Cao, Mengjie Ren, Hongyu Lin, Xianpei Han, Feng Zhang, Junfeng Zhan, and Le Sun.
StructEval: Deepen and broaden large language model assessment via structured evalu-
ation. InFindings of the Association for Computational Linguistics: ACL 2024, pp. 5300–5318.
Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.findings-acl.
314.
Jasin Cekinmez, Omid Ghahroodi, Saad Fowad Chandle, Dhiman Gupta, and Ehsaneddin
Asgari. ADAM: A diverse archive of mankind for evaluating and enhancing LLMs in
biographical reasoning.arXiv preprint arXiv:2509.22991, 2025. doi: 10.48550/arXiv.2509.
22991.
Nguyen Tien Dong, Minh-Anh Nguyen, Thanh Dat Hoang, Nguyen Tuan Ngoc, Dao
Xuan Quang Minh, Phan Phi Hai, Nguyen Thi Ngoc Anh, Dang Van Tu, and Binh Vu.
VLegal-Bench: Cognitively grounded benchmark for vietnamese legal reasoning of large
language models.arXiv preprint arXiv:2512.14554, 2025. doi: 10.48550/arXiv.2512.14554.
Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou, Zhuo Han, Alan Huang, Songyang
Zhang, Kai Chen, Zhixin Yin, Zongwen Shen, Jidong Ge, and Vincent Ng. LawBench:
Benchmarking legal knowledge of large language models. InProceedings of the 2024
Conference on Empirical Methods in Natural Language Processing, pp. 7933–7962. Associa-
tion for Computational Linguistics, 2024. doi: 10.18653/v1/2024.emnlp-main.452. URL
https://aclanthology.org/2024.emnlp-main.452/.
Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin
Johnson. XTREME: A massively multilingual multi-task benchmark for evaluating cross-
lingual generalization. InInternational Conference on Machine Learning, pp. 4411–4421,
2020.
11

Published as a conference paper at COLM 2026
Ali Khoramfar, Ali Ramezani, Mohammad Mahdi Mohajeri, Mohammad Javad Dousti,
Majid Nili Ahmadabadi, and Heshaam Faili. DEEPQUESTION: Systematic generation
of real-world challenges for evaluating LLMs performance. InProceedings of the Language
Resources and Evaluation Conference, pp. 11451–11460. European Language Resources As-
sociation (ELRA), 2026.
David R. Krathwohl. A revision of Bloom’s taxonomy: An overview.Theory Into Practice,
41(4):212–218, 2002. doi: 10.1207/s15430421tip4104 2.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al. Retrieval-
augmented generation for knowledge-intensive NLP tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pp. 9459–9474, 2020.
Daliang Li, Ankit Singh Rawat, Manzil Zaheer, Xin Wang, Michal Lukasik, Andreas Veit,
Felix X. Yu, and Sanjiv Kumar. Large language models with controllable working mem-
ory. InFindings of the Association for Computational Linguistics: ACL 2023, pp. 1774–1793.
Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.findings-acl.
112.
Junhong Lin, Bing Zhang, Song Wang, Ziyan Liu, Dan Gutfreund, Julian Shun, and Yada
Zhu. How much reasoning do retrieval-augmented models add beyond LLMs? a bench-
marking framework for multi-hop inference over hybrid knowledge. InProceedings of
the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V .2, pp. 9403–
9414. ACM, 2026. doi: 10.1145/3770855.3817481.
Andrew Liu, Henry Prior, Gargi Balasubramaniam, Rivka Moroshko, Amir Zait, Ilia
Labzovsky, Danny Karmon, Ishita Dasgupta, Kim Stachenfeld, and Kenneth Marino.
ReCogLab: A framework testing relational reasoning & cognitive hypotheses on LLMs.
InThe Thirteenth International Conference on Learning Representations, 2025.
Shayne Longpre, Kartik Perisetla, Anthony Chen, Nikhil Ramesh, Chris DuBois, and
Sameer Singh. Entity-based knowledge conflicts in question answering. InConference
on Empirical Methods in Natural Language Processing, pp. 7052–7063. Association for Com-
putational Linguistics, 2021. doi: 10.18653/v1/2021.emnlp-main.565.
Yinghang Ma, Jiho Shin, Leuson da Silva, Zhen Ming Jiang, Song Wang, Foutse Khomh,
and Shin Hwei Tan. BloomAPR: A bloom’s taxonomy-based framework for assessing
the capabilities of LLM-powered APR solutions.arXiv preprint arXiv:2509.25465, 2025.
doi: 10.48550/arXiv.2509.25465. URLhttps://arxiv.org/abs/2509.25465.
Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing fac-
tual associations in GPT. InAdvances in Neural Information Processing Systems, volume 35,
pp. 17359–17372, 2022.
Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying
large language models and knowledge graphs: A roadmap.IEEE Transactions on Knowl-
edge and Data Engineering, 36(7):3580–3599, 2024. doi: 10.1109/TKDE.2024.3352100.
Bianca Raimondi and Maurizio Gabbrielli. Mechanistic interpretability of cognitive
complexity in LLMs via linear probing using Bloom’s taxonomy.arXiv preprint
arXiv:2602.17229, 2026. doi: 10.48550/arXiv.2602.17229.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christo-
pher D. Manning. RAPTOR: Recursive abstractive processing for tree-organized re-
trieval. InThe Twelfth International Conference on Learning Representations, 2024.
Liangtai Sun, Yang Han, Zihan Zhao, Da Ma, Zhennan Shen, Baocai Chen, Lu Chen, and
Kai Yu. SciEval: A multi-level large language model evaluation benchmark for scien-
tific research.Proceedings of the AAAI Conference on Artificial Intelligence, 38(17):19053–
19061, 2024. doi: 10.1609/aaai.v38i17.29872. URLhttps://ojs.aaai.org/index.php/
AAAI/article/view/29872.
12

Published as a conference paper at COLM 2026
Travis Thompson, Seung-Hwan Lim, Paul Liu, Ruoying He, and Dongkuan Xu. Infer-
ence scaled GraphRAG: Improving multi hop question answering on knowledge graphs.
arXiv preprint arXiv:2506.19967, 2025. doi: 10.48550/arXiv.2506.19967.
Sara Todorovikj, Lars-Peter Meyer, and Michael Martin. Characterizing knowledge graph
tasks in LLM benchmarks using cognitive complexity frameworks. InProceedings of SE-
MANTiCS 2025 Poster Track, 2025. doi: 10.48550/arXiv.2509.19347.
Xiaozhi Wang, Tianyu Gao, Zhaocheng Zhu, Zhengyan Zhang, Zhiyuan Liu, Juanzi Li,
and Jian Tang. KEPLER: A unified model for knowledge embedding and pre-trained
language representation.Transactions of the Association for Computational Linguistics, 9:
176–194, 2021. doi: 10.1162/tacl a00360.
Liangliang Zhang, Zhuorui Jiang, Hongliang Chi, Haoyang Chen, Mohammed ElKoumy,
Fali Wang, Qiong Wu, Zhengyi Zhou, Shirui Pan, Suhang Wang, and Yao Ma. Diagnos-
ing and addressing pitfalls in KG-RAG datasets: Toward more reliable benchmarking.
InAdvances in Neural Information Processing Systems, volume 38, pp. 75490–75527, 2025.
Yuxuan Zhou, Xien Liu, Chenwei Yan, Chen Ning, Xiao Zhang, Boxun Li, Xiangling Fu,
Shijin Wang, Guoping Hu, Yu Wang, et al. Evaluating LLMs across multi-cognitive lev-
els: From medical knowledge mastery to scenario-based problem solving. InInterna-
tional Conference on Machine Learning, 2025. doi: 10.48550/arXiv.2506.08349.
Maria-Eleni Zoumpoulidi, Georgios Paraskevopoulos, and Alexandros Potamianos.
BloomWise: Enhancing problem-solving capabilities of large language models using
Bloom’s-taxonomy-inspired prompts. InProceedings of The 3rd Workshop on Mathematical
Natural Language Processing (MathNLP 2025), pp. 34–49. Association for Computational
Linguistics, 2025. doi: 10.18653/v1/2025.mathnlp-main.3.
A Appendix
A.1 Model Checkpoint Inventory
We evaluate 43 model checkpoints in total.
Qwen2.5 Base.Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B, Qwen2.5-14B,
and Qwen2.5-32B.
Qwen2.5 Instruct.Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct,
Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, and Qwen2.5-32B-Instruct.
Qwen3.Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, and Qwen3-14B.
Qwen3.5.Qwen3.5-0.8B, Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B, and Qwen3.5-27B.
Gemma.Gemma 3 270M IT, Gemma 3 4B IT, Gemma 3 12B IT, Gemma 3 27B IT, Gemma 4
E4B IT, Gemma 4 26B-A4B IT, and Gemma 4 31B IT.
LLaMA.Llama 3.2 1B Instruct, Llama 3.2 3B Instruct, Llama 3.1 8B Instruct, and Llama 3.1
70B Instruct.
Granite.Granite 3.3 2B Instruct, Granite 4.0 Micro, Granite 4.0 H-Tiny, Granite 3.3 8B
Instruct, and Granite 4.0 H-Small.
DeepSeek.DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B, and
DeepSeek-R1-Distill-Qwen-14B.
Gemini.Gemini 3 Flash and Gemini 3.1 Pro.
13

Published as a conference paper at COLM 2026
A.2 Full Hypothesis Definitions and Statistical Procedures
This appendix provides the complete mathematical definitions corresponding to the com-
pact hypothesis descriptions in the main paper. For hypotheses involving model scale,A S,
AM, andA Ldenote mean accuracy within the Small (<7B), Medium (≥7B and<14B),
and Large (≥14B) buckets of a coherent model series.
H1: Diminishing Gains with Model ScaleLet
GSM=A M−A S
denote the gain from Small to Medium and
GML=A L−A M
the gain from Medium to Large. We define
Dscale =G SM−G ML.
H1 is directionally supported when
GSM>0,G ML>0,D scale>0.
Thus, both scaling transitions must improve accuracy, and the Small-to-Medium gain must
exceed the Medium-to-Large gain.
H2: Accuracy Increases with Model SizeH2 tests the two adjacent scaling contrasts
AM>A S
and
AL>A M.
Support is reported separately by transition. A Medium result indicates thatA M>A S,
while a Large result indicates thatA L>A M.
H3: Apply-to-Analyze Performance DropWe define
DAA=A Apply−A Analyze .
H3 tests
DAA>0.
In addition to directional and statistical support, we mark an Apply-to-Analyze “cliff”
when the drop is statistically significant and the lower bound of its 95% relation-cluster
bootstrap interval exceeds 10 percentage points.
H4: Veracity OrderingH4 tests the ordering
ANeutralized >A Factual >A Counterfactual .
A size bucket supports H4 only when both component inequalities
ANeutralized >A Factual
and
AFactual >A Counterfactual
hold in the expected direction.
14

Published as a conference paper at COLM 2026
H5: Factual–Counterfactual Gap Widens with Cognitive ComplexityFor Bloom level
b, define
Gb=A Factual,b −A Counterfactual,b .
We summarize lower-complexity levels as
Glow=GRemember +G Understand
2,
and higher-complexity levels as
Ghigh =GAnalyze +G Evaluate
2.
Apply is omitted because it occupies the middle position in the five-level sequence.
We define
DFC=G high−G low,
and H5 tests
DFC>0.
H6: Relation Usage Frequency Has a Negligible EffectFor each size bucket, let
ALow,A Medium ,A High
denote mean accuracy on relations in the three usage-frequency tiers.
We define the observed spread as
Rusage =max{A Low,AMedium ,AHigh} −min{A Low,AMedium ,AHigh}.
The effect is considered practically negligible when
Rusage<0.05,
corresponding to a maximum spread below five percentage points.
For statistical equivalence, we consider the three pairwise contrasts:
ALow−A Medium ,
ALow−A High,
and
AMedium −A High.
A bucket receives statistical support only when the corresponding 95% relation-cluster
bootstrap intervals lie entirely inside the±5 percentage-point equivalence region.
The confirmatory H6 claim concerns models at or above 7B parameters, so Medium and
Large are the confirmatory buckets. Small is reported only as an exploratory diagnostic.
H7: Neutralized Advantage Is Concentrated at Higher Bloom LevelsFor each Bloom
levelb, define the Neutralized-over-Factual advantage
Nb=A Neutralized,b −A Factual,b .
We summarize the lower levels by
Nlow=NRemember +N Understand
2,
and the higher levels by
Nhigh =NAnalyze +N Evaluate
2.
15

Published as a conference paper at COLM 2026
Apply is omitted as the middle level.
We define
DNF=N high−N low,
and H7 tests
DNF>0.
Thus, H7 asks whether removing entity identity provides a larger relative benefit under
higher cognitive demand.
H8: Evaluate Is a Recurring Bloom BottleneckLet
Aother =ARemember +A Understand +A Apply +A Analyze
4.
We define the Evaluate bottleneck gap as
DEval =A other−A Evaluate .
H8 tests
DEval>0.
This definition asks whether Evaluate is harder than the average of the other four Bloom
levels rather than requiring it to be worse than each level individually.
H9: Item-Level Difficulty Is Stable Across LanguagesFor itemiand languageℓ, let
ai,ℓdenote its mean accuracy within a model series and size bucket. For every pair of
languages(ℓ 1,ℓ2), we compute
ρitem
ℓ1,ℓ2=Spearman {ai,ℓ1}i,{a i,ℓ2}i
.
With six languages, this yields 15 language-pair correlations. We summarize item-level
cross-language stability by
Ritem =1
15∑
ℓ1<ℓ2ρitem
ℓ1,ℓ2,
and H9 tests
Ritem>0.
Significance is evaluated using a one-sided sign-flip test over the 15 pairwise Spearman
correlations, testing whether their mean is greater than zero. Confidence intervals are ob-
tained by bootstrap resampling the 15 language-pair correlation values and recomputing
their mean.
H10: Relation-Level Difficulty Hierarchy Is Stable Across LanguagesFor relationrand
languageℓ, letA r,ℓdenote mean accuracy on that relation. For every pair of languages,
compute
ρrel
ℓ1,ℓ2=Spearman {Ar,ℓ1}r,{A r,ℓ2}r
.
We summarize relation-level stability by
Rrel=1
15∑
ℓ1<ℓ2ρrel
ℓ1,ℓ2.
H10 tests
Rrel>0.
A positive result means that relations that are comparatively difficult in one language tend
to retain their relative difficulty in the others.
16

Published as a conference paper at COLM 2026
Statistical InferenceQuestions derived from the same Wikidata relation share a source
triple and therefore are not treated as independent observations.
For H1–H5, H7, and H8, we first construct the hypothesis-specific matched contrast and
then average the contrast within each of the 30 source relations. Directional hypotheses
are evaluated using one-sided relation-cluster sign-flip permutation tests with 105Monte
Carlo permutations. We report 95% relation-cluster bootstrap confidence intervals based
on 2×104resamples.
Holm correction is applied atα=0.05 using hypothesis-specific correction families. For
H1, the series-levelp-values are corrected jointly across the available open-weight model
series. For H2, the two adjacent size-transition tests are corrected within each series. For
H4, the two component veracity contrasts are corrected jointly across the available size
buckets within each open-weight series. For H3, H5, and H7–H10, the available S/M/L
p-values are corrected within each open-weight series. For Gemini, H4 corrects its two
component contrasts within each named model, whereas for H3, H5, and H7–H10 the two
named-modelp-values are corrected together.
H6 is an equivalence hypothesis rather than a directional hypothesis. We therefore use
the predefined five-percentage-point equivalence region and assess whether the relation-
cluster bootstrap intervals for the relevant pairwise usage-frequency contrasts lie entirely
inside that region.
For H9, the statistic is the mean of the 15 pairwise item-level Spearman correlations. Signif-
icance is evaluated using a one-sided sign-flip test over the 15 correlation values, and con-
fidence intervals are obtained by bootstrap resampling those language-pair correlations.
For H10, significance is evaluated using a relation-identity permutation test. Relation
correspondences are independently shuffled across languages to break any stable cross-
language relation ranking, after which the mean of the 15 pairwise Spearman correlations
is recomputed. Confidence intervals are obtained by resampling relations and recomput-
ing the full correlation statistic.
A.3 Full Hypothesis Support Matrices
Tables 6 and 7 give per-series and per-bucket support for all ten hypotheses; Table 4 in the
main text summarizes them.
A.4 Benchmark Construction Details
Relation candidate pool and usage tiers.We construct a candidate pool of Wikidata
properties starting from the relations in Wikidata5M (Wang et al., 2021). For each relation,
we take its primary English alias as a human-readable handle, deduplicate, and randomly
shuffle the resulting list using a fixed seed, retaining up toN=1000 candidates for down-
stream sampling. We estimate relation usage frequency by querying the MediaWiki search
API withhaswbstatement:PIDand using the returnedtotalhitsas a proxy. This measures
the prevalence of the property in Wikidata-indexed content and should not be interpreted
as a direct estimate of model pretraining exposure. We apply a log(1+hits)transforma-
tion to reduce skew and split values into tertiles, defining Low, Medium, and High usage
tiers. We then sample up to 30 properties per tier using a fixed random seed, yielding a
candidate pool of up to 90 properties for manual curation.
Triple extraction and manual curation.For each sampled propertyp, we extract exactly
one representative factual triple(h,p,t)from popular English Wikipedia pages, ranked by
sitelink count as a proxy for popularity. From the automatically sampled candidates, we
manually curate a final set of 30 relations, 10 per usage tier, excluding relations involving
sensitive topics, overly technical semantics, or cases where multiple relations feature the
same entity. We prioritize semantic clarity and domain diversity across geography, science,
engineering, sports, entertainment, and linguistics.
17

Published as a conference paper at COLM 2026
Table 6: Hypothesis support by model series in thefully translated setting. Cells are
shaded to convey strength of support at a glance: all present buckets significantand
an H3 cliff (†); all present buckets significant (starred); partial significance (some
buckets starred, some directional only); direction observed but no statistical support;
no support (x); not applicable (—). Reading the cell contents: S, M, and L indicate
that the required direction is observed in the Small, Medium, and Large model buckets,
respectively; starred labels indicate statistical support under the hypothesis-specific proce-
dure; Holm correction is applied where applicable. † marks an H3 cliff whose lower 95%
relation-cluster bootstrap bound exceeds 10 percentage points. H1 is a single series-level
test rather than a per-bucket test. Gemini models are reported individually because their
parameter counts are undisclosed and are therefore not evaluated under H1 or H2; for
Gemini,∗denotes statistical support and✓denotes the required direction without statisti-
cal support. ‡ marks a value at the H6 five-percentage-point boundary under the rounded
output.
Series H1 H2 H3 H4 H5 H6 H7 H8 H9 H10
Qwen2.5 Base * S,M*,L* S*,M*†,L*† S,M,L S,M M S*,M*,L* S*,M*,L* S*,M*,L* S*,M*,L*
Qwen2.5 Instruct * S,M*,L* S*,M*†,L*† S,M,L* S x S*,M*,L* S*,M*,L* S*,M*,L* S*,M*,L*
Qwen3 * S,M*,L* S*,M*†,L*† S,M,L S,M*,L M,L S*,M,L S*,M*,L* S*,M*,L* S*,M*,L*
Qwen3.5 * S,M*,L* S*†,M*†,L* S,M S,L x S*,M S*,M*,L* S*,M*,L* S*,M*,L*
Gemma * S,M*,L* S*†,M*†,L* S,M S,M,L x S*,M S*,M*,L* S*,M*,L* S*,M*,L*
LLaMA x S,M*,L* S*,M*†,L*† L x S S*,M,L* S*,M,L* S*,M*,L* S*,M*,L*
Granite x S,M*,L* S*,M*†,L*† S,M,L S,M x S*,M*,L* S*,M*,L S*,M*,L* S*,M*,L*
DeepSeek x S,M* S,M*,L* M,L S,L x S,M*,L S*,M*,L* S*,M*,L* S*,M*,L*
Gemini 3 Flash — — x x *✓‡✓ * * *
Gemini 3.1 Pro — — x x * ✓ x * * *
Table 7: Hypothesis support by model series in thehybrid cross-lingual setting. Cell
shading follows the same scale as Table 6 (deeper darkblue=stronger support; rose=no
support). Gemini models were not evaluated in this setting.
Series H1 H2 H3 H4 H5 H6 H7 H8 H9 H10
Qwen2.5 Base * S,M*,L* S*†,M*†,L*† S,M,L x M,L S*,M*,L* S*,M*,L* S*,M*,L* S*,M*,L*
Qwen2.5 Instruct * S,M*,L* S*,M*†,L*† S,M,L* x M S*,M*,L* S*,M*,L* S*,M*,L* S*,M*,L*
Qwen3 * S,M*,L* S*,M*†,L*† S,M,L S,M,L M,L S*,M*,L* S*,M*,L* S*,M*,L* S*,M*,L*
Qwen3.5 * S,M*,L* S*†,M*†,L* S,M*,L S,L* x S*,M,L S*,M*,L* S*,M*,L* S*,M*,L*
Gemma * S,M*,L* S*†,M*†,L* S*,M,L S,M,L* L S*,M,L S*,M*,L* S*,M*,L* S*,M*,L*
LLaMA x S,M*,L* S,M*†,L*† L* L S,M S*,M*,L* S*,M,L* S*,M*,L* S*,M*,L*
Granite x S,M*,L* S*†,M*†,L*† S,M*,L M M S*,M*,L* S*,M*,L S*,M*,L* S*,M*,L*
DeepSeek * S,M*,L S,L*† M,L L S S*,M*,L* M*,L* S*,M*,L* S*,M*,L*
A.5 Item Quality Control
All generated questions, answer options, and answer keys were manually reviewed by the
authors for factual correctness, preservation of the intended relation, answer uniqueness,
and distractor plausibility. As an additional quality-control step, every item was indepen-
dently evaluated using Gemini and Grok. Each model received the fact sentence, ques-
tion, answer options, and intended answer key and was asked to identify cases in which
it disagreed with the provided answer. All flagged cases were manually inspected. When
disagreement occurred, we examined whether it reflected model error or plausible ambi-
guity in the question or answer options. Valid objections led to revisions of wording or
distractors, followed by re-evaluation before inclusion.
A.6 Language-Specific Performance in the Fully Translated Setting
Figures A1–A6 provide a language-specific detailed overview of model performance in the
fully translated setting across English, Arabic, Chinese, Hindi, Persian, and Spanish. For
each language, three heatmaps illustrate accuracy under the Factual, Counterfactual, and
Neutralized conditions. The 43 model checkpoints (Appendix A.1) are arranged along the
rows, while the columns represent the five Bloom-inspired cognitive levels: Remember,
18

Published as a conference paper at COLM 2026
Understand, Apply, Analyze, and Evaluate. Each cell reports accuracy computed from 30
questions, with a consistent 0–100% color scale applied across all 18 heatmaps to enable
direct comparisons.
19

Published as a conference paper at COLM 2026
Remember UnderstandApply
AnalyzeEvaluateQwen2.5-0.5B
Qwen2.5-1.5B
Qwen2.5-3B
Qwen2.5-7B
Qwen2.5-14B
Qwen2.5-32B
Qwen2.5-0.5B-Instruct
Qwen2.5-1.5B-Instruct
Qwen2.5-3B-Instruct
Qwen2.5-7B-Instruct
Qwen2.5-14B-Instruct
Qwen2.5-32B-Instruct
Qwen3-0.6B
Qwen3-1.7B
Qwen3-4B
Qwen3-8B
Qwen3-14B
Qwen3.5-0.8B
Qwen3.5-2B
Qwen3.5-4B
Qwen3.5-9B
Qwen3.5-27B
Gemma 3 270M IT
Gemma 3 4B IT
Gemma 3 12B IT
Gemma 3 27B IT
Gemma 4 E4B IT
Gemma 4 26B-A4B IT
Gemma 4 31B IT
Llama 3.2 1B Instruct
Llama 3.2 3B Instruct
Llama 3.1 8B Instruct
Llama 3.1 70B Instruct
Granite 3.3 2B Instruct
Granite 4.0 Micro
Granite 4.0 H-Tiny
Granite 3.3 8B Instruct
Granite 4.0 H-Small
DeepSeek-R1-Distill-Qwen-1.5B
DeepSeek-R1-Distill-Qwen-7B
DeepSeek-R1-Distill-Qwen-14B
Gemini 3 Flash
Gemini 3.1 ProFactual
Remember UnderstandApply
AnalyzeEvaluateCounterfactual
Remember UnderstandApply
AnalyzeEvaluateNeutralized
0 25 50 75 100English  |  Fully translated
Accuracy (%)
Figure A1: English: model accuracy by Bloom level and veracity condition in the fully
translated setting.
20

Published as a conference paper at COLM 2026
Remember UnderstandApply
AnalyzeEvaluateQwen2.5-0.5B
Qwen2.5-1.5B
Qwen2.5-3B
Qwen2.5-7B
Qwen2.5-14B
Qwen2.5-32B
Qwen2.5-0.5B-Instruct
Qwen2.5-1.5B-Instruct
Qwen2.5-3B-Instruct
Qwen2.5-7B-Instruct
Qwen2.5-14B-Instruct
Qwen2.5-32B-Instruct
Qwen3-0.6B
Qwen3-1.7B
Qwen3-4B
Qwen3-8B
Qwen3-14B
Qwen3.5-0.8B
Qwen3.5-2B
Qwen3.5-4B
Qwen3.5-9B
Qwen3.5-27B
Gemma 3 270M IT
Gemma 3 4B IT
Gemma 3 12B IT
Gemma 3 27B IT
Gemma 4 E4B IT
Gemma 4 26B-A4B IT
Gemma 4 31B IT
Llama 3.2 1B Instruct
Llama 3.2 3B Instruct
Llama 3.1 8B Instruct
Llama 3.1 70B Instruct
Granite 3.3 2B Instruct
Granite 4.0 Micro
Granite 4.0 H-Tiny
Granite 3.3 8B Instruct
Granite 4.0 H-Small
DeepSeek-R1-Distill-Qwen-1.5B
DeepSeek-R1-Distill-Qwen-7B
DeepSeek-R1-Distill-Qwen-14B
Gemini 3 Flash
Gemini 3.1 ProFactual
Remember UnderstandApply
AnalyzeEvaluateCounterfactual
Remember UnderstandApply
AnalyzeEvaluateNeutralized
0 25 50 75 100Arabic  |  Fully translated
Accuracy (%)
Figure A2: Arabic: model accuracy by Bloom level and veracity condition in the fully
translated setting.
21

Published as a conference paper at COLM 2026
Remember UnderstandApply
AnalyzeEvaluateQwen2.5-0.5B
Qwen2.5-1.5B
Qwen2.5-3B
Qwen2.5-7B
Qwen2.5-14B
Qwen2.5-32B
Qwen2.5-0.5B-Instruct
Qwen2.5-1.5B-Instruct
Qwen2.5-3B-Instruct
Qwen2.5-7B-Instruct
Qwen2.5-14B-Instruct
Qwen2.5-32B-Instruct
Qwen3-0.6B
Qwen3-1.7B
Qwen3-4B
Qwen3-8B
Qwen3-14B
Qwen3.5-0.8B
Qwen3.5-2B
Qwen3.5-4B
Qwen3.5-9B
Qwen3.5-27B
Gemma 3 270M IT
Gemma 3 4B IT
Gemma 3 12B IT
Gemma 3 27B IT
Gemma 4 E4B IT
Gemma 4 26B-A4B IT
Gemma 4 31B IT
Llama 3.2 1B Instruct
Llama 3.2 3B Instruct
Llama 3.1 8B Instruct
Llama 3.1 70B Instruct
Granite 3.3 2B Instruct
Granite 4.0 Micro
Granite 4.0 H-Tiny
Granite 3.3 8B Instruct
Granite 4.0 H-Small
DeepSeek-R1-Distill-Qwen-1.5B
DeepSeek-R1-Distill-Qwen-7B
DeepSeek-R1-Distill-Qwen-14B
Gemini 3 Flash
Gemini 3.1 ProFactual
Remember UnderstandApply
AnalyzeEvaluateCounterfactual
Remember UnderstandApply
AnalyzeEvaluateNeutralized
0 25 50 75 100Chinese  |  Fully translated
Accuracy (%)
Figure A3: Chinese: model accuracy by Bloom level and veracity condition in the fully
translated setting.
22

Published as a conference paper at COLM 2026
Remember UnderstandApply
AnalyzeEvaluateQwen2.5-0.5B
Qwen2.5-1.5B
Qwen2.5-3B
Qwen2.5-7B
Qwen2.5-14B
Qwen2.5-32B
Qwen2.5-0.5B-Instruct
Qwen2.5-1.5B-Instruct
Qwen2.5-3B-Instruct
Qwen2.5-7B-Instruct
Qwen2.5-14B-Instruct
Qwen2.5-32B-Instruct
Qwen3-0.6B
Qwen3-1.7B
Qwen3-4B
Qwen3-8B
Qwen3-14B
Qwen3.5-0.8B
Qwen3.5-2B
Qwen3.5-4B
Qwen3.5-9B
Qwen3.5-27B
Gemma 3 270M IT
Gemma 3 4B IT
Gemma 3 12B IT
Gemma 3 27B IT
Gemma 4 E4B IT
Gemma 4 26B-A4B IT
Gemma 4 31B IT
Llama 3.2 1B Instruct
Llama 3.2 3B Instruct
Llama 3.1 8B Instruct
Llama 3.1 70B Instruct
Granite 3.3 2B Instruct
Granite 4.0 Micro
Granite 4.0 H-Tiny
Granite 3.3 8B Instruct
Granite 4.0 H-Small
DeepSeek-R1-Distill-Qwen-1.5B
DeepSeek-R1-Distill-Qwen-7B
DeepSeek-R1-Distill-Qwen-14B
Gemini 3 Flash
Gemini 3.1 ProFactual
Remember UnderstandApply
AnalyzeEvaluateCounterfactual
Remember UnderstandApply
AnalyzeEvaluateNeutralized
0 25 50 75 100Hindi  |  Fully translated
Accuracy (%)
Figure A4: Hindi: model accuracy by Bloom level and veracity condition in the fully trans-
lated setting.
23

Published as a conference paper at COLM 2026
Remember UnderstandApply
AnalyzeEvaluateQwen2.5-0.5B
Qwen2.5-1.5B
Qwen2.5-3B
Qwen2.5-7B
Qwen2.5-14B
Qwen2.5-32B
Qwen2.5-0.5B-Instruct
Qwen2.5-1.5B-Instruct
Qwen2.5-3B-Instruct
Qwen2.5-7B-Instruct
Qwen2.5-14B-Instruct
Qwen2.5-32B-Instruct
Qwen3-0.6B
Qwen3-1.7B
Qwen3-4B
Qwen3-8B
Qwen3-14B
Qwen3.5-0.8B
Qwen3.5-2B
Qwen3.5-4B
Qwen3.5-9B
Qwen3.5-27B
Gemma 3 270M IT
Gemma 3 4B IT
Gemma 3 12B IT
Gemma 3 27B IT
Gemma 4 E4B IT
Gemma 4 26B-A4B IT
Gemma 4 31B IT
Llama 3.2 1B Instruct
Llama 3.2 3B Instruct
Llama 3.1 8B Instruct
Llama 3.1 70B Instruct
Granite 3.3 2B Instruct
Granite 4.0 Micro
Granite 4.0 H-Tiny
Granite 3.3 8B Instruct
Granite 4.0 H-Small
DeepSeek-R1-Distill-Qwen-1.5B
DeepSeek-R1-Distill-Qwen-7B
DeepSeek-R1-Distill-Qwen-14B
Gemini 3 Flash
Gemini 3.1 ProFactual
Remember UnderstandApply
AnalyzeEvaluateCounterfactual
Remember UnderstandApply
AnalyzeEvaluateNeutralized
0 25 50 75 100Persian  |  Fully translated
Accuracy (%)
Figure A5: Persian: model accuracy by Bloom level and veracity condition in the fully
translated setting.
24

Published as a conference paper at COLM 2026
Remember UnderstandApply
AnalyzeEvaluateQwen2.5-0.5B
Qwen2.5-1.5B
Qwen2.5-3B
Qwen2.5-7B
Qwen2.5-14B
Qwen2.5-32B
Qwen2.5-0.5B-Instruct
Qwen2.5-1.5B-Instruct
Qwen2.5-3B-Instruct
Qwen2.5-7B-Instruct
Qwen2.5-14B-Instruct
Qwen2.5-32B-Instruct
Qwen3-0.6B
Qwen3-1.7B
Qwen3-4B
Qwen3-8B
Qwen3-14B
Qwen3.5-0.8B
Qwen3.5-2B
Qwen3.5-4B
Qwen3.5-9B
Qwen3.5-27B
Gemma 3 270M IT
Gemma 3 4B IT
Gemma 3 12B IT
Gemma 3 27B IT
Gemma 4 E4B IT
Gemma 4 26B-A4B IT
Gemma 4 31B IT
Llama 3.2 1B Instruct
Llama 3.2 3B Instruct
Llama 3.1 8B Instruct
Llama 3.1 70B Instruct
Granite 3.3 2B Instruct
Granite 4.0 Micro
Granite 4.0 H-Tiny
Granite 3.3 8B Instruct
Granite 4.0 H-Small
DeepSeek-R1-Distill-Qwen-1.5B
DeepSeek-R1-Distill-Qwen-7B
DeepSeek-R1-Distill-Qwen-14B
Gemini 3 Flash
Gemini 3.1 ProFactual
Remember UnderstandApply
AnalyzeEvaluateCounterfactual
Remember UnderstandApply
AnalyzeEvaluateNeutralized
0 25 50 75 100Spanish  |  Fully translated
Accuracy (%)
Figure A6: Spanish: model accuracy by Bloom level and veracity condition in the fully
translated setting.
25
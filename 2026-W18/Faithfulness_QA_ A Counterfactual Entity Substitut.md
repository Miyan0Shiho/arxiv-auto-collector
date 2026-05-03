# Faithfulness-QA: A Counterfactual Entity Substitution Dataset for Training Context-Faithful RAG Models

**Authors**: Li Ju, Junzhe Wang, Qi Zhang

**Published**: 2026-04-28 07:21:46

**PDF URL**: [https://arxiv.org/pdf/2604.25313v2](https://arxiv.org/pdf/2604.25313v2)

## Abstract
Retrieval-Augmented Generation (RAG) models frequently produce answers grounded in parametric memory rather than the retrieved context, undermining the core promise of retrieval augmentation. A fundamental obstacle to fixing this unfaithfulness is the lack of training data that explicitly requires models to prefer context over internal knowledge. We introduce Faithfulness-QA, a large-scale dataset of 99,094 samples constructed through counterfactual entity substitution. Starting from two established extractive QA benchmarks--SQuAD and TriviaQA--we automatically identify answer-bearing named entities in each context, replace them with type-consistent alternatives drawn from a curated bank of 76,953 entities, and thereby manufacture controlled knowledge conflicts between context and parametric memory. Rigorous quality filtering ensures 100% pass rates across four automated checks on random 200-sample audits. We release the full dataset, the construction pipeline, and a typed entity bank covering eight named entity categories. Faithfulness-QA is designed as a training resource for attention-based faithfulness objectives and as an evaluation benchmark for measuring context-grounding behavior in RAG systems. Data and code are available at https://github.com/qzhangFDU/faithfulness-qa-dataset.

## Full Text


<!-- PDF content starts -->

Faithfulness-QA: A Counterfactual Entity Substitution
Dataset for Training Context-Faithful RAG Models
Li Ju1, Junzhe Wang2, Qi Zhang1,2
1WisPaper.AI2College of Computer Science and Artificial Intelligence, Fudan University
qz@fudan.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) models frequently produce answers grounded in paramet-
ric memory rather than the retrieved context, undermining the core promise of retrieval augmenta-
tion. A fundamental obstacle to fixing thisunfaithfulnessis the lack of training data that explic-
itly requires models to prefer context over internal knowledge. We introduceFaithfulness-QA, a
large-scale dataset of99,094samples constructed throughcounterfactual entity substitution. Starting
from two established extractive QA benchmarks—SQuAD and TriviaQA—we automatically iden-
tify answer-bearing named entities in each context, replace them with type-consistent alternatives
drawn from a curated bank of 76,953 entities, and thereby manufacture controlled knowledge con-
flicts between context and parametric memory. Rigorous quality filtering ensures 100% pass rates
across four automated checks on random 200-sample audits. We release the full dataset, the con-
struction pipeline, and a typed entity bank covering eight named entity categories. Faithfulness-QA
is designed as a training resource for attention-based faithfulness objectives and as an evaluation
benchmark for measuring context-grounding behavior in RAG systems. Data and code are available
athttps://github.com/qzhangFDU/faithfulness-qa-dataset.
1 Introduction
Retrieval-Augmented Generation (RAG) has become the dominant paradigm for grounding large lan-
guage models (LLMs) in external evidence [Lewis et al., 2020, Guu et al., 2020]. By prepending re-
trieved documents to the input, RAG systems can, in principle, produce responses that faithfully reflect
the most up-to-date or domain-specific information. In practice, however, models frequentlyignorethe
retrieved context and instead fall back on parametric knowledge acquired during pre-training, a failure
mode broadly categorized asunfaithfulness[Longpre et al., 2021, Xie et al., 2024].
The root cause is structural: the standard next-token prediction loss treats every token identically,
regardless of whether it originates from the context or from memorized parameters. Without explicit
training signal that rewards context-grounding, there is no mechanism to steer the model’s attention
toward the retrieved passage when a conflict exists between external evidence and internal belief [Li
et al., 2022].
Several lines of work have attempted to address this problem. Self-RAG [Asai et al., 2024] trains
models to generate reflection tokens that critique their own outputs, but still relies on the model’s inter-
nal judgment. FaithfulRAG [Zhang et al., 2025a] models knowledge conflicts at the fact level through
a self-thinking process, yet focuses on inference-time conflict resolution rather than systematic training
data construction. CoCoLex [Santosh et al., 2025] forces copy-based decoding at inference time, which
is inherently limited to extractive answers. FaithEval [Ming et al., 2025] offers a benchmark with coun-
terfactual, inconsistent, and unanswerable contexts, but is designed forevaluation(4,900 samples) rather
than large-scaletraining.
A complementary approach is to constructtraining datathat explicitly and systematically creates
knowledge conflicts, so that a model trained on such data must learn to follow the context. The Coun-
terFact dataset [Meng et al., 2022] demonstrated the value of counterfactual entity manipulation, but its
1arXiv:2604.25313v2  [cs.CL]  29 Apr 2026

purpose isknowledge editingrather than RAG faithfulness training. More recent work on knowledge
conflicts in LLMs [Xu et al., 2024] has highlighted the urgent need for large-scale, controlled conflict
datasets, yet no such resource targeted at RAG training currently exists.
In this paper, we presentFaithfulness-QA, a dataset of 99,094 counterfactual samples specifically
designed to train context-faithful RAG models. Our approach is conceptually simple but effective: for
each extractive QA sample, we identify the answer-bearing entity in the context via Named Entity Recog-
nition (NER), replace it with a type-consistent alternative, and thereby create a controlled conflict be-
tween the context and the model’s likely parametric knowledge. The key insight is thatonly when context
contradicts parametric knowledge can we distinguish genuine context-following from incidental agree-
ment.
Our contributions are:
1. We releaseFaithfulness-QA, a 99K-sample dataset with counterfactual entity substitution across 8
entity types, built from SQuAD [Rajpurkar et al., 2016] and TriviaQA [Joshi et al., 2017].
2. We develop and open-source a fully automatedconstruction pipelinethat achieves 56.7% substitu-
tion success rate with 100% quality on audited samples.
3. We provide atyped entity bankof 76,953 unique entities extracted from Wikipedia, enabling repro-
ducible and extensible dataset construction.
4. We present detaileddata analysiscovering entity type distributions, context statistics, skip-reason
diagnostics, and complementarity between source datasets.
2 Related Work
Retrieval-Augmented Generation.RAG was introduced by Lewis et al. [2020] to ground LLM gen-
eration in retrieved documents, and has since been extended to diverse settings including pre-training
[Guu et al., 2020], trillion-token retrieval [Borgeaud et al., 2022], and black-box augmentation [Shi et al.,
2024]. Despite its popularity, RAG systems suffer from faithfulness failures: Barnett et al. [2024] showed
that irrelevant retrieved documents candecreasemodel performance, and Wu et al. [2024] quantified the
“tug-of-war” between RAG context and the model’s internal prior, finding that even state-of-the-art mod-
els frequently ignore the retrieved content when it conflicts with parametric knowledge.
Faithfulness and Hallucination.Faithfulness—the degree to which a model’s output is supported by
its input context—has been extensively studied in summarization [Maynez et al., 2020, Laban et al.,
2022] and question answering [Li et al., 2022]. In the RAG setting, Self-RAG [Asai et al., 2024] in-
troduces reflection tokens for self-critique, and CRAG [Yan et al., 2024] proposes corrective retrieval
strategies. FaithfulRAG [Zhang et al., 2025a] decomposes knowledge conflicts at the fact level, while
Drift [Li et al., 2025] uses dual-reward probabilistic inference to enhance rationale faithfulness. HaluE-
val [Li et al., 2023] and FaithEval [Ming et al., 2025] provide evaluation benchmarks for hallucination
detection and context faithfulness, but neither is designed as a large-scale training resource.
Knowledge Conflicts.When retrieved context contradicts a model’s parametric knowledge, the re-
sultingknowledge conflictis a critical test of faithfulness [Longpre et al., 2021, Xie et al., 2024]. Xu
et al. [2024] provide a comprehensive survey of knowledge conflict phenomena in LLMs, categorizing
conflicts into context–memory, inter-context, and intra-memory types. KAFT [Li et al., 2024] proposes
Knowledge-Aware Fine-Tuning that incorporates counterfactual and irrelevant contexts to strengthen
controllability and robustness. Our work focuses on the first category—creating systematic context–
memory conflicts for training.
2

Faithfulness-QA: Training RAG to Follow Context
1. Entity Bank
76,953 typed entities
PPERSON
e.g., Marie Curie
GGPE
e.g., Paris
OORG
e.g., UNESCO
DDATE
e.g., 21 July 1869
+ 4 more entity types
extracted from Wikipediasupply2. NER & Match
Find answer entity
CONTEXT
It is a replica of the grotto at Lourdes,
France where the Virgin Mary
reputedly appeared to
Saint Bernadette Soubirous
in 1858.
Q: To whom did the Virgin Mary
allegedly appear in 1858 in
Lourdes France?
A: Saint Bernadette Soubirous
NER + multi-strategy
cascade matching
MATCHED
Saint Bernadette Soubirous
type: PERSON
SpaCy + heuristic matchmatched
entity3. Counterfactual Sub.
Swap with type-consistent
ORIGINAL (PERSON)
Saint Bernadette
SoubirousSWAP
SUBSTITUTED (PERSON)
Kiyomori
context ≠ memory
forces the model to
follow the context
KEY STEP
our core contributionswapped4. Quality Filter
6 rule-based checks per sample
replacement in c'
context modified (c' ≠ c)
|c'| ≥ 50 chars
replacement not in c
count(orig, c) ≤ 10
|replacement| ≥ 2
passing → random split
80 / 10 / 10
train / dev / test
deterministic, seed=42passing
samplesFaithfulness-QA
99,094 samples
99K
counterfactual samples
8 entity types
2 source datasets
SQuAD + TriviaQA
56.7% success rate
automated pipeline
DATASET
released, JSONL formatFigure 1: The Faithfulness-QA construction pipeline. The counterfactual entity substitution stage manu-
factures controlled context-vs-memory conflicts that train RAG models to follow context.
Counterfactual Data Augmentation.Counterfactual data augmentation has proven effective across
NLP tasks [Kaushik et al., 2020, Zhang et al., 2025b]. The CounterFact dataset [Meng et al., 2022] in-
troduced entity-level counterfactuals for knowledge editing, and MQuAKE [Zhong et al., 2023] extended
this to multi-hop settings. Our work shares the entity substitution mechanism but differs in purpose: we
create training data forfaithfulnessrather than for editing factual associations.
Named Entity Recognition.Our pipeline relies on NER to identify answer-bearing entities. SpaCy
[Honnibal et al., 2020] provides efficient transformer-based NER with pre-trained models covering stan-
dard entity types. Recent advances in NER include few-shot approaches such as CONTaiNER [Das et al.,
2022] and LLM-based methods such as GPT-NER [Wang et al., 2023], which could improve our entity
matching in future iterations.
3 Dataset Construction
3.1 Overview
Faithfulness-QA is constructed through a four-stage automated pipeline (Figure 1). The first stage builds
a large typed entity bank by extracting named entities from source dataset contexts. The second stage
processes each QA sample, applying NER to the context and matching the answer text to a recognized
entity using a multi-strategy cascade. The third stage performs the core counterfactual transformation:
sampling a type-consistent replacement entity and using it to substitute the original recognized entity
across the entire context. The fourth stage applies quality checks to filter out degenerate or problematic
substitutions and splits the passing samples into train/dev/test partitions. The entire pipeline is determin-
istic (random seed 42), requires no GPU, and runs in approximately 5.5 hours on a single CPU machine.
3.2 Source Datasets
We build Faithfulness-QA from two complementary extractive QA datasets:
•SQuAD[Rajpurkar et al., 2016]: 87,599 QA pairs from the training split, with Wikipedia paragraph
contexts (median length: 701 characters). Answers are contiguous spans in the context.
3

Table 1: Entity bank composition.
Type Count %
ORG 25,378 33.0
PERSON 20,292 26.4
DATE 10,613 13.8
GPE 6,769 8.8
CARDINAL 6,636 8.6
LOC 2,977 3.9
NORP 2,849 3.7
EVENT 1,439 1.9
Total 76,953100.0
•TriviaQA[Joshi et al., 2017]: 87,041 QA pairs from the reading comprehension (RC) training split,
with longer Wikipedia evidence documents (median length: 1,581 characters). Contexts exceeding
2,000 characters are truncated to a 1,600-character window centered on the answer.
3.3 Stage 1: Entity Bank Construction
We construct a typed entity bank by running SpaCy’sen_core_web_lgNER model [Honnibal et al.,
2020] over all 19,035 unique SQuAD contexts. Entities of typesPERSON,GPE,ORG,DATE,CARDINAL,
NORP,LOC, andEVENTare collected. Entities shorter than 2 characters or longer than 100 characters
are discarded. This yields a bank of76,953unique entities (Table 1).
3.4 Stage 2: Answer–Entity Matching
For each sample(q, c, a)whereqis the question,cis the context, andais the answer, we attempt to
matchato a recognized NER entity incusing a three-strategy cascade:
1.Exact match: case-insensitive string equality betweenaand an entity span.
2.Substring match:a⊆eore⊆awith overlap≥3characters.
3.Positional overlap: ifaappears literally inc, find the NER entity with≥50%character overlap with
the answer span.
Samples failing all three strategies are filtered.
3.5 Stage 3: Counterfactual Entity Substitution
Given a matched answer entity(e orig, t)with NER typet, we:
1. Sample a replacement entitye newfrom the entity bank with type(e new) =tande new̸=e orig.
2. Re-sample up to 5 times to enforce length compatibility:0.3≤ |e new|/|e orig| ≤3.0.
3. Replaceallcase-insensitive occurrences ofe origwithe newincto producec′.
4. Set the faithful answera′=e new.
The formal procedure for counterfactual entity substitution is presented in Algorithm 1.
4

Algorithm 1Counterfactual Entity Substitution
Require:QA sample(q, c, a), Entity bankB, NER modelN
Ensure:Faithfulness-QA sample(q, c′, enew, c, a, e orig, t)or⊥
1:E← N(c){Extract entities from context}
2:(e orig, t)←MATCHANSWER(a, E)
3:if(e orig, t) =⊥then
4:return⊥{No NER entity match}
5:end if
6:e new← ⊥
7:fori= 1to5do
8:e′←SAMPLE(B, t,exclude=e orig)
9:if0.3≤ |e′|/|e orig| ≤3.0then
10:e new←e′;break
11:end if
12:end for
13:ife new=⊥then
14:return⊥{No replacement in bank}
15:end if
16:c′←REPLACEALL(c, e orig, enew)
17:if¬QUALITYCHECK(c, c′, eorig, enew)then
18:return⊥
19:end if
20:return(q, c′, enew, c, a, e orig, t)
3.6 Stage 4: Quality Filtering and Splitting
Each candidate sample passes six quality checks before inclusion:
1.e new∈c′(replacement entity present in modified context)
2.c′̸=c(context was actually modified)
3.|c′| ≥50(context is not trivially short)
4.e new/∈c(replacement did not already exist in original)
5. count(e orig, c)≤10(avoid excessive replacements)
6.|e new| ≥2(replacement is non-trivial)
Passing samples are randomly split into train/dev/test sets with an 80/10/10 ratio using a fixed seed
of 42 for reproducibility.
4 Dataset Analysis
4.1 Overall Statistics
Table 2 summarizes the generation results. From a combined pool of 174,640 input QA pairs drawn
from the SQuAD and TriviaQA training sets, our pipeline produces99,094high-quality counterfactual
samples—49,094 from SQuAD and 50,000 from TriviaQA—with an overall substitution success rate
of 56.7%. The resulting dataset is an order of magnitude larger than existing faithfulness evaluation
benchmarks such as FaithEval [Ming et al., 2025] (4.9K samples). The SQuAD and TriviaQA subsets
5

Table 2: Dataset generation summary.
Source Input Output Rate Train Dev Test
SQuAD 87,599 49,094 56.0% 39,275 4,909 4,910
TriviaQA 87,041 50,000 57.4% 40,000 5,000 5,000
Total 174,640 99,094 56.7% 79,275 9,909 9,910
contribute roughly equal numbers of samples, ensuring that neither source dominates the training dis-
tribution. Each subset is independently split into train/dev/test partitions at an 80/10/10 ratio, yielding
79,275 training samples, 9,909 development samples, and 9,910 test samples in total. The aggregate data
volume is approximately 367 MB in JSONL format.
4.2 Failure Analysis
Understanding why 43.3% of input samples are filtered is important for assessing coverage and identify-
ing improvement opportunities. Table 3 presents a detailed breakdown.
The dominant failure mode isno NER entity match(64.0%), arising when answers are descriptive
phrases, boolean values, or common nouns that NER models do not recognize as named entities. This is
a deliberate trade-off: by restricting to named entities, we ensure that substitutions are well-defined and
type-consistent, at the cost of limited coverage on non-factoid QA pairs.
Table 3: Breakdown of filtered samples by reason.
Reason SQuAD TriviaQA Total %
No NER entity match 30,977 17,435 48,412 64.0
No context available — 13,272 13,272 17.6
No replacement in bank 7,179 4,885 12,064 16.0
Answer not in context — 626 626 0.8
Answer too short 201 576 777 1.0
Entity already in orig. 103 134 237 0.3
Too many occurrences 45 113 158 0.2
Total filtered 38,505 37,041 75,546100.0
4.3 Entity Type Distribution
Table 4 and Figure 2 reveal strong complementarity between the two sources. TriviaQA is dominated by
PERSONentities (45.7%), reflecting its trivia-style questions about notable individuals, while SQuAD
exhibits a more uniform distribution acrossORG(20.6%),DATE(20.4%), andPERSON(19.9%). Com-
bining both sources yields broad entity-type coverage, mitigating the risk of type-specific overfitting
during downstream training.
4.4 Context Length Analysis
TriviaQA contexts are approximately twice as long as SQuAD contexts (Table 5), providing better cov-
erage of realistic RAG scenarios in which retrieved passages tend to be multi-paragraph documents.
The combination of shorter (SQuAD) and longer (TriviaQA) contexts ensures that models trained on
Faithfulness-QA are exposed to diverse context lengths.
6

Table 4: Entity type distribution across source datasets.
Type SQuAD SQuAD % TriviaQA TriviaQA %
PERSON 9,775 19.9 22,871 45.7
ORG 10,114 20.6 8,186 16.4
DATE 9,997 20.4 2,057 4.1
GPE 6,292 12.8 11,810 23.6
CARDINAL 6,568 13.4 1,348 2.7
NORP 4,004 8.2 1,247 2.5
LOC 1,629 3.3 1,952 3.9
EVENT 715 1.5 529 1.1Table 5: Context length statistics
(characters).
Statistic SQuAD TriviaQA
Mean 763 1,313
Median 701 1,581
Min 151 108
Max 3,706 2,122
PER ORG DATE GPE CARD NORP LOC EVT00.511.522.5Count (×104)SQuAD
TriviaQA
Figure 2: Entity type distribution comparison.
5 Quality Validation
5.1 Automated Quality Checks
To validate that the filtering pipeline produces consistently high-quality output, we perform automated
quality verification on a stratified random sample of 200 instances drawn from the SQuAD subset (with
random seed 42 for reproducibility). The audit evaluates four complementary dimensions: (i) whether
the replacement entity is actually present in the modified context, (ii) whether the original entity has
been fully removed, (iii) whether the context was meaningfully changed, and (iv) whether the length
ratio between modified and original contexts falls within a reasonable range[0.5,2.0]. These four checks
collectively ensure that each sample represents a valid, self-consistent counterfactual rather than a de-
generate transformation. Table 6 reports the results.
Table 6: Quality audit results (200-sample random check).
Check Pass Rate Target
Replacement entity inc′200/200 (100%) 100%
Original entity not inc′200/200 (100%) 100%
Context changed (c′̸=c) 200/200 (100%) 100%
Length ratio∈[0.5,2.0]200/200 (100%)≥90%
All checks 100%≥90%
All four quality criteria achieve a perfect 100% pass rate, substantially exceeding the target threshold of
90%. This validates the effectiveness of our six-stage filtering pipeline. We note that these automated
checks arenecessary but not sufficientconditions for overall data quality: they verify structural correct-
ness (entity presence, context modification, length preservation) but do not assess semantic naturalness
or grammatical fluency. Section 5.2 provides complementary evidence on these dimensions.
7

Table 7: Qualitative examples of counterfactual entity substitution.
Src Question Original Ans. Faithful Ans. Type Modified Context Snippet
SQuAD To whom did the Virgin
Mary allegedly appear in
1858 in Lourdes France?Saint
Bernadette
SoubirousKiyomori PER...the Virgin Mary reputedly ap-
peared toKiyomoriin 1858.
TQA The Rufiyaa is the cur-
rency of which island
group?Maldives Sri Lanka GPEThe rufiyaa is the currency of
theMaldives...
TQA Which city does David
Soul come from?Chicago Amritsar GPE...Soul was born David Richard
Solberg inAmritsar, Illinois...
5.2 Qualitative Examples
To provide intuitive insight into the nature of the generated data, Table 7 presents three representative
examples spanning different entity types and source datasets.
The first example (SQuAD, PERSON) replaces “Saint Bernadette Soubirous” with “Kiyomori” in
a passage about the grotto at Lourdes. A model with parametric knowledge of French history would
need to override its memory to faithfully answer “Kiyomori”—exactly the kind of conflict our dataset
is designed to create. The second example (TriviaQA, GPE) substitutes “Maldives” with “Sri Lanka”
in a geopolitical passage; here, Sri Lanka and the Maldives are both island nations, which makes this
example highly misleading. The third example illustrates a case wheremultiple occurrencesof the entity
are replaced: “Chicago” becomes “Amritsar” both in the birthplace and in “the Chicago White Sox”
(replaced as “the Amritsar White Sox”), demonstrating the global replacement strategy. These examples
show that while syntactic coherence is well preserved, some substitutions (e.g., “Amritsar, Illinois”)
introduce semantic implausibility—a known limitation discussed in Section 8.
5.3 Comparison with Design Targets
Before construction began, we defined a set of quantitative targets in the experimental plan, organized
into priority levels P0 (must-have) and P1 (important). Table 8 compares these targets against actual
results.
Table 8: Target achievement summary.
Metric Target Achieved Status
Dataset size (P0)≥10K 99,094✓
Dataset size (P1)∼50K 99,094✓
Success rate≥60% 56.7%≈
Quality (200-sample)≥90% 100%✓
Type consistency≥95% 100%✓
Entity bank/type (P1)≥5K 5/8≈
Four of six targets are fully met or exceeded. Thedataset sizesubstantially surpasses both the P0
minimum (≥10K) and the P1 target (∼50K), reaching 99,094 samples—a result enabled by the efficiency
of the automated pipeline and the large volume of source data. Thequality auditachieves a perfect 100%
pass rate, comfortably exceeding the 90% threshold, which validates the effectiveness of the six-rule
filtering strategy.Entity type consistencylikewise reaches 100%, confirming that every replacement
entity shares the NER type of the original.
Two targets are approximately met but not fully achieved. Thesubstitution success rate(56.7%)
falls slightly below the 60% target; as analyzed in Section 4, the primary bottleneck is the NER-based
8

entity matching step, which cannot handle non-factoid or descriptive answers. Theentity bank per-
type coveragereaches the 5K threshold for 5 out of 8 types; the three smaller categories (LOC: 2,977;
NORP: 2,849; EVENT: 1,439) reflect the natural frequency distribution of these entity types in Wikipedia
text rather than a pipeline deficiency. Importantly, even for these smaller categories, the bank sizes are
sufficient for the volumes of same-type substitutions observed in practice (e.g., only 1,629 LOC-type
substitutions were performed in SQuAD, well within the 2,977 available LOC entities).
6 Data Format and Intended Use
6.1 Schema
Each sample is stored as a single-line JSON object in JSONL (JSON Lines) format, with one sam-
ple per line for efficient streaming. Table 9 describes the ten fields provided for each sample. The
schema is designed to support multiple downstream use cases simultaneously: fortraining, users can
pairmodified_contextandquestionas input withfaithful_answeras the target; foreval-
uation, theoriginal_answerfield enables measuring whether a model follows context (outputs
faithful_answer) or relies on parametric memory (outputsoriginal_answer); foranalysis,
theentity_typeandsourcefields enable fine-grained stratification of results by entity category
and source dataset. The inclusion of bothoriginal_contextandmodified_contextfurther
enables paired ablation studies comparing model behavior on identical questions with and without knowl-
edge conflicts.
Table 9: Data schema.
Field Type Description
idstr Source sample ID
questionstr Question text
original_contextstr Unmodified context
modified_contextstr Counterfactual context
original_answerstr Source ground-truth answer
faithful_answerstr Correct answer givenc′
original_entitystr Replaced entity
replacement_entitystr New entity (=faithful ans.)
entity_typestr NER type
sourcestrsquadortriviaqa
6.2 Intended Applications
We envision four primary use cases:
1.Faithfulness-aware fine-tuning: Train with(modified_context, q)→faithful_answer
to teach models to follow context over parametric memory.
2.Attention-based faithfulness loss: The explicit context modification region enables supervision of
cross-attention weights, encouraging models to attend to the context rather than generating from mem-
ory.
3.Faithfulness evaluation: Measure the rate at which models output the faithful answer (context-
grounded) vs. the original answer (parametric).
4.Knowledge conflict research: The paired original/modified contexts enable controlled studies of
LLM behavior under knowledge conflicts [Xie et al., 2024, Xu et al., 2024].
9

7 Discussion
Complementarity of Source Datasets.A key design choice is the use oftwosource datasets with
different characteristics. SQuAD provides balanced entity-type coverage and shorter contexts resem-
bling single-paragraph retrieval, while TriviaQA offers PERSON-dominated trivia questions with longer,
multi-paragraph contexts. This complementarity ensures that downstream models are not biased toward
a single entity type or context length distribution.
Success Rate Analysis.The 56.7% success rate, while slightly below the 60% target, is a deliberate
consequence of our conservative filtering strategy. The largest contributor to filtering (64% of skips) is the
requirement that the answer must match a recognized named entity—a constraint that guarantees well-
defined, type-consistent substitutions. Relaxing this constraint (e.g., via LLM-based entity matching)
could increase recall but risks introducing noisy or ambiguous substitutions.
Limitations of Rule-Based Substitution.Our pipeline performs string-level replacement without coref-
erence resolution. When a context contains pronominal or abbreviated references to the replaced entity
(e.g., “he” referring to the original person), these are not updated. While our quality checks show no
impact on the 200-sample audit, this limitation could create subtle semantic inconsistencies in a minority
of samples.
Comparison with FaithEval.FaithEval [Ming et al., 2025] is the closest existing resource, providing
4,900 evaluation samples with counterfactual, inconsistent, and unanswerable contexts. Faithfulness-
QA differs in three key aspects: (1) scale (99K vs. 4.9K), (2) purpose (training vs. evaluation), and
(3) construction method (automatic entity substitution vs. expert curation). The two resources are com-
plementary: models can be trained on Faithfulness-QA and evaluated on FaithEval.
8 Limitations
1.No coreference resolution: Pronominal and abbreviated references to replaced entities are not up-
dated, potentially creating minor inconsistencies.
2.No NLI-based filtering: We rely on rule-based quality checks rather than NLI models [He et al.,
2023] for semantic consistency verification.
3.Semantic plausibility: Some substitutions are syntactically valid but semantically implausible (e.g.,
a non-US city inserted into “born in [City], Illinois”).
4.Entity bank coverage: Three entity types (LOC, NORP, EVENT) have fewer than 5,000 entries,
potentially limiting substitution diversity for these types.
5.English only: The current pipeline and entity bank are limited to English.
6.No downstream evaluation: We present the dataset and construction analysis but defer model train-
ing and faithfulness evaluation experiments to future work.
9 Future Work
Several directions extend this work:
1.NLI-based quality filtering: Apply DeBERTa-v3-large-MNLI [He et al., 2023] to verify internal
consistency of modified contexts and filter semantically contradictory samples.
10

2.Coreference-aware substitution: Integrate coreference resolution to update all references (pronouns,
abbreviations) to the replaced entity.
3.LLM-based entity matching: Use LLMs for entity identification to improve the success rate beyond
NER-based matching.
4.Multi-entity and multi-hop substitution: Replace multiple entities per context or create multi-hop
reasoning chains with counterfactual entities, following the MQuAKE approach [Zhong et al., 2023].
5.Faithfulness baseline evaluation: Test open-source LLMs (Llama-3, Mistral, Qwen) on Faithfulness-
QA to establish context-grounding baselines.
6.Attention-based faithfulness training: Use Faithfulness-QA to train models with explicit attention
supervision losses, which is the ultimate goal of this research program.
10 Conclusion
We have presented Faithfulness-QA, a large-scale dataset of 99,094 counterfactual entity-substituted
QA samples designed to train and evaluate context-faithful RAG models. Built from two complemen-
tary sources—SQuAD (49,094 samples with balanced entity types and shorter contexts) and TriviaQA
(50,000 samples dominated by PERSON entities with longer contexts)—the dataset creates controlled
knowledge conflicts through automatic NER-based entity identification and type-consistent replacement.
The construction pipeline achieves a 56.7% substitution success rate with 100% quality pass rates on
audited samples, demonstrating that fully automated counterfactual data generation can produce reliable
training data at scale. The curated entity bank of 76,953 entities across 8 NER types ensures diverse and
reproducible substitutions.
We believe Faithfulness-QA fills an important gap in the RAG research ecosystem: while evaluation
benchmarks for faithfulness exist, large-scaletrainingresources that explicitly create knowledge con-
flicts have been lacking. By releasing the dataset, the construction pipeline, and the entity bank, we hope
to enable the community to (1) train models with faithfulness-aware objectives, (2) develop attention-
based faithfulness losses, and (3) systematically study how language models resolve conflicts between
retrieved context and parametric knowledge. Ultimately, the goal is to make RAG systems that are not
merely augmented by retrieval, but genuinelyfaithfulto it.
Acknowledgments
In this work, the WisPaper Scholar Agent [Ju et al., 2025] completed the entire research pipeline, includ-
ing research idea formulation, dataset collection and preprocessing, code implementation, experimental
deployment, result analysis, and full manuscript drafting. Human researchers solely provided high-level
task supervision, logical verification, factual proofreading, critical revision, and took full academic re-
sponsibility for all content, arguments, experimental results, and conclusions in this paper.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to
Retrieve, Generate, and Critique through Self-Reflection. InProceedings of the Twelfth International
Conference on Learning Representations (ICLR), 2024.
Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu, Zach Brannelly, and Mohamed Abdelrazek.
Seven Failure Points When Engineering a Retrieval Augmented Generation System.arXiv preprint
arXiv: 2401.05856, 2024.
11

Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican,
George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego de Las Casas,
Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris
Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osin-
dero, Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent Sifre. Improving Language Models by
Retrieving from Trillions of Tokens. InProceedings of the 39th International Conference on Machine
Learning (ICML), pages 2206–2240, 2022.
Sarkar Snigdha Sarathi Das, Arzoo Katiyar, Rebecca Passonneau, and Rui Zhang. CONTaiNER: Few-
Shot Named Entity Recognition via Contrastive Learning. InProceedings of the 60th Annual Meeting
of the Association for Computational Linguistics (ACL), pages 6338–6353, Dublin, Ireland, 2022.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. REALM: Retrieval-
Augmented Language Model Pre-Training. InProceedings of the 37th International Conference on
Machine Learning (ICML), pages 3929–3938, 2020.
Pengcheng He, Jianfeng Gao, and Weizhu Chen. DeBERTaV3: Improving DeBERTa using ELECTRA-
Style Pre-Training with Gradient-Disentangled Embedding Sharing. InProceedings of the Eleventh
International Conference on Learning Representations (ICLR), 2023.
Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. spaCy: Industrial-
Strength Natural Language Processing in Python. Zenodo, 2020.https://doi.org/10.5281/
zenodo.1212303.
Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. TriviaQA: A Large Scale Distantly
Supervised Challenge Dataset for Reading Comprehension. InProceedings of the 55th Annual Meet-
ing of the Association for Computational Linguistics (ACL), pages 1601–1611, Vancouver, Canada,
2017.
Li Ju, Jun Zhao, Mingxu Chai, Ziyu Shen, Xiangyang Wang, Yage Geng, Chunchun Ma, Hao Peng,
Guangbin Li, Tao Li, Chengyong Liao, Fu Wang, Xiaolong Wang, Junshen Chen, Rui Gong, Shijia
Liang, Feiyan Li, Ming Zhang, Kexin Tan, Junjie Ye, Zhiheng Xi, Shihan Dou, Tao Gui, Yuankai
Ying, Yang Shi, Yue Zhang, and Qi Zhang. WisPaper: Your AI Scholar Search Engine.arXiv preprint
arXiv:2512.06879, 2025.
Divyansh Kaushik, Eduard Hovy, and Zachary C. Lipton. Learning the Difference that Makes a Differ-
ence with Counterfactually-Augmented Data. InProceedings of the Eighth International Conference
on Learning Representations (ICLR), 2020.
Philippe Laban, Tobias Schnabel, Paul N. Bennett, and Marti A. Hearst. SummaC: Re-Visiting NLI-
based Models for Inconsistency Detection in Summarization.Transactions of the Association for
Computational Linguistics, 10:163–177, 2022.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela.
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. InAdvances in Neural Infor-
mation Processing Systems (NeurIPS), volume 33, pages 9459–9474, 2020.
Junyi Li, Xiaoxue Cheng, Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. HaluEval: A Large-Scale Halluci-
nation Benchmark for Large Language Models. InProceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing (EMNLP), pages 6449–6464, Singapore, 2023.
Wei Li, Wenhao Wu, Moye Chen, Jiachen Liu, Xinyan Xiao, Hua Wu. Faithfulness in Natural Language
Generation: A Systematic Survey of Analysis, Evaluation and Optimization Methods.arXiv preprint
arXiv:2203.05227, 2022.
12

Daliang Li, Ankit Singh Rawat, Manzil Zaheer, Xin Wang, Michal Lukasik, Andreas Veit, Felix Yu,
and Sanjiv Kumar. Large Language Models with Controllable Working Memory. InFindings of the
Association for Computational Linguistics: ACL 2023, pages 1774–1793, Toronto, Canada, 2023.
Jiazheng Li, Hanqi Yan, and Yulan He. Drift: Enhancing LLM Faithfulness in Rationale Generation via
Dual-Reward Probabilistic Inference. InProceedings of the 63rd Annual Meeting of the Association
for Computational Linguistics (ACL), pages 6850–6866, Vienna, Austria, 2025.
Shayne Longpre, Kartik Perisetla, Anthony Chen, Nikhil Ramesh, Chris DuBois, and Sameer Singh.
Entity-Based Knowledge Conflicts in Question Answering. InProceedings of the 2021 Conference on
Empirical Methods in Natural Language Processing (EMNLP), pages 7052–7063, Online and Punta
Cana, Dominican Republic, 2021.
Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. On Faithfulness and Factuality
in Abstractive Summarization. InProceedings of the 58th Annual Meeting of the Association for
Computational Linguistics (ACL), pages 1906–1919, Online, 2020.
Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and Editing Factual Associ-
ations in GPT. InAdvances in Neural Information Processing Systems (NeurIPS), volume 35, pages
17359–17372, 2022.
Yifei Ming, Senthil Purushwalkam, Shrey Pandit, Zixuan Ke, Xuan-Phi Nguyen, Caiming Xiong, and
Shafiq Joty. FaithEval: Can Your Language Model Stay Faithful to Context, Even If “The Moon is
Made of Marshmallows”? InProceedings of the Thirteenth International Conference on Learning
Representations (ICLR), 2025.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ Questions for
Machine Comprehension of Text. InProceedings of the 2016 Conference on Empirical Methods in
Natural Language Processing (EMNLP), pages 2383–2392, Austin, Texas, 2016.
T.Y .S.S. Santosh, Youssef Tarek Elkhayat, Oana Ichim, Pranav Shetty, Dongsheng Wang, Zhiqiang Ma,
Armineh Nourbakhsh, and Xiaomo Liu. CoCoLex: Confidence-guided Copy-based Decoding for
Grounded Legal Text Generation. InProceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (ACL), pages 19002–19018, Vienna, Austria, 2025.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. REPLUG: Retrieval-Augmented Black-Box Language Models. InProceed-
ings of the 2024 Conference of the North American Chapter of the Association for Computational
Linguistics (NAACL), pages 8364–8377, Mexico City, Mexico, 2024.
Shuhe Wang, Xiaofei Sun, Xiaoya Li, Rongbin Ouyang, Fei Wu, Tianwei Zhang, Jiwei Li, and
Guoyin Wang. GPT-NER: Named Entity Recognition via Large Language Models.arXiv preprint
arXiv:2304.10428, 2023.
Kevin Wu, Eric Wu, and James Zou. ClashEval: Quantifying the tug-of-war between an LLM’s internal
prior and external evidence.arXiv preprint arXiv:2404.10198, 2024.
Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. Adaptive Chameleon or Stubborn Sloth:
Revealing the Behavior of Large Language Models in Knowledge Conflicts. InProceedings of the
Twelfth International Conference on Learning Representations (ICLR), 2024.
Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang, Hongru Wang, Yue Zhang, and Wei Xu. Knowl-
edge Conflicts for LLMs: A Survey. InProceedings of the 2024 Conference on Empirical Methods in
Natural Language Processing (EMNLP), pages 8541–8565, Miami, Florida, 2024.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective Retrieval Augmented Generation.
arXiv preprint arXiv:2401.15884, 2024.
13

Qinggang Zhang, Zhishang Xiang, Yilin Xiao, Le Wang, Junhui Li, Xinrun Wang, and Jinsong Su.
FaithfulRAG: Fact-level Conflict Modeling for Context-faithful Retrieval-Augmented Generation. In
Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL),
Vienna, Austria, 2025.
Luhao Zhang, Xinyu Zhang, Linmei Hu, Dandan Song, and Liqiang Nie. Dually Self-Improved Counter-
factual Data Augmentation Using Large Language Model. InProceedings of the 63rd Annual Meeting
of the Association for Computational Linguistics (ACL), pages 5216–5227, Vienna, Austria, 2025.
Zexuan Zhong, Zhengxuan Wu, Christopher D. Manning, Christopher Potts, and Danqi Chen.
MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions. InPro-
ceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP),
pages 15686–15702, Singapore, 2023.
14
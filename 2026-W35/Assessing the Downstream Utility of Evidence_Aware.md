# Assessing the Downstream Utility of Evidence-Aware Retrieval in RAG

**Authors**: Utshab Kumar Ghosh, Debayan Mukhopadhyay, Shubham Chatterjee

**Published**: 2026-08-26 20:07:45

**PDF URL**: [https://arxiv.org/pdf/2608.26379v1](https://arxiv.org/pdf/2608.26379v1)

## Abstract
Retrieval evaluation for retrieval-augmented generation (RAG) is increasingly designed around whether retrieved passages contain evidence that can support generation, rather than topical relevance alone. We study whether this closer alignment with downstream evidence needs also makes retrieval evaluation more useful for the decisions built from it.
  Across five retrieval benchmarks and an end-to-end TREC RAG 2025 setting, we examine an answer-support signal in four roles: comparing retrievers, guiding retrieval training and system selection, predicting downstream answer quality, and filtering the evidence supplied to a generator. The signal changes retrieval rankings, but its downstream value is not uniform. It does not reliably improve retriever training; the benefit of using it for system selection depends on how the generator is instructed to use the retrieved evidence; and retrieval scores based on it do not robustly predict answer quality on unseen topics. In a direct evidence intervention, human annotators confirm that filtering preferentially preserves passages containing useful answer evidence, yet different answer evaluators reach different conclusions about whether the resulting answers improve.
  These results show that making retrieval evaluation more closely reflect the evidence needed for generation does not by itself make every downstream use of that evaluation more reliable. RAG evaluation methods should therefore be assessed with respect to the particular comparisons, decisions, and conclusions they are intended to support.

## Full Text


<!-- PDF content starts -->

Assessing the Downstream Utility of Evidence-Aware Retrieval in
RAG
Utshab Kumar Ghosh
Department of Computer Science
Missouri University of Science and
Technology
Rolla, MO, USA
u.ghosh@mst.eduDebayan Mukhopadhyay
University of Calcutta
Kolkata, West Bengal, India
debayan.mukherjee14@gmail.comShubham Chatterjee
Department of Computer Science
Missouri University of Science and
Technology
Rolla, MO, USA
shubham.chatterjee@mst.edu
Abstract
Retrieval evaluation for retrieval-augmented generation (RAG) is
increasingly designed around whether retrieved passages contain
evidence that can support generation, rather than topical relevance
alone. We study whether this closer alignment with downstream
evidence needs also makes retrieval evaluation more useful for the
decisions built from it.
Across five retrieval benchmarks and an end-to-end TREC RAG
2025 setting, we examine an answer-support signal in four roles:
comparing retrievers, guiding retrieval training and system selec-
tion, predicting downstream answer quality, and filtering the evi-
dence supplied to a generator. The signal changes retrieval rankings,
but its downstream value is not uniform. It does not reliably im-
prove retriever training; the benefit of using it for system selection
depends on how the generator is instructed to use the retrieved
evidence; and retrieval scores based on it do not robustly predict
answer quality on unseen topics. In a direct evidence intervention,
human annotators confirm that filtering preferentially preserves
passages containing useful answer evidence, yet different answer
evaluators reach different conclusions about whether the resulting
answers improve.
These results show that making retrieval evaluation more closely
reflect the evidence needed for generation does not by itself make
every downstream use of that evaluation more reliable. RAG evalu-
ation methods should therefore be assessed with respect to the par-
ticular comparisons, decisions, and conclusions they are intended
to support.
CCS Concepts
•Information systems →Evaluation of retrieval results;Rel-
evance assessment;Test collections;Retrieval models and ranking;
Question answering.
Keywords
Retrieval-augmented generation, Retrieval evaluation, Answer sup-
port, Evaluation validity, LLM judges, System selection
1 Introduction
Retrieval-augmented generation (RAG) depends on retrieving evi-
dence that can support the generated answer, motivating evaluation
criteria that go beyond topical relevance. TREC RAG 2025 [ 35], for
example, distinguishes merely related passages from those that
cover requested sub-narratives, while RAGTIME 2025 [ 23] rewards
information useful for constructing the final report. This shift is also
empirically motivated: downstream-aware retrieval evaluation canalign more closely with RAG performance [ 30], and utility-aware
retrieval or evidence selection can improve generation itself [ 28].
Together, these developments suggest that an evaluation signal
better aligned with answer-bearing evidence may also provide a
better basis for improving and choosing retrievers.
That inference, however, has not been established. An evalua-
tion criterion can change which retrievers appear better without
making the resulting comparison a better basis for training, system
selection, or predicting downstream quality. Thus, showing that
answer support is a more meaningful retrieval criterion is distinct
from showing that the decisions subsequently built from it are
empirically justified.
This motivates our central question:If retrieval evaluation is
better aligned with the evidence that RAG needs, how far does that
improvement carry through to the downstream decisions built from it?
We follow the same answer-support signal through a sequence of
increasingly consequential uses: comparing retrievers, using it for
training and system selection, predicting answer quality, and finally
determining whether an intervention that improves the retrieved
evidence enhances the RAG system.
We find that the benefit does not propagate automatically. Answer-
support-aware evaluation changes retrieval conclusions, but using
the same signal for training does not reliably improve retrieval. The
retrievers it favors improve held-out answers under one generation
regime but not another, and its retrieval scores do not robustly
predict answer quality on unseen topics. These failures cannot be
dismissed as consequences of an arbitrary evidence signal: in our
experiments, human annotators confirm that answer-support fil-
tering gives the generator more answer-bearing evidence. We then
evaluate the same generated answers with two different answer
evaluators. Qwen finds that the filtering improves answer quality,
whereas Claude finds essentially no improvement. Thus, even after
verifying that the generator received better evidence, whether the
resulting answers are judged to be better depends on how answer
quality is measured.
Together, these findings expose what we call thevalidity-composition
problem: evidence supporting an evaluation signal for one use does
not automatically support the next inference or decision built from
it. This is consistent with the broader measurement principle that
validity concerns particular interpretations and uses of a mea-
sure [ 21,27], but we show empirically how this problem arises
across the stages of a modern RAG pipeline. The methodological
implication is that the validity of an evaluation signal must be es-
tablished with respect to the decisions it is used to support, not
merely with respect to the construct it is intended to measure.
arXiv:2608.26379v1  [cs.IR]  26 Aug 2026

Conference’17, July 2017, Washington, DC, USA Ghosh, Mukhopadhyay, and Chatterjee
Contributions.We make the following contributions:
•We show when evidence-aware retrieval evaluation changes
system conclusions.Large disagreement with answer-bearing
evidence is not sufficient: the effect becomes consequential when
that mismatch is system-selective, affecting competing retrievers
differently.
•We uncover a disconnect between better-aligned retrieval
evaluation and downstream RAG decisions.Answer-support-
aware evaluation can materially change which retriever appears
best without reliably improving optimization, held-out system
selection, or prediction of answer quality.
•We show that this disconnect persists even after validating
the evidence improvement with humans.Independent anno-
tators support the evidence distinction and confirm that answer-
support filtering preferentially preserves evidence-bearing pas-
sages. Yet, with the generated outputs held fixed, different answer
evaluators can lead to different conclusions about whether the
same intervention improved answer quality.
•We identify the resulting validity-composition problem for
RAG evaluation.Our findings show that evaluation methods
must be validated for the decisions they are intended to sup-
port, not only for the constructs they measure. This applies both
to using retrieval metrics for system choice and optimization
and to using answer evaluators to determine whether a RAG
intervention worked.
Below, we follow this chain from evidence judgments to end-to-end
conclusions. We test in turn whether answer-support-aware evalu-
ation changes retrieval comparisons, whether those comparisons
support better system decisions, whether retrieval scores predict
answer quality, and whether a human-supported improvement in
retrieved evidence yields a robust downstream conclusion.
2 Related Work
§1 raises a tension: evidence-aware retrieval is a natural response
to what RAG needs, yet a more meaningful signal need not support
better downstream decisions. Prior work makes both sides plau-
sible: evidence- and downstream-aware evaluation can improve
RAG, while IR and measurement research show that a measure’s
usefulness depends on the inference or decision it supports.
From Relevance to Evidence Utility in RAG.The move be-
yond topical relevance predates modern RAG evaluation. EXAM
evaluates retrieve-and-generate systems through questions that
the returned information should enable users to answer [ 31], while
subsequent work develops question-, nugget-, and rubric-based
measures of useful information [ 10,15]. RAGAS and ARES simi-
larly separate properties of retrieved context from faithfulness and
answer quality [ 12,29]. More recently, TREC RAG 2025 rewards ev-
idence that contributes to requested sub-narratives, and RAGTIME
2025 distinguishes answer-useful information from material that is
merely topical [23, 35].
This progression reflects a broader IR insight: individually rel-
evant results need not form a useful result set. Diversity-aware
evaluation, for example, rewards coverage of distinct aspects rather
than repeated evidence about the same aspect [ 6]. RAG sharpens
this distinction because retrieved material is not the final product;
it is evidence supplied to a generator. eRAG makes this downstreamrole explicit by evaluating retrieved documents through their effect
on generation and reports stronger alignment with RAG perfor-
mance than conventional relevance judgments [ 30]. Such results
make the expectation studied in this paper plausible: evaluation
designed around answer-useful evidence can be more informative
than relevance alone.
When Better Retrieval Leads to Better Generation.Other
work shows that downstream-aware retrieval signals can some-
times be acted upon successfully. Uplift-RAG defines document
utility through marginal contribution to generation and uses that
signal to improve reranking and evidence selection [ 28]. Lee et
al. [24] similarly estimate passage utility with respect to generator
behavior, while sub-question coverage has been used to measure
and improve whether retrieved evidence addresses important parts
of an information need [41].
At the same time, better retrieval does not determine better gen-
eration by itself. RGB shows that generators differ in their ability
to use noisy, incomplete, or conflicting evidence [ 4]; RAGGED and
related work report interactions among retriever, reader, and con-
text size [ 19,36]; and recent work distinguishes retrieval utility
from final answer quality rather than treating them as interchange-
able [ 34]. Thus, downstream-aware retrieval signalscanimprove
RAG, but their value depends on how the retrieved evidence is
used. What remains unclear is whether a signal that better captures
answer-useful evidence also becomes a better objective for opti-
mization, a better basis for system selection, or a useful predictor
of answer quality.
From Evaluation Reliability to Decision Validity.IR evalua-
tion has long separated properties of judgments from properties of
the system conclusions drawn from them. Assessor disagreement,
incomplete pools, topic sampling, and metric choice can all affect
effectiveness estimates [ 2,3,32,38,44]. Importantly, substantial
disagreement among assessors can coexist with stable relative sys-
tem rankings [ 38]. The converse is also important for our setting:
changing judgments or a leaderboard does not establish that the
new ordering is a better basis for choosing systems.
The growing use of LLMs for relevance assessment makes this
distinction particularly salient. Faggioli et al. caution against treat-
ing agreement with existing human assessments as sufficient justifi-
cation for replacing them with LLM judgments [ 13,14]. Clarke and
Dietz similarly argue that reproducing human relevance judgments
does not establish that LLM-generated qrels are safe reusable eval-
uation targets [ 7], while Dietz et al. articulate broader principles
for determining when LLM judges are appropriate [11].
Measurement theory provides the general principle behind these
concerns: validity supports a particular interpretation or use of a
measure, not the measure in isolation [ 21,27]. In Kane’s argument-
based framework, each additional inference or decision requires
supporting evidence. Applied to RAG, showing that answer support
captures a meaningful evidence distinction does not establish that
it is also a useful optimization objective, selection criterion, or pre-
dictor of downstream quality. These uses must be tested separately.
Evaluator Dependence in RAG Conclusions.The same issue
arises at the other end of the pipeline, where generated answers be-
come the objects of evaluation. LLM judges exhibit systematic biases
and sensitivity to evaluation conditions [ 43]. Recent work therefore
examines consequences beyond item-level agreement. JuStRank

Downstream Utility of Evidence-Aware Retrieval Conference’17, July 2017, Washington, DC, USA
studies LLM judges as system rankers and shows that aggregation
can expose biases not apparent from individual judgments [ 17]; the
Progress Illusion shows that strong aggregate meta-evaluation can
conceal poor discrimination among systems of similar quality [ 42].
RAGTIME reports differences between automatic and human eval-
uation of RAG runs [ 23], while Auto-Judge studies reliability and
vulnerabilities of judges for citation-grounded RAG [16].
For system development, however, the consequential question
is often not whether two evaluators assign the same scores, but
whether they support the same decision. If the generated outputs
are fixed, score disagreement is unsurprising. More consequential
is disagreement about whether an intervention improved the sys-
tem. Evaluator robustness should therefore be examined not only
through agreement on individual outputs or system rankings, but
also through the stability of the substantive conclusion drawn from
an experiment.
This section establishes that both evidence- and downstream-aware
signals can produce genuine improvements, while evaluation the-
ory cautions that success for one use does not establish success
for another. What remains unresolved is how far an improvement
in evidence evaluation actually carries:when retrieval evaluation
becomes better aligned with answer-bearing evidence, which down-
stream decisions does that improvement support?
3 From Evidence Judgments to RAG Decisions
We useanswer supportas a concrete probe of evidence-aware re-
trieval evaluation. A passage isanswer-supportingwhen it con-
tains evidence that contributes directly to answering the informa-
tion need, rather than merely discussing the same topic. To assess
this distinction, we first decompose each passage into atomic, self-
contained claims using Gemma3-27B. An answer-support judge
(GPT 4.1) evaluates these claims against the information need and
assigns a single 0–3 grade to the query–passage pair. Grade 0 indi-
cates no answer-supporting evidence, 1 means partial or insufficient
evidence, and 2–3 satisfy the answer-support criterion.
The central question of this paper is what follows once such
judgments are available. They can be used to change how retrievers
are evaluated, to guide system selection, to construct retrieval scores
treated as proxies for downstream answer quality, or to determine
which retrieved evidence reaches the generator. These are different
uses of the same underlying signal, and each supports a different
claim about what that signal can tell us. Figure 1 follows these
claims from retrieval evaluation to the final RAG conclusion.
A meaningful answer-support judgment does not guarantee that
these downstream uses are also justified. Changing the evidence cri-
terion may change which retriever appears better without making
that comparison a better basis for optimization or system selection.
A retrieval score may align more closely with downstream perfor-
mance without predicting answer quality on unseen topics. And
even an intervention that supplies better evidence does not by itself
establish that the resulting answers improved if that conclusion
depends on how answer quality is evaluated.
We call this thevalidity-composition problem: evidence support-
ing an evaluation signal for one use does not automatically validate
the next inference or decision built from it. This is RAG-specific
Answer -
Support 
JudgementsRetriever 
ComparisonOptimization/
System 
SelectionEvidence 
Supplied to 
GeneratorGenerated 
AnswersFinal 
Evaluation 
and 
Conclusion
RQ4 RQ1 RQ2
RQ3Does the retrieval score predict downstream answer 
quality?Does better 
evidence 
judgement 
improve retriever 
comparison?Does that 
comparison 
support better 
system choice?Does the 
chosen system 
provide better 
evidence?Does better 
retrieved 
evidence 
produce better 
answers?Can we reliably 
conclude that 
the intervention 
helped?Figure 1: From answer-support judgments to end-to-end RAG
conclusions. Each arrow represents a distinct empirical claim
tested by RQ1–RQ4. The dashed RQ3 link represents the
cross-stage claim that retrieval evaluation should predict
downstream answer quality.
shorthand for the established principle that validity concerns par-
ticular interpretations and uses of a measure [ 21]. The relevant
question is therefore not whether an evaluation signal is “valid”
in isolation, but which uses of that signal the available evidence
actually supports.
Research Questions.We test four links in this chain:
RQ1: When does answer-support-aware evaluation change
retrieval conclusions?We test when answer support changes re-
triever rankings, whether those changes exceed comparable random
perturbations of the judgments, and why some benchmarks are
more affected than others.
RQ2: Does answer support provide a better basis for opti-
mizing or selecting retrievers?We test whether using answer
support as retrieval supervision improves retrieval and whether
retrievers selected by answer-support-aware evaluation produce
better answers on held-out topics.
RQ3: Do answer-support-aware retrieval scores predict
answer quality?We test whether these scores provide reliable
out-of-sample information about downstream RAG performance.
RQ4: If the retrieved evidence improves, can we reliably
conclude that the RAG system improved?We use answer-
support judgments to change the evidence supplied to the generator,
verify with human judgments that the resulting context contains
more answer-bearing evidence, and test whether the conclusion
that the answers improved is robust to evaluator choice.
4 When Evidence-Aware Evaluation Matters
We begin with the first link in Figure 1: when does answer-support-
aware evaluation change how retrievers are compared? We test this
by evaluating the same retrieval runs under conventional relevance
and under a criterion that rewards answer-supporting evidence.
Experimental setup.We use five test sets. These include three
BEIR [ 33] collections: TREC-COVID [ 37] (50 queries), NFCorpus [ 1]
(323), and SciFact [ 39] (300), together with TREC-DL 2019 [ 9] (43)
and TREC-DL 2020 [ 8] (54). For the three BEIR collections, we
evaluate 10 systems: BM25, SPLADE-v3 [ 22], BGE [ 5], and Con-
triever [ 20] as first-stage retrievers, together with six rerankers over

Conference’17, July 2017, Washington, DC, USA Ghosh, Mukhopadhyay, and Chatterjee
the BM25 top-100. The rerankers are three cross-encoders (MiniLM-
L6, MiniLM-L12, and BGE-reranker-base) and three bi-encoders
(BGE, E5 [ 40], and GTE [ 25]). For TREC-DL 2019 and 2020, we
evaluate 11 systems: BM25, SPLADE-v3, BGE, TCT-ColBERT [ 26],
and TAS-B [ 18], together with the same six rerankers. First-stage
runs use Pyserini. NFCorpus has one retrieval-coverage asymmetry.
BM25 returns no results for 15 of its 323 queries, so BM25 and the
six BM25-seeded rerankers cover 308 queries, whereas the three
neural first-stage systems cover all 323. We retain the full 323-query
evaluation set, treating a system that returns no results for a query
as receiving zero effectiveness on that query.
Constructing answer-support-aware qrels.We use answer
support as a stricter criterion on passages already judged relevant,
rather than constructing a new judgment pool from scratch. We
first define the original positive set using each benchmark’s na-
tive relevance threshold: grade ≥1for BEIR and grade ≥2for
TREC-DL, and then apply the answer-support procedure from §3
to those positives. This design isolates the effect of requiring origi-
nally relevant passages to also contain answer-bearing evidence:
an original positive either retains its benchmark gain if its answer-
support grade is at least 2 or receives zero gain otherwise. We do
not add newly discovered positives from previously nonrelevant
or unjudged passages. This also gives a clean matched-random
counterfactual in which the same number of original positive judg-
ments can be removed at random, allowing us to ask whether the
particular positives rejected by the answer-support criterion alter
system conclusions more than comparable judgment attrition.
Comparing retrieval conclusions.We then hold every retrieval
run fixed and score the same rankings under the two qrel sets. A pas-
sage that survives answer-support filtering keeps its original rele-
vance grade; a filtered passage receives zero gain. We use nDCG@10
with linear gain and compare the resulting system orderings with
Kendall’s𝜏 𝑏.
Answer support changes every leaderboard.Relevance and an-
swer support disagree substantially. Among the three BEIR collec-
tions, 63.4% of SciFact positives satisfy the answer-support criterion,
compared with 43.4% for TREC-COVID and 18.6% for NFCorpus.
That difference changes the system ordering on every benchmark.
Table 1 summarizes the result. Kendall’s 𝜏𝑏between the original
and answer-support-aware rankings ranges from 0.378 to 0.746,
with 6–14 pairwise reversals. The top-ranked system changes on
four of the five collections.
System-selective mismatch.A large change in the qrels can it-
self move a leaderboard. We therefore compare answer-support
filtering with matched-random removal. For each query and rel-
evance grade, we remove exactly the same number of positive
judgments as answer-support filtering, but choose them randomly.
We repeat this100 ,000times. TREC-COVID is the only collection
whose observed reordering exceeds the matched-random distribu-
tion (𝑝rand=0.0243). The other four collections have 𝑝rand=0.1097–
0.4783. The amount of judgment removal does not account for
this difference: NFCorpus loses 81.4% of its positives, compared
with 56.6% for TREC-COVID, yet its reordering remains consistent
with matched-random removal. The distinguishing factor is how
unevenly the removed judgments benefit competing systems. ToTable 1: RQ1 results. 𝜏𝑏compares the original and answer-
support-aware system orderings. 𝑝randcompares the observed
reordering with matched-random judgment removal. Expo-
sure spread is the largest between-system difference in the
share of original DCG@10 contributed by positives removed
by answer-support filtering.
Collection𝜏 𝑏Rev.𝑝 rand Exp. spread
TREC-COVID 0.378 14 0.0243 33.9 pp
SciFact 0.556 10 0.3120 5.5 pp
DL2019 0.636 10 0.4783 5.4 pp
NFCorpus 0.733 6 0.1097 7.1 pp
DL2020 0.746 7 0.2108 5.3 pp
Figure 2: System-selective exposure to relevance–answer-
support mismatch. Each point represents a retrieval system
and shows the share of its original evaluated DCG@10 con-
tributed by positive judgments that fail the answer-support
criterion. TREC-COVID exhibits a 33.9 percentage-point
spread between systems, compared with at most 7.1 points
on the other collections.
measure how strongly each system depends on the judgments be-
ing removed, we define its exposure as the fraction of its original
DCG@10 gain contributed by positives that fail the answer-support
criterion. On TREC-COVID, this share is 47.5% for Contriever and
13.6% for SPLADE-v3, a 33.9 percentage-point (pp) difference. The
largest spread on any other collection is only 7.1 points (Figure 2).
Across these benchmarks, relevance–evidence mismatch be-
comes most consequential when it issystem-selective. A large amount
of mismatch can leave relative system comparisons largely intact
when it affects systems similarly. When the mismatched judgments
benefit competing systems differently, changing the evidence crite-
rion can substantially change the leaderboard.
The effect persists across judges.For this judge-sensitivity anal-
ysis, GPT-4.1, Qwen3-30B, and Llama-3-8B evaluate exactly the
same query–passage pairs from a shared BM25 top-100 candidate
pool. Binary answer-support agreement yields Gwet’s AC1 values
of 0.626–0.731 across collections. More importantly, all 15 judge–
collection combinations reorder the relevance-based leaderboard.

Downstream Utility of Evidence-Aware Retrieval Conference’17, July 2017, Washington, DC, USA
All three judges select the same answer-support-aware winner on
three collections, and two of three agree on the remaining two. The
RQ1 effect therefore persists across all three answer-support judges:
changing the evidence criterion changes retrieval conclusions.
Takeaway (RQ1).Retrieval conclusions are not independent of
what the benchmark counts as useful. Moving from topical rele-
vance to answer-supporting evidence can change which systems
appear better, especially when systems differ in how much their
retrieved relevance actually consists of answer-bearing evidence.
The important issue is therefore not only how much relevance and
evidence disagree, but whether that disagreement is distributed un-
evenly across systems. RQ2 asks whether acting on these changed
conclusions leads to better RAG systems.
5 Evidence-Aware System Decisions
§4 showed that answer-support-aware evaluation can change which
retrievers appear better. RQ2 asks whether acting on that signal
leads to better systems. We test two consequential uses: shaping
retriever training and selecting a retriever for downstream RAG.
Using answer support for retrieval training.We first ask whether
answer-support judgments provide useful training supervision.
Training setup.We fine-tune ms-marco-MiniLM-L-6-v2 with
a RankNet objective for three epochs (learning rate2 ×10−5), using
five query-disjoint folds and dev nDCG@10 for checkpoint selec-
tion. Before constructing the folds, we restrict each collection to
queries for which the BM25 candidate set contains at least one
training positive: a passage with original relevance grade ≥2for
NFCorpus and TREC-COVID, or grade ≥1for binary-qrel SciFact,
and answer-support grade ≥2. This yields 87 of 323 NFCorpus
queries, all 50 TREC-COVID queries, and 188 of 300 SciFact queries.
Every retained query is held out exactly once across the five folds.
Within each query, retrieved assumed negatives are separated
into answer-support grade 0 (C0: no answer-supporting evidence)
and grade 1 (C1: partial or insufficient evidence). We compare equal
C0/C1 weighting with a variant that gives C1 negatives1 .5×weight.
The matched control uses the same number of negatives from
the same candidate pool, matched by BM25 rank, but ignores the
answer-support distinction.
Training results.Answer-support-aware training does not re-
liably improve nDCG@10 over the matched control. With equal
weighting, the differences are +0.0027on NFCorpus (95% CI: −0.0050
to+0.0108),+0.0021on TREC-COVID (95% CI: −0.0059to+0.0098),
and+0.0072on SciFact (95% CI: −0.0026to+0.0185). Weighting C1
more strongly likewise yields no reliable gain: +0.0030,−0.0035, and
+0.0069, respectively, with every confidence interval including zero.
Thus, although answer support exposes distinctions that conven-
tional relevance collapses, using those distinctions as hard-negative
supervision does not reliably produce a better retriever.
From retrieval rankings to system selection.We next ask: if
answer-support-aware evaluation favors a different retriever, does
choosing that retriever improve downstream answers?
Experimental setup.We use the official TREC RAG 2025 re-
trieval task. Of its 22 judged topics, four are reserved for prompt
calibration and excluded from all reported results, leaving 18 eval-
uation topics. We begin from the 46 official retrieval submissions.
Forty satisfy our prespecified eligibility requirements: at least 95%coverage of the evaluation topics, retrieval depth of at least 100,
valid MS MARCO v2.1 segment identifiers with no within-topic
duplicates, a publicly documented automatic retrieval method with
no manual alteration of test-topic rankings, top-100 segment iden-
tifiers that all resolve to corpus text, and successful evaluation by
the released TREC retrieval scorer without modification.
We then construct a fixed downstream-analysis roster with one
representative run per team, avoiding over-representation of teams
with multiple submissions. We use a team-declared primary run
when available; otherwise, we choose that team’s eligible run with
the highest official original-qrel retrieval score. This de-duplication
yields 11 systems and is distinct from the cross-fitted retriever-
selection experiment below. Neither answer-support judgments
nor downstream answer quality is used to construct the roster.
For each system, we evaluate retrieval with nDCG@30 under
either the original TREC RAG 2025 relevance judgments or answer-
support-aware judgments. For RAG25, the same claim-based instru-
ment as in §4 uses the full topic narrative as the information need,
with GPT-4.1 as the primary judge. An originally relevant passage
contributes gain only when its answer-support grade is at least 2.
Generation conditions.We generate an answer for each retriever–
topic pair using Gemma3-27B and the passages returned by that
retriever. We evaluate two fixed generation strategies. TheStan-
dard Groundedprompt asks the model to answer the query as
fully as the supplied passages allow, use only information sup-
ported by those passages, cite every factual statement, and explic-
itly identify requested information that the passages do not support.
TheCoverage-Disciplinedprompt retains these requirements but
additionally encourages broader coverage of distinct supported
facts, nonredundant citations, atomic claims, explicit treatment of
supported negative findings, and representation of disagreement
among passages.
We use four held-out topics to choose the primary generation
condition before analyzing the remaining 18 topics. Under a prespec-
ified rule, Coverage-Disciplined would replace Standard Grounded
only if it improved coverage of fully supported vital nuggets by at
least0.02while reducing citation precision by no more than0 .01.
Although it improved citation precision, it reduced vital-nugget cov-
erage by0.027; Standard Grounded therefore remains the primary
condition, with Coverage-Disciplined retained as a prespecified
sensitivity condition.
Answer-quality evaluation.We evaluate the generated an-
swers with RAGDoll’s Nuggetizer, which compares an answer
against the information nuggets defined for each TREC RAG topic.
Our primary answer-quality measure is strict_vital_score : it
considers only nuggets designated as vital and gives credit only
when a vital nugget is fully supported by the generated answer;
partial support receives no credit. Qwen3-30B provides the primary
Nuggetizer judgments. Claude Sonnet 5 independently evaluates
the same generated answers as an evaluator sensitivity.
Cross-fitted system selection.Within the fixed 11-system ros-
ter, retriever selection is cross-fitted over the 18 evaluation topics.
In each fold, each retrieval criterion selects the system with the
highest mean nDCG@30 on the training topics, and we measure
that system’s answer quality on the held-out topics. Thus, held-out
topics do not participate in the fold-specific retriever selection, and
downstream answer quality never participates in selection.

Conference’17, July 2017, Washington, DC, USA Ghosh, Mukhopadhyay, and Chatterjee
Table 2: Held-out change in downstream answer quality
when retrievers are selected using answer-support-aware
rather than original relevance evaluation. Positive values
favor answer-support-aware selection.
Generation EvaluatorΔanswer quality 95% CI
Standard Grounded Qwen3-30B−0.0065[−0.0617,+0.0497]
Standard Grounded Claude Sonnet 5+0.0018[−0.0235,+0.0260]
Coverage-Disciplined Qwen3-30B+0.0583[+0.0202,+0.1069]
Coverage-Disciplined Claude Sonnet 5+0.0324[+0.0072,+0.0588]
Selection results.At the system-ranking level, answer-support-
aware evaluation appears promising. With Qwen evaluating an-
swer quality, Kendall’s 𝜏𝑏between retrieval effectiveness and down-
stream answer quality increases from0 .2364under conventional
relevance to0 .3455under answer-support-aware evaluation ( Δ𝜏=
+0.1091). With Claude evaluating the same generated answers, it
increases from0 .2364to0.4182( Δ𝜏=+ 0.1818). Both changes are
directionally favorable, although their topic-bootstrap confidence
intervals include zero.
The held-out decision tells a different story. Under the primary
Standard Grounded generation condition, selecting a retriever by
answer-support-aware rather than conventional relevance evalu-
ation does not improve downstream answer quality. With Qwen,
the mean held-out difference is −0.0065(95% CI[−0.0617,+0.0497],
𝑝=0.828); with Claude it is +0.0018(95% CI[−0.0235,+0.0260],
𝑝=0.902). The null result is not because the two evaluation criteria
make the same decision: in the prespecified five-fold assignment,
they select the same retriever in only one fold. Answer-support-
aware evaluation therefore changes which system is chosen, but
under the primary generation condition that changed choice does
not improve held-out answers.
Decision value depends on evidence use.We now repeat the
same selection experiment under the Coverage-Disciplined genera-
tion condition. The retrieval systems, answer-support judgments,
retrieval metric, cross-fitted selection procedure, and Gemma3-27B
generator remain unchanged; only the instructions governing how
the generator uses the retrieved evidence differ.
Answer-support-aware system selection improves held-out an-
swer quality. The gain is +0.0583with Qwen (95% CI [+0.0202,+0.1069],
𝑝= 0.007) and+0.0324with Claude (95% CI [+0.0072,+0.0588],
𝑝= 0.030). Selection regret—the gap between the selected re-
triever’s answer quality and that of the best available retriever on
the held-out topics—decreases by0.0569and0.0311, respectively.
The contrast identifies the central RQ2 result. Answer-support-
aware evaluation can change retrieval rankings and move them
closer to downstream answer-quality rankings without providing a
universally better system-selection rule. Under Standard Grounded,
the changed decision provides no held-out benefit. Under Coverage-
Disciplined, the same retrieval criterion and selection procedure
produce a reliable gain. The selection value of answer-support-
aware evaluation, therefore, depends on how the downstream gen-
erator is instructed to use the evidence that the criterion favors.Takeaway (RQ2).A better evidence signal is not automatically a
better decision rule. Answer-support distinctions do not reliably im-
prove retrieval training in our experiments, while answer-support-
aware system selection improves held-out answers only under one
generation regime. The value of an evaluation signal is therefore
partly a property of the pipeline and decision in which it is used,
not of the metric in isolation. RQ3 asks whether retrieval scores
themselves nevertheless generalize as predictors of downstream
answer quality.
6 Predictive Validity of Evidence-Aware Scores
RQ2 showed that answer-support-aware evaluation can align more
closely with downstream system rankings without consistently
supporting better system selection. RQ3 asks whether retrieval
scores themselves nonetheless provide useful information about
answer quality on unseen topics.
Predictive setup.We use the same 18 topics and 11 systems as
in §5. For each topic–system pair, we observe original nDCG@30
(𝑂), answer-support-aware nDCG@30 ( 𝐺), and downstream an-
swer quality. We fit three ordinary least-squares models: 𝑂predicts
answer quality from conventional retrieval effectiveness, 𝐺uses
answer-support-aware effectiveness, and 𝑂𝐺uses both. Thus, 𝐺
tests whether answer-support-aware evaluation is a better predic-
tor than conventional relevance, while 𝑂𝐺tests whether it adds
predictive information beyond conventional relevance.
We evaluate transfer with leave-one-topic-out cross-validation.
Within each training fold, retrieval features and answer quality
are centered within topic before estimating slopes. For the held-
out topic, only retrieval features are centered using that topic’s
retrieval scores; no held-out answer score is used. We report pooled
cross-validated 𝑅2relative to the pooled held-out mean, so negative
values indicate greater squared prediction error than that baseline.
Predictive validity therefore requires a relationship learned across
training topics to generalize to systems on an unseen topic.
Results.Under the primary Standard Grounded condition, none
of the retrieval features provides useful out-of-topic prediction.
With Qwen evaluation, 𝑅2
CVis−0.0726for𝑂,−0.0800for𝐺, and
−0.0802for𝑂𝐺. With Claude, the corresponding values are −0.0495,
−0.0491, and−0.0503. Thus, answer-support-aware retrieval scores
are no more predictive than conventional relevance in the primary
condition, and combining the two signals does not help.
The linear result is consistent across the full analysis: across
two generation prompts, five generator–grounding-judge configu-
rations, and two answer evaluators, all 60 leave-one-topic-out OLS
models have negative cross-validated 𝑅2. Adding answer-support-
aware retrieval to relevance improves 𝑅2
CVnumerically in only four
of the 20 conditions, and all four models remain negative.
We also test whether this failure is an artifact of assuming a linear
relationship. Repeating the same leave-one-topic-out protocol with
low-capacity cubic-spline regression yields negative 𝑅2in 55/60
tested cells, while monotone isotonic regression is negative in 56/60.
The few positive nonlinear values are negligible ( 𝑅2≤0.007) and
occur only under Claude evaluation with Coverage-Disciplined
Gemma3 generation. Thus, allowing smooth or monotone nonlinear
relationships does not reveal a robust transferable relationship
between retrieval effectiveness and answer quality.

Downstream Utility of Evidence-Aware Retrieval Conference’17, July 2017, Washington, DC, USA
0.2 0.3 0.4 0.5 0.6 0.7 0.8
Agreement with answer quality (τb)Std/Qwen
Std/Claude
Cov/Qwen
Cov/ClaudeΔ=+0.109
Δ=+0.182
Δ=+0.182
Δ=+0.182
(a)Ranking alignment (◦O,⋄G)
−0.05 0.00 0.05 0.10 0.15
Held-out answer-quality Δ\n(G-selected − O-selected)Std/Qwen
Std/Claude
Cov/Qwen
Cov/Claude
p=0.828
p=0.902
p=0.007
p=0.030(b)Held-out selection value
O G OG
Retrieval predictors−0.08−0.06−0.04−0.020.00LOTO CV R20/60 positive CV R2
(c)Out-of-topic prediction
Figure 3: Downstream value of answer-support-aware retrieval. (a) Answer-support-aware scores show higher observed
alignment with answer quality. (b) This does not consistently improve held-out system selection: gains appear under Coverage-
Disciplined but not Standard Grounded generation. (c) Neither original nor answer-support-aware scores provide transferable
out-of-topic prediction; all 60 OLS models have negative cross-validated 𝑅2, with nonlinear sensitivities yielding the same
overall conclusion.
Taken together with RQ2, these results separate three distinct
uses of the same retrieval signal: ranking alignment, held-out sys-
tem selection, and out-of-topic prediction (Figure 3). Answer-support-
aware evaluation can move the aggregate system ordering closer
to the downstream ordering without becoming a reliable selection
rule under every generation regime or a score that robustly predicts
answer quality on unseen topics.
Takeaway (RQ3).RQ1–RQ3 reveal a validity-composition problem.
Answer support can be more directly aligned with answer-bearing
evidence without every downstream use of that signal becoming
valid in turn. It changes retrieval conclusions; its value for retrieval
training is not established; its value for system selection depends on
the generation regime; and its scores provide no robust transferable
prediction of answer quality on unseen topics. Better alignment
at one stage therefore does not establish the validity of the down-
stream use. This motivates a more direct test. RQ4 leaves retrieval
scores behind and intervenes on the evidence supplied to the gener-
ator itself. We test whether answer-support judgments can identify
evidence worth preserving and, if so, whether changing that evi-
dence changes the quality of the generated answer.
7 Does Better Evidence Improve RAG?
RQ3 found no robust out-of-topic prediction from retrieval scores.
RQ4 therefore intervenes directly on the retrieved context, asking
whether a human-validated shift toward answer-bearing evidence
yields an evaluator-robust improvement in answer quality.
Controlled evidence intervention.RQ1–RQ3 asked what hap-
pens when answer support is used to evaluate, optimize, select, or
predict retrieval. RQ4 asks a more direct question:does the signal
actually identify evidence that is more useful to the generator?We
therefore hold the retriever fixed and use answer support only to
change which of its retrieved passages reach the generator.
For each held-out topic, we begin with the retriever that would
have been selected using the original RAG25 retrieval judgments.
Within each of five folds, we select the retriever with the highest
nDCG@30 on that fold’s training topics and apply it to the held-out
topics. This simulates the context that conventional benchmarkevaluation would have led us to use while keeping the held-out
topics out of the fold-specific retriever selection decision.
We then construct an answer-support-filtered version of the
retrieved context using the judgments defined in §3. We remove
passages that fall below the answer-support criterion and backfill,
in ranked order, with lower-ranked passages that satisfy it until
reaching 20 passages or the fixed 12,000-token budget. This directly
tests whether preferentially supplying answer-supporting passages
gives the generator better evidence.
Simply changing which passages reach the generator can itself
affect generation, so we compare this intervention against an equal-
magnitude matched-random control. For each passage displaced by
answer-support filtering, the control removes a passage matched
on original TREC relevance grade category, retrieval-rank band,
and passage-length quartile. When an exact match is unavailable,
matching follows a fixed relaxation sequence: drop the length quar-
tile, merge adjacent rank bands, retain only the relevance category,
and finally use the global candidate pool. The context is then back-
filled from the same ranked list in original order. We generate 20
matched-random controls per topic using seeds 20260731–20260750
and average their answer outcomes within topic before comparison.
Thus, both conditions perturb contexts from the same retriever
under the same context budget, while answer support determines
which passages the intervention preferentially preserves.
Human validation of the mechanism.Because the interven-
tion is defined by GPT-4.1 answer-support judgments, we vali-
date two distinct links before interpreting its downstream effect.
H1asks whether the answer-support distinction itself is human-
recognizable;H2asks whether using that distinction to filter con-
text actually moves the supplied evidence in the intended direction.
We construct two independently sampled 60-passage studies
from the retrieved contexts of the same 18 TREC RAG 2025 eval-
uation topics used in the preceding experiments, with 3–4 pas-
sages from every topic. H1 is deliberately a challenge sample rather
than a prevalence estimate: 30 passages come from cases where

Conference’17, July 2017, Washington, DC, USA Ghosh, Mukhopadhyay, and Chatterjee
GPT-4.1, Qwen, and Llama agree on the answer-support bound-
ary and 30 from cases where the judges disagree, with both GPT-
4.1-positive and GPT-4.1-negative decisions represented. H2 in-
stead samples passages according to the decisions made by answer-
support filtering and the prespecified matched-random control:
passages uniquely removed by either policy, together with passages
retained by both. The two studies overlap on 13 passages, yielding
107 unique passages for annotation.
Two independent annotators inspect the original passage text
and label each passage as containing no meaningful evidence (0),
related but insufficient information (1), or answer-supporting evi-
dence (2). They do not see the decomposed claims, study stratum,
automated judgments, or filtering decision. Across all passages,
exact agreement is 93.5%, with Cohen’s 𝜅=0.660and Gwet’s AC1
=0.928; at the binary 0/1-versus-2 evidence boundary, AC1 is0 .919.
ForH1, GPT-4.1 agrees with the two annotators on 73.3% and
68.3% of the 60 challenge cases. The disagreement is strongly asym-
metric. Both annotators classify all 16 passages labeled answer-
supporting by all three automated judges as evidence-bearing. By
contrast, among the 14 passages labeled non-supporting by all
three, 71.4% and 78.6% are still judged evidence-bearing by humans.
The automated boundary therefore captures a human-recognizable
evidence distinction, but GPT-4.1’s negative decisions are compara-
tively conservative and should not be treated as ground truth.
ForH2, we test the intervention itself using a prespecified matched-
random control (seed 20260731). Passages preserved by answer-
support filtering but removed by this control are judged evidence-
bearing in 14/15 cases by both annotators. Passages removed by
answer-support filtering but retained by the random control are
evidence-bearing in 11/15 and 12/15 cases, differences of 20.0 and
13.3 percentage points. Thus, although the filter does not perfectly
separate useful from useless passages, it preferentially preserves
evidence that humans recognize as answer-bearing.
Downstream effect.Having established that answer-support fil-
tering preferentially preserves human-recognized evidence, we next
ask whether that improvement carries through to the generated an-
swer. Under the primary Qwen3-30B evaluator, and using the same
primary downstream measure as in RQ2–RQ3, strict_vital_score
(the fraction of vital nuggets fully supported by the answer), answer-
support filtering increases the score by +0.0289relative to the
matched-random control (95% CI [+0.0051,+0.0518]), with im-
provements on 13 of 18 topics ( 𝑝= 0.0305, exact sign-flip test).
Neither strict_all_score (full support across all nuggets) nor
hard_recall (strict citation-support recall) shows the same reli-
able improvement. The positive effect is therefore specific to the
primary measure rather than a uniform gain across the evaluated
answer dimensions.
Evaluator sensitivity.The remaining question is whether this
downstream conclusion is robust to how answer quality itself is
measured. We therefore evaluate the exact same generated answers
with Claude Sonnet 5, designated as an independent evaluator
sensitivity before the treatment-effect analysis. Claude does not
reproduce the Qwen result: the mean effect is −0.0003(95% CI
[−0.0162,+0.0173]), with 8 wins, 1 tie, and 9 losses across the 18
topics (𝑝=0.9780). Thus, at the aggregate level, the same interven-
tion supports a positive treatment effect under Qwen and essentially
no effect under Claude (Fig. 4(b)).The disagreement is not simply a consequence of one 𝑝-value
crossing a significance threshold. Across the 18 topics, the treatment
effects estimated by the two evaluators correlate only 𝑟=0.258and
agree in sign on 8 topics (Fig. 4(a)). The evaluator dependence is
therefore visible not only in the aggregate estimate, but in how the
two evaluators assess the intervention topic by topic. Even after
human validation confirms that answer-support filtering moves
the context toward more answer-bearing evidence, whether that
change appears to improve the final answer depends substantially
on how answer quality is evaluated.
Takeaway (RQ4).RQ4 isolates the final break in the validity chain.
Human judgments confirm that the intervention moves the re-
trieved context toward answer-bearing evidence, so the down-
stream disagreement cannot be explained simply by the interven-
tion failing to manipulate what it intended to manipulate. Yet the
same contexts and the same generated answers support an im-
provement under one evaluator and essentially no effect under
another. The answer evaluator is therefore part of the validity of
the downstream conclusion, not merely a reporting device. Across
RQ1–RQ4, this exposes a validity-composition problem: evidence
that a measurement or intervention is meaningful at one stage does
not automatically validate the comparison, decision, prediction, or
conclusion drawn at the next.
8 Discussion
Our experiments show that “better RAG evaluation” is not a sin-
gle property. An evaluation signal may better reflect the evidence
needed for generation, change which systems appear better, sup-
port some downstream decisions, and fail at others. Answer support
exhibits exactly this pattern. It changes retrieval conclusions and
identifies evidence that humans recognize as answer-bearing, yet
it does not reliably improve retrieval training, its value for system
selection depends on the generation regime, and its retrieval scores
do not robustly predict answer quality on unseen topics. Even after
a direct intervention improves the supplied evidence, the conclu-
sion that the resulting answers improved depends on the answer
evaluator. These results separate several properties that are often
treated as if they followed from one another: construct alignment,
system-comparison validity, decision validity, predictive validity,
and robustness of outcome measurement.
Evaluation utility is relational.Two patterns help explain
why a more meaningful retrieval signal does not have uniform
downstream value. First, the amount of disagreement between rele-
vance and answer support alone is not sufficient to determine how
consequential the change will be. NFCorpus loses a large share of
its original positive judgments, yet that loss is distributed relatively
evenly across systems. In TREC-COVID, by contrast, competing
retrievers differ much more in their exposure to judgments that fail
the answer-support criterion, and changing the criterion has a cor-
respondingly larger effect on their relative ordering. This suggests
that the consequence of an evaluation change depends not only on
what the criterion measures, but also on how its disagreements are
distributed across the systems being compared.
Second, the value of the resulting comparison depends on what
happens after retrieval. The same answer-support-aware selection
procedure provides no reliable held-out benefit under Standard

Downstream Utility of Evidence-Aware Retrieval Conference’17, July 2017, Washington, DC, USA
−0.10 −0.05 0.00 0.05 0.10
Qwen topic effect−0.10−0.050.000.050.10Claude topic effectPearson r=0.258
same sign=8/18 (44.4%)
(a)Topic-level evaluator agreement
Qwen
(primary)Claude
(sensitivity)−0.02−0.010.000.010.020.030.040.050.06Mean topic effect on strict-vital score
p=0.030
+0.0289
p=0.978
-0.0003 (b)Aggregate treatment effect
Figure 4: Evaluator sensitivity of the RQ4 treatment effect. (a) Qwen and Claude show weak topic-level agreement ( 𝑟=0.258;
same sign on 8/18 topics). (b) The aggregate conclusion changes: Qwen finds a positive intervention effect, while Claude
estimates essentially none.
Grounded generation but improves answer quality under Coverage-
Disciplined generation. The retrieval criterion, systems, and selec-
tion procedure are unchanged; what changes is how the generator is
instructed to use the retrieved evidence. Evaluation utility is there-
fore not solely a property of a metric. It arises from the interaction
between the evaluation criterion, the systems being compared, the
downstream component consuming their outputs, and the decision
the evaluation is meant to support.
Validation should follow intended use.This distinction mat-
ters because retrieval evaluation increasingly functions as decision
infrastructure. Offline metrics are not used only to describe systems;
they guide optimization, model selection, and deployment. Valida-
tion should therefore test the use for which an evaluation signal is
intended. If a metric is used to select a retriever, its value should be
tested through held-out system selection rather than inferred from
stronger correlation with downstream quality. If retrieval effective-
ness is used as a proxy for end-to-end performance, the relationship
should transfer to unseen topics rather than hold only within the
observed benchmark. And when an automated answer evaluator
is used to determine whether an intervention improved a system,
robustness should be assessed at the level of that scientific conclu-
sion, not only through item-level agreement between evaluators.
The broader methodological implication is that meta-evaluation
should mirror the downstream inference or decision that the metric
will actually support.
Beyond passage-level answer support.Our experiments also
expose a limitation of passage-level evidence judgments. A genera-
tor consumes an evidence set, not isolated passages. Individually
answer-supporting passages may be redundant, collectively omit
important aspects of the information need, or consume limited
context budget with overlapping evidence. Conversely, a passage
with modest value in isolation may be useful because it comple-
ments what has already been retrieved. Evidence-set utility may
therefore depend on coverage, redundancy, complementarity, con-
tradiction, and context budget in addition to the answer-support
value of individual passages. The stronger system-selection resultunder Coverage-Disciplined generation is consistent with the idea
that the downstream value of retrieved evidence depends on how
effectively the generator exploits such structure, although our ex-
periments do not establish that mechanism.
Taken together, these results suggest a broader direction for RAG
evaluation. The field is right to move beyond topical relevance to-
ward evidence that is more closely connected to downstream needs,
but richer constructs alone do not resolve the evaluation problem.
The next question is not only whether a metric captures something
meaningful, but whether it supports the particular comparison,
decision, prediction, or experimental conclusion for which it is
used. In multi-stage RAG systems, those transitions are themselves
objects of validation.
9 Conclusion
We asked how far the benefits of answer-support-aware retrieval
evaluation carry through the RAG pipeline. Across retrieval compar-
ison, training, system selection, prediction, and a direct evidence
intervention, the answer is: not automatically. Answer support
changes retrieval conclusions and identifies evidence that humans
recognize as more answer-bearing, yet its downstream value de-
pends on the decision being made, how the generator uses the
retrieved evidence, and how the resulting answers are evaluated.
We call this thevalidity-composition problem: evidence that an
evaluation is meaningful for one use does not automatically validate
the next inference or decision built from it. As RAG evaluation
increasingly guides system development and deployment, validity
must therefore be established for the particular use an evaluation
signal is intended to support rather than assumed to propagate
through the pipeline.
Ethical Considerations
This work uses public information-retrieval benchmarks and does
not involve private user data or deployment on real users. Human
annotation is limited to judging the evidential value of retrieved

Conference’17, July 2017, Washington, DC, USA Ghosh, Mukhopadhyay, and Chatterjee
passages. Because several analyses rely on LLM-based judgments,
these judgments may inherit model-specific biases or systematic
errors; we therefore evaluate cross-judge robustness and include
independent human validation for the central evidence interven-
tion. More broadly, our results caution against treating automated
RAG metrics as decision-neutral: when such metrics guide system
selection, optimization, or deployment, their errors can propagate
into consequential engineering decisions. We therefore view trans-
parency about evaluator choice, validation against the intended use,
and reproducible reporting of evaluation procedures as important
safeguards for responsible RAG evaluation.
References
[1]Vera Boteva, Demian Gholipour, Artem Sokolov, and Stefan Riezler. 2016. A Full-
Text Learning to Rank Dataset for Medical Information Retrieval.Proceedings
of the 38th European Conference on Information Retrieval. http://www.cl.uni-
heidelberg.de/~riezler/publications/papers/ECIR2016.pdf
[2]Chris Buckley and Ellen M. Voorhees. 2000. Evaluating Evaluation Measure
Stability. InProceedings of the 23rd Annual International ACM SIGIR Conference
on Research and Development in Information Retrieval(Athens, Greece)(SIGIR
’00). Association for Computing Machinery, New York, NY, USA, 33–40. doi:10.
1145/345508.345543
[3]Chris Buckley and Ellen M. Voorhees. 2004. Retrieval Evaluation with Incomplete
Information. InProceedings of the 27th Annual International ACM SIGIR Conference
on Research and Development in Information Retrieval(Sheffield, United Kingdom)
(SIGIR ’04). Association for Computing Machinery, New York, NY, USA, 25–32.
doi:10.1145/1008992.1009000
[4]Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2024. Benchmarking Large
Language Models in Retrieval-Augmented Generation. InProceedings of the
Thirty-Eighth AAAI Conference on Artificial Intelligence and Thirty-Sixth Confer-
ence on Innovative Applications of Artificial Intelligence and Fourteenth Sympo-
sium on Educational Advances in Artificial Intelligence (AAAI’24/IAAI’24/EAAI’24).
AAAI Press, Article 1980, 9 pages. doi:10.1609/aaai.v38i16.29728
[5]Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu.
2025. M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity
Text Embeddings Through Self-Knowledge Distillation. arXiv:2402.03216 [cs.CL]
https://arxiv.org/abs/2402.03216
[6]Charles L.A. Clarke, Maheedhar Kolla, Gordon V. Cormack, Olga Vechtomova,
Azin Ashkan, Stefan Büttcher, and Ian MacKinnon. 2008. Novelty and Diversity in
Information Retrieval evaluation. InProceedings of the 31st Annual International
ACM SIGIR Conference on Research and Development in Information Retrieval
(Singapore, Singapore)(SIGIR ’08). Association for Computing Machinery, New
York, NY, USA, 659–666. doi:10.1145/1390334.1390446
[7]Charles L. A. Clarke and Laura Dietz. 2025. LLM-based Relevance Assessment
Still Can’t Replace Human Relevance Assessment. InProceedings of the Tenth
International Workshop on Evaluating Information Access (EVIA 2025). Tokyo,
Japan, 1–5. doi:10.20736/0002002105
[8]Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and Daniel Campos. 2021. Overview
of the TREC 2020 deep learning track. arXiv:2102.07662 [cs.IR] https://arxiv.org/
abs/2102.07662
[9]Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Ellen M.
Voorhees. 2020. Overview of the TREC 2019 deep learning track.
arXiv:2003.07820 [cs.IR] https://arxiv.org/abs/2003.07820
[10] Laura Dietz. 2024. A Workbench for Autograding Retrieve/Generate Sys-
tems. InProceedings of the 47th International ACM SIGIR Conference on Re-
search and Development in Information Retrieval(Washington DC, USA)(SIGIR
’24). Association for Computing Machinery, New York, NY, USA, 1963–1972.
doi:10.1145/3626772.3657871
[11] Laura Dietz, Oleg Zendel, Peter Bailey, Charles L. A. Clarke, Ellese Cotterill, Jeff
Dalton, Faegheh Hasibi, Mark Sanderson, and Nick Craswell. 2025. Principles and
Guidelines for the Use of LLM Judges. InProceedings of the 2025 International ACM
SIGIR Conference on Innovative Concepts and Theories in Information Retrieval
(ICTIR)(Padua, Italy)(ICTIR ’25). Association for Computing Machinery, New
York, NY, USA, 218–229. doi:10.1145/3731120.3744588
[12] Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024. RAGAs:
Automated Evaluation of Retrieval Augmented Generation. InProceedings of the
18th Conference of the European Chapter of the Association for Computational
Linguistics: System Demonstrations, Nikolaos Aletras and Orphee De Clercq (Eds.).
Association for Computational Linguistics, St. Julians, Malta, 150–158. doi:10.
18653/v1/2024.eacl-demo.16
[13] Guglielmo Faggioli, Laura Dietz, Charles L. A. Clarke, Gianluca Demartini,
Matthias Hagen, Claudia Hauff, Noriko Kando, Evangelos Kanoulas, Martin
Potthast, Benno Stein, and Henning Wachsmuth. 2023. Perspectives on LargeLanguage Models for Relevance Judgment. InProceedings of the 2023 ACM SI-
GIR International Conference on Theory of Information Retrieval(Taipei, Taiwan)
(ICTIR ’23). Association for Computing Machinery, New York, NY, USA, 39–50.
doi:10.1145/3578337.3605136
[14] Guglielmo Faggioli, Laura Dietz, Charles L. A. Clarke, Gianluca Demartini,
Matthias Hagen, Claudia Hauff, Noriko Kando, Evangelos Kanoulas, Martin
Potthast, Benno Stein, and Henning Wachsmuth. 2024. Who Determines What
Is Relevant? Humans or AI? Why Not Both?Commun. ACM67, 4 (March 2024),
31–34. doi:10.1145/3624730
[15] Naghmeh Farzi and Laura Dietz. 2024. Pencils Down! Automatic Rubric-based
Evaluation of Retrieve/Generate Systems. InProceedings of the 2024 ACM SIGIR
International Conference on Theory of Information Retrieval(Washington DC,
USA)(ICTIR ’24). Association for Computing Machinery, New York, NY, USA,
175–184. doi:10.1145/3664190.3672511
[16] Naghmeh Farzi, Tim Hagen, Eugene Yang, Maik Fröbe, Ronak Pradeep, Hos-
sein A. Rahmani, Xi Wang, Oleg Zendel, Martin Potthast, and Laura Dietz. 2026.
Auto-Judge: A Cross-Task Benchmark for Comparing LLM Judges for Citation-
Grounded RAG Systems. InProceedings of the 49th International ACM SIGIR
Conference on Research and Development in Information Retrieval(Australia)(SI-
GIR ’26). Association for Computing Machinery, New York, NY, USA, 3159–3166.
doi:10.1145/3805712.3808601
[17] Ariel Gera, Odellia Boni, Yotam Perlitz, Roy Bar-Haim, Lilach Eden, and Asaf
Yehudai. 2025. JuStRank: Benchmarking LLM Judges for System Ranking. In
Proceedings of the 63rd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), Wanxiang Che, Joyce Nabende, Ekaterina
Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational
Linguistics, Vienna, Austria, 682–712. doi:10.18653/v1/2025.acl-long.34
[18] Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, and Allan
Hanbury. 2021. Efficiently Teaching an Effective Dense Retriever with Balanced
Topic Aware Sampling. InProceedings of the 44th International ACM SIGIR Confer-
ence on Research and Development in Information Retrieval(Virtual Event, Canada)
(SIGIR ’21). Association for Computing Machinery, New York, NY, USA, 113–122.
doi:10.1145/3404835.3462891
[19] Jennifer Hsia, Afreen Shaikh, Zora Zhiruo Wang, and Graham Neubig. 2025.
RAGGED: Towards Informed Design of Scalable and Stable RAG Systems. In
Forty-second International Conference on Machine Learning. https://openreview.
net/forum?id=4ufjBV6S4I
[20] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bo-
janowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised Dense
Information Retrieval with Contrastive Learning. arXiv:2112.09118 [cs.IR]
https://arxiv.org/abs/2112.09118
[21] Michael T. Kane. 2013. Validating the Interpretations and Uses of Test Scores.
Journal of Educational Measurement50, 1 (2013), 1–73. doi:10.1111/jedm.12000
[22] Carlos Lassance, Hervé Déjean, Thibault Formal, and Stéphane Clinchant. 2024.
SPLADE-v3: New Baselines for SPLADE. arXiv:2403.06789 [cs.IR] https://arxiv.
org/abs/2403.06789
[23] Dawn Lawrie, Sean MacAvaney, James Mayfield, Luca Soldaini, Eugene Yang,
and Andrew Yates. 2026. Overview of the TREC 2025 RAGTIME Track.
arXiv:2602.10024 [cs.IR] https://arxiv.org/abs/2602.10024
[24] Youngwon Lee, Seung-won Hwang, Daniel F Campos, Filip Graliński, Zhewei
Yao, and Yuxiong He. 2025. Inference Scaling for Bridging Retrieval and Aug-
mented Generation. InFindings of the Association for Computational Linguis-
tics: NAACL 2025, Luis Chiruzzo, Alan Ritter, and Lu Wang (Eds.). Associ-
ation for Computational Linguistics, Albuquerque, New Mexico, 7339–7354.
doi:10.18653/v1/2025.findings-naacl.409
[25] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan
Zhang. 2023. Towards General Text Embeddings with Multi-stage Contrastive
Learning. arXiv:2308.03281 [cs.CL] https://arxiv.org/abs/2308.03281
[26] Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy Lin. 2020. Distilling Dense Repre-
sentations for Ranking using Tightly-Coupled Teachers. arXiv:2010.11386 [cs.IR]
https://arxiv.org/abs/2010.11386
[27] Samuel Messick. 1995. Validity of Psychological Assessment: Validation of In-
ferences from Persons’ Responses and Performances as Scientific Inquiry into
Score Meaning.American Psychologist50, 9 (1995), 741–749. doi:10.1037/0003-
066X.50.9.741
[28] Changle Qu, Sunhao Dai, Hengyi Cai, Yiyang Cheng, Jun Xu, Shuaiqiang Wang,
and Dawei Yin. 2025. Uplift-RAG: Uplift-Driven Knowledge Preference Align-
ment for Retrieval-Augmented Generation. InFindings of the Association for
Computational Linguistics: EMNLP 2025, Christos Christodoulopoulos, Tanmoy
Chakraborty, Carolyn Rose, and Violet Peng (Eds.). Association for Computational
Linguistics, Suzhou, China, 9632–9644. doi:10.18653/v1/2025.findings-emnlp.511
[29] Jon Saad-Falcon, Omar Khattab, Christopher Potts, and Matei Zaharia. 2024.
ARES: An Automated Evaluation Framework for Retrieval-Augmented Gen-
eration Systems. InProceedings of the 2024 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), Kevin Duh, Helena Gomez, and Steven Bethard
(Eds.). Association for Computational Linguistics, Mexico City, Mexico, 338–354.
doi:10.18653/v1/2024.naacl-long.20

Downstream Utility of Evidence-Aware Retrieval Conference’17, July 2017, Washington, DC, USA
[30] Alireza Salemi and Hamed Zamani. 2024. Evaluating Retrieval Quality in
Retrieval-Augmented Generation. InProceedings of the 47th International ACM
SIGIR Conference on Research and Development in Information Retrieval(Wash-
ington DC, USA)(SIGIR ’24). Association for Computing Machinery, New York,
NY, USA, 2395–2400. doi:10.1145/3626772.3657957
[31] David P. Sander and Laura Dietz. 2021. EXAM: How to Evaluate Retrieve-
and-Generate Systems for Users Who Do Not (Yet) Know What They Want. In
Biennial Conference on Design of Experimental Search & Information Retrieval
Systems. https://api.semanticscholar.org/CorpusID:238207962
[32] Mark Sanderson and Justin Zobel. 2005. Information Retrieval System Evaluation:
Effort, Sensitivity, and Reliability. InProceedings of the 28th Annual International
ACM SIGIR Conference on Research and Development in Information Retrieval
(Salvador, Brazil)(SIGIR ’05). Association for Computing Machinery, New York,
NY, USA, 162–169. doi:10.1145/1076034.1076064
[33] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna
Gurevych. 2021. BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of
Information Retrieval Models. arXiv:2104.08663 [cs.IR] https://arxiv.org/abs/
2104.08663
[34] Fangzheng Tian, Debasis Ganguly, and Craig Macdonald. 2026. Predicting Re-
trieval Utility and Answer Quality in Retrieval-Augmented Generation. InAd-
vances in Information Retrieval: 48th European Conference on Information Re-
trieval, ECIR 2026, Delft, The Netherlands, March 29 – April 2, 2026, Proceedings,
Part I(Delft, The Netherlands). Springer-Verlag, Berlin, Heidelberg, 368–385.
doi:10.1007/978-3-032-21289-4_24
[35] Shivani Upadhyay, Nandan Thakur, Ronak Pradeep, Nick Craswell, Daniel Cam-
pos, and Jimmy Lin. 2026. Overview of the TREC 2025 Retrieval Augmented Gen-
eration (RAG) Track. arXiv:2603.09891 [cs.IR] https://arxiv.org/abs/2603.09891
[36] Juraj Vladika and Florian Matthes. 2025. On the Influence of Context Size and
Model Choice in Retrieval-Augmented Generation Systems. InFindings of the
Association for Computational Linguistics: NAACL 2025, Luis Chiruzzo, Alan Ritter,
and Lu Wang (Eds.). Association for Computational Linguistics, Albuquerque,
New Mexico, 6739–6751. doi:10.18653/v1/2025.findings-naacl.375
[37] Ellen Voorhees, Tasmeer Alam, Steven Bedrick, Dina Demner-Fushman,
William R Hersh, Kyle Lo, Kirk Roberts, Ian Soboroff, and Lucy Lu Wang. 2020.
TREC-COVID: Constructing a Pandemic Information Retrieval Test Collection.
arXiv:2005.04474 [cs.IR] https://arxiv.org/abs/2005.04474
[38] Ellen M. Voorhees. 1998. Variations in Relevance Judgments and the Measure-
ment of Retrieval Effectiveness. InProceedings of the 21st Annual InternationalACM SIGIR Conference on Research and Development in Information Retrieval
(Melbourne, Australia)(SIGIR ’98). Association for Computing Machinery, New
York, NY, USA, 315–323. doi:10.1145/290941.291017
[39] David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen,
Arman Cohan, and Hannaneh Hajishirzi. 2020. Fact or Fiction: Verifying Scientific
Claims. InProceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), Bonnie Webber, Trevor Cohn, Yulan He, and
Yang Liu (Eds.). Association for Computational Linguistics, Online, 7534–7550.
doi:10.18653/v1/2020.emnlp-main.609
[40] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang,
Rangan Majumder, and Furu Wei. 2024. Text Embeddings by Weakly-Supervised
Contrastive Pre-training. arXiv:2212.03533 [cs.CL] https://arxiv.org/abs/2212.
03533
[41] Kaige Xie, Philippe Laban, Prafulla Kumar Choubey, Caiming Xiong, and Chien-
Sheng Wu. 2025. Do RAG Systems Cover What Matters? Evaluating and Optimiz-
ing Responses with Sub-Question Coverage. InProceedings of the 2025 Conference
of the Nations of the Americas Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume 1: Long Papers), Luis Chiruzzo,
Alan Ritter, and Lu Wang (Eds.). Association for Computational Linguistics,
Albuquerque, New Mexico, 5836–5849. doi:10.18653/v1/2025.naacl-long.301
[42] Tianruo Rose Xu, Vedant Gaur, Liu Leqi, and Tanya Goyal. 2025. The Progress Il-
lusion: Revisiting meta-evaluation standards of LLM evaluators. InFindings of the
Association for Computational Linguistics: EMNLP 2025, Christos Christodoulopou-
los, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (Eds.). Association for
Computational Linguistics, Suzhou, China, 19033–19043. doi:10.18653/v1/2025.
findings-emnlp.1036
[43] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu,
Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-
bench and Chatbot Arena. InProceedings of the 37th International Conference on
Neural Information Processing Systems(New Orleans, LA, USA)(NIPS ’23). Curran
Associates Inc., Red Hook, NY, USA, Article 2020, 29 pages.
[44] Justin Zobel. 1998. How Reliable are the Results of Large-scale Information
Retrieval Experiments?. InProceedings of the 21st Annual International ACM SIGIR
Conference on Research and Development in Information Retrieval(Melbourne,
Australia)(SIGIR ’98). Association for Computing Machinery, New York, NY, USA,
307–314. doi:10.1145/290941.291014
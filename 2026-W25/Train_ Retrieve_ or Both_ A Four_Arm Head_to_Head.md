# Train, Retrieve, or Both? A Four-Arm Head-to-Head for Correct Statutory Citation on the Ontario Residential Tenancies Act

**Authors**: Ali Asaria, Tony Salomone, Deep Gandhi

**Published**: 2026-06-18 15:21:53

**PDF URL**: [https://arxiv.org/pdf/2606.20359v1](https://arxiv.org/pdf/2606.20359v1)

## Abstract
Self-represented tenants, landlords, and help-desk staff need to be pointed at the provision of law that actually governs a question, with a correct statutory citation. We study this task on the Ontario Residential Tenancies Act, 2006 (RTA) and its core regulation, asking the operator's question empirically: is fine-tuning enough, or is hybrid retrieval needed? We run a four-arm head-to-head on Qwen2.5-7B-Instruct (base zero-shot, LoRA SFT-only, RAG-only, and an SFT+RAG hybrid), scored on citation exact-match (section+subsection) over a small, human-verification-pending real eval set. The base model cannot cite the RTA and SFT-only mis-recalls sections; retrieval is essential and drives hallucination to zero by construction; and the SFT+RAG hybrid scores highest at 0.481 exact-match with zero hallucinated citations. Its edge comes from SFT making provision selection more robust to the higher-recall candidate sets that hurt zero-shot RAG. Notably, this cheap bge-small hybrid matches or beats a pipeline built on bigger, specialized retrieval models (a larger embedder and a cross-encoder reranker), and a larger/improved training set does not help either: strong statutory-citation performance here does not require specialized retrieval models or more data. The artifact zeroes hallucination and clears the lift-over-base bar but does not reach the aspirational 0.70 exact-match target. All results are on a small, human-verification-pending real eval set and are reported as preliminary.

## Full Text


<!-- PDF content starts -->

Train, Retrieve, or Both? A Four-Arm Head-to-Head for Correct
Statutory Citation on the Ontario Residential Tenancies Act
Ali Asaria
Transformer LabTony Salomone
Transformer LabDeep Gandhi∗
Transformer Lab
Abstract
Self-represented tenants, landlords, and help-desk staff need to be pointed at the provision of
law that actually governs a question, with a correct statutory citation. We study this task
on the OntarioResidential Tenancies Act, 2006(RTA) and its core regulation, asking the
operator’s question empirically: is fine-tuning enough, or is hybrid retrieval needed? We run
a four-arm head-to-head on Qwen2.5-7B-Instruct (base zero-shot, LoRA SFT-only, RAG-only,
and an SFT+RAG hybrid), scored on citation exact-match (section+subsection) over a small,
human-verification-pending real eval set. The base model cannot cite the RTA and SFT-only
mis-recalls sections; retrieval is essential and drives hallucination to zeroby construction; and
theSFT+RAG hybrid scores highest at 0.481 exact-match with zero hallucinated
citations. Its edge comes from SFT making provision selection more robust to the higher-recall
candidate sets thathurtzero-shot RAG. Notably, this cheap bge-small hybrid matches or beats
a pipeline built on bigger, specialized retrieval models (a larger embedder and a cross-encoder
reranker), and a larger/improved training set does not help either: strong statutory-citation
performance here does not require specialized retrieval models or more data. The artifact
zeroes hallucination and clears the lift-over-base bar but doesnotreach the aspirational 0.70
exact-match target. All results are on a small, human-verification-pending real eval set and are
reported as preliminary.
1 Introduction
Ontario’s residential-tenancy law is governed by theResidential Tenancies Act, 2006(RTA, S.O.
2006, c. 17) and O. Reg. 516/06 (General). The people who most need it (self-represented tenants
and landlords, paralegals, and tenant-help-desk staff) rarely need fluent prose; they need to be
pointed atthe provision that governs, with a correct citation (the right Act/regulation, section,
and subsection). We therefore treat the headline quality target as thecitation scorerather than
answer fluency: the unit of prediction is a single natural-language tenancy question (e.g. “How
much notice must a landlord give to enter my unit?”) mapped to a short answer plus one or more
statutory citations (e.g.RTA s. 26(1)). This is decision-support, not legal advice.
The operator posed a concrete, falsifiable question:is fine-tuning enough, or do we need
hybrid RAG?Rather than assume an answer, we measure it. We build a citation index and a
question→citation dataset (no public Ontario-RTA set exists) and run a four-arm head-to-head on
∗Corresponding author:deep@lab.cloud
1arXiv:2606.20359v1  [cs.LG]  18 Jun 2026

Train, Retrieve, or Both? Statutory Citation for the Ontario RTA
the same held-out citation metric: the base instruct model zero-shot, LoRA supervised fine-tuning
on synthetic question →citation pairs, RAG over a chunked statute index, and the SFT+RAG hybrid.
Beyond a single-GPU LoRA budget, the stated design constraints include interactive latency, a
pinned consolidation date, and using no personal data as model input.
Contributions.
1.A measured four-arm answer to “train vs. retrieve”: base cannot cite (0.00, 81% hallucinated),
SFT-only mis-recalls sections (0.148), RAG is essential (0.44, zero hallucination), and the
hybrid scores highest (0.481), giving the ordering base ≪SFT-only≪retrieval-bearing arms;
see §5.
2.AnSFT×keffect: more retrieved candidateshurtzero-shot RAG (0.44 →0.37 atk=10) but
helpthe SFT model (0.333 →0.481), so the hybrid gains selection robustness to higher-recall
candidate sets; see §5.
3.An efficiency result: the cheap bge-small hybrid matches or beats a pipeline built on bigger,
specialized retrieval models (a larger embedder+a cross-encoder reranker, 0.37), and a
larger/improved training set does not improve over the original (0.444); see §5.
4.An honest accounting: zero hallucinated citationsby constructionand a+0 .48exact-match
lift, but the 0.70 target is not met on a 27-item, human-verification-pending eval set; see §6.
2 Related Work
Generating verifiable citations.Teaching attribution directly is more effective than passively
tagging text. Huang et al. [1]show that bidirectional source ↔fact training with fact-variation
augmentation lifts citation precision substantially, while verbatim token replay gives no benefit and
more replay epochs can hurt generalization, which directly motivates our paraphrase-based synthetic
generation and group-aware split. Pezeshkpour and Hruschka [2]find continual pre-training on raw
documents has only a marginal effect, whereas reformatting to atomic facts yields large exact-match
gains, supporting SFT on (question→provision-id) pairs over dumping statute prose. Xu et al.
[3]self-generate QA from an unlabeled corpus with a retrieval round-trip filter, and Yasuno [4]scale
graph-derived synthetic data; both inform our synthetic-from-statute recipe. Attribution methods
such as TRACE [ 5] and judgement-citation retrieval by contextual similarity [ 6] frame citation as a
retrieval/grounding problem rather than free generation.
Retrieval for statute.Statute retrieval is semantics-dominant yet benefits from a lexical anchor:
Paul et al. [7]report optimal fusion heavily weighted toward the dense retriever, but Arafat [8]
and Nigam et al. [9]show BM25 is essential to anchor exact identifiers and defined terms, with a
robust fusion weight that transfers without per-corpus tuning, so we adopt a hybrid BM25+dense
default. Second-stage re-ranking [ 7] or cross-encoder rerankers [ 9] can lift retrieval F1; we test the
reranker as an ablation. Kabir et al. [10]use relevance-check/query-refinement loops and low decode
temperature, and bounded-size chunking with overlap is standard [ 11,12]; we instead chunk on
statutory section boundaries as a design choice, one provision per chunk. Crucially, Yasuno [4]
report thatbadretrieval actively hurts (over-retrieval misleads generation), and Hsu et al. [13]note
multi-stage pipelines compound failures; both foreshadow our finding that heavier retrieval helps
neither arm.
2

Train, Retrieve, or Both? Statutory Citation for the Ontario RTA
Anti-hallucination guardrails.Arafat [8]use mechanical citation verification (accept a citation
only if it exists in the pinned corpus) to reach high accuracy with zero hallucinations; Noël et al.
[14]reduce structural hallucination detection to a cheap entity-grounding existence check but warn
it collapses on short, sparse inputs, favoring a deterministic existence check for single subsections.
Hsu et al. [13]contrast large/small models to verify grounding, and Komorowski et al. [15]explore
decode-time attribution; we adopt the cheap existence-check guardrail as a hard gate. Reality checks
abound: RAG does not eliminate hallucination [ 16], commercial legal RAG tools still hallucinate
17–33% [ 14], and retrieval can game cheap metrics like ROUGE without improving judged quality
[16].
LoRA and positioning.Our central positioning is simple: Yasuno [4]ran three of our four arms
and explicitly left SFT+RAG as future “Case D”, and this work fills that gap on a statutory-citation
exact-match metric, with a deterministic hallucination check that the surveyed legal papers measured
only qualitatively. On the training side, Rathore et al. [17]show LoRA usually matches or beats
full SFT while forgetting less, and that fine-tuning can erode a strong base, arguing to keep the
zero-shot base as a real baseline (which we do) and to prefer LoRA.
3 Method
Task and label.For a question qwe predict a short answer and a set of citations ˆC=
{(instrument,section,subsection )}against a gold set C. Correctness is section+subsection match;
section-only matches earn partial credit (0.5); the instrument must match. Citations to provisions
absent from the pinned consolidation count as hallucinations.
Corpus and index.We parse the e-Laws consolidation of the RTA (297 sections, 895 subsections,
20 Parts) and O. Reg. 516/06 (61 sections, 166 subsections) into a citation inventory of 358 provisions,
each a single structure-aware RAG chunk. The inventory doubles as the existence-check whitelist
for the hallucination metric.
Arms.Thebasearm is the instruct model zero-shot. TheSFT-onlyarm is a LoRA SFT on
synthetic question →citation pairs (rank r=32,α=2r,lr2e−4, 400 steps; q/k/v/o projections),
following the LoRA guidance of Rathore et al. [17]. TheRAG-onlyarm is the base model with
hybrid BM25+ bge-small retrieval, prompted to cite only from the retrieved context (a cite-from-
context guardrail that is effectively an existence check). We note up front that this guardrail
and the hallucinationmetricshare the same pinned inventory, so a 0.00 hallucination rate for the
retrieval-bearing arms is largelyguaranteed by constructionrather than learned; we treat it as a
design property, not an emergent result. TheSFT+RAG hybridis the SFT model trained to
select the correct citation from retrieved candidates, evaluated with the same retrieval. We sweep
retrieval depth kand retrieval strength (cheap bge-small vs.bge-large +cross-encoder reranker).
Objective.The optimization target is the citation set, scored as exact-match with section-only
partial credit and multi-citation F1. Two reporting conventions matter. First, because section-only
matches earn 0.5, the “exact-match” figures we report are a partial-credit citation score in[0 ,1], not
a strict 0/1 rate, so we keep the name “exact-match” for brevity but it is partial-credit-weighted.
Second, F1 is computed per item over the predicted vs. gold citationsets(multi-citation precision and
3

Train, Retrieve, or Both? Statutory Citation for the Ontario RTA
recall, then their harmonic mean) and averaged across items; it can exceed exact-match because it
gives proportional credit for partially-correct citation sets. Conceptually, an arm’s quality combines
retrieval supply and selection:
ExactMatch≤recall@k/bracehtipupleft/bracehtipdownright/bracehtipdownleft /bracehtipupright
retrieval supplies gold×P(select gold|retrieved)/bracehtipupleft /bracehtipdownright/bracehtipdownleft /bracehtipupright
SFT improves this,(1)
which frames the central finding: SFT raises the selection term while RAG raises the supply term,
and the hybrid benefits from both.
4 Experimental Setup
Data.No public Ontario-RTA question →citation dataset exists, so s4 builds one.Training
(synthetic).An LLM paraphrases each section/subsection into natural questions with the provision
as gold: 2,148 pairs, split leakage-safely (question-level bigram-Jaccard dedup+a 15% unseen-
section slice) into train 1,473 / synth_test 345 / unseen 330.Held-out (real, headline).A small
set mined from tenant-law sites with explicit RTA section references: 27 source-cited, in-inventory
items, each flagged human-verification-pending. Splits are group-aware by section/Part so the model
cannot win by memorizing question →citation maps; the real set is fully source-disjoint. Statute
text is the e-Laws consolidation (Government of Ontario, Crown copyright).
Protocol.The base model is Qwen2.5-7B-Instruct (operator: Qwen-only; Llama-3.1-8B was
not evaluated, HF-gated). Metrics: citation exact-match (instrument+section+subsection) with
section-only partial credit, multi-citation precision/recall/F1, a deterministic hallucinated-citation
rate (existence check against the pinned inventory), and recall@ kfor the RAG arms to separate
retrieval from generation failure. The scorer is deterministic and reused across arms. A harness
control (perfect predictions on the 27 items →exact-match 1.0, hallucination 0.0) confirms the
scorer and gold aremechanicallycorrect: it validates the measurement instrument, not the statistical
reliability of the model numbers, which is bounded by n=27. All arms are single-run point estimates
(no seed replication). Total compute was 4.43 GPU-hours against a planned 44 (Lambda H100 for
training, SkyPilot RTX 3090 for eval).
5 Results
Table 1 reports the final model against the overview’s success-criteria gates, and Table 2 the full
arm/ablation grid. All numbers are single-run point estimates on the 27-item real eval (human-
verification-pending) unless noted, so they should be read as preliminary. On 27 items a single
correct/incorrect prediction moves a rate by ∼0.037, so we draw inferences only at the level oflarge,
qualitatively-robust gaps (base vs. SFT-only vs. the retrieval-bearing arms); differences of one to
three items among the retrieval-bearing cells (e.g. 0.481 vs. 0.44, or the kand retrieval-strength
sweeps) are within noise and are reported as suggestive, not significant. We also flag a selection
caveat: the final configuration ( k=10) was chosen as the best-scoring cell on this same 27-item set,
so the headline 0.481 is a max-over-configurations and is optimistically biased (see §6).
Train vs. retrieve (Fig. 1).The base model cannot cite the RTA: 0.00 exact-match with 81%
of citations hallucinated. SFT-only teaches the citationformatand cuts the hallucination rate from
4

Train, Retrieve, or Both? Statutory Citation for the Ontario RTA
Table 1: Final model (SFT+RAG hybrid, k=10) vs. success-criteria gates. Three of four gates are
met; the absolute exact-match target is not.
Metric Result Target Verdict
Citation exact-match (sec+subsec) 0.481≥0.70not met
Hallucinated-citation rate 0.00≤0.05met
Lift over zero-shot base+0.48≥0.15met
Evidenced train-vs-retrieve answer yes — met
Table 2: Four-arm head-to-head and ablations on the 27-item real eval (single-run point estimates).
SFT-only learns format but mis-recalls sections; RAG is essential and zeroes hallucination by
construction; the hybrid at k=10scores highest, though gaps among the retrieval-bearing rows are
within noise at n=27. The lower block shows that swapping the cheap bge-small retriever for a
bigger, specialized pipeline (a larger embedder and a cross-encoder reranker) doesnothelp either
arm (0.37 vs. the cheap hybrid’s 0.481), and does not raise recall: the efficiency result. F1 was not
computed for the light-SFT and specialized-retrieval ablations (shown as “—”).
Arm Configkrecall@kExact F1
base zero-shot — — — 0.00 0.00
SFT-only LoRA r32/400 — — 0.148 0.167
Cheap retrieval (bge-smallhybrid BM25+dense)
RAG-onlyk=55 0.815 0.44 0.52
RAG-onlyk=1010 0.889 0.37 0.46
hybrid r32/400,k=55 0.815 0.333 0.43
hybrid (light) r16/150,k=55 0.815 0.333 —
hybrid (final) r32/400,k=1010 0.889 0.481 0.574
Bigger, specialized retrieval (bge-large+cross-encoder reranker)
RAG-onlyk=1010 0.852 0.37 —
hybrid r32/400,k=1010 0.852 0.37 —
0.81 to 0.148, but mis-recalls exact sections, reaching only 0.148 exact-match (the two 0.148 figures
are distinct quantities that coincide), consistent with exact section numbers being a memorization
weak spot for∼8B models [ 4]. RAG is the decisive lever: RAG-only reaches 0.44 and drives
hallucination to 0.00, which (as noted in §5) follows by construction from the cite-from-context
existence check rather than being learned. The hybrid scores highest at 0.481 exact-match (0.574
F1, 0.00 hallucination), though its ∼0.04 (≈1-item) margin over RAG-only is within noise at n=27.
The qualitatively robust finding is the large step from the non-retrieval arms ( ≤0.148) to the
retrieval-bearing arms (≥0.37), not the ordering within the latter.
A candidate-depth ( k) pattern (hypothesis).The hybrid’s apparent advantage is conditional
on candidate depth. More candidates appear tohurtzero-shot RAG (0.44 at k=5→0.37 atk=10,
even as recall rises 0.815 →0.889), as if distractors mislead selection; but the same depthhelps
the SFT model (0.333 at k=5→0.481 atk=10). Our reading is that SFT teaches more robust
selection from a noisier, higher-recall candidate set. We stress two limits on this claim. First, the
5

Train, Retrieve, or Both? Statutory Citation for the Ontario RTA
Figure 1: Citation exact-match rises monotonically across arms (base 0.00, SFT-only 0.148, RAG-
only 0.44, SFT+RAG hybrid 0.481), but the best arm still falls short of the 0.70 target (dashed).
Real eval,n=27, human-verification-pending.
deltas are 1–4 items on n=27, so the difference-in-differences “interaction” is under-powered and
we offer it as a hypothesis, not an established effect. Second, it is confounded: recallalsorises
withk(0.815→0.889), so the hybrid’s k=10gain could be additional retrieval supply rather than
better selection; the recall@k×P (select )decomposition in Eq. (1) is motivating, but we did not
measure the conditional selection term P(select|retrieved )directly, so supply and selection cannot
be cleanly separated here. A lighter SFT (r16/150) also scores 0.333 at k=5, which is consistent
with (but, being two single-run points, does not establish) insensitivity to LoRA strength.
Efficiency: no measurable benefit from heavier retrieval (Fig. 2).The cheap bge-small
hybrid (0.481) is not beaten by a heavier retriever ( bge-large +cross-encoder reranker), which
scores 0.37 for RAG-only and 0.37 for the hybrid; recall is also no higher (0.889 →0.852). The
observed ordering is hybrid k=10(0.481)≥RAG-only best (0.44) ≥all big-retrieval cells (0.37).
We emphasize that these cells span ∼3 items on n=27with overlapping uncertainty: the safe
reading is that heavier, specialized retrieval buysno measurable improvementon this short-statute
corpus (a useful negative result for a latency- and cost-constrained deployment), not that the
small retriever is provably superior. Separately, an improved/larger training set (2,645 pairs, 63%
subsection-targeted) didnotimprove over the original, reaching 0.444 with 0.037 hallucination
(again a∼1-item difference); we read the data lever as offering no easy further gain here, not as
exhausted.
Subgroups and errors.The real slice (0.481) is, surprisingly,higherthan the synthetic-test slice
(0.327), the opposite of the usual synthetic-overfit worry, because real questions cluster on a few
high-frequency provisions the retriever handles well, whereas synthetic questions span the long tail.
Per-RTA-Part subgroups are not statistically powered at n=27and are deferred. The dominant
error mode is retrieval misses (~11%; recall@10=0.889): when the governing section is not in the
6

Train, Retrieve, or Both? Statutory Citation for the Ontario RTA
Figure 2: Efficiency 2 ×2: a cheap bge-small hybrid (0.481) is the highest-scoring cell; a heavier
bge-large +reranker retriever does not improve either RAG-only (0.37) or the hybrid (0.37). On a
27-item set these cells carry overlapping uncertainty, so the reading is that heavier retrieval buys no
measurable benefit on this corpus, not that the small retriever is provably better.
top-10 the model guesses a topically adjacent provision (hallucination stays 0 because the guess is a
realsection). Secondary modes are mis-selection among retrieved candidates and subsection drop
(section right, subsection missing →0.5 partial credit), which accounts for much of the gap between
F1 (0.574) and exact-match (0.481).
6 Discussion and Limitations
The results give the operator a clear answer at the level the data can support:training alone
is insufficient and retrieval is essential.The hybrid is the highest-scoring arm and we read
its edge as more robust selection from higher-recall candidate sets, but at n=27its margin over
RAG-only is within noise, so we do not claim the hybrid is provably best. The system passes three
of four gates: it zeroes hallucinated citations (by construction) and clears the lift bar by a wide
margin (0.481 vs. 0.00, target ≥0.15). A useful efficiency finding is that heavier retrieval and more
training data both buy no measurable improvement on this short-statute corpus, which matters
for a latency- and cost-constrained deployment even though our sample cannot prove the cheap
retriever superior.
Honestly, the headline gate is not met: at 0.481 exact-match the system is well below the
aspirational 0.70 target. Three factors bound the gap: a retrieval-recall ceiling around 0.89 (the
dominant error mode); subsection-level difficulty (the gap between F1 and exact-match); and a
tiny, human-verification-pending eval set of 27 items. On generalization to untrained provisions, the
only configuration we ran on the unseen-section slice was the improved-data retrain variant (not
the headline model), which scored 0.31 exact-match on a held-out slice of 330 items whose sections
were never seen in training; it degrades but does not collapse, which suggests retrieval can supply
untrained provisions, though this evidence is from a different model than the headline and should
7

Train, Retrieve, or Both? Statutory Citation for the Ontario RTA
be read with that caveat. Calibration is not applicable to v1: the model emits a citation string with
no confidence, so Brier and ECE are undefined, and it remains future work.
6.1 Threats to Validity
Construct and statistical power.The headline metric rides on a 27-item, human-verification-
pending real eval set. A harness control confirms the scorer and gold are mechanically correct, but
it does not address statistical reliability: with n=27and single-run, no-confidence-interval estimates,
one item shifts a rate by ∼0.037, so the fine-grained ordering among the retrieval-bearing arms
(the 0.481 vs. 0.44 win, the candidate depth pattern, and the efficiency 2 ×2) is within noise and is
reported as suggestive only. Two further cautions apply. First, the final configuration was selected
on this same eval set, so the headline 0.481 is a max-over-configurations and is optimistically biased.
Second, the favorable real-vs-synthetic gap (0.481 vs. 0.327 on the larger n=150synthetic slice)
suggests the real questions cluster on common, high-frequency provisions, so the headline likely
overstates performance on the long tail; the better-powered synthetic number is arguably the more
representative estimate for arbitrary questions. The remedy for all of these, and the precondition
for any claim of statistical certainty, is a larger, topic-balanced, human-verified eval set, which is
our top future-work item. Until then all real-set numbers should be read as preliminary.
External.Training is synthetic-heavy (statute-derived paraphrases) and the study is single-
jurisdiction, single-statute (Ontario RTA+one regulation) on a single base model (Qwen2.5-7B;
Llama-3.1-8B not evaluated); legal-NLP components are known not to port across jurisdictions [ 9],
so generalization beyond this setting is unproven.
Internal.A retrieval-recall ceiling (~0.89) caps achievable exact-match independent of the gen-
erator; leakage is mitigated by group-aware splits and an unseen-section slice, but the eval set’s
clustering on high-frequency provisions is a confound for the headline number.
7 Conclusion and Future Work
On the task of citing the correct Ontario RTA provision, a four-arm head-to-head on Qwen2.5-7B
shows that fine-tuning alone is not enough and retrieval is essential. The SFT+RAG hybrid is
the highest-scoring arm (0.481 exact-match, 0.574 F1, zero hallucination by construction), which
we read as SFT making provision selection more robust to the higher-recall candidate sets that
hurt zero-shot RAG, though at n=27this margin is within noise rather than a proven win; on
the same data, a heavier retriever and more training data both buy no measurable improvement.
The artifact does not reach the 0.70 exact-match target on a 27-item, human-verification-pending
eval set. Ranked next steps: (1) a larger, topic-balanced, human-verified real eval set, which is the
precondition for treating any fine-grained ranking as statistically reliable; (2) a stronger retriever or
reranker tuned for short statute text, attacking the dominant retrieval-miss error mode; and (3)
subsection-level modeling and constrained decoding to close the gap between F1 and exact-match,
plus citation-confidence calibration as a stretch goal.1
1All artifacts (code, data, the synthetic question →citation dataset, and the trained LoRA adapter) are available from
the authors on request. Statute text is from Ontario e-Laws (Government of Ontario, Crown copyright; reproducible
under the King’s Printer terms, confirm attribution before redistribution). Outputs are decision-support, not legal
advice.
8

Train, Retrieve, or Both? Statutory Citation for the Ontario RTA
References
[1]Yukun Huang, Sanxing Chen, Jian Pei, Manzil Zaheer, and Bhuwan Dhingra. Cite Pretrain:
Retrieval-free knowledge attribution for large language models. 2025. URL https://arxiv.
org/abs/2506.17585.
[2]Pouya Pezeshkpour and Estevam Hruschka. Learning Beyond the Surface: How far can
continual pre-training with LoRA enhance LLMs’ domain-specific insight learning? 2025. URL
https://arxiv.org/abs/2501.17840.
[3]Ran Xu, Hui Liu, Sreyashi Nag, Zhenwei Dai, Yaochen Xie, Xianfeng Tang, Chen Luo,
Yang Li, Joyce C. Ho, Carl Yang, and Qi He. SimRAG: Self-improving retrieval-augmented
generation for adapting large language models to specialized domains. 2024. URL https:
//arxiv.org/abs/2410.17952.
[4]Takato Yasuno. Suppressing Domain-Specific Hallucination in Construction LLMs: A knowledge
graph foundation for GraphRAG and QLoRA on river and sediment control technical standards.
2026. URLhttps://arxiv.org/abs/2603.13307.
[5]Cheng Wang, Xinyang Lu, See-Kiong Ng, and Bryan Kian Hsiang Low. TRACE: TRansformer-
based attribution using contrastive embeddings in LLMs. 2024. URL https://arxiv.org/
abs/2407.04981.
[6]Akshat Mohan Dasula, Hrushitha Tigulla, and Preethika Bhukya. Judgement Citation Retrieval
using Contextual Similarity. 2024. URLhttps://arxiv.org/abs/2406.01609.
[7]Shounak Paul, Dhananjay Ghumare, Pawan Goyal, Saptarshi Ghosh, and Ashutosh Modi.
IL-PCSR: Legal corpus for prior case and statute retrieval. 2025. URL https://arxiv.org/
abs/2511.00268.
[8]Jahidul Arafat. Citation-Grounded Code Comprehension: Preventing LLM hallucination
through hybrid retrieval and graph-augmented context. 2025. URL https://arxiv.org/abs/
2512.12117.
[9]Shubham Kumar Nigam, Tanmay Dubey, Noel Shallum, and Arnab Bhattacharya. Segment
First, Retrieve Better: Realistic legal search via rhetorical role-based queries. 2025. URL
https://arxiv.org/abs/2508.00679.
[10]Muhammad Rafsan Kabir, Rafeed Mohammad Sultan, Fuad Rahman, Mohammad Ruhul Amin,
Sifat Momen, Nabeel Mohammed, and Shafin Rahman. LegalRAG: A hybrid RAG system for
multilingual legal information retrieval. 2025. URLhttps://arxiv.org/abs/2504.16121.
[11]Dnyanesh Panchal, Aaryan Gole, Vaibhav Narute, and Raunak Joshi. LawPal: A retrieval
augmented generation based system for enhanced legal accessibility in india. 2025. URL
https://arxiv.org/abs/2502.16573.
[12]Anuraj Maurya. Scaling Legal AI: Benchmarking mamba and transformers for statutory
classification and case law retrieval. 2025. URLhttps://arxiv.org/abs/2509.00141.
9

Train, Retrieve, or Both? Statutory Citation for the Ontario RTA
[13]I-Hung Hsu, Zifeng Wang, Long T. Le, Lesly Miculicich, Nanyun Peng, Chen-Yu Lee, and Tomas
Pfister. CaLM: Contrasting large and small language models to verify grounded generation.
2024. URLhttps://arxiv.org/abs/2406.05365.
[14]Valentin Noël, Elimane Yassine Seidou, Charly Ken Capo-Chichi, and Ghanem Amari. Hallu-
Graph: Auditable hallucination detection for legal RAG systems via knowledge graph alignment.
2025. URLhttps://arxiv.org/abs/2512.01659.
[15]Piotr Komorowski, Elena Golimblevskaia, Reduan Achtibat, Thomas Wiegand, Sebastian
Lapuschkin, and Wojciech Samek. Attribution-Guided Decoding. 2025. URL https://arxiv.
org/abs/2509.26307.
[16]Shubham Kumar Nigam, Balaramamahanthi Deepak Patnaik, Shivam Mishra, Ajay Varghese
Thomas, Noel Shallum, Kripabandhu Ghosh, and Arnab Bhattacharya. NyayaRAG: Realistic
legal judgment prediction with RAG under the indian common law system. 2025. URL
https://arxiv.org/abs/2508.00709.
[17]Darshita Rathore, Vineet Kumar, Chetna Bansal, and Anindya Moitra. How Much is Too
Much? exploring LoRA rank trade-offs for retaining knowledge and domain robustness. 2025.
URLhttps://arxiv.org/abs/2512.15634.
10
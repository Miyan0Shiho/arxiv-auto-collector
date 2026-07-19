# MamaBench: Benchmarking LLM Robustness in Maternal and Child Health Diagnosis through Counterfactual Clinical Perturbation

**Authors**: Thanni Adewuyi, Anuoluwa Sotome, Samuel Okoko, Angel Ezendu, Oluwafunke Akinbuwa, Oluwaseun Odunsi, Oluwasegun Oguntuase, Oluwadarasimi Oguntuase, Ifeoma Nwabueze, Abiodun Adereni

**Published**: 2026-07-15 21:56:19

**PDF URL**: [https://arxiv.org/pdf/2607.14385v1](https://arxiv.org/pdf/2607.14385v1)

## Abstract
Large language models achieve strong scores on medical benchmarks, yet these benchmarks evaluate each question in isolation, providing no measure of whether a system can distinguish clinically similar presentations requiring different interventions. We introduce MamaBench, the first counterfactual benchmark for maternal and paediatric AI: 434 expert-authored clinical narratives in 217 pairs across 371 pathologies, evaluated via the Bias Trap Rate (BTR), the conditional probability that a model fails the counterfactual given success on the base case. We propose Evidence-Anchored RAG (EA-RAG), a three-stage retrieval method that replaces aggregate similarity with an evidence coverage objective through clinical parameter extraction, coverage auditing, and contrastive sub-queries. Across eight configurations of four frontier LLMs, base accuracy overstates robust accuracy by 16-28 percentage points in every model. EA-RAG achieves 20.3% BTR and 65.0% robust accuracy on Claude Sonnet 4.6, a 5.5 percentage point BTR reduction without degrading base accuracy. The residual 20% BTR confirms that counterfactual robustness in clinical AI remains an open challenge. Keywords: counterfactual evaluation, clinical AI, maternal healthcare, retrieval-augmented generation, diagnostic robustness

## Full Text


<!-- PDF content starts -->

MamaBench: Benchmarking LLM Robustness in Maternal
and Child Health Diagnosis through Counterfactual
Clinical Perturbation
Thanni Adewuyi *1,2, Anuoluwa Sotome1, Samuel Okoko1, Angel Ezendu1,
Oluwafunke Akinbuwa1, Oluwaseun Odunsi1, Oluwasegun Oguntuase1,
Oluwadarasimi Oguntuase1, Ifeoma Nwabueze1, Abiodun Adereni†1
1Helpmum Africa
2University of Ibadan
Abstract
Large language models achieve strong scores on medical benchmarks, yet these benchmarks
evaluate each question in isolation, providing no measure of whether a system can dis-
tinguish clinically similar presentations requiring different interventions. We introduce
MamaBench, the first counterfactual benchmark for maternal and paediatric AI: 434 expert-
authored clinical narratives in 217 pairs across 371 pathologies, evaluated via theBias Trap
Rate (BTR), the conditional probability that a model fails the counterfactual given success
on the base case. We proposeEvidence-Anchored RAG (EA-RAG), a three-stage retrieval
method that replaces aggregate similarity with an evidence coverage objective through
clinical parameter extraction, coverage auditing, and contrastive sub-queries. Across eight
configurations of four frontier LLMs, base accuracy overstates robust accuracy by 16-28
percentage points in every model. EA-RAG achieves 20.3% BTR and 65.0% robust accuracy
on Claude Sonnet 4.6, a 5.5 percentage point BTR reduction without degrading base accuracy.
The residual 20% BTR confirms that counterfactual robustness in clinical AI remains an
open challenge.
Keywords:counterfactual evaluation, clinical AI, maternal healthcare, retrieval-augmented
generation, diagnostic robustness
1 Introduction
LLMs now pass medical licensing exams (Singhal et al., 2023; Nori et al., 2023) and outperform
physician baselines on clinical QA (Jin et al., 2021; Pal et al., 2022), accelerating deployment
via retrieval-augmented generation for domain-grounded decision support (Lewis et al., 2020;
Zakka et al., 2024). In maternal healthcare where a diagnostic error threatens two lives, such
systems carry both exceptional promise and exceptional risk (Topol, 2019).
Yet current benchmarks (MedQA, PubMedQA, MedMCQA) score each question indepen-
dently, hiding a dangerous failure mode: a system may answer correctly when the presentation
*tadewuyi949@stu.ui.edu.ng
†Corresponding author.biodun@helpmum.org
arXiv:2607.14385v1  [cs.CL]  15 Jul 2026

matches its prior, yet fail when a single discriminative parameter shifts the correct diagnosis.
Contrast sets (Gardner et al., 2020) and counterfactual augmentation (Kaushik et al., 2020)
exposed this fragility in general NLP; MedEinst (Chen et al., 2026) extended it to medical LLMs
via cases derived from DDXPlus (Fansi Tchango et al., 2022). No counterfactual benchmark yet
exists for maternal and paediatric care the domain where clinical stakes are highest.
We close this gap with two contributions:
1.MamaBench: 434 expert-authored clinical narratives in 217 counterfactual pairs across 371
pathologies, written in first-person patient-reported format by a three-member clinical team.
We formaliseDiagnostic Fixation(a model gets the base case right but rigidly persists when
discriminative parameters shift) and adopt theBias Trap Rate (BTR)(Chen et al., 2026) to
quantify it.
2.Evidence-Anchored RAG (EA-RAG): A three-stage inference-time pipeline that replaces
aggregate similarity retrieval with anevidence coverageobjective: extracting typed clinical
parameters, auditing retrieval coverage, and filling gaps via contrastive sub-queries, guided
by a taxonomy-grounded reasoning scaffold.
Across eight configurations of four frontier LLMs we find: (i) base accuracy overstates robust
accuracy by 16–28 percentage point ineverymodel; (ii) vanilla RAG provides no counterfactual
benefit; (iii) EA-RAG reduces BTR by 5.5 percentage point on the strongest model without
degrading base accuracy; (iv) even the best system still fails one in five counterfactual pairs.
2 Related Work
Medical QA benchmarks.MedQA (Jin et al., 2021), PubMedQA (Jin et al., 2019), and MedM-
CQA (Pal et al., 2022) evaluate isolated questions, enabling landmark results from Med-PaLM
(Singhal et al., 2023) and GPT-4 (Nori et al., 2023) but providing no pair-level discrimination
signal.
Counterfactual and robustness evaluation.Contrast sets (Gardner et al., 2020) and counterfac-
tually augmented data (Kaushik et al., 2020) showed that high-accuracy models exploit surface
cues; CheckList (Ribeiro et al., 2020) formalised behavioural testing via structured perturbations.
MedEinst (Chen et al., 2026) benchmarks the Einstellung effect (Luchins, 1942) through coun-
terfactual diagnosis on DDXPlus (Fansi Tchango et al., 2022), while Li et al. (2025) evaluated
clinical RAG robustness using a specialized counterfactual retrieval testbed. MamaBench differs
in being (a) entirely expert-authored, (b) focused on maternal/paediatric care, and (c) paired
with a retrieval-level intervention. Complementary robustness studies address paraphrasing
and demographic perturbations (Pfohl et al., 2024) but not diagnostic fixation.
Retrieval-augmented generation.RAG (Lewis et al., 2020) has been refined by dense retrieval
(Karpukhin et al., 2020), adaptive strategies (FLARE, Jiang et al. 2023; Self-RAG, Asai et al.
2024; IRCoT, Trivedi et al. 2023), clinical example-selection frameworks like MMRAG (Zhan
et al., 2025), and clinical deployment (Zakka et al., 2024). None addressesretrieval stagnation:
near-identical embeddings for base–counterfactual pairs yield the same retrieved context (Shi
et al., 2023). EA-RAG fills this gap through evidence coverage, drawing on query decomposition
(Khattab et al., 2023; Wang et al., 2023) while operating entirely at inference time.
2

Structured reasoning.CoT (Wei et al., 2022) and Tree of Thoughts (Yao et al., 2023) improve
reasoning but are task-agnostic. EA-RAG’s scaffold derives from empirical failure taxonomy,
targeting specific failure categories (missed critical info, underthinking, factual inaccuracy,
overthinking) rather than providing generic instructions.
3 MamaBench Benchmark
3.1 Problem Formulation
We formalise diagnosis as f:X →Y , where Xis the space of patient-reported narratives and Y
the label space of 371 pathologies. ACounterfactual Pair (xb,xc)comprises a base case xbwith
ground truth yband a counterfactual xc—a minimal perturbation shifting the ground truth to
yc̸=y bwhile preserving shared context.
Definition 1 (Diagnostic Fixation).Model fexhibits Diagnostic Fixation on (xb,xc)
iff:
f(x b) =y b∧f(x c) =y b∧y c̸=y b (1)
The model succeeds in the base case, but persists rigidly when discriminative parameters
change analogously to the Einstellung effect (Luchins, 1942). Diagnostic Fixation is architecture-
agnostic: it can arise from memorised priors, retrieval stagnation, or heuristic shortcuts.
3.2 Construction and Statistics
A three-member medical team (obstetrics, gynaecology, paediatrics) authored each narrative
infirst-person patient-reported format: e.g.,“My baby was delivered at 37 weeks by a traditional
birth attendant. He is 10 days old and refusing to eat. . . ”introducing naturalistic ambiguity absent
from structured vignettes (Jin et al., 2021). Unlike benchmarks derived from existing datasets
(Fansi Tchango et al., 2022) or LLM pipelines (Chen et al., 2026), every case is individually
expert-authored.
For each base case, the team constructs a counterfactual by perturbing the fewest parameters
necessary to shift the diagnosis, following five categories: symptom substitution, severity
escalation, risk factor modification, temporal shift, and comorbidity introduction.
MamaBench totals217 pairs(434 cases): paediatrics 254 (58.5%), obstetrics/gynaecology
178 (41.0%), unclassified 2 (0.5%), covering 371 unique pathologies. All cases passed multi-stage
clinical review for diagnostic sufficiency, counterfactual validity, and minimal edit properties.
3.3 Evaluation Metrics
Each pair is classified asPP(both correct),PF(base correct, counterfactual wrong),FP, orFF.
We report three metrics:
•Base Accuracy:Acc base= (|PP|+|PF|)/N
•Bias Trap Rate(primary metric):
BTR=|PF|
|PP|+|PF|×100% (2)
•Robust Accuracy:Acc rob=|PP|/N
Predictions are assessed by an LLM-as-judge (Claude Opus 4.5, T=0): lenient on nomenclature,
strict on clinically significant modifiers that change management (Zheng et al., 2023).
3

Figure 1:Standard RAG (left) vs. EA-RAG (right). Standard RAG retrieves top- kchunks by aggregate
query similarity, returning near-identical context for base and counterfactual cases (retrieval stagnation).
EA-RAG introduces three additional mechanisms: (1) evidence extraction of typed clinical parameters,
(2) coverage auditing with contrastive sub-queries to fill gaps, and (3) a taxonomy-grounded generation
scaffold.
4 Evidence-Anchored RAG
Standard RAG retrieves the top- kchunks by maximising aggregate query similarity: R(q) =
top-ksim(ϕ(q) ,ϕ(c i)). For a base–counterfactual pair (q,q′)with sim(ϕ(q) ,ϕ(q′))≈ 0.94, this
yields R(q)≈R(q′)—the generator reasons from the same context and produces the same
diagnosis (retrieval stagnation). EA-RAG replaces this with anevidence coverageobjective
(Fig. 1).
4.1 Stage 1: Evidence Extraction
A structured LLM call (GPT-5.4-nano, T=0) parses the patient narrative into typed clinical
parameters Eq={e 1,. . .,em}spanning patient demographics, presenting condition and severity
markers, risk factors, contraindications, and management-relevant features. Null fields are
informative: their absence signals the query does not specify that parameter. If extraction fails,
the system falls back to single-pass retrieval, ensuring EA-RAG never degrades below baseline.
4.2 Stage 2: Two-Pass Retrieval with Coverage Auditing
Pass 1retrieves the top- k(k=5) chunks via cosine similarity over text-embedding-3-large
embeddings ( d=3072). Acoverage auditthen scores each evidence element ejagainst the
retrieved set: cov(e j,R(1)) =maxci∈R(1)cos(ϕ(e j),ϕ(c i)). Elements with cov<θ (θ=0.6) form
the gap setG.
Pass 2generates acontrastive sub-queryfor each ej∈G framed as “How does ejchange
the management ofcondition?” and retrieves supplementary chunks. After deduplication
(δ=0.82), the final set Rfinal=R(1)∪R(2)is capped at k+3=8 chunks. This mechanism ensures
4

Table 1:MamaBench results. Best per column inbold. All values are percentages.
Model SetupAcc base BTR↓Acc rob↑
GLM-5 Standalone 54.8 50.4 27.2
Kimi K2.5 Standalone 76.0 40.0 45.6
GPT-4o RAG Baseline 77.4 39.9 46.5
Nemotron 120B Standalone 72.4 35.0 47.0
GPT-4oEA-RAG (Ours)77.0 35.9 49.3
GPT-5.4 Standalone 77.4 25.6 57.6
Claude Sonnet 4.6 Standalone82.025.8 60.8
Claude Sonnet 4.6EA-RAG (Ours)81.620.3 65.0
discriminative parameters surface in the context even when the overall query embedding fails
to distinguish them.
4.3 Stage 3: Taxonomy-Grounded Generation
The generator receives the narrative, Rfinal, and ascaffold instruction setderived from error
taxonomy analysis of 113 baseline failures:
Failure Mode Freq. Scaffold Response
Missed critical info 48.7% Analyse ALL clinical parameters
Underthinking 31.9% Match to specific severity level
Factual inaccuracy 10.6% Check contraindications first
Overthinking 8.8% State conclusion decisively
The scaffold enforces aclinical-reasoning-firsthierarchy: the model must diagnose from the pre-
sentation before consulting retrieved evidence, may override retrieval when clinical reasoning
diverges, and must verify the final diagnosis accounts for all key findings. This contrasts with
generic CoT prompting (Wei et al., 2022) and training-time approaches like Self-RAG (Asai
et al., 2024) EA-RAG operates entirely at inference time with no fine-tuning.
5 Experiments and Results
5.1 Setup
We evaluate eight configurations: five standalone models (GLM-5, Kimi K2.5, Nemotron Su-
per 120B, GPT-5.4, Claude Sonnet 4.6), a vanilla RAG baseline (GPT-4o, k=2), and EA-RAG on
GPT-4o (k=5) and Claude Sonnet 4.6. All atT=0; all 217 pairs evaluated per configuration.
Data separation.To ensure evaluation integrity, no MamaBench case narratives or ground
truth labels were included in the RAG or EA-RAG knowledge base. The retrieval corpus
consists exclusively of clinical reference material (textbook guidelines, WHO protocols, and
clinical management standards); the benchmark data remains unseen by the retrieval system at
all times.
5.2 Main Results
Base accuracy is misleading.Therobustness gap—the difference between Acc baseand Acc rob—
ranges from 16.2 pp (Claude + EA-RAG: 82.0 →65.0) to 27.6 pp (GLM-5: 54.8 →27.2). GPT-4o
achieves 77.4% base accuracy but only 46.5% robust accuracy ( BTR= 39.9%): nearly two in five
“correct” diagnoses are fragile. No model closes the gap below 16 pp, confirming counterfactual
vulnerability as a systemic property of current clinical AI.
5

Figure 2:Bias Trap Rate (BTR) vs. Robustness across different model configurations. The x-axis denotes
the Bias Trap Rate (where lower percentages are better), and the y-axis represents Robustness (where
higher percentages are better). Standalone models are marked as red circles, standard RAG as blue
squares, and the proposed EA-RAG configurations as gold stars. The top-left corner represents the ideal
performance quadrant. The plot illustrates that while standard RAG improves baseline robustness, the
EA-RAG framework uniquely minimizes the bias trap while simultaneously maximizing robustness,
driving configurations toward the ideal zone.
EA-RAG improves robustness without harming base accuracy.GPT-4o: BTR drops 39.9 →
35.9% ( −4.0 pp); Claude: 25.8 →20.3% ( −5.5 pp). Base accuracy decreases by only 0.4 pp in
both cases. The stronger model benefits more, consistent with evidence coverage being most
effective when reasoning capacity is sufficient and retrieval quality is the bottleneck.
Vanilla RAG does not help.GPT-4o with RAG ( BTR= 39.9%) matches standalone models of
similar capability (Kimi K2.5: 40.0%). Retrieval stagnation renders standard top- kretrieval in-
visible to the counterfactual perturbation—confirming that the problem lies inwhatis retrieved,
notwhetherretrieval occurs.
Scale reduces but does not eliminate the problem. BTR drops from 50.4% (GLM-5) to 25.6%
(GPT-5.4) across standalone models, but even Claude Sonnet 4.6 standalone (25.8%) fails on
more than one in four pairs where it demonstrates base competence.
The best system still fails 1 in 5.Claude + EA-RAG achieves the lowest BTR (20.3%) and
highest Acc rob(65.0%). In deployment, this means a system thatappearsreliable will miss
discriminative clinical parameters in ∼20% of diagnostically ambiguous cases—underscoring
that counterfactual robustness remains an open research challenge.
5.3 Ablation Study
To isolate the contribution of each EA-RAG component, we conduct an additive ablation using
GPT-4o as the base model. Starting from the k5configuration (top- k= 5, no scaffold, no
coverage), we incrementally enable each module (Table 2).
Coverage alone is insufficient. k5_coverage adds evidence extraction and coverage auditing
but no contrastive sub-queries or scaffold. It produces identical BTR and Acc robto the k5
6

/uni0000002a/uni0000002f/uni00000030/uni00000010/uni00000018 /uni0000002e/uni0000004c/uni00000050/uni0000004c/uni00000003/uni0000002e/uni00000015/uni00000011/uni00000018 /uni00000035/uni00000024/uni0000002a
/uni0000000b/uni0000002a/uni00000033/uni00000037/uni00000010/uni00000017/uni00000052/uni0000000c/uni00000031/uni00000048/uni00000050/uni00000052/uni00000057/uni00000055/uni00000052/uni00000051
/uni00000014/uni00000015/uni00000013/uni00000025/uni00000028/uni00000024/uni00000010/uni00000035/uni00000024/uni0000002a
/uni0000000b/uni0000002a/uni00000033/uni00000037/uni00000010/uni00000017/uni00000052/uni0000000c/uni0000002a/uni00000033/uni00000037/uni00000010/uni00000018/uni00000011/uni00000017 /uni00000026/uni0000004f/uni00000044/uni00000058/uni00000047/uni00000048/uni00000003/uni00000036/uni00000052/uni00000051/uni00000051/uni00000048/uni00000057/uni00000003/uni00000017/uni00000011/uni00000019 /uni00000028/uni00000024/uni00000010/uni00000035/uni00000024/uni0000002a
/uni0000000b/uni00000036/uni00000052/uni00000051/uni00000051/uni00000048/uni00000057/uni00000003/uni00000017/uni00000011/uni00000019/uni0000000c/uni00000013/uni00000008/uni00000015/uni00000013/uni00000008/uni00000017/uni00000013/uni00000008/uni00000019/uni00000013/uni00000008/uni0000001b/uni00000013/uni00000008/uni00000014/uni00000013/uni00000013/uni00000008/uni00000033/uni00000055/uni00000052/uni00000053/uni00000052/uni00000055/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni00000052/uni00000049/uni00000003/uni00000033/uni00000044/uni0000004c/uni00000055/uni00000056/uni00000003/uni0000000b/uni00000008/uni0000000c/uni00000033/uni00000044/uni0000004c/uni00000055/uni00000010/uni0000002f/uni00000048/uni00000059/uni00000048/uni0000004f/uni00000003/uni00000032/uni00000058/uni00000057/uni00000046/uni00000052/uni00000050/uni00000048/uni00000003/uni00000027/uni0000004c/uni00000056/uni00000057/uni00000055/uni0000004c/uni00000045/uni00000058/uni00000057/uni0000004c/uni00000052/uni00000051
/uni00000033/uni00000033/uni00000003/uni0000000b/uni00000025/uni00000052/uni00000057/uni0000004b/uni00000003/uni00000026/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057/uni0000000c /uni00000033/uni00000029/uni00000003/uni0000000b/uni00000025/uni0000004c/uni00000044/uni00000056/uni00000003/uni00000037/uni00000055/uni00000044/uni00000053/uni0000000c /uni00000029/uni00000033/uni00000003/uni0000000b/uni00000035/uni00000048/uni00000059/uni00000048/uni00000055/uni00000056/uni00000048/uni0000000c /uni00000029/uni00000029/uni00000003/uni0000000b/uni00000025/uni00000052/uni00000057/uni0000004b/uni00000003/uni0000003a/uni00000055/uni00000052/uni00000051/uni0000004a/uni0000000cFigure 3:Pair-level outcome distribution. PF (red) = bias traps: model appears competent but fails under
perturbation. EA-RAG shrinks PF while expanding PP .
Table 2:Ablation study on EA-RAG components (GPT-4o). Each row adds one module to the k5baseline.
PP/PF/FP/FF are pair-level outcome counts (N=217).
Variant PP PF FP FFAcc base BTR↓Acc rob↑
k594 61 36 26 71.4% 39.4% 43.3%
k5_coverage94 61 37 25 71.4% 39.4% 43.3%
k5_scaffold101 62 35 19 75.1% 38.0% 46.5%
ea_rag_full107 6032 1877.0% 35.9% 49.3%
baseline (39.4% / 43.3%), indicating thatdetectingcoverage gaps withoutfillingthem orguiding
generation has negligible effect. The retrieval audit alone does not change what the generator
sees.
The scaffold drives the largest single gain. k5_scaffold adds the taxonomy-grounded rea-
soning scaffold without coverage-aware retrieval. This lifts Acc baseby 3.7 pp (71.4% →75.1%),
Acc robby 3.2 pp, and reduces BTR by 1.4 pp. Seven additional pairs shift from PF →PP (and
seven from FF →FP/PP), confirming that structured generation instructions particularly the
clinical-reasoning-first hierarchy and permission to override retrieved context—improve diag-
nostic reasoning even with unchanged retrieval.
The full pipeline is greater than the sum of its parts. ea_rag_full combines all three stages
and achieves the best results on every metric: Acc base=77.0%, BTR= 35.9%, Acc rob=49.3%.
Compared to k5_scaffold , the addition of coverage-aware retrieval with contrastive sub-queries
converts six more pairs from incorrect to correct ( PP: 101→107), yielding a further 2.1 pp BTR
reduction and 2.8 pp Acc robgain. This confirms that the scaffold and coverage modules are
complementary: the scaffold tells the generatorhowto reason, while coverage-aware retrieval
ensures it has theevidenceto reason over.
6 Conclusion
We introduced MamaBench, the first counterfactual benchmark for maternal and paediatric
AI (217 expert-authored pairs, 371 pathologies), and EA-RAG, an evidence-anchored retrieval
7

method. Base accuracy systematically overstates clinical competence by 16–28 pp across all
models. EA-RAG reduces BTR by 5.5 pp on the strongest model without degrading base
accuracy, confirming retrieval quality as the primary bottleneck. The residual 20% BTR shows
retrieval-level interventions alone are insufficient.
Limitations.MamaBench’s 434 cases cover a single clinical domain in English only, which
constrains how far our findings generalize to other specialties or languages. Our reliance on
LLM-as-judge evaluation may introduce systematic biases correlated with the judge model’s
own blind spots (Zheng et al., 2023), and while expert authoring gives us high-quality counter-
factuals, it caps how quickly the benchmark can scale to new cases or domains.
References
Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). Self-RAG: Learning to retrieve, generate, and
critique through self-reflection.ICLR.
Chen, W., Huang, G., Wang, W., & Zhu, Z. (2026). MedEinst: Benchmarking the Einstellung Effect in
medical LLMs through counterfactual differential diagnosis.Proceedings of the 64th Annual Meeting of
the Association for Computational Linguistics (ACL 2026).
Fansi Tchango, A., Goel, R., Wen, Z., Martel, J., & Ghosn, J. (2022). DDXPlus: A new dataset for automatic
medical diagnosis.NeurIPS.
Gardner, M., Artzi, Y., Basmova, V ., et al. (2020). Evaluating models’ local decision boundaries via
contrast sets.Findings of EMNLP, 1307–1323.
Jiang, Z., Xu, F. F., Gao, L., et al. (2023). Active retrieval augmented generation.EMNLP, 7969–7992.
Jin, D., Pan, E., Oufattole, N., et al. (2021). What disease does this patient have?Applied Sciences, 11(14),
6421.
Jin, Q., Dhingra, B., Liu, Z., Cohen, W. W., & Lu, X. (2019). PubMedQA: A dataset for biomedical research
question answering.EMNLP, 2567–2577.
Karpukhin, V ., Oguz, B., Min, S., et al. (2020). Dense passage retrieval for open-domain question
answering.EMNLP, 6769–6781.
Kaushik, D., Hovy, E., & Lipton, Z. C. (2020). Learning the difference that makes a difference with
counterfactually-augmented data.ICLR.
Khattab, O., Santhanam, K., Li, X. L., et al. (2023). Demonstrate-Search-Predict: Composing retrieval and
language models for knowledge-intensive NLP.arXiv:2212.14024.
Lewis, P ., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive
NLP tasks.NeurIPS, 9459–9474.
Luchins, A. S. (1942). Mechanization in problem solving: The effect of Einstellung.Psych. Monographs,
54(6), i–95.
Nori, H., King, N., McKinney, S. M., et al. (2023). Capabilities of GPT-4 on medical competency examina-
tions.arXiv:2303.13375.
Pal, A., Umapathi, L. K., & Sankarasubbu, M. (2022). MedMCQA: A large-scale multi-subject multi-choice
dataset for medical domain question answering.CHIL, 248–260.
Pfohl, S. R., Cole-Lewis, H., Sayres, R., et al. (2024). A toolbox for surfacing health equity harms and
biases in large language models.arXiv:2403.12025.
8

Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (2020). Beyond accuracy: Behavioral testing of NLP
models with CheckList.ACL, 4902–4912.
Shi, F., Chen, X., Misra, K., et al. (2023). Large language models can be easily distracted by irrelevant
context.ICML.
Singhal, K., Azizi, S., Tu, T., et al. (2023). Large language models encode clinical knowledge.Nature, 620,
172–180.
Topol, E. J. (2019). High-performance medicine: The convergence of human and artificial intelligence.
Nature Med., 25, 44–56.
Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2023). Interleaving retrieval with chain-of-
thought reasoning for knowledge-intensive multi-step questions.ACL.
Wang, L., Yang, N., & Wei, F. (2023). Query2doc: Query expansion with large language models.EMNLP,
9414–9423.
Wei, J., Wang, X., Schuurmans, D., et al. (2022). Chain-of-thought prompting elicits reasoning in large
language models.NeurIPS, 24824–24837.
Yao, S., Yu, D., Zhao, J., et al. (2023). Tree of thoughts: Deliberate problem solving with large language
models.NeurIPS.
Zakka, C., Shad, R., Chaurasia, A., et al. (2024). Almanac: Retrieval-augmented language models for
clinical medicine.NEJM AI, 1(2).
Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot
Arena.NeurIPS.
Li, M., Zhan, Z., Yang, H., Xiao, Y., Zhou, H., Huang, J., & Zhang, R. (2025). Benchmarking retrieval-
augmented large language models in biomedical NLP: Application, robustness, and self-awareness.
Science Advances, 11(47), eadr1443.
Zhan, Z., Wang, J., Zhou, S., Deng, J., & Zhang, R. (2025). MMRAG: multi-mode retrieval-augmented
generation with large language models for biomedical in-context learning.Journal of the American
Medical Informatics Association, 32(10), 1505–1516.
9
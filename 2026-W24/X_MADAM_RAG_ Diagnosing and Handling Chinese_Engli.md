# X-MADAM-RAG: Diagnosing and Handling Chinese-English Evidence Conflict in Retrieval-Augmented Generation

**Authors**: Yongqi Kang, Yu Fu, Yong Zhao

**Published**: 2026-06-11 04:59:35

**PDF URL**: [https://arxiv.org/pdf/2606.12903v1](https://arxiv.org/pdf/2606.12903v1)

## Abstract
Retrieval-augmented generation (RAG) systems may receive evidence that is not merely noisy but mutually contradictory. This issue becomes particularly salient in multilingual settings, where retrieved Chinese and English evidence may support incompatible answer candidates. We study this problem through X-RAMDocs-ZHEN, a controlled Chinese-English benchmark derived from RAMDocs for diagnosing evidence conflict in RAG. The benchmark contains 300 examples across six balanced conditions, including monolingual support, bilingual agreement, reversed conflict directions, and conflict with optional noise. We further examine X-MADAM-RAG, an interpretable pipeline that decomposes evidence handling into per-document candidate extraction, visible-evidence repair, deterministic candidate grouping, and conflict-aware aggregation. On the original controlled benchmark with Qwen2.5-7B-Instruct, X-MADAM-RAG achieves 0.9667 strict accuracy and 0.9767 conflict-aware success, outperforming an evidence-normalized single-call baseline. However, a zero-call rule-only extractor reaches 1.0000 on the same benchmark, revealing strong template regularity. To probe this limitation, we construct a deterministic naturalized stress test that removes explicit answer templates while preserving candidate strings. On its 100-sample subset, rule-only extraction falls to 0.0000, but X-MADAM-RAG also drops to 0.3000 strict accuracy, below both naive and evidence-normalized baselines. A privileged oracle remains perfect, indicating that document-level extraction is the main bottleneck. These findings position X-RAMDocs-ZHEN and X-MADAM-RAG as diagnostic tools for controlled evidence conflict rather than as evidence of general hallucination detection or robustness to natural retrieval.

## Full Text


<!-- PDF content starts -->

X-MADAM-RAG: Diagnosing and Handling
Chinese-English Evidence Conflict in
Retrieval-Augmented Generation
Yongqi Kang, Yu Fu, and Yong Zhao∗
Sichuan University, Chengdu, China
2023141520239@stu.scu.edu.cn
fuyu05@stu.scu.edu.cn
yong.zhao@scupi.cn
Abstract.Retrieval-augmented generation (RAG) systems may receive
evidence that is not merely noisy but mutually contradictory. This is-
sue becomes particularly salient in multilingual settings, where retrieved
Chinese and English evidence may support incompatible answer candi-
dates. We study this problem throughX-RAMDocs-ZHEN, a con-
trolled Chinese-English benchmark derived from RAMDocs for diagnos-
ing evidence conflict in RAG. The benchmark contains 300 examples
across six balanced conditions, including monolingual support, bilingual
agreement, reversed conflict directions, and conflict with optional noise.
We further examineX-MADAM-RAG, an interpretable pipeline that
decomposes evidence handling into per-document candidate extraction,
visible-evidence repair, deterministic candidate grouping, and conflict-
aware aggregation. On the original controlled benchmark with Qwen2.5-
7B-Instruct,X-MADAM-RAGachieves0.9667strictaccuracyand0.9767
conflict-aware success, outperforming an evidence-normalized single-call
baseline. However, a zero-call rule-only extractor reaches 1.0000 on the
same benchmark, revealing strong template regularity. To probe this lim-
itation, we construct a deterministic naturalized stress test that removes
explicit answer templates while preserving candidate strings. On its 100-
sample subset, rule-only extraction falls to 0.0000, but X-MADAM-RAG
also drops to 0.3000 strict accuracy, below both naive and evidence-
normalized baselines. A privileged oracle remains perfect, indicating that
document-levelextractionisthemainbottleneck.Thesefindingsposition
X-RAMDocs-ZHEN and X-MADAM-RAG as diagnostic tools for con-
trolled evidence conflict rather than as evidence of general hallucination
detection or robustness to natural retrieval.
Keywords:Retrieval-augmentedgeneration·Knowledgeconflict·Mul-
tilingual NLP·Evidence aggregation.
1 Introduction
Retrieval-augmented generation has become a widely used approach for improv-
ing the factual grounding of large language models. By conditioning generationarXiv:2606.12903v1  [cs.CL]  11 Jun 2026

2 F. Author et al.
on external evidence, RAG systems aim to reduce reliance on parametric mem-
ory and produce responses that are better supported by retrieved documents.
However,retrievedevidenceisnotalwaysconsistent.Inrealisticmultilingualsce-
narios, a system may retrieve Chinese and English sources that support different
answers to the same question—for example, a Chinese snippet may support one
date while an English snippet supports another, forcing the system to preserve
disagreement rather than silently select one candidate.
Most multilingual RAG work focuses on whether cross-lingual evidence can
help answer questions. We focus on a complementary diagnostic question: how
do RAG systems behave when evidence in different languages supports conflict-
ing answers? We are particularly interested in whether a system can preserve
evidence-supported candidates, recognize conflict, and avoid unsupported can-
didate generation.
Tostudythis,weconstructX-RAMDocs-ZHENv0.1,acontrolledChinese-
English benchmark with 300 examples across six balanced evidence conditions.
We also studyX-MADAM-RAG, an interpretable pipeline that extracts can-
didates per document, applies label-blind repair, groups candidates determinis-
tically, and aggregates the extracted candidates in a conflict-aware manner. On
the original benchmark, X-MADAM-RAG substantially outperforms single-call
LLM baselines, but a rule-only extractor also solves it perfectly. A naturalized
stress test removes explicit answer templates: rule-only extraction collapses, but
X-MADAM-RAG also degrades below single-call baselines while a privileged
oracle remains perfect, identifying extraction as the primary bottleneck.
Our contributions are threefold:
1. WeintroduceX-RAMDocs-ZHENv0.1,aleakage-awarecontrolledChinese-
English evidence-conflict benchmark with six balanced conditions.
2. We presentX-MADAM-RAG, an auditable pipeline with per-document
extraction,visible-evidencerepair,deterministicgrouping,andconflict-aware
aggregation.
3. We provide diagnostic analyses showing original-template success, ablation
contributions, and naturalized extraction fragility—identifying document-
level extraction robustness as the central open problem.
We do not claim that the benchmark is human-validated, that X-MADAM-
RAG solves hallucination detection, or that the results generalize to natural
multilingual retrieval.
2 Related Work
Prior work on RAG faithfulness and hallucination diagnostics evaluates whether
generatedoutputsaresupportedbyretrievedcontext.FaithEvalexaminesmodel
faithfulness under parametric-contextual conflict [6], FaithfulRAG models fact-
level conflict [12], RAGChecker provides fine-grained RAG diagnostics [8], and
RefChecker targets knowledge-centric hallucination [4]. Multilingual hallucina-
tion has also been studied in shared tasks such as Mu-SHROOM [9]. Our work

Diagnosing Chinese-English Evidence Conflict in RAG 3
is narrower: we diagnose answer-candidate behavior under controlled Chinese-
English evidence conflict using automatic lexical metrics rather than open-ended
semantic faithfulness.
Knowledgeconflictariseswhenretrievedevidencecontainsincompatibleclaims
or contradicts parametric knowledge [11,2,1,3]. RAMDocs and MADAM-RAG
directly motivate our benchmark design by studying RAG with conflicting evi-
dence [10]. We extend this setting with a cross-lingual dimension: Chinese and
English evidence are controlled to agree, disagree, or reverse which language car-
ries the reference candidate. Decomposition-based approaches such as MADAM-
RAG and FaithfulRAG process evidence in smaller units before aggregation; X-
MADAM-RAG follows this line but uses deterministic candidate grouping after
document-level extraction, making each stage separately inspectable.
Multilingual and cross-lingual RAG systems retrieve and generate across lan-
guages.XRAGstudiescross-lingual retrieval-augmented generation[5],andmul-
tilingualRAGworkhasexaminedknowledge-intensivequestionansweringacross
languages [7]. Our work complements these by focusing specifically on answer-
level disagreement between Chinese and English evidence, with controlled con-
ditions that isolate conflict direction from retrieval noise.
3 X-RAMDocs-ZHEN Benchmark
3.1 Task and Dataset
Given a question and a set of Chinese and/or English evidence snippets, the sys-
tem must produce an answer or a conflict-aware response using only the visible
evidence. Evidence may support a single candidate, multiple matching candi-
dates, or incompatible candidates across languages. A desirable system should
preserve evidence-supported candidates, recognize disagreement, avoid unsup-
ported candidates, and provide traceable evidence use. The task is diagnostic:
evidence conditions are controlled and private metadata is used only for con-
struction and evaluation.
X-RAMDocs-ZHEN v0.1is derived from RAMDocs and contains 300
samples across 534 evidence documents, split evenly between Chinese and En-
glish. Questions are bilingual (175 Chinese, 125 English). Conflicts cover entities
(144), dates (101), and numbers (55). The benchmark uses 54 unique source
records from 68 usable RAMDocs records, with 100% provenance coverage and
zeroautomaticsevereerrorsorwarnings.Nocompletedhumanauditisavailable.
3.2 Evidence Conditions
The benchmark contains six balanced evidence conditions (Table 1).
3.3 Construction and Leakage Control
Figure1summarizestheconstructionandleakage-controldesignofX-RAMDocs-
ZHEN. The benchmark starts from RAMDocs records, normalizes question, an-

4 F. Author et al.
Table 1.The six balanced evidence conditions. Internal labels are preserved for repro-
ducibility.
Display Label Internal Label Description N
C1: Chinese-only
correctC1_ZH_ONLY_CORRECTCorrect evidence in Chinese. 50
C2: English-only
correctC2_EN_ONLY_CORRECTCorrect evidence in English. 50
C3: Bilingual con-
sistentC3_BILINGUAL_CONSISTENTBoth languages support same can-
didate.50
C4: Chinese true,
EN falseC4_ZH_TRUE_EN_FALSEChinese supports reference; English
supports competing candidate.50
C5: English true,
ZH falseC5_EN_TRUE_ZH_FALSEEnglish supports reference; Chinese
supports competing candidate.50
C6: Conflict with
optional noiseC6_CONFLICT_WITH_NOISEBilingual conflict; noise present in
34 of 50 cases.50
swer, and document fields, renders controlled Chinese–English evidence snip-
pets, assigns one of the six balanced evidence conditions, and replaces private
source identifiers with public evidence IDs. Standard methods receive only pub-
lic prompt fields: the question, visible evidence text, public evidence IDs, and
language tags. Gold answers, document roles, language roles, condition labels,
expected behavior, and supported-answer metadata are hidden from standard
prompts and used only for construction and automatic evaluation.
This separation ensures that standard models cannot infer the target behav-
ior from hidden labels or construction metadata. The oracle diagnostic is the
only exception: it accesses document-level supported-answer metadata and is
therefore treated only as a privileged upper-bound analysis, not as a deployable
baseline.
3.4 Naturalized Stress Test
Because the original benchmark uses controlled evidence templates, we construct
a naturalized stress test. This deterministic transformation removes explicit an-
swer templates while preserving answer candidate strings and private metadata,
testing whether methods depend on surface cues such as “the answer is” or “答案
是”. The full transformed set covers all 300 samples and 534 documents with zero
remaining templates and perfect candidate and metadata preservation. We eval-
uate a 100-sample approximately balanced subset; it remains controlled prose
rather than natural retrieval.

Diagnosing Chinese-English Evidence Conflict in RAG 5
Fig. 1.Construction and leakage-control flow of X-RAMDocs-ZHEN. Standard meth-
ods receive only the question, visible evidence text, public evidence IDs, and language
tags. Gold answers, document roles, conflict labels, language roles, expected behavior,
and supported-answer metadata are hidden from standard prompts and used only for
construction and automatic evaluation. The oracle diagnostic is the only method that
accesses document-level supported-answer metadata and is therefore non-deployable.
4 Methods
4.1 Baselines and Diagnostics
Thenaive RAGbaseline concatenates all visible evidence into one prompt and
asks the model to return an answer, evidence identifiers, a conflict flag, and
notes. Theevidence-normalized RAGbaseline groups visible evidence by
language before a single model call, testing whether explicit language organiza-
tion helps. Therule-only extractorapplies conservative Chinese and English
surface patterns, groups normalized candidates deterministically, and generates
a fixed response with zero model calls; it measures template dependence, not
semantic extraction. Theoracle diagnosticreplaces document-level extraction
with privatesupported_answermetadata and applies the same downstream
grouping and aggregation; it is privileged and non-deployable, used only to esti-
mate the impact of imperfect extraction.
4.2 X-MADAM-RAG
Figure2givesanoverviewofX-MADAM-RAG.Thepipelinedecomposesevidence-
conflict handling into document-level extraction, visible-evidence repair, candi-
datenormalization,deterministicgrouping,andconflict-awareaggregation.Each
evidence document is processed independently by an extraction agent that re-
ceives only the question, public evidence ID, language tag, and visible evidence
text.

6 F. Author et al.
Fig. 2.X-MADAM-RAG pipeline. Eachevidencedocument isprocessed independently
byanextractionagent.Missingorunusablecandidatesmayberepairedfromvisibleev-
idence only. Normalized candidates are then grouped deterministically. A single group
yields a direct answer, multiple groups yield conflict enumeration, and no group yields
abstention. The no-repair and no-grouping ablations remove the corresponding com-
ponents, while the oracle diagnostic replaces extraction with privileged document-level
metadata.
When extraction fails or returns an unusable candidate, a label-blind repair
module searches parsed fields, raw agent output, and visible evidence; it cannot
access hidden metadata. Repaired candidates are normalized and grouped de-
terministically. A single candidate group yields a direct answer, multiple groups
yield explicit conflict enumeration, and no candidate yields abstention.
We evaluate two ablations. Theno-repairvariant removes the repair stage
while keeping extraction, grouping, and aggregation. Theno-groupingvari-
ant replaces deterministic grouping with a model-based conflict decision. These
isolate the contribution of each component. The oracle diagnostic replaces only
the extraction stage with private document-level supported-answer metadata,
distinguishing extraction failures from downstream grouping and aggregation
failures.

Diagnosing Chinese-English Evidence Conflict in RAG 7
Table 2.Results on the original controlled benchmark (300 samples). Best values per
column in bold. Rule-only and oracle rows are diagnostic: rule-only measures template
dependence; oracle uses privilegedsupported_answermetadata and is non-deployable.
Method Coverage Strict Conf.-aware Conf. F1 Awareness Unsupported
Naive 1.0000 0.8700 0.8667 0.9579 0.7400 0.0800
Evidence-normalized 1.0000 0.8900 0.8867 0.9766 0.7533 0.1067
Rule-only extractor 1.00001.0000 1.0000 1.0000 1.0000 0.0000
X-MADAM-RAG 1.0000 0.9667 0.9767 0.98041.00000.0200
Oracle extraction 1.00001.0000 1.0000 1.0000 1.0000 0.0000
5 Experimental Setup
AllLLMexperimentsuselocalQwen2.5-7B-Instructinferencewithdeterministic
decoding (temperature 0), bfloat16 weights, SDPA attention, and a maximum
of 512 new tokens.1Only one model is evaluated. We evaluate on the original
300-sample benchmark and the 100-sample naturalized stress subset; component
ablations are run on the original benchmark only.
We use five automatic metrics.Strict accuracymeasures normalized lex-
ical match between the returned answer and the reference.Conflict-aware
successrequires conflict recognition and reference-candidate preservation on
conflict samples, and reduces to strict correctness on non-conflict samples.Con-
flict F1evaluates expected versus predicted conflict.Conflict awarenessmea-
sures explicit conflict expression among expected-conflict cases.Unsupported-
candidate ratemeasures whether generated candidates fail to lexically match
any evidence-supported candidate in private metadata. None of these metrics
constitute semantic entailment, factual verification, or human judgments.
For original-benchmark comparisons we use 1,000 bootstrap resamples with
seed 13; paired sign tests are exploratory and uncorrected for multiple compar-
isons.
6 Results and Analysis
6.1 Original Controlled Benchmark
Among deployable LLM methods, X-MADAM-RAG performs best on the orig-
inal benchmark. It improves strict accuracy over evidence-normalized RAG by
0.0767 and lowers unsupported-candidate rate from 0.1067 to 0.0200 (95% boot-
strap interval for the strict-accuracy gap:[0.0367,0.1167]; exploratory sign-test
p= 0.0004). However, the rule-only extractor also achieves perfect performance,
showingthattheoriginalbenchmarkisstronglytemplate-regular.Originalbench-
markresultsshouldthereforebeinterpretedascontrolleddiagnosticperformance
rather than evidence of semantic extraction robustness.
1Python 3.10.20, PyTorch 2.6.0+cu124, Transformers 4.57.6, NVIDIA RTX 4090
(≈48GiB VRAM).

8 F. Author et al.
Table 3.Component ablations on the original benchmark.
Variant Coverage Strict Conf.-aware Conf. F1 Awareness Unsupported
X-MADAM without
repair1.0000 0.8033 0.6967 0.7089 0.6467 0.0933
X-MADAM without
det. grouping1.0000 0.8600 0.9433 0.9521 0.9933 0.0367
Full X-MADAM 1.00000.9667 0.9767 0.9804 1.0000 0.0200
Table 4.Results on the 100-sample naturalized stress subset.
Method Coverage Strict Conf.-aware Conf. F1 Awareness Unsupported
Rule-only extractor 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000
Naive 1.0000 0.6500 0.6300 0.8762 0.6200 0.1900
Evidence-normalized 1.00000.6800 0.66000.94950.64000.3000
X-MADAM-RAG 1.0000 0.3000 0.2600 0.1481 0.5800 0.0500
Oracle extraction 1.00001.0000 1.0000 1.0000 1.0000 0.0000
6.2 Component Ablations
Both components contribute within the original X-MADAM pipeline. Disabling
repair produces the largest drop, reducing strict accuracy from 0.9667 to 0.8033.
Replacing deterministic grouping with model conflict judgment also degrades
performance, especially strict accuracy and unsupported-candidate rate.
6.3 Naturalized Stress Test
Table 4 reports results on the naturalized-100 stress subset, and Figure 3 com-
pares strict accuracy before and after naturalization. Naturalization removes the
explicit rule cues: the rule-only extractor drops from 1.0000 strict accuracy on
the original benchmark to 0.0000 on the naturalized subset. This confirms that
its original success depends on surface templates rather than semantic extrac-
tion.
X-MADAM-RAG also degrades substantially, from 0.9667 strict accuracy
on the original benchmark to 0.3000 on the naturalized subset. By contrast,
the naive and evidence-normalized single-call baselines retain 0.6500 and 0.6800
strict accuracy, respectively. This shows that the current X-MADAM extractor
is sensitive to naturalized phrasing and should not be described as robust in this
setting.
The oracle diagnostic remains at 1.0000 in both settings. Since the oracle
replaces only document-level extraction with privileged supported-answer meta-
data while retaining the downstream grouping and aggregation stages, this pat-
tern identifies extraction as the principal bottleneck. The naturalized stress test
is therefore a negative robustness result, but also a useful diagnostic result: it
shows that improving document-level extraction is more urgent than adding
further aggregation complexity.

Diagnosing Chinese-English Evidence Conflict in RAG 9
Fig. 3.Strict accuracy before and after naturalization. Rule-only extraction drops
from 1.0000 to 0.0000 after explicit answer templates are removed. X-MADAM-RAG
also drops from 0.9667 to 0.3000, while the oracle remains at 1.0000, indicating that
document-level extraction is the main bottleneck under naturalized phrasing.
Additional breakdowns by answer type, conflict direction, and C6 optional-
noise subgroups are omitted from the main paper for space; they are consistent
with the main findings.
7 Discussion and Limitations
On the original benchmark, X-MADAM-RAG is the strongest deployable LLM
method and its internal components are demonstrably useful: repair recovers
missing candidates from visible evidence and deterministic grouping improves
auditability and performance.
At the same time, the perfect rule-only result shows that the original bench-
mark is highly template-regular, limiting what can be concluded from original
benchmark performance. X-RAMDocs-ZHEN v0.1 is best understood as a con-
trolled diagnostic tool for evidence-conflict behavior, not a general robustness
benchmark.
The naturalized stress test sharpens this conclusion. X-MADAM-RAG de-
grades below single-call baselines while the oracle remains perfect, indicating
that the extraction stage—not grouping or aggregation—is the central bottle-
neck. Conflict-aware aggregation alone is insufficient: a system must first extract
candidate answers robustly from varied evidence forms. Future work should fo-
cusondocument-levelextractionunderparaphrase,richeranswernormalization,
and semantic candidate verification.

10 F. Author et al.
This study has several limitations. X-RAMDocs-ZHEN uses controlled snip-
pets derived from RAMDocs fields rather than naturally retrieved multilingual
passages. The original benchmark is template-regular and cannot establish se-
mantic extraction ability. The naturalized stress test is still deterministic con-
trolled prose. All LLM experiments use one model, Qwen2.5-7B-Instruct; re-
sults may differ for larger or closed models. The benchmark covers only Chinese
and English. All metrics are automatic lexical diagnostics, not human semantic
judgments. No completed human audit is available. The oracle is privileged and
non-deployable. Our claims are limited to controlled Chinese-English evidence-
conflict diagnosis; we do not claim broad hallucination detection, robustness to
natural retrieval, or human-validated faithfulness.
8 Conclusion
We presented X-RAMDocs-ZHEN, a controlled leakage-aware Chinese-English
benchmark for diagnosing evidence conflict in RAG, and studied X-MADAM-
RAG,aninterpretablepipelinebasedonper-documentextraction,visible-evidence
repair, deterministic candidate grouping, and conflict-aware aggregation. On the
original controlled benchmark, X-MADAM-RAG outperforms single-call LLM
baselines and ablations confirm that repair and deterministic grouping con-
tribute. However, a rule-only extractor achieves perfect performance, revealing
strong template regularity. A naturalized stress test removes explicit rule cues:
X-MADAM-RAG degrades below single-call baselines while a privileged ora-
cle remains perfect, identifying extraction robustness as the central unresolved
problem. Future work should evaluate natural retrieval, complete human audit,
test additional models and languages, and develop more robust document-level
extraction and semantic candidate verification.
Acknowledgments.The authors thank the contributors of the RAMDocs dataset,
on which X-RAMDocs-ZHEN is based.
Disclosure of Interests.The authors have no competing interests to declare that
are relevant to the content of this article.
References
1. Cai, L., Kang, Y., Yu, C., Fu, Y., Zhang, H., Zhao, Y.: Bringing two worlds to-
gether: The convergence of large language models and knowledge graphs. In: 2024
3rd International Conference on Automation, Robotics and Computer Engineering
(ICARCE). pp. 207–216. IEEE (2024)
2. Cai, L., Yu, C., Kang, Y., Fu, Y., Zhang, H., Zhao, Y.: Practices, opportunities and
challenges in the fusion of knowledge graphs and large language models. Frontiers
in computer science7, 1590632 (2025)
3. Fu, Y., Kang, Y., Zhao, Y.: Reasoning in large language models: A survey (2025)

Diagnosing Chinese-English Evidence Conflict in RAG 11
4. Hu, X., Ru, D., Qiu, L., Guo, Q., Zhang, T., Xu, Y., Luo, Y., Liu, P., Zhang, Y.,
Zhang, Z.: Knowledge-centric hallucination detection. In: Proceedings of the 2024
Conference on Empirical Methods in Natural Language Processing. pp. 6953–6975
(2024)
5. Liu, W., Trenous, S., Ribeiro, L.F., Byrne, B., Hieber, F.: Xrag: Cross-lingual
retrieval-augmented generation. arXiv (2025)
6. Ming, Y., Purushwalkam, S., Pandit, S., Ke, Z., Nguyen, X.P., Xiong, C., Joty, S.:
Faitheval: Can your language model stay faithful to context, even if" the moon is
madeofmarshmallows".In:InternationalConferenceonLearningRepresentations.
vol. 2025, pp. 29430–29456 (2025)
7. Ranaldi, L., Haddow, B., Birch, A.: Multilingual retrieval-augmented generation
for knowledge-intensive task. arXiv preprint arXiv:2504.03616 (2025)
8. Ru, D., Qiu, L., Hu, X., Zhang, T., Shi, P., Chang, S., Jiayang, C., Wang, C., Sun,
S., Li, H., et al.: Ragchecker: A fine-grained framework for diagnosing retrieval-
augmented generation. vol. 37, pp. 21999–22027 (2024)
9. Vázquez, R., Mickus, T., Zosa, E., Vahtola, T., Tiedemann, J., Sinha, A., Segonne,
V., Sánchez-Vega, F., Raganato, A., Libovick` y, J., et al.: Semeval-2025 task 3:
Mu-shroom, the multilingual shared-task on hallucinations and related observable
overgeneration mistakes. In: Proceedings of the 19th International Workshop on
Semantic Evaluation (SemEval-2025). pp. 2472–2497 (2025)
10. Wang, H., Prasad, A., Stengel-Eskin, E., Bansal, M.: Retrieval-augmented gener-
ation with conflicting evidence. arXiv preprint arXiv:2504.13079 (2025)
11. Xu, R., Qi, Z., Guo, Z., Wang, C., Wang, H., Zhang, Y., Xu, W.: Knowledge
conflicts for llms: A survey. In: Proceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing. pp. 8541–8565 (2024)
12. Zhang, Q., Xiang, Z., Xiao, Y., Wang, L., Li, J., Wang, X., Su, J.: Faithfulrag:
Fact-level conflict modeling for context-faithful retrieval-augmented generation.
In: Proceedings of the 63rd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). pp. 21863–21882 (2025)
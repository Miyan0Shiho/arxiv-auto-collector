# De Jure: Iterative LLM Self-Refinement for Structured Extraction of Regulatory Rules

**Authors**: Keerat Guliani, Deepkamal Gill, David Landsman, Nima Eshraghi, Krishna Kumar, Lovedeep Gondara

**Published**: 2026-04-02 17:06:50

**PDF URL**: [https://arxiv.org/pdf/2604.02276v1](https://arxiv.org/pdf/2604.02276v1)

## Abstract
Regulatory documents encode legally binding obligations that LLM-based systems must respect. Yet converting dense, hierarchically structured legal text into machine-readable rules remains a costly, expert-intensive process. We present De Jure, a fully automated, domain-agnostic pipeline for extracting structured regulatory rules from raw documents, requiring no human annotation, domain-specific prompting, or annotated gold data. De Jure operates through four sequential stages: normalization of source documents into structured Markdown; LLM-driven semantic decomposition into structured rule units; multi-criteria LLM-as-a-judge evaluation across 19 dimensions spanning metadata, definitions, and rule semantics; and iterative repair of low-scoring extractions within a bounded regeneration budget, where upstream components are repaired before rule units are evaluated. We evaluate De Jure across four models on three regulatory corpora spanning finance, healthcare, and AI governance. On the finance domain, De Jure yields consistent and monotonic improvement in extraction quality, reaching peak performance within three judge-guided iterations. De Jure generalizes effectively to healthcare and AI governance, maintaining high performance across both open- and closed-source models. In a downstream compliance question-answering evaluation via RAG, responses grounded in De Jure extracted rules are preferred over prior work in 73.8% of cases at single-rule retrieval depth, rising to 84.0% under broader retrieval, confirming that extraction fidelity translates directly into downstream utility. These results demonstrate that explicit, interpretable evaluation criteria can substitute for human annotation in complex regulatory domains, offering a scalable and auditable path toward regulation-grounded LLM alignment.

## Full Text


<!-- PDF content starts -->

Preprint. Under review.
De Jure: Iterative LLM Self-Refinement for Structured Extraction of
Regulatory Rules
Keerat Guliani†, Deepkamal Gill†, David Landsman, Nima Eshraghi, Krishna Kumar & Lovedeep Gondara
†Equal contribution
The Vanguard Group, Inc.∗
{keerat_guliani, deepkamal_gill,david_landsman,nima_eshraghi,krishna_kumar,lovedeep_gondara}@vanguard.com
Abstract
Regulatory documents encode legally binding obligations that LLM-based systems must
respect. Yet converting dense, hierarchically structured legal text into machine-readable
rules remains a costly, expert-intensive process. We presentDe Jure, a fully automated,
domain-agnostic pipeline for extracting structured regulatory rules from raw documents,
requiring no human annotation, domain-specific prompting, or annotated gold data. De Jure
operates through four sequential stages: normalization of source documents into structured
Markdown; LLM-driven semantic decomposition into structured rule units; multi-criteria
LLM-as-a-judge evaluation across 19 dimensions spanning metadata, definitions, and rule
semantics; and iterative repair of low-scoring extractions within a bounded regeneration
budget, where upstream components are repaired before rule units are evaluated, ensuring
definitional context is maximally accurate before the most demanding decomposition stage.
We evaluate De Jure across four models on three regulatory corpora spanning finance,
healthcare, and AI governance. On the finance domain, De Jure yields consistent and
monotonic increase in extraction quality, reaching peak performance within at most three
judge-guided iterations. De Jure generalizes effectively to the structurally distinct healthcare
and the AI governance domains, maintaining similar high performance across all domains
and both open- and closed-source models. In a downstream evaluation, using compliance
question answering via RAG, responses grounded in De Jure extracted rules are preferred
by a judge LLM over prior work in 73.8% of cases at single-rule retrieval depth, rising to
84.0% under broader retrieval, confirming that extraction fidelity translates directly into
downstream utility. These results demonstrate that explicit, interpretable evaluation criteria
can substitute for human annotation in highly complex regulatory domains, offering a scalable
and auditable path toward regulation-grounded LLM alignment.
1 Introduction
The alignment of large language models (LLMs) with human values has become one of the central challenges
in modern AI research (Ouyang et al., 2022; Bai et al., 2022b). While most alignment work focuses on making
models helpful and harmless (Bai et al., 2022b; Christiano et al., 2017), the rapid deployment of LLMs in
high-stakes domains including finance (Wu et al., 2023; Yang et al., 2023), healthcare (Singhal et al., 2023),
and law (Chalkidis et al., 2020; Nay, 2023) demands a broader conception of alignment: one grounded not
only in subjective human preferences but in explicit, codified regulatory obligations. Regulatory documents
such as healthcare privacy rules (HIPAA(U.S. Department of Health and Human Services, 2003)), financial
conduct standards (SEC Advisers Act(U.S. Securities and Exchange Commission, 1940)), and responsible
AI frameworks (EU AI Act(European Union, 2024)) encode legally binding requirements that LLM-based
systems must respect. Yet transforming these dense, hierarchically structured texts into machine-readable
rules remains a largely manual, expert-intensive process, creating a critical bottleneck for compliance-aware
AI deployment at scale.
∗©2026 The Vanguard Group, Inc. All rights reserved.
This material is provided for informational purposes only and is not intended to be investment advice or a recommen-
dation to take any particular investment action.
1arXiv:2604.02276v1  [cs.AI]  2 Apr 2026

Preprint. Under review.
Input
PDF / HTML
regulatory
documents1Pre-processing
Convert to Markdown
Clean & normalise
Split by section
Index by ID
SHA-256 fingerprint2Rule Generation
Prompt + JSON schema
(domain-agnostic)
Section→struct.JSON
Non-actionable→null3Judgment
LLM-as-a-judge scoring
S1Metadata6 criteria
S2Definitions5 criteria
S3Rule units8 criteria4Repair
Avg score<90%:
re-generate (max 3)
Keep best generationOutput
High-fidelity
structured
rule set✓scores
avg score<90%, retries≤3
⟨/⟩Section Extraction
section_metacitation, title, effective_dates, notes
definitions[]term, text, scope, cross_refs
extracted_rules[]rule_id, label, type, targets, citations, score
statement→action, action_object, method, conditions,
constraints, exceptions, penalties, purpose, verbatimoutput validates
Figure 1: De Jure pipeline overview. Input documents are pre-processed into structured Markdown (Stage 1),
parsed into JSON rule units via a domain-agnostic LLM prompt (Stage 2), scored by an LLM judge across
19 criteria in three stages (Stage 3), and iteratively repaired until the per-stage average score reaches 90% or
the retry budget (max 3) is exhausted (Stage 4).
Constitutional AI (CAI) (Bai et al., 2022a) and its extensions (Lee et al., 2023; Sun et al., 2023) have demon-
strated that LLMs can generate and evaluatealignment principlesfor general helpfulness and harmlessness
from plain-language policy text, reducing reliance on costly human preference annotation. However, in high-
stakes domains such as finance, healthcare, and law, alignment is governed by explicit regulatory obligations
rather than general-purpose safety principles, which are often structurally complex and exception-laden.
Existing approaches either depend on hand-curated seed principles requiring significant domain expertise (Bai
et al., 2022a), or produce coarse-grained rules ill-suited for the structural complexity of legal and regulatory
corpora (Sun et al., 2023). More targeted work on LLM-driven legal rule extraction has shown promise.
Defeasible deontic logic formulae have been extracted from telecommunications regulations (Governatori
et al., 2016), and multi-stage pipelines have been applied to regulatory text (Sleimi et al., 2018). The work
most closely related to ours (Datla et al., 2025), extracts governance principles fromHIPAAand theEU
AI Actvia few-shot prompting with judge-and-repair steps benchmarked against a human-annotated gold
set. However, it requires costly expert annotation, captures a relatively flat rule representation without
decomposing the finer-grained semantic structure of regulatory obligations, and treats repair as a single,
undifferentiated correction step without exploiting the hierarchical dependencies between section metadata,
term definitions, and rule units.
We proposeDe Jure(DocumentExtraction withJudge-RefinedEvaluation) (Figure 1), a fully automated,
domain-agnostic pipeline that transforms raw regulatory documents into structured, machine-readable rule
sets with no human annotation and no domain-specific prompts. De Jure pre-processes input documents into
normalized Markdown, prompts an LLM to extract typed rule units conforming to a schema that decomposes
each rule into a rich set of semantic fields capturing actions, conditions, constraints, exceptions, penalties,
and verbatim source spans, and applies a multi-criteria LLM-as-a-judge (Zheng et al., 2023) to score each
extraction across 19 dimensions (summarized in Appendix B). A key design principle ishierarchical decoupling,
where the components that rules depend on are verified and repaired before rule units are evaluated, ensuring
that rule-level repair always operates on reliable context. Stages that fall below a quality threshold of 90%
are repaired through targeted regeneration, and the highest-scoring output is retained within a configurable
user-fined retry budget (defaulting to three attempts). Our ablation studies confirm that this budget is both
necessary and sufficient. Our main contributions are as follows:
•De Jure pipeline and extraction schema.A four-stage, domain-agnostic pipeline for end-to-end
regulatory rule extraction requiring no annotated data, domain-specific prompts, or logical formalisms,
anchored by a structured schema that decomposes each rule unit into a rich set of semantic fields
generalizable across regulatory corpora without modification (Section 3).
•Multi-stage LLM judge with hierarchical repair.A 19-criterion evaluation framework with three
judges applied in hierarchical dependency order, so that each stage repairs on previously-verified
context before the next begins. A targeted regeneration mechanism retains the highest-scoring output
within a bounded compute budget (Section 3.3).
2

Preprint. Under review.
•Cross-domain generalization.With no changes to prompts, schema, or model configuration, De Jure
generalizes across three structurally distinct regulatory domains:HIPAA(healthcare), theSEC
Advisers Act(finance), and theEU AI Act(AI governance), achieving total average scores above
94% (4.70/5.00) across all domain and model combinations. This demonstrates that the pipeline
transfers to new regulatory domains without any re-engineering (Section 4.3).
•Downstream and ablation evaluation.A RAG-based question answering comparison against Datla
et al. (2025) onHIPAAshows that De Jure-grounded responses are preferred in 73.8% of cases at
single-rule retrieval, rising to 84.0% at ten-rule retrieval, confirming that extraction quality gains
translate directly to downstream utility. Targeted ablation studies validate the contribution of each
key design choice to overall pipeline quality (Section 4.4, Appendix A).
The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 describes the
De Jure pipeline, covering preprocessing, extraction schema, judge design, and iterative repair. Section 4
presents our evaluation, including extraction quality, cross-domain generalization, and downstream RAG
assessment. Finally, the concluding remarks are given in Section 6. Supplementary material is provided in
the appendix, including ablation studies (Appendix A), implementation and algorithmic details (Appen-
dices B and D), a qualitative end-to-end pipeline case study (Appendix C), extended experimental results
(Appendices E and F), and complete prompts and extraction schemas (Appendices G and H).
2 Related Work
De Jure sits at the intersection of regulatory NLP, structured information extraction, and LLM-based
self-refinement. We review the work most relevant to these three areas and clarify where our approach
diverges from prior methods.
2.1 Rule Extraction from Regulatory and Policy Text
Converting dense regulatory text into structured, machine-readable representations is a long-standing problem
in legal informatics, healthcare, and finance. Early work applied syntactic parsing and logic-based formalisms
to identify normative statements and support automated compliance checking (Dragoni et al., 2016; Wyner
& Peters, 2011; Governatori, 2024), while information extraction approaches have been applied to financial
regulations (Lam et al., 2021) and clinical guidelines (Weng et al., 2010). More recent work leverages LLMs
for rule extraction across domains, including traffic regulations (Zin et al., 2025), clinical decision support (He
et al., 2024; Tang et al., 2026), and legal contract understanding (Koreeda & Manning, 2021; Hendrycks et al.,
2021). Despite this progress, existing approaches either require labeled data, operate within a single domain,
or produce extractions too coarse to capture the conditional and exception-laden structure of regulatory
obligations. De Jure addresses all three limitations simultaneously.
2.2 Deontic Logic and Defeasible Rule Formalisms
A dominant paradigm for making extracted rules machine-actionable isdeontic logic(von Wright, 1951),
extended bydefeasible deontic logic(DDL) to support reasoning under exceptions and conflicts (Governatori
& Rotolo, 2023). At scale, DDL extraction has been applied to telecommunications regulations using carefully
designed prompts and fine-tuning (Horner et al., 2025). While powerful, DDL-based extraction is inherently
domain-specific: the target formalism must be defined in advance and does not transfer to new regulatory
domains without substantial re-engineering.
2.3 LLM pipelines for structured principle extraction and iterative refinement
CLAUDETTE (Lippi et al., 2019) and OPP-115 (Wilson et al., 2016) established that structured, criterion-
level analysis of policy text is both feasible and practically valuable. The most closely related work, of
Datla et al. (2025) extracts governance principles from texts includingHIPAAand theEU AI Actvia
chunking, clause mining, and structured LLM extraction with judge-and-repair steps evaluated against a
3

Preprint. Under review.
human-annotated gold set. On the refinement side, iterative self-refinement (Madaan et al., 2023), verbal
reinforcement (Shinn et al., 2023), and Constitutional AI (Bai et al., 2022a) have demonstrated that LLM-
generated critiques can systematically improve output quality without annotated data, and LLM-as-a-judge
evaluation with decomposed criteria has been shown to correlate more closely with human judgment than
holistic scoring (Zheng et al., 2023; Liu et al., 2023b).
Positioning De Jure.Prior approaches address complementary aspects of the problem but differ from De Jure
in three key respects. DDL-based methods (Horner et al., 2025) require a pre-specified logical formalism
and do not transfer across domains. The method in Datla et al. (2025) depends on costly human annotation
and applies repair as a single flat step, without accounting for structural dependencies in the extraction.
General-purpose refinement methods (Madaan et al., 2023; Shinn et al., 2023) provide no structured quality
criteria and are not designed for regulatory text. De Jure addresses all three limitations. It requires no formal
specification and no human annotation. Furthermore, repair steps are applied hierarchically, so that upstream
components are corrected before rule units are evaluated. Finally, criterion-driven, field-level judgment at
each stage enables scalable quality control across heterogeneous regulatory corpora with minimal domain
expertise.
3 De Jure
We presentDe Jure(DocumentExtraction withJudge-RefinedEvaluation), an automated, domain-agnostic
pipeline for transforming raw regulatory documents into structured, machine-readable rule sets (Figure 1).
Unlike prior works that rely on human-annotated gold sets (Sleimi et al., 2018) or domain-specific logical
formalisms (Governatori et al., 2016), De Jure requires no labelled data and no domain expertise beyond the
source document itself. A full procedural specification of the pipeline is provided in Appendix D (Algorithm 1),
and a concrete end-to-end example of De Jure in action is given in Appendix C (Figure 3).
Thepipelineconsistsoffourstages: (1)documentpre-processingintonormalized, section-segmentedMarkdown;
(2) LLM-drivenrule generationinto a typed JSON schema capturing a rich set of semantic fields per rule unit;
(3)multi-criteria judgmentby an LLM judge across 19 dimensions (summarized in Appendix B) organized in
three sequential validation stages; and (4)selective repair by regeneration, where any stage scoring below a
quality threshold is retried and the highest-scoring output is retained. Two core principles drive the De Jure
pipeline design. First,decoupling of extraction from verification: by separating what is generated from how it
is evaluated, the same judgment-repair loop applies uniformly across document types, regulatory regimes,
and LLMs without modifying the core pipeline. Second,hierarchical repair ordering: the three judgment
stages are applied in dependency order, so that upstream components are verified and repaired before rule
units are evaluated, ensuring that rule-level repair always operates on reliable context.
3.1 Pre-processing
The pre-processing stage (Figure 1, Stage 1) converts heterogeneous regulatory source formats into structured
section–content pairs suitable for downstream LLM processing. Each input document is converted to
Markdown using Docling (Auer et al., 2024), which preserves section boundaries, list structure, and table
formatting, then cleaned to remove formatting artifacts and extraneous whitespace. The clean text is
segmented by splitting on regulatory section delimiters (e.g., “§”, “Article”, “Rule”), yielding an ordered
setSof section–content pairs, each indexed by its regulatory identifier, assigned a SHA-256 fingerprint
for traceability, and stored alongside standardized metadata fields (title, version, effective dates). This
deterministic representation ensures every downstream extraction traces back to an exact source span, a hard
requirement for regulatory auditability.
3.2 Rule Generation
Each section s∈Sis independently submitted to the rule generation stage (Figure 1, Stage 2), where an
LLM is prompted to produce a structured JSON extraction conforming to our Section Extraction schema
(Figure 1). The schema decomposes each section into three typed components: (i)section metadata(citation,
4

Preprint. Under review.
title, effective dates, notes); (ii)definitions(term, text, scope, cross-references); and (iii)rule units, each
carrying an identifier, rule type, summary label, citation, and a nine-field statement decomposition: action,
action object, method, conditions, constraints, exceptions, penalties, purpose, and verbatim source span. This
fine-grained decomposition enables downstream systems: constitutional AI frameworks (Bai et al., 2022a) or
compliance verification engines to query and enforce individual semantic components rather than treating
rules as monolithic strings. The generation prompt is schema-driven with no domain-specific examples or seed
rules, and returns nullfor non-actionable sections, suppressing non-normative passages such as preambles
and cross-reference tables to prevent rule set inflation. Full prompt and schema are provided in Appendix G
and Appendix H, respectively.
3.3 Multi-Criteria Judgment
Since De Jure operates in anunsupervisedsetting with no reference annotations, quality control cannot rely
on surface-form matching against a gold standard. Instead, we adopt an LLM-as-a-judge framework (Zheng
et al., 2023; Liu et al., 2023a; Zhong et al., 2022) organized intothree sequential validation stagesmirroring
the schema hierarchy: section metadata is judged and repaired first, followed by definitions, then rule units.
This ordering ensures each stage receives the benefit of all prior corrections. By the time rule units are
evaluated, both the section metadata and the definitional vocabulary have already been refined, maximizing
the contextual accuracy available to the rule repair stage and increasing the likelihood of producing high-fidelity
rule decompositions.
Stage 1: Metadata validation (6 criteria).We validate section-level metadata including headings, effective
dates, citation strings, and optional contextual fields across six criteria: completeness, fidelity to source,
non-hallucination, title quality, citation and date precision, and optional field population. Accurate metadata
is foundational: it anchors every downstream extraction to a traceable regulatory provision, and errors such
as incorrect effective dates or misattributed citations propagate silently into all derived rule units.
Stage 2: Definition validation (5 criteria).We validate all extracted definitions, covering both affirmative
(“X means Y”) and exclusionary (“X does not include Y”) forms. Where a definitional span also encodes an
operative rule, it must appear in both the definitions and rule-unit components and is evaluated independently
in each. Criteria include completeness, source fidelity, non-hallucination, precision and formatting, and term
quality. Non-hallucination is particularly critical here: a fabricated or paraphrased definition silently corrupts
the interpretation of every rule referencing that term, making this the highest-leverage point for factual
grounding in the pipeline.
Stage 3: Rule-unit validation (8 criteria).We validate each rule unit at the field level across criteria spanning
the full statement decomposition: completeness, label conciseness, rule-type classification accuracy, fidelity
to source, neutrality, target consistency, actionability, and non-hallucination. Several criteria warrant brief
motivation.Rule-type accuracyis essential because misclassifying an obligation as a permission silently inverts
compliance semantics.Neutralityprevents the model from injecting interpretive framing that would bias
downstream enforcement.Actionabilityensures each rule is expressed in a form that a compliance system
can operationalize.Label concisenessmatters because summary labels serve as retrieval keys in downstream
question answering RAG systems, where verbose or ambiguous labels directly degrade retrieval precision.
This is the most demanding stage: a single source section may yield multiple rule units, each required to
satisfy all criteria independently.
Each criterion is scored on a0–5scale by the judge LLM, which also produces a natural-language justification
for each score. The full set of criteria is summarized in Table 8, Appendix B.
3.4 Selective Repair by Regeneration
Low judgment scores indicate specific, localizable defects such as a misclassified rule type, an incomplete
label, a hallucinated condition rather than wholesale generation failure. De Jure exploits this through a
selective repairmechanism: each stage is repaired independently if and only if its average score falls below
θ=0.90. The LLM is re-prompted with the original section text, the current extraction, per-criterion scores,
and the judge’s natural-language critiques, and asked to correct only the deficient fields. This repeats for up
5

Preprint. Under review.
tor=3attempts per stage, retaining the best-scoring output across all attempts, guaranteeing monotonically
non-decreasing quality.
This design has three practical consequences.First, repair cost is bounded, with at most radditional LLM
calls per stage per section.Second, structured scores and critiques make the repair signal substantially richer
than a generic re-prompt, empirically producing targeted field-level corrections rather than wholesale rewrites.
Third, retaining the best rather than the final generation provides a soft safety net: when re-generation
degrades quality, the pipeline falls back gracefully to the best previously seen output.
4 Experiments and Results
We evaluate De Jure across four backbone extraction and repair models with a separate fixed judge model,
on three regulatory corpora, assessing extraction quality, cross-domain generalization, downstream utility,
and core design decisions through ablation studies.
Pipeline Configuration.Extraction and judgment are performed by strictly separate models. Extraction
is performed by the backbone model under evaluation while judgment is performed by a fixed model,
Compass (Cohere, 2025), designed for structured, criterion-aligned evaluation of long-form outputs. Three
judges operate sequentially, each instantiated from Compass with stage-specific criteria targeting the failure
modes of its semantic layer:Judge 1evaluates section metadata quality (6 criteria),Judge 2assesses
extracted definitions (5 criteria), andJudge 3evaluates the correctness and completeness of extracted rules
(8 criteria). Each produces per-criterion scores and natural-language critiques. If the average normalized
score falls below θ=90%, the backbone model is re-prompted with the critique, the prior extraction, and the
original input, for at most r=3attempts per judge, retaining the highest-scoring output across all attempts.
Inference Settings.All backbone models use identical hyperparameters: temperature τ=0.1, maximum
output tokensT max=4096, and nucleus samplingp=0.95, encouraging deterministic, faithful extraction.
4.1 Datasets
We evaluate on three regulatory corpora spanning distinct high-stakes domains (Table 1), selected to probe
generalization across diverse legal styles and rule structures rather than in-domain memorization.
SEC Investment Advisers Act.TheInvestment Advisers Act(U.S. Securities and Exchange Commission,
1940) governs investment adviser conduct in U.S. financial markets. Its densely nested conditionals and
extensive cross-references make it a demanding primary benchmark.
EU Artificial Intelligence Act.TheRegulation (EU) 2024/1689(European Union, 2024) is the world’s
first comprehensive AI governance framework. Its mix of binding technical obligations, broad governance
principles, and non-binding recitals presents a structurally distinct extraction challenge.
HIPAA Privacy Rule.TheHIPAA Privacy Rule(U.S. Department of Health and Human Services, 2003)
governs the use and disclosure of protected health information in the United States. Its exception-laden
permission structures, where broad prohibitions are qualified by narrow condition-dependent exemptions, test
a pipeline’s ability to faithfully decompose rule scope.
The three corpora span a natural complexity gradient:SECprovisions are rigidly alphanumerically indexed
with well-delimited boundaries;HIPAAorganizes mandates around exception-laden permission hierarchies
with less consistent demarcation; and theEU AI Actemploys discursive, principle-based prose that
interleaves binding obligations with non-binding recitals. Ordered from most to least structurally regular,
they constitute ana prioridifficulty ranking for extraction.
6

Preprint. Under review.
Dataset Domain Jurisdiction Sections Rule Density
SEC Advisers Act Financial securities United States∼50 High
EU AI Act AI governance European Union 113 Medium–High
HIPAA Privacy Rule Healthcare privacy United States∼30 High
Table 1: Summary of evaluation datasets.
4.2 Experiment 1: Extraction Quality
Experimental Setup.We evaluate on the SEC Investment Advisers Act corpus using four backbone models
spanning open- and closed-source families:Llama-3.1-8B-Instruct(Dubey et al., 2024) andQwen3-VL-8B-
Instruct(Qwen Team, 2025) as cost-efficient, controllable open-source options, andClaude-3.5-Sonnet(An-
thropic, 2024) andGPT-5-mini(OpenAI, 2025) as high-capability closed-source systems. This selection
enables a controlled comparison across model families and access paradigms under identical pipeline conditions.
Model J1 (Metadata) J2 (Definitions) J3 (Per-Rule) Overall
Llama-3.1-8B 4.934.75 4.53 4.74
Qwen3-VL-8B 4.934.874.64 4.81
Claude-3.5-Sonnet4.994.83 4.67 4.83
GPT-5-mini4.994.834.75 4.85
Table 2: Averaged judge (J) scores on the SEC Investment Advisers Act corpus (scale 1–5; higher is better).
Per-criterion breakdowns for all three judges are provided in Appendix E (Tables 9–11).
Results.Table 2 reports the averaged score across all judge criteria, for each model and judge stage. Two
observations stand out. First, performance degrades monotonically from metadata ( ≈4.96) to definitions
(≈4.82) to per-rule quality ( ≈4.65), directly reflecting increasing task complexity: section-level metadata is
structurally salient and consistently recoverable, whereas decomposing fine-grained rule components such
as conditions, exceptions, and penalties demands substantially deeper semantic parsing. Second, model
divergence is most pronounced at Judge 3 (J3), where GPT-5-mini leads (Avg =4.75) over Llama-3.1-8B
(4.53), driven by superior Accuracy (4.95) and Fidelity to Source (4.74) on individual rule decomposition
(Appendix E). Notably, Qwen3-VL-8B achieves the highest J2 average (4.87), outperforming both closed-source
models on definition extraction. The narrow overall gap between the best open-source model (Qwen3-VL-8B,
4.81) and the best closed-source model (GPT-5-mini, 4.85) suggests that capable open-source models, when
paired with structured extraction and iterative refinement, can approach proprietary model performance, a
practically significant finding for privacy-sensitive regulatory deployments where reliance on external APIs is
undesirable. Further, the per-criterion breakdowns in Appendix E reveal that Non-Hallucination is uniformly
perfect (5.00) across all models and all three judges, confirming that schema-constrained extraction eliminates
factual fabrication as a failure mode regardless of model family.
4.3 Experiment 2: Generalization Across Regulatory Domains
Experimental Setup.Does De Jure generalize,without any modification, across structurally diverse regulatory
domains? We evaluate on all three corpora from Section 4.1, usingGPT-5-mini(best closed-source) and
Qwen3-VL-8B-Instruct(best open-source) from Experiment 1, with all settings unchanged.
Results.Table 3 shows a consistent result: overall scores remain above 4.70 across every domain and model
combination, the range spanning only 0.14 points in total. Sustaining near-ceiling performance across three
structurally distinct regulatory domains, with no domain-specific adaptation, presents strong evidence of
broad domain generalizability. The monotonic decline from SEC ( ≈4.84) to HIPAA ( ≈4.76) to EU AI
Act (≈4.71) tracks an intuitive ordering of structural regularity: SEC text is rigidly indexed, HIPAA is
more loosely organized, and the EU AI Act is the most discursive. This monotonic decline mirrors thea
prioristructural complexity ordering established in Section 4.1, which was derived independently of any
7

Preprint. Under review.
Dataset Model J1 (Metadata) J2 (Definitions) J3 (Per-Rule) Overall
SECGPT-5-mini4.994.834.75 4.85
Qwen3-VL-8B4.934.874.64 4.82
HIPAAGPT-5-mini4.934.614.75 4.76
Qwen3-VL-8B4.904.714.66 4.76
EU AI ActGPT-5-mini4.724.694.714.71
Qwen3-VL-8B4.654.824.654.71
Table 3: Generalization across three regulatory corpora (scale 1–5; higher is better). De Jure is applied
without domain-specific modification. Per-criterion breakdown details are provided in Appx. F (Tables 12–14).
experimental result. A permissive judge would yield uniformly near-ceiling scores across all corpora; the
systematic score variation is consistent only with a judge that is sensitive to intrinsic differences in extraction
difficulty. This finding is further supported by the qualitative evidence in Appendix C, where the judge assigns
a failing normalized average score of 0.55 to a semantically deficient extraction, with targeted per-criterion
feedback that directly identifies the defective fields. Upon repair, the same judge awards a passing normalized
average score of 0.90 to the corrected output, while leaving scores unchanged on fields that were already
correct. This behavior confirms that the judge is discriminative rather than permissive. It penalizes specific
deficiencies, recognizes genuine improvement, and does not reward outputs indiscriminately.
Two patterns at the averaged and at the per-criterion level (Table 3 and Appendix F) merit attention. First,
J2 is the most variable judge and the primary driver of the cross-domain gap: GPT-5-mini’s J2 drops from
4.83 (SEC) to 4.61 (HIPAA), consistent with the intuition that linking rules to definitional context is harder
when section boundaries are loosely organized rather than rigidly indexed. Model-level variance is most
pronounced at J2, with scores diverging by up to 0.21 points across domains, suggesting that definitional
enrichment is sensitive to corpus structure and model characteristics. Second, J3 remains the hardest stage
across all three corpora, consistent with Experiment 1, confirming that precise recovery of conditional triggers,
quantitative thresholds, and nested exception structures is a universal bottleneck independent of jurisdiction
or legal style, and identifying fine-grained rule decomposition as the primary target for future work.
4.4 Experiment 3: Downstream Evaluation, Compliance QA via RAG
Intrinsic extraction quality does not fully capture practical utility. We therefore evaluate extracted rules as
the knowledge base of a RAG system tasked with answering HIPAA compliance questions, providing a direct
task-grounded comparison against Datla et al. (2025).
Experimental Setup.We restrict evaluation to HIPAA sections covered by the publicly released extractions
of Datla et al. (2025),1ensuring any performance gap is attributable solely to rule quality rather than
coverage. Our rules are extracted using GPT-5-mini. We prompt Qwen3-VL-8b-Instruct to generate 100
evaluation questions from the same HIPAA sections at temperature τ=0.8(Further details in Appendices G.4
and I), yielding a lexically and semantically diverse set in which every question has a verifiable, source-
grounded answer. Two independent vector databases are constructed, one per rule set, both encoded with
all-mpnet-base-v2 (Reimers & Gurevych, 2019). Answers are generated using Claude 3.5 Sonnet under
retrieval depths k∈{ 1, 5, 10}, spanning precise single-rule to broad multi-rule regimes. A pairwise LLM
judge (τ=0.1) evaluates each answer pair across six criteria: Completeness, Factual Grounding, Handling
Ambiguity, Practical Actionability, Regulatory Precision, and Overall Preference (Appx. B.1 provides further
details). To eliminate positional bias, each pair is judged twice with swapped ordering and win rates are
averaged.
Results.Table 4 shows De Jure outperforming Datla et al. (2025) across every criterion and retrieval depth,
with a widening margin as kgrows. At k=1, where aggregation effects are absent, the 73.75% win rate,
1https://github.com/gautamvarmadatla/Policy-Tests-P2T-for-operationalizing-AI-governance/blob/
90059a7d2b59d705a80d212b3a1a8adfb30ef58b/Annotator%20Data/Annotator%20Docs/out/HIPAA/HIPAA_removed.extracted.jsonl
8

Preprint. Under review.
Criterionk=1k=5k=10
Completeness 78.00 80.50 83.50
Factual Grounding 80.50 76.00 85.50
Handling Ambiguity 53.50 66.50 84.00
Practical Actionability 78.00 80.50 84.00
Regulatory Precision 74.50 80.50 83.50
Overall Preference 78.00 80.50 83.50
Aggregated 73.75 77.42 84.00
Table 4: Pairwise win rates (%) of De Jure against Datla et al. (2025) on the HIPAA downstream QA task
across three retrieval depths.
23.75 points above parity, reflects a fundamental representational advantage: our rules preserve conditional
logic, scope qualifiers, and exception clauses that the baseline’s flatter representations omit, with factual
grounding already pronounced at 80.50%. The most diagnostic trajectory is handling ambiguity, which begins
near parity at k=1(53.50%) and surges to 84.00% at k=10(+30.50 pts). Ambiguous queries are inherently
multi-provision and cannot be resolved by any single rule; that our advantage materializes precisely as kgrows
confirms that our structured decomposition produces rules that integrate compositionally across provisions,
while the baseline exhibits diminishing returns consistent with redundant, less differentiated extractions. The
monotonically increasing aggregate win rate from 73.75% to 77.42% to 84.00% further confirms that De Jure’s
rules are mutually complementary, each additional retrieved rule contributes distinct, non-redundant context,
and that extraction fidelity translates directly into downstream utility.
5 Ablation Studies
We conduct four targeted ablations on core design decisions (full results in Appendix A). First, extraction
quality improves monotonically with acceptance threshold θ, with the largest gains concentrated in Step 2
(definitions, +0.30 points, that is 6.0%, from θ= 0.6 to 0.9), confirming θ= 0.9 as the strongest configuration
within the evaluated range. Second, the retry budget exhibits a striking non-linearity: a single retry yields
negligible improvement, while the qualitative shift occurs at r= 2, where Step 2 recovers 1.25 points (25.0%)
as the candidate pool becomes large enough for the best-of- rselection mechanism to escape low-quality basins;
gains saturate beyond r= 2, and we adopt r= 3 as a conservative default. Third, our section-aware chunking
strategy outperforms that of Datla et al. (2025) by +0.16 points (3.2%) overall, with the benefit concentrated
entirely in early pipeline stages (+0.33 points, i.e., 6.6%, at Step 1), confirming that downstream refinement
cannot compensate for incoherent inputs. Fourth, conditional on an adequate retry budget, the choice of
regeneration trigger (average-score vs. per-criterion) has no measurable effect on final quality, establishing
the retry budget as the dominant control variable and trigger granularity as second-order.
6 Conclusion
We presented De Jure, a fully automated, domain-agnostic pipeline that converts raw regulatory documents
into structured, machine-readable rule sets by interleaving typed semantic extraction with a hierarchically
ordered multi-stage LLM judge and an iterative repair mechanism. Metadata and definitions are corrected
before rule units are evaluated, ensuring that each downstream stage operates on the best available upstream
context and that errors do not silently propagate into rule decompositions. De Jure achieves strong extraction
quality on financial securities regulation and generalizes without modification to healthcare privacy and AI
governance, maintaining consistently high performance across all model families and access paradigms. A
downstream RAG-based evaluation demonstrates that De Jure-grounded responses are strongly preferred
over those from a strong prior approach, with the margin widening as retrieval depth increases, confirming
that extraction fidelity translates directly into downstream utility. Ablation studies further validate the
core design decisions: the retry budget is the dominant quality lever, and input chunking quality directly
conditions extraction fidelity in early pipeline stages. Together, these results demonstrate that explicit,
interpretable evaluation criteria can substitute for human annotation in annotation-scarce regulatory settings,
and that iterative judge-guided refinement offers a scalable and auditable path toward regulation-grounded
LLM alignment.
9

Preprint. Under review.
References
Anthropic. Claude 3.5 sonnet.https://www.anthropic.com/news/claude-3-5-sonnet, 2024.
Christoph Auer, Maksym Lysak, Ahmed Nassar, Michele Dolfi, Nikolaos Unski, Kasper Dinkla, Luke
Zhong-Hua, and Peter Staar. Docling technical report.arXiv preprint arXiv:2408.09869, 2024.
Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain,
Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Constitutional AI: Harmlessness from AI feedback.
arXiv preprint arXiv:2212.08073, 2022a.
Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain,
Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with
reinforcement learning from human feedback.arXiv preprint arXiv:2204.05862, 2022b.
Ilias Chalkidis, Manos Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos.
LEGAL-BERT: The muppets straight out of law school. InFindings of the Association for Computational
Linguistics (EMNLP), 2020.
Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement
learning from human preferences. InAdvances in Neural Information Processing Systems (NeurIPS), 2017.
Cohere. Compass: A structured evaluation model.https://cohere.com/compass, 2025. Accessed: 2026.
Gautam Varma Datla, Anudeep Vurity, Tejaswani Dash, Tazeem Ahmad, Mohd Adnan, and Saima Rafi.
Executable governance for ai: Translating policies into rules using llms.arXiv preprint arXiv:2512.04408,
2025.
Mauro Dragoni, Serena Villata, Williams Rizzi, and Guido Governatori. Combining nlp approaches for rule
extraction from legal documents. In1st Workshop on MIning and REasoning with Legal texts (MIREL
2016), 2016.
Abhimanyu Dubey et al. The Llama 3 herd of models.arXiv preprint arXiv:2407.21783, 2024.
European Union. Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial
Intelligence Act). Regulation L 2024/1689, Official Journal of the European Union, July 2024. Published
12 July 2024.
Guido Governatori. An asp implementation of defeasible deontic logic.KI-Künstliche Intelligenz, 38(1):79–88,
2024.
Guido Governatori and Antonino Rotolo. Deontic ambiguities in legal reasoning. InProceedings of the
Nineteenth International Conference on Artificial Intelligence and Law, pp. 91–100, 2023.
Guido Governatori, Mustafa Hashmi, Ho-Pun Lam, Serena Villata, and Monica Palmirani. Semantic business
process regulatory compliance checking using LegalRuleML. InInternational Conference on Advanced
Information Systems Engineering (CAiSE), 2016.
Yuxin He, Buzhou Tang, and Xiaoling Wang. Generative models for automatic medical decision rule extraction
from text. InProceedings of the 2024 Conference on Empirical Methods in Natural Language Processing,
pp. 7034–7048, 2024.
Dan Hendrycks, Collin Burns, Anya Chen, and Spencer Ball. CUAD: An expert-annotated NLP dataset for
legal contract review. InAdvances in Neural Information Processing Systems (NeurIPS), 2021.
Elias Horner, Cristinel Mateis, Guido Governatori, and Agata Ciabattoni. Toward robust legal text formal-
ization into defeasible deontic logic using llms.arXiv preprint arXiv:2506.08899, 2025.
Yuta Koreeda and Christopher D. Manning. ContractNLI: A dataset for document-level natural language
inference for contracts. InFindings of the Association for Computational Linguistics (EMNLP), 2021.
Tin Wai Lam et al. Extracting structured information from financial regulations using NLP. InProceedings
of the Natural Legal Language Processing Workshop, 2021.
10

Preprint. Under review.
Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune,
and Abhinav Rastogi. RLAIF: Scaling reinforcement learning from human feedback with AI feedback.
arXiv preprint arXiv:2309.00267, 2023.
Marco Lippi, Przemysław Pałka, Giuseppe Contissa, Francesca Lagioia, Hans-Wolfgang Micklitz, Giovanni
Sartor, and Paolo Torroni. CLAUDETTE: An automated detector of potentially unfair clauses in online
terms of service. volume 27, pp. 117–139, 2019.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-eval: NLG evaluation
usinggpt-4withbetterhumanalignment. InHoudaBouamor, JuanPino, andKalikaBali(eds.),Proceedings
of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 2511–2522, Singapore,
December 2023a. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.153. URL
https://aclanthology.org/2023.emnlp-main.153/.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-Eval: NLG evaluation
using GPT-4 with better human alignment. InProceedings of the Conference on Empirical Methods in
Natural Language Processing (EMNLP), 2023b.
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha
Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. In
Advances in Neural Information Processing Systems (NeurIPS), 2023.
John J. Nay. Law informs code: A legal informatics approach to aligning artificial intelligence with humans.
Northwestern Journal of Technology and Intellectual Property, 2023.
OpenAI. GPT-5mini. https://openai.com/index/gpt-5-mini-advancing-cost-efficient-intelligence/ ,
2025.
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with
human feedback. InAdvances in Neural Information Processing Systems (NeurIPS), 2022.
Qwen Team. Qwen3 technical report.arXiv preprint, 2025.https://huggingface.co/Qwen.
Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In
Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for
Computational Linguistics, 11 2019. URLhttps://arxiv.org/abs/1908.10084.
Noah Shinn, Federico Cassano, Beck Labash, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao.
Reflexion: Language agents with verbal reinforcement learning. InAdvances in Neural Information
Processing Systems (NeurIPS), 2023.
Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales,
Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. Large language models encode clinical knowledge.
Nature, 620:172–180, 2023.
Amin Sleimi, Nicolas Sannier, Mehrdad Sabetzadeh, Lionel Briand, and John Dann. Automated extraction of
semantic legal metadatausing naturallanguage processing. InIEEE International Requirements Engineering
Conference (RE), 2018.
Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David Cox, Yiming Yang,
and Chuang Gan. SALMON: Self-alignment with principle-following reward models.arXiv preprint
arXiv:2310.05910, 2023.
Bingxin Tang, Jiajie Zhuang, Zhiang Wu, Hongru Lu, and Changjian Fang. From policy documents to audit
logic: A llms-based framework for extracting executable audit rules. 2026.
U.S. Department of Health and Human Services. Standards for privacy of individually identifiable health
information (HIPAA privacy rule). Federal Regulation 45 C.F.R. Part 164 Subpart E, U.S. Department of
Health and Human Services, 2003. Privacy Rule under the Health Insurance Portability and Accountability
Act (HIPAA).
U.S. Securities and Exchange Commission. Investment advisers act of 1940. Federal Statute 15 U.S.C.
§§ 80b-1 et seq., U.S. Securities and Exchange Commission, 1940. URLhttps://www.sec.gov/.
11

Preprint. Under review.
Georg Henrik von Wright. Deontic logic.Mind, 60(237):1–15, 1951.
Chunhua Weng et al. Formal representation of eligibility criteria: A literature review. volume 43, pp. 451–467,
2010.
Shomir Wilson, Florian Schaub, Aswarth Abhilash Dara, Frederick Liu, Sushain Cherivirala, Pedro Giovanni
Leon, Mads Schaarup Andersen, Sebastian Zimmeck, Kanthashree Mysore Sathyendra, N. Cameron Russell,
et al. The creation and analysis of a website privacy policy corpus. InProceedings of the Annual Meeting
of the Association for Computational Linguistics (ACL), 2016.
Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan
Kambadur, David Rosenberg, and Gideon Mann. BloombergGPT: A large language model for finance.
arXiv preprint arXiv:2303.17564, 2023.
Adam Wyner and Wim Peters. On rule extraction from regulations. InLegal knowledge and information
systems, pp. 113–122. IOS Press, 2011.
Hongyang Yang, Xiao-Yang Liu, and Christina Dan Wang. FinGPT: Open-source financial large language
models.arXiv preprint arXiv:2306.06031, 2023.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, et al. Judging LLM-as-a-judge with MT-bench and chatbot arena.
InAdvances in Neural Information Processing Systems (NeurIPS), 2023.
Ming Zhong, Yang Liu, Da Yin, Yuning Mao, Yizhu Jiao, Pengfei Liu, Chenguang Zhu, Heng Ji, and Jiawei
Han. Towards a unified multi-dimensional evaluator for text generation. In Yoav Goldberg, Zornitsa
Kozareva, and Yue Zhang (eds.),Proceedings of the 2022 Conference on Empirical Methods in Natural
Language Processing, pp. 2023–2038, Abu Dhabi, United Arab Emirates, December 2022. Association for
Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.131. URL https://aclanthology.org/
2022.emnlp-main.131/.
May Myo Zin, Georg Borges, Ken Satoh, and Wachara Fungwacharakorn. Towards machine-readable traffic
laws: Formalizing traffic rules into prolog using llms. InProceedings of the Twentieth International
Conference on Artificial Intelligence and Law, pp. 327–336, 2025.
12

Preprint. Under review.
A Ablation Studies
We present four ablations targeting the core design decisions of our pipeline. Unless stated otherwise, all
experiments use the following defaults: maximum regeneration retries r=3, acceptance threshold θ=0.9,
Llama-3.1-8B-Instructas the backbone, and HIPAA as the evaluation corpus.
A.1 Impact of Acceptance Threshold
At each pipeline stage, an output is accepted only if its average quality score meets or exceeds θ; otherwise
regeneration is triggered. We ablateθ∈{0.6, 0.7, 0.8, 0.9}and report per-step quality in Table 5.
θStep 1 Step 2 Step 3 Total
0.6 4.86 4.12 4.47 4.48
0.7 4.86 4.21 4.49 4.52
0.84.924.31 4.58 4.61
0.9 4.92 4.42 4.65 4.67
∆(0.6→0.9) +0.06 +0.30 +0.18 +0.19
Table 5: Average quality score per pipeline step across acceptance thresholds θ(scale 1–5). Bold row denotes
our default.∆reports the absolute gain from the weakest to the strongest threshold.
Quality improves monotonically with θ, with the total average rising from 4.48 to 4.67. The ∆row exposes a
structurally meaningful asymmetry: Step 2 captures the dominant share of the gain (+0.30 points, i.e., 6%),
compared to only +0.06 points (1.2%) for Step 1. Step 1 operates over well-scoped, bounded inputs where a
first-pass extraction is typically sufficient; Step 2 involves compositionally harder extraction decisions over
loosely structured definitional content, where the higher bar imposed by θ=0.9meaningfully increases the
probability that a higher-quality candidate is surfaced through regeneration. The consistent gains atθ=0.9
across all stages, at modest additional regeneration cost, confirm it as the optimal operating point.
A.2 Impact of Maximum Regeneration Retries
The retry budget rcontrols how many regeneration attempts a pipeline stage may make before the best-scoring
output is accepted. We ablate r∈{ 0, 1, 2, 3}, wherer=0corresponds to a single-pass pipeline with no
regeneration. Full per-step scores across all values ofrare plotted in Figure 2.
The results reveal a striking non-linearity concentrated entirely in Step 2. Increasing rfrom 0 to 1 yields a
negligible total gain of 0.01 points. This near-zero first-retry return is a diagnostic finding: the distributional
tendencies that produced the initial suboptimal output persists under minimally perturbed sampling, making
a single alternative draw unlikely to materially improve upon the failure.
The qualitative shift occurs at r=2, delivering a total gain of 0.47 points (9.4%) over r=1, driven almost
entirely by Step 2 recovering 1.25 points (25%). This threshold behavior indicates that escaping the low-quality
basin at Step 2 requires a minimum candidate pool of three: the initial output establishes the failure mode,
the first retry explores a partial correction, and the second provides sufficient diversity for the selection
mechanism to identify a genuinely superior output. This is consistent with the qualitative progression in
Figure 3, where judge critiques from earlier attempts provide increasingly targeted feedback that steers the
model away from prior failure modes rather than resampling from the same distribution. Steps 1 and 3 do
not exhibit this pattern, confirming the non-linearity is specific to the compositional complexity of Step 2.
Beyondr=2, total gains saturate at +0.02, and we adopt r=3as the default, providing a conservative safety
margin at negligible additional cost.
13

Preprint. Under review.
0 1 2 3
Maximum Retry Budget r3.003.253.503.754.004.254.504.755.00Average Quality Score
Step 1 (Metadata)
Step 2 (Definitions)
Step 3 (Per-Rule)
Overall
Figure 2: Average quality score per pipeline step as a function of retry budget r(scale 1–5; higher is better).
Steps 1 and 3 remain largely flat throughout, while Step 2 exhibits a sharp threshold effect: negligible
gain fromr=0tor=1, followed by a 1.25-point (25%) recovery at r=2. The shaded region marks the
negligible-gain zone. All steps saturate beyondr=2.
A.3 Impact of Chunking Strategy
Input chunk quality is a silent but consequential variable: each pipeline stage can only reason over the
regulatory context it receives, so structural artifacts introduced at the chunking stage propagate forward.
We compare our chunking strategy against that of Datla et al. (2025) by routing both through an identical
pipeline on the same HIPAA subset.
Chunking StrategyStep 1 Step 2 Step 3 Total
Ours4.97 4.47 4.60 4.68
Datla et al. (2025) 4.64 4.34 4.59 4.52
∆(Ours−comparison target) +0.33 +0.13 +0.01 +0.16
Table 6: Average quality score per pipeline step for each chunking strategy on the HIPAA subset (scale 1–5).
∆reports the absolute gain of our strategy.
The∆row reveals a gain profile whose non-uniformity is diagnostically informative. Step 1 captures nearly
the entire benefit (+0.33 points, i. e., 6.6% ): a chunk that cleanly encapsulates a single regulatory provision
allows precise, well-scoped extraction on the first attempt, whereas a poorly bounded chunk introduces
structural ambiguity that degrades initial quality before regeneration has any opportunity to help. Step 3, by
contrast, is virtually identical across strategies (+0.01), suggesting it operates against a quality ceiling set by
model capacity rather than input structure. Chunking exerts its leverage exclusively in the early pipeline
stages; downstream refinement cannot compensate for incoherent inputs, it can only refine coherent ones.
A.4 Regeneration Trigger Strategy
We evaluate two trigger strategies:avg-trigger, which regenerates if the normalized average score across all
criteria falls below θ=0.9; andindividual-trigger, a strictly more conservative criterion that regenerates if any
single criterion falls below a raw score of 4. Both strategies are evaluated under retry budgets r∈{ 1, 3}to
disentangle the effect of trigger policy from retry budget.
Two findings stand out. First, the retry budget is the dominant factor by a wide margin: increasing rfrom
1 to 3 yields a total gain of 0.48 points (9.6%), with Step 2 alone recovering 1.26 points (25.2%). This
asymmetry is structurally expected: Step 2 faces the most compositionally demanding extraction task and
routinely encounters diverse, co-occurring failure modes that a single retry cannot simultaneously correct.
14

Preprint. Under review.
Max Retries (r) TriggerStep 1 Step 2 Step 3 Total
r=3Avg (default) 4.92 4.42 4.65 4.66
Individual 4.92 4.44 4.63 4.66
r=1Avg 4.87 3.16 4.49 4.18
Individual 4.87 3.21 4.43 4.17
∆(r: 1→3, Avg) +0.05 +1.26 +0.16 +0.48
Table 7: Average quality score per pipeline step across trigger strategies and retry budgets. ∆reports the
gain fromr=1tor=3under the default avg-trigger.
The budget of r=3provides the minimum candidate pool necessary for the best-selection mechanism to
identify a materially superior output. Second, conditional on r=3, both trigger strategies reach identical
total quality (4.66), differing by at most 0.02 points (0.4%) on any individual step. With an adequate retry
budget, the pipeline’s best-of- rselection naturally corrects per-criterion weaknesses without needing them
to be individually flagged. The individual-trigger’s added conservatism imposes higher regeneration and
inference cost with zero measurable return. These results confirm that trigger granularity is a second-order
concern: retry budget is the primary control variable.
B Judgment Criteria
De Jure employs three specialized judges operating sequentially, each evaluating a distinct semantic layer:
section-level metadata (Judge 1), definitional content (Judge 2), and individual rule units (Judge 3). Criteria
within each judge are designed to beorthogonal and collectively exhaustiveover the failure modes specific
to that layer. All criteria are scored on a 1–5 scale, and the per-stage average determines whether the
regeneration thresholdθis met.
Judge 1: Section Metadata.Criteria target both factual accuracy (citations, dates, titles) and extraction
completeness.Non-HallucinationandFidelity to Source Textact as hard correctness gates: a metadata
extractionthatdistortsregulatoryidentifiersisunusableregardlessofstructuralquality.Meaningful Population
of Optional Fieldsserves as a proxy for thoroughness beyond mandatory fields.
Judge 2: Definitions.This stage evaluates whether the extracted glossary is complete, source-faithful, and
well-formed.Precision and Formattingis included because malformed definitions degrade both retrieval
and rule grounding downstream.Quality of Termsassesses whether extracted terms are genuine regulatory
primitives rather than incidental phrases, a distinction requiring semantic judgment beyond surface copying.
Judge 3: Per-Rule Quality.The most demanding stage, assessing each rule unit across eight criteria organized
into three functional clusters:Completeness,Accuracy, andActionabilityassess whether the rule captures
the full operative content of the source provision;Fidelity to Source Text,Neutrality, andNon-Hallucination
form a faithfulness cluster ensuring the extraction neither omits nor introduces content; andConciseness
andConsistencyassess structural quality, as redundant or contradictory rules impose unnecessary burden on
downstream retrieval and synthesis.
B.1 Pairwise Judge Criteria for Downstream Compliance QA
While the three judges above assess intrinsic extraction quality, Experiment 3 (Section 4.4) requires a separate
instrument to measure downstream utility. A pairwise LLM judge compares RAG responses produced from
De Jure extractions against those of Datla et al. (2025) across six criteria.Completenesscaptures whether all
aspects of the question are addressed;Factual Groundingpenalizes claims not traceable to the retrieved rule
set;Handling Ambiguityassesses whether the response correctly distinguishes mandates, permissions, and
unresolved provisions rather than forcing false certainty;Practical Actionabilitymeasures whether regulatory
15

Preprint. Under review.
Judge Evaluates Criteria
Judge 1 Section MetadataCompleteness
Fidelity to Source Text
Non-Hallucination
Title Quality
Precision of Citations and Dates
Meaningful Population of Optional Fields
Judge 2 DefinitionsCompleteness
Fidelity to Source Text
Non-Hallucination
Precision and Formatting
Quality of Terms
Judge 3 Per-Rule QualityCompleteness
Conciseness
Accuracy
Fidelity to Source Text
Neutrality
Consistency
Actionability
Non-Hallucination
Table 8: Judgment criteria grouped by pipeline stage. Each judge targets the failure modes specific to
its semantic layer: structural integrity (Judge 1), definitional precision (Judge 2), and fine-grained rule
correctness and utility (Judge 3).
language is translated into concrete guidance rather than accurate but inert quotation;Regulatory Precision
evaluates correct reflection of scope, conditions, and exceptions; andOverall Preferenceprovides a holistic
judgment integrating all five dimensions, reported separately to surface trade-offs not captured by any single
criterion.
These criteria collectively operationalizedownstream utility. In regulatory compliance, a well-formed extraction
has no value unless it enables a system to answer questions that are complete, grounded, unambiguous, and
actionable. Each criterion directly targets a failure mode that poor rule extraction introduces into the RAG
pipeline: incomplete rules truncate responses, hallucinated content propagates into assertions, imprecise
definitions blur scope, and incoherent structure impedes actionable synthesis. Evaluating at this level provides
a direct, task-grounded measure of whether extraction fidelity translates into practical compliance support.
16

Preprint. Under review.
C Pipeline Success Example: Judgment and Repair in Practice
Thissectionwalksthroughaconcreteexampleofjudge-guidedrepair. Astructurallycompletebutsemantically
flawed extraction is identified by the judge, corrected in a single repair step, and re-evaluated to a passing
score. The example also shows that the judge scores each criterion independently: it penalizes only the fields
that are wrong and preserves high scores on fields that are already correct.
(a) Raw PDF Input
§164.306 Security standards: General rules.
(a)General requirements.Covered entities and business associates
must do the following:
(1) Ensure the confidentiality, integrity, and availability of all
electronic protected health information...
(2) Protect against any reasonably anticipated threats or hazards
to the security or integrity of such information.
(3) Protect against any reasonably anticipated uses or disclosures
not permitted under subpart E.
(4) Ensure compliance with this subpart by its workforce.
(b)Flexibility of approach.
(1) Covered entities and business associates may use any security
measures that allow... reasonably and appropriately implement the
standards...
(2) In deciding which security measures to use, must take into
account:
(i) Size, complexity, and capabilities...
(ii) Technical infrastructure, hardware, and software security
capabilities.
(iii) The costs of security measures.
[68 FR 8376, Feb.20, 2003; 78 FR 5693, Jan.25, 2013](b) Preprocessed Markdown
Security standards: General rules.
-(a)General requirements.Covered entities and business
associates must do the following:
-(1) Ensure the confidentiality, integrity, and
availability of all ePHI ...
-(2) Protect against reasonably anticipated threats or
hazards ...
-(3) Protect against reasonably anticipated uses or
disclosures ...
-(4) Ensure compliance by its workforce.
-(b)Flexibility of approach.
-(1) CE and BA may use any security measures ...
-(2) Must consider:
(i) Size, complexity ...
(ii) Technical infrastructure ...
(iii) Costs of security measures.
[68 FR 8376, Feb. 20, 2003; ... 78 FR 5693](c) Initial Generation
rule_id: RULE-008
label:“Covered entities must implement security
measures.”×incomplete
type: clarification×wrong
action: “Implement security measures”
method: “Use any security measures that allow the
CE/BA
to reasonably implement...”
purpose: “Ensure confid.,integrity, avail. of ePHI”
conditions: [3]constr.: [2]
(d) LLM Judge Scores
Completeness 2/5
Conciseness 3/5
Accuracy 2/5
Consistency 4/5
Fidelity 2/5
Neutrality 4/5
Actionability 2/5
Non-halluc. 3/5
Avg: 55%FAIL<90%
“rule_type is inaccurately classified as ‘clarification’. The label does not capture the full scope of
the rule...”(e) Improved Generation
rule_id: RULE-008
label:“Covered entities and business associates must implement security mea-
sures that balance flexibility with compliance, considering size, complexity, techni-
cal infrastructure, and costs.”✓
type: definition-application✓corrected
action: “Implement security measures”
method: “Use any security measures that allow the CE/BA to reasonably and
appropriately implement...”
purpose: “Ensure confidentiality, integrity, and availability of all ePHI”
conditions: [3]constraints: [2]
Avg: 90%PASS≥90%De Jure Pipeline Example: HIPAA §164.306 — RULE-008
Figure 3: De Jure applied to HIPAA §164.306. Panels (c)→(d)→(e) read left-to-right across the bottom
row. Panels (a)–(b) show the raw PDF source and its pre-processed Markdown. Panel (c) shows the initial
extraction with two field-level defects: an incomplete label and a misclassified rule type. Panel (d) shows
the judge evaluation (avg. 0.55,fail) with per-criterion scores and targeted critiques. Panel (e) shows the
corrected extraction after a single repair iteration (avg. 0.90,pass), with only deficient fields revised and all
others preserved.
Figure 3 traces RULE-008 from HIPAA §164.306 through the full pipeline. Panels (a)–(b) show the raw
source and its pre-processed Markdown. The four blocks below correspond to: initial extraction, judge
evaluation of the initial extraction, corrected extraction, and judge re-evaluation of the corrected extraction.
Block 1: Initial Extraction.The initial extraction is structurally complete as per the extraction schema (see
Appendix H), and free of hallucinations. Targets, conditions, and the verbatim span are all correct. However,
two fields are defective. The label is too brief:"Covered entities must implement security measures"omits
the multi-factor balancing that is central to the provision. The rule_type is misclassified as clarification
instead ofdefinition-application, silently reversing the compliance semantics.
17

Preprint. Under review.
Block 1: Initial Extraction (RULE-008)
{
"rule_id": "RULE-008",
"label": "Covered entities must implement security measures.",
"rule_type": { "type":"clarification", "other_label": "implementation-guidance" },
"targets": [ { "role": "covered_entity" }, { "role": "business_associate" } ],
"statement": {
"action": "Implement security measures",
"action_object": "security measures",
"method": "Use any security measures that allow the covered entity or business
associate to reasonably and appropriately implement the standards and
implementation specifications as specified in this subpart, taking into
account the size, complexity, and capabilities of the covered entity or
business associate, technical infrastructure, hardware, and software
security capabilities, and costs of security measures.",
"constraints": [
{ "text": "Reasonably anticipated threats or hazards to the security or
integrity of electronic protected health information" },
{ "text": "Reasonably anticipated uses or disclosures of electronic protected
health information not permitted or required under subpart E" }
],
"conditions": [
{ "trigger": "size, complexity, and capabilities of the covered entity or business associate" },
{ "trigger": "technical infrastructure, hardware, and software security capabilities" },
{ "trigger": "costs of security measures" }
],
"exceptions": [],
"penalties_or_consequences": null,
"purpose": "Ensure the confidentiality, integrity, and availability of all
electronic protected health information",
"verbatim": "Covered entities and business associates may use any security
measures that allow the covered entity or business associate to reasonably
and appropriately implement the standards and implementation specifications
as specified in this subpart."
},
"citations": { "text": "164.306: Security standards: General rules.
[68 FR 8376, Feb. 20, 2003; 68 FR 17153, Apr. 8, 2003; 78 FR 5693, Jan. 25, 2013]" },
"examples": []
}
Block 2: Judge Evaluation (Fail, 0.55).The judge returns a normalized average score of 0.55 and flags the
extraction asfail. The critique is targeted: it assigns low scores toCompleteness,Accuracy,Fidelity, and
Actionability, which are the affected criteria, while keeping high scores onConsistencyandNeutrality, which
are correct. This field-level signal is what makes the repair step precise. The feedback for each criteria is
provided to guide the regeneration in a direction to improve the score on the respected criteria.
Block 2: Judge Evaluation of Initial Extraction|Avg. 0.55 (fail)
"Completeness": {"Score": 2, "Justification": "Label incompletely summarizes the rule,
omitting the multi-factor balancing requirement central to the provision.
rule_type is misclassified; should be definition-application." },
"Conciseness": {"Score": 3, "Justification": "Label is brief but sacrifices accuracy
for brevity; the scope of the rule is inadequately represented." },
"Accuracy": {"Score": 2, "Justification": "rule_type classified as clarification
instead of definition-application, misrepresenting the rule’s nature." },
"Consistency": {"Score": 4, "Justification": "Targets correctly identified as
covered_entity and business_associate, consistent with source." },
"Fidelity": {"Score": 2, "Justification": "Statement omits the full balancing
framework including both targets; significant omission relative to source." },
"Neutrality": {"Score": 4, "Justification": "Label and statement are neutrally
presented; no interpretive bias introduced." },
"Actionability": {"Score": 2, "Justification": "Statement fails to convey what both
covered entities and business associates must do; insufficiently operational." },
"Non_hallucination": {"Score": 3, "Justification": "No fabricated content, but
significant omissions reduce faithfulness to the source provision." }
Average Normalized Score: 0.55 – FAIL – Regeneration triggered.
Block 3: Corrected Extraction.After one repair iteration, only the two flagged fields change. The
label is expanded to capture the multi-factor balancing framework. The rule_type is corrected to
definition-application . All other fields, including conditions, constraints, verbatim span, and cita-
tions, remain exactly as before. The repair targets specific deficiencies rather than regenerating the rule from
scratch.
18

Preprint. Under review.
Block 3: Corrected Extraction (RULE-008)
{
"rule_id": "RULE-008",
"label":"Covered entities and business associates must implement security
measures that balance flexibility with compliance, considering size,
complexity, technical infrastructure, and costs.",
"rule_type": {"type": "definition-application", "other_label": "implementation-guidance" },
"targets": [ { "role": "covered_entity" }, { "role": "business_associate" } ],
"statement": {
"action": "Implement security measures",
"action_object": "security measures",
"method": "Use any security measures that allow the covered entity or business
associate to reasonably and appropriately implement the standards and
implementation specifications as specified in this subpart, taking into
account the size, complexity, and capabilities of the covered entity or
business associate, technical infrastructure, hardware, and software
security capabilities, and costs of security measures.",
"constraints": [
{ "text": "Reasonably anticipated threats or hazards to the security or
integrity of electronic protected health information" },
{ "text": "Reasonably anticipated uses or disclosures of electronic protected
health information not permitted or required under subpart E" }
],
"conditions": [
{ "trigger": "size, complexity, and capabilities of the covered entity or business associate" },
{ "trigger": "technical infrastructure, hardware, and software security capabilities" },
{ "trigger": "costs of security measures" }
],
"exceptions": [],
"penalties_or_consequences": null,
"purpose": "Ensure the confidentiality, integrity, and availability of all
electronic protected health information",
"verbatim": "Covered entities and business associates may use any security
measures that allow the covered entity or business associate to reasonably
and appropriately implement the standards and implementation specifications
as specified in this subpart."
},
"citations": { "text": "164.306: Security standards: General rules.
[68 FR 8376, Feb. 20, 2003; 68 FR 17153, Apr. 8, 2003; 78 FR 5693, Jan. 25, 2013]" },
"examples": []
}
Block 4: Judge Re-evaluation (Pass, 0.90).The judge re-evaluates the corrected extraction and returns
a normalized average score of 0.90. All four previously low criteria recover:CompletenessandAccuracy
reach 5, reflecting the corrected rule type and the richer label.ActionabilityandFidelitysimilarly improve.
Concisenessdrops slightly, as the longer label trades brevity for coverage. Criteria that were already correct
retain their scores, confirming that the judge does not penalize correct fields for the failures of others. The
score meets the thresholdθ=0.90, and the repair loop halts.
Block 4: Judge Evaluation of Corrected Extraction|Avg. 0.90 (pass)
"Completeness": {"Score": 5, "Justification": "Label now comprehensively captures
the multi-factor balancing requirement. All required fields populated correctly.
rule_type corrected to definition-application." },
"Conciseness": {"Score": 4, "Justification": "Label is necessarily detailed to
reflect the provision’s scope; slight verbosity is justified given the
complexity of the rule." },
"Accuracy": {"Score": 5, "Justification": "rule_type correctly classified as
definition-application; accurately represents the provision’s regulatory
function." },
"Consistency": {"Score": 4, "Justification": "Targets consistently and correctly
identify both covered_entity and business_associate throughout." },
"Fidelity": {"Score": 4, "Justification": "Statement closely reflects the source
text; label now captures both targets and the balancing framework.
Minor omission: flexibility aspect could be made more explicit." },
"Neutrality": {"Score": 5, "Justification": "Label and statement remain neutrally
framed with no interpretive bias introduced." },
"Actionability": {"Score": 4, "Justification": "Statement now provides operationally
useful guidance for both covered entities and business associates, though
method field is dense and could be more concisely structured." },
"Non_hallucination": {"Score": 5, "Justification": "All extracted content is
grounded in the source text; no fabricated or inferred content introduced." }
Average Normalized Score: 0.90 – PASS – Repair loop terminates.
19

Preprint. Under review.
The jump of the normalized averaged score from 0.55 to 0.90 was driven by two field corrections, with
everything else unchanged. This pattern holds across all corpora and aligns with the quantitative results in
Section A.2.
This example illustrates two key properties of the judge. First, it reliably flags weak extractions: the low
scores in Block 2 are not generic penalties but precise signals tied to specific fields, paired with actionable
feedback that directly guides the repair. Second, it recognizes improvement: once those fields are corrected,
the judge awards high scores to the updated output without penalizing fields that were already correct.
These two properties are what make automated iterative repair practical. The judge distinguishes poor
extractions from good ones, and its feedback is specific enough to drive targeted corrections rather than blind
regeneration. This allows the extraction and verification to operate as mutually reinforcing stages and drive
De Jure’s self-correcting nature.
20

Preprint. Under review.
D De Jure Pipeline: Algorithm Psuedo Code
Algorithm 1 provides a complete procedural specification of the De Jure pipeline, complementing the
stage-by-stage description in Section 3.
Algorithm 1:De Jure: Regulatory Rule Extraction with Judgment and Repair
Input :DocumentD(PDF/HTML)
Output:Structured rule setR
1/* Stage 1: Pre-processing */
2M←ConvertToMarkdown(D)
3M←CleanAndNormalise(M)
4S←SplitBySections(M) // split on section markers
5foreachsections∈Sdo
6IndexByID(s)
7Attach metadata andSHA256(D)tos
8end
9/* Stage 2: Generation, Judgment, and Selective Repair */
10R←∅
11r←3// max regeneration attempts per stage
12θ←0.90// avg-score acceptance threshold
13stages←[(metadata, 6), (definitions, 5), (rule_units, 8)]
14foreachsections∈Sdo
15e←LLMGenerate(s,JSONSchema)
16ife=nullthen continue// non-actionable section
17
18foreach(name,n c)∈stagesdo
19ˆe←e
20ˆµ←0// best score seen for this stage
21fort←1tordo
22scores,critique←JudgeStage(e,name,n ccriteria)
23µ←avg(scores)
24ifµ>ˆµthen
25ˆe←e
26ˆµ←µ
27end
28ifµ≥θthen break// threshold met
29
30e←LLMGenerate(s,e,scores,critique,JSONSchema)
31end
32e←ˆe// commit best generation for this stage
33end
34R←R∪{e}
35end
36returnR
The total LLM call budget per section is bounded by1 +3×rin the worst case (one initial generation plus
at mostrrepair attempts across each of the three stages), and reduces to1 +3in the best case where all
stages pass on the first judgment.
21

Preprint. Under review.
E Extraction Quality Experiment: Full Results
This section provides further details and extended results for the experiments presented in Section 4.2.
Table 9 presents the complete Judge 1 metadata criteria scores for Experiment 1 across the four different
models evaluated on SEC corpus. One observation that stands out is that Non-Hallucination and Fidelity
to source text is uniformly perfect (5.00) across all models, confirming that schema-constrained extraction
eliminates factual fabrication as a failure mode regardless of model family or access paradigm.
Model Completeness Fidelity Non-Halluc. Title Quality Citation & Date Opt. Avg
Claude-3.5-Sonnet4.95 5.00 5.00 5.00 5.00 5.004.99
Llama-3.1-8B 4.845.00 5.004.95 4.89 4.924.93
GPT-5-mini4.95 5.00 5.00 5.00 5.004.974.99
Qwen3-VL-8B 4.855.00 5.00 5.004.97 4.774.99
Table 9: Full per-criteria scores on the SEC Investment Advisers Act corpus forJudge 1(Metadata) criteria
(scale 1–5; higher is better). Abbreviations: Non-Halluc. = Non-Hallucination.; Opt. = Optional Field
Population; Avg = Average Score
Table 10 presents the complete Judge 2 definition criteria scores for Experiment 1 across the four different
models evaluated on SEC corpus. Non-Hallucination remains uniformly perfect (5.00), indicating strong
faithfulness under the structured extraction setup. Compared to metadata, greater variation is observed in
Completeness and Precision, reflecting the higher compositional complexity of definitions, withQwen3-VL-8B
achieving the highest overall score (4.87) while all models remain closely clustered.
Model Completeness Fidelity Non-Halluc. Precision Term Quality Avg
Claude-3.5-Sonnet 4.59 4.905.004.77 4.904.83
Llama-3.1-8B 4.50 4.765.004.66 4.844.75
GPT-5-mini 4.544.95 5.004.694.954.83
Qwen3-VL-8B4.694.905.00 4.854.924.87
Table 10: Full per-criteria scores on the SEC Investment Advisers Act corpus forJudge 2(Definitions) criteria
(scale 1–5; higher is better). Abbreviations: Non-Halluc. = Non-Hallucination; Precision = Precision &
Formatting.
Table 11 presents the complete Judge 3 per-rule criteria scores for Experiment 1 across the four different
models evaluated on SEC corpus. All the models again obtain perfect scores for Non-Hallucination as well as
Neutrality, indicating strong faithfulness under the structured extraction setup. Performance differences are
most pronounced in Accuracy, Consistency, and Fidelity, whereGPT-5-minileads overall (4.75Avg), while
open-source models remain competitive, withQwen3-VL-8Bachieving comparable scores (4.64), highlighting
the robustness of the pipeline across model families.
Model Comp. Conc. Accuracy Cons. Fidelity Neut. Actionab. Non-Halluc. Avg
Claude-3.5-Sonnet 4.114.684.80 4.95 4.435.004.425.004.67
Llama-3.1-8B 4.06 4.40 4.57 4.72 4.265.004.255.004.53
GPT-5-mini4.274.494.95 4.98 4.74 5.00 4.54 5.004.75
Qwen3-VL-8B 4.10 4.48 4.81 4.93 4.435.004.365.004.64
Table 11: Full per-criteria scores on the SEC Investment Advisers Act corpus forJudge 3(Per-Rule) criteria
(scale 1–5; higher is better). Abbreviations: Comp. = Completeness; Conc. = Conciseness; Cons. =
Consistency; Neut. = Neutrality; Actionab. = Actionability; Non-Halluc. = Non-Hallucination.
22

Preprint. Under review.
F Domain Generalization Experiment: Full Results
This section provides further details and extended results for the experiments presented in Section 4.3.
Table 12 presents the complete Experiment 2 results for Judge 1 metadata criteria across SEC, EU AI Act,
and HIPAA.Qwen3-VL-8BandGPT-5-miniperform comparably overall on metadata extraction, with only
small differences across most criteria. Notably, the EU AI Act is consistently lower than the other two
datasets on bothCompletenessandCitation & Date, indicating that metadata recovery is more challenging
in this corpus.
Dataset Model Completeness Fidelity Non-Halluc. Title Quality Citation & Date Opt.
SECQwen3-VL-8B 4.85 5.00 5.00 5.00 4.97 4.77
GPT-5-mini 4.95 5.00 5.00 5.00 5.00 4.97
EU AI ActQwen3-VL-8B 3.89 5.00 5.00 5.00 4.67 4.33
GPT-5-mini 4.00 5.00 5.00 5.00 4.78 4.56
HIPAAQwen3-VL-8B 4.85 4.95 4.93 5.00 4.90 4.76
GPT-5-mini 4.85 5.00 5.00 5.00 4.95 4.76
Table 12: Generalization across three regulatory corpora forJudge 1(Metadata) criteria (scale 1–5; higher is
better). Abbreviations: Non-Halluc. = Non-Hallucination.; Opt. = Optional Field Population
Table 13 presents the complete Experiment 2 results for Judge 2 definition criteria across SEC, EU AI Act,
and HIPAA.GPT-5-miniunderperforms on definitionCompletenessandPrecision, with the largest drop
observed on HIPAA. More broadly, HIPAA appears to be the most difficult corpus for definition extraction,
with both models showing lower scores than on SEC and EU AI Act in key definition quality dimensions.
Dataset Model Completeness Fidelity Non-Halluc. Precision Term Quality
SECQwen3-VL-8B 4.69 4.90 5.00 4.85 4.92
GPT-5-mini 4.54 4.95 5.00 4.69 4.95
EU AI ActQwen3-VL-8B 4.67 4.89 5.00 4.67 4.89
GPT-5-mini 4.44 4.78 5.00 4.44 4.78
HIPAAQwen3-VL-8B 4.51 4.71 5.00 4.63 4.68
GPT-5-mini 4.20 4.76 5.00 4.32 4.78
Table 13: Generalization across three regulatory corpora forJudge 2(Definitions) criteria (scale 1–5; higher
is better). Abbreviations: Non-Halluc. = Non-Hallucination; Precision = Precision & Formatting.
Table 14 presents the complete Experiment 2 results for Judge 3 per-rule criteria across SEC, EU AI Act,
and HIPAA.Qwen3-VL-8Bunderperforms relative toGPT-5-minion ruleCompleteness,Fidelity, and
Actionabilityacross datasets. At the same time, both models perform strongly onAccuracy,Consistency,
andNeutrality. Finally, consistent with Tables 12 and 13, hallucination remains minimal across all three
judges, with near-ceiling non-hallucination scores throughout.
Dataset Model Comp. Conc. Accuracy Cons. Fidelity Neut. Actionab. Non-Halluc.
SECQwen3-VL-8B 4.10 4.48 4.81 4.93 4.43 5.00 4.36 5.00
GPT-5-mini 4.27 4.49 4.95 4.98 4.74 5.00 4.54 5.00
EU AI ActQwen3-VL-8B 4.28 4.41 4.90 4.93 4.51 5.00 4.19 5.00
GPT-5-mini 4.42 4.35 4.95 4.98 4.65 5.00 4.34 5.00
HIPAAQwen3-VL-8B 4.15 4.52 4.84 4.96 4.50 5.00 4.32 4.99
GPT-5-mini 4.31 4.52 4.95 4.99 4.76 5.00 4.46 5.00
Table 14: Generalization across three regulatory corpora forJudge 3(Per-Rule) criteria (scale 1–5; higher
is better). Abbreviations: Comp. = Completeness; Conc. = Conciseness; Cons. = Consistency; Neut. =
Neutrality; Actionab. = Actionability; Non-Halluc. = Non-Hallucination.
23

Preprint. Under review.
G Prompts
De Jure uses seven prompts organized into three functional groups: an initial extraction prompt that generates
the structured rule representation from raw text (Section G.1), three stage-specific regeneration prompts that
incorporate judge feedback to correct failing extractions (Section G.2), and three judge prompts that produce
the per-criterion scores and natural-language critiques driving the repair loop (Section G.3). All prompts are
fully domain-agnostic and contain no corpus-specific examples or seed rules.
G.1 Initial Extraction Prompt
Prompt 1 is invoked once per regulatory section at Stage 2 of the pipeline. It instructs the backbone LLM to
decompose the source text into three typed components: section metadata, definitions, and rule units, each
conforming to a fixed schema. The prompt encodes quality constraints inline, including null-filtering for non-
actionable sections and negation-preserving rules for action fields, to minimize the number of judge-triggered
repairs on well-formed inputs.
Prompt 1|Rule Generation (Stage 2)
You are an expert at analyzing regulatory documents to extract structured, actionable rules from {}.
## Task
Extract all rules, requirements, obligations, prohibitions, permissions, and procedures. Each distinct rule = one RuleUnit.
Requirements:
•Exhaustive: Capture every rule in the text
•Objective: Use only source text – no inference or interpretation
•Precise: Include verbatim quotes to ground extractions
•Structured: Decompose into clear components
## Extraction Guidelines
### Section Metadata
Citation
•Official regulatory section number (e.g., "17 CFR 275.0-2", "Rule 144", "§230.405")
•DO NOT use Federal Register citations (e.g., "51 FR 32907") assection_cite
•If no regulatory section identifier exists in source, set to null
Title
•Section’s official heading
Effective Dates
•Extract dates in EXACT format from source (e.g., "Sept. 17, 1986" NOT "September 17, 1986")
•Preserve abbreviations, punctuation, and spacing
•Extract ALL dates with appropriate event types
•Multiple FR citations typically indicate: first = "adopted", subsequent = "amended"
•Look for context clues in brackets (e.g., "amended at 64 FR...")
•Common event types: adopted, amended, effective, rescinded
notes
•Set to null unless source contains additional context not captured in other fields
x_extensions
•Set to null unless source contains non-standard metadata
Definitions
Extract terms whose meaning is established by the text:
•Positive definitions: "X means Y"
•Negative definitions: "X is not Y" or "X does not include Y"
•Example: "A transaction...is not an assignment"→defines what "assignment" excludes
Note: Definitional text may ALSO constitute a rule (e.g., "X is not considered Y" both defines Y’s scope AND exempts X). Extract to BOTH definitions and
extracted_ruleswhen applicable.
### For Each Rule Extract:
#### 1. Rule Identification
•rule_id: Unique identifier (e.g., "RULE-001"). Use consistent nomenclature across document.
•label: Concise summary (5–25 words)
•rule_type: obligation | prohibition | permission | exemption | definition-application | safe-harbor | procedure | clarification | deeming | condition-precedent | other
(provideother_label)
•Useexemptionfor: Carves out exceptions or states what is NOT covered (e.g., "X is not deemed Y")
•Useclarificationfor: Explains scope or meaning without imposing new obligations
#### 2. Targets (WHO must comply)
CRITICAL: Target = WHO is being instructed to perform the action per this rule, NOT who is the action being performed on!
Example:
•"Issuers must file to Commission"→Target isissuer(not Commission)
24

Preprint. Under review.
•"Any person may serve by furnishing Commission..."→Target isany_person
•"If⟨event⟩, the head of Commission must..."→Target ishead of Commission(not Commission)
For each target:
•role: answers the "who must follow this rule?" question
•Preserve exact entity names: "Secretary of the Commission" NOT "commission"
•qualifiers: captures any additional conditions that apply (e.g. "with 15+ employees")
IMPORTANT – Resolve Vague References:
•If source text uses "you", "the registrant", "such person", determine the actual entity from:
– Section title (e.g., "Rules for Investment Advisers"→target is "investment adviser")
– Regulatory context (e.g., Form CRS is for broker-dealers and investment advisers)
– Earlier definitional text in the section
#### 3. Statement
Extract the complete rule decomposition as a single structured object with the following fields:
3.1 action
•Primary regulatory verb phrase (multi-word OK: "file reports", "serve legal process")
•CRITICAL: For negative statements, INCLUDE the negation in the action field
•WHAT must/must not/may be done?
Examples of negative actions:"is not deemed"•"does not constitute"•"shall not be required"•"are not subject to"
×Wrong: "deemed"✓Correct: "is not deemed"
×Wrong: "considered"✓Correct: "is not considered"
3.2 action_object
•WHO/WHAT is acted upon (e.g., "to the Commission", "non-resident advisers", "assignment")
3.3 method_or_conditions
•HOW performed – mechanisms, procedures (e.g., "by furnishing documents", "by delivering the amended Form CRS or by communicating through another
disclosure")
•DO NOT include temporal/quantitative constraints here– extract those toconstraintsfield
3.4 constraints
Extract temporal/quantitative/qualitative limits as an array.If none, return [].
For each constraint:
•text: Clear description (e.g., "within 10 business days", "not less than $5,000", "without charge")
•applies_to: What is constrained (null if ambiguous)
3.5 conditions
Capture ALL conditions that define when, where, or under what circumstances the rule applies.If unconditional, return [].
IMPORTANT: The trigger should describe the QUALIFYING circumstance, not just repeat the subject
×Wrong: trigger: "transaction"
✓Correct: trigger: "transaction does not result in change of actual control or management"
Types of conditions to capture:
1.Event-based triggers(something happens):
•"If process is served..."→trigger: "process is served on the Commission"
•"When trading volume exceeds..."→trigger: "trading volume exceeds threshold"
•"Upon receipt of notice..."→trigger: "receipt of notice"
2.Scope/jurisdictional conditions(regulatory framework):
•"Under Forms ADV and ADV-NR..."→trigger: "Under Forms ADV and ADV-NR"
•"For purposes of Section 5..."→trigger: "For purposes of Section 5"
•"As provided in Regulation S-K..."→trigger: "As provided in Regulation S-K"
3.Status-based conditions(entity characteristics):
•"If the issuer is foreign..."→trigger: "issuer is foreign"
•"For reporting companies..."→trigger: "entity is a reporting company"
For each condition:
•trigger: The condition that must be met (can be event, scope, or status)
•time_window: Start/end times and timezone (only for temporal conditions)
•cross_references: Regulatory citations mentioned in the condition (e.g., "section 205(a)(2) of the Act")
3.6 exceptions
Extract exemptions/carve-outs.If none, return [].
For each exception:
•text: Exception description
•cross_references: Regulatory citations elaborating exception
3.7 penalties_or_consequences
Extract stated consequences of non-compliance.If none, return null.
For each penalty/consequence:
•text: Description of penalty or consequence
•cross_references: Regulatory citations related to the penalty
3.8 purpose
•Stated objective (only if explicit; else null)
3.9 verbatim
•REQUIRED– Exact source quote establishing rule
•Must include ALL sentences that contribute to this RuleUnit
#### 4. Additional Elements
•citations: ALL regulatory references mentioned in the rule, regardless of where they appear in the statement. Include references from conditions, exceptions, and
the main statement. For each citation provide the reference text and context of its use.
•examples: Illustrative examples from source (lift verbatim; else null)
## Output Format
JSON conforming toSecActSectionExtractionschema:
25

Preprint. Under review.
•section_meta: Citation, title, effective dates
•definitions: Terms defined in section with explanations
•extracted_rules: List of RuleUnit objects
## Quality Standards
•Completeness: Every regulatory statement→RuleUnit with all applicable fields filled
•Fidelity: Verbatim = actual quotes, not paraphrases
•Granularity: Multiple distinct obligations→separate RuleUnits
•No Hallucination: Only explicit content; use null/[] when absent
•Preserve Specificity: Exact entity names, titles, "if" statements – no simplification
•Regulatory Citations: Federal Register citations (e.g., [51 FR 32907, Sept. 17, 1986]) indicate effective dates but should not be extracted to notes or x_extensions
unless they provide substantive context beyond dating.
## Special Cases
•Compound Rules: Multiple obligations on different targets→split into RuleUnits
•Cross-References: "as defined in §240.10b-5"→capture incross_references
•Implicit Requirements: Extract only if clearly stated (e.g., "failure to file within 10 days requires..." implies filing obligation)
## CRITICAL OUTPUT REQUIREMENTS
Your JSON output MUST include these fields for EVERY RuleUnit (use[]for empty):
{
"rule_id": "...",
"label": "...",
"rule_type": {...},
"targets": [...],
"statement": {
"action": "...",
"action_object": "...",
"method": "...",
"constraints": [...],// REQUIRED – use [] only if genuinely absent
"conditions": [...],// REQUIRED – use [] only if genuinely absent
"exceptions": [...],// REQUIRED – use [] only if genuinely absent
"penalties_or_consequences": [...],// REQUIRED – use null only if genuinely absent
"purpose": "...",
"verbatim": "..."
},
"citations": {...},
"examples": [...]
}
Missing any of these fields will cause validation failure.
FINAL VERIFICATION PROTOCOL
Before returning your JSON output:
1. Section Metadata Verification:
•section_cite: null if not in source, exact format if present (DO NOT use FR citations)
•effective_dates: ALL dates with correct event types ("adopted" for first, "amended" for subsequent)
•Dates: exact format match including abbreviations (e.g., "Sept." not "September")
•definitions: Extract negative definitions too (e.g., "X is not Y")
•Optional fields (notes, x_extensions): explicitly null if unused
2. Rule Extraction Verification:
•action: For negative statements, verify negation is INCLUDED (e.g., "is not deemed" NOT "deemed")
•rule_type: "exemption" for exclusionary rules (e.g., "X is not Y")
•conditions: trigger describes the qualifying circumstance, not just the subject
•conditions:cross_referencespopulated if regulatory citations mentioned
•citations: ALL regulatory references included with context
3. Completeness Check:For each RuleUnit, verify that you’ve populated:
•conditions array (even if single item)
•constraints array (check for temporal/quantitative limits)
•exceptions array (check for carve-outs)
•penalties_or_consequences (check for stated consequences)
4. Source Cross-Check:For any field marked as[], re-read the verbatim quote and confirm no extractable information exists.
5. Scope Conditions:Specifically check if the rule begins with regulatory context like "Under [Form/Regulation]" or "For purposes of [Section]" – this is ALWAYS a
condition.
Now analyze the document and extract all rules:
{}
Prompt 1: Initial rule generation prompt (Stage 2 of De Jure). The first placeholder {}is filled with the regulatory domain;
the second with the source section text. No domain-specific examples or seed rules are included.
G.2 Regeneration Prompts
When a pipeline stage fails to meet the acceptance threshold θ, the backbone LLM is re-prompted with
three inputs: the original source text, the failing extraction, and the judge’s structured critique. Three
stage-specific regeneration prompts are used, one per judgment stage, each targeting the exact output type of
26

Preprint. Under review.
its corresponding judge. Crucially, each prompt asks the model to correct only the deficient fields identified
in the critique rather than regenerating the full extraction, keeping repairs surgical and computationally
bounded.
Prompt 2|Stage 1 Regeneration: Section Metadata
Your task is to extract section metadata from a source text containing regulatory content. You have been provided with a previous extraction that was assessed based
on specific criteria and determined to be incorrect.
Using the critique, generate corrected metadata from the source text to achieve the highest score on all mentioned criteria:
•Completeness
•Fidelity to Source Text
•Non-Hallucination
•Title Quality
•Precision of Citations and Dates
•Reasonable Population of Optional Fields
Follow the format below:
{
"section_cite": "citation from source",
"title": "section title",
"effective_dates": [{"event": "event_type", "date": "exact date format"}],
"notes": "additional context if present in source",// (optional)
"x_extensions": {}// (optional)
}
Only generate the corrected metadata JSON and nothing else.
Inputs:
Source Text: {}
Incorrect Metadata: {}
Critique: {}
Prompt 2: Stage 1 regeneration prompt. Placeholders are filled at runtime with the source section text, the failing metadata
extraction, and the Judge 1 critique.
Prompt 3|Stage 2 Regeneration: Definitions
Your task is to extract definitions from a source text containing regulatory content. You have been provided with a previous extraction that was assessed based on
specific criteria and determined to be incorrect.
Using the critique, generate corrected definitions from the source text to achieve the highest score on all mentioned criteria:
•Completeness
•Fidelity to Source Text
•No Hallucination or Fabrication
•Precision and Formatting
•Quality of Terms
Follow the format below:
{
"definitions": [
{
"term": term,
"text": definition
}
]
}
Only generate the corrected definitions JSON and nothing else.
Inputs:
Source Text: {}
Incorrect Definitions: {}
Critique: {}
Prompt 3: Stage 2 regeneration prompt. Placeholders are filled at runtime with the source section text, the failing definitions
extraction, and the Judge 2 critique.
27

Preprint. Under review.
Prompt 4|Stage 3 Regeneration: Rule Units
Your task is to extract rule units from a source text containing regulatory content. You have been provided with a previous extraction that was assessed based on
specific criteria and determined to be incorrect.
Using the critique, generate a corrected rule unit from the source text to achieve the highest score on all mentioned criteria:
•Completeness
•Conciseness (for label)
•Accuracy (of rule_type)
•Consistency (of targets)
•Fidelity to Source Text (statements)
•Neutrality
•Actionability
•No Hallucination
Follow the format below:
{
"rule_id": "rule_id from the source/generated",
"label": "concise summary (5–25 words)",
"rule_type": "obligation" | "prohibition" | "permission" | "exemption" |
"definition-application" | "safe-harbor" | "procedure" |
"clarification" | "deeming" | "condition-precedent" | "other",
"targets": ["WHO must comply, is prohibited, or is granted permission"],
"statement": {
"action": "primary regulatory action",
"action_object": "direct object or recipient of the action",
"method": "HOW the action must be performed",
"constraints": [...],// REQUIRED – [] if genuinely absent
"conditions": [...],// REQUIRED – [] if genuinely absent
"exceptions": [...],// REQUIRED – [] if genuinely absent
"penalties_or_consequences": [...],// REQUIRED – null if genuinely absent
"purpose": "stated objective",
"verbatim": "source quote establishing rule"
},
"citations": [...],// REQUIRED – [] if genuinely absent
"examples": [...]// REQUIRED – [] if genuinely absent
}
Only generate the corrected RuleUnit JSON and nothing else.
Inputs:
Source Text: {}
Incorrect RuleUnit: {}
Critique: {}
Prompt 4: Stage 3 regeneration prompt. Placeholders are filled at runtime with the source section text, the failing rule unit
extraction, and the Judge 3 critique.
G.3 Judge Prompts
The three judge prompts below are invoked sequentially during the multi-criteria evaluation stage to assess
section metadata (Judge 1), definitions (Judge 2), and individual rule units (Judge 3). Each prompt
instructs the judge LLM to score the extraction independently on per-stage criteria and to produce structured
natural-language critiques. These critiques are subsequently passed verbatim as input to the corresponding
regeneration prompts in Section G.2, forming the closed judge-repair loop at the core of De Jure. All prompts
share the same 0–5 scoring rubric and instruct the judge to assign 5 to inapplicable criteria, ensuring the
thresholdθis not penalized by structural variation across regulatory sections.
Prompt 5|Judge 1: Section Metadata Evaluation
Evaluate whether the extracted section metadata is substantially accurate based on the following criteria:
1. Completeness
Major metadata elements should be extracted and populated:
•section_citeshould be present and identify the correct section
•titleshould be captured if clearly present in the source
•effective_datesshould include at least the primary temporal event (if any)
•notes(optional) may capture additional context
•x_extensions(optional) may include non-standard metadata
Missing critical fields (section_cite,titlewhen present) are significant issues. Missing optional fields or secondary dates are minor issues.
28

Preprint. Under review.
2. Fidelity to Source Text
Notes and x_extensions should reasonably reflect the source content. Direct quotes, close paraphrasing, or reasonable interpretations are all acceptable. Minor
rewording or normalization of language is acceptable.
3. Non-Hallucination
Fields should only be populated when corresponding information exists in the source. Do not fabricate dates, citations, or contextual information. Event types
ineffective_dates should be grounded in the source (normalized terminology acceptable, e.g., "enacted" →"adopted"). This criterion is strict: hallucinated
information is a significant problem.
4. Title Quality
Title should accurately reflect the section title if present. Minor formatting variations are acceptable. Null is appropriate if no title exists.
5. Precision of Citations and Dates
Section citations should identify the correct section (minor formatting differences acceptable). Dates should be correct to at least the month/year level. If multiple
effective dates exist, capturing the primary date is essential; missing secondary dates is a minor issue.
6. Reasonable Population of Optional Fields
Whennotesorx_extensionsare populated, they should add value. Omitting these fields when relevant information exists is a minor issue, not a major deficiency.
Scoring Guidelines (per criterion):
•5.0: Fully satisfied with no errors
•4.0–4.9: Mostly satisfied with minor issues
•3.0–3.9: Partially satisfied with notable gaps or minor inaccuracies
•2.0–2.9: Poorly satisfied with significant omissions or errors
•1.0–1.9: Barely satisfied with major problems
•0.0–0.9: Not satisfied – critical failures or fabrications
Score each of the 6 criteria independently. If a criterion is not applicable, assign 5. Report the average as the final score.
Inputs:
Source Text: {}
Extracted Metadata: {}
Prompt 5: Judge 1 evaluation prompt for section metadata (6 criteria). Output scores and critiques are consumed by
Prompt 2.
Prompt 6|Judge 2: Definitions Evaluation
Evaluate whether the extracted definitions and related rule extractions are substantially accurate based on the following criteria:
1. Completeness
Major definitional statements should be captured, including:
•Primary positive definitions ("X means Y", "X includes Y")
•Significant negative definitions or exclusions ("X is not Y", "X does not include Y")
•Important context such as major exceptions or limitations
Missing primary definitions or major exclusions are significant issues. Missing minor qualifications or secondary cross-references are minor issues.
2. Fidelity to Source Text
Extracted terms and definitions should reasonably reflect the source. Direct quotes, close paraphrasing, or reasonable interpretations preserving core meaning are all
acceptable. Definitions should not substantially contradict the source or misrepresent the term’s meaning.
3. No Hallucination or Fabrication
Extract only definitions present in the source. Do not invent terms, definitions, or context not grounded in the text. This criterion is strict: fabricated content is a
significant problem. Reasonable interpretations of existing content are acceptable.
4. Precision and Formatting
Terms should be substantially accurate in spelling and punctuation. Important references (e.g., to statutes or sections) should be captured. Each extracted definition
should be clear and understandable. Minor formatting inconsistencies are acceptable if core meaning is preserved.
5. Quality of Terms
Each extracted term should reasonably match the terminology used in the source. Terms should accurately represent the intended meaning and context. Minor
variations in term format or phrasing are acceptable if they do not misrepresent the definition.
Scoring Guidelines (per criterion):
•5.0: Fully satisfied with no errors
•4.0–4.9: Mostly satisfied with minor issues
•3.0–3.9: Partially satisfied with notable gaps or minor inaccuracies
•2.0–2.9: Poorly satisfied with significant omissions or errors
•1.0–1.9: Barely satisfied with major problems
•0.0–0.9: Not satisfied – critical failures or fabrications
Score each of the 5 criteria independently. If a criterion is not applicable, assign 5. Report the average as the final score.
Inputs:
Source Text: {}
Extracted Definitions: {}
Prompt 6: Judge 2 evaluation prompt for definitions (5 criteria). Output scores and critiques are consumed by Prompt 3.
29

Preprint. Under review.
Prompt 7|Judge 3: Rule Unit Evaluation
You are an expert evaluator assessing structured rule extraction from regulatory documents. You will receive the original source text and an extracted RuleUnit.
Evaluate each component based on the following criteria:
1. Completeness
Core components ( label,rule_type ,targets,action,action_object ) are significant if missing. Secondary components ( method,constraints ,conditions ) and
optional fields when absent are minor issues.
2. Conciseness
The label should be reasonably brief, summarizing the rule while preserving important meaning. Slight wordiness is acceptable. Significant deviation from the rule’s
meaning or scope should be noted.
3. Accuracy
Therule_type should reasonably represent the type of rule in the source. Minor classification judgment calls are acceptable. Significant misclassification that
misrepresents the rule’s fundamental nature is a problem.
4. Consistency
Targets should reasonably align with the source. Minor terminology variations that preserve meaning are acceptable. Targets should not significantly contradict or
misrepresent who the rule applies to.
5. Fidelity to Source Text
Statement components should reasonably reflect the source. Conditions, exceptions, and constraints should capture the primary requirements. Minor deviations in
phrasing that do not affect legal interpretation are acceptable. Significant alterations of scope or omission of critical qualifiers should be noted.
6. Neutrality
Labels and statements should present the source in a balanced manner. Minor interpretive choices are acceptable. Significant bias or misrepresentation of the rule’s
intent should be noted.
7. Actionability
Action and action_object should provide reasonably clear guidance, understandable and usable in a business context. Minor ambiguity is acceptable if the core action
and object are identifiable. Excessive abstraction that obscures what must be done is problematic.
8. Non-Hallucination
Extract only rule units present in the source. Do not invent rule components, conditions, targets, or context not grounded in the text. This criterion is strict: fabricated
content is a significant problem. Reasonable interpretations of existing content are acceptable.
Scoring Guidelines (per criterion):
•5.0: Fully satisfied with no errors
•4.0–4.9: Mostly satisfied with minor issues
•3.0–3.9: Partially satisfied with notable gaps or minor inaccuracies
•2.0–2.9: Poorly satisfied with significant omissions or errors
•1.0–1.9: Barely satisfied with major problems
•0.0–0.9: Not satisfied – critical failures or fabrications
Score each of the 8 criteria independently. If a criterion is not applicable, assign 5. Report the average as the final score. Focus on significant errors that materially
affect accuracy, completeness, or usability of the extracted rule.
Inputs:
Source Text: {}
Extracted RuleUnit: {}
Prompt 7: Judge 3 evaluation prompt for per-rule quality (8 criteria). Output scores and critiques are consumed by
Prompt 4.
G.4 Question Generation Prompt
The following prompt is used to generate evaluation questions for Experiment 3 (Section 4.4). Qwen3-VL-
8B-Instruct is prompted with HIPAA regulatory document to produce compliance-grounded questions fully
answerable from the passage alone, spanning factual, conditional, and analytical types. The question count
and passage content are injected at the{}placeholders.
Prompt 8|Question Generation: Downstream RAG Evaluation
You are a senior legal analyst and compliance expert with over two decades of experience specializing in U.S. healthcare law, HIPAA regulations (45 CFR Parts 160
and 164), and regulatory compliance frameworks.
Areas of Expertise:
•HIPAA Privacy Rule, Security Rule, and Breach Notification Rule
•Protected Health Information (PHI) handling, use, and disclosure requirements
•Covered entities, business associates, and their obligations
•Administrative, physical, and technical safeguards
•Enforcement, penalties, and compliance procedures
Your role is to generate precise, grounded, and legally meaningful questions from regulatory passages. You think like both a compliance officer and a litigator: you
understand nuance, edge cases, and the practical implications of regulatory language. Given the following regulatory passage, generate exactly {} questions that a
compliance officer, legal analyst, or auditor might ask.
Strict Rules:
•Every question must be directly and fully answerable from the passage alone
30

Preprint. Under review.
•Do NOT introduce concepts, entities, or scenarios not present in the passage
•Do NOT ask questions requiring outside knowledge or inference beyond the passage
•Questions must be diverse: cover who, what, when, how, and under what conditions
•Questions must be specific and unambiguous
•Questions should vary in complexity: mix factual, conditional, and analytical types
•Each question must stand alone as a complete, clear sentence
Output Format:
Return ONLY a valid Python list of {} strings. No explanation, no preamble, no commentary.
Example:
["Question one?", "Question two?", ...]
Input:
Passage: {}
Prompt 8: Question generation prompt for downstream evaluation on compliance QA via RAG (Experiment 3). Injected
with the target passage and desired question count; output is a Python list used to construct the 100-question evaluation
set.
31

Preprint. Under review.
H Extraction Schema
De Jure structures LLM outputs using guided JSON generation, with Pydantic-defined schemas enforcing
the extraction of discrete rule units from the source text.
Schema 1|SectionExtraction(top-level)
Top-level container returned by the extraction pipeline. All sub-objects useadditionalProperties: false.
Root fields
section_metaSectionMetareqMetadata about the regulatory section
definitionsDefinitionEntry[ ]opt Defined terms extracted from the section
extracted_rulesRuleUnit[ ]reqList of structured rule extractions
SectionMeta
schema_versionstringopt (“1.0.0”) Version of the extraction schema
section_citestringreqCitation of the section
titlestringreqTitle of the section
effective_datesEffectiveDate[ ]opt Timeline of adoptions, amendments, rescissions
notesstringopt Additional context
x_extensionsobjectopt Custom extension namespace
EffectiveDate
eventenumreqadopted|amended|rescinded|note
datestringreqISO date, e.g.2023-02-27
fr_citationstringopt Federal Register citation, if available
DefinitionEntry
termstringreqThe term being defined
textstringreqDefinition text from source
scopeenumoptsection|part|act|context-dependent ; leave null
if unstated
cross_referencesCrossRef[ ]opt Related regulatory references
Schema 1: Top-level extraction container with section metadata and definitions. All objects enforce strict schemas
(additionalProperties: false). Required (’req’) and Optional (’opt’) categories are tagged separately.
Schema 2|RuleUnit
A single extracted regulatory rule with type classification, targets, decomposed statement, citations, and judge scores populated post-evaluation.
RuleUnit
rule_idstringreqUnique identifier for this rule
labelstringreqShort summary of the rule
rule_typeRuleTypereqClassification of the rule
targetsTarget[ ]reqEntities subject to the rule
statementStatementreqDecomposed regulatory requirement
citationsCitationsopt Supporting regulatory citations
judge_scoreJudgeScoreopt Aggregated evaluation scores
examplesstring[ ]opt Examples lifted from source text, if available
RuleType
typeenumreqobligation|prohibition|permission|exemption
|definition-application|safe-harbor
|procedure|clarification|deeming|
condition-precedent|other
other_labelstringopt Required when type=‘other’ ; descriptive label for cus-
tom type
Target
The entity/role subject to the obligation, prohibition, or permission — WHO must comply, not the recipient or intermediary.
rolestringreqRole/entity subject to the rule
qualifiersstringopt Narrowing qualifiers (e.g. “foreign private issuer”)
Citations
textstring|string[ ]opt Textual citation(s) to supporting sources
Schema 2: Individual rule unit schema with detailed type taxonomy (11 categories), target identification, and citation
support.
32

Preprint. Under review.
Schema 3|Statementwithin a rule unit
Decomposition of a regulatory requirement into action, object, method, constraints, conditions, exceptions, and verbatim source text.
Statement
actionstringreqPrimary regulatory action as verb phrase
action_objectstringopt Direct object or recipient of the action
methodstringopt How the action must be performed
constraintsConstraint[ ]reqLimits imposed on applying this rule (default[])
conditionsCondition[ ]reqPrerequisites for this rule to apply (default[])
exceptionsExceptionItem[ ]reqPlaces where this rule may not apply (default[])
penalties_or_PenaltiesOr-opt Penalties or consequences
consequencesConsequences[ ]of a rule action
purposestringopt Stated purpose/objective if explicit in source
verbatimstringreqExact quoted text from the source
Constraint
textstringreqConstraint description
applies_tostringopt Entity the constraint binds to
Condition
triggerstringreqTriggering event or conditional text
time_windowTimeWindowopt Temporal boundaries for the condition
cross_referencesCrossRef[ ]opt Related regulatory cross-references
TimeWindow
startstringopt Start of the time window
endstringopt End of the time window
zoneenumoptET|EST|EDT
ExceptionItem
textstringreqMain exception description
cross_referencesCrossRef[ ]opt Related regulatory references
PenaltiesOrConsequences
textstringreqPenalty/consequence description
cross_referencesCrossRef[ ]opt Related regulatory references
CrossRef
typeenumreqCFR|Rule|Form|USC|Release|Regulation|
Note|Other
citestringreqCitation string
summarystringopt Brief summary of the cross-reference
Schema 3: Statement decomposition with conditions, constraints, exceptions, penalties, and cross-references. Arrays default
to[]; optional scalars default tonull.
Schema 4|Judge Evaluation Schemas(Judges 1, 2, 3)
Multi-step evaluation framework for extraction quality. Each metric carries an integer Score∈[ 0, 5]and a textual Justification . All fields in every judge step are
required.
Metric(shared unit)
Scoreintegerreq0≤Score≤5
JustificationstringreqTextual justification for the score
Step1Judge— Section-level metadata extraction
CompletenessCoverage of all required metadata fields
Fidelity_to_source_textFaithfulness to the original text
Non_hallucinationAbsence of fabricated content
Title_QualityAppropriateness and clarity of the extracted title
Precision_of_Citations_and_DatesAccuracy of citation strings and date formats
Reasonable_Population_of_Optional_FieldsSensible use of optional fields
Step2Judge— Definition extraction
CompletenessAll defined terms captured
Fidelity_to_Source_TextDefinitions faithful to source
No_Hallucination_or_FabricationNo invented terms or definitions
Precision_and_FormattingCorrect formatting and precision
Quality_of_TermsAppropriateness and clarity of extracted terms
33

Preprint. Under review.
Step3Judge— Per-rule unit evaluation
CompletenessAll rule components present
ConcisenessLanguage is brief while preserving meaning
AccuracyRule type correctly classifies the source
ConsistencyTargets align with the source
Fidelity_to_source_textStatement reflects the source faithfully
NeutralityBalanced, unbiased presentation
ActionabilityClear, usable guidance with reasonable abstraction and minimal ambiguity
Non_hallucinationNo fabricated rule components
JudgeScore(aggregate)
step1Step1JudgereqStep 1 evaluation
step2Step2JudgereqStep 2 evaluation
step3Step3JudgereqStep 3 evaluation
Finalintegeropt Overall (average) score∈[0, 5]
Notesstringopt Free-text evaluator notes
Schema 4: Three-step judge framework — Step 1 evaluates metadata (6 metrics), Step 2 evaluates definitions (5 metrics),
Step 3 evaluates each rule unit (8 metrics). All metrics use the shared Metrictype with Score∈[ 0, 5]andJustification .
I Sample Evaluation Questions
Table 15 presents a representative sample of 10 questions from the 100-question evaluation set used in
Experiment 3 (Section 4.4), generated by Qwen3-VL-8B-Instruct from HIPAA regulatory passages using the
prompt in Appendix G.4. The questions span five distinct HIPAA topic areas (general use and disclosure
standards, permitted uses, business associate obligations, genetic information protections, and reproductive
health privacy) and mix factual, conditional, and analytical question types, collectively stress-testing whether
a RAG system can retrieve and reason over structurally and semantically diverse regulatory provisions.
# Question
1Who may not use or disclose protected health information except as permitted or required by
this subpart?
2When is a covered entity permitted to use or disclose protected health information for treatment,
payment, or health care operations?
3What conditions must be met for a use or disclosure to be considered incident to a permitted
use or disclosure?
4 What are the required disclosures of protected health information by a covered entity?
5What are the limitations on a business associate’s use and disclosure of protected health
information?
6What is prohibited regarding the use and disclosure of genetic information for underwriting
purposes?
7 What activities are considered “underwriting purposes” with respect to genetic information?
8 What constitutes a “sale of protected health information”?
9Under what conditions does the prohibition on using reproductive health care information apply?
10What is presumed about the lawfulness of reproductive health care provided by another person,
and what can override that presumption?
Table 15: Ten representative questions from the 100-question evaluation set used in Experiment 3. Each
question is fully answerable from its source passage alone. The set spans five HIPAA topic areas and three
question types, reflecting the diversity requirements specified in the generation prompt (Appendix G.4).
34
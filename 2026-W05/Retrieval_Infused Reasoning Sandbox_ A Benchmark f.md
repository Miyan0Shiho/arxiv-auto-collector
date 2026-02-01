# Retrieval-Infused Reasoning Sandbox: A Benchmark for Decoupling Retrieval and Reasoning Capabilities

**Authors**: Shuangshuang Ying, Zheyu Wang, Yunjian Peng, Jin Chen, Yuhao Wu, Hongbin Lin, Dingyu He, Siyi Liu, Gengchen Yu, YinZhu Piao, Yuchen Wu, Xin Gui, Zhongyuan Peng, Xin Li, Xeron Du, Libo Qin, YiXin Cao, Ge Zhang

**Published**: 2026-01-29 16:26:19

**PDF URL**: [https://arxiv.org/pdf/2601.21937v1](https://arxiv.org/pdf/2601.21937v1)

## Abstract
Despite strong performance on existing benchmarks, it remains unclear whether large language models can reason over genuinely novel scientific information. Most evaluations score end-to-end RAG pipelines, where reasoning is confounded with retrieval and toolchain choices, and the signal is further contaminated by parametric memorization and open-web volatility. We introduce DeR2, a controlled deep-research sandbox that isolates document-grounded reasoning while preserving core difficulties of deep search: multi-step synthesis, denoising, and evidence-based conclusion making. DeR2 decouples evidence access from reasoning via four regimes--Instruction-only, Concepts (gold concepts without documents), Related-only (only relevant documents), and Full-set (relevant documents plus topically related distractors)--yielding interpretable regime gaps that operationalize retrieval loss vs. reasoning loss and enable fine-grained error attribution. To prevent parametric leakage, we apply a two-phase validation that requires parametric failure without evidence while ensuring oracle-concept solvability. To ensure reproducibility, each instance provides a frozen document library (drawn from 2023-2025 theoretical papers) with expert-annotated concepts and validated rationales. Experiments across a diverse set of state-of-the-art foundation models reveal substantial variation and significant headroom: some models exhibit mode-switch fragility, performing worse with the Full-set than with Instruction-only, while others show structural concept misuse, correctly naming concepts but failing to execute them as procedures.

## Full Text


<!-- PDF content starts -->

Retrieval-Infused Reasoning Sandbox: A Benchmark for
Decoupling Retrieval and Reasoning Capabilities
ByteDance Seed, M-A-P
Full author list in Contributions
Abstract
Despite strong performance on existing benchmarks, it remains unclear whether large language
models can reason over genuinely novel scientific information. Most evaluations score end-to-end
RAG pipelines, where reasoning is confounded with retrieval and toolchain choices, and the signal is
further contaminated by parametric memorization and open-web volatility. We introduceDeR2, a
controlled deep-research sandbox that isolates document-grounded reasoning while preserving core
difficulties of deep search: multi-step synthesis, denoising, and evidence-based conclusion making.
DeR2decouples evidence access from reasoning via four regimes—Instruction-only, Concepts
(gold concepts without documents), Related-only (only relevant documents), and Full-set (relevant
documentsplustopicallyrelateddistractors)—yieldinginterpretableregimegapsthatoperationalize
retrieval loss vs. reasoning loss and enable fine-grained error attribution. To prevent parametric
leakage, we apply a two-phase validation that requires parametric failure without evidence while
ensuring oracle-concept solvability. To ensure reproducibility, each instance provides a frozen
document library (drawn from 2023–2025 theoretical papers) with expert-annotated concepts
and validated rationales. Experiments across a diverse set of state-of-the-art foundation models
reveal substantial variation and significant headroom: some models exhibit mode-switch fragility,
performing worse with the Full-set than with Instruction-only, while others show structural concept
misuse, correctly naming concepts but failing to execute them as procedures.
Date:January 30, 2026
Correspondence:Ge Zhang atzhangge.eli@bytedance.com
Project Page:https://retrieval-infused-reasoning-sandbox.github.io/
1 Introduction
Recent foundation models have advanced from fluent assistants to strong problem solvers, achieving high
performance on difficult, objectively graded tasks such as competition-style mathematics (e.g., AIME) and
real-world software engineering (e.g., SWE-bench) [ 14,22]. Meanwhile, agents—autonomous systems that
navigate the internet to acquire and consolidate evidence—are increasingly used to extend models beyond
parametric knowledge by coupling external search with internal reasoning and synthesizing information across
multiple sources [ 8,20]. Within this broader shift, deep search has emerged as a foundational evaluation
scenario: it stresses an agent’s ability to uncover hard-to-find facts or entities through multi-step exploration,
denoising, and evidence-driven synthesis. Motivated by these demands, recent work has proposed stronger
agentic designs [ 13,17–19,28] and introduced more challenging benchmarks that better capture deep-research
behavior, including BrowseComp for hard-to-find web browsing tasks and HLE for frontier, expert-level
closed-ended questions [24, 32].
However, we find that current deep-research evaluation practice suffers from three major limitations.(1)
1arXiv:2601.21937v1  [cs.AI]  29 Jan 2026

(a) Closed-book QA
 (b) RAG Pipeline
relatedDoc noiseDoc
concept
concept
concept
(c) Decoupled Retrieval and Reasoning Benchmark
Instruction
OnlyConcepts 
OnlyFull DocSetRelated 
Only
Frontier 
Question
Figure 1Closed-book QA measures intrinsic knowledge, while standard RAG entangles retrieval and reason-
ing—distractors can hide whether errors come from bad retrieval or failed evidence-based reasoning.DeR2decouples
the two by evaluating the same question under controlled inputs (instruction/concepts/related/full), isolating failure
causes.
End-to-end pipeline confounding with weak diagnosis:many benchmarks score the full stack—retrieval,
reranking, summarization/compression, context stitching, and reasoning—so variance is often dominated by
toolchain choices rather than the model’s latent research reasoning capability. Moreover, most setups expose
only a binary “correct/incorrect” outcome, which makes it difficult to attribute errors to distinct failure types
such as missing the key concept, identifying the concept but failing to apply it, or being diverted by topically
related noise that shifts the reasoning starting point and triggers mode-switch derailment. As a result, these
evaluations provide limited guidance for model selection, debugging, or targeted training.(2) Parametric
leakage / memorization:if tasks can be solved via memorized facts, broad encyclopedic knowledge, or easily
guessable numeric forms, then correctness does not reliably indicate that the model consulted the crucial
evidence and reasoned with it, contaminating the evaluation signal.(3) Web volatility and irreproducibility:
benchmarks that depend on open-web retrieval inherit non-stationarity across time, region, indexing updates,
and page instability (movement, deletion, or content changes), producing high variance across repeated runs
and hindering controlled comparisons and ablations [32].
To address these limitations, we proposeDeR2, a controlled research sandbox that isolates document-
grounded reasoning while preserving the core difficulties of deep search: multi-step synthesis, noise filtering,
and evidence-based conclusion making. We mitigate(1) pipeline confounding and weak diagnosisby decoupling
evidence access from reasoning via four evaluation regimes:Instruction-only(parametric knowledge),Concepts
(instruction + gold concept set; concept composition/scheduling without document extraction),Related-only
(instruction + only relevant documents; extraction and reasoning under clear evidence), andFull-set(relevant
documents plus topically related distractors; denoising and reasoning). This design yields interpretable regime-
to-regime gaps that operationalize “retrieval loss” versus “reasoning loss,” and it supports error attribution,
including missing concepts, concept misuse, and noise-induced mode switching. We mitigate(2) parametric
leakagethrough a rigorous two-phase validation protocol: each problem must exhibit parametric failure
(low accuracy without documents) while remaining document-solvable (a correct solution is achievable when
provided oracle concepts). This jointly enforces novelty (not answerable from memorized knowledge alone)
and tractability (answerable given the intended evidence). Finally, we address(3) web volatilityby replacing
open-web dependence with a frozen, per-instance document library (mean: 6.5 documents) drawn from
2023–2025 theoretical papers. Each library includes both necessary evidence and carefully selected distractors
that mimic realistic, imperfect literature gathering. Each instance further provides expert-annotated concept
sets and validated chain-of-thought rationales, enabling process-level evaluation of evidence usage rather than
relying solely on final-answer accuracy.
Our experiments reveal two consistent failure patterns.(1) Mode-switch fragility:Some models perform
better under Instruction-only than under Full-set, indicating that additional documents can disrupt the
controller responsible for shifting from parametric reasoning to evidence-grounded reasoning, thereby degrading
performance.(2) Structural concept misuse:even when models identify and restate the correct concepts,
2

a) Existing Benchmarks b) Retrieval-Infused Reasoning Sandbox
Closed-book QA
only evaluate intrisic knowledge
RAG Pipeline
reasoning-retrieval tangle 
masks failure causeInput: Mixed DocSet
Evaluation
Output
Why is the 
result wrong...
the latest 
research in 
2026 indicates...
Related Documents
Frotier Question: Does the addition of a 2,2′-bipyridine coating on Pt/C and Ir/C 
electrodes enhance or weaken the HER?
Noise Documents
EDG For a given set 
of target interatomic 
distances...Doc with concept A:
Lattice Matrix and 
Coordinate Consistency 
Condition...Doc with concept B: Noise Doc: Noise Doc:
irrelevant definations
...outmoded expositions
...
Instruction-Only Full DocSet RelatedDoc-Only Concepts-Only
concept A
concept B
concept N
Answer:
 The optimization objective 
function for equilibrium...CoT:
Step1: Given the lattice matrix �=... 
Step2: Given target equilibrium...
...Results
Failure in reasoning mode 
switching and concept execution.
...Figure 2 (a) Existing benchmarks.Closed-book QA only evaluates intrinsic (parametric) knowledge. RAG pipelines
couple retrieval and reasoning end-to-end, creating a reasoning–retrieval tangle that masks the true cause of failure.(b)
Decoupled Retrieval and Reasoning Benchmark.DeR2provides a controlled setting that decouples retrieval from
reasoning by comparing Instruction-only, Full DocSet (related + noise), Related-only, and Concepts-only conditions,
enabling diagnosis of grounding, mode-switching, and concept execution errors.
they often fail to execute them as procedures (e.g., applying rules step-by-step or instantiating abstractions),
instead reverting to generic heuristics that appear plausible but produce incorrect conclusions.
Our contributions are threefold:
•We articulate desiderata for evaluating research-oriented LLMs and show that prior QA, RAG, and
reasoning benchmarks do not sufficiently isolate evidence-grounded synthesis.
•We releaseDeR2, a rigorously curated benchmark featuring frozen document libraries with embedded
noise, expert concept annotations, and validated CoT rationales drawn from frontier scientific literature.
•We provide a systematic analysis of leading LLMs under this controlled setup, revealing substantial
headroom even for top-tier systems and identifying concrete failure modes that future architectures and
training regimes must address.
2 Benchmark
2.1 Benchmark Overview
We introduceDeR2, a controlled benchmark for evaluating retrieval-infused reasoning in scientific problem
solving. The benchmark is designed to expose failure modes that are difficult to diagnose in end-to-end
retrieval-augmented generation (RAG) or agent-based deep research evaluations.
DeR2explicitly separates (a) a model’s ability to identifyandextractsolution-relevant concepts from a
document set, from (b) its ability to compose andschedule these concepts to derive the correct answer. This
decomposition enables controlled diagnosis of where performance collapses in retrieval-infused reasoning.
Compared to conventional RAG benchmarks,DeR2provides (i) a controlled document set with both relevant
anddistractordocuments, (ii)concept-levelsupervisionthatsupportscapabilityisolation, and(iii)averifiability
3

protocol ensuring that problems are not solvable by parametric memorization alone, but become solvable once
the right concepts are provided.
2.2 Task Definition
Each benchmark example asks the model to answer a frontier scientific question by grounding its reasoning
in a provided set of documents. The target capability is not merely recalling definitions, but performing
multi-step derivations that require selecting, instantiating, and coordinating multiple concepts.
Each example is defined by a tuple:
•Instruction: A high-level academic question derived from cutting-edge scientific papers. The question
should be solvable only if the model is provided with the correct concepts, making it impossible to
answer without the relevant knowledge.
•Concepts: All the concepts, theorems, or formulas necessary to solve the instruction. These should not
be overly specific experimental data, but rather generalizable scientific knowledge.
•CoT (Chain of Thought): The reasoning process that connects the concepts to the final answer, illustrating
how the model should use the provided knowledge step by step.
•Answer: The answer to the instruction, which could be a product, scalar value, or conclusion derived
from the concepts and reasoning.
•Doc Set: A collection of documents that provide the necessary concepts and theories to answer the
question. These documents include noise documents that are unrelated to the task, ensuring a balanced
challenge for the model.
DeR2emphasizes research-style reasoning, where the model must transform theoretical knowledge from
documents into actionable steps for solving complex scientific problems. The task focuses on concept
operationalization (e.g., theorem instantiation, algorithm execution) rather than simply summarizing surface-
level content.
2.3 Data Collection
We recruited annotators and reviewers from a highly selective pool: all contributors are current PhD students
enrolled at top-tier Chinese universities (Project 985). To reduce domain-mismatch errors, both annotators
and reviewers were strictly restricted to working onlywithin their own specialized academic fields. To attract
top-tier annotators, a competitive compensation package was offered, with each task rewarded at a rate of
2500 RMB per annotated question. Reviewers were compensated at 350 RMB per audited item. A total of 81
annotators across various scientific disciplines participated in the data collection process.
Step 1: Source paper acquisition and screening. Each annotation begins with collecting a sourcepaper
published in2023 between 2025. We place special emphasis on provenance: when a paper has an arXiv preprint,
we use the arXiv URL date as the publication timestamp for eligibility. We exclude appliedpapers—including
but not limited to physics/chemistry experiments, social surveys, and numerical simulations—because their
conclusions are often primarily experimental and do not yield instructions solvable via concept-driven logical
derivation. Eligible source papers must be theory-oriented , spanning (i) theoretical foundational disciplines,
(ii) theoretical engineering disciplines, and (iii) theory-driven interdisciplinary fields. As we know, the general
workflow of annotators involves selecting source papers by tracking leading research groups in each domain
and referencing materials published by prominent professors in the field.
Step 2: Constructing the quadruple(Instruction, Answer, Concepts, CoT). After an annotator identifies
an eligible source paper via abstract-level screening, they read the full paper to understand the problem
setting, assumptions, and core argument. They then construct: (i) anInstructionaligned with the paper’s
primary research question, with a scope that is neither overly broad (admitting multiple valid answers) nor
overly narrow (only true within the specific paper instance); (ii) a completeAnswerthat directly resolves the
Instruction, prioritizing a short artifact,scalar, orconciseconclusion . If the Answer is necessarily long, the
annotator must provide a rigorouschecklistof key scoring items to make verification objective; (iii) a set of
4

Reviewer
Challenge 
Question?
PreciseSolution?
AnnotatorStep1: Find Source Paper
Frontier Source Paper
 (2023-2025)
InstructionConcepts CoT AnswerStep2: Construct Items
Offline AI TestNo Concepts
❌ ❌ ❌ 
With Concepts
✅ ✅ ❌ 
Step3: Difficulty 
Control
   Step4: Collect Doc Set
ref
related documents noise documents
The paper’s year 
is unacceptable.       OK，
i will change 
the source
     paper
Experimental data 
do not qualify as 
concepts.Figure 3 Pipeline for constructing DeR2data:PhD annotators gather theoretical papers from academic platforms,
extracting instructions, concepts, CoT, and concise answers. These questions are designed to be challenging and
represent cutting-edge academic problems. During the difficulty control phase, offline AI models must fail to answer
correctly three times without concepts, but have a chance to answer correctly when provided with the relevant concepts.
When the difficulty control conditions are satisfied, retrieve the related documents and noise documents from the
references of the source paper. Finally, the reviewer conducts logical quality inspection and format quality inspection
on the submitted data, and verifies the difficulty control conditions again.
Conceptscapturing all and only the necessary theoretical concepts/theorems/formalisms required to solve
the Instruction (not incidental experimental measurements); (iv) a step-by-stepCoT(reasoning trace) from
Instruction to Answer, derived from the paper’s theoretical argumentation, ensuring that every knowledge
item used in the CoT is included in the Concepts list.
Annotators may use AI tools for brainstorming or summarization, but the Instruction and Answer must
be finalized by the annotator after fully understanding the paper. We explicitly observed two recurring AI
failure modes: (a) selecting peripheral intermediate quantities as the Answer, yielding an Instruction that
only holds inside the specific paper context; (b) generating overly open-ended Instructions where the provided
Answer becomes only one of many plausible solutions. Therefore, human authorship is required for the final
Instruction–Answer pair.
Step 3: Difficulty calibration under offline LLM testing. To ensure that each item is sufficiently frontier and
non-trivial, we perform controlled difficulty checks using offlinemodels (e.g., DeepSeek-R1-0528 or the Doubao
web client with networking disabled). We run each condition three times and record correctness:
•Instruction-onlythe model answers the Instruction without Concepts. All three attempts must be
incorrect.
•Concepts-onlythe model answers with Concepts provided. Across three attempts, it must be correct
at least once and incorrect at least once.
If the model is correct in all three Instruction+Concepts trials, we apply one of two remedies. Option A
adds adeepvalidation step by replacing Concepts with a document set (Related + Noise) and retesting;
if the model remains always-correct, the item is discarded, whereas if it fails at least twice, the item can
be retained (this option is discouraged when the doc set exceeds three PDFs due to context limitations in
common LLM interfaces). At this point, option B can still be adopted, whereby annotators can iteratively
5

Figure 4Distribution of problem domains and answer types, showing the disciplinary coverage of the benchmark and
the structural diversity of target answers.
edit the Instruction and/or Concepts (by adding/removing concepts or narrowing the question scope) to
increase the difficulty until the calibration criteria are satisfied.
Step 4: Document set construction. Once the quadruple passes difficulty calibration, the annotator constructs
the document set. Starting from the source paper’s references, the annotator locates at least oneRelated
document for each Concept (a single related document may cover multiple Concepts). The annotator then
curates severalNoisedocuments that are topically adjacent to the Instruction but do not contribute solution-
relevant concepts. For all documents, the annotator must verify: (i) Related documents contain the needed
Concepts; (ii) Noise documents do not contain any solution Concepts while remaining on-topic; and (iii) no
document contains the Answer verbatim or in an equivalent directly-recoverable form.
Review and quality assurance(QA). Reviewers are selected from annotators withat least three accepted
annotations. Each reviewed item is checked for both format compliance and scientific validity: (i) the
Instructionisafrontieracademicquestiongroundedina2023–2025theory-orientedsourcepaper, self-contained,
and appropriately scoped; (ii) the Answer is unique, unambiguous, and easy to verify (or accompanied by
a strict checklist when long); (iii) the Concepts are necessary knowledge items for solving the Instruction,
contain no irrelevant content, and are theoretical constructs rather than paper-specific experimental values;
(iv) the CoT is a coherent concept-grounded derivation that uses allConcepts (no omissions), does not rely
on experimental measurements as premises, and yields a unique conclusion consistent with the Answer; and
(v) the difficulty calibration tests are re-run by the reviewer and the document set is re-audited to confirm
that no document leaks the Answer.
Collaboration loop.Annotators and reviewers collaborated in a shared online document workspace with
high-frequency feedback. This iterative workflow enabled the rapid correction of ambiguity, scope issues,
concept leakage, and document contamination, thereby improving consistency and ensuring that all accepted
items met the benchmark’s structural and scientific requirements.
3 Experiments
3.1 Evaluation Settings
We evaluate models under four controlled input configurations that explicitly decouple retrieval capability
(selecting and extracting solution-relevant evidence from long contexts) from reasoning capability (composing
6

and scheduling the extracted concepts to derive the final answer). The key idea is to provide the model with
progressively more information — ranging from none to oracle concepts, to clean evidence, and finally to
evidence mixed with distractors — so that performance gaps can be attributed to specific failure sources
rather than being conflated in an end-to-end score.
We consider the following four settings, each evaluated independently (see 1).
These settings enable fine-grained diagnosis of three recurring failure sources: (i)Knowledge loss(Concepts-
only→Instruction-only gap), (ii)Retrieval loss(Related-only →Concepts-only gap), and (iii)Noise-induced
loss(Full-set →Related-only gap). To ensure the decoupling is meaningful, we keep the input format and
evaluation procedure identical across settings; only the provided information differs.
3.2 Models and Evaluation Protocol
We evaluate a diverse set of state-of-the-art models. Concretely, our evaluated pool includes GPT-5.1 [ 21],
Gemini-3-Pro [ 11] and Gemini-2.5-Pro [ 6], Claude-Opus-4.1 [ 2] and Claude-Sonnet-4.5 [ 3], DeepSeek-V3.1
and DeepSeek-V3.2 [ 7], Doubao-1.6 and Doubao-1.8 [ 4], Qwen-3-max [ 33], Moonshot-Kimi-k2-thinking [ 30],
and GLM-4.6 [ 1]. We aim to evaluate the retrieval-enhanced reasoning capabilities of different models in a
controlled sandbox environment, rather than optimizing for any specific vendor stack.
Controlled evaluation interface.All models are evaluated under the same protocol: each model receives
a fixed input instance and produces a single prediction, with no web access. All retrieval signals must
originate solely from the provided document set (Related-only / Full-set) or the oracle concept list (Concepts).
This design enforces the intended capability decoupling between (i) knowledge selection and de-noising, (ii)
document-to-concept extraction, and (iii) concept-level composition and scheduling.
Because some models cannot reliably ingest the full document set (e.g., due to context-length limits that
can lead to runtime errors or empty generations), we apply a lightweight pre-processing utility to ensure fair
evaluation across models.
Specifically, we use a deterministic long-text truncation rule that takes an input string content and a
maximum character budget max_chars (default: 30000). If max_chars isNone, negative, or the input length
is below the limit, the utility returns content unchanged. Otherwise, it sets half = max_chars // 2 , keeps
the first halfcharacters and the last halfcharacters, and inserts a fixed marker line in between: [...
Observation truncated due to length ...]. Newlines are added on both sides of the marker.
To reduce evaluation overhead while keeping decoding conditions consistent across models, we use a single
shared sampling configuration: temperature = 1 and nucleus sampling with p= 0.7(no additional decoding
constraints beyond the API defaults). Each model-setting pair is run twice, and we report the average score
over the two runs.
To minimize prompt-induced artifacts, we never instruct the model to quote documents verbatim; instead, we
ask for concise, verifiable final answers consistent with the benchmark’s answer format. All raw generations
are logged verbatim. Answers are evaluated by doubao-seed-1-6-251015, using a fixed evaluation prompt
provided in Appendix A.
Setting Problem Concepts Documents Capability
Instruction-only✓- - parametric knowledge
Concepts-only✓ ✓(oracle) - concept-level composition
and scheduling reasoning
Related-only✓-✓(related doc) knowledge selection and
reasoning
Full-set✓-✓(related+noise doc) knowledge selection, de-
noising and reasoning
Table 1Evaluation settings with different input configurations
7

We report the answer-level accuracy score as the primary evaluation metric. A prediction is considered
correct if it matches the ground-truth answer according to task-specific normalization rules, including numeric
tolerance, symbolic equivalence, or checklist-based matching for structured answers.
Additionally, we use a simple retrieval loss, defined as the difference between the Concepts-only score and
the Full-set score: Retrieval Loss (RLoss) =Score Concepts-only −Score Full-set .A larger retrieval loss indicates
greater degradation due to retrieval.
3.3 Results and Analysis
We present results in three layers: (i) overall accuracy across the four evaluation settings, (ii) controlled factor
analyses that stress specific retrieval–reasoning bottlenecks, and (iii) qualitative CoT-based error attribution.
Overall performance and setting-wise gaps.Table 2 reports answer-level accuracy for each model under
Instruction-only , Concepts-only , Related-only , and Full-set . Across models, we observe a consistent
orderingConcepts-only(avg. 75.39) >Related-only(avg. 62.89) >Full-set(avg. 51.25), indicating that (i)
providing gold concepts substantially improves outcomes by removing concept-extraction errors, (ii) concept
extraction from documents remains a major source of loss, and (iii) adding distractor documents further
degrades performance through evidence selection failures. Notably, we also observe a counter-intuitive but
recurrent phenomenon whereInstruction-only(avg. 55.89) can match or even exceedFull-set(avg. 51.21) for
most models, suggesting that the performance drop is not merely attributable to noise volume but to a shift
in the model’s reasoning regime once external context is introduced.
Decomposing loss: retrieval vs. reasoning.The controlled settings allow us to attribute end-to-end degradation
to two interpretable gaps: document-to-concept loss(the performance drop fromConcepts-onlytoRelated-
only) and noise-induced loss(the drop fromRelated-onlytoFull-set). The first gap captures failures
torecoverandusethe required concepts from evidence—including locating the right passages, extracting
the relevant conceptual content, and operationalizing it as executable reasoning steps. The second gap
isolates robustness to distractors: given that the necessary evidence is present, it measures whether the model
can ignore topic-adjacent noise and sustain a correct, evidence-conditioned reasoning trajectory. Overall,
this decomposition matches the benchmark’s intended separation of (i) evidence selection / de-noising, (ii)
structured concept extraction from long-context documents, and (iii) concept-level coordination and scheduling
in multi-step reasoning.
Controlled factor analysis.We next stratify instances by dataset factors that are designed to surface distinct
failure modes.
Noise documents.As the number of noise documents increases,Full-setperformance exhibits a gradual
but non-linear decline, consistent with the hypothesis that distractors do not simply ’dilute’ signal, but can
redirect early-step abstraction and cause irreversible trajectory drift, as illustrated in Fig.6. We additionally
compare cases where models retrieve relevant concepts but fail to usethem: A frequent pattern is that the
model cites superficially relevant evidence while executing a generic template solution, indicating evidence is
present but not integrated into the causal chain of inference.
Required concept count.Accuracy decreases as the number of required concepts grows, even under the
Concepts-onlysetting. This shows that providing all necessary concepts does not eliminate errors: models
still struggle with concept-level working memory, dependency tracking, and coordinating multiple constraints.
The effect is amplified inConcepts-onlyandFull-set, where concept extraction errors and spurious concept
injection further enlarge the hypothesis space and increase coordination burden.
Reasoning depth and answer type.We bucket problems by reasoning depth (e.g., number of steps in
the reference CoT) and by answer type (e.g., formula derivation, numeric, boolean, or conclusion-style).
Deeper reasoning typically yields larger gaps betweenConcepts-onlyand document-based settings, reflecting
compounding failures in multi-step grounding and intermediate-state maintenance. In parallel, formula-
derivation items benefit the most from explicit concepts, suggesting that document-based settings often fail at
8

(a) Distribution of Doc Quantities  (b) Distribution of Concept and Step Quantities
Figure 5Structural and contextual complexity of the dataset.(a)Reports the distributions of reasoning step counts
and required concept counts per problem, reflecting multi-step and multi-concept reasoning demands.(b)Presents the
distributions of related documents, noise documents, and total documents per instance, characterizing the degree of
retrieval difficulty and noise exposure in the document sets. Together, these statistics illustrate the controlled diversity
of DeR2across reasoning depth, conceptual load, and retrieval conditions.
operationalizing constructive or procedural concepts (e.g., algorithm instantiation, theorem application) even
when high-level definitions are recognized.
Chain-of-thought and error attribution.To move beyond accuracy, we analyze predicted CoTs against
reference concepts and reasoning structure. We categorize failures into three dominant modes:
(1) Reasoning mode switching failure.When external documents are provided, models often abandon a
viable parametric reasoning path but fail to reliably anchor a new evidence-driven chain. This produces
the paradoxicalInstruction-only ≥Full-setbehavior on a non-trivial subset (e.g., Gemini-3-Pro-Preview
scores 64.2 under Instruction-only , but only 53.7 when provided with more knowledge in the Full-set ;
Claude-Opus-4.1-thinking scores 49.3 under Instruction-only , but only 40 in the Full-set ; similar patterns
are observed for models like OpenAI-GPT-5.1-high, DeepSeek-V3.2-Exp-Thinking, etc.) and indicates that
switchcontrol between internal-knowledge reasoning and context-grounded reasoning is itself a bottleneck.
(2) Structural retrieval errors.Errors frequently arise not from missing definitions but from failing to execute
constructive mechanisms described in documents (e.g., algorithmic steps, instantiation details, auxiliary-
variable constructions). Models replacethesemechanisms withfamiliarheuristics , yielding CoTs that are
locally coherent but globally incorrect.
(3) Concept coordination breakdown.Even with correct concepts available (inConcepts), models sometimes
fail due to poor coordination: They activate only a subset of necessary constraints, apply concepts out of
order, or lose track of intermediate invariants. This highlights that retrieval-infused reasoning is not a simple
’retrieval + reasoning’ composition; it requires explicitscheduling andmulti-constraint control.
Table 3 presents a detailed breakdown of the main failure modes at the reasoning chain level, covering four
controlled evaluation settings. Each sub-table within the main table includes two key types of information:
first, the concept-level performance metrics (precision, recall, and accuracy) calculated via a model-assisted
annotation protocol; second, the distribution of error types extracted from the predicted reasoning trajectories
of evaluated models.
For the concept-level metrics, the calculation process follows a strict workflow: we first use the model
doubao-seed-1.6-251015 to extract the set of concepts explicitly applied by each evaluated model from
its predicted chain-of-thought ( pred_cot ). Then, with the corresponding ground-truth concept set ( true_-
9

Model Instruction-only Full-set Related-only Concepts-only RLoss
OpenAI-GPT-5.2-high65.8 71.1 71.4 83.812.7
Gemini-3-Pro-Preview 64.2 53.7 68.3 80.9 27.2
Gemini-3-Flash-Preview 60.5 66 .0 64.8 78.312.3
OpenAI-GPT-5.1-high 59.8 57.0 66.9 81.4 24.4
DeepSeek-V3.2-Exp-Thinking 57.6 49.3 61.3 75.3 26.0
Moonshot-kimi-k2-thinking 55.9 52.7 65.0 74.4 21.7
Gemini-2.5-Pro 54.0 51.5 66.6 78.5 27.0
GLM-4-6 53.9 43.2 57.7 70.8 27.6
QwenAPI-3-max-0923 53.2 41.8 61.1 70.6 28.8
Claude-Sonnet-4.5 53.2 45.2 59.1 74.1 28.9
Doubao-1.8-1228-high 52.3 52.0 64.8 74.3 22.3
DeepSeek-V3.1-terminus-thinking 52.2 50.2 62.1 69.8 19.6
Doubao-1.6-1015-high 50.6 43.3 59.4 70.8 27.5
Claude-Opus-4.1-thinking 49.3 40.0 52.0 72.4 32.4
Average Score 55.9 51.2 62.9 75.4 24.2
Table 2Answer-level accuracy (%) under four controlled evaluation settings, ranked by RE score.Boldindicates best
performance, underline indicates second-best per column. Instruction-only reflects parametric knowledge; Concepts-only
provides an empirical upper bound on concept-level reasoning; Related-only and Full-set evaluate retrieval-conditioned
reasoning under weak and noisy evidence, respectively.
concepts ) as a reference, the same doubao-seed-1.6-251015 model is prompted to align the predicted and
reference concepts, and output the counts of true positives (TP), false positives (FP), and false negatives (FN).
Precision and recall are computed using standard formulas based on these counts, while concept accuracy is
derived by aggregating the per-instance performance results across the entire dataset.
For the error type distribution, the data is collected through a targeted sampling and analysis process: for
each evaluated model and each of the four evaluation settings, we uniformly sample 50 instances where
the model’s final answer is incorrect (to focus on failure modes beyond simple concept omission). We then
usedoubao-seed-1.6-251015 as a structured evaluator, providing it with four inputs for each sampled
instance—the predicted concepts ( pred_concept ), predicted chain-of-thought ( pred_cot ), ground-truth
concepts ( true_concept ), and reference reasoning ( true_cot )—and prompt it to diagnose the root causes of
failure. Each incorrect instance is labeled with one or more of four predefined error categories:(1) reasoning-
process error(flaws in the logical flow of reasoning),(2) numeric or formalization error(mistakes in calculations
or symbolic representation),(3) missing core concept(failure to include a key concept required for the correct
answer), and(4) incorrect understanding or misuse of a core concept(misinterpreting a concept or applying it
in an inappropriate context). The full prompt used to guide the doubao-seed-1.6-251015 evaluator during
the error diagnosis process is provided in the appendix.A for reproducibility.
InInstruction-only, the errors mainly manifest as missing core concepts and issues with general reasoning
heuristics, reflecting the limitations of pure parametric knowledge. ByConcepts-only, overall accuracy
significantly improves, and errors related to concepts are largely eliminated; the remaining failures are
predominantly procedural, indicating that simply retrieving the correct concepts does not guarantee the
correct execution of multi-step reasoning.
It is further observed that in the Full-set , Related-only , and Instruction-only settings, the missing core
concept(MC) is the most frequent error. This aligns with the knowledge input scaling relationship, but
it is worth noting that the gap in the number of missing core concepts between Instruction-only and the
knowledge-providing settings (Full-set and Related-only ) is not large, with all hovering around 40, reflecting
a limitation in the model’s ability to retrieve the correct knowledge points, or the model sometimes does not
even know which concepts to extract to answer the current question. Additionally, Gemini-3-Pro only made
29 errors in the Related-only setting and delivered its optimal performance in Instruction-only (with merely
34 mistakes). This indicates that Gemini may possess more endogenous knowledge, enabling it to retrieve and
comprehend concepts effectively even with limited background information.
10

(a) Correlation Between Noise Doc and Score 
(b) Correlation Between Concept and Score  Number of Noise Documents Number of ConceptsAverage ScoreFigure 6 Impact of Document Characteristics on Model Performance.(a) Under theFull-setsetting, model
score as a function of the number of noise documents. (b) Under theFull-setsetting, model score as a function of the
number of concepts.
InConcepts-only, the model must recover concepts from less relevant evidence, and both concept omission
and reasoning errors reappear, revealing structural retrieval failure rather than simple definition gaps. Finally,
in theFull-setsetting, although some retrievals are successful, the presence of topic-related noise exacerbates
reasoning errors and misalignment of concepts.
Overall, these tables indicate that the performance degradation under retrieval-augmented settings is not
only due to missing concepts but also stems from failures in procedural and conceptual coordination during
evidence-conditioned reasoning.
Additionally, in the concepts-only setting, the model’s precision and recall both fail to reach 1, indicating that
the model cannot comprehend the provided unseen concepts, even when usage instructions for those concepts
are supplied.
4 Related Work
4.1 Existing Benchmarks
EarlyreadingcomprehensionbenchmarkssuchasSQuAD[ 25]andNaturalQuestions[ 16]establishedevaluation
protocols for extractive question answering over short passages. Multi-hop extensions such as HotpotQA [ 34]
increased reasoning complexity by requiring evidence aggregation across passages, though answers remained
primarily extractive. While recent variants incorporate more abstractive generation, evaluation metrics, and
task designs still tend to favor localized extraction over genuine multi-document synthesis [15].
Retrieval-augmented generation benchmarks, including KILT [ 23], and FreshQA [ 31], evaluate end-to-end RAG
pipelines over large corpora. By coupling retrieval with generation, they measure system-level performance,
but this coupling conflates multiple capabilities, making error attribution difficult: failures may stem from
retrieval gaps, reasoning deficits, or evidence integration issues. Moreover, many tasks emphasize factoid
lookup or temporal fact verification rather than multi-step synthesis that requires integrating novel concepts
across multiple documents.
In parallel, mathematical and formal reasoning benchmarks evaluate deductive capabilities largely in isolation
from document comprehension. Benchmarks like GSM8K [ 5], MATH [ 12], Omni-MATH [ 10] and Putnam-
AXIOM [ 9] and test multi-step problem solving with increasing complexity. Graduate-level benchmarks such
as GPQA [ 26] and broad suites like BIG-Bench [ 27] increase difficulty but remain predominantly parametric:
models rely on training-time knowledge rather than synthesizing information from provided evidence.
Finally, long-context evaluations (e.g., recent long-context model work [ 29]) demonstrate improved capacity for
processing extensive documents, yet they still lack systematic measurement of synthesis when evidence must be
extracted from noisy, realistic corpora. Related work on document-level reasoning [ 15,35] highlights limitations
11

Model Prec. Rec. Score Err.-type dist.
Full-set setting
Claude-Opus-4.1 0.242 0.224 40.0 MC:42 UC:7R:35NF:3
DeepSeek-V3.1 0.323 0.283 50.2 MC:42 UC:5 R:44NF:2
Doubao-1.6-1015 0.270 0.237 43.3MC:37UC:10 R:43 NF:4
Gemini-3-Pro0.3330.315 53.7 MC:41 UC:9 R:38 NF:8
GPT-5.1 0.3180.330 57.0MC:39 UC:4R:41 NF:3
Average0.297 0.278 48.8 MC:40.2 UC:7 R:40.2 NF:4
Related-only setting
Claude-Opus-4.1 0.241 0.212 52.0 MC:38 UC:7 R:38 NF:2
DeepSeek-V3.1 0.308 0.270 62.1 MC:37 UC:4 R:40 NF:2
Doubao-1.6-1015 0.276 0.247 59.4 MC:32 UC:6 R:37 NF:2
Gemini-3-Pro0.3490.332 68.3 MC:29UC:2 R:37NF:3
GPT-5.1 0.329 0.33966.9 MC:34UC:1 R:33NF:5
Average0.301 0.280 61.7 MC:34 UC:4 R:37 NF:2.8
Concepts-only setting
Claude-Opus-4.1 0.718 0.651 72.4 MC:9 UC:5 R:11 NF:0
DeepSeek-V3.1 0.695 0.650 69.8 MC:13 UC:6 R:7 NF:2
Doubao-1.6-1015 0.724 0.648 70.8 MC:11 UC:5 R:10 NF:0
Gemini-3-Pro 0.740 0.730 80.9MC:7 UC:2R:7NF:0
GPT-5.10.754 0.75481.4 MC:6 UC:0R:7 NF:0
Average0.766 0.686 75.0 MC:9.2 UC:3.8 R:8.4 NF:0.4
Instruction-only setting
Claude-Opus-4.1 0.246 0.236 49.3 MC:42 UC:4R:30NF:3
DeepSeek-V3.1 0.301 0.260 52.2 MC:40 UC:4 R:43 NF:4
Doubao-1.6-1015 0.293 0.267 50.6 MC:39 UC:3 R:41 NF:4
Gemini-3-Pro0.341 0.336 64.2 MC:34 UC:0R:42NF:0
GPT-5.1 0.300 0.297 59.8 MC:42 UC:2 R:38NF:5
Average0.296 0.279 55.2 MC:39.4 UC:2.6 R:38.8 NF:3.2
Table 3CoT-based error attribution under four evaluation settings. Abbreviations: MC=missing core concept;
UC=misused/incorrect core concept; R=reasoning-process error; NF=numeric/formalization error.
of current QA benchmarks, but does not provide controlled environments for measuring multi-document
synthesis under adversarial distractors.
4.2 Decoupled Retrieval and Reasoning Benchmark
Our benchmark is designed as a controlled environment that disentangles retrieval from reasoning for document-
grounded scientific synthesis. Concretely, each instance is organized around a five-field structure (Instruction,
Concepts, CoT, Answer, Document Set) and a two-stage verifiability protocol: problems are unsolvable in
instruction-only mode, yet become solvable once the correct concepts are provided. This explicitly targets
scientific reasoning that requires integrating novel concepts rather than relying on parametric recall.
Unlike end-to-end RAG benchmarks that conflate retrieval and generation, we introduce evaluation settings
that decouple capabilities by construction:Instruction-onlyisolates parametric knowledge;Conceptsisolates
concept-level reasoning;Related-onlymeasures weak concept retrieval plus reasoning; andFull-setstress-tests
evidence selection and de-noising under distractors. This structure enables attribution of performance loss to
(i) evidence selection and de-noising, (ii) document-to-concept extraction, and (iii) concept-level composition
and scheduling.
Finally, by explicitly including topically adjacent but concept-irrelevant distractors in the document set, our
benchmark operationalizes adversarial noise as a first-class variable. This targets failure modes that are
12

typically masked in prior benchmarks—especially the difficulty of switching from parametric reasoning to
evidence-grounded reasoning, and the brittleness of multi-concept coordination under evidence overload.
5 Conclusion
We introduceDeR2, a controlled benchmark for decoupling retrieval from reasoning in retrieval-infused
problem solving. Each instance is built with a two-stage verifiability protocol and a structured doc set
(related + distractors), enabling stable attribution of performance loss to (i) evidence selection/de-noising, (ii)
document-to-concept extraction, and (iii) concept-level composition and scheduling.
Across strong commercial and open models, we observe consistent non-trivial failure modes: (1)reasoning
mode switching failureswhereInstruction-only can outperform Full-set, indicating that adding documents
may disrupt a previously viable parametric reasoning path; (2)structural (procedural) retrieval failureswhere
models recognize definitions but cannot execute constructive mechanisms; (3)Concepts-only is not an upper
bound, revealing bottlenecks in multi-concept coordination and long-range dependency tracking; and (4)
nonlinear noise effectsthat alter early reasoning topology and lead to irrecoverable trajectory drift.
These results suggest that “deep research” capability is nota simple extension of end-to-end RAG accuracy:
robust systems must explicitly support evidence-grounded mode control, executable concept utilization, and
adaptive hypothesis management under noisy contexts. We releaseDeR2to facilitate principled model
selection and diagnosis for retrieval-infused reasoning and to provide a practical target for future training and
evaluation.
13

Contributions and Acknowledgements
Multimodal Art Projection (M-A-P) is a non-profit open-source AI research community, ran by donation. The
community members are working on research topics in a wide range of spectrum, including but not limited to
the pre-training paradigm of foundation models, large-scale data collection and processing, and the derived
applications on coding, reasoning and music generation.
Leading Authors
•Shuangshuang Ying, M-A-P
•Zheyu Wang, ByteDance Seed, China
•Yunjian Peng, ByteDance Seed, China
•Jin Chen, ByteDance Seed, China
Contributors
•Yuhao Wu, ByteDance Seed, China
•Hongbin Lin, ByteDance Seed, China
•Dingyu He, ByteDance Seed, China
•Siyi Liu, ByteDance Seed, China
•Gengchen Yu, ByteDance Seed, China
•YinZhu Piao, ByteDance Seed, China
•Yuchen Wu, ByteDance Seed, China
•Xin Gui, M-A-P
•Zhongyuan Peng, Fudan University
•Xin Li, Nanyang Technological University
•Xeron Du, M-A-P
Advisors
•Libo Qin, Harbin Institute of Technology, Shenzhen
•YiXin Cao, Fudan University
Corresponding Authors
•Ge Zhang, ByteDance Seed, China
14

References
[1]Zhipu AI. Glm-4.6: An open large language model, 2025. URL https://huggingface.co/zai-org/GLM-4.6 .
Hugging Face Model Card.
[2] anthropic. claude-opus-4-1, 2025. URLhttps://www.anthropic.com/news/claude-opus-4-1.
[3] anthropic. Claude 4.5 sonnet, 2025. URLhttps://www.anthropic.com/news/claude-sonnet-4-5.
[4] ByteDance. Doubao large language model, 2024. URLhttps://www.doubao.com/. Accessed: 2025-01-16.
[5]Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert,
Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve
math word problems, 2021.
[6]Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel
Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning,
multimodality, long context, and next generation agentic capabilities. arXivpreprint arXiv:2507.06261, 2025.
[7]DeepSeek-AI. Deepseek-v3.2: Pushing the frontier of open large language models. arXivpreprint arXiv:2512.02556 ,
2024. URLhttps://arxiv.org/abs/2512.02556.
[8]Mohamed Amine Ferrag, Norbert Tihanyi, and Merouane Debbah. From llm reasoning to autonomous ai agents:
A comprehensive review, 2025. URLhttps://arxiv.org/abs/2504.19678.
[9]Kai Fronsdal, Aryan Gulati, Brando Miranda, Eric Chen, Emily Xia, Bruno de Moraes Dumont, and Sanmi Koyejo.
Putnam-axiom: A functional and static benchmark for measuring higher level mathematical reasoning. NeurIPS
2024Workshop onMATH-AI , October 2024. URL https://openreview.net/pdf?id=YXnwlZe0yf . Published:
09 Oct 2024, Last Modified: 09 Oct 2024.
[10]Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen,
Runxin Xu, et al. Omni-math: A universal olympiad level mathematic benchmark for large language models.
arXivpreprint arXiv:2410.07985, 2024.
[11]Google DeepMind. Gemini 3 pro: Advanced ai model. https://deepmind.google/models/gemini/pro/ , nov
2025. Released November 2025.
[12]Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob
Steinhardt. Measuring mathematical problem solving with the math dataset. arXivpreprint arXiv:2103.03874 ,
2021.
[13]Pengcheng Jiang, Xueqiang Xu, Jiacheng Lin, Jinfeng Xiao, Zifeng Wang, Jimeng Sun, and Jiawei Han. s3: You
don’t need that much data to train a search agent via rl, 2025. URLhttps://arxiv.org/abs/2505.14146.
[14]CarlosE.Jimenez, JohnYang, AlexanderWettig, ShunyuYao, KexinPei, OfirPress, andKarthikNarasimhan. Swe-
bench: Can language models resolve real-world github issues?, 2024. URL https://arxiv.org/abs/2310.06770 .
[15]Ehsan Kamalloo, Charles LA Clarke, and Davood Rafiei. Limitations of open-domain question answering
benchmarks for document-level reasoning. In Proceedings ofthe46thInternational ACMSIGIRConference on
Research andDevelopment inInformation Retrieval, pages 2123–2128, 2023.
[16]Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle
Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering
research. Transactions oftheAssociation forComputational Linguistics, 7:453–466, 2019.
[17]Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei
Tao, Xinyu Wang, Weizhou Shen, Junkai Zhang, Dingchu Zhang, Xixi Wu, Yong Jiang, Ming Yan, Pengjun
Xie, Fei Huang, and Jingren Zhou. Websailor: Navigating super-human reasoning for web agent, 2025. URL
https://arxiv.org/abs/2507.02592.
[18]Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou.
Search-o1: Agentic search-enhanced large reasoning models, 2025. URLhttps://arxiv.org/abs/2501.05366.
[19]Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yongkang Wu, Ji-Rong Wen, Yutao Zhu, and Zhicheng
Dou. Webthinker: Empowering large reasoning models with deep research capability, 2025. URL https:
//arxiv.org/abs/2504.21776.
15

[20]Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu
Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin
Button, Matthew Knight, Benjamin Chess, and John Schulman. Webgpt: Browser-assisted question-answering
with human feedback, 2022. URLhttps://arxiv.org/abs/2112.09332.
[21] OpenAI. Introducing gpt-5.1, 2025. URLhttps://openai.com/index/gpt-5-1/.
[22]Bhrij Patel, Souradip Chakraborty, Wesley A. Suttle, Mengdi Wang, Amrit Singh Bedi, and Dinesh Manocha.
Aime: Ai system optimization via multiple llm evaluators, 2024. URLhttps://arxiv.org/abs/2410.03131.
[23]Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne,
Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktäschel, and Sebastian Riedel.
KILT: a benchmark for knowledge intensive language tasks. In Kristina Toutanova, Anna Rumshisky, Luke Zettle-
moyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou,
editors,Proceedings ofthe2021Conference oftheNorthAmerican Chapter oftheAssociation forComputational
Linguistics: HumanLanguage Technologies , pages 2523–2544, Online, June 2021. Association for Computational
Linguistics. doi: 10.18653/v1/2021.naacl-main.200. URLhttps://aclanthology.org/2021.naacl-main.200/.
[24]Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Sean Shi, Michael Choi, Anish
Agrawal, Arnav Chopra, Adam Khoja, Ryan Kim, Jason Hausenloy, Oliver Zhang, Mantas Mazeika, Daron
Anderson, Tung Nguyen, Mobeen Mahmood, Fiona Feng, Steven Y. Feng, Haoran Zhao, Michael Yu, Varun
Gangal, Chelsea Zou, Zihan Wang, Jessica P. Wang, Pawan Kumar, Oleksandr Pokutnyi, Robert Gerbicz, Serguei
Popov, John-Clark Levin, Mstyslav Kazakov, Johannes Schmitt, Geoff Galgon, Alvaro Sanchez, Yongki Lee, Will
Yeadon, Scott Sauers, Marc Roth, Chidozie Agu, Søren Riis, Fabian Giska, Saiteja Utpala, Zachary Giboney,
Gashaw M. Goshu, Joan of Arc Xavier, Sarah-Jane Crowson, Mohinder Maheshbhai Naiya, Noah Burns, Lennart
Finke, Zerui Cheng, Hyunwoo Park, Francesco Fournier-Facio, John Wydallis, Mark Nandor, Ankit Singh, Tim
Gehrunger, Jiaqi Cai, Ben McCarty, Darling Duclosel, Jungbae Nam, Jennifer Zampese, Ryan G. Hoerr, Aras
Bacho, Gautier Abou Loume, Abdallah Galal, Hangrui Cao, Alexis C Garretson, Damien Sileo, Qiuyu Ren,
Doru Cojoc, Pavel Arkhipov, Usman Qazi, Lianghui Li, Sumeet Motwani, Christian Schroeder de Witt, Edwin
Taylor, Johannes Veith, Eric Singer, Taylor D. Hartman, Paolo Rissone, Jaehyeok Jin, Jack Wei Lun Shi, Chris G.
Willcocks, Joshua Robinson, Aleksandar Mikov, Ameya Prabhu, Longke Tang, Xavier Alapont, Justine Leon
Uro, Kevin Zhou, Emily de Oliveira Santos, Andrey Pupasov Maksimov, Edward Vendrow, Kengo Zenitani,
Julien Guillod, Yuqi Li, Joshua Vendrow, Vladyslav Kuchkin, Ng Ze-An, Pierre Marion, Denis Efremov, Jayson
Lynch, Kaiqu Liang, Andrew Gritsevskiy, Dakotah Martinez, Ben Pageler, Nick Crispino, Dimitri Zvonkine,
Natanael Wildner Fraga, Saeed Soori, Ori Press, Henry Tang, Julian Salazar, Sean R. Green, Lina Brüssel, Moon
Twayana, Aymeric Dieuleveut, T. Ryan Rogers, Wenjin Zhang, Bikun Li, Jinzhou Yang, Arun Rao, Gabriel Loiseau,
Mikhail Kalinin, Marco Lukas, Ciprian Manolescu, Subrata Mishra, Ariel Ghislain Kemogne Kamdoum, Tobias
Kreiman, Tad Hogg, Alvin Jin, Carlo Bosio, Gongbo Sun, Brian P Coppola, Tim Tarver, Haline Heidinger, Rafael
Sayous, Stefan Ivanov, Joseph M Cavanagh, Jiawei Shen, Joseph Marvin Imperial, Philippe Schwaller, Shaipranesh
Senthilkuma, Andres M Bran, Ali Dehghan, Andres Algaba, Brecht Verbeken, David Noever, Ragavendran P
V, Lisa Schut, Ilia Sucholutsky, Evgenii Zheltonozhskii, Derek Lim, Richard Stanley, Shankar Sivarajan, Tong
Yang, John Maar, Julian Wykowski, Martí Oller, Jennifer Sandlin, Anmol Sahu, Yuzheng Hu, Sara Fish, Nasser
Heydari, Archimedes Apronti, Kaivalya Rawal, Tobias Garcia Vilchis, Yuexuan Zu, Martin Lackner, James
Koppel, Jeremy Nguyen, Daniil S. Antonenko, Steffi Chern, Bingchen Zhao, Pierrot Arsene, Alan Goldfarb,
Sergey Ivanov, Rafał Poświata, Chenguang Wang, Daofeng Li, Donato Crisostomi, Andrea Achilleos, Benjamin
Myklebust, Archan Sen, David Perrella, Nurdin Kaparov, Mark H Inlow, Allen Zang, Elliott Thornley, Daniil
Orel, Vladislav Poritski, Shalev Ben-David, Zachary Berger, Parker Whitfill, Michael Foster, Daniel Munro, Linh
Ho, Dan Bar Hava, Aleksey Kuchkin, Robert Lauff, David Holmes, Frank Sommerhage, Keith Schneider, Zakayo
Kazibwe, Nate Stambaugh, Mukhwinder Singh, Ilias Magoulas, Don Clarke, Dae Hyun Kim, Felipe Meneguitti
Dias, Veit Elser, Kanu Priya Agarwal, Victor Efren Guadarrama Vilchis, Immo Klose, Christoph Demian, Ujjwala
Anantheswaran, Adam Zweiger, Guglielmo Albani, Jeffery Li, Nicolas Daans, Maksim Radionov, Václav Rozhoň,
Ziqiao Ma, Christian Stump, Mohammed Berkani, Jacob Platnick, Volodymyr Nevirkovets, Luke Basler, Marco
Piccardo, Ferenc Jeanplong, Niv Cohen, Josef Tkadlec, Paul Rosu, Piotr Padlewski, Stanislaw Barzowski, Kyle
Montgomery, Aline Menezes, Arkil Patel, Zixuan Wang, Jamie Tucker-Foltz, Jack Stade, Tom Goertzen, Fereshteh
Kazemi, Jeremiah Milbauer, John Arnold Ambay, Abhishek Shukla, Yan Carlos Leyva Labrador, Alan Givré,
Hew Wolff, Vivien Rossbach, Muhammad Fayez Aziz, Younesse Kaddar, Yanxu Chen, Robin Zhang, Jiayi Pan,
Antonio Terpin, Niklas Muennighoff, Hailey Schoelkopf, Eric Zheng, Avishy Carmi, Adam Jones, Jainam Shah,
Ethan D. L. Brown, Kelin Zhu, Max Bartolo, Richard Wheeler, Andrew Ho, Shaul Barkan, Jiaqi Wang, Martin
Stehberger, Egor Kretov, Kaustubh Sridhar, Zienab EL-Wasif, Anji Zhang, Daniel Pyda, Joanna Tam, David M.
Cunningham, Vladimir Goryachev, Demosthenes Patramanis, Michael Krause, Andrew Redenti, Daniel Bugas,
16

David Aldous, Jesyin Lai, Shannon Coleman, Mohsen Bahaloo, Jiangnan Xu, Sangwon Lee, Sandy Zhao, Ning
Tang, Michael K. Cohen, Micah Carroll, Orr Paradise, Jan Hendrik Kirchner, Stefan Steinerberger, Maksym
Ovchynnikov, Jason O. Matos, Adithya Shenoy, Benedito Alves de Oliveira Junior, Michael Wang, Yuzhou Nie,
Paolo Giordano, Philipp Petersen, Anna Sztyber-Betley, Priti Shukla, Jonathan Crozier, Antonella Pinto, Shreyas
Verma, Prashant Joshi, Zheng-Xin Yong, Allison Tee, Jérémy Andréoletti, Orion Weller, Raghav Singhal, Gang
Zhang, Alexander Ivanov, Seri Khoury, Hamid Mostaghimi, Kunvar Thaman, Qijia Chen, Tràn Quôc Khánh,
Jacob Loader, Stefano Cavalleri, Hannah Szlyk, Zachary Brown, Jonathan Roberts, William Alley, Kunyang
Sun, Ryan Stendall, Max Lamparth, Anka Reuel, Ting Wang, Hanmeng Xu, Sreenivas Goud Raparthi, Pablo
Hernández-Cámara, Freddie Martin, Dmitry Malishev, Thomas Preu, Tomek Korbak, Marcus Abramovitch,
Dominic Williamson, Ziye Chen, Biró Bálint, M Saiful Bari, Peyman Kassani, Zihao Wang, Behzad Ansarinejad,
Laxman Prasad Goswami, Yewen Sun, Hossam Elgnainy, Daniel Tordera, George Balabanian, Earth Anderson,
Lynna Kvistad, Alejandro José Moyano, Rajat Maheshwari, Ahmad Sakor, Murat Eron, Isaac C. McAlister,
Javier Gimenez, Innocent Enyekwe, Andrew Favre D.O., Shailesh Shah, Xiaoxiang Zhou, Firuz Kamalov, Ronald
Clark, Sherwin Abdoli, Tim Santens, Khalida Meer, Harrison K Wang, Kalyan Ramakrishnan, Evan Chen,
Alessandro Tomasiello, G. Bruno De Luca, Shi-Zhuo Looi, Vinh-Kha Le, Noam Kolt, Niels Mündler, Avi Semler,
Emma Rodman, Jacob Drori, Carl J Fossum, Milind Jagota, Ronak Pradeep, Honglu Fan, Tej Shah, Jonathan
Eicher, Michael Chen, Kushal Thaman, William Merrill, Carter Harris, Jason Gross, Ilya Gusev, Asankhaya
Sharma, Shashank Agnihotri, Pavel Zhelnov, Siranut Usawasutsakorn, Mohammadreza Mofayezi, Sergei Bogdanov,
Alexander Piperski, Marc Carauleanu, David K. Zhang, Dylan Ler, Roman Leventov, Ignat Soroko, Thorben
Jansen, Pascal Lauer, Joshua Duersch, Vage Taamazyan, Wiktor Morak, Wenjie Ma, William Held, Tran Ðuc
Huy, Ruicheng Xian, Armel Randy Zebaze, Mohanad Mohamed, Julian Noah Leser, Michelle X Yuan, Laila
Yacar, Johannes Lengler, Hossein Shahrtash, Edson Oliveira, Joseph W. Jackson, Daniel Espinosa Gonzalez, Andy
Zou, Muthu Chidambaram, Timothy Manik, Hector Haffenden, Dashiell Stander, Ali Dasouqi, Alexander Shen,
Emilien Duc, Bita Golshani, David Stap, Mikalai Uzhou, Alina Borisovna Zhidkovskaya, Lukas Lewark, Mátyás
Vincze, Dustin Wehr, Colin Tang, Zaki Hossain, Shaun Phillips, Jiang Muzhen, Fredrik Ekström, Angela Hammon,
Oam Patel, Nicolas Remy, Faraz Farhidi, George Medley, Forough Mohammadzadeh, Madellene Peñaflor, Haile
Kassahun, Alena Friedrich, Claire Sparrow, Taom Sakal, Omkar Dhamane, Ali Khajegili Mirabadi, Eric Hallman,
Mike Battaglia, Mohammad Maghsoudimehrabani, Hieu Hoang, Alon Amit, Dave Hulbert, Roberto Pereira,
Simon Weber, Stephen Mensah, Nathan Andre, Anton Peristyy, Chris Harjadi, Himanshu Gupta, Stephen Malina,
Samuel Albanie, Will Cai, Mustafa Mehkary, Frank Reidegeld, Anna-Katharina Dick, Cary Friday, Jasdeep
Sidhu, Wanyoung Kim, Mariana Costa, Hubeyb Gurdogan, Brian Weber, Harsh Kumar, Tong Jiang, Arunim
Agarwal, Chiara Ceconello, Warren S. Vaz, Chao Zhuang, Haon Park, Andrew R. Tawfeek, Daattavya Aggarwal,
Michael Kirchhof, Linjie Dai, Evan Kim, Johan Ferret, Yuzhou Wang, Minghao Yan, Krzysztof Burdzy, Lixin
Zhang, Antonio Franca, Diana T. Pham, Kang Yong Loh, Joshua Robinson, Shreen Gul, Gunjan Chhablani,
Zhehang Du, Adrian Cosma, Colin White, Robin Riblet, Prajvi Saxena, Jacob Votava, Vladimir Vinnikov,
Ethan Delaney, Shiv Halasyamani, Syed M. Shahid, Jean-Christophe Mourrat, Lavr Vetoshkin, Renas Bacho,
Vincent Ginis, Aleksandr Maksapetyan, Florencia de la Rosa, Xiuyu Li, Guillaume Malod, Leon Lang, Julien
Laurendeau, Fatimah Adesanya, Julien Portier, Lawrence Hollom, Victor Souza, Yuchen Anna Zhou, Yiğit Yalın,
Gbenga Daniel Obikoya, Luca Arnaboldi, Rai (Michael Pokorny), Filippo Bigi, Kaniuar Bacho, Pierre Clavier,
Gabriel Recchia, Mara Popescu, Nikita Shulga, Ngefor Mildred Tanwie, Thomas C.H. Lux, Ben Rank, Colin Ni,
Alesia Yakimchyk, Huanxu (Quinn) Liu, Olle Häggström, Emil Verkama, Himanshu Narayan, Hans Gundlach,
Leonor Brito-Santana, Brian Amaro, Vivek Vajipey, Rynaa Grover, Yiyang Fan, Gabriel Poesia Reis e Silva,
Linwei Xin, Yosi Kratish, Jakub Łucki, Wen-Ding Li, Justin Xu, Kevin Joseph Scaria, Freddie Vargus, Farzad
Habibi, Long (Tony) Lian, Emanuele Rodolà, Jules Robins, Vincent Cheng, Declan Grabb, Ida Bosio, Tony
Fruhauff, Ido Akov, Eve J. Y. Lo, Hao Qi, Xi Jiang, Ben Segev, Jingxuan Fan, Sarah Martinson, Erik Y. Wang,
Kaylie Hausknecht, Michael P. Brenner, Mao Mao, Yibo Jiang, Xinyu Zhang, David Avagian, Eshawn Jessica
Scipio, Muhammad Rehan Siddiqi, Alon Ragoler, Justin Tan, Deepakkumar Patil, Rebeka Plecnik, Aaron Kirtland,
Roselynn Grace Montecillo, Stephane Durand, Omer Faruk Bodur, Zahra Adoul, Mohamed Zekry, Guillaume
Douville, Ali Karakoc, Tania C. B. Santos, Samir Shamseldeen, Loukmane Karim, Anna Liakhovitskaia, Nate
Resman, Nicholas Farina, Juan Carlos Gonzalez, Gabe Maayan, Sarah Hoback, Rodrigo De Oliveira Pena, Glen
Sherman, Hodjat Mariji, Rasoul Pouriamanesh, Wentao Wu, Gözdenur Demir, Sandra Mendoza, Ismail Alarab,
Joshua Cole, Danyelle Ferreira, Bryan Johnson, Hsiaoyun Milliron, Mohammad Safdari, Liangti Dai, Siriphan
Arthornthurasuk, Alexey Pronin, Jing Fan, Angel Ramirez-Trinidad, Ashley Cartwright, Daphiny Pottmaier,
Omid Taheri, David Outevsky, Stanley Stepanic, Samuel Perry, Luke Askew, Raúl Adrián Huerta Rodríguez,
Abdelkader Dendane, Sam Ali, Ricardo Lorena, Krishnamurthy Iyer, Sk Md Salauddin, Murat Islam, Juan
Gonzalez, Josh Ducey, Russell Campbell, Maja Somrak, Vasilios Mavroudis, Eric Vergo, Juehang Qin, Benjámin
Borbás, Eric Chu, Jack Lindsey, Anil Radhakrishnan, Antoine Jallon, I.M.J. McInnis, Alex Hoover, Sören Möller,
Song Bian, John Lai, Tejal Patwardhan, Summer Yue, Alexandr Wang, and Dan Hendrycks. Humanity’s last
17

exam.arXiv, 2025.
[25]Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine
comprehension of text. arXivpreprint arXiv:1606.05250, 2016.
[26]David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Ju-
lian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. arXivpreprint
arXiv:2311.12022, 2023.
[27]Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R.
Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, Agnieszka Kluska, Aitor Lewkowycz, Akshat Agarwal,
Alethea Power, Alex Ray, Alex Warstadt, Alexander W. Kocurek, Ali Safaya, Ali Tazarv, Alice Xiang, Alicia
Parrish, Allen Nie, Aman Hussain, Amanda Askell, Amanda Dsouza, Ameet Rahane, Anantharaman S. Iyer,
Anders Andreassen, Andrea Santilli, Andreas Stuhlmüller, Andrew M. Dai, Andrew La, Andrew K. Lampinen,
Andy Zou, Angela Jiang, Angelica Chen, Anh Vuong, Animesh Gupta, Anna Gottardi, Antonio Norelli, Anu
Venkatesh, Arash Gholamidavoodi, Arfa Tabassum, Arul Menezes, Arun Kirubarajan, Asher Mullokandov,
Ashish Sabharwal, Austin Herrick, Avia Efrat, Aykut Erdem, Ayla Karakas, and et al. Beyond the imitation
game: Quantifying and extrapolating the capabilities of language models. CoRR, abs/2206.04615, 2022. doi:
10.48550/arXiv.2206.04615. URLhttps://doi.org/10.48550/arXiv.2206.04615.
[28]Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou, Yong Jiang, Pengjun Xie, Yan Zhang, Fei
Huang, and Jingren Zhou. Zerosearch: Incentivize the search capability of llms without searching, 2025. URL
https://arxiv.org/abs/2505.04588.
[29]Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024. URL
https://arxiv.org/abs/2403.05530.
[30]Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen,
Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang
Du, Dikang Du, Yulun Du, Yu Fan, Yichen Feng, Kelin Fu, Bofei Gao, Hongcheng Gao, Peizhong Gao, Tong
Gao, Xinran Gu, Longyu Guan, Haiqing Guo, Jianhang Guo, Hao Hu, Xiaoru Hao, Tianhong He, Weiran He,
Wenyang He, Chao Hong, Yangyang Hu, Zhenxing Hu, Weixiao Huang, Zhiqi Huang, Zihao Huang, Tao Jiang,
Zhejun Jiang, Xinyi Jin, Yongsheng Kang, Guokun Lai, Cheng Li, Fang Li, Haoyang Li, Ming Li, Wentao Li,
Yanhao Li, Yiwei Li, Zhaowei Li, Zheming Li, Hongzhan Lin, Xiaohan Lin, Zongyu Lin, Chengyin Liu, Chenyu
Liu, Hongzhang Liu, Jingyuan Liu, Junqi Liu, Liang Liu, Shaowei Liu, T. Y. Liu, Tianwei Liu, Weizhou Liu,
Yangyang Liu, Yibo Liu, Yiping Liu, Yue Liu, Zhengying Liu, Enzhe Lu, Lijun Lu, Shengling Ma, Xinyu Ma,
Yingwei Ma, Shaoguang Mao, Jie Mei, Xin Men, Yibo Miao, Siyuan Pan, Yebo Peng, Ruoyu Qin, Bowen Qu,
Zeyu Shang, Lidong Shi, Shengyuan Shi, Feifan Song, Jianlin Su, Zhengyuan Su, Xinjie Sun, Flood Sung, Heyi
Tang, Jiawen Tao, Qifeng Teng, Chensi Wang, Dinglu Wang, Feng Wang, Haiming Wang, Jianzhou Wang, Jiaxing
Wang, Jinhong Wang, Shengjie Wang, Shuyi Wang, Yao Wang, Yejie Wang, Yiqin Wang, Yuxin Wang, Yuzhi
Wang, Zhaoji Wang, Zhengtao Wang, Zhexu Wang, Chu Wei, Qianqian Wei, Wenhao Wu, Xingzhe Wu, Yuxin Wu,
Chenjun Xiao, Xiaotong Xie, Weimin Xiong, Boyu Xu, Jing Xu, Jinjing Xu, L. H. Xu, Lin Xu, Suting Xu, Weixin
Xu, Xinran Xu, Yangchuan Xu, Ziyao Xu, Junjie Yan, Yuzi Yan, Xiaofei Yang, Ying Yang, Zhen Yang, Zhilin
Yang, Zonghan Yang, Haotian Yao, Xingcheng Yao, Wenjie Ye, Zhuorui Ye, Bohong Yin, Longhui Yu, Enming
Yuan, Hongbang Yuan, Mengjie Yuan, Haobing Zhan, Dehao Zhang, Hao Zhang, Wanlu Zhang, Xiaobin Zhang,
Yangkun Zhang, Yizhi Zhang, Yongting Zhang, Yu Zhang, Yutao Zhang, Yutong Zhang, Zheng Zhang, Haotian
Zhao, Yikai Zhao, Huabin Zheng, Shaojie Zheng, Jianren Zhou, Xinyu Zhou, Zaida Zhou, Zhen Zhu, Weiyu
Zhuang, and Xinxing Zu. Kimi k2: Open agentic intelligence, 2025. URLhttps://arxiv.org/abs/2507.20534.
[31]Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny
Zhou, Quoc Le, and Thang Luong. Freshllms: Refreshing large language models with search engine augmentation.
InAnnualMeeting oftheAssociation forComputational Linguistics , 2023. URL https://api.semanticscholar.
org/CorpusID:263672149.
[32]Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung,
Alex Tachard Passos, William Fedus, and Amelia Glaese. Browsecomp: A simple yet challenging benchmark for
browsing agents, 2025. URLhttps://arxiv.org/abs/2504.12516.
[33]An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen
Huang, Chenxu Lv, et al. Qwen3 technical report. arXivpreprint arXiv:2505.09388, 2025.
[34]Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Ellen Riloff, David
18

Chiang, Julia Hockenmaier, and Jun’ichi Tsujii, editors, Proceedings ofthe2018Conference onEmpirical Methods
inNaturalLanguage Processing, Brussels, Belgium, October 31-November 4,2018, pages 2369–2380. Association
for Computational Linguistics, 2018. doi: 10.18653/v1/d18-1259. URL https://doi.org/10.18653/v1/d18-1259 .
[35]Yilun Zhao, Yitao Long, Hongjun Liu, Ryo Kamoi, Linyong Nan, Lyuhao Chen, Yixin Liu, Xiangru Tang, Rui
Zhang, and Arman Cohan. Docmath-eval: Evaluating math reasoning capabilities of llms in understanding long
and specialized documents. arXivpreprint arXiv:2311.09805, 2023.
19

A Prompts
Prompt for infer answers in Related-only and Full-set
You are a literature analysis agent.
Problem: {{Instruction}}
Context Source:
•Use the built-in function fetch_md to list and read local markdown files under {{local_dir}}.
•First, call fetch_md with instance_id: "{{local_dir}}" and no filename to get the file list.
•Then, call fetch_md with filename to read selected files.
Guidelines:
•Tool call policy: Within a single assistant turn, call fetch_md at most once; read only one
document per tool call (one filename). If more documents are required, continue in subsequent
turns.
•Sufficiency policy: Answers may not be contained in a single document. After reading, assess
whether the currently fetched documents are sufficient to derive the correct answer. If uncertain,
continue reading more documents in subsequent turns before finalizing.
•Before the final answer, output your step-by-step reasoning process (Reasoning) explaining how
the fetched documents support each claim; include citations like [file: xxx.md] inline.
•Do not guess; provide the final result only once you are confident and can justify it with the read
materials.
•IMPORTANT: The final line must be exactly:
Final answer: <your final answer>
Prompt for infer answers in Concepts-only
You are a concept-based reasoning agent.
Problem: {{Instruction}}
Concepts: {{Concepts}}
Output Requirements:
•First, output a section starting with the exact header: Reasoning:
Provide step-by-step reasoning to support each claim.
•Then provide the final concise conclusion on the last line in the exact format:
Final answer: <your final answer>
20

Prompt for infer answers in Instruction-only
You are an agent skilled at solving challenging problems.
Problem: {{Instruction}}
Output Requirements:
•First, output a section starting with the exact header: Reasoning:
Provide step-by-step reasoning to support each claim.
•Then provide the final concise conclusion on the last line in the exact format:
Final answer: <your final answer>
Prompt for extract concepts in CoT
# Task Steps
1. Concept Identification: Carefully read the Model’s Chain-of-Thought (CoT). Identify and extract
every single core concept, theorem, or formula that the model explicitly *used* in its reasoning
process.
2. Create Used Concept List: List these extracted items as the Used Concepts.
3. Comparison and Calculation: Compare the Used Concepts list against the Ground Truth
Concept List (Standard Concepts) to determine:
- True Positives (TP): Concepts that are in Used Concepts AND in Standard Concepts. (Correctly
used necessary concepts)
- False Positives (FP): Concepts that are in Used Concepts BUT NOT in Standard Concepts.
(Incorrectly used or irrelevant concepts)
- False Negatives (FN): Concepts that are NOT in Used Concepts BUT are in Standard Concepts.
(Necessary concepts that were missed or not used)
—
# Input Data
1. Model’s Chain-of-Thought (CoT): The step-by-step reasoning generated by the model.
2. Ground Truth Concept List (Standard Concepts): The authoritative list of core concepts,
theorems, or formulas that are *necessary* for a correct solution.
# INPUT
[Model’s Chain-of-Thought (CoT)]
{{pred_cot}}
[Ground Truth Concept List (Standard Concepts)]
{{concepts}}
—
# OUTPUT
You must output a JSON object with the following structure, followed by your final calculation.
‘‘‘json { "Used Concepts": [ "Concept A", "Concept B", ... ], "True Positives (TP)": count, "False
Positives (FP)": count, "False Negatives (FN)": count, }‘‘‘
21

Prompt for error cause analysis
# Task
Based on the model’s overall reasoning process (pred_cot) and the concepts it used (pred_concepts),
compare and match them against the ground-truth core concepts (true_concepts) and reasoning
process (true_cot), determine the cause(s) of the model’s error, and provide a detailed explanation.
—
# Core Requirements
Error type labeling rules: When the matching quality is insufficient, the error cause must be
selected strictly from the following labels (the characters must match exactly):
- Errors in reasoning process - Numerical or formal errors - Missing core concepts - Errors in
understanding or using core concepts If multiple error types co-exist, you may combine up to two
labels, separated by “/”.
Specific justification:
- You must explicitly point out the concrete issues in pred_concepts / pred_cot - Clearly explain
why they cannot form an effective match with true_concepts - Avoid vague statements; the
explanation must be verifiable
—
# INPUT
true_concepts: {{true_concepts}}
true_cot:{{true_cot}}
pred_concepts: {{pred_concepts}}
pred_cot: {{pred_cot}}
—
# OUTPUT
The output must be a valid JSON in the following format. Do not include any extra fields or
explanatory text.
‘‘‘json { "error reason": "error in reasoning process", "specific reason": "xxxxxxxx in pred_-
concepts, and the reasoning process of pred_cot is xxxxxxxx..." }‘‘‘
22

Prompt for scoring LLM-generated answers
The evaluation must adhere strictly to the following Rubric, which assesses three dimensions: A)
Factual and Logical Correctness, B) Completeness, and C) Reasoning Quality (based on the CoT).
—
[RUBRIC: Correctness Score (0-1)]
1 (Excellent)
A: Correctness: The answer is fully correct, logically rigorous, and perfectly aligns with the core
conclusion of the true answer.
B: Completeness: The answer addresses all aspects of the instruction and includes all key
information points from the true answer.
C: Reasoning: The CoT is clear, well-structured, and correctly and effectively utilizes all necessary
concepts.
0.75 (Good)
A: Correctness: Mostly correct with minor issues that do not affect the core conclusion.
B: Completeness: Covers main parts but may omit secondary details.
C: Reasoning: Generally correct and understandable but may have skipped steps or imprecision.
0.5 (Acceptable)
A: Correctness: Captures the core idea but contains noticeable factual errors or logical flaws.
B: Completeness: Only covers the core part, omitting multiple important points.
C: Reasoning: Understandable but messy, or fails to sufficiently apply key concepts.
0.25 (Poor)
A: Correctness: Fundamentally incorrect or based on faulty logic.
B: Completeness: Severely deviates from instruction requirements with minimal valid information.
C: Reasoning: Serious logical fallacies or irrelevant to the problem.
0.0 (Unacceptable)
A: Correctness: Completely incorrect, irrelevant, or nonsensical.
B: Completeness: Does not address the instruction at all.
C: Reasoning: Missing or completely erroneous.
—
[INPUT DATA]
Instruction (The Problem to Solve): {{instruction}}
True Answer (Benchmark Standard): {{true_answer}}
Predicted Answer (Model Output to Evaluate): {{pred_answer}}
Concepts (Knowledge Base): {{concepts}}
Chain-of-Thought (Model’s Reasoning Process): {{pred_cot}}
—
[EVALUATION TASK]
Analyze the Predicted Answer and Chain-of-Thought against the True Answer and the Instruction.
Assign a single score between 0 and 1 based on the provided Rubric.
Please output only the final score in the exact format:
score: <float between 0 and 1>
IMPORTANT: The last line must be exactly "score: <float between 0 and 1>", and there must be
no content after this line.
23

B Case Analysis
B.1 bad case of Claude-Opus-4.1 in Instruction-only
InstructionDetermine the time complexity for the proposed algorithm to provably learn a multi-head
attention layer Ffrom random labeled examples drawn uniformly from {±1}k×d. The learning process must
operate under the assumption that the ground truth attention and projection matrices {Θi, Wi}m
i=1satisfy
a set of non-degeneracy conditions, and it should output estimates {ˆWi,ˆΘi}m
i=1that result in a multi-head
attention layer ˆFachieving a mean squared error of at most( kd)−Ω(m)with a specified success probability of
at least1−δ.
Pred CoTReasoning:
To determine the time complexity for learning a multi-head attention layer, I need to analyze several key
components:
1. Problem Structure: We’re learning mattention heads, each with matricesΘ iandWi. The input space is
{±1}k×d, giving us(2)kdpossible inputs.
2. Target Accuracy: The required MSE of( kd)−Ω(m)is exponentially small in m. This extremely high
accuracy requirement significantly impacts the complexity.
3. Sample Complexity Analysis:
- To achieve MSE of ϵ= (kd)−Ω(m), standard learning theory suggests we need O(1/ϵ2)samples for concentra-
tion
- This translates to(kd)Ω(m)samples - The success probability requirement1−δadds alog(1/δ)factor
4. Computational Cost per Sample: - For each sample, computing multi-head attention involves: - Attention
score computation: O(k2d)per head - Applying attention weights: O(k2d)per head - Total per sample:
O(mk2d)for allmheads
5. Learning Algorithm Complexity: - Provable learning algorithms for neural networks often use moment-
based or tensor decomposition methods - For multi-head structures with mheads, these methods typically
require computing O(m)-th order statistics - The computational cost scales exponentially with the order of
moments/tensors needed
6. Total Time Complexity: - Number of samples:( kd)Ω(m)- Cost per sample processing: O(mk2d)- Success
probability factor:log(1/δ)- Combined:(kd)O(m)·log(1/δ)
The dominant term is(kd)O(m), which arises from both the sample complexity needed for the exponentially
small error and the computational requirements of provable learning methods for multi-head structures.
Final answer:(kd)O(m)·log(1/δ)
True ConceptsConcept 1: A crude, noisy estimate for the sum of projection matrices,P
iWi, can be
obtained by calculating the empirical correlation between the input sequence Xand the output label Yvia
the approximationE1
kX⊤JY
≈P
iWi, whereJis the all-ones matrix.
Concept 2: A convex body that approximates the affine hull of the attention matrices can be constructed by
using linear regression to identify input examples Xthat induce similar attention patterns across all heads and
leveraging these examples to form affine linear constraints that are simultaneously satisfied by all attention
matricesΘ 1, . . . ,Θ m.
Concept 3: The estimate of the projection matrix sum can be significantly refined by using the minimum-norm
point within the sculpted convex body as a proxy for the average attention matrix. This allows for the
identification of examples that induce large-margin (approximately 1-sparse) attention patterns across all
heads, which in turn provides highly accurate linear constraints on the sum of projection matrices.
24

Concept 4: The convex body sculpting process can be re-run using the refined estimate of the projection
matrix sum to produce a new, much more accurate convex body K∗that is significantly closer to the true
affine hull of the attention matrices.
Concept 5: An epsilon-net containing candidate attention matrices is constructed not over the full d×dspace,
but specifically within the estimated m-dimensional linear span of the true attention matrices {Θi}m
i=1. This
crucial dimensionality reduction step reduces the search space for the subsequent brute-force step from being
exponential ind2to being exponential only inm.
Concept 6: A brute-force search is performed over all m-tuples of candidate matrices from the low-dimensional
epsilon-net, with linear regression solved for each. The final selection of the best model is confirmed on a
held-out validation set, and standard generalization bounds are used to certify that this selection is correct
with probability at least1 −δ. This validation step introduces a logarithmic dependency on1 /δto the overall
runtime.
Answer statusIncorrect
Causes of errorsMethod hallucination due to lack of specific domain knowledge
Missing/incorrect contentOmission of core algorithm mechanisms: Key steps such as "Convex Body",
"epsilon-net (ϵ-net)", "Linear Span", and "Brute-force search" were not mentioned.
Incorrect reasoning contentErroneously introduced "standard learning theory" concentration result O(1/ϵ2)
as the source of complexity; incorrectly used single-sample computational cost O(mk2d)as the multiplier for
total time complexity, neglecting the significant overhead in the algorithm’s training/search phase.
Concepts Recall StatusUnretrieved / completely missing. Failed to retrieve any core algorithm steps, fully
missing the algorithmic process based on convex body geometry and subspace enumeration.
Reasoning stateLogic completely deviated.
Correct logicCalculate time complexity based on constructive algorithms, with the brute-force search
step (searching for mmatrices in an m-dimensional subspace) dominating the complexity, resulting in an
exponential term ofO(m3)(from covering number computation or combinatorial search).
Model logicUsed a "statistical learning theory" template, erroneously simplifying it to "sample complexity
×single forward pass computation cost", assuming resolution via moment estimation/tensor decomposition,
and treating the sample size( kd)Ω(m)as the dominant term while neglecting the algorithm’s search space
complexity.
B.2 Bad Case of DeepSeek-V3.1 in Concepts-only
InstructionUsing the lifted formulation for certifiable Schubert calculus in type A, determine the total number
of bilinear equations in the systemSA(u, v, w ◦w)required to test for non-vanishing Schubert coefficients for
the specific case in S4with permutations u= 2143, v= 3124, and w= 4132. In this context, Schubert
coefficients are the structure constants in the ring of Schubert polynomials, and the lifted formulation is a
method to create a system of polynomial equations whose solutions correspond to the geometric problem of
intersecting Schubert varieties.
Pred CoTReasoning:
- The lifted formulation for certifiable Schubert calculus in type A involves creating a system of bilinear
equationsSA(u, v, w ◦w)whose solutions correspond to the intersection of Schubert varieties.
25

- The total number of bilinear equations in the system is determined by the sum of the number of descents for
each permutation involved:u,v, andw ◦w.
- For a permutation, the descent set Des(w)consists of indices iwhere w(i)u003ew(i+ 1). - Compute descent
sets:
- Foru= 2143: Des(u) ={1,3}, so|Des(u)|= 2.
- Forv= 3124: Des(v) ={1}, so|Des(v)|= 1.
- For w◦w, first compute w◦forS4:w◦= 4321. Then w◦w=w◦(w(i)), with w= 4132, so w◦w= 1423.
Des(w ◦w) ={2}, so|Des(w ◦w)|= 1.
- Total number of bilinear equations =|Des(u)|+|Des(v)|+|Des(w ◦w)|= 2 + 1 + 1 = 4.
Final answer: 4
True ConceptsConcept 1: Schubert Coefficients are non-negative integers cw
u,vthat are defined as the
structure constants for the multiplication of Schubert polynomials, which form a linear basis in the ring of
polynomialsZ[x 1, x2, . . .].
Concept 2: ’Geometric Interpretation of Schubert Coefficients states that the coefficient cw
u,vcounts the
number of points in the intersection of three generically translated Schubert varieties, expressed as cw
u,v=
#{X u(E•)∩X v(F•)∩X w◦w(G•)}.
Concept 3: A Schubert Variety Xwis the Zariski closure of a Schubert cellΩ w. The problem of finding the
Schubert coefficient is equivalent to finding the number of solutions to a system of equations that define
membership in an intersection of these varieties.
Concept 4: Stiefel Coordinates are a set of variables used to parameterize the flags within a Schubert cellΩA
w.
These variables form the entries of a matrixχA
wwhich is central to constructing the polynomial system.
Concept 5: The Lifted Formulation is a method that characterizes membership in a Schubert cell using a
system of bilinear equations by introducing auxiliary variables (e.g.,α, β), avoiding the use of higher-degree
determinantal equations.
Concept 6: The Maximal Descent, denoted by d:=max(Des(u)∪Des (v)∪Des (w)), determines the number
of column vectors to consider from the Stiefel coordinate matrix and sets the range of indices for the equations
in the system.
Theorem 1: "The Characterization of Schubert Cell Membership states that a flagΦ •is in the Schubert cell
ΩA
w(E•)if and only if for each k∈[d], there exist unique coefficients such that a specific linear combination of
the flag’s basis vectors lies within a specific subspace defined by the permutation w, i.e., gk∈Ewk−1−Ewk.
This theorem provides the direct rule for converting the geometric containment problem into algebraic
equations.
Answer statusIncorrect (predicted value: 4, true value: 7)
Causes of errorsHallucinatory reasoning due to lack of core concepts. The model lacked knowledge of the
specific construction of "Lifted Formulation" and fabricated an incorrect heuristic rule (number of equations
= sum of descent set element counts).
Missing/incorrect contentOmission of constructive details: Did not mention using d(Maximal Descent) to
determine the column range of the Stiefel matrix, introduction of auxiliary variables (to avoid high-degree
determinants), or the uniqueness constraint on linear combination coefficients in Theorem 1.
ErrorIncorrect formula ∥Des (u)∥+∥Des (v)∥+∥Des (w◦w)∥(erroneous simplification of the algebraic
geometric structure).
26

Concepts Recall StatusPartially retrieved. Mentioned "Lifted Formulation" and "Schubert Variety", but
severely omitted:
Concept 4 (Definition of Stiefel coordinate matrixχA
w);
Concept 6 (Definition of Maximal Descentd);
Theorem 1 (Theorem on algebraic conversion of Schubert cell membership).
Reasoning stateLogical break (premise error). The break occurred between the first and second steps, where
the model fabricated an incorrect formula "total number of equations determined by the sum of descent
set sizes" instead of deriving based on the actual construction of Lifted Formulation (Stiefel coordinate
dimensions, auxiliary variable introduction, rank constraints), replacing algebraic geometric derivation with
simple addition.
B.3 Bad Case of Gemini-3-Pro in Full-set
InstructionFor the bumblebee theory, if there exists an external coupled matter field LM, and the B-field is
(0, b(r),0,0), under the static spherically symmetric coordinate system ds2=−A(r)dt2+S(r)dr2+r2dΩ2,
then what is the black hole solutionA(r)?
Pred CoTThe problem asks for the black hole solution component A(r)in the context of the bumblebee
theory with an external coupled matter field and a radial bumblebee fieldB µ= (0, b(r),0,0).
Based on the file **Än exact Schwarzschild-like solution in a bumblebee gravity model.md¨** (which corresponds
to the work by Casana et al.), the solution is a spherically symmetric vacuum solution (where the potential V
ensures the bumblebee field stays at its vacuum expectation value).
Although the explicit derivation in Section III was truncated in the retrieved text, the metric form is revealed
in Section IV.C ( ¨Time delay of light¨). Equation (75) describes the null geodesic condition ds2= 0for the
spacetime (labeled as equation 36 in the full text) as:
−
1−2M
r
dt2+ (1 +ℓ)
1−2M
r−1
dr2+r2dϕ2= 0
Comparing this with the static spherically symmetric metric ds2=−A(r)dt2+S(r)dr2+r2dΩ2, we can
identify the metric coefficients:
A(r) = 1−2M
r, S(r) =1 +ℓ
1−2M
r
where ℓ=ξb2is the Lorentz-violating parameter derived from the non-minimal coupling ξand the vacuum
expectation valueb2.
The temporal component A(r)remains formally identical to the Schwarzschild solution, while the radial
componentS(r)is modified by the parameterℓ.
Reasoning:
The relevant document [file: An exact Schwarzschild-like solution in a bumblebee gravity model.md] discusses
a spherically symmetric solution in bumblebee gravity with a radial background field. The text explicitly
writes the line element for null geodesics in Eq. (75) as having a dt2coefficient of −(1−2M/r)and a dr2
coefficient involving(1 + ℓ). This implies that the metric potential A(r), which corresponds to −gtt, is
1−2M/r. This result is consistent with the known Casana et al. (2018) solution where the radial bumblebee
field modifies only the grrcomponent of the metric (and gθθeffective potentials in some contexts, but here
A(r)is the focus).
Final answer:A(r) = 1−2M
r
27

True ConceptsTheorem 1: bumblebee theory:
S=Z
d4x√−g1
2κR+ξ
2κBµBνRµν−1
4BµνBµν
−Z
d4x√−gL M
Theorem 2: We consider the matter field to be an electromagnetic field non-minimally coupled with the
bumblebee vector field. Its Lagrangian density expression reads as
LM=1
2κ 
FµνFµν+γBµBµFαβFαβ
, Then through Maxwell equation, we can derive Ftr=p
A(r)S(r)ϕ′(r). Building upon this, we solve the
modified Maxwell’s equations to obtain,ϕ(r) =q
r.
Concept 1: For convenience, we set ℓ=ξb2as Lorentz-violating parameter, we consider a spacelike background
bµwith the form bµ= (0, br(r),0,0). Utilizing the aforementioned condition bµbµ=b2=const, we can
deriveb r(r) =bp
S(r).
Concept 2: By simplifying Einstein equations, we obtain:[ S(r)A(r)]′= 0, which implies that S(r) =C1
A(r),
where C1is a constant. Similar to the Schwarzschild-like black hole solution in bumblebee gravity, we set the
value ofC 1to be1 +ℓ. Solving Einstein equations{t, t}, we can getA(r).
A′(r)S′(r)
4A(r)S(r)−A′′(r)
2A(r)+A′(r)2
4A(r)2+S′(r)
rS(r)+2ϕ′(r)2S(r)
(2 +ℓ)= 0
Answer statusIncorrect
Causes of errorsIncorrect derivation path due to neglecting key constraints. The model ignored the "external
coupled matter field LM" setting in the prompt, erroneously simplifying the problem to a vacuum solution
(non-charged solution) under pure Bumblebee gravity.
Missing/incorrect contentOmission of electromagnetic field contribution: Did not mention the Lagrangian
of the non-minimally coupled electromagnetic field or its Maxwell equation solutionϕ(r) =q/r.
Omission of source terms in Einstein equations: Failed to include the energy-momentum tensor contribution
fromL Mwhen solving forA(r).
Missing term in the answer formulaThe finalA(r)lacks the charge-related correction term2(1+ℓ)q2
(2+ℓ)r2.
Concepts Recall StatusPartially retrieved. Mentioned the definition of Lorentz-violating parameter ℓ, but
severely omitted:
The complete action form of Bumblebee theory;
The matter field LagrangianL Mand its Maxwell equation solutionϕ(r) =q/r;
The solution process of Einstein equations including the matter field (derivation of the correction to A(r)
fromϕ′(r)).
Reasoning stateLogical break. The break occurred in the initial phase of understanding the problem premises
and selecting formulas. The problem explicitly states the existence of an "external coupled matter field LM",
but the model erroneously claimed the solution is a spherically symmetric vacuum solution, neglected the
matter field, and used a charge-free Schwarzschild-like solution, resulting in the metric coefficient missing the
chargeqcontribution.
28
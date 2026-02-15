# Closing Reasoning Gaps in Clinical Agents with Differential Reasoning Learning

**Authors**: Jinsong Liu, Yuhang Jiang, Ramayya Krishnan, Rema Padman, Yiye Zhang, Jiang Bian

**Published**: 2026-02-10 16:29:32

**PDF URL**: [https://arxiv.org/pdf/2602.09945v1](https://arxiv.org/pdf/2602.09945v1)

## Abstract
Clinical decision support requires not only correct answers but also clinically valid reasoning. We propose Differential Reasoning Learning (DRL), a framework that improves clinical agents by learning from reasoning discrepancies. From reference reasoning rationales (e.g., physician-authored clinical rationale, clinical guidelines, or outputs from more capable models) and the agent's free-form chain-of-thought (CoT), DRL extracts reasoning graphs as directed acyclic graphs (DAGs) and performs a clinically weighted graph edit distance (GED)-based discrepancy analysis. An LLM-as-a-judge aligns semantically equivalent nodes and diagnoses discrepancies between graphs. These graph-level discrepancy diagnostics are converted into natural-language instructions and stored in a Differential Reasoning Knowledge Base (DR-KB). At inference, we retrieve top-$k$ instructions via Retrieval-Augmented Generation (RAG) to augment the agent prompt and patch likely logic gaps. Evaluation on open medical question answering (QA) benchmarks and a Return Visit Admissions (RVA) prediction task from internal clinical data demonstrates gains over baselines, improving both final-answer accuracy and reasoning fidelity. Ablation studies confirm gains from infusing reference reasoning rationales and the top-$k$ retrieval strategy. Clinicians' review of the output provides further assurance of the approach. Together, results suggest that DRL supports more reliable clinical decision-making in complex reasoning scenarios and offers a practical mechanism for deployment under limited token budgets.

## Full Text


<!-- PDF content starts -->

Closing Reasoning Gaps in Clinical Agents with Differential
Reasoning Learning
Jinsong Liu
jil4047@med.cornell.edu
Dept. of Population Health Sciences,
Weill Cornell Medicine, Cornell
University
New York, NY, USAYuhang Jiang
yj38@iu.edu
Dept. of Biostatistics and Health Data
Science,
Indiana University School of Medicine
Indianapolis, IN, USARamayya Krishnan
rk2x@cmu.edu
Heinz College of Information Systems
and Public Policy,
Carnegie Mellon University
Pittsburgh, PA, USA
Rema Padman
rpadman@cmu.edu
Heinz College of Information Systems
and Public Policy,
Carnegie Mellon University
Pittsburgh, PA, USAYiye Zhang
yiz2014@med.cornell.edu
Dept. of Population Health Sciences,
Weill Cornell Medicine, Cornell
University
New York, NY, USAJiang Bian
bianji@regenstrief.org
Dept. of Biostatistics and Health Data
Science,
Indiana University School of Medicine
Indianapolis, IN, USA
Abstract
Clinical decision support requires not only correct answers but also
clinically valid reasoning. We propose Differential Reasoning Learn-
ing (DRL), a framework that improves clinical agents by learning
from reasoning discrepancies. From reference reasoning rationales
(e.g., physician-authored clinical rationale, clinical guidelines, or
outputs from more capable models) and the agent’s free-form chain-
of-thought (CoT), DRL extracts reasoning graphs as directed acyclic
graphs (DAGs) and performs a clinically weighted graph edit dis-
tance (GED)-based discrepancy analysis. An LLM-as-a-judge aligns
semantically equivalent nodes and diagnoses discrepancies between
graphs. These graph-level discrepancy diagnostics are converted
into natural-language instructions and stored in a Differential Rea-
soning Knowledge Base (DR-KB). At inference, we retrieve top- 𝑘
instructions via Retrieval-Augmented Generation (RAG) to aug-
ment the agent prompt and patch likely logic gaps. Evaluation on
open medical question answering (QA) benchmarks and a Return
Visit Admissions (RVA) prediction task from internal clinical data
demonstrates gains over baselines, improving both final-answer
accuracy and reasoning fidelity. Ablation studies confirm gains
from infusing reference reasoning rationales and the top- 𝑘retrieval
strategy. Clinicians’ review of the output provides further assurance
of the approach. Together, results suggest that DRL supports more
reliable clinical decision-making in complex reasoning scenarios
and offers a practical mechanism for deployment under limited
token budgets.
Keywords
Clinical Decision Support, Large Language Models, Reasoning Align-
ment, Graph Edit Distance, Retrieval-Augmented Generation
1 Introduction
Clinical decision support requires not only correct answers but
also clinically valid reasoning [ 2,43]. In real-world clinical settings,
clinicians do not blindly accept a model’s output as a standalone
label; they pay attention to whether the conclusion is supported by
patient-specific evidence, whether key contraindications have beenconsidered, and whether the recommended actions are consistent
with established clinical guidelines and standard practice [ 1]. As
large language models (LLMs) become the backbone of “clinical
agents” [ 7,9,32], the systems that read patient context and gen-
erate diagnostic suggestions, treatment recommendations, or risk
predictions, the gap between superficially plausible and reliable rea-
soning becomes a central barrier to real-world deployment and user
acceptance. This gap is especially acute in high-stakes settings such
as emergency medicine [ 11,17], where omission of a single risk
factor or an unjustified inference can change disposition decisions,
follow-up plans, and ultimately patient outcomes.
A major practical challenge is that contemporary LLM agents
typically express reasoning in free-form text (e.g., chain-of-thought,
CoT) [ 28]. CoT can be helpful for prompting models to expose in-
termediate steps, but it is difficult to compare, audit, and improve
systematically across cases. The same clinical concept may be para-
phrased in countless ways; important evidence may be scattered
across a narrative; and many clinically meaningful failures are struc-
tural rather than lexical. For example, an agent may reach a correct
answer for the wrong reasons, omit an essential intermediate hy-
pothesis, or suggest an inappropriate clinical action even while
citing superficially relevant facts. In other cases, the agent may
hallucinate irrelevant findings that accidentally steer the conclu-
sion toward the correct option but degrade trust and safety. These
issues indicate that focusing only on final-answer accuracy is in-
sufficient: what is needed is a mechanism to detect and correct
reasoning gaps, the missing, spurious, or misconnected elements
in the agent’s reasoning process.
At the same time, clinical domains often provide rich but under-
utilized sources of high-quality reasoning. These sources include
physician-authored clinical rationale, clinical guidelines, expert
knowledge, and even the rationales produced by more capable
teacher models. Such references encode not just what the answer
should be, but why it should be so in a way that is grounded in
medical principles. However, turning these references into system-
atic improvements for an LLM agent remains challenging. Naively
adding more exemplars at inference time can be ineffective and
quickly runs into context-length limits, while fine-tuning can bearXiv:2602.09945v1  [cs.AI]  10 Feb 2026

Jinsong Liu, Yuhang Jiang, Ramayya Krishnan, Rema Padman, Yiye Zhang, and Jiang Bian
Figure 1: Pipeline for Differential Reasoning Learning (DRL), including Differential Knowledge Mining and Differential
Knowledge-Augmented Inference.
costly and may still fail to generalize when the deployment distri-
bution shifts. This motivates a different question:Can we learn
directly from the discrepancy between an agent’s reasoning
and an expert’s reasoning, and reuse that knowledge to patch
future reasoning failures?
This paper aims to answer the question by casting reasoning
alignment as a knowledge discovery problem. Scientific reasoning
(and clinical reasoning in particular) is inherently relational: facts
support or contradict hypotheses; hypotheses motivate actions;
and actions update beliefs or disposition plans. Errors therefore
manifest not only as wrong statements but as broken relational
structure. We propose to (i) represent reasoning as a structured
object, (ii) quantify the difference between agent and reference
reasoning, and (iii) distill those differences into reusable guidance
or instructions. The result is a framework that turns “reasoning
discrepancies” into a growing resource for improving scientific and
clinical decision-making.
We introduceDifferential Reasoning Learning (DRL), a frame-
work that improves clinical agents by learning from reasoning
discrepancies. DRL has two stages: Differential Knowledge Min-
ing (training) and Differential Knowledge-Augmented Inference
(testing). During training, the framework compares the agent’s
reasoning with a reference reasoning trace for the same case and
mines the discrepancy signal. During inference, it retrieves the
most relevant mined “patches” to guide the agent on new cases.
The complete procedure is summarized in Algorithm 1.
DRL is built around three design goals that matter in clinical
settings. (i)Process-level supervision:the method should target
how the answer is obtained, not only whether the final answer is
correct. (ii)Interpretability and auditability:the improvement
signal should be inspectable by researchers and clinicians, enabling
analysis of recurring failure modes. (iii)Practicality under do-
main shift:the mechanism should help an agent generalize to new
tasks and distributions without requiring expensive re-training for
every deployment environment.To make reasoning comparable across cases, we represent it as
adirected acyclic graph (DAG) 𝐺=(𝑉,𝐸) , where nodes 𝑉cor-
respond to clinical entities and edges 𝐸represent inference steps.
Each graph contains nodes forfacts(symptoms, labs, history, demo-
graphics),hypotheses(diagnoses or clinical conditions, including
ruled-out possibilities), andactions(tests, treatments, assessments,
monitoring plans). Edges encode clinically meaningful relationships
such as a fact supporting a hypothesis, a hypothesis motivating an
action, or a fact directly triggering a test [ 6]. This representation
captures the compositional structure of clinical reasoning in a way
that is difficult to recover from free-form CoT alone.
We use an LLM as a semantic parser to extract these graphs
from unstructured clinical text (Phase 1). For each training case,
we construct (i) areference reasoning graph 𝐺𝑟𝑒𝑓from (near)
gold-standard logic derived from physician-authored clinical ra-
tionale, clinical guidelines, expert knowledge, or higher-capacity
models, and (ii) anagent reasoning graph 𝐺𝑎𝑔𝑒𝑛𝑡 extracted from
the agent’s CoT for the same case. This design addresses a key
practical constraint: while physician-authored graphs are costly
and difficult to scale for learning, high-quality textual rationales
are far more readily available.
Graph extraction also enables a key advantage for knowledge
discovery: it makes error analysis compositional. For example, miss-
ing an entity such as an important risk factor or a social support
related factor can be detected as a missing node, regardless of how
it was phrased in the text. Similarly, incorrect clinical actions can be
detected as action nodes or edges that should not appear under the
case context. This modularity is essential for accumulating reusable
knowledge from many diverse cases.
Given𝐺𝑟𝑒𝑓and𝐺𝑎𝑔𝑒𝑛𝑡 , DRL performs a clinically weighted graph
edit distance (GED)-based discrepancy analysis (Phase 2) to pro-
duce an edit-based diagnostic report. We use the underlying edit
structure to capture three clinically meaningful categories of dis-
crepancy: missing clinically relevant factors ( 𝑣𝑚𝑖𝑠𝑠), hallucinated or
irrelevant factors ( 𝑣ℎ𝑎𝑙𝑙𝑢𝑐 ), and incorrect or incomplete reasoning
paths (𝑒𝑑𝑖𝑓𝑓). This diagnostic view directly targets process-level

Closing Reasoning Gaps in Clinical Agents with Differential Reasoning Learning
errors: it can identify cases where the agent reaches the correct
answer while omitting essential reasoning steps, and it can flag
erroneous reasoning that introduces unsupported claims.
Clinical reasoning is rich in paraphrase and contextual equiv-
alence; thus, exact string matching between nodes is insufficient.
To make GED robust, we use LLM-as-a-judge to perform semantic
node matching and context-based reasonableness checks [ 10]. Node
types are weighted to reflect clinical importance. The judge outputs
an interpretable discrepancy profile: (i) which clinically important
nodes are missing or mismatched, (ii) which nodes are hallucinated
or irrelevant, and (iii) whether the key reasoning path/edge struc-
ture is consistent with the reference. When relevant, we summarize
the severity per category (node mismatch, hallucination, wrong
reasoning path), but the downstream supervision of instruction gen-
eration is driven by the identified edits and affected nodes/edges.
The correctness of the answers is also used as a sanity check for
the plausibility of the discrepancy analysis. This structured discrep-
ancy signal is crucial for transforming reasoning differences into
actionable feedback.
A discrepancy analysis alone does not improve a model; it must
be translated into supervision the agent can apply. The core novelty
of DRL is to treat graph-level discrepancy diagnostics as opportuni-
ties to mine reusable corrective knowledge. An “Insight Generator”
LLM consumes the discrepancy report (matched/missing nodes, hal-
lucinations, and path/edge errors) together with the case context
and produces a structured, actionablenatural-language instruc-
tion(Phase 3). We generate instructions whenever the analysis
surfaces non-trivial missing/hallucinated/path issues. Each instruc-
tion is designed to be directly usable in prompts and to generalize
beyond the originating case: it specifies what went wrong, why
it matters clinically, risks of the error, prevention steps, and con-
textual triggers (e.g., keywords or patterns that indicate when the
guidance applies).
These instructions are stored in aDifferential Reasoning
Knowledge Base (DR-KB). DR-KB can be viewed as a repository
of distilled “patches” that close recurring logic gaps. For example,
if the agent misses “living alone” as a relevant factor for an elderly
fall patient, DRL may store an instruction such as: “In geriatric fall
cases, explicitly assess social support.” By converting graph-level
discrepancies into compact guidance and instructions, DR-KB sup-
ports accumulation of corrective knowledge across many cases, a
property that mirrors the real-world practice of patient care quality
and safety review, where lessons learned from error analysis are
codified into checklists and guidelines.
At inference time, DRL uses DR-KB to proactively steer the agent
toward clinically valid reasoning (Phase 4). Given a new clinical
case or question, we form a retrieval query from the current context,
retrieve the top- 𝑘most relevant instructions, and inject them into
the agent prompt. This is a form of retrieval-augmented generation
(RAG), but with a crucial difference: what is retrieved is not raw
documents or exemplars, but error-targeted reasoning guidance
learned from prior reasoning discrepancies for each case.
This retrieval-based design has several practical advantages. First,
it improves the agent without parameter updates, making it easy
to deploy and iterate in settings with strict governance or limited
resource availability. Second, it is inherently auditable: the retrievedinstructions can be inspected, filtered, or edited, and clinical stake-
holders can review the system’s “reasoning patches.” Third, it sup-
ports fast adaptation. When new failure modes are observed (e.g., in
a new patient population or institutional workflow), DR-KB can be
updated by mining additional discrepancies without re-training the
base model. Finally, the top- 𝑘mechanism provides a direct handle
on the trade-off between guidance coverage and context budget,
enabling controlled use of external reasoning knowledge.
We evaluate DRL on both open benchmarks and an internal clin-
ically grounded task. For open-source medical question answering
(QA), we use MedQA and MedMCQA, two widely adopted multiple-
choice datasets that assess medical knowledge and reasoning. To
test robustness under realistic distribution shift, we further curate
a real-world hospital dataset focused onReturn Visit Admission
(RVA)prediction: predicting whether a patient will be readmitted
after an emergency department (ED) visit based on the patient’s
ED assessment. RVA is defined as an ED return visit resulting in
hospital admission within 9 days of the index visit, and is clinically
meaningful because such patients exhibit higher morbidity and
mortality than those admitted initially. We converted clinical notes
written by ED physicians into RVA-focused QA pairs, yielding a
dataset similar in format to MedQA and MedMCQA.
Our experiments demonstrate that learning from reasoning-
graph differences yields substantial improvements, especially under
domain shift. On the RVA-QA dataset, our DRL framework achieves
81.28±0.47accuracy, outperforming the strongest baseline among
the evaluated models (Qwen3-8B at 56.97 ±0.57) by24.31 points
with low variance across runs (Table 1). Notably, several baselines
that are fine-tuned on open-source medical QA data perform poorly
on RVA, highlighting the limits of transferring answer-focused fine-
tuning to institution-specific clinical prediction tasks.
This work makes the following contributions:
•A discrepancy-driven alignment framework.We pro-
pose Differential Reasoning Learning (DRL), which im-
proves clinical agents by explicitly learning from struc-
tured discrepancies between agent reasoning and reference
reasoning derived from physicians, clinical guidelines, ex-
pert knowledge, physician-authored clinical rationale, or
higher-capacity models.
•Graph-based reasoning representation and discrep-
ancy analysis.We represent clinical reasoning as DAGs
and introduce a clinically weighted GED-based discrepancy
analysis with an LLM-as-a-judge for semantic matching,
yielding interpretable edit diagnostics (missing factors, hal-
lucinations, and path/edge errors) beyond final-answer cor-
rectness.
•Differential Reasoning Knowledge Base (DR-KB).We
convert graph-level discrepancies into reusable natural lan-
guage instructions and store them in a retrievable knowl-
edge base, enabling accumulation of corrective “patches”
across future, unseen cases.
•Retrieval-based inference-time patching.We present
a simple but effective differential knowledge-augmented
inference mechanism that retrieves top- 𝑘DR-KB instruc-
tions and injects them into the agent prompt to close likely
logic gaps without parameter updates.

Jinsong Liu, Yuhang Jiang, Ramayya Krishnan, Rema Padman, Yiye Zhang, and Jiang Bian
•Empirical validation on benchmarks and internal
EHR data.We evaluate on MedQA, MedMCQA, and the
RVA-QA task, and provide ablations isolating the effects of
physician rationales and retrieval depth.
The remainder of the paper details the DRL pipeline (reasoning
graph extraction, GED-based discrepancy measurement, instruc-
tion generation, and retrieval-augmented inference) and evaluates
it on open medical QA benchmarks and the internal RVA setting.
Overall, our results suggest that learning from reasoning-graph dif-
ferences offers a practical route toward more reliable clinical agents:
it produces explicit, auditable diagnostics of reasoning failures, dis-
tills them into reusable knowledge, and applies that knowledge to
guide future reasoning in clinically grounded settings. By mining
the cognitive gap between agents and experts, we move closer to
the goal of trustworthy, reasoning-aligned AI in medicine, ensuring
that right clinical decision support is provided for the right reasons.
2 Related Work
Aligning clinical agents requires not only accurate predictions but
also faithful reasoning processes that can be evaluated, corrected,
and improved over time. Our work is most closely related to re-
search in (a) structured representations of clinical reasoning, (b)
process-level reasoning supervision and alignment, and (c) agentic
memory and error-driven self-improvement. Our work also draws
parallels with existing approaches such as instruction tuning and
preference learning [ 13,42]; while instruction tuning improves
task compliance and preference learning optimizes outputs toward
human-judged quality, neither provides explicit supervision over
reasoning structure, leaving clinical grounding, guideline consis-
tency, and robustness to missing risk factors largely implicit.
2.1 Structured Representations of Clinical
Reasoning
Recent advances have leveraged Large Language Models (LLMs)
as semantic parsers to extract structured representations from un-
structured clinical text. While early work focused on entity-centric
clinical knowledge graph (KG) construction [ 23,24,27,29], re-
cent systems increasingly emphasize document-specific, dynami-
cally induced structures. For example, Zhang et al . [36] proposed
MedKGent, a multi-agent framework that constructs temporally
evolving medical graphs from large-scale biomedical corpora, while
Wang et al . [26] introduced a retrieval-augmented pipeline that
grounds extracted entities and relations in clinical guidelines.
Beyond entity-relation extraction, graph-based representations
have been shown to better support reasoning-intensive clinical
queries. Wu et al . [31] demonstrated that hierarchically indexed
graph structures outperform flat retrieval for longitudinal and
context-dependent medical reasoning. These approaches motivate
the use of structured graphs as an intermediate representation. In
contrast to prior work centered on knowledge accumulation or
retrieval efficiency, our focus is on extracting reasoning graphs that
explicitly encode hypotheses, evidence, and inferential steps, en-
abling direct comparison between model-generated and reference
reasoning processes.2.2 Process-Level Reasoning Alignment and
Supervision
Despite strong performance on clinical benchmarks, LLMs often
exhibit a mismatch between answer correctness and reasoning
validity, a phenomenon commonly referred to as the reasoning
gap [ 30]. Prior studies emphasize that clinical safety depends on
alignment with expert reasoning processes rather than outcome
accuracy alone [ 12,19]. This insight has driven a shift from outcome-
supervised reward models to process-supervised approaches [ 40].
For instance, Wang et al . [22] showed that supervising step-by-step
reasoning enables reliable detection of latent logical errors that
outcome-based supervision overlooks. Complementarily, structural
metrics have emerged as principled tools for evaluating reason-
ing quality. Verma et al . [21] used GED to quantify topological
discrepancies between generated and expert reasoning structures,
effectively penalizing unsupported or hallucinated inference steps.
Our work builds on this line of research by using structural discrep-
ancies not only for evaluation but also as an explicit signal to drive
downstream correction and learning.
2.3 Agentic Memory and Error-Driven
Self-Improvement
Learning from past experience is a central theme in agentic systems,
where memory mechanisms enable agents to accumulate, retrieve,
and reuse knowledge over time [ 39]. Existing approaches explore
diverse memory formats, including textual buffers [ 15], latent em-
beddings [ 25], and structured graph memories [ 3,33]. Most prior
work emphasizes learning from successful trajectories or task com-
pletions [ 20,41], with memory serving as a repository of positive
experiences.
More recently, researchers have begun to investigate learning
from failures. Ouyang et al . [14] proposed ReasoningBank, which
curates incorrect reasoning traces to improve future performance.
Our approach aligns with this emerging direction but differs in
mechanism: rather than storing raw past experiences, we distill
reasoning discrepancies into reusable, structured guidance. This
formulation connects agentic memory with process-level reasoning
alignment, enabling targeted retrieval of corrective instructions
conditioned on observed reasoning gaps.
3 Methodology
Differential Reasoning Learning (DRL)improves a clinical
agent by learning from reasoning discrepancies between the agent’s
reasoning and a reference reasoning trace derived from physicians,
clinical guidelines, expert rationales, or a more capable teacher
model. Because clinical questions are typically multi-factorial in-
volving findings, intermediate hypotheses, and recommended ac-
tions, we represent each reasoning trace as a reasoning graph: a
directed acyclic graph (DAG) that makes the underlying relational
structure explicit. DRL then (i) performs a clinically weighted GED-
based discrepancy analysis (with an LLM-as-a-judge for semantic
matching) to obtain structured edit diagnostics: what is missing,
what is hallucinated, and where the reasoning path deviates, (ii) dis-
tills high-discrepancy cases into reusable natural-language “patches”
(instructions), and (iii) retrieves the most relevant patches at infer-
ence time to close likely logic gaps. The overall procedure follows

Closing Reasoning Gaps in Clinical Agents with Differential Reasoning Learning
two stages, Differential Knowledge Mining (training) and Differen-
tial Knowledge-Augmented Inference (testing), implemented in the
four phases below and summarized in Algorithm 1.
3.1 Phase 1: Reasoning Graph Extraction
We use a large language model (LLM) as a semantic parser to extract
DAG-structured reasoning graphs from unstructured clinical text.
For each case, we construct two graphs:
•Reference reasoning graph ( 𝐺ref):a case-specific “gold-
standard” reasoning trace grounded in clinical guidelines,
clinical knowledge, physician-authored clinical rationale,
or a higher-capacity model.
•Agent reasoning graph ( 𝐺agent):the reasoning trace in-
duced from the agent’s chain-of-thought (CoT) for the same
case, extracted by the same graph extractor to avoid under-
specified comparisons between free-form rationales.
Each graph is denoted as 𝐺=(𝑉,𝐸) , but we additionally im-
pose a typed, evidence-grounded representation so that reasoning
discrepancies can be measured reliably. Because manually curated
reasoning graphs by physicians are expensive, we rely on a strong
instruction-following LLM to extract 𝐺refdirectly from clinical text,
incorporating physician-reviewed rationales when available to im-
prove faithfulness. The same schema is used for 𝐺agent so that down-
stream matching reflects substantive reasoning gaps rather than
formatting artifacts. Concretely, the extractor outputs a structured
JSON artifact (Appendix A.1) that partitions 𝑉into four compo-
nents:
•Facts{𝐹𝑖}capturing observable findings (e.g., symptoms,
vitals, labs, demographics, history) with an optional value
field and apolarityin {present,absent,uncertain};
•Hypotheses{𝐻𝑗}capturing diagnoses or clinical condi-
tions with a level of confidence in {high ,medium ,low,
uncertain};
•Actions{𝐴𝑘}capturing clinical actions with action_type
in {TEST ,TREAT ,ASSESS ,OBSERVE ,PRESCRIBE } and astatus
field (e.g.,recommended,contraindicated);
•Finalnode representing the QA answer or final disposition
or diagnosis.
Each node carries an evidence list support containing verbatim
quotes from the case text and (when applicable) references to other
node IDs, enabling both manual inspection and LLM-based adjudi-
cation.
Edges𝐸are directed, binary relations with a single source and
a single destination, encoded as tuples (src, dst, type) where
type∈ {supports ,contradicts (i.e., rules out), suggests_test }.
The extractor instantiates edges following common clinical rea-
soning patterns, including Fact →Hypothesis, Hypothesis →Action,
Fact→Action, Action→Hypothesis, and links into the Final node.
Each edge additionally stores a short justification field in nat-
ural language, which we retain as auxiliary metadata for inter-
pretability. We constrain the overall structure to a DAG to mirror
the forward flow from evidence to intermediate hypotheses to ac-
tions and conclusions, and to keep downstream graph matching
well-behaved. The detailed JSON schema of the graphs is provided
in Appendix A.1.3.2 Phase 2: Measuring the Reasoning Gap
We quantify reasoning alignment using a clinically weighted GED-
based discrepancy analysis, defined via the minimum-cost edit
script that aligns 𝐺agent to𝐺ref. For interpretability, we explicitly
separate the edits into clinically relevant missing factors, halluci-
nated factors, and structural (edge/path) inconsistencies, and we
treat these components as the primary output:
D(𝐺 agent,𝐺ref)= 𝑑miss,𝑑halluc,𝑑path
=∑︁
𝑐(𝑣miss),∑︁
𝑐(𝑣halluc),∑︁
𝑐(𝑒diff)
.(1)
Here𝑣missare clinically important nodes present in 𝐺refbut ab-
sent in𝐺agent,𝑣halluc are nodes introduced by the agent that are
unsupported or irrelevant in context, and 𝑒diffcaptures edge-level
discrepancies (e.g., incorrect directional support, missing links re-
quired for the reference reasoning path, or spurious inference steps).
DRL conditions instruction generation on the discrepancy report
(the categorized edits and affected nodes/edges), which is stable
across tasks and prompts.
To make GED robust to clinical paraphrases, we use an LLM-as-a-
judge to perform semantic node matching and context-sensitive rea-
sonableness checks, rather than relying on exact string overlap. We
assign higher penalties to failures on clinically consequential node
types, weighting Facts by1 .0×, Hypotheses by1 .5×, and Actions
by2.0×. The judge outputs a discrepancy profile over three cate-
gories: missing/mismatched nodes, hallucinated/irrelevant nodes,
and reasoning path/edge errors. When helpful for summarization,
the judge also reports a normalized severity per category. Answer
correctness is used only as a sanity check for plausibility of the
discrepancy analysis (e.g., to flag inconsistent evaluations), not to
impose fixed numeric regimes. The full prompt for discrepancy
analysis and instruction generation is provided in Appendix A.2
(Step 1).
3.3 Phase 3: Differential Reasoning Knowledge
Base
Rather than treating discrepancies as isolated failures, DRL con-
verts discrepancies into reusable guidance. An Insight Generator
LLM consumes the GED diagnostics including matched and missing
nodes, hallucinations, and path errors, and produces a structured,
actionable instruction that specifies (i) the error type, (ii) the clinical
context in which it arises, and (iii) concrete prevention or remedi-
ation steps (including risks, contraindications, trigger keywords,
and the underlying clinical principle):
I=LLM(“Graph difference𝐷𝑖𝑓𝑓(𝐺 agent,𝐺ref)in context𝐶”)
We store the resulting instructions in theDifferential Reasoning
Knowledge Base (DR-KB)for later retrieval. Appendix A.2 (Step 2)
shows the prompt used for instruction generation.
Example:If the agent fails to account forliving alonewhen rea-
soning about disposition after a geriatric fall, the generated patch
may be:“In geriatric fall cases, explicitly assess social support (e.g.,
living situation) before determining discharge safety. ”

Jinsong Liu, Yuhang Jiang, Ramayya Krishnan, Rema Padman, Yiye Zhang, and Jiang Bian
Algorithm 1DR Learning: Training and Inference Process
1:Input:Training setD={(𝑞 𝑖,𝑎𝑖)}𝑁
𝑖=1
2:Initialize:DR-KBK←∅
3:Stage 1: Differential Knowledge Mining
4:foreach case(𝑞 𝑖,𝑎𝑖)inDdo
5:𝐺 ref←ExtractGraph(𝑞 𝑖,𝑎𝑖,physician rationales)
6: ˆ𝑎𝑖,CoT𝑖←Agent(𝑞 𝑖)
7:𝐺 agent←ExtractGraph(CoT 𝑖)
8:D𝑖←AnalyzeDiscrepancyGED(𝐺 ref,𝐺agent)⊲semantic
matches + missing/hallucinated/path edits
9:ifD 𝑖≠∅then
10:InstructionI←GenerateInsight(𝐺 ref,𝐺agent,D𝑖,𝑞𝑖)
11:K.add(I)
12:end if
13:end for
14:Stage 2: Differential Knowledge-Augmented Inference
15:fornew clinical question𝑞 newdo
16:I retrieved←Retrieve(K,𝑞 new,𝑘)
17:Prompt←“Question:𝑞 new. Guidelines:I retrieved ”
18:Prediction←Agent(Prompt)
19:end for
3.4 Phase 4: Differential Knowledge-Augmented
Inference
At inference, we operationalize DR-KB as an error-targeted retrieval
module. We index instructions with BM25 [ 18] for lexical retrieval
and, given a new clinical question (and its associated context), we:
(1) form a BM25 query from the current clinical context,
(2) retrieve the top-𝑘most relevant DR-KB instructions, and
(3)inject the retrieved instructions into the agent prompt to
dynamically patch likely reasoning gaps.
This is a retrieval-augmented generation (RAG) mechanism in
which the retrieved items are not raw documents or exemplars, but
discrepancy-derived reasoning patches mined from prior failures.
4 Experiments
4.1 Experimental Setup
We conduct comprehensive experiments to evaluate DRL on open-
source medical QA benchmarks and an internal, clinically grounded
dataset. For public benchmarks, we use MedQA [ 8] and MedMCQA
[16], which are widely used to assess medical QA and reasoning [ 2,
30,35]. We additionally curate a dataset for Return Visit Admission
(RVA) prediction [ 37] from electronic health records (EHR) at a
large, urban ED by converting ED clinical notes into RVA-focused
QA pairs. The two sets of experiments, on MedQA/MedMCQA and
on RVA, are meant to test DRL’s ability on different tasks. The
former tests reasoning for facts and diagnostic tasks, whereas the
latter tests reasoning for prognosis and predictive tasks. Each task
has only one right answer.
Across all experiments, we use Qwen3-8B (Qwen) [ 34] and
LLaMA-3.1-8B-Instruct (LLaMA) [ 5] as the backbone agent models
due to their strong performance and practical deployment size, and
to show the robustness of DRL across different base models. For
reasoning-graph extraction and GED-based discrepancy analysis,we use Gemini-2.5-Flash [ 4]. This choice is driven by our require-
ment to run the EHR dataset in a HIPAA-compliant environment
where only HIPAA-compliant Gemini models are available; for con-
sistency, we use Gemini-2.5-Flash across all datasets. For converting
internal EHR data to the QA format, we use Gemini-2.5-Pro [4].
We compare DRL against strong medical QA agent baselines:
MedReason-8B [ 30], fine-tuned on synthetic CoT data derived from
MedQA and MedMCQA; MedPRM-8B [ 35], also trained on these
benchmarks; and HuatuoGPT-o1-8B [ 2], a state-of-the-art medi-
cal reasoning model fine-tuned on multiple medical QA datasets,
including MedQA and MedMCQA.
To account for the stochasticity of LLM generation, we repeat
each experiment five times and report mean accuracy with standard
deviation. Three expert clinicians reviewed the reasoning output
for validation.
4.2 Datasets
4.2.1 Open Medical QA Datasets.We evaluate on open-source
medical QA benchmarks MedQA [ 8] and MedMCQA [ 16], which
consist of multiple-choice questions covering clinical knowledge
and reasoning. As both datasets consist of diverse medical topics,
we restrict the evaluation to questions related to sepsis, chest pain,
and stroke by keyword matching to align with the clinical scope of
our internal EHR dataset. For each dataset, we randomly sample
500 training examples and 100 test examples, ensuring a controlled
experimental setting that aligns the clinical scope and data scale
across benchmarks. Accuracy serves as the primary metric.
4.2.2 Return Visit Admission Prediction.Return visit admissions
(RVA) are ED return visits resulting in hospital admission within 9
days of the index visit [ 37]. These patients exhibit higher morbidity
and mortality than those admitted initially. Drivers include rapid
disease progression, patient non-adherence, and physician error.
We therefore convert ED notes into RVA-focused QA pairs, framing
a binary prediction task that asks whether a patient will experience
an RVA based on the summarized ED encounter, yielding a dataset
comparable in format to open QA benchmarks such as MedQA and
MedMCQA.
We curated 218 ED visits with RVA from electronic health records
(EHR), further reviewed by experienced ED physicians who pro-
vided reasoning rationales. Reasoning rationales include initial and
return diagnostic impressions, and classification for readmission
reasons. The reasoning indicates whether the return visit is illness
related, physician related, healthcare system related, patient related,
or unpreventable. These rationales provide expert knowledge for
constructing high-quality reference reasoning graphs 𝐺𝑟𝑒𝑓. For a
balanced dataset, we randomly sampled 218 ED visits without RVA,
totaling 436 QA pairs. To ensure conversion quality, we use the
most capable model available at the study site (Gemini-2.5-Pro)
to generate initial QA pairs. The prompt for this conversion is
provided in Appendix A.3.
4.3 Experimental Results
Figure 2 summarizes accuracy across the three datasets for DRL
with two backbone models: Qwen and LLaMA. DRL consistently
outperforms its corresponding base model, indicating that gains
stem from the proposed discrepancy-driven augmentation rather

Closing Reasoning Gaps in Clinical Agents with Differential Reasoning Learning
than model-specific idiosyncrasies. Improvements on MedQA and
MedMCQA are modest but consistent (roughly 2–4 points), whereas
the largest gains are observed on RVA (approximately +24 points
for Qwen and +15 points for LLaMA). This pattern suggests that
DRL may be especially effective in the more reasoning-intensive
clinical setting. Because RVA QA is derived from long, information-
dense ED notes, and the question asks for prognosis, it imposes
substantially higher reasoning demands than typical benchmark
questions. Correspondingly, the base models perform comparably
or worse on RVA than on MedQA/MedMCQA despite RVA being a
binary prediction task.
Figure 2: Performance gain of DRL over baselines across
datasets.
Table 1 summarizes the comparison between DRL and other
strong baseline models across datasets.
Table 1: Main accuracy comparison results across datasets.1
Method MedQA MedMCQA RVA
DRL(Qwen) 72.20±1.64 64.80±2.1781.28±0.47
Qwen3-8B 70.20±1.10 62.80±0.84 56.97±0.57
DRL(LLaMA) 53.60±3.36 56.60±3.05 65.23±0.37
LLaMA-3.1-8B-Instruct 51.20±4.09 52.20±0.58 49.91±0.69
MedReason-8B72.40±0.5556.80±0.37 53.49±1.43
HuatuoGPT-o1-8B 60.80±1.1065.00±1.1049.54±0.99
MedPRM-8B 62.20±0.84 52.20±0.49 47.98±0.87
Open medical QA benchmarks.With Qwen as the backbone, DRL
achieves 72.20±1.64 on MedQA and 64.80 ±2.17 on MedMCQA. It
consistently outperforms the base Qwen3-8B and all fine-tuned
baselines, with the exception of HuatuoGPT-o1-8B on MedMCQA
and MedReason-8B on MedQA. Notably, DRL performs on par with
HuatuoGPT-o1-8B on MedMCQA and MedReason-8B on MedQA,
despite being restricted to access only a small fraction of the train-
ing data. With LLaMA as the backbone, DRL similarly improves
over LLaMA-3.1-8B-Instruct (53.60 ±3.36 vs. 51.20±4.09 on MedQA;
56.60±3.05 vs. 52.20±0.58 on MedMCQA), though it remains be-
low specialized medical models, while avoiding the cost and re-
duced flexibility of fine-tuning. Overall, DRL yields reliable gains
1DRL reports the best-performing top-𝑘; RVA results use physician rationales.across backbones, whereas general-purpose instruction-tuned mod-
els without domain-specific reasoning support (e.g., LLaMA-3.1-8B-
Instruct) perform substantially worse, underscoring the difficulty
of these benchmarks without targeted reasoning guidance.
RVA-QA and domain shift.The largest gain is observed on RVA-
QA: DRL(Qwen) achieves 81.28±0.47 accuracy, outperforming the
strongest baseline (Qwen3-8B, 56.97 ±0.57) by 24.31 points with low
variance. DRL(LLaMA) yields similar improvements over LLaMA-
3.1-8B-Instruct (65.23 ±0.37 vs. 49.91±0.69; +15.32 points). In con-
trast, models fine-tuned on open medical QA (HuatuoGPT-o1-8B,
MedReason-8B, MedPRM-8B) transfer poorly to RVA-QA, indicat-
ing limited robustness and generalizability under clinical task shift.
These results suggest that discrepancy-mined, retrieval-based in-
structions more effectively address the missing factors and struc-
tural reasoning demands induced by long, information-dense clini-
cal QA tasks.
Ablations: physician rationales, retrieval depth, and ICL.To fur-
ther validate and analyze our method’s effectiveness, we conduct
the following ablation studies on the RVA-QA dataset using Qwen
as the base model:
(1)Include physician-authored clinical rationales in the prompt
when constructing the reference reasoning graph 𝐺𝑟𝑒𝑓(re-
sults reported in Table 1).
(2)Exclude physician rationales from the prompt when con-
structing𝐺 𝑟𝑒𝑓to isolate their impact on performance.
(3)Compare against an in-context learning (ICL) baseline using
the same training dataset to quantify the benefit of learning
from reasoning differences versus learning from exemplars.
Table 2: Ablation on physician rationales and top- 𝑘retrieval.2
Top-𝑘=1 Top-𝑘=3 Top-𝑘=5 Top-𝑘=10
With PHY. RAT. 74.59±0.47 80.00±0.23 81.28±0.47 81.28±0.53
No PHY. RAT. 72.75±0.43 80.37±0.27 79.63±0.34 80.92±0.47
ICL 68.72±0.53 77.80±0.49 74.68±0.47 -
Table 2 compares our method with and without physician ratio-
nales across different top- 𝑘values, alongside the ICL baseline. ICL
was not applicable at top- 𝑘=10 due to context length constraints.
The results demonstrate that physician rationales consistently im-
prove performance, especially at lower top- 𝑘values, indicating that
expert knowledge enhances reference reasoning graph quality and
enables more effective learning from differential reasoning. This
suggests that including physician rationale helps identify critical
reasoning components the agent may overlook, thereby construct-
ing more reliable near-gold-standard reasoning graphs that guide
more accurate instruction generation. Furthermore, the ICL baseline
consistently underperforms both variants of our method, demon-
strating that DRL provides more targeted guidance than traditional
in-context learning. This confirms that learning from reasoning
graph differences under reference model supervision yields more
effective clinical reasoning improvements than merely providing
exemplar cases.
2ICL𝑘=10 exceeds token limit

Jinsong Liu, Yuhang Jiang, Ramayya Krishnan, Rema Padman, Yiye Zhang, and Jiang Bian
Effect of backbone strength.Table 3 shows that Qwen3-8B is sub-
stantially stronger than LLaMA-3.1-8B-Instruct on RVA-QA across
all Top-𝑘values, and DRL improves both backbones consistently.
This suggests that stronger base reasoning capabilities and retrieval-
based patching are complementary: higher-capacity backbones
benefit from the same discrepancy-mined guidance, while weaker
backbones still gain from targeted corrections.
Table 3: Impact of base models on alignment accuracy for
different top-𝑘values.
Top-𝑘=0 Top-𝑘=1 Top-𝑘=3 Top-𝑘=5 Top-𝑘=10
Qwen 56.97±0.57 74.59±0.47 80.00±0.23 81.28±0.47 81.28±0.53
LLaMA 49.91±0.69 62.29±0.79 64.13±0.39 61.65±0.11 65.23±0.37
Physician qualitative review (RVA-QA).To complement auto-
matic metrics with clinical face validity, three experienced clin-
icians reviewed reasoning outputs from both the base agent and
DRL-corrected outputs for RVA-QA cases (see Appendix B for com-
plete case details). Clinicians were given the clinical vignette, the
agent’s answer and the ground truth (i.e., did an RVA actually
occur), the agent reasoning, and DRL-generated assessments and
instructions for improvement. We report summaries of three rep-
resentative RVA cases to illustrate both the advantages and the
limitations identified by clinician review.
•Urinary retention (agent: no RVA; ground-truth: no
RVA):The clinician noted key discharge cues (well-appearing
or functional status; clear ED intervention such as Foley).
They also agreed with the DRL assessment that the agent
reasoning over-branched and referenced undocumented
labs.
•Breast rash/wound infection (agent: no RVA; ground-
truth: RVA):The clinician judged RVA risk as moderate
to high due to baseline risk, leukocytosis, and concerning
wound features, and emphasized that labs + explicit 24–
48 h return precautions signal ED concern. They endorsed
DRL’s assessment that the agent under-synthesized these
risk factors; DRL guidance aligned with clinician feedback.
•Fall + rib fracture + tube bleeding (agent: RVA; ground-
truth: RVA):The clinician noted a reasoning weakness
in the agent: mixing evidence across problems and under-
weighting key admission cues. The DRL guidance was con-
sistent with this feedback. The clinician also flagged an in-
consistency in the DRL assessment, which stated the agent
failed to predict disposition despite a correct prediction.
Overall, clinicians’ reviews indicate that DRL-corrected rea-
soning better aligns with clinical judgment. In cases where the
base agent reasoning exhibited deficiencies (speculative branch-
ing, evidence mismatches, under-weighting of key clinical signals),
DRL instructions systematically corrected these gaps. However,
in some cases, DRL failed to surface certain nuanced deficiencies
that clinicians identified (e.g., subtle cues about treatment response
likelihood). These findings suggest that while DRL substantially
improves reasoning quality, further refinement of the base agent
prompt, specifically, including relevant clinical assessments andgrounding all reasoning strictly in documented evidence, would
reduce the initial deficiency rate and enhance DRL’s effectiveness.
5 Conclusion
We introducedDifferential Reasoning Learning (DRL), a rea-
soning graph discrepancy-driven framework for improving clinical
agents beyond final-answer supervision. DRL contrasts reference
rationales (from physicians, guidelines, or higher-capacity teach-
ers) and an agent’s CoT as typed reasoning DAGs, performs a
clinically weighted GED-based discrepancy analysis with semantic
node alignment, and distills recurring failure patterns into reusable
natural-language patches stored in a Differential Reasoning Knowl-
edge Base (DR-KB). At inference time, DRL retrieves and injects
the most relevant patches to proactively close potential logic gaps.
Across MedQA, MedMCQA, and an RVA-QA task, DRL is on par
with baseline models for simple questions and outperforms baseline
models for complex reasoning tasks derived from real-world care.
This paper has several limitations. First, DRL depends on LLM-
based graph extraction and LLM-as-a-judge matching; extraction
noise or adjudication bias can propagate into the mined patches,
motivating stronger schema validation, uncertainty-aware discrep-
ancy scoring, and selective human review for high-stakes cases.
Second, DRL assumes access to high-quality reference rationales;
thus, broadening coverage to settings with weaker supervision
(e.g., noisy guidelines or incomplete physician rationales) remains
important. Relatedly, we acknowledge that while gold standards
such as guidelines exist, much of clinical medicine is considered
more an art than a science and a gold standard is not available in
all cases encountered in the real world clinical setting [ 38]. Hence,
triangulating between reference reasoning rationales tries to ap-
proximate evidence-based decision making with the best available
knowledge. Third, while our evaluations had some clinicians’ re-
views, future work should include larger-scale expert assessment.
Since each phase of knowledge mining is currently specific to the
training cases, future work should also evaluate the scalability of
DR-KB. Finally, future work can move from flat patch retrieval to
extracting and validating conditional decision-rule fragments from
discrepancy-context pairs, approximating an explicit latent decision
policy that yields more reliable and auditable context augmentation.
Acknowledgments
The work of Ramayya Krishnan and Rema Padman was funded
in part by the National Institute of Standards and Technology
(NIST) through Federal Award ID Number 60NANB24D231 to the
Carnegie Mellon University AI Measurement Science and Engi-
neering Center (AIMSEC). This research was also funded in part
by the National Institute of Health (NIH) awards R01AG076998,
R01AG089445, R01AG083039, and R01AI172875. We thank Drs. Jus-
tine Ko, Rochelle Joly, and Edward J. Schenck for providing clinical
reviews of DRL outputs.

Closing Reasoning Gaps in Clinical Agents with Differential Reasoning Learning
References
[1] Stephanie Cabral, Daniel Restrepo, Zahir Kanjee, Philip Wilson, Byron Crowe,
Raja-Elie Abdulnour, and Adam Rodman. 2024. Clinical reasoning of a generative
artificial intelligence model compared with physicians.JAMA internal medicine
184, 5 (2024), 581–583.
[2]Junying Chen, Zhenyang Cai, Ke Ji, Xidong Wang, Wanlong Liu, Rongsheng
Wang, Jianye Hou, and Benyou Wang. 2024. Huatuogpt-o1, towards medical
complex reasoning with llms.arXiv preprint arXiv:2412.18925(2024).
[3]Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Ya-
dav. 2025. Mem0: Building production-ready ai agents with scalable long-term
memory.arXiv preprint arXiv:2504.19413(2025).
[4]Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen
Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen,
et al.2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multi-
modality, long context, and next generation agentic capabilities.arXiv preprint
arXiv:2507.06261(2025).
[5] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan,
et al. 2024. The llama 3 herd of models.arXiv e-prints(2024), arXiv–2407.
[6] David Hitchcock and Bart Verheij. 2006.Arguing on the Toulmin model. Vol. 10.
Springer.
[7] Yixing Jiang, Kameron C Black, Gloria Geng, Danny Park, James Zou, Andrew Y
Ng, and Jonathan H Chen. 2025. MedAgentBench: a virtual EHR environment to
benchmark medical LLM agents.Nejm Ai2, 9 (2025), AIdbp2500144.
[8]Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter
Szolovits. 2021. What disease does this patient have? a large-scale open domain
question answering dataset from medical exams.Applied Sciences11, 14 (2021),
6421.
[9] Qiao Jin, Zhizheng Wang, Yifan Yang, Qingqing Zhu, Donald Wright, Thomas
Huang, Nikhil Khandekar, Nicholas Wan, Xuguang Ai, W John Wilbur, et al .
2025. Agentmd: Empowering language agents for risk prediction with large-scale
clinical tool learning.Nature Communications16, 1 (2025), 9377.
[10] Haitao Li, Qian Dong, Junjie Chen, Huixue Su, Yujia Zhou, Qingyao Ai, Ziyi
Ye, and Yiqun Liu. 2024. Llms-as-judges: a comprehensive survey on llm-based
evaluation methods.arXiv preprint arXiv:2412.05579(2024).
[11] Lars Masanneck, Linea Schmidt, Antonia Seifert, Tristan Kölsche, Niklas Hunte-
mann, Robin Jansen, Mohammed Mehsin, Michael Bernhard, Sven G Meuth,
Lennert Böhm, et al .2024. Triage performance across large language models,
ChatGPT, and untrained doctors in emergency medicine: comparative study.
Journal of medical Internet research26 (2024), e53297.
[12] Harsha Nori, Nicholas King, Scott Mayer McKinney, Dean Carignan, and Eric
Horvitz. 2023. Capabilities of gpt-4 on medical challenge problems.arXiv preprint
arXiv:2303.13375(2023).
[13] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela
Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al .2022.
Training language models to follow instructions with human feedback.Advances
in neural information processing systems35 (2022), 27730–27744.
[14] Siru Ouyang, Jun Yan, I Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han,
Long T Le, Samira Daruki, Xiangru Tang, et al .2025. Reasoningbank: Scaling
agent self-evolving with reasoning memory.arXiv preprint arXiv:2509.25140
(2025).
[15] Charles Packer, Vivian Fang, Shishir_G Patil, Kevin Lin, Sarah Wooders, and
Joseph_E Gonzalez. 2023. MemGPT: Towards LLMs as Operating Systems. (2023).
[16] Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. 2022.
Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain
question answering. InConference on health, inference, and learning. PMLR,
248–260.
[17] Carl Preiksaitis, Nicholas Ashenburg, Gabrielle Bunney, Andrew Chu, Rana
Kabeer, Fran Riley, Ryan Ribeira, and Christian Rose. 2024. The role of large
language models in transforming emergency medicine: scoping review.JMIR
medical informatics12 (2024), e53787.
[18] Stephen Robertson, Hugo Zaragoza, et al .2009. The probabilistic relevance
framework: BM25 and beyond.Foundations and trends®in information retrieval
3, 4 (2009), 333–389.
[19] Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Mohamed
Amin, Le Hou, Kevin Clark, Stephen R Pfohl, Heather Cole-Lewis, et al .2025.
Toward expert-level medical question answering with large language models.
Nature Medicine31, 3 (2025), 943–950.
[20] Xiangru Tang, Tianyu Hu, Muyang Ye, Yanjun Shao, Xunjian Yin, Siru Ouyang,
Wangchunshu Zhou, Pan Lu, Zhuosheng Zhang, Yilun Zhao, et al .[n. d.]. Chema-
gent: Self-updating memories in large language models improves chemical rea-
soning. InThe Thirteenth International Conference on Learning Representations.
[21] Samidha Verma, Arushi Goyal, Ananya Mathur, Ankit Anand, and Sayan Ranu.
2025. GRAIL: Graph edit distance and node alignment using llm-generated code.
arXiv preprint arXiv:2505.02124(2025).
[22] Hanyin Wang, Chufan Gao, Qiping Xu, Bolun Liu, Guleid Hussein,
Hariprasad Reddy Korsapati, Mohamad El Labban, Kingsley Iheasirim, MohamedHassan, Gokhan Anil, et al .2025. Process-supervised reward models for verifying
clinical note generation: A scalable approach guided by domain expertise. In
Proceedings of the 2025 Conference on Empirical Methods in Natural Language
Processing. 19138–19158.
[23] Ling Wang, Haoyu Hao, Xue Yan, Tie Hua Zhou, and Keun Ho Ryu. 2025. From
biomedical knowledge graph construction to semantic querying: a comprehen-
sive approach.Scientific Reports15, 1 (2025), 8523.
[24] Ruiyu Wang, Tuan Vinh, Ran Xu, Yuyin Zhou, Jiaying Lu, Carl Yang, and Fran-
cisco Pasquel. 2025. Knowledge Graph Augmented Large Language Models for
Disease Prediction.arXiv preprint arXiv:2512.01210(2025).
[25] Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang, Shiyang Li, Jingfeng Yang,
Qingyu Yin, Zheng Li, Xian Li, Bing Yin, et al .2024. MEMORYLLM: Towards
Self-Updatable Large Language Models. InInternational Conference on Machine
Learning. PMLR, 50453–50466.
[26] Zhengda Wang, Daqian Shi, Jingyi Zhao, Xiaolei Diao, Xiongfeng Tang, and
Yanguo Qin. 2025. Automated Construction of Medical Indicator Knowledge
Graphs Using Retrieval Augmented Large Language Models.arXiv preprint
arXiv:2511.13526(2025).
[27] Chih-Ping Wei, Pei-Yuan Tsai, and Jih-Jane Li. 2025. Biomedical knowledge
graph verification with multitask learning architectures.Journal of Biomedical
Informatics(2025), 104894.
[28] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi,
Quoc V Le, Denny Zhou, et al .2022. Chain-of-thought prompting elicits reason-
ing in large language models.Advances in neural information processing systems
35 (2022), 24824–24837.
[29] Guanchen Wu, Yuzhang Xie, Huanwei Wu, Zhe He, Hui Shao, Xiao Hu, and Carl
Yang. 2025. Utilizing Large Language Models for Zero-Shot Medical Ontology
Extension from Clinical Notes.arXiv preprint arXiv:2511.16548(2025).
[30] Juncheng Wu, Wenlong Deng, Xingxuan Li, Sheng Liu, Taomian Mi, Yifan Peng,
Ziyang Xu, Yi Liu, Hyunjin Cho, Chang-In Choi, et al .2025. Medreason: Eliciting
factual medical reasoning steps in llms via knowledge graphs.arXiv preprint
arXiv:2504.00993(2025).
[31] Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min Xu, Filippo Menolascina,
Yueming Jin, and Vicente Grau. 2025. Medical graph rag: Evidence-based medical
large language model via graph retrieval-augmented generation. InProceedings of
the 63rd Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers). 28443–28467.
[32] Ran Xu, Yuchen Zhuang, Yishan Zhong, Yue Yu, Xiangru Tang, Hang Wu,
May Dongmei Wang, Peifeng Ruan, Donghan Yang, Tao Wang, et al .2025. Meda-
gentgym: Training llm agents for code-based medical reasoning at scale. InThe
Second Workshop on GenAI for Health: Potential, Trust, and Policy Compliance.
[33] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang.
2025. A-mem: Agentic memory for llm agents.arXiv preprint arXiv:2502.12110
(2025).
[34] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al .2025. Qwen3 technical
report.arXiv preprint arXiv:2505.09388(2025).
[35] Jaehoon Yun, Jiwoong Sohn, Jungwoo Park, Hyunjae Kim, Xiangru Tang, Daniel
Shao, Yong Hoe Koo, Ko Minhyeok, Qingyu Chen, Mark Gerstein, et al .2025.
Med-PRM: Medical reasoning models with stepwise, guideline-verified process
rewards. InProceedings of the 2025 Conference on Empirical Methods in Natural
Language Processing. 16565–16582.
[36] Duzhen Zhang, Zixiao Wang, Zhong-Zhi Li, Yahan Yu, Shuncheng Jia, Jiahua
Dong, Haotian Xu, Xing Wu, Yingying Zhang, Tielin Zhang, et al .2025. Med-
KGent: A Large Language Model Agent Framework for Constructing Temporally
Evolving Medical Knowledge Graph.arXiv preprint arXiv:2508.12393(2025).
[37] Yiye Zhang, Yufang Huang, Anthony Rosen, Lynn G Jiang, Matthew McCarty,
Arindam RoyChoudhury, Jin Ho Han, Adam Wright, Jessica S Ancker, and
Peter AD Steel. 2024. Aspiring to clinical significance: Insights from developing
and evaluating a machine learning model to predict emergency department
return visit admissions.PLOS Digital Health3, 9 (2024), e0000606.
[38] Yiye Zhang, Rema Padman, and Nirav Patel. 2015. Paving the COWpath: Learning
and visualizing clinical pathways from electronic health record data.Journal of
biomedical informatics58 (2015), 186–197.
[39] Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Jieming Zhu,
Zhenhua Dong, and Ji-Rong Wen. 2025. A survey on the memory mechanism of
large language model-based agents.ACM Transactions on Information Systems
43, 6 (2025), 1–47.
[40] Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen
Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. 2025. The lessons of de-
veloping process reward models in mathematical reasoning.arXiv preprint
arXiv:2501.07301(2025).
[41] Longtao Zheng, Rundong Wang, Xinrun Wang, and Bo An. [n. d.]. Synapse:
Trajectory-as-Exemplar Prompting with Memory for Computer Control. InThe
Twelfth International Conference on Learning Representations.
[42] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario
Amodei, Paul Christiano, and Geoffrey Irving. 2019. Fine-tuning language models
from human preferences.arXiv preprint arXiv:1909.08593(2019).

Jinsong Liu, Yuhang Jiang, Ramayya Krishnan, Rema Padman, Yiye Zhang, and Jiang Bian
[43] Yuxin Zuo, Shang Qu, Yifei Li, Zhangren Chen, Xuekai Zhu, Ermo Hua, Kaiyan
Zhang, Ning Ding, and Bowen Zhou. 2025. Medxpertqa: Benchmarking expert-
level medical reasoning and understanding.arXiv preprint arXiv:2501.18362
(2025).

Closing Reasoning Gaps in Clinical Agents with Differential Reasoning Learning
A Prompts
A.1 Reasoning Graph extraction prompt for QA pairs
CLINICAL_NOTE_QA_PROMPT ='''
You are a medical AI expert. You will be give a question generated from clinical note, and corresponding answer.
Extract the reasoning graph from this question-answer pair.
## QUESTION generated from CLINICAL NOTE:
{question}
## ANSWER:
{answer}
## EXTRACTION INSTRUCTIONS:
1. **FACTS (F1, F2, ...)**: Extract observable clinical findings:
- Vital signs, lab values, physical exam findings
- Patient demographics, medical history
- Symptoms reported, medications listed
- Use "present", "absent", or "uncertain" for polarity
- Include verbatim quote from the note as support
2. **HYPOTHESES (H1, H2, ...)**: Extract diagnoses or clinical conditions:
- Primary and secondary diagnoses mentioned
- Suspected or ruled-out conditions
- Problem list items that represent clinical reasoning
- Confidence based on how definitively stated in the note
3. **ACTIONS (A1, A2, ...)**: Extract clinical actions documented:
- Medications prescribed (PRESCRIBE)
- Tests ordered (TEST)
- Treatments given (TREAT)
- Assessments performed (ASSESS)
- Monitoring plans (OBSERVE)
4. **FINAL**: The final answer of the question, which is given
5. **EDGES (E1, E2, ...)**: Infer connections between nodes:
- Fact→Hypothesis: What findings support which diagnoses?
- Hypothesis→Action: What conditions led to which treatments?
- Fact→Action: Direct fact-to-action relationships
- Action→Hypothesis: What actions suggest which diagnoses?
- Fact or Hypothesis or Action→FINAL: What overall conclusion is supported?
- Use "supports", "contradicts", or "suggests_test" as edge types
- For each edge, there should be only one source and one destination
## OUTPUT SCHEMA:
{{
"artifact_id": "{artifact_id}",
"qid": "{qid}",
"guideline_bundle_id": "extracted",
"facts": [
{{
"fid": "F1",
"label": "finding description",
"value": "specific value or null",
"polarity": "present|absent|uncertain",
"support": [{{"source": "case", "ref": null, "quote": "verbatim from note"}}]
}}
],
"hypotheses": [
{{
"hid": "H1",
"label": "diagnosis or condition",

Jinsong Liu, Yuhang Jiang, Ramayya Krishnan, Rema Padman, Yiye Zhang, and Jiang Bian
"confidence": "high|medium|low|uncertain",
"support": [{{"source": "case", "ref": "F1", "quote": "supporting text"}}]
}}
],
"actions": [
{{
"aid": "A1",
"label": "action description",
"action_type": "TEST|TREAT|ASSESS|OBSERVE|PRESCRIBE",
"status": "recommended|contraindicated|optional|not_specified",
"support": [{{"source": "case", "ref": null, "quote": "from note"}}]
}}
],
"edges": [
{{
"eid": "E1",
"src": "F1",
"dst": "H1",
"type": "supports|contradicts|suggests_test",
"justification": "why this connection exists based on the note",
"guideline_support": null
}}
],
"final": {{
"answer": "{answer}",
"answer_text": "reasoning behind the answer",
"confidence": "high|medium|low|uncertain",
"uncertainties": ["list of unclear aspects"]
}}
}}
Return ONLY the JSON object.
'''
A.2 GED and instruction prompt
prompt = """You are a medical expert evaluating an agent's reasoning quality using Graph Edit Distance (GED).
QUESTION: {question}
CORRECT ANSWER: {answer}
AGENT ANSWER CORRECT: {'YES'if answer_correct else'NO'}
REFERENCE GRAPH (G_REF):
Facts: {g_ref_facts}
Hypotheses: {g_ref_hyps}
Actions: {g_ref_actions}
AGENT GRAPH (G_AGENT):
Facts: {g_agent_facts}
Hypotheses: {g_agent_hyps}
Actions: {g_agent_actions}
---
STEP 1: CALCULATE GED SCORE
Evaluate GED between G_REF and G_AGENT using these rules:
1. SEMANTIC NODE MATCHING: Match by meaning, not exact text
2. CONTEXT-BASED REASONABLENESS: Accept facts from question context
3. NODE IMPORTANCE: Facts=1.0x, Hypotheses=1.5x, Actions=2.0x
4. ANSWER IMPACT:
- Correct + correct path: GED 0.1-0.3
- Correct + wrong path: GED 0.6-0.8

Closing Reasoning Gaps in Clinical Agents with Differential Reasoning Learning
- Wrong: GED > 0.8
5. REASONING PATH QUALITY: Check edge connectivity
IMPORTANT CALCULATION FORMULA:
- Calculate three penalty components (each between 0.0-1.0):
* node_mismatch_penalty: Penalty for missing or mismatched nodes
* hallucination_penalty: Penalty for hallucinated (extra) nodes
* reasoning_path_penalty: Penalty for incorrect reasoning path structure
- Calculate total_ged_score EXACTLY as:
total_ged_score = node_mismatch_penalty + hallucination_penalty + reasoning_path_penalty
- The total_ged_score can exceed 1.0 if the sum of penalties is greater than 1.0
---
STEP 2: GENERATE ENHANCED IMPROVEMENT INSTRUCTIONS
Generate improvement instructions that could be used in a future prompt
when the agent encounters similar medical questions. These instructions should help
the agent avoid the same mistakes and reason more effectively.
Each instruction should include:
{
"title": "Brief error title",
"error_type": "diagnosis_error|treatment_error|reasoning_error|missed_fact",
"clinical_domain": "<domain>",
"situation_context": "This instruction applies when: [clinical scenario where error occurred]",
"example_situation": "For example: [specific example from this case]",
"similar_case_patterns": [
"Pattern 1: [how to recognize similar cases]",
"Pattern 2: [clinical signs to watch for]",
"Pattern 3: [key indicators]"
],
"what_went_wrong": "[Specific mistake made]",
"why_it_matters": "[Clinical consequences and impact]",
"potential_risks": [
"Risk 1: [specific bad outcome if error repeated]",
"Risk 2: [patient safety impact]",
"Risk 3: [clinical consequences]"
],
"prevention_steps": [
"Step 1: [concrete action]",
"Step 2: [concrete action]",
"Step 3: [concrete action]"
],
"contraindications": [
"DO NOT: [specific thing to avoid]",
"NEVER: [explicit warning]",
"AVOID: [specific practice]"
],
"trigger_keywords": ["keyword1", "keyword2"],
"medical_principle": "[Clinical foundation]",
"similar_case_examples": "[When this guidance applies]"
}
Important: Include practical, actionable guidance that prevents the EXACT error that occurred.

Jinsong Liu, Yuhang Jiang, Ramayya Krishnan, Rema Padman, Yiye Zhang, and Jiang Bian
---
CRITICAL: When calculating total_ged_score, you MUST:
1. First calculate node_mismatch_penalty, hallucination_penalty, and reasoning_path_penalty (each 0.0-1.0)
2. Then calculate: total_ged_score = node_mismatch_penalty + hallucination_penalty + reasoning_path_penalty
3. Do NOT use any other formula or normalization - use the exact sum of the three penalties
---
OUTPUT FORMAT:
Return ONLY valid JSON with these fields:
{{
"qid": "{qid}",
"answer_correct": {str(answer_correct).lower()},
"node_matching": {{
"matched_facts": <number>,
"matched_hypotheses": <number>,
"matched_actions": <number>,
"hallucinated_nodes": <list or empty>,
"hallucination_severity": "none|minor|moderate|severe"
}},
"graph_structure": {{
"reasoning_path_correct": <boolean>,
"missing_edges": <list>,
"extra_edges": <list or empty>
}},
"ged_calculation": {{
"node_mismatch_penalty": <0.0-1.0>,
"hallucination_penalty": <0.0-1.0>,
"reasoning_path_penalty": <0.0-1.0>,
"total_ged_score": <EXACTLY: node_mismatch_penalty + hallucination_penalty + reasoning_path_penalty>,
"explanation": "<brief explanation>"
}},
"assessment": {{
"overall": "Excellent|Good|Fair|Poor|Very Poor",
"strengths": <list of 1-2 strengths>,
"weaknesses": <list of 1-2 weaknesses>
}},
"improvement_instructions": [
{{
"title": "Brief title",
"error_type": "diagnosis_error|treatment_error|reasoning_error|missed_fact",
"clinical_domain": "<domain>",
"situation_context": "This instruction applies when: ...",
"example_situation": "For example: ...",
"similar_case_patterns": ["Pattern 1: ...", "Pattern 2: ..."],
"what_went_wrong": "...",
"why_it_matters": "...",
"potential_risks": ["Risk 1: ...", "Risk 2: ...", "Risk 3: ..."],
"prevention_steps": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
"contraindications": ["DO NOT: ...", "NEVER: ...", "AVOID: ..."],
"trigger_keywords": ["keyword1", "keyword2"],
"medical_principle": "...",
"similar_case_examples": "..."
}}
]
}}

Closing Reasoning Gaps in Clinical Agents with Differential Reasoning Learning
Return ONLY JSON, no other text."""
A.3 Prompt for converting clinical notes into QA pairs
prompt = """
You are a clinician-grade medical language model. Your task is to transform emergency department (ED) clinical notes into a
single, comprehensive clinical question suitable for clinical prognosis and reasoning.
You are given an emergency department clinical note composed of the following sections:
- Chief Complaint
- Patient History
- Review of Systems
- Physical Exam
- Medical Decision Making
- Assessment and Plan
Rewrite the note as one self-contained clinical question that could be posed to a clinical decision-support system or clinician
at the time of ED discharge. The clinical question is concerned with whether this patient will return to ED and get admitted to
inpatient within 9 days after the discharge of the index ED visit. The answer should be yes or no.
Constraints:
1. Do not omit any clinical facts, findings, symptoms, comorbidities, diagnostics, treatments, or disposition details.
2. Do not introduce new information or reinterpret findings.
3. Preserve uncertainty, negations, and clinician reasoning exactly as stated.
4. Use medically precise language.
5. Do not answer the question.
6. Do not include probabilities, risk estimates, or predictions.
7. Remove all names of patients and physicians and all other identifying information.
Output format:
1. A single paragraph written as a clinical question beginning with:“Given the following emergency department presentation and
evaluation. . . ”
2. The question must end with: “Will this patient return to ED and get admitted to inpatient within 9 days after the discharge
of the index ED visit”
Clinical Note:
{clinical_note}
"""
B Physician review details
B.1 Medical scenario 1 (de-identified)
[...]
Medical background information omitted due HPI data.
[...]
Question:Will this patient return to ED and get admitted to inpatient within 9 days after the discharge of the index ED visit?
True answer:No
Agent answer:No
Agent reasoning:
The patient is an 84-year-old male with a history of BPH and HLD, but the medical history isn’t fully documented. He’s presenting with
urinary retention, decreased output, dysuria, and intermittent hematuria. His son reported some fever and chills a week ago. The physical
exam shows suprapubic tenderness, and the ED doctor suspects acute urinary retention, UTI, or cystitis. They plan a Foley catheter,
antibiotics, and discharge.
The question is whether he’ll return to ED and be admitted within 9 days. The answer choices are 0 (no) or 1 (yes).
First, I need to consider the possible diagnoses. The main issues here are urinary retention and possible UTI. BPH is a known cause of urinary
retention, especially in older men. The suprapubic tenderness suggests bladder distension, which aligns with retention. The hematuria and
dysuria could indicate a UTI or even a more severe issue like a bladder infection or stones. The fever and chills a week ago might have been
a past infection, but the current symptoms are more acute.
The ED plan includes a Foley catheter to relieve retention, which should address the immediate problem. Antibiotics are being considered,
which is good for a UTI. However, the patient’s history of BPH might mean that the retention is due to an obstructive issue, which could be
managed with catheterization and possibly further urological evaluation.

Jinsong Liu, Yuhang Jiang, Ramayya Krishnan, Rema Padman, Yiye Zhang, and Jiang Bian
But wait, the differential includes possible cystitis and UTI. If the UTI is severe, it could lead to complications like pyelonephritis, but the ED
doctor has low suspicion for that. The patient’s vital signs are normal, and he’s not in acute distress. The plan is for discharge, so they might
think the issue is manageable outpatient.
However, the presence of hematuria and the history of fever and chills might indicate a more serious infection. If the UTI is not adequately
treated, it could lead to sepsis or kidney involvement. But the ED is planning antibiotics, which should cover that. Also, the Foley catheter
would help with the retention, which might resolve the symptoms.
Another angle: the patient’s age and BPH increase the risk of complications. If the retention is due to BPH, then after catheterization, he
might need further management like medication or surgery. But the question is about returning to ED and being admitted. If the initial
treatment is effective, maybe he won’t need to come back. However, if there’s a complication like sepsis, AKI, or if the UTI is not responding
to antibiotics, he might need readmission.
The ED doctor’s differential doesn’t include high suspicion for pyelonephritis or AKI, so maybe they think it’s a straightforward UTI or
BPH-related retention. The planned evaluations are CBC, chem, UA, and UCx. If those results are normal, then discharge is appropriate. But
if there’s a complication, like a high WBC count or signs of infection, they might admit.
But the question is about the likelihood of return within 9 days. If the patient is treated with a Foley and antibiotics, and the underlying
cause (BPH or UTI) is managed, he might not need to come back. However, if the UTI is severe or if there’s a complication, he might. But the
ED’s assessment is that the most likely is acute urinary retention, which is being treated. The answer choices are 0 (no) or 1 (yes).
Considering the information given, the ED is planning for discharge, and the differential doesn’t suggest high suspicion for severe
complications. So the answer might be 0 (no). But I need to be careful. The presence of hematuria and fever could indicate a more serious
issue. However, the ED doctor is considering UTI and cystitis, which are treatable. If the patient’s symptoms resolve with treatment, he might
not return. But if there’s a complication, he might. However, the question is about the likelihood. Since the ED is planning for discharge and
the differential doesn’t include high suspicion for severe issues, the answer is probably 0.
</think>
1. The patient presents with acute urinary retention, dysuria, hematuria, and a history of BPH, which are classic signs of urinary retention
or a urinary tract infection (UTI).
2. The ED team has initiated a Foley catheter and planned antibiotics, addressing the immediate issue of retention and potential infection.
3. The absence of severe symptoms like flank pain, high fever, or signs of sepsis, along with normal vital signs, suggests a less severe
condition.
4. The differential diagnosis includes UTI and BPH-related retention, both of which are typically managed with outpatient care, including
antibiotics and urological follow-up.
5. The low suspicion for pyelonephritis or acute kidney injury (AKI) further supports the likelihood of a manageable outpatient course.
6. Without evidence of complications or treatment failure, the patient is likely to be discharged and not require readmission within 9 days.
The answer is no.
DRL-generated assessment on agent reasoning:

Closing Reasoning Gaps in Clinical Agents with Differential Reasoning Learning
OverallFair
StrengthCorrectly identified the primary diagnoses (urinary retention, UTI, BPH) and key symptoms
Correctly identified the immediate ED plan (Foley, antibiotics, discharge) and reached the correct final answer
weaknesses Significant number of hallucinated hypotheses and actions, some of which contradict the case (e.g., ’admit’ vs. ’anticipated
discharge’)
Missed several specific facts from the ROS and physical exam, and miscategorized ’discharge’ from a fact to an action
generated instruction These guidelines are crucial for enhancing medical reasoning, particularly in Emergency Medicine, by addressing common
errors in interpreting clinical information. Firstly, avoid hallucinating speculative diagnoses or hypotheses; generating
broad, unsupported diagnoses not explicitly considered by the ED physician leads to unfocused and misleading differentials,
obscuring the actual assessment and suggesting a lack of contextual understanding. This error often occurs when agents
list potential diagnoses beyond the immediate ED scope or not directly supported by clinical findings. To prevent this,
strictly adhere to the physician’s stated differential and level of suspicion, inferring only directly supported hypotheses, and
differentiating between comprehensive textbook knowledge and the specific differential being considered, especially when
trigger keywords like "low suspicion for" or "assessed as" are present. Secondly, it is imperative to adhere to the physician’s
stated plan and avoid contradictory actions. Generating actions not part of the immediate ED plan, or directly contradicting
the stated disposition (e.g., listing ’admit’ when ’discharge’ is planned), misrepresents the physician’s decision-making
and care pathway. This includes inferring long-term management not relevant to the acute ED visit. Prevention involves
prioritizing the physician’s explicit ’Evaluations planned,’ ’Treatments, ’ and ’Disposition, ’ ensuring consistency, and avoiding
speculation on future treatments beyond the current encounter, particularly when phrases like "Disposition is anticipated"
are used. Finally, maintain node type fidelity by accurately categorizing information as Facts, Hypotheses, or Actions.
Miscategorizing a descriptive statement about a plan or outcome (like ’Disposition anticipated discharge’ being a Fact) as
an active intervention (an Action) distorts the graph’s structure and misrepresents the clinical narrative. To prevent this,
carefully distinguish between descriptive statements, diagnostic considerations, and planned interventions, understanding
that a ’disposition’ is a plan/outcome statement (a fact), not an action, especially when trigger keywords like "plan is" or
"reveals" are encountered. In all cases, avoid adding unmentioned or unimplied information, never include contradictory or
highly speculative elements, and always ensure precise categorization to reflect the physician’s immediate assessment and
plan.,
Clinician’s review: Urinary retention (Agent predicted no RVA; clinician agrees)
The clinician emphasized that bedside appearance and functional status (e.g., well-appearing, ambulatory, tolerating PO) and a clear ED
intervention (Foley catheter) are key signals supporting discharge. The base agent reasoning occasionally over-elaborated counterfactual
admit/discharge branches and referenced undocumented laboratory results. DRL instructions explicitly corrected these issues by: (i)
anchoring reasoning on documented findings only, and (ii) prioritizing functional status and direct interventions. The clinician agreed with
DRL-generated assessment that the agent gives multiple possible admit/discharge branches, indicating these instructions are reasonable.
B.2 Medical scenario 2 (de-identified)
[...]
Medical background information omitted due HPI data.
[...]
Question:Will this patient return to ED and get admitted to inpatient within 9 days after the discharge of the index ED visit?
True answer:Yes
Agent answer:No (wrong answer)
Agent reasoning:
The patient is a 65-year-old female (though referred to with female pronouns, but the medical history mentions male with ESI 4, which
might be a typo or confusion. Wait, the initial note says "male with an ESI of 4" but then refers to her with female pronouns. Maybe that’s a
mistake, but I’ll proceed with the info given.
She has a rash on inner thighs and left breast, described as raw, weeping dark fluid with a foul smell. Physical exam shows erythematous,
tender skin with satellite lesions, and some raw areas. Lab shows WBC 14.12, which is elevated. She was treated for intertrigo with nystatin
and Keflex. She was discharged with instructions to return in 24-48 hours or sooner if worsening, and to see dermatology.
The question is whether she’ll return to ED and get admitted within 9 days. The answer choices are 0 (no) or 1 (yes).
First, consider the initial diagnosis: intertrigo. But the presentation includes foul-smelling discharge, which might suggest a fungal infection
(like Candida) or bacterial infection. The left breast has an ulcer with fat layer exposed and cellulitis. The groin and thighs have superficial
ulcerations with foul smell and satellite lesions, suggesting a fungal infection with possible secondary bacterial infection.
The patient has some comorbidities listed (asthma, beta thalassemia, cyclic neutropenia, sarcoidosis), and diabetes is in the PMH but she
denies it. Her glucose is normal. But cyclic neutropenia could predispose to infections.
She is treated with antibiotics (Keflex) and nystatin, and referred to dermatology. She is discharged stable, ambulatory.
The question is about return within 9 days and admission. Given that she has significant skin infection and ulceration, and elevated WBC,
there is a risk of worsening infection, sepsis, or needing IV antibiotics. But they treated with oral antibiotics and antifungal, and she’s stable.

Jinsong Liu, Yuhang Jiang, Ramayya Krishnan, Rema Padman, Yiye Zhang, and Jiang Bian
However, the ulcer of breast with fat layer exposed is concerning. Also, the foul smell suggests possible necrotizing infection? But no
mention of systemic symptoms or high fever. Her vitals are normal and she is not ill-appearing.
She is instructed to return in 24-48 hours for wound check. That suggests the ED expects possible progression or need for further care. But
that could be outpatient wound check, not necessarily admission. But if she returns, she might not be admitted.
Still, the elevated WBC and underlying neutropenia and sarcoidosis could mean she is immunocompromised, which increases risk.
But the question is admission within 9 days, not just return. Many intertrigo cases can be managed outpatient with topical and oral meds.
If she returns, it might be for worsening cellulitis or failure of oral antibiotics. But admission is possible if infection worsens.
But given she is stable, no fever, and treated, I lean towards no admission.
So answer likely 0.
</think>
1. Patient has intertrigo with ulceration and cellulitis, treated with appropriate outpatient medications (nystatin, Keflex).
2. Vital signs stable, no systemic symptoms, and patient ambulatory and not ill-appearing.
3. Although instructed to return for wound check, this is likely precautionary and for outpatient follow-up.
4. Without evidence of treatment failure, severe systemic infection, or inability to manage at home, inpatient admission is unlikely within 9
days.
5. Therefore, the patient is not expected to return and be admitted.
<answer>0</answer>
DRL-generated assessment on agent reasoning:
OverallVery Poor
Strength"Identified the primary diagnosis (intertrigo) and key treatments.",
Recognized the penicillin allergy and the rationale for Keflex use."
weaknesses "Failed to synthesize critical risk factors (elevated WBC, comorbidities, severity of rash, explicit return instructions) to
predict the likelihood of readmission/admission.",
"Hallucinated outcomes and predictions as ’facts’ or ’actions’, which directly led to the incorrect answer and skewed its
reasoning."
generated instruction These instructions highlight critical reasoning errors in medical practice, emphasizing the need for precise information
processing. First, when evaluating patients for ED discharge, especially those with infections or complex medical histories,
it is crucial to integrate a multi-factor risk assessment to accurately predict readmission or inpatient admission. The
common error is failing to synthesize cumulative risks from factors like age, comorbidities, objective infection severity (e.g.,
elevated WBC, foul smell, raw skin, difficulty walking), and explicit discharge instructions (e.g., "return to ED for wound
check"), which leads to underestimating risk, potential patient harm, delayed care, and increased costs. To prevent this,
systematically review all relevant patient data, identify ’red flag’ indicators, and pay close attention to discharge instructions
suggesting early follow-up, rather than solely relying on a patient’s subjective ’feeling well’ or discharge status. Second,
avoid hallucinating or misclassifying outcomes as facts or actions in clinical reasoning. This error occurs when clinicians or
models assume a patient’s future course (e.g., "will likely not return") as an established fact or treat it as a planned action,
leading to flawed decision-making. Prevention requires strict adherence to the presented clinical facts and physician’s
explicit plan, ensuring that predictions remain separate from documented events or intended interventions. When extracting
or generating clinical reasoning graphs, categorize information correctly: facts are objective and documented, hypotheses
are diagnostic considerations, actions are planned interventions, and outcomes are distinct future events that must not be
assumed unless explicitly stated. Always avoid adding unmentioned details, ensure consistent categorization, and prioritize
cautious, evidence-based risk synthesis for patient safety.
Clinician’s review: Rash / wound infection under breast (Agent predicted no RVA; clinician disagreed)
The clinician assessed return risk as moderate to high given baseline risk factors (diabetes; sarcoidosis/possible immunosuppression;
history of neutropenia), leukocytosis (WBC14 .12), and concerning wound features (foul odor, yellow drainage, multifocal involvement).
The clinician further noted that obtaining labs for a wound infection and providing explicit 24–48 h return precautions are strong signals
of clinician concern. The base agent oscillated between low and high risk without anchoring on baseline vulnerability or likelihood of
treatment failure. DRL instructions corrected this by: (i) explicitly weighting baseline comorbidities and immunosuppression, (ii) flagging
when lab workup indicates higher concern, and (iii) recognizing explicit return precautions as high-risk signals. These corrections directly
addressed the reasoning gaps the clinician identified. The weaknesses list generated by DRL also matched what the clinician said: elevated
WBC, comorbidities, rash appearance, explicit return precautions.
B.3 Medical scenario 3 (de-identified)
[...]
Medical background information omitted due HPI data.
[...]

Closing Reasoning Gaps in Clinical Agents with Differential Reasoning Learning
Question:Will this patient return to ED and get admitted to inpatient within 9 days after the discharge of the index ED visit?
True answer:Yes
Agent answer:Yes (correct answer)
Agent reasoning:
The patient is an 81-year-old male with a complex medical history. He presented to the ED with a fall and difficulty breathing. The main
issues are left-sided rib and back pain, along with pink-tinged urine from his left PCNU. He also has a history of CKD stage 4, heart failure,
and multiple other conditions.
First, I need to consider the possible reasons for his symptoms. The chief complaints are a fall and difficulty breathing. The fall caused
left-sided chest trauma, leading to rib and back pain. The PCNU site has pink-tinged urine and surrounding erythema, which might indicate
a complication like infection or bleeding. His weight loss and decreased oral intake could contribute to weakness and possibly malnutrition,
which might affect his recovery.
The differential diagnosis includes rib fracture, musculoskeletal injury, PCNU complications (like infection or blockage), and maybe even a
urinary tract infection (UTI) or pyelonephritis. The CXR didn’t show pneumonia, so that’s less likely. His shortness of breath could be due
to COPD exacerbation or heart failure exacerbation, especially given his low EF and recent admissions.
The ED attending anticipates admission. The patient is ill-appearing, cachectic, tachycardic, and has multiple comorbidities. He has had
multiple admissions in the last month and was discharged yesterday. This suggests he is unstable and likely to require inpatient care.
Additionally, the fact that he’s too weak to cook and has decreased oral intake suggests functional decline. His fall and pain could limit
mobility. His shortness of breath at rest for days suggests a significant pulmonary or cardiac issue. Also, the nephrostomy tube complications
could indicate infection, which in a frail patient could require admission.
Even though the chest X-ray didn’t show pneumonia, he could still have CHF exacerbation or COPD exacerbation. The ED planned labs
including BNP and troponin, which suggests concern for cardiac issues.
Given all these factors, it’s highly likely that he will be admitted during this visit. The question asks whether he will return to ED and be
admitted within 9 days after discharge of the index ED visit. But note, the index visit is this one, and disposition is anticipated admission.
Wait, the question: "return to ED and get admitted within 9 days after discharge of the index ED visit." If he is admitted now, then he hasn’t
been discharged yet. But perhaps the index ED visit ends with discharge, and then we look at return visits. Hmm. Actually, in the prompt,
the disposition is anticipated admission, but maybe he was discharged? It’s confusing. However, since the true answer is yes and AI answer
is yes, likely he was discharged and then returned.
But the ED plan anticipates admission, so he likely will be admitted. The patient has multiple risk factors for readmission, including chronic
illness, recent admissions, functional decline, and device complications. So yes, he is likely to return and be admitted.
</think>
1. The patient is an 81-year-old male with multiple chronic comorbidities (CKD stage 4, HFrEF with EF 15-20%, CAD, COPD, cachexia) and
has had multiple recent admissions, including discharge just yesterday.
2. He presents with significant acute issues: a fall with severe left-sided rib pain, shortness of breath at rest for several days, tachycardia, and
signs of nephrostomy tube complications (pink output, erythema), all of which increase the likelihood of clinical deterioration.
3. He is ill-appearing, cachectic, and functionally declining (decreased oral intake, too weak to cook), suggesting inability to safely manage
at home.
4. The ED clinician’s assessment includes concerns for serious conditions (rib fracture, COPD/CHF exacerbation, nephrostomy infection)
and anticipates admission.
5. Given his frailty, complexity, and acute presentation, the risk of readmission/inpatient admission within 9 days is high.
<answer>1</answer>
DRL-generated assessment on agent reasoning:

Jinsong Liu, Yuhang Jiang, Ramayya Krishnan, Rema Padman, Yiye Zhang, and Jiang Bian
OverallVery Poor
Strength"Identified some key patient facts and the primary diagnostic hypotheses considered by the clinician."
weaknesses"Failed to correctly predict the patient’s disposition (admission/readmission).",
"Significant hallucination of general hypotheses and miscategorization of outcomes as ’actions’.",
Missed a large number of specific facts from the detailed patient presentation.,
Demonstrated severe internal inconsistency by listing ’admission’ as an action while concluding ’NO’ to admission."
generated instruction This instruction addresses two critical reasoning errors: inaccurate risk assessment for admission/readmission in complex
patients and inaccurate node categorization with excessive hallucination. When evaluating complex, elderly patients
with multiple comorbidities presenting with new symptoms (e.g., falls, weight loss, device issues, tachycardia), avoid
underestimating cumulative risk factors, as this reasoning error can lead to inappropriate discharge, patient harm, and
readmissions. Instead, synthesize all significant risk factors, consider their cumulative burden, prioritize patient safety by
erring on the side of caution, and recognize red flags like tachycardia or new device-related symptoms. Simultaneously,
when extracting and categorizing clinical information into Facts, Hypotheses, and Actions, prevent the reasoning error of
miscategorization and vague hypotheses by strictly adhering to definitions: Facts are objective patient data, Hypotheses are
diagnostic considerations, Actions are planned clinician interventions. Never include predicted outcomes as Actions, and
avoid generating speculative hypotheses not grounded in the physician’s stated differential. This ensures accurate clinical
reasoning representation and avoids misleading graphs or downstream decisions.
Clinician’s review: Fall + rib fracture + tube bleeding (Agent predicted RVA; clinician agreed, but flagged reasoning)
Although the final prediction matched, the clinician noted the base agent sometimes conflated evidence across concurrent problems
(e.g., reasoning that a normal chest X-ray addresses renal/tube issues when these require different diagnostic modalities). The clinician also
highlighted high-salience admission cues that the base agent underweighted, including ill-appearing/cachectic presentation and functional
decline (reduced intake; “too weak to cook”). DRL instructions addressed these by: (i) delineating reasoning for concurrent complaints
separately, and (ii) elevating functional status and appearance as primary disposition signals. The clinician pointed out that the issue in the
DRL-generated assessment that the agent failed to predict disposition even though it reasoned correctly.
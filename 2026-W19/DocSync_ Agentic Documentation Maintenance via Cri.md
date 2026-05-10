# DocSync: Agentic Documentation Maintenance via Critic-Guided Reflexion

**Authors**: Sidhesh Badrinarayan, Adithya Parthasarathy

**Published**: 2026-05-04 02:41:33

**PDF URL**: [https://arxiv.org/pdf/2605.02163v1](https://arxiv.org/pdf/2605.02163v1)

## Abstract
Software documentation frequently drifts from executable logic as codebases evolve, creating technical debt that degrades maintainability and causes downstream API misuse. While static analysis tools can detect the absence of documentation, they cannot evaluate its semantic consistency. Conversely, standard Large Language Models (LLMs) offer generative flexibility but frequently hallucinate when updating documentation without deep structural awareness of the underlying code. To address this gap, we propose DocSync, an agentic workflow that frames documentation maintenance as a structurally grounded, iterative generation task. DocSync bridges syntactic changes and natural language descriptions by fusing Abstract Syntax Tree (AST) representations and Retrieval-Augmented Generation (RAG) to provide dependency-aware context. Furthermore, to ensure factual consistency, we incorporate a critic-guided refinement loop based on the Reflexion paradigm, allowing the model to self-correct candidate updates against the source code. We empirically evaluate a resource-constrained implementation of DocSync-using a LoRA-adapted small language model - on a proxy code-to-text maintenance task. Our findings demonstrate that this AST-aware agentic approach substantially outperforms standard encoder-decoder baselines across semantic alignment, summary-line faithfulness, and automated judge preferences (e.g., achieving an automated judge score of 3.44/5.0 compared to 1.91 for CodeT5-base). Crucially, the iterative critic loop yields measurable improvements in semantic correctness without requiring scaled-up parameter counts. These results provide strong evidence that coupling structural retrieval with agentic refinement is a highly promising direction for autonomously mitigating documentation debt.

## Full Text


<!-- PDF content starts -->

DocSync: Agentic Documentation Maintenance via
Critic-Guided Reflexion
Sidhesh Badrinarayan
California, USA
0009-0002-5203-1485Adithya Parthasarathy
California, USA
0009-0001-6839-9527
Abstract—Software documentation frequently drifts from ex-
ecutable logic as codebases evolve, creating technical debt that
degrades maintainability and causes downstream API misuse.
While static analysis tools can detect the absence of documenta-
tion, they cannot evaluate its semantic consistency. Conversely,
standard Large Language Models (LLMs) offer generative flex-
ibility but frequently hallucinate when updating documentation
without deep structural awareness of the underlying code. To
address this gap, we proposeDocSync, an agentic workflow that
frames documentation maintenance as a structurally grounded,
iterative generation task. DocSync bridges syntactic changes
and natural language descriptions by fusing Abstract Syntax
Tree (AST) representations and Retrieval-Augmented Generation
(RAG) to provide dependency-aware context. Furthermore, to
ensure factual consistency, we incorporate a critic-guided refine-
ment loop based on the Reflexion paradigm, allowing the model
to self-correct candidate updates against the source code. We
empirically evaluate a resource-constrained implementation of
DocSync-using a LoRA-adapted small language model - on a
proxy code-to-text maintenance task. Our findings demonstrate
that this AST-aware agentic approach substantially outperforms
standard encoder-decoder baselines across semantic alignment,
summary-line faithfulness, and automated judge preferences
(e.g., achieving an automated judge score of 3.44/5.0 compared
to 1.91 for CodeT5-base). Crucially, the iterative critic loop
yields measurable improvements in semantic correctness without
requiring scaled-up parameter counts. These results provide
strong evidence that coupling structural retrieval with agentic
refinement is a highly promising direction for autonomously
mitigating documentation debt.
Index Terms—Software Documentation, Documentation Debt,
Agentic AI, Large Language Models, Retrieval-Augmented Gen-
eration, Abstract Syntax Tree, Reflection, Code Maintenance.
I. INTRODUCTION
A. The Economic and Operational Impact of Documentation
Debt
In modern software lifecycles, Code-Documentation Incon-
sistency (CDI) is not merely a technical nuisance; it is a
critical bottleneck. While source code is subjected to rigorous
validation through compilers and continuous integration (CI)
pipelines, accompanying documentation often lacks equivalent
enforcement, representing a significant gap in current DevOps
and MLOps practices. This leads to “bit rot,” where tacit ar-
chitectural knowledge becomes decoupled from the codebase.
The costs of this divergence are concrete and high. Outdated
documentation increases onboarding friction for new develop-
ers, who must “archeologize” the codebase to understand true
behavior. This high barrier to entry poses a direct challengeto the goals of AI in Education, as it hinders the ability of
students and newcomers to contribute to real-world projects.
More critically, it leads to production incidents when down-
stream consumers rely on incorrect API contracts. In legacy
open-source ecosystems, this friction contributes to maintainer
burnout and project stagnation.
B. The Failure of Static Maintenance
Historically, documentation maintenance has been reactive,
driven by user complaints, or limited to static analysis. Tools
like Doxygen or Javadoc can verify thepresenceof documen-
tation tags but are blind tosemantic correctness. This semantic
blindness manifests in several dangerous failure modes:
•Silent Constraint Shifts: If a function’s retry logic changes
from seconds to milliseconds, or a maximumtimeout
parameter drops from 300 to 60, the docstring may
incorrectly remain “Calculates backoff in seconds.” A
static linter reports no error because the@paramtag
is present, despite the documentation being functionally
lethal to downstream consumers.
•Unrecorded Side-Effects: A simplegetUser()function
might be modified during a refactor to also initialize a
user session or write to a cache. If the documentation
continues to describe it as a “pure, lightweight getter,”
developers may unknowingly call it in tight loops, caus-
ing severe performance regressions that static tools cannot
foresee.
•Tutorial Rot: Modern documentation requires under-
standing the intent behind multi-file dependencies. A
change in a low-level utility (e.g., migrating a database
driver from synchronous to asynchronous) often inval-
idates code snippets in high-level architectural guides
likeTUTORIAL.md. Static linters are structurally inca-
pable of traversing these semantic, cross-file dependency
chains.
These examples highlight a fundamental limitation: rule-based
systems can check syntax, but they cannot verify truth. True
maintenance requires “dependency awareness,” a capability
that demands the fusion of structural parsing (AST) with the
semantic reasoning of Large Language Models (LLMs).
C. Bridging the Gap: The DocSync Agentic Paradigm
The advent of Agentic AI offers a paradigm shift. However,
Large Language Models (LLMs) alone are insufficient; theyarXiv:2605.02163v1  [cs.SE]  4 May 2026

frequently hallucinate details when not grounded in the code’s
structural reality. Conversely, static tools lack the semantic
flexibility to generate human-readable explanations.
We introduceDocSync, a framework that bridges this gap
by combining the determinism of Abstract Syntax Trees (AST)
with the semantic reasoning of LLMs. DocSync operates as a
“digital gardener” via a multi-phase architecture:
1) Impact Analysis: Filtering noise to focus on semantic
changes.
2) Structural Retrieval: Using AST parsing to understand
the scope of changes (e.g., parameter flux, type migra-
tion).
3) Generative Synthesis: Using Large Language Models
(LLMs) adapted via Low-Rank Adaptation (LoRA) [1]
to rewrite documentation.
4) Verification: A “Reflexion” loop where a critic model
evaluates the consistency of the update [2].
D. Contributions
•A Practical Framework for Doc Repair: We introduce
DocSync, a blueprint for an agent that fixes stale docu-
mentation. It’s designed to be smart about code structure
by using ASTs, context-aware using RAG, and self-
correcting thanks to a critic-guided loop.
•Proof on a Budget: Our small-scale experiment demon-
strates that a lightweight, quantized Phi-3 Mini running
the DocSync workflow handily beats a standard CodeT5-
base baseline.
•A Fresh Look at Evaluation: We introduce a new metric
for checking summary-line accuracy and explore why
what looks good to a human (or a judge model) often
gets a poor score from metrics like BLEU.
•Open-Source Code: We’ve made our complete prototype
available on GitHub [14].
II. BACKGROUND ANDRELATEDWORK
The problem of Code-Documentation Inconsistency (CDI)
is well-documented, yet solutions have historically struggled
to bridge the semantic gap between code execution and natural
language description.
A. Taxonomy of Drift
To design an effective agent, one must categorize the types
of drift. We identify three primary categories:
•Signature-Level Inconsistencies: Explicit changes to the
API surface, such asParameter Flux(renaming, adding,
or removing arguments),Type Migration(e.g., broaden-
ing an input fromstrtoUnion[str, Path]), or
Return Value Divergence(e.g., returning an object instead
of a boolean). While static analysis or type checkers
can sometimes catch these in strongly typed languages,
they routinely fail in dynamically typed ecosystems like
Python, leaving users to discover the mismatch only at
runtime through crypticTypeErrorexceptions.•Semantic Inconsistencies: Subtler and arguably more
dangerous changes occur when the function signature re-
mains constant but its internal behavior shifts. Examples
includeSide-Effect Introduction(e.g., a simple memory
getter is modified to perform a blocking database query)
orConstraint Changes(e.g., a valid input range for a
parameter shrinks). Because the API contract appears
unbroken to static linters, these inconsistencies silently
propagate, leading to logical errors or performance re-
gressions in downstream applications.
•Tutorial Rot: High-level architectural guides and
README.mdfiles frequently contain embedded code
snippets that demonstrate how to orchestrate multiple
components. When underlying APIs are refactored or
renamed, these cross-file narrative examples are rarely
updated in tandem. This silent decay leads to broken
workflows for new users attempting to follow the ”quick
start” instructions, creating significant friction for project
adoption.
B. Iterative Reasoning: ReAct vs. Reflexion
As the field moves beyond single-pass text generation,
two primary agentic paradigms have emerged for complex
problem-solving: ReAct (Reasoning + Acting) [3] and Reflex-
ion [2]. ReAct prompts a model to emit a verbalized reasoning
trace before deciding on a tool action (e.g., “I notice the
timeout argument was deleted; I will now search the docstring
for references to it”). While ReAct is powerful for exploratory
tool orchestration, it typically executes actions in a forward-
only trajectory without inherently proofreading its final output.
Reflexion, on the other hand, introduces an explicit post-
generation critique loop. In this paradigm, an agent produces
a candidate solution, which is then evaluated by a discrete
“Critic” module against the original constraints. If the critic
detects an error or hallucination, it generates natural language
feedback, prompting the agent to revise its output. We se-
lected Reflexion as the foundational mechanism for DocSync
because documentation maintenance is inherently an editing
and verification task. Initial LLM generations often suffer from
minor formatting artifacts or subtle semantic omissions that are
difficult to prevent in a single forward pass, but are readily
caught by a focused semantic critic.
C. Agentic Workflows
While Code-LLMs like StarCoder [4] and Qwen2.5-Coder
have achieved parity with human developers on generation
tasks, applying them to maintenance requires operationalizing
the iterative approaches discussed above. DocSync implements
this by fusing AST-aware retrieval with the Reflexion critic-
guided generation loop, ensuring updates are not only gener-
ated but rigorously verified against the source code.
III. METHODOLOGY: THEDOCSYNCFRAMEWORK
A. Mathematical Formulation
We formalize the documentation update task as finding the
optimal documentation stringD∗given the current code state

Cnew, the previous documentationD old, and the set of changes
∆C.
LetM θbe a language model parameterized byθ. We seek
to maximize the probability:
D∗= argmax
DP(D|C new, Dold,K,∆C;θ)(1)
whereKrepresents the retrieved context. This context is a
union of structural and semantic information:
K=AST(C new)∪RAG(C new,Dcorpus )(2)
Here, AST(·)extracts function signatures and dependency
graphs via Tree-sitter [5], and RAG(·)retrieves semanti-
cally relevant documentation chunks from the vector store
Dcorpus [6].
B. Architecture
The DocSync architecture follows a multi-stage pipeline
visualized in Figure 1.
Code Change∆C
Impact Analysis
Relevant? Ignore
Retrieval (AST + RAG)
Generation (LoRA)
Critic Check
(IsGood?)Refine
Update DocNo
Yes
Fail
Pass
Fig. 1. DocSync Agentic Workflow.
C. Algorithm
The core logic is implemented as an iterative refinement
loop. Algorithm 1 details the procedure. The execution begins
by evaluating the code difference to determine if a documen-
tation update is actually necessary. If the changes are deemedirrelevant (e.g., whitespace formatting or purely internal logic
adjustments), the original documentation is preserved. For
relevant changes, the system extracts structural constraints
via an AST parser and semantic context via a RAG module.
These elements are combined with the new code and stale
documentation to form a composite prompt. The generation
model then produces an initial documentation draft, which
is immediately evaluated by an automated critic. If the critic
provides a ”GOOD” evaluation, the draft is accepted. Other-
wise, the critic’s natural language feedback is appended to
the prompt, and the model generates a refined draft. This
self-correction loop repeats until the draft is accepted or the
maximum number of retries is exhausted, at which point the
best available draft is returned.
Algorithm 1DocSync Update Loop
Require:C new(New Code),D old(Stale Doc)
Ensure:D new(Updated Doc)
1:∆C←Diff(C old, Cnew)
2:ifnot IsRelevant(∆C)then
3:returnD old
4:end if
5:K AST←ParseAST(C new)
6:K RAG←RetrieveContext(C new)
7:Prompt←Construct(C new, Dold, KAST, KRAG)
8:D draft← M θ(Prompt)
9:Attempts←0
10:whileAttempts < MaxRetriesdo
11:IsGood, Reason←Critic(D draft, Cnew)
12:ifIsGoodthen
13:returnD draft
14:end if
15:Prompt←Prompt+”Critic: ”+Reason
16:D draft← M θ(Prompt)
17:Attempts←Attempts+ 1
18:end while
19:returnD draft
D. Prompt Engineering and AST Injection
To bridge the gap between raw code and natural language,
we construct a composite prompt that explicitly separates
structural facts from semantic intent. The AST parser extracts
a linearized signature summaryS AST containing function
definitions and argument lists (e.g.,def connect(host,
port) | class DB). This summary is injected alongside
the code changeC newand the stale documentationD old:
Prompt=I sys⊕Cnew⊕D old⊕“AST: ”S AST (3)
whereI sysis the system instruction. This structural anchoring
reduces hallucinations by forcing the model to attend to the
verified API surface before generating the description.
E. Model and Training
We fine-tune Phi-3 Mini 4K Instruct [7] with LoRA (r=8,
α=16, dropout=0.05) in 4-bit quantization. Training uses batch

size 2, max source length 256, target length 96, and one
epoch on 8,192 training examples from the Python split
of CodeXGLUE code-to-text. The code dynamically detects
VRAM and pins minimal batch sizes to avoid OOM on T4/L4
GPUs.
F . Baselines
We compare against CodeT5-base [8], a standard encoder-
decoder model for code tasks. Baseline caching avoids re-
peated downloads; a sanitization step flattens special tokens
to prevent tokenizer instantiation errors.
IV. EXPERIMENTALSETUP
We conduct a resource-constrained, small-scale empirical
study to evaluate DocSync’s core mechanics. Our proto-
col uses a proxy task—updating a single function’s docu-
mentation—to test the architectural components (AST-aware
prompting, RAG, Reflexion loop) on consumer-grade hard-
ware.
A. Dataset and Task
We use the Python subset of the CodeXGLUE code-to-
text dataset [9], sampling 8,192 examples for training and
32 for evaluation. To simulate documentation drift, we create
a ”stale” docstring by truncating the ground-truth reference
to its first sentence. The model’s task is to repair this stale
documentation. The input prompt combines the code, the stale
docstring, and an AST-derived signature summary, tokenized
to a maximum of 256 tokens. Generated outputs are capped
at 96 tokens and normalized before evaluation to remove
artifacts.
B. Model and Training
We fine-tune a 4-bit quantized Phi-3 Mini 4K Instruct
model [7] using LoRA (r= 8, α= 16, p= 0.05) for one
epoch on the training set. We use the AdamW optimizer with
a learning rate of2×10−4and a batch size of 2. This setup
is designed to be executable on consumer-grade GPUs (16GB
VRAM).
C. Baselines and Metrics
We compare against CodeT5-base [8], a standard code-
to-text model. We evaluate performance using BLEU-4,
BERTScore (F1) [10], summary-line faithfulness (exact
match), and an LLM-as-a-Judge score [11] (1-5 scale). All
metrics are computed on normalized docstring payloads to
isolate content quality from generation artifacts.
V. RESULTS
A. Quantitative
Table I reports the main comparison. On this held-out
subset, DocSync (Final) improves substantially over CodeT5-
base on all reported metrics: BLEU (+0.382), BERTScore F1
(+0.105), summary-line exact match (+0.781), and judge score
(+1.53). These gains support an “early-stage promise” narra-
tive: the method is not yet production-ready, but it is clearly
stronger than the baseline on the current proxy task.TABLE I
MAIN EVALUATION RESULTS.
Model BLEU F1 Summary Exact Judge
DocSync (Final) 0.575 0.985 0.969 3.44
DocSync (Initial) 0.578 0.980 0.938 3.25
CodeT5-base 0.193 0.880 0.188 1.91
Oracle (Gemini-2.5-Pro) 0.138 0.868 0.031 4.13
B. Judge Metric and Error Analysis
We report an LLM-as-a-judge [11] signal where a teacher
model rates each docstring on a 1-5 scale (1=Irrelevant,
5=Perfect). Under the revised decoding and cleanup pipeline,
DocSync (Final) reaches 3.44 compared with CodeT5-base’s
1.91. The approximate 95% confidence intervals from the
current run are 2.87-4.00 for DocSync (Final), 1.41-2.41
for CodeT5-base, and 3.66-4.59 for the Oracle, which is
encouraging given the small evaluation set. A qualitative audit
of the saved outputs reveals three residual failure modes:
1) Boundary Artifacts: Some generations retain minor
quoting or punctuation debris at the start or end
of the docstring. For example, a model might in-
clude literal docstring markers in the payload (e.g.,
""" Parses the input string.) or leave trail-
ing, unfulfilled punctuation (e.g.,Returns the
updated configuration object:).
2) Over-Elaboration: The model occasionally adds plausi-
ble but unsupported parameter or return details, espe-
cially on longer structured docstrings. For instance, it
might hallucinate specific keyword arguments (e.g., de-
tailing atimeoutparameter for a generic **kwargs
pass-through) or invent specific error conditions (e.g.,
“Returns False if the connection times out”) that do not
exist in the source code.
3) Truncation and Repetition: A small number of examples
collapse into repeated summaries or drop the detailed
sections of longer reference docstrings. Examples in-
clude truncating a comprehensive parameter list after
the first argument, or falling into a generative loop
of the summary sentence (e.g., “Creates a wrapped
environment. Creates a wrapped environment.”).
C. Training Dynamics
Figure 2 shows the training loss over steps. The curve
flattens over the course of the first epoch, suggesting that a
single epoch is a reasonable low-cost operating point for this
prototype. We do not interpret this as evidence that longer
training would not help; rather, it indicates that meaningful
gains are achievable without large-scale compute.
VI. DISCUSSION
Our results indicate that DocSync is a promising direction
for documentation maintenance, substantially outperforming a
CodeT5-base baseline on a proxy task. We position this as
an early-stage empirical study, as the evaluation uses a small,

Fig. 2. Training loss curve.
benchmark-derived dataset. The critic-guided Reflexion loop
provides modest but consistent semantic refinement; as shown
in Table II, the final pass improves the judge score (+0.19) and
summary-line exact match (+0.031) with a negligible change
in BLEU, suggesting it acts as a semantic cleanup mechanism.
Finally, we observe a notable misalignment between overlap-
based metrics and judge preference. The oracle model achieves
the highest judge score (4.13) but the lowest BLEU score
(0.138), reinforcing that semantic quality and reference-based
overlap are distinct properties and justifying our multi-faceted
evaluation approach.
VII. ABLATIONS ANDSENSITIVITY
We report the current sensitivity analysis for the Reflexion
loop. We do not report refreshed AST/RAG ablations here
because those variants were not rerun under the revised prompt
and normalization pipeline used in the final results. Isolating
those contributions remains important future work.
TABLE II
REFLEXION LOOP SENSITIVITY ANALYSIS.
Setting BLEU BERTScore F1 Summary Exact Judge
DocSync (Initial) 0.578 0.980 0.938 3.25
DocSync (Final) 0.575 0.985 0.969 3.44
VIII. LIMITATIONS
While the early-stage results are promising, this study has
several limitations:
•Training Scope: Our runs are limited to a single epoch
and focus exclusively on Python docstrings. This scope
was chosen to isolate the impact of the architectural con-
tributions under tight compute constraints, but it leaves
cross-language generalization unverified.
•Evaluation Task: The evaluation task is a proxy derived
from the CodeXGLUE code-to-text dataset rather than a
true historical repository replay benchmark. The “stale
documentation” is artificially simulated, which may not
fully capture the complex reality of organic codebase
drift.
•Metric Normalization: We normalize generated outputs to
extract the core docstring payload before scoring. Whilethis correctly focuses the evaluation on semantic content
rather than formatting artifacts, it means we are not
evaluating the raw, end-to-end string generation.
•Workflow Integration: We do not yet execute embedded
code snippets (doctests) or test the mergeability of pro-
posed updates in real maintainer workflows. The pipeline
currently stops at generation and semantic verification.
•Artifact Constraints: From a deployment perspective,
the current pipeline leaves the trained LoRA adapters
unmerged from the base model, meaning the resulting
checkpoints are not fully self-contained.
IX. QUALITATIVEANALYSIS
To better understand practical model behavior beyond ag-
gregate metrics, Table III presents a comparative qualitative
analysis. The “Success” example highlights DocSync’s ability
to cleanly extract semantic meaning and halt generation ap-
propriately, whereas the baseline model leaks raw code into
its output. Conversely, the “Failure” example demonstrates
a key limitation we term “Long-Form Compression,” where
DocSync correctly captures the high-level summary of a com-
plex function but aggressively truncates the structured, detailed
parameter descriptions, favoring brevity over completeness.
TABLE III
QUALITATIVE ANALYSIS EXAMPLES.
Success: Clean Semantic Extraction
Reference:“Create placeholder to feed observations into of the size
appropriate to the observation space, and add input encoder of the
appropriate type.”
Baseline Output:Repeats the summary sentence but fails to halt,
appending raw code artifacts (e.g., “code snippet:def
observation_input(...)”).
DocSync Output:“Creates an observation placeholder and its
corresponding encoded representation.” (Captures the intent cleanly and
halts appropriately).
Failure: Long-Form Compression
Reference:A detailed, multi-line docstring explaining thesmooth
function, including specific formulas fortwo_sidedandcausal
modes, and the behavior of thevalid_onlyflag.
DocSync Output:“Smooth signal y, where radius is determines the size
of the window.” (Retains only the first sentence, completely dropping
the crucial parameter operation details).
X. ETHICAL ANDSOCIETALIMPACT
The automation of software maintenance presents a dual
impact. On one hand, tools like DocSync can democratize
open-source contributions by lowering the cognitive barrier
for new developers and reducing maintainer burnout. On the
other hand, it introduces risks of automation bias, where
developers might uncritically accept AI-generated text. A
hallucinated constraint could lead to security vulnerabilities
or production outages. To mitigate these risks, DocSync is
designed for a Human-in-the-Loop (HITL) workflow, where its
outputs are treated as drafts for human review. Furthermore, to
address privacy and copyright concerns, our model was trained
exclusively on the permissively licensed, public CodeXGLUE
dataset, avoiding proprietary or personal data.

XI. FUTUREWORK
This research lays the groundwork for more reliable docu-
mentation agents. Several key directions for future investiga-
tion could bridge the gap between this prototype and broader
maintainer-facing deployment.
A. Historical Replay at Scale
One promising direction is evaluating agents via Historical
Replay, using git history to simulate real-world code evolution.
This would move beyond static benchmarks by testing against
actual developer updates, but would require robust filtering to
isolate clean documentation commits [13].
B. Execution-Based Verification
To move beyond proxy metrics, execution-based verification
could be implemented. This would involve extracting and
running code snippets (doctests) from generated documenta-
tion. Execution failures could then provide a strong negative
signal for a Reinforcement Learning (RL) loop, directly testing
functional correctness.
C. Multi-Modal Documentation
Another area for future work is multi-modal documentation.
Integrating Vision-Language Models (VLMs) could enable
the agent to understand and update diagrams (e.g., UML,
architecture flows), helping to ensure they remain consistent
with code changes.
D. Human-in-the-Loop and RLHF
To build trust and improve the agent, Reinforcement Learn-
ing from Human Feedback (RLHF) could be integrated. Main-
tainer interactions (accepting, rejecting, or editing suggestions)
could be used to fine-tune a reward model, which would help
calibrate the agent’s confidence and align it with developer
preferences.
XII. CONCLUSION
The relentless decay of software documentation has long
been accepted as an unavoidable friction in the software de-
velopment lifecycle. With DocSync, we demonstrate that this
no longer has to be the case. By fusing AST-aware structural
retrieval with the semantic reasoning of a LoRA-adapted Phi-
3 Mini and a robust Reflexion loop, DocSync unequivocally
outperforms a standard CodeT5 baseline across all evaluated
metrics—even when constrained to consumer-grade hardware.
These highly promising results validate the agentic paradigm
for documentation maintenance. The practical implications of
this architectural direction are profound. As a prime example
for AI in Education, DocSync dramatically lowers the barrier
for contributors to new codebases. It can also help with
DevOps and MLOps as its integration into CI/CD pipelines
can automate a critical phase of software delivery, eradicat-
ing ”documentation debt” and alleviating maintainer burnout.
DocSync establishes a powerful and proven foundation for this
new paradigm. We stand at the threshold of a new era in soft-
ware engineering, where documentation evolves autonomouslyalongside executable logic. As a pioneering force in this
transition, DocSync stands as a robust, intelligent guardian of
codebase truth, ensuring that the critical knowledge embedded
within our global open-source infrastructure remains accurate,
accessible, and eternally alive.
REFERENCES
[1] E. J. Hu, Y . Shen, P. Wallis, Z. Allen-Zhu, Y . Li, S. Wang, L. Wang, and
W. Chen, ”LoRA: Low-Rank Adaptation of Large Language Models,”
inInt. Conf. Learning Representations, 2022.
[2] N. Shinn, B. Labash, and A. Gopinath, ”Reflexion: Language Agents
with Verbal Reinforcement Learning,” inAdvances in Neural Informa-
tion Processing Systems, 2023.
[3] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y . Cao,
”ReAct: Synergizing Reasoning and Acting in Language Models,” in
Int. Conf. Learning Representations, 2023.
[4] R. Li et al., ”StarCoder: may the source be with you!,” arXiv preprint
arXiv:2305.06161, 2023.
[5] M. Brunsfeld, ”Tree-sitter,” 2017-2024: https://tree-sitter.github.io/tree-
sitter/.
[6] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal, H.
K¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel, S. Riedel, and D. Kiela,
”Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,”
inAdvances in Neural Information Processing Systems, vol. 33, 2020,
pp. 9459–9474.
[7] M. Abdin et al., ”Phi-3 Technical Report: A Highly Capable Language
Model Locally on Your Phone,” arXiv preprint arXiv:2404.14219, 2024.
[8] Y . Wang, W. Wang, S. Joty, and S. C. H. Hoi, ”CodeT5: Identifier-aware
Unified Pre-trained Encoder-Decoder Models for Code Understanding
and Generation,” inProc. Conf. Empirical Methods Natural Language
Processing, 2021, pp. 8696–8708.
[9] S. Lu et al., ”CodeXGLUE: A Machine Learning Benchmark Dataset for
Code Understanding and Generation,” inProc. Conf. Empirical Methods
Natural Language Processing, 2021.
[10] T. Zhang, V . Kishore, F. Wu, K. Q. Weinberger, and Y . Artzi,
”BERTScore: Evaluating Text Generation with BERT,” inInt. Conf.
Learning Representations, 2020.
[11] L. Zheng, W.-L. Chiang, Y . Sheng, S. Zhuang, Z. Wu, Y . Zhuang, Z.
Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica,
”Judging LLM-as-a-judge with MT-Bench and Chatbot Arena,” arXiv
preprint arXiv:2306.05685, 2023.
[12] C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K.
Narasimhan, ”SWE-bench: Can Language Models Resolve Real-World
GitHub Issues?,” inInt. Conf. Learning Representations, 2024.
[13] C. E. Jimenez et al., ”SWE-Replay: Efficient Test-Time Scaling for
Software Engineering Agents,” arXiv preprint arXiv:2401.12129, 2024.
[14] S. Badrinarayan, ”DocSync: Agentic Documentation Maintenance
via Critic-Guided Reflexion,” GitHub repository: https://github.com/
TheSidhesh/DocSync.
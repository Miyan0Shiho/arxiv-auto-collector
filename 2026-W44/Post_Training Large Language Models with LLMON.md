# Post-Training Large Language Models with LLMON

**Authors**: Basel Shbita, Michael Hind, Chad DeLuca, Nathan Fulton, Shubhi Asthana, Farhan Ahmed, Anisa Halimi, Yada Zhu, Mauro Martino, David Cox, Dan Gutfreund

**Published**: 2026-10-28 00:56:01

**PDF URL**: [https://arxiv.org/pdf/2610.38120v1](https://arxiv.org/pdf/2610.38120v1)

## Abstract
Large Language Models (LLMs) treat all context tokens uniformly, lacking explicit separation between executable instructions and passive data. This leads to prompt injection vulnerabilities, ambiguous instruction binding, and unreliable execution control. We present a post-training methodology using LLMON (LLM Object Notation), a framework that enables separating instructions from data and requires explicit identifier-based execution. We construct a large-scale structured training corpus of 4.6B tokens spanning general, RAG (Retrieval-Augmented Generation), and tool-calling data, augmented with distractor-infused variants in which competing instructions and irrelevant data segments are present in context but not referenced. Post-training two Granite-4.1 base models (3B and 8B) through supervised fine-tuning followed by reinforcement learning, we show that LLMON-structured supervision reduces attack success rates on indirect prompt-injection benchmarks from 53-100% to 0.02-5.6% while preserving general capabilities. We further introduce attention-based interpretability metrics that diagnose failure modes at individual transformer layers for structured inputs, revealing how our approach reshapes internal attention patterns. Our results demonstrate that explicit execution semantics can be taught through standard post-training without architectural modification.

## Full Text


<!-- PDF content starts -->

Post-Training Large Language Models with LLMON
Basel Shbita1, Michael Hind2, Chad DeLuca1, Nathan Fulton3, Shubhi Asthana1,
Farhan Ahmed1, Anisa Halimi4, Yada Zhu2, Mauro Martino3, David Cox3, Dan Gutfreund3
1IBM Research, San Jose, CA, USA
2IBM Research, Yorktown Heights, NY , USA
3IBM Research, Cambridge, MA, USA
4IBM Research, Dublin, Ireland
basel@ibm.com,{hindm, delucac}@us.ibm.com,nathan@ibm.com,sasthan@us.ibm.com,
{farhan.ahmed, anisa.halimi}@ibm.com,{yzhu, mmartino}@us.ibm.com,david.d.cox@ibm.com,dgutfre@us.ibm.com
Abstract
Large Language Models (LLMs) treat all con-
text tokens uniformly, lacking explicit separa-
tion between executable instructions and pas-
sive data. This leads to prompt injection vulner-
abilities, ambiguous instruction binding, and
unreliable execution control. We present a post-
training methodology using LLMON (LLM
Object Notation), a framework that enables
separating instructions from data and requires
explicit identifier-based execution. We con-
struct a large-scale structured training corpus of
4.6B tokens spanning general, RAG (Retrieval-
Augmented Generation), and tool-calling data,
augmented with distractor-infused variants in
which competing instructions and irrelevant
data segments are present in context but not
referenced. Post-training two Granite-4.1 base
models (3B and 8B) through supervised fine-
tuning followed by reinforcement learning, we
show that LLMON-structured supervision re-
duces attack success rates on indirect prompt-
injection benchmarks from 53-100% to 0.02-
5.6% while preserving general capabilities. We
further introduce attention-based interpretabil-
ity metrics that diagnose failure modes at in-
dividual transformer layers for structured in-
puts, revealing how our approach reshapes in-
ternal attention patterns. Our results demon-
strate that explicit execution semantics can be
taught through standard post-training without
architectural modification.
1 Introduction
Large Language Models (LLMs) process all to-
kens in a context window through the same at-
tention mechanism, without any built-in distinc-
tion between executable instructions, passive data,
and auxiliary context (Vaswani et al., 2017; Kad-
dour et al., 2023). This architectural uniformity
means that any token can equally influence model
behavior, whether it originates from a user instruc-
tion, a retrieved document, or an injected adver-
sarial payload. The practical consequences arewell-documented: prompt injection attacks that hi-
jack model execution (Greshake et al., 2023; Perez
and Ribeiro, 2022), ambiguous referential binding
when multiple instructions coexist in context (e.g.,
“execute the second task”), and degraded reliabil-
ity in compositional scenarios involving retrieval-
augmented generation (RAG) and tool calling (Yao
et al., 2022; Schick et al., 2023). These failure
modes stem from a single root cause: the model
has no mechanism to distinguish executable instruc-
tions from passive data at the token level. Current
instruction-tuning approaches (Ouyang et al., 2022;
Wei et al., 2022) train models to follow instructions
implicitly, based on position, formatting conven-
tions (e.g., “chat templates”), and statistical pat-
terns, thus lack the mechanism to distinguish which
instruction to execute when multiple candidates ex-
ist in a given prompt.
In this paper, we present a structured post-
training methodology built on LLMON (Hind et al.,
2026; LLMON Team, 2026) (LLM Object Nota-
tion), a hierarchical tagging framework that in-
troduces explicit instruction-data separation and
identifier-based execution binding into the LLM
interface. Under LLMON, each instruction and
data segment is wrapped in typed, uniquely iden-
tified blocks, and execution occurs only through
explicit execution calls that reference specific in-
struction and data identifiers. This design steers
the model toward a clear semantic principle where
only explicitly invoked instructions are executed.
Our contributions are:
1.We describe a scalable data pipeline that
transforms instruction-tuning corpora (gen-
eral, RAG, and tool-calling) into LLMON-
structured training data, including distractor-
augmented variants with up to three compet-
ing instructions and irrelevant data segments.
2.We conduct a post-training evaluation on
two base models (Granite-4.1-3B-Baseand
arXiv:2610.38120v1  [cs.AI]  28 Oct 2026

Granite-4.1-8B-Base) under full fine-tuning
with 4.6B tokens, followed by reinforcement
learning.
3.We introduce attention-based interpretability
metrics: Instruction Focus Ratio (IFR), Data
Attention Score (DAS), and Leakage (LEAK),
to provide mechanistic insight into how struc-
tured training reshapes model attention.
2 Related Work
Instruction tuning and alignment.Instruction
tuning (Wei et al., 2022; Ouyang et al., 2022) and
preference-based alignment (Rafailov et al., 2023;
Christiano et al., 2017) have become the standard
post-training pipeline for modern LLMs. These
methods train models to follow natural-language
instructions but rely on implicit positional cues
and chat templates (Hugging Face, 2025)-based
formatting conventions, rather than explicit struc-
tural semantics. When multiple instructions appear
in context, models lack a principled mechanism
for disambiguation. Our work is complementary:
LLMON-structured supervision can be applied on
top of any instruction-tuning corpus to add explicit
execution binding behavior.
Prompt injection and adversarial robustness.
Prompt injection exploits the absence of privilege
separation between instructions and data (Greshake
et al., 2023; Perez and Ribeiro, 2022). Spotlight-
ing (Hines et al., 2024) marks untrusted spans with
in-band text delimiters at the prompt level, while
StruQ (Chen et al., 2025a) and SecAlign (Chen
et al., 2025b) make the boundary atrainingobjec-
tive, fine-tuning on structured prompts and prefer-
ence pairs in which injected instructions are present
but must not be followed. We differ in two respects.
First, the boundary is carried by reserved single to-
kens from a disjoint region of the vocabulary rather
than by in-band delimiters or role names, so it can-
not be formed by subword composition of ordinary
content. Second, privilege attaches to an explicit
binding rather than to a role: every span is indi-
vidually addressable and authority follows from
which pointer references it, so several same-role
instructions can coexist with only one bound for
execution.
Tool use and function calling.Modern LLM
systems integrate external tools through structured
generation (Schick et al., 2023; Patil et al., 2024;
Yao et al., 2022). These approaches define toolschemas and train models to emit structured calls,
but the instruction-data boundary within the prompt
remains implicit. LLMON’s execution semantics
are orthogonal and compatible with tool-calling
pipelines: tool definitions, retrieved documents,
and user instructions each occupy typed segments
with distinct identifiers.
Structured generation and constrained decod-
ing.Constrained decoding approaches (Willard
and Louf, 2023; Zheng et al., 2024; Beurer-Kellner
et al., 2023) enforce output structure at infer-
ence time through grammar-based token masking.
Prompting frameworks (Khattab et al., 2023; Liu
et al., 2026; POML) and programming-language
abstractions (Dantanarayana et al., 2025) orches-
trate LLM calls, and markup-based prompt lan-
guages impose document structure on the prompt,
but none of these modifies the model’s internal
representation of instruction-data boundaries. Un-
like these inference-time or application-layer ap-
proaches, LLMON operates at thetraining data
level, teaching models to internalize structural dis-
tinctions through supervised learning.
3 Using LLMON (LLM Object Notation)
This section describes how we leverage LL-
MON (Hind et al., 2026) to construct structured
post-training data that teaches models explicit exe-
cution control. LLMON is a hierarchical tagging
framework that introduces instruction-data separa-
tion and identifier-based execution binding at the to-
ken level, operating within standard autoregressive
language modeling without architectural modifica-
tion. We briefly summarize the notation relevant
to our training methodology, then describe the data
construction pipeline.
3.1 Notation Overview
LLMON defines six special delimiter tokens
(“<|open|> ”, “ <|close|> ”, “ <|open_end|> ”,
“<|self_close|> ”, “<|:|> ”, “<|.|> ”) that form a
typed tagging system. The key structural elements
relevant to training are:
Typed segments.All content is wrapped in iden-
tified and tagged blocks.Instructionscarry role
qualifiers (“ instr<|:|>ID ”),datasegments hold
passive content (“ data<|:|>ID ”),requirements
hold output constraints (“ req<|:|>ID ”), and
documentsuse nested paths (“ doc<|.|>title ”,
“doc<|.|>text ”).Tool definitionsare flattened

<|open|> exec <|:|> EXEC_ID <|close|>
<|open|> exec <|.|> instruction <|close|>
instr <|:|> INSTR_ID
<|open_end|>
<|open|> exec <|.|> input <|close|>
data <|:|> DATA_ID
<|open_end|>
<|open|> exec <|.|> output <|close|>
assistant <|:|> OUT_ID
<|open_end|>
<|open_end|>
Figure 1: Example of an execution block in LLMON
syntax. All special tokens are highlighted in light-pink.
so that every leaf of the JSON schema becomes
its own addressable sub-field (“ tool<|.|>name ”,
“tool<|.|>parameters<|.|> arg<|.|>type ”),
and a tool result returns as a “ data ” block like any
other input. Asystemblock (“ system<|:|>ID ”)
carries authoritative policy and persists across
turns.
Execution binding.An execution block explic-
itly references which instruction to execute, which
data to use as input, and where the output should
be placed via identifier references as shown in
Figure 1. This design enforces a clear semantic
paradigm where multiple instructions can coexist
in context without interference, as execution re-
quires referencing a specific identifier regardless of
position.
The privilege invariant.Execution binding
yields the invariant our training targets: con-
tent reached through “ exec<|.|>instruction ”
or “exec<|.|>system ” isauthoritative, whereas
everything reached through “ exec<|.|>input ”
(data, documents, tool schemas, and tool results) is
material to be read, not a command to be followed,
however it is phrased. Blocks that no pointer ref-
erences are rendered into the context but bound
to nothing. Privilege is therefore a property of
the binding rather than of a block’s position or role
name, and the delimiters that express it are reserved
tokens occupying a disjoint region of the vocabu-
lary, so the structure cannot be formed by subword
composition of ordinary content.
3.2 Data Construction Pipeline
We develop an automated pipeline that trans-
forms diverse instruction-tuning data into LLMON-
structured training examples. The pipeline handlesthree distinct data modalities:
General instruction data.Single-turn and multi-
turn conversations are serialized with per-turn in-
struction blocks, “ exec ” bindings, and assistant
data blocks. Each turn receives unique identifiers,
and multi-turn examples chain consecutive “ exec ”
cycles.
RAG and document-grounded data.Re-
trieved documents are serialized as typed “ doc”
segments with nested “ doc<|.|>title ” and
“doc<|.|>text ” sub-fields. The “ exec ” block’s
input field references both the user’s data and the
relevant document identifiers, teaching the model
to ground its response in explicitly referenced
context.
Tool-calling data.Tool definitions are
serialized using flattened path-based nota-
tion (e.g., “ tool<|.|>function<|.|>name ”,
“tool<|.|>function<|.|>description ”). Tool
invocations produce a first “ exec ” cycle (user
instruction →assistant tool call), followed by a
tool-result data block and a second “ exec ” cycle
(tool result→final assistant response).
3.3 Distractor Augmentation
The central innovation in our training data isdis-
tractor augmentation: systematically injecting
competing instructions and irrelevant data into the
context while keeping the “ exec ” block pointed at
only the target. This directly trains the model to
resolve referential bindings under interference.
For each base example we generate variants
withd∈ {1,2,3} distractor instructions (sampled
from the instruction pool, excluding the target) and
m∈ {0,1,2,3} irrelevant data segments (sampled
from other examples). The distractor instructions
and data occupy their own identified blocks in the
context, but the “ exec ” block referencesonlythe
original target instruction and data, and the target
output is identical across all variants of an example.
Variants are sampled per base example, yielding
8 categories parameterized by whether target data
is present and by the distractor and irrelevant-data
counts. A model that keys on the “ exec ” block
is unaffected by the decoys; one that keys on sur-
face cues such as position or imperative phrasing
is penalized by the loss.
We illustrate one such example with a multi-
instruction and multi-data distractors execution in
Appendix A.1.

4 Experiments
4.1 Training Data
Following the pipeline of Section 3.2, we assemble
the LLMON-structured corpus from the sources
below, totaling 4.4M examples and 4.6B tokens.
•General SFT data: 3.25M single-turn (2.79B
tokens) and 59k multi-turn (120M tokens)
examples drawn from the Granite-4.1 post-
training SFT corpora.
•RAG data: 27k single-turn (45.5M tokens)
and 8k multi-turn (17.8M tokens) retrieval-
augmented examples from the same corpora.
•Tool-calling data: 451k single-turn (707M
tokens) and 251k multi-turn (562M tokens)
function-calling examples from the same cor-
pora.
•Alpaca: 52k clean examples (Taori et al.,
2023) (17.2M tokens) plus 103k distractor-
augmented variants (39.4M tokens).
•Dolly: 15k clean examples (Conover et al.,
2023) (3.3M tokens) plus 30k distractor-
augmented variants (16.3M tokens).
•Prompt-injection data: 130k examples
(211M tokens) in which an adversarial instruc-
tion is planted inside a payload the model is
meant to read rather than execute, spanning
general (40k), RAG (11k), and tool-calling
(80k) modalities.
The general, RAG, and tool-calling exam-
ples teach the structural format across single-
turn instruction following, multi-turn conversation,
document-grounded QA, and tool invocation, and
account for 93.6% of all tokens, which is what
keeps capability close to the conventionally-tuned
baseline. The Alpaca and Dolly expansions supply
the distractor-augmented variants of Section 3.3,
expanding 67k base examples into 133k variants.
The prompt-injection examples are the training-
time counterpart of the evaluation in Section 4.3.2:
the injected instruction is rendered as a block that
no execution pointer references, and the target out-
put is the one that ignores it.
For the reinforcement-learning stages we use
two further sources: approximately 169k prompts
across five sequential policy-optimization stages
spanning multi-domain reasoning, RLHF, knowl-
edge calibration, and two mathematics boosters;and approximately 1k on-policy preference pairs
for the final preference-optimization stage, con-
structed from each model’s own responses at agen-
tic decision points where a tool result carries an
injected instruction.
4.2 Models and Training
We post-train two base models from the same fam-
ily and generation at different scales:Granite-4.1-
3B-BaseandGranite-4.1-8B-Base(IBM Granite
Team, 2026; Granite Team, 2024). Both are base
(pre-trained) checkpoints without prior instruction
tuning, so the LLMON behavior is learned dur-
ing post-training rather than layered on top of an
existing chat format.
Supervised fine-tuning.We first install the six
LLMON delimiters in the tokenizer as reserved sin-
gle tokens by remapping unused vocabulary slots,
which avoids growing the vocabulary or resizing
the embedding matrix. We then run full fine-tuning,
updating all parameters on the LLMON-serialized
corpus: three epochs over 4.4M examples and 4.6B
tokens at a sequence length of 4096 tokens, sharded
across 8 ×H100 GPUs with FSDP. We report the
final-epoch checkpoint.
Reinforcement learning.Starting from the su-
pervised checkpoint, we run five sequential Group
Relative Policy Optimization (GRPO) stages
against a reward model over the ≈169k prompts
described above, which optimize end behavior di-
rectly rather than the token-level target. A final Di-
rect Preference Optimization (Rafailov et al., 2023)
(DPO) stage then trains on the ≈1k on-policy injec-
tion preference pairs ( β= 0.1 , one epoch, learning
rate 5×10−7, effective batch size 32). This stage
is placed last so that the general-capability stages,
which carry no injection objective, cannot erode the
boundary behavior installed earlier. Unless noted
otherwise, results reported as “LLMON” refer to
this final SFT→GRPO→DPO checkpoint.
4.3 Evaluation
4.3.1 Standard Benchmarks
To verify that LLMON post-training preserves gen-
eral capabilities, we evaluate across seven cate-
gories, reporting the unweighted mean over the
benchmarks in each:General(MMLU (Hendrycks
et al., 2021), MMLU-Pro (Wang et al., 2024),
BBH (Suzgun et al., 2023), AGI-Eval (Zhong et al.,
2024), GPQA (Rein et al., 2023));Alignment(IFE-
val (Zhou et al., 2023));Math(GSM8K (Cobbe

et al., 2021), GSM-Symbolic (Mirzadeh et al.,
2025), Minerva Math (Lewkowycz et al., 2022),
DeepMind Math (Saxton et al., 2019));Code(Hu-
manEval (Chen et al., 2021), HumanEval+ (Liu
et al., 2023), MBPP (Austin et al., 2021),
MBPP+ (Liu et al., 2024a), MultiPL-E (Cassano
et al., 2023), CRUXEval-O (Gu et al., 2024));
Tool-Calling(BFCL v3 (Patil et al., 2025));Mul-
tilingual(Global-MMLU (Singh et al., 2025),
MGSM (Shi et al., 2022), INCLUDE (Romanou
et al., 2025)); andSafety(6-task safety aggregate
metric from Tulu3 (Zhao et al., 2026), AttaQ (Kour
et al., 2023), SALAD-Bench (Li et al., 2024)).
Each LLMON model is compared against the in-
struct counterpart at the same scale, trained on the
matched chat-template corpus of Section 4.1.
4.3.2 Indirect Prompt-Injection Benchmarks
Indirect prompt injection is the failure mode the
privilege invariant is designed to prevent: a mali-
cious instruction is planted in a data payload, such
as a retrieved document or a tool response, rather
than supplied by the user. We evaluate on two
published benchmarks that place the injection on
different surfaces.OPI(Liu et al., 2024b) injects
into the data of classic NLP tasks and defines the
five-attack taxonomy used throughout this evalua-
tion (naive, escape, ignore, fake-completion, and
combine), together with the attack-success metric.
ASB(Zhang et al., 2025) moves the attack into an
agent’s tool observations, which is the surface that
matters for tool-using deployments; we report its
observation-injection arm.
We report attack success rate (ASR), the frac-
tion of adversarial prompts in which the injected
instruction is followed; lower is better. For the LL-
MON models the injected payload arrives inside a
“data ” block reached through “ exec<|.|>input ”,
so following it would require acting on a span the
model was trained to read rather than execute; the
instruct baselines receive the same payloads under
conventional chat formatting.
4.3.3 Adversarial Red-Teaming
The two benchmarks above measure known, pub-
lished attacks. Separately from them, and go-
ing beyond what a fixed benchmark can surface,
our security team red-teamed both models with
ARES (Nedoshivina et al., 2025), a robustness-
evaluation harness that expands 808 seed prompts
from five datasets (Purple (Bhatt et al., 2023), Gan-
dalf (Pfister et al., 2025), and TensorTrust (Toyeret al., 2024) for direct attacks; Alpaca and a
safety-policy set for indirect ones) into approxi-
mately 15k attack prompts by mutating each seed
through eight attack strategies organized by at-
tacker sophistication, from basic (asking directly)
through intermediate (append, ignore, escape-
separation, completion-real, append-and-prepend)
to advanced. The advanced arm is a multi-turn
variant of Crescendo (Russinovich et al., 2025), in
which an attacker model generates each next turn
and a judge decides whether the adversarial objec-
tive was met; we use DeepSeek-V3.2 (DeepSeek-
AI et al., 2025) as the attacker,GPT-OSS(OpenAI
et al., 2025) (120B) as the judge, and a budget of 20
turns. Results are reported asrobustness( 1−ASR ),
so that higher is better, which is the convention of
the harness; note that this is the opposite direction
to Table 2.
4.3.4 Attention-Based Interpretability Metrics
Beyond output-level evaluation, we introduce three
per-layer attention metrics designed to provide
mechanistic insight into model behavior. These
metrics quantify how attention from the final gener-
ated token is distributed across target and distractor
spans. Let A(S) denote the total attention mass
assigned to a span set S. We define ItandIdas
the target and distractor instruction spans, respec-
tively, and DtandDdas the corresponding target
and distractor data spans.
TheInstruction Focus Ratio(IFR) measures the
proportion of attention allocated to the target in-
struction relative to all instruction spans:
IFR=A(I t)
A(I t) +A(I d)(1)
Similarly, theData Attention Score(DAS) quan-
tifies the proportion of attention assigned to the
target data relative to all data spans:
DAS=A(D t)
A(D t) +A(D d)(2)
Finally,Leakage(LEAK) is the share of all span
attention that reaches distractor content, across
both instruction and data spans. Writing Sfor
every labeled span in the prompt (the target and
distractor instructions and data, together with the
structural spans: the execution block, the delim-
iters, and the remaining prompt), LEAK is
LEAK=A(I d) +A(D d)
A(S)(3)

These metrics are computed per layer across
all attention heads (averaged over attention heads
within each KV group for grouped-query-attention
models). This enables layer-wise failure diagnosis:
higher IFR and DAS values indicate stronger fo-
cus on task-relevant content, whereas larger LEAK
values suggest increased attention allocated to ir-
relevant or competing information.
5 Results and Discussion
5.1 General Capabilities
Table 1 compares LLMON-structured post-training
against theInstructchat-template baselines at both
scales under matched full fine-tuning. The objec-
tive here is parity rather than improvement: LL-
MON training is meant to install an execution-
binding behavior, not to make the model better
at mathematics or code. Averaged over the six
capability categories, the 3B model is unchanged
(66.7 both ways) and the 8B model loses 1.4 points
(74.2 to 72.8), with mathematics at 8B the largest
single regression (84.1 to 79.6). LLMON improves
on Code and Multilingual at 3B. Safety improves
at both scales, by 8.8 points at 3B and 6.2 at 8B,
which is where the structural supervision is ex-
pected to show.
5.2 Indirect Prompt Injection
Table 2 reports attack success rate on the two in-
jection benchmarks of Section 4.3.2. On text-level
injection (OPI) the attack essentially stops working:
ASR falls from 100.0 to 0.1 at 3B and from 96.3
to 0.02 at 8B. On agentic injection (ASB), where
the payload arrives in a tool observation rather than
in task data, ASR falls from 99.2 to 5.6 at 3B and
from 52.8 to 3.1 at 8B. These are reductions of
roughly 17× on agentic injection and three orders
of magnitude on text-level injection, and they hold
at both scales despite the 8BInstructbaseline al-
ready being markedly more robust on ASB than the
3B one. The mechanism is structural rather than a
learned refusal: the injected instruction occupies a
“data ” block that no execution pointer references,
so acting on it would require the model to execute
a span it was trained to read. Section 5.4 examines
what this looks like inside the network.
5.3 Adversarial Red-Teaming
Independently of the published benchmarks of Sec-
tion 5.2, Table 3 reports the ARES red-teaming
campaign of Section 4.3.3, which uses its own seeddatasets and attack strategies and separates results
by where the adversarial instruction enters. Scores
are robustness (1−ASR), so higher is better.
Onindirectattacks the gain is large and is what
the design predicts: robustness rises from 2.8% to
84.6% at 3B and from 0.7% to 79.8% at 8B, be-
cause the payload lands in a span that no execution
block designates as an operable instruction.Direct
andmulti-turnattacks also improve, roughly dou-
bling at both scales, but remain far weaker in abso-
lute terms (32-37%). That asymmetry is expected
rather than disappointing. When the adversarial
instructionisthe user’s own instruction there is no
instruction-data boundary left to enforce, and an
attacker who steers a conversation across twenty
turns is not smuggling an instruction past the bind-
ing but building towards a request the model ac-
cepts on its own terms. The privilege invariant is
therefore a defense against untrustedinput, and
should be composed with defenses of a different
kind rather than relied upon alone.
5.4 Attention Interpretability Analysis
Both the injection reductions and the capability pic-
ture are consistent across the two scales tested, so
the remainder of this section drills intohowthat
robustness arises. Using the per-layer attention
metrics defined in Section 4.3.4, we analyze the
Granite-4.1-3Bpair (the LLMON-trained model
and itsInstructcounterpart) on distractor prompts
in which several instructions, and in most cases sev-
eral data segments, coexist while the “ exec ” block
binds only one pair. We first report the per-layer
behavior on a representative example to expose the
mechanism, then aggregate over all examples to
verify that the trends are not driven by ordering.
Where the binding lives.Both models are mea-
sured at the same layers, and the separation is
largest in the late-middle of the stack rather than at
the output. Table 4 reports the mean of each metric
over the distractor examples at five representative
layers. The LLMON-trained model holds a mean
IFR of 86-87% at layers 25 and 28 against 26-38%
for theInstructmodel, with mean leakage of 6-9%
against 35-45%. At the final layer both models col-
lapse towards chance (IFR 28.0% vs 17.2%), which
is why we report the span distribution at layer 31
rather than at layer 39: the execution binding is a
mid-to-late-stack computation, and reading it off
the last layer alone would understate it for both
models.

Model Training General Alignment Math Code Tool-Calling Multilingual Safety
Granite-4.1-3BInstruct57.91 82.30 75.1964.1960.8059.89 80.89
LLMON 57.16 81.02 73.2565.4560.7162.68 89.64
Granite-4.1-8BInstruct64.95 87.06 84.09 72.38 68.27 68.6884.19
LLMON 63.31 86.69 79.63 71.01 67.93 68.2490.36
Table 1: LLMON-structured post-training vs.Instructchat-template baselines under matched full fine-tuning.
Each column is the unweighted mean over the benchmarks of that category (Section 4.3.1). Capability is broadly
preserved (the six-category capability mean is unchanged at 3B and 1.4 points lower at 8B) while Safety improves
at both scales. Best per model/category inbold.
Model Training OPI↓ASB↓
Granite-4.1-3BInstruct 100.0 99.2
LLMON0.1 5.6
Granite-4.1-8BInstruct 96.3 52.8
LLMON0.02 3.1
Table 2: LLMON-structured post-training vs.Instruct
chat-template baselines, measured byattack success
rate (ASR, %)on two published indirect prompt-
injection benchmarks: OPI (text-level injection) and
ASB (agentic tool-observation injection).Lower is bet-
ter: ASR is the fraction of adversarial prompts that
succeed, so a smaller value means a more robust model.
Best (lowest ASR) per model/benchmark inbold.
Model TrainingIndirect Direct Multi-turn
Granite-4.1-3BInstruct 2.8 17.6 15.0
LLMON84.6 37.1 28.0
Granite-4.1-8BInstruct 0.7 9.2 15.0
LLMON79.8 32.7 37.0
Table 3: Adversarial red-teaming with ARES, reported
asrobustness ( 1−ASR , %),higher is better, the
opposite direction to Table 2. Indirect attacks use
StruQ-style payloads planted in input, averaged over in-
distribution and out-of-distribution splits (ID: 90.3/83.6,
OOD: 79.0/75.9 for 3B/8B); direct attacks place the
adversarial instruction in the user turn; multi-turn is a
Crescendo variant with a 20-turn budget. Best per mod-
el/column inbold.
Layer-wise Instruction Focus Ratio.Figure 2
shows the IFR computed at each transformer layer
for both models on a representative distractor ex-
ample, in which three instructions (“summarize”,
“count the words”, “translate”) and two data seg-
ments coexist and the “ exec ” block binds only the
word-counting instruction. The two models are in-
distinguishable through the first twenty layers, both
fluctuating around 25-45%. They separate sharply
from layer 22: the LLMON-trained model rises
into a 55-89% band across layers 25-38, peaking at
89% at layer 28, whereas theInstructmodel peaksIFR↑DAS↑LEAK↓
Layer LLMON Inst. LLMON Inst. LLMON Inst.
2586.525.859.138.08.644.9
2887.237.762.236.66.334.9
3163.815.353.937.212.240.6
3475.826.357.935.66.447.2
3928.017.247.040.213.739.7
Table 4: Attention metrics (%) averaged over the distrac-
tor examples, at five representative layers, forGranite-
4.1-3BLLMON-trained vs.Instruct. The LLMON
model holds higher instruction focus and lower leakage
at every layer measured; the gap is widest at layers 25-
34 and narrows at the output.
at 62% and spends most of that range below 45%.
The implication is that the binding is not a surface
property of the prompt but a computation the net-
work performs, and that LLMON training installs it
in a specific region of the stack rather than shifting
attention uniformly.
Layer-wise Distractor Leakage.Figure 3
presents the complementary LEAK metric across
layers. Both models leak heavily in the first six
layers, peaking above 50%, which is consistent
with early layers aggregating context indiscrimi-
nately. From layer 7 the LLMON-trained model
drops to a 5-25% band and settles below 15% after
layer 25, while theInstructmodel stays in the 20-
50% range throughout and rises again in the last
ten layers. Distractor suppression is therefore not
a property of the input encoding (where the two
models behave alike) but something acquired in the
same region of the stack where instruction focus is
established.
Attention span distribution.Figure 4 presents
the per-span attention distribution and the three
metrics at layer 31 for both models. The LLMON-
trained model splits its span attention between the
target instruction (I-T, 23.4%) and the target data
it is bound to (D-T, 23.0%), with 35.6% on the

Figure 2: Instruction Focus Ratio (IFR) per transformer
layer for the LLMON-trainedGranite-4.1-3B(teal) vs.
itsInstructcounterpart (pink) on a distractor example.
The two are indistinguishable through layer 20 and sep-
arate sharply from layer 22, with the LLMON model
sustaining 55-89% across layers 25-38. Both fall at the
final layer.
Figure 3: Distractor leakage (LEAK) per transformer
layer. Both models leak heavily in the first six layers;
from layer 7 the LLMON-trained model (teal) settles
into a 5-25% band while theInstructmodel (pink) re-
mains between 20% and 50% and rises again in the late
layers.
prompt and execution structure and only 5.6% on
the largest distractor instruction: IFR = 71.4%,
DAS = 72.6%, LEAK = 9.9%. TheInstructmodel
inverts this. Its single largest recipient is a dis-
tractor instruction (I-D1, 47.5%), with a second
distractor taking 21.8% and the target instruction
only 14.5%; target data attention is negligible. Its
metrics record the failure directly: IFR = 17.3%,
DAS = 22.1%, LEAK = 49.9%. The two models
receive identical token sequences, so the difference
is entirely in what the network does with them.
Layer-wise Data Attention Score.Figure 5 com-
pletes the picture with DAS. It follows the same
shape as IFR: the two models track each other
through the first twenty layers, then the LLMON-
trained model rises into a 46-94% band across lay-
ers 23-38 while theInstructmodel stays between
14% and 56%. All three metrics therefore sepa-
rate in the same region of the stack, which is whatone expects if a single mechanism of resolving the
execution binding is responsible for all three.
Token-level attention flow.Figure 6 traces how
attention is allocated during generation, showing
the correlation between each generated token and
the KV-cache spans for the LLMON-trained model
at layer 31. The target data span (D-T) activates
precisely while the model is quoting the sentence
it was asked to count, and falls away once it moves
on to report the count; the target instruction span
(I-T) carries weight at the opening tokens and again
as the answer is formed. The distractor spans (I-
D1, I-D2, D-D1) stay near zero throughout decod-
ing. This token-resolved view complements the
per-layer metrics: the model does not merely arrive
at the right binding by the end of the forward pass,
it holds it step by step as it produces the answer.
6 Conclusion
We presented a structured post-training methodol-
ogy that uses LLMON’s explicit instruction-data
separation and identifier-based execution binding to
improve controlled execution in LLMs. Our evalua-
tion on twoGranite-4.1base models demonstrates
that: (i) LLMON-structured supervision reduces
attack success rates on indirect prompt-injection
benchmarks from 53-100% to 0.02-5.6%, with-
out catastrophic degradation on standard capability
measures; (ii) the effect holds at both 3B and 8B
scale, and extends to adversarial red-teaming on in-
direct attacks, though direct and multi-turn attacks
remain substantially harder; and (iii) attention-
based interpretability metrics (IFR, DAS, LEAK)
provide actionable per-layer failure diagnosis, re-
vealing how structural training reshapes internal
attention patterns. These findings suggest that ex-
plicit execution structure can be reliably taught to
LLMs through standard post-training, offering a
training-time complement to inference-time safety
mechanisms.
Limitations
We discuss several important limitations of this
work. First, our evaluation covers two model scales
(3B and 8B) within a single model family and
generation. While LLMON’s structural supervi-
sion is architecture-agnostic, it remains to be val-
idated whether the same magnitude of improve-
ment transfers to other pre-training corpora and
architectures, and to larger scales where models

Figure 4: Attention metrics and per-span attention distribution at layer 31 for the LLMON-trainedGranite-4.1-
3B(left) vs. itsInstructcounterpart (right). The LLMON model divides its attention between the bound target
instruction (I-T) and target data (D-T); theInstructmodel’s largest single recipient is a distractor instruction (I-D1,
47.5%).
Figure 5: Data Attention Score (DAS) per transformer
layer for the LLMON-trainedGranite-4.1-3B(teal) vs.
itsInstructcounterpart (pink). DAS separates in the
same region of the stack as IFR and LEAK, rising into
a 46-94% band across layers 23-38 for the LLMON
model.
may already exhibit stronger implicit instruction-
following. Second, LLMON’s delimiter tokens
increase context length by approximately 7-15%
depending on the example complexity. We do not
systematically quantify the impact on inference la-
tency and throughput, though sequence packing
at training time largely amortizes this cost. Third,
the privilege invariant addressesindirectinjection,
where the adversarial instruction arrives as input.
It offers far less protection when the adversarial
instructionisthe user’s instruction, since there is
then no instruction-data boundary to enforce: red-
teaming robustness reaches only 33-37% on direct
attacks and 28-37% on multi-turn attacks, against
80-85% on indirect ones, and it is a learned behav-
ior rather than an enforced guarantee: it substan-tially reduces, but does not eliminate, the chance
that an instruction embedded in data is followed,
and should be one layer of a defense rather than the
only one. Fourth, while our capability evaluation in-
cludes multilingual benchmarks, the LLMON train-
ing corpus itself is predominantly English, and we
have not characterized how tag overhead behaves
on morphologically rich languages or on tasks with
very long documents. Finally, our IFR/DAS/LEAK
metrics provide correlation between attention pat-
terns and output quality, but we do not establish a
causal mechanism. Attention weights are an im-
perfect proxy for information flow in transformers,
and the relationship between attention-level met-
rics and generation-level performance may not be
monotonic.
References
Jacob Austin, Augustus Odena, Maxwell Nye, Maarten
Bosma, Henryk Michalewski, David Dohan, Ellen
Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. 2021.
Program synthesis with large language models.arXiv
preprint arXiv:2108.07732.
Luca Beurer-Kellner, Marc Fischer, and Martin Vechev.
2023. Prompting is programming: A query language
for large language models.Proc. ACM Program.
Lang., 7(PLDI).
Manish Bhatt, Sahana Chennabasappa, Cyrus Niko-
laidis, Shengye Wan, Ivan Evtimov, Dominik Gabi,
Daniel Song, Faizan Ahmad, Cornelius Aschermann,
Lorenzo Fontana, et al. 2023. Purple llama cyber-
seceval: A secure coding benchmark for language
models.arXiv preprint arXiv:2312.04724.

Figure 6: Attention correlation weights between each generated token (columns, in decoding order) and the KV-
cache spans (rows) for the LLMON-trainedGranite-4.1-3Bat layer 31. Brighter cells indicate stronger attention.
The target data span (D-T) activates while the model quotes the sentence it was asked to count, and the distractor
spans (I-D1, I-D2, D-D1) remain near zero throughout decoding.
Federico Cassano, John Gouwar, Daniel Nguyen, Syd-
ney Nguyen, Luna Phipps-Costin, Donald Pinckney,
Ming-Ho Yee, Yangtian Zi, Carolyn Jane Anderson,
Molly Q Feldman, et al. 2023. Multipl-e: A scal-
able and polyglot approach to benchmarking neural
code generation.IEEE Transactions on Software
Engineering, 49(7):3675–3691.
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming
Yuan, Henrique Ponde De Oliveira Pinto, Jared Ka-
plan, Harri Edwards, Yuri Burda, Nicholas Joseph,
Greg Brockman, et al. 2021. Evaluating large
language models trained on code.arXiv preprint
arXiv:2107.03374.
Sizhe Chen, Julien Piet, Chawin Sitawarin, and David
Wagner. 2025a. {StruQ}: Defending against prompt
injection with structured queries. In34th USENIX
Security Symposium (USENIX Security 25), pages
2383–2400.
Sizhe Chen, Arman Zharmagambetov, Saeed Mahlouji-
far, Kamalika Chaudhuri, David Wagner, and Chuan
Guo. 2025b. Secalign: Defending against prompt in-
jection with preference optimization. InProceedings
of the 2025 ACM SIGSAC Conference on Computer
and Communications Security, pages 2833–2847.
Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan
Martic, Shane Legg, and Dario Amodei. 2017. Deep
reinforcement learning from human preferences. In
Advances in Neural Information Processing Systems
30: Annual Conference on Neural Information Pro-
cessing Systems 2017, December 4-9, 2017, Long
Beach, CA, USA.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian,
Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias
Plappert, Jerry Tworek, Jacob Hilton, Reiichiro
Nakano, Christopher Hesse, and John Schulman.
2021. Training verifiers to solve math word prob-
lems.Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie,
Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell,
Matei Zaharia, and Reynold Xin. 2023. Free dolly:
Introducing the world’s first truly open instruction-
tuned llm.
Jayanaka L. Dantanarayana, Yiping Kang, Kugesan
Sivasothynathan, Christopher Clarke, Baichuan Li,
Savini Kashmira, Krisztian Flautner, Lingjia Tang,
and Jason Mars. 2025. Mtp: A meaning-typed
language abstraction for ai-integrated programming.
Proc. ACM Program. Lang., 9(OOPSLA2).
DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingx-
uan Wang, Bochao Wu, Chengda Lu, Chenggang
Zhao, Chengqi Deng, Chenyu Zhang, et al. 2025.
Deepseek-v3 technical report.
IBM Granite Team. 2024. Granite 3.0 language mod-
els.URL: https://github.com/ibm-granite/
granite-3.0-language-models.
Kai Greshake, Sahar Abdelnabi, Shailesh Mishra,
Christoph Endres, Thorsten Holz, and Mario Fritz.
2023. More than you’ve asked for: A comprehen-
sive analysis of novel prompt injection threats to
application-integrated large language models.CoRR,
abs/2302.12173.
Alex Gu, Baptiste Rozière, Hugh Leather, Armando
Solar-Lezama, Gabriel Synnaeve, and Sida I Wang.
2024. Cruxeval: A benchmark for code reason-
ing, understanding and execution.arXiv preprint
arXiv:2401.03065.
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul
Arora, Steven Basart, Eric Tang, Dawn Song, and
Jacob Steinhardt. 2021. Measuring mathematical
problem solving with the math dataset.
Michael Hind, Basel Shbita, Bo Wu, Farhan Ahmed,
Chad DeLuca, Nathan Fulton, David Cox, and Dan
Gutfreund. 2026. LLMON: An LLM-native markup

language to leverage structure and semantics at the
llm interface.arXiv preprint arXiv:2603.22519.
Keegan Hines, Gary Lopez, Matthew Hall, Federico
Zarfati, Yonatan Zunger, and Emre Kiciman. 2024.
Defending against indirect prompt injection attacks
with spotlighting.arXiv preprint arXiv:2403.14720.
Hugging Face. 2025. Templates for chat models.
https://huggingface.co/docs/transformers/
en/chat_templating. Accessed: 2026-03-17.
IBM Granite Team. 2026. Granite 4.1 LLMs: How
they’re built. Hugging Face Blog. Published April
29, 2026.
Jean Kaddour, Joshua Harris, Maximilian Mozes, Her-
bie Bradley, Roberta Raileanu, and Robert McHardy.
2023. Challenges and applications of large language
models.arXiv preprint arXiv:2307.10169.
Omar Khattab, Arnav Singhvi, Paridhi Maheshwari,
Zhiyuan Zhang, Keshav Santhanam, Sri Vard-
hamanan, Saiful Haq, Ashutosh Sharma, Thomas T.
Joshi, Hanna Moazam, Heather Miller, Matei Za-
haria, and Christopher Potts. 2023. Dspy: Compiling
declarative language model calls into self-improving
pipelines.
George Kour, Marcel Zalmanovici, Naama Zwerdling,
Esther Goldbraich, Ora Fandina, Ateret Anaby Tavor,
Orna Raz, and Eitan Farchi. 2023. Unveiling safety
vulnerabilities of large language models. InProceed-
ings of the Third Workshop on Natural Language
Generation, Evaluation, and Metrics (GEM), pages
111–127.
Aitor Lewkowycz, Anders Andreassen, David Dohan,
Ethan Dyer, Henryk Michalewski, Vinay Ramasesh,
Ambrose Slone, Cem Anil, Imanol Schlag, Theo
Gutman-Solo, et al. 2022. Solving quantitative rea-
soning problems with language models.Advances
in neural information processing systems, 35:3843–
3857.
Lijun Li, Bowen Dong, Ruohui Wang, Xuhao Hu, Wang-
meng Zuo, Dahua Lin, Yu Qiao, and Jing Shao.
2024. Salad-bench: A hierarchical and comprehen-
sive safety benchmark for large language models. In
Findings of the Association for Computational Lin-
guistics: ACL 2024, pages 3923–3954.
Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Ling-
ming Zhang. 2023. Is your code generated by chatgpt
really correct? rigorous evaluation of large language
models for code generation.Advances in neural in-
formation processing systems, 36:21558–21572.
Jiawei Liu, Songrun Xie, Junhao Wang, Yuxiang Wei,
Yifeng Ding, and Lingming Zhang. 2024a. Evalu-
ating language models for efficient code generation.
arXiv preprint arXiv:2408.06450.
Xiaoxia Liu, Jingyi Wang, Xiaohan Yuan, Jun Sun, Guo-
liang Dong, Peng Di, Wenhai Wang, and Dongxia
Wang. 2026. Prompting frameworks for large lan-
guage models: A survey.ACM Comput. Surv.Yupei Liu, Yuqi Jia, Runpeng Geng, Jinyuan Jia, and
Neil Zhenqiang Gong. 2024b. Formalizing and
benchmarking prompt injection attacks and defenses.
In33rd USENIX Security Symposium (USENIX Se-
curity 24), pages 1831–1847.
IBM LLMON Team. 2026. LLMON.URL: https:
//github.com/generative-computing/llmon.
Iman Mirzadeh, Keivan Alizadeh-Vahid, Hooman
Shahrokhi, Oncel Tuzel, Samy Bengio, and Mehrdad
Farajtabar. 2025. Gsm-symbolic: Understanding the
limitations of mathematical reasoning in large lan-
guage models. InInternational Conference on Learn-
ing Representations, volume 2025, pages 94743–
94765.
Liubov Nedoshivina, Kieran Fraser, Mark Purcell, Am-
brish Rawat, Giulio Zizzo, Muhammad Zaid Hameed,
Stefano Braghin, Anisa Halimi, Cristian Morasso,
Ibrahim Malik, Naoise Holohan, and Giandomenico
Cornacchia. 2025. Ares: Ai robustness evaluation
system.
OpenAI, :, Sandhini Agarwal, Lama Ahmad, Jason
Ai, Sam Altman, Andy Applebaum, Edwin Arbus,
Rahul K. Arora, Yu Bai, Bowen Baker, Haiming Bao,
Boaz Barak, et al. 2025. gpt-oss-120b & gpt-oss-20b
model card.
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida,
Carroll L. Wainwright, Pamela Mishkin, Chong
Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray,
John Schulman, Jacob Hilton, Fraser Kelton, Luke
Miller, Maddie Simens, Amanda Askell, Peter Welin-
der, Paul F. Christiano, Jan Leike, and Ryan Lowe.
2022. Training language models to follow instruc-
tions with human feedback. InAdvances in Neural
Information Processing Systems 35: Annual Confer-
ence on Neural Information Processing Systems 2022,
NeurIPS 2022, New Orleans, LA, USA, November 28
- December 9, 2022.
Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie
Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E
Gonzalez. 2025. The berkeley function calling leader-
board (bfcl): From tool use to agentic evaluation of
large language models. InForty-second International
Conference on Machine Learning.
Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E
Gonzalez. 2024. Gorilla: Large language model
connected with massive apis.Advances in Neural
Information Processing Systems, 37:126544–126565.
Fábio Perez and Ian Ribeiro. 2022. Ignore previous
prompt: Attack techniques for language models.
arXiv preprint arXiv:2211.09527.
Niklas Pfister, Václav V olhejn, Manuel Knott, Santiago
Arias, Julia Bazi ´nska, Mykhailo Bichurin, Alan Com-
mike, Janet Darling, Peter Dienes, Matthew Fiedler,
et al. 2025. Gandalf the red: Adaptive security for
llms.arXiv preprint arXiv:2501.07927.
POML. 2025. Accessed: 2026-03-17. [link].

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christo-
pher D. Manning, Stefano Ermon, and Chelsea Finn.
2023. Direct preference optimization: Your language
model is secretly a reward model. InAdvances in
Neural Information Processing Systems 36: Annual
Conference on Neural Information Processing Sys-
tems 2023, NeurIPS 2023, New Orleans, LA, USA,
December 10 - 16, 2023.
David Rein, Betty Li Hou, Asa Cooper Stickland, Jack-
son Petty, Richard Yuanzhe Pang, Julien Dirani, Ju-
lian Michael, and Samuel R Bowman. 2023. Gpqa: A
graduate-level google-proof q&a benchmark.arXiv
preprint arXiv:2311.12022.
Angelika Romanou, Negar Foroutan, Anna Sotnikova,
Sree Harsha Nelaturu, Shivalika Singh, Rishabh Ma-
heshwary, Micol Altomare, Zeming Chen, Mohamed
Haggag, Alfonso Amayuelas, et al. 2025. Include:
Evaluating multilingual language understanding with
regional knowledge. InInternational Conference
on Learning Representations, volume 2025, pages
83291–83322.
Mark Russinovich, Ahmed Salem, and Ronen Eldan.
2025. Great, now write an article about that: The
crescendo {Multi-Turn }{LLM}jailbreak attack. In
34th USENIX Security Symposium (USENIX Security
25), pages 2421–2440.
David Saxton, Edward Grefenstette, Felix Hill, and
Pushmeet Kohli. 2019. Analysing mathematical rea-
soning abilities of neural models.arXiv preprint
arXiv:1904.01557.
Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta
Raileanu, Maria Lomeli, Eric Hambro, Luke Zettle-
moyer, Nicola Cancedda, and Thomas Scialom. 2023.
Toolformer: Language models can teach themselves
to use tools.Advances in neural information process-
ing systems, 36:68539–68551.
Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang,
Suraj Srivats, Soroush V osoughi, Hyung Won Chung,
Yi Tay, Sebastian Ruder, Denny Zhou, et al. 2022.
Language models are multilingual chain-of-thought
reasoners.arXiv preprint arXiv:2210.03057.
Shivalika Singh, Angelika Romanou, Clémentine Four-
rier, David Ifeoluwa Adelani, Jian Gang Ngui, Daniel
Vila-Suero, Peerat Limkonchotiwat, Kelly Marchi-
sio, Wei Qi Leong, Yosephine Susanto, et al. 2025.
Global mmlu: Understanding and addressing cultural
and linguistic biases in multilingual evaluation. In
Proceedings of the 63rd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 18761–18799.
Mirac Suzgun, Nathan Scales, Nathanael Schärli, Se-
bastian Gehrmann, Yi Tay, Hyung Won Chung,
Aakanksha Chowdhery, Quoc Le, Ed H Chi, Denny
Zhou, et al. 2023. Challenging big-bench tasks and
whether chain-of-thought can solve them. InFind-
ings of the Association for Computational Linguistics:
ACL 2023, pages 13003–13051.Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann
Dubois, Xuechen Li, Carlos Guestrin, Percy Liang,
and Tatsunori B Hashimoto. 2023. Stanford alpaca:
An instruction-following llama model.
Sam Toyer, Olivia Watkins, Ethan Mendes, Justin Sveg-
liato, Luke Bailey, Tiffany Wang, Isaac Ong, Karim
Elmaaroufi, Pieter Abbeel, Alan Ritter, et al. 2024.
Tensor trust: Interpretable prompt injection attacks
from an online game. InInternational Conference
on Learning Representations, volume 2024, pages
18714–18746.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all
you need. InAdvances in Neural Information Pro-
cessing Systems 30: Annual Conference on Neural
Information Processing Systems 2017, December 4-9,
2017, Long Beach, CA, USA.
Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni,
Abhranil Chandra, Shiguang Guo, Weiming Ren,
Aaran Arulraj, Xuan He, Ziyan Jiang, et al. 2024.
Mmlu-pro: A more robust and challenging multi-task
language understanding benchmark.Advances in
Neural Information Processing Systems, 37:95266–
95290.
Jason Wei, Maarten Bosma, Vincent Y . Zhao, Kelvin
Guu, Adams Wei Yu, Brian Lester, Nan Du, An-
drew M. Dai, and Quoc V . Le. 2022. Finetuned
language models are zero-shot learners. InThe Tenth
International Conference on Learning Representa-
tions, ICLR 2022, Virtual Event, April 25-29, 2022.
OpenReview.net.
Brandon T Willard and Rémi Louf. 2023. Efficient
guided generation for large language models.arXiv
preprint arXiv:2307.09702.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2022.
React: Synergizing reasoning and acting in language
models.arXiv preprint arXiv:2210.03629.
Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao,
Zhenting Wang, Chenlu Zhan, Hongwei Wang, and
Yongfeng Zhang. 2025. Agent security bench (asb):
Formalizing and benchmarking attacks and defenses
in llm-based agents. InInternational Conference
on Learning Representations, volume 2025, pages
35331–35366.
Zhengyue Zhao, Liwei Jiang, Yejin Choi, and Chaowei
Xiao. 2026. Safety mid-training: Internalizing LLM
safety as a foundational capability.arXiv preprint
arXiv:2610.26809.
Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue
Sun, Jeff Huang, Cody H Yu, Shiyi Cao, Christos
Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. 2024.
Sglang: Efficient execution of structured language
model programs.Advances in neural information
processing systems, 37:62557–62583.

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang,
Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen,
and Nan Duan. 2024. Agieval: A human-centric
benchmark for evaluating foundation models, 2023.
URL https://arxiv. org/abs/2304.06364.
Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha
Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and
Le Hou. 2023. Instruction-following evaluation for
large language models.
A LLMON Examples
A.1 Multi-instruction and Multi-data
Distractors
Figure 7 illustrates a distractor example: three
instructions and two data segments coexist, but
the “ exec ” block unambiguously binds only one
instruction-data pair. The model must learn to at-
tend selectively to the referenced segments and
ignore all others.

<|open|> instr<|:|>êí <|close|>
Summarize the text in one short sentence
<|open_end|>
<|open|> data<|:|>¡¿ <|close|>
The smart rabbit escaped the field
<|open_end|>
<|open|> data<|:|>Äî <|close|>
The quick brown fox jumps over the lazy dog
<|open_end|>
<|open|> instr<|:|>íä <|close|>
Count the number of words in the given sentence
<|open_end|>
<|open|> instr<|:|>Úê <|close|>
Translate the text into French
<|open_end|>
<|open|> exec<|:|>áÂ <|close|>
<|open|> exec<|.|>instruction <|close|> instr<|:|>íä <|open_end|>
<|open|> exec<|.|>input <|close|> data<|:|>Äî <|open_end|>
<|open|> exec<|.|>output <|close|> assistant<|:|>ðã <|open_end|>
<|open_end|>
<|open|> assistant<|:|>ðã <|close|>
The sentence "The quick brown fox jumps over the lazy dog" consists of 9 words.
<|open_end|> <|end_of_text|>
Figure 7: Multi-instruction distractor example. Three instructions and two data segments coexist; the bound target
instruction and data are shown in green and yellow, the distractors in lime and light-yellow. The exec block (orange)
references only “ instr<|:|>íä ” with input “ data<|:|>Äî ”. The model must execute only the word-counting
instruction, ignoring the summarization and translation distractors. Other special tokens are highlighted in light-pink.
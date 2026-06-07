# Discourse-Role Labels as Presentation-Time Variables for Context Use in Language Models

**Authors**: Jianguo Zhu

**Published**: 2026-06-02 18:12:57

**PDF URL**: [https://arxiv.org/pdf/2606.04109v1](https://arxiv.org/pdf/2606.04109v1)

## Abstract
Context-augmented language model systems often wrap supplied content with labels such as Reference:, Evidence:, Instruction:, Note:, or Example:, but the effect of these labels on reader-model behavior remains underexplored. We introduce a paired fixed-content probe over 500 MMLU-Pro items: each item receives the same misleading answer-bearing assertion under different discourse-role labels, and adoption is measured by whether the model outputs the injected wrong option. Across GPT-5.5, DeepSeek V4 Pro, Llama-3-8B-Instruct, and Qwen2.5-7B-Instruct, Misleading Adoption Rate shifts by 56-84 percentage points. Binding or source-like labels such as Instruction: and Reference: produce high adoption, whereas Example: consistently suppresses it. Paired tests, bootstrap intervals, final-instruction ablations, and Qwen final-step log-probability probes support a label-conditioned candidate preference. Boundary probes show where the effect weakens or persists: arithmetic tasks reduce adoption, passage-shaped external context preserves smaller label gaps, short-answer evaluation rules out option-letter copying, and nested-label conflicts suggest that illustrative framing can delimit adoption scope. A 200-case single-author manual audit confirms that the short-answer contrasts are stable under conservative adjudication. The resulting claim is bounded but practical: context-utilization and reader-side RAG benchmarks should report and control wrapper labels, because presentation choices can change measured reliance on supplied context.

## Full Text


<!-- PDF content starts -->

Highlights
Discourse-Role Labels as Presentation-Time Variables for Context
Use in Language Models
Jianguo Zhu
•Discourse-role labels shift misleading adoption by 56–84pp on MMLU-
Pro.
•An aligned no-label/instruction/example subset supports cross-model
replication.
•Passage-wrapper probes show the effect persists in passage-shaped
context.
•Log-probability and nested-label probes indicate preference and scope
effects.
•Wrapper labels should be reported in context-utilization benchmarks.arXiv:2606.04109v1  [cs.CL]  2 Jun 2026

Discourse-Role Labels as Presentation-Time Variables
for Context Use in Language Models
Jianguo Zhua
aChengdu University of Information Technology, Chengdu University of Information
Technology, No. 366, Section 5, Yinhe Road, Shuangliu
District, Chengdu, 610225, Sichuan, China
Abstract
Context-augmented language model systems often wrap supplied content with
labels such as Reference: ,Evidence: ,Instruction: ,Note:, or Example: ,
but the effect of these labels on reader-model behavior remains underexplored.
We introduce a paired fixed-content probe over 500 MMLU-Pro items: each
item receives the same misleading answer-bearing assertion under different
discourse-role labels, and adoption is measured by whether the model outputs
the injected wrong option. Across GPT-5.5, DeepSeek V4 Pro, Llama-3-
8B-Instruct, and Qwen2.5-7B-Instruct, Misleading Adoption Rate shifts by
56–84 percentage points. Binding or source-like labels such as Instruction:
and Reference: produce high adoption, whereas Example: consistently
suppresses it. Paired tests, bootstrap intervals, final-instruction ablations, and
Qwen final-step log-probability probes support a label-conditioned candidate
preference. Boundary probes show where the effect weakens or persists:
arithmetic tasks reduce adoption, passage-shaped external context preserves
smallerlabelgaps, short-answerevaluationrulesoutoption-lettercopying, and
nested-label conflicts suggest that illustrative framing can delimit adoption
scope. A 200-case single-author manual audit confirms that the short-answer
contrasts are stable under conservative adjudication. The resulting claim is
bounded but practical: context-utilization and reader-side RAG benchmarks
should report and control wrapper labels, because presentation choices can
change measured reliance on supplied context.
Keywords:Large language models, Context utilization, Retrieval-augmented
generation, Presentation-time formatting, Discourse-role labels, Evaluation
methodology

Retrieval /
tools / memory
supplied
external
informationWrapper layer
Reference:
/Evidence:
Instruction:
/Example:Reader model
answer decision
presentation-time variable studied here
Figure 1: Location of discourse-role labels in a context-augmented retrieval-reader pipeline.
The paper studies the wrapper layer while holding the supplied answer-bearing content
fixed.
1. Introduction
A retrieval-reader pipeline does not only decidewhatinformation to pass
to a language model. It also decideshowthat information is presented.
Retrieved passages, tool outputs, memory snippets, demonstrations, and
prompt-template blocks are commonly wrapped with short labels such as
Reference: ,Evidence: ,Instruction: ,Note:, or Example: . In many sys-
tems these labels are treated as cosmetic organization for human readers.
This paper asks whether they are also functional presentation-time variables
for machine readers.
The question matters for information processing and management because
context-augmented systems are increasingly evaluated by how faithfully and
selectively a model uses supplied information. If identical answer-bearing
content is adopted when labeled as a reference but suppressed when labeled
as an example, then a benchmark may be measuring not only context content
but also the role assigned to that content. Figure 1 locates the studied variable
in a retrieval-reader pipeline: after information is retrieved, generated, or
recalled, but before the reader model converts it into an answer.
Previous studies have shown that large-language-model behavior can
be sensitive to prompt wording, presentation-time formatting, punctuation,
underspecification, demonstrations, and retrieved context (Sclar et al., 2024;
He et al., 2024; Seleznyov et al., 2025; Liu et al., 2024, 2025; Zhang et al.,
2025). We study a narrower and more diagnostic problem: when the answer-
bearing content is fixed, does the discourse-role label attached to that content
determine whether the model adopts it? To connect this controlled question to
2

context-augmented settings, the paper treats passage-shaped wrapper probes
as a main reader-side evidence layer rather than as a cosmetic appendix check.
We call this phenomenonrole-conditioned reader adoption of supplied
content. The experimental design holds the question, answer choices, injected
wrongoption, wrong-optiontext, promptposition, andfinalanswerinstruction
fixed; onlythelocaldiscourse-rolelabelchanges. Theoutcomeisnotaggregate
accuracy drift across broad prompt variants, but paired within-item adoption
of the same controlled misleading assertion. This design turns a wrong answer
into a measurement device: if the model outputs the injected wrong option,
it has adopted the supplied claim under that label.
Our central contribution is an evaluation design that treats wrapper labels
as controlled variables. First, we introduce a paired protocol that isolates
discourse-role labels while holding the supplied content fixed. This makes
wrapper choice observable rather than implicit in context-use evaluation. Sec-
ond, we report cross-system replication across four reader models, including
a fully aligned no-label/instruction/example subset, with misleading adop-
tion shifting by 56–84 percentage points in the audited MMLU-Pro setting.
Third, we provide RAG-relevant reader-side evidence through passage-shaped
external context, while explicitly separating this probe from end-to-end
retrieval evaluation. Fourth, we provide decoding-level evidence from final-
instruction ablations and final-step log probabilities. Fifth, we characterize
task-affordance and output-format boundaries using GSM8K, mixed-language
prompts, label taxonomy probes, template variants, nested-label conflicts,
short-answer output, and a single-author manual audit of short-answer judg-
ments. Together, these results support a bounded methodological claim:
context-utilization benchmarks should report and control the wrapper labels
surrounding supplied or retrieved content, because those labels can change
measured reliance on external information.
2. Related work
2.1. Prompt sensitivity and presentation effects
A large body of prompt-sensitivity work has already made the broad point
that form matters: wording, formatting, punctuation, underspecification,
scoring artifacts, and prompt variants can all change model behavior (Sclar
et al., 2024; Chatterjee et al., 2024; Zhuo et al., 2024; Lu et al., 2024; He
et al., 2024; Razavi et al., 2025; Seleznyov et al., 2025; Hua et al., 2025; Pecher
et al., 2026; Liu and Chu, 2026). We use that literature as motivation rather
3

than as the comparison target. The narrower question here is whether a local
role label changes adoption when the assertion text, answer options, wrong
option, prompt position, and final answer instruction are held fixed.
2.2. In-context demonstrations
Examples are a natural source of ambiguity for this study. In-context-
learning work has shown that demonstrations, their order, and their presen-
tation affect model behavior (Wang et al., 2024a; Peng et al., 2024; Zhang
et al., 2024; Su et al., 2024; Qin et al., 2024; Agarwal et al., 2024; Bertsch
et al., 2025). Our use of Example: is different: the supplied sentence is not a
worked demonstration selected for imitation. It is the same counterfactual
answer-bearing assertion used in the other conditions, with only the wrapper
label changed. This lets us test the role assigned to the content rather than
the quality of an example set.
2.3. RAG faithfulness, context conflict, and source attribution
The closest application setting is retrieval-augmented reading. Prior work
has documented that models may underuse sufficient context, ignore retrieved
evidence, show position effects, or mix parametric and retrieved knowledge
under conflict (Liu et al., 2024; Wu et al., 2024; Qi et al., 2024; Es et al.,
2024; Shen et al., 2024; Liu et al., 2025; Hagström et al., 2025; Zhang et al.,
2025; Ming et al., 2025; Lin et al., 2026). Source-attribution and evidence-use
studies usually ask whether the answer is supported by the right material. We
ask a smaller reader-side question: if the material is fixed, can the label around
it change whether it is adopted? The passage-wrapper probe is included for
that reason, and is not reported as an end-to-end retriever–reader benchmark.
2.4. External-context security
There is also a security-adjacent reading of the result. Prompt injection,
indirect prompt injection, instruction/data separation, and external-context
security work study how untrusted text can affect model-integrated systems
(OWASP Foundation, 2025; Russinovich, 2024; Microsoft Agent Framework
Team,2026;Hinesetal.,2024;Chenetal.,2025;Zverevetal.,2025;Zhanetal.,
2024; Debenedetti et al., 2024; Yi et al., 2025; Zou et al., 2025; Al Masoud
et al., 2026; Khodayari et al., 2026). We do not evaluate attacks or propose a
defense. The relevance is more limited: ordinary context labels can change
adoption of a fixed external assertion, so label choice should be treated as
part of context-presentation design.
4

Work area Varies
content?Fixed
assertion?Wrapper role? Paired
adoption?Reporting
guidance?
Prompt
sensitivityOften Usually no Broad formatting Usually no Limited
In-context
demonstrationsY N Example
presentationUsually no Demo-focused
RAG faithful-
ness/conflictY Sometimes Rarely isolated Task
dependentEvidence-focused
Source
attributionY Usually no Source markers Usually no Citation-focused
This work N Y Y Y Wrapper labels
Table 1: Positioning relative to adjacent literatures. The contribution is not that prompts
matter in general, but that a local discourse-role wrapper changes paired adoption of
the same fixed misleading assertion and yields a concrete reporting recommendation for
context-utilization benchmarks.
3. Framework and methodology
Acontextual assertionis a statement prepended to a task input that
contains an answer or claim relevant to the current question. In the main
experiments, the assertion contains a multiple-choice option and its option
text, for example Reference: The answer is (B). <option text> or
Example: The answer is (B). <option text> . Adiscourse-role labelis the
short prefix assigning a role to that assertion. Some labels are binding or
evidential, some are suggestive or illustrative, and the nonce label Zorple:
preserves label syntax without carrying an interpretable discourse role.
For each item i, a wrong option wiis selected and reused across label
conditions. If the model output ˆyi,ℓunder label ℓequals wi, the supplied
assertion is counted as adopted. The primary metric is Misleading Adoption
Rate (MAR):
MAR(ℓ) =1
nX
i1[ˆyi,ℓ=w i].(1)
MAR should not be read as ordinary task error. It is a targeted adoption
measure: under a fixed counterfactual conflict, the model either follows or
resists the supplied misleading claim. Intuitively, labels such as Instruction:
and Reference: make the assertion feel closer to the current answer deci-
sion, while Example: frames the same sentence as illustrative rather than
operational.
5

The main task is MMLU-Pro-style multiple-choice question answering
(Wang et al., 2024b). Its ten-option format makes adoption of a specific wrong
answer directly observable and reduces ambiguity in the primary adoption
metric. For each sampled item, one incorrect option is selected using a fixed
random seed and reused across all label conditions, so label comparisons
are paired within item rather than confounded by wrong-answer plausibility.
GSM8K is used as a boundary task because arithmetic word problems require
independent derivation and make direct answer reuse less appropriate. Two
GPT-5.5 reader-setting probes use the same 500 paired MMLU-Pro items: a
passage-wrapper probe embeds the misleading answer text in paragraph-like
external context, and a short-answer probe requests a textual answer instead
of an option letter. The aligned cross-model subset uses the shared no-label,
Instruction: , and Example: conditions; broader label inventories are treated
as model-specific extensions rather than directly comparable label sets.
The model set is chosen to separate a detailed primary run from replication
and diagnostic probes. GPT-5.5 is used for the cleanest six-label run, final-
instruction ablations, the mixed-language rerun, and the reader-setting probes.
Qwen2.5-7B-Instruct supports the open-weight final-step log-probability anal-
ysis. DeepSeek V4 Pro and Llama-3-8B-Instruct add structural replication
with different model families and prompt implementations. API experiments
use deterministic decoding where available; local open-weight experiments
use greedy or temperature-zero generation. Since sample indices and wrong
options are paired across conditions, primary comparisons use exact McNe-
mar tests and paired bootstrap confidence intervals. Accuracy, none rate,
and other-output rate are reported where available so that adoption is not
conflated with generic answer failure.
4. Results
4.1. Discourse-role labels create a large adoption gradient
Figure 2 visualizes the main GPT-5.5 result. With 500 MMLU-Pro
examples per condition, Instruction: reaches 95.6% MAR and Reference:
reaches 80.2%, while Example: falls to 11.4%. The largest within-model
contrast, Instruction: vsExample: , is 84.2 percentage points. Paired tests
confirm the within-item nature of the effect: Instruction: vsExample: has
421 vs 0 discordant pairs ( p= 3.69×10−127), and Reference: vsExample:
has 345 vs 1 (p= 4.84×10−102).
6

Instruction ReferenceNo-label EvidenceZorpleExample
Wrapper label, sorted by marginal MARP1: binding-family adoption
P2: instruction-only adoption
P3: universal adoption
P4: top-binding-only adoption
P5: structural-slot activated
P6: full robustness
P7: binding-family skips Evidence
P8: binding-family skips No-labelEach row is one six-label signature; filled = adopted wrong option, hollow = not adopted.
Rows shown cover 478/500 items; rare patterns with <16 items omitted (total 22).(a) Dominant adoption-pattern signatures
Adopted
Not adopted
0 50 100 150 200 250 300
Items per pattern251
63
49
31
27
21
20
16
0 20 40 60 80 100
MAR with Wilson 95% CI (%)Instruction
Reference
No-label
Evidence
Zorple
Example95.6%
80.2%
72.4%
71.6%
15.4%
11.4%Δ = 84.2 pp; McNemar p<10−126; 421/0 discordant(b) Marginal misleading adoption rateGPT-5.5 fixed-content label probe (n = 500 paired items)Figure 2: GPT-5.5 fixed-content label probe over 500 paired MMLU-Pro items per condition.
Panel (a) groups items by identical six-label adoption patterns; the row label reports the
pattern index and the annotation reports the number of items in that pattern. Panel (b)
shows MAR with Wilson 95% confidence intervals and annotates the largest paired contrast
between Instruction: and Example: .No-label and Evidence: have nearly identical
marginal MAR (72.4% vs. 71.6%) but adopt non-identical item subsets, which is visible in
the grouped paired patterns.
Table 2 summarizes the same GPT-5.5 finding together with cross-model
structural replication. Because the full label inventories differ slightly across
model scripts, Table 3 reports the completely aligned subset shared by the four
main reader models: no label, Instruction: , and Example: . This aligned
view is more conservative than comparing each model’s strongest label against
its weakest label.
The cross-model picture is deliberately read at the role-family level.
Example: is the lowest-adoption condition in every evaluated model, and the
top-bottom spread ranges from 56.4pp to 84.2pp. The aligned subset gives the
same boundary without relying on model-specific high labels: Instruction:
exceeds Example: by 53.4–84.2pp, and even no-label prompts exceed Example:
by 39.2–61.0pp. At the same time, the full rankings are not identical; the
mean pairwise Spearman correlation over full rankings is moderate ( ρ= 0.59).
We therefore do not treat labels as having one universal scalar authority. What
replicates is coarser: binding or source-like roles tend to admit adoption, while
illustrative roles suppress it.
7

Model / setting High-adoption role Bare or neutral Example Spread
GPT-5.5Instruction:95.6%
Reference:80.2%no label 72.4%
Evidence:71.6%11.4% 84.2pp
DeepSeek V4 ProInstruction:74.8%
Note:73.6%no-format 44.6% 5.4% 69.4pp
Qwen2.5-7B-
InstructHint:89.0%
Instruction:81.2%no-format 61.0% 7.8% 81.2pp
Llama-3-8B-
InstructZorple:88.4%
Note:88.0%no-format 73.4% 32.0% 56.4pp
Table 2: Main adoption gradient and four-model structural replication. Values are MAR
percentages. Exact labels differ across scripts, so cross-model comparison emphasizes role
families and spread.
4.2. Labels interact with global instructions and pre-generation preferences
The decoding-level evidence is summarized in Table 4. First, GPT-5.5
final-instruction ablations show that global context-use instructions amplify
adoption but do not erase the local label boundary. Under a reference-based
final instruction, Reference: reaches 99.6% MAR, yet Example: remains
at 26.8%. A fixed-effect logistic analysis over 9,000 records finds significant
label, final-instruction, and label-by-instruction terms (label p <10−300; final
instructionp= 5.28×10−243; interactionp= 4.90×10−19).
Second, Qwen2.5-7B-Instruct final-step log-probability probes show label-
conditioned differences in candidate-answer preference before generation. The
wrong-correct gap is9 .149under Reference: ,9.196under Instruction: ,
and−5.697under Example: . The paired Reference: –Example: gap is 14.846
log-probability points (95% CI [14.257, 15.444]); the Instruction: –Example:
gap is 14.893 points (95% CI [14.301, 15.485]). Thus, before generation, the
same wrong option receives higher relative probability under binding labels
but not under the illustrative label.
This goes beyond a final-output formatting effect. Before generation,
the same supplied assertion already has a different standing relative to the
candidate answer depending on the local discourse role.
4.3. Nested-label conflicts reveal scope-sensitive role assignment
The preceding experiments vary a single wrapper at a time. A natural
follow-up is whether nested, conflicting wrappers combine as simple authority
cues. We therefore run a nested-label conflict probe on the same 500-item
8

Modeln/cond. No labelInstruction: Example: Aligned con-
trast
GPT-5.5 500 72.4% 95.6% 11.4% Inst.–Ex
+84.2pp;
No-label–Ex
+61.0pp
DeepSeek V4 Pro 500 44.6% 74.8% 5.4% Inst.–Ex
+69.4pp;
No-label–Ex
+39.2pp
Qwen2.5-7B-Instruct 500 61.0% 81.2% 7.8% Inst.–Ex
+73.4pp;
No-label–Ex
+53.2pp
Llama-3-8B-Instruct 500 73.4% 85.4% 32.0% Inst.–Ex
+53.4pp;
No-label–Ex
+41.4pp
Table 3: Completely aligned cross-model core-label subset. Values are all-trial MAR
percentages on the pure-English fixed-wrong-assertion setting. The table uses only labels
shared by all four main models, so it avoids comparing model-specific label inventories.
fixed-wrong-assertion setting for GPT-5.5 and DeepSeek V4 Pro. The probe
includestwosingle-labelbaselines, Reference: andExample: , andfournested
wrappers: Reference⊃Example ,Example⊃Reference ,Instruction⊃
Example, and Example⊃Instruction . We analyze this probe internally
within its own six-condition run, because its single-label baseline differs
numerically from the main six-label run. The result should therefore be read
as evidence about nested-label conflict structure, not as a replacement for the
main MAR estimates.
Table 5 shows that nested labels do not behave as simple authority stack-
ing. In both models, pure Reference: produces high MAR, while pure
Example: produces low MAR. However, placing Reference orInstruction
outside an inner Example frame does not restore high adoption. On GPT-5.5,
Reference⊃Example reaches 26.2% MAR, far below Reference: alone at
82.6%; Instruction⊃Example falls to 3.0%. On DeepSeek V4 Pro, the
same nested conditions reach 9.8% and 3.8%, respectively, again far below
9

Probe Measure Binding/source-likeExample:Contrast
GPT-5.5, neutral final
instructionMARInstruction:96.2% 11.6% +84.6pp
GPT-5.5, reference-based
final instructionMARReference:99.6% 26.8% +72.8pp
Qwen2.5 final-step
probabilityWrong-pref.
rateInstruction:89.6% 28.4% +61.2pp
Qwen2.5 final-step
probabilitylogp(w)−
logp(c)Instruction:9.196 -5.697 +14.893
Table 4: Pre-generation preference evidence. Binding/source-like labels remain above
Example: under stronger final instructions, and Qwen2.5-7B-Instruct assigns higher relative
probability to the misleading candidate before generation.
Condition GPT-5.5 DeepSeek V4 Pro
MAR Acc. None Other MAR Acc. None Other
Reference:82.6 17.2 0.0 0.2 64.4 12.4 22.8 0.4
Example:11.0 80.6 0.0 8.4 7.2 64.6 23.2 5.0
Reference⊃Example26.2 67.4 0.0 6.4 9.8 61.8 22.6 5.8
Example⊃Reference28.2 65.6 0.0 6.2 14.4 56.0 24.6 5.0
Instruction⊃Example3.0 86.0 0.0 11.0 3.8 68.2 21.6 6.4
Example⊃Instruction5.8 84.2 0.0 10.0 5.0 63.8 25.0 6.2
Table 5: Nested-label conflict probe over 500 paired items per condition. Values are
percentages. Nested illustrative framing keeps MAR far below the pure Reference: baseline
in both models, but the response distributions differ: GPT-5.5 converts most non-adoption
cases into correct answers or other explicit answers, whereas DeepSeek V4 Pro maintains a
stable non-answer component.
Reference: alone at 64.4%. Conversely, Example⊃Reference remains far
below Reference: alone in both models. This pattern suggests that illustra-
tive framing can act as a scope delimiter: when answer-bearing content is
explicitly embedded as an example, it is less likely to be treated as operational
evidence even when an authority-like label appears elsewhere in the wrapper.
The two models also differ in how low adoption is realized. GPT-5.5 has
no non-answer outputs in this probe, so lower MAR mostly corresponds to
correct-answer recovery or other explicit non-adoption outputs. DeepSeek
V4 Pro maintains a stable 21–25% non-answer rate across conditions, with
a smaller but persistent other-output component. Thus, similar aggregate
reductions in adoption can hide different response-distribution patterns. We
do not use this probe to claim a general decision mechanism. Instead, it
shows that MAR should be reported together with accuracy, none rate, and
other-output rate, because adoption behavior and final response type need
10

Probe Setting High role Low role Interpretation
Task boundary MMLU-Pro direct
assertion, GPT-5.595.6% 11.4% Direct answer reuse
is available
Task boundary GSM8K previous-
solution, GPT-5.50.0% – Independent deriva-
tion resists adoption
Task boundary GSM8K Chinese an-
swer hint, DeepSeek
Chat7.5% – Stronger cue yields
weak residual adop-
tion
Mixed language English labels,
Chinese assertion,
GPT-5.593.8% 13.0% Coarse boundary
persists
Template variants Qwen2.5 six asser-
tion templates23.5% 10.2% Pattern survives
weaker cues
Table 6: Boundary and robustness checks. High role reports the strongest binding/source-
like condition; low role reportsExample:when applicable. Values are MAR percentages.
not vary in the same way across models.
4.4. The effect is bounded by task affordance
Table 6 summarizes boundary and robustness checks. GSM8K sharply
attenuates adoption: when solving requires arithmetic derivation, adoption
falls near zero even with misleading previous-solution prompts. This supports
the task-affordance interpretation. Role-conditioned adoption is strongest
when the supplied assertion can be directly reused as an answer.
The mixed-language rerun narrows the claim without overturning it. Using
English labels with Chinese assertion content on the same 500 sample indices,
the coarse boundary remains: Instruction: reaches 93.8%, Reference:
70.8%, and Example: 13.0%. However, pure-English prompts remain signifi-
cantly higher for some labels, so the result supports robustness of the coarse
boundary rather than full multilingual invariance.
Label-taxonomy experiments further show that the effect is not driven
by a single word pair such as Reference: versus Example: . Instruction-
like, source/evidence, neutral-context, and suggestive labels cluster above
illustrative labels. Template-robustness experiments show that the pattern
survives alternatives to The answer is (X) , although direct answer cues
remainstrongest. Candidate-onlyQwenlogit-lensprobesprovideaconsistency
11

check: final-layer wrong-correct gaps follow the same structure and correlate
highly with final-step log-probability gaps ( r≈0.97–0.99), although we do
not infer neural circuits from this probe.
4.5. Reader-setting probes: passage wrappers and short answers
The original probe uses a direct assertion such as The answer is (B) and
a multiple-choice output format. Table 7 tests whether the effect survives
two reader-setting changes. The passage-wrapper probe is the main bridge
to context-augmented reading: it embeds the misleading answer text inside
short paragraph-like external context and varies only the wrapper. It does not
implement retrieval, indexing, reranking, corpus construction, or a deployed
retriever–generator pipeline; it is a synthetic external-context probe. Even
under this weaker, less direct cue, Reference: and Instruction: remain
aboveExample:by about 16pp.
The short-answer probe removes option-letter output as the target. Can-
didate answer texts are shown without A/B/C labels, and ambiguous textual
answers are judged with a two-stage procedure. The label effect not only
persists but sharpens: Reference: reaches 76.0%, Instruction: 52.6%,
and Example: 5.2%. Explicit option-letter outputs occur in only 0.4% of
responses. To audit the judgment step, we manually labeled 200 model-
judged short-answer cases, stratified as 50 per condition, using a pre-specified
four-label rubric ( ADOPT_WRONG , CORRECT, OTHER, AMBIGUOUS). The
automatic and manual labels agree on 87.5% of audited cases, with Cohen’s
κ= 0.765. Agreement is balanced across conditions (86–90%). The largest
disagreement source is automatic OTHER versus manual CORRECT (11
cases), while direct ADOPT_WRONG /CORRECT cross-confusions account for
only five cases. A conservative adjudication changes every condition-level
MAR by at most 0.6pp: the Reference: –Example: contrast remains 70.8pp,
and the Instruction: –Example: contrast changes from 47.4pp to 46.8pp.
Thus, the short-answer result is not an option-letter-copying artifact and is
stable under manual adjudication.
4.5.1. Wrapper labels and output format are independent adoption channels
A useful implication emerges from the short-answer probe: wrapper labels
and output format are partially independent adoption channels. Removing
option-letter output sharply reduces no-label adoption from 72.4% to 15.2%,
butReference: remains high at 76.0%. Thus, the binding label can preserve
adoption even when the direct letter-copying channel is largely removed.
12

Probe No
labelReference: Instruction: Example:Primary
contrast
Direct assertion 72.4 80.2 95.6 11.4 Inst.–Ex
+84.2pp
Passage-shaped context 29.8 39.4 39.2 23.2 Ref.–Ex
+16.2pp;
Inst.–Ex
+16.0pp
Short-answer output 15.2 76.0 52.6 5.2 Ref.–Ex
+70.8pp;
Inst.–Ex
+47.4pp
Table 7: Reader-setting probes. Values are MAR percentages over 500 paired GPT-5.5
items per condition. Passage-shaped context is a synthetic external-context reader probe,
not a deployed RAG pipeline. The short-answer probe has a 0.4% explicit option-letter
output rate.
5.Methodological recommendation for context-utilization bench-
marks
The practical recommendation is modest: wrapper text should be treated
as part of the benchmark specification. A benchmark that labels a passage
asReference: may measure a different reader behavior from one that labels
the same passage as Example: ,Note:, or no wrapper. This matters when
comparing model reliance on provided context, robustness to conflicting
information, or faithfulness to retrieved evidence.
Three low-cost changes would make this variable visible.
1.Report wrapper labels and delimiters.Benchmark descriptions
should include the exact text surrounding supplied context, not only
the retrieved passage or answer format.
2.Add content-fixed paired variants.When evaluating whether a
reader model uses or ignores external information, studies should include
conditions where the supplied content is identical and only the wrapper
changes.
3.Separate passage-shaped probes from end-to-end retrieval
benchmarks.A prompt containing passage-like context can diagnose
reader behavior, but it should not be reported as a full retriever–reader
evaluation unless retrieval, indexing, reranking, and corpus construction
are actually implemented.
13

Audit item Result
Manual audit sample 200 short-answer cases; 50 per condition
Automatic–manual agreement 87.5% (175/200); Cohen’sκ= 0.765
Per-condition agreement 86–90%, balanced across conditions
Main disagreement source automatic OTHER vs. manual CORRECT: 11
cases
Direct label conflictADOPT_WRONGvs. CORRECT cross-confusions:
5 cases
Conservative MAR shift all condition-level shifts≤0.6pp
Key contrasts after
adjudicationRef.–Ex 70.8pp; Inst.–Ex 46.8pp
Table 8: Manual audit of short-answer judgments. The audit was stratified by condition and
conducted using a pre-specified four-label rubric over ADOPT_WRONG , CORRECT, OTHER,
and AMBIGUOUS. Conservative adjudication leaves the main short-answer contrasts
essentially unchanged.
These changes are intentionally lightweight. They do not require every
evaluation to become a full end-to-end RAG benchmark; they make the
reader-side presentation choice visible enough that wrapper effects are not
mistaken for model-level context-utilization differences.
6. Discussion
6.1. Design implications for context-augmented systems
For prompt and system designers, the immediate lesson is not to treat
Reference: ,Instruction: ,Evidence: , and Example: as interchangeable
decoration. A non-binding context block may become more adoptable when
framed as a reference or instruction; an illustrative block should be marked
as such. The numeric pattern will not necessarily transfer unchanged to every
retrieval-augmented system, but the design variable itself should be logged
and evaluated rather than left implicit.
6.2. Safety implications
Although this is not a security paper, the results have safety implications.
Authority-like labels can increase adoption of misleading content, while il-
lustrative or non-binding labels can reduce adoption in controlled settings.
14

This suggests that discourse-role framing may be useful as one component
in broader context-risk mitigation, but it is not a standalone defense against
prompt injection, RAG poisoning, or instruction/data confusion.
7. Limitations and reproducibility
Theclaimisintentionallyboundedbydesignchoicesthatisolatethereader-
side label effect. First, the evidence is behavioral and decoding-level; we do
not identify internal circuits, perform activation interventions, or establish
a causal neural mechanism. Second, the stable pattern is block-level rather
than a universal fine-grained ranking: absolute MAR values and within-block
rankings vary across model families, language settings, and final instructions.
For this reason, we include both a structural replication table and a completely
aligned no-label/Instruction:/Example:cross-model subset.
Third, the main paired probe uses a 500-item MMLU-Pro subset because
it allows precise wrong-answer adoption measurement. We deliberately trade
benchmark breadth for within-item statistical control. The broader evidence
spans several distributional axes already available in the study: GSM8K,
Chinese assertions with English wrappers, nested-label conflicts, passage-
shaped external context, short-answer output, and four reader-model families.
We do not claim generalization to all QA datasets, option formats, or item
construction protocols. Fourth, the passage-wrapper probe is not deployed
RAG because it intentionally omits retrieval, indexing, reranking, source
diversity, and corpus construction. Its smaller contrasts should therefore be
read as controlled reader-side evidence that wrapper effects can persist in
passage-shaped contexts, not as a claim about effect sizes in deployed retrieval
systems.
Fifth, the nested-label conflict probe should be interpreted as a within-run
structural probe rather than as a replacement for the main six-label MAR
estimates, because its single-label baselines differ numerically from the main
run. Finally, the short-answer probe requires judging ambiguous textual
answers. We report judge-method counts, letter-output rates, a 200-case
single-author manual audit, and a conservative adjudication rerun. The audit
supports the reliability of the short-answer result, but it remains a single-
annotator manual audit rather than a multi-annotator annotation study. We
therefore use the short-answer probe to validate the output-format boundary
and the non-letter-copying explanation, while keeping the primary claim
anchored in the paired multiple-choice evidence.
15

The experiments use fixed random seeds and paired sample indices wher-
ever possible. Formal raw records include sample index, original index, con-
dition, correct option, injected wrong option, prompt, raw response, parsed
prediction, and adoption indicators. The reader-setting probes also record
passage text, raw short answer, judge method, matched option, and letter-
output flags. A complete replication package should include prompt templates,
wrapper text, sampling scripts, model names, decoding parameters, parser
code, judge prompts for short-answer ambiguity, aggregate result tables, and
scripts for McNemar tests and paired bootstrap confidence intervals. For the
short-answer audit, the structured artifact should additionally include the
anonymized answer text, automatic judge label, manual audit label, adju-
dicated label, and a disagreement flag for every audited case. If raw model
responses cannot be released in full, a useful supplement should still provide
de-identified examples, annotation rules, and enough structured records to
reproduce the reported tables.
8. Ethical considerations
This work studies how models adopt misleading context. Such findings
could be misused to craft more persuasive misleading prompts. We therefore
frame the experiments as controlled behavioral analysis rather than attack
optimization, avoid releasing a toolkit for maximizing attack success, and do
not claim that specific labels constitute universal attacks. The constructive
implication is that system designers should report and control how retrieved,
generated, or user-supplied context is framed.
9. Conclusion
The experiments point to a simple conclusion: supplied context is not
defined only by its content, but also by the role assigned to it in the prompt.
Withanswer-bearingcontentheldfixed, bindingandsource-likelabelspromote
adoption, while illustrative labels such as Example: suppress it. This pattern
recurs across four model families in the audited setting, survives stronger
final instructions, appears in Qwen final-step candidate preferences, weakens
on tasks that require independent reasoning, persists in passage-shaped
context, and remains visible under manually audited short-answer evaluation.
Nested-label conflicts add a further boundary case by showing scope-sensitive
behavior rather than simple authority stacking. The claim is bounded, but
16

the reporting implication is direct: context-utilization research and reader-
side RAG evaluation should document and control the discourse-role labels
surrounding supplied or retrieved content.
References
Agarwal, R., et al., 2024. Many-shot in-context learning. arXiv preprint URL:
https://arxiv.org/abs/2404.11018. arXiv:2404.11018.
Al Masoud, A., Arazzi, M., Nocera, A., 2026. SD-RAG: A prompt-injection-resilient
framework for selective disclosure in retrieval-augmented generation. arXiv preprint
URL:https://arxiv.org/abs/2601.11199. arXiv:2601.11199.
Bertsch, A., et al., 2025. In-context learning with long-context models: An in-depth
exploration. arXiv preprint URL:https://arxiv.org/abs/2405.00200.
arXiv:2405.00200.
Chatterjee, A., et al., 2024. POSIX: A prompt sensitivity index for large language models,
in: Findings of the Association for Computational Linguistics: EMNLP. URL:
https://aclanthology.org/2024.findings-emnlp.852/.
Chen, S., Piet, J., Sitawarin, C., Wagner, D., 2025. StruQ: Defending against prompt
injection with structured queries, in: USENIX Security Symposium. URL:
https://arxiv.org/abs/2402.06363. arXiv:2402.06363.
Debenedetti, E., Zhang, J., Balunovic, M., Beurer-Kellner, L., Fischer, M., Tramer, F.,
2024. AgentDojo: A dynamic environment to evaluate prompt injection attacks and
defenses for LLM agents. arXiv preprint URL:https://arxiv.org/abs/2406.13352.
arXiv:2406.13352.
Es, S., James, J., Espinosa-Anke, L., Schockaert, S., 2024. RAGAs: Automated evaluation
of retrieval augmented generation, in: Proceedings of the 18th Conference of the
European Chapter of the Association for Computational Linguistics: System
Demonstrations. URL:https://aclanthology.org/2024.eacl-demo.16/.
Hagström, L., et al., 2025. A reality check on context utilisation for retrieval-augmented
generation, in: Proceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics. URL:https://aclanthology.org/2025.acl-long.968/.
He, J., et al., 2024. Does prompt formatting have any impact on LLM performance?
arXiv preprint URL:https://arxiv.org/abs/2411.10541. arXiv:2411.10541.
Hines, K., Lopez, G., Hall, M., Zarfati, F., Zunger, Y., Kiciman, E., 2024. Defending
against indirect prompt injection attacks with spotlighting. arXiv preprint URL:
https://arxiv.org/abs/2403.14720. arXiv:2403.14720.
17

Hua, A., Tang, K., Gu, C., Gu, J., Wong, E., Qin, Y., 2025. Flaw or artifact? rethinking
prompt sensitivity in evaluating LLMs, in: Proceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing. URL:
https://aclanthology.org/2025.emnlp-main.1006/. arXiv:2509.01790.
Khodayari, S., Zhang, X., Acharya, B., Pellegrino, G., 2026. Indirect prompt injection in
the wild: An empirical study of prevalence, techniques, and objectives. arXiv preprint
URL:https://arxiv.org/abs/2604.27202. arXiv:2604.27202.
Lin, C., Wen, Y., Su, D., Tan, H., Sun, F., Chen, M., Bao, C., Lv, Z., 2026. Resisting
contextual interference in RAG via parametric-knowledge reinforcement, in:
International Conference on Learning Representations. URL:
https://openreview.net/forum?id=6Qc6sO1jh9.
Liu, A., Press, O., Smith, N.A., Hajishirzi, H., 2025. Sufficient context: A new lens on
retrieval augmented generation systems, in: International Conference on Learning
Representations. URL:https://arxiv.org/abs/2411.06037.
Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., Liang, P., 2024.
Lost in the middle: How language models use long contexts. Transactions of the
Association for Computational Linguistics URL:
https://aclanthology.org/2024.tacl-1.9/.
Liu, Y., Chu, C., 2026. Understanding the prompt sensitivity. arXiv preprint URL:
https://arxiv.org/abs/2604.18389. arXiv:2604.18389.
Lu, S., Schuff, H., Gurevych, I., 2024. How are prompts different in terms of sensitivity?,
in: Proceedings of the 2024 Conference of the North American Chapter of the
Association for Computational Linguistics. URL:
https://aclanthology.org/2024.naacl-long.325/.
Microsoft Agent Framework Team, 2026. Stop prompt injection from hijacking your agent:
New security capabilities now released within agent framework. Microsoft Agent
Framework Blog. URL:https://devblogs.microsoft.com/agent-framework/fides/.
Ming, Y., et al., 2025. FaithEval: Can your language model stay faithful to context, even
if “the moon is made of marshmallows”?, in: International Conference on Learning
Representations. URL:https://arxiv.org/abs/2410.03727. arXiv:2410.03727.
OWASP Foundation, 2025. OWASP top 10 for large language model applications.
Technical report. URL:https://owasp.org/www-project-top-10-for-large-
language-model-applications/.
Pecher, B., Spiegel, M., Belanec, R., Cegin, J., 2026. Revisiting prompt sensitivity in large
language models for text classification: The role of prompt underspecification. arXiv
preprint URL:https://arxiv.org/abs/2602.04297. arXiv:2602.04297.
18

Peng, K., et al., 2024. Revisiting demonstration selection strategies in in-context learning,
in: Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics. URL:https://aclanthology.org/2024.acl-long.492/.
Qi, J., Sarti, G., Fernandez, R., Bisazza, A., 2024. Model internals-based answer
attribution for trustworthy retrieval-augmented generation, in: Proceedings of the 2024
Conference on Empirical Methods in Natural Language Processing. URL:
https://aclanthology.org/2024.emnlp-main.347/.
Qin, C., et al., 2024. In-context learning with iterative demonstration selection, in:
Findings of the Association for Computational Linguistics: EMNLP. URL:
https://aclanthology.org/2024.findings-emnlp.438/.
Razavi, A., Soltangheis, M., Arabzadeh, N., Salamat, S., Zihayat, M., Bagheri, E., 2025.
Benchmarking prompt sensitivity in large language models, in: European Conference on
Information Retrieval. URL:https://arxiv.org/abs/2502.06065. arXiv:2502.06065.
Russinovich, M., 2024. How microsoft discovers and mitigates evolving attacks against AI
guardrails. Microsoft Security Blog. URL:
https://www.microsoft.com/en-us/security/blog/2024/04/11/how-microsoft-
discovers-and-mitigates-evolving-attacks-against-ai-guardrails/.
Sclar, M., Choi, Y., Tsvetkov, Y., Suhr, A., 2024. Quantifying language models’ sensitivity
to spurious features in prompt design, or: How i learned to start worrying about
prompt formatting, in: International Conference on Learning Representations. URL:
https://openreview.net/forum?id=RIu5lyNXjT.
Seleznyov, M., Chaichuk, M., Ershov, G., Panchenko, A., Tutubalina, E., Somov, O., 2025.
When punctuation matters: A large-scale comparison of prompt robustness methods for
LLMs, in: Findings of the Association for Computational Linguistics: EMNLP. URL:
https://aclanthology.org/2025.findings-emnlp.1109/. arXiv:2508.11383.
Shen, X., et al., 2024. Assessing “implicit” retrieval robustness of large language models,
in: Proceedings of the 2024 Conference on Empirical Methods in Natural Language
Processing. URL:https://aclanthology.org/2024.emnlp-main.507/.
Su, Y., et al., 2024. Demonstration augmentation for zero-shot in-context learning, in:
Findings of the Association for Computational Linguistics: ACL. URL:
https://aclanthology.org/2024.findings-acl.846/.
Wang, L., Yang, N., Wei, F., 2024a. Learning to retrieve in-context examples for large
language models, in: Proceedings of the 18th Conference of the European Chapter of
the Association for Computational Linguistics. URL:
https://aclanthology.org/2024.eacl-long.105/.
Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X.,
Jiang, Z., Li, T., Ku, M., Wang, K., Zhuang, A., Fan, R., Yue, X., Chen, W., 2024b.
19

MMLU-Pro: A more robust and challenging multi-task language understanding
benchmark, in: Advances in Neural Information Processing Systems. URL:
https://arxiv.org/abs/2406.01574.
Wu, D., et al., 2024. Synchronous faithfulness monitoring for trustworthy
retrieval-augmented generation, in: Proceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing. URL:
https://aclanthology.org/2024.emnlp-main.527/.
Yi, J., Xie, Y., Zhu, B., Kiciman, E., Sun, G., Xie, X., Wu, F., 2025. Benchmarking and
defending against indirect prompt injection attacks on large language models, in: ACM
SIGKDD Conference on Knowledge Discovery and Data Mining. URL:
https://arxiv.org/abs/2312.14197. arXiv:2312.14197.
Zhan, Q., Liang, Z., Ying, Z., Kang, D., 2024. InjecAgent: Benchmarking indirect prompt
injections in tool-integrated large language model agents, in: Findings of the
Association for Computational Linguistics: ACL. URL:
https://aclanthology.org/2024.findings-acl.624/.
Zhang, M., et al., 2024. The impact of demonstrations on multilingual in-context learning:
A multidimensional analysis, in: Findings of the Association for Computational
Linguistics: ACL. URL:https://aclanthology.org/2024.findings-acl.438/.
Zhang, Q., Xiang, Z., Xiao, Y., Wang, L., Li, J., Wang, X., Su, J., 2025. FaithfulRAG:
Fact-level conflict modeling for context-faithful retrieval-augmented generation, in:
Proceedings of the 63rd Annual Meeting of the Association for Computational
Linguistics. URL:https://aclanthology.org/2025.acl-long.1062/.
Zhuo, J., et al., 2024. ProSA: Assessing and understanding the prompt sensitivity of
LLMs, in: Findings of the Association for Computational Linguistics: EMNLP. URL:
https://aclanthology.org/2024.findings-emnlp.108/.
Zou, W., Geng, R., Wang, B., Jia, J., 2025. PoisonedRAG: Knowledge corruption attacks
to retrieval-augmented generation of large language models, in: USENIX Security
Symposium. URL:https://arxiv.org/abs/2402.07867. arXiv:2402.07867.
Zverev, E., Abdelnabi, S., Tabesh, S., Fritz, M., Lampert, C.H., 2025. Can LLMs separate
instructions from data? and what do we even mean by that?, in: International
Conference on Learning Representations. URL:https://arxiv.org/abs/2403.06833.
arXiv:2403.06833.
20
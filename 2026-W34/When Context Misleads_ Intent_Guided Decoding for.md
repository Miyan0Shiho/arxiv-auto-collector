# When Context Misleads: Intent-Guided Decoding for Robust Retrieval-Augmented Generation

**Authors**: Haolin Jin, Pengyue Yang, Huaming Chen

**Published**: 2026-08-17 12:56:43

**PDF URL**: [https://arxiv.org/pdf/2608.16515v1](https://arxiv.org/pdf/2608.16515v1)

## Abstract
Retrieval-augmented generation (RAG) improves large language models by grounding generation in external evidence, but it also introduces a source trust problem: retrieved context may be useful, irrelevant, or even misleading. Existing RAG systems often apply a fixed trust policy toward retrieved evidence, which can either over-trust incorrect context or underuse context when the user explicitly asks for context-following behavior. Therefore, we propose Intent-Guided Decoding (IGD), a framework that arbitrates between retrieved context and parametric memory according to user intent. IGD uses answer-level filtering and token-level correction to steer the final decoding trajectory between retrieved context and parametric memory. We evaluate IGD on three faithful QA benchmarks and three factual-conflict benchmarks across five LLMs, IGD substantially improves factual recovery, achieving gains of up to 65.4 percentage points on factual-conflict benchmarks over Direct RAG, while preserving or improving strict context-following behavior, this findings highlight the importance of balancing factuality and faithfulness in RAG.

## Full Text


<!-- PDF content starts -->

When Context Misleads: Intent-Guided Decoding for
Robust Retrieval-Augmented Generation
Haolin Jin, Pengyue Yang, Huaming Chen
School of Electrical and Computer Engineering
The University of Sydney
Sydney, NSW, Australia
{haolin.jin, pengyue.yang, huaming.chen}@sydney.edu.au
Abstract—Retrieval-augmented generation (RAG) improves
large language models by grounding generation in external
evidence, but it also introduces a source trust problem: retrieved
context may be useful, irrelevant, or even misleading. Existing
RAG systems often apply a fixed trust policy toward retrieved
evidence, which can either over-trust incorrect context or underuse
context when the user explicitly asks for context-following
behavior. Therefore, we propose Intent-Guided Decoding (IGD),
a framework that arbitrates between retrieved context and
parametric memory according to user intent. IGD uses answer-
level filtering and token-level correction to steer the final decoding
trajectory between retrieved context and parametric memory. We
evaluate IGD on three faithful QA benchmarks and three factual-
conflict benchmarks across five LLMs, IGD substantially improves
factual recovery, achieving gains of up to 65.4 percentage points on
factual-conflict benchmarks over Direct RAG, while preserving or
improving strict context-following behavior, this findings highlight
the importance of balancing factuality and faithfulness in RAG.
Index Terms—Retrieval-Augmented Generation, Large Lan-
guage Models, Factuality, Faithfulness
I. INTRODUCTION
Retrieval-augmented generation (RAG) has become a widely
adopted paradigm for connecting large language models
(LLMs) with external knowledge. By retrieving relevant evi-
dence at inference time, retrieval-augmented generation extends
the static parametric knowledge of LLMs with updatable non-
parametric memory [1]. This design improves performance
on knowledge-intensive tasks such as open-domain question
answering, where models require access to facts beyond their
internal parametric knowledge [2]–[4].
However, the effectiveness of RAG depends not only on
whether useful evidence is retrieved, but also on whether the
model can ground the generation in the retrieved evidence
appropriately. In practice, RAG systems may still generate
statements that are unsupported by, or even contradictory to, the
provided context. RAGTruth systematically annotates nearly
18K RAG responses, and finds that non-factual statements
remain common under standard RAG pipelines [5]. On the other
hand, retrieval introduces a source-trust problem. Standard RAG
pipelines often implicitly treat retrieved context as authoritative,
which content may be false, adversarially corrupted, or incon-
sistent with stable world knowledge [6]. FaithEval formalizes
this issue as contextualfaithfulnessevaluation, and constructs
4.9K high-quality examples across unanswerable, inconsistent,and counterfactual settings, showing that even strong contem-
porary LLMs often fail to maintain an appropriate degree
of faithfulness to given context [7]. More broadly, situated
faithfulness argues that models should dynamically calibrate
their trust in external context against internal knowledge, rather
than blindly following either source [8]. These findings suggest
that a central challenge in RAG is not only how to retrieve
useful evidence, but also how to decide whether the retrieved
evidence should dominate generation [9], [10].
This issue exposes a fundamental trade-off betweenfaithful-
nessandfactualityin RAG generation. In some settings, the
user explicitly expects the model to follow the provided context,
for example when answering questions about a document,
policy, or passage [11]. In other settings, the user expects the
model to use retrieved context only as auxiliary evidence and
remain skeptical when that context is misleading or conflicts
with world knowledge [9], [10]. A fixed RAG behavior is
therefore insufficient: always trusting context may improve
contextual faithfulness but can harm factual correctness under
misleading retrieval. Conversely, always discounting retrieved
context may recover factuality against corrupted evidence but
undermine user intent in strict context-following scenarios
[12]. Practical RAG systems therefore require an intent-aware
mechanism that can follow context when the user asks for
contextual grounding, while resisting context when the user
seeks factual robustness against unreliable retrieval.
To address this challenge, we argue that RAG requires not
only retrieval and grounding with external evidence, but also
an explicit decoding-time arbitration mechanism that decides
when generation should be controlled by retrieved context
and when it should rely on parametric memory. In this work,
we proposeIntent-Guided Decoding(IGD), a decoding-time
framework to address both factuality and faithfulness in RAG.
In Figure 1, IGD decomposes generation into three conditional
branches: auser branchconditioned on the original user prompt,
acontext branchthat explicitly follows the retrieved context,
and amemory branchthat answers in a closed-book manner.
IGD then performs source arbitration at two granularities. First,
anAnswer-Level Memory Filterhandles high-confidence cases
by routing to a stable memory answer when the retrieved
context is likely misleading. Second, for the remaining cases,
we propose a novelToken-Level Correctionmechanism to adjust
the user branch decoding distribution only when intervention
arXiv:2608.16515v1  [cs.CL]  17 Aug 2026

Context branch
preview: 1968
top-K next-token distribution
0.78
0.040.18
Memory branch
preview: 1980
top-K next-token distribution
0.320.280.40Distributional Conflict
no intervention full interventionexampleSoft activation gate
1968 1980 other
1968 1980 other Entropy Confidence Calculation
Branch Confidence V alues
0.46 0.02Faithful example
SQuAD, truth prompt
When was the UMC founded?Question
Founded in 1968  by the union of the
Methodist Church and the Evangelical
United Brethren Church.Correct retrieved context
Direct RAG Answer: 1968
Factual-conflict example
NQ-Swap, truth prompt
Question
Dan Stevens  as Agent Malcolm Turner /
Big MommaMisleading retrieved context
Direct RAG Answer: Dan StevensWho plays Big Momma in Big Momma's
House?Context branch
preview: Dan Stevens
top-K next-token distribution
0.76
0.050.19
Memory branch
preview: Martin Lawrence
top-K next-token distribution
0.040.84
0.12Dan Martin other
Dan Martin otherFaithful case:
0.39 0.52Factual case:
Context Preference Calculation
Faithful case: Factual case:
context preference = 0.91 context preference = 0.24  
memory pull context pullscale the selected pull
from snippet reliability
from memory reliabilityReliability Scaling
Decoding Shift
shifts decoding toward context
shifts decoding toward memory
Final Output
Faithful
1968Factual
Martin Lawrence
Token-Level Correction Branch DecodingFig. 1: The figure illustrates two truth mode examples: a faithful QA case where the retrieved context correctly supports the
answer, and a factual-conflict case where the retrieved context is misleading. For each case, IGD first obtains branch-level
previews and top- Ktoken distributions, then applies token-level correction only when the context and memory branches exhibit
distributional conflict. The correction direction is determined by entropy-based branch confidence and the prompt-mode prior,
while the intervention strength is scaled by the reliability of the selected source.
is needed. Specifically, we leverage an activation gate to detect
context-memory conflict, a confidence measurement to decide
whether generation should move toward context or memory, and
reliability scaling to determine the strength of the intervention.
As a result, IGD pulls decoding toward memory when retrieved
context is unreliable, while still pulling decoding toward context
when the user explicitly requests faithful context following.
Our method IGD introduces a novel decoding-time control
layer for RAG, which shifts the focus from retrieving better
evidence to arbitrating which knowledge source should steer
generation. Unlike retrieval-centric methods that improve
what or when to retrieve [13]–[15], IGD assumes retrieval
has already occurred and addresses downstream conflict
between retrieved context and parametric memory. Also,
unlike context-faithfulness alignment methods that generally
encourage stronger adherence to retrieved evidence [16], [17],
IGD performs bidirectional trust calibration. It adapts the source
preference according to user intent. Moreover, unlike prompt-
based confidence reasoning or multi-agent evidence aggregation,
IGD performs lightweight logit arbitration during generation.
Its token-level correction is activated only when the context and
memory branches exhibit distributional conflict, enabling local,
conservative and intent-aware intervention without retraining
the model or replacing the retrieval pipeline.
Our contributions are summarized as follows. First, we
formulate the factuality and faithfulness trade-off in RAG
as anintent-conditioned source arbitrationproblem, where
the appropriate trust policy depends on whether the user
requests truth seeking robustness or strict context following.
Second, we introduce IGD, a decoding-time framework thatcombines answer-level memory filtering with conservative
token-level correction over user, context, and memory branches.
Third, we evaluated and analysis the performance of IGD on
three faithful benchmark settings and three factual-conflict
benchmark settings. Fourth, we compare IGD against Direct
RAG, confidence reasoning baselines such as SCR/RCR,
and a multi-agent baseline in strict mode, and further test
whether IGD preserves context-following behavior when the
user explicitly asks to follow the provided context. Lastly, we
release the replication package in 1.
II. RELATEDWORK
Retrieval-augmented generation combines parametric lan-
guage models (LM) with external evidence to improve knowl-
edge intensive generation [1], [18]. Early and representative
RAG systems differ in how they retrieve, encode, and consume
evidence. Fusion-in-Decoder conditions generation on multiple
retrieved passages and fuses evidence inside the decoder [4].
Atlas jointly trains a retrieval-augmented model for few-shot
knowledge intensive learning [19]. In-Context RALM shows
that LMs can benefit from retrieved documents inserted directly
into the prompt [20]. Recent work has made retrieval more
adaptive, with most remaining retrieval-centric by improving
the retriever or training retrieval-aware generators. FLARE re-
trieves during generation based on predicted future content [13],
Self-RAG trains models to retrieve and critique the outputs
using reflection tokens [14], and CRAG evaluates retrieval
quality and corrects low-confidence retrieved evidence [15].
Other works study hallucination, contextual faithfulness, and
knowledge conflict in RAG. RAGTruth provides a corpus to

analyze hallucinations in RAG and shows that retrieved evi-
dence does not eliminate unsupported generation [5]. FaithEval
evaluates whether models remain faithful under unanswerable,
inconsistent, and counterfactual contexts [7]. ClashEval stud-
ies conflicts between an LLM’s internal prior and external
evidence, showing that models can adopt incorrect retrieved
content even when their internal knowledge is correct [6].
Context-DPO improves context faithfulness through prefer-
ence optimization [16], while FaithfulRAG models fact-level
conflicts between retrieved context and parametric knowledge
to improve context-faithful generation [17]. Work on situated
faithfulness further argues that LLMs should calibrate trust
in external context based on both contextual evidence and
internal confidence [8]. Our method shares this motivation of
trust calibration, but targets a different control point. Instead
of uniformly enforcing stronger adherence to retrieved context,
IGD performs intent-aware source arbitration directly in the
decoding distribution, allowing the model to favor either context
or memory according to the prompt mode.
III. METHOD
Intent-Guided Decoding (IGD) addresses context memory
conflict by separating source arbitration into two decision granu-
larities. Our method first applies an answer-level memory filter,
which performs hard replacement only for high-confidence
cases, all remaining examples are handled by token-level
correction, which continuously adjusts the decoding distribution
around the original user prompt. Figure 1 illustrates this token-
level stage: IGD compares the context and memory branch
distributions to activate correction only under meaningful
conflict, estimates which branch is more reliable through
entropy based confidence and source specific reliability, and
then applies a signed logit correction that shifts the user branch
distribution toward either the context or memory source.
A. Conditioned Branches
At decoding step t, let puser,t(v)denote the next token
distribution induced by the original user prompt for candidate
token v. In a standard RAG prompt, this distribution already
reflects the interaction between the user instruction, the
question, and the retrieved context. IGD augments this original
branch with two conditional branches:
•pctx,t(v): acontext-following branchexplicitly instructed
to answer according to the supplied context.
•pmem,t (v): amemory branchinstructed to answer in a
closed-book manner, independent of the supplied context.
The user branch is the unmodified RAG branch: it conditions
on the original instruction, question, and retrieved context. In
contrast, the context branch is a deliberately source specific
branch, we instantiate it with a support snippet selected from
the retrieved context, rather than the full context, so that pctx,t
represents the local evidence most relevant to the question. The
support snippet is selected using a lightweight IDF-based lexical
localizer over context segments, following the classical term-
specificity weighting idea of inverse document frequency [21].The context branch therefore estimates what the model
would generate under a strict context-following prompt, while
the memory branch estimates what the model’s parametric
knowledge supports when the retrieved context is removed.
To reduce prompt sensitivity in closed-book decoding, we
implement the memory branch as an ensemble of Mprompt
variants, following the common use of ensembling to stabilize
model predictions [22]. In our implementation, M= 3 ,
corresponding to a base memory prompt, an explicitly closed-
book prompt, and a best-guess memory prompt:
pmem,t(v) =1
MMX
i=1p(i)
mem,t(v).(1)
B. Answer-Level Memory Filter
The first stage of IGD performs answer-level filtering before
decoding, particularly for high-confidence cases in which the
memory branch assigns substantially higher likelihood to the
memory answer than to the context answer, while the original
user branch does not prefer the context answer over the memory
answer. Let ˆyctxdenote a short preview answer generated by
the context branch, and let ˆymem denote the memory preview
selected from the memory ensemble. For a candidate answer
y= (y 1, . . . , y L)and branch b, we define the length-normalized
answer likelihood, and for the memory ensemble we average
this likelihood across memory views:
ℓb(y) =1
LLX
ℓ=1logp b(yℓ|y<ℓ),
ℓmem-ens (y) =1
MMX
i=1ℓ(i)
mem(y).(2)
The filter evaluates the memory candidate against the context
candidate under two distributions:
mM=ℓ mem-ens (ˆymem)−ℓ mem-ens (ˆyctx),(3)
mU=ℓ user(ˆymem)−ℓ user(ˆyctx).(4)
Here, mMmeasures whether the memory branch itself strongly
prefers the memory candidate, while mUchecks whether the
same candidate remains compatible with the full user prompt.
This second condition prevents the filter from overriding the
user instruction with a memory answer that the original prompt
distribution strongly disfavors. Since this stage performs hard
replacement, IGD uses a conservative dominance test rather
than an averaged routing score:
Dmem= min(m M−logρ M, mU−logρ U),(5)
where ρMandρUare likelihood-ratio thresholds. Since
mMandmUare differences of length-normalized log likeli-
hoods, exp(m M)andexp(m U)can be interpreted as length-

normalized candidate-level likelihood ratios. IGD routes di-
rectly to the memory preview only if
valid mem= 1,
ˆyctx̸≡ˆymem,
Amem≥0.67,
Dmem≥0.(6)
When these conditions hold, IGD sets ˆyfinal= ˆymem; otherwise,
decoding falls back to token-level correction. We use ρM= 3.0
andρU= 1.0 by default, requiring the memory branch to prefer
the memory candidate by at least a factor of three while the
user branch must not prefer the context candidate.
C. Token-Level Correction
IGD performs token-level source arbitration around the
original user distribution, the goal is not to replace the
distribution, but to gently adjust it when the context and
memory branches disagree. This follows the general intuition
of decoding time control, where generation is steered by
modifying next token scores rather than updating model
parameters [23]–[25].
pfinal,t(v)∝p user,t(v)pctx,t(v)
pmem,t(v)λt
(7)
The scalar λtcontrols both the direction and magnitude of the
intervention, positive values favor the context branch, negative
values favor the memory branch, and values near zero leave
decoding close to the original user distribution. IGD computes
λtin three steps:activation,direction, andreliability scaling.
1) Activation:IGD should intervene only when context and
memory provide meaningfully different next token distribution.
We therefore measure distributional conflict at step tbetween
the context and memory branches using Jensen-Shannon
divergence [26]:
δt= JSD 
pctx,t∥pmem,t
(8)
For efficiency, JSD(·∥·) is computed over a renormalized top-
Ktoken set rather than the full vocabulary we use K= 16 .
The conflict score is converted into a soft activation gate:
at=
clipδt−τ low
τhigh−τ low,0,1γ
(9)
with τlow= 0.10 ,τhigh = 0.35 , and γ= 2.0 . This gate
suppresses correction when the two branches already agree and
gradually activates intervention as source conflict increases.
2) Confidence Direction:Conditioned on activation, IGD
determines which source should be favored. We combine two
signals: an instruction mode and branch level confidence, the
instruction encodes the user’s intended trust policy or hard
context-follow policy, while the branch confidence estimates
how concentrated each branch’s next token distribution is. We
use entropy-based confidence, which is a standard proxy for
predictive uncertainty and calibration behavior in neural models,
for each branchb∈ {ctx,mem}, we define:
rb,t= 1−H(topK(p b,t))
logK(10)where H(·) is the entropy of the renormalized top- Kdistribu-
tion, and lower entropy corresponds to higher confidence. Let
dmode∈(0,1) denote the prior probability of trusting context
under the current instruction mode, we use dstrict= 0.9 , and
dtruth= 0.3. The resulting context preference is:
πctx,t=σ(logit(d mode) + log(r ctx,t)−log(r mem,t))(11)
and the corresponding signed base coefficient is:
λbase,t =λmaxat 
2πctx,t−1
(12)
This parameterization induces a intent consistent intervention
direction: strict context following prompts start with a strong
context prior, truth seeking prompts start with a memory prior,
and the final direction can still change when one branch is
substantially more confident than the other.
3) Reliability Scaling:The base coefficient determines the
direction of intervention, IGD then rescales its magnitude
by the reliability of the source being favored. This prevents
confident but poorly supported context from exerting excessive
influence, and similarly prevents unstable memory predictions
from overriding context.
a) Context reliability:When λbase,t >0, IGD favors
the context branch and estimates whether the selected snippet
genuinely supports the context preview. Let ˆyctxbe the context
preview andx ctxbe the selected support snippet, we define:
sctx= 0.7·Supp(ˆy ctx, xctx) + 0.3·Qual(x ctx) (13)
where Supp(·,·) measures local textual support for the preview
answer and Qual(·) captures the evidential quality of selected
snippet. Let valid ctx∈ {0,1} indicate whether the context
preview is a valid short-form answer, the context reliability is:
qctx=qmin
ctx+ (1−qmin
ctx) valid ctxsctx (14)
b) Memory reliability:When λbase,t <0, IGD favors the
memory branch and scales the intervention by the stability of
the closed-book memory previews.
qmem=qmin
mem+ (1−qmin
mem) valid memAmem (15)
We use qmin
ctx= 0.35 andqmin
mem= 0.55 . Here, Amemis computed
as the average pairwise answer consistency among the M
closed-book memory previews:
Amem=2
M(M−1)X
1≤i<j≤M
AnsCons
ˆy(i)
mem,ˆy(j)
mem(16)
where AnsCons(·,·)∈[0,1] measures whether two short-form
memory previews are equivalent answers.
c) Final coefficient:The final token-level coefficient is:
λt=

λbase,tqctx, λ base,t>0,
λbase,tqmem, λ base,t<0,
0, λ base,t = 0.(17)

Thus, IGD intervenes only under source conflict, chooses the
direction according to user intent and branch confidence, and
scales the update by the reliability of the favored source.1
IV. EXPERIMENTSETUP
A. Evaluation Design
We evaluate IGD in retrieval-augmented short form answer-
ing questions. Each example consists of a question, a document-
tagged context, and a prompt mode. We consider two prompt
modes that represent different user intents. InSTRICTmode,
the model is instructed toanswer strictly according to the
provided context. InTRUTHmode, the model is instructed
totreat the context as potentially noisy or misleading and
to prioritize the factually correct answer when context
and world knowledge conflict. The two modes use the same
question and context, only the user instruction changes. Direct
RAG and IGD are evaluated under identical prompts, to isolate
the effect of source arbitration during the decoding phase.
We organize the evaluation into two groups. Thefaithful
group contains standard QA settings where the context supports
the gold answer, to test whether IGD preserves the benefits
of retrieval. Thefactual-conflictgroup contains examples
where the context answer conflicts with the world answer,
to test whether IGD follows the context inSTRICTmode while
recovering the world-knowledge answer inTRUTHmode.
B. Benchmarks
We evaluate on six QA benchmarks, grouped into three faith-
ful QA benchmarks and three factual-conflict QA benchmarks,
each contains 500 samples. The faithful group includes KILT-
NQ, TriviaQA, and SQuAD. KILT-NQ is based on Natural
Questions and the KILT Wikipedia corpus [27], [28]; TriviaQA
contains trivia style open-domain questions with answer aliases
and Wikipedia/web evidence [29]; and SQuAD is a paragraph
level reading comprehension dataset built from Wikipedia [30].
For KILT-NQ and TriviaQA, we construct multi-document
contexts by selecting an answer bearing passage and adding
distractors; for SQuAD, we use the original paragraph as
support and add distractor paragraphs from other examples.
The factual-conflict group includes ConflictBank, NQ-Swap,
and CounterFact. ConflictBank contains Wikidata-derived
knowledge conflicts with generated misleading evidence [31];
NQ-Swap is constructed from Natural Questions/MRQA by
replacing the original answer in the context with a substituted
answer [27], [32]; and CounterFact is adapted from the factual
association editing benchmark in ROME [33]. In these factual-
conflict benchmarks, the original answer is treated as the world
answer, while the misleading or substituted answer supported
by the context is treated as the context answer. We apply alias
cleaning and semantic filtering to remove ambiguous cases
where the world and context answers are equivalent.
1Github RepositoryC. Models and Baselines
We evaluate five instruction-tuned LLMs: Qwen3-32B [34],
Qwen2.5-14B-Instruct [35], Llama-3-8B-Instruct [36], Mistral-
7B-Instruct-v0.3 [37], and Phi-4 14B [38]. These models cover
different families and scales, allowing us to test whether
IGD generalizes across models with different parametric
knowledge and instruction-following behavior. Experiments
were conducted on local GPU servers equipped with NVIDIA
RTX A6000 GPU and two NVIDIA GeForce RTX 5090 GPUs.
We compare IGD with six baselines.Closed-book Q-only
receives only the question and no context, it is a diagnostic
reference for parametric knowledge rather than a context
using RAG method.Direct RAGreceives the same question,
context, and prompt-mode instruction as IGD, but generates
directly from the original prompt without source arbitration.
We also include confidence reasoning baselines from situated
faithfulness work [8]:ExplicitSCR, which asks the model
to explicitly reason about whether to trust the context or its
internal knowledge, and three rule-based confidence reasoning
variants,RCR-InternalEval,RCR-ContextEval, andRCR-
InternalConf, which extract confidence or evaluation signals
for internal and context-based answers and then select a final
answer according to predefined rules. In addition, we compare
withMADAM-RAG, a multi-agent RAG framework designed
to aggregate and resolve conflicting retrieved evidence [9].
D. Evaluation Metrics
We evaluate the answer accuracy using normalized alias
matching, following standard QA evaluation practice [29],
[30]. For faithful benchmarks, predictions are always matched
against gold answers. For factual-conflict benchmarks, the
target depends on the prompt mode:STRICTmode reports
context accuracy, whileTRUTHmode reportsworld accuracy.
Our main aggregate metric is theIntent-Aligned Score(IA),
defined as the macro-average over the faithful and factual-
conflict benchmark groups:
IAm=P
d∈D faithAccgold
d+P
d∈D confAccm
d
|Dfaith|+|D conf|.(18)
Here, m∈ {STRICT,TRUTH} . For d∈ D conf,Accm
ddenotes
context accuracy inSTRICTmode and world accuracy inTRUTH
mode, IA therefore measures whether the model behavior
matches the user’s intended trust policy rather than rewarding
a single fixed preference for either context or memory.
V. RESULTS
A. Closed-Book Performance Reveals the Recoverable Para-
metric Signal
We first evaluate each model in the closed-book Q-only
setting, where the model receives only the question without
any retrieved context. It is important to interpret IGD because
factual recovery from a misleading context is only possible
when the underlying model can access the correct answer from
its parametric memory, if the closed-book model cannot recover
the world answer, the upper bound of any memory oriented
correction is naturally limited [39], [40].

TABLE I: Truth-mode comparison across three faithful benchmarks and three factual-conflict benchmarks. IA is the macro-
average over the six benchmark columns. Bold marks the best context-using method in each column, and underline marks the
second-best. IGD also reports deltas against Direct RAG under the same model and benchmark.
Model Method IAFaithful benchmarks Factual-conflict benchmarks
KILT-NQ TriviaQA SQuAD ConflictBank NQ-Swap CounterFact
Qwen3-32B Closed-book Q-only†56.8 35.4 62.0 32.4 56.4 70.0 84.6
Direct RAG 51.0 74.8 88.8 90.4 15.0 27.0 10.0
ExplicitSCR 12.9 16.0 18.4 15.8 4.0 3.0 20.2
RCR-InternalEval 62.1 58.2 74.6 47.8 47.8 70.5 73.6
RCR-ContextEval 53.6 68.8 80.0 83.2 21.0 37.0 31.4
RCR-InternalConf 59.9 50.0 69.0 41.8 46.677.574.8
MADAM-RAG 65.0 62.6 82.4 66.2 44.0 61.0 73.8
IGD74.0↑23.0 72.2↓2.684.8 ↓4.089.8 ↓0.6 50.2↑35.271.5↑44.575.4↑65.4
Qwen2.5-14B Closed-book Q-only†63.3 37.2 65.6 31.2 61.2 91.5 92.8
Direct RAG 58.1 74.2 90.2 94.2 12.0 42.0 36.0
ExplicitSCR 10.8 13.0 16.4 14.2 4.8 9.5 6.6
RCR-InternalEval 63.5 64.8 80.6 65.0 39.2 57.0 74.6
RCR-ContextEval 59.6 57.8 76.4 69.8 26.4 57.5 69.8
RCR-InternalConf 65.0 54.8 75.6 49.2 55.0 67.0 88.2
MADAM-RAG 59.7 57.4 71.8 61.0 38.2 63.0 66.6
IGD77.6↑19.5 70.8↓3.485.6 ↓4.687.8 ↓6.4 55.4↑43.477.0↑35.089.4↑53.4
Llama-3-8B Closed-book Q-only†55.7 37.8 69.8 26.8 55.6 68.5 75.8
Direct RAG 59.6 73.2 88.4 89.4 27.6 49.0 30.2
ExplicitSCR 11.2 9.8 8.8 8.8 2.4 16.0 21.2
RCR-InternalEval 59.7 59.8 81.0 58.8 41.0 55.0 62.8
RCR-ContextEval 53.3 42.6 68.6 33.2 42.2 61.5 71.6
RCR-InternalConf 52.1 45.0 75.2 40.6 32.2 56.0 63.8
MADAM-RAG 45.9 51.6 64.6 60.4 28.0 40.0 30.6
IGD69.3↑9.6 68.4↓4.885.8 ↓2.685.0 ↓4.4 46.8↑19.262.0↑13.067.6 ↑37.4
Mistral-7B Closed-book Q-only†51.8 39.2 66.2 24.2 39.4 68.5 73.0
Direct RAG 42.7 61.4 77.4 77.0 11.4 17.5 11.4
ExplicitSCR 26.6 23.2 32.0 22.6 17.6 21.0 43.2
RCR-InternalEval 45.7 50.0 72.8 41.4 26.2 44.0 40.0
RCR-ContextEval 47.0 57.0 76.4 72.8 10.6 22.5 42.4
RCR-InternalConf 50.1 49.6 74.2 38.2 36.6 40.561.4
MADAM-RAG 44.4 50.6 59.6 58.6 19.252.525.6
IGD54.2↑11.5 58.4↓3.076.0 ↓1.473.0 ↓4.0 37.0↑25.643.0↑25.547.8 ↑36.4
Phi-4 14B Closed-book Q-only†51.8 32.6 66.6 28.2 41.8 60.0 81.4
Direct RAG 56.9 71.0 87.2 85.8 15.8 35.5 46.2
ExplicitSCR 6.1 11.8 14.6 3.0 2.6 2.0 2.8
RCR-InternalEval 55.8 58.8 77.6 49.2 30.8 47.5 70.8
RCR-ContextEval 57.5 65.8 81.8 66.8 21.6 45.5 63.4
RCR-InternalConf 55.7 54.0 76.2 45.8 32.8 53.0 72.4
MADAM-RAG 61.1 61.6 79.8 59.0 32.2 57.0 77.2
IGD67.1↑10.2 63.8↓7.285.8 ↓1.482.2 ↓3.6 34.8↑19.059.0↑23.577.0↑30.8
The closed-book results in Table I show that the difficulty
of the six benchmarks differs substantially. Among the faithful
benchmarks, closed-book accuracy is generally low on KILT-
NQ and SQuAD, indicating that these datasets rely heavily on
the provided context [27], [28], [30]. Across the five models,
KILT-NQ closed-book accuracy remains around the mid 30%
range, and SQuAD is even lower, mostly around 25-32%. This
is expected: KILT-NQ is derived from open-domain queries
grounded in Wikipedia evidence, while SQuAD questions are
written against specific paragraphs and often require access to
the original passage. In contrast, TriviaQA is substantially easier
in the closed-book setting, with all models reaching above 60%,
suggesting that many trivia-style answers are already encoded
in the models’ parameters.For the factual-conflict benchmarks, closed-book perfor-
mance is much higher overall, especially on CounterFact and
NQ-Swap. CounterFact is the easiest factual-conflict benchmark
under closed-book evaluation: all models achieve between
73.0% and 92.8%, suggesting that many subject relation facts
remain accessible from parametric memory. NQ-Swap is also
highly recoverable for several models, with Qwen2.5-14B
reaching 91.5% and Phi-4 reaching 60.0%. ConflictBank is
more challenging, with closed-book accuracy ranging from
39.4% to 61.2%. Overall, Qwen2.5-14B-Instruct shows the
strongest closed-book IA score, while Qwen3-32B also exhibits
strong recoverability on the factual-conflict benchmarks. These
diagnostics suggest that IGD is evaluated in a meaningful
regime: the factual-conflict examples are difficult for Direct

ConflictBankNQ-SwapCounterFact020406080100PRR (%)75.491.987.7Qwen3-32B
ConflictBankNQ-SwapCounterFact88.2
70.794.0Qwen2.5-14B
ConflictBankNQ-SwapCounterFact68.666.782.0Llama-3-8B
ConflictBankNQ-SwapCounterFact55.7
50.059.1Mistral-7B
ConflictBankNQ-SwapCounterFact73.195.9
87.5Phi-4 14BParametric Recovery Rate Across ModelsFig. 2: Parametric recovery rate (PRR) of IGD on factual-conflict benchmarks, the dashed line denotes full recovery relative to
the closed-book reference.
RAG because the context is misleading, but many of their
correct answers are still recoverable from model memory.
B. Truth Prompting Alone Does Not Calibrate Context Trust
Table I shows that explicit truth seeking instructions alone
are insufficient to prevent RAG models from being misled
by incorrect context. In truth mode, the prompt explicitly
instructs the model to prioritize the correct answer and to
treat the provided context with skepticism when it may be
misleading [6], [8]. Nevertheless, Direct RAG performs poorly
on factual-conflict benchmarks across all models, this failure
is most visible when comparing Direct RAG with the closed-
book diagnostic. For example, Qwen2.5-14B answers NQ-Swap
correctly 91.5% of the time in the closed-book setting, but drops
to 42.0% once the misleading context is provided. The same
model reaches 92.8% closed-book accuracy on CounterFact,
but Direct RAG falls to 36.0%.
The additional baselines further clarify that this problem
is not fully solved by answer-level reasoning or confidence-
based source selection. ExplicitSCR and RCR variants are
designed to improve situated faithfulness by reasoning over, or
extracting confidence from, internal and context-based answers
[8], while MADAM-RAG uses multi-agent aggregation to
handle conflicting retrieved evidence [9]. These methods often
improve over Direct RAG on factual-conflict benchmarks,
confirming that explicit source comparison is useful. However,
their improvements are uneven: some variants recover more
world answers but substantially reduce faithful QA accuracy,
while others preserve context grounded performance but remain
weak under misleading context. In contrast, IGD achieves the
best IA score for every evaluated backbone in Table I. This
suggests that the key difficulty is not only detecting whether
context or memory is correct at the answer level, but also
controlling how strongly each source should influence decoding.
the desired behavior is not to simply suppress context, but to
preserve context grounded accuracy on faithful examples while
resisting misleading evidence in factual conflict cases.
C. IGD Balances Factuality and Faithfulness
a) Truth mode: recovering world answers under mislead-
ing context.:Table I shows that IGD substantially improvestruth mode factual recovery while largely preserving faithful QA
accuracy. Compared with Direct RAG, IGD improves the IA
score for every model, with gains of 23.0 points for Qwen3-32B,
19.5 for Qwen2.5-14B, 9.6 for Llama-3-8B, 11.5 for Mistral-
7B, and 10.2 for Phi-4. These gains are driven primarily by
large improvements on factual-conflict benchmarks, where IGD
consistently shifts generation away from misleading context
and toward recoverable world knowledge.
The comparison with external baselines provides a more
informative view of this trade-off. Confidence-reasoning and
multi-agent baselines can be strong on individual factual-
conflict columns, but they often sacrifice accuracy on faithful
benchmarks. For instance, RCR-style methods may recover
more world-knowledge answers on a particular conflict set, but
their faithful accuracy drops sharply because they rely on a
global answer-level source decision. Similarly, MADAM-RAG
improves some conflict cases, but its aggregation strategy does
not consistently preserve original context-grounded behavior.
The factual gains are large and systematic. Across all models,
IGD improves every factual-conflict benchmark over Direct
RAG, with the largest gain reaching 65.4 percentage points on
Qwen3-32B CounterFact. Importantly, these gains do not come
from discarding retrieval altogether. On faithful benchmarks,
IGD remains close to Direct RAG, with moderate drops that
are much smaller than the factual-conflict gains. This pattern
suggests that token-level correction is more conservative than
answer-level rerouting alone: it can weaken misleading context
when necessary, while still preserving useful retrieved evidence
when the context is correct.
To better quantify how much of the recoverable parametric
signal is restored, Figure 2 reports the parametric recovery rate
(PRR), defined as the fraction of the gap between Direct RAG
and closed-book Q-only that IGD recovers on factual-conflict
benchmarks:
PRR =AccIGD−Acc Direct
AccClosed−Acc Direct.(19)
The PRR results show that IGD recovers a large portion
of the available parametric knowledge for stronger models.
For example, Qwen3-32B recovers most of performance gap
on NQ-Swap and CounterFact, whereas Mistral-7B exhibits

TABLE II: Strict context following prompt. Factual-conflict benchmarks are evaluated by context accuracy, testing whether IGD
preserves user intent when the user explicitly asks to follow the provided context.
Model Method IAFaithful benchmarks Factual-conflict benchmarks
KILT-NQ TriviaQA SQuAD ConflictBank NQ-Swap CounterFact
Qwen3-32BDirect RAG 83.0 72.6 89.4 91.0 83.2 63.0 99.0
IGD 84.3 69.4↓3.286.8↓2.690.0↓1.0 87.2↑4.0 72.5↑9.5 99.6↑0.6
Qwen2.5-14BDirect RAG 84.2 74.6 89.2 94.0 84.4 63.0 100.0
IGD 88.1 76.4↑1.888.6↓0.693.6↓0.4 89.2↑4.8 81.0↑18.0 100.0↔0.0
Llama-3-8BDirect RAG 80.8 73.0 88.0 88.4 85.4 50.0 99.8
IGD 82.0 71.0↓2.086.4↓1.687.6↓0.8 88.0↑2.6 59.0↑9.0 100.0↑0.2
Mistral-7BDirect RAG 77.1 64.0 79.6 80.8 82.6 59.5 96.0
IGD 77.6 60.6↓3.479.2↓0.4 82.0↑1.2 84.4↑1.8 62.5↑3.0 97.0↑1.0
Phi-4 14BDirect RAG 81.3 73.2 84.0 86.6 82.4 63.0 98.4
IGD 82.5 71.6↓1.683.2↓0.8 87.0↑0.4 85.6↑3.2 68.5↑5.5 99.2↑0.8
weaker recovery. This indicates that IGD’s factual correction is
bounded by both the model’s accessible parametric knowledge
and the reliability of the memory branch. The pattern in
Figure 2 therefore reinforces the interpretation that IGD is
not hallucinating new facts, but recovering parametric answers
that are otherwise suppressed by misleading context.
b) Strict mode: preserving context-following intent:
Table II evaluates the complementary setting: the user explicitly
asks the model to follow the provided context. In this mode,
factual-conflict benchmarks are evaluated by context accuracy
rather than world accuracy, because the desired behavior is
to follow the supplied evidence even when it conflicts with
world knowledge. Since the external baselines in Table I are
designed primarily for truth-seeking or correctness-oriented
source selection, we keep the strict-mode comparison focused
on Direct RAG and IGD.
The results show that IGD preserves, and in many cases im-
proves, strict context-following behavior. These strict prompt’s
mode results are central to the claim that IGD balances
factuality and faithfulness rather than optimizing only one side
of the trade-off [12]. A method that simply discounts retrieved
context would improve truth mode factual accuracy but damage
strict context following [16], [17]. In contrast, IGD uses the
prompt mode to determine the direction of intervention: under
truth-seeking prompts it can pull decoding toward memory,
while under strict context-following prompts it preserves or
strengthens context guided decoding.
VI. FACTUALITY ANDFAITHFULNESSTRADE-OFF
We further analyze how IGD balances factual recovery
and context faithfulness through component ablations and
hyperparameter sensitivity. This analysis is motivated by
the central tension in RAG, stronger reliance on parametric
memory can recover world answers but may undermine strict
context following [7], [8]. We conduct ablations on Qwen2.5-
14B-Instruct over the same six-benchmark suite used in the
main experiments, reporting macro averaged results over
the three faithful benchmarks and the three factual-conflictTABLE III: Core ablations of IGD on Qwen2.5-14B. Values are
macro-averaged percentages over the three faithful benchmarks
and the three factual-conflict benchmarks. Shaded cells mark
notable changes relative to IGD full: light red indicates drops
of roughly 3–10 percentage points, darker red indicates drops
larger than 10 points, and light green indicates improvements
over IGD full.
Variant Removed / changed componentStrict mode Truth mode
Faithful
GoldFactual
CtxFaithful
GoldFactual
World
IGD full none 86.2 90.1 81.4 73.7
w/o Answer Filter answer-level memory filter 85.8 86.0 81.3 63.0
w/o Token Correction token-level correction 86.9 83.1 82.7 52.8
w/o Activation Gate JSD conflict gate 85.9 86.2 77.5 70.5
w/o Mode Prior instruction-mode prior 86.6 83.4 82.7 63.1
w/o Reliability Gateq ctx/qmem scaling 85.0 85.5 76.1 68.6
Single Memory View memory ensemble 86.1 85.8 85.5 54.5
benchmarks. Table III ablates six core components of IGD: the
answer level memory filter, token-level correction, activation
gate, instruction-mode prior, reliability scaling, and memory
ensemble, we choose these ablations because each component
directly affects the trade-off rather than merely changing
implementation details.
Table III shows that IGD’s gains arise from coordinated
source arbitration rather than a single heuristic. Removing
token-level correction causes the largest degradation, reducing
truth-mode factual world accuracy from 73.7 to 52.8 and
strict-mode factual context accuracy from 90.1 to 83.1. This
indicates that token-level correction is not merely a memory
side intervention, but the main mechanism for steering: it
pulls decoding toward memory under misleading context while
preserving context-following behavior when required. The
answer-level memory filter provides a complementary high
precision route for stable factual-conflict cases; removing it
drops factual world accuracy to 63.0 while leaving faithful accu-
racy nearly unchanged, suggesting that the filter is conservative
enough not to disrupt ordinary context-grounded QA. The
remaining ablations explain how IGD avoids over-intervention,

0.8 1.0 1.2 1.4
max
767880828486Faithful accuracy (%)
A. Context-faithful benchmarks
0.8 1.0 1.2 1.4
max
3040506070Factual world accuracy (%)
B. Factual-conflict benchmarksdtruth=0.1dtruth=0.2dtruth=0.3 default router Direct RAGFig. 3: Sensitivity of truth mode performance to token-level
steering hyperparameters on Qwen2.5-14B-Instruct.
Fig. 4: Answer distribution on factual-conflict benchmarks
under truth mode.
a single memory view sharply reduces factual recovery to
54.5, showing that memory side correction requires a stable
closed-book signal; removing the activation or reliability gate
degrades both faithful and factual columns, indicating that
intervention must be triggered only under meaningful conflict
and scaled by source reliability. Finally, removing the mode
prior lowers both truth-mode factual recovery and strict-mode
context accuracy, confirming that context and memory conflicts
cannot be resolved with a fixed source preference, whereas
the full IGD model provides a stable balance between factual
recovery and context faithfulness.
Figure 3 further exposes the limitation of using fixed global
steering hyperparameters. As λmax increases, factual world ac-
curacy on factual-conflict benchmarks generally improves, but
faithful accuracy on faithful benchmarks decreases. Similarly,
smaller dtruth values make the router more memory biased
and improve factual recovery, but they also make the model
more likely to deviate from correct retrieved evidence. This
creates a clear tuning trade-off: aggressive correction helps
resist misleading context, whereas conservative correction better
preserves faithful context use, this is a core limitation of the
current framework. Future work could reduce this limitation by
learning example adaptive intervention strengths, calibrating the
router with validation feedback, or estimating source reliability
at a finer granularity.
Beyond aggregate accuracy, we further analyze what models
output on factual-conflict benchmarks. Main results report
whether an answer matches the target, but they do not distin-
guish whether errors come from copying the misleading context,
refusing to answer, or producing an unrelated alternative. We
Faithful snippet Factual snippet020406080100Accuracy (%)83.0
79.099.0
87.0A. Snippet support accuracy
Faithful acc Factual world acc IAS020406080100Accuracy (%)74.5
57.566.072.5
56.564.5B. Downstream IGD performance
IDF
GPT-5 localizerFig. 5: Snippet localization accuracy versus downstream IGD
performance.
therefore categorize each output into four types:correct answer,
misleading-context answer,unknown/refusal, andother answer.
As shown in Figure 4, Direct RAG is dominated by misleading
context answers: 57.9% of its outputs follow the incorrect
context, while only 25.8% recover the correct world answer,
IGD reverses this pattern, increasing correct answers to 58.3%
and reducing misleading-context answers to 22.5%. At the same
time, IGD does not convert all misleading context answers
into correct answers, some mass moves into theother answer
category. We interpret this as a residual uncertainty effect
under source conflict: once decoding is pushed away from
the misleading context, the model may still fail to land on
the gold world answer if the memory branch is unstable, if
competing aliases are plausible, or if the evidence conflict
is not cleanly resolved. This is consistent with recent work
showing that RAG models remain vulnerable under conflicting
or noisy evidence and that models often generate plausible
incorrect answers rather than abstaining when uncertainty is
not explicitly rewarded [10], [41], [42]. Thus, IGD substantially
mitigates context over-trust, but factual-conflict resolution
remains imperfect when memory evidence is ambiguous or
insufficiently calibrated.
Figure 5 highlights a surprising mismatch between snippet
localization quality and downstream IGD performance. The
GPT-5 localizer substantially improves standalone snippet
support accuracy, increasing faithful snippet accuracy from
83.0% to 99.0% and factual snippet accuracy from 79.0% to
87.0%. Intuitively, one might expect a better support localizer to
improve IGD, since IGD uses snippet reliability to scale context-
side intervention. However, the downstream results show the
opposite: replacing the IDF localizer slightly reduces faithful
accuracy, factual world accuracy, and IAS. This suggests
that IGD does not only need a snippet that is semantically
supportive under human or LLM judgment, it needs a snippet
whose evidence is useful for the specific downstream generator
and reliability function. The IDF localizer may select shorter,
lexically concentrated spans that align better with answer-
bearing tokens and IGD’s support heuristics, while the GPT-
5 localizer may prefer semantically richer snippets that are
correct in isolation but less discriminative for logit-level source
arbitration [43], [44]. The result also aligns with evidence that
robust RAG depends on preserving fine-grained textual details,
not only high-level semantic support [45]. More broadly, LLM-
based scoring or localization can introduce its own judgment

biases and objective mismatch [46].
VII. CONCLUSION
We presented Intent-Guided Decoding (IGD), a decoding-
time framework for resolving the factuality and faithfulness
trade-off in retrieval-augmented generation. Rather than treating
retrieved context as either universally authoritative or inherently
suspect, IGD performs intent-conditioned source arbitration
across the user prompt, retrieved context, and parametric
memory. Experiments on faithful and factual-conflict bench-
marks show that IGD improves intent-aligned generation across
multiple LLMs, while ablations confirm that its gains arise
from coordinated routing, conflict activation, reliability scaling,
and memory stabilization.
REFERENCES
[1]P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin,et al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,” inAdvances in
Neural Information Processing Systems, vol. 33, pp. 9459–9474, 2020.
[2]Y . Hu and Y . Lu, “Rag and rau: A survey on retrieval-augmented language
model in natural language processing,”arXiv preprint arXiv:2404.19543,
2024.
[3]P. Jiang, S. Ouyang, Y . Jiao, M. Zhong, R. Tian, and J. Han, “Retrieval
and structuring augmented generation with large language models,”
inProceedings of the 31st ACM SIGKDD Conference on Knowledge
Discovery and Data Mining V . 2, pp. 6032–6042, 2025.
[4]G. Izacard and E. Grave, “Leveraging passage retrieval with generative
models for open domain question answering,” inProceedings of the 16th
Conference of the European Chapter of the Association for Computational
Linguistics, pp. 874–880, 2021.
[5]C. Niu, Y . Wu, J. Zhu, S. Xu, K. Shum, R. Zhong, J. Song, and T. Zhang,
“Ragtruth: A hallucination corpus for developing trustworthy retrieval-
augmented language models,”arXiv preprint arXiv:2401.00396, 2024.
[6]K. Wu, E. Wu, and J. Zou, “Clasheval: Quantifying the tug-of-war
between an llm’s internal prior and external evidence,”arXiv preprint
arXiv:2404.10198, 2024.
[7]Y . Ming, S. Purushwalkam, S. Pandit, Z. Ke, X.-P. Nguyen,et al.,
“Faitheval: Can your language model stay faithful to context, even if"
the moon is made of marshmallows",” inInternational Conference on
Learning Representations, vol. 2025, pp. 29430–29456, 2025.
[8]Y . Huang, S. Chen, H. Cai, and B. Dhingra, “Enhancing large language
models’ situated faithfulness to external contexts,”arXiv preprint
arXiv:2410.14675, 2024.
[9]H. Wang, A. Prasad, E. Stengel-Eskin, and M. Bansal, “Retrieval-
augmented generation with conflicting evidence,”arXiv preprint
arXiv:2504.13079, 2025.
[10] Z. Ge, Y . Wu, D. W. K. Chin, R. K.-W. Lee, and R. Cao, “Resolving
conflicting evidence in automated fact-checking: A study on retrieval-
augmented llms,”arXiv preprint arXiv:2505.17762, 2025.
[11] X.-P. Nguyen, S. Pandit, S. Purushwalkam, A. Xu, H. Chen, Y . Ming,
Z. Ke, S. Savarese, C. Xiong, and S. Joty, “Sfr-rag: Towards contextually
faithful llms,”arXiv preprint arXiv:2409.09916, 2024.
[12] E. Fadeeva, A. Rubashevskii, R. Vashurin, S. Dhuliawala, A. Shelmanov,
T. Baldwin, P. Nakov, M. Sachan, and M. Panov, “Faithfulness-aware
uncertainty quantification for fact-checking the output of retrieval
augmented generation,”arXiv preprint arXiv:2505.21072, 2025.
[13] Z. Jiang, F. Xu, L. Gao, Z. Sun, Q. Liu, J. Dwivedi-Yu, Y . Yang, J. Callan,
and G. Neubig, “Active retrieval augmented generation,” inProceedings
of the 2023 Conference on EMNLP, (Singapore), pp. 7969–7992, 2023.
[14] A. Asai, Z. Wu, Y . Wang, A. Sil, and H. Hajishirzi, “Self-rag: Learning
to retrieve, generate, and critique through self-reflection,” inThe Twelfth
International Conference on Learning Representations, 2024.
[15] S.-Q. Yan, J.-C. Gu, Y . Zhu, and Z.-H. Ling, “Corrective retrieval
augmented generation,”arXiv preprint arXiv:2401.15884, 2024.
[16] B. Bi, S. Huang, Y . Wang, T. Yang, Z. Zhang,et al., “Context-dpo:
Aligning language models for context-faithfulness,”arXiv preprint
arXiv:2412.15280, 2024.
[17] Q. Zhang, Z. Xiang, Y . Xiao, L. Wang, J. Li, X. Wang, and J. Su,
“Faithfulrag: Fact-level conflict modeling for context-faithful retrieval-
augmented generation,”arXiv preprint arXiv:2506.08938, 2025.[18] X. Zhang, J. Zhang, F. Mo, D. K. Chandra, Y .-Z. Chen, F. Xie, and
K. Liu, “Retrieval-augmented feature generation for domain-specific
classification,” in2025 IEEE International Conference on Data Mining
(ICDM), pp. 943–952, IEEE, 2025.
[19] G. Izacard, P. Lewis, M. Lomeli, L. Hosseini, F. Petroni,et al., “Atlas:
Few-shot learning with retrieval augmented language models,”arXiv
preprint arXiv:2208.03299, 2022.
[20] O. Ram, Y . Levine, I. Dalmedigos, D. Muhlgay, A. Shashua, K. Leyton-
Brown, and Y . Shoham, “In-context retrieval-augmented language models,”
arXiv preprint arXiv:2302.00083, 2023.
[21] K. Sparck Jones, “A statistical interpretation of term specificity and
its application in retrieval,”Journal of Documentation, vol. 28, no. 1,
pp. 11–21, 1972.
[22] T. G. Dietterich, “Ensemble methods in machine learning,” inMultiple
Classifier Systems, vol. 1857 ofLecture Notes in Computer Science,
pp. 1–15, Springer, 2000.
[23] B. Krause, A. D. Gotmare, B. McCann, N. S. Keskar, S. Joty, R. Socher,
and N. F. Rajani, “GeDi: Generative discriminator guided sequence
generation,” inFindings of the Association for Computational Linguistics:
EMNLP 2021, (Punta Cana, Dominican Republic), pp. 4929–4952, 2021.
[24] A. Liu, M. Sap, X. Lu, S. Swayamdipta, C. Bhagavatula, N. A. Smith,
and Y . Choi, “Dexperts: Decoding-time controlled text generation with
experts and anti-experts,” inProceedings of the 59th Annual Meeting of
the Association for Computational Linguistics and the 11th International
Joint Conference on Natural Language Processing, 2021.
[25] X. L. Li, A. Holtzman, D. Fried, P. Liang, J. Eisner, T. Hashimoto,
L. Zettlemoyer, and M. Lewis, “Contrastive decoding: Open-ended text
generation as optimization,”arXiv preprint arXiv:2210.15097, 2022.
[26] J. Lin, “Divergence measures based on the shannon entropy,”IEEE
Transactions on Information Theory, vol. 37, no. 1, pp. 145–151, 1991.
[27] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh,et al.,
“Natural questions: A benchmark for question answering research,”
Transactions of the Association for Computational Linguistics, vol. 7,
pp. 452–466, 2019.
[28] F. Petroni, A. Piktus, A. Fan, P. Lewis, M. Yazdani,et al., “Kilt: a
benchmark for knowledge intensive language tasks,” inProceedings of
the 2021 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies, (Online),
pp. 2523–2544, 2021.
[29] M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer, “Triviaqa: A large
scale distantly supervised challenge dataset for reading comprehension,”
inProceedings of the 55th Annual Meeting of the Association for
Computational Linguistics, pp. 1601–1611, 2017.
[30] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, “Squad: 100,000+
questions for machine comprehension of text,” inProceedings of the
2016 Conference on Empirical Methods in Natural Language Processing,
pp. 2383–2392, 2016.
[31] Z. Su, J. Zhang, X. Qu, T. Zhu, Y . Li, J. Sun, J. Li, M. Zhang, and
Y . Cheng, “Conflictbank: A benchmark for evaluating the influence of
knowledge conflicts in llm,”arXiv preprint arXiv:2408.12076, 2024.
[32] A. Fisch, A. Talmor, R. Jia, M. Seo, E. Choi, and D. Chen, “Mrqa 2019
shared task: Evaluating generalization in reading comprehension,” in
Proceedings of the 2nd Workshop on Machine Reading for Question
Answering, pp. 1–13, 2019.
[33] K. Meng, D. Bau, A. Andonian, and Y . Belinkov, “Locating and editing
factual associations in gpt,” inAdvances in Neural Information Processing
Systems, vol. 35, 2022.
[34] A. Yanget al., “Qwen3 technical report,”arXiv preprint
arXiv:2505.09388, 2025.
[35] A. Yanget al., “Qwen2.5 technical report,”arXiv preprint
arXiv:2412.15115, 2024.
[36] A. Grattafioriet al., “The llama 3 herd of models,”arXiv preprint
arXiv:2407.21783, 2024.
[37] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot,
et al., “Mistral 7b,”arXiv preprint arXiv:2310.06825, 2023.
[38] M. Abdinet al., “Phi-4 technical report,”arXiv preprint
arXiv:2412.08905, 2024.
[39] A. Mallen, A. Asai, V . Zhong, R. Das, D. Khashabi,et al., “When not
to trust language models: Investigating effectiveness of parametric and
non-parametric memories,”arXiv preprint arXiv:2212.10511, 2023.
[40] N. Kandpal, H. Deng, A. Roberts, E. Wallace, and C. Raffel, “Large
language models struggle to learn long-tail knowledge,” inProceedings
of the 40th International Conference on Machine Learning, PMLR, 2023.

[41] J. Chen, B. Bi, W. Zhang, J. Sui, X. Zhu,et al., “Rethinking all evidence:
Enhancing trustworthy retrieval-augmented generation via conflict-driven
summarization,”arXiv preprint arXiv:2507.01281, 2025.
[42] A. T. Kalai, O. Nachum, S. S. Vempala, and E. Zhang, “Why language
models hallucinate,”arXiv preprint arXiv:2509.04664, 2025.
[43] J. Deng, Y . Shen, Z. Pei, Y . Chen, and L. Huang, “Influence guided
context selection for effective retrieval-augmented generation,”arXiv
preprint arXiv:2509.21359, 2025.
[44] T. E. Kim and F. Diaz, “Ltrr: Learning to rank retrievers for llms,”arXiv
preprint arXiv:2506.13743, 2025.
[45] Y . Jin, K. Sharma, V . Rakesh, Y . Dou, M. Pan, M. Das, and S. Kumar,
“Sara: Selective and adaptive retrieval-augmented generation with context
compression,”arXiv preprint arXiv:2507.05633, 2025.
[46] Q. Li, S. Dou, K. Shao, C. Chen, and H. Hu, “Evaluating scoring bias
in llm-as-a-judge,”arXiv preprint arXiv:2506.22316, 2025.
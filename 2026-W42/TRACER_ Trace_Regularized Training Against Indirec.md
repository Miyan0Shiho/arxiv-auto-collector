# TRACER: Trace-Regularized Training Against Indirect Prompt Injection

**Authors**: Kshitij Mishra, Salem Lahlou, Nils Lukas

**Published**: 2026-10-13 20:03:37

**PDF URL**: [https://arxiv.org/pdf/2610.19133v1](https://arxiv.org/pdf/2610.19133v1)

## Abstract
Indirect prompt injection uses untrusted retrieved content, tool outputs, or agent messages to override an LLM's intended behavior. We show that defense robustness is channel-conditioned: the same attack succeeds 18% of the time as a follow-up message but 53% when placed in retrieved context. We propose TRACER (Trace-Regularized Adversarial Channel Exposure with Reward-shaping), a channel-aware adversarial training method that samples presentation wrappers and combines an answer-anchored, evidence-gated reward with ATRC, which aligns attacked reasoning with clean evidence when an injection is invalid. On GSM8K with Qwen2.5-3B-Instruct, TRACER reduces RAG-framed ASR from 65.5% to 6.1% while preserving 78-82% clean accuracy. Joint RAG+tool training achieves 13.9% average ASR across framings at 3B scale, while a single epoch at 7B reaches 9.4%. These results demonstrate that robust defenses must explicitly train and evaluate across deployment channels.

## Full Text


<!-- PDF content starts -->

TRACER: Trace-Regularized Training Against Indirect Prompt Injection
Kshitij Mishra and Salem Lahlou and Nils Lukas
Mohamed bin Zayed University of Artificial Intelligence
Abu Dhabi, United Arab Emirates
{kshitij.mishra,salem.lahlou,nils.lukas}@mbzuai.ac.ae
Abstract
Indirect prompt injection uses untrusted
retrieved content, tool outputs, or agent
messages to override an LLM’s intended
behavior. We show that defense robustness
is channel-conditioned: the same attack
succeeds 18% of the time as a follow-up
message but 53% when placed in retrieved
context. We propose TRACER (Trace-
Regularized Adversarial Channel Exposure
with Reward-shaping), a channel-aware
adversarial training method that samples
presentation wrappers and combines an
answer-anchored, evidence-gated reward with
ATRC, which aligns attacked reasoning with
clean evidence when an injection is invalid.
On GSM8K with Qwen2.5-3B-Instruct ,
TRACER reduces RAG-framed ASR from
65.5% to 6.1% while preserving 78–82%
clean accuracy. Joint RAG+tool training
achieves 13.9% average ASR across framings
at 3B scale, while a single epoch at 7B
reaches 9.4%. These results demonstrate
that robust defenses must explicitly train
and evaluate across deployment channels.
Research work’s codebase can be accessed at
https://github.com/Mishrakshitij/TRACER.git.
1 Introduction
Modern LLM systems rarely operate on trusted
prompts alone. Retrieval-augmented generation
(RAG) pipelines ingest documents from search or
knowledge bases (Lewis et al., 2020; Gao et al.,
2023); tool-using assistants consume API out-
puts (Qin et al., 2024; Patil et al., 2024); and LLM
agents act on observations from interactive environ-
ments (Liu et al., 2024) and on messages from other
agents (Li et al., 2023). In each case, the model
must act on text that it did not author and cannot
fully trust. This creates the setting forindirect
prompt injection: malicious instructions hidden in-
side external content rather than entered as direct
user prompts (Greshake et al., 2023).A central but under-tested assumption in this
literature is that robustness transfers across pre-
sentation channels. If the same malicious payload
is tolerated as a conversational follow-up, will it
also be tolerated when wrapped as a retrieved docu-
ment, a tool response with provenance metadata, or
a message from another agent? Existing defenses
are usually trained and evaluated within a single
wrapper, so this question is rarely isolated directly.
We begin by empirically testing this transfer as-
sumption. Specifically, we ask whether a model
trained to resist malicious instructions in one pre-
sentation channel remains robust when the same
attack content is re-wrapped in another. The an-
swer is no: across six intermediate model snapshots
trained against follow-up-style attacks, simply re-
wrapping the same attack strings as retrieved doc-
uments increases average ASR from 18% to 53%.
The base model also exhibits a strong authority gra-
dient, where the same payload is less effective as a
casual follow-up than as a document, tool output,
or agent message. We call this phenomenon the
framing gap.
To address this gap, we propose TRACER
(Trace-RegularizedAdversarialChannelExposure
withReward-shaping), a deployment-matched ad-
versarial training framework that treats the chan-
nel of untrusted context as part of the optimiza-
tion problem. TRACER combines a three-turn
robustness game with two training components de-
signed for indirect injection: an answer-anchored,
evidence-gated composite reward for thepost-
attackresponse, and ATRC, a committed-spanAd-
versarial Trace Recovery and Consistencyloss that
aligns attacked reasoning to clean anchor spans
when injected evidence is invalid. This lets us ask
not only whether a defense works in one wrapper,
but whether the learned robustness transfers across
wrappers with different authority cues. Ourkey
contributions are:
arXiv:2610.19133v1  [cs.LG]  13 Oct 2026

Indirect Prompt Injection, the Framing Gap, and TRACER
Untrusted Contexts
(same payload, four wrappers)
πθUser follow-up
malicious
instruction
RAG document
malicious
instruction
Tool output
malicious
instruction
Agent message
malicious
instruction
Identical attack content is
embedded in wrappers with
different provenance cues.Framing Gap
same payload, different authority
follow-up
RAG
tool
agent
low ASR high ASR
higher apparent authority
Single-wrapper training can learn
robustness to content plus wrap-
per, not the underlying attack.TRACER Training Game
channel-aware adversarial optimization
Turn 1
injectTurn 2
defendTurn 3
evaluateanswer-anchored evidence-gated
reward
ATRC: align attacked trace to clean
anchors
sample wrapper f∈ {followup ,rag,tool,agent}
Robustness Transfer Evaluation
native robustness vs. cross-framing transfer
3B model
baseline
multi-framebest average7B model
baseline
multi-frametrade-off
reduced
Transfer is asymmetric; breadth helps, while
scale can reduce the depth–breadth trade-off.
Framing gap: increased attack success from wrapper authority cues. TRACER: Trace-Regularized Adversarial Channel Exposure with Reward-shaping.
ASR: attack success rate.Figure 1: Left: the same malicious payload is injected through different untrusted-context channels. Middle:
changing only the wrapper creates a framing gap; apparent authority/provenance cues are one possible explanation
for the measured effect. Right: TRACER trains on channel-sampled three-turn games using an answer-anchored,
evidence-gated reward and ATRC trace alignment, and evaluates both native robustness and cross-framing transfer.
•Framing gap.We identify and quantify a
channel-conditioning effect in indirect prompt
injection: identical adversarial content becomes
substantially more successful when presented
through a different untrusted-context channel.
•Channel-aware training.We introduce
TRACER, an adversarial training framework
that combines wrapper-sampled channel expo-
sure, an answer-anchored evidence-gated reward,
and a committed-span ATRC trace-alignment
loss to preserve clean reasoning under invalid
injected evidence.
•Cross-framing evaluation.We show that trans-
fer across wrappers is asymmetric, revealing a
depth–breadth trade-off between native robust-
ness and generalization. Multi-framing training
gives the best 3B average performance, while 7B
models substantially reduce the trade-off.
2 Related Work
Indirect prompt injection and benchmarks.In-
direct prompt injection was first highlighted as a
practical threat to LLM-integrated applications by
Greshake et al. (2023). Subsequent benchmarks
broaden the attack surface from single applications
to agents, dynamic environments, and web-native
settings, including BIPIA, InjecAgent, AgentDojo,
ASB, WASP, and Hidden-in-Plain-Text (Yi et al.,2025; Zhan et al., 2024; Debenedetti et al., 2024;
Zhang et al., 2025; Evtimov et al., 2025; Guo and
Wei, 2026). Our work is complementary to these
benchmarks: rather than introducing another envi-
ronment, we isolate one factor that cuts across them
all, namely the wrapper through which untrusted
content reaches the model.
RAG robustness, poisoning, and attack design.
A related literature studies how retrieved context
can be noisy, conflicting, or poisoned. Conflic-
tQA, RGB, and PoisonedRAG evaluate failures un-
der contradictory or corrupted retrieval (Xie et al.,
2024; Chen et al., 2024; Zou et al., 2025). More
recent work explores realistic indirect injections
such as TopicAttack (Chen et al., 2025c). These
studies mainly treat robustness as a property of a
fixed pipeline. By contrast, we vary thepresenta-
tion channelwhile holding the underlying payload
fixed, allowing us to ask whether learned robust-
ness generalizes across wrappers.
Defenses.Existing defenses include prompt/data
separation with structured instruction tuning (Chen
et al., 2025a), data marking at inference
time (Hines et al., 2024), alignment-based training
such as SecAlign (Chen et al., 2025b), and run-
time detector, re-execution, parsing, or correction
defenses such as MELON, CaMeL, instruction de-
tection, robustness-via-referencing, PromptArmor,

tool-result parsing, ICON, and AttriGuard (Zhu
et al., 2025; Debenedetti et al., 2026; Wen et al.,
2025; Chen et al., 2026; Shi et al., 2025; Yu et al.,
2026; Wang et al., 2026; He et al., 2026). These
approaches usually harden a fixed pipeline or add
an auxiliary inference-time module. Our focus is
complementary: we study how the base model it-
self can be trained to maintain correct post-attack
behavior when identical malicious content arrives
through different untrusted channels.
Reinforcement learning for reasoning.RL with
verifiable rewards, most prominently GRPO, is now
a standard way to improve LLM reasoning (Shao
et al., 2024; Guo et al., 2025; Yu et al., 2025), along-
side process rewards for individual steps (Lightman
et al., 2024; Wang et al., 2024). Follow-up work
encourages exploration and diversity (Cui et al.,
2025; Cheng et al., 2026; Song et al., 2025; Walder
and Karkhanis, 2025; Mishra et al., 2026b) or lets
several models reason and learn together (Du et al.,
2024; Subramaniam et al., 2025; Park et al., 2025;
Wan et al., 2025; Mishra et al., 2026a). These meth-
ods improve reasoning on clean inputs; TRACER
instead uses the clean reasoning trace as an anchor
that attacked reasoning must stay consistent with
when injected evidence is invalid.
How prompt-injection defenses should be eval-
uated.Recent evaluations argue that prompt-
injection defenses should be assessed jointly on
security and utility, and against adaptive attacks
rather than only static strings (Jia et al., 2025; Zhan
et al., 2025). Our framing-gap analysis fits that
agenda: a defense that appears robust in one wrap-
per may fail badly in another even when the adver-
sarial payload is unchanged. We therefore empha-
size cross-framing transfer, final-checkpoint com-
parisons, and conditional ASR rather than only a
single in-channel score.
3 TRACER
Problem setup.Let D={(x i, yi)}N
i=1denote a
task dataset, wherex iis the user task andy iis the
gold answer. Let aidenote adversarial content, and
letc∈ C denote the channel through which this
content is presented. In our experiments,
C={followup,rag,tool,agent}.
We write ρc(ai)for the channel-specific render-
ing of the same adversarial payload. Thus, themalicious instruction is held fixed while only its
presentation channel changes.
For a policyπ, let
ˆy0
i= ans(π(x i))
be the answer extracted from the clean response,
and let
ˆyc
i= ans(π(x i, ρc(ai)))
be the answer extracted after adversarial content
is introduced through channel c. We report three
empirical quantities:
dAcc 0(π) =1
NNX
i=11[ˆy0
i=yi],(1)
dAccc(π) =1
NNX
i=11[ˆyc
i=yi],(2)
[ASR c(π) =PN
i=11[ˆy0
i=yi]1[ˆyc
i̸=yi]PN
i=11[ˆy0
i=yi].(3)
HeredAcc 0measures clean task performance, dAccc
measures post-attack accuracy under channel c,
and[ASR cmeasures conditional attack success
rate. The conditioning is important: an example
counts as a successful attack only when the model
solved the task before seeing the injected content
and failed after seeing it. This separates genuine
injection-induced failures from ordinary task er-
rors.
For a model πStrained on channel set S⊆ C ,
we define the cross-channel transfer gap from a
training channel ctr∈Sto an evaluation channel
cev∈ Cas
b∆ctr→cev(πS) =[ASR cev(πS)−[ASR ctr(πS).(4)
A positive value indicates that robustness learned
under ctrdoes not fully transfer to cev. In other
words, the model has learned channel-specific skep-
ticism rather than channel-invariant robustness.
Threat model.We consider indirect prompt in-
jection in which the attacker can modify onlyun-
trusted external context. The attacker cannot mod-
ify the original user task, hidden system or devel-
oper instructions, model weights, training proce-
dure, or evaluation harness. The defender must
therefore solve a semantic-control problem: the
model should use external context as evidence
when it is relevant and valid, but should not treat
adversarial text inside that context as an instruc-
tion to obey. This threat model covers retrieved

documents, tool outputs, and messages from other
agents. These channels are especially challenging
because their surface form often carries implicit
authority even though their contents may be un-
trusted.
Three-turn robustness game.Each training
episode is formulated as a three-turn robustness
game. First, the model receives the clean task xi
and produces a clean reasoning trace τ0
iand clean
answer ˆy0
i. Second, the same adversarial payload
aiis rendered through a sampled channel cias
ρci(ai). Third, the model produces a post-attack
reasoning traceτci
iand answerˆyci
i.
The central design choice is that the channel ci
is sampled as part of training:
ci∼pS(c),supp(p S) =S⊆ C.
This allows TRACER to instantiate both single-
channel specialists, such as RAG-only training with
S={rag} , and multi-channel training, such as
joint RAG+tool training with S={rag,tool} .
The wrapper is therefore not an evaluation af-
terthought; it is a random variable in the training
objective.
Composite post-attack reward.TRACER as-
signs reward only to the post-attack response. The
reward is designed to encouragecontextual skepti-
cism: the model should preserve a correct answer
when injected evidence is invalid, but should still
be able to revise its answer when new evidence
is valid. For a single training episode, the scalar
reward is
r=X
k∈Kβkrk,K={ans,proc,anc,corr,len}.
(5)
The five terms correspond to answer correctness,
process quality, anchor retention, evidence-gated
corrigibility, and length control1. We describe
the reward terms for a single episode and omit the
example index i. The answer term rewards post-
attack correctness:
rans=1[ˆyc=y].(6)
The process term rewards local validity of the post-
attack reasoning trace:
rproc=1
|Steps(τc)|X
s∈Steps(τc)V(s),(7)
1The weights for answer correctness, process quality, an-
chor retention, corrigibility, and length control are 0.25,0.25,
0.20,0.20, and0.10, respectively.where Steps(τc)denotes the reasoning steps ex-
tracted from the post-attack trace and V(s) scores
whether stepsis locally valid.
The anchor-retention term measures whether
propositions from the clean trace remain supported
after the attack. Let A0=A(τ0)be the set of
anchor propositions extracted from the clean trace.
We use a verifier ϕthat assigns support and con-
tradiction scores, denoted s+
ϕands−
ϕ, respectively.
The anchor-retention reward is
ranc=1
|A0|X
p∈A0
s+
ϕ(p, τc)−s−
ϕ(p, τc)
.(8)
The corrigibility term is evidence-gated. Let
q0=1[ˆy0=y] andqc=1[ˆyc=y] de-
note clean and post-attack correctness, and let
v=1[Valid(a, c)] indicate whether the external
evidence introduced through channel cis valid. We
write
rcorr=C(q0, qc, v),(9)
where Cassigns positive reward when the model
preserves a correct answer under invalid evidence
or corrects itself under valid evidence, and assigns
negative reward when invalid evidence flips an oth-
erwise correct answer. Appendix B gives the full
truth table. This term is what prevents the reward
from collapsing into blanket refusal: the model is
rewarded for resisting invalid evidence, but not for
ignoring genuinely useful evidence.
Finally, the length-control term discourages un-
necessary post-attack verbosity while avoiding a
penalty for incorrect answers:
rlen=−λ len1[ˆyc=y] max{0,|τc| − |τ0| −δ}.
(10)
This reward differs from answer-only supervi-
sion in two important ways. First, it distinguishes
robustness from blanket refusal: the model should
preserve its answer when external evidence is in-
valid, but should still revise when the evidence is
valid. Second, it makes the clean reasoning trace
part of the optimization target, which matters in
reasoning tasks where an attack may corrupt inter-
mediate steps rather than only the final answer.
ATRC: Adversarial Trace Recovery and Con-
sistency.The second algorithmic ingredient is
ATRC, a trace regularizer that preserves clean rea-
soning under invalid evidence. For example i, let
Midenote the set of committed response positions
induced by anchor spans in the clean trace τ0
i. Let
gi=1[ˆy0
i=yi]1[Invalid(a i, ci)](11)

Input Sampling Block
task
instance xgold
answer yadversarial
content achannel
ci∼pS(c)
rag tool followup agentTRACER
Clean Attempt
πθ(xi)→(τ0
i,ˆy0
i)
h0
i= (xi, τ0
i,ˆy0
i)Channel Rendering
ρci(ai)
Post-attack Response
πθ(h0
i, ρci(ai))
→(τci
i,ˆyci
i)Anchors & Mask
anchors A0
imask MiATRC Trace Loss
gategi: clean correct
+ invalid evidence
align response logits
on mask Mi
penalize drift:
KL + clean-token NLL
LATRCPost-attack Reward
rans
answerrproc
processranc
anchorrcorr
corrig.rlen
lengthReward
ri
ri=/summationtext
k∈Kβkri,k
Update πθ
ri↑post-attack reward; KL keeps
πθnearπ0; ATRC preserves spans
updated πθFigure 2:TRACER:The method samples a task, payload, and channel; generates clean and post-attack traces;
computes the composite reward; applies ATRC on committed clean spans when the gate is active; and updates πθ
with the post-attack objective.
be the evidence gate. The gate activates ATRC only
when the clean response is correct and the injected
evidence is invalid. This prevents the method from
forcing the model to preserve an incorrect clean
trace or to ignore genuinely valid evidence.
Letp0
i,tandpci
i,tdenote the clean and post-attack
token distributions at response-relative position t,
and let z0
i,tbe the clean target token at that position.
We define the per-token ATRC loss as
ℓATRC
i,t =αKL
sg[p0
i,t]∥pci
i,t
−(1−α) logpci
i,t(z0
i,t).(12)
The full ATRC regularizer is then
LATRC =1
BBX
i=1gi
|Mi|X
t∈M iℓATRC
i,t .(13)
The KL term preserves the clean distribution
over the vocabulary, while the token-recovery term
encourages exact recovery of committed clean to-
kens. The stop-gradient operator sg[·] treats the
clean distribution as a fixed target. The implemen-
tation aligns clean and post-attack responses by
response-relative position before applying Mi; this
alignment is necessary because the attacked prompt
contains additional channel-rendered context be-
fore the model response begins.Overall training objective.The policy is opti-
mized with GRPO (Shao et al., 2024) and ATRC
regularization. For a channel-sampling distribution
pSover training channelsS, the objective is
max
θE(x,y)∼D, a∼A, c∼p S[rθ(x, y, a, c)]
−λKLKL(π θ∥π0)−λ regLATRC ,(14)
where rθ(x, y, a, c) is the composite reward in
Equation 5, computed from the post-attack re-
sponse generated after rendering athrough channel
c. A training step with TRACER is given in Algo-
rithm 1.
The optimizer itself is standard GRPO. The nov-
elty lies in the training objective: TRACER treats
the presentation channel as a sampled training vari-
able, uses a structured post-attack reward rather
than answer-only correctness, and adds a response-
aligned trace loss over committed clean reasoning
spans, as shown in Figure 2. This makes robust-
ness channel-aware: the model is trained not only
to answer correctly after an attack, but also to pre-
serve valid clean reasoning when untrusted content
arrives through different deployment channels.
4 Experimental Setup
We primarily trainQwen2.5-3B-Instructwith 4-
bit QLoRA (Qwen Team, 2024; Dettmers et al.,

Algorithm 1TRACER Training
1:Sample a batch {(xi, yi)}B
i=1, adversarial payloads
{ai}B
i=1, and channelsc i∼pS(c)
2:Generate clean traces and answers (τ0
i,ˆy0
i)fromπθ(xi)
3: Form the clean dialogue prefixh0
i= (x i, τ0
i,ˆy0
i)
4:Render each payload through its sampled channel as
ρci(ai)
5:Generate post-attack traces and answers (τci
i,ˆyci
i)from
πθ(h0
i, ρci(ai))
6: Compute the composite rewardr i=P
k∈Kβkri,k
7:Extract clean anchors A0
i=A(τ0
i)and build the
committed-position maskM i
8:Construct the ATRC gate gi=1[ˆy0
i=
yi]1[Invalid(a i, ci)]
9:Align clean and post-attack logits by response-relative
position and computeL ATRC over{M i, gi}B
i=1
10:Update πθwith the GRPO post-attack objective, KL reg-
ularization, and ATRC
2023; Hu et al., 2022); we also report a 7B scaling
study. The main training task is GSM8K (Cobbe
et al., 2021), using 2,000 training examples and
200 evaluation examples per checkpoint. Evalua-
tion wrappers are follow-up, RAG, tool-output, and
agent-message.
Attacks.We evaluate three malicious attack fam-
ilies.Reasoning rewritetries to replace the model’s
solution path with an alternative chain of reasoning.
Preemptive answerpresents a supposedly verified
final answer.Prompt injectionissues explicit in-
structions to ignore prior reasoning or higher-level
intent. Train and evaluation tasks are disjoint; addi-
tional tests use held-out paraphrases and defense-
aware attacks. Training also includes a 15% benign
ratio so that robustness is not achieved by uncon-
ditional refusal. Appendix A details the split and
adversarial construction.
Training protocol.The main 3B configuration
uses three epochs, learning rate 2×10−6, batch
size 1, gradient accumulation 8, group size 4, KL
coefficient 0.10, ATRC weight 0.10, and vLLM-
accelerated rollouts (Kwon et al., 2023). We train
channel specialists for RAG, tool-output, and agent-
message framing, plus a joint model that alter-
nates between RAG and tool framings. Full hy-
perparameters appear in Appendix H. Unless oth-
erwise stated, cross-framing comparisons usefi-
nal checkpointsfor every specialist model. This
avoids comparing a cherry-picked best checkpoint
for one channel against a final checkpoint for an-
other. When we analyze optimization dynamics,
we separately report checkpoint-level curves and
the best intermediate checkpoint reached during
training.Evaluation protocol.For transfer experiments,
the adversarial payload is held fixed and only the
wrapper changes. This separates “learning the
content of an attack” from “learning the channel
through which that content arrives.” Together, the
evaluation axes let us answer three distinct ques-
tions: does framing create a real failure mode, does
TRACER close the deployment-matched gap, and
how well does the learned robustness transfer be-
yond the native channel?
Terminology and metric boundary.We use
channelfor the source or pathway by which un-
trusted context enters the model,wrapperfor its
concrete textual rendering, andframing gapfor the
change in ASR when a fixed payload is re-rendered
through another channel. Evidence validity is la-
beled by construction; training-only step and an-
chor verifiers do not enter reported accuracy or
ASR, which depend only on final-answer correct-
ness. Appendix A documents their implementation
and audit.
5 Results and Analysis
We organize the results around the paper’s three-
part claim: a scientific diagnosis of the framing gap,
an algorithmic validation of the reward and trace-
regularized TRACER objective, and an empirical
characterization of transfer across channels.
The framing gap is large and consistent.Fig-
ure 3 shows two complementary views of the prob-
lem. The left panel measures base-model vulner-
ability by framing and reveals a clear authority
gradient: for prompt injection, ASR rises from
70% under casual follow-ups to roughly 92–96%
under agent, tool, and RAG wrappers. The right
panel isolates transfer failure directly. Across six
intermediate model snapshots trained only against
follow-up-style attacks, merely switching to RAG
presentation raises average ASR from 18% to 53%
(+35pp).
This result has a direct deployment implication.
A model hardened against conversational follow-
up attacks should not be assumed robust in a RAG
pipeline, a tool-using assistant, or a multi-agent
system. Robustness learned under one wrapper is
often robustness to thatwrapper, not to the under-
lying attack content.
We use “authority” as an interpretive shorthand,
not as an isolated causal conclusion. Format-
ting, content position, provenance markers, and

Figure 3:The framing gap.Left: the base model is much more vulnerable when the same attack appears inside a
more formal wrapper, revealing an authority gradient. Right: six checkpoints trained only on follow-up attacks fail
to transfer to RAG evaluation even when the attack payload is unchanged.
instruction-tuning priors also change with the wrap-
per and may contribute to the measured gap. The
controlled claim is that the channel-specific ren-
dering changes ASR while payload content is held
fixed.
TRACER closes the native RAG vulnerability.
We next train directly in the deployment-matched
RAG wrapper. Under held-out RAG evaluation, the
3B final checkpoint reduces ASR from 65.5% (base
model) to 6.1%. Figure 4 shows the full checkpoint
trajectory: the sharpest gains appear between the
late first epoch and early second epoch, and the best
intermediate checkpoint reaches 3.8% ASR while
maintaining clean accuracy in the same 78–82%
band as neighboring checkpoints.
The gain is not merely a reduction in average fail-
ures. Figure 5 shows that prompt-injection attacks
drop from 77.5% to 2.3% ASR, while preemptive-
answer attacks drop from 54.5% to 3.4%. Because
ASR is measured only on examples the model
solved in the clean condition, this improvement
cannot be explained away as capability loss alone.
The model still solves the task and becomes far less
willing to obey injected content.
This is the main algorithmic win of TRACER.
The wrapper exposes the model to deployment-
matched authority cues; the composite reward
makes post-attack correctness, anchor retention,
and evidence-sensitive revision directly valuable;
and ATRC discourages the model from drifting
away from clean anchor spans when the injected
evidence is invalid.Stress tests, mixed evidence, and component at-
tribution.Table 1 consolidates the evaluations
added to test the security boundary and isolate
the method components. On held-out paraphrases,
TRACER retains 10.0% ASR versus 62.7% for the
base model, showing that the result is not explained
by exact-template recognition. Channel-aware es-
calation and defense-aware attacks are stronger,
raising TRACER to 18.1% and 23.8% ASR, re-
spectively, but the corresponding baselines remain
68.8–70.0%. Thus, TRACER substantially nar-
rows adaptive vulnerability without claiming im-
munity to a fully optimized white-box adversary.
The direct ablation rules out channel exposure as
the sole explanation: answer-only training reaches
21.3% ASR, the composite reward lowers this to
13.8%, and ATRC further lowers it to 6.1%. The
mixed-evidence evaluation addresses blanket con-
text rejection. Clean/evidence accuracy remains
81.0–83.0%, while injection-induced flip rates fall
from 46.3–58.1% for the base model to 13.8–17.5%
with TRACER.
The principal uncertainty estimates support
the large framing-gap claim. Follow-up evalu-
ation is 18.0% ASR (173/960; Wilson 95% CI
[15.7,20.6] ), whereas RAG evaluation is 53.0%
(509/960; [49.9,56.2] ). For native RAG, the
base model is 65.6% (99/151; [57.7,72.7] ) and
TRACER is 6.1% (10/164; [3.3,10.9] ). We do
not claim that smaller differences among structured
wrappers are all statistically distinct.
Cross-framing transfer reveals a depth–breadth
trade-off.The full picture emerges when we eval-

Figure 4:RAG training trajectory across model snapshots.Clean accuracy stays near 78–82% while robust
accuracy rises from 38% to above 80%. The best intermediate model snapshot reaches 3.8% ASR.
(a) Attack generalization (b) Objective ablation (c) Valid evidence + injection
Setting Base TRACER Variant ATRC ASR Channel Ev. acc. Base flip TRACER flip
Original 65.5 6.1 Base no 65.5 RAG 83.0 58.1 13.8
Held-out paraphrase 62.7 10.0 Answer-only no 21.3 Tool 82.5 52.5 17.5
Channel-aware 68.8 18.1 Composite reward no 13.8 Agent 81.0 46.3 16.3
Defense-aware 70.0 23.8 Full TRACER yes 6.1
Table 1:Stress tests and attribution (ASR in %).(a) Generalization beyond the original RAG templates. (b)
Direct objective ablations under RAG evaluation. (c) Mixed-evidence evaluation, where useful channel content also
contains an injection; “flip” is conditional ASR from a correct clean-evidence response.
Figure 5:Per-attack effects of RAG-framed training.
Prompt injection is both the most dangerous attack and
the one most reduced by TRACER.
uate trained models across multiple framings. Fig-
ure 6 shows three distinct regimes. First, the un-
trained 3B base model exhibits a strong author-
ity gradient: ASR is lowest for casual follow-
ups (41.4%) and highest for retrieved documents
(65.5%), with tool outputs and agent messages in
between. Formal wrappers make attacks more per-
suasive.
Second, RAG-specific training yields the deepest
nativedefense but relatively weak transfer: the 3BRAG specialist reaches 6.1% on RAG yet remains
at 25.7% and 26.5% on tool and agent framings.
Tool-output training shows the opposite pattern: its
native ASR is higher (15.0%), but it is nearly flat
across all four framings (15.0–20.2%). In other
words, RAG training learns depth, whereas tool
training learns breadth.
Third, multi-framing training partially resolves
this trade-off. The joint RAG+tool model achieves
the best 3B average ASR (13.9%) and even slightly
improves native RAG robustness over the final
RAG specialist (5.2% vs. 6.1%). This suggests that
broader channel coverage during training helps the
model acquire a more transferable notion of “un-
trusted external context.” Hence, transfer asymme-
try shows that robustness is not one-dimensional:
strong in-channel defense can coexist with cross-
channel brittleness. Single-wrapper evaluation
therefore misses the depth–breadth trade-off.
Ablations, scaling, and broader transfer
strengthen the story.Three additional results
sharpen the main claim. First, the RAG wrapper
and the RAG-specific system prompt are comple-
mentary: using both yields 5.5% ASR, compared
with 10.0% for wrapper-only, 11.3% for prompt-
only, and 60.5% for neither (Appendix F). This sup-

Figure 6:Cross-framing transfer at 3B scale.Gold
boxes mark native training framing(s). RAG-specific
training yields the deepest native defense, tool-output
training yields the flattest profile, and multi-framing
training gives the best average performance.
ports the view that deployment-matched robustness
depends on both structural cues and the surround-
ing instruction context.
Second, the phenomenon persists but becomes
easier to mitigate at larger scale. The 7B base
model already shows a milder authority gradient
than the 3B base, and a single epoch of RAG
training reaches 9.4% average ASR across the
four framings, outperforming the best 3B sys-
tem. Evaluation-only diagnostics on Llama-3.1-8B-
Instruct, Gemma-2-9B-It, and Mistral-7B-Instruct
also exhibit 22.0–24.7 ppframing spreads (Ap-
pendix A). The final 7B model reaches 10.8% av-
erage ASR, indicating that most of the gain is real-
ized early and that early stopping may be preferable
when transfer, rather than only native-channel ASR,
is the target.
Third, we test transfer beyond the GSM8K train-
ing domain. On RAG-framed Natural Questions
and TriviaQA, framing-aware models retain rela-
tively low ASR, but clean accuracy remains low
for 3B models, indicating that robustness transfers
more readily than task-specific competence. In
other words, the model can learn to be skeptical of
untrusted context before it becomes broadly strong
at every downstream task.
These results show that indirect prompt-injection
robustness is channel-conditioned: changing only
the wrapper can substantially change attack success.
TRACER improves robustness by training on the
channel that carries untrusted context, rather than
treating the wrapper as an evaluation detail. Cross-
channel evaluation further reveals a depth–breadth
trade-off hidden by single-wrapper reporting: spe-
cialized training is strongest in-channel, whereasmulti-channel training transfers better across de-
ployment framings.
6 Conclusion
We study indirect prompt injection as a channel-
conditioned robustness problem. Holding payloads
fixed, we find that a model trained on malicious
follow-up messages remains vulnerable when the
same attacks are rendered as retrieved context,
increasing ASR from 18% to 53%. We intro-
duce TRACER, which samples presentation chan-
nels during training and combines composite post-
attack rewards with ATRC trace regularization. On
GSM8K with Qwen2.5-3B-Instruct , TRACER
reduces RAG-framed ASR from 65.5% to 6.1%
while maintaining 78–82% clean accuracy; joint
RAG+tool training further achieves the best 3B
cross-channel average ASR of 13.9%. Held-out,
adaptive, and mixed-evidence evaluations preserve
substantial gains while exposing a clear residual
attack surface. These results show that defenses
should train on deployment-matched framings, re-
port cross-channel transfer, and remain paired with
runtime safeguards.
Limitations
Our primary attacks are template-based and de-
signed to hold the adversarial payload fixed while
varying the channel. This supports a clean measure-
ment of the framing gap. Held-out paraphrases and
a targeted defense-aware suite broaden the test, but
do not cover arbitrary optimization-based or multi-
turn adversaries. In addition, the framing-gap diag-
nostic spans several model families, whereas full
TRACER training is evaluated within Qwen2.5
at 3B and 7B scale. Broader architectures, tasks,
seeds, and adaptive strategies remain important
next steps. Further, TRACER is a training-time
defense. It is complementary to inference-time
safeguards such as retrieval filtering, tool-output
parsing, policy monitors, and sandboxing. We ex-
pect the strongest deployed systems to combine
channel-aware robust training with such runtime
protections.
Ethical Considerations
This work studies indirect prompt injection, a
security-relevant failure mode in LLM systems.
The research has dual-use aspects: the same at-
tack templates used to evaluate robustness could
be adapted to probe deployed systems. To reduce

this risk, our experiments use controlled task set-
tings and synthetic template-based injections rather
than targeting real users, private data, or produc-
tion services. The primary purpose of the attack
construction is to measure and improve robustness
under different presentation channels.
The proposed method is intended as a defen-
sive training procedure. It may reduce suscepti-
bility to malicious retrieved documents, tool out-
puts, and agent messages, but it should not be
viewed as a complete security boundary. In de-
ployment, channel-aware robust training should be
combined with runtime safeguards such as retrieval
filtering, tool-output validation, monitoring, sand-
boxing, and least-privilege tool access. We also
note that overly aggressive defenses can harm use-
fulness by causing unwarranted refusal or distrust
of valid evidence. Our reward design explicitly
separates invalid-evidence resistance from valid-
evidence correction to mitigate this risk.
References
Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun.
2024. Benchmarking large language models in
retrieval-augmented generation.Proceedings of
the AAAI Conference on Artificial Intelligence,
38(16):17754–17762.
Sizhe Chen, Julien Piet, Chawin Sitawarin, and David
Wagner. 2025a. StruQ: Defending against prompt
injection with structured queries. In34th USENIX
Security Symposium (USENIX Security 25), pages
2383–2400. USENIX Association.
Sizhe Chen, Arman Zharmagambetov, Saeed Mahlouji-
far, Kamalika Chaudhuri, David Wagner, and Chuan
Guo. 2025b. SecAlign: Defending against prompt in-
jection with preference optimization. InProceedings
of the 2025 ACM SIGSAC Conference on Computer
and Communications Security (CCS), pages 2833–
2847. ACM.
Yulin Chen, Haoran Li, Yuexin Li, Yue Liu, Yangqiu
Song, and Bryan Hooi. 2025c. TopicAttack: An
indirect prompt injection attack via topic transition.
InProceedings of the 2025 Conference on Empiri-
cal Methods in Natural Language Processing, pages
7327–7345.
Yulin Chen, Haoran Li, Yuan Sui, Yue Liu, Yufei He,
Xiaoling Bai, Chi Fei, Yabo Li, Haozhe Ma, Yangqiu
Song, and Bryan Hooi. 2026. Robustness via refer-
encing: Defending against prompt injection attacks
by referencing the executed instruction. InFind-
ings of the Association for Computational Linguis-
tics: ACL 2026, pages 1194–1215. Association for
Computational Linguistics.Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai,
Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei.
2026. Reasoning with exploration: An entropy per-
spective. InProceedings of the AAAI Conference
on Artificial Intelligence, volume 40, pages 30377–
30385.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian,
Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias
Plappert, Jerry Tworek, Jacob Hilton, Reiichiro
Nakano, and 1 others. 2021. Training verifiers
to solve math word problems.arXiv preprint
arXiv:2110.14168.
Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan,
Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan,
Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng,
Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, and
Ning Ding. 2025. The entropy mechanism of rein-
forcement learning for reasoning language models.
arXiv preprint arXiv:2505.22617.
Edoardo Debenedetti, Ilia Shumailov, Tianqi Fan, Jamie
Hayes, Nicholas Carlini, Daniel Fabian, Christoph
Kern, Chongyang Shi, Andreas Terzis, and Florian
Tramèr. 2026. Defeating prompt injections by de-
sign. InIEEE Conference on Secure and Trustworthy
Machine Learning (SaTML). ArXiv:2503.18813.
Edoardo Debenedetti, Jie Zhang, Mislav Balunovi ´c,
Luca Beurer-Kellner, Marc Fischer, and Florian
Tramèr. 2024. AgentDojo: A dynamic environment
to evaluate prompt injection attacks and defenses for
LLM agents. InAdvances in Neural Information
Processing Systems (NeurIPS), Datasets and Bench-
marks Track, volume 37.
Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and
Luke Zettlemoyer. 2023. QLoRA: Efficient finetun-
ing of quantized LLMs. InAdvances in Neural In-
formation Processing Systems (NeurIPS), volume 36,
pages 10088–10115.
Yilun Du, Shuang Li, Antonio Torralba, Joshua B.
Tenenbaum, and Igor Mordatch. 2024. Improving
factuality and reasoning in language models through
multiagent debate. InProceedings of the 41st Inter-
national Conference on Machine Learning (ICML),
volume 235 ofProceedings of Machine Learning
Research, pages 11733–11763.
Ivan Evtimov, Arman Zharmagambetov, Aaron
Grattafiori, Chuan Guo, and Kamalika Chaudhuri.
2025. WASP: Benchmarking web agent security
against prompt injection attacks. InAdvances in
Neural Information Processing Systems (NeurIPS),
Datasets and Benchmarks Track, volume 38.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2023. Retrieval-augmented gen-
eration for large language models: A survey.arXiv
preprint arXiv:2312.10997.
Kai Greshake, Sahar Abdelnabi, Shailesh Mishra,
Christoph Endres, Thorsten Holz, and Mario Fritz.

2023. Not what you’ve signed up for: Compromis-
ing real-world LLM-integrated applications with in-
direct prompt injection. InProceedings of the 16th
ACM Workshop on Artificial Intelligence and Security
(AISec), pages 79–90.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang,
Shirong Ma, Xiao Bi, and 1 others. 2025. DeepSeek-
R1 incentivizes reasoning in LLMs through reinforce-
ment learning.Nature, 645(8081):633–638.
Haoze Guo and Ziqi Wei. 2026. Hidden-in-plain-text: A
benchmark for social-web indirect prompt injection
in RAG. InProceedings of the ACM Web Conference
2026 (WWW ’26). ArXiv:2601.10923.
Yu He, Haozhe Zhu, Yiming Li, Shuo Shao, Hongwei
Yao, Zhihao Liu, and Zhan Qin. 2026. AttriGuard:
Defeating indirect prompt injection in LLM agents
via causal attribution of tool invocations. In35th
USENIX Security Symposium (USENIX Security 26),
pages 1547–1566.
Keegan Hines, Gary Lopez, Matthew Hall, Federico
Zarfati, Yonatan Zunger, and Emre Kıcıman. 2024.
Defending against indirect prompt injection attacks
with spotlighting. InProceedings of the Conference
on Applied Machine Learning in Information Secu-
rity (CAMLIS), volume 3920 ofCEUR Workshop
Proceedings, pages 48–62.
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and
Weizhu Chen. 2022. LoRA: Low-rank adaptation of
large language models. InInternational Conference
on Learning Representations (ICLR).
Yuqi Jia, Zedian Shao, Yupei Liu, Jinyuan Jia, Dawn
Song, and Neil Zhenqiang Gong. 2025. A critical
evaluation of defenses against prompt injection at-
tacks.arXiv preprint arXiv:2505.18333.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E Gon-
zalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serv-
ing with PagedAttention. InProceedings of the ACM
SIGOPS 29th Symposium on Operating Systems Prin-
ciples, pages 611–626.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented
generation for knowledge-intensive NLP tasks.Ad-
vances in Neural Information Processing Systems,
33:9459–9474.
Guohao Li, Hasan Abed Al Kader Hammoud, Hani
Itani, Dmitrii Khizbullin, and Bernard Ghanem. 2023.
CAMEL: Communicative agents for “mind” explo-
ration of large language model society. InAdvances
in Neural Information Processing Systems (NeurIPS),
volume 36.Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harri-
son Edwards, Bowen Baker, Teddy Lee, Jan Leike,
John Schulman, Ilya Sutskever, and Karl Cobbe.
2024. Let’s verify step by step. InInternational
Conference on Learning Representations (ICLR).
Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu
Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen
Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Ao-
han Zeng, Zhengxiao Du, Chenhui Zhang, Sheng
Shen, Tianjun Zhang, Yu Su, Huan Sun, and 3 others.
2024. AgentBench: Evaluating LLMs as agents. In
International Conference on Learning Representa-
tions (ICLR).
Kshitij Mishra, Mirat Aubakirov, Martin Takac, Nils
Lukas, and Salem Lahlou. 2026a. CoRe: Collabo-
rative reasoning via cross teaching. InProceedings
of the 43rd International Conference on Machine
Learning (ICML). ArXiv:2601.21600.
Kshitij Mishra, Nils Lukas, and Salem Lahlou. 2026b.
SD-E2: Semantic exploration for reasoning under
token budgets. InFindings of the Association for
Computational Linguistics: EACL 2026, pages 6144–
6157. Association for Computational Linguistics.
Chanwoo Park, Seungju Han, Xingzhi Guo, Asuman E.
Ozdaglar, Kaiqing Zhang, and Joo-Kyung Kim. 2025.
MAPoRL: Multi-agent post-co-training for collab-
orative large language models with reinforcement
learning. InProceedings of the 63rd Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 30215–30248. Asso-
ciation for Computational Linguistics.
Shishir G. Patil, Tianjun Zhang, Xin Wang, and
Joseph E. Gonzalez. 2024. Gorilla: Large language
model connected with massive APIs. InAdvances in
Neural Information Processing Systems (NeurIPS),
volume 37.
Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan
Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang,
Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian,
Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li,
Zhiyuan Liu, and Maosong Sun. 2024. ToolLLM:
Facilitating large language models to master 16000+
real-world APIs. InInternational Conference on
Learning Representations (ICLR).
Qwen Team. 2024. Qwen2.5 technical report.arXiv
preprint arXiv:2412.15115.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, Y . K. Li, Y . Wu, and Daya Guo. 2024.
DeepSeekMath: Pushing the limits of mathematical
reasoning in open language models.arXiv preprint
arXiv:2402.03300.
Tianneng Shi, Kaijie Zhu, Zhun Wang, Yuqi Jia, Will
Cai, Weida Liang, Haonan Wang, Hend Alzahrani,
Joshua Lu, Kenji Kawaguchi, Basel Alomair, Xuan-
dong Zhao, William Yang Wang, Neil Gong, Wenbo

Guo, and Dawn Song. 2025. Promptarmor: Sim-
ple yet effective prompt injection defenses.arXiv
preprint arXiv:2507.15219.
Yuda Song, Julia Kempe, and Rémi Munos. 2025.
Outcome-based exploration for LLM reasoning.
arXiv preprint arXiv:2509.06941.
Vighnesh Subramaniam, Yilun Du, Joshua B. Tenen-
baum, Antonio Torralba, Shuang Li, and Igor Mor-
datch. 2025. Multiagent finetuning: Self improve-
ment with diverse reasoning chains. InInternational
Conference on Learning Representations (ICLR).
Christian Walder and Deep Tejas Karkhanis. 2025.
Pass@K policy optimization: Solving harder rein-
forcement learning problems. InAdvances in Neu-
ral Information Processing Systems (NeurIPS), vol-
ume 38, pages 169225–169254.
Ziyu Wan, Yunxiang Li, Xiaoyu Wen, Yan Song, Han-
jing Wang, Linyi Yang, Mark Schmidt, Jun Wang,
Weinan Zhang, Shuyue Hu, and Ying Wen. 2025.
ReMA: Learning to meta-think for LLMs with multi-
agent reinforcement learning. InAdvances in Neu-
ral Information Processing Systems (NeurIPS), vol-
ume 38, pages 140399–140445.
Che Wang, Fuyao Zhang, Jiaming Zhang, Ziqi
Zhang, Yinghui Wang, Longtao Huang, Jianbo Gao,
Zhong Chen, and Wei Yang Bryan Lim. 2026.
Icon: Indirect prompt injection defense for agents
based on inference-time correction.arXiv preprint
arXiv:2602.20708.
Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai
Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui.
2024. Math-Shepherd: Verify and reinforce LLMs
step-by-step without human annotations. InProceed-
ings of the 62nd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 9426–9439. Association for Computa-
tional Linguistics.
Tongyu Wen, Chenglong Wang, Xiyuan Yang, Haoyu
Tang, Yueqi Xie, Lingjuan Lyu, Zhicheng Dou, and
Fangzhao Wu. 2025. Defending against indirect
prompt injection by instruction detection. InFind-
ings of the Association for Computational Linguistics:
EMNLP 2025, pages 19472–19487. Association for
Computational Linguistics.
Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and
Yu Su. 2024. Adaptive chameleon or stubborn sloth:
Revealing the behavior of large language models in
knowledge conflicts. InInternational Conference on
Learning Representations (ICLR).
Jingwei Yi, Yueqi Xie, Bin Zhu, Emre Kiciman,
Guangzhong Sun, Xing Xie, and Fangzhao Wu.
2025. Benchmarking and defending against indirect
prompt injection attacks on large language models.
InProceedings of the 31st ACM SIGKDD Conference
on Knowledge Discovery and Data Mining (KDD),
pages 1809–1820.Qiang Yu, Xinran Cheng, and Chuanyi Liu. 2026. De-
fense against indirect prompt injection via tool result
parsing.arXiv preprint arXiv:2601.04795.
Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan,
Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan,
Gaohong Liu, Juncai Liu, Lingjun Liu, Xin Liu, and
1 others. 2025. DAPO: An open-source LLM rein-
forcement learning system at scale. InAdvances in
Neural Information Processing Systems (NeurIPS),
volume 38, pages 125532–125554.
Qiusi Zhan, Richard Fang, Henil Shalin Panchal, and
Daniel Kang. 2025. Adaptive attacks break defenses
against indirect prompt injection attacks on LLM
agents. InFindings of the Association for Computa-
tional Linguistics: NAACL 2025, pages 7116–7132.
Qiusi Zhan, Zhixiang Liang, Zifan Ying, and Daniel
Kang. 2024. InjecAgent: Benchmarking indirect
prompt injections in tool-integrated large language
model agents. InFindings of the Association for
Computational Linguistics: ACL 2024, pages 10471–
10506.
Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao,
Zhenting Wang, Chenlu Zhan, Hongwei Wang, and
Yongfeng Zhang. 2025. Agent security bench (ASB):
Formalizing and benchmarking attacks and defenses
in LLM-based agents. InInternational Conference
on Learning Representations (ICLR).
Kaijie Zhu, Xianjun Yang, Jindong Wang, Wenbo Guo,
and William Yang Wang. 2025. MELON: Provable
defense against indirect prompt injection attacks in
AI agents. InProceedings of the 42nd International
Conference on Machine Learning (ICML), volume
267 ofProceedings of Machine Learning Research,
pages 80310–80329.
Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan
Jia. 2025. PoisonedRAG: Knowledge corruption at-
tacks to retrieval-augmented generation of large lan-
guage models. In34th USENIX Security Symposium
(USENIX Security 25), pages 3827–3844. USENIX
Association.
A Camera-ready evaluation additions
This section records the additional controls re-
quested during author–reviewer discussion. They
refine the scope of the main claim without changing
the original training objective or evaluation proto-
col.
A.1 Attack construction and split
The train and evaluation sets use disjoint GSM8K
task instances. The high-level attack families are
shared intentionally so that cross-channel experi-
ments vary the presentation pathway rather than
the attack taxonomy. Held-out evaluation addition-
ally changes the surface form through templates

and paraphrases not used in training. The targeted
adaptive suite strengthens wrapper provenance cues
and explicitly references the defense’s expected be-
havior.
The original result should therefore be read as
controlled robustness to known or family-matched
attacks, not as robustness to arbitrary attacks. Held-
out paraphrases show that the gain is not exact-
string memorization. The stronger adaptive results
identify the remaining attack surface and motivate
combining TRACER with runtime safeguards.
A.2 Training signals and verifier audit
Evidence validity is a ground-truth construction
label attached when an episode is generated; it
is not inferred from the policy output. The step
scorer segments the post-attack trace and applies
the training verifier/rules to local reasoning steps.
The anchor verifier checks whether propositions
extracted from the clean trace remain supported
or are contradicted in the attacked trace. These
two signals affect reward computation only. Final-
answer extraction compares the normalized answer
with the gold label and is the only learned-output
judgment used by reported clean accuracy, robust
accuracy, and ASR.
The objective-ablation panel in Table 1 further
separates the sources of improvement: answer-only
channel exposure reaches 21.3% RAG ASR, the
composite reward reaches 13.8%, and ATRC re-
duces the final result to 6.1%.
A.3 Statistical uncertainty and denominators
Because ASR is conditioned on clean correctness,
its denominator is the number of clean-solved ex-
amples, not the full evaluation set. Table 5 reports
the supplied exact counts and Wilson intervals for
the headline comparisons. The large follow-up-to-
RAG transfer gap remains well separated; small
pairwise differences among structured wrappers
should not be interpreted as a strict ordering.
For the held-out and defense-aware TRACER
evaluations, the corresponding exact outcomes are
16/160 (10.0%) and 38/160 (23.8%), respectively.
A.4 Mixed valid evidence and embedded
injection
The original three-turn protocol is an isolation ex-
periment: it starts from a clean solution and mea-
sures attack-induced correct-to-incorrect flips. A
deployment system may instead receive useful ev-
idence that also contains an injection. We there-fore place task-relevant evidence and a malicious
instruction in the same RAG, tool, or agent chan-
nel. Table 6 shows that TRACER retains high
clean/evidence accuracy while substantially reduc-
ing the contamination-induced flip rate.
For evidence-dependent tasks, a clean-evidence
baseline is preferable to the clean-no-evidence an-
chor used for GSM8K. The mixed-evidence test is
a deployment-facing extension rather than a change
to the isolation protocol.
A.5 Broader-model framing diagnostics
Table 7 evaluates the framing phenomenon, with-
out additional TRACER training, on several
instruction-tuned model families. Absolute robust-
ness varies with scale and model family, but every
tested model shows a 22.0–27.1 ppspread across
renderings. The evidence supports a cross-model
framing-gap diagnosis; the training-time mitigation
is established on Qwen2.5-3B/7B, and full cross-
architecture training remains future work.
B Additional method details
Full corrigibility truth table.The corrigibility
reward implements six cases: invalid evidence with
a correct clean answer is rewarded when the model
stays correct and penalized when it flips; valid evi-
dence is rewarded when the model corrects a wrong
clean answer and mildly penalized when it fails to
do so. This makes “good stubbornness” and “good
revision” distinct behaviors rather than opposites
on one axis.
Anchor spans and gating.Anchor propositions
are extracted from the clean reasoning trace and
filtered through a verifier gate so that ATRC applies
only when the clean answer is worth preserving.
In the implementation, anchor spans are aligned
in response-relative coordinates before clean and
attacked logits are compared.
Channel wrappers.The four wrappers corre-
spond to realistic deployment pathways: user
follow-up, retrieved document, tool response, and
agent message. The wrapper changes the model’s
interpretation of the same string without changing
its underlying adversarial content.
C Complete RAG training trajectory
Figure 7 serves as the full checkpoint-level eval-
uation in place of a long checkpoint table. The

Attack family Goal Train/evaluation relation
Reasoning rewrite Replace the solution path with an alternative chain Disjoint tasks + held-out paraphrases
Preemptive answer Inject a false “verified” final answer Disjoint tasks + held-out paraphrases
Prompt injection Issue an explicit instruction override Disjoint tasks + held-out paraphrases
Table 2:Attack families and split.Families are shared to isolate channel transfer; task instances and held-out
surface forms are disjoint.
Figure 7:Complete RAG training evaluation (200 samples per checkpoint).Clean accuracy, robust accuracy,
and ASR across 28 checkpoints. Most of the gain appears before the end of epoch 2.
RAG evaluation Base TRACER
Original templates 65.5% 6.1%
Held-out paraphrases 62.7% 10.0%
Channel-aware escalation 68.8% 18.1%
Defense-aware suite 70.0% 23.8%
Table 3:Adaptive evaluation.The defense-aware suite
explicitly targets the learned training behavior.
steepest improvement occurs once the model be-
gins treating retrieved content as potentially adver-
sarial rather than automatically authoritative.
D Complete cross-framing results
Table 8 and Figures 8–9 provide the full appendix
view behind the condensed main-paper heatmap.
The important point is that robustness is not a sin-
gle scalar property: each trained model has both a
levelof robustness (its average ASR across fram-
ings) and ashapeof robustness (how unevenly
that ASR is distributed across framings). The 3B
base model averages 53.2% ASR across the four
wrappers, with a 24.1 ppspread between follow-up
and RAG presentation. This authority gradient is
present before any finetuning, which means that
wrapper semantics alone materially change how
persuasive the exact same injected content appears
to the model.Depth versus breadth.The 3B RAG specialist
gives the deepest native defense: 6.1% ASR in its
matched RAG setting, a 59.4 ppreduction from
the RAG base model. But its transfer profile re-
mains uneven, with 25.7% ASR on tool outputs
and 26.5% on agent messages. Tool-output train-
ing exhibits the opposite behavior. At the end of
epoch 1 it is still weak everywhere (32.1–40.3%),
but already noticeably flatter than the RAG spe-
cialist. After convergence, the final tool-trained
model reaches 15.0–20.2% ASR across all four
wrappers, reducing the spread to only 5.2 pp. The
final multi-framing model partially combines these
benefits: it retains very strong RAG performance
(5.2%) while achieving the best overall 3B average
ASR (13.9%). These patterns are the empirical ba-
sis for the depth–breadth trade-off described in the
main text.
Interpreting the asymmetry.Figure 9 makes the
asymmetry easier to see by plotting improvement
relative to the corresponding base-model framing
rather than raw ASR. RAG training gives its largest
gain in RAG itself and smaller but still substantial
reductions on follow-up, tool, and agent evaluation.
Agent-message training occupies an intermediate
regime: it does not match the uniformity of the fi-
nal tool specialist, but it improves all four framings
and achieves its best final transfer to RAG (15.7%),

Signal Used in Implementation Used by evaluation? Audit
Evidence validity Corrigibility, ATRC gate Construction label No Exact
Step validity Process reward Verifier + rules No 90.0% agreement
Anchor support/contradiction Anchor reward Verifier No 91.0% agreement
Final-answer correctness Accuracy, ASR Answer extraction Yes 99.5% agreement
Table 4:Training and evaluation signals.Verifier audits report agreement on a manually checked sample;
evaluation metrics do not depend on the training-only step or anchor verifiers.
Figure 8:Full transfer matrix visualization.The heatmap includes all 3B and 7B settings considered in the
paper. The base model shows an authority gradient, while trained models differ in how concentrated or broad their
robustness becomes.
suggesting that structured agent messages and re-
trieved documents share some authority cues. The
practical implication is that cross-framing trans-
fer is systematic rather than random: neighboring
channels can share robustness, but the amount of
sharing depends on the authority structure of the
wrapper.Model-size effects.The 7B rows are included
explicitly to show that scale reduces, but does
not eliminate, framing dependence. The 7B base
model starts at a much lower average ASR than
the 3B base (24.3% vs. 53.2%), yet still exhibits a
pronounced spread between follow-up (9.5%) and
tool/RAG framing (34–37%). After one epoch of

Figure 9:Transfer asymmetry as reduction from the base model.Specialists reduce ASR unevenly across
evaluation wrappers. RAG training gives the deepest native reduction, tool training gives the flattest profile, and
multi-framing plus 7B training improve the overall shape.
Result ASR Successes/denom. Wilson 95% CI
Follow-up eval. average 18.0% 173/960 [15.7, 20.6]
RAG eval. average 53.0% 509/960 [49.9, 56.2]
Base RAG 65.6% 99/151 [57.7, 72.7]
TRACER RAG 6.1% 10/164 [3.3, 10.9]
Table 5:Conditional-ASR denominators and uncer-
tainty.Minor differences between aggregated and plot-
ted percentages arise from the reporting granularity of
the corresponding evaluations.
Mixed-evidence channel Evidence acc. Base flip ASR TRACER flip ASR
RAG document 83.0% 58.1% 13.8%
Tool output 82.5% 52.5% 17.5%
Agent message 81.0% 46.3% 16.3%
Table 6:Valid evidence with an embedded injection.
High evidence accuracy argues against blanket refusal
or wholesale context ignoring.
RAG-framed training, the 7B model reaches 9.4%
average ASR with only 8.3 ppspread, outperform-
ing every 3B configuration in the table. Additional
epochs slightly worsen this cross-framing average,
which is why the main paper treats early stopping
as an important design choice when transfer is the
target metric rather than only native-channel perfor-
mance. Overall, the complete matrix reinforces the
main claim: wrapper-conditioned training changes
not only how much robustness is learned, but also
what kind of robustness is learned.
E Tool-output and agent-message
training
These runs test whether the TRACER recipe is
specific to RAG documents or whether the same
reward- and trace-regularized training objective canadapt to other high-authority channels. The an-
swer is yes, but the optimization dynamics differ
by wrapper. Tool outputs and agent messages both
present the attack as external system state rather
than as a direct user request, yet the resulting ro-
bustness profiles are not identical.
Figure 10:Tool-output training curve.Tool-output
training converges more gradually than RAG training
but ultimately reaches 15.0% native ASR.
Tool-output training.Table 9 and Figure 10
show that tool-output training improves more grad-
ually than RAG training. Native tool-output ASR
falls from 59.4% in the base model to 39.6% at the
end of epoch 1, 28.8% by late epoch 2, and 15.0%
at the final checkpoint. Clean accuracy simultane-
ously rises from 64.0% to 77.5%. Compared with
the sharper phase transition observed in RAG train-
ing, this smoother curve suggests that tool wrappers
encourage a broader but less sharply localized form
of skepticism: the model becomes progressively
less willing to treat tool-returned text as executable
instruction, rather than learning a single document-
specific counter-pattern.

Model Follow-up RAG Tool Agent Spread
Qwen2.5-3B-Instruct 41.4% 65.5% 59.4% 46.5% 24.1pp
Qwen2.5-7B-Instruct 9.5% 34.3% 36.6% 16.8% 27.1pp
Llama-3.1-8B-Instruct 17.8% 42.5% 39.0% 24.0% 24.7pp
Gemma-2-9B-It 12.0% 31.5% 34.0% 19.0% 22.0pp
Mistral-7B-Instruct 24.5% 49.0% 45.5% 30.5% 24.5pp
Table 7:Evaluation-only framing diagnostics across model families.Spread is the maximum minus minimum
ASR across channels.
Model Followup RAG Tool-Output Agent-Msg Spread
3B Base (no LoRA) 41.4% 65.5% 59.4% 46.5% 24.1pp
3B RAG-trained 14.5% 6.1% 25.7% 26.5% 20.4pp
3B Tool-trained (epoch 1) 32.1% 33.1% 39.6% 40.3% 8.2pp
3B Tool-trained (final) 20.0% 20.2% 15.0% 15.8% 5.2pp
3B Multi-framing (RAG+tool) 11.3% 5.2% 21.7% 17.5% 16.5pp
3B Agent-trained 19.6% 15.7% 26.3% 18.4% 10.6pp
7B Base (no LoRA) 9.5% 34.3% 36.6% 16.8% 27.1pp
7B RAG-trained (epoch 1) 5.2% 10.4% 13.5% 8.5% 8.3pp
7B RAG-trained (epoch 2) 5.7% 18.0% 19.2% 6.5% 13.5pp
7B RAG-trained (final) 5.1% 13.6% 18.2% 6.2% 13.1pp
Table 8:Complete cross-framing transfer matrix.Attack success rate (%) across training conditions and
evaluation framings. Spread is the maximum minus minimum ASR across the four framings for a given trained
model.
State Epoch Clean Robust ASR
Base – 64.0% 24.5% 59.4%
step-250 1 66.0% 32.5% 57.6%
step-1750 1 72.0% 52.0% 39.6%
step-2500 2 73.0% 58.5% 32.2%
step-2750 2 69.5% 60.5% 28.8%
final 3 77.5% 75.5% 15.0%
Table 9:Tool-output training trajectory.Three epochs
of tool-output training improve native tool-output ro-
bustness from 59.4% to 15.0%.
Why tool training transfers broadly.The cross-
framing numbers support that interpretation in Ta-
ble 10. Even before full convergence, the epoch-
1 tool checkpoint is already relatively flat across
follow-up, RAG, tool, and agent evaluation (32.1%,
33.1%, 39.6%, 40.3%), despite not yet being strong
in any single framing. After full training, the same
recipe reaches a near-uniform 15–20% ASR pro-
file. This is qualitatively different from the RAG
specialist, whose native ASR is much lower but
whose transfer is less even. One plausible explana-
tion is that tool outputs mix provenance, status, and
content fields in a way that forces the model to rely
less on any single authority cue and more on the
underlying semantic question: is this external text
evidence to reason over, or an instruction to obey?State Followup RAG Tool Agent Avg
Base 41.4% 65.5% 59.4% 46.5% 53.2%
Epoch 1 21.0% 30.3% 43.3% 31.9% 31.6%
Epoch 2 12.4% 21.8% 20.1% 23.4% 19.4%
Final 19.6% 15.7% 26.3% 18.4% 20.0%
Table 10:Agent-message training trajectory.Agent-
message training improves average ASR, but its
strongest final transfer is to RAG framing rather than
native agent framing.
Agent-message training.Agent-message train-
ing also improves robustness substantially, reduc-
ing average ASR from 53.2% in the base model
to 31.6% after epoch 1 and 19.4% after epoch 2.
The final epoch-3 checkpoint slightly worsens the
average to 20.0%, which is why we report it as
a specialized final model rather than as the best
checkpoint. Notably, the strongest final transfer is
to RAG framing (15.7%) rather than to native agent
framing (18.4%). This suggests that the model
treats agent messages and retrieved documents as
partially overlapping sources of delegated author-
ity. In other words, the wrapper taxonomy is not
arbitrary: some channels are closer than others in
the trust geometry learned by the model.
Takeaway.These results show that TRACER is
not just a RAG-specific defense. The same training
game, composite reward, and ATRC regularization

Condition Sys Wrapper ASR∆
Full RAG RAG 5.5% —
Wrapper only Default RAG 10.0% +4.5pp
Prompt only RAG None 11.3% +5.8pp
Neither Default None 60.5% +55.0pp
Table 11:RAG ablation.Both the RAG wrapper and
the RAG-specific system prompt matter. The wrapper
contributes slightly more, but the strongest robustness
appears only when both are present.
can be instantiated for multiple untrusted-context
channels. What changes from channel to chan-
nel is the learned robustness profile: RAG train-
ing gives deep specialization, tool training gives
the broadest transfer, and agent training lands be-
tween them. That pattern is exactly what the cross-
framing hypothesis would predict if wrappers affect
the model’s trust calibration rather than merely its
formatting preferences.
F Ablation and scaling
We use two compact studies to separate
TRACER’s ingredients from confounds. The first
ablation asks whether the RAG gain comes mainly
from the wrapper, from the RAG-specific system
prompt, or from their combination. The second
asks how the framing gap changes with model
scale and training duration.
Wrapper versus system prompt.Table 11 and
Figure 12 show that both components matter. Re-
moving the RAG-specific system prompt but keep-
ing the document wrapper increases ASR from
5.5% to 10.0%; removing the wrapper but keep-
ing the prompt increases ASR to 11.3%. Remov-
ing both raises ASR to 60.5%, close to the unde-
fended baseline. Two conclusions follow. First,
the gain is not reducible to prompt wording alone:
the wrapper structure itself carries crucial authority
cues. Second, the gain is not reducible to wrap-
per structure alone either: telling the model how
to treat retrieved content still adds a meaningful
margin. This complements the method claim in the
main paper. Channel-sampled exposure matters be-
cause the model must actually see the deployment
wrapper during training, while reward shaping and
ATRC matter because the model must learn what
behavior to preserve once that wrapper is present.
Scaling behavior.Table 12 and Figure 11 show
a favorable scaling trend. At 7B, the base model is
already substantially more robust than the 3B base,Model Followup RAG Tool Agent Avg
7B Base 9.5% 34.3% 36.6% 16.8% 24.3%
7B RAG ep1 5.2% 10.4% 13.5% 8.5% 9.4%
7B RAG ep2 5.7% 18.0% 19.2% 6.5% 12.3%
7B RAG final 5.1% 13.6% 18.2% 6.2% 10.8%
Table 12:7B scaling results.Larger models are sub-
stantially more robust before training and reach even
lower average ASR after framing-aware training.
Figure 11:7B scaling.Larger models begin from a
better robustness baseline and achieve their best cross-
framing average after a single epoch of RAG-framed
training.
especially under follow-up and agent framing. Yet
the framing gap does not disappear: RAG and tool
wrappers still raise ASR into the mid-30s. Training
closes much of this remaining gap quickly. After
a single epoch of RAG-framed training, the 7B
model reaches 9.4% average ASR across the four
framings, with 10.4% native RAG ASR and only
8.3ppspread overall. Later checkpoints remain
strong but do not improve this average, ending at
10.8% for the final model. This is why we interpret
7B scaling as evidence that larger models learn
contextual skepticism more readily, not as evidence
that scale alone solves the problem.
Implications for method design.The ablation
and scaling results together support three practi-
cal recommendations. First, deployment-matched
wrappers should be part of the training distribution
rather than treated as cosmetic formatting. Sec-
ond, when transfer across wrappers is the target,
model selection should use cross-framing averages
rather than only native-channel ASR; the best 7B
checkpoint by that criterion appears after one epoch
rather than at the end of training. Third, stronger
base models reduce the amount of channel-specific
training needed, but they do not remove the need
for channel-aware evaluation. Even at 7B, the dif-
ference between follow-up and retrieved-document

Figure 12:Wrapper vs. system prompt.The doc-
ument wrapper and the RAG-specific system prompt
provide complementary gains.
Figure 13:Cross-domain robustness transfer.
Framing-aware skepticism transfers across domains
more readily than task-specific clean accuracy.
vulnerability remains large enough that a single in-
channel robustness number would be misleading.
G Cross-domain robustness
We also evaluate framing-aware robustness outside
the GSM8K training domain. On RAG-framed
Natural Questions and TriviaQA, framing-aware
models maintain relatively low ASR, but clean ac-
curacy is low for 3B models, indicating that capa-
bility rather than robustness is the main bottleneck
in these cross-domain settings.
H Hyperparameters and reproducibility
Primary 3B configuration.We use Qwen2.5-
3B-Instruct with 4-bit QLoRA (rank 64, α= 128 ),
three epochs of training, batch size 1, gradient ac-
cumulation 8, group size 4, warmup ratio 0.1, and
learning rate 2×10−6. The KL coefficient is 0.10,
the PPO clip parameter is 0.2, the ATRC regular-
ization weight is 0.1, and the ATR/TCR blend pa-
rameter is0.5.Reward weights.The reward weights are
wans= 0.25 ,wprm= 0.25 ,wanchor = 0.20 ,
wcorr= 0.20, andw len= 0.10.
Compute notes.All models are trained with
vLLM-accelerated rollouts and LoRA-style adapta-
tion rather than full dense finetuning. We provide
the configuration needed to reproduce the exper-
iments, but do not report exact wall-clock mea-
surements for every run; we therefore answer the
checklist conservatively.
I Assets and licenses
The experiments rely on public assets cited in the
paper, including base models, datasets, and veri-
fier backbones. The submission package contains
only manuscript sources and figures; it does not
redistribute third-party model weights or dataset
contents. Subject to the licenses of the underlying
assets, training wrappers, evaluation scripts, and
configuration files are available through the project
repository linked in the abstract.
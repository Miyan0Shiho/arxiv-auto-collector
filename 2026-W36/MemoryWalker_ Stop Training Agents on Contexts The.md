# MemoryWalker: Stop Training Agents on Contexts They Never Saw

**Authors**: Zinco J, Xunjie Zhu, Shen Huang, Zhenyi Wang, Pengjun Xie, Jieping Ye

**Published**: 2026-09-01 08:01:27

**PDF URL**: [https://arxiv.org/pdf/2609.00865v1](https://arxiv.org/pdf/2609.00865v1)

## Abstract
Production agent harnesses such as Claude Code and Qwen-Agent compress context during rollout, but training under compression creates a conditioning problem: every eviction branches the effective history, so the learning object is a tree rather than a sequence. Existing linearizations either retain the rightmost path, causing time-travel leakage, or replay a depth-first traversal, causing train-inference mismatch. We introduce two exact, gradient-equivalent corrections: LogitTree, a segmented K-forward traversal, and a packed 4D attention mask. LogitTree requires K+1 backward passes; the 4D mask requires a custom kernel and white-box eviction records. We also propose SDCC (Self-Distillation for Conditioning Consistency), a single-backward-pass variational relaxation. At each eviction, it minimizes forward KL between the compressed student and a stop-gradient teacher on the reconstructed pre-eviction prefix. A residual per-junction KL of epsilon_KL gives an O(sqrt(epsilon_KL)) bound on the train-deployment total-variation gap. SDCC also applies to black-box harnesses. On seven web-search benchmarks with TC-RAG, AgentFold, MemexRL, Claude Code, and OpenCode, naive training inflates the train-rollout log-probability gap, especially on eviction-heavy batches. The exact methods stay at the no-compression floor, and SDCC substantially closes the gap, with lower logit drift and higher rollout rewards.

## Full Text


<!-- PDF content starts -->

2026-09-02
MemoryWalker: Stop Training Agents on Contexts They Never Saw
“YourMemory-CompressingHarness Makes Trainingand InferenceInconsistent”
Zinco J, Xunjie Zhu, Shen Huang, Zhenyi Wang, Pengjun Xie, Jieping Y e
T oken Foundry, Alibaba Group
Abstract
Productionagentharnesses, includingClaudeCodeandQwen-Agent, compressanagent’scontextduringroll-
outtomakelong-horizoninteractiontractable. Trainingpoliciesundersuchcompression,however,introduces
a fundamental conditioning problem: each compression eviction branches the effective interaction history,
making the object presented to the learning objective a treerather than a single sequence. Existing pipelines
typically linearize this tree in one of two ways: retaining only the rightmost root-to-leaf path , which causes
time-travel leakage , or replaying the full depth-first traversal , which induces a train–inference mismatch .
Thus, we introduce two exact conditioning-level corrections and prove their gradient equivalence: LogitTree ,
a segmented K-forward traversal of the logits tree, and an equivalent packed 4D attention mask . LogitTree
requires K+1backward passes, whereas the 4D formulation requires both a custom masked-attention kernel
andwhite-boxaccesstoevictionrecords. Toavoidthesecosts,wefurtherpropose SDCC(Self- Distillationfor
Conditioning Consistency), a training-friendly variational relaxation requiring only a single backward pass.
At each eviction junction, SDCC minimizes the forward KL divergence between the compressed student pol-
icy and a stop-gradient teacher evaluated on the reconstructed pre-eviction prefix. By Pinsker’s inequality,
a residual per-junction KL divergence of εKLyields an O(√εKL)bound on the resulting train–deployment
behavioral gap in total variation. Unlike the exact corrections, SDCC applies without modification to black-
box harnesses. We evaluate our framework with three white-box context editors—TC-RAG, AgentFold, and
MemexRL—and two black-box harnesses—Claude Code and OpenCode. Across seven web-search bench-
marks, naive compressed-stream training substantially inflates the train–rollout log-probability gap relative
to the no-compression floor, with the largest inflation on eviction-heavy batches. In contrast, both exact
corrections—LogitTree and the 4D attention mask—remain at the no-compression floor, while SDCC sub-
stantially closes the gap. These gains in conditioning consistency are accompanied by lower logit drift and
higher rollout rewards than naive compressed-stream training.
1 Introduction
Long-horizon agents , including deep-search assistants ( Xi et al. ,2025), coding copilots ( Wang et al. ,2025),
browser-based research agents ( Shi et al. ,2025), and multi-turn planners ( Huang et al. ,2024)—can accumulate
trajectories spanning tens to hundreds of thousands of tokens. Retaining the full interaction history is not only
computationally expensive, but can also impair decision making: ¶outdated observations, abandoned plans, and
failed attempts introduce substantial contextual noise ( Jiang et al. ,2025a);·state rollback can invalidate the ac-
tions and observations associated with a reverted branch ( Jiang et al. ,2025a); and¸critical goals and instructions
maybedilutedoroverlookedasthecontextgrows,exacerbatingthe lost-in-the-middle effect(Liuetal. ,2023;Chen
et al.,2026). Production agent harnesses therefore actively compress and rewrite interaction histories to maintain
compact, decision-relevant contexts. For example, Claude Code ( Anthropic ,2024) and Qwen-Agent ( Team,2024)
maintain bounded context windows and replace evicted prefixes with automatically generated summaries. Other
approaches,includingTC-RAG( Jiangetal. ,2025a),MemexRL( Wangetal. ,2026)andAgentFold( Yeetal.,2025;
Zhou et al. ,2025) introduce model-aware memory mechanisms that allow the agent to determine whether, when,
what and how to compress, offload, or retrieve information on demand.
Across these context-compression designs, a memoryeditor (hook) rewrites the interaction history into a bounded
live context before each decoding step. Although this improves inference-time tractability and decision quality,
it creates a subtle problem for post-training (SFT and RL): when the trainer recomputes the log probabilities of
rollouttokens, the contextunder whicha token wasoriginally generatedmay no longerbe available .
Training on the conditioning tree, not the sequence. This missing-context problem arises because harness-level
compressionbreaksthecorrespondencebetweentokenorderandconditioninghistory. Ateachevictionjunction,the
originalprefixisreplacedbyacompressedview: tokensprecedingtheeditweregeneratedundertheoriginalprefix,
1
arXiv:2609.00865v1  [cs.LG]  1 Sep 2026

whereas subsequent tokens are generated under its compressed replacement. Repeated edits therefore induce a tree
ofconditioninghistories ,ratherthanasinglesequencethatcanbereplayedconsistently,asillustratedinFigure1(a).
Existing pipelines ( Ye et al.,2025;Zhou et al. ,2025;Wang et al. ,2026;Jiang et al. ,2025a) typically serialize this
tree in one of two ways. ¶Training on the surviving compressed stream follows only the rightmost root-to-leaf
path, causing earlier tokens to be reevaluated as if they had been conditioned on summaries created only later—an
error we call time-travel leakage .·Training on the full raw trace instead follows a depth-first traversal , causing
later tokens to be reevaluated with evicted content that is unavailable during rollout and deployment—a train–
inferenceconditioningmismatch .In both cases, recomputing token logits under the resulting sequence assigns
tokens contexts different from those under which they were originally generated, producing substantial train–
inference logit inconsistency.
From the conditioning tree to exact solutions. To restore the context under which each rollout token was origi-
nally generated, we formulate training as a traversal of the conditioning tree and introduce two exact approaches to
walk the tree, gradient-equivalent constructions. ¶ LogitTree performs a segmented traversal, separately evaluat-
ingthe Kpre-evictionbranches, ·whereasapacked 4D attention mask encodesthesamebranchstructurewithin
a single forward pass. As shown in § 4, both constructions yield the same policy gradients and exactly recover the
correctrolloutconditioning. Thisexactness,however,comesatacomputationalorsystemscost: LogitTreerequires
(K+1) segmented model evaluations, whereas the packed formulation relies on a custom masked-attention kernel
that is difficult to integrate efficiently with fused attention implementations and is restricted to white-box harnesses
with access to eviction records. To recover conditioning consistency without explicitly replaying every branch of
the context tree, we further propose SDCC(Self- Distillation for Conditioning Consistency; § 5). Instead of exactly
reconstructing the conditioning context for every rollout token, SDCC trains the policy under the surviving com-
pressedcontexttomatchitsbehaviorundertheoriginalpre-evictioncontext. Specifically, ateachevictionjunction,
it minimizes the forward KL divergence from the compressed student policy to a stop-gradient teacher evaluated
under the reconstructed pre-eviction prefix. A wake–sleep argument determines the KL direction, and Pinsker’s
inequality converts a residual per-junction KL divergence of εKLinto an O(√εKL)bound on the corresponding
behavioral discrepancy in total variation. This relaxation requires only a single backward pass per trajectory and
extends to black-box harnesses without requiring white-box modification of their attention computation.
Contributions. Our contributions are fourfold:
•We formalize context compression training as a conditioning-inconsistency problem (§3). We prove that the
two common rollout compression— Naive-Compressed andNaive-Full —lead to time-travel leakage and train–
inference conditioning mismatch, respectively.
•We propose two exact, gradient-equivalent solutions to walk the context tree (§4). LogitTree explicitly
traverses the conditioning tree, while a packed 4D attention mask represents the same structure in a single
forward pass.
•We introduce SDCC, a training-eﬀicient relaxation (§5). SDCCdistillspolicybehaviorundertheoriginalpre-
eviction context into the surviving compressed context, requires only one backward pass, and generally applies
to white-box and black-box harnesses.
•We validate the framework across compression mechanisms and agent tasks (§6). Across three white-box
and two black-box memory editors, on seven web-search benchmarks, our methods reduce train–inference incon-
sistency and logit drift while improving rollout rewards over naive compressed-stream training.
2 Related Work
Harness context compression. Mostproductionlong-horizonagentharnessesincludeanexternaleditorthatmu-
tates the physical history before the next decoding step. Claude Code ( Anthropic ,2024) and Qwen-Agent ( Team,
2024) maintain a bounded context window and inject an auto-compact summary in place of the evicted prefix.
MemexRL ( Wang et al. ,2026) and MEMGPT ( Packer et al. ,2023) pages between the live context and an external
store. TC-RAG and StackPlanner ( Jiang et al. ,2025a;Zhang et al. ,2026a) manages context as an explicit stack
whose popevictstheoldestretrievalenvelope. AgentFoldandMem1( Yeetal.,2025;Zhouetal. ,2025)periodically
fold the prefix into a self-generated summary. A complementary line studies working-memory design for agents
operatingoverlongdocuments,retaininganswerableevidencewhilecompactingtherestofthecontext( Zhouetal. ,
2026). Aparallellinetrainsdedicatedcompressors,includingtoken-budgetcompressionwithsmallauxiliarymod-
els such as LLM-LINGUA ( Jiang et al. ,2023) and RECOMP ( Xu et al. ,2024), which act as scheduled deterministic
editors. Across these editor families ( Zhang et al. ,2023;Li et al.,2024;Xiao et al. ,2024;Borgeaud et al. ,2022),
thecontextavailableattrainingtimeneednotmatchtheliveviewunderwhichtokenwasoriginallydecoded. Thus,
context the model conditioned on while generating is not, what remains available when training.
Agent RL over edited contexts. Recently, policy-gradient RL ( Schulman et al. ,2017;Shao et al. ,2024;Zheng
et al.,2025) and their agentic extensions ( Jin et al.,2025;Chen et al. ,2025;Jiang et al. ,2025b) fine-tune LLMs on
rollout trajectories, typically treating the edited context as ordinary sequential data. Memory-R1 ( Yan et al. ,2026;
2

Zhangetal. ,2026a)andMemexRL( Wangetal. ,2026)gofurtherbymakingmemorymanagementitselflearnable:
the policy can offload spans to external storage and retrieve on demand, thereby optimizing the memory-editing
process. However, neither line of work asks which context the training gradient should condition on once the
rollout history has been rewritten . Relatednotionsoftrain–inferenceconsistencyaddressdifferentfailuremodes.
TITO (Gallouédec & Rasul ,2026), whose token-level scheme our LogitTree extends to trajectory trees, studies
token-level consistency—whether the token IDs observed during rollout are replayed identically during training—
but does not address harness-induced history rewriting. Our focus is orthogonal: the conditioning context itself
changes, which extends block-causal masking for long-context training ( Ding et al. ,2024;Reid et al. ,2024;Zhou
et al.,2025).
3 Pitfalls: T wo Default Ways to Train on an Edited Rollout
Every harness in § 2rewrites the agent’s context mid-rollout, posing one question: which context did each token
condition on, and does training match it? An edited rollout is therefore a trajectory tree , not a single sequence
that can be replayed consistently. Existing pipelines flatten it in one of two ways, which fail in opposite directions
yet violate the same token-level conditioning invariant.
3.1 Train–inference logits drift
We first define the train–inference logits drift that quantifies this mismatch. Fix a generated token yt; write crollout
t
for the context the policy held when it emitted ytandctrain
tfor the prefix under which the loss later scores it. The
rollout engine logs log Pθ(yt|crollout
t)at decode time and the training pass recomputes log Pθ(yt|ctrain
t)under the
sameweights; averagedovertheloss-carryingresponsetokens M,theirdisagreementisthe train–inference logits
drift(logdiffin §6):
logdiff =1
|M|X
t∈MlogPθ(yt|ctrain
t)−logPθ(yt|crollout
t). (1)
Sharing one set of weights makes drift immune to sampler staleness: it is either the small numerical gap between
the training and inference kernels when nothing is compressed, where the two contexts coincide by construction
and the statistic sits at its 0.014floor (§6); such residual differences can arise from floating-point computation and
optimization implementation details ( Zhang et al. ,2026b) — or a genuine ctrain
t̸=crollout
t. Drift is thus a test with a
calibrated zero, and the two pitfalls below fail it in opposite directions.
3.2 Setup: the live view and the trajectory tree
We model the agent as a ReAct-style policy ( Yao et al. ,2023) that generates each action from the physical history
Ht= (s0, a0, o0, . . . , s t−1, at−1, ot−1), where sk,ak, and okdenote the thought, tool call, and tool observation
at turn k, respectively. A memory-editing harness inserts an editor EDIT (·)into the rollout loop to perform dele-
tion (Wang et al. ,2026) or summarization ( Ye et al.,2025;Jiang et al. ,2025a). LetE≤tdenote the ordered edits
fired by step t. The policy conditions on the live context view as:
H′
t=VIEW t 
H≤t;E≤t
,E≤t=EDIT (H≤t), a t∼πθ(· | H′
t), (2)
where VIEW tapplies all edits fired up to step t, yielding the partially observable context available to the policy. At
the end of the rollout, the harness reports the finalcompressedwalk Hcomp. Crucially,
VIEW t 
H≤t;E≤t
| {z }
live view at decode of yt̸= Hcomp[:t]|{z}
prefix of final walk=VIEW T(H≤T;E≤T), (3)
whenever an edit fires after step t: the prefix on the right has been rewritten by edits that did not yet exist when yt
was decoded.
To see why, consider an eviction at position Jkthat removes or replaces a span Ek. Tokens generated before Jk
were conditioned on a context in which Ekremained visible, whereas subsequent generation proceeds from the
edited context in which Ekhas been removed or replaced. Each edit therefore forks the conditioning history into a
generation-timeleg ,whichpreservesthepre-editcontext,anda compressedspine ,fromwhichtherolloutcontinues.
Unrolling Ksuch evictions at positions J1<···< J K, removing spans E1, . . . , E K, yields a trajectory tree T
rooted at the initial context. Every ytis a leaf of this tree: pt(yt)denotes its root-to-leaf generation path, with
pt(yt) = H′
tby definition, while ps(yt) = Hcomp[:t]denotes its corresponding prefix in the final compressed walk.
Figure1illustrates a running example with E1,2,3={y5},{y8},{y11}atJ1,2,3={7,10,12}; the corresponding
per-leaf prefixes are reported in Table 4and Appendix B.4.
3.3 The two pitfalls and the conditioning invariant
Two sequence representations of Tare natural, and neither preserves the generation-time path of every token: ¶
the final compressed walk Hcompkeeps only the compressed spine, ·the full physical trace HTconcatenates every
3

(a) Trajectory tree and two flattenings ( K=3 junctions).
Trajectory tree T: one row per live context H′
t; editJkdeletes a columnrollout timesy1y2y3y4y5y6y7prompt
older contextdecoded here
decoded, evicted next
yjno box: gone from the live context J1=7: evict y5
sy1y2y3y4y5y6y7y8y9y10
J2=10: evict y8
sy1y2y3y4y5y6y7y8y9y10y11y12
J3=12: evict y11
sy1y2y3y4y5y6y7y8y9y10y11y12yans
Pitfall A: Naive-Compressed : scoreseverytoken on the last row Hcomp— tooshort
sy1y2y3y4y5y6y7y8y9y10y11y12yans
y6, y7scored as if y5had never existed
Pitfall B: Naive-Full : revives every evicted column, tape HT— toolong
sy1y2y3y4y5y6y7y8y9y10y11y12yans
y9, y10still attend to
y5, dead since J1(b) Per-token ∆logpon untrained Qwen3-4B, three
harnesses on one axis.
Figure 1: The two pitfalls. (a) The trajectory tree Tof the K=3running example, drawn as the staircase of livecontexts : row
klists every token visible during layer k, so a token decoded there conditions on exactly what lies to its left in its own row, i.e.
onitsliveview H′
t= pt(yt)(boxfillsarekeyedinthelegend). Betweentworowsaneditfires— E1={y5}atJ1=7,E2={y8}
atJ2=10,E3={y11}atJ3=12(orange arrows) — so every later row carries a hole at the evicted column. The two default
flattenings collapse this staircase in opposite directions: strip A scores every token on the last row alone, the walk Hcomp, so
y6, y7are scored on a prefix that already skips y5(no box at that column); strip B revives every evicted column into the tape
HT, leaving y5, y8, y11visible to tokens decoded long after their eviction (red arc). (b)Train–inference logit inconsistency for
each generated token of an untrained Qwen3-4B, overlaid across three white-box harnesses. ∆comp(dashed, left tail) is the logit
difference when training replays a token using the final compressed context, which is missing information present at inference;
∆full(solid, right tail) is the difference when training replays the full physical trace, which contains information unavailable at
inference. Their per-harness means have opposite signs, µcomp∈[−4.0,−1.6]versus µfull∈[+0.2,+0.7]; full aggregation is
in Table 7(§F.2).
generated token in decoding order, a depth-first traversal. The former applies edits too early, the latter keeps edited-
away content too long.
¶ Pitfall A: Naive-Compressed follows the final compressed path (time-travel leakage). Naive-Compressed
trains directly on Hcomp. Consequently, each surviving target ytis scored under its prefix in the final compressed
walk, ps(yt), rather than under the live view pt(yt) = H′
tfrom which it was generated:
ps(yt)( pt(yt) = H′
t=ROOTTORIGHTMOSTLEAF (T). (4)
The mismatch arises because ps(yt)may already reflect edits that fired only after ytwas decoded. Future compres-
sion is thus propagated backward in time, a failure mode we call time-travel leakage . In the running example, this
scores y6, y7as ify5never existed — though the model conditioned on y5. On an untrained Qwen3-4B, one such
leaf contributes ∆comp=−22.8nats to the per-token log-probability gap; SFT/RL views appear in Appendix B.4.
· Pitfall B: Naive-Full trains on the depth-first traversal (train–inference mismatch). Naive-Full instead
trains on HT, which preserves every generated token in decoding order. A target generated after an eviction is
therefore reevaluated under the physical prefix H<t, which may still contain spans that had already been removed
from the live view:
DEPTHFIRSTTRAVERSAL (T) = H<t) pt(yt)⊇ ps(yt). (5)
Thus, y9, y10are scored with y5, y8still in the prefix even though each was already evicted; the model learns to
exploit signal unavailable at deployment, with credit-assignment error accumulating linearly with K(App.B.4,
Eq.14). A matched leaf contributes ∆full= +18 .5nats: opposite in sign to Pitfall A, with comparable magnitude.
Werefertothisfailureas stale-contextleakage : trainingexposesthemodeltoinformationunavailableduringrollout
and deployment.
¸ Mirror symmetry and the conditioning invariant. Pitfall A conditions on too little context, whereas Pit-
fall B conditions on too much (Table 5, App.B.4). Accordingly, on an untrained Qwen3-4B, ∆compis predomi-
nantlynegativeand ∆fullpredominantlypositiveacrossthreewhite-boxharnesses(Figure 1). Undereviction-heavy
compression, the Naive-Compressed train–inference log-probability gap reaches 0.366on AgentFold, ≈26×the
no-compression floor of 0.014(§6). Both failures violate the same requirement. In the notation of § 3.1, the loss
prefix is ctrain
t= ps(yt)for Pitfall A and H<tfor Pitfall B, against crollout
t = H′
tin both cases.
Definition 1 (Conditioning consistency) .A training pipeline is conditioning-consistent if, at every step tin every
trajectory, using only the edits fired through step t, never future ones:
∀t, ctrain
t≡crollout
t≡VIEW t 
H≤t;E≤t
= H′
t, (ConditioningInvariant )
4

(a) Trajectory tree T
Each edit forks the trajectory.
shared trunk
sy1y2y3y4y5y6y7
y6y7y8y9y10
y9y10y11y12
y12yansJ1=7
J2=10
J3=12branch 0
branch 1
branch 2
branch 3 = Hcomp
orange dot: fork; green: decoded/kept; red dash: evicted
grey: re-shown; green underline: deployed walk Hcomp(b) LogitTree
Walk the K+1paths separately.
fwd 0
fwd 1
fwd 2
fwd 3s
s
s
sy1:4
y1:4
y1:4
y1:4y5y6y7
y6y7y8y9y10
y6y7 y9y10y11y12
y6y7 y9y10 y12yansscored here, evicted at next junction
evicted earlier: not on this branch
P
π 0 1 1 1 1 1 1 1 1 1 1
green =scored on this branch; grey =carried, loss-masked
theK+1blocks tile the response:P
π=1per token, never twice
cost:K+1forward and K+1backward passes
(c) 4D mask
Pack the same tree into one forward pass.
14tokens, nothing replicateds
s
y1
y1
y2
y2
y3
y3
y4
y4
y5
y5
y6
y6
y7
y7
y8
y8
y9
y9
y10
y10
y11
y11
y12
y12
yans
yansfwd 0
fwd 1
fwd 2
fwd 3tree of (a)
rowiadmits exactly the keys live when yiwas decoded;
an evicted column (dashed, red) drops out for all later rows
cost: 1 forward, 1 backward +a custom mask kernel(d) SDCC
One walk, pulled toward the teacher legs.
teacher legs (stop-grad) (Eq. 9), with ιthe slot Ekwas cut from:
pt(yp) = Hcomp[:ι]⊕Ek⊕ Hcomp[ι:p]
leg 2
leg 1
leg 0
students y1:4 y6y7y9y10y11y12
s y1:4 y6y7y8y9y10
s y1:4 y5y6y7
s y1:4 y6y7y9y10y12yansDKLleg 0:Ptarget
s y1:4y5y6y7
re-inserted
DKL
evicted
s y1:4 ∅y6y7
student: Pθ
read at y6; two
prefixes differ by y5
legk=branch k; student =branch 3; rebuilt from Hcomp;
dashed red =reinserted span; solid red =DKLreadout
cost: 1 backward; the Klegs are no-grad and share the trunk KV
Figure 2: From the trajectory tree to three consistent materializations. (a) A memory editor turns one rollout into a tree,
not a sequence: evicting Ekat junction Jkforks the prefix, because the pre-eviction path keeps the evicted token while the
post-eviction path skips it, giving K+1root-to-leaf branches over a shared trunk whose bottom-most path is the compressed
walk Hcompshipped at deployment. Every token is decoded on its own root-to-leaf prefix pt(yt) = H′
t; the two pitfalls of
Figure1are what happens when training instead scores it on some other path. (b)LogitTree (ours, §4) restores the invariant
by literally walking the K+1branches, one forward each, with the loss masked to the tokens decoded on that branch; the∑
π
strip tallies that mask, reading 1on every response column — including the trunk tokens y1:4, replicated K+1times — so no
token contributes its policy gradient twice. (c)The4Dpackedmask (ours, §4) folds the same tree into a single forward over the
physical union HTand recovers each branch through the attention mask: the row bands are the tree’s legs (right). Its gradient is
identicalto(b)(Theorem 1).(d)SDCC(ours,§5)keepsthesinglecompressedforwardasthestudent,andrebuildstheotherlegs
as stop-gradient teachers by re-inserting each evicted span at its junction (Eq. 9); a forward KL at the diverging leaves only then
pulls the student’s next-token distribution toward the teacher’s. The inset isolates one such leaf: the two conditioning prefixes
differbyexactlytheevictedspan,andtheKListhegapthisopensbetweenthetwonext-tokendistributions. Onebackwardpass,
no kernel, and an O(√εKL)residual (Proposition 2) instead of exactness.
Proposition 1 (Failure of naive training) .Hcomp[:t]yields Pitfall A, while H<tyields Pitfall B; neither matches
the live view H′
tfrom which ytwas decoded. Whenever a nontrivial edit changes the prefix used to evaluate some
target yt,bothNaive-CompressedandNaive-Fullviolatetheconditioninginvariant. § 4givestwoexact(bias-zero)
fixes, while § 5givesa soft O(√εKL)one.
4 Exact Solutions: T wo Ways to Walk the Context Tree
To eliminate the conditioning errors introduced by the two pitfalls, we propose two exact solutions that enforce
conditioning-invariance. Across the K+1branches of the conditioning tree T, each target ytis scored on the
unique branch whose prefix matches its inference-time live view, pt(yt) = H′
t(App.B). Both methods therefore
recover the correct training context for every target and incur zero conditioning bias. ¶LogitTree (§4.1) explicitly
materializes and evaluates the branches as separate sequences, ·whereas the 4D attention mask (§4.2) packs the
same branches into a single forward pass. As established in Theorem 1, they implement the same tree traversal at
different levels of the training stack (Figure 2).
4.1¶LogitTree materializes branches as separate sequences
LogitTree (ours; extending the token-level scheme of ( Gallouédec & Rasul ,2026) to trajectory trees) decomposes
Tinto its K+1root-to-leaf branches and processes each branch as a separate sequence:
{H′π}=ROOTTOLEAF (T), Logitsπ=fθ({H′π}), π ∈ {1, . . . , K + 1}, (6)
5

where {H′π}denotes the sequence on branch πand logits is extracted from LLMs f(·)parameterized by θ. For
each target yt, LogitTree selects the unique branch whose prefix equals the live view under which ytwas generated,
pt(yt) = H′
t. The loss for ytis therefore computed from that branch only, so its training-time context exactly
matches its rollout-time context. In implementation this uniqueness is enforced by a per-branch tokenlossmask : a
token is unmasked on exactly one of the K+1branches — the one whose prefix is its live view — and masked out
at every other occurrence, so each token contributes its policy gradient exactly once and the shared-trunk tokens,
which are physically replicated across all branches, are never counted K+1times.
LogitTree places the K+1legs in physically disjoint sequences, so standard causal attention restricts each target
ytto its predecessors on the same leg—namely, pt(yt)—with no attention across legs. In the running example
(J1,2,3= 7,10,12), the pre- J1leg containing y6andy7retains y5, which was visible when they were decoded. In
contrast, the compressed-spine leg containing yansexcludes y5,y8, and y11, all of which had already been evicted.
Thus, neither time-travelleakagenor stale-contextleakagehasa validattention pathway. Shared-trunk KV caching
keeps aggregate memory well below (K+1)×; compute stays ≈(K+1)×. Per-branch SFT/RL gradient equiva-
lences are in App. C(Proposition 3; the branch layout for the running example is drawn in Figure 2(b)); the con-
struction is backbone-agnostic — it consumes only the per-leg live views the harness already exposes (App. E.2).
4.2·The 4D attention mask packs all branches into one sequence
Performing K+1backward passes per rollout is prohibitively expensive in deep-search settings. We therefore ask
whether the entire conditioning tree can be traversed in a single masked forward pass. Our realization follows the
4D attention-mask formulation used for stack-memory agent trajectories in AgenticRag-R1 ( Jiang et al. ,2026): we
keep HTpacked as one sequence while restricting each query token to attend only to tokens on its own root-to-leaf
branch. Let
τ(j), ω(j)
be token j’s logical lifetime ( ω(j) = +∞if never evicted), we define
Mlogical
i,j =0, τ (j)≤i < ω (j),
−∞,otherwise,Mi,j=Mcausal
i,j⊕Mlogical
i,j, (7)
Thecausalmask Mcausalpreventsattentiontofuturetokens, while Mlogicalfurtherremovesattentionedgesbetween
different branches of the conditioning tree. We also reassign position ids so that the tokens visible to each query
form a gap-free logical sequence ( Zhou et al. ,2025). As a result, each query yiattends to exactly the root-to-leaf
pathof Tunderwhichitwasgenerated, allowingallbranchestobeevaluatedinasingleforwardpass(Figure 2(c)).
Throughout, both materializations share one standing conditioning invariant proposition — and gradient equiva-
lence(Proposition 4,App.C),whichholdsunderfiveconditions: (i)exact-visibilityencoding,(ii)row-wiseposition
reassignment, (iii) decoupled loss/attention masks, (iv) no cross-mask normalization, and (v) rollout and training
sharingthesame Mlogical. However,linearattention,sparse/top- kselection,andvLLM/SGLangviolatethepremise
or one of (i)–(v) (App. E.2).
4.3¸The centerpiece: 4D masks and LogitTree are the same walk
Theorem 1 (Materialization equivalence) .Under dense softmax attention, the packed 4D mask and the LogitTree
K-forward decomposition implement the same traversal of the conditioning tree and compute identical training
gradients. Bothcomputethesum overbranchesof theper-branchgradientin Proposition 3(App.C).
Cost and deployment splits. LogitTree requires K+1backward passes per rollout, resulting in a 5–20×wall-
clockoverheadindeep-searchsettings,whereasthe4Dformulationrequiresonlyone. The4Dformulation,however,
imposes a stronger systems requirement: it requires more larger attention-mask and white-box access to both the
harness and the model. Implementing equation 7requires the harness to expose the eviction records that determine
[τ(j), ω(j)), as well as the model to accept a custom (Batchsize, NumHeads, T, T )attention mask. Neither
capability is generally available through black-box APIs such as Claude Code or OpenCode. Moreover, construct-
ing and applying a per-query 4D mask has high memory and computational complexity, making it particularly
unfriendly for training with very long contexts. By contrast, LogitTree requires only the live views exposed during
rollout and can reconstruct the K+1branch inputs by concatenating the corresponding per-leg prompts. Its main
limitation is therefore computational rather than architectural. The concrete 4D realization used in § 6is deferred
to App.E.2, a packaging within the equivalence class of equation 7that inherits Theorem 1.
5 SDCC: Self-Distillation for Conditioning Consistency
A compressed rollout leaves two trajectories of the same interaction : thelive rollout trajectory , whose contexts
produced each token, and the finalcompressedreplaytrajectory Hcomp, which remains after all edits and is used by
Naive-Compressed for training. They agree until a later edit rewrites a prefix. SDCC aligns their next-token dis-
tributions at exactly those divergences: the student under the compressed replay prefix matches a stopped-gradient
teacher under the original live prefix. It aligns conditional policy distributions rather than the two text sequences.
6

The exact methods in Section 4achieve this alignment by scoring every target on the branch on which it was
generated. They are exact, but require either K+1backward passes or a custom attention kernel. Self-Distillation
for Conditioning Consistency (SDCC) is the cheaper alternative: it retains the usual single backward pass on the
compressed walk and aligns its predictions with the original live trajectory only at the diverging leaves. SDCC is
therefore an approximation to the exact walks, not another way to materialize the tree.
5.1 Aligning the live and compressed trajectories
For a diverging response token yp, write
zp:= pt(yp), x p:= ps(yp)(zp, (8)
where zpis the live prefix from which ypwas originally decoded and xpis its prefix in the final compressed walk.
We call ypadivergingleaf , and let Ccollect all such leaves in a trajectory; each pair (zp, xp)is one local alignment
pair. In the running example, if y5is evicted only after y6was decoded, then z6contains y5whereas x6does not.
Naive-Compressed trains y6onx6; the exact methods would instead score it on z6.
SDCC does not put zpback into the gradient-carrying training sequence. Instead, it uses the policy evaluated on
zpas a stopped-gradient teacher, and the policy evaluated on xpas the gradient-carrying student. Here “teacher”
does not mean a second model: it is the current policy with gradients stopped. Thus, at a diverging leaf, SDCC
teachesthecompressed-contextpolicytoreproducethenext-tokendistributionithadundertheoriginallivecontext
(Figure2(d)).
The procedure has three steps. First, run the ordinary task forward pass on the final compressed walk Hcomp; this is
the student pass and is the only pass through which gradients flow. Second, reconstruct the original live prefix for
each diverging leaf and evaluate it without gradients. For the common case in which one eviction Eact(p)is active
atp, with ι(Eact(p))its original insertion slot, the reconstruction is
pt(yp) = Hcomp[:ι(Eact(p))]| {z }
shared trunk⊕Eact(p)|{z}
re-inserted span⊕Hcomp[ι(Eact(p)) :p]| {z }
post-junction suffix. (9)
With overlapping evictions, SDCC re-inserts all spans active at p; Appendix D.1gives the general construction.
Third, add a KL penalty only at those diverging leaves:
LSDCC = Ltask(θ;Hcomp)|{z }
ordinary compressed-stream task loss+λX
p∈CDKL0
B@Ptarget(· | pt(yp))| {z }
live-context teacher; stop-gradPθ(· | ps(yp))p|{z }
compressed-context student; gradient1
CA,(10)
where Ptarget(· | pt(yp)) = fsg(θ)(pt(yp))andPθ(· | ps(yp))p=fθ(Hcomp)p. The KL is forward: the teacher
distribution appears first, so it acts as a fixed soft target and requires the student to cover outcomes that are likely
under the original live context.
Thisconstructionmakesthecomputationaltrade-offexplicit. ThestudentusesthesamecompressedinputasNaive-
Compressed, and the teacher legs are gradient-free; SDCC therefore needs one backward pass per trajectory. The
penalty is leaf-gated : if no context differs, Cis empty and the loss is exactly the usual task loss. Likewise, setting
λ= 0recovers Naive-Compressed exactly, making it SDCC’s matched baseline. Because HTnever enters the
student input, SDCC does not introduce Naive-Full’s stale-context leakage. It instead softens Naive-Compressed’s
time-travel error by matching distributions rather than by exactly replaying every branch.
The variational derivation of the forward direction, including its wake–sleep interpretation and the role of λ, is
deferred to Appendix D. In brief, the forward KL has the same zero set as the conditioning gap induced by the
variational objective, while giving a stopped-gradient cross-entropy update for the student.
5.2 What the residual KL guarantees
Letεp:=DKL(Ptarget(· | pt(yp))∥Pθ(· | ps(yp))p)be the residual left by Equation ( 10) at diverging leaf p.
Proposition 2 (Behavioral Pinsker bound on the conditioning gap) .Forevery divergingleaf p,
∥πθ(· | pt(yp))−πθ(· | ps(yp))∥TV≤q
εp/2.
Thus the regularizer controls a behavioral quantity, not merely a logit diagnostic: the smaller the teacher–student
KL at a junction, the smaller the difference between their next-action distributions. If εp= 0, SDCC matches the
exact walk’s next-token distribution at that leaf; for nonzero residual it remains an explicitly bounded approxima-
tion. Appendix Dproves the proposition and further gives the variational derivation, a policy-gradient-bias bound,
conditions for a zero-KL solution, and the convergence analysis.
7

6 Experiments
Wevalidatetheproposedconditioning-inconsistencyframeworkacrossmultiplemodelscales,white-boxandblack-
box agent harnesses, and multiple logit recomputation schemes. Across these settings, we consistently observe
that harness-level context editing introduces a measurable train–inference mismatch, which manifests as elevated
logit drift and degraded rollout reward under naive training. In contrast, conditioning-consistent recomputation
substantially reduces this mismatch and improves downstream agent performance.
6.1 Experimental Setup
We validate conditioning inconsistency across different backbone models, agent harnesses, and training-time logit
recomputation schemes.
¶Backbone Models. We use Qwen-family models, with Qwen3-4B and Qwen3.7-Air as the main backbones.
·Harnesses. Forwhite-boxcontexteditors, weusethreerepresentativelearnable-actioneditors: TC-RA G (Jiang
et al.,2025a), which emits pop();AgentFold (Ye et al. ,2025), which folds spans past a length threshold; and
MemexRL (Wang et al. ,2026), which offloads spans to an external store. A Search-R1 no-compression control
emits tool calls but never evicts, anchoring the no-compression drift floor. Knobs and per-method inputs are given
in §F.1–F.3. For black-box production-style harnesses, we instrument their HTTPS traffic, recover the per-turn
rendered context prefixes, and construct the corresponding LogitTree views from these observed prefix states. We
evaluate two such harnesses: Claude Code and OpenCode .
¸ Logit Recomputation Schemes. We compare replay on the final compressed stream ( Naive-Compressed ),
replay on the full physical trace ( Naive-Full ), two exact conditioning-consistent replay methods ( 4D-Mask and
LogitTree ), and our proposed single-backward approximation SDCC.
¹ Datasets. Train:a pooled composite-QA corpus of 81,638instances drawn from REDSEARCHER ( Chu et al. ,
2026) and the ASEARCHER agentic-search training set, whose multi-hop structure reliably induces long-horizon
interaction and frequent context editing. Test:NQ (Kwiatkowski et al. ,2019), TRIVIAQA ( Joshi et al. ,2017),
HOTPOTQA ( Yang et al. ,2018), 2WIKI ( Ho et al. ,2020), MUSIQUE ( Trivedi et al. ,2022), BAMBOOGLE ( Press
et al.,2023), and FRAMES ( Krishna et al. ,2025) (38,270questions per checkpoint, using the same live pipeline
as training).
ºSearch Infrastructure. All rollouts use live web tools rather than a local retrieval server. Web search is backed
by the online DashScope text-search API, which returns up to 10 ranked results per query. For full-page evidence
extraction, we use a two-stage pipeline: retrieved pages are first fetched via Firecrawl and then summarized by
Qwen3-Turbo.
» Evaluation Metrics. We report the following quantities, each with a distinct role. (1) Recorded-token logdiff
=logπtrain(y|ctrain)−logπrollout(y|crollout)is a per-token conditioning-fidelity diagnostic rather than an efficacy
metric; (2) Full-distribution conditioning KL KL(π(·|Hteacher)∥π(·|Hstudent))is the population-level counter-
part, used in § 6.3to measure the direction and magnitude of the conditioning mismatch. (3) SDCC training KL
KLSDCC(§6.2, §F.6) is SDCC’s leaf-gated consistency regularizer, and is not intended as a cross-method compara-
tor. (4) Rollout reward measuresthebehavioralconsequenceofconditioningmismatchduringtrainingandrollout.
(5) Compression depth depth =E[ci|ci>0], forcithe fraction of rendered context evicted in rollout episode i,
is how much an editor removes when it fires. Its companion factor freq =Pr[ci>0]is tabulated per cell in § F.6
instead of here: what triggers a firing is harness-specific, so freq compares down a column, not across. (6) Agentic
EMis the primary downstream task metric and determines the final correctness ranking in § 6.4. Full definitions
are given in § F.
¼Optimization. All cells are trained with GRPO: group size G= 16, learning rate 10−6, KL coefficient βKL=
10−3, and a live-compression rollout pipeline, so training and deployment use the same rollout setting. Each cell
retains at least 400steps; a frozen dev split of 1,325questions, disjoint from every test benchmark, shortlists
candidatesforfull-suiteevaluation,andeachreportedEMisthebestfull-suiteresultamongthecell’sfullyevaluated
checkpoints (§ F).
6.2 Grand result matrix (overview)
Tree-consistent methods recover the drift floor; the two Naives inflate with eviction. Under live model-
triggered compression ( memory_offload ,popemitted by the policy; AgentFold folds when the rendered context
is>3ktokens), the no-compression controlsits at logdiff = 0.014. The discriminatingquantityis the slopeagainst
evictionfrequencyratherthanthelevel: whereacellbarelyevicts,everymethodsitsatthefloor,baselinesincluded.
From each method’s least- to its most-evicting harness, Naive-Compressed moves 30.5×(0.012→0.366for freq
2.3→28.6%) and Naive-Full 9.7×(0.021→0.203, freq 6.5→41.5%), whereas 4D spans a comparably wide
frequency range ( 3.8→49.2%) and moves only 3.7×, ending at 0.022; SDCC moves 6.5×to0.071and LogitTree
8

Table 1: Grand result matrix (Qwen3-4B, live compression, real web tools). Five training methods across three
white-box editors, plus two black-box deployed agents. logdiffis measured at each cell’s maximum-reward rollout
iteration, as the mean and max within that iteration rather than over the run. Per-cell compression frequencies
are in §F.6. For the two black-box harnesses, context management is internal and eviction spans are not exposed,
so compression depth is N/A; their observed maximum context lengths are reported, and junctions are recovered
indirectly (§ E.2).
reward ↑ logdiff ↓ compression Agentic EM per bench (%) ↑
Method Harness max mean max depth (%) max tok. NQ TQA HQA 2Wiki MSQ Bamb. Frames avg EM ↑
Qwen3-4B(without RL)
No-CompressionBase N/A N/A N/A 0.0 289 3.2 21.6 8.4 2.6 2.5 23.2 8.0 9.9
Search-R1 N/A N/A N/A 0.0 1,806 2.3 22.2 8.7 2.5 2.7 18.4 8.4 9.3
CompressionTC-RAG N/A N/A N/A 2.00 2,117 3.1 23.2 9.2 3.1 3.9 16.0 9.7 9.7
AgentFold N/A N/A N/A 4.78 1,971 2.0 21.3 7.3 2.5 2.9 13.6 6.1 7.9
MemexRL N/A N/A N/A 12.58 1,670 1.2 15.8 5.5 1.1 1.7 12.0 7.3 6.4
Claude Code N/A N/A N/A N/A 33,804 22.1 50.4 23.2 29.8 7.4 37.6 13.7 26.3
OpenCode N/A N/A N/A N/A 34,285 31.0 55.7 26.6 32.7 8.4 36.6 12.1 29.0
Qwen3-4B(with RL)
No-compression Search-R1 0.697 0.0140.015 0.0 36,370 19.8 57.6 19.9 11.1 7.9 32.0 12.7 23.0
Naive-FullTC-RAG 0.838 0.197 0.335 60.5 37,044 30.1 64.4 28.2 34.1 12.9 52.0 22.8 34.9
AgentFold 0.600 0.203 0.333 78.2 14,190 25.6 58.8 24.0 18.7 9.1 40.8 17.1 27.7
MemexRL 0.734 0.021 0.051 43.2 65,630 29.2 63.1 28.0 31.3 9.7 48.0 15.2 32.1
Naive-CompressedTC-RAG 0.850 0.015 0.041 54.3 38,171 30.9 64.3 31.9 46.0 11.8 50.4 21.5 36.7
AgentFold 0.734 0.366 1.095 68.4 16,708 30.3 64.1 25.4 24.8 12.6 56.0 18.9 33.2
MemexRL 0.750 0.012 0.014 55.0 37,101 23.9 60.5 22.9 20.5 9.7 47.2 17.6 28.9
4D mask (ours)TC-RAG 0.800 0.006 0.013 44.6 40,182 29.4 60.5 30.2 44.9 13.0 58.4 23.7 37.2
AgentFold 0.632 0.022 0.091 79.7 15,383 33.4 66.0 31.9 43.9 11.1 37.6 15.8 34.2
MemexRL 0.416 0.013 0.017 45.4 36,528 33.7 65.9 31.2 36.2 11.7 40.8 14.3 33.4
LogitTree (ours)TC-RAG 0.914 0.012 0.013 53.0 39,743 36.2 65.6 36.1 54.1 13.9 55.2 19.3 40.1
AgentFold 0.629 0.012 0.018 83.3 16,595 34.2 67.9 34.7 46.8 12.5 46.4 19.8 37.5
MemexRL 0.715 0.013 0.013 64.1 40,485 37.169.1 42.0 61.5 19.1 64.0 28.5 45.9
Claude Code 0.839 0.016 0.016 N/A 65,167 34.8 37.0 26.5 30.8 21.7 58.8 42.0 35.9
OpenCode 0.929 0.012 0.012 N/A 73,589 34.5 63.2 33.7 39.2 16.0 42.8 15.9 35.0
SDCC (ours)TC-RAG 0.922 0.013 0.015 57.8 37,590 37.8 70.2 42.9 63.8 17.7 60.2 29.6 46.0
AgentFold 0.713 0.071 0.156 73.3 9,416 33.9 67.1 39.7 57.9 14.4 58.4 25.1 42.4
MemexRL 0.750 0.011 0.012 50.3 36,364 37.0 69.239.5 56.9 15.6 59.2 24.5 43.1
Claude Code 0.648 0.015 0.015 N/A 82,779 37.2 67.7 34.1 40.7 20.8 42.8 19.5 37.5
OpenCode 0.717 0.013 0.013 N/A 76,807 32.7 62.3 35.4 41.5 17.0 47.7 21.5 36.9
is flat at 0.012–0.013. In the two non-folding editors all six tree-consistent cells sit at 0.006–0.013, at or below the
control (§ 3.3). Table 1reports the full training matrix. N/A = undefined. rewardand logdiffare read at each cell’s
argmax-rewarditerationandcompressionatrunlevel, byonerecipeappliedtoeverycell. Semantics: § F.4. Forthe
untrained seven-benchmark diagnostic, compression depth is the mean fraction removed conditional on an editor
firing. “Max tok.” is the largest single-request input length observed during rollout, rendered with the same chat
template and tool schema used by the adapter. The Base and Search-R1 controls have depth 0.0by construction.
Structural pattern of Table 1.Exactwalksrecoverthefloor(LogitTree 0.012–0.013,4D0.006–0.022;No-comp
anchor 0.014). The Naives separate on the same axis: Naive-Compressed reaches 0.366on AgentFold, ≈26×the
floor. Where a 4D maximum exceeds LogitTree’s it does so at the floor itself ( 0.017vs.0.013on MemexRL),
which does not contradict Theorem 1: the theorem equates gradients on the same batch , whereas each row is an
independentlytrainedrun;thetwo means—thequantitytheinvariantconstrains—agreeat 0.013. SDCC’sKL SDCC
scaleswithteacher–studentmismatch,spanningthreeordersofmagnitudeacrossharnesses( 10−4onMemexRLup
to0.356on AgentFold). Every white-box trained row exceeds the Search-R1 no-compression control at EM 23.0
(vs. untrained-Base 9.9). For the black-box harnesses, SDCC reaches 37.5on Claude Code and 36.9on OpenCode.
6.3 Q1: The two pitfalls fail in opposite directions
Both pitfalls exhibit their predicted signs on an untrained model. On an untrained Qwen3-4B, Naive-Comp
under-places ( ∆comp<0) and Naive-Full-leak over-places ( ∆full>0) across all three white-box harnesses, with the
predicted sign holding for a majority of tokens in every cell — structural rather than RL-absorbable. Per token,
∆comp=logp(yt|Hcomp[:t])−logp(yt|pt(yt))(diverging leaves) and ∆full=logp(yt|cleak
t)−logp(yt|Hcomp[:t])
(post-eviction; cleak
treinjects Naive-Full’s training-only content); magnitudes are ranked TC-RAG ≈AgentFold ≫
MemexRL for ∆compand are reversed for ∆full, interpreted at the sign level only. Figure 1(b); §F.2.
9

6.4 Q2: SDCC ≈LogitTree ≈4D≫ both Naives
On the anchor cell, the exact walks pin the floor and SDCC’s logdiff decreases over the logged window. On
the low-eviction MemexRL anchor at 4B, both exact walks sit at the no-comp floor by construction (LogitTree
0.0133, 4D 0.0140vs. Search-R1 0.0135; Naive-Comp 0.0237; cost multipliers in Table 9). On the eviction-heavy
AgentFold cell, SDCC’s mean logdifffalls by 55% over the logged window and crosses below the same-window
Naive-Comptrace,whichshowsnotrend. Thatcomparisonisononeeditoratonescaleandweattachnoconfidence
interval to it; whether these conditioning-side movements translate into learning is decided by EM, not logdiff.
Naive-Compressed is the λ= 0limit of SDCC’s objective: setting λ= 0reduces the loss to the standard GRPO
objective on the compressed walk Hcomp(same underlying policy-loss implementation), so the SDCC vs. Naive-
Comp comparison is the natural matched-baseline ablation of the junction KL. Table 9and Figure 8close the
argument; training dynamics appear in § F.6.
6.5 Q3: Cross-harness (white-box and black-box)
White-box ordering is editor-dependent. The gap between the exact walks and Naive-Comp tracks how much
the editor rewrites. It is widest on length-triggered AgentFold, where LogitTree and 4D fall far below Naive-Comp
(0.012/0.022vs.0.366); it narrows on TC-RAG and closes on the low-eviction MemexRL cell, where all five
methods lie within 0.011–0.021and no method is separated from the floor. Naive-Full is elevated on the two
eviction-heavy harnesses, opposite to Pitfall B. Per-cell values are in Table 1.
Black-box transfer follows the same task-level ordering. On Claude Code, SDCC reaches 37.5average EM,
above LogitTree’s 35.9; on OpenCode, the corresponding values are 36.9and35.0. The observed logit-drift values
remain small in both harnesses (Claude Code: 0.015for SDCC versus 0.016for LogitTree; OpenCode: 0.013
versus 0.012). Because the two harnesses do not expose their internal evictions, we compare them by downstream
EM and observed maximum context length rather than compression depth; the latter is therefore reported as N/A in
Table1.
6.6 Large-scale evaluation on WideSearch
Weadditionallyevaluateon the WIDESEARCHbenchmark( Wonget al. ,2025)with the Claude Code harness. This
setting is distinct from the Qwen3-4B matrix: it uses Qwen3.7-Air , is trained on 256 NVIDIA H100 GPUs , and
evaluates 200questions with 4independent trials per question ( 800executions per model). We compare an SFT
checkpoint ( Base) against LogitTree after 15training steps. Pass@1 is the success rate of the first trial (“ _1”),
while Pass@4 counts a question as successful if any of its four trials succeeds.
Table2: WideSearch results with Claude Code (200 questions, four trials each). Pass@1usestrial _1; Pass@4
credits any successful trial. “Row” evaluates matching output rows and “Item” evaluates individual answer items.
P/R denote precision/recall; all Row/Item metrics are means over 800trial-level verifier outputs.
Method Pass@1 Pass@4 Row Precision Row Recall Row F1 Item Precision Item Recall Item F1
Base (SFT) 4.5 7.0 42.50 37.91 38.99 71.53 63.57 65.41
LogitTree (15 steps) 5.0 10.0 49.88 43.64 45.27 77.39 67.48 69.99
Relative improvement +11.1% +42.9% +17.4% +15.1% +16.1% +8.2% +6.2% +7.0%
LogitTreeimprovesPass@1from 9/200(4.5%)to 10/200(5.0%), orarelative 11.1%improvement; Pass@4rises
from 14/200(7.0%) to 20/200(10.0%), a relative 42.9% improvement. Rowprecision is the fraction of predicted
output rows that match a reference row, whereas row recall is the fraction of reference rows recovered; they rise
by17.4% and 15.1%, respectively. Item precision anditem recall apply the same definitions to individual answer
items, rising by 8.2% and 6.2%. Thus the row- and item-level F1 scores improve by 16.1% and 7.0%, respectively.
These verifier metrics complement binary success by giving partial credit for structured multi-item answers.
7 Conclusion and Future Work
Viewing a compressed rollout as a conditioning tree reveals a distinct train–inference mismatch in editor-based
agent RL. The two default replay schemes fail in opposite directions: Naive-Compressed conditions tokens on pre-
fixes that are too short, while Naive-Full conditions them on prefixes that are too long. Across different models,
white-boxandblack-boxharnesses, andmultiplelogitrecomputationschemes,weshowthatthismismatchappears
as measurable logit drift and degraded rollout reward. To address it, we present two exact conditioning-consistent
solutions, LogitTree and the 4D attention mask , together with a training-efficient approximation, SDCC. Exact
replay restores the no-compression drift floor, while SDCC substantially narrows the gap with a single-backward
10

objective. Future work includes extending the framework to broader compression policies (e.g., dropping tool ob-
servationsorusinglatentsummaries),testingacrossabroaderrangeofharnesses,andadaptingexacttree-consistent
replay to sparse and linear-attention architectures.
Provenance and acknowledgment. Ourwhite-boximplementationbuildsontheSLIMEexamplestack( Zhuetal. ,
2025),whichalreadyprovidedaTITO-style( Gallouédec&Rasul ,2026)losshooksketchingtheper-turnsegmented
objective—one forward per response segment, scored against a reconstructed rollout-time prefix. We reuse that
scaffold and its interfaces, and thank its authors. What this paper adds is the conditioning-tree formulation, the
equivalencebetweenexplicitbranchmaterializationandthepacked4Dmask,theSDCCrelaxation,therollout-side
splitting that makes exact replay executable, and the measurements of Table 1—not the idea of replaying tokens
under the prefix that produced them.
References
AlexanderA.Alemi, BenPoole, IanFischer, JoshuaV.Dillon, RifA.Saurous, andKevinMurphy. Fixingabroken
ELBO. In International Conference on MachineLearning (ICML) , 2018.
Anthropic. Claude code. https://docs.anthropic.com/en/docs/claude-code , 2024.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, et al. Improving language models by retrieving from
trillions of tokens. In International Conference on MachineLearning (ICML) , 2022.
Vivek S. Borkar. Stochastic approximation with two time scales. Systems&ControlLetters , 29(5):291–294, 1997.
Jörg Bornschein and Yoshua Bengio. Reweighted wake-sleep. In InternationalConferenceonLearningRepresen-
tations(ICLR) , 2015. Oral presentation; arXiv:1406.2751.
Ming-Bin Chen, Jey Han Lau, and Lea Frermann. Cig: Measuring conversational information gain in deliberative
dialogues with semantic memory dynamics, 2026. URL https://arxiv.org/abs/2604.15647 .
Mingyang Chen, Linzhuang Sun, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z. Pan,
WenZhang,HuajunChen,FanYang,ZenanZhou,andWeipengChen. ReSearch: Learningtoreasonwithsearch
for LLMs via reinforcement learning. In Advancesin NeuralInformation ProcessingSystems (NeurIPS) , 2025.
Zheng Chu, Xiao Wang, Jack Hong, Huiming Fan, Yuqi Huang, Yue Yang, Guohai Xu, Chenxiao Zhao, Cheng
Xiang, Shengchao Hu, Dongdong Kuang, Ming Liu, Bing Qin, and Xing Yu. Redsearcher: A scalable and
cost-efficient framework for long-horizon search agents, 2026. URL https://arxiv.org/abs/2602.14234 .
Hantian Ding, Zijian Wang, Giovanni Paolini, Varun Kumar, Anoop Deoras, Dan Roth, and Stefano Soatto. Fewer
truncations improve language modeling. In International Conferenceon MachineLearning (ICML) , 2024.
Quentin Gallouédec and Kashif Rasul. Agentic RL: Token-in, token-out done right. Hugging Face blog, https:
//huggingface.co/blog/huggingface/tito , 2026. Token-level loss-mask baseline; defers on history rewriting.
Saeed Ghadimi and Guanghui Lan. Stochastic first- and zeroth-order methods for nonconvex stochastic program-
ming.SIAM Journal on Optimization , 23(4):2341–2368, 2013.
Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed,
and Alexander Lerchner. β-VAE: Learning basic visual concepts with a constrained variational framework. In
International Conferenceon Learning Representations(ICLR) , 2017.
Geoffrey E Hinton, Peter Dayan, Brendan J Frey, and Radford M Neal. The wake-sleep algorithm for unsupervised
neural networks. Science, 268(5214):1158–1161, 1995.
XanhHo, Anh-KhoaDuongNguyen, SakuSugawara, andAkikoAizawa. Constructingamulti-hopQAdatasetfor
comprehensive evaluation of reasoning steps. In Proceedingsofthe28thInternationalConferenceonComputa-
tionalLinguistics(COLING) , pp. 6609–6625, 2020.
Xu Huang, Weiwen Liu, Xiaolong Chen, Xingmei Wang, Hao Wang, Defu Lian, Yasheng Wang, Ruiming Tang,
and Enhong Chen. Understanding the planning of llm agents: A survey, 2024. URL https://arxiv.org/abs/
2402.02716 .
Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. Llmlingua: Compressing prompts for
accelerated inference of large language models. In Conference on Empirical Methods in Natural Language
Processing(EMNLP) , 2023.
11

XinkeJiang,YueFang,RihongQiu,HaoyuZhang,YongxinXu,HaoChen,WentaoZhang,RuizheZhang,Yuchen
Fang, Xinyu Ma, Xu Chu, Junfeng Zhao, and Yasha Wang. TC-RAG: Turing–complete RAG’s case study on
medical LLM systems. In Proceedings of the 63rd Annual Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers) , pp. 11400–11426, Vienna, Austria, 2025a. Association for Computational
Linguistics.
Xinke Jiang, Jiaran Gao, Rihong Qiu, Zhixin Zhang, Wentao Zhang, Yue Fang, and Hongxin Ding. Agentic
rag-r1: Enhance agentic rag reasoning capacity via reinforcement learning. https://github.com/jiangxinke/
Agentic-RAG-R1 , 2025b. GitHub repository.
Xinke Jiang, Yue Fang, Zhibang Yang, Jiaran Gao, Zhixin Zhang, Tao Feng, Rihong Qiu, Wentao Zhang,
Hongxin Ding, Ruizhe Zhang, et al. Agenticrag-r1: Agentic reinforcement learning with stack memory for
multi-step reasoning, retrieval and memorizing. In EMNLP, 2026. URL https://github.com/jiangxinke/
Agentic-RAG-R1 .
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han.
Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint
arXiv:2503.09516 , 2025.
Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised
challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for
ComputationalLinguistics , 2017.
SatyapriyaKrishna,KalpeshKrishna,AnhadMohananey,StevenSchwarcz,AdamStambler,ShyamUpadhyay,and
ManaalFaruqui. Fact,fetch,andreason: Aunifiedevaluationofretrieval-augmentedgeneration. In Proceedings
ofthe2025ConferenceoftheNationsoftheAmericasChapteroftheAssociationforComputationalLinguistics:
HumanLanguageTechnologies(NAACL)—Volume1: LongPapers ,pp.4745–4759,Albuquerque,NewMexico,
2025. arXiv preprint arXiv:2409.12941.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle
Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: A benchmark for question answer-
ing research. Transactionsof theAssociation forComputational Linguistics(TACL) , 7:453–466, 2019.
Sergey Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review. arXiv preprint
arXiv:1805.00909 , 2018.
Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick
Lewis, and Deming Chen. SnapKV: LLM knows what you are looking for before generation. In Advances in
NeuralInformation ProcessingSystems , 2024.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang.
Lost in the middle: How language models use long contexts, 2023. URL https://arxiv.org/abs/2307.03172 .
Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez.
Memgpt: Towards llms as operating systems. arXiv preprintarXiv:2310.08560 , 2023.
OfirPress,MuruZhang,SewonMin,LudwigSchmidt,NoahA.Smith,andMikeLewis. Measuringandnarrowing
the compositionality gap in language models. In Findings of the Association for Computational Linguistics:
EMNLP2023 , 2023.
Ali Razavi, Aaron van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with VQ-VAE-2. In
Advances in NeuralInformation ProcessingSystems(NeurIPS) , 2019.
Machel Reid, Nikolay Savinov, Denis Teplyashin, et al. Gemini 1.5: Unlocking multimodal understanding across
millions of tokens of context. arXiv preprintarXiv:2403.05530 , 2024.
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization
algorithms. arXiv preprintarXiv:1707.06347 , 2017.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang,
Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language
models.arXiv preprintarXiv:2402.03300 , 2024.
Zhengliang Shi, Yiqun Chen, Haitao Li, Weiwei Sun, Shiyu Ni, Yougang Lyu, Run-Ze Fan, Bowen Jin, Yixuan
Weng,MinjunZhu,QiujieXie,XinyuGuo,QuYang,JiayiWu,JujiaZhao,XiaqiangTang,XinbeiMa,Cunxiang
Wang, Jiaxin Mao, Qingyao Ai, Jen-Tse Huang, Wenxuan Wang, Yue Zhang, Yiming Yang, Zhaopeng Tu, and
Zhaochun Ren. Deep research: A systematic survey, 2025. URL https://arxiv.org/abs/2512.02038 .
Alibaba Cloud Model Team. Qwen-agent: An agent framework for qwen large language models, 2024.
12

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via
single-hop question composition. Transactions of the Association for Computational Linguistics , 10:539–554,
2022.
Alexandre B. Tsybakov. IntroductiontoNonparametricEstimation . Springer, 2009.
Huanting Wang, Jingzhi Gong, Huawei Zhang, Jie Xu, and Zheng Wang. Ai agentic programming: A survey of
techniques, challenges, and opportunities, 2025. URL https://arxiv.org/abs/2508.11126 .
Zhenting Wang, Huancheng Chen, Jiayun Wang, and Wei Wei. Memex(rl): Scaling long-horizon llm agents via
indexed experience memory, 2026. URL https://arxiv.org/abs/2603.04257 .
Ryan Wong, Jiawei Wang, Junjie Zhao, Li Chen, Yan Gao, Long Zhang, Xuan Zhou, Zuo Wang, Kai Xiang,
Ge Zhang, Wenhao Huang, Yang Wang, and Ke Wang. WideSearch: Benchmarking agentic broad info-seeking.
arXivpreprintarXiv:2508.07999 , 2025.
YunjiaXi,JianghaoLin,YongzhaoXiao,ZheliZhou,RongShan,TeGao,JiachenZhu,WeiwenLiu,YongYu,and
Weinan Zhang. A survey of llm-based deep search agents: Paradigm, optimization, evaluation, and challenges,
2025. URL https://arxiv.org/abs/2508.05668 .
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models
with attention sinks. In International Conferenceon Learning Representations , 2024.
FangyuanXu,WeijiaShi, andEunsolChoi. RECOMP:Improvingretrieval-augmentedLMswithcontextcompres-
sion and selective augmentation. In International Conferenceon Learning Representations(ICLR) , 2024.
Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Jinhe Bi, Kristian
Kersting, Jeff Z. Pan, Hinrich Schütze, Volker Tresp, and Yunpu Ma. Memory-r1: Enhancing large language
model agents to manage and utilize memories via reinforcement learning, 2026. URL https://arxiv.org/abs/
2508.19828 .
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Conferenceon
EmpiricalMethodsin NaturalLanguageProcessing(EMNLP) , 2018.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergiz-
ing reasoning and acting in language models. In InternationalConferenceonLearningRepresentations(ICLR) ,
2023.
RuiYe,ZhongwangZhang,KuanLi,HuifengYin,ZhengweiTao,YidaZhao,LiangcaiSu,LiwenZhang,ZileQiao,
Xinyu Wang, Pengjun Xie, Fei Huang, Siheng Chen, Jingren Zhou, and Yong Jiang. AgentFold: Long-horizon
web agents with proactive context management. arXiv preprintarXiv:2510.24699 , 2025.
Ruizhe Zhang, Xinke Jiang, Zhibang Yang, Zhixin Zhang, Jiaran Gao, Yuzhen Xiao, Tao Feng, Yue Fang, Yuxuan
Liu, Ruiqing Li, Hongbin Lai, Huheng Huang, Xu Chu, Junfeng Zhao, and Yasha Wang. Stackplanner: A
centralized hierarchical multi-agent system with task-experience memory management, 2026a. URL https://
arxiv.org/abs/2601.05890 .
YaxiangZhang, Yingru Li, Jiacai Liu, JiaweiXu, Ziniu Li, Qian Liu, and HaoyuanLi. Beyond precision: Training-
inference mismatch is an optimization problem and simple lr scheduling fixes it, 2026b. URL https://arxiv.
org/abs/2602.01826 .
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian,
Christopher Ré, Clark Barrett, Zhangyang Wang, and Beidi Chen. H 2o: Heavy-hitter oracle for efficient genera-
tive inference of large language models. In Advancesin NeuralInformation ProcessingSystems , 2023.
ChujieZheng,ShixuanLiu,MingzeLi,Xiong-HuiChen,BowenYu,ChangGao,KaiDang,YuqiongLiu,RuiMen,
AnYang,JingrenZhou,andJunyangLin. Groupsequencepolicyoptimization. arXivpreprintarXiv:2507.18071 ,
2025.
DongzhuoranZhou,YuqichengZhu,YuleLiu,ZhenYang,RuiLu,YuxiaoDong,JieTang,andEvgenyKharlamov.
Awm: Answerable working memory for long-document vqa agents, 2026. URL https://arxiv.org/abs/2608.
25618.
Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang
Low, and Paul Pu Liang. Mem1: Learning to synergize memory and reasoning for efficient long-horizon agents,
2025. URL https://arxiv.org/abs/2506.15841 .
ZilinZhu,ChengxingXie,XinLv,andslimeContributors. slime: AnLLMpost-trainingframeworkforRLscaling.
https://github.com/THUDM/slime , 2025. GitHub repository. Corresponding author: Xin Lv.
13

Part I Position in the agentic-RL literature
At the time of writing, published baselines for harness-RL training on QA-style benchmarks (Search-R1 ( Jin et al. ,
2025), ReSearch ( Chen et al. ,2025)) all train on the physical trajectory Ht. AgenticRag-R1 ( Jiang et al. ,2026)
and MEM1 ( Zhou et al. ,2025) have begun exploring RL with stack-based or learned memory mechanisms for
long-horizon agents. Recent work in the long-horizon coding-agent literature (e.g., agentic SWE-bench solvers)
has also informally reported that training on edited transcripts is harmful. To our knowledge, our work is the first
to formalize the conditioning invariant, characterize the failure modes that follow from violating it, and propose a
soft remedy with a provable bias bound.
Part II Method supplements: pitfalls, structure, proofs, variational derivation,
convergence
This part supplies the technical scaffolding behind the development of § 3–§5: a worked tool-using rollout in which
a single eviction places two target spans on opposite sides of it, so that both pitfalls and the mirror symmetry
between them can be read off one concrete trajectory (§ A); the trajectory-tree object itself (§ B); every proof stated
in §4–§5, restated so the appendix reads self-contained (§ C); and the end-to-end β-VAE derivation of SDCC with
its convergence analysis (§ D, §D.6).
A A worked example: one weather lookup, two pitfalls
§3statesbothpitfallsoverabstracttokens yt; herebothsitononerolloutwithoneeviction, sothemirrorsymmetry
of Table 5is read off rather than derived. The rollout is constructed for exposition: it is not a logged trajectory,
and no quantity in it is a measurement.
Setup. TC-RAG’s learnable pop(§E.2) deletes an envelope outright, leaving no summary behind to carry the
forecast forward. Asked about an umbrella for tomorrow’s meeting in Shanghai, the policy calls get_weather ,
receives envelope o1, and writes span S— “rain is likely tomorrow afternoon — bring an umbrella ” — while o1
is in view . The budget is then exceeded, popevicts the largest envelope o1at junction J1(E1=o1), and a later
get_transit turnproducesspan Tafter the eviction . Below,rolloutistheliveviewatargetwasdecodedunderand
training the constructed prefix the objective scores it under; the one row where they disagree is the whole failure.
A.1¶Pitfall A on S: the model is taught that it knows the weather
Naive-Compressed, time-travel leakage: “I alreadyknowtheweather.”
Naive-Compressed trains on the final compressed walk Hcomp, soSis scored under ps(S):J1firesafter Sis
decoded, yet its effect is already applied (Eq. 4,ps(S)( pt(S)).
rollout training
pt(S) ps(S)
[user] umbrella for tomorrow’s Shanghai meeting? ✓ ✓
[call] get_weather(”Shanghai”, +1d) ✓ ✓
[obs ] weather envelope (afternoon rain) ✓ ×
[tgt ] S: ” ...rain tomorrow afternoon... ” scoredhere
The toolcallsurvives, only its resultis gone, so −logPθ(S|ps(S))demands the forecast from a prefix that
no longer holds it; the only way to cut the term is to move mass onto “rain tomorrow afternoon” as a priorover
Shanghaiweather . A tool-grounded assertion becomes a from-memory one, and the deployed model states the
forecast before reading the tool. T oo little context.
A.2·Pitfall B on T: the model is taught to read a discarded context
Naive-Full, stale-context leakage: “Icompressedthat— why is it stillin my context?”
Naive-Full trains on the depth-first tape HT, which deletes nothing, so Tis scored under the physical prefix
H<t, which still holds the o1its live view had lost (Eq. 5,H<t) pt(T)).
14

rollout training
pt(T) H<t
[user] umbrella for tomorrow’s Shanghai meeting? ✓ ✓
[call] get_weather(”Shanghai”, +1d) ✓ ✓
[obs ] weather envelope (afternoon rain) × ✓
[span] S: ” ...rain tomorrow afternoon... ” ✓ ✓
[pop ] J1:E1= weather envelope ✓ ✓
[turn] get_transit(...) → transit envelope ✓ ✓
[tgt ] T: ”Line 2 runs every four minutes... ” scoredhere
The gradient therefore rewards reading the forecast back afterthe pop— a move the deployed agent cannot
make, because there the deletion is real. T oo much context.
The mirror . On row [obs] weather envelope , Pitfall A reads (✓,×)and Pitfall B reads (×,✓): A drops the
forecastfromatargetthatusedit,Bkeepsitforonethathadlostit,sonoreweightingofasingleserializationrepairs
both. §3.3measures this — ∆comp =−22.8against ∆full= +18 .5nats on an untrained Qwen3-4B, opposite in
sign and comparable in magnitude. Those two numbers are measurements; the weather rollout is not. Both defects
vanish under one repair: score every target under the live view it was decoded from (Definition 1).
B The trajectory tree: a unified structural analysis
This appendix collects, in one place, the tree-theoretic view underlying every construction in the main text. § 3
introduces the running-example tree T; §3.3states the conditioning invariant on its leaves; § 4and §5materialize
that invariant with a 4D mask, with LogitTree, and with SDCC. The purpose here is to make the object Texplicit
as a graph, count its components (junctions, leaves, branches), identify where each method places gradient, and
read off the compute budget from the counts.
B.1 Definition of the trajectory tree
Definition 2 (Trajectory tree) .Fix a rollout with physical trajectory HT= (x1, . . . , x T)and eviction records
{(Jk, Ek)}K
k=1,wherethe k-thcompressionfiresatphysicalposition Jkandremovesthepositions Ek⊂ {1, . . . , J k}.
The trajectory tree T= (V, E, ϕ )has aroot(the initial user prompt), a junction Jkfor each k, and aleaffor each
generatedtoken yt; edgesarelabeledbyphysical-tokenspans,sofollowingedgesfromtheroottoanode vrecovers
the physical prefix that produced v. Two colorings ϕspine, ϕleg:V→2{1,...,T}record the two conditioning sets of
interest: ϕspine(v)is the prefix visible on the final compressed walk, with everyeviction applied, and ϕleg(v)is the
live view of equation 2— the prefix visible when vwas decoded, carrying exactly the evictions that had fired by
that step and no others. A node decoded at step tisdiverging iff the two disagree,
ϕleg(v)\ϕspine(v) =[
j:Jj>tEj∩ {1, . . . , t −1} ̸=∅,
i.e. iff some eviction that fires aftertreaches back into v’s prefix.
Two regions of Tare non-diverging by construction: nodes whose prefixes precede every evicted span, and nodes
decodedatorafterthefinaljunction JK,forwhichallevictionshavealreadybeenappliedandthetree re-converges
(cf. leaf yansin Table 4). Nodes decoded within [Jk−1, Jk)form the k-thbranchlayer , with J0= 0.
Counts. Write Rforthenumberofresponsetokensand Dforthenumberofdivergingleaves. Then Tcarriesone
root,Kjunctions, Rleaves, and K+1root-to-leafbranches, andeachleafhasanevictionmass |ϕleg\ϕspine|—the
tokens it conditioned on at generation that its final-walk prefix no longer contains. Only two of these numbers are
read off downstream: the junction depth K, which upper-bounds the number of independent forwards a LogitTree-
style method needs, and the diverging-leaf count D, which is exactly the set of positions where SDCC places KL
gradients. Table 3reports both for the three white-box harnesses at Tmax= 5, so tree size can be read directly off
theharnessknob. Tworegularitiesmatterlater: theper-evictionmass |Ek|ranksinverselywith K(TC-RAGevicts
large spans rarely, MemexRL small slices often), and these are structural capacities under a forced schedule, not
live firing rates — MemexRL has the deepest tree here but the lowestmodel-triggered eviction density in Table 10.
B.2 Where each method places gradient on T
•Naive-Full runs one forward on the union HTand takes gradients at every response token position. On Tthis
collapsesthetreeintoitsdepth-firsttraversal: everyleafisconditionedonthe entirephysicalprefix H<t⊇ϕleg(v),
which contains all evicted spans — including spans that had already left the live view when vwas decoded. The
training conditioning strictly exceeds the live view, so Naive-Full trains on information the policy never had at
any point in the rollout (Pitfall B, § 3.3).
15

Table 3: Tree sizes induced by the three white-box harnesses on the running example (probe of Table 7);K=
junctions per rollout, D= diverging leaves per rollout, |Ek|= mean evicted-span length per junction.
Harness mean Kmean diverging leaves Dmean|Ek|
TC-RAG 0.87 29 156
AgentFold 1.94 58 84
MemexRL 2.00 125 41
•Naive-Compressed runsoneforwardonthecompressedwalk Hcompandtakesgradientsateveryresponsetoken.
OnTthis collapses all branches into the final walk: the training forward sees ϕspine(v), which at the Ddiverging
leavesis a strictsubset of the live view ϕleg(v)that generated the token — and, worse, contains summary content
created after t. The loss on those leaves is therefore computed under conditioning the model never decoded from
(Pitfall A, § 3.3).
•LogitTree (ours, K-forward) runs one forward per branch layer, so K+1forwards per rollout. Each forward
covers a root-to-leaf path with ϕspine=ϕlegalong the entire branch (because the branch’s forward uses the live
viewas it existed during that layer). Gradients are placed on the leaves of that branch only, so the union of
LogitTree’s gradient-carrying positions equals the leaf set of Twithout double-counting.
•4D mask (ours, packed) realizes the same partition as LogitTree but in a singleforward: one structured mask
admits, in each query row, exactly the keys that were live when that token was decoded, so ϕspine=ϕlegholds
row by row and the branches share the tree trunk instead of being replayed. Gradient placement is identical to
LogitTree’s; the difference is the pass count: one forward instead of K+1.
•SDCC (ours) runsonestudentforwardon Hcomp(withgradient)plusonegradient-freeteacherforwardperbranch
layer that holds a diverging leaf — at most K, since the layer after JKre-converges. The policy-gradient loss
remains on the student’s R-length response tokens. The consistency KL DKL(πθ,teach(· | H′
t)∥πθ,stud(· | Hcomp[:
t]))issummedonlyoverthe Ddivergingleaves,whosepositionsaregivenbythediverging-leafmaskconstructed
fromtheharness’s {Jk, Ek}records. On T,SDCCpullsthestudent’sper-leafdistributiontowardtheteacher’sat
exactlythedivergingpositions, withouttouchingthetrunk. Thisiswherethe O(√εKL)boundof§ 5.2applies: as
SDCC’s KL contracts, the student’s leaf distributions approach the teacher’s, and the diverging-leaf gap closes.
B.3 Compute budget as tree traversal cost
Three lengths govern the budget: the union length L=|HT|; the compressed walk C=|Hcomp|=L− |S
kEk|,
shorter than Lby the token massevicted and not by the eviction count; and ℓi, the live view in force during branch
layer i. Consecutive branches share the trunk, so C≤ℓi≤L— with ℓK+1=C, every eviction having fired by
the last layer — andPK+1
i=1ℓi≥Lwith equality only at K= 0: LogitTree re-reads the shared prefix once per
layer rather than splitting LintoK+1disjoint pieces. Table 6tabulates the resulting per-rollout pass and token
budgets next to the infrastructure each method demands.
Tokens are a proxy, not a cost: a forward over ntokens costs F(n) = Θ( nd2+n2d), and for Qwen3-4B the
quadratic term overtakes the linear one at n≈12k — inside the range of contexts we observe (Table 1). A dense
kernel scores the packed sequence in full and discards the masked entries, paying Θ(L2d), while K+1segmented
forwardspay Θ(P
iℓ2
id),soneitherdominates—itturnsontheevictedmass. The4Dmask’sdependableadvantage
over LogitTree is one backward instead of K+1and one well-filled kernel launch instead of K+1short ones, not
fewer FLOPs; a genuine FLOP saving needs a block-sparse kernel. SDCC saves nothing on the forward: because
ℓK+1=C, the student pass isthe last branch layer’s, and SDCC’s token budget C+PK
i=1ℓiequals LogitTree’sPK+1
i=1ℓiterm for term. The entire saving is the backward, 1instead of K+1; and unlike either exact walk it gives
up exactness, down to O(√εKL).
B.4 Per-leaf divergence and the two pitfalls
Recallthattheeditorfiresrepeatedly insidetherolloutloop: atstep tthepolicyseestheliveview H′
t=VIEW t(H≤t;E≤t)
ofequation 2—thephysicalhistorywithexactlytheeditsthathavefiredbystep tapplied,andnoothers—whereas
thelength- tprefix Hcomp[:t]ofthefinalcompressedwalkretroactivelyapplies futureeditsE>ttoatokengenerated
before those edits existed. Table 4instantiates the divergence for each leaf of the running example.
Pitfall A, SFT view (time-travel leakage).
LA
SFT=−X
tlogPθ(yt|ps(yt))̸=−X
tlogPθ(yt|pt(yt)) =L⋆
SFT. (11)
The model is fitted to a shorter-prefix distribution than the one that generated the target token, while at inference
the harness still delivers the live view H′
t= pt(yt)— a prefix the model was never trained to match.
16

Table 4: Per-leaf live-view prefix vs. final-walk prefix for the running example ( E1={y5}atJ1= 7;E2={y8}
atJ2= 10;E3={y11}atJ3= 12). The leaf whose two prefixes coincide ( yans) does not need any correction.
Evicting one token per junction keeps each row on a single line; widening a span enlarges Ekand each leaf’s
eviction mass, but changes neither which leaves diverge nor where the tree re-converges.
LeafytLive-view prefix pt(yt) Final-walk prefix ps(yt) Diverges by
y6 {y1, y2, y3, y4, y5} {y1, y2, y3, y4} +y5
y7 {y1, y2, y3, y4, y5, y6} {y1, y2, y3, y4, y6} +y5
y9 {y1, y2, y3, y4, y6, y7, y8} {y1, y2, y3, y4, y6, y7} +y8
y10 {y1, y2, y3, y4, y6, y7, y8, y9} {y1, y2, y3, y4, y6, y7, y9} +y8
y12 {y1, y2, y3, y4, y6, y7, y9, y10, y11}{y1, y2, y3, y4, y6, y7, y9, y10}+y11
yans {y1, y2, y3, y4, y6, y7, y9, y10, y12}same —
Pitfall A, RL view (broken importance ratio). The rollout policy sampled at∼πθold(· | pt(yt)), so the training
ratio
ρA
t=πθ(at|ps(yt))
πθold(at|pt(yt))̸=πθ(at|pt(yt))
πθold(at|pt(yt))=ρt (12)
invalidates PPO’s clip guarantee and mis-weights GRPO’s group-relative advantage; empirically, the clip-rate map
concentrates immediately after each junction, precisely where the two prefixes diverge.
Pitfall B, SFT view (train–inference mismatch).
LB
SFT=−X
tlogPθ(yt|H<t). (13)
Cross-entropyiscomputedusingaprefixthatcarriesstrictlymoreinformation—includingalready-evictedcontent
— than the deployed model will ever see.
Pitfall B, RL view (advantage on the wrong observation). Eventually-evicted tokens remain present during
training, so they still absorb a share of the outcome reward rout/T, and the advantage baseline is fitted to V(H<t)
instead of V(H′
t); the step-level credit-assignment error accumulates linearly in K(Jiang et al. ,2025b):
E
∥B(K)
e∥2
≃εe·K·E
∥∇θlogπθ(ae|se)∥2
, (14)
where B(K)
eis the cumulative advantage-baseline bias at an evicted-token position eafterKeviction events occur
along the trajectory, and εeis the per-event bias scale (the misfit between V(Ht)andV(H′
t)attributable to one
evicted span).
Table5summarizes the mirror symmetry of the twopitfalls; concrete per-leaf examples(a ∆comp=−22.8-nat leaf
for Pitfall A and a ∆full= +18 .5-nat leaf for Pitfall B, both on an untrained Qwen3-4B) are provided in the main
text (§3.3).
Table 5: Two symmetric failure modes of the same conditioning invariant.
Naive-Compressed (A) Naive-Full (B)
Training input Hcomp(rightmost path) HT(full DFS)
Prefix relation ctrain(cinferctrain)cinfer
Failure mode time-travel leakage train–inference mismatch
Directional signature ctoo short ctoo long
Intuition “train saw less than infer” “train saw future info”
Fixable by any tree-consistent method any tree-consistent method
ForQwen3-4Bunderlivemodel-triggeredcompression(§ 6, Table1), themismatchisdirectlyvisibleinthe logdiff
statistic: the No-compression (Search-R1) control remains at 0.014(numerical noise — with nothing compressed,
the live view equals the physical history), whereas Naive-Compressed rises to 0.024–0.059on average across the
three editors and peaks at 0.23–0.29on eviction-heavy batches ( ∼20×the baseline), scaling monotonically with
eviction density: the direct empirical signature of ctrain̸=cinfer, present from the very first steps of training.
Correct gradients. Under the invariant, both the supervised loss and the policy gradient condition uniformly on
H′
t—theobservation oftheeditor-inducedPOMDP(§ 3.3): therolloutandtrainingpoliciesintheimportanceratio
must share this observation, and the advantage baseline must approximate V(H′
t), notV(Ht). A fix must therefore
restore ∇θdLtask(θ) =∇θL⋆
task(θ)exactly (§ 4) or up to a quantifiable O(√εKL)bias at the loss level (§ 5).
17

C Proofs
We proceed bottom-up: the per-branch tree decomposition (§ C.1) is the primitive, the packed 4D construction
(§C.2) reproduces it in one masked forward, and their equivalence (§ C.3) follows. Every proof opens with a one-
line sketch. Throughout, pt(yt) = H′
tis the live-view root-to-leaf prefix of leaf ytin the conditioning tree Tof
definition 2, and ps(yt) = Hcomp[:t]its prefix on the final compressed walk.
C.1 Per-branch equivalence on the tree (proposition 3)
Proposition 3 (SFT and RL equivalence on the tree) .LetLtree
SFTbe the summed cross-entropy over the K+1per-
branchforwards,eachloss-maskedtoitsownbranch. Then ∇θLtree
SFT=P
π∇θL⋆
SFT(H′π). ForRL,theimportance
ratio,advantagebaseline, and KL regularizerarecomputedon thesame H′πduringrolloutand update.
Proof.Sketch: the K+1branches partition the loss-carrying leaves, so regrouping the per-leaf loss by branch is
an identity,notan approximation.
Each ytlies on exactly one branch π(yt)— the one live when it was produced — so H′π(yt)[:t] = pt(yt), and the
per-branch loss masks are disjoint and jointly exhaustive on the response. RegroupingP
t−logPθ(yt|pt(yt))by
branchreturns Ltree
SFTwithnoleafdouble-countedormissed,andthefinitesumcommuteswith ∇θ. ForProposition 3,
theimportanceratio,advantageweightandKLregularizerat ytareallfunctionsof Pθ(· | pt(yt)),whichthebranch-
πforward evaluates under exactly the rollout-time conditioning.
C.2 One packed forward reproduces the tree (proposition 4)
Proposition 4 (SFTandRLequivalenceforthe4Dmask) .Suppose(i) Mlogicalexactlyencodeslogicalvisibility,(ii)
position ids are reassigned for each query row, (iii) loss and attention masks are decoupled, (iv) no normalization
crossesmaskboundaries,and(v)rolloutandtrainingsharethesame Mlogical. Then,underdensesoftmaxattention,
∇θL4D
SFT=∇θL⋆
SFT,andtheimportanceratioandKLregularizerarecomputedunderidenticalconditioningduring
rolloutand update.
Proof.Sketch: a masked query row and its image in the compact sequence attend to the same keys at the same
relative offsets, hence produce the same logits; induction over layers and the shared loss mask lift this to gradient
equality.
Let Edit delete Et⊂ Htand let Mimplement Mlogical— the (T×T)mask of § 4.2admitting key jin row iiff
jwas live when yiwas decoded — so Mij=−∞forj∈Et,j < i, and causal otherwise. By (iii) it suffices
to prove Logitsi(Ht, M) = Logitsi(H′
t)fori/∈Et. Write i′, j′for the images of i, junder the deletion, so the
visible set Vi:={j < i :j/∈Et}maps onto the prefix of the compact sequence H′
t. Row imixes only over Vi
by construction, and by (ii) the reassigned position ids give (i, j)the same relative offset as (i′, j′); since RoPE
— or any relative-positional encoding — depends only on that offset, q⊤
ikj=q⊤
i′kj′, so softmax weights and
mixed values agree. Condition (iv) closes the only remaining channel of difference, per-token operations reading
across mask boundaries (a layer norm reduced over an unmasked axis, MoE routing pooling evicted positions),
so the residual stream at row iequals the compact-forward stream at i′. Induction over layers carries this to the
unembedding, and the shared loss mask turns logit equality into the gradient identity.
C.3 Materialization equivalence (theorem 1)
(Recalled.) Undertheassumptionsofproposition 4,the4D-maskedsingleforwardonthephysicalunion HT(§4.2)
and the K-segment logits-tree forward produce the same logits at every non-evicted position.
Proof.Sketch: both materializations are the same scalar function of θ, and identical functions have identical
gradients.
By proposition 4each construction computes Logitsi(H′
t)at every loss-carrying position under the identical loss
mask; they differ only in cost (union mask: one attention call with a (B,1, T, T )structured mask; segmented walk:
K+1separate forwards). Both losses are therefore the samescalarfunctionof θ— the leaf sum −P
tlogPθ(yt|
pt(yt))of proposition 3— so their gradients coincide up to floating-point association order.
D Full V ariational Derivation of the SDCC Forward-KL Self-Distillation Loss
This appendix derives the SDCC objective (§ 5, Eq. equation 10) as a forward-KL conditioning-consistency (self-
distillation) regularizer, using amortized variational inference as a scaffold. The correctness of SDCC does not rest
on the ELBO being tight: it rests on the zero set of the KL residual being exactly the conditioning invariant (§ D.3),
18

and on that residual controlling deployment behavior (§ D.4). §5.1gives the main-text algorithmic view; here we
state the assumptions and give the proofs.
Why the forward direction. The slack in the ELBO is a reverseKL, and descending it directly is the wake-phase
updateofamortizedinference( Hintonetal. ,1995;Bornschein&Bengio ,2015). Atthesequencelevelthatrequires
sampling whole continuations from the compressed context: the resulting score-function estimator is weighted by
log(qθ/pθ), so its variance is unbounded on the surplus side, while the deficit side — which is Pitfall A itself — is
almost never drawn from the compressed proposal and is therefore invisible to the update. SDCC instead descends
the forward-KL (sleep-phase) surrogate, which samples from the well-behaved teacher side, has a plain bounded-
variance cross-entropy gradient, and, as § D.3shows, has exactly the same zero set. The four subsections follow
that substitution: § D.1fixes the probabilistic model, § D.2derives the bound the KL comes from, § D.3shows the
forward direction loses nothing, and § D.4bounds what a nonzero residual costs at deployment.
D.1 Probabilistic model and the conditioning invariant
Fix a diverging leaf position p∈ C. All quantities below are tied to this particular p; sums over Care taken at the
end. Write A(p) ={k:ι(Ek)≤p≤e(Ek)}for the evictions active at p. The common case is |A(p)|= 1; when
several evictions are active the teacher context re-inserts every span in A(p)and nothing below changes.
Random variables.
•Clean context zp:= pt(yp): the pre-eviction prefix that was live at rollout time and that the harness will again
deliveratdeploymenttime. Itisan externallysupplied conditioningvariable,nevermarginalizedout: atinference
the harness passes it directly to the policy.
•Compressed context xp:= ps(yp)(zp: the post-eviction prefix produced by the memory editor, which has
removed every span in A(p).
•Action y∈ V: the next generated token.
•Outcome y⋆, with likelihood p(y⋆|y): an oracle target for y. We take the RL reading as primary, in which
logp(y⋆|y) =r(y)/τistheexponentiated-advantagelog-likelihoodofcontrol-as-inference( Levine,2018); the
SFT reading, log p(y⋆|y) =log 1[y=y⋆], is its τ→0limit.1
Distributions.
pθ(y|zp) :=Pθ 
· | pt(yp)
deployed / target policy (teacher forward), (15)
qθ(y|xp) :=Pθ 
· | ps(yp)
pamortized recognition policy (student forward), (16)
p(y⋆|y) outcome likelihood given action. (17)
The two policies share parameters θ; the amortization is parametric , not architectural. The conditioning invariant
of §3.3is the statement that this amortization is exact,
qθ(· |xp) = pθ(· |zp)∀p∈ C. (CI)
equation CIisnotamodelingassumption—itistheconditionwewantto achievethroughtraining, andeverything
that follows is an argument that the SDCC penalty achieves it.
D.2 The ELBO and where the forward KL comes from
This subsection does three things. It derives the variational bound whose slack is the conditioning gap and shows
that slack to be an exact KL divergence; it identifies the multiplier λof equation 10as the βof aβ-VAE free
energy, which fixes both its interpretation and its failure mode at large values; and it shows why that slack cannot
be descended in the direction the bound presents it in — the fact that forces the forward-KL substitution justified
in §D.3.
Proposition 5 (Single-step ELBO for the amortized policy) .Forevery θand every divergingleaf p∈ C,
logpθ(y⋆|zp)≥E
y∼qθ(·|xp)
logp(y⋆|y)
−DKL
qθ(· |xp)pθ(· |zp)
, (18)
withequalityiff qθ(· |xp) =pθ(· |y⋆, zp)almosteverywhere.
1The hard-indicator limit is degenerate as a variational target: whenever qθplaces mass off y⋆the bound equation 18reads
−∞ ≥ −∞ and is vacuously satisfied. Every statement below is made at finite τ, where the expressions remain finite.
19

Proof.Marginalizing the action gives the evidence pθ(y⋆|zp) =P
yp(y⋆|y)pθ(y|zp). Multiplying and
dividing by qθ(y|xp), rewriting the sum as a qθ-expectation and applying Jensen to the concave logarithm,
logpθ(y⋆|zp) =logE
y∼qθp(y⋆|y)pθ(y|zp)
qθ(y|xp)
≥E
y∼qθ
logp(y⋆|y)pθ(y|zp)
qθ(y|xp)
(19)
=E
y∼qθ
logp(y⋆|y)
−DKL 
qθ(· |xp)pθ(· |zp)
,
which is equation 18. The importance-weighting step needs the proposal qθ(· |xp)to dominate the integrand; this
is automatic here, since both distributions are softmax outputs of the same network and so have full support on V
(up to numerical underflow). The reverse domination — pθ(· |zp)dominating qθ(· |xp)— isnotneeded for the
bound, and is precisely what fails for the reverse-KL estimator below.
Fortheequalitycondition,observethattheJensengapisitselfaKLdivergence. Bayes’ruleatfixed zpreads pθ(y|
y⋆, zp) =p(y⋆|y)pθ(y|zp)/pθ(y⋆|zp), so log pθ(y⋆|zp) =logp(y⋆|y) +logpθ(y|zp)−logpθ(y|y⋆, zp)
for every yin the support. The left-hand side does not depend on y, so it is unchanged by taking the qθ-expectation
of the right-hand side; adding and subtracting Eqθ[logqθ(y|xp)]and regrouping the three resulting expectations
gives the exactdecomposition
logpθ(y⋆|zp) =E
qθ
logp(y⋆|y)
−DKL 
qθ(· |xp)∥pθ(· |zp)
| {z }
the bound equation 18
+ DKL 
qθ(· |xp)∥pθ(· |y⋆, zp)
| {z }
tightness gap, against the outcome-conditioned posterior.(20)
The residual term is non-negative by Gibbs’ inequality — which re-derives equation 18without invoking Jensen —
and vanishes iff qθ(· |xp) =pθ(· |y⋆, zp)a.e.
Reading the bound. The first term of equation 18is the (negated) task loss evaluated under the compressed
conditioningthattrainingactuallysees;thesecondisthe conditioninggap betweenthecompressedandpre-eviction
views, and it is the only term that references zp. Decomposition equation 20is worth isolating because it separates
two quantities that are easy to conflate in the main text: the conditioning gap, which SDCC exists to close, and
the tightness gap DKL(qθ(· |xp)∥pθ(· |y⋆, zp))against the outcome-conditioned posterior, which SDCC never
touches and does not need to. § D.3makes that separation precise.
The multiplier λis the βof aβ-V AE. Nothing in equation 18fixes the relative weight of its two terms. Treating
the conditioning gap as a constraint rather than a fixed-weight penalty — maximize the task evidence subject toP
p∈CDKL≤δ— and forming the Lagrangian returns equation 10exactly, with λthe multiplier and the constant
λδdiscarded. In β-VAE terms ( Higgins et al. ,2017), writing β:=λand using the task loss as the distortion,
Fβ(θ) =Ltask(θ;Hcomp)|{z }
distortion+βX
p∈CDKL 
psg(θ)(· |zp)qθ(· |xp)
| {z }
rate, (21)
withβ= 1the plain evidence bound and β= 0Naive-Compressed. The distortion–rate tension is genuine
here precisely because xp(zp: distortion is minimized by exploiting the compressed tokens, rate by making
the student’s output independent of them, and any task-relevant content of zp\xpmust be recovered through the
parameters rather than read off the input. Pushing βtoo far therefore reproduces the classical failure mode.
Proposition 6 (Collapse regime at large β).Letθcollbe a parameter at which the student output is functionally
independent of xpand equals the teacher, qθcoll(· |xp) =psg(θcoll)(· |zp)for all p∈ C. Then the rate term of
equation21vanishesat θcollwhileitsdistortiontermisgenericallypositive,andforevery β >Ltask(θcoll;Hcomp)/ε⋆
—where ε⋆>0lower-boundstherateatatask-optimal θ⋆—onehas Fβ(θcoll)<Fβ(θ⋆),sodescenton Fβfrom
an uninformed initialization is attractedtothecollapsed solution.
Proof.Theratevanishesat θcollbyconstruction. Thedistortionispositivebecauseastudentthatignores xpcannot
fit a task loss evaluated on Hcomp. Substituting both into equation 21givesFβ(θcoll) =Ltask(θcoll;Hcomp), while
Fβ(θ⋆)≥βε⋆; the stated threshold on βis exactly the crossing point. This is β-VAE posterior collapse ( Alemi
et al.,2018;Razavi et al. ,2019).
Prop.6is why λcarries a warm-up schedule and the target a stop-gradient rather than being a free hyperparameter:
KL annealing holds β= 0until task-loss SGD has moved off θcoll, and a frozen (or EMA) target keeps it from
tracking the student into collapse. Both are the standard β-VAE remedies, and both are analyzed in § D.6.
20

Why the slack cannot be descended in the direction it appears. The remaining step is the direction of the
KL. The obvious move is to descend the term of equation 18as written — the wake-phase update of amortized
inference ( Hinton et al. ,1995;Bornschein & Bengio ,2015). Because the gradient must then pass through the
sampling distribution, its estimator is a score function reweighted by a log-ratio.
Lemma 1 (Wake-phasegradient,andthevarianceofitsestimator) .LetLwake(θ) :=DKL 
qθ(· |xp)∥psg(θ)(· |zp)
withthetargetbranchfrozen,and write gθ(y) :=∇θlogqθ(y|xp)andρθ(y) :=qθ(y|xp)/psg(θ)(y|zp). Then
∇θLwake(θ) =E
y∼qθ
logρθ(y)gθ(y)
, (22)
and its N-sample Monte-Carloestimatorhas variance
Var\∇θLwake
=1
N
E
qθ
(logρθ)2∥gθ∥2
−E
qθ
logρθgθ2
. (23)
Proof. Lwake(θ) =P
yqθ(y|xp)logρθ(y),andonly qθcarriesagradient,so ∇θLwake=P
y∇θqθ(y)logρθ(y)+P
y∇θqθ(y). The log-derivative identity ∇θqθ=qθgθturns the first sum into equation 22, and the second is
∇θP
yqθ(y) =∇θ1 = 0. Equation equation 23is then the variance of a mean of Ni.i.d. copies of X:=
logρθ(y)gθ(y), namely (E∥X∥2− ∥EX∥2)/N.
Identity equation 23is exact — it uses no log ρ≈ρ−1small-ratio expansion — and it exposes two failure modes,
one on each side of the eviction:
•Surplus side: unbounded variance. A token y†the compressed context finds plausible ( qθ(y†|xp)≥q0>0)
but the pre-eviction context does not ( psg(θ)(y†|zp)→0) — a redundant re-query that looks reasonable only
once the retrieved span is gone — sends log ρθ(y†)→+∞, and the second moment in equation 23grows like
(logρθ(y†))2. No variance bound holds uniformly over Θ.
•Deficit side: sampling blindness. A token the pre-eviction context supports ( psg(θ)(y†|zp)≥p0>0) but
the compressed one has lost ( qθ(y†|xp)→0) is Pitfall A itself. Here the variance does notblow up, since
qlog2q→0, and that is exactly the problem: the summand qθ(y†)logρθ(y†)gθ(y†)→0, so the update draws
no signal from the one token whose conditioning it is meant to repair, and an N-sample estimator needs N=
Ω 
1/qθ(y†)
draws to observe it even once.
Both are properties of the sequence-level Monte-Carlo estimator rather than of the reverse KL as a function: a
per-token vocabulary sum evaluates Lwakeexactly and has neither pathology. What survives even in that exact
form is the reverse direction’s zero-forcing bias, which under-covers precisely the modes an eviction removes. Two
substitutionsthereforeseparateequation 18fromequation 10—theKListakenforwardratherthanreverse,andits
right argument is frozen by a stop-gradient — and § D.3justifies both, showing that the forward direction samples
from the well-behaved side, has a bounded-variance gradient, and retains the same zero set.
D.3 Why the forward KL suﬀices
Two things must hold before the ELBO’s reverse KL may be replaced by the forward KL that SDCC minimizes:
the reverse-KL slack must vanish exactly at the conditioning invariant, so that the ELBO points at the right target;
and the forward KL must have the same zero set, so that the substitution loses nothing.
The slack is the conditioning gap, and is weaker than tightness.
Proposition 7 (Slack-zero implication) .DKL 
qθ(· |xp)∥pθ(· |zp)
= 0if and only if qθ(· |xp) =pθ(· |zp)
a.e., i.e. if and only if the per-leaf conditioning invariant equation CIholds. This is strictly weaker than tightness
of equation 18, which by Prop. 5additionally requires the recognition policy to equal the outcome-conditioned
posterior pθ(· |y⋆, zp)ratherthantheoutcome-marginal pθ(· |zp).2
Proof.Gibbs’ inequality gives the first equivalence: a KL divergence is zero iff its two arguments coincide almost
everywhere. The second claim is the equality condition of Prop. 5.
TheELBOisthereforescaffoldingratherthantheobjectofinterest: drivingitsslacktozeroclosestheamortization
gap between the two conditioning distributions withouttightening the bound, and SDCC’s justification needs only
the former.
2The two targets coincide only when p(y⋆|y)isy-independent on the support of pθ(· |zp), i.e. when the outcome carries
no discriminative signal — a degenerate configuration, and not the regime of interest here.
21

The sleep-phase surrogate. Wake–sleep ( Hinton et al. ,1995) flips the sampling direction: instead of sampling
from qθand fitting it to pθ, it samples from the frozen psg(θ)and fits qθby maximum likelihood on those samples.
Proposition 8 (Thesleep-phaseupdateisaforwardKLandrecoversequation 10).LetLsleep(θ) :=Ey∼psg(θ)(·|zp)
−logqθ(y|
xp)
. Then
Lsleep(θ) = DKL
psg(θ)(· |zp)qθ(· |xp)
+H 
psg(θ)(· |zp)
, (24)
withH(·)theShannonentropy. Sincethestop-gradientmakestheentropy θ-independent, ∇θLsleep=∇θDKL(psg(θ)∥qθ) =
−Ey∼psg(θ)[∇θlogqθ(y|xp)], a plain cross-entropy score whose N-sample Monte-Carlo estimator has variance
at mostEy∼psg(θ)∥∇θlogqθ(y|xp)∥2/N— no q/pre-weighting appears. Summing over p∈ Cand adding the
task loss recoverstheboxedSDCC objectiveequation 10verbatim:
LSDCC(θ) =Ltask(θ;Hcomp) +λX
p∈CDKL
psg(θ)(· |zp)|{z}
teacher,stop-gradqθ(· |xp)|{z}
student,with-grad
. (25)
Proofsketch. Addandsubtractlog psg(θ)(y|zp)insidetheexpectationdefining Lsleepandidentifythetworesulting
expectationsasaKLdivergenceandanentropy; thatisequation 24. Differentiatingwiththetargetheldfixedleaves
the score expectation stated, whose estimator variance is its second moment; the bound is finite because qθhas
full support (softmax) and no ratio-based re-weighting inflates the summand. For the last claim, sum equation 24
overp∈ C, discard the θ-independent entropy sum, weight the KL sum by λ, add the task loss, and substitute
qθ(· |xp) =Pθ(· | ps(yp))pandpsg(θ)(· |zp) =Ptarget(· | pt(yp))with the notation of § 5.1.
The two directions have the same zero set.
Theorem 2 (Zero SDCC residual ⇔conditioning invariant) .Letεp(θ) :=DKL 
psg(θ)(· |zp)∥qθ(· |xp)
be the
per-leafforward-KLresidualthatSDCC minimizes. Then
DKL 
psg(θ)qθ
= 0 ⇐⇒ psg(θ)=qθa.e.⇐⇒ DKL 
qθpsg(θ)
= 0, (26)
and consequentlyP
p∈Cεp(θ) = 0if and only if the conditioning invariant equation CIholds at every diverging
leaf — in whichcase theELBO’sreverse-KLslackof Prop. 7iszeroas well.
Proof.Both equivalences in equation 26are Gibbs’ inequality: a KL divergence vanishes iff its two arguments
agree almost everywhere, and that condition is symmetric in the two arguments even though the two divergences
are not. A sum of non-negative terms is zero iff every term is, which lifts the per-leaf statement to C.
Away from the zero set the two divergences are different functions of θ, and the forward direction is the one that
can actually be descended in a single-backward training loop: the frozen teacher supplies the target distribution,
the gradient flows only through the student, and the update is an ordinary cross-entropy rather than a log (qθ/pθ)-
weightedscore-functionestimatorwhosevariancedivergesexactlyonthetokenstheevictiondamaged. FullELBO
tightness (Prop. 5) is strictly stronger than equation 26and is neither targeted nor required.
Existence of a zero-KL solution. The residual can be driven to zero at all only if the model can reproduce its
pre-eviction conditional from the compressed prefix alone.
Proposition 9 (Existence of a zero-KL solution) .If model capacity admits θ⋆withPθ⋆(· | ps(yp))p=Pθ⋆(· |
pt(yp))for all p∈ C, then the SDCC KL loss attains 0. This holds when the evicted span is redundant with the
surviving trunk, and fails when the span carries a strictly necessary bit absent from Hcomp, in which case εpstays
positiveand SDCC is bias-dominatedrelativetotheexact methods.
Proof.Each per-leaf term εp(θ)is non-negative, so K(θ) =P
p∈Cεp(θ)≥0with equality iff student and teacher
coincide on every diverging leaf (theorem 2); whenever capacity admits such a θ⋆the zero set ΘKis non-empty
and the loss attains 0. This is existence only — that SGD reaches it is theorem 4in section D.6, at rate O(1/√
T)
to the noise floor and exactly 0under the decaying-step-size schedule of corollary 3.
D.4 How the KL residual controls deployment and gradient bias
Infinitetrainingtheresidual εp(θ)issmallbutnonzero,sowhatmattersisnotthezerosetoftheorem 2butthechain
small KL ⇒small total variation ⇒small deployment deviation ⇒small policy-gradient bias . This subsection
states that chain once. Its first link is unconditional; only the last requires bounded advantage and score.
Proposition 10 (Behavioral Pinsker bound; restates and proves proposition 2).Forevery divergingleaf p∈ C,
pθ(· |zp)−qθ(· |xp)
TV≤q
εp(θ)/2. (27)
22

Proof.Pinsker’s inequality ( Tsybakov ,2009) states that ∥µ−ν∥TV≤p
DKL(µ∥ν)/2for any two distributions on
a common measurable space. Take µ=psg(θ)(· |zp)andν=qθ(· |xp), whose divergence is exactly the per-leaf
SDCC residual εp(θ); at convergence the stop-gradient branch equals pθand equation 27follows. No bounded-
score, same-advantage or parameterization assumption enters, so the bound is unconditional. Because the harness
redelivers zpat inference, the left-hand side is literally the deployment gap: the behavioral distance between the
context the update was computed under and the one the deployed policy is served.
Assumption 1 (Boundedadvantageandlast-layerscore) .(A1)∥ˆA∥∞≤Amax;(A2)thesamebehavioraladvantage
ˆA(y)isusedonboth pt(yp)and ps(yp);(A3)softmaxparametrizationwith ∥h∥∞≤Hmax,andthepolicygradient
is taken w.r.t. the last-layer parameters, for which ∇θlogπ(y|c) =hy(c)−Ey′∼π[hy′(c)]is uniformly bounded
by2Hmax; (A4) advantagesareaveragedoveran outerbatch.
Corollary 1 (Conditional gradient-bias bound) .Underassumption 1,theSDCCper-leafpolicy-gradientdeviation
isO(√εp);in aggregate,with C:= 5AmaxHmax,
∇θLtask(θ;Hcomp)− ∇ θL⋆
task(θ)≤CX
p∈Cq
εp/2 = OP
p∈C√εp
, (28)
whereL⋆
taskis the tree-consistent task loss that 4D masks and LogitTree compute exactly. Writing εKL:=max pεp,
thisis theper-leaf O(√εKL)form quotedin § 5.2.
Proof.Sketch: split the two policy gradients into a distribution-difference term and a score-difference term; (A3)
makes bothLipschitzin totalvariation,whichproposition 10converts into√εp.
Write π:=πθ(· | pt(yp)),eπ:=πθ(· | ps(yp)),sπ(y) :=∇θlogπ(y),g:=P
yπ(y)ˆA(y)sπ(y),and ˜ganalogously
with the same ˆAby (A2). Then
∥g−˜g∥ ≤P
y 
π(y)−eπ(y)ˆA(y)sπ(y)+P
yeπ(y)ˆA(y) 
sπ(y)−seπ(y)
≤2AmaxHmax∥π−eπ∥1+AmaxE
eπsπ−seπ. (29)
Totalvariationishalfthe ℓ1distance,soby(A1),(A3)andproposition 10thefirsttermisatmost 4AmaxHmaxp
εp/2.
In the second, (A3) makes the two scores share hyat the query token and differ only in the normalizer, sπ(y)−
seπ(y) =Eeπ[h]−Eπ[h], bounded by Hmax∥π−eπ∥TV≤Hmaxp
εp/2(scope: remark 1). Summing over per-leaf
residuals and per-trajectory tokens as in (A4) gives equation 28.
Remark 1 (Scope of the gradient corollary: last-layer restriction) .proposition 2is unconditional; only corollary 1
invokes assumption 1, where (A2) is the standard PPO/GRPO advantage-identifiability convention and (A3) sup-
plies the closed-form score. That identity needs θto index the softmax head, so the two contexts share hyat the
query token and differ only in the normalizer Eπ[h]. For full transformer parameters the score acquires an extra
∇θhy(c)termthatsoftmaxstructurealonedoesnotbounduniformly,andthecorollarythenneedsaseparatebound
on∥∇θhy(pt)− ∇ θhy(ps)∥, e.g. via Lipschitz assumptions on the backbone. This restriction is intrinsic to the
ratio-difference decomposition, not an artifact of our proof.
Corollary 2 (Ratetozerobiasundertraining) .UnderAssumptions(i)–(iv)of§ D.6,Theorem 4yieldsmint≤TE[εp(θt)] =
O(1/√
T);combined withcorollary 1,thedeploymentpolicy-gradientbias decaysat therate O(T−1/4).
Cost, and where SDCC sits relative to the exact methods. The bound above is what SDCC buys with its cost
profile. SDCC requires exactly one backward pass per trajectory against LogitTree’s K+1; the teacher forwards
share KV-cache trunks and are stop-gradient, so they retain no activations. A 4D mask is cheaper still in pure
FLOPs — one forward, one backward — but requires attention-kernel surgery and cannot be applied in black-box
harnesses where eviction spans are never exposed. The three methods therefore trade off along two independent
axes, correctness bias and infrastructure cost: the exact walks have zero bias at heavy cost, while SDCC pays
O(√εKL)bias — decaying during training by corollary 2— for one backward pass and no kernel changes. The
claim is not that SDCC dominates the exact methods: where their cost is acceptable, LogitTree and the 4D mask
are the correctness ground truth; where it is not, or in black-box regimes, SDCC is the only one of the three that
applies.
D.5 LogitTree and SDCC optimize the same objective up to a λ-tunable slack
The exact walks of Section 4and SDCC of Section 5charge a diverging leaf p∈ Cin visibly different ways:
LogitTree places the policy loss on the pre-eviction branch zp= pt(yp), so its per-leaf loss is an expectation under
the teacher distribution pθ(· |zp); SDCC keeps the policy loss on the compressed branch xp= ps(yp), so its
per-leaf loss is an expectation under the student distribution qθ(· |xp), augmented with a stop-gradient forward KL
that penalizes the two distributions for disagreeing (§ 5.1). The two constructions therefore differ in whichbranch
23

ofTenters the policy loss — LogitTree scores every branch (Prop. 3); SDCC scores only the deployed branch and
charges the others through the KL. Because the KL residual controls the behavioral gap between the two branches
(proposition 10), the two losses can be made arbitrarily close by choosing λ; this subsection makes that statement
precise as a two-sided bound.
Fixθandp∈ C; write εp:=εp(θ). Following the paper’s per-leaf normalization convention, let
LLT
p(θ) :=−E
y∼pθ(·|zp)ˆA(y)
,LSDCC
p(θ) :=−E
y∼qθ(·|xp)ˆA(y)
+λ εp
denote the per-diverging-leaf contributions of the two objectives; on non-diverging leaves zp=xp, both reduce to
the identical single-branch term, so a sum-over- Csuffices. ˆAis the same behavioral advantage on both branches
(Assumption 1, A2), and no rollout re-sampling appears.
Theorem 3 (Two-sided objective equivalence between LogitTree and SDCC) .UnderAssumption 1(A1),forevery
θ,every p∈ C,and every λ >0,LSDCC
p(θ)− LLT
p(θ)−λ εp≤Amaxp
2εp. (30)
Consequently,Young’sinequality( Amaxp2εp≤A2
max/(2λ)+λεp)yieldsa θ-uniformlowerboundanda θ-tracking
upper bound,
LLT
p(θ)−A2
max
2λ≤ LSDCC
p(θ)≤ LLT
p(θ) + 2λ εp+A2
max
2λ, (31)
and summing over p∈ CgivesLLT(θ)− |C|A2
max/(2λ)≤ LSDCC(θ)≤ LLT(θ) + 2λP
pεp+|C|A2
max/(2λ). On
the zero set {θ:εp(θ) = 0∀p∈ C}of theorem 2, equation 30holds with equality at 0, so LogitTree and SDCC
coincide exactly as functions of θ; the relaxed form equation 31still carries its A2
max/(2λ)slack there, Young’s
inequalitybeing loose at εp= 0.
Proof.Rewrite the difference of the two policy-loss terms as a change of measure and apply Hölder:
 
E
qθ(·|xp)−E
pθ(·|zp)
[ˆA(y)] =X
y 
qθ(y|xp)−pθ(y|zp)ˆA(y)≤ ∥ ˆA∥∞· ∥qθ−pθ∥1, (32)
and analogously for the reverse sign. By Assumption 1(A1) the leading factor is at most Amax; total variation is
half the ℓ1distance and proposition 10bounds it byp
εp/2. Together, |(Eqθ−Epθ)[ˆA]| ≤Amaxp2εp. Substi-
tuting into LSDCC
p− LLT
p= (Epθ−Eqθ)[ˆA] +λεpgives equation 30. Young’s inequality supplies Amaxp2εp=
2p
(A2max/(2λ))·(λεp/1)≤A2
max/(2λ) +λεp; adding and subtracting this on the two sides of equation 30yields
the clean form equation 31. On the zero set every εpvanishes, hence pθ(· |zp) =qθ(· |xp)by theorem 2and the
change-of-measure term equation 32is0, so equation 30is an equality between zeros.
What the bound says, and where it stops. Read in the maximization convention J:=−L, the lower bound of
equation 31becomes JSDCC(θ)−|C|A2
max/(2λ)≤JLT(θ): SDCC’sobjective,offsetbya θ-independent constant,is
alowerboundontheexactper-branchobjective , somaximizingitraisesthatboundatevery θandateveryresidual
εp,withtheoffsetshrinkingastheKLweight λgrows. Equivalently,inthelossconvention LSDCCplusthatconstant
majorizes LLT,sominimizingSDCCminimizesanupperboundontheexactwalk’sloss. Theupperboundislooser
— it retains a λP
pεpterm — because SDCC pays the extra KL even when the exact walk does not; on the CI
zero set that term is zero, so the two objectives coincide there. The classical VAE correspondence of App. D.2
reappears: the same λthat makes the lower bound tight is the βwhosetoo-large regime triggers posterior collapse
(Prop.6), soλcannot be sent to infinity naively. Two consequences follow. (i) The two objectives share a common
global minimizer set on the CI zero set of theorem 2, and their gradients on that set agree (corollary 1restates
the away-from-zero version under the same assumptions); this is the precise sense in which the two objectives are
“essentially equivalent”. (ii) The gap is not zero in general : it scales as λεpon the upper side, which is precisely
the main-text trade-off in § 5.1: SDCC buys single-backward compute at a residual that is θ-controlled by the KL
weight rather than structurally zero, and this residual is what theorem 4contracts at O(1/√
T).
D.6 Convergence: contraction toward the zero-KL set
Setup and assumptions. With θt∈Θ⊂Rdthe step- tparameters, Hcompthe compressed student input (§ 5.1)
and(H′
t,Hcomp)∼π, the task loss is J(θ)≜Eπ[Ltask(θ;Hcomp)]and the SDCC penalty is
K(θ)≜E
πhP
t∈CDKL 
Psg(θ)(·|H′
t)Pθ(·|Hcomp)ti
, (33)
thestop-gradientmaking Kdependon θonlythroughtheupdatebranch. Thecompositeobjectiveis Ft=J+λtK,
with step θt+1=θt−ηt(b∇J+λtb∇K)(θt)and schedule λt= 0fort < T warm,λt→λ⋆>0after. The post-
warm-up Lyapunov function Φ≜(J−J⋆) +λ⋆K,J⋆:=infθJ, is non-negative, zero at a joint minimizer, and
LΦ-smoothwith LΦ≤LJ+λ⋆LK. Weassumethroughout((i)–(iv)aredistinctfrom(A1)–(A4)ofAssumption 1):
24

(i)Smoothness: J,KareLJ-,LK-smooth.
(ii)Bounded gradientnoise: E∥b∇J∥2≤σ2
J,E∥b∇K∥2≤σ2
K.
(iii)Joint realizability: {θ:K(θ) = 0}is nonempty (Prop. 9) and meets argmin θJ; fixθ⋆in the intersection, so
Φ(θ⋆) = 0. This adds to Prop. 9the capacity condition that conditioning-invariance and task-optimality be
jointly attainable.
(iv)Composite Polyak–Łojasiewicz: ∥∇Φ∥2≥2µΦΦnearθ⋆for some µΦ>0; stronger than PL on Kalone, as it
excludes cancellations ∇J≈ −λ⋆∇KatΦ>0.
Only(iv)isrestrictive: near θ⋆bothHessiansareFishermatrices, hencePSD,andPSDalonedoesnotimplyPL.It
holdswhen Φislocallystronglyconvexalongdescent-relevantdirections(e.g.thebounded-scorelast-layersoftmax
of Cor.1); under PSD only, dropping the PL step of Theorem 4still leaves min tE∥∇Φ(θt)∥2→0, forfeiting the
linear rate on Φ(hence K) but not convergence to a critical point.
Warm-up. Fort < Twarmtraining is pure task-loss SGD, ε-stationary in O(σ2
J/ε2)steps under (i)–(ii) ( Ghadimi
&Lan,2013); asthecollapsed“ignoreeverything”solutionisnon-stationaryfor Jonnon-degeneratetasks, warm-
up moves off collapse before the KL is switched on, decoupling cold-start collapse from steady-state alignment.
The linear ramp used in practice replaces λ⋆by its running average, inflating the constants below by at most 2.
Steady-state contraction. Proofs here are sketches, each naming the standard step it invokes.
Theorem 4 (Steady-state contraction toward zero-KL set) .Under(i)–(iv)withµΦthe constant of (iv), constant
stepsize η=min(1/LΦ, c/√
T), and Tpost-warm-up SGD steps(distinctfromtherollout-end Tof§3),
min
t≤TE
Φ(θt)
≤Φ(θTwarm)
(µΦηT) + 
LΦη/2µΦ 
σ2
J+ (λ⋆)2σ2
K
, (34)
an initial-gap decay plus a noise floor. As Φ≥λ⋆Kpointwise, equation 34overλ⋆boundsmint≤TE[K(θt)] =
O(1/√
T)atη=c/√
T.
Proofsketch. LΦ-smoothness with (ii) gives the standard descent step EΦ(θt+1)≤Φ(θt)−η
2∥∇Φ(θt)∥2+
η2LΦ
2(σ2
J+ (λ⋆)2σ2
K)forη≤1/LΦ; telescope, divide by T, apply (iv). Only (iv) is non-routine: PL on K
alone would notsuffice, as ∇Jandλ⋆∇Kmay cancel, while (iii) forces J−J⋆andKto zero together.
EMA targets. With θ−
t+1=αθ−
t+ (1−α)θt+1replacing sg (θt),(θt, θ−
t)is a two-timescale stochastic ap-
proximation ( Borkar,1997): differencing the updates and applying discrete Grönwall gives E∥θ−
t−θt∥2≤
2η0(σ2
J+ (λ⋆)2σ2
K)/(1−α)whenever ηt≤η0and1−α≤η0. As the target KL is O(∥θ−
t−θt∥2)near
θ⋆in the Fisher metric, Theorem 4extends to EMA targets with its noise floor inflated by O(η/(1−α)), vanishing
asη→0. Composing it with the behavioral Pinsker bound of Prop. 2turns the KL rate into the deployment TV
bias against exact-walk (LogitTree / 4D) conditioning, quoted without proof in § 5.2.
Corollary 3 (Conditioning bias rate) .Under(i)–(iv), theSDCC policy πθTsatisfies
EπθT(·|H′
t)−πθT(·|Hcomp)
TV≤ O 
T−1/4
+O p
η σ2/λ⋆
, (35)
an optimizationterm plus a noise floor,bothdriventozeroby λ⋆= Θ(1)andηt= Θ(1/√
t).
Adding Assumption 1and replacing Prop. 2by Cor.1lifts equation 35to a policy-gradient bias rate with the same
O(T−1/4)scaling; without that assumption the TV bound still holds unconditionally.
Part III Experiment supplements: implementation, harness knobs, metric semantics,
training curves
This part supports § 6at the level of detail needed to reproduce every measured entry of Table 1from a fresh
container: the rollout–training contract shared by every cell of the matrix, the three white-box harnesses (TC-
RAG’s pop(), AgentFold’s periodic fold, MemexRL’s memory(op) ), the two black-box harnesses (Claude Code,
OpenCode),andthefivetraining-methodimplementationsintheSLIMEstack(§ E);thenthereproductionknobs,the
full per-harness aggregation of the Q1 offline probe, what logdiffdoes and does not certify under live compression
(§F.4), and the training curves with SDCC’s eviction-density profile (§ F).
E Implementation of the harnesses and the training methods
This appendix documents howeach of the five harnesses edits the live context at rollout time, and howeach of
the five training methods materializes its training input from the resulting record. It complements § F, which lists
25

the settings needed for reproduction (optimizer schedule, retriever backend, per-method and per-harness settings,
evaluation protocol); here the emphasis is on the mechanisms — what data structure each component maintains,
whatitemits, andwherethetreeof§ 3(formalizedin§ B)physicallyresidesinthecode. Allcomponentsruninside
the SLIME RL stack (Megatron actor +SGLang rollout engine).
E.1 The rollout–training contract
One record per rollout. Every harness, whether white-box or black-box, reduces to one per-rollout record that
all five methods consume identically:
Hcomp
|{z}
compressed walk, HT|{z}
physical union,{(Jk, Ek)}K
k=1|{z }
eviction records,
where Jkis the physical position at which the k-th compression fired and Ekis the set of physical positions it
removed (Definition 2). The compressed walk Hcompis exactly the token sequence the rollout engine rendered for
thefinalturn,and HTistheunionofalltokensthateverexistedintheliveview. Therecordtravelswitheachtraining
sample,togetherwiththeper-tokenlog-probabilitiesloggedbytherolloutengineatdecodetime(thereferenceside
of the logdiffdiagnostic of § 6.2).
The harness bridge. Eachwhite-boxharnessimplementsonebridgeadapterwithtworesponsibilities: (i) execute
the edit inside the rollout loop — rewrite the surviving message list before the next decoding step, so that the next
request payload sent to the rollout engine is truly the compressed view; and (ii) logthe edit as an eviction record
(Jk, Ek)in token coordinates of HT. Because the bridge is the only component that knows the harness’s internal
semantics (stack, fold, memory store), the downstream loss hooks are fully harness-agnostic: they see only the
record above.
E.2 Harnesses and methods
We now describe, at the level of the mechanism rather than the code, how each editor rewrites the live context and
howeachmethodreadstheresultingrecord. Theharnessandthemethodareselectedbytwoindependentswitches:
anymethodrunsagainstanywhite-boxharnesswithoutper-cellcodedivergence,whichistheengineeringproperty
behind the cross-harness sweep of § 6.5.
White-box editors. All three white-box editors follow the learnable-action route: wherever the original system
specifies a compression operation, we expose it to the policy as a first-class tool call, so the compression decision
itself is trainable. They differ in trigger and tree shape. TC-RAG (a learnable popover a stack of tool-result
envelopes): a popremoves the oldest envelope entirely, giving few junctions with heavy spans (mean |Ek| ≈
156tokens; Table 3); it is model-triggered, so an untrained policy leaves K= 0and the tree collapses to its
trunk.AgentFold (length-triggeredfolding): whentherenderedcontextexceeds 3,000tokensthepolicysummarizes
the foldable prefix and the prefix is replaced by that summary in the live view; the summary is on-policy and
receives gradients, while the folded prefix is logged as Ek(mean |Ek|≈84). Being length-triggered, it fires even
for an untrained policy, which makes it the natural pilot harness. MemexRL (an external key–value store with
memory_offload /memory_retrieve ): offload evicts a named span, retrieve re-injects it, and a terminal restore
pulls all stored entries back before the answer turn. This yields the richest tree — many small-span junctions
(mean |Ek|≈41) andnon-monotone spans (a token can leave and re-enter), each event logged so the live view is
reconstructible by replay. Per-turn state transitions are illustrated in Figure 3.
Black-box editors. ClaudeCodeandOpenCodearedeployedagentswhosecontextmanagement(slidingwindow
+auto-compact) is internal and unobservable: they surface the per-turn rendered transcript but not eviction spans
orsummaryprovenance. Runningthroughaseparateagent-platformpipeline, werecoverperturntheexactrequest
payload the framework sent to the model and convert it into the same per-rollout record, with two degradations:
junctions are detected as prefix divergences between consecutive payloads (the evicted span is not identifiable, so
we store the payloads rather than a token-set), and the teacher prefix is taken to be the previous payload. Only
per-turn LogitTree and SDCC are executable here — both need eviction events only to be detectable, not span-
exposed — whereas the 4D mask and the segmented K-forward require span-level records. Scoring uses the same
EM/format reward as the white-box cells.
The five methods. All five methods are standard SLIME loss hooks reading only the one record above; none
modifies the forward pass or the attention kernel. Naive-Compressed andNaive-Full use the reference policy loss
unchanged and differ only in the training sequence: Naive-Compressed trains on Hcompas rendered (condition-
ing post-junction tokens on later-created summaries, Pitfall A), while Naive-Full re-injects every evicted span to
reconstruct HT(conditioning on already-evicted content, Pitfall B); both roll out under live compression, so Ta-
ble1isolates the training-time conditioning choice alone. LogitTree splits the rollout at its Kjunctions into K+1
26

sub-samples, each carrying the live view as it existed during that layer and only that layer’s generated tokens;
SLIME trains them as independent samples ( K+1forward/backward passes) with ctrain
t=crollout
tby construction,
theepisoderewardinheritedunchanged. The4Dmask reusesthatsamebranchsplitbutfoldsitintoasingleforward:
the branches are packed into one sequence and the attention mask of Prop. 4admits, in each query row, exactly the
keys that were live when that token was decoded, with position ids restarted per branch so that RoPE phases match
the rollout. SDCCruns a student forward on Hcomp(with gradients, the Naive-Compressed compute path) plus one
gradient-freeteacherforwardperjunctiononthereconstructedpre-evictionprefix H′
t(Eq.9),andaddstheforward-
KLDKL(πθ(· | H′
t)∥πθ(· | Hcomp[:t]))only at diverging-leaf positions, with coefficient λ. The teacher shares the
student’s weights (no second model in memory), and the direction is fixed — student =compressed view, teacher
=pre-eviction view, never the reverse — because the student is scored on the compressed replay while the original
live conditional is the stop-gradient target (§ 5; §D).
Cost and applicability. The three consistency-restoring methods pay for the invariant along different axes (Ta-
ble6). LogitTree pays in compute :Kadditional backward passes per trajectory, a 5–20×per-step penalty in
deep-searchorcodingregimes( K= 5–20);itisbackbone-agnosticandalignsnativelywithvLLM/SGLangprefix-
sharing. The 4D mask needs one forward and one backward, but pays in infrastructure : the model must accept an
arbitrary per-row mask, position ids must be reassigned row-wise, and mask equivalence has to be re-checked at
serving time. SDCC keeps a single backward pass (though, with ℓK+1=C, no smaller forward budget) and needs
eviction events only to be detectable, so it is the only one of the three that also runs in the black-box case, at the
price of an O(√εKL)correctness residual.
Table 6: The three consistency-restoring methods vs. the two pitfalls, across dimensions that matter for production
RL stacks, with the per-rollout token budget of § Bfolded in: L=|HT|is the physical union, C=|Hcomp|the
compressed walk, and ℓithe live view in force during branch layer i, soC≤ℓi≤LwithℓK+1=C. Tokens are
not FLOPs (§ B).
Dimension Naive-Comp / Naive-Full 4D mask LogitTree SDCC
Restores conditioning invariant × ✓(dense only) ✓(any backbone) ✓up toO(√εKL)
Forward passes / trajectory 1 1 K+1 ≤K+1(Kstop-gradient)
Backward passes / trajectory 1 1 K+1 1
Tokens per trajectory C/L L(packed)∑K+1
i=1ℓi≥L ≤C+∑K
i=1ℓi
Custom attention kernel required no yes no no
Tree-structured data pipeline no no yes no (leaf mask only)
Position-id reassignment required no yes(row-wise) no no
Needs eviction spans exposed no yes yes(segmented) no (divergence suffices)
vLLM / SGLang alignment native × ✓ native
F Experiment Details
This appendix is designed to let an external reader reproduce every entry of table 1and table 9. The training stack
is SLIME (Megatron actor + SGLang rollout engine); we give the macroscopic setup below, with pinned versions
and every non-default setting shipped in the supplementary code.
F.1 Experimental Settings Details
Compute environment. AllrunsusetheSLIMERLframework(MEGATRON-LMactor +SGLANGrolloutengine)
on 80GB A100- or H800-class accelerators with CUDA/PyTorch and FlashAttention; pinned versions and the full
dependency list ship with the supplementary code.
Optimizer and schedule. Adam ( β1=0.9,β2=0.98) with constant learning rate 1×10−6, weight decay 0.01and
gradient clip 1.0; GRPO with group size G=16samples per prompt, clip range ε=0.2(low side) / 0.28(high side)
andKLcoefficient β=0.001(low-varianceestimator). ForSDCC,theconsistencycoefficient λrampslinearlyfrom
0to0.1over the first 20% of update steps and stays constant thereafter; we deliberately do not tune this schedule
per editor, so that no reported gain is an artifact of hyperparameter sweeping. The headline sweep uses a fixed step
budget per cell, whereas the convergence campaign trains each cell to a reward plateau (early stopping once the
trailing reward mean stops improving), capped at 200rollouts.
Rollout. At most T=30tool-call turns with top- k=3retrieved chunks per call, rollout temperature 1.0,n=16
samplesperpromptand 4promptsperrollout,giving 64trajectoriesperrollout,whichareconsumedin 8optimizer
steps of 8trajectories each; advantages are normalized within each 16-sample group (GRPO default), so the group,
not the optimizer step, sets the baseline.
27

Retriever and search backend. All rollouts — during both training and evaluation — issue live web searches
ratherthanqueryingalocalindex. WebsearchisservedbytheonlineDashScopetext-searchAPI,whichreturnsup
to10ranked results per query, and full-page evidence is extracted by a two-stage pipeline that fetches the retrieved
pages with Firecrawl and summarizes them with Qwen3-Turbo. Because the same endpoint serves training and
evaluation, the inference-time search distribution matches the training-time one.
Training corpus. ThetrainingcorpuspoolsREDSEARCHER( Chuetal. ,2026)andtheASEARCHERagentic-search
training set into 81,638composite QA instances, normalized to a uniform question–answer schema. Its composite,
multi-hop questions typically require at least two retrieval rounds, giving the model-triggered editors (TC-RAG
pop, MemexRL memory_offload ) and AgentFold’s length-triggered fold the opportunity to actually fire (see the
eviction-density profile in Table 10).
Evaluation benchmarks. Each trained checkpoint is evaluated on seven heterogeneous held-out open-domain
QA benchmarks ( 38,270questions per checkpoint in total), all run through the full agentic loop (multi-turn search,
live compression by the cell’s harness): NQ ( Kwiatkowski et al. ,2019) (3,610questions, single-hop factoid), TRIV-
IAQA (Joshi et al. ,2017) (11,313, single-hop trivia), HOTPOTQA ( Yang et al. ,2018) (7,405, two-hop), 2WIKI-
MULTIHOPQA ( Ho et al. ,2020) (12,576, structured multi-hop), MUSIQUE ( Trivedi et al. ,2022) (2,417, 2–4-hop
compositional), BAMBOOGLE ( Press et al. ,2023) (125, hand-curated compositional), and FRAMES ( Krishna et al. ,
2025) (824, multi-constraint retrieval-and-reasoning).
Harness knobs. The three white-box harnesses are selected by a single harness switch. All three compress live,
inside the rollout turn loop — the surviving messages are rewritten before the next decoding step, and the training
and evaluation code consume the same compressed view — and all three emit the same eviction-record contract
({Jk, Ek}), so the downstream loss hook is harness-agnostic. MemexRL is model-triggered, via a memory_
offload/memory_retrieve tool pair over a session-local store with a terminal restore before the answer turn
(§E.2);TC-RA G is model-triggered, keeping a stack of tool-result envelopes with a learnable popthat evicts the
oldest and no rule-based schedule (§ E.2);AgentFold is length-triggered, folding the prefix into a self-generated
summary once the live context exceeds 3,000tokens — the only rule-based trigger, and hence the highest-density
editor at cold start (Table 10; §E.2). Complete mechanism descriptions — action spaces, triggers, and what each
editor stores and restores — are in § E.2. For the two black-box harnesses (Claude Code, OpenCode) the rollout
channel is supplied by a separate integration pipeline outside this codebase; we consume its outputs in the same
per-rollout record format, so the SDCC / LogitTree loss hooks are unchanged (§ E.2), and the per-cell scoring uses
the same internal EM/format reward as the white-box cells, making the black-box column directly comparable.
Evaluation protocol. The reward (M1) is Exact-Match against the gold answer with a 0.2format bonus for cor-
rectly tagged but factually wrong answers (matching Jin et al. ,2025); Pass@1 (M2) is measured on the held-out
test split with a single rollout at temperature 0; drift (M3) is DKL 
πθ(· | H′
t)∥πθ,train(· |ctrain
t)
at inference time,
averaged over editor activations.
F.2 Q1 offline probe: full per-harness aggregation
Theper-tokendistributionsinFigure 1areplottedfromthesame32synthetic3-turnrolloutsperharness(untrained
Qwen3-4B,forwardpassesonly). Table 7reportsthefullper-harnessaggregationbehindthatpanel: samplecounts,
means, and majority-sign fractions for both directions of the pitfall ∆, together with the theoretical prediction each
rowmustsatisfyfortheinvariantof§ 3.3tohold. Everyharnessmeetsthepredictioninbothdirections. Thereading
is qualitative: leaves are clustered within rollouts and prompts, so the per-token means and |∆|magnitudes here
should be interpreted as sign-and-scale indicators rather than as unit-independent population estimates. We do not
attach confidence intervals to the means for that reason; instead we report the majority-sign fraction per harness as
the primary read. The qualitative pattern (aggressive-eviction editors amplify Naive-Comp, high-payload editors
amplify Naive-Full-leak) is discussed in the prose of § 6.3.
F.3 Method-input recipe under the same adapter record
All five methods construct their training input from the same per-rollout adapter record (Hcomp,HT,{Jk, Ek}K
k=1),
plus a No-comp (Search-R1) control that disables the editor (so Hcomp≡ HTand its drift is the numerical floor).
Table8summarizes the input, teacher-forward count, and extra bookkeeping. All five methods roll out under live
compression; Naive-Full re-injects the evicted text into the student input.
F.4 What logdiff certifies under live compression
Recall the four-quantity split of § 6.1Metrics: (1) the recorded-token logdiff, (2) the full-distribution condition-
ing KL, (3) SDCC’s training KL KL SDCC, (4) agentic EM. This subsection expands on (1). The logdiffstatistic
28

(a) No-comp control ( K=0 ): Hcomp≡HT.
sy1y2y3y4y5y6y7y8y9yano editor ever fires, so Tstays a single branch
Reference: training prefix =decode-time prefix.(b) Naive-Compressed Hcomp(Pitfall A).
sy1y2y3y4J1y6y7J2y9yay5 y8
!y6, y7conditioned on y5, scored without it.
(c) Naive-Full H T(Pitfall B).
sy1y2y3y4y5y6y7y8y9ya
!y9, yascored with y5, y8still in prefix.(d) TC-RA G: pop evicts oldest envelope.
before: s env1 env2 pop
JkEk=env1
after: s env2 y7y8ya× discarded
Heavy spans, few junctions (mean |Ek| ≈156).
(e) MemexRL: offload then retrieve .
before: s y1 y2 y3 offload
mid: s store y4 retrieve
after: s y4 y1 y2 y3 ya
Small spans; tokens leave and re-enter ( non-monotone ).(f) AgentFold: fold prefix into summary.
before: s y1 y2 y3 y4 y5 y6
live tokens >fold-threshold
JkEk=y1:6
after: s summary y7 ya
Rule-triggered; summary receives gradient.
Tokenroles: query s live token spine yaanswer token ya restored (MemexRL) evicted token ( ∈Ek)
Harness objects: envTC-RAG envelope summary AgentFold summary opharness action Jkjunction (edit event)
Arrows(events): leaves the live view restored to live view folded into summary leaked prefix (Pitfall B)
Figure 3: Six cases on the shared token-node vocabulary of Fig. 1, whose running example and subscripts these
panels reuse; J3falls inside the abridged tail ya. Panels split by role, not by row : Part I (a)–(c)are training-input
recipes on one rollout, exposing the Pitfall A “too-short” and Pitfall B “too-long” mismatches; Part II (d)–(f)draw
one live compression event per harness (before →after; Jkis generic, indices are panel-local). See § E.2for harness
code.
Table 7: Q1: the two pitfalls are real and directionally distinct. Untrained Qwen3-4B, 32 synthetic rollouts per
harness. Bothpitfallsexhibitthepredictedsignonallthreewhite-boxharnesses. ∆comp: TC-RAG/AgentFoldevict
mostaggressively(larger |∆|);MemexRLexposesthemostleaves( ncomp=4000)becauseitstwo-stepoffloaddelay
keepseachevictedspanalivefortwophysicalsteps. nfullisidenticalacrossharnesses( 3072)becauseallthreeevict
before the answer turn. Figure 1plots the underlying distributions.
∆comp(Naive-Comp vs. teacher) ∆full(Naive-Full-leak vs. current view)
Harness n mean frac. <0 n mean frac.>0
MemexRL 4000 −1.62 0.603072 +0.68 0.56
TC-RAG 928 −3.96 0.783072 +0.24 0.71
AgentFold 928 −3.81 0.753072 +0.34 0.55
Predicted <0(under) →1 >0(over) →1
compares, per response token, the training-side re-forward under the method’s training conditioning with the log-
probability recorded by the rollout engine at generation time. Under the live-compression protocol, the latter is
conditionedonthe compressedviewthatthemodelactuallysawatthatturn —sothetwosidesaredirectlycompa-
rable, and logdiffis a faithful per-step measure of the training–rollout conditioning gap for the recorded token. As
29

(a) The tap
Messages in, tokens out.
black-box harness
Claude Code / OpenCode
served policyrendered payload pk
decoded tokens
the harness compacts between turns;
the stream carries noeviction event(b) Consecutive payloads diverge
The first mismatch dates Jkand names Ek.
k=0sy1y2y3
J1:E1={y2}
k=1sy1y2y3summary y4y5y6
J2:E2={y5}
k=2sy1y2y3summary y4y5y6yans
rowkis the live view H′
tthe policy actually saw(c) One record, two uses
Same type a white-box run logs.
recovered T
one record per session
LogitTree
replay every
H′
texactlySDCC
align one walk
at each Jk
Figure 4: Recovering the trajectory tree from a black-box harness. (a) At the model boundary the only observable is one
(renderedpayload,decodedtokens)pairperturn: theharnesscompactsitsowncontextbetweenturnsandputsnoevictionevent
on the wire. (b)Consecutive payloads are laid on a single column grid in the token vocabulary of Figure 1, whose colour key
applies here unchanged; the one addition is the purple slot, text the harness wrote for itself, as in Figure 3. What turn kdecoded
(green) reappears as carried context (grey) in the next payload exceptwhere the harness dropped it, and a dropped slot leaves no
box at all. The first position at which two consecutive payloads disagree therefore dates the junction Jkand names the evicted
setEk(orangebandandarrow),sobotharerecoveredfromthemessagestreamalone,withnospan-levellogandnocooperation
from the harness. (c)Each row of (b) is a live view H′
t, so the recovered Thas the same type as the tree a white-box harness
logs, and the LogitTree and SDCC loss hooks consume it unchanged (§ E.2).
Table 8: Method inputs from the single adapter record. CC_TOKENS = compressed walk, UN_TOKENS = union.
Method Student input Teacher forwards? Extra bookkeeping
Naive-Compressed Hcompno —
Naive-Full HT no —
LogitTree (ours) per-segment slices no Ksegment boundaries
4D mask (ours) HT no 4D attn + hole-free posids
SDCC (ours) HcompK(no-grad) Kreconstructed teacher prefixes
a diagnostic, it certifies conditioning fidelity of the training forward, not end-task correctness; the efficacy verdict
is delivered by (4). The residual gap Naive-Compressed incurs is now precisely identified: a token emitted at turn
kwas sampled under the turn- kview, which still contained material that a latereviction removed; re-forwarding
the final compressed walk therefore conditions turn- ktokens on a history under which they were never sampled.
LogitTree( K-forwardonexactper-turnprompttokenids)andthe4Dmask(onepackedforwardrealizingthesame
conditioning)eliminatethisgap byconstruction ,andtheirobserved logdiffindeedreturnsto(orwithinnoiseof)the
no-compression baseline in every cell of Table 1. SDCC deliberately keeps the cheap Naive-Compressed forward
(and thus shares its elevated diagnostic on mismatched batches) and closes the gap through the training-KL objec-
tive (3); its progress is read off KL SDCC =KL(πtrain(·|Hcomp)∥πteacher (·|Hctx))(the training loss itself, § 5.1) plus
downstream EM (4). We therefore read the matrix as: logdiffranks conditioning fidelity of the training forward
(LogitTree =4D=baseline ≪Naive-Compressed, by construction for the two exact walks), while SDCC-KL and
EM rank end-task correction quality.
F.5 Training curves
This subsection plots the training dynamics behind the grand matrix. The white-box curves come from the 4B
RL runs on the pooled training corpus, one curve per method ×harness cell; the editor-free control enters this
subsection only as a numeric reference in the text. Figure 6additionally reports reward traces from the Claude
Code and OpenCode black-box runs. The curves diagnose optimization behaviour and are not themselves results:
Table1remains the sole source of final numbers.
Reward. Figure5shows all fifteen white-box cells rising over the windows it draws, by between 0.07and0.39
fromfirstdrawnpointtopeak, withtheper-iterationsignalnoisyenoughthatthesecurvescannotseparatemethods
by eye. We deliberately draw no “best” marker: the logged windows differ in length across cells, and the rescaled
axis aligns their endpoints rather than removing that difference, so any endpoint contrast would in part reflect
unequal training length rather than method. Note also that rollout/rewards is a post-normalization quantity, not
a reward: its per-cell median lies between −7.5×10−3and−1.0×10−8, i.e. centred on zero rather than on any
reward level, so it appears nowhere in this figure.
Figure6separatelyreportstheblack-boxrewardtracesforClaudeCodeandOpenCodeovertheirfirst 40optimizer
steps. These panels use their native step grids rather than the white-box normalized-progress axis. Their noisy,
short windowsand overlappingvariability bands makethem descriptivetraining traces, not a reward-basedmethod
30

0 50 100 150 200
normalised progress0.10.20.30.40.50.6mean raw rewardMemexRL
0 50 100 150 200
normalised progressTC-RAG
0 50 100 150 200
normalised progressAgentFold
Naive-Full
Naive-CompressedLogitTree (ours)
4D mask (ours)SDCC (ours)Figure 5: Mean raw reward over training, by harness. The axis is normalised progress: each cell’s iterations
are rescaled by x= 200iter/itermax, with iter maxthat cell’s last logged iteration, so iteration 0sits at x= 0.
Onex-unit is therefore a different number of iterations in each curve. Thick lines are a centred rolling median of
rollout/raw_reward over a±15-iteration window, taken on the true iteration grid before the rescale; the band is
that window’s interquartile range.
(a) Claude Code
 (b) OpenCode
Figure 6: Black-box training reward in the first 40 steps. Mean raw reward for SDCC (red) and the GRPO
baseline (blue); shaded bands show the variability recorded by the training runs.
ranking.
Conditioning drift. Section F.4states what logdiffcertifies; this paragraph states how we measure it. Every
number below is a pooled median over steadysteps, defined by one rule applied uniformly to every cell: within
each launch we discard the first 20optimizer steps and retain only launches with at least 10steps remaining. The
discardisnecessarybecausethestatisticreadsmorethananorderofmagnitudehighuntilthetrainerandtherollout
enginehavere-aligned;conditioningonthewithin-launchstepindexremovesmostbutnotallofthat,andwereport
the residual rather than claim it away.
Training length is a second confound, and it is the one that decides the ordering. Because the editor-free control
covers iterations up to 58, a median taken over each cell’s own window would compare cells trained for different
lengths. We therefore fix the comparison window in advance to the window the control covers — steady steps at
iteration ≤58, against the control’s median there ( 0.0138,n=448). Ten of the fifteen cells sit on the floor at 0.93–
1.04×thecontrol; Naive-Full’stwoeviction-heavycellssitjustaboveit,bothat 1.19×; andthreecellsareelevated:
Naive-Compressed ×AgentFold 2.17×, Naive-Compressed ×TC-RAG 2.69×and SDCC ×AgentFold 4.51×. That
31

Table 9: Q2: 5-method comparison, Qwen3-4B / MemexRL. Drift from the live-compression MemexRL cell;
Search-R1 drift from the editor-free control (numerical floor). EM columns are seven-bench macro-EM under the
agentic-eval protocol (§ F.1);†rows are grand-matrix same-source runs (cross-harness EM in Table 1).
Method EM↑Drift↓Cost (×base)
Search-R1 (no-comp.) 23.0†0.0135 1.05
Naive-Compressed 28.9†0.0237 1.00
Naive-Full 32.1†0.0163 1.05
LogitTree (ours) 45.9†0.0133 4.20
4D mask (ours) 33.4†0.0140 1.35
SDCC (ours) 43.1†0.0165 1.55
istheorderingthemaintextpredicts—thetwoeviction-heavyharnessesseparateNaive-Compressedfromthefloor
and MemexRL does not ( 0.99×) — and SDCC is elevated by construction, since it keeps the cheap compressed
forward and corrects through its KL term. Restricting to the eleven cells that cover the window to within a single
iteration leaves the floor band at 0.94–1.04×and moves no cell across a group boundary.
Onthecold-startMemexRLcells—wherecompressionbarelyfires—allfivemethodsstaywithin 0.0115–0.0151
overtheirfullwindows(Naive-Compressed 0.0133),bracketingthecontrol’s 0.0138,exactlyastheeviction-density
argument of Section F.6requires. One caution on reading these numbers against the rest of the paper: a median
over per-iteration medians, a median pooled over a cell’s steady steps, and the drift column of Table 1are three
different estimators of the same quantity, none numerically interchangeable with another. Only the pooled form is
used here.
F.6 SDCC training dynamics and eviction-density profile
Q2 headline 5-method table. Table9providesthe5-methoddrift/EMcomparisononthelow-evictionMemexRL
anchor cell at 4B; the drift column is measured under the live-compression protocol, and the three daggered EM
rows(Search-R1,Naive-Compressed,Naive-Full)arepopulatedfromthesameseven-benchagenticrunsthatsupply
Table1. The three ours rows (LogitTree/4D/SDCC) already show the drift ordering; cross-harness EM support
appearsinTable 1: ontheeviction-heavyharnesses, 4DmatchesorexceedsNaive-CompressedEM(ahead 34.2vs.
33.2on AgentFold, ahead 37.2vs.36.7on TC-RAG) while its drift stays at the no-compression floor ( 0.017–0.021
vs. Naive-Compressed’s 0.054–0.058), so ours does not trade EM for drift consistency. MemexRL is the cold-start
regime ( 8/256sessions evict), so Naive-Comp drift here is at the floorof the eviction-density scaling reported in
Table1(main text).
Selectivity of the SDCC regularizer under live compression. Figure7trackstheon-lineevolutionoftheSDCC
regularizer on the AgentFold cell — the eviction-heavy editor (length-triggered fold at 3,000 tokens; compression
ratio 0.06–0.15), and hence the cell in which the KL contract has the most work to do. Four regularities emerge.
(i) The KL fires sparsely and precisely. KLSDCCisnon-zeroonslightlyoverhalfoftheloggedoptimizerstepsand
exactly zero elsewhere, and the non-zero steps are exactly those whose batches contain live folds. Conditioning
on the gate separates the diagnostic by a factor of 5.6in median: steps with KL SDCC >0have median logdiff
0.081(IQR 0.058–0.125), whereas steps with KL SDCC = 0sit at the no-compression baseline, median 0.015
(IQR 0.014–0.017). The regularizer engages at mismatched positions and only at mismatched positions — the
per-leaf gating of § 5operating as designed.
(ii)The contract stays armed for the duration of the run rather than saturating or dying: the diverging-leaf set is
non-empty on a majority of micro-batches throughout, so the KL term never becomes structurally inert.
(iii)The SDCC KL residual does not grow over the run, while KL refagainst the reference model grows smoothly
from 0.002to≈0.03: the policy moves but does not detach from the reference, and no run-away is observed
without an entropy bonus.
(iv)Mean raw reward rises over the same window in which the training-KL residual does not grow — the training-
time consistency signal and the eval-time task signal are not in tension — which motivates the head-to-head
against Naive-Compressed in Table 9: absent SDCC’s regularizer, the same policy optimizes against physical
logits it never trains on. The load-bearing part of that comparison is structural rather than numerical: the two
armssharethesamepolicy-losscodepathmodulotheKLterm, i.e.Naive-Compressed =SDCCat λ=0(§6.4).
Magnitudes here come from one editor and are not offered as cross-method effect sizes.
Thiscelliscold-start: theuntrained4Bpolicytriggersevictionsinonlyafractionofsessions,sotheKLmagnitudes
are small in absolute terms. The point establishedhere is not the magnitude but the selectivity —εKLis confined to
exactlythedivergingleaves,sothe O(√εKL)boundof§ 5.2isnon-vacuousfromthefirststepsoftraining. Figure 7
32

0 20 40 60 80 100 120
training step0.0000.0050.0100.0150.0200.0250.0300.0350.040KLSDCC
(a) KLSDCC: sparse spike train
0 20 40 60 80 100 120
training step0.00.20.40.60.81.0diverging-leaf activity(b) leaf activity and effective λ
0 20 40 60 80 100 120
training step0.00.10.20.30.40.5logdiff(c) logdiff by gate state; reward
logdiff
median, KL>0: 0.081
median, KL=0: 0.015
raw reward
0.000.020.040.060.080.10
effective λ
0.000.050.100.150.200.250.300.350.40
raw reward
Figure 7: SDCC dynamics on Qwen3-4B × AgentFold under live compression. (a) KL SDCCis a sparse spike
train, non-zero on slightly over half of the logged optimizer steps and exactly zero elsewhere. (b) Diverging-leaf
activity and the effective KL weight λ. (c) logdiffrises on eviction batches (median 0.081) and sits at the no-
compression baseline otherwise (median 0.015); mean reward drifts upward.
plots the co-evolving signals: (a) the sparse KL spike train; (b) the diverging-leaf activity and the effective KL
weight; (c) logdiffspiking on eviction batches and returning to baseline elsewhere, with reward drifting up.
Per-method mechanism activation under live compression. Table11reports, for each method, the run-mean
ofthequantitythatitsowncorrectivemechanismcontrols,perwhite-boxeditor. Thisisnotanoutcome(EM)table:
itestablishesthateachmechanismengagesat4Bscaleunder live,model-triggeredcompression,andquantifieshow
strongly. Two things stand out.
(i)Foreveryeditor, eachcorrectivemethod’sownchannelisnon-zero: LogitTreesplitseachrolloutinto ¯K= 3.2–
4.1segments, the 4D packer activates on 1.2–1.4sub-samples per micro-batch, and SDCC’s leaf gate is armed
on a large majority of micro-batches. The Naive rows have no such channel by construction.
(ii)SDCC’s KL magnitude tracks the editor’s eviction density: ≈10−5on MemexRL (8/256 sessions evict at cold
start), 1.1×10−4on TC-RAG, and 2.9×10−3on AgentFold (132/960 sessions fold; compression ratio 0.10).
The correction therefore scales precisely with how much the editor actually rewrites — consistent with the per-
leaf gating derivation of § 5.
Persistence of Pitfall A under training. A natural hope is that the naive-compressed drift will self-correct as
GRPO adapts the policy to the compressed prefixes. The AgentFold cell gives no sign of that correction: over the
logged window the per-third mean of logdiffmoves 0.069→0.045→0.064, and the count of eviction-heavy steps
(logdiff >0.05) does not fall — the mismatch neither shrinks nor is absorbed, because its source (folds keep firing
onnewrollouts)remainsstationary. Onthesamecell,LogitTree’s logdiffstayspinnedat 0.013–0.015ineverythird:
the exact correction removes the drift at its source rather than asking the optimizer to absorb it. This window is
early-trainingandthereforelow-eviction;attheargmax-rewarditerationofthefullrunthesameNaive-Compressed
×AgentFold cell reads 0.366(Table1). The persistence claim is thus made at the conservative end of the density
range.
0 20 40 60 80 100 120
training step0.000.050.100.150.200.250.300.35logdiff  |logtrain logrollout|
SDCC (rolling mean)
Naive-compressed (rolling mean)
No-compression floor (0.0135)
Figure8: SDCC’s conditioning gap decreases over the logged training window; Naive-Compressed’s does not.
Per-step logdiffonQwen3-4B ×AgentFoldunderlivecompression. SDCC(red)trendstowardtheno-compression
floor (green dashed, 0.0135); Naive-Compressed (grey) drifts without directional trend; LogitTree stays pinned at
0.013–0.015.
33

SDCC’s logdiff decreases over the logged window. SDCC shares Naive-Compressed’s inexpensive student for-
ward, so both begin with an elevated diagnostic — but only SDCC includes a mechanism that should shrinkit.
Figure8tests this on the same eviction-heavy AgentFold cell: SDCC’s mean logdifffalls by 55% from the start to
the end of the logged window and does so monotonically across successive quarters, whereas Naive-Compressed’s
shows no comparable directional trend. SDCC starts 60%aboveNaive-Compressed (its leaf-gated KL initially
perturbs exactly the mismatched positions) and ends belowit, still descending toward the no-compression floor of
0.0135at the end of the logged window. This is one editor at one scale; it is the qualitative counterpart of the order-
ing predicted in the main text — LogitTree/4D own the floor by construction, SDCC approaches it by optimization,
and Naive-Compressed has no route there at all — and not a trend estimate across seeds.
Table10: Live eviction-density profile per editor. Triggeranddensitystatistics;driftfromtheNaive-Compressed
cell; KL from the SDCC cell (mean ×10−3). Both diagnostics scale monotonically with eviction density.
eviction density Naive-Comp SDCC KL
Editor Trigger sess. w/ evict ratio mean drift ×10−3
MemexRL model tool ( memory_offload )8/256 0.012 0.024 0.01
TC-RAG model tool ( pop) 17/320 0.024 0.039 0.11
AgentFold length fold (3k tokens) 132/960 0.104 0.059 2.86
Table 11: Per-method mechanism activation under live compression. Run-mean signature channels per white-
box editor. seg= LogitTree segments per rollout; pack= 4D packed sub-samples per micro-batch; KL = SDCC
leaf-gated KL ( ×10−3);act= fraction of micro-batches with a non-empty diverging-leaf set. —= channel not
applicable to that method.
Method Channel MemexRL TC-RAG AgentFold
Naive-Full — — — —
Naive-Compressed — — — —
LogitTree (ours, K-forward) seg 4.10 3.90 3.23
4D mask (ours, packed) pack 1.44 1.28 1.16
SDCC (ours) KL / act0.01 / 0.99 0.11 / 0.87 2.86 / 0.82
Compression frequency and depth. The depth column of Table 1and its companion factor freq =Pr[ci>0]
are both measured per episode over each cell’s full training run; their product is the single compression rate a one-
number summary would report. We keep freq out of the main table because it is not one quantity across columns:
AgentFoldfoldsonalengththreshold,soitsfreqcountsepisodesthatoutgrow 3krenderedtokens,whereasTC-RAG
andMemexRLevictonlywhenthepolicyemits memory_offload orpop,sotherefreqcountsapolicydecisionand
not a length — comparable down a column but not across. depth has one meaning everywhere. Across the fifteen
trainedcellsfreqvariesbymorethananorderofmagnitudewhiledepthvariesbyunder 2×, sotheproductisclose
to a monotone rescaling of frequency and inherits its harness-dependence, whereas the two factors reported apart
show what it hid: whichever editor fires removes about three fifths of the rendered context. Since depth conditions
on the episodes that evict, it is N/A, not 0, for the no-compression control.
Eviction-density profile across editors. Thethreewhite-boxeditorsspantwoordersofmagnitudeinliveeviction
density (Table 10): MemexRL and TC-RAG are model-triggered (the policy must decide to call memory_offload
/pop), so from a cold start the untrained 4B model rarely invokes them, whereas AgentFold is length-triggered
and fires on every long rollout, folding physical contexts of up to 121k tokens into ≤5.2k-token logical views. The
low-densitycellsserveasin-runnegativecontrols: asevictiondensityfalls, Naive-Compressed’sdriftandSDCC’s
KL both collapse toward the no-compression baseline — confirming that the pitfall is compression-driven, not an
artifact of harness dispatch.
G Implementation note: branch-replicated packing vs. the physical-union view
The main text (§ 4.2, Eq.7) presents the 4D mask as a per-row visibility schedule over the physical union HT, in
which each token appears once and different queries see different subsets of the same hidden states. Our implemen-
tation instead follows LogitTree’s branch decomposition: it materializes one copyof each shared-trunk token per
branch, so that every copy carries a hidden state computed from its branch-local causal prefix alone. The copies
are then packed into a single sequence with a block-diagonal attention mask that prevents cross-branch interaction;
position ids restart per branch.
34

Under dense softmax attention and the five conditions of Proposition 4, the two constructions produce identical
per-target logits and gradients: the physical-union view is the row-wise characterization of what the block-packed
execution computes. We retain the physical-union formulation in the main text because it gives a compact, closed-
form mask definition (Eq. 7) and a direct proof path (§ C.2); the branch-replicated form is what the training loop
executes.
The token budget in Table 6reports L(the physical union length) as a lower bound; the actual packed length is
Npacked =P
πℓπ≥Ldue to trunk replication, and Table 1’s “max tok.” column reflects the true packed input
size.
35
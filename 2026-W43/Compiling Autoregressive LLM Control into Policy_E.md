# Compiling Autoregressive LLM Control into Policy-Exact Parallel Inference for Hierarchical RAG

**Authors**: Zehao Lin, Qianfeng Shen, Xixuan Hao, Shaobo Cui, Chunyu Li, Kai Chen, Bo Tang, Zhenjun Wang, Jin Yan, Zhiyu Li, Feiyu Xiong

**Published**: 2026-10-22 03:25:21

**PDF URL**: [https://arxiv.org/pdf/2610.30664v1](https://arxiv.org/pdf/2610.30664v1)

## Abstract
Large language models increasingly act as controllers in structured decision pipelines, yet their decisions are still executed by autoregressive text decoding. Hierarchical RAG is one such setting: at each frontier of a document tree, an LLM commits to $b$ of $k$ sections under an input-dependent action grammar, executed as either failure-prone free-form JSON or reliable but strictly serial grammar-constrained generation. We show that this serialization is unnecessary for bounded-horizon control: the reachable future forms a shallow action-prefix tree, which we compile --- one root prefill plus one tree-causal probe evaluate every prefix-conditioned action distribution with serial decoding's tokens, positions, and causal ancestry. A prefix-state lemma makes compiled control policy-exact in exact arithmetic: the same policy with a two-forward critical path and zero output tokens. One further probe pass certifies the exact order-marginalized MAP set. Across four document-disjoint splits and two scales on the primary MetaX device, compiled control matches the serial policy on $84$--$89\%$ of questions --- every instrumented divergence a floating-point near-tie --- and cuts control latency by $35$--$47\%$ versus token-wise serial decoding and $6$--$7\%$ versus an idealized chunked engine; an H800 repetition preserves the ordering.

## Full Text


<!-- PDF content starts -->

COMPILING AUTOREGRESSIVE LLM CONTROL INTO POLICY-EXACT
PARALLEL INFERENCE FOR HIERARCHICAL RAG
Zehao Lin1, Qianfeng Shen2, Xixuan Hao1,3, Shaobo Cui4, Chunyu Li1, Kai Chen1,
Bo Tang1, Zhenjun Wang2, Jin Yan2, Zhiyu Li1∗, Feiyu Xiong1
1MemTensor (Shanghai) Technology Co., Ltd.2China Haisum Engineering Co., Ltd.
3The Hong Kong University of Science and Technology (Guangzhou)4Shanghai Jiao Tong University
{linzh,chenk,licy,tangb,lizy,xiongfy}@memtensor.cn, shaobo.cui@sjtu.edu.cn
{shenqf,wangzhenjun,yanjin}@haisum.com, xhao390@connect.hkust-gz.edu.cn
ABSTRACT
Large language models increasingly act as controllers in structured
decision pipelines, yet their decisions are still executed by autore-
gressive text decoding. Hierarchical RAG is one such setting: at
each frontier of a document tree, an LLM commits tobofksec-
tions under an input-dependent action grammar, executed as either
failure-prone free-form JSON or reliable but strictly serial grammar-
constrained generation. We show that this serialization is unnec-
essary for bounded-horizon control: the reachable future forms a
shallow action-prefix tree, which wecompile— one root prefill plus
one tree-causal probe evaluate every prefix-conditioned action dis-
tribution with serial decoding’s tokens, positions, and causal ances-
try. A prefix-state lemma makes compiled controlpolicy-exact in ex-
act arithmetic: the same policy with a two-forward critical path and
zero output tokens. One further probe pass certifies the exact order-
marginalized MAP set. Across four document-disjoint splits and two
scales on the primary MetaX device, compiled control matches the
serial policy on84–89%of questions — every instrumented diver-
gence a floating-point near-tie — and cuts control latency by35–
47%versus token-wise serial decoding and6–7%versus an ideal-
ized chunked engine; an H800 repetition preserves the ordering.
Index Terms—retrieval-augmented generation, constrained de-
coding, parallel inference, LLM control, tree attention
1. INTRODUCTION
Hierarchical retrieval [1, 2, 3, 4] turns evidence gathering into se-
quential control: an LLM repeatedly selectsbofk≤20candi-
date sections and descends into the selected subtrees. The action
space is finite and input-dependent, yet deployed systems execute
this controller bydecoding text. That choice imposes a three-way
dilemma. Free-form JSON leaves a protocol surface that dominates
the error budget (3–65%of questions fail on syntax, identifiers, or
cardinality in our companion interface study). Scoring allkverified
single-token labels in one prefill — the unary shortcut — removes
the protocol surface but also the controller’s memory: selection can
no longer condition on earlier selections. Grammar-constrained de-
coding [5, 6] keeps the history-dependent policy and syntactic va-
lidity, but pays a serial critical path ofbaction tokens plus forced
syntax tokens, one model forward each.
Our central observation is that this dilemma is an artifact of the
execution model, not of the controller. A constrained controller with
∗Corresponding author.horizonbhas abounded reachable future: every state it can visit
is a prefix over at mostbdistinct actions — for greedy control at
our deployed horizon of three actions, at mostkstates remain once
the first is fixed. We thereforecompilethe action grammar into its
prefix tree and evaluate all reachable states at once: one root prefill
resolves the first action; onetree-causal probe— a single forward
pass over the concatenated prefix continuations, with a block-causal
mask and serial position indices — returns every prefix-conditioned
action distribution the controller can consult. Because each tree state
sees exactly the tokens, positions, and causal ancestry it would see
under serial decoding, its logitsarethe serial controller’s logits, and
the greedy chain is recovered arithmetically. We do not approximate
or distill the serial controller; we recompute it on a shallower exe-
cution graph (finite precision can still flip near ties; Sec. 4.1). The
critical path drops fromb+m syndependent forwards to two, inde-
pendent ofb; no token is ever generated; the protocol surface is zero
by construction.
Compilation also upgrades what the controller can decide. The
probe exposes the controller’sjointpreferences, so unordered multi-
action selection becomes exact posterior inference rather than a se-
rialization artifact: we marginalize action order and certify the MAP
set with a branch-and-bound certificate — no truncation heuristics.
And the probe compares the two policies: the AR-vs-unary KL di-
vergence bounds — via Pinsker — the expected-utility loss of the
unary shortcut per input. The construction applies to any finite-
action constrained LLM controller (tool choice, routing, agenda se-
lection); hierarchical RAG is our case study.
Contributions.(1)Finite-action control compilation: constrained
controllers executed as one root prefill plus one tree-causal probe,
zero decoded tokens (Sec. 3.2). (2)Policy-exactness theory: prefix-
state equivalence and policy equivalence under matched tokens, po-
sitions, masks, and tie-breaking, with serial depth exchanged for
bounded parallel width (Sec. 3.3). (3)Certified set inference and
interaction mass: an order-marginalized set posterior with a branch-
and-bound MAP certificate, and a bound on the expected cost of
unary execution (Sec. 3.4). (4)Evaluation: four document-disjoint
splits, two scales, two accelerators, and a hierarchy-corruption stress
test — policy fidelity,100%certification in non-trivial rounds, and
35–47%lower control latency (Sec. 4).
2. RELATED WORK
Hierarchical RAG as control.Tree- and memory-structured re-
trieval [1, 2, 3, 4, 7] steers an LLM through document structure;
retrieval grounding remains a standard hallucination mitigation [8],
arXiv:2610.30664v1  [cs.CL]  22 Oct 2026

(a) Serial grammar decoding
1[2["3["A4["A"5["A",⋮⋮9["A","C"](b) Compiled: root prefill +ONE tree-causal probe
shared prompt KV · cached, zero copies
["
A
⟨sep⟩trunk
A
⟨sep⟩
B
⟨sep⟩
A
⟨sep⟩
C
⟨sep⟩
A
⟨sep⟩
D
⟨sep⟩…
prompttrbr₁br₂trbr₁br₂
serialb + m_syntax forwardscompiled1 prefill + 1 probe(c) Measuredfresh split D, 7B — latency / questionserial grammar0.56 scompiled0.36 s−36%unary (bound)0.29 s00.20.40.6 sVERIFIED0output tokens0protocol failures0.05mean logit deviation, fp32 replay (split A)
critical path ∝ emitted lengthFig. 1. Compile, don’t decode. (a) Serial grammar decoding spends one dependent forward per emitted token (b+m syn≈9). (b) Compilation:
one root prefill, then a single tree-causal probe evaluates the trunk and every candidate branch under a block-causal mask — each row sees
only the cached prompt and its own ancestors, so its logits equal the serial controller’s in exact arithmetic (Lemma 1). (c) Fresh split D, 7B:
mean control latency−36%, zero output tokens and protocol failures; fp32 replay (split A): mean graph deviation≈0.05.
and evidence quality is sensitive to structure and OCR noise [9,
10]. A companion interface study (under review) locates this stack’s
dominant failures in the protocol layer; here we change the execution
of its strongest constrained controller.
Constrained decoding and logit selection.Grammar- and schema-
constrained decoding [5, 6, 11, 12] enforces validity but keeps the
serial path; first-token logit selection [13] avoids decoding but aban-
dons the sequential policy, inheriting label-position sensitivities [14,
15]. Compilation sits between the two: it preserves the sequential
policy at the cost of one additional forward pass.
Tree-parallel execution.Token-tree verification is an established
primitive in speculative and multi-head decoding [16, 17], with IO-
aware kernels [18]. Speculationacceleratesdecoding — draft, ver-
ify, reject. Our compiled tree is not a guess but theexhaustivereach-
able state space of a bounded-horizon control grammar: nothing to
reject, no decoding left to accelerate, and an equivalence that is prov-
able rather than distributional.
Parallel structured selection.Contextual Plackett–Luce learns
pairwise interaction heads for subset selection [19]; mean-field
parallel decoding coordinates token commitments via predictive
interactions [20]. Both approximate or learn joint structure; we eval-
uate the frozen model’s exact prefix conditionals without training
and certify the MAP. SimpleTool retrains the function-call represen-
tation for parallel fields [21]; LLMCompiler parallelizesexternal
call orchestration [22]; neither compiles a frozen controller’s own
conditionals, as we do exactly.
3. COMPILED AUTOREGRESSIVE CONTROL
3.1. The serial constrained controller
At a traversal round with candidates bound to contextually verified
single-token labelsA x(|Ax|=k; larger frontiers are batched de-
terministically), the grammarG xforces a JSON array of exactly
n= min(b, k)distinct labels: forced syntax segments alternatewith label slots masked to unused labels. Serial constrained decod-
ing emitsσ= (a 1, . . . , a n)greedily,
at= arg max
a∈Ax\{a<t}z 
a|x⊕ρ t
,(1)
withρ tthe emitted token prefix before slottandzthe next-token
logits. Each of then+m synemitted tokens costs one dependent
forward: critical pathD serial =n+m syn≈9forn=3. (Even
an idealizedchunkedengine that fast-forwards known forced spans
still pays one dependent forward per label slot,D=n; we measure
it as a second baseline in Sec. 4.) Selected nodes’ children form the
next frontier; selected sections are packed into the reader budgetB
in selection order.
3.2. Compilation to an action-prefix tree
Oncea 1is fixed, the greedy controller can only ever consult
N(k, n) =Pn−2
d=0(k−1)!/(k−1−d)!(2)
prefix states — forn=3, exactlyk: the trunk(a 1)and the branches
(a1, c),c̸=a 1. Compiled execution is therefore two steps.Root
prefill: one pass over the prompt plus the forced opening syntax
yields slot-1 logits and the shared KV cache;a 1resolves by Eq. (1).
Tree-causal probe: theN(k, n)prefix continuations — each the ex-
act forced token sequence serial decoding would emit — are con-
catenated into one query of a few hundred tokens and evaluated in
a single forward under a block-causal mask: every position attends
to the cached prompt and its own ancestors only, with serial position
indices (branches may share or replicate the trunk; attended key–
value sets coincide). The last position of each row is that prefix’s
action distribution;a 2, a3follow arithmetically. The critical path
isD comp = 2forwards for anyn, at parallel widthΘ(N(k, n))
probe rows: serialdepthis exchanged for bounded parallelwidth,
and compilation is profitable whenever
Tprobe 
N(k, n)
<Pn+m syn
t=1 Tstep,(3)

the regime of everyn≤3configuration we measure at the deployed
frontier width (Sec. 4.4).
3.3. Policy exactness
Lemma 1 (prefix-state equivalence).If tree-packed and serial eval-
uation give a prefixρthe same token ids, position ids, ancestor-
visible attention set, and parameters, then all hidden states overρ
coincide at every layer, hencez tree(· |x, ρ) =z serial(· |x, ρ).
Proof:induction over layers: embeddings agree; attention at each
position reads an identical key/value set by the mask construction;
MLPs and normalizations are deterministic per-position maps.□
Theorem 1 (policy equivalence).With identical admissible-label
masks, candidate order, and first-maximum tie-breaking, compiled
and serial greedy control select the same action sequence for ev-
ery input:π comp(x) =π serial(x).Proof:by Lemma 1 the two
graphs produce identical logits at every slot; identical masks and tie-
breaking then make each argmax identical, and induction over slots
propagates the shared prefix.□
Corollary 1 (margin stability).If a slot’s decision margin under ei-
ther graph exceeds2ε max, whereε maxbounds the cross-graph logit
deviation, then the compiled and serial decisions at that slot coin-
cide.Margins come at no extra cost from the probe, so decisions in-
side the flip window can be flagged online against a deviation bound
calibrated on replays.
Exact arithmetic is the premise; hardware executes the two
graphs with different floating-point reduction orders, so logits de-
viate by a data-dependentεand near-tie slots can flip. This is a
property of precision, not of compilation: empirically, every ob-
served divergence sits at a slot whose margin lies within twice the
observed deviation — the window where Corollary 1 permits flips
(Sec. 4.1) — so the two executions coincide up to the precision the
deployment already accepts.
3.4. Certified set inference and interaction mass
The reader consumes the selection as aset; we marginalize or-
der for thedecisionand serialize the chosen set for packing by its
highest-probabilityexpandedordering — order enters only there.
The probe makes the order-marginalized posteriorP(S|x) =P
σ∈Perm(S)Q
tp(σt|x, σ <t)computableexactly, and its MAP
certifiable without full expansion: any unexpanded prefixρbounds
all completions byP(ρ), givingL(S)≤P(S)≤U(S); expanding
ordered pairs by mass until
L(S⋆)>max
S̸=S⋆U(S)(4)
certifiesS⋆as the exact MAP, else the remaining pairs are expanded
in one further probe — the readout is always exact, never truncated.
(Set-MAP adds one probe pass — a second only when the certifi-
cate defers; split E:0.44s/question vs.0.37compiled,0.63serial.)
Empirically the certificate fires after the first24of up to380pairs
in every non-trivial round (frontiers withk≥4; smaller frontiers
admit a single set) across four splits (527/527).
Finally, letP Ube the Plackett–Luce policy induced by root log-
its with removal renormalization. Thesequential interaction mass
Iseq(x) = KL(P AR∥PU), computable in closed form from the
probe, bounds the expected cost of the unary shortcut foranyutility
u∈[0,1]:|E PARu−E PUu| ≤p
Iseq/2(Pinsker) — a state-
ment about the two stochastic policies; greedy agreement is mea-
sured directly in Sec. 4.1. This is apost-compilationfidelity diag-
nostic: smallI seqguarantees the unary shortcut loses little expectedTable 1. Policy fidelity and control cost. “Agree”: exact action-
sequence identity with the serial controller over the full traversal
(divergences are floating-point near-ties, Sec. 4.1). Latency: mean
(median) s/question. Out tok.: mean output tokens per question, se-
rial / compiled.
Split Agree Serial lat. Compiled lat. Out tok.
A (90q) 80/90 0.62 (0.50) 0.40 (0.33) 11.7 / 0
B (120q) 101/120 0.52 (0.46) 0.32 (0.28) 10.4 / 0
D (90q) 79/90 0.56 (0.48) 0.36 (0.32) 11.2 / 0
E (60q) 52/60 0.63 (0.50) 0.37 (0.30) 12.9 / 0
D, 3B 78/90 0.53 (0.44) 0.28 (0.24) 11.2 / 0
utility; predicting itbeforecompilation is future work. Measured
Iseqis heavy-tailed (median2.6, p908.3nats over the139fully ex-
panded rounds of split B). Only14%of rounds fall belowI seq=0.5
(expected-utility gap at most0.5): history dependence is pervasive,
so the unary shortcut is rarely provably benign — exactly where pre-
serving the autoregressive policy cheaply matters.
4. EXPERIMENTS
Setup.QASPER [23] with byte-verified evidence spans. The met-
ric is AllEvidence@B: all spans of an annotated evidence group
≥80%covered in the budget-Bpacked context; error rows count
zero. Five splits, labeled A–E in collection order, are all accounted
for below. Prompts and interfaces were frozen on development split
A (30 docs/90 questions); B (120 questions) is the development-
confirmatory batch of the companion interface study; C/D are
document-disjoint splits from the official test split (30 docs/90
questions each; C was consumed by the companion study’s policy
analysis, so only D and E remain fresh); E (20 docs/60 questions)
is preregistered: frozenafterthe compiler design, identity hashes
recorded before its gold was read. Qwen2.5-7B/3B-Instruct [24],
bf16, eager attention, one MetaX C500, offline; latency is per-
question wall-clock over full traversals, with identical configuration
across interfaces and per-run code-hash manifests; document-cluster
bootstrap CIs (10k resamples, seed 20260916). Baselines: serial
grammar decoding, token-by-token as deployed grammar engines
do (the compiled controller’s source policy); an idealizedchunked
serial engine that fast-forwards forced spans (one forward per slot);
unary projection; sequential-label decoding; exact-bfree genera-
tion; legacy capped JSON; and embedding-MMR diversity [25]. A
byte-preserving false-nesting corruption stresses the hierarchy prior,
and the four-arm runs and benchmark are repeated on NVIDIA
H800s with the same runner, prompts, weights, and eager-attention
configuration (one run per device).
4.1. Compilation reproduces the serial policy
Compiled control reproduces the serial controller’s exact action se-
quences (Table 1) on84–89%of questions per split (87%at 3B) with
zero output tokens, cutting mean control latency by35–41%(47%
at 3B). Selectedsetsagree on87–90%: some divergences merely
permute a selection. Agreement is higher per round (86–92%) than
per question: one flipped round redirects the remaining traversal. A
same-run four-arm comparison on D and E (one code hash, arms in-
terleaved) adds the chunked engine; compiled control beats this fair
baseline too: serial/chunked/compiled run at0.58/0.37/0.35s per
question on D and0.64/0.40/0.37on E.

1.4k 2.1k 8.2k
total prompt tokens (k=20, 7B)2.55.0selection
latency (s)
2.8×
serial
compiledchunked serial
compiled, b=4Fig. 2. Work–depth scaling at the deployed frontier (k=20, 7B,
synthetic): serial grammar decoding, the idealized chunked engine
(D=n; it overlaps compiled here, while Sec. 4.1 separates them on
real workloads), and compiled control (D=2); the dottedb=4probe
shows the measured break-even of Eq. (3). The 3B model repro-
duces the pattern.
Both cluster-bootstrap CIs on the compiled−chunked delta ex-
clude zero; a reversed arm order on E reproduces the gap (−0.028s).
The divergences themselves are precision, not policy. Instru-
mented replays on identical prefixes show per-slot logit deviations
between the graphs averaging1.2(p952.6) against serial decision
margins of median4.1, and on the margin-instrumented replays
(B/D/E) all39diverging rounds occur at slots with margin≤2.5
— inside the2εflip window of Corollary 1 — never at a confi-
dently separated decision. A full fp32 replay of split A collapses
the deviations by over20×(mean0.05, p950.12), and under bf16
the compiled-vs-serial deviation on a slot-level diagnostic (1.25–2.5
across slots) is of the same order as re-chunking the same serial
prompt (2.0).
Nor are near-tie flips specific to compilation. Chunking alone
changes final selections at least as often (13vs.11of90on D;7
vs.7of60on E), and an NVIDIA H800 repetition preserves the la-
tency ordering (0.28/0.16/0.15s on D;0.31/0.17/0.15on E) while
final-selection agreement drops to65–77%— the per-token engine
agrees withitselfacross the two accelerators on just67–73%of
questions. The realized bf16 policy is accelerator- and backend-
dependent: compiled-versus-serial disagreement is comparable to
the serial policy’s cross-accelerator disagreement.
4.2. Quality tracks the source policy
Whether history-dependent selection beats unary marginals is
distribution-dependent: the AR policy led unary by+5.9/+4.4
points at 2K on splits B/C yet trailed by1.1on D (CI[−8.9,+6.7]);
embedding-MMR leaves84/90selections unchanged. Compilation
makes this uncertainty cheap to hedge: both policies come from the
same probe, andI seq(Sec. 3.4) flags when they can differ. Paired
deltas quantify the tracking error: pooled over four splits and three
budgets, compiled minus serial is−0.6points, CI[−1.7,+0.5]
(1080cells); on the two fresh splits alone,−2.4(CI[−4.9,−0.4])
— a deficit consistent with bf16 near-tie resolution: fp32 on A re-
stores89/90identical sequences. Corollary 1 supplies aparity dial:
rounds whose minimum margin falls belowθare replayed serially
— serial decisions by construction. Atθ=2, set post hoc above every
observed flip margin, the hybrid’s final selections match the serial
engine’s on all150/150D+E questions (60%of rounds gated) —
but at0.70–0.71s it is slower than serial itself, and on H800 three
flips escape the gate (147/150):θ, likeε, is hardware-specific. Ex-
act parity via the dial is not cheap: compiled trades it for41%lowerlatency, and parity-critical deployments keep the serial engine. Un-
der false-nesting corruption compiled control keeps a zero protocol
surface (44.4AllEvidence@1K,0errors); free generation fails on
7.8–14.4%of questions.
4.3. Ablations
On development split A, reading only the controller’sfirstlabel slot
(top-bof slot-1 marginals) collapses mid-budget quality:42.2vs.
51.1AllEvidence@1K for the full compiled chain — later, history-
conditioned slots carry information no single readout recovers. Con-
ditionals aregrammar-specific: a different serialization of the same
task (sequential labels, no JSON syntax) yields a visibly different
policy (44.4vs.50.0@1K) — so we compile the deployed gram-
mar, not a surrogate prompt. Repeat runs of every interface are bit-
identical on fixed hardware; Table 1’s variance comes from the exe-
cution graph or precision, never the controller.
4.4. Work–depth scaling
Synthetic sweeps (k∈ {4, . . . ,20},n∈ {2,3,4}, prompts1.4k–
8.2k tokens, both scales) isolate execution cost (Fig. 2). At the de-
ployed frontier widthk=20, compiled selection is1.3–2.8×faster
than serial decoding forn≤3, and on the MetaX the advantage
growswith prompt length (2.8×at 8k tokens;≥2.2×at both scales)
because every serial step re-reads the long KV cache; on the H800,
whose decode kernels blunt that penalty, gains sit at1.4–2.6×, peak-
ing at short prompts. The idealized chunked engine (D=n) removes
most of that per-token penalty (2.09s at 8k,n=3, vs.5.85serial);
compiled control matches it there (2.09s) and beats it by6–7%on
the real workloads of Sec. 4.1, where the remaining dependent for-
wards still bind. Atn=4the362-row probe loses at short prompts
(0.4–0.6×vs. token-wise serial), recovers only1.4×at 8k (1.1×at
3B), and never catches the chunked engine (4.34vs.2.15s; the wall
persists on the H800 at0.5–0.9×) — the width wall that Eq. (2)
predicts; past it, staged or chunked execution plans win.
Compilation spends parallel work to shorten the critical path,
not to reduce total computation: atk=20,n=3the probe evalu-
ates117parallel token positions where serial decoding emits9se-
quential ones (13×the work, off the critical path), and atn=4it
grows to362rows and3195positions (about270×) — exactly
why the break-even recedes. An fp32 probe (89/90parity) runs at
0.55s/question — no more than bf16serialdecoding (0.56s); fp32
serial pays1.21s; probe memory overhead is≤5MB atn≤3and
≈3.7GB atn=4.
5. CONCLUSION
A bounded-horizon constrained LLM controller does not need to be
decoded; it can be compiled. Its reachable future is a shallow pre-
fix tree that one tree-causal probe evaluates exactly, reproducing the
serial policy in exact arithmetic with a depth-2 critical path and zero
output tokens; certified set inference and an interaction-mass bound
follow. Limits are explicit: probe width grows factorially beyond the
greedy horizon, and bf16 leaves near-tie divergences that fp32 nar-
rows to89/90. The probe is an ordinary attention call, composable
with tree-attention IO optimizations [18]; the guarantees transfer to
any finite-action agent loop, and the compiler and benchmark har-
ness will be released. The controller’s semantics are now a model-
ing choice decoupled from the serial critical path — for hierarchical
RAG and every finite-action controller an LLM is asked to be.

6. COMPLIANCE WITH ETHICAL STANDARDS
This study uses only the publicly available QASPER dataset of sci-
entific papers and involves no human subjects, animal experiments,
or personally identifiable data; no ethical approval was required.
7. FUNDING AND CONFLICTS OF INTEREST
This work was supported by the Shanghai Special Fund for Pro-
moting High-Quality Industrial Development: Pilot Industry Innova-
tion and Development Program (Artificial Intelligence Track, Grant
No. 2025-GZL-RGZN-02006). The authors declare no conflicts of
interest.
8. REFERENCES
[1] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna,
Anna Goldie, and Christopher D. Manning, “RAPTOR: Re-
cursive abstractive processing for tree-organized retrieval,” in
Proc. ICLR, 2024.
[2] Howard Chen, Ramakanth Pasunuru, Jason Weston, and Asli
Celikyilmaz, “Walking down the memory maze: Beyond
context limit through interactive reading,”arXiv preprint
arXiv:2310.05029, 2023.
[3] Mingtian Zhang, Yu Tang, and PageIndex Team, “PageIn-
dex: Next-generation vectorless, reasoning-based RAG,”
https://pageindex.ai/blog/pageindex-intro,
2025, Accessed 2026-08-29.
[4] Nilesh Gupta, Wei-Cheng Chang, Ngot Bui, Cho-Jui Hsieh,
and Inderjit S. Dhillon, “LLM-guided hierarchical search
for end-to-end reasoning intensive retrieval,”arXiv preprint
arXiv:2510.13217, 2025.
[5] Torsten Scholak, Nathan Schucher, and Dzmitry Bah-
danau, “PICARD: Parsing incrementally for constrained auto-
regressive decoding from language models,” inProc. EMNLP,
2021, pp. 9895–9901.
[6] Saibo Geng, Martin Josifoski, Maxime Peyrard, and Robert
West, “Grammar-constrained decoding for structured NLP
tasks without finetuning,” inProc. EMNLP, 2023, pp. 10932–
10952.
[7] Lingnan Xu, Chong Feng, Kaiyuan Zhang, Liu Zhengyong,
Wenqiang Xu, and Fanqing Meng, “Equipping retrieval-
augmented large language models with document structure
awareness,” inFindings of EMNLP, 2025, pp. 24608–24631.
[8] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni,
Vladimir Karpukhin, Naman Goyal, Heinrich K ¨uttler, Mike
Lewis, Wen-tau Yih, Tim Rockt ¨aschel, Sebastian Riedel,
and Douwe Kiela, “Retrieval-augmented generation for
knowledge-intensive NLP tasks,” inProc. NeurIPS, 2020.
[9] Junyuan Zhang, Qintong Zhang, Bin Wang, Linke Ouyang,
Zichen Wen, Ying Li, Ka-Ho Chow, Conghui He, and Wentao
Zhang, “OCR hinders RAG: Evaluating the cascading impact
of OCR on retrieval-augmented generation,” inProc. ICCV,
2025, pp. 17443–17453.
[10] Joongmin Shin, Chanjun Park, Jeongbae Park, Jaehyung Seo,
and Heuiseok Lim, “MultiDocFusion: Hierarchical and multi-
modal chunking pipeline for enhanced RAG on long industrial
documents,” inProc. EMNLP, 2025, pp. 20985–21004.[11] Brandon T. Willard and R ´emi Louf, “Efficient guided
generation for large language models,”arXiv preprint
arXiv:2307.09702, 2023.
[12] Saibo Geng, Hudson Cooper, Michał Moskal, Samuel Jenkins,
Julian Berman, Nathan Ranchin, Robert West, Eric Horvitz,
and Harsha Nori, “JSONSchemaBench: A rigorous bench-
mark of structured outputs for language models,”arXiv
preprint arXiv:2501.10868, 2025.
[13] Mingyu Derek Ma, Yanna Ding, Zijie Huang, Jianxi Gao,
Yizhou Sun, and Wei Wang, “Inferring from logits: Explor-
ing best practices for decoding-free generative candidate se-
lection,” inProc. ACL, 2025, pp. 33125–33144.
[14] Yuval Reif and Roy Schwartz, “Beyond performance: Quan-
tifying and mitigating label bias in LLMs,” inProc. NAACL,
2024, pp. 6784–6798.
[15] Pouya Pezeshkpour and Estevam Hruschka, “Large language
models sensitivity to the order of options in multiple-choice
questions,” inFindings of NAACL, 2024, pp. 2006–2017.
[16] Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng,
Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan
Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen,
Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia, “SpecIn-
fer: Accelerating large language model serving with tree-based
speculative inference and verification,” inProc. ASPLOS,
2024, pp. 932–949.
[17] Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Ja-
son D. Lee, Deming Chen, and Tri Dao, “Medusa: Simple
LLM inference acceleration framework with multiple decod-
ing heads,” inProc. ICML, 2024, pp. 5209–5235.
[18] Jinwei Yao, Kaiqi Chen, Kexun Zhang, Jiaxuan You, Binhang
Yuan, Zeke Wang, and Tao Lin, “DeFT: Decoding with flash
tree-attention for efficient tree-structured LLM inference,” in
Proc. ICLR, 2025.
[19] Noam Mizrachi, Nadav Har-Tuv, and Shai Shalev-Shwartz,
“Contextual Plackett-Luce: An efficient neural model for prob-
abilistic sequence selection under ambiguity,”arXiv preprint
arXiv:2605.09112, 2026.
[20] Tamim Zoabi, Ameen Ali, Liran Ringel, and Lior Wolf,
“Mean-field parallel decoding for discrete diffusion language
models,”arXiv preprint arXiv:2606.15805, 2026.
[21] Xiaoxin Shi, Jiaxin Wan, Linkang Dong, Wei Jiang, Yue Liu,
and Zengfeng Huang, “SimpleTool: Parallel decoding for real-
time LLM function calling,”arXiv preprint arXiv:2603.00030,
2026.
[22] Sehoon Kim, Suhong Moon, Ryan Tabrizi, Nicholas Lee,
Michael W. Mahoney, Kurt Keutzer, and Amir Gholami, “An
LLM compiler for parallel function calling,” inProc. ICML,
2024, pp. 24370–24391.
[23] Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A.
Smith, and Matt Gardner, “A dataset of information-seeking
questions and answers anchored in research papers,” inProc.
NAACL, 2021, pp. 4599–4610.
[24] Qwen Team, “Qwen2.5 technical report,”arXiv preprint
arXiv:2412.15115, 2024.
[25] Jaime Carbonell and Jade Goldstein, “The use of MMR,
diversity-based reranking for reordering documents and pro-
ducing summaries,” inProc. SIGIR, 1998, pp. 335–336.
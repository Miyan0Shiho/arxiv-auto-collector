# Copying Is Not Grounding: Training Faithful Long-Form RAG with Authored-Sentence Credit

**Authors**: Hoang Pham, Duc Anh Vu, Duy Tung Doan, Cong-Duy T Nguyen, Anh Tuan Luu

**Published**: 2026-10-13 14:16:38

**PDF URL**: [https://arxiv.org/pdf/2610.18442v1](https://arxiv.org/pdf/2610.18442v1)

## Abstract
In search engines and assistants, long-form RAG answers can mislead readers with claims their sources do not support. Faithfulness rewards, the share of supported sentences, train models toward fewer such claims. We show that these rewards are hacked by copying. A sentence copied from the passages is supported almost by construction, so trained models learn to copy rather than ground what they write. Common defenses only move the hack: a KL or copy penalty leaves copying in place or trades it for shorter, emptier answers and refusals. GRPO, the standard RL algorithm, makes it worse by giving every sentence its answer's advantage, reinforcing copied and grounded sentences alike. We propose Authored-Sentence Credit (ASC) , which fixes both by moving GRPO's reward and advantage from the answer to the sentence level. The reward credits only sentences that are authored rather than copied, say more than the question, and are supported by the passages. The advantage compares each sentence with every sentence sampled for the question, so one that earns no credit ranks last and is never reinforced. Across four models and two in-domain, three held-out corpora, only ASC lowers the error rate of authored sentences and writes more supported ones in every setting, by 38% and 46% at the median under held-out evaluators, giving readers more grounded information. It takes none of these shortcuts, and two LLM judges comparing whole answers prefer no baseline to the base model but rate ASC's answers on par with it. We conclude that a faithfulness reward should credit what a model writes, not what it copies.

## Full Text


<!-- PDF content starts -->

Copying Is Not Grounding: Training Faithful Long-Form RAG
with Authored-Sentence Credit
Hoang Pham1Duc Anh Vu1Duy Tung Doan2
Cong-Duy Nguyen3Anh Tuan Luu1,3*
1Nanyang Technological University2Viettel AI, Viettel Group3VinUniversity
Email: pham0100@ntu.edu.sg
Abstract
In search engines and assistants, long-form
RAG answers can mislead readers with claims
their sources do not support. Faithfulness re-
wards, the share of supported sentences, train
models toward fewer such claims. We show
that these rewards are hacked by copying. A
sentence copied from the passages is supported
almost by construction, so trained models learn
to copy rather than ground what they write.
Common defenses only move the hack: a KL or
copy penalty leaves copying in place or trades
it for shorter, emptier answers and refusals.
GRPO, the standard RL algorithm, makes it
worse by giving every sentence its answer’s ad-
vantage, reinforcing copied and grounded sen-
tences alike. We propose Authored-Sentence
Credit ( ASC )1, which fixes both by moving
GRPO’s reward and advantage from the answer
to the sentence level. The reward credits only
sentences that are authored rather than copied,
say more than the question, and are supported
by the passages. The advantage compares each
sentence with every sentence sampled for the
question, so one that earns no credit ranks last
and is never reinforced. Across four models
and two in-domain, three held-out corpora, only
ASC lowers the error rate of authored sentences
and writes more supported ones in every set-
ting, by 38% and 46% at the median under held-
out evaluators, giving readers more grounded
information. It takes none of these shortcuts,
and two LLM judges comparing whole answers
prefer no baseline to the base model but rate
ASC ’s answers on par with it. We conclude
that a faithfulness reward should credit what a
model writes, not what it copies.
1 Introduction
Retrieval-augmented generation (RAG; Lewis
et al., 2020) now answers open questions in search
*Corresponding author.
1Our code is available at https://anonymous.
4open.science/r/asc14.
Qwen Llama01020unsupported (%)15
43.7× 18
63.2×(a) Error rate
authored copied
20 40 60 80
copy share (%)012supported own
sentences / answercopies less,
writes more
base
GRPOGRPO + KLASC(b) Trained policiesFigure 1:A faithfulness reward can be hacked by
copying(ASQA).(a)Under the reward’s checker, the
sentences a model writes itself are unsupported 3.2 to
3.7 times as often as those it copies.(b)Qwen3.5-9B
after training: GRPO copies and nearly stops writing,
GRPO + KL writes less, and ASC copies less and writes
more supported sentences of its own (held-out checker).
engines and assistants, yet its long-form answers
often state claims their sources do not support, in
web search (Liu et al., 2023) as in law and medicine
(Magesh et al., 2025; Wu et al., 2025). Faithfulness,
the share of an answer’s sentences a checker finds
supported by the passages (Gao et al., 2023b; Min
et al., 2023; Tang et al., 2024), is now a training sig-
nal against such claims, as a reward for reinforce-
ment learning or to rank preference pairs (Huang
et al., 2024; Chen et al., 2026; Tian et al., 2024).
The reward has a cheaper way up. A sentence
copied from the passages is supported almost by
construction: on ASQA, the sentences Qwen3.5-
9B writes itself are unsupported 3.7 times as often
as those it copies, and those of Llama-3.1-8B 3.2
times (Figure 1a). The reward therefore rises with
every sentence a model copies instead of writing.
Trained policies exploit this gap, a form of re-
ward hacking (Skalse et al., 2022). GRPO (Shao
et al., 2024) raises the copy share of Qwen3.5-9B
from 38.8 to 72.8 (Figure 1b) and shrinks its an-
swers from 74 to 14 words: the reward climbs
arXiv:2610.18442v1  [cs.CL]  13 Oct 2026

because the model stops writing, not because it
grounds better. Common defenses only move the
hack between threeshortcuts(copying, writing less
and writing emptier sentences): on the same model,
a KL penalty holds copying at the base level, but re-
fusals more than double and answers grow shorter
and emptier, and a copy penalty makes it decline
most questions. GRPO itself makes it worse: it
gives every token of an answer the same advantage,
its update weight, so a copied or empty sentence in
a high-reward answer is reinforced and a grounded
one in a low-reward answer pushed down.
We propose Authored-Sentence Credit ( ASC ),
to our knowledge the first faithfulness reward that
pays only for what the model writes itself (Fig-
ure 4). It changes GRPO in two places and adds no
model. Its reward credits a sentence only if it is not
copied, says more than the question (notvacuous),
and is supported by the passages. Its advantage
scores each sentence against all sentences sampled
for the question, so each sentence is judged by its
own credit, not its answer’s. Together, they guaran-
tee that a copied, vacuous or unsupported sentence
ranks last in its group and is never reinforced.
We train four models from two families on two
corpora and test them on five, three held out. Every
baseline, even a sentence-level reward, lowers the
error rate of authored sentences at the median, yet
takes a shortcut in nearly every setting. ASC alone
lowers it while writing more, with no shortcut in
any setting: at the median, it cuts that error rate by
38% under an independent judge and writes 46%
more supported sentences of its own, giving readers
more grounded information; human annotators con-
firm the drop without the judge. A higher reward
alone does not give better answers: every baseline
raises the reward, yet two LLM judges comparing
whole answers prefer none to the base model but
rate ASC on par with it and above every baseline.
In summary, our contributions are as follows.
•Analysis.We show that faithfulness rewards
pay for copying, that KL and copy penalties
only move the hack to other shortcuts, and
that GRPO’s answer-level advantage barely tells
copied from grounded sentences.
•Method.We propose ASC , to our knowledge the
first faithfulness reward that credits only authored
sentences, and prove that no uncredited sentence
is ever reinforced.
•Results.Only ASC lowers the authored error
rate and writes more supported sentences in allsettings, with no shortcut and no quality loss.
2 Related Work
Faithfulness metrics favor copying.Long-form
answers are scored statement by statement against
their sources (Gao et al., 2023b; Min et al., 2023),
and small checkers make this scoring cheap enough
to serve as a reward (Tang et al., 2024; Zha et al.,
2023). In summarization, such metrics favor extrac-
tive output, so faithfulness gains often come from
copying (Ladhak et al., 2022; Dreyer et al., 2023),
and an entailment reward can trade informativeness
for consistency (Roit et al., 2023). We study this
bias as a reward hack in long-form RAG, where
copying has a cost for readers: in generative search,
more precise citations come with more copying
from the sources and lower perceived utility (Liu
et al., 2023). Trivially true claims also inflate fac-
tuality scores (Jiang et al., 2025); our vacuity test
removes them by checking each sentence against
the question alone, as context-contrast methods do
(Shi et al., 2024; Cohen-Wang et al., 2024).
Grounded generation and reward hacking.
Faithfulness and factuality objectives train mod-
els through preference optimization (Tian et al.,
2024; Song et al., 2025), reinforcement learning
(Si et al., 2025; Huang et al., 2024), self-reflection
(Asai et al., 2024) and citation tuning (Ye et al.,
2024), and one method even trains RAG models to
copy more so that they hallucinate less (Long et al.,
2026). Optimizing such a proxy invites reward
hacking (Skalse et al., 2022; Gao et al., 2023a),
best studied for length (Singhal et al., 2024); for
factuality, precision-only metrics and rewards favor
short, less detailed answers (Wei et al., 2024; Chen
et al., 2026). A KL penalty does not prevent hack-
ing under heavy-tailed reward error (Kwa et al.,
2024), and optimization amplifies a reward’s biases
(Shapira et al., 2026; Moya et al., 2026). We show
that faithfulness rewards are hacked by copying,
and that common defenses only move the hack.
Fine-grained credit assignment.Dense signals
score each sentence (Wu et al., 2023) or reasoning
step (Lightman et al., 2024). GRPO normalizes
rewards within a group of answers (Shao et al.,
2024), and GiGPO within groups of steps (Feng
et al., 2025). The closest rewards score facts or
sentences for closed-book correctness (Wen et al.,
2025; Gu et al., 2025) or for support in RAG
(Huang et al., 2024), but none asks whether a
sentence was copied. ASC withholds credit from

copied and vacuous sentences and compares sen-
tences, not answers, so these are never reinforced.
3 Problem Formulation and Setup
Task and reward.A prompt consists of a ques-
tionqand retrieved passages P, and a policy πθ
writes a long-form answer y= (y 1, . . . , y n)ofn
sentences. A checker g(e.g., Bespoke-MiniCheck-
7B; Tang et al., 2024; Bespoke Labs, 2024) returns
g(ys, P)∈[0,1] , its probability that Psupports
ys, and the sentence counts as supported when
g(ys, P)≥κ . The faithfulness reward that training
maximizes and evaluations report is the share of
supported sentences,
R(y) =1
nPn
s=11[g(y s, P)≥κ].(1)
Training.GRPO (Shao et al., 2024) samples G
answers to each prompt and gives every token of an-
swerithe same advantage, its reward standardized
over the group, A(i)= (R(y(i))−µ)/(σ+ϵ) with
µ,σthe group’s mean and standard deviation; DPO
(Rafailov et al., 2023) instead learns offline from
pairs of answers, preferring the one with higher R.
Copied, authored and vacuous sentences.The
copy share c(ys)of a sentence is the fraction of
its words inside verbatim matches of at least m
words with P, and c(y) is the same fraction over
an answer. A sentence iscopiedif c(ys)≥τ and
authoredotherwise; ρ(y) is the share of copied
sentences in an answer. A sentence isvacuousif
it passes the same support test with the question
in place of the passages, g(ys, q)≥κ : it needs no
evidence to count as supported, like “The song
was released.” for a question about a song. Only
authored, non-vacuous sentences show whether a
model can support what it writes itself; we call one
groundedwhenPsupports it.
Goal and shortcuts.Training should make a
model’s own sentences more reliable: it should
lower theauthored error rate ea(π), the share of
authored sentences that the passages do not support,
which is 2–40% for the base model π0. It should do
so without writing less: the copy share c, vacuous
share vand refusal rate d(the share of answers that
decline) should not rise above those of π0, nor the
answer length ℓfall below it. Rdoes not encode
this goal and can be hacked through threeshortcuts,
copying more, writing less and writing emptier sen-
tences, with refusal as the extreme; each raises R,
and writing less can even lower eaby droppingrisky sentences, with no authored sentence becom-
ing more reliable. Section 4 shows that training on
Rtakes these shortcuts, and Section 5 introduces a
signal that improves onπ 0without them.
Metrics.Two lead metrics measure the goal; the
others detect shortcuts and check quality (Table 6).
•Reliability (Err). ea, judged with the passages
by Gemma-4-31B (Gemma Team, 2026), which
comes from a different family than the reward’s
checker and is never used in training.
•Content (Supp.).Supported authored sentences
per answer under the held-out MiniCheck-Flan-
T5-Large (Tang et al., 2024); unlike Err, it falls
when a model simply writes less.
•Shortcuts (Copy, Vac., Ref., Words). c,v,d
andℓ; a policytakes a shortcutwhen c,vor
dexceeds that of π0by a set margin, or ℓfalls
below a set fraction of it (values in Section 6.1).
•Checks.AlignScore (Zha et al., 2023) as a sec-
ond held-out checker, ASQA exact match (EM;
Gao et al., 2023b), and two LLM judges of whole
answers (Section 6.2).
Models and data.We train four instruction mod-
els from two families, Qwen3.5-9B, Qwen3.5-4B
(Qwen Team, 2026), Llama-3.1-8B and Llama-3.2-
3B (Grattafiori et al., 2024), and evaluate each on
five corpora, 20settingsof model and corpus in
all. Training pools ASQA (Stelmakh et al., 2022)
with ALCE’s top five passages (Gao et al., 2023b)
and the question-answering part of RAGTruth (Niu
et al., 2024); we evaluate in domain on their evalu-
ation sets. Three further corpora, ClapNQ (Rosen-
thal et al., 2025), ELI5 (Fan et al., 2019) with
ALCE’s passages and ExpertQA (Malaviya et al.,
2024), are held out entirely, so a gain on them can-
not come from fitting the training data. On ELI5
and ExpertQA, where base models err most and
copy least, a good answer has to be written, not
assembled from the passages.
4Why Faithfulness Rewards Are Hacked
We ask what Rpays for in base-model answers,
what a policy trained on it learns, and what GRPO’s
advantage sees inside an answer; each finding
yields one requirement that Section 5 meets.
Finding 1: copied sentences are rarely unsup-
ported, so the reward pays for copying.We
sample eight answers per question from each base
model on ASQA, RAGTruth, ClapNQ and ELI5.
Under the reward’s checker, authored sentences are

ASQA RAGTruth ClapNQ ELI50102030unsupported (%)3.4×
2.3×3.3×4.4×(a) Reward checker
authored
copied
T5 Align Human2.7×
2.1×6.4×(b) Held-out checksFigure 2: Unsupported rate of authored and copied sen-
tences, with their ratio.(a)Base-model samples under
the reward’s checker (mean over models).(b)The same
samples under MiniCheck-Flan-T5 (T5) and AlignScore
(Align), mean over models and corpora, and human la-
bels on RAGTruth’s responses.
unsupported 2.3 to 4.4 times as often as copied ones
on average (Figure 2a), and more often for every
model and corpus. The gap is not an artifact of the
checker: two held-out checkers find it almost ev-
erywhere, and so do human labels on RAGTruth’s
responses, 12.2% against 1.9% (Figure 2b). With
scandsathe support rates of copied and authored
sentences, the reward of Eq. 1 decomposes as
R=s a+ρ(s c−sa),(2)
a line in ρwith slope sc−sa>0: at fixed authored
reliability, every sentence copied instead of written
raises R(proof in Appendix A.1). Sampled an-
swers agree: among answers to the same question,
those that copy more earn more reward for every
model and corpus (slopes in Appendix C.1). Yet a
copied sentence inherits its source’s reliability, so
its credit says nothing of the model’s own ground-
ing. This gives the first requirement,(R1): a copied
sentence earns nothing, however well supported.
Training Reward Copy Words Vac. Ref. EM
Base model 87.1 38.8 73.5 21.5 6.1 45.5
GRPO 98.972.8 13.718.1 0.0 27.6
GRPO + KL 92.3 39.260.0 28.5 14.741.8
Copy penalty 92.0 11.836.9 67.4 64.520.2
DPO 94.653.4 55.716.9 1.7 41.8
DPO, copy-matched 90.444.9 59.721.5 6.1 43.8
Table 1: Qwen3.5-9B trained to raise R, on ASQA; met-
rics of Section 3, rates in %.Bold: a shortcut. Llama-
3.1-8B and training curves: Appendix C.2.
Finding 2: training takes the shortcuts, and com-
mon defenses only move between them.We train
Qwen3.5-9B to raise Rwith GRPO, alone or with
one of two defenses, a KL penalty (Ziegler et al.,
0 .5 1020406080copy share (%)
(a) Copy share
0 .5 1
training progress020406080words per answer
(b) Length
0 .5 1020406080refusals (%)
(c) RefusalsGRPO GRPO + KL Copy penaltyFigure 3: Qwen3.5-9B on ASQA over training (0: base
model): GRPO copies and nearly stops writing, and
each defense swaps one shortcut for another.
2020; Ouyang et al., 2022) or a copy penalty, and
with DPO (Table 1). Plain GRPO nearly saturates
the reward (98.9) through the cheapest shortcuts,
and the answers get worse: copy share nearly dou-
bles within half of training, answers shrink from 74
to 14 words (Figure 3), and exact match falls from
45.5 to 27.6. Each defense closes one shortcut and
opens another: the KL penalty holds copying at
the base level, but answers grow shorter and emp-
tier and refusals rise, and the copy penalty cuts
copying, but its vacuous share rises to 67.4% and it
declines 64.5% of questions. DPO copies too, even
on pairs whose answers copy nearly equally: prefer-
ring better-supported answers alone pulls it toward
the passages. The two penalties take a shortcut on
nine in ten model–corpus pairs (grey in Table 2). R
pays for any sentence the checker cannot reject, so
penalizing one quantity only reroutes the optimizer.
This gives the second requirement,(R2): a vacuous
sentence earns nothing, and dropping sentences or
refusing never pays.
Finding 3: GRPO’s advantage barely tells a
copied sentence from a grounded one.GRPO’s
advantage belongs to the whole answer, so a sen-
tence is rewarded for the answer it sits in: a copied
or vacuous sentence in a high-reward answer is
pushed up, and a grounded one in a low-reward
answer is pushed down. Copying therefore barely
changes a sentence’s chance of being reinforced:
on Finding 1’s samples, of the sentences GRPO
moves, it pushes up 72.0% of the copied ones and
75.5% of the grounded ones, and even 58.8% of
the vacuous ones; grounded ones are pushed up
more often everywhere, but by at most 17 points
(Appendix C.3). An answer-level advantage thus
cannot steer the policy away from copying: it rein-

1) Sample a group
𝐺answers per prompt
𝑞:How do the genetics
of three-parent
children work?
𝑃:/file-alt/file-alt/file-alt/file-alt/file-alt
♂robot𝜋𝜃
𝑦(𝑖)
𝑦(𝑖)∼𝜋𝜃old(·|𝑞,𝑃)2) Credit each sentence
checker𝑔reads𝑦 𝑠with𝑃, or𝑞alone
sentence𝑦𝑠of𝑦(𝑖)
♂quote-rightcopied from𝑃?
𝑐(𝑦𝑠)≥𝜏yes
♂quote-right𝑟𝑠=0
/commen◎supported by𝑞alone?
𝑔(𝑦𝑠,𝑞)≥𝜅yes
/commen◎𝑟𝑠=0
/timesunsupported by𝑃?
𝑔(𝑦𝑠,𝑃)<𝜅yes
/times𝑟𝑠=0no
no
no
✓𝑟𝑠=𝜙(︁𝑔(𝑦𝑠,𝑃))︁3) Sentence baseline
𝜇,𝜎over the whole group
𝑎(𝑖)
𝑠=𝑟(𝑖)
𝑠−𝜇
𝜎+𝜖
𝑎𝑠<0𝑎𝑠>0
♂quote-right♂quote-right♂quote-right♂quote-right♂quote-right♂quote-right
♂quote-right♂quote-right/times/times/commen◎
✓✓✓✓✓✓✓
✓✓✓✓✓𝜇
0 1
credit𝑟𝑠of each sentence4) Policy update
each token takes its sentence’s𝑎 𝑠
♂quote-right ✓ /commen◎ 𝑦(𝑖):
ASC
GRPO−1.0+1.0−1.0
+0.5ˆ𝐴𝑖,𝑡=𝑎(𝑖)
𝑠for𝑡∈𝑦(𝑖)
𝑠
ˆ𝐴𝑖,𝑡=𝐴(𝑖)for all𝑡
ASC objective
J(𝜃)=1
𝑁∑︁
𝑖,𝑡(︂𝜋𝜃
𝜋oldˆ𝐴𝑖,𝑡−𝛽D𝑖,𝑡)︂
sentence ♂quote-rightcopied /commen◎vacuous /timesunsupported ✓grounded wrong sign
𝑐: copy share of a sentence𝑔: the checker’s support probability𝜏,𝜅: thresholds𝜙(𝑝)=max{0,1−ln𝑝/ln𝜅}: graded creditFigure 4:One ASC step in GRPO.1) The policy samples Ganswers to a question qwith passages P. 2) A
sentence earns credit only if it is authored, not vacuous and supported by P. 3) Each credit is compared with those
of all sentences in the group, across answers. 4) Every token takes its sentence’s advantage; the rest of GRPO is
unchanged. Example (Qwen3.5-9B, ELI5): answer y(i)earns the full reward although one sentence is copied and
one vacuous; GRPO pushes all three up (+0.5), ASC only the grounded one (+1.0; the others−1.0).
forces whatever the high-reward answers contain.
This gives the third requirement,(R3): sentences
are compared with sentences, so that each advan-
tage reflects the sentence’s own reliability.
5 ASC: Authored-Sentence Credit
ASC meets (R1)–(R3) with two changes to GRPO,
which sentences earn credit (Section 5.1) and what
each credit is compared with (Section 5.2), and
leaves the rest unchanged (Section 5.3; Figure 4).
5.1 Which sentences earn credit
For a sampled answer ywith sentences y1, . . . , y n,
the credit ofy sis the product of three tests,
rs=1[c(y s)<τ]|{z}
copy1[g(y s, q)<κ]|{z}
vacuityϕ(g(y s, P))|{z}
support.(3)
The copy test meets (R1): a copied sentence owes
its support to repeating P, not to the policy’s
grounding, so the test gives it what an unsupported
sentence earns. The vacuity test meets the first
half of (R2): a sentence that the checker accepts
from the question alone needs no evidence, so its
support is no evidence of grounding either. The
support test credits the remaining sentences with
the graded credit
ϕ(p) = maxn
0,1−lnp
lnκo
.(4)
Properties of the credit. ϕis zero for p≤κ
and one at p= 1 , so the credited sentences are
exactly those above the checker’s threshold. Writ-
ingh(p) =−lnp for the checker’s surprisal at“supported”, ϕ(p) = 1−h(p)/h(κ) on(κ,1] :ϕis
the only credit that is affine in surprisal, zero at the
threshold and one at certainty, and it is strictly in-
creasing and concave in p, rising fastest just above
the threshold (Appendix A.2). The binary credit
1[p≥κ] pays a sentence just above the threshold
as much as one the checker is certain of; the graded
credit keeps the checker’s confidence and pulls the
policy toward sentences the checker is sure of. Ev-
ery credit lies in [0,1] , and a sentence that fails any
test earns the minimum,r s= 0.
5.2 A sentence baseline
Where GRPO standardizes answers (Section 3),
ASC standardizes sentences. For a group of G
answers with Msentences in total, let µandσbe
the mean and standard deviation of the credits of
allMsentences. Sentence sof answer ireceives
the advantage
a(i)
s=r(i)
s−µ
σ+ϵ, ϵ >0.(5)
Proposition 1 shows that the credit and the baseline
together meet (R1)–(R3): whatever the answers
contain, a sentence failing a test ranks strictly below
every sentence that earns credit.
Proposition 1(No shortcut earns advantage).For
any group and the advantage of Eq. 5:
(i)Credit alone sets the order: a(i)
s< a(j)
tif and
only if r(i)
s< r(j)
t, for sentences of the same
or of different answers (R3).
(ii)Shortcuts rank last: a copied, vacuous or
unsupported sentence receives the lowest ad-
vantage of its group, −µ/(σ+ϵ) , negative if

any sentence earns credit (R1, R2).
(iii) Average sentences earn nothing: advantages
sum to zero over the group, and a sentence
of average credit receives zero.
(iv)Length-free advantages: given the group’s µ
andσ, a sentence’s advantage depends on
neither its own length nor its answer’s (R2).
All four parts follow from two facts: Eq. 5 ap-
plies one strictly increasing affine map to every
credit of the group, and a sentence that fails a test
has the minimum credit. GRPO’s answer-level ad-
vantage breaks (i) and (ii), as Finding 3 measured,
and (iv) as well (proofs in Appendix A.3).
5.3 Training objective
Every token tof sentence sin answer itakes
that sentence’s advantage, ˆAi,t=a(i)
s, in place
of GRPO’s A(i); tokens outside every sentence,
the whitespace between sentences and the end-of-
sequence token, take zero. The objective is other-
wise GRPO’s (Algorithm 1, Appendix B):
J(θ) =1
NP
i,t
wi,t(θ)ˆAi,t−βD i,t
,(6)
where wi,t(θ) =π θ(y(i)
t|q, P, y(i)
<t)/πθold(y(i)
t|
q, P, y(i)
<t)is the importance ratio, Di,tGRPO’s per-
token KL estimate (Shao et al., 2024) toward the
reference πref, the base model, and Nthe number
of answer tokens in the batch (Yu et al., 2025).
Gradient form.Each batch is used for one update,
so the gradient is taken where wi,t= 1, and the
log-probabilities of a sentence’s tokens add up to
its log-likelihood. Up to the KL term, the gradient
of Eq. 6 is a policy gradient over sentences,
∇J=1
NP
i,sa(i)
s∇lnπ θ 
y(i)
s|y(i)
<s
,(7)
with∇=∇ θ,y(i)
<sthe text before sentence sand
the prompt (q, P) left implicit (Appendix A.4).
GRPO’s gradient weights the log-likelihood of the
whole answer by A(i), so every sentence moves
with its answer; ASC moves each sentence by its
own advantage, whatever the rest of its answer con-
tains, which is the formal content of (R3).
Relation to GRPO and to sentence rewards.If
every answer had a single sentence, Eq. 5 would
reduce to GRPO’s own standardization with ras
the reward, so ASC is GRPO with a finer unit of
comparison. Without the first two tests and with the
binary credit, ASC becomes a sentence-level sup-
port reward (Wu et al., 2023; Huang et al., 2024),the sentence-reward baseline of Section 6.1. The
only added checker cost is one question-only call
per authored sentence. ASC needs no critic, no
reference answers and no penalty weight.
6 Experiments
6.1 Setup
Training.All policies are trained with GRPO at
full weight on the pooled ASQA and RAGTruth
training questions, with G= 8 answers per ques-
tion, a KL coefficient β= 0.04 , one zero-shot
prompt and the same hyper-parameters for every
method (Appendix B). Every method uses the same
reward checker, Bespoke-MiniCheck-7B (Tang
et al., 2024; Bespoke Labs, 2024), and the same
thresholds, κ=τ=1
2andm= 5 words, so the
methods differ only in the training signal.
Baselines.We compare ASC with three baselines.
GRPO + KLtrains on the faithfulness reward Rof
Eq. 1, the standard recipe.Copy penaltytrains on
R−λ c(y) , the direct fix for the copy shortcut, with
λset to the support gap sc−saof the model’s own
samples on the training questions, so that by Eq. 2
a copied sentence earns no more than an authored
one.Sentence reward, a sentence-level support
reward in the style of fine-grained RLHF (Wu et al.,
2023), is ASC without its first two tests and graded
credit, so it keeps the sentence baseline.
Evaluation.We decode greedily and evaluate on
all five corpora with the metrics of Section 3. A
policy takes a shortcut on a corpus when its copy
share, vacuous share or refusal rate exceeds the
base model’s by more than 3 points, or its answers
fall below 85% of the base model’s words. Rows
average the training seeds where a method has sev-
eral ( ASC on the Qwen models, binary credit) and
are single runs otherwise; Appendix D gives per-
corpus results, every run and bootstrap significance
tests, and Appendix I example answers.
6.2 Main results
Only ASC lowers the error rate while writing
more.Every method lowers the authored error
rate at the median, ASC the most (38%, against 31
to 36%), but a model can also lower it by writing
less, so Err alone does not separate them (Table 2).
What separates them is how much grounded con-
tent the model still writes (Supp.): ASC writes the
most supported authored sentences on every model,
46% more than the base model at the median and
more in every setting, whereas GRPO + KL writes

MethodAuthored error rate (Err)↓Answer metrics (mean) ASQA
EMASQA RAGT. ClapNQ ELI5 Expert Mean Supp.↑Copy Words Vac. Ref.
Qwen3.5-9B
Base model 5.09 2.31 3.61 6.04 6.21 4.65 2.17 28.06 97.40 27.88 4.94 45.45
GRPO + KL 3.97 1.20 3.19 5.38 5.85 3.92 2.03 27.17 83.00 32.23 23.32 41.78
Copy penalty 1.45 1.20 0.40 1.13 0.42 0.92 1.22 6.18 29.48 82.31 79.65 20.22
Sentence reward 3.74 0.85 3.09 3.51 4.25 3.09 2.16 30.85 92.08 25.69 11.78 44.15
ASC 3.96 1.13 2.15 3.37 3.74 2.87 3.04 22.18 108.85 13.28 0.22 44.93
Qwen3.5-4B
Base model 8.57 2.87 5.84 8.84 8.72 6.97 2.16 29.84 103.77 26.31 5.33 46.24
GRPO + KL 3.17 2.03 2.94 3.94 3.49 3.11 1.59 22.67 59.41 53.77 43.87 36.23
Copy penalty 8.17 3.06 4.92 6.73 9.07 6.39 2.30 19.73 86.35 36.27 28.13 36.34
Sentence reward 4.282.224.17 5.91 5.91 4.50 2.42 32.85 108.07 21.98 4.40 46.42
ASC 6.47 2.25 4.00 6.58 7.69 5.40 2.97 23.98 110.77 12.89 0.95 45.67
Llama-3.1-8B
Base model 11.69 8.46 13.95 27.89 27.34 17.87 1.06 37.98 50.41 24.24 4.65 35.14
GRPO + KL 6.56 6.695.5821.05 22.48 12.47 1.10 41.26 49.41 25.50 4.03 34.50
Copy penalty 8.04 6.02 8.75 17.63 20.53 12.19 1.41 15.54 39.79 29.63 6.96 28.17
Sentence reward 4.77 5.06 5.53 18.18 19.96 10.70 1.04 42.58 49.76 23.03 4.00 34.25
ASC 6.03 3.94 7.23 17.38 19.89 10.89 1.57 24.77 55.13 13.71 0.68 35.61
Llama-3.2-3B
Base model 18.87 11.71 14.47 38.67 40.53 24.85 0.93 40.28 52.58 20.05 4.27 34.73
GRPO + KL 14.41 5.68 9.95 25.42 28.00 16.69 0.89 49.39 50.98 20.47 3.20 32.42
Copy penalty 10.73 8.99 9.94 26.76 27.06 16.70 1.23 18.86 37.82 28.29 7.53 25.79
Sentence reward 11.82 7.61 6.31 23.84 24.01 14.72 0.93 49.76 52.31 17.36 2.00 33.03
ASC 15.00 7.06 8.70 24.88 24.96 16.12 1.57 24.54 55.86 14.09 2.47 35.66
Table 2: Main results (0–100). Err: authored error rate on two in-domain (ASQA, RAGTruth) and three held-out
corpora, with its mean; answer metrics: means over the five corpora; EM: exact match on ASQA. Grey Err: a
shortcut on that corpus (Section 6.1), not ranked; grey mean: a shortcut on at least one corpus. Bold: lowest
remaining Err and most supported authored sentences per answer (Supp.); underlined: second-lowest Err.
0 5 10 15 20 25 30 35 40
authored error rate, base model (%)051015202530authored error rate, ASC (%)median: 38% lower error rateno change Qwen3.5-9B
Qwen3.5-4B
Llama-3.1-8B
Llama-3.2-3B
in domain
Figure 5: Authored error rate of the base model against
ASC , one point per model and corpus (hollow: in do-
main); all lie below the diagonal (dashed: median).
fewer than the base model on three of the four mod-
els. The lower rate is not a dilution by the extra
sentences: unsupported authored sentences per an-
swer fall too, in 16 of the 20 settings. Figure 5
shows that the cut holds at every error level, includ-
ing the hardest settings, where the base model errs
on about 40% of its sentences. It is even larger on
the held-out corpora, never seen in training, than in
domain (median 38% against 32%).
ASC takes no shortcut; the baselines take onealmost everywhere.In every setting ASC copies
less than the base model, writes fewer vacuous sen-
tences and refuses less often, so none of its Err
values is grey in Table 2, under any margin from
1 to 10 points. The baselines buy their lower error
rates with a shortcut, and closing one opens another,
as in Finding 2: GRPO + KL declines up to 43.9%
of questions on the Qwen models, against about
5% before training, and copies more on the Llama
models; the copy penalty cuts copying but declines
79.7% of questions on Qwen3.5-9B; and the sen-
tence reward copies more than the base model in
every setting, since scoring sentences one by one
does not close the copy shortcut. Where a baseline
takes no shortcut (5 of 60), its error rate is never
significantly below ASC’s, and ASC writes signif-
icantly more supported authored sentences in all
five, so no baseline matches it even there.
ASC gains faithfulness at no cost in quality.Two
LLM judges, Gemma-4-31B and Mistral-Small-
3.2-24B (Mistral AI, 2025) from a third model
family, pick, in both orders, the answer that serves
a reader better, under a rubric fixed on base-model
probes before any trained answer was judged, to
favor neither copied text nor padded answers (Ap-
pendix E). For every baseline the faithfulness gains

cost quality (Table 3): each raises the reward over
the base model, yet neither judge prefers any of
them to it. ASC is the only method that pays noth-
ing: both judges rate its answers on par with the
base model’s while they carry fewer errors and
more supported sentences, and prefer them to those
of every baseline. Blind human annotators, with
the same rubric, prefer ASC ’s answer to the base
model’s more often than the reverse (22 against
12 of 75 pairs) and GRPO + KL’s (28 against 14;
Appendix G). Exact match on ASQA stays within
one point of the base model for ASC , while the
copy penalty loses up to 25 points (Table 2).
Method RewardAgainst base ASC against itCov.
Gemma Mistral Gemma Mistral
GRPO + KL 92.7 43.6 43.4 56.4 55.9 42.0
Copy penalty 90.2 34.8 30.8 64.6 68.9 31.4
Sentence reward 92.2 49.5 48.0 51.5 51.1 45.1
ASC 90.6 51.0 49.1 46.8
Table 3: Answer quality over the 20 settings: train-
ing reward, win rate (%, a tie counts half) against the
base model and of ASC against each method under two
judges, and coverage (%) of the reference answer; the
base model’s reward is 83.5 and its coverage 45.0.
6.3 Each part of ASC contributes
VariantErr Mean
ASQA RAGT. ClapNQ ELI5 Expert Supp. Copy Vac.
ASC 4.0 1.1 2.2 3.4 3.7 3.0 22.2 13.3
−copy test 2.4 1.5 1.6 3.5 3.5 2.6 33.3 11.2
−vacuity test 5.21.0 1.64.6 7.5 2.9 21.3 19.1
Sentence reward 3.7 0.9 3.1 3.5 4.3 2.2 30.9 25.7
Answer advantage 5.4 2.3 1.9 5.1 5.1 3.3 25.1 18.3
Binary credit 4.6 1.7 2.5 4.1 4.4 3.0 20.9 12.8
Table 4: Ablations on Qwen3.5-9B, one part of ASC
changed at a time: Err on each corpus, and Supp., copy
and vacuous share averaged over the five corpora (grey
Err: a shortcut, as in Table 2). Binary credit: 1[p≥κ]
in place of ϕ; the sentence reward also drops both tests.
Table 4 changes one part of ASC at a time on
Qwen3.5-9B: removing any part brings back a
shortcut or raises the error rate on most corpora.
Without the copy test, copying returns (copy share
33.3 against 22.2, above the base model’s 28.1), so
the variant’s lower error rates are grey, and they
come with fewer supported authored sentences (2.6
against 3.0 per answer). Without the vacuity test,
vacuous sentences return (19.1% against 13.3%)
and the error rate on held-out ExpertQA doubles,
from 3.7 to 7.5. With GRPO’s whole-answer ad-
vantage in place of the sentence baseline, the policy
writes more (Supp. 3.3) but less reliably: the errorrate rises on four of the five corpora and the vac-
uous share to 18.3%, as Finding 3 predicts. With
binary credit 1[p≥κ] in place of the graded credit
ϕof Eq. 4, the error rate rises on all five corpora at
similar copy and vacuous shares: the graded credit
favors sentences the checker is sure of, which the
judge accepts more often. The sentence reward,
which drops both tests and the graded credit, copies
more than the base model and writes nearly twice
as many vacuous sentences as ASC.
6.4 Robustness
The gain holds across runs, evaluators and training,
and it does not game the copy test.
•Runs.Every run of ASC , each seed of each
model, has a lower authored error rate than the
base model on every corpus.
•Evaluators.Two checkers never used in training
also find more of ASC ’s authored sentences sup-
ported than the base model’s, in all 20 settings
under MiniCheck-Flan-T5 and in 19 under Align-
Score. With no automatic evaluator at all, the hu-
man annotators find 24.8% of ASC ’s sentences
unsupported against 33.2% of the base model’s
(250 ELI5 questions, Llama-3.1-8B;p= 0.01).
•Copy test. ASC does not learn to copy just below
its threshold: fewer of its authored sentences sit
just below it than the base model’s (9.2% against
11.3%), and with the threshold lowered to 0.4, its
share of copied sentences stays below the base
model’s in every setting (Appendix D.5).
•Training. ASC does not drift into a shortcut: on
every model, its copy share drops within the first
quarter of training and then stays flat, and its re-
fusals never exceed the base model’s, whereas
plain GRPO ends at a copy share of 72.8 (Find-
ing 2; curves in Figure 7, Appendix D.6).
7 Conclusion
In this work, we showed that faithfulness rewards
for long-form RAG pay for copying, and that com-
mon defenses only move this shortcut to shorter,
emptier answers and refusals. We then introduced
ASC , which credits only the sentences a model
authors, that the question alone does not support
and that the passages do, and compares each sen-
tence with every sentence of its group. On four
models, it lowers the authored error rate while writ-
ing more grounded sentences, held-out corpora in-
cluded, without these shortcuts or any loss in an-
swer quality. More broadly, we suggest asking of

any reward what a policy can earn without the be-
havior it is meant to teach. A RAG model should be
credited for what it writes, not for what it copies.
Limitations
ASC needs no model beyond the reward’s checker:
its copy test matches strings, and its vacuity and
support tests reuse the checker. This keeps the
method simple and lets it adopt a stronger checker
as one appears, but its tests are only as reliable as
that checker, which all methods in our comparison
share. The copy test is lexical, so a close para-
phrase counts as authored; ASC writes fewer such
near-copies than the base model in 17 of the 20
settings (Appendix D.5). We train models of up to
9B parameters on two training corpora, with one
training run for each baseline; the gain holds on all
four models and on the three held-out corpora, and
we leave larger models to future work.
ASC rewards the grounded sentences a model
writes, not declining, so it refuses less than the
base model, also on questions the passages do not
answer, where the judge prefers the base model’s re-
fusal (Appendix E). Pairing ASC with a grounded-
refusal objective (Song et al., 2025) is a natural
next step. Finally, ASC targets the reliability of
what a model writes itself; where an application
prefers verbatim quotation, plain faithfulness re-
wards already deliver it.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations.
Bespoke Labs. 2024. Bespoke-minicheck-7b.
Xilun Chen, Ilia Kulikov, Vincent-Pierre Berges, Barlas
Oguz, Rulin Shao, Gargi Ghosh, and Wen tau Yih.
2026. Learning to reason for factuality. InForty-
third International Conference on Machine Learning.
Yung-Sung Chuang, Benjamin Cohen-Wang, Zejiang
Shen, Zhaofeng Wu, Hu Xu, Xi Victoria Lin,
James R. Glass, Shang-Wen Li, and Wen-Tau Yih.
2025. SelfCite: Self-supervised alignment for con-
text attribution in large language models. InPro-
ceedings of the 42nd International Conference on
Machine Learning, volume 267 ofProceedings of
Machine Learning Research, pages 10839–10858.
PMLR.
Benjamin Cohen-Wang, Harshay Shah, Kristian
Georgiev, and Aleksander Madry. 2024. Contextcite:Attributing model generation to context. InThe
Thirty-eighth Annual Conference on Neural Infor-
mation Processing Systems.
Markus Dreyer, Mengwen Liu, Feng Nan, Sandeep
Atluri, and Sujith Ravi. 2023. Evaluating the tradeoff
between abstractiveness and factuality in abstractive
summarization. InFindings of the Association for
Computational Linguistics: EACL 2023, pages 2089–
2105, Dubrovnik, Croatia. Association for Computa-
tional Linguistics.
Angela Fan, Yacine Jernite, Ethan Perez, David Grang-
ier, Jason Weston, and Michael Auli. 2019. ELI5:
Long form question answering. InProceedings of
the 57th Annual Meeting of the Association for Com-
putational Linguistics, pages 3558–3567, Florence,
Italy. Association for Computational Linguistics.
Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An.
2025. Group-in-group policy optimization for LLM
agent training. InThe Thirty-ninth Annual Confer-
ence on Neural Information Processing Systems.
Leo Gao, John Schulman, and Jacob Hilton. 2023a.
Scaling laws for reward model overoptimization. In
Proceedings of the 40th International Conference
on Machine Learning, volume 202 ofProceedings
of Machine Learning Research, pages 10835–10866.
PMLR.
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen.
2023b. Enabling large language models to generate
text with citations. InProceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing, pages 6465–6488, Singapore. Associa-
tion for Computational Linguistics.
Gemma Team. 2026. Gemma 4 technical report.
Preprint, arXiv:2607.02770.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, et al. 2024. The llama 3 herd of mod-
els.Preprint, arXiv:2407.21783.
Yuzhe Gu, Wenwei Zhang, Chengqi Lyu, Dahua Lin,
and Kai Chen. 2025. Mask-DPO: Generalizable fine-
grained factuality alignment of LLMs. InThe Thir-
teenth International Conference on Learning Repre-
sentations.
Or Honovich, Roee Aharoni, Jonathan Herzig, Hagai
Taitelbaum, Doron Kukliansy, Vered Cohen, Thomas
Scialom, Idan Szpektor, Avinatan Hassidim, and
Yossi Matias. 2022. TRUE: Re-evaluating factual
consistency evaluation. InProceedings of the 2022
Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies, pages 3905–3920, Seattle,
United States. Association for Computational Lin-
guistics.

Chengyu Huang, Zeqiu Wu, Yushi Hu, and Wenya
Wang. 2024. Training language models to gener-
ate text with citations via fine-grained rewards. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 2926–2949, Bangkok, Thailand.
Association for Computational Linguistics.
Zhengping Jiang, Jingyu Zhang, Nathaniel Weir,
Seth Ebner, Miriam Wanner, Kate Sanders, Daniel
Khashabi, Anqi Liu, and Benjamin Van Durme. 2025.
Core: Robust factual precision with informative sub-
claim identification. InFindings of the Association
for Computational Linguistics: ACL 2025, pages
19833–19856, Vienna, Austria. Association for Com-
putational Linguistics.
Kalpesh Krishna, Aurko Roy, and Mohit Iyyer. 2021.
Hurdles to progress in long-form question answering.
InProceedings of the 2021 Conference of the North
American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies,
pages 4940–4957, Online. Association for Computa-
tional Linguistics.
Thomas Kwa, Drake Thomas, and Adrià Garriga-
Alonso. 2024. Catastrophic goodhart: regularizing
RLHF with KL divergence does not mitigate heavy-
tailed reward misspecification. InThe Thirty-eighth
Annual Conference on Neural Information Process-
ing Systems.
Faisal Ladhak, Esin Durmus, He He, Claire Cardie, and
Kathleen McKeown. 2022. Faithful or extractive?
on mitigating the faithfulness-abstractiveness trade-
off in abstractive summarization. InProceedings
of the 60th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 1410–1421, Dublin, Ireland. Association for
Computational Linguistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harri-
son Edwards, Bowen Baker, Teddy Lee, Jan Leike,
John Schulman, Ilya Sutskever, and Karl Cobbe.
2024. Let’s verify step by step. InThe Twelfth Inter-
national Conference on Learning Representations.
Chin-Yew Lin. 2004. ROUGE: A package for auto-
matic evaluation of summaries. InText Summariza-
tion Branches Out, pages 74–81, Barcelona, Spain.
Association for Computational Linguistics.
Nelson Liu, Tianyi Zhang, and Percy Liang. 2023. Eval-
uating verifiability in generative search engines. In
Findings of the Association for Computational Lin-
guistics: EMNLP 2023, pages 7001–7025, Singapore.
Association for Computational Linguistics.Yongchao Long, Yingying Zhang, Xianbin Wen, Xian
Wu, Yuxi Zhou, and Shenda Hong. 2026. Copy-paste
to mitigate large language model hallucinations. In
The Fourteenth International Conference on Learn-
ing Representations.
Varun Magesh, Faiz Surani, Matthew Dahl, Mirac Suz-
gun, Christopher D. Manning, and Daniel E. Ho.
2025. Hallucination-free? Assessing the reliabil-
ity of leading AI legal research tools.Journal of
Empirical Legal Studies, 22(2):216–242.
Chaitanya Malaviya, Subin Lee, Sihao Chen, Elizabeth
Sieber, Mark Yatskar, and Dan Roth. 2024. Ex-
pertQA: Expert-curated questions and attributed an-
swers. InProceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), pages 3025–3045,
Mexico City, Mexico. Association for Computational
Linguistics.
R. Thomas McCoy, Ellie Pavlick, and Tal Linzen. 2019.
Right for the wrong reasons: Diagnosing syntactic
heuristics in natural language inference. InProceed-
ings of the 57th Annual Meeting of the Association for
Computational Linguistics, pages 3428–3448, Flo-
rence, Italy. Association for Computational Linguis-
tics.
Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis,
Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettle-
moyer, and Hannaneh Hajishirzi. 2023. FActScore:
Fine-grained atomic evaluation of factual precision
in long form text generation. InProceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing, pages 12076–12100, Singa-
pore. Association for Computational Linguistics.
Mistral AI. 2025. Mistral-Small-3.2-24B-Instruct-2506.
Model card.
Christian Moya, Alex Semendinger, Guang Lin, and El-
liott Thornley. 2026. Spurious correlation learning in
preference optimization: Mechanisms, consequences,
and mitigation via tie training. InForty-third Inter-
national Conference on Machine Learning.
Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu,
KaShun Shum, Randy Zhong, Juntong Song, and
Tong Zhang. 2024. RAGTruth: A hallucination cor-
pus for developing trustworthy retrieval-augmented
language models. InProceedings of the 62nd An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 10862–
10878, Bangkok, Thailand. Association for Compu-
tational Linguistics.
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida,
Carroll Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, John
Schulman, Jacob Hilton, Fraser Kelton, Luke Miller,
Maddie Simens, Amanda Askell, Peter Welinder,
Paul F Christiano, Jan Leike, and Ryan Lowe. 2022.
Training language models to follow instructions with

human feedback. InAdvances in Neural Information
Processing Systems, volume 35, pages 27730–27744.
Curran Associates, Inc.
Qwen Team. 2026. Qwen3.5: Towards native multi-
modal agents. Blog post.
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christo-
pher D Manning, Stefano Ermon, and Chelsea Finn.
2023. Direct preference optimization: Your language
model is secretly a reward model. InThirty-seventh
Conference on Neural Information Processing Sys-
tems.
Paul Roit, Johan Ferret, Lior Shani, Roee Aharoni, Ge-
offrey Cideron, Robert Dadashi, Matthieu Geist, Ser-
tan Girgin, Leonard Hussenot, Orgad Keller, Nikola
Momchev, Sabela Ramos Garea, Piotr Stanczyk,
Nino Vieillard, Olivier Bachem, Gal Elidan, Avinatan
Hassidim, Olivier Pietquin, and Idan Szpektor. 2023.
Factually consistent summarization via reinforce-
ment learning with textual entailment feedback. In
Proceedings of the 61st Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 6252–6272, Toronto, Canada.
Association for Computational Linguistics.
Sara Rosenthal, Avirup Sil, Radu Florian, and Salim
Roukos. 2025. CLAPnq: Cohesive long-form an-
swers from passages in natural questions for RAG
systems.Transactions of the Association for Compu-
tational Linguistics, 13:53–72.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, Y . K. Li, Y . Wu, and Daya Guo. 2024.
Deepseekmath: Pushing the limits of mathemati-
cal reasoning in open language models.Preprint,
arXiv:2402.03300.
Itai Shapira, Gerdus Benade, and Ariel D. Procaccia.
2026. How rlhf amplifies sycophancy.Preprint,
arXiv:2602.01002.
Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia
Tsvetkov, Luke Zettlemoyer, and Wen-tau Yih. 2024.
Trusting your evidence: Hallucinate less with context-
aware decoding. InProceedings of the 2024 Confer-
ence of the North American Chapter of the Associ-
ation for Computational Linguistics: Human Lan-
guage Technologies (Volume 2: Short Papers), pages
783–791, Mexico City, Mexico. Association for Com-
putational Linguistics.
Shuzheng Si, Haozhe Zhao, Cheng Gao, Yuzhuo Bai,
Zhitong Wang, Bofei Gao, Kangyang Luo, Wenhao
Li, Yufei Huang, Gang Chen, Fanchao Qi, Minjia
Zhang, Baobao Chang, and Maosong Sun. 2025.
Teaching large language models to maintain contex-
tual faithfulness via synthetic tasks and reinforcement
learning.Preprint, arXiv:2505.16483.
Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg
Durrett. 2024. A long way to go: Investigating length
correlations in RLHF. InFirst Conference on Lan-
guage Modeling.Joar Max Viktor Skalse, Nikolaus H. R. Howe, Dmitrii
Krasheninnikov, and David Krueger. 2022. Defining
and characterizing reward gaming. InAdvances in
Neural Information Processing Systems.
Maojia Song, Shang Hong Sim, Rishabh Bhardwaj,
Hai Leong Chieu, Navonil Majumder, and Soujanya
Poria. 2025. Measuring and enhancing trustworthi-
ness of LLMs in RAG through grounded attributions
and learning to refuse. InThe Thirteenth Interna-
tional Conference on Learning Representations.
Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-
Wei Chang. 2022. ASQA: Factoid questions meet
long-form answers. InProceedings of the 2022 Con-
ference on Empirical Methods in Natural Language
Processing, pages 8273–8288, Abu Dhabi, United
Arab Emirates. Association for Computational Lin-
guistics.
Liyan Tang, Philippe Laban, and Greg Durrett. 2024.
MiniCheck: Efficient fact-checking of LLMs on
grounding documents. InProceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 8818–8847, Miami, Florida,
USA. Association for Computational Linguistics.
Katherine Tian, Eric Mitchell, Huaxiu Yao, Christo-
pher Manning, and Chelsea Finn. 2024. Fine-tuning
language models for factuality. InInternational Con-
ference on Learning Representations, volume 2024,
pages 44851–44866.
Liam van der Poel, Ryan Cotterell, and Clara Meis-
ter. 2022. Mutual information alleviates hallucina-
tions in abstractive summarization. InProceedings
of the 2022 Conference on Empirical Methods in Nat-
ural Language Processing, pages 5956–5965, Abu
Dhabi, United Arab Emirates. Association for Com-
putational Linguistics.
Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei
Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong,
Qi Liu, Tianyu Liu, and Zhifang Sui. 2024. Large lan-
guage models are not fair evaluators. InProceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 9440–9450, Bangkok, Thailand. Association
for Computational Linguistics.
Jerry Wei, Chengrun Yang, Xinying Song, Yifeng Lu,
Nathan Zixia Hu, Jie Huang, Dustin Tran, Daiyi Peng,
Ruibo Liu, Da Huang, Cosmo Du, and Quoc V Le.
2024. Long-form factuality in large language models.
InThe Thirty-eighth Annual Conference on Neural
Information Processing Systems.
Xueru Wen, Jie Lou, Xinyu Lu, Yuqiu Ji, Xinyan Guan,
Yaojie Lu, Hongyu Lin, Ben He, Xianpei Han, De-
bing Zhang, and Le Sun. 2025. On-policy self-
alignment with fine-grained knowledge feedback for
hallucination mitigation. InFindings of the Asso-
ciation for Computational Linguistics: ACL 2025,
pages 5215–5231, Vienna, Austria. Association for
Computational Linguistics.

Kevin Wu, Eric Wu, Kevin Wei, Angela Zhang, Allison
Casasola, Teresa Nguyen, Sith Riantawan, Patricia
Shi, Daniel Ho, and James Zou. 2025. An automated
framework for assessing how well LLMs cite rele-
vant medical references.Nature Communications,
16(1):3615.
Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane
Suhr, Prithviraj Ammanabrolu, Noah Smith, Mari
Ostendorf, and Hannaneh Hajishirzi. 2023. Fine-
grained human feedback gives better rewards for
language model training. InAdvances in Neural
Information Processing Systems, volume 36, pages
59008–59033. Curran Associates, Inc.
Xi Ye, Ruoxi Sun, Sercan Arik, and Tomas Pfister. 2024.
Effective large language model adaptation for im-
proved grounding and citation generation. InPro-
ceedings of the 2024 Conference of the North Amer-
ican Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume
1: Long Papers), pages 6237–6251, Mexico City,
Mexico. Association for Computational Linguistics.
Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xi-
aochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gao-
hong Liu, Juncai Liu, LingJun Liu, Xin Liu, Haibin
Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan
Tong, Chi Zhang, Mofan Zhang, and 17 others. 2025.
DAPO: An open-source LLM reinforcement learning
system at scale. InAdvances in Neural Information
Processing Systems, volume 38, Main Conference,
pages 113222–113244. Curran Associates, Inc.
Yuheng Zha, Yichi Yang, Ruichen Li, and Zhiting Hu.
2023. AlignScore: Evaluating factual consistency
with a unified alignment function. InProceedings
of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 11328–11348, Toronto, Canada. Association
for Computational Linguistics.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging
LLM-as-a-judge with MT-bench and chatbot arena.
InThirty-seventh Conference on Neural Information
Processing Systems Datasets and Benchmarks Track.
Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B.
Brown, Alec Radford, Dario Amodei, Paul Chris-
tiano, and Geoffrey Irving. 2020. Fine-tuning lan-
guage models from human preferences.Preprint,
arXiv:1909.08593.
A Formal Statements and Proofs
This section proves the formal claims of Sections 4
and 5 in the order they appear: the reward as a
line in the share of copied sentences (Eq. 2), the
properties of the graded credit, Proposition 1, and
the sentence form of the gradient (Eq. 7). Notation
follows Section 3, and∇stands for∇ θ.A.1 The reward is a line in the share of copied
sentences
Lemma 1(Reward decomposition).Let an an-
swer yhave nsentences, ncof them copied and
na=n−n cauthored (markup-only sentences,
rare in our data, are set aside), and let scandsa
be the shares of its copied and of its authored sen-
tences that the checker supports (either is 0 when
the answer has no sentence of that kind). With
ρ=n c/n, the reward of Eq. 1 is
R(y) =s a+ρ(s c−sa).
Proof. Split the sum of Eq. 1 by provenance. The
copied sentences contribute ncscsupported sen-
tences and the authored onesn asa, so
R(y) =ncsc+nasa
n=ρ s c+ (1−ρ)s a
=s a+ρ(s c−sa).
At fixed support rates, Ris therefore affine in ρ
with slope sc−sa. Turning one authored sentence
into a copied one raises ρby1/nand the reward
by(sc−sa)/n, although no authored sentence has
become more reliable. The slope is positive when-
ever copied sentences are supported more often
than authored ones, which holds under the reward’s
checker for every base model and corpus (Table 8).
A.2 The graded credit
Lemma 2(Graded credit).Let κ∈(0,1) and
ϕ(p) = 1−lnp/lnκ on[κ,1] , the branch of Eq. 4
above the threshold. Then
(a)ϕ(κ) = 0andϕ(1) = 1;
(b)ϕ is strictly increasing and strictly concave on
[κ,1];
(c)with the surprisal h(p) =−lnp ,ϕ(p) = 1−
h(p)/h(κ) , and ϕis the only function affine in
hthat satisfies (a).
Proof. (a)ln 1 = 0 gives ϕ(1) = 1 , and
lnκ/lnκ= 1givesϕ(κ) = 0. (b) Sincelnκ <0,
ϕ′(p) =−1
plnκ>0, ϕ′′(p) =1
p2lnκ<0.
(c) The identity follows from lnp/lnκ=
h(p)/h(κ) . Iff(h) =α+γh satisfies (a), then
f(h(1)) =f(0) =α= 1 andf(h(κ)) =
1 +γ h(κ) = 0 , soγ=−1/h(κ) andf(h(p)) =
ϕ(p).
Below the threshold Eq. 4 sets the credit to zero,
so a sentence is credited exactly when the checker’s
probability exceeds κ; above it, the credit grows
with the checker’s confidence, fastest nearκ.

A.3 Proof of Proposition 1
Fix a group of Ganswers and index its Msen-
tences by k, across all answers. Their credits
r1, . . . , r M∈[0,1] have mean µand standard de-
viationσ, and Eq. 5 givesa k=T(r k)with
T(r) =r−µ
σ+ϵ, ϵ >0.
The proof rests on two facts.(F1) Tis one map
for the whole group, whichever answer a sentence
belongs to, and it is strictly increasing, with slope
1/(σ+ϵ)>0 .(F2)A sentence that is copied,
vacuous or unsupported has rk= 0, the smallest
credit in [0,1] : Eq. 3 zeroes copied and vacuous
sentences, and Eq. 4 unsupported ones.
Proof. (i) Credit alone sets the order.By (F1),
T(rk)< T(r l)if and only if rk< rl, that is ak<
alif and only if rk< rl, whether sentences kandl
belong to the same answer or to different ones.
(ii) Shortcuts rank last.By (F2) and (i), a copied,
vacuous or unsupported sentence has the smallest
advantage of the group, ak=−µ/(σ+ϵ) . If any
sentence of the group earns credit, then µ >0
and this advantage is negative. A refusal earns
zero whenever the checker finds it unsupported
or vacuous, as for the copy penalty’s refusal in
Example 1 (Appendix I).
(iii) Average sentences earn nothing.Since
Mµ=P
krk,
MX
k=1ak=1
σ+ϵMX
k=1rk−Mµ
= 0,
andak= 0 exactly when rk=µ, so a sentence
gains only if its credit exceeds the mean.
(iv) Length-free advantages.Given the group’s µ
andσ,akdepends on the sentence only through rk,
which is computed from its text, the question and
the passages (Eq. 3); neither its number of tokens
nor the length of its answer enters, except through
µandσ, which every sentence of the group shares.
Part (iv) concerns the advantage; in the gradient,
each token of the sentence carries it (Lemma 3).
Scope.The proposition concerns sentences that
fail a test: it does not bound how many sentences
an answer has, and a paraphrase whose copy share
stays below τcounts as authored; Appendix D.5
measures how often such sentences occur.
Remark1 (GRPO’s answer-level advantage).Un-
der GRPO with the reward R, every token of an-
swerireceives A(i)= (R i−µR)/(σ R+ϵ), withµR,σRthe group’s mean and standard deviation of
R, and (i), (ii) and (iv) fail. Take a group of two
answers: answer 1 has three supported sentences,
one of them copied, so R1= 1; answer 2 has two
grounded sentences and one unsupported sentence,
soR2= 2/3 . Then A(1)> A(2), and the copied
sentence of answer 1 is reinforced more than the
two grounded sentences of answer 2. Deleting the
unsupported sentence of answer 2 raises R2to 1,
and with it the advantage of every remaining token,
so writing less pays.
A.4 The gradient as a policy gradient over
sentences
Lemma 3(Sentence form of the gradient).With
one update per batch, the gradient of the objective
in Eq. 6 atθ=θ oldis
∇J=1
NX
i,sa(i)
s∇lnπ θ 
y(i)
s|y(i)
<s
−β∇1
NX
i,tDi,t,
with the prompt (q, P) left implicit in the condi-
tioning, as in Eq. 7. Under GRPO, the first term is
1
NP
iA(i)∇lnπ θ(y(i)).
Proof. Fix an answer iand drop its index. At θ=
θoldthe importance ratio is wt= 1, so∇wt=
wt∇lnπ θ(yt|y<t) =∇lnπ θ(yt|y<t). Every
token tof sentence scarries ˆAt=as, and tokens
outside every sentence carry zero and drop out, so
grouping the tokens by sentence gives
X
tˆAt∇lnπ θ(yt|y<t)
=X
sasX
t∈s∇lnπ θ(yt|y<t)
=X
sas∇lnπ θ(ys|y<s),
where the last step is the chain rule of probability:
the log-probabilities of a sentence’s tokens add up
to the log-probability of the sentence given every-
thing before it. Under GRPO every token of answer
icarries the same A(i), and the same grouping over
the whole answer gives A(i)∇lnπ θ(y(i)). The KL
term does not depend on the advantages and is the
same for every method.
B Setup Details
This section gives what is needed to reproduce the
experiments: the data, the sentence tests and the

metrics built on them, and the training settings. The
prompts are listed in Appendix H.
Corpus Role Train Eval Eval (small) Passages
ASQA train + in domain 2,850 948 300 5
RAGTruth train + in domain 839 150 150 3
ClapNQ held out – 300 300 1
ELI5 held out – 1,000 300 5
ExpertQA held out – 1,183 300 3
Table 5: The five corpora. Train: training questions;
Eval: evaluation questions for the 9B and 8B models;
Eval (small): those for the 4B and 3B models; Passages:
passages shown with each question.
Data.
•Training.ASQA and RAGTruth are pooled for
training (3,689 questions) and evaluated on their
own evaluation sets, ALCE’s 948 ASQA ques-
tions and RAGTruth’s QA test sources.
•Held out.ClapNQ (its answerable development
questions), ELI5 and ExpertQA are never seen in
training. ELI5 is known to be hard to ground in
retrieved documents (Krishna et al., 2021), and
ExpertQA was added only after every training
run had finished, so nothing was tuned on it.
•No overlap.Training questions are disjoint from
every evaluation set by identifier, source docu-
ment and normalized question text.
•Size.The two smaller models are evaluated on at
most 300 questions per corpus, to bound the cost
of the judge.
Sentence tests. ASC and the metrics share three
sentence tests; the reward uses only the second.
•Copy.Answer and passages are lowercased and
split into words. A word of the answer is covered
if some window of five consecutive answer words
containing it also occurs in the passages, and the
copy share is the covered fraction. A sentence
is copied if the copy share of its text, with list
markers, headings and emphasis stripped, is at
least τ= 0.5 ; a sentence made only of such
markup is neither copied nor authored and earns
no credit.
•Support.A sentence is supported when the
checker’s probability is at least κ= 0.5 . Long
passages are split into chunks aligned with sen-
tences, and the maximum over chunks is taken.
•Vacuity.A sentence is vacuous when the support
check passes with the question in place of the
passages, a context ablation in the spirit of con-
ditional mutual information (van der Poel et al.,
2022) and SelfCite (Chuang et al., 2025).Bespoke-MiniCheck-7B (Bespoke Labs, 2024) is
the checker of the reward and of ASC ; MiniCheck-
Flan-T5-Large (Tang et al., 2024) and AlignScore
(Zha et al., 2023) are the held-out checkers, used
only in evaluation.
Metric Definition Instrument
Err authored sentences judged un-
supported, over authored sen-
tencesGemma-4-
31B
T5, AS authored sentences supported,
over authored sentencesMiniCheck-
Flan-T5,
AlignScore
Supp. supported authored sentences
per answerMiniCheck-
Flan-T5
Copy answer words inside a verba-
tim five-word match with the
passages, over wordsstring match
Vac. supported authored sentences
also supported by the question
alone, over supported authored
sentencesBespoke-
MiniCheck-
7B
Ref. answers whose every sentence
declines to answer, over an-
swersrule-based
Words words per answer –
Reward sentences supported, over sen-
tences (Eq. 1)Bespoke-
MiniCheck-
7B
No auth. answers without an authored
sentence, over answersstring match
EM short answers of ASQA found
in the answerALCE string
match
Table 6: The metrics of Tables 2 and 4 and of the ap-
pendix, on a 0–100 scale except counts per answer. Only
Bespoke-MiniCheck-7B is also used in training.
Metrics.The judge, Gemma-4-31B (Gemma
Team, 2026) at full precision, reads the passages
and one authored sentence (prompt in Appendix H);
a sentence is an error when the judge rejects it out-
right, with a probability of “Yes” below 3.5×10−10.
Its probabilities are nearly binary (46% of sen-
tences below 0.001 and 42% above 0.999), and this
rule, chosen on one half of RAGTruth’s human-
labeled responses and tested on the other, reaches a
balanced accuracy of 0.81 against the human labels,
above the 0.75 of the reward’s checker. Copying
counts as a shortcut even though Err leaves copied
sentences out: a policy that copies more submits
fewer of its own sentences to the judge, so Err can
fall without any sentence becoming more reliable,
which Supp. makes visible. An answer refuses only
when every sentence declines to answer, so a single
hedge is not a refusal.
Training.All methods share the settings of Ta-
ble 7: one epoch over the pooled training questions

Setting Value
Answers per questionG8
KL coefficientβ0.04
Learning rate 1e-6
Warm-up steps 36
Epochs 1
Updates per batch 1
Loss token-normalized
Max. answer tokens 1024
Sampling temperature / top-p0.7 / 0.95
Gradient clipping 1.0
Precision bf16, full weight
Standardizationϵ1e-4
Table 7: GRPO settings of every method and model.
(922 steps, 32 answers per step), with rollouts sam-
pled at the listed temperature; they differ only in
the training signal, and Algorithm 1 gives one ASC
update. Evaluation decodes greedily with the same
token budget. For the copy penalty, λis the support
gapsc−saof the model’s own samples on the train-
ing questions under the reward’s checker: 0.071
for Qwen3.5-9B, 0.087 for Qwen3.5-4B, 0.077 for
Llama-3.1-8B and 0.123 for Llama-3.2-3B. The
sentence reward uses the sentence baseline of Eq. 5
with rs=1[g(y s, P)≥κ] for every sentence
of the answer. ASC is trained with two seeds on
Qwen3.5-9B and three on Qwen3.5-4B, and the
binary-credit ablation with three seeds on Qwen3.5-
9B; every other policy is one training run. Every-
thing runs on eight H100 80GB GPUs: one training
run on a 9B model takes about 3.5 hours on six of
them, with the checker served on the remaining
two, and evaluating one policy on all corpora, in-
cluding the judge, takes about two hours on two.
Penalty strengths.Neither penalty is tuned
on the evaluation corpora. The KL coefficient
β= 0.04 is GRPO’s default (Shao et al., 2024),
and no other value removes the shortcuts: the KL-
regularized objective is maximized by π∗(y)∝
πref(y) exp(R(y)/β) (Rafailov et al., 2023), which
favors whatever raises Rat every β >0 , so a larger
βonly shrinks the reward gain together with the
shortcut. The copy penalty’s λis the smallest value
at which, by Eq. 2, a copied sentence earns no
more than an authored one; a larger λwould only
push the policy further from the passages, which
its refusals already show.
C Analysis Details
This section gives the evidence behind the three
findings of Section 4, one subsection each.Algorithm 1One ASC update for a prompt (q, P)
Require: policy πθ, reference πref, checker g, thresholds
τ, κ, group sizeG
1: Sampley(1), . . . , y(G)∼πθold(· |q, P)
2:foreach sentencey(i)
sof each answery(i)do
3:ifc(y(i)
s)≥τorg(y(i)
s, q)≥κthen
4:r(i)
s←0▷copied or vacuous
5:else
6:r(i)
s←ϕ 
g(y(i)
s, P)
▷graded support
7:end if
8:end for
9:µ, σ←mean, s.d. of all credits in the group
10:ˆAi,t←(r(i)
s−µ)/(σ+ϵ)for eacht∈y(i)
s
11:θ←θ+η∇ θJ(θ)▷Eq. 6, step sizeη
C.1 Copied versus authored sentences
(Finding 1)
Model CorpusReward checker MiniCheck-Flan-T5 AlignScore
Auth. Copied Auth. Copied Auth. Copied
Qwen3.5-9B ASQA 15.3 4.2 17.4 6.1 12.2 6.3
RAGTruth 7.7 3.4 13.7 5.8 6.4 4.9
ClapNQ 3.6 1.1 10.9 5.0 3.1 0.7
ELI5 23.9 3.0 29.2 8.6 10.7 8.5
Qwen3.5-4B ASQA 18.1 5.5 19.0 5.3 13.9 7.5
RAGTruth 8.3 4.5 14.0 6.6 7.4 5.0
ClapNQ 5.7 1.5 13.2 5.6 3.7 1.2
ELI5 26.3 5.5 29.4 8.4 13.4 8.8
Llama-3.1-8B ASQA 18.3 5.8 19.3 7.4 16.5 8.0
RAGTruth 9.1 4.6 14.9 11.2 6.2 6.7
ClapNQ 6.0 2.3 16.7 7.7 5.6 0.5
ELI5 24.3 6.0 29.5 9.0 22.5 9.1
Llama-3.2-3B ASQA 26.2 7.7 21.7 7.3 19.6 8.3
RAGTruth 14.7 4.6 19.5 8.0 10.7 4.1
ClapNQ 10.5 3.0 21.6 8.4 8.1 0.9
ELI5 32.6 9.6 34.3 9.5 26.5 8.2
Table 8: Unsupported rate (%) of authored (Auth.) and
copied sentences in base-model samples (eight answers
per question) under the reward’s checker and the two
held-out checkers of Appendix B.
Under the reward’s checker, authored sentences
are unsupported 1.8 to 8.1 times as often as copied
ones for every model and corpus, most on ELI5,
where base models write most of the answer them-
selves. The held-out checkers find the same gap,
MiniCheck-Flan-T5 everywhere and AlignScore in
all but one case, so it is not a quirk of one checker:
entailment models accept hypotheses that overlap
their premise (McCoy et al., 2019), and a copied
sentence overlaps it entirely. Human span labels
on RAGTruth (131,087 sentences, no checker in-
volved) agree: copied sentences carry errors at
1.9% and authored ones at 12.2%, and the error
rate falls steadily with copy share, from 13.7% in
the lowest decile to 0.6% in the highest.
Within a question.Real samples follow Lemma 1:
among the answers to the same question, the reward
rises with the copy share c(y) for every model and
corpus. The pull acts even before training: picking
the highest-reward answer of the eight raises the

copy share for every model and corpus.
Tercile ROUGE-L Sentences Reward checker MiniCheck-Flan-T5
least similar 0.27 7,362 74.8 73.3
middle 0.40 7,285 86.2 82.9
closest 0.57 7,323 91.4 89.2
Table 9: Authored, non-vacuous sentences of Qwen3.5-
9B and Llama-3.1-8B base-model samples on ASQA
and RAGTruth, by tercile of ROUGE-L similarity to
their closest passage sentence: mean similarity, count,
and share (%) supported under each checker.
Closeness to the passages.The preference does
not hinge on the copy test’s five-word cut-off: both
checkers accept an authored sentence more often
the closer it stays to its closest passage sentence
(Table 9), so the reward’s preference for copying is
a property of the checker, not of the threshold.
C.2 Training on the faithfulness reward
(Finding 2)
Training Reward T5 Copy Words Vac. Ref. No auth. EM
Qwen3.5-9B
Base model 87.1 82.5 38.8 73.5 21.5 6.1 15.4 45.5
GRPO 98.9 96.1 72.8 13.7 18.1 0.0 81.0 27.6
GRPO + KL 92.3 87.7 39.2 60.0 28.5 14.7 19.5 41.8
Copy penalty 92.0 94.3 11.8 36.9 67.4 64.5 1.3 20.2
DPO 94.6 91.0 53.4 55.7 16.9 1.7 37.2 41.8
DPO, copy-matched 90.4 85.8 44.9 59.7 21.5 6.1 24.7 43.8
Llama-3.1-8B
Base model 87.3 83.0 45.5 24.0 25.8 7.0 47.9 35.2
GRPO 99.8 99.8 4.6 7.6 58.9 0.0 5.2 0.5
Table 10: Policies trained on R, evaluated on ASQA
at the final checkpoint. T5: authored support under
MiniCheck-Flan-T5; No auth.: answers without an au-
thored sentence; other columns as in Table 1. DPO,
copy-matched: pairs whose two answers copy equally.
Every policy trained on Rraises the reward
through a shortcut rather than through better au-
thored sentences (Table 10): authored support rises
too, but on fewer authored sentences. Plain GRPO
copies until 81.0% of its answers contain no au-
thored sentence; GRPO + KL refuses and writes
emptier answers; DPO copies more even when the
two answers of every pair copy equally. On Llama-
3.1-8B, plain GRPO settles on a vacuous template
whose most frequent sentences are “A film was
made.” and “A song was released.”, which earns a
reward of 99.8 at an exact match of 0.5.
Over training.Shortcuts set in early and persist
(Figure 6): GRPO’s answers shrink within the first
quarter of training and its copy share passes 70 by
mid-training, and the copy penalty’s refusals climb
to 69.3% by three quarters of training, while ASC
brings copy share and refusals down within the first
quarter and keeps them there.Runs.Plain GRPO is GRPO + KL with β= 0 .
The DPO policies are trained for one epoch on pairs
of base-model samples ranked by R; in the copy-
matched pairs, the two answers’ copy shares differ
by at most 0.02 and their sentence counts by at
most one. Table 1, Table 10 and the GRPO point of
Figure 1b come from the evaluation pass of these
runs, which decoded the base model separately
from the runs of Section 6, so their base values
differ slightly from Tables 12 and 13 (by at most 2.7
points, Llama-3.1-8B’s vacuous share on ASQA).
C.3 Whole-answer and sentence advantages
(Finding 3)
Sentence CountAll sentences (%) Pushed up, if moved (%)
Up Down Unmoved Pooled Range
Copied 38,106 36.3 14.1 49.6 72.0 66.8–83.0
Vacuous 21,344 40.9 28.7 30.5 58.8 54.7–75.6
Grounded 72,765 50.1 16.3 33.6 75.5 70.1–93.3
Authored, unsupported 18,011 19.7 75.7 4.6 20.7 3.0–27.8
Table 11: How GRPO’s answer-level advantage moves
each kind of sentence, on the base-model samples
of Finding 1. Kinds, in the order of ASC ’s tests:
copied; vacuous (authored, supportable from the ques-
tion alone); grounded (authored, not vacuous, supported
by the reward’s checker); authored, unsupported. Count:
sentences of that kind. Up, Down: the answer’s reward
is above, below its group’s mean; Unmoved: equal to it,
so the advantage is zero. Pushed up, if moved: the share
pushed up among the sentences GRPO moves, pooled
and its range over models and corpora.
Signs of GRPO’s advantage.Every sentence
takes the sign of its answer’s advantage, so Table 11
counts how GRPO moves each kind of sentence.
It leaves a third of all sentences, and half of the
copied ones, unmoved, because their answer’s re-
ward equals the group’s mean; among the sentences
it moves, it pushes up copied and grounded ones
at nearly the same rate, and most vacuous ones
too. Under ASC , a copied, vacuous or unsupported
sentence always takes the lowest advantage of its
group (Proposition 1(ii)), so none is pushed up.
Correlation with held-out support.On the same
pool as Table 9 (21,970 authored, non-vacuous sen-
tences in 1,967 groups), we compute two advan-
tages from the binary form of the credit in Eq. 3:
GRPO’s answer-level advantage, with the credit
summed over the answer, and the sentence advan-
tage of Eq. 5. Their correlation with the support of
the authored sentences under MiniCheck-Flan-T5
is 0.07 and 0.21: the same credit tracks a sentence’s
reliability far more closely when sentences are com-
pared with sentences rather than answers with an-

0 .5 120406080copy share (%)
(a) Copy share
0 .5 120406080words per answer
(b) Length
0 .5 1
training progress204060vacuous share (%)
(c) Vacuous
0 .5 10204060refusals (%)
(d) Refusals
0 .5 1859095authored support (%)
(e) SupportGRPO GRPO + KL Copy penalty ASCFigure 6: Qwen3.5-9B on ASQA over training (0: base model): every metric of Figure 3, with vacuous share and
authored support under MiniCheck-Flan-T5, and ASC for comparison. Policies trained on Rraise authored support
by copying (GRPO) or by writing less and refusing (KL, copy penalty); ASC copies less and writes longer answers.
swers. Answer length explains less than 2% of the
variance of either advantage, so the difference does
not come from length.
D Full Results
This section expands Table 2: every metric on every
corpus, counts per answer with their significance,
every training run, the robustness of the shortcut
rule, near-copies, and how each model trains. Sig-
nificance marks test a difference with a paired boot-
strap over questions (1,000 resamples, seeds pooled
per question); they cover the sampling of questions,
not of training runs (Appendix D.3 lists the runs),
and each comparison is tested on its own.
D.1 Every metric on every corpus
Faithfulness.The one setting in which AlignScore
does not confirm the gain of Section 6.4 is a 0.2-
point dip (Qwen3.5-9B on ELI5). The absolute
gain is largest where the base model errs most,
up to 15.6 points for Llama-3.2-3B on ExpertQA,
which is the same relative cut as elsewhere (38%,
Figure 5).
Shortcuts.The baselines’ shortcuts concentrate
on ELI5 and ExpertQA, the held-out corpora where
the base model errs most and an answer must be
written rather than assembled (Section 3). There
GRPO + KL declines 44 to 76% of the questions on
the Qwen models, against at most 28% on the other
corpora, and the copy penalty declines 94 to 95%
on Qwen3.5-9B. On the Llama models, GRPO +
KL and the sentence reward copy more than the
base model on every corpus, and the copy penalty
shortens the answers instead, to 56 to 89% of thebase model’s words. ASC does the opposite: it
copies less, writes fewer vacuous sentences and re-
fuses less than the base model on every corpus, and
writes longer answers on every held-out corpus.
D.2 Counts per answer and significance
Err is a rate over authored sentences, and ASC
writes more of them, so a lower rate could hide
more errors per answer; Table 14 therefore counts
them. Err is lower in every setting, significantly
in 14 of the 20. Per answer, ASC writes 0.3 to 1.0
more judge-supported authored sentences, and its
unsupported ones fall in 16 settings and rise signifi-
cantly in none. Counting copied sentences too, the
unsupported share of the whole answer falls in 18
settings and rises significantly in none, so answers
become more faithful as a whole. Within bins of
answer length, ASC still writes more supported
authored sentences in all 20 settings, so the added
content is not a by-product of length.
D.3 Every training run
Table 15 lists every ASC training run separately.
Each run has a lower authored error rate than the
base model on every corpus it was evaluated on, so
no single seed carries the gain.
D.4 Robustness of the shortcut rule
The shortcut rule of Section 6.1 has two free
choices, the margin and the word floor, and Ta-
ble 16 recomputes the ranking of Table 2 under
other values of both. ASC is flagged in no set-
ting under any of them. The number of settings in
which it has the lowest Err falls at wider margins,

MethodASQA RAGTruth ClapNQ ELI5 ExpertQA
Err T5 AS Supp. Err T5 AS Supp. Err T5 AS Supp. Err T5 AS Supp. Err T5 AS Supp.
Qwen3.5-9B
Base model 5.1 82.3 89.9 1.6 2.3 87.9 94.2 3.0 3.6 88.4 96.7 1.8 6.0 70.9 89.6 2.2 6.2 71.5 89.7 2.3
GRPO + KL 4.0 87.7 91.2 1.4 1.2 88.8 93.0 3.0 3.2 90.6 97.3 1.8 5.4 79.5 91.1 2.0 5.9 81.6 92.9 2.1
Copy penalty 1.5 94.3 95.1 1.4 1.2 73.5 92.8 1.2 0.4 87.6 98.0 1.5 1.1 94.0 98.8 1.1 0.4 74.5 98.6 0.9
Sentence reward 3.7 87.1 – 1.5 0.9 90.9 – 2.9 3.1 90.7 – 1.7 3.5 80.0 – 2.4 4.3 80.4 90.0 2.5
ASC 4.0 87.9 90.7 2.3 1.1 90.9 94.6 3.8 2.2 90.8 98.3 2.5 3.4 78.8 89.4 3.2 3.7 84.9 90.7 3.4
Qwen3.5-4B
Base model 8.6 80.0 87.1 1.7 2.9 87.9 92.2 2.9 5.8 89.6 96.1 1.8 8.8 72.7 88.1 2.3 8.7 64.7 85.1 2.1
GRPO + KL 3.2 91.8 94.3 1.4 2.0 86.3 93.9 2.3 2.9 88.6 98.2 1.5 3.9 90.5 95.0 1.5 3.5 80.8 95.9 1.4
Copy penalty 8.2 90.9 – 1.7 3.1 86.9 – 3.2 4.9 90.0 – 2.3 6.7 79.6 – 2.2 9.1 75.0 91.1 2.1
Sentence reward 4.3 85.3 – 1.72.289.7 – 3.0 4.2 91.5 – 1.8 5.9 76.8 – 2.8 5.9 76.3 87.1 2.8
ASC 6.5 87.3 90.5 2.4 2.3 89.9 94.3 3.5 4.0 91.5 97.5 2.6 6.6 80.9 89.4 3.2 7.7 81.7 88.7 3.2
Llama-3.1-8B
Base model 11.7 82.6 86.3 0.5 8.5 85.7 91.9 1.6 14.0 81.4 93.0 0.7 27.9 69.1 76.6 1.4 27.3 72.1 79.6 1.2
GRPO + KL 6.6 90.2 92.1 0.4 6.7 90.0 94.4 1.65.688.5 97.0 0.8 21.1 77.1 82.3 1.3 22.5 78.4 84.1 1.4
Copy penalty 8.0 91.6 90.9 0.8 6.0 87.1 92.8 2.0 8.8 88.3 94.4 1.1 17.6 80.2 84.0 1.5 20.5 78.0 85.1 1.5
Sentence reward 4.8 90.5 – 0.4 5.1 88.6 – 1.4 5.5 88.9 – 0.8 18.2 78.8 – 1.4 20.0 78.7 85.9 1.3
ASC 6.0 86.6 91.4 0.8 3.9 88.2 95.8 2.2 7.2 90.4 97.9 1.3 17.4 80.8 84.8 1.8 19.9 81.5 86.8 1.8
Llama-3.2-3B
Base model 18.9 84.9 81.8 0.5 11.7 83.8 89.2 1.2 14.5 75.4 91.7 0.6 38.7 63.3 72.7 1.4 40.5 63.6 72.6 1.0
GRPO + KL 14.4 89.0 91.5 0.4 5.7 89.8 90.3 1.1 10.0 88.1 97.5 0.6 25.4 77.2 83.2 1.4 28.0 73.8 82.4 1.1
Copy penalty 10.7 92.0 – 0.8 9.0 91.4 – 1.7 9.9 85.8 – 1.0 26.8 73.8 – 1.3 27.1 75.3 82.8 1.4
Sentence reward 11.8 92.7 – 0.3 7.6 89.1 – 1.1 6.3 90.5 – 0.7 23.8 76.9 – 1.4 24.0 76.9 83.7 1.1
ASC 15.0 86.2 85.4 0.8 7.1 88.2 93.5 2.0 8.7 84.4 96.3 1.2 24.9 75.5 83.2 2.1 25.0 77.5 84.9 1.8
Table 12: Faithfulness on every corpus: authored error rate under the judge (Err; grey: a shortcut, not ranked; bold,
underlined: lowest and second-lowest remaining Err, as in Table 2), authored support under MiniCheck-Flan-T5
(T5) and AlignScore (AS), and supported authored sentences per answer (Supp.); –: not scored.
MethodASQA RAGTruth ClapNQ ELI5 ExpertQA
Copy Words Vac. Ref. Copy Words Vac. Ref. Copy Words Vac. Ref. Copy Words Vac. Ref. Copy Words Vac. Ref.
Qwen3.5-9B
Base model 38.9 73.3 22.5 6.1 30.1 119.3 16.6 2.7 38.6 82.8 21.2 4.3 13.5 102.1 38.6 4.3 19.2 109.6 40.5 7.3
GRPO + KL 39.3 60.0 28.4 14.7 29.7 113.6 18.6 7.3 39.1 81.3 21.7 5.0 11.7 77.0 43.4 45.9 16.2 83.1 49.0 43.7
Copy penalty 11.8 36.9 67.4 64.5 5.0 34.3 81.6 78.7 11.4 41.3 72.7 66.0 0.9 16.3 95.2 95.4 1.8 18.7 94.8 93.7
Sentence reward 42.3 66.4 22.8 7.3 34.0 116.7 15.9 2.7 42.2 80.6 19.0 2.0 14.5 92.2 35.9 28.2 21.3 104.5 34.8 18.8
ASC 28.6 80.9 10.9 0.2 22.7 122.4 12.6 0.3 28.4 89.7 13.7 0.2 12.9 123.5 14.1 0.3 18.3 127.8 15.2 0.1
Qwen3.5-4B
Base model 38.2 84.0 23.9 5.7 32.7 118.9 19.2 3.3 40.3 86.0 20.2 3.0 15.9 109.0 37.2 7.0 22.2 121.0 31.1 7.7
GRPO + KL 32.6 49.7 43.7 27.7 28.4 93.8 34.5 22.7 35.8 68.0 39.9 22.7 6.9 38.8 75.9 75.7 9.8 46.8 74.9 70.7
Copy penalty 24.4 58.0 37.9 28.0 23.2 115.5 24.0 10.7 27.7 82.5 26.4 12.3 9.8 83.8 46.4 48.7 13.7 92.0 46.8 41.0
Sentence reward 43.2 83.9 17.8 2.0 34.0 119.2 18.2 1.3 43.7 87.6 19.9 1.3 17.9 118.1 28.7 13.7 25.5 131.5 25.4 3.7
ASC 28.4 86.3 12.1 0.4 25.0 119.9 12.5 0.4 29.4 91.9 12.0 0.4 15.3 124.7 13.5 2.4 21.9 131.0 14.4 1.0
Llama-3.1-8B
Base model 45.4 24.1 28.5 6.9 43.1 69.5 17.8 0.7 43.4 36.5 21.0 0.3 25.3 60.2 26.5 7.8 32.7 61.8 27.5 7.6
GRPO + KL 50.1 21.3 29.5 3.4 44.0 69.0 23.0 2.7 45.2 38.6 20.5 0.3 31.3 55.1 26.2 7.7 35.7 63.0 28.3 6.1
Copy penalty 14.2 13.7 34.2 10.4 22.8 61.5 21.9 4.0 17.6 29.9 22.3 1.0 9.3 44.4 36.5 12.1 13.8 49.5 33.2 7.3
Sentence reward 51.7 22.4 27.9 3.2 47.0 68.7 18.7 2.0 47.1 40.0 19.0 0.0 29.9 56.5 24.3 9.2 37.3 61.1 25.3 5.7
ASC 30.2 27.7 9.9 1.3 26.2 74.3 16.1 0.0 26.2 44.8 14.1 0.0 17.9 61.5 13.2 1.1 23.4 67.3 15.3 1.0
Llama-3.2-3B
Base model 49.3 22.4 21.3 7.0 45.7 68.1 15.3 2.0 45.8 40.4 26.0 3.7 27.0 71.8 15.1 2.7 33.6 60.2 22.5 6.0
GRPO + KL 57.9 22.5 25.8 4.7 55.4 63.9 17.4 1.3 54.6 41.8 22.5 2.0 34.8 62.1 16.5 2.7 44.3 64.6 20.2 5.3
Copy penalty 14.7 12.6 34.5 10.0 29.4 60.2 17.8 2.0 22.0 26.8 28.9 3.0 12.4 42.2 28.1 12.0 15.9 47.3 32.2 10.7
Sentence reward 59.5 22.0 18.5 2.0 54.7 64.8 12.8 1.3 54.3 46.1 18.4 1.0 35.5 64.7 19.4 1.3 44.8 64.0 17.7 4.3
ASC 31.0 26.6 9.8 3.7 26.0 66.5 13.4 1.3 28.9 45.8 17.5 0.7 15.4 73.5 14.0 2.3 21.5 67.0 15.9 4.3
Table 13: Shortcut metrics on every corpus: copy share, words per answer, vacuous share (Vac.) and refusal rate
(Ref.). A policy takes a shortcut on a corpus when its copy share, vacuous share or refusal rate exceeds the base
model’s by more than 3 points, or its words fall below 85% of the base model’s.
mostly because they stop flagging the sentence re-
ward, which copies a median 3.8 points more than
the base model and 14.0 more than ASC . With-
out any threshold the picture is the same: in 19
of the 20 settings, no baseline has both a lower
Err and at least as many supported authored sen-tences per answer as ASC ; the exception, the copy
penalty on ASQA for Llama-3.2-3B, writes an-
swers half as long. Where a baseline takes no
shortcut, in 5 of its 60 results, ASC ’s error rate
is lower in one (Llama-3.1-8B on ExpertQA, 19.9
against 22.5 for GRPO + KL) and no different in

CorpusErr Unsupp. (J) Supp. (J) Whole
Base ASC Base ASC Base ASC Base ASC
Qwen3.5-9B
ASQA 5.1 3.9∗0.10 0.10 1.8 2.5 13.4 10.5∗
RAGT. 2.3 1.1 0.08 0.05 3.4 4.1 10.2 8.5
ClapNQ 3.6 2.2 0.07 0.06 2.0 2.7 9.5 8.8
ELI5 6.0 3.4∗0.19 0.14∗2.9 3.9 27.4 20.6∗
Expert 6.2 3.7∗0.20 0.15∗2.9 3.9 24.6 13.9∗
Qwen3.5-4B
ASQA 8.6 6.5 0.18 0.17 1.9 2.5 14.6 10.9∗
RAGT. 2.9 2.3 0.09 0.09 3.2 3.8 10.4 9.4
ClapNQ 5.8 4.0∗0.12 0.11 1.9 2.7 9.1 7.9
ELI5 8.8 6.6∗0.28 0.26 2.9 3.7 24.6 18.1∗
Expert 8.7 7.7 0.28 0.30 3.0 3.6 30.0 16.2∗
Llama-3.1-8B
ASQA 11.7 6.0∗0.07 0.05 0.5 0.8 11.9 11.5
RAGT. 8.5 3.9∗0.15 0.10 1.7 2.4 11.1 11.6
ClapNQ 14.0 7.2∗0.12 0.10 0.7 1.3 13.3 9.1∗
ELI5 27.9 17.4∗0.55 0.38∗1.4 1.8 26.1 17.9∗
Expert 27.3 19.9∗0.46 0.43 1.2 1.7 21.5 16.2∗
Llama-3.2-3B
ASQA 18.9 15.0 0.10 0.13 0.4 0.7 10.2 11.8
RAGT. 11.7 7.1∗0.17 0.16 1.3 2.1 12.2 11.4
ClapNQ 14.5 8.7∗0.11 0.13 0.7 1.3 15.9 13.8
ELI5 38.7 24.9∗0.84 0.69∗1.3 2.1 31.6 23.6∗
Expert 40.5 25.0∗0.66 0.57 1.0 1.7 28.0 20.4∗
Table 14: Base model against ASC on every corpus
(ASC ’s seeds pooled per question): authored error rate
(Err), unsupported and supported authored sentences per
answer under the judge (Unsupp. (J), Supp. (J)), and the
share (%) of all sentences, copied ones included, that
MiniCheck-Flan-T5 does not support (Whole).∗(on
Err, Unsupp. and Whole): the 95% bootstrap interval
over questions of the difference from the base model
excludes zero. Pooling seeds per question makes Err
differ slightly from Table 2.
Model Run ASQA RAGT. ClapNQ ELI5 Expert
Qwen3.5-9B Base model 5.1 2.3 3.6 6.0 6.2
Seed 1 4.5 1.5 2.0 3.3 4.1
Seed 2 3.4 0.8 2.3 3.5 3.4
Qwen3.5-4B Base model 8.6 2.9 5.8 8.8 8.7
Seed 1 6.6 2.0 3.5 6.3 6.5
Seed 2 5.7 2.6 4.0 6.3 8.6
Seed 3 7.1 2.1 4.5 7.1 7.9
Llama-3.1-8B Base model 11.7 8.5 14.0 27.9 27.3
Seed 1 6.0 3.9 7.2 17.4 19.9
Llama-3.2-3B Base model 18.9 11.7 14.5 38.7 40.5
Seed 1 15.0 7.1 8.7 24.9 25.0
Table 15: Authored error rate of every ASC training run
on every corpus, with the base model above.
the other four, while ASC writes significantly more
judge-supported authored sentences in all five (0.4
to 0.8 per answer).
D.5 Near-copies
A policy trained against a copy test could learn to
copy just below its threshold. Section 6.4 rules this
out for verbatim copying (just below: a copy shareMargin ASC flaggedASC lowest Err Baseline
cells flagged75% 85% 95%
1 0 20 20 20 59
2 0 18 18 18 57
3 0 18 18 18 55
4 0 15 15 16 48
5 0 15 15 16 43
7.5 0 12 12 14 37
10 0 8 8 12 29
Table 16: Table 2 under other shortcut rules. Margin:
points above the base model at which copy share, vac-
uous share or refusals flag a shortcut. ASC flagged:
settings (of 20) in which ASC takes a shortcut. ASC
lowest Err: settings in which it has the lowest Err among
policies without a shortcut, for word floors of 75%, 85%
and 95% of the base model’s words. Baseline cells
flagged: baseline Err cells (of 60) with a shortcut, floor
85%. Table 2 uses a margin of 3 and a floor of 85%.
of 0.40 to 0.50), and with the threshold lowered to
0.4,ASC also keeps its lower error rate. Beyond
verbatim matches, an authored sentence of ASC
is as close to its closest passage sentence as one
of the base model (median ROUGE-L precision
0.42 for both; Lin, 2004), and ASC writes fewer
near-copies, with a ROUGE-L of at least 0.8, in 17
of the 20 settings. Within terciles of closeness, its
authored error rate is below the base model’s in 58
of the 60 tercile and setting pairs, so its gain is not
a product of closeness.
D.6 Training dynamics
Figure 7 follows GRPO + KL and ASC through
training on every model, and the copy penalty
on Qwen3.5-9B and Llama-3.1-8B. The baselines’
shortcuts build up over training: GRPO + KL’s re-
fusals rise from the first checkpoint on the Qwen
models, up to 29% of answers for Qwen3.5-4B, and
its copy share rises on the Llama models, while the
copy penalty’s answers shrink as its refusals and
vacuous share grow. Their authored support ends
at or above ASC ’s, but over the fewer sentences
they still write. ASC settles early on every model:
its copy share falls within the first quarter of train-
ing and then stays flat, its vacuous share falls by
mid-training, and it ends with longer answers and
fewer refusals than the base model.
E Answer Quality
Section 6.2 compares whole answers with LLM
judges. This section gives the protocol, how its
rubric was chosen without favoring any method,
the result in every setting, a second judge, and

10203040Qwen3.5-9B
Copy share (%)
406080
Words per answer
204060
Vacuous share (%)
0204060
Refusals (%)
8590
Authored support (%)
3035Qwen3.5-4B
6080
10203040
0102030
808590
2040Llama-3.1-8B
152025
102030
051015
8590
0 .5 1304050Llama-3.2-3B
0 .5 122242628
0 .5 1
training progress102030
0 .5 146
0 .5 1848688
GRPO + KL Copy penalty ASCFigure 7: All four models on ASQA over training (0: base model): GRPO + KL, ASC , and the copy penalty where
its intermediate checkpoints were evaluated.
checks that need no judge preference: reference
coverage, refusals and answerable questions.
Protocol.The judge is Gemma-4-31B, the judge
of the authored error rate. It sees the question, the
passages and two answers and returns the answer
that serves a reader better, or a tie, with a one-
sentence reason (prompt in Appendix H). Every
pair is judged in both orders; a pair is a win only if
both orders agree, any other outcome is a tie, and
a tie counts half in the win rate. We compare ASC
(first seed) with the base model and each baseline,
and each baseline with the base model, on up to
300 questions per setting, about 5,400 pairs each.
Choosing the rubric.LLM judges favor longer
answers and a fixed position (Zheng et al., 2023;
Wang et al., 2024), and a judge that favored quoted
or long text would rank methods by style. We
wrote three rubrics and measured each on probes
built from base-model answers only (Table 17): the
same content as a verbatim passage sentence orProbe(ideal value)Rubric
123
Copied form preferred (50) 44.8 47.2 45.9
Padded answer preferred (≤50) 3.8 15.5 11.5
Intact answer preferred (high) 99.3 99.3 99.7
Table 17: Probes (%) for the three candidate rubrics,
built from base-model answers only: two biases (copied
form, padding) and the sensitivity to a planted error.
Rubric 2 (bold) is the one used.
in the model’s own words (404 pairs), an answer
padded with a restated sentence (200 pairs), and an
answer with one supported sentence replaced by an
unsupported one (153 pairs), which a useful judge
must notice. The rule, fixed in advance, keeps the
rubrics that prefer the intact answer at least 80%
of the time, takes the one closest to 50% on the
copy probe, and breaks near-ties by the smaller
preference for padding. The chosen rubric states
that quoting is neither a merit nor a fault and that

length is no merit, and counts a refusal as right
only when the passages lack the answer; it was
fixed, and its hash recorded, before any answer of
a trained model was judged.
0 5 10 15
Δ reward (points)304050win rate against base (%)(a) Answer quality
0 5 10 15
Δ reward (points)−0.50.00.51.0Δ supported sentences / answer(b) ContentBase model GRPO + KL Copy penalty Sentence reward ASC
Figure 8: Gain in the training reward over the base
model against (a) the win rate against the base model
and (b) the change in judge-supported authored sen-
tences per answer. Markers: mean over the 20 settings
with 95% intervals; light points: ASC in each setting.
Every method raises the reward, but only ASC keeps
the base model’s quality and adds content.
Preference.Table 18 gives every setting. Signifi-
cant verdicts favor ASC more often than the other
method in every comparison: over the base model
in 5 settings (the base model in 2), and over the
copy penalty, GRPO + KL and the sentence reward
in 14, 9 and 3 (each of them in at most 2). Figure 8
adds the spread over the 20 settings to Table 3, and
the change in supported authored sentences: only
ASC adds supported content. Controlled for length,
by a logistic fit of the verdict on the difference
in words, its win rates are 48.9% against the base
model, 50.1% against GRPO + KL, 54.5% against
the copy penalty and 49.2% against the sentence
reward; this control also removes the credit for the
extra supported sentences, which come with more
words, so it likely understates ASC ’s win rates. A
second seed on Qwen3.5-9B gives the same order-
ing (52.1% and 56.4% against the base model and
GRPO + KL).
Second judge.Mistral-Small-3.2-24B-Instruct,
from a third model family, judges the same pairs
with the same rubric and ranks the methods as
Gemma does (Table 3), although its copy probe
shows a bias against ASC (it prefers the copied
form in 62.9% of the pairs). On the pairs that
both judges decide, 45% of all pairs, they agree on
81.7% (κ0.61).
Reference coverage.ASQA, ClapNQ and ELI5
come with reference answers; the judge splits each
reference into key points and checks which ones an
answer conveys. ASC covers the most key pointson average (Table 3), significantly more than the
base model in 4 of the 12 settings and less in none.
The NLI model TRUE (Honovich et al., 2022),
which is not an LLM judge, agrees with Gemma
on 90.7% of the key points ( κ0.80) and gives the
same direction (+1.3 points).
Refusals.Refusing is right when the passages lack
the answer, and the base model mostly refuses for
that reason: the judge finds that the passages do
not answer 85.4% of its 575 refusals. GRPO + KL,
the sentence reward and ASC also refuse mostly on
such questions (79.0%, 93.7% and 88.8% of their
refusals), but the copy penalty does not: 47.2%
of its 3,821 refusals fall on questions the passages
answer fully or in part. ASC answers most of the
questions the base model rightly refuses, with an
authored error rate of 5.5%, and there the judge
prefers the refusal ( ASC wins 32.3% of 181 pairs);
where both models answer, ASC wins 51.5%.
Answerable questions.On the questions whose
passages contain the answer (labeled by the judge;
Mistral agrees on 84% of 300 sampled questions),
ASC has a lower error rate than the base model in
19 of the 20 settings and writes significantly more
supported authored sentences in all 20 (Table 19).
F Ablation Details
Each ablation of Section 6.3 changes one part of
ASC on Qwen3.5-9B and keeps every other setting
of Appendix B; the answer-advantage variant sums
the graded credit over the answer. Table 20 gives
every corpus. The vacuity test matters where vac-
uous sentences are common: without it, Err rises
on ASQA, ELI5 and ExpertQA, the three corpora
where the base model writes the most vacuous sen-
tences, and falls slightly on RAGTruth and ClapNQ.
Removing the copy test or the graded credit, or us-
ing the answer-level advantage, raises Err or brings
copying back on at least four of the five corpora.
G Human Evaluation
Two graduate-level annotators experienced in the
field label sentences and compare answers, blind
to the method that wrote them.
Sentence labels.The annotators label authored
sentences as supported or unsupported by the pas-
sages. On a judge-validity set drawn from base-
model and trained answers on ASQA and ELI5,
the judge agrees with people at least as often on the
sentences of ASC (90.8%) as on the base model’s

Model CorpusWin rate of ASC against Controlled for length Coverage
Base GRPO + KL Copy pen. Sent. rew. Base GRPO + KL Copy pen. Sent. rew. Base ASC
Qwen3.5-9B ASQA45.552.072.250.743.9 43.957.445.250.9 52.0
RAGTruth 52.3 55.088.352.7 52.5 52.6 61.1 50.9 – –
ClapNQ54.253.078.352.7 52.7 51.459.150.7 89.593.5
ELI5 51.2 51.776.553.2 47.443.655.9 47.4 14.4 14.9
ExpertQA 50.559.5 83.8 54.845.048.8 51.7 48.8 – –
Qwen3.5-4B ASQA 49.561.8 59.748.8 49.4 50.9 54.2 48.5 50.9 51.2
RAGTruth 54.365.052.0 53.7 52.3 55.0 50.4 52.8 – –
ClapNQ 51.062.752.8 52.0 49.2 51.3 50.2 51.1 91.092.6
ELI541.362.744.0 42.7 39.453.642.5 42.114.1 14.9
ExpertQA 46.772.2 57.249.345.753.0 51.1 50.3 – –
Llama-3.1-8B ASQA 53.058.0 69.4 57.647.7 50.8 51.8 53.2 37.9 38.3
RAGTruth 48.7 49.061.448.3 43.7 43.2 50.541.8– –
ClapNQ58.0 57.3 72.852.3 47.5 50.6 54.8 46.7 63.671.2
ELI5 49.2 49.8 51.244.749.2 48.4 48.343.213.1 13.1
ExpertQA 48.2 49.560.751.5 45.4 47.756.848.1 – –
Llama-3.2-3B ASQA 52.0 53.367.9 57.550.8 51.0 53.655.136.6 37.6
RAGTruth 50.7 51.0 55.7 54.7 52.4 48.3 48.4 54.6 – –
ClapNQ55.2 56.2 73.849.5 51.4 51.9 56.3 48.5 65.270.6
ELI554.353.2 54.3 53.2 54.3 52.1 50.1 52.0 12.7 12.0
ExpertQA54.853.259.552.3 52.6 52.256.651.0 – –
Table 18: Win rate (%) of ASC against the base model and each baseline in every setting under Gemma-4-31B, as
judged and controlled for the difference in words, and reference coverage (%) of the base model and ASC (–: no
reference answers). Bold: the 95% bootstrap interval over questions lies above 50 ( ASC preferred), or for coverage,
ASC’s difference from the base model lies above zero; italic: below. ASC: first seed.
Corpus QuestionsErr Supp.
Base ASC Base ASC
Qwen3.5-9B
ASQA 663 3.9 3.3 1.54 2.22∗
RAGT. 91 1.2 1.3 3.16 3.96∗
ClapNQ 235 3.1 2.4 1.83 2.57∗
ELI5 248 3.7 2.5∗3.08 3.53∗
Expert 363 5.2 3.5∗3.12 3.82∗
Qwen3.5-4B
ASQA 214 8.5 5.2∗1.67 2.37∗
RAGT. 91 2.6 1.6 3.07 3.75∗
ClapNQ 235 5.2 3.6 1.87 2.65∗
ELI5 70 7.3 4.3∗2.74 3.38∗
Expert 86 6.3 4.6 3.05 3.66∗
Llama-3.1-8B
ASQA 663 7.1 3.1∗0.46 0.78∗
RAGT. 91 3.9 1.2 1.71 2.42∗
ClapNQ 235 9.7 5.2∗0.75 1.35∗
ELI5 248 10.6 5.8∗1.56 1.94∗
Expert 363 10.3 6.9∗1.25 1.78∗
Llama-3.2-3B
ASQA 214 16.0 10.4 0.43 0.75∗
RAGT. 91 8.4 4.2 1.13 2.11∗
ClapNQ 235 10.2 6.9 0.56 1.26∗
ELI5 70 17.2 9.8 1.41 1.93∗
Expert 86 15.5 11.8 1.07 1.73∗
Table 19: Base model against ASC on the questions
whose passages contain the answer: questions, authored
error rate (Err) and supported authored sentences per
answer (Supp.).∗: a significant difference from the base
model (bootstrap over questions, Appendix D).(85.5%), in both model families. In a direct test
with no judge, one random authored sentence from
the base model and one from ASC for each of
250 ELI5 questions answered by Llama-3.1-8B,
24.8% of the sentences of ASC are unsupported
against 33.2% of the base model’s (exact McNemar
p= 0.01 ). On 150 shared sentences, the annotators
agree with a Cohen’s kappa of 0.65.
Answer preference.The annotators compare 150
pairs of answers, 75 of ASC against the base model
and 75 against GRPO + KL, drawn with a fixed
seed across the four models and five corpora; the
order of the two answers is random, and they follow
the rubric of Appendix E. Counting a tie as half a
win, they prefer ASC to the base model in 56.7% of
the pairs and to GRPO + KL in 59.3%. They agree
with a Cohen’s kappa of 0.66, and on the pairs
where both decide, their verdict matches Gemma’s
in 89.8% and Mistral’s in 88.9%.
H Prompts
Figure 9 gives every prompt verbatim. All models
answer with the same prompt in one user turn, and
thinking is disabled on the Qwen models. The sen-
tence judge’s verdict is read as its probability of
answering “Yes”; the pairwise rubric was fixed be-
fore any trained answer was judged (Appendix E).

MethodASQA RAGTruth ClapNQ ELI5 ExpertQA
Err Copy Vac. Supp. Err Copy Vac. Supp. Err Copy Vac. Supp. Err Copy Vac. Supp. Err Copy Vac. Supp.
Qwen3.5-9B
ASC 4.0 28.6 10.9 2.3 1.1 22.7 12.6 3.8 2.2 28.4 13.7 2.5 3.4 12.9 14.1 3.2 3.7 18.3 15.2 3.4
−copy test 2.4 46.3 11.1 1.4 1.5 31.8 10.0 3.4 1.6 42.2 11.5 1.9 3.5 19.1 10.4 3.1 3.5 26.9 12.7 3.2
−vacuity test 5.2 26.9 17.2 2.01.024.0 15.9 3.71.626.7 17.8 2.4 4.6 11.3 23.4 3.0 7.5 17.8 21.4 3.3
Sentence reward 3.7 42.3 22.8 1.5 0.9 34.0 15.9 2.9 3.1 42.2 19.0 1.7 3.5 14.5 35.9 2.4 4.3 21.3 34.8 2.5
Answer advantage 5.4 33.1 15.2 2.8 2.3 26.3 12.5 4.4 1.9 33.4 15.2 2.9 5.1 13.0 24.0 3.2 5.1 19.9 24.6 3.3
Binary credit 4.6 26.6 10.1 2.3 1.7 21.7 12.0 3.8 2.5 26.6 12.9 2.5 4.1 12.3 13.7 2.9 4.4 17.4 15.2 3.3
Table 20: Ablations on Qwen3.5-9B on every corpus: authored error rate (Err; grey, bold and underlined as in
Table 4), copy and vacuous share, and supported authored sentences per answer (Supp.).
I Example Answers
Examples 1–4 show one question for each model,
two from training corpora and two from held-out
ones, answered by all five policies of Table 2. Each
example is chosen to be representative of its set-
ting. Among the questions whose passages and five
answers fit in one box, and which the base model
answers without declining any part, we take the
one whose answers best show each policy’s typ-
ical behavior in that setting (Appendix D): ASC
copies no more than the base model and writes
more grounded sentences, as it does at the median,
and as many baselines as one question allows take
their most frequent shortcut there. Each box gives
the question, every passage as the model saw it,
and the five answers. In the passages, highlighted
text is repeated verbatim, five words or more, by at
least one answer, so the copied material is visible at
a glance. In the answers, every sentence is marked
as copied C, authored and supported A, authored and
unsupported Uorvacuous V(supported or not under
the judge; vacuous under the reward’s checker),
and ASC’s answer is shaded.
In Example 1, the base model copies its first two
sentences and adds an unsupported claim about the
Qt toolkit; GRPO + KL keeps the two copied sen-
tences and the sentence reward copies a third, and
the copy penalty declines the question, as it does
on 95.4% of ELI5 questions for Qwen3.5-9B. ASC
conveys the same content in its own words, in four
grounded sentences, with no error. In Example 2,
GRPO + KL and the sentence reward copy two
and three of their three sentences, the copy penalty
ends on a vacuous one, and ASC copies one of
its four sentences and grounds the other three. In
Example 3, the base model overstates the passage
(California taxes no lottery winnings at all), GRPO
+ KL and the copy penalty decline a question the
passage answers, the sentence reward opens with
a copied sentence, and ASC writes four groundedsentences, including the federal withholding rules.
In Example 4, GRPO + KL and the sentence reward
shrink the answer to one copied sentence and the
copy penalty to a vacuous one; ASC writes three
grounded sentences.

Answer prompt (all models, one user turn)
You are answering a question using only the passages provided below.
Write a complete, self-contained answer in your own words, as connected prose. Use only information that is stated in the
passages; do not add facts, examples, or explanations from anywhere else. If the passages do not contain the answer, say
that they do not.
Do not refer to the passages by number, do not write citation markers such as [1], and do not restate the question or add a
preamble. Give the answer and nothing else.
{passages}
{question}
Sentence judge behind Err (Gemma-4-31B)
Below are a source document and a single statement taken from a response written about it. Judge only whether the statement
is supported by the source document: every fact it asserts must be stated in the document or follow directly from it.
Answer No if the statement contradicts the document, or if it adds any detail the document does not contain—even a
plausible one. Do not use outside knowledge, and do not judge whether the statement is well written or useful.
### Source document
{passages}
### Statement
{sentence}
Is the statement fully supported by the source document? Answer Yes or No.
Pairwise judge (Gemma-4-31B and Mistral-Small-3.2-24B)
You are comparing two answers to the same question. Both were written from the passages below, which are the only
information the writer had. Decide which answer serves a reader better.
First check the content. Go through the claims of each answer and compare each one with the passages: is it stated in the
passages or does it follow directly from them? A claim that adds a detail the passages do not contain, or that contradicts
them, is an error, and errors weigh heavily. Then ask which answer gives the reader more of what the question asks for,
correctly. Then, only to separate answers that are otherwise equal, prefer the clearer one.
Judge what the claims say, not how they are worded. Many accurate answers restate the passages in new words; others quote
them. Neither form is better: do not reward an answer for sounding close to the passages, and do not reward it for sounding
original. Do not reward length by itself; a repeated or padded point adds nothing. If the passages do contain the answer,
declining to answer is a failure; if they do not, saying so is correct. Ignore the order in which the answers appear. Choose
“tie” if neither answer is better.
### Question
{question}
### Passages
{passages}
### Answer A
{answer_a}
### Answer B
{answer_b}
Reply with one JSON object and nothing else: {"reason": "<one sentence>", "verdict": "A" | "B" | "tie"}
Figure 9: The three prompts, verbatim: the answer prompt shared by all models (top), the sentence judge behind the
authored error rate (middle), and the pairwise judge of Appendix E (bottom).

Example 1: Qwen3.5-9B on ELI5 (held out)
Question: What is Linux, what does it do, & why is it so popular?
Passages
[1]Top 50 Most Asked Linux Interview Questions & Answers in 2019.
Q-25: What is “PR” Command in Linux? Q-26: What is “inode” in
Linux? Q-27: What is the use of “man” command in Linux? Q-28:
What Does “pipe” Do in Linux? Q-29: What Does “mkdir” Do in
Linux? Q-30: What Does “ls” Do in Linux? Q-31: What Does “TR”
command Do in Linux? Q-32: What are the Linux “Logs”? Q-33: How
to Check “Logs” in Linux? Q-34: How to Play Games on Linux? Q-35:
What is “zsh” in Linux? Q-36: Why “zsh” Should Be Used Instead of
Bash? Q-37: Why “top” Command is Used in Linux? Q-38: Which
“File System” is
[2]What is Linux? |.What is Linux and why is it so popular? Browse
the article What is Linux and why is it so popular? Every desktop com-
puter uses an operating system. The most popular operating systems in
use today are: Linux isanoperatingsystem — very much like UNIX
— that has become very popularover thelastseveralyears. Operating
systems are computer programs. An operating system isthefirstpiece
ofsoftware that the computer executes when you turn the machine on.
The operating system loads itself intomem oryandbegins managing
theresources avail able onthecomputer. It then
[3]Who Made Linux and Why Is It Free?.available on Unix. It used
the Qt toolkit and was the first complete desktop environment for Linux.
The Qt toolkit was under a proprietary license, so it wasn’t fully free
software. This left other developers wanting an alternative. Miguel de
Icaza and Federico Mena founded GNOME GNOME Explained: A
Look at One of Linux’s Most Popular Desktops GNOME Explained:
A Look at One of Linux’s Most Popular Desktops You’re interested
in Linux, and you’ve come across “GNOME”, an acronym for GNU
Network Object Model Environment. GNOME is one of the most
popular open source interfaces, but what does that mean?
[4]Rute User’s Tutorial and Exposition. D. LINUX Advocacy FAQ.
different distributions? What is the best distribution to use? Where do I
get LINUX? D.4 LINUX Support Where does a person get LINUX sup-
port? My purchased software is supported; how does LINUX compete?
D.5 LINUX Compared to Other Systems What is the most popular
UNIX in the world? How many LINUX systems are there out there?
What is the total cost of installing and running LINUX compared to a
proprietary non-UNIX system? What is the total cost of installing and
running a LINUX system compared to a proprietary UNIX system?
How does LINUX compare to other operating systems in performance?
[5]Windows or Linux Hosting? Does it matter? - Small Planet Online,
LLC.The only real consideration is what type of technology your site
is being developed in. For the record, Linux is almost unheard of at
home, but isbyfarthemost popularoptionforweb hosting. It’s secure,
fast, and reliable. We normally steer our client towards open source
solutions likeLinux and PHP, because they work well and you will be
in very very good company. Here’s a brief article from a few years ago
that discusses why Linux is so darn popular for web hosting.Answers
Base model:Linux is an operating system that is very much
like UNIX and has become increasingly popular over the last
several years. CAs an operating system, it is the first piece of
software a computer executes when turned on; it loads into memory
and begins managing the resources available on the computer. C
Linux is popular for several reasons: it is secure, fast, and reliable,
making it the most popular option for web hosting. AAdditionally,
it is an open-source solution that allows developers to work well
together, and it serves as a free alternative to proprietary software,
such as the Qt toolkit which was not fully free. U
GRPO + KL:Linux is an operating system that is very much like
UNIX and has become increasingly popular over the last several
years. CAs an operating system, it is the first piece of software
a computer executes when turned on; it loads into memory and
begins managing the resources available on the machine. CLinux is
popular for several reasons: it is secure, fast, and reliable, making
it the most popular option for web hosting despite being rarely
used in home environments. AAdditionally, it is favored because it
utilizes open source solutions that work well, placing users in very
good company. A
Copy penalty:The provided passages do not contain the answer
to this question. V
Sentence reward:Linux is an operating system that is very much
like UNIX and has become increasingly popular over the last
several years. CAs an operating system, it is the first piece of
software a computer executes when turned on; it loads into memory
and begins managing the resources available on the machine. C
Linux is popular for web hosting because it is secure, fast, and
reliable, and it is considered a secure, fast, and reliable option that
is by far the most popular choice for this purpose. AAdditionally,
developers often prefer open source solutions like Linux because
they work well and are widely used. C
ASC :Linux is an operating system that is very similar to UNIX
and has gained significant popularity over the last several years. A
As an operating system, it is the first software executed when a
computer is turned on; it loads into memory and manages the
available resources on the machine. ALinux is widely used for
web hosting, where it is considered the most popular option
because it is secure, fast, and reliable. AIts popularity is further
supported by the fact that it is an open-source solution that works
well with technologies like PHP, placing users in good company
with many others who utilize it. A

Example 2: Llama-3.1-8B on RAGTruth (in domain)
Question: symptoms and causes of ibs
Passages
[1]The cause of irritable bowel syndrome
is currently unknown. Itisthought toresult
from acombinationof abnormal gastroin-
testinal (GI) tract move ments, increased
aware ness ofbodilyfunctions, and a dis-
ruption in the communicationbetween the
brain andtheGItract. IBS-D is irritable
bowel syndrome with diarrhea.
[2]The exact cause of irritable bowel syn-
drome is unknown. It is believed tobedue
toanumberoffactors, including alteration
in the gastrointestinal (GI) tract motil ity,ab-
normalnervous systemsignals, increased
sensitivitytopain, andfood intolerances.
The following are risk factors thought to
cause IBS: 1 Abnormalmove ments ofthe
colon andsmall intestines (too fast or slow,
or too strong) 2 Hypersensitivity to pain
from a full bowel or gas. 3 Food sensitivi-
ties, possibly caused by poor absorption of
sugars or acids in food.
[3]People with IBS have symptoms that
can include: 1 Diarrhea (often described
as violent episodes of diarrhea) 2 Consti-
pation. 3 Constipationalternatingwith di-
arrhea. Belly pains orcramps, usually in
the lower half of the belly, that get worse
after meals and feel better after a bowel 1
movement. Alotofgasorbloat ing.Answers
Base model:Irritable bowel syndrome, or IBS, is a condition with unknown causes,
but it is thought to result from a combination of factors. AThese factors may include
abnormal movements of the colon and small intestines, hypersensitivity to pain, and
food intolerances. CPeople with IBS can experience a range of symptoms, including
diarrhea, constipation, or alternating episodes of both, as well as belly pains or cramps,
especially after eating, and a lot of gas or bloating. A
GRPO + KL:Irritable bowel syndrome, or IBS, is a condition with unknown causes,
thought to result from a combination of factors such as abnormal gastrointestinal move-
ments, increased awareness of bodily functions, and disrupted communication between
the brain and the GI tract. CIt is also believed to be due to factors like altered GI tract
motility, abnormal nervous system signals, increased sensitivity to pain, and food intol-
erances. CPeople with IBS may experience symptoms including diarrhea, constipation,
or alternating episodes of both, accompanied by belly pains or cramps, especially after
meals, and a lot of gas or bloating. A
Copy penalty:Irritable bowel syndrome, or IBS, is a condition with unknown causes, but
it is thought to be the result of a combination of factors, including abnormal movements
of the GI tract, increased sensitivity to bodily functions, and disrupted communication
between the brain and the GI tract. ASymptoms of IBS can include diarrhea, constipation,
or alternating between the two, as well as belly pains or cramps, gas, and bloating. V
Sentence reward:Irritable bowel syndrome, or IBS, is a condition with unknown
causes, but it is thought to result from a combination of factors including abnormal
gastrointestinal tract movements, increased awareness of bodily functions, and disrupted
communication between the brain and the GI tract. CThe exact cause is also believed to
be due to a number of factors, including alterations in the GI tract motility, abnormal
nervous system signals, increased sensitivity to pain, and food intolerances. CSymptoms
of IBS can include diarrhea, constipation, constipation alternating with diarrhea, belly
pains or cramps, and a lot of gas or bloating. C
ASC :Irritable bowel syndrome, or IBS, is a condition with unknown causes, but it
is thought to be the result of a combination of factors. AThese factors may include
abnormal movements of the gastrointestinal tract, increased awareness of bodily
functions, and disruptions in communication between the brain and the GI tract. CIBS
can also be caused by factors such as abnormal nervous system signals, hypersensitivity
to pain, and food intolerances. ASymptoms of IBS can vary, but they often include
diarrhea, constipation, or a combination of both, as well as belly pains or cramps, gas,
and bloating. A

Example 3: Qwen3.5-4B on ClapNQ (held out)
Question: what is the tax rate for lottery winnings in california
Passages
[1]California State Lottery.All prizes for Fantasy5,Daily
Derby, Daily 3,Daily 4,andnon-jackpotSuperLotto Plus,
Mega Millions, andPower ball prizes, are paid out in one
payment, less 25 % or 33 % (depending upon thewinner’s
taxdocumentation) Federal withholding if the prize is over
$5,000. Merchan dise prizes over $5,000 aresubjectto 33
% Federal withholding. Scratchers tickets are generally
one - payment prizes; however, some games have annuity
options for payments each year, or per week. California
does nottaxCaliforniaLotterywinnings, however it taxes
lotterywinnings from other jurisdictions. For SuperLotto
Plus andMega Millions jackpots, the player may choose a
single cash payout for a floating percentage of the jackpot,
or an annuity. The SuperLotto Plus, Mega Millions, and
Power ball payment schedule are on agraduated basisover
30annual payments. Until 2005, when California joined
Mega Millions, the payment choice on SuperLotto Plus had
to be made when the ticket was bought. Since then, there
is a 60 - day window after winning, in which the choice of
cash or annuity is to be made.Answers
Base model:California does not tax lottery winnings. UHowever, the
state does tax lottery winnings from other jurisdictions. CAdditionally,
federal withholding applies to prizes over $5,000, with rates of 25% or 33%
depending on the winner’s tax documentation, and merchandise prizes over
$5,000 are subject to a 33% federal withholding rate. A
GRPO + KL:The provided passages do not contain the answer to this
question. V
Copy penalty:The provided passages do not contain the answer to what
the specific tax rate is for lottery winnings in California. V
Sentence reward:The provided passages indicate that California does not
tax California Lottery winnings. CHowever, the text notes that California
does tax lottery winnings from other jurisdictions. A
ASC :California does not impose a tax on lottery winnings won within
the state, although it does tax winnings obtained from other jurisdictions. A
Regarding federal withholding, prizes exceeding $5,000 for games such
as Fantasy 5, Daily Derby, Daily 3, Daily 4, and non-jackpot SuperLotto
Plus, Mega Millions, and Powerball are subject to a federal withholding
rate of either 25% or 33%, depending on the winner’s tax documentation. A
Merchandise prizes over $5,000 are specifically subject to a 33% federal
withholding rate. AFor the jackpots of SuperLotto Plus and Mega Millions,
players may elect a single cash payout or an annuity option, with the
payment schedule for these games, as well as Powerball, following a
graduated basis over 30 annual payments. A
Example 4: Llama-3.2-3B on ASQA (in domain)
Question: What kind of series is game of thrones?
Passages
[1]A Game of Thrones.A Game of Thrones A Game of Thrones is the first novel in “A Song of Ice
and Fire”, aseriesoffantasynovelsby American authorGeorge R.R.Martin. It was first published
on August 1, 1996. The novel won the 1997 Locus Award and was nominated for both the 1997
Nebula Award and the 1997 World Fantasy Award. The novella “Blood of the Dragon”, comprising the
Daenerys Targaryen chapters from the novel, won the 1997 Hugo Award for Best Novella. In January
2011 the novel became a “New York Times” Bestseller and reached #1 on the list
[2]Game of Thrones.in a fantasy setting of magic and dragons. In a 2012 study of deaths per episode,
it ranked second out of 40 recent U.S. TV drama series (with an average of 14). The series is generally
praised for what is perceived as a sort of medieval realism. George R.R. Martin set out to make the
story feel more like historical fiction than contemporary fantasy, with less emphasis on magic and
sorcery and more onbattles, politicalintrigue, and the characters, believing that magic should be used
moderately in the epic fantasy genre. Martin has stated that “the true horrors of
[3]Game of Thrones (season 1).The majority of reviews for the first season were very positive, with
critics noting the high production values, the well-realized world, compelling characters, and giving
particular note to the strength of the child actors. Tim Goodman’s review for “The Hollywood Reporter”
stated, “a few minutes into HBO’s epic Game of Thrones series, it’s clear that the hype was right
and the wait was worth it”. Mary McNamara from the “Los Angeles Times” called it “...a great and
thundering series of political and psychological intrigue bristling with vivid characters, cross-hatched
with tantalizing plotlines and seasoned with a splash of fantasy”. “New
[4]Game of Thrones.in the pilot, playing a character described as “a charismatic socialite hiding
a dark secret.” Game of Thrones Game of Thrones is an American fantasy drama television series
created by David Benioff and D. B. Weiss. It is an adaptation of “A Song of Ice and Fire”, George R.
R. Martin’s series of fantasy novels, the first of which is “A Game of Thrones”. It is filmed in Belfast
and elsewhere in Northern Ireland, Canada, Croatia, Iceland, Malta, Morocco, Scotland, Spain, and the
United States. The series premiered on HBO in the United States on April 17, 2011, and its
[5]A Game of Thrones (board game).A Game of Thrones (board game) A Game ofThrones isa
strategy board game created by Christian T. Petersen and released by Fantasy Flight Games in 2003.
The game is based on the “A Song of Ice and Fire” fantasy series byGeorge R.R.Martin. It was
followed in 2004 by the expansion “A Clash of Kings”, and in 2006 by the expansion “A Storm of
Swords”. “A Game of Thrones” allows the players to take on the roles of several of the Great Houses
vying for control of the Seven Kingdoms, including House Stark, House Lannister, HouseAnswers
Base model:A fantasy series set in a
medieval-like world with magic and drag-
ons, with a focus on battles, political in-
trigue, and character development, and an
adaptation of a series of fantasy novels by
George R.R. Martin. A
GRPO + KL:Game of Thrones is a fan-
tasy series. C
Copy penalty:The series is a fantasy
series. V
Sentence reward:Game of Thrones is a
fantasy series. C
ASC :The series is a fantasy series with
elements of historical fiction, and it is
set in a medieval-like world with magic
and dragons. AThe author, George R.R.
Martin, aimed to create a more realistic
story with a focus on battles, politics,
and character development, rather than
relying heavily on magic. AThe series
has been adapted into a television show,
which was praised for its production val-
ues, characters, and plotlines. A
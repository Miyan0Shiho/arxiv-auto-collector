# AWM: Answerable Working Memory for Long-Document VQA Agents

**Authors**: Dongzhuoran Zhou, Yuqicheng Zhu, Yule Liu, Zhen Yang, Rui Lu, Yuxiao Dong, Jie Tang, Evgeny Kharlamov

**Published**: 2026-08-26 10:35:25

**PDF URL**: [https://arxiv.org/pdf/2608.25618v1](https://arxiv.org/pdf/2608.25618v1)

## Abstract
Long-document visual question answering increasingly relies on VLM agents that retrieve candidate pages, inspect page images, write findings to working memory, and synthesize answers. Working memory should carry answer-supporting evidence across page inspections for later grounded answering, yet existing evaluation mainly checks final-answer correctness and evidence-page access. This creates a memory-quality blind spot: an agent may reach the right page and answer correctly while leaving behind memory too generic or incomplete to support answering once page context is removed. We introduce \emph{memory-only answerability}, a diagnostic that asks whether a reader can answer from the question and terminal working memory alone. Building on this diagnostic, \emph{Answerable Working Memory} (AWM) treats terminal working memory as an answerable evidence artifact, and AWM-GRPO incorporates this signal into the GRPO reward while preserving final-answer priority. Under GRPO, this reward assigns higher advantages to answer-correct trajectories whose terminal working memory remains answerable. On \textsc{MMLongBench-Doc}, even when gold evidence pages are provided, 42.5\% of correct answers still cannot be answered from terminal working memory alone. AWM-GRPO improves final-answer accuracy over the RAG baseline by 8.1 and 11.9 points on \textsc{MMLongBench-Doc} and \textsc{LongDocURL} and reduces the memory-missing-correct rate by 2.7 points over answer-only GRPO.

## Full Text


<!-- PDF content starts -->

A WM: Answerable Working Memory for Long-Document VQA Agents
Dongzhuoran Zhou1,5,*, Yuqicheng Zhu2,5, Yule Liu3,*, Zhen Yang4,
Rui Lu4,Yuxiao Dong4,Jie Tang4,Evgeny Kharlamov1,5
1University of Oslo,2University of Stuttgart
3Hong Kong University of Science and Technology (Guangzhou)
4Tsinghua University,5Bosch Center for AI
Abstract
Long-document visual question answering in-
creasingly relies on VLM agents that retrieve
candidate pages, inspect page images, write
findings to working memory, and synthesize an-
swers. Working memory should carry answer-
supporting evidence across page inspections
for later grounded answering, yet existing eval-
uation mainly checks final-answer correctness
and evidence-page access. This creates a
memory-quality blind spot: an agent may reach
the right page and answer correctly while leav-
ing behind memory too generic or incomplete
to support answering once page context is re-
moved. We introducememory-only answerabil-
ity, a diagnostic that asks whether a reader can
answer from the question and terminal working
memory alone. Building on this diagnostic,An-
swerable Working Memory(AWM) treats termi-
nal working memory as an answerable evidence
artifact, and AWM-GRPO incorporates this sig-
nal into the GRPO reward while preserving
final-answer priority. Under GRPO, this reward
assigns higher advantages to answer-correct
trajectories whose terminal working memory
remains answerable. On MMLONGBENCH-
DOC, even when gold evidence pages are pro-
vided, 42.5% of correct answers still cannot
be answered from terminal working memory
alone. AWM-GRPO improves final-answer ac-
curacy over the RAG baseline by 8.1 and 11.9
points on MMLONGBENCH-DOCand LONG-
DOCURL and reduces the memory-missing-
correct rate by 2.7 points over answer-only
GRPO.
1 Introduction
Answering questions over long documents resem-
bles goal-directed information seeking: a reader
searches for relevant evidence, keeps intermedi-
ate findings available, and integrates them into an
*Corresponding authors.
Code is available at github.com/DongzhuoranZhou/AWManswer. Recent long-document VQA systems in-
stantiate a similar loop with VLM agents: they
retrieve candidate pages, inspect page images, up-
date working memory with findings, and synthe-
size a final answer (Faysse et al., 2025; Yu et al.,
2025; Cho et al., 2024; Zheng et al., 2026; Lin
et al., 2026; Lim et al., 2026). Working memory
should carry answer-supporting evidence across
page inspections for later grounded answering. Yet
current evaluation checks final-answer correctness
and evidence-page access, but not whether terminal
working memory remains answer-supporting. An
agent may therefore answer correctly from page
context while leaving behind memory too generic
to support the answer on its own.
We introducememory-only answerability: a
reader receives only the question and the agent-
written terminal working memory, without page
images or trajectory context, and must answer from
that memory alone. This exposes a memory-quality
gap: even when gold evidence pages are supplied,
42.5% of correctly answered MMLONGBENCH-
DOCexamples fail memory-only answerability. A
controlled memory comparison further shows that
improving terminal working memory improves an-
swering, motivating optimization of the memory
artifact itself.
Motivated by this observation, we propose
AWM-GRPO. For each trajectory, the agent out-
puts a final answer and terminal working memory.
A frozen reader answers from the question and ter-
minal working memory alone, and a judge scores
both the final answer and the memory-only answer.
Under GRPO, this reward ranks trajectories by both
final-answer correctness and memory answerabil-
ity, creating a preference for trajectories that an-
swer correctly with answerable terminal working
memory.
We evaluate a controlled training pipeline in
which the three AWM-Agent variants share the
same Qwen3-VL-4B policy, RAG Top-3 retrieval
arXiv:2608.25618v1  [cs.CL]  26 Aug 2026

harness, tools, memory schema, prompts, and
optimizer; only post-training and reward dif-
fer. We compare direct input, RAG Top-3,
SFT, answer-only GRPO, and AWM-GRPO. On
MMLONGBENCH-DOC/LONGDOCURL, AWM-
GRPO exceeds RAG Top-3 by 8.1/11.9 points,
SFT by 4.7/4.4 points, and answer-only GRPO by
2.3/2.7 points. It also reduces the memory-missing-
correct rate by 2.7 points over answer-only GRPO
on MMLONGBENCH-DOC(Sec. 4.3).
This paper makes three contributions. (1)
We identify a memory-quality gap in long-
document VQA agents: final-answer correct-
ness and evidence-page access do not deter-
mine whether terminal working memory preserves
answer-supporting evidence. (2) We propose
AWM-GRPO, which uses memory-only answer-
ability as a GRPO reward signal and ranks trajec-
tories by both final-answer correctness and mem-
ory answerability, creating a preference for answer-
correct trajectories with answerable terminal work-
ing memory. (3) We evaluate AWM-GRPO against
direct-input, RAG, SFT, and answer-only GRPO
baselines, showing improvements in both final-
answer accuracy and memory-quality diagnostics.
2 Agent Setup and Memory-Quality
Motivation
This section introduces the agent-memory frame-
work and the memory-only answerability metrics
used throughout the paper. It then presents two
diagnostics: current agent-written memory is often
insufficient even when page access is controlled,
and improving terminal working memory improves
answering.
Agent-memory framework.A long-document
VQA instance is a tuple (D, q, y) , where D=
{p1, . . . , p N}is a multi-page document, qis a ques-
tion, and yis the ground-truth answer; when avail-
able, E⊆D denotes gold evidence pages. The
policy πuses evidence-page retrieval and mem-
ory update. A retrieval call returns candidate page
images directly in the observation; after inspect-
ing these images, the agent must append a source-
linked finding to working memory Mt. The inter-
action proceeds until the agent outputs both a final
answer ˆyand terminal working memory Mterm, or
reaches a step budget (Figure 1a).
As illustrated in Figure 1a, memory entries are
source-linked findings whose quality depends on
whether they preserve question-relevant evidence.Final answer Memory answerable Case name
correct yes memory-supported correct
correct nomemory-missing correct
wrong yes answering error
wrong no unresolved error
Table 1: Outcome cases induced by final-answer cor-
rectness and memory-only answerability.
A generic description such as “the chart shows
trends” is less useful than a question-specific find-
ing such as “the 2020 self-sufficiency rate is 4.7%.”
Memory-only answerability.To measure
whether terminal working memory Mtermalone
supports answering, we evaluatememory-only
answerability: a frozen reader Rreceives only the
question qandMterm, without the trajectory or
page images, and produces a memory-only answer
R(q, Mterm). The official judge Jevaluates
both the agent’s final answer and the reader’s
memory-only answer against the gold answer y.
For the reported four-cell analysis, we map any
positive score from Jto1and zero to 0. This gives
two binary scores per instance: final-answer cor-
rectness sans=1[J(ˆy, y)>0] and memory-only
answerability smem=1[J(R(q, Mterm), y)>0] .
Implementation details for RandJappear in
Appendix B.
The pair (sans, smem)partitions instances into
four cases (Table 1). The focal case ismemory-
missing correct(MMC): sans= 1andsmem= 0,
i.e., the agent answered correctly but its terminal
working memory alone does not support answer-
ing.
We summarize each run with Pmmc =P
i1[s(i)
ans=1∧s(i)
mem=0]
P
i1[s(i)
ans=1], the fraction of final-answer-
correct instances whose terminal working mem-
ory is not answerable. Conditioning on sans=1
removes general answer failure as a confound and
asks whether memory remains insufficient even
after the agent answered correctly.
Current memory quality.We measure Pmmc
on a fixed 500-example answerable subset of
MMLONGBENCH-DOC(excluding the bench-
mark’s unanswerable questions). Thefull multi-
turnsetting runs the agent end-to-end with its
own retrieval and memory-update process. The
evidence-page-given(EP-given) setting replaces re-
trieval with gold evidence pages, so page access
is controlled and retrieval is no longer a confound.
Table 2 reports the results.

(a) Multi -turn agent -memory framework
page
retrieval
update
memoryq
ŷ
Mterm
append 
finding
Mterm
 = [ {source_page: pi, finding: "..."},  
{source_page: pj, finding: "..."},  ... ]
ŷ = final answer
Mterm
 = terminal working memory(b) Memory -quality diagnostics
(i) Standard evaluation
agent
 ŷ
 judge J
gold ysans
Mterm
(ii) Memory -only answerability
q, 
Mterm
 reader R
 ŷM
 judge J
gold ysmem
(iii) Controlled comparison
memory
gold 
pages
empty
original
improved
reader R
ŷempty
ŷorig
ŷimpFigure 1: Combined agent-memory framework and memory-quality diagnostics. (a) The agent retrieves page
images, writes source-linked findings to working memory, and outputs both a final answer and terminal working
memory. (b) Standard evaluation scores the final answer, memory-only evaluation asks whether the question can be
answered from terminal working memory alone, and the controlled comparison varies terminal working memory
while holding evidence-page access fixed.
SettingNFinal-ans.
acc. (%)Memory-only
acc. (%)MMC /
correctP mmc (%)
Full multi-turn 500 33.4 27.4 43 / 172 25.0
EP-given 500 36.2 30.2 77 / 181 42.5
Table 2: Memory-quality diagnostic for the Base AWM-
Agent (Qwen3-VL-4B, no post-training) on a fixed
500-example answerable MMLONGBENCH-DOCsub-
set. Unanswerable questions are excluded. Accu-
racy follows the official mean per-question score; for
MMC/correct and Pmmc, we binarize the official score
by treating a positive score as correct.
In the full multi-turn setting, one in four correct
trajectories ( Pmmc= 25.0% ) leaves behind termi-
nal working memory that cannot support answering
on its own. The EP-given setting controls for page
access by replacing retrieval with gold evidence
pages; Pmmc rises to 42.5% , showing that page
access and final-answer correctness do not imply
answerable terminal working memory.
Controlled memory comparison.We next ask
whether higher-quality memory improves answer-
ing. We construct a controlled setting in which
retrieval returns gold evidence pages and compare
three terminal-memory variants: (1)empty mem-
ory; (2)original agent memory; and (3)improved
memory, generated from the same gold pages by
a stronger model. To isolate memory quality, the
comparison keeps gold-page access and the eval-
uation protocol fixed while substituting different
terminal working memories. The answer gener-
ator, memory-only reader, and judge are shared
across conditions, so only terminal-memory con-Memory conditionFinal-ans.
binary rateMemory-only
binary rate
Empty memory 2.8 2.6
Original agent memory (4B) 36.2 30.2
Improved memory (GPT-4o) 44.4 44.8
Table 3: Controlled memory comparison on a fixed
500-example MMLONGBENCH-DOCEP-given answer-
able subset. The reader and scorer are fixed; only the
terminal-memory condition changes. Improved memory
is produced by GPT-4o.
tent changes (Figure 1b). Appendix D gives the
full setup, confidence intervals, outcome-cell break-
down, and artifact audit.
Table 3 isolates the role of terminal mem-
ory. With empty memory, both final-answer and
memory-only binary rates are near zero. Reusing
the 4B agent’s original memory yields final-answer
and memory-only binary rates of 36.2 and 30.2,
while GPT-4o-improved memory raises them to
44.4 and 44.8. Improving terminal memory alone
therefore improves both readouts with the answer
generator, reader, and judge held fixed.
Takeaway.Page access and final-answer correctness
do not guarantee answerable terminal working memory.
Memory-only answerability exposes this gap, and the
controlled comparison shows that improving terminal
working memory alone can improve answering.
3 Optimizing Answerable Working
Memory with GRPO
Sec. 2 identifies the failure mode that AWM-GRPO
targets: trajectories can answer correctly while

Rollout and memory-only scoring
policy 
πθrollout group
{τ1...τG}ŷ
Mterm
reader 
RŷM
gold 
y
judge Jsans
smemqAWM-
GRPO
policy updateFigure 2: Overview of AWM-GRPO. For each sampled trajectory, the agent produces both a final answer and
terminal working memory. A frozen reader answers from the terminal working memory alone, and a frozen
scorer evaluates the final answer and the memory-only answer. The two scores define the AWM reward, which is
normalized within each GRPO group to update the policy.
leaving terminal working memory unanswerable.
Since GRPO updates the policy through group-
relative advantages, the reward should make this
memory-quality distinction visible in the advantage
ranking. We first specify the desired advantage be-
havior, then instantiate a four-cell reward that real-
izes it, and finally use synthetic outcome mixtures
to show how the induced preferences change with
group composition.
3.1 Desired advantage behavior under GRPO
For each prompt, GRPO (Shao et al., 2024) sam-
plesGtrajectories and normalizes each trajectory’s
reward within the group:
Ai=r(τi)−µ g
σg+ϵ,
where µgandσgare the group mean and standard
deviation and ϵ= 10−6stabilizes zero-variance
groups. Because GRPO updates through these nor-
malized advantages, what matters is not the raw
reward scale but theorderingit induces: for any
two outcome cellsz, z′,
Az−Az′=rz−rz′
σg+ϵ.
The reward ordering therefore determines the ad-
vantage ordering. We design the AWM reward to
induce three pairwise preferences:
Memory refinement.Among answer-correct tra-
jectories, those whose terminal working memory
remains answerable should receive higher advan-
tage than those whose memory is not answerable.
Final-answer gate.Final-answer correctness re-
mains primary: an answer-correct but memory-
imperfect trajectory should still outrank an answer-wrong trajectory, even if the latter leaves answer-
able memory.
Useful-memory failure.An answer-wrong tra-
jectory with answerable memory is not success,
but it is less severe than complete failure, because
useful evidence was preserved.
GRPO gives positive advantage only to cells
whose reward exceeds the current group mean.
The same cell can therefore change role as group
composition changes: answer-correct but memory-
imperfect trajectories may be tolerated when most
samples fail, but become disfavored once memory-
supported-correct trajectories are common. For-
mally,A z>0iffr z> µg.
3.2 A WM reward instantiation
To realize these preferences with a minimal scalar
reward, we reuse the two binary variables from
Sec. 2: final-answer correctness sansand memory-
only answerability smem. During online training,
a frozen local Qwen3-14B model generates the
memory-only answer and scores both answers for
the reward. The official judge Jfrom Sec. 2 is
used for reported diagnostics. The AWM reward is
defined over the resulting four outcome cells:
r11=β, r 10= 0, r 01=γ, r 00=ω,
β >0> γ > ω.
We write RAWM(sans, smem) =r sanssmem; the
answer-only baseline uses Rans(sans) =s ans.
Here βis the reward for full success (memory-
supported correct), γis the reward for answer-
wrong but memory-answerable trajectories, and
ωis the reward for complete failure. The an-
chor r10= 0 fixes answer-correct but memory-
imperfect trajectories. The resulting normalized

margins are:
Preference Margin Ranking
Memory refinementA 11−A10=β/(σ g+ϵ) (1,1)≻(1,0)
Final-answer gateA 10−A01=−γ/(σ g+ϵ) (1,0)≻(0,1)
Useful-memory failureA 01−A00= (γ−ω)/(σ g+ϵ) (0,1)≻(0,0)
The memory-refinement margin is exactly what
answer-only reward cannot express: it scores (1,1)
and(1,0)identically, collapsingAans
11=Aans
10.
Each cell’s advantage sign depends on the group
mean (A z>0iffr z> µg):
Cell Positive iff Interpretation
(1,1)µ g< βfull success favored
(1,0)µ g<0flips atµ g=0
(0,1)µ g< γflips atµ g=γ
(0,0)never zero only ifµ g=ω
The main experiments instantiate β= 2 ,γ=
−0.1 , and ω=−1 , yielding (r11, r10, r01, r00) =
(2,0,−0.1,−1) . Under this default,
the wrong-answer but memory-answerable cell
(r01=γ=−0.1 ) is slightly below the answer-correct
/ memory-imperfect cell ( r10=0), so it becomes
negative first as the group mean rises. When
the group mean exceeds r10, the answer-correct /
memory-imperfect cell also becomes negative and
positive advantage concentrates on full success.
This gives the reward an answer-first, memory-
refinement-later tendency without implying a de-
terministic curriculum guarantee.
3.3 Advantage distributions under synthetic
outcome mixtures
Figure 3 uses a controlled Monte Carlo simulation
to show how the answer-only and AWM rewards
translate into GRPO advantages. For each row,
we fix a categorical distribution over the four out-
come cells of Table 1 and draw 15,000 independent
GRPO groups, each containing 8 trajectories. The
answer-only simulation samples from the corre-
sponding final-answer marginal, while the AWM
simulation samples the four cells directly. Rows
increase p11, the probability of memory-supported
correct (1,1) , so later rows contain more memory-
supported correct trajectories. The simulation uses
the default reward instantiation β= 2 ,γ=−0.1 ,
ω=−1 . It isolates reward-induced group-relative
preferences; curves are smoothed for visualiza-
tion only. Implementation details appear in Ap-
pendix C.
Three effects are visible:
•Answer-only hides memory quality: in the
left column, trajectories are grouped only by
final-answer correctness, so Ans ✓Mem✓and
012p11=0.08
012p11=0.30
2
 0 2
normalized advantage012p11=0.60
2
 0 2
normalized advantageAnswer-only GRPOAns× (reward 0)Ans (reward 1)
AWM-GRPOAns× Mem×Ans× Mem
Ans Mem×
Ans Mem
Figure 3: Simulated GRPO advantage distributions.
Each row samples synthetic groups of size G=8 from
a four-cell outcome mixture; p11is the probability of
memory-supported correct (1,1) . Left: answer-only
reward collapses the four cells into answer-wrong and
answer-correct regions. Right: AWM assigns a distinct
raw reward to each of the four cells before group nor-
malization.
Ans✓Mem×are learned together as a single
answer-correct region; memory answerability
is invisible.
•A WM distinguishes the answer-correct cells:
in the right column, Ans ✓Mem×and Ans ✓
Mem✓receive distinct raw rewards and can
therefore receive different normalized advan-
tages. This is the memory-refinement prefer-
ence missing from answer-only reward.
•The preference changes with group compo-
sition: as p11increases, the red Ans ✓Mem×
curve moves below zero while the blue Ans ✓
Mem✓curve remains positive. The nearby
green Ans ×Mem✓curve is expected because
r01=−0.1 is close to r10=0; the key compari-
son is between the two answer-correct cells.
What the reward does not claim.The reward
uses memory-only answerability as a training sig-
nal, not as a claim of full source-grounding verifi-
cation. It requires no gold terminal working mem-
ory: the frozen reader Ranswers from the agent-
written Mterm, and the frozen training scorer eval-
uates both the agent’s final answer and the memory-
only answer against the gold answer. The reward
does not score memory style, length, or formatting,
and memory-only answerability does not override

final-answer correctness; r10> r01preserves final-
answer priority. Appendix C discusses reward vari-
ants, simulation details, and why we adopt this
minimal form.
4 Experiments
We evaluate whether AWM-GRPO improves final-
answer accuracy and terminal-working-memory
quality over direct-input, RAG, SFT, and answer-
only GRPO baselines.
4.1 Experimental Setup
Benchmarks.We evaluate on LONG-
DOCURL (Deng et al., 2025) ( N=2325 )
and MMLONGBENCH-DOC(Ma et al., 2024)
(N=1082 ), under each benchmark’s official
evaluation protocol (Appendix A).
Metrics.We report final-answer accuracy,
memory-only accuracy, and Pmmc from Sec. 2.
Both final and memory-only answers are evaluated
by the official judge Jdefined in Sec. 2. The judge
uses GPT-4o to extract a concise candidate answer
and then applies each benchmark’s deterministic
rules to compute generalized accuracy.
Agent and retrieval harness.The policy is
Qwen3-VL-4B-Instruct served via SGLang. The
agent runs the retrieve-inspect-update memory loop
of Sec. 2 with an append-only mandatory memory-
update policy and a maximum of 15agent steps per
question. Page retrieval uses Jina v4 page-image
embeddings indexed by Qdrant, returning the top
three candidate pages per query (RAG Top-3).
Post-training data.SFT and GRPO use post-
training prompts from DOC-750K (Duan et al.,
2025), disjoint from the evaluation sets. A 2000 -
question rendered subset yields 994SFT trajecto-
ries via difficulty-balanced bucket sampling, with
kimi-k2.5 as the teacher; the remaining 1006 ques-
tions form the RL-unseen pool, from which 226
mixed-difficulty groups are retained for GRPO
training. Online GRPO uses a frozen, locally
hosted Qwen3-14B model to generate the memory-
only answer and compute both reward scores; only
the 4B policy is updated.
Variants.We evaluate two no-memory baselines
and three AWM-Agent variants. Thedirect-input
baseline receives all document pages in one pass,
without retrieval, tools, or terminal working mem-
ory. The second,VLM + RAG Top-3, retrieves thetop three pages and answers directly without writ-
ing terminal working memory. The three AWM-
Agent variants share the same model, tools, mem-
ory schema, retrieval harness, prompts, and opti-
mizer; only the post-training step or reward dif-
fers:(i) AWM-Agent (SFT): trained on the 994-
trajectory DOC-750K subset.(ii) AWM-Agent +
Answer-GRPO: the SFT checkpoint with GRPO un-
der the answer-only reward Rans.(iii) AWM-Agent
+ AWM-GRPO: the SFT checkpoint with GRPO
under the AWM reward RAWM at default values
(2,0,−0.1,−1)(Sec. 3.2).
4.2 Main Results
Table 4 reports final-answer accuracy for external
baselines and our variants. The external rows use
different backbones and published protocols, so
they provide context rather than controlled com-
parisons. Our method comparisons use the five
Qwen3-VL-4B rows, which share the backbone
and official scoring protocol.
The direct-input and RAG rows provide no-
memory baselines, while Sec. 4.3 analyzes mem-
ory quality only for methods that produce termi-
nal working memory. On MMLONGBENCH-DOC,
AWM-GRPO reaches 53.9 and improves over di-
rect input, RAG Top-3, SFT, and answer-only
GRPO by 10.4, 8.1, 4.7, and 2.3 points, respec-
tively. It also improves over RAG Top-3, SFT, and
answer-only GRPO by 11.9, 4.4, and 2.7 points on
LONGDOCURL.
4.3 Analysis
Memory-quality analysis.Tables 5 and 6 report
memory-quality metrics across the three AWM-
Agent variants on MMLONGBENCH-DOC. The
direct-input and RAG baselines are excluded be-
cause they do not produce terminal working mem-
ory. Table 5 uses the full multi-turn setting; Table 6
uses the EP-given setting, where retrieval is re-
placed by gold evidence pages to isolate memory
extraction from page access.
SFT already achieves a relatively low Pmmc of
17.7%. Answer-only GRPO improves final-answer
accuracy but raises Pmmc to 19.9%, because it as-
signs the same reward to memory-supported cor-
rect and memory-missing correct trajectories; its
effect on memory quality is incidental. AWM-
GRPO separates these two cells through the AWM
reward (Sec. 3). Compared with answer-only
GRPO, AWM-GRPO improves memory-only ac-
curacy from 42.5 to 44.5 and reduces Pmmc from

Method Backbone Param Paradigm MMLONGBENCH-DOCLONGDOCURL
Open-source / RAG
Qwen2.5-VL (Zheng et al., 2026) Qwen2.5-VL 7B VLM 28.0 32.9
SV-RAG (Chen et al., 2024) InternVL2 4B RAG 23.0 –
M3DocRAG (Cho et al., 2024) Qwen2-VL 7B RAG 21.0 35.1
VisRAG (Yu et al., 2025) MiniCPM-V 8B RAG 18.8 41.9
MoLoRAG (Wu et al., 2025) Qwen2.5-VL 7B RAG 41.0 51.9
URaG-7B (Shi et al., 2026) Qwen2.5-VL 7B RAG 33.8 52.2
Agentic / RL
VRAG-RL (Wang et al., 2025) Qwen2.5-VL 7B Agent + RL 26.6 44.9
Doc-V⋆(SFT) (Zheng et al., 2026) Qwen2.5-VL 7B Agent + SFT 39.8 53.0
Doc-V⋆(GRPO) (Zheng et al., 2026) Qwen2.5-VL 7B Agent + GRPO 42.1 56.3
Ours (Qwen3-VL-4B)
Qwen3-VL-4B (direct input) Qwen3-VL 4B VLM 43.5 –
VLM + RAG Top-3 (no memory) Qwen3-VL 4B RAG 45.8 48.2
AWM-Agent (SFT) Qwen3-VL 4B Agent + SFT 49.2 55.7
+ Answer-GRPO Qwen3-VL 4B GRPO (R ans) 51.6 57.4
+ AWM-GRPO Qwen3-VL 4B GRPO (R AWM )53.9 60.1
Table 4: Final-answer accuracy; higher is better. External rows follow their reported protocols and provide context
only. Our five Qwen3-VL-4B rows use the official scoring protocol.
Variant Final acc. Mem acc.P mmc (%)
SFT 40.8 38.5 17.7
+ Answer-GRPO 43.2 42.5 19.9
+ AWM-GRPO 45.4 44.5 17.2
Table 5: Memory-quality analysis of the three AWM-
Agent variants on 1,082 MMLONGBENCH-DOCtrajec-
tories in the full multi-turn setting.
Variant SubsetFinal
acc.Mem
acc.Pmmc
(%)
+ Answer-GRPO MM EP 45.4 41.2 19.1
+ AWM-GRPO MM EP 48.0 43.5 16.4
Table 6: Memory-quality analysis of Answer-GRPO and
AWM-GRPO on 733 answerable MMLONGBENCH-
DOCexamples in the EP-given setting.
19.9% to 17.2%, while also improving final-answer
accuracy. AWM-GRPO also improves both final-
answer and memory-only accuracy over SFT (+4.6
and +6.0 points) while keeping Pmmc slightly lower
(17.2 vs. 17.7). The targeted failure is narrow: af-
ter a relevant page is observed, the agent must
commit a question-conditioned finding to mem-
ory. AWM-GRPO supplies trajectory-level reward
for this terminal-working-memory artifact, rather
than changing retrieval tools or the answer prompt.
Table 6 reports the EP-given comparison: with
retrieval replaced by gold evidence pages, AWM-
GRPO improves final-answer accuracy from 45.4
to 48.0, memory-only accuracy from 41.2 to 43.5,
and reduces Pmmc from 19.1% to 16.4% relative
to answer-only GRPO, showing that the memory-
40 160 280
Training step363840424446Memory-only accuracy (%)
38.840.443.5
AWM-GRPOFigure 4: AWM-GRPO memory-only accuracy over
training steps on MMLONGBENCH-DOC. Memory-
only accuracy increases from 38.8 at step 40 to 43.5 at
step 280.
aware reward remains beneficial even when page
access is controlled.
Training dynamics.We track memory-only ac-
curacy across AWM-GRPO training steps on
MMLONGBENCH-DOC. Figure 4 shows a mono-
tonic increase from 38.8 at step 40 to 43.5 at step
280, suggesting that the reward progressively im-
proves the answerability of terminal working mem-
ory during training.
Evidence-source analysis.Table 7 compares
Answer-GRPO and AWM-GRPO by answer-
source category. AWM-GRPO lowers Pmmc for
mixed visual+text, text-multi, figure, chart, and
none categories, while table and layout questions
show higher Pmmc. This suggests that the mem-
ory reward helps in several cross-source settings
but does not uniformly solve structured-evidence

Final acc.P mmc (%)
SourceNAns AWM Ans AWM
Pure-text 135 46.158.5 22.823.1
Table 143 36.739.5 15.223.9
Chart 107 45.746.021.417.5
Figure 187 41.043.920.316.2
Layout 3275.974.34.815.8
visual+text (mixed) 180 36.737.336.222.2
visual (multi) 3 – – – –
text (multi) 62 40.148.331.824.0
none 233 47.549.210.45.4
All 1082 43.245.419.917.2
Table 7: Evidence-source comparison on
MMLONGBENCH-DOCin the full multi-turn
setting. “Ans” denotes Answer-GRPO and “AWM”
denotes AWM-GRPO; bold marks the better value
within each source.
preservation.
5 Related Work
Long-document VQA agents.
MMLONGBENCH-DOC(Ma et al., 2024)
and LONGDOCURL (Deng et al., 2025) are two
recent benchmarks for long-document VQA over
PDFs that average tens to hundreds of pages and
mix textual, tabular, and figure-grounded ques-
tions. Doc-V⋆treats this task as sequential page
navigation: the agent starts from a low-resolution
overview, fetches selected page images, and
records findings in structured working memory be-
fore answering (Zheng et al., 2026). Its navigation
policy is trained with imitation learning and GRPO.
MemSearcher studies compact working memory
across turns in text-based search agents rather than
page-image VQA, using multi-context GRPO to
propagate trajectory-level advantages to individual
turns (Yuan et al., 2025). AgenticRAG-R1 uses
stack memory for multi-step reasoning, retrieval,
and memorizing, while TC-RAG studies Turing-
complete RAG for medical LLM systems (Jiang
et al., 2026, 2025). SCAIR, GR-Agent, and
EigentSearch structure retrieval or reasoning with
schemas or tools (Chaturvedi et al., 2026; Zhou
et al., 2025b; Zhang et al., 2026; Li et al., 2025).
AWM keeps the agent architecture and memory
format, but evaluates whether terminal memory is
answerableon its own and uses that signal as an
RL reward.Intermediate-state supervision.TableQA work
filters questions and prunes tables to retain answer-
critical content (Ye et al., 2026; Guo et al., 2026).
For visual inputs, CapRL rewards answerable
captions, while Vision-SR1 optimizes question-
conditioned visual descriptions and answers (Xing
et al., 2026; Li et al., 2026b,a; Xiao et al., 2026b,
2025). Perception-R1 checks visual content in
a reasoning response against reference annota-
tions (Xiao et al., 2026a). AWM instead evaluates
the answerability of query-specific terminal work-
ing memory accumulated across sequential page
inspections.
Memory evaluation.Evidence-citation work
asks whether an agent’s final answer is supported
by its cited evidence (Liu et al., 2023; Gao et al.,
2023). Related RAG work tests reasoning when
retrieved knowledge is incomplete or exposes the
argument structure behind an answer (Zhou et al.,
2026a; Zhu et al., 2025b). Memory-only an-
swerability asks whether saved memory is suffi-
cient to answer after the page images and inter-
action trajectory are removed, a question related
to self-evaluation in retrieval-augmented genera-
tion (Asai et al., 2024). Neither test subsumes
the other: answerable memory may contain un-
supported claims, while grounded memory may
omit a fact needed to answer. Other diagnostics
audit prompt-conditioned traces left by RLVR (Liu
et al., 2025); they target data membership, not
whether terminal memory supports answering. In
our diagnostic, a frozen language model reads only
(q, Mterm)and produces a memory-only answer,
which the official benchmark judge scores along-
side the agent’s final answer. This reader-based di-
agnostic is related to LLM-based evaluation (Zheng
et al., 2023; Gilardi et al., 2023). Memory probes
in interpretability (Ghandeharioun et al., 2024) ask
whether internal activations encode a fact, whereas
we ask whether an externalized, source-linked ter-
minal record encodes it.
6 Conclusion
Memory-only answerability identifies cases in
which an agent answers correctly but leaves ter-
minal working memory that cannot support the an-
swer on its own. AWM-GRPO improves accuracy
over answer-only GRPO on both benchmarks and
lowers Pmmc in both settings. Two benchmarks
and one model size leave grounding unverified.

Limitations
All controlled AWM comparisons use Qwen3-VL-
4B on LONGDOCURL and MMLONGBENCH-
DOC; whether AWM-GRPO helps at 30B+ or on
other document distributions, such as scientific pa-
pers, financial filings, and slide decks, remains
open. Memory-only answerability may depend
on the frozen Qwen3-14B reader, and we did not
test robustness with a second reader. Errors in the
fixed judge’s answer-extraction step can affect the
reported metrics. Our 100-example audit found
six disagreements but no consistent direction of er-
ror; a larger audit would provide stronger evidence.
The AWM reward measures answerability, not full
source-grounding verification; future work can add
source-conditioned checks for individual memory
claims. Memory-only answerability also requires
an extra reader pass and an extraction pass during
offline evaluation, although neither pass is needed
when the trained agent is deployed.
Ethics Statement
This work uses publicly released benchmarks
(MMLONGBENCH-DOC, LONGDOCURL), a pub-
lic open-weight VLM (Qwen3-VL-4B-Instruct),
and a fixed local Qwen3-14B model for memory
reading and reward scoring during online training.
We collect no new data, do not work with human
subjects, and do not target deployment. The official
judge follows each benchmark’s answer-extraction
and rule-based scoring pipeline. Separately, the
controlled intervention in Appendix D uses GPT-4o
to construct improved memory from gold evidence
pages; those memories are not used for training.
The fixed training scorer, reader, and answer ex-
tractor may introduce systematic errors; our man-
ual audit does not establish broader scoring robust-
ness or claim-level source grounding. Per ARR /
EMNLP policy, AI assistants were used for editing;
all technical content is the authors’ own.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to re-
trieve, generate, and critique through self-reflection.
InInternational conference on learning representa-
tions, volume 2024, pages 9112–9141.
Prateek Chaturvedi, Yuqicheng Zhu, Hongkuan Zhou,
Dongzhuoran Zhou, Yunjie He, Steffen Staab, Fei
Du, Jie Tang, and Evgeny Kharlamov. 2026. SCAIR:
Schema-conditioned agentic iterative reasoning forenterprise knowledge graphs. InProceedings of the
64th Annual Meeting of the Association for Computa-
tional Linguistics (Volume 6: Industry Track), pages
1089–1104.
Jian Chen, Ruiyi Zhang, Yufan Zhou, Tong Yu,
Franck Dernoncourt, Jiuxiang Gu, Ryan A. Rossi,
Changyou Chen, and Tong Sun. 2024. Sv-
rag: Lora-contextualizing adaptation of mllms for
long document understanding.arXiv preprint
arXiv:2411.01106.
Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie
He, and Mohit Bansal. 2024. M3docrag: Multi-
modal retrieval is what you need for multi-page
multi-document understanding.arXiv preprint
arXiv:2411.04952.
Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, Zhong-
Zhi Li, Jian Xu, Xiao-Hui Li, Yuan Gao, Jun Song,
Bo Zheng, and Cheng-Lin Liu. 2025. Longdocurl:
a comprehensive multimodal long document bench-
mark integrating understanding, reasoning, and locat-
ing. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1135–1159.
Yuchen Duan, Zhe Chen, Yusong Hu, Weiyun Wang,
Shenglong Ye, Botian Shi, Lewei Lu, Qibin Hou,
Tong Lu, Hongsheng Li, Jifeng Dai, and Wenhai
Wang. 2025. Docopilot: Improving multimodal mod-
els for document-level understanding. InProceed-
ings of the Computer Vision and Pattern Recognition
Conference, pages 4026–4037.
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani,
Gautier Viaud, Céline Hudelot, and Pierre Colombo.
2025. Colpali: Efficient document retrieval with
vision language models. InInternational Conference
on Learning Representations, volume 2025, pages
61424–61449.
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen.
2023. Enabling large language models to generate
text with citations. InProceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing, pages 6465–6488.
Asma Ghandeharioun, Avi Caciularu, Adam Pearce,
Lucas Dixon, and Mor Geva. 2024. Patchscopes:
A unifying framework for inspecting hidden rep-
resentations of language models.arXiv preprint
arXiv:2401.06102.
Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli.
2023. Chatgpt outperforms crowd-workers for text-
annotation tasks.Proceedings of the National
Academy of Sciences, 120(30):e2305016120.
Yu Guo, Shenghao Ye, Shuangwu Chen, Zijian Wen,
Tao Zhang, Qirui Bai, Dong Jin, Yunpeng Hou,
Huasen He, Jianyang, and Xiaobin Tan. 2026. Re-
thinking table pruning in TableQA: From sequential
revisions to gold trajectory-supervised parallel search.
InProceedings of the 64th Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 12960–12976.

Yuan He, Bailan He, Zifeng Ding, Alisia Maria Lupidi,
Yuqicheng Zhu, Shuo Chen, Caiqi Zhang, Jiaoyan
Chen, Yunpu Ma, V olker Tresp, and Ian Horrocks.
2025. Supposedly equivalent facts that aren’t? en-
tity frequency in pre-training induces asymmetry in
LLMs. InSecond Conference on Language Model-
ing.
Xinke Jiang, Yue Fang, Rihong Qiu, Haoyu Zhang,
Yongxin Xu, Hao Chen, Wentao Zhang, Ruizhe
Zhang, Yuchen Fang, Xu Chu, and 1 others. 2025.
TC-RAG: Turing-complete RAG’s case study on
medical LLM systems.ACL oral 2025.
Xinke Jiang, Yue Fang, Zhibang Yang, Jiaran Gao,
Zhixin Zhang, Tao Feng, Rihong Qiu, Wentao Zhang,
Hongxin Ding, Ruizhe Zhang, and 1 others. 2026.
Agenticrag-r1: Agentic reinforcement learning with
stack memory for multi-step reasoning, retrieval and
memorizing. InEMNLP.
Songze Li, Xiaoke Guo, Tianqi Liu, Biao Yi, Zhaoyan
Gong, Zhiqiang Liu, Huajun Chen, and Wen Zhang.
2026a. What’s missing in screen-to-action? towards
a ui-in-the-loop paradigm for multimodal gui reason-
ing. InFindings of the Association for Computa-
tional Linguistics: ACL 2026, pages 17674–17690.
Songze Li, Zhiqiang Liu, Zhengke Gui, Huajun Chen,
and Wen Zhang. 2025. Enrich-on-graph: Query-
graph alignment for complex reasoning with LLM
enriching. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing,
pages 7683–7703.
Zongxia Li, Wenhao Yu, Chengsong Huang, Zhenwen
Liang, Rui Liu, Fuxiao Liu, Jingxi Chen, Dian Yu,
Jordan Lee Boyd-Graber, Haitao Mi, and Dong Yu.
2026b. Vision-SR1: Self-rewarding vision-language
model via reasoning decomposition and multi-reward
policy optimization. InInternational Conference on
Learning Representations.
Gyubeum Lim, Yemo Koo, and Vijay Krishna Madis-
etti. 2026. Scope VLM: selective context processing
for efficient document navigation in vision-language
models. InProceedings of the 19th Conference of
the European Chapter of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
95–140.
Jiahang Lin, Kai Hu, Binghai Wang, Yuhao Zhou, Zhi-
heng Xi, Honglin Guo, Shichun Liu, Junzhe Wang,
Shihan Dou, Enyu Zhou, Hang Yan, Zhenhua Han,
Tao Gui, Qi Zhang, and Xuanjing Huang. 2026. Mm-
doc-r1: Training agents for long document visual
question answering through multi-turn reinforcement
learning.arXiv preprint arXiv:2604.13579.
Nelson F. Liu, Tianyi Zhang, and Percy Liang. 2023.
Evaluating verifiability in generative search engines.
InFindings of the Association for Computational
Linguistics: EMNLP 2023, pages 7001–7025.
Yule Liu, Heyi Zhang, Jinyi Zheng, Zhen Sun, Zifan
Peng, Jiaheng Wei, Tianshuo Cong, Yilong Yang,and Xinlei He. 2025. Auditing data membership in
reinforcement learning with verifiable rewards.arXiv
preprint arXiv:2511.14045.
Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen,
Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan
Ma, Xiaoyi Dong, Pan Zhang, Liangming Pan, Yu-
Gang Jiang, Jiaqi Wang, Yixin Cao, and Aixin
Sun. 2024. MMLONGBENCH-DOC: benchmarking
long-context document understanding with visualiza-
tions.Advances in Neural Information Processing
Systems, 37:95963–96010.
Nico Potyka, Yuqicheng Zhu, Yunjie He, Evgeny Khar-
lamov, and Steffen Staab. 2024. Robust knowledge
extraction from large language models using social
choice theory. InProceedings of the 23rd Interna-
tional Conference on Autonomous Agents and Multi-
agent Systems, pages 1593–1601.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, Y . K. Li, Y . Wu, and Daya Guo. 2024.
Deepseekmath: Pushing the limits of mathemat-
ical reasoning in open language models.URL
https://arxiv. org/abs/2402.03300, 2(3):5.
Yongxin Shi, Jiapeng Wang, Zeyu Shan, Dezhi Peng,
Zening Lin, and Lianwen Jin. 2026. Urag: Unified
retrieval and generation in multimodal llms for effi-
cient long document understanding. InProceedings
of the AAAI Conference on Artificial Intelligence,
volume 40, pages 25357–25365.
Qiuchen Wang, Ruixue Ding, Yu Zeng, Zehui Chen,
Lin Chen, Shihang Wang, Pengjun Xie, Fei Huang,
and Feng Zhao. 2025. VRAG-RL: empower vision-
perception-based RAG for visually rich information
understanding via iterative reasoning with reinforce-
ment learning.CoRR, abs/2505.22019.
Xixi Wu, Yanchao Tan, Nan Hou, Ruiyang Zhang, and
Hong Cheng. 2025. Molorag: Bootstrapping doc-
ument understanding via multi-modal logic-aware
retrieval. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing,
pages 14024–14045.
Tong Xiao, Xin Xu, Zhenya Huang, Hongyu Gao, Quan
Liu, Qi Liu, and Enhong Chen. 2026a. Perception-
R1: Advancing multimodal reasoning capabilities of
MLLMs via visual perception reward. InInterna-
tional Conference on Learning Representations.
Xi Xiao, Chen Liu, Chih-Ting Liao, Yunbei Zhang,
Qizhen Lan, Yuxiang Wei, Lin Zhao, Janet Wang,
Jianyang Gu, Muchao Ye, and 1 others. 2026b. Stay-
ing VIGILant: Mitigating visual laziness via counter-
factual visual alignment in MLLMs.arXiv preprint
arXiv:2606.26387.
Xi Xiao, Yunbei Zhang, Xingjian Li, Tianyang Wang,
Xiao Wang, Yuxiang Wei, Jihun Hamm, and Min Xu.
2025. Visual instance-aware prompt tuning. InPro-
ceedings of the 33rd ACM International Conference
on Multimedia, pages 2880–2889.

Can Xie, Ruotong Pan, Xiangyu Wu, Yunfei Zhang,
Jiayi Fu, Tingting Gao, and Guorui Zhou. 2026a. Un-
locking exploration in RLVR: Uncertainty-aware ad-
vantage shaping for deeper reasoning. InFindings of
the Association for Computational Linguistics: ACL
2026, pages 19057–19076.
Can Xie, Yuyi Zhou, Wen Yang, Ziyi Zhang, Siyao
Song, Yingzhuo Deng, Shuo Ren, and Jiajun Zhang.
2026b. EDGE: Experience-distillation for guided ex-
ploration in agentic reinforcement learning.Preprint,
arXiv:2608.21946.
Long Xing, Xiaoyi Dong, Yuhang Zang, Yuhang Cao,
Jianze Liang, Qidong Huang, Jiaqi Wang, Feng Wu,
and Dahua Lin. 2026. CapRL: Stimulating dense im-
age caption capabilities via reinforcement learning.
InInternational Conference on Learning Representa-
tions.
Shenghao Ye, Yu Guo, Dong Jin, Yuxiang Wang, Yikai
Shen, Yunpeng Hou, Shuangwu Chen, Jianyang, and
Xiaofeng Jiang. 2026. When TableQA meets noise:
A dual denoising framework for complex questions
and large-scale tables. InProceedings of the 64th
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 24022–
24045.
Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Jun-
hao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang,
Xu Han, Zhiyuan Liu, and Maosong Sun. 2025. Vis-
rag: Vision-based retrieval-augmented generation on
multi-modality documents. InInternational Con-
ference on Learning Representations, volume 2025,
pages 21074–21098.
Qianhao Yuan, Jie Lou, Zichao Li, Jiawei Chen, Yaojie
Lu, Hongyu Lin, Le Sun, Debing Zhang, and Xianpei
Han. 2025. Memsearcher: Training llms to reason,
search and manage memory via end-to-end reinforce-
ment learning.arXiv preprint arXiv:2511.02805.
Boer Zhang, Mingyan Wu, Dongzhuoran Zhou,
Yuqicheng Zhu, Wendong Fan, Puzhen Zhang, Zifeng
Ding, Guohao Li, and Yuan He. 2026. EigentSearch-
Q+: Enhancing deep research agents with structured
reasoning tools. InProceedings of the ACM Confer-
ence on AI and Agentic Systems, pages 1114–1118.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging
llm-as-a-judge with mt-bench and chatbot arena. In
NeurIPS.
Yuanlei Zheng, Pei Fu, Hang Li, Ziyang Wang, Yuyi
Zhang, Wenyu Ruan, Xiaojin Zhang, Zhongyu Wei,
Zhenbo Luo, Jian Luan, Wei Chen, and Xiang
Bai. 2026. Doc-v*:coarse-to-fine interactive visual
reasoning for multi-page document VQA.arXiv
preprint arXiv:2604.13731.
Dongzhuoran Zhou, Yuqicheng Zhu, Xiaxia Wang,
Yuan He, Jiaoyan Chen, Steffen Staab, and EvgenyKharlamov. 2025a. Evaluating knowledge graph
based retrieval augmented generation methods un-
der knowledge incompleteness.arXiv preprint
arXiv:2504.05163.
Dongzhuoran Zhou, Yuqicheng Zhu, Xiaxia Wang,
Hongkuan Zhou, Jiaoyan Chen, Steffen Staab, Yuan
He, and Evgeny Kharlamov. 2025b. GR-Agent:
Adaptive graph reasoning agent under incomplete
knowledge.arXiv preprint arXiv:2512.14766.
Dongzhuoran Zhou, Yuqicheng Zhu, Xiaxia Wang,
Hongkuan Zhou, Yuan He, Jiaoyan Chen, Steffen
Staab, and Evgeny Kharlamov. 2026a. What breaks
knowledge graph based RAG? benchmarking and
empirical insights into reasoning under incomplete
knowledge. InProceedings of the 19th Conference of
the European Chapter of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
2522–2538.
Yixiao Zhou, Dongzhou Cheng, Zhiliang Wu, Yi Yang,
Yu Cheng, and Hehe Fan. 2026b. One refiner to
unlock them all: Inference-time reasoning elicitation
via reinforcement query refinement. InProceedings
of the 64th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 38957–38978.
Yixiao Zhou, Yang Li, Dongzhou Cheng, Hehe Fan,
and Yu Cheng. 2026c. Look inward to explore out-
ward: Learning temperature policy from LLM in-
ternal states via hierarchical RL.arXiv preprint
arXiv:2602.13035.
Yuqicheng Zhu, Daniel Hernández, Yuan He, Zifeng
Ding, Bo Xiong, Evgeny Kharlamov, and Steffen
Staab. 2025a. Predicate-conditional conformalized
answer sets for knowledge graph embeddings. In
Findings of the Association for Computational Lin-
guistics: ACL 2025, pages 4145–4167.
Yuqicheng Zhu, Nico Potyka, Daniel Hernández, Yuan
He, Zifeng Ding, Bo Xiong, Dongzhuoran Zhou,
Evgeny Kharlamov, and Steffen Staab. 2025b. Ar-
gRAG: Explainable retrieval augmented generation
using quantitative bipolar argumentation. InPro-
ceedings of the 19th International Conference on
Neurosymbolic Learning and Reasoning, volume 284
ofProceedings of Machine Learning Research, pages
697–718. PMLR.
Yuqicheng Zhu, Nico Potyka, Mojtaba Nayyeri,
Bo Xiong, Yunjie He, Evgeny Kharlamov, and Stef-
fen Staab. 2024. Predictive multiplicity of knowl-
edge graph embeddings in link prediction. InFind-
ings of the Association for Computational Linguistics:
EMNLP 2024, pages 334–354.
Yuqicheng Zhu, Nico Potyka, Jiarong Pan, Bo Xiong,
Yunjie He, Evgeny Kharlamov, and Steffen Staab.
2025c. Conformalized answer set prediction for
knowledge graph embedding. InProceedings of
the 2025 Conference of the Nations of the Ameri-
cas Chapter of the Association for Computational

Linguistics: Human Language Technologies (Volume
1: Long Papers), pages 731–750.
Yuqicheng Zhu, Jialin Yu, Lin Li, Gengyuan Zhang,
Zhen Yang, Steffen Staab, Puneet Dokania, Philip
Torr, Jie Tang, and Evgeny Kharlamov. 2026. Confor-
malized large language models under configuration
shift.arXiv preprint arXiv:2608.01460.
A Dataset Details
Benchmark sources.MMLONGBENCH-
DOCuses all 1,082 examples in the orig-
inal samples.json . LONGDOCURL
uses all 2,325 examples in the original
LongDocURL_public.jsonl . The
main-results table uses the original full-size
MMLONGBENCH-DOCpage images.
EP-given evaluation.In the diagnostic experi-
ments, EP-given replaces retrieval with the anno-
tated gold evidence pages and is reported on fixed
MMLONGBENCH-DOCsubsets specified in the
corresponding tables. We do not report a LONG-
DOCURL EP-given diagnostic in the main experi-
ments.
Routing verification.All reported runs pass a
check that each question is paired with its source
document.
B Evaluation Protocol
Official benchmark scoring.All final-answer
and memory-only metrics reported for our Qwen3-
VL-4B runs on MMLONGBENCH-DOCand
LONGDOCURL use the official judge Jdefined
in Sec. 2. The judge follows the official evaluation
pipeline for the corresponding benchmark. It first
sends the model’s free-form output to GPT-4o at
temperature 0. GPT-4o extracts a concise candi-
date answer in the benchmark’s expected format. A
benchmark-specific deterministic rule scorer then
compares the extracted candidate with the gold an-
swer and assigns the official generalized-accuracy
score.
Type-aware rule scoring.Both official scorers
normalize the extracted candidate and gold an-
swer to the benchmark’s answer types before com-
parison. Integer answers require an exact match.
Floating-point answers allow a 1%relative toler-
ance and account for equivalent percentage scales.
For string and None answers, the scorer uses exact
matching for designated special strings and other-
wise uses average normalized Levenshtein similar-
ity (ANLS) with a 0.5threshold. For list answers,the MMLONGBENCH-DOCscorer requires equal
list lengths, sorts both lists, and uses the minimum
element-level ANLS as the question score. The
LONGDOCURL scorer instead matches list ele-
ments greedily by ANLS and applies a square-root
penalty when the list lengths differ.
Final-answer and memory-only metrics.For
final-answer accuracy, the candidate is the agent’s
final answer. For memory-only accuracy, a frozen
reader receives the question qand terminal work-
ing memory Mterm, but no trajectory, page images,
or tool instructions. It then produces the candi-
date answer. The reader is the frozen Qwen3-14B
model defined in Sec. 2. The same judge evalu-
ates both candidates. Unless stated otherwise, re-
ported accuracy is the mean official per-question
generalized-accuracy score. The four-cell memory
analysis requires binary outcomes, so it treats any
positive official score as correct and a zero score
as incorrect. The controlled intervention in Sec. 2
reports this binary positive-score rate so that each
percentage matches its correct count and Wilson
confidence interval.
Manual audit.We manually checked 100 scored
examples against human correctness judgments.
The audit found six disagreements, mainly involv-
ing answer aliases, list formatting, and partial-
credit edge cases. The six disagreements were not
concentrated on one side of the comparison.
Training-time reward.Online GRPO uses a
frozen local Qwen3-14B model to generate the
memory-only answer and compute the two reward
scores. This training-time scorer is separate from
the official judge used for reported metrics. Only
the Qwen3-VL-4B agent policy is updated; the
Qwen3-14B model remains fixed.
C Design Choices and Reward Variants
This appendix expands on the design choices and
reward analyses summarized in the main text.
C.1 Why a conditional rate and not a
marginal gap
For a set of Nexamples, let n10andn01de-
note the numbers of memory-missing-correct and
answering-error examples, respectively. The
marginal gap between final-answer and memory-
only accuracy is (n10−n 01)/N. Because the two
failures enter with opposite signs, they can can-
cel: the gap can be near zero even when many

correct answers have unanswerable terminal work-
ing memory. It therefore does not directly answer
the question that motivates AWM: among the final
answers that the agent gets right, how often is the
saved memory still insufficient?
The conditional rate Pmmc =n 10/Ncorrect
measures the fraction of binary-correct final an-
swers with unanswerable terminal working mem-
ory. Answering-error examples enter neither its
numerator nor its denominator, so they cannot can-
cel the memory failures of interest. Across datasets,
subsets, and checkpoints, Pmmc retains the same
interpretation: the share of correct final answers
that cannot be recovered from memory alone. We
therefore report it rather than the signed marginal
gap.
C.2 Why mandatory commit does not collapse
memory-only accuracy onto final-answer
accuracy
Our agent loop makes a memory update mandatory
after every retrieval. A natural worry is that this
construction makes memory-only accuracy track
final-answer accuracy because every retrieval is
followed by a memory update. It does not. Each
update is free-form within a schema that limits find-
ings to 40 words and requires a source page. An
agent can satisfy that schema with a generic page
description, omit the fact needed by the question,
and still answer from raw page context at the final
step. The diagnostic in Sec. 2 tests this gap between
writing an update and preserving what the later an-
swer needs. Mandatory commit therefore isolates
whatthe agent extracts at each update rather than
whenit chooses to update memory.
C.3 Reward variants we considered
The minimal form in Sec. 3 pairs one answer from
the frozen reader with one score for that answer.
We also considered three richer variants.
Per-finding source-image grounding.A more
granular signal asks a model to check, for each find-
ing in Mterm, whether the cited source_page
image actually supports the claim. This is a stricter
per-finding evidence-grounding signal and would
provide a finer-grained training signal. We did not
adopt it for the main experiments because (i) it re-
quires one scoring call per finding rather than one
per trajectory, raising compute and scoring noise at
training time, and (ii) it conditions the reward on a
fixed schema, sacrificing the schema-agnosticismthat we view as a feature of our minimal form.
Source-grounding audits using this variant are a
natural follow-up evaluation, separate from the re-
ward signal used during training.
Self-consistency over multiple reader rollouts.
The binary memory score inherits the variance
of a single reader rollout. Sampling Kanswers
from R(q, Mterm)and scoring their correct frac-
tion would reduce variance but multiply scoring
cost byK; we do not evaluate this variant.
Length penalty on Mterm.We considered a
training-time penalty for Mtermentries that closely
mirror the predicted final-answer string. We did
not adopt it because such a penalty could suppress
legitimate answer-supporting findings that contain
the answer value (e.g., when the answer is a value
on the page); answer-echo detection remains a sep-
arate audit.
C.4 Computation overhead relative to
standard GRPO
Training-time overhead.Relative to standard
answer-only GRPO, AWM-GRPO adds two short
text-only passes through a local Qwen3-14B model
for each rollout. The first pass answers from termi-
nal working memory alone, and the second scores
that memory-only answer for the training reward.
The answer-only baseline already scores the agent’s
final answer. The Qwen3-14B model remains
frozen, and only the Qwen3-VL-4B agent policy
is trained. We report the added model passes in-
stead of a wall-clock percentage because elapsed
time depends on hardware, batching, and serving
configuration.
Deployment-time overhead.The memory
reader and training scorer are not called at
deployment. AWM-GRPO therefore adds no
deployment-time model call relative to the same
agent policy and tool loop.
C.5 Reward discrimination under A WM
This appendix gives a structural statement of how
the AWM reward (Sec. 3) differs from an answer-
only reward over the four outcome cases of Ta-
ble 1. The statement is purely structural and as-
sumes only that the scoring and reader interfaces
are fixed within each pass. During online train-
ing, the frozen local Qwen3-14B model supplies
the reader output and the two reward scores. For

reported diagnostics, Ris the frozen Qwen3-14B
reader andJis the official judge from Sec. 2.
Letsans, smem∈ {0,1} denote final-answer cor-
rectness and memory-only answerability for a tra-
jectory τ, as in Sec. 3.2, and let RAWM be the piece-
wise AWM reward with values (r11, r10, r01, r00)
satisfying r11> r 10> r 01> r 00. The reward
mapping (sans, smem)7→R AWM strictly orders
the four outcome cells: memory-supported correct
receives the highest value, memory-missing cor-
rect sits above answering error, and unresolved
error receives the lowest value. Answer-only re-
ward, by contrast, takes only two values and identi-
fies(sans, smem)=(1,1) with (sans, smem)=(1,0)
and identifies (0,1) with(0,0) . The AWM reward
therefore distinguishes pairs of policies that agree
on final-answer accuracy but differ in memory-
only answerability; answer-only reward cannot.
The default values (2,0,−0.1,−1) also preserve
final-answer priority because r10> r01, so a final-
answer-correct trajectory with unanswerable mem-
ory is never outranked by a final-answer-wrong
trajectory with answerable memory.
The same property has a direct GRPO conse-
quence (Xie et al., 2026a,b; Zhou et al., 2026c,b).
In a GRPO group of Gtrajectories sampled for
one prompt, sampled trajectories that fall into dif-
ferent reward levels can produce non-zero group-
relative advantage after normalization. The key
case is a group containing both memory-supported
correct and memory-missing correct trajectories:
answer-only reward assigns them the same value,
while the AWM reward assigns values that differ
byr11−r10. By the diagnostic of Sec. 2, this is
the cell pair where the pre-training agent already
shows a measurable memory-preservation gap.
C.6 Advantage-distribution simulation details
This appendix records the simulation behind
Figure 3. For mixture m, letρ(m)be a categorical
distribution over the four cells z= (s ans, smem).
For each reward scheme and mixture, we sample
B= 15,000 groups of size G= 8 . AWM samples
zb,1:G∼Categorical(ρ(m)), while answer-only
sampling uses the corresponding final-answer
marginal. We assign either answer-only or AWM
rewards, compute the group mean µband standard
deviation σb, and form Ab,i= (r b,i−µb)/(σ b+ϵ)
withϵ= 10−6. Figure 3 shows kernel-smoothed
empirical advantage distributions grouped by
final-answer correctness for answer-only reward
and by outcome cell for AWM. Finite groupsand discrete rewards produce a finite set of
normalized advantage values; kernel smoothing
affects only the plot, not reward or training
computation. In (0,0),(0,1),(1,0),(1,1)
order, the three mixtures are
(0.82,0.05,0.05,0.08) ,(0.60,0.05,0.05,0.30) ,
and(0.30,0.05,0.05,0.60) . The off-diagonal cells
(0,1) and(1,0) therefore retain probability 0.05
across rows, while diagonal mass shifts from (0,0)
toward (1,1) . The change in (1,0) advantage
therefore comes from the rising group mean rather
than a change in its cell frequency.
D Improved-Memory Intervention
Details
This appendix expands the controlled memory com-
parison in Sec. 2. The intervention is a controlled
sanity check rather than a deployed method or a
full-benchmark result, and the improved memories
are not used as training data. The intervention uses
a fixed set of 500 answerable MMLONGBENCH-
DOCquestions in the evidence-page-given (EP-
given) setting. Here,answerablemeans that the
benchmark annotations identify sufficient evidence
for a reference answer. It does not mean that the
agent’s terminal working memory preserves that
evidence.
D.1 Setup
EP-given evaluation provides the gold evidence
pages, which removes page retrieval as a source
of error. The empty condition supplies no termi-
nal memory. The original condition uses memory
written by the base Qwen3-VL-4B agent, while
the improved condition uses memory that GPT-4o
constructs from the same gold evidence pages. The
same 500 questions are evaluated under all three
terminal-memory conditions. The final-answer gen-
erator and judge are held fixed across conditions.
For memory-only evaluation, the fixed reader is
Qwen3-14B.
All reported outputs use the official judge de-
fined in Sec. 2.
The original agent partitions the 500 examples
into the four outcome cells from Sec. 2: 104
memory-supported correct, 77 memory-missing
correct, 47 answering error, and 272 unresolved er-
ror. We fix this partition from the original-memory
run, then compare the same questions under origi-
nal and improved memory.

Memory condition Correct / N Rate (%) 95% Wilson CI (%)
Empty memory 14 / 500 2.8 [1.7, 4.6]
Original agent memory (4B) 181 / 500 36.2 [32.1, 40.5]
Improved memory (GPT-4o) 222 / 500 44.4 [40.1, 48.8]
Table 8: Final-answer positive-score rate in the con-
trolled memory intervention. Intervals are 95% Wilson
confidence intervals.
Original outcome NOriginal memory
correctImproved memory
correctChange
(count; pp)
Memory-supported correct 104 104 88−16;−15.4
Memory-missing correct 77 0 48+48;+62.3
Answering error 47 47 29−18;−38.3
Unresolved error 272 0 59+59;+21.7
Total 500 151 224+73;+14.6
Table 9: Memory-only correctness after grouping ex-
amples by the original agent’s outcome. Change is
measured within each fixed row.
D.2 Final-answer binary correctness
Under the shared evaluation procedure, final-
answer binary correctness is 44.4% with improved
memory and 36.2% with original agent memory.
With empty memory, 14 of 500 final answers re-
ceive a positive official score. These rates describe
the controlled 500-example comparison and are not
a full-benchmark accuracy claim.
D.3 Memory-only results by original outcome
The original-memory column follows directly from
the fixed outcome labels: memory is answerable
in the memory-supported-correct and answering-
error cells, and unanswerable in the other two
cells. Improved memory makes 48 of the 77
memory-missing-correct examples and 59 of the
272 unresolved-error examples answerable. The
same replacement makes memory unanswerable
for 16 originally memory-supported-correct exam-
ples and 18 answering-error examples. The net
result is 73 more memory-answerable examples,
an increase from 151 to 224 out of 500. The inter-
vention therefore improves memory-only answer-
ability overall but does not dominate the original
memory in every outcome cell.
D.4 Artifact audit
All 500 improved memories include a
source_page citation to a page in the
corresponding gold evidence_pages set. This
audit verifies that every memory points to an
eligible evidence page. It does not verify that every
generated statement is entailed by that page or that
the memory contains all evidence needed for the
answer.We also apply a strict answer-string filter. The fil-
ter removes every example whose improved mem-
ory contains the normalized gold answer string
after whitespace and punctuation normalization. It
leaves 304 examples, of which the memory-only
reader receives a positive official score on 98, for
a positive-score rate of 98 / 304 = 32.2%. This re-
maining rate shows that exact answer-string copies
do not explain all successful memory-only answers.
Because the filter selects a different subset rather
than editing memory in place, its 32.2% rate is not
a like-for-like estimate of the full-set effect. The fil-
ter also cannot rule out paraphrased answer leakage
or establish claim-level grounding.
D.5 Answerability versus grounding
Memory-only answerability asks whether the fixed
reader can recover an accepted answer from the
question and terminal working memory alone.
Grounding asks whether the memory’s claims are
supported by the cited source pages. These proper-
ties are related but not equivalent. An answerable
memory may contain an unsupported guess or an
answer echo. A grounded memory may still omit
a needed fact, or the reader may fail to use it. The
audits check only source-page validity and exact
answer copying; they do not establish claim-level
grounding. Accordingly, the intervention shows
that changing memory content can improve answer-
ing under a fixed protocol; it does not show that
every improved memory is fully grounded.
D.6 Limitations
This single-seed intervention is an upper-bound
analysis on one fixed subset, reader, judge, and
improved-memory model: GPT-4o is stronger than
the original 4B memory writer and receives gold ev-
idence pages. Table 9 shows that improved memory
also hurts some outcome cells, and the improved
memories are used neither for training nor in the
main AWM diagnostic. Repeated runs and claim-
level audits are needed.
E Relation to Other Reliability Criteria
AWM treats terminal working memory as an
instance-level artifact. Predictive multiplicity in
knowledge-graph embeddings and disagreement
across repeated language-model queries expose
cases where aggregate performance hides query-
level variation (Zhu et al., 2024; Potyka et al.,
2024); AWM instead tests whether a fixed reader

can recover the answer after the pages and inter-
action trace are removed. Conformal methods in-
stead construct prediction sets or study coverage
under predicate and configuration shift (Zhu et al.,
2025c,a, 2026). We make no uncertainty or cover-
age claim; the output is a per-trajectory memory-
only answerability test.
Other studies modify or inspect the evidence-to-
reasoning path. EigentSearch-Q+ uses structured
tools and SCAIR uses schema-conditioned traver-
sal (Zhang et al., 2026; Chaturvedi et al., 2026).
Knowledge-graph RAG studies test missing facts,
while ArgRAG represents retrieved evidence as
supporting or attacking arguments (Zhou et al.,
2025a, 2026a; Zhu et al., 2025b). Related anal-
ysis also finds asymmetric responses for logically
equivalent facts under different pretraining frequen-
cies (He et al., 2025). AWM leaves the agent archi-
tecture and memory schema unchanged and evalu-
ates the source-linked memory they produce: the
memory passes only when the reader can answer
without the pages or the trajectory.
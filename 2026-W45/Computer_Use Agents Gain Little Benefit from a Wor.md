# Computer-Use Agents Gain Little Benefit from a World Model When Conditioned on Tutorials

**Authors**: Anh Nguyen, Tianyi Xu, Phat Tran, Harry Yu, Huazheng Wang, Stefan Lee

**Published**: 2026-11-02 23:36:18

**PDF URL**: [https://arxiv.org/pdf/2611.04073v1](https://arxiv.org/pdf/2611.04073v1)

## Abstract
Recent work has improved computer-use agents (CUAs) by combining retrieved task-relevant tutorials and look-ahead planning using world models with MLLM judges. However, the contribution of each component -- tutorials, the judge, and the world model -- to overall improvement has not been evaluated in isolation. To fill this gap, we first construct OSWorld-Tutorial and ScienceBoard-Tutorial: two datasets for CUAs with curated multimodal tutorials, covering everyday and scientific domains. We then propose RAG-PolicyJudge (RAG-PJ), a policy-judge framework for CUAs conditioned on task-specific tutorials: the policy proposes candidate actions, and the judge selects among them. To isolate the effect of world models given tutorials, we compare RAG-PJ with and without common world models: using either zero-shot text-based descriptions of next states or an oracle that returns ground-truth visual observations. We find that RAG-PJ gains little benefit from these rollouts when tutorials are present, even with perfect visual futures. This suggests prior world-model gains in tutorial-conditioned methods come mainly from tutorial understanding and the judge's ability to select the proposed actions, rather than from predicted rollouts. Error analysis reveals that the bottleneck is not rollout fidelity, but the judge's ability to reason over future states and recognize task completion. This highlights the need to improve not just next-state prediction in world models, but also reasoning over their predicted rollouts. Datasets and code are available at https://github.com/realjoenguyen/RAGPJ-official.

## Full Text


<!-- PDF content starts -->

Computer-Use Agents Gain Little Benefit from a World Model When
Conditioned on Tutorials
Anh (Joe) Nguyen Tianyi Xu*Phat Tran*Harry Yu*
Huazheng Wang Stefan Lee
Oregon State University
{nguyejoe,xut2,tranphat,yuhar,huazheng.wang,leestef}@oregonstate.edu
Abstract
Recent work has improved computer-use
agents (CUAs) by combining retrieved task-
relevant tutorials and look-ahead planning us-
ing world models with MLLM judges. How-
ever, the contribution of each component—
tutorials, the judge, and the world model—
to overall improvement has not been evalu-
ated in isolation. To fill this gap, we first
construct OSWorld-Tutorial and ScienceBoard-
Tutorial: two datasets for CUAs with curated
multimodal tutorials, covering everyday and
scientific domains. We then propose RAG-
PolicyJudge (RAG-PJ), a policy-judge frame-
work for CUAs conditioned on task-specific
tutorials: the policy proposes candidate actions,
and the judge selects among them. To isolate
the effect of world models given tutorials, we
compare RAG-PJ with and without common
world models: using either zero-shot text-based
descriptions of next states or an oracle that re-
turns ground-truth visual observations. We find
that RAG-PJ gains little benefit from these roll-
outs when tutorials are present,even with per-
fect visual futures. This suggests prior world-
model gains in tutorial-conditioned methods
come mainly from tutorial understanding and
the judge’s ability to select the proposed ac-
tions, rather than from predicted rollouts. Error
analysis reveals that the bottleneck is not roll-
out fidelity, but the judge’s ability to reason
over future states and recognize task comple-
tion. This highlights the need to improve not
just next-state prediction in world models, but
also reasoning over their predicted rollouts.1
1 Introduction
People rely on desktop and web software for day-
to-day tasks and scientific workflows. Computer-
use agents (CUAs) aim to automate these tasks by
*Equal contribution.
1Datasets and code are available at
github.com/realjoenguyen/RAGPJ-official.
Without tutorial With tutorial4045505560Success rate (%)47.449.450.657.4
51.557.7No rollout Text rollout Oracle visual rolloutFigure 1: Success rate on OSWorld-Tutorial (Qwen3-
VL-32B) of RAG-PJ, a policy–judge CUA, whose judge
sees no rollout, a predicted text rollout (R-WoM (Mei
et al., 2026)), or an oracle visual rollout (real future
screenshots). Rollouts help RAG-PJ without a tutorial
but add almost nothing with one. Full results in § 6 and
Appendix C.
interacting with computer interfaces given natural-
language user instructions. Modern CUAs are typi-
cally built on top of multimodal LLMs (MLLMs)
trained on static offline data. As software is up-
dated and user interfaces shift, these MLLM-based
CUAs may lack current procedural knowledge
about new environments, resulting in degraded per-
formance (Ishmam and Marino, 2026).
A natural approach to address this gap is to pro-
vide CUAs with up-to-date task-specific tutorials.
These readily available tutorials, contain environ-
ment descriptions and step-by-step instructions in
the form of text and screenshots.2
Recent work has integrated tutorial retrieval as
a component in CUA systems (Yang et al., 2026a;
Xu et al., 2025a). However, these systems do not
compare against alternative tutorial-use approaches,
leaving their effectiveness unclear. A parallel body
of work has considered test-time scaling for CUAs
through a policy-judge approach with look-ahead
planning (Luo et al., 2026; Chae et al., 2025).
2An example tutorial is given in Appendix G.
1
arXiv:2611.04073v1  [cs.LG]  2 Nov 2026

These methods use a world model to predict the
outcome of action candidates under a policy, over
one or many steps. They then use an MLLM-as-a-
judge to select the best action among the candidates,
conditioned on these predicted rollouts. Recently,
tutorial-augmented methods of this approach have
been introduced (Mei et al., 2026). However, these
methods conflate the contributions of the tutorial,
the judge, and world-model rollouts without proper
evaluation.
To fill this gap, we study the effect of tutorial
information on CUAs with and without look-ahead
planning, while retaining the MLLM-as-a-judge
framework. Intuitively, high-quality tutorials al-
ready provide guidance for the policy to generate
actions and demonstrations of their effects that can
be used by the judge. This can potentially reduce
the need for costly world model training and infer-
ence. With this motivation, we propose a simple
yet effective method: RAG-PolicyJudge (RAG-PJ),
a retrieval-augmented policy-judge framework for
CUAs. RAG-PJ provides a task-specific tutorial
to both (1) the policy, which proposes action can-
didates, and (2) the judge, which selects the best
action among them.
To evaluate tutorials for CUAs across do-
mains, we build curated tutorials for exist-
ing datasets: OSWorld-Verified for day-to-day
tasks (Xie et al., 2024) and ScienceBoard for scien-
tific workflows (Sun et al., 2026), resulting in two
new tutorial benchmarks: OSWorld-Tutorial and
ScienceBoard-Tutorial. The two original datasets
are widely adopted for CUAs, facilitating easy
adoption and reuse by the research community.
To study the effect of world models on policy-
judge frameworks in these two tutorial datasets,
using the same base MLLM for both the policy and
the judge, we compare RAG-PJ with two types
of world models for CUAs: text-based and vi-
sual. First, zero-shot text-based rollouts (Mei et al.,
2026) (i.e., the text description of what the pre-
dicted next state would look like). Second, oracle
visual rollouts (i.e., the real screenshot of the next
state). We argue that these oracle rollouts are an up-
per bound on the next-state fidelity a learned visual
world model (like ViMo (Luo et al., 2026)) could
achieve.
We find that RAG-PJ matches both rollout vari-
ants when tutorials are present. This indicates that
current zero-shot MLLM-as-a-judge models gain
little extra benefit from a world model, even with
oracle visual futures, when conditioned on tuto-rials (Figure 1). Analyzing failures of the oracle
rollout, we find that the agent’s failure tounder-
stand task success and progress, even when the
oracle future is available, is a primary bottleneck
to higher performance. This suggests that progress
in CUAs may require not just better world models,
but also stronger reasoning over the futures these
world models simulate.
In summary, our contributions are as follows:
–We introduce RAG-PJ, a simple yet effective
tutorial-conditioned policy-judge method for
CUAs (§ 4.2).
–We build OSWorld-Tutorial and ScienceBoard-
Tutorial, two curated tutorial datasets with exist-
ing tasks from OSWorld (Xie et al., 2024) and
ScienceBoard (Sun et al., 2026) (§ 3), facilitating
tutorial study for CUAs.
–We propose an oracle visual rollout infrastructure
for CUAs, providing an important ablation on
the effect of visual world-model rollouts with
perfect future information (§ 5, Appendix F).
–We provide a controlled comparison of tutorial-
only judging, text-based rollouts, and oracle vi-
sual rollouts-demonstrating that current MLLM-
as-a-judge models gain little benefit from world
models given task-specific tutorials (§ 6).
2 Related Work
World model evaluation for CUAs.Several
works find that zero-shot rollouts from pretrained
MLLMs used as world models (Ha and Schmidhu-
ber, 2018) are not yet faithful or consistent proxies
for real environments (Li et al., 2026; Xu et al.,
2026; Fu et al., 2026). Beyond MLLMs as world
models, other works use oracle world models to
study upper-bound look-ahead performance in em-
bodied tasks (Qian et al., 2026) and, at the first step
only, in computer-use tasks (Gonzalez-Pumariega
et al., 2026). We extend these analyses to (1) or-
acle image-based rollouts with task-specific tuto-
rials and (2) look-ahead planning at every step of
computer-use tasks.
Concurrent to our work, Tang et al. (2026) show
that code-based visual world model add little bene-
fit if agents can already have a well-trained MLLM-
as-a-judge. Our work complements theirs by study-
ing the benefit of world model rollouts against
zero-shot MLLM-as-judge, in the presence of task-
specific tutorials. Further, we study two types of
common world models: we adopt (1) text-based,
and build (2) visual ground truths, which is the
2

upper bound of current visual world models.
CUAs that use tutorials (or manuals).Prior
work uses tutorials either to create training data (Ou
et al., 2024; Xu et al., 2025b; Zhang et al., 2026,
2025b; Song et al., 2026; Lu et al., 2026) or as
guidance at inference time (Liu et al., 2026b; Xu
et al., 2025a; Agashe et al., 2025a; Mei et al., 2026;
Yang et al., 2026a), where the tutorial guides ac-
tion generation (policy) or selection (judge). RAG-
GUI (Xu et al., 2025a) and Agent S (Agashe et al.,
2025a) retrieve text-only tutorials to extract guid-
ance for policy-only agents.3Liu et al. (2026b) re-
trieve tutorial videos at inference time as in-context
guidance for the policy. Similar to our benchmarks,
there are two recent multimodal tutorial datasets
for CUA/GUI: WebOne Long et al. (2026) and Do-
cOS Liu et al. (2026a), where agents must search
the open web for documentation before executing a
task. However, WebOne is only for web agents and
tutorials are not task-conditioned, but rather than
software documentation in general. DocOS is not
publicly available, preventing direct comparison.
In contrast, our work focuses on task-specific mul-
timodal tutorials for general CUAs and test-time
scaling approach with a MLLM-as-a-judge, which
is evaluated against world model rollout methods.
We also release our benchmark and trained agents
for reproducibility.
Most similar to our approach are OS-
Symphony (Yang et al., 2026a) and R-WoM (Mei
et al., 2026). OS-Symphony couples retrieved
tutorials with a multi-agent history-reflection
policy, but does not compare against other
tutorial-using baselines. Our tutorial-conditioned
judge is orthogonal and could be combined with its
policy. Like R-WoM, we focus on settings where
multimodal task-specific tutorials are available, but
R-WoM’s results do not disentangle the tutorial,
the MLLM-as-a-judge, and the rollout, and it is
evaluated on a non-public subset of ∼85 OSWorld
tasks. R-WoM reports that its rollouts improve
over a no-tutorial baseline, but never compares
against a tutorial-conditioned judge without rollout.
We add that comparison with our method RAG-PJ,
and find the judge alone matches R-WoM; we
further test oracle visual rollouts. See Appendix D
for a more detailed discussion of related work.
3See Appendix D.3 for a comparison with RAG-GUI.3 Tutorial datasets
Tutorial dataset domains.To evaluate tutori-
als and tutorial-conditioned CUAs across domains,
we pairexisting taskswith newly collected tuto-
rials to build two datasets: (1) OSWorld-Tutorial,
from general-purpose desktop tasks in OSWorld-
Verified (Xie et al., 2024), and (2) ScienceBoard-
Tutorial, from scientific-software workflow tasks in
ScienceBoard (Sun et al., 2026). General-purpose
tasks from OSWorld-Verified reflect everyday com-
puter use, with abundant online tutorials. Scien-
tific software from ScienceBoard is harder on both
fronts. First, its applications are niche: step-by-step
task tutorials are scarce, and even ScienceBoard’s
annotators relied on official documentation (Ta-
ble 6 of Sun et al. (2026)). Second, current open
MLLMs still struggle on these tasks, with as low as
∼30% success rate with Qwen3.8-27B (Table 2).
This makes it a natural testbed for whether tutori-
als, even scarce ones, can compensate for limited
MLLM familiarity with these specialized domains.
Generating tutorials.We pair each task with
one multimodal tutorial (text and possibly images)
drawn from web sources. The tutorial type depends
on what is available online for that task.
First, human-verified online tutorials: we search
Google with the task instruction, and when a re-
trieved tutorial covers the task, one of three annota-
tors experienced with Ubuntu environments selects
the most helpful one and verifies it by following
its instructions to complete the task; we keep the
pair only when this succeeds. This accounts for
31% of OSWorld-Tutorial tasks and none of the
ScienceBoard-Tutorial tasks, as the latter’s scien-
tific software is niche and lacks task-level tutorials
online. These tutorials rule out retrieval errors, let-
ting us measure the effect of tutorials independently
of the retrieval pipeline.
Second, MLLM-generated tutorials: when no
single online tutorial covers the task (none exists,
or the task requires combining several), we launch
the task in a virtual machine and prompt Google
Gemini 3.8 Flash (Google DeepMind, 2026) with
the task instruction, the initial screenshot, and the
tutorial-generation prompt in Appendix H; Gem-
ini searches the web and writes a task-specific
tutorial, which we keep without human verifica-
tion. The screenshot lets it refer to the actual files,
columns, and menus of the task instead of inventing
names. MLLM-generated tutorials mirror practi-
cal real-world guidance: users rarely have a cu-
3

rated tutorial database and instead ask a search-
augmented MLLM for task-specific instructions.
MLLM-generated tutorials cover 69% of OSWorld-
Tutorial tasks and all ScienceBoard-Tutorial tasks.
Dataset statistics.We create total 529 tutorials
across both original datasets. OSWorld-Tutorial
contains 360 existing OSWorld-Verified tasks4:
110 with human-verified and 250 with MLLM-
generated tutorials (examples in Appendix H).
ScienceBoard-Tutorial pairs all 169 existing Sci-
enceBoard tasks with MLLM-generated tutorials
(examples in Appendix I).
4 Tutorial-conditioned CUAs
4.1 Preliminaries
CUA architectures.A CUA is a policy
π(at|o≤t, a<t, L, M) over screenshots o, actions
a, instruction L, and tutorial M. See Appendix J
for the full problem formulation. Typically, CUA
architectures are built on MLLMs. Designs are
either end-to-end, with one MLLM mapping inputs
directly to actions (Qin et al., 2025), or modular,
combining multiple MLLMs and specialized mod-
els (Agashe et al., 2025b; Yang et al., 2026a). We
follow the common two-stage modular design. A
policy π(ˆat|·)outputs a natural-language action
(e.g.,click(the red button) ). A grounding
model g(at|ˆat,·)then maps it to a low-level ac-
tion (e.g.,click(731,244)).
Judges and world models.Other work sam-
plesKaction candidates ˆa1
t, . . . ,ˆaK
t∼π(ˆa t|·)
and introduces a judge model that selects
the index of the best candidate, k∗=
j(ˆa1
t, . . . ,ˆaK
t|o≤t, a<t, L, M)∈ {1, . . . , K} ,
passing only the selected action ˆak∗
tto the ground-
ing model for execution (Yang et al., 2026c). This
sample-then-select paradigm has shown promising
results when coupled with look-ahead planning, us-
ing a world model to estimate the outcome of each
action over a fixed number of time steps. See Ap-
pendix D.1 for an overview of world model meth-
ods following this approach for CUAs. Each can-
didate is rolled out for Hsteps by alternating the
world model and the policy, and the judge selects
among candidates based on their rollouts (details in
Appendix J). However, look-ahead planning greatly
4OSWorld-Verified at commit 83e8534 (Jun 24, 2026),
excluding 9 tasks that require Google Drive credentials or are
blocked by website bot detection (Appendix H).increases computational cost by requiring multiple
policy and world-model executions per step.
4.2 RAG-PolicyJudge (RAG-PJ)
To test if the additional compute for world model
is valuable when strong policy guidance is avail-
able as a tutorial, we design a simple tutorial-
conditioned method, RAG-PolicyJudge (RAG-PJ),
shown in Figure 2. We adapt a two-stage GUI-
agent framework, consisting of a policy and a
grounding model as described above (§ 4.1). Im-
plementation details are in Appendix K.
Policy and Grounding Models.We instantiate
the policy π(ˆat|o≤t, a<t, L, M) as an MLLM that
outputs Kcandidate actions in a single call (Yang
et al., 2026c). To bound context length, in prac-
tice we condition πonˆo≤tin place of o≤t, where
ˆo≤tconsists of: the current observation ot; all
previous actions; the chain-of-thought of the five
most recent steps; and the screenshots of up to
five recent steps. Additionally, the task instruction,
CUA-specific policy rules, and the tutorial are al-
ways available at each time step. We adopt the
action space from Agent-S2 (Agashe et al., 2025b).
The policy prompt is available in Appendix L. We
adopt UI-Venus-1.5 (Venus Team et al., 2026) as it
is the state-of-the-art grounding model for CUAs
and has been used as the grounding model in prior
work (Gong et al., 2026; Jiang et al., 2025). The
grounding prompt is provided in Appendix N. The
grounding model g(at|ˆat, ot)accesses only the
current observationo tand the policy actionˆa t.
Judge Model.The judge jshares the policy’s
MLLM and inputs: Kcandidate actions, task in-
struction, the trajectory encoding ˆo≤t, and tutorials.
It ranks the candidates with a ranking prompt (Ap-
pendix M) and sends the top-ranked action to the
grounding model for execution.
5 Experimental Settings
Evaluation metric: task success rate.We use
the task success rate metrics defined by OSWorld-
Verified and ScienceBoard. For OSWorld-Verified,
the metrics are computed within the original max-
imum of15 steps(Xie et al., 2024; Kuntz et al.,
2025; Ye et al., 2025; Wang et al., 2025) and
the extended50-stepbudget adopted in follow-up
work (Abhyankar et al., 2026; OpenAI, 2025; Jia
et al., 2026). For ScienceBoard, we follow each
task’s time limit from the original benchmark (Sun
et al., 2026).
4

RAG-Policy-Judge (RAG-PJ)
User instructionCanyoumakeBingthemainsearchenginewhenIlookstuffupontheinternet?Current screenshot 
Trajectory history
𝑎!𝑎"⋮𝑎#$%ACTIONHISTORYK-STEP DETAILEDLOCAL HISTORY
𝑇!"#,𝑎!"#𝑇!"$,𝑎!"$𝑇!"%,𝑎!"%
POLICYThought𝑻𝒕:Ihavenavigatedtothe'Searchengine'settingspageinChrome.Thetutorialinstructsmetoselect'Change'underthe'Searchengine'section(Step4).Icanseea'Change'buttonnexttothecurrentdefaultsearchengine,whichisGoogle.Clickingthisbuttonshouldallowmetoselectanewdefaultsearchengine,suchasBing,asrequestedinthetaskinstruction.
Action 𝒂𝒕𝟐	:agent.hotkey(keys=['ctrl', 'shift', 's'])
Thought:Thetutorialinstructstoclick'Change'underthe'Searchengine'section(Step4)toselectanewdefaultsearchengine.Thecurrentscreenshotshowsthe'Change'buttonnexttothecurrentsearchengine(Google),whichisthecorrectnextsteptoproceedtowardsettingBingasthedefault.Clicking'Change'willopenalistofavailablesearchengines,includingBing,allowingmetoselectit.Thisactiondirectlyalignswiththetutorialandmovesthetaskforwardeffectively.Action 𝒂𝒕𝟎	:agent.click(instruction="the 'Manage search engines and site search' option below the current search engine setting", num_clicks=1, button_type='left', hold_keys=[])
Action 𝒂𝒕𝟏:agent.click(instruction="the 'Change' button next to the current search engine (Google) in the Search engine settings section", num_clicks=1, button_type='left', hold_keys=[])
GUI ENVIRONMENTS1. On your computer, open Chrome.4. Under "Search engine," select Change.2. At the top right, select MoreSettings.5. Select a new default search engine. . . .3. Select Search engine.
TUTORIAL
2JUDGE3OBSERVATION AT TIME t1
import pyautogui; import pyautogui; pyautogui.click(1275, 354, clicks=1,button='left');GROUNDING4Figure 2:Overview of RAG-PJ.Given
 the instruction, current screenshot, and trajectory history,
 a tutorial-
conditioned policy proposes candidate actions,
 a tutorial-conditioned judge selects one, and
 a grounding
model maps it to keyboard/mouse inputs.
Baselines.We use the following baselines with
the same two-stage policy-grounding frame-
work (Gou et al., 2025), the same harness
(rules/guidelines), and the same input/output space
as RAG-PJ, allowing us to isolate the effects of
the world model, the judge, and the tutorial. We
adopt the hyperparameters from their best publicly
reported configurations. Details are in Appendix K.
In all settings with a judge, the policy proposes
K= 3 candidate actions per step5and the judge
selects one; in Policy Only settings, the policy
proposes a single action.
–Policy Only drops the judge from RAG-PJ.
When including the tutorial, we denote this
model asPolicy Only + RAG.
–Policy + Judge (GTA-1) (Yang et al., 2026c)
is a policy-judge framework, without a world
model or tutorial.
–R-WoM H=# is a re-implementation of the recent
tutorial-augmented look-ahead method from
Mei et al. (2026) described in § 4.1. R-WoM
uses an MLLM-based world model to produce
zero-shot text-based next-state descriptions. We
setH=1,2,3 to adjust how many steps are
rolled out during look-ahead search. Our re-
implementation details are in Appendix O.6
5K= 3follows the original codebase of our baselines.
6Without tutorials, our re-implementation reproduces R-
WoM’s reported gain over a judge-only baseline, as shown–WebDreamer H=# (Gu et al., 2025) is a tutorial-
free, text-based world model look-ahead method
originally proposed for web agents, which we
adapt to CUAs. Similar to R-WoM, we use look-
ahead horizons H=1,2 . Our re-implementation
details are in Appendix P.
–(Visual) GT-rollout H=# : an oracle experi-
ment in which we use ground-truth screenshots
as future rollouts.7For each action candidate
at a given time step, we save and reset the
state of the virtual machine hosting the envi-
ronment before collecting each rollout. Stochas-
tic mismatches between GT-rollout and live
states are rare ( <1.0% of completed-task states).
These mismatches are mostly distracting web-
page changes (color, ads), not inherent environ-
ment changes that would require a different pol-
icy (Appendix F). Similar to R-WoM H=# , we use
H=1,2,3 look-ahead steps. Prior world-model
work (Chae et al., 2025; Gu et al., 2025; Luo
et al., 2026; Zheng et al., 2026; Cao et al., 2026;
Mei et al., 2026) predicts only one step ahead
in Table 3; we use this as a sanity check that the baseline is
functional before adding tutorials.
7The common usage of the term “world model” is to de-
scribe a predictor of the next state given a current state and
action. This is separate from planning that might use the world
model. In this sense, the visual rollout is extremely accurate
except for the stochastic elements described in Appendix S,
which are fundamentally unpredictable. We do not imply the
system as a whole is the best possible planning system.
5

MethodQwen3-VL-8B Qwen3-VL-32B Qwen3.8-27B
#15 steps 50 steps 15 steps 50 steps 15 steps 50 steps
Policy only
1Policy Only 29.0±0.934.5±1.142.5±0.744.9±2.057.4±2.665.7±0.3
2+ RAG 40.6±2.745.6±2.147.8±0.751.7±2.460.7±0.568.2±1.1
Policy + Judge
3Policy + Judge (GTA-1) 29.9±0.232.5±2.343.8±0.147.4±1.356.8±0.865.1±0.7
3aPolicy + RAG Judge 32.4±0.339.5±1.945.2±1.747.9±1.260.1±1.467.1±0.8
3bRAG Policy + Judge 40.5±3.145.7±1.950.2±1.455.4±0.161.3±1.070.9±0.8
4RAG-PJ (Ours) 42.2±0.4 46.8±1.9 51.7±1.4 57.4±3.6 61.7±0.5 71.0±1.2
Policy + RAG Judge + Text-based rollout (R-WoM Mei et al. (2026))
5R-WoMH=130.8±1.332.6±1.636.4±1.045.8±2.156.9±1.866.5±1.0
6R-WoMH=230.5±1.533.5±2.240.9±0.948.0±1.958.2±0.465.5±2.7
7R-WoMH=331.8±0.636.9±1.842.1±0.846.2±0.255.6±1.566.2±0.6
8+Policy RAGH=134.8±0.940.2±0.847.8±0.554.0±0.1 63.0±1.3 71.7±2.7
9+Policy RAGH=238.6±0.639.4±0.950.1±0.751.5±1.0 59.7±0.3 70.2±1.4
10+Policy RAGH=336.3±1.839.3±1.847.2±0.252.3±0.6 60.2±0.8 71.4±0.5
RAG Policy + RAG Judge + Ground-truth visual rollout
11H=1 40.7±0.8 43.1±4.0 49.4±0.2 57.5±1.8 62.8±0.9 70.7±1.0
12H=2 43.1±1.0 45.6±1.3 51.6±0.9 57.7±1.0 63.8±0.7 70.1±0.4
13H=3 41.4±1.5 47.5±2.5 50.5±1.1 58.0±0.7 63.6±1.4 72.3±0.6
Table 1: Success rate (%) of RAG-PJ (Ours) and baselines on 360 OSWorld-Tutorial tasks, at 15 and 50 steps. Mean
±sample std. over 3 completed runs. “RAG” indicates tutorial use. Shaded cells are not significantly different
from RAG-PJ (row 4) in the same column. See more details in Appendix A.
(rarely two); we evaluate up to three. Details
are in Appendix F.
Base MLLMs.We use Qwen3-VL-8B, Qwen3-
VL-32B, and Qwen3.8-27B (Bai et al., 2025; Qwen
Team, 2026)8as the base MLLMs for the policy,
judge, and world model in both our method and
the baselines. These models are commonly used
in recent CUA works (Xue et al., 2026; Yan et al.,
2025; Bai et al., 2026; Wang et al., 2026).
6 Results and findings
Given a task anda corresponding task-specific tu-
torial(either a human-verified online tutorial or an
MLLM-generated tutorial, which can be noisy), we
want to answer the following questions:
6.1 Can tutorials improve CUAs?
Finding 1: Tutorials improve task success rate
over non-tutorial baselines.In Table 1 (OSWorld-
Tutorial) and Table 2 (ScienceBoard-Tutorial), we
show that including tutorials in the policy ( row 2 )
improves over the non-tutorial baseline ( Policy
8Qwen3-VL-8B-Instruct, Qwen3-VL-32B-Instruct-FP8,
and Qwen3.8-27B-FP8 from the official Hugging Face ver-
sions of Qwen.Only ,row 1 ) by∼3–12% (Table 1) and ∼15–21%
(Table 2). The policy-judge framework with tu-
torials (RAG-PJ, row 4 ) improves over the non-
tutorial baseline ( GTA-1 ,row 3 ) by∼5–14% (Ta-
ble 1) and ∼18–22% (Table 2). These improve-
ments are larger on ScienceBoard (Table 2), whose
specialized scientific apps are likely less covered in
MLLM training data. This suggests tutorials help
most where MLLMs lack prior knowledge.
Table 7 (Appendix A) and Tables 8 and 9 (Ap-
pendix B) break results down per category in both
benchmarks; the improvement from tutorial condi-
tioning is consistent across all categories. Notably,
tutorial conditioning in the policy yields most of
the gains, suggesting that guidance for producing
action candidates is the main source of improve-
ment. Yet policy conditioning alone ( row 2 , Ta-
bles 1 and 2) does not account for all of RAG-PJ’s
gains, so judge action selection still helps. Finally,
rows 3a, 3bvs.row 4(Table 1) show that using
tutorials in both policy and judge performs best.
9Hierarchical bootstrap, sign-flip permutation, and
Wilcoxon signed-rank tests over tasks; p-values in Ap-
pendix B.
6

#MethodQwen3-VL
32BQwen3.8
27B
Policy only
1Policy Only 14.8±1.029.0±0.6
2+ RAG 36.1±0.644.2±1.4
Policy + Judge
3Policy + Judge (GTA-1) 18.3±2.430.4±1.2
4RAG-PJ (Ours) 40.2±0.6 48.3±0.7
Policy + RAG Judge + Text rollout (R-WoM)
5R-WoMH=118.5±0.930.6±2.4
6R-WoMH=220.3±2.532.7±1.2
7R-WoMH=320.5±0.730.6±0.7
8+Policy RAGH=135.3±0.942.6±1.6
9+Policy RAGH=237.5±1.443.2±0.6
10+Policy RAGH=333.7±2.644.8±0.3
RAG Policy + RAG Judge + Ground-truth visual rollout
11H=1 39.1±2.0 49.1±0.6
12H=2 40.4±0.7 49.3±0.3
13H=3 42.4±1.9 49.9±1.2
Table 2: Success rate (%) on ScienceBoard-Tutorial.
Mean±sample std. over 3 completed runs. “RAG”
indicates tutorial use. Shaded cells (ground-truth roll-
outs,rows 11–13 ) are not significantly different from
RAG-PJ (row 4).9
6.2 Do CUAs benefit from zero-shot
text-based world models?
Finding 2: Given a tutorial, CUAs gain no
benefit from zero-shot text-based world models.
As shown in Tables 1 (OSWorld-Tutorial) and 2
(ScienceBoard-Tutorial) and their breakdowns in
Appendix A, we find that R-WoM does not improve
over a judge without world model rollout ( RAG-PJ ).
R-WoM without policy-side tutorial conditioning
(rows 5–7 , Table 1) matches or underperforms the
Policy + RAG Judge without rollout ( row 3a ,
Table 1) at different horizons (H=1,2,3) . Ap-
pendix O.2 traces this to hallucination in rollouts.
When policy-side RAG is added ( rows 8–10 , Ta-
bles 1 and 2), R-WoM is worse than or statistically
similar to RAG-PJ ( row 4 ) on both benchmarks.
This indicates that gains from R-WoM come from
tutorial conditioning and the judge, rather than
text-based rollouts. Further, we observe no consis-
tent trend across rollout horizons: longer horizons
(H=2,3 ) do not always improve over H=1 , and
in some settings longer horizons slightly degrade
performance. This suggests compounding errors
across simulated steps add noise to the judge.#Method (no tutorial) 15 steps 50 steps
1Policy + Judge (GTA-1) 43.8±0.147.4±1.3
+ Text rollout, WebDreamer (Gu et al., 2025)
2H=141.4±1.4 (−2.4)47.0±2.0 (−0.4)
3H=240.1±1.0 (−3.7)47.5±1.1 (+0.1)
+ Text rollout, R-WoM (Mei et al., 2026)
4H=144.2±1.2 (+0.4)48.2±3.5 (+0.8)
5H=244.9±0.6 (+1.1)49.4±2.5 (+2.0)
+ Ground-truth visual rollout
6H=144.7±1.9 (+0.9)46.8±1.4 (−0.6)
7H=248.0±1.1 (+4.2)50.6±0.8 (+3.2)
Table 3: Success rate (%) on OSWorld-Tutorialwithout
tutorials(Qwen3-VL-32B policy and judge). Mean ±
sample std. over 3 completed runs. Parentheses: change
(pp) over Policy + Judge (row 1).
6.3 Do CUAs benefit from ground-truth visual
rollouts?
Motivated by the observation that text-based roll-
outs failed to improve over RAG-PJ, we consider
an oracle visual rollout ( GT-rollout ) to establish
an upper bound on world model performance.
Finding 3: Given oracle or noisy MLLM-
generated tutorials, zero-shot MLLM judges gain
little from ground-truth visual rollouts.In Table
1 (OSWorld-Tutorial) and Table 2 (ScienceBoard-
Tutorial), with tutorials, GT-rollout clearly im-
proves over text-based R-WoM (rows 11–13
vs.8–10 ) but is not significantly better than RAG-
PJ on either benchmark, for both human-verified
and MLLM-generated tutorials (Appendix A). As
before, the judge can already decide well from tu-
torials without reasoning over future consequences.
Without tutorials, GT-rollout does help: H=2
improves over Policy+Judge by∼3–4% (Table 3,
row 7vs.row 1).
With noisy MLLM-generated tutorials,
GT-rollout gives small nominal gains over
RAG-PJ in some settings (Tables 4 and 2). In 40
sampled tasks that RAG-PJ fails but GT-rollout
solves, the tutorials are partially correct but miss
steps.GT-rollouthelps the judge recover when:
–the horizon is too short for RAG-PJ to try the
alternative steps the tutorial suggests, e.g., Sci-
enceBoard categories with only 5 steps;
–the action’s effect is visually obvious in the next
state, e.g., opening a website or creating a file.
This matters more in ScienceBoard-Tutorial,
whose scientific software is less familiar to the
judge, so it is harder to infer the effect from the
7

misunder-
stands task
successjudge fails
future-state
reasoningfails tutorial
followinggrounding
errorcorrect action,
hits max stepstutorial
errorweb server /
connection
errorobservation
hallucinationtutorial
incomplete080160240320400480560640720Failed trajectories650
470
236 230
174
12499 92
411865155
86 71
12 1179
23Failure reasons count in Ground-truth world model
Primary reason Mentioned reasonFigure 3: Error analysis of 2,116 failed trajectories with ground-truth visual rollouts in OSWorld-Tutorial. Given
a failed task, we use Gemini 3.8 Flash to identify a primary failure reason and additional secondary reasons. We
categorize the failure reasons into 9 categories and show their frequency. For a detailed breakdown, see Appendix R.
tutorial text alone.
Per-category results are in Appendices A and B.
Finding 4: Agents fail to use ground-truth future
information due to poor understanding of task
success and progress.To understand why ground-
truth visual rollouts do not improve success, we
analyze failed tasks in OSWorld-Tutorial with the
help of Gemini 3.8 Flash (Google DeepMind, 2026)
and summarize failure reasons and their frequency
in Figure 3.
The most frequent failure category is the agent
(both the policy and the judge) failing to understand
task success: whether the current state satisfies the
task instruction. For instance, a task is not com-
pleted in the current state, yet the policy and the
judge predict that the current state already satis-
fies the instruction. The judge therefore selects the
action that ends the task prematurely.
The second most frequently tagged category is
the judge failing to reason over the future informa-
tion abouttask progress: what has been done and
what is left to do. Even if the policy generates good
action candidates, the judge fails to select the one
that leads to task progress. The judge often repeats
the same failed action instead of recognizing the
failure and choosing other candidates.
We show examples of these failures in Fig-
ures 28, 32, and 29. Overall, most failures stem
from the judge’s inability toreason over future
states generated by world models, not from world-
model quality. Examples of other less significant
failure reasons and the error-analysis implementa-
tion are shown in Appendix R.
Implication: CUAs need not only better world
models but also better mechanisms for reason-
ing over future rollouts.Improving world-model
fidelity alone may therefore be insufficient for look-
ahead planning. Current world-model methodsMethod15 steps 50 steps
#Human MLLM Human MLLM
Policy only
1Policy Only 60.2±3.463.8±1.4
2+ RAG 66.6±1.470.4±1.568.7±2.374.7±1.9
Policy + Judge
3Policy + Judge (GTA-1) 61.7±0.969.6±3.7
4RAG-PJ (Ours) 72.6±2.771.4±1.174.5±3.375.0±4.1
RAG-PJ + Ground-truth visual rollout
5H=166.3±0.968.9±0.972.6±3.671.4±6.0
6H=272.3±1.470.1±4.174.2±1.076.7±0.6
7H=370.4±2.972.2±4.275.3±1.375.5±3.7
Table 4: Success rate (%) on 110 OSWorld-Tutorial
tasks with human-verified (Human) vs. MLLM-
generated (MLLM) tutorials (Qwen3-VL-32B policy/-
judge, 15/50 steps). Tutorial-free baselines shown once.
Mean±sample std. over 3 completed runs.
for CUAs lackreward predictionfor rollout states,
which the judge needs to compare candidates by
their futures, and judges need better ways to reason
over rollouts, e.g., via prompting or training. This
is consistent with Qian et al. (2026), who find that
even a ground-truth world model does not improve
embodied agents, as using future information is the
main bottleneck.
6.4 Can current MLLM systems synthesize
tutorials for CUAsas good as
human-verified online tutorials?
Without an available online tutorial for a given
task, can MLLM systems with web search synthe-
size equally useful ones? Since Tables 5 and 6
(Appendix A) suggest that tasks with MLLM-
generated tutorials may be harder, we use OSWorld-
Tutorial to compare both tutorial typeson the
same tasks. Specifically, we select 110 tasks
with human-verified online tutorials. We use our
dataset pipeline (described in § 3) to create MLLM-
generated tutorials for these tasks. We then evaluate
8

both RAG-PJ and the ground-truth visual rollout
method under each tutorial condition and report the
results in Table 4.
Finding 5: Task-specific tutorials written by a
search-grounded frontier MLLM are as useful
to the policy and judge as human-verified online
tutorials.In Table 4, we find no statistically sig-
nificant difference between the two tutorial types,
in RAG-PJ ( row 4 ) or inGT-rollout (rows 5–7 ).
This is likely because OSWorld-Tutorial tasks in-
volve common software with abundant online tuto-
rials. We provide examples of MLLM-generated
vs. human-verified tutorial pairs and an analysis
of a sample of 20 pairs in Appendix E. In this
sample, a human verifier judges every MLLM-
generated tutorial as correct or partially correct,
and prefers it over its human-verified counterpart
in 17 of 20 pairs. MLLM-generated tutorials are
task-conditioned, naming the on-screen objects and
values the task requires, whereas human-verified
online tutorials describe an application feature in
general, leaving the CUA to ground each action on
the current screen.
7 Limitations
Our study has several limitations:
First, our MLLM-generated tutorials (Gemini
3.8 Flash with web search) cover noisyonline
retrieval, but we do not evaluateofflinesettings,
where a local retriever over a smaller corpus (e.g.,
local documentation) may yield noisier or missing
tutorials, which could change the relative benefit
of look-ahead planning.
Second, both benchmarks run desktop applica-
tions in Linux VMs. ScienceBoard-Tutorial has
only MLLM-generated tutorials, so our human
vs. MLLM tutorial comparison (Finding 5) cov-
ers only 110 general-purpose OSWorld-Tutorial
tasks.
Third, all models come from one MLLM family
(Qwen3-VL-8B, Qwen3-VL-32B, Qwen3.8-27B),
and all judges are zero-shot prompted; other model
families, larger frontier MLLMs, or judges trained
to reason over rollouts may use future states more
effectively.
Fourth, our error analysis (Figure 3) uses Gemini
3.8 Flash rather than human annotators to catego-
rize failures, so the categories may carry labeling
noise or MLLM biases.8 Ethical Considerations
Compute used.We used 272 GPU-hours in total
on NVIDIA H100/H200 GPUs, plus about 1,080
hours on an NVIDIA DGX Spark (GB10).
Risks.Like all assistive technologies, advances
in computer-use agents make it easier for individu-
als to take action—both good and ill. Given how
modern infrastructure relies on digital interfaces,
this may pose significant risks for personal digital
security, coordinated astroturfing, or model errors
causing users harm.
Acknowledgments
We thank everyone from the VIRL lab and
Huazheng Wang’s lab (Oregon State University).
We thank Yaolun Zhang, Andrew Brandenfels,
Artin Lahni, and Pranav Kulkarni for their help
with annotation, analysis, and discussion, and for
providing project resources. We thank Khanh
Nguyen for the brainstorming sessions that inspired
the first author to pursue ideas in computer-use
agents. We thank Kai Mei, the first author of R-
WoM (Mei et al., 2026), for detailed discussions of
R-WoM’s implementation.
The first author thanks Nguyen Thi Ngoc Anh,
Amanda Putiza, Ngo Thi Bich Lan, Tran Thanh
Nhu, Bui Thuy Tien, and Nguyen Hoang Kieu Anh
for their personal support. This work is supported
by NSF CAREER Award 2339676. We also thank
the anonymous reviewers for their valuable feed-
back and suggestions.
References
Reyna Abhyankar, Qi Qi, and Yiying Zhang. 2026.
OSWorld-human: Benchmarking the efficiency of
computer-use agents. InMLSys.
Saaket Agashe, Jiuzhou Han, Shuyu Gan, Jiachen Yang,
Ang Li, and Xin Eric Wang. 2025a. Agent S: An
open agentic framework that uses computers like a
human. InICLR.
Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang,
Ang Li, and Xin Eric Wang. 2025b. Agent S2: A
compositional generalist-specialist framework for
computer use agents. InCOLM.
Hao Bai, Alexey Taymanov, Tong Zhang, Aviral Kumar,
and Spencer Whitehead. 2026. WebGym: Scaling
training environments for long-horizon visual web
agents with realistic tasks. InCVPR.
Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen,
Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei
9

Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhi-
fang Guo, Qidong Huang, Jie Huang, Fei Huang,
Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng
Li, and 45 others. 2025. Qwen3-VL technical report.
arXiv preprint arXiv:2511.21631.
Yilin Cao, Yufeng Zhong, Zhixiong Zeng, Liming
Zheng, Jing Huang, Haibo Qiu, Peng Shi, Wenji
Mao, and Wan Guanglu. 2026. MobileDreamer: Gen-
erative sketch world model for GUI agent.arXiv
preprint arXiv:2601.04035.
Hyungjoo Chae, Namyoung Kim, Kai Tzu-Iunn Ong,
Minju Gwak, Gwanwoo Song, Jihoon Kim, Sungh-
wan Kim, Dongha Lee, and Jinyoung Yeo. 2025.
Web agents with world models: Learning and lever-
aging environment dynamics in web navigation. In
ICLR.
Dongping Chen, Ruoxi Chen, Shilin Zhang, Yinuo Liu,
Yaochen Wang, Huichi Zhou, Qihui Zhang, Yao Wan,
Pan Zhou, and Lichao Sun. 2024a. MLLM-as-a-
judge: Assessing multimodal LLM-as-a-judge with
vision-language benchmark. InICML.
Minghao Chen, Yihang Li, Yanting Yang, Shiyu Yu,
Binbin Lin, and Xiaofei He. 2024b. AutoManual:
Constructing instruction manuals by LLM agents via
interactive environmental learning. InNeurIPS.
Xiong-Hui Chen, Ziyan Wang, Yali Du, Shengyi Jiang,
Meng Fang, Yang Yu, and Jun Wang. 2024c. Pol-
icy learning from tutorial books via understanding,
rehearsing and introspecting. InNeurIPS.
Nicola Dainese, Pekka Marttinen, and Alexander Ilin.
2023. Reader: Model-based language-instructed rein-
forcement learning. InEMNLP, pages 16583–16599.
Lin Fu, Zheyuan Yang, Tianhui Zhang, Jinbiao Wei,
Guo Gan, Boxu Liu, Yilun Zhao, and Yu Rong. 2026.
GUI-CC: Benchmarking contextual consistency of
GUI world models as agent environments. InFind-
ings of EMNLP.
Yichen Gong, Zhuohan Cai, Sunhao Dai, Yuqi Zhou,
Zhangxuan Gu, Changhua Meng, and Shuheng Shen.
2026. VenusBench-Mobile: A challenging and user-
centric benchmark for mobile GUI agents with capa-
bility diagnostics. InICML, pages 35862–35887.
Gonzalo Gonzalez-Pumariega, Vincent Tu, Chih-Lun
Lee, Jiachen Yang, Ang Li, and Xin Eric Wang. 2026.
Scaling agents for computer use.Transactions on
Machine Learning Research (TMLR).
Google DeepMind. 2026. Gemini 3.8 Flash
model card. https://deepmind.google/models/
model-cards/gemini-3-8-flash/ . Accessed:
2026-09-11.
Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie,
Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su.
2025. Navigating the digital world as humans do:
Universal visual grounding for GUI agents. InICLR.Yu Gu, Kai Zhang, Yuting Ning, Boyuan Zheng, Boyu
Gou, Tianci Xue, Cheng Chang, Sanjari Srivastava,
Yanan Xie, Peng Qi, Huan Sun, and Yu Su. 2025. Is
your LLM secretly a world model of the internet?
model-based planning for web agents.Transactions
on Machine Learning Research (TMLR).
David Ha and Jürgen Schmidhuber. 2018. Recurrent
world models facilitate policy evolution. InNeurIPS.
Yu Han and Tianwen Qian. 2026. WM-R1: Train-
ing GUI agents to reason and leverage world mod-
els with reinforcement learning.arXiv preprint
arXiv:2608.27508.
Md Farhan Ishmam and Kenneth Marino. 2026. Time-
Warp: Evaluating web agents by revisiting the past.
arXiv preprint arXiv:2603.04949.
Hongrui Jia, Jitong Liao, Xi Zhang, Haiyang Xu, Tian-
bao Xie, Chaoya Jiang, Ming Yan, Si Liu, Wei Ye,
and Fei Huang. 2026. OSWorld-MCP: Benchmark-
ing MCP tool invocation in computer-use agents. In
ICLR.
Zhiyuan Jiang, Shenghao Xie, Wenyi Li, Wenqiang Zu,
Peihang Li, Jiahao Qiu, Siqi Pei, Lei Ma, Tiejun
Huang, Mengdi Wang, and Shilong Liu. 2025. Zoom
in, click out: Unlocking and evaluating the poten-
tial of zooming for GUI grounding.arXiv preprint
arXiv:2512.05941.
Thomas Kuntz, Agatha Duzan, Hao Zhao, Francesco
Croce, Zico Kolter, Nicolas Flammarion, and
Maksym Andriushchenko. 2025. OS-harm: A bench-
mark for measuring safety of computer use agents.
InNeurIPS.
Shufan Li, Konstantinos Kallidromitis, Akash Gokul,
Yusuke Kato, Kazuki Kozuka, and Aditya Grover.
2025. MobileWorldBench: Towards semantic
world modeling for mobile agents.arXiv preprint
arXiv:2512.14014.
Yixia Li, Hongru Wang, Jiahao Qiu, Zhenfei Yin, Dong-
dong Zhang, Cheng Qian, Zeping Li, Xiaoteng Ma,
Guanhua Chen, and Heng Ji. 2026. From word to
world: Can large language models be implicit text-
based world models? InACL, pages 8084–8111.
Jessy Lin, Yuqing Du, Olivia Watkins, Danijar Hafner,
Pieter Abbeel, Dan Klein, and Anca Dragan. 2024.
Learning to model the world with language. In
ICML.
Jingjing Liu, Ziye Huang, Zihao Cheng, Zeming Liu,
Jiahong Wu, Yuhang Guo, Kehai Chen, Yunhong
Wang, and Haifeng Wang. 2026a. DocOS: Towards
proactive document-guided actions in GUI agents. In
ICML.
Yujian Liu, Ze Wang, Hao Chen, Ximeng Sun, Xi-
aodong Yu, Jialian Wu, Jiang Liu, Emad Barsoum,
Zicheng Liu, and Shiyu Chang. 2026b. Learning
from online videos at inference time for computer-use
agents.Transactions on Machine Learning Research
(TMLR).
10

Xinwei Long, Kai Tian, Peng Xu, Weibo Gao, Yihua
Shao, Guoli Jia, Haozhe Geng, Sa Yang, Jingxuan Li,
Huayong Hu, Kaiyan Zhang, Jiaqi Wang, and Bowen
Zhou. 2026. Guiding the blind: Generalizing GUI
agents to unseen websites via multimodal tutorials.
InECCV.
Dunjie Lu, Yiheng Xu, Junli Wang, Haoyuan Wu,
Xinyuan Wang, Zekun Wang, Junlin Yang, Hongjin
Su, Jixuan Chen, Junda Chen, Yuchen Mao, Jingren
Zhou, Junyang Lin, Binyuan Hui, and Tao Yu. 2026.
VideoAgentTrek: Computer use pretraining from un-
labeled videos. InICLR.
Dezhao Luo, Bohan Tang, Kang Li, Georgios Pa-
poudakis, Jifei Song, Shaogang Gong, Jianye Hao,
Jun Wang, and Kun Shao. 2026. ViMo: A generative
visual GUI world model for app agents. InICLR.
Kai Mei, Jiang Guo, Shuaichen Chang, Mingwen
Dong, Dongkyu Lee, Xing Niu, and Jiarong Jiang.
2026. R-WoM: Retrieval-augmented world model
for computer-use agents. InICLR.
Anh Nguyen and Stefan Lee. 2025. Language-
conditioned world model improves policy general-
ization by reading environmental descriptions. In
NeurIPS LAW Workshop.
OpenAI. 2025. Introducing operator. https://openai.
com/index/introducing-operator/ . Accessed:
2026-05-15.
Tianyue Ou, Frank F Xu, Aman Madaan, Jiarui Liu,
Robert Lo, Abishek Sridhar, Sudipta Sengupta, Dan
Roth, Graham Neubig, and Shuyan Zhou. 2024.
Synatra: Turning indirect knowledge into direct
demonstrations for digital agents at scale. In
NeurIPS.
Cheng Qian, Emre Can Acikgoz, Bingxuan Li, Xiusi
Chen, Yuji Zhang, Bingxiang He, Qinyu Luo, Dilek
Hakkani-Tür, Gokhan Tur, Yunzhu Li, and Heng Ji.
2026. Current agents fail to leverage world model as
tool for foresight. InACL, pages 13686–13723.
Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang,
Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li,
Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye
Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu,
Xu Jiang, Qianli Ma, Jingyu Li, and 16 others. 2025.
UI-TARS: Pioneering automated GUI interaction
with native agents.arXiv preprint arXiv:2501.12326.
Qwen Team. 2026. Qwen3.8-27B model card.
https://huggingface.co/Qwen/Qwen3.8-27B .
Accessed: 2026-10-13.
Allen Z Ren, Bharat Govil, Tsung-Yen Yang, Karthik
Narasimhan, and Anirudha Majumdar. 2023. Lever-
aging language for accelerated learning of tool ma-
nipulation. InCoRL.
Chan Hee Song, Yiwen Song, Palash Goyal, Yu Su,
Oriana Riva, Hamid Palangi, and Tomas Pfister. 2026.
Watch and learn: Learning to use computers from
online videos. InCVPR.Qiushi Sun, Zhoumianze Liu, Chang Ma, Zichen Ding,
Fangzhi Xu, Zhangyue Yin, Haiteng Zhao, Zhenyu
Wu, Kanzhi Cheng, Zhaoyang Liu, Jianing Wang,
Qintong Li, Xiangru Tang, Tianbao Xie, Xiachong
Feng, Xiang Li, Ben Kao, Wenhai Wang, Biqing
Qi, and 2 others. 2026. ScienceBoard: Evaluating
multimodal autonomous agents in realistic scientific
workflows. InICLR.
Sizhe Tang, Hanhan Zhou, Jeffrey Svajlenko, and
Rashmi Gangadharaiah. 2026. Beyond next-screen
prediction: The decision value of GUI world models.
arXiv preprint arXiv:2610.24372.
Venus Team, Changlong Gao, Zhangxuan Gu, Yulin
Liu, Xinyu Qiu, Shuheng Shen, Yue Wen, Tianyu
Xia, Zhenyu Xu, Zhengwen Zeng, Beitong Zhou,
Xingran Zhou, Weizhi Chen, Sunhao Dai, Jingya
Dou, Yichen Gong, Yuan Guo, Zhenlin Guo, Feng
Li, and 8 others. 2026. UI-venus-1.5 technical report.
arXiv preprint arXiv:2602.09082.
Xinyuan Wang, Bowen Wang, Dunjie Lu, Junlin Yang,
Tianbao Xie, Junli Wang, Jiaqi Deng, Xiaole Guo,
Yiheng Xu, Chen Henry Wu, Zhennan Shen, Zhuokai
Li, Ryan Li, Xiaochuan Li, Junda Chen, Boyuan
Zheng, Peihang Li, Fangyu Lei, Ruisheng Cao, and
23 others. 2025. OpenCUA: Open foundations for
computer-use agents. InNeurIPS.
Yinjie Wang, Tianbao Xie, Ke Shen, Mengdi Wang, and
Ling Yang. 2026. RLAnything: Forge environment,
policy, and reward model in completely dynamic RL
system. InICML.
Jiannan Xiang, Yun Zhu, Lei Shu, Maria Wang, Lijun
Yu, Gabriel Barcik, James Lyon, Srinivas Sunkara,
and Jindong Chen. 2025. UISim: An interactive
image-based UI simulator for dynamic mobile envi-
ronments. InNeurIPS LAW Workshop.
Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan
Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhou-
jun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu,
Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming
Xiong, Victor Zhong, and Tao Yu. 2024. OSWorld:
Benchmarking multimodal agents for open-ended
tasks in real computer environments. InNeurIPS.
Ran Xu, Kaixin Ma, Wenhao Yu, Hongming Zhang,
Joyce C Ho, Carl Yang, and Dong Yu. 2025a.
Retrieval-augmented GUI agents with generative
guidelines. InEMNLP, pages 17866–17875. Associ-
ation for Computational Linguistics.
Weikai Xu, Kun Huang, Yunren Feng, Jiaxing Li,
Yuhan Chen, Yuxuan Liu, Zhizheng Jiang, Heng Qu,
Pengzhi Gao, Wei Liu, Jian Luan, Xiaolin Hu, and
Bo An. 2026. How mobile world model guides GUI
agents?arXiv preprint arXiv:2605.10347.
Yiheng Xu, Dunjie Lu, Zhennan Shen, Junli Wang,
Zekun Wang, Yuchen Mao, Caiming Xiong, and Tao
Yu. 2025b. AgentTrek: Agent trajectory synthesis
via guiding replay with web tutorials. InICLR.
11

Taofeng Xue, Chong Peng, Mianqiu Huang, Linsen
Guo, Tiancheng Han, Haozhe Wang, Jianing Wang,
Xiaocheng Zhang, Xin Yang, Dengchang Zhao, Jin-
rui Ding, Xiandi Ma, Yuchen Xie, Peng Pei, Xunliang
Cai, and Xipeng Qiu. 2026. EvoCUA: Evolving com-
puter use agents via learning from scalable synthetic
experience.arXiv preprint arXiv:2601.15876.
Tianci Xue, Weijian Qi, Tianneng Shi, Chan Hee Song,
Boyu Gou, Dawn Song, Huan Sun, and Yu Su. 2025.
An illusion of progress? assessing the current state
of web agents. InCOLM.
Haolong Yan, Jia Wang, Xin Huang, Yeqing Shen,
Ziyang Meng, Zhimin Fan, Kaijun Tan, Jin Gao,
Lieyu Shi, Mi Yang, Shiliang Yang, Zhirui Wang,
Brian Li, Kang An, Chenyang Li, Lei Lei, Meng-
meng Duan, Danxun Liang, Guodong Liu, et al.
2025. Step-GUI technical report.arXiv preprint
arXiv:2512.15431.
Bowen Yang, Kaiming Jin, Zhenyu Wu, Zhaoyang Liu,
Qiushi Sun, Zehao Li, JingJing Xie, Zhoumianze Liu,
Fangzhi Xu, Kanzhi Cheng, Yian Wang, Qingyun
Li, Yu Qiao, Zun Wang, and Zichen Ding. 2026a.
OS-Symphony: A holistic framework for robust and
generalist computer-using agents. InACL, pages
22300–22330.
Cheng Yang, Yifan Wu, Yutao Huang, Zhaohua Zhang,
Beiduo Chen, Muxi Chen, Chenchen Zhao, Hexuan
Deng, Haolin Yang, Geyuan Zhu, Sa Zhu, Jianhuan
Zhuo, Qiuyong Xiao, Jianhao Ruan, Yiran Peng, Jiayi
Zhang, Tian Ye, Xinlei Yu, Tianwen Jiang, and 2
others. 2026b. AutoGUIWorld: Image generators as
visual world models for GUI agent.arXiv preprint
arXiv:2610.01215.
Yan Yang, Dongxu Li, Yutong Dai, Yuhao Yang, Ziyang
Luo, Zirui Zhao, Zhiyuan Hu, Junzhe Huang, Am-
rita Saha, Zeyuan Chen, Ran Xu, Liyuan Pan, Sil-
vio Savarese, Caiming Xiong, and Junnan Li. 2026c.
GTA1: GUI test-time scaling agent. InICLR.
Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Jun-
yang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao,
Junjie Cao, Zhengxi Lu, Jitong Liao, Qi Zheng, Fei
Huang, Jingren Zhou, and Ming Yan. 2025. Mobile-
agent-v3: Fundamental agents for GUI automation.
arXiv preprint arXiv:2508.15144.
Alex Zhang, Khanh Nguyen, Jens Tuyls, Albert Lin, and
Karthik Narasimhan. 2024. Language-guided world
models: A model-based approach to AI control. In
ACL SpLU-RoboNLP Workshop.
Bofei Zhang, Zirui Shang, Zhi Gao, Wang Zhang, Rui
Xie, Xiaojian Ma, Tao Yuan, Xinxiao Wu, Song-
Chun Zhu, and Qing Li. 2026. TongUI: Internet-
scale trajectories from multimodal web tutorials for
generalized GUI agents. InAAAI.
Jian Zhang, Hanbo Zhang, Anxing Xiao, and David
Hsu. 2025a. Robot operation of home appliances by
reading user manuals. InCoRL.Ziyun Zhang, Xinyi Liu, Xiaoyi Zhang, Jun Wang,
Gang Chen, and Yan Lu. 2025b. UI-evol: Auto-
matic knowledge evolving for computer use agents.
InICML WCUA Workshop.
Yuhao Zheng, Li’an Zhong, Yi Wang, Rui Dai, Kaikui
Liu, Xiangxiang Chu, Linyuan Lv, Philip Torr, and
Kevin Qinghong Lin. 2026. Code2World: A GUI
world model via renderable code generation.arXiv
preprint arXiv:2602.09856.
Yuxin Zuo, Zikai Xiao, Li Sheng, Fei Huang, Jianhong
Tu, Yuxuan Liu, Tianyi Tang, Xiaomeng Hu, Yang
Su, Qingfeng Lan, Yantao Liu, Qin Zhu, Yinger
Zhang, Bowen Yu, Haiquan Zhao, Haiyang Xu,
Jianxin Yang, Jiayang Cheng, Junyang Wang, and
14 others. 2026. Qwen-AgentWorld: Language
world models for general agents.arXiv preprint
arXiv:2606.24597.
12

Appendix Contents
A Results on OSWorld-Tutorial per category of tutorials. . . . . . . . . . . . . . . . . . .14
B Results on ScienceBoard per application. . . . . . . . . . . . . . . . . . . . . . . . . .16
C Policy + Judge with visual rollout, without tutorials. . . . . . . . . . . . . . . . . . . .17
D Detailed related work. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .19
D.1 World models for CUAs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
D.2 Agents that use tutorials (manuals) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
D.3 Comparison with RAG-GUI . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
E Human-verified online tutorials vs. AI-generated tutorials. . . . . . . . . . . . . . . . .20
F Parallel Ground-Truth Rollouts. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .24
F.1 Parallel Warm-Rollout Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
G Example of a Retrieved Tutorial. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .29
H Building the OSWorld-Tutorial Dataset. . . . . . . . . . . . . . . . . . . . . . . . . . .29
I ScienceBoard-Tutorial Examples. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .34
J Problem formulation. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .35
K RAG-PJ implementation details. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .36
L Policy Prompt. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .37
M Judge Prompt. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .40
N Grounding Prompt. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .42
O R-WoM reimplementation. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .42
O.1 Implementation details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
O.2 R-WoM error analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
O.3 Policy prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
O.4 Rollout prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
O.5 R-WoM judge . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
P WebDreamer reimplementation. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .49
P.1 WebDreamer world model: state change . . . . . . . . . . . . . . . . . . . . . . . . . . 50
P.2 WebDreamer world model: imagined action proposal . . . . . . . . . . . . . . . . . . . 50
P.3 WebDreamer world model: imagined state change . . . . . . . . . . . . . . . . . . . . . 51
P.4 WebDreamer value function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
Q Visual Ground-Truth implementation details. . . . . . . . . . . . . . . . . . . . . . . .52
R Ground-Truth Error Analysis. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .56
S Example of stochastic dynamics in OSWorld environment. . . . . . . . . . . . . . . . .65
T GT-Rollout Failure Analysis Prompt. . . . . . . . . . . . . . . . . . . . . . . . . . . . .65
U LLM Usage Statement. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .68
13

A Results on OSWorld-Tutorial per category of tutorials
MethodQwen3-VL-8B Qwen3-VL-32B Qwen3.8-27B
#15 steps 50 steps 15 steps 50 steps 15 steps 50 steps
Policy only
1Policy Only 47.2%±0.9% 55.9%±0.5% 60.2%±3.4% 63.8%±1.4% 68.9%±3.1% 75.3%±1.0%
2+ RAG 59.3%±2.3% 61.5%±3.2% 66.6%±1.4% 68.7%±2.3% 73.2%±2.6% 76.2%±4.0%
Policy + Judge
3Policy + Judge (GTA-1) 44.7%±3.7% 50.8%±5.7% 61.7%±0.9% 69.6%±3.7% 67.1%±3.1% 76.5%±1.9%
4RAG-PJ (Ours) 61.6%±2.2% 62.1%±2.9% 72.6%±2.7% 74.5%±3.3% 75.1%±2.8% 79.0%±0.9%
Policy + RAG Judge + Text-based rollout (R-WoM Mei et al. (2026))
5R-WoMH=146.0%±1.8% 49.0%±1.8% 53.2%±1.9% 66.0%±3.5% 66.6%±1.4% 76.8%±1.3%
6R-WoMH=249.5%±1.2% 50.9%±7.1% 61.1%±1.4% 68.0%±5.0% 71.1%±1.1% 79.5%±0.5%
7R-WoMH=347.8%±0.5% 54.7%±6.2% 60.2%±1.4% 66.0%±1.9% 67.1%±2.4% 75.7%±3.7%
8+Policy RAGH=150.0%±2.4% 59.5%±3.9% 68.5%±3.7% 73.7%±1.8% 73.5%±0.9% 80.0%±1.5%
9+Policy RAGH=255.4%±2.4% 56.9%±1.9% 70.9%±1.6% 72.0%±2.8% 72.6%±2.7% 79.8%±1.6%
10+Policy RAGH=349.6%±1.3% 57.1%±1.7% 67.8%±1.9% 71.1%±3.8% 73.8%±2.7% 79.4%±1.3%
RAG Policy + RAG Judge + Ground-truth visual rollout
11H=1 59.5%±1.9% 60.1%±3.3% 66.3%±0.9% 72.6%±3.6% 75.3%±1.8% 80.2%±2.3%
12H=2 59.9%±2.7% 60.5%±2.3% 72.3%±1.4% 74.2%±1.0% 75.1%±1.4% 80.7%±0.8%
13H=3 60.5%±3.6% 65.8%±0.6% 70.4%±2.9% 75.3%±1.3% 76.0%±2.9% 81.7%±1.1%
Table 5: Success rate of RAG-PJ and other baselines on the 110 OSWorld-Tutorial tasks withhuman-verified
online tutorials, over maximum runtimes of 15 and 50 steps. Results are mean ±sample std. across 3 completed
runs. “RAG” indicates tutorial use. Shaded cells are not significantly different from RAG-PJ (row 4) in the same
column.
MethodQwen3-VL-8B Qwen3-VL-32B Qwen3.8-27B
#15 steps 50 steps 15 steps 50 steps 15 steps 50 steps
Policy only
1Policy Only 20.9%±0.9% 25.1%±1.4% 34.7%±1.8% 36.6%±2.4% 51.7%±3.1% 61.0%±0.2%
2+ RAG 32.4%±2.9% 38.6%±1.7% 39.5%±1.2% 44.3%±2.5% 54.5%±2.0% 64.3%±1.6%
Policy + Judge
3Policy + Judge (GTA-1) 23.3%±1.6% 24.5%±1.4% 35.8%±0.3% 37.7%±1.5% 51.6%±0.6% 59.5%±0.3%
4RAG-PJ (Ours) 33.7%±0.8% 40.1%±1.9% 42.5%±0.9% 49.9%±4.1% 55.1%±0.6% 67.0%±1.4%
Policy + RAG Judge + Text-based rollout (R-WoM Mei et al. (2026))
5R-WoMH=124.1%±2.6% 25.3%±1.9% 29.0%±0.9% 36.9%±3.1% 52.1%±3.1% 61.4%±1.2%
6R-WoMH=222.1%±1.9% 25.9%±1.8% 32.0%±1.4% 39.2%±4.3% 51.8%±1.2% 58.6%±3.8%
7R-WoMH=324.7%±0.8% 29.1%±0.2% 34.1%±1.1% 37.5%±0.5% 49.8%±1.1% 61.5%±1.6%
8+Policy RAGH=128.1%±0.4% 31.8%±1.2% 38.6%±1.1% 45.4%±0.7% 57.7%±1.7% 67.6%±3.3%
9+Policy RAGH=231.2%±0.7% 31.8%±1.8% 40.9%±1.3% 42.5%±0.7% 53.2%±1.1% 65.5%±2.6%
10+Policy RAGH=330.5%±2.4% 31.4%±1.8% 38.1%±1.0% 44.1%±1.5% 53.4%±0.4% 67.5%±1.1%
RAG Policy + RAG Judge + Ground-truth visual rollout
11H=1 32.4%±0.3% 35.7%±4.8% 42.0%±0.5% 50.8%±3.1% 56.5%±0.9% 65.9%±1.7%
12H=2 35.8%±1.0% 39.0%±1.4% 42.4%±0.8% 50.4%±1.8% 58.1%±1.8% 64.9%±0.4%
13H=3 33.0%±1.1% 39.2%±1.4% 41.8%±0.5% 50.4%±0.4% 57.5%±2.4% 67.2%±1.4%
Table 6: Success rate of RAG-PJ and other baselines on the 250 OSWorld-Tutorial tasks withAI-generated
tutorialsfrom tutorial_osworld over maximum runtimes of 15 and 50 steps. Results are mean ±sample std.
across 3 completed runs. “RAG” indicates tutorial use. Shaded cells are not significantly different from RAG-PJ
(row 4) in the same column.
14

MethodQwen3-VL-32B
#Overall OS Office Daily Professional Workflow
15 steps
1Policy only baseline 42.5%±0.7% 65.3%±4.8% 33.5%±1.8% 63.5%±4.7% 60.5%±1.2% 21.0%±3.3%
2Policy only + RAG policy 47.8%±0.7% 88.9%±2.4% 35.6%±3.5% 68.8%±3.4% 72.8%±1.2% 22.0%±2.1%
3RAG policy + RAG judge (RAG-PJ) 51.7%±1.4% 91.7%±4.2% 41.5%±2.0% 66.6%±2.7% 83.7%±2.0% 25.0%±2.2%
4GT rolloutH=149.4%±0.2% 87.5%±7.2% 39.9%±0.9% 66.2%±2.7% 72.1%±5.9% 25.7%±2.1%
5GT rolloutH=251.6%±0.9% 91.7%±4.2% 42.3%±0.7% 65.8%±4.6% 80.3%±4.2% 26.0%±2.4%
6GT rolloutH=350.5%±1.1% 91.7%±4.2% 39.7%±2.3% 66.4%±3.2% 81.6%±2.0% 24.0%±1.2%
50 steps
1Policy only baseline 44.9%±2.0% 73.6%±4.8% 32.4%±2.3% 63.9%±3.7% 68.0%±9.4% 25.4%±3.7%
2Policy only + RAG policy 51.7%±2.4% 88.9%±6.4% 39.5%±3.2% 72.2%±5.3% 73.5%±2.0% 29.2%±0.7%
3RAG policy + RAG judge (RAG-PJ) 57.4%±3.6% 95.8%±0.0% 48.7%±5.1% 72.3%±3.0% 81.6%±3.5% 33.5%±3.8%
4GT rolloutH=157.5%±1.8% 97.2%±2.4% 47.1%±4.4% 72.1%±0.7% 80.3%±5.1% 36.1%±4.1%
5GT rolloutH=257.7%±1.0% 94.4%±4.8% 47.7%±5.0% 73.0%±3.3% 83.0%±4.2% 34.6%±1.6%
6GT rolloutH=358.0%±0.7% 97.9%±2.9% 48.9%±4.4% 73.7%±7.3% 82.7%±1.4% 33.1%±1.6%
Table 7: Success rate per category on 360 OSWorld-Tutorial tasks over 15 and 50 steps, using Qwen3-VL-32B as
the policy and judge. Categories are OS, Office, Daily, Professional, and Workflow. Mean ±sample std. across 3
completed runs.
Statistical tests.We use Hierarchical bootstrap, sign-flip permutation, and Wilcoxon signed-rank tests
over tasks in Table 1, 6, and 5 to test whether ground-truth visual rollouts and R-WoM improve over
RAG-PJ on OSWorld-Tutorial.
Per-task-category analysis.To understand when and why the agent gains or loses a benefit from
ground-truth visual rollout (GT rollouts), we break down the results of Qwen3-VL-32B in Table 1 by
task category in Table 7. We inspected 40 cases where GT rollouts succeeded but RAG-PJ failed. In
these cases, candidate actions produced clear, visible changes in the next state, such as opening a new
website or creating a file. These visible effects make the correct action easier for the judge to spot, which
matters most when there is little time to recover from a mistake. These gains do not generalize to other
task categories. Applications like VS Code and GIMP have denser interfaces, so the effect of an action
is often less visually clear. In one failure case, the judge misread a VS Code workspace with unwanted
Explorer folders as a success—consistent with our earlier finding that the judge misinterprets task success
even from perfect future screenshots. The agent simply has more time to observe the consequences of its
actions and recover from mistakes without needing look-ahead.
15

B Results on ScienceBoard per application
#Method Celestia ChimeraX GrassGIS KAlgebra Lean TeXstudio Overall
(33) (29) (34) (31) (26) (16) (169)
Policy only
1Policy Only 20.2%±1.7%26.4%±2.0%8.8%±2.9%18.3%±1.9%0.0%±0.0%12.5%±0.0%14.8%±1.0%
2+ RAG 31.3%±3.5%74.7%±2.0%22.5%±1.7%62.4%±3.7%0.0%±0.0%12.5%±0.0%36.1%±0.6%
Policy + Judge
3Policy + Judge (GTA-1) 25.3%±1.7%31.0%±6.0%10.8%±4.5%26.9%±10.4%0.0%±0.0%10.4%±3.6%18.3%±2.4%
4RAG-PJ (Ours) 28.3%±1.7%74.7%±5.3%24.5%±1.7%73.1%±1.9%15.4%±0.0%12.5%±0.0%40.2%±0.6%
Policy + RAG Judge + Text rollout (Mei et al., 2026)
5R-WoMH=121.2%±0.0%25.3%±4.0%14.7%±0.0%34.4%±8.1%0.0%±0.0%8.3%±7.2%18.5%±0.9%
6R-WoMH=224.2%±0.0%34.5%±6.9%16.7%±1.7%30.1%±7.4%0.0%±0.0%8.3%±3.6%20.3%±2.5%
7R-WoMH=325.3%±1.7%29.9%±8.7%14.7%±2.9%34.4%±1.9%0.0%±0.0%12.5%±0.0%20.5%±0.7%
8+Policy RAGH=127.3%±3.0%71.3%±2.0%27.5%±1.7%62.4%±1.9%0.0%±0.0%8.3%±3.6%35.3%±0.9%
9+Policy RAGH=232.3%±3.5%70.1%±2.0%25.5%±3.4%69.9%±1.9%1.3%±2.2%10.4%±3.6%37.5%±1.4%
10+Policy RAGH=331.3%±3.5%66.7%±5.3%18.6%±9.0%61.3%±3.2%2.6%±2.2%8.3%±3.6%33.7%±2.6%
RAG Policy + RAG Judge + Ground-truth visual rollout
11H=134.3%±4.6%73.6%±2.0%27.5%±4.5%71.0%±5.6%1.3%±2.2%10.4%±3.6%39.1%±2.0%
12H=241.4%±6.3%73.6%±5.3%30.4%±1.7%68.8%±3.7%0.0%±0.0%10.4%±3.6%40.4%±0.7%
13H=340.4%±3.5%75.9%±3.4%32.4%±2.9%72.0%±6.7%3.8%±3.8%12.5%±0.0%42.4%±1.9%
Table 8: Per-application success rate on the 169 ScienceBoard tasks with theQwen3-VL-32Bpolicy (15 steps).
Mean±sample std. over 3 completed runs; task counts in parentheses. “RAG” marks tutorial use.
#Method Celestia ChimeraX GrassGIS KAlgebra Lean TeXstudio Overall
(33) (29) (34) (31) (26) (16) (169)
Policy only
1Policy Only 19.2%±4.6%56.3%±2.0%27.5%±4.5%46.2%±1.9%2.6%±2.2%12.5%±0.0%29.0%±0.6%
2+ RAG 34.3%±3.5%79.3%±3.4%29.4%±2.9%79.6%±1.9%14.1%±2.2%12.5%±0.0%44.2%±1.4%
Policy + Judge
3Policy + Judge (GTA-1) 25.3%±1.7%62.1%±3.4%27.5%±6.1%40.9%±4.9%3.8%±3.8%12.5%±0.0%30.4%±1.2%
4RAG-PJ (Ours) 41.4%±3.5%80.5%±2.0%31.4%±4.5%79.6%±1.9%28.2%±2.2%12.5%±0.0%48.3%±0.7%
Policy + RAG Judge + Text rollout (Mei et al., 2026)
5R-WoMH=127.3%±3.0%67.8%±4.0%23.5%±7.8%41.9%±8.5%0.0%±0.0%12.5%±0.0%30.6%±2.4%
6R-WoMH=228.3%±1.7%60.9%±5.3%34.3%±6.1%46.2%±11.3%1.3%±2.2%12.5%±0.0%32.7%±1.2%
7R-WoMH=323.2%±1.7%58.6%±0.0%27.5%±4.5%48.4%±3.2%2.6%±2.2%12.5%±0.0%30.6%±0.7%
8+Policy RAGH=131.3%±4.6%73.6%±5.3%30.4%±1.7%77.4%±3.2%15.4%±3.8%12.5%±0.0%42.6%±1.6%
9+Policy RAGH=230.3%±5.2%81.6%±4.0%31.4%±4.5%77.4%±5.6%10.3%±4.4%12.5%±0.0%43.2%±0.6%
10+Policy RAGH=338.4%±4.6%81.6%±2.0%27.5%±4.5%77.4%±0.0%15.4%±3.8%12.5%±0.0%44.8%±0.3%
RAG Policy + RAG Judge + Ground-truth visual rollout
11H=145.5%±6.1%77.0%±2.0%36.3%±4.5%83.9%±0.0%20.5%±2.2%12.5%±0.0%49.1%±0.6%
12H=254.5%±0.0%79.3%±3.4%32.4%±0.0%80.6%±0.0%16.7%±2.2%12.5%±0.0%49.3%±0.3%
13H=348.5%±0.0%79.3%±0.0%39.2%±4.5%81.7%±4.9%17.9%±2.2%12.5%±0.0%49.9%±1.2%
Table 9: Per-application success rate on the 169 ScienceBoard tasks with theQwen3.8-27Bpolicy (15 steps). Mean
±sample std. over 3 completed runs; task counts in parentheses. “RAG” marks tutorial use.
Significance tests.We test whether ground-truth visual rollouts improve over RAG-PJ on Science-
Board (Table 2, rows 11–13 vs.row 4 ) with task-level tests. For Qwen3-VL-32B ( H=1/2/3 ): hier-
archical bootstrap p=0.62/0.97/0.32 , sign-flip permutation p=0.58/1.00/0.24 , Wilcoxon signed-rank
p=0.51/0.99/0.20 . For Qwen3.8-27B ( H=1/2/3 ): sign-flip permutation p=0.20/0.20/0.11 , Wilcoxon
signed-rank p=0.20/0.15/0.14 . Welch’s t-test on run-level totals gives p=0.43/0.73/0.18 (32B) and
16

p=0.013/0.001/0.052 (27B), but it ignores per-task variation, so we do not treat the 27B differences as
significant.
Why do GT rollouts help in Celestia and GrassGIS?GT visual rollouts improve performance over
RAG-PJ in Celestia and GrassGIS across the evaluated horizons. We sample 40 cases where GT rollouts
succeeded but RAG-PJ failed. The common reason is that the tutorials given to agents are only partially
correct. They omit steps that the task needs in the current application state, and they present options as
equivalent when only some of them work.
For example, in Celestia task B-02 (Figure 4) the instruction is“Go to Neptune in Celestia. ”RAG-PJ
executes Method 1 of the retrieved tutorial, and tries to follow the highlighted step in its tutorial. In that
step, the tutorial proposes three options to travel to Neptune: pressing G, right-clicking, or double-clicking.
RAG-PJ then executes the first option, pressing G, but it fails to travel to Neptune. The reason is that its
tutorial fails to include the step that gives the 3D viewport keyboard focus (by clicking on it), right before
the step that instructs the user to press Gto travel to Neptune. Therefore, it fails to do that step and run out
of the 5-step budget of that task, before trying the other options.
GT-rollout insteads executes all three options in the tutorial on a copy of the VM, and sees that only the
right-click option works, so it chooses that one and completes the task in 4 steps. Details of three different
outcomes of the three options are shown in Figure 5.
C Policy + Judge with visual rollout, without tutorials
0204060SR (%)15 steps
+4.2−0.1
Without tutorial With tutorial0204060SR (%)50 steps
+3.2+0.3rollout bars: lighter =H=1, darker =H=2Policy + Judge (GTA-1)
RAG-PJ (ours)
WebDreamer (text-based VM)R-WoM (text-based WM)
R-WoM + RAG policy
GT rollout
Figure 6: Success rate with vs. without tutorials (Qwen3-VL-32B). Green arrows: gap from the judge-only reference
(dashed) to the best rollout method. Without tutorials, rollout beats the judge by up to 4.2%; with tutorials, the gap
nearly vanishes. Numbers from Tables 3 and 1.
We note down the differences between WebDreamer and R-WoM, which are both text-based world model
look-ahead methods. While WebDreamer and R-WoM both use greedy policy in rollout states, they
17

Task instruction:Go to Neptune in Celestia.
Tutorial:
Method 1: Using the Celestial Browser (Visible Panel)
1.In theCelestial Browserpanel on the left (under theSolar Systemtab), click onNeptunein the list under Solar
System Barycenter→Solto select it.
2.Press theGkeyonyour keyboard (orright -click Neptune inthelistandselectGoto,ordouble-click Neptune )to
travel totheplanet.
Method 2: Using Direct Planet Number Shortcut
1. Click on the main 3D star viewport to ensure it has focus.
2. Press the number key8on your keyboard to select Neptune (the 8th planet from the Sun).
3. Press theGkey to travel directly to Neptune.
Method 3: Using the Target Name Bar
1. Click on the main 3D star viewport and pressEnter(Return) to bring up the target entry prompt.
2. TypeNeptuneand pressEnterto confirm the selection.
3. Press theGkey to travel to Neptune.
Method 4: Using the Navigation Menu
1. ClickNavigationin the top menu bar.
2. ClickGo to Object...(orSelect Object..., typeNeptune, and confirm).
3. In the dialog window, ensure the target is set toNeptuneand click theGo to(orOK) button to begin navigation.
Figure 4: Celestia task B-02 (5-step budget): the task instruction, the initial screenshot, and the retrieved tutorial.
Method 1 omits the step that gives the 3D viewport keyboard focus, so after Neptune is clicked in the list, pressing G
or double-clicking does not travel to Neptune.
(a) Press G: the list jumps to Gonggong;
the view stays on Earth.
(b) Double-click: the list expands Nep-
tune’s moons; the view stays on Earth.
(c) Right-click: Celestia’s object menu,
withGoto.
Figure 5: The three alternatives the tutorial gives for one step of “Go to Neptune” (Qwen3.8-27B), each executed by
GT rollout on a copy of the VM after Neptune is selected in the list. RAG-PJ executed (a) for real.
18

differ in how rollouts are generated and scored. In WebDreamer (Gu et al., 2025), for each candidate
action, the rollout is generated many times with different imagined trajectories. Each trajectory is scored
independently, either complete (1), on track (0.5), or incorrect (0). The scores of these trajectories are
averaged to produce a score for that candidate action. In R-WoM (Mei et al., 2026), only one imagined
trajectory is generated for each candidate action. All candidate actions are then ranked listwise based on
the single trajectory score, and the best candidate action is selected.
D Detailed related work
D.1 World models for CUAs
Recent work explores world models (Ha and Schmidhuber, 2018) for CUA planning, which predict
future states in one of two forms: (1) textual descriptions of the next state (Chae et al., 2025; Li et al.,
2025; Cao et al., 2026; Mei et al., 2026; Zuo et al., 2026), or (2) visual predictions of the next state,
either by generating images directly (Luo et al., 2026; Xiang et al., 2025) or by generating code that
renders them (Zheng et al., 2026; Han and Qian, 2026). Most recently, Yang et al. (2026b) show that
pretrained image generators can act as visual world models, iteratively editing screenshots to synthesize
action-grounded training trajectories without deploying the software. In this work, we evaluate whether
CUAs with an MLLM-as-a-judge can leverage both forms of world models: text-based rollouts and oracle
visual rollouts, an upper bound for visual world models, when given access to task-relevant tutorials.
D.2 Agents that use tutorials (manuals)
Tutorials or manuals can describe both step-by-step instructions (what to do) and how an environment
changes over time (thedynamics of the environment). Many works study how agents can use tutorials
in different domains, including (1) games (Nguyen and Lee, 2025; Zhang et al., 2024; Lin et al., 2024;
Dainese et al., 2023; Chen et al., 2024c), (2) robotics (Zhang et al., 2025a; Ren et al., 2023), and (3)
embodied environments (Chen et al., 2024b). While our work targets the Computer-use Agent (CUA)
domain, some of these works share a similar goal: using tutorials at inference time to help agents handle
new tasks or environments. In particular, our work is closely related to (Nguyen and Lee, 2025; Zhang
et al., 2024; Dainese et al., 2023). However, these works focus on symbolic 2D environments, where
language is restricted to a single sentence per entity—typically 3–5 sentences total—and covers only
environmental dynamics. Meanwhile, we focus on real-world GUI environments with longer tutorials that
cover both task instructions and environmental dynamics.
D.3 Comparison with RAG-GUI
While RAG-GUI (Xu et al., 2025a) shares with us the same motivation of using tutorials for GUI agents
at inference time, we highlight several key differences between RAG-GUI and our method (RAG-PJ):
•RAG-GUI discards visual information in tutorials.RAG-GUI uses a lightweight VLM to judge
whether each retrieved multimodal tutorial is relevant and then generates a “concise textual guidance
summary” for the VLM. This removes visual information that might be necessary for GUI agents to
complete a given task, e.g., clicking on an icon of a button shown in the tutorial. Meanwhile, our
method (RAG-PJ) uses raw multimodal tutorials and directly feeds them to the LLM-as-judge to
judge the policy’s action, thus potentially taking advantage of visual clues in tutorials.
•RAG-GUI is challenging to reproduce and compare.RAG-GUI does not publish its implementation
or its dataset (tutorials). Further, it requires extensive training. This makes it challenging for us to
directly compare our method (RAG-PJ) with RAG-GUI. Meanwhile, one of our contributions is
that we publish our dataset and implementation to foster further research in CUA / GUI agents with
tutorials.
•RAG-GUI does not have tutorial-used baselines.RAG-GUI does not compare with other GUI / CUA
methods that use tutorials. Thus, it is hard to know if RAG-GUI’s design choice is the best way to
use tutorials. Meanwhile, we compare against other tutorial-using methods that involve planning
ahead with world models.
19

•RAG-GUI focuses on retrieving tutorials, whereas we focus on how to use a given tutorial.RAG-GUI
develops a retrieval module to obtain the most relevant tutorials for a given task. Meanwhile, we
do not focus on the retrieval part, and offload this retrieval role to frontier LLMs acting as an
LLM-augmented retriever, such as Gemini. Retrieval-focused methods like RAG-GUI would be
more suitable for CUA settings that involve local software / domain knowledge, where the agent
needs a retrieval module that can run locally. Meanwhile, our method (RAG-PJ) is more relevant to
general-purpose CUA that can access the internet and frontier LLMs.
E Human-verified online tutorials vs. AI-generated tutorials
One author manually reviewed 20 pairs of a human-verified online tutorial and an AI-generated tutorial
written for the same task. We drew the tasks at random with a fixed seed from the 110 tasks with both
tutorials in Section 6.4. For each pair, the reviewer read the instruction, the post-setup screenshot and
both tutorials, judged whether following each tutorial as written would complete the task, and chose the
more useful one. Table 10 reports the verdicts.
Human-verified online AI-generated
Correct 15 18
Partially correct 4 2
Wrong 0 0
Preferred by the reviewer 3 17
Table 10: Verdicts on 20 random tutorial pairs. A tutorial is correct if following it as written completes the task.
Task-specific AI tutorials.The reviewer judged 18 of the 20 AI-generated tutorials correct and preferred
them for 17 tasks. Because the generator sees the instruction and the screenshot, its tutorials name the
objects on screen, such as the three colored text boxes on a slide or the file 1.png on the desktop, and use
the values the task asks for. They also include the final steps that commit the change, such as confirming a
dialog or saving the file. The online tutorials are mostly correct but address a general reader. Five leave it
to the reader to map the steps onto the task, four miss a required step such as entering Master View in
Figure 7, and one explains double spacing instead of separating sentences, shown in Figure 9.
Errors in AI tutorials.Two AI-generated tutorials miss a required step. The VLC tutorial in Figure 8
never restarts VLC, and one VS Code alternative stops after a single ENTERin the command palette.
Every AI-generated tutorial in the sample offers at least two methods, so the median length is 252 words
against 147 for the online tutorials. Most remaining errors are in these extra methods, for example a menu
entry from another LibreOffice version or a file name guessed from a truncated label.
20

Task instruction:I am preparing a PPT in Libreoffice impress. The slide number is barely visible to me. Please
help me change the color of the slide number to red?
Post-setup screenshot supplied to the tutorial generator.
Human-verified online tutorial:partially correct
Source: Super User answer
1. Right click on the textbox containing the number, select Character.
2. On the Fonts tab, select the font size.
3.On the Font Effets tab, on the Font Color section, select the color. Click OK.
Reviewer:Never enters Master View, where the slide number is set. In Normal
view the field may not be selectable.AI-generated tutorial:correct
Source: Gemini 3.8 Flash with Google Search, prompt in Figure 13
Method 1: Change Slide Number Color via Master Slide View (Recommended)
1.Enter Master Slide View:In the top menu bar, clickView>Master Slide
(or click theMaster Viewbutton located in the right-handProperties
sidebar under theSlidesection).
2.Select the Slide Number Field:In the slide canvas, locate the slide number
placeholder box in the bottom-right corner (labeled<number>). Click to
select the<number>text box (or double-click to highlight the<number>
text).
3.Set the Font Color to Red:Right-click inside the selected<number>box
and chooseCharacter...from the context menu (or clickFormat>
Character...in the top menu). In theCharacterdialog window, switch to
theFont Effectstab. Under theFont colorsection, open the color palette
dropdown and selectRed. ClickOKto apply the color.
4.Exit Master View:Click theClose Master Viewbutton on the floating
Master View toolbar, or go to the top menu bar and selectView>Normal.
Method 2: Change Slide Number Color via the Formatting Toolbar in Master
View
1.Enter Master Slide View:ClickView>Master Slidefrom the top menu
(or clickMaster Viewunder thePropertiespanel on the right sidebar).
2.Select the Slide Number Field:Click on the <number> text box located at
the bottom-right of the master slide. Highlight the text <number> inside the
box.
3.Apply Red Font Color:On the topText Formattingtoolbar (or under the
Charactersection in the right sidebar), click the drop-down arrow next to
theFont Coloricon (represented by the letterAwith a colored bar
underneath). SelectRedfrom the palette.
4.Exit Master View:ClickClose Master Viewon the toolbar, or navigate to
View>Normal.
Reviewer:Enters and leaves Master View.
Figure 7: Impress task ac9bb6cb-1888-43ab-81e4-a98a547918cd . The online tutorial never enters Master View,
where the slide number is set.
21

Task instruction:I like watching movies (using VLC) on my laptop and sometimes the volume is too low for my
taste even when the volume in VLC is set to the maximum of 125% on the volume control. Can you increase the
max volume of the video to the 200% of the original volume?
Post-setup screenshot supplied to the tutorial generator.
Human-verified online tutorial:correct
Source: Super User answer
How can I increase the maximum volume output by VLC?
Increase the Maximum Volume to 200%
To increase the audio output beyond VLC’s default settings, follow these steps:
Change the Maximum Volume Displayed Setting
1. PressCtrl+Por selectTools→Preferences.
2. Select theAllradio button underShow settingsat the bottom of the
preferences window.
3. Expand theInterfacetree.
4. ExpandMain interfaces.
5. SelectQtand change the default value ofMaximum Volume displayed
from 125 to 200.
6. ClickSave.
7. Restart VLC.
Now, VLC should allow you to increase the volume to 200%.
Reviewer:Complete, including the restart.AI-generated tutorial:partially correct
Source: Gemini 3.8 Flash with Google Search, prompt in Figure 13
Method 1: Change Maximum Volume in VLC Preferences (Recommended)
1. In the open VLC media player window, click onToolsin the top menu bar
and selectPreferences(or pressCtrl + P).
2. In the bottom-left corner of the Preferences dialog, locateShow settings
and select theAllradio button to display advanced settings.
3. In the left navigation tree: scroll down to theInterfacecategory; click the
arrow next toMain interfacesto expand it; click onQt.
4. In the right pane, scroll down to the bottom to find theMaximum Volume
displayedsetting.
5. Change the value in the input box from125to200.
6.Click theSavebutton in the bottom-right corner of the Preferences window
to apply the change.
Method 2: Update the VLC Configuration File via Terminal
1. Open the Terminal application (or pressCtrl + Alt + T).
2.Run the following command to update the qt-max-volume setting to 200in
the VLC configuration file:
sed -i ’s/^#*qt-max-volume=.*/qt-max-volume=200/’
~/.config/vlc/vlcrc
3. If the setting does not already exist in the file, append it by running:
grep -q "^qt-max-volume=" ~/.config/vlc/vlcrc || echo
"qt-max-volume=200" >> ~/.config/vlc/vlcrc
Reviewer:Never restarts VLC, so the new maximum does not apply. Method 2
editsvlcrcwhile VLC runs, and VLC may overwrite it on exit.
Figure 8: VLC task 9195653c-f4aa-453d-aa95-787f6ccfaae9 . Only the online tutorial restarts VLC, which the
new maximum needs.
22

Task instruction:I am making a guideline for students of my course and would like to separate each sentence in
the the first paragraph to improve readability. Please separate each sentence by creating one empty line space
after each of them, as I am having hard time separating them one by one.
Post-setup screenshot supplied to the tutorial generator.
Human-verified online tutorial:wrong
Source: Super User answer
How can I double space a document in LibreOffice?
1. Select the paragraph
2. Right click and select “Paragraph”
3. Select “Indents & Spacing”
4. Set “Line Spacing” to “Double” and click “OK”
5. Change “Below paragraph” as well if you want extra spacingbetween
paragraphs.
Before:
After:
Reviewer:Double spacing does not put each sentence on its own line, so the task
fails.AI-generated tutorial:correct
Source: Gemini 3.8 Flash with Google Search, prompt in Figure 13
Method 1: Automated Search and Replace (Using Regular Expressions)
1.Highlight the First Paragraph:Click and drag your mouse across the
entire first paragraph (from“The purpose of the tutorial... ”to“...proactive
and talkative!”) to select it.
2.Open the Find and Replace Dialog:PressCtrl + H(or clickEditin the
top menu bar and selectFind and Replace...).
3.Configure Search Settings:If not already expanded, clickOther options
at the bottom of the dialog. Check the box forCurrent selection only.
Check the box forRegular expressions.
4.Replace Sentence Endings: Handle periods:In theFindbox, enter:\.␣
(note the trailing space after the dot). In theReplacebox, enter:.\n\n.
ClickReplace All.Handle exclamation marks:In theFindbox, enter: !␣
(note the trailing space after the exclamation mark). In theReplacebox,
enter:!\n\n. ClickReplace All.
5.Close and Save:Click theClosebutton (or pressEsc) to close the dialog.
Press Ctrl + S (or click theSavefloppy disk icon on the standard toolbar)
to save your changes.
Method 2: Manual Line Break Separation
1. In the first paragraph, locate the end of each sentence.
2. For each sentence boundary: click directly between the punctuation mark
and the following space (or delete the space between the sentences). Press
Enter twice to move the next sentence down and create an empty line space
between them. Repeat this after each of the following sentences: after
...related to our lectures.; after...develop your own ideas.;
after...then you really understand it. ; after...you don’t! ; after
...understood in class.; after...participation and
contribution.
3.Verify that each of the individual sentences sits on its own line separated by
a blank line space.
4. PressCtrl + Sto save the document.
Reviewer:Correct, and the sentence endings it lists are visible in the screenshot.
Its 24 search queries also include
"88fe4b2d-3040-4c70-9a70-546a47764b48.json",
site:github.com/xlang-ai/OSWorld "libreoffice_writer"and
"CCCH9003_Tutorial_guidelines_Gold.docx" test.
Figure 9: Writer task 88fe4b2d-3040-4c70-9a70-546a47764b48 . The online tutorial only sets double spacing.
The AI-generated tutorial is correct, but its generator searched for the task’s files.
23

F Parallel Ground-Truth Rollouts
At each interaction step, the policy proposes Kcandidate first actions {a(1)
k}K
k=1. Rather than asking a
learned world model to predict their effects, we evaluate every candidate in a separate virtual machine
(VM) restored to the agent’s current state. Within branch k, the first action is executed and the rollout
may continue for a horizon of up to Hreal environment transitions. Later actions are conditioned on the
observations produced within that branch. This gives a grounded trajectory
τk=
a(h)
k, o(k)
t+hHk
h=1,1≤H k≤H,
where Hkmay be shorter than Hif the branch terminates early. A multimodal judge compares the Kreal
trajectories and selects
k⋆= arg max
k∈{1,...,K}J(ot, τk).
Only the first action a(1)
k⋆from the selected trajectory is then executed on the main VM. Rollout VMs
are speculative branches: their states provide evidence for selection but never directly replace the live
environment state. Of 237,709 completed-task states, 2,298 (0.97%) are GT-rollout mismatches. We
manually label about 40 sampled mismatches: around 70% are webpage changes, and the remaining
∼30% are program execution differences. For example, installing Spotify can take 4s in the live VM but
>5s in the rollout VM. Examples are shown in Appendix S.
We execute the Kbranches concurrently and retain their VM processes in a warm pool. The first
decision step pays the cost of launching the rollout VMs. On later steps, the same processes are refreshed
from the newest main-VM checkpoint and reused. This design preserves the fidelity of ground-truth
execution while amortizing VM startup cost across agent steps.
Figure 11 expands the state-copy operation used before branch execution. The system does not copy
a running main VM independently Ktimes. It first captures one consistent checkpoint Ct= (M t, Dt)
with a QEMU snapshot-save job, where Mtcontains CPU, memory, and device state and Dtcontains
persistent disk state. Dtis stored as an internal qcow2 snapshot inside the main VM’s disk image, and Mt
is written to a separate vmstate image in host shared memory ( /dev/shm ). This checkpoint is then treated
as an immutable source. In parallel, branch kreceives a private copy of the machine state M(k)
tand a
private full copy of the disk imageD(k)
t:
C(k)
t=
M(k)
t, D(k)
t
,
restore
C(k)
t
=st.
The disk is copied in full rather than as a thin overlay on a shared base, because the snapshot load restores
every writable qcow2 device in the VM, so each branch disk must carry the main disk’s internal snapshot
table itself. The private copies ensure that changes made while evaluating one action cannot affect another
branch. Each package is attached to an already-running warm VM process: the paused VM’s active disk
is switched to its branch copy at the block layer, and the vmstate copy is attached as a temporary block
device. A QMP snapshot-load (loadvm ) operation stops that VM, loads the branch’s private checkpoint
state, and resumes the same process. The temporary vmstate source is then detached and deleted, guest
services are allowed to recover until the screen is stable, and the resulting State 0 is compared with that of
the other branches before any candidate action is executed. Thus, warming reuses the VM process but
replaces its complete guest execution state.
24

Main VM
current state st
Common checkpoint (State 0)
K-ACTION POLICY
propose {a(1)
k}K
k=1
WARM POOL
Rollout VM 1
restore stRollout VM 2
restore stRollout VM K
restore st···a(1)
1a(1)
2a(1)
K
State 1:o(1)
t+1
State 1:o(2)
t+1
State 1:o(K)
t+1···only ifH >1SINGLE-ACTION POLICY— re-invoked independently inside each branch k
H<t, ot, 
a(i)
k, o(k)
t+ih−1
i=17−→a(h)
kforh= 2, . . . , H
StateH:o(1)
t+H
StateH:o(2)
t+H
StateH:o(K)
t+H···...a(h)
1...a(h)
2...a(h)
K
JUDGE
select k⋆from{τk}K
k=1
Commit only a(1)
k⋆
on main VM
Main VM
next state st+1History
H<t
Figure 10: Parallel warm ground-truth rollout with horizon H. All candidate VMs restore the same pre-action
checkpoint and generate real H-step trajectories concurrently. The judge compares these trajectories, while only the
first action of the selected branch is committed to the canonical main VM. Vertically stacked insets separate rollout
depth from branch-level parallelism: the upper row shows State 1; each branch then re-invokes the same policy
model, prompted for a single action, on the history, ot, and that branch’s own actions and observations so far to
produce the next action, repeating this process through State H; this within-branch policy call occurs only when
H >1 , marked by the amber region. Both the K-action policy and JUDGE additionally condition on the interaction
history H<talongside the current state. Warm-pool VM processes are refreshed from the newest checkpoint and
reused rather than relaunched. Insets come from an actual rollout with K= 3 andH= 2 ; the construction supports
arbitraryKandH. The dashed return path indicates process reuse after state refresh at the next decision step.
25

Main VM
live state st
CHECKPOINT ONCE
capture a consistent state
Ct: immutable fan-out source
Mt
CPU, memory, devicesDt
disk + internal snapshot
parallel fan-out
private M(1)
t
private full disk
copyD(1)
tprivate M(2)
t
private full disk
copyD(2)
tprivate M(K)
t
private full disk
copyD(K)
tBranch package1 Branch package2 Branch packageK
···
PARALLEL WARM REFRESH
apply independently for k= 1, . . . , K
WARM POOL
WARM VM k
reuse existing QEMU process
Swap in disk D(k)
t, attach M(k)
t
snapshot-load
stop, restore, resume
Detach temporary vmstate source
Recover guest services (stable screen)
checked State 0
s(k)
t≈s t
READINESS BARRIER
continue only after all Kbranches are ready
Begin candidate execution
Figure 11: Parallel fan-out of the current state st. A single consistent checkpoint Ctis captured once and then
treated as an immutable source (navy region). Its machine state Mtand its disk image Dt, which carries the internal
snapshot, are both copied privately into each of the Kbranch packages; branches therefore share no mutable state.
The dispatch band applies one warm-restore protocol independently to every branch k; the teal region draws that
protocol once rather than repeating it Ktimes: the warm VM process is reused, its active disk is switched to the
branch copy, the vmstate copy is attached and loaded via a QMP snapshot-load (loadvm ), the temporary source
is detached, and guest services recover into State 0, which is compared across branches by perceptual hash. A
readiness barrier ensures candidate execution begins only after every one of the Kbranches has been restored; if
any branch fails, the pool is discarded and theKVMs are cold-launched.
26

F.1 Parallel Warm-Rollout Implementation
Initial pool construction.At a decision step, the system checkpoints the main VM before any candidate
action is committed. The checkpoint is a QEMU snapshot-save : disk state goes into an internal qcow2
snapshot of the main disk, and CPU, memory, and device state go into an external vmstate image on host
shared memory. Snapshot creation runs on a background thread and overlaps policy inference. If the
policy returns a single unique candidate, no rollout or judge is run. For the first parallel rollout, the system
copies the disk image and the vmstate image once per candidate and cold-launches one rollout VM per
candidate from its private copy. Each rollout VM therefore begins with the same disk, memory, device,
and interface state. The system captures a settled pre-action screenshot for every candidate and executes
the corresponding first grounded action. It then continues each branch on the same VM for up to Hsteps,
conditioning later actions on that branch’s actions and observations and recording the settled state after
every transition. A branch stops early when the policy emits a terminal action ( done ,fail , oranswer ),
when its action cannot be generated or grounded, or when no screenshot can be captured; a terminal first
action is not executed. Instead of terminating these VMs after selection, the system transfers them to the
warm pool for reuse.
Warm refresh.At every later decision step, all candidates must start from the new live state. The
refresh is prepared in the background while the policy runs, and its protocol is:
1. checkpoint the main VM and pause the warm rollout VMs;
2. makeKfull copies of the main disk image and of the vmstate image, one per branch;
3.switch each paused warm VM’s active disk to its branch copy at the block layer, attach its vmstate
copy, and restore it with a QMPsnapshot-loadin the same QEMU process;
4.detach and delete the vmstate copy, verify the active disk, resume and recover each rollout VM until
its screen is stable, and compare the pre-action states across branches; and
5.execute the Kbranch trajectories concurrently, with transitions within each branch proceeding
sequentially for up toHsteps.
Refreshing an existing pool avoids repeated VM process creation and guest bootstrapping. The refresh
still replaces each branch’s complete disk and execution state, ensuring that no disk or memory changes
from a previous candidate survive into the next decision step. After a restore, the guest network state is
repaired, since the restored TCP connections are stale.
Snapshot isolation.Each branch receives an independent full copy of the disk image and of the vmstate
image. Thin copy-on-write overlays on a shared base are not used: the snapshot load restores every
writable qcow2 device in the VM, so every branch disk must carry the main disk’s internal snapshot table.
Sharing mutable snapshot state between branches can also introduce interference: restoring or modifying
one candidate may alter another candidate that has already been restored. Per-candidate copies eliminate
this dependency and allow all restores and action executions to proceed concurrently. All copies are kept
in host shared memory ( /dev/shm ), and each vmstate copy is deleted as soon as its VM has loaded it.
Before a run, a resource preflight checks that host memory and shared-memory space can hold the main
VMs, the Krollout VMs per worker, and their copies; before each copy, a host-wide admission check
confirms that enough free space remains.
Selection and state invariants.Before action execution, every rollout VM should recover to the same
State 0 as the checkpointed main VM. The implementation captures stable State 0 images and compares
each candidate’s State 0 with that of the first candidate by perceptual (dHash) distance; mismatches are
logged and recorded with the trajectory. If the warm refresh fails for any branch, because of a failed disk
copy, snapshot load, disk verification, or guest recovery, or because Kchanged, the pool is discarded and
theKrollout VMs are cold-launched. The judge receives each high-level candidate together with its real
rollout trajectory: the later actions and the screenshots of State 1 through State Hk. While the judge runs,
27

the main VM is reloaded to State 0 in parallel. After it selects k⋆, the grounded first action a(1)
k⋆from the
selected rollout is dispatched to the live environment without re-grounding; later actions of the selected
branch are discarded, and the next step proposes new candidates from scratch. The warm pool persists
across the steps of a task until the candidate count changes, recovery fails, the task ends, or the worker
shuts down.
Resource accounting and recorded evidence.For Wtask workers, the method requires Wmain VMs
andWK rollout VMs, together with their checkpoint copies. Increasing either WorKtherefore raises
memory and shared-memory use substantially. For every step, the trajectory stores the candidate actions,
grounded actions, rollout actions and thoughts, State 0 screenshots, and per-horizon rollout screenshots.
These artifacts permit visual inspection that all branches began from a common state and that the judge’s
evidence corresponds to actual VM execution.
28

G Example of a Retrieved Tutorial
Figure 12 shows a Chrome task with a retrieved web tutorial.
Task instruction.Can you make Bing the main search engine when I look stuff up on the internet?
Tutorial: Set your default search engine
1. On your computer, open Chrome.
2. At the top right, select More
 Settings.
3. Select Search engine.
4. Under “Search engine,” select Change.
5. Select a new default search engine.
Figure 12: A Chrome task example with a retrieved web tutorial. The tutorial provides updated step-by-step
browser instructions with inline UI icons.
H Building the OSWorld-Tutorial Dataset
Never search for or use this benchmark's own materials: task IDs, task JSON files, evaluator/
checker code, expected answers, or published solutions (for example the OSWorld or ScienceBoard
GitHub repositories, Hugging Face datasets, or solution/trajectory repos). Write the tutorial
only from the application's own documentation and general knowledge.
Never call the task infeasible unless the application's documentation confirms the required
feature is missing. Do not decide feasibility from the screenshot: data, layers, menus and files
not visible in it may still exist.
Before writing the tutorial, use the google_search tool aggressively. Perform at least 8 distinct
Google Search queries before answering. Use separate searches for current UI/menu names,
settings page names, keyboard shortcuts, official help or documentation, Ubuntu/Linux behavior
when relevant, and task-specific caveats. Do not stop after one search if more verification
queries are available. After searching, output only the requested tutorial instructions.
Give detailed instructions for completing the given task in Ubuntu (<ubuntu version>) with <
program> <version>, assuming the given start state steps are already complete. If the application
is already open with the task's files loaded, do not launch, relaunch or reopen it.
Output only the instructions, with no questions or extra text. You should not solve the task: you
are providing instructions for solving the task. Give one primary method, the one you are most
sure of, as short numbered steps; never lead with a guess. Add at most one short alternative. Do
not close anything unless it is instructed by user. Change only what the task asks; keep every
other line, name and setting as it is.
Rules for the tutorial:
- <step budget>The primary method ends as soon as the task's goal is reached: no steps the task
does not ask for, such as compiling, previewing, running or verifying.
- Never invent names of files, maps, layers, columns, icons or records. If a name is not in the
task or the screenshot, the first step reads it from the application; later steps call it "the
item found in step 1".
- If the application has a command line, the primary method is the single exact command, using
only options from the official documentation.
- Every typed input must be complete and valid as entered, with exact syntax, separators and
formats. Write needed content (citation entry, formula, code, value) out in full instead of
getting it from a website.
- Cover the whole object the task names, not one branch, half or special case; prefer a form that
represents all of it (for example an implicit equation). Read ambiguous wording with its
standard technical meaning.
- If the task asks a question, the last step gives the answer found, in the format the task asks
for (for example`ANS 42`when answers are submitted with ANS).
- For tasks about dates or times, give generic steps that do not depend on today's date. For
multi_app tasks, you may install new tools and apps.
29

Before answering, cross-check every quoted variable, path, or placeholder in the task instruction
against the attached screenshot, and check that the end state of your steps matches the task's
target exactly, including sign, direction, units and values.
Treat the attached screenshot as the raw, native-resolution source of truth; do not rely on a
reduced, resized, or max-pixel version. For spreadsheet and table tasks, explicitly verify every
relevant column letter, header, source range, destination range, and example cell reference
against the screenshot before writing instructions. Map the task's quoted column names to the
visible column letters yourself. Never shift a source or destination column based only on the
wording of the task. Before outputting any formula or copy step, confirm that it reads from the
visible source column and writes to the visible destination column without overwriting the source
or touching irrelevant cells.
If the task is infeasible (cannot be done with <program> alone), say "Infeasible." followed by a
brief explanation of why. Do NOT suggest workarounds, alternative tools, or command-line
solutions --- just state why it is infeasible and stop.
Task:
<task instruction>
<the first screenshot>
Figure 13: Tutorial-generation prompt used to retrieve or synthesize task-specific tutorials. The same prompt is
used for OSWorld and ScienceBoard tasks. The placeholders are replaced with the program category, task
instruction, and initial task screenshot from each task. For "multi_apps" tasks,<version> lists the versions of all
applications in Table 11; ScienceBoard versions are in Table 12. <step budget> is “The agent has only Nactions
in total; the primary method must fit in that.” for ScienceBoard tasks with step budget N, and empty for OSWorld.
Table 11: Software versions supplied to the tutorial-generation prompt for OSWorld tasks.
Program Version
Ubuntu22.04.3 LTS
Google Chrome130.0.6723.58
GIMP2.10.30
LibreOffice (Calc, Impress, Writer)7.3.7.2
Mozilla Thunderbird115.6.0
VLC3.0.16
Visual Studio Code1.85.1
Multi-app tasks all of the above
Table 12: Software versions supplied to the tutorial-generation prompt for ScienceBoard tasks.
Program Version
Ubuntu22.04 LTS
UCSF ChimeraX1.9with thechimerax-states 0.5bundle
GRASS GIS8.4.1(wxPython GUI)
Celestia1.7-dev(Qt6 GUI)
KAlgebra KDE Education, Qt 6 build
TeXstudio Ubuntu 22.04 PPA build
Lean Lean 4v4.14.0in Visual Studio Code with the Lean 4 extension
Overview.To robustly evaluate the impact of tutorials, we construct the OSWorld-Tutorial dataset by
augmenting 360 tasks from the OSWorld-Verified benchmark (Xie et al., 2024).10We recruit three student
annotators with experience in software tasks similar to those in OSWorld to collect a tutorial for each task.
For each task, we first launch it in a virtual machine and provide the task instruction together with the
initial screenshot to the tutorial-generation prompt shown in Figure 13. We submit this prompt to Google
Gemini 3.8 Flash (Google DeepMind, 2026). It searches the web for relevant tutorials and synthesizes a
comprehensive summary over the retrieved results. The annotators then review the retrieved tutorials and
10We use OSWorld-Verified at commit83e8534.
30

select the most helpful one for each task. To verify tutorials from publicly available articles, they follow
the article’s instructions to complete the task and confirm that it is correct and helpful.
We find that many OSWorld tasks require combining multiple tutorials, while others have no relevant
tutorials available online. For these tasks, we use the response generated by Google Gemini 3.8 Flash
directly as the tutorial.We do not verify the correctness of AI-synthesized tutorials.
Figures 14 and 15 show examples of retrieved publicly available tutorials and MLLM-generated
tutorials. Each figure shows task instruction, initial screenshot, and the corresponding tutorial content.
Statistics.In total, we collect 360 tasks with verified tutorials, covering all eight OSWorld task categories.
Among these, 110 tasks have human-verified tutorials retrieved from the web (31%), while the remaining
69% are synthesized by Google Gemini 3.8 Flash. We follow Agent-S3 (Gonzalez-Pumariega et al., 2026)
and OSWorld-MPC (Jia et al., 2026) to exclude eight Google Drive tasks. Recently, these tasks have
presented technical challenges in using Google accounts and Google Drive credentials in the sandbox. We
notice that some web tasks have just been upgraded to use more aggressive bot detection systems, which
block automated access even through residential proxies provided by OSWorld. We therefore exclude one
task that is blocked by website-level bot-detection or access restrictions, as listed by task ID in Figure 16.
# Excluded task IDs – one per line, comments start with #
# tripadvisor.com – "Access is temporarily restricted" behavioral bot detection.
b7895e80-f4d1-4648-bee0-4eb45a6f1fa8
Figure 16: Excluded task IDs for websites that block automated access through bot detection, CAPTCHAs, or
access-denial systems.
31

Task instruction. Enable the “Do Not Track” feature in Chrome to enhance online privacy.
Turn “Do Not Track” on or off
When browsing the web on computers or Android devices, users can send a request to websites not to collect or track
browsing data. The setting is turned off by default.
However, what happens to the data depends on how a website responds to the request. Many websites still collect and use
browsing data for security, content, services, ads, recommendations, and reporting statistics.
Most websites and web services, including Google’s, do not change their behavior when they receive a Do Not Track request.
1. On your computer, open Chrome.
2. At the top right, select More
 Settings.
3. Select Privacy and security
 Third-party cookies.
4. Turn Send a “Do not track” request with your browsing traffic on or off.
Figure 14:Chrome task example with a retrieved web tutorial.The top panel shows the initial screenshot; the
lower panel shows the retrieved markdown tutorial with inline images preserved.Task ID:030eeff7-b492-4218-
b312-701ec99ee0cc.Source:Google Chrome Help.
32

Task instruction. Could you assist me in placing my photo on the desktop and renaming it toexport.jpg?
Category:gimp.
GIMP treats standard image formats such as.jpgas exports rather than saves.
1. In the top menu, go to File > Export As... or press Ctrl+Shift+E.
2. In the Name box at the top, delete the existing text and typeexport.jpg.
3. On the left-hand sidebar, under Places, click Desktop to ensure it is being saved to the right spot.
4. Click the Export button in the bottom right.
5. An “Export Image as JPEG” window will pop up. Click Export again to confirm the settings.
Figure 15:GIMP task example with an MLLM-synthesized tutorial.The top panel shows the initial screenshot;
the lower panel shows the synthesized tutorial for exporting the image asexport.jpgon the desktop.Task ID:
77b8ab4d-994f-43ac-8930-8ca087d7c4b4.
33

I ScienceBoard-Tutorial Examples
ScienceBoard-Tutorial pairs each of the 169 ScienceBoard tasks with a tutorial generated by Google
Gemini 3.8 Flash, using the same prompt and inputs as OSWorld-Tutorial (Figure 13). Figures 17 and 18
show a Celestia task and a ChimeraX task with their generated tutorials. The Celestia tutorial lists several
alternative methods; Appendix B (Figure 5) analyzes how the choice among them affects RAG-PJ on this
task. The ChimeraX tutorial instead gives a single command, with a menu-based alternative.
Task instruction:Go to Neptune in Celestia.App:Celestia
Method 1: Using the Celestial Browser (Visible Panel)
1.In theCelestial Browserpanel on the left (under theSolar Systemtab), click onNeptunein the list under Solar
System Barycenter→Solto select it.
2.Press theGkey on your keyboard (or right-clickNeptunein the list and selectGo to, or double-clickNeptune) to
travel to the planet.
Method 2: Using Direct Planet Number Shortcut
1. Click on the main 3D star viewport to ensure it has focus.
2. Press the number key8on your keyboard to select Neptune (the 8th planet from the Sun).
3. Press theGkey to travel directly to Neptune.
Method 3: Using the Target Name Bar
1. Click on the main 3D star viewport and pressEnter(Return) to bring up the target entry prompt.
2. TypeNeptuneand pressEnterto confirm the selection.
3. Press theGkey to travel to Neptune.
Method 4: Using the Navigation Menu
1. ClickNavigationin the top menu bar.
2. ClickGo to Object...(orSelect Object..., typeNeptune, and confirm).
3. In the dialog window, ensure the target is set toNeptuneand click theGo to(orOK) button to begin navigation.
Figure 17:Celestia task example with an MLLM-generated tutorial.The top panel shows the initial screenshot;
the lower panel shows the tutorial generated by Google Gemini 3.8 Flash.Task ID:Celestia/B-02.
34

Task instruction:Toggle a scale bar of length 10 and thinkness 4 in ChimeraX.App:ChimeraX
1. Click the command input field next toCommand:at the bottom of the ChimeraX window.
2. Typescalebar length 10 thickness 4and press Enter.
Alternative: SelectToolsfrom the top menu, open the scale bar tool under depiction utilities, set the length to 10 and
thickness to 4, and toggle the scale bar on.
Figure 18:ChimeraX task example with an MLLM-generated tutorial.The top panel shows the initial
screenshot; the lower panel shows the tutorial generated by Google Gemini 3.8 Flash. The task instruction is quoted
verbatim from ScienceBoard, including its typo.Task ID:ChimeraX/B-06.
J Problem formulation
We cast the computer-use task as a standard finite-horizon partially observable Markov decision process
(POMDP) ⟨S, O, A, T, R, L, M⟩ , conditioned on the task instruction Land a multimodal tutorial M. The
computing environment itself (typically a virtual machine) serves as a mostly-deterministic11transition
function T:S×A→S . As the agent only accesses the environment visually, the observation ot∈O
at each time step tis simply the current screenshot. Actions correspond to movements or clicks of
the mouse and keystrokes as well as stop actions (more details in Figure 19). The reward function
R:S×A→ {0,1} is sparse and indicates task success when the agent issues stop actions or reaches
the horizon H. We consider a CUA to be a stochastic policy π(at|o≤t, a<t, L, M) , which produces a
distribution over actions atgiven previous observations o≤t, prior actions a<t, the instruction L, and the
tutorial M. We want to find a policy which achieves high expected cumulative reward (i.e., task success)
over CUA tasks.
Look-ahead rollouts.Specifically, given a world model w:O×A→O that predicts the next
observation given past observations and the current action, the outcome of each of the Kcandidate actions
can be estimated as ˜o1
t, ...,˜oK
tin a one-step look-ahead. Following (Mei et al., 2026), the policy πis then
executed with greedy decoding to generate a single next action ˆaj
t+1= arg max ˆaπ(ˆa|˜oj
t+1)for each
predicted observation ˜oj
t+1. This loop can be executed until some fixed look-ahead horizon is reached to
generate a rollout trajectory ξj= (o t,ˆaj
t,˜oj
t+1,ˆaj
t+1, . . . ,˜oj
t+k)for each action candidate ˆaj
t. The judge
model can then select between actions based on their rollout trajectories.
11In our setting,Tis deterministic for most tasks. For examples of non-deterministic tasks, see Appendix S.
35

K RAG-PJ implementation details
History information.Following GTA-1 (Yang et al., 2026c)12,the policy and the judgereceive the
same execution history. It includes the actions of all previous steps, the intermediate thought tokens of the
five most recent steps, and the screenshots of up to five recent steps. The screenshot window drops the
oldest steps four at a time instead of one per step, so the prompt prefix can be reused across steps. When
the history text grows too long, the oldest steps outside the screenshot window are replaced by a single
line noting that they were omitted. We find that including intermediate thought tokens for these steps does
not significantly improve performance, while it does increase the input length and inference time for both
the policy and the judge. We include the ablation study on the effect of including intermediate thought
tokens in the execution history in Table 13.
MLLM base15 steps
w/ thought w/o thought
Qwen3-VL-8B 28.5%±1.6% 28.1%±1.2%
Qwen3-VL-32B 39.0%±1.2% 38.8%±0.4%
Table 13: Ablation of including intermediate thought tokens in the execution history. We use the setting vanilla
policy without tutorial retrieval. Each result is mean and sample std. over 3 completed runs. We find that including
intermediate thought tokens does not significantly improve performance.
Policy.We use the default decoding hyperparameters for our policies13, shown in Table 14. We use the
prompt template shown in Figure 19 for the policy, which includes the retrieved tutorial and execution
history as context for action generation.
Hyperparameter Value
temperature0.7
top_p0.8
top_k20
repetition_penalty1.0
presence_penalty1.5
max_output_length16384
Table 14: Default decoding hyperparameters used for the RAG-PJ policy.
Judge.Following WebJudge (Xue et al., 2025) and MLLM-as-the-judge (Chen et al., 2024a), we use
deterministic decoding for the judge, shown in Table 15. We use the prompt template shown in Figure 20
for the judge, which includes the retrieved tutorial and execution history as context for action evaluation.
We also apply the same set of hyperparameters for other judges in our baselines, including R-WoM judge
(ranker) and ground-truth judge.
Hyperparameter Value
temperature0
top_p1
Table 15: Default decoding hyperparameters used for the RAG-PJ judge.
Grounding.We use UI-Venus-1.5 (Venus Team et al., 2026) as the grounding model for RAG-PJ. We
use the prompt template shown in Figure 21 for the grounding model.
12Note that the official implementation of GTA-1 https://github.com/Yan98/GTA1 is different from GTA-1’s paper
description. In the implementation, it only uses thought tokens and actions, and does not use observation caption descriptions for
history information. We follow the implementation instead of the paper description.
13Hugging Face model cards: Qwen3-VL-8B-Instruct; Qwen3-VL-32B-Instruct-FP8.
36

L Policy Prompt
The listing below shows the fixed- Kvariant used when adaptive policy is disabled before the final
step. The system message is SHARED_SYSTEM_PROMPT (a one-line intro, shared with the judge, and
the full action space; the judge gets a compact action space instead) followed by the policy role text
POLICY_K_ROLE_WITH_TUTORIAL ; the user message is POLICY_K_USER . On OSWorld, at current_step
>= max_steps , the system message is unchanged (so its prefix cache is reused) and the user message
appends a # FINAL STEP note after the role selector: instead of exactly Kcandidates, return 1–2 different
candidates using only terminal actions, each at most once. On ScienceBoard, the last step instead appends
a# LAST STEP note asking for the action that completes the task. When adaptive policy is enabled, the
runtime uses the adaptive variant instead: the output contract changes from exactly Kcandidates to 1–K
candidates, and theADAPTIVE ACTION RULESblock is added.
[SYSTEM MESSAGE]
Ubuntu desktop computer-use agent (1920x1080; internet available). The user message gives the
task, an optional tutorial, step history, and screenshots, then names your role. Follow only the
rules and output format of that role.
# ACTION SPACE
def click(self, instruction: str, num_clicks: int = 1, button_type: str ='left', hold_keys: List
= []):
'''Click the target element.
instruction: one clear visual description distinguishing the target.
num_clicks: 1 or 2. button_type: "left"|"right"|"middle". hold_keys: e.g. ["ctrl"].
'''
def done(self, return_value: Union[Dict, str, List, Tuple, int, float, bool, NoneType] = None):
'''Call ONLY after fully satisfying the task. Do not use agent.done(return_value=...) for
impossible tasks; call agent.fail().
'''
def drag_and_drop(self, starting_description: str, ending_description: str, hold_keys: List = []):
'''Drag start to end. Prefer cut/paste for file ops and highlight_text_span for visible text/code
selection. Use for drawing, moving objects, or non-text selection.
'''
def fail(self):
'''Call ONLY for proven impossibility: CAPTCHA/bot-check, login walls, missing credentials,
unavailable sites, or absent required settings/options.
'''
def highlight_text_span(self, starting_phrase: str, ending_phrase: str):
'''Select visible text/code between phrases. Prefer this over drag_and_drop for text selection,
including multi-line selections.
'''
def hover(self, instruction: str):
'''Move to target without clicking.
'''
def hold_and_press(self, hold_keys: List, press_keys: List):
'''Hold hold_keys; press press_keys in sequence.
'''
def hotkey(self, keys: List):
'''Press a hotkey. Prefer over menus. Use switch_applications to switch apps.
'''
def open(self, app_or_filename: str):
'''Open a closed app or file by name; use switch_applications if open. Give filename, not full
path.
Available: Google Chrome, LibreOffice Writer, LibreOffice Calc, LibreOffice Impress, Thunderbird,
VSCode, VLC Media Player, GIMP. Do not use other programs.
'''
37

def scroll(self, instruction: str, clicks: int, shift: bool = False):
'''Scroll target. clicks: positive=up, negative=down; shift=True means horizontal. 1 ~ 3 lines;
small 1-5, section 5-10, long 10-20; max 50.
'''
def set_cell_values(self, cell_values: Dict[str, Any], app_name: str, sheet_name: str):
'''Set spreadsheet cells; ALWAYS use instead of click+type.
cell_values: e.g. {"A1": "Name", "B1": 42}; formulas start with "="; max 30 cells per call.
app_name: file name from title (e.g. "data.xlsx"), not "LibreOffice Calc". sheet_name: tab name.
'''
def switch_applications(self, app_code):
'''Switch to an open app by its listed code.
'''
def type(self, element_description: Optional[str] = None, text: str ='', overwrite: bool = False,
enter: bool = False):
'''Type text into a field. If element_description is provided, grounding locates and clicks that
element first. Omit element_description only when current keyboard focus/caret is known and
unchanged; omission skips grounding and clicking and types at current focus. Mouse pointer
position alone is not keyboard focus. Otherwise provide element_description for grounding.
overwrite=True clears possible existing text before replacement (dialogs, search/address bars,
cells, Save As, Find/Replace); False appends intentionally. In a multi-line text or code editor,
overwrite=True erases the whole document, not just the clicked word or line: to replace part of a
document, first select that part (agent.highlight_text_span, or a double-click on one word),
then type with element_description omitted and overwrite=False. Single-line GUI inputs still
showing an old entry (calculator/command line after Enter, rename field; not terminals) need
overwrite=True; False after grounding clicks inserts mid-string. If the whole old text is already
selected, omit element_description. In console commands never nest single quotes; use "..."
outside'...', e.g. where="name LIKE'%
'''
def wait(self, time: float):
'''Wait seconds only for a visible progress/load indicator.
'''
# ROLE: POLICY
Control Ubuntu desktop (1920x1080; internet available). Use task, optional tutorial, history,
screenshots to choose best next actions.
# POLICY RULES:
- Follow task exactly; preserve unrelated content/tabs; visible targets only; current screenshot
is ground truth.
- For each UI-targeted candidate, classify target as ENABLED, DISABLED, or NOT_VISIBLE; inspect
checked/unchecked and open/closed state. Greyed-out controls are DISABLED; never click or type
into a disabled control. First use its visible enabling checkbox/toggle/parent.
- For lists, never click a hidden option: a mouse click is allowed only when the list is visibly
open and the exact option text is visible. Prefer arrow keys + Enter over clicking the row. When
the combobox/list control is visibly focused, keyboard navigation with arrow keys + Enter is
allowed whether the popup is open or closed, even if the target row is not visible. When the
control is not focused, first focus it. Verify committed text/selection in the next screenshot
before proceeding.
- Prefer hotkeys/terminal over equivalent GUI. For spreadsheet value/formula entry or fill,
candidate 0 must use agent.set_cell_values() for the requested range, at most 30 cells per call (
continue a larger range next step); never click, type, drag, or highlight to enter/fill cells.
- If an action's expected effect is absent, do not repeat it unchanged. After two no-effect
actions, prohibit that target+method pair; inspect the dialog for an unchecked prerequisite or
disabled control group, then recover differently.
- Save with Ctrl+S. Before agent.done(), finish/apply dialogs and verify every requested value,
text, format, and path in the final artifact. Dialogs, highlights, and partial edits are not
completion.
- Cancel "Authentication required". Never agent.fail() for uncertainty or recoverable UI.
- At max_steps, return only evidence-supported agent.done() or agent.fail() candidates.
- Never kill/restart desktop, log out, or reboot; keep media changes moderate unless exact/
extreme requested.
Sudo password:'{CLIENT_PASSWORD}'(type value, not placeholder).
38

# OUTPUT FORMAT
{"thought": "(1) Step by Step Progress Assessment;(2) Next Action(s) Analysis;(3) UI State/
Evidence: each returned candidate`Candidate i`; N/A when no coordinate-grounded UI target;(4)
Tutorial Reference","action_candidates":[{"action_code":{"action_type":"hotkey","parameters":{"
keys":["ctrl","s"]}}}]}
# OUTPUT REQUIREMENTS
Thought: Must have the following sections:
(1) Step by Step Progress Assessment---assess completion and potential errors/unexpected results;
predict a recovery step if previous action incorrect.
(2) Next Action(s) Analysis---evaluate next actions from current state/history; propose best.
(3) UI State/Evidence---write one candidate-indexed entry for each returned candidate (`Candidate
i: ...`). Each UI-targeted candidate: ENABLED, DISABLED, or NOT_VISIBLE; checked/unchecked or
open/closed; exact screenshot evidence from the current screenshot. For any candidate without a
coordinate-grounded UI target, write`N/A (no coordinate-grounded UI target)`; never invent UI
state or evidence. Non-UI: focused agent.type(...), hotkey/hold_and_press/open/
switch_applications/wait/set_cell_values/done/fail/answer. Never provide one shared UI State/
Evidence entry for multiple candidates.
(4) Tutorial Reference: QUOTE the exact relevant sentence(s) from the tutorial. If none, say`No
tutorial reference for this step.`If method fails, switch to another tutorial method.
Action requirements:
- One currently valid action per candidate, confidence-ordered; vary method/target. Never type
into buttons or click static text.
- For visible text/code, use highlight_text_span with phrase anchors, including multi-line
selections. Never use it for spreadsheet cell ranges. Use drag_and_drop only for objects/drawing
or text without visible unique anchors.
Return only JSON matching the structured action schema. Return exactly {K} different and diverse
action_candidates. Each needs action_code.action_type and action_code.parameters; no markdown,
code blocks, or agent.* calls. Use at most one done and one fail; never create terminal fillers
or return_value variants. For click use button_type='right'or num_clicks=2. For type use
element_description (not instruction). Keyboard arrays must be bounded: hold_keys <= 4, keys <= 6,
and press_keys <= 20.
# TUTORIAL RULES:
- Tutorial is a hint: use relevant steps; prefer a simpler reliable method.
- INFEASIBLE OVERRIDE (HIGHEST PRIORITY; MUST FOLLOW): If the tutorial's first non-empty line
begins with`Infeasible`or`infeasible`(case-insensitive), stop: the first and only valid
semantic action is`agent.fail()`. In JSON use action_type='fail', action_code.parameters={};
never emit agent.*. Ignore all later text and candidate-count/diversity/recovery/workaround rules.
Do not emit any non-fail action candidate.
- For ordinary tutorials, after 2 failed attempts switch methods; for streaming media, allow 2
buffering steps.
[USER MESSAGE]
# TUTORIAL
{tutorial}
# TASK INSTRUCTION
{instruction}
# HISTORY
{history}{current_screenshot}
# YOUR ROLE
Act as the POLICY: follow the ROLE: POLICY section of the system message.
# STEP
You are on step {current_step} of maximum {max_steps} steps.
Figure 19: Policy prompt used to generate candidate actions from the task instruction, retrieved tutorial, interaction
history, and current screenshot.
39

M Judge Prompt
The system message starts with the same one-line introduction as the policy’s (Appendix L), followed by
a compact action space ( AGENT_METHODS_LIST : one line per method instead of the policy’s full signatures
and docstrings) and the judge role textJUDGE_ROLE_PROMPT; the user message isJUDGE_USER_PROMPT.
[SYSTEM MESSAGE]
Ubuntu desktop computer-use agent (1920x1080; internet available). The user message gives the
task, an optional tutorial, step history, and screenshots, then names your role. Follow only the
rules and output format of that role.
# ACTION SPACE
Use only these exact methods/arguments; no right_click/double_click:
- agent.click(instruction, num_clicks=1, button_type='left', hold_keys=[]) # button_type='right
'|'middle'; double-click=num_clicks=2
- agent.done(return_value=None)
- agent.drag_and_drop(starting_description, ending_description, hold_keys=[])
- agent.fail() # proven impossible: CAPTCHA/login/missing required user account credentials/
unavailable site/absent required setting or option
- agent.highlight_text_span(starting_phrase, ending_phrase)
- agent.hover(instruction)
- agent.hold_and_press(hold_keys, press_keys)
- agent.hotkey(keys)
- agent.open(app_or_filename) # closed app/file; filename, not path; switch if open. Only Google
Chrome, LibreOffice Writer/Calc/Impress, Thunderbird, VSCode, VLC Media Player, GIMP; no other
programs
- agent.scroll(instruction, clicks, shift=False) # +up/-down; shift=horizontal; 1~3 lines; small
=1-5, section=5-10, long=10-20; max=50
- agent.set_cell_values(cell_values, app_name, sheet_name) # ALWAYS for sheet cells; formulas
start "="; max 30 cells/call; app=title filename, sheet=tab
- agent.switch_applications(app_code)
- agent.type(element_description=None, text='', overwrite=False, enter=False) # omit
element_description only when keyboard focus/caret is known unchanged from history or immediately
preceding focus action; pointer!=focus; click/app switch/dialog/restore uncertainty requires
grounding; overwrite=True replaces, False appends intentionally; in a multi-line text/code editor
overwrite=True erases the whole document, so select the part to replace (highlight_text_span)
and type over it without element_description; overwrite=True for a single-line GUI input still
showing old text (calculator/command line, rename field; not terminals); if it is already
selected, omit element_description; console commands: no nested single quotes; Enter only submits
(click live-filter results)
- agent.wait(time) # visible progress/load only
# ROLE: JUDGE
Evaluate the candidate actions and rank them by how well each advances the overall goal.
# EVALUATION CRITERIA:
Screenshots are ground truth; tutorials are guidance unless they start with'Infeasible'.
1. Feasibility - Target must be visible now; after-screenshots/rollouts must prove the action
worked.
2. Exactness - Advance the task without extra steps.
3. Effectiveness - Make visible progress toward the final state.
4. Completion - Use agent.done() only when already complete.
5. Tutorial alignment - Prefer feasible tutorial actions, but trust pixels when they differ.
6. No regression - Reject broken state, navigation away, or lost progress.
7. Terminal evidence - Consider agent.done()/agent.fail() only with hard screenshot/history proof.
Prefer any visible action needed to complete, save, apply, close a required dialog, or verify
the final artifact.
# REQUIRED DECISION PROCESS
Before deciding:
1. Derive exact task-specific success conditions from the task instruction.
2. Apply all applicable named evaluation criteria to every candidate.
3. Judge progress toward the overall goal, not mere action execution.
4. Compare observed or expected resulting states and cumulative progress across candidates.
5. Decide only after that comparison.
40

In`thought`, connect screenshot/history evidence to success conditions and criteria; do not
merely narrate screenshots.
# TUTORIAL GROUNDING GUIDANCE (IF TUTORIAL IS PROVIDED):
Prefer feasible tutorial operations that reach task milestones.
# RULES:
(1) **OVERRIDE - Infeasible tutorial**: If a tutorial is provided and starts with'Infeasible',
agent.fail() is the only correct action.
(2) Prefer terminal over GUI, then hotkeys over multiple clicks.
(3) ANTI-LOOP: If current and latest HISTORY screenshots are substantially unchanged, reject any
candidate matching an action's method+target from the last 3 steps. If all feasible candidates
repeat failures, prefer a different agent.* method family and explain the rejection.
(4) Do not treat agent.done(return_value=...) as an infeasibility report; prefer agent.fail() for
proven impossibility.
(5) On the LAST step, only agent.done() or agent.fail() can succeed.
(6) Google Chrome is the ONLY available browser.
(7) TEXT-SELECTION METHOD PREFERENCE: For the same text range, prefer agent.highlight_text_span
(...) over agent.drag_and_drop(...) unless anchors are not visible or the target is non-text. Do
not reject highlight_text_span as unsupported for multi-line selection.
(8) FOCUSED TEXT ENTRY: Omit element_description only when current keyboard focus/caret is known
and unchanged. Then agent.type(text=..., overwrite=..., enter=...) intentionally skips grounding
and clicking. History or the current screenshot must establish the intended focus; any
intervening click, app switch, dialog transition, or restore uncertainty changes the rule. Do not
reject omitted element_description solely when focus evidence is strong. If focus is uncertain,
the target is another field, or only the mouse pointer is over the field, require agent.type(
element_description=..., ...) so grounding can locate it.
# EVIDENCE-FIRST VERIFICATION:
Candidate prose, rationale, tutorials, and simulated/text-only trajectories are hypotheses, not
execution evidence. Simulated/text-only trajectories may be FEASIBLE, REJECTED, or UNKNOWN, never
VERIFIED.
- Check exact requested conditions: values, text, counts, range endpoints, selection boundaries,
dates, names, file paths, source/provenance, and persistence; generic change is insufficient.
- Compare the latest HISTORY action with the current screenshot first. Missing expected effect is
failure.
- For each unexecuted candidate, require a visible target and preconditions; mark FEASIBLE; never
claim execution or mark VERIFIED.
- For rollouts, inspect screenshots state by state only for actual executed actions. Mark
VERIFIED only when an actual state visibly proves the effect; cite it (e.g.`Candidate 1 State 2`)
.
- Missing, ambiguous, or unreadable evidence is UNKNOWN, never success.
- Visible proof: selection shows highlight/handles/caret/status; saving shows a completed dialog
plus visible target/location or explicit saved/applied history; formatting shows values;
persistence shows Save/Apply/OK or reopen/check.
- DONE: If any candidate is`agent.done(...)`,`thought`MUST begin`DONE_CHECK: <condition> =
MET | NOT-MET; ...`, cover every requested final condition, then give`CANDIDATE_CHECKS`. MET
means the requested end state holds (a removal/hide goal is MET when the item is gone) and
requires direct current-screen proof or explicit saved/applied confirmation; dialogs, highlights,
menus, and partial edits are not completion. Select done only when every condition is MET.
Missing/incomplete DONE_CHECK, UNKNOWN, or NOT-MET rejects done; choose progress addressing the
missing condition.
Selection rules:
- Prefer verification for required UNKNOWN evidence; prefer recovery after two failures in the
current GUI branch.
- Do not reject all candidates when one can safely reveal missing evidence or recover. If none
advances the task, choose the least-irreversible visible verification/recovery; never unsupported
done/fail.
[USER MESSAGE]
# TUTORIAL
{tutorial}
41

# TASK INSTRUCTION
{instruction}
# HISTORY
{history}{current_screenshot}
# YOUR ROLE
Act as the JUDGE: follow the ROLE: JUDGE section of the system message.
# CANDIDATE ACTIONS
{candidate_actions}
# STEP
You are at step {current_step} of the max step {max_steps}.
# THOUGHT FORMAT
In`thought`, output these fields in order:
TASK_CHECK: <condition> = VERIFIED | FAILED | UNKNOWN (<exact evidence>); ...
LAST_EFFECT: SEEN | NOT-SEEN | UNKNOWN (<exact evidence>)
CANDIDATE_CHECKS:
0 = FEASIBLE (Feasibility: target visible; Effectiveness: expected effect advances a condition)
1 = REJECTED (Feasibility: target/prerequisite missing; Effectiveness: no task progress)
2 = ...
If any candidate is`agent.done(...)`,`thought`MUST begin with the`DONE_CHECK:`line required
by the
evidence rules above, followed immediately by CANDIDATE_CHECKS, then TASK_CHECK and LAST_EFFECT.
# OUTPUT FORMAT
Output your response in the following JSON format:
{
"thought": "your rationale for the ranking result",
"ranking": [x, x, x] # "indexes of the action candidates, most promising first"
}
Figure 20: Judge prompt used to rank the candidate actions (the top-ranked one is executed) based on the current
state, task instruction, retrieved tutorial, and evaluation criteria.
N Grounding Prompt
You are a GUI grounding agent. Given a screenshot and the user's grounding instruction, locate
the UI element. Examine the screenshot carefully, think step by step about which element matches,
then provide the coordinate.
Match the requested semantic label, not merely a similar-looking icon. In LibreOffice Calc, "
Background Color", "Fill Color", or "paint bucket" means the cell background/fill-color bucket
control, not the funnel-shaped AutoFilter icon. The output should just be the coordinates of a
point, in the format [x,y]. Additionally, if the task is infeasible (e.g., the task is not
related to the image), the output should be [-1,-1].
# Task instruction:
{}.
Figure 21: Grounding prompt used by UI-Venus-1.5 to localize UI elements from grounding instructions and
screenshots.
O R-WoM reimplementation
O.1 Implementation details
We reimplement the R-WoM method based on the description in the paper. We make the following
implementation changes to R-WoM to ensure compatibility with our environment and comparability with
other baselines for a more controlled comparison. Changes 1, 3 and 4 below are shared by every method in
42

Table 1, including ours, and are required for a controlled comparison. Change 2 fixes a format violation in
the original rollout prompt that left the ranker with malformed input. Table 3 shows the re-implementation
reproduces R-WoM’s gain over a judge-only baseline without tutorials.
1.We use the action space defined by Agent-S2 (Agashe et al., 2025b) , instead of the original
action space of OSWorld (Xie et al., 2024) used in R-WoM. This action space makes the agent
act more effectively and is more widely adopted in other GUI-agent works, such as the baseline
GTA-1 (Yang et al., 2026c), Agent-S family (Agashe et al., 2025b; Gonzalez-Pumariega et al., 2026)
and UITAR1.5 (Qin et al., 2025).
2.We notice that the text-based rollout of R-WoM does not produce a well-structured rollout that
follows the rules in the prompt, which makes it hard for the R-WoM judge (ranker) to judge from the
rollout. For example, given the horizon H=2 , it does not stop at the state 2 as required, and it stops
at rollout actions instead. Therefore, we design a new rollout prompt with stricter rules and a more
structured output format, as shown in the rollout prompt in Figure 23.
3.To make it fair to compare R-WoM with other baselines such as GTA-1 (Yang et al., 2026c), we add
judge rules from GTA-1 to the R-WoM judge prompt.
4.We keep the GTA-1 (Yang et al., 2026c) policy rules and action space but adopt R-WoM’s policy
output format: a shared observation and a separate thought for each action candidate.
O.2 R-WoM error analysis
To better understand the limitations of R-WoM, we manually analyzed more than 30 rollout samples. We
found that its failures primarily arise from two sources.
•Because R-WoM originally does not have RAG policy, policy can propose wrong actions and
therefore the judge does not have a good candidate among proposed candidates. For example, if the
tutorial already says “Infeasible”, the agent’s policy should produce “agent.fail()” to end the task.
However, its policy does not produce this action, instead proposing different plans to complete the
task and hallucinating that the task is completed.
•R-WoM hallucinates or fails to produce high-quality text-based rollouts, making it difficult for (1)
the policy to produce good rollout actions and/or (2) the judge to select the correct action. These
failures range from outright incorrect descriptions to inadequate descriptions of the next state.
To improve the R-WoM baseline and allow for fair comparison, we extended R-WoM with a RAG
policy. We show that the performance of R-WoM + RAG policy improves over R-WoM, confirming
R-WoM’s policy limitation in the first source of error.
O.3 Policy prompt
The R-WoM policy uses the GTA-1 policy prompt (Figure 19) with the same system introduction, action
space, policy rules, action requirements, and tutorial rules. Only the output section changes, to follow
the R-WoM policy format: one shared observation, then a separate thought inside each action candidate.
The structured-output JSON schema matches: it requires observation and athought in every candidate,
with no shared thought. Each candidate’s thought is passed to its world-model rollout and to the judge, and
the selected candidate’s thought is stored in the history. Without tutorial evidence, the Tutorial Reference
parts are omitted as in the GTA-1 prompt. In the diff below, red “-” lines are the GTA-1 output section
and green “+” lines are the R-WoM replacement; “...” marks unchanged text.
R-WoM policy prompt (output section)
--- GTA-1 POLICY PROMPT (OUTPUT SECTION)
+++ R-WOM POLICY PROMPT (OUTPUT SECTION)
# OUTPUT FORMAT
43

-{"thought": "(1) Step by Step Progress Assessment;(2) Next Action(s) Analysis;(3) UI State/
Evidence: each returned candidate`Candidate i`; N/A when no coordinate-grounded UI target;(4)
Tutorial Reference","action_candidates":[{"action_code":{"action_type":"hotkey","parameters":{"
keys":["ctrl","s"]}}}]}
+{"observation": "Current state, changes since the previous step, and progress assessment; no
intended actions","action_candidates":[{"thought": "Why this action is appropriate given the
observation; UI State/Evidence; Tutorial Reference","action_code":{"action_type":"hotkey","
parameters":{"keys":["ctrl","s"]}}}]}
# OUTPUT REQUIREMENTS
-Thought: Must have the following sections:
-(1) Step by Step Progress Assessment---assess completion and potential errors/unexpected results;
predict a recovery step if previous action incorrect.
-(2) Next Action(s) Analysis---evaluate next actions from current state/history; propose best.
- (3) UI State/Evidence---write one candidate-indexed entry for each returned candidate (`
Candidate i: ...`). Each UI-targeted candidate: ENABLED, DISABLED, or NOT_VISIBLE; checked/
unchecked or open/closed; exact screenshot evidence from the current screenshot. For any
candidate without a coordinate-grounded UI target, write`N/A (no coordinate-grounded UI target)`;
never invent UI state or evidence. Non-UI: focused agent.type(...), hotkey/hold_and_press/open/
switch_applications/wait/set_cell_values/done/fail/answer. Never provide one shared UI State/
Evidence entry for multiple candidates.
-
-(4) Tutorial Reference: QUOTE the exact relevant sentence(s) from the tutorial. If none, say`No
tutorial reference for this step.`If method fails, switch to another tutorial method.
+Observation (shared, written once): (1) describe the current state and what changed since the
previous step; (2) assess step-by-step progress and errors/unexpected results; predict a recovery
step if the previous action was incorrect. Keep it concise and do not describe intended actions
here.
+Thought (one inside each candidate): why this action is appropriate given the observation, then
its UI State/Evidence: for a UI-targeted candidate, ENABLED, DISABLED, or NOT_VISIBLE; checked/
unchecked or open/closed; exact evidence from the current screenshot. For a candidate without a
coordinate-grounded UI target, write`N/A (no coordinate-grounded UI target)`; never invent UI
state or evidence.
+Each candidate thought ends with its Tutorial Reference: QUOTE the exact relevant sentence(s)
from the tutorial. If none, say`No tutorial reference for this step.`If method fails, switch to
another tutorial method.
...
Return only JSON matching the structured action schema. Return exactly {K} different and diverse
action_candidates. Each needs +thought, action_code.action_type-and+, and action_code.parameters;
...
Figure 22: Unified diff of the GTA-1 policy output section and the R-WoM policy output section used for R-WoM.
O.4 Rollout prompt
TheTUTORIAL USAGE GUIDELINE block is included only when tutorial evidence exists; when no tutorial
evidence is available, the runtime removes this block from the rollout prefix. Plain text is from the original
R-WoM paper or implementation. In the unified diff below, red “-” lines are original text removed or
replaced, green “+” lines are current runtime additions or replacements, and black lines are unchanged
context;—and+++identify the compared versions. Only word-level semantic changes are colored;
capitalization, Markdown markers, list markers, punctuation, and symbol-only differences are normalized
and not treated as changes.
R-WoM rollout prompt
--- ORIGINAL R-WoM PROMPT
+++ OUR CURRENT RUNTIME PROMPT
You are a world-model assistant with extensive knowledge of desktop and web UIs. Given the task
objective, the candidate action, and the observation before the candidate action, you must "
simulate
the future" and describe the plausible future states.
-Available actions
44

-# Action space definition
+# ACTION SPACE
+Use only these exact methods/arguments; no right_click/double_click:
+- agent.click(instruction, num_clicks=1, button_type='left', hold_keys=[]) # button_type='right
'|'middle'; double-click=num_clicks=2
+- agent.done(return_value=None)
+- agent.drag_and_drop(starting_description, ending_description, hold_keys=[])
+- agent.fail() # proven impossible: CAPTCHA/login/missing required user account credentials/
unavailable site/absent required setting or option
+- agent.highlight_text_span(starting_phrase, ending_phrase)
+- agent.hover(instruction)
+- agent.hold_and_press(hold_keys, press_keys)
+- agent.hotkey(keys)
+- agent.open(app_or_filename) # closed app/file; filename, not path; switch if open. Only
Google Chrome, LibreOffice Writer/Calc/Impress, Thunderbird, VSCode, VLC Media Player, GIMP; no
other programs
+- agent.scroll(instruction, clicks, shift=False) # +up/-down; shift=horizontal; 1~3 lines;
small=1-5, section=5-10, long=10-20; max=50
+- agent.set_cell_values(cell_values, app_name, sheet_name) # ALWAYS for sheet cells; formulas
start "="; max 30 cells/call; app=title filename, sheet=tab
+- agent.switch_applications(app_code)
+- agent.type(element_description=None, text='', overwrite=False, enter=False) # omit
element_description only when keyboard focus/caret is known unchanged from history or immediately
preceding focus action; pointer!=focus; click/app switch/dialog/restore uncertainty requires
grounding; overwrite=True replaces, False appends intentionally; in a multi-line text/code editor
overwrite=True erases the whole document, so select the part to replace (highlight_text_span)
and type over it without element_description; overwrite=True for a single-line GUI input still
showing old text (calculator/command line, rename field; not terminals); if it is already
selected, omit element_description; console commands: no nested single quotes; Enter only submits
(click live-filter results)
+- agent.wait(time) # visible progress/load only
# TUTORIAL USAGE GUIDELINE
1. Use the tutorials to identify efficient workflow patterns that should be predicted as likely
outcomes.
2. Provide a reference to the tutorial if the current situation matches the standard operations
in the tutorials. If the current situation does not align with tutorials, rely on internal world
knowledge instead.
# Environment awareness checklist
- Visible UI elements: text, icons, menus, modals, tooltips
- Element states: enabled/disabled, focused/hovered, loading progress
- Hidden or off-screen affordances revealed by scrolling or clicking
- Cursor position, caret position, selection highlights
- Global context: file system changes, network requests, OS dialogs
# Output Format
-Produce an ordered chain from STATE 0 (current) up to STATE n (1 <= n <= {k}); you may stop
early
-if no further prediction is useful.
+Produce an ordered chain from ACTION 0 up to STATE {k_steps}.
+
+Strict rules:
+- Do NOT describe the current state. The reader already has the current screenshot; restating it
wastes tokens and risks contradicting the pixels. Never output a STATE 0 line.
+- Start with ACTION 0: followed by the candidate action exactly as provided in the user message.
+- The highest allowed STATE index is {k_steps}. Never output STATE {k_steps} + 1 or any larger
index.
+- Between every two consecutive states, you MUST output an ACTION line. Every STATE i (for 1 <=
i < {k_steps}) MUST be followed by ACTION i: before STATE i+1. There is no STATE 0, so ACTION 0
is the first line of the response.
+- Describe only what the action would CHANGE, relative to the current screenshot. Do not re-
describe unchanged UI.
+- Keyboard shortcuts: predict the effect of the shortcut's real default binding in this
application, not the effect the candidate intends. If you are not sure what the shortcut does
45

here, write "effect uncertain" instead of assuming success.
+- Repeated actions: if HISTORY shows the same method on the same target was already executed and
the current screenshot does not show its expected effect, that action did not work here. Predict
that it again causes no visible change, unless the screen now differs in a way that removes the
reason it failed (e.g., a blocking dialog has closed or the needed focus or selection now exists).
+- STOP IMMEDIATELY after STATE {k_steps}. Do NOT output ACTION {k_steps} or any text after the
last STATE. The response must end with STATE {k_steps}: <description>.
+- Do not add explanations, code fences, headings, or extra text before or after the rollout.
+
+# EXAMPLE OUTPUT (FOR K_STEPS={k_steps}):
+{rollout_example}
+
+WRONG (do NOT do this):
+STATE {k_steps}: <description>
+ACTION {k_steps}: <any action here> <- FORBIDDEN, response must end at STATE {k_steps}
# CANDIDATE THOUGHT AND ACTION
Thought: {thought}
Action: {action_code}
# YOUR TASK
Simulate the plausible future states after executing the candidate action. Follow the rollout-
length limit from the system prompt exactly. Do not output extra states or actions after the
final STATE line. Use the output format specified above.
# CURRENT STEP
You are on step {current_step} of maximum {max_steps} steps.
Figure 23: Unified diff of the original R-WoM rollout prompt and our current runtime prompt.
Rollout Example, K=1
ROLLOUT_EXAMPLE_K1 = """ACTION 0: agent.typewrite(text="downloads")
STATE 1: The search field contains "downloads", and matching download-related settings are
visible."""
Rollout Example, K=2
ROLLOUT_EXAMPLE_K2 = """ACTION 0: agent.typewrite(text="downloads")
STATE 1: The search field contains "downloads", and matching download-related settings are
visible.
ACTION 1: agent.click(instruction="Click the Change button for download location")
STATE 2: A file chooser dialog opens with folder choices and Cancel and Select buttons."""
Rollout Example, K=3
ROLLOUT_EXAMPLE_K3 = """ACTION 0: agent.typewrite(text="downloads")
STATE 1: The search field contains "downloads", and matching download-related settings are
visible.
ACTION 1: agent.click(instruction="Click the Change button for download location")
STATE 2: A file chooser dialog opens with folder choices and Cancel and Select buttons.
ACTION 2: agent.click(instruction="Select the Documents folder")
46

STATE 3: The Documents folder is highlighted in the file chooser, ready to be selected as the new
download location."""
We use different examples with different rollout lengths.
O.5 R-WoM judge
Plain text is from the original R-WoM paper or implementation. In the unified diff below, red “-” lines are
original text removed or replaced, green “+” lines are current runtime additions or replacements, and black
lines are unchanged context; —and+++identify the compared versions. Only word-level semantic changes
are colored; capitalization, Markdown markers, list markers, punctuation, and symbol-only differences
are normalized and not treated as changes.
R-WoM judge prompt
--- ORIGINAL R-WoM PROMPT
+++ OUR CURRENT RUNTIME PROMPT
You are an agent that evaluates actions by considering the observation before the action
candidates and the potential outcomes of these actions.
+Ubuntu desktop computer-use agent (1920x1080; internet available).
+Evaluate the candidate actions and rank them by how well each advances the overall goal.
+# ACTION SPACE
+Use only these exact methods/arguments; no right_click/double_click:
+- agent.click(instruction, num_clicks=1, button_type='left', hold_keys=[]) # button_type='right
'|'middle'; double-click=num_clicks=2
+- agent.done(return_value=None)
+- agent.drag_and_drop(starting_description, ending_description, hold_keys=[])
+- agent.fail() # proven impossible: CAPTCHA/login/missing required user account credentials/
unavailable site/absent required setting or option
+- agent.highlight_text_span(starting_phrase, ending_phrase)
+- agent.hover(instruction)
+- agent.hold_and_press(hold_keys, press_keys)
+- agent.hotkey(keys)
+- agent.open(app_or_filename) # closed app/file; filename, not path; switch if open. Only
Google Chrome, LibreOffice Writer/Calc/Impress, Thunderbird, VSCode, VLC Media Player, GIMP; no
other programs
+- agent.scroll(instruction, clicks, shift=False) # +up/-down; shift=horizontal; 1~3 lines;
small=1-5, section=5-10, long=10-20; max=50
+- agent.set_cell_values(cell_values, app_name, sheet_name) # ALWAYS for sheet cells; formulas
start "="; max 30 cells/call; app=title filename, sheet=tab
+- agent.switch_applications(app_code)
+- agent.type(element_description=None, text='', overwrite=False, enter=False) # omit
element_description only when keyboard focus/caret is known unchanged from history or immediately
preceding focus action; pointer!=focus; click/app switch/dialog/restore uncertainty requires
grounding; overwrite=True replaces, False appends intentionally; in a multi-line text/code editor
overwrite=True erases the whole document, so select the part to replace (highlight_text_span)
and type over it without element_description; overwrite=True for a single-line GUI input still
showing old text (calculator/command line, rename field; not terminals); if it is already
selected, omit element_description; console commands: no nested single quotes; Enter only submits
(click live-filter results)
+- agent.wait(time) # visible progress/load only
+# EVALUATION CRITERIA:
+Screenshots are ground truth; tutorials are guidance unless they start with'Infeasible'.
+1. Feasibility - Target must be visible now; after-screenshots/rollouts must prove the action
worked.
+2. Exactness - Advance the task without extra steps.
+3. Effectiveness - Make visible progress toward the final state.
+4. Completion - Use agent.done() only when already complete.
+5. Tutorial alignment - Prefer feasible tutorial actions, but trust pixels when they differ.
+6. No regression - Reject broken state, navigation away, or lost progress.
+7. Terminal evidence - Consider agent.done()/agent.fail() only with hard screenshot/history
proof. Prefer any visible action needed to complete, save, apply, close a required dialog, or
verify the final artifact.
47

+# REQUIRED DECISION PROCESS
+Before deciding:
+1. Derive exact task-specific success conditions from the task instruction.
+2. Apply all applicable named evaluation criteria to every candidate.
+3. Judge progress toward the overall goal, not mere action execution.
+4. Compare observed or expected resulting states and cumulative progress across candidates.
+5. Decide only after that comparison.
+In`thought`, connect screenshot/history evidence to success conditions and criteria; do not
merely narrate screenshots.
# Tutorial Grounding Guidance +(IF TUTORIAL IS PROVIDED):
+Prefer feasible tutorial operations that reach task milestones. Prioritize action sequences that
follow the standard operations in the tutorials and have captured the milestones and conditions
to make more meaningful progress to achieve the task objective.
+# RULES:
+(1) **OVERRIDE - Infeasible tutorial**: If a tutorial is provided and starts with'Infeasible',
agent.fail() is the only correct action.
+(2) Prefer terminal over GUI, then hotkeys over multiple clicks.
+(3) ANTI-LOOP: If current and latest HISTORY screenshots are substantially unchanged, reject any
candidate matching an action's method+target from the last 3 steps. If all feasible candidates
repeat failures, prefer a different agent.* method family and explain the rejection.
+(4) Do not treat agent.done(return_value=...) as an infeasibility report; prefer agent.fail()
for proven impossibility.
+(5) On the LAST step, only agent.done() or agent.fail() can succeed.
+(6) Google Chrome is the ONLY available browser.
+(7) TEXT-SELECTION METHOD PREFERENCE: For the same text range, prefer agent.highlight_text_span
(...) over agent.drag_and_drop(...) unless anchors are not visible or the target is non-text. Do
not reject highlight_text_span as unsupported for multi-line selection.
+(8) FOCUSED TEXT ENTRY: Omit element_description only when current keyboard focus/caret is known
and unchanged. Then agent.type(text=..., overwrite=..., enter=...) intentionally skips grounding
and clicking. History or the current screenshot must establish the intended focus; any
intervening click, app switch, dialog transition, or restore uncertainty changes the rule. Do not
reject omitted element_description solely when focus evidence is strong. If focus is uncertain,
the target is another field, or only the mouse pointer is over the field, require agent.type(
element_description=..., ...) so grounding can locate it.
+# EVIDENCE-FIRST VERIFICATION:
+Candidate prose, rationale, tutorials, and simulated/text-only trajectories are hypotheses, not
execution evidence. Simulated/text-only trajectories may be FEASIBLE, REJECTED, or UNKNOWN, never
VERIFIED.
+
+- Check exact requested conditions: values, text, counts, range endpoints, selection boundaries,
dates, names, file paths, source/provenance, and persistence; generic change is insufficient.
+- Compare the latest HISTORY action with the current screenshot first. Missing expected effect
is failure.
+- For each unexecuted candidate, require a visible target and preconditions; mark FEASIBLE;
never claim execution or mark VERIFIED.
+- For rollouts, inspect screenshots state by state only for actual executed actions. Mark
VERIFIED only when an actual state visibly proves the effect; cite it (e.g.`Candidate 1 State
2`).
+- Missing, ambiguous, or unreadable evidence is UNKNOWN, never success.
+- Visible proof: selection shows highlight/handles/caret/status; saving shows a completed dialog
plus visible target/location or explicit saved/applied history; formatting shows values;
persistence shows Save/Apply/OK or reopen/check.
+
+- DONE: If any candidate is`agent.done(...)`,`thought`MUST begin`DONE_CHECK: <condition> =
MET | NOT-MET; ...`, cover every requested final condition, then give`CANDIDATE_CHECKS`. MET
means the requested end state holds (a removal/hide goal is MET when the item is gone) and
requires direct current-screen proof or explicit saved/applied confirmation; dialogs, highlights,
menus, and partial edits are not completion. Select done only when every condition is MET.
Missing/incomplete DONE_CHECK, UNKNOWN, or NOT-MET rejects done; choose progress addressing the
missing condition.
+
+Selection rules:
+- Prefer verification for required UNKNOWN evidence; prefer recovery after two failures in the
current GUI branch.
48

+- Do not reject all candidates when one can safely reveal missing evidence or recover. If none
advances the task, choose the least-irreversible visible verification/recovery; never unsupported
done/fail.
+# THOUGHT FORMAT
+In`thought`, output these fields in order:
+TASK_CHECK: <condition> = VERIFIED | FAILED | UNKNOWN (<evidence from the CURRENT SCREENSHOT>);
...
+LAST_EFFECT: SEEN | NOT-SEEN | UNKNOWN (<evidence from the CURRENT SCREENSHOT>)
+CANDIDATE_CHECKS:
+0 = FEASIBLE (Feasibility: target visible; Effectiveness: expected effect advances a condition)
+1 = REJECTED (Feasibility: target/prerequisite missing; Effectiveness: no task progress)
+2 = ...
+
+LAST_EFFECT reports whether the latest executed action visibly changed the screen. It is about
the action already taken, never about a simulation: a rollout can never make LAST_EFFECT SEEN.
+If LAST_EFFECT is NOT-SEEN, that action had no effect: REJECT every candidate repeating its
method+target however confident its simulation, and prefer a candidate that establishes the
missing prerequisite (focus, selection, caret, or an open dialog) or switches to a different
agent.* method family.
+Judge feasibility against the CURRENT SCREENSHOT; a simulation may only argue about an action's
expected effect, never establish what is on screen now.
+
+If any candidate is`agent.done(...)`,`thought`MUST begin with the`DONE_CHECK:`line required
by the evidence rules above, followed immediately by CANDIDATE_CHECKS, then TASK_CHECK and
LAST_EFFECT.
# Output Format
Output your response in the following JSON format:
{
+ "thought": "your rationale for the ranking result",
"ranking": [x, x, x] # "indexes of the action candidates, most
promising first"
- "thought": "your rationale for the ranking result"
}
# TASK INSTRUCTION
{instruction}
# TUTORIAL EVIDENCE (IF ANY)
{evidence_text}
# STEP HISTORY
{step_history}
+# CURRENT SCREENSHOT
+{current_screenshot}
# SIMULATED ROLLOUT TRAJECTORIES (ONE PER CANDIDATE ACTION)
Each block below shows one candidate's policy thought and its world-model simulation. Evaluate
each candidate's feasibility, correctness, and effectiveness, then rank them.
{rollouts_block}
# CURRENT STEP
You are on step {current_step} of maximum {max_steps} steps.
Figure 24: Unified diff of the original R-WoM judge prompt and our current runtime prompt.
P WebDreamer reimplementation
We run WebDreamer (Gu et al., 2025) from its released code (commit e589411 ), pinned as a read-only
submodule. Its multi-step imagination loop and pointwise scoring are unchanged: for each of the K=3
policy candidates we draw 3 independent simulations of planning horizon H; each simulation predicts
the state change of the candidate action and then, for up to H−1 further imagined steps (stopping early
if a proposed action is stop ), proposes the next action and predicts its state change. A value function
49

scores each simulation with 10 samples (success = 1, on the right track = 0.5 , otherwise 0); the candidate
with the highest mean score is executed. We adapt the prompts to the desktop (web wording becomes
desktop wording, the web operation list becomes our desktop operation list, and the current URL is
dropped) and pass the current screenshot as the policy’s encoded image. Both components also receive
the real executed-step history, so that WebDreamer sees the same history as the other baselines. When
steps have already been executed, each world-model user message starts with the text-only step history
used by the R-WoM rollout, under the header “Steps already executed before the screenshot below (real
history, text only; not part of PREVIOUS ACTIONS)”; the released code gives the world model no history.
Instead of the trajectory’s screenshots and an action list, the value function receives the same history
as the policy+judge judge (older steps as text, recent steps with screenshots), followed by the current
screenshot under the heading “Current state of the screen” and the simulated steps. The value prompt also
gains one rule: a simulation whose first action is agent.fail() counts as a success only when the history
and the screen prove the task impossible, mirroring the policy+judge judge’s rule for infeasible tasks.
World-model and value calls use the judge model (Qwen3-VL-32B, as for the policy) instead of GPT-4o,
with OpenAI’s default sampling (temperature 1.0, top- p1.0, no top- k). The world model generates at
most 2048 new tokens, the released code’s cap for its Claude path (its GPT-4o path sets none). The value
function generates at most 1024 new tokens instead of the released 256, because Qwen’s longer reasoning
was otherwise cut off before the Status: line on about 10–16% of samples, which then scored 0. We
simulate all K=3 candidates and do not apply the optional LLM filter that prunes candidates before
simulation.
In the unified diffs below, red “-” lines are original WebDreamer text removed or replaced, green “+”
lines are our replacements, and black lines are unchanged.
P.1 WebDreamer world model: state change
WebDreamer state-change prompt (step 0)
--- ORIGINAL WebDreamer PROMPT
+++ OUR CURRENT RUNTIME PROMPT
-You are an agent that predicts the effect of an action on a webpage.
-You will be given a screenshot of a webpage and an operation to perform on the webpage.
-You are required to predict the changes that will occur on the webpage after the operation is
performed, such as the appearance of new elements, the disappearance of existing elements, or
changes in the content of existing elements.
+You are an agent that predicts the effect of an action on a computer screen.
+You will be given a screenshot of a computer desktop and an operation to perform on it.
+You are required to predict the changes that will occur on the screen after the operation is
performed, such as the appearance of new elements, the disappearance of existing elements, or
changes in the content of existing elements.
The operation type and the element to operate will be provided in the prompt.
Directly output'State changes: ...'and don't output anything else.
Try to be as comprehensive and detailed as possible.
P.2 WebDreamer world model: imagined action proposal
WebDreamer action-proposal prompt
--- ORIGINAL WebDreamer PROMPT
+++ OUR CURRENT RUNTIME PROMPT
-You are an autonomous intelligent agent tasked with navigating a web browser.
-You will be given web-based tasks.
+You are an autonomous intelligent agent tasked with operating a computer desktop (Ubuntu).
+You will be given computer tasks.
These tasks will be accomplished through the use of specific actions you can issue.
Available operation types include
```click```:
-Use this to click.
+Use this to click an element (left, right or middle button; single or double click).
```type```:
50

Use this to type the content into a field.
```hover```:
Hover over an element.
-```press```:
+```hotkey```:
Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
```scroll down```or```scroll up```:
-Scroll the page up or down.
-URL Navigation Actions:
-```goto```:
-Navigate to a specific URL.
-```go_back```:
-Navigate to the previously viewed page.
-```go_forward```:
-Navigate to the next page (if a previous'go_back'action was performed).
+Scroll an element up or down.
+```drag_and_drop```:
+Drag an element from a start position to an end position.
+```highlight_text_span```:
+Select visible text between a starting and an ending phrase.
+```set_cell_values```:
+Set spreadsheet cell values.
+Application Actions:
+```open```:
+Open an application or file by name.
+```switch_applications```:
+Switch to an already open application.
Completion Action:
```stop```:
Issue this action when you believe the task is complete.
If the objective is to find a text-based answer, provide the answer in the bracket.
P.3 WebDreamer world model: imagined state change
WebDreamer imagined state-change prompt
--- ORIGINAL WebDreamer PROMPT
+++ OUR CURRENT RUNTIME PROMPT
-You are an agent that predicts the effect of an action on a webpage.
-You will be given a screenshot of a webpage, a sequence of actions and state changes applied to
the initial screenshot, and an operation to perform on the webpage.
-You are required to predict the new changes that will occur on the webpage after the operation
is performed, such as the appearance of new elements, the disappearance of existing elements, or
changes in the content of existing elements.
+You are an agent that predicts the effect of an action on a computer screen.
+You will be given a screenshot of a computer desktop, a sequence of actions and state changes
applied to the initial screenshot, and an operation to perform on it.
+You are required to predict the new changes that will occur on the screen after the operation is
performed, such as the appearance of new elements, the disappearance of existing elements, or
changes in the content of existing elements.
The operation type and the element to operate will be provided in the prompt.
Directly output'State changes: ...'and don't output anything else.
Try to be as comprehensive and detailed as possible.
P.4 WebDreamer value function
WebDreamer value prompt
--- ORIGINAL WebDreamer PROMPT
+++ OUR CURRENT RUNTIME PROMPT
-You are an expert in evaluating the performance of a web navigation agent.
-The agent is designed to help a human user navigate a website to complete a task.
-Given the user's intent, the agent's action history, the current state of the webpage, your goal
is to decide **whether the simulated steps by the agent indicate a successful execution of the
user intent**.
51

+You are an expert in evaluating the performance of a computer-use agent.
+The agent is designed to help a human user operate a computer desktop to complete a task.
+Given the user's intent, the agent's action history, the current state of the screen, your goal
is to decide **whether the simulated steps by the agent indicate a successful execution of the
user intent**.
-In particular, if the predicted state (i.e., the current state represented by the last image
plus all the predicted changes so far) corresponds to a successful final state.
+In particular, if the predicted state (i.e., the current state shown under "Current state of the
screen" plus all the predicted changes so far) corresponds to a successful final state.
If it is a failure but it looks like the simulated steps are on the right track towards success,
you should also output as such.
-Note that, in the simulated steps, all the state changes are predicted by the agent's world
model, and they may not actually be faithful to the real website interactions (e.g., some
proposed actions may not be avaiable in a realistic website).
+Note that, in the simulated steps, all the state changes are predicted by the agent's world
model, and they may not actually be faithful to the real computer interactions (e.g., some
proposed actions may not be avaiable in a real application).
You should also account for this in your evaluation (e.g., if the predicted state changes are not
reasonable then it's probably a failure).
+If the first proposed action is agent.fail(), the agent declares the task impossible and stops.
Output "success" only if the action history and the current state of the screen prove the task
cannot be done (e.g., a required setting, option, file, or feature does not exist, or the task
needs a CAPTCHA, a login, or credentials the user did not provide). Otherwise output "failure"
and "no".
*IMPORTANT*
Format your response into two lines as shown below:
Thoughts: <your thoughts and reasoning process>
Status: "success" or "failure"
On the right track to success: "yes" or "no"
Q Visual Ground-Truth implementation details
To find the ground-truth screenshot for each step in the serial option, we first use a QEMU virtual machine
supported by Singularity as our environment for OSWorld. We first save a QEMU snapshot of the current
VM state before evaluating any candidate action. This snapshot saves the exact state of the VM, including
the screen, memory, and disk, allowing us to restore it to this state after each candidate action is executed.
We then execute each candidate action one at a time on the same VM, wait for the interface to settle,
and capture the resulting screenshot as that candidate’s ground-truth next state. After each candidate is
executed, the VM is restored back to the saved snapshot, so the next candidate starts from the exact same
screen state. This makes the comparison fair and deterministic on most of the tasks, with a few exceptions
when the task involves internet access and the content on the screen may change (e.g., Google search
results). See more details in Appendix S. For multi-horizon trajectories (depth >1), the runtime inserts
the bold notice shown below; single-step runs omit it.
[SYSTEM MESSAGE]
You are a desktop computer agent on Ubuntu with internet access that evaluates actions by
considering the observation before the action candidates and the potential outcomes of these
actions. Rank the candidate actions by how effectively each advances toward the goal, considering
both the current observation and actual executed rollout outcomes.
# ACTION SPACE
Use only these exact methods/arguments; no right_click/double_click:
- agent.click(instruction, num_clicks=1, button_type='left', hold_keys=[]) # button_type='right
'|'middle'; double-click=num_clicks=2
- agent.done(return_value=None)
- agent.drag_and_drop(starting_description, ending_description, hold_keys=[])
- agent.fail() # proven impossible: CAPTCHA/login/missing required user account credentials/
unavailable site/absent required setting or option
- agent.highlight_text_span(starting_phrase, ending_phrase)
- agent.hover(instruction)
- agent.hold_and_press(hold_keys, press_keys)
- agent.hotkey(keys)
- agent.open(app_or_filename) # closed app/file; filename, not path; switch if open. Only Google
Chrome, LibreOffice Writer/Calc/Impress, Thunderbird, VSCode, VLC Media Player, GIMP; no other
52

programs
- agent.scroll(instruction, clicks, shift=False) # +up/-down; shift=horizontal; 1~3 lines; small
=1-5, section=5-10, long=10-20; max=50
- agent.set_cell_values(cell_values, app_name, sheet_name) # ALWAYS for sheet cells; formulas
start "="; max 30 cells/call; app=title filename, sheet=tab
- agent.switch_applications(app_code)
- agent.type(element_description=None, text='', overwrite=False, enter=False) # omit
element_description only when keyboard focus/caret is known unchanged from history or immediately
preceding focus action; pointer!=focus; click/app switch/dialog/restore uncertainty requires
grounding; overwrite=True replaces, False appends intentionally; in a multi-line text/code editor
overwrite=True erases the whole document, so select the part to replace (highlight_text_span)
and type over it without element_description; overwrite=True for a single-line GUI input still
showing old text (calculator/command line, rename field; not terminals); if it is already
selected, omit element_description; console commands: no nested single quotes; Enter only submits
(click live-filter results)
- agent.wait(time) # visible progress/load only
**Multi-horizon rollout (depth={depth})**: Each candidate shows a trajectory of {depth} steps
executed on the real VM. The first action is the candidate's proposed action; subsequent actions
were chosen by the policy model based on the resulting screenshot. Evaluate the FULL trajectory;
prefer the candidate whose trajectory makes the most cumulative progress.
# EVALUATION CRITERIA:
Screenshots are ground truth; tutorials are guidance unless they start with'Infeasible'.
1. Feasibility - Target must be visible now; after-screenshots/rollouts must prove the action
worked.
2. Exactness - Advance the task without extra steps.
3. Effectiveness - Make visible progress toward the final state.
4. Completion - Use agent.done() only when already complete.
5. Tutorial alignment - Prefer feasible tutorial actions, but trust pixels when they differ.
6. No regression - Reject broken state, navigation away, or lost progress.
7. Terminal evidence - Consider agent.done()/agent.fail() only with hard screenshot/history proof.
Prefer any visible action needed to complete, save, apply, close a required dialog, or verify
the final artifact.
# REQUIRED DECISION PROCESS
Before deciding:
1. Derive exact task-specific success conditions from the task instruction.
2. Apply all applicable named evaluation criteria to every candidate.
3. Judge progress toward the overall goal, not mere action execution.
4. Compare observed or expected resulting states and cumulative progress across candidates.
5. Decide only after that comparison.
In`thought`, connect screenshot/history evidence to success conditions and criteria; do not
merely narrate screenshots.
# TUTORIAL GROUNDING GUIDANCE (IF TUTORIAL IS PROVIDED):
Prefer feasible tutorial operations that reach task milestones.
# RULES:
(1) **OVERRIDE - Infeasible tutorial**: If a tutorial is provided and starts with'Infeasible',
agent.fail() is the only correct action.
(2) Prefer terminal over GUI, then hotkeys over multiple clicks.
(3) ANTI-LOOP: If current and latest HISTORY screenshots are substantially unchanged, reject any
candidate matching an action's method+target from the last 3 steps. If all feasible candidates
repeat failures, prefer a different agent.* method family and explain the rejection.
(4) Do not treat agent.done(return_value=...) as an infeasibility report; prefer agent.fail() for
proven impossibility.
(5) On the LAST step, only agent.done() or agent.fail() can succeed.
(6) Google Chrome is the ONLY available browser.
(7) TEXT-SELECTION METHOD PREFERENCE: For the same text range, prefer agent.highlight_text_span
(...) over agent.drag_and_drop(...) unless anchors are not visible or the target is non-text. Do
not reject highlight_text_span as unsupported for multi-line selection.
(8) FOCUSED TEXT ENTRY: Omit element_description only when current keyboard focus/caret is known
and unchanged. Then agent.type(text=..., overwrite=..., enter=...) intentionally skips grounding
and clicking. History or the current screenshot must establish the intended focus; any
intervening click, app switch, dialog transition, or restore uncertainty changes the rule. Do not
reject omitted element_description solely when focus evidence is strong. If focus is uncertain,
the target is another field, or only the mouse pointer is over the field, require agent.type(
53

element_description=..., ...) so grounding can locate it.
# EVIDENCE-FIRST VERIFICATION:
Candidate prose, rationale, tutorials, and simulated/text-only trajectories are hypotheses, not
execution evidence. Simulated/text-only trajectories may be FEASIBLE, REJECTED, or UNKNOWN, never
VERIFIED.
- Check exact requested conditions: values, text, counts, range endpoints, selection boundaries,
dates, names, file paths, source/provenance, and persistence; generic change is insufficient.
- Compare the latest HISTORY action with the current screenshot first. Missing expected effect is
failure.
- For each unexecuted candidate, require a visible target and preconditions; mark FEASIBLE; never
claim execution or mark VERIFIED.
- For rollouts, inspect screenshots state by state only for actual executed actions. Mark
VERIFIED only when an actual state visibly proves the effect; cite it (e.g.`Candidate 1 State 2`)
.
- Missing, ambiguous, or unreadable evidence is UNKNOWN, never success.
- Visible proof: selection shows highlight/handles/caret/status; saving shows a completed dialog
plus visible target/location or explicit saved/applied history; formatting shows values;
persistence shows Save/Apply/OK or reopen/check.
- DONE: If any candidate is`agent.done(...)`,`thought`MUST begin`DONE_CHECK: <condition> =
MET | NOT-MET; ...`, cover every requested final condition, then give`CANDIDATE_CHECKS`. MET
means the requested end state holds (a removal/hide goal is MET when the item is gone) and
requires direct current-screen proof or explicit saved/applied confirmation; dialogs, highlights,
menus, and partial edits are not completion. Select done only when every condition is MET.
Missing/incomplete DONE_CHECK, UNKNOWN, or NOT-MET rejects done; choose progress addressing the
missing condition.
Selection rules:
- Prefer verification for required UNKNOWN evidence; prefer recovery after two failures in the
current GUI branch.
- Do not reject all candidates when one can safely reveal missing evidence or recover. If none
advances the task, choose the least-irreversible visible verification/recovery; never unsupported
done/fail.
- Evaluate each candidate by its full observed trajectory, not just the action text.
- Evaluate each candidate independently. Do NOT use the outcome of Candidate A as evidence that
Candidate B is complete.
- If Candidate B is agent.done(), it has no post-action state, so judge it only from the current
screenshot/history.
- Prefer the candidate whose resulting state makes the most progress toward the task goal.
- For multi-step trajectories, evaluate cumulative progress across ALL steps and focus on the
FINAL screenshot.
# TASK
Based on the ACTUAL screenshots above, rank the candidates from most to least progress toward the
task goal.
# THOUGHT FORMAT
In`thought`, output these fields in order:
TASK_CHECK: <condition> = VERIFIED | FAILED | UNKNOWN (<evidence from the CURRENT SCREENSHOT>);
...
LAST_EFFECT: SEEN | NOT-SEEN | UNKNOWN (<evidence from the CURRENT SCREENSHOT>)
CANDIDATE_CHECKS: 0 = VERIFIED | FEASIBLE | REJECTED | UNKNOWN (<evidence>); ...
LAST_EFFECT reports whether the latest executed action visibly changed the screen. It is about
the action
already taken, never about a candidate's rollout: a rollout can never make LAST_EFFECT SEEN.
If LAST_EFFECT is NOT-SEEN, that action had no effect: REJECT every candidate repeating its
method+target
unless its rollout screenshots visibly show the missing effect, and prefer a candidate that
establishes the
missing prerequisite (focus, selection, caret, or an open dialog) or switches to a different
agent.* method family.
If any candidate is`agent.done(...)`,`thought`MUST begin with the`DONE_CHECK:`line required
by the
54

evidence rules above, followed immediately by CANDIDATE_CHECKS, then TASK_CHECK and LAST_EFFECT.
# OUTPUT FORMAT
Output your response in the following JSON format:
{
"thought": "your rationale for the ranking result",
"ranking": [x, x, x] # "indexes of the action candidates, most promising first"
}
[USER MESSAGE]
TUTORIAL/REFERENCE:
{tutorial}
# TASK INSTRUCTION
{instruction}
# HISTORY
{step_history}
# CURRENT SCREENSHOT
{current_screenshot}
# CANDIDATE ACTIONS
{candidate_actions}
# STEP
You are on step {current_step} of maximum {max_steps} steps.
Figure 25: Visual ground-truth judge prompt used to rank candidate action trajectories using screenshots captured
from the virtual machine.
55

R Ground-Truth Error Analysis
We follow the same procedure described in Qian et al. (2026) to analyze errors in the ground-truth visual
rollout. In total, we collect 2,116 failed trajectories from all runs with ground-truth rollouts. We first
sample 30 tasks and manually go through trajectory data: screenshots, policy "thought" tokens, judge
"thought" tokens, and proposed actions. We then identify and summarize the following core categories of
errors that lead to agent failure:
1.The judge fails to understand task success: We find a notable portion of cases where the GT-rollout
judge fails to understand task success, which is a more specific failure mode. GT-rollout judges fail
to understand whether future states satisfy the task instruction. We show an example of this case in
Figure 29.
2.The judge fails to reason over future states: GT-rollout judges can fail to reason over future
states to wrongly judge the progress of the task. We show an example of this case in Figure 32 and
Figure 28.
3.Wrong MLLM-generated tutorial: MLLM-generated tutorials are not correct and the agent doesn’t
know how to recover from wrong steps in the tutorial. We show an example of this case in Figure 26.
4.Incomplete tutorial: The tutorial does not provide fine-grained enough steps to complete the task,
and the policy fails to propose the missing steps by itself.
5.Policy fails to follow the tutorial correctly: The policy itself misunderstands, skips, contradicts, or
wrongly executes a tutorial step. We show an example of this case in Figure 27.
6.Grounding error: The grounding model maps the same action description to different regions in the
visual input, which leads to wrong grounding and thus failure. We show an example of this case in
Figure 31.
7.Observation hallucination in policy: The policy misreads the current state and thus reasons toward
a wrong action. We show an example of this case in Figure 30.
8.Web server error: The web server hosting the task environment has issues, such as the target website
being down. This is an external factor that can lead to failure, and it is not caused by the agent’s
decision-making process.
9.Agent doing correctly but runs out of max steps: The agent is following a correct path but reaches
the 15-step or 50-step limit before completing the task. We show an example of this case in Figure 33.
We then use an MLLM (Gemini 3.8 Flash (Google DeepMind, 2026)) to annotate the rest of the failed
tasks. Given the trajectory data of a task, we prompt it (see Figure 38) to classify failure reasons into the
categories above. Because a failed task might have many root causes, we follow Qian et al. (2026) and ask
the MLLM to classify potential errors into: (1) a primary reason and (2) mentioned (secondary) reasons.
Note that mentioned reasons cannot be duplicated with the primary reason.
To assess the reliability of this MLLM-as-a-judge framework, we run the MLLM on the manually
annotated examples above and find that the agreement rate for the primary reason reaches 90%. Further-
more, we randomly sample 20 examples from MLLM results to examine manually. All of the results are
reasonable and aligned with our predefined categories of error. Based on the MLLM’s results, we plot the
number of primary and mentioned reasons in Figure 3.
56

Task ID:a96b564e-dbe9-42c3-9ccf-b4498073438a
Tutorial.<...>
Once you are on the “Top” page, you can narrow down the timeframe to see which discussion has the most replies in
history:
• Look for the dropdown menu that usually says “this week” or “this month” (located just below the main tabs).
• Change this selection to “All Time”.
•Incorrect tutorial step.The list will refresh. Look at the “Replies” column on the right side of the list. The
discussion at the very top of this list will be the one with the most replies in FlightAware history.
<...>
Correct step.Click “Replies” to sort the discussions by number of replies, then select the discussion at the top of the sorted
list.
The list will refresh. Look at the “Replies” column on the right side of the list. The discussion at the very top of this list
will be the one with the most replies in FlightAware history.
Figure 26: An example of a wrong tutorial. The top panel shows the tutorial given to the agent, and the bottom
panel shows the screenshot where the agent fails.
57

Task ID:eb303e01-261e-4972-8c07-c9b4e7a4922a
Task.Tomorrow, I am scheduled to deliver a talk, and my PowerPoint slides and speaking notes are saved on the desktop.
Help me insert my planned remarks for each slide into the “note” section of the PowerPoint as a reminder. I have completed
this task for some slides; assist me in completing the remaining part.
Failure.The agent should copy the planned remarks from notes.docx and paste them into the active notes area, but
instead it hallucinates and types the slide’s own content into the notes section.
Tutorial.
3. Enable the Notes View
•If the notes input area is not visible at the bottom of the main
workspace, go to the top menu and selectView > Notes.
•Alternatively, you can click the Notes tab located directly above the
main slide workspace.
4. Copy and Paste the Remarks
• Switch back to thenotes.docxwindow.
•Highlight and copy ( Ctrl +C) the specific text corresponding to the
active slide.4. Copy and Paste the Remarks (continued)
•Switch back to the presentation window and click inside the area
labeled “Click to add notes” below the slide image.
• Paste the text (Ctrl+V).
5. Repeat for Remaining Slides
• Select the next slide in the sequence from the left pane.
•Repeat the copy-paste process for all subsequent slides until all
planned remarks from the Word document are successfully transferred
to the PowerPoint file.
(a) Step 6.
 (b) Step 7.
(c) Step 8.
 (d) Step 9.
Figure 27: Example of agents failing to follow the tutorial correctly (continued on next page). The policy should
copy planned remarks from notes.docx and paste them into the PowerPoint notes section, but instead types slide
content directly into the notes area.
58

Tutorial-consistent action (should have done):
Copy the remaining remarks fromnotes.docx, return to the presentation, and paste them into the active notes area.
Executed action at Step 9:
agent.type(element_description="the ‘Click to add Notes’ area",
text="Course Essentials\n... Student can...", enter=True)
Other candidate actions:
agent.switch_applications(app_code="libreoffice-writer")
agent.done()
Policy thought at Step 9.The Notes area for Slide 2 is active and ready for pasting the remarks. I have already copied
the text for Slide 2 from the notes.docx file in a previous step. The next logical action is to paste the text into this active
Notes area using Ctrl+V. This aligns with the tutorial’s instruction to paste the text after switching back to the presentation
window.
Figure 27: Example of agents failing to follow the tutorial correctly (continued).
59

Task ID:7b7617bd-57cc-468e-9c91-40c4ec2bcb3d
Judge tokens.GT-Rollout judge: The goal is to set the minimum number of undo steps to 100. The current screenshot
shows the “Document History” section, but the specific “Minimal number of undo levels” setting is not visible. Previous
scrolling attempts have not revealed it. The “Document History” section might need to be expanded or further scrolled to
locate the setting. However, the hotkey action did not yield any visible progress toward the goal. The best course of action
is to continue scrolling within the “Document History” section to locate the desired setting.
(a) Current state.
Chosen Action.
agent.scroll(instruction="at the center of the windows Preference", clicks=-5)
(b) GT future state.
Figure 28: An example where the judge fails to reason about task progress using the future state. The judge
recommends continued scrolling from the current state, even when it realizes the GT future state does not show
progress.
60

Example:Fail to find the target state.
Task ID:06fe7178-4491-4589-810f-2e2bc9502122
Task.Can you make my computer bring back the last tab I shut down?
(a) State 1.
Chosen Action 1.agent.hotkey(keys=[’ctrl’, ’shift’, ’t’])
(b) State 2.
Index 0 (should have chosen):
agent.done()
Index 1 (chosen):
agent.hotkey(keys=[’ctrl’, ’shift’, ’t’])
Index 2:
agent.click(instruction="the ’Tripadvisor’ tab in the browser",
num_clicks=1, button_type=’left’, hold_keys=[])
Judge tokens.GT-Rollout judge: The task is to restore the last closed tab. The tutorial confirms that Ctrl+Shift+T is the
correct method. Candidate 1 successfully executed this shortcut and brought up the history page, which indicates the tab
was restored. Candidate 0 does nothing and is not useful. Candidate 2 clicks on the current tab, which does not restore a
closed tab. Therefore, Candidate 1 is the best choice as it directly fulfills the task.
Figure 29: An example where the judge fails to understand task success. The judge selects the restore-tab shortcut
in State 1, observes State 2 after that action, and still selects the highlighted restore-tab option again.
61

Example:Wrong reasoning over the current state.
Task ID:82bc8d6a-36eb-4d2d-8801-ef714fb1e55a
Task.On next Monday, look up a flight from Mumbai to Stockholm.
Policy thought.The current date selected is May 21, 2026, which is a Wednesday. The task requires selecting the upcoming
Monday. The next Monday after May 21 is May 26, 2026. I need to click on May 26 in the calendar to set the correct
departure date.
Figure 30: Wrong reasoning over the current state. The policy misreads the currently selected departure date and
reasons toward an incorrect next-Monday target.
62

Example:Wrong grounding.
Task ID:37608790-6147-45d0-9f20-1137bb35703d
Task.The information is mixed in one field. Help me split them and fill in the columns of First Name, Last Name and
Rank. Finish the work and don’t touch the original data.
Grounded action.agent.drag_and_drop(
starting_description=‘the fill handle of cell B2’,
ending_description=‘cell B22’, hold_keys=[])
(a) Grounding visualization.
(b) Resulting state after the grounded action.
Figure 31: Wrong grounding example. The grounding model maps the drag-and-drop action to an incorrect
spreadsheet region, selecting the wrong range in the resulting state.
63

Example:Failure to reason over future states.
Task ID:67890eb6-6ce5-4c00-9e3d-fb4972699b06
Task.I am an NLP researcher. Check out the best long paper awards of ACL from 2019 to 2022 and record the title ,
year ,author list , andPDF link into table best_awards_acl.xlsx on the desktop. Separate authors only by commas,
use official ACL Anthology URLs, not arXiv, and sort rows by year in descending order.
(a) State 1.
 (b) State 2.
Index 0 (should have chosen):
agent.type(element_description="the search bar on the ACL Anthology website",
text="ACL 2021", enter=True)
Index 1 (chosen):
agent.switch_applications(
app_code="libreoffice-calc")
Index 2:
agent.hotkey(keys=["ctrl", "s"])
Judge tokens.GT-Rollout judge: Candidate 1 switches to LibreOffice Calc, showing the spreadsheet
best_awards_acl.xlsx with data already entered for the 2022 award, which directly progresses the
task of recording award information. Candidate 0 results in a failed search, and Candidate 2 triggers
a save dialog unrelated to the task. Selected: 1.
(c) State 2 future.
Figure 32: The judge fails to reason over the future. The judge selects the candidate that returns to the partially
completed spreadsheet because it observes existing 2022 data, instead of preferring the candidate that continues
searching for missing ACL award entries.
64

Example:Agent runs out of max steps.
Task ID:6f4073b8-d8ea-4ade-8a18-c5d1d5d5aa9a
Task.I now want to count the meeting cities of the three machine learning conferences in the past ten years from 2013 to
2019 (including 2013 and 2019). I have listed the names and years of the conferences in Excel. Please fill in the vacant
locations.
Failure step.Step 50.
Figure 33: An example where an agent runs out of steps. The agent is still working on the spreadsheet at step 50 but
has not completed all missing conference locations before reaching the step limit.
S Example of stochastic dynamics in OSWorld environment
We provide examples of rollout mismatch issues in the OSWorld environment in Figures 34–37. In these
cases, the screenshot captured in the rollout VM differs from the screenshot captured in the real VM after
executing the same action.
T GT-Rollout Failure Analysis Prompt
You are an expert desktop UI-agent and GT-rollout failure analyst.
Return JSON only with this exact top-level shape:
{
"primary_reason": {
"category": "grounding error" | "tutorial error" | "tutorial is not completed" | "observation
hallucination in policy thought" | "agent doing correctly but runs out of max steps" | "web
server error, agent cannot connect" | "the judge fails to reason over the future state" | "the
judge fails to reason over the future state, fails to understand task success" | "agents fail to
follow the tutorial correctly",
"error_step": integer or null,
"details": string
},
"mentioned_reasons": [
{
"category": "grounding error" | "tutorial error" | "tutorial is not completed" | "
observation hallucination in policy thought" | "agent doing correctly but runs out of max steps"
| "web server error, agent cannot connect" | "the judge fails to reason over the future state" |
"the judge fails to reason over the future state, fails to understand task success" | "agents
65

fail to follow the tutorial correctly",
"has_reason": true,
"error_step": integer or null,
"details": string
}
],
"step_evidence": [
{
"step": integer or null,
"screenshot_ids": [integer],
"action": string,
"grounded_action": string,
"details": string
}
],
"reason_presence": {
"grounding error": boolean,
"tutorial error": boolean,
"tutorial is not completed": boolean,
"observation hallucination in policy thought": boolean,
"agent doing correctly but runs out of max steps": boolean,
"web server error, agent cannot connect": boolean,
"the judge fails to reason over the future state": boolean,
"the judge fails to reason over the future state, fails to understand task success": boolean,
"agents fail to follow the tutorial correctly": boolean,
}
}
Use only category strings exactly as listed.
Category definitions:
- grounding error: The high-level action or selected candidate is correct, but the grounded UI
operation targets with (x,y) coordinate are wrong, maybe wrong coordinates, window, field, or
application state. For example, the agent outputs "click the'Submit'button", but the grounded
UI operation clicks on a wrong position that is not the'Submit'button.
- tutorial error: tutorials are not correct and the agent doesn't know how to recover from wrong
steps in tutorials.
- tutorial is not completed: the tutorial doesn't provide enough fine-grained actions to complete
the task, and the agent fails to propose these fine-grained actions correctly by itself.
- observation hallucination in policy thought: policy hallucinates wrong observation in its
thought process, which leads to wrong next action. For example, the policy hallucinates that it
has already opened the target application and thus tries to interact with the application, which
leads to failure in completing the task.
- agent doing correctly but runs out of max steps: The agent appears to be following a valid path
and has not made a decisive wrong action, but the task remains incomplete because the run
reaches the configured max step limit, including either 15-step or 50-step runs. This can be
caused by any of the above reasons or by other unknown reasons; use it as primary only when
exhausting the step budget is the latest decisive failure.
- web server error, agent cannot connect: the web server hosting the task environment has issues,
such as a website being down.
- the judge fails to reason over the future state: Use when the GT-rollout judge/selector is the
decisive failing component: it selects, approves, or keeps preferring a wrong candidate/action
because it misreads the current screenshot, candidate rollout screenshots, or future/resulting
state. If the policy proposed a correct tutorial-following action but the judge chose a different
candidate by hallucinating that the wrong future state was successful, this category should be
primary, not tutorial-following failure. In mentioned_reasons, this category is mutually
exclusive with the more specific task-success category below.
- the judge fails to reason over the future state, fails to understand task success: The policy
or GT-rollout judge fails to understand whether the current/future state satisfies the task
instruction, especially when it terminates with agent.done() despite visible evidence that the
task is still incorrect. Use this more specific category instead of the broader judge future-
state category when task-success or termination evidence is involved; if this category appears in
mentioned_reasons, do not also include the broader judge future-state category.
- agents fail to follow the tutorial correctly: Use when the policy/agent itself misunderstands,
skips, contradicts, or wrongly executes a tutorial step, and that policy/agent behavior is the
decisive cause of failure. Do not use this as primary when the policy/proposed actions include
the correct tutorial-following action but the GT-rollout judge selects a different wrong
candidate; in that case use the judge future-state reasoning category as primary.
66

Do not output "other", "unknown", "unclear", or any category not listed above. If no category
fits perfectly, choose the closest listed category and explain the mismatch in details.
The primary_reason must be the latest decisive reason that directly explains the final failed
outcome, usually associated with steps near the end of the trajectory. Earlier mistakes should be
secondary mentioned_reasons unless there is no later decision that preserves or confirms the
failure.
Critical distinction between tutorial-following failure and judge future-state failure:
- Use the tutorial-following category only when the policy/agent itself misunderstands, skips,
contradicts, or wrongly executes a tutorial step, and that policy/agent action is the decisive
cause.
- Use the judge future-state reasoning category when the GT-rollout judge selects, approves, or
keeps preferring a wrong candidate/action because it misreads the current screenshot, misreads
candidate/future screenshots, or hallucinates that an incorrect future state satisfies the task.
- If a correct tutorial-following candidate/action was available or proposed, but the judge chose
a different candidate by claiming its resulting state was correct, the primary_reason must be
the judge future-state reasoning category. The tutorial-following category may be mentioned only
if the policy/agent also independently failed to follow the tutorial.
If the final or near-final step marks the task done even though screenshots/evaluator evidence
show the task is not successful, prefer the task-success/termination category as primary over
earlier action or grounding errors.
mentioned_reasons must include only secondary categories with evidence in the trajectory. Do not
include the same category as primary_reason in mentioned_reasons.
The two judge future-state categories are mutually exclusive in mentioned_reasons. If "the judge
fails to reason over the future state, fails to understand task success" applies anywhere, do not
include "the judge fails to reason over the future state" in mentioned_reasons; use only the
more specific task-success category for judge evidence.
reason_presence must contain every allowed category with true or false.
Keep the JSON compact: primary_reason.details and each mentioned_reasons.details must be one
short sentence.
Return at most 6 mentioned_reasons entries, only for categories with concrete evidence.
Return at most 6 step_evidence entries total, prioritizing the primary reason step and the latest
decisive evidence near the end.
step_evidence must cite screenshot IDs from the provided Image index. Do not invent screenshot
IDs. Do not generate screenshots. The renderer will embed the referenced images.
For each step_evidence item, include the trajectory step, the high-level action proposed by the
policy, the grounded/executed action from traj.jsonl or pre_grounded fields, and one short
sentence explaining why the screenshot/action matters.
Use the evaluation JSON and result details as the source of truth for the final failed state.
Use GT-rollout candidate images/actions to decide whether the failure came from policy proposal,
grounding, rollout prediction, selection, tutorial conditioning, evaluator issues, or termination
behavior.
Do not invent evidence. If evidence is weak, say so in details.
No markdown fences. No prose outside JSON.
Analyze this failed OSWorld GT-rollout task.
You are given the full task/evaluation bundle, the full traj.jsonl text, and every raster image
referenced by the trajectory. Images have been preprocessed with max_image_pixels={
max_image_pixels}; they are not raw screenshots.
Task instruction:
{task_instruction}
Image index:
{image_index}
Classify the primary failure reason and every category that is mentioned by evidence in the
trajectory.
Also produce compact step_evidence entries for the screenshots/steps that justify the
classification. Refer to screenshots by Image number only.
Task bundle and full trajectory:
{bundle_markdown}
Figure 38: Gemini prompt used to classify failed GT-rollout trajectories and extract compact evidence for the failure
analysis.
67

U LLM Usage Statement
The authors used LLM-based assistants for limited editorial and technical support, including grammar and
clarity edits, LaTeX debugging, and draft phrasing for non-substantive text. We also use LLM-assisted
programming like Claude Code and Codex to implement infrastructure code for ground-truth visual
rollouts and other baselines. All LLM-assisted text and code changes were inspected and revised by the
authors. The authors are responsible for the final manuscript, including all claims, analyses, tables, figures,
citations, and conclusions.
68

(A) Screenshot captured in a rollout VM.
(B) Screenshot captured in a real VM.
Figure 34: Rollout vs. real VM (case 0).
69

(A) Screenshot captured in a rollout VM.
(B) Screenshot captured in a live VM.
Figure 35: Rollout vs. live VM (case 1).
70

(A) Screenshot captured in a rollout VM.
(B) Screenshot captured in a live VM.
Figure 36: Rollout vs. live VM (case 2).
71

(A) Screenshot captured in a rollout VM.
(B) Screenshot captured in a live VM.
Figure 37: Rollout vs. live VM (case 3).
72
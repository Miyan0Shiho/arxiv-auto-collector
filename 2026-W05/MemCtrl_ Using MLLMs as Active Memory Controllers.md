# MemCtrl: Using MLLMs as Active Memory Controllers on Embodied Agents

**Authors**: Vishnu Sashank Dorbala, Dinesh Manocha

**Published**: 2026-01-28 18:31:17

**PDF URL**: [https://arxiv.org/pdf/2601.20831v1](https://arxiv.org/pdf/2601.20831v1)

## Abstract
Foundation models rely on in-context learning for personalized decision making. The limited size of this context window necessitates memory compression and retrieval systems like RAG. These systems however often treat memory as large offline storage spaces, which is unfavorable for embodied agents that are expected to operate under strict memory and compute constraints, online. In this work, we propose MemCtrl, a novel framework that uses Multimodal Large Language Models (MLLMs) for pruning memory online. MemCtrl augments MLLMs with a trainable memory head μthat acts as a gate to determine which observations or reflections to retain, update, or discard during exploration. We evaluate with training two types of μ, 1) via an offline expert, and 2) via online RL, and observe significant improvement in overall embodied task completion ability on μ-augmented MLLMs. In particular, on augmenting two low performing MLLMs with MemCtrl on multiple subsets of the EmbodiedBench benchmark, we observe that μ-augmented MLLMs show an improvement of around 16% on average, with over 20% on specific instruction subsets. Finally, we present a qualitative analysis on the memory fragments collected by μ, noting the superior performance of μaugmented MLLMs on long and complex instruction types.

## Full Text


<!-- PDF content starts -->

MemCtrl: Using MLLMs as Active Memory Controllers on Embodied
Agents
Vishnu Sashank Dorbala Dinesh Manocha
University of Maryland, College Park
Abstract
Foundation models rely on in-context learn-
ing for personalized decision making. The
limited size of this context window necessi-
tates memory compression and retrieval sys-
tems like RAG. These systems however often
treat memory as large offline storage spaces,
which is unfavorable for embodied agents that
are expected to operate under strict mem-
ory and compute constraints, online. In this
work, we propose MemCtrl, a novel frame-
work that uses Multimodal Large Language
Models (MLLMs) for pruning memory online.
MemCtrl augments MLLMs with atrainable
memory headµthat acts as a gate to determine
which observations or reflections to retain, up-
date, or discard during exploration. We evalu-
ate with training two types ofµ, 1) via an of-
fline expert, and 2) via online RL, and observe
significant improvement in overall embod-
ied task completion ability onµ-augmented
MLLMs. In particular, on augmenting two low
performing MLLMs with MemCtrl on mul-
tiple subsets of the EmbodiedBench bench-
mark, we observe thatµ-augmented MLLMs
show an improvement of around16%on aver-
age, with over20%on specific instruction sub-
sets. Finally, we present an qualitative anal-
ysis on the memory fragments collected by
µ, noting the superior performance ofµaug-
mented MLLMs on long and complex instruc-
tion types.
1 Introduction
An overarching goal of Embodied AI is the devel-
opment of a generalist agent that can perform con-
sistently well with high success on diverse tasks,
environments and instructions (Szot et al., 2025).
A common paradigm to achieve this has been to
utilize foundation models to develop task solv-
ing frameworks (Mu et al., 2023; Yang et al.,
2025). While a few of these methods general-
ize well to diverse tasks and instructions (Driess
Figure 1:Overview: We present MemCtrl, a novel
memory filterting scheme to improve decision mak-
ing performance onsmallMLLMs tackling embodied
tasks. Our approach proposes atrainable memory head
(green box labeled “Memory”) that learns to actively
filter out redundant observations on-the-go. This form
ofactivefiltering alleviates issues with inefficient re-
trieval from stored observations, while also enabling
scalability as a detachable memory head.
et al., 2023; Zawalski et al., 2024), they are con-
strained by high training costs, prohibiting them
from quickly being able to adapt to novel real-
time settings where the data is out of distribution.
Further, finetuning large foundation models incurs
significant computational capacity, proving to be a
significant hurdle in the democratization of these
methods (Liang et al., 2022), especially in the con-
text of robotics, where computation on the edge is
of vital importance.
An alternative, more feasible paradigm has been
a modular system where foundation models are
used in conjunction with memory banks (Zhong
et al., 2023; Wang et al., 2024) of past expe-
riences and reflections. Foundation models in-
cluding very large Multimodal Large Language
Models (MLLMs) such as LLaMA 4 (Touvron
et al., 2023) and Deepseek V3 (DeepSeek-AI
et al., 2025) are limited by the size of their con-arXiv:2601.20831v1  [cs.AI]  28 Jan 2026

Figure 2:Comparison with Prior Work: We present MemCtrl, a novel approach to train“memory heads”to
filter observations on the go. Prior work either used the entirety of stored observations as context (left) or filtered
them via a variety of Retrieval Augmented Generation (RAG) based schemes (red arrows), both of which assume
the parsing of large amounts of data offline. MemCtrl introduces transferrable heads to use on MLLM backbone
(green arrows) to actively filter observations.
text window, and developing methods to refine
and selectively pass memory as context is an ac-
tive area of research (Wu et al., 2025). Prior
work in this area includes use of intrinsic model
editing techniques for memory injection (Mitchell
et al., 2022; Meng et al., 2023), or extrinsic inter-
actions with episodic logs, Retrieval-Augmented
Generated (RAG) (Gao et al.), or long-range latent
states (Park et al., 2023; Wang et al., 2023).
While both paradigms have shown improved
performance in multi-step reasoning, their imple-
mentation on embodied robot agents raises practi-
cal issues. Embodied agents providing assistance
often use small models (<20B parameters) that
work locally on-device, with often limited or only
cloud access to large memory storage. Moreover,
these agents need to generalize to novel settings,
making it preferable to have modular, lightweight
segments that are easilytransferrable.
To model a more efficient memory frame-
work for embodied agents, we draw inspiration
from how humans store memories of experiences.
While performing various embodied tasks, hu-
mans do not accumulate every observation for
later retrieval, but rather learn to actively filter
out only certain vital fragments that we assume to
be relevant to our task (He et al., 2025). Whenqueried, we reconstruct the missing fragments of
memory through commonsense reasoning. This
makes us humans highly efficient reasoners even
with limited storage.
We aim to endow compact embodied agents
with a similar ability: rather than relying on
large external memory banks or complex retrieval
pipelines, the agent must actively learn to store
vital memories while filtering redundant ones on
the go. Learning this skill across a wide range of
tasks would enable scalable self-improvement un-
der tight computational and memory budgets.
Main Results:To address these issues, we present
MemCtrl, a transferrable memory augmentation
scheme that aims to improve the embodied deci-
sion making performance ofsmallmodels. Mem-
Ctrl introduces a trainable memory head that
learns to selectively storememories of importance,
increasing both parameter and memory efficiency
for self-improving embodied agents. Our contri-
butions are as follows:
•Active Memory Filtering:We introduce
two lightweight memory headsµtrained on
top of a frozen MLLM backbone to actively
filter observations to determine which to keep
and which to discard in memory. Unlike
prior retrieval-based work involving filtering

large observational data offline,µenables the
MLLM to engage in real-time filtering, which
is particularly useful in memory-constrained
settings involving small models.
•Transferrable Heads:µis model-agnostic
and attaches to any off-the-shelf MLLM
without having to finetune or edit the back-
bone to remove redundant observations for
more prudent decision making. This modu-
lar design allows MemCtrl to transfer across
embodied setups and vision-language back-
bones, enabling its scalable transfer across
embodied agents in diverse settings.
•Improving Small Model Performance:Fi-
nally, attachingµto the worst performing
agents on the Habitat (Puig et al., 2023) and
ALFRED (Shridhar et al., 2020) splits of Em-
bodiedBench (Yang et al., 2025) shows a sig-
nificant improvement on task performance of
around16%on average, while also storing
significantly fewer observations. This effi-
ciency makes MemCtrl favorable for real-
world deployment.
2 Related Work
2.1 MLLMs in Embodied AI
Several works in recent literature have lever-
aged large language models (LLMs) for high-level
robot planning. For example, Ahn et al. (2022)
introduce a framework (SayCan) where an LLM
translates natural-language instructions into feasi-
ble robot actions constrained by a set of learned
skills. This approach demonstrated that LLMs can
provide semantic task knowledge, but it relies on
a fixed library of affordance-grounded skills and
struggles with adapting to novel situations. Re-
cent multimodal LLMs extend this idea by directly
integrating visual inputs. For instance, PaLM-E
(Driess et al., 2023) is a vision-language model
that outputs robotic actions, achieving good gen-
eralizability across manipulation and navigation
tasks. However, PaLM-E requires a lot of train-
ing data, limiting its real-time adaptability. Sim-
ilarly, RT-2 (Zitkovich et al., 2023) augments a
vision-language model with web-scale pretraining
to create a vision-language-action (VLA) agent for
improved zero-shot object understanding. How-
ever, it makes use of short temporal contexts with-
out an explicit memory mechanism. In contrast,
our work uses an MLLM as an active memorycontroller, enabling the agent to retain and re-
call cross-modal information over long horizons.
Our design empowers embodied agents with small
models to handle complex, extended tasks without
requiring large-scale training.
2.2 Memory-Augmented Agents
Recent works have explored augmenting LLM-
based embodied agents with memory to address
challenges with long-horizon task solving. Mai
et al. (2023) propose using an LLM itself as a
‘robotic brain’ that maintains an egocentric mem-
ory of the agent’s observations and dialogue. Their
system shows that a textual memory stream can
help the agent refer back to important context, im-
proving consistency in multi-step tasks. Another
approach is to attach an external memory to the
agent. In the HELPER framework (Sarch et al.,
2023), they maintain a repository of dialogue-to-
action examples, retrieving relevant past interac-
tions to condition the LLM when parsing new in-
structions. They show that dynamic memory of
prior events or user preferences can overcome the
limitations of fixed prompts or short context win-
dows to improve task completion success. How-
ever, this still relies on a separate module for popu-
lating memory, with the foundation model having
a passive role. In contrast, our work introduces
a framework that enables the MLLM toactively
control what and what not to store in the agent’s
memory. Figure 2 highlights this idea. The Mem-
Ctrl module takes in the embedding of the cur-
rent observation from the MLLM and determines
whether or not the observation should be stored.
3 MLLM-based Embodied Agents
LetMbe an MLLM,O cbe the current observa-
tion, andIbe the instruction. A baseline method
utilizesMto translate the observations and in-
structions into actionable outputs for the agent as
follows:
a=M(O c,I),
wherea∈ Asuch thatArepresents a set of fea-
sible actions for the agent in the environment. Us-
ingMas a prior in this “zero-shot” manner leads
to subpar performance, since the agent does not
have any continual context of the environment,
and takes actions solely based off of the current
image observed and its capacity for commonsense
reasoning (Dorbala et al., 2022; Majumdar et al.,
2022; Gadre et al., 2023; Shah et al., 2023).

3.1 Memory-Augmented MLLM Agents
One way to improve zero-shot performance ofM
is to provide continued context to the agent as a
set of past observations, i.e.,{O c,I,C}, whereC
is the context passed toM:
C={R i,Oi}i:1→n.
RiandO irespectively represent theithreflection
and observation inntimesteps.
EmbodiedBench (Yang et al., 2025) highlights
that adding history as context greatly influences
performance, particularly when it comes to em-
bodied tasks involving long-horizon instructions.
Further, they highlight benefits to adding memory
in the form of agent reflections, where the agent
reflects on its past interactions with the environ-
ment to refine its future plan.
While adding history improves zero-shot per-
formance,Mis limited by a context windowh,
andn < h. To circumvent this issue, several ap-
proaches explore retrieval pipelines to compress
memory to obtain contextc=F(C,I)forM.F
here is retrieval function that selects the most rele-
vant parts of the whole context given an objective,
which in this case is the instructionI.
Inefficient Retrieval: In alignment to other re-
ported results (Liu et al., 2024; Packer et al.,
2023), we similarly observe that as the size of con-
textn=sizeof(C)increases it leads to more
inefficient retrieval, especially since we are lim-
ited by a context window of sizeh << n. Robot
agents running in the wild often collect observa-
tions at high frequencies (>1observation per
second), which quickly increases the size ofn.
Further, having redundant observations in memory
only adds to this issue, prompting the development
of better strategies for write-time memory control,
especially on small models.
To achieve such active memory control, we pro-
poseMemCtrl, a learnable memory augmentation
scheme that allows active write-time memory con-
trol on the modelM. We describe this in the fol-
lowing section.
4 MemCtrl: Training Memory Heads (µ)
To achieve write-time memory control, we con-
sider three natural augmentations toM, illustrated
in Figure 3. Our objective is to improve de-
cision making onsmallvisual-language models,
which also translates to small context windows
Figure 3:MemCtrl: We experiment with3augmenta-
tions. The simple case acts as a non-trained baseline,
where the MLLM is directly queried about storage. In
the offline supervised case,µis first pretrained using
expert answers from a high performing, expert MLLM
(GPT-4o here). This trained binary classifier then acts
as a head on top of the MLLM backbone. In the Online
RL case, we train the memory head online as a policy
network. We use a sparse reward on task success and a
dense reward on action success. Note that MemCtrl is
trained as a detachable head that takes the visuolingual
MLLM embeddings as input.
C’s, by empowering them to better filter observa-
tions prior to storing them in memory.
Filtering observations in this way fully avoids
the problem of inefficient retrieval described in
the previous section, since the agent’s context is
now driven by its own decisions, much akin to hu-
mans deciding only to remember moments of im-
portance that might be meaningful to them in the
future.
For this, we propose atrainable memory head
µas a simple binary classifier that learns to either
keep or discard memory.µintegrates with theM
backbone, to make decisions online about whether
or not to store the current observation. Following

on the our definitions so far, we get
c=F(C,I),
a=M a(Oc,I, c),
b=M µ(Oc,I, c).(1)
whereb∈ {0,1}is a binary classifier that deter-
mines if the current observation must be added or
discarded. The updated context,C′can then be
written as,
C′=(
C∪ {(O c, a)}b= 1,
Cotherwise.(2)
We consider2ways to integrateµontoM:
•Offline, Fully-Supervised: We first gather
offline data from a high-performingM
treated as an expert. Using the gathered neg-
ative and positive samples, we trainµoffline
as a binary classifier to determine which ob-
servations led to success and which led to
failure. We transfer thispretrainedmemory
head onto a low-performing modelM, as an
expert supervision gate. We train the network
with a binary cross-entropy loss function:
L(y,ˆp) =ylog(ˆp)+(1−y) log(1−ˆp),(3)
wherey i∈[0,1]is the ground-truth label
andˆp iis the predicted probability fromµ
of whether the current observation should be
stored.
•Online RL: We directly train the memory
head and the action head via an online RL
policy. We model two rewards: 1) a sparse
reward for episode success, and 2) a dense
reward for picking valid actions:
R(r, a) =r+1 a∈A,(4)
wherer∈ {0,1}is the binary reward sig-
nal for task completion, and1is an indica-
tor function. In our approach,r= 0for all
steps except for goal-completing ones. These
reward functions ensure thatµpicks helpful
observations and the action heads make valid
decisions.
In both these cases, the memory headµempow-
ers the MLLM to play an active role in filtering
observations, as highlighted in Figure 3. Further,µis a head, and can be transferred across arbi-
trary MLLMs helping alleviate the cost of directly
finetuning MLLMs. Algorithm 1 and 2 in the Ap-
pendix present the details of our algorithm for the
supervised and RL variants, respectively.
5 Experimental Setup
Datasets.Our objective is to improve the perfor-
mance of an MLLM by augmenting it with a mem-
ory head. For this, we choose EmbodiedBench
(Yang et al., 2025) as a benchmark for evaluation,
since it provides us with tools to automate embod-
ied task evaluation on both ALFRED (Shridhar
et al., 2020) and Habitat (Puig et al., 2023) simu-
lators, making it easy to evaluate the performance
of multiple MLLMs on various tasks. We modify
this evaluator with memory heads for our task.
LLM backbones.To showcase improvement,
we choose two low-performing models on Embod-
iedBench,Qwen2-VL-7B-InsandGemma-3-12B-
ITaiming to showcase an improvement in perfor-
mance when augmented with MemCtrl. We run
all models locally on a NVIDIA A5000 GPU.
Baseline: Simple, In-Context Learning.We
promptMfor a binary output on whether or not
to store the current observation in its memory. We
do not train a memory head here, but ablate with
combinations ofµandM.
Memory Heads.Bothµ’s are paramterized as
Linear MLP’s with 3 layers, that map the back-
bone MLLM’s embedding to a binary output.
For the offline, supervisedµ, we first gather ex-
pert data using a high performing MLLM, GPT-4o
which gives us a set ofX= [x 1, x2, . . . , x n]em-
beddings per episode mapped to a binary episode
success or failurel, giving us[n,1]training pairs
for each episode. We ensure balancing of the
dataset with negative and positive samples and
then train the MLP to overfit using a cross-entropy
loss.
For the onlineµ, we similarly define an MLP
as the policy network that predicts a binary out-
come. We define a sparse and a dense reward as
described in the previous section, and train using
REINFORCE (Williams, 1992).
6 Results
Table 1 presents the main results, showcasing the
benefits of MemCtrl across two different LLM

Model EB-ALFRED EB-Habitat
Avg Base Common Complex Spatial Long Avg Base Common Complex Spatial Long
Gemma-3-12B-IT 25.6 32 26 38 20 12 23.0 58 10 24 24 4
Gemma-3-12B-IT +µ Simple 28 41 27 34 16 22 31.2 62 15 30 31 18
Gemma-3-12B-IT +µ Offline Sup. 32.248 31 32 23 27 31.8 66 14 35 27 17
Gemma-3-12B-IT +µ Online RL 27.8 38 29 41 21 26 33.860 13 37 35 24
Qwen2.5-VL-7B-Ins 4.7 10 8 6 0 2 14.3 32 2 26 14 2
Qwen2.5-VL-7B-Ins +µ Simple 9.6 15 10 12 2 14 21.4 39 10 31 14 13
Qwen2.5-VL-7B-Ins +µ Offline Sup. 12.2 9 10 14 2 26 22.839 5 33 17 20
Qwen2.5-VL-7B-Ins +µ Online RL 14.210 13 21 3 24 22.2 37 3 37 16 18
Table 1:Results: We augmentQwen2.5-VL-7B-InsandGemma-3-12B-ITwith the 3 variations of MemCtrl on 5
subsets of EB-ALFRED and EB-Habitat (Yang et al., 2025). Note the improve performance overall of adding the
memory headµ. In particular, we note superior performance on long context and complex instructions, which tend
to be long horizon where memory helps. Performance on Habitat is also much better than ALFRED overall, hinting
that navigation heavy tasks that are common on the Habiat dataset might benefit from memory augmentation more
than manipulation ones common in ALFRED.
backbones,Gemma-3-12B-ITandQwen2.5-VL-
7B-Ins. We pick these two models as they perform
among the worst on the EmbodiedBench bench-
mark, aiming to show improved performance with
the inclusion of our memory head.
Increased Performance withµ.The overall
performance improves across EB-Alfred and EB-
Habitat with adding any type of memory headµ.
This is expected, since any form of memory pro-
vides continual context that is more meaningful
for the MLLM.
In particular, we observe huge improvements on
Long instructions, with results onGemma-3-12B-
ITbumping from12on the baseline to26with the
µRLaugmentation, andQwen2.5-VL-7B-Insgo-
ing from2to24. We believe this to be a result of a
more strategic memory storage needed for longer-
horizon tasks, like with EB-Habitat.
We also observe an overall improvement in the
task performance on complex instructions, where
the instructions are not just long, but also contain
irrelevant information. This is very evident with
the Qwen model, where it goes up3x from6to
21. For instance, the following is the difference
between a base and complex instruction:-
Base: “Move one of the pear items to
the indicated sofa.”
Complex: “When you find the fridge
door open, go ahead and move an bowl
to the sofa; otherwise, transport an ham-
mer to the sofa.”
The agent is expected to finish these tasks in a
fixed set of timesteps, and over time, gathers more
and more information about the environment as
potential context for determining its next action.The base query here is fairly simple, requiring to
track just a single object (‘pear’). In contrast, the
complex query not only has multiple objects to
track (‘fridge, sofa, bowl, hammer’), but is also
sophisticated in its framing, requiring better rea-
soning. While more context would help with bet-
ter reasoning, it also leads to more redundant in-
formation storage, which a trained memory head
can help actively filter.
Performance across Gemma-3 and Qwen2.5.
The baseline performance of Qwen2.5 as reported
by EmbodiedBench is far lower than Gemma-
3. We note much higher performance gains on
Qwen2.5 compared to Gemma-3. Being one of
the worst performing models on EmbodiedBench
due to it’s small size, adding a lightweightµfor
active memory control bumps its performance up.
Qwen2.5-VL +µ RLis comparable to Ovis2-16B,
a model with over twice the number of parameters
that had16%on the original benchmark.
Alfred vs Habitat: Finally, we also notice better
performance on Habitat overall compared to Al-
fred. Tasks in habitat tend to be more navigation
centric, requiring more long-horizon planning. In
contrast, Alfred focuses more on reasoning on cur-
rent observations for manipulating objects. We in-
fer that memory filtration is potentially more im-
pactful on long-horizon navigation tasks, since it
helps reduce redundant frames gathered.
7 Qualitative Analysis
Figure 6 in the Appendix shows a qualitative ex-
ample of MemCtrl in action.

Figure 4:Long Horizon performance on EB-Habitat: We notice that on long horizon tasks, the expert tends
to end the task early by hastily assuming that it is done (finishing after placingoneplate instead ofallplates).
Memory heads highlight unique performance improvements, withµ RLexhibiting a moreexploratorynature by
continuing to placenewobjects at the right counter, andµ Exp.being moreexploitativeby repeating the same
activity over and over, with a single plate. Note: Grayed out images indicate discarded memories.
7.1 Visualization
Figure 5 and Figure 4 compare a base and long
episode, using GPT-4o and Qwen with and with-
out memory. Note Qwen performs poorly on the
EB-Habitat baseline, with only14.3%average vs
GPT-4o that has59%. Augmenting Qwen with
a lightweight memory headµbumps this up to
around22%, both in the case ofµ RLandµ Expert .
In the base case, note the good zero shot perfor-
mance of the GPT-4o agent in being able to com-
plete the task. The complete memory agent how-
ever shows an invalid action after the first step, and
this is also seen with the case of Qwen +µ Expert .
However, with Qwen +µ RLnote the improved
performance in being able to complete the task in
fewer number of steps. Qwen is a much smaller
model than GPT-4o, and with a small memory
head augmentation, it is able to perform at par with
its GPT-4o counterpart.
In the long horizon case, we make a few inter-
esting observations. First, none of the4meth-
ods seem to be able to successfully complete the
task. The instruction requiresallplates to be trans-
ported to the right corner, requiring long-horizon
task planning to keep track of whichinstancesofplates have been moved.
• Thezero-shot expertassumes that the task
has ended after transferring the first plate, and
hence sends no executable plan after being
done. This stops the episode earlier than ex-
pected, hence failing the task.
• In thecomplete memory case, the agent is
overloaded with too many experiences from
the past in its memory, and this floods the
MLLM with too much context, leading to a
bad action being selected (navigate to the re-
frigerator).
• In theQwen +µ RLcase, the agent success-
fully placesoneplate from the sink onto
the right counter. While not ending the
episode early like with the expert, it contin-
ues to place other objects including apples
and wrenches in the vicinity to the right, ul-
timately running out of timesteps. This be-
havior can be associated with more activeex-
ploration, as the agent seeks to explore new
directions to complete the instruction. Note
that the memory head filters out most of these
extraneous observations (images in gray).

Figure 5:Base Performance on EB-Habitat: Here, we compare the performance of GPT-4o vs Qwen2.5-VL-7B-
Ins, with various memory augmentations. While GPT-4V gives superior zero-shot performance, it is a very large
model that is not easily finetunable. In this instance it also takes more steps to complete the task. On the other
hand,µboosts the performance of a significantly weaker model, and in this scenario, even doing it quicker with
µRL. The expert head fails here however, similar to the complete memory, causing the episode to end early. Note:
Grayed out images indicate discarded memories.
• In theQwen +µ Expert case, we observe the
opposite behaviour orexploitation, where the
agent just starts to repeat the same activ-
ity over and over again. It navigates to the
sink, picks up a plate, transfers it to the right
counter, then goes back to the sink, and so
on. This form of exploitative behavior seems
desirable, especially since the agent is tasked
with transporting all plates to the sink. The
expert memory head as a result decides to
keep almost all of the observations, as the
agent keeps doing the right thing in following
the instruction. However, as thesame plate
is constantly being moved, it highlights a lim-
itation of the expert memory head in being
too conservative with classifying significant
memories.8 Conclusions
In this work, we present MemCtrl, a lightweight,
transferrable memory framework that introduces
a memory headµon a backbone MLLM to ac-
tively filtermemories of importance. Instead of
editing the MLLM directly or using external re-
trieval methods like RAG,µis trained as a binary
classifier decide whether or not to store current
observations on the go. We introduce two ways
to trainµ, and present a qualitative and quantita-
tive analysis ofµ-augmented low parameter Qwen
and Gemma models. Our results show significant
performance improvement of around16%on av-
erage, across ALFRED and Habitat splits of the
EmbodiedBench dataset. Further, we note the su-
perior performance on instructions involving com-
plex or long-horizon language.

9 Limitations
Our work has a few limitations. The super-
vised learning method requires expert demonstra-
tions from a stronger model to understand which
observations contribute to success, and the RL
variants suffer from the inefficiencies that come
with sparse reward structures. Future work could
look into designing better reward functions that
can capture theinteresting-nessof an observation
(Gardezi et al., 2021). Furthermore, the benefits of
MemCtrl degrade over short horizons, suggesting
a limited need to train a memory head to filter ob-
servations in more basic settings. Another possi-
ble avenue we are excited to explore is to incorpo-
rate audio observations to increase the complexity
of observations stored. Sim-to-real transfer of our
work via real world experiments is also a potential
extension.
References
Michael Ahn, Anthony Brohan, Noah Brown, Yevgen
Chebotar, Omar Cortes, Byron David, Chelsea Finn,
Chuyuan Fu, Keerthana Gopalakrishnan, Karol
Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu,
Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang,
Rosario Jauregui Ruano, Kyle Jeffrey, and 26 oth-
ers. 2022. Do as i can, not as i say: Grounding
language in robotic affordances. InarXiv preprint
arXiv:2204.01691.
Kwesi Adu Cobbina and Tianyi Zhou. 2025. Where to
show demos in your prompt: A positional bias of in-
context learning. InProceedings of the 2025 Con-
ference on Empirical Methods in Natural Language
Processing, pages 29548–29581, Suzhou, China.
Association for Computational Linguistics.
DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingx-
uan Wang, Bochao Wu, Chengda Lu, Chenggang
Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan,
Damai Dai, Daya Guo, Dejian Yang, Deli Chen,
Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai,
and 181 others. 2025. Deepseek-v3 technical report.
Preprint, arXiv:2412.19437.
Vishnu Sashank Dorbala, Gunnar Sigurdsson, Robin-
son Piramuthu, Jesse Thomason, and Gaurav S
Sukhatme. 2022. Clip-nav: Using clip for zero-
shot vision-and-language navigation.arXiv preprint
arXiv:2211.16649.
Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey
Lynch, Aakanksha Chowdhery, Brian Ichter, , and
1 others. 2023. PaLM-E: An embodied multimodal
language model.arXiv preprint arXiv:2303.03378.
Yufeng Du, Minyang Tian, Srikanth Ronanki,
Subendhu Rongali, Sravan Bodapati, Aram Gal-
styan, Azton Wells, Roy Schwartz, Eliu A Huerta,and Hao Peng. 2025. Context length alone hurts
llm performance despite perfect retrieval.Preprint,
arXiv:2510.05381.
Samir Yitzhak Gadre, Mitchell Wortsman, Gabriel Il-
harco, Ludwig Schmidt, and Shuran Song. 2023.
Cows on pasture: Baselines and benchmarks for
language-driven zero-shot object navigation. In
Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 23171–
23181.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Jiawei Sun, and Haofen Wang.
Retrieval-augmented generation for large language
models: A survey.
Maham Gardezi, King Hei Fung, Usman Mirza Baig,
Mariam Ismail, Oren Kadosh, Yoram S Bonneh, and
Bhavin R Sheth. 2021. What makes an image in-
teresting and how can we explain it.Frontiers in
psychology, 12:668651.
Zizhan He, Maxime Daigle, and Pouya Bashivan.
2025. Building spatial world models from sparse
transitional episodic memories.arXiv preprint
arXiv:2505.13696.
Sanghwan Kim, Daoji Huang, Yongqin Xian, Otmar
Hilliges, Luc Van Gool, and Xi Wang. 2024. Palm:
Predicting actions through language models. InEu-
ropean Conference on Computer Vision, pages 140–
158. Springer.
Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol
Hausman, Brian Ichter, Pete Florence, and Andy
Zeng. 2022. Code as policies: Language model
programs for embodied control.arXiv preprint
arXiv:2209.07753.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language
models use long contexts.Transactions of the Asso-
ciation for Computational Linguistics, 12:157–173.
Jinjie Mai, Jun Chen, Bing Li, Guocheng Qian,
Mohamed Elhoseiny, and Bernard Ghanem. 2023.
LLM as A robotic brain: Unifying egocentric mem-
ory and control.arXiv preprint arXiv:2304.09349.
Arjun Majumdar, Gunjan Aggarwal, Bhavika Devnani,
Judy Hoffman, and Dhruv Batra. 2022. Zson: Zero-
shot object-goal navigation using multimodal goal
embeddings.Advances in Neural Information Pro-
cessing Systems, 35:32340–32352.
Kevin Meng, Arnab Sen Sharma, Alex J. Andonian,
Yonatan Belinkov, and David Bau. 2023. Mass edit-
ing memory in a transformer.Proceedings of the
11th International Conference on Learning Repre-
sentations (ICLR).
Eric Mitchell, Charles Lin, Antoine Bosselut, Christo-
pher D. Manning, and Chelsea Finn. 2022.

Memory-based model editing at scale. InPro-
ceedings of the 39th International Conference on
Machine Learning (ICML), pages 15817–15831.
PMLR.
Yao Mu, Qinglong Zhang, Mengkang Hu, Wen-
hai Wang, Mingyu Ding, Jun Jin, Bin Wang,
Yu Qiao, and Ping Luo. 2023. EmbodiedGPT:
Vision-language pre-training via embodied chain of
thought. InAdvances in Neural Information Pro-
cessing Systems (NeurIPS).
Charles Packer, Sarah Wooders, Kevin Lin, Vivian
Fang, Shishir G. Patil, Ion Stoica, and Joseph E.
Gonzalez. 2023. MemGPT: Towards LLMs as op-
erating systems.arXiv preprint arXiv:2310.08560.
Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai,
Meredith R. Morris, Percy Liang, and Michael S.
Bernstein. 2023. Generative agents: Interactive sim-
ulacra of human behavior. InProceedings of the
36th Annual ACM Symposium on User Interface
Software and Technology (UIST), pages 1–22.
Xavier Puig, Eric Undersander, Andrew Szot,
Mikael Dallaire Cote, Tsung-Yen Yang, Ruslan
Partsey, Ruta Desai, Alexander William Clegg,
Michal Hlavac, So Yeon Min, and 1 others. 2023.
Habitat 3.0: A co-habitat for humans, avatars and
robots.arXiv preprint arXiv:2310.13724.
Gabriel Sarch, Yue Wu, Michael Tarr, and Katerina
Fragkiadaki. 2023. Open-ended instructable em-
bodied agents with memory-augmented large lan-
guage models. InFindings of the Association for
Computational Linguistics: EMNLP 2023, pages
3468–3500, Singapore. Association for Computa-
tional Linguistics.
Dhruv Shah, Ajay Sridhar, Nitish Dashora, Kyle Sta-
chowicz, Kevin Black, Noriaki Hirose, and Sergey
Levine. 2023. Vint: A foundation model for visual
navigation.arXiv preprint arXiv:2306.14846.
Mohit Shridhar, Jesse Thomason, Daniel Gordon,
Yonatan Bisk, Winson Han, Roozbeh Mottaghi,
Luke Zettlemoyer, and Dieter Fox. 2020. Alfred:
A benchmark for interpreting grounded instructions
for everyday tasks. InProceedings of the IEEE/CVF
conference on computer vision and pattern recogni-
tion, pages 10740–10749.
Andrew Szot, Bogdan Mazoure, Omar Attia, Aleksei
Timofeev, Harsh Agrawal, Devon Hjelm, Zhe Gan,
Zsolt Kira, and Alexander Toshev. 2025. From mul-
timodal llms to generalist embodied agents: Meth-
ods and lessons. InProceedings of the Computer
Vision and Pattern Recognition Conference, pages
10644–10655.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro,
Faisal Azhar, Aurelien Rodriguez, Armand Joulin,
Edouard Grave, and Guillaume Lample. 2023.
Llama: Open and efficient foundation language
models.Preprint, arXiv:2302.13971.Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Man-
dlekar, Chaowei Xiao, Yuke Zhu, Linxi “Jim” Fan,
and Anima Anandkumar. 2023. V oyager: An open-
ended embodied agent with large language models.
arXiv preprint arXiv:2305.16291.
Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang,
Shiyang Li, Jingfeng Yang, Qingyu Yin, Zheng Li,
Xian Li, Bing Yin, and 1 others. 2024. Memo-
ryllm: Towards self-updatable large language mod-
els.arXiv preprint arXiv:2402.04624.
Ronald J Williams. 1992. Simple statistical gradient-
following algorithms for connectionist reinforce-
ment learning.Machine learning, 8(3):229–256.
Yaxiong Wu, Sheng Liang, Chen Zhang, Yichao Wang,
Yongyue Zhang, Huifeng Guo, Ruiming Tang, and
Yong Liu. 2025. From human memory to ai mem-
ory: A survey on memory mechanisms in the era of
llms.arXiv preprint arXiv:2504.15965.
Mengge Xue, Zhenyu Hu, Liqun Liu, Kuo Liao,
Shu’ang Li, Honglin Han, Meng Zhao, and Cheng-
guo Yin. 2024. Strengthened symbol binding makes
large language models reliable multiple-choice se-
lectors. InProceedings of the 62nd Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 4331–4344.
Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao,
Cheng Qian, Kangrui Wang, Qineng Wang, Teja V .
Koripella, Marziyeh Movahedi, Manling Li, Heng
Ji, Huan Zhang, and Tong Zhang. 2025. Embodied-
Bench: Comprehensive benchmarking multi-modal
large language models for vision-driven embodied
agents. InProceedings of the 42nd International
Conference on Machine Learning (ICML).
Michał Zawalski, William Chen, Karl Pertsch, Oier
Mees, Chelsea Finn, and Sergey Levine. 2024.
Robotic control via embodied chain-of-thought rea-
soning.arXiv preprint arXiv:2407.08693.
Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou,
and Minlie Huang. 2024. Large language models
are not robust multiple choice selectors.Preprint,
arXiv:2309.03882.
Wanjun Zhong, Lianghong Guo, Qiqi Gao, and Yan-
lin Wang. 2023. Memorybank: Enhancing large
language models with long-term memory.arXiv
preprint arXiv:2305.10250. Preprint. See also
_AAAI_ 2024 version.
Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu,
Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan
Welker, Ayzaan Wahid, Quan Vuong, Vincent Van-
houcke, Huong Tran, Radu Soricut, Anikait Singh,
Jaspiar Singh, Pierre Sermanet, Pannag R. Sanketi,
Grecia Salazar, and 35 others. 2023. Rt-2: Vision-
language-action models transfer web knowledge to
robotic control. InProceedings of The 7th Con-
ference on Robot Learning (CoRL), volume 229 of
PMLR, pages 2165–2183.

A Algorithms
Algorithm 1Training Memory Headsµwith Of-
fline, Supervised Learning.
Require:Labeled ground-truth expert answers
E={(y i,OE
i)}K
i=1, Memory HeadM µ, Ac-
tion HeadM a
1:Pretrainingµ
2:foriin[1, K]do
3:ˆp i=µ(OE
i)
4:Calculate lossL(y i,ˆpi)with Equation 3
5:Updateµ
6:end for
7:Test-time with trainedµ
8:c←F(C, I)
9:a← M a(Oc,I, c)
10:b← M µ(Oc,I, c)
11:ifb= 1then
12:C ← C ∪ {(O c, a)}
13:end if
Algorithm 2Training Memory Headsµwith On-
line, RL.
Require:Current observationO c, InstructionI,
Total ContextC, Memory HeadM µ, Action
HeadM a
1:c←F(C, I)
2:a← M a(Oc,I, c)
3:b← M µ(Oc,I, c)
4:ifb= 1then
5:C ← C ∪ {(O c, a)}
6:end if
7:Calculate reward with Equation 4
8:Updateµ
B Experimental Details
We train the memory headµboth via offline su-
pervised learning with an expert, and online RL. In
both cases,µis an MLP initialized with3layers
to map the hidden MLLM dimension to a binary
output.
Expert, Offline Supervised: Here, we first gather
a dataset using an expert model that performs
well on EmbodiedBench. We use GPT-4o for
this, which has high success rates of56.3%and
59.0%on EB-ALFRED and EB-Habitat respec-
tively (Yang et al., 2025). We gather obser-
vations, actions predicted, and episode success.
We use this to create labels for our loss func-
tion, where if the action predicted was valid(last_action_successvariable in EmbodiedBench
is true) or the episode was successful, we associate
the image and it’s visual state description (text)
to a positive label, and negative otherwise. We
make sure to balance out this dataset for optimal
training. We then trainµwith the gathered labels
as ground truth, and embeddings from the MLLM
backbone (Qwen or Gemma) model taking in the
image and visual scene description as input. We
use a Binary Cross-Entropy Loss with the Adam
optimizer. Our learning rate is1e−3.
RL, Online: Here, we train online using REIN-
FORCE. At each timestep, we sample a binary ac-
tion from the current policy, which in this case is
whether to keep or discard the memory. We then
compute the cumulative reward after executing the
actions predicted by the action head, and update
the policy. To compute the cumulative reward, we
use 1) a dense reward from the action head pre-
dicting a valid action, and 2) a sparse reward from
episode success. Observe that both these rewards
are not directly related to keeping or discarding
memory, but are instead a result of the agent per-
forming expected behavior to improve success.
Modeling direct rewards to determine which mem-
ory to keep or discard is challenging, as they are
tied to the nature of the task. For instance, an
observation containing a white wall might not be
useful for tasks involving picking up objects, but
becomes necessary when it comes to answering
questions about the environment. However, given
enough training data, an agent can learn about
the types of tasks being asked and corresponding
memory fragments to keep in order to successfully
complete them. This is analogous to a lifelong
learning agent that must continually adapt to its
surroundings to provide personalized assistance.
C MLLM Prompts & Decoder
Modifications
We modify the base prompt provided by Embod-
iedBench with stricter constraints on choosing ac-
tions by adding the following:
Important Rules:
The action_id must be picked from the
available ids provided above.
Make sure the action_id matches the
corresponding action_name.
A valid path is guaranteed to exist. If
the image does not contain the required
object for completing the task, you may

Figure 6:Active Filtering: The transferrable memory head performs active filtering, i.e., filters observations on
the go. The grayed out images represent discarded ones. In sequence 1, notice MemCtrl filters out redundant
images taken at odd angles while the agent looks around. In sequence 2, the agent makes a bunch of invalid
actions in the middle of the sequence, before ultimately completing its task of placing a phone on the bed. These
extra observations are filtered out as an outcome of our negative dense reward on invalid actions. In sequence 3,
notice the repeating pattern towards the end, which starts to get filtered out byµ. In each of these cases, the active
involvement of the MLLM in filtering allows for better write-time memory control.
have to navigate there.
We also decode by action name instead of ac-
tion id, and note that this leads to a significant in-
crease in the number of valid actions taken while
planning. For instance, Qwen outputs:
visual_state_description: “The scene
..."
reflection_and_reasoning: “The user
wants ... "
executable_plan: {51: “pick up the
hammer", ...}
We note that the action of “pick up the hammer”
exists and is valid, but it is linked to a different
action_id,28, causing the action decoding to fail
on the base EmbodiedBench code.
We speculate that this may be connected to prior
literature showing that LLMs struggle with multi-
ple choice selection (Xue et al., 2024; Zheng et al.,
2024). The action selection of this work can be
viewed as a multiple choice problem; therefore,
some of the issues of selecting a natural language
phrase with an ID can exist here as well. Further-
more, the action to ID mapping is at the beginning
of the prompt, and there is research that shows that
where information is positioned in the prompt af-
fects reasoning ability (Cobbina and Zhou, 2025).
Another reason could be that smaller sized LLMs
are better at generating descriptive language witha larger token counts, and lack the capacity for in-
teger action mapping requiring consolidation to-
wards a smaller token count (Kim et al., 2024).
For this work, we modify the decoding func-
tion to map theaction nameinstead of the ID and
notice improved performance, especially on the
Qwen model. Further work could potentially try
moving the mapping information to different loca-
tions in the prompt. Another potential cause could
be prompt length, where irrelevant information or
even increased whitespace can degrade LLM ac-
curacy, as shown in Du et al. (2025). Reducing
the prompt to remove unnecessary whitespace or
characters could potentially show improvement in
outputting both the correct action name and the
corresponding ID.
D Analyzing Memory
The memory head enables a form of selective
memory as an alternative to passing a complete
history of observations. To highlight its effective-
ness, we perform an ablation with complete mem-
ory, where none of the observations are discarded,
and all of them are passed back to the MLLM,
while maintaining a token horizon to prevent over-
flow. Table 2 presents results for the complete
memory case on EB-Habitat and EB-ALFRED, in
comparison toµ RL.

Model EB-ALFRED EB-Habitat
Avg Base Common Complex Spatial Long Avg Base Common Complex Spatial Long
Qwen2.5-VL-7B-Ins 4.7 10 8 6 0 2 14.3 32 2 26 14 2
Qwen2.5-VL-7B-Ins +µ Complete 7.8 8 7 18 1 5 8.8 28 0 8 8 0
Qwen2.5-VL-7B-Ins +µ RL 14.210 13 21 3 24 22.237 3 37 16 18
Table 2:Complete vs Selective: We compare the performance of complete memoryµ Complete where all observa-
tions are passed as context to our best performing selective memory agent,µ RL. Note the improved performance of
our selective memory agent, highlighting the importance of being picky about what to store in memory, especially
when model capacity is limited.
EB-ALFRED EB-Habitat
Method Avg Base Common Complex Spatial Long Avg Base Common Complex Spatial Long
Memory Efficiencyµ E(%)↓
Qwen2.5 (Baseline, No Mem.) N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A
Qwen2.5 +µ RL 39.42 35.6 42.8 39.9 38.7 40.1 27.56 39.2 13.7 15.7 22.9 46.3
Qwen2.5 +µ Expert 38.6637.9 40.1 45.6 36.4 33.3 26.3837.2 10.2 14.8 36.5 33.2
Qwen2.5 +µ Complete 100 100 100 100 100 100 100 100 100 100 100 100
Invalid ActionsI ↓
Qwen2.5 (Baseline, No Mem.) 3.50 3.1 2.9 4.6 3.8 3.1 3.0 0.5 5.2 4.8 2.7 1.8
Qwen2.5 +µ RL 2.22 2.0 1.8 2.9 2.1 2.3 1.36 0.4 3.0 2.4 0.6 0.3
Qwen2.5 +µ Expert 2.102.7 1.5 2.4 1.8 2.1 1.020.6 2.2 1.3 0.6 0.4
Qwen2.5 +µ Complete 3.10 2.7 4.2 3.3 2.3 3.0 2.12 1.5 4.6 2.8 0.8 0.9
Table 3:Statistics: Memory efficiency (↑) and invalid actions (↓) across all five splits for EB-Habitat and EB-
Alfred.µ Efor the expert memory head is slightly better on average, but is much worse on ALFRED. This can be
attributed to tasks in ALFRED being slightly harder than Habitat.µaugmented Qwen models also make lesser
number of invalid actions per episode. Overall, adding a memory head shows significant improvement over no
memory and complete memory baselines.
D.1 Statistics
Table 3 highlights statistics gathered across all5
splits on EB-Habitat and EB-Alfred for Qwen and
the memory augmentations. We measure the fol-
lowing across20randomly chosen episodes:
Memory EfficiencyE: To determine the effec-
tiveness of our approach, we compute the memory
efficiency per episode as,
E= 1−Number of Memories Kept
Total Steps Taken
This gives us the fraction of memories that were
stored in the memory bank per episode. When
all the memories have been store,Eresolves to0,
meaning the agent was inefficient in its memory
management.
Invalid ActionsI: This is the average number of
times that the MLLM responds with an invalid ac-
tion for execution. For instance, the MLLM asks
the agent to‘pick up a spoon’, but there is no
spoon visible in the observation.
Inferences: The table highlights the improved
memory efficiency of both our models. In our
main results table, we noted the improved perfor-
mance of the online RL memory head on Qwen.
By multiplying values from that table with the ef-
ficiency values here, we get aweighted efficiencyscore that aims to capture both success and frugal-
ity of memory usage. This can be written as,
W(m) b=Succ.(m)· 
1−µ(m)
E
100!
,
where Succ(m)is the task success (in %) of
methodmon benchmarkb∈ {Alfred, Habitat},
andµ(m)
Eis the corresponding memory efficiency.
We then aggregate this into a single score per
method as,
W(m)=W(m)
Alf.+W(m)
Hab.
2
Substituting the values from Table 1 in the main
text and Table 2 in this supplementary, we get
WExp.= 12.13andWRL= 10.95, meaning that
the expert has slightly better overall weighted effi-
ciency.
We also note that the invalid actionsIare sig-
nificantly lower on both our memory heads when
compared to the baseline and complete memory
approaches. This further highlights the effective-
ness of selective memory whereµactively decides
to populate the context for the MLLM on the go.
Across the 3 sequences in Figure 6, note how
our approach actively filters redundant and re-
peated observations. A trained memory head can

be transferred to novel settings, learning to distin-
guish between interesting and non-interesting ob-
servational data (Gardezi et al., 2021).
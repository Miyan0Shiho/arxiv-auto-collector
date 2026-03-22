# PaAgent: Portrait-Aware Image Restoration Agent via Subjective-Objective Reinforcement Learning

**Authors**: Yijian Wang, Qingsen Yan, Jiantao Zhou, Duwei Dai, Wei Dong

**Published**: 2026-03-17 18:41:36

**PDF URL**: [https://arxiv.org/pdf/2603.17055v1](https://arxiv.org/pdf/2603.17055v1)

## Abstract
Image Restoration (IR) agents, leveraging multimodal large language models to perceive degradation and invoke restoration tools, have shown promise in automating IR tasks. However, existing IR agents typically lack an insight summarization mechanism for past interactions, which results in an exhaustive search for the optimal IR tool. To address this limitation, we propose a portrait-aware IR agent, dubbed PaAgent, which incorporates a self-evolving portrait bank for IR tools and Retrieval-Augmented Generation (RAG) to select a suitable IR tool for input. Specifically, to construct and evolve the portrait bank, the PaAgent continuously enriches it by summarizing the characteristics of various IR tools with restored images, selected IR tools, and degraded images. In addition, the RAG is employed to select the optimal IR tool for the input image by retrieving relevant insights from the portrait bank. Furthermore, to enhance PaAgent's ability to perceive degradation in complex scenes, we propose a subjective-objective reinforcement learning strategy that considers both image quality scores and semantic insights in reward generation, which accurately provides the degradation information even under partial and non-uniform degradation. Extensive experiments across 8 IR benchmarks, covering six single-degradation and eight mixed-degradation scenarios, validate PaAgent's superiority in addressing complex IR tasks. Our project page is \href{https://wyjgr.github.io/PaAgent.html}{PaAgent}.

## Full Text


<!-- PDF content starts -->

1
PaAgent: Portrait-Aware Image Restoration Agent
via Subjective-Objective Reinforcement Learning
Yijian Wang, Qingsen Yan, Jiantao Zhou, Duwei Dai, Wei Dong
Abstract—Image Restoration (IR) agents, leveraging multi-
modal large language models to perceive degradation and invoke
restoration tools, have shown promise in automating IR tasks.
However, existing IR agents typically lack an insight summa-
rization mechanism for past interactions, which results in an
exhaustive search for the optimal IR tool. To address this limi-
tation, we propose a portrait-aware IR agent, dubbed PaAgent,
which incorporates a self-evolving portrait bank for IR tools and
Retrieval-Augmented Generation (RAG) to select a suitable IR
tool for input. Specifically, to construct and evolve the portrait
bank, the PaAgent continuously enriches it by summarizing
the characteristics of various IR tools with restored images,
selected IR tools, and degraded images. In addition, the RAG
is employed to select the optimal IR tool for the input image by
retrieving relevant insights from the portrait bank. Furthermore,
to enhance PaAgent’s ability to perceive degradation in complex
scenes, we propose a subjective-objective reinforcement learning
strategy that considers both image quality scores and semantic
insights in reward generation, which accurately provides the
degradation information even under partial and non-uniform
degradation. Extensive experiments across 8 IR benchmarks,
covering six single-degradation and eight mixed-degradation
scenarios, validate PaAgent’s superiority in addressing complex
IR tasks. Our project page is PaAgent.
Index Terms—Image restoration, agent, portrait bank, rein-
forcement learning.
I. INTRODUCTION
IMAGE Restoration (IR) aims to recover high-quality im-
ages from low-quality ones that are degraded by snow
[1], noise [2], haze [3], rain [4], low-light [5], blur [6], or
composite degradation patterns [7]. Task-specific IR methods
[4], [8]–[18] focus on handling a single degradation; however,
real-world images often suffer from multiple degradations
[19], which significantly limits the practicality of these meth-
ods. To address this, numerous All-in-One (AiO) IR methods
[20]–[29] have been presented, which can handle multiple
degradations simultaneously within a unified model by prompt
*Corresponding author: Qingsen Yan.
Yijian Wang and Qingsen Yan are with the School of Computer Science,
Northwestern Polytechnical University, Xi’an 710072, China, and Qingsen
Yan is also with the Shenzhen Research Institute of Northwestern Polytechni-
cal University, Shenzhen 518057, China (e-mail: wangyijian1017@163.com;
qingsenyan@nwpu.edu.cn).
Jiantao Zhou is with the State Key Laboratory of Internet of Things
for Smart City, University of Macau, Macau 999078, China (e-mail:
jtzhou@um.edu.mo).
Duwei Dai is with the National-Local Joint Engineering Research Center of
Biodiagnosis and Biotherapy, the Second Affiliated Hospital of Xi’an Jiaotong
University, Xi’an 710006, China, and also with Xi’an Jiaotong University,
Xi’an 710049, China (e-mail: duweidai@xjtu.edu.cn).
Wei Dong is with the College of Information and Control Engineering,
Xi’an University of Architecture and Technology, Xi’an 710064, China (e-
mail: dongwei156@outlook.com).
Quality
PerceptionTask
PlanningSequential
StepsTask
Execution
ReflectionIf unsatisfactory
If satisfactory(a) One -step Decision
(b) Multi -step Decision
. . .Quality
PerceptionTask
PlanningSingle
StepTask
Execution
. . .Fig. 1. Execution workflows of existing agent-based IR methods [30]–[37]. (a)
One-step decision methods follow a rigid sequential pipeline, which is prone
to error accumulation. (b) Multi-step decision methods introduce a reflection-
based iterative mechanism for evaluating and refining restoration results.
learning or instruction tuning. Unfortunately, these AiO IR
methods [20]–[29] struggle with a trade-off between gener-
alization and restoration accuracy due to potential conflicts
among various optimization objectives.
Recently, agent-based IR methods [30]–[37] have demon-
strated remarkable performance in complex scenarios. As
shown in Fig. 1, agent-based IR methods [30]–[37] can be
broadly categorized into one-step [33]–[37] and multi-step
decision [30]–[32] methods. One-step decision methods [33]–
[37] mimics human expertise to perceive quality degradation,
arrange a sequence of restoration tasks, and invoke IR tools.
However, any misjudgment at the quality assessment can cause
error accumulation throughout the entire execution chain.
Multi-step decision methods [30]–[32] employ an iterative
framework with a rollback mechanism, which evaluates the
restored output after each step to decide whether to execute
the next task or rollback the current operation for a retry. This
iterative framework provides better scalability and flexibility
in handling complex degradations. Nevertheless, the afore-
mentioned methods [30]–[37] force the agent into exhaustive
exploration of IR tools in the task execution, even when for
simple degradations, because they treat each restoration in-
stance in isolation and fail to leverage insights from historical
interactions. Furthermore, existing agent-based IR methods
[30]–[37] typically integrate multiple NR-IQA scores into an
overall quality score as optimization targets when improving
degradation perception capabilities [34]. Although the overall
quality score represents the visual quality of images to a
certain extent, a single score is insufficient to capture the
spatial distribution of degradations. As shown in Fig. 2, two
images possess a similar overall quality score, yet one exhibits
significant degradation in local regions while the other shows
slight hazing globally. Such ambiguity in quality assessment
often leads to erroneous decision-making, as the agent cannot
distinguish between localized severe degradations and globallyarXiv:2603.17055v1  [cs.CV]  17 Mar 2026

2
(a) Local Degradation (b) Global DegradationOverall Quality: 0.887 Overall Quality: 0.881
Fig. 2. The illustration of quality score ambiguity. Despite possessing similar
overall quality scores (calculated following [34]), these two images exhibit
distinct spatial degradation patterns.
minor ones.
In this paper, we propose PaAgent, a portrait-aware IR agent
designed to overcome the above limitations through two key
innovations: 1) a tool portrait bank that enables PaAgent to
choose the suitable IR tool by leveraging past interaction
insights. 2) a Subjective-Objective Reinforcement Learning
(SORL) strategy that enhances PaAgent’s degradation per-
ception capabilities through reward signals derived from both
image quality scores and semantic insights. Specifically, we
construct a tool portrait bank that stores extensive historical
interaction insights from various IR tools. The portrait bank
encompasses 14 restoration tools, 14 degradation scenarios,
and over 100,000 interaction insights. By retrieving relevant
insights via RAG [38], PaAgent preserves the scalability
and flexibility of multi-step decision methods while avoiding
excessive IR trials and rollbacks. In addition, each restora-
tion record generated by the PaAgent is dynamically stored
in the tool portrait bank, allowing PaAgent to improve its
restoration performance over prolonged usage. To improve
PaAgent’s degradation perception, we design a SORL strat-
egy comprising two core components: a Multimodal Large
Language Model (MLLM)-based reward generator and the
Group Relative Policy Optimization (GRPO) [39] algorithm.
The MLLM-based reward generator is designed to generate
fine-grained reward signals by synthesizing both MLLM’s
semantic insights and quality scores from No-Reference Image
Quality Assessment (NR-IQA) metrics [40]–[46]. The GRPO
[39] algorithm is employed to translate the fine-grained reward
signals into intra-group relative advantages, which allows
PaAgent to refine its policy (i.e., the next-task selection and
degradation information) by comparing multiple restoration
trajectories, ensuring that the policy updates are driven by a
nuanced understanding of both quality scores and semantic
insights.
In summary, our main contributions are as follows:
•We propose an efficient PaAgent method, which is the
first work to tackle image restoration using an agent with
insight summarization mechanism. Benefiting from RAG,
PaAgent achieves informed tool invocation by retrieving
historical interaction insights, effectively avoiding exces-
sive tool trials.
•We design a Subjective-Objective Reinforcement Learn-
ing (SORL) strategy to sharpen the degradation per-
ception capability of PaAgent. SORL enables PaAgent
not only to perceive the overall quality of images but
also to understand the semantic characteristics of local
degradation, thereby achieving precise decision-makingin complex scenarios.
•We conduct extensive experiments to demonstrate that the
proposed PaAgent method outperforms several leading
approaches in both subjective results and objective assess-
ments. Additionally, we show that PaAgent effectively
overcomes error accumulation and premature termination
across diverse degradation scenarios.
II. RELATEDWORK
A. All-in-One Image Restoration
To handle multiple distortions with a unified model weight,
numerous AiO IR models have been proposed. Inspired by
prompt learning in natural language processing, Potlapalliet
al.[20] presented a prompt-based learning approach for dy-
namically handling diverse image degradations. Qinet al.[22]
developed a mask image modeling framework that effectively
learns the inherent content of images while ensuring high-
quality reconstruction results. By leveraging the advantages
of ordinary differential equation solvers, Wanget al.[24]
introduced a diffusion generalist solver with universal poste-
rior sampling for image restoration. Rajagopalanet al.[25]
proposed a visual context learning to replace the prompt-
based learning in [20], thereby extracting degradation features
more effectively. Motivated by the observation that different
degradation types uniquely impact image content across dis-
tinct frequency subbands, Cuiet al.[29] proposed an AiO IR
network based on frequency mining and modulation. Although
these AiO IR methods achieve significant improvements by
unifying multiple degradation tasks within a single framework,
training these methods is inherently difficult due to the opti-
mization conflicts arising from diverse restoration objectives.
As a result, it often leads to a sub-optimal balance between
generalization ability and reconstruction performance.
B. Agent-based Image Restoration
With the rise of agents, IR tasks have ushered in a new
research line. As a pioneering work, Chenet al.[30] fine-
tuned an MLLM on synthetic datasets to enable autonomous
degradation assessment and tool execution. Another parallel
work by Zhuet al.[31] introduced a rollback-based agent
framework to select a suitable IR tool at each decision step
from a large tool repository. To balance execution efficiency
and restoration performance, Liet al.[32] proposed a hybrid
IR agent that adaptively invokes perception agents of varying
complexity based on user prompts. Jianget al.[33] proposed
a three-stage IR Agent based on the inherent physical order of
image degradation. Zhouet al.[34] proposed a greedy strategy
driven by NR-IQA metrics [34] to explore high-quality restora-
tion sequences. Linet al.[36] designed a training strategy
based on Supervised Fine-Tuning (SFT) and human feedback
alignment to enhance the agent’s robustness in autonomous
driving systems. Liuet al.[37] proposed a perturbation-driven
image quality optimization strategy to improve the agent’s
decision-making capabilities in terms of tool selection and
execution ordering. However, these agent-based IR methods
lack insight-driven tool invocation and powerful degradation
perception capabilities. In contrast, the proposed PaAgent

3
Fig. 3. Overview of the proposed PaAgent architecture. (a) illustrates the entire workflow of PaAgent, which leverages Qwen3.5-9B [47] for degradation
perception and task recommendation, followed by an RAG [38] module that queries the constructed tool portrait bank for optimal tool invocation. (b) depicts
the evolution of the tool portrait bank, where interaction insights are summarized by Qwen3.5-Plus [47] and stored for future utilization. (c) shows the SORL
strategy, which integrates MLLM’s insights and NR-IQA scores [40]–[46] via Qwen3.5-Plus [47] to generate reward signals, thereby driving the GRPO [39]
algorithm to fine-tune Qwen3.5-9B. [47]
is equipped with a tool portrait bank that stores extensive
historical interaction insights, enabling it to directly invoke the
appropriate IR tool. Moreover, our SORL strategy empowers
PaAgent with fine-grained degradation perception capabilities
by constructing an optimization objective jointly guided by
semantic insights and quality scores.
III. PROPOSEDMETHOD
In this section, we first present the overall architecture of
the proposed PaAgent, and we then detail the proposed tool
portrait bank and subjective-objective reinforcement learning
strategy.
A. Overall Architecture
The overall architecture of our proposed Portrait-aware
Agent (PaAgent) is shown in Fig. 3. Given a degraded image,
we first leverage the Qwen3.5-9B [47] model to conduct com-
prehensive perception, which additionally incorporates NR-
IQA scores [40]–[46] and previously executed tasks. This
perception stage yields degradation information and a se-
lected restoration task, both of which serve as queries for
the Retrieval-Augmented Generation (RAG) module [38]. As
illustrated in Fig. 4, these queries are first encoded via an
Query
Low-light EnhancementSelected Task
The image appears dark, ...Degradation Information
Vector Database
 Tool Portrait Bank
 Embedding Model
Similarity RetrievalChunksChunking
Embedding Model
Answer
CIDNetSelected Tool
Qwen3.5 -Plus
Related Chunks
Question :Please select the optimal
restoration tool based on the context.
Please answer the above questions
based on the { Query } and following
information:
Chunk 1 : ....; Chunk 2 : ....; Chunk
3: ....; ........ Chunk n : ....;Prompt TemplatesFig. 4. The illustration of the RAG [38] process. It mainly consists of two
phases: (1) offline knowledge base construction (upper stream), where the tool
portrait bank is chunked, embedded via an embedding model, and stored in a
vector database; and (2) online retrieval and reasoning (lower stream), where
the query is embedded to retrieve relevant chunks through similarity search.
The retrieved chunks are then integrated into prompt templates and fed into
the QWen3.5-Plus [47] to select the IR tool.
embedding model to perform similarity retrieval against the
vector database, which stores chunked interaction insights
from the tool portrait bank. The top-krelevant chunks retrieved

4
from the vector database are dynamically concatenated with
the query context into the prompt template. These enriched
prompts are subsequently fed into Qwen3.5-Plus [47] for
context-aware reasoning, enabling the inference of the most
appropriate IR tool. Ultimately, the selected IR tool is adopted
to obtain restored images.
B. Tool Portrait Bank
Agent-based IR methods have demonstrated impressive
capabilities in handling complex degradations; however, they
fail to leverage historical interaction insights, which leads to
excessive tool trials and rollback operations. To address this
limitation, we develop a tool portrait bank designed to store
past interaction insights that map diverse degradation scenarios
to their corresponding restoration solutions. As illustrated
in Fig. 3 (a), each entry in the portrait bank is organized
as a triplet comprising degradation information, the applied
restoration tool, and a subjective evaluation of the restoration
quality. To support efficient retrieval within the RAG [38]
framework, these structured triplets undergo a preprocessing
pipeline as depicted in Fig. 4. These interaction insights
are first segmented into semantic chunks via a chunking
strategy, which are subsequently encoded by an embedding
model into dense vector representations and organized within a
vector database. This vectorized organization enables semantic
similarity-based retrieval of relevant chunks given a degra-
dation query. Consequently, the search space for candidate
IR tools is effectively narrowed, allowing our PaAgent to
make informed decisions and avoid redundant explorations
during the restoration process. Furthermore, our constructed
tool portrait bank is equipped with an evolution mechanism to
ensure the continuous accumulation of interaction insights. As
shown in Fig. 3 (b), once a restoration task is performed, the
triple consisting of the degraded image, the restored image,
and the suggested tool is processed by the Qwen3.5-Plus [47]
model to summarize the interaction insight and provide a
subjective evaluation. The new insight is then integrated into
the tool portrait bank, facilitating a self-evolving knowledge
base that progressively enhances PaAgent’s proficiency in
handling complex restoration tasks.
C. Subjective-Objective Reinforcement Learning
Training agent-based IR methods that rely on numerical
scores makes it difficult to capture the subtle impact of dif-
ferent degradation patterns on image quality, thereby limiting
their ability to accurately describe degradation information
and make task-specific decisions. To this end, we employ the
GRPO [39] algorithm, a transformative advancement in RL
distinguished by its ability to refine model policies without a
complex value network, to enhance the accuracy of degrada-
tion perception. By utilizing the mean reward of a sampled
group as a baseline, the GRPO [39] algorithm effectively
identifies superior reasoning patterns and penalizes suboptimal
outputs, achieving a more efficient policy alignment process.
Specifically, we propose a Subjective-Objective Reinforcement
Learning (SORL) strategy, as shown in Fig. 3 (c). Given a
restored image, its subjective evaluation is first obtained byutilizing an MLLM (i.e., Qwen3.5-Plus [47]) to summarize the
triplet consisting of the degraded image, the selected IR tool,
and the restored image itself. Then, inspired by the [34], [37],
multiple NR-IQA metrics, including CLIP-IQA [40], Hyper-
IQA [41], NIQE [42], CPBD [43], BRISQUE [44], MUSIQ
[45], and LIQE [46], are used to compute its objective quality
scores and form an objective evaluation. These subjective and
objective evaluations are then integrated to constitute thei-th
outputO iwithin a sampled group. As illustrated in Fig. 3(c),
for each restoration task, PaAgent generates a group ofnsuch
outputs{O 1, O2, . . . , O n}. These outputs{O 1, O2, . . . , O n}
are fed into the reward generator (i.e., Qwen3.5-Plus [47]),
which performs dual-perspective analysis and assigns a binary
rewardr i∈ {0,1}to eachO i. A reward ofr i= 1indicates
that the selected restoration task effectively addressed the
identified degradation and produced a result consistent with
both subjective visual preferences and high objective quality
scores. Conversely, a reward ofr i= 0signifies that the chosen
restoration task either failed to resolve the image defect or
yielded a perceptually unsatisfactory outcome. Based on these
binary reward signals{r 1, r2, . . . , r n}, we first calculate the
normalized advantageA ifor each samplei:
Ai=ri−mean(r)
std(r) +ϵ,(1)
wheremean(·)denotes the mean operation,std(·)represents
the standard deviation, andϵis a small positive constant added
for numerical stability. Next, we leverage the normalized
advantages{A 1, A2, . . . , A n}within the GRPO [39] algorithm
to update the current policy modelπ θ(i.e., Qwen3.5-9B [47]),
whereθdenotes the trainable parameters of the model. The
core update mechanism follows a policy gradient scheme that
maximizes the expected advantage for actions taken by the
policy. Formally, we first define the probability ratio between
the current policyπ θand the old policyπ θoldas:
ρi(θ) =πθ(Ti|Ideg)
πθold(Ti|Ideg),(2)
whereπ θ(Ti|Ideg)denotes the probability of selecting
restoration taskT igiven a degraded input imageI deg. There-
after, the clipped surrogate lossL CLIP (θ)is formulated as:
LCLIP (θ) =1
nnX
i=1min [ρ i(θ)A i,clip(ρ i(θ),1−ϵ,1 +ϵ)A i],
(3)
where the clip(·)operation constrains the ratio to[1−ϵ,1+ϵ]to
prevent overly large policy steps, and themin(·)operator en-
sures pessimistic updates. Positive advantagesA ireinforce the
selection of taskT i, whereas negative advantages discourage
similar strategies. In practice, to prevent excessive updates and
maintain training stability, we adopt a clipped surrogate ob-
jective augmented with a KL-divergence regularization term.
Therefore, the final loss functionL final(θ)is obtained by
incorporating a KL-divergence penalty:
Lfinal(θ) =−L CLIP (θ) +β·D KL[πθ∥πref],(4)
where the hyperparameterβis the regularization coefficient
and the KL-divergence termD KL[πθ∥πref][39] penalizes

5
deviations from the reference policyπ ref(e.g., the initial pre-
trained model). To ensure computational stability, we employ
the following unbiased estimator [39]:
DKL[πθ∥πref] =πref(Ti|Ideg)
πθ(Ti|Ideg)−logπref(Ti|Ideg)
πθ(Ti|Ideg)−1,(5)
which guarantees non-negative values and exhibits lower vari-
ance than standard estimators. Consequently, the GRPO [39]
algorithm enables the policy modelπ θto learn effectively
from these intra-group advantages rather than absolute reward
scales, thereby enhancing training stability while maintaining
distributional consistency with the reference modelπ ref.
IV. EXPERIMENTS
A. Experiment Settings
Implementation Details.We adopt LoRA [48], a parameter-
efficient fine-tuning technique, to adapt the projection layers
within all self-attention modules of both the vision encoder
and the LLM. For our experimental setup, we configure the
LoRA rank to 16 and employ the SWIFT framework [49] to
facilitate the training pipeline. The PaAgent is trained for 10
epochs on 2 NVIDIA A100 GPUs with a batch size of 4, using
the Adam optimizer with a learning rate of10−6. For the IR
toolkit of PaAgent, we integrate a diverse collection of pre-
trained models: denoising (Restormer [8] and MPRNet [4]),
dehazing (OKNet [10], FocalNet [11], DehazeFormer [9], and
Dehamer [12]), deraining (Restormer [8] and MPRNet [4]),
deblurring (Restormer [8], MPRNet [4], and DRBNet [13]),
desnowing (InvDSNet [14], OKNet [10], and FocalNet [11]),
low-light enhancement (ZERO-DCE [16], ZERO-DCE++ [17],
LLFlow [18], and CIDNet [15]), and composite degradation
enhancement (LEDNet [50] and MoCE-IR [51]).
Compared Methods.We compare our PaAgent method
against eleven IR approaches, including ten AiO IR methods
(PromptIR [20], GridFormer [21], RAM [22], NDR-Restore
[23], DGSolver [24], AWRaCLe [25], DFPIR [26], DA-RCOT
[27], CPLIR [28], and AdaIR [29]), and one agent-based IR
method (AgenticIR [31]).
Benchmark Datasets.The training set consists of 15,013
images, which include 13,013 mixed degraded images from
the CDD-11 training subset [7], along with 2,000 motion
blurred images from the GoPro training subset [52]. The CDD-
11 [7] dataset contains 11 categories of image degradations
including single degradations (low-light, haze, rain, snow) and
their combinations (low+haze, low+rain, low+snow, haze+rain,
haze+snow, low+haze+rain, low+haze+snow). The test set has
16,230 image pairs, including 2,222 motion blurred image
pairs from the GoPro testing subset [52], 1,000 haze image
pairs from the SOTS [3] dataset, 408 noise image pairs from
the BSD68 [2] dataset, 4,300 rainy image pairs from the
Rain13K [4] dataset, 2,000 snow image pairs from the CSD
[1] dataset, 500 low-light image pairs from the LOL-V1 [5]
dataset, 3,600 low-light and motion blurred image pairs from
the LOL Blur [50] dataset, and 2,200 mixed degraded images
from the CDD-11 testing subset [7].
Evaluation Metrics.We adopt the peak signal-to-noise ratio
(PSNR) and the structural similarity index measure (SSIM)[53] to measure the performance of different methods. Higher
scores of PSNR and SSIM [53] denote that the enhanced
results are closer to the reference images in terms of image
content and structure. PSNR and SSIM [53] metrics are
produced on the Python platform.
B. Qualitative Evaluation
We conduct the visual comparisons of different methods on
various benchmark datasets [1]–[5], [7], [50], [52], as shown
in Figs. 5, 6, 7, and 8. Fig. 5 presents the results on desnowing
(CSD [1]) and denoising (BSD68 [2]) tasks. While DGSolver
[24] and AgenticIR [31] demonstrate robustness under dense
snowfall, they compromise the fidelity of fine textural details.
RAM [22] and NDR-Restore [23] achieve visually plausible
results yet suffer from residual noise and artifacts within com-
plex textural regions. By contrast, our PaAgent effectively sup-
presses snow streaks and noise across diverse scenarios while
preserving fine structural details, such as architectural textures,
animal fur, and foliage. As illustrated in Fig. 6, we evaluate
the restoration performance on dehazing (SOTS [3]) and
deraining (Rain13K [4]) tasks. Existing methods demonstrate
satisfactory capability in handling global atmospheric haze and
rain streaks, yet closer inspection exposes their struggles with
fine-detail preservation. Furthermore, RAM [22], DGSolver
[24] and CPLIR [28] exhibit impressive performance in the
deraining task but leave residual rain streaks in local regions
of the image. In contrast, our PaAgent not only recovers
sharper building boundaries in complex haze but also removes
dense rain streaks that competing methods fail to address. Fig.
7 shows the results on LLIE (LOL-V1 [5]) and deblurring
(GoPro [52]) tasks. RAM [22] DGSolver [24], DFPIR [26],
DA-RCOT [27], and AdaIR [29] demonstrate strong capability
in enhancing global visibility; however, they inevitably in-
troduce noticeable color casts in the stuffed toy regions or
fail to recover the local fine details of the drum. AWRaCLe
[25], AdaIR [29], and AgenticIR [31] successfully restore the
general shape of the petals and reduce motion blur, yet struggle
with fine texture reconstruction. In comparison, our proposed
PaAgent method efficiently addresses illumination deficiency
and motion blurring, yielding outputs with richer semantic
content and sharper structural details. As shown in Fig. 8, we
present the results under composite degradation scenarios (i.e.,
LOL Blur [50] and CDD-11 [7]). DGSolver [24] and DFPIR
[26] demonstrate competence in illumination adjustment, but
they fail to simultaneously eliminate motion blur. GridFormer
[21] and DGSolver [24] exhibit impressive performance in
handling snow+haze+low-light degradations, but they sacrifice
fine textural details such as tree barks. AgenticIR [31] excels
in recovering high-frequency information yet exhibits limited
capability in contrast adjustment. By intelligently coordinat-
ing specialized restoration experts, our PaAgent method can
restore results of both fine details and clear structures, which
suggests that our method is more robust in handling diverse
real-world degradations.
C. Quantitative Assessment
We quantify the performance of different IR methods using
PSNR and SSIM [53] metrics. As reported in Table I, our PaA-

6
(a) Input (b) PromptIR [20] (c) GridFormer [21] (d) RAM [22] (e) NDR-Restore [23] (f) DGSolver [24] (g) AWRaCLe [25]
(h) DFPIR [26] (i) DA-RCOT [27] (j) CPLIR [28] (k) AdaIR [29] (l) AgenticIR [31] (m) PaAgent (n) Reference
Fig. 5. Visual comparison of different methods on snow (CSD [1]) and noise (BSD68 [2]) images. Our PaAgent yields better visual results on different
degraded images, with outputs closer to the reference images.
TABLE I
QUANTITATIVE RESULTS OF DIFFERENT METHODS ON DESNOWING AND DENOISING DATASETS.σDENOTES THE NOISE LEVELTHE BEST RESULTS ARE
MARKED IN RED.
Desnowing Denoising(BSD68 [2])
CSD [1] σ=5 σ=10 σ=15 σ=25 σ=35 σ=50 Method Venue
PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM
PromptIR [20] NIPS’23 20.45 0.79 27.00 0.80 33.11 0.93 31.64 0.90 28.41 0.82 26.22 0.75 23.62 0.65
GridFormer [21] IJCV’24 15.94 0.79 24.64 0.69 28.82 0.88 27.00 0.81 23.85 0.67 21.68 0.56 19.18 0.44
RAM [22] ECCV’24 20.08 0.79 30.60 0.86 33.16 0.93 31.94 0.91 30.06 0.86 28.60 0.82 26.74 0.76
NDR-Restore [23] TIP’24 20.06 0.78 27.38 0.79 32.29 0.91 31.98 0.90 29.29 0.84 27.67 0.79 23.78 0.65
DGSolver [24] NIPS’25 32.69 0.96 25.47 0.65 29.62 0.85 26.82 0.76 22.91 0.59 20.29 0.47 17.73 0.36
AWRaCLe [25] AAAI’25 15.46 0.78 24.92 0.69 29.65 0.89 27.29 0.82 23.43 0.68 20.51 0.56 17.73 0.43
DFPIR [26] CVPR’25 13.03 0.69 21.19 0.67 26.75 0.88 24.95 0.82 23.83 0.70 21.53 0.59 17.85 0.44
DA-RCOT [27] TPAMI’25 20.21 0.78 18.46 0.66 19.11 0.79 31.00 0.88 30.21 0.86 27.03 0.76 21.98 0.56
CPLIR [28] TPAMI’25 15.56 0.73 24.03 0.74 29.30 0.88 26.47 0.83 21.55 0.74 19.11 0.67 17.02 0.58
AdaIR [29] ICLR’25 20.44 0.79 24.19 0.74 33.28 0.93 31.93 0.91 28.05 0.81 26.32 0.75 22.99 0.61
AgenticIR [31] ICLR’25 34.06 0.97 10.13 0.31 12.35 0.36 9.59 0.30 10.24 0.33 10.05 0.33 9.77 0.33
Our - 37.95 0.98 33.89 0.88 36.74 0.96 34.39 0.94 31.40 0.89 29.36 0.85 27.03 0.79
gent yields the highest scores in both desnowing and denoising
tasks, which reflects its superior capabilities in recovering
structural details and preserving color fidelity. Moreover, even
under extreme noise conditions (e.g.,σ= 50), the proposed
PaAgent still maintains a leading PSNR of 27.03 dB. As
summarized in Table II, compared with other competitors, ourPaAgent achieves optimal or suboptimal results in both dehaz-
ing and deraining tasks, demonstrating its superior capability
in effectively removing haze and rain streaks. From the results
in Table III, our method consistently achieves the highest
performance in both low-light image enhancement (LLIE) and
deblurring tasks, suggesting richer image details and better

7
(a) Input (b) PromptIR [20] (c) GridFormer [21] (d) RAM [22] (e) NDR-Restore [23] (f) DGSolver [24] (g) AWRaCLe [25]
(h) DFPIR [26] (i) DA-RCOT [27] (j) CPLIR [28] (k) AdaIR [29] (l) AgenticIR [31] (m) PaAgent (n) Reference
Fig. 6. Visual comparison of different methods on haze (SOTS [3]) and rainy (Rain13K [4]) images. Our PaAgent yields better visual results on different
degraded images, with outputs closer to the reference images.
TABLE II
QUANTITATIVE RESULTS OF DIFFERENT METHODS ON DEHAZING AND DERAINING DATASETS. THE BEST RESULTS ARE MARKED IN RED.
Dehazing(SOTS [3]) Deraining(Rain13K [4])
Indoor Outdoor Rain100L Rain100H Test100 Test1200 Test2800 Method Venue
PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM
PromptIR [20] NIPS’23 23.52 0.93 27.12 0.93 37.34 0.98 15.66 0.53 22.37 0.73 24.25 0.76 25.46 0.83
GridFormer [21] IJCV’24 11.96 0.70 17.30 0.84 25.40 0.85 12.22 0.39 21.80 0.69 23.12 0.71 23.87 0.77
RAM [22] ECCV’24 13.09 0.74 28.23 0.95 28.16 0.90 26.44 0.85 25.72 0.86 31.14 0.90 31.20 0.92
NDR-Restore [23] TIP’24 17.27 0.85 27.12 0.95 36.58 0.98 14.63 0.49 22.17 0.72 23.28 0.73 25.29 0.84
DGSolver [24] NIPS’25 36.41 0.98 31.78 0.95 36.02 0.97 26.98 0.85 22.69 0.79 32.15 0.92 29.87 0.89
AWRaCLe [25] AAAI’25 12.30 0.71 28.00 0.93 35.48 0.97 27.29 0.86 28.01 0.88 31.78 0.91 31.06 0.91
DFPIR [26] CVPR’25 10.56 0.66 12.68 0.71 24.09 0.84 11.06 0.38 19.17 0.65 19.69 0.68 19.92 0.74
DA-RCOT [27] TPAMI’25 19.20 0.87 27.45 0.93 37.92 0.98 14.85 0.50 21.68 0.68 23.55 0.71 24.70 0.79
CPLIR [28] TPAMI’25 12.06 0.69 16.62 0.77 30.32 0.92 17.71 0.60 22.63 0.80 26.78 0.84 27.69 0.87
AdaIR [29] ICLR’25 18.99 0.88 26.80 0.93 37.88 0.98 15.59 0.53 22.40 0.73 24.03 0.74 25.54 0.83
AgenticIR [31] ICLR’25 33.46 0.99 20.43 0.86 35.04 0.96 28.63 0.88 28.66 0.88 30.47 0.89 31.45 0.92
Our - 40.65 0.99 37.01 0.99 37.64 0.98 29.57 0.90 30.55 0.91 31.74 0.91 32.37 0.93
visibility of our results. Furthermore, in the more challenging
LLIE+deblurring task, our method maintains leading perfor-
mance, which demonstrates the effectiveness of our PaAgent
in addressing composite degradations. To further evaluate the
robustness of our PaAgent in handling complex degradation
scenes, we conduct experiments on the CDD-11 dataset. Asshown in Table IV, our PaAgent consistently outperforms
all competitors across all 11 evaluation scenarios, including
single, double, and triple degradations. This significant per-
formance gap proves its generalization capability in complex
degradations. The comprehensive quantitative results across
the aforementioned diverse degradation tasks demonstrate that

8
(a) Input (b) PromptIR [20] (c) GridFormer [21] (d) RAM [22] (e) NDR-Restore [23] (f) DGSolver [24] (g) AWRaCLe [25]
(h) DFPIR [26] (i) DA-RCOT [27] (j) CPLIR [28] (k) AdaIR [29] (l) AgenticIR [31] (m) PaAgent (n) Reference
Fig. 7. Visual comparison of different methods on low-light (LOL-V1 [5]) and blur (GoPro [52]) images. Our PaAgent yields better visual results on different
degraded images, with outputs closer to the reference images.
TABLE III
QUANTITATIVE RESULTS OF DIFFERENT METHODS ON LOW-LIGHT IMAGE ENHANCEMENT(LLIE)AND DEBLURRING DATASETS. BLUR GAMMA
DENOTES BLURRED IMAGES WITH GAMMA CORRECTION. THE BEST RESULTS ARE MARKED IN RED.
LLIE Deblurring(GoPro [52]) LLIE+Deblurring(LOL Blur [50])
LOL-V1 [5] Blur gamma Blur Low-light Low-light+Blur Method Venue
PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM
PromptIR [20] NIPS’23 7.66 0.18 22.14 0.75 22.15 0.75 8.18 0.18 8.15 0.16
GridFormer [21] IJCV’24 7.73 0.18 24.89 0.78 25.14 0.78 8.75 0.27 8.77 0.25
RAM [22] ECCV’24 26.24 0.90 27.08 0.86 27.63 0.86 9.10 0.29 8.79 0.26
NDR-Restore [23] TIP’24 7.61 0.17 23.72 0.77 23.84 0.77 8.48 0.23 8.45 0.21
DGSolver [24] NIPS’25 24.89 0.91 27.16 0.87 27.91 0.87 17.38 0.74 14.67 0.59
AWRaCLe [25] AAAI’25 7.55 0.16 23.72 0.77 23.85 0.78 8.57 0.24 8.60 0.22
DFPIR [26] CVPR’25 9.96 0.36 21.95 0.76 21.71 0.75 12.46 0.50 12.53 0.47
DA-RCOT [27] TPAMI’25 29.61 0.91 27.37 0.87 28.03 0.88 12.57 0.51 10.44 0.35
CPLIR [28] TPAMI’25 7.79 0.19 24.30 0.78 24.52 0.78 8.73 0.26 8.73 0.24
AdaIR [29] ICLR’25 29.33 0.91 27.01 0.86 27.47 0.86 13.31 0.54 10.61 0.36
AgenticIR [31] ICLR’25 9.32 0.39 29.11 0.92 30.59 0.93 11.02 0.50 10.93 0.45
Our - 31.95 0.93 29.97 0.94 31.81 0.94 21.44 0.88 20.82 0.83
our PaAgent offers a powerful solution for IR.
D. Ablation Study
We conduct an ablation study on the tool portrait bank, the
subjective-objective reinforcement learning strategy, and theinput context to show the effects of each key component. The
following ablation experiments are presented:
•w/ RS:Randomly select one from the IR tool candidates.
•w/ MS:Manually select one from the IR tool candidates
by human experts.

9
(a) Input (b) PromptIR [20] (c) GridFormer [21] (d) RAM [22] (e) NDR-Restore [23] (f) DGSolver [24] (g) AWRaCLe [25]
(h) DFPIR [26] (i) DA-RCOT [27] (j) CPLIR [28] (k) AdaIR [29] (l) AgenticIR [31] (m) PaAgent (n) Reference
Fig. 8. Visual comparison of different methods on composite degradation (LOL Blur [50] and CDD-11 [7]) images. Our PaAgent yields better visual results
on different degraded images, with outputs closer to the reference images.
TABLE IV
QUANTITATIVE RESULTS OF DIFFERENT METHODS ON THECDD-11 [7]DATASET. THE BEST RESULTS ARE MARKED IN RED.
Single Degradation Double Degradation Triple Degradation
Haze (H) Rain (R) Snow (S) Low-light (L) H+R H+S L+H L+R L+S L+H+R L+H+S Method
PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM
PromptIR [20] 17.52 0.76 18.76 0.74 17.95 0.73 14.49 0.63 17.43 0.72 16.83 0.73 15.27 0.65 15.12 0.61 13.80 0.57 15.67 0.64 14.96 0.62
GridFormer [21] 17.32 0.78 18.92 0.77 18.83 0.77 14.62 0.64 18.21 0.79 16.79 0.76 15.67 0.68 14.85 0.60 13.34 0.60 15.95 0.66 15.51 0.67
RAM [22] 15.73 0.72 18.49 0.76 16.90 0.69 14.41 0.64 15.89 0.72 15.15 0.68 15.25 0.65 14.54 0.66 13.87 0.57 15.66 0.65 14.89 0.61
NDR-Restore [23] 17.02 0.74 18.40 0.73 17.53 0.72 14.43 0.62 17.19 0.72 16.17 0.72 15.16 0.63 15.12 0.61 13.78 0.57 15.58 0.63 14.85 0.60
DGSolver [24] 16.70 0.75 17.98 0.74 18.21 0.74 13.86 0.64 16.14 0.72 16.75 0.74 13.84 0.65 13.81 0.62 12.79 0.56 14.83 0.65 13.68 0.63
AWRaCLe [25] 15.89 0.74 18.19 0.75 18.95 0.74 13.50 0.62 15.81 0.73 15.59 0.72 14.35 0.65 13.46 0.59 12.46 0.58 15.59 0.66 15.13 0.65
DFPIR [26] 14.85 0.69 16.47 0.69 15.62 0.68 14.93 0.62 15.25 0.69 14.25 0.66 14.98 0.64 15.55 0.60 14.20 0.55 15.19 0.63 14.82 0.61
DA-RCOT [27] 15.10 0.72 17.40 0.73 17.00 0.72 13.33 0.64 15.05 0.70 14.65 0.70 11.36 0.60 15.70 0.66 15.15 0.62 11.56 0.60 11.49 0.59
CPLIR [28] 16.02 0.71 18.58 0.77 17.84 0.71 15.27 0.66 16.31 0.72 15.51 0.67 15.70 0.65 15.33 0.70 14.76 0.62 16.09 0.67 15.28 0.61
AdaIR [29] 17.10 0.75 18.40 0.73 17.72 0.72 14.49 0.63 16.91 0.71 16.50 0.72 14.88 0.64 15.25 0.62 14.02 0.57 15.34 0.64 14.58 0.61
AgenticIR [31] 11.99 0.36 12.68 0.31 12.31 0.37 10.62 0.29 12.00 0.29 11.94 0.37 10.60 0.28 11.16 0.28 10.68 0.28 11.07 0.25 10.37 0.28
Our 28.37 0.92 29.57 0.92 29.39 0.91 27.24 0.88 28.10 0.92 27.83 0.91 26.46 0.87 27.70 0.88 27.02 0.86 26.46 0.87 25.95 0.86
•w/ SFT: Apply the SFT strategy to improve PaAgent’s
degradation perception capability.
•w/ SE: Apply the GRPO [39] algorithm driven by
MLLMs’ subjective evaluation to improve PaAgent’s
degradation perception capability.
•w/o NE: Exclude NR-IQA scores [40]–[46] and executed
tasks from PaAgent’s input context.•w/o ET: Exclude executed tasks from PaAgent’s input
context.
•w/ OSD: Determine the image restoration sequence
through the one-step decision.
•full model: Using our full PaAgent.
1) Effects of the Tool Portrait bank: The qualitative and
quantitative results are shown in Fig. 9 and Table V, respec-

10
(a) Input (b)w/ RS(c)w/ MS(d)w/ SFT(e)w/ SE(f)full model(g) Reference
Fig. 9. Visualization results of ablation study on the tool portrait bank and the subjective-objective reinforcement learning strategy. In the zoomed-in areas,
ourfull modelyields better visual results on different degraded images, with outputs closer to the reference images.
" T h ou gh t":"Theimageshowsasignificantamountof
grainytextureacrosstheentireframe,whichis
particularly noticeable intheuniformareaslikethe
buildingwalls.Althoughtherearevisibleweather
effects,theunderlying noiseistheprimaryfactor
reducingimagequality.Therefore, i m age d e n oi s i n g
shouldbeperformedtocleanupthegrain."
w/oNE
" T h ou gh t":"Theimageshowsanarchitectural
buildingwithsevererainstreakartifactsvisibleacross
theentireimage.ThelowCLIP-IQAscore(0.297)and
thehighNIQEscore(9.506)confirmsignificant
deviationfromnaturalimagestatistics,bothconsistent
withheavyraindegradation .Themostappropriate
restorationtaskis i m age d e r ai n i n g."
full model
" T h ou gh t":"Theimagedisplayssignificant blur
degradation withseverely reducedsharpness
throughout thescene.TheextremelyhighNIQEscore
of9.506indicatesseveredeviationfromnaturalimage
statistics,whiletheverylowCLIP-IQAof0.297
confirmspoorperceptualquality. I m a ge d e b l u r r i n g is
necessarytorecoverthearchitectural details."
w/oET
" T h ou gh t":"Basedonthevisualcharacteristics ofthe
image,thesceneappearstohavelowbrightnessand
poorvisibility.Theoverallcontrastseemsreduced,and
theimagequalitycouldbenefitfrombrightness
adjustment.Therefore, l o w - l i gh t i m age e n h an c e m e n t
shouldbethenextrestoration tasktoimprovethe
visibilityandbrightness."" T h ou gh t":"Aftercompleting theImageRainingtask,
theimagestillexhibitssignificant atmospheric haze
effects.TheCLIP-IQAscore(0.390)remainsrelatively
lowandNIQE(7.652)isstillhigh,indicating
substantial roomforqualityimprovement . I m a ge
d e h az i n gistheappropriate nexttasktoremovethe
hazeandimprovetheoverallvisibility."" T h ou gh t":"Theimagecurrentlyexhibitsastrong
grayishcastandsignificantly reducedcontrast,
characteristic ofheavyhazeorfog.TheNIQEscore
hasincreaseddrasticallyto7.652,whichindicatesthat
theimagestatisticsarefarfromnatural.Therefore,
i m age d e h az i n gisthenextsteptorestorecontrast,and
recoverthetruecolorsanddetailsofthebuilding."
" T h ou gh t":"Theimagedepictsanarchitectural
structureunderanovercast,resultinginascenethatis
overalldarkandhaslowcontrast.Theprimaryvisual
issueisthelackofsufficientillumination .Therefore,
applying l o w - l i gh t i m age e n h an c e m e n t wouldbethe
mostappropriate tasktobrightenthescene,improve
contrast,andrevealmoredetailsintheshadows."" T h ou gh t":"Althoughderaininganddehazingreduced
theNIQEto4.567,theCLIP-IQAscoreof0.415
indicatesremainingdeficiencies inperceptualquality.
Visualinspectionrevealstheimageremainsdarkand
low-contrastdespiteweatherremoval.Therefore, l o w -
l i gh t i m age e n h an c e m e n t isselectedtoimprove
brightness, contrast,andcolorvividness."" T h ou gh t":"Theimagedepictsalargearchitectural
structureundergloomyweather,resultinginascene
thatisvisiblydim.ThehighNIQEscore(4.567)and
lowCLIP-IQAscore(0.415)furtherindicatepoor
perceptual quality,likelystemming fromthedark
lighting.Therefore, applying l o w - l i gh t i m age
e n h an c e m e n t isthemostappropriate nextstep."I n p u tL o w - l i gh t + H az e + R ai n L o w - l i gh t + H az e L o w - l i gh t
Fig. 10. Ablation study on the input context of our proposed PaAgent. Incorrect and correct restoration task suggestions are marked inpurpleandred,
respectively. Our full model consistently generates accurate restoration strategies across various degradation scenarios.
TABLE V
THE AVERAGEPSNRANDSSIMSCORES OF ABLATION STUDY OF THE
TOOL PORTRAIT BANK AND THE SUBJECTIVE-OBJECTIVE
REINFORCEMENT LEARNING STRATEGY ON THECDD-11 [7]DATASET.
THE BEST RESULTS ARE MARKED IN RED.
Double Degradation Triple Degradation
H+S L+S L+H+R L+H+S Method
PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM
w/ RS 19.33 0.71 18.27 0.70 17.57 0.68 17.39 0.65
w/ MS 23.36 0.74 22.94 0.72 21.58 0.70 20.09 0.68
w/ SFT 25.31 0.86 24.84 0.78 24.10 0.75 23.59 0.71
w/ SE 26.68 0.88 26.01 0.82 25.53 0.81 24.21 0.79
full model 27.83 0.91 27.02 0.86 26.46 0.87 25.95 0.86
tively.w/ RSproduces the poorest performance, as random
tool selection fails to address specific degradations, leaving
heavy noise and artifacts, as shown in Fig.9 (b). In Fig.9
(c),w/ MSimproves restoration performance, but it lacks
the flexibility to handle complex degradations. In contrast,the results of ourfull modelare closer to the reference
images in Fig. 9 (f). Moreover, ourfull modelpossesses
the best PSNR and SSIM scores in Table V. This indicates
that the insight-driven tool selection effectively narrows the
search space to optimal restoration strategies, avoiding both
the blindness of random selection and the rigidity of fixed
matching schemes, thereby substantially improving restoration
quality across diverse degradation scenarios.
2) Effects of the SORL strategy: As shown in Table V,w/
SFTyields significant gains in PSNR and SSIM compared
tow/ RSandw/ MS. This demonstrates that the tool portrait
bank requires accurate degradation perception to effectively
deploy the appropriate IR tools for specific restoration tasks.
As shown in Fig. 9 (e),w/ SErecovers finer textures and
more natural illumination thanw/ SFT, demonstrating that the
subjective insights of MLLMs are crucial for improving PaA-

11
"Thought":"Theinputimagesuffersfromsevereunderexposure
duetoextremelylowambientillumination, resultinginpredominant
darkregionswherecriticalstructuraldetailsarenearly
indistinguishable .Fromtheseobservations, thereisonetypeof
degradation (severelowillumination) inthisimageandtheimage
restorationisperformedinonestep.Consequently, theorderof
imagerestorationissetto[low-light image enhancement ]."
w/ OSD
Low -light
Step1: LLIE Zoom -In
 Step1: LLIE Step2: Motion Deblurring Zoom -In
full model
"Thought":"Theinputimage
suffersfromunderexposure due
toextremely lowambient
illumination, ......Consequently,
applyingthe low-light image
enhancement isessentialto
recoverhiddendetails,......"
full model
"Thought":"Althoughthelow-
lightimageenhancement
operationisattempted, the
outcomestillsuffersfromsevere
motionblur......Therefore,an
motion deblurring processis
imperative......"
full model
"Thought":"Thecurrentimage
exhibitssharp,well-defined
edgesonarchitectural elements
suchasstaircasesandpillars......
Consequently, theterminating
restoration isanappropriate
decision."
"Thought":"Theinputimageexhibitsdensesnowfall,characterized
bynumerousbright,randomlydistributedsnowflakesthatheavily
occludethebackground scene,includingwatersurfacesand
vegetationregions.Fromtheseobservations, thereisonetypeof
degradation (densesnow)inthisimageandtheimagerestorationis
performedinonestep.Consequently, theorderofimagerestoration
issetto[image desnowing]."
w/ OSD
Snow
Step1: Desnowing Zoom -In
 Step1: Desnowing Step2: Dehazing Zoom -In
full model
"Thought":"Theinputimage
exhibitsseveredegradation
causedbydensesnowfall,......
Consequently, applying an
image desnowing processis
essentialtoeliminatethese
obstructivesnowparticles,......"
full model
"Thought":"Although the
imagedesnowingoperationis
attempted,thescenestillsuffers
fromatmospheric haze......
Therefore,animage dehazing
processisimperativetofurther
remove......"
full model
"Thought":"Theimagenow
possessesadequateclarityand
visualfidelitysuitablefor
downstream computervision
tasks......Consequently, the
terminating restoration isan
appropriatedecision."
Fig. 11. Ablation study on the multi-step decision mechanism of our PaAgent. Our full model dynamically evaluates intermediate results and triggers additional
restoration steps when necessary, thereby avoiding premature termination caused by the fixed restoration sequence.
gent’s degradation perception capability. By contrast, ourfull
modeleliminates the subtle color biases and over-smoothing
issues sometimes found inw/ SE, and produces results that are
visually pleasing and structurally accurate. This indicates that
the SORL strategy can further enhance PaAgent’s degradation
perception capability.
3) Effects of the input context: As shown in Fig. 10, the
agent’s restoration path varies significantly depending on the
richness of the input context. Specifically, when process-
ing images with rain streaks,w/o NEconfuses them with
noise and erroneously selects image denoising; similarly, it
mistakes haze degradation for low-light conditions, resulting
in fundamentally incorrect restoration strategies. Whilew/o
ETenables correct distinction between atmospheric haze and
noise, the absence of executed task history still causes er-
roneous sequencing in multi-step restoration pipelines, such
as attempting to deblur before deraining. In contrast, our
full model, equipped with a comprehensive input context,
accurately perceives degradation types and determines the
optimal restoration task. This indicates that NR-IQA scores
provide essential quantitative guidance for degradation percep-
tion, while executed task history provides sequential guidance
for multi-step restoration, where both components are crucial
for robust restoration planning.
4) Effects of multi-step decision mechanism: As shown in
Fig. 11,w/ OSDdemonstrates the capability to generate a
complete restoration sequence in a single forward pass, which
significantly improves computational efficiency. However,w/
OSDfollows a predetermined sequence without adapting to
the actual restoration outcomes. When the enhanced imagestill exhibits motion blur or when the low-light enhancement
produces suboptimal results,w/ OSDproceeds to execute the
next operation or terminates the restoration process, leading
to incomplete restoration. By contrast, ourfull modelemploys
the multi-step decision mechanism that dynamically evaluates
the intermediate results after each restoration operation. As
illustrated in Fig. 11, after completing image desnowing,
ourfull modeldetects the residual atmospheric haze and
determines that an additional image dehazing step is impera-
tive. This progressive perceive-restore-evaluate cycle process
continues until the image achieves adequate visual quality, as
evidenced by the final terminating restoration decision. These
observations confirm that the multi-step decision mechanism
is crucial for handling complex degradations and ensuring
thorough restoration.
V. CONCLUSION
In this paper, we present PaAgent, a portrait-aware IR
agent designed to boost the paradigm of intelligent IR. The
cornerstone of our approach is a RAG-powered framework
that enables precise IR tool invocation by leveraging past in-
teraction insights. Moreover, we propose a subjective-objective
reinforcement learning strategy based on the GRPO algo-
rithm to refine PaAgent’s degradation perception. This strategy
aligns policy updates with both NR-IQA scores and human
visual preferences, successfully achieving a fine-grained per-
ception of degradation characteristics. Extensive experiments
across eight benchmark datasets demonstrate that our PaAgent
achieves superior performance across diverse degradation sce-
narios, underscoring its robustness in complex degradations.

12
REFERENCES
[1] W. Chen, H. Fang, C. Hsieh, C. Tsai, I. Chen, J. Ding, and S. Kuo, “All
snow removed: Single image desnowing algorithm using hierarchical
dual-tree complex wavelet representation and contradict channel loss,”
inProc. IEEE Int. Conf. Comput. Vis. (ICCV), pp. 4196–4205, 2021.
[2] D. Martin, C. Fowlkes, D. Tal, and J. Malik, “A database of human
segmented natural images and its application to evaluating segmentation
algorithms and measuring ecological statistics,” inProc. IEEE Int. Conf.
Comput. Vis. (ICCV), vol. 2, pp. 416–423, 2001.
[3] B. Li, W. Ren, D. Fu, D. Tao, D. Feng, W. Zeng, and Z. Wang,
“Benchmarking single-image dehazing and beyond,”IEEE Trans. Image
Process., vol. 28, no. 1, pp. 492–505, 2019.
[4] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M. Yang, and
L. Shao, “Multi-stage progressive image restoration,” inProc. IEEE
Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 14821–14831, 2021.
[5] C. Wei, W. Wang, W. Yang, and J. Liu, “Deep retinex decomposition for
low-light enhancement,” inProc. Brit. Mach. Vis. Conf. (BMVC), 2018.
[6] W. Zhao, G. Hu, F. Wei, H. Wang, Y . He, and H. Lu, “Attacking defocus
detection with blur-aware transformation for defocus deblurring,”IEEE
Trans. Multimedia, vol. 26, pp. 5450–5460, 2024.
[7] Y . Guo, Y . Gao, Y . Lu, H. Zhu, R. W. Liu, and S. He, “OneRestore:
A universal restoration framework for composite degradation,” inProc.
Eur. Conf. Comput. Vis. (ECCV), pp. 255–272, 2024.
[8] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M. Yang,
“Restormer: Efficient transformer for high-resolution image restoration,”
inProc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 5728–
5739, 2022.
[9] Y . Song, Z. He, H. Qian, and X. Du, “Vision transformers for single
image dehazing,”IEEE Trans. Image Process., vol. 32, pp. 1927–1941,
2023.
[10] Y . Cui, W. Ren, and A. Knoll, “Omni-kernel network for image
restoration,” inProc. AAAI Conf. Artif. Intell. (AAAI), vol. 38, pp. 1426–
1434, 2024.
[11] Y . Cui, W. Ren, X. Cao, and A. Knoll, “Focal network for image
restoration,” inProc. IEEE Int. Conf. Comput. Vis. (ICCV), pp. 12955–
12965, 2023.
[12] C. Guo, Q. Yan, S. Anwar, R. Cong, W. Ren, and C. Li, “Image dehazing
transformer with transmission-aware 3d position embedding,” inProc.
IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 5802–5810,
2022.
[13] L. Ruan, B. Chen, J. Li, and M. Lam, “Learning to deblur using light
field generated and real defocus images,” inProc. IEEE Conf. Comput.
Vis. Pattern Recognit. (CVPR), pp. 16304–16313, 2022.
[14] Y . Quan, X. Tan, Y . Huang, Y . Xu, and H. Ji, “Image desnowing via
deep invertible separation,”IEEE Trans. Circuits Syst. Video Technol.,
vol. 33, no. 7, pp. 3133–3144, 2023.
[15] Q. Yan, Y . Feng, C. Zhang, G. Pang, K. Shi, P. Wu, W. Dong, J. Sun, and
Y . Zhang, “HVI: A new color space for low-light image enhancement,”
inProc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 5678–
5687, 2025.
[16] C. Guo, C. Li, J. Guo, C. C. Loy, J. Hou, S. Kwong, and R. Cong, “Zero-
reference deep curve estimation for low-light image enhancement,” in
Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 1777–
1786, 2020.
[17] C. Li, C. Guo, and C. C. Loy, “Learning to enhance low-light image via
zero-reference deep curve estimation,”IEEE Trans. Pattern Anal. Mach.
Intell., vol. 44, no. 8, pp. 4225–4238, 2022.
[18] Y . Wang, R. Wan, W. Yang, H. Li, L. Chau, and A. Kot, “Low-light
image enhancement with normalizing flow,” inProc. AAAI Conf. Artif.
Intell. (AAAI), vol. 36, pp. 2604–2612, 2022.
[19] W. Zhao, H. Cui, H. Wang, Y . He, and H. Lu, “FreeFusion: Infrared
and visible image fusion via cross reconstruction learning,”IEEE Trans.
Pattern Anal. Mach. Intell., vol. 47, no. 9, pp. 8040–8056, 2025.
[20] V . Potlapalli, S. W. Zamir, S. H. Khan, and F. Shahbaz Khan, “PromptIR:
Prompting for all-in-one image restoration,” inProc. Adv. Neural Inf.
Process. Syst. (NeurIPS), vol. 36, pp. 71275–71293, 2023.
[21] T. Wang, K. Zhang, Z. Shao, W. Luo, B. Stenger, T. Lu, T. Kim, W. Liu,
and H. Li, “GridFormer: Residual dense transformer with grid structure
for image restoration in adverse weather conditions,”Int. J. Comput.
Vis., vol. 132, no. 10, pp. 4541–4563, 2024.
[22] C. Qin, R. Wu, Z. Liu, X. Lin, C. Guo, H. H. Park, and C. Li, “Restore
Anything with Masks: Leveraging mask image modeling for blind all-
in-one image restoration,” inProc. Eur. Conf. Comput. Vis. (ECCV),
pp. 364–380, 2024.[23] M. Yao, R. Xu, Y . Guan, J. Huang, and Z. Xiong, “Neural degradation
representation learning for all-in-one image restoration,”IEEE Trans.
Image Process., vol. 33, pp. 5408–5423, 2024.
[24] H. Wang, J. Zhang, H. Guo, D. Wang, J. Ma, and B. Du, “DGSolver:
Diffusion generalist solver with universal posterior sampling for image
restoration,” inProc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2025.
[25] S. Rajagopalan and V . M. Patel, “AWRaCLe: All-weather image restora-
tion using visual in-context learning,” inProc. AAAI Conf. Artif. Intell.
(AAAI), vol. 39, pp. 6675–6683, 2025.
[26] X. Tian, X. Liao, X. Liu, M. Li, and C. Ren, “Degradation-aware
feature perturbation for all-in-one image restoration,” inProc. IEEE
Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 28165–28175, 2025.
[27] X. Tang, X. Gu, X. He, X. Hu, and J. Sun, “Degradation-aware residual-
conditioned optimal transport for unified image restoration,”IEEE Trans.
Pattern Anal. Mach. Intell., vol. 47, no. 8, pp. 6764–6779, 2025.
[28] G. Wu, J. Jiang, K. Jiang, X. Liu, and L. Nie, “Beyond Degradation Re-
dundancy: Contrastive prompt learning for all-in-one image restoration,”
IEEE Trans. Pattern Anal. Mach. Intell., pp. 1–18, 2025.
[29] Y . Cui, S. W. Zamir, S. Khan, A. Knoll, M. Shah, and F. S. Khan,
“AdaIR: Adaptive all-in-one image restoration via frequency mining and
modulation,” inProc. Int. Conf. Learn. Represent. (ICLR), pp. 57335–
57356, 2025.
[30] H. Chen, W. Li, J. Gu, J. Ren, S. Chen, T. Ye, R. Pei, K. Zhou, F. Song,
and L. Zhu, “RestoreAgent: Autonomous image restoration agent via
multimodal large language models,” inProc. Adv. Neural Inf. Process.
Syst. (NeurIPS), vol. 37, pp. 110643–110666, 2024.
[31] K. Zhu, J. Gu, Z. You, Y . Qiao, and C. Dong, “An intelligent agentic
system for complex image restoration problems,” inProc. Int. Conf.
Learn. Represent. (ICLR), vol. 2025, pp. 57985–58013, 2025.
[32] B. Li, X. Li, Y . Lu, and Z. Chen, “Hybrid agents for image restoration,”
inProc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2026.
[33] X. Jiang, G. Li, B. Chen, and J. Zhang, “Multi-agent image restoration,”
arXiv preprint arXiv:2503.09403, 2025.
[34] Y . Zhou, J. Cao, Z. Zhang, F. Wen, Y . Jiang, J. Jia, X. Liu, X. Min, and
G. Zhai, “Q-Agent: Quality-driven chain-of-thought image restoration
agent through robust multimodal large language model,”arXiv preprint
arXiv:2504.07148, 2025.
[35] Y . Zuo, Q. Zheng, M. Wu, X. Jiang, R. Li, J. Wang, Y . Zhang, G. Mai,
L. Wang, J. Zou, X. Wang, M. Yang, and Z. Tu, “4KAgent: Agentic any
image to 4k super-resolution,” inProc. Adv. Neural Inf. Process. Syst.
(NeurIPS), 2025.
[36] Y . Lin, Z. Lin, H. Chen, P. Pan, C. Li, S. Chen, K. Wen, Y . Jin, W. Li,
and X. Ding, “JarvisIR: Elevating autonomous driving perception with
intelligent image restoration,” inProc. IEEE Conf. Comput. Vis. Pattern
Recognit. (CVPR), pp. 22369–22380, 2025.
[37] F. Liu, J. Xu, and X. Hu, “Real-world adverse weather image restoration
via dual-level reinforcement learning with high-quality cold start,” in
Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2025.
[38] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W. Yih, T. Rockt ¨aschel,et al., “Retrieval-
augmented generation for knowledge-intensive NLP tasks,” inProc. Adv.
Neural Inf. Process. Syst. (NeurIPS), vol. 33, pp. 9459–9474, 2020.
[39] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang,
Y . Li, Y . Wu,et al., “DeepSeekMath: Pushing the limits of mathematical
reasoning in open language models,”arXiv preprint arXiv:2402.03300,
2024.
[40] J. Wang, K. C. Chan, and C. C. Loy, “Exploring clip for assessing
the look and feel of images,” inProc. AAAI Conf. Artif. Intell. (AAAI),
vol. 37, pp. 2555–2563, 2023.
[41] S. Su, Q. Yan, Y . Zhu, C. Zhang, X. Ge, J. Sun, and Y . Zhang,
“Blindly assess image quality in the wild guided by a self-adaptive hyper
network,” inProc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR),
pp. 3664–3673, 2020.
[42] A. Mittal, R. Soundararajan, and A. C. Bovik, “Making a “completely
blind” image quality analyzer,”IEEE Signal Process. Lett., vol. 20, no. 3,
pp. 209–212, 2013.
[43] N. D. Narvekar and L. J. Karam, “A no-reference image blur metric
based on the cumulative probability of blur detection (CPBD),”IEEE
Trans. Image Process., vol. 20, no. 9, pp. 2678–2683, 2011.
[44] A. Mittal, A. K. Moorthy, and A. C. Bovik, “No-reference image quality
assessment in the spatial domain,”IEEE Trans. Image Process., vol. 21,
no. 12, pp. 4695–4708, 2012.
[45] J. Ke, Q. Wang, Y . Wang, P. Milanfar, and F. Yang, “MUSIQ: Multi-
scale image quality transformer,” inProc. IEEE Int. Conf. Comput. Vis.
(ICCV), pp. 5128–5137, 2021.

13
[46] W. Zhang, G. Zhai, Y . Wei, X. Yang, and K. Ma, “Blind image quality
assessment via vision-language correspondence: A multitask learning
perspective,” inProc. IEEE Conf. Comput. Vis. Pattern Recognit.
(CVPR), pp. 14071–14081, 2023.
[47] Qwen Team, “Qwen3.5: Towards native multimodal agents,” February
2026.
[48] E. J. Hu, Y . Shen, P. Wallis, Z. Allen-Zhu, Y . Li, S. Wang, L. Wang,
W. Chen,et al., “LoRA: Low-rank adaptation of large language models,”
inProc. Int. Conf. Learn. Represent. (ICLR), 2022.
[49] Y . Zhao, J. Huang, J. Hu, X. Wang, Y . Mao, D. Zhang, Z. Jiang, Z. Wu,
B. Ai, A. Wang, W. Zhou, and Y . Chen, “Swift: A scalable lightweight
infrastructure for fine-tuning,” inProc. AAAI Conf. Artif. Intell. (AAAI),
vol. 39, pp. 29733–29735, 2025.
[50] S. Zhou, C. Li, and C. C. Loy, “LEDNet: Joint low-light enhancement
and deblurring in the dark,” inProc. Eur. Conf. Comput. Vis. (ECCV),
pp. 573–589, 2022.
[51] E. Zamfir, Z. Wu, N. Mehta, Y . Tan, D. P. Paudel, Y . Zhang, and
R. Timofte, “Complexity experts are task-discriminative learners for any
image restoration,” inProc. IEEE Conf. Comput. Vis. Pattern Recognit.
(CVPR), pp. 12753–12763, 2025.
[52] S. Nah, T. Hyun Kim, and K. Mu Lee, “Deep multi-scale convolutional
neural network for dynamic scene deblurring,” inProc. IEEE Conf.
Comput. Vis. Pattern Recognit. (CVPR), pp. 3883–3891, 2017.
[53] Z. Wang, A. Bovik, H. Sheikh, and E. Simoncelli, “Image quality
assessment: from error visibility to structural similarity,”IEEE Trans.
Image Process., vol. 13, no. 4, pp. 600–612, 2004.
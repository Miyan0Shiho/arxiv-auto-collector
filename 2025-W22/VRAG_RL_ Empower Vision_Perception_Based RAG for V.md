# VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning

**Authors**: Qiuchen Wang, Ruixue Ding, Yu Zeng, Zehui Chen, Lin Chen, Shihang Wang, Pengjun Xie, Fei Huang, Feng Zhao

**Published**: 2025-05-28 06:30:51

**PDF URL**: [http://arxiv.org/pdf/2505.22019v1](http://arxiv.org/pdf/2505.22019v1)

## Abstract
Effectively retrieving, reasoning and understanding visually rich information
remains a challenge for RAG methods. Traditional text-based methods cannot
handle visual-related information. On the other hand, current vision-based RAG
approaches are often limited by fixed pipelines and frequently struggle to
reason effectively due to the insufficient activation of the fundamental
capabilities of models. As RL has been proven to be beneficial for model
reasoning, we introduce VRAG-RL, a novel RL framework tailored for complex
reasoning across visually rich information. With this framework, VLMs interact
with search engines, autonomously sampling single-turn or multi-turn reasoning
trajectories with the help of visual perception tokens and undergoing continual
optimization based on these samples. Our approach highlights key limitations of
RL in RAG domains: (i) Prior Multi-modal RAG approaches tend to merely
incorporate images into the context, leading to insufficient reasoning token
allocation and neglecting visual-specific perception; and (ii) When models
interact with search engines, their queries often fail to retrieve relevant
information due to the inability to articulate requirements, thereby leading to
suboptimal performance. To address these challenges, we define an action space
tailored for visually rich inputs, with actions including cropping and scaling,
allowing the model to gather information from a coarse-to-fine perspective.
Furthermore, to bridge the gap between users' original inquiries and the
retriever, we employ a simple yet effective reward that integrates query
rewriting and retrieval performance with a model-based reward. Our VRAG-RL
optimizes VLMs for RAG tasks using specially designed RL strategies, aligning
the model with real-world applications. The code is available at
\hyperlink{https://github.com/Alibaba-NLP/VRAG}{https://github.com/Alibaba-NLP/VRAG}.

## Full Text


<!-- PDF content starts -->

Technical Report
 Qwen
VRAG-RL: Empower Vision-Perception-Based RAG for Visually
Rich Information Understanding via Iterative Reasoning with Re-
inforcement Learning
Qiuchen Wang1, Ruixue Ding2, Yu Zeng1, Zehui Chen1, Lin Chen1
Shihang Wang2, Pengjun Xie2, Fei Huang2, Feng Zhao1†
1MoE Key Laboratory of Brain-inspired Intelligent Perception and Cognition, USTC
2Tongyi Lab, Alibaba Group
Abstract
Effectively retrieving, reasoning and understanding visually rich information remains a
challenge for traditional Retrieval-Augmented Generation (RAG) methods. On the one
hand, traditional text-based methods cannot handle visual-related information. On the
other hand, current vision-based RAG approaches are often limited by fixed pipelines
and frequently struggle to reason effectively due to the insufficient activation of the
fundamental capabilities of models. As reinforcement learning (RL) has been proven
to be beneficial for model reasoning, we introduce VRAG-RL , a novel RL framework
tailored for complex reasoning across visually rich information. With this framework,
VLMs interact with search engines, autonomously sampling single-turn or multi-turn
reasoning trajectories with the help of visual perception tokens and undergoing con-
tinual optimization based on these samples. Our approach highlights key limitations
of RL in RAG domains: (i) Prior Multi-modal RAG approaches tend to merely incor-
porate images into the context, leading to insufficient reasoning token allocation and
neglecting visual-specific perception; and (ii) When models interact with search en-
gines, their queries often fail to retrieve relevant information due to the inability to
articulate requirements, thereby leading to suboptimal performance. To address these
challenges, we define an action space tailored for visually rich inputs, with actions
including cropping and scaling, allowing the model to gather information from a coarse-
to-fine perspective. Furthermore, to bridge the gap between users’ original inquiries
and the retriever, we employ a simple yet effective reward that integrates query rewrit-
ing and retrieval performance with a model-based reward. Our VRAG-RL optimizes
VLMs for RAG tasks using specially designed RL strategies, aligning the model with
real-world applications. Extensive experiments on diverse and challenging benchmarks
show that our VRAG-RL outperforms existing methods by 20% (Qwen2.5-VL-7B) and
30% (Qwen2.5-VL-3B), demonstrating the effectiveness of our approach. The code is
available at https://github.com/Alibaba-NLP/VRAG.
1 Introduction
Retrieval-Augmented Generation (RAG) Gao et al. (2023); Jin et al. (2024); Chen et al. (2025b) enables
Language Models (LMs) to leverage external information to tackle problems. Due to the limitations
of traditional textual RAG methods in handling visually rich information , efforts have been made
to introduce RAG into the visual domain by integrating Vision-Language Models (VLMs) Bai et al.
(2025); Chen et al. (2024d); OpenAI (2024); Jaech et al. (2024); Pichai et al. (2024) with search engines.
However, current visual RAG methods still fall short in effectively reasoning with search engines and
understanding complex visual information. Reinforcement Learning (RL) has been recognized as an
†Corresponding author
1arXiv:2505.22019v1  [cs.CL]  28 May 2025

Technical Report
 Qwen
LLM-based Evaluation…Stacked retrieval results fromRound 1 to N(b) Visual Perception Action Space
Pre-defined Visual Perception Actions
Region of Interest
Select & Crop & Zoom(a) Reinforcement Learning Driven Visual Iterative Reasoning Framework
(c) Fine-grained Retrieval-based Reward
Retrieval-based RewardGolden ImageUnrelatedEvaluate retrieval performance using NDCGPattern RewardModel-based RewardPattern and Action VerificationRule-basedFunction
Answer ExtractionOutcome judgement:True or FalseInstructions
+ Human QueryRollout Module:Interaction with external environmentSearch EngineVisual Perception FunctionPolicy Model (VLMs)
%='(%)*++*(%,-.++(%/0+Reward Func (RAG Specific)Reference ModelH1,H2,...HG.r1,r2,...rG.Adv1,Adv2,...AdvG.GroupComputationKLUpdate Policy ModelAdv.
Trajectories H1,H2,…
Trjs.
Figure 1: Overall Framework of our Reinforcement Learning Framework. (a) demonstrates the
interaction process between the model and the external environment, as well as the implementation
of the GRPO algorithm. (b) shows the proposed visual perception action space which allows the
model to extract information from a coarse-to-fine perspective. (c) is the specially designed reward
for RAG, which combines outcome and retrieval performance across the entire sampling process.
effective approach for optimizing VLMs in complex reasoning tasks Sutton et al. (1999); Kaelbling
et al. (1996); Huang et al. (2025); Meng et al. (2025); Yu et al. (2025a). Therefore, RL offers a promising
approach to address the challenges faced by visual RAG methods.
Inspired by these advancements, we introduce VRAG-RL , a novel multimodal RL framework
specifically designed for iterative reasoning in visually rich information RAG. Our approach is based
on three critical observations: (i) Insufficient activation of reasoning capabilities with visual infor-
mation. Existing methods underutilize the reasoning potential of VLMs when incorporating visual
information. For instance, prior approaches tend to merely embed images into the context without
adequately addressing visual-specific perception processes, resulting in insufficient reasoning token
allocation and limiting the models’ ability to fully leverage visual data for complex reasoning tasks.
(ii) Inefficient and disjointed Retrieval. In previous work, limited by the inability to articulate
complex requirements, models struggled to retrieve relevant information efficiently, which may lead
to repetitive and meaningless interactions, restricting the overall effectiveness. (iii) Inconsistent
multi-turn reasoning and unstable training with VLMs. Current RL frameworks for LMs often
struggle with maintaining stability and consistency during multi-turn reasoning. Handling complex,
multi-step reasoning tasks can be particularly challenging, as models may encounter difficulties in
maintaining effective reasoning across interactions with external environments, leading to inconsis-
tent performance and suboptimal results. This challenge is further exacerbated for VLMs, which are
limited by their instruction-following and reasoning capabilities.
Building upon these insights, VRAG-RL introduces improvements in various modules: (i) We
propose a visual perception action space that includes selecting regions of interest and zooming
into these areas. VLMs with visual perception tokens in the action space are capable of acquiring
information from coarse-to-fine perspective. As shown in Figure 1(b), when dealing with images or
charts within documents, VLMs can give higher attention to information-dense areas through the
proposed perception tokens. This allows the model to more effectively activate reasoning abilities
within a limited context length, preventing the overlooking of details. (ii) Furthermore, rather than
relying solely on a simple outcome-based reward, we factor in the effectiveness of the retrieval
process as part of the reward structure. In particular, during the interaction between the model and
the search engine, retrieving pertinent images promptly enhances the model’s ability to address
questions effectively, whereas persistently retrieving irrelevant documents adds noise and hampers
the reasoning process. As illustrated in Figure 1(c), by integrating retrieval performance into reward,
2

Technical Report
 Qwen
we establish comprehensive guidance for retrieval-augmented generation frameworks. (iii) Inspired
by the current think-then-answer approach and the ReAct paradigm, we model the interaction
between the VLMs and the search engine, along with the visual perception action space, as a
process of iterative reasoning and tool invocation. Figure 1(a) illustrates our training pipeline, which
supports automatic sampling and integrates the GRPO algorithm. To ensure stability in multi-turn
sampling and training, we have carefully designed the sampling strategy including post-processing
for each interaction, and model-based reward together with the retrieval reward mentioned above
guides the model training. Additionally, we have re-annotated existing datasets of visually rich
documents and developed a data construction pipeline to efficiently scale data for RL and SFT.
Our major contributions are as follows:
•We propose VRAG-RL, a novel reinforcement learning framework tailored for training
VLMs to effectively reason, retrieve, and understand visually rich information.
•We define a visual perception action space that includes selecting, cropping, and scaling
regions of interest, allowing VLMs to gather information progressively from coarse-grained
to fine-grained levels. This action space enhances the models’ ability to focus on information-
dense areas and activates their vision-specific reasoning capabilities more effectively.
•We introduce a comprehensive reward structure that integrates retrieval performance and
model-based outcome reward. This reward mechanism aligns the model more closely
with real-world applications, bridging the gap between users’ original intentions and the
retriever.
•Extensive experiments demonstrate the effectiveness of our method. VRAG-RL significantly
outperforms strong baselines, achieving over 20% improvement on various benchmarks.
2 VRAG-RL
In this section, drawing on insights and foundational ideas, we present a comprehensive description
of our VRAG-RL framework. We start with the formulation of the problem (§2.1), then introduce the
action space designed for visual perception (§2.2) and the fine-grained reward specifically defined
for the RAG task (§2.3). Finally, we illustrate the model interaction process in the rollout module
and the reinforcement learning training implementation of our framework (§2.4).
2.1 Problem Formulation
Given a query denoted as q, we have a huge collection of images C={I1,I2,. . .,IN}, consisting of
Nimages. Each image contains a variety of visually rich elements, such as flowcharts, charts, tables,
and diverse layouts, derived from real-world documents across multiple domains, including slides
and reports. Our goal is to efficiently reason, accurately retrieve the most relevant images, extract
valuable information from the complex visual data, and generate the final answer ato the query q.
2.2 Visual Perception Action Integration for Understanding Information-Dense Regions
Previous works merely involved migrating textual RAG to the multi-modal domain, which simply
meant inserting images into the context and then reasoning and responding. However, these efforts
overlooked the characteristics of image data, where the efficiency of visual perception is closely
related to image resolution, visual element layouts, information density, and other visually related
factors. Motivated by these findings, we introduce a dynamic novel visual perception paradigm into
VLMs that involves region selection and re-encoding at the token level, as illustrated in Figure 2.
Definition of Visual Perception Actions. We define the visual perception action space for VLMs by
taking into account the specific characteristics of visual information. This enables the model to select
regions with high information density or regions relevant to the query for a detailed view, acquiring
3

Technical Report
 Qwen
information from a coarse to fine perspective. We integrate search queries, answer summaries, and
visually specific actions into a unified action space to align with the model’s pre-training domain.
Language Model Decoder…………………
Vision Encoder(T1, A1)(T2, A2)(Tk, Ak)Language Model Decoder……………
Vision Encoder(T1, A1)(T2, A2)(Tk, Ak)…Action ExtractionMapping Region Selection
CropO1: Retrieved Imagere-input
Retrieved Image 1:Retrieved Image N:…
Precise retrieval and Efficient perception from coarse to fine(a) Traditional RAG with search action only(b) Our VRAG-RL with unified visual perception action space
Inefficient perception and Repetitive retrievalselect
Figure 2: Comparison between our VRAG-RL and the traditional RAG in terms of perception
methods. (a) Traditional methods lack effective perception, which easily leads to repetitive and
ineffective retrieval calls and suboptimal outcomes. (b) Our VRAG-RL is efficient and accurate,
enabling the model to perceive information-dense regions from a coarse-to-fine perspective.
The policy model πθinteracts with the environment in the Thought-Action-Observation (T,A,O)
paradigm. In each interaction, the model generates the next action At∼πθ(· | H t−1)based on the
trajectory Ht−1from step t−1and earlier. A role-based function is used to extract visual perception
tokens < region > and < /region >, whose main purpose is to select, crop, and zoom in on the region
of interest of the image that has already been retrieved in the context:
At× O k→ O t,k∈ {1, 2, . . . , t−1}, (1)
Given a w×himage as an observation Ok, a bounding box [xmin,ymin,xmax,ymax]within perception
tokens can precisely delineate the position of region R, where (xmin,ymin)and(xmax,ymax)represent
the coordinates of the top-left and bottom-right pixels of region R. Some current models’ pre-training
domains for grounding tasks normalize the coordinates to [0,δ], resulting in actual coordinates
of(x×w
δ,y×h
δ), while other models, such as Qwen2.5VL, directly use the original coordinates
without normalization. Then we will map the selected region Rfrom the image tokens in context to
thewraw×hrawraw image, and crop this raw image to obtain ˆR:
ˆR=Crop(Iraw,[xmin×wraw
wencoder,ymin×hraw
hencoder,xmax×wraw
wencoder,ymax×hraw
hencoder]). (2)
where (wraw,hraw)are the shape of the original image Iraw,(wencoder ,hencoder )are determined by the
vision encoder such that wencoder×hencoder =Pixels max. Finally, ˆRis integrated into the context as
an observation: ˆR → O t. Actually, the image token embedded in the context does not represent
the original size of the image. The maximum pixel size Pixels maxfor the vision encoder is often
considerably smaller than the pixel of visually rich documents found in real-world applications. This
is the reason why the region cropped from the original image and scaled within the vision encoder
has a higher density of vision tokens. This simple yet effective "crop and re-input" strategy enhances
visual perception performance by directly increasing perceptual resolution Yu et al. (2025b); Liu et al.
(2024); Shao et al. (2024).
Trajectory Data Scaling-Up Based on Multi-Expert Sampling. To effectively train the model,
especially smaller-scale models, to learn the utilization of Visual Perception Tokens while retaining
their foundational capabilities, we need to train them with high-quality data through Supervised
Fine-Tuning before applying RL. We propose a multi-expert sampling strategy to scale up the
trajectory data, aiming to sample diverse interactions within the same reasoning trajectory for each
data.
4

Technical Report
 Qwen
The core idea is to utilize large-scale models πLMto effectively guide the reasoning process and
tool selections within a trajectory, while smaller expert models πEMannotate coordinate under the
guidance of large-scale models. At the tthinteraction between the model and the environment:
Ht={T1,A1,O1,· · ·,Ot−1,Tt,At,Ot}, (3)
where Htis the trajectory, representing the sequence of past observations and actions leading up
to the current step. The πLMequipped with extensive capacities for understanding and process-
ing complex multi-modal interactions, act as pioneers in determining the overarching reasoning
pathway:
{Tt,At}=πLM(· | H t−1), (4)
We use a rule-based function to extract action and thought. If the action is search, the engine returns
the original image as Ot. Otherwise, each time a visual perception token is output, we employ
grounding-specific expert models to re-locate the coordinates of regions of interest:
ˆAt=πEM(· | H t−1;Tt), (5)
where the expert models πEMbenefit from the guidance provided by the large model’s thought Tt,
leveraging these insights to enhance their precision in region localization. The newly generated
coordinates of the region of interest ˆAtwill replace the old visual perception tokens Atgenerated by
πLM, and the re-encoded image serves as observation ˆOt:
ˆOt=PV(Ot−1,ˆAt). (6)
where PVrepresents the visual processing function, the selected region will undergo cropping,
zooming in, and re-encoding before being inserted into the context.
2.3 Fine-Grained Reward Function Tailored for Enhancing RAG Framework
Inference w/ NoiseUpper Bound63.5
34.754.156.455.350.142.4Num of Reference ImagesAcc1234567060504030
Figure 3: Experiments on the impact of
context length on model performance.Unlike traditional RL methods that focus only on out-
put results, VRAG-RL emphasizes optimizing retrieval
in RAG, as retrieval quality directly affects overall per-
formance. We designed a reward function with three
components: pattern reward, retrieval efficiency reward,
and model-based outcome reward, guiding the model to
efficiently retrieve information and generate high-quality
answers.
Retrieval Efficiency Reward. As shown in Figure 3,
when the information is sufficient, an excessively long
context can interfere with the model. Therefore, the ear-
lier and more comprehensive the retrieval of relevant information, the better the model can construct
a coherent and informative context for generating high-quality answers. Inspired by Normalized
Discounted Cumulative Gain, and using our predefined relevance of the recalled images, we define:
DCG(Dtrj) =|Dtrj|
∑
i=12si−1
log2(i+1),si=(
1, if di∈ D rel
0, if di/∈ D rel, (7)
where di∈ D trjrepresents stacked retrieved images within the trajectory, Drelis the collection of
relevant golden images, siis the predefined relevance score. We believe that the performance is
optimal when all relevant documents are retrieved first, the Ideal-DCG is defined as:
IDCG (Drel) =|Drel|
∑
i=12srel−1
log2(i+1)+n
∑
i=|Drel|+12sunrel−1
log2(i+1)=|Drel|
∑
i=11
log2(i+1), (8)
where srel=1and sunrel=0respectively represent the relevance scores of ideally relevant and
irrelevant documents. Our Retrieval Efficiency Reward is defined as:
rRet=DCG(Dtrj,Drel)
IDCG (Drel). (9)
where rRet, the modified NDCG, is directly used as the reward to reflect retrieval performance.
5

Technical Report
 Qwen
Pattern Consistency and Model-Based Outcome Rewards. The rule-based pattern reward is
designed to encourage the model to follow the reasoning patterns during the interaction process:
rPat∼Parse (H), (10)
where His the generated trajectory. Parse (·)employ action tokens <search> and </search> to
extract predefined actions in the action space. This is crucial for a reasoning agent with a predefined
action space, as it helps effectively extract actions and thoughts. Regarding outcome reward, unlike
rule-based methods that are prone to falling into local optima, we adopt a model-based reward:
rAns∼πRM(·|Q,Agolden ,Apred), (11)
where Qrepresents the input query, Agolden is the reference golden answer, and Apred is the answer
generated by the VLMs. Based on these inputs, the evaluation model πRMassesses the correctness
of the final answer.
Integrated Reward Function. The final reward function is a weighted combination of the three
components described above, with weights used to balance the contributions of each component:
rϕ=α·rRet+β·rAns+γ·rPat. (12)
where α+β+γ=1. In practice, we usually set γ=0as the model can effectively learn the pattern
after SFT. We set γ=0.1when performing RL with cold start to help the model learn the predefined
pattern. By integrating these three components into the reward function, our VRAG-RL provides
a comprehensive and fine-grained evaluation mechanism that guides the model in optimizing its
reasoning and retrieval capabilities in a way that aligns closely with real-world applications.
2.4 Reinforcement Learning Framework with Iterative Reasoning
We apply RL to multimodal RAG agent tasks to enhance the capability of VLMs in retrieving and
reasoning. Our RL framework is primarily divided into two parts for discussion: the rollout process
for multimodal agent and the reinforcement learning training strategy for multi-turn interactions.
Multi-Round Generation with Search Engine and Visual Perception Actions. As shown in
Algorithm 1, the model interacts with the external environment in multiple turns, where the
observation, which is the image, is inserted into the trajectory in the role of the user. This is necessary
to align with the model’s pre-training domain, where only the user token can insert image tokens.
Algorithm 1 Interaction of VLM with the External Environment through Iterative Reasoning
Input: Input query x, Policy model πθ, External environment V, Maximum iterations T.
Output: Final trajectory y.
1: Initialize rollout sequence y←∅and action count t←0
2:while t<Tdo
3: Generate VLM response sequence yt∼πθ(· |x,y)
4: Concatenate ytto the ysequence with the role of assistant: y←y+yt
5: if<search> </search> detected in ytthen
6: Extract search query q←Parse (yt)and Retrieve related image It=Ret(q)
7: else if <region> </region> detected in ytthen
8: Extract visual perception tokens loc←Parse (yt)and Processing image It=PV(loc,y)
9: else if <answer> </answer> detected in ytthen
10: return final generated trajectory y
11: end if
12: Concatenate vision tokens Itto the sequence ywith the role of user: y←y+It
13: Increment action count t←t+1
14:end while
15:return final generated trajectory y
6

Technical Report
 Qwen
Table 1: Main Results. The best performance are marked in bold. SlideVQA and ViDoSeek mainly
focus on reasoning type, while MMLongBench focuses on the visual type of reference content.
OCR-based ( ) RAG and purely visual ( ) RAG are evaluated with the same prompt and setting.
METHODSLIDE VQA VIDOSEEK MML ONG BENCHOVERALLSingle-hop Multi-hop Extraction Logic Text Table Chart Figure Layout
Qwen2.5-VL-3B-Instruct
Vanilla RAG 15.1 12.1 8.8 14.3 3.9 5.1 1.7 3.1 2.5 11.2
ReAct 11.8 9.9 5.3 7.4 6.5 3.7 3.9 5.2 2.5 8.4
Search-R1 17.5 13.8 13.3 20.7 3.4 3.2 4.5 4.1 6.8 14.1
Vanilla RAG 19.4 12.2 10.1 17.3 2.2 4.1 5.2 4.7 4.3 13.2
ReAct 15.7 10.9 6.7 14.2 2.7 3.6 3.4 3.1 5.1 10.9
Search-R1-VL 26.3 20.1 20.1 29.8 8.5 7.8 7.9 9.3 7.6 21.3
VRAG-RL 65.3 38.6 63.1 73.8 22.7 16.1 21.9 21.4 19.5 53.5
Qwen2.5-VL-7B-Instruct
Vanilla RAG 26.1 10.6 24.7 30.9 8.5 5.4 11.7 4.4 3.3 20.9
ReAct 21.2 13.3 14.3 21.3 5.9 5.1 7.3 5.5 1.7 15.8
Search-R1 28.4 19.7 20.8 30.6 9.9 6.0 7.9 10.1 5.9 22.2
Vanilla RAG 29.1 17.4 26.4 41.3 13.1 14.7 15.9 4.3 7.6 24.2
ReAct 34.8 20.4 27.5 42.1 10.1 12.4 10.2 6.2 7.1 26.9
Search-R1-VL 48.3 42.3 40.5 50.3 19.9 13.4 12.9 11.4 10.2 37.4
VRAG-RL 69.3 43.1 60.6 74.8 26.1 26.3 24.8 25.9 21.2 57.1
Training Strategy for Reinforcement Learning in Multi-Step Interactions. We propose a RL
framework that enables VLM to learn how to interact with search engines and gather visually rich
information from a coarse-to-fine perspective. The optimization objective is formulated as:
max
πθEx∼D,y∼πθ(·|x;V)
rϕ(x,y)−βDKL[πθ(y|x;V)||πref(y|x;V)], (13)
where the πθis the policy model, πre fis the reference model, DKLis KL-divergence, and y∼
πθ(· |x;V) =πθ(· |x)⊗ V is the rollout process. Our approach implements Group Relative Policy
Optimization (GRPO) Guo et al. (2025), which optimizes the model’s retrieval-augmented reasoning
capability with group-sampled role-play trajectories.
3 Experiments
3.1 Experimental Settings
Datasets, Metric and Baselines. To evaluate the effectiveness of VRAG-RL, we compare our
method with the text-based and vision-based baselines: (1) Vanilla RAG Faysse et al. (2024) uses the
original question as a query for the search engine, then VLMs perform direct inference. (2) ReAct
Yao et al. (2023): The model performs rewriting, retrieving, and reasoning in the think-then-act
paradigm. (3) Search-R1(-VL) is the baseline adapted from Search-R1 Jin et al. (2025), and the
settings are aligned across all experiments to ensure fairness. We evaluate our method on three
challenging, visually rich benchmarks: ViDoSeek Wang et al. (2025a), SlideVQA Tanaka et al. (2023)
and MMLongBench Ma et al. (2024). The model-based evaluation metric is binary 0 or 1, indicating
the accuracy of the model’s responses.
Training and Inference Setups. We conducted SFT and RL on llama-factory Zheng et al. (2024)
and verl Sheng et al. (2024) respectively. We use full parameter fine-tuning and cosine learning
scheduler with a warmup ratio of 0.1 during SFT. When training with the GRPO algorithm, we set
the group size to 5 and the coefficient for the KL loss is typically set to 0.01, but if we perform cold
start, we set it to 0 to disable the KL loss constraint on the model. During training and inference, we
built a search engine from a database of approximately ∼70k visual documents.
7

Technical Report
 Qwen
3.2 Results
Main Results. As shown in Table 1, compared to purely visual methods, OCR-based methods
exhibit significant limitations on visually intensive benchmarks. On the one hand, visual information
inherently contains elements that cannot be represented by text, such as element positions, layout,
and color, etc. On the other hand, the perceptual capabilities of OCR models are considerably
inferior to those of the current advanced VLMs, which restricts the overall performance ceiling
of the framework. Visual-based methods have proven to be a more elegant solution compared to
OCR-based methods, especially in tasks related to visual understanding. For prompt-based baselines
of vision domain, Vanilla RAG and ReAct exhibit poor performance, far behind RL-based baselines
and our method on various benchmarks. The 7B model, compared to the 3B model, possesses
superior perception and understanding capabilities, exhibiting strong performance across various
datasets. For RL-based baselines, our method also performs better than search-R1-VL on both
Qwen2.5-VL-7B-Instruct (34.7 →57.1) and Qwen2.5-VL-3B-Instruct (21.3 →53.5). The evaluation
results on SlideVQA and ViDoSeek demonstrate our model’s significant improvement in reasoning
capabilities across various reasoning tasks. Furthermore, as MMLongBench includes multiple
visual elements, which indicates the model’s improvement in visual perception capabilities, this
phenomenon is related to our proposed visual perception action space. The results across various
benchmarks prove the effectiveness and generalization of our method in the retrieval and reasoning
of visually rich information.
Table 2: Ablation study on three benchmarks.
REWARD ACTION SPACEAccuracyVanilla RAG-Specific Search Visual-Perception
✓ ✓ 47.2
✓ ✓ ✓ 49.3
✓ ✓ 54.9
✓ ✓ ✓ 57.1Approach Ablations. As shown in Table 2,
taking Qwen2.5-VL-7B-Instruct as an example,
we decompose the key components of VRAG-
RL to examine the impact of different rewards
and action space on performance separately. In
a macro view, removing each module results in
a clear drop in the accuracy, which validates the
power of our RAG-specific reward and Visual-
perception action space. The action space module we defined shows a certain degree of improvement
on different bases, which proves the effectiveness of the visual perception-based strategy . Consis-
tent with the findings demonstrated in MMLongBench in Figure 5, the visual perception action space
we introduced has generally enhanced the framework’s performance, particularly in improving
high-density visual information. Furthermore, ablation experiments on the reward model further
demonstrate that retrieving relevant information is a prerequisite for high-quality generation, high-
lighting the role of high-quality retrieval in RAG, which proves the importance of our RAG-specific
reward . Comparisons and analyses of experiments across different settings collectively demon-
strate the effectiveness and generalization of our modules, and their combination comprehensively
enhances end-to-end performance from various perspectives.
3.3 Analysis
40506070
SlideVQAViDoSeekMMLongBenchVRAG-RLReActDirect RetrievalRecall@1
Figure 4: Retrieval perfor-
mance of our approach.Better retrieval facilitates high-quality generation. Our VRAG-
RL framework significantly enhances the retrieval efficiency, which
is crucial for constructing a coherent and informative context for
high-quality generation. As demonstrated in Figure 3, the context
length has a substantial impact on model performance. When the
context is too long, it can introduce noise and interfere with the
model’s ability to generate accurate answers. In contrast, when
relevant information is retrieved early and comprehensively, the
model can build a more focused and informative context. As shown in Figure 4, our model is more
effective at retrieving relevant information compared to traditional prompt-based rewrite methods.
Our approach provides the vision model with a better context for generating high-quality answers.
8

Technical Report
 Qwen
TextTable
ChartFigureLayout
Vanillaw/ Visual Perceptionw/o Visual Perception
Figure 5: Relative perfor-
mance on MMLongBench.Visual perception action space provides a fine-grained perspective.
The visual perception action space introduced in our framework further
enhances understaning by allowing the model to focus on information-
dense regions of images. Figure 5 illustrates the relative performance
comparison between our approach with visual perception action space
and various baselines, from which we can observe that VRAG-RL not
only performs well in textual tasks but also shows noticeable improve-
ments in tasks requiring visual perception abilities, particularly in
Layout, Chart, and Figure. This is particularly important given the cur-
rent limitations in computational resources, especially considering that
VLMs are highly memory-intensive. Using this dynamic resolution
strategy, the model can achieve more detailed perception within the constraints of limited computa-
tional resources, rather than simply maximizing the resolution of the original image. Our method
achieves an improvement in perceptual abilities while optimizing resource utilization. Perhaps this
human-like way of thinking and acting is the key to AGI.
Table 3: Average Finish Rate ( %) and
Average Invalid Action Rate (%).
Method Invalid Action Rate ↓Finish Rate ↑
SFT 9.4 84.2
+ RL 5.1 97.1Reinforcement learning helps the model to perform multi-
step reasoning effectively. One major challenge of the
prompt-based method is that as the number of interactions
increases, the model’s capability to follow instructions weak-
ens. However, pre-training with SFT helps the model reason
in a pre-defined pattern compared to cold strat, but it also im-
pacts the model’s inherent foundational capabilities to some extent. To further explore the activation
of multi-turn reasoning abilities in models by RL, we compared the iterative reasoning performance
of models with and without RL, as shown in Table 3. For our method with action space, effective
actions are crucial for interacting with the external environment. The Invalid Action Rate indicates
incorrect action responses, which include not only pattern errors but also hallucinations caused by
wrong cropping, answering before retrieval, and so on. Inefficient reasoning often includes repeated
meaningless searches, leading to a decrease in the finish rate. Our method with RL effectively re-
duces the invalid rate and increases the finish rate. It guides the model to make optimal decisions at
each step of the reasoning process, enabling it to flexibly adjust strategies when faced with different
types of out-of-domain visual information, thereby better completing complex reasoning tasks.
Model-based reward offers more stable training compared to rule-based reward. Previous works
often use EM as the reward, which is too strict. Unlike short answers for data-related questions,
it is difficult for the model’s responses to exactly match the golden answer, resulting in inefficient
training. However, using recall as a reward may lead to misjudgments and cause models to hack
the function, resulting in repetitive responses that destabilize training. In contrast, a model-based
reward leverages an evaluation model to assess the quality and relevance of generated responses
in a more flexible manner. This approach not only aligns better with real-world applications but
also provides a more stable and effective training signal, as demonstrated in Appendix A. The
model-based reward thus enables VRAG-RL to achieve more robust performance across visual
reasoning tasks.
AnswerSearchPerceptionAvg. Action CountVanilla RAGReAct RAGVRAG-RL
Figure 6: Latency Analysis on Generation.Time efficiency. As shown in Figure 6, our
method’s multi-turn interaction with external envi-
ronments can lead to increased latency. The latency
of vanilla RAG remains consistent, as it only per-
forms a single search and provides an answer. ReAct
RAG, a prompt-based method, also demonstrates
multi-turn interaction capabilities due to the funda-
mental reasoning abilities of the model. However, it is limited to only two defined actions: answer
and search. Due to the lack of sufficient perception capabilities, it often falls into repetitive search
loops. Our approach equips the model with a visual perception space that can effectively understand
visually rich images. The model can quickly extract answers after retrieval, thus avoiding ineffective
9

Technical Report
 Qwen
searches. Despite the increase in latency, the overall performance improves due to the higher quality
of generated answers, making the trade-off between latency and accuracy highly beneficial for
visually rich retrieval and understanding tasks.
Case Study. In Figure 7 and 8 (Appendix H), we list the trajectories of our VRAG-RL to illustrate
how our model reasons and interacts with the environment. These cases highlight two challenges
in visually rich information RAG: (1) accurately retrieving relevant images, and (2) the reference
information often requires higher-resolution perception. In Figure 7, we can observe that the model
demonstrated reflective capability, and eventually identified subtle clues in the relevant images.
Moreover, as shown in Figure 8, the model engages in visual perception actions only when required,
showcasing human-like reasoning instead of simply replicating patterns from its training data.
4 Related Work
Vision-based Retrieval-augmented Generation. RAG demonstrates significant advantages in
addressing knowledge-intensive problems Lewis et al. (2020); Gao et al. (2023); Chen et al. (2024a).
Traditional text-based RAG methods typically involve designing different agents to interact with
search engines Wu et al. (2025b); Chen et al. (2024b;c); Wu et al. (2025a); Li et al. (2023); Moreira et al.
(2024); Lee et al. (2024). However, with the widespread adoption of electronic documents, knowledge
is no longer confined to text. Recently, there has been an increasing amount of research on OCR-free
retrieval methods that directly align textual queries with imagesYu et al. (2024); Faysse et al. (2024).
Furthermore, more and more work is focusing on multimodal RAG agents Wang et al. (2025a); Cho
et al. (2024); Jiang et al. (2024); Li et al. (2024); Xia et al. (2024), enabling more accurate retrieval and
extraction of visual information. Our work builds upon these developments by incorporating visual
perception actions into visual-based RAG, effectively activating the reasoning and understanding
capabilities of VLMs.
Reinforcement Learning with Large Models. Reasoning capabilities are crucial for models to
effectively address complex problems, and RL has been proven to be a powerful approach to enhance
these capabilitiesGuo et al. (2025); Jaech et al. (2024). Previous work applied RL in the training of
LLMs Meng et al. (2024); Williams (1992); Rafailov et al. (2023); Schulman et al. (2017); Guo et al.
(2025). Additionally, more and more works aim to use RL to enhance the reasoning capabilities
of VLMs Chen et al. (2025a); Meng et al. (2025); Liu et al. (2025). Recent advancements have seen
RL being widely applied to the training of large model-driven agents Wang et al. (2025b). These
agents, especially RAG agents, require robust multi-step reasoning capabilities to interact effectively
with external environments Jiang et al. (2025); Li et al. (2025). However, there is still a scarcity of RL
frameworks specifically tailored for multimodal iterative reasoning, which is essential for handling
visually rich information. Our work aims to fill this gap by introducing a novel RL framework that
enables VLMs to perform iterative reasoning with visual perception actions, thereby enhancing their
reasoning capabilities in complex, multi-modal retrieval-augmented reasoning tasks.
5 Conclusion and Future Work
In this paper, we introduce VRAG-RL, a novel reinforcement learning framework tailored for com-
plex reasoning across visually rich information. Our approach enables Vision Language Models to
interact with search engines more effectively, significantly enhancing their reasoning and retrieval ca-
pabilities. Extensive evaluations on various benchmarks have demonstrated significant advantages
in visual information reasoning, retrieval, and understanding with our model. For future work, we
plan to introduce more actions that mimic how humans handle complex information, allowing the
model to focus more on deep thinking. Additionally, we aim to reduce hallucinations by leveraging
more advanced models, further improving the accuracy and reliability of our framework.
10

Technical Report
 Qwen
References
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang,
Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 , 2025.
Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. Benchmarking large language models in
retrieval-augmented generation. In Proceedings of the AAAI Conference on Artificial Intelligence , pp.
17754–17762, 2024a.
Liang Chen, Lei Li, Haozhe Zhao, Yifan Song, and Vinci. R1-v: Reinforcing super generalization
ability in vision-language models with less than $3. https://github.com/Deep-Agent/R1-V ,
2025a. Accessed: 2025-02-02.
Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z Pan,
Wen Zhang, Huajun Chen, Fan Yang, et al. Research: Learning to reason with search for llms via
reinforcement learning. arXiv preprint arXiv:2503.19470 , 2025b.
Zehui Chen, Kuikun Liu, Qiuchen Wang, Jiangning Liu, Wenwei Zhang, Kai Chen, and Feng Zhao.
Mindsearch: Mimicking human minds elicits deep ai searcher. arXiv preprint arXiv:2407.20183 ,
2024b.
Zehui Chen, Kuikun Liu, Qiuchen Wang, Wenwei Zhang, Jiangning Liu, Dahua Lin, Kai Chen, and
Feng Zhao. Agent-flan: Designing data and methods of effective agent tuning for large language
models. arXiv preprint arXiv:2403.12881 , 2024c.
Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong
Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal
models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271 , 2024d.
Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. M3docrag: Multi-
modal retrieval is what you need for multi-page multi-document understanding. arXiv preprint
arXiv:2411.04952 , 2024.
Yuning Du, Chenxia Li, Ruoyu Guo, Xiaoting Yin, Weiwei Liu, Jun Zhou, Yifan Bai, Zilin Yu, Yehua
Yang, Qingqing Dang, and Haoshuang Wang. Pp-ocr: A practical ultra lightweight ocr system,
2020. URL https://arxiv.org/abs/2009.09941 .
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre
Colombo. Colpali: Efficient document retrieval with vision language models. In The Thirteenth
International Conference on Learning Representations , 2024.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen
Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey.
arXiv preprint arXiv:2312.10997 , 2:1, 2023.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via
reinforcement learning. arXiv preprint arXiv:2501.12948 , 2025.
Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and
Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models.
arXiv preprint arXiv:2503.06749 , 2025.
Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec
Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint
arXiv:2412.16720 , 2024.
Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanmin Wu, Jiayi Lei, Pengshuo Qiu, Pan Lu, Zehui Chen,
Chaoyou Fu, Guanglu Song, et al. Mmsearch: Benchmarking the potential of large models as
multi-modal search engines. arXiv preprint arXiv:2409.12959 , 2024.
11

Technical Report
 Qwen
Pengcheng Jiang, Jiacheng Lin, Lang Cao, Runchu Tian, SeongKu Kang, Zifeng Wang, Jimeng
Sun, and Jiawei Han. Deepretrieval: Hacking real search engines and retrievers with large
language models via reinforcement learning. arXiv preprint arXiv: 2503.00223 , 2025. URL https:
//arxiv.org/abs/2503.00223 .
Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O Arik. Long-context llms meet rag: Overcoming
challenges for long inputs in rag. arXiv preprint arXiv:2410.05983 , 2024.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and
Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement
learning. arXiv preprint arXiv:2503.09516 , 2025.
Leslie Pack Kaelbling, Michael L Littman, and Andrew W Moore. Reinforcement learning: A survey.
Journal of artificial intelligence research , 4:237–285, 1996.
Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro,
and Wei Ping. Nv-embed: Improved techniques for training llms as generalist embedding models.
arXiv preprint arXiv:2405.17428 , 2024.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks. Advances in neural information processing systems , 33:
9459–9474, 2020.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and
Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models. arXiv preprint
arXiv:2501.05366 , 2025.
Yangning Li, Yinghui Li, Xinyu Wang, Yong Jiang, Zhen Zhang, Xinran Zheng, Hui Wang, Hai-Tao
Zheng, Fei Huang, Jingren Zhou, et al. Benchmarking multimodal retrieval augmented generation
with dynamic vqa dataset and self-adaptive planning agent. arXiv preprint arXiv:2411.02937 , 2024.
Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards
general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281 ,
2023.
Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction
tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp.
26296–26306, 2024.
Jerry Liu. LlamaIndex, 11 2022. URL https://github.com/jerryjliu/llama_index .
Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi
Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785 , 2025.
Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan
Ma, Xiaoyi Dong, et al. Mmlongbench-doc: Benchmarking long-context document understanding
with visualizations. arXiv preprint arXiv:2407.01523 , 2024.
Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi,
Wenhai Wang, Junjun He, Kaipeng Zhang, et al. Mm-eureka: Exploring visual aha moment with
rule-based large-scale reinforcement learning. arXiv preprint arXiv:2503.07365 , 2025.
Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a reference-
free reward. Advances in Neural Information Processing Systems , 37:124198–124235, 2024.
Gabriel de Souza P Moreira, Radek Osmulski, Mengyao Xu, Ronay Ak, Benedikt Schifferer, and
Even Oldridge. Nv-retriever: Improving text embedding models with effective hard-negative
mining. arXiv preprint arXiv:2407.15831 , 2024.
12

Technical Report
 Qwen
OpenAI. Hello gpt-4o. https://openai.com/index/hello-gpt-4o/ , 2024.
Sundar Pichai, D Hassabis, and K Kavukcuoglu. Introducing gemini 2.0: our new ai model for the
agentic era, 2024.
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea
Finn. Direct preference optimization: Your language model is secretly a reward model. Advances
in Neural Information Processing Systems , 36:53728–53741, 2023.
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy
optimization algorithms. arXiv preprint arXiv:1707.06347 , 2017.
Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and
Hongsheng Li. Visual cot: Advancing multi-modal language models with a comprehensive
dataset and benchmark for chain-of-thought reasoning. Advances in Neural Information Processing
Systems , 37:8612–8642, 2024.
Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng,
Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint
arXiv: 2409.19256 , 2024.
Richard S Sutton, Andrew G Barto, et al. Reinforcement learning. Journal of Cognitive Neuroscience ,
11(1):126–134, 1999.
Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito.
Slidevqa: A dataset for document visual question answering on multiple images. In Proceedings of
the AAAI Conference on Artificial Intelligence , pp. 13636–13645, 2023.
Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun Xie, and Feng Zhao.
Vidorag: Visual document retrieval-augmented generation via dynamic iterative reasoning agents.
arXiv preprint arXiv:2502.18017 , 2025a.
Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin,
Kefan Yu, Minh Nhat Nguyen, Licheng Liu, Eli Gottlieb, Yiping Lu, Kyunghyun Cho, Jiajun Wu,
Li Fei-Fei, Lijuan Wang, Yejin Choi, and Manling Li. Ragen: Understanding self-evolution in llm
agents via multi-turn reinforcement learning, 2025b. URL https://arxiv.org/abs/2504.20073 .
Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement
learning. Machine learning , 8:229–256, 1992.
Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang, Linhai Zhang, Yulan
He, Deyu Zhou, Pengjun Xie, et al. Webwalker: Benchmarking llms in web traversal. arXiv
preprint arXiv:2501.07572 , 2025a.
Weiqi Wu, Shen Huang, Yong Jiang, Pengjun Xie, Fei Huang, and Hai Zhao. Unfolding the head-
line: Iterative self-questioning for news retrieval and timeline summarization. arXiv preprint
arXiv:2501.00888 , 2025b.
Peng Xia, Kangyu Zhu, Haoran Li, Hongtu Zhu, Yun Li, Gang Li, Linjun Zhang, and Huaxiu Yao.
Rule: Reliable multimodal rag for factuality in medical vision language models. In Proceedings of
the 2024 Conference on Empirical Methods in Natural Language Processing , pp. 1081–1093, 2024.
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li,
Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang,
Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia,
Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu
Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115 ,
2024.
13

Technical Report
 Qwen
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao.
React: Synergizing reasoning and acting in language models. In International Conference on Learning
Representations (ICLR) , 2023.
En Yu, Kangheng Lin, Liang Zhao, Jisheng Yin, Yana Wei, Yuang Peng, Haoran Wei, Jianjian Sun,
Chunrui Han, Zheng Ge, et al. Perception-r1: Pioneering perception policy with reinforcement
learning. arXiv preprint arXiv:2504.07954 , 2025a.
Runpeng Yu, Xinyin Ma, and Xinchao Wang. Introducing visual perception token into multimodal
large language model. arXiv preprint arXiv:2502.17425 , 2025b.
Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang,
Xu Han, Zhiyuan Liu, et al. Visrag: Vision-based retrieval-augmented generation on multi-
modality documents. arXiv preprint arXiv:2410.10594 , 2024.
Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and
Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceed-
ings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System
Demonstrations) , Bangkok, Thailand, 2024. Association for Computational Linguistics. URL
http://arxiv.org/abs/2403.13372 .
14

Technical Report
 Qwen
A Model-Based Reward
We employ a model-based reward to evaluate the quality and relevance of generated responses.
Specifically, we utilize Qwen2.5-7B-Instruct Yang et al. (2024) as our reward model. This model is
deployed on 4 NVIDIA A100 GPUs to enable efficient batch evaluation. The prompt used for the
reward model is illustrated in Figure 12. Given the input query, reference answer, and generated
response, the reward model assesses the correctness of the generated response and outputs a binary
value (0 or 1) to represent the accuracy of the answer. Compared to the rule-based reward like exact
match (EM) or Recall, used in previous work Jin et al. (2025); Chen et al. (2025a), our model-based
reward provides a more flexible and comprehensive evaluation of the generated response. This
leads to higher training efficiency and better generalization to diverse datasets.
B The implementation of the search engine
To effectively support the retrieval-augmented generation tasks in our VRAG-RL framework, we
implemented OCR-based and vision-based pipeline separately. The vision-based retriever is built
upon the state-of-the-art embedding model ColPali Faysse et al. (2024), which is specifically designed
for aligning textual queries with images. For the textual retrieval pipeline, we employ the PP-OCR
Du et al. (2020) to extract text from images. We utilize the Llama-Index Liu (2022) to ensure an
efficient indexing and querying mechanism for large-scale image datasets. In our experiments, we
deployed the search engine on a single NVIDIA A100 80G GPU, allowing us to handle large-scale
queries efficiently. The use of batch querying further optimizes the retrieval speed, making it suitable
for real-time applications.
C Reinforcement Learning Framework with GRPO
Our framework implements the Group Relative Policy Optimization (GRPO), which leverages the
average reward of multiple sampled outputs as a baseline rather than relying on a learned value
function. The policy model is optimized by maximizing the following objective function:
JGRPO(θ) =Ex∼D,{yi}G
i=1∼πold(·|x;V)"
1
GG
∑
i=11
∑|yi|
t=1I(yi,t)|yi|
∑
t=1:I(yi,t)=1min 
πθ(yi,t|x,yi,<t;V)
πold(yi,t|x,yi,<t;V)ˆAi,t,
clip 
πθ(yi,t|x,yi,<t;V)
πold(yi,t|x,yi,<t;V), 1−ϵ, 1+ϵ!
ˆAi,t!
−βDKL[πθ||πref]
where rollout module samples a group of trajectories {y1,y2,. . .,yG}from the reference policy
πreffor each input question xby interacting with the external environment V.ˆAi,trepresent the
advantage, computed based on the relative rewards of outputs within each group.
D Expert Trajectories Collection
Data Collection. To train our model effectively, we collected expert trajectories using Qwen-VL-
max-latest for prompt-based data collection. Specifically, we utilized the React-based prompt to
gather data, ensuring that the model could perform complex reasoning tasks. During the data
collection process, whenever grounding was required to focus on specific regions of interest within
images, we employed Qwen2.5VL-72B to perform the grounding tasks. This was done under the
guidance of the historical trajectories.
Data Proportions. To ensure that our model could perform diverse multi-step reasoning during
Reinforcement Learning (RL), we carefully balanced the training data. Specifically, we balanced
the trajectories based on the number of steps (2-6) and the types of actions involved (search and
15

Technical Report
 Qwen
perception). This approach ensured that the model was exposed to a wide range of reasoning tasks
and could learn to handle different types of interactions with the environment effectively.
E Dataset Information
We evaluate our method on three visually rich document datasets: SlideVQA, ViDoSeek, and
MMLongbench.
1.SlideVQA Tanaka et al. (2023) is a dataset for document visual question answering focused
on understanding slides. It contains over 2,600 slide decks with more than 52,000 slide
images and 14,500 questions that require complex reasoning skills such as single-hop, multi-
hop, and numerical reasoning. The dataset is designed to support various reasoning types
and includes annotated arithmetic expressions for numerical questions to enhance reasoning
capabilities.
2.ViDoSeek Wang et al. (2025a) is a dataset specifically designed for visually rich document
retrieval-reason-answer tasks. It aims to evaluate the performance of RAG systems on
large-scale document collections. Unlike traditional VQA datasets that focus on single
images or documents, ViDoSeek contains queries with unique answers across a collection
of approximately 6,000 images, covering diverse content types such as text, charts, tables,
and layouts. This dataset provides a more comprehensive and challenging benchmark for
evaluating the retrieval and reasoning capabilities of RAG models in real-world scenarios.
3.MMLongbench Ma et al. (2024) is a dataset designed to evaluate the document under-
standing capabilities of VLMs with an emphasis on long-context, multi-modal documents
composed of text, images, charts, tables, and layout structures.
F Compared Baselines
Here we detailedly introduce the baselines we compare with and our re-produce details.
1.Vanilla RAG . There are two types of Vanilla RAG: text-based and visual-based. Text-based
Vanilla RAG uses text as the retrieval corpus, which is reflected in text search engines and
text modality generation. During the retrieval phase, it directly uses the original question
to search for relevant text, which is then inserted into the context to answer the question.
Visual-based Vanilla RAG uses images as the corpus. During the retrieval phase, it directly
uses the original question to search for relevant images, which are then inserted into the
context to answer the question.
2.ReAct RAG Yao et al. (2023). The method incorporates Chain-of-Thought (COT) prompting
in RAG agent tasks with a format of a Thought-Action-Observation loop. The main dif-
ference between text-based and visual-based approaches lies in the retrieval corpus of the
search engine and the modality of the information inserted.
3.Search-R1 Jin et al. (2025). The method introduces multi-turn reasoning RL into the text
RAG. We used our framework for reproducing, which includes multi-turn interactions and
rule-based rewards.
4.Search-R1-VL . This is a vision-based baseline implemented on our framework based on
search-R1. We used the same reward and post-process methods and trained models based
on cold start with the same dataset as VRAG-RL.
G Hyperparameters
The detailed hyperparameters we use during training are shown in Table 4 and Table 5. We employ
identical hyperparameters for different models.
16

Technical Report
 Qwen
Table 4: Key hyperparameters for SFT.
Name Value
Finetuning type Full
Freeze vision tower True
Freeze multi-modal projector True
Freeze language model False
Cutoff len 16384
Epochs 3
Batch size 16
Gradient accumulation steps 2
Learning rate 1.0e-5
LR scheduler type cosine
Warmup ratio 0.1Table 5: Key hyperparameters for RL.
Name Value
Number of agent groups 5
Warmup steps ratio 0.285
Mini batch size 64
Micro batch size per GPU 2
Learning rate (Actor) 1.0e-6
KL loss coefficient 0.01 (optional)
Tensor model parallel size 4
Total epochs 1
Max prompt length 8192
Max response length 2048
GPU memory utilization 0.6
H Case Study
In Figure 7 and 8, we list the trajectories of our VRAG-RL to illustrate how our model reasons and
interacts with the environment. These cases highlight two challenges in visually rich information
RAG: (1) accurately retrieving relevant images, and (2) the reference information often requires
higher-resolution perception. In Figure 7, we can observe that the model demonstrated reflective
capability, and eventually identified subtle clues in the relevant images. Moreover, as shown in
Figure 8, the model engages in visual perception actions only when required, showcasing human-like
reasoning instead of simply replicating patterns from its training data.
I Prompts
In this section, we illustrate all the prompts used in our paper. Part of our prompts are taken from
Search-R1 Jin et al. (2025).
I.1 Vanilla RAG Prompt
See Figure 11.
I.2 Search-R1 Prompt
See Figure 10.
I.3 ReAct RAG Prompt
ReAct RAG uses the same prompt as Search-R1, as shown in Figure 10.
I.4 VRAG-RL Prompt
See Figure 9.
I.5 Model-based Reward Prompt
See Figure 12.
17

Technical Report
 Qwen
Question:HowmanymoreDugwellsaretherethanShallowTubewellsintheyearsofIndia'sEnergyDividewhosemapwaspreparedbyIWMIbasedonMinorIrrigationCensus?
<think>Toanswerthisquestion,IneedtofindthespecificdatarelatedtoDugwellsandShallowTubewellsinIndia'sEnergyDividemappreparedbyIWMIbasedonMinorIrrigationCensus.Thisrequiressearchingfortherelevantinformation.</think><search>India'sEnergyDivideIWMIMinorIrrigationCensusDugwellsShallowTubewells</search>
<think>TheimageprovidedisamapshowingtheproportionofelectricstructuresinIndia'sEnergyDivide.Itindicatestheproportionofelectricstructuresusingdifferentcolorcodes.However,itdoesnotprovidespecificnumbersforDugwellsandShallowTubewells.IneedtosearchformoredetaileddataonDugwellsandShallowTubewellsfromtheMinorIrrigationCensus.</think><search>DetaileddataonDugwellsandShallowTubewellsdata</search>
<think>ThecroppedimageprovidesthespecificnumbersforDugwells(9.20million)andShallowTubewells(9.10million).TofindhowmanymoreDugwellstherearethanShallowTubewells,IneedtosubtractthenumberofShallowTubewellsfromthenumberofDugwells.</think><answer>Thereare0.10million(or100,000)moreDugwellsthanShallowTubewells.</answer><think>Theimagecontainsatablewiththenumberofstructures,includingDugwellsandShallowTubewells.Ineedtocropthispartoftheimagetofocusontherelevantdata.</think><region>[361,366,721,517]</region>
Select & Crop & Zoom in
Image Retrieval
Image Retrieval
Figure 7: Case 1 for VRAG-RL.
18

Technical Report
 Qwen
Question:WhichH2antagonisthasalowerbioavailability:thedrugthatinhibitsseveralCYP-450isoenzymesandreduceshepaticbloodfloworFamotidine?
<think>Toanswerthisquestion,IneedtoidentifytheH2antagonistthatinhibitsseveralCYP-450isoenzymesandreduceshepaticbloodflow.Then,IwillcompareitsbioavailabilitywiththatofFamotidine.</think><search>H2antagonistthatinhibitsseveralCYP-450isoenzymes</search>
<think>Theimageprovidesinformationaboutdruginteractions,specificallymentioningCimetidine'seffectonCYP-450isoenzymesandhepaticbloodflow.Thisalignswiththesearchquery.Ineedtolookatthelowerrightcorneroftheimagetoconfirmthis.</think><region>[556,337,722,505]</region>
<think>Basedontheprovidedtable,Cimetidinehasabioavailabilityof80%,whileFamotidinehasabioavailabilityof40%.Therefore,FamotidinehasalowerbioavailabilitycomparedtoCimetidine.ThedrugthatinhibitsseveralCYP-450isoenzymesandreduceshepaticbloodflowisCimetidine,asindicatedinpreviousinformation.</think><answer>Famotidinehasalowerbioavailability(40%)comparedtoCimetidine(80%).Therefore,theH2antagonistwithlowerbioavailabilityisFamotidine.</answer><think>ThediagramconfirmsthatCimetidineinhibitstheP-450enzymes,leadingtoincreasedserumconcentrationsofdrugsmetabolizedbytheseenzymes.Now,IneedtofindinformationonthebioavailabilityofCimetidineandFamotidinetodeterminewhichhaslowerbioavailability.</think><search>bioavailabilityofCimetidineandFamotidine</search>
Image Retrieval
Image Retrieval
Select & Crop & Zoom in
Figure 8: Case 2 for VRAG-RL.
19

Technical Report
 Qwen
VRAG-RL Prompt.
System Prompt:
Answer the given question. You must conduct reasoning inside <think> and </think> first
every time you get new information. After reasoning, if you find you lack some knowledge,
you can call a search engine by <search> query </search> and user will return the searched
results. Every time you retrieve an image, you have the option to crop it to obtain a clearer
view, the format for coordinates is <bbox>[x1, y1, x2, y2]</bbox>. You can search as many
times as your want. If you find no further external knowledge needed, you can directly
provide the answer inside <answer> and </answer>, without detailed illustrations. For
example, <answer> Beijing </answer>.
User Prompt:
Query: {Query Description}
Figure 9: Prompt of VRAG-RL.
Search-R1(-VL) Prompt.
System Prompt:
Answer the given question. You must conduct reasoning inside <think> and </think> first
every time you get new information. After reasoning, if you find you lack some knowledge,
you can call a search engine by <search> query </search> and user will return the searched
results. You can search as many times as your want. If you find no further external knowledge
needed, you can directly provide the answer inside <answer> and </answer>, without
detailed illustrations. For example, <answer> Beijing </answer>.
User Prompt:
Query: {Query Description}
Figure 10: Prompt of Search-R1(-VL) and ReAct RAG.
20

Technical Report
 Qwen
Vanilla RAG Prompt.
System Prompt:
Answer the given question. You must conduct reasoning inside <think> and </think> first
every time you get new information. After reasoning, you should directly provide the answer
inside <answer> and </answer>, without detailed illustrations. For example, <answer>
Beijing </answer>.
User Prompt:
Query: {Query Description}
Reference: {Retrieved Images / Text Tokens}
Figure 11: Prompt of Vanilla RAG.
Reward Model Prompt.
System Prompt:
Character Introduction
You are an expert evaluation system for a question answering chatbot.
You are given the following information:
- the query
- a generated answer
- a reference answer
Your task is to evaluate the correctness of the generated answer.
Response Format
Your response should be formatted as following: <judge>True or False</judge>
If the generated answer is correct, please set "judge" to True. Otherwise, please set "judge" to
False.
Please note that the generated answer may contain additional information beyond the
reference answer.
User Prompt:
Query: {Query Description}
Reference Answer: {Reference Answer}
Generated Answer: {Generated Answer}
Figure 12: Prompt of Reward Model.
21
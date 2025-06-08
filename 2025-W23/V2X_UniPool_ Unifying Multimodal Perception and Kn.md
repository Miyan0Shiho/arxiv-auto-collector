# V2X-UniPool: Unifying Multimodal Perception and Knowledge Reasoning for Autonomous Driving

**Authors**: Xuewen Luo, Fengze Yang, Fan Ding, Xiangbo Gao, Shuo Xing, Yang Zhou, Zhengzhong Tu, Chenxi Liu

**Published**: 2025-06-03 08:00:57

**PDF URL**: [http://arxiv.org/pdf/2506.02580v1](http://arxiv.org/pdf/2506.02580v1)

## Abstract
Knowledge-driven autonomous driving systems(ADs) offer powerful reasoning
capabilities, but face two critical challenges: limited perception due to the
short-sightedness of single-vehicle sensors, and hallucination arising from the
lack of real-time environmental grounding. To address these issues, this paper
introduces V2X-UniPool, a unified framework that integrates multimodal
Vehicle-to-Everything (V2X) data into a time-indexed and language-based
knowledge pool. By leveraging a dual-query Retrieval-Augmented Generation (RAG)
mechanism, which enables retrieval of both static and dynamic knowledge, our
system enables ADs to perform accurate, temporally consistent reasoning over
both static environment and dynamic traffic context. Experiments on a
real-world cooperative driving dataset demonstrate that V2X-UniPool
significantly enhances motion planning accuracy and reasoning capability.
Remarkably, it enables even zero-shot vehicle-side models to achieve
state-of-the-art performance by leveraging V2X-UniPool, while simultaneously
reducing transmission cost by over 99.9\% compared to prior V2X methods.

## Full Text


<!-- PDF content starts -->

V2X-UniPool: Unifying Multimodal Perception and
Knowledge Reasoning for Autonomous Driving
Xuewen Luo1, Fengze Yang2∗, Fan Ding1∗,
Xiangbo Gao3, Shuo Xing3, Yang Zhou3, Zhengzhong Tu3, Chenxi Liu2†
1Monash University
2University of Utah3Texas A&M University
Abstract
Knowledge-driven autonomous driving systems(ADs) offer powerful reasoning
capabilities, but face two critical challenges: limited perception due to the short-
sightedness of single-vehicle sensors, and hallucination arising from the lack of real-
time environmental grounding. To address these issues, this paper introduces V2X-
UniPool, a unified framework that integrates multimodal Vehicle-to-Everything
(V2X) data into a time-indexed and language-based knowledge pool. By leveraging
a dual-query Retrieval-Augmented Generation (RAG) mechanism, which enables
retrieval of both static and dynamic knowledge, our system enables ADs to perform
accurate, temporally consistent reasoning over both static environment and dynamic
traffic context. Experiments on a real-world cooperative driving dataset demonstrate
that V2X-UniPool significantly enhances motion planning accuracy and reasoning
capability. Remarkably, it enables even zero-shot vehicle-side models to achieve
state-of-the-art performance by leveraging V2X-UniPool, while simultaneously
reducing transmission cost by over 99.9% compared to prior V2X methods.
1 Introduction
Cooperatively utilizing both ego-vehicle and infrastructure sensor data via Vehicle-to-Everything
(V2X) communication has emerged as a promising approach to enhance perception and decision-
making in autonomous driving systems (ADs) [ 15,36]. V2X communication enables the fusion of
sensor data from multiple infrastructure sources, including roadside cameras, LiDARs, and traffic
signals, thus extending the perception range and mitigating the occlusion limitations inherent in single
vehicle sensing [ 42]. Moreover, with the development of ADs, the field is witnessing a paradigm
shift from traditional data-driven pipelines toward knowledge-driven reasoning frameworks [ 35]. In
this transition, Large Language Models (LLMs) are attracting growing attention due to their powerful
contextual understanding and generalization abilities [ 30]. Unlike conventional models that rely on
fixed labels or supervised tasks, LLMs can synthesize diverse inputs into coherent, interpretable
knowledge, offering a new level of flexibility and intelligence for real-world autonomous decision-
making [42, 11].
Despite their promising capabilities, LLMs face notable limitations when applied to autonomous
driving contexts. First, these models are inherently designed for natural language processing and thus
lack native compatibility with the multimodal sensor inputs prevalent in V2X datasets [ 26]. This
modality gap restricts their ability to directly interpret raw environmental data without intermediate
translation layers. Second, they are prone to hallucination, generatingg inaccurate or fabricated
information[ 13] duee to the lack ofstructured real-time grounding information. This can lead to
∗Fan Ding and Fengze Yang contributed equally as co-second authors.
†Corresponding Author
Preprint. Under review.arXiv:2506.02580v1  [cs.AI]  3 Jun 2025

false perceptions (e.g., misidentifying obstacles or imagining non-existent events) and inconsistent
decisions [ 2]. These issues are further exacerbated in multi-turn interactions, where limited context
memory may cause the model to forget or contradict earlier observations.
To address these limitations, we propose V2X-UniPool, an end-to-end framework that transforms
multimodal V2X data into a language-based knowledge and store in a SQL database. This pool
encodes both static and dynamic elements within temporally organized windows, enabling temporally
consistent reasoning. By leveraging a Retrieval-Augmented Generation (RAG) mechanism, the
system allows LLMs to proactively query and integrate relevant context on demand. This design
bridges the modality gap and reduces hallucinations by grounding the model’s decisions in structured,
real-time environmental information.
Our contributions in this work are summarized as follows:
•We propose V2X-UniPool, the first end-to-end framework that unifies multimodal data fusion,
knowledge pool construction, and RAG reasoning into a single pipeline for knowledge-driven ADs.
Experimental results on real-world DAIR-V2X datasets demonstrate that our framework achieves
state-of-the-art planning performance and lowest transmission cost even in zero-shot vehicle-side
model.
•A time indexed knowledge pool is constructed that unifies heterogeneous infrastructure-side sensors’
records including LiDAR, images, and structured data, into a unified language representation. By
organizing information into static and dynamic pools across time windows, the pool supports
efficient storage, real-time retrieval, and consistent reasoning over both persistent construction and
evolving traffic context and states.
• This framework incorporates a proactive RAG mechanism that enables LLMs to query the knowl-
edge pool for scene information. This approach bridges the modality gap and mitigates hallu-
cinations by grounding reasoning and decision-making in structured, real-time environmental
data.
2 Related Work
2.1 V2X Collaborative Perception
Collaborative perception plays a vital role in enhancing the safety, robustness, and adaptability of
ADs, especially as traffic scenarios become increasingly complex and dynamic [ 25,12]. Traditional
independent perception, which relies solely on onboard sensors and intra-vehicle fusion, often suffers
from limited field of view and occlusions, making it difficult to ensure reliable situation awareness in
dense or unstructured environments [ 21]. In contrast, cooperative perception—enabled by Vehicle-to-
Everything (V2X) communication—facilitates the sharing of sensory information among vehicles and
infrastructure, leading to a more comprehensive and holistic understanding of the traffic environment
[12,34,17]. This approach has proven particularly valuable in high-occlusion scenarios, where
external perspectives can compensate for blind spots and improve safety-critical decision-making
[23, 31, 18].
V2X collaborative perception methods are commonly divided into three categories: early fusion,
intermediate fusion, and late fusion. Early fusion aggregates raw sensor data across agents before any
local processing, offering fine-grained cooperation but incurring high bandwidth and strict temporal
synchronization requirements [ 3,1]. Intermediate fusion shares encoded features among agents to
achieve a balance between communication cost and accuracy, and has thus become a widely adopted
paradigm [ 17,28,33,27]. However, it faces a major challenge in heterogeneous environments [ 9,14].
In contrast, late fusion, especially when implemented through LLM-based reasoning, can better
accommodate agent heterogeneity by deferring integration to a high-level semantic space [4, 8].
In this work, by treating shared outputs as language-level abstractions rather than previous data-fusion
methods that require transmitting raw data or high-dimensional features, our approach transforms
infrastructure-side outputs into compact, symbolic text fields stored in a SQL-based knowledge pool.
This not only significantly reduces communication cost, but also supports efficient, interpretable
reasoning under realistic bandwidth constraints.
2

2.2 LLM-Empowered Autonomous Driving
The rapid advancement of LLMs is reshaping the field of ADs, enabling a shift from traditional,
limited situation-aware pipelines to more holistic and knowledge-driven paradigms. By leveraging
their ability to understand complex driving scenes, LLMs are increasingly regarded as foundational
components in modern ADs architectures [ 43,19]. Their powerful contextual reasoning capabili-
ties allow ADs to interpret intricate traffic scenarios and support more informed decision-making.
Recent studies have demonstrated that LLMs not only enhance perception but also contribute to
downstream tasks such as trajectory prediction and motion planning, thereby improving the overall
safety, adaptability, and intelligence of AD systems [35, 10].
Building on this foundation, recent work has shown the potential of LLMs to support end-to-
end system from initial sensing and understanding to planning and control [ 5]. However, their
reasoning capacity is often limited by static pre-training, which restricts real-time adaptability and
environmental grounding [ 16]. To address this, the RAG paradigm has emerged as a promising
enhancement, allowing models to proactively query external databases for up-to-date and context-
specific knowledge [ 7]. Systems such as RAG-Driver [ 41] and RAG-Guided [ 40] demonstrate how
integrating RAG can reduce hallucinations, improve decision consistency, and expand the effective
knowledge scope of LLMs in autonomous driving contexts [20].
In contrast to prior LLM-based ADs that suffer from hallucinations and lack environmental grounding,
our framework anchors reasoning in a structured knowledge pool, providing temporally aligned,
interpretable, and query-efficient context. Furthermore, unlike generic RAG-based methods, V2X-
UniPool enables precise and proactive retrieval of spatial-temporal knowledge tailored to driving
needs, ensuring both consistency and real-time responsiveness.
3 Methodology
3.1 Framework Overview
Figure 1: Framework of V2X-UniPool
To empower knowledge-driven AD systems with robust situational awareness and accurate reasoning,
we introduce V2X-UniPool, a unified multimodal framework deployed on the infrastructure side,
that bridges the gap between real-world infrastructure raw sensor data and structured language
information. The framework operates by continuously aggregating heterogeneous multimodal inputs
into a temporally structured V2X Knowledge Pool on the infrastructure-side SQL database, while the
onboard knowledge-driven AD system proactively interacts with this pool via a RAG mechanism. At
each decision-making step, the vehicle-side model encodes the current perception and intent into a
latent query to retrieve both static context and dynamic states from the knowledge pool. This enables
both reactive responses and anticipatory planning, supporting real-time, grounded, and adaptive
reasoning under complex traffic conditions.
The core of the framework is a temporally indexed V2X Knowledge Pool. At its essence, the
V2X Knowledge Pool serves as a comprehensive repository for environmental scene understanding.
Environmental scene understanding refers to the perception system’s ability to extract, recognize,
and interpret features of all surrounding elements relevant to driving [ 22]. In this paper, the V2X
Knowledge Pool formalizes this scene understanding as a integration of language-based text of the
3

surrounding world, including both static elements (road geometry, traffic signs, map features) and
dynamic elements (traffic context and states). Therefore, the V2X Knowledge Pool is organized
into a Static Pool and a Dynamic Pool . The Static Pool stores static elements, while the Dynamic
Pool stores dynamic elements and it is further divided into a a High-Frequency Update Sub-Pool
(updating at 10 Hz) for rapidly changing states and a low-Frequency Update Sub-Pool (updating
at 1 Hz) for gradually evolving contextual information. The storage and retrieval of the V2X
Knowledge Pool are supported by a SQL database, which enables efficient spatiotemporal indexing
and low-latency access to information.
A RAG reasoning system is integrated with a structured V2X knowledge database to enable situation-
awareness and interpretable driving decision-making. This system employs a bridge encoder to
convert vehicle-side perception and intent into a latent query vector, which is used to retrieve
semantically aligned entries within the V2X knowledge database: static semantics, high-frequency
real-time traffic states, and low-frequency contextual updates. The retrieved knowledge is fused into
a language textual representation and combined with the vehicle’s local sensory input to support joint
reasoning. The fused context is then processed by the vehicle-side model to generate final driving
decisions.
3.2 V2X Knowledge Pool Construction
Figure 2: Overview of the V2X Knowledge Pool Construction
Infrastructure-Side Data Processing To construct a temporally aligned and semantically struc-
tured knowledge representation, the infrastructure-side raw sensor data are first transformed into a
language-based format. The data are categorized into two modalities: Unstructured Data , including
camera images and LiDAR point clouds; and Structured Data , comprising HD maps, traffic states,
traffic density, and other infrastructure-sourced signals. This focus leverages the global field-of-view
and fixed positioning of infrastructure sensors, offering more comprehensive and stable environmental
coverage than the ego vehicle’s limited and dynamic perception.
All collected data are first temporally synchronized using precise timestamps and spatially calibrated
within a unified coordinate frame to ensure cross-modality alignment. The processing pipeline is
divided according to data modality and preprocessing requirements:
ForUnstructured Data , camera and LiDAR inputs from roadside units are converted into inter-
pretable semantic representations. High-resolution bird’s-eye-view (BEV) images are denoised,
standardized, and normalized before being processed by the vision-language model GPT-4o (2024).
Carefully designed prompts guide the model to extract scene-level semantics, outputting structured
descriptions that include a reason field for interpretability and a prediction field to infer short-
term dynamics. This reasoning-aware design improves both transparency and utility [ 29,24,2]. In
parallel, LiDAR point clouds are geometrically filtered to remove noise and normalized for consistent
density and frame alignment. Combined with image semantics, these 3D spatial cues enhance the
detection and localization of road users and infrastructure elements, yielding a spatially coherent
representation of the traffic scene.
4

ForStructured Data , inputs are cleaned, temporally synchronized, and spatially aligned. Relevant
fields are parsed, missing values are interpolated or replaced with defaults, and numerical fields are
standardized. These records are then matched with unstructured data by timestamp and location,
enabling unified multimodal integration.
The resulting dataset comprises temporally aligned, semantically normalized heterogeneous data,
where both structured and unstructured modalities are transformed into a unified language-based
representation. This forms the basis of the V2X knowledge pool, supporting interpretable and efficient
reasoning across complex traffic scenarios.
Static Pool The static pool of the V2X knowledge pool is defined as the collection of infrastructure-
level environmental semantics whose temporal dynamics are negligible within operational planning
horizons [ 14]. These elements are considered time-invariant in practice as they do not change over
short time intervals, and thus provide a stable grounding for reasoning and decision-making across
time-aligned traffic scenarios. Formally, a data element Dstaticis classified as part of the static pool if
it satisfies the constraint of temporal invariance:
∂Dstatic
∂t< ϵ, ∀t∈[t0, t0+T] (1)
where ϵis a predefined stability threshold (e.g., ϵ≪1) and Tdenotes the duration of the planning
horizon. This condition ensures that static elements exhibit negligible change throughout the interval
and can be reliably reused without frequent updates.
The static pool is typically constructed offline or during system initialization for a given region.
It is refreshed only when long-term infrastructure changes occur, such as construction updates or
map revisions. Each entry in the static pool is represented in a language-based format, allowing
downstream modules to access interpretable, causally grounded descriptions of the environment.
These descriptions include traffic signs, markings, and roadway geomatrix.
By maintaining a stable, high-level abstraction of the physical scene, the static pool serves as the
long-term memory of the reasoning system. It enables consistent grounding for dynamic perception,
supports scenario understanding, and reduces the computational burden of repeatedly processing and
extracting static semantics in real time.
Dynamic Pool The dynamic pool of the V2X knowledge pool captures temporally evolving traffic
states. Unlike the static pool that includes invariant infrastructure, the dynamic pool is continuously
updated to reflect changes in traffic participants, signal states, and environmental conditions. Formally,
a data element Ddynamic is classified as part of the dynamic pool if it violates the constraint of temporal
invariance and exhibits perceptible change within the planning horizon:
∂Ddynamic
∂t≥ϵ,∃t∈[t0, t0+T] (2)
where ϵis the minimum change rate required to be considered temporally dynamic. The threshold is
task-defined and reflects the system’s sensitivity to environmental variation. This definition ensures
that only non-stationary elements requiring timely updates are included in the dynamic pool, enabling
responsive and adaptive planning.
To balance responsiveness and semantic granularity, we divide it into two sub-pools: a high-frequency
High-Frequency Update Sub-Pool and a low-frequency low-Frequency Update Pool .
High-Frequency Update Sub-Pool contains data streams that change rapidly over short time intervals
and are critical for short-term prediction and motion planning. This includes object trajectories,
velocities, types, and traffic light statuses updated at rates of 10 Hz. Formally, a dynamic data instance
Ddynamic is assigned to the high-frequency sub-pool if it satisfies:
∂Ddynamic
∂t≥ϵhigh, f s(Ddynamic )≥10Hz (3)
where ϵhighdenotes the minimum rate of change for high-dynamic content, and fs(·)represents the
sampling frequency. This condition ensures only fast-evolving, high-resolution data are routed to this
sub-pool for real-time planning.
5

Low-Frequency Update Sub-Pool stores moderately dynamic yet semantically rich information that
evolves over slower time scales. This includes aggregated traffic density, collision alerts, construction
updates, and abnormal event, typically updated at rates around 1 Hz. Formally, a dynamic data
instance Ddynamic is assigned to the low-frequency sub-pool if it satisfies:
ϵlow≤∂Ddynamic
∂t< ϵ high, f s(Ddynamic )<1Hz (4)
where ϵlowandϵhighdefine the acceptable range of perceptual change for moderate dynamics. This
ensures mid-rate signals are preserved for semantic fusion and planning over longer horizons.
Together, these two sub-pools provide a comprehensive and temporally resolved understanding of
the current traffic environment, enabling Knowledge-driven ADs to anticipate, reason, and respond
adaptively under dynamic conditions.
SQL Database Construction To support structured and low-latency retrieval of V2X knowledge,
all processed elements are consolidated into a SQL database. This database is logically partitioned
into three sub-databases, each corresponding to a distinct data modality and temporal profile.
Static Pool Database (DB static) is composed of persistent entries Dstaticthat describe time-invariant
environmental structures. These entries are generated offline or during system initialization and
indexed by location identifiers.
High-Frequency Dynamic Database (DB HF) consists of dynamic data instances Dhighthat satisfy
high-rate update conditions. The database is optimized for sliding-window queries and supports
real-time query.
Low-Frequency Dynamic Database (DB SF) comprises update dynamic data Dslowupdated at lower
frequencies. These entries are served as mid-horizon context for scene understanding and risk
assessment.
Each entry across all three partitions is associated with a timestamp and geospatial anchor (intersection
ID or local map coordinates), and structured in a language-based text format. The complete SQL
database is represented as:
DB V2X=DB static∪ DB HF∪ DB SF (5)
3.3 LLM-V2X RAG Reasoning
We propose an LLM-V2X collaborative reasoning approach that integrates perception of the ego
vehicle with the external V2X Knowledge Pool. Unlike traditional RAG pipelines that rely on natural
language queries, our method employs a bridge encoder to directly map vehicle-side requirements to
SQL queries, enabling efficient and structured information retrieval.
Encoder-Based V2X Query Interface An encoder module is deployed alongside the infrastructure.
It takes as input the vehicle-side perception and driving intent, collectively denoted as the requirement
R, and encodes it into a latent query vector q∈Rd, where dis the semantic embedding dimension:
q= Encoder( R) (6)
The query qis used to retrieve relevant information from the V2X SQL database. Each retrieval
targets a specific temporal resolution: DB staticfor Static Pool database, DB HFfor High-Frequency
Update Sub-Pool database , and DB SFfor Low-Frequency Update Sub-Pool database. The retrieval
process is defined as:
To support multi-resolution reasoning, the latent query vector qis used to retrieve semantically
aligned records from the three temporally partitioned sub-databases:
Estatic= Search( q;DB static),
EHF= Search( q;DB HF),
ESF= Search( q;DB SF)(7)
Each retrieval operation returns a set of records E∗that are selected based on spatial-temporal
alignment and semantic relevance to the query. These context subsets are then integrated to form a
unified external knowledge representation:
E= Fuse( Estatic, E HF, E SF) (8)
6

Context Fusion and Joint Reasoning Once the external context Eis retrieved, it is fused with
the vehicle-side sensory input V, which includes the ego agent’s current perception and internal
state. A fusion module Fuse( V, E)maps both sources into a shared embedding space, allowing
cross-modal reasoning over local observations and global semantic context. The fused representation
is then processed by the vehicle planning model LLM(·), which outputs the final decision ˆP, such as
trajectory waypoints or action plan:
ˆP= LLM (Fuse( V, E)) (9)
This architecture allows the onboard system to jointly reason over vehicle-local and infrastructure-
global information in real time, enabling interpretable, situation-aware planning under complex urban
traffic scenarios.
4 Experiment
4.1 Experiment Settings
We conduct comprehensive evaluations of V2X-UniPool on the DAIR-V2X-Seq dataset [ 38], a large-
scale real-world V2X benchmark featuring both vehicle-side and infrastructure-side sensors. The
dataset comprises sequential perception and trajectory forecasting subsets. The sequential perception
dataset includes over 15,000 frames across 95 scenarios, collected at 10 Hz from vehicle and roadside
sensors. The trajectory forecasting dataset is significantly larger, with 210,000 scenarios from 28
intersections, including 50,000 cooperative-view, 80,000 ego-view, and 80,000 infrastructure-view
cases. Each scenario is supplemented with 10-second trajectories, high-definition vector maps,
and real-time traffic light signals. This comprehensive setup enables fine-grained exploration of
cooperative perception and planning, allowing assessment of infrastructure-enhanced reasoning in
diverse urban traffic conditions.
We argue that the true utility of any V2X framework should be assessed in the context of end-to-end
autonomous driving tasks. This includes not only perception accuracy or communication latency, but
also how well the system supports downstream planning and control decisions. To this end, we adopt
representative and promising language models as vehicle-side base models, ensuring that the overall
system reflects real-world driving decision-making pipelines.
All models follow the OpenEMMA-style planning formulation [ 32], where each input consists of
front-view images and a history of ego vehicle speeds and curvatures, and the output is a prediction of
future speed and curvature vectors. This design reflects real-world driving requirements and supports
precise evaluation of planning performance. For the experiment, we construct over 10,000 ego-view
scenarios from DAIR-V2X-Seq, each spanning 10 seconds and formatted as structured planning
prompts with historical motion states and aligned future trajectory labels. During inference, all
vehicle-side models receive the same V2X-UniPool RAG retrieval results, ensuring that observed
performance differences are solely attributed to the models’ reasoning ability over external semantic
context.
4.2 Experiment Results
Table 1: Planning Evaluation Results. V2X-UniPool integrates external knowledge via structured
RAG reasoning. The vehicle-side model is based on Qwen-3-8B.
Method L2 Error (m) ↓ Collision Rate (%) ↓ Transmission Cost ↓
2.5s 3.5s 4.5s Avg. 2.5s 3.5s 4.5s Avg.
V2VNet [28] 2.31 3.29 4.31 3.30 0.00 1.03 1.47 0.83 8.19×107
CooperNaut [6] 3.83 5.26 6.69 5.26 0.59 1.92 1.63 1.38 8.19×107
UniV2X - Vanilla [39] 2.21 3.31 4.46 3.33 0.15 0.89 2.67 1.24 8.19×107
UniV2X [39] 2.60 3.44 4.36 3.00 0.00 0.74 0.49 0.41 8.09×105
V2X-VLM [37] 1.21 1.21 1.23 1.22 0.01 0.01 0.01 0.01 1.24×107
V2X-UniPool 1.04 1.54 2.10 1.60 0.01 0.01 0.02 0.01 2.74×103
Table 1 reports the performance of our proposed V2X-UniPool framework against representative
baselines across three key metrics: L2 planning error, collision rate, and transmission cost. Compared
7

to traditional cooperative perception models like V2VNet and CooperNaut, V2X-UniPool reduces the
average L2 error from 3.30m and 5.26m to 1.60m . In terms of safety, our model maintains a minimal
collision rate of 0.01% , substantially outperforming UniV2X-Vanilla (1.24%) and CooperNaut
(1.38%).
A critical advantage of V2X-UniPool lies in its efficient knowledge representation and communication
design. Unlike V2X-VLM, V2X-UniPool queries a structured SQL knowledge database and transmits
only compact, symbolic JSON fields representing key environmental semantics, including lane
markings, traffic signs, weather, traffic conditions, and objects. Each query result is serialized in
UTF-8 with minimal overhead, yielding a total transmission cost of only 2.74×103bytes . This
is over 4500×more efficient than the 1.24×107bytes required by V2X-VLM. The reduction is
achieved without compromising planning accuracy or safety, validating the effectiveness of our
language-level abstraction and structured retrieval mechanism under realistic bandwidth constraints.
Overall, V2X-UniPool achieves the best trade-off among accuracy, safety, and communication
efficiency. By grounding planning decisions in structured, retrieved knowledge and minimizing
bandwidth usage, our approach demonstrates a practical and scalable solution for real-world V2X-
based autonomous driving systems.
4.3 Ablation Study
To evaluate the effectiveness and general applicability of the proposed V2X-UniPool framework,
we conduct a comprehensive ablation study across a diverse set of vehicle-side language models.
Specifically, we compare the planning performance of each model with and without the integration of
V2X-UniPool. The evaluation spans both proprietary and open-source paradigms, including: GPT-4o
(2024) , a strong multimodal language model with state-of-the-art visual reasoning; GPT-4.1 Mini ,
a lightweight version optimized for real-time inference; Gemini-2.0 , an advanced vision-language
model from Google DeepMind; and Qwen-3-8B , a new high-performance small-scale open-source
VLM developed for general reasoning.
By applying the same front-view image and motion history as input and comparing the planning
outputs with and without external V2X knowledge, we isolate the effect of structured knowledge
integration. This setup enables a direct comparison of how each model benefits from V2X-UniPool’s
semantic grounding. The results show that incorporating V2X-UniPool improves not only accuracy
(measured by L2 error), but also trajectory smoothness and comfort (reflected in Comfort Scores),
regardless of the underlying language model architecture. These findings validate the effectiveness
andgenerality of V2X-UniPool as a plug-in enhancement to a wide range of vehicle-side planning
models.
We omit the collision rate comparison table, as all vehicle-side models, including those without V2X-
UniPool, already achieve extremely low collision rates (below 0.03) across all horizons. Given the
rarity of collision events, outcomes are highly sensitive to stochastic variations and model fluctuations,
making this metric statistically unstable for fine-grained ablation. Instead, we focus on L2 Error
(ADE) in Table 2, which provides a more consistent and discriminative measure of planning quality.
Table 2: Ablation study on L2 Error (m) of different
vehicle-side models with and without V2X-UniPool.
Results are reported at 2.5s, 3.5s, and 4.5s horizons.
Model w/o V2X-UniPool w/ V2X-UniPool
2.5s 3.5s 4.5s 2.5s ↓3.5s↓4.5s↓
GPT-4o (2024) 1.25 1.93 2.74 1.18 1.81 2.56
GPT-4.1 Mini 1.45 2.21 3.11 1.41 2.18 3.10
Gemini-2.0 1.37 2.18 3.17 1.29 2.01 2.92
Qwen-3-8b 1.37 2.05 2.75 1.04 1.54 2.10The results in Table 2 show that integrating
V2X-UniPool significantly improves trajec-
tory prediction accuracy across all evalu-
ated vehicle-side models. The average L2
error is consistently reduced at all future
horizons (2.5s, 3.5s, 4.5s), demonstrating
the value of structured external knowledge
in refining long-horizon planning. GPT-
4obenefits notably, with L2 error reduced
from 1.25 to 1.18 at 2.5s and from 2.74 to
2.56 at 4.5s. Qwen-3-8B shows the most
dramatic improvement, especially at long
range, reducing its 4.5s error from 2.75 to 2.10, a relative improvement of over 23%. These gains
suggest that even high-performing models can enhance their planning accuracy when grounded in
V2X-UniPool’s structured semantic context. The consistent improvements across diverse model
architectures confirm the generality and effectiveness of our method.
8

To evaluate trajectory smoothness and driving comfort, we introduce a differentiable Comfort Score ,
which penalizes abrupt acceleration, jerk, and yaw rate fluctuations over the predicted trajectory. The
score is computed as:
ComfortScore = 1−tanh 
α·¯|¨|v+β·¯|...v|+γ·¯|˙|ω
(10)
Here, ¨vdenotes longitudinal acceleration,...vis jerk (the rate of change of acceleration), and ˙ωis the
derivative of yaw rate (indicating lateral instability). The average is computed over nfuture frames.
The coefficients α,β, andγcontrol the contribution of each term, set to 1.0, 2.0, and 0.5 respectively.
The resulting score lies within [0,1], where higher values indicate smoother and more comfortable
planned trajectories.
Table 3: Ablation study on trajectory Comfort Score at
different time horizons, with and without V2X-UniPool.
Higher scores indicate smoother and more comfortable
plans.
Model w/o V2X-UniPool w/ V2X-UniPool
2.5s 3.5s 4.5s 2.5s ↑3.5s↑4.5s↑
GPT-4o (2024) 0.44 0.46 0.48 0.53 0.54 0.55
GPT-4.1 Mini 0.53 0.55 0.57 0.54 0.56 0.57
Gemini-2.0 0.23 0.25 0.26 0.26 0.28 0.29
Qwen-3-8B 0.69 0.70 0.70 0.72 0.74 0.74The ablation results in Table 3 show
that integrating V2X-UniPool consistently
improves trajectory smoothness across
all tested vehicle-side models. Notably,
GPT-4o shows the most significant gain,
with Comfort Scores increasing from
0.44/0.46/0.48 to 0.53/0.54/0.55 across the
2.5s–4.5s horizons, reflecting a substan-
tial improvement in trajectory stability and
comfort. Gemini-2.0 , despite having the
lowest baseline scores, benefits clearly
from V2X-UniPool (+0.03 to +0.04), in-
dicating that external structured knowledge
can complement weaker motion reasoning capabilities. Qwen-3-8B , which already performs well,
further improves to a high comfort level of 0.74. These results confirm that V2X-UniPool enhances
both accuracy and smoothness of planned trajectories across a diverse range of language model
architectures.
5 Conclusion
In this paper, we propose V2X-UniPool, a unified end-to-end framework that integrates multimodal
V2X data into a structured, language-based knowledge pool to support knowledge-driven autonomous
driving. By constructing a temporally indexed SQL-based database composed of static and dynamic
pools, our system enables efficient and interpretable RAG reasoning for planning tasks. We further
design a bridge encoder to map vehicle-side perception into structured semantic queries for scene
understanding, enabling real-time, grounded decision-making. Experimental results on the DAIR-
V2X-Seq benchmark show that V2X-UniPool significantly outperforms both traditional V2X methods
and recent LLM-based approaches. In particular, it achieves the lowest planning error (1.85m average
L2) and the lowest collision rate (0.01%) across all time horizons, while maintaining an extremely low
transmission cost ( 2.74×103), validating the efficiency, accuracy, and scalability of our framework.
The consistent performance across diverse ablation settings further demonstrates the generalizability
of our approach.
Although V2X-UniPool achieves promising results, several directions remain for future work. First,
while our current framework demonstrates generalizability across multiple language models’ back-
bones, further gains may be achieved by customizing and fine-tuning vehicle-side language models
for V2X planning tasks. Specifically, we plan to explore the alignment between lightweight language
models and our structured V2X-UniPool flow pipeline, enabling efficient onboard reasoning under
computational constraints. Additionally, scaling the framework to incorporate larger language models
with task-specific adaptation techniques (e.g., LoRA or instruction tuning) may further enhance plan-
ning quality, especially in rare or long-tail traffic scenarios. We also intend to investigate real-world
deployment with continuous data streaming and feedback-driven updates to validate the robustness
and practicality of our system in complex, open-world driving environments.
References
[1]Eduardo Arnold, Mehrdad Dianati, Robert de Temple, and Saber Fallah. Cooperative perception
for 3d object detection in driving scenarios using infrastructure sensors. IEEE Transactions on
Intelligent Transportation Systems , 23(3):1852–1864, 2020.
9

[2]Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng
Shou. Hallucination of multimodal large language models: A survey. arXiv preprint
arXiv:2404.18930 , 2024.
[3]Qi Chen, Sihai Tang, Qing Yang, and Song Fu. Cooper: Cooperative perception for connected
autonomous vehicles based on 3d point clouds. In 2019 IEEE 39th International Conference on
Distributed Computing Systems (ICDCS) , pages 514–524. IEEE, 2019.
[4]Hsu-kuang Chiu, Ryo Hachiuma, Chien-Yi Wang, Stephen F Smith, Yu-Chiang Frank Wang,
and Min-Hung Chen. V2v-llm: Vehicle-to-vehicle cooperative autonomous driving with multi-
modal large language models. arXiv preprint arXiv:2502.09980 , 2025.
[5]Can Cui et al. A survey on multimodal large language models for autonomous driving. In
Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision , pages
958–979, 2024.
[6]Jiaxun Cui, Hang Qiu, Dian Chen, Peter Stone, and Yuke Zhu. Coopernaut: End-to-end
driving with cooperative perception for networked vehicles. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pages 17252–17262, 2022.
[7]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua,
and Qing Li. A survey on rag meeting llms: Towards retrieval-augmented large language
models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and
Data Mining , pages 6491–6501, 2024.
[8]Xiangbo Gao, Yuheng Wu, Rujia Wang, Chenxi Liu, Yang Zhou, and Zhengzhong Tu. Langcoop:
Collaborative driving with language. arXiv preprint arXiv:2504.13406 , 2025.
[9]Xiangbo Gao, Runsheng Xu, Jiachen Li, Ziran Wang, Zhiwen Fan, and Zhengzhong Tu. Stamp:
Scalable task and model-agnostic collaborative perception. arXiv preprint arXiv:2501.18616 ,
2025.
[10] Xianda Guo, Ruijun Zhang, Yiqun Duan, Yuhang He, Chenming Zhang, Shuai Liu, and Long
Chen. Drivemllm: A benchmark for spatial understanding with multimodal large language
models in autonomous driving. arXiv preprint arXiv:2411.13112 , 2024.
[11] Muhammad Usman Hadi, Qasem Al Tashi, Abbas Shah, Rizwan Qureshi, Amgad Muneer,
Muhammad Irfan, Anas Zafar, Muhammad Bilal Shaikh, Naveed Akhtar, Jia Wu, et al. Large
language models: a comprehensive survey of its applications, challenges, limitations, and future
prospects. Authorea Preprints , 2024.
[12] Yushan Han, Hui Zhang, Huifang Li, Yi Jin, Congyan Lang, and Yidong Li. Collabora-
tive perception in autonomous driving: Methods, datasets, and challenges. IEEE Intelligent
Transportation Systems Magazine , 2023.
[13] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qiang-
long Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large
language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on
Information Systems , 43(2):1–55, 2025.
[14] Tao Huang, Jianan Liu, Xi Zhou, Dinh C Nguyen, Mostafa Rahimi Azghadi, Yuxuan Xia,
Qing-Long Han, and Sumei Sun. V2x cooperative perception for autonomous driving: Recent
advances and challenges. arXiv preprint arXiv:2310.03525 , 2023.
[15] Henry Alexander Ignatious, Manzoor Khan, et al. An overview of sensors in autonomous
vehicles. Procedia Computer Science , 198:736–741, 2022.
[16] Xin Li, Yeqi Bai, Pinlong Cai, Licheng Wen, Daocheng Fu, Bo Zhang, Xuemeng Yang, Xinyu
Cai, Tao Ma, Jianfei Guo, et al. Towards knowledge-driven autonomous driving. arXiv preprint
arXiv:2312.04316 , 2023.
[17] Si Liu, Chen Gao, Yuan Chen, Xingyu Peng, Xianghao Kong, Kun Wang, Runsheng Xu, Wentao
Jiang, Hao Xiang, Jiaqi Ma, et al. Towards vehicle-to-everything autonomous driving: A survey
on collaborative perception. arXiv preprint arXiv:2308.16714 , 2023.
10

[18] Jia Quan Loh, Xuewen Luo, Fan Ding, Hwa Hui Tew, Junn Yong Loo, Ze Yang Ding, Susilawati
Susilawati, and Chee Pin Tan. Cross-domain transfer learning using attention latent features for
multi-agent trajectory prediction, 2024. URL https://arxiv.org/abs/2411.06087 .
[19] Xuewen Luo, Fan Ding, Yinsheng Song, Xiaofeng Zhang, and Junnyong Loo. Pkrd-cot: A
unified chain-of-thought prompting for multi-modal large language models in autonomous
driving. arXiv preprint arXiv:2412.02025 , 2024.
[20] Xuewen Luo, Chenxi Liu, Fan Ding, Fengze Yang, Yang Zhou, Junnyong Loo, and Hwa Hui
Tew. Senserag: Constructing environmental knowledge bases with proactive querying for
llm-based autonomous driving. In Proceedings of the Winter Conference on Applications of
Computer Vision , pages 989–996, 2025.
[21] Lili Miao, Shang-Fu Chen, Yu-Ling Hsu, and Kai-Lung Hua. How does c-v2x help autonomous
driving to avoid accidents? Sensors , 22(2):686, 2022.
[22] Khan Muhammad, Tanveer Hussain, Hayat Ullah, Javier Del Ser, Mahdi Rezaei, Neeraj Kumar,
Mohammad Hijji, Paolo Bellavista, and Victor Hugo C de Albuquerque. Vision-based semantic
segmentation in scene understanding for autonomous driving: Recent achievements, challenges,
and outlooks. IEEE Transactions on Intelligent Transportation Systems , 23(12):22694–22715,
2022.
[23] Vandana Narri, Amr Alanwar, Jonas Mårtensson, Christoffer Norén, Laura Dal Col, and
Karl Henrik Johansson. Set-membership estimation in shared situational awareness for auto-
mated vehicles in occluded scenarios. In 2021 IEEE Intelligent Vehicles Symposium (IV) , pages
385–392. IEEE, 2021.
[24] Nazneen Fatema Rajani, Bryan McCann, Caiming Xiong, and Richard Socher. Explain yourself!
leveraging language models for commonsense reasoning. arXiv preprint arXiv:1906.02361 ,
2019.
[25] Shunli Ren, Siheng Chen, and Wenjun Zhang. Collaborative perception for autonomous driving:
Current status and future trend. In Proceedings of 2021 5th Chinese Conference on Swarm
Intelligence and Cooperative Control , pages 682–692. Springer, 2022.
[26] Shiva Sreeram, Tsun-Hsuan Wang, Alaa Maalouf, Guy Rosman, Sertac Karaman, and Daniela
Rus. Probing multimodal llms as world models for driving. arXiv preprint arXiv:2405.05956 ,
2024.
[27] Rujia Wang, Xiangbo Gao, Hao Xiang, Runsheng Xu, and Zhengzhong Tu. Cocmt:
Communication-efficient cross-modal transformer for collaborative perception. arXiv preprint
arXiv:2503.13504 , 2025.
[28] Tsun-Hsuan Wang, Sivabalan Manivasagam, Ming Liang, Bin Yang, Wenyuan Zeng, and Raquel
Urtasun. V2vnet: Vehicle-to-vehicle communication for joint perception and prediction. In
Computer vision–ECCV 2020: 16th European conference, Glasgow, UK, August 23–28, 2020,
proceedings, part II 16 , pages 605–621. Springer, 2020.
[29] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le,
Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models.
Advances in neural information processing systems , 35:24824–24837, 2022.
[30] Licheng Wen, Daocheng Fu, Xin Li, Xinyu Cai, Tao Ma, Pinlong Cai, Min Dou, Botian Shi,
Liang He, and Yu Qiao. Dilu: A knowledge-driven approach to autonomous driving with large
language models. arXiv preprint arXiv:2309.16292 , 2023.
[31] Zhu Xiao, Jinmei Shu, Hongbo Jiang, Geyong Min, Hongyang Chen, and Zhu Han. Overcoming
occlusions: Perception task-oriented information sharing in connected and autonomous vehicles.
IEEE Network , 37(4):224–229, 2023.
[32] Shuo Xing, Chengyuan Qian, Yuping Wang, Hongyuan Hua, Kexin Tian, Yang Zhou, and
Zhengzhong Tu. Openemma: Open-source multimodal model for end-to-end autonomous
driving. In Proceedings of the Winter Conference on Applications of Computer Vision , pages
1001–1009, 2025.
11

[33] Runsheng Xu, Hao Xiang, Zhengzhong Tu, Xin Xia, Ming-Hsuan Yang, and Jiaqi Ma. V2x-vit:
Vehicle-to-everything cooperative perception with vision transformer. In Computer Vision –
ECCV 2022 , 2022.
[34] Xun Yang, Yunyang Shi, Jiping Xing, and Zhiyuan Liu. Autonomous driving under v2x
environment: state-of-the-art survey and challenges. Intelligent Transportation Infrastructure ,
1:liac020, 2022.
[35] Zhenjie Yang, Xiaosong Jia, Hongyang Li, and Junchi Yan. Llm4drive: A survey of large
language models for autonomous driving. In NeurIPS 2024 Workshop on Open-World Agents ,
2023.
[36] Takahito Yoshizawa, Dave Singelée, Jan Tobias Muehlberg, Stephane Delbruel, Amir Taherko-
rdi, Danny Hughes, and Bart Preneel. A survey of security and privacy issues in v2x communi-
cation systems. ACM Computing Surveys , 55(9):1–36, 2023.
[37] Junwei You, Haotian Shi, Zhuoyu Jiang, Zilin Huang, Rui Gan, Keshu Wu, Xi Cheng, Xiaopeng
Li, and Bin Ran. V2x-vlm: End-to-end v2x cooperative autonomous driving through large
vision-language models. arXiv preprint arXiv:2408.09251 , 2024.
[38] Haibao Yu, Wenxian Yang, Hongzhi Ruan, Zhenwei Yang, Yingjuan Tang, Xu Gao, Xin
Hao, Yifeng Shi, Yifeng Pan, Ning Sun, et al. V2x-seq: A large-scale sequential dataset for
vehicle-infrastructure cooperative perception and forecasting. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pages 5486–5495, 2023.
[39] Haibao Yu, Wenxian Yang, Jiaru Zhong, Zhenwei Yang, Siqi Fan, Ping Luo, and Zaiqing
Nie. End-to-end autonomous driving through v2x cooperation. In Proceedings of the AAAI
Conference on Artificial Intelligence , volume 39, pages 9598–9606, 2025.
[40] Jun Yu, Yunxiang Zhang, Zerui Zhang, Zhao Yang, Gongpeng Zhao, Fengzhao Sun, Fanrui
Zhang, Qingsong Liu, Jianqing Sun, Jiaen Liang, et al. Rag-guided large language models for
visual spatial description with adaptive hallucination corrector. In Proceedings of the 32nd ACM
International Conference on Multimedia , pages 11407–11413, 2024.
[41] Jianhao Yuan, Shuyang Sun, Daniel Omeiza, Bo Zhao, Paul Newman, Lars Kunze, and Matthew
Gadd. Rag-driver: Generalisable driving explanations with retrieval-augmented in-context
learning in multi-modal large language model. arXiv preprint arXiv:2402.10828 , 2024.
[42] Syed Adnan Yusuf, Arshad Khan, and Riad Souissi. Vehicle-to-everything (v2x) in the au-
tonomous vehicles domain–a technical review of communication, sensor, and ai technologies
for road user safety. Transportation Research Interdisciplinary Perspectives , 23:100980, 2024.
[43] Yuxuan Zhu, Shiyi Wang, Wenqing Zhong, Nianchen Shen, Yunqi Li, Siqi Wang, Zhiheng Li,
Cathy Wu, Zhengbing He, and Li Li. Will large language models be a panacea to autonomous
driving? arXiv preprint arXiv:2409.14165 , 2024.
12
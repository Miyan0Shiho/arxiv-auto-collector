# Fail-RAG : A Retrieval Augmented Generation Informed Framework for Robot Failure Identification

**Authors**: Ameya Salvi, Jie Hu

**Published**: 2026-06-17 21:02:47

**PDF URL**: [https://arxiv.org/pdf/2606.19598v1](https://arxiv.org/pdf/2606.19598v1)

## Abstract
Industry automation is witnessing an evolution in robotics driven by both technological breakthroughs and societal changes: progress towards generalist robots, embodied and physical artificial intelligence (AI), and increasing labor shortage in manufacturing.An intelligent autonomous robot needs to not only act according to planned motions but also react to any unexpected events. In this study, we focus on such unexpected events in warehouses where robots are used for material handling. Specifically, we refer to any unexpected events as failures and develop methods to detect robot operations related failures. Rule-based detection methods may break since the form of failures could change due to the dynamic nature of both environments and tasks. We propose 'Fail-RAG', a Retrieval Augmented Generation (RAG)-based failure detection framework where failure images and context information are embedded and queried against a failure database by calculating their similarities. Vision-Language Models (VLMs) are further used to analyze failures and provide details by following our instruction template. We evaluated the performance of Fail-RAG by conducting both simulation and physical experiments using fixed robot arms and a mobile manipulator for multiple tasks that are common in warehouse automation. Fail-RAG achieved 25 percentage point higher failure detection accuracy on average across five types of robot operations compared to using off-the-shelf VLMs, indicating its effectiveness for real-world failure detection.

## Full Text


<!-- PDF content starts -->

Fail-RAG : A Retrieval Augmented Generation Informed Framework
for Robot Failure Identification
Ameya Salvi1and Jie Hu1
Abstract— Industry automation is witnessing an evolution
in robotics driven by both technological breakthroughs and
societal changes: progress towards generalist robots, embodied
and physical artificial intelligence (AI), and increasing labor
shortage in manufacturing. An intelligent autonomous robot
needs to not onlyactaccording to planned motions but also
reactto any unexpected events. In this study, we focus on
such unexpected events in warehouses where robots are used
for material handling. Specifically, we refer to any unexpected
events asfailuresand develop methods to detect robot opera-
tions related failures. Rule-based detection methods may break
since the form of failures could change due to the dynamic
nature of both environments and tasks. We propose Fail-RAG, a
Retrieval Augmented Generation (RAG)-based failure detection
framework where failure images and context information are
embedded and queried against a failure database by calculating
their similarities. Vision-Language Models (VLMs) are further
used to analyze failures and provide details by following our
instruction template. We evaluated the performance of Fail-
RAG by conducting both simulation and physical experiments
using fixed robot arms and a mobile manipulator for multiple
tasks that are common in warehouse automation. Fail-RAG
achieved 25 percentage point higher failure detection accuracy
on average across five types of robot operations compared to
using off-the-shelf VLMs, indicating its effectiveness for real-
world failure detection.
I. INTRODUCTION
The focus of automation has been shifting to more com-
plex tasks and those that are natural to human but difficult
to machines, examples exist from handling large volume of
packages with great variety in logistics and warehouses[1], to
outdoor deliveries[2], assistance in hospitals[3], and house-
hold tasks[4]. While several automation processes have been
perfected over decades, the surge in inexpensive hardware
and development of cutting edge software has led to a wider
attraction and adoption of robot technologies in increasingly
challenging and unstructured environments. Conventionally,
automation operations necessitate implementation of failure
and anomaly identification frameworks for enhancing the
reliability and safety of the operation. While several methods
implementing specialized hardware and rule-based softwares
have contributed to this area, the adoption of these method
for more general purpose robots is significantly challenging
due to complexities in scaling hardware and rule-base soft-
ware. Inasmuch, vision language models (VLMs) provide a
viable solution for monitoring robot operations purely based
on camera vision information, allowing for their flexible
application in a large number of robot operations.
1Authors are with Hitachi America, Ltd., Michigan, U.S.A.,
ameya.salvi, jie.hu@hal.hitachi.com
Fig. 1. Overview of the proposed RAG-based failure detection framework.
Multiple robot processes such as (a) object manipulation, (b) robot navi-
gation, and (c) robot coordination are encoded as RAG context and used
for real-time monitoring of robot operations using multi-modal language
models.
Large language models (LLMs) have gained tremendous
popularity in recent years for the development ofagentic
systems (systems and workflows that can be automated to
operate with minimum human supervision. Within the LLM
ecosystem, VLMs have gained attention within the robotics
community to utilize video and image data by leveraging
multi-modal architectures, thus finding applications in scene
understanding, behavior planning and robot control. More
crucially, VLMs generate a human comprehensible output
in the form of natural language which thus forms a critical
bridge in understating robot’s reasoning capabilities. Fur-
thermore, the adoption of camera vision (with or without
natural language) for model inference now allows to scale
the operations in a wide variety of robotics tasks without in-
stallation of task specific sensors or development of operation
specific algorithms. To this end, VLMs provide a promising
opportunity to be implemented as a failure and anomaly
identification mechanism in a variety of robot operations.
To this end, a RAG informed VLM framework, Fail-
RAG, is proposed for identifying anomalies and failures
in industrial robot operations. Figure 1 now captures the
overall idea of the proposed framework. As compared to the
existing state of the art, the frameworks does not require
any fine-tuning for introducing new edge cases and can
be implemented for real-time operations on significantly
less compute. Furthermore, a statistical analysis for several
parametric variations of the RAG formulation is presented
to consolidate the framework’s performance. More critically,
deeper insights on the framework’s performance driven by
the nature of the vector embeddings has been presented in
the analysis section.
Section II reviews recent work in failure and anomalyarXiv:2606.19598v1  [cs.RO]  17 Jun 2026

detection in robotics domain. Section III presents the prob-
lem formulation and the proposed methods. Section IV
discusses the experiment design and results followed up by
section V providing insights in the operation of Fail-RAG.
Finally, section VI provides conclusions and some future
work directions.
II. RELATED WORK
Anomaly detection and Out-of-Distribution detection are
two related research topics and methods developed in these
field can be potentially applied for robot operation failures.
While a rich set of literature indulges in adopting conven-
tional or rule-based methods towards anomaly detection, they
face challenges in scaling up to larger set of operations and
their subsequent edge cases [5], [6], [7]. To this end, we focus
on recent study of using generative models for failure de-
tection, such as LLMs, VLMs and Vision-Language-Action
(VLAs).
A large body of work builds upon the conventional idea of
finding discrepancies in camera vision information (images)
as compared to a baseline or ‘normal’ image. For exam-
ple, failures where robot performs tasks that are different
from instructions but semantically normal are considered
as “Semantic Misalignment Failures” in [8]. Detection of
such failures is based on post-training VLMs using collected
semantics misalignment dataset and adding classification
heads. Similarly, [9] emphasizes the importance of failure
data with fine-grained failure categories and reasoning, and a
VLM training on multi-view images. As contrast to compar-
ing images, [10] proposes to construct a unified scene graph
from multi-view scene images to provide full knowledge
about the environment, which is further used to compare with
an expected scene graph generated during planning to detect
any failures. A similar framework of integrating ontologies,
logical rules, and LLM-based planners is proposed for online
failure identification and recovery for robotic operations in
[11].
While the body of work introduced above focuses on
difference driven failure identification, a recent set of work
goes a step further for failure identification and subsequent
recovery. AHA [12], proposes fine-tuning a language model
on a failure dataset. Similarly, FailSafe [13] is a finetuned
VLM that detects failures and outputs executable actions to
recover from detected failures. ViFailback was introduced
in [14], where visual symbols are generated to provide
recovery guidance. SAFE, proposed in [15] uses a single
scalar to predict failures by learning the internal latent
features.
Thus, a large set of contemporary works focus on fine-
tuning existing model for failure identification or/and recov-
ery. While powerful, the major drawback of these methods is
(a) the necessity to capture vision-action failure data, and, (b)
sourcing heavy compute resources for tuning vision-language
models (which may sometimes go upwards of a few billion
parameters).
In contrast, several methods also explore the idea of
identifying failures without the use of extensive datasets orfine-tuning. For example, anomalies are classified into robot-
driven and environment-drive in [16], and a mix-of-experts
framework was proposed to detect anomalies from both
sources by combining VLM and Gaussian-mixture models.
VLMs are directly used as monitoring tooling during task
execution in [17]. Further, skills for robot to address such
failures are generated by VLMs and incorporated into the
robot behavior tree. While interesting, the above works are
limited by their capacity to identifying categorical failures
(via outliers or deterministic behavior trees) and do not
exploit the powerful capabilities of VLMs in providing
semantic reasoning for the failures. Such a reasoning or
insight can be critical in identifying effective means of
intervention and failure corrections, especially in operations
requiring human-robot collaborations.
Thus, the primary focus of our work tries to address two
potential gaps in the existing literature:
•Formulate a framework free of fine-tuning and thus
can be adopted relatively easily within robot operation
workflows.
•Utilizes the powerful language model capabilities in not
only identifying the failures, but also provide insights
on the potential cause of the failure.
III. FORMULATION
Figure 2 illustrates the flow of the proposed framework.
Inspection camera video stream is received by an image
processing layer which compiles the video frames as a
single image containing time-sequence images. This images
is provide to a vision language model along with (a) a process
specific prompt and, (b) a process specific answer template.
The provisioned answer template follows a structured format
such asjsonoryamlthat allows for seamless integration
with an application programming interface (API).
A. Retrieval Augmented Generation (RAG) Embeddings
As discussed in section II majority of the state-of-the art
frameworks rely heavily on fine-tuning on failure scenarios
requiring significant compute resources for VLM-finetuning.
To avoid this, a minimalistic RAG based framework is
proposed where failure scenarios are provided to the VLM
as reference documents to predict operational failure. In this
novel framework, the pairs of time-sequenced images with
their respective ground truth scenarios are embedded within
a memory stack. For efficient processing of multi-modal
data (image and text), Contrastive Language-Image Pre-
Training (CLIP) encoder is used [18]. As illustrated in Fig. 2,
the real-time image data is processed as CLIP embeddings
and is used to retrieve the relevant information from the
memory stash by utilizing comparison metrics between the
vector embeddings.
As witnessed in contemporary works, different metrics
used within the RAG formulation to measure the distance
between the indexed embeddings and the query embeddings
lead to different performance [19]. Typical strategies for
making the choice of the measurement metric depends on
the data that is used to create the embeddings (such as

Fig. 2. Overview of the proposed motion anomaly detection framework. A view planner captures frames from a video streaming object based on a one
of the fixed frame rates5,10,15,20,25Frames. The single image compiled of several frames is encoded as aR512vector using the CLIP model. A
pre-compiled RAG dataset tries to gather the relevant failure modes using one of the vector comparison metrics (cosine similarity, dot product, or,L 2
distance). The ranked retrieval information is presented to a off-the-shelf VLM agent along with a standardized prompt and expected response template.
The structures response can be integrated within existing API tools for human interpretation and subsequent operation intervention.
text, graphics or audio). In this work, since we are dealing
with multi-modal embeddings, we have chosen three metrics
to compare and investigate for their performance.RAG
distance metrics:For given two vector embeddings,ˆu 1and
ˆu2, and angleθbetween the two, different distance metrics
for RAG computation are given as:
cosine similarity= 1−cos(θ)(1)
L2=||ˆu 1−ˆu 2||(2)
dot product= ˆu 1·ˆu2 (3)
Thus, the cosine similarity metric disregards the magni-
tude of the vectors and accounts for only for their relative
orientation.
B. Instruction structuring
As illustrated in Fig. 2, a reinforced instructional struc-
turing is adopted in this work. First, a standardizedJSON
template is created for each task. Such a JSON template
includes four primary attributes: (a) types of failures for a
particular operation, (b) deterministic status of for that par-
ticular failure type, (c) a sub-failure for each type of failure
object, and (d) the reason for each type of sub-failure. For
instance, for an operation like palletization, types of failure
objects could include instances such as robot-arm, loading
pallet, flipping-station, etc. For this work, the deterministic
status for failure objects have been constrained to ‘normal’,
‘anomalous’ or ‘unknown’. Within the scope of this work, it
has been observed that such a constraint while limiting the
response options for the VLM, significantly improves the
performance (absolute accuracy) and also provides a concise
metric for quantifying the model’s performance. For each
Fig. 3. Five experiment setups in simulation ([1] Palletization, [2]
MoMa - Transport, [4] MoMa - Traverse) and physical experiments ([3]
Depalletization, [5] Assembly) used to evaluate proposed failure detection
methods
failure type, a sub-failures are specified attributes that can
allow to identify the failure cause. For example, the failure
of robot-arm could be attribute to failure of suction-gripper,
or limitations of joint-motions due to motion singularities.
Finally, the reasons for sub-failures are unconstrained natural
language responses that can allow operators/inspectors to
debug the failure scenario.
IV. EXPERIMENTS
We conducted simulation using two robot platforms for
a set of tasks in Nvidia Isaac Sim: a fixed robot arm
and a mobile manipulator. The fixed robot arm (UR5 from
Universal Robots) is programmed for palletizing. The mo-
bile manipulator (MoMa) consists of a mobile robot base
(Clearpath Ridgeback) and a 7 DoF robot arm (Franka).
Further more, we used a physical UR5 and a Fanuc M-20iB
robot arm for assembly, cleaning and depalletizing.

TABLE I
FRAMEWORKATTRIBUTES
Operation Variations Sampling Rate (Hz)
Palletization6 0.05to0.27
MoMa – Transport3 0.5to2.5
Depalletization3 0.5to2.5
MoMa – Traverse2 0.5to2.5
Assembly2 0.2to1
Inference Tasks
Vector Encoding CLIP :ViT-B-32
VLM Inference Qwen :qwen2.5vl:32b
A. Robot operations
Figure 3 shows the five experiment setups used to evaluate
the proposed RAG-based failure detection methods.
1. Palletizing:a UR5 robot is programmed to pick
incoming boxes from the conveyor and place them on a
pallet. Boxes are three-walled (one side open), the robot
arm should place all boxes with the open side on the pallet.
When an incoming box has its open side facing upwards, the
robot should flip the box by using the L-shaped station next
to the conveyor. We simulate this palletizing process and
potential failures caused by: irregular box size, inaccurate
L-shape station location.
2. MoMa - Transport:a MoMa is transporting some
goods inside a warehouse. The goods are placed on the
mobile robot platform. Stack of goods might be unstable
during transportation due to MoMa accelerates/decelerates.
3. Depalletizing:a Fanuc robot (M-20iB) is depalletizing
packages from a pallet to a moving conveyor. Packages
include cardboard boxes and shrink-wrapped items. Soft
plastic packaging material might cause picking failure or
loose grasping.
4. MoMa - Traverse:a MoMa is moving from point A to
point B after receiving the command from fleet management
software. Some minor layout changes to the environment that
haven’t been updated to the map might cause collisions.
5. Assembly:a UR5 robot is used to assemble a
mechanical kit by picking components from one tray and
placing them into target locations. Inaccuracy in workstation
setup might cause assembly errors.
B. RAG Dataset
Table I gives an overview of the RAG dataset composed
for failure identification. Every operation can have several
failure scenarios apart from the baseline (expected) function.
Further, each Operation-failure pair has been captured at
different sampling frequencies. The idea behind capturing
different frequencies is to provide insights on the tradeoff be-
tween the framework’s real-time operational limits (too high
sampling frequency may result in computational overload)
vs information contained withing the query image (too low
frequency may not capture relevant operation feature and
may result in poor absolute performance/ absence of ade-
quate reasoning). In this work, the frame sampling frequency
ranges between0.05Hz to2.5Hz for different operations innet5gradations. For every set{operation,failure scenario,
sampling frequency},5instances of distinct images have
been provided for improve the generalization of the frame-
work. Thus, for every operation a total offailure scenarios×
5 (sampling frequencies)×5 (instances)data points have
been captured. The performance validation occurs on3
images not used for composing the RAG dataset, thus making
a roughly60−40training validation split.
While categorizing different processes via sampling fre-
quencies makes sense from an engineering perspective, it is
difficult to translate that from the LLM-processing point-of-
view. From the language model perspective (including RAG)
the difference in different failure scenarios and their cor-
responding visual representations can only be distinguished
by feature similarities or differences, especially in the latent
embedding sense. Thus, for improved interpretation of the
VLM’s performance, different images embeddings (for query
images) are utilized as parametric variation and utilized to
investigate the variability in the retrieval performance and
the VLM’s performance.
C. VLM Models
Over the last couple of years, several competitive VLMs
have emerged (both proprietary and open-source) that could
potentially be utilized for this framework. For this work,
the open-sourceQwenfamily of models have be chosen for
benchmarking. The choice of these models comes from : (a)
The primary emphasis of the work being on improvements
brought about by RAG augmentaion to off-the-shelf VLM
models and not the standalone performance of different
models, (b) the models being available in different parameter
sizes (3b, 11b & 32b) for a variable compute constraints,
and, (c) being open-source and widely adaptable to different
application programming interfaces such as ‘Ollama’ to
enable recreation of our benchmarks by the community.
Table I illustrates the details of the different parameters and
models utilized for the proposed framework.
D. Results
Based on the experiments outlined in sec IV, the perfor-
mance of the framework has been outline in two steps. The
RAG-only performance captures how different RAG metrics
work in pure retrieval of the relevant RAG data given the
query image. This is purely mathematical, and hallucination
free performance evaluation to investigate image retrieval. In
the second step, VLM-RAG is compared against pure VLM.
1) Retrieval Performance:As outlined in section III-
A, the retrieval only performance is evaluated across all
operations and their failure scenarios for all three RAG
distance metrics. Figure 4 illustrates this performance with
the sampling frequency for different task increasing across
the horizontal axis. The figure also utilize distinct colors and
markers for differentiating between different distance mea-
surement metric and different robot operations respectively.
The retrieval scores ranging from0to1indicate the mean
score for identifying different queries which individually

5 10 15 20 25
Image sequence frames0.30.40.50.60.70.80.91.01.1T op-1 Retrieval
Retrieval-only performance
Metric (color)
Dot Product
Cosine Similarity
2
Operation (marker)
Palletizing
MoMa - Transport
MoMa - Traverse
Depalletizing
AssemblyFig. 4. Retrieval-only performance captures the ability of different RAG
measurement metrics (III) such as (a) cosine similarity, (b)L 2distance, and
(c) vector dot product in retrieving relevant failure cases.
return0and1. For instance, if6out of10queried images
return the correct failure mode, the score is0.6.
The evaluations indicated that more than a specific dis-
tance measurement metric, or a specific frame rate for data
collection across all operation, it is a combination of a
distance metirc and operation that may create variable perfor-
mance. In general, for all operations,L 2distance and cosine
similarity work quiet well irrespective of the sampling fre-
quencies. Within different robot operations, while most other
operations show expected performance,MoMa - Transport
shows poor performance across all distance measurement
metric as well as different sampling frequencies. Based on
the results cosine distance is chosen as the preferred method
to measure the distance based for RAG framework as it
allows for equivalent performance for vector embeddings
based on images and texts alike.
2) VLM-RAG:Figure 5 illustrates the comparative per-
formance between the the VLM model (baseline) and
Fail-RAG for all different operations and across differ-
ent frame rates. The evaluations capture the accuracy of
the agent in identifying not only if the image indicates
‘failure’, but also the type of failure as provided in the
in the RAG documents. For example, for an operation
such as Palletizing, different sub-failure modes may in-
cluderobot_arm,pallet,flipping_station, and,
overall_operationalong with the corresponding rea-
sons. The evaluations illustrated in Fig. 5 capture if the
agent identifies correctly if the sub-failures arenormalor
anomalous.
Results indicate in general an average of around25%
increase in the accuracy performance across all operations
with a few increasing by as much as40%. Fail-RAG
outperforms significantly for all operations expect for a
single scenario forMoMa - Transport. Compared to the
simulation scenarios, results are much more improved over
6080100Palletizing
6080100MoMa - Transport
6080100MoMa - Traverse
6080100Depalletizing
5 10 15 20 256080100AssemblyFailure Identification Accuracy (VLM vs Fail-RAG)Accuracy (%)
Image Sequence Frames
VLM Fail-RAGFig. 5. Comparison of the VLM and the proposed Fail-RAG
framework for accurate failure identification. The metrics capture
the correct characterization of several operation indicators such as
robot arm,payload,overall operation, etc. asnormalor
anomalousfrom the agent’s response. For all operations, Fail-RAG
consistently shows better performance as compared to vanilla VLM, except
for an outlier for one of the cases of MoMa-Transport.
the baseline for real experiments such asDepalletizingand
Assembly. One potential reason for this is that as compared
to simulation, there is more pronounced impact of shadows
and lighting in physical experiments, thus making RAG more
relevant.
Unfortunately, there wasn’t any noticeable pattern for Fail-
RAG’s performance for all operations as function of increas-
ing/decreasing the frame rate. While there were variable re-
sults for individual operations (such as the best performance
forPalletizingwas for15frames and forMoMa-Transport
was for5frames), there was no common trend across all
operations. This lead to analysis from the the perspective of
vector embeddings used within the RAG database.
V. ANALYSIS
Within the RAG framework, every image and text is
encoded and stored as aR512vector. When presented with
a query image the CLIP model encodes the query image
asR512and uses the distance metric (cosine similarity) to
retrieve the vectors mostly aligned with the query vector.
For a RAG database containing several failure scenarios of
a robot operation as a vector embeddings, we hypothesize
that the RAG performance can be dictated based on how

Palletizing
0.1374 (N=48)
 0.1070 (N=48)
 0.0783 (N=48)
 0.0808 (N=48)
 0.0708 (N=48)
MoMa - Transport
0.1154 (N=24)
 0.0488 (N=24)
 0.0491 (N=24)
 0.0332 (N=24)
 0.0303 (N=24)
MoMa - Traverse
0.1244 (N=16)
 0.0467 (N=16)
 0.0441 (N=16)
 0.0510 (N=16)
 0.0394 (N=16)
Depalletizing
0.0715 (N=24)
 0.1068 (N=24)
 0.0878 (N=24)
 0.0955 (N=24)
 0.0899 (N=24)
5Assembly
0.0342 (N=16)
100.0545 (N=16)
150.0623 (N=16)
200.0466 (N=16)
250.0449 (N=16)
0.000.020.040.060.080.10
Cosine DistanceOperations
Image Sequence FramesFig. 6. Heatmaps illustrating the cosine distance between any two vector embeddings of an operation. For every operation, there are8distinct vectors
for every scenario. Thus, total number of vectors per tile isN= 8×scenarios . For a given set ofNvectors, the horizontal and the vertical axes aligns
them inversely such that the diagonal is the the same vector comparing with itself resulting all diagonal values to be0.00. Tiles with dense yellow hue
correspond to operation - image sequence combinations where the scenario embeddings are very distinct from one another where are the ones with blue
hue are the ones with vector embeddings similar to each other. Similarity is measured by cosine distance (1−cosine similarity) which indicates0.0for
identical vectors and1.0when vectors are orthogonal. Below each tile is mean cosine distance between all the vectors for that operation-image sequence
combination along withN, the total number of vectors used for averaging.
distinct the vector embeddings for each failure scenario can
be distinct from one another.
Figure 6 illustrates the phenomenon of comparing vectors
within a database with one another. The25tiles indicate
25distinct RAG databases for the image embeddings. Each
tile represents a set ofNvectors measured against the same
Nvectors arranged inversely. This results in all diagonal
measurements being the measurements between identical
vectors rendering the distance to be0.0. The hues yellow
and blue indicate how distinct and similar the vectors are
from each other respectively. Thus, a tile with dense yellow
hue indicates that all vectors are very distinct from each other
and one with deep blue hues indicate that all vectors are quite
similar to each other. Furthermore, the pixel density is based
on the total number of embeddings in the database. Thus,
tiles with higher number of distinct vectors (N= 48) have
higher pixel density relative to the ones with lower vector
count (N= 16).
When treating every operation independently, and com-
paring with the results of Fig. 5, it can be observed that ingeneral higher mean cosine distance is indicative of higher
Fail-RAG performance. While there are some outliers to
the trend, the general trend indicates alignment with our
proposed hypothesis. Figure 7 illustrates this effect more
explicitly. The mean cosine distances (one for each tile) and
corresponding Fail-RAG performance are plotted against one
another. The trendline now indicates a general trend in which
increasing cosine distance improves the RAG based VLM
performance.
VI. DISCUSSION
In this work, Fail-RAG : a RAG based VLM framework
has been proposed towards failure identification in robotics
processes. Fail-RAG is significantly effort-efficient as com-
pared to the majority of the contemporary works which focus
on fine-tuning VLMs/VLAs with failure-datasets. Further-
more, this work presents a framework for curating failure
data samples and provides insights on how different data
collection procedures may result in performance variability.
Fail-RAG also demonstrates significant performance im-

0.04 0.06 0.08 0.10 0.12 0.14
Mean Cosine Distance60708090100Accuracy (%)Cosine Distance vs Accuracy
TrendlineFig. 7. The accuracy retrieved from the Fail-RAG evaluations (Fig. 5)
can be arranged against the mean cosine distance calculations illustrated in
Fig. 6. Such a perspective now maps how the Fail-RAG performance varies
as the mean cosine distance changes. As expected, increasing the cosine
distance (which indicates failure scenarios are encoded distinctly from one
another) results in improved Fail-RAG performance.
provement over the baseline demonstrating the utility of off-
the-shelf VLMs without any expensive fine-tuning.
More critically, we emphasize on diving deeper on in-
ternal mechanics of the RAG framework to investigate the
sensitivity of the RAG framework as a function of how the
vector embeddings are encoded in the dataset, how their
similarity is measured, and how their distinct directions in
the embedding space impacts the framework’s performance.
The interesting implications of this effort now scales beyond
just robot failure identification and can potentially be adopted
for general purpose vision based monitoring and inspection
beyond robotics.
While composing several frames together and presenting
a single query image makes the framework lightweight, it
also introduces some delay in identification of the failure.
Depending on the task, such a delay might be acceptable/
undesirable due to potential disruptions to automation work-
flows. Thus, minimizing the delay can be a potential future
direction to build on top of this work. Working in the latent
space or/and introducing new operational views can be some
methods that could be adopted towards early prediction of
the failure thus reducing the identification delay.
Furthermore, while figure 6 illustrates that variability in
vector embeddings is crucial for improving performance,
the exact mathematical modeling or formalizing the relation
between them can further strengthen the adoption of this
framework. Such a relationship can now be utilize to curate
the RAG dataset thus improving the efficiency of the data-
collection process.
REFERENCES
[1] T. Yano, D. Hagihara, N. Kimura, N. Chihara, and K. Ito, “Param-
eterized b-rep-based surface correspondence estimation for category-level 3d object matching applicable to multi-part items,” in2022 IEEE
18th International Conference on Automation Science and Engineering
(CASE). IEEE, 2022, pp. 611–618.
[2] N. Kim and S. S. Kwak, “Investigating behavioral and cognitive
changes induced by autonomous delivery robots in incidentally co-
present persons,” in2024 IEEE/RSJ International Conference on
Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 2514–2519.
[3] R. S. Novin, A. Yazdani, T. Hermans, and A. Merryweather, “Dynamic
model learning and manipulation planning for objects in hospitals
using a patient assistant mobile (pam) robot,” in2018 IEEE/RSJ
International Conference on Intelligent Robots and Systems (IROS).
IEEE, 2018, pp. 1–7.
[4] G. Kazhoyan, S. Stelter, F. K. Kenfack, S. Koralewski, and M. Beetz,
“The robot household marathon experiment,” in2021 IEEE Interna-
tional Conference on Robotics and Automation (ICRA). IEEE, 2021,
pp. 9382–9388.
[5] M. Visinsky, J. Cavallaro, and I. Walker, “Robotic fault detection
and fault tolerance: A survey,”Reliability Engineering & System
Safety, vol. 46, no. 2, pp. 139–158, 1994. [Online]. Available:
https://www.sciencedirect.com/science/article/pii/0951832094901325
[6] O. Pettersson, “Execution monitoring in robotics: A
survey,”Robotics and Autonomous Systems, vol. 53,
no. 2, pp. 73–88, 2005. [Online]. Available:
https://www.sciencedirect.com/science/article/pii/S092188900500134X
[7] I. Hwang, S. Kim, Y . Kim, and C. E. Seah, “A survey of fault detection,
isolation, and reconfiguration methods,”IEEE Transactions on Control
Systems Technology, vol. 18, no. 3, pp. 636–653, 2010.
[8] C. Grislain, H. Rahimi, O. Sigaud, and M. Chetouani, “I-FailSense:
Towards General Robotic Failure Detection with Vision-Language
Models,” Sep. 2025, arXiv:2509.16072 [cs]. [Online]. Available:
http://arxiv.org/abs/2509.16072
[9] P. Pacaud, R. Garcia, S. Chen, and C. Schmid, “Guardian: Detecting
Robotic Planning and Execution Errors with Vision-Language
Models,” Dec. 2025, arXiv:2512.01946 [cs]. [Online]. Available:
http://arxiv.org/abs/2512.01946
[10] H. Chong, J. Lee, and H. Ahn, “Robust Task Planning
via Failure Detection Using Scene Graph From Multi-View
Images,”IEEE Robotics and Automation Letters, vol. 11,
no. 2, pp. 1986–1993, Feb. 2026. [Online]. Available:
https://ieeexplore.ieee.org/document/11302797/
[11] C. Cornelio and M. Diab, “Recover: A neuro-symbolic framework
for failure detection and recovery,” 2024. [Online]. Available:
https://arxiv.org/abs/2404.00756
[12] J. Duan, W. Pumacay, N. Kumar, Y . R. Wang, S. Tian, W. Yuan,
R. Krishna, D. Fox, A. Mandlekar, and Y . Guo, “AHA: A Vision-
Language-Model for Detecting and Reasoning Over Failures in
Robotic Manipulation,” Oct. 2024, arXiv:2410.00371 [cs]. [Online].
Available: http://arxiv.org/abs/2410.00371
[13] Z. Lin, J. Duan, H. Fang, D. Fox, R. Krishna, C. Tan, and
B. Wen, “FailSafe: Reasoning and Recovery from Failures in
Vision-Language-Action Models,” Oct. 2025, arXiv:2510.01642 [cs].
[Online]. Available: http://arxiv.org/abs/2510.01642
[14] X. Zeng, X. Zhou, Y . Li, J. Shi, T. Li, L. Chen, L. Ren, and Y .-L. Li,
“Diagnose, Correct, and Learn from Manipulation Failures via Visual
Symbols,” Dec. 2025, arXiv:2512.02787 [cs]. [Online]. Available:
http://arxiv.org/abs/2512.02787
[15] Q. Gu, Y . Ju, S. Sun, I. Gilitschenski, H. Nishimura, M. Itkina, and
F. Shkurti, “SAFE: Multitask Failure Detection for Vision-Language-
Action Models,” Oct. 2025, arXiv:2506.09937 [cs]. [Online].
Available: http://arxiv.org/abs/2506.09937
[16] C. Willibald, D. Sliwowski, and D. Lee, “Multimodal Anomaly
Detection with a Mixture-of-Experts,” in2025 IEEE/RSJ International
Conference on Intelligent Robots and Systems (IROS). Hangzhou,
China: IEEE, Oct. 2025, pp. 20 020–20 027. [Online]. Available:
https://ieeexplore.ieee.org/document/11245878/
[17] F. Ahmad, J. Styrud, and V . Krueger, “Addressing Failures in Robotics
using Vision-Based Language Models (VLMs) and Behavior Trees
(BT),” Nov. 2024, arXiv:2411.01568 [cs]. [Online]. Available:
http://arxiv.org/abs/2411.01568
[18] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh,
S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark,
G. Krueger, and I. Sutskever, “Learning transferable visual models
from natural language supervision,” 2021. [Online]. Available:
https://arxiv.org/abs/2103.00020

[19] H. Elkiran and J. Rasheed, “Evarag: Evaluating advanced rag tech-
niques with indexing and distance metrics,”IEEE Access, vol. 13, pp.
215 724–215 747, 2025.
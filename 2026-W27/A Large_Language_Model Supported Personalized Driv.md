# A Large-Language-Model Supported Personalized Driving Framework for Lane Change in Highway Scenarios

**Authors**: Dong Bi, Yongqi Zhao, Paul Kovacevic, Tomislav Mihalj, Ji Zhou, Jiayuan Gong, Arno Eichberger

**Published**: 2026-06-30 10:58:21

**PDF URL**: [https://arxiv.org/pdf/2606.31483v2](https://arxiv.org/pdf/2606.31483v2)

## Abstract
Personalized driving can improve the user acceptance of automated driving systems. However, existing methods still provide limited support for translating natural-language driving preferences, especially when such preferences are expressed implicitly, into executable and distinguishable driving behaviors. This paper proposes a large language model (LLM)-supported personalized driving framework for highway lane-change scenarios. The framework maps natural-language driving commands to executable planning parameters in the open-source Apollo automated driving stack according to three driving styles: aggressive, normal, and conservative. To establish this mapping, candidate planning parameters are evaluated based on the resulting lane-change behaviors, and style-specific parameter sets are constructed through clustering and style-intensity ranking. For command interpretation, a retrieval dataset is constructed to support retrieval-augmented generation (RAG), enabling LLM-based interpretation of implicit user commands. Experimental results show that the derived parameter sets generate distinguishable personalized lane-change behaviors, while RAG consistently improves preference interpretation, particularly for implicit commands. These results indicate the potential of integrating LLM-based natural-language interaction with Apollo to support personalized lane-change behavior generation. The source code and the relevant datasets are available at: https://github.com/ftgTUGraz/LLM-Personalized-Driving.

## Full Text


<!-- PDF content starts -->

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 1
A Large-Language-Model Supported Personalized Driving
Framework for Lane Change in Highway Scenarios
Dong Bi, Yongqi Zhao, Paul Kovacevic, Tomislav Mihalj, Ji Zhou, Jiayuan Gong, and Arno
Eichberger,Member, IEEE
Abstract—Personalized driving can improve the user accep-
tance of automated driving systems. However, existing methods
still provide limited support for translating natural-language
driving preferences, especially when such preferences are ex-
pressed implicitly, into executable and distinguishable driving
behaviors. This paper proposes a large language model (LLM)-
supported personalized driving framework for highway lane-
change scenarios. The framework maps natural-language driving
commands to executable planning parameters in the open-
source Apollo automated driving stack according to three driving
styles: aggressive, normal, and conservative. To establish this
mapping, candidate planning parameters are evaluated based on
the resulting lane-change behaviors, and style-specific parame-
ter sets are constructed through clustering and style-intensity
ranking. For command interpretation, a retrieval dataset is
constructed to support retrieval-augmented generation (RAG),
enabling LLM-based interpretation of implicit user commands.
Experimental results show that the derived parameter sets gen-
erate distinguishable personalized lane-change behaviors, while
RAG consistently improves preference interpretation, particu-
larly for implicit commands. These results indicate the potential
of integrating LLM-based natural-language interaction with
Apollo to support personalized lane-change behavior generation.
The source code and the relevant datasets are available at:
https://github.com/ftgTUGraz/LLM-Personalized-Driving.
Index Terms—Automated driving, personalized lane change,
LLMs, retrieval-augmented generation, natural-language inter-
action, virtual testing.
I. INTRODUCTION
FUTURE automated vehicles are expected to move be-
yond safety-oriented automation toward more flexible,
intelligent, and interactive mobility solutions [1]. Personal-
ized driving technologies have attracted increasing attention
for their potential to improve user acceptance, comfort, and
driving experience in automated driving systems (ADSs) [2]–
[4]. In personalized automated driving, a key problem is how
to translate user preferences into vehicle behaviors. This is
This work has been submitted to the IEEE for possible publication.
Copyright may be transferred without notice, after which this version may
no longer be accessible. (Corresponding author: Yongqi Zhao)
Dong Bi is with the School of Intelligent Connected Vehicle, Hubei Univer-
sity of Automotive Technology, 442002 Shiyan and Institute of Automotive
Engineering, Graz University of Technology, Graz 8010, Austria (e-mail:
dong.bi@tugraz.at)
Yongqi Zhao, Paul Kovacevic, Tomislav Mihalj, Ji Zhou,
and Arno Eichberger are with the Institute of Automotive
Engineering, Graz University of Technology, Graz 8010, Austria
(e-mail: yongqi.zhao@tugraz.at; paul.kovacevic@tugraz.at; tomis-
lav.mihalj@tugraz.at; ji.zhou@student.tugraz.at; arno.eichberger@tugraz.at)
Jiayuan Gong is with the School of Intelligent Connected Vehicle,
Hubei University of Automotive Technology, 442002 Shiyan (e-mail: jy-
gong@huat.edu.cn)particularly challenging for lane changes [5], where vehicle be-
havior depends on coupled longitudinal and lateral control and
interactions with surrounding traffic participants. Therefore,
personalized lane-change adaptation through natural language
requires reliable interpretation of user preferences and gener-
ation of executable, distinguishable lane-change behaviors for
different driving styles.
Previous works have explored user-preference modeling
based on historical driving data [6], predefined driving-style
labels [7], [8], and manually designed criteria [9]. These stud-
ies demonstrate the feasibility of adapting vehicle behaviors
to individual preferences, but they provide limited support
for direct human-vehicle interaction through natural language.
Recent large language model (LLM)-based automated driving
studies have introduced language reasoning into high-level task
planning [10], system reconfiguration [11], human-like inter-
action [12], decision support [13], and style-aware trajectory
generation [14], [15]. However, the use of natural-language
interaction to generate distinguishable and adjustable personal-
ized lane-change behaviors within an automated driving stack
remains insufficiently explored.
Another unresolved issue is the interpretation of implicit
driving preferences, where users do not directly specify a
driving style but express the desired lane-change behavior
indirectly. Such preferences may be conveyed through urgency,
comfort needs, safety concerns, or traffic situations, rather than
explicit labels such as aggressive, normal, or conservative,
and are common in human–vehicle interaction [16]. However,
existing LLM-based driving studies offer limited support for
reliably mapping implicit preferences to executable, style-
specific driving behaviors.
In summary, prior studies still present three limitations.
First, limited support is available for intuitive and flexible
driving-style adaptation through natural-language interaction,
especially when users express explicit preferences for ag-
gressive, normal, or conservative driving styles. Second, the
generation of distinguishable and adjustable personalized lane-
change behaviors within an open-source automated driving
stack remains insufficiently explored. Third, existing methods
provide limited support for mapping implicit driving prefer-
ences to executable, style-specific lane-change behaviors.
To address these limitations, this paper proposes an LLM-
based personalized driving framework for lane-change adap-
tation in highway scenarios. The framework is implemented
in the open-source Apollo automated driving stack [17]. The
main contributions are summarized as follows:
1) A personalized lane-change framework is proposed to
translate natural-language driving commands into ex-arXiv:2606.31483v2  [cs.RO]  1 Jul 2026

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 2
ecutable lane-change behaviors through driving-style
interpretation and planning-parameter mapping.
2) Lane-change parameter sets are constructed for different
driving styles to generate distinguishable behaviors and
support multiple intensity levels for aggressive, normal,
and conservative driving.
3) A natural-language preference dataset is constructed
for retrieval-augmented generation (RAG), supporting
more accurate interpretation of explicit and implicit
commands.
II. RELATED WORK
A. Personalized Driving and Lane-Change Adaptation in ADS
Existing surveys indicate that adapting ADS to driver prefer-
ences can improve user experience, whereas generic ADS may
neglect individual characteristics. Personalization has been
explored in representative ADS functions, including forward
collision warning [18], adaptive cruise control [19], [20],
automated driving speed adaptation [21], data-driven driving
style learning [6], and user-driven adaptation to dynamic
preferences [22]. The personalization mechanisms of these
studies mainly rely on historical behavior data, predefined
preference representations, or explicit feedback, rather than
direct natural-language preference expression.
As a representative and challenging ADS maneuver, lane
change provides an important scenario for studying person-
alized driving behavior [5]. Existing studies have modeled
personalized lane-change responses [7], selected trajectories
according to user preferences on safety, comfort, and stabil-
ity [9], and generated human-like lane-change trajectories con-
sidering driver characteristics and traffic-environmental fac-
tors [8]. Hu et al. [23] further used user takeover interventions
to update perceived safe driving zones and personalize lane-
change trajectory planning online. However, these methods
do not focus on natural-language-based preference expression,
nor do they explicitly address how to generate distinguishable
and adjustable lane-change behaviors for different driving style
within a mature ADS stack.B. LLM-Based Language Interaction for ADS
According to [24]–[27], recent advances in LLMs have cre-
ated new opportunities for human-centered automated driving
by enabling natural-language understanding, intent interpre-
tation, and high-level decision support. Cui et al. [11], [12],
[28] developed LLM-based interaction frameworks to translate
verbal commands into executable control programs. Xu et
al. [29] used LLMs to personalize warning messages and
human-machine interaction, while Ma et al. [10] proposed
an LLM-based programming planner that converts natural-
language instructions and driving contexts into executable
policies in CARLA [30]. These studies mainly focus on
command execution, communication, or general policy gen-
eration, rather than mapping driver preference expressions to
executable and distinguishable driving-style behaviors.
Recent multimodal large language models and vision-
language-action models have further used to explore per-
sonalized or style-conditioned driving behavior generation.
StyleVLA [14] fine-tunes a vision-language-action model to
generate physically plausible trajectories under predefined
driving-style instructions, such as comfort, sporty, and safety.
PADriver [15] enables switching among slow, normal, and
fast driving modes through predefined personalized prompts
and selects discrete driving actions according to the traffic
environment. In addition, Ge et al. [31] proposed an LLM-
based operating-system architecture for task understanding,
module coordination, and system management. These studies
mainly focus on predefined style instructions, instruction-
driven ADS adaptation, or system-level coordination, while
the interpretation of open-ended, implicit user preferences into
personalized driving styles remains insufficiently explored.
C. Natural-Language Understanding for Driving Preference
Interpretation
To bridge the gap between natural human preference expres-
sion and machine-interpretable driving systems, recent studies
have explored natural-language understanding in ADS. Yang
et al. [32] used LLMs to infer structured system requirements
from in-cabin verbal commands, while Liao et al. [33] im-
proved command grounding by combining textual, emotional,
visual, and contextual cues. Dataset-oriented works such as
TABLE I
COMPARISON OFRELATEDSTUDIES ONPERSONALIZEDDRIVING ANDLLM-BASEDLANGUAGEINTERACTION
Work Personalized
BehaviorPersonalized Driving
StyleLane-Change Focus LLM-Based Language
InteractionImplicit Preference
Dataset
[6]✓ ✓– – –
[8]✓Partial✓– –
[9]✓ ✓ ✓– –
[10]✓– –✓–
[11]✓– –✓–
[12], [26]✓– –✓–
[21]✓– – – –
[22]✓–✓– –
[28]✓– –✓–
[36]✓– –✓–
Ours✓ ✓ ✓ ✓ ✓

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 3
Fig. 1. Architecture of the proposed personalized driving simulation framework.
doScenes [34], Talk2Car [35], and NuPrompt [36] connect
natural-language instructions or prompts with real driving
scenes, objects, and trajectories. LMDrive [37] further incor-
porates natural-language instructions into closed-loop end-to-
end driving in CARLA. Although these works demonstrate the
potential of natural language for driving-scene understanding
and behavior generation, limited attention has been paid to
interpreting implicit preferences across different personalized
driving styles.
In summary, existing studies have explored personalized
driving, LLM-based driving interaction, and natural-language
scene understanding from different perspectives. However, as
summarized in Table I, translating natural-language prefer-
ences into executable personalized driving styles within a
ADS framework remains insufficiently studied. In particular,
implicit expressions are challenging to interpret.
III. METHODOLOGY
Figure 1 illustrates the architecture of the proposed person-
alized driving simulation framework. The framework enables
natural-language interaction between user and automated vehi-
cles, allowing personalized lane-change behaviors to be gener-
ated according to user preferences. The LLM-based interaction
program, described in Section III-A, receives user commands
and driving feedback and infers the underlying driving prefer-
ences. To improve the interpretation of implicit commands, a
retrieval dataset is constructed as described in Section III-B,
providing reference examples for LLM inference. The inferred
driving styles are then converted into Robot Operating System
(ROS) messages and transmitted to the personalized simulation
platform described in Section III-C, where the correspondingpersonalized driving parameter set is selected and executed. To
construct this parameter set, a personalized simulation dataset
is generated by varying lane-change planning parameters and
recording the resulting vehicle behaviors, as described in
Section III-D. This dataset contains samples generated by
varying lane-change planning parameters and recording the
corresponding vehicle behaviors.
A. LLM-based Interaction Program
Figure 2 presents the workflow of the proposed LLM-
based interaction program, which consists of two main stages:
index construction and RAG-based inference. In the index
construction stage, the RAG dataset, which contains natural-
language preference examples, is encoded into dense embed-
dings using theall-MiniLM-L6-v2[38] text encoder, and the
embeddings are stored in a Facebook AI Similarity Search
(FAISS) index [39], [40]. During RAG-based inference, the
user command is encoded by the same text encoder and used as
a query to retrieve the top-3 examples from the FAISS index.
The retrieved examples are then incorporated into the RAG
prompt, which is combined with the system prompt to form
the final prompt for LLM-based driving style classification.
B. RAG Dataset
To support LLM-based driving style classification, a RAG
dataset covering driver commands is required. Because man-
ually constructing such data is time-consuming, a multi-
agent LLM-based data generation framework is designed, as
shown in Figure 3. The framework follows a ”generation-
labeling-validation” workflow. First, a question agent generates

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 4
Fig. 2. Workflow of LLM-based interaction program for driving style
classification.
TABLE II
LLMSUSED BYDIFFERENTAGENTS
Agent Models
Question AgentGPT-4.1; DeepSeek-Chat;
Claude-Haiku-4.5;
Gemini-2.5-Flash;
Grok-4-Fast; Qwen3.5-Plus;
Kimi-K2.5; MiniMax-M2.1
Answer Agent GPT-4.1
Validation Agent DeepSeek-Chat
candidate driving commands according to predefined driving-
style targets and command configurations, including explicit-
ness, difficulty, and length. An answer agent then assigns a
driving-style label to each command, and a validation agent
independently checks the consistency of the generated sample.
Specifically, the question agent uses eight different LLMs, as
listed in Table II, to improve command diversity. Samples with
consistent target labels, answer-agent labels, and validation-
agent labels are directly accepted, whereas the remaining
samples are assigned to a ambiguous set for manual review.
C. Personalized Simulation Platform
The personalized simulation platform consists of two main
parts: the co-simulation system and the personalized driv-
ing parameter set. After receiving the inferred driving style
through ROS messages, the platform selects the corresponding
parameter configuration and updates the lane-change planning
parameters in the co-simulation system, thereby generating
personalized lane-change behaviors.
1) Co-simulation System:A co-simulation system integrat-
ing Apollo and CarMaker [41] was developed in previous
works [42], [43]. Apollo provides the automated driving
stack, whereas CarMaker provides the vehicle dynamics modeland simulation environment. The two systems communicate
through ROS to exchange vehicle states, planning information,
and control commands. In this study, this system serves as the
execution environment for personalized lane-change behavior
generation, where the selected parameter configuration is
applied to the lane-change planning module.
2) Tunable Planning Parameters for Personalized Driv-
ing:The personalization capability is achieved by adjusting
selected planning parameters in Apollo’s longitudinal and
lateral planning modules. This subsection identifies the tun-
able parameters used for personalized lane-change behavior
generation.
a) Longitudinal Planning Parameters:Personalized lon-
gitudinal behavior during lane-change scenarios is achieved
by adjusting the speed planning parameters in Apollo’s final
speed optimization stage. In this study, two types of longi-
tudinal parameters are considered: objective-function weights
and acceleration bounds. The objective-function weights reg-
ulate the trade-off among reference tracking, smoothness,
and responsiveness, while the acceleration bounds define the
allowable acceleration and deceleration range for different
driving styles. The complete theoretical analysis process is
detailed in Appendix A.
According to Apollo’s planning pipeline, the piecewise-jerk
speed optimizer determines the longitudinal speed trajectory
by solving
min
xJ(x) =w sJs(x) +w vJv(x) +w aJa(x) +w jJj(x),(1)
wherex={s i, vi, ai}N−1
i=0 denotes the discretized longitudinal
trajectory at time stepi, andJ(x)is the total cost of a
candidate trajectory. The termsJ s,Jv,Ja, andJ jrepresent
the accumulated costs of position reference tracking, velocity
reference tracking, acceleration, and jerk, respectively. The
corresponding longitudinal weighting parameters are defined
as
wlon={w s, wv, wa, wj}.(2)
In the piecewise-jerk speed optimizer, each weight deter-
mines the penalty strength of its corresponding trajectory term.
Larger weights impose stronger suppression on the related
deviation or motion component. Together with the accelera-
tion bounds, these parameters regulate longitudinal tracking,
smoothness, and responsiveness, as summarized in Table III.
In addition to the objective-function weights, the accel-
eration bounds are used to limit the feasible longitudinal
motion. For different driving styles, the acceleration constraint
is formulated as
ak
min≤ai≤ak
max, k∈ {agg,nor,con},(3)
wherea iis the longitudinal acceleration, andkdenotes the
driving style, including aggressive, normal, and conservative
styles. The acceleration bounds for different driving styles
are predefined according to driving-style-related acceleration
characteristics reported in previous studies [44].
b) Lateral Planning Parameters:Personalized lateral be-
havior during lane-change scenarios is achieved by adjusting
the path planning parameters in Apollo’s piecewise-jerk path
optimizer. In this study, three types of lateral parameters are

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 5
Fig. 3. Multi-agent data generation framework for constructing the RAG dataset used in driving style classification.
TABLE III
PERSONALIZATIONPARAMETERS INLONGITUDINALSPEEDPLANNING
Symbol Parameter Meaning and Behavioral Effect
ws Position tracking weightTracks the upstream reference position; largerw sencourages closer adherence to
the coarse speed profile.
wv Velocity tracking weight Tracks the desired speed; largerw vencourages stronger cruising-speed tracking.
wa Acceleration weightPenalizes longitudinal acceleration; largerw asuppresses aggressive acceleration
and deceleration.
wj Jerk weightPenalizes acceleration variation; largerw jimproves smoothness but may reduce
responsiveness.
considered: objective-function weights, geometric constraints,
and derivative bounds. The objective-function weights regulate
the trade-off among lateral offset, lateral slope, curvature-
related motion, and high-order smoothness. The geometric
constraints define the feasible lateral corridor, while the deriva-
tive bounds limit the derivative bounds constrain the lateral
slope and curvature-related variation. The complete theoretical
analysis process is detailed in Appendix B.
According to Apollo’s planning pipeline, the piecewise-jerk
path optimizer determines the lateral path by solving
min
yJ(y) =w lJl(y) +w dlJdl(y)
+wddlJddl(y) +w dddlJdddl(y),(4)
wherey={l i, dli, ddli}N−1
i=0 denotes the discretized lateral
state sequence in the Frenet frame [45], andJ(y)is the total
cost of a candidate lateral path. Here,l i,dli, andddl irepresent
the lateral offset, first-order lateral derivative, and second-order
lateral derivative, respectively. The termsJ l,Jdl,Jddl, and
Jdddl represent the accumulated costs of lateral offset, lateral
slope, second-order lateral variation, and high-order lateral
smoothness, respectively. The corresponding lateral weightingparameters are defined as
wlat={w l, wdl, wddl, wdddl}.(5)
In addition to the objective-function weights, the feasible
lateral region is constrained by the lane boundaries, obstacle
constraints, and vehicle safety margin. The geometric con-
straint is expressed as
lmin(ξi;δadc)≤l i≤lmax(ξi;δadc),(6)
whereξ iis the spatial sampling point along the reference line,
liis the lateral offset atξ i, andδ adcdenotes the vehicle buffer
margin used to adjust the feasible lateral corridor.
The lateral transition is further limited by the lateral slope
bound
|dli| ≤dl max.(7)
wheredl idenotes the first-order lateral derivative with respect
to the reference-line arc length, anddl max defines the maxi-
mum allowable lateral slope.
In the piecewise-jerk path optimizer, each weight determines
the penalty strength of its corresponding lateral trajectory
term. Together with the geometric constraints and derivative
bounds, these parameters regulate the lateral aggressiveness,

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 6
TABLE IV
TUNABLELATERALPLANNINGPARAMETERS
Symbol Parameter Category Meaning and Qualitative Effect
wl Lateral offset weight Objective functionPenalizes lateral offset; largerw ldiscourages large lateral
deviations.
wdl Slope weight Objective functionPenalizes lateral slope; largerw dlsuppresses steep lateral
transitions.
wddl Curvature weight Objective functionPenalizes second-order lateral variation; largerw ddlsuppresses
second-order lateral variation.
wdddl Jerk weight Objective functionPenalizes third-order lateral variation; largerw dddl improves
smoothness but may reduce responsiveness.
δadc Vehicle buffer margin Geometric constraintExpands the initial feasible lateral region; largerδ adcimproves
feasibility but may allow boundary-near paths.
dlmax Lateral slope bound Derivative constraintBounds the lateral derivative; largerdl max allows steeper and
faster lateral transitions.
smoothness, and feasibility of lane-change trajectories, as
summarized in Table IV.
c) Personalized Parameter Mapping:Based on these
selected longitudinal and lateral planning parameters, a person-
alized driving parameter mapping is constructed to associate
each driving style with multiple executable parameter config-
urations. As summarized in Table V, each style is represented
by three intensity levels, denoted asL 1,L2, andL 3. These
levels are obtained through the intra-class ranking procedure
described in Section III-D, where a higher level indicates a
stronger expression of the corresponding driving style.
D. Personalized Dataset
This subsection presents the construction process of the
personalized dataset. A highway lane-change scenario is first
defined in the co-simulation system. Then, selected Apollo
planning parameters are varied within feasible ranges to gen-
erate offline simulation samples and record the corresponding
lane-change behaviors. Based on the valid samples, clustering
is used to identify aggressive, normal, and conservative driving
styles, and intra-class ranking is further applied to select
representative parameter sets with different intensity levels.
1) Lane-Change Scenario:As shown in Figure 4, the
scenario includes ego vehicle and three surrounding vehicles
with predefined parameters. The ego vehicle is initialized at
Fig. 4. A typical lane change scenario.
126 km/h, while all surrounding vehicles travel at 80 km/h.
The initial longitudinal spacing is determined according to
time-to-collision (TTC) criterion. A target TTC of 5.5 s is
adopted, which is slightly larger than the 4.5–5 s range
suggested for motorway collision-avoidance warning strategies
[41]. The spacing is calculated as
d=v rel·TTC.(8)
wherev reldenotes the relative speed between the ego vehicle
and the traffic vehicle.
2) Offline Simulation:Table VI lists the input parameter
ranges and the corresponding behavioral outputs recorded from
each simulation. Each selected input parameter is discretized
TABLE V
REPRESENTATIVEPARAMETERMAPPING FORPERSONALIZEDDRIVINGSTYLES
Style LevelLongitudinal Parameters Lateral Parameters
ws wv wa wj amin amax wl wdl wddl wdddl δadc dlmax
AggressiveL1 0.6 30 0.3 0.2 -3.0 3.0 3.0 5.0 200 5000 0.5 3.0
L2 0.6 30 0.3 0.2 -3.0 3.0 2.0 4.0 100 3000 0.3 3.0
L3 0.8 25 0.5 0.2 -3.0 3.0 1.0 4.0 300 8000 0.5 2.0
NormalL1 0.8 15 0.7 1.5 -2.75 2.75 0.7 10 700 12000 0.7 1.2
L2 0.8 15 0.7 1.5 -2.75 2.75 0.7 10 500 20000 0.7 1.2
L3 0.8 15 0.7 1.5 -2.75 2.75 0.7 12 700 20000 0.7 1.6
ConservativeL1 1.0 10 1.0 3.0 -2.5 2.5 0.4 20 1200 20000 1.0 0.5
L2 1.0 10 1.5 3.0 -2.5 2.5 0.2 20 1200 50000 1.0 0.4
L3 1.0 10 1.5 3.0 -2.5 2.5 0.4 20 1200 60000 1.0 1.0

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 7
TABLE VI
PERSONALIZEDPARAMETERSPACE ANDBEHAVIORALEFFECTS
Dim. Parameters Range Output
Long.ws
wv
wa
wj
amin
amax[0.6,1.2]
[10,30]
[0.3,1.5]
[0.2,3.0]
[−3.0,−2.5]
[2.5,3.0]Tlc(s)
vego(m/s)
ax(m/s2)
Xj(m),j= 1,2,3,4
Lat.wl
wdl
wddl
wdddl
δadc
dlmax[0.1,4.0]
[5,20]
[100,1200]
[3000,50000]
[0.3,1.0]
[0.4,4.0]δsw(rad)
rego(rad/s)
ay(m/s2)
Yj(m),j= 1,2,3,4
into several representative values with equal spacing for offline
dataset generation.
The ranges of the selected longitudinal and lateral planning
parameters are determined from Apollo’s default configu-
ration and simulation-based boundary exploration. Starting
from the default values, parameter-sweeping simulations are
performed to identify settings that can maintain valid lane-
change behavior. Settings causing planning failure, unsuccess-
ful lane changes, speed-tracking failure, repeated trajectory
corrections, or road-boundary violations are excluded. The
remaining values are used as engineering ranges for offline
dataset generation.
The output variables are selected to capture the personalized
lane-change behavior produced by different parameter settings.
The lane-change timeT lcis used to describe the temporal
efficiency of the maneuver, as illustrated in Figure 4. The
ego velocityv egoand longitudinal accelerationa xcharacter-
ize the longitudinal speed-tracking and acceleration behavior
during the lane-change process. For lateral behavior, Previous
research collected vehicle-state signals such as lateral acceler-
ationa y, steering wheel angleδ sw, and yaw rater ego, and used
their characteristic values for driver-behavior classification into
cautious, normal, and aggressive groups [46]. In addition, the
absolute positions of the ego vehicle and surrounding vehicles,
denoted asX jandY j, are recorded to characterize the spatial
evolution of the lane-change process and the surrounding
traffic context.
TABLE VII
DATASETGENERATION ANDCLUSTERINGSTATISTICS
Group Category Number
Dataset GenerationTotal samples 465
Valid samples 425
Invalid samples 40
Clustering Results
(K-means,K= 3)Aggressive 97
Normal 175
Conservative 153TABLE VIII
CLUSTERINGFEATURES FORDRIVINGSTYLEIDENTIFICATION
Category Feature Weight
Lane change timeT lc 25.40%
Longitudinal accelerationmax|a x|3.67%
Eax 3.67%
Lateral accelerationmax|a y|15.04%
Eay 16.60%
Steering wheel anglemax|δ sw|6.08%
Eδsw11.55%
Yaw ratemax|r ego|6.81%
Erego 11.18%
3) Clustering:A detailed statistical summary of the dataset
generation and clustering results is provided in Table VII.
488 simulation samples were generated, among which 425
were valid and 40 were identified as invalid samples. K-means
clustering was then applied to the valid samples to identify
three representative driving styles. K-means partitions samples
intoKclusters by minimizing the within-cluster distance
between each sample and its assigned cluster centroid [47].
In this study,Kis set to 3 to obtain aggressive, normal, and
conservative driving styles.
For driving-style identification, the clustering feature vector
is constructed from the longitudinal and lateral dynamic char-
acteristics of each lane-change maneuver. The selected features
are summarized in Table VIII, where the accumulated squared
responseE qis used to describe the overall intensity of each
dynamic variable. Here,qdenotes a generic signal selected
froma x,ay,δsw, andr ego.
For a dynamic variableq(t), the accumulated squared
response is defined as
Eq=Ztend
tstartq2(t)dt,(9)
wheret start andt enddenote the start and end times of the lane-
change maneuve. This term represents the overall intensity of
the corresponding motion response during the maneuver. A
larger value indicates a stronger or more persistent dynamic
response.
Figure 5 provides a visual illustration of the sample dis-
tributions for the three driving styles obtained via clustering.
Subfigure (1) shows that the lane-change time increases from
aggressive to conservative driving styles, with statistically
significant differences between the groups based on Student’s
t-test (p <0.05andp <0.001). Subfigures (2)–(4) show
the probability distributions of lateral acceleration, steering
wheel angle, and yaw rate during lane changes. The aggressive
style generally presents larger values and broader distributions,
indicating stronger lateral dynamic responses. In contrast, the
conservative style shows more concentrated distributions with
lower magnitudes, reflecting smoother lane-change behavior.
The normal style lies between these two extremes. These
results indicate that the constructed dataset can effectively

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 8
(1) (2)
(3) (4)
Fig. 5. Statistical distributions of lane-change time and lateral dynamic
responses across driving styles.
distinguish different driving styles through both lane-change
duration and lateral dynamic characteristics.
4) Style-Intensity Ranking:Based on the personalized
dataset obtained from clustering, the diversity of input pa-
rameter combinations and the varying influence of different
parameters on personalized lane-change behavior may lead to
insufficiently distinguishable differences among samples. To
further evaluate the proposed framework’s ability to distin-
guish not only different driving styles but also different style
intensities within the same driving style, three representative
control parameter sets with different intensity levels are se-
lected from each driving style.
To rank the samples within each driving-style category,
the behavioral features are first normalized using z-score
standardization to eliminate the influence of different units and
numerical scales
zij=xij−µj
σj,(10)
wherex ijdenotes the original value of thej-th behavioral
feature of thei-th sample, andµ jandσ jare the mean
and standard deviation of thej-th feature over all valid
samples, respectively. After standardization, thei-th sample
is represented by the standardized behavioral feature vector
zi= [zi1, zi2, . . . , z ip],(11)
wherepis the number of behavioral features.
Since each standardized feature describes the deviation of
a sample from the average value of that feature, the distance
from a sample to the origin of the standardized feature space is
used to quantify its overall deviation from average lane-change
behavior. This distance is defined as the style-intensity score
Si=∥z i∥2=q
z2
i1+z2
i2+···+z2
ip.(12)
A largerS iindicates a stronger deviation from average lane-
change behavior and is therefore interpreted as a higher style
intensity.Within each driving-style category, samples are ranked
according toS i. Three representative samples are then selected
to correspond to low, medium, and high intensity levels.
IV. EXPERIMENT
The experiments are designed to evaluate the effectiveness
of the RAG dataset and the lane-change behavior differences
under the nine style-parameter configurations defined in Ta-
ble V. As summarized in Table IX, the experiments consist
of an LLM-based preference interpretation test and a system-
level behavior test. The interpretation test uses explicit, mixed,
and implicit command sets, where the mixed set contains
50% explicit and 50% implicit commands. The system-level
test evaluates the resulting vehicle behaviors under different
driving styles and intensity levels. For each test run, the lane-
change events along the route are extracted and statistically
evaluated using lane-change time, maximum lateral accelera-
tion, and accumulated lateral-acceleration response.
In addition to the quantitative experiments, a complete
usage example is provided online [Video] to demonstrate the
workflow from natural-language command input, LLM-based
style inference, and parameter selection to co-simulation-based
behavior execution.
TABLE IX
EXPERIMENTALDESIGN
Part No. Test Condition Evaluation
LLM
interpretation
test1 explicit samples
Accuracy 2 mixed samples
3 implicit samples
System-level
behavior test1 Aggressive style, L1
TLC,
max|a y|,
Eay2 Aggressive style, L2
3 Aggressive style, L3
4 Normal style, L1
5 Normal style, L2
6 Normal style, L3
7 Conservative style, L1
8 Conservative style, L2
9 Conservative style, L3
A. Performance Comparison of Zero-Shot and RAG-Based
Preference Interpretation
Three test settings are designed to evaluate LLM-based
driving-style classification under different levels of command
explicitness. Each test set contains 80 samples. As summarized
in Table X, RAG-based inference improves classification ac-
curacy across all three test settings. For the explicit command
set, both zero-shot and RAG-based inference achieve high
accuracy, indicating that explicit style commands can already
be accurately interpreted by most LLMs. For the mixed
command set, RAG provides consistent gains, showing its
benefit when explicit and implicit expressions appear together.
The largest improvement is observed for the implicit command
set, where the average gain reaches 10.2 percentage points,
indicating that retrieved examples are particularly helpful for
interpreting indirect user commands.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 9
TABLE X
ACCURACYCOMPARISON ONEXPLICIT, MIXED,ANDIMPLICIT
COMMANDSETS
Set Model Zero-shot RAG Gain
ExplicitGemma-4-31B-IT 97.5% 98.8% +1.3
Qwen-2.5-7B-Instruct 97.5% 98.8% +1.3
Llama-3.1-8B-Instruct 87.5% 95.0% +7.5
Ministral-8B-2512 98.8% 98.8% +0.0
Gemma-3-4B-IT 80.0% 88.8% +8.8
Qwen3-8B 97.5% 98.8% +1.3
Gemma-3-12B-IT 96.2% 100.0% +3.8
Average 93.6% 97.0% +3.4
MixedGemma-4-31B-IT 93.8% 95.0% +1.2
Qwen-2.5-7B-Instruct 86.2% 92.5% +6.3
Llama-3.1-8B-Instruct 86.2% 91.2% +5.0
Ministral-8B-2512 86.2% 87.5% +1.3
Gemma-3-4B-IT 81.2% 91.2% +10.0
Qwen3-8B 81.2% 87.5% +6.3
Gemma-3-12B-IT 83.8% 96.2% +12.4
Average 85.5% 91.6% +6.1
ImplicitGemma-4-31B-IT 83.8% 86.2% +2.4
Qwen-2.5-7B-Instruct 71.2% 87.5% +16.3
Llama-3.1-8B-Instruct 71.2% 80.0% +8.8
Ministral-8B-2512 67.5% 77.5% +10.0
Gemma-3-4B-IT 68.8% 78.8% +10.0
Qwen3-8B 67.5% 81.2% +13.7
Gemma-3-12B-IT 72.5% 82.5% +10.0
Average 71.8% 82.0% +10.2
Gain denotes the accuracy difference between RAG and zero-shot prompting, measured
in percentage points.
B. System-Level Evaluation of Personalized Lane-Change Be-
havior
The experiments are conducted on a road network recon-
structed from a motorway in Austria [48], [49], as shown
in Figure 6. Background traffics are generated using Car-
Maker’s stochastic traffic-flow function, where vehicles are
randomly distributed along predefined routes according to a
relative density parameter. A total of nine experiments were
conducted, where each experiment corresponds to a complete
driving process from the starting point to the destination.
The results are analyzed from two perspectives. First, Fig-
ure 7 groups the extracted lane-change events by driving
style and presents the statistical distributions of aggressive,
Fig. 6. Road network used for the experiments.
Fig. 7. Lane-change KPI distributions across driving styles.
normal, and conservative behaviors. Second, Table XI further
summarizes the results by both driving style and intensity
level, enabling a comparison among L1, L2, and L3 within
each driving style.
TABLE XI
LANE-CHANGEKPIS ACROSS DRIVING STYLES ON THEA2MOTORWAY.↑:
HIGHER VALUES INDICATE A MORE INTENSIVE LANE-CHANGE DYNAMIC
PROCESS;↓:LOWER VALUES INDICATE A GENTLER LANE-CHANGE
DYNAMIC PROCESS.
StyleStrength
GradeLane-change performance
TLC↓[s]max|a y| ↑ Eay↑
AggressiveL12.4790 3.5362 17.6299
L2 3.3800 3.3918 16.4673
L3 3.7150 3.3688 14.1935
NormalL1 3.7675 3.1043 12.0857
L2 4.2017 2.9980 10.4531
L3 4.4467 2.9901 9.2197
ConservativeL1 5.1692 2.8463 7.6423
L2 5.5450 2.6409 4.1332
L3 6.1725 2.0355 2.4949
Figure 7 shows clear behavioral differences among the three
driving styles. The aggressive style generally presents shorter
lane-change time and stronger lateral dynamic responses,
including larger lateral acceleration intensity, wider steering-
wheel-angle distributions, and higher yaw-rate values. In con-
trast, the conservative style shows longer lane-change time and
more concentrated distributions around smaller lateral accel-
eration, steering-wheel angle, and yaw rate values, indicating
smoother and more stable lane-change behavior. The normal
style generally lies between these two extremes. Table XI
presents the statistical analysis of representative personalized
lane-change KPIs. As the style shifts from aggressive to
conservative, the lane-change time increases, while both the
maximum lateral acceleration and lateral acceleration intensity
decrease, indicating smoother maneuvers.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 10
V. CONCLUSION
This study proposed an LLM-enabled personalized lane-
change framework for highway automated driving. The frame-
work maps natural-language driver commands to personal-
ized driving styles and executable planning parameters within
the Apollo planning pipeline. The results show that differ-
ent parameter settings generate distinguishable lane-change
behaviors in lane-change time, lateral acceleration, steering
response, and yaw-rate characteristics.
To improve the interpretation of implicit preference com-
mands, a RAG dataset was constructed for retrieval-augmented
generation. Three test settings were used to compare zero-
shot and RAG-based inference across multiple LLMs. The
results show that RAG provides consistent improvements,
especially for implicit preference commands and relatively
smaller models. This indicates that retrieved implicit examples
can provide useful semantic guidance for mapping indirect
natural-language expressions to personalized driving styles.
Future work will extend the proposed framework to broader
traffic conditions, more complex scenarios, and richer driver
preference expressions. The current framework represents each
driving style using three discrete intensity levels. We can
explore continuous style-intensity modeling, so that person-
alized driving parameters can be adjusted more smoothly and
precisely according to individual user preferences. Moreover,
driver-in-the-loop experiments will be conducted to validate
whether the generated driving styles align with human subjec-
tive preferences and comfort expectations.
APPENDIXA
PERSONALIZEDLONGITUDINALPLANNINGANALYSIS
Personalized longitudinal behavior during lane-change sce-
narios is achieved by regulating the trade-off among ve-
locity tracking, smoothness, and responsiveness within the
final speed optimization stage. Specifically, personalization is
realized by tuning the cost-function weights of the piecewise-
jerk speed optimizer, while preserving the original structure
of Apollo’s LaneFollow planning pipeline.
Fig. 8. The longitudinal speed planning process in Apollo.
Figure 8 illustrates the longitudinal speed planning pipeline
in Apollo. The pipeline first constructs spatio-temporal (ST)
boundaries and obstacle-related longitudinal decisions, such
asstop,follow,yield, andovertake. A coarse speed
profile is generated by dynamic programming, and the final
smooth speed profile is obtained by thePiecewise Jerk Speed
optimizer, which solves a constrained quadratic optimization
problem using Operator Splitting Quadratic Program (OSQP)
solver. In this work, the selected longitudinal parameters are
mainly associated with this optimization process and are usedto adjust acceleration behavior, longitudinal progress, and
lane-change duration.
According to this pipeline, the piecewise-jerk speed op-
timizer computes the optimal trajectory within the feasible
region determined by upstream modules. The constraints of
the optimization problem can be categorized into feasibility
constraints, dynamic consistency constraints, and smoothness-
related constraints. The longitudinal state at time stept i
is defined as(s i, vi, ai), wheres i,vi, anda idenote the
longitudinal position, velocity, and acceleration.
The feasibility constraints define the admissible longitudinal
motion region. The ST boundaries impose the positional
constraint
smin(ti)≤s i≤smax(ti),(13)
wheres min(ti)ands max(ti)are determined by obstacle-
induced ST boundaries after applying longitudinal decisions.
The velocity is constrained by
0≤v i≤vlimit
i,(14)
wherevlimit
i denotes the allowable upper speed bound at
time stept i, which is determined by the local road speed
limit, global planning speed bound, and scenario-dependent
constraints. In addition, the acceleration is bounded by vehicle
dynamic limits
amin≤ai≤amax,(15)
wherea min anda max denote the minimum and maximum
longitudinal acceleration, determined by the vehicle’s physical
limits. These bounds define the admissible dynamic envelope
of the vehicle and indirectly influence the aggressiveness of
the generated speed profile.
The dynamic consistency constraints enforce the physical
relationships among these states
vi+1−vi−∆t
2(ai+ai+1) = 0,(16)
si+1−si−∆t v i−∆t2
3ai−∆t2
6ai+1= 0,(17)
where∆t= 0.1 sis the discretization interval. These
constraints ensure that the optimized trajectory is physically
realizable.
The smoothness constraint is imposed through the jerk
bound
jmin≤ai+1−ai
∆t≤jmax,(18)
which limits the rate of change of acceleration and improves
ride comfort and control stability.
Within this constrained feasible region, the longitudinal
trajectory is obtained by solving a quadratic optimization
problem. The objective function is formulated as
min
{si,vi,ai}J=N−1X
i=0ws 
si−sref
i2+N−1X
i=0wv 
vi−vref
i2
+N−1X
i=0waa2
i+N−2X
i=0wjai+1−ai
∆t2
.
(19)

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 11
The reference positionsref
iis obtained by temporally sam-
pling the upstream coarse speed profile generated by the
heuristic optimizer. The reference velocity is defined as
vref
i= min 
vlimit
i, vcruise
,(20)
wherev cruise is the desired cruising speed.
APPENDIXB
PERSONALIZEDLATERALPLANNINGANALYSIS
To enable personalized lane-change behaviors, this work
focuses on Apollo’s piecewise-jerk lateral path optimizer
and adjusts its geometric constraints, derivative bounds, and
objective-function weights. Figure 9 illustrates the lane-change
trajectory generation process in Apollo. The pipeline first
updates the lane-change state and internal lane-change status
through theLane Change State Updatemodule. TheTrigger
Checkmodule then evaluates whether the routing-based lane-
change request satisfies the required feasibility conditions.
Once the request is accepted, theInitial State Extraction
module obtains the initial Frenet state, while theFeasible
Lateral Region Constructionmodule builds the feasible lateral
corridor based on self-lane boundaries, forward zones, obstacle
constraints, and a vehicle-dependent safety margin. Based on
these inputs, theLateral Path Optimizationmodule formulates
and solves a constrained quadratic optimization problem using
OSQP. Finally, the generated path is checked by thePath
Assessmentmodule to ensure geometric validity and road
feasibility. In this work, the selected lateral parameters are
mainly associated with the feasible lateral region, derivative
bounds, and objective-function weights, and are used to adjust
lane-change smoothness, lateral aggressiveness, and spatial
feasibility.
In this pipeline, the personalized parameters considered are
mainly associated with the feasible lateral corridor, derivative
bounds, and the objective-function weights of the optimizer.
The feasible lateral corridor constrains the lateral offset at each
discretized station point
lmin(ξi;δadc)≤l i≤lmax(ξi;δadc),(21)
whereξ i=ξ 0+i∆ξis the discretized reference-line arc
length,l i=l(ξ i)is the lateral offset, andδ adcdenotes the
safety margin used to adjust the feasible lateral region.
In addition to the lateral corridor, the optimized path is
constrained by derivative bounds
|dli| ≤dl max,(22)
Fig. 9. The lane-change trajectory generation process in Apollo.ddli∈[ddlmin
i, ddlmax
i],(23)
ddli+1−ddl i
∆ξ∈[dddlmin
i, dddlmax
i],(24)
wheredl i=l′(ξi),ddl i=l′′(ξi), and(ddl i+1−ddl i)/∆ξap-
proximate the first-order, second-order, and third-order lateral
derivatives with respect to the reference-line arc length.
The dynamic consistency constraints enforce the geometric
relationships among consecutive lateral states
dli+1−dli−∆ξ
2(ddli+ddl i+1) = 0,(25)
li+1−li−∆ξ dl i−∆ξ2
3ddli−∆ξ2
6ddli+1= 0,(26)
where∆ξis the discretization interval along the reference-
line arc length. These constraints ensure that the optimized
lateral trajectory remains geometrically consistent in the Frenet
frame.
Given the above constraints, the lateral trajectory is obtained
by solving a constrained quadratic optimization problem. For
clarity, the main tunable terms used for personalization are
written as
min
{li, dli, ddl i}J=N−1X
i=0wll2
i+N−1X
i=0wdldl2
i+N−1X
i=0wddlddl2
i
+N−2X
i=0wdddlddli+1−ddl i
∆ξ2
.
(27)
REFERENCES
[1] J. Ge, C. Chang, J. Zhang, L. Li, X. Na, Y . Lin, L. Li, and F.-Y .
Wang, “Llm-based operating systems for automated vehicles: A new
perspective,”IEEE Transactions on Intelligent Vehicles, vol. 9, no. 4,
pp. 4563–4567, 2024.
[2] M. Hasenj ¨ager, M. Heckmann, and H. Wersing, “A survey of personal-
ization for advanced driver assistance systems,”IEEE Transactions on
Intelligent Vehicles, vol. 5, no. 2, pp. 335–344, 2020.
[3] D. Yi, J. Su, L. Hu, C. Liu, M. Quddus, M. Dianati, and W.-H. Chen,
“Implicit personalization in driving assistance: State-of-the-art and open
issues,”IEEE Transactions on Intelligent Vehicles, vol. 5, no. 3, pp.
397–413, 2019.
[4] X. Liao, Z. Zhao, M. J. Barth, A. Abdelraouf, R. Gupta, K. Han, J. Ma,
and G. Wu, “A review of personalization in driving behavior: Dataset,
modeling, and validation,”IEEE Transactions on Intelligent Vehicles,
2024.
[5] D. Bevly, X. Cao, M. Gordon, G. Ozbilgin, D. Kari, B. Nelson,
J. Woodruff, M. Barth, C. Murray, A. Kurtet al., “Lane change and
merge maneuvers for connected and automated vehicles: A survey,”
IEEE Transactions on Intelligent Vehicles, vol. 1, no. 1, pp. 105–120,
2016.
[6] M. L. Schrum, E. Sumner, M. C. Gombolay, and A. Best, “Maveric:
A data-driven approach to personalized autonomous driving,”IEEE
Transactions on Robotics, vol. 40, pp. 1952–1965, 2024.
[7] V . A. Butakov and P. Ioannou, “Personalized driver/vehicle lane change
models for ADAS,”IEEE Transactions on Vehicular Technology, vol. 64,
no. 10, pp. 4422–4431, 2014.
[8] S. Yang, H. Zheng, J. Wang, and A. El Kamel, “A personalized human-
like lane-changing trajectory planning method for automated driving
system,”IEEE Transactions on Vehicular Technology, vol. 70, no. 7,
pp. 6399–6414, 2021.
[9] C. Huang, H. Huang, P. Hang, H. Gao, J. Wu, Z. Huang, and C. Lv,
“Personalized trajectory planning and control of lane-change maneuvers
for autonomous driving,”IEEE Transactions on Vehicular Technology,
vol. 70, no. 6, pp. 5511–5523, 2021.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 12
[10] Y . Ma, X. Cao, W. Ye, C. Cui, K. Mei, and Z. Wang, “Learning
autonomous driving tasks via human feedbacks with large language
models,” inFindings of the Association for Computational Linguistics:
EMNLP 2024, 2024, pp. 4985–4995.
[11] Z. Song, M. Lv, T. Ren, C. J. Xue, J.-M. Wu, and N. Guan, “Autoware.
flex: Human-instructed dynamically reconfigurable autonomous driving
systems,” in2025 IEEE 31st International Conference on Embedded
and Real-Time Computing Systems and Applications (RTCSA). IEEE,
2025, pp. 1–11.
[12] C. Cui, Y . Ma, X. Cao, W. Ye, and Z. Wang, “Drive as you speak:
Enabling human-like interaction with large language models in au-
tonomous vehicles,” inProceedings of the IEEE/CVF Winter Conference
on Applications of Computer Vision, 2024, pp. 902–909.
[13] L. Wen, D. Fu, X. Li, X. Cai, T. Ma, P. Cai, M. Dou, B. Shi, L. He, and
Y . Qiao, “Dilu: A knowledge-driven approach to autonomous driving
with large language models,” inInternational Conference on Learning
Representations, 2024, pp. 34 503–34 522.
[14] Y . Gao, D. Hua, M. Piccinini, F. R. Sch ¨afer, K. Moller, L. Li, and
J. Betz, “Stylevla: Driving style-aware vision language action model for
autonomous driving,”arXiv preprint arXiv:2603.09482, 2026.
[15] G. Kou, F. Jia, W. Mao, Y . Liu, Y . Zhao, Z. Zhang, O. Yoshie, T. Wang,
Y . Li, and X. Zhang, “Padriver: Towards personalized autonomous
driving,” in2025 International Joint Conference on Neural Networks
(IJCNN). IEEE, 2025, pp. 1–8.
[16] M. Hasenj ¨ager, M. Heckmann, and H. Wersing, “A survey of personal-
ization for advanced driver assistance systems,”IEEE Transactions on
Intelligent Vehicles, vol. 5, no. 2, pp. 335–344, 2019.
[17] ApolloAuto, “Apollo: An open autonomous driving platform,” https:
//github.com/ApolloAuto/apollo, gitHub repository. Accessed: Jun. 21,
2026.
[18] N. Xie, R. Yu, W. Sun, S. Qiu, K. Zhong, M. Xu, G. Wu, and
Y . Yang, “Personalized forward collision warning model with learning
from human preferences,”Accident Analysis & Prevention, vol. 208, p.
107791, 2024.
[19] Y . Wang, Z. Wang, K. Han, P. Tiwari, and D. B. Work, “Personalized
adaptive cruise control via gaussian process regression,” in2021 IEEE
International Intelligent Transportation Systems Conference (ITSC).
IEEE, 2021, pp. 1496–1502.
[20] I. Koglbauer, J. Holzinger, A. Eichberger, and C. Lex, “Drivers’ inter-
action with adaptive cruise control on dry and snowy roads with various
tire-road grip potentials,”Journal of Advanced Transportation, vol. 2017,
no. 1, p. 5496837, 2017.
[21] M. Delmas, V . Camps, and C. Lemercier, “Personalizing automated driv-
ing speed to enhance user experience and performance in intermediate-
level automated driving,”Accident Analysis & Prevention, vol. 199, p.
107512, 2024.
[22] M. Zhang, J. Li, N. Li, E. Kang, and K. Tei, “User-driven adaptation:
Tailoring autonomous driving systems with dynamic preferences,” in
Extended Abstracts of the CHI Conference on Human Factors in
Computing Systems, 2024, pp. 1–8.
[23] J. Hu, M. Lei, H. Wang, Z. Liu, and F. Yang, “Accelerating the evolution
of personalized automated lane change through lesson learning,”IEEE
Transactions on Intelligent Transportation Systems, 2025.
[24] C. Cui, Y . Ma, S.-Y . Park, Z. Yang, Y . Zhou, P. Liu, J. Lu, J. Peng,
J. Zhang, R. Zhanget al., “Large language models for autonomous
driving—concept, review, benchmark, experiments, and future trends,”
Proceedings of the IEEE, 2026.
[25] Z. Yang, X. Jia, H. Li, and J. Yan, “Llm4drive: A survey of
large language models for autonomous driving,”arXiv preprint
arXiv:2311.01043, 2023.
[26] X. Zhou, M. Liu, E. Yurtsever, B. L. Zagar, W. Zimmer, H. Cao, and
A. C. Knoll, “Vision language models in autonomous driving: A survey
and outlook,”IEEE Transactions on Intelligent Vehicles, 2024.
[27] Y . Zhao, J. Zhou, D. Bi, T. Mihalj, J. Hu, and A. Eichberger, “A
survey on the application of large language models in scenario-based
testing of automated driving systems,”IEEE Transactions on Intelligent
Transportation Systems, 2026.
[28] C. Cui, Z. Yang, Y . Zhou, Y . Ma, J. Lu, L. Li, Y . Chen, J. Panchal,
and Z. Wang, “Personalized autonomous driving with large language
models: Field experiments,” in2024 IEEE 27th International Conference
on Intelligent Transportation Systems (ITSC). IEEE, 2024, pp. 20–27.
[29] Z. Xu, T. Chen, Z. Huang, Y . Xing, and S. Chen, “Personalizing
driver agent using large language models for driving safety and smarter
human–machine interactions,”IEEE intelligent transportation Systems
magazine, 2025.[30] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V . Koltun, “Carla:
An open urban driving simulator,” inConference on robot learning.
PMLR, 2017, pp. 1–16.
[31] J. Ge, C. Chang, J. Zhang, L. Li, X. Na, Y . Lin, L. Li, and F.-Y .
Wang, “Llm-based operating systems for automated vehicles: A new
perspective,”IEEE Transactions on Intelligent Vehicles, vol. 9, no. 4,
pp. 4563–4567, 2024.
[32] Y . Yang, Q. Zhang, C. Li, D. S. Marta, N. Batool, and J. Folkesson,
“Human-centric autonomous systems with llms for user command
reasoning,” inProceedings of the IEEE/CVF Winter Conference on
Applications of Computer Vision, 2024, pp. 988–994.
[33] H. Liao, H. Shen, Z. Li, C. Wang, G. Li, Y . Bie, and C. Xu, “Gpt-
4 enhanced multimodal grounding for autonomous driving: Leveraging
cross-modal attention with large language models,”Communications in
Transportation Research, vol. 4, p. 100116, 2024.
[34] P. Roy, S. Perisetla, S. Shriram, H. Krishnaswamy, A. Keskar, and
R. Greer, “doscenes: An autonomous driving dataset with natural lan-
guage instruction for human interaction and vision-language navigation,”
in2025 IEEE 28th International Conference on Intelligent Transporta-
tion Systems (ITSC). IEEE, 2025, pp. 1651–1658.
[35] T. Deruyttere, S. Vandenhende, D. Grujicic, L. Van Gool, and M. F.
Moens, “Talk2car: Taking control of your self-driving car,” inPro-
ceedings of the 2019 conference on empirical methods in natural
language processing and the 9th international joint conference on
natural language processing (EMNLP-IJCNLP), 2019, pp. 2088–2098.
[36] D. Wu, W. Han, Y . Liu, T. Wang, C.-z. Xu, X. Zhang, and J. Shen,
“Language prompt for autonomous driving,” inProceedings of the AAAI
conference on artificial intelligence, vol. 39, no. 8, 2025, pp. 8359–8367.
[37] H. Shao, Y . Hu, L. Wang, G. Song, S. L. Waslander, Y . Liu, and H. Li,
“Lmdrive: Closed-loop end-to-end driving with large language models,”
inProceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2024, pp. 15 120–15 130.
[38] Sentence Transformers, “sentence-transformers/all-MiniLM-L6-v2,”
[Online]. Available: https://huggingface.co/sentence-transformers/
all-MiniLM-L6-v2, 2021, accessed: Jun. 15, 2026.
[39] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazar ´e,
M. Lomeli, L. Hosseini, and H. J ´egou, “The faiss library,” 2024.
[40] J. Johnson, M. Douze, and H. J ´egou, “Billion-scale similarity search
with GPUs,”IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535–
547, 2019.
[41] IPG Automotive GmbH, “CarMaker: The simulation solution for
virtual test driving,” https://www.ipg-automotive.com/solutions/
product-portfolio/carmaker, accessed: Jun. 21, 2026.
[42] D. Bi, Y . Zhao, Z. Gu, T. Mihalj, J. Hu, and A. Eichberger, “Toward
a full-stack co-simulation platform for testing of automated driving
systems,” in2025 IEEE 28th International Conference on Intelligent
Transportation Systems (ITSC), 2025, pp. 980–986.
[43] Y . Zhao, W. Xiao, T. Mihalj, J. Hu, and A. Eichberger, “Chat2scenario:
Scenario extraction from dataset through utilization of large language
model,” in2024 IEEE Intelligent Vehicles Symposium (IV), 2024, pp.
559–566.
[44] K. Wang, Y . Yang, S. Wang, and Z. Shi, “Research on car-following
model considering driving style,”Mathematical Problems in Engineer-
ing, vol. 2022, no. 1, p. 7215697, 2022.
[45] M. Werling, J. Ziegler, S. Kammel, and S. Thrun, “Optimal trajectory
generation for dynamic street scenarios in a frenet frame,” in2010 IEEE
international conference on robotics and automation. IEEE, 2010, pp.
987–993.
[46] B. Zhu, S. Yan, J. Zhao, and W. Deng, “Personalized lane-change as-
sistance system with driver behavior identification,”IEEE Transactions
on Vehicular Technology, vol. 67, no. 11, pp. 10 293–10 306, 2018.
[47] J. B. McQueen, “Some methods of classification and analysis of mul-
tivariate observations,” inProc. of 5th Berkeley Symposium on Math.
Stat. and Prob., 1967, pp. 281–297.
[48] D. Nalic, A. Eichberger, G. Hanzl, M. Fellendorf, and B. Rogic,
“Development of a co-simulation framework for systematic generation
of scenarios for testing and validation of automated driving systems,” in
2019 IEEE Intelligent Transportation Systems Conference (ITSC), 2019,
pp. 1895–1901.
[49] Y . Zhao, X. Zhang, T. Mihalj, M. Schabauer, L. Putzer, E. Reichmann-
Blaga, ´A. Borony ´ak, A. R ¨ovid, G. So ´os, P. Zhang, L. Xiong, J. Hu, and
A. Eichberger, “A communication-latency-aware co-simulation platform
for safety and comfort evaluation of cloud-controlled icvs,”IEEE
Internet of Things Journal, vol. 13, no. 4, pp. 6217–6229, 2026.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 13
Dong Bireceived the M.Sc. degree in Mechanical
Engineering from Hubei University of Automotive
Technology, Shiyan, China, in 2017. From 2017 to
2024, he served as a Lecturer at the same university,
where he was involved in research and teaching
in the areas of advanced driver assistance systems
and functional safety testing for automated driving
systems. Since October 2024, he has been pursuing
the Ph.D. degree in the field of automated driving at
Graz University of Technology, Graz, Austria.
Yongqi Zhaoreceived the bachelor’s degree from
the China University of Petroleum (East China),
Qingdao, China, in 2019, and the master’s degree
from Technical University of Braunschweig, Braun-
schweig, Germany, in 2022. He is currently pursuing
the Ph.D. degree with the Institute of Automotive
Engineering, Graz University of Technology, Graz,
Austria, with a research focus on virtual testing of
automated driving systems. While pursuing his mas-
ter’s degree, he gained practical experience through
internships with Momenta, Stuttgart, Germany, and
V olkswagen Group, Wolfsburg, Germany.
Paul Kovacevicreceived his bachelor’s degree
from Graz University of Technology, Graz, Austria,
in 2023, and his master’s degree from the same
institution in 2025. He is currently pursuing his
Ph.D. at the Institute of Automotive Engineering,
where his research focuses on Cooperative Intel-
ligent Transportation Systems (C-ITS). Alongside
his academic career, he gained practical industry
experience at A VL List and Magna International.
Tomislav Mihaljreceived his degree in Mechan-
ical Engineering from the University of Zagreb,
Croatia, in 2014, and earned his PhD from Graz
University of Technology, Austria, in 2024. From
2014 to 2019, he worked as a Research Engineer
at Virtual Vehicle, Graz, Austria, where he focused
on the mechanical efficiency of combustion engines,
vibration analysis, and crack propagation in wheel-
rail contact. Between 2019 and 2024, he served as a
University Project Assistant at the Institute of Auto-
motive Engineering, Graz University of Technology,
concentrating on the virtual verification of automated driving systems. Since
2024, he has been a Postdoctoral Researcher at the same institute, where
he focuses on the verification and validation of driver assistance systems
and supervises related research projects. He has authored or co-authored 12
peer-reviewed publications on rail vehicles and driver assistance systems. He
has also contributed to several national and EU-funded projects related to
automated driving.
Ji Zhouobtained the master’s degree (M.Sc)
in Automotive Engineering from Jilin University,
Changchun, China, in 2009. He is currently pur-
suing a Ph.D. degree at the Institute of Automo-
tive Engineering, Graz University of Technology,
Graz, Austria. He has worked in Ricardo, former
PSA group and now in Stellantis group for more
than 15 years. He has rich technical experience in
powertrain control Software development, validation
& calibration, vehicle Electrical & Electronic (EE)
architecture integration validation, and EE product
industrialization. His current research interests are mainly in the advanced
methodology of Automated Driving Systems integration and validation. He
has published 4 technical papers indexed by EI Compendex, 2 patents
officially granted, and 11 other patents currently under publication.
Jiayuan Gongreceived his PhD degree in Acoustics
from University of Chinese Academy of Sciences,
and completed postdoctoral research at Harbin Engi-
neering University, is an associate professor of Hubei
University of Automotive Technology, and the vice
dean of School of Intelligent Connected Vehicle. He
is a Senior Member of the China Computer Federa-
tion (CCF), serving as an Executive Member of CCF
Intelligent Vehicle Committee, and a Member of the
Wuhan Chapter. Additionally, he is a Committee
Member of the Artificial Intelligence Division of the
China Society of Automotive Engineers and a Committee Member of the
Embodied Intelligence Committee of the Chinese Association for Artificial
Intelligence. He leads the Electro-Electronic Information Architecture &
Vehicle-Road-Cloud Collaboration Team, and serves as Deputy Director of
the Shiyan Key Laboratory of Air-Ground Swarm Collaborative Intelligence.
His main research interests include CPS, HPC, Data Science, and AI.
Arno Eichberger(Member, IEEE) received the
degree in mechanical engineering and the Ph.D.
degree (Hons.) in technical sciences from the Graz
University of Technology, Graz, Austria, in 1995 and
1998, respectively.
From 1998 to 2007, he was employed with Magna
Steyr Fahrzeugtechnik AG&Company, Graz, where
he dealt with different aspects of active and passive
safety. Since 2007, he has been working with the
Institute of Automotive Engineering, Graz Univer-
sity of Technology, dealing with driver assistance
systems, vehicle dynamics, and suspensions. Since 2012, he has been an
Associate Professor holding a “venia docendi” of automotive engineering.
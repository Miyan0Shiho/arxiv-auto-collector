# A Novel Integrated Architecture for Intent Based Approach and Zero Touch Networks

**Authors**: Neelam Gupta, Dibakar Das, Tamizhelakkiya K, Uma Maheswari Natarajan, Sharvari Ravindran, Komal Sharma, Jyotsna Bapat, Debabrata Das

**Published**: 2025-09-25 11:35:27

**PDF URL**: [http://arxiv.org/pdf/2509.21026v1](http://arxiv.org/pdf/2509.21026v1)

## Abstract
The transition to Sixth Generation (6G) networks presents challenges in
managing quality of service (QoS) of diverse applications and achieving Service
Level Agreements (SLAs) under varying network conditions. Hence, network
management must be automated with the help of Machine Learning (ML) and
Artificial Intelligence (AI) to achieve real-time requirements. Zero touch
network (ZTN) is one of the frameworks to automate network management with
mechanisms such as closed loop control to ensure that the goals are met
perpetually. Intent- Based Networking (IBN) specifies the user intents with
diverse network requirements or goals which are then translated into specific
network configurations and actions. This paper presents a novel architecture
for integrating IBN and ZTN to serve the intent goals. Users provides the
intent in the form of natural language, e.g., English, which is then translated
using natural language processing (NLP) techniques (e.g., retrieval augmented
generation (RAG)) into Network Intent LanguagE (Nile). The Nile intent is then
passed on to the BiLSTM and Q-learning based ZTN closed loop framework as a
goal which maintains the intent under varying network conditions. Thus, the
proposed architecture can work autonomously to ensure the network performance
goal is met by just specifying the user intent in English. The integrated
architecture is also implemented on a testbed using OpenAirInterface (OAI).
Additionally, to evaluate the architecture, an optimization problem is
formulated which evaluated with Monte Carlo simulations. Results demonstrate
how ZTN can help achieve the bandwidth goals autonomously set by user intent.
The simulation and the testbed results are compared and they show similar
trend. Mean Opinion Score (MOS) for Quality of Experience (QoE) is also
measured to indicate the user satisfaction of the intent.

## Full Text


<!-- PDF content starts -->

This is a preprint version. The published version will be available in the proceedings of IEEE FNWF 2025 on IEEEXplore subsequently.
A Novel Integrated Architecture for Intent Based
Approach and Zero Touch Networks
Neelam Gupta1, Dibakar Das1, Tamizhelakkiya K1,Uma Maheswari Natarajan1, Sharvari Ravindran1, Komal Sharma2,
Jyotsna Bapat1, and Debabrata Das1
1Networking and Communication Research Lab, IIIT Bangalore, India
2Toshiba Software (India) Private Limited, Bangalore, India
Email: {neelam.gupta, dibakar.das, tamizhelakkiya.k, umamaheswari.natarajan, sharvari.r, jbapat, ddas}@iiitb.ac.in,
komal.sharma@toshiba-tsip.com
Abstract —The transition to Sixth Generation (6G) networks
presents challenges in managing quality of service (QoS) of
diverse applications and achieving Service Level Agreements
(SLAs) under varying network conditions. Hence, network man-
agement must be automated with the help of Machine Learning
(ML) and Artiﬁcial Intelligence (AI) to achieve real-time require-
ments. Zero touch network (ZTN) is one of the frameworks to
automate network management with mechanisms such as closed
loop control to ensure that the goals are met perpetually. Intent-
Based Networking (IBN) speciﬁes the user intents with diverse
network requirements or goals which are then translated into
speciﬁc network conﬁgurations and actions. This paper presents
a novel architecture for integrating IBN and ZTN to serve the
intent goals. Users provides the intent in the form of natural
language, e.g., English, which is then translated using natural
language processing (NLP) techniques (e.g., retrieval augmented
generation (RAG)) into Network Intent LanguagE (Nile) . The
Nile intent is then passed on to the BiLSTM and Q-learning
based ZTN closed loop framework as a goal which maintains
the intent under varying network conditions. Thus, the proposed
architecture can work autonomously to ensure the network
performance goal is met by just specifying the user intent in
English. The integrated architecture is also implemented on a
testbed using OpenAirInterface (OAI) . Additionally, to evaluate
the architecture, an optimization problem is formulated which
evaluated with Monte Carlo simulations. Results demonstrate
how ZTN can help achieve the bandwidth goals autonomously
set by user intent. The simulation and the testbed results are
compared and they show similar trend. Mean Opinion Score
(MOS) for Quality of Experience (QoE) is also measured to
indicate the user satisfaction of the intent.
Keywords— zero touch network, intent based network,
integrated architecture, QoS, QoE, retrieval augmented gen-
eration, BiLSTM, Q-learning
I. I NTRODUCTION
As we move towards the 6G era [1], the research attention
shifts from simple connectivity to connected intelligence.
Future networks are expected to meet demanding require-
ments, including high reliability, mobility, ultra-fast speeds,
low latency, massive connectivity, and seamless integration
across diverse applications [2]. Achieving these ambitious
goals necessitate signiﬁcant enhancements in network ca-pabilities, driven by increased automation. The vision of
intelligent autonomous networks aims to minimize human
intervention through features, such as, self-conﬁguration, self-
learning, and self-optimization [3] [4]. Such systems can
detect issues, dynamically allocate resources, and adapt in
real time to ensure continuous operation. The ZTN concept is
designed to create an automated and intelligent infrastructure
with minimal human oversight for design, monitoring, and
optimization. This aligns with standards such as the ETSI
Zero-Touch Service Management (ZSM) framework [5].
Despite the growing adoption of AI and ML in network
automation, deploying ML models still presents challenges,
particularly in terms of scalability and real-time decision-
making. Advancements in network slicing and resource man-
agement, along with tools like AutoML [6], have simpliﬁed
ML model development for ZTN security. Adaptive network
management requires a resilient closed-loop framework with
intelligent analysis, automated actions, and real-time monitor-
ing [7]. Future networks must autonomously fulﬁll user intent,
shifting from manual conﬁgurations to IBN. It expresses goals
through Graphical User Interfaces (GUI), templates, Natural
Language Processing (NLP) [8], or intent languages like
YANG, LAI, and NEMO [9], making intent expression more
accessible, especially for non-technical users than traditional
conﬁgurations. It interprets user intents, translates them into
actionable policies and enforces user-speciﬁc conﬁgurations
to ZTN. Then, ZTN needs to continuously monitor and act
on the network to ensure alignment with these intent based
polices which lead seamless and autonomous networks.
With this motivation, this work proposes an architecture to
integrate IBN and ZTN. The intention is that once an intent
is set by the user in the IBN framework the ZTN can ensure
that the intent is served continuously and autonomously. The
proposed integrated of ZTN framework and IBN is shown in
Fig. 1. This architecture is also implemented in OAI platform
(Fig. 5). The user intent is provided in natural language, e.g.,
English. The IBN block converts it into intermediate Nile
language using a retrieval augmented generation (RAG) basedarXiv:2509.21026v1  [cs.NI]  25 Sep 2025

approach [10]. The IBN component also does basic conﬂict
detection of user intents. The Nile intent is forwarded to the
ZTN framework. The ZTN framework with its BiLSTM based
network state prediction and a Q-learning based closed loop
control maintains the user intent requirements under varying
network conditions [11]. Thus, the proposed architecture can
autonomously serve, with the user just specifying its usage
requirement in natural language English. Rest of the assurance
of the intent is taken care by the AI/ML models in the ZTN
network.
Two types of scenarios are considered to evaluate the
proposed architecture. Firstly, an in-distribution (ID) intent
scenario is considered where the ZTN framework has already
learnt the actions to maintain a certain range of intent QoS re-
quirements (e.g. bandwidth). Secondly, an out-of-distribution
(OOD) case is experimented to evaluate the performance
of the ZTN framework with a new intent QoS requirement
unknown to it. These two scenarios are measured to ﬁnd out
if the expected values of QoS parameters (e.g., bandwidth)
match the user intent requirements over a period of time
under varying network conditions. Experimental results show
how the proposed architecture is able to handle requirement
for end-to-end (E2E) throughput to serve the user intents
satisfactorily. To the best of the knowledge of the authors,
no prior work has proposed such an integrated architecture,
implemented on a testbed, and moving towards futuristic
autonomous networks.
Fig. 1. Integrated architecture of ZTN and IBN implemented on OpenAir-
Interface
The following are the main contributions of our work: 1)
We proposed a novel integrated architecture that combines
ZTN with IBN, extracting bandwidth requirement from En-
glish language user intents, and programming them into ZTN
closed loop as a goal. 2) This architecture is implemented
on OAI based testbed (Fig. 5). 3) Bandwidth variations are
introduced in the network and ZTN closed loop is trained to
handle these changing conditions to meet the E2E throughput
requirements of user intents perpetually using trafﬁc shaping
(TS) actions. 4) We formulate an optimization problem to
evaluate the proposed integrated architecture. The optimiza-
tion problem is evaluated with Monte Carlo simulations. 5)
MOS values are also measured for QoE to indicate the usersatisfaction. 6) We present a comparative analysis of the
effectiveness of optimal and suboptimal ZTN actions in both
ID and OOD user intent scenarios. Results show how the ZTN
can ensure that user intents satisfactorily.
The rest of the paper is structured as follows. We describe
our proposed novel integrated architecture in Section II. The
results and discussion of our experiments are presented in
Section III. Section IV concludes the paper conclusions with
directions for future research.
II. S YSTEM MODEL
This section describes the proposed architecture of inte-
grating IBN and ZTN. It involves translation of user intent
speciﬁed in natural language (English) to Nile syntax. This
Nile intent is then passed on to the ZTN subsystem as a goal.
ZTN closed loop job is to assure that the intent is sustained
under varying network condition (e.g. bandwidth variations).
A. Translation of Intent
This function aims to convert natural language intents into
executable Nile code using large language models (LLMs)
using the methodology explained in [10]. This approach
performs efﬁcient translation of user intents in IBN context
using the Retrieval-Augmented Generation (RAG) inspired
approach [12]. The method translates an intent in English
to Nile format using a few-shot prompting approach with
zero training and a small dataset. It also has a conﬂict
detection mechanism with previously conﬁgured intents. The
user-provided natural language intent INLis translated into
the NILE format INILE =βtarget ∈R+using a Retrieval-
Augmented Generation (RAG)-based LLM translator TLLM,
in equation 1.
INile=TLLM(INL)where INile=βtarget∈R+(1)
Where:
•INL: User intent in Natural Language
•INile: Intent translated into NILE format (e.g., bandwidth
requirement βtarget)
•TLLM(.): RAG-based intent translator
B. Network Topology
Network topology comprises a core network (CN) con-
nected with application servers. gNBs are connected with CN
and support user equipments (UEs). Variations QoS parame-
ters, i.e., bandwidth, latency, jitter, and packet loss, may be
experienced during data communications between UEs and
the application servers. These variations of parameters may
be saved as a dataset for training the ZTN closed loop control.
C. ZTN Closed Loop
The BiLSTM model is trained on network dataset to predict
network states, such as, bandwidth, for proactive decisions
and resource optimization as explained in [11] The predicted
bandwidth at time t+ 1, denoted as ˆBt+1, is generated using
a BiLSTM model fBiLSTM applied over the past ntime steps
of bandwidth Bt−n:t, with model parameters θ, in equation
(2). Predictions of the BiLSTM model are compared with the

ground truths to compute residuals, which are then used to
train an XGBoost model. This secondary model learns and
corrects the BiLSTM’s errors. The resulting hybrid model en-
hances network state prediction accuracy. Given the predicted
state St=ˆBt+1, the Q-learning agent selects an optimal
action Atusing a policy π(St) = arg max a∈AQ(St, a), and
updates the Q-value using the Bellman equation, as shown in
equations (3) and (4).
ˆBt+1=fBiLSTM (Bt−n:t;θ) (2)
where Bt−n:trepresents the sequence of bandwidth values
from time t−ntot, and θdenotes the learned parameters of
the BiLSTM model.
At=π(St) = arg max
a∈AQ(St, a) (3)
Q(St, At)←Q(St, At)+α[
Rt+γmax
a′Q(St+1, a′)−Q(St, At)]
(4)
where Q(St, a)is the Q-value representing the expected
future reward for taking action atin state St. Action Atwill
allocate bandwidth B(a)
t.
The reward at time tis denoted as Rt, which is computed
based on how well the selected action meets the intent deﬁned
byINile.
The hybrid model and Q-learning model try to select
actions based on predicted network states [13]. Each state
reﬂects network conditions, allowing ZTN to select optimal
conﬁgurations as actions and proactively enhance performance
perpetually in a closed loop.
D. Extraction of Bandwidth from Nile Intent
The Nile intent acts as a guiding policy for the ZTN
framework to function. For bandwidth requirements, using
the speciﬁed clause: setbandwidth′max′, Bi,′kbps′where
Biindicates the expected bandwidth of the user intent. In
this approach, we extract the value of Bifrom the Nile intent
using regular expressions and is then set as a goal for the ZTN
system. We considered two different goal thresholds to see
how the system performs. The ID scenario reﬂects expected
or previously observed trafﬁc patterns, and the OOD scenario
represents unexpected or new network goals.
E. Evaluation of Suboptimal Vs Optimal Action
To measure the performance of Q-learning, we evaluate and
contrast system performance in two different scenarios. The
optimal action case, in which decisions are trained into the
Q-learning model, and the sub-optimal condition, in which
the ZTN system makes judgments without being completely
trained.
Rt={
+1,ifBt+1≥βtarget
−1,otherwise(5)
The reward Rtis assigned based on whether the actual
bandwidth Bt+1meets or exceeds the target intent βtarget,where Rt= +1 ifBt+1≥βtarget, and Rt=−1otherwise,
in equation (5). Let Edenote the set of network episodes
over which the model operates. For each episode e∈ E, the
objective of the closed-loop system is to maximize the number
of time steps δtfor which the bandwidth Btsatisﬁes the intent
over a period of time Tin equations (6) and (7).
∀e∈ E, maxT∑
t=1δt (6)
δt={
1,ifBt≥βtarget
0,otherwise(7)
Here, δt= 1 indicates that the actual bandwidth at time t
meets or exceeds the target intent threshold βtarget.
The optimal action Atis selected by a Q-learning policy
based on the BiLSTM-predicted network state, ensuring that
the resulting bandwidth Bt+1meets the threshold derived
from the user’s natural language intent via a RAG-based LLM
translator, in equation (8).
At=π(fBiLSTM (Bt−n:t;θ)),subject to: Bt+1≥TLLM(INL)
(8)
Where:
•At: Action selected by the Q-learning policy at time t
•fBiLSTM (Bt−n:t;θ): BiLSTM model predicting future
bandwidth based on previous ntime steps
•π(·): Q-learning policy that selects the optimal action
based on the predicted state
•Bt+1: Actual bandwidth observed after applying action
•TLLM(INL): Bandwidth threshold is determined using a
RAG-based LLM to interpret the user’s natural language
intent.
The total user experience is measured by the expected
Mean Opinion Score (MOS), which measures Quality of
Experience (QoE). This score is based on how often the
network successfully reaches the user’s intended bandwidth
threshold. This score is a human-centered measure of network
performance. It is the normalized ratio of time steps where
the actual bandwidth meets the IBN-derived target, scaled
between a minimum and maximum MOS range, in equation
(9).
The predicted MOS [14] score is computed as:
MOS pred=MOS min+(
1
TT∑
t=1δt)
·(MOS max−MOS min)
(9)
Where:
•T: Total number of time steps
•MOS max: Maximum MOS score (ID: 5 for excellent
QoE)
•MOS min: Minimum MOS score (OOD: 1 for poor QoE)

0 50 100 150 200 250 300
Time (s)260280300320340Bandwidth V ariation (Kbps)Bandwidth V ariation Over TimeFig. 2. Injected bandwidth variation between CN and AS
F . Optimization formulation
The following optimization is formulated.
Objective:
O1: min
at∈A|βtarget−B(at)
t| (10)
(11)
subject to,
c1:Bmin≤B(at)
t≤Bmax
where BminandBmax denotes the lower and upper bounds
of bandwidth provided by the underlying network.
c2:at={
aoptimal , ifˆBt∈Dtrain
asuboptimal ,ifˆBt/∈Dtrain
where Dtrainis the set for ID, and aoptimal andasuboptimal
are actions chosen during ID and OOD scenarios.
The above optimization problem is evaluated using Monte
Carlo simulations.
III. R ESULTS AND DISCUSSION
This section presents the results from the testbed based on
our system model. We will also show how the closed-loop
control mechanism works for ZTN and IBN integration based
on BiLSTM-based bandwidth prediction. This integration is
for dynamic QoS assurance of user intents. For simplicity,
we are considering only the downlink E2E bandwidth as our
QoS parameter. This allows ZTN to constantly check whether
its predicted throughput meets or exceeds the intent-driven
thresholds. Naturally, other parameters can also be taken
into account in examining our generalized system model.
We use an i9CPU and 64 GB RAM based system for this
research with OAI. The models are all implemented in python
programming language.
A. Experimental Setup
The E2E network topology uses the OAI platform which
includes a Core Network (CN) with gNB, User Plane Function
(UPF), Session Management Function (SMF), and Access
and Mobility Management Function (AMF), enabling up-
link/downlink data ﬂow from UE through the UPF to the
AS. The RAN comprises gNB and UEs. The downlink E2E
0 50 100 150 200 250 300
Time (s)280288296304312320Measured Throughput (Kbps)Throughput Over Time (300 Kbps)Fig. 3. E2E throughput due to bandwidth variation between CN and AS
without ZTN closed loop control (ID scenario)
0 50 100 150 200 250 300
Time (s)430435440445450455460Measured Throughput (Kbps)Throughput Over Time (450 Kbps)
Fig. 4. E2E throughput due to bandwidth variation between CN and AS
without ZTN closed loop control (OOD scenario)
throughput between AS and UE, recorded over 100 seconds
when bandwidth variations are applied to the interface be-
tween UPF and the AS. The collected time series data are used
to train the BiLSTM model ofﬂine. The Q-learning training
with TS happens online.
We used iperf for controlled testing within an OAI 5G
testbed environment to determine the inﬂuence of the injected
network bandwidth variation. Bandwidth variations is simu-
lated randomly for 300 randomized bandwidth values. Trafﬁc
control ( tc) command was utilized at the UPF interface to
simulate bandwidth variation by enforcing burst sizes and
variable rate bounds. The E2E throughput was observed
for bandwidth variations at the UPF, and these results are
shown in Fig. 2. Figs. 3 and 4 respectively show the E2E
throughputs for ID and OOD scenarios of 300 Kbps and
Fig. 5. Screenshot of the OAI testbed

450 kbps user intents without the ZTN functionality. These
are raw throughputs for the two scenarios. As will be seen
latter, applying ZTN closed loop actions leads to sustained
throughput as required by the respective intents.
B. Intent Translation
ID and OOD User intents in English show in Fig. 6 are
provided to the RAG based translator. The RAG based model
translates this to Nile format (Fig. 7). This Nile intent is then
passed on to the ZTN framework as a goal to be ensured by
its closed loop.
Fig. 6. In-Distribution and Out-of-Distribution Input Intents
Fig. 7. Nile Intent for in-distribution and out-of-distribution
C. ZTN Closed Loop Performance
For predicting the network state, i.e., bandwidth, E2E
throughput for ID scenario is used for training the BiLSTM
model. During inference on live downlink data transmission,
the predicted state is passed on the Q-learning based closed
loop to choose the appropriate trafﬁc shaping action so that
the user intent is met.
In the ID case, we assign the E2E throughput goal of
300 kbps from the Nile intent. As indicated in Fig. 8 for
the optimal scenario when the Q-learning is fully trained it
meets the intended throughput of 300 kbps most of the time.
Interesting, even for sub-optimal case (when the Q-learning
model has not fully learnt), there are several occasions the
intent threshold is met.
In the OOD case of 450 kbps threshold in Fig. 9, when
the Q-learning model has to ensure a new goal, even for the
optimal scenario there are several instances when the intent
is not met. This is because the current actions known to
the ZTN closed loop are not sufﬁcient to meet the intent
requirements. Hence, in this case, more exploration of the
action space is necessary. As expected, in the suboptimal
case the performance is also in similar lines with lower
performance.
0 20 40 60 80 100 120 140
Time (s)200220240260280300Throughput (Kbps)In-Distribution (300 Kbps)
Sub-Optimal
Optimal
Threshold = 300 KbpsFig. 8. E2E throughput achieved applying ZTN closed loop to meet ID user
intent of 300 kbps
0 20 40 60 80 100 120 140
Time (s)100200300400500Throughput (Kbps)Out-of-Distribution (450 Kbps)
Sub-Optimal
Optimal
Threshold = 450 Kbps
Fig. 9. E2E throughput achieved applying ZTN closed loop to meet OOD
user intent of 450 kbps
The predicted (MOS pred) are computed by taking a time-
normalized average of the number of time steps in which
actual throughput meets or exceeds the IBN deﬁned intent
bandwidth goal. Out of 148 evaluation runs in the ID scenario,
146 runs met the intent goal ( δt= 1) with the optimal actions
and 112 runs for the suboptimal ones. In contrast, in the OOD
scenario, only 47 optimal and 23 suboptimal runs satisﬁed the
same requirement. The observed MOS values for both ID and
OOD situations are summarized in Table 1.
TABLE I
MEAN OPINION SCORE COMPARISON FOR IN-DISTRIBUTION AND
OUT-OF-DISTRIBUTION SCENARIOS
Scenario Allocation Type MOS Value MOS Rating
In-Distribution (ID)Suboptimal 3.8 Good
Optimal 4.6 Excellent
Out-of-Distribution (OOD)Suboptimal 1.4 Poor
Optimal 2.2 Average
To evaluate the proposed architecture, two observations
are compared. Firstly, the formulated optimization problem is
evaluated on the testbed. Secondly, Monte Carlo simulation
are performed to evaluate the same problem. Both ID and
OOD scenarios are considered. The comparison for intent
bandwidths - 300 Kbps for ID is shown in Fig. 10, and
the same with 450 Kbps for OOD shown in Fig. 11. These
comparisons are plotted along the Y-axis, with episode indices
on the X-axis. Monte Carlo simulation results are shown with
blue line while the testbed performance shown in red. In case
of ID, the intent goal is reached whereas for the OOD there is
a divergence due to sub-optimal actions. However, the trends
are similar in both cases. This again highlights the need for

0 5 10 15 20 25 30 35 40
Episode−200−175−150−125−100−75−50−250Objective  Value f ( x
t)Monte Carlo Simulation
T e t-bed PerformanceFig. 10. Theoretical (Monte Carlo) vs. Practical (Testbed) Performance Under
In-Distribution Case
0 5 10 15 20 25 30 35 40
Episode−175−150−125−100−75−50−250Objective  Value f ( B
t)Monte Carlo Simulation
T e t-bed Performance
Fig. 11. Theoretical (Monte Carlo) vs. Practical (Testbed) Performance Under
Out-Of-Distribution Case
perpetual exploration of action space to meet new and unseen
intent goals.
IV. D ISCUSSION
The intents provide a goal that translates to ZTN decisions
and actions applied to the underlying network to meet user
requests. The BiLSTM model provides network prediction,
with optimal actions chosen by ZTN closed loop to closely
meet different bandwidth intents under varying network con-
ditions. This analysis conﬁrms that the integration of IBN-
speciﬁed bandwidth requirements into ZTN control logic
supports automated, intent-aware throughput sustenance in a
move towards autonomous networks. Though, the ID case
performs satisfactorily, perpetual exploration of the action
space is required for OOD scenario.
V. C ONCLUSION
Automation is essential for the efﬁcient operation of 6G
networks. We proposed a novel architecture that integrates
IBN and ZTN. The architecture takes an intent in English,
translates into intermediate Nile format using a RAG based
approach. The Nile intent is passed on to the ZTN component
as a goal to be ensured by the underlying network. ZTN closed
loop proactively predicts the network state e.g. bandwidth
using XGBoosted BiLSTM model and then Q-learning to
provide optimal actions to the network under varying network
condition. This architecture is implemented using OAI based
testbed. Results show how the architecture can autonomously
maintain intents in natural language both for IN and OODscenarios. Further, an optimization problem is formulated
which is evaluated through Monte Carlo simulations and
compared with the testbed results which show similar trends.
However, for OOD case, there is some amount of divergence
and provides scope of further improvement. The MOS scores
for user intent QoE also follow similar patterns for ID and
OOD scenarios.
Future work will concentrate on considering other QoS
parameters beyond bandwidth and exploration of the action
space for the OOD scenario.
ACKNOWLEDGMENT
The authors would like to thank Toshiba Software India Pvt
Ltd for sponsoring this research project.
REFERENCES
[1]L. U. Khan, I. Yaqoob, M. Imran, Z. Han, and C. S. Hong, “6g wireless
systems: A vision, architectural elements, and future directions,” IEEE
access , vol. 8, pp. 147 029–147 044, 2020.
[2]Y. Wei, M. Peng, and Y. Liu, “Intent-based networks for 6g: Insights
and challenges,” Digital Communications and Networks , vol. 6, no. 3,
pp. 270–280, 2020.
[3]H. Chergui, A. Ksentini, L. Blanco, and C. Verikoukis, “Toward zero-
touch management and orchestration of massive deployment of network
slices in 6g,” IEEE Wireless Communications , vol. 29, no. 1, pp. 86–93,
2022.
[4]E. Coronado, R. Behravesh, T. Subramanya, A. Fernandez-Fernandez,
M. S. Siddiqui, X. Costa-Pérez, and R. Riggio, “Zero touch man-
agement: A survey of network automation solutions for 5g and 6g
networks,” IEEE Communications Surveys & Tutorials , vol. 24, no. 4,
pp. 2535–2578, 2022.
[5]G. ETSI, “Zero-touch network and service management (zsm); refer-
ence architecture,” Group Speciﬁcation (GS) ETSI GS ZSM , vol. 2, 2019.
[6]L. Yang, M. El Rajab, A. Shami, and S. Muhaidat, “Enabling automl
for zero-touch network security: Use-case driven analysis,” IEEE Trans-
actions on Network and Service Management , 2024.
[7]P. Bhattacharya, A. Mukherjee, S. Tanwar, and E. Pricop, “Zero-load:
a zero touch network based router management scheme underlying 6g-
iot ecosystems,” in 2023 15th International Conference on Electronics,
Computers and Artiﬁcial Intelligence (ECAI) . IEEE, 2023, pp. 1–7.
[8]E. El-Rif, A. Leivadeas, and M. Falkner, “Intent expression through
natural language processing in an enterprise network,” in 2023 IEEE
24th International Conference on High Performance Switching and
Routing (HPSR) . IEEE, 2023, pp. 1–6.
[9]M. U. Hadi, R. Qureshi, A. Shah, M. Irfan, A. Zafar, M. B. Shaikh,
N. Akhtar, J. Wu, S. Mirjalili et al. , “A survey on large language models:
Applications, challenges, limitations, and practical usage,” Authorea
Preprints , 2023.
[10] U. M. Natarajan, R. B. Diddigi, and J. Bapat, “Rag-inspired intent-
based solution for intelligent autonomous networks,” in 2025 17th
International Conference on COMmunication Systems and NETworks
(COMSNETS) . IEEE, 2025, pp. 413–421.
[11] K. Tamizhelakkiya, D. Das, J. Bapat, D. Das, and K. Sharma, “Novel
closed loop control mechanism for zero touch networks using bilstm and
q-learning,” in 2025 National Conference on Communications (NCC) .
IEEE, 2025, pp. 1–6.
[12] J. Lin, K. Dzeparoska, A. Tizghadam, and A. Leon-Garcia, “Apple-
seed: Intent-based multi-domain infrastructure management via few-
shot learning,” in 2023 IEEE 9th International Conference on Network
Softwarization (NetSoft) . IEEE, 2023, pp. 539–544.
[13] W. Zhuang and Y. Cao, “Short-term trafﬁc ﬂow prediction based on cnn-
bilstm with multicomponent information,” Applied Sciences , vol. 12,
no. 17, p. 8714, 2022.
[14] International Telecommunication Union, “ITU-T Recommendation
P.800: Methods for subjective determination of transmission quality,”
ITU, Tech. Rep., 1996, accessed: 2025-07-04. [Online]. Available:
https://www.itu.int/rec/T-REC-P.800
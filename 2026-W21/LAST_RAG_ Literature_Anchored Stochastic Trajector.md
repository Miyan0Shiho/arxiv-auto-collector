# LAST-RAG: Literature-Anchored Stochastic Trajectory Retrieval-Augmented Generation for Knowledge-Conditioned Degradation Model Selection

**Authors**: Hanbyeol Park, Hyerim Bae

**Published**: 2026-05-18 06:07:16

**PDF URL**: [https://arxiv.org/pdf/2605.17902v1](https://arxiv.org/pdf/2605.17902v1)

## Abstract
Stochastic-process-based degradation modeling is a core approach for estimating the distribution of remaining useful life (RUL); however, the selection of an appropriate stochastic process has not been sufficiently addressed. Existing model selection methods mainly rely on the statistical fit of the observed health indicator (HI) trajectory, but this approach may select a model that is inconsistent with the underlying degradation mechanism when the observation window is short or the signal is highly noisy. To address this issue, this paper proposes Literature-Anchored Stochastic Trajectory Retrieval-Augmented Generation (LAST-RAG). The proposed method uses both the observed HI trajectory and domain-specific context, and hierarchically conditions the candidate degradation model space based on theoretical and mechanical evidence retrieved from a local evidence bank. In addition, Rule-based Confidence Reasoning with Uncertain State (RCRUS) is introduced to prevent candidate models from being prematurely eliminated when hierarchical decisions are uncertain. Simulation-based experiments demonstrate that the proposed method outperforms statistical, prognostic, and uncertainty-aware baselines in both Wiener/gamma family classification and detailed degradation model classification. Ultimately, this study reframes degradation model selection from a purely statistical goodness-of-fit problem into a knowledge-conditioned decision-making problem that integrates observed data with domain knowledge.

## Full Text


<!-- PDF content starts -->

LAST-RAG: LITERATURE-ANCHOREDSTOCHASTIC
TRAJECTORYRETRIEVAL-AUGMENTEDGEMERATION FOR
KNOWLEDGE-CONDITIONEDDEGRADATIONMODELSELECTION
Hanbyeol Park
Department of Industrial Engineering
Pusan National University
Busan, Republic of Korea
pb104@pusan.ac.krHyerim Bae
Department of Industrial Engineering
Pusan National University
Busan, Republic of Korea
hrbae@pusan.ac.kr
May 19, 2026
ABSTRACT
Stochastic-process-based degradation modeling is a core approach for estimating the distribution
of remaining useful life (RUL); however, the selection of an appropriate stochastic process has not
been sufficiently addressed. Existing model selection methods mainly rely on the statistical fit of the
observed health indicator (HI) trajectory, but this approach may select a model that is inconsistent
with the underlying degradation mechanism when the observation window is short or the signal is
highly noisy. To address this issue, this paper proposes Literature-Anchored Stochastic Trajectory
Retrieval-Augmented Generation (LAST-RAG). The proposed method uses both the observed HI
trajectory and domain-specific context, and hierarchically conditions the candidate degradation model
space based on theoretical and mechanical evidence retrieved from a local evidence bank. In addition,
Rule-based Confidence Reasoning with Uncertain State (RCRUS) is introduced to prevent candidate
models from being prematurely eliminated when hierarchical decisions are uncertain. Simulation-
based experiments demonstrate that the proposed method outperforms statistical, prognostic, and
uncertainty-aware baselines in both Wiener/gamma family classification and detailed degradation
model classification. Ultimately, this study reframes degradation model selection from a purely
statistical goodness-of-fit problem into a knowledge-conditioned decision-making problem that
integrates observed data with domain knowledge.
1 Introduction
As the importance of operational stability, reliability, and cost reduction continues to increase, remaining useful life
(RUL) prediction has become increasingly crucial across various domains including the aerospace [ 1], transportation and
logistics [ 2], and energy sectors [ 3]. RUL prediction directly links condition monitoring with operational maintenance
decision-making, and is widely regarded as a core component of condition-based maintenance [ 4]. A central task
in RUL prediction is to model the future deterioration of health indicators (HIs) collected from, or extracted from,
equipment [ 5]. Because inappropriate degradation models can underestimate or overestimate RUL uncertainty, choosing
an appropriate degradation model is a critical issue [6].
Despite this need, studies on stochastic process-based degradation modeling have primarily focused on structural
improvements of degradation models [ 7] and parameter estimation or approximation methods [ 8], whereas degradation
model selection has received less attention. Existing degradation model selection methods can be broadly divided into
likelihood-based and prognostics-oriented criteria based on RUL prediction performance [ 6]. Likelihood-based criteria
can enable robust performance when sufficient data are available; however, when the observation window is short or
the HI contains substantial noise, they may select a model that does not reflect the actual degradation mechanism [ 9].
Prognostics-oriented criteria can identify degradation trajectories suitable for RUL prediction, but some model selection
procedures may implicitly use information that is available only after the prediction time or even the full life-cyclearXiv:2605.17902v1  [cs.AI]  18 May 2026

APREPRINT- MAY19, 2026
trajectory, thereby causing data leakage. In other words, existing degradation model selection methods rely mainly on
the observed HI trajectory; consequently, they are sensitive to data quality and limited in their ability to incorporate
knowledge about system-level or equipment-specific failure mechanisms.
In industrial practice, expert knowledge, experience, and technical expertise are recognized as being essential for
equipment maintenance [ 10]. This knowledge includes information on the system type, structural characteristics,
failure modes, sensor types, and operating conditions, as well as individual tacit knowledge. For example, irreversible
damage accumulation processes such as corrosion, cumulative wear, and crack propagation are naturally associated
with gamma-family models, which assume non-negative increments and monotonic degradation behavior [ 11]. In
contrast, vibration-based HIs in rotating machinery may exhibit repeated local increases and decreases due to rotational
frequency variations, load fluctuations, and changes in operating conditions [ 12]. In such cases, Wiener-family models
may be more appropriate candidates. Furthermore, even within each degradation family, extended models can be further
subdivided according to linearity, the presence of an initial degradation point, change points, and random effects [ 13].
Under these more realistic conditions, degradation model selection should be regarded not merely as a statistical model
selection problem, but as a scientific decision-making problem that integrates statistical evidence with an understanding
of system behavior and failure mechanisms.
To this end, we propose Literature-Anchored Stochastic Trajectory Retrieval-Augmented Generation (LAST-RAG), a
local-evidence-based RAG framework for stochastic degradation model selection. Given an observed HI trajectory and
domain-specific context, LAST-RAG retrieves proposition-level evidence from a pre-constructed local evidence bank
and uses the retrieved evidence to condition the selection of candidate stochastic degradation models. The performance
of the proposed method was validated using simulation-generated datasets. LAST-RAG achieved an average F1 score
of 0.925 in binary stochastic-process family classification. Notably, even when only 30% of the HI trajectory up to the
actual failure time was observed, LAST-RAG achieved an F1 score of 0.983, corresponding to an improvement of 0.394
over the state-of-the-art baseline. The main contributions of this study are summarized as follows:
•We extend degradation model selection beyond a purely statistics-based problem by formulating it as a
knowledge-integrated decision-making problem that incorporates mechanical characteristics and tacit expert
knowledge.
•We propose a degradation model conditioning method based on LAST-RAG, which integrates theoretical
knowledge with observed degradation evidence.
•We develop a hierarchical model selection framework that combines model conditioning with statistical
degradation model selection methods.
The remainder of this paper is organized as follows: Section 2 reviews previous studies on degradation model selection in
the field of prognostics and health management (PHM). Section 3 presents the proposed LAST-RAG-based knowledge-
conditioned degradation model selection framework. Section 4 reports a validation of the effectiveness of the proposed
method using simulation-generated data. Finally, Section 5 concludes the paper and discusses future research directions.
2 Related works
Degradation modeling has been widely adopted because it enables the estimation of the RUL distribution through
the first hitting time (FHT) to a predefined failure threshold [ 6]. Among representative stochastic processes, the
gamma process assumes monotonically increasing degradation trajectories and is therefore well suited to describing
cumulative damage systems. In practice, however, HIs may exhibit repeated local increases and decreases due to sensor
noise, variations in operating conditions, or recovery effects; in such cases, the assumption of monotonicity may be
inappropriate. In contrast, Wiener-process-based models can represent non-monotonic degradation trajectories through
Gaussian increments and a diffusion structure, making them suitable for degradation modeling under local variability.
Nevertheless, many RUL estimation studies assume that the underlying degradation process is known in advance, which
limits their applicability in environments lacking sufficient domain knowledge and expert experience [ 14]. Therefore,
techniques that can select an appropriate degradation model by integrating the observed HI trajectory with limited
domain knowledge are required.
In a representative study addressing degradation model selection, Nguyen et al. [6] compared model selection criteria
for stochastic process models, including Brownian motion, Ornstein–Uhlenbeck processes, and gamma process families,
from the perspectives of goodness-of-fit, model complexity, and prognostic relevance. In addition to conventional
criteria, such as the Akaike information criterion (AIC), Bayesian information criterion (BIC), minimum description
length (MDL), empirical average log-likelihood (EAL), and cross-validation (CV), they also examined RUL-prediction-
oriented criteria, including the prognostic horizon criterion (PHC), prognostic accuracy criterion (PAC), and hybrid
2

APREPRINT- MAY19, 2026
criterion (HyC). Their results showed that CV tended to favor models with many parameters, whereas complexity-
penalized criteria such as AIC, BIC, and MDL tended to favor simpler models with fewer parameters. They also noted
that posterior information related to operating conditions was not sufficiently incorporated. Moreover, PHC was found
to be weak in identifying the degradation model family, whereas PAC exhibited a paradoxical tendency to select an
incorrect model family as the observation period increased.
Yu et al. [ 9] highlighted model selection uncertainty in online RUL prediction and considered both prior parameter
uncertainty and degradation model uncertainty. Specifically, they employed a Bayesian-updated expectation conditional
maximization algorithm to update uncertain prior parameters and proposed a modified Bayesian model averaging
method to represent uncertainty among candidate degradation models using posterior model probabilities. Rather than
forcing the selection of a single model at a given time point, this approach sequentially updates the probabilities of
candidate models and incorporates them into the RUL estimation.
Zhou et al. [ 15] proposed another composite indicator for degradation model selection. Their study used the number of
model parameters as an indicator of model complexity, the mean absolute error (MAE) as an indicator of goodness-of-fit,
and L1 normalization as an indicator of generalization. These three indicators were then normalized and linearly
transformed, and their weights were estimated based on the RUL prediction performance of historical degradation data.
This method can be regarded as an attempt to jointly consider model complexity, prediction accuracy, and generalization
performance.
In summary, early studies on degradation model selection mainly considered likelihood-based model complexity,
whereas more recent studies have attempted to further improve RUL prediction performance using criteria such as PHC,
PAC, HyC, and MAE. However, in real-world settings, degradation model selection does not rely solely on observed
data; it is also closely related to physical degradation mechanisms, operating environments, mechanical characteristics,
and tacit knowledge derived from field experts’ experience. In particular, when the amount of observed data is limited
or the data exhibit high variability, robust model selection using purely data-driven criteria alone is difficult. Existing
studies have largely assumed such knowledge implicitly through researchers’ intuition or prior model designs, without
sufficiently addressing practical constraints. This study aims to improve the explainability of degradation model
selection while additionally accounting for theoretical degradation models, mechanical characteristics, and recursively
accumulated empirical knowledge.
3 Methodology
This study addresses the problem of selecting a stochastic degradation model using an observed HI trajectory and
domain-specific context. The proposed method is a hybrid model selection framework that hierarchically conditions
the structural properties of degradation models using LAST-RAG and applies statistical model selection only to
the uncertain branch identified by LAST-RAG. To improve the logical validity and reliability of the large language
model (LLM) while explicitly extracting uncertain cases, this study introduces Rule-based Confidence Reasoning with
Uncertain State (RCRUS). RCRUS refers to a deterministic arbitration rule that first generates an internal LLM answer
and a RAG-based LLM answer at each hierarchy, and then determines the final hierarchical decision based on the
agreement between the two answers and the difference in their confidence scores. This mechanism is proposed to
mitigate excessive model-set conditioning caused by errors in hierarchical conditioning.
In PHM environments, condition-monitoring data and operational information are often security-sensitive [ 16]. In
addition, PHM systems frequently have limited computational resources and real-time inference requirements [ 17],
making lightweight designs essential. Accordingly, this study assumes an offline local LLM setting that does not
depend on external APIs and proposes a LAST-RAG-based hierarchical model selection framework combined with
domain-specific local RAG.
3.1 Problem formulation
For thei-th inference scenario, the HI trajectory observed up to inspection timeτis defined as follows:
Di
τ=
(tij, yij) :j= 1, . . . , ni
τ, tij≤τ	
.(1)
Here, tijdenotes the j-th inspection time in the i-th scenario, and yijdenotes the HI value observed at that time. The
termni
τis the number of HI samples observed up to timeτ.
In this study, the operator uses domain context provided by an operator or analyst together with the HI sequence. This
is defined as
Xi
τ= 
Di
τ,Ki
τ
,(2)
3

APREPRINT- MAY19, 2026
Figure 1: Proposed hierarchical framework combining local LAST-RAG and stochastic degradation model selection.
where Ki
τdenotes mechanical characteristics or experiential knowledge provided in textual form by an operator or
analyst. Mechanical characteristics may include, for example, equipment type, the physical meaning of the HI, sensor
characteristics, operating environment, wear, cracking, corrosion, fatigue, and battery capacity fading. Experiential
knowledge may include experience regarding the variability of the observed HI, past records, and qualitative judgments
based on visual factors. However, Xi
τexcludes information that is only available after time τand information directly
linked to the degradation model selection outcome, such as failure mode, future observations, failure time, and RUL
labels. This exclusion reflects realistic constraints in online inference.
The objective of this study is to select the final degradation model bmi
τ∈ M from the candidate set of stochastic
degradation models usingXi
τ:
M=
m1, m2, . . . , m |M|	
,(3)
where the elements ofMmay include, for example, a linear Wiener process or a non-homogeneous gamma process.
3.2 Hierarchical representation of the candidate model space
Each candidate model m∈ M is defined as a combination of multiple structural attributes. In this study, these attributes
are defined as hierarchies, and the complete set of hierarchies is represented as
H={F, T},(4)
where Fdenotes the stochastic process family and Tdenotes the trend structure. Let Yhbe the label space for each
hierarchyh∈ H. Then,
YF={W, G},(5)
YT={L, NL},(6)
where YFis the family condition set, and WandGdenote the Wiener and gamma families, respectively. The set YT
represents the form of the trend function, whereLandNLdenote linearity and nonlinearity, respectively.
The structural label assigned to each candidate modelmat hierarchyhis defined by the following mapping:
ψh:M → Y h, ψ h(m)∈ Y h.(7)
Accordingly, the complete structural attributes of candidate modelmcan be expressed as
ψ(m) = (ψ h(m))h∈H.(8)
For example, if a modelmbelongs to the Wiener family and exhibits a nonlinear trend, it can be represented as
ψ(m) = (W, NL).(9)
4

APREPRINT- MAY19, 2026
3.3 Local evidence bank construction and evidence retrieval
This study assumes an offline RAG setting, in which all evidence is restricted to a pre-constructed local evidence bank
(LEB).
3.3.1 Offline proposition evidence bank construction
The offline literature corpus is defined as
P={p l|l= 1, . . . , L},(10)
where pldenotes the l-th document used to construct the LEB. The corpus Pconsists of three categories of literature.
First, literature on mechanical engineering mechanisms describes physical degradation mechanisms and the meanings of
HIs, including wear, corrosion, crack propagation, bearing degradation, and battery capacity fading. Second, stochastic
process theory literature provides the mathematical properties of the Wiener and gamma processes. Specifically,
Wiener-process-based models are suitable for describing continuous sample paths, Gaussian increments, measurement
noise, and local fluctuations. In contrast, gamma-process-based models are suitable for describing non-negative
increments, monotonic accumulation, and irreversible degradation. Third, degradation modeling review literature
provides theoretical or empirical evidence linking specific degradation patterns to stochastic processes.
Eachp lis decomposed into atomic propositions using an offline proposition tokenizerΦ:
Ql= Φ(p l) ={q l1, . . . , q lNl}.(11)
The role of Φis to transform the document into self-contained, minimal, and semantically interpretable proposition-level
retrieval units [ 18]. In this study, the publicly available propositionizer-wiki-flan-t5-large1was used as Φ;
further details are provided in [18].2Here,N ldenotes the number of propositions extracted fromp l.
The offline LEB is then constructed as follows:
Eoff={e r= (q r, sr)|r= 1, . . . , R},(12)
R=LX
l=1Nl,(13)
where qrdenotes the r-th proposition, and srdenotes its provenance metadata, which serve as auxiliary information for
tracing the literature source from which the retrieved evidence originates.
3.3.2 Query construction
The retrieval query is constructed fromXi
τas follows:
gi
τ= Ω 
Xi
τ
,(14)
where Ω(·) denotes a prompt that transforms the HI trajectory and domain-specific context into a textual representation
suitable for retrieval. The query gi
τsummarizes the main characteristics of the observed HI trajectory and domain
context. For example, a query may include descriptions of monotonicity, local fluctuations, abrupt changes, nonlinear
trends, and the possibility of measurement noise. An example ofΩused in this study is provided in Appendix B.
3.3.3 Proposition-level evidence retrieval
Based on the generatedgi
τ, the local LLM retrieves the top-Kproposition-level evidence fromEoff:
Ri
τ= Retrieve K 
gi
τ,Eoff
,(15)
Ri
τ=
er= (q r, sr)|r∈ Ii
τ	
,Ii
τ⊆ {1, . . . , R}.(16)
Here,Ii
τdenotes the evidence index set retrieved at inspection time τin the i-th inference scenario. The retrieved
evidence is subsequently used as evidence-scenario input when generating hierarchy-specific LLM decisions.
1https://huggingface.co/chentong00/propositionizer-wiki-flan-t5-large
2https://github.com/chentong0/factoid-wiki
5

APREPRINT- MAY19, 2026
3.4 RCRUS: Rule-based Confidence Reasoning with Uncertain State
In hierarchical model conditioning, an error can condition the entire model set toward a specific subset, which may
lead to excessive elimination under weak evidence. RCRUS introduces an uncertain state to mitigate this problem.
Specifically, the local LLM generates an answer and a confidence score within the label space of each hierarchy. Then,
based on a deterministic arbitration rule, RCRUS evaluates whether the two answers conflict and whether the difference
between their confidence scores is sufficiently large, thereby assigning an uncertain state when appropriate.
For each hierarchyh, the extended decision space is defined as
eYh=Yh∪ {U h},(17)
whereU hdenotes the uncertain state in hierarchyh.
First, the answer from the internal local LLM is generated as
ai
int,h,τ = 
zi
int,h,τ , ci
int,h,τ
= Λ int,h 
Xi
τ
,(18)
where Λint,h is the internal local LLM decision function for hierarchy h. The term zi
int,h,τ denotes the answer label
produced byΛ int,h, andci
int,h,τ denotes the confidence score of that answer.
Second, using the retrieved evidenceRi
τ, the evidence-conditioned answer is generated as
ai
ctx,h,τ = 
zi
ctx,h,τ , ci
ctx,h,τ
= Λ ctx,h 
Xi
τ,Ri
τ
,(19)
where Λctx,h is the evidence-conditioned local LLM decision function for hierarchy h. The term zi
ctx,h,τ denotes the
answer label produced byΛ ctx,h, andci
ctx,h,τ denotes the corresponding confidence score.
Third, the final hierarchical decision of RCRUS is determined by comparing zi
int,h,τ ,ci
int,h,τ ,zi
ctx,h,τ , andci
ctx,h,τ . Let
the final decision be
zi
h,τ∈eYh.(20)
Then, the RCRUS arbitration rule is defined as follows:
bzi
h,τ=

zi
ctx,h,τ ,ifzi
int,h,τ =zi
ctx,h,τ ,
zi
int,h,τ ,ifzi
int,h,τ̸=zi
ctx,h,τ andci
int,h,τ > ci
ctx,h,τ +δ,
zi
ctx,h,τ ,ifzi
int,h,τ̸=zi
ctx,h,τ andci
ctx,h,τ > ci
int,h,τ +δ,
Uh,ifzi
int,h,τ̸=zi
ctx,h,τ andci
int,h,τ−ci
ctx,h,τ≤δ.(21)
Here, δis the confidence margin for the hierarchy. In this study, δ= 0.05 is used as the default value for all hierarchies,
although it may also be treated as a tunable hyperparameter.
The RCRUS arbitration rule in Eq. (21) follows three principles. First, if the internal and evidence-conditioned answers
support the same label, that label is selected. Second, if the two answers support different labels, their confidence scores
are compared. If one confidence score exceeds the other by at least δ, the answer with the higher confidence score is
selected. Third, if the two answers differ but the confidence-score gap is no larger than δ, the uncertain state Uhis
returned for that hierarchy, and no conditioning is performed.
3.5 RCRUS-based hierarchical space conditioning
Once the final decision bzi
h,τfor each hierarchy is determined by RCRUS, the candidate model space is conditioned
accordingly.
First, the candidate subspace corresponding to labely∈ Y hin hierarchyhis defined as
Mh(y) ={m∈ M |ψ h(m) =y}.(22)
Ifbzi
h,τ=y, only models belonging toM h(y)are retained in hierarchyh. In contrast, forbzi
h,τ=Uh, no conditioning
is performed. Therefore, the retained candidate model set at timeτis defined as
Mi
τ=
m∈ M | ∀h∈ H,bzi
h,τ=Uhorψh(m) =bzi
h,τ	
.(23)
Equivalently, if the set of hierarchies for which confident decisions are generated is defined as
Hi
conf,τ =
h∈ H |bzi
h,τ̸=Uh	
,(24)
6

APREPRINT- MAY19, 2026
then
Mi
τ=\
h∈Hi
conf,τMh 
bzi
h,τ
,(25)
with the convention that the intersection equalsMwhenHi
conf,τ is empty.
For example, if the hierarchy consists of the family and trend, and the RCRUS decisions are
bzi
F,τ=Wandbzi
T,τ=L,(26)
then the retained candidate set is
Mi
τ=M F(W)∩ M T(L) ={m∈ M |ψ F(m) =W, ψ T(m) =L}.(27)
As another example, if the RCRUS decisions include an uncertain state, such as
bzi
F,τ=Wandbzi
T,τ=UT,(28)
then the retained candidate set is
Mi
τ=M F(W) ={m∈ M |ψ F(m) =W}.(29)
In this case, because the trend hierarchy is assigned the uncertain state, bzi
T,τis determined in the subsequent statistical
model selection stage.
3.6 Stochastic model selection within the retained candidate set
After RCRUS-based hierarchical space conditioning, if the retained candidate set Mi
τcontains only a single model,
that model is selected as the final degradation model. Formally,
bmi
τ=m,ifMi
τ={m}.(30)
In contrast, if multiple candidate models remain because at least one uncertain state Uhexists, statistical model selection
is performed based on the observed HI trajectory Di
τ. Let S(m;Di
τ)denote the model selection score for candidate
model m, where a smaller value of S(·) indicates a more suitable model. The final selected model is then determined as
bmi
τ= arg min
m∈MiτS 
m;Di
τ
.(31)
In this study, EAL is used as S(·); however, the proposed framework is not dependent on a specific model selection
criterion.
4 Experiments
4.1 Dataset
Because it is difficult to obtain ground-truth labels for specific degradation curves, this study defined conventional
stochastic process models and used datasets generated through simulations for validation. The experiments were divided
into Cases 1 and 2 according to the task difficulty, as summarized in Table 1. Case 1 addresses binary classification at
the gamma- and Wiener-family level, whereas Case 2 considers a four-class classification task that distinguishes not
only the gamma and Wiener families but also linear/nonlinear trends and homogeneity/non-homogeneity.
Table 1: Stochastic process models included in each experimental case.
Experiment case Family Degradation model Problem
Case 1 Wiener Wiener family 2-class classification
Gamma Gamma family
Case 2 Wiener Linear Wiener 4-class classification
Nonlinear Wiener
Gamma Homogeneous gamma
Non-homogeneous gamma
7

APREPRINT- MAY19, 2026
4.2 Experimental settings
For each stochastic degradation model described in Section 4.1, independent run-to-failure (RtF) degradation trajectories
were generated using the simulation parameters summarized in Appendix A. Each trajectory corresponded to one
independent unit, and no trajectory shared noise realizations or random seeds with any other trajectory. The generated
trajectories were divided into training, validation, and test sets at a ratio of 60%, 20%, and 20%, respectively. Splitting
was performed at the trajectory level.
The training and validation sets were used for parameter estimation and model fitting, whereas the final performance
was exclusively evaluated on the test set. All comparative methods were evaluated using the same training, validation,
and test splits and the same set of candidate models.
To examine model selection performance under different observation horizons, each RtF trajectory was truncated
according to the degradation progression rate n(%), where n∈ {30,50,70} . The stochastic processes considered in
this study are illustrated in Figure 2.
Figure 2: Four models in the Wiener- and gamma-process family.
Table 2: Baselines for degradation model selection.
Category Author Selection criterion
Statistical – AIC, BIC, MDL, CV , EAL
Prognostic Nguyen et al. PHC, PAC, HyC
Zhuo et al. LC
Uncertainty Yu et al. BAM
The performance of the proposed method was compared with that of three groups of model selection baselines, as
summarized in Table 2. The first group consisted of statistical criteria including AIC, BIC, MDL, CV , and EAL. The
second group comprised prognostic criteria including PHC, PAC, HyC, and LC. The third group was the Bayesian
8

APREPRINT- MAY19, 2026
averaging model (BAM), which considers uncertainty in model selection. Classification performance was measured
using accuracy (Acc), precision (Prec), recall (Rec), and F1-score (F1).
4.3 Experimental results
This section reports two experimental results. First, the classification performance of the degradation models is
evaluated for Cases 1 and 2. Second, the robustness of the proposed family classification framework is examined under
information perturbation in the input data.
Table 3: Classification performance comparison of degradation model selection methods under different observation
lengths in Cases 1 and 2. Bold values denote the proposed method; underlined values denote the strongest baseline(s)
by F1-score within each observation-length block.
Case 1: family classification (sample: 24) Case 2: detailed classification (sample: 96)
nModel Acc Prec Rec F1nModel Acc Prec Rec F1
30 AIC 0.478 0.244 0.478 0.323 30 AIC 0.375 0.594 0.375 0.333
BIC 0.478 0.244 0.478 0.323 BIC 0.250 0.088 0.250 0.130
MDL 0.478 0.244 0.478 0.323 MDL 0.250 0.088 0.250 0.130
CV 0.597 0.604 0.597 0.589 CV 0.250 0.155 0.250 0.183
EAL 0.478 0.244 0.478 0.323 EAL 0.500 0.303 0.500 0.361
PHC 0.500 0.250 0.500 0.333 PHC 0.250 0.079 0.250 0.120
PAC 0.490 0.485 0.490 0.443 PAC 0.333 0.250 0.333 0.286
HyC 0.471 0.288 0.471 0.325 HyC 0.625 0.562 0.625 0.548
LC 0.531 0.686 0.531 0.408 LC 0.250 0.083 0.250 0.125
BAM 0.478 0.244 0.478 0.323 BAM 0.500 0.303 0.500 0.361
Proposed 0.968 1.000 0.968 0.983 Proposed 0.750 0.667 0.667 0.667
50 AIC 0.500 0.250 0.500 0.333 50 AIC 0.250 0.088 0.250 0.130
BIC 0.500 0.250 0.500 0.333 BIC 0.250 0.088 0.250 0.130
MDL 0.500 0.250 0.500 0.333 MDL 0.250 0.088 0.250 0.130
CV 0.597 0.608 0.597 0.586 CV 0.167 0.067 0.167 0.095
EAL 0.500 0.250 0.500 0.333 EAL 0.500 0.303 0.500 0.361
PHC 0.500 0.250 0.500 0.333 PHC 0.250 0.079 0.250 0.120
PAC 0.484 0.476 0.484 0.436 PAC 0.292 0.250 0.292 0.269
HyC 0.487 0.246 0.487 0.327 HyC 0.542 0.562 0.542 0.452
LC 0.531 0.758 0.531 0.399 LC 0.250 0.094 0.250 0.136
BAM 0.500 0.250 0.500 0.333 BAM 0.500 0.303 0.500 0.361
Proposed 0.906 1.000 0.906 0.950 Proposed 0.875 0.688 0.750 0.714
70 AIC 0.500 0.250 0.500 0.333 70 AIC 0.250 0.083 0.250 0.125
BIC 0.500 0.250 0.500 0.333 BIC 0.250 0.083 0.250 0.125
MDL 0.500 0.250 0.500 0.333 MDL 0.250 0.083 0.250 0.125
CV 0.609 0.628 0.609 0.595 CV 0.208 0.078 0.208 0.114
EAL 0.500 0.250 0.500 0.333 EAL 0.500 0.333 0.500 0.375
PHC 0.500 0.250 0.500 0.333 PHC 0.250 0.079 0.250 0.120
PAC 0.487 0.485 0.487 0.468 PAC 0.292 0.188 0.292 0.217
HyC 0.500 0.250 0.500 0.333 HyC 0.500 0.333 0.500 0.375
LC 0.525 0.706 0.525 0.391 LC 0.250 0.083 0.250 0.125
BAM 0.500 0.250 0.500 0.333 BAM 0.500 0.333 0.500 0.375
Proposed 0.750 1.000 0.750 0.843 Proposed 0.625 0.667 0.750 0.625
Table 3 summarizes the degradation model classification results for Cases 1 and 2 with different observation lengths.
Overall, the baseline methods showed case-dependent and unstable performance. In Case 1, CV achieved the strongest
performance among the statistical selection criteria, while PAC showed relatively competitive results among the
prognostics-oriented criteria. However, their F1-scores remained limited, indicating insufficient discriminative capability
9

APREPRINT- MAY19, 2026
even in the binary classification setting. In Case 2, the best-performing baseline varied across n, with HyC, EAL, and
BAM showing better results, but no baseline consistently maintained high performance.
By contrast, the proposed method achieved the best overall performance in both cases. In Case 1, its F1-scores were
0.983, 0.950, and 0.843 for n= 30 , 50, and 70, respectively. In Case 2, it achieved F1-scores of 0.667, 0.714, and
0.625, outperforming the strongest baseline in all settings. These results indicate that the proposed method provides
more reliable degradation model classification than the conventional information criteria, likelihood-based criteria,
and prognostics-oriented selection measures. The consistent improvement across cases suggests that the proposed
framework can effectively exploit both trajectory-level degradation patterns and model-relevant evidence for robust
model selection.
Table 4: Robustness analysis of stochastic-process family classification under HI trajectory and domain-specific
perturbations.
nHI trajectory Domain context Acc Prec Rec F1
30Correct Correct 0.8751.0000.875 0.929
Correct Wrong 0.750 1.000 0.750 0.833
Wrong Correct 0.000 0.000 0.000 0.000
Wrong Wrong 0.000 0.000 0.000 0.000
50Correct Correct 1.0001.0001.000 1.000
Correct Wrong 0.750 1.000 0.750 0.833
Wrong Correct 0.000 0.000 0.000 0.000
Wrong Wrong 0.000 0.000 0.000 0.000
70Correct Correct 1.0001.0001.000 1.000
Correct Wrong 0.750 1.000 0.750 0.833
Wrong Correct 0.000 0.000 0.000 0.000
Wrong Wrong 0.000 0.000 0.000 0.000
Table 4 presents an evaluation of the effect of information perturbation in the HI trajectory and domain context on
performance at the family classification stage. When both types of information were correctly provided, the proposed
method achieved an F1-score of 0.929 at n= 30 , and all performance metrics reached 1.000 at n= 50 andn= 70 .
When only the domain context was perturbed, accuracy and F1-score decreased to 0.750 and 0.833, respectively, but the
classification performance remained high. This suggests that the proposed method utilizes domain context without
excessively relying on it. In contrast, when the HI trajectory was perturbed, all performance metrics decreased to 0.000,
regardless of the correctness of the domain context. These results indicate that the family classification decision of
the proposed method depends more strongly on the dynamic characteristics of the observed HI trajectory than on the
domain context. Therefore, ensuring the quality and reliability of HI trajectories is a critical prerequisite for practical
applications.
5 Conclusion
This study redefined stochastic degradation model selection in PHM as a knowledge-conditioned decision-making prob-
lem that jointly considers observed data and domain knowledge. The proposed LAST-RAG framework hierarchically
conditions the stochastic process family and trend structure by leveraging literature-grounded evidence retrieved from a
local evidence bank, while RCRUS prevents excessive elimination of candidate models when decisions are uncertain.
The experimental results demonstrate that the proposed method achieves more stable performance than existing model
selection criteria, even under limited observation windows, suggesting that knowledge-based model conditioning is
effective for online RUL inference. Nevertheless, this study was based on simulated degradation trajectories. Future
work should validate the generalizability of the proposed framework using real-world PHM datasets and extend it to
multivariate HIs, random effects, change-point structures, and covariate-dependent degradation models.
10

APREPRINT- MAY19, 2026
A Implementation Details for Stochastic Process Generation
This appendix describes the complete data-generation process, including the distributional assumptions for increments
and the definition of change points.
This study considers a four-class classification problem consisting of the linear Wiener process, nonlinear Wiener
process, homogeneous gamma process, and non-homogeneous gamma process. For each unit i, the degradation state is
denoted by Xij, where the observation time step is given by j= 0,1, . . . , N i. The initial state is fixed as Xi0= 0, and
the increment is defined as∆X ij=X ij−X i,j−1 .
A.1 Wiener process family
The Wiener process is a stochastic degradation process with Gaussian increments. The normal distribution is denoted
byN(µ, σ2), whereµandσ2denote the mean and variance, respectively.
The linear Wiener process represents the case in which both the degradation rate and noise scale remain constant over
time, while negative degradation increments are allowed. Its increment is defined as
∆Xij∼N(m, s2),(32)
wheremis the constant drift parameter andsis the diffusion parameter.
The nonlinear Wiener process represents the case in which the degradation rate varies over time while negative
increments are still allowed. In this study, nonlinearity is introduced using the time-transformed increment function
dj(β) =jβ−(j−1)β.(33)
Accordingly, the increment of the nonlinear Wiener process is defined as
∆Xij∼N 
mdj(β), s2
.(34)
A.2 Gamma process family
The gamma process is a degradation process whose increments follow a gamma distribution; it is characterized by
cumulative and monotonic degradation behavior. The gamma distribution is denoted by Γ(k, ϑ) , where kis the shape
parameter andϑis the scale parameter.
The homogeneous gamma process represents the case in which the non-negative degradation increments follow a
time-invariant structure. Its increment is defined as
∆Xij∼Γ(α, ϑ).(35)
The non-homogeneous gamma process represents the case in which the rate of non-negative degradation varies over
time. As in the nonlinear Wiener process, the time-transformed increment function dj(β) =jβ−(j−1)βis used to
represent nonlinearity. Accordingly, the increment of the non-homogeneous gamma process is defined as
∆Xij∼Γ (αd j(β), ϑ).(36)
11

APREPRINT- MAY19, 2026
B Prompt examples
Figure 3: Internal and external LLM prompts used in RCRUS. Bold text indicates the main topic, and red text denotes
variables.
12

APREPRINT- MAY19, 2026
C Domain-specific machinery descriptions
Figure 4: Results retrieved by the local LLM for four types of mechanical information.
D Example evidence chunks extracted from the LEB
Figure 5: Examples of LEB evidence chunks consisting of mechanical theory, stochastic process theory, and review
papers. For each corpus, one representative example corresponding to the Wiener process and one corresponding to the
gamma process is presented.
13

APREPRINT- MAY19, 2026
References
[1]Xiaopeng Xi, Donghua Zhou, Maoyin Chen, Narayanaswamy Balakrishnan, and Hanwen Zhang. Remaining
useful life prediction for multivariable stochastic degradation systems with non-markovian diffusion processes.
Quality and Reliability Engineering International, 36(4):1402–1421, 2020.
[2]Yu Zang, Wei Shangguan, Baigen Cai, Huasheng Wang, and Michael G. Pecht. Hybrid remaining useful life
prediction method. a case study on railway d-cables.Reliability Engineering & System Safety, 213:107746, 2021.
[3]Yaogang Hu, Hui Li, Pingping Shi, Zhaosen Chai, Kun Wang, Xiangjie Xie, and Zhe Chen. A prediction method
for the real-time remaining useful life of wind turbine bearings based on the wiener process.Renewable Energy,
127:452–460, 2018.
[4]Zhengxin Zhang, Changhua Hu, Xiaosheng Si, Jianxun Zhang, and Jianfei Zheng. Stochastic degradation process
modeling and remaining useful life estimation with flexible random-effects.Journal of the Franklin Institute,
354(6):2477–2499, 2017.
[5]Khanh Le Son, Mitra Fouladirad, Anne Barros, Eric Levrat, and Benoît Iung. Remaining useful life estimation
based on stochastic deterioration models: A comparative study.Reliability Engineering & System Safety, 112:165–
175, 2013.
[6]Khanh T. P. Nguyen, Mitra Fouladirad, and Antoine Grall. Model selection for degradation modeling and prognosis
with health monitoring data.Reliability Engineering & System Safety, 169:105–116, 2018.
[7]Shuyi Zhang, Qingqing Zhai, Xin Shi, and Xuejuan Liu. A wiener process model with dynamic covariate for
degradation modeling and remaining useful life prediction.IEEE Transactions on Reliability, 72(1):214–223,
2023.
[8]Xiao-Sheng Si, Wenbin Wang, Chang-Hua Hu, Dong-Hua Zhou, and Michael G. Pecht. Remaining useful life
estimation based on a nonlinear diffusion degradation process.IEEE Transactions on Reliability, 61(1):50–67,
2012.
[9]Yong Yu, Xiaosheng Si, Changhua Hu, Jianfei Zheng, and Jianxun Zhang. Online remaining-useful-life estima-
tion with a bayesian-updated expectation-conditional-maximization algorithm and a modified bayesian-model-
averaging method.Science China Information Sciences, 64(1), 2020.
[10] Ling Li, Min Liu, Weiming Shen, and Guoqing Cheng. An expert knowledge-based dynamic maintenance task
assignment model using discrete stress–strength interference theory.Knowledge-Based Systems, 131:135–148,
2017.
[11] Khanh Le Son, Mitra Fouladirad, and Anne Barros. Remaining useful lifetime estimation and noisy gamma
deterioration process.Reliability Engineering & System Safety, 149:76–87, 2016.
[12] Zhengxin Zhang, Xiaosheng Si, Changhua Hu, and Yaguo Lei. Degradation data analysis and remaining useful life
estimation: A review on wiener-process-based methods.European Journal of Operational Research, 271(3):775–
796, 2018.
[13] Ameneh Forouzandeh Shahraki. A review on degradation modelling and its engineering applications.International
Journal of Performability Engineering, 2017.
[14] Yong Yu, Changhua Hu, Xiaosheng Si, and Jianxun Zhang. Degradation data-driven remaining useful life
estimation in the absence of prior degradation knowledge.Journal of Control Science and Engineering, 2017:1–
11, 2017.
[15] Yue Zhuo, Jianxun Zhang, Lei Feng, Zhengxin Zhang, and Xiaosheng Si. A novel criterion of degradation model
selection for remaining life prediction, 2025/05/09 2025.
[16] Behrad Bagheri, Maryam Rezapoor, and Jay Lee. A unified data security framework for federated prognostics and
health management in smart manufacturing.Manufacturing Letters, 24:136–139, 2020.
[17] Hanbyeol Park, Sunghyun Sim, Yunkyung Park, and Hyerim Bae. Trend-fluctuation correlated attention unit for
remaining useful life prediction.IEEE Access, 13:164486–164507, 2025.
[18] Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang, and Dong Yu.
Dense x retrieval: What retrieval granularity should we use?, 2024.
14
# How Many Iterations to Jailbreak? Dynamic Budget Allocation for Multi-Turn LLM Evaluation

**Authors**: Shai Feldman, Yaniv Romano

**Published**: 2026-05-07 17:25:15

**PDF URL**: [https://arxiv.org/pdf/2605.06605v1](https://arxiv.org/pdf/2605.06605v1)

## Abstract
Evaluating and predicting the performance of large language models (LLMs) in multi-turn conversational settings is critical yet computationally expensive; key events -- e.g., jailbreaks or successful task completion by an agent -- often emerge only after repeated interactions. These events might be rare, and under any feasible computational budget, remain unobserved.
  Recent conformal survival frameworks construct reliable lower predictive bounds (LPBs) on the number of iterations to trigger the event of interest, but rely on static budget allocation that is inefficient in multi-turn setups. To address this, we introduce \emph{Dynamic Allocation via PRojected Optimization} (DAPRO), the first theoretically valid dynamic budget allocation framework for bounding the time-to-event in multi-turn LLM interactions.
  We prove that DAPRO satisfies the budget constraint and provides distribution-free, finite-sample coverage guarantees without requiring the conditional independence between censoring and event times assumed by prior conformal survival approaches.
  A key theoretical contribution is a novel coverage bound that scales with the square root of the mean censoring weight rather than the worst-case weight, yielding provably tighter guarantees than prior work. Furthermore, DAPRO can be employed to obtain unbiased, low-variance estimates of population-level evaluation metrics, such as the jailbreak rate, under limited computing resources.
  Comprehensive experiments across agentic task success, adversarial jailbreaks, toxic content generation, and RAG hallucinations using LLMs such as Llama 3.1 and Qwen 2.5 demonstrate that DAPRO consistently achieves coverage closer to the nominal level with lower variance than static baselines, while satisfying the budget constraint.

## Full Text


<!-- PDF content starts -->

How Many Iterations to Jailbreak? Dynamic Budget
Allocation for Multi-Turn LLM Evaluation
Shai Feldman
Department of Computer Science
Technion, Israel
shai.feldman@cs.technion.ac.il
Yaniv Romano
Departments of Electrical and Computer Engineering and of Computer Science
Technion, Israel
yromano@cs.technion.ac.il
Abstract
Evaluating and predicting the performance of large language models (LLMs) in
multi-turn conversational settings is critical yet computationally expensive; key
events—e.g., jailbreaks or successful task completion by an agent—often emerge
only after repeated interactions. These events might be rare, and under any feasible
computational budget, remain unobserved. Recent conformal survival frameworks
construct reliable lower predictive bounds (LPBs) on the number of iterations to
trigger the event of interest, but rely on static budget allocation that is inefficient in
multi-turn setups. To address this, we introduceDynamic Allocation via PRojected
Optimization( DAPRO ), the first theoretically valid dynamic budget allocation frame-
work for bounding the time-to-event in multi-turn LLM interactions. We prove that
DAPRO satisfies the budget constraint and provides distribution-free, finite-sample
coverage guarantees without requiring the conditional independence between cen-
soring and event times assumed by prior conformal survival approaches. A key
theoretical contribution is a novel coverage bound that scales with the square root
of the mean censoring weight rather than the worst-case weight, yielding provably
tighter guarantees than prior work. Furthermore, DAPRO can be employed to obtain
unbiased, low-variance estimates of population-level evaluation metrics, such as
the jailbreak rate, under limited computing resources. Comprehensive experiments
across agentic task success, adversarial jailbreaks, toxic content generation, and
RAG hallucinations using LLMs such as Llama 3.1 and Qwen 2.5 demonstrate
thatDAPRO consistently achieves coverage closer to the nominal level with lower
variance than static baselines, while satisfying the budget constraint.
1 Introduction
Large language models (LLMs) are increasingly deployed in high-stakes applications ranging from
healthcare and education to customer service [ 1,2]. As their use expands, evaluating their safety and
utility becomes essential. We define an unsafe response as content that contains toxic language [ 3],
reveals confidential data [ 4], or produces factual hallucinations [ 5,6]. For example, in educational AI
tutors or customer service chatbots [ 7,8], users rarely elicit harmful behavior with a single query;
instead, they iteratively tweak prompts to bypass guardrails [ 9,10]. To evaluate safety in these multi-
turn adversarial settings, researchers must simulate these interactions under a global computational
budget constraint B∈N , e.g., 1,000 exchanges across all conversations. These conversations are
monitored by an independent auditor, such as an LLM-as-a-judge [ 11]. Since these events can be rare,arXiv:2605.06605v1  [cs.LG]  7 May 2026

Figure 1: Illustration of our framework: (i) collecting data via dynamic budget allocation; (ii)
calibrating a pre-trained model; and (iii) deploying the model at inference time to serve as a guardrail.1
particularly for well-aligned models [ 12], recently introducedtime-to-unsafe-sampling. In our setting,
this metric represents the number of conversational interactions required to elicit an unsafe response
given an initial prompt. Utilizing this metric, one can frame LLM prompt risk assessment as a survival
analysis task, which is advantageous for two reasons. First, the metric is well-defined even when LLM
outputs are time-dependent or adversarial, such as in jailbreaking [ 9]. Second, it allows constructing
a calibratedlower predictive bound(LPB) ˆL(X test)for the unknown time-to-unsafe-sampling Ttestof
a new test prompt Xtestwith formal, finite-sample coverage guarantees P(T test≥ˆL(X test))≥1−α ,
where 1−α is the user-specified target coverage rate, e.g., 1−α= 90% . Informally, one can reliably
expect the LLM to survive at least ˆL(X test)−1 exchanges before yielding unsafe content. With
this reliability guarantee, this LPB can be used for safety evaluation and as a proactive guardrail,
as a lower value indicates a higher safety risk. While we frame our discussion around safety, this
framework applies to evaluating utility, such as constructing an upper predictive bound (UPB) for the
time-to-successof an agent assisting a target model (see Figure 7 in Appendix B). For simplicity of
exposition, we focus primarily on the safety scenario and LPBs hereafter.
To construct this LPB, we must first acquire a dataset of annotated conversations under the budget B,
generated by running an attacking algorithm starting from a set of initial prompts, and continuously
evaluating the outputs of the target model using the auditor. To satisfy the computational budget
constraint, we set a censoring time Cifor each conversation i, which limits the maximum allowed
conversation length. [ 12] proposes an optimized budget allocation strategy that minimizes the
variance of the resulting LPB. However, this strategy sets static censoring times that are fixedbefore
any interaction occursand are not updated over time, even as the conversation unfolds. Consequently,
this static method is not optimal for our multi-turn setup. Furthermore, it often under-utilizes the
budget; when a conversation elicits unsafe content before its allocated budget is fully consumed, the
remaining budget that was dedicated to that prompt is lost: it cannot be redistributed to other prompts
that may benefit from additional sampling.
However, dynamically allocating the censoring times is mathematically challenging. First, to provide
a theoretically valid coverage guarantee for the constructed LPBs, different conversations cannot
actively share information or coordinate their decisions to stop or continue, since such communication
breaks the independence assumptions required for this theoretical guarantee. Second, the online
adaptation of Cimust satisfy the shared global budget constraint B, even though the trajectory and
true length of each conversation are initially unknown. Third, the allocation must not merely be valid,
butoptimal, fully utilizing the budget to minimize the variance of the resulting LPBs.
In this work, we tackle these challenges and propose the first adaptive budget allocation framework
that dynamically updates Ciwhile providing a theoretical coverage guarantee and satisfying the
global budget constraint. To illustrate the necessity and efficiency of this dynamic adaptation, we
conduct an experiment on the RealToxicityPrompts [ 3] dataset, where both the attacker and the target
utilize the Qwen 2.5 14B Instruct [ 13] model. In Appendix E.4 we provide more details on this
setup. Figure 2 shows that the static allocation strategy of [12] sets a static censoring time that does
not change as the attacker-target interactions evolve. In contrast, our proposed strategy adapts the
censoring times online as the conversation unfolds, achieving a coverage rate closer to the nominal
coverage level with a lower variance. We emphasize that obtaining a lower variance is crucial in
1This figure was generated using Google Gemini based on a prompt designed by the authors and subsequently refined.
2

Figure 2: Adaptive vs. static budget allocation. By dynamically adjusting the budget over time
(Left+Mid), our adaptive approach achieves lower empirical coverage error (Right).
practice, where only a single dataset is typically available. A lower variance means that our LPBs
attain a coverage rate closer to the target level1−αgiven the single dataset.
1.1 Our contribution
In this work, we introduceDynamic Allocation via PRojected Optimization( DAPRO ), a novel paradigm
that treats budget allocation as a sequential decision. In contrast to prior work that sets static censoring
times, DAPRO dynamically updates them in response to ongoing conversational interactions. By
splitting the available data to first learn an optimal allocation policy and then adaptively deploying it,
DAPRO invests computational resources in the most informative exchanges. This reduces the variance
of the constructed predictive bounds while satisfying the global budget constraint. Specifically, our
main contributions are:
•Tighter Theoretical Guarantees: We prove that DAPRO attains the desired coverage level
with a bound strictly tighter than prior methods [ 12,14]. Crucially, our novel proof technique
removes the restrictive assumption that the censoring time Cand true event time Tmust be
conditionally independent given the prompt X. This result is of independent interest as it
applies to a broad class of survival analysis tasks.
•Budget Validity: Under certain assumptions, we formally prove that our dynamic allocation
strategy satisfies the global sampling budget constraint in expectation.
•Population-Level Performance Evaluation Under Censoring: Beyond individual bounds,
DAPRO can efficiently evaluate LLM at the population level, e.g., providing unbiased esti-
mates of the unsafe event rate, reducing the costs of LLM calls (Appendix F.6).
•Comprehensive Empirical Evaluation: We validate DAPRO across diverse tasks, including
agentic task success, toxic content, adversarial jailbreaks, and RAG-based hallucinations,
demonstrating our proposal consistently achieves empirical coverage closer to the nominal
level and utilizes the budget more efficiently compared to existing baselines.
Software implementing the proposed method and reproducing our experiments is available at https:
//github.com/Shai128/dapro
1.2 Problem setup
We consider a set of base objectives {Vi}N
i=1, e.g., harmful requests, from which an attacker model
Agenerates an initial adversarial prompt for each objective. We define our calibration dataset as
Dcal={X i}N
i=1, where Xi=A(V i)is the initial attack prompt for the i-th objective. We assume
that the training data Dtrain, which is used to fit the predictive model, has a separate budget constraint.
The combined dataset is denoted as Dall=D train∪ Dcal. For each prompt in Dcal, an attacker A
interacts sequentially with a target modelGto elicit an unsafe response.
At step t= 1 , the history is simply the initial prompt: Hi(1) ={A i(1)}, where Ai(1) :=X i=
A(V i). At any step t≥1 , the target model processes this prompt and outputs a response: Ri(t) =
G(A i(t)). Next, we employ an audit function Jto obtain a binary label Yi(t) =J(A i(t), R i(t))∈
{0,1} , where Yi(t) = 1 ifRi(t)is an unsafe response, and Yi(t) = 0 otherwise. If the attack
is successful, we terminate this process. Otherwise, we continue to step t+ 1 , where the attacker
3

generates Ai(t+1) =A(H i(t)∪{(R i(t), Y i(t))}) , and we update the interaction history: Hi(t+1) =
Hi(t)∪ {(R i(t), Y i(t), A i(t+ 1))} . For notational convenience, we set Hi={H i(t)}tmax
t=1, where
tmaxis the maximal sequence length.
We define the uncensored time-to-event as the minimum number of steps required to achieve an unsafe
response: Ti= min{t≥1 :Y i(t) = 1} . The censored time-to-event is thus ˜Ti= min(T i, Ci).
Since the budget expended on thei-th sample is ˜Ti, the budget constraint can be formulated as:
E"NX
i=1˜Ti#
≤B.(1)
Our goal is to construct an LPB ˆL(X test)given an initial test prompt Xtestfor the test event time
Ttest. Importantly, at test time, we construct the LPBwithoutrunning any attacker-target exchanges,
relying solely on the initial prompt Xtest. This design choice reflects two practical considerations.
First, a test-time sampling budget may be extremely limited or entirely unavailable. Second, and
more critically, when the target model is an autonomous agent operating in a real deployment, we
must assess the safety of a promptbeforethe agent has any opportunity to act, as any unsafe action
taken by the agent might be irreversible. Finally, given a target coverage level 1−α∈(0,1) and a
tolerance levelδ∈(0,1), the LPB should satisfy:
P
ˆL(X test)≤T test| Dall
≥1−α,(2)
with probability at least 1−δ over the randomness of the data Dall. We remark that the probability
in(2)is taken over random draws of Xtest, Ttest. An LPB ˆLsatisfying this requirement is called a
Probably Approximately Correct (PAC) LPB at levelαwith toleranceδ.
In Section 3, we introduce our approach for designing the censoring mechanism and allocating the
sampling budget to satisfy the budget constraint (1). After obtaining the censoring and survival times,
we can construct a calibrated LPB with the approach introduced in [ 12], which is presented hereafter.
2 Background
Our approach builds on the LLM safety evaluation framework of [ 12], which refines conformalized
survival analysis [ 14] to construct a valid LPB for the time-to-unsafe under a known censoring
mechanism. While we adopt their LPB construction, their budget allocation strategy is static, which
is inefficient in our multi-turn setup. We briefly outline their LPB construction procedure below.
Suppose we are given quantile estimates ˆqτ(x)for the time-to-event from a pre-trained model. Since
the raw estimates ˆqα(x)might not be sufficiently accurate, using them directly as LPBs might not
achieve 1−α coverage. Thus, we find an adjusted quantile level ˆτsuch that the resulting LPB ˆqˆτ(x)
satisfies the coverage requirement. We obtain this calibrated quantile level by weighting the errors of
the raw quantile estimates to account for the distribution shift induced by the censoring:
ˆα(τ) =1
|Ical|X
i∈Icalwτ(i)I˜Ti<ˆqτ(Xi)≤C i	
,(3)
where the weights are formulated as: wτ(i) =P
ˆqτ(Xi)≤C i{Xj}j∈Ical−1. Notice that these
weights are known since the mechanism is actively designed by the algorithm. The calibrated LPB is
thenˆL(x) = ˆq ˆτ(x), whereˆτis defined as the largest level achieving valid coverage:
ˆτ= sup
τ∈ T: sup
τ′<τˆα(τ′)≤α	
,
where Tis the search space for τ. As proven in [ 12], this LPB holds a PAC-type coverage guarantee,
where the coverage gap scales with the maximal weight wτ(i), such that larger weights lead to
a higher variance. Motivated by this insight, [ 12] introduces two refinements to reduce variance
that we adopt. First, the raw estimates are capped at M≤t max, yielding trimmed estimates
ˆfτ(x) := min(ˆq τ(x), M) . This trimming bounds the maximal weight, and thus controls the tightness
of the PAC-type coverage guarantee. Furthermore, by setting ˆL(x)≤M≤t max, we can determine
4

whether Ttestfalls below or above the LPB, so we can correctly evaluate the empirical coverage rate
of our LPBs. Thus, throughout this work, we assume M≤t max. Second, since a conversation i
only contributes to ˆα(τ) when ˆfτ(Xi)≤C i, the authors propose fixing a prior quantile level τprior,
which represents the prior belief that ˆτ∈[0, τ prior]. For instance, for 1−α= 90% and a reasonably
accurate model, τprior= 30% is appropriate since the true αquantile should lie below 30%. They
design Cito maximize P(ˆfprior(Xi)≤C i), where ˆfprior(·) := ˆfτprior(·). This, in turn, maximizes
this probability for all τ≤τ prior by the monotonicity of ˆfτ. Thus, the search space is reduced to
T= [0, τ prior]. For completeness, this procedure is summarized in Algorithm 1 in Appendix D.1.
Due to space constraints, we include an extended discussion of additional related works in conformal
prediction, survival analysis, and LLM safety in Appendix A.
3 Proposed Method
The proposed adaptive budget allocation strategy operates in two phases. Before these phases, we
first randomly partition our calibration points, indexed by Ical={1, ..., N} , into two disjoint sets,
indexed by Ical1andIcal2, of sizes N1andN2, respectively. We re-index the samples in the first
split asi∈ {1, . . . , N 1}for notational convenience.
In the first phase, we learn an optimal acquisition policy on Ical1. Our allocation strategy relies on
a scoring function St:Ht→R that maps the conversation trajectory observed up to time tto a
real-valued score. A higher score indicates a stronger signal to spend a unit of budget and advance
the attack to the next step. For example, this score could be the estimated probability of observing
an event at the current step, or the expected number of additional iterations required to trigger one.
This scoring function can be trained over the training data Dtrain. Our proposed policy operates
iteratively: at each step t, we compute the current risk score Si(t)and map it to a continuation
probability Pi(t)∈[0,1] . Then, with probability Pi(t), we advance to the next step, i.e., acquire
a new conversational interaction; otherwise, the acquisition for this sample is terminated. This
score-to-probability mapping is optimized over Ical1to minimize the variance of our resulting LPBs
while satisfying a global budget constraint, similarly to [ 12]. In the second phase, we deploy this
policy on the remaining indices Ical2to acquire new interactions. By leveraging the interactions
observed in Ical1, we selectively terminate conversations that are unlikely to contribute to reducing
the LPB variance.
3.1 Phase I: Learning the Optimal Acquisition Policy onI cal1
To learn an optimal mapping from scores to probabilities, we must first expend budget to observe
the full conversation history for all samples in Ical1, until we observe an event, or reach the prior
ˆfprior(Xi). For all i∈ I cal1, we set the censoring times as Ci=ˆfprior(Xi), and expend bi:=
min(T i,ˆfprior(Xi))budget units. We require the total budget Bis sufficient for this observation phase,
such that B >P
i∈Ical1bi. Thus, the weights of these samples are simply wτ(i) = 1,∀i∈ I cal1.
The budget per sample remaining for Phase II is therefore ¯B2:=1
|Ical2| 
B−P
i∈Ical1bi
. Next, we
compute the scores for all observed time steps across the subsetI cal1:
Si(t) =S t(Hi(t)),∀t∈ {1, . . . , b i}, i∈ I cal1.
We use these scores to assign a continuation probability Pi(t)∈[0,1] to each step based on the
scores Si(t)that minimizes the variance of our produced LPBs. For t > b i, we set Pi(t) = 0 since
advancing beyond bidoes not contribute to (3). Following the theoretical coverage bound we develop
in Section 3.3, an optimal probability matrix should minimize the inverse of P(ˆfprior(Xi)≤C i),
without exceeding the budget units that remain for the second step. We also require that these
probabilities are monotonic with respect to the scores Si(t), so we can derive a sound mapping
from the scores to probabilities. By the construction of our policy, we get P(ˆfprior(Xi)≤C i) =
5

Qbi
t=1Pi(t), and therefore the optimization problem is formulated as:
min
P∈[0,1]N1×tmax1
N1N1X
i=11
Qbi
t=1Pi(t)
s.t.1
N1N1X
i=1B(Pi)≤¯B2, P i(t)≤P j(t)⇐⇒S i(t)≤S j(t),∀t, i, j.
Here,B(Pi)is the expected budget expended on the i-th sample: B(Pi) =Pbi
t=1Qt
j=1Pi(j).
We solve this optimization problem using coordinate descent, yielding an optimal sequence of
continuation probabilities {Pi(t)}bi
t=1for each sample i∈ I cal1. To generalize this mapping to the
unseen points in Ical2, we fit a model Mtover the paired tuples {Si(t), P i(t)}i∈Ical1to transport the
risk score into an optimal continuation probability.
3.2 Phase II: Adaptive Acquisition onI cal2
Armed with the sequence of score-to-probability mapping models {Mt}tmax
t=1, we adaptively acquire
interactions for the second split, Ical2. We initialize the process at t= 1 and define a continuation
indicator ξi(0) = 1 . At each step t, we compute the score Si(t) =S t(Hi(t))and transport it into
a continuation probability Pi(t) =M t(Si(t)). We then draw a Bernoulli random variable with
parameterP i(t)to determine if the interaction should proceed. Formally, the continuation indicator
at step tupdates as ξi(t) =ξ i(t−1) with probability Mt(Si(t)), and ξi(t) = 0 otherwise. If
ξi(t) = 1 , we advance to the next step: we acquire the new conversation exchange Hi(t+ 1) ,
sett←t+ 1 , and repeat the process. The interaction terminates when ξi(t) = 0 or we reach
the predefined boundary t= min( ˆfprior(Xi), Ti). Finally, we define the censoring time for each
sample i∈ I cal2as the last successful step: Ci=ˆfprior(Xi)ifξi(Ti) = 1 orξi(ˆfprior(Xi)) = 1 ,
andCi= max{t∈ {0,1, . . . , t max}:ξ i(t) = 1} otherwise. When an event occurs at step
Ti≤ˆfprior(Xi), we terminate the procedure early at this step, and set Ci=ˆfprior(Xi)to indicate that
the budget for this sample is fully accounted for, as the event has been directly observed. Consequently,
for samples i∈ I cal2that successfully reach their target sequence length, i.e., ˆfprior(Xi)≤C i, the
probability of achieving this event is the product of the sequential continuation probabilities:
P(ˆfprior(Xi)≤C i|Xi, Hi, Ti,{(X j, Hj, Tj)}j∈Ical1,Dtrain) =min( ˆfprior(Xi),Ti)Y
t=1Pi(t).
The censored event times are given by ˜Ti= min(C i, Ti). Lastly, we employ Algorithm 1 using
the samples we acquired {(Ci, w(i) =P( ˆfprior(Xi)≤C i)−1,˜Ti)}i∈Ical. Note that while the true
probabilities P(ˆfprior(Xi)≤C i)are unknown for samples where Ci<ˆfprior(Xi), Algorithm 1
evaluates weights for samples satisfying Ci≥ˆfprior(Xi). That is, the unobserved probabilities are
unnecessary for the LPB construction. For convenience, this procedure is summarized in Algorithm 5
in Appendix D.5. In what follows, we prove that this process generates an LPB that achieves the
desired coverage rate while satisfying the budget constraint.
3.3 Theoretical guarantees
In this section, we analyze the coverage rate and budget used by our proposed approach. First, we
show that the LPB constructed by Algorithm 5 holds a PAC-type coverage guarantee.2
Theorem 1(Coverage validity (informal)).Fix a miscoverage level α∈(0,1) and a tolerance level
δ∈(0,1) . Suppose that {(Xi, Ti, Hi)}i∈Icaland(Xtest, Ttest, Htest)are drawn i.i.d., and that ˆfτ(x)
is non-decreasing and continuous in τ. Further assume that there exists a constant ¯w≥1 , such that
almost surely: E[wτ(i)| {(X i, Hi, Ti)}i∈Ical1,Dtrain]≤¯w,∀i∈ I cal2. Then, with probability at
least1−δ over the draw of Dall, the lower predictive bound ˆL(x) generated by Algorithm 5 satisfies:
Ph
Ttest≥ˆL(X test)Dalli
≥1−α−log(1/δ)
3|Ical|−s
log2(1/δ)
9|Ical|2+2 
¯w−α2
log(1/δ)
|Ical|.
2An analogous UPB guarantee holds by reversing the inequality, and is omitted for brevity.
6

Figure 3: Theoretical coverage lower bounds of Theorem 1 (green) and [ 12, Theorem 4.1] (blue) as a
function of the maximum weight,γ, assuming the weights are uniformly distributed.
We defer the proof of Theorem 1 to Appendix C.1. Although the theorem relies on an almost-sure
upper bound ¯won the expected weight, we refer to this bound as the mean weight as optimizing the
mean weight minimizes this upper bound. Furthermore, the almost-sure assumption can be replaced
with a high-probability PAC guarantee over the calibration splits, and we omit it for clarity. While
this proof builds on the theoretical foundations in [ 12,14], we highlight several key implications
that distinguish our result. First, this coverage guarantee holds in finite samples for any quantile
estimator ˆqτ(·), any LLM, and any distribution of the data. Crucially, unlike prior conformal
frameworks [ 12,14], our result does not rely on the assumption of conditional independence between
the censoring times Cand event times Tgiven the covariates X.DAPRO violates this assumption as it
usesTfor early stopping when an event is elicited. Second, our coverage gap scales asp
¯w/N , where
¯wis the mean inverse-censoring weight bound. This is tighter than the bounds derived in [ 12,14],
which scale withp
γ2/N, where γ≥¯w is the maximal, worst-case, weight across all calibration
samples. Motivated by this insight, our Phase I optimization explicitly minimizes the mean weight
to achieve coverage closer to the nominal level. Finally, we remark that this tighter bound is of
independent interest, as our proof technique can be directly applied to the static procedure of [ 12].
Their original motivation for minimizing the mean weight relied on an oracle assumption, in which
the true conditional quantiles are known and the variance of the miscoverage estimator scales linearly
with the mean weight. Our analysis bridges this gap by formally demonstrating that, even in finite
samples and without oracle quantiles, the coverage gap is bounded by the mean weight, providing a
theoretical justification for their optimization objective.
To demonstrate the tightness of our proposed bound, we compare in Figure 3 our theoretical coverage
guarantee with the one developed by [ 12]. We consider a realistic setup with N= 3000 calibration
samples, a target coverage level of 1−α= 0.9 , and a tolerance of δ= 0.05 . For the sake of this
illustration, we assume the inverse-censoring weights are uniformly distributed between 1and a
maximum value γ, yielding a mean weight of ¯w=1
2(1 +γ) . The figure shows that while the prior
bound decreases rapidly as the maximum weight increases, our proposed bound remains significantly
tighter and informative across the entire range.
We now show that when the estimates of our Phase I are sufficiently accurate, the expected budget
consumed by Algorithm 5 satisfies the budget constraint, and we develop a finite-sample bound for
the budget utilized in practice.
Theorem 2(Budget validity (informal)).Suppose that {(Xi, Ti, Hi)}i∈Icalare drawn i.i.d. Further-
more, assume that the empirical probabilities {Pi(t)}t∈{1,...,b i},i∈I cal1learned in Phase I approx-
imate the optimal oracle allocation, and the models {Mt}tmax
t=1 are sufficiently accurate. Then, the
expected budget used by Algorithm 5 is upper bounded by B. Additionally, with probability at least
1−δ, the average budget per sample consumed by Algorithm 5 is bounded by:
1
|Ical|X
i∈Ical˜Ti≤B
|Ical|+tmaxlog(1/δ)
3|Ical|+1
|Ical|s
tmax2log2(1/δ)
9+ 2|I cal2|tmax¯B2log(1/δ).
This theorem is formally stated and proved in Appendix C.2. In Appendix C.2.4, we relax the
assumptions that the models are sufficiently accurate and develop an upper bound that accounts
for the estimation errors of the score-to-probability mapping. While this budget bound scales with
O(tmax2/√Ical1), it is highly conservative: it assumes that estimation errors at every step aggregate
7

and do not cancel each other. In practice, positive and negative errors mostly cancel, and Algorithm 5
satisfies the nominal budget constraint across our experiments. We therefore interpret this bound as a
sample complexity result: it establishes the asymptotic consistency of our approach.
4 Experiments
We evaluate the coverage rate and budget consumption of our approach on attacker-target con-
versations generated using the PAIR algorithm [ 9]. We sample 10,000 prompts from each of the
following datasets: RealToxicityPrompts [ 3] (where the target event Y= 1 is a toxic response) and
the Anthropic Red Team dataset [ 15,16] (where Y= 1 is a successful jailbreak). To demonstrate the
framework’s applicability to utility evaluation, we also include the AutoIF dataset [ 17], a benchmark
for complex instruction following. For AutoIF, the "attacker" is a helper agent, and the event ( Y= 1 )
occurs when the target model successfully satisfies all instructions. To generate the interactions, we
utilize Qwen 2.5 14B Instruct [ 13] as the attacker model, as its outputs had the lowest formatting
error rate among the models we examined. For the target models, we use Qwen 2.5 14B Instruct,
Llama 3.1 8B Instruct [ 18], Phi 4 Mini Instruct [ 19], and Gemma 3 4B Instruct [ 20]. We define
the target event ( Y(t) = 1 ) using an audit function: for the Toxicity dataset, following [ 12], we
use Detoxify [ 21] with a threshold of 0.5 to identify toxic responses. For the Red-Team dataset,
we employ Qwen 2.5 14B Instruct as an LLM-as-a-judge [ 11,22], which scores responses on a
1–10 scale; a score of 10 denotes a successful jailbreak. Finally, for the AutoIF dataset, a successful
instruction-following response event is verified programmatically. In Appendix F.3, we conduct an
experiment with Llama-Guard [ 23] serving as the audit function, which outputs a binary success
label. Additionally, Appendix F.4 evaluates DAPRO on a RAG-based dataset, in which the event is a
hallucination. We cap the maximum sequence length at tmax= 200 . The full details regarding the
dataset generation are provided in Appendix E.1.
We split the data into training (4000), calibration (3000), and test (3000) sets. We fix the training data
and fit a transformer as the predictive model over it, using the full conversation history. We assume
an unlimited budget during the training phase; this work focuses on the calibration phase budget only.
At any given time t1, the model estimates the probability of an event occurring at any future time step
t2> t1. Specifically, it outputs a tmax×(tmax+1) matrix containing P(Y(t 2) = 1|T > t 1, H(t 1))
and the probability of reaching the horizon without an unsafe event, P(T > t max|T > t 1, H(t 1)).
All methods use the same predictive models. In all experiments, we evaluate the methods over 50
independent random splits of the calibration and test sets. During each trial, the calibration algorithms
are employed using the calibration data and evaluated on the test set. We set the average budget per
sample asB/|I cal|= 20. Following [12], we setτ prior= 0.56, andM= 200.
Figure 4: Coverage rate and LPB size of various methods over the RedTeam datasets with Qwen 2.5
serving as attacker, target, and judge.
We compare our proposal to an uncalibrated baseline, which constructs the LPB directly from the raw
model predictions at the target αlevel, ˆqα, and to the static optimized allocation strategy proposed
in [12]. In addition to our main method, we developed two alternative dynamic allocation strategies,
which we introduce in Appendix D. The first is designed to greedily discover as many unsafe events
as possible. The second one is a locally adaptive variant that optimizes the continuation probabilities
step-by-step rather than globally. Since DAPRO attains a lower variance compared to these variants
across our experiments, we focus our main presentation on it and defer the evaluation of these
dynamic variants to Appendix F.
8

Figure 4 presents the empirical coverage rate and LPB size of each method as a function of the target
coverage level on the RedTeam dataset with the Qwen 2.5 model serving as attacker, target, and
judge. This figure demonstrates that all calibration methods attain the desired coverage level, while
the uncalibrated model does not. This is anticipated by our theoretical results. Figure 5 presents the
deviation from the target 90% coverage rate and average budget per sample used by each method
on the Toxicity dataset. This figure reveals that our proposed method satisfies the budget constraint
while attaining the lowest coverage deviation. In addition, Figure 6 details the same metrics on the
AutoIF dataset for both an LPB (target coverage 90%) and an upper predictive bound (UPB, target
coverage 70%). We evaluate the UPB at this lower target coverage since a 90% target pushes the
bound too close to the maximum tmax= 200 , which artificially reduces the variance differences
between methods. Across both bound types, the results reveal the same trend: DAPRO attains a
coverage rate closer to the nominal level compared to the baselines.
Figure 5:Toxicitydataset: absolute coverage deviation (left) and average budget utilized (right)
across four target LLMs. Target coverage rate:90%and target ¯B= 20budget per sample.
Additional experimental setups, including an analysis of the empirical mean weights and the number of
observed unsafe events, are deferred to Appendix F. Furthermore, our ablation study in Appendix F.7
reveals that even when degrading its components, i.e, the score or first data-split size N1,DAPRO is
robust and maintains a tighter coverage rate than the baseline. Finally, beyond constructing individual
LPBs, our framework can be utilized to construct unbiased estimates of population-level evaluation
metrics, such as the unsafe event rate. As demonstrated in Appendix F.6, DAPRO accurately recovers
the true metric quantities with zero bias and a lower variance than existing baselines.
Figure 6:AutoIFdataset: coverage deviation (left) and budget utilized (right) over Qwen 2.5 target
model. Target coverage rate:90%for LPB and70%for UPB, and ¯B= 20budget per sample.
5 Discussion and impact statement
We introduced DAPRO , a novel dynamic budget allocation approach for LLM evaluation. Our approach
can be employed to construct reliable LPBs for individual prompts and extract unbiased population-
level metrics, such as the jailbreak rate. By updating censoring times online based on ongoing
attacker-target interactions, our method achieves lower variance, as supported by our PAC-type
coverage guarantee, which is tighter than prior approaches. Our comprehensive experiments on real
datasets validate these theoretical claims, demonstrating tighter bounds compared to baselines.
While our method satisfies the budget constraint in our empirical simulations, our theoretical expected
budget guarantee relies on the accuracy of the estimated score mapping. Establishing a finite-sample
budget guarantee that does not rely on the accuracy of the model could be a promising research
direction, possibly by relying on risk control theory [ 24,25] or leave-one-out to handle scarce
9

data [ 26–28]. For applications requiring a strict budget guarantee, practitioners can utilize the variants
presented in Appendices D.3 and D.4, which provide exact budget control without relying on the
accuracy of a score mapping. Furthermore, the calibrated quantile level ˆτin our current method is
a single scalar applied identically to all test prompts. An intriguing extension would be to make ˆτ
adaptive to the test prompt Xtest, possibly leveraging the newly acquired calibration data to select a
prompt-specific quantile level, e.g., by utilizing ideas from [ 29,30]. This would allow the LPB to be
tighter for predictable interactions, and more conservative for those with high variance.
We note that, like all conformal prediction-based methods, formal guarantees rely on the standard
conformal assumption of exchangeability between calibration and test sets. Specifically, we assume
the test attacker, target, judge models, and prompt distribution are the same as those used during
calibration. In practice, these may shift over time as models are updated or new attack strategies
emerge. Developing methods that are robust to such distributional shifts [ 31,32] or audit annotation
errors [ 33,34] is an important direction for future work. Finally, while our framework provides
rigorous guarantees to help developers proactively assess prompt risks and set usage limits, its insights
into model failures could potentially be exploited by adversaries, akin to many developments in ML.
Therefore, our proposal must be deployed as a complementary tool within a broader safety, alignment,
and governance pipeline.
Acknowledgments and Disclosure of Funding
Y . R. and S. F. were supported by the European Union (ERC, SafetyBounds, 101163414). Views and
opinions expressed are however those of the authors only and do not necessarily reflect those of the
European Union or the European Research Council Executive Agency. Neither the European Union
nor the granting authority can be held responsible for them. Y .R. thanks the Career Advancement
Fellowship, Technion.
References
[1]Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von
Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the
opportunities and risks of foundation models.arXiv preprint arXiv:2108.07258, 2021.
[2]Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min,
Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models.arXiv
preprint arXiv:2303.18223, 2023.
[3]Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A Smith. RealToxic-
ityPrompts: Evaluating Neural Toxic Degeneration in Language Models. InFindings of the
Association for Computational Linguistics: EMNLP, pages 3356–3369, 2020.
[4]Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-V oss, Kather-
ine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training
data from large language models. In30th USENIX security symposium (USENIX Security 21),
pages 2633–2650, 2021.
[5]Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang,
Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation.
ACM computing surveys, 55(12):1–38, 2023.
[6]Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qiang-
long Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large
language models: Principles, taxonomy, challenges, and open questions.ACM Transactions on
Information Systems, 43(2):1–55, 2025.
[7]Keivalya Pandya and Mehfuza Holia. Automating customer service using langchain: Building
custom open-source gpt chatbot for organizations.arXiv preprint arXiv:2310.05421, 2023.
[8]Yangxinyu Xie, Xuyang Chen, Zhimei Ren, and Weijie J Su. Watermark in the classroom: A
conformal framework for adaptive ai usage detection.arXiv preprint arXiv:2507.23113, 2025.
10

[9]Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J Pappas, and
Eric Wong. Jailbreaking black box large language models in twenty queries. In2025 IEEE
Conference on Secure and Trustworthy Machine Learning (SaTML), pages 23–42. IEEE, 2025.
[10] Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia
Glaese, Nat McAleese, and Geoffrey Irving. Red teaming language models with language
models.arXiv preprint arXiv:2202.03286, 2022.
[11] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and
chatbot arena.Advances in neural information processing systems, 36:46595–46623, 2023.
[12] Hen Davidov, Shai Feldman, Gilad Freidkin, and Yaniv Romano. Calibrated predictive lower
bounds on time-to-unsafe-sampling in LLMs. InThe 29th International Conference on Artificial
Intelligence and Statistics, 2026.
[13] Qwen Team. Qwen2.5: A party of foundation models, September 2024.
[14] Yu Gui, Rohan Hore, Zhimei Ren, and Rina Foygel Barber. Conformalized survival analysis
with adaptive cut-offs.Biometrika, 111(2):459–477, 2024.
[15] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn
Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless
assistant with reinforcement learning from human feedback.arXiv preprint arXiv:2204.05862,
2022.
[16] Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath,
Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, et al. Red teaming language
models to reduce harms: Methods, scaling behaviors, and lessons learned.arXiv preprint
arXiv:2209.07858, 2022.
[17] Guanting Dong, Keming Lu, Chengpeng Li, Tingyu Xia, Bowen Yu, Chang Zhou, and Jingren
Zhou. Self-play with execution feedback: Improving instruction-following capabilities of large
language models. InThe Thirteenth International Conference on Learning Representations,
2025.
[18] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian,
Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The
llama 3 herd of models.arXiv preprint arXiv:2407.21783, 2024.
[19] Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach,
Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, et al. Phi-4-mini
technical report: Compact yet powerful multimodal language models via mixture-of-loras.
arXiv preprint arXiv:2503.01743, 2025.
[20] Gemma Team. Gemma 3 technical report.arXiv preprint arXiv:2503.19786, 2025.
[21] Laura Hanu and Unitary team. Detoxify. Github. https://github.com/unitaryai/detoxify, 2020.
[22] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan
Shen, Shengjie Ma, Honghao Liu, et al. A survey on llm-as-a-judge.The Innovation, 2024.
[23] Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao,
Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, et al. Llama guard: Llm-based
input-output safeguard for human-ai conversations.arXiv preprint arXiv:2312.06674, 2023.
[24] Anastasios N Angelopoulos, Stephen Bates, Emmanuel J Candès, Michael I Jordan, and Lihua
Lei. Learn then test: Calibrating predictive algorithms to achieve risk control.arXiv preprint
arXiv:2110.01052, 2021.
11

[25] Anastasios N Angelopoulos. Conformal risk control for non-monotonic losses.arXiv preprint
arXiv:2602.20151, 2026.
[26] Rina Foygel Barber, Emmanuel J Candès, Aaditya Ramdas, and Ryan J Tibshirani. Predictive
inference with the jackknife+.The Annals of Statistics, 49(1):486–507, 2021.
[27] Kiljae Lee and Yuan Zhang. Leave-one-out stable conformal prediction.arXiv preprint
arXiv:2504.12189, 2025.
[28] Chirag Gupta, Arun K Kuchibhotla, and Aaditya Ramdas. Nested conformal prediction and
quantile out-of-bag ensemble methods.Pattern Recognition, 127:108496, 2022.
[29] Shai Feldman, Stephen Bates, and Yaniv Romano. Improving conditional coverage via orthogo-
nal quantile regression.Advances in neural information processing systems, 34:2060–2071,
2021.
[30] Isaac Gibbs, John J Cherian, and Emmanuel J Candès. Conformal prediction with condi-
tional guarantees.Journal of the Royal Statistical Society Series B: Statistical Methodology,
87(4):1100–1126, 2025.
[31] Isaac Gibbs and Emmanuel Candes. Adaptive conformal inference under distribution shift.
Advances in Neural Information Processing Systems, 34:1660–1672, 2021.
[32] Shai Feldman, Liran Ringel, Stephen Bates, and Yaniv Romano. Achieving risk control in
online learning settings.Transactions on Machine Learning Research, 2023.
[33] Bat-Sheva Einbinder, Shai Feldman, Stephen Bates, Anastasios N Angelopoulos, Asaf Gendler,
and Yaniv Romano. Label noise robustness of conformal prediction.Journal of Machine
Learning Research, 25(328):1–66, 2024.
[34] Coby Penso and Jacob Goldberger. A conformal prediction score that is robust to label noise.
arXiv preprint arXiv:2405.02648, 2024.
[35] Bertie Vidgen, Nino Scherrer, Hannah Rose Kirk, Rebecca Qian, Anand Kannappan, Scott A
Hale, and Paul Röttger. SimpleSafetyTests: a Test Suite for Identifying Critical Safety Risks in
Large Language Models.arXiv preprint arXiv:2311.08370, 2023.
[36] Traian Rebedea, Razvan Dinu, Makesh Narsimhan Sreedhar, Christopher Parisien, and Jonathan
Cohen. Nemo guardrails: A toolkit for controllable and safe llm applications with programmable
rails. InProceedings of the 2023 conference on empirical methods in natural language
processing: system demonstrations, pages 431–445, 2023.
[37] Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson.
Universal and transferable adversarial attacks on aligned language models.arXiv preprint
arXiv:2307.15043, 2023.
[38] Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. Jailbroken: How does llm safety training
fail?Advances in neural information processing systems, 36:80079–80110, 2023.
[39] Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron
Singer, and Amin Karbasi. Tree of attacks: Jailbreaking black-box llms automatically.Advances
in Neural Information Processing Systems, 37:61065–61105, 2024.
[40] Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, and
Himabindu Lakkaraju. Certifying llm safety against adversarial prompting.arXiv preprint
arXiv:2309.02705, 2023.
[41] Alexander Robey, Eric Wong, Hamed Hassani, and George J Pappas. Smoothllm: Defending
large language models against jailbreaking attacks.arXiv preprint arXiv:2310.03684, 2023.
12

[42] Jannik Kossen, Sebastian Farquhar, Yarin Gal, and Tom Rainforth. Active testing: Sample-
efficient model evaluation. InInternational Conference on Machine Learning, pages 5753–5763.
PMLR, 2021.
[43] Yang Li, Jie Ma, Miguel Ballesteros, Yassine Benajiba, and Graham Horwood. Active evaluation
acquisition for efficient llm benchmarking.arXiv preprint arXiv:2410.05952, 2024.
[44] Jin Peng Zhou, Christian K Belardi, Ruihan Wu, Travis Zhang, Carla P Gomes, Wen Sun,
and Kilian Q Weinberger. On speeding up language model evaluation. InThe Thirteenth
International Conference on Learning Representations, 2025.
[45] Ganghua Wang, Zhaorun Chen, Bo Li, and Haifeng Xu. Cer-eval: Certifiable and cost-efficient
evaluation framework for llms.arXiv preprint arXiv:2505.03814, 2025.
[46] Vladimir V ovk, Alexander Gammerman, and Glenn Shafer.Algorithmic Learning in a Random
World, volume 29. Springer, 2005.
[47] Glenn Shafer and Vladimir V ovk. A tutorial on conformal prediction.Journal of machine
learning research, 9(3), 2008.
[48] Harris Papadopoulos, Kostas Proedrou, V olodya V ovk, and Alex Gammerman. Inductive
confidence machines for regression. InEuropean conference on machine learning, pages
345–356. Springer, 2002.
[49] Ryan J Tibshirani, Rina Foygel Barber, Emmanuel Candes, and Aaditya Ramdas. Conformal
prediction under covariate shift.Advances in neural information processing systems, 32, 2019.
[50] Emmanuel Candès, Lihua Lei, and Zhimei Ren. Conformalized survival analysis.Journal of
the Royal Statistical Society Series B: Statistical Methodology, 85(1):24–45, 2023.
[51] Hen Davidov, Shai Feldman, Gil Shamai, Ron Kimmel, and Yaniv Romano. Conformalized
Survival Analysis for General Right-Censored Data. InInternational Conference on Learning
Representations, 2025.
[52] Sangdon Park, Osbert Bastani, Nikolai Matni, and Insup Lee. Pac confidence sets for deep
neural networks via calibrated prediction.arXiv preprint arXiv:2001.00106, 2019.
[53] Stephen Bates, Anastasios Angelopoulos, Lihua Lei, Jitendra Malik, and Michael Jordan.
Distribution-Free, Risk-Controlling Prediction Sets.Journal of the ACM (JACM), 68(6):1–34,
2021.
[54] Stephen Bates, Emmanuel Candès, Lihua Lei, Yaniv Romano, and Matteo Sesia. Testing for
outliers with conformal p-values.The Annals of Statistics, 51(1):149–178, 2023.
[55] Burr Settles. Active learning literature survey.University of Wisconsin, Madison, 52, 07 2010.
[56] Valerii Vadimovich Fedorov.Theory of optimal experiments. Elsevier, 2013.
[57] Tijana Zrnic and Emmanuel Candès. Active Statistical Inference. InInternational Conference
on Machine Learning, 2024.
[58] Jinyong Hahn, Keisuke Hirano, and Dean Karlan. Adaptive experimental design using the
propensity score.Journal of Business & Economic Statistics, 29(1):96–108, 2011.
[59] Martin J Wainwright.High-dimensional statistics: A non-asymptotic viewpoint, volume 48.
Cambridge university press, 2019.
[60] Aleksandrs Slivkins. Introduction to multi-armed bandits.Foundations and Trends® in Machine
Learning, 12(1-2):1–286, 2019.
13

[61] Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed
bandit problem.Machine learning, 47(2):235–256, 2002.
[62] Shipra Agrawal and Navin Goyal. Analysis of thompson sampling for the multi-armed bandit
problem. InConference on learning theory, pages 39–1. JMLR Workshop and Conference
Proceedings, 2012.
[63] Anastasios Nikolas Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster.
Conformal risk control. InThe Twelfth International Conference on Learning Representations,
2024.
[64] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+
questions for machine comprehension of text. In Jian Su, Kevin Duh, and Xavier Carreras,
editors,Proceedings of the 2016 Conference on Empirical Methods in Natural Language
Processing, pages 2383–2392, Austin, Texas, November 2016. Association for Computational
Linguistics.
[65] Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don’t know: Unanswerable
questions for SQuAD. In Iryna Gurevych and Yusuke Miyao, editors,Proceedings of the 56th
Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers),
pages 784–789, Melbourne, Australia, July 2018. Association for Computational Linguistics.
[66] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge,
Yu Han, Fei Huang, et al. Qwen technical report.arXiv preprint arXiv:2309.16609, 2023.
[67] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony
Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. Huggingface’s transform-
ers: State-of-the-art natural language processing.arXiv preprint arXiv:1910.03771, 2019.
[68] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu,
Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language
model serving with pagedattention. InProceedings of the 29th symposium on operating systems
principles, pages 611–626, 2023.
[69] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei.
Multilingual e5 text embeddings: A technical report.arXiv preprint arXiv:2402.05672, 2024.
[70] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is all you need.Advances in neural information
processing systems, 30, 2017.
[71] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. InInternational
Conference on Learning Representations, 2019.
[72] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan,
Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas
Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy,
Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style,
high-performance deep learning library. InAdvances in Neural Information Processing Systems
32, pages 8024–8035. Curran Associates, Inc., 2019.
[73] F. Pedregosa, G. Varoquaux, A. Gramfort, V . Michel, B. Thirion, O. Grisel, M. Blondel,
P. Prettenhofer, R. Weiss, V . Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher,
M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python.Journal of Machine
Learning Research, 12:2825–2830, 2011.
14

A Additional Related work
Our work sits at the intersection of LLM safety evaluation, conformal prediction, and survival analysis.
Prior efforts in safety evaluation mostly focus on detecting harmful content in model outputs [ 3,35] or
constructing guardrails at inference time [ 23,36]. A complementary line of work takes an adversarial
perspective, generating red-teaming attacks to probe model vulnerabilities [ 9,10,37–39]. These
works aim to maximize attack success. On the defensive side, recent efforts in certified LLM
safety aim to provide statistical guarantees against such attacks, typically by bounding the failure
probability of a single prompt using techniques like randomized smoothing [ 40,41]. In contrast,
our goal is to provide rigorous statistical guarantees on prompt risk using the multi-turn interaction
structure that adversarial settings naturally produce, and explicitly evaluating the time-to-event. While
existing adversarial red-teaming approaches effectively investigate these multi-turn vulnerabilities,
they do not provide statistical guarantees, as they do not account for the distribution shift induced by
right-censoring.
While our framework was designed to provide rigorous safety bounds, it can be employed for sample-
efficient LLM evaluation [ 42]. To reduce computational costs and API calls when evaluating large
models, recent methods utilize active testing [ 43], multi-armed bandits [ 44], and adaptive dataset
partitioning [ 45]. Our framework directly complements these efforts: by framing metric evaluation as
a sequential, budget-constrained survival problem, DAPRO produces theoretically valid estimates with
low variance under resource constraints.
Our theoretical framework builds upon conformal prediction [ 46,47], a powerful framework for con-
structing prediction sets with finite-sample validity guarantees and without distributional assumptions.
Split conformal prediction [ 48] and its extension to covariate shift via weighting [ 49], are utilized by
the calibration procedure we adopt from [ 12]. Conformal prediction has recently been extended to
survival analysis [ 14,50,51], where the goal is to construct lower predictive bounds on event times
under censoring. Our work builds directly on [ 12], who apply conformalized survival analysis to
LLM safety by introducing the time-to-unsafe-sampling metric and a static, variance-minimizing
budget allocation scheme. We extend this framework to the adaptive setting where conversation
evolves, and our coverage proof provides a tighter PAC-type guarantee [52–54].
Finally, the variance-minimization phase of our framework is related to active learning [ 55] and
optimal experimental design [56], which study how to allocate a limited budget across observations
to minimize estimator variance. In particular, [ 57] formalizes active data collection for statistical
inference while maintaining valid guarantees, and [ 58] develops adaptive designs that minimize
treatment effect estimation variance. Our contribution is adapting these ideas to the survival analysis
and LLM safety setting, where observations arrive sequentially, and the allocation policy must be
learned from a first data split before being projected on a second.
B Extension to LLM utility evaluation and upper predictive bounds (UPB)
While the main text focuses on safety risk assessment, our framework extends to evaluating LLM
utility in multi-turn interactions. In this setting, the adversarial "attacker" is a "helper agent" aiming
to guide the target model toward successful task completion, such as satisfying complex constraints,
e.g., using a specific word multiple times, as in the AutoIF dataset [ 17]. Here, the event of interest
(Y= 1 ) is defined as task success. Instead of bounding the time-to-unsafe with a lower predictive
bound (LPB), we construct an upper predictive bound (UPB), denoted by ˆU(X) . The UPB provides
a finite-sample guarantee that the time-to-success Twill not exceed ˆU(X) with probability 1−α ,
i.e.,P(T≤ ˆU(X))≥1−α . As illustrated in Figure 7, the dynamic allocation mechanism remains
identical to the safety setting. Technically, a low UPB indicates that the task is highly feasible,
requiring few interactions to succeed, whereas a UPB reaching the maximum horizon tmaxflags the
task as practically infeasible. The theoretical coverage guarantees derived for the LPB apply to the
UPB by simply reversing the target inequality direction.
15

Figure 7: Illustration of our framework for LLM utility evaluation: (i) collecting data via dynamic
budget allocation; (ii) calibrating a pre-trained model; and (iii) deploying the model at inference time
to assess task feasibility.3
C Theory
In this section, we present our theoretical guarantees. For ease of notations, we abuse some of the
notations made in the first part and set Dcal={(X i, Hi, Ti)}i∈Ical,Dcal1={(X i, Hi, Ti)}i∈Ical1
andD cal2={(X i, Hi, Ti)}i∈Ical2. Hence, the combined data is denoted asD all=D train∪ Dcal.
C.1 Coverage rate guarantee of Algorithm 5
Before presenting the coverage validity guarantee of our proposed algorithm, we first show that the
weights it uses are correct.
Proposition 1(Valid weights of Algorithm 5).The weights used by Algorithm 5 equal to wτ(i) =
Ph
Ci≥ˆfτ(Xi)Xi, Hi, Ti,Dcal1,Dtraini−1
for all i∈ I calsuch that Ci=ˆfprior(Xi)andτ∈
T= [0, τ prior].
Proof. First, for all i∈ I cal1,Ci=ˆfprior(Xi)deterministically, and hence, wτ(i) = 1 =
Ph
Ci≥ˆfprior(Xi)Xi, Hi, Ti,Dcal1,Dtraini−1
. The above holds since ˆfτis non-decreasing in
τandτ≤τ prior, we have ˆfτ(Xi)≤ ˆfprior(Xi) =C ifor all τ∈ T , giving P(Ci≥ˆfτ(Xi)|
Xi, Hi, Ti,Dcal1,Dtrain) = 1 deterministically, and hence wτ(i) = 1 . For any i∈ I cal2, the weights
used by the algorithm are:
wτ(i) =
min( ˆfprior(Xi),Ti)Y
t=1Pi(t)
−1
.
Above, ∀i∈ I cal2:Pi(t) =M t(Si(t))are the outputs of the mapping models. By our construction,
Pi(t)≡P(ξ i(t) = 1|ξ i(t−1) = 1, X i, Hi, Ti,Dcal1,Dtrain). Furthermore, ξi(t) =ξ i(t−1)·Z(i)
t
where Zi
t∼Bern(P i(t))are mutually independent given Xi, Hi, Ti,Dcal1,Dtrain. Since the event
1{ξi(k) = 1} is equivalent to {Z(i)
1= 1, . . . , Z(i)
k= 1} , the chain rule and mutual independence
give:
min( ˆfprior(Xi),Ti)Y
t=1Pi(t) =min( ˆfprior(Xi),Ti)Y
t=1P(ξi(t) = 1|ξ i(t−1) = 1, X i, Hi, Ti,Dcal1,Dtrain)
=P(ξ i(min( ˆfprior(Xi), Ti)) = 1|X i, Hi, Ti,Dcal1,Dtrain).
By the definition ofC ias:
Ci:=

ˆfprior(Xi)ifi∈ I cal1,
ˆfprior(Xi)ifi∈ I cal2, ξi(min(T i,ˆfprior(Xi))) = 1,
max{t∈ {0, . . . , t max}:ξ i(t) == 1}otherwise,
3This figure was generated using Google Gemini based on a prompt designed by the authors and subsequently refined.
16

we get:
P(ˆfprior(Xi)≤C i|Xi, Hi, Ti,Dcal1,Dtrain)
=P(C i=ˆfprior(Xi)|X i, Hi, Ti,Dcal1,Dtrain)
=P(ξ i(min( ˆfprior(Xi), Ti)) = 1|X i, Hi, Ti,Dcal1,Dtrain)
=min( ˆfprior(Xi),Ti)Y
t=1Pi(t).
Observe that for Ci=ˆfprior(Xi)andτ≤τ prior we get Ci≥ˆfτ(Xi)>0 by the monotonicity of ˆfτ
and therefore:
P(ˆfprior(Xi)≤C i|Xi, Hi, Ti,Dcal1,Dtrain) =P( ˆfτ(Xi)≤C i|Xi, Hi, Ti,Dcal1,Dtrain).
We now turn to present the main coverage rate guarantee.
Theorem 3.Fix a miscoverage level α∈(0,1) and a tolerance level δ∈(0,1) . Suppose that
{(Xi, Ti, Hi)}i∈Icaland(Xtest, Ttest, Htest)are drawn i.i.d., and that ˆfτ(x)is non-decreasing and
continuous in τ. Further assume that there exists a constant ¯w≥1 , which may depend on Dtrain but
not onD cal1, such that almost surely overD cal1:
E[wτ(i)| D cal1,Dtrain]≤¯wfor alli∈ I cal2.
Define the coverage gap:
∆ :=log(1/δ)
3|Ical|+s
log2(1/δ)
9|Ical|2+2 
¯w−α2
log(1/δ)
|Ical|.
Then, with probability at least 1−δ over the draw of Dall= (D train,Dcal), the lower predictive
bound ˆL(x)generated by Algorithm 5 satisfies:
Ph
Ttest≥ˆL(X test)Dalli
≥1−α−∆.
Proof. First, we note that, in practice, we do not compute the entire sequential continuation indicators
ξi(t). However, for the sake of this proof, indicators ξi(t)can be viewed as an infinite sequence
generated via Zi
t∼Ber(M t(Si(t))) , which is dependent only on the history Hi, features Xi, and
calibration data Dcal1. The early stopping at Tiis conducted for practical purposes to satisfy the
budget constraint, but the underlying distribution of the censoring variable Cirelies solely on
Bernoulli random variables that are perfectly independent of the target sequence length Ti. Thus, by
the construction of the censoring times, and following the i.i.d. assumption, we have
1.By construction, for all i∈ I cal2:(ξi, Ti)⊥ {(ξ j, Tj)}j∈Ical2\{i} |
{(Xk, Hk, Tk)}k∈I cal2,Dcal1. Therefore, given {(Xk, Hk, Tk)}k∈I cal2andDcal1, the cen-
soring times {Ci}i∈Ical2are mutually independent, since Ciis a deterministic function of
(ξi, Ti, Xi, Hi).
2.The marginal law of (Xi, Ti, Hi)fori∈ I cal2is independent of Dcal1givenDtrain, and
{(Xi, Ti, Hi)}i∈Ical2are mutually independent given(D cal1,Dtrain).
By Proposition 1, the weights used by the algorithm are correct: wτ(i) =
Ph
Ci≥ˆfτ(Xi)Xi, Hi, Ti,Dcal1,Dtraini−1
for all i∈ I calsuch that Ci=ˆfprior(Xi)and
τ∈ T= [0, τ prior]. Every i∈ I calsuch that Ci̸=ˆfprior(Xi), does not contribute to the weighted
sum in (3), so we do not need access to its true weight. Therefore, all assumptions of Theorem 4 hold,
and thus the LPB generated by Algorithm 5 is a valid PAC-type LPB with the desired bounds.
17

We now present the general theorem for coverage validity guarantee.
Theorem 4(General Coverage Validity).Fix a miscoverage level α∈(0,1) and a tolerance level
δ∈(0,1) . LetDtrain be the training set, and let Ical=I cal1⊔ Ical2be a disjoint partition of
the calibration set indexes, with |Ical1|+|I cal2|=|I cal|. Suppose that {(Xi, Ti, Hi)}i∈Icaland
(Xtest, Ttest, Htest)are drawn i.i.d., that ˆfτ(x)is non-decreasing and continuous in τ, with ˆfτ(x)>0
for allxandτ >0. We assume that fori∈ I cal1the weights are given byw τ(i) = 1, and
wτ(i) =1
Ph
Ci≥ˆfτ(Xi)Xi, Hi, Ti,Dcal1,Dtrainifori∈ I cal2.
Assume the following conditions hold:
1.(Mutual independence)For all i∈ I cal2:(Ci, Ti)⊥ {(C j, Tj)}j∈Ical2\{i}|
{(Xk, Hk, Tk)}k∈I cal2,Dcal1.
2.(Marginal independence of calibration data)The marginal law of (Xi, Ti, Hi)fori∈ I cal2
is independent of Dcal1givenDtrain, and{(Xi, Ti, Hi)}i∈Ical2are mutually independent
given(D cal1,Dtrain).
3.(Bounded mean weight)There exists a constant ¯w≥1 , which may depend on Dtrain but
not onD cal1, such that almost surely overD cal1:
E[wτ(i)| D cal1,Dtrain]≤¯wfor alli∈ I cal2.
Define the estimated miscoverage rate and calibrated quantile level by:
ˆα(τ) :=1
|Ical|X
i∈Icalwτ(i)In
˜Ti<ˆfτ(Xi)≤C io
,ˆτ:= sup
τ∈ T: sup
τ′≤τˆα(τ′)≤α
.
We remark that for ˆτto be well defined, we assume there exists τ′∈ T= [0, τ prior]such that
ˆα(τ′)≤α . This assumption can be trivially satisfied by including τ= 0+inTwith ˆf0(x) = 0 ,
which yieldsˆα(0) = 0≤α.
Define the coverage gap:
∆ :=log(1/δ)
3|Ical|+s
log2(1/δ)
9|Ical|2+2 
¯w−α2
log(1/δ)
|Ical|.
Then, with probability at least 1−δ over the draw of Dall= (D train,Dcal), the lower predictive
bound ˆL(x) = ˆfˆτ(x)satisfies:
Ph
Ttest≥ˆL(X test)Dalli
≥1−α−∆.
Proof.We define the oracle miscoverage level by:
τ(α+ ∆) = supn
λ∈ T:P
T < ˆfλ(X)| D train
≤α+ ∆o
.
The above set is non-empty since τ= 0 satisfies P(T < ˆf0(X)| D train) =P(T <0| D train) =
0≤α+ ∆by the convention ˆf(x) = 0.
Observe that it suffices to show 1−δ≤P(ˆτ≤τ(α+∆)| D train), since on the event {ˆτ≤τ(α+∆)} ,
the monotonicity of ˆfτinτand the left-continuity ofτ7→P(T≥ ˆfτ(X)| D all)give:
P(T≥ ˆfˆτ(X)| D all)≥P(T≥ ˆfτ(α+∆) (X)| D all)≥1−α−∆.
It therefore remains to show P(ˆτ≤τ(α+ ∆)| D train)≥1−δ . Fix ε >0 and set λ:=
τ(α+ ∆) +ε . We claim that {ˆα(λ)> α} ⊆ {ˆτ≤λ} . Indeed, if ˆα(λ)> α , then for every τ≥λ ,
supτ′≤τˆα(τ′)≥ˆα(λ)> α , so no τ≥λ belongs to the set {τ∈ T: supτ′≤τˆα(τ′)≤α} , hence
ˆτ= sup{τ∈ T: supτ′≤τˆα(τ′)≤α} ≤λ. It therefore suffices to show:
P(ˆα(λ)≤α| D train)≤δ,(4)
18

since this gives P(ˆτ≤λ| D train)≥1−δ for every ε >0 . Taking ε→0+and applying continuity
of the probability measure to the decreasing family of events {ˆτ≤τ(α+∆)+ε} (whose intersection
equals{ˆτ≤τ(α+ ∆)}) gives:
P(ˆτ≤τ(α+ ∆)| D train) = lim
ε→0+P(ˆτ≤τ(α+ ∆) +ε| D train)≥1−δ,
as required.
We define:
Sλ:=X
i∈Icalwλ(i)I{T i<ˆfλ(Xi)}I{C i≥ˆfλ(Xi)},
S(1)
λ:=X
i∈Ical1wλ(i)I{T i<ˆfλ(Xi)}I{C i≥ˆfλ(Xi)},
S(2)
λ:=X
i∈Ical2wλ(i)I{T i<ˆfλ(Xi)}I{C i≥ˆfλ(Xi)},
so that Sλ=S(1)
λ+S(2)
λandˆα(λ) =S λ/|Ical|. By construction of Cifori∈ I cal1:Ci=ˆfprior(Xi)
deterministically and wλ(i) = 1 (since wλ(i)−1=P(C i≥ˆfλ(Xi)|Xi, Hi, Ti,Dcal1,Dtrain) = 1 ),
soI{C i≥ˆfλ(Xi)}= 1deterministically, giving:
S(1)
λ=X
i∈Ical1I{Ti<ˆfλ(Xi)}=:N(1)
λ.
Lastly, we setA:= ¯w−α2≥1−α2>0forα∈(0,1).
Step 1: Chernoff bound givenD cal1.By the tower property:
P(ˆα(λ)≤α| D train) =Eh
P(N(1)
λ+S(2)
λ≤ |I cal|α| D cal1,Dtrain)Dtraini
.(5)
Conditioning on Dcal1makes N(1)
λa deterministic non-negative integer. Notice that ¯wis also
deterministic given Dtrain by its definition. Since N(1)
λis deterministic given Dcal1, Markov’s
inequality applied to the non-negative random variable exp
−t(N(1)
λ+S(2)
λ− |I cal|α)
≥0gives,
for anyt >0:
P(N(1)
λ+S(2)
λ≤ |I cal|α| D cal1,Dtrain)≤et|Ical|α−tN(1)
λ·Eh
e−tS(2)
λ| Dcal1,Dtraini
.(6)
Step 2: Bounding E[e−tS(2)
λ| D cal1,Dtrain].For i∈ I cal2, define Vi:=w λ(i)I{T i<
ˆfλ(Xi)}I{C i≥ˆfλ(Xi)}, soS(2)
λ=P
i∈Ical2Vi.
Mutual independence of {Vi}i∈Ical2 given (Dcal1,Dtrain).By the i.i.d. assumption,
{(Xi, Ti, Hi)}i∈Ical2are mutually independent given (Dcal1,Dtrain)(the global i.i.d. assumption
gives cross- iindependence, and the marginal law of each (Xi, Ti, Hi)is independent of Dcal1given
Dtrain). By the mutual independence assumption, given {(Xk, Hk, Tk)}k∈I cal2andDcal1, the cen-
soring times {Ci}i∈Ical2are mutually independent. Combining these via the tower property: for any
measurablef i(Xi, Ti, Hi, Ci)and any finiteS⊆ I cal2,
E"Y
i∈Sfi(Xi, Ti, Hi, Ci)Dcal1,Dtrain#
=E"
E"Y
i∈Sfi(Xi, Ti, Hi, Ci){(Xk, Hk, Tk)}k∈I cal2,Dcal1,Dtrain#Dcal1,Dtrain#
=E"Y
i∈SE[fi(Xi, Ti, Hi, Ci)|X i, Ti, Hi,Dcal1,Dtrain]Dcal1,Dtrain#
=Y
i∈SE[fi(Xi, Ti, Hi, Ci)|D cal1,Dtrain],(7)
19

where the second equality uses mutual independence of {Ci}i∈Ical2given{(Xk, Hk, Tk)}k∈I cal2and
Dcal1, and the third uses mutual independence of {(Xi, Ti, Hi)}i∈Ical2given (Dcal1,Dtrain). Since
Viis a measurable function of (Xi, Ti, Hi, Ci),(7)establishes mutual independence of {Vi}i∈Ical2
given(D cal1,Dtrain).
Mean of Vi(IPCW identity).By the tower property, conditioning first on (Xi, Hi, Ti,Dcal1,Dtrain)
and using the weight definitionw λ(i)−1=P(C i≥ˆfλ(Xi)|X i, Hi, Ti,Dcal1,Dtrain):
E[Vi| Dcal1,Dtrain]
=E
wλ(i)I{T i<ˆfλ(Xi)}Eh
I{Ci≥ˆfλ(Xi)} |X i, Hi, Ti,Dcal1,Dtraini
| {z }
=wλ(i)−1Dcal1,Dtrain

=Eh
I{Ti<ˆfλ(Xi)}Dcal1,Dtraini
=P(T < ˆfλ(X)| D train) =:P λ,
where the last equality uses the i.i.d assumption.
Second moment and variance ofV i.
SinceV iis a product of three terms, we have:
V2
i=
wλ(i)I{T i<ˆfλ(Xi)}I{C i≥ˆfλ(Xi)}2
=w λ(i)2I{Ti<ˆfλ(Xi)}2I{Ci≥ˆfλ(Xi)}2
=w λ(i)2I{Ti<ˆfλ(Xi)}I{C i≥ˆfλ(Xi)}.
By the tower property of conditional expectation:
E
V2
iXi, Hi, Ti,Dcal1,Dtrain
=w λ(i)2I{Ti<ˆfλ(Xi)}w λ(i)−1=w λ(i)I{T i<ˆfλ(Xi)},
therefore:
E[V2
i| Dcal1,Dtrain] =Eh
wλ(i)I{T i<ˆfλ(Xi)}Dcal1,Dtraini
≤E[w λ(i)| D cal1,Dtrain]≤¯w.
One-sided Bernstein MGF bound for S(2)
λ.Define mean-zero variables Y(2)
i:=P λ−Vifori∈ I cal2.
SinceV i≥0, we haveY(2)
i≤Pλ≤1.
Since Y(2)
i:=P λ−ViandPλ=E[V i| Dcal1,Dtrain]is deterministic given (Dcal1,Dtrain), we
have:
Var(Y(2)
i| Dcal1,Dtrain) = Var(P λ−Vi| Dcal1,Dtrain)
= Var(V i| Dcal1,Dtrain)
=E[V2
i| Dcal1,Dtrain]−P2
λ
≤¯w−P2
λ
≤¯w−α2=:A
where the second equality uses the fact that adding or subtracting a constant does not change
variance and the last inequality follows from the definition of the supremum τ(α+ ∆) and since
λ=τ(α+ ∆) +ε > τ(α+ ∆), we haveP λ> α+ ∆.
We apply [ 59, Proposition 2.14] for Y(2)
iwithY(2)
i≤1 a.s., and E
Y(2)
i2
| Dcal1,Dtrain
=
Varh
Y(2)
i| Dcal1,Dtraini
≤Ato get:
E"
exp 
tX
i∈Ical2Y(2)
i!Dcal1,Dtrain#
≤expt2|Ical2|A/2
1−t/3
,0< t <3.
20

SinceS(2)
λ=|I cal2|Pλ−P
i∈Ical2Y(2)
i:
Eh
e−tS(2)
λ| Dcal1,Dtraini
≤exp
−t|I cal2|Pλ+t2|Ical2|A/2
1−t/3
,0< t <3.(8)
The right-hand side of (8)depends only on the deterministic quantities Pλ,A,|Ical2|, and t, and in
particular does not depend onD cal1.
Step 3: Bounding E[e−tN(1)
λ| Dtrain].Under our assumptions, {(Xi, Ti)}i∈Ical1are i.i.d. with
marginal law independent of Dcal2givenDtrain. Hence Z(1)
i:=I{T i<ˆfλ(Xi)} ∼Bernoulli(P λ)
are i.i.d. givenD train, andN(1)
λ=P
i∈Ical1Z(1)
i.
Define mean-zero Ui:=P λ−Z(1)
ifori∈ I cal1, so{Ui}i∈Ical1are i.i.d. given Dtrain. Since
Z(1)
i≥0,U i≤Pλ≤1, so the one-sided boundU i≤1holds.
Variance of Uibounded by A.Var(U i| Dtrain) =P λ(1−P λ) =P λ−P2
λ. Since wλ(i)≥1 implies
¯w≥1≥P λ(Pλis a probability), and sinceP λ≥α+ ∆> αimpliesP2
λ> α2:
Pλ(1−P λ) =P λ−P2
λ≤¯w−α2=A.
We apply [ 59, Proposition 2.14] with b= 1 andν=A to each i.i.d. Ui, summing over |Ical1|terms,
and writingN(1)
λ=|I cal1|Pλ−P
i∈Ical1Ui:
Eh
e−tN(1)
λ| Dtraini
=e−t|I cal1|Pλ·E"
exp 
tX
i∈Ical1Ui!Dtrain#
≤exp
−t|I cal1|Pλ+t2|Ical1|A/2
1−t/3
,0< t <3.(9)
Step 4: Combining the bounds.Applying (6)pointwise in Ical1, then taking the outer expectation
overI cal1via (5):
P(ˆα(λ)≤α| D train)≤et|Ical|α·Eh
e−tN(1)
λ·Eh
e−tS(2)
λ| Dcal1,DtrainiDtraini
.
Since the upper bound in (8)is deterministic (does not depend on Dcal1), it factors out of the outer
expectation:
P(ˆα(λ)≤α| D train)≤et|Ical|α·exp
−t|I cal2|Pλ+t2|Ical2|A/2
1−t/3
·Eh
e−tN(1)
λ| Dtraini
.
Applying (9) and using|I cal1|+|I cal2|=|I cal|:
P(ˆα(λ)≤α| D train)≤exp
t|Ical|α−t|I cal2|Pλ−t|I cal1|Pλ+t2(|Ical2|+|I cal1|)A/2
1−t/3
= exp
−t|I cal|(Pλ−α) +t2|Ical|A/2
1−t/3
≤exp
−t|I cal|∆ +t2|Ical|A/2
1−t/3
=: exp(E(t)),
for0< t <3 , where the last inequality uses Pλ−α≥∆ (since λ > τ(α+ ∆) implies
Pλ=P(T < ˆfλ(X)| D train)≥α+ ∆, by definition ofτ(α+ ∆)and continuity of ˆfτ).
Step 5: Evaluation at the Bernstein point and solving for ∆.Set N:=|I cal|, and recall
A= ¯w−α2>0. We evaluateEat:
t∗:=∆
A+ ∆/3.
21

Notice thatt∗∈(0,3)since∆, A >0. Substituting intoE(t∗):
E(t∗) =−N∆2
A+ ∆/3+NA
2·∆2
(A+ ∆/3)2·1
A/(A+ ∆/3)
=−N∆2
A+ ∆/3+NA
2·∆2
A(A+ ∆/3)
=−N∆2
A+ ∆/3+N∆2
2(A+ ∆/3)
=−N∆2
2(A+ ∆/3)
Therefore,
exp(E(t∗))≤δ⇐⇒N∆2
2(A+ ∆/3)≥log1
δ
.(10)
Multiplying both sides by2(A+ ∆/3)>0and rearranging:
|Ical|∆2−2
3log(1/δ) ∆−2( ¯w−α2) log1
δ
= 0.
This quadratic in ∆has positive leading coefficient |Ical|>0 and negative constant term
−2Alog(1/δ)<0. By Descartes’ rule of signs, it has exactly one positive root:
∆ :=log(1/δ)
3|Ical|+s
log2(1/δ)
9|Ical|2+2( ¯w−α2) log(1/δ)
|Ical|.
For the above ∆, condition (10) holds with equality, so P(ˆα(λ)≤α| D train)≤exp(E(t∗))≤δ ,
satisfying (4).
Step 6: Conclusion.Since (4)holds for every ε >0 withλ=τ(α+ ∆) +ε , we have P(ˆτ≤λ|
Dtrain)≥1−δ for every ε >0 . Taking ε→0+and applying continuity of the probability measure
to the decreasing family of events{ˆτ≤τ(α+ ∆) +ε}, whose intersection is{ˆτ≤τ(α+ ∆)}:
P(ˆτ≤τ(α+ ∆)|D train) = lim
ε→0+P(ˆτ≤τ(α+ ∆) +ε|D train)≥1−δ.
On the event {ˆτ≤τ(α+ ∆)} , monotonicity of ˆfτinτgives ˆfˆτ(x)≤ ˆfτ(α+∆) (x)for all x, hence
{T≥ ˆfτ(α+∆) (X)} ⊆ {T≥ ˆfˆτ(X)} . Since ˆfτ(α+∆) isDtrain-measurable and (Xtest, Ttest)is
independent of(D cal1,Dcal2)givenD train:
Ph
Ttest≥ˆfˆτ(Xtest)Dalli
≥Ph
Ttest≥ˆfτ(α+∆) (Xtest)Dalli
≥1−α−∆,
where the last inequality uses the definition of τ(α+ ∆) and left-continuity of τ7→P(T < ˆfτ(X)|
Dtrain), which give P(T < ˆfτ(α+∆) (X)| D all)≤α+∆ . Notice that P(T < ˆfτ(α+∆) (X)| D all) =
P(T < ˆfτ(α+∆) (X)| D train)since ˆfτ(α+∆) isDtrain-measurable and (Xtest, Ttest)⊥ ⊥ D cal| Dtrain.
This holds with probability at least 1−δ overDcalgivenDtrain, hence over Dall, completing the
proof.
C.2 Budget validity results
In this section, we prove that the expected budget used by Algorithm 5 is bounded by the nominal
budget constraint B. Unless stated otherwise, all expectations and probabilities in this section are
conditional onD train; we omit this explicit conditioning for notational convenience.
C.2.1 Assumptions
We begin by setting the following assumptions.
22

Assumption 1(i.i.d. data).The pairs {(Sk, Pk)}N
k=1areindependent and identically distributed, each
drawn from a joint distribution νonRtmax×[0,1]tmax. We denote by µthe marginal distribution of Sk
onRtmaxand by µtthe marginal distribution of the t-th coordinate Sk(t)onR, fort∈ {1, . . . , t max}.
Assumption 2(Split independence).The data is randomly split by π, which is a random variable,
taking values in the collection of subsets of {1, . . . , N} of size N1, distributed uniformly over all N
N1
such subsets. The random splitπis independent of the full dataset{(S k, Pk)}N
k=1.
Recall that for a given split π, Algorithm 5 splits the calibration data, indexed by {1, . . . , N} , as
follows:
Ical1:=π⊂ {1, . . . , N},|I cal1|=N 1,
Ical2:={1, . . . , N} \ I cal1,|I cal2|=N 2,
Dcal1:={(X i, Hi, Ti)}i∈Ical1 (calibration dataset),
Scal2:=S jfor a fixedj∈ I cal2.
By symmetry of the i.i.d. assumption and the uniform split, the choice of jwithin Ical2does not
affect any expectation below. We re-index the calibration (cal1) samples as i∈ {1, . . . , N 1}for
notational convenience.
Assumption 3(Oracle mapping).For each β≥0 and each t∈ {1, . . . , t max}, there exists a
measurable function
M∗
t(·;β) :R−→[0,1]
that is monotonically increasing in its first argument. For s= (s(1), . . . , s(t max))∈Rtmaxwe write
M∗(s;β) := (M∗
1(s(1);β), . . . , M∗
tmax(s(tmax);β)).
Recall that ¯B2= (B−B 1)/N2is the remaining budget per sample for Phase II, which is a
deterministic function of Dcal1. For a given realization of Dcal1(and hence a given value of ¯B2), the
oracle mappingM∗(·;¯B2)satisfies the following two conditionsD cal1-almost surely:
1.(Pathwise oracle fit)For alli∈ {1, . . . , N 1}:
Pi(t) =M∗
t(Si(t);¯B2)P-almost surely.
2.(Pathwise population budget constraint)
ES∼µ
B(M∗(S;¯B2))
≤¯B2.
Assumption 4(Estimated mapping).Given Dcal1, the estimated mapping Mt(·;D cal1) :R→[0,1]
is measurable and monotonically increasing for each t∈ {1, . . . , t max}. Write M(s;D cal1) :=
(M1(s(1);D cal1), . . . , M tmax(s(tmax);Dcal1)).
Assumption 5(vanishing cross-term moments).For each t∈ {1, . . . , t max}and each non-empty
subsetA⊆ {1, . . . , t}, the following holds forD cal1-almost every realization:
EScal2
Y
k∈Aεk(Scal2(k);D cal1)·Y
k∈{1,...,t}\AM∗
k(Scal2(k))Dcal1
= 0,
whereε k(s;D cal1) =M k(s;D cal1)−M∗
k(s;¯B2).
Conceptually, Assumption 5 requires that these mixed products have zero conditional expectation over
Scal2, given the calibration data Dcal1. Intuitively, this implies that the projection errors {εk}kare
unbiased and mutually uncorrelated across time steps. We emphasize that this condition is not a mere
mathematical artifact of our proof; rather, it is required to guarantee that the projection inaccuracies
do not inflate the expected budget consumed during Phase II.
In practice, there are two cases where this assumption is satisfied. First, it holds exactly in finite
samples if the coordinates of the score vector Scal2are mutually independent, and the estimated
mapping is coordinate-wise unbiased, i.e., EScal2(k)[εk(Scal2(k);D cal1)] = 0 almost surely, for all k.
Alternatively, even without assuming independence or finite-sample unbiasedness, this assumption
holds asymptotically assuming Mt(· | D cal1)is a consistent estimator of M∗
t. Specifically, if
23

EScal2(j)[|Mj(Scal2(j);D cal1)−M∗
j(Scal2(j))| | D cal1]→0 asN1→ ∞ ,Dcal1-almost surely,
then, the magnitude of the cross-terms for any fixed non-empty setAvanishes:
EScal2
Y
j∈Aεj·Y
j /∈AM∗
j(Scal2;¯B2)Dcal1
≤Y
j∈AEScal2(j)[|εj| | D cal1]→0asN 1→ ∞,
where the inequality uses the fact that M∗
j∈[0,1] and Hölder’s inequality. Thus, Assumption 5
holds asymptotically for any fixed Dcal1and target budget ¯B2, meaning the budget constraint can be
satisfied with arbitrary precision given sufficient calibration data in Phase I.
C.2.2 Expected budget validity proofs
To establish the expected budget bound, we first prove the following lemmas.
Independence lemma
Lemma 1(Independence ofD cal1andS cal2).Under Assumptions 1 and 2,
Dcal1⊥S cal2.
Proof. Given the split π,Dcal1 andScal2 correspond to disjoint subsets of the i.i.d. array
{(Sk, Pk)}N
k=1. Disjoint subsets of an i.i.d. sequence are independent of each other. Since πis
independent of the data (Assumption 2), conditioning on πdoes not introduce dependence, and so
Dcal1⊥Scal2.
Properties of the budget functionThe budget functionB: [0,1]tmax→Ris given by:
B(P) =min(T i,ˆfprior(Xi))X
t=1tY
j=1P(j), P= (P(1), . . . , P(t max))∈[0,1]tmax.
Lemma 2(Properties of the budget function).1. Boundedness: 0≤ B(P)≤t maxfor all
P∈[0,1]tmax.
2.Multilinearity: Bis a polynomial in {P(j)}tmax
j=1 that is linear in each coordinate P(j) when
all other coordinates are held fixed.
3.Monotonicity:Bis monotonically increasing in each coordinateP(j).
Proof. (i) Each productQt
j=1P(j)∈[0,1] and there are tmaxterms. (ii) Direct inspection of
the polynomial structure. (iii) For any j0, increasing P(j0)increases every termQt
j=1P(j) with
t≥j 0.
Estimation error does not affect expected loss
Lemma 3(Cross-term cancellation).Under Assumptions 1, 2, 3, 4, and 5, the following holds for
Dcal1-almost every realization of the calibration data:
EScal2
B(M(S cal2;Dcal1))| D cal1
=E Scal2
B(M∗(Scal2;¯B2))| D cal1
.
HereEScal2[· | D cal1]denotes the conditional expectation over Scal2givenDcal1. Since Dcal1⊥Scal2
(Lemma 1), this conditional expectation equals the marginal expectation over Scal2∼µ, withDcal1
treated as a fixed parameter.
Proof.Fix a realization ofD cal1(a.s.). Define for eachj∈ {1, . . . , t max}:
pj:=M∗
j(Scal2(j);¯B2), e j:=ε j(Scal2(j);D cal1) =M j(Scal2(j);D cal1)−M∗
j(Scal2(j);¯B2).
Given Dcal1, both pjandejare bounded, measurable functions of Scal2(j)alone. For each t∈
24

{1, . . . , t max}, expand each factor asM j(Scal2(j);D cal1) =p j+ejand expand the product by the
distributive law:
tY
j=1Mj(Scal2(j);D cal1) =tY
j=1(pj+ej) =X
A⊆{1,...,t}
Y
j∈Aej

Y
j∈{1,...,t}\Apj
.
We prove the above by induction over t. For t= 1 :Q1
j=1(pj+ej) =p 1+e1, which satisfies this
identity. Assume that the identity holds for some arbitrary integerk≥1. Then, fort=k+ 1:
k+1Y
j=1(pj+ej) =
kY
j=1(pj+ej)
(pk+1+ek+1)
By our inductive assumption, we can substitute the expansion for the product up tok:
=
X
A⊆S k
Y
j∈Aej

Y
j∈Sk\Apj

(pk+1+ek+1)
Distributing the termsp k+1ande k+1across the sum gives two separate sums:
=X
A⊆S k
Y
j∈Aej

Y
j∈Sk\Apj
pk+1+X
A⊆S k
Y
j∈Aej

Y
j∈Sk\Apj
ek+1
=X
X⊆S k+1
Y
j∈Xej

Y
j∈Sk+1\Xpj

=k+1Y
j=1(pj+ej).
We now take the conditional expectation givenD cal1.
EScal2
tY
j=1Mj(Scal2(j);D cal1)Dcal1

=X
A⊆{1,...,t}EScal2
Y
j∈Aej·Y
j∈{1,...,t}\ApjDcal1
.(11)
Exchanging sum and expectation is valid since the sum is finite and all terms are bounded
(Lemma 2(i)). Next, we isolate theA=∅term.
(11)=E Scal2
tY
j=1pjDcal1

| {z }
A=∅+X
∅̸=A⊆{1,...,t}EScal2
Y
j∈Aej·Y
j∈{1,...,t}\ApjDcal1
.(12)
Now, for each non-emptyA⊆ {1, . . . , t}, we apply Assumption 5:
EScal2
Y
j∈Aej·Y
j∈{1,...,t}\ApjDcal1
= 0D cal1-a.s.
25

Hence the entire sum over non-emptyAin (12) is zero, and:
EScal2
tY
j=1Mj(Scal2(j);D cal1)Dcal1
=EScal2
tY
j=1M∗
j(Scal2(j);¯B2)Dcal1
.(13)
Summing (13) overt∈ {1, . . . , t max}(Pi(t) =M∗
t(Si;¯B2) = 0 fort > b i) and using linearity of
conditional expectation:
EScal2
B(M(S cal2;Dcal1))| D cal1
=EScal2
B(M∗(Scal2);¯B2)| D cal1
.
Oracle budget is bounded
Lemma 4(Oracle budget bound).Under Assumption 3, forD cal1-almost every realization:
EScal2
B(M∗(Scal2;¯B2))| D cal1
≤¯B2.
Proof. This follows directly from condition (ii) of Assumption 3, which states that
EScal2[B(M∗(Scal2;¯B2))| D cal1]≤¯B2holdsD cal1-almost surely.
Proof of the Holdout Budget Bound
Proposition 2(Holdout sample budget bound).Under Assumptions 1, 2, 3, 4, and 5, for Dcal1-almost
every realization:
EScal2
B(M(S cal2;Dcal1))| D cal1
≤¯B2.
Proof of Proposition 2.By Lemma 3 (cross-term cancellation), forD cal1-almost every realization:
EScal2
B(M(S cal2;Dcal1))| D cal1
=E Scal2
B(M∗(Scal2;¯B2))| D cal1
.
By Lemma 4 (oracle budget bound), forD cal1-almost every realization:
EScal2
B(M∗(Scal2;¯B2))| D cal1
≤¯B2.
Chaining these two relations gives the result.
Proof of the Algorithm’s Budget Validity
Theorem 5(Algorithm’s budget bound).Under Assumptions 1, 2, 3, 4, and 5, the expected budget
used by Algorithm 5 is bounded byB.
Proof of Theorem 5. Condition on Dcal1. Given Dcal1, the Phase I budget B1=P
i∈Ical1biis a
fixed constant, and ¯B2= (B−B 1)/N2is a fixed constant as well. The total realized budget is
B1+B 2, whereB 2=P
i∈Ical2Uiis random over Phase II draws, withU i=Pbi
t=1Qt
j=1ξi(j).
By the tower property and the mutual independence of{U i}i∈Ical2conditioned onD cal1:
E[B2| Dcal1] =X
i∈Ical2E[Ui| Dcal1].
For eachi∈ I cal2, by the tower property andD cal1⊥ ⊥S i(Lemma 1):
E[Ui| Dcal1] =E Sih
E[Ui|Si, Ti,Dcal1]Dcal1i
=EScal2[B(M(S cal2;Dcal1))],
where the second equality uses E[Ui|Si, Ti,Dcal1] =B(M(S i;Dcal1))(the expected stopping time
equals the budget function evaluated at the continuation probabilities, given the Bernoulli draws are
independent).
By Proposition 2, forD cal1-almost every realization:
EScal2[B(M(S cal2;Dcal1))| D cal1]≤¯B2.
ThereforeE[B 2| Dcal1]≤N 2¯B2, and:
E[B1+B 2| Dcal1]≤B 1+N 2¯B2
=B 1+N 2·B−B 1
N2
=B.
Taking the expectation overD cal1givesE[B 1+B 2]≤B.
26

C.2.3 Finite-sample budget guarantee
In this section, we provide a finite-sample bound for the budget consumed by Algorithm 5. Throughout
this section, all probabilities and expectations are conditioned on Dtrain. For notational convenience,
we omit this conditioning.
Theorem 6.Under Assumptions 1, 2, 4, and 5, with probability at least 1−δ , the average budget
per sample consumed by Algorithm 5 is bounded by:
1
NX
i∈Ical˜Ti≤B
N+tmaxlog(1/δ)
3N+1
Ns
tmax2log2(1/δ)
9+ 2N 2tmax¯B2log(1/δ).
Proof. We denote B1=P
i∈Ical1bi, where bi= min(T i,ˆfprior(Xi)), and Ui=Pbi
t=1Qt
j=1ξi(j)
for anyi∈ I cal2, so thatB 2=P
i∈Ical2Ui.
For a fixed Dcal1, and fixed (Si, Ti), the continuation probabilities Mj(Si(j);D cal1)are deterministic.
The Bernoulli draws{ξ i(j)}jare then mutually independent, so:
E[U i| Dcal1, Si, Ti] =biX
t=1tY
j=1Mj(Si(j);D cal1) =B(M(S i;Dcal1)).
Taking a further expectation over(S i, Ti)and usingD cal1⊥ ⊥S i(Lemma 1):
E[U i| Dcal1] =E Scal2[B(M(S cal2;Dcal1))| D cal1].
By Proposition 2, forD cal1-almost every realization:
E[U i| Dcal1] =E Scal2[B(M(S cal2;Dcal1))Dcal1]≤¯B2.
Hence:
E[B 2| Dcal1] =X
i∈Ical2E[U i| Dcal1]≤N 2¯B2.
Notice that the collection {Ui}i∈Ical2is mutually independent given Dcal1. Since Ui∈[0, t max]
almost surely, by Lemma 2, andE[U i| Dcal1]≤¯B2, the Bhatia–Davis inequality gives:
Var[U i]≤(t max−E[U i| Dcal1])·E[U i| Dcal1]≤t max¯B2.
Observe that:
E"
Ui−1
N2E[B2| Dcal1]2Dcal1#
=E[(U i−E[U i| Dcal1])2| Dcal1] = Var(U i| Dcal1).
We now apply the Bernstein inequality to the sum B2=P
i∈Ical2Ui. Notice that {Ui}i∈Ical2are
mutually independent variables givenD cal1, where eachU i∈[0, t max].
P(B 2−E[B 2| Dcal1]≥ε| D cal1)≤exp 
−ε2/2P
i∈Ical2Var(U i| Dcal1) +t maxε/3!
≤exp
−ε2/2
N2tmax¯B2+tmaxε/3
.
We now show that by setting theεas given below, the probability above is bounded byδ:
ε=tmaxlog(1/δ)
3+s
tmax2log2(1/δ)
9+ 2N 2tmax¯B2log(1/δ)
For ease of notation, we denoteV=P
i∈Ical2tmax¯B2.
ε−tmaxlog(1/δ)
3=s
tmax2log2(1/δ)
9+ 2Vlog(1/δ)
27

Squaring both sides yields:

ε−tmaxlog(1/δ)
32
=tmax2log2(1/δ)
9+ 2Vlog(1/δ)
ε2−2tmaxlog(1/δ)
3ε+tmax2log2(1/δ)
9=tmax2log2(1/δ)
9+ 2Vlog(1/δ)
Notice that thetmax2log2(1/δ)
9term appears on both sides and cancels out:
ε2−2tmaxlog(1/δ)
3ε= 2Vlog(1/δ)
Divide the entire equation by 2:
ε2
2−tmaxε
3log(1/δ) =Vlog(1/δ)
Move theεterm to the right side:
ε2
2=Vlog(1/δ) +tmaxε
3log(1/δ)
ε2
2=
V+tmaxε
3
log(1/δ)
The exponent in the original inequality bound is:
−ε2/2
V+t maxε/3
Substituting the equivalent expression we just found forε2
2:
− 
V+tmaxε
3
log(1/δ)
V+t maxε/3=−log(1/δ)
Therefore,
exp(−log(1/δ)) = exp(log(δ)) =δ
Putting it all together into a single block to match the original formulation:
P(B 2−E[B 2| Dcal1]≥ε| D cal1)≤exp 
−ε2/2P
i∈Ical2tmax¯B2+tmaxε/3!
≤exp 
−ε2/2P
i∈Ical2tmax¯B2+tmaxε/3!
= exp 
− P
i∈Ical2tmax¯B2+tmaxε
3
log(1/δ)P
i∈Ical2tmax¯B2+tmaxε/3!
= exp (−log(1/δ))
=δ.
Therefore, with probability at least1−δ, the budget used in Phase II is bounded by:
B2≤N 2¯B2+tmaxlog(1/δ)
3+s
tmax2log2(1/δ)
9+ 2N 2tmax¯B2log(1/δ).
By the construction of ¯B2of Algorithm 5, we have:
B1+N 2¯B2≤B.
Therefore, the average budget per sample used by Algorithm 5 is bounded by:
1
NX
i∈Ical˜Ti≤B
N+tmaxlog(1/δ)
3N+1
Ns
tmax2log2(1/δ)
9+ 2N 2tmax¯B2log(1/δ).
28

C.2.4 Budget Guarantee with Bounded Estimation Errors
We develop a bound for the utilized budget, replacing Assumption 5 with a weaker condition that
bounds the mean absolute estimation error at each step instead of requiring that the estimation errors
cancel out. Specifically, we rely on the following assumption:
Assumption 6(Bounded Mean Absolute Estimation Error).For each j∈ {1, . . . , t max}, there exists
a constantη j≥0such that, almost surely overD cal1:
EScal2(j)[|εj(Scal2(j);D cal1)| | D cal1]≤η j,
whereε j(s;D cal1) :=M j(s;D cal1)−M∗
j(s;¯B2)is the pointwise estimation error at stepj.
Observe that Assumption 5 requires the cross-term expectations E[Q
j∈Aεj(s;D cal1)·Q
j /∈AM∗
j(s;¯B2)| D cal1] = 0 for every non-empty A, which implies ηj= 0 in case where
Scal2has independent coordinates. In contrast, Assumption 6 is strictly weaker: it requires only
that the marginal mean absolute error at each coordinate is bounded, without making assumptions
on the joint cross-term. In particular, it does not require that the score coordinates are indepen-
dent. Furthermore, this assumption is reasonable for consistent estimators Mj(·;D cal1)that satisfy
EScal2(j)[|εj(Scal2(j);D cal1)| | D cal1]→0asN 1→ ∞, and thusη j→0.
Theorem 7(Expected Budget with Estimation Error).Under Assumptions 1, 2, 3, 4 and 6, the
expected total budget consumed by Algorithm 5 satisfies:
EDcal"X
i∈Ical˜Ti#
≤B+N 2Γ,
whereΓ =Ptmax
k=1(tmax−k+ 1)η k.
To prove it, we first develop an upper bound on the expected budget used in Phase II:
Lemma 5.Under Assumptions 1, 2, 3, 4, forD cal1-almost every realization:
EScal2[B(M(S cal2;Dcal1))| D cal1]≤E Scal2
B(M∗(Scal2;¯B2))
+ Γ,
where the total bias is:
Γ :=tmaxX
k=1(tmax−k+ 1)η k.
Proof. We first show a telescoping product identity. For any two sequences of real numbers {aj}t
j=1
and{b j}t
j=1, the following identity holds:
tY
j=1aj−tY
j=1bj=tX
k=1
k−1Y
j=1aj
(ak−bk)
tY
j=k+1bj
,
where empty products equal 1. We verify by induction. For t= 1 :a1−b1= (a 1−b1), which is
trivially true. Assuming it holds fort−1:
tY
j=1aj−tY
j=1bj=att−1Y
j=1aj−btt−1Y
j=1bj
= (a t−bt)t−1Y
j=1aj+bt
t−1Y
j=1aj−t−1Y
j=1bj

= (a t−bt)t−1Y
j=1aj+btt−1X
k=1
k−1Y
j=1aj
(ak−bk)
t−1Y
j=k+1bj

=tX
k=1
k−1Y
j=1aj
(ak−bk)
tY
j=k+1bj
,
29

completing the induction.
We now apply this identity to the continuation probabilities. Set aj=M j(Scal2(j);D cal1)and
bj=M∗
j(Scal2(j);¯B2), soaj−bj=ε j(Scal2(j);D cal1)andaj, bj∈[0,1] . Then for each
t∈ {1, . . . , t max}:
tY
j=1Mj(Scal2(j);D cal1)−tY
j=1M∗
j(Scal2(j);¯B2)
=tX
k=1
k−1Y
j=1Mj(Scal2(j);D cal1)

| {z }
=:W−
k∈[0,1]·εk(Scal2(k);D cal1)·
tY
j=k+1M∗
j(Scal2(j);¯B2)

| {z }
=:W+
k∈[0,1].
Since W−
k, W+
k∈[0,1] (being products of values in [0,1] ), we have |W−
k·εk·W+
k| ≤ |ε k|pointwise.
Therefore, taking the conditional expectation givenD cal1:
EScal2
tY
j=1Mj−tY
j=1M∗
jDcal1
≤EScal2
tY
j=1Mj−tY
j=1M∗
jDcal1

≤EScal2"tX
k=1W−
k|εk(Scal2(k);D cal1)|W+
kDcal1#
≤tX
k=1EScal2[|εk(Scal2(k);D cal1)| | D cal1],
where the final inequality uses W−
kW+
k≤1 and the fact that |εk(Scal2(k);D cal1)|depends
only on the k-th coordinate S(k) , so that EScal2[W−
k|εk(Scal2(k);D cal1)|W+
k| D cal1]≤
EScal2[|εk(Scal2(k);D cal1)| | D cal1] =E Scal2(k)[|εk(Scal2(k);D cal1)| | D cal1]regardless of the
joint distribution of coordinates.
Applying Assumption 6:E Scal2(k)[|εk(Scal2(k);D cal1)| | D cal1]≤η k, giving:
EScal2
tY
j=1Mj(Scal2;Dcal1)−tY
j=1M∗
j(Scal2;¯B2)Dcal1
≤tX
k=1ηk.
We now sum overt:
EScal2[B(M(S cal2;Dcal1))| D cal1]−E Scal2[B(M∗(Scal2;¯B2))]
=tmaxX
t=1EScal2
tY
j=1Mj(Scal2(j);D cal1)−tY
j=1M∗
j(Scal2(j)¯B2)Dcal1

≤tmaxX
t=1tX
k=1ηk
=tmaxX
k=1ηk· |{t:t≥k, t≤t max}|
=tmaxX
k=1ηk(tmax−k+ 1) = Γ,
where we interchanged the order of summation:Ptmax
t=1Pt
k=1=Ptmax
k=1Ptmax
t=k, andPtmax
t=k1 =
tmax−k+ 1.
Armed with the above Lemma, we turn to prove Theorem 7.
30

Proof of Theorem 7. The Phase I budget B1=P
i∈Ical1bisatisfies E[B1] =N 1E[b1], and by
construction ¯B2= (B−B 1)/N2soB 1+N 2¯B2=Bdeterministically. For the Phase II budget:
E[B2| Dcal1] =N 2EScal2[B(M(S cal2;Dcal1))| D cal1].
By Lemma 5 and Lemma 4:
EScal2[B(M(S cal2;Dcal1))| D cal1]≤E Scal2[B(M∗(Scal2;¯B2))] + Γ≤ ¯B2+ Γ.
Therefore:
E[B2| Dcal1]≤N 2(¯B2+ Γ) =B−B 1+N 2Γ.
Taking the total expectation and addingE[B 1]:
E[B1+B 2]≤B+N 2Γ.
Lastly, we develop a finite-sample budget bound based on the above result.
Theorem 8.Under the conditions of Theorem 7, with probability at least1−δ:
1
NX
i∈Ical˜Ti≤B
N+N2
NΓ+tmaxlog(1/δ)
3N+1
Ns
tmax2log2(1/δ)
9+ 2N 2tmax(¯B2+ Γ) log(1/δ).
Proof.Identical to Theorem 6 except that the bound overE[B 2| Dcal1]is replaced with:
E[B2| Dcal1]≤N 2¯B2+N 2Γ.
The above inequality follows from Lemma 5 and Lemma 4.
While Theorem 8 establishes the asymptotic consistency of DAPRO , the finite-sample relative budget
gap of O(tmax2/(¯B2√N1))is highly conservative. This looseness stems from the absolute value
inequality applied in Lemma 5. By taking the absolute value of the errors at each time step, the proof
assumes a worst-case scenario where the estimation inaccuracies at every time step t∈ {1, . . . , t max}
aggregate in the same direction, with no cancellation.
In practice, if the projection models {Mt}tmax
t=1 are reasonably well-calibrated, e.g., yielding approx-
imately mean-zero errors conditionally, the pointwise errors tend to cancel each other out as the
sequence progresses. This explains why our empirical evaluations in Section 4 show that DAPRO
satisfies the target budget Beven when tmax= 200 andN1= 100 is relatively small. Therefore,
rather than viewing this bound as a strict operational limitation, it should be interpreted as a sample
complexity guideline, and demonstrates that the Phase I calibration size N1must grow quadratically
with the maximum conversation horizon tmaxto satisfy the budget constraint at this worst-case setup.
D Algorithms
D.1 LPB calibration with a known censoring mechanism
In this section, we outline in Algorithm 1 the procedure introduced in [ 12] for constructing a calibrated
LPB with a known censoring mechanism.
D.2 Optimized budget allocation
In this section, we present in Algorithm 2 the budget allocation approach developed in [12].
D.3 Proposed greedy adaptive budget allocation
We introduce a Greedy budget allocation strategy as an alternative to our main proposal. Like our
primary method, this strategy dynamically updates censoring times in response to new exchanges in
the conversation. The core motivation behind this approach is to increase the effective sample size,
that is, the number of samples actively contributing to the miscoverage estimator in (3), by observing
more unsafe events early on. We do so by dictating a portion of the budget to exploring conversations
31

Algorithm 1Constructing a calibrated LPB with a known censoring mechanism
Require: Calibration data {Xi, Ci,˜Ti}i∈Ical, censoring weights {wτ(i)}i∈Ical,τ∈T , quantile esti-
mates{ ˆfτ(·)}τ∈T, target miscoverage rateα, prior quantileτ prior, search spaceT ⊆(0,1).
1:forτ∈ T ∩[0, τ prior]do
2:ˆα(τ)←1
|Ical|X
i∈Icalwτ(i)I˜Ti<ˆfτ(Xi)≤C i	
▷miscoverage est.
3:end for
4:ˆτ←supn
τ∈ T ∩[0, τ prior] : supτ′∈T
τ′≤τˆα(τ′)≤αo
▷calibrated quantile level
5:returnLower predictive bound for a test pointX test=x, ˆL(x) = ˆfˆτ(x).
Algorithm 2Optimized static budget allocation
Require: Calibration data {Xi}i∈Ical, attacker model A, target model G, audit function J(·) , pre-
trained quantile regression model {ˆqτ(·)}τ∈T, target miscoverage rate α, prior quantile τprior,
quantile trimming thresholdM, total budgetB.
1:{ˆfτ(Xi)}τ∈T,i∈I cal← {min (ˆq τ(Xi), M)} τ∈T,i∈I cal ▷trim the quantile est.
2:Compute the optimalλ∗=
1
B·P
i∈Icalq
ˆfprior(Xi)2
3:Obtain optimal static probabilitiesπ∗
i= min
1,1/q
λ∗ˆfprior(Xi)
4:C i←(
ˆfprior(Xi)w.p.π i,
0otherwise.
5:For each i∈ I cal, run the conversation between AandGforCiiterations using the initial prompt
Xito obtain the censored event time ˜T←min(C i, Ti).
6:w τ(i)←1
πi, i∈ I cal
7:returnCensoring and event times{( ˜Ti, Ci, wτ(i))} i∈Ical.
that have a high estimated probability of yielding unsafe outputs. Specifically, we partition the total
available budget Binto two stages using a ratio ρ∈(0,1) , where a budget of B·ρ is reserved for
greedy exploration. The greedy exploration is related to the multi-armed bandit framework [ 60].
While any bandit algorithm [ 61,62] could be employed for this exploration phase, we adopt a greedy
strategy that selects the samples with the highest estimated probability of encountering an unsafe
event.
Phase 1: Stochastic Greedy ExplorationLet ξgreedy
i denote the number of acquisitions we have
conducted for the i-th sample, initialized to ξgreedy
i = 0for all i∈ I cal. In this first phase, we rely on
the outputs of our predictive model, ˆp(Hi(t)), which estimates the probability of an unsafe response
at the next step P 
Yi(t+ 1) = 1|H i(t)
. We iteratively spend our exploration budget by focusing
on the most promising candidates. At each step, we identify the set of active samples Iactive, those for
which we have neither encountered the unsafe event nor reached the prior sequence length ˆfprior(Xi).
We compute the predicted unsafe probabilities for these active samples, select the top khighest
probabilities, and sample an index i∗proportionally to their estimated unsafe output probability, from
this top- kset. We advance the interaction for sample i∗by one step ( ξgreedy
i∗←ξgreedy
i∗+ 1), update its
history, and decrement our exploration budget. We repeat this process until the designated exploration
budget ofB·ρis entirely consumed.
Phase 2: Static Optimization for Theoretical GuaranteesAfter the exploration phase concludes,
we use the remaining budget, B·(1−ρ) , to employ the static optimized approach of [ 12], which
provides a theoretical guarantee. Crucially, we concentrate this remaining budget only on the
unresolved samples. For the unresolved sample i /∈ I active—where the unsafe event was successfully
32

triggered, or the prior was reached during Phase 1— the allocation process is complete. We assign
their final censoring time as Ci=ˆfprior(Xi)and their inverse censoring weight as wτ(i) = 1 . For
the remaining active samples i∈ I active, we calculate the remaining number of generations required
to reach the prior: di=ˆfprior(Xi)−ξgreedy
i . We then apply Algorithm 2 over these active samples,
using dias the new target prior and distributing the remaining budget among them. This optimization
yields inverse censoring probability w′(i)and an additional drawn censoring time C′
ifor each active
sample. Finally, we formalize the final censoring times across all samples as:
Ci=(
ξgreedy
i +C′
i, i∈ I active,
ˆfprior(Xi),otherwise,
with the corresponding inverse censoring weights defined as:
wτ(i) =w′(i), i∈ I active,
1,otherwise.
The censored event times are then computed as ˜Ti= min(T i, Ci). Having obtained the set
{(˜Ti, Ci, wτ(i))} i∈Ical, we construct the final calibrated LPB using Algorithm 1. This greedy
budget allocation is summarized in Algorithm 3. We note that the coverage validity of this method
follows from Theorem 4 with Ical1=I active. It also satisfies the budget constraint since in the first
stage we use at most ρ·B budget units, and the second stage uses at most (1−ρ)·B in expectation,
as guaranteed by [12].
33

Algorithm 3Greedy adaptive budget allocation (Greedy)
Require: Calibration data {Xi}i∈Ical, attacker model A, target model G, audit function J(·) ,
predictive model ˆp(·), prior quantiles {ˆfprior(Xi)}i∈Ical, total budget B, exploration ratio ρ∈
(0,1), top-kparameterk.
1:Initialize acquisitionsξgreedy
i←0, and event timesT i← ∞for alli∈ I cal.
2:B explore←B·ρ ▷allocate exploration budget
3:Phase 1: Stochastic Greedy Exploration
4:whileB explore >0do
5:I active← {i∈ I cal:ξgreedy
i <ˆfprior(Xi)andT i> ξgreedy
i}▷update active set
6:ifI active is emptythen
7:break
8:end if
9:Compute probabilitiesp i←ˆp(H i(ξgreedy
i))for alli∈ I active
10:LetKbe the indices of the topmin(k,|I active|)values in{p i}i∈I active
11:Sample an indexi∗fromKwith probability proportional top i∗
12:Updateξgreedy
i∗←ξgreedy
i∗+ 1
13:Advance interactionA ↔ Gby one step to obtain new exchange, updateH i∗(ξgreedy
i∗)
14:ifAudit(new exchange) == 1then
15:T i∗←ξgreedy
i∗ ▷unsafe event observed
16:end if
17:B explore←B explore−1
18:end while
19:Phase 2: Static Optimization for Theoretical Guarantees
20:I active← {i∈ I cal:ξgreedy
i <ˆfprior(Xi)andT i=∞}
21:B remain←B·(1−ρ)
22:Set remaining target lengthsd i←ˆfprior(Xi)−ξgreedy
i for alli∈ I active
23:{(C′
i, w′(i))} i∈I active←Algorithm 2 applied with priors{d i}i∈I activeand budgetB remain
24:Finalize Censoring Times and Weights
25:foreachi∈ I caldo
26:ifi∈ I active then
27:C i←ξgreedy
i +C′
i
28:w τ(i)←w′(i)
29:else
30:C i←ˆfprior(Xi)▷interaction resolved or prior reached during phase 1
31:w τ(i)←1
32:end if
33: ˜Ti←min(T i, Ci)
34:end for
35:returnCensoring and event times{( ˜Ti, Ci, wτ(i))} i∈Ical.
D.4 Proposed locally optimized adaptive budget allocation
We propose a LocallyAdaptive allocation method as an additional alternative to our main proposal.
While our primary method solves a global optimization problem to map risk scores to continuation
probabilities across all future time steps, this baseline computes the optimal probabilitieslocallyat
each step.
Similarly to our main approach, we partition the calibration data into two disjoint subsets: a small
set indexed byI cal1(e.g.,N 1= 100) on which we find an optimal parameter, and a deployment set
Ical2, on which we deploy this parameter. The algorithm operates in two phases.
Phase I: Tuning the Global Parameter.In the first phase, we first fully observe all samples in
Ical1, until we either reach the prior ˆfprior(Xi)or observe the unsafe event Ti. After this, we have ¯B2
34

average budget remaining for the second phase. In this first phase, we use Ical1to calibrate a single
global parameter, λ∈R+, which balances the tradeoff between the consumed sampling budget and
the probability of discovering unsafe events.
The expected cost is the sum of the survival probabilities up to the known stopping time:
ˆRN1(λ) =1
N1X
i∈Ical1min(T i,ˆfprior(Xi))X
t=1t−1Y
j=0Pi(j;λ),
where Pi(j;λ) is the local continuation probability evaluated at λ. We then conduct a binary search
to find the optimal valueλ∗:= infn
λ≥0 :N1
N1+1ˆRN1(λ) +tmax
N1+1≤¯B2o
.
Phase II: Local Step-by-Step Acquisition.Equipped with the optimal parameter λ∗, we deploy
the policy on the deployment set Ical2. At any given step t, for each active sample i, we use our
predictive model to estimate the probability of an unsafe event occurring at every future time step
t2> t, conditioned on the current conversation history Hi(t). Let ˆp(t+k|t) denote the estimate of
the probability that the unsafe event occurs exactly ksteps in the future, given that the interaction has
safely reached stept, and the history up to this point,P(T i=t+k|T i> t, H i(t)).
Using these future probabilities, we compute the expected remaining cost Ei(t), defined as the
anticipated number of generations required to either observe an unsafe event or reach the prior
sequence length ˆfprior(Xi)without observing an unsafe event. Formally, letd i(t) = ˆfprior(Xi)−t
denote the maximum number of remaining steps. The expected remaining cost is computed as the
weighted sum of future steps:
Ei(t) =di(t)X
k=1k·ˆp(t+k|t) +d i(t)tmax−t+1X
k=di(t)+1ˆp(t+k|t),(14)
where the second term captures the maximum cost di(t)paid if the unsafe event does not occur before
reaching the prior ˆfprior(Xi). We note that ˆp(tmax+ 1|t) estimates the probability that no unsafe
events occur at timesj≤t max.
Guided by the Karush-Kuhn-Tucker (KKT) conditions for budget-constrained variance minimiza-
tion [ 12], the optimal probability of reaching the end of the sequence from the current state is defined
as:
Ptarget
i(t) = min 
1,1p
λ·E i(t)!
.(15)
Because our acquisition process operates step-by-step, we must translate this probability into a local
continuation probability. Let Paccum
i(t) :=Qt−1
j=0Pi(j)denote the cumulative probability that sample
ihas survived up to step t, where Paccum
i(0) := 1 . Then, the immediate continuation probability is
given by:
Pi(t) = min
1,Ptarget
i(t)
Paccum
i(t)
.(16)
Finally, we draw a Bernoulli random variable Zi(t)∼Ber(P i(t)). IfZi(t) = 1 , we expend one unit
of budget to acquire the next conversational exchange Hi(t+ 1) ; otherwise, we halt the interaction
and set the censoring time as Ci=t. Furthermore, to ensure the inverse-censoring weights remain
bounded for samples halted early by the policy, we lower bound the global minimal probability by
pmin(e.g., 0.005 ) on the final cumulative probability. For convenience, we summarize this procedure
in Algorithm 4. The coverage validity of this process follows from Theorem 4. We remark that
the budget validity of this approach can be formally established using the conformal risk control
framework [63], as presented next.
Proposition 3(Budget validity of LocallyAdaptive ).Suppose that {(Xi, Ti, Hi)}i∈Icalare drawn
i.i.d. Then the expected total budget consumed by Algorithm 4 satisfies
E"X
i∈Icalbi#
≤B,
35

where bidenotes the number of attacker–target exchanges conducted for sample i, and the expectation
is taken over the randomness of the data {(Xi, Ti, Hi)}i∈Icaland the random continuation draws of
the policy.
Proof. We begin by analyzing the budget used in Phase I and the remaining budget per sample. For
eachi∈ I cal1, the algorithm runs the conversation to completion (without any stopping policy),
expending bi= min ˆfprior(Xi), Ti
budget units. The total Phase I budget is B1:=P
i∈Ical1bi.
By construction of Algorithm 4, the target average budget per Phase II sample is
¯B2:=B−B 1
N2,
soB 1+N 2¯B2=Bholds deterministically for every realization of the Phase I data.
We now turn to examining the Phase II budget as a function of λ. For each j∈ I cal2and any
λ≥0 , letbj(λ)denote the budget expended on sample junder the policy parameterized by λ, i.e.,
the final value of the acquisition counter at termination. We note two properties. First, bj(λ)is
bounded: since the active set removes sample jonce it reaches ˆfprior(Xj), we have bj(λ)∈[0, t max]
for all λ≥0 . Second, bj(λ)is monotone non-increasing in λ: increasing λdecreases the KKT
target probability Ptarget
j(t) = min 
1,1/p
λ·E j(t)
at every step t, and hence decreases the local
continuation probability Pj(t). Consequently, sample jis more likely to be halted earlier, so bj(λ)is
non-increasing inλfor each fixed sample.
Because Phase I conversations are observed to completion, the full histories {Hi}i∈Ical1are available.
For each i∈ I cal1and any λ≥0 , we can therefore obtain the expected budget that the policy λ
would have expended on sample i; denote this by bexpi(λ) :=Pbi
t=1Qt
k=1Pi(k;λ) , where Pi(k;λ)
is defined as in (16), with the correspondingλ. The empirical average expected budget onI cal1is
ˆRN1(λ) :=1
N1X
i∈Ical1bexp
i(λ).
Following the conformal risk control framework of [63], Phase I tunes
ˆλ:= inf
λ≥0 :N1
N1+ 1ˆRN1(λ) +tmax
N1+ 1≤¯B2
.(17)
This infimum is well-defined: since Ptarget
i(t)→0 asλ→ ∞ , all continuation probabilities vanish
andˆRN1(λ)→0.
By the definition of ˆλand the monotone non-increasing property of bexpi(λ), the selector satisfies
almost surely:
N1
N1+ 1ˆRN1(ˆλ) +tmax
N1+ 1≤¯B2.(18)
Multiplying both sides of (18) byN 1+ 1yields:
X
i∈Ical1bi(ˆλ) +t max≤(N 1+ 1) ¯B2(almost surely).(19)
We now bound the expected budget used in Phase II. Fix any j∈ I cal2. Denote by bempj(ˆλ)the
empirical budget expended for the j-th sample by the policy employed with ˆλ. Since bempj(ˆλ)≤t max,
we may replacet maxon the left-hand side of (19) bybempj(ˆλ)to obtain:
X
i∈Ical1bexp
i(ˆλ) +bemp
j(ˆλ)≤(N 1+ 1) ¯B2(almost surely).(20)
Taking expectations on both sides of (20):
E"X
i∈Ical1bexp
i(ˆλ)| D cal1#
+Eh
bemp
j(ˆλ)| D cal1i
≤(N 1+ 1)E[ ¯B2| Dcal1].
36

SinceEh
bempj(ˆλ)| D cal1i
=Eh
bexpj(ˆλ)| D cal1i
, we set:
E"X
i∈Ical1bexp
i(ˆλ)| D cal1#
+Eh
bexp
j(ˆλ)| D cal1i
≤(N 1+ 1)E[ ¯B2| Dcal1].
The data {(Xi, Hi, Ti)}i∈Ical1∪{j} contains N1+ 1 i.i.d. (hence exchangeable) samples, with ˆλ
determined solely by the Ical1portion. By Theorem 1 of [ 63] applied to the non-increasing, [0, tmax]-
bounded loss bexp(λ), the selector (17), and these N1+ 1exchangeable samples, the expected loss on
the held-out samplejsatisfies:
Eh
bexp
j(ˆλ)| D cal1i
≤E[ ¯B2| Dcal1]for allj∈ I cal2.(21)
We note that while Theorem 1 of [ 63] is formulated for a deterministic risk level, its proof also holds
for a risk level that is a random variable ¯B2, and we can take the expectation to obtain the above
inequality.
We now conclude by summing the budget over all remaining samples in Phase II. Summing (21)
over all j∈ I cal2and applying linearity of expectation (valid since all terms are non-negative and
bounded byt max):
E
X
j∈Ical2bexp
j(ˆλ)| D cal1
=X
j∈Ical2Eh
bexp
j(ˆλ)| D cal1i
≤N 2E[¯B2| Dcal1].
Combining with the Phase I budget and using B1+N 2¯B2=B deterministically, and by taking an
expectation overD cal1, we get:
E"X
i∈Icalbi#
=E"X
i∈Ical1bi#
+E
X
j∈Ical2bexp
j(ˆλ)

≤E[B 1] +N 2E[¯B2]
=E[B 1] +N 2EB−B 1
N2
=E[B 1] +E[B−B 1]
=B,
where the last equality uses the fact that Bis a deterministic constant. This completes the proof.
37

Algorithm 4Locally optimized adaptive budget allocation (LocallyAdaptive)
Require: Calibration data {Xi}i∈Ical, attacker model A, target model G, audit function J(·) ,
predictive model ˆp(· | ·) , prior quantiles {ˆfprior(Xi)}i∈Ical, total budget B, first-split size N1,
minimum probabilityp min.
1:Randomly partitionI calintoI cal1(sizeN 1) andI cal2(sizeN−N 1).
2:LetB 1be the budget required to fully observeI cal1up to ˆfprior(Xi)or unsafe event.
3:¯B2←(B−B 1)/|I cal2|▷target average budget for phase 2
4:Phase 1: Tuning the Globalλparameter
5:Define the expected budget function overI cal1for a givenλ:
6: ˆRN1(λ)←1
N1P
i∈Ical1Pmin(T i,ˆfprior(Xi))
t=1Qt−1
j=0Pi(j;λ)▷usingP ifrom (16)
7:Find optimal λ∗∈[0, λ max]satisfying: λ∗:= infn
λ≥0 :N1
N1+1ˆRN1(λ) +tmax
N1+1≤¯B2o
via binary search.
8:Phase 2: Local Step-by-Step Acquisition
9:Initialize acquisitionsξlocally
i←0, cumulative probabilitiesPaccum
i←1, for alli∈ I cal2.
10:I active← {i∈ I cal2:ξlocally
i <ˆfprior(Xi)}
11:whileI active is not emptydo
12:foreachi∈ I active do
13:Estimate future unsafe probabilitiesˆp(ξlocally
i +k|ξlocally
i)using the predictive model
14:d i←ˆfprior(Xi)−ξlocally
i ▷remaining allowed steps
15:Compute expected remaining costE i(ξlocally
i)using (14)
16:Compute KKT target probabilityPtarget
i using (15)
17:Compute local continuation probabilityP i(ξlocally
i)using (16)
18:SampleZ i∼Ber(P i(ξlocally
i))
19:ifZ i== 1then
20:Paccum
i←Paccum
i·Pi(ξlocally
i)
21:ξlocally
i←ξlocally
i + 1
22:Advance interactionA ↔ G, updateH i(ξlocally
i)
23:ifAudit(new exchange) == 1then
24:T i←ξlocally
i ▷unsafe event observed
25:RemoveifromI active
26:end if
27:else
28:RemoveifromI active ▷halted by policy
29:end if
30:end for
31:UpdateI active← {i∈ I active:ξlocally
i <ˆfprior(Xi)}
32:end while
33:Finalize Censoring Times and Weights
34:foreachi∈ I caldo
35:ifi∈ I cal1then
36:C i←ˆfprior(Xi)andw τ(i)←1
37:else
38:C i←ξlocally
i + 1ifξlocally
i + 1<min(T i,ˆfprior(Xi)), otherwiseC i←ˆfprior(Xi)
39:w τ(i)←1/Paccum
i ▷bound max weight
40:end if
41: ˜Ti←min(T i, Ci)
42:end for
43:returnCensoring and event times{( ˜Ti, Ci, wτ(i))} i∈Ical.
38

D.5 ProposedDAPRO
We presentDAPROin Algorithm 5.
While the overall computational cost is dominated by the BLLM API calls, the internal algorithmic
overhead is lightweight. Specifically, the optimization process requires O(N 1tmax)operations
per iteration. In our experiments, we conduct at most 60outer iterations and 10inner iterations.
Furthermore, training the projection model Mtat each timestep ttakesO(N 1)operations. During
Phase II, deploying the acquisition policy costs O(N 2tmax)operations, and computing the final
calibrated thresholdˆτrequires onlyO(N)steps.
E Experimental details
E.1 Data generation
Across all experiments, we generate the raw multi-turn attacker-target conversations using the PAIR
algorithm [ 9], with initial prompts (attack goal) taken from three popular safety benchmarks: we
sample 10,000 harmful prompts from the RealToxicityPrompts dataset [ 3], 10,000 prompts from the
Anthropic Red Team dataset [ 15,16], 10,000 prompts from SQuAD v2 data [ 64,65], and 10,000
instruction-following tasks from the AutoIF dataset [ 17]. We deploy Qwen 2.5 14B Instruct [ 13,66]
as the attacking model, since it had the fewest formatting errors when generating adversarial prompts
compared to the other models we examined. For the target models, we evaluate safety across three
distinct architectures: Qwen 2.5 14B Instruct, Llama 3.1 8B Instruct [ 18], Phi 4 Mini Instruct [ 19],
and Gemma 3 4B Instruct [20].
We used both HuggingFace [ 67] and vLLM [ 68] to load and infer with these models, using bfloat16
precision for all models. The system prompts and hyperparameters for the LLMs we use follow those
of [9]. Specifically, we set the attacker’s temperature, top-p, and max new tokens to 1, 0.9, and 500,
respectively, and the target’s to 0, 1, and 150, respectively. If the attacker failed to generate an attack
in a valid JSON format, we continue to retry for at most 10 attempts, and finish the conversation
if it failed to generate a valid response for ten attempts. The number of responses saved in the
conversation history of the attack model is set to 4. The parameters of the judge model are 50 max
tokens, 0 temperature, and top-p of 1. If we fail to obtain a rating from the judge, we set the score of
the judge to 0 manually and continue the conversation process. Each conversation runs as a single
stream.
To define a successful jailbreak, i.e., an unsafe event where Y(t) = 1 , we rely on LM safety judges.
For the Toxicity dataset, we employ Detoxify [ 21] and flag any response scoring above a threshold
of 0.5 as unsafe, consistent with the evaluation protocol in [ 12]. For the Red Team dataset, we use
two different judging mechanisms: Qwen 2.5 14B Instruct serving as an LLM-as-a-judge, which
outputs a continuous score from 1 to 10, where a score of 10 triggers the unsafe event flag, and
Llama-Guard [ 23], which directly outputs a binary safety label. In all generated conversations, we
cap the maximum sequence length att max= 200turns.
For the Toxicity dataset, the attacker is asked to construct a prompt using the "prompt" part of
RealToxicityPrompts so that the target model outputs the "continuation" part of the sample.
For the Anthropic Red Team dataset, we first select the 10,000 most harmful examples based on the
min_harmlessness_score_transcript parameter, and use the first human text as the initial attack. If the
attack was successful, i.e., had a rating ≥4, we use the response of the LLM as the target output.
Otherwise, we set the default objective to "Sure, here is a...". Our hallucination dataset builds on
the SQuAD v2 data [ 64,65], which contains subjects with a paragraph describing it. We consider
the paragraphs from SQuAD as the "ground truth" source for the target model and judge, and set
the goal of the attacker to trick the target model into answering a question about the subject that is
NOT covered by its provided text. For this data, we engineered two attacker prompts based on [ 9]:
one utilizing ’leading questions’ to trick the target into referencing outside knowledge, and another
employing ’gaslighting’ to compel the target into generating false information not present in the
reference text.
Finally, for the AutoIF dataset, we extract tasks with existing programmatic verifiers. To increase the
complexity and evaluate multi-constraint interference, we programmatically inject between three and
six additional structural constraints into each prompt: explicit word counts, forbidden vocabulary,
39

Algorithm 5Dynamic Allocation via PRojected Optimization (DAPRO)
Require: Calibration prompts Dcal={X i}N
i=1, target lengths {ˆfprior(Xi)}N
i=1, horizon tmax, total
budgetB, score functions{S t}tmax
t=1.
1:Randomly partition dataset Dcal1into disjoint sets, indexed by: Ical1(sizeN1) andIcal2(size
N2).
2:Phase I: Learning the Optimal Acquisition Policy
3:foreachi∈ I cal1do
4: Run conversation until unsafe event or prior is reached; let bi= min(T i,ˆfprior(Xi))denote
the observed length.
5:fort= 1, . . . , b ido
6:Compute intermediate risk score:S i(t) =S t(Hi(t))
7:end for
8:end for
9:Compute ¯B2←1
|Ical2|(B−P
i∈Ical1bi)
10:Solve for optimal probability matrixP∈[0,1]N1×tmax:
min
P1
N1X
i∈Ical11
Qbi
t=1Pi(t)s.t.1
N1X
i∈Ical1B(Pi)≤¯B2,andP i(t)monotonic w.r.tS i(t)
11:fort= 1, . . . , t maxdo
12: Fit a projection model Mtover pairs {Si(t), P i(t)}i∈Ical1. If{Si(t), P i(t)}i∈Ical1=∅, we
manually setM t(s) = 1for alls.
13:end for
14:Phase II: Adaptive Acquisition
15:foreachi∈ I cal2do
16:Initialize continuation indicatorξ i(0) = 1andξ i(t) = 0∀t≥1, and step countert= 1
17:whileξ i(t−1) == 1andt≤ ˆfprior(Xi)do
18:Compute current risk score:S i(t) =S t(Hi(t))
19:Infer optimal continuation probability:P i(t) =M t(Si(t))
20:Sample Bernoulli random variableZi
t∼Ber(P i(t))
21:ifZi
t== 1then
22:setξ i(t) = 1.
23:Advance attacker-target interaction and obtainH i(t+ 1).
24:IfY i(t) = 1, halt early, and break.
25:else
26:ξ i(t) = 0▷Halt interaction
27:Break
28:end if
29:t=t+ 1
30:end while
31:Find the last successful step:tstop
i= max{t∈ {0, . . . , t max}:ξ i(t) == 1}
32:end for
33:foreachi∈ I caldo
34:ifi∈ I cal1then
35:C i←ˆfprior(Xi)andw τ(i)←1
36:else
37:C i←tstop
iiftstop
i<min(T i,ˆfprior(Xi)), otherwiseC i←ˆfprior(Xi)
38:w τ(i)←1/Qtstop
i
t=1Pi(t)▷bound max weight
39:end if
40:Set ˜Ti←min(T i, Ci)
41:end for
42:returncensoring and event times{ ˜Ti, Ci, wτ(i)}i∈Ical.
40

mandatory markdown elements, or specific paragraph quantities. These new rules are appended to
the prompt alongside their corresponding Python verification functions. Notably, for this dataset,
the attacking model is a helper agent. Its goal is not to elicit harmful content, but rather to improve
the presentation of the complex task, e.g., by generating an explicit output contract or checklist, to
maximize the probability that the target model passes all programmatic constraints, explicitly without
solving the task itself. Specifically, an event is defined as a response that successfully satisfies all
prompt requirements, which we verify programmatically using deterministic Python scripts.
The resulting dataset of interactions is divided into three disjoint subsets: a training set (4,000
prompts), a calibration set (3,000 prompts), and a test set (3,000 prompts). We fix the training data
and use it to fit our predictive model, assuming an unlimited budget during the training phase, where
the full conversation histories are fully observed.
E.2 Predictive model and hyperparameters
First, we embed all texts into vectors using Multilingual-E5-large [ 69]. We use these embeddings as
the inputs for our predictive model, which estimates the raw risk scoresS t(H(t)).
The model architecture begins by projecting the high-dimensional embeddings into a lower-
dimensional latent space of 256 using a linear layer, followed by a ReLU activation and dropout of
rate 0.2. To capture the temporal dependencies of the conversational history, we apply a Transformer
Encoder [ 70]. Specifically, we use 3 pre-normalized Transformer layers with 6 attention heads, a
feed-forward dimension of 1024, and GELU activations. We apply an upper-triangular attention
mask during encoding to prevent the model from using future time steps. Following the sequence
encoding, we concatenate the hidden states of the Transformer with the initial projected embeddings.
This concatenated representation is passed through a multi-layer perceptron consisting of three linear
layers, separated by ReLU activations and dropout of rate 0.2, to produce the raw temporal logits.
Specifically, the model receives the text embeddings of the history and estimates the probability of an
unsafe event occurring at any future time step t2> t1. The output is an tmax×(t max+ 1) matrix
containing the probabilities P(T=t 2|T > t 1, H(t 1)), and the probability of the conversation
safely reaching the maximum horizon without an unsafe event,P(T > t max|T > t 1, H(t 1)).
We train this model using an AdamW optimizer [ 71] with a learning rate of 10−4, weight decay of
0.001, and a batch size of 64. We optimize the network for a maximum of 500 epochs to minimize
the survival negative log-likelihood loss. This loss function explicitly accounts for right-censored
data by evaluating the exact event probability for uncensored sequences and the aggregated survival
probability for censored sequences. We stop the training early if the validation loss does not improve
for 50 consecutive epochs. Subsequently, we calibrate the outputs of the model using a temperature
scaling L-BFGS optimizer to tune a time-dependent temperature parameter on a held-out validation
set, minimizing the discrete survival NLL. The neural networks are implemented using the PyTorch
library [ 72]. The validation set is a portion of 10% samples from the training set that was held-out so
that the model is not trained over it.
E.3 Calibration algorithms and evaluation protocol
Once the predictive model is trained and frozen, we evaluate our proposed DAPRO against the
optimized method introduced in [ 12] and two other dynamic techniques we develop in this work. The
experiments are taken over 50 independent random splits of the remaining 6,000 prompts into the
calibration and test sets.
When constructing an Upper Predictive Bound (UPB), we define the coverage event as the condition
where the UPB is either greater than or equal to the true time-to-event T, or when it reaches the
maximum conversational horizon tmax= 200 . In this formulation, tmaxserves as a practical infinite
time horizon. This design choice is necessary as treating tmaxas an effective infinity preserves
the theoretical coverage semantics while keeping the UPB size finite. This allows us to effectively
compare the bound sizes and their tightness across different calibration methods.
For the calibration algorithms, we simulate the budget-constrained environment by setting the average
allowed budget per sample to B/|I cal|= 20 . Following the static allocation setup in [ 12], we set our
prior quantile level toτ prior= 0.56when constructing LPBs. When constructing a UPB, we set the
prior level at τprior= 0.97 . To control the maximum inverse-censoring weight, we set the clipping
41

parameter to γ= 100 , which correspondingly yields an upper bound of M= 200 for the trimmed
quantile estimates ˆfτ(x). For the LPB construction, across all calibration methods, we define the
search space Tas a logarithmically spaced grid of 1,000 candidate values in the range [0.001,0.977] ,
following [ 12]. For the UPB construction, we set the search space as a linearly spaced grid of 3000
values in the range [0.5,0.95] . For each τ∈ T , we compute the estimated miscoverage rate ˆα(τ)
using (3)and select the calibrated ˆτthat yields an estimated miscoverage closest to the target level α.
During Phase I of DAPRO , we implement the projection models {Mt}tmax
t=1 as Platt scaling regressors
using the scikit-learn package [ 73]. We implement the budget-constrained Phase I optimization,
with a bisection search over the Lagrange multiplier λ∈[10−8,1014]running up to 60 outer steps,
combined with an inner Gauss-Seidel Block Coordinate Descent over the log-probabilities, running up
to 10 inner passes, with tolerance 10−9. For numerical stability, all probability variables are optimized
in log-space. We initialize the variables using a geometric series approximation, Pinit=¯B2/(1 + ¯B2)
and warm-start the inner solver using the state from the λboundary that yielded a budget closest to
the target average per sample ¯B2. We enforce the monotonicity of the probabilities with respect to the
risk scores by applying the Pool Adjacent Violators Algorithm (PA V A) only when the monotonicity
is violated during coordinate updates. Once the algorithm converges, we apply a bisection that shifts
the log-probabilities to guarantee the budget constraint is satisfied over this set of samples.
In our implementation of the LocallyAdaptive method, we compute the global parameter λwithout
the finite-sample correction term. Specifically, we set: λ:= inf{λ≥0 : ˆRN1(λ)≤ ¯B2}, where
ˆRN1(λ)denotes the empirical expected budget consumed on the first data split Ical1, and ¯B2is the
target average budget for the second phase. While the correction term is required for the rigorous
theoretical guarantees to control the budget, we observed empirically that this uncorrected formulation
consistently satisfies the nominal budget constraint in practice, making the additional conservative
penalty unnecessary for our experimental setting.
E.4 Setup of the experiment in Section 1
In the experiment from Section 1, we employed our proposed method, DAPRO , and the static optimized
budget allocation introduced in [ 12]. We chose two specific samples to illustrate how the distribution
of the censoring time changes over time. For the static method, the censoring time is defined as
Ci=ˆfprior(Xi)·Ber(π i), where Ber(π) is a Bernoulli random variable with probability πi. Hence,
the expected censoring time is given by ˆfprior(Xi)·πi. This expected censoring time is static, as
it is computed once and does not change over time. In contrast, our dynamic method changes the
advancement probabilityP i(t) =P(ξ i(t) = 1).
To evaluate the expected censoring time at step twithout relying on the future probabilities, e.g.,
P(ξi(t+ 1) = 1), we define the expected censoring time as:
t+P(ξ i(t) = 1)·( ˆfprior(Xi)−t)
This formulation is a conservative, causal estimate. It assumes that if the process successfully advances
past the current step t, it will continue to run until it reaches ˆfprior(Xi). Since we intentionally
avoid relying on the unobserved future probabilities, this metric reflects the real-time, sequential
decision-making process of the algorithm. Finally, the reported coverage differences are taken across
50 independent random splits of the calibration and test sets.
E.5 Machine specifications
The computational infrastructure used to generate the datasets and run the experiments includes:
•CPU: Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz, Intel(R) Xeon(R) Gold 5318Y CPU
@ 2.10GHz, Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz.
•GPU: NVIDIA A40, NVIDIA TITAN X (Pascal), NVIDIA 2080 TI, NVIDIA RTX 2060
SUPER.
•OS: Ubuntu 20.04.6.
The data generation via the PAIR algorithm required approximately four days over ten GPUs for
sequences of length tmax= 200 per dataset and one attacker-target-judge configuration. The GPUs
42

used for this data generation have 48GB of memory. Training the predictive model and calibrating
it using either calibration method takes several minutes per configuration. Across all experiments,
random seeds were fixed for reproducibility.
F Additional experiments
In this section, we present extended experiments and performance metrics to further evaluate our
proposed framework. We compare against additional baselines and examine the effect of the score
function and projection method. We evaluate two distinct score functions for quantifying prompt
risk: (1) the estimated probability to observe an unsafe response at the current iteration, and (2) the
estimated 1−α quantile of the time-to-unsafe given that an unsafe response was not observed up to
the current time. Furthermore, we analyze the effect of the score-to-probability mapping mechanism
by comparing Platt scaling and Beta-distribution scaling. Specifically, we evaluate the following
eight methods:
•Optimized:The static, variance-minimizing allocation strategy [12].
•Greedy (10%) & Greedy (95%):The idea is to dynamically observe as many unsafe
events as possible on a portion of the data, and then employ the static approach of [ 12] to
obtain a coverage guarantee. See Appendix D.3 for additional details on this method.
•LocallyAdaptive :A locally optimized version of our main proposal, where the continua-
tion probabilities are optimized for each step separately, and not globally, in contrast to our
flagship method. See Appendix D.4 for more information about this method.
•DAPRO (Score 1, Platt) & DAPRO (Score 2, Platt):Our optimally adaptive method with two
different score functions (estimated probability of unsafe event and estimated quantile of
unsafe event time), with Platt scaling mapping of scores to continuation probabilities .
•DAPRO (Score 1, Beta) & DAPRO (Score 2, Beta):Our optimally adaptive method with the
same two score functions, but mapped via Beta distribution scaling.
For each experimental setup, we evaluate the performance of these methods across all target LLMs:
Qwen 2.5 14B Instruct, Llama 3.1 8B Instruct, Phi 4 Mini Instruct, and Gemma 3 4B Instruct. We
report the following metrics: (1) empirical coverage rate at a 90% target level, (2) the average LPB
size, (3) the coverage deviation from the 90% nominal target, (4) the total number of unsafe events
observed (5) the empirical mean inverse-censoring weight, (6) the variance of the empirical coverage
rate, and (7) the variance of the LPB size constructed by each calibration algorithm.
F.1 Toxicity dataset
Figures 8 and 9 present the performance of all methods on the Toxicity dataset using Detoxify as the
safety judge. As theoretically guaranteed, all methods achieve a valid coverage rate and satisfy the
budget constraint across all evaluated target LLMs. However, the static optimized baseline exhibits
a high coverage deviation and variance. In contrast, our adaptive DAPRO observes significantly
more unsafe responses than the competitors, resulting in the lowest coverage deviation and variance.
LocallyAdaptive also exhibits low variance compared to the static baseline, although it does not
match the performance of DAPRO , since its optimization is conducted locally rather than globally.
Interestingly, DAPRO achieves this low variance despite having a higher mean inverse-censoring
weight. This counterintuitive increase in mean weight occurs since DAPRO invests budget in the initial
policy-learning phase ( Ical1), especially for resilient models like Llama and Phi 4 Mini that require
longer interactions to jailbreak, leaving a smaller remaining budget for the deployment phase ( Ical2).
As indicated by the results over the Qwen2.5 and Gemma3 LLM models, when the budget remaining
for the second phase increases,DAPROachieves a lower mean weight than the static baseline.
F.2 RedTeam dataset with Qwen as a judge
In Figures 10 and 11, we present the metrics over the Anthropic Red Team dataset, utilizing Qwen
2.5 14B Instruct as an LLM-as-a-judge.
These figures closely mirror those observed in the Toxicity experiments, demonstrating the robustness
of our method across different safety benchmarks and scoring mechanisms. Once again, all evaluated
43

Figure 8:Toxicitydataset: coverage rate, LPB size, coverage deviation, and budget utilized by
various methods across four target LLMs. Target coverage rate: 90% and target ¯B= 20 budget per
sample. Performance metrics are taken over 50 random splits of the calibration and test sets.
methods successfully attain valid coverage and satisfy the budget constraint. However, the static
optimized baseline still suffers from a high coverage deviation and variance. In contrast, DAPRO
dynamically adapts to the interactions to uncover the highest number of unsafe events, yielding
the LPBs with the lowest variance and coverage deviation. Furthermore, we observe the exact
phenomenon with the mean-weight: since DAPRO expends many budget units during the initial policy-
learning phase ( Ical1) to learn an optimal policy, it occasionally has a higher mean inverse-censoring
weight due to the limited budget remaining for Ical2. These figures also show that the Greedy method
is unreliable since it tends to exhibit a very high variance when a high portion of the budget is used
for the early greedy exploration.
44

Figure 9:Toxicitydataset: number of observed unsafe events, mean inverse-probability weight,
coverage variance, and LPB variance by various methods across four target LLMs. Target coverage
rate: 90% and target ¯B= 20 budget per sample. Performance metrics are taken over 50 random
splits of the calibration and test sets.
F.3 RedTeam dataset with LlamaGuard as a judge
Figures 12 and 13 detail the results on the Red Team dataset when evaluated with the Llama-Guard
judge. This judge marks responses as unsafe at earlier stages of the conversation, so we set the target
average budget per sample at ¯B= 10 for this experiment. These figures demonstrate the same trend
observed in the previous experiments. All methods are valid in terms of both coverage and budget. In
this setup, the unsafe event occurs very early, and thus DAPRO does not expend too many budget units
in the first phase, which leaves more budget for the second phase. Consequently, DAPRO attains the
lowest mean inverse-censoring weights while still observing the highest number of unsafe events.
As a result, it constructs LPBs with the lowest variance and achieves empirical coverage rates that
are closer to the nominal level. The Greedy approach still tends to exhibit a high variance if its
45

Figure 10:RedTeamdataset withQwen 2.5 14B Instructas a judge: coverage rate, LPB size,
coverage deviation, and budget utilized by various methods across four target LLMs. Target coverage
rate: 90% and target ¯B= 20 budget per sample. Performance metrics are taken over 50 random
splits of the calibration and test sets.
budget split parameter is not tuned correctly. Once again, the LocallyAdaptive technique attains a
variance lower than the static baseline, but higher than our globally optimizedDAPRO.
F.4 RAG hallucinations experiment
In this experiment, we evaluate our framework in a Retrieval-Augmented Generation (RAG) hallu-
cinations setup. The target LLM is provided a context paragraph regarding a specific subject, such
as "Macintosh", or "Beyoncé". The objective of the attacker is to construct adversarial queries that
trick the target model to output a fabricated information that does not exist in the source context. To
46

Figure 11:RedTeamdataset withQwen 2.5 14B Instructas a judge: number of observed unsafe
events, mean inverse-probability weight, coverage variance, and LPB variance by various meth-
ods across four target LLMs. Target coverage rate: 90% and target ¯B= 20 budget per sample.
Performance metrics are taken over 50 random splits of the calibration and test sets.
evaluate the success of the attack, the judge is provided with the ground-truth context, the target’s
response, and outputs a hallucination score ranging from 1to10, where a score of 10is considered
as a hallucination ( Y= 1 ). We set the budget per sample constraint to B/|I cal|= 10 since the event
rate is significantly high for this dataset, of approximately95−99%hallucination rate.
Figures 14 and 15 present the performance of all evaluated methods on the Hallucination dataset.
These results present the same phenomenon observed in the Toxicity and RedTeam experiments:
DAPRO maintains the nominal coverage rate and satisfies the budget constraints across all evaluated
target LLMs. Similarly, DAPRO observes significantly more hallucinated responses than the baselines,
and yields the lowest coverage deviation and variance among the tested methods. LocallyAdaptive
47

Figure 12:RedTeamdataset withLlama-Guardas a judge: coverage rate, LPB size, coverage
deviation, and budget utilized by various methods across four target LLMs. Target coverage rate:
90% and target ¯B= 10 budget per sample. Performance metrics are taken over 50 random splits of
the calibration and test sets.
again exhibits lower variance than the static baseline, though not as low as the globally optimized
DAPRO.
F.5 AutoIF: evaluating utility of helper agents
We demonstrate our proposal in a helper LLM setup. The "attacker" LLM is an agent aiming to guide
the target model to output a response that satisfies complex constraints, such as including a specific
word multiple times. The judge is a set of pre-defined Python scripts that programmatically evaluate
the constraints. If the response of the target satisfies all constraints, the judge outputs10; otherwise,
48

Figure 13:RedTeamdataset withLlama-Guardas a judge: number of observed unsafe events,
mean inverse-probability weight, coverage variance, and LPB variance by various methods across
four target LLMs. Target coverage rate: 90% and target ¯B= 10 budget per sample. Performance
metrics are taken over 50 random splits of the calibration and test sets.
it outputs 1. We repeat our analysis for this experiment once for constructing a calibrated LPB and
once for a UPB.
Figures 16 and 17 display the performance of the LPBs constructed by each method on the AutoIF
dataset. These figures show the same trend observed in the previous experiments: DAPRO attains
the nominal coverage rate and satisfies the budget constraints. Moreover, DAPRO observes the
highest number of successful events compared to the static baseline and exhibits the lowest variance.
LocallyAdaptive exhibits low variance as well. In contrast, the static optimized baseline achieves
a very high variance, and the uncalibrated model does not achieve the desired coverage level. In
addition, these figures show that DAPRO performs similarly across both scores and projection models
49

Figure 14:Hallucinationdataset: coverage rate, LPB size, coverage deviation, and budget utilized
by various methods across four target LLMs. Target coverage rate: 90% and target ¯B= 10 budget
per sample. Performance metrics are taken over 50 random splits of the calibration and test sets.
we examined, indicating that our approach is robust to these choices. Finally, these figures indicate
thatGreedy is not robust to the choice of the data split size parameter, as it heavily affects the
variance of its produced LPBs.
We now evaluate our approach for constructing a calibrated UPB. Since constructing a UPB requires
observing the conversations at higher iterations, we increase the average budget per sample to ¯B= 30
for this setup. We summarize the performance of each method for UPB constructions with the AutoIF
dataset in Figures 18 and 19. These figures show that all calibration algorithms attain the target 70%
coverage rate while satisfying the budget constraint, as guaranteed in theory. The proposed DAPRO
observes more successful events than the baseline and exhibits a lower variance compared to it. Yet,
50

Figure 15:Hallucinationdataset: number of observed successful events, mean inverse-probability
weight, coverage variance, and LPB variance by various methods across four target LLMs. Target
coverage rate: 90% and target ¯B= 10 budget per sample. Performance metrics are taken over 50
random splits of the calibration and test sets.
the variance of all methods is very similar, as the UPBs they generate are closer to the maximum
horizont max, which naturally reduces variance.
F.6 Unbiased estimation of safety metrics at the population level
Beyond evaluating the efficiency of individual adversarial attacks, a critical objective in many
applications, such as large-scale red-teaming, is assessing LLM safety at the population level. For
this purpose, we consider two safety metrics evaluated over a maximum conversational horizon
oftmax= 200 turns: theUnsafe Event Rate ( UER), which measures the proportion of prompts
that elicit an unsafe output within tmaxsteps, and theRestricted Mean Time-to-Unsafe ( RMTTU ),
51

Figure 16:AutoIF(LPB) dataset: coverage rate, LPB size, coverage deviation, and budget utilized
by various methods across four target LLMs. Target coverage rate: 90% and target ¯B= 20 budget
per sample. Performance metrics are taken over 50 random splits of the calibration and test sets.
which quantifies the average number of turns required for the model to produce that unsafe response.
When evaluating LLM utility, e.g., via the AutoIF dataset, where the target event is successful task
completion, we symmetrically refer to these metrics as theSuccessful Event Rate ( SER)and the
Restricted Mean Time-to-Success (RMTTS).
Estimating these metrics under limited computing resources is challenging, as when the budget
allocator halts a conversation early, it introduces a right-censoring bias. Treating these early stopped
samples as safe will increase the safety perceived by the model. Conversely, running every con-
versation for the full tmaxiterations to find the true Oracle outcome is computationally infeasible.
However, by employing a stochastic budget allocator that tracks the probability of reaching the
52

Figure 17:AutoIF(LPB) dataset: number of observed successful events, mean inverse-probability
weight, coverage variance, and LPB variance by various methods across four target LLMs. Target
coverage rate: 90% and target ¯B= 20 budget per sample. Performance metrics are taken over 50
random splits of the calibration and test sets.
censoring time, such as DAPRO , we can construct unbiased estimates of the true population metrics
using only a fraction of the compute.
We achieve this by re-weighting the observed outcomes using the inverse probability of censoring,
similar to our miscoverage rate estimator in (3). Specifically, let m(T) represent an individual-level
metric, such as the unsafe event indicator m(T) =I
T < t max	
. Taking the expectation over the
population yields the true UER:E[m(T)] =P(T < t max). We estimate this population metric using:
ˆm=1
|Ical|X
i∈Icalw(i)I
Ti≤Ci	
m(T i),(22)
53

Figure 18:AutoIF(UPB) dataset: coverage rate, UPB size, coverage deviation, and budget utilized
by various methods across four target LLMs. Target coverage rate: 70% and target ¯B= 30 budget
per sample. Performance metrics are taken over 50 random splits of the calibration and test sets.
where, w(i) =P(T i≤Ci| {(X i, Hi, Ti)}i∈Ical1,Dtrain)−1is the corresponding weight. In our
experiments, since conversation lengths are capped at tmax, any metric depending on this horizon is
fully resolved if we either observe the event ( Ti≤C i) or reach the maximal horizon ( Ci=tmax).
Consequently, our capped estimated metric is given by:
ˆmtmax=1
|Ical|X
i∈Icalwtmax(i)I
Ti≤CiorC i=tmax	
m(T i),
54

Figure 19:AutoIF(UPB) dataset: number of observed successful events, mean inverse-probability
weight, coverage variance, and UPB variance by various methods across four target LLMs. Target
coverage rate: 70% and target ¯B= 30 budget per sample. Performance metrics are taken over 50
random splits of the calibration and test sets.
where the adjusted weights are wtmax(i) =P(T i≤CiorC i=tmax| {(X i, Hi, Ti)}i∈Ical1,Dtrain)−1.
Following the arguments in Theorem 4, ˆmis an unbiased estimator of m(T) , guaranteeing that
E[ ˆm] =E[m(T)].
To demonstrate this empirically, we compute the Oracle UERandRMTTU values using both calibration
and test sets under an unlimited budget setting, so that all unsafe events up to time tmaxare fully
observed. We then run each allocation method with a nominal average budget per sample constraint
set to ¯B= 20 to obtain censoring times, censored event times, and the weights, and construct the
metric estimator ˆmfrom (22). We repeat this process across 50 random calibration-test splits to
measure the mean of estimated metrics and their variance.
55

We compare DAPRO against several baselines: anUncalibrated Uniformallocation, which allocates
the budget uniformly and naively averages the raw censored outcomes without re-weighting, a
Calibrated Uniformallocation that applies re-weighting, theStatic Optimizedbaseline of [ 12], and
theLocallyAdaptive ,Greedy dynamic allocators. Notably, in this setting, theStatic Optimized
baseline functions identically to theCalibrated Uniformallocation. A sample only contributes to the
weighted sum if the conversation reaches the maximum horizon ( C=t max) or successfully reveals an
unsafe event ( C≥T i); if a sample is censored ( C < t maxandC < T i), its contribution is effectively
zero. For this reason, we must set the ˆfprior =tmaxfor all samples, so the static optimization
algorithm has no preference between any two prompts. As a result, the static optimization allocates
the budget uniformly across the entire calibration set.
Figure 20: Population-level safety metric estimation on the Toxicity dataset with Qwen 2.5 14B
Instruct model serves as both the target and attacker. The top two rows display the estimated
Unsafe Event Rate (UER) and Restricted Mean Time-to-Unsafe (RMTTU), alongside their respective
variances. The red dashed lines are the oracle quantities. The bottom row displays the utilized budget
per sample with a red dashed line indicating the target level, and the total number of observed unsafe
events. While the unweighted baseline severely underestimates true vulnerability, and naive calibrated
baselines suffer from extreme variance, DAPRO accurately estimates the true Oracle metrics with zero
bias and minimal variance. Furthermore, DAPRO achieves the highest budget efficiency, successfully
observing the most unsafe events under the same resource constraints. Results are aggregated over 50
random calibration-test splits.
Figure 20 presents the estimated UERandRMTTU , along with their variance, on the Toxicity dataset
with Qwen 2.5 14B Instruct as both target and attacker. TheUnweighted Uniformbudget allocator
underestimates the UERsince it does not re-weight the contribution of the uncensored samples to the
sum. By correcting for this bias with re-weighting, theCalibrated Uniformachieves the correct
value in expectation, but exhibits a high variance. The static optimized approach has zero bias as well,
although it exhibits a higher variance, as its allocation is highly inefficient. In stark contrast, DAPRO
achieves the best of both worlds: it recovers the true Oracle metrics with zero bias while achieving
the lowest variance and satisfying the budget constraint. By globally optimizing the advancement
56

probabilities, DAPRO expends the budget on highly vulnerable conversations, efficiently observing the
highest number of unsafe events.
Figures 21 through 25 present the estimated UERandRMTTU , alongside their variance, across all
four experimental setups. We compare the Static Optimized baseline, the dynamic Greedy approach
(configured with a data split level of 0.1, as higher values resulted in extreme instability), and our
LocallyAdaptive andDAPRO . Since all evaluated methods correctly address the right-censoring
distribution shift by applying re-weighting, they successfully attain the true Oracle value (marked
by the red dashed line) in expectation. As all methods have zero bias, the estimator’s measure
of reliability is its variance. The bar plots in these figures demonstrate that the static baseline
attains a high variance. The Greedy allocator tends to attain a variance lower than the static baseline,
although not consistently, as its data-split ratio parameter is not tuned. In contrast, LocallyAdaptive
consistently attains variance lower than the static baseline, and occasionally even lower than our global
optimized DAPRO . This establishes LocallyAdaptive as a competitive alternative, particularly as it
provides an exact, finite-sample budget guarantee in expectation. Crucially, this theoretical guarantee
holds purely under the mild assumption of data exchangeability; it does not require an accurate
score-to-probability mapping to satisfy the budget constraint. While DAPRO generally maintains lower
variance than the static approach, the rare instances where it exhibits a higher variance can likely
be attributed to an inaccurate score-to-probability mapping. Overall, this experiment demonstrates
that while proper statistical re-weighting is necessary to construct an unbiased estimator, efficient
budget allocation is essential to obtain low-variance, reliable safety metrics for large-scale tasks under
limited compute resources.
Figure 21: Population-level metric estimation on theToxicitydataset. Boxplots display the distribu-
tion of the estimated Unsafe Event Rate ( UER) and Restricted Mean Time-to-Unsafe ( RMTTU ) across
50 random splits. The red dashed lines represent the true Oracle metrics. While all methods achieve
the correct metric in expectation, LocallyAdaptive andDAPRO yield the tightest variance compared
to the static baselines andGreedyapproach.
F.7 Ablation study forDAPRO
In this section, we analyze the robustness of our proposed DAPRO to three factors: the size of the first
calibration set split ( N1), the quality of the score, and the nominal budget per sample. This study
reveals that DAPRO is effective and valid under varying experimental conditions. Even under extreme
conditions, such as a limited budget or severely corrupted scores, our framework consistently achieves
the desired coverage level, and its budget consumption is close to the target level. We evaluate the
performance on the Toxicity dataset using the Qwen 2.5 14B Instruct model serving both as attacker
and target.
57

Figure 22: Population-level metric estimation on theRed Teamdataset evaluated with aQwenjudge.
The red dashed lines denote the true Oracle metrics. Our proposed LocallyAdaptive andDAPRO
consistently achieve lower variance than the static optimized baseline across various target models.
Figure 23: Population-level metric estimation on theRed Teamdataset evaluated with aLlama
Guardjudge. The red dashed lines denote the true Oracle metrics. Both the static and greedy
baselines exhibit high variance across different data splits, whereas DAPRO remains highly stable and
consistently recovers the Oracle values.
F.7.1 The effect of the calibration set split
We employ DAPRO with various values of the first calibration set split size ( N1) and display its
performance in Figure 26. DAPRO consistently achieves the nominal coverage rate, 90%, and satisfies
the target budget per sample of 20across all tested values of N1. This figure reveals that the budget
used by DAPRO converges to the target level as N1increases. However, DAPRO attains a higher mean
weight for higher values of N1. Yet, DAPRO achieves a lower coverage difference and lower coverage
variance compared to the static baseline allocation of [12] for all examined values ofN 1.
F.7.2 The effect of the quality of the score
To evaluate the robustness of our approach to the quality of the scores, we conduct an ablation where
the scores used in Phase I are artificially corrupted. Specifically, we add random noise to the original
scores:
Si(t) = (1−λ)· S(H i(t)) +λ·U i(t)
58

Figure 24: Population-level metric estimation on theHallucinationdataset. The red dashed lines
denote the true Oracle metrics. The LocallyAdaptive andDAPRO exhibit the lowest variance,
resulting in the most reliable estimation of both theUERandRMTTUmetrics.
Figure 25: Population-level metric estimation on theAutoIFdataset. The red dashed lines denote the
true Oracle metrics. The LocallyAdaptive andDAPRO exhibit the lowest variance, resulting in the
most reliable estimation of both theSERandRMTTSmetrics.
where Ui(t)∼Uni(0,1) is a uniform random variable between 0 and 1. The parameter λ∈[0,1]
controls the score degradation level. We fix the first-split set size to N1= 100 . As shown in
Figure 27, DAPRO maintains a valid empirical coverage ( ≈90% ) across all noise levels, indicating
that its theoretical guarantees hold regardless of the quality of the score. Furthermore, DAPRO
satisfies the budget constraint of 20units per sample for low values of λ, while exceeding the budget
usage for higher values of λ. Asλ→1 and the scores lose their predictive signal, DAPRO can no
longer distinguish between promising and unpromising conversations. Nevertheless, DAPRO attains a
coverage rate closer to the desired level with a lower variance compared to the static baseline for all
values ofλ. This demonstrates the robustness of our proposal to scores that are not informative.
F.7.3 The effect of the available budget
We employ DAPRO and the static baseline of [ 12] with varying levels of budget per sample and present
their performance in Figure 28. To adapt our method, and its data split to the varying budget per
sample, we adapt its N1parameter to the nominal average budget per sample. For an average budget
per sample lower than or equal to 10, we set N1= 25 , and for an average budget per sample of 25we
59

Figure 26: Impact of first calibration set split size ( N1) on empirical coverage, budget per sample,
and mean weight on the Toxicity dataset. The Qwen 2.5 14B Instruct model serves as both the
attacker and target. Nominal coverage level is set to 1−α= 90% and the target budget per sample
is20. Shaded regions denote semi-deviations over 50 random calibration-test splits. As N1increases,
DAPRO consistently maintains valid coverage and satisfies to the budget constraint, with variance
notably decreasing at larger sample sizes.
setN1= 50 . For any higher value of budget per sample, we set N1= 100 . This figure demonstrates
that under a low budget constraint, the static baseline suffers from a high coverage variance that
highly deviates from the target level. It requires a significant budget increase to eventually converge
towards coverage values attained by our method with a less available budget. In stark contrast, DAPRO
perfectly maintains valid empirical coverage ( ≈90% ) across all budget levels. Furthermore, as the
nominal budget increases, our approach attains lower values of mean weight. This indicates a more
efficient resource utilization ofDAPRO.
60

Figure 27: Impact of score informativeness ( λ) on empirical coverage, budget consumed per sample,
and mean weight on the Toxicity dataset. The Qwen 2.5 14B Instruct model serves as both attacker
and target, with a first-split set size of N1= 100 . Scores are corrupted by injecting random noise
with levelλ. The nominal coverage level is set to1−α= 90%and the target budget is20. Shaded
regions denote semi-deviations over 50 random calibration-test splits.
61

Figure 28: Impact of the nominal budget per sample on empirical coverage, budget consumed per
sample, and mean allocation weight on the Toxicity dataset. The Qwen 2.5 14B Instruct model
serves as both attacker and target. The target coverage level is 1−α= 0.90 . Shaded regions denote
semi-deviations across 50 random calibration-test splits. DAPRO satisfies both coverage and budget
constraints across all scenarios, whereas the static baseline exhibits high variance and coverage
deviation under low-budget regimes.
62
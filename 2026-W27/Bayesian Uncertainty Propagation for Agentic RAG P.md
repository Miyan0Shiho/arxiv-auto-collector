# Bayesian Uncertainty Propagation for Agentic RAG Pipelines: A Proof-of-Concept Study on Multi-Hop Question Answering

**Authors**: Louis Donaldson, Connor Walker, Koorosh Aslansefat, Yiannis Papadopoulos

**Published**: 2026-07-01 14:08:58

**PDF URL**: [https://arxiv.org/pdf/2607.00972v1](https://arxiv.org/pdf/2607.00972v1)

## Abstract
Trustworthy deployment of Agentic Retrieval-Augmented Generation (RAG) systems requires mechanisms for estimating when multi-stage reasoning pipelines may fail. This paper presents an uncertainty-aware Agentic Retrieval-Augmented Generation (RAG) framework in which planner, evaluator and generator stages produce uncertainty signals derived from semantic divergence and generator self-evaluation. These signals are propagated through a Bayesian Network (BN) to estimate system-level uncertainty and provide node-level indicators of potential failure points across the workflow. The approach is evaluated on StrategyQA and HotpotQA using GPT-3.5-Turbo and GPT-4.1-Nano, with Area Under the Receiver Operating Characteristic Curve (AUROC), Area Under the Accuracy-Rejection Curve (AUARC), Expected Calibration Error (ECE), and Brier Score used to assess discrimination, selective prediction and calibration. Results show that Bayesian propagation is more effective on HotpotQA, where uncertainty accumulates across multi-hop reasoning stages, while StrategyQA exposes limitations caused by miscalibration and unreliable upstream signals. The study positions Bayesian uncertainty propagation as a promising but preliminary mechanism for monitoring Agentic RAG systems, with future validation required in industrial domains such as Offshore Wind (OSW) maintenance decision support.

## Full Text


<!-- PDF content starts -->

Bayesian Uncertainty Propagation for Agentic
RAG Pipelines: A Proof-of-Concept Study on
Multi-Hop Question Answering
Louis Donaldson1[0009-0000-3862-8061], Connor
Walker1[0009-0008-8181-1644], Koorosh Aslansefat1[0000-0001-9318-8177], and
Yiannis Papadopoulos1[0000-0001-7007-5153]
University of Hull, Hull HU6 7RX, UK
l.donaldson-2020@hull.ac.uk
Abstract.TrustworthydeploymentofAgenticRetrieval-AugmentedGen-
eration (RAG) systems requires mechanisms for estimating when multi-
stage reasoning pipelines may fail. This paper presents an uncertainty-
aware Agentic Retrieval-Augmented Generation (RAG) framework in
which planner, evaluator and generator stages produce uncertainty sig-
nalsderivedfromsemanticdivergenceandgeneratorself-evaluation.These
signals are propagated through a Bayesian Network (BN) to estimate
system-level uncertainty and provide node-level indicators of potential
failure points across the workflow. The approach is evaluated on Strate-
gyQA and HotpotQA using GPT-3.5-Turbo and GPT-4.1-Nano, with
Area Under the Receiver Operating Characteristic Curve (AUROC),
Area Under the Accuracy-Rejection Curve (AUARC), Expected Cal-
ibration Error (ECE), and Brier Score used to assess discrimination,
selective prediction and calibration. Results show that Bayesian prop-
agation is more effective on HotpotQA, where uncertainty accumulates
across multi-hop reasoning stages, while StrategyQA exposes limitations
caused by miscalibration and unreliable upstream signals. The study po-
sitions Bayesian uncertainty propagation as a promising but preliminary
mechanism for monitoring Agentic RAG systems, with future validation
required in industrial domains such as Offshore Wind (OSW) mainte-
nance decision support.
Keywords:Multi-Agent Systems·Uncertainty Quantification·Run-
time Monitoring·AI-Enabled Industrial Decision Support Systems
1 Introduction
As Large Language Models (LLMs) become increasingly integrated into indus-
trial workflows, their ability to automate routine tasks and provide human-
understandable interpretations of complex data has the potential to improve op-
erational efficiency across a wide range of sectors. As confidence in these systemsarXiv:2607.00972v1  [cs.AI]  1 Jul 2026

2 L. Donaldson et al.
grows, they are expected to play increasingly important roles in business-critical
processes. One potential application area is Offshore Wind (OSW) maintenance
scheduling, where increasing turbine complexity and expanding wind farm ca-
pacity require the interpretation of large volumes of operational data to support
maintenance planning. Recent work has demonstrated the potential of LLM-
based decision-support and safety monitoring within this domain, while also
highlighting the need for reliable run-time monitoring and trustworthy uncer-
tainty estimation [16].
1.1 Background and Motivation
The adoption of LLMs within specialised domains is accompanied by signifi-
cant challenges regarding transparency, reliability and trust. In healthcare, the
inability to inspect the reasoning process behind generated recommendations
may affect clinical decision making and patient safety, while privacy consider-
ations further complicate deployment [13,1]. Similar concerns arise in finance
and legal applications, where opaque reasoning makes regulatory compliance
and decision auditing difficult [1,13]. In scheduling, manufacturing and Predic-
tive Maintenance (PdM), operators often require understandable explanations
before trusting automated decisions, yet conventional black-box models provide
limited insight into the reasoning behind their outputs [13,20,9].
Retrieval-AugmentedGeneration(RAG)partiallyaddressestheselimitations
by grounding generated responses in retrieved knowledge sources, allowing sup-
porting evidence to be inspected and verified by human operators [2,3,15]. Hy-
brid and Multi-Agent Architectures (MAAs) further improve reasoning by de-
composing complex tasks across specialised agents or combining LLM reasoning
with domain-specific models, producing more interpretable reasoning workflows
[8,19]. In parallel, prompting paradigms such as ReAct and context engineering
improve reasoning consistency through external tool use and structured contex-
tual guidance [1,14].
Although these approaches improve reasoning quality and interpretability,
they do not explicitly model how uncertainty propagates across sequential rea-
soningstages.Inmulti-agentRAGpipelines,uncertaintyintroducedduringplan-
ning, retrieval or evaluation may influence downstream decisions and ultimately
affect the reliability of the final response. This motivates the development of
lightweight uncertainty propagation methods capable of monitoring confidence
throughoutthereasoningworkflow.Thecontributionsofthispaperarethreefold:
1. Weproposealightweightuncertainty-awaremonitoringarchitectureforAgen-
tic RAG systems, where Planner, Evaluator and Generator agents produce
complementary uncertainty signals during multi-stage reasoning.
2. WeintroduceaBayesianNetwork(BN)-baseduncertaintypropagationmodel
that combines node-level uncertainty estimates into an interpretable system-
level confidence estimate capable of indicating which stages contribute most
to system-level uncertainty.

Uncertainty Monitoring for LLM Agents 3
3. We present a proof-of-concept empirical evaluation on the StrategyQA and
HotpotQA multi-hop question answering benchmarks using GPT-3.5-Turbo
and GPT-4.1-Nano, demonstrating the strengths and limitations of Bayesian
uncertainty propagation across multiple uncertainty evaluation metrics, in-
cluding Area Under the Receiver Operating Characteristic Curve (AUROC),
Area Under the Accuracy-Rejection Curve (AUARC), Expected Calibration
Error (ECE) and Brier Score.
2 Methodology on Uncertainty Quantification
Agentic RAG systems operate as sequential decision-making processes that can
be modelled as a Partially Observable Markov Decision Process (POMDP) [12].
Because eachstage dependson previousdecisions,uncertaintyintroduced during
planning, retrieval or evaluation may propagate through the workflow, leading
to cascading errors and semantic drift. Consequently, reliable uncertainty quan-
tification is required to monitor confidence throughout multi-stage reasoning.
2.1 Token-level Entropy
Also referred to as naive entropy, it represents the uncertainty of a LLM at the
level of individual word-pieces or characters during the text generation process
[5]. In autoregressive models, this uncertainty is derived from the softmax proba-
bility distribution over all possible next tokens in the model’s vocabulary. Given
an inputxand model parametersθ, the predictive distributions of the outputy
canbewrittenasp θ(y|x),andforageneratedsequenceofoutputsy= (z 1, ..., z L)
with lengthL, the probability can be decomposed into conditional token prob-
abilities:
pθ(y|x) =LY
t=1pθ(zt|z<t, x)(1)
Taking the logarithm to yield the log probability of the sequence and apply-
ing length normalisation to ensure a fair comparison between different response
lengths, this can be computed as:
si=1
LiLiX
t=1logp θ(zt|z<t, x)(2)
which is then transformed into a feature of confidence denoted asq i=exp(s i).
This allows the probability of uncertainty to be estimated as
P(X i=success|q i) =f cal(qi)
wheref caldenotes a calibration function estimated using runs. This value is
then incorporated into the BN as probabilistic evidence.

4 L. Donaldson et al.
Although naive entropy metrics are not direct indicators of the truthfulness
of model outputs, in multi-agent systems, they can be used to parameterise the
Prior Probabilities of nodes in a BN. They can act as local confidence estimates
that are transformed into bounded features and then propagated through the
BN to understand how the uncertainty accumulates across workflow stages.
The use of logprobs in this system is utilised for the Generator node using the
P(True) self-evaluation approach. This is a supervised "in-context" uncertainty
estimation to evaluate the truthfulness of its own outputs [5]. Once the RAG
pipelinehasfinishedexecutingandproducedananswer,itisthenposedafurther
question "Are you sure that the answer provided is correct?" to which it can only
reply with a single token (true/false or yes/no). The confidence score, or P(True)
is the probability that the model assigns to the affirmative token [5].
2.2 Semantic Divergence
Alsoreferredtoas"semanticdrift",canoccurduringiterativeretrievalprocesses.
As an agent autonomously reformulates its search queries across multiple steps,
it can gradually diverge from the original information requirement [12]. Because
Agentic RAG systems utilise a sequential decision process, minor deviation in
early steps can compound across iterations, leading to "reasoning collapse" and
unnecessary computational costs [12]. Semantic divergence can be used as a
metric to identify confabulations; hallucinations that are both arbitrary and
semantically misaligned with the input context [7].
Theoretical FoundationsThe Semantic Divergence formulation in this pa-
per is motivated by the UProp framework of Duan et al,. [4], which decomposes
sequential decision uncertainty into Intrinsic Uncertainty (IU) and Extrinsic Un-
certainty (EU). In UProp, EU at steptis estimated via the Pointwise Mutual
Information (PMI) between the current step’s decision distribution and each
preceding step’s committed decision:
dPMI(y t;y(k)
t−1|x) =−logNX
n=1KN
d
y(n)
t−1, y(k)
t−1
(3)
whereK Nis a Gaussian kernel applied to the fuzzy string distancedbetween the
Nsamples at stept−1and the committed decisiony(k)
t−1. Semantic Divergence
instantiatesananalogousmeasurementwithineachnode,replacingtheGaussian
kernel over string distances with the Wasserstein distance between the commit-
ted output distribution and the re-sampled output distribution in a projected
embedding space. Both metrics measure the spread of the decision distribution
around the committed trajectory point; Semantic Divergence therefore provides
a per-node IU estimate that is directly comparable to UProp’s PMI-based EU
accumulation.

Uncertainty Monitoring for LLM Agents 5
Comparison with Semantic EntropySemantic Entropy (SE) [10] clusters
Ksamples by semantic equivalence and computes Shannon entropy over the
resulting cluster distribution:
SE=−X
cp(c) logp(c)(4)
SE measures uncertainty by clustering multiple sampled outputs according
to semantic equivalence and computing the entropy of the resulting distribution.
To prevent repeated JavaScript Object Notation (JSON) outputs collapsing into
a single SE cluster, this paper proposes the use of semantic divergence to anchor
measurements to the committed output rather than measuring within-sample
spread.
2.3 Bayesian Networks
In this Multi-Agent System (MAS) approach, BNs are used to represent uncer-
tainty propagation by modelling the dependencies between agents. Each stage
of the workflow is modelled as a node, and the dependencies of each node are
captured through directed edges. A combination of semantic-divergence scores
and token-level log-probabilities are used to quantify uncertainty at each node,
which is then propagated through the BN to quantify the overall uncertainty
level.
BNs are probabilistic graphical models that represent conditional depen-
dencies between random variables using directed acyclic graphs [11]. LetX=
{X1, ..., X n}denote the workflow tasks, whereX 1,X 2andX 3correspond to the
Planner, Evaluator and Generator agents respectively.
P(X 1, ..., X n) =nY
i=1P(X i|Pa(X i))(5)
wherePa(X i)denotes the set of parent nodes ofX i. The corresponding
ConditionalProbabilityTable(CPT)specifiestheconditionalprobabilityofeach
node given its parents. Within the proposed Agentic RAG framework, each node
representstheuncertaintystateoftheAgenticRAGstage,withconfidencescores
parameterising the prior probability according toP= (X i=success) =c(y i).
The BN representation used in this study represents an abstracted execution
trace of the workflow rather than the full control logic of the Agentic system.
Temporal modelling does not extend further than parent-child dependencies,
nor is there any task priority or repeated executions, feedback loops or iterative
cycles.IftheEvaluatordeemsthereisinsufficientevidencetoproduceaconfident
answer, the final accepted cycle is used as evidence for the corresponding node.
For the Overall node that relies on preceding steps in the workflow, the confi-
dence scores contribute to the CPT. As seen in Table 1 the relationship between
Overall and its parent nodes is characterised for this study as a deterministic
OR gate. This implies that Overall is FAIL when one node in the pipeline is un-
certain or FAIL. The threshold for determining a FAIL or OK in this study is set

6 L. Donaldson et al.
Fig.1: Agentic RAG pipeline with BN uncertainty propagation used to flag/ac-
cept/abstain answers based on Overall OK/FAIL.
Table 1: CPT forOverallunder a deterministic OR gate.
Planner Evaluator Generator P(Overall=OK)
OK OK OK 1.0
OK OK FAIL 0.0
OK FAIL OK 0.0
OK FAIL FAIL 0.0
FAIL OK OK 0.0
FAIL OK FAIL 0.0
FAIL FAIL OK 0.0
FAIL FAIL FAIL 0.0
at 0.5 (or 50 %). If an uncertainty value of a node exceeds this, it is considered
a FAIL. Using a deterministic OR gate in this study is intended to test if the
simplification of the uncertainty propagation behaviour throughout an Agentic
RAG pipeline is a viable method to quantify the overall uncertainty of responses
and lead to detection of possible hallucinations or confabulations.
New evidence is also included in the network and the posterior probabilities
are updated alongside the workflow run. The interpretability of the BNs graphi-
cal structure makes it possible to identify which components can contribute most
to the overall uncertainty.
The objective of performing probabilistic inference, once the BN structure
and CPTs are defined, is to estimate posterior probabilities across the workflow.
The objective of this is to determine how uncertainty in each node affects the
reliability of the final workflow outcome.
3 Experiments & Results
In this section, we provide the experimental setup and discuss the performance
of multi-step reasoning benchmarks including StrategyQA [6] and HotpotQA

Uncertainty Monitoring for LLM Agents 7
Table2:AUROCresultspernodeandBNposterior(Overall)acrossbenchmarks.
Bold denotes the best BN score per row. AUROC>0.5 indicates better-than-
random discrimination.
Benchmark: StrategyQA (200 Samples)
Model Planner Evaluator Generator BN (Ours) UProp
GPT-3.5-Turbo 0.4138 0.47990.58810.5592 0.6040
GPT-4.1-Nano 0.4696 0.46750.62130.6083 0.5440
Benchmark: HotpotQA (200 Samples)
Model Planner Evaluator Generator BN (Ours) UProp
GPT-3.5-Turbo 0.6209 0.6117 0.69930.71600.7470
GPT-4.1-Nano 0.5919 0.5369 0.71450.72400.7188
[17]. These multi-hop, question-answering benchmarks were chosen due to their
popularity and academic presence. In these, LLMs are tasked with answering
questions that require multi-hop reasoning where each action will provide a key-
word or phrase request to a Wikipedia engine for retrieval using a Reasoning and
Acting(ReAct)[18]methodology.Bothbenchmarkexperimentsarerunwith200
randomly sampled questions.
In this study we use commercial LLMs provided by OpenAI, GPT-3.5-Turbo
and GPT-4.1-Nano. For the generative hyper-parameters, we assign the Planner
and Generator agents a temperature of 0.2, the Evaluator a temperature of 0.8
with a maximum number of new tokens as 256. By default the per-step sample
number ofNis set to 10.
As an indicative comparison from the aforementioned benchmarks, we are
usingUProp’s[4]AUROCscoresasanexternalreferencefortheresultsproduced
in the approach proposed in this study.
As shown in Table 2, the AUROC results demonstrate contrasting behaviour
acrossthetwobenchmarks.OnStrategyQA,theGeneratorprovidesthestrongest
individual uncertainty signal, while the Planner and Evaluator perform close to
random, limiting the effectiveness of Bayesian uncertainty propagation. Conse-
quently, the Overall BN remains below the UProp baseline for GPT-3.5-Turbo,
although it exceeds the baseline for GPT-4.1-Nano. In contrast, all nodes achieve
above random discrimination on HotpotQA, and the Overall BN becomes com-
petitive with UProp, exceeding the baseline for GPT-4.1-Nano while remaining
only slightly below it for GPT-3.5-Turbo. These results suggest that Bayesian
uncertainty propagation is most effective when uncertainty accumulates across
multiple reasoning stages.
The AUARC results presented in Table 3 complement the AUROC analysis
by evaluating selective prediction performance. On StrategyQA, the Generator
provides the strongest individual uncertainty estimate, indicating that P(True)
self-evaluation is sufficient for effective selective prediction on the simpler bench-
mark. In contrast, the Overall BN achieves the highest AUARC on HotpotQA,

8 L. Donaldson et al.
Table 3: AUARC results across benchmarks. Higher values indicate greater
improvement in accuracy under selective prediction. Bold denotes the highest
AUARC value in each row.
Benchmark ModelPlanner Evaluator GeneratorBN (Ours)
StrategyQAGPT-3.5-Turbo 0.4968 0.54470.63870.5952
GPT-4.1-Nano 0.5554 0.56800.68310.6764
HotpotQAGPT-3.5-Turbo 0.3920 0.3611 0.41480.4254
GPT-4.1-Nano 0.3213 0.2733 0.36690.3862
Table 4: ECE across benchmarks, per node and the BN posterior (Overall).
Lower values indicate better calibration. Bold denotes the best ECE score in
each row.
Benchmark ModelPlanner Evaluator GeneratorBN (Ours)
StrategyQAGPT-3.5-Turbo 0.21860.21270.3703 0.4814
GPT-4.1-Nano 0.15040.12050.4231 0.4942
HotpotQAGPT-3.5-Turbo 0.3410 0.4728 0.25020.1376
GPT-4.1-Nano 0.3684 0.4199 0.27430.1120
suggesting that combining uncertainty across multiple pipeline stages becomes
increasingly beneficial as reasoning complexity increases. Although AUARC de-
creases on HotpotQA despite improved AUROC, this likely reflects the substan-
tially lower baseline accuracy of the benchmark, illustrating that discrimination
quality and selective prediction utility capture different aspects of uncertainty
estimation.
The ECE results presented in Table 4 indicate that calibration is strongly
benchmark dependent. On StrategyQA, the Overall BN substantially overesti-
mates failure probability, suggesting that the deterministic OR gate propagates
conservative uncertainty estimates even when the models answer correctly. In
contrast, calibration improves considerably on HotpotQA, where the higher task
difficulty aligns more closely with the BNs uncertainty estimates. On HotpotQA,
the Generator remains the strongest individual uncertainty signal, though its
calibration is weaker than the Planner’s and Evaluator’s on StrategyQA. The
calibration of the Evaluator node itself is inconsistent across benchmarks, re-
inforcing the Generator node uncertainty signal is the most reliable source at
identifying failures, if not yet calibrated in absolute terms.
The Brier Score results presented in Table 5 reinforce the calibration trends
observed in the ECE analysis. StrategyQA exhibits poor probabilistic calibra-
tion, reflecting a systematic mismatch between predicted and observed failure
probabilities. In contrast, the Overall BN achieves substantially improved Brier

Uncertainty Monitoring for LLM Agents 9
Table5:BrierScoreacrossbenchmarks,pernodeandtheBNposterior(Overall).
Lower values indicate better probabilistic accuracy. The random baseline is 0.25;
values below this indicate better-than-random calibration. Bold denotes the best
Brier Score in each row.
Benchmark ModelPlanner Evaluator GeneratorBN (Ours)
StrategyQAGPT-3.5-Turbo0.29680.2989 0.3906 0.4831
GPT-4.1-Nano 0.28050.26350.4253 0.4816
HotpotQAGPT-3.5-Turbo 0.3197 0.4289 0.27810.2023
GPT-4.1-Nano 0.3286 0.3745 0.28020.1788
Scores on HotpotQA, demonstrating that its probability estimates become more
accurate as task complexity increases. Together, the ECE and Brier Score re-
sults indicate that the proposed Bayesian uncertainty propagation framework
produces better calibrated confidence estimates when the underlying uncertainty
better reflects the true difficulty of the reasoning task.
4 Evaluation
4.1 Does the Bayesian Network Add Value Over Individual Nodes?
The evaluation demonstrates that the benefit of Bayesian uncertainty propa-
gation depends on the reliability of the uncertainty signals being combined.
On HotpotQA, the Overall BN consistently outperforms the individual pipeline
nodes, indicating that propagating uncertainty across multiple reasoning stages
provides additional predictive value. In contrast, StrategyQA is dominated by
the Generator uncertainty signal, suggesting that propagating weaker Planner
and Evaluator signals can reduce overall performance and the use of P(True)
self-evaluation techniques for the Generator node alone is sufficient for simpler
queries to predict uncertainty. This behaviour highlights a limitation of the de-
terministic OR gate, which treats all uncertainty signals equally and cannot
down-weight unreliable upstream nodes.
4.2 Does Task Complexity Affect Uncertainty Propagation?
The results indicate that task complexity has a substantial influence on the effec-
tiveness of Bayesian uncertainty propagation. Compared with StrategyQA, Hot-
potQA exhibits stronger discrimination and calibration performance, suggesting
that uncertainty estimates become more informative when reasoning requires
multiple retrieval and reasoning stages. Although selective prediction perfor-
mance decreases on HotpotQA, this is likely a consequence of the benchmark’s
substantially lower baseline accuracy rather than poorer uncertainty estimation.
Overall, these findings suggest that uncertainty propagation provides the great-
est benefit when uncertainty accumulates throughout multi-step reasoning.

10 L. Donaldson et al.
4.3 Calibration Versus Discrimination Trade-off
The experiments demonstrate a clear trade-off between discrimination and cal-
ibration that depends on task difficulty. While the BN provides stronger cali-
bration on HotpotQA, StrategyQA reveals a tendency to overestimate failure
probability, reflecting the conservative behaviour introduced by the determin-
istic OR gate. This conservatism is undesirable for well-calibrated probability
estimation on simpler tasks but may be advantageous in safety-critical settings,
where failing to identify an incorrect response is often more costly than issuing
unnecessary warnings. Consequently, although the current framework requires
improved calibration for general purpose applications, its risk-averse behaviour
may be well suited to industrial decision support domains such as OSW main-
tenance scheduling.
4.4 Limitations
There are three principal limitations that constrain the generalisability of the
currentresults.FirstisthatthedeterministicORgatecannotadapttobenchmark-
specific node reliability. As demonstrated in Table 2, the Planner and Evalua-
tor nodes show below-random AUROC scores on StrategyQA yet are weighted
equally to the Generator in the OR gate, introducing noise into the Overall pos-
terior and contributes to the below-UProp AUROC values on that benchmark
(with GPT-4.1-Nano Overall AUROC as the exception). A noisy-OR parametri-
sation with learned inhibition parameters would address this issue by down-
weighting nodes whose signals are empirically unreliable. This represents the
most impactful architectural improvement available within the existing frame-
work.
Secondly,theP(True)self-evaluationsignalexhibitssystematicconservatism,
as the Generator assigns high uncertainty scores across a wide range of answers.
This produces strong AUROC but poor Brier Score on StrategyQA. Using ap-
proaches such as post-hoc calibration with isotonic regression or Platt scaling
fitted on a held-out validation set would allow the decoupling of the discrimina-
tory quality from the calibration quality. This would be expected to drastically
improve Brier Score on StrategyQA without degrading AUROC scores.
The third limitation is that the Wikipedia retrieval environment used in
both benchmarks for this study differs from the knowledge bases that would be
used in OSW maintenance scheduling applications. Uncertainty signals would
behave differently in domain-specific settings and the current results should be
interpreted as a proof-of-concept on publicly available multi-hop benchmarks
rather than a direct validation of performance in the targeted industrial domain.
5 Conclusion
This paper presented a lightweight uncertainty-aware Agentic RAG framework
in which per-node uncertainty signals derived from Wasserstein semantic diver-
gence and P(True) self-evaluation are propagated through a BN to estimate

Uncertainty Monitoring for LLM Agents 11
system-level confidence. The framework was evaluated on the StrategyQA and
HotpotQAmulti-hopreasoningbenchmarksusingGPT-3.5-TurboandGPT-4.1-
Nano across AUROC, AUARC, ECE and Brier Score. The proposed architecture
provides a modular and computationally lightweight mechanism for monitoring
uncertainty within multi-stage reasoning pipelines.
The experimental results demonstrate that Bayesian uncertainty propaga-
tion is most effective on tasks requiring genuine multi-hop reasoning, where
uncertainty accumulates across multiple reasoning stages. While the Genera-
tor node’s P(True) self-evaluation approach consistently provides the strongest
individual uncertainty signal, combining uncertainty across pipeline stages im-
proves overall performance on the more challenging benchmark. These findings
support Bayesian uncertainty propagation as a promising approach for run-time
uncertainty monitoring in Agentic RAG systems while also highlighting the im-
portance of reliable upstream uncertainty estimates.
The current framework remains limited by the deterministic OR gate, conser-
vative P(True) calibration and evaluation on general purpose question-answering
benchmarks rather than industrial data. Future work will investigate adap-
tive Bayesian structures, improved calibration techniques and validation using
domain-specific retrieval corpora, including OSW maintenance decision support
scenarios.
Data and Code Availability
The datasets used in this article are publicly available and the code produced
for this study, including the Agentic RAG pipeline, Bayesian Network inference
scripts, and evaluation notebooks, is publicly available at:
https://github.com/LouisDonaldson/BN-Uncertainty-Propagation.git
References
1. Bilal, A., Ebert, D., Lin, B.: Llms for explainable ai: A comprehensive survey
(2025),https://arxiv.org/abs/2504.00125
2. DENG, H., Namoano, B., ZHENG, B., Khan, S., Ahmet Erkoyuncu, J.: From
prediction to prescription: Large language model agent for context-aware main-
tenance decision support. PHM Society European Conference8, 10 (06 2024).
https://doi.org/10.36001/phme.2024.v8i1.4114
3. Dogra, S., Erras, M., Farrell-Morris, C., Maple, C., Hairs, P., McCahon, W.,
Niven, T., Thornely, B., Zitani, L.: Generative ai in action: Opportunities & risk
management in financial services january 2025 (04 2025),https://committees.
parliament.uk/writtenevidence/140284/pdf/
4. Duan, J., Diffenderfer, J., Madireddy, S., Chen, T., Kailkhura, B., Xu, K.: Uprop:
Investigating the uncertainty propagation of llms in multi-step agentic decision-
making (2025),https://arxiv.org/abs/2506.17419

12 L. Donaldson et al.
5. Farquhar, S., Kossen, J., Kuhn, L., Gal, Y.: Detecting hallucinations in
large language models using semantic entropy. Nature630, 625–630 (06
2024). https://doi.org/10.1038/s41586-024-07421-0,https://www.nature.com/
articles/s41586-024-07421-0#Sec2
6. Geva, M., Khashabi, D., Segal, E., Khot, T., Roth, D., Berant, J.: Did aristotle
use a laptop? a question answering benchmark with implicit reasoning strategies.
arXiv (Cornell University) (01 2021). https://doi.org/10.48550/arxiv.2101.02235
7. Halperin, I.: Prompt-response semantic divergence metrics for faithfulness hal-
lucination and misalignment detection in large language models (2025),https:
//arxiv.org/abs/2508.10192
8. Harbola, C., Purwar, A.: Prescriptive agents based on rag for automated mainte-
nance (param) (2025),https://arxiv.org/abs/2508.04714
9. Hughes, P., Perinpanayagam, S., Ball, P.: Cost-efficiency and cost-effectiveness
of xai in predictive maintenance. IEEE Access13, 151664–151670 (2025).
https://doi.org/10.1109/access.2025.3601385
10. Kuhn, L., Gal, Y., Farquhar, S.: Semantic uncertainty: Linguistic invariances for
uncertainty estimation in natural language generation. (2023),https://arxiv.
org/abs/2302.09664
11. Liu, Z., Liu, Y.: A bayesian network based method for reliability analysis of sub-
sea blowout preventer control system. Journal of Loss Prevention in the Process
Industries59, 44–53 (05 2019). https://doi.org/10.1016/j.jlp.2019.03.004
12. Mishra, S., Niroula, S., Yadav, U., Thakur, D., Gyawali, S., Gaire, S.: Sok: Agen-
tic retrieval-augmented generation (rag): Taxonomy, architectures, evaluation, and
research directions (2026),https://arxiv.org/abs/2603.07379
13. Powell, C., Riccardi, A.: Generating textual explanations for scheduling systems
leveragingthereasoningcapabilitiesoflargelanguagemodels.JournalofIntelligent
Information Systems63, 1287–1337 (04 2025). https://doi.org/10.1007/s10844-
025-00940-w
14. Venkatachalam, S.: Integrating large language models with network optimization
for interactive and explainable supply chain planning: A real-world case study
(2025),https://arxiv.org/abs/2508.21622
15. Walker, C., Aslansefat, K., Akram, M.N., Papadopoulos, Y.: Raguard: A novel ap-
proach for in-context safe retrieval augmented generation for llms. arXiv preprint
arXiv:2509.03768 (2025). https://doi.org/10.48550/arXiv.2509.03768,https://
arxiv.org/abs/2509.03768
16. Walker, C., Rothon, C., Aslansefat, K., Papadopoulos, Y., Dethlefs, N.:
Safellm: Domain-specific safety monitoring for large language models: A case
study of offshore wind maintenance. arXiv preprint arXiv:2410.10852 (2024).
https://doi.org/10.48550/arXiv.2410.10852,https://arxiv.org/abs/2410.10852
17. Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W.W., Salakhutdinov, R., Manning,
C.D.: Hotpotqa: A dataset for diverse, explainable multi-hop question answering.
arXiv:1809.09600 [cs] (09 2018),https://arxiv.org/abs/1809.09600
18. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., Cao,
Y.: React: Synergizing reasoning and acting in language models (03 2023).
https://doi.org/10.48550/arXiv.2210.03629,https://arxiv.org/abs/2210.03629
19. Yuan, X., Huang, Q., Guo, M., Ma, H., Xu, M., Liu, Z., He, X.: Towards next-
generation intelligent maintenance: Collaborative fusion of large and small models
(2025),https://arxiv.org/abs/2506.05854
20. Zhang,L.,Zhao,C.,Gao,Q.,Zhao,X.,Bai,G.,Lv,J.:Dschellm:Enablingdynamic
scheduling through a fine-tuned dual-system large language model (2026),https:
//arxiv.org/abs/2601.09100
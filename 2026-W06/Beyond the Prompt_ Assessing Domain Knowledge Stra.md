# Beyond the Prompt: Assessing Domain Knowledge Strategies for High-Dimensional LLM Optimization in Software Engineering

**Authors**: Srinath Srinivasan, Tim Menzies

**Published**: 2026-02-02 20:07:15

**PDF URL**: [https://arxiv.org/pdf/2602.02752v1](https://arxiv.org/pdf/2602.02752v1)

## Abstract
Background/Context: Large Language Models (LLMs) demonstrate strong performance on low-dimensional software engineering optimization tasks ($\le$11 features) but consistently underperform on high-dimensional problems where Bayesian methods dominate. A fundamental gap exists in understanding how systematic integration of domain knowledge (whether from humans or automated reasoning) can bridge this divide.
  Objective/Aim: We compare human versus artificial intelligence strategies for generating domain knowledge. We systematically evaluate four distinct architectures to determine if structured knowledge integration enables LLMs to generate effective warm starts for high-dimensional optimization.
  Method: We evaluate four approaches on MOOT datasets stratified by dimensionality: (1) Human-in-the-Loop Domain Knowledge Prompting (H-DKP), utilizing asynchronous expert feedback loops; (2) Adaptive Multi-Stage Prompting (AMP), implementing sequential constraint identification and validation; (3) Dimension-Aware Progressive Refinement (DAPR), conducting optimization in progressively expanding feature subspaces; and (4) Hybrid Knowledge-Model Approach (HKMA), synthesizing statistical scouting (TPE) with RAG-enhanced prompting. Performance is quantified via Chebyshev distance to optimal solutions and ranked using Scott-Knott clustering against an established baseline for LLM generated warm starts.
  Note that all human studies conducted as part of this study will comply with the policies of our local Institutional Review Board.

## Full Text


<!-- PDF content starts -->

Beyond the Prompt: Assessing Domain Knowledge Strategies for
High-Dimensional LLM Optimization in Software Engineering
Srinath Srinivasan
ssrini27@ncsu.edu
North Carolina State University
Raleigh, North Carolina, USATim Menzies
timm@ieee.org
North Carolina State University
Raleigh, North Carolina, USA
Abstract
Background/Context: Large Language Models (LLMs) demon-
strate strong performance on low-dimensional software engineer-
ing optimization tasks ( ≤11 features) but consistently underperform
on high-dimensional problems where Bayesian methods dominate.
A fundamental gap exists in understanding how systematic inte-
gration of domain knowledge (whether from humans or automated
reasoning) can bridge this divide.
Objective/Aim: We compare human versus artificial intelli-
gence strategies for generating domain knowledge. We systemati-
cally evaluate four distinct architectures to determine if structured
knowledge integration enables LLMs to generate effective warm
starts for high-dimensional optimization.
Method: We evaluate four approaches on MOOT1datasets strat-
ified by dimensionality: (1)Human-in-the-Loop Domain Knowledge
Prompting (H-DKP), utilizing asynchronous expert feedback loops;
(2)Adaptive Multi-Stage Prompting (AMP), implementing sequen-
tial constraint identification and validation; (3)Dimension-Aware
Progressive Refinement (DAPR), conducting optimization in pro-
gressively expanding feature subspaces; and (4)Hybrid Knowledge-
Model Approach (HKMA), synthesizing statistical scouting (TPE)
with RAG-enhanced prompting. Performance is quantified via Cheby-
shev distance to optimal solutions and ranked using Scott-Knott
clustering against an established baseline for LLM generated warm
starts.
Note that all human studies conducted as part of this study will
comply with the policies of our local Institutional Review Board.
Keywords
Optimization, warm starts, LLM, active learning, configuration
1 Introduction
Software engineering optimization requires balancing competing
objectives (e.g. runtime versus memory, quality versus cost, speed
versus energy consumption). Active learning addresses these prob-
lems efficiently by using models to select the most informative
examples to label, achieving good results with minimal data [40].
However, to learn good optimizations from data, some ground
truth data must initially be available. In the domain of software
engineering, obtaining these labels is often prohibitively expensive
or time-consuming [43, 24].
Large Language Models (LLMs) offer a potential solution for
generating these initial “warm start” samples without the cost of
execution. Yet, their reliability in this domain is debated; Treude
et al. [3] conclude that LLMs function best as assistive rather than
1Available at: tiny.cc/mootauthoritative agents in software tasks. Conversely, insights from
Nair et al. [34] on “bad learners” suggest that even imperfect mod-
els can successfully guide optimization if they can suggest valid
partial rankings. This raises a critical question: can we leverage the
assistive nature of LLMs to act as heuristic guides for optimization,
even if they lack authoritative ground truth?
Recent work [40] studying dozens of SE optimization problems
demonstrates that LLMs can indeed generate effective warm starts,
reducing labeling requirements from hundreds to dozens of ex-
amples. On low-dimensional problems ( <6features), LLM-based
warm starts achieve top performance 100% of the time. On medium-
dimensional problems (6-11 features), they succeed 50% of the time.
However, performance collapses for high-dimensional problems
(>11features), where traditional Bayesian methods like Gauss-
ian Process Models remain superior. This dimensional barrier is
problematic: many real-world SE optimization tasks in the MOOT
[29] repository involve more than 11 features, including software
configuration (18-38 features), process modeling (23 features), and
hyperparameter tuning (14-38 features).
The cause of this failure likely stems from training data limita-
tions. LLMs excel on “common” problems well-represented in their
training corpus but struggle with specialized, high-dimensional
SE tasks that lack abundant public examples. Cloud configuration,
aerospace software processes, and domain-specific optimizations
rarely have publicly documented optimal solutions. Without suffi-
cient training data, LLMs cannot learn the complex feature inter-
actions that characterize high-dimensional spaces. This problem
will intensify: recent projections suggest LLMs will exhaust avail-
able textual training data by 2028, making data-scarce optimization
domains increasingly important [44]. No prior work has systemati-
cally explored whether injecting domain-specific knowledge can
overcome these limitations.
We propose four complementary approaches to incorporate do-
main knowledge into LLM warm starts, moving from human-centric
to fully automated strategies:
(1)Human-in-the-Loop Domain Knowledge Prompting
(H-DKP):We leverage human experts to verify and re-
fine constraints in an iterative loop, augmenting prompts
with structured feature relationships and heuristics at three
levels of detail.
(2)Adaptive Multi-Stage Prompting (AMP):We decompose
the generation process into sequential reasoning stages:
analysis, constraint identification, generation, and valida-
tion. We allow the LLM to generate its own knowledge
before proposing solutions.
(3)Dimension-Aware Progressive Refinement (DAPR):
We address high-dimensional complexity by optimizing
1arXiv:2602.02752v1  [cs.SE]  2 Feb 2026

Srinivasan et al.
in reduced feature spaces (identified via statistical rank-
ing) and progressively expanding dimensions, effectively
guiding the LLM through the search space.
(4)Hybrid Knowledge-Model Approach (HKMA):We em-
ploy Retrieval-Augmented Generation (RAG) [25] com-
bined with rapid statistical exploration using Tree of Parzen
Estimators (TPE) [5] to leverage both data-driven patterns
and semantic understanding from online texts and docu-
mentation.
We will evaluate all approaches on MOOT multi-objective SE
optimization tasks stratified by dimensionality.
2 Related Work
Modern software engineering is characterized by massive param-
eter search spaces that must be configured [7]. Xu et al. [47] and
Van Aken et al. [4] report that as systems mature, these spaces
explode exponentially. Consequently, users often ignore configu-
ration options or rely on obsolete defaults, which can lead to sig-
nificant performance degradations (up to 480 ×in some industrial
cases [13, 18, 20]). To manage this complexity, researchers employ
configuration optimization algorithms to balance competing con-
straints [17]. In the realm of software analytics, such optimization is
critical; studies by Fu et al. [14] and Agrawal et al. [1] demonstrate
that optimizing learners (e.g., finding hyperparameter settings) can
fundamentally alter experimental conclusions, turning a "worst"
performing algorithm into the "best" [42, 2, 49, 48].
Despite the necessity of optimization, practical application is
hindered by the high cost of data collection. Exploring configuration
landscapes for systems like x264 can require thousands of hours
of compile time [43], necessitating low-resource approaches that
operate within strict budgets of fewer than 50 evaluations [35, 9]. To
maximize efficiency within these limits, researchers utilize iterative
refinement algorithms [30, 52] and Active Learning [38]. However,
these methods are sensitive to initialization; poor "cold starts" waste
the limited labeling budget. The standard solution has historically
been "warm starting" the optimization using prior knowledge from
Subject Matter Experts (SMEs) [16, 26, 50].
While effective, reliance on human SMEs is not scalable across
the diverse ecosystem of modern software libraries. Recent work
attempted to use Large Language Models (LLMs) as automated
proxies for SMEs to generate these warm starts [40]. However, these
initial studies reveal a critical "dimensional barrier": while LLMs
perform well on simple, low-dimensional tasks, their performance
collapses on high-dimensional, multi-objective tabular data ( >11
features), often performing worse than random sampling.
3 Research Questions
•RQ1 (Comparative Efficacy):Which domain knowledge
integration strategy (HDKP, AMP, DAPR, HKMA) yields the
highest quality warm starts compared to standard baselines
across the MOOT corpus?
•RQ2 (The Dimensional Barrier):How does the effective-
ness of each strategy vary across dimensionality tiers?
•RQ3 (The Human Factor):Does human-in-the-loop feed-
back (H-DKP) provide statistically significant performance
gains over the fully automated methods?•RQ4 (Cost-Benefit Analysis):What are the quantitative
trade-offs between solution quality improvements and com-
putational overhead for each approach?
•RQ5 (Knowledge Attribution):Which specific categories
of domain knowledge (structural constraints, feature corre-
lations, heuristics, or statistical priors) contribute most to
performance improvements?
4 Data Sources & Collection
4.1 The MOOT Repository
To evaluate our hypotheses, we utilize MOOT, a curated repository
of software engineering optimization datasets. These datasets come
from papers published in top SE venues such as the International
Conference on Software Engineering [10, 45, 31, 15], Foundations
of SE (FSE) conference [34, 21] IEEE Trans. SE [11, 46, 24, 6, 23],
the Information Software Technology journal [8, 14], Empirical
Softw. Eng. [19, 36, 22], Mining Software Repositories [32], IEEE
Access [27], ACM Trans. SE Methodologies [28] and the Automated
Software Engineering Journal [33]. The repository currently houses
over 120 datasets spanning diverse domains, including real-world
system traces and software process simulations. Table. 1 provides a
summary of the datasets in MOOT.
4.1.1 Dataset Selection and Stratification.From this corpus, we
curate a representative subset to form our experimental testbed.
We enforce a selection criterion ofat least 10 datasets per com-
plexity tierto ensure generalizability across the dimensionality
spectrum in accordance with prior work [40]:
•Low Dimensionality ( <6features):Simple configura-
tion tasks where LLMs historically perform well.
•Medium Dimensionality (6 −11features):Intermediate
complexity tasks.
•High Dimensionality ( >11features):Complex, sparse
landscapes where prior work indicates LLMs fail.
This selection strategy is designed to balance statistical power
with economic constraints. Our study requires four distinct knowl-
edge integration methods (plus baselines) executed 20 independent
trials across datasets. Due to costs associated with high-volume API
access for state-of-the-art LLMs, analyzing the entire repository is
infeasible. By restricting our testbed to a stratified subset, we ensure
sufficient statistical power to detect the "dimensional barrier" while
managing API costs.
The final column of Table 1 indicates the feasibility of recruit-
ment for the H-DKP protocol. Datasets marked with a ✓indicate
domains where Subject Matter Experts (SMEs) are accessible either
locally at the NC State campus or through our direct academic
network. This accessibility filter will serve as a secondary criterion
when choosing datasets from MOOT for H-DKP.
4.2 Participants: Subject Matter Experts (SMEs)
We will recruit human experts following approval from the North
Carolina State University Institutional Review Board (IRB). We clas-
sify this study as “Minimal Risk” regarding human subjects, as data
collection is limited to professional email correspondence regarding
2

Beyond the Prompt: Assessing Domain Knowledge Strategies for High-Dimensional LLM Optimization in Software Engineering
Table 1: Summary of datasets in the MOOT repository. “x/y” denotes the number of independent and dependent attributes.
# Datasets Dataset Type File Names Primary Objective x/y # Rows Experts
25Specific Software
ConfigurationsSS-A to SS-X, billing10k Optimize software system settings 3-88/2-3 197–86,059
12PromiseTune Software
Configurations7z, BDBC, HSQLDB, LLVM, PostgreSQL,
dconvert, deeparch, exastencils, javagc,
redis, storm, x264Software performance optimization 9-35/1 864-
166,975✓
1 Cloud HSMGP num Hazardous Software, Management Program data 14/1 3,457
1 Cloud Apache AllMeasurements Apache server performance optimization 9/1 192
1 Cloud SQL AllMeasurements SQL database tuning 39/1 4,654
1 Cloud X264 AllMeasurements Video encoding optimization 16/1 1,153
7 Cloud (rs—sol—wc)* misc configuration tasks 3-6/1 196–3,840
35 Software Project Health Health-ClosedIssues, -PRs, -Commits Predict project health and developer activity 5/2-3 10,001✓
3 Scrum Scrum1k, Scrum10k, Scrum100k Configurations of the scrum feature model 124/3 1,001–100,001 ✓
8 Feature Models FFM-*, FM-* Optimize number of variables, constraints and
Clause/Constraint ratio128-1,044/3 10,001
1 Software Process Model nasa93dem Optimize effort, defects, time and LOC 24/3 93✓
1 Software Process Model COC1000 Optimize risk, effort, analyst experience, etc 20/5 1,001✓
4 Software Process Model POM3 (A–D) Balancing idle rates, completion rates and cost 9/3 501–20,001✓
4 Software Process Model XOMO (Flight, Ground, OSP) Optimizing risk, effort, defects, and time 27/4 10,001✓
3 Miscellaneous auto93, Car_price, Wine_quality Miscellaneous 5-38/2-5 205–1,600✓
4 Behavioral all_players, student_dropout,
HR-employeeAttrition, player_statisticsAnalyze and predict behavioral patterns 26-55/1-3 82–17,738✓
4 Financial BankChurners, home_data, Loan,
Telco-ChurnFinancial analysis and prediction 19-77/2-5 1,460–20,000
3 Human Health Data COVID19, Life_Expectancy,
hospital_ReadmissionsHealth-related analysis and prediction 20-64/1-3 2,938–25,000 ✓
2 Reinforcement Learning A2C_Acrobot, A2C_CartPole Reinforcement learning tasks 9-11/3-4 224–318✓
5 Sales accessories, dress-up, Market-
ing_Analytics, socks, wallpaperSales analysis and prediction 14-31/1-8 247–2,206
2 Software testing test120, test600 Optimize the class 9/1 5,161✓
127Total
technical domain knowledge, with no collection of sensitive per-
sonal data, health information, or identifiers linked to vulnerable
populations.
4.2.1 Recruitment Channels and Criteria.We leverage the exten-
sive industry connections of the NC State Computer Science De-
partment, where many graduates have remained in the Research
Triangle Park area for over a decade. Through the departmental
alumni group, advisory board, and faculty contacts, we will con-
duct a targeted email campaign to identify qualified experts. Based
on prior experience with similar SE surveys, we anticipate a re-
sponse rate of 2-10%. To achieve our target of at least one expert per
dataset category, we plan to issue approximately 500 recruitment
requests. We will conduct a pilot study with two experts local to
the university to calibrate expert time commitment before wider
recruitment.
We define a qualified Subject Matter Expert (SME) based on
meeting at least one of the following criteria:
•Primary Authorship:Authors of the original research
papers contributing datasets to the MOOT repository.
•Project Maintenance:Active maintainers or core contrib-
utors to the specific software systems under consideration.
Gold Standard Vetting:For experts who are not primary
authors or known maintainers, we will administer a brief
"Gold Standard" questionnaire containing 7 multiple choice
questions about the domain. These questions will focus
on specific, non-obvious domain constraints found in thedocumentation. Only participants who correctly answer
these control questions will be admitted to the study.
4.2.2 Contingency for Dropouts and Asynchronous Responses.We
recognize that expert availability is variable. Therefore, we adopt a
flexible Asynchronous Iterative Protocol:
•Non-Consecutive Iterations:The experimental design
requires𝑇logical feedback iterations, not 𝑇consecutive
calendar days. Experts may respond at their own pace; the
LLM state remains frozen until feedback is received.
•Minimum Viable Threshold ( 𝑇𝑚𝑖𝑛):To ensure statistical
validity, we establish a minimum threshold of 𝑇𝑚𝑖𝑛=5
iterations. Datasets where experts provide fewer than 5
feedback cycles will be excluded from the H-DKP specific
analysis.
•Variable𝑇Analysis:We anticipate that the final number
of iterations 𝑇will vary across datasets (e.g.,5 ≤𝑇≤
10). We pre-register an analysis to correlate the number
of expert feedback rounds ( 𝑇) with the final optimization
improvement ( ΔChebyshev distance). This allows us to
quantify the marginal utility of human effort (e.g.,"Does
performance plateau after𝑇=7?").
5 Analysis Plan & Evaluation Criteria
5.1 Baseline Methods
We will compare our methods against the following warm start
baselines:
3

Srinivasan et al.
•Random Sampling (Random):A naive baseline that se-
lects samples uniformly at random from the search space.
This establishes the lower bound of performance.
•Gaussian Process Model (UCB_GPM):The current state-
of-the-art for high-dimensional optimization in this domain.
We use a Gaussian Process regressor with the Upper Confi-
dence Bound (UCB) [41] acquisition function.
•Standard LLM Warm Start (BS_LLM):Prior state-of-
the-art few-shot prompting approach where the LLM is
provided with 4 randomly selected examples (labeled as
"Best" or "Rest") and basic feature metadata (name, type,
median) [40]. This serves as our primary control to measure
improvements from domain knowledge integration.
5.2 Primary Metric: Chebyshev Distance
We quantify the quality of generated warm starts using the Cheby-
shev distance to the optimal configuration as done in prior work
[40]. Since objectives in software optimization often have vastly
different scales, we first normalize all objective values 𝑦𝑖to the
range[0,1]. The Chebyshev distance 𝐷for a candidate solution 𝑥
is defined as:
𝐷(𝑥)=max
𝑖∈{1,...,𝑚}(|𝑓𝑖(𝑥)−𝑧∗
𝑖|)(1)
Where𝑚is the number of objectives and 𝑧∗
𝑖is the ideal value (0
for minimization) for the 𝑖-th objective. A lower Chebyshev distance
indicates a solution closer to the theoretical optimum. We perform
our experiments for 20 trials across each method (in H-DKP, the
final prompt is used 20 times to generate warm starts). For each
trial, we report the minimum Chebyshev distance achieved among
the generated warm start examples.
5.3 Secondary Metrics
5.3.1 Generated Example Diversity.To ensure the LLM is not sim-
ply generating identical variations of one good example, we mea-
sure the diversity of the generated set E𝑔𝑒𝑛. We calculate theAverage
Pairwise Euclidean Distancebetween all generated vectors in the
feature space. Higher diversity implies better exploration of the
search space.
5.3.2 Computational Cost (API Tokens).We track the economic
feasibility of each method by logging the total number of input
and output tokens consumed per trial. We report the average cost
per successful warm start. This allows us to analyze the tradeoff
between performance gains and the increased inference cost relative
to the single-shot baseline.
5.4 Statistical Analysis: Scott-Knott and Effect
Size
To determine if our proposed methods provide a statistically signif-
icant improvement over the baseline, we employ the Scott-Knott
Effect Size Difference (ESD) test.
5.4.1 Scott-Knott Clustering Algorithm.The Scott-Knott algorithm
[39] recursively partitions the set of treatment means into two
subsets to maximize the difference between groups. The splitting
criterion maximizes theBetween-Group Sum of Squares( 𝐵0). For aset of treatments with sizes 𝑁1and𝑁2and sums of responses 𝑇1
and𝑇 2, the algorithm seeks a partition that maximizes:
𝐵0=𝑇2
1
𝑁1+𝑇2
2
𝑁2−(𝑇1+𝑇2)2
𝑁1+𝑁 2(2)
The algorithm follows these steps:
(1)Sort:Order the treatment distributions by their median
Chebyshev distance.
(2)Split:Identify the partition point that yields the maximum
𝐵0.
(3)Significance Test:Check if the split is statistically signifi-
cant using a bootstrap sampling method (to avoid assump-
tions of normality).
(4)Effect Size Check (ESD):Even if significant, the split is
rejected if the magnitude of the difference is negligible
(Cliff’s Delta <0.147). This ensures distinct ranks represent
practically meaningful differences.
(5)Recurse:If the split is valid, recursively apply the proce-
dure to each subgroup; otherwise, terminate and group the
treatments into a single rank.
5.4.2 Effect Size (Cliff’s Delta).To ensure that observed differences
are not just statistically significant but practically meaningful, we
calculate Cliff’s Delta [12] ( 𝛿), a non-parametric effect size measure.
We interpret the magnitude of difference between the proposed
method and the baseline as follows:
• |𝛿|<0.147: Negligible
•0.147≤|𝛿|<0.33: Small
•0.33≤|𝛿|<0.474: Medium
• |𝛿|≥0.474: Large
We consider a hypothesis validated only if the proposed method
achieves a better Scott-Knott rankandshows at least a "Small"
effect size improvement over the BS_LLM baseline.
6 Methods & Execution Plan
6.1 Human-in-the-Loop Domain Knowledge
Prompting (H-DKP)
6.1.1 Overview and Rationale.While LLMs can parse documenta-
tion, they lack the tacit knowledge possessed by domain experts
(unwritten rules, edge cases, and intuition gained through experi-
ence). We proposeHuman-in-the-Loop Domain Knowledge
Prompting (H-DKP), a methodology to extract this tacit knowl-
edge through a structured, asynchronous dialogue with human ex-
perts. Unlike static few-shot prompting, H-DKP treats the prompt
construction as an iterative software design process, evolving the
LLM’s "mental model" of the domain over a fixed time window. The
process is outlined in Algorithm 1.
6.1.2 Expert Recruitment and Elicitation Protocol.To execute H-
DKP, we will identify and recruit domain experts for the datasets in
the MOOT repository as outlined previously. To minimize expert
cognitive load, we utilize the Recognition over Recall principle. Ex-
perts will not be asked to write prompts. Instead, they will critique
the LLM’s outputs.
6.1.3 The 10-Day Iterative Refinement Sprint.We employ anAsyn-
chronous Iterative Refinement (AIR)protocol. For a period of
4

Beyond the Prompt: Assessing Domain Knowledge Strategies for High-Dimensional LLM Optimization in Software Engineering
10 consecutive days, each expert engages in a daily feedback loop
with the LLM.
•Day 1 (Initialization):The LLM generates an initial “Belief
State” (hypothesized constraints and feature relationships)
based on documentation about the data, variable names and
problem objectives.This is sent to the expert for a baseline
validity check (Valid/Invalid/Modify).
•Days 2-9 (The Feedback Loop):Each day, we run the
current Prompt State to generate warm start configurations.
We identify the “Most Confusing Failure”—a configuration
the LLM predicted would be optimal but which performed
poorly in reality. The expert receives a structured email
containing:
(1) The current rule set the LLM is following.
(2)The specific failure case (e.g.,"The model set ‘threads=100‘
expecting high throughput, but latency spiked. ").
(3)A single question:"What domain rule is the model miss-
ing that explains this failure?"
The expert’s email reply is parsed and appended to the
prompt context for the next day’s run.
•Day 10 (Finalization):The accumulated knowledge base
is frozen and used for the final experimental evaluation.
Algorithm 1H-DKP via Asynchronous Expert Feedback
Require:DatasetD, Human ExpertH
Require:Duration𝑇=10days
Ensure:Optimized Knowledge Base𝐾 𝑓 𝑖𝑛𝑎𝑙 and Warm StartsE
1:Day 1: Bootstrapping
2:𝐾 1←LLM(Docs,"Propose constraints")
3:Send Email(H, "Verify these baseline constraints:𝐾 1")
4:𝐾 1←Update(𝐾 1,EmailReply(H))
5:for𝑡=2to𝑇do
6:{Generate candidates using current knowledge}
7:E 𝑡←LLM(prompt=𝐾 𝑡−1)
8:EvaluateE 𝑡against ground truthD
9:{Identify knowledge gap}
10:𝑒 𝑓 𝑎𝑖𝑙←FindMaxError(E 𝑡){High confidence, low reward}
11:{Daily Asynchronous Query}
12:Query←"Model believed𝑒 𝑓 𝑎𝑖𝑙was optimal. Why did it fail?"
13:Send Email(H, Query)
14:{Wait for asynchronous reply}
15:Feedback 𝑡←EmailReply(H)
16:𝐾 𝑡←𝐾 𝑡−1∪Feedback 𝑡
17:end for
18:Final Evaluation
19:E 𝑓 𝑖𝑛𝑎𝑙←LLM(prompt=𝐾 10)
20:returnE 𝑓 𝑖𝑛𝑎𝑙
6.1.4 H-DKP analysis consideration.Given the nature of human
studies and mercurial nature of expert recruitment, we do not an-
ticipate our analysis with H-DKP to span across all datasets used
in the other discussed methods. We will restrict our comparison of
H-DKP and other algorithms only for datasets where the study is
completed with an expert without dropout.6.2 Adaptive Multi-Stage Prompting (AMP)
6.2.1 Overview and Rationale.Standard "single-shot" prompting
treats optimization as a pattern-matching task, asking the LLM to
generate solutions immediately after seeing a few examples. We
proposeAdaptive Multi-Stage Prompting (AMP), a sequential
reasoning pipeline that forces the LLM to explicitly articulate its
"mental model" of the optimization landscape before generating
configuration values. By separating analysis from generation, AMP
aims to reduce the logical inconsistencies and constraint violations
common in single-shot warm starts.
6.2.2 Stage 1: Analysis.The first stage functions as a filter to distin-
guish signal from noise. The LLM is provided with dataset metadata
(feature names, types, ranges) and the initial few-shot examples. It
is prompted to output a structured analysis identifying:
(1)Feature Ranking:A prioritized list of the 3-5 most influ-
ential features driving the objective values.
(2)Tradeoff Identification:Explicit notes on conflicting ob-
jectives observed in the few-shot examples.
(3)Directionality:The hypothesized direction of improve-
ment for continuous variables.
6.2.3 Stage 2: Constraint Discovery.Using the analysis from Stage
1, the LLM infers explicit boundaries for valid configurations. This
stage distinguishes between:
•Hard Constraints:Inviolable rules based on physical or
logical limits (e.g.,"parallel_threads cannot exceed available
cpu_cores").
•Soft Constraints:Heuristic preferences that generally lead
to better outcomes but may be violated for exploration.
We expect that explicitly generating these rules will reduce the
search space for the subsequent generation step and provide a
"rulebook" for the validation stage.
6.2.4 Stage 3: Constrained Generation.The third stage performs
the actual warm start generation. Unlike the baseline single-shot
approach, the prompt for this stage is dynamically constructed to
include the prioritized Feature List (from Stage 1) and the Validation
Rules (from Stage 2) as strict instructions. The LLM is tasked to
generate configurations that optimize the identified key features
while strictly adhering to the discovered hard constraints.
6.2.5 Stage 4: Self-Validation.In the final stage, the LLM acts as
a critic. It reviews its own generated configurations against the
constraint set defined in Stage 2.
•Verification:Each generated example is checked for strict
logical consistency based on the generated constraints.
•Refinement:If a configuration violates a hard constraint,
the model is prompted to revise the specific value while
preserving the rest of the configuration.
6.2.6 AMP Ablation study.To study the effect of the different strate-
gies for prompting the LLM to decipher knowledge we will study the
various stages in 3 experimental conditions: Condition 1 (AMP-2):
Analysis + Generation, Condition 2 (AMP-3): Analysis + Constraints
+ Generation & Condition 3 (AMP-4): Full 4-stage pipeline.
5

Srinivasan et al.
6.3 Dimension-Aware Progressive Refinement
(DAPR)
6.3.1 Feature Importance Ranking.To minimize possible bias and
failures of specific methods, we propose feature importance calcula-
tion through 3 different statistical methods - Spearman coefficient,
mutual information & feature importance calculated through ran-
dom forest. The calculated values through these methods will then
be normalized and averaged to calculate the final feature importance
score. It is to be noted that this calculation will only be performed
with the few-shot samples random chosen as knowledge for the
LLM.
6.3.2 Progressive Expansion Algorithm.The progressive refinement
portion of DAPR, as illustrated in Algorithm. 2, begins by initializing
the current feature set 𝜒𝑐𝑢𝑟with the top 𝑘most important features
(line 2). At each iteration, the algorithm projects four random ex-
amples onto the current reduced subspace (line 4) and prompts
the LLM to generate optimized configurations using only these
features (line 5). Generated examples are mapped to their nearest
neighbors (following the methodology in prior work [37, 51]) in the
full dataset to obtain labels (lines 6-8), and the best configuration
is tracked across iterations (lines 9-12). The feature space is then
progressively expanded by adding the next 𝑠most important fea-
tures (lines 13-14), with newly added features anchored to values
from the current best configuration to maintain continuity (lines
15-17). This process repeats until all 𝑛features are included. Finally,
the algorithm generates warm start examples in the full dimen-
sional space, using the best configuration found during progressive
refinement as an anchor point (lines 18-20).
Algorithm 2Dimension-Aware Progressive Refinement (DAPR)
Require:DatasetDwith featuresX={𝑥 1,...,𝑥 𝑛}
Require:Ranked featuresF=[𝑓 1,...,𝑓 𝑛], initial size𝑘, step𝑠
Ensure:Warm start examples in full𝑛-dimensional space
1:X cur←{𝑓 1,...,𝑓 𝑘}; best←null
2:while|X cur|<𝑛do
3:E fs←project 4 random samples ontoX cur
4:E gen←LLM(E fs,"Optimize onX cur")
5:for𝑒∈E gendo
6:𝑒 full←nearest neighbor of𝑒inD
7:Evaluate Chebyshev(𝑒 full)
8:end for
9:𝑒∗←arg min 𝑒∈E genChebyshev(𝑒 full)
10:ifbest=null or Chebyshev(𝑒∗)<Chebyshev(best)then
11:best←𝑒∗
12:end if
13:𝑑←|X cur|;X new←{𝑓 𝑑+1,...,𝑓 min(𝑑+𝑠,𝑛)}
14:X cur←X cur∪X new
15:for𝑥 𝑖∈X newdo
16: Anchor𝑥𝑖to value from best(ormedian(𝑥 𝑖)if unavailable)
17:end for
18:end while
19:E fs←4 random samples in full space
20:E final←LLM(E fs,anchored to best)
21:returnE final6.4 Hybrid Knowledge-Model Approach
(HKMA)
6.4.1 Overview and Rationale.We posit that LLMs and statistical
models suffer from orthogonal blind spots: LLMs possess seman-
tic understanding but hallucinate quantitative relationships, while
Bayesian models (like TPE) identify quantitative patterns but lack
semantic causal reasoning. HKMA bridges this gap using a light-
weight statistical model to "scout" the terrain and identify empirical
priors, which are then fed to the LLM to ground its generation in
observed reality.
6.4.2 Phase 1: Statistical Scouting.Before invoking the LLM, we
perform a rapid, low-budget exploration using the Tree-structured
Parzen Estimator (TPE). We allocate a small "scouting budget"
(𝐵𝑠𝑐𝑜𝑢𝑡=10) to perform a rapid exploration of the search space us-
ing TPE. We specifically utilize the TPE exploit capability to quickly
identify high-performing regions versus low-performing ones. By
comparing the distribution of the top-performing configurations
(𝑆𝑏𝑒𝑠𝑡) against the remaining samples ( 𝑆𝑟𝑒𝑠𝑡), we extract Empirical
Priors. These priors are formalized as natural language descriptions
of observed phenomena, such as directional trends and boundary
conditions.
6.4.3 Phase 2: Retrieval-Augmented Synthesis.We employ Retrieval-
Augmented Generation (RAG) to provide semantic context for the
observed statistical patterns. We index the domain documentation
and academic literature collected from the MOOT repository into a
vector store.Using the extracted Empirical Priors as search queries,
we retrieve relevant documentation that explains the physical or
logical mechanisms behind the statistics (e.g., querying "Why does
high buffer size improve throughput?" to retrieve specific mem-
ory management docs). The final prompt to the LLM constructs a
synthesis task: the model is provided with the Empirical Evidence
and the Semantic Explanation, and is tasked with generating warm
start configurations that satisfy both.
6.4.4 Study design and ablation.We plan to ablate the RAG and
scouting phases by using them separately and together to under-
stand the implication of these treatments on the quality of warm
starts generated.
References
[1] A. Agrawal, W. Fu, D. Chen, X. Shen, and T. Menzies. 2019. How to “dodge”
complex software analytics.IEEE Trans Softw Eng, 47, 10, 2182–2194.
[2] Amritanshu Agrawal and Tim Menzies. 2018. Is “better data” better than “bet-
ter data miners”?: on the benefits of tuning smote for defect prediction. In
Proceedings of the 40th International Conference on Software Engineering. ACM,
(May 2018), 1050–1061. doi:10.1145/3180155.3180197.
[3] Toufique Ahmed, Premkumar Devanbu, Christoph Treude, and Michael Pradel.
2025. Can llms replace manual annotation of software engineering artifacts?
(2025). https://arxiv.org/abs/2408.05534 arXiv: 2408.05534[cs.SE].
[4] Dana Van Aken, Andrew Pavlo, Geoffrey J. Gordon, and Bohan Zhang. 2017.
Automatic database management system tuning through large-scale machine
learning. InSIGMOD.
[5] James Bergstra, Rémi Bardenet, Yoshua Bengio, and Balázs Kégl. 2011. Algo-
rithms for hyper-parameter optimization. InAdvances in Neural Information
Processing Systems. J. Shawe-Taylor, R. Zemel, P. Bartlett, F. Pereira, and K.Q.
Weinberger, (Eds.) Vol. 24.
[6] Jianfeng Chen, Vivek Nair, Rahul Krishna, and Tim Menzies. 2019. “sampling”
as a baseline optimizer for search-based software engineering.IEEE Trans.
Softw. Eng., 45, 597–614.
6

Beyond the Prompt: Assessing Domain Knowledge Strategies for High-Dimensional LLM Optimization in Software Engineering
[7] Jianfeng Chen, Vivek Nair, Rahul Krishna, and Tim Menzies. 2018. “sampling” as
a baseline optimizer for search-based software engineering.IEEE Transactions
on Software Engineering, 45, 6, 597–614.
[8] Jianfeng Chen, Vivek Nair, and Tim Menzies. 2018. Beyond evolutionary algo-
rithms for search-based software engineering.Inf. Softw. Technol.y, 95, 281–
294.
[9] Junjie Chen, Ningxin Xu, Peiqi Chen, and Hongyu Zhang. 2021. Efficient
compiler autotuning via bayesian optimization. In43rd IEEE/ACM Int. Conf.
Softw. Eng. ICSE 2021, Madrid, Spain, 22-30 May 2021, 1198–1209.
[10] Pengzhou Chen and Tao Chen. 2026. Promisetune: unveiling causally promising
and explainable configuration tuning. InProc. of the 48th IEEE/ACM Int. Conf.
Softw. Eng.
[11] Pengzhou Chen, Jingzhi Gong, and Tao Chen. 2025. Accuracy can lie: on the
impact of surrogate model in configuration tuning.IEEE Trans. Softw. Eng., 51,
2, 548–580.
[12] Norman Cliff. 1993. Dominance statistics: ordinal analyses to answer ordinal
questions.Psychological Bulletin, 114, 494–509.
[13] Yuanyuan Zhou et al. 2011. Understanding and detecting software configuration
errors.USENIX OSDI.
[14] Wei Fu, Tim Menzies, and Xipeng Shen. 2016. Tuning for software analytics: is
it really necessary?Inf. Softw. Technol., 76, 135–146.
[15] H. Ha and H. Zhang. 2019. Deepperf: performance prediction for configurable
software with deep sparse neural network. InICSE.
[16] Guy Hacohen, Avihu Dekel, and Daphna Weinshall. 2022. Active learning
on a budget: opposite strategies suit high and low budgets.arXiv preprint
arXiv:2202.02794.
[17] Mark Harman, S Afshin Mansouri, and Yuanyuan Zhang. 2012. Search-based
software engineering: trends, techniques and applications.ACM Computing
Surveys (CSUR), 45, 1, 11.
[18] Herodotos Herodotou and Shivnath Babu. 2011. Starfish: A self-tuning system
for big data analytics. InProceedings of the 5th Biennial Conference on Innovative
Data Systems Research (CIDR ’11), 261–272.
[19] Jeremy Hulse, Nasir U Eisty, and Tim Menzies. 2025. Shaky structures: the
wobbly world of causal graphs in software analytics.Empir. Softw. Eng.
[20] P. Jamshidi and G. Casale. 2016.Uncertainty-aware self-adaptation in cloud
computing. Ph.D. Dissertation. Imperial College London. Ph.D. dissertation pro-
posal. Cited by other works for the Storm configuration performance example.
[21] P. Jamshidi, M. Velez, C. Kästner, and N. Siegmund. 2018. Learning to sample:
exploiting similarities across environments to learn performance models for
configurable systems.FSE.
[22] G. Jianmei, Y. Dingyu, S. Norbert, A. Sven, S. Atrisha, V. Pavel, C. Krzysztof,
W. Andrzej, and Y. Huiqun. 2018. Data-efficient performance learning for
configurable systems.Empir. Softw. Eng.
[23] Joseph Krall, Tim Menzies, and Misty Davies. 2015. Gale: geometric active
learning for search-based software engineering.IEEE Trans. Softw. Eng., 41, 10,
1001–1018.
[24] Rahul Krishna, Vivek Nair, Pooyan Jamshidi, and Tim Menzies. 2020. Whence
to learn? transferring knowledge in configurable systems using beetle.TSE, 47,
12, 2956–2972.
[25] Patrick Lewis et al. 2020. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information processing systems, 33,
9459–9474.
[26] Tennison Liu, Nicolás Astorga, Nabeel Seedat, and Mihaela van der Schaar.
2024. Large language models to enhance bayesian optimization.arXiv preprint
arXiv:2402.03921.
[27] Andre Lustosa and Tim Menzies. 2024. Isneak: partial ordering as heuristics
for model- based reasoning in software engineering.IEEE Access, 12, 142915–
142929.
[28] Andre Lustosa and Tim Menzies. 2024. Learning from very little data: on the
value of landscape analysis.TOSEM, 33, 3, 1–22.
[29] Tim Menzies, Tao Chen, Yulong Ye, Kishan Kumar Ganguly, Amirali Rayegan,
Srinath Srinivasan, and Andre Lustosa. 2025. Moot: a repository of many multi-
objective optimization tasks. (2025). arXiv: 2511.16882[cs.SE].
[30] Wiem Mkaouer, Marouane Kessentini, Adnan Shaout, Patrice Koligheu, Slim
Bechikh, Kalyanmoy Deb, and Ali Ouni. 2015. Many-objective software remod-
ularization using nsga-iii.ACM Trans. Softw. Eng. Methodol., 24, 3, Article 17,
(May 2015), 45 pages. doi:10.1145/2729974.
[31] S. Mühlbauer, F. Sattler, C. Kaltenecker, J. Dorn, S. Apel, and N. Siegmund.
2023. Analysing the impact of workloads on modeling the performance of
configurable software systems. InICSE, 2085–2097.
[32] V. Nair, A. Agrawal, J. Chen, W. Fu, G. Mathew, T. Menzies, L. L. Minku, M.
Wagner, and Z. Yu. 2018. Data-driven search-based software engineering. In
MSR.
[33] Vivek Nair, Tim Menzies, Norbert Siegmund, and Sven Apel. 2018. Faster
discovery of faster system configurations with spectral learning.Autom. Softw.
Eng., 25, 247–277.
[34] Vivek Nair, Tim Menzies, Norbert Siegmund, and Sven Apel. 2017. Using bad
learners to find good configurations. InFSE, 257–267.[35] Vivek Nair, Zhe Yu, Tim Menzies, Norbert Siegmund, and Sven Apel. 2020.
Finding faster configurations using FLASH.IEEE Trans. Software Eng., 46, 7,
794–811.
[36] Kewen Peng, Christian Kaltenecker, Norbert Siegmund, Sven Apel, and Tim
Menzies. 2023. Veer: enhancing the interpretability of model-based optimiza-
tions.Empir. Softw. Eng., 28, 61.
[37] Florian Pfisterer, Lennart Schneider, Julia Moosbauer, Martin Binder, and Bernd
Bischl. 2022. Yahpo gym - an efficient multi-objective multi-fidelity benchmark
for hyperparameter optimization. In vol. 188. PMLR, (25–27 Jul 2022), 3/1–39.
[38] Amirali Rayegan and Tim Menzies. 2025. Minimal data, maximum clarity: a
heuristic for explaining optimization.arXiv preprint arXiv:2509.08667.
[39] A. J. Scott and M. Knott. 1974. A cluster analysis method for grouping means
in the analysis of variance.Biometrics, 30, 3, 507–512. Retrieved Jan. 13, 2026
from http://www.jstor.org/stable/2529204.
[40] Lohith Senthilkumar and Tim Menzies. 2024. Can large language models im-
prove se active learning via warm-starts? (2024). arXiv: 2501.00125[cs.SE].
[41] Niranjan Srinivas, Andreas Krause, Sham M. Kakade, and Matthias W. Seeger.
2012. Information-theoretic regret bounds for gaussian process optimization in
the bandit setting.IEEE Transactions on Information Theory, 58, 5, (May 2012),
3250–3265. doi:10.1109/tit.2011.2182033.
[42] Chakkrit Tantithamthavorn, Shane McIntosh, Ahmed E Hassan, and Kenichi
Matsumoto. 2016. Automated parameter optimization of classification tech-
niques for defect prediction models. InProc. of the 38th Int. Conf. Softw. Eng.
321–332.
[43] Pavel Valov, Jean-Christophe Petkovich, Jianmei Guo, Sebastian Fischmeister,
and Krzysztof Czarnecki. 2017. Transferring performance prediction models
across different hardware platforms. InProc. of the 8th ACM/SPEC on Int. Conf.
Perf. Eng.39–50.
[44] Pablo Villalobos, Anson Ho, Jaime Sevilla, Tamay Besiroglu, Lennart Heim,
and Marius Hobbhahn. 2024. Will we run out of data? limits of llm scaling
based on human-generated data. (2024). arXiv: 2211.04325[cs.LG].
[45] M. Weber, C. Kaltenecker, F. Sattler, S. Apel, and N. Siegmund. 2023. Twins or
false friends? A study on energy consumption and performance of configurable
software. InICSE, 2098–2110.
[46] Tianpei Xia, Rui Shu, Xipeng Shen, and Tim Menzies. 2020. Sequential model
optimization for software effort estimation.IEEE Trans. Softw. Eng., 48, 1994–
2009.
[47] Tianyin Xu, Long Jin, Xuepeng Fan, Yuanyuan Zhou, Shankar Pasupathy, and
Rukma Talwadker. 2015. Hey, you have given me too many knobs!: under-
standing and dealing with over-designed configuration in system software. In
FSE.
[48] Rahul Yedida, Hong Jin Kang, Huy Tu, Xueqi Yang, David Lo, and Tim Menzies.
2023. How to find actionable static analysis warnings.IEEE Trans. Softw. Eng.,
49, 2856–2872.
[49] Rahul Yedida and Tim Menzies. 2021. On the value of oversampling for deep
learning in software defect prediction.IEEE Trans. Softw. Eng., 48, 8, 3103–3116.
[50] Ofer Yehuda, Avihu Dekel, Guy Hacohen, and Daphna Weinshall. 2022. Active
learning through a covering lens.Advances in Neural Information Processing
Systems, 35, 22354–22367.
[51] Arber Zela, Julien Siems, Lucas Zimmer, Jovita Lukasik, Margret Keuper, and
Frank Hutter. 2022. Surrogate nas benchmarks: going beyond the limited search
spaces of tabular nas benchmarks. (2022). arXiv: 2008.09777[cs.LG].
[52] Guofu Zhang, Zhaopin Su, Miqing Li, Feng Yue, Jianguo Jiang, and Xin Yao. 2017.
Constraint handling in nsga-ii for solving optimal testing resource allocation
problems.IEEE Transactions on Reliability, 66, 4, 1193–1212.
7
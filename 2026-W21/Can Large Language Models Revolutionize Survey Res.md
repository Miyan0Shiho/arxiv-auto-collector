# Can Large Language Models Revolutionize Survey Research? Experiments with Disaster Preparedness Responses

**Authors**: Yan Wang, Ziyi Guo, Christopher McCarty

**Published**: 2026-05-19 00:58:36

**PDF URL**: [https://arxiv.org/pdf/2605.19229v1](https://arxiv.org/pdf/2605.19229v1)

## Abstract
Survey research faces mounting structural challenges: declining response rates, sample bias, block-wise missingness among at-risk respondents, and AI-assisted fraudulent completions in online panels. Large language models (LLMs) have been proposed as a remedy, yet rigorous evaluations across the full survey workflow remain scarce, particularly in disaster contexts where data quality matters most. We present and evaluate a five-stage framework for LLM integration covering questionnaire design, sample selection, pilot testing, missing-data imputation, and post-collection analysis, using the 2024 Hurricane Milton preparedness survey of Florida residents (n=946) as a shared empirical testbed. We introduce a Protection Motivation Theory (PMT)-constrained co-occurrence knowledge graph and develop seven LLM configurations spanning zero-shot inference, retrieval-augmented baselines, and novel theory-informed variants. Our proposed Anchored Marginal Theory-Informed LLM (A-TLM) outperforms all three classical imputation baselines (IPW/MI, MICE+PMM, missForest) on RMSE under disaster-relevant block-wise MNAR conditions (S4 RMSE 1.439 vs. 1.496 for the next-best), while achieving near-zero signed bias (-0.121) where the random-forest imputer produces the largest absolute bias (-0.631). Organizing retrieval around PMT causal structure and integrating all evidence in a single model call outperforms unstructured retrieval and staged sequential inference (MAE 0.993 vs. 1.097 for standard RAG). We document that near-zero aggregate bias can mask opposing subgroup errors and propose subgroup-stratified bias auditing as a reporting standard. A retrieval-constrained knowledge-graph chatbot demonstrates that hallucination is architecturally manageable through grounded refusal.

## Full Text


<!-- PDF content starts -->

Can Large Language Models Revolutionize Survey Research?
Experiments with Disaster Preparedness Responses
Yan Wanga∗, Ziyi Guob, Christopher McCartyc
aDept. of Urban and Regional Planning & Florida Institute for Built Environment Resilience, University of Florida, P.O. Box 115706,
Gainesville, FL 32611.yanw@ufl.edu*Corresponding author
bDept. of Urban and Regional Planning & Florida Institute for Built Environment Resilience, University of Florida, 1480 Inner Rd.,
Gainesville, FL 32611.ziyiguo@ufl.edu
cCollege of Liberal Arts and Sciences, Bureau of Economic and Business Research, University of Florida, Suite 150, 720 SW 2nd Ave.,
Gainesville, FL 32611.ufchris@ufl.edu
Abstract
Survey research faces mounting structural challenges: declining response rates, sample bias, block-wise missingness concentrated among
the most at-risk respondents, and a growing wave of AI-assisted fraudulent completions in online panels. Large language models (LLMs)
have been proposed as a remedy, yet rigorous empirical evaluations of their performance across the full survey workflow remain scarce,
particularly in disaster contexts where data quality and population coverage matter most. We present and evaluate a five-stage framework for
LLM integration in survey research covering questionnaire design, sample selection, pilot testing, missing-data imputation, and post-collection
analysis, using the 2024 Hurricane Milton preparedness survey of Florida residents ( n=946 ) as a shared empirical testbed throughout
(Wang et al., 2026). We introduce a Protection Motivation Theory (PMT)-constrained co-occurrence knowledge graph and use it to develop
seven LLM configurations spanning zero-shot inference, retrieval-augmented baselines, and novel theory-informed variants. Our proposed
Anchored Marginal Theory-Informed LLM (A-TLM) outperforms all three classical imputation baselines (IPW/MI, MICE+PMM, and
missForest) on root-mean-square error under disaster-relevant block-wise missing-not-at-random conditions (S4 RMSE 1.439 vs. 1.496
for the next-best method), while achieving near-zero overall signed bias ( −0.121 ) where the random-forest imputer produces the largest
absolute bias ( −0.631 ). Organizing retrieval around PMT’s causal structure and integrating all evidence in a single model call outperforms
both unstructured nearest-neighbor retrieval and staged sequential inference (Marginal-TLM combined MAE 0.993 vs. 1.097 for standard
RAG). We document that near-zero aggregate bias can mask opposing subgroup errors of substantial magnitude, with compound-vulnerable
respondents systematically under-predicted across LLM configurations, and propose subgroup-stratified bias auditing as a reporting standard
for policy-relevant LLM-augmented workflows. A retrieval-constrained knowledge-graph chatbot demonstrates that hallucination risk is
architecturally manageable through grounded refusal. Across all five stages, we identify specific boundaries on LLM utility and provide
reproducible per-stage scripts anchored to real disaster survey data.
Keywords:generative AI; natural language processing; surveys; retrieval-augmented generation; in-context learning; disaster response;
missing data imputation; Protection Motivation Theory.
1. Introduction
Survey research stands as a cornerstone of the social sci-
ences, gauging public attitudes, uncovering patterns in hu-
man behavior, and shaping policy. However, the field is con-
fronted by escalating methodological challenges, including
frequent low response rates, sample bias that fails to repre-
sent the target population, and the proliferation of disengaged
or fraudulent answers. These challenges have intensified as
survey modes have shifted from traditional telephone inter-
views to self-administered online panels (Dillman and Chris-
tian, 2005; Stern et al., 2014; Wang and Fan, 2004). Histor-
ically, approaches such as post-stratification weighting and
demographic imputation, often guided by benchmarks like the
American Community Survey (ACS), have aimed to preserve
representativeness and correct for nonresponse. Yet these
techniques are increasingly limited in their ability to addresspartial responses, shifting population dynamics, and complex
patterns of missingness.
Compounding these difficulties, reliance on commercial
survey panels introduces further opacity: recruitment meth-
ods and sample maintenance are often undisclosed, and re-
sulting panels can diverge substantially from their intended
populations (Bentley et al., 2020; Callegaro et al., 2014; Hays
et al., 2015). Reviews of consumer sentiment surveys have
highlighted how incentives and self-paced online panels can
unintentionally inflate results and introduce quality risks that
are difficult to trace or correct (Hacıo ˘glu Hoke et al., 2025).
Survey data quality is also compromised by insincere or care-
less respondents who may provide hasty, random, or fabricated
answers (Meade and Craig, 2012).
Recent advances in large language models (LLMs) (Zhao
et al., 2025) have prompted researchers to reconsider long-
1arXiv:2605.19229v1  [cs.AI]  19 May 2026

standing limitations in survey methodology (Arora et al.,
2025; Jung et al., 2025; Perc, 2025; Qu and Wang, 2024;
Salecha et al., 2024). LLMs offer the potential to augment or
streamline nearly every stage of the survey research process
(Chakraborty and Nishimura, 2025): dynamic questionnaire
development, adaptive translations for diverse subpopulations,
real-time detection and intervention for disengaged respon-
dents, and sophisticated synthesis of both quantitative and
qualitative data (see Table 1). With the ability to process and
cluster thousands of open-ended responses, LLMs present new
opportunities to uncover latent patterns in attitudes and behav-
iors, going beyond traditional variable-by-variable analysis.
Yet integrating LLMs into survey workflows is not without
risk. These models can inherit or amplify biases present in
their training corpora, raising concerns about fairness and rep-
resentativeness, especially when deployed in culturally diverse
contexts (Ashwin et al., 2025; Gao et al., 2025). Data privacy
is an ongoing concern, as is the possibility of “hallucination,”
meaning plausible but misleading or entirely incorrect model
output (Zhao et al., 2025). Moreover, practical and ethical con-
straints in model deployment, supervision, and interpretation
challenge their scalability, reinforcing the need for vigilant
human oversight at every step (Perc, 2025).
The stakes are especially high indisaster survey research,
where methodological challenges are intensified by rapidly
changing populations, heightened respondent distress, and the
urgent need for timely, actionable information. Pre-disaster
studies grapple with uncertain population boundaries, low
engagement with hypothetical scenarios, and sparse prepared-
ness data (Hao et al., 2022), while post-disaster surveys face
disrupted communications, increased respondent distress, and
compressed timelines (King, 2002). The 2024 Hurricane Mil-
ton preparedness survey of Florida residents, which forms the
empirical backbone of this study, documents how routine time
constraints interact with hurricane preparation behavior across
household types (Wang et al., 2026). In these demanding con-
texts, traditional approaches to sampling, data collection, and
quality control can quickly become inadequate.
The intersection of LLMs and human expertise in disaster
research represents a promising yet underexplored frontier.
This study uses disaster preparedness and response as a test
case to systematically investigate these questions, drawing
on experimental analyses and the Hurricane Milton RAPID
survey to identify practical strategies and future directions for
effective human-AI collaboration. We empirically evaluate the
strengths and pitfalls of LLMs across five operational stages
of the survey-research workflow on the 2024 Hurricane Milton
disaster-preparedness survey (Wang et al., 2026), benchmark-
ing LLM-based imputation against three established classi-
cal baselines (IPW/MI, MICE+PMM, and missForest). Our
goal is to illuminate both the promise and the limits of LLM-
enhanced survey research, providing groundwork for responsi-ble guidelines and future innovation at the intersection of AI
and social measurement.
2. Background
2.1 Current Challenges in Survey Research
2.1.1 Evolution of Survey Research Methods
Survey methodology has evolved through a sequence of
distinct technological eras, each defined by trade-offs among
coverage, cost, and data quality. Face-to-face interviewing
produced the highest response rates but at prohibitive per-
interview cost (Groves and Couper, 1998). The telephone
survey era reduced costs substantially while preserving many
quality advantages of human contact. The theoretical frame-
works developed during these decades remain foundational:
Groves and Couper’s (1998) leverage-saliency model of re-
spondent cooperation and Tourangeau et al.’s (2000) four-stage
model of survey response continue to anchor contemporary
item design and mode-effect research.
The transition to self-administered web surveys further dis-
rupted the field’s economics while intensifying quality con-
cerns. Response rates fell steeply, and the Total Survey Error
(TSE) framework, which decomposes survey estimates into er-
rors of non-observation and observation, became the standard
vocabulary for diagnosing these losses (Groves and Lyberg,
2010). The Tailored Design Method (Dillman et al., 2014) cod-
ified mixed-mode best practices, while Couper’s (2017) review
documented the full scope of web displacement: collapsing re-
sponse rates, the proliferation of probability-based and opt-in
panel designs, and the integration of passive behavioral data
into survey architectures. Recent innovations in mixed-mode
design, including sequential web-phone protocols and push-to-
web strategies, have recovered some ground, with documented
gains of 12 to 25 percentage points over single-mode designs
(Coffey, 2024), but the secular decline in public cooperation
shows no sign of reversal. Federal economic surveys have
seen response rates fall by approximately 15 to 30 percentage
points over the past decade (Leduc et al., 2025).
2.1.2 Data Quality Challenges in Modern Surveys
Beneath the headline problem of declining response rates
lies a more troubling development: the growing unreliabil-
ity of completed responses. Multiple imputation, formalized
by Rubin (1987) as the principled replacement of each miss-
ing value with Mdraws from its posterior predictive distri-
bution combined via Rubin’s Rules, remains the statistical
standard for item-level nonresponse. MICE with predictive
mean matching (van Buuren and Groothuis-Oudshoorn, 2011)
has become the dominant implementation. Inverse-probability
weighting combined with multiple imputation extends cov-
erage to block-level missingness (Seaman et al., 2012), and
non-parametric random-forest imputers handle mixed-type
data with non-linear dependencies (Stekhoven and Bühlmann,
2

Table 1:Large Language Models in Survey Research: An Adoption Framework.
Survey Research
StageHow LLMs Can Assist Role of Human
Supervisor / Partner
Research &
Questionnaire
DesignStreamline and create survey questionnaires. Adapt and translate for different sub-
populations and modes (landlines, cellphone, online). Leverage extensive databanks
for thorough item generation. Suggest multidimensional constructs or latent vari-
ables.Ensure model is trained on
relevant data; oversee
ethical considerations in
instrument development.
Sample Selection Model sample characteristics for targeted recruitment. Optimize representativeness
across survey modes. Aid prediction of difficult-to-reach samples and inform alloca-
tion of resources. Advance beyond opt-in panels as LLM may help subpopulation
sampling.Guide selection criteria;
evaluate representativeness
and sampling quality;
interpret mode-sensitive
sampling implications.
Pilot Testing Generate synthetic respondent data; anticipate analytic challenges (e.g., by preparing
mock tables and outputs); test survey design robustness prior to fielding.Interpret synthetic datasets;
refine pilot instruments
based on LLM outputs;
verify utility for real
samples.
Data Collection Impute missing data. Generate synthetic responses as needed. Conduct bias mitiga-
tion. Identify and flag “bogus” or disengaged respondents in real time. Intervene on
quality (e.g., mid-survey clarifications or motivational prompts).Monitor ongoing data
quality; adjudicate
intervention strategies;
oversee ethical management
of respondent flagging and
replacement.
Data Analysis Extract key themes from open-ended responses; conduct statistical modeling and
formal analyses; produce rapid visualizations and summaries.Review and interpret
analytic outputs; ensure
validity of findings;
contextualize results for
reporting and
decision-making.
2012). These three methods constitute the classical compari-
son set for the LLM-based approach evaluated here.
The integrity of completed responses has also come under
mounting pressure. Pinzon-Espitia (2024) audited 31 fraud-
detection strategies currently deployed in commercial pan-
els and found that none adequately maintain analytic-quality
samples, with usable completion rates falling from roughly
75% to 10% in some panels between 2021 and 2024. John-
son et al. (2024) report that approximately 40% of submis-
sions in a participatory-mapping web survey were classified
as fraudulent despite multi-layer behavioral and IP screening.
Westwood (2025) demonstrates that autonomous AI agents
can pass standard attitudinal surveys at a 99.8% success rate
against conventional attention checks, and documents that over
one-third of Prolific respondents admitted using LLMs to com-
pose open-ended answers. The macroeconomic consequences
are already visible: consumer sentiment indicators diverged
sharply from verified retail-purchase behavior during 2023
to 2024, attributable in part to mode effects and recruitment-
composition shifts in underlying panels (Hacıo ˘glu Hoke et al.,
2025).2.2 Potential of Large Language Models
2.2.1 Overview of LLMs
Large language models are transformer-based neural net-
works trained on massive text corpora to model the conditional
probability of the next token given its context. GPT-3 demon-
strated that scaling alone yields a general-purpose model capa-
ble of competitive performance across diverse natural language
tasks using only in-context examples, without task-specific
fine-tuning (Brown et al., 2020). This finding reoriented the
NLP paradigm from task-specific architectures to prompt-
conditioned general models, a shift subsequently extended
and safety-aligned in frontier systems such as GPT-4 (OpenAI,
2023). Comprehensive surveys document this landscape in
detail, covering model families, training paradigms, bench-
mark performance, and open challenges (Minaee et al., 2025;
Naveed et al., 2025; Stanford Institute for Human-Centered
AI, 2025).
The principal limitation constraining scientific deployment
of LLMs is hallucination: the generation of outputs that are
syntactically plausible but factually incorrect or internally
inconsistent. Huang et al. (2024) distinguish factuality hal-
3

lucination from faithfulness hallucination and document that
mitigation strategies must be addressed throughout the entire
model lifecycle. Even frontier systems hallucinate on a non-
trivial share of prompts in some task categories, with rates
varying systematically with prompt structure and retrieval de-
sign (Ravi et al., 2025; Saxena, 2025; Frontiers in Artificial
Intelligence, 2025). Retrieval-augmented generation (RAG),
which grounds model outputs in explicitly retrieved evidence
rather than relying on parametric memory (Lewis et al., 2020),
provides partial but meaningful mitigation and is the primary
architectural strategy adopted in the pipeline evaluated here.
2.2.2 LLMs and Survey Research
The most influential early study of LLMs in survey con-
texts is Argyle et al.’s (2023) “silicon sampling” framework, in
which GPT-3 was conditioned on detailed demographic back-
stories derived from real survey respondents. The model’s
outputs matched human response distributions and cross-item
correlation patterns with what the authors termed high “algo-
rithmic fidelity.” Sun et al. (2024a) extended this to group-
level conditioning, showing that LLM-generated distributions
closely approximate real public-opinion marginals for many
items while performing worst on socially sensitive topics
where models systematically produce what the authors term
harmlessness bias, a shift toward socially desirable answers.
Horton (2023) evaluated LLMs as simulated economic agents
and found internally coherent but training-data-anchored be-
havior, positioning LLMs as instruments for theory exploration
rather than human-subject substitutes.
A parallel body of work has catalogued the risks of this ap-
proach. Bisbee et al. (2024) find that silicon-sampled GPT-3.5
responses inflate estimates of partisan and racial affective po-
larization by a factor of roughly seven relative to human bench-
marks, attributing the distortion to stereotype amplification in
persona conditioning. Ashwin et al. (2025) demonstrate that
LLM-based qualitative coding produces errors systematically
correlated with interviewee characteristics such as refugee
status, gender, and education level, leading to biased inference.
Performance gaps are especially severe in non-Western and
non-English contexts, consistent with training-corpus imbal-
ance (Sun et al., 2024b). Synthesizing 285 silicon-human
comparisons, Sarstedt et al. (2024) conclude that LLMs repli-
cate surface-level patterns but fail to reproduce many deeper
behavioral regularities, making them most credible at upstream
design stages. Crockett and Messeri (2025) argue further that
AI surrogates can reinforce existing sampling biases, creating
an illusion of generalizability rather than genuinely expanding
it.
On the constructive side, AI-assisted conversational inter-
viewing has been shown to enhance the specificity and ex-
planatory depth of open-ended responses, though with trade-
offs in respondent burden and attrition (Barari et al., 2025).
LLM-based phone agents achieve lower item-nonresponsethan self-administered web forms while presenting elevated
risk of leading-question artifacts (Kaiyrbekov et al., 2025).
Loru et al. (2025) caution against delegating evaluative judg-
ment to models susceptible to what they termepistemia, the
illusion of knowledge produced when surface plausibility
substitutes for contextual verification. Augmenting LLMs
with variational autoencoders to induce individual heterogene-
ity substantially improves performance on collective-opinion
benchmarks (CrowdLLM; Lin et al., 2024), and PersonaFuse
demonstrates that dynamically adapting personality traits to
conversational context opens new possibilities for adaptive sur-
vey interviewing (Tang et al., 2025). Hybrid LLM-statistical
approaches outperform either alone on distributional recon-
struction (Miranda and Balbi, 2025).
2.3Disaster Survey Research: Unique Contexts and Con-
straints
Disaster survey research operates under constraints that am-
plify every methodological challenge described above. Popula-
tion boundaries are inherently unstable: evacuation, sheltering,
displacement, and mortality all alter the sampling frame be-
tween hazard onset and the earliest feasible data collection,
often in ways that cannot be fully characterized until the sur-
vey itself is complete (King, 2002). Pre-disaster studies face
the additional problem of measuring behavioral intentions
toward low-probability, high-consequence events that many
respondents have never experienced, generating responses
anchored to hypothetical scenarios rather than to revealed
behavior (Lazo et al., 2015). Post-disaster studies face the
inverse: real behavior did occur, but the respondents who ex-
perienced it most intensely are those most likely to be missing
from the sample due to infrastructure disruption, psycholog-
ical distress, or temporary displacement (Hao et al., 2022).
The resulting data frequently exhibit block-wise missingness,
where entire question modules are absent for identifiable sub-
groups, a pattern that standard item-level imputation routines
are not designed to address.
Protection Motivation Theory (PMT) provides the dom-
inant explanatory framework for preparedness behavior in
this context (Floyd et al., 2000). PMT posits that protective
action follows from the joint output of threat appraisal, inte-
grating perceived severity, perceived vulnerability, and prior
experience, and coping appraisal, integrating response effi-
cacy, self-efficacy, and response cost. In disaster-preparation
contexts, response cost has both a financial and a temporal
dimension, and for time-constrained households the temporal
dimension is often the binding constraint on protective behav-
ior (Lazo et al., 2015; Slovic et al., 2007; Wang et al., 2026).
The Hurricane Milton instrument is specifically designed to
index this temporal dimension in depth, providing the theo-
retical scaffold for the knowledge-graph methods evaluated
here.
4

3.Data, Survey Instrument, and Theoretical
Framework
3.1 Data and Survey Instrument
All operations are anchored in the 2024 Hurricane Milton
preparedness survey of Florida residents ( n=946 ), collected
via online panel during the weeks following landfall and de-
scribed in detail in Wang et al. (2026). The instrument is orga-
nized into three blocks.Block A(always observed; 16 items)
contains demographic items including age, gender, race or
ethnicity, marital status, education, employment, occupation,
income bracket, housing tenure, household size, caregiving
burden, dependent age categories, a health-issue indicator, and
prior hurricane experience.Block B(routine time use; 8 or-
dinal items) measures daily time allocation, perceived time
scarcity, and schedule flexibility.Block C(hurricane prepara-
tion; 8 ordinal items) measures awareness timing, preparation
timing and duration, preparation stress, time spent on depen-
dents, and personal disruption.
A deterministic 80/20 split (seed =42 , by respondent identi-
fier) yielded a training set of 757 records and a validation set of
189 records. We derived four binary vulnerability flags from
Block A: low-income households (income at or below $45,999
USD), minority (race other than “White”), renter (apartment
or mobile home), and disabled (reported health issue other
than “No difficulties experienced”). Respondents satisfying at
least two flags were classified ascompound-vulnerable. The
validation set contained 72 compound-vulnerable respondents
(38%), sufficient for stratified evaluation. Where comparison
to population baselines was required, we used the U.S. Cen-
sus Bureau American Community Survey 5-year estimates
for Florida adults from vintages 2018 to 2022 (U.S. Census
Bureau, 2024).
3.2Theoretical Framework: Protection Motivation The-
ory
PMT specifies the cognitive sequence by which an in-
dividual translates a perceived threat into protective action
(Floyd et al., 2000). The individual first conducts a threat
appraisal, estimating perceived severity and perceived vulner-
ability weighted by prior experience and affective response.
The individual then conducts a coping appraisal, estimating
response efficacy (whether the action works), self-efficacy
(whether one can perform it), and response cost (what the ac-
tion consumes in time and resources). The differential between
the two appraisals produces a protection motivation that, when
sufficiently activated, translates into protective behavior. In
disaster-preparation contexts, for time-constrained households
the temporal dimension of response cost is often the binding
constraint (Lazo et al., 2015; Slovic et al., 2007; Wang et al.,
2026).
The Hurricane Milton instrument indexes the temporal di-
mension of response cost in depth, with 16 outcome items
Figure 1:Theory-constrained staged prediction pipeline. The six-
stage PMT cascade maps Block A demographic inputs through tem-
porally ordered constructs to hurricane preparation outcomes. Edges
retained in the PMT-constrained graph are shown; validated edges
(Spearman sign-concordant) are highlighted.
spanning routine time allocation, perceived time constraints,
and preparation timing. It does not, however, provide multi-
item scales for threat appraisal, coping appraisal, or protection
motivation in the conventional PMT sense. We therefore oper-
ationalize PMT as a six-stage temporal-cost cascade aligned
to the available items (Table 2), with each construct estimated
solely from Block A demographics or from upstream cascade
stages, ensuring that no prediction target enters its own condi-
tioning set.
We instantiated the cascade as a PMT-constrained co-
occurrence graph built from the 757 training records. Edges
were retained only between source and target fields in which
the source occupied an upstream PMT stage relative to the tar-
get. This procedure yielded 204 source nodes, 9,605 weighted
edges, and 344 permitted source-to-target field pairs, with
edge weights estimated as conditional cell counts. To assess
whether the graph’s directional assumptions held empirically,
we tested each edge for agreement between its theoretical
prediction and the observed Spearman correlation in the train-
ing data. This produced two operational variants: the full
PMT-constrained graph, retaining every theoretically licensed
edge with its empirical weight; and a validated-edge subgraph,
restricted to edges whose theoretical sign matches the em-
pirical Spearman correlation. The graph organizes evidence
rather than enforcing a fixed inference path (Figure 1): it spec-
ifies which conditional relationships are theoretically licensed,
while leaving the integration of that evidence either to a staged
process or to a single model call.
5

Table 2:PMT Cascade Operationalized on the Hurricane Milton Instrument.
Stage PMT Construct Block Survey Variables
1 Routine time allocation
(situational baseline)B Time_Family, Time_Personal, Time_Sleep
2 Time constraint /
temporal response costB Constraint_Work, Constraint_Household, Pressure_Overwhelmed, Flex_Work,
Time_Scarcity
3 Threat appraisal C Aware_Time, Helene_Impact, Hurricane_Experience
4 Coping appraisal C Prep_Stress, Prep_Time
5 Protective behavior C Prep_Start, Prep_Action_Time
6 Temporal impact C Personal_Disruption, Time_Spent_Dependents
4.LLM-Based Methods for Survey Research
Operations
A survey-research project unfolds through a sequence of
five operational stages: designing the instrument, selecting the
sample, pilot-testing the questionnaire, collecting and clean-
ing the data, and analyzing it. As introduced in Table 1, we
mapped a series of LLM operations to each of the five stages.
For each stage we specify the LLM’s input, output, and val-
idation strategy, and whether the operation is evaluated as a
controlled experiment (Stages 3 and 4) or as a demonstration
grounded in real survey data (Stages 1, 2, and 5).
Methods for predicting survey responses from a respon-
dent’s persona, defined here as the Block A demographic
profile, vary along two design choices: what task-specific
evidence the language model receives at inference time, and
how that evidence is structured. We evaluate seven methods
spanning this design space; their prompt configurations are
summarized in Table 3.
Two baseline methods give the model no task-specific evi-
dence and rely on its pretrained knowledge alone. TheZero-
Shot LLM (ZS-LLM)issues the prediction request together
with the persona and the survey question. TheFew-Shot
LLM (FS-LLM)extends the same prompt by inserting five
complete training respondents drawn at random as illustrative
input-output pairs.
Three retrieval-augmented methods follow the RAG pattern:
at inference time an external information store is queried for
evidence relevant to the target persona, and the retrieved evi-
dence is inserted into the prompt before the model is asked to
answer (Lewis et al., 2020). The three methods share this
overall pattern and differ in the structure of the evidence
store, providing a controlled comparison of three progres-
sively more informative organizations of the same training
data. TheEmbedding-Retrieval LLM (ER-LLM)is the
standard RAG configuration: the store is a flat index of the
757 training-set respondents, each represented as a numerical
embedding; retrieval selects the training respondents whose
embeddings lie nearest the target’s, treating the training set
as an unstructured collection. TheGraph-Retrieval LLM(GR-LLM)replaces the flat index with a knowledge graph in
which respondent attributes and survey answers are nodes con-
nected by weighted edges recording empirically observed co-
occurrences, so that retrieval traces a relational structure rather
than a similarity score alone. TheStaged Theory-Informed
LLM (Staged-TLM)operates on the PMT-constrained knowl-
edge graph, with retrieval proceeding one PMT stage at a time
and each stage’s point estimate becoming an input to the next.
The three methods together isolate three structural increments:
the addition of semantic similarity (ER-LLM), of empirical
relational structure (GR-LLM), and of theoretical direction
(Staged-TLM).
The remaining two methods hold the PMT-constrained
graph as the evidence source and vary how the model in-
tegrates retrieved evidence. TheMarginal Theory-Informed
LLM (Marginal-TLM)supplies the model with the graph’s
full set of marginal and conditional probability distributions in
a single prompt, so that the model weighs the evidence in one
integration step rather than committing to staged point esti-
mates. TheAnchored Marginal Theory-Informed LLM (A-
TLM), the configuration we propose, extends Marginal-TLM
with two persona-targeted signals: the five training respon-
dents nearest the target persona on key demographics (peer
examples), and, when the target persona meets at least two
vulnerability criteria, the empirical mean difference in each
outcome between the compound-vulnerable training subsam-
ple and the full training sample, conveyed as a soft directional
cue rather than a numerical override.
Stage 3 evaluates all seven LLM methods. The Stage 4 main-
text comparison evaluates five of them (ZS-LLM, FS-LLM,
Staged-TLM, Marginal-TLM, A-TLM) against three classical
imputation baselines: inverse-probability-weighted multiple
imputation (IPW/MI; Seaman et al. 2012), multivariate impu-
tation by chained equations with predictive mean matching
(MICE+PMM; van Buuren and Groothuis-Oudshoorn 2011),
and a random-forest iterative imputer (missForest; Stekhoven
and Bühlmann 2012). The remaining two LLM methods (ER-
LLM, GR-LLM) and a Validated-Edge Theory-Informed LLM
(VE-TLM) are reported in supplementary materials. All LLM
calls usedclaude-sonnet-4.5at temperature 0.1.
6

Table 3:Learning Contexts in LLMs: Method Descriptions and Illustrative Examples.
Method Description Example
ZS-LLM Uses only pretrained knowledge and the task instruction; no task-specific examples in the prompt. Predicts 16 Block-B/C
answers from Block A
demographics alone; no
training-set evidence
supplied.
FS-LLM Augments the zero-shot prompt with five randomly selected complete training respondents as input–output
pairs.Five training respondents
inserted verbatim before
the prediction request.
ER-LLM Queries a dense-vector flat index at inference time; top- ksemantically similar training narratives appended to
prompt (vanilla RAG).LightRAG encoder
retrieves nearest training
respondents by embedding
similarity.
GR-LLM Queries a data-driven co-occurrence graph; conditional cell distributions consistent with the target persona
returned as evidence (graph-RAG).Weighted graph queried
for cell distributions
consistent with target
demographics; cells
summarized in prompt.
Staged-TLM Traverses the PMT-constrained graph in a fixed six-stage causal cascade; each stage commits to a point
estimate that conditions the next.Stage 1 predicts time use;
Stage 2 predicts
constraints conditional on
Stage 1; cascade continues
through coping appraisal,
protective behavior, and
temporal impact.
Marginal-TLM Summarizes the full PMT-constrained graph as marginal/conditional distributions; all evidence supplied in a
single prompt.Prompt reports distribution
of training respondent
answers by demographic
profile and upstream
PMT-stage values; model
returns all 16 predictions
in one call.
A-TLM Extends Marginal-TLM with (i) the k=5 nearest training peers and (ii) empirical subgroup-shift parameters
for compound-vulnerable personas as a soft directional signal.Peer examples appended;
for compound-vulnerable
targets, per-item mean
difference between
vulnerable and full training
samples also included in
prompt.
4.1Stage 1: Questionnaire Design via the LLM Instru-
ment Audit
Before fielding a survey, the research team must verify that
the instrument measures the theoretical constructs of inter-
est, a step traditionally performed by manual expert review.
This stage automates that audit with a single LLM call. We
provided the LLM with the 16 outcome items together with
the full specification of the nine PMT constructs, namely Per-
ceived Severity, Perceived Vulnerability, Fear Arousal, Prior
Experience, Response Efficacy, Self-Efficacy, Response Cost,
Protection Motivation, and Protective Behavior, and required
a structured response covering per-construct adequacy scores
(1 to 5), identified construct gaps with literature-grounded rec-
ommendations for additional items, identified redundancies,
and an overall verdict on instrument sufficiency.
Validation proceeded along three axes. First, we assessed
the face validity of identified gaps by comparing them againstpublished PMT-based hurricane instruments (Floyd et al.,
2000; Lazo et al., 2015; Slovic et al., 2007). Second, each
flagged redundancy was independently inspected in the code-
book. Third, we assessed whether the audit’s verdict correctly
reconciled the full PMT framework with the instrument’s ac-
tual research focus on time-constrained disaster preparation
(Wang et al., 2026).
4.2Stage 2: Sample Selection via the LLM Sample-
Coverage Prior
Survey samples almost never match the population on ev-
ery dimension, and the resulting under-representation distorts
inferences. This stage uses the LLM as a pre-fielding plan-
ning tool: it forecasts which subgroups are likely to be under-
represented in a post-hurricane survey, and that forecast is then
validated against the empirical gap measured after fielding.
The LLM was prompted to predict, from the published
disaster-survey literature (Fussell et al., 2014), which demo-
7

graphic subgroups are typically under-represented in post-
hurricane convenience samples, without access to the Hur-
ricane Milton sample. The model returned a ranked list of
subgroups with literature-grounded justifications. We then
computed the empirical gap as the Florida ACS population
percentage minus the Hurricane Milton sample percentage for
each of eight subgroups. Validation used the Spearman rank
correlation between the LLM’s prior ranking and the empirical
gap ranking as the primary performance metric, with a thresh-
old of ρ>0.5 as evidence of a useful pre-fielding planning
tool.
4.3 Stage 3: Pilot Testing via Response Prediction
Pilot testing has historically required recruiting real respon-
dents to verify that items produce variation, check that planned
cross-tabulations will populate at usable sample sizes, refine
ambiguous wording, and budget statistical power before com-
mitting to full fielding. We tested whether an LLM can fill the
same design-validation role by generating simulated responses
conditioned only on a respondent’s demographic profile. The
purpose is design support, not data fabrication: simulated
responses inform questionnaire and analysis-plan decisions
before fielding and are not introduced into the final analytic
dataset.
Each of the seven LLM methods was asked to predict the
full set of 16 Block B and Block C answers for each of the 189
held-out validation respondents, conditioning only on Block
A demographics. The full A-TLM specification, including the
0.10 Likert-level inclusion threshold for the subgroup-shift
signal, is given in the supplementary material. Evaluation was
restricted to ordinal-mappable cells; fields containing skip-
logic missing values were excluded. For each method we
report mean absolute error (MAE) on the ordinal Likert map-
ping, within-1 accuracy, quadratic-weighted Cohen’s κ, and
semantic similarity computed via the SentenceTransformer
all-MiniLM-L6-v2 model.
4.4Stage 4: Data Collection via Simulated-Missingness
Imputation
Real surveys rarely collect complete responses. Missing-
ness is typically related both to the answer itself and to respon-
dent characteristics, a pattern known as missing not at random
(MNAR). This stage tests whether an LLM guided by theory
and demographic anchoring can impute missing responses
more reliably than classical methods under progressively de-
manding conditions.
We evaluated each method under four reproducible miss-
ingness mechanisms applied to the 189-respondent validation
set (seed =42 ).Mechanism S1(MCAR; sanity baseline)
deleted 20% of Block B and Block C cells uniformly at ran-
dom.Mechanism S2(MAR; demographic-driven) assigned
higher deletion probability to low-income and minority respon-
dents, conditional on Block A demographics.Mechanism S3(MNAR-moderate; response-driven) elevated Block C dele-
tion probability for respondents with lower preparation scores.
Mechanism S4(MNAR-severe; block-wise) deleted the entire
Block C for respondents meeting at least two vulnerability
criteria, simulating permanent displacement; this produced
60% Block C missingness concentrated among compound-
vulnerable respondents.
IPW/MI was implemented following Seaman et al. (2012):
respondents with at least 50% block-level missingness were
excluded and inverse-probability weighted using logistic-
regression weights conditional on Block A; included respon-
dents with isolated missing items received M=5 multiple
imputations with the inverse probability of inclusion as a
covariate; Rubin’s Rules combined results and a sandwich
variance estimator was used for the weighted-mean estima-
tor. A component ablation evaluated four A-TLM variants on
Block C across the four scenarios: Marginal-TLM alone, plus
peer examples only, plus vulnerability deltas only, and the full
A-TLM combining both signals.
For each method, scenario, and block, the primary metrics
are root-mean-square error (RMSE) and signed bias on deleted
cells. Secondary metrics include 95% confidence-interval cov-
erage via Rubin pooling, symmetric Kullback-Leibler diver-
gence between imputed and ground-truth Likert distributions,
and the ℓ2norm of the coefficient difference between OLS fits
on imputed versus ground-truth datasets. All metrics are com-
puted on both the full sample and the compound-vulnerable
subgroup. Five preregistered sanity checks gated the results:
(1) positive between-imputation variance under Rubin pooling;
(2) IPW weights bounded at 10; (3) zero out-of-range imputed
values on the 1 to 5 scale; (4) missingness patterns in MNAR
scenarios significantly different from MCAR and MAR by
Fisher’s exact test; and (5) exact preservation of ground-truth
values wherever the deletion mask was zero.
4.5Stage 5: Data Analysis via the Graph-Grounded Sur-
vey Assistant
Survey datasets are typically delivered as static files that
require statistical expertise to query. This stage tests whether
routing user questions through an LLM that retrieves cells of
a knowledge graph can let non-specialist stakeholders receive
numerically grounded answers. We deployed a web chatbot
that mediates between natural-language survey questions and
the PMT-constrained co-occurrence graph built from the 757
training records. The pipeline parses each question via an
LLM call to identify relevant survey variables, retrieves the
corresponding marginal distributions, cross-tabulations, and
conditional answer patterns from the graph, and returns a
data-grounded answer that explicitly references sample sizes,
percentages, and PMT-stage interpretations.
We validated the system against three criteria: every nu-
meric claim traceable to a cell of retrieved graph evidence
(quantitative grounding); findings linked to PMT-stage inter-
8

pretations where applicable (theory integration); and the assis-
tant declines to answer when retrieved evidence is insufficient
rather than confabulating a response (epistemic honesty).
5. Experimental Results
5.1 Stage 1: Construct-Adequacy Audit
We asked the LLM to score the coverage of each of the nine
PMT constructs by the 16-item Hurricane Milton instrument
on a 1 to 5 adequacy scale and to recommend additional items
for each construct scoring below the threshold. Per-construct
scores are reported in Table 4. These results should be inter-
preted as an illustrative demonstration of LLM-assisted audit
capability rather than a definitive validation, given that no for-
mal inter-rater reliability comparison was conducted against
an expert-panel baseline.
Only one construct, the temporal dimension of response
cost, received a score of 4 or higher; it was mapped to seven
items spanning routine time allocation and time constraint.
Three constructs (Perceived Severity, Response Efficacy, and
Protection Motivation) scored at the floor of the scale because
the instrument contains no items mapped to them. The audit’s
overall verdict was that the instrument is fit for the project’s
research focus on time-constrained disaster preparation but
does not support full PMT pathway testing, an assessment con-
sistent with the instrument’s original design goals (Wang et al.,
2026). Eight literature-grounded gap-item recommendations
accompanied the per-construct scores. The same assessment
by an expert reviewer would typically require several hours
of manual cross-referencing; the LLM returned an equivalent
scoping diagnostic in a single inference call.
5.2 Stage 2: Sample-Coverage Prior
We asked the LLM to rank demographic subgroups most
likely to be under-represented in a post-hurricane convenience
sample, conditioning only on the published disaster-survey
literature and not on the Hurricane Milton respondents. We
then computed the actual deviation between the Milton sample
composition and the Florida adult population from the ACS
5-year estimates. Both quantities are reported in Table 5.
The Spearman rank correlation between the LLM’s
literature-prior ranking and the empirical gap ranking was
ρ=0.12 , indicating limited correspondence between what
the literature predicts and what the online-panel recruitment
produced. The LLM correctly identified Hispanic respon-
dents as among the most under-represented, consistent with
the disaster-survey literature (Fussell et al., 2014), but ranked
Black respondents as the most under-represented when their
sample share was within one percentage point of the popula-
tion reference. The two largest discrepancies involved over-
representation of respondents reporting any disability (29.1%
versus 14.0%) and under-representation of high-income re-
spondents (10.5% versus 15.0%); neither pattern is antici-pated by the disaster-survey literature, and both are consistent
with online-panel self-selection dynamics rather than disaster-
specific coverage gaps. The value of the LLM prior in this
stage lies less in the accuracy of its rankings and more in estab-
lishing a structured baseline: the divergences between the prior
and the observed sample composition identify subgroups that
warrant targeted recruitment attention in subsequent fielding
waves and that would not have been flagged by the literature
consensus alone.
5.3 Stage 3: Response Prediction for Pilot Testing
Each of the seven LLM methods was asked to predict the
full set of 16 Block B and Block C answers for each of the 189
held-out validation respondents, conditioning only on Block
A demographic profiles. Mean absolute error on the ordinal
coding, computed separately for Block B and Block C, is
reported in Table 6. These results reflect the performance of
each method as a design-support tool for pilot testing, not as a
substitute for primary data collection.
The two blocks differed substantially in predictability.
Block B MAE was lower than Block C MAE across all seven
methods because routine time use is largely a function of de-
mographic role characteristics, such as age, household size,
employment type, and schedule structure, that are directly
recoverable from Block A. Hurricane-preparation answers de-
pend additionally on perceived threat, prior experience, house-
hold resources, and local risk perception, none of which are
recoverable from demographics alone. Block C therefore
shows both higher overall error and a wider spread between
the best (Marginal-TLM, 1.067) and worst (ZS-LLM, 1.159)
methods.
The seven methods represent a sequence of design choices,
each adding a different layer of evidence. ER-LLM, the
standard retrieval-augmented configuration, was the worst-
performing method (combined MAE 1.097), suggesting that
nearest-neighbor retrieval without relational or theoretical
structure can mislead the model when respondents with similar
narratives differ in the demographic dimensions that drive hur-
ricane behavior. GR-LLM recovers the accuracy lost by ER-
LLM by replacing the embedding index with a co-occurrence
graph (1.016). Staged-TLM layers a directional PMT cascade
on the same graph but traverses it one stage at a time; the re-
sulting error propagation causes it to underperform GR-LLM
(1.030). Marginal-TLM integrates the full PMT-organized
evidence in a single call, achieving the lowest combined MAE
(0.993). A-TLM performs on par with Marginal-TLM (1.003):
the persona-targeted signals it adds do not further reduce
prediction error at this stage, where all respondents are ob-
served, but become consequential at Stage 4 where compound-
vulnerable respondents are systematically absent. Across all
methods, the absolute MAE values of approximately one or-
dinal level on a five-point scale are appropriate for design-
support decisions such as prototyping cross-tabulations and
9

Table 4:LLM Adequacy Audit of Hurricane Milton Instrument Against Nine PMT Constructs.
PMT Construct Score
(1–5)Instrument Items Mapped
Perceived Severity 1 None
Perceived Vulnerability 2 Helene_Impact
Fear Arousal 2 Prep_Stress
Prior Experience 2 Helene_Impact
Response Efficacy 1 None
Self-Efficacy 2 Flex_Household
Response Cost
(temporal)4 Prep_TotalTime, Dep_Time_Pct, Impact_Personal, Constraint_Work, Constraint_Household,
Time_Scarcity, Pressure_Overwhelmed
Protection Motivation 1 None
Protective Behavior 3 Aware_Time, Prep_Start, Prep_TotalTime
Table 5:Sample Composition versus Florida ACS Reference. Positive gaps indicate under-representation in the Milton sample relative to the
population. Rank 1 = most under-represented; Rank 8 = most over-represented.
Subgroup Milton (%) ACS Florida (%) Gap (ACS−Milton, pp) LLM Rank Empirical Rank
White non-Hispanic 66.7 52.0−14.7 7 7
Hispanic 10.8 26.0+15.2 2 1
Black 17.0 16.0−1.0 1 4
Asian 3.5 3.0−0.5 6 3
Income<$35k 27.6 24.0−3.6 3 6
Income>$150k 10.5 15.0+4.5 8 2
Renter 34.1 33.0−1.1 4 5
Disabled (18+) 29.1 14.0−15.1 5 8
Note.Spearmanρbetween LLM prior ranking and empirical gap ranking=0.12.
Table 6:MAE on Response Prediction, 189 Validation Respondents
(lower is better). Bold = best.
Method Blk B Blk C Combined
Marginal-TLM 0.949 1.067 0.993
A-TLM 0.960 1.074 1.003
GR-LLM 0.952 1.131 1.016
Staged-TLM 0.953 1.156 1.030
FS-LLM 0.984 1.117 1.031
ZS-LLM 0.999 1.159 1.060
ER-LLM 1.081 1.123 1.097
Note.Cell-weighted avg; skip-logic cells excluded.
identifying low-frequency cells, but fall short of what would
be required for population-level inferential precision.
5.4 Stage 4: Imputation Under Simulated Missingness
5.4.1 Main Comparison
Root-mean-square error and signed bias on the deleted
Block C cells across the four missingness mechanisms are
reported in Table 7 and visualized in Figure 2. These results
are directionally informative; the performance differences be-
Figure 2:Block C imputation RMSE across four missingness mech-
anisms. A-TLM achieves the lowest RMSE under S4, the scenario
most representative of post-disaster block-wise displacement.
tween closely ranked methods, particularly under S4, should
be interpreted with the caveat that the compound-vulnerable
validation subsample comprises 72 respondents and formal
inferential statistics on the RMSE differences were not com-
puted.
A-TLM achieved a lower RMSE than IPW/MI across all
four scenarios, a lower RMSE than MICE+PMM across all
four scenarios, and the lowest RMSE of any method evaluated
in S4 (1.439, compared with 1.496 for missForest). missForest
10

Table 7:Block C Imputation: RMSE and Signed Bias Across the Four Missingness Mechanisms. Bold denotes the lowest RMSE under S4.
Method S1 RMSE S1 Bias S2 RMSE S2 Bias S3 RMSE S3 Bias S4 RMSE S4 Bias
ZS-LLM 1.316+0.526 1.385+0.430 1.580+0.866 1.677+0.611
FS-LLM 1.361+0.164 1.486+0.000 1.533+0.536 1.616+0.096
Staged-TLM 1.597+0.603 1.588+0.616 1.910+1.031 1.853+0.768
Marginal-TLM 1.287+0.328 1.347+0.302 1.506+0.742 1.586+0.485
A-TLM 1.316+0.026 1.402−0.081 1.436+0.3921.439−0.121
IPW/MI 1.828+0.283 1.607−0.177 1.788+0.511 1.716−0.096
MICE+PMM 1.412+0.071 1.482+0.012 1.529+0.330 1.671+0.145
missForest 1.141−0.043 1.258−0.140 1.320+0.216 1.496−0.631
Figure 3:Bias-RMSE frontier on Block C imputation by missingness
scenario. Methods below and to the left of A-TLM are preferable on
both metrics simultaneously; no method clearly dominates A-TLM
in any scenario.
achieved the lowest RMSE in S1, S2, and S3 but the largest ab-
solute bias of any method in S4 ( −0.631 ordinal levels), reflect-
ing a systematic under-prediction of compound-vulnerable
respondents that would distort any policy-relevant marginal-
mean estimate derived from the imputed data. Staged-TLM
had the highest RMSE in three of the four scenarios. The
bias-versus-RMSE frontier in Figure 3 shows that no method
clearly dominates A-TLM on both metrics simultaneously in
any scenario, though the margin in S4 is modest and future
evaluations with larger validation samples will be needed to
confirm this ordering.
5.4.2 Component Ablation
To attribute A-TLM’s improvement over Marginal-TLM
to its two added signals, we ran four ablations on Block C
imputation: Marginal-TLM with neither signal, with peer
examples only, with the vulnerability cue only, and with both.
Signed bias on deleted Block C cells is reported in Table 8.
Adding peer examples to Marginal-TLM reduced absolute
bias by 0.22 ordinal levels in S1, 0.23 in S2, 0.23 in S3, and
0.12 in S4. Adding the vulnerability cue alone produced
smaller but consistent reductions across all four scenarios. The
two signals combined super-additively only in S4 (A-TLM
bias 0.298 versus peer-only bias 0.379), the scenario in which
compound-vulnerable respondents are entirely absent from
the observed evidence. There, the vulnerability cue carries
non-redundant information that peer examples alone cannot
supply.
5.4.3 Subgroup-Stratified Bias
A near-zero overall bias can mask substantial subgroup-
specific errors of opposite sign. We computed bias separately
Figure 4:Subgroup-stratified bias on Block C imputation under S2
(MAR). Bars show signed bias separately for compound-vulnerable
(n=39 ) and non-vulnerable ( n=47 ) validation respondents. Near-
zero overall bias in several methods conceals opposing errors of
opposite sign across subgroups.
for non-vulnerable and compound-vulnerable respondents un-
der S2 on the deleted Block C cells (Figure 4).
Under S2, FS-LLM produced an overall bias of essen-
tially zero on Block C. This figure was the cancellation of
a+0.340 -level over-prediction among non-vulnerable respon-
dents ( n=47 ) and a −0.410 -level under-prediction among
compound-vulnerable respondents ( n=39 ). A-TLM pro-
duced a smaller but qualitatively similar pattern ( +0.234 and
−0.462 respectively). The cancellation is a general feature
of LLM imputation under this scenario rather than a property
of any single method. For policy-relevant applications where
estimates for high-missingness subgroups are the inferential
target, reporting overall bias alone is insufficient; subgroup-
stratified estimates should be reported alongside aggregate
figures as standard practice.
5.4.4 Routine Time-Use Items (Block B)
All theory-informed methods outperformed the baseline
LLM methods on Block B. However, A-TLM offered no mea-
surable improvement over Marginal-TLM, with absolute-bias
differences below 0.05 ordinal levels across all four scenarios.
The persona-targeted signals in A-TLM contribute specifically
to Block C, where respondent-level heterogeneity not captured
by demographics is the binding constraint. Routine time-use
behavior is sufficiently predicted by Block A demographic role
characteristics that the model’s pretraining prior already pro-
11

Table 8:Component Ablation: Signed Bias on Block C Across the Four Missingness Scenarios.
Variant S1 S2 S3 S4
Marginal-TLM (baseline)+0.353+0.337+0.794+0.500
+Peer examples only+0.138+0.105+0.567+0.379
+Vulnerability cue only+0.276+0.233+0.691+0.409
+Both signals (A-TLM)+0.164+0.105+0.588+0.298
vides adequate signal. Detailed Block B results are reported
in supplementary Table S3.
5.5 Stage 5: Knowledge-Graph Chatbot
We deployed the Graph-Grounded Survey Assistant, a web
chatbot that retrieves evidence from the PMT-constrained
co-occurrence graph in response to natural-language ques-
tions and constructs an answer grounded in the retrieved cells.
We evaluated three captured exchanges against the three pre-
specified criteria.
In one exchange, the user asked whether respondents with
flexible work schedules report less preparation stress. The
assistant retrieved the relevant cross-tabulation and reported,
with sample sizes, that 26.2% of respondents with “Very flexi-
ble” schedules also reported “Not at all stressed,” compared
with 9.6% among respondents with “Not flexible” schedules.
Each percentage was traceable to a specific retrieved cell, and
the assistant contextualized the finding within PMT by not-
ing that schedule flexibility (a Stage 2 time-constraint item)
moderates coping appraisal. This exchange satisfied both the
quantitative-grounding and theory-integration criteria.
In a second exchange, the user asked whether respondents
with prior multi-hurricane experience start preparing earlier.
The assistant retrieved the marginal distributions of the two rel-
evant items but found no joint cross-tabulation in the retrieved
evidence and declined to answer (Figure 5). This refusal be-
havior is architectural: the system is constrained to ground
every numeric claim in evidence present in the prompt, and
in its absence states which additional data would be required.
This exchange satisfied the epistemic-honesty criterion.
Figure 5:Captured exchanges from the Graph-Grounded Survey
Assistant. Top: successful quantitative retrieval with PMT-stage
contextualization. Bottom: epistemic refusal when retrieved evidence
is insufficient.
6. Discussion
6.1 Contributions to Knowledge
This paper operationalizes a five-stage framework for em-
bedding large language models within the survey-research
workflow and evaluates each stage on the 2024 Hurricane
Milton preparedness survey of Florida residents (Wang et al.,
2026). The contribution claimed here is the operationaliza-
tion itself: each of the five stages carries a reproducible script
grounded in the same dataset and the same PMT-constrained
co-occurrence graph, and every numerical claim traces to a
specific table or figure. We discuss the contributions at three
levels: the conceptual framework, the methodological innova-
tions, and the broader implications for how AI systems should
complement human expertise in survey science.
6.1.1 A Unified Operational Framework for LLM-Augmented
Survey Research
Prior work on LLMs in survey contexts has largely ad-
dressed individual operations in isolation: whether silicon
sampling can replicate human response distributions (Argyle
et al., 2023; Sun et al., 2024a), whether LLM-assisted inter-
viewing improves open-ended data quality (Barari et al., 2025;
Kaiyrbekov et al., 2025), or whether AI-generated responses
introduce systematic error (Bisbee et al., 2024; Ashwin et al.,
2025). No prior study evaluates LLM contributions across
12

the full survey lifecycle on a single shared empirical testbed.
By anchoring all five stages to the same dataset and theoreti-
cal structure, this framework enables cumulative evaluation:
subsequent studies can extend or challenge individual stages
without rebuilding the evaluation infrastructure from scratch.
The framework also reframes a normative question that has
run through this literature. Rather than asking whether LLMs
can replace human judgment in survey research, it asks where
in the workflow LLMs can serve as a reliable assistant, un-
der what conditions, and subject to what constraints. This
reframing aligns with the position that AI systems are most
productively understood as tools that augment human capac-
ity in well-defined tasks rather than as general substitutes for
domain expertise (Crockett and Messeri, 2025; Sarstedt et al.,
2024). The five-stage structure makes that position opera-
tionally concrete by assigning a human oversight role at each
stage alongside the LLM operation.
6.1.2 Theory-Constrained Retrieval as a Methodological Con-
tribution
The experimental results across Stages 3 and 4 establish
that how the evidence store is organized matters as much as
the decision to retrieve at all. Standard RAG grounds model
outputs in empirical evidence but treats that evidence as an un-
structured collection. Replacing the flat embedding index with
a co-occurrence graph that encodes relational structure recov-
ers accuracy lost by embedding-based retrieval. Crucially, the
gain from theory-constrained graph retrieval is realized only
when evidence is integrated in a single model call. Staged-
TLM, which applies the same PMT structure but commits to a
point estimate at each step, amplifies bias across the cascade
rather than correcting it, confirming that the theoretical struc-
ture is not itself the source of gain unless the model can weigh
all evidence simultaneously. This finding connects to broader
observations in NLP on the risks of committing to interme-
diate outputs before all evidence is integrated (Huang et al.,
2024), extending that insight to a structured social-science
application.
In the survey methodology literature, the finding that theo-
retical structure improves imputation performance by shaping
how auxiliary information is weighted is novel. PMT has
served primarily as a post-hoc explanatory framework for pre-
paredness behavior (Floyd et al., 2000; Lazo et al., 2015; Wang
et al., 2026), but it has not previously been operationalized
as a retrieval constraint during missing-data imputation. The
demonstration that this operationalization reduces signed bias
on compound-vulnerable respondents represents a meaningful
integration of substantive domain theory and computational
method.
6.1.3 Imputation Performance Under Disaster-Relevant Miss-
ingness
The Stage 4 results surface a pattern with practical con-
sequences for disaster research. Classical methods such asmissForest achieve competitive overall RMSE but produce
the largest absolute bias on compound-vulnerable respondents
under block-wise displacement conditions. This occurs be-
cause aggregate RMSE can remain favorable even when errors
of opposite sign cancel across respondent groups. A-TLM
reduces this subgroup-differential bias by anchoring predic-
tions to demographically matched peer examples and, when
the target respondent meets multiple vulnerability criteria,
to an empirical directional signal derived from the training
distribution of respondents with comparable characteristics.
Subgroup-stratified bias reporting is therefore proposed not as
an optional sensitivity check but as a standard reporting prac-
tice for any imputation workflow applied to disaster survey
data where high-missingness subgroups are also the primary
inferential targets.
6.1.4 LLMs as Bounded Research Assistants: Scope and Epis-
temic Honesty
A recurring challenge in applied AI research is communicat-
ing what a system can and cannot do without either overstating
its performance or dismissing genuine utility. This study ad-
dresses that challenge operationally by reporting, for each
stage, both what the LLM contribution achieves and where
it falls short. The Stage 2 result is illustrative: a literature-
derived prior for sample coverage produced a Spearman corre-
lation of 0.12 with the observed ACS gap. The informational
content lay in the divergences: the two subgroups where the
prior most differed from observation identified recruitment-
mode artifacts that the disaster-survey literature does not de-
scribe and that would not have been flagged by the literature
alone. The LLM is useful here not because its rankings were
accurate but because its structured hypothesis provided a base-
line that the empirical comparison could test and revise, con-
sistent with Horton 2023’s positioning of LLMs as instruments
for hypothesis generation and with Sarstedt et al. 2024’s rec-
ommendation that silicon samples be used at upstream design
stages.
The Stage 5 chatbot extends this principle architecturally.
By constraining the system to ground every numerical claim
in explicitly retrieved graph cells and to decline when those
cells are absent, the chatbot converts hallucination from an
inherent model tendency into a manageable design parameter.
The refusal behavior demonstrated in Figure 5 is a feature,
not a limitation: it makes the boundary between what the
data support and what they do not visible to the user, directly
addressing the epistemia risk identified by Loru et al. (2025).
6.2 Limitations and Future Directions
We address the principal limitations of this study alongside
the research directions they motivate.
6.2.1 Evaluation Scope and Generalizability
The framework was evaluated on a single post-hurricane
event in one U.S. state using an online panel. Performance in
13

face-to-face field settings, other disaster types, or non-English
survey environments is not yet established. LLM performance
varies with the linguistic and cultural profile of the training
corpus (Sun et al., 2024b), and post-hurricane convenience
samples in the United States are structurally distinct from sam-
ples collected through other modes or in other regions (King,
2002; Hao et al., 2022). Within these bounds, the Hurricane
Milton data provide an ecologically valid testbed grounded
in a real post-disaster event with documented composition
and ACS population benchmarks (Wang et al., 2026). The
single-event scope motivates a replication agenda: applying
the same five-stage operations to post-disaster datasets from
other events and settings will enable systematic evaluation
of what generalizes and what is context-specific. Extending
the framework to non-English contexts will require multilin-
gual foundation models and evaluation datasets that reflect the
linguistic diversity of global disaster-affected populations.
6.2.2 Sample Scale and Statistical Precision
The Stage 4 imputation benchmark for compound-
vulnerable respondents rests on 72 validation respondents
under the most demanding missingness scenario. The per-
formance differences reported are directionally informative
and consistent with a methodological pilot study, but formal
inferential statistics such as bootstrap confidence intervals
on RMSE differences were not computed. Future evalua-
tions should target compound-vulnerable subsamples large
enough to support such tests. Linking survey data to admin-
istrative records, such as disaster-relief registrant files, could
also provide verified ground truth that goes beyond held-out
validation-set simulation.
6.2.3 Demonstration Versus Controlled Experiment
Three of the five stages, the instrument audit, the sample-
coverage prior, and the knowledge-graph chatbot, are evalu-
ated as demonstrations grounded in real data rather than as
controlled experiments with preregistered held-out ground
truth. Their quantitative contributions are suggestive rather
than definitive. Standardized evaluation protocols for these
stages do not yet exist. For Stage 1, independent expert pan-
els could audit the same instrument and provide a basis for
measuring agreement with the LLM’s recommendations. For
Stage 5, a held-out question bank with adjudicated answers
would enable precision and recall reporting for both correct
retrieval and appropriate refusal behavior. Developing these
protocols is a concrete priority for follow-on research.
6.2.4 Synthetic Respondents: Appropriate Use and Limits
The Stage 3 results show that the best-performing LLM
configurations agree with held-out real respondents to within
roughly one ordinal level on a five-point scale. This level
of fidelity supports design-support tasks, including prototyp-
ing cross-tabulations, checking item variation, and calibrating
power expectations, but falls short of what population-levelinferential precision requires. Synthetic respondents are appro-
priately used in the design phase rather than the analytic phase.
This distinction is especially valuable in rapid-onset disaster
settings where access to target populations is constrained, but
responsible deployment requires clear articulation of these
limits and ongoing evaluation against real respondent data.
6.2.5 Privacy and Deployment Infrastructure
Routing survey microdata through commercial model APIs
raises privacy considerations that vary by jurisdiction and in-
stitutional policy. This study uses anonymized experimen-
tal data and presents the framework as a research proto-
type. Translating it into operational deployment will require
privacy-compliant infrastructure, audit-trail standards, and
data-handling protocols that most survey organizations do not
yet have in place. Locally deployed open-weight models offer
a partial path forward. Beyond privacy, the computational re-
sources and expertise required to implement LLM-augmented
workflows may be out of reach for smaller research teams,
particularly those working in resource-constrained settings.
Developing practical guidelines, including IRB protocol tem-
plates and training resources for survey methodologists, is a
collaborative task that deserves sustained institutional atten-
tion.
7. Conclusion
This paper demonstrates that large language models, em-
bedded within a theoretically grounded and architecturally dis-
ciplined framework, can make meaningful and reproducible
contributions at each stage of the survey-research process.
The contributions are specific and bounded: they are strongest
where the LLM operates within a structured evidence architec-
ture that organizes what the model can retrieve and constrains
what it can assert.
The central methodological finding is that how retrieval is
organized determines whether the model assists or misleads.
Grounding retrieval in a theory-derived causal structure and
integrating all evidence in a single model call consistently out-
performs unstructured similarity search and staged sequential
reasoning. This result suggests that productive combination of
domain theory and machine learning in social measurement de-
pends on specifying precisely how theoretical structure shapes
the evidence integration process, not merely on adding re-
trieval to a language model.
The finding that aggregate imputation metrics can mask sys-
tematic errors for specific respondent groups carries particular
weight for disaster research, where the households most diffi-
cult to reach are often those most relevant to relief planning
and recovery policy. Reporting imputation bias separately for
groups defined by their missingness risk, rather than in aggre-
gate alone, is a straightforward methodological standard whose
adoption would improve the reliability of LLM-augmented
survey analysis for policy-relevant applications.
14

Ultimately, the empirical responses of real survey partici-
pants remain the irreplaceable foundation of survey science.
Every measurement, including those augmented by LLMs, is
at best an estimate shaped by methodological choices, respon-
dent behavior, and practical constraints. What LLMs can offer,
when deployed responsibly within the kind of framework eval-
uated here, is help researchers make better use of the data
they have: more efficiently at the design stage, with greater
interpretive nuance under missing-data conditions, and more
transparently at the analysis stage. Realizing that potential will
require the same interdisciplinary rigor, ethical scrutiny, and
commitment to human oversight that the best survey research
has always demanded.
Declaration of Generative AI Use
During the preparation of this manuscript, the authors made
use of Claude Sonnet 4.6 (Anthropic) for assistance with lan-
guage refinement and proofreading. The authors have re-
viewed and edited the output and take full responsibility for
the content of this publication.
Acknowledgements
This manuscript is based on work supported by the Na-
tional Science Foundation under Grant No. 2505675 and No.
2440023. Any opinions, findings, and conclusions or recom-
mendations expressed in this material are those of the authors
and do not necessarily reflect the views of the National Science
Foundation.
References
Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting,
C., and Wingate, D. (2023). Out of one, many: Using
language models to simulate human samples.Political
Analysis, 31(3):337–351.
Arora, N., Chakraborty, I., and Nishimura, Y . (2025). AI-
human hybrids for marketing research: Leveraging large
language models (LLMs) as collaborators.Journal of Mar-
keting, 89(2):43–70.
Ashwin, J., Chhabra, A., and Rao, V . (2025). Using large lan-
guage models for qualitative analysis can introduce serious
bias.Sociological Methods & Research.
Barari, S., Angbazo, J., Wang, N., Christian, L. M., Dean,
E., Slowinski, Z., and Sepulvado, B. (2025). AI-assisted
conversational interviewing: Effects on data quality and
respondent experience. Technical report, NORC at the
University of Chicago.
Bentley, F., O’Neill, K., Quehl, K., and Lottridge, D. (2020).
Exploring the quality, efficiency, and representative nature
of responses across multiple survey panels. InProceed-ings of the 2020 CHI Conference on Human Factors in
Computing Systems, pages 1–12.
Bisbee, J., Clinton, J. D., Dorff, C., Kenkel, B., and Larson,
J. M. (2024). Synthetic replacements for human survey
data? The perils of large language models.Political Analy-
sis, pages 1–16.
Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.,
Dhariwal, P., and Amodei, D. (2020). Language models are
few-shot learners.arXiv preprint arXiv:2005.14165.
Callegaro, M., Baker, R., Bethlehem, J., Goritz, A. S., Kros-
nick, J. A., and Lavrakas, P. J. (2014). Online panel research.
InOnline Panel Research, pages 1–22. John Wiley & Sons.
Chakraborty, I. and Nishimura, Y . (2025). AI-human hybrids
for marketing research: Leveraging large language models
(LLMs) as collaborators.Journal of Marketing, 89(2):43–
70.
Coffey, S. (2024). Recent innovations and advances in mixed-
mode surveys.Journal of Survey Statistics and Methodol-
ogy, 12(3):507–528.
Couper, M. P. (2017). New developments in survey data
collection.Annual Review of Sociology, 43:121–145.
Crockett, M. J. and Messeri, L. (2025). AI surrogates and
illusions of generalizability in cognitive science.Trends in
Cognitive Sciences.
Dillman, D. A. and Christian, L. M. (2005). Survey mode as
a source of instability in responses across surveys.Field
Methods, 17(1):30–52.
Dillman, D. A., Smyth, J. D., and Christian, L. M. (2014).
Internet, Phone, Mail, and Mixed-Mode Surveys: The Tai-
lored Design Method. John Wiley & Sons, 4th edition.
Floyd, D. L., Prentice-Dunn, S., and Rogers, R. W. (2000). A
meta-analysis of research on Protection Motivation Theory.
Journal of Applied Social Psychology, 30(2):407–429.
Frontiers in Artificial Intelligence (2025). Survey and analysis
of hallucinations in large language models.Frontiers in
Artificial Intelligence, 8:1622292.
Fussell, E., Curtis, K. J., and DeWaard, J. (2014). Recov-
ery migration to the City of New Orleans after Hurricane
Katrina: A migration systems approach.Population and
Environment, 35(3):305–322.
Gao, Y ., Lee, D., Burtch, G., and Fazelpour, S. (2025). Take
caution in using LLMs as human surrogates.Proceedings of
the National Academy of Sciences, 122(24):e2501660122.
Groves, R. M. and Couper, M. P. (1998).Nonresponse in
Household Interview Surveys. John Wiley & Sons.
15

Groves, R. M. and Lyberg, L. (2010). Total survey error: Past,
present, and future.Public Opinion Quarterly, 74(5):849–
879.
Hacıo ˘glu Hoke, S., Feler, L., Mitchell, S., and Chylak, J.
(2025). Tracking consumer sentiment versus how con-
sumers are doing based on verified retail purchases.FEDS
Notes, Federal Reserve Board.
Hao, H., Wang, Y ., and Kang, S. (2022). Examining “dig-
ital” vulnerability to flooding among subsidized housing
residents in Florida.International Journal of Disaster Risk
Reduction, 82:103302.
Hays, R. D., Liu, H., and Kapteyn, A. (2015). Use of Internet
panels to conduct surveys.Behavior Research Methods,
47(3):685–690.
Horton, J. J. (2023). Large language models as simulated
economic agents: What can we learn from Homo Silicus?
arXiv preprint arXiv:2301.07543.
Huang, Y . et al. (2024). A survey on hallucination in large
language models: Principles, taxonomy, challenges, and
open questions.ACM Transactions on Information Systems,
1(1).
Johnson, C. et al. (2024). Addressing fraudulent responses in
online surveys.People and Nature, 6(2).
Jung, S.-g., Salminen, J., Aldous, K. K., and Jansen, B. J.
(2025). PersonaCraft: Leveraging language models for
data-driven persona development.International Journal of
Human-Computer Studies, 197:103445.
Kaiyrbekov, K., Dobbins, N. J., and Mooney, S. D. (2025). Au-
tomated survey collection with LLM-based conversational
agents.JAMIA Open, 8(5):ooaf103.
King, D. (2002). Post disaster surveys: Experience and
methodology.The Australian Journal of Emergency Man-
agement, 17(3):39–47.
Lazo, J. K., Bostrom, A., Morss, R. E., Demuth, J. L., and
Lazrus, H. (2015). Factors affecting hurricane evacuation
intentions.Risk Analysis, 35(10):1837–1857.
Leduc, S., Oliveira, L. E., and Paulson, C. (2025). Do low
survey response rates threaten data dependence? Technical
Report 2025-07, Federal Reserve Bank of San Francisco.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., and Kiela, D. (2020). Retrieval-augmented gen-
eration for knowledge-intensive NLP tasks.arXiv preprint
arXiv:2005.11401.
Lin, Y . et al. (2024). CrowdLLM: Building digital populations
augmented with generative models.arXiv preprint.Loru, E., Nudo, J., Di Marco, N., Santirocchi, A., Atzeni, R.,
Cinelli, M., and Quattrociocchi, W. (2025). The simulation
of judgment in large language models.Proceedings of the
National Academy of Sciences, 122(42):e2518443122.
Meade, A. W. and Craig, S. B. (2012). Identifying careless re-
sponses in survey data.Psychological Methods, 17(3):437–
455.
Minaee, S., Mikolov, T., Nikzad, N., Chenaghlu, M., Socher,
R., Amatriain, X., and Gao, J. (2025). Large language
models: A survey.arXiv preprint arXiv:2303.18223.
Miranda, F. and Balbi, P. P. (2025). Simulating public opinion:
Comparing distributional and individual-level predictions
from LLMs and random forests.Entropy, 27(9):923.
Naveed, H. et al. (2025). A survey of large language models:
Evolution, architectures, adaptation, benchmarking, appli-
cations, challenges, and societal implications.Electronics,
14(18):3580.
OpenAI (2023). GPT-4 system card. Technical report, Ope-
nAI.
Perc, M. (2025). Counterfeit judgments in large language
models.Proceedings of the National Academy of Sciences,
122(48):e2528527122.
Pinzon-Espitia, O. L. (2024). AI-powered fraud and the ero-
sion of online survey integrity.Frontiers in Research Met-
rics and Analytics, 9:1432774.
Qu, Y . and Wang, J. (2024). Performance and biases of large
language models in public opinion simulation.Humanities
and Social Sciences Communications, 11:1095.
Ravi, S., Rovira, P., Acharya, A., Petrov, S., Lee, A., Gokul,
H., and Patwary, M. (2025). HalluLens: LLM hallucination
benchmark. InProceedings of the 63rd Annual Meeting
of the Association for Computational Linguistics, pages
22154–22171.
Rubin, D. B. (1987).Multiple Imputation for Nonresponse in
Surveys. John Wiley & Sons.
Salecha, A., Ireland, M. E., Subrahmanya, S., Sedoc, J., Ungar,
L. H., and Eichstaedt, J. C. (2024). Large language models
display human-like social desirability biases in Big Five
personality surveys.PNAS Nexus, 3(12):pgae533.
Sarstedt, M., Adler, S. J., Rau, L., and Schmitt, B. (2024).
Using large language models to generate silicon samples in
consumer and marketing research.Psychology & Marketing,
41:1254–1270.
16

Saxena, D. (2025). Hallucination in generative artificial intelli-
gence: Challenges, causes, and mitigation strategies.SSRN
Working Paper No. 5976335.
Seaman, S. R., White, I. R., Copas, A. J., and Li, L. (2012).
Combining multiple imputation and inverse-probability
weighting.Biometrics, 68(1):129–137.
Slovic, P., Finucane, M. L., Peters, E., and MacGregor, D. G.
(2007). The affect heuristic.European Journal of Opera-
tional Research, 177(3):1333–1352.
Stanford Institute for Human-Centered AI (2025). Artificial
intelligence index report 2025. Technical report, Stanford
HAI.
Stekhoven, D. J. and Bühlmann, P. (2012). MissForest: Non-
parametric missing value imputation for mixed-type data.
Bioinformatics, 28(1):112–118.
Stern, M. J., Bilgen, I., and Dillman, D. A. (2014). The state
of survey methodology: Challenges, dilemmas, and new
frontiers in the era of the tailored design.Field Methods,
26(3):284–301.
Sun, S., Lee, E., Nan, D., Zhao, X., Lee, W., Jansen, B. J.,
and Kim, J. H. (2024a). Random silicon sampling: Sim-
ulating human sub-population opinion using a large lan-
guage model based on group-level demographic informa-
tion.arXiv preprint arXiv:2402.18144.
Sun, S., Schoenegger, P., Kapoor, S., Pan, L., and Narayanan,
A. (2024b). Performance and biases of large languagemodels in public opinion simulation.Humanities and Social
Sciences Communications, 11:1095.
Tang, Y ., Yang, Y ., and Abbasi, A. (2025). PersonaFuse: A per-
sonality activation-driven framework for enhancing human–
LLM interactions.arXiv preprint arXiv:2509.07370.
Tourangeau, R., Rips, L. J., and Rasinski, K. (2000).The
Psychology of Survey Response. Cambridge University
Press.
U.S. Census Bureau (2024). American community survey 5-
year estimates, 2018–2022. Technical report, U.S. Census
Bureau.
van Buuren, S. and Groothuis-Oudshoorn, K. (2011). mice:
Multivariate imputation by chained equations in R.Journal
of Statistical Software, 45(3):1–67.
Wang, L. and Fan, X. (2004). Missing data in disguise
and implications for survey data analysis.Field Methods,
16(3):332–351.
Wang, Y ., Guo, Z., and McCarty, C. (2026). Time poverty
and disaster readiness: How routine constraints shape hur-
ricane preparation.International Journal of Disaster Risk
Reduction, page 106037.
Westwood, S. J. (2025). The potential existential threat of large
language models to online survey research.Proceedings of
the National Academy of Sciences, 122(47):e2518075122.
Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y .,
and Wen, J.-R. (2025). A survey of large language models.
arXiv preprint arXiv:2303.18223.
17
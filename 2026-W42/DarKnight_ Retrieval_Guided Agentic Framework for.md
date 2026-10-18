# DarKnight: Retrieval-Guided Agentic Framework for Breaking Dark Web CAPTCHA Gates

**Authors**: Binghui Wu, Yeo Zi Yi, Darren Tan, Manit Baser, Levente Csikor, Dinil Mon Divakaran

**Published**: 2026-10-14 09:04:50

**PDF URL**: [https://arxiv.org/pdf/2610.19830v1](https://arxiv.org/pdf/2610.19830v1)

## Abstract
Cyber Threat Intelligence (CTI) investigators who monitor dark web marketplaces for leaked credentials and breach listings must pass the diverse CAPTCHAs that operators deploy to block automated collection. Existing solvers either need unscalable per-site fine-tuning or rely on commercial Vision-Language Model (VLM) APIs, which surrender operational sovereignty: provider policies, Tor-exit blocking, government regulation, or model updates outside the investigator's control can halt or alter the solver overnight, and per-call costs recur with every session and retry. We present DarKnight, a self-improving agentic framework designed to run on a single workstation GPU inside an investigator's Tor crawler and solve CAPTCHAs with small, locally hosted open-weight VLMs that no provider decision can withdraw, without per-market training or third-party services. For each CAPTCHA, DarKnight retrieves successful and failed past attempts on similar images (dual-signal RAG), composes an image-specific preprocessing pipeline, and learns from the site's pass/fail verdict. In a case study on four real-world dark web marketplaces, agentic tool selection outperformed parameter scaling. A deployable open-weight VLM with DarKnight reached 83.2\% on the most heavily defended market, 37.2 percentage points above its base model, outperforming even the frontier Claude Opus~5 (44%). DarKnight recovered 48.2% of baseline failures at a 7.6% degeneration cost, indicating that selective tool use can avoid much of the degradation caused by uniform preprocessing.

## Full Text


<!-- PDF content starts -->

DARKNIGHT: Retrieval-Guided Agentic Framework
for Breaking Dark Web CAPTCHA Gates
Binghui Wu, Yeo Zi Yi, Darren Tan, Manit Baser
National University of Singapore
SingaporeLevente Csikor, Dinil Mon Divakaran
A*STAR Institute of Advanced Intelligence and Compute
Singapore
Abstract—Cyber Threat Intelligence (CTI) investigators who
monitor dark web marketplaces for leaked credentials and breach
listings must pass the diverse CAPTCHAs that operators deploy
to block automated collection. Existing solvers either need unscal-
able per-site fine-tuning or rely on commercial Vision-Language
Model (VLM) APIs, which surrender operational sovereignty:
provider policies, Tor-exit blocking, government regulation, or
model updates outside the investigator’s control can halt or alter
the solver overnight, and per-call costs recur with every session
and retry. We present DARKNIGHT, a self-improving agentic
framework designed to run on a single workstation GPU inside
an investigator’s Tor crawler and solve CAPTCHAs with small,
locally hosted open-weight VLMs that no provider decision can
withdraw, without per-market training or third-party services.
For each CAPTCHA, DARKNIGHTretrieves successful and failed
past attempts on similar images (dual-signal RAG), composes
an image-specific preprocessing pipeline, and learns from the
site’s pass/fail verdict. In a case study on four real-world dark
web marketplaces, agentic tool selection outperformed parame-
ter scaling. A deployable open-weight VLM with DARKNIGHT
reached 83.2% on the most heavily defended market, 37.2
percentage points above its base model, outperforming even the
frontier Claude Opus 5 (44%). DARKNIGHTrecovered 48.2%
of baseline failures at a 7.6% degeneration cost, indicating that
selective tool use can avoid much of the degradation caused by
uniform preprocessing.
Index Terms—Dark web, VLM, Agentic, Threat Intelligence
I. INTRODUCTION
Dark web marketplaces demonstrate resilience against law
enforcement efforts, with new platforms rapidly emerging to
replace those that are disrupted [1]. The scale of illicit activity
continues to grow: data breaches posted on underground fo-
rums increased by 43% in 2024, compromised credentials rose
by 34%, and the number of ransomware group leak sites grew
by 53% over the same period [2]. Beyond data and credentials,
dark web forums host exploit listings that are subsequently
linked to known vulnerabilities in production systems [3].
Phishing kits are openly traded and deployed at industrial
scale [4], [5]. The emergence of malicious LLM services, sold
openly on underground markets to lower the barrier to crafting
phishing lures and malware, further amplifies the threat [6].
Cryptocurrency facilitates pseudonymous settlement of nearly
all transactions across these markets [7].
Security teams monitor dark web marketplaces to detect,
as early as possible, newly listed data breaches, emerging
ransomware strains, and exposed customer records before they
are exploited [1], [8]. The value of this monitoring scaleswith coverage and speed, which makes automated, large-scale
collection a requirement rather than a convenience. To resist
such automation, marketplaces deploy several defensive layers,
including headless-browser blocking, dynamic link generation,
request-rate limiting, and CAPTCHAs [9]–[11]. Most of these
layers can be circumvented through careful crawler engineer-
ing [11], [12]. CAPTCHAs are the exception. Deployed at
registration, login, and checkout, they are designed to admit
humans and turn away bots [13], [14]. It is the chokepoint
that if overcome cheaply and at scale, makes automated dark-
web measurement feasible; otherwise collection stalls behind
manual intervention [10].
What sits at that gate is not a clear web CAPTCHA.
Mainstream services such as reCAPTCHA, hCaptcha, and
Cloudflare Turnstile are largely absent from dark web sites for
structural reasons [13]–[16]. They require the client to contact
a third-party endpoint the site operator does not control. On a
hidden service, that adds the risky de-anonymization layer and
logging surface; neither do legit CAPTCHA provider serve the
dark web market. Moreover, those endpoints also degrade or
block Tor exit traffic. For these anonymity reasons mainstream
CAPTCHAs are not used on the dark web [14]. The visitors
compound the issue: many Tor users disable JavaScript to
shrink their fingerprinting surface, which makes the behav-
ioral, slider, and interactive challenges common on the clear
web impractical to deploy. Dark web site operators therefore
build their own [9], [10], [13]; the result is a population
of bespoke, locally hosted, server-rendered CAPTCHAs [13].
Due to the lack of JavaScript, they tend to be less diverse
than commercial offerings using standard image libraries,
random fonts and additive noise. The answer is bound to
a server-side session, often at low character entropy. The
dominant form is the OCR text CAPTCHA. It appears in a
few recurring variants such as distorted Latin alphanumeric
strings, image-rendered arithmetic, and non-Latin script. Non-
Latin rendering is a low-cost anti-OCR measure on regionally
targeted markets. It defeats common off-the-shelf recognizers
trained predominantly on Latin text.
Existing empirical evaluations of modern CAPTCHAs study
the clear web systems above [17], [18]. They explicitly
place dark web challenges out of scope [19], leaving this
design space comparatively unmeasured. Prior works approach
solving dark web CAPTCHAs from two directions, neither
of which fits a dark web Cyber Threat Intelligence (CTI)
arXiv:2610.19830v1  [cs.CR]  14 Oct 2026

deployment. The first trains a recognizer per CAPTCHA type.
CNN (Convolutional Neural Network) and GAN (Generative
Adversarial Network) solvers reach high accuracy on the styles
they are trained on [10], [13], [14]. However, they require
labeled data and retraining for each new style. Besides, they
do not generalize across the heterogeneous, churning set of
markets a collector must track. The second is general visual
solving with vision-language models (VLMs). Currently, they
can solve a range of clear web CAPTCHAs [17], [20], [21].
On the dark web, a VLM has so far been used only as
a component: an off-the-shelf model embedded in a single-
market navigation tool [11]. There, solving is a step in data
acquisition, not the object of study. Consequently, prior dark-
web work has not characterised how VLM-based solvers
behave across model scales, preprocessing pipelines, or agent
designs. These are the questions a measurement study of
CAPTCHA solving must answer. Neither direction studies
dark web CAPTCHA solving as a problem in itself. The
unstudied space spans heterogeneous market styles, including
the non-Latin scripts of regionally targeted markets. It sits
under the operational constraints of CTI deployment. And it
calls for a solver that adapts without per-type retraining.
Frontier commercial LLM APIs are not good options for
a CTI solver, because they leave the investigator without
operational AI sovereignty: control over whether, how, and at
what cost the solver keeps running. Even a frontier model that
solves every CAPTCHA today would have to be removed from
the pipeline by decisions the investigator does not control.
First, provider usage policies prohibit using the models to
circumvent CAPTCHAs, and such policies can be tightened at
any time. Second, regulation can cut access without warning:
in June 2026, a US export-control directive led Anthropic to
suspend access to two of its frontier models for all customers
worldwide for about three weeks [22]. Third, providers decide
which networks they serve. When the collection host routes
its traffic through Tor, major providers challenge or block
Tor exit traffic with access-denied errors and forced human-
verification prompts, so reachability cannot be assumed [23].
Moreover, Tor rotates its exit endpoint regularly; to a provider
this resembles an account hopping over IP addresses, which
injects additional verification mid-run and stalls an unattended
pipeline until a human intervenes. Fourth, providers update
and retire model versions on their own schedule, and solving
behavior changes with them: on our most heavily defended
dataset, four Claude models range from 1.4% to 44% accuracy
(cf. App. D). Finally, the per-call cost recurs with every session
and retry, since several attempts may be needed before a page
opens. Closed APIs are therefore not a practical option (nor
do they perform well; see App. D); the deployed solver must
be open-weight and self-hosted, so that the investigator runs a
fixed model checkpoint that no provider decision can withdraw
or alter.
Two further requirements follow from the field setting. The
solver must be small enough to run on a single workstation
GPU (see Sec. VI-A for details) for CTI deployments (≤30B
models) rather than a GPU cluster in a datacenter. The solverhas one hard timing constraint and one soft one. The hard
constraint is server-side session expiry. A correct answer ar-
riving after the session has timed out is rejected, and there is no
client-side workaround [13]. The soft constraint is submission-
rate detection. Markets that escalate on rapid submission can
be defeated by holding the solver output and submitting after a
randomised delay. The hard constraint binds the solver design.
The soft constraint binds only the submission policy.
We identify three gaps that this paper addresses.
G1Small open-weight VLMs for solving CAPTCHAs
from the dark web within the operation constraints are
understudied.Prior dark web CAPTCHA solvers use per-
style trained models that require labelled data per market [10],
[14]. Other work treats solving as a downstream step in a navi-
gation pipeline [11]. Modern VLM-based solvers are evaluated
on the clear web, predominantly with frontier-closed or large-
parameter models [20], [21]. The capability of small open-
weight VLMs in solving real dark web CAPTCHAs under
time-bounded operational constraints remains understudied.
G2Understudied deployable enhancement methods for
open-weight VLMs.Prior work can be categorized broadly
into two approaches: per-style retraining and frontier-API
agentic solving. No prior method combines agentic enhance-
ment with a deployable open-weight backbone.
G3Non-Latin and heterogeneous-style CAPTCHAs from
dark web are yet to be addressed.Existing dark web
CAPTCHA work targets English-language, Western markets,
and Latin-script inputs [10], [13], [14]. Recent threat intel-
ligence reports identify active Chinese-language underground
markets as hubs that preserve operational anonymity across
clearnet, darknet, and messaging platforms [24]. The study of
any solver, classical or VLM-based, on non-Latin dark web
CAPTCHAs is missing in literature.
We argue that these gaps leave us with a single viableprofile
to address the challenges: an open-weight solver that runs
locally, stays small, and operates within a timing window. We
hypothesize that the bottleneck is not model capacity but tool-
selection strategy, informed by past solves.
To this, we design and develop DARKNIGHT, a self-
improving agentic solver. DARKNIGHTis given a small li-
brary of image-preprocessing tools (see Sec. V). For each
CAPTCHA, DARKNIGHTperceives the input and retrieves
past attempts on visually similar images. Then, it composes a
preprocessing pipeline, solves the cleaned image, and updates
from the site’s verdict [25]. The same open-weight VLM
does both planning and solving, with no additional model to
host. That closed verdict-feedback loop enables stream-level
improvement without retraining.
We present this work as a case study of one framework
on four marketplaces, evaluated offline; where we generalize,
we show what is feasible, not what holds for every market or
model. Our primary contributions are summarized as follows:
1) We design DARKNIGHT, an agentic enhancement layer
for open-weight small VLMs operating under dark web
deployment constraints. DARKNIGHTcombines chain-of-
thought visual analysis, a dual-signal RAG knowledge
2

database (retains both successful and failed records of
CAPTCHA solving, §V-C), five preprocessing operations,
and automated ground-truth feedback (§V-E). Across
2,000 CAPTCHAs from four real-world dark web mar-
ketplaces and nine model backbones, DARKNIGHTim-
proves accuracy in 30 out of 32 (93.75%) configurations,
with an average absolute improvement of 13.33%. In
particular, on the most challenging dataset (D 1), the
average improvement across all backbones is +30.48%.
Such improvement on all four marketplaces is achieved
without any per-site retraining, with a single configuration
handling alphanumeric, math, and Chinese CAPTCHAs
within the same mixed-style market.
2) We present the first systematic evaluation of small open-
weight VLMs (1B–30B) on real dark web CAPTCHAs
under operational constraints. We show that ageneral-
purpose27B model equipped with DARKNIGHT(83.2%)
onD 1outperforms the raw baselines of Step3-VL-
10B [26] (63.6%), the strongest bare model we evaluated,
alargergeneral-purpose vision language model, Qwen3-
VL-30B-A3B (42.4%), and even a frontier model Claude
Opus 5 (44%), showing that adding DARKNIGHTto a
lightweight model beats scaling up the bare model.
3) We identify and quantify the degeneration-recovery effect
in CAPTCHA preprocessing. Across all Planner deci-
sions, including cases where no preprocessing is selected,
DARKNIGHTrecovers 48.2% of baseline failures while
degrading only 7.6% of correct predictions, yielding a
5.8×recovery-to-degeneration ratio. The Planner selects
no preprocessing in 86.9% of cases onD 2and only 26.6%
on the heavily defendedD 1, indicating that it intervenes
selectively rather than uniformly.
To support reproducibility, we publicly release the complete
source code, experiment configurations, and all four annotated
CAPTCHA datasets1.
II. RELATEDWORK
A. Dark Web Analytics and Anti-Crawling Mechanisms
Dark web marketplaces continuously deploy robust anti-
crawling defenses to shield illicit trade operations from au-
tomated CTI monitoring [27]. Unlike the clear web, where
operators frequently deploy sophisticated, behaviorally driven
validation modules (e.g., reCAPTCHA or hCaptcha), dark web
administrators operate under strict Tor network constraints
that fundamentally preclude third-party JavaScript execution
to prevent user de-anonymization [13]–[16]. Consequently,
marketplaces rely predominantly on bespoke, locally hosted,
and server-rendered CAPTCHAs. These mechanisms are char-
acterized by aggressive background noise, hyper-kerning, vari-
able character lengths, and mathematical puzzles intentionally
engineered to defeat conventional scraping tools [9].
B. Classical ML-Based CAPTCHA Solvers
Early automated CAPTCHA resolution relied on specialized
computer vision and deep learning pipelines [28], [29]. Even
1https://github.com/Binghui99/DarKnightprior to the emergence of multimodal foundation models,
researchers demonstrated that machine learning systems could
successfully attack visual reasoning CAPTCHAs that were
explicitly designed to resist automated solving [30]. For dark
web CTI, frameworks like DW-GAN [10] and “Tick Tock
Break the Clock” [14] achieve high success against bespoke
challenges. However, they are operationally brittle, as they
overfit to specific data distributions and require labor-intensive
manual retraining whenever marketplace administrators update
their visual obfuscation logic. We do not compare against them
on accuracy. Given a labeled training set for a target style these
solvers score highly, but that assumption is exactly what fails
in CTI: a collector tracking churning, heterogeneous markets
has no per-style labels at the moment of solving and cannot
retrain as styles change. DARKNIGHTtargets this regime,
achieving deployable accuracy without per-style supervision.
C. Generalized VLM Solvers and Agentic Frameworks
Recent works have demonstrated the capability of VLMs
in zero-shot multimodal reasoning. Recent benchmark suites
such as MCA-Bench [17] and CAPTURE [18] systemat-
ically evaluate the effectiveness of modern VLMs against
diverse CAPTCHA schemes. MCA-Bench [17] demonstrates
that fine-tuned VLMs can consistently overcome cross-modal
verification challenges, while CAPTURE [18] provides a
dedicated evaluation framework for assessing LVLM perfor-
mance on CAPTCHA-solving tasks. More recently, Cogni-
tion provides a large-scale evaluation of multimodal LLM
CAPTCHA-solving capabilities and investigates defensive
strategies against emerging MLLM-based attackers [31]. State-
of-the-art agentic frameworks have further operationalized
these capabilities against complex security mechanisms. Hal-
ligan is an agentic VLM framework, which reduces intricate
visual CAPTCHAs into heuristic search problems overlaid
onto graphical user interface action spaces [20]. Concur-
rently, the Oedipus framework utilizes an LLM-enhanced
reasoning pipeline to parse and bypass advanced logic-
based CAPTCHAs using a specialized Domain Specific Lan-
guage [21]. Despite their adaptability, these frontier agentic
solvers uniformly rely on commercial cloud APIs for infer-
ence. This reliance violates the strict operational requirements
of dark web CTI deployments discussed above. As with the
trained solvers above, we therefore cannot run these frontier-
API agents under our deployment constraints, and exclude
them from our evaluation.
III. THREATMODEL
Setting.We consider a security investigator who has to
automate CAPTCHA solving on dark web marketplaces for
scalable CTI gathering (see Fig. 1). The CAPTCHA’s purpose
is to gate automated access; the investigator’s goal is to
bypass that gate at scale. We study the defender (the site’s
CAPTCHA system) by playing the attacker (the automated
solver), for the larger social good. From the marketplace
operator’s perspective, the investigator is a bot. We discuss
ethical considerations in Appendix A.
3

CTI Investigator Dark-web site
Crawler
Local VLM     
Memory
StoreCAPTCHACAPTCHA
generatorVisitTor Network
Session /
Access control
Protected
ContentBinary feedbackCandidate solutionFig. 1: Threat model
Investigator.The investigator operates an automated crawler
over Tor. The CAPTCHA solver interacts with the target site
through a standard browser session. It receives CAPTCHA
images as rendered in the browser and submits candidate
text solutions. Each submission yields binary feedback: the
protected page loads (solution accepted) or a new CAPTCHA
is served (solution rejected). The solver has no view of the
CAPTCHA generation algorithm, character set, font library,
or distortion parameters. No labelled dataset of marketplace-
specific image and solution pairs is available at the start.
The solver may, however, accumulate experience over time. It
may record candidate solutions, outcomes, and preprocessing
strategies that succeeded or failed on visually similar images.
However, this memory is feedback-derived, not a labelled
training set. Per Sec. I, the solver must be open-weight, locally
hosted, and resource-bounded.
Site operator.The site operator controls the full CAPTCHA
stack: generation algorithm, fonts, distortion, script choice,
length, and presentation. The operator also controls the server-
side access-control policy. The operator enforces request-
frequency thresholds. Exceeding a threshold triggers escalating
countermeasures. These include serving harder CAPTCHAs,
invalidating sessions, and blocking the current Tor exit IP [23].
Exit-IP blocking forces the client to rotate circuits, since
circuit control belongs to the client. The operator may also
update CAPTCHA designs over time in response to observed
bypassing attacks.
Scope and assumptions.This work targetstext-based
CAPTCHAs, the dominant variant on dark-web market-
places [10], [13]. The text-based scope spans Latin alphanu-
meric, arithmetic, and non-Latin (e.g., Chinese) variants ob-
served in our data. For this current work, we exclude slider
puzzles, image-selection grids, audio CAPTCHAs, and any
challenge requiring JavaScript execution beyond text render-
ing. We assume the Tor Browser renders CAPTCHA images
faithfully; i.e., what the solver receives matches what a human
user would see. We assume the operator does not adversari-
ally adapt CAPTCHA generation to our specific system. Tor
network availability and anonymity are out of scope. A solve
is successful when the candidate solution is accepted and the
protected page loads. All experimental results are obtained
offline against a static labeled dataset (Section VI-B). Live-
deployment characteristics that depend on operator adaptation
are not exercised in this evaluation.IV. DESIGNRATIONALE
Below we present the design principles behind DARK-
NIGHT(Sec. V).
1From Monolithic Solvers to Decomposable Modules.
A single VLM invocation that simultaneously analyzes a
CAPTCHA to perform multiple operations (e.g., removing
background noise and OCR), treats solving as a monolithic
task. A CAPTCHA-solving framework should therefore de-
compose the task into distinct perception, planning, and ex-
ecution stages, allowing each component to focus on a well-
defined objective. This modular design improves controlla-
bility over individual components and enables targeted error
diagnosis. This principle motivates the three-phase perceive-
plan-execute architecture described in Sec. V.
2Utilizing Transformation Experts.Beyond staging the
pipeline, the question is who performs each step. Identifying
visual characteristics is a perception task suited to VLMs; the
resulting image transformation, by contrast, is typically deter-
ministic. VLM should be used to analyze visual distortions
and interference patterns, while preprocessing operations such
as denoising, line removal, and binarization should be dele-
gated to classical computer vision techniques. This principle
motivates the separation between the Planner’s visual analysis
component (Sec. V-B) and the deterministic preprocessing
toolbox (Sec. V-D).
3Deterministic Tasks are Programmable.Separately from
image preprocessing, the symbolic steps after recognition are
also deterministic. Once text is extracted, many downstream
operations such as evaluating an arithmetic expression can
be solvedprogrammaticallyrather than by the model. Pro-
grammable tasks thus should be delegated to dedicated tools,
reserving the VLM for perception and interpretation while
avoiding unnecessary sources of stochastic error. This prin-
ciple motivates the Executor’s use of a dedicated arithmetic
solver described in Sec. V-D.
4Learn from Experience.CAPTCHAs from the same
marketplace often share recurring visual characteristics, so
a preprocessing strategy that succeeds or fails on one is
likely to behave the same on similar ones. A solver should
therefore retain both successful and unsuccessful attempts,
and retrieve visually similar records when handling a new
CAPTCHA, reusing what worked and avoiding what degraded
performance. This principle motivates the dual-signal RAG
knowledge base (Sec. V-C).
5Plug-and-Play for Scalability.Dark web CAPTCHA
styles evolve continuously, making repeated model retraining
impractical for operational deployments. The preprocessing
capabilities should comprise modular tools rather than learned
model behaviors. The framework is designed so that a new
tool can be added without modifying the underlying model
weights; we leave empirical validation of adding new tools
to future work. This plug-and-play design may facilitate rapid
adaptation to emerging defenses, and enable scaling without
incurring the cost of continual retraining. This principle mo-
4

tivates the modular toolbox and feedback-driven adaptation
mechanisms described in Sec. V-D and V-E.
V. THEDARKNIGHT FRAMEWORK
A. System Overview
Fig. 2 illustrates the architecture of DARKNIGHT. Given a
CAPTCHA image from a dark web marketplace, the system
operates in a three-phase loop:plan,execute, andrecord.
i) In theplanphase, the Planner Agent receives the raw
CAPTCHA image and produces a structured visual analysis
describing the image’s properties. This analysis is used to
query the RAG Knowledge Base, which retrieves the most
relevant past solving records (both successes and failures)
based on visual and semantic similarity. The planner agent then
uses the retrieved records alongside its own visual analysis
and composes a preprocessing pipeline tailored to the current
image. ii) In theexecutephase, The Executor runs this pipeline
step-by-step from the toolbox, passes the processed image
to the OCR engine and submits the recognized text (or the
solution of an equation) to the target website. iii) In therecord
phase, the website returns a binary pass/fail verdict. The com-
plete attempt, including image, visual analysis, pipeline, OCR
output, and outcome, is written back to the RAG Knowledge
Base regardless of the result, closing the feedback loop.
CAPTCHA Database 
Planner Agent （Vision LLM ）
Visual Analysis 
Chain of Thought 
Action Pipeline Similarity Check Description 
Image encoder 
Top-k recall Image 
Executor Agent 
Tool execution 
Output: compare 
with ground truth 
(pass/fail) 
Toolbox 
●color_extract 
●remove_line 
●denoise 
●binarize 
●sharpen 
Record {"noise":        "none", 
 "background":   "clean", 
 "distortion":   "none", 
 "overlap":      "none", 
 "char_quality": "normal", 
 "captcha_type": 
"alphanumeric", 
 "char_count":   4, 
 "is_math":      false, 
 "is_chinese":   false, 
 "description":  "..."} Sentence Transformer 
OCR 
(Vision LLM) math Math 
Solver 
Fig. 2: Architecture Diagram of the pipeline
As the system encounters more CAPTCHAs from the same
marketplace, the RAG store accumulates site-specific experi-
ence. This enables the Planner to progressively make better
tool-selection decisions based on the past experience,with no
model retraining or fine-tuning. This is the key design property
that distinguishes DARKNIGHTfrom prior per-style trained
solvers. A critical design choice is that the Planner VLM
servesdual roles: it performs the visual analysis and planning
in the perceive phase, and also serves as the OCR engine in
the execute phase. Since the solver is already loaded, routing
planning through it adds no memory overhead. This keeps the
system lightweight since only a single model is needed to
be loaded, even if VLM invocation happens twice, resulting
in the same memory requirements as a naive single VLM
execution pipeline. The full agentic loop introduces additional
local inference time overhead. Section VII-E quantifies this
computational cost across the evaluated VLMs.B. Planner Agent
The Planner Agent is a lightweight, open-weight VLM that
operates in three stages to perform visual analysis, experience
retrieval, and pipeline planning.
Stage 1: Visual Analysis.The Planner receives a raw
CAPTCHA image and outputs a structured JSON describing
eight visual properties shown in Table I. This structured output
serves two purposes: i) it informs the RAG retrieval query
(Stage 2) and ii) it provides the Planner with an explicit rep-
resentation of the image’s properties for reasoning in Stage 3.
TABLE I: Visual features extracted by Stage 1 visual analysis.
Features Values
captcha_type alphanumeric|math|chinese
char_countInteger count of CAPTCHA characters
noise none|color_lines: multiple coloured wavy/diagonal
lines of a different hue from the text|strikethrough:
one thin dark line crossing all characters|arc: dark
curved or arc-shaped bands|vertical_lines: thin dark
vertical strokes through characters|dots: scattered isolated
dot clusters|salt_pepper: fine pixel-level random grain
background clean|gradient|noisy|patterned
distortion none|wave|blur|perspective
char_quality normal|hollow|faded|thin|broken|thick
is_math, Boolean flag ifcaptcha_typeis math
is_chineseBoolean flag ifcaptcha_typeis chinese
descriptionFree-text sentence for RAG text-embedding similarity
Stage 2: Experience Retrieval.Using the structured anal-
ysis from Stage 1, the system queries the RAG Knowledge
Base (Sec. V-C) for past solving records on visually similar
CAPTCHAs. The query uses two signals: text similarity of
the image description and perceptual hash [32] similarity
of the image itself. The top-krecords, annotated with their
preprocessing pipelines and website pass/fail outcomes, are
appended to the Planner’s context for Stage 3. This stage is
automated and does not require additional VLM inference.
Stage 3: Pipeline Planning.The Planner receives its own
visual analysis (Stage 1) and the retrieved records (Stage 2)
as context, and produces a complete processing plan. The plan
specifies two components: (i)preprocess_pipeline, an
ordered list of tools to apply from the toolbox (Sec. V-D), each
with its parameters (e.g.,binarizewithmethod=otsu);
and (ii)post_ocr, a flag indicating whether to activate the
math expression solver after OCR. The Planner is prompted
with a system instruction that defines the available tools, their
expected inputs and effects, and the output JSON format.
When retrieved records are available, the Planner is instructed
to prefer strategies that previously succeeded on similar im-
ages and avoid strategies that previously failed. Table XV
(Appendix. F) summarizes the API parameters for VLM calls.
C. Dual-signal RAG Knowledge Base
The optimal preprocessing strategy for a given CAPTCHA
cannot be deduced from static visual analysis alone. Hence,
DARKNIGHTmaintains a persistent dual-signal RAG knowl-
edge base of past execution records, which the Planner
queries dynamically at inference time. Such RAG knowledge
base retainsbothsuccessful and failed solving attempts in
5

distinct partitions (Table II), bounded to 100 records per
experimental configuration. Successful records function as
positive blueprints, guiding the Planner to replicate proven
pipelines when encountering matching visual features. Failed
records serve as negative constraints, explicitly warning the
Planner against repeating historically destructive tool se-
quences. This dual-signal retrieval design directly addresses
the degeneration-recovery trade-off (Sec. VII-B). Rather than
blindly applying an operation and discovering at test time
that it degrades a specific CAPTCHA archetype, the Planner
actively retrieves prior failure evidence to proactively prune
hazardous tools from its execution plan. In all experiments,
the RAG store applies a cold-start protocol. It begins empty
and is populated incrementally as each dataset is processed
in a single sequential pass: the first image is solved with no
retrieved experience, and each subsequent image benefits from
the records accumulated by all preceding attempts.
1) Storage Layout:The RAGStore adopts a file-per-
combination storage scheme: one JSON file is created for each
unique [dataset, planner model, ocr model] triplet, stored
underrag_store/with the naming convention:
{dataset}__{planner}__{ocr}.json
For the primary per-dataset experiments, this storage scheme
ensures isolation between different configurations, and no
cross-dataset retrieval is performed. Each file holds up to200
records, distributed across two independent slots as shown in
Table II: up to 100 successful and 100 failed records. This
design allows us to measure the effect of DARKNIGHTinde-
pendently on each marketplace. In Sec. VII-D, we additionally
evaluate a mixed-site setting in which CAPTCHAs from all
four marketplaces are processed sequentially using a single
shared RAG store for each evaluated backbone.
TABLE II: RAGStore slot structure.
Slot Content Capacity Eviction Policy
correctSuccessful predictions 100 records Least-retrieved first
failedFailed predictions 100 records Least-retrieved first
2) Quality-Based Eviction:Each record tracks a
_retrieval_countfield that is incremented each time the
record is surfaced to the planner. When a slot reaches capacity,
the system applies a least frequently used (LFU) eviction
policy, displacing the record with the lowest cumulative
retrieval count [33]. This policy is self-reinforcing as records
that frequently match new queries survive indefinitely, while
records that are never retrieved and thus never contribute to
planning, are replaced by more relevant newcomers. The net
effect is a memory that organically specializes toward the
most frequently encountered CAPTCHA configurations.
3) Deduplication:On everyadd()call, the system com-
putes the weighted similarity between the incoming record and
all existing records in the target slot. If any existing record
has a similarity score≥0.99, the insert is suppressed. Thisprevents near-duplicate examples from crowding out diverse
records and preserves the breadth of the memory.
4) Similarity Metric:The composite similarity score com-
bines two signals:
s= 0.7·cos(f q,fr) + 0.3·cos(e q,er)(1)
wheref∈ {0,1}192is the perceptual hash (pHash-RGB)
of the image ande∈R384is the sentence-transformer
embedding of the free-text description. The pHash is computed
over the RGB channels of the image (pHash-RGB) [32], which
captures structural and color gradient patterns without storing
the full image. The free-text description is embedded using
all-MiniLM-L6-v2[34], [35], a lightweight sentence-
transformer that maps visual descriptions to a dense semantic
space. Feature cosine (weight0.7) dominates because it is
deterministic and encodes the exact visual properties that drive
tool selection. Text cosine (weight0.3) contributes semantic
context: two CAPTCHAs described as “wavy blue lines over
Chinese characters on a gradient background” may share
only partial feature overlap but should still trigger similar
preprocessing decisions. Records whose weighted similarity
falls belowτ= 0.7(RAG_SIMILARITY_THRESHOLD) are
dropped before being returned to the planner.
5) Split Retrieval:Thequery_split(description,
features, top_k)method searches thecorrectand
failedslotsindependentlyand returns the topk/2records
from each. We setk=5throughout all experiments, retrieving
up to two successful and two failed records per query (with
one additional record from whichever slot has the higher top
match). This guarantees balanced exposure to positive and
negative examples regardless of the relative sizes of the two
slots. Each retrieved record is presented to the planner with
the following fields:
•similarity score: the weighted score from Equation (1).
•description: the free-text visual description.
•tools applied: the ordered list of preprocessing tools
used in that run.
•slot:correctorfailed, indicating the outcome.
•note: a natural-language annotation “✓succeeded repli-
cate if features match” or “×failed avoid the same tools
on similar images”.
•ocr result: the raw OCR output, included only for
failedrecords to assist diagnosis.
Relevance thresholds.Table III summarizes the threshold
policy applied to retrieved RAG records. Correct-slot records
are adopted, verified, or ignored according to their similarity
to the query; failed-slot records serve as negative signals
regardless of similarity, and a tool that appears repeatedly in
failed records is treated as banned for that CAPTCHA style.
Positive and negative records.A distinguishing feature of
the RAG store is that it retainsbothsuccessful and failed
attempts. Failed records are explicitly valuable: when the
Planner sees that a particular pipeline failed on a visually
similar CAPTCHA, it avoids repeating the same strategy. In
the prompt, failed records are formatted as “the following
6

TABLE III: Planner policy for retrieved RAG records.
Slot Similarity Action
Correct≥0.8Adopt pipeline if features match
0.6–0.8Verify each feature before copying
<0.6Ignore entirely
Failed≥0.6Avoid same tools
0.5–0.6Still a negative signal
Repeated Treat that tool as banned
pipeline was tried on a similar image and FAILED: [...]” to
clearly signal that the strategy should be avoided.
D. Executor and Toolbox
The Executor receives the Planner’s pipeline specification
and runs it in three phases.
Phase 1: Image Preprocessing.The toolbox provides five
preprocessing operations to neutralise a specific class of visual
degradation commonly encountered in CAPTCHA images.
Table IV summarises the tools, their target degradation, and
the conditions under which each should be applied.
TABLE IV: Preprocessing tools and their intended use cases.
Tool Description and Usage Condition
color_extractRemoves colored interference lines by iso-
lating non-color pixels. Use when wavy or
diagonal lines of a distinct color cross the
text region.
sharpenRestores character edge crispness via un-
sharp masking. Use when blur distortion
has softened character boundaries.
denoiseRemoves small connected components be-
low an area threshold. Use to clean residual
noise dots or on patterned backgrounds.
remove_lineDetects and removes a single thin dark
diagonal or near-horizontal line via the
Hough transform. Use when such a line
crosses all characters.
binarizeConverts a faded-grayscale image to black-
and-white using Otsu’s global thresh-
old. Useonlywhenbackground =
"clean".
Three hard ordering constraints govern pipeline composition
and are surfaced as explicit rules in the Planner prompt:
1)color_extractmust precedesharpenand
binarize, as both destroy the color information on
which color-based line removal relies.
2)denoiseis applied aftercolor_extractto elimi-
nate residual dot artefacts produced by color separation.
3)binarizationis used only whenbackground =
‘‘clean’’; applying Otsu’s global threshold to a struc-
tured or gradient background corrupts the result.
The Planner may select zero, one, or several tools depending
on the visual analysis, and the tools are designed to be freely
composable within these constraints.
As shown in Fig. 3, for instance,color_extract→
denoise→binarizeis a valid three-step pipeline for a
Before
 After
(a) Color Extract onD 1Chinese CAPTCHA.
Before
 After
(b) Denoise onD 4
Fig. 3: Effective use of preprocessing tools.
clean-background CAPTCHA overlaid with colored lines and
residual dot noise. Conversely, misapplied preprocessing oper-
ations cause more visual obfuscation, making the CAPTCHA
even harder for the underlying VLM (see the comparison in
Fig. 4). These cases concretely illustrate design principle 4
(Sec. IV): the same tool can help one CAPTCHA archetype
and harm another, so tool selection must be guided by retrieved
success and failure records rather than applied uniformly. This
is the trade-off the dual-signal RAG store (Sec. V-C) is built
to manage, and which we quantify in Sec. VII-B.
Before
 After
(a) Remove Line onD 2
Before
 After
(b)FailedBinarize onD 2
Fig. 4: Effective vs. degenerative use of preprocessing tools.
Phase 2: OCR engine.After preprocessing, the Executor
invokes the same VLM used for planning, now with a type-
specific OCR prompt:
•Alphanumeric prompt: instructs the model to output ex-
actly the characters visible in the image, handling mixed
letters and digits.
•Math prompt: instructs the model to extract the arithmetic
expression as written (e.g.,3+5=?), without solving it.
•Chinese prompt: instructs the model to output each char-
acter independently, without inferring semantic meaning
between characters.
The prompt type is determined by thecaptcha_typefield
from the Planner’s Stage 1 analysis.
Phase 3: Post-processing.For math CAPTCHAs, the OCR
output (e.g.,3+5=?) is passed to a rule-based arithmetic
solver. Before evaluation, the system validates the expres-
sion: it must contain at least one arithmetic operator and
7

an equals sign or question mark, and must not contain al-
phabetic characters. If validation fails, the system reclassifies
the CAPTCHA as alphanumeric and re-runs OCR with the
alphanumeric prompt. This fallback catches misclassifications
where the Planner incorrectly identified a CAPTCHA as a
math expression.
One additional recovery mechanism: If the Planner labels a
CAPTCHA asalphanumericbut the OCR output contains
=or?, the executor automatically attempts to solve the math
before returning a result.
E. Ground-truth Feedback Loop and Self-Improvement
After each solve attempt, the Executor writes a complete
record to the RAG Knowledge Base:
{image_hash, description, pipeline, params,
ocr_output, outcome, timestamp}
This write occurs for both passed and failed attempts. The re-
sulting feedback loop drives DARKNIGHT’s self-improvement
through three mechanisms:
Positive reinforcement.When a pipeline succeeds, future
CAPTCHAs with similar visual properties will retrieve this
record as a high-confidence reference, biasing the Planner
toward the same strategy. Over consecutive images from the
same site, successful strategies are reinforced and converge.
Negative avoidance.When a pipeline fails, the record is
stored with aFAILlabel. Future retrievals surface this record
alongside successful ones, explicitly informing the Planner
that the failed strategy should be avoided. This is particularly
important for addressing the degeneration-recovery trade-off:
if a preprocessing step degraded a similar image in the past,
the Planner learns to skip it.
Cold start and warm-up.When DARKNIGHTfirst encounters
a new marketplace, the RAG store contains no records for that
site’s CAPTCHA style. In this cold-start scenario, the Planner
relies solely on its own visual analysis (Stage 1) without
retrieved experience. As attempts accumulate, typically within
5-10 images as per our observation, the RAG store builds
sufficient site-specific history for retrieval to become effective.
Cross-site adaptation.Because retrieval is based on visual
similarity rather than site identity, experience can be shared
across marketplaces with similar CAPTCHA characteristics.
We empirically evaluate this capability using the mixed-site
experiment in Sec. VII-D.
VI. EXPERIMENTSETUP
A. Models and Hardware Configurations
We evaluate nine models across three categories: one OCR
specialist, five general-purpose instruct models, and three
general-purpose thinking models.
HunyuanOCR [36] is a 1B end-to-end OCR specialist
covering multiple languages and document types. It is the
lowest-cost point in our comparison and represents the OCR-
specialist category.
The Qwen3-VL family [37] contributes three instruct mod-
els at 4B, 8B, and 30B-A3B. The first two are dense; the thirdmodel is a mixture-of-experts model with 3B active parameters
per forward pass.
Gemma4-E4B-it and Gemma4-12B-it [38] are instruct mod-
els having 8B total parameters with 4.5B effective and 12B
parameters (all effective), respectively.
Qwen3.5-9B [39] and Qwen3.6-27B [40] are native mul-
timodal thinking models that integrate vision and language
through early fusion.
Step3-VL-10B [26] is a 10B thinking model pairing a
Perception Encoder with a Qwen3-8B backbone. It introduces
Parallel Coordinated Reasoning (PaCoRe) for test-time com-
pute scaling.
All experiments were conducted on a single NVIDIA
RTX Pro 6000 GPU (96 GB VRAM) running Debian 13
(trixie) with NVIDIA driver 610.57.04. To simulate a resource-
constrained production environment, we capped GPU memory
at 30% of its total capacity ( 29 GB), using vLLM (v0.25.1)
to enforce this limit. All models were evaluated in their
official, unmodified form: Qwen3-VL and Qwen3.5 (Qwen),
Gemma4 (Google), and Step3-VL (StepFun). No third-party
quantization, fine-tuning, or other modifications were applied.
B. Datasets
We evaluate DARKNIGHTon four CAPTCHA datasets
collected from real dark web marketplaces, namely ChangAn
(D1), Freecity (D 2), Russian market (D 3), Mobilestore (D 4).
CAPTCHA images were collected via automated visits to
each marketplace’s login page, rate-paced with randomized
delays to stay below detection thresholds. Each visit yielded a
fresh CAPTCHA, which was downloaded and stored. For each
marketplace, we sampled 500 CAPTCHA instances, resulting
in 2,000 images in total. Table V summarizes the resulting
dataset structure and visual characteristics. Dataset availability
and responsible-use safeguards are discussed in Appendix B.
(a)D 1
Alphanumeric, colored
interference lines, wave distortion.
(b)D 2
Alphanumeric, clean background,
single strikethrough line.
(c)D 3
Alphanumeric, gradient background,
wave distortion and dot noise.
(d)D 4
Alphanumeric, patterned mesh
background, 60×30 px.
Fig. 5: Representative samples from each dataset.
D1is the most diverse dataset, containing CAPTCHAs span-
ning three types: alphanumeric, math equation, and Chinese
characters. The images exhibit a range of visual defenses, e.g.,
colored interference lines, wave-distorted characters, overlap-
ping strokes, and mixed background textures, making it the
primary benchmark for evaluating the Planner’s ability to
select appropriate tools for heterogeneous inputs.
8

TABLE V: Summary of dataset visual properties.
Dataset Type Sample size Background Noise Distortion Char Quality Difficulty Dark web site
D1 Alpha / Math / Chinese 500 Mixed colored lines Wave Mixed High ChangAn
D2 Alphanumeric 500 Clean Strikethrough line None Normal Low Freecity
D3 Alphanumeric 500 Gradient Dots/specks Wave Normal Medium Russian market
D4 Alphanumeric 500 Patterned None (mesh texture) None Thick Medium Mobilestore
D2contains alphanumeric CAPTCHAs with clean back-
grounds and a dark line crossing the text. This dataset specifi-
cally tests whether DARKNIGHTcorrectly applies minimal or
nopreprocessing to easy images. A well-functioning Planner
should recognize that preprocessing is unnecessary here and
leave the image untouched, directly evaluating the system’s
handling of the degeneration-recovery effect.
D3contains alphanumeric CAPTCHAs with moderate noise
and distortion, serving as an intermediate-difficulty benchmark
betweenD 2andD 1.
D4contains alphanumeric CAPTCHAs with thick characters
on a patterned mesh background.
Dataset Annotation and Verification.Ground-truth labels
were produced independently prior to the model experiments,
ensuring that annotation was not influenced by model predic-
tions. Three annotators participated in the annotation process
and inter-annotator reliability was measured before any rec-
onciliation using pairwise exact-string agreement (ESA) and
character error rate (CER). Disagreements were subsequently
reviewed and manually resolved through consensus to estab-
lish the final ground-truth labels. More details and dataset-level
agreement statistics are reported in Appendix C.
C. Baselines and Models
We evaluate nine open-weight models, with model size
spanning from 1B to 30B parameters (Table VI). Each model
is tested in two configurations:
Baseline (raw VLM).A CAPTCHA image is passed directly
to the model with one generic prompt for all types. There
is no preprocessing, no type-specific prompting, and no RAG
retrieval involved. This set of evaluations demonstrates what
each model achieves on its own.
DARKNIGHT(agentic pipeline).The DARKNIGHTsystem is
applied on top of the same model, with visual analysis, RAG
retrieval, selective preprocessing, type-specific prompting, and
feedback recording.
Throughout this paper,backbonerefers to the underlying
base VLM on top of which the DARKNIGHTagentic pipeline
is applied.
Eight of the nine evaluated models are general-purpose
VLMs. We include HunyuanOCR as our sole OCR-specialized
baseline to determine whether dedicated text-recognition fine-
tuning is sufficient on its own. The selected models span
1B to 30B parameters across four distinct architectures, to
understand if our findings generalize across different model
families and training paradigms. A key distinction among
these models is theirinference mode(Table VI). Instruct-onlyTABLE VI: Baseline models evaluation: Parameter size, type,
domain, and development source.
Model Name Size Mode Domain Date Source
HunyuanOCR 1BInstOCR Nov 2025 Tencent
Gemma4-E4B-it 8BInstGeneral Apr 2026 Google
Qwen3-vl-4B 4BInstGeneral Oct 2025 Alibaba
Qwen3-vl-8B 8BInstGeneral Oct 2025 Alibaba
Qwen3.5-9B 9BThinkGeneral Mar 2026 Alibaba
Step3-VL-10B 10BThinkGeneral Jan 2026 Stepfun
Gemma4-12B-it 12BInstGeneral Jul 2026 Google
Qwen3.6-27B 27BThinkGeneral Apr 2026 Alibaba
Qwen3-VL-30B-A3B 30BInstGeneral Oct 2025 Alibaba
models (markedInst) execute structured commands with-
out internal chain-of-thought deliberation, prioritizing speed
over planning reliability. Conversely, thinking-enabled models
(markedThink) perform extended internal reasoning, yield-
ing more deliberate tool-selection decisions at the expense of
higher latency. This dichotomy is critical: DARKNIGHTre-
quires the backbone to bothplan(reason about tool selection)
andread(execute OCR). These are two distinct tasks that
inherently benefit from different inference modes.
D. Evaluation Metrics
Exact-match accuracy.The primary metric: the fraction of
CAPTCHAs for which the website returns “pass” upon sub-
mission. Matching is case-insensitive for alphabetic characters,
following standard practice [20], which we also experimented
on the markets we gathered our dataset from. For math
CAPTCHAs, the numeric result must match exactly.
Inference time.Wall-clock time per CAPTCHA on local hard-
ware, covering all pipeline stages (visual analysis, RAG query,
preprocessing, OCR, and result handling). This measures local
compute only; the Tor network round-trip is excluded.
Degeneration and recovery rates. For each preprocessing
tool, we measure: (i)degeneration rate: fraction of images
correct without the tool but incorrect with it (correct→
wrong); (ii)recovery rate: fraction incorrect without the tool
but correct with it (wrong→correct). Statistical significance
is assessed via McNemar’s test (p <0.05).
VII. RESULTS ANDANALYSES
A. CAPTCHA-solving Accuracy
Fig. 6 presents the full empirical results across all four
datasets; the corresponding exact numerical values are com-
piled in Table XIII (Appendix G).
Baseline models performance.Without any agentic assis-
tance, open-weight VLMs exhibit highly variable performance
9

gemma4-e4b-itqwen3-vl-4b qwen3-vl-8b qwen3.5-9b step3-vl-10bgemma4-12b-itqwen3.6-27b
qwen3-vl-30b-a3b0%20%40%60%80%100%Accuracy (%)
628403764
2246424756677376
4683811
gemma4-e4b-itqwen3-vl-4b qwen3-vl-8b qwen3.5-9b step3-vl-10bgemma4-12b-itqwen3.6-27b
qwen3-vl-30b-a3b0%20%40%60%80%100%Accuracy (%)71
31437899
658594
6673
4994100
81100 1002
gemma4-e4b-itqwen3-vl-4b qwen3-vl-8b qwen3.5-9b step3-vl-10bgemma4-12b-itqwen3.6-27b
qwen3-vl-30b-a3b0%20%40%60%80%Accuracy (%)51
27335854
436467
56
353863
57
4868723
gemma4-e4b-itqwen3-vl-4b qwen3-vl-8b qwen3.5-9b step3-vl-10bgemma4-12b-itqwen3.6-27b
qwen3-vl-30b-a3b0%20%40%60%80%100%Accuracy (%)74
3429858189 91 93
70
55
3992
8295 93 934
Baseline DarKnightFig. 6: Baseline vs. DARKNIGHT: accuracy (%) across four
benchmark datasets.
across dark web CAPTCHA datasets. OnD 1, the most di-
verse dataset, containing alphanumeric, math, and Chinese
CAPTCHAs with heavy visual defenses, baseline accuracy is
strikingly low, ranging from 6% with Gemma4-e4b-it (see Ta-
ble XIII) to 64% (achieved by Step3-VL-10B). Most general-
purpose VLMs cluster around 22–46%, indicating that dark
web CAPTCHAs with multiple defense features remain chal-
lenging for current models. Conversely, on the less defended
alphanumeric CAPTCHAs ofD 4, several models exceed 80%
baseline accuracy. Models also perform reliably onD 2’s
clean, minimal-noise backgrounds (Step3-VL-10B: 98.6%).
D3represents an intermediate difficulty, with baselines falling
between 27–67%.
Model scale does not predict accuracy. OnD 1, parameter
count is a poor predictor of baseline accuracy. The dense
Qwen3.6-27B (46.0%) improves on Qwen3-VL-8B (39.6%)
by only 6.4 pp despite a 3.4×larger parameter count, and
despite also being a newer model with built-in reasoning.
The mixture-of-experts Qwen3-VL-30B-A3B (42.4%), which
activates only 3B parameters per token, performs on par with
its dense 8B sibling. Conversely, Step3-VL-10B reaches 63.6%
onD 1, outperforming every larger open-weight model we
evaluate, while Gemma4-12B-IT (22.4%) falls below the 4B
Qwen3-VL (28.0%). However, pure OCR systems lacking
visual reasoning struggle on complex datasets. HunyuanOCR
(cf. Table XIII) achieves 90.0% on the simpleD 4but collapses
to 12.4% onD 1.
Agentic Tool Selection vs. Model Scaling.Beyond base-
line performance, in our evaluation, DARKNIGHT’s agentic
framework yields higher accuracy gains than scaling model
parameters. Within the dense Qwen3-VL family, doubling
parameters from 4B to 8B raisesD 1accuracy by 11.6 pp
(28.0%→39.6%), whereas adding DARKNIGHTto the 4B
model raises it by 27.6 pp (to 55.6%), so the 4B model withDARKNIGHTexceeds the base 8B by 16 pp. This improvement
holds across model families: even Qwen3-VL-30B-A3B rises
from 42.4% to 80.6% with DARKNIGHT(+38.2 pp). Gemma4-
E4B-IT with DARKNIGHT(46.8%) also slightly exceeds the
bare Qwen3-VL-30B-A3B (42.4%).
We additionally evaluate two conventional OCR systems
and four frontier cloud VLMs as reference baselines; although
the latter remains outside our target deployment setting as out-
lined in Sec. I. In the first group, PaddleOCR [41] substantially
outperforms Tesseract [42], particularly on the less distorted
datasets, but remains considerably weaker on the heavily
defendedD 1dataset (below 5%). Recent frontier API models
from Claude should provide stronger general-purpose visual
reasoning, although their performance was below expectations;
even with the Claude Opus 5, the accuracy is less than 50%
(see Appendix D for more details).
B. The Degeneration-Recovery Effect
A central design premise of DARKNIGHTis that prepro-
cessing must be applied selectively rather than uniformly.
To validate this, we trace every tool-selection decision made
by the Planner across the evaluation suite and classify
its ultimate effect on the prediction. By comparing the
DARKNIGHTpipeline’s outcome against the raw VLM base-
line, we map each attempt to one of the four transition
states: correct→correct (C→C, preserving a correct predic-
tion), correct→wrong (C→W,degeneration), wrong→correct
(W→C,recovery), and wrong→wrong (W→W, no effect).
Based on these transition states, we derive three core metrics
to quantify the Planner’s efficacy:Degen%(the fraction of
originally correct baseline predictions degraded by unnec-
essary tools),Recov%(the fraction of originally incorrect
baseline predictions successfully recovered), andNet(the
overall performance gain, calculated as Recov%−Degen%).
TABLE VII: Per-tool degeneration-recovery profile across all
models and datasets. Tools ordered by net effect.
Tool N % C→C C→W W→C W→W Degen% Recov% Net
(no tool) 4406 72.8 2357 159 733 1157 6.3 38.8+32.5
color extract 1216 20.1 371 58 508 279 13.5 64.5+51.0
sharpen 197 3.3 66 3 90 38 4.3 70.3+66.0
remove straight line 184 3.0 111 9 53 11 7.5 82.8+75.3
denoise 43 0.7 17 12 6 8 41.4 42.9+1.5
binarize 3 0.0 0 0 0 3 — 0.0 —
Total 6049 100 2922 241 1390 1496 7.6 48.2 +40.5
Table VII summarises the effect of each preprocessing
tool applied by DARKNIGHTacross three operational models
(Qwen3-VL-4B, Qwen3.5-9B, and Qwen3-VL-30B-A3B; see
Sec. VII-F) and all datasets. The bottom row reports the pooled
degeneration and recovery rates across all Planner decisions,
including cases in which no preprocessing tool was selected.
Across all Planner decisions, DARKNIGHTrecovers 48.2%
of baseline failures at a cost of 7.6% degeneration, yielding
1,390 recoveries against only 241 degenerations (a5.8×
recovery-to-degeneration ratio).
The preprocessing tools exhibit different risk-reward pro-
files.remove_straight_lineachieves the strongest net
10

effect (+75.3), recovering 82.8% of baseline failures while
degrading 7.5% of correct predictions.sharpenprovides the
next strongest profile (+66.0), with a low degeneration rate
of 4.3% and a recovery rate of 70.3%.color_extract,
the most frequently selected preprocessing tool, achieves a
+51.0 net effect, recovering 64.5% of failures at a 13.5%
degeneration rate. In contrast,denoiseprovides only a
marginal net benefit (+1.5), whilebinarizeis selected only
three times and therefore does not provide sufficient evidence
for a meaningful risk-reward estimate.
TABLE VIII: Planner tool-selection behavior per dataset.No-
preproc rateis the fraction of decisions where the Planner
selects an empty pipeline.
D1 D2 D3 D4
No-preproc rate 26.6% 86.9% 97.0% 56.7%
Avg baseline acc 35.7% 67.7% 50.5% 70.6%
Avg∆with DARKNIGHT+34.2 +21.5 +6.5 +9.5
Table VIII demonstrates that the Planner’s intervention be-
havior varies substantially across datasets. OnD 3, the Planner
selects no preprocessing in 97.0% of cases, while the no-
preprocessing rate remains high at 86.9% onD 2. Conversely,
onD 1, which contains substantially heavier visual obfuscation,
the Planner intervenes in 73.4% of cases. This indicates that
tool selection is driven by the visual characteristics of the
CAPTCHA rather than by a uniform preprocessing policy.
Importantly, selecting no preprocessing is itself an explicit
Planner decision. When intervention is unnecessary, withhold-
ing tools avoids potential degeneration. Overall, in our eval-
uation, effective planning required both selecting appropriate
transformations and knowing when not to intervene.
C. Effect of Retrieval Memory
To isolate the contribution of retrieval memory, we compare
the raw VLM baseline, the DARKNIGHTpipeline without
retrieval, and the full DARKNIGHTframework with retrieval.
As shown in Fig. 7, enabling retrieval consistently improves
accuracy over the no-retrieval pipeline across all evaluated
configurations. This demonstrates that the gains of DARK-
NIGHTare not solely attributable to preprocessing and prompt-
ing. Retrieving prior successful and failed experiences pro-
vides additional benefit by guiding tool selection and avoiding
previously harmful actions. We further examine the sensitivity
to RAG memory size in App. E, where a memory size of 100
achieves the best average performance across the evaluated
settings and is adopted as the default configuration.
D. Cross-Site Adaptation
To evaluate whether DARKNIGHTcan operate across het-
erogeneous marketplaces using a shared experience memory,
we construct a mixed-site stream from all four datasets.
Specifically, we sample 125 CAPTCHAs from each ofD 1
toD 4, combine them into a merged set with 500 images, and
qwen3-vl-4b-instruct qwen3-vl-8b-instruct0%20%40%60%80%Accuracy (%)1
qwen3-vl-4b-instruct qwen3-vl-8b-instruct0%20%40%60%80%Accuracy (%)2
qwen3-vl-4b-instruct qwen3-vl-8b-instruct0%20%40%Accuracy (%)3
qwen3-vl-4b-instruct qwen3-vl-8b-instruct0%20%40%60%Accuracy (%)4
Baseline Pipeline w/o Retrieval Pipeline w/ Retrieval (DarKnight)Fig. 7: Ablation of retrieval memory using Qwen3-VL-4B and
Qwen3-VL-8B acrossD 1− D 4.
randomly shuffle the samples. The resulting stream is pro-
cessed sequentially using a single shared RAG store, allowing
experience accumulated from one marketplace to be retrieved
when solving visually similar CAPTCHAs originating from
another. Table IX reports the results. DARKNIGHTimproves
accuracy for all four evaluated backbones in this mixed-
site setting. Qwen3-VL-8B improves from 65.0% to 75.0%,
while Qwen3.6-27B improves from 65.0% to 78.0%. We
also observe substantial improvements for Gemma4-E4B and
Qwen3.5-9B. These results indicate that DARKNIGHTcan
maintain a shared experience memory across heterogeneous
CAPTCHA sources rather than requiring an independently
engineered solver for each marketplace.
TABLE IX: Accuracy (%) in the cross-site mixed-stream
experiment.
Model Baseline DARKNIGHT∆
Qwen3-VL-8B 65.0 75.0 +10.0
Gemma4-E4B 50.8 60.8 +10.0
Qwen3.5-9B 16.5 80.4 +63.9
Qwen3.6-27B 65.0 78.0 +13.0
E. Inference Time and Computational Overhead
Table X presents the end-to-end inference time across all
datasets. As expected, integrating the DARKNIGHTagentic
pipeline inherently increases latency compared to raw base-
lines. This delay represents the computational cost of the
perceive-plan-execute loop, which involves dual VLM invoca-
tions and intermediate RAG retrieval. However, this overhead
varies substantially across the evaluated VLMs.
The Accuracy-Latency Sweet Spot.Among the evaluated
models, Qwen3-VL-30B-A3B-Instruct provides the strongest
accuracy–latency trade-off. With DARKNIGHT, it achieves an
average accuracy of 86.5% across the four datasets while re-
quiring only 2.64 s per CAPTCHA on average. In comparison,
Qwen3.6-27B reaches a similar average accuracy of 86.1%
but requires 20.77 s, while Step3-VL-10B achieves 78.8% at
11

TABLE X: Mean end-to-end inference time (in seconds) per
CAPTCHA across dark web datasets.
D1 D4 D2 D3
Model Params B DK B DK B DK B DK
HunyuanOCR 1B 1.82 – 0.10 – 0.07 – 0.13 –
Gemma4-E4B-IT 8B 0.156 3.81 0.154 3.25 0.137 3.12 0.157 2.97
Qwen3-VL-4B-Instruct 4B 0.122 2.72 0.102 2.21 0.120 2.57 0.108 2.34
Qwen3-VL-8B-Instruct 8B 0.170 4.26 0.164 3.72 0.273 3.89 0.165 3.67
Qwen3.5-9B 9B 0.168 5.38 0.162 4.70 0.163 5.17 0.166 4.43
Step3-VL-10B 10B 4.500 68.19 1.026 44.50 2.203 48.20 0.971 50.11
Gemma4-12B-IT 12B 0.156 4.44 0.164 5.47 0.144 4.03 0.172 3.77
Qwen3.6-27B 27B 2.247 23.03 1.714 18.34 1.898 21.00 1.732 20.70
Qwen3-VL-30B-A3B-Instruct 30B 0.170 2.92 0.155 2.11 0.148 3.16 0.162 2.38
52.75 s. These results indicate that efficient model architecture
and agentic orchestration matter more than model size alone.
The Latency Bottleneck of Heavy Reasoning.Conversely,
some models incur higher local inference cost when integrated
with DARKNIGHT. Step3-VL-10B is the most expensive, re-
quiring 44.50–68.19 s per CAPTCHA across the four datasets,
with an average of 52.75 s. Qwen3.6-27B also incurs consider-
ably higher latency, averaging 20.77 s, despite achieving strong
accuracy. In contrast, Qwen3-VL-30B-A3B-Instruct achieves
comparable or higher accuracy with an average latency of
only 2.64 s. These results show that the agentic overhead
depends on model architecture and reasoning behavior rather
than parameter count.
For the backbones we evaluated, effective CAPTCHA solv-
ing did not depend on parameter scaling alone. Instead,
the best accuracy-latency trade-off came from combining an
efficient VLM backbone with selective agentic orchestration.
F . Cross-Model Accuracy-Latency Analysis
Table XI shows that the benefit introduced by DARKNIGHT
varies substantially across backbones and is not determined
simply by model size or inference mode. Qwen3-VL-4B ob-
tains the largest incremental improvement (+24.9%), showing
that a relatively weak raw backbone can benefit substantially
from agentic assistance. In contrast, Qwen3-VL-30B-A3B-
Instruct achieves the highest final average accuracy (86.5%),
closely followed by Qwen3.6-27B (86.1%). These results also
highlight the distinction between pipeline-induced gain and ab-
solute performance, a large improvement does not necessarily
imply the highest final accuracy.
TABLE XI: Cross-model accuracy and latency summary. Avg
B and Avg DK denote the average raw-baseline and DARK-
NIGHTaccuracy across the four datasets; Avg∆denotes the
incremental improvement introduced by DARKNIGHT, and
tDKdenotes the average local inference time per CAPTCHA.
Model Params Avg B Avg DK Avg∆t DK(s)
Qwen3-VL-4B-Instruct 4B 29.9 54.8+24.9 2.46
Qwen3-VL-8B-Instruct 8B 36.3 48.3+12.0 3.89
Gemma4-E4B-IT 8B 50.8 59.9+9.1 3.29
Qwen3.5-9B 9B 64.6 80.9+16.3 4.92
Step3-VL-10B 10B 74.2 78.8+4.6 52.75
Gemma4-12B-IT 12B 54.9 67.4+12.6 4.43
Qwen3.6-27B 27B 71.4 86.1+14.7 20.77
Qwen3-VL-30B-A3B-Instruct 30B 73.9 86.5+12.6 2.64
The latency results further show that parameter count alone
is a poor predictor of computational cost. Qwen3-VL-30B-A3B-Instruct achieves the highest average DARKNIGHTac-
curacy while requiring only 2.64 s per CAPTCHA, providing
the strongest accuracy–latency trade-off among the evaluated
backbones. By comparison, Qwen3.6-27B reaches a similar
accuracy of 86.1% but requires 20.77 s, while Step3-VL-10B
requires 52.75 s despite achieving a lower final accuracy of
78.8%. Overall, the results suggest that selecting an efficient
backbone is more important than either parameter scaling or
model category alone when deploying DARKNIGHT.
VIII. CONCLUSION
In this work, we introduced DARKNIGHT, an autonomous,
self-improving agentic framework that redefines the approach
to defeating dark web CAPTCHAs. By shifting from per-site
model retraining and large parameter scaling toward dynamic,
experience-driven tool orchestration, DARKNIGHTenables
lightweight, open-weight VLMs to reach high accuracy among
deployable open-weight solvers. Equipping a small 9B VLM
with DARKNIGHTyields a +36.8% accuracy gain on the most
heavily defended dataset, with the local agentic loop com-
pleting in 5.38s per CAPTCHA. Furthermore, the dual-signal
RAG architecture navigates the degeneration-recovery trade-
off by learning to avoid preprocessing steps that previously
degraded similar CAPTCHAs. In our evaluation, the agent’s
efficacy rests as much on restraint, knowing when to bypass
preprocessing, as on active intervention.
Implications.Beyond our specific tool, we see three im-
plications; the last two are speculative. First, for CTI teams:
in our case study, one workstation GPU, an open-weight
model, and five classical image tools solved most CAPTCHAs
without labelled data or per-market training. Practitioners can
start from our results on backbone choice (the mixture-of-
experts Qwen3-VL-30B-A3B gave the best accuracy–latency
trade-off, Sec. VII-E), on failure memory (Sec. VII-B and
VII-C), and on per-market warm-up (Sec. V-E). Second,
server-rendered text CAPTCHAs, on the dark or clear web,
may offer limited protection against low-cost local models; we
expect operators to move to other challenge types or stricter
rate limits, and collectors should plan for this. Third, learning
from failures when not to act may benefit other agentic tool-
use settings where a tool can degrade its input.
ACKNOWLEDGMENT
This research/project is supported by the National Research
Foundation, Singapore, and the Cyber Security Agency of
Singapore under the National Cybersecurity R&D Programme
and the CyberSG R&D Programme Office (Award CRPO-
GC2-ASTAR-001). Any opinions, findings, conclusions, or
recommendations expressed in these materials are those of
the author(s) and do not reflect the views of the National
Research Foundation, Singapore, the Cyber Security Agency
of Singapore, or the CyberSG R&D Programme Office.
12

REFERENCES
[1] Europol,The evolving threat landscape: how encryption, proxies
and AI are expanding cybercrime – Internet Organised Crime
Threat Assessment (IOCTA). Luxembourg: Publications Office of
the European Union, 2026. [Online]. Available: https://www.europol.
europa.eu/cms/sites/default/files/documents/IOCTA-2026.pdf
[2] D. Lerner, “State of the underground 2025: Key trends shaping cyber
risk today,” May 2025. [Online]. Available: https://www.bitsight.com/
blog/state-of-the-underground-2025
[3] S. Samtani, Y . Chai, and H. Chen, “Linking exploits from the
dark web to known vulnerabilities for proactive cyber threat
intelligence: An attention-based deep structured semantic model,”MIS
Quarterly, vol. 46, no. 2, pp. 911–946, 06 2022. [Online]. Available:
https://doi.org/10.25300/MISQ/2022/15392
[4] H. Bijmans, T. Booij, A. Schwedersky, A. Nedgabat, and R. van
Wegberg, “Catching Phishers By Their Bait: Investigating the Dutch
Phishing Landscape through Phishing Kit Detection,” inUSENIX
Security, Aug. 2021, pp. 3757–3774. [Online]. Available: https:
//www.usenix.org/conference/usenixsecurity21/presentation/bijmans
[5] D. Kim, H. Zhao, D. Kim, and S. Hwang, “Unveiling the Underground
Phishing Ecosystem: A 12-Year Longitudinal Study of Deep and Dark
Web Forums,” inACM Web Conference, 2026, p. 3531–3542. [Online].
Available: https://doi.org/10.1145/3774904.3792737
[6] Z. Lin, J. Cui, X. Liao, and X. Wang, “Malla: Demystifying Real-world
Large Language Model Integrated Malicious Services,” inUSENIX
Security, Aug. 2024, pp. 4693–4710. [Online]. Available: https:
//www.usenix.org/conference/usenixsecurity24/presentation/lin-zilong
[7] T. E. Dearden and S. E. Tucker, “Follow the Money: Analyzing Darknet
Activity Using Cryptocurrency and the Bitcoin Blockchain,”Journal of
Contemporary Criminal Justice, vol. 39, no. 2, Mar. 2023.
[8] S. L. Schr ¨oer, N. Canevascini, I. Pekaric, P. Widmer, and P. Laskov,
“The dark side of the web: Towards understanding various data sources
in cyber threat intelligence,” inIEEE European Symposium on Security
and Privacy Workshops (EuroS&PW), 2025, pp. 79–89.
[9] Y . Wang, B. Arief, and J. Hernandez-Castro, “Secure in the dark? An
in-depth analysis of dark web markets security,”International Journal
of Information Security, vol. 24, no. 3, p. 107, 2025.
[10] N. Zhang, M. Ebrahimi, W. Li, and H. Chen, “Counteracting
Dark Web Text-Based CAPTCHA with Generative Adversarial
Learning for Proactive Cyber Threat Intelligence,”ACM Trans.
Manage. Inf. Syst., vol. 13, no. 2, Mar. 2022. [Online]. Available:
https://doi.org/10.1145/3505226
[11] M. Vibhute, N. Gutierrez, K. Radivojevic, and P. R. Brenner, “Multi-
modal Web Agents for Automated (Dark) Web Navigation,” inICAART,
2025, pp. 437–444.
[12] D. De Pascale, G. Cascavilla, D. A. Tamburri, and W. J. Van
Den Heuvel, “CRATOR a CRAwler for Tor: Turning Dark Web Pages
into Open Source INTelligence,” in9th European Symposium on
Research in Computer Security. Springer-Verlag, 2024, p. 144–161.
[Online]. Available: https://doi.org/10.1007/978-3-031-70890-9 8
[13] Y . Yannikos and J. Heeger, “Captchas on Darknet Marketplaces:
Overview and Automated Solvers,”Electronic Imaging, vol. 36, no. 4,
pp. 330–1–330–1, 2024.
[14] D. H. Audran, M. B. Andersen, M. H. Hansen, M. M. Andersen, T. B.
Frederiksen, K. H. Hansen, D. Georgoulias, and E. Vasilomanolakis,
“Tick Tock Break the Clock: Breaking CAPTCHAs on the Darkweb,”
inSECRYPT, 2022, pp. 357–365.
[15] L. von Ahn, B. Maurer, C. McMillen, D. Abraham, and M. Blum,
“reCAPTCHA: Human-Based Character Recognition via Web Security
Measures,”Science, vol. 321, no. 5895, pp. 1465–1468, 2008.
[16] M. Guerar, L. Verderame, M. Migliardi, F. Palmieri, and A. Merlo,
“Gotta CAPTCHA ’Em All: A Survey of 20 Years of the Human-or-
computer Dilemma,”ACM Comput. Surv., vol. 54, no. 9, Oct. 2021.
[Online]. Available: https://doi.org/10.1145/3477142
[17] Z. Wu, Y . Xue, Y . Feng, X. Wang, and Y . Song, “MCA-Bench: A
multimodal benchmark for evaluating CAPTCHA robustness against
VLM-based attacks,” inAAAI Conference on Artificial Intelligence,
vol. 40, no. 44, 2026, pp. 38 039–38 047. [Online]. Available:
https://doi.org/10.1609/aaai.v40i44.41141
[18] J. Zhang, Z. Zhou, X. Ji, S. Liu, and Z. Zhao, “CAPTURE: A
Benchmark and Evaluation for LVLMs in CAPTCHA Resolving,”arXiv
preprint arXiv:2512.11323, 2025.[19] A. Searles, Y . Nakatsuka, E. Ozturk, A. Paverd, G. Tsudik, and
A. Enkoji, “An empirical study & evaluation of modern CAPTCHAs,”
inUSENIX Security, 2023.
[20] X. Teoh, Y . Lin, S. Li, R. Liu, A. Sollomoni, Y . Harel, and J. S. Dong,
“Are CAPTCHAs still bot-hard? Generalized visual CAPTCHA solving
with agentic vision language model,” inUSENIX Security, 2025, pp.
3747–3766.
[21] G. Deng, H. Ou, Y . Liu, J. Zhang, T. Zhang, and Y . Liu, “Oedipus:
LLM-enchanced Reasoning CAPTCHA Solver,” inACM CCS, 2025,
p. 6–20. [Online]. Available: https://doi.org/10.1145/3719027.3744872
[22] Anthropic, “Statement on the US government directive to suspend
access to Fable 5 and Mythos 5,” https://www.anthropic.com/news/
fable-mythos-access, Jun 2026, accessed: 2026-10-14.
[23] Z. Zhang, W. Zhou, and M. Sherr, “Bypassing Tor Exit Blocking with
Exit Bridge Onion Services,” inACM CCS, 2020, p. 3–16. [Online].
Available: https://doi.org/10.1145/3372297.3417245
[24] CrowdStrike, “CrowdStrike 2025 APJ eCrime Landscape Report:
Chinese Underground Marketplaces Drive Billions in Illicit
Transactions; AI-accelerated Ransomware Surges,” 2025. [Online].
Available: https://ir.crowdstrike.com/news-releases/news-release-details/
crowdstrike-2025-apj-ecrime-landscape-report-chinese-underground
[25] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y . Cao,
“ReAct: Synergizing reasoning and acting in language models,” inICLR,
2023.
[26] StepFun AI, “Step3-VL-10B: Compact Yet Frontier Multimodal
Intelligence,”arXiv preprint arXiv:2601.09668, 2026. [Online].
Available: https://arxiv.org/abs/2601.09668
[27] D. Georgoulias, J. M. Pedersen, M. Falch, and E. Vasilomanolakis, “A
qualitative mapping of darkweb marketplaces,” inAPWG eCrime, 2021,
pp. 1–15.
[28] E. Bursztein, S. Bethard, C. Fabry, J. C. Mitchell, and D. Jurafsky, “How
good are humans at solving CAPTCHAs? A large scale evaluation,” in
IEEE Security and Privacy, 2010, pp. 399–413.
[29] H. Gao, J. Yan, F. Cao, Z. Zhang, L. Lei, M. Tang, P. Zhang, X. Zhou,
X. Wang, and J. Li, “A simple generic attack on text captchas.” inNDSS,
2016.
[30] Y . Gao, H. Gao, S. luo, Y . Zi, S. Zhang, W. Mao, P. Wang, Y . Shen,
and J. Yan, “Research on the security of visual reasoning CAPTCHA,”
inUSENIX Security, Aug. 2021, pp. 3291–3308. [Online]. Available:
https://www.usenix.org/conference/usenixsecurity21/presentation/gao
[31] J. Wang, C. Zhu, Y . Zhou, L. Li, X. He, M. Wei, and J. Xiong,
“COGNITION: From Evaluation to Defense against Multimodal LLM
CAPTCHA Solvers,”arXiv preprint arXiv:2512.02318, 2025.
[32] C. Zauner, “Implementation and Benchmarking of Perceptual Image
Hash Functions,” Master’s thesis, Upper Austria University of Applied
Sciences, Hagenberg Campus, Hagenberg, Austria, 2010. [Online].
Available: https://www.phash.org/docs/pubs/thesis zauner.pdf
[33] W. F. King, “Analysis of demand paging algorithms,”Information
Processing, vol. 71, pp. 485–490, 1971.
[34] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, “MiniLM:
Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-
Trained Transformers,” inAdvances in Neural Information Processing
Systems, vol. 33, 2020, pp. 5776–5788.
[35] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence Embeddings
using Siamese BERT-Networks,” inEMNLP-IJCNLP. Hong Kong,
China: Association for Computational Linguistics, 2019, pp. 3982–3992.
[36] Hunyuan Vision Teamet al., “HunyuanOCR Technical Report,” 2025.
[Online]. Available: https://arxiv.org/abs/2511.19575
[37] Qwen Team, “Qwen3-VL Technical Report,” 2025. [Online]. Available:
https://arxiv.org/abs/2511.21631
[38] Google DeepMind, “Gemma 4,” Hugging Face, 2026, accessed: 2026-
06-12. [Online]. Available: https://huggingface.co/google/gemma-4-E4B
[39] Qwen Team, “Qwen3.5: Towards Native Multimodal Agents,” Qwen
Blog, Feb 2026, accessed: 2026-08-12. [Online]. Available: https:
//qwen.ai/blog?id=qwen3.5
[40] ——, “Qwen3.6-27B: Flagship-Level Coding in a 27B Dense Model,”
Qwen Blog, Apr 2026, accessed: 2026-06-12. [Online]. Available:
https://qwen.ai/blog?id=qwen3.6-27b
[41] Y . Du, C. Li, R. Guo, X. Yin, W. Liu, J. Zhou, Y . Bai, Z. Yu, Y . Yang,
Q. Dang, and H. Wang, “PP-OCR: A practical ultra lightweight
OCR system,” 2020, paddleOCR, PP-OCRv4 recognition model:
https://github.com/PaddlePaddle/PaddleOCR. [Online]. Available: https:
//arxiv.org/abs/2009.09941
13

[42] R. Smith, “An overview of the Tesseract OCR engine,” inNinth Inter-
national Conference on Document Analysis and Recognition (ICDAR
2007), vol. 2. IEEE, 2007, pp. 629–633.
APPENDIX
A. Ethical Considerations
This work studies CAPTCHA robustness in the context of
dark web threat intelligence collection. While CAPTCHA-
solving techniques can be misused to automate access to
protected services (e.g., on the clear web), our objective is dark
web analytics for the larger social good. We therefore release
the research artifacts used in this study, including the anno-
tated datasets, evaluation code, preprocessing tools, agentic
pipeline, experiment configurations, and reproduction scripts.
To reduce unnecessary exposure of operational infrastructure,
the released datasets contain no onion addresses, credentials,
or other marketplace access information.
B. Data Availability and Responsible Release
To support reproducibility and future research, we pub-
licly release the complete source code, experiment configura-
tions, reproduction scripts, and all four annotated CAPTCHA
datasets at: https://github.com/Binghui99/DarKnight. The re-
leased data contain no onion addresses or other marketplace
access information. We also withhold operational solver assets
and deployment tooling that could directly facilitate automated
CAPTCHA bypass at scale.
C. Inter-Annotator Agreement
Table XII reports inter-annotator reliability for CAPTCHA
transcription. Three annotators participated in the annotation
process, two of whom are Chinese-speaking, resulting in
two annotators labelled 2,000 CAPTCHAs, while the third
annotator skipped the 100+ Chinese CAPTCHAs fromD 1.
All agreement statistics were computed on independently
produced transcriptions before any disagreement resolution.
We report exact-string agreement (ESA), which measures the
proportion of independently produced transcription pairs that
match exactly, and character error rate (CER), which quantifies
character-level disagreement using edit distance. For samples
annotated by three annotators, the reported statistics aggre-
gate agreement across all annotator pairs; for Chinese-script
samples, agreement is evaluated between the two Chinese-
speaking annotators. After agreement measurement, disagree-
ments were reviewed and resolved by consensus to establish
the final ground-truth labels.
TABLE XII: Inter-annotator agreement for CAPTCHA tran-
scription. Agreement statistics were calculated before consen-
sus resolution.
Dataset Samples ESA (%) CER (%) Resolution
D1 500 90.4 2.90 Consensus
D2 500 99.4 0.16 Consensus
D3 500 96.4 0.70 Consensus
D4 500 94.4 1.29 ConsensusThe results show high annotation agreement across all four
datasets. D2 exhibits the highest reliability (99.4% ESA and
0.16% CER), consistent with its relatively clean visual struc-
ture, whereasD 1shows the largest disagreement (90.4% ESA
and 2.90% CER), reflecting its greater visual and script com-
plexity. Importantly, even forD 1, character-level disagreement
remains low, indicating that most transcription differences
involve only a small number of characters.
D. Additional Baseline Comparison
To provide additional reference points against established
OCR systems and frontier VLMs, we evaluate two locally
hosted OCR baselines (PaddleOCR [41] and Tesseract [42])
and four frontier Claude models on the ChangAn (D 1) dataset.
The Claude models are evaluated only as reference baselines
and are not considered deployment candidates under the op-
erational constraints discussed in Sec. I.
1
 2
 3
 4
0%20%40%60%80%100%Accuracy (%)
0% 0%PaddleOCR
Tesseract
step3-vl-10b (VL baseline)
qwen3.6-27b (VL baseline)
(a) Classical OCR and VLM baselines acrossD 1− D 4.
Claude
Haiku 4-5Claude
Sonnet 4-6Claude
Opus 4-6Claude
Opus 50%10%20%30%40%50%60%70%80%90%Accuracy (%)+43.2%+51.6%+42.4%+37.0%Baseline
DarKnight
(b) Frontier Claude models onD 1, with and without DARKNIGHT.
Fig. 8: Comparison with other baselines.
E. Sensitivity to RAG Memory Size
We further evaluate the sensitivity of DARKNIGHTto the
per-slot RAG memory capacity using Qwen3-VL-4B onD 1
andD 2. Here, a RAG size ofNdenotes a capacity ofN
successful andNfailed records, for a maximum of2Nrecords
in total. As shown in Table XIV, moderate memory sizes
generally provide strong performance, while increasing the
capacity does not consistently yield further improvements.
Among the evaluated settings, a per-slot capacity of 100 (up
14

TABLE XIII: Accuracy (%) on dark web CAPTCHA datasets.B= baseline (raw VLM),DK= DARKNIGHT(agentic pipeline).
∆= absolute change. Best agentic accuracy per dataset inbold. Models are ordered by size. HunyuanOCR is a pure OCR
system that cannot be used as a Planner and therefore has no agentic results.
D1 D2 D3 D4
Model Params BDK∆ BDK∆ BDK∆ BDK∆
HunyuanOCR 1B 12.4 – – 61.4 – – 45.7 – – 90.0 – –
Gemma4-E4B-IT 8B 5.8 46.8 +41.0 71.4 65.8−5.6 51.5 56.5 +5.0 74.4 70.4−4.0
Qwen3-VL-4B-Instruct 4B 28.0 55.6 +27.6 30.8 73.4 +42.6 26.8 35.3 +8.5 33.8 54.8 +21.0
Qwen3-VL-8B-Instruct 8B 39.6 66.6 +27.0 43.4 49.0 +5.6 33.4 38.3 +4.9 28.8 39.4 +10.6
Qwen3.5-9B 9B 36.6 73.4 +36.8 78.2 94.2 +16.0 58.1 63.4 +5.3 85.4 92.4 +7.0
Step3-VL-10B 10B 63.6 76.0 +12.4 98.6100.0+1.4 53.6 57.0 +3.4 80.8 82.0 +1.2
Gemma4-12B-IT 12B 22.4 46.0 +23.6 65.0 81.0 +16.0 43.0 47.7 +4.7 89.095.0+6.0
Qwen3.6-27B 27B 46.0 83.2 +37.2 85.2100.0+14.8 63.6 68.0 +4.4 90.6 93.0 +2.4
Qwen3-VL-30B-A3B 30B 42.480.6+38.2 94.0100.0+6.0 66.772.3+5.6 92.6 93.0 +0.4
TABLE XIV: Accuracy (%) for varying RAG memory size.
RAG Size D1D2
No RAG 52.8 51.0
10 57.2 53.0
20 56.8 73.2
50 56.0 73.2
80 57.2 64.8
100 57.8 73.4
150 56.8 73.2
200 59.4 59.0
to 200 records in total) achieves the highest average accuracy
across the two datasets and is therefore adopted as the default
configuration.
The results in Fig. 8 show that conventional OCR systems
perform poorly on ChangAn (D 1), with PaddleOCR achieving
3.4% and Tesseract 0%. The frontier Claude models provide
substantially stronger raw performance, reaching up to 44.0%.
DARKNIGHTfurther improves all four Claude models, with
final accuracy reaching up to 81.0%.
F . API parameters for VLM calls
Tab. XV specifies the exact hyperparameter configurations
and inference payloads used for the two distinct VLM invo-
cations (planning and execution) within the framework.
TABLE XV: API parameters for the two VLM calls per
sample.
Parameter visual analysis&plan run ocr
max_tokens2048 1024
temperature0.0 0.1
Input image+text+RAG processed image+text
G. Detailed results
This section provides the complete granular dataset for
the evaluation discussed in Sec. VII. Table XIII details the
exact accuracy percentages and absolute changes (∆) for all
evaluated VLMs across the four dark web datasets, serving as
the numerical foundation for Fig. 6.H. Discussion
Local CAPTCHA Solving for CTI.Historically, defeating
specially designed dark web CAPTCHAs has required either
labor-intensive per-site engineering or increasingly large vision
models. DARKNIGHTdemonstrates that strong performance
can instead be achieved by combining open-weight VLMs with
adaptive, experience-driven tool orchestration. In particular,
our results show that model size alone does not determine
either accuracy or inference cost; an efficient backbone paired
with DARKNIGHTcan provide a favorable accuracy–latency
trade-off without relying on third-party VLM APIs. This
makes local CAPTCHA solving more accessible for automated
CTI collection.
The Necessity of Negative Memory in Visual Agents.A core
insight from our evaluation is that agentic visual systems fail
not just from an inability to act, but from an inability to restrain
themselves. When applied uniformly, aggressive preprocessing
consistently degrades clean inputs. DARKNIGHTovercomes
this via its dual-signal RAG architecture. By treating historical
failures as explicit negative constraints rather than discarding
them, the Planner learns a conservative execution policy. This
suggests that multimodal agents deployed in volatile secu-
rity contexts may benefit from incorporating explicit failure-
memory mechanisms to prevent self-induced degradation.
Limitations and Adversarial Adaptation.While our evalu-
ation uses real-world dark web CAPTCHA datasets, a live
deployment faces dynamic adversarial conditions that are
not fully captured by an offline study. First, marketplace
operators may introduce targeted perturbations or entirely
new CAPTCHA mechanisms designed to disrupt VLM-based
solvers. Although DARKNIGHTcan adapt its preprocessing
strategy through accumulated experience, its formal robust-
ness against targeted adversarial examples remains an open
question. Second, our latency measurements capture local
computational overhead only. In a live deployment, end-to-
end latency would additionally depend on variable Tor network
delays and crawler request pacing, which we did not measure.
Future work will integrate DARKNIGHTinto a live adaptive
crawler to evaluate its long-term behavior under operator
countermeasures and network variability.
15
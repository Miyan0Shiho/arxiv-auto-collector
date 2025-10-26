# AutoMT: A Multi-Agent LLM Framework for Automated Metamorphic Testing of Autonomous Driving Systems

**Authors**: Linfeng Liang, Chenkai Tan, Yao Deng, Yingfeng Cai, T. Y Chen, Xi Zheng

**Published**: 2025-10-22 10:11:05

**PDF URL**: [http://arxiv.org/pdf/2510.19438v1](http://arxiv.org/pdf/2510.19438v1)

## Abstract
Autonomous Driving Systems (ADS) are safety-critical, where failures can be
severe. While Metamorphic Testing (MT) is effective for fault detection in ADS,
existing methods rely heavily on manual effort and lack automation. We present
AutoMT, a multi-agent MT framework powered by Large Language Models (LLMs) that
automates the extraction of Metamorphic Relations (MRs) from local traffic
rules and the generation of valid follow-up test cases. AutoMT leverages LLMs
to extract MRs from traffic rules in Gherkin syntax using a predefined
ontology. A vision-language agent analyzes scenarios, and a search agent
retrieves suitable MRs from a RAG-based database to generate follow-up cases
via computer vision. Experiments show that AutoMT achieves up to 5 x higher
test diversity in follow-up case generation compared to the best baseline
(manual expert-defined MRs) in terms of validation rate, and detects up to
20.55% more behavioral violations. While manual MT relies on a fixed set of
predefined rules, AutoMT automatically extracts diverse metamorphic relations
that augment real-world datasets and help uncover corner cases often missed
during in-field testing and data collection. Its modular architecture
separating MR extraction, filtering, and test generation supports integration
into industrial pipelines and potentially enables simulation-based testing to
systematically cover underrepresented or safety-critical scenarios.

## Full Text


<!-- PDF content starts -->

1
AUTOMT: A Multi-Agent LLM Framework for
Automated Metamorphic Testing of Autonomous
Driving Systems
Linfeng Liang1, Chenkai Tan2, Yao Deng1, Yingfeng Cai2, T.Y Chen3, Xi Zheng1,
1School of Computing, Macquarie University, Australia2Automotive Engineering Research Institute, Jiangsu
University, China3School of Science, Computing and Emerging Technologies, Swinburne University of
Technology, Australia
Abstract—Autonomous Driving Systems (ADS) are safety-
critical, where failures can be severe. While Metamorphic Testing
(MT) is effective for fault detection in ADS, existing methods
rely heavily on manual effort and lack automation. We present
AUTOMT, a multi-agent MT framework powered by Large
Language Models (LLMs) that automates the extraction of
Metamorphic Relations (MRs) from local traffic rules and the
generation of valid follow-up test cases. AUTOMT leverages
LLMs to extract MRs from traffic rules in Gherkin syntax
using a predefined ontology. A vision-language agent analyzes
scenarios, and a search agent retrieves suitable MRs from a
RAG-based database to generate follow-up cases via computer
vision. Experiments show that AUTOMT achieves up to 5× higher
test diversity in follow-up case generation compared to the best
baseline (manual expert-defined MRs) in terms of validation
rate, and detects up to 20.55% more behavioral violations.
While manual MT relies on a fixed set of predefined rules,
AUTOMT automatically extracts diverse metamorphic relations
that augment real-world datasets and help uncover corner cases
often missed during in-field testing and data collection. Its
modular architecture—separating MR extraction, filtering, and
test generation—supports integration into industrial pipelines
and potentially enables simulation-based testing to systematically
cover underrepresented or safety-critical scenarios.
Index Terms—Metamorphic testing, autonomous driving, test-
ing, large language model, text-to-video.
I. INTRODUCTION
Fig. 1. A high-level overview of AUTOMT.
Autonomous Driving Systems (ADSs) are increasingly de-
ployed in real-world settings [1] and are safety-critical, as
failures can be catastrophic [2], [3]. Thorough testing is thus
essential. Metamorphic Testing (MT) has proven effective for
addressing the oracle problem in ADS testing, enabling early
*Corresponding author. Email: james.zheng@mq.edu.aubug detection and lowering development costs [4]–[7]. Prior
work has applied MT to ADSs using real-world data and
simulators [6], [7], but significant manual effort hinders full
automation. Two key gaps remain: (1) automatically extracting
feasible MRs, which prior work relies on human experts for
[4], [6], and (2) identifying suitable original test cases—e.g.,
a highway test is invalid for the MR“The vehicle should slow
down in a school zone”. This raises the central question:How
can we automate metamorphic testing for ADSs?
Recent advances in AI, particularly Large Language Models
(LLMs), have shown strong performance in language and
vision tasks [8]–[10]. We propose AUTOMT, a fully auto-
mated MT framework for ADSs powered by multiple LLM-
based agents. As shown in Figure 1, AUTOMT first uses a
predefined ontology based on local traffic rules to extract MRs
in Gherkin syntax [11] via a MR extraction agent (M-Agent).
The extracted MRs are embedded using retrieval-augmented
generation (RAG) to populate an MR-RAG database. A vision-
language agent (T-Agent) then analyzes test case context (e.g.,
speed, steering, road type). Finally, a follow-up generation
agent (F-Agent) retrieves valid MRs and generates follow-
up test cases using computer vision tools. We summarize our
key contributions below.
•Automated MR Extraction from Traffic Rules via
LLM:We propose an MR extraction framework that
leverages LLMs to automatically extract feasible meta-
morphic relations from traffic rules, reducing reliance on
manual effort and enabling scalable MR discovery.
•RAG-Enhanced Original Test Case Identification:We
introduce a pipeline that combines RAG with LLMs
to identify suitable original test cases from real-world
datasets or simulators, ensuring the applicability of ex-
tracted MRs and enabling valid follow-up test case gen-
eration.
•End-to-End Automated Metamorphic Testing:By inte-
grating automated MR extraction and test case selection,
we enable end-to-end automated metamorphic testing,
capable of generating follow-up cases for real-world data.
•Extensive Experiments:We conduct comprehensive ex-
periments, including qualitative analysis of MR extraction
accuracy and follow-up case realism, user perception ofarXiv:2510.19438v1  [cs.SE]  22 Oct 2025

2
MR violations in relation to ADS safety, and quantitative
comparisons with baseline methods for follow-up test
case generation.
This paper is organized as follows: Section II reviews related
work. Section III introduces our method, and Section IV
outlines the experimental design. Results are presented in Sec-
tion V, followed by discussion and future work in Section VI.
Section VII concludes the paper.
II. RELATEDWORK
A. ADS Metamorphic Testing
The oracle problem poses a major challenge in ADS testing
[12], as real-world traffic complexity makes it infeasible to
enumerate all expected outcomes [7]. Metamorphic Testing
(MT) [13] has been widely adopted to address this issue. Prior
work uses a small set of human-defined MRs on real-world
datasets, such as DeepTest [4], DeepRoad [5], RMT [6], and
MetaSem [14], often applying computer vision tools to detect
violations.
Other studies apply MT in simulators [7], [15]. For example,
CoCoMEGA [7] uses handcrafted MRs and genetic operators
to explore scenario variations, while Baresi et al. [15] ma-
nipulate ego vehicle sensor inputs. However, both real-world
and simulation-based approaches rely heavily on manual MR
definition and follow-up test case validation [16].
AUTOMT automates this process by leveraging LLMs to
extract MRs from traffic rules and using a test case analysis
agent to verify their applicability to test cases, enabling fully
automated MT for ADSs. For simplicity, we focus on real-
world data, though the modular design supports simulator-
based agents for MR-specific follow-up generation.
B. Automated MR Generation
Prior studies have explored MR generation. RMT [6] and
MetaSem [14] manually parse traffic rules to derive MRs.
To reduce manual effort, automated techniques have been
proposed. Xu et al. [17] synthesize MRs from test cases,
but their method targets structured code testing and lacks
generalizability to ADS. Cho et al. [18] use a genetic algorithm
to expand a human-written MR set, while Zhang et al. [19]
introduce three abstract MR patterns, both still requiring expert
input and lacking generality.
LLMs exhibit strong reasoning and generalization abilities
[20], and have been applied to knowledge extraction [10], [21].
Recent work has explored LLM-based MR extraction [8], [9],
[16]. For example, Zhang et al. [12] generate MRs for ADS
modules using LLMs. Others extract MRs from artifacts [22]
or requirements [9]. However, due to LLM hallucinations,
directly generating MRs often yields many invalid cases.
Our work goes a step further by coordinating multiple
specialized agents for fully automated MT. An LLM-based
agent extracts MRs from local traffic rules into a RAG
database using Gherkin syntax and ontology elements [6],
[11]. A vision-language agent parses original test cases to
match applicable MRs, and a generation agent uses computer
vision tools to synthesize valid follow-up test cases based on
the matched MRs.III. APPROACH
A. Overview ofAUTOMT
AUTOMT comprises three modular components detailed in
the following sections. Section III-B describes the MR extrac-
tion agent (M-Agent), which leverages LLMs to construct a
retrieval-ready MR database from traffic rules. Section III-C
presents the test case analysis agent (T-Agent), which uses
a vision-language model to extract scenario semantics from
original test cases. Section III-D introduces the follow-up
generation agent (F-Agent), which retrieves applicable MRs
and synthesizes valid follow-up test cases using computer
vision techniques.
B. Metamorphic Relation Extraction Agent (M-Agent)
Fig. 2. The workflow of M-Agent
To enable automated MR extraction, AUTOMT introduces
the M-Agent. As shown in Figure 2, we define a rule parser
using Gherkin syntax [11], a predefined ontology, and LLMs.
Given a traffic rule, multiple LLM-based parsers generate
candidate MRs, which are validated via SelfCheckGPT [23] to
select the optimal one. The selected MRs are then embedded
into a RAG database.
1) LLM-based rule parser:To systematically extract MRs
from traffic rules, we design a structured rule parser
based on Gherkin syntax [11], following the “Given-When-
Then” pattern commonly used in behavior-driven development
(BDD) [24]. This format maps traffic rule components to three
key ontology elements:
•Given:Road Type(e.g., intersection, crosswalk, high-
way)
•When:Manipulation(e.g., adding traffic signals,
changing weather)
•Then:Ego-Vehicle Expected Behavior(e.g.,
slow down, turn left)
This structure ensures that each MR captures a driving
context, a scenario transformation, and the expected system
response, aligned with traffic regulations.
To define the ontology, we first use an LLM to extract
candidate elements from traffic rules using a prompting strat-
egy adapted from Shin [9], as shown in Table I. Then, we
refine and categorize them with two ADS experts. Conflicts are
resolved by removing duplicates and elements not grounded

3
TABLE I
THEPROMPT FORONTOLOGYELEMENTSEXTRACTION
Role Setting
You are an expert in traffic rules.
Prompt
1. Examples of ontology elements like Table II.
2. I want to derive Road network, Traffic infrastructure, Object, Environ-
ment from a rules document. Can you assist me? Please identify each
item that I did not list but is present in the rules document.
User:One rules document
TABLE II
PRE-DEFINED ONTOLOGY.
CategoryLevel-1
SubcategoryLevel-2 Subcategory
Road Type Road network Intersection, Crosswalk, Field path
ManipulationTraffic
infrastructureSign: STOP Sign etc., Light: Red Light
etc., Barriers: Guardrail etc., Line:
Crosswalk Markings etc.
Object Vehicle, Pedestrian, Cyclist
Environment Rain, Mud, Snowy, Fog, Night
Ego-Vehicle
Expected
Behavior/Slow down, Turn left, Turn right, Keep
current
in the rules. The resulting ontology constrains MR generation,
ensuring syntactic validity and semantic fidelity. Table II
presents the summarized ontology categories; the full version
is provided in theSection 1 of the supplementary material.
For example, consider traffic rule “Steady Red Light (Stop)
Stop before entering the crosswalk or intersection”. The parser
would extract the following MR:
MR example:
Given the ego-vehicle approaches to anintersection
When AUTOMTaddsared light on the roadside
Then ego-vehicle shouldslow down
In this example, theRoad Typeisintersection, defining
the driving context. TheManipulationisadds a red
light on the roadside, representing the scenario change. The
Ego-Vehicle Expected Behaviorisslow down, indi-
cating the ADS should reduce speed in the follow-up scenario
(with red light) compared to the source (without red light).
To extract MRs from traffic rules, we adopt a Chain-
of-Thought (CoT) prompting strategy [25] to guide LLMs
through structured reasoning, as shown in Table III. The pro-
cess begins by identifying theEgo-Vehicle Expected
Behavior, followed by extracting theRoad Typeand
Manipulationelements. These are then composed into a
Gherkin-style MR. To improve accuracy, we provide demon-
stration examples and key concept explanations for in-context
learning.
2) MR Validation:To address LLM hallucination, we adopt
SelfCheckGPT [23], which assumes that consistent outputs
across multiple generations indicate reliable knowledge. As
shown in Figure 2, we generate multiple candidate MRs
per traffic rule using different LLM-based rule parsers. Each
output is assessed using a validation prompt with three binary
questions (Table IV), scoring 0 for “yes” and 1 for “no.” We
compute the average score per MR and select the one with
the lowest hallucination score as the final output, ensuring
consistency and reliability across LLM generations.TABLE III
THE PROMPT FORLLM-BASED RULE PARSER
Role Setting
You are an expert in traffic rules and scene analysis. Metamorphic Testing
(MT) is a method used in autonomous vehicle testing. Your task is
to convert traffic rules into structured ”Given-When-Then” metamorphic
relations (MRs) for vehicle testing.
# Key Concepts #
1. traffic rule: Define how the ego-vehicle should behavioral in the specific
driving scenario.
2.Road Type: Road elements are specified in the traffic rule, such as
crosswalk.
3.Manipulation:“adds”objects specified in the traffic rule, such
as red light (those items can be added “on the road” or “by the road
side” based on the prior knowledge of LLM), or“replaces”environmental
conditions, such as a rainy day.
4.Ego-Vehicle Expected Behavior: The expected ego-vehicle
behavior in the traffic rule, such as slow down, turn right.
# EXAMPLE # User: Traffic rule: ”Steady Red Light (Stop) Stop before
entering the crosswalk or intersection”
Assistant: Given the ego-vehicle approaches to an intersection
When AUTOMT adds a red light on the roadside
Then ego-vehicle should slow down
Prompt
You are given:
1. Details of the MRs: ontology elements ofRoad Type,
ManipulationandEgo-Vehicle Expected Behavior.
2. To ensure consistency, follow a step-by-step process to extract the MR
from traffic rule.
Step 1, Determine one appropriateRoad Typeontology element based
on the rule. Step 2, Determine one appropriateManipulationon-
tology element based on the rule. Step 3, Determine the verb for
Manipulation, use“adds”for objects with optional presence (e.g.,
pedestrians, vehicles), and“replaces”for objects with mandatory presence
(e.g., weather, lighting conditions). Step 4, Determine one appropriate
Ego-Vehicle Expected Behaviorontology element based on the
rule.
Finally, compose the MR using the selected elements in the following
format:
Given the ego-vehicle approaches toRoad Type
When AUTOMTManipulation
Then ego-vehicle shouldEgo-Vehicle Expected Behavior
User:{One Traffic rule}
TABLE IV
THEPROMPT FORMR VALIDATION
Role Setting
# CONTEXT # Based on the list of close-ended yes or no questions,
generate a JSON answer.
# Key Concepts # Same as Table III.
Questions:
1. AreRoad Type,Manipulation, andEgo-Vehicle
Expected Behaviorall mentioned in the traffic rule?
2. Is the traffic rule supported by MR?
3. Are all parts of the MR consistent with each other?
Prompt
# EXAMPLE # User: Traffic rule: ”Steady Red Light (Stop) Stop before
entering the crosswalk or intersection”,
MR: ”Given the ego-vehicle approaches to an intersection
When AUTOMT adds a red light on the roadside
Then ego-vehicle should slow down”
Assistant: [”yes”, ”yes”, ”yes”]
User:{One Traffic rule:,MR:}
As shown in Table IV, the three validation questions as-
sess different aspects of each MR: (1) coverage of all three
ontology elements, (2) alignment with the original traffic
rule, and (3) internal and grammatical coherence. Since MRs
are synthesized via LLMs using predefined syntax, this step
ensures proper integration and correctness.

4
TABLE V
THEPROMPT FORT-AGENT
Prompt
# Analyze this driving scenario. Describe the time of day, weather
conditions, road type (such as intersection, crosswalk, etc.), and any
objects around the ego-vehicle. Reply format: time: , weather: , road type:
, objects:
Images Input
User:{image or a list of images}
3) Metamorphic Relation RAG Database:To align test
cases with MRs and generate valid follow-ups, we intro-
duce the MR-RAG database. Retrieval-Augmented Genera-
tion (RAG) improves LLM performance on domain-specific
tasks by grounding outputs in external knowledge [26]–
[28]. We compile extracted MRs into a structured CSV
with fields:Index,MRs,Road Type,Manipulation,
Ego-Vehicle Expected Behavior, andExecution
Count, where the latter tracks MR usage to avoid redundancy.
MRs are embedded using OpenAI embeddings [29] and stored
in a FAISS-based vector store [30], enabling the F-Agent to
retrieve semantically relevant MRs for follow-up test genera-
tion.
C. Test Case Analysis Agent (T-Agent)
Vision-language models (VLMs) [31] can act as interactive
visual agents. We leverage this capability in the T-Agent to an-
alyze test cases and generate structured representations. These
are then matched with the MR-RAG database by the F-Agent
to retrieve relevant MRs. Inspired by DRIVEVLM [32], the T-
Agent is prompted to describe the driving environment—e.g.,
weather, time, road conditions, and nearby objects—as shown
in Table V.
Next, the T-Agent combines its analysis with ground-truth
ego-vehicle data (e.g., speed and steering) from the source test
case to construct a structured test case representation. This
representation is formatted as JSON:
Example of test case representation
{
"Test Case Representation": {
"Time": "Afternoon",
"Weather": "Clear",
"RoadType": "Intersection",
"Objects": "Cars, buildings,
pedestrians, bicycles, trees",
"EgoVehicle": {
"Speed": "10.649 km/h",
"Steering Angle": "-3.689 rad"
}
}
}
D. Follow-up Test Case Generation Agent (F-Agent)
1) MR Match:As shown in Table VI, the F-Agent uses
the test case representation from the T-Agent to retrieve the
most relevant MR from the MR-RAG dataset, prioritizingTABLE VI
THEPROMPT FORF-AGENT WITHMR-RAG DATABASE
Role Setting
You are an assistant for question-answering tasks. Use the following pieces
of retrieved context to answer the question.
Prompt
#Given the test case description:T-Agent output, select one MR from
the retrieved context where:
1. TheTime,Weather,Road type, andObjectsin theT-Agent
outputshould best match those in the MR.
2. Ego-vehicle’s speed and steering angle should match in this MR.
3. Among all matched MRs, prefer the one with the lowestExecution
Countvalue.
User:{test case description}
those with the lowest execution count to reduce repetition. The
selected MR’sManipulationelement is then used to guide
image editing (for real-world data) or scenario modification (in
simulation).
2) Image Editing:In theManipulationontology ele-
ment, two image operations are defined: “add” and “replace.”
For “adds,” F-Agent uses FLUX.1-Fill [33], a state-of-the-
art diffusion model designed for localized image edits with
high visual fidelity. It accounts for lighting, shadows, and
object placement to seamlessly blend modifications into the
original image, improving upon prior metamorphic testing
techniques [6], [14]. FLUX.1-Fill [34] requires two inputs: an
editing prompt and a mask defining the editable region. Masks
are generated via semantic segmentation using the OneFormer
model [35], fine-tuned on Cityscapes [36], which classifies
pixels into classes such as person, rider, car, truck, bus,
train, motorcycle, and bicycle—key dynamic agents in driving
scenarios. For “replaces,” F-Agent uses InstructPix2Pix [37], a
text-guided model capable of making global scene edits (e.g.,
“make it rainy”) with realistic environmental changes, enabling
diverse test scenarios through natural language prompts.
To ensure temporal consistency, we avoid per-frame editing
and adopt VISTA [38], a video generation model trained on
driving data that synthesizes realistic sequences using the
modified image and vehicle dynamics (e.g., speed). Although
test cases generated by AUTOMT may not perfectly match
real-world fidelity, they preserve critical semantics. Since
added elements (e.g., pedestrians, traffic signs) come from real
data, unsafe ADS behavior (e.g., failing to slow) remains a
valid safety concern.
Our method is modular—image editing tools can be re-
placed with newer models as they emerge. We use FLUX.1-
Fill and InstructPix2Pix due to their strong performance at
the time of writing. Additionally, this framework can support
test generation in simulation by building libraries that translate
matched MRs into scenario scripts.
IV. EXPERIMENT
A. Research Questions
We evaluate AUTOMT through four research questions:
RQ1: Are the extracted MRs consistent with the underlying
traffic rules, as judged by human oracles?RQ2: How many
MR violations can AUTOMT detect across multiple ADSs

5
compared to state-of-the-art baselines?RQ3: Are the gener-
ated follow-up test cases semantically consistent with their
original counterparts?RQ4: Do the detected MR violations
align with human-perceived safety violations in the tested
ADSs?
B. General Experiment Setup
We evaluate AUTOMT on test sets from A2D2 [39] and
Udacity [40], two large-scale ADS datasets. A2D2 includes
217 cases from Germany (1920×1280), and Udacity contains
167 cases from California (640×480). To standardize and
accelerate processing, we downsampled all videos to 10 FPS,
which is sufficient for human-perceived fluency [41], resulting
in 10 frames per test case. To align with regional traffic
laws, we manually reviewed official rulebooks [42], [43],
identifying 38 distinct rules from Germany and 72 from
California (details of those traffic rules can be foundhere). For
our AI-agents, the MR generation agent (M-Agent) uses three
SOTA models: ChatGPT-4o [44], Claude 3.7 Sonnet [45], and
Qwen3-8B [46], with ChatGPT-4o also used for SelfCheck-
GPT validation. Both the test case analysis agent (T-Agent)
and follow-up generation agent (F-Agent) use ChatGPT-4o, a
leading vision-language model.
C. Specific Experiment Design
1) RQ1: Qualitative Evaluation of Extracted Metamorphic
Relations:We conducted a qualitative user study with three
MT experts to assess whether extracted MRs align with traffic
rules. Each expert reviewed all rules and their corresponding
MRs and answered:“Does the metamorphic relation correctly
align with the traffic rule?”Responses were recorded on a 5-
point Likert scale. We computed weighted Fleiss’ Kappa [21]
to measure inter-rater agreement by region.
2) RQ2: Quantitative Performance ofAUTOMT:As the
first fully automated MT pipeline, we compare AUTOMT
against three adapted baselines based on prior work. All
methods take a source test case as input and generate a follow-
up test case. Each experiment is run five times, and we report
averaged results with statistical analysis. The source code of
those baselines can be found in ourrepo.
•Auto MT Pipeline w/o Traffic Rules [12]: Uses
ChatGPT-4o to generate an MR directly from a test case
without traffic rules or predefined ontology. The model
is prompted with MT background and MR structure.
Follow-up test cases are generated using the same com-
puter vision tools as AUTOMT.
•Auto MT Pipeline w/ Traffic Rules [9]: Uses ChatGPT-
4o to generate MRs using traffic rules but without ontol-
ogy grounding. The model is similarly prompted with
MT knowledge and MR format. Follow-up cases are
generated using the same tools as AUTOMT.
•Auto MT with Manually Defined MRs: To compare
with expert-defined MRs, we reviewed prior works [6],
[7], [14] and identified 9 universal, mutually exclusive
MRs. For each test case, one MR is randomly selected
to generate a follow-up. This design aligns with our
approach and other baselines, which generate exactlyone follow-up per test case, ensuring a fair and consis-
tent comparison across methods. Random selection also
avoids over-representation of particular MRs and reflects
realistic scenarios where only one transformation may be
applied at a time. Those MRs include adding vehicles,
pedestrians, cyclists, and traffic lights or signs, as well as
changing the weather to include rainy days, snowy days,
and nighttime conditions.
We evaluate follow-up test cases using six ADSs. Four are
single-frame models commonly used in MT for ADSs [6],
[14]:
•PilotNet[47]: NVIDIA’s end-to-end driving model,
•Epoch[48]: a high-performing model from Udacity,
•ResNet101-ADS[6]: based on ResNet101 [49] with a
fully connected output layer,
•VGG16-ADS[6]: similar to above, using VGG16 [50].
Two models process multiple frames:
•CNN-LSTM[51]: a CNN combined with an LSTM and
a fully connected layer,
•CNN3D[51]: a 3D CNN with a fully connected output
layer.
All ADSs are trained on a combined A2D2 and Udacity
dataset with ground truth labels for steering angle (radians)
and speed (m/s). We use 80% for training, 10% for validation,
and 10% for testing. Images are resized to 320×160 pixels,
and training is run to convergence.
For RQ2, we use two metrics:Follow-up Test Case Vali-
dation RateandSafety Violation Rate. The validation rate
is the ratio of valid follow-up test cases to the total generated.
Validity is determined using three binary evaluation metrics; if
any metric is 0, the follow-up test case is considered invalid.
1)Scenario Alignment:We check whether the road type
remains consistent between the source and follow-up test
cases using ChatGPT-4o. A mismatch marks the follow-
up as invalid.
2)Logical Alignment:We assess consistency between
theGiven,When, andThencomponents of the MR
using SelfCheckGPT. Misaligned logic—e.g., contradic-
tory expected behaviors or context violations—results in
an invalid MR.
3)Manipulation Verification:We use the Difference Co-
herence Estimator [52] to verify whether the intended
Manipulation is visually reflected. Formally:
C(I original , Ifollow-up ,e) =(
1,if visual change aligns withe,
0,otherwise.
(1)
whereeis the manipulation prompt, andC(·)is computed
via a VLM. A return value of 0 marks the follow-up as invalid.
Safety Violation Rateis defined as the proportion of
follow-up test cases that trigger MR violations, which we treat
as indicators of safety violations in this study. For each test
case, ADSs predict frame-level speed and steering; we use
the median to mitigate outliers. We then compute prediction
variance across all ADSs. If theManipulationspecifies
Slow Down, the follow-up speed must fall below the variance
lower bound. ForKeep Current, both speed and steering

6
TABLE VII
QUANTITATIVE RESULT OF USER STUDY ON METAMORPHIC RELATION
GENERATED BYAUTOMT.
Region Strongly Agree Agree Neutral Disagree Strongly Disagree
Germany 74.4% 18.9% 1.1% 0% 5.6%
California 91.1% 3.3% 4.4% 1.1% 0%
must stay within bounds. ForTurn Left/Right, steering must
fall within the expected directional range. MR Violations
are treated as safety-critical behaviors. In RQ4, we further
investigate the alignment between MR violations and human-
perceived safety violations through a user study.
3) RQ3: Consistency of Generated Follow-up Test Cases:
To evaluate whether follow-up test cases are semantically
consistent with their corresponding source cases, we conducted
a user study on 118 randomly sampled valid test pairs. We
recruited 15 licensed drivers (aged 18–60) via Prolific [53],
with diverse demographics (details can be foundhere). Each
participant was shown a source and follow-up test case and
asked: “How realistic is the follow-up video (e.g., visual
quality, consistency with the original video)?” Responses were
recorded on a 5-point Likert scale, with justifications required
for ratings below “Neutral.” Inter-rater agreement was assessed
using weighted Fleiss’ Kappa [21].
4) RQ4: Alignment Between MR Violations and Perceived
Safety Violations:Using the same samples and participants
as RQ3, we evaluated whether MR violations align with
perceived safety. Participants answered: “Given the transfor-
mation, is the ADS’s predicted motion (e.g., speed or steer-
ing angle) in the follow-up video reasonable?” A mismatch
between MR-based and user judgment indicates a potential
false positive or false negative. Responses used a 5-point
Likert scale; ratings above “Neutral” suggest the motion was
reasonable. Disagreements required justification. Agreement
was measured using weighted Fleiss’ Kappa [21].
V. RESULT
A. RQ1: Evaluation of MR Extraction
Table VII shows human expert evaluations of MRs extracted
by AUTOMT.93.3%(Germany) and94.4%(California) of
MRs were rated above “Agree”, demonstrating strong align-
ment with traffic rules. Fleiss’ Kappa scores were0.421(mod-
erate agreement) for Germany and0.941(almost perfect) for
California. The lower agreement in Germany may stem from
unfamiliar patterns (e.g., St. Andrew’s cross) and translation
ambiguity. We further analyze outlier cases with “Strongly
Disagree” ratings to identify causes of disagreement.
Case Study: The traffic rule from Germany that received the
first “Strongly Disagree” rating states:
“Maximum Speed Limit Sign: Command or Pro-
hibition. A person driving a vehicle must not
exceed the speed limit indicated on the sign.”
The generated MR is:
“Given the ego-vehicle approaches any roads, when
AUTOMTadds a maximum 50km/h speed limit signon the roadside, then the ego-vehicle should slow
down. ”
However, the user believe that if the vehicle is already
traveling below the speed limit, requiring it to slow down
further is unreasonable.
A similar issue arises in the “Disagree” case involving a
traffic rule from California, which states:
“A green traffic signal light means GO.”
The extracted MR is:
“Given the ego-vehicle approaches an intersection,
whenAUTOMTadds a green traffic signal light
on the roadside, then the ego-vehicle should keep
current speed. ”
If the ego-vehicle is stationary, maintaining the same state
(i.e., staying still) after the light turns green contradicts the
intended rule—it should begin moving instead.
These two cases highlight a limitation: an MR may be
considered invalid depending on the current status of the ego-
vehicle. However, AUTOMT effectively addresses this issue
by leveraging the MR-RAG database to filter out original
test cases that are incompatible with the MR. Specifically, it
selects only those scenarios where the vehicle’s speed exceeds
the posted limit or where the vehicle is not stationary. This
demonstrates AUTOMT’s capability to handle such contextual
nuances effectively.
B. RQ2: Quantitative Performance ofAUTOMT
Table VIII presents the quantitative performance of AU-
TOMT and baseline methods across two regions, showing val-
idation rates and the underlying evaluation metrics. AUTOMT
outperforms all baselines, includingMT with Manual MR,
achieving up to 28.26% improvement in validation rate. Al-
though manual MRs perform slightly better in Germany and in
some metrics in California, they are expert-defined and limited
in diversity and adaptability. In contrast, AUTOMT produces
more context-specific and diverse MRs, as highlighted in our
diversity analysis.
Figure 3 shows that AUTOMT generates 46 distinct manipu-
lations (marked in blue) compared to only 9 inMT with Man-
ual MR(marked in red), demonstrating significantly greater
MR diversity. With support from the T-Agent and MR-RAG
database, AUTOMT more effectively matches MRs to varied
test cases, contributing to its superior overall performance.
The overall validation rate difference between AUTOMT
andMT with Manual MRis less than 0.6% across both
regions, other than this, AUTOMT achieves the highest scores
in semantic consistency between follow-up and source test
cases. To assess statistical significance, we conducted a t-test
on the validation rate. The p-values comparing AUTOMT with
MT without Traffic RuleandMT with Traffic Ruleare as
low as 0.002 and 0.0007, respectively, confirming a significant
improvement over these baselines.
To understand why baselines struggle with follow-up gen-
eration, we analyzed the causes of invalid cases across all
methods. A primary issue is failure in theManipulation
Verificationmetric, largely due to limitations of image editing
tools—even theMT with Manual MRbaseline is affected.

7
TABLE VIII
VALIDATION RATES AND CORRESPONDING UNDERLYING EVALUATION METRICS OF DIFFERENT BASELINES INGERMANY ANDCALIFORNIA. BEST
RESULTS ARE MARKED IN BOLD.
Method Overall Validation RateUnderlying Evaluation Metrics
Scenario Alignment Manipulation Verification Logical Alignment
Germany
AUTOMT40.74% 98.90% 46.36% 89.77%
MT without Traffic Rule36.13% 82.58% 44.89% 97.88%
MT with Traffic Rule31.52% 80.37% 39.08% 95.30%
MT with Manual MR 50.41%99.17%50.60%100%
California
AUTOMT 47.90% 97.72%50.18% 98.32%
MT without Traffic Rule32.10% 78.44% 43.59% 91.38%
MT with Traffic Rule19.64% 60.48% 32.57% 80.72%
MT with Manual MR38.80%99.76% 38.92%100%
Red light
Yellow lightGreen lightStop signYield sign
Work zone signPedestrian
Road work vehicle
Emergency vehicle flashing lightHeavy trafficRainy weatherWet road
Speed limit signStopped vehicleDust storm
Foggy weatherTraffic break
Red arrow signal
Right lane must turn right signRed and White Regulatory SignWrong way signNo entry signWarning signPerson walking
Person traveling on bicyclePedestrian on roadRoller skater
Wheelchair userGuide dog user
Emergency vehicle
Stationary flashing vehicleFlashing lights vehicleAnimalCyclist
Livestock
School bus yellow lights
Oncoming vehicle turns leftObstacle ahead
Construction vehicleNight
Road obstacle
School bus red lightssCollision
Snowy weatherDrizzle weather
Emergency vehicle ahead03691215182124Matched CountAutoMT
MT with Manual MR
Fig. 3. The manipulation operation distribution in the valid follow-up test case generated by AUTOMT (blue) and MT with Manual MR (red).
Moreover, the two baselines show significantly lower pass
rates than AUTOMT, indicating that their manipulations are
often not properly reflected.
Another major issue isScenario Alignment, where
both baselines perform substantially worse than AUTOMT,
which—along with the manual MR baseline—achieves over
98% alignment. These problems stem from the lack of on-
tology guidance in baseline MR generation, leading to overly
complex MRs that are difficult to execute correctly during
editing.
Figure 4 illustrates failure cases from the two baselines,
where their MRs do not passManipulation Verificationand
Scenario Alignment. Given the same source test case,MTwith Traffic Ruledescribes the scene as “a residential street
with parked vehicles on both sides”, andMT without Traffic
Ruleuses “a narrow urban street with parked cars”, both
overly detailed and inaccurate. In contrast, AUTOMT produces
a concise and accurate description guided by the ontology.
Similarly, the baselines generate overly specific manipu-
lation instructions that image editing tools struggle to ex-
ecute. AUTOMT leverages ontology-based selection to pro-
duce clearer, more actionable manipulations, resulting in valid
follow-up test cases.
ForLogical Alignment, both our method and the baselines
show high pass rates. However, none reach 100%, as some
MRs are only conditionally valid. Our MR-RAG database

8
TABLE IX
VIOLATION RATES OF DIFFERENT BASELINES ACROSS SIXADSS,EACH EVALUATED INGERMANY ANDCALIFORNIA. BEST RESULTS ARE MARKED IN
BOLD.
Method PilotNet Epoch ResNet101-based ADS
Germany California Germany California Germany California
AUTOMT 19.08%14.37%9.12%14.49%25.71%21.68%
MT without Traffic Rule10.41% 7.78% 8.85% 9.22% 11.24% 11.74%
MT with Traffic Rule12.17% 4.31% 4.33% 6.11% 5.16% 9.34%
MT with manual MR13.09% 10.54% 4.88% 10.66% 11.43% 15.69%
Method VGG16-based ADS CNN-LSTM CNN3D
Germany California Germany California Germany California
AUTOMT 27.83%17.01%16.68%12.69%18.16%11.62%
MT without Traffic Rule9.31% 11.86% 9.77% 7.07% 13.27% 6.35%
MT with Traffic Rule7.74% 6.47% 11.89% 3.71% 16.22% 3.83%
MT with manual MR16.22% 15.93% 12.53% 10.06% 16.31% 8.86%
(a) Original test case
 (b) Follow-up test case generated
by AUTOMT- MR: Given the ego-
vehicle approaches to any roads,
when method adds a school bus with
hazard light on the road, then ego-
vehicle should slow down.
(c) Follow-up test case generated by
MT with traffic rule - MR: Given
the ego-vehicle approaches to a res-
idential street with parked vehicles
on both sides, When method adds a
pedestrian walking close to the road
edge, Then ego-vehicle should slow
down.
(d) Follow-up test case generated by
MT without traffic rule - MR: Given
the ego-vehicle approaches to a nar-
row urban street with parked cars
on both sides, When method adds a
pedestrian crossing the street ahead,
Then ego-vehicle should slow down
Fig. 4. Comparison of the original test case, follow-up test case generated
by AUTOMT and baselines.
helps address this by filtering MRs based on the specific test
case context, as also observed in RQ1. Overall, the two LLM-
based baselines suffer from hallucinations and overly complex
MRs, making it difficult for image editors to generate valid
follow-up test cases. This highlights the importance of using
predefined ontology elements to guide MR generation and
improve validation rates.
We next examine the violation rate. Table IX presents
the safety violation rates across six ADSs in two regions.
AUTOMT achieves the highest violation rate in nearly all
settings, outperforming baselines by up to 20.55%. A t-test on
the violation rates yields p-values of 0.000193, 0.000096, and
0.000826 when comparing AUTOMT withMT without Traf-
fic Rule,MT with Traffic Rule, andMT with Manual MR,
respectively—indicating statistically significant differences.AlthoughMT with Manual MRslightly outperforms AU-
TOMT in validation rate, AUTOMT achieves higher violation
rates due to a broader range of manipulations (Figure 3).
The RAG-based matching aligns MRs more effectively with
original test cases, uncovering more diverse unsafe patterns.
Some manual MRs may overlap with existing scene elements
and thus fail to introduce new violations, as seen in Fig-
ure 5(b). In contrast, AUTOMT introduces impactful patterns
like collisions or extreme weather (Figures 5(c) and 5(d)),
which are more likely to trigger violations. This highlights
AUTOMT’s ability to avoid redundancy and generate diverse,
safety-relevant follow-up cases.
(a) Original test case
 (b) Follow-up test case generated by
MT with manual MR - MR: Given the
ego-vehicle approaches to any roads,
when method adds a vehicle on the
road, then ego-vehicle should slow
down.
(c) Follow-up test case generated by
AUTOMT - MR: Given the ego-
vehicle approaches to any roads, when
method adds a collision on the road,
then ego-vehicle should slow down.
(d) Follow-up test case generated by
AUTOMT - MR: Given the ego-
vehicle approaches to any roads, when
method replaces the weather with a
dust storm, then ego-vehicle should
slow down.
Fig. 5. Comparison of the original test case and follow-up test cases generated
by MT with manual MR and AUTOMT.
C. RQ3: Follow-up Test Case realism Evaluation
To report consistency ratings for 118 follow-up test cases,
we summarize descriptive statistics in the main paper and

9
Fig. 6. Case study of representative follow-up test cases that are rated as
unrealistic by users. For each row, the left image is the original test case, and
the right image is the follow-up test case.
provide two boxplots for user ratings and variance by region
(seeSection 8 and Figure 5 & 6 of the supplementary
material). The mean rating is 3.61, median is 4, and 64.5%
of cases are rated above “Neutral”, indicating most follow-ups
are viewed as semantically consistent. Inter-rater agreement
via weighted Fleiss’ Kappa isκ= 0.27, suggesting fair
agreement. We also analyze user feedback on low-rated cases,
presenting three examples in a case study.
Figure 6 shows three representative cases rated as unrealistic
by users. In the first case, the required transformation is
“add a group of pedestrians crossing the road”. AUTOMT
successfully adds pedestrians, but users noted missing trees,
commenting, “the scenario is not the same as the original.” The
tree removal was caused by FLUX.1-Fill [34], which clears
space for edits. Since the removed items are non-critical, the
test purpose remains valid. Our method is editing-tool agnostic
and can readily adopt more advanced tools.
The second case involves a transformation that adds a
pedestrian to the road. A user deemed it unrealistic, noting
that pedestrians rarely appear in such settings. However, while
uncommon, this is a valid corner case. AUTOMT is designed
to surface such rare but plausible scenarios, improving test
coverage by uncovering potential safety risks often missed by
conventional methods.
The third row involves a transformation of adding a yellow
roadside light. Users commented that the follow-up appeared
blurry and unnaturally smooth, stating it did “not resemble
a real driving scenario.” Such feedback reflects limitations in
current image editing and video generation tools, which is
caused by VISTA [38]. This phenomenon is caused by irra-
tional dynamics with respect to historical frames, a common
limitation of existing driving world models [38]. These open-
source driving world models lack sufficient priors about the
future motion tendencies of objects, especially when these
objects are generated by diffusion-based models. Despite this,
the majority of follow-up test cases were rated realistic, and
AUTOMT consistently exposed safety violations.D. RQ4: Safety Violation Validity Evaluation
Fig. 7. The confusion matrix of safety violation.
To assess whether MR violations correspond to real system-
level safety issues, we conducted a user study. Figure 7 shows
the confusion matrix comparing user judgments with MR-
based violations. The false positive and false negative rates are
14.8% and 19%, while the true positive and true negative rates
are 29.3% and 36.9%, respectively. The inter-rater agreement,
measured by weighted Fleiss’ Kappa, isκ= 0.25, indicating
fair agreement. Overall, users and MRs aligned on 66.2% of
the cases, demonstrating AUTOMT’s strong ability to uncover
real safety violations.
Fig. 8. Case study of representative follow-up test cases that exhibit
disagreement on whether the case constitutes a violation between the user
and MR. For each row, the left image shows the original test case, and the
right image shows the follow-up test case.
Despite the overall agreement, some disagreements re-
mained. Figure 8 shows two representative cases. The first
row depicts a false negative: the MR was not violated, but
users believed a violation occurred. Here, AUTOMT added
a roadside red light, and although the ego-vehicle slowed
down, users expected a complete stop. This reflects a limitation
of end-to-end ADSs, which rarely output zero speed due
to dataset biases. We plan to address this by extending our

10
pipeline to a simulator and integrating with CoCoMEGA [7]
for more nuanced testing.
The second row shows a false positive: the MR is violated,
but users perceive no violation. Snowy weather is added, yet
the ADS maintains its speed. While MR logic deems this un-
safe, users judged the conditions safe enough for steady speed.
This discrepancy reflects differing driving habits; however, in
principle, reduced speed is expected in snowy conditions to en-
sure safety. Overall, AUTOMTdemonstrates strong capability
in revealing real system-level safety violations, even if some
edge cases remain subject to human interpretation.
VI. VALIDITY DISCUSSION
A. External Validity
One limitation of our approach is its current focus on offline
test case generation using real-world images rather than full-
system simulation. While recent work such as CoCoMEGA [7]
explores MT in simulators, our approach reflects the ADS in-
dustry’s reliance on large-scale in-field data collection, which
still suffers from limited scenario diversity [54]. AUTOMT
complements this by automatically generating diverse test
cases through MR-guided augmentation. Our main contri-
bution lies in automated MR extraction from traffic rules,
source case analysis for MR applicability, and LLM-based
matching via RAG. These modular components can be adapted
to simulation-based environments in future work, allowing
substitution of image editors with simulation APIs.
B. Internal Validity
A key internal threat is the reliance on current image editing
tools. Even with manual MRs, follow-up test case validation
rates remain below 50% due to rendering limitations. As the
CV community advances, we plan to integrate stronger editing
backends. Another concern is LLM reliability. Despite using
multiple agents across models (GPT-4o [44], Claude [45],
Qwen [46]), hallucination remains an issue, as reflected in less-
than-perfect logical alignment scores. Ontology constraints
and BDD-style syntax help mitigate this issue, further strength-
ened by our multi-agent collaboration framework. Finally, our
violation assessments rely on outputs from VISTA, which
may introduce rendering artifacts. We acknowledge this and
plan to extend our pipeline to simulation-based testing, using
interfaces like CoCoMEGA for higher fidelity.
C. Construct Validity
We chose not to include other baselines in RQ3 and
RQ4 due to three reasons. First RQ2 already compared
AUTOMT against multiple baselines using three automated
metrics—Scenario Alignment, Logical Alignment, and Ma-
nipulation Verification—with DCE [21] quantifying semantic
change. Second, RQ3–RQ4 focus on whether the generated
follow-up cases are perceived as realistic and whether MR-
predicted violations align with human judgments. These out-
comes are influenced more by the image/video rendering tools
than by MR quality itself. Including baselines that rely on
alternative editing tools would introduce unfair competitionand obscure our primary contribution. Third, our technical
novelty lies in end-to-end automated MR generation and
matching—not in image synthesis—and baseline comparisons
in RQ3/RQ4 would misattribute quality differences to the
wrong system components.
VII. CONCLUSION
We propose the first multi-agent framework for extracting
MRs from traffic rules with consistency validation, stor-
ing them in a RAG-based repository, and using vision-
language models to analyze real-world datasets. The pipeline
transforms raw videos into structured test case representa-
tions and matches them with suitable MRs via reasoning
agents guided by diversity and applicability. Extensive ex-
periments—including expert MR validation, testing across
multiple ADSs with baselines, assessment of follow-up test
case consistency, and alignment of MR violations with human-
perceived safety issues—demonstrate the framework’s effec-
tiveness. Future work includes extending the modular frame-
work to simulation-based testing and integrating advanced
image editing tools to improve diversity coverage for real-
world datasets.
REFERENCES
[1] W. N. Caballero, D. Rios Insua, and D. Banks, “Decision support issues
in automated driving systems,” International Transactions inOperational
Research, vol. 30, no. 3, pp. 1216–1244, 2023.
[2] S. Zhai, S. Gao, L. Wang, and P. Liu, “When both human and machine
drivers make mistakes: Whom to blame?” Transportation research part
A:policy andpractice, vol. 170, p. 103637, 2023.
[3] L. Liang, Y . Deng, K. Morton, V . Kallinen, A. James, A. Seth,
E. Kuantama, S. Mukhopadhyay, R. Han, and X. Zheng, “Garl: Ge-
netic algorithm-augmented reinforcement learning to detect violations
in marker-based autonomous landing systems,” in 2025 IEEE/ACM
47th International Conference onSoftware Engineering (ICSE). IEEE
Computer Society, 2025, pp. 613–613.
[4] Y . Tian, K. Pei, S. Jana, and B. Ray, “Deeptest: Automated testing
of deep-neural-network-driven autonomous cars,” in Proceedings ofthe
40th international conference onsoftware engineering, 2018, pp. 303–
314.
[5] M. Zhang, Y . Zhang, L. Zhang, C. Liu, and S. Khurshid, “Deeproad:
Gan-based metamorphic testing and input validation framework for
autonomous driving systems,” in Proceedings ofthe33rd ACM/IEEE
International Conference onAutomated Software Engineering, 2018, pp.
132–142.
[6] Y . Deng, X. Zheng, T. Zhang, H. Liu, G. Lou, M. Kim, and T. Y . Chen,
“A declarative metamorphic testing framework for autonomous driving,”
IEEE Transactions onSoftware Engineering, 2022.
[7] H. Yousefizadeh, S. Gu, L. C. Briand, and A. Nasr, “Using cooperative
co-evolutionary search to generate metamorphic test cases for au-
tonomous driving systems,” IEEE Transactions onSoftware Engineering,
2025.
[8] J. Zhang, C.-a. Sun, H. Liu, and S. Dong, “Can large language models
discover metamorphic relations? a large-scale empirical study,” in 2025
IEEE International Conference onSoftware Analysis, Evolution and
Reengineering (SANER). IEEE, 2025, pp. 24–35.
[9] S. Y . Shin, F. Pastore, D. Bianculli, and A. Baicoianu, “Towards
generating executable metamorphic relations using large language mod-
els,” in International Conference ontheQuality ofInformation and
Communications Technology. Springer, 2024, pp. 126–141.
[10] V . S. A. Duvvuru, B. Zhang, M. Vierhauser, and A. Agrawal,
“Llm-agents driven automated simulation testing and analysis of
small uncrewed aerial systems,” in Proceedings oftheIEEE/ACM
47th International Conference onSoftware Engineering, ser. ICSE
’25. IEEE Press, 2025, p. 385–397. [Online]. Available: https:
//doi.org/10.1109/ICSE55347.2025.00223

11
[11] E. C. dos Santos and P. Vilain, “Automated acceptance tests as software
requirements: An experiment to compare the applicability of fit tables
and gherkin language,” in Agile Processes inSoftware Engineering and
Extreme Programming: 19th International Conference, XP2018, Porto,
Portugal, May 21–25, 2018, Proceedings 19. Springer, 2018, pp. 104–
119.
[12] Y . Zhang, D. Towey, and M. Pike, “Automated metamorphic-
relation generation with chatgpt: An experience report,” in 2023
IEEE 47th Annual Computers, Software, andApplications Conference
(COMPSAC). IEEE, 2023, pp. 1780–1785.
[13] T. Y . Chen, S. C. Cheung, and S. M. Yiu, “Metamorphic test-
ing: a new approach for generating next test cases,” arXiv preprint
arXiv:2002.12543, 2020.
[14] Z. Yang, S. Huang, T. Bai, Y . Yao, Y . Wang, C. Zheng, and C. Xia,
“Metasem: metamorphic testing based on semantic information of au-
tonomous driving scenes,” Software Testing, Verification andReliability,
vol. 34, no. 5, p. e1878, 2024.
[15] L. Baresi, D. Y . Xian Hu, A. Stocco, and P. Tonella, “ Efficient Domain
Augmentation for Autonomous Driving Testing Using Diffusion
Models ,” in 2025 IEEE/ACM 47th International Conference on
Software Engineering (ICSE). Los Alamitos, CA, USA: IEEE
Computer Society, May 2025, pp. 398–410. [Online]. Available:
https://doi.ieeecomputersociety.org/10.1109/ICSE55347.2025.00206
[16] C. Xu, S. Chen, J. Wu, S.-C. Cheung, V . Terragni, H. Zhu, and
J. Cao, “Mr-adopt: Automatic deduction of input transformation func-
tion for metamorphic testing,” in Proceedings ofthe39th IEEE/ACM
International Conference onAutomated Software Engineering, 2024, pp.
557–569.
[17] C. Xu, V . Terragni, H. Zhu, J. Wu, and S.-C. Cheung, “Mr-scout:
Automated synthesis of metamorphic relations from existing test cases,”
ACM Transactions onSoftware Engineering andMethodology, vol. 33,
no. 6, pp. 1–28, 2024.
[18] E. Cho, Y .-J. Shin, S. Hyun, H. Kim, and D.-H. Bae, “Automatic
generation of metamorphic relations for a cyber-physical system-of-
systems using genetic algorithm,” in 2022 29th Asia-Pacific Software
Engineering Conference (APSEC). IEEE, 2022, pp. 209–218.
[19] Y . Zhang, D. Towey, M. Pike, J. Cheng Han, Z. Quan Zhou, C. Yin,
Q. Wang, and C. Xie, “Scenario-driven metamorphic testing for
autonomous driving simulators,” Software Testing, Verification and
Reliability, p. e1892, 2024.
[20] X. Wang, H. Kim, S. Rahman, K. Mitra, and Z. Miao, “Human-llm
collaborative annotation through effective verification of llm labels,” in
Proceedings oftheCHI Conference onHuman Factors inComputing
Systems, 2024, pp. 1–21.
[21] Y . Deng, Z. Tu, J. Yao, M. Zhang, T. Zhang, and X. Zheng, “Target:
Traffic rule-based test generation for autonomous driving via vali-
dated llm-guided knowledge extraction,” IEEE Transactions onSoftware
Engineering, vol. 51, no. 7, pp. 1950–1968, 2025.
[22] C. Tsigkanos, P. Rani, S. M ¨uller, and T. Kehrer, “Large language models:
The next frontier for variable discovery within metamorphic testing?”
in2023 IEEE International Conference onSoftware Analysis, Evolution
andReengineering (SANER). IEEE, 2023, pp. 678–682.
[23] P. Manakul, A. Liusie, and M. J. Gales, “Selfcheckgpt: Zero-resource
black-box hallucination detection for generative large language models,”
arXiv preprint arXiv:2303.08896, 2023.
[24] Y . Deng, G. Lou, X. Zheng, T. Zhang, M. Kim, H. Liu, C. Wang,
and T. Y . Chen, “Bmt: Behavior driven development-based metamor-
phic testing for autonomous driving models,” in 2021 IEEE/ACM 6th
International Workshop onMetamorphic Testing (MET). IEEE, 2021,
pp. 32–36.
[25] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V . Le,
D. Zhou etal., “Chain-of-thought prompting elicits reasoning in large
language models,” Advances inneural information processing systems,
vol. 35, pp. 24 824–24 837, 2022.
[26] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel etal., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,” Advances in
neural information processing systems, vol. 33, pp. 9459–9474, 2020.
[27] M. Fatehkia, J. K. Lucas, and S. Chawla, “T-rag: lessons from the llm
trenches,” arXiv preprint arXiv:2402.07483, 2024.
[28] Y . Zhang, Y . Li, L. Cui, D. Cai, L. Liu, T. Fu, X. Huang, E. Zhao,
Y . Zhang, Y . Chen etal., “Siren’s song in the ai ocean: A survey on
hallucination in large language models,” Computational Linguistics, pp.
1–45, 2025.
[29] OpenAI, “Openai embeddings,” 2024, accessed: 2025-05-27. [Online].
Available: https://platform.openai.com/docs/guides/embeddings[30] J. Johnson, M. Douze, and H. J ´egou, “Billion-scale similarity search
with gpus,” IEEE Transactions onBigData, vol. 7, no. 3, pp. 535–547,
2019.
[31] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang,
S. Wang, J. Tang etal., “Qwen2. 5-vl technical report,” arXiv preprint
arXiv:2502.13923, 2025.
[32] X. Tian, J. Gu, B. Li, Y . Liu, Y . Wang, Z. Zhao, K. Zhan, P. Jia, X. Lang,
and H. Zhao, “Drivevlm: The convergence of autonomous driving and
large vision-language models,” arXiv preprint arXiv:2402.12289, 2024.
[33] M. Li, Y . Lin, Z. Zhang, T. Cai, X. Li, J. Guo, E. Xie, C. Meng, J.-
Y . Zhu, and S. Han, “Svdqunat: Absorbing outliers by low-rank com-
ponents for 4-bit diffusion models,” arXiv preprint arXiv:2411.05007,
2024.
[34] B. F. Labs, “Flux,” https://github.com/black-forest-labs/flux, 2024.
[35] J. Jain, J. Li, M. T. Chiu, A. Hassani, N. Orlov, and H. Shi, “Oneformer:
One transformer to rule universal image segmentation,” in Proceedings
ofCVPR, 2023, pp. 2989–2998.
[36] M. Cordts, M. Omran, S. Ramos etal., “The cityscapes dataset,” in
CVPR Workshop ontheFuture ofDatasets inVision, vol. 2, 2015, p. 1.
[37] T. Brooks, A. Holynski, and A. A. Efros, “Instructpix2pix: Learning
to follow image editing instructions,” in Proceedings oftheIEEE/CVF
Conference onComputer Vision and Pattern Recognition, 2023, pp.
18 392–18 402.
[38] S. Gao, J. Yang, L. Chen, K. Chitta, Y . Qiu, A. Geiger, J. Zhang, and
H. Li, “Vista: A generalizable driving world model with high fidelity
and versatile controllability,” Advances inNeural Information Processing
Systems, vol. 37, pp. 91 560–91 596, 2024.
[39] J. Geyer, Y . Kassahun, M. Mahmudi, X. Ricou, R. Durgesh, A. S. Chung,
L. Hauswald, V . H. Pham, M. M ¨uhlegg, S. Dorn etal., “A2d2: Audi
autonomous driving dataset,” arXiv preprint arXiv:2004.06320, 2020.
[40] Udacity, “Self driving car challenge 2 dataset,” Available at https:
//github.com/udacity/self-driving-car/tree/master/datasets/CH2, 2016.
[41] L. Liang, Y . Deng, Y . Zhang, J. Lu, C. Wang, Q. Sheng, and X. Zheng,
“Cueing: a lightweight model to capture human attention in driving,”
arXiv preprint arXiv:2305.15710, 2023.
[42] Bundesministerium der Justiz, “Straßenverkehrs-ordnung (stvo),”
2013. [Online]. Available: https://www.gesetze-im-internet.de/stvo
2013/StVO.pdf
[43] California Department of Motor Vehicles. (2025) California driver
handbook. Accessed: 2025-05-24. [Online]. Available: https://www.
dmv.ca.gov/portal/handbook/california-driver-handbook/
[44] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark,
A. Ostrow, A. Welihinda, A. Hayes, A. Radford etal., “Gpt-4o system
card,” arXiv preprint arXiv:2410.21276, 2024.
[45] Anthropic, “Claude 3.7 sonnet,” https://www.anthropic.com/news/
claude-3-7-sonnet, 2025, accessed: 2025-05-24.
[46] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao,
C. Huang, C. Lv etal., “Qwen3 technical report,” arXiv preprint
arXiv:2505.09388, 2025.
[47] M. Bojarski, “End to end learning for self-driving cars,” arXiv preprint
arXiv:1604.07316, 2016.
[48] C. Gundling, “cg23,” https://bit.ly/2VZYHGr, 2017.
[49] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” in Proceedings oftheIEEE conference oncomputer vision
andpattern recognition, 2016, pp. 770–778.
[50] K. Simonyan and A. Zisserman, “Very deep convolutional networks for
large-scale image recognition,” arXiv preprint arXiv:1409.1556, 2014.
[51] Z. Lai and T. Br ¨aunl, “End-to-end learning with memory models for
complex autonomous driving tasks in indoor environments,” Journal of
Intelligent &Robotic Systems, vol. 107, no. 3, p. 37, 2023.
[52] L. Baraldi, D. Bucciarelli, F. Betti, M. Cornia, N. Sebe, and R. Cuc-
chiara, “What changed? detecting and evaluating instruction-guided
image edits with multimodal large language models,” arXiv preprint
arXiv:2505.20405, 2025.
[53] Prolific, “General citation guidelines,” https://www.prolific.com, 2024,
accessed: September 2025.
[54] G. Lou, Y . Deng, X. Zheng, M. Zhang, and T. Zhang, “Testing
of autonomous driving systems: where are we and where should
we go?” in Proceedings ofthe30th ACM Joint European Software
Engineering Conference andSymposium ontheFoundations ofSoftware
Engineering, 2022, pp. 31–43.
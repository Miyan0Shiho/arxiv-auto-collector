# Quantifying Return on Security Controls in LLM Systems

**Authors**: Richard Helder Moulton, Austin O'Brien, John D. Hastings

**Published**: 2025-12-17 04:58:09

**PDF URL**: [https://arxiv.org/pdf/2512.15081v1](https://arxiv.org/pdf/2512.15081v1)

## Abstract
Although large language models (LLMs) are increasingly used in security-critical workflows, practitioners lack quantitative guidance on which safeguards are worth deploying. This paper introduces a decision-oriented framework and reproducible methodology that together quantify residual risk, convert adversarial probe outcomes into financial risk estimates and return-on-control (RoC) metrics, and enable monetary comparison of layered defenses for LLM-based systems. A retrieval-augmented generation (RAG) service is instantiated using the DeepSeek-R1 model over a corpus containing synthetic personally identifiable information (PII), and subjected to automated attacks with Garak across five vulnerability classes: PII leakage, latent context injection, prompt injection, adversarial attack generation, and divergence. For each (vulnerability, control) pair, attack success probabilities are estimated via Laplace's Rule of Succession and combined with loss triangle distributions, calibrated from public breach-cost data, in 10,000-run Monte Carlo simulations to produce loss exceedance curves and expected losses. Three widely used mitigations, attribute-based access control (ABAC); named entity recognition (NER) redaction using Microsoft Presidio; and NeMo Guardrails, are then compared to a baseline RAG configuration. The baseline system exhibits very high attack success rates (>= 0.98 for PII, latent injection, and prompt injection), yielding a total simulated expected loss of $313k per attack scenario. ABAC collapses success probabilities for PII and prompt-related attacks to near zero and reduces the total expected loss by ~94%, achieving an RoC of 9.83. NER redaction likewise eliminates PII leakage and attains an RoC of 5.97, while NeMo Guardrails provides only marginal benefit (RoC of 0.05).

## Full Text


<!-- PDF content starts -->

Quantifying Return on Security Controls in LLM
Systems
Richard Helder Moulton , Austin O’Brien , John D. Hastings
The Beacom College of Computer & Cyber Sciences
Dakota State University
Madison, SD, USA
rich.moulton@trojans.dsu.edu,{austin.obrien,john.hastings}@dsu.edu
Abstract—Although large language models (LLMs) are in-
creasingly used in security-critical workflows, practitioners lack
quantitative guidance on which safeguards are worth deploy-
ing. This paper introduces a decision-oriented framework and
reproducible methodology that together quantify residual risk,
convert adversarial probe outcomes into financial risk estimates
and return-on-control (RoC) metrics, and enable monetary com-
parison of layered defenses for LLM-based systems. A retrieval-
augmented generation (RAG) service is instantiated using the
DeepSeek-R1 model over a corpus containing synthetic personally
identifiable information (PII), and subjected to automated attacks
with Garak across five vulnerability classes: PII leakage, latent
context injection, prompt injection, adversarial attack generation,
and divergence. For each (vulnerability, control) pair, attack suc-
cess probabilities are estimated via Laplace’s Rule of Succession
and combined with loss triangle distributions, calibrated from
public breach-cost data, in 10,000-run Monte Carlo simulations to
produce loss exceedance curves and expected losses. Three widely
used mitigations, attribute-based access control (ABAC); named
entity recognition (NER) redaction using Microsoft Presidio;
and NeMo Guardrails, are then compared to a baseline RAG
configuration. The baseline system exhibits very high attack
success rates (≥0.98 for PII, latent injection, and prompt
injection), yielding a total simulated expected loss of $313k per
attack scenario. ABAC collapses success probabilities for PII
and prompt-related attacks to near zero and reduces the total
expected loss by≈94%, achieving an RoC of 9.83. NER redaction
likewise eliminates PII leakage and attains an RoC of 5.97, while
NeMo Guardrails provides only marginal benefit (RoC of 0.05).
Index Terms—Adversarial machine learning, Artificial intelli-
gence security, Large language models, Monte Carlo methods,
Privacy, Retrieval-augmented generation, Risk analysis.
I. INTRODUCTION
Large language models (LLMs) are increasingly deployed
across diverse fields, including healthcare, finance, education,
law, medicine, and cybersecurity [1], [2], [3], [4]. Their
widespread adoption is driven by their capabilities, such as
natural language understanding, contextual reasoning, text
generation, summarization, translation, and code synthesis [5],
[6], [7]. These models facilitate automation in customer ser-
vice, legal document drafting, medical report generation, and
software development [2], [3], [6]. Furthermore, their ability to
perform few-shot and zero-shot learning enables adaptation to
new tasks with minimal domain-specific training, accelerating
their integration into various industries [5], [8].Despite their advantages, LLMs also exhibit important se-
curity and trustworthiness challenges that limit their safe and
effective use. Studies have identified vulnerabilities including
adversarial attacks, prompt injection, data leakage, and model
bias, raising concerns about their robustness and real world
applicability [9], [10], [11], [12], [13]. In high stakes contexts,
these weaknesses can result in failures on sentiment related
tasks, unintended disclosure of sensitive information, suscepti-
bility to jailbreak and prompt injection attempts, hallucinated
or fabricated content, and biased or otherwise inappropriate
outputs with harmful societal consequences [14], [15].
A. Statement of the Problem
In their current form, LLMs lack consistent safeguards
capable of preventing these vulnerabilities in the face of
adaptive, sophisticated attacks [10], [12], [16]. Addressing
these risks requires reliable, targeted, and computationally
affordable mitigation strategies.
A variety of mitigation strategies have been proposed in-
cluding Attribute-Based Access Control (ABAC) [17] as a
higher-level system control, Named Entity Recognition (NER)
[1] as a component of content filtering or detection, and
differential privacy [18] as a technique to limit the leak-
age of sensitive information during training or generation.
Other proposed defenses include perplexity filtering [19], input
preprocessing (e.g., paraphrasing and retokenization) [12],
adversarial training [20], the use of auxiliary classifier models
(“detectors”) [21], red-teaming frameworks [22], and intention
analysis [23]. However, a critical gap remains in understanding
the real-world effectiveness and computational costs associated
with deploying or not deploying these safeguards, especially
against sophisticated and adaptive attacks.
B. Objectives of the Project
To address this gap, this study aims to systematically eval-
uate, quantify, and analyze the effectiveness and feasibility of
various safeguards designed to mitigate known vulnerabilities
in LLMs. Specifically, it will:
•Assess the effectiveness of known LLM vulnerability
safeguards by applying them to a model with well-
documented security flaws [24], [25]. This includes eval-
uating how well different mitigation techniques preventarXiv:2512.15081v1  [cs.CR]  17 Dec 2025

unintended information leakage, resist adversarial ma-
nipulations such as prompt injections and jailbreaks,
and reduce biases, hallucinations, and other undesirable
behaviors.
•Measure and analyze the costs of implementing or es-
chewing these safeguards, considering factors such as cost
of implementation and cost of breach. By quantifying
the resource demands of each approach, this study will
provide insights into the trade-offs between security risk,
costs, and practical deployment feasibility.
•Compare multiple mitigation techniques across differ-
ent vulnerability categories to determine their strengths,
weaknesses, and suitability for real-world applications.
This comparative analysis will highlight the most effec-
tive and cost-efficient strategies while identifying areas
where further optimization or refinement is needed.
•Evaluate the adaptability and resilience of safeguards
against evolving adversarial techniques, including auto-
mated jailbreak strategies, adaptive attacks, and sophis-
ticated prompt engineering methods. By testing these
defenses under varying conditions, this study will help
establish their robustness and reliability.
C. Significance of the Study
As LLMs become increasingly embedded in sensitive do-
mains, ensuring their trustworthiness and security is critical.
Although a wide range of mitigation strategies exist, including
prompt filtering; access control; and privacy-preserving train-
ing, current evaluations focus primarily on qualitative security
outcomes or functional correctness with limited attention to
modeling financial or operational risk [26], [27], [28]. As a
result, the cost effectiveness of mitigation strategies remains
underexplored. This research addresses that need by delivering
a comprehensive assessment of multiple safeguards under con-
trolled, adversarial conditions, quantifying both the security
benefits and economic costs. The findings will help developers,
security practitioners, and policy-makers make informed deci-
sions about which protections are most viable for real-world
use, enabling a more balanced approach to trustworthy AI that
considers not only security but also scalability, usability, and
resource constraints.
The main contributions of this work are as follows:
1) A reproducible methodology that integrates probabilistic
risk modeling, Monte Carlo simulation, and cost-benefit
analysis to evaluate LLM security safeguards.
2) An empirical comparison of three security controls
(ABAC, NER, and NeMo Guardrails) against five rep-
resentative vulnerability classes (personally identifiable
information (PII) leakage, latent context injection, ad-
versarial attack generation, direct prompt injection, and
divergence).
3) Decision-support metrics, including Return on Controls
(RoC), that quantify the reduction in expected loss per
unit cost, supporting practical security investment deci-
sions.The remainder of the paper is organized as follows. Section
II presents related work; Section III details the theoretical
framework and methodology; Section IV reports results and
discussion; Section V outlines future work; and Section VI
concludes the paper.
II. RELATEDWORK
A. Taxonomies and Attack Vectors
A growing body of research explores the risks LLMs pose
when exposed to adversarial behavior. Cui et al. [9] proposed
a comprehensive taxonomy of vulnerabilities organized across
four system modules: input, language model, toolchain, and
output. This framework provides a modular lens for analyzing
threats and has informed subsequent evaluations of LLM
behavior under attack.
Zhang et al. [4] surveyed the use of LLMs in cybersecurity,
highlighting both defensive applications and attack surfaces.
They emphasized the urgent need for systematic defenses as
LLMs are increasingly used in sensitive workflows. Specific
attack vectors such as prompt injection, suffix-based jailbreaks,
and adversarial prompting have been studied in depth.
Zou et al. [16] introduced transferable adversarial prompts
that can bypass alignment safeguards across models. Carlini
et al. [29] demonstrated that LLMs may memorize and leak
training data, raising serious concerns for privacy and data
protection.
Beyond suffix-based and universal jailbreaks, targeted jail-
breaking methods and datasets (e.g., [30], [31]) further indicate
that carefully crafted prompts can consistently elicit restricted
content across models and settings. These risks are amplified
in agentic configurations that use external tools or APIs, where
indirect prompt injection and cross-domain context attacks
enlarge the effective attack surface [11].
B. Benchmarks and Security Evaluation Tooling
Several benchmarking frameworks have been proposed to
evaluate LLM robustness. HarmBench [26] standardizes the
assessment of harmful prompt generation and model refusal
behavior. Brokman et al. [27] compared popular open-source
scanners such as Garak and Giskard, identifying inconsistent
coverage of known threats. Li et al. [32] focused specifically
on privacy risks and proposed PrivLM-Bench to measure
vulnerability to membership inference and data extraction.
Complementary suites such as PromptBench [28] and related
adversarial-prompt tooling [33] help probe jailbreakability and
robustness, while framework-level scanners like Garak [34]
support automated, repeatable security probing in practice.
C. Mitigation Strategies and Privacy-Preserving Approaches
Mitigation strategies have included both preventive and
reactive techniques. Feretzakis and Verykios [1] assessed the
use of inference-time filtering, Named Entity Recognition, and
access control mechanisms (e.g., RBAC and ABAC) to prevent
sensitive information disclosure. However, such approaches
may be limited in their coverage or scalability. He et al. [11]
further analyzed security risks in LLM agents, which combine

language models with external tools and APIs, exposing more
complex and less understood attack surfaces.
Privacy-preserving methods complement inference-time
controls: differentially private training and tooling (e.g., Ten-
sorFlow Privacy) [35] reduce the risk of memorization and
membership inference; federated and decentralized training
paradigms discussed in privacy surveys [32] mitigate central-
ized data exposure; and parameter-efficient adaptation (e.g.,
LoRA/QLoRA) can limit retraining footprint and reduce
inadvertent regressions introduced during fine-tuning [36].
Rule-based and policy-driven guardrail systems (e.g., NeMo
Guardrails) [37] are increasingly used to constrain outputs
at inference, though their effectiveness depends greatly on
configuration and integration depth.
D. Offensive Uses and Operational Threats
Recent work has explored how LLMs can be weaponized
for offensive operations. Mozes et al. [13] cataloged illicit
uses such as automated phishing and malware generation. Fang
et al. [38] demonstrated how LLM agents can autonomously
perform web exploitation tasks. Jailbreaking techniques that
manipulate models into generating restricted content were
surveyed by Jin et al. [31], revealing ongoing challenges in
enforcing alignment and compliance. Reports of real-world
leakage incidents and organizational restrictions further un-
derscore the operational costs of misconfigured retrieval, weak
output filtering, or insufficient privacy controls, reinforcing the
need for layered mitigations in production settings [39].
E. Gaps and Positioning
While prior work has identified and categorized a wide
range of vulnerabilities, most existing studies focus either on
attack demonstration or qualitative assessments. There remains
a gap in empirically evaluating how mitigation strategies per-
form under adversarial conditions, especially in terms of their
effectiveness, resource consumption, and operational trade-
offs. The present work aims to fill this gap by conducting
a structured evaluation of LLM security controls using adver-
sarial testing tools and resource monitoring metrics.
Consistent with observations from recent reviews, current
evaluation methods remain fragmented: many benchmarks
target isolated harms (e.g., jailbreakability or privacy leakage)
and rarely quantify the full cost–risk–latency trade space of
mitigations [26], [27], [28]. The present work combines ad-
versarial probing (via Garak) with probabilistic risk modeling
to compare controls on both security impact and economic
efficiency, complementing prior qualitative assessments.
III. THEORY ANDMETHODOLOGY
This study evaluates the effectiveness and cost efficiency of
several security mitigations for large language models through
a quantitative and decision-oriented framework. The objective
is not to identify new vulnerabilities but rather to measure
how well selected controls reduce the impact of known fail-
ure modes under controlled and repeatable conditions. The
methodology applies adversarial probing, probabilistic riskestimation, Monte Carlo simulation, and cost benefit analysis
to provide a coherent basis for comparing multiple mitigation
strategies.
The theoretical framework draws upon prior work in ad-
versarial robustness and trustworthy artificial intelligence, in-
cluding the work of Feretzakis and Verykios [1]. The approach
also incorporates the probing methods implemented in Garak
[34], which supplies adversarial prompts designed to expose
model vulnerabilities. This strategy is informed by research on
the transferability of jailbreak attacks across aligned language
models [16], which demonstrates that a general and systematic
procedure is needed to test the security of language model
systems.
A. Theory
The theoretical foundation of this work is rooted in three
main areas within adversarial machine learning and trustwor-
thy artificial intelligence.
•Adversarial robustness, which concerns the ability of
a language model to resist manipulation from crafted
prompts intended to induce harmful or unintended be-
haviors [40], [41], [42].
•Inference time defenses, which operate during model exe-
cution rather than during training. These include controls
intended to filter prompts or outputs in real time to reduce
the likelihood of harmful behavior [21], [43].
•Trustworthy artificial intelligence, which emphasizes
transparency, safety, accountability, and responsible be-
havior of model driven systems [44], [45].
These concepts determine the types of controls selected for
evaluation and guide the interpretation of the empirical results.
B. Methodology
This research applies a structured and repeatable method-
ology that combines empirical adversarial testing with prob-
abilistic risk modeling drawn from Hubbard and Seiersen’s
cyber risk methods [46]. The evaluation process contains four
main components: construction of the retrieval augmented
generation system, execution of adversarial probes, estimation
of attack success probabilities, and simulation of financial
losses.
1) System Construction and Baseline Test Procedure:A
FastAPI server was created with a single endpoint for retrieval
augmented question answering and fallback generation. Fig-
ure 1 illustrates the workflow. The system loads a pretrained
DeepSeek R1 Distill Qwen 1.5B model through the Hug-
gingFace transformers library and executes the model using
automatic device mapping and 16-bit floating point precision.
A FAISS vector store created from one hundred thousand lines
of synthetic names and social security numbers provides the
retrieval corpus. Retrieved documents must exceed a similarity
score threshold to be included in the context. If no retrieved
document meets this threshold, the system responds with direct
model generation.
A custom probe and detector was implemented for PII.
Garak [34] generated adversarial probes that were sent to the

Fig. 1. Workflow of the retrieval augmented application with fallback to direct
model generation
FastAPI endpoint. The probes and detectors other than PII
were taken from the default Garak configuration and included:
•atkgen, which uses a separate attack model to create
prompts intended to induce specific failures
•divergence, which attempts to cause the model to repeat
attacker supplied strings
•latentinjection, which embeds semi overt instructions
inside retrieved context and is used commonly in attacks
against retrieval augmented systems
•promptinject, which applies a broad range of prompt
instructions from the Prompt Inject library [47]
Each probe family was executed once per mitigation sce-
nario using the standard parameters of the tool. The number of
trials per probe varied by design and matched Garak defaults.
2) Security Control One: Attribute Based Access Control:
The first mitigation represents an access control approach
based on user or request attributes. In a typical attribute based
access control system, access decisions are determined by
evaluating attributes of the subject, object, and environment
against policy rules. In this research, attribute based access
control was modeled by rendering the PII corpus inaccessible
during retrieval. This simulates a policy that denies retrieval
of sensitive documents for requests that do not meet required
attributes.
Figure 2 shows the applied structure. Attributes are evalu-
ated before retrieval. If the request is not authorized, the model
receives no PII from the document store. The remainder of the
workflow proceeds unchanged.
Fig. 2. Workflow of the retrieval augmented application with attribute based
access control
3) Security Control Two: Named Entity Recognition Redac-
tion:The second mitigation applies named entity recognitionto documents before they are provided as context to the model.
Microsoft Presidio’s AnalyzerEngine and AnonymizerEngine
were used to detect and replace sensitive entities in any
retrieved content. Only documents that passed the similarity
threshold were processed. The remainder of the retrieval
augmented pipeline remained unchanged.
Figure 3 illustrates the integration of this approach into
the workflow. Retrieved documents pass through the named
entity recognition module before entering the context for
generation. If similarity criteria fail, the system falls back to
direct generation without redaction.
Fig. 3. Workflow of the retrieval augmented application with named entity
recognition based redaction
4) Security Control Three: NeMo Guardrails:The third
mitigation applies a rule based output filtering layer using
NVIDIA NeMo Guardrails. This method validates the gen-
erated response against a policy file defining allowed and
disallowed behaviors. The core retrieval augmented logic
is wrapped inside a Guardrails control flow so that every
response is checked before being returned to the user.
Figure 4 depicts this structure. The model generates a
response, then the Guardrails policy evaluates the content.
If any policy rule is triggered, the system returns a safe
alternative response. This control does not affect retrieval or
internal reasoning, and operates only at the final output stage.
Fig. 4. Workflow of the retrieval augmented application with NeMo
Guardrails
5) Estimating Attack Success with Laplace’s Rule of Suc-
cession:The Laplace Rule of Succession provides a conser-
vative estimate of attack success probability in the presence
of limited data. The rule is expressed as:

Pattack success =s+ 1
n+ 2,(1)
wheresis the observed number of attack successes andn
is the number of trials. This formulation prevents a zero prob-
ability estimate when no failures are observed and provides
a reasonable estimate for low sample sizes. These estimated
probabilities were used as inputs to the loss simulation model.
In this study, the terms attack success and model failure are
used as synonyms.
6) Monte Carlo Simulation and Loss Distribution Model-
ing:Uncertain loss values were represented using triangle
distributions with minimum, mode, and maximum parameters
taken from the IBM 2024 breach report [48]. These triangle
distributions are given in Table III. For each scenario, ten
thousand Monte Carlo trials were executed. Each trial first
determines whether an attack succeeds based on the Laplace
probability, and if so, samples a loss from the triangle dis-
tribution. A fixed random seed ensures reproducibility across
scenarios.
Loss Exceedance Curves were then created as shown in
equation 2.
P(Loss> L)(2)
These curves summarize both central tendencies and tail
behavior of the simulated loss distribution.
7) Quantifying Control Effectiveness with Return on Con-
trol:Return on Control provides a normalized measure of cost
efficiency and is defined as
RoC=E[Loss] baseline−E[Loss] control
Cost control.(3)
This metric indicates the reduction in expected loss obtained
per dollar spent on the control. It serves as the final step in
comparing the effectiveness of the three mitigation strategies.
Because the goal of this study is to compare the relative
reduction in risk produced by each control, all controls are
given an identical implementation cost of $30,000 USD.
This controlled assumption avoids introducing cost variability
across mitigations and ensures that differences in RoC arise
only from differences in their effect on expected loss.
8) Evaluation Conditions:All four configurations were
evaluated under identical conditions. The same probes, vector
store, retrieval threshold, model parameters, triangle distribu-
tions, simulation structure, and random seed were used in
every scenario.
C. Assumptions
Several assumptions guide the design of the system.
•Threats are modeled through adversarial prompts and
evaluated through the response of the language model.
•Adversaries have query access to the system but no access
to model weights.
•Inference time controls can reduce certain categories of
model failure without compromising utility.•The evaluation methods including Garak and Monte Carlo
simulation offer reasonable approximations of risk.
•The DeepSeek R1 model is representative of challenges
found in modern large language models.
D. Constraints
The study operates under several practical constraints.
•Only inference time controls are evaluated.
•The DeepSeek R1 model is used exclusively.
•Experiments are limited to available compute resources.
•Garak results are qualitative and require probabilistic
modeling to estimate true risk.
E. Validation
Validation of the methodology used several components:
•Adversarial testing with five probe families
•Verification of the custom detector for PII
•Automated scoring through Garak for consistent labeling
•Reproducibility through controlled experimental condi-
tions
This framework provides a consistent basis for evaluating
the three controls and prepares the ground for the quantitative
results in Section IV .
IV. RESULTS ANDDISCUSSION
This section presents the quantitative results of the vulnera-
bility testing and risk modeling. Each of the following subsec-
tions detail one of the controls starting with the baseline. For
comparison purposes, Table I presents the vulnerability test
results and LRS across all configurations. Table II summarizes
simulated expected losses and RoC for all configurations. Loss
exceedance curves for all five attack types, comparing the
baseline and each control, are shown in Figs. 5-9 (Appendix
A). The triangular loss distributions assumed for each vulner-
ability type are held fixed for all scenarios.
A. Baseline RAG Model
Testing the baseline RAG enabled system with the Garak
probes revealed that the model was highly susceptible to all
major categories of attack. As shown in Table I in the baseline
columns, failure rates for PII leakage, latent injection, and
prompt injection reached the maximum observed values: all
50 PII attempts, all 160 latent injection attempts, and all
500 prompt injection attempts succeeded. The corresponding
Laplace adjusted success probabilities were therefore 0.980,
0.993, and 0.998, respectively, which indicates that these
attacks should be treated as essentially certain to succeed under
baseline conditions. Divergence also occurred at a high rate,
with 138 failures in 180 attempts, and attack generation probes
succeeded in 1 of 25 attempts, confirming that even without
any additional controls the system exhibited limited robustness
across all tested vulnerability types. These baseline success
probabilities form the inputs to the simulated expected losses
reported in Table II, which serves as the reference point for
evaluating each control configuration.

For each vulnerability type, Table I reports the total num-
ber of probe attempts (“Trials”), the number of successful
attacks (“Failures”), and the resulting Laplace adjusted success
probability (“LRS”) for the baseline and each controlled
configuration.
The loss exceedance curves for the baseline configuration,
shown in Figs. 5–9, further illustrate the implications of these
findings. Because the Laplace estimates for several attack
categories are very close to one, the baseline curves output
by the Monte Carlo simulations largely track the shapes of
the respective triangle distributions. Although the ranges differ
by vulnerability type, the distributions share a right skew
with substantial mass in the upper tail, particularly for PII
leakage and latent injection, which leads to steep curves and
pronounced tail risk in the baseline loss exceedance curves.
Taken together, Table I, Table II, and Figs. 5–9 establish
the baseline system as a reference point for the subsequent
analyses. Because most baseline vulnerabilities show very high
or nearly certain probabilities of success, any effective control
must produce visible changes in both the failure counts and the
Laplace adjusted probabilities, as well as a downward shift in
the corresponding loss exceedance curves, especially for high
impact categories such as PII leakage, latent injection, and
prompt injection. The degree to which each control achieves
these changes defines its comparative effectiveness in the
subsections that follow.
B. RAG Model with ABAC Control
Figure 2 illustrates the ABAC workflow incorporated into
the RAG pipeline. Before a response is generated, user and
request attributes are evaluated against a policy store, and
requests that fail these checks are denied or rewritten before
reaching the model. In effect, ABAC restricts the categories
of documents, prompts, or retrieval contexts that the model is
allowed to use, thereby limiting the opportunity for harmful
inputs to propagate through the system. The control operates
upstream of both retrieval and inference, so it can prevent
high-risk context from entering the generation path at all.
The impact of this control is reflected directly in Table I.
Relative to the baseline, ABAC eliminates all observed failures
for PII leakage (50 to 0) and prompt injection (500 to 0),
and dramatically reduces latent-injection failures from 160 to
18. These are substantial reductions, and the corresponding
Laplace-adjusted success probabilities fall from near certain-
ties (0.980, 0.998, and 0.993) to values near zero (0.019,
0.001, and 0.117). Two categories exhibit slight increases in
failures, divergence (138 to 146) and attack generation (1 to
2), but the absolute magnitudes remain small compared to the
categories where ABAC provides large improvements. These
changes indicate that ABAC is highly effective at preventing
the most consequential vulnerabilities, with only minor trade-
offs elsewhere.
Figures 5–9 show the corresponding loss exceedance curves.
For the vulnerability types substantially mitigated by ABAC,
PII leakage; latent injection; and prompt injection, the curves
shift downward by more than an order of magnitude across theentire range of exceedance probabilities. This displacement
indicates not only a reduction in expected loss but also a
meaningful reduction in tail risk, particularly for PII leakage,
whose baseline curve shows the steepest and largest upper tail
in the system. In contrast, the divergence and attack-generation
curves remain close to their baseline positions, consistent with
the small changes in their underlying success probabilities. The
LECs therefore mirror the failure-pattern changes in Table I,
demonstrating strong control effectiveness concentrated in the
highest-impact categories.
These changes propagate into the simulated expected losses
reported in Table II, where the total expected loss decreases
from $313,712 under the baseline to $18,966 with ABAC in
place. Using an estimated annualized implementation cost of
$30,000, the resulting Return on Control (RoC) is 9.83, mean-
ing that every dollar invested in ABAC yields an estimated
$9.83 reduction in loss. The RoC summarizes the combined
effect of the large reductions in attack success for PII, latent
injection, and prompt injection, which dominate the system’s
overall financial risk profile. As the subsequent subsections
show, no other control produces a comparable reduction in
both failure rates and loss exceedance behavior.
C. RAG Model with NER Control
As shown in Fig. 3, the NER control processes each
retrieved document during inference to identify and redact
entities that may contain sensitive information such as names
or contact details before the content is passed to the language
model. When retrieval does not return any document that
meets the similarity threshold, the system falls back to direct
generation without redaction, and a flag records whether
redaction occurred. In this way, the control is designed to
reduce the risk of sensitive data exposure while preserving
the overall structure of the RAG workflow.
The effect of this control on vulnerability outcomes is sum-
marized in Table I. Relative to the baseline, NER eliminates
all observed PII leakage, reducing the number of failures
from 50 to 0 and lowering the Laplace adjusted success
probability from 0.980 to 0.019. Attack generation also im-
proves modestly, with failures decreasing from 1 to 0 and
the corresponding probability dropping from 0.074 to 0.037.
In contrast, the control leaves divergence, latent injection,
and prompt injection unchanged: divergence failures remain
at 138 of 180 attempts, latent injection remains at 160 of 160
attempts, and prompt injection remains at 500 of 500 attempts,
with the same high Laplace adjusted probabilities as in the
baseline. These patterns indicate that NER provides targeted
protection against PII leakage, with limited effect on the other
failure modes.
The loss exceedance curves in Figs. 5–9 reflect this selective
mitigation. The curve for PII shifts sharply downward across
the full range of exceedance probabilities compared to the
baseline, indicating both lower expected loss and reduced tail
risk for that category. The attack generation curve also shows
a small downward movement, consistent with the modest
reduction in its success probability. In contrast, the curves for

TABLE I
LLM VULNERABILITYTESTRESULTS ANDLAPLACE-ADJUSTEDSUCCESSPROBABILITIESACROSSCONFIGURATIONS
Vulnerability type Trials Baseline ABAC NER NeMo
Failures LRS Failures LRS Failures LRS Failures LRS
Attack generation (Atkgen) 25 1 0.074 2 0.111 0 0.037 1 0.074
Divergence 180 138 0.763 146 0.807 138 0.763 143 0.791
Latent injection 160 160 0.993 18 0.117 160 0.993 160 0.993
PII 50 50 0.980 0 0.019 0 0.019 50 0.980
Prompt injection 500 500 0.998 0 0.001 500 0.998 500 0.998
TABLE II
SIMULATEDEXPECTEDLOSSES ANDRETURN ONCONTROL(ROC)FORALLCONFIGURATIONS
Vulnerability type Baseline EL ($) ABAC EL ($) ABAC RoC NER EL ($) NER RoC NeMo EL ($) NeMo RoC
Atkgen 2,598 3,838 -0.04 1,208 0.05 2,624 0.00
Divergence 2,863 3,021 -0.01 2,807 0.00 2,894 0.00
Latent injection 73,310 8,856 2.15 73,043 0.01 73,048 0.01
PII 181,223 3,157 5.94 3,685 5.89 180,077 0.04
Prompt injection 53,718 94 1.78 53,682 0.00 53,613 0.00
Total313,712 18,966 9.83 134,425 5.97 312,256 0.05
TABLE III
ESTIMATEDTRIANGLEDISTRIBUTIONS FORLOSS BYLLM
VULNERABILITYTYPE
Vulnerability Type Min ($) Mode ($) Max ($)
Atkgen 500 5k 100k
Divergence 100 1k 10k
Latent Injection 1k 20k 200k
PII 5k 50k 500k
Prompt Injection 1k 10k 150k
latent injection, prompt injection, and divergence closely track
their baseline counterparts, since the underlying probabilities
remain nearly identical. As a result, the overall shape of the
combined risk profile is still dominated by categories that NER
does not address.
These changes are carried through to the simulated expected
losses in Table II. The total expected loss decreases from
$313,712 under the baseline configuration to $134,425 with
NER, largely due to the removal of high impact PII losses. Us-
ing an estimated annualized implementation cost of $30,000,
the resulting Return on Control is 5.97, which indicates that
each dollar spent on NER yields an estimated $5.97 reduction
in loss. Although this represents a strong cost effectiveness for
environments where PII exposure is the primary concern, the
persistence of high latent injection and prompt injection risk
suggests that NER alone is insufficient to address the broader
vulnerability landscape characterized in the baseline results.
D. RAG Model with Guardrails Control
Figure 4 shows the integration of NeMo Guardrails into the
RAG pipeline. The control sits at the response stage, applying
an output filtering layer designed to block responses that vio-
late predefined policy rules. Because it operates after retrieval
and generation, the control does not prevent harmful context
from being retrieved or incorporated into model reasoning;instead, it attempts to filter problematic content only at the
final stage before returning the response to the user.
The results of this control are summarized in Table I. In
contrast to ABAC and NER, NeMo shows minimal change
relative to the baseline. Latent injection and prompt injection
remain at their maximum observed failure counts (160 and
500, respectively), with Laplace adjusted success probabilities
unchanged at 0.993 and 0.998. PII leakage likewise remains
fully successful (50 of 50 failures), yielding the same high
LRS value of 0.980. Divergence shows a slight increase in
failures (138 to 143), and attack generation remains effectively
unchanged (1 to 1). Taken together, these patterns indicate that
the tested configuration of NeMo does not materially reduce
the vulnerability of the system to the categories of attacks
identified in the baseline. This is consistent with the control’s
focus on output constraints rather than restricting the model’s
internal reasoning or the content of retrieved documents.
The loss exceedance curves in Figs. 5–9 reinforce this
interpretation. For each vulnerability type, the NeMo curves
nearly overlap their baseline counterparts across the full range
of exceedance probabilities. In particular, the heavy upper
tails of the PII, latent injection, and prompt injection curves
remain unchanged, reflecting the lack of reduction in their
underlying success probabilities. The divergence and attack
generation curves show only minimal shifts, consistent with
the small changes in those categories. As a result, the overall
risk profile of the system remains dominated by unmitigated
vulnerabilities, and the control does not meaningfully alter the
distribution of potential losses.
These outcomes translate directly into the simulated ex-
pected losses reported in Table II. The total expected loss
decreases only marginally, from $313,712 under the baseline
configuration to $312,256 with NeMo Guardrails. Given an
estimated annualized implementation cost of $30,000, the
resulting Return on Control is only 0.05, indicating negligible
financial benefit.

Because NeMo does not modify the success probabilities
of the highest impact vulnerabilities, its risk-reduction perfor-
mance is limited in this setting. Further tuning, broader policy
coverage, or deeper integration earlier in the RAG pipeline
may be necessary for the control to produce meaningful
reductions in model vulnerability.
E. Comparative Synthesis of Control Effectiveness
The results from all three control configurations reveal
clear differences in their ability to reduce model vulnerability
and financial risk. The baseline system exhibited very high
probabilities of failure for several attack categories, which
resulted in heavy tail behavior in the corresponding loss
exceedance curves. This pattern established a reference point
that any effective control needed to shift by producing both
fewer failures and reduced tail risk.
ABAC produced the most substantial improvements. It
eliminated all failures related to PII and prompt injection
and greatly reduced failures for latent injection. The loss
exceedance curves for these categories showed large down-
ward shifts across the entire range of exceedance probabilities.
These changes explain the large reduction in total expected
loss that appears in Table II. ABAC therefore provides broad
and substantial risk reduction in this setting.
NER also produced a marked reduction in PII leakage but
did not reduce failures for latent injection, prompt injection,
or divergence. As a result, only the PII loss exceedance
curve shifted downward, while the curves for the remaining
categories remained close to their baseline forms. The total
expected loss for NER was therefore substantially lower
than the baseline, although the reduction was limited by the
persistence of the other high impact vulnerabilities.
NeMo Guardrails provided only very limited improvement
in any attack category. The loss exceedance curves for all vul-
nerabilities were nearly identical to their baseline counterparts.
Because the control did not reduce the success probabilities
of the highest impact vulnerabilities, the total expected loss
changed very little. In this configuration, NeMo offers little
risk reduction for the types of failures evaluated here.
Across all results, ABAC provided the most extensive reduc-
tion in both failure likelihood and financial risk, followed by
NER with selective but meaningful improvement. NeMo did
not reduce risk in a measurable way. These findings demon-
strate that controls which regulate the information available
to the model during retrieval and processing can change the
risk profile significantly, while controls that operate only at
the response stage may be less effective for the vulnerability
types examined in this study.
V. FUTUREWORK
This research presents a methodology for quantifying the
risks associated with language model vulnerabilities and as-
sessing the effectiveness of several LLM mitigation strategies.
However, numerous opportunities exist to extend and deepen
this work. One significant area for future research involvesexpanding the test suite used for vulnerability detection. Al-
though the current implementation leverages Garak’s default
and custom probes effectively, new detectors and probes could
be developed to assess additional failure modes, such as
multilingual leakage, model inversion, or bias amplification,
thereby broadening the scope of empirical risk discovery.
Another critical direction is the creation of a more quan-
tifiable and scientifically grounded testing framework. While
Garak offers broad coverage and insightful qualitative signals,
it does not natively support reproducible benchmarking or
statistical significance testing across runs. A next-generation
testing tool could incorporate probabilistic scoring, normaliza-
tion techniques, and configurable control conditions to enable
robust comparisons between models and configurations. Such
a tool would bridge the gap between practical red-teaming and
academically rigorous experimentation.
Further investigation is also warranted into the capabilities
and robustness of NeMo Guardrails. This research evaluates
the utility of Guardrails for applying output-level constraints;
although its current configuration showed limited impact under
tested scenarios, the underlying Colang scripting language,
used to define safety; factuality; and behavior rules, remains
underexplored. Future work could evaluate the language’s ex-
pressiveness, composability, and resilience against adversarial
bypass attempts, potentially positioning Colang as a formalism
for secure LLM governance.
Additionally, integrating alternative evaluation platforms
such as PrivBench, HarmBench, and similar datasets would
strengthen the external validity of this framework. These tools
provide curated prompts and ground-truth expectations for
privacy, toxicity, and fairness harms, making them well suited
for comparative validation. By incorporating such benchmarks,
researchers could triangulate Garak-derived results and better
characterize the limitations of each tool.
Moreover, the research applied controls primarily at the
retrieval and output stages. There remains significant oppor-
tunity to explore control points within the language model
architecture itself, or during initial input processing.
Finally, the methodology should be applied to a broader
array of LLMs. Testing across models of varying scale, ar-
chitecture, and training paradigms, including open-source and
proprietary systems, would yield insights into how vulnerabil-
ity profiles differ and which mitigations generalize effectively.
This line of research would also support the development
of model selection criteria based on risk, complementing
performance-based evaluation metrics with robust security
considerations.
VI. CONCLUSION
This research pursued four core objectives aligned with
evaluating the security posture of LLM-based retrieval sys-
tems. These objectives were addressed through the design,
implementation, and empirical evaluation of a FastAPI-based
retrieval system incorporating the DeepSeek-R1 language
model.

To meet the first objective, the system was tested against
a suite of probes targeting personally identifiable information
leakage, prompt injections, attack generation, latent context
manipulation, and output divergence. Initial results confirmed
that the baseline system was highly susceptible to multiple
classes of attack, particularly in the absence of safeguards
around document retrieval and output regulation. Subsequent
testing evaluated the effectiveness of applied mitigations,
demonstrating measurable reductions in vulnerability exposure
for access restrictions and entity redaction, though the response
filtering mechanism showed minimal impact under the tested
configuration.
The second objective was fulfilled by modeling the proba-
bilistic risks associated with adversarial attacks and the finan-
cial consequences of failing to mitigate them. Using Laplace’s
Rule of Succession, the likelihood of successful exploitation
was estimated based on observed probe outcomes. These
probabilities were combined with loss estimates drawn from
industry-informed triangle distributions to reflect potential
financial impacts. Monte Carlo simulations were then used to
model a range of threat scenarios, accounting for uncertainty
in both attack success rates and damage severity. The resulting
loss exceedance curves provided visibility into worst-case
outcomes and the long-tail nature of AI-related breach risks,
offering a foundation for assessing whether mitigation is
warranted under varying operational threat conditions.
The third objective was addressed by translating the model’s
quantitative outputs into actionable insights to inform decision-
making. Rather than emphasizing simulation mechanics, the
analysis prioritized comparing mitigation strategies in terms of
their practical impact on risk exposure and resource efficiency.
Return on control metrics enabled ranking of safeguards by
cost-effectiveness, helping identify which controls provided
the greatest reduction in projected loss relative to their imple-
mentation burden. This comparative framework supports real-
world deployment decisions by aligning technical effective-
ness with economic viability, offering a structured approach
to selecting appropriate defenses across varying operational
contexts.
The final objective was accomplished by evaluating the
adaptability and resilience of three mitigation strategies,
attribute-based access control; named entity recognition filter-
ing; and NeMo Guardrails, under evolving adversarial tech-
niques, including automated jailbreaks and advanced prompt
engineering attacks. Each control was subjected to a battery
of tests simulating dynamic threat conditions. To quantify
their cost-effectiveness, return on control was calculated for
each strategy. Access control and redaction proved both effec-
tive and cost-efficient, while NeMo Guardrails yielded only
marginal benefits, highlighting the importance of configuration
and context in determining control efficacy.
The results of this study have important real-world im-
plications. Organizations that deploy LLMs in regulated or
risk-sensitive environments—such as healthcare, finance, and
government—can use this framework to select and justify
security mitigations. By translating probe-level vulnerabilitiesinto projected financial losses and return on control metrics,
the framework aligns technical decisions with executive-level
risk acceptance and cost-benefit thresholds. Additionally, the
reproducibility of this approach makes it a viable candidate
for formalizing LLM security audits, product evaluations,
or procurement requirements. As AI regulation evolves, the
quantitative lens offered by this study may inform both internal
policy and external compliance standards.
VII. AVAILABILITY
All code, models, and data are available in a version-
controlled GitHub repository [49].
REFERENCES
[1] G. Feretzakis and V . S. Verykios, “Trustworthy ai: Securing sensitive
data in large language models,”AI, vol. 5, no. 4, pp. 2773–2800, 2024.
DOI: 10.3390/ai5040134
[2] J. Lai, W. Gan, J. Wu, Z. Qi, and P. S. Yu, “Large language models in
law: A survey,”AI Open, vol. 5, pp. 181–196, 2024,ISSN: 2666-6510.
DOI: https://doi.org/10.1016/j.aiopen.2024.09.002
[3] H. Zhou et al.,A survey of large language models in medicine:
Progress, application, and challenge, 2023. arXiv: 2311 . 05112
[cs.CL].
[4] J. Zhang et al., “When LLMs meet cybersecurity: A systematic
literature review,”Cybersecurity, vol. 8, no. 1, pp. 1–41, 2025.DOI:
10.1186/s42400-025-00361-w
[5] T. B. Brown et al., “Language models are few-shot learners,” in
Proceedings of the 34th International Conference on Neural Informa-
tion Processing Systems (NIPS ’20), Vancouver, BC, Canada: Curran
Associates Inc., 2020,ISBN: 9781713829546. [Online]. Available:
https : / / proceedings . neurips . cc / paper files / paper / 2020 / file /
1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf
[6] R. Bommasani et al.,On the opportunities and risks of foundation
models, 2022. arXiv: 2108.07258[cs.LG].
[7] M. Chen et al.,Evaluating large language models trained on code,
2021. arXiv: 2107.03374[cs.LG].
[8] S. Bubeck et al.,Sparks of artificial general intelligence: Early
experiments with gpt-4, 2023. arXiv: 2303.12712[cs.CL].
[9] T. Cui et al.,Risk taxonomy, mitigation, and assessment benchmarks
of large language model systems, 2024. arXiv: 2401.05778[cs.CL].
[10] E. Derner, K. Batisti ˇc, J. Zah ´alka, and R. Babu ˇska, “A security risk
taxonomy for prompt-based interaction with large language models,”
IEEE Access, vol. 12, pp. 126 176–126 187, 2024.DOI: 10 . 1109 /
ACCESS.2024.3450388
[11] F. He, T. Zhu, D. Ye, B. Liu, W. Zhou, and P. S. Yu, “The emerged
security and privacy of llm agent: A survey with case studies,”ACM
Comput. Surv., vol. 58, no. 6, Dec. 2025,ISSN: 0360-0300.DOI: 10.
1145/3773080
[12] N. Jain et al.,Baseline defenses for adversarial attacks against aligned
language models, 2023. arXiv: 2309.00614[cs.LG].
[13] M. Mozes, X. He, B. Kleinberg, and L. D. Griffin,Use of LLMs
for illicit purposes: Threats, prevention measures, and vulnerabilities,
2023. arXiv: 2308.12833[cs.CL].
[14] A. Hu, J. Gu, F. Pinto, K. Kamnitsas, and P. Torr,As firm as their
foundations: Can open-sourced foundation models be used to create
adversarial examples for downstream tasks?2024. arXiv: 2403.12693
[cs.CV].
[15] C. Chen and K. Shu,Can LLM-generated misinformation be detected?
2023. arXiv: 2309.13788[cs.CL].
[16] A. Zou, Z. Wang, N. Carlini, M. Nasr, J. Z. Kolter, and M. Fredrikson,
Universal and transferable adversarial attacks on aligned language
models, 2023. arXiv: 2307.15043[cs.CL].
[17] V . C. Hu et al., “Guide to attribute based access control (ABAC)
definition and considerations,” Tech. Rep. NIST Special Publication
800-162, 2013, pp. 1–54.DOI: 10.6028/NIST.SP.800-162
[18] M. Abadi et al., “Deep learning with differential privacy,” inPro-
ceedings of the 2016 ACM SIGSAC Conference on Computer and
Communications Security (CCS ’16), Vienna, Austria: Association for
Computing Machinery, 2016, 308–318,ISBN: 9781450341394.DOI:
10.1145/2976749.2978318

[19] G. Wenzek et al., “CCNet: Extracting high quality monolingual
datasets from web crawl data,” inProceedings of the Twelfth Language
Resources and Evaluation Conference, European Language Resources
Association, May 2020, pp. 4003–4012,ISBN: 979-10-95546-34-4.
[Online]. Available: https://aclanthology.org/2020.lrec-1.494/
[20] T. Miyato, A. M. Dai, and I. Goodfellow, “Adversarial training methods
for semi-supervised text classification,” inInternational Conference on
Learning Representations, 2017.
[21] S. Gehman, S. Gururangan, M. Sap, Y . Choi, and N. A. Smith, “Re-
alToxicityPrompts: Evaluating neural toxic degeneration in language
models,” inFindings of the Association for Computational Linguistics:
EMNLP 2020, Online: Association for Computational Linguistics, Nov.
2020, pp. 3356–3369.DOI: 10.18653/v1/2020.findings-emnlp.301
[22] P. Liang et al., “Holistic evaluation of language models,”Transactions
on Machine Learning Research, 2023,ISSN: 2835-8856. [Online].
Available: https://openreview.net/forum?id=iO4LZibEqW
[23] Y . Zhang, L. Ding, L. Zhang, and D. Tao, “Intention analysis makes
LLMs a good jailbreak defender,” inProceedings of the 31st Inter-
national Conference on Computational Linguistics, Abu Dhabi, UAE:
Association for Computational Linguistics, Jan. 2025, pp. 2947–2968.
[Online]. Available: https://aclanthology.org/2025.coling-main.199/
[24] D. Schultz, A. Bili ´c, D. Jurin ˇci´c, D. Grano ˇsa, and L. Kamber,
“Deepseek-r1 vs. OpenAI-o1: The ultimate security showdown,”
Splx.ai, Tech. Rep., 2025. [Online]. Available: https://splx.ai/blog/
deepseek-r1-vs-openai-o1-the-ultimate-security-showdown
[25] WIRED,Deepseek’s safety guardrails failed every test researchers
threw at its ai chatbot, 2025. Accessed: Mar. 28, 2025. [Online].
Available: https : / / www. wired . com / story / deepseeks - ai - jailbreak -
prompt-injection-attacks/
[26] M. Mazeika et al., “HarmBench: A standardized evaluation framework
for automated red teaming and robust refusal,” inProceedings of the
41st International Conference on Machine Learning, ser. ICML’24,
Vienna, Austria: JMLR.org, 2024.
[27] J. Brokman et al., “Insights and current gaps in open-source LLM
vulnerability scanners: A comparative analysis,” in47th International
Conference on Software Engineering (ICSE 2025), in 3rd International
Workshop on Responsible AI Engineering, Ottawa, Ontario, Canada,
2025.
[28] K. Zhu, Q. Zhao, H. Chen, J. Wang, and X. Xie, “Promptbench: A
unified library for evaluation of large language models,”Journal of
Machine Learning Research, vol. 25, no. 254, pp. 1–22, 2024.
[29] N. Carlini et al., “Extracting training data from diffusion models,”
in32nd USENIX Security Symposium (USENIX Security 23), 2023,
pp. 5253–5270.
[30] X. Guo, F. Yu, H. Zhang, L. Qin, and B. Hu, “COLD-attack: Jailbreak-
ing LLMs with stealthiness and controllability,” inProceedings of the
41st International Conference on Machine Learning, ser. ICML’24,
Vienna, Austria: JMLR.org, 2024.
[31] H. Jin et al.,JailbreakZoo: Survey, landscapes, and horizons in
jailbreaking large language and vision-language models, 2024. arXiv:
2407.01599[cs.CL].
[32] H. Li et al.,Privacy in large language models: Attacks, defenses and
future directions, 2024. arXiv: 2310.10383[cs.CL].
[33] A. Zhou, B. Li, and H. Wang, “Robust prompt optimization for
defending language models against jailbreaking attacks,” inAdvances
in Neural Information Processing Systems, A. Globerson et al., Eds.,
vol. 37, Curran Associates, Inc., 2024, pp. 40 184–40 211.[34] L. Derczynski, E. Galinkin, J. Martin, S. Majumdar, and N. Inie,garak:
A framework for security probing large language models, 2024. arXiv:
2406.11036[cs.CL].
[35] N. Papernot, “Machine learning at scale with differential privacy in
TensorFlow,” in2019 USENIX Conference on Privacy Engineering
Practice and Respect (PEPR 19), Santa Clara, CA: USENIX Associ-
ation, Aug. 2019. [Online]. Available: https://www.usenix.org/node/
238163
[36] M. Feffer, A. Sinha, W. H. Deng, Z. C. Lipton, and H. Heidari,
“Red-teaming for generative ai: Silver bullet or security theater?” In
Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society,
vol. 7, 2024, pp. 421–437.DOI: 10.1609/aies.v7i1.31647
[37] Y . Dong et al., “Safeguarding large language models: A survey,”
Artificial intelligence review, vol. 58, no. 12, p. 382, 2025.DOI: 10.
1007/s10462-025-11389-2
[38] R. Fang, R. Bindu, A. Gupta, Q. Zhan, and D. Kang,LLM agents can
autonomously hack websites, 2024. arXiv: 2402.06664[cs.CR].
[39] A. Mudaliar,ChatGPT leaks sensitive user data, OpenAI suspects hack,
Spiceworks, 2024. Accessed: Mar. 26, 2025. [Online]. Available: https:
//www.spiceworks.com/tech/artificial-intelligence/news/chatgpt-leaks-
sensitive-user-data-openai-suspects-hack/
[40] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing
adversarial examples,” inProceedings of the 3rd International Confer-
ence on Learning Representations (ICLR), Computational & Biological
Learning Society, 2015, pp. 1–11.
[41] D. Jin, Z. Jin, J. T. Zhou, and P. Szolovits, “Is BERT really robust?
a strong baseline for natural language attack on text classification
and entailment,” inProceedings of the AAAI Conference on Artificial
Intelligence, vol. 34, 2020, pp. 8018–8025.DOI: 10.1609/aaai.v34i05.
6311
[42] E. Wallace, S. Feng, N. Kandpal, M. Gardner, and S. Singh, “Universal
adversarial triggers for attacking and analyzing NLP,” inProceedings
of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural Lan-
guage Processing (EMNLP-IJCNLP), Association for Computational
Linguistics, Nov. 2019, pp. 2153–2162.DOI: 10.18653/v1/D19-1221
[43] B. Vidgen and L. Derczynski, “Directions in abusive language training
data, a systematic review: Garbage in, garbage out,”PLOS one, vol. 15,
no. 12, 2020.DOI: 10.1371/journal.pone.0243300
[44] European Commission High-Level Expert Group on Artificial Intelli-
gence,Ethics guidelines for trustworthy AI, 2019. Accessed: Aug. 11,
2025. [Online]. Available: https://digital- strategy.ec.europa.eu/en/
library/ethics-guidelines-trustworthy-ai
[45] A. Jobin, M. Ienca, and E. Vayena, “The global landscape of ai ethics
guidelines,”Nature machine intelligence, vol. 1, no. 9, pp. 389–399,
2019.DOI: 10.1038/s42256-019-0088-2
[46] D. W. Hubbard and R. Seiersen,How to Measure Anything in Cyber-
security Risk, 2nd. Wiley, 2023,ISBN: 978-1394163795.DOI: 10.1002/
9781119892335
[47] F. Perez and I. Ribeiro,Ignore previous prompt: Attack techniques for
language models, 2022. arXiv: 2211.09527[cs.CL].
[48] IBM Security and Ponemon Institute, “Cost of a data breach report
2024,” IBM Corporation, Tech. Rep., 2024. [Online]. Available: https:
//www.ibm.com/reports/data-breach
[49] R. Moulton,DSU: Quantifying return on controls in LLM cybersecu-
rity, 2025. Accessed: May 8, 2025. [Online]. Available: https://github.
com/rhmoult/DSU

APPENDIXA
LOSSEXCEEDANCECURVES
Fig. 5. Loss Exceedance Curve for Atkgen
Fig. 6. Loss Exceedance Curve for Divergence

Fig. 7. Loss Exceedance Curve for Latentinjection
Fig. 8. Loss Exceedance Curve for PII

Fig. 9. Loss Exceedance Curve for Promptinject
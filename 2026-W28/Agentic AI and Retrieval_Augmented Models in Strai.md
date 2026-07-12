# Agentic AI and Retrieval-Augmented Models in Straight-Through Underwriting

**Authors**: Robert Richardson, Josh Meyers, Brian Hartman, David Sandberg

**Published**: 2026-07-08 18:43:34

**PDF URL**: [https://arxiv.org/pdf/2607.07858v1](https://arxiv.org/pdf/2607.07858v1)

## Abstract
Artificial intelligence (AI) is beginning to reshape actuarial practice, particularly in domains that require reasoning over unstructured documents, heterogeneous data sources, and regulated decision workflows. Actuaries now face a design space that ranges from traditional rule-based automation to large language models (LLMs), retrieval-augmented generation (RAG), and multi-agent ``agentic'' systems that plan, retrieve, call tools, and reflect. This paper examines how these emerging architectures can support actuarial priorities such as transparency, auditability, and human-in-the-loop governance, with a focus on straight-through decision processes. To make these ideas concrete, we develop and analyze an agentic AI framework for straight-through underwriting of small commercial Business Owner Policies (BOPs). We construct a synthetic but realistic experimental environment and compare three underwriting pipelines: (i) a single-LLM baseline, (ii) a naive RAG system, and (iii) a multi-agent ``Agentic RAG'' pipeline that combines targeted retrieval, third-party data checks, and explicit multi-step rule evaluation. The agentic system performs best overall, with the largest gains in multi-step and missing-information scenarios, where structured retrieval and reflection help the model avoid unsupported straight-through decisions.

## Full Text


<!-- PDF content starts -->

Agentic AI and Retrieval-Augmented Models in
Straight-Through Underwriting
Robert Richardson, Josh Meyers, Brian Hartman, and David Sandberg
Brigham Young University
Abstract
Artificial intelligence (AI) is beginning to reshape actuarial practice, particu-
larly in domains that require reasoning over unstructured documents, heteroge-
neous data sources, and regulated decision workflows. Actuaries now face a design
space that ranges from traditional rule-based automation to large language models
(LLMs), retrieval-augmented generation (RAG), and multi-agent “agentic” systems
that plan, retrieve, call tools, and reflect. This paper examines how these emerg-
ing architectures can support actuarial priorities such as transparency, auditabil-
ity, and human-in-the-loop governance, with a focus on straight-through decision
processes. To make these ideas concrete, we develop and analyze an agentic AI
framework for straight-through underwriting of small commercial Business Owner
Policies (BOPs). We construct a synthetic but realistic experimental environment
and compare three underwriting pipelines: (i) a single-LLM baseline, (ii) a naive
RAG system, and (iii) a multi-agent “Agentic RAG” pipeline that combines tar-
geted retrieval, third-party data checks, and explicit multi-step rule evaluation.
The agentic system performs best overall, with the largest gains in multi-step and
missing-information scenarios, where structured retrieval and reflection help the
model avoid unsupported straight-through decisions.
Keywords:Retrieval-Augmented Generation (RAG); Human-AI Interactions; Syn-
thetic Data Generation; Insurance Automation;
1 Introduction
Artificial intelligence (AI) has undergone rapid evolution in recent years, moving from
rule-based automation toward models with increased adaptability and autonomy. Tra-
ditional automation systems execute predefined workflows, often lacking flexibility when
confronted with unstructured or unforeseen data. In contrast, large language models
(LLMs) such as those powered by transformer architectures have enabled richer capabil-
ities: they can process unstructured natural language inputs, generate fluent text, and
perform a variety of downstream tasks.
The distinction between rule-based automation and LLM-based systems should not
be interpreted too sharply. In many applications, explicit rules, statistical models, and
LLMs often play complementary roles. A rule engine is well suited to apply known
decision logic once the relevant facts are available. An LLM, by contrast, may be useful
when those facts are embedded in messy business descriptions, underwriting notes, or
other unstructured text. In some cases, LLMs may also help surface recurring patterns
1
arXiv:2607.07858v1  [cs.AI]  8 Jul 2026

or exceptions that can later be reviewed by domain experts and translated into explicit
underwriting rules. The practical question is therefore not whether rules or LLMs are
better, but how each can be used in the parts of the workflow where it is most reliable.
One of the pivotal developments in this space is the technique known as retrieval-
augmented generation (RAG). In the foundational work by Lewis et al. (2020), RAG
models are defined as combining a parametric memory with a non-parametric retrieval
mechanism. The authors show that such a hybrid architecture improves performance
on knowledge-intensive tasks by enabling the system to access external factual content
rather than relying solely on internal model parameters. Subsequent surveys and reviews
reinforce that RAG architectures help mitigate phenomena like “hallucination” (where
the model asserts facts not grounded in the external world) and improve factuality and
traceability (Gupta et al., 2024).
While RAG enhances knowledge access, it remains a relatively narrow paradigm:
retrieval plus generation. The next frontier is what many authors term “agentic AI,”
which are systems that behave more like autonomous agents, capable of setting sub-goals,
invoking external tools, coordinating workflows, and even collaborating across multiple
agents. Throughout this discussion, “agentic” does not imply sentience or true autonomy,
but rather structured orchestration of statistical models. Recent taxonomy distinguishes
multi-agent collaboration, task decomposition, memory, and coordinated autonomy as
key features of agentic architectures (Botti, 2025). In this view, agentic systems go
beyond simply responding to prompts. They exhibit what the literature calls “reasoning”
about how to achieve higher-level goals—again, a statistical planning heuristic rather
than genuine human reasoning. Importantly, throughout this paper, we use terms such
as “reasoning” or “thinking” in the sense adopted by the AI research community. These
systems do not think or reason in the human cognitive sense; rather, they estimate outputs
conditioned on learned statistical structure. Their effectiveness remains dependent on
careful human oversight and domain expertise.
In the domain of actuarial science where decisions often involve uncertain data, regu-
latory constraints, interpretability, and risk modelling, there is an opportunity to deploy
these agentic frameworks. Traditional actuarial tasks such as underwriting, reserving,
claims processing, and risk monitoring are increasingly data-rich and complex. By in-
tegrating retrieval, LLM-based inferential heuristics, structured statistical models, and
tool orchestration into an agentic pipeline, one can imagine underwriting systems that not
only process applications but generate structured, auditable steps in support of actuarial
decision-making.
The motivation for using an LLM in this setting is not that it replaces traditional
underwriting rules. Many underwriting decisions still depend on clear thresholds or ex-
clusions that can be applied directly once the relevant facts are known. The challenge is
that those facts are often not presented cleanly. They may appear in a business narrative,
be implied by several fields, or be missing from the application but available in another
source. The LLM component is used to help map this unstructured information to the
relevant guidebook concepts, summarize missing or ambiguous facts, and produce a draft
rationale. The RAG and agentic components then place structure around that language
capability by grounding the output in retrieved guidebook text, requiring structured de-
cisions, and routing uncertain cases to human review.
This paper presents an exploration of such an agentic AI framework applied to
straight-through underwriting in a business owner policy (BOP) insurance setting. Af-
ter defining the conceptual distinctions between automation, LLMs, retrieval-augmented
2

generation, and agentic AI, we review agentic use cases in actuarial science, discuss syn-
thetic/generated data pipelines, describe our specific framework, and present empirical
results. We conclude with implications, limitations, and directions for future work.
2 Agentic AI Concepts
Artificial intelligence has evolved through distinct stages of capability, from rule-based
automation to generative and ultimately agentic systems. Each stage represents an ex-
pansion of what constitutes “intelligence” in practical systems, from executing predefined
logic to performing parameter-driven reasoning heuristics with goals, tools, and memory.
These advances remain computational and statistical in nature; human judgment remains
essential for interpreting outputs responsibly.
2.1 Large Language Models (LLMs)
Early automation systems followed fixed decision rules. In business processes such as
underwriting or claims handling, these systems relied on deterministic logic designed to
replicate human procedures, such as if-then trees, linear scorecards, and expert systems
(Russell and Norvig, 2016). They excelled at efficiency and repeatability but failed when
encountering data or scenarios outside their programmed domain.
Machine learning extended automation by allowing systems to learn parameters from
data rather than explicit instructions. Supervised models such as logistic regression and
gradient-boosted trees offered statistical inference but not genuine reasoning or contextual
awareness.
Large Language Models represent a significant leap from classical automation. Based
on transformer architectures (Vaswani et al., 2017), they learn latent structure in lan-
guage, enabling flexible generation, summarization, and what the AI literature refers to
as “reasoning.” These are not acts of cognition but probabilistic pattern extrapolations.
LLMs remain parametric, meaning all information is encoded in their weights, making
them prone to hallucination (Ji et al., 2023a).
In practical terms, naive LLM deployments behave as sophisticated text generators,
capable of producing coherent narratives but lacking direct access to verified facts, struc-
tured databases, or organizational knowledge. Their value is realized only when paired
with rigorous human review and well-designed workflows.
2.2 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) addresses key limitations of purely parametric
language models by coupling an LLM with a non-parametric memory (Lewis et al., 2020).
In a RAG pipeline, the model:
1. Encodes a user query into a dense vector representation.
2. Searches a vector database of domain documents for semantically similar content.
3. Injects the retrieved passages into the LLM prompt.
4. Generates an output grounded in this external context.
3

This architecture enhances factual accuracy and interpretability by ensuring each re-
sponse can be traced to specific supporting documents. For regulated domains such as
finance or healthcare, this grounding is critical for compliance, auditability, and explain-
ability (Gupta et al., 2024).
Limitations.Despite its strengths, RAG systems are not without drawbacks. Per-
formance depends heavily on the quality of document retrieval—errors in search or em-
bedding similarity can cause the model to operate over irrelevant or outdated passages
(Asai et al., 2024; Izacard and Grave, 2022). Moreover, LLMs may still hallucinate or
misattribute facts even when provided with relevant context (Ji et al., 2023b). RAG
pipelines are also reactive: they answer a single query without persistent memory, plan-
ning, or reflection, limiting their ability to refine or validate their inferential process over
multiple steps. In high-stakes applications such as actuarial or financial modeling, these
shortcomings must be managed through human judgment.
Recent Enhancements.Several approaches aim to mitigate these issues. Self-RAG
architectures introduce self-reflective verification, prompting models to critique and re-
rank retrieved passages before generation (Asai et al., 2024). Dynamic retrieval and query
rewriting methods update search queries iteratively to reduce context drift (Trivedi et al.,
2023). Memory-augmented RAG systems further maintain persistent state, allowing
multiple rounds of retrieval and reasoning (Shuster et al., 2022). These improvements
begin to blur the boundary between retrieval-based and agentic reasoning systems.
2.3 Agentic AI
Agentic AI extends beyond retrieval to orchestration. According to recent frameworks
from Microsoft’sAutoGenproject and open research in multi-agent reasoning (Li et al.,
2023; Botti, 2025), an agentic system is characterized by autonomy, tool use, memory,
and reflective reasoning. Instead of a single prompt–response cycle, it maintains a stateful
interaction loop:
1. The agent interprets the user goal (statistically, by estimating intent from input
text).
2. It plans sub-tasks and invokes external tools (retrievers, calculators, APIs, models).
3. It reflects on intermediate results, adapting subsequent actions.
4. It terminates when the goal is achieved or uncertainty thresholds are met.
In this paper, we use the term “agentic” in a narrow and practical sense. The system
is not autonomous in the human sense, and it does not reason in the way an underwriter
reasons. Instead, it organizes a set of defined workflow steps: retrieving guidebook ev-
idence, routing the application into accept, reject, or refer pathways, checking whether
decisive facts are missing, using simulated third-party data when available, applying
multi-step underwriting rules, and reviewing the final decision. This narrower definition
is important because the value of the system comes from making the decision process
more structured and inspectable, not from giving the model independent authority.
This framework embeds LLM-based inference within a control architecture, often
implemented with orchestration libraries such as LangChain or LangGraph, where the
model not only generates text but also decideswhento retrieve, compute, or delegate.
4

Because this behavior is algorithmic rather than cognitive, effective use depends critically
on smart human design, validation, and supervision.
Limitations.While promising, multi-agent systems introduce new points of failure.
Communication errors, inconsistent context passing, and cascading prompt noise can lead
to instability or non-deterministic behavior across agents (Wang et al., 2024). Each ad-
ditional agent adds complexity to alignment and safety verification, making the system
more difficult to audit or reproduce (Park et al., 2023). Furthermore, multi-agent or-
chestration often incurs higher computational cost and latency, and reflective loops may
amplify biases if not properly constrained.
Emerging Mitigations.Recent work addresses these vulnerabilities through im-
proved coordination and verification mechanisms. For instance, delegation-aware archi-
tectures constrain inter-agent communication to predefined schemas (Liu et al., 2024).
Critic–executor frameworks employ specialized “critic” agents that evaluate and revise
outputs from task agents, improving reliability and stability (Madaan et al., 2024). Other
research focuses on memory management and role consistency, ensuring agents retain co-
herent goals across iterative interactions (Chen et al., 2024). Together, these refinements
aim to balance autonomy with control—an essential requirement in professional domains
such as actuarial science, where traceability and reproducibility are mandatory.
3 AI in Actuarial Science
The actuarial profession is increasingly incorporating advanced artificial intelligence (AI)
capabilities. These approaches are reshaping how actuaries approach reserving, pricing,
and risk monitoring by enabling parameter-driven “reasoning” over unstructured data,
dynamic workflows, and semi-autonomous decision pipelines. Their advantages, however,
materialize only when skilled actuaries design, monitor, and interpret these systems.
Several studies and professional reports highlight how actuaries and insurers are al-
ready applying AI methods. A foundational literature review by the Society of Actuaries
(SOA) emphasized that AI and machine learning will “transform actuarial work through
the creation of new technologies, products, and services” (Yeo et al., 2019). More recently,
the SOA’sPrimer on Generative AI for Actuariesrecommends that generative systems
be deployed as augmentation tools for actuarial judgment rather than full automation
(Carlin and Mathys, 2024). Industry analysis from Oliver Wyman further reports that
AI in insurance is already enhancing accuracy, fraud detection, and customer service,
while introducing new data-governance challenges (Balona, 2023).
Recent work titledAdvanced Applications of Generative AI in Actuarial Sciencepro-
vides concrete demonstrations of RAG and multi-agent systems performing document
comparison, vision-assisted claim classification, and automated reporting (Hatzesberger
and Nonneman, 2025). The Casualty Actuarial Society has called for research on lever-
aging large language models (LLMs) to process unstructured claims data for actuarial
use (Richman, 2024). An AI-based automated pricing and underwriting model has been
demonstrated for the general insurance sector, integrating machine learning with rule-
based decision processes (Mahohoho et al., 2024). Case studies highlight LLMs deriving
features from text, RAG automating document retrieval, and agentic multi-step sys-
tems orchestrating data ingestion and statistical inference (Hatzesberger and Nonneman,
2025). The application of AI in actuarial science is new and underdeveloped, but active
research is underway to determine how these tools can best support rather than replace
5

professional actuarial expertise.
3.1 Use Cases of Agentic AI in Actuarial Practice
Agentic AI should be viewed not as a substitute for actuarial expertise but as an amplifier
of it. Its strength lies in coordinating retrieval, tool use, and reflection, which are activi-
ties that are procedural yet time-consuming, while preserving professional judgment and
accountability. The following four scenarios are intended as illustrative use cases rather
than predictions of immediate production capability. In practice, the feasibility of each
workflow depends on data quality, system integration, governance constraints, and the
degree to which legacy platforms can expose reliable inputs to the agentic system.
Scenario 1: Pricing Cycle Management and Regulatory Filing Support
Context.Actuaries performing quarterly rate reviews must reconcile exposure and loss
data, run rate indications, prepare exhibits, and document the rationale for regulatory
filings. These cycles blend data engineering, modeling, and narrative justification, guided
by theCAS Statement of Principles on Ratemaking(Casualty Actuarial Society, 2019).
Agentic augmentation.
•Workflow-assistance toolscould help extract data from internal warehouses,
populate portions of indication templates, and identify reconciliation issues in earned
exposures, premiums, and losses, provided that source systems, data dictionaries,
and schema mappings have been validated.
•Agentic review modulescould provide preliminary checks of pipeline outputs,
examining assumptions for consistency and reasonableness across exhibits (e.g.,
trend selections, development patterns, inflation factors), while leaving actuarial
selections and final judgment to the practitioner.
•Reflection componentscould draft explanatory exhibits or generate diagnostic
code (e.g., SQL queries) to help investigate period-over-period changes, such as
shifts in written premium trends. These outputs would require review, especially
when source databases are complex, poorly documented, or maintained across mul-
tiple legacy systems.
•Retrieval systemscould compare assumptions and methodologies against prior
filings, competitor submissions, and state bulletins to help identify possible devia-
tions or emerging regulatory considerations (Balona, 2023).
•Draft summarieswith data provenance, assumption documentation, and cross-
references to supporting materials could be assembled for actuarial review and sign-
off.
Benefits.When the underlying data connections are reliable, the combination of
automated data preparation and agentic review may improve consistency, reduce some
manual coordination, and support more complete documentation. Actuaries retain full
authority over rate indications and filing decisions, and the system should be treated as
a documentation and review aid rather than an independent filing engine.
Risks.Overreliance on automated diagnostics may obscure data quality issues, regu-
latory nuance, or business context that actuaries are better positioned to interpret. These
6

risks are especially important in insurance environments with legacy databases, incon-
sistent coding practices, complex product hierarchies, or multiple downstream reporting
systems. Agentic modules may also surface false positives or misleading comparisons,
requiring careful human validation to avoid unjustified adjustments or misaligned filings.
Scenario 2: Reserve Diagnostics and Portfolio Monitoring
Context.Reserving actuaries analyze loss development across lines and methods, ex-
plain movements, and communicate results to finance, claims, and regulatory stakehold-
ers. Classical methods such as the chain-ladder, Bornhuetter–Ferguson, and stochastic
reserving approaches (Mack, 1993; England and Verrall, 2002; W¨ uthrich and Merz, 2008)
remain foundational, yet diagnostic synthesis and documentation remain labor-intensive.
Agentic augmentation.
•Automated pipeline checkscould help schedule data refreshes, execute pre-
specified reserving techniques (e.g., chain-ladder, Bornhuetter–Ferguson, GLM- or
Bayesian-based stochastic methods), and compare outputs across valuation periods,
subject to existing data controls and actuarial review.
•Variance analysis agentscould draft preliminary movement explanations by
analyzing changes in loss triangles and identifying possible drivers such as large-loss
emergence, development pattern shifts, or operational changes (e.g., “Increase in
paid severity for AY 2021→+3.2 points IBNR”).
•RAG-based claim linkingcould help search individual claim notes and transac-
tion histories to identify claims associated with unusual cell-level movements, while
recognizing that claim narratives may be incomplete, inconsistent, or affected by
operational coding practices.
•Assumption monitoring modulescould compare current selections (e.g., tail
factors, a priori loss ratios, case reserve adequacy assumptions) against prior memos
and methodological documentation to flag material deviations beyond established
tolerances (Hatzesberger and Nonneman, 2025).
•Governance logscould store model outputs, diagnostic notes, assumption lineage,
and human overrides to support reproducibility, audit trails, and internal model
governance.
Benefits.The actuary could receive a structured diagnostic aid that highlights areas
requiring expert judgment rather than a fully automated reserving conclusion. Inter-
pretability, communication consistency, and governance discipline may improve while
methodological choices remain fully under actuarial control.
Risks.Automatically generated explanations may overstate confidence or misat-
tribute drivers of development, especially in sparse or volatile triangles. Agentic systems
may also focus on statistically detectable changes while missing operational insights ap-
parent to experienced reserving actuaries, reinforcing the need for human interpretation.
For this reason, generated movement explanations should be treated as hypotheses to
investigate, not as final reserve diagnostics.
7

Scenario 3: Claims Triage and Straight-Through Processing Support
Context.P&C insurers pursue straight-through processing (STP) for simple claims
while maintaining rigorous oversight of complex or high-risk ones. Actuaries contribute
by designing triage and severity models that balance efficiency with fairness (Richman,
2024).
Agentic augmentation.
•The agent could retrieve policy terms, prior claims, and adjuster notes to contex-
tualize new submissions, subject to data availability and access controls.
•Reasoning modules could apply triage rules and invoke fraud or severity scor-
ing models, while preserving the distinction between model indicators and claim-
handling decisions.
•Reflection steps could compare outputs with historical experience and escalate un-
certain cases with documented rationale.
•Transparent logs could show which rules, models, and data sources informed each
recommendation.
Benefits.The framework could support faster handling of low-complexity claims
while improving documentation and auditability. Actuaries retain control of model cali-
bration and validation but gain enhanced visibility into operational performance.
Risks.Automated triage may overweight efficiency at the expense of fairness if not
carefully monitored; actuaries are typically better at identifying equity considerations,
claim-specific nuance, and unintended model bias. Escalation logic may also be brit-
tle, requiring oversight to avoid inappropriate straight-through approvals or rejections.
Because claims decisions can have direct policyholder impact, any straight-through work-
flow should include monitoring for error patterns, disparate impact, and inappropriate
automation of borderline cases.
Scenario 4: Catastrophe Exposure and Accumulation Management
Context.Catastrophe (CAT) modeling integrates hazard, exposure, vulnerability, and
financial modules to estimate loss distributions (Grossi and Kunreuther, 2005). Actuaries
oversee validation, communication of PML/TVaR, and governance per industry best
practices (Roberts et al., 2023).
Agentic augmentation.
•A planning agent could help coordinate exposure updates and vendor model runs
when the required data feeds and model interfaces are sufficiently standardized.
•Retrieval components could pull prior exposure reports, treaty terms, and hazard
maps to support review of model inputs and assumptions.
•Reflection layers could flag unusual exposure growth or potential concentration
issues for actuarial review, producing annotated dashboards where appropriate.
•When model results exceed pre-specified thresholds, the system could draft alerts
with location-level summaries and comparable-event references.
8

Benefits.Actuaries may spend less time assembling routine inputs and more time in-
terpreting results, testing assumptions, and communicating uncertainty. Agentic orches-
tration could improve governance by preserving clearer records of inputs, model versions,
assumptions, and review steps.
Risks.Excessive reliance on automated workflows may shift focus toward process
compliance rather than the substantive quality of scenario interpretation. Actuaries must
ensure that automated anomaly detection does not overshadow expert judgment regard-
ing exposure growth, data limitations, and model uncertainty. This is especially impor-
tant because CAT results are sensitive to exposure quality, vendor model assumptions,
and the treatment of low-probability high-severity events.
3.1.1 Additional Opportunities for Augmentation
Beyond these four cases, agentic orchestration may also support selected components of:
•Reinsurance analytics:retrieving treaty wordings, summarizing key provisions,
and supporting scenario comparisons.
•Operational data quality:helping coordinate reconciliations between policy,
billing, and claims systems, while flagging exceptions for human review.
•Regulatory and financial reporting:drafting disclosure narratives that can be
checked against prior filings and current assumptions.
•Portfolio steering:monitoring loss-ratio drift and identifying emerging exposure
clusters for further analysis.
•Agent / Broker Monitoring:reviewing changes in mix of business from agents
or brokers and flagging patterns that may warrant underwriting review.
Across these use cases, the most realistic near-term role for agentic AI is to reduce
repetitive coordination work, improve documentation, and surface issues for professional
review. Its value depends on careful implementation, reliable data infrastructure, and
actuarial oversight consistent with the profession’s standards of accuracy, fairness, and
accountability.
3.2 Ethical Considerations
Ethical reflection is central to the responsible adoption of artificial intelligence in actuarial
work. The actuarial profession has long emphasized professionalism, accountability, and
the public interest, while AI ethics frameworks stress fairness, transparency, and human
oversight. As actuaries deploy large language models, retrieval systems, and agentic
frameworks, both traditions must converge to ensure responsible practice.
This governance framing is consistent with current practitioner concerns in actuarial
and insurance settings. Guidance from actuarial and insurance organizations generally
emphasizes that AI tools should support, rather than replace, professional judgment.
For this framework, that means the system should preserve the information needed for
review: the data used, the guidebook passages retrieved, the model output, the reason for
any escalation, and any human override. The goal is not only to improve accuracy, but
also to make the workflow easier to inspect, challenge, and govern. Because regulatory
expectations for AI continue to evolve, this paper should be read as a proof of concept for
a governance-aware workflow rather than as a deployment-ready compliance framework.
9

3.2.1 Alignment with Actuarial Professional Standards
Actuarial ethics, as codified in theCAS Code of Professional Conductand theSOA Code
of Professional Conduct, require actuaries to act with integrity, competence, and due care,
using appropriate models and methods. Precept 3 requires that any model used in pro-
fessional work be fit for purpose and adequately communicated. AI and agentic systems
introduce new responsibilities: validating algorithmic components, disclosing limitations,
and ensuring that automated reasoning supports and not supplants human judgment.
These align with model-risk-management expectations under the NAIC’sModel Risk
Management Handbook(2023) and emerging AI governance guidance.
From a broader AI ethics perspective, five recurring principles appear across ma-
jor frameworks such as the OECD AI Principles and the NAICPrinciples on Artificial
Intelligence(2020):(1)fairness and non-discrimination,(2)accountability and human
oversight,(3)transparency and explainability,(4)privacy and security, and(5)robust-
ness and reliability. For actuarial applications, these translate into clear documentation
of data sources and assumptions, interpretability of decision logic, and audit trails that
regulators and policyholders can trust. Scholarly reviews note that responsible AI sys-
tems require proportional human involvement, outcome monitoring, and ongoing impact
assessment (Jobin et al., 2019; Floridi et al., 2018). Such principles are consistent with
actuarial professionalism’s focus on verifiable methods and communication of uncertainty.
3.2.2 Practical Integration in Actuarial Workflows
In practice, ethical AI governance for actuarial systems should include:
•Human-in-the-loop design:ensuring actuaries retain authority for acceptance,
override, and sign-off decisions.
•Model transparency:maintaining documentation that explains algorithms, data
provenance, and limitations.
•Bias and fairness testing:routinely checking for disparate impact across demo-
graphic or geographic segments.
•Auditability and traceability:storing reasoning traces, retrieval evidence, and
model outputs for regulatory review.
•Continuous validation:treating AI components as models subject to annual
review under model-risk frameworks.
•Implementation controls:documenting data interfaces, system dependencies,
known data-quality limitations, and circumstances in which automated outputs
should not be relied upon without additional review.
Ultimately, the ethical integration of AI within actuarial science should rest on a
shared foundation: AI ethics contributes principles of responsible autonomy and account-
ability, while actuarial professionalism provides the structures of peer review, disclosure,
and the public-interest mandate. An agentic system that enhances transparency, re-
duces bias, and maintains human oversight can thus be viewed as a natural extension of
actuarial ethics rather than a disruption of it.
10

4 Synthetic Data Generation and the Business Owner
Policy (BOP) Use Case
4.1 Use Case Motivation and Context
The motivating problem for this work is the automation of straight-through underwriting
(STU) for small commercial Business Owner Policies (BOPs). In conventional workflows,
underwriting requires actuaries, analysts, and underwriters to interpret lengthy policy
manuals, evaluate diverse data sources, and decide whether an application should be
accepted, rejected, or referred for human review. The process is information-intensive,
and small differences in how facts are interpreted can change the final decision.
The envisioned agentic pipeline uses:
•A large language model (LLM) to read and summarize application materials.
•A retrieval component to ground decisions in the synthetic underwriting guidebook.
•A structured orchestration layer that checks completeness, retrieves third-party
information when needed, decomposes multi-step rules, and escalates cases when
the decisive information is unavailable.
Testing such a pipeline requires data that reflect business diversity, missing-information
patterns, edge cases, and underwriting rules that may depend on information distributed
across several parts of an application. Real commercial underwriting data cannot gen-
erally be shared for regulatory, privacy, and competitive reasons, and it rarely includes
explicit ground-truth rationales. At the same time, a synthetic BOP dataset necessarily
falls short of a production underwriting or pricing environment. Real BOP workflows
may involve hundreds of variables, company-specific rating plans, legacy data systems,
detailed coverage forms, and complex interactions among classification, exposure, payroll,
revenue, premises characteristics, loss history, and operational controls. Consequently,
our synthetic dataset should be interpreted as a controlled proof-of-concept environment
rather than a full representation of production BOP underwriting.
Within that limitation, the dataset was designed to preserve several important log-
ical and structural properties of real underwriting decisions. The applications include
complete cases, explicit rule violations, missing-information cases, and multi-step reason-
ing cases in which the correct decision depends on connecting information from different
parts of the application, the underwriting guide, and simulated third-party data. For
example, an application may report total square footage in one field, describe that only
a portion of the premises is used for a particular operation in the narrative, and require
comparison with a guidebook rule that applies only if that operation exceeds a speci-
fied square-footage threshold. Other cases require interpreting operating hours, employee
counts, revenue limits, or conditional exclusions from messy business descriptions. These
examples do not make the dataset fully realistic, but they create a transparent benchmark
for evaluating whether agentic retrieval and review steps improve decision accuracy and
explanation quality relative to simpler LLM and RAG baselines.
This dataset allows us to address several research challenges:
1.Incomplete information:Many applications omit details essential for risk eval-
uation.
11

2.Multiple decision pathways:A case can fail because of a direct appetite restric-
tion, a multi-step rule violation, or unresolvable ambiguity.
3.Need for consistent reasoning:Evaluation requires not only the correct outcome
but also a defensible explanation.
4.Integration of heterogeneous data:Underwriting decisions depend on struc-
tured fields, unstructured narratives, guidebook clauses, and third-party informa-
tion.
By including simulated third-party data, rule-based guidebook clauses, and multi-step
reasoning cases, the synthetic dataset enables end-to-end testing of the entire agentic un-
derwriting pipeline, from data ingestion and retrieval to decision justification and human
escalation.
4.2 Policy Guidebook Generation
We first generated a 143-page synthetic policy guidebook using a large language model.
The guidebook encodes general underwriting standards and business-specific rules for 127
business types. Each business type has its own appetite language, including eligibility
criteria, exclusion criteria, and thresholds involving property characteristics, operational
details, revenue mix, or business activities. In total there are 16 global requirements and
6 business-specific requirements for each business type. We recognize that the guidebook
does not reproduce any insurer’s internal underwriting manual, but it provides a con-
trolled and realistic basis for generating applications that should be accepted, rejected, or
referred. The full guidebook can be found athttps://github.com/drbob-richardson/
agentic-bop-underwriting/blob/main/data/BOP_Guidebook.pdf.
Example clause:
“For restaurants serving alcohol, coverage is excluded if more than 50% of rev-
enue is derived from alcohol sales, unless a liquor-liability rider is attached.”
This guidebook forms the primary retrieval corpus for the RAG and agentic compo-
nents. In the multi-step scenarios, the relevant clause is often not sufficient by itself. The
model must also extract or infer the numerical quantity or operational condition needed
to apply the rule.
4.3 Application and Third-Party Data Generation
For each of the 127 business types, we created five application scenarios, producing 635
applications in total. Each scenario represents a distinct truth condition:
1.Compliant case:The application contains the necessary information and meets
all relevant guidebook criteria (Accept).
2.Single-issue violation:The application contains one explicit breach of the guide-
book rules, such as a prohibited operation or a threshold violation stated directly
in the application (Reject).
12

3.Multi-step reasoning:The correct decision requires combining at least two facts
before applying the guidebook rule. The decisive issue may involve a derived square
footage, revenue share, activity mix, subcontractor count, or shared-premises con-
dition (AcceptorReject, depending on the derived result).
4.Missing information (recoverable):Key information is omitted from the ap-
plication but can be recovered from simulated third-party data. The model must
retrieve the missing information and then apply the guidebook rule.
5.Missing information (irrecoverable):The decisive information is absent from
both the application and the third-party data, so the proper outcome isRefer to
human review.
Each record combines structured elements such as business class, location, revenue,
employee count, property size, and prior losses with unstructured text such as narrative
operations descriptions, underwriting notes, and third-party business summaries. The
third-party data emulate external information sources that an agentic system could query
in practice, such as public business descriptions, marketing pages, property summaries,
or registry-style records.
Every rejection or referral scenario is paired with a consistent ground-truth rationale
to support evaluation of explanation quality. For instance:
True reason:Reject because the application reports 14,000 total square feet,
the operations note states that 50% is used for grocery sales, and the guide-
book permits grocery operations only below 5,000 square feet. The derived
grocery area is 7,000 square feet, exceeding the threshold.
Model rationale example:“The business should be declined because its
grocery operation occupies approximately 7,000 square feet, which exceeds
the 5,000-square-foot eligibility limit in the underwriting guide.”
Evaluation compares the model’s textual explanation to the true rationale using
cosine-similarity scoring, enabling quantitative assessment of reasoning alignment in ad-
dition to decision accuracy.
A human validation team manually reviewed the synthetic applications, guidebook ex-
cerpts, and expected outcomes. Reviewers verified that each record’s data, decision label,
and rationale were logically consistent and realistic. This step prevented circular vali-
dation, because the systems were not evaluated solely against unreviewed AI-generated
labels. The final dataset contains 127 applications in each of the five scenario categories.
After validation and relabeling, the final decision outcomes are:
•127 applications that should be accepted (20.0%).
•345 applications that should be rejected (54.3%).
•163 applications that should be referred to human review (25.7%).
The final dataset can be found athttps://github.com/drbob-richardson/agentic-bop-underwriting/
blob/main/BOP_Synthetic_Dataset.csv.
13

4.4 Examples
To illustrate the structure of the synthetic dataset, consider a small specialty grocery and
cafe risk. The guidebook includes the following simplified clauses:
“Specialty grocery operations are eligible only when the grocery portion oc-
cupies less than 5,000 square feet. Cafe operations are eligible if seating is
limited and alcohol sales are not offered. Shared premises with excluded
manufacturing operations require referral unless the occupancy separation is
documented.”
These clauses create several possible decision paths.
(1) Compliant Case.Application excerpt:“Maple Street Cafe operates a 2,800-
square-foot coffee and sandwich shop with no alcohol sales and no shared manufacturing
or warehouse operations.” All material facts are present and within appetite.Decision:
Accept.
(2) Single-Issue Violation.Application excerpt:“Maple Street Cafe offers evening
service and derives 60% of revenue from wine and beer sales.” The alcohol condition is
directly stated and violates the guidebook rule.Decision:Reject.
(3) Multi-Step Reasoning.Application excerpt:“Maple Street Market leases 12,000
square feet. The front half of the premises is used for specialty grocery sales, with the
remaining space used for office and storage.” The guidebook threshold applies to the
grocery portion, not the total premises. The model must compute that 50% of 12,000
square feet is 6,000 square feet, then compare that derived amount with the 5,000-square-
foot grocery limit.Decision:Reject.
(4) Missing Information (Recoverable).Application excerpt:“Maple Street Mar-
ket operates a neighborhood grocery and cafe.” The application does not state square
footage by operation. Third-party property data state that the business leases 8,000
square feet and a public description states that the grocery area occupies one-third of the
premises. The derived grocery area is approximately 2,667 square feet, which is below
the threshold.Decision:Acceptif all other guidebook conditions are satisfied.
(5) Missing Information (Irrecoverable).Application excerpt:“Maple Street Mar-
ket operates a neighborhood grocery and cafe in a mixed-use building.” Neither the ap-
plication nor third-party data disclose the grocery square footage or whether the premises
are separated from other occupancies. The decisive information is unavailable.Decision:
Refer to human review.
These examples show why the revised experiment is harder than simple rule lookup.
In the multi-step and missing-information cases, the model must determine what fact is
needed, locate or derive that fact, and then decide whether the guidebook rule can be
applied safely.
14

5 Agentic Pipeline and Experimental Results
5.1 Model Descriptions
The main purpose of the AI system is to determine whether a BOP application should
be accepted, rejected, or referred to human review. The decision is based on whether
the application is within the underwriting appetite described in the synthetic guidebook.
In the revised experiment, there is no logistic-regression or quantitative risk-score layer.
Instead, the third scenario category tests whether the model can apply multi-step un-
derwriting logic: extracting relevant facts, deriving any needed intermediate quantity,
retrieving the correct rule, and applying that rule consistently.
To put the agentic workflow into context, we evaluate three systems, from simplest to
most structured. Conceptually, the goal of the agentic pipeline is not to replace under-
writing judgment but to triage work. The AI should handle the easiest high-confidence
passes and fails, while routing nuanced or low-confidence applications to human under-
writers. Finding this balance requires careful calibration of prompts, retrieval rules, and
escalation thresholds against human decisions.
5.1.1 Single-LLM Baseline
Figure 1 shows the simplest underwriting workflow, in which a single large language model
performs end-to-end decision-making based on the application data and static policy-
guide context. We use the term “Determine Appetite” because it is the underwriting-
appropriate term for the task, but again emphasize that the LLM is a parameter-based
model and does not reason or think in the human sense.
Figure 1:Single-LLM Baseline.A single LLM determines the underwriting dispo-
sition directly from application text and static guidebook context. Decisions rely on a
fixed prompt and do not use dynamic retrieval, explicit completeness checks, third-party
lookup, or reflection.
In this configuration, the model assesses appetite directly from the provided text.
15

Although simple and fast, this baseline lacks a structured mechanism for resolving am-
biguous or missing-context cases. It may also miss multi-step violations when the decisive
fact must be derived rather than read directly.
5.1.2 Retrieval-Augmented Workflow (RAG)
Figure 2 extends the baseline by adding a retrieval-augmented generation layer that
supplements each application with relevant passages from the synthetic underwriting
guidebook.
Figure 2:RAG Workflow.An LLM receives both the application and retrieved guide-
book context. This enables evidence-grounded appetite classification, though the process
remains single-pass and lacks explicit diagnostic, completeness, or reflective feedback.
The RAG system improves factual grounding by retrieving domain-specific underwrit-
ing rules prior to model inference. However, retrieval alone does not guarantee that the
model will identify the decisive missing fact, combine facts correctly, or recognize that a
decision should be referred rather than forced.
5.1.3 Agentic Pipeline
Figure 3 illustrates the complete agentic straight-through underwriting pipeline, com-
posed of three cooperating agents orchestrated through a stateful workflow. Each agent
specializes in a distinct task: routing, clarification, and reflection.
The sequence proceeds as follows:
Agent 1: Routing and Appetite Check.Evaluates whether the application appears
to bein appetite,out of appetite, orunclear. Direct violations are rejected, clear
compliant cases are accepted, and uncertain cases are passed to Agent 2.
16

Figure 3:Agentic Multi-Agent Pipeline.A coordinated three-agent system with
targeted retrieval and reflection loops. Agent 1 performs initial appetite routing, Agent 2
diagnoses missing or ambiguous information and retrieves third-party data when avail-
able, and Agent 3 checks multi-step logic before returning the case for reevaluation.
Resolved cases are accepted or rejected; unresolved cases are referred for human review.
Agent 2: Contextual Clarification.Determines why uncertainty exists. If the miss-
ing element is a guidebook rule, the agent requests targeted retrieval. If the missing
element is a business fact, the agent searches the simulated third-party data. If the
decisive fact remains unavailable, the case is marked for human review.
Agent 3: Multi-Step Reflection.Checks whether the case requires derived reasoning,
such as allocating square footage, combining employee and subcontractor counts,
converting revenue percentages into activity-specific revenue, or linking a shared-
premises note to an exclusion. The agent synthesizes the relevant facts and returns
a structured rationale to Agent 1 for final evaluation.
This design creates an explicit pathway for deciding when the application contains
enough information to support a straight-through decision and when it should be referred.
The value of the agentic architecture is not simply that it retrieves more text, but that it
organizes the underwriting task into completeness, evidence, and rule-application checks.
5.2 Implementation Details
All three pipelines were implemented in Python 3.12 using the OpenAI API and the
LangChain ecosystem, includinglangchain,langchain-openai,langchain-community,
andlanggraph. No local model inference was used. We evaluated two foundation mod-
els:gpt-4o-mini, used as a smaller lower-cost model, andgpt-5.2, used as the more
capable model. The same pipeline code was used for both models, with only prompt cali-
bration adjusted across model families. Evaluation calls usedtemperature= 0 to reduce
17

randomness, while synthetic data generation used a higher temperature to encourage
diversity in the generated applications.
The underwriting guidebook was parsed into a FAISS vector index for retrieval. We
usedPyPDFLoaderto extract guidebook text and aRecursiveCharacterTextSplitter
with 1000-character chunks and 200-character overlap. The na¨ ıve RAG pipeline retrieved
the top guidebook chunks from a single query built from the business type and operations
description. The Agentic RAG pipeline allowed additional targeted retrieval when an
application was unclear, including queries tied to the business type, global exclusions,
and any specific missing or ambiguous issue identified during the intermediate review
step.
Prompt design was intentionally structured so that routing decisions could be audited.
Each model was instructed to return a short JSON object containing a decision label, a
rationale, and any missing or decisive underwriting facts. The single-LLM baseline used
one prompt containing the application and static guidebook context. The na¨ ıve RAG
model used one prompt containing the application and retrieved guidebook excerpts. The
Agentic RAG workflow used a LangGraph state machine with separate steps for appetite
review, missing-information diagnosis, targeted retrieval or third-party lookup, multi-step
rule evaluation, and final decision review. If a response could not be parsed reliably, the
system defaulted to human review rather than forcing an accept or reject decision.
To reduce label leakage, each application record was sanitized before evaluation.
The model-facing record excluded the scenario label, ground-truth decision, genera-
tion rationale, and internal validation fields. The pipelines therefore observed only
the application fields, operations narrative, relevant guidebook context, and simulated
third-party data. The released artifacts include the synthetic application dataset, per-
application evaluation results, pipeline code, and a data dictionary for the applica-
tion fields. The main implementation notebook is available athttps://github.com/
drbob-richardson/agentic-bop-underwriting/blob/main/Agentic_RAG_Underwriting.
ipynb.
5.3 Experimental Setup
We tested the framework on the synthetic dataset described in Section 4, covering 127
business types with five scenario categories each: compliant, single-issue violation, multi-
step reasoning, missing information recoverable, and missing information irrecoverable.
This produces a total of 635 applications.
Three configurations were evaluated:
1.LLM-Only:A baseline large language model making single-pass decisions without
dynamic retrieval.
2.Na¨ ıve RAG:A standard retrieval-augmented model that retrieves guidebook con-
text but does not use agentic orchestration.
3.Agentic RAG:A multi-agent framework with retrieval, completeness checking,
third-party data access, and reflection over multi-step logic.
Each model was asked to provide both a decision label and a textual justification.
These were compared to the human-validated ground-truth outcomes and reasons. Eval-
uation metrics include:
18

•Decision accuracy:the percentage of correctly classified outcomes.
•Per-scenario accuracy:accuracy within each of the five scenario categories.
•Rationale similarity:mean cosine similarity between the model-generated ex-
planation and the human-validated ground-truth reason, computed across all 635
applications for each pipeline.
•Latency:average time required to process an application.
To prevent label leakage, every record is stripped of its scenario tag, ground-truth
label, and generation rationale before any pipeline observes it. The pipelines see only
the application form, retrieved guidebook context where applicable, and simulated third-
party data where the workflow permits it. We report each pipeline under two base
models: a small, inexpensive model (gpt-4o-mini) and a substantially more capable
model (gpt-5.2).
5.4 Experimental Results
Table 1 reports overall decision accuracy. Under both base models the Agentic RAG
pipeline attains the highest accuracy. The gains are especially meaningful because the
revised dataset is designed to test not only direct rule recognition, but also multi-step
rule application and appropriate referral when decisive information is unavailable.
Table 1: Overall decision accuracy (%) against human-validated ground truth,n= 635.
Best per column in bold.
Pipelinegpt-4o-mini gpt-5.2
Single-LLM 73.4 77.6
Na¨ ıve RAG 71.5 76.9
Agentic RAG85.1 86.5
5.5 Per-Scenario Accuracy
Table 2 decomposes accuracy by scenario for thegpt-5.2configuration. The agentic
pipeline performs best on four of the five scenario categories and is only slightly behind
the single-LLM baseline on single-issue violations. The largest practical gains occur in
the multi-step and missing-information settings, where a single-pass model is more likely
to overlook a derived fact or force a decision when the evidence is incomplete.
The multi-step reasoning row is particularly important because it replaces the earlier
logistic-failure scenario. These cases are not failures of quantitative scoring; they are
failures or successes of document-grounded logic. The model must identify which facts
matter, combine them, and then apply the guidebook rule. The agentic pipeline improves
this category from 70.1% for the single-LLM baseline and 66.1% for na¨ ıve RAG to 78.0%.
The missing-information rows show a related pattern. When information is recover-
able, the agentic pipeline benefits from an explicit process for seeking third-party data.
When information is irrecoverable, the agentic pipeline is much more likely to recognize
that the application should be referred rather than forcing an unsupported accept or
reject.
19

Table 2: Per-scenario decision accuracy (%) undergpt-5.2. Best per row in bold.
Scenario Single-LLM Na¨ ıve RAG Agentic RAG
Compliant (accept) 96.9 95.3100.0
Single-issue violation84.383.5 81.9
Multi-step Reasoning 70.1 66.178.0
Missing info (recoverable) 80.3 85.888.2
Missing info (irrecoverable) 56.7 53.584.3
5.6 Rationale Similarity and Latency
Table 3 reports two secondary diagnostics undergpt-5.2: rationale similarity and la-
tency. Rationale similarity is an overall pipeline-level metric, not a scenario-specific
metric and not a metric restricted to applications referred for human review. For each of
the 635 applications, the model-generated rationale is compared with the corresponding
human-validated ground-truth rationale using embedding-based cosine similarity. The
table reports the mean of these 635 similarities for each pipeline. Higher values indicate
that the model’s explanation is semantically closer to the human-validated reason for the
decision.
Table 3: Mean rationale–ground-truth cosine similarity and mean latency undergpt-5.2.
Rationale similarity is averaged across all 635 applications for each pipeline.
Pipeline Rationale cos. Latency (s)
Single-LLM 0.709 1.64
Na¨ ıve RAG 0.741 2.14
Agentic RAG0.7794.72
The agentic pipeline has the highest mean rationale similarity, increasing from 0.709
for the single-LLM baseline and 0.741 for na¨ ıve RAG to 0.779. This suggests that the
agentic workflow not only improves decision accuracy but also produces explanations that
are more closely aligned with the human-validated rationales. The metric should still
be interpreted cautiously: cosine similarity measures semantic proximity, not actuarial
correctness, legal sufficiency, or regulatory adequacy. It is therefore best viewed as a
supporting diagnostic alongside decision accuracy and per-scenario accuracy.
The improvement in rationale similarity comes with higher latency. The single-LLM
baseline averages 1.64 seconds per application, na¨ ıve RAG averages 2.14 seconds, and
Agentic RAG averages 4.72 seconds. This reflects the additional retrieval, third-party
checks, and multi-step review used by the agentic workflow. In deployment, the depth
of retrieval and reflection should be tuned to the cost of delay, the cost of unnecessary
referral, and the risk of an incorrect straight-through decision.
5.7 Discussion
The agentic advantage is concentrated in complex and incomplete cases.Over-
all accuracy favors the Agentic RAG pipeline under both base models: 85.1% under
gpt-4o-miniand 86.5% undergpt-5.2. The per-scenario results explain why. The
agentic system is not uniformly better on every simple case; the single-LLM baseline is
slightly stronger on single-issue violations. However, the agentic pipeline is strongest on
20

compliant accepts, multi-step reasoning, recoverable missing information, and irrecover-
able missing information. This is the pattern one would hope to see from an architecture
designed around retrieval, clarification, and reflection.
Retrieval alone is not enough.The naive RAG system improves access to guidebook
text, but it does not consistently improve decisions. In the multi-step reasoning scenario,
naive RAG performs worse than the single-LLM baseline (66.1% versus 70.1%). This
suggests that the limiting factor is not merely whether the relevant rule is present in
context. The system must also know what facts to extract, what intermediate quantity to
compute, and when the available evidence is insufficient. Retrieval supplies information;
agentic orchestration supplies a workflow for using that information.
Missing information requires an explicit escalation pathway.The irrecoverable
missing-information category shows the clearest architectural difference. The correct
response in these cases is not to guess; it is to refer the application for human review.
The Agentic RAG pipeline reaches 84.3% accuracy in this category, compared with 56.7%
for the single-LLM baseline and 53.5% for naive RAG. This result supports the central
claim of the paper: agentic structure is most valuable when the underwriting task requires
recognizing the limits of the available evidence.
The accept/reject/refer boundary remains a policy choice.The results should
not be interpreted as showing a single universally optimal automation threshold. In
practice, the decision to accept, reject, or refer depends on the relative costs of false ac-
ceptance, false rejection, unnecessary human review, and customer delay. A conservative
insurer may prefer more referrals, while a high-volume digital channel may prefer a nar-
rower referral boundary. The agentic framework is useful because it makes this boundary
explicit and adjustable rather than burying it inside a single prompt response.
Rationale quality improves with structure, but latency increases.The agentic
pipeline produces the highest rationale similarity undergpt-5.2, with a mean cosine
similarity of 0.779. This supports the idea that structured intermediate reasoning can
improve not only the final decision but also the explanation attached to that decision.
The cost is latency: the agentic workflow averages 4.72 seconds, compared with 1.64
seconds for the single-LLM baseline and 2.14 seconds for naive RAG. For straight-through
underwriting, this additional cost may be acceptable for difficult cases but unnecessary for
obvious accepts or explicit violations. This motivates future work on adaptive reflection,
where the system invokes the full agentic loop only when the expected value of additional
reasoning exceeds its cost.
Limitations.The corpus is synthetic, and the guidebook rules are stylized. Although
the scenarios were manually reviewed for consistency and realism, they cannot fully rep-
resent the ambiguity, noise, and institutional variation of production underwriting. The
multi-step cases are designed to test a specific form of logical composition, but real under-
writing may involve more subtle judgment, conflicting evidence, regulatory constraints,
or broker-specific context. In addition, the reported results use a fixed retrieval setup
and prompt design. Production deployment would require calibration against real under-
writer decisions, monitoring for drift, and careful governance around what the system is
allowed to decide automatically.
21

6 Conclusion and Future Work
This study demonstrates that agentic artificial intelligence can provide a coherent frame-
work for automating and enhancing selected parts of the underwriting process in small
commercial insurance. By integrating retrieval, completeness checking, third-party data
access, and reflection over multi-step rules, the system mirrors realistic decision pathways
more closely than a single-pass LLM or a naive RAG workflow. The revised experiment
removes the earlier logistic-regression-style simulated screen and instead focuses on a
more document-grounded problem: determining whether the correct underwriting deci-
sion can be reached by connecting facts across the application, third-party information,
and the guidebook.
The results suggest that the agentic approach is most valuable in the cases where
straight-through underwriting is most fragile. Simple accepts and direct violations can
often be handled by simpler systems. The harder cases involve derived quantities, missing
facts, or uncertainty about whether the available evidence is sufficient. In those settings,
the agentic pipeline improves accuracy and produces more aligned explanations, while
also providing a clearer audit trail.
In practice, the full value of an agentic pipeline is realized when AI and humans
work together. The system can reduce cognitive load by handling straightforward cases,
retrieving relevant context, and documenting intermediate reasoning. Human reviewers
benefit from structured explanations and clearer uncertainty signals, while retaining au-
thority over nuanced or high-impact decisions. This is consistent with actuarial standards
of professionalism, transparency, and accountability.
Several limitations should be emphasized. The experiment uses a synthetic BOP
dataset rather than production underwriting data. Although the revised dataset includes
missing-information cases and multi-step reasoning scenarios, it does not capture the full
complexity of real insurer data, policy forms, legacy systems, or operational constraints.
The guidebook and third-party data are also simulated, so the results should be inter-
preted as a controlled comparison of workflow designs rather than a production validation
study. In addition, the results depend on the prompts, retrieval setup, and foundation
model used in the experiment. These choices would need to be recalibrated in a different
underwriting environment. Most importantly, the system is intended to support triage
and documentation, not to replace underwriter or actuarial judgment.
A key direction for future research lies in the decision economics of agentic workflows.
Every retrieval or reflection call carries an implicit cost in latency, compute expenditure,
and potentially external data licensing fees. Optimally, an agent should weigh the ex-
pected value of improved certainty against these marginal costs. Formalizing this trade-off
could yield an economic control model of reasoning depth, where the decision to requery,
retrieve third-party data, or invoke reflection is governed by an expected utility function.
Future work should also compare retrieval-based systems with fine-tuned language
models. Fine-tuning may be useful for stable extraction, routing, or output-formatting
tasks, and may reduce latency or cost when smaller local models are sufficient. However,
RAG is better suited to underwriting rules that change over time or require auditable
citation to current guidebook language. Our preliminary fine-tuning experiments did
not show a clear improvement over the foundation-model baselines, but they were not
broad enough to support a formal conclusion. A likely production approach is a hybrid
system that uses fine-tuning for stable recurring tasks and RAG for dynamic context and
evidence retrieval.
22

Future work will also focus on empirical validation in partnership with insurers and
insurtech firms. Synthetic data are valuable for controlled experimentation, but true
robustness requires evaluation on heterogeneous, production-grade submissions. Collab-
orative pilots can test the system’s reliability under real ingestion conditions such as
missing fields, OCR noise, ambiguous narrative text, conflicting third-party signals, or
changing underwriting appetite.
Several technical enhancements remain open for exploration:
•Adaptive reflection loops:dynamically deciding when to terminate reasoning
based on confidence thresholds and cost penalties.
•Context optimization:selective retrieval that balances semantic relevance with
token economy.
•Robust data ingestion:improving resilience to unstructured or noisy input using
hybrid symbolic-neural preprocessing.
•Human-AI collaboration models:formalizing escalation triggers and account-
ability boundaries between automated and actuarial review stages.
•Cross-domain generalization:extending beyond BOP to personal lines, rein-
surance analytics, and claims reserving while maintaining interpretability.
Agentic AI offers a path toward structured, economically aware automation. By align-
ing computational effort with actuarial principles of efficiency, fairness, and auditability,
such systems can evolve from experimental prototypes into operational decision partners.
The long-term vision is not to replace human underwriters or actuaries, but to strengthen
their ability to supervise, calibrate, and continually improve modern insurance decision
workflows.
Acknowledgements
The author gratefully acknowledges the contributions of the students who provided hu-
man oversight during the evaluation of model outputs. Their careful review, structured
feedback, and thoughtful discussions substantially improved the realism and reliability of
the agentic AI framework developed in this study. In alphabetical order, sincere thanks go
to Breanne Bair, Nathan Brueck, Vincent Galbraith, Cade Sanders, and Riley Williams.
Their work strengthened both the empirical rigor and the practical relevance of this
research.
References
Asai, A., He, Y., Chen, X., and Hajishirzi, H. (2024). Self-rag: Learning to retrieve,
generate, and critique through self-reflection. InInternational Conference on Learning
Representations (ICLR).
Balona, C. (2023). Actuarygpt: Applications of large language models to insurance and
actuarial work. SSRN pre-print.
23

Botti, V. (2025). Agentic ai and multiagentic: Are we reinventing the wheel?arXiv
preprint arXiv:2506.01463.
Carlin, S. and Mathys, S. (2024). A primer on generative ai for actuaries. Technical
report, Society of Actuaries Research Institute.
Casualty Actuarial Society (2019). Statement of principles on ratemaking. CAS State-
ment. Available from the Casualty Actuarial Society website.
Chen, M., Wang, J., Qiu, Y., et al. (2024). Mindagent: Emergent dynamics and autonomy
in llm-based multi-agent systems.arXiv preprint arXiv:2403.13890.
England, P. D. and Verrall, R. J. (2002). Stochastic claims reserving in general insurance.
British Actuarial Journal, 8(3):443–518.
Floridi, L., Cowls, J., Beltrametti, M., Chatila, R., Chazerand, P., Dignum, V., Luetge,
C., Madelin, R., Pagallo, U., Rossi, F., Schafer, B., Valcke, P., and Vayena, E. (2018).
Ai4people—an ethical framework for a good ai society: Opportunities, risks, principles,
and recommendations.Minds and Machines, 28(4):689–707.
Grossi, P. and Kunreuther, H., editors (2005).Catastrophe Modeling: A New Approach
to Managing Risk. Huebner International Series on Risk, Insurance and Economic
Security. Springer, New York, NY.
Gupta, S., Ranjan, R., and Singh, S. N. (2024). A comprehensive survey of retrieval-
augmented generation (rag): Evolution, current landscape and future directions.arXiv
preprint arXiv:2410.12837.
Hatzesberger, S. and Nonneman, I. (2025). Advanced applications of generative ai in
actuarial science: Case studies beyond chatgpt.arXiv pre-print arXiv:2506.18942.
Izacard, G. and Grave, E. (2022). Few-shot learning with retrieval augmented language
models. InTransactions on Machine Learning Research.
Ji, Z., Lee, N., Frieske, R., et al. (2023a). Survey on hallucination in natural language
generation.ACM Computing Surveys.
Ji, Z., Lee, N., Frieske, R., et al. (2023b). Survey on hallucination in natural language
generation.ACM Computing Surveys, 55(12).
Jobin, A., Ienca, M., and Vayena, E. (2019). The global landscape of ai ethics guidelines.
Nature Machine Intelligence, 1(9):389–399.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., K¨ uttler, H., Lewis,
M., Yih, W.-t., Rockt¨ aschel, T., Riedel, S., and Kiela, D. (2020). Retrieval-augmented
generation for knowledge-intensive nlp tasks. InAdvances in Neural Information Pro-
cessing Systems (NeurIPS) 33. arXiv:2005.11401.
Li, Z., Yang, Y., Zhang, H., et al. (2023). Autogen: Enabling next-gen llm applications
via multi-agent conversation. arXiv preprint arXiv:2308.08155.
Liu, H., Zhao, H., Ren, X., et al. (2024). Agentbench: Evaluating llms as agents.arXiv
preprint arXiv:2308.03688.
24

Mack, T. (1993). Distribution-free calculation of the standard error of chain ladder reserve
estimates.ASTIN Bulletin, 23(2):375–393.
Madaan, A., Yazdanbakhsh, A., et al. (2024). Self-refine: Iterative refinement with self-
feedback in large language models.arXiv preprint arXiv:2303.17651.
Mahohoho, B., Chimedza, C., Matarise, F., and Munyira, S. (2024). Artificial intelligence-
based automated actuarial pricing and underwriting model for the general insurance
sector.Open Journal of Statistics, 14(3):294–340.
Park, J., Ouyang, L., Chen, B., et al. (2023). Generative agents: Interactive simulacra
of human behavior.arXiv preprint arXiv:2304.03442.
Richman, R. (2024). An ai vision for the actuarial profession.e-forum of the Casualty
Actuarial Society.
Roberts, K., Schmidt, J., and Peterson, S. (2023). Informed decision making with catas-
trophe risk insights.The Insurer. Guy Carpenter viewpoint article.
Russell, S. and Norvig, P. (2016).Artificial Intelligence: A Modern Approach. Pearson,
3rd edition.
Shuster, K., Xu, J., Smith, E., et al. (2022). Blenderbot 3: A 175b-parameter, open-
domain chatbot that learns to retrieve and refine. InarXiv preprint arXiv:2208.03188.
Trivedi, H., Karpukhin, V., Lewis, M., et al. (2023). Rethinking retrieval-augmented
generation from first principles.arXiv preprint arXiv:2309.01603.
Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. InAdvances
in Neural Information Processing Systems.
Wang, Y., Chen, Y., Liang, H., et al. (2024). A survey on multi-agent systems for large
language models.arXiv preprint arXiv:2402.01680.
W¨ uthrich, M. V. and Merz, M. (2008).Stochastic Claims Reserving Methods in Insurance.
Wiley, Chichester.
Yeo, N., Lai, R., Ooi, M. J., and Liew, J. Y. (2019). Literature review: Artificial intelli-
gence and its use in actuarial work. Technical report, Society of Actuaries.
A Construction of Multi-Step Reasoning Scenarios
The revised experiment replaces the earlier logistic-regression-style quantitative screen
with a set of multi-step underwriting scenarios. These scenarios are designed to test
whether a model can combine facts across documents before applying an underwriting
rule.
A typical multi-step case has four components:
1.A guidebook rulethat defines a threshold, exclusion, or eligibility condition.
2.An application factthat provides part of the information needed to apply the
rule.
25

3.A second application or third-party factthat must be connected to the first
fact.
4.A derived conclusionthat determines whether the application should be ac-
cepted, rejected, or referred.
The scenarios were generated so that the decisive issue is not usually stated in a single
sentence. Instead, the model must identify the relevant rule, determine what quantity or
condition the rule requires, extract the needed facts, and apply the rule to the derived
value. The following templates were used repeatedly across business types.
Square-footage allocation.The application states total leased square footage, while
a separate note states the percentage devoted to a specific operation. The guidebook
threshold applies only to that operation. For example, if a business leases 14,000 square
feet and 50% is used for a grocery operation, the derived grocery area is 7,000 square
feet. If the guidebook permits grocery operations only below 5,000 square feet, the correct
decision is reject.
Revenue-mix thresholds.The application states total revenue and the percentage
generated by a restricted activity. The guidebook may exclude a risk if more than a
certain share of revenue comes from that activity. The model must connect the revenue
share to the relevant exclusion and determine whether the threshold is exceeded.
Shared-premises conditions.The application may describe an eligible business, while
a lease note or third-party description indicates that the business shares premises with
an excluded operation. The model must recognize that the shared-premises rule applies
even when the primary operation is otherwise eligible.
Combined worker counts.Some rules apply to total on-site workers, while the ap-
plication lists employees, contractors, and seasonal workers separately. The model must
combine the relevant counts before applying the guidebook threshold.
Activity-specific operations.A business may be eligible for its main operation but
ineligible if a secondary activity exceeds a stated frequency, area, or revenue threshold.
For example, a studio may be acceptable for standard instruction but require referral if
it hosts high-attendance events more than a specified number of times per month.
These examples are intentionally simple enough to permit human validation, but
complex enough to distinguish rule lookup from structured rule application. They also
create a natural role for agentic orchestration: one agent can retrieve the rule, another
can identify missing or derived facts, and a reflection step can check whether the final
decision follows from the evidence.
26
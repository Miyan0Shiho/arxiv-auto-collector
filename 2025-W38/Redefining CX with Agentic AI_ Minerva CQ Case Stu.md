# Redefining CX with Agentic AI: Minerva CQ Case Study

**Authors**: Garima Agrawal, Riccardo De Maria, Kiran Davuluri, Daniele Spera, Charlie Read, Cosimo Spera, Jack Garrett, Don Miller

**Published**: 2025-09-16 02:30:33

**PDF URL**: [http://arxiv.org/pdf/2509.12589v1](http://arxiv.org/pdf/2509.12589v1)

## Abstract
Despite advances in AI for contact centers, customer experience (CX)
continues to suffer from high average handling time (AHT), low first-call
resolution, and poor customer satisfaction (CSAT). A key driver is the
cognitive load on agents, who must navigate fragmented systems, troubleshoot
manually, and frequently place customers on hold. Existing AI-powered
agent-assist tools are often reactive driven by static rules, simple prompting,
or retrieval-augmented generation (RAG) without deeper contextual reasoning. We
introduce Agentic AI goal-driven, autonomous, tool-using systems that
proactively support agents in real time. Unlike conventional approaches,
Agentic AI identifies customer intent, triggers modular workflows, maintains
evolving context, and adapts dynamically to conversation state. This paper
presents a case study of Minerva CQ, a real-time Agent Assist product deployed
in voice-based customer support. Minerva CQ integrates real-time transcription,
intent and sentiment detection, entity recognition, contextual retrieval,
dynamic customer profiling, and partial conversational summaries enabling
proactive workflows and continuous context-building. Deployed in live
production, Minerva CQ acts as an AI co-pilot, delivering measurable
improvements in agent efficiency and customer experience across multiple
deployments.

## Full Text


<!-- PDF content starts -->

Redefining CX with Agentic AI: Minerva CQ Case Study
Garima Agrawal Riccardo De Maria Kiran Davuluri Daniele Spera
Charlie Read Cosimo Spera Jack Garrett Don Miller∗
Minerva CQ, California, USA
{garima, riccardo.demaria, kdavuluri, daniele}@minervacq.com
{cread, cosimo, jack, dmiller}@minervacq.com
Abstract
Despite advances in AI for contact cen-
ters, customer experience (CX) contin-
ues to suffer from high average handling
time (AHT), low first-call resolution,
and poor customer satisfaction (CSAT).
A key driver is the cognitive load on
agents, who must navigate fragmented
systems, troubleshootmanually, andfre-
quently place customers on hold. Exist-
ing AI-powered agent-assist tools are of-
ten reactive—driven by static rules, sim-
ple prompting, or retrieval-augmented
generation (RAG) without deeper con-
textual reasoning.
We introduceAgentic AI: goal-driven,
autonomous, tool-using systems that
proactively support agents in real time.
Unlike conventional approaches, Agen-
tic AI identifies customer intent, trig-
gers modular workflows, maintains
evolving context, and adapts dynam-
ically to conversation state.
This paper presents a case study of
Minerva CQ, a real-time Agent As-
sistproductdeployedinvoice-basedcus-
tomer support. Minerva CQ integrates
real-time transcription, intent and senti-
ment detection, entity recognition, con-
textual retrieval, dynamic customer pro-
filing, and partial conversational sum-
maries—enabling proactive workflows
and continuous context-building. De-
ployed in live production, Minerva CQ
acts as an AI co-pilot, delivering mea-
surableimprovementsinagentefficiency
∗All authors contributed equally to this work. This
work was conducted as part of the Minerva CQ AI
Research Division.and customer experience across multi-
ple deployments.
1 Introduction
Customer experience (CX) has become a
defining differentiator across industries—from
telecommunications and retail to healthcare
and automotive [ Arkadan et al.2024 ]. Within
this landscape,contact centers and service
teamsserve as the frontline of brand engage-
ment. Yet despite increasing investments in au-
tomation and AI, service quality often remains
inconsistent. Persistent challenges include:
•High average handling time (AHT) due to
manual troubleshooting and fragmented
tool usage
•Customer frustration from being placed
on hold or repeating information
•Agent overload navigating complex work-
flows and unstructured knowledge bases
(KBs)
•Variability in AI-generated answers due to
lack of structure and contextual awareness
in LLM/RAG systems
•Low first-call resolution (FCR) and declin-
ing customer satisfaction (CSAT) and Net
Promoter Score (NPS)
To address these challenges, many en-
terprises have adoptedagent-assist sys-
temsdesigned to surface relevant an-
swers, summarize interactions, or suggest
next steps [ Ye et al.2025 ,Yang et al.2025 ,
Sapkota et al.2025 ]. However, most existing
solutions are inherentlyreactive—they respond
only when prompted, lack temporal awareness,
and fail to support agents proactively through-
out the customer journey.arXiv:2509.12589v1  [cs.AI]  16 Sep 2025

Limitations of Reactive Agent-Assist
Systems
Today’s agent-assist tools typically fall into
three categories:
•Rule-based or decision-tree systems that
are rigid and prone to failure under con-
versational drift
•Prompt-based LLMs that generate plau-
sible responses but lack continuity and
memory
•Retrieval-Augmented Generation (RAG)
systems that fetch documents from a KB
based on queries, but rely on explicit
prompting and offer limited contextual
integration
These solutions are passive by design. They
do not track user goals, update internal mem-
ory, or orchestrate multi-step reasoning. As
a result, the cognitive burden remains on the
human agent to interpret outputs and deter-
mine next actions—especially under real-time
pressure.
The Shift Toward Agentic AI
To move beyond the limitations of reactive sys-
tems, we advocate for a new paradigm:Agen-
tic AI—intelligent, autonomous, goal-driven
systemsthatoperatewithcontinuityandintent.
AgenticAIsystemsdonotmerelyrespond; they
act. They infer intent, maintain evolving mem-
ory, plan actions, and invoke tools in service of
customer outcomes. While the term “agentic”
is gaining popularity, true agentic behavior re-
quires integrated planning, workflow orchestra-
tion, and sustained contextual reasoning—not
just LLMs augmented with tools.
Minerva CQ: Real-Time Agent-Assist,
Not Just RAG
We present a case study ofMinerva CQ, a
real-time agent-assist product purpose-built for
voice-based customer support. Unlike stan-
dalone RAG platforms, Minerva CQ leverages
RAG selectively as one component within a
broader agentic framework. The system in-
tegrates multilingual automatic speech recog-
nition (ASR), dynamic intent and sentiment
detection, autonomous workflow triggering, en-
tity tracking, contextual profiling, and live con-
versation state management.Minerva CQ incorporates several distinctive
capabilities that elevate it beyond traditional
agent-assist tools. The system automatically
identifies customer intent from live conversa-
tions and generates AI-suggested queries, al-
lowing agents to retrieve answers without man-
ual typing or search. Frequently asked ques-
tions—mined from both historical transcripts
and live queries—are validated and cached to
provide instant responses when applicable, re-
ducing latency, cost, and reliance on LLMs.
It also generates compact, incrementalpar-
tial summariesthat preserve and evolve the
context across conversation turns, supporting
downstream reasoning and recommendations.
Additionally, MinervaCQsurfacesintent-aware
workflowsandnext-bestactionstoguideagents
in real time, and concludes each interaction
with an automatic call summary, CSAT/NPS
estimation, and sentiment tracking throughout
the call.
By embedding Agentic AI principles across
the full interaction pipeline, Minerva CQ trans-
forms agent assist into a proactive AI co-
pilot—enhancing both agent efficiency and cus-
tomer satisfaction in high-stakes, real-time en-
vironments. Notably, Minerva CQ has demon-
stratedsuccess inbothcustomerserviceand
sales scenarios, dynamically adapting to the
tone, intent, and goals of the interaction. This
flexibility enables consistent support across a
wide range of workflows, from resolving techni-
cal issues to driving conversions.
Contributions
The key contributions of this paper are:
•We present a deployed agentic AI system
for real-time voice-based contact centers,
integrating multilingual ASR, contextual
understanding, and workflow automation.
•We describe a novel combination of par-
tial conversational summaries, proactive
query generation, validated FAQ caching,
and intent-triggered workflows that reduce
cognitive load and operational latency.
•We report results from live production pi-
lots demonstrating measurable improve-
ments in AHT, FCR, conversion rates, and
customer satisfaction metrics.

The remainder of this paper outlines our
system design, operational insights, and the
practical implications of deploying Agentic AI
at scale in enterprise contact centers.
2 End-to-End Agentic AI Workflow
for Customer Experience
Minerva CQ’s Agentic AI platform is designed
to transform customer experience (CX) by sup-
porting human agents throughout the full life-
cycle of a conversation—proactively, intelli-
gently, and in real time. Unlike traditional
systems that offer isolated suggestions or reac-
tive responses, Minerva CQ functions as an in-
tegrated, goal-driven co-pilot, embedded seam-
lessly within live voice-based interactions.
This section outlines the key phases of a typ-
ical customer call, showing how Minerva CQ’s
agentic capabilities align with natural touch-
points in the agent–customer journey. Each
system component is described along with its
motivation and design rationale.
2.1 Call Initiation: Establishing
Customer Context
Atcallstart, thesystemidentifiesthecallerand
retrieves relevant CRM data. Beyond static
context fetching, Minerva CQ’s entity recogni-
tionmoduleparsesthelivetranscripttoextract
identifiers (e.g., name, email, account number)
as they are mentioned. This eliminates repeti-
tive verification questions, accelerates authen-
tication, and improves early engagement.
2.2 Intent Recognition and Workflow
Triggering
As the conversation progresses, Minerva CQ
continuously analyzes the transcript to detect
underlying customer intent. Once intent confi-
dence is high, it proactively triggers the corre-
sponding workflow—such as a billing correction
or plan change—and surfaces next-best actions
tailored to that scenario. This removes the
need for manual search, enabling faster and
more consistent resolution.
2.3 Customer Profiling in Sales
Conversations
For outbound or sales-oriented calls, Minerva
CQ passively builds a behavioral profile by
detecting expressions of interest, hesitation,
or goal-oriented language. These cues feedinto lightweight dynamic profiling, enabling
tailored offer recommendations, upsell paths,
or probing questions—supporting conversions
without breaking conversational flow.
2.4 Context-Aware Knowledge
Querying: Beyond Keyword-Based
RAG
Navigating fragmented knowledge bases (KBs)
mid-conversation is a major agent pain point.
Traditional keyword lookups via portals, PDFs,
or spreadsheets slow down calls and increase
cognitive load.
While many modern agent-assist tools inte-
grate RAG search bars, they still rely on agents
to manually craft queries—an unrealistic expec-
tation under real-time pressure. Poorly formed
queries further degrade retrieval quality.
Minerva CQ automates this step by generat-
ing semantically rich, context-grounded queries
directly from the live transcript. It identifies
implicit or explicit customer asks and formu-
lates well-structured, KB-compatible questions,
presented as clickable prompts.
To reduce latency and cost, Minerva CQ
maintains a validated FAQ cache mined from
transcripts and live usage. Matching queries
retrieve instant responses, bypassing RAG en-
tirely. New FAQs undergo heuristic and ontol-
ogy checks before being added to the cache.
Example:If a customer says,“I want to
get a travel plan”, a naive system might
ask“Which offer is the customer ask-
ing about?”—too vague for KB retrieval.
Minerva CQ reformulates this into action-
able queries such as“Which travel offers
are available?”or“How to activate
a travel plan?”, bridging conversational
language and structured KB content.
2.5 Partial Summarization:
Maintaining Conversation Context
Minerva CQ produces compact, incremen-
talpartial summariesthroughout the call.
These evolve turn-by-turn, capturing salient
facts while filtering noise—helpful for both
agents (quick reference) and downstream mod-
ules (maintaining cross-turn context for reason-
ing, workflow triggers, and sentiment analysis).

2.6 Agent Guidance: Sentiment,
Intent, and Interaction Indicators
In parallel, Minerva CQ tracks sentiment shifts,
intent changes, and CSAT/NPS likelihood in
real time. Indicators appear on the live agent
dashboard, enabling on-the-fly adjustments in
tone, pacing, and phrasing—critical for de-
escalation, upselling, and emotionally sensitive
situations.
2.7 Call Conclusion and Compliance
At call end, Minerva CQ auto-generates a fi-
nal summary including primary intent, resolu-
tion path, agent actions, and sentiment trajec-
tory. Personally identifiable information (PII)
is redacted to ensure compliance and reduce
post-call documentation.
2.8 A Unified Agentic Framework
Minerva CQ orchestrates these capabilities
through anobserve→understand→decide
→act→assist→learnloop. Unlike mod-
ular tools that operate in isolation, Minerva
CQ connects each stage to sustain momentum
toward resolution. Its proactive, goal-oriented
design reduces agent burden while enhancing
consistency and personalization.
Key Features Summary.The following ca-
pabilities collectively distinguish Minerva CQ
as a real-time agent-assist solution built on
Agentic AI principles. Their operational bene-
fits and KPI impact are summarized in Table 1
in Section 3:
•Real-time entity extraction, intent recog-
nition, and dynamic workflow triggering
•Proactive AI-suggested query generation
with validated FAQ caching
•Partialconversationsummariesforcontext
retention
•Live sentiment/CSAT/NPS tracking and
automated final summaries
3 Measuring Feature Impact and
ROI
While Section 2 detailed Minerva CQ’s multi-
phase agentic workflow, this section examines
how specific features translate into measurableoperational benefits and business outcomes.
Drawing on real-world pilot deployments in
a sales usecase, we quantify improvements in
agent productivity, customer experience, and
revenue metrics.
3.1 Operational Impact of Key
Features
Table 1 maps Minerva CQ’s major agentic fea-
tures to their operational benefits and the KPIs
they most directly influence. This condensed
view connects the system capabilities described
in Section 2 to downstream performance mea-
sures without repeating the detailed descrip-
tions.
3.2 Pilot ROI Analysis
Evaluation Methodology.The pilot was
structured as a live production A/B test with
two cohorts of 50 agents each: one group using
Minerva CQ’s agentic AI, the other operat-
ing without AI support. Both handled the
same mix of inbound service and sales calls
under identical routing rules and product of-
ferings. The evaluation covered approximately
40,000 production voice interactions over the
pilot period. KPIs—Average Handling Time
(AHT), Lead-to-Enquiry (L2E) conversion, and
booking conversion (enquiry→booking)—were
computed from call logs and CRM outcomes.
While seasonality and time-of-day effects were
notfullyeliminated, thebalancedcohortdesign
reduces major confounders.
Key ROI Insights.
•38% reduction in Average Handling
Time (AHT)– AHT reduced from4m
43s to 2m 55s, enabling faster call reso-
lution and lower cost-to-serve.
•33% uplift in Lead-to-Enquiry (L2E)
conversion– Proactive guidance and AI-
suggested queries improved pipeline effi-
ciency.
•4.8% uplift in booking conversion–
Consistent, validated responses and con-
textual guidance resulted in measurable
revenue impact.
As shown in Figure 2, these improvements
are reflected in all three KPIs, with Minerva-
assistedagentsoutperformingthecontrolgroup
across efficiency and conversion measures.

Conversation Timeline
Call Start Problem Discovery Resolution & Guidance Call Closure
Entity Recognition
•Real-time parsing
•CRM data retrieval
•Eliminates re-
peat questionsIntent & Workflow
•Continuous analysis
•High-
confidence class
•Next-best actionsKnowledge
Querying
•AI-generated
auto queries
•Clickable prompts
•Query reformulationSummarization
•Incremental
partial summaries
•PII redaction
•Post-call fi-
nal summary
Customer
Profiling
•Behavior analysis
•Interest detection
•Personal-
ized offersFAQ Cache
•Cached responses
•Latency
reduction
•Validation
Agent Guidance
Dashboard
•Real-time sentiment
•CSAT/NPS
likelihood
Human Agent
Conversation Timeline System Connections
Figure 1: Minerva CQ’s Agentic AI Framework: Human agents are supported by intelligent
AI modules that activate during specific conversation phases. The system provides real-time
transcript analysis, proactive knowledge retrieval, behavioral profiling, and continuous context
preservation—transforming the agent experience from reactive assistance to goal-driven co-
piloting.
Latency and Cost Savings from FAQ
Matching.FAQ matching delivered both
speed and cost advantages. In deployment
logs from 10,000 calls, approximately 7,000
queries were answered from the validated FAQ
cache instead of invoking RAG. Each avoided
RAG call saved agents an average of 6 seconds
of wait time (cumulative ∼11.7 hours latency
saved) and reduced LLM API usage, lowering
inference costs. Traditional RAG retrieval via
OpenSearch typically took 5–9 seconds, while
FAQ matching returned results in under 0.5
seconds—helping to compress handling times
and improve customer experience.
Multilingual Capability.Minerva CQ sup-
ports multilingual and mixed-language conver-
sations in live deployments, includingEnglish
for US contact centers,Italianand other Eu-
ropean languages for EU contact centers, andHindi/HinglishforIndiainAsia. IntheIndia
deployment, the client setup has no language
flag for outbound calls—customers may speak
Hindi, English, or switch freely between the
two. Agents can speak both languages, but not
all can read Hindi. To address this, Minerva
CQ’s speech-to-text pipeline:
1.Accurately transcribes Hindi, Indian En-
glish, or Hinglish speech in real time.
2.Converts transcriptions into English text
for AI-suggested prompts, ensuring all
agents can read suggestions without delay.
This design enables true code-switching sup-
port without introducing additional latency,
maintaining real-time AI guidance in dynamic
multilingual settings.
Summary.Integrating Minerva CQ into live
voice workflows led to:

Feature Operational Benefit KPI Impact
Entity Recognition &
CRM Context RetrievalReduces repetitive verification questions and ac-
celerates call initiationLower AHT
Intent Recognition &
Workflow TriggeringGuides agents with next-best actions; enables
proactive resolutionLower AHT, Higher FCR
AI Query Generation &
FAQ CacheEliminates manual search; provides validated, low-
latency responsesLower AHT, Consistent CX
Partial & Final Summa-
rizationReduces after-call work; supports compliance with
PII redactionLower Wrap-up Time, Improved
Compliance
Table 1: Mapping Minerva CQ features to operational benefits and corresponding KPI improve-
ments.
Figure 2: Key performance improvements with Minerva CQ in a live client case study: 38%
reduction in Average Handling Time (AHT), 33% uplift in Lead-to-Enquiry (L2E), and 4.8%
uplift in booking conversion. Client details withheld; results from production voice interactions.
•Reduced operational cost through faster
call handling
•Improved pipeline engagement and lead
conversion
•Measurable uplift in revenue-driving KPIs
These results validate the transition fromreac-
tive agent-assisttoproactive agentic AI
co-piloting.
4 Related Work
Customer experience and AI-assisted
contact centers -Field studies show that
AI assistance can improve service produc-
tivity at scale. For example, a random-
ized deployment of a generative assistant to
over 5,000 agents increased issues resolved
per hour by 14–15% [ Brynjolfsson et al.2025 ].
Contact-center research also documents persis-
tent agent challenges such as manual knowl-
edge lookups, after-call documentation, andfragmented tool usage, motivating real-time
transcription and summarization to reduce cog-
nitive load [ Sachdeva et al.2023 ]. While adop-
tion is growing, most deployed tools remain
reactive: they respond only when prompted
and lack sustained conversation-level state or
proactive guidance.
From reactive to proactive, context-
aware assistance -A key capability in proac-
tive systems is conversational query reformula-
tion: under-specified utterances must be rewrit-
ten into self-contained queries for accurate re-
trieval. Manual rewrites in TREC CAsT signif-
icantly outperform automatic baselines, high-
lighting the importance of robust reformulation
[Elgohary et al.2019, Dalton et al.2020].
Retrieval-augmented generation and its
trade-offs -Retrieval-Augmented Generation
(RAG) can improve grounding but is highly
sensitive to retrieval quality and can intro-
duce latency and cost in production settings

Figure 3: Minerva-assisted vs. non-assisted performance in the same environment: faster handling
time, higher L2E and booking conversion, and stronger overall volume metrics. Totals shown are
anonymized aggregates from the pilot period.
[Salemi and Zamani2024 ]. This makes fre-
quent or repeated questions an ideal target
for low-latency alternatives.
FAQ retrieval for efficiency -FAQ retrieval
systems offer such an alternative, returning cu-
rated, validated answers with millisecond-level
latency. Early methods relied on query clus-
tering [Sakata et al.2019 ], while recent work
uses dense bi-encoder models with intent-aware
matching [Chen et al.2023].
Our system builds on these strands by (i)
proactively generating KB-compatible queries
from the live transcript, (ii) serving validated
FAQ answers to reduce RAG calls and cost
[Agrawal et al.2024 ], and (iii) maintaining par-
tial, streaming summaries to preserve conver-
sational context across turns .
5 Discussion
5.1 From Reactive Assistance to
Agentic Co-Piloting
Thiscasestudydemonstratesthatshiftingfrom
reactive, prompt-driven agent assist to an agen-
tic, goal-oriented architecture measurably im-proves workflows. Unlike systems that leave
interpretation and next-step selection to the
agent, Minerva CQ maintains evolving context,
plans actions, and invokes tools to advance cus-
tomer outcomes—reducing cognitive load and
variability in real time. Its design integrates
entity extraction and CRM retrieval at call
start, continuous intent detection with work-
flow triggering during the call, and automated
summarization with compliance at close, ensur-
ing continuity across the interaction lifecycle.
5.2 Design Drivers Behind KPI Gains
Two mechanisms stand out as primary contrib-
utors to the observed improvements:
1.Proactive, KB-compatible query gen-
eration with validated FAQ caching:
Reformulating under-specified customer
asks into retrievable queries removes agent
query-writingeffortandmitigatesaknown
RAG failure mode. The validated FAQ
cache returns less than 0.5s responses,
avoiding costly RAG calls. In 10k calls,
∼7k queries were served from cache, sav-

ing∼11.7 hours of cumulative latency.
2.Partial, streaming summaries with
workflow guidance:Turn-by-turn sum-
maries preserve state for downstream rea-
soning, while intent-aware next-best ac-
tions reduce decision latency and support
consistent execution.
Together, these mechanisms align with the
system-to-KPI mapping in Table 1 and match
the efficiency and conversion gains observed in
the A/B pilot.
5.3 Business Impact in Live
Production
In an A/B pilot covering ∼40,000 produc-
tion voice interactions across matched cohorts
(50 agents per arm), Minerva-assisted agents
achieved a 38% reduction in AHT (4m43s →
2m55s), a 33% uplift in L2E conversion, and a
4.8% uplift in bookings.
Shorter AHT directly correlates with in-
creased agent productivity. When agents spend
less time handling each call, they can resolve
more customer issues within the same shift
duration. This allows the same workforce to
handle a higher volume of interactions with-
out increasing headcount—offering a win-win
scenario for both operational efficiency and
business scalability.
Beyond real-time assistance, Minerva CQ
functions as a strategic intelligence layer
through its proactive “voice of the customer”
reporting. By aggregating and analyzing cus-
tomer pain points, agent challenges, and senti-
ment trends, these reports provide stakeholders
with a holistic view of what is working well and
where targeted changes in products, policies, or
service flows could yield greater CX gains. This
positions Minerva CQ not only as an efficiency
driver but also as a business insight engine, en-
abling organizations to act on evidence-based
priorities that extend beyond the call.
These results substantiate the central claim:
moving from reactive assistance to agentic co-
piloting delivers efficiency, conversion, and in-
telligence gains in live, voice-based CX envi-
ronments.5.4Operational Breadth: Multilingual,
Code-Switching, Compliance
The deployment also demonstrates robustness
to multilingual, code-switching environments
(e.g., Hindi/English/Hinglish), where the ASR
pipeline produces readable English prompts for
all agents without added latency—maintaining
guidance quality despite language shifts. Auto-
mated, PII-aware final summaries reduce after-
call work while supporting compliance require-
ments.
5.5 Positioning Within the Literature
and Market
Minerva’s architecture directly addresses well-
documented pain points in contact cen-
ters—manual KB lookups, fragmented tools,
and after-call documentation—by maintaining
conversation-level state and proactively guid-
ing the agent, in line with the broader indus-
try shift from reactive tools to context-aware,
workflow-integrated assistance.
The system also occupies a pragmatic point
in the RAG design space: retrieval is used
selectively, while frequent questions are routed
to a validated, low-latency FAQ path. This
approach reflects known latency/cost trade-
offs of RAG in production and the operational
value of purpose-built FAQ retrieval.
5.6 Limitations
While the live A/B design improves ecological
validity, residual temporal confounders (e.g.,
seasonality, time-of-day) may influence out-
comes. CSAT and NPS remain partly shaped
by factors outside Minerva CQ’s direct control,
such as agent training, customer history, or
offer competitiveness. Minerva mitigates this
through “voice of customer" reporting that sur-
faces systemic friction points, agent feedback,
and sentiment trends.
5.7 Implications for Industry Track
For practitioners, the key takeaway is an
integrative pattern that blends agentic
planning with retrieval pragmatism: (1)
proactive query reformulation; (2) FAQ
caching as the fast path; (3) partial summaries
to maintain state; (4) workflow triggers for
next-best actions; and (5) compliance-ready
documentation—all embedded in anob-
serve→understand→decide→act→assist→learn

loop suited to real-time voice. This pattern
is both deployable and measurable in ROI
terms, while remaining robust to multilingual,
code-switching realities of global CX.
5.8 Future Directions for Industry
Standardized evaluation:Establish shared,
voice-centric evaluation slices (e.g., repeated-
FAQ vs. novel-policy queries, escalation/de-
escalation segments) and report latency distri-
butions alongside accuracy to reflect real-time
constraints.
FAQ governance and drift:Ex-
tend the validated FAQ lifecycle (min-
ing→validation→cache→expiry) with auto-
matic drift detection and policy/version con-
trol, preserving the latency advantage while
maintaining correctness.
Cross-channel generalization:Investi-
gate how partial summarization and intent-
triggered workflows transfer from voice to
chat/email while preserving turnaround-time
guarantees and compliance controls.
Human factors and training:Quantify
how guidance frequency, timing, and UI presen-
tation affect agent trust, handoff smoothness,
and learning curves—particularly in multilin-
gual environments.
Cost–latency knobs:Systematically char-
acterize when to route to FAQ vs. RAG vs. tool
actions under SLA/throughput constraints, ex-
tending the selective-retrieval strategy already
deployed here.
Summary.The evidence indicates
that an agentic, workflow-integrated ap-
proach—anchored by proactive query genera-
tion, validated FAQ caching, and persistent
conversation state—can deliver both efficiency
and revenue benefits at production scale, while
also generating strategic intelligence for the
business. By reducing the average handling
time (AHT), the system effectively increases
agent throughput—enabling the same team
to handle a greater volume of calls without
adding headcount. This translates into higher
productivity per agent and contributes to
scalable growth in support operations. This
positions Minerva CQ as a credible blueprint
for real-time, voice-first agent co-pilots in
enterprise CX.
In sum, Minerva CQ exemplifies the broaderindustry shift toward agentic AI in customer
experience—moving from reactive assistance
to integrated, context-aware co-pilots that not
onlyoptimizeliveinteractionsbutalsogenerate
strategic intelligence for continuous business
transformation.
6 Conclusion
Minerva CQ unifies real-time reasoning, selec-
tive retrieval, and workflow automation to sup-
port human agents end-to-end in live customer
interactions. By maintaining conversational
context, proactively generating high-quality
queries, leveraging validated FAQ caching, and
triggering next-best actions, it addresses per-
sistent contact center challenges of information
overload, fragmented tools, and inconsistent
query handling.
Deployed at production scale, Minerva CQ
delivered measurable efficiency and conversion
gains while maintaining robustness in multi-
lingual, code-switching settings and ensuring
compliance through automated summaries. By
enabling agents to resolve more conversations
within the same operational bandwidth, the
system boosts individual productivity and sup-
ports scalable service delivery. The results
highlights a broader industry shift from re-
active assistance toward integrated, context-
aware co-piloting—where AI systems actively
shape customer outcomes in real time and gen-
erate strategic intelligence for continuous im-
provement.
References
[Agrawal et al.2024] Garima Agrawal,
Sashank Gummuluri, and Cosimo Spera.
2024. Beyond-RAG: Question Identification
and Answer Generation in Real-Time Con-
versations.arXiv preprint arXiv:2410.10136.
https://arxiv.org/abs/2410.10136.
[Brynjolfsson et al.2025] Erik Brynjolfsson,
Jin Li, and Lee G. Raymond. 2025. Gener-
ative AI at scale: The productivity effects
of real-world agent assist.The Quarterly
Journal of Economics. Preprint available at
arXiv:2304.11771.
[Sachdeva et al.2023] Akash Sachdeva, Ming
Li, Brian Strope, and Françoise Beaufays.
2023. Real-time conversation summarization

for customer support agents. InProceedings
of Interspeech 2023, pages 1234–1238.
[Elgohary et al.2019] Ahmed Elgohary, Denis
Peskov, and Jordan Boyd-Graber. 2019.
Can you unpack that? Learning to rewrite
questions-in-context. InProceedings of
EMNLP-IJCNLP 2019, pages 520–526.
[Dalton et al.2020] Jeffrey Dalton, Chenyan
Xiong, and Jamie Callan. 2020. TREC
CAsT 2019: The conversational assis-
tance track overview. InProceedings of
The Text REtrieval Conference (TREC).
arXiv:2003.13624.
[Salemi and Zamani2024] Alireza Salemi and
Hamed Zamani. 2024. Evaluating retrieval
quality in retrieval-augmented generation. In
Proceedings of SIGIR 2024.
[Sakata et al.2019] Takuma Sakata, Hiroaki
Ohshima, Tomohide Shibata, and Tetsuya
Nasukawa. 2019. FAQ retrieval using BERT-
based dense representations. InProceedings
of the 42nd International ACM SIGIR Con-
ference, pages 1113–1116.
[Chen et al.2023] Zheng Chen, Xinya Du, and
Hema Raghavan. 2023. Intent-aware FAQ
retrieval in product search. InProceedings of
the 2023 Conference on Empirical Methods
in Natural Language Processing: Industry
Track, pages 371–381.
[Arkadan et al.2024] Farah Arkadan, Emma
K. Macdonald, and Hugh N. Wilson. 2024.
Customer experience orientation: Concep-
tual model, propositions, and research direc-
tions. InJournal of the Academy of Market-
ing Science, 52:1–24.
[Ye et al.2025] Anbang Ye, et al. 2025. SOP-
Agent: Empower General Purpose AI Agent
with Domain-Specific SOPs. InProceedings
of the AAAI Conference on Artificial Intelli-
gence.
[Yang et al.2025] Bufang Yang, Lilin Xu, et
al. 2025. ContextAgent: Context-Aware
Proactive LLM Agents with Open-World
Sensory Perceptions. InProceedings of the
Annual Meeting of the Association for Com-
putational Linguistics (ACL).
[Sapkota et al.2025] Ranjan Sapkota, Kon-
stantinos I. Roumeliotis, and Manoj Karkee.2025. AI Agents vs. Agentic AI: A Concep-
tual Taxonomy, Applications and Challenges.
arXiv preprint arXiv:2505.10468.
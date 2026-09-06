# InsightToast: Proactive Information Retrieval & Glanceable Visualization in the Side Channel of Data-Rich Meetings

**Authors**: Mohammad Abolnejadian, Matthew Brehmer

**Published**: 2026-08-31 17:22:43

**PDF URL**: [https://arxiv.org/pdf/2608.31115v1](https://arxiv.org/pdf/2608.31115v1)

## Abstract
Missing institutional context during meetings can impede effective participation. Retrieving relevant information, often scattered across heterogeneous internal and external sources, requires costly task-switching that disrupts both individual focus and collective conversational flow, particularly detrimental during cognitively demanding tasks such as decision-making. We introduce InsightToast, a mixed-initiative application that monitors verbal discourse in real time, identifies topics and informational needs as they emerge, and proactively retrieves relevant information through a multi-agent large language model (LLM)-based pipeline integrating retrieval-augmented generation (RAG) to produce source-grounded insights as succinct text and glanceable interactive charts, delivered through a peripheral interface as ephemeral toasts in the conversation's side channel. To demonstrate the potential for yielding serendipitous insights, we showcase a usage scenario involving a knowledge base of legislative documents as the meeting's context. We then report on a comparative study (N=16), in which participants arrived at informed policy decisions while maintaining natural conversation flow.

## Full Text


<!-- PDF content starts -->

InsightToast: Proactive Information Retrieval & Glanceable
Visualization in the Side Channel of Data-Rich Meetings
Mohammad Abolnejadian
School of Computer Science
University of Waterloo
Waterloo, Ontario, Canada
mabolnej@uwaterloo.caMatthew Brehmer
School of Computer Science
University of Waterloo
Waterloo, Ontario, Canada
mbrehmer@uwaterloo.ca
Figure 1:InsightToastis a mixed-initiative interface that lives in the periphery of a meeting, proactively retrieves information
as knowledge gaps emerge, and surfaces glanceable, source-traceable, and context-aware insights through ephemeral toasts.
Abstract
Missing institutional context during meetings can impede effec-
tive participation. Retrieving relevant information, often scattered
across heterogeneous internal and external sources, requires costly
task-switching that disrupts both individual focus and collective
conversational flow, particularly detrimental during cognitively
demanding tasks such as decision-making. We introduceInsight-
Toast, a mixed-initiative application that monitors verbal discourse
in real time, identifies topics and informational needs as they
emerge, and proactively retrieves relevant information through a
multi-agent large language model (LLM)-based pipeline integrating
retrieval-augmented generation (RAG) to produce source-grounded
insights as succinct text and glanceable interactive charts, delivered
through a peripheral interface as ephemeral toasts in the conver-
sation’s side channel. To demonstrate the potential for yielding
serendipitous insights, we showcase a usage scenario involving a
knowledge base of legislative documents as the meeting’s context.
We then report on a comparative study ( 𝑁= 16), in which par-
ticipants arrived at informed policy decisions while maintaining
natural conversation flow.
CCS Concepts
•Human-centered computing →Collaborative interaction;
Visualization systems and tools.
This work is licensed under a Creative Commons Attribution 4.0 International License.
UIST ’26, Detroit, MI, USA
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2856-3/2026/11
https://doi.org/10.1145/3830398.3830522Keywords
Meeting and decision support, Retrieval-augmented generation,
Visualization, Natural language interaction, Proactive AI
ACM Reference Format:
Mohammad Abolnejadian and Matthew Brehmer. 2026.InsightToast: Proac-
tive Information Retrieval & Glanceable Visualization in the Side Channel of
Data-Rich Meetings. InThe 39th Annual ACM Symposium on User Interface
Software and Technology (UIST ’26), November 02–05, 2026, Detroit, MI, USA.
ACM, New York, NY, USA, 16 pages. https://doi.org/10.1145/3830398.3830522
1 Introduction
Every meeting has context. Whenever people meet to make deci-
sions, plan, or ideate together, context includes institutional knowl-
edge, the agendas and transcripts of prior meetings, and data rel-
evant to the topic at hand [ 15]. However, apart from prepared
materials such as slide presentations or agendas that establish con-
text, retrieving and sharingadditionalcontext and data during a
meeting can be highly disruptive to the flow of the conversation.
Disruptions arise from application-switching to search interfaces
and by side channel conversations in which meeting participants
share missing context. Irrespective of how missing context is sought,
people require the right keywords to search or ask about, and this
poses a challenge when the meeting’s domain is unfamiliar and
they lack the terminology (and attentional resources) to articulate
what information they need [ 75]. On the other hand, a conscious
avoidance of these disruptions can lead participants to either defer
information retrieval until after the meeting or to act on whatever
information is available during the meeting, aWhat You See Is All
There Isbias [60] which risks making under-informed decisions.
With the advent of large language models (LLMs), many video-
conferencing platforms (e.g.Microsoft Teams [ 81] and Google Meet
arXiv:2608.31115v1  [cs.HC]  31 Aug 2026

UIST ’26, November 02–05, 2026, Detroit, MI, USA Mohammad Abolnejadian and Matthew Brehmer
[42]) now offer meeting support through transcription and retro-
spective summaries, as well as in-meeting assistance via conver-
sational question answering (e.g.Zoom’s AI Companion [ 133]).
However, these tools remain reactive, requiring conscious initia-
tion and thus perpetuating the interruption costs described above.
We instead posit that information retrieval during synchronous
human-to-human conversations should be as effortless and ambi-
ent as possible, operating in the background without demanding
conscious initiation and delivering results unobtrusively. In doing
so, a communication platform can enhance meeting effectiveness
by establishing common ground among the group through address-
ing participants’ individual information needs as they emerge. [ 63]
While prior work has explored proactive retrieval [ 97] and com-
munication facilitation [ 1,5,71,125], we have yet to encounter an
integrated approach that determineswhenan emerging collabora-
tive need merits delivery andhowto synthesize retrieved content
into a form amenable to synchronous meetings, where information
retrieval and sensemaking are supplemental to the conversation.
In this paper, we present
 InsightToast, a mixed-initiative
interface that monitors spoken discourse in real time, detects knowl-
edge gaps as they emerge, and proactively delivers glanceable,
source-grounded information in the form of ephemeral ‘toast’ notifi-
cations, which are dynamically organized in a meeting side channel
according to detected meeting topics. Led with a concise title that
previews its content, each insight contains either a text snippet or
an interactive chart prioritizing both serendipity and brevity, trace-
able both to its conversational trigger and to its contributing doc-
ument sources.InsightToastemploys a multi-agent LLM-based
pipeline that tracks conversational context, detects information
needs and topics as they emerge, formulates multi-faceted search
queries that address complementary aspects of each identified need,
and ultimately generates insights using a retrieval-augmented gen-
eration (RAG) approach that draws from an indexed knowledge
base and supplementary open web sources. We demonstrateIn-
sightToast’s application through a legislative policy deliberation
scenario, collecting and processing approximately11 𝐺𝐵𝑠 of leg-
islative documents from Canada’s House of Commons open data
and conducting a simulated meeting based on recorded commit-
tee sessions. We then report findings from a comparative study
(𝑁= 16) in which participants engaged in conversations leading to
their decision to support alternative legislative petitions, both using
InsightToastand a baseline search interface. We found that the
former’s multi-faceted retrieval and proactive delivery preserved
natural conversational flow, broadened the scope of deliberation,
and yielded more source-grounded decision justifications, all while
reducing manual retrieval effort. Ourcontributionsinclude:
1.The distillation of three design goals for proactive information
retrieval and presentation during synchronous data-rich meet-
ings, informed by interdisciplinary literature spanning human-
computer interaction, information retrieval, visualization, and
judgment and decision-making;
2.InsightToast, a mixed-initiative interface that lowers the cogni-
tive and temporal cost of information discovery during meetings
by proactively surfacing insights as glanceable, source-grounded
text and interactive charts, using its multi-agent pipeline to inferwhatto retrieve andwhento deliver it subject to organizational
context and the pragmatics of the conversation;
3.Findings from a mixed-methods evaluation ( 𝑁= 16) reporting
on participants’ experiences with glanceable insights surfaced
proactively in the conversation’s periphery during collaborative
deliberation and reflecting on the implications for integrating
such capabilities into synchronous communication platforms.
2 Related Work
We are informed by prior work in synchronous collaboration around
data, proactive information delivery, decision support under cogni-
tive load, and AI-mediated conversational support.
2.1 Data-Rich Synchronous Communication
From large organizations to small collaborative settings, data and
institutional knowledge serve as the basis for meetings where goals
include analyzing trends [ 74], brainstorming ideas [ 101], making
decisions [ 31], and monitoring unfolding events [ 85]. Foundational
work in CSCW characterizes group activity along dimensions of
time and space [ 36,58]; our focus is on the synchronous side of
this space. With respect to space, our work applies to co-located,
remote, and/or hybrid collaboration [ 28]. Irrespective of where
these meetings take place, and as remarked upon in the introduction,
meeting participants face substantial barriers engaging with data
during synchronous meetings [ 15,112]; live conversations routinely
surface questions that expand the meeting scope beyond prepared
materials [ 16], resulting in improvisation that no amount of advance
curation can anticipate, the deferral of discussion to future meetings,
and a risk of under-informed decisions.
Right data, right time.Often the challenge is not that data is
unavailable, but that it exists somewhere within an organization’s
reports or broader web sources [ 20]; it is practically inaccessible
in the moment it is needed. Effective synchronous collaboration
depends on a continuous, low-effort stream of shared contextual
information [ 45], a stream that is broken when participants disen-
gage from the conversation to locate missing evidence. The result
is both an individual cognitive cost, as recovering focus following
interruptions can consume precious minutes [33], as well as a col-
lective cost, as the fluid, exploratory engagement that makes live
conversations productive is brought to a halt [47].
Tracking the conversation.Keeping a coherent record of what
a data-rich conversation produces is itself demanding: it requires
maintaining common ground [ 21], capturing multiple layers of
analytic provenance generated through speech [ 95], and ensuring
that partial findings can be handed off to others [ 130]. Mahyar et al.
[73] observed that record-keeping in collaborative visual analytics
sessions directly competed for the attentional resources needed
to engage in the analysis itself. This problem extends to decision-
making, where traceability between claims and their supporting
data matters for accountability and decision quality [74].
Together, these challenges shape the problem space that we ad-
dress: supporting synchronous collaboration by facilitating informa-
tion retrieval while keeping track of evolving topics of conversation,
ideally without disrupting its flow.

InsightToast UIST ’26, November 02–05, 2026, Detroit, MI, USA
2.2 Data Retrieval and Presentation
In meetings, data is typically surfaced through slide presentations,
spreadsheets, and dashboards, allowing participants to reference
and co-interpret data with visual aids [ 16,62]. While primary dis-
plays attend to these prepared data assets, prior research has exam-
ined backchannel communication as a mechanism for participants
to augment understanding in parallel with the ongoing discussion
[23,32]. Platforms such as MeetCues [ 7] and CommunityClick [ 56]
demonstrate how visual and interactive backchannels can improve
engagement and awareness during meetings. However, engaging
with backchannels imposes a dual-task cost, diverting participants’
visual and cognitive attention from the primary discussion [2, 8].
Proactive retrieval.A key tension arises along the dimension of
information delivery within the mixed-initiative spectrum [ 50,98].
Proactive delivery offers the potential to surface contextually rel-
evant data without requiring people to break conversational flow
[80,118], an idea explored by just-in-time information retrieval
agents [ 97] and more recently by SearchBot [ 5], which listens to spo-
ken conversations, detects entities, and proactively decideswhatre-
lated document to surface. In the information retrieval community,
frameworks [ 102] such as Information Fostering and workshops
including ProActLLM [ 18] have sought to formalize taxonomies for
whena system should initiate retrieval in dialogue contexts. Yet,
these efforts fall short onhowto present proactively retrieved data
in a manner that respects the attentional constraints of meetings.
Glanceable and peripheral presentation.To mitigate this atten-
tional cost, researchers have explored interfaces positioned at the
edge of attention that convey information through ambient, low-
demand encodings [ 77,82,121]. The Peripheral Displays Toolkit
(PTK) [ 76] formalizes this design space, structuring attention man-
agement around abstraction, notification levels, and display transi-
tions. For such displays to be useful without sustained focus, they
must be glanceable, conveying information during the brief peeks
characteristic of constrained display contexts [ 12]. Within this con-
straint, text has been found to be among the fastest media for
information extraction, followed by simple encodings leveraging
pre-attentive visual features [67].
Together, this body of work informs the design ofInsightToast
as a system that combines proactive, context-driven retrieval with
glanceable textual and visual presentations; a pairing that, to our
knowledge, has not been previously prototyped or evaluated in the
context of synchronous data-driven meetings.
2.3 Data-Driven Decision-Making
Decision-making is a cognitively demanding process that relies
heavily on working memory and deliberate reasoning [ 60,110]. In
synchronous conversational settings, this load compounds, where
participants must simultaneously track dialogue, process new in-
formation, and reason about trade-offs [ 9,110]. These individual
burdens extend to the group level: under-informed members tend to
discuss already-shared information and withhold unshared knowl-
edge, biasing the resulting decision [ 108], while reduced participa-
tion makes meetings less effective [ 26,51]. These tensions motivate
the need for lightweight, well-timed decision support that can op-
erate within the constraints of an ongoing conversation [99].Visualization for decision-makers.Decision-makers require
concise overviews of alternatives, criteria, and trade-offs rather
than exhaustive analysis [ 31,87]. Despite visualization being widely
positioned as a decision-support tool, prior work has found that
explicit decision tasks remain underrepresented in visualization
research [ 30], and that most tools focus on the intelligence stage
(data exploration) while providing limited support for generating
alternatives [ 29,87]. Systems such as Dust & Magnet [ 128] and
WeightLifter [ 89] represent targeted efforts toward multi-criteria
decision support, helping people assess trade-offs interactively.
Unknown unknowns.A deeper challenge is not what people
know they are missing, but what they do not know to look for
[115,119]. This is compounded by the vocabulary problem [ 38]:
people struggle to translate information needs into proper queries,
particularly when unfamiliar with a domain [ 75,122]. This dynamic
also surfaces in recommendation systems, where optimizing for ac-
curacy tends to reinforce what people already know at the expense
of relevant alternatives that could reframe a decision [64, 79].
Our work draws on these threads to design lightweight, ambient
data interventions that surface relevant but potentially overlooked
facets of a decision problem in real time, reducing the cognitive cost
of processing information while steering decision-makers toward
information they would not have thought to seek.
2.4 AI Support for Conversations
A growing number of commercial tools now leverage AI to support
conversations. Microsoft Teams [ 81] and Otter [ 88] provide post-
meeting summaries and action items, while platforms like Google
Meet [ 42] and Zoom’s AI Companion [ 133] extend this support by
drawing on conversational and organizational context to answer
participant questions during meetings. Tools such as Cluely [ 14,
22] are beginning to shift toward proactive assistance by drawing
on LLMs’ knowledge. In parallel, research has explored how NLP
and AI-driven systems can support meetings along dimensions
including understanding spoken references to data [ 104], enabling
in-meeting goal reflection [ 19], and designing generative interfaces
that bridge planning, execution, and follow-up [90, 91, 117].
Mixed-initiative AI interfaces for synchronous conversation.
Several works have explored using language models to enrich con-
versations with contextually relevant content. CrossTalk [ 125] cre-
ates intelligent substrates to enable dynamic annotations and in-
meeting queries grounded in the live conversation. Visual Cap-
tions [ 71] and Crosscast [ 124] similarly augment spoken commu-
nication with on-the-fly visuals derived from what is being said.
While these systems demonstrate the value of proactive AI support
in conversational settings, they do not synthesize relevant informa-
tion from a meeting-specific knowledge base into a source-traceable
form, reinforcing the fragmentation between organizational knowl-
edge and synchronous collaboration [15].
Retrieval-augmented generation (RAG) for conversation sup-
port.RAG [ 68] addresses these limitations by grounding model
outputs in external documents, improving factual accuracy and en-
abling source attribution [ 37,129]. Recent extensions handle sense-
making over larger corpora through graph-based indexing [ 35,44],
multi-hop reasoning [ 70], and agentic retrieval pipelines [ 105].

UIST ’26, November 02–05, 2026, Detroit, MI, USA Mohammad Abolnejadian and Matthew Brehmer
Social-RAG [ 120] adopts this framework in asynchronous conver-
sations by retrieving from group interaction histories to socially
ground proactive suggestions. AInsight [ 1] extends this thread to
synchronous settings with a proactive, peripheral interface, though
it offers limited glanceability and integration into the natural con-
versational flow, and lacks a formal evaluation with human subjects.
Our work builds on these foundations by designing and evaluat-
ing a multi-agent AI assistant for during-meeting support that ac-
tively monitors a conversation, tracks topics, identifies knowledge
gaps, and delivers retrieval-augmented insights in the periphery.
3 Design Goals
Based on the challenges discussed in Section 2, we derive twelve
[DesignConcepts] organized around three [DesignGoals]: how
data should be retrieved, how it should be presented, and how meet-
ing participants maintain control and trust over surfaced informa-
tion. Figure 1 illustrates the design concepts through a motivating
scenario of a co-located team meeting about market expansion.
3.1 Proactive, Context-Aware Retrieval
DG1 Manual information seeking forces participants to choose
between staying engaged in discussion and pursuing the context
they need [ 16,33]. To avoid this trade-off, meeting support tools
should autonomously monitor the discussion and trigger retrieval
when participants exhibit a genuine knowledge gap, such as an
explicit information request, an uncertain claim being raised, or a
decision stalled for lack of evidence. Rather than issuing a single
query, tools should decompose each need into multiple comple-
mentary retrievals to mitigate the vocabulary problem [ 38] and
surface perspectives participants may not have considered [ 60]
(e.g.a remark about expanding to a new market might yield sepa-
rate retrievals on market size, competitor presence, and regulatory
landscape). Retrieval should draw from a primary organizational
knowledge base supplemented by broader web sources [ 20], and
should weight recent conversational context more heavily while
remaining aware of the full discussion arc [21].
3.2 Glanceable Peripheral Presentation
DG2 Content delivery must not itself become a source of dis-
ruption [ 2,8]. Drawing on prior work on glanceable and peripheral
interfaces introduced in Section 2.2, we propose that content be
delivered to a secondary display region that participants consult at
their own pace, using non-modal, time-limited notifications that
signal availability without demanding attention. Each piece of con-
tent should be synthesized into glanceable forms, such as concise
text or compact visualizations suited to the content type, optimized
for rapid comprehension during a live discussion [ 12,31], and share-
able across participants to maintain common ground [ 45]. Finally,
consistent with the usability principle of system status visibility
[84], retrieval and synthesis progress (initiated as in
 DG1 )
should be surfaced through ephemeral indicators, giving partici-
pants awareness of what is being generated and when to expect it,
without persisting beyond its moment of relevance.3.3 Traceability and Control
DG3 Surfaced content should carry dual attribution: both its
retrieved data source(s) and its triggering conversational moment(s),
providing accountability, grounding trust in synthesized content,
and clarifying the context for judging relevance [ 95,129]. The
salience of information should shift with the conversation’s focus,
so participants should be able to curate this content, promoting
useful items and dismissing irrelevant ones, preserving human
initiative over the system’s proactively retrieved and synthesized
results [ 50]. Finally, the brevity of this content
 DG2 invites
progressive disclosure (i.e.details-on-demand) [4].
4 InsightToast
Guided by the aforementioned design goals, we designed and de-
veloped
 InsightToast, a system that actively monitors live con-
versation, tracks topics as they emerge and evolve, and detects
information gaps to proactively generate representations of this
information delivered to the conversation’s side channel; as a goal
of these deliveries is serendipity, we hereafter refer to them asin-
sights. Each insight is a succinct, synthesized information snippet
grounded in the meeting’s knowledge base and supplemented by
the open web, traceable to both its conversational trigger and its
underlying data sources and affording curation control over them.
To anchor our explanation, we follow a running example through-
out this section in which a sustainability team at an energy company
deliberates on an emission-reduction target, with the meeting’s
knowledge base comprising internal organizational documents and
a corpus of national legislative and regulatory records.
4.1 Interface and Interaction
To realize
 DG2 and
 DG3 ,InsightToast’s interface jux-
taposes two side panels with a main panel; the latter is either a
video-conferencing view for remote collaboration (Figure 2), or a
participant’s desktop environment in co-located scenarios (Figure 1).
Inspired by macOS’s notification center placement [ 6], ephemeral
Insight Toastsappear as notification overlays in the top-right corner
of the conversation’s main panel, conveying both real-time progress
on information retrieval and a concise title for each newly synthe-
sized insight. AnInsight Sidebarprovides an expandable panel that
aggregates all generated insights, organized under dynamically
identified meeting topics, allowing meeting participants to explore
and curate synthesized content at their own pace.
Toasts.GivenInsightToast’s system-initiated retrieval, partici-
pants must remain aware of when new content becomes available.
We employtoastsas transient, non-modal, time-based UI compo-
nents [ 54] to communicate system status DC2.2 .Progress Toasts
appear once the system detects a knowledge gap and updates incre-
mentally throughout retrieval and synthesis, surfacing high-level
status indicators such as the number of source documents retrieved
and considered for synthesis DC2.4 .Insight Toastsappear upon
the completion of synthesis as the first level of insight disclosure,
displaying a title of up to five words that previews the synthesized
content while limiting conversational disruption [ 86] alongside
speaker attribution and its conversational moment. Participants
can click the toast to be directed to the insight’s full representation
in an Insight side panel (Figure 2.A). Depending on the structure

InsightToast UIST ’26, November 02–05, 2026, Detroit, MI, USA
Figure 2:InsightToast’s interface from Abhi’s view during a remote meeting. (A) Toast notifications ephemerally surface
retrieval progress and synthesized insights as Matt expresses an information need, with insights organized under an expandable
(B) side panel sorted into dynamically detected meeting topics. (C) Each insight appears as an interactive chart or a succinct text
snippet, traceable to its retrieved document sources and originating conversational trigger via a (D) secondary transcript panel.
of the retrieved data, the system’s pipeline selects the appropriate
presentation modality (see Section 4.2), transforming evidence into
glanceable, source-grounded textual or visual insights.
Text Insightssummarize relevant unstructured evidence. They
lead with the most salient finding, anchor claims in temporal and
institutional context (e.g.specifying when testimony was given or
which board meeting a report originates from), and attach inline ci-
tations so that every claim can be traced to its underlying document
DC3.1 . To support glanceability, each text insight is constrained
to a tweet-length snippet capped at 280 characters, a length concise
enough to convey a self-contained message, yet short enough to
be peeked at under five seconds [ 25], consistent with the glance
durations observed for constrained peripheral displays [ 67]DC2.3 .
Meanwhile,Chart Insightsencode relevant structured evidence in
a compact form while embedding finer detail within an interactive
layer accessible on demand DC3.4 . Informed by the glanceable
visualization literature,Chart Insightsare drawn from a restricted
chart inventory favoring encodings known to support rapid human
judgment: length along a common scale for quantities (bar charts)
and temporal patterns (line charts), and arc length for part-to-whole
relationships (donut charts) [ 12]. To preserve legibility within the
compact peripheral display, the system enforces high data-ink ra-
tios [ 55] by suppressing non-essential elements such as gridlines,
redundant legends, and auxiliary annotations.
In Figure 2, which captures a moment in our sustainability meet-
ing scenario, Kai raises a potential tax concern, after which Matt
poses an information need beyond the data readily available in theconversation, stating“I wonder how fiscal gap precedents compare
globally.”Drawing on both meeting and conversational moment
context,InsightToastsurfaces serendipitous insights comparing
corporate tax rate trends at a global scale, notifying all participants
of retrieval and synthesis progress through dismissable toast no-
tifications (Figure 2.A). Abhi, choosing to explore the synthesized
insights further, clicks on an insight toast, which expands the In-
sight Sidebar and brings the full content into his personal view.
There, he can compare tax rate trends among major economies at
a glance, with the ability to inspect exact values by hovering over
any data point in the chart (Figure 2.C).
Side panels.Each participant can access a side panel that can be ex-
panded when further context is needed DC2.1 , either by browsing
synthesized insights under their corresponding topics or by per-
forming reactive retrieval through a direct knowledge-base search,
an LLM chatbot, or a web search engine, complementingInsight-
Toast’s proactive retrieval. The panel helps participants maintain
awareness of subjects discussed during a meeting via topic tracking
DC3.2 . Without requiring meeting participants to define topics at
the outset, the system automatically identifies and creates distinct
topic categories as the conversation unfolds, transitioning between
them as discussion shifts and surfacing the active topic’s content
without requiring manual navigation. To afford control over proac-
tively generated content, participants can collaboratively curate
insights by archiving those deemed irrelevant or pinning those
particularly valuable to the deliberation into shared pin and archive
collections DC3.3 . To facilitate collaboration around a specific

UIST ’26, November 02–05, 2026, Detroit, MI, USA Mohammad Abolnejadian and Matthew Brehmer
Figure 3:InsightToast’s multi-agent LLM-based pipeline processes diarized speech utterances sequentially through a three-
stage process: (1) maintaining a running conversational context to facilitate topic tracking, (2) multi-faceted proactive retrieval
across indexed and web sources, and (3) insight synthesis.
piece of synthesized information, any participant can bring an in-
sight into focus across all participants’ views by selecting it DC2.5
(Figure 2.C). The primary insight side panel is complemented by a
secondary transcript side panel, which allows meeting participants
to reference specific conversational moments at three levels of de-
tail:(1)the full running transcript,(2)utterances belonging to a
specific topic by expanding that topic in the insight side panel, or
(3)the specific conversational trigger of an insight DC3.1.
4.2 Proactive Retrieval and Synthesis Pipeline
InsightToastemploys a three-stage natural language process-
ing pipeline (Figure 3) that continuously ingests transcribed and
diarized speech utterances. It determines when, what, and how
to retrieve and synthesize relevant information, collectively an-
swering the requirements posed by
 DG1 . As the pipeline’s
objective is to support a running meeting from start to finish, it
processes input incrementally, maintaining an evolving picture of
the broader discussion so that later turns can still be interpreted
in light of earlier references and topics. The pipeline is realized
as a multi-agent LLM-based directed graph in which specialized
agents operate over a shared state, enabling task specialization,
independent verification of intermediate results, and distributed
reasoning that collectively surpass the capacity of a single mono-
lithic language model [ 34,43,114], a pattern demonstrated across
different frameworks [123] and application domains [49].
(i) Utterance assessment and transformation.Upon receiving
a transcribed utterance, a content filter agent, informed by the
DAMSL dialogue act annotation scheme [ 24], separates communi-
cation management utterances from task-level content. Retained
segments are either assigned to an existing topic or instantiate a
new one via few-shot classification [ 17,109], while a meeting-level
tracker maintains a running summary and key factual anchors
(e.g., stated targets, participant roles, decisions in progress) to keep
all downstream agents context-aware across assessment, retrieval,
and synthesis. Using this context built around the utterance, the
pipeline next evaluates whether the current moment exhibits a
genuine knowledge gap DC1.1 by weighing the expected utility of
surfacing new information against the cost of an unsolicited toast
[50], operationalized through five trigger categories (detailed in
Appendix A.1), drawing on established information-seeking signals
such as Belkin’s ASK model [ 10], linguistic hedges as markers ofepistemic uncertainty [ 65], and clarification requests signaling in-
complete understanding [ 92]. When a gap is confirmed, a query
construction agent decomposes the underlying information need
into multiple complementary search queries [ 132], each augmented
with hypothetical document descriptions [ 40] to improve semantic
alignment while surfacing perspectives participants may not have
considered DC1.2 . For instance, a remark such as“I’m not sure
what regulatory risks would come with a more aggressive target?”is
decomposed into queries targetingfederal and sub-national report-
ing requirements,feasibility constraints associated with accelerated
emissions reductions, andprior ambitious enforcement precedents,
three aspects the team may not have had the time or thought to
seek individually, before being dispatched to the retrieval stage.
(ii) Source retrieval.Queries are dispatched to parallel subgraphs
following a map–reduce pattern [ 27]. Within each subgraph, a plan-
ner agent refines the query with the help of synthetic hypothetical
documents, applies metadata filters, calibrates retrieval scope, and
determines whether to route retrieval to the meeting’s indexed
knowledge base, a web search API, or both DC1.3 . Following an
adaptive retrieval strategy [ 57,126], a secondary feedback agent
evaluates the retrieved document set for sufficiency and relevance,
adjusting the query and re-executing retrieval in an iterative refine-
ment loop [ 59] that terminates once adequate coverage is confirmed
or a fixed iteration ceiling is reached. Once all subgraphs conclude,
their results are merged into a unified, ranked evidence set.
(iii) Insight synthesis.A routing agent assesses the aggregated
evidence against the current conversational context to determine
whether to surface a new insight, groups queries addressing facets
of the same information need, and routes each group to the appro-
priate output modality: a textual summary, a chart, or both.Text
Insightsare generated following a RAG approach [ 41], ensuring
that every claim remains traceable to its underlying source docu-
ment.Chart Insightsfollow a two-phase plan-then-generate process
inspired by stepwise chart reasoning [ 111], in which a planning
agent first determines the primary analytic intent of the evidence,
either comparison, trend, composition, or ranking, and selects a
chart type from a restricted inventory to support glanceability; Ap-
pendix A.2 provides more detail about this process. A generation
agent then produces the chart as a declarative Vega-Lite [ 100] spec-
ification by writing and executing Altair [ 116] code in a sandboxed
environment, leveraging interleaved reasoning and action [ 127] to

InsightToast UIST ’26, November 02–05, 2026, Detroit, MI, USA
iteratively verify both the structural validity of the specification
and the semantic fidelity of the rendered output.
4.3 Implementation
InsightToast’s backend implements the pipeline described in Sec-
tion 4.2, exposed via a RESTful API to the front-end interface de-
tailed in Section 4.1. The backend is built on LangGraph [ 66] as the
agent orchestration framework, with Gemini Flash Lite Latest (2.5
Flash Lite at development time) serving as the language model for
the majority of pipeline agents and Gemini Flash Latest (3 Flash at
development time) assigned to the chart planner and generation
agents given its stronger tool-use capabilities. The API layer is
implemented in FastAPI [ 96], providing an endpoint that accepts
transcribed speech utterances and streams asynchronous updates
to connected clients. The front-end is a React web application using
Chakra UI [ 3] as its component library and LiveKit [ 72] for video
conferencing. For co-located settings, the interface is embedded
within a simulated macOS desktop environment [ 94] to emulate its
integration within participants’ native environments. Automatic
Speech Recognition (ASR) is handled by Speechmatics [ 106], which
provides speaker identification alongside transcription to fulfill
the pipeline’s diarized utterance requirement. The source code is
available atgithub.com/ubixgroup/InsightToast.
5 Evaluation
To empirically evaluate our approach, we conducted a within-
subjects comparative study. In a
 BaseLine(BL) condition, we
provided participants with an interface that embedded three re-
active search widgets consisting of direct knowledge base search,
a chatbot (powered by the Gemini API), and web search in the
conversation’s side channel to approximate current practices for
retrieving information during meetings. In the
 Insight Toast(IT)
condition, we provided participants with the interface described in
Section 4.1, to which we added the same reactive search capabilities
available in the
 BaseLine(BL) interface.
Figure 4: (A) We first demonstratedInsightToast’s func-
tionality to study participants by synchronizing it with the
playback of an open government webcast recording. The
main study protocol accommodated both remote (Figure 2)
and (B) face-to-face conversations.
Scenario and knowledge base.Policy deliberations typically draw
heavily on precedents and accumulated evidence to inform decision-
making. Legislative committee meetings exemplify this pattern, as
they are accompanied by extensive context, including prior re-
ports, research briefings, and legislative documents such as bills
and amendments. These sessions are routinely published alongsidetheir supporting materials as part of governmental transparency
mandates, providing both recorded deliberations and a rich, openly
available knowledge base well-suited for developing and evaluating
InsightToaston real-world, data-rich meetings. Through web
scraping and open APIs [ 52,69,83], we collected over 40 thousand
records spanning Hansard debates, petitions, recorded votes, and
related legislative material from Canada’s parliamentary open data
sources and indexed them into a Qdrant vector database [ 93]. We
then used this indexed knowledge base to simulateInsightToast
across multiple recorded committee sessions, informing develop-
ment and serving as a live demo during our studies. The collected
dataset, together with its processed and embedded vector store
used as the knowledge base in this evaluation is open sourced at
zenodo.org/records/21502545.
Task.In each task, we asked participants to assume the role of a
citizen advisor, a lay stakeholder recommending a policy position
to inform a legislative decision, to support one of two competing
federal legislative petitions on a healthcare or housing policy issue
through deliberation with a conversation partner (a member of
the research team). We assigned one such task per condition to
mitigate learning effects, counterbalancing both the task-condition
pairing and presentation order with a Latin square yielding four
orderings assigned evenly across participants. After the delibera-
tion, we asked participants to write a justification for their decision
based on financial validity, precedents of effectiveness, and political
feasibility. This post-conversation justification served to external-
ize participants’ decision process, requiring them to deliberately
ground their recommendation in the meeting’s knowledge base.
We provide the task briefs in Appendix B.1.
Procedure.Each study session lasted approximately 80 minutes,
beginning by obtaining informed consent from all participants. We
(1)first asked participants to comment on their current practices for
collaborative decision-making, information retrieval, and discourse
tracking during meetings (5 min).(2)Next, we provided a guided
walkthrough of the system and introduced the interface through a
recording of a recent federal committee meeting on environment
and sustainable development as depicted in Figure 4.A (retrieved
from the government’s official webcasting repository). This allowed
participants to observeInsightToastas a passive meeting attendee
in a realistic data-rich meeting context (20 min).(3)Following the
demo, participants completed both study conditions: reviewing
the task cards, engaging in a 15-minute deliberation conversation,
making their decision, writing their justification, and completing
a post-task questionnaire. The conversation partner used scripted
open-ended cues (e.g. “Has any bill on national rent capping made
it through the legislative assembly?”; a list of cues appears in Ap-
pendix B.2) to build rapport and to ensure that the conversation
did not veer off-topic. The questionnaire probed the perceived in-
formation retrieval and decision-making effectiveness, integrating
questions from information retrieval [ 61] and Satisfaction with
Decision (SWD) [ 48] evaluation as well as workload [ 46], conversa-
tional engagement, and system usability.(4)The session concluded
with a semi-structured interview in which participants compared
their experience across conditions and reflected on the viability of
InsightToastfor their existing meeting practices (15 min).

UIST ’26, November 02–05, 2026, Detroit, MI, USA Mohammad Abolnejadian and Matthew Brehmer
We deployed the
 ITand
 BLinterfaces on Vercel, commu-
nicating with a backend hosted on a server where all session logs
were collected. The study was approved by University of Waterloo’s
research ethics board (approval number #47676), and participants
were remunerated with a $25 CAD multi-retailer gift card.
Participants.We recruited 16 participants (8 men, 8 women; age:
𝑀= 24.4,𝑆𝐷= 3.4, range 19–35; all fluent English speakers)
through purposive convenience sampling via departmental mail-
ing lists. Participants were predominantly graduate students (13
graduate, 3 undergraduate), with backgrounds spanning computer
science (𝑛= 11; including HCI, information visualization, usable
security, and data science) and policy and management sciences
(𝑛= 5; including public service, political science, and management
science engineering). To ensure the sample reflected a range of
prior exposure to the study’s core activities, we stratified recruit-
ment along two dimensions: familiarity with federal politics (2 very
familiar, 7 somewhat, 5 slightly, 2 not at all) and frequency of par-
ticipation in meetings involving data (7 several times per week, 8
a few times per month, 1 rarely). Eleven participants attended in
person (Figure 4.B showing the co-located setup), while five joined
remotely using screens ranging from 13–14 in ( 𝑛= 2), 15–26 in
(𝑛= 2), to 27 in or larger ( 𝑛= 1), providing variation to account for
potential effects of screen size on the interface experience.
5.1 Findings
All participants were able to complete both tasks. We performed
a thematic analysis of the sessions, organizing our findings into
three themes that interleave qualitative with quantitative results
shown in Figure 5, Figure 6, and Figure 7, following the ‘weaving’
mixed-methods reporting methodology [78].
T1: Proactive and serendipitous information retrieval results
in less task switching.As reflected in the significantly higher
ratings for perceived natural flow and engagement in the conversa-
tion (Figure 5.A), participants consistently reported that a proac-
tive delivery of content allowed them to stay more present in the
discourse, rather than diverting attention to manual information re-
trieval. This sense of flow can be attributed to the insights surfaced
in the side channel, with participants rating them favorably across
measures of timeliness, glanceability (empirically supporting the
text insight length design choice described in Section 4.1), and their
ability to help advance the conversation (Figure 7). P15 described
this as“a big game changer. . . very subtle and very seamlessly pro-
viding me insights, ”adding that they appeared“at moments of our
conversation where it felt appropriate. . . whenever there’s ambiguity
in the conversation, or when more context is needed. ”
Some participants described how the
 BLcondition forced
them into a dual-task mode, simultaneously conversing and search-
ing [P04, P13, P16], with P13 noting:“you can’t do efficient discussion,
thinking, note taking, and being in a normal mental state. . . but this
[InsightToast], you have it in front of you.”In the
 ITcondition,
participants offloaded the search process and reported lower physi-
cal demand ( 𝑀=1.75vs.𝑀=2.31;𝑝=.030), mentioning that“it was
doing the hard part of searching for information for me, and actually
use legit sources”[P07]. This pivot in search behavior is evident in
fewer reactive searches (Figure 6.B), with P01 stating that“I didn’t
have to think of a question to ask”with proactive retrieval.In the
 BLcondition, the struggle of search began even be-
fore typing a query, with the participants consistently identifying
crafting a search term as a central barrier to effective informa-
tion retrieval, illustrated by remarks such as“I’m struggling to
find keywords to search”[P05],“am I searching up the right key-
words?”[P15],“I don’t know how to even ask[chatbot]and formulate
my prompt”[P06]. Compounding this was the challenge of deciding
whichsource would be of most help between the chatbot, Google,
and the knowledge base search. Some participants resolved to only
query the chatbot [P07, P08, P09, P14], albeit with noted reserva-
tions about trust, as P06 put it:“My justification is just the[chatbot].
I don’t trust that, but I have to go with this. ”This cumulative cost of
finding information in the baseline was reflected in significantly
higher effort and frustration (Figure 5.B), often making participants
abandon or delay search altogether, leading to a“huge break in the
continuity of the conversation. ”[P12]. In contrast, P09 noted that in
the
 ITcondition,“we can actually discuss it in the meeting, and
just not postpone it to the next meeting, after I’ve done the research. ”
Beyond changing post-meeting search behavior, proactive de-
livery also reshaped how participants spent their time within a
session. As illustrated by Figure 6.A, a pattern emerged across both
conditions: the first ∼60%of session time was dominated by ex-
ploration, while the remaining ∼40%shifted toward deliberation,
with these phases manifesting differently across conditions. In the
exploration phase, behavior in the
 BLcondition followed apause–
search–resumecycle in which participants repeatedly halted the
conversation to formulate queries and scan documents, producing
“long gaps”[P16], whereas in the
 ITcondition, exploration hap-
pened mostly through interactions with proactively synthesized
insights, which P16 described as allowing ideas to“build layer on
layer. ”The deliberation phase also differed, with participants largely
moving off the main collaboration interface to read opened docu-
ment tabs in the
 BLcondition: Figure 6.A-bottom shows long
intervals containing no interaction with the system. In the
 IT
condition, participants remained within the conversation interface
and engaged with citations directly.
T2: Deliberation broadens and decisions feel more informed.
Participants in the
 ITcondition reported feeling significantly
more informed about their decisions ( 𝑀=4.06vs.𝑀=2.62;𝑝=.004;
Figure 5.B). P13 contrasted the two experiences, describing how
the“first decision[
 BL]was mostly based on my perception and
my thoughts, not really grounded . . . while the second decision [
IT] was explained by external studies or sources of laws. ”To examine
whether this subjective sense of being more informed translated to
observable differences, we assessed participants’ post-task written
justifications along three axes inspired by the Toulmin Argument
Model [ 113]:claims(distinct arguments made),evidence(factual
support for claims), andcitations(references to specific sources). To
complete this assessment, one member of the research team along
with two LLM-as-judges [ 39,131] (GPT-5.4 and Claude Opus 4.6)
coded participants’ justifications with a high degrees of inter-rater
reliability (Krippendorff’s 𝛼=0.861). We observed that the number
of claims (𝑀=3.50vs.𝑀=3.88) and evidence pieces ( 𝑀=2.50vs.
𝑀=1.94) did not differ significantly between conditions, indicating
that participants constructed a similar number of arguments backed
by a comparable amount of factual support. However, participants

InsightToast UIST ’26, November 02–05, 2026, Detroit, MI, USA
1 2 3 4 5IT
BL
IT
BL
IT
BL
IT
BL
IT
BL*
**
*
ns
**Information Retrieval & Decision Making
Effectively accessed relevant data
Considered all important aspects
Data broadened my perspective
Confident in my decision
Made an informed decision
1 2 3 4 5IT
BL
IT
BL
IT
BL
IT
BL
IT
BLns
*
**
*
*TLX
Mental
Physical
Temporal
Effort
Frustration
1 2 3 4 5IT
BL
IT
BL
IT
BL
IT
BL
IT
BL*
*
ns
*
*Conversation Engagement
Information seeking disrupted flow (R)
Conversation flowed naturally
Easily tracked discussion points
Felt fully engaged in conversation
Enjoyed the overall experience
Figure 5: Questionnaires in our study asked participants to compare
 Insight Toast(IT) and
 BaseLine(BL) interfaces with
respect to information retrieval [61], decision satisfaction [48], workload [46], engagement, and usability.
ITInteraction Timeline
0 20 40 60 80 100
Session Progress (%)BL1.97.4
IT BLReactive Searches
Insight Engagement Citation Engagement Topic Browsing Knowledge Base Search Chatbot Web Search Navigation/UI
Figure 6: (A) Aggregated interaction frequencies and types
are compared between the
 ITand
 BLinterfaces across all
study sessions. (B) Overall, participants performed far fewer
reactive searches (knowledge base search, chatbot interac-
tion, and Google search) with
 ITrelative to with
 BL.
in the
 ITcondition included significantly more source citations
in their justifications ( 𝑀=2.31vs.𝑀=1.25;𝑝=.006), suggesting that
they found proactively-retrieved sources relevant enough to cite
in support of their arguments. P02 qualifies this observation by
stating:“I felt like I could cite more to say why I thought certain
things. Whereas in the first one[
 BL], I felt more like I was just
saying things that I believed, but I couldn’t back it up. ”This source
traceability corresponded with feeling significantly more informed,
with P07 remarking how they felt“more informed with the AI system
[
IT], specifically because I had more sources to back my point. ”The
comparable volume of evidence across conditions appears to have
provided sufficient justification for participants’ reasoning, yielding
no significant difference in decision confidence. Ignorance, it would
seem, was bliss in the
 BLcondition, where participants could
construct plausible arguments and feel reasonably confident.
Participants in both conditions came to the conversation with
pre-existing views on the policy topics, and several described how,
withoutInsightToast, they would have searched only for informa-
tion confirming their existing position. P09 explained,“I was kind of
biased to the other position. . . I believe I would ask[chatbot]questions
more toward that position, and do not even think about the other po-
sition, ”and P12, who described themselves as“a pretty opinionated
person, ”noted that“the insights were like a good, factually grounded
way for me to consider other opinions, ”. P01 spoke to the root of
this mitigation, stating that they were getting“more perspectives”
because“it[InsightToast]told me about both. ”This broadening of
perspectives was also reflected in questionnaire results, (𝑀=4.44
vs.𝑀= 3.81;𝑝=. 048) with participants reporting a markedly
greater sense of having considered all important aspects ( 𝑀= 3.44
vs.𝑀= 2.19;𝑝=. 003). These broader perspectives often surfaced
as unexpected surprises, presenting information that participantswould not have thought to seek on their own. P04, for instance, de-
scribed encountering a specific policy detail that was“not something
I would have even considered, ”noting that it“made me completely
recontextualize my thoughts, ”while P07 reported shifting positions
entirely after encountering unanticipated evidence, reflecting that
without it, they would“probably have dug my heels into position
B. ”Crucially, participants perceived this broadening as expanding
rather than constraining their agency (Figure 7), with P06 valu-
ing that the system“was not giving me a direct answer, but it was
supporting my thoughts. ”This sense of support over substitution
also fostered trust, particularly through grounded citations, as P12,
who described having“very low trust towards AI systems in general, ”
noted that“the fact that it generated these insights and gave me direct
[policy]sources, I felt like I was able to trust those insights. ”
T3: Adapting to proactive meeting intelligence elicits new
usage scenarios.Participants reported significantly higher enjoy-
ment in the
 ITcondition (Figure 5.C), and rated the interface
favorably across its evaluation dimensions. Participants specifically
valued the titled summaries that enabled quick scanning, mention-
ing that“when I was searching, I could just skim based on titles, ”[P07],
and the clickable citations that provided transparent sourcing, as
they could“click the button and immediately see what it’s talking”
[P02]. Reactions to chart-based insights, however, were mixed, with
participants who described themselves as "visual learners" [P06,
P13, P15] finding them most useful, stating that“the charts were
really helpful for me to understand, ”[P06] and appreciating that they
“helped really quickly visualize stats, ”[P04], while some“preferred the
text insight generations over the charts, because they were smaller and
digestible”[P07], consistent with prior findings on the efficiency of
text for information extraction [ 67]. Participants who described how
demanding gathering data could be in their current meeting prac-
tices on the pre-study questionnaire were particularly enthusiastic
about proactive meeting intelligence. Correlating responses from
the pre-study questionnaire against participant’s gain from
 BL
to
 ITconditions, we found significant positive correlations with
perceived advantages in information retrieval and decision-making
(Spearman𝜌=0.50,𝑝=.049) and engagement ( 𝜌=0.61,𝑝=.012), as
well as a trending reduction in workload ( 𝜌=− 0.47,𝑝=.069), indicat-
ing thatInsightToastespecially benefits those with high search
overhead in their current practices.
Participants were enthusiastic about the prospect of using ap-
plications likeInsightToastin future meetings (Figure 7). Even

UIST ’26, November 02–05, 2026, Detroit, MI, USA Mohammad Abolnejadian and Matthew Brehmer
P07, who expressed concerns about AI’s environmental impact, ac-
knowledged that“this seems like a really good place”for AI, as it is
“not really replacing the creative aspect of being a human”but rather
“replacing some of the grunt work. ”Participants connected the system
to diverse knowledge bases suited to their own contexts, including
academic research literature [P05, P07, P12], cross-domain collab-
oration where knowledge gaps are known to exist [P06], policy
analysis with government archives [P15, P16], and even casual set-
tings like vacation planning [P02]. Notably, P10 described a scenario
of missing institutional context within an organization: after senior
team members depart and juniors“reinvent the wheel”because they
are unaware of past activities documented in an organizational
knowledge base; they envisionInsightToastas a bridge to this
institutional knowledge.
Participants also described how they adapted toInsightToast,
with P08 stating“it’s like getting a phone for the first time, ”while P05
observed it was“a lot to get started with, but just like any software, I
think I will get comfortable using it, ”mentioningInsightToast’s
novelty effect. This feeling of adaptation is notable given the in-
formation density of the 15-minute task durations across study
sessions, with an average of28 .53insights ( 𝑆𝐷= 8.58) generated
over each session (≈one insight every 30s). However, this density
of proactively delivered content also introduced its own disruptions,
with P04 describing how their“train of thought was hijacked”by
insight toasts, P03 noting they“had to sift through a lot more noise
compared to manually searching, ”and P08 pointing to“the cogni-
tive load of having a conversation while also deciphering what these
insights meant. ”Beyond adaptingtothe system, participants also
envisioned how the system could adapttothem. A recurring desire
was reactive insight generation alongside the proactive mechanism,
using proactivity for breadth while allowing targeted depth on
demand: P03 wanted“proactive themes, but the insights that I’m
looking for within the theme, I would want more agency . . . to manu-
ally do it. ”However, P04 indicated that reactive insight generation
does not necessarily mean chatbot interaction, but rather“a button
in the transcript where I could select[an utterance]and say, generate
me insights on this thing.”Several participants also identified the
potential value for longer meeting sessions, where they expected
the topic tracker to be more useful [P03, P08, P11], and in meet-
ings with more participants, where one can read insights while
others speak [P04, P05]. Beyond usingInsightToastduring con-
versations, participants also saw value in retrospective engagement.
P04, who found real-time interaction overwhelming, would instead
“turn it on, let it run,[and]review it after the meeting, ”fully defer-
ring engagement with synthesized content to post-meeting, while
P11 would“dive deeper into them[insights]later”, only skimming
during the meeting. P10 suggested pinning as a mechanism for
more efficient use of insights after the meeting and envisioned a
“conversation after the conversation with the[retrieved]documents. ”
5.2 Technical Evaluation
To assess the performance ofInsightToast’s pipeline, we analyzed
processing logs from all 16 study sessions along three dimensions:
(i) processing latency:Each 15-minute session processed an average
of 113 transcribed utterances ( 𝑆𝐷= 18), with a mean end-to-end
pipeline latency of7 .5𝑠per chunk ( 𝑆𝐷= 13.8s). Utterances that
1 1 6 8Insights were relevant to our discussion
1 1 2 8 4Insights appeared at the right moments
4 1 4 7Insights were easy to read at a glance
2 3 11Insights helped advance the conversation
2 2 12Helped discover unexpected information
1 6 9Supported decisions without overriding agency
1 3 4 8Would use this system in future meetings
Strongly disagree Somewhat disagree Neither Somewhat agree Strongly agreeFigure 7: Participants’ agreement with statements regarding
the perceived utility ofInsightToast(𝑁=16).
triggered insight synthesis required substantially more process-
ing (𝑀=29.1s,𝑆𝐷= 22.0s) compared to those that resolved without
synthesis (𝑀=3.4s,𝑆𝐷= 5.3s).(ii) LLM inference cost:Each session
consumed approximately2 .17M input (𝑆𝐷= 0.54M) and141K output
(𝑆𝐷= 42K) tokens across all pipeline agents. At the time of writing,
Gemini Flash, used for chart insight agents, is priced at $0.50/$3.00
per 1M input/output tokens, while Gemini Flash Lite, used for the
remaining pipeline agents (see Section 4.3), is priced at $0.10/$0.40,
resulting in a mean LLM inference cost of $0.45 per session.(iii)
proactive retrieval quality:To assess the accuracy of knowledge gap
detection that triggers retrieval, a member of the research team
annotated whether each utterance across all sessions warranted
retrieval. We compared these labels against the system’s binary clas-
sification, yielding a mean 𝐹1=0.81(precision =0.70, recall =0.99),
indicating the system rarely missed genuine information needs,
with occasional over-triggering at adjacent turns.
6 Discussion
We reflect onInsightToast’s design and evaluation, the limitations
of our research, and future work in proactive meeting intelligence.
Reflections.Although proactive retrieval reduced task switching
by delegating query formulation and evidence synthesis to the
system and reducing the temporal and spatial separation between
communication and information discovery, some participants char-
acterized the frequency of toast notifications asnoise. We attribute
this to occasional over-triggering in the pipeline and high informa-
tion density in the session. While this relocation of task switching
was unfamiliar early in the study sessions, participants appeared
to grow more comfortable identifying relevant content as they
adapted to the interface. Rather than relying on such adaptation
alone, a more granular model of notification criticality that goes
beyond knowledge-gap detection could inform more intelligent
interruption decisions [ 2]. By attaching levels of criticality to toasts,
accounting for conversational state and task progress, the system
could allow participants to tune the notification threshold through
meeting settings, balancing the exploration and exploitation [ 11] for

InsightToast UIST ’26, November 02–05, 2026, Detroit, MI, USA
the meeting’s information needs. For instance, participants could
opt to mute new notifications during retrospective meeting review.
Beyond calibratingwheninsights surface, our findings also point
to refiningwhattriggers their generation. While ambient discourse
currently serves as the exclusive input modality initiating insight
synthesis and implicitly capturing participant feedback, direct voice
commands, gestural cues [ 13], and inline UI controls could add reac-
tive channels for on-demand generation and for reporting failures
such as irrelevant or erroneous insights, shifting toward higher
levels of control in the mixed-initiative design spectrum [98]. Our
study participants valued insights not merely for being proactive
but for their context-aware and digestible form that eliminated
the burden of formulating search queries, suggesting that reac-
tive mechanisms could preserve this quality while offering targeted
depth on demand. Context detection could also be refined by attend-
ing to deictic expressions and referential cues in speech [ 107], such
as demonstrative references to entities or concepts under discus-
sion, using these signals alongside speech utterances to better scope
what the system retrieves. More broadly, whileInsightToastcur-
rently establishes its meeting and knowledge base context through
explicit configuration in its codebase and implicit inference from its
running memory of the discourse, richer pre-meeting specifications,
including agendas and a priori meeting goals [ 19], could further
steer both retrieval and synthesis.
Limitations.On the technical side, the mean synthesis latency of 29
seconds per insight-triggering utterance risks brief conversational
disruptions as participants pause to await the results. Addition-
ally, the inference cost, measured at approximately half a dollar
per fifteen-minute session, may limit broader adoption. Adapting
open-weight models hosted on internal servers within adopting
organizations presents one path forward, though this introduces a
quality-cost trade-off, as smaller models may degrade the pipeline’s
quality, while also addressing within-organization data privacy
requirements, pointing toInsightToast’s potential adaptability
to domains involving sensitive data. Moreover, as with any sys-
tem reliant on LLMs, bias and hallucination persist even under
retrieval grounding [ 53,103]. WhileInsightToasttempers such
risks through multi-faceted retrieval and traceable citations that in-
vite verification, disclosing this fallibility to participants alongside
adopting more reliable models can serve as additional safeguards.
Methodologically, our evaluation involved trade-offs for a within-
subjects contrast within a single study session. While this setup
provided an initial demonstration ofInsightToast’s capabilities,
it does not capture the dynamics of larger groups deliberating over
longer periods with multiple subject-matter experts.
Future Work.Our methodological limitations call for further eval-
uation with varying group sizes, conversation topics, and meeting
durations. Data-rich collaborative tasks beyond decision-making
are also exciting directions to consider [ 15], such as group brain-
storming and real-time data monitoring. Beyond limitations, the
extension of applications likeInsightToastinto specialized appli-
cation domains suggests new opportunities for proactive meeting
information retrieval. For instance, conversations that are concur-
rent with collaborative interaction around maps and Geographic In-
formation System (GIS) interfaces could support discussions among
urban planners. Similarly, multimodal discussion and interactionwith medical imagery and electronic health records (EHRs) could
surface clinically-relevant insights. In either setting, the delivery
of notifications may need to be adapted to appear adjacent to or
near the object, image, or map location being manipulated, and the
palette of chart insights will likely need to be expanded beyond
quantitative and temporal attributes. Beyond charts, other insight
modalities merit further investigation in accordance with their
application contexts. In meeting settings, while our findings sup-
ported the glanceability ofInsightToast’s tweet-length snippet
cap, a focused study of text length under the constrained conversa-
tional bandwidth of live discourse could better inform the design
of informative yet non-disruptive of insights in text insights.
More explicit configurability could better align proactive sup-
port with participants’ desired degree of control over serendipitous
information discovery. Promising dimensions include tunable proac-
tivity levels, alternative interruption and triggering mechanisms,
and assignable meeting roles, as participants who differ in data
access privileges and domain knowledge may hold asymmetric
information needs for maintaining common ground. We envision
a system likeInsightToast, adaptable across meeting modalities
and generalizable through the substitution of its indexed knowl-
edge base, being adopted across diverse settings, from corporate
boardrooms to academic advisory meetings, where surfacing the
right data at the right moment is critical.
7 Conclusion
We contributed an approach to proactive information delivery for
synchronous meetings, one with the aim of filling gaps in knowl-
edge to ground the conversation in data. We encapsulated our ap-
proach in a meeting application calledInsightToast, suitable for
both co-located and remote/hybrid meetings; as its name implies,
it delivers ephemeral toast notifications in participants’ periphery
at opportune moments in a conversation. These notifications are
(ideally) insightful, summarizing content primarily from an insti-
tutional knowledge base and secondarily from the open web, in
the form of either pithy text statements or glanceable charts, with
dual attribution to the conversation and to source documents. We
evaluatedInsightToastin a study ( 𝑁= 16) where we contrasted
the experience of using it in a meeting emulating a legislative
decision-making scenario with a baseline reactive search inter-
face. Our findings suggest that participants were quite receptive to
proactive meeting assistance, especially for identifyingunknown
unknownsand serendipitously surfacing content that would oth-
erwise have gone amiss. Ultimately, this work is an initial step
toward generalized proactive support for the collaborative analysis
of heterogeneous data types and the integration of this support
into the longitudinal fabric of collaborative knowledge work.
Acknowledgments
We thank S. Onay, D. Vogel, A. Crisan, S. Amirshahi, and members
of the CS HCI Lab and UBIX research group at U. Waterloo.
References
[1]Mohammad Abolnejadian, Shakiba Amirshahi, Matthew Brehmer, and Anamaria
Crisan. 2025. AInsight: Augmenting expert decision-making with on-the-fly
insights grounded in historical data. InACM Conf. Conversational User Interfaces
(CUI). ACM, New York, NY, USA, 1–7. doi:10.1145/3719160.3737633

UIST ’26, November 02–05, 2026, Detroit, MI, USA Mohammad Abolnejadian and Matthew Brehmer
[2]Piotr D Adamczyk and Brian P Bailey. 2004. If not now, when?: the effects of
interruption at different moments within task execution. InACM Conf. Human
Factors in Computing Systems (CHI). ACM, 271–278. doi:10.1145/985692.985727
[3]Segun Adebayo. 2024. Chakra UI: A Simple, Modular and Accessible Component
Library for React. https://chakra-ui.com. Accessed: 2024.
[4]Christopher Ahlberg and Ben Shneiderman. 1994. Visual information seeking:
tight coupling of dynamic query filters with starfield displays. InACM Conf.
Human Factors in Computing Systems (CHI). ACM Press, New York, New York,
USA, 313–317. doi:10.1145/191666.191775
[5]Salvatore Andolina, Valeria Orso, Hendrik Schneider, Khalil Klouche, Tuukka
Ruotsalo, Luciano Gamberini, and Giulio Jacucci. 2018. Investigating proactive
search support in conversations. InACM Conf. Designing Interactive Systems
(DIS). ACM, New York, NY, USA, 1141–1152. doi:10.1145/3196709.3196734
[6]Apple Inc. 2026. Use Notification Center on Mac. Apple Support – Mac User
Guide. Accessed: 2026-03-30. https://support.apple.com/en/guide/mac-help/
mchl2fb1258f/mac
[7]Bon Adriel Aseniero, Marios Constantinides, Sagar Joglekar, Ke Zhou, and
Daniele Quercia. 2020. MeetCues: Supporting online meetings experience. In
IEEE Proc. Visualization & Visual Analytics (VIS). 236–240. doi:10.1109/VIS47514.
2020.00054
[8]Brian P Bailey and Joseph A Konstan. 2006. On the need for attention-aware
systems: Measuring effects of interruption on task performance, error rate,
and affective state.Computers in Human Behavior22, 4 (July 2006), 685–708.
doi:10.1016/j.chb.2005.12.009
[9]G Baker. 2002. The effects of synchronous collaborative technologies on decision
making: A study of virtual teams.Inf. Resour. Manag. J.15, 4 (Oct. 2002), 79–93.
doi:10.4018/irmj.2002100106
[10] Nicholas J Belkin. 1980. Anomalous states of knowledge as a basis for infor-
mation retrieval.Canadian Journal of Information Science5, 1 (May 1980),
133–143.
[11] Oded Berger-Tal, Jonathan Nathan, Ehud Meron, and David Saltz. 2014. The
exploration-exploitation dilemma: a multidisciplinary framework.PLoS ONE9,
4 (April 2014), e95693. doi:10.1371/journal.pone.0095693
[12] Tanja Blascheck, Lonni Besancon, Anastasia Bezerianos, Bongshin Lee, and
Petra Isenberg. 2019. Glanceable Visualization: Studies of Data Comparison
Performance on Smartwatches.IEEE Trans. Visualization & Computer Graphics
(TVCG)25, 1 (Jan. 2019), 630–640. doi:10.1109/TVCG.2018.2865142
[13] Richard A Bolt. 1980. “Put-that-there”: Voice and gesture at the graphics inter-
face. InACM Conf. Computer Graphics and Interactive Techniques (SIGGRAPH).
ACM, 262–270. doi:10.1145/800250.807503
[14] Gabriella Bonic. 2025. The Proactive AI Revolution: How Cluely Signals a New
Era. https://gabriellabonic.substack.com/p/the-proactive-ai-revolution-how-
cluely. Accessed: 2026-3-15.
[15] Matthew Brehmer, Maxime Cordeil, Christophe Hurter, Takayuki Itoh, Wolfgang
Büschel, Mahmood Jasim, Arnaud Prouzeau, David Saffo, Lyn Bartram, Sheelagh
Carpendale, Chen Zhu-Tian, Andrew Cunningham, Tim Dwyer, Samuel Huron,
Masahiko Itoh, Alark Joshi, Kiyoshi Kiyokawa, Hideaki Kuzuoka, Bongshin Lee,
Gabriela Molina León, Harald Reiterer, Bektur Ryskeldiev, Jonathan Schwabish,
Brian A Smith, Yasuyuki Sumi, Ryo Suzuki, Anthony Tang, Yalong Yang, and
Jian Zhao. 2026. Challenges in synchronous & remote collaboration around
visualization. InACM Conf. Human Factors in Computing Systems (CHI). 1–17.
doi:10.1145/3772318.3791117
[16] Matthew Brehmer and Robert Kosara. 2022. From jam session to recital: Syn-
chronous communication and collaboration around data in organizations.IEEE
Trans. Visualization & Computer Graphics (TVCG)28, 1 (Jan. 2022), 1139–1149.
doi:10.1109/tvcg.2021.3114760
[17] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan,
Rewon Child, Aditya Ramesh, Daniel M Ziegler, Jeffrey Wu, Clemens Winter,
Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin
Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya
Sutskever, and Dario Amodei. 2020. Language Models are Few-Shot Learners.
InConf. Neural Information Processing Systems (NeurIPS), Vol. 33. 1877–1901.
[18] Shubham Chatterjee, Xi Wang, Shuo Zhang, Sajad Ebrahimi, Zhaochun Ren,
Debasis Ganguly, Gareth J F Jones, Emine Yilmaz, and Hamed Zamani. 2025.
ProActLLM: Proactive conversational information seeking with large language
models. InACM Intl. Conf. Information and Knowledge Management (CIKM).
6894–6897. doi:10.1145/3746252.3761593
[19] Xinyue Chen, Lev Tankelevitch, Rishi Vanukuru, Ava Elizabeth Scott, Payod
Panda, and Sean Rintel. 2025. Are we on track? AI-assisted active and passive
goal reflection during meetings. InACM Conf. Human Factors in Computing
Systems (CHI). ACM, New York, NY, USA, 1–22. doi:10.1145/3706598.3714052
[20] Charles L Citroen. 2011. The role of information in strategic decision-making.
Int. J. Inf. Manage.31, 6 (Dec. 2011), 493–501. doi:10.1016/j.ijinfomgt.2011.02.005
[21] Herbert H Clark and Susan E Brennan. 1991. Grounding in communication. In
Perspectives on Socially Shared Cognition, Lauren B Resnick, John M Levine, and
Stephanie D Teasley (Eds.). American Psychological Association, Washington,DC, 127–149. doi:10.1037/10096-006
[22] Cluely. 2026. Cluely - Live AI Meeting Assistant. https://cluely.com/. Accessed:
2026-3-15.
[23] Sharon Cogdill, Tari Lin Fanderclai, Judith Kilborn, and Marian G Williams.
2001. Backchannel: whispering in digital conversation. InHawaii Intl. Conf. on
Systems Science (HICSS). doi:10.1109/HICSS.2001.926500
[24] Mark G Core and James F Allen. 1997. Coding Dialogs with the DAMSL Anno-
tation Scheme. InWorking Notes of the AAAI Fall Symposium on Communicative
Action in Humans and Machines. Cambridge, MA, 28–35.
[25] Scott Counts and Kristie Fisher. 2011. Taking It All In? Visual Attention in
Microblog Consumption.Proc. Intl. AAAI Conf. Web and Social Media (ICWSM)
5, 1 (2011), 97–104. doi:10.1609/icwsm.v5i1.14103
[26] Ross Cutler, Yasaman Hosseinkashi, Jamie Pool, Senja Filipi, Robert Aichner,
Yuan Tu, and Johannes Gehrke. 2021. Meeting effectiveness and inclusiveness
in remote collaboration.Proc. ACM on Human-Computer Interaction (PACM HCI)
5, CSCW1 (April 2021), 1–29. doi:10.1145/3449247
[27] Jeffrey Dean and Sanjay Ghemawat. 2008. MapReduce: Simplified Data
Processing on Large Clusters.Comm. ACM (CACM)51, 1 (2008), 107–113.
doi:10.1145/1327452.1327492
[28] Evan DeFilippis, Stephen Michael Impink, Madison Singell, Jeffrey T Polzer,
and Raffaella Sadun. 2022. The impact of COVID-19 on digital communication
patterns.Humanit. Soc. Sci. Commun.9, 1 (May 2022), 180. doi:10.1057/s41599-
022-01190-9
[29] Evanthia Dimara, Anastasia Bezerianos, and Pierre Dragicevic. 2018. Conceptual
and methodological issues in evaluating multidimensional visualizations for
decision support.IEEE Trans. Visualization & Computer Graphics (TVCG)24, 1
(Jan. 2018), 749–759. doi:10.1109/tvcg.2017.2745138
[30] Evanthia Dimara and John Stasko. 2022. A critical reflection on visualization
research: Where do decision making tasks hide?IEEE Trans. Visualization &
Computer Graphics (TVCG)28, 1 (Jan. 2022), 1128–1138. doi:10.1109/tvcg.2021.
3114813
[31] Evanthia Dimara, Harry Zhang, Melanie Tory, and Steven Franconeri. 2022. The
unmet data visualization needs of decision makers within organizations.IEEE
Trans. Visualization & Computer Graphics (TVCG)28, 12 (Dec. 2022), 4101–4112.
doi:10.1109/tvcg.2021.3074023
[32] Marian Dörk, Daniel Gruen, Carey Williamson, and Sheelagh Carpendale. 2010.
A visual backchannel for large-scale events.IEEE Trans. Visualization & Com-
puter Graphics (TVCG)16, 6 (Nov. 2010), 1129–1138. doi:10.1109/TVCG.2010.129
[33] Sara Doubleday. 2023. What is Context Switching and How Does it Impact My
Team? https://www.seerinteractive.com/insights/context-switching-impact-
team. Accessed: 2025-12-30.
[34] Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mor-
datch. 2024. Improving Factuality and Reasoning in Language Models through
Multiagent Debate. InIntl. Conf. Machine Learning (ICML), Vol. 235. PMLR,
11733–11763.
[35] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A graph RAG approach to query-
focused summarization.arXiv [cs.CL](April 2024). arXiv:2404.16130 [cs.CL]
doi:10.48550/arXiv.2404.16130
[36] Clarence A Ellis, Simon J Gibbs, and Gail Rein. 1991. Groupware: some issues
and experiences.Comm. ACM (CACM)34, 1 (Jan. 1991), 39–58. doi:10.1145/
99977.99987
[37] Maryia Fokina. 2023. When Machines Dream: A Dive in AI Hallucinations
[Study]. https://www.tidio.com/blog/ai-hallucinations/. Accessed: 2025-12-30.
[38] G W Furnas, T K Landauer, L M Gomez, and S T Dumais. 1987. The vocabulary
problem in human-system communication.Comm. ACM (CACM)30, 11 (Nov.
1987), 964–971. doi:10.1145/32206.32212
[39] Jie Gao, Yuchen Guo, Gionnieve Lim, Tianqin Zhang, Zheng Zhang, Toby Jia-
Jun Li, and Simon Tangi Perrault. 2024. CollabCoder: A lower-barrier, rigorous
workflow for inductive collaborative qualitative analysis with large language
models. InACM Conf. Human Factors in Computing Systems (CHI). ACM, 1–29.
doi:10.1145/3613904.3642002
[40] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise Zero-Shot
Dense Retrieval without Relevance Labels. InAnnual Meeting of the Association
for Computational Linguistics (ACL). Toronto, Canada, 1762–1777. doi:10.18653/
v1/2023.acl-long.99
[41] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi
Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2023. Retrieval-Augmented
Generation for Large Language Models: A Survey.arXiv [cs.CL](Dec. 2023).
arXiv:2312.10997 [cs.CL] doi:10.48550/arXiv.2312.10997
[42] Google. 2026. AI for Meetings & Video Conferencing. https://workspace.google.
com/intl/en/resources/ai-for-meetings/. Accessed: 2026-3-15.
[43] Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V
Chawla, Olaf Wiest, and Xiangliang Zhang. 2024. Large Language Model based
Multi-Agents: A Survey of Progress and Challenges. InIntl. Joint Conf. Artificial
Intelligence (IJCAI), Survey Track. 8048–8057. doi:10.24963/ijcai.2024/890

InsightToast UIST ’26, November 02–05, 2026, Detroit, MI, USA
[44] Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu
Su. 2024. HippoRAG: Neurobiologically inspired long-term memory for large
language models. InConf. Neural Information Processing Systems (NeurIPS),
Vol. 37. 59532–59569.
[45] Carl Gutwin and Saul Greenberg. 2002. A descriptive framework of workspace
awareness for real-time groupware.Comput. Support. Coop. Work11, 3-4 (Sept.
2002), 411–446. doi:10.1023/A:1021271517844
[46] Sandra G Hart and Lowell E Staveland. 1988. Development of NASA-TLX (task
load index): Results of empirical and theoretical research. InHuman Mental
Workload. Advances in Psychology, Vol. 52. Elsevier, 139–183. doi:10.1016/s0166-
4115(08)62386-9
[47] Chen He, Luana Micallef, Barış Serim, Tung Vuong, Tuukka Ruotsalo, and Giulio
Jacucci. 2021. Interactive visual facets to support fluid exploratory search. In
Intl. Symp. Visual Information Communication and Interaction (VINCI). ACM,
1–10. doi:10.1145/3481549.3481565
[48] M Holmes-Rovner, J Kroll, N Schmitt, D R Rovner, M L Breer, M L Rothert, G
Padonu, and G Talarczyk. 1996. Patient satisfaction with health care decisions:
the satisfaction with decision scale.Med. Decis. Making16, 1 (Jan. 1996), 58–64.
doi:10.1177/0272989x9601600114
[49] Sirui Hong, Mingchen Zhuge, Jiaqi Chen, Xiawu Zheng, Yuheng Cheng, Ceyao
Zhang, Jinlin Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou,
Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. 2024.
MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework. In
Intl. Conf. Learning Representations (ICLR). https://openreview.net/forum?id=
VtmBAGCN7o
[50] Eric Horvitz. 1999. Principles of mixed-initiative user interfaces. InACM Conf.
Human Factors in Computing Systems (CHI). doi:10.1145/302979.303030
[51] Yasaman Hosseinkashi, Lev Tankelevitch, Jamie Pool, Ross Cutler, and Chinmaya
Madan. 2024. Meeting effectiveness and inclusiveness: Large-scale measurement,
identification of key features, and prediction in real-world remote meetings.
Proc. ACM on Human-Computer Interaction (PACM HCI)8, CSCW1 (April 2024),
1–39. doi:10.1145/3637370
[52] House of Commons of Canada. 2026. House of Commons of Canada — Offi-
cial Website. https://www.ourcommons.ca. Order papers, journals, committee
reports, and petitions. Crown copyright; reproduced under the Speaker’s Per-
mission. Accessed 2026-07-23.
[53] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian
Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu.
2025. A survey on hallucination in large language models: Principles, taxonomy,
challenges, and open questions.ACM Trans. Information Systems (TOIS)43, 2
(Jan. 2025), 1–55. doi:10.1145/3703155
[54] IBM Carbon Design System. 2023. Notification – Usage. https://
carbondesignsystem.com/components/notification/usage/. Accessed: 2026-03-
30.
[55] Ohad Inbar, Noam Tractinsky, and Joachim Meyer. 2007. Minimalism in In-
formation Visualization: Attitudes towards Maximizing the Data-Ink Ratio.
InEuropean Conf. Cognitive Ergonomics (ECCE). ACM, New York, NY, USA,
185–188. doi:10.1145/1362550.1362587
[56] Mahmood Jasim, Pooya Khaloo, Somin Wadhwa, Amy X Zhang, Ali Sarvghad,
and Narges Mahyar. 2021. CommunityClick: Capturing and reporting commu-
nity feedback from town halls to improve inclusivity.Proc. ACM on Human-
Computer Interaction (PACM HCI)4, CSCW3 (Jan. 2021), 1–32. doi:10.1145/
3432912
[57] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park.
2024. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
Models through Question Complexity. InConf. North American Chapter of the
Association for Computational Linguistics (NAACL). Mexico City, Mexico, 7036–
7050. doi:10.18653/v1/2024.naacl-long.389
[58] Robert Johansen, Jeff Charles, Robert Mittman, and Paul Saffo. 1988.Groupware:
Computer Support for Business Teams. Free Press, New York.
[59] Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly,
and Cyrus Rashtchian. 2025. Sufficient Context: A New Lens on Retrieval
Augmented Generation Systems. InIntl. Conf. Learning Representations (ICLR).
https://arxiv.org/abs/2411.06037
[60] Daniel Kahneman. 2011.Thinking, Fast and Slow. Farrar, Straus and Giroux.
[61] Diane Kelly. 2009. Methods for evaluating interactive information retrieval
systems with users.Found. Trends Inf. Retr.3, 1-2 (April 2009), 1–224. doi:10.
1561/1500000012
[62] Hyeok Kim, Arjun Srinivasan, and Matthew Brehmer. 2024. Bringing data into
the conversation: Adapting content from business intelligence dashboards for
threaded collaboration platforms. InIEEE Proc. Visualization & Visual Analytics
(VIS). 81–85. doi:10.1109/VIS55277.2024.00024
[63] Joseph Kim and Julie A Shah. 2016. Improving team’s consistency of under-
standing in meetings.IEEE Trans. Hum. Mach. Syst.46, 5 (Oct. 2016), 625–637.
doi:10.1109/thms.2016.2547186
[64] Denis Kotkov, Alan Medlar, and Dorota Glowacka. 2023. Rethinking Serendipity
in Recommender Systems. InACM Conf. Human Information Interaction and
Retrieval (CHIIR). ACM, 383–387. doi:10.1145/3576840.3578310[65] George Lakoff. 1973. Hedges: A Study in Meaning Criteria and the Logic of Fuzzy
Concepts.J. Philos. Logic2, 4 (Oct. 1973), 458–508. doi:10.1007/BF00262952
[66] LangChain AI. 2024. LangGraph: A Library for Building Stateful, Multi-Actor
Applications with LLMs. https://github.com/langchain-ai/langgraph. Accessed:
2024.
[67] Bongshin Lee, Raimund Dachselt, Petra Isenberg, and Eun Kyoung Choe (Eds.).
2021.Mobile Data Visualization. CRC Press, Boca Raton, FL. doi:10.1201/
9781003090823
[68] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-Tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. InConf. Neural Information
Processing Systems (NeurIPS), Vol. 33. 9459–9474.
[69] Library of Parliament, Canada. 2026. Library of Parliament Catalogue and
Sessional Papers. https://parl-gc.primo.exlibrisgroup.com. Accessed 2026-07-
23.
[70] Hao Liu, Zhengren Wang, Xi Chen, Zhiyu Li, Feiyu Xiong, Qinhan Yu, and
Wentao Zhang. 2025. HopRAG: Multi-hop reasoning for logic-aware retrieval-
Augmented Generation. InFindings of the Association for Computational Lin-
guistics (ACL). 1897–1913. doi:10.18653/v1/2025.findings-acl.97
[71] Xingyu “Bruce” Liu, Vladimir Kirilyuk, Xiuxiu Yuan, Alex Olwal, Peggy Chi,
Xiang “Anthony” Chen, and Ruofei Du. 2023. Visual captions: Augmenting
verbal communication with on-the-fly visuals. InACM Conf. Human Factors in
Computing Systems (CHI). ACM, New York, NY, USA, 1–20. doi:10.1145/3544548.
3581566
[72] LiveKit Inc. 2025. LiveKit: Open Source Realtime Communication Infrastructure.
https://livekit.io. Accessed: 2025.
[73] Narges Mahyar, Ali Sarvghad, and Melanie Tory. 2012. Note-taking in co-located
collaborative visual analytics: Analysis of an observational study.Information
Visualization11, 3 (July 2012), 190–204. doi:10.1177/1473871611433713
[74] Narges Mahyar and Melanie Tory. 2014. Supporting Communication and Coor-
dination in Collaborative Sensemaking.IEEE Trans. Visualization & Computer
Graphics (TVCG)20, 12 (Nov. 2014), 1633–1642. doi:10.1109/TVCG.2014.2346573
[75] Gary Marchionini. 2006. Exploratory search: from finding to understanding.
Comm. ACM (CACM)49, 4 (April 2006), 41–46. doi:10.1145/1121949.1121979
[76] Tara Matthews, Anind K Dey, Jennifer Mankoff, Scott Carter, and Tye Rattenbury.
2004. A toolkit for managing user attention in peripheral displays. InACM Symp.
User Interface Software and Technology (UIST). 247–256. doi:10.1145/1029632.
1029676
[77] Tara Matthews, Jodi Forlizzi, and Stacie Rohrbach. 2006.Designing Glance-
able Peripheral Displays. Technical Report UCB/EECS-2006-113. University of
California, Berkeley, Electrical Engineering and Computer Sciences.
[78] Katrina McChesney and Jill Aldridge. 2019. Weaving an interpretivist stance
throughout mixed methods research.Int. J. Res. Method Educ.42, 3 (May 2019),
225–238. doi:10.1080/1743727x.2019.1590811
[79] Sean M McNee, John Riedl, and Joseph A Konstan. 2006. Being accurate is
not enough: how accuracy metrics have hurt recommender systems. InCHI
Extended Abstracts. ACM, 1097–1101. doi:10.1145/1125451.1125659
[80] Chuan Meng, Francesco Tonolini, Fengran Mo, Nikolaos Aletras, Emine Yilmaz,
and Gabriella Kazai. 2025. Bridging the gap: From ad-hoc to proactive search in
conversations. InACM Conf. Research and Development in Information Retrieval
(SIGIR). 64–74. doi:10.1145/3726302.3729915
[81] Microsoft. 2026. Boost teamwork with AI in Microsoft Teams. https://www.
microsoft.com/en-us/microsoft-teams/teams-ai. Accessed: 2026-3-15.
[82] Todd Miller and John Stasko. 2002. Artistically conveying peripheral information
with the InfoCanvas. InConf. Advanced Visual Interfaces (AVI). 43–50. doi:10.
1145/1556262.1556268
[83] Michael Mulley. 2010. openParliament.ca: Parliament of Canada Data and API.
https://openparliament.ca. Independent aggregator of Canadian parliamentary
data. API at https://api.openparliament.ca. Accessed 2026-07-23.
[84] Jakob Nielsen. 1994. Enhancing the explanatory power of usability heuristics.
InACM Conf. Human Factors in Computing Systems (CHI). 152–158. doi:10.1145/
191666.191729
[85] Stan Nowak and Lyn Bartram. 2024. Designing for ambiguity in visual analyt-
ics: Lessons from risk assessment and prediction.IEEE Trans. Visualization &
Computer Graphics (TVCG)30, 1 (Jan. 2024), 924–933. doi:10.1109/TVCG.2023.
3326571
[86] Eyal Ofek, Shamsi T Iqbal, and Karin Strauss. 2013. Reducing Disruption from
Subtle Information Delivery during a Conversation: Mode and Bandwidth In-
vestigation. InACM Conf. Human Factors in Computing Systems (CHI). ACM,
New York, NY, USA, 3111–3120. doi:10.1145/2470654.2466425
[87] Emre Oral, Ria Chawla, Michel Wijkstra, Narges Mahyar, and Evanthia Dimara.
2024. From information to choice: A critical inquiry into visualization tools for
decision making.IEEE Trans. Visualization & Computer Graphics (TVCG)30, 1
(Jan. 2024), 359–369. doi:10.1109/tvcg.2023.3326593
[88] Otter.ai. 2026. Otter Meeting Agent - AI Notetaker, Transcription, Insights.
https://otter.ai/. Accessed: 2026-3-15.

UIST ’26, November 02–05, 2026, Detroit, MI, USA Mohammad Abolnejadian and Matthew Brehmer
[89] Stephan Pajer, Marc Streit, Thomas Torsney-Weir, Florian Spechtenhauser,
Torsten Möller, and Harald Piringer. 2017. WeightLifter: Visual weight space
exploration for Multi-Criteria Decision Making.IEEE Trans. Visualization & Com-
puter Graphics (TVCG)23, 1 (Jan. 2017), 611–620. doi:10.1109/tvcg.2016.2598589
[90] Gun Woo Park, Frederik Brudy, George Fitzmaurice, and Fraser Anderson. 2026.
GroundLink: Exploring how contextual meeting snippets can close common
ground gaps in editing 3D scenes for virtual Production. InACM Conf. Human
Factors in Computing Systems (CHI). ACM, 1–27. doi:10.1145/3772318.3790793
[91] Gun Woo Park, Payod Panda, Lev Tankelevitch, and Sean Rintel. 2024. The
CoExplorer technology probe: A generative AI-powered adaptive interface
to support intentionality in planning and running video meetings. InACM
Conf. Designing Interactive Systems (DIS). ACM, New York, NY, USA, 1638–1657.
doi:10.1145/3643834.3661507
[92] Matthew Purver, Jonathan Ginzburg, and Patrick Healey. 2003. On the Means
for Clarification in Dialogue. InCurrent and New Directions in Discourse and
Dialogue, Jan van Kuppevelt and Ronnie W Smith (Eds.). Text, Speech and
Language Technology, Vol. 22. Springer Netherlands, Dordrecht, 235–255. doi:10.
1007/978-94-010-0019-2_11
[93] Qdrant Team. 2024. Qdrant: High-Performance Vector Database and Vector
Search Engine. https://github.com/qdrant/qdrant. Accessed: 2024.
[94] quanla. 2024. macos-demo. https://github.com/quanla/macos-demo. Accessed:
2025.
[95] Eric D Ragan, Alex Endert, Jibonananda Sanyal, and Jian Chen. 2016. Char-
acterizing provenance in visualization and data analysis: An organizational
framework of provenance types and purposes.IEEE Trans. Visualization & Com-
puter Graphics (TVCG)22, 1 (Jan. 2016), 31–40. doi:10.1109/TVCG.2015.2467551
[96] Sebastián Ramírez. 2024. FastAPI: Modern, Fast Web Framework for Building
APIs with Python. https://fastapi.tiangolo.com. Accessed: 2024.
[97] Bradley J Rhodes and Pattie Maes. 2000. Just-in-time information retrieval
agents.IBM Systems Journal39, 3.4 (2000), 685–704. doi:10.1147/sj.393.0685
[98] Victor Riley. 1989. A General Model of Mixed-Initiative Human-Machine
Systems.Proc. Human Factors Soc. Annu. Meet.33, 2 (Oct. 1989), 124–128.
doi:10.1177/154193128903300227
[99] C Rochat. 2002. Possible solutions to information overload.S. Afr. J. Inf. Manag.
4, 2 (Dec. 2002). doi:10.4102/sajim.v4i2.170
[100] Arvind Satyanarayan, Dominik Moritz, Kanit Wongsuphasawat, and Jeffrey Heer.
2017. Vega-Lite: A Grammar of Interactive Graphics.IEEE Trans. Visualization
& Computer Graphics (TVCG)23, 1 (Jan. 2017), 341–350. doi:10.1109/TVCG.2016.
2599030
[101] Orit Shaer, Angelora Cooper, Osnat Mokryn, Andrew L Kun, and Hagit
Ben Shoshan. 2024. AI-Augmented Brainwriting: Investigating the use of LLMs
in group ideation. InACM Conf. Human Factors in Computing Systems (CHI).
ACM, 1–17. doi:10.1145/3613904.3642414
[102] Chirag Shah. 2018. Information fostering - being proactive with information
seeking and retrieval: Perspective paper. InACM Conf. Human Information
Interaction and Retrieval (CHIIR). 62–71. doi:10.1145/3176349.3176389
[103] Nikhil Sharma, Q Vera Liao, and Ziang Xiao. 2024. Generative echo chamber?
Effect of LLM-powered search systems on diverse information seeking. InACM
Conf. Human Factors in Computing Systems (CHI). ACM, 1–17. doi:10.1145/
3613904.3642459
[104] Rizul Sharma, Tianyu Jiang, Seokki Lee, and Jillian Aurisano. 2025. Can AI agents
understand spoken conversations about data visualizations in online meetings?.
InIEEE VIS Workshop on Multimodal Experiences for Remote Communication
Around Data Online (MERCADO).
[105] Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Talaei Khoei. 2025. Agentic
retrieval-Augmented Generation: A survey on agentic RAG.arXiv [cs.AI](Jan.
2025). arXiv:2501.09136 [cs.AI] doi:10.48550/arXiv.2501.09136
[106] Speechmatics. 2024. Speechmatics: Automatic Speech Recognition API. https:
//www.speechmatics.com. Accessed: 2024.
[107] Arjun Srinivasan and Matthew Brehmer. 2023. Combining Voice and Gesture
for Presenting Data to Remote Audiences. InMERCADO Wkshp. Multimodal
Experiences for Remote Communication Around Data Online, IEEE VIS.
[108] Garold Stasser and William Titus. 1985. Pooling of unshared information in
group decision making: Biased information sampling during discussion.J. Pers.
Soc. Psychol.48, 6 (June 1985), 1467–1478. doi:10.1037/0022-3514.48.6.1467
[109] Xiaofei Sun, Xiaoya Li, Jiwei Li, Fei Wu, Shangwei Guo, Tianwei Zhang, and
Guoyin Wang. 2023. Text Classification via Large Language Models. InFind-
ings of the Association for Computational Linguistics: EMNLP 2023. Association
for Computational Linguistics, Singapore, 8990–9005. doi:10.18653/v1/2023.
findings-emnlp.603
[110] John Sweller. 2011. Cognitive Load Theory. InPsychology of Learning and
Motivation. Vol. 55. Elsevier, 37–76. doi:10.1016/b978-0-12-387691-1.00002-8
[111] Yuan Tian, Weiwei Cui, Dazhen Deng, Xinjing Yi, Yurun Yang, Haidong Zhang,
and Yingcai Wu. 2025. ChartGPT: Leveraging LLMs to Generate Charts from
Abstract Natural Language.IEEE Trans. Visualization & Computer Graphics
(TVCG)31, 3 (2025), 1731–1745. doi:10.1109/TVCG.2024.3368621
[112] Melanie Tory, Lyn Bartram, Brittany Fiore-Gartland, and Anamaria Crisan. 2023.
Finding their data voice: Practices and challenges of dashboard users.IEEEComputer Graphics & Applications (CGA)43, 1 (Jan. 2023), 22–36. doi:10.1109/
MCG.2021.3136545
[113] Stephen E Toulmin. 2003.The Uses of Argument(updated ed.). Cambridge
University Press. doi:10.1017/cbo9780511840005
[114] Khanh-Tung Tran, Dung Dao, Minh-Duong Nguyen, Quoc-Viet Pham, Barry
O’Sullivan, and Hoang D Nguyen. 2025. Multi-Agent Collaboration Mechanisms:
A Survey of LLMs.arXiv [cs.AI](Jan. 2025). arXiv:2501.06322 [cs.AI] doi:10.
48550/arXiv.2501.06322
[115] Amos Tversky and Daniel Kahneman. 1973. Availability: A heuristic for judging
frequency and probability.Cogn. Psychol.5, 2 (Sept. 1973), 207–232. doi:10.1016/
0010-0285(73)90033-9
[116] Jacob VanderPlas, Brian Granger, Jeffrey Heer, Dominik Moritz, Kanit Wong-
suphasawat, Arvind Satyanarayan, Eitan Lees, Ilia Timofeev, Ben Welsh, and
Scott Sievert. 2018. Altair: Interactive Statistical Visualizations for Python.J.
Open Source Softw.3, 32 (Dec. 2018), 1057. doi:10.21105/joss.01057
[117] Rishi Vanukuru, Payod Panda, Xinyue Chen, Ava Elizabeth Scott, Lev Tankele-
vitch, and Sean Rintel. 2025. Designing interfaces that support temporal work
across meetings with generative AI. InACM Conf. Designing Interactive Systems
(DIS). ACM, New York, NY, USA, 3600–3620. doi:10.1145/3715336.3735833
[118] Somin Wadhwa and Hamed Zamani. 2021. Towards System-Initiative Conversa-
tional Information Seeking. InDesign of Experimental Search & Information RE-
trieval Systems (DESIRES). 102–116. https://ceur-ws.org/Vol-2950/paper-17.pdf
[119] Daniel J Walters, Philip M Fernbach, Craig R Fox, and Steven A Sloman. 2017.
Known unknowns: A critical determinant of confidence and calibration.Manage.
Sci.63, 12 (Dec. 2017), 4298–4307. doi:10.1287/mnsc.2016.2580
[120] Ruotong Wang, Xinyi Zhou, Lin Qiu, Joseph Chee Chang, Jonathan Bragg, and
Amy X Zhang. 2025. Social-RAG: Retrieving from group interactions to socially
ground AI generation. InACM Conf. Human Factors in Computing Systems (CHI).
ACM, New York, NY, USA. doi:10.1145/3706598.3713749
[121] Mark Weiser and John Seely Brown. 1995. Designing Calm Technology. Xerox
PARC. http://www.ubiq.com/hypertext/weiser/calmtech/calmtech.htm
[122] Ryen W White, Bill Kules, Steven M Drucker, and m c Schraefel. 2006. Supporting
exploratory search, introduction, special issue, Communications of the ACM.
Comm. ACM (CACM)49, 4 (April 2006), 36–39. doi:10.1145/1121949.1121978
[123] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li
Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah,
Ryen W White, Doug Burger, and Chi Wang. 2024. AutoGen: Enabling Next-
Gen LLM Applications via Multi-Agent Conversation. InFirst Conf. Language
Modeling (COLM). https://openreview.net/forum?id=BAakY1hNKS
[124] Haijun Xia, Jennifer Jacobs, and Maneesh Agrawala. 2020. Crosscast: Adding
visuals to audio travel podcasts. InACM Symp. User Interface Software and
Technology (UIST). ACM, New York, NY, USA, 735–746. doi:10.1145/3379337.
3415882
[125] Haijun Xia, Tony Wang, Aditya Gunturu, Peiling Jiang, William Duan, and
Xiaoshuo Yao. 2023. CrossTalk: Intelligent substrates for language-oriented
interaction in video-based communication and collaboration. InACM Symp.
User Interface Software and Technology (UIST). ACM, New York, NY, USA, 1–16.
doi:10.1145/3586183.3606773
[126] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. 2024. Cor-
rective Retrieval Augmented Generation.arXiv [cs.CL](Jan. 2024).
arXiv:2401.15884 [cs.CL] doi:10.48550/arXiv.2401.15884
[127] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R
Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Act-
ing in Language Models. InIntl. Conf. Learning Representations (ICLR). https:
//openreview.net/forum?id=WE_vluYUL-X
[128] Ji Soo Yi, Rachel Melton, John Stasko, and Julie A Jacko. 2005. Dust & magnet:
Multivariate information visualization using a magnet metaphor.Information
Visualization4, 4 (Dec. 2005), 239–256. doi:10.1057/palgrave.ivs.9500099
[129] Yunfeng Zhang, Q Vera Liao, and Rachel K E Bellamy. 2020. Effect of confidence
and explanation on accuracy and trust calibration in AI-assisted decision making.
InACM Conf. Fairness, Accountability, and Transparency (FAccT). ACM, New
York, NY, USA, 295–305. doi:10.1145/3351095.3372852
[130] Jian Zhao, Michael Glueck, Petra Isenberg, Fanny Chevalier, and Azam Khan.
2018. Supporting handoff in asynchronous collaborative sensemaking using
knowledge-transfer graphs.IEEE Trans. Visualization & Computer Graphics
(TVCG)24, 1 (Jan. 2018), 340–350. doi:10.1109/TVCG.2017.2745279
[131] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu,
Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P Xing, Hao Zhang,
Joseph E Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-bench
and Chatbot Arena. InConf. Neural Information Processing Systems (NeurIPS).
46595–46623. doi:10.52202/075280-2020
[132] Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi
Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, and Ed Chi.
2023. Least-to-Most Prompting Enables Complex Reasoning in Large Language
Models. InIntl. Conf. Learning Representations (ICLR). https://openreview.net/
pdf?id=WZH7099tgfM
[133] Zoom Communications. 2026. Customize your AI Companion. https://www.
zoom.com/en/products/custom-ai/. Accessed: 2026-3-15.

InsightToast UIST ’26, November 02–05, 2026, Detroit, MI, USA
A Appendix: Pipeline Implementation Details
This appendix details the proactive retrieval-trigger and chart-
design criteria referenced in Section 4.2.
A.1 Knowledge Gap Detection Criteria
Retrieval is triggered when the current utterance, read against the
recent conversational window and the meeting’s running context
(Section 4.2), satisfies at least one of the conditions in Table A.1;
for pragmatic reasons, topical relevance or an isolated hedge word
alone does not qualify. Categories are disjunctive while each cate-
gory’s own conditions remain conjunctive, weighting the resulting
judgment toward recall over precision (Section 5.2), since a dismiss-
able toast is a lower-cost outcome than a missed gap in a high-stakes
deliberation. Detection is thus aimed at resolving ambiguity and
preventing discussion from stalling, opening room for participants
to pursue avenues they might not otherwise raise.
A.2 Chart Design Constraints and Verification
Chart type selection follows the analytic intent and data shape of the
retrieved evidence, drawn from the restricted inventory in Table A.2.
Suppression of non-essential elements (Section 4.1) is enforced as
fixed generation rules rather than case-by-case judgment: legends
are added only after simplifying categories fails to resolve ambiguity,
gridlines appear only when they aid reading a quantitative axis,
and exact values are exposed via tooltip rather than on-chart labels.
The generation agent verifies the resulting specification against
these rules before returning it, regenerating it if validation fails.
Table A.2: Restricted chart type inventory used during vi-
sualization planning, organized by data shape and analytic
intent. Secondary types require explicit justification that no
primary type serves the intent.
Data Shape (Intent) Chart Types
Category×Value
(Comparison/Ranking)Simple Bar, Horizontal Bar, Sorted Bar
Secondary:Grouped Bar, Horizontal Grouped Bar, Bar w/
Highlighted Bar
Category×Value
(Composition)Stacked Bar, Horizontal Stacked Bar, Normalized Stacked Bar,
Horizontal Normalized Stacked Bar
Secondary:Diverging Stacked Bar
Time×Value
(Trend)Simple Line, Multi-Series Line, Step
Secondary:Stacked Area, Slope Graph, Bump Chart
Parts-of-Whole
(≤6 categories)Donut, Pie
Event Distribution Strip Plot
Grid/MatrixSecondary:Heatmap
B Appendix: Evaluation Task Materials
This appendix documents the materials used to administer the
deliberation task described in Section 5.
B.1 Task Briefs
As described in Section 5, each task was framed as a citizen advisory
brief in which participants recommended one of two competing
policy positions. We gave each participants a task card presenting
the policy question, both positions with a brief description and
arguments for and against to better introduce different aspects ofthe positions, and three justification criteria to consider during
deliberation:value for public money(whether the approach is a
responsible use of government funding),evidence of effectiveness
(whether the approach has been studied or discussed by govern-
ment), andpolitical feasibility(whether there is evidence of prior
legislative support or opposition). The task cards below were pre-
sented to participants, one per condition, each framing a policy
decision between two competing positions.
Task 1: Rental Housing Affordability
How should the federal government address rental housing affordabil-
ity for citizens?
 
Position A: Introduce a national rent increase cap
Limits how much landlords can raise rent each year, tied to the rate of
inflation (CPI).
For:Protects existing tenants from sudden large rent increases while
longer-term solutions are developed.
Against:Critics argue it discourages landlords from building new rental
units, shrinking supply over time.

 
Position B: Fund purpose-built rental construction
Provides federal tax incentives and direct funding to developers who
build new rental-only housing.
For:Increases the number of available rental units, addressing the root
cause of high prices.
Against:Critics argue the benefits are long-term and do nothing for
renters struggling with costs right now.
Task 2: Healthcare Wait Times
What should the federal government do to reduce healthcare wait
times for citizens?
 
Position A: Conditional Health Funding
Makes a portion of federal health funding to provinces/states conditional
on meeting and publicly reporting specific wait time targets.
For:Creates measurable accountability for how provinces/states spend
federal health dollars.
Against:Critics argue it penalizes provinces/states where chronic un-
derfunding and staff shortages are already severe.

 
Position B: Federal hospital staffing & infrastructure investment
Provides direct federal funding to hospitals and health authorities specif-
ically to hire clinical staff.
For:Directly addresses the staffing shortages and capacity gaps driving
long wait times.
Against:Critics argue that without conditions attached,
provinces/states have no obligation to maintain improvements
after funding ends.
The full study instruments, including the pre- and post-task
questionnaires, the closing semi-structured interview guide, and
the original task cards as presented to participants are provided as
supplemental material.

UIST ’26, November 02–05, 2026, Detroit, MI, USA Mohammad Abolnejadian and Matthew Brehmer
Table A.1: Trigger conditions evaluated by the knowledge gap detection agent. Retrieval is initiated only when all conditions
within at least one trigger category are satisfied.
Trigger Condition
Explicit knowledge gapA speaker explicitly requests or admits lacking specific background information, past data, or historical context
The gap is concrete and specific, not vague or rhetorical
The gap cannot be resolved from information already available within the conversation
Factual or historical inquiryA direct question is posed about past events, decisions, precedents, or official records
The speaker indicates that obtaining this information would help the discussion proceed
The question cannot be answered from within the conversation itself
Contested or uncertain
claimsA verifiable claim or assumption is potentially incorrect or actively contested
The uncertainty is visibly affecting the direction of the discussion
Evidence-dependent
decisionA concrete, unresolved question is on the table that participants are actively trying to resolve
Resolving it requires factual evidence not currently available to any participant
Stalled discussionThe discussion has become circular, with no forward progress across recent turns
Participants lack the factual grounding needed to advance
B.2 Conversation Partner Cues
During each deliberation, a member of the research team served as
the participant’s conversation partner, using the scripted cues below
to build rapport, sustain the conversation, and keep the discussion
on topic without steering participants toward either position. We
organized cues into three phases matching the deliberation stage
the participant was in, and bracketed placeholders (e.g.[topic])
indicate the specific pair of positions assigned to each task.
Introduction
•“Both sound reasonable, but reading them, what is your initial
reaction to them?”
•“That makes sense. But I’m wondering: do we actually know if
Parliament has looked at this seriously before? Has there been
any committee work on [topic] specifically?”
•“What were the findings on similar approaches before?”
Knowledge Base Exploration
•“I wonder if there’s been a committee study on this, with actual
recommendations, not just debate.”•“The card mentions [an argument]. Has Parliament actually heard
from [experts/committee researchers] on that? I’d want to know
what the evidence says before we take that argument at face
value.”
•“Has any bill on [this problem] actually made it through Parlia-
ment, or do they all die on the order paper?”
•“What do the voting records look like on [bills supporting this]?
Is this something with cross-party support or is it completely
partisan?”
•“There must be petitions on [this problem]. I wonder how much
public pressure Parliament has actually received on this.”
Deliberation
•“OK, so based on what we’ve found, which of these positions
actually holds up on the value-for-money side?”
•“Is there enough political will in Parliament for either of these to
actually pass? Because a great policy that goes nowhere doesn’t
help anyone.”
•“Even if we’re not fully certain, which position do we feel we can
defend better with what we found?”
# Event Extraction in Large Language Model

**Authors**: Bobo Li, Xudong Han, Jiang Liu, Yuzhe Ding, Liqiang Jing, Zhaoqi Zhang, Jinheng Li, Xinya Du, Fei Li, Meishan Zhang, Min Zhang, Aixin Sun, Philip S. Yu, Hao Fei

**Published**: 2025-12-22 16:22:14

**PDF URL**: [https://arxiv.org/pdf/2512.19537v1](https://arxiv.org/pdf/2512.19537v1)

## Abstract
Large language models (LLMs) and multimodal LLMs are changing event extraction (EE): prompting and generation can often produce structured outputs in zero shot or few shot settings. Yet LLM based pipelines face deployment gaps, including hallucinations under weak constraints, fragile temporal and causal linking over long contexts and across documents, and limited long horizon knowledge management within a bounded context window. We argue that EE should be viewed as a system component that provides a cognitive scaffold for LLM centered solutions. Event schemas and slot constraints create interfaces for grounding and verification; event centric structures act as controlled intermediate representations for stepwise reasoning; event links support relation aware retrieval with graph based RAG; and event stores offer updatable episodic and agent memory beyond the context window. This survey covers EE in text and multimodal settings, organizing tasks and taxonomy, tracing method evolution from rule based and neural models to instruction driven and generative frameworks, and summarizing formulations, decoding strategies, architectures, representations, datasets, and evaluation. We also review cross lingual, low resource, and domain specific settings, and highlight open challenges and future directions for reliable event centric systems. Finally, we outline open challenges and future directions that are central to the LLM era, aiming to evolve EE from static extraction into a structurally reliable, agent ready perception and memory layer for open world systems.

## Full Text


<!-- PDF content starts -->

ARXIV PREPRINT 1
Event Extraction in Large Language Model:
A Holistic Survey of Method, Modality, and Future
Bobo Li, Xudong Han, Jiang Liu, Yuzhe Ding Liqiang Jing, Zhaoqi Zhang, Jinheng Li,
Xinya Du, Fei Li, Meishan Zhang, Min Zhang, Aixin Sun, Philip S. Yu, Hao Fei
/githubhttps://github.com/unikcc/AwesomeEventExtraction
Abstract—Large language models (LLMs) and multimodal
LLMs are changing event extraction (EE): prompting and
generation can often produce structured outputs in zero shot
or few shot settings. Yet LLM based pipelines face deployment
gaps, including hallucinations under weak constraints, fragile
temporal and causal linking over long contexts and across
documents, and limited long horizon knowledge management
within a bounded context window. We argue that EE should be
viewed as a system component that provides a cognitive scaffold
for LLM centered solutions. Event schemas and slot constraints
create interfaces for grounding and verification; event centric
structures act as controlled intermediate representations for
stepwise reasoning; event links support relation aware retrieval
with graph based RAG; and event stores offer updatable episodic
and agent memory beyond the context window. This survey
covers EE in text and multimodal settings, organizing tasks and
taxonomy, tracing method evolution from rule based and neural
models to instruction driven and generative frameworks, and
summarizing formulations, decoding strategies, architectures,
representations, datasets, and evaluation. We also review cross
lingual, low resource, and domain specific settings, and highlight
open challenges and future directions for reliable event centric
systems. Finally, we outline open challenges and future directions
that are central to the LLM era, aiming to evolve EE from static
extraction into a structurally reliable, agent ready perception and
memory layer for open world systems.
CONTENTS
I Introduction. . . . . . . . . . . . . . . . . . . . 2
II Tasks & Taxonomy. . . . . . . . . . . . . . . . 3
II-A Text-based EE. . . . . . . . . . . . . . . . . . 3
II-A1 Trigger Detection & Typing. . . . . . . . 3
II-A2 Argument Extraction. . . . . . . . . . . . 4
II-A3 Event Coreference Resolution. . . . . . . . 4
II-A4 Event–Event Relations. . . . . . . . . . . 5
II-B Multimodal EE. . . . . . . . . . . . . . . . . 5
II-B1 Visual EE. . . . . . . . . . . . . . . . . 5
II-B2 Video EE. . . . . . . . . . . . . . . . . 6
II-B3 Audio/Speech EE. . . . . . . . . . . . . . 6
II-B4 Cross-modal EE. . . . . . . . . . . . . . 6
III Methodology. . . . . . . . . . . . . . . . . . . . 7
III-A Rule-based Methods. . . . . . . . . . . . . . . 7
Bobo Li, Jinheng Li, and Hao Fei are with the National University of
Singapore, Singapore. Xudong Han is with the University of Sussex, U.K.
Jiang Liu, Yuzhe Ding, and Fei Li are with Wuhan University, China. Liqiang
Jing and Xinya Du are with the University of Texas at Dallas, USA. Zhaoqi
Zhang and Aixin Sun are with Nanyang Technological University, Singapore.
Meishan Zhang and Min Zhang are with the Harbin Institute of Technology
(Shenzhen), China. Philip S. Yu is with the University of Illinois at Chicago,
USA. Corresponding author: Hao Fei (e-mail: haofei7419@gmail.com)III-B Classic ML Methods. . . . . . . . . . . . . . 7
III-C Deep Learning Methods. . . . . . . . . . . . 7
III-C1 CNN-based Methods. . . . . . . . . . . . 8
III-C2 RNN-based Methods. . . . . . . . . . . . 8
III-C3 Transformer-based Methods. . . . . . . . . 9
III-C4 Graph-based Methods. . . . . . . . . . . 9
III-D LLM-based Methods. . . . . . . . . . . . . . 9
III-D1 Instruction-tuned Models.. . . . . . . . . . 9
III-D2 In-context Learning.. . . . . . . . . . . . 10
III-D3 Chain-of-Thought (CoT).. . . . . . . . . . 11
III-D4 Multi-agent Methods. . . . . . . . . . . . 11
III-D5 Data Augmentation with LLMs. . . . . . . 11
III-D6 Multimodal LLMs (MLLMs).. . . . . . . . 12
IV Formulations & Decoding. . . . . . . . . . . . 12
IV-A Classification. . . . . . . . . . . . . . . . . . 12
IV-B Sequence Labeling. . . . . . . . . . . . . . . 12
IV-C Span/Pointer. . . . . . . . . . . . . . . . . . . 12
IV-D Table/Grid. . . . . . . . . . . . . . . . . . . . 13
IV-E Generation. . . . . . . . . . . . . . . . . . . 13
V System Architectures. . . . . . . . . . . . . . . 14
V-A Pipeline. . . . . . . . . . . . . . . . . . . . . 14
V-B Joint/Global. . . . . . . . . . . . . . . . . . . 15
V-C One-Stage/Unified. . . . . . . . . . . . . . . . 15
VI Representations & Feature Engineering. . . . 15
VI-A Lexicon. . . . . . . . . . . . . . . . . . . . . 15
VI-B Syntactic & Semantic. . . . . . . . . . . . . . 15
VI-C Knowledge Retrieval. . . . . . . . . . . . . . . 16
VI-D Pretrained Embeddings. . . . . . . . . . . . . 16
VI-E Visual Feature. . . . . . . . . . . . . . . . . . 16
VII Dataset & Evaluation. . . . . . . . . . . . . . . 17
VII-A Text Datasets. . . . . . . . . . . . . . . . . . 17
VII-B Multimodal Datasets. . . . . . . . . . . . . . 18
VII-C Evaluation Metrics. . . . . . . . . . . . . . . . 18
VII-D Tools. . . . . . . . . . . . . . . . . . . . . . 19
VIII EE under Diverse Settings. . . . . . . . . . . . 20
VIII-A Language and Resource Conditions. . . . . . . 20
VIII-B Discourse Scope (Granularity). . . . . . . . . . 21
VIII-C Vertical Domains. . . . . . . . . . . . . . . . 22
IX Open Challenges and Future Directions. . . . 24
IX-A Agentic Perception. . . . . . . . . . . . . . . 24
IX-B Neuro-Symbolic Reasoning. . . . . . . . . . . 24
IX-C Interactive Open-World Discovery. . . . . . . . 25
IX-D Cross-Document Synthesis. . . . . . . . . . . . 25
IX-E Physically Grounded World Models. . . . . . . 25
IX-F Utility-Driven Evaluation. . . . . . . . . . . . 25
X Conclusion. . . . . . . . . . . . . . . . . . . . . 26
References. . . . . . . . . . . . . . . . . . . . . . . . 26arXiv:2512.19537v1  [cs.CL]  22 Dec 2025

2 ARXIV PREPRINT
MUC-3, [1], 1991
(Task Initialization)TimeML&TimeBank, [2], 2003
(Time Annotation & Corpus)
ACE-2004, [3], 2005
(Multilingual Dataset)ACE-2005, [4], 2006
(Multilingual Dataset)
Stage-Split, [5], 2006
(Stage Modeling)BioNLP’09, [6], 2009
(BioNLP Dataset)
CrossEvent Inference, [7], 2010
(Document Level Inference)MaxEnt, [8], 2013
(Joint Outperform Pipeline)
DMCNN, [9], 2015
(Dynamic CNN)DomainBiLSTM, [10], 2015
(Domain Adaptation)
RichEE, [11], 2015
(Domain Adaptation)JRNN, [12], 2016
(bidirectional RNN)
JointEventEntity, [13], 2016
(Joint Events & Entities)FBRNN, [14], 2016
(BiRNN for Event Detection)
GCN-EN, [15], 2018
(GCN for Event Detection)JMEE, [16], 2018
(Attention-based GCN)
Bert, [17], 2018
(PLM Era Begins)
HMEAE, [18], 2019
(Hierarchy Modular EE)
MA VEN, [19], 2020
(General Domain Dataset)OneIE, [20], 2020
(Joint EE Framework)
M2E2, [21], 2020
(Multimedia EE Dataset)RAME, [22], 2020
(Document-level Arg Data)
EEMRC, [23], 2020
(EE as MRC)BERT-QA, [24], 2020
(QA-style EE)
CASIE, [25], 2020
(Cybersecurity EE Data)MGR, [26], 2020
(Role Filler EE)
WikiEvents, [27], 2021
(Document Event)VM2E2, [28], 2021
(Video EE Data)
Text2Event, [29], 2021
(Structure Generation)TANL, [30], 2021
(EE as Translation)
DEGREE, [31], 2022
(Generative-based EE)OneEE, [32], 2022
(Grid-Tagging EE)
ChatGPT, [33], 2022
(LLM Era Begins)
ChatIE, [34], 2023
(ChatGPT for EE)OmniEvent, [35], 2023
(LLM-based EE Tools)
CAMEL, [36], 2023
(Generation Argument EE)MosaiCLIP, [37], 2023
(Contrastive Learning for MMEE)
InstructUIE, [38], 2023
(Instruction-tuned EE)DIE-EC, [39], 2024
(Cross-Document ECR Data)
CDEE, [40], 2024
(Cross-Document EE Data)DiscourseEE, [41], 2024
(Implicit EE Data)
HD-LoA, [42], 2024
(In-Context Learning for EE)ULTRA, [43], 2024
(Enhance LLM for EE)
UMIE, [44], 2024
(Instruction-tuned MMEE)GEMS, [45], 2025
(Ontology-guided EE)
XMTL, [46], 2025
(Multi-task Learning for EE)LSED, [47], 2025
(LLM for Social EE)
Sed-Aug, [48], 2025
(Data Augmentation for EE)SEOE, [49], 2025
(Open Domain EE)Dataset Pre-LLM Methods
PLM-based Methods LLM-based Methods
Fig. 1. A roadmap of Event Extraction (1991–2025): Key milestones in the Pre-PLM, PLM, and LLM eras.
I. INTRODUCTION
EVent extraction (EE) is a core task in natural language
processing that aims to identify event triggers, event
types, and participant roles from unstructured text, and to
organize them into a computable structured representation
[27]. Unlike static facts at the entity or relation level, events
capture what happened, who was involved, when and where
it happened, how it unfolded, and what outcomes followed.
This capability is important in application settings that require
tracking and interpreting real-world dynamics, including finan-
cial risk control and public opinion monitoring, clinical course
tracking, situational awareness, and public safety and emer-
gency warning. Over the past two decades, the community has
developed many datasets and benchmarks, and has advanced
methods from rule-based and feature-engineering approaches
to neural and graph-based modeling [12], [16], [27]. These
efforts have also supported the construction and use of event
knowledge bases and event graphs, making EE a key pillar
within the broader information extraction landscape.
The rise of large language models (LLMs) is reshapingthe practice of information extraction. Models that previously
required task-specific training can now often be replaced by
prompting a general-purpose LLM to directly produce outputs
that resemble structured records, sometimes even in zero-shot
or few-shot settings [50], [51]. This shift raises an unavoidable
question:in an era where LLMs can process text end to end
and generate structured outputs, is event extraction still
necessary as an independent research direction?In many
real deployments, the prevailing practice is increasingly to feed
raw text to an LLM, rather than to first extract structured
events and then perform downstream reasoning and decision
making on top of those structures. As a result, EE may appear
less central than before, and it can be mistakenly viewed as
something that end-to-end generation can simply replace.
This survey argues that LLMs do not diminish the value of
event extraction. Instead, they push EE from a task- or model-
centric problem toward a system-level structured interface
and constraint layer. The key distinction is that practical
deployments care not only about producing an answer, but
also about meeting system requirements such as reliability,

LIet al.: EVENT EXTRACTION 3
traceability, and long-horizon knowledge management. Under
these requirements, relying solely on unconstrained end-to-
end generation exposes substantialcognitive gaps. First, gen-
erative outputs are probabilistic. Without explicit structural
constraints, models can hallucinate and errors can accumulate
across multi-step pipelines [52]. Second, when evidence is
dispersed across long contexts or multiple documents, models
often struggle to maintain stable links among temporal order,
causal chains, and role coreference. This makes the reasoning
process brittle and difficult to audit. Third, similarity-based re-
trieval does not guarantee access to precise temporal or causal
relations, and limited context windows cannot accommodate
a continuous stream of experiences in open environments.
Consequently, simply stacking more text is often insufficient
for long-term planning and consistent behavior.
Event extraction offers a structured complement that directly
targets these system-level gaps. Because EE outputs are ex-
plicit, constrained, and computable, they can serve as interme-
diate representations and external memories in LLM-centered
systems. In this sense, EE evolves from a static prediction task
into acognitive scaffold. First, for reliability, event schemas
and slot constraints provide concrete interfaces for grounding
and verification, narrowing the space of free-form generation
and supplying anchors for checking and correction. Second,
for reasoning, event chains decompose narratives into discrete
steps and can function as controlled intermediate structures
analogous to Chain-of-Thought reasoning, improving control-
lability and reproducibility [53]. Third, for knowledge access
and memory, events and their temporal, causal, and role links
enable retrieval to move beyond similarity matching toward
relation-navigable, graph-based retrieval-augmented genera-
tion [54], [55]. This organization further supports updatable
episodic memory, which is useful for agents that require
long-horizon planning without being constrained by context
overflow [56]. Therefore, in the LLM era, the value of EE lies
less in being the only path to structured outputs, and more
in providing a structural backbone for verification, reasoning,
retrieval, and agent memory.
Motivated by this perspective, this paper revisits event
extraction along the axis oftasks, datasets and evaluation,
and methodological paradigms. We start from the classic
definitions and decompositions of textual EE, review repre-
sentative datasets, evaluation protocols, and metrics, and then
summarize the evolution of modeling approaches from rule-
based and traditional learning methods to neural, generative,
and instruction-driven frameworks. We further discuss how
multimodal and cross-document settings extend the boundary
of EE. Building on this foundation, we examine how emerging
generative models and agentic systems reshape the functional
role of EE in practice, highlighting common collaboration
patterns with LLMs, including structured constraints, verifi-
able workflows, graph-based retrieval, and external memory.
Finally, we distill open challenges and future directions guided
by application requirements, aiming to inform the design of
reliable, controllable, and deployable event-centric intelligent
systems.
The remainder of this survey is organized as follows.
Section II introduces the task definition and taxonomy of eventextraction, unifies the core subtasks in textual EE, and extends
the discussion to multimodal settings such as vision, video,
and speech. Section III to Section VI review methods and
modeling routes, covering rule-based and traditional learning
approaches, deep learning, and the instruction-driven and
generative paradigms in the LLM and multimodal LLM era,
together with common task formulations, decoding strategies,
system architectures, and representation design. Section VII
summarizes datasets, evaluation metrics, and toolchains, and
Section VIII discusses diverse application settings such as
cross-lingual and low-resource scenarios, different granulari-
ties, and vertical domains. Section IX outlines open challenges
and future directions, and Section X concludes the survey.
Evolution: Static Task vs. Cognitive Scaffold
•Traditionally, EE was viewed as a standalone prediction
task to populate static knowledge bases.
•In the LLM era, EE evolves into a structural interface
(Scaffold) that enhances system reliability, reasoning,
and memory.
II. TASKS& TAXONOMY
To systematically navigate the diverse landscape of Event
Extraction (EE), we present a comprehensive taxonomy
in Fig. 2, which organizes the field into tasks, methods,
paradigms, and applications. In this section, we focus specif-
ically on theTasksdimension to define the fundamental
problem scope. We categorize EE tasks into two primary
streams: the establishedText-based EE, comprising subtasks
like trigger detection and argument extraction, and the expand-
ingMultimodal EE, which incorporates visual and acoustic
signals.
A. Text-based EE
Text-based event extraction [12], [147], [198] is the most
established and widely studied setting in the broader event
understanding landscape. The objective is to transform un-
structured text into structured event knowledge—typically,
machine-readable records that specify an event type, its partic-
ipants, attributes, and links to other events. In practice, the task
is commonly decomposed into a pipeline of interconnected
sub-tasks: trigger detection [199], argument extraction [18],
event coreference resolution [79], and event-to-event relation
extraction [200]. Each stage feeds forward signals that refine
subsequent stages, while later stages provide opportunities for
joint or global consistency checks over earlier decisions.
1) Trigger Detection & Typing:Trigger detection and typ-
ing [68], [199] is the foundational step in event extraction:
the system identifies the text spans that most clearly signal
an event’s occurrence and assigns them a type within an
event schema. Triggers are the lexical anchors of an event
mention. As shown in Fig. 3a, they are often verbs (e.g.,
“attacked,” “resigned”) or nominalizations (e.g., “an attack,”
“the resignation”), but can also include adjectives or multiword
expressions depending on the schema and domain.

4 ARXIV PREPRINTEvent Extraction
in LLM EraTask DimensionsExtraction ScopeSentence: ACE05 [57], FewEvent [58], PHEE [59], MAVEN [19], CASIE [25], GENEVA [60], RAMS [22], MINION [61]
Document: Doc2EDAG [62], WIKIEVENTS [27], DE-PPN [63], DocEE [64], RoleEE [65], FEED [66], DuEE [67], HBTNGMA [68]
Cross-Doc: EventStoryLine [69], WEC-Eng [70], RECB [71], ECB+ [72], MCECR [73], FCC [74], LegalCore [75], GVA [76]
Event SemanticsCoreference: MAVEN-ERE [19], KBP2017 [77], GraphECR [78], WEC-Zh [39], Joint ECR [79], RJoint ECR [80], DIE-EC [40]
Relations: MATRES [81], Causal-TimeBank [82], HiEve [83], TimeBank-Dense [84], TDDiscourse [85], DocRED [64], MECI [86]
Ontology: RAMS [22], PHEE [59], OntoEvent [87], RichERE [88], MACCROBAT -EE [89], MLEE [90], MUC-4 [91]
MultimodalityVisual/Video: imSitu [92], VidSitu [93], SWiG [94], VASR [95], Grounded VidSitu [96], Video Swin [97], SlowFast [98], I3D [99]
Audio/Speech: SpeechEE [100], DeepSpeech [101], VGGish [102], wav2vec 2.0 [103], Whisper [104], ASR-based [105]
Cross-modal: M2E2[21], VOANews [106], MUIE [107], CMMEvent [108], VM2E2[28], MultiHiEve [109], M-VAE [110]
Modeling ParadigmsDiscriminativeSequence Labeling: OneIE [20], DCFEE [111], BEESL [112], ILP [113], SP-IE [114], SSED [115], MTTLADE [116], DLRNN [117]
Span/Pointer: DyGIE++ [118], PLMEE [119], MQAEE [120], BERT -QA [24], PAIE [121], RCEE ER [23], CasEE [122], RAAT [123]
Graph/GNN: JMEE [16], S-CNNs [124], JRNN [12], DMCNN [9], PMCNN [125], DBRNN [126], JointEventEntity [13]
GenerativeSeq2Seq: Text2Event [29], UIE [127], LasUIE [128], OneEE [32], EDEE [129], DualCor [130], ODEE [131], JEF-HM [132]
Prompting: BART -Gen [27], DEGREE [31], PAIE [121], GTEE-DynPref [133], DE-PPN [63], SCPRG [134], AMPERE [135]
Unified Models: DeepStruct [64], InstructUIE [38], OmniEvent [35], ADELIE [136], TimeLlaMA [137], RLQG [138], CollabKG [139]
LLM-SpecificInstruction Tuning: InstructUIE [38], ChatUIE [140], LLMERE [141], ITAG [142], ECIE [143], LLMR [144], GLF [145], GCIE [146]
Code Generation: Code4Struct [140], CodeIE [140], GoLLIE [147], LC4EE [148], RUIE [149]
Agent/Reasoning: ChatIE [34], STAR [150], LLM-ERL [151], DAO [152], TALOR-EE [153], SBDA [154]
Learning StrategiesData EfficiencyFew-Shot: FewEvent [58], FewFC [155], FewDocAE [156], ZSEE [157], Title2Event [158]
Unsupervised: CrossCluster [159], Type-Induction [160], R2E [161], Genia 2011 [162]
Knowledge
AugmentedRetrieval (RAG): RCEE ER [23], GREE [163], GRIT [164], GENEVA [165], TOPJUDGE [166], MoRAG-FD [167], EAESR [168]
Schema/Gloss: GEANet [169], SROLEPRED [65], SemSynGTN [170], BRAD [171], Seq2EG [172]
RepresentationEmbedding: BERT [17], SpanBERT [173], RoBERTa [174], BART [175], GATE [176], EABERT [177], Mabert [178], PLMEE [119]
Syntax/Structure: TSAR [179], AMPERE [135], TAG [180], EPIG-EAE [181], GTEE-DynPref [133], PAIE [121], SemSynGTN [170]
Eco-systemDomain ApplicationsBiomedical: PHEE [59], DeepEventMine [182], SPEED++ [183], BioNLP [162], BioDEX [184], MACCROBAT -EE [89], AniEE [185]
Finance: FinEvent [186], Doc2EDAG [64], CFinDEE [187], DCFEE [111], ChFinAnn [64], OEE-CFC [188], CFinEE [189]
Legal/Cyber: LEVEN [190], CASIE [25], LEEC [191], LegalCore [75], ExcavatorCovid [192], AEs [193], SEOE [49]
Tools & PlatformsFrameworks: TextEE [160], OneEE [32], DeepStruct [64], OmniEvent [35], DeepKE [194], LFDe [195]
Integrated Models: DyGIE++ [118], SpERT [196], RESIN [197], CasEE [122], UIE [127], CollabKG [139], ChatUIE [140]
EvaluationStandard: Strict/Soft Match F1, Trigger/Argument ID, Head/Exact Match (ACE05/MAVEN metrics), Mention Detection Rate
Advanced: Coreference (MUC/B3/CEAF/BLANC), Temporal F1, Hallucination Rate, Cross-modal Alignment Score
Fig. 2. Taxonomy of Event Extraction research in the LLM era, structured by tasks, modeling paradigms, learning strategies, and ecosystem.
Successfully identifying and typing these triggers, however,
involves overcoming several core challenges. First, the na-
ture of the event schema itself is a primary consideration.
Closed schemas with a predefined ontology (e.g., the types
in ACE-style resources [4]) constrain the label space and
facilitate supervised learning, whereas open schemas demand
models that can generalize to previously unseen or fluid event
types discovered from data. Second, further complexity arises
from the linguistic form of triggers. Beyond single tokens,
they can manifest as multiword expressions (e.g., “blew up”)
or even discontinuous phrases (e.g., “set . . . off”), which
challenges token-contiguous span assumptions and motivates
more sophisticated structured prediction. Finally, a significant
challenge lies in handling nested events, such as “the an-
nouncement of a merger.” These hierarchical structures strain
flat, span-based tagging approaches, requiring systems that can
recognize such compositions and maintain consistent typing
across levels [32].
2) Argument Extraction:Argument extraction [18] identi-
fies the participants and attributes associated with a detected
event and assigns them semantically meaningful roles. As
shown in Fig. 3b, given a specific trigger, the system selects
text spans, typically entity mentions, temporal expressions, andlocations, and links them as arguments to that event, thereby
answering the “who, what, when, and where” of the event
mention [18], [27], [201].
Accurately extracting these arguments poses several sig-
nificant challenges. A primary task is to assign the correct
semantic roles (e.g.,Attacker,Victim,Place) to each argument
based on a predefined schema. This process requires sophisti-
cated role disambiguation, which becomes particularly difficult
when dealing with label imbalance or domain shifts. The
complexity is further compounded because arguments do not
always appear adjacent to the event trigger. Participants may
be referenced elsewhere in the text through anaphora (e.g.,
pronouns) or bridging descriptions, compelling robust systems
to leverage coreference resolution and broader discourse con-
text. Finally, ensuring the semantic integrity of the extracted
structure is crucial. A single entity may legitimately fill
multiple roles, and schemas often impose strict type constraints
(e.g., aVictimmust be a person). Addressing these overlaps
and enforcing constraints often requires models capable of
joint inference and applying global logic across the document.
3) Event Coreference Resolution:Event coreference reso-
lution [78], [79], [202] clusters event mentions that denote
the same underlying occurrence. As shown in Fig. 3d, the

LIet al.: EVENT EXTRACTION 5
Google launched Pixel 8 in San Francisco on October 4th.Trigger: Product-Launch
(a) Event Trigger Detection
Trigger: Product-Launch
Google  launched  Pixel 8  in  San Francisco  on  October 4th.role: Agent role: Product role: Placerole: Time (b) Event Extraction
Trigger: Product-Launch Trigger: Communication-Speech
Trigger: Finance-PriceChange Trigger: Business-MeetingAt the Pixel 8 launch, the CEO spoke about the new AI camera.
This successful presentation later caused the  stock to surge.
Rel: Sub-EventRel: Sub-Event
(c) Event Relation Extraction
Event id: 002 Event id: 003Event id: 001
Google's hardware event took place on Tuesday. The highly 
anticipated presentation focused on the new phone. This 
launch was watched by millions.
CorefCoref (d) Event Coreference Resolution
Fig. 3. Task illustration for Text-based Event Extraction.
goal is to aggregate information, track event evolution across
mentions, and avoid redundant or fragmented representations
by merging co-referential mentions into a single canonical
event [80], [203].
This task is fundamentally challenging due to the multi-
dimensional nature of coreference signals and the varying
scope of analysis. A central difficulty is that signals for
coreference span across triggers, arguments, and attributes.
For triggers, this involves matching lexical identity, morpho-
logical variants, and semantic similarities. For arguments, it
requires assessing the compatibility of key roles, such as
shared participants or locations. For attributes, it demands
consistency in time, polarity, and modality. Reconciling partial
overlaps—where mentions share some but not all of these
features—remains a key hurdle. The complexity is further
shaped by the task’s scope. Within a single document (intra-
document), resolution can leverage strong local discourse cues,
while across multiple documents (cross-document), systems
must contend with greater lexical variability and temporal
drift, necessitating more robust normalization and external
knowledge.
4) Event–Event Relations:As presented in Fig. 3c, event-
to-event relation extraction [118], [200], [204] identifies struc-
tural links among events, enabling timeline construction,
causal interpretation, and narrative composition. This is a
higher-level reasoning task that abstracts beyond individual
mentions and frames to infer relations that organize events
into interpretable structures.
These relations are commonly categorized into three pri-
mary types. First, temporal relations [205] order events along
a timeline. This entails temporal anchoring—linking events
to absolute time expressions and normalizing them (e.g.,
TIMEX3 [206])—and recognizing relative relations such as
BEFORE, AFTER, and OVERLAPS. Temporal reasoning
must reconcile document creation time with vague expressionsand cross-sentence cues. Second, causal relations [207] capture
directed influence (e.g., “The resignation was caused by the
scandal”). Signals range from explicit connectives (“because,”
“therefore”) to implicit world knowledge, requiring models to
handle directionality and complex chains of causes and effects.
Third, compositional relations [208] address part–whole struc-
tures, where sub-events (e.g., “a signing ceremony”) instantiate
components of a super-event (e.g., “a peace negotiation”),
which supports the creation of hierarchical narratives.
Nevertheless, extracting these event relations remains chal-
lenging. It typically requires deep semantic understanding and
discourse-level analysis, including long-range dependencies,
rhetorical structure, and background knowledge. Moreover,
relation extraction interacts bidirectionally with earlier stages:
reliable triggers, arguments, and coreference substantially im-
prove relation inference, while relational constraints can, in
turn, regularize and correct lower-level decisions.
B. Multimodal EE
The multimodal EE tasks can be divided into four cate-
gories: Visual EE, Video EE, Audio/Speech EE, and Cross-
modal EE, as shown in Fig. 4.
Key Difference: Linguistic Context vs. Grounding
•Text-based EE relies on linguistic cues and discourse
context to disambiguate meaning.
•Multimodal EE requires grounding, which aligns sym-
bolic roles with physical regions or temporal segments.
1) Visual EE:Extract events and their semantic roles di-
rectly from still images, without relying on accompanying text
[92]. This task, also known as situation recognition or visual
semantic role labeling, requires predicting both the event

6 ARXIV PREPRINT
Role:
Agent:
Source:
Destination:
Place:Value
Boy
Cliff
Water
Lake
Type:
Trigger:
Product:
Time:Launch
Released
iPhone12
2020
EventEvent: jumping
Arg0 (deflector):
Arg1 (thing deflected):
Scene:Woman with shieldVerb: deflect (block, avoid)
Visual
Argument
Textual
ArgumentsDeploy
[Movement, Transport]
Vehicle truck
truck Vehicle
Agent
Artifact soldiersUnited StatesLast week, U.S. Secretary of State Rex Tillerson visited [Movement.Transport] Ankara, the 
first senior administration official to visit [Movement.Transport] Turkey, to try to seal a 
deal about the battle [Conflict.Attack] ……
Event
boulder
City park
a) Visual EE b) Video EE
c) Audio EE d) Cross-modal EE
Fig. 4. Task illustration for different multimodal EE tasks.
trigger (verb) (e.g., cutting, riding) and the associated roles
(e.g., agent, tool, object, place), grounding each role to specific
visual regions. For instance, given an image of a woman
cutting a tomato with a knife in a kitchen, the system should
identify the event as cutting, with agent=woman, tool=knife,
item=tomato, and place=kitchen.
2) Video EE:The task of Video EE is to detect and extract
events from a given video clip. We follow the definition of
[209]. Given an input video sequenceV={f i}F
i=1withF
frames, the objective of Video EE is to automatically construct
a set of structured eventsE={e 1, e2, . . . , e m}that describe
the actions and their participants within the video. Formally,
each evente∈ Eis represented as,
e= 
v,⟨r0, a0⟩,⟨r1, a1⟩, . . .
,
wherev∈ Vis an action predicate selected from a predefined
verb setV, and each pair⟨rk, ak⟩associates a semantic role
rk∈ R(v)with a corresponding argumentak. The role
setR(v)defines the expected participants for the actionv
(e.g., AGENT, TARGET, LOCATION, etc.). For instance, if the
action verb isknock, a possible extracted event could be:
⟨AGENT,gray bull⟩,⟨TARGET,person in gray hoodie⟩, and
⟨PLACE,ground⟩. In this way, Video EE transforms raw video
into a set of symbolic event representations, enabling down-
stream reasoning about “who did what, to whom, and where”
in the visual domain.3) Audio/Speech EE:Following the definition of [100],
we define the Audio/Speech EE task as: given a speech
signal represented as a sequence of acoustic framesS=
(f1, f2, . . . , f U), the goal is to identify and structure events
expressed within the audio stream. We assume a predefined
set of event typesEand a corresponding set of argument
rolesR. Each extracted event record is represented with four
components: 1) an event typeϵ∈ E, 2) the corresponding
event trigger in the audio (e.g., a word or phrase aligned
with speech), 3) an argument roler∈ R, and 4) the event
argument itself. Compared with text-based event extraction,
Audio/Speech EE faces unique challenges such as noisy
signals, disfluencies, speaker variability, and the necessity to
align acoustic features with semantic roles. The task therefore
requires both robust speech recognition and accurate mapping
from speech content to structured event representations.
4) Cross-modal EE:Cross-modal EE aims to extract struc-
tured events by jointly leveraging information from multiple
modalities, such as text, vision, and audio [21]. Formally,
given multimodal inputsM={m(t), m(r)}, wherem(t)is
the text andm(r)corresponds to another modality (i.e., image,
video, or speech), the goal is to identify event typesϵ∈ E,
event triggers, and argument-role pairs⟨r, a⟩across modal-
ities. Different modalities provide complementary evidence:
for instance, visual cues may reveal objects and actions, text
can supply explicit semantic triggers, and audio can indicate
speaker roles or affective states. The key challenges in Cross-

LIet al.: EVENT EXTRACTION 7
modal EE include aligning heterogeneous signals, grounding
arguments consistently across modalities, and reasoning about
missing or conflicting evidence. By fusing multimodal infor-
mation, Cross-modal EE can improve event typing accuracy,
enhance argument grounding, and support more robust infer-
ence compared to unimodal approaches.
Key Distinction: Scope & Boundary
•Event Extraction (EE) focuses on high-level, schema-
defined occurrences with specific semantic roles.
•It is distinct from general semantic parsing (like SR-
L/AMR) and low-level signal processing tasks.
III. METHODOLOGY
Event Extraction (EE) systems fundamentally rely on en-
coding methods to transform unstructured text into struc-
tured representations that are amenable to computation. The
evolution of these methods reflects the broader history of
Natural Language Processing (NLP), progressing from manu-
ally crafted linguistic rules to sophisticated, data-driven deep
learning architectures. This section provides a comprehen-
sive overview of this evolution, categorized into four major
paradigms: rule-based approaches, traditional machine learn-
ing, deep learning, and approaches based on large language
models (LLMs). Each paradigm offers a distinct approach
to capturing the lexical, syntactic, and semantic features
necessary to identify event triggers and their corresponding
arguments. To provide a quantitative comparison, we summa-
rize the performance of representative methods across these
paradigms in Table I.
A. Rule-based Methods
The foundational paradigm for event extraction was built
upon rule-based systems. As shown in Fig. 5a, these ap-
proaches leverage handcrafted patterns, linguistic heuristics,
and syntactic structures to identify event components. Charac-
terized by high precision and interpretability, they often require
significant domain expertise and manual effort, which can limit
their recall and adaptability to new domains. The development
in this area shows a clear trajectory from domain-specific,
syntax-heavy systems towards more generalized frameworks
and hybrid models that combine rules with statistical methods.
Rule-based methods represent the foundational approach
to event extraction, relying on manually crafted patterns and
heuristics. Early work in the biomedical domain by [210]
utilized syntactic dependency heuristics to identify biological
events. This dependency-based paradigm was further explored
by [211], which framed event extraction as a dependency
parsing task. As the field matured, the focus shifted towards
creating more generalised and adaptable rule-based systems.
For instance, Dutkiewicz et al. [161] introduced a system that
could automatically learn rules, reducing the need for extensive
manual effort. Similarly, Valenzuela et al. [212] developed a
framework applicable across various domains by abstracting
event structures. The application of rule-based methods alsoexpanded to diverse areas like socio-political news analysis
[213] and artificial social intelligence [214]. To overcome the
inherent rigidity of purely rule-based systems, hybrid models
emerged. These models combine rule-based components with
machine learning to improve performance and adaptability.
Kova et al. [215] demonstrated the efficacy of such a hybrid
approach for extracting information from clinical texts. This
trend was continued by later research, which integrated rules
with machine learning for general text [216], [217], showcas-
ing the enduring relevance of structured linguistic knowledge
in the machine learning era. Most recently, Guda et al. [218]
continued to refine the core rule-based extraction techniques
from natural language.
B. Classic ML Methods
The machine learning era for event extraction shifted the
focus from manually crafting rules to training statistical mod-
els on annotated data. As presented in Fig. 5a, this paradigm
was characterized by intensive feature engineering, where
researchers designed a rich variety of lexical, syntactic, and
semantic features to feed into classifiers such as Support
Vector Machines (SVMs) and Maximum Entropy models. A
key evolutionary trend during this period was the recognition
that sentence-level information is often insufficient, leading to
innovative methods that incorporated document-level context
to resolve local ambiguities.
Early work established the viability of treating event ex-
traction as a series of classification subtasks. Ahn et al. [5]
conceptualized the task as a pipeline of stages, including
anchor identification and argument identification, and used
machine-learned classifiers with a rich feature set derived from
lexical information and syntactic parsers. Naughton et al. [219]
focused on merging event descriptions from multiple news
sources, developing methods to group sentences referring to
the same event. To improve performance, researchers began
incorporating wider contexts. For example, Ji et al. [159]
and Liao et al. [7] both leveraged cross-document inference
to refine event extraction. Concurrently, efforts were made
to apply these techniques to real-time systems for crisis
monitoring [220] and social media analysis [221].
The focus then shifted towards richer feature engineering
and sophisticated modeling. Patwardhan et al. [222] developed
a unified model incorporating both phrasal and sentential
evidence. Miwa et al. [223] demonstrated the benefits of using
a rich feature set for complex event classification. Hong et al.
[224] proposed cross-entity inference, a method motivated by
the observation that entities of the same type often participate
in similar events. These methods were successfully adapted
to specialized domains such as clinical text [225], biomedical
literature [226], and social media for detecting adverse drug
events [227]. More recently, machine learning approaches have
been applied to new languages like Arabic [228] and enhanced
with visualization techniques to improve model interpretability
[229].
C. Deep Learning Methods
As illustrated in Fig. 5b, The advent of deep learning
marked a pivotal paradigm shift in event extraction, moving

8 ARXIV PREPRINT
Texual Input
a) Rule-based & Classic ML b) Deep Learing Methods c) Generative Methods d) LLM-based MethodsML AlgorithmSVM
CRF Random ForestEnsemble LearningCRFArg-0 Trigger Arg-1 O O B-Trigger B-Arg
Linguistic Rule &
Feature EngineeringPOS TaggingDependency 
Parsing
Constituency
ParsingPosition
Head Word
Texual InputEmbeddingWord2vec Glove BERT<Acquisition> acquired <Arg> 
Elon Musk </Arg> <Role>Bu 
yer</Role><Arg>Twitter</Ar
g> <Role> Artifact </Role>
Graph Convolution Network
ML TransformersCNN RNN / GRU / LSTM
Attention
Texual InputSeq2seq Backbone
PromptLineared Event Structure
BART T5 GPT PaLMLarge Language Models
In-context LearningCoT & Reasoning AlignmentStructured Event Output
GPT Llama
DeepSeekQwen
Gemini
Prompt Texual InputStructural Event  GenneratorCopy Mechanism
Constrained DecodingClassifier{ "event_type": "Acquisition",
  "trigger": "acquired",
  "arguments": {
    "Buyer": "Elon Musk",
    "Artifact": "Twitter",}}Softmax
DL Algorithm
Fig. 5. Different Methods for Event Extraction.
from manual feature engineering to automatic representation
learning. Neural networks, with their ability to learn hierarchi-
cal and distributed features from data, began to systematically
outperform traditional machine learning models. This section
chronicles the progression of deep learning architectures,
starting with Convolutional and Recurrent Neural Networks,
the revolutionary impact of Transformer-based pre-trained
language models, and the explicit modeling of syntax with
Graph Neural Networks.
1) CNN-based Methods:CNN-based models were among
the first deep learning approaches for event extraction, ef-
fective at capturing local n-gram features around potential
triggers. Nguyen et al. [10] applied CNNs for event trigger
detection and showed that a CNN trained on one domain
can be adapted to another via domain adaptation techniques.
By leveraging pre-trained embeddings and fine-tuning on a
target domain, their model achieved robust event detection
performance across newswire and novel domains, highlighting
CNNs’ ability to learn transferable feature representations.
Further improving trigger and argument representation, Chen
et al. [9] introduced a Dynamic Multi-Pooling CNN (DM-
CNN) that segments the sentence by event arguments and
applies multiple pooling operations. This architecture captures
features for different sentence segments (before, between, after
arguments) separately, which yielded state-of-the-art results by
more precisely localizing informative context for each argu-
ment role. To jointly extract triggers and arguments, Zhang
et al. [124] proposed a skip-window CNN that can handle
non-contiguous context. Their model uses wider convolutional
windows that skip certain distances to capture global sentence
features, and it extracts triggers and arguments simultaneously.
By expanding the receptive field in the convolution, this joint
model better grasped dependencies between distant trigger-
argument pairs, improving both trigger classification and argu-
ment role labeling. CNNs were also successful in specializeddomains: Li et al. [125] developed a Parallel Multi-Pooling
CNN (PMCNN) for biomedical event extraction. PMCNN
applies multiple pooling layers in parallel on different parts
of the dependency tree or sentence, thereby capturing diverse
semantic features (e.g., one pool focusing on trigger context,
another on argument context). This parallel design improved
extraction of complex biomedical events (like protein mod-
ifications) by combining information from multiple context
windows. Finally, researchers explored combining CNNs with
iterative self-training. Kodelja et al. [230] integrated a CNN-
based trigger detector with a bootstrapping approach to incor-
porate global context. They iteratively refined a global con-
text representation (aggregated from the document or related
documents) and fed it into the CNN, allowing the model
to use broader topical information. This bootstrapped CNN
achieved higher precision on event detection by correcting
errors using the larger context (for example, reinforcing that
if earlier sentences mention an earthquake, a later sentence’s
“magnitude” likely relates to that event).
2) RNN-based Methods:RNNs, particularly Long Short-
Term Memory (LSTM) and Gated Recurrent Unit (GRU)
networks, became a dominant architecture due to their natural
ability to model the sequential nature of text and capture
long-range dependencies. Nguyen et al. [12] proposed a joint
framework using bidirectional RNNs to predict event triggers
and arguments simultaneously, mitigating error propagation by
using novel memory features. Chen et al. [231] introduced a
framework using bidirectional LSTMs with a dynamic multi-
pooling layer and a tensor layer to explore the interaction
between candidate arguments and predict them jointly.
To address the limitations of local context, Duan et al. [117]
developed a document-level RNN (DLRNN) model capable
of automatically extracting cross-sentence clues to improve
sentence-level event detection. Researchers also integrated
explicit syntactic information into RNNs. Sha et al. [126]

LIet al.: EVENT EXTRACTION 9
enhanced the RNN architecture with “dependency bridges”
to carry syntactically related information when modeling each
word. Similarly, Li et al. [232] introduced a knowledge-driven
Tree-LSTM framework for biomedical event extraction, which
explicitly encoded dependency structures and external knowl-
edge from ontologies. To improve feature learning, Zhang
et al. [233] proposed a multi-task learning framework that
incorporated a bidirectional neural language model into the
event detection model, allowing it to extract more general
patterns from raw data.
3) Transformer-based Methods:Transformer architectures,
with their self-attention mechanism and ability to capture
long-range dependencies, have been widely adopted for event
extraction, especially as they can be pre-trained on large
corpora and fine-tuned for extraction tasks. Early explorations
integrated transformers with event structures. Yang et al. [119]
examined how pre-trained transformers (like BERT) could
be leveraged for event extraction. They proposed separating
argument role prediction by roles to handle overlapping argu-
ments and also experimented with using the model to generate
event descriptions from structured data. Wadden et al. [118]
proposed DyGIE++, a unified multi-task framework using
contextualized span representations to jointly perform entity,
relation, and event extraction. The generative capabilities of
Transformer-based models also enabled new task formulations.
For instance, Zheng et al. [62] introduced an end-to-end frame-
work that generates an entity-based directed acyclic graph,
and Lu et al. [29] proposed a paradigm that directly gener-
ates structured event records from text. To further improve
performance, researchers integrated syntactic knowledge into
Transformers, as seen in the GATE model [176], or designed
relation-augmented attention mechanisms like RAAT to model
argument dependencies in document-level extraction [123].
Finally, Scaboro et al. [234] provided a valuable benchmark
by conducting an extensive evaluation of various Transformer
architectures for extracting adverse drug events.
Building on the Transformer, BERT and its variants have
been specifically adapted for event extraction. A key inno-
vation was reframing the task itself, such as recasting it
as a machine reading comprehension (MRC) problem where
a BERT-based model answers questions to identify event
details [23]. To improve performance, researchers developed
specialized frameworks like CasEE, which uses a cascade
decoding strategy to handle complex overlapping events [122].
Other works focused on adapting the BERT architecture itself,
for example, by modifying the self-attention mechanism with
mask matrices for Chinese text in MABERT [178], or by
explicitly integrating event schema annotations into the model
input in EABERT [177]. These advanced models have been
successfully applied to critical real-world scenarios, such as
extracting information related to COVID-19 [192], and have
been adapted for new languages like Arabic through the
creation of dedicated corpora and novel modeling approaches
[235].
4) Graph-based Methods:Graph neural networks (GNNs)
have been leveraged to capture structured information (like
dependency trees or event graphs) for event extraction. These
models propagate node representations through edges, whichis intuitive for modeling relationships between triggers and ar-
guments. Nguyen et al. [15] first applied Graph Convolutional
Networks (GCNs) to event detection. This approach modelled
syntactic dependencies as graphs and introduced argument-
aware pooling to emphasise potential argument words, improv-
ing trigger classification by leveraging dependency structures
and argument clues. Liu et al. [16] developed a model that
performs multi-order graph convolution, meaning it not only
considers direct neighbors in the dependency graph but also
second-order and third-order connections.
Subsequent work refined the GNN approach by incorpo-
rating more complex structural information. Yan et al. [236]
extended GCNs to model and aggregate multi-order syntactic
representations, moving beyond just first-order dependency
relations. Lai et al. [237] introduced a novel gating mech-
anism into GCNs to filter noisy information based on the
trigger candidate and incorporated syntactic importance scores.
Balali et al. [238] proposed a joint framework that applies
GCNs along the shortest dependency path between words,
eliminating irrelevant context. Ahmad et al. [176] introduced
GATE, a Graph Attention Transformer Encoder that fuses
structural information from dependency parses into a self-
attention mechanism to improve cross-lingual event extraction.
More recently, Wan et al. [239] built a multi-channel GAT
that processes multiple types of relations (dependency, co-
reference, etc.) in separate channels and then hierarchically
combines them. This approach effectively handles open-event
extraction by not assuming predefined event types and by
drawing together various relational cues through graph atten-
tion. Zhang et al. [240] proposed a multifocal graph-based
scheme for extracting topic events, essentially identifying a
representative event that characterizes a topic from a set of
documents.
D. LLM-based Methods
Current methodologies leveraging LLMs for event extrac-
tion can be broadly categorized into six paradigms:Instruc-
tion Tuning,In-context Learning (ICL),Chain-of-Thought
(CoT)reasoning,Data Augmentation,Multi-agentframe-
works, andMultimodal LLMs. A schematic overview of
these approaches is illustrated in Fig. 5d.
Key Difference: Representation vs. Instruction
•Deep Learning approaches focus on learning complex
feature representations to map inputs to labels.
•LLM-based approaches emphasize instruction follow-
ing and reasoning to synthesize structures directly.
1) Instruction-tuned Models.:Instruction-tuned approaches
adapt LLMs to specific event extraction tasks and schemas by
directly training them to follow task instructions, thereby pro-
viding a promising and cost-effective solution. For example,
Wang et al. [64] proposed DeepStruct, which unifies diverse
structure prediction tasks into triple generation via structure
pretraining. Trained on task-agnostic and multi-task corpora,
it enhances structural understanding and supports zero-shot

10 ARXIV PREPRINT
TABLE I
PERFORMANCE COMPARISON OF SEVERAL REPRESENTATIVE METHODS ONEVENTEXTRACTION TASKS.
Category ModelTrigger Id (%) Trigger Cls (%) Argument Id (%) Argument Cls (%)
P R F1 P R F1 P R F1 P R F1
Classic
MLCross-Event [7] - - - 68.7 68.9 68.8 50.9 49.7 50.3 45.1 44.1 44.6
Cross-Entity [224] - - - 72.9 64.3 68.3 53.4 52.9 53.1 51.6 45.5 48.3
JointBeam [8] 76.9 65.0 70.4 73.7 62.3 67.5 69.8 47.9 56.8 64.7 44.4 52.7
PSL [241] - - - 75.3 64.4 69.4 - - - - - -
Deep
LearningDMCNN [9] 80.4 67.7 73.5 75.6 63.6 69.1 68.8 51.9 59.1 62.2 46.9 53.5
JRNN [12] 68.5 75.7 71.9 66.0 73.0 69.3 61.4 64.2 62.8 54.2 56.7 55.4
dbRNN [126] - - - 74.1 69.8 71.9 71.3 64.5 67.7 66.2 52.8 58.7
ANN-FN [242] - - - 79.5 60.7 68.8 - - - - - -
ANN-AugATT [201] - - - 78.0 66.3 71.7 - - - - - -
JMEE [16] 80.2 72.1 75.9 76.3 71.3 73.7 71.465.668.4 66.8 54.9 60.3
Pretrained
Language
ModelPLMEE [119] 84.8 83.7 84.2 81.0 80.4 80.7 71.4 60.1 65.3 62.3 54.2 58.0
Joint3EE [243] 70.5 74.5 72.5 68.0 71.8 69.8 - - - 52.1 52.1 52.1
GAIL-ELMo [244] 76.8 71.2 73.9 74.8 69.4 72.0 63.3 48.7 55.1 61.6 45.7 52.4
DYGIE++ [118] - - - - - 69.7 - - 55.4 - - 52.5
BERT-QA [26] 74.3 77.4 75.8 71.1 73.7 72.4 58.9 52.1 55.3 56.8 50.2 53.3
and multi-task transfer. Wang et al. [38] proposed InstructUIE,
a unified information extraction framework using instruction
tuning that reformulates IE tasks in a text-to-text format.
Evaluated on the expert-written IE INSTRUCTIONS bench-
mark, it matches BERT in supervised settings and surpasses
GPT-3.5 and prior SOTA in zero-shot performance. Li et
al. [245] proposed KnowCoder, which converts IE schemas
into Python-style class representations and uses code pre-
training with instruction tuning to improve schema under-
standing and enable universal information extraction. Wei et
al. [137] proposed the first explainable complex temporal
reasoning task for predicting future event times with reasoning
explanations. They built the 26K-sample ExpTime dataset
and released TimeLlaMA, achieving state-of-the-art results in
event prediction and explanation generation. Sainz et al. [147]
proposed GoLLIE, which fine-tunes large language models
to follow complex annotation guidelines, thereby significantly
improving performance on zero-shot information extraction
tasks.
Furthermore, recent studies have introduced reinforcement
learning and specific optimization strategies to further enhance
extraction capabilities. Qi et al. [136] introduced ADELIE,
an IE-aligned large language model trained with the high-
quality IEInstruct dataset via instruction tuning and DPO.
Its variants, ADELIESFT and ADELIEDPO, achieve state-
of-the-art performance across multiple IE benchmarks. Xu
et al. [140] proposed ChatUIE, a unified information ex-
traction framework based on ChatGLM, which integrates
reinforcement learning and generation constraints to effec-
tively improve information extraction performance. Hong et
al. [138] proposed a reinforcement learning–based question
generation method, RLQG, which leverages four evaluation
criteria to produce high-quality, generalizable, and context-
dependent questions, thereby enhancing the performance of
QA-based event extraction. Zhang et al. [246] proposed UL-
TRA, a framework that improves event argument extractionby hierarchically reading document segments and refining
candidate sets, while incorporating LEAFER to better locate
argument boundaries. Cai et al. [247] built DivED, an au-
tomatically generated dataset with diverse event types and
definitions to improve models’ understanding and zero-shot
event detection. Fine-tuning LLaMA-2-7B on DivED yields
significant gains over traditional methods and even surpasses
GPT-3.5 on three open benchmarks. Hu et al. [141] proposed
LLMERE, which reformulates Event Relation Extraction as a
QA task. It extracts all related events simultaneously and uses
partitioning with rationale generation to enhance efficiency,
coverage, and interpretability. Srivastava et al. [142] integrated
human- and machine-generated event annotation guidelines
into LLMs for event extraction, improving performance in
both full-data and low-resource settings, especially for cross-
schema generalization and low-frequency event types.
2) In-context Learning.:ICL-based approaches rely on
providing a few-shot context within prompts, enabling LLMs
to infer structured information without explicit parameter
updates. For example, Li et al. [143] systematically evaluated
ChatGPT on seven fine-grained IE tasks, finding it underper-
forms in standard IE but excels in OpenIE and provides high-
quality explanations. Ma et al. [144] proposed combining SLM
filtering with LLM reranking, using small models for efficient
sample filtering and large models for reranking difficult cases,
effectively integrating both strengths. Pang et al. [145] pro-
posed Guideline Learning (GL), which automatically generates
and retrieves task guidelines to enhance in-context information
extraction and mitigate underspecified task descriptions. Wang
et al. [140] proposed Code4Struct, reformulating structured
prediction as code generation. Using event argument extrac-
tion, it maps text to class-based event–argument structures,
leveraging type annotations and inheritance to integrate ex-
ternal knowledge and constraints. Li et al. [140] proposed
converting IE outputs into code and using Code-LLMs for
code-based extraction, achieving consistent gains over IE-

LIet al.: EVENT EXTRACTION 11
TABLE II
PERFORMANCE COMPARISON ONVISUAL ANDMULTIMODAL SETTINGS. THE TASKS INCLUDEEVENTTRIGGEREXTRACTION ANDARGUMENTROLE
EXTRACTION. BEST RESULTS ARE BOLDED.
MethodVisual (Image-only) Multimodal
Trigger Argument Trigger Argument
P R F1 P R F1 P R F1 P R F1
Flat [21] 27.1 57.3 36.7 4.3 8.9 5.8 33.9 59.8 42.2 12.9 17.6 14.9
WASE [21] 43.1 59.2 49.9 14.5 10.1 11.9 43.0 62.1 50.8 19.5 18.9 19.2
CLIP-EVENT [248] 41.3 72.8 52.7 21.1 13.1 17.1 - - - - - -
UniCL [249] 54.6 60.9 57.6 16.9 13.8 15.2 44.1 67.7 53.4 24.3 22.6 23.4
CAMEL [36] 52.1 66.8 58.5 21.4 28.4 24.4 55.6 59.5 57.5 31.4 35.1 33.2
UMIE [44] - - - - - - - - 62.1 - - 24.5
MMUTF [250] 55.1 59.1 57.0 23.6 18.8 20.9 47.9 63.4 54.6 39.9 20.8 27.4
X-MTL [46] 73.1 70.3 71.7 33.2 31.3 32.2 78.3 57.3 66.2 40.3 42.6 41.4
Qwen2VL-7B [251] 69.11 65.38 64.83 27.30 27.63 27.31 83.77 73.14 70.64 47.13 42.70 43.04
MKES [252] 69.32 65.77 65.30 27.30 27.63 27.31 89.82 87.70 87.39 48.11 43.89 44.25
specific and prompt-based LLMs across seven benchmarks.
Complementary to these format-oriented approaches, recent
research focuses on optimizing demonstration selection and
integrating hybrid workflows to ensure stability and accuracy.
He et al. [253] proposed DRAGEAE, which integrates a
knowledge-injected generator with a demonstration retriever
to address format inconsistency, multi-argument handling, and
contextual deviation in generative event argument extraction.
Kan et al. [254] reformulated event extraction as multi-turn
dialogues, guiding LLMs to learn event schemas and generate
structured outputs. They also introduced an LLM-based data
generation method that significantly improves performance on
long-tail event types. Zhu et al. [148] proposed LC4EE, which
combines SLMs’ event extraction with LLMs’ error correc-
tion via auto-generated feedback, allowing LLMs to refine
SLM predictions effectively. Zhou et al. [255] proposed HD-
LoA prompting for document-level event argument extraction,
combining heuristic-driven and analogy-based prompts to help
LLMs learn task heuristics from few examples and generalize
via analogical reasoning. Fu et al. [256] proposed TISE,
an example selection method that uses semantic similarity,
diversity, and event correlation, applying Determinantal Point
Processes to choose contextual examples effectively. Bao et
al. [146] proposed GCIE, a unified information extraction
framework that combines LLMs and SLMs in a two-stage
pipeline to handle noise, abstract label semantics, and varied
span granularity. Zhang et al. [257] found that large language
models exhibit spurious associations in information extraction
and proposed two strategies, forward label extension and
backward label validation, to leverage the extended labels.
Wei et al. [258] proposed CAT (Choose-After-Think), a two-
phase “Think–Choose” framework that mitigates LLM pref-
erence traps in event argument extraction, greatly enhancing
unsupervised EAE. In zero-shot settings, a local 7B model
with CAT matches DeepSeek-R1 performance while cutting
time costs. Liao et al. [149] proposed RUIE, a retrieval-based
unified IE framework that combines LLM preferences with
a keyword-enhanced reward model and trains its retriever via
contrastive learning and knowledge distillation for efficient in-context generalization.
3) Chain-of-Thought (CoT).:Inject intermediate reasoning
into the decoding. Improve step-by-step structure induction.
Wei et al. [34] proposed ChatIE, a prompt-based zero-shot
IE method that reformulates extraction as multi-turn QA and
leverages ChatGPT across multiple datasets and languages.
Ma et al. [150] proposed STAR, which uses LLMs to gen-
erate high-quality training data from limited seed examples,
boosting performance in low-resource IE tasks like event and
relation extraction. Chen et al. [151] analyzed LLMs’ weak-
nesses in event relation reasoning and proposed generative,
retrieval-based, and fine-tuning methods with a new dataset,
LLM-ERL, significantly improving logical consistency and
performance in event relation extraction. Shuang et al. [168]
proposed EAESR, a document-level event argument extraction
method using guided summarization and reasoning to leverage
LLMs’ emergent abilities for key feature extraction and cross-
event association.
4) Multi-agent Methods:Wang et al. [152] proposed DAO
(Debate as Optimization), a multi-agent system that refines
LLM event extraction outputs through debate without tuning. It
integrates DRAG for diverse retrieval and AdaCP for filtering
unreliable answers, significantly narrowing the gap between
tuning-free and supervised methods on ACE05 and CASIE.
Guan et al. [259] proposed MMD-ERE, a multi-agent debate
framework for event relation extraction, where cooperative and
confrontational debates with audience feedback enhance rela-
tion understanding. It outperforms baselines across multiple
ERE tasks and LLMs, demonstrating the effectiveness of the
debate mechanism.
5) Data Augmentation with LLMs:Wang et al. [153]
proposed TALOR-EE, which enhances low-resource event
extraction through targeted augmentation, negative sampling,
and back-validation, achieving notable gains in zero- and few-
shot settings. Jin et al. [154] proposed a schema-based data
augmentation method that leverages event schemas to generate
synthetic data, thereby alleviating the scarcity of annotated
data in event extraction tasks. Choudhary et al. [260] pro-
posed QAEVENT, a document-level event representation that

12 ARXIV PREPRINT
models events as question–answer pairs, removing predefined
role schemas and improving annotation efficiency and cover-
age. Zhao et al. [167] proposed MoRAG-FD, a biomedical
event causality framework using retrieval-augmented multi-
perspective data expansion and fine-grained denoising via
syntactic dependency–based weighting of irrelevant entity
pairs. Chen et al. [261] used LLMs as expert annotators
to expand event extraction datasets, aligning generated data
with benchmark distributions to mitigate data scarcity and
imbalance. Uddin et al. [262] proposed automatic question
generation methods for document-level event argument ex-
traction, producing both contextualized and uncontextualized
questions without human input. Combining the two notably
improves performance, especially for cross-sentence triggers
and arguments. Meng et al. [263] proposed CEAN, a con-
trastive event aggregation network with LLM-based augmen-
tation that uses event semantics and contrastive learning to
reduce noise and boost low-resource event extraction. Zhou
et al. [264] proposed BiTer, a bidirectional feature learning
method for zero-shot event argument extraction that jointly
learns contextual and labeled features while using LLMs to
generate pseudo-arguments to reduce context bias and improve
feature representation.
6) Multimodal LLMs (MLLMs).:Extend LLMs with vision
or audio inputs. Enable cross-modal trigger and role reasoning.
Bao et al. [265] proposed a multimodal Chinese event
extraction model that incorporates character glyph images
to capture intra- and inter-character morphological features.
Ma et al. [110] proposed the M-V AE task for extracting
and localizing abnormal event quadruples in videos. Their
Sherlock model, featuring Global-local Spatial-enhanced MoE
and Spatial Imbalance Regulator modules, effectively cap-
tures spatial information and significantly outperforms exist-
ing Video-LLMs on the M-V AE dataset. Bao et al. [266]
proposed a Literary Vision-Language Model for classical
Chinese event extraction that integrates literary annotations,
historical context, and glyph features to capture rich semantic
information. Zhang et al. [107] proposed the first framework
of Multimodal Universal Information Extraction (MUIE) and
developed a multimodal large language model, REAMO,
which can perform information recognition and fine-grained
grounding across various modalities.
We provide a schematic illustration of these multimodal EE
methodologies in Fig. 5.
IV. FORMULATIONS& DECODING
Event extraction typically involves multiple subtasks, such
as trigger word and argument extraction and their type clas-
sification, event relationship extraction, and event corefer-
ence resolution. Therefore, in this section, we categorize and
summarize existing event extraction methods based on their
decoding approach, including classification methods, sequence
labeling methods, span/pointer methods, table/grid methods,
and generation methods. The decoding diagrams of these
methods are shown in Fig. 6.A. Classification
Classification methods typically classify the types of entities
such as extracted trigger words and arguments, or classify
the relations between events. Some existing methods improve
model performance by incorporating lexical, syntactic, and
semantic knowledge into classification models [8], [12], [13],
[16], [267]. For example, Chen et al. [268] proposed a
knowledge-rich linguistic feature approach that not only effec-
tively utilizes character-level features but also incorporates the
results of zero pronoun resolution and noun phrase coreference
resolution, as well as features such as trigger word probability
and trigger word type consistency. These features capture
rich linguistic information from the character to the utterance
level. Furthermore, some methods transform trigger words and
arguments into graph or tree structures [62], [232], [269]–
[271]. For example, Han et al. [272] utilize structured pre-
diction to simultaneously extract event and temporal relations.
They model event relations as a graph structure that includes
three types of edges: event-relation consistency, transitivity
of temporal relations, and symmetry. Whether incorporating
additional knowledge or constructing a graph (tree) structure,
the ultimate goal of these methods is to better model trigger
words, arguments, and their relations for better classification.
B. Sequence Labeling
Sequence labeling methods treat event extraction as a se-
quence labeling problem. Unlike classification methods, se-
quence labeling methods can simultaneously extract trigger
words and arguments, as well as their types. However, such
methods have difficulty handling nested or multi-event sce-
narios. Typically, sequence labeling methods use a labeling
system similar to named entity recognition (NER) (such as
BIO tags) and use CRF for label prediction. Most existing
research uses weak supervision to generate more training
data to improve model performance [113], [115], [273]. For
example, Kan et al. [195] used queries, internet retrieval,
and LLM-based question answering to generate pseudo-labels
to form a weakly labeled dataset. They then used the auto-
matically labeled weakly labeled data to pre-train the event
extraction model, and finally fine-tuned the pre-trained model
on a manually labeled dataset. In addition, He et al. [274]
repeatedly embedded the trigger words of candidate events
into the gaps between each word in the sentence to form a new
extended sequence, thereby strengthening the trigger word’s
presence in the sequence. Ramponi et al. [112] designed a
multi-label-aware encoding strategy that represents the label
of each token as a triple and handles nested event structures
through relative position encoding. Lu et al. [114] introduced
customizable “structural preferences” (such as grammatical
rules and event patterns) to guide model learning.
C. Span/Pointer
Span-based methods extract events by predicting the spans
(start and end positions) of trigger words and arguments.
These methods can handle nested structures with multiple
trigger words or arguments within the same sentence, but

LIet al.: EVENT EXTRACTION 13
TRI ARG1 ARG0TRI ARG0 ARG1
Alibaba  laid off  5000 employees in 2023 Alibaba    laid    off  5000 employees in     2023
OutputInputB-ARG  B-TRI  I-TRI  O            O      O  B-ARG
Alibaba  laid off  5000  employees in 2023ARG0 TRI ARG1
Alibaba laid off 5000 employees in 2023
Alibaba  
laid
off
5000
...
2023Alibaba laid off 5000 employees in 2023
<argument >2023</argument> <argument > Alibaba  
</argument > <tigger > laid off </trigger >b) Sequence Labeling c) Span / Pointer
d) Table / Gride) Generationa) Classification
ARG0
TRI
ARG1
Fig. 6. Different Decoding Paradigms for Event Extraction.
their drawback is the need to enumerate all possible span
combinations, resulting in low efficiency. Existing span-based
methods fall into two main categories. One approach cap-
tures global relationships by constructing graph structures to
better predict spans [168], [179], [181]. For example, Yang
et al. [180] compress unrelated subgraphs and edge types,
integrates text span information, and highlights surrounding
events within the same document. Finally, they identify event
arguments by predicting edges between event trigger words
and other nodes. The other approach treats event extraction as
a question-answering task, predicting spans through question-
ing [23], [24], [120], [275], [276]. For example, Zhou et al.
[155] extract spans through a collaborative question-answering
process: identifying arguments by asking, “What plays [role]
in [event type]?” and identifying roles by asking, “What role
does [argument] play in [event type]?”
D. Table/Grid
The grid tagging method converts text into a two-
dimensional grid to predict events and their relations. The
advantage of this method lies in its ability to handle com-
plex scenarios such as nested and multiple events simultane-
ously, making it more time-efficient than span-based methods.
However, since each cell in the grid represents a word-pair
relationship, the size of the two-dimensional grid increases
exponentially with longer text, consuming a significant amount
of space. Cao et al. [32] pioneered the grid tagging method for
event extraction, designing two grids to extract trigger words
and arguments, respectively. Cui et al. [130] later developed
a dual-grid annotation scheme to capture correlations between
event parameters within and across events, thereby addressing
event causality. Pu et al. [132] improved on the method of Cui
et al. by designing a heterogeneous relational perception graphmodule and a multi-channel label enhancement module. Ning
et al. [131] initially used a single grid and four vertex labels
to simultaneously extract trigger words and arguments. Chen
et al. [277] later tackled event causality identification using
a single grid and six labels. Wan et al. [129] incorporated
the eType-Role1-Role2 composite labeling and a complete
subgraph-based decoding strategy into the grid tagging model
to handle more complex document-level event extraction.
E. Generation
Generative methods use generative models to generate struc-
tured event representations directly from text. The advantage
of these methods is that they don’t require decomposition
into subtasks, directly outputting event types, trigger words,
and arguments, thus avoiding error propagation between sub-
tasks. They can also handle a wide range of complex events.
However, generative methods often suffer from hallucination
problems, generating trigger words or arguments that don’t
exist in the text, or outputting results that don’t conform to the
structured representation. Existing methods typically construct
task-specific prompt templates, generate these templates, and
then decode the corresponding results [27], [29], [31], [128],
[133], [197]. With the rapid development of LLMs, LLM-
based prompt engineering and instruction tuning techniques
are gaining increasing attention among researchers [42]–[44],
[107], [278]. Furthermore, some work considers EE as a QA
task. In fact, QA can also be considered a form of prompt
engineering, essentially still directly generating events by con-
structing task-specific question and answer templates [279]–
[281]. Furthermore, Wang et al. [282] first applied generative
adversarial networks to open-domain event extraction. GANs
generate events by learning the mapping between document-
event distributions and event-related word distributions. Ren

14 ARXIV PREPRINT
a) Image Event Extraction b) Video Event Extraction c) Audio Event ExtractionInteraction / ModelingEvent Output
{Trigger: “Attact”, Arg-Object: “Tank 
(BBox [100, 200, 200, 300])”}
Image-Text MatchingGNN (Scene Graph + Dependency Tree)
Visual GroundingCross-Modal Attention
Visual Encoder
ImageFaster R-CNN
CLIP-ViT
ResNetTextual Encoder
RoBERTa
BiLSTM
BERT
CaptionInteraction / ModelingEvent Output
Timestamp: “00:15-00:20” Trigger: 
“arrive”, ARG-Place: “New York” 
Temporal Model (LSTM / Transformer)
Cross-modal Gating (Audio-Visual-Text)
Boundary Detection Head
Video Subtitle AudioVideo EncoderBackbone
[VideoMAE]
[C3D | I3D]
[TimeSformer]Audio Encoder
[VGGish]
[HuBert]
[Wav2Vec 2.0]Text
EncoderInteraction / ModelingEvent Output
{Trigger: “buy”, Arg-intent: “Purchase”,
Arg-object: “seafood”
Prosody-Semantic Fusion
Monotonic Alignment 
Contrastive Learning CTC Loss
Raw Waveform or SpectrogramAudio EncoderBackbone
[Wav2vec 2.0]
[HuBert] [Whisper]Text Encoder
[ASR Decoder]
[Phonetic Feature]
Fig. 7. Different Methods for Multimodal Event Extraction.
et al. [283] further improved the performance of generative
models using retrieval enhancement techniques. They designed
three retrieval methods: context-consistent retrieval (retrieval
of similar documents in the input space), pattern-consistent
retrieval (retrieval of similar tags in the tag space), and
adaptive hybrid retrieval.
Trade-off: Flexibility vs. Faithfulness
•Generative models support open schemas and end-to-
end extraction without cascading errors.
•However, they are prone to hallucination, creating plau-
sible but non-existent details lacking source evidence.
V. SYSTEMARCHITECTURES
Event extraction is a complex information extraction task
that inherently involves multiple interdependent subtasks, in-
cluding trigger word extraction, event type classification, ar-
gument identification, and argument role classification. The
relationships between these subtasks are critical, as the iden-
tification of an event trigger often dictates the schema for
potential arguments. In this section, we categorize existing
methodologies based on their architectural design choices.
These choices fundamentally determine how the dependencies
between subtasks are modeled and how information flows
through the system. We classify the architectures into three pri-
mary categories: Pipeline, Joint/Global, and One-Stage/Unified
models. The schematic overview of these architectures is
illustrated in Fig. 6.
A. Pipeline
The pipeline architecture represents the classical approach
to event extraction, where the problem is decomposed intoa series of distinct, sequential subtasks. In this framework,
separate models are trained for each stage, and the output of
one stage serves as the input for the next. Typically, the process
begins with trigger word identification, followed by event type
classification, and concludes with argument extraction and
role classification [7], [9], [24], [119], [159]. This modular
design offers the advantage of simplicity and interpretability,
as each sub-model focuses on a specific, constrained problem.
Furthermore, it allows for the flexible combination of different
algorithms for different stages.
However, the sequential nature of pipeline models intro-
duces two significant limitations. First, they suffer from severe
error propagation cascades. Since downstream models for
argument extraction rely entirely on the predictions of up-
stream trigger classifiers, any error in trigger identification or
classification is irreversible and inevitably leads to failures in
argument extraction. The downstream models have no mech-
anism to correct upstream errors. Second, this architecture
leads to information fragmentation. By treating subtasks as
statistically independent steps, the system fails to leverage the
semantic interactions between tasks. For instance, the presence
of specific arguments often provides strong contextual clues
that can help disambiguate the event type, but a strict pipeline
prevents this reverse information flow.
A representative example is the work of Liu et al. [23],
who formulated event extraction as a Machine Reading Com-
prehension (MRC) task. Their system operates in a strict
sequence: it first utilizes a special query token [EVENT] to
identify triggers and classify event types. Subsequently, it
constructs natural language questions based on the predicted
types to extract arguments using a BERT-based MRC model.
While effective, the dependence is unidirectional; if the initial
trigger detection fails, the subsequent question generation
relies on incorrect premises, rendering the argument extraction
step futile.

LIet al.: EVENT EXTRACTION 15
B. Joint/Global
To mitigate the limitations of the pipeline approach, Joint
(or Global) architectures model multiple subtasks within a
single unified framework. Unlike pipeline methods where
modules are trained in isolation, joint models optimize sub-
tasks simultaneously, typically sharing a common encoder or
feature representation layer. The core philosophy is that trigger
detection and argument extraction are mutually informative;
knowing the arguments can clarify the event type, and vice
versa. Consequently, these models aim to learn a global
representation that captures the dependencies between triggers
and arguments [8], [12], [16], [20], [126].
Joint architectures effectively alleviate the error propagation
problem by replacing the hard decisions of upstream pipeline
steps with soft, shared parameter optimization. This allows the
model to adjust its internal representations based on the loss
from all subtasks during training. For example, Graph Neural
Networks (GNNs) are frequently employed in this category to
explicitly model the structural connections between words. Xu
et al. [284] proposed a sophisticated heterogeneous graph net-
work that integrates sentence nodes and entity mention nodes.
Their model captures global interactions through various edge
types, including inter-sentence edges and cross-sentence men-
tion edges. By performing message passing over this graph, the
model achieves collaborative learning, where the features for
entity extraction, event detection, and argument role labeling
are jointly refined. This holistic view ensures that the extracted
events are structurally consistent and semantically coherent.
C. One-Stage/Unified
The One-Stage, or Unified, architecture represents a
paradigm shift in event extraction, moving away from task-
specific classification heads towards a fully end-to-end gener-
ation or prediction process. In this approach, the distinction
between trigger detection and argument extraction is mini-
mized. Instead of designing separate loss functions or modules
for different subtasks, unified methods employ a singular
architecture to output the complete event structure directly.
These models are predominantly generative, often leveraging
the capabilities of Pre-trained Language Models (PLMs) or
Large Language Models (LLMs) to transform event extraction
into a sequence generation problem [27], [44], [107], [121],
[133].
The primary advantage of unified frameworks is their
streamlined design and flexibility. By converting the extrac-
tion task into a text-to-structure or text-to-text format, these
methods avoid the complexity of designing specific network
components for each subtask. Furthermore, they are highly
adaptable to new event schemas through prompt engineering
rather than architectural re-engineering. Researchers can de-
sign task-specific prompts that instruct the model to linearize
the event structure. For instance, Lin et al. [45] introduced a
multi-perspective prompt design strategy. Their method does
not treat event extraction as a labeling task but rather queries
the model with various prompt arrangements for each event
type. This allows the model to leverage diverse semantic
perspectives to understand the context and generate the fullevent record in a single pass. By optimizing a single objective
function, unified models ensure that all components of an
event are generated in a globally optimal manner, reducing
the disconnect often seen in multi-stage systems.
Architecture: Pipeline vs. Unified
•Pipeline models decompose tasks sequentially, leading
to irreversible error propagation.
•Unified models treat EE as a single prediction, maxi-
mizing information flow across subtasks.
VI. REPRESENTATIONS& FEATUREENGINEERING
A. Lexicon
Early EE approaches relied heavily on curated lexical
resources such as trigger lists and gazetteers to bootstrap
event recognition in the absence of large annotated corpora.
These resources [169], [170] provided simple but effective
anchors for detecting candidate triggers and arguments, often
combined with rule-based or pattern-matching systems to
enhance recall. As research progressed, lexicons were increas-
ingly integrated into statistical and neural pipelines as weak
supervision signals [171], supplying prior knowledge to guide
trigger identification or constrain argument spans. More recent
work [131], [172] explores hybrid strategies where lexicon-
derived cues complement distributed representations, showing
that gazetteer augmentation and dictionary-based triggers can
still provide robustness under domain shift or low-resource
scenarios. Recent studies [285], [286] continue to highlight the
auxiliary role of lexical priors within large-scale or domain-
adaptive EE frameworks, revisiting curated trigger word lists
as a lightweight yet interpretable supervision source.
B. Syntactic & Semantic
Syntactic and semantic structures remain central to event
extraction because they encode relational signals that guide
how triggers connect to arguments and roles. Early ap-
proaches [170], [287] integrated POS tags, constituency
boundaries, and dependency arcs into neural encoders, show-
ing that structural cues improve argument boundary precision
and help reduce error propagation in complex sentences.
Graph-based models [176], [288] further leveraged syntactic
trees to propagate evidence across spans, improving trig-
ger–role interactions and global coherence.
Beyond syntax, SRL and frame semantics provide pred-
icate–argument abstractions that naturally align with event
schemas. SRL-based supervision [289], [290] has been used as
auxiliary signals, constraints, and multi-task objectives, consis-
tently enhancing role disambiguation and generalization. Sev-
eral studies [65], [291], [292] design explicit role interaction
networks or fuse role semantics with span–relation modeling,
further improving robustness in multilingual and biomedical
settings. More recently, researchers have tested whether large
language models [261], [293] can act as annotators of semantic
roles, with mixed results depending on domain specificity.

16 ARXIV PREPRINT
Abstract meaning representation (AMR) introduces a
higher-level structural bias: nodes and edges normalize lexical
variation and facilitate reasoning beyond sentence boundaries.
Document-level event extraction [135], [179], [180] benefits
from AMR-enhanced models, including two-stream architec-
tures, prefix injection, and AMR-based link prediction. Dual-
level AMR [181], [294] injection and event pattern instance
graphs further extend this line by embedding AMR into
pretrained models for cross-sentence role resolution.
At the discourse level, structural cues such as coreference
and narrative flow help connect argument mentions scattered
across documents. Systems that explicitly project narrative
knowledge or exploit discourse signals [27], [295] enhance
argument completion and role consistency. Coupling coref-
erence with dependency and SRL information [26], [296],
[297] further enables implicit role reasoning and temporal
coherence across events. Beyond explicit discourse modeling,
graph-augmented architectures [298], [299] capture intra- and
inter-event dependencies, highlighting that explicit structure
remains indispensable for enforcing discourse-level coherence
and role resolution.
Building on these insights, recent research moves from
treating syntactic and semantic structures as standalone alter-
natives to pretrained embeddings toward integrating them as
complementary inductive biases. Syntax-guided latent variable
models [300], [301], reinforcement-driven boundary refine-
ment, and soft structural constraints inject syntax into PLMs
to improve calibration in long-range argument attachment
and low-resource scenarios. Hybrid approaches [302] further
demonstrate that schema-aware decoding and role relevance
reallocation can effectively combine structural signals with
contextual embeddings. This line of work [164], [303]–[305]
consistently shows that while pretrained embeddings capture
rich local semantics, explicit structural modeling is crucial for
supporting reliable generalization across both sentence- and
document-level extraction.
C. Knowledge Retrieval
Research on event extraction has increasingly emphasized
that local sentence context alone is often insufficient for robust
trigger identification or argument filling, particularly when
roles are implicit or span multiple sentences. To address this
limitation, a diverse body of work has explored the integration
of external knowledge sources—including structured ontolo-
gies, curated schemas, knowledge graphs, retrieved documents,
and commonsense resources—into extraction pipelines. Early
studies [169], [200], [306], [307] grounded arguments in
domain ontologies or legal schemas, showing that factual con-
straints help disambiguate types and enforce role consistency.
These ideas motivated retrieval-based architectures [308]–
[311] in which supporting passages or entity descriptions
are gathered dynamically from external corpora and then
fused with contextual encoders via attention or gating, thereby
reducing errors in argument completion.
Knowledge graphs [123], [312] quickly became central
in this paradigm: aligning mentions with KG nodes and
propagating signals along edges provides schema compatibil-
ity, while retrieving prototypical schema instances improveszero-shot transfer. Commonsense KGs such as ConceptNet
and ATOMIC have also been employed, with subgraph re-
trieval [23], [180], [298]. guiding implicit argument reasoning
and improving recall where surface cues are sparse. Retrieval
has likewise been coupled with cross-domain and cross-lingual
adaptation [165], [313], [314], where external knowledge mit-
igates data sparsity and enhances generalization in specialized
or low-resource scenarios.
More recent [279], [283], [316] advances integrate re-
trieval into generative pipelines. Retrieval-augmented gener-
ation (RAG) models combine dense retrievers with generative
decoders that construct structured frames under schema con-
straints, outperforming purely end-to-end baselines in open
or evolving domains. Other approaches [163], [166], [317]
retrieve schema descriptors, event exemplars, or temporal
patterns that regularize role inventories and support schema-
aware decoding. Large-scale pipelines [318], [319] automat-
ically build or synthesize event knowledge, offering broader
coverage and complementing retrieval modules in emerging
domains. In parallel, generative and continual learning frame-
works [164], [176], [303] highlight how retrieval can be
integrated with role reasoning and structural constraints for
greater robustness.
D. Pretrained Embeddings
Pretrained language models (PLMs) [17], [173], [174] make
event extraction largely a feature–engineering problem: con-
textual token and span representations are harvested from
a shared encoder and reused everywhere—trigger scoring,
argument identification, and role typing—so that one universal
embedding space underpins the entire pipeline. In sentence-
level EE, PLM features replace brittle lexical lists and hand-
built templates, yielding stronger trigger detectors and tighter
argument boundaries, with span pooling or biaffine heads
operating directly on contextualized vectors. Domain-tailored
PLMs [320], [321]. further align specialized terminology with
role inventories, improving calibration and recall without be-
spoke features.
Document-level models [135], [170], [176], [294] keep the
same recipe—one encoder, many heads—but add structure-
aware aggregation atop PLM features (graph layers, AMR-
aware prefixes, role-guided attention) to propagate evidence
across mentions while staying in the PLM embedding space.
Parameter-efficient conditioning [121], [133], [133], [322],
[323] then shapes these universal features toward schema con-
straints and label spaces with minimal finetuning, improving
few-shot trigger recognition and argument-role generalization.
More recently, generative adaptations [175], [253], [262],
[279], [324] still rely on the same pretrained representation
core—now used to produce frames directly under decoding
constraints—and can be fused with retrieval or demonstrations
when roles are implicit, reinforcing the view that PLM-derived
contextual embeddings function as universal features across
detection, role typing, and cross-sentence aggregation.
E. Visual Feature
Visual feature extraction for situation recognition and video
event extraction typically combines local, object-centric cues

LIet al.: EVENT EXTRACTION 17
TABLE III
REPRESENTATIVE EVENT EXTRACTION BENCHMARKS. “SCALE”IS A ROUGH RELATIVE INDICATOR(SMALL/MEDIUM/LARGE)TO HELP COMPARE
DATASET SIZES AT A GLANCE.ERE: EVENTRELATIONEXTRACTION;ECR: EVENTCOREFERENCERESOLUTION.
Name Task Domain Scale Language Notes
ACE05 [57] Event Extraction Newswire, broadcast Medium EN / ZH / AR trigger, argument
MUC-4 [91] Event Extraction Terrorism newswire Small EN trigger, argument
MA VEN [19] Event Extraction Wikipedia Large EN trigger, argument
RAMS [22] Event Extraction News Medium EN trigger, argument
WikiEvents [27] Event Extraction Wikipedia + news Medium EN trigger, argument
CASIE [25] Event Extraction Cybersecurity news Small–Medium EN trigger, argument
PHEE [59] Event Extraction Pharmacovigilance Small–Medium EN trigger, argument
DuEE [67] Event Extraction Open-domain news Medium–Large ZH trigger, argument
DuEE-Fin [315] Event Extraction Finance news Medium ZH trigger, argument
FewEvent [58] Event Extraction Open-domain Small–Medium ZH trigger, argument
Rich ERE [88] ERE News, forums Medium EN / ZH / AR trigger, argument, relation
MA VEN-ERE [19] ERE Wikipedia Large EN trigger, argument, relation
HiEve [83] ERE News (temporal) Small–Medium EN trigger, argument, relation
GENIA [162] ECR Paper Medium EN trigger, argument, coreference
MLEE [90] ECR Biomedical Paper Medium EN trigger, argument, coreference
KBP2017 [77] ECR News + forums Medium EN trigger, argument, coreference
with global, scene-level context. Early works mainly used
global CNN embeddings (e.g., VGG-16 [325] or ResNet-50
[326]) for situation recognition [92], [327]–[331], focusing on
capturing the holistic scene for verb prediction. Subsequent
grounded approaches explicitly incorporated local object de-
tectors such as Faster R-CNN [332] and RoI-pooled embed-
dings to jointly predict role labels and bounding boxes [94],
[333], while later extensions further exploited transformers
and collaborative attention to refine local-global fusion [334]–
[336]. Beyond static images, video-based models rely on both
global spatio-temporal encoders (e.g., I3D [99], SlowFast [98],
Video Swin Transformer [97]) to capture temporal dynamics
[93], [96], [337], [338] and local tracklet-based features that
trace entities and their evolving states over time [209], [338]–
[340]. More recent works extend these pipelines with structural
modeling, such as spatio-temporal scene graphs, visual analo-
gies, and causality-guided attention [94], [95], [330], [333],
[335], [336], [341]. Together, these efforts highlight a clear
trend: local features provide fine-grained entity grounding
[342] and argument role understanding, while global represen-
tations capture holistic context and temporal coherence, and
their integration has become the standard backbone for robust
visual event extraction.
VII. DATASET& EVALUATION
A. Text Datasets
We survey sentence- and document-level datasets for text-
based event extraction, covering resources annotated for event
detection, event relations, and event coreference.
ACE05: The ACE 2005 dataset [57] is the first large-scale,
multilingual corpus to systematically define and annotate event
tasks. It covers five main event types—Interaction, Movement,
Transfer, Creation, and Destruction—each with detailed sub-
types. Events are annotated with triggers, participants and theirroles, and attributes such as time, location, instrument, and
purpose, yielding a rich event structure. The dataset includes
English, Chinese, and Arabic newswire, broadcast, and news-
paper texts totaling about 500,000 words and thousands of
event instances.
Causal-TimeBank: Causal-TimeBank [82] extends the
TimeBank corpus with causality annotations across 183 news
documents. It adds annotations for 137 events to the original
6,811, yielding 318 causal links and 171 causal signals. Com-
patible with TimeML temporal tags, it supports research on
causality extraction and temporal–causal interaction analysis.
HiEve: The HiEve dataset [83] is a news event hierarchy
corpus with 100 documents, 1,354 sentences, and 33,000
tokens, with an average of 32 events per document. It manually
annotates spatiotemporal relations to form DAG-structured
hierarchies and serves as the first public resource for event
hierarchy modeling in extraction and summarization tasks.
ERE: The ERE dataset [88], developed by LDC under
DARPA DEFT, is a multilingual corpus in English, Chinese,
and Spanish. Light ERE provides simplified entity, relation,
and event annotations, while Rich ERE adds Realis labels and
Event Hoppers for event coreference. Covering newswire and
forums, it supports large-scale knowledge-base construction
and event extraction research.
KBP2017: The KBP 2017 dataset [77] includes both formal
news and informal forum texts, challenging models to handle
varied structures and noise. It supports tasks such as event
trigger detection, type classification, realis identification, and
document-level coreference, and serves as a key benchmark
for cross-domain event extraction.
MA VEN: MA VEN [19] is a large, human-annotated event
detection dataset from Wikipedia, with 4,480 documents,
118,732 event mentions, and 168 types—about twenty times
larger than ACE 2005. With broad coverage and realistic long-
tail, multi-event distributions, it is a key resource for general-

18 ARXIV PREPRINT
TABLE IV
MULTIMODAL EVENT EXTRACTION BENCHMARKS.
Name Modality Domain Scale Language Output
imSitu [92] Image General 126,102 English Verb, semantic roles and values
SWiG [94] Image General 126,102 EnglishVerb, semantic roles, values,
bounding boxes
V ASR [95] Image General 3,820 English Verb, semantic roles and values
VidSitu [93] Video Movie 126,102 English Verb and values
Grounded VidSitu [96] Video Movie 126,102 English Verb, values, bounding boxes
SpeechEE [100] AudioNews, medical, biology,
cybersecurity, movies5,240 human +
49,585 synthEN / ZH Event type, trigger, argument role
M2E2[21] Text + Image News 245 English Event type, trigger, image object
VOANews [106] Text + Image News 106,875 English Event type, trigger, image object
CMMEvent [108] Text + Image News 5,709 Chinese Event type, trigger, role
VM2E2[28] Text + Video News 852 EnglishArgument role type, entity type,
co-referential text event
MultiHiEve [109] Text + Video News 100,000 English Event-event relations
M-V AE [110] Text + Video General 1,680 English Natural language answer
MM Chinese EE [343] Text + SpeechBroadcast news,
newswire, weblog6,694 Chinese Event type
MUIE [107] Text + Image + Audio General 3,000 EnglishEvent type, entity label, argument,
role
domain event detection research.
ECB+: The ECB+ dataset [72] extends the EventCorefBank
(ECB) corpus with 502 additional news articles describing dif-
ferent instances of the same event types, increasing linguistic
diversity and representativeness for studying cross-document
event coreference in news texts.
For additional datasets and details, see Table III.
B. Multimodal Datasets
imSitu: The imSitu dataset was introduced in [92] to
support a structured understanding of what is happening in
an image. It contains over 500 activities, 1,700 roles, 11,000
objects, 125,000 images, and 200,000 unique situations. Each
image is annotated with one or more “situations” defined by
a verb and a set of semantic roles (e.g., Agent, Tool, Place)
filled by noun entities grounded in the image. Notably, imSitu
draws its role schema from linguistic resource FrameNet [344],
and its noun entities from ImageNet [345], enabling a rich yet
consistent mapping between visual entities and semantic roles.
SWiG: The SWiG (Situations With Groundings) dataset
[94] extends the imSitu situation recognition benchmark by
enriching it with explicit grounding annotations. While imSitu
provides event categories and semantic role labels, SWiG
further associates each argument with bounding-box annota-
tions in the image, thereby linking semantic roles to concrete
visual regions. In total, the dataset contains 126,102 images
spanning 504 verbs, 278,336 bounding boxes, and more than
11,538 entity categories, covering a wide variety of actions and
participants. This design enables models not only to recognize
what event is happening and which entities are involved, but
also to localize these entities in the visual scene, making it a
comprehensive benchmark for studying structured visual event
extraction and grounded situation recognition.
V ASR: The V ASR dataset [95] introduces a novel task
of visual analogy of situation recognition. Using the imSitusituation recognition annotations, the authors automatically
generated over 500,000 analogy instances of the form (A,
A’, B, B’), where the change from A to A’ (e.g., agent,
tool, or item role swapped) is mirrored by a corresponding
change from B to B’. The silver-labeled analogies were further
validated via crowdsourcing, yielding a gold-standard subset
of 3,820 high-confidence analogies.
VidSitu: The VidSitu dataset [93] is a large-scale bench-
mark for video situation recognition. It consists of about
29,000 ten-second clips sampled from around 3,000 movies,
each densely annotated at 2-second temporal intervals. Ev-
ery clip contains up to five salient events, and each event
is labeled with a verb sense together with semantic roles
grounded in video captions. The annotations also include entity
co-reference across event segments within a video, as well
as inter-event relations such as causation, contingency, and
reaction. On average, each clip contains 4.2 unique verbs and
6.5 distinct entities, covering a verb vocabulary of roughly
1500 unique verbs (with over 200 verbs having at least 100
examples) and diverse noun entities (5600 unique nouns with
350 nouns occurring in at least 100 videos).
For additional datasets and details, see Table IV.
C. Evaluation Metrics
We categorize evaluation methodologies into three primary
domains: Event Extraction (EE), Event Relation Extraction
(ERE), and Event Coreference Resolution (ECR).
a) Event Extraction Tasks:Current literature typically
evaluates EE performance under six distinct settings:
•Trigger Identification (Tri-I): the predicted trigger span
exactly matches the gold span.
•Trigger Classification (Tri-C): both the span and the
event type are correct.
•Argument Identification (Arg-I): the predicted argu-
ment span exactly matches the gold span.

LIet al.: EVENT EXTRACTION 19
•Argument Classification (Arg-C): both the argument
span and its role are correct.
•Head Classification (Head-C): evaluates the correctness
of predicted headwords (primarily for WIKIEVENT).
•Coref Match F1: credits predictions where the argument
belongs to the same coreference cluster as the gold entity.
To demonstrate the calculation protocol, we takeTrigger
Identification (Tri-I)as a representative example. LetT pred
denote the set of predicted trigger spans andT golddenote the
set of gold trigger spans. The Precision (P), Recall (R), and
F1-score (F 1) are calculated as:
PTri-I=|Tpred∩ Tgold|
|Tpred|,
RTri-I=|Tpred∩ Tgold|
|Tgold|,
F1Tri-I=2·P Tri-I·RTri-I
PTri-I+R Tri-I.(1)
Metrics for the other five tasks (Tri-C, Arg-I, etc.) follow the
same formulation, differing only in the matching criteria used
to define the intersection set (True Positives).
b) Relation Extraction:For Temporal, Causal, and
Subevent relation extraction, the field adopts the standard
micro-averagedP,R, andF 1scores. The calculation mirrors
Eq. 1, where a True Positive is defined by the specific relation
matching rules of the target dataset.
c) Event Coreference Resolution:Event coreference is
standardly evaluated using four cluster-based metrics:
•MUC[346]: A link-based metric measuring the minimum
number of missing or spurious links.
•B3[347]: A mention-based metric averaging precision
and recall over individual mentions.
•CEAF e[348]: An entity-based metric solving for the
optimal one-to-one alignment between gold and system
clusters.
•BLANC[349]: A metric designed to balance perfor-
mance between coreference and non-coreference links.
d) Semantic and Generative Evaluation:To address the
limitations of Exact Span Matching, particularly for Genera-
tive Event Extraction (GEE), recent surveys and benchmarks
have introduced semantic evaluation protocols. Common ap-
proaches include word-overlap metrics (e.g., BLEU, ROUGE),
embedding distance (e.g., BERTScore), and LLM-based eval-
uation agents.
Notably, Lu et al. [350] proposed RAEE, employing LLMs
to compute semantic-levelF 1via adaptive prompting. Simi-
larly, Lu et al. [49] introduced SEOE, a framework integrating
multi-source ontologies to enable semantic matching across
open-domain events. For argument evaluation, Fane et al.
[305] developed BEMEAE, which combines deterministic
text normalization with semantic similarity to better correlate
with human judgments.TABLE V
CATEGORIZATION OFTOOLS
Tool Category
DyGIE++ [118] EE Toolkit
SpERT [196] EE Toolkit
OneIE [20] EE Toolkit
CasEE [122] EE Toolkit
RESIN [197] EE Toolkit
UIE [127] EE Toolkit
InstructUIE [351] EE Toolkit
DeepKE [194] EE Toolkit
OneEE [32] EE Toolkit
DocEE [352] EE Toolkit
SPEED++ [353] EE Toolkit
OmniEvent [35] EE Toolkit
TextEE [160] EE Toolkit & Evaluation scripts
ACE [354] Evaluation scripts
TAC-KBP [355], [356] Evaluation scripts
ERE [11] Evaluation scripts
MA VEN-ERE [357] Evaluation scripts
BRAT [358] Annotation Platform
RESIN [197] [358] Annotation Platform
CollabKG [139] Annotation Platform
LFDe [195] Annotation Platform
Evaluation Gap: Exact Match vs. Semantic Utility
•Exact Match (F 1) penalizes valid paraphrases, failing
to reflect true model capability in generation.
•Semantic Evaluation shifts focus to utility and mean-
ing equivalence, which is essential for assessing LLM
outputs.
D. Tools
Over the past few years, a variety of reusable toolkits
have been released to support event extraction, reflecting the
steady maturation of research infrastructure. Early frameworks
such asDyGIE++[118] andSpERT[196] introduced span-
based modeling for entities, relations, and events. Build-
ing on this,OneIE[20] provided a unified document-level
pipeline that integrates entities, relations, and events. Sub-
sequent toolkits focused on specialization and accessibility.
CasEE[122] proposed a cascade decoding framework for
overlapping events, whileRESIN[197] released a dockerized
schema-guided pipeline for cross-document event extraction.
In parallel, the focus shifted toward generalization.UIE[127]
and its extensionInstructUIE[351] framed extraction as a
generative task, enabling zero- and few-shot adaptation, while
DeepKE[194] offered an open-source toolkit with ready-to-
use pipelines covering entities, relations, and events. More
recently, new platforms have targeted broader coverage and
scalability.OneEE[32] andDocEE[352] addressed one-stage
fast EE and document-level EE respectively.SPEED++[353]
introduced a multilingual EE framework with cross-lingual
evaluation, andOmniEvent[35] provided a large-scale open-
source toolkit that integrates datasets, models, and evaluation
scripts.TextEE[160] consolidated these developments by of-

20 ARXIV PREPRINT
fering a benchmark suite with runnable code and standardized
evaluation, marking the current stage of maturity in EE toolkit
development.
In parallel with toolkit development, the community has
also established standardized evaluation scripts to ensure re-
producibility and comparability. Early benchmarks such as the
ACE[354] program introduced task definitions and official
scorers for precision, recall, and F1. The subsequentTAC-
KBP[355], [356] evaluations extended this tradition to large-
scale knowledge base population, providing official evaluation
code for entity linking, slot filling, and event extraction. To
support richer annotation, theERE[11] framework unified
entity, relation, and event labeling with corresponding scoring
scripts. More recent benchmarks have followed the same
practice.MA VEN-ERE[357] released a large-scale dataset
with an official evaluation pipeline, whileTextEE[160]
consolidated prior resources and offered publicly available
scoring scripts for standardized re-evaluation. These evaluation
packages, often distributed through dataset repositories or task
organizers, have become a de facto standard in event extrac-
tion, enabling fair comparison across models and accelerating
methodological progress.
Beyond toolkits and evaluation scripts, researchers have
also developed annotation platforms and schema converters
to facilitate dataset creation and adaptation. Early annotation
tools such asBRAT[358] provided web-based interfaces
for labeling entities, relations, and events, and have become
widely adopted in constructing new corpora. Subsequently,
schema-aware infrastructures were introduced to address het-
erogeneous annotation guidelines:RESIN[197] released a
dockerized schema-guided pipeline for cross-document and
cross-lingual information extraction, enabling alignment of
entities, relations, and events across datasets. More recently,
human–machine collaboration has been explored to further
streamline annotation:CollabKG[139] proposes a cooper-
ative framework for knowledge graph and event annotation,
whileLFDe[195] presents a lighter and more data-efficient
workflow. Together, these platforms and converters reduce an-
notation costs and enhance dataset interoperability, providing
crucial support for large-scale event extraction research. For
clarity, Table V presents a categorical summary of existing
event extraction resources.
VIII. EEUNDERDIVERSESETTINGS
A. Language and Resource Conditions
In event extraction research, language setting defines the
scope of applicability and generalization across linguistic con-
texts. This section categorizes EE research along four major
dimensions: monolingual, multilingual, cross-lingual, and low-
resource.
a) Monolingual:Early EE research largely targets
single-language settings—most often English—under stan-
dardized schemas and evaluation protocols such as ACE [354],
TAC-KBP [355], [356], and ERE [11]. Within this paradigm,
modeling advances span formulation changes, decoders, and
evaluation practice. Casting EE [23] as machine reading com-
prehension emphasizes span-centric reasoning and question-style conditioning to couple trigger decisions. Architectural re-
finements [122], [359] for overlapping or interdependent struc-
tures further improve extraction fidelity in single-language
news and web domains. Beyond triggers, several works [134],
[360], [361] push document-level argument modeling with
specialized structures or pipelines, reflecting a shift from
sentence-local to discourse-level inference. New monolingual
resources also expand the modality and language coverage
while retaining a single-language assumption, e.g., speech-
based EE benchmarks and non-English single-language cor-
pora [100], [362], which test robustness to acoustic noise
and domain shift. As large language models (LLMs) become
stronger few-shot reasoners, recent analyses [363] probe pref-
erence biases and instruction sensitivity in argument extrac-
tion, highlighting evaluation nuances specific to monolingual
pipelines.
b) Multilingual:Multilingual EE aims to extract events
across many languages with a unified schema and modeling
backbone, often coordinating parallel or comparable corpora
to stabilize cross-language label semantics. Resource design
has therefore focused on schema alignment and language
coverage. New multilingual datasets [364] formalize cross-
language consistency for triggers and provide broader typolog-
ical diversity. On the modeling side, unified or massively mul-
tilingual encoders [353], [365], [366] serve as the backbone
for schema-conditioned extraction, enabling a single model to
operate across diverse languages while amortizing supervision.
Beyond generic benchmarks, specialized multilingual cor-
pora [293], [367], such as historical news and domain-specific
frames, stress-test schema transfer and semantic normalization
when language-specific lexicalizations diverge.
c) Cross-lingual:Cross-lingual EE trains on a source
language, typically English and transfers to target languages
with limited or no labels, emphasizing zero-shot transfer,
alignment, and adaptation. Early work [368] reframed EE
through transfer-friendly interfaces to leverage cross-lingual
sentence representations and reduce label mismatch. Transfer-
based methods [116], [289], [369]–[371], including multi-task
pretraining, priming, and the use of SRL as an auxiliary task,
improve argument-role generalization by leveraging shared
semantics. Prompt–based approaches [313], [323] paired with
instruction-like conditioning further improve zero-shot ro-
bustness by decoupling surface forms from schema roles,
while contextualized prompting addresses label drift across
languages. Recent studies [372]–[374] explicitly evaluate zero-
shot cross-lingual structure recommendation and role transfer
quality, clarifying where alignment succeeds or fails and how
language-independent signals can mitigate lexical gaps.
d) Low-resource:When labeled data are scarce, EE
performance hinges on how effectively models can construct
surrogate supervision and generalize from a few examples.
Data-centric methods [153], [154] dominate, while targeted
or schema-aware augmentation synthesizes diverse trigger
contexts, improving coverage of rare types and roles. Few-
shot document-level argument extraction [156] explores how
to transfer role semantics and coreference cues with min-
imal labels, often relying on discourse structure and prior
knowledge to compensate for annotation sparsity. Orthogo-

LIet al.: EVENT EXTRACTION 21
Fig. 8. An overview of diverse research settings in Event Extraction, categorized into two primary dimensions: language and resource conditions, and
discourse scope granularity.
nally, lightweight or data-efficient pretraining [195] reduces
reliance on large labeled corpora while preserving competitive
accuracy, indicating that pretraining objectives and architecture
choices can be tailored to low-resource regimes. Broader po-
sitioning work [254] emphasizes liberating EE from rigid data
constraints by integrating weak supervision, augmentation, and
careful evaluation design to better capture the realities of
under-resourced languages and domains.
B. Discourse Scope (Granularity)
In event extraction (EE) research, the choice of discourse
scope granularity defines the contextual boundaries of event
analysis and directly impacts task complexity and applica-
tion scenarios. This section categorizes EE granularity into
four core levels: sentence/span-level, document-level, cross-
document, and dialogue/conversation-level.
a) Sentence/Span-level:Event extraction focuses on the
single sentence or phrase level, where event triggers and
arguments often reside within the same sentence, making
this task relatively simple. Early work utilized traditional
machine learning methods to extract event triggers and classify
arguments [8], [375], [376]. Other researchers also improved
performance by incorporating lexical, syntactic, and semantic
information into models [224], [269], [377]. Subsequently,
models such as CNNs and RNNs were introduced, enhancing
the performance of event trigger identification and argument
classification by enhancing their ability to model context
[9], [126], [232]. At this time, researchers not only im-
proved performance by incorporating syntactic information
[12], [16], [267], [271] but also utilized unsupervised (or
weakly supervised) approaches to reduce reliance on large
amounts of high-quality, manually annotated data [378]–[380].
With technological advancements, deep learning has gradually
become mainstream. Researchers are generating data using
pre-trained language models, reducing reliance on manually
annotated data while also providing high-quality data [119],
[261]. Furthermore, researchers have viewed EE as a naturallanguage generation task [23], [29], for example, by asking
questions to allow pre-trained language models to directly
generate answers [24], [120], [363], or by constructing specific
prompt templates to achieve event extraction [31], [45], [381].
b) Document-level:Document-level event extraction
(DEE) aims to extract structured event information from an
entire document. Unlike sentence-level event extraction (SEE),
DEE requires the integration of event information scattered
across sentences in a document, which involves complex sce-
narios such as the scattered distribution of event arguments and
interrelationships among multiple events. Early researchers
used the same methods as SEE, such as traditional machine
learning methods [7], or incorporating syntactic information
to enhance event extraction performance [13], [268] and using
supervised learning to automatically generate large amounts of
annotated data [273]. However, this method cannot effectively
handle complex issues such as the dispersion of arguments
across sentences and the coexistence of multiple events. Gen-
erative methods [44], [63] are favored by researchers, who
extract events by constructing and filling in specific prompt
templates [27], [121], [133]. With the rapid development of
large language models, generative methods have received more
attention in DEE [42], [43], [278].
Scope: Local vs. Global Reasoning
•Sentence-level extraction focuses on local syntax and
identifying immediate triggers.
•Document-level extraction demands memory mecha-
nisms to track entities and logic to resolve cross-
sentence conflicts.
c) Cross-document:Cross-document event extraction
(CDEE) is a method that integrates information from multiple
documents to extract structured events. Unlike traditional
document-level and sentence-level methods, CDEE aims to
mine descriptions of the same event from different sources

22 ARXIV PREPRINT
through collaborative analysis of multiple documents, ulti-
mately forming a comprehensive and consistent view of the
event. Compared to DEE and SEE, CDEE faces challenges
such as information heterogeneity and inconsistency, as well as
cross-document coreference resolution. Most existing research
clusters documents with similar content through clustering or
semantic similarity, then analyzes each cluster [159], [382]–
[385]. In addition, some researchers have attempted to generate
events directly from multiple documents using generative
models [197], [386], but generative approaches face higher
time and space overhead when processing large numbers of
documents simultaneously.
d) Dialogue/Conversation-level:Conversation-level
event extraction is the task of identifying and extracting
event information from multi-turn conversations. While
similar to DEE, which requires processing multiple sentences,
conversation-level data differs in that it is nonlinear,
colloquial, involves dynamic interactions between multiple
actors, and is highly fragmented. Event information is
scattered across multiple conversation turns and depends on
speaker identity and contextual intent. It also requires handling
ellipsis, references, and conversational behavior variations.
Currently, research on conversation-level event extraction
is still in its infancy. Eisenberg et al. [387] constructed the
Personal Events in Dialogue Corpus (PEDC) and proposed
a method for automatically extracting personal events from
conversations. They used a support vector machine (SVM)
model for event classification and explored four feature types
and different machine learning protocols. Sharif et al. [41]
also constructed a conversation-level dataset, DiscourseEE,
and proposed a relaxed matching evaluation method based
on semantic similarity, addressing the problem of “exact
matching” evaluation leading to severe underestimation of
performance in generative event extraction.
C. Vertical Domains
Event extraction (EE) has been extensively studied in open
domains, yet its deployment in vertical domains has become
increasingly essential as real-world applications often operate
within specialized contexts. Each domain is characterized
by its own linguistic patterns, terminologies, and knowledge
structures, leading to substantial variation in data distributions.
As a result, models trained on generic corpora often face
severe performance degradation when transferred to domain-
specific texts, highlighting the need for domain adaptation
and knowledge-guided learning strategies. At the same time,
the growing prevalence of multimodal data—spanning text,
images, and videos—has expanded the scope of EE beyond
language alone, prompting researchers to explore how domain-
specific and cross-modal cues can jointly enhance event un-
derstanding.
In this survey, we examine both general domain and vertical-
domain event extraction. We include general domain EE not
as a vertical domain in the strict sense, but as a comple-
mentary setting that deals with unrestricted event types and
heterogeneous text sources, offering a valuable contrast to
domain-constrained scenarios. Alongside general domain EE,we discuss representative vertical domains such as biomedical,
financial, and legal EE, where rich domain knowledge and
task-specific schemas drive distinctive modeling challenges
and applications. Together, these perspectives provide a com-
prehensive view of how event extraction evolves from general-
purpose frameworks toward specialized and multimodal intel-
ligence.
a) General domain :General domain event extraction
(EE) targets text without a pre-specified ontology, aiming to
identify events across heterogeneous sources such as news
articles, web pages, and social media streams. Unlike vertical
domains, where schemas are tightly defined, general domain
EE must cope with continuously emerging event types and
evolving triggers. This diversity raises challenges in cover-
age, adaptability, and generalization. To address these issues,
several benchmarks have been proposed. The ACE 2005
corpus [354] and its extensions through TAC KBP evaluations
[388] provided early large-scale resources. More recently, the
MA VEN dataset [19] significantly extended coverage by anno-
tating 168 event types across 118,000 documents, establishing
the largest human-annotated general domain benchmark. The
RAMS corpus [22] emphasized document-level arguments,
highlighting the importance of contextual reasoning.
More recently, data augmentation with generated images
and captions has been employed to enhance multimedia EE
[389]. At the same time, Theia [390] proposed weakly su-
pervised multimodal EE with incomplete annotations, while
TSEE [391] introduced a three-stream contrastive framework
for text–video EE. On the language modeling side, instruction-
tuned large language models [392] have been shown to signif-
icantly improve the flexibility of open-domain EE, and GLEN
[393] extended the coverage of event types to thousands,
pushing the boundary of schema scalability.
In addition to mainstream corpora, some studies have inves-
tigated unusual sources such as historical newspapers [367] or
video transcripts [310], highlighting the adaptability of open-
domain EE to diverse modalities and data conditions. Col-
lectively, these advances demonstrate that open-domain EE is
moving toward schema-flexible, multimodal, and generalizable
approaches, which are essential for real-world applications
such as news monitoring, crisis management, and social media
event tracking.
b) Biomedical/Clinical:Biomedical and clinical event
extraction (EE) aims to identify complex interactions among
biomedical entities (e.g., genes, proteins, drugs, diseases) or
clinically relevant events such as diagnoses, treatments, and
adverse drug reactions. Compared with open-domain EE,
this domain is characterized by lengthy terminology, frequent
abbreviations, and nested entity structures, which together
pose significant modeling challenges. Importantly, biomedical
and clinical EE has received more attention and produced
more work than many other domains. This is largely due to
the abundance of publicly available textual resources such as
PubMed abstracts, PMC full-text articles, and electronic health
records, as well as pressing real-world needs in drug discov-
ery, disease understanding, and healthcare. Furthermore, long-
running benchmark campaigns such as the BioNLP Shared
Tasks [6] [162] [394] [395] have consistently provided datasets

LIet al.: EVENT EXTRACTION 23
Fig. 9. A three-layer architecture for Event Extraction applications in various domains.
and schemas that stimulate research in this area.
Among recent methods, DeepEventMine [182] introduced
an end-to-end framework for nested biomedical events and
remains a strong baseline. Reinforcement learning has also
been explored in biomedical event extraction. Zhao [396]
formulated multi-event extraction as a sequential decision
process and incorporated external biomedical knowledge bases
to guide argument detection. A subsequent study by Zhao
[397] improved this RL framework through self-supervised
data augmentation, enabling more robust learning under sparse
annotation. These works illustrate how RL can be strengthened
by domain knowledge and auxiliary supervision in biomedical
settings. Beyond reinforcement learning–based approaches,
several recent studies have further diversified the methodolog-
ical landscape of biomedical event extraction. Fine-grained
attention mechanisms have been explored to capture subtle se-
mantic distinctions in complex biomedical interactions [398].
Tree-structured attentive models such as Child-Sum EATree-
LSTMs [399] leverage syntactic hierarchies to enhance event
representation. Knowledge-guided hierarchical graph networks
[400] incorporate curated biomedical knowledge to improve
causal relation extraction. Constraint-based multi-task frame-
works like CMBEE [401] promote structural consistency by
jointly learning triggers, arguments, and event schemas under
explicit constraints. More recently, framing biomedical EE
as a semantic segmentation problem [402] has introduced a
fully end-to-end and schema-flexible perspective, demonstrat-
ing strong performance in handling complex and low-resource
scenarios.
Clinical event extraction has likewise attracted increasing
attention. Datasets such as PHEE for pharmacovigilance [59],
discharge summaries annotated with adverse events [403], and
ACES for cohort event-stream data [404] provide valuable
resources. Models like DICE [89] introduced data-efficient
clinical EE with generative models, while disorder-aware
attention mechanisms [405] improved recognition of clinicalevents from electronic health records. More recently, multi-
lingual frameworks such as SPEED++ [183] have expanded
biomedical and epidemic event extraction to global contexts.
Collectively, these advances demonstrate that biomedical
and clinical EE has evolved from feature-based methods on
BioNLP benchmarks to diverse neural and generative ap-
proaches, supported by specialized datasets. The richness of
resources, the continuity of community benchmarks, and the
urgent societal needs explain why this domain has generated
a larger body of work compared to others. It remains vital
for applications such as biomedical knowledge base construc-
tion, drug discovery, pharmacovigilance, and clinical decision
support.
c) Finance:Financial event extraction (EE) focuses on
identifying market- and economy-related events such as merg-
ers and acquisitions, bankruptcies, stock fluctuations, policy
changes, and equity pledges. Compared with open-domain
settings, financial EE often requires reasoning over tempo-
ral and causal relations, since the sequence of events (e.g.,
policy issuance followed by market reaction) is essential for
investment and risk analysis.
In terms of resources, financial EE exhibits a clear imbal-
ance across languages. Chinese datasets are far more abundant,
largely due to the availability of structured company announce-
ments and financial disclosures in the Chinese market. Repre-
sentative benchmarks include DCFEE [111], the first large-
scale document-level financial EE system based on distant
supervision, and Doc2EDAG [64], which modeled document-
level financial events as directed acyclic graphs. Subsequent
datasets such as FinEvent [186], CFinDEE [187], and CFERE
[406] further enriched fine-grained and multi-type financial
event coverage. More recently, OEE-CFC [188] explored open
EE from Chinese financial commentary, and Probing into the
Root [407] provided a benchmark for reasoning about the
causes of structural events.
By contrast, English financial EE datasets remain scarce.

24 ARXIV PREPRINT
While efforts such as automatic extraction of financial events
from news [12] and the Event Causality in Finance benchmark
[408] have begun to fill the gap, most English studies still
rely on self-constructed corpora from financial news (e.g.,
Reuters, Bloomberg) that are not fully open due to licensing
restrictions. As a result, many methodological innovations in
financial EE—such as reinforcement learning, graph-based
document modeling, or causal reasoning—have been validated
primarily on Chinese datasets, with only limited evaluation on
English data.
Overall, financial EE has grown from early document-
level systems to datasets and models capable of fine-grained,
causal, and open extraction. The uneven resource distribution
between Chinese and English underscores both the maturity
of research in the Chinese community and the need for
broader multilingual benchmarks. These advances highlight
the practical value of financial EE for applications such as
investment decision-making, risk management, and financial
knowledge graph construction.
d) Legal:Legal event extraction seeks to extract struc-
tured events from legal texts such as court judgments, con-
tracts, case reports, and other judicial documents. Compared
with general open-domain or vertical domains, the legal do-
main imposes stricter semantics, domain-specific schemas,
and high requirements for interpretability and correctness.
The scarcity of publicly annotated legal event datasets has
constrained model development, but recent work is beginning
to fill this gap.
One significant dataset is LEVEN [190], which provides
8,116 Chinese legal documents annotated with 150,977 event
mentions across 108 event types, laying a foundation for
scale in legal event detection. To further enrich granularity,
LEEC [191] defines an extensive label system of 159 elements
for criminal documents, supporting fine-grained extraction
tasks. On the methodological side, Hierarchical legal event
extraction via Pedal Attention Mechanism [307] introduced
hierarchical features and attention mechanisms specific to
Chinese legal events. More recently, Event Grounded Criminal
Court View Generation [409] combines event extraction and
legal document generation using LLMs, injecting fine-grained
legal events into court view summarization. Earlier work such
as Event Extraction for Legal Case Building and Reasoning
[410] laid the conceptual groundwork for event-based legal
case analysis. These developments illustrate a trajectory: from
entity/event detection toward integrated event-centered legal
reasoning, generation, and retrieval. Applications include legal
knowledge graph construction, case retrieval, judgment sum-
marization, and intelligent legal assistants.
IX. OPENCHALLENGES ANDFUTUREDIRECTIONS
The advent of generative LLMs and MLLMs has not
merely pushed the performance boundaries on standard EE
benchmarks; it has fundamentally precipitated a paradigm shift
in how we conceptualize the task itself. We argue that EE
is undergoing a critical transition from a static, sentence-
level information extraction problem to a dynamic, document-
level knowledge acquisition process intended for intelligentsystems. Moving beyond the traditional limitations of span-
based accuracy within isolated sentences, we identify six
transformative directions that represent the true frontier of
this field, demanding deeper investigation and architectural
innovation.
A. Agentic Perception
The traditional view of event extraction as a pipeline to
populate static knowledge bases is increasingly insufficient in
the era of autonomous agents. An intelligent agent operating
in a complex environment does not need a static snapshot of
history; it requires a continuously updated understanding of
state changes and evolving situations to inform its actions. In
this context, EE must be reimagined as the primary “perception
module” intended to digest continuous text streams (e.g., news
feeds, operational logs, dialogue history) and convert them into
structured observations. The fundamental challenge shifts from
simply identifying that an event occurred to understanding its
implications for the agent’s current state.
Future research must focus on integrating extracted events
directly into the agent’s cognitive architecture, specifically its
episodic memory and world model. This means moving away
from generating isolated JSON objects toward maintaining a
coherent, temporally linked narrative of evolving realities. A
critical research direction is developing mechanisms where
newly extracted events can trigger updates to previously stored
information, resolve conflicting observations over time, and
directly feed into downstream modules responsible for causal
reasoning, planning, and decision-making under uncertainty.
The goal is to transform text into actionable situational aware-
ness.
Future Vision: Static vs. Dynamic
•Static extraction traditionally populates offline
databases from historical text.
•Dynamic perception serves as an Agent’s Memory,
continuously updating state from streaming data.
B. Neuro-Symbolic Reasoning
While LLMs exhibit impressive “System 1” intuition in
identifying potential triggers and arguments based on surface-
level semantic correlations, they remain fundamentally prone
to probabilistic failures. They often struggle with complex
structural constraints (e.g., an entity cannot be both an ’At-
tacker’ and a ’Victim’ in the same ’Attack’ event unless
specified) and suffer from factual hallucinations, generating
plausible-looking but incorrect structures. The core limitation
is the lack of explicit reasoning mechanisms to verify that the
generated structure adheres to logical and ontological rules
inherent to the domain.
To achieve robust and trustworthy extraction, a critical
frontier is instilling “System 2” capabilities through neuro-
symbolic integration. This goes beyond simply prompting
an LLM to “think step-by-step.” Future frameworks should
explore utilizing event schemas and logical rules not just

LIet al.: EVENT EXTRACTION 25
as data preprocessing steps, but as hard constraints during
the decoding process (e.g., constrained beam search via au-
tomata) to physically prevent invalid structures. Furthermore,
we envision architectures that incorporate an intrinsic “critic”
module—potentially a separate, logic-driven system—that de-
liberately verifies the generated event arguments against source
evidence before final output, trading inference speed for struc-
tural guarantee.
Future Direction: System 1 vs. System 2
•System 1 (LLMs) provides probabilistic intuition but
struggles with complex structural constraints.
•Future System 2 frameworks integrate Neuro-Symbolic
verification to enforce logical validity and reliability.
C. Interactive Open-World Discovery
The reliance on rigid, pre-defined ontologies (such as ACE
or ERE) severely curtails the real-world applicability of EE
systems. The long tail of complex scenarios contains myriad
event types that cannot be anticipated ab initio. While recent
zero-shot and few-shot approaches show promise, they often
still operate under a closed-world assumption where the new
types are known to the user beforehand. The true challenge lies
in handling unknown unknowns—enabling systems to proac-
tively identify clusters of information that indicate previously
undefined event types evolving in the data.
The future paradigm must shift from passive extraction
based on fixed instructions to interactive knowledge discovery.
Future systems should not merely output a best guess when
faced with ambiguity; they should possess the meta-cognitive
ability to recognize their uncertainty and engage in clarifica-
tion dialogues with human users (e.g., Does this text describe
a new type of ‘Cyberattack’, or is it a sub-event of ’Fraud’?).
This collaborative approach allows the model to incrementally
build and refine its own schema repository, evolving from a
static classifier into an adaptive, lifelong-learning knowledge
acquisition engine.
Future Paradigm: Passive vs. Interactive
•Passive extraction operates on fixed ontologies, forcing
ambiguous inputs into pre-defined slots.
•Interactive discovery possesses meta-cognition to rec-
ognize uncertainty and proactively query users to learn
new types.
D. Cross-Document Synthesis
A significant disconnect exists between academic bench-
marks, which predominantly focus on sentence-level ex-
traction, and real-world information dynamics. Complex
events—such as a pandemic outbreak, a corporate merger, or a
geopolitical conflict—are rarely encapsulated within a single
paragraph. Their core arguments (who, when, where, why)are fragmented across long documents or scattered among dis-
parate sources published over extended periods. Current LLM-
based approaches, despite larger context windows, struggle
to synthesize conflicting information and maintain coherence
over long horizons due to the lost-in-the-middle phenomenon
and attention dilution.
Addressing this requires architectural innovations geared
toward massive-scale context integration. Future research
needs to develop specialized Retrieval-Augmented Generation
(RAG) systems optimized for structured event data, capable of
retrieving not just relevant sentences, but related existing event
structures to inform current extraction. Furthermore, cracking
cross-document event coreference resolution and temporal
ordering is paramount. A sophisticated system must be able to
link an arrest mentioned in today’s news with an investigation
mentioned last week, recognizing them as parts of the same
larger event chain, rather than treating them as isolated data
points.
E. Physically Grounded World Models
Current multimodal event extraction efforts often suffer
from a shallow alignment limitation, where visual data is
treated merely as supplementary signals to disambiguate tex-
tual entities (e.g., bounding box matching). This misses the
profound potential of multimodal data: grounding language in
physical reality. Text is inherently lossy; it rarely explicitly
states common-sense physical details (e.g., that a break event
implies an instrument and an irreversible state change). Re-
lying solely on text limits the depth of understanding causal
dynamics.
The next generation of multimodal models must leverage
vast amounts of video data to learn the intuitive physics and
temporal cause-and-effect chains of the real world. By pre-
training on visual dynamics, future models should be able to
infer implicit arguments that are visually obvious but textually
unstated. For instance, presented with text about a car accident,
a physically grounded model should implicitly understand the
likely involvement of high velocity and impact forces, allowing
it to predict potential consequences (e.g., injury, damage)
even if unmentioned. This bridging of linguistic semantics
with physical scene understanding is key to deeper event
comprehension.
Multimodal Evolution: Alignment vs. Grounding
•Shallow alignment uses visual data merely as auxiliary
signals to locate or disambiguate textual entities.
•Physical grounding learns intuitive physics from video
to infer implicit causes and consequences unstated in
text.
F . Utility-Driven Evaluation
As EE shifts toward generative paradigms, traditional exact-
match metrics like F1 scores based on character offsets are be-
coming functionally obsolete, penalizing valid paraphrases and
failing to capture structural correctness. While shifting toward

26 ARXIV PREPRINT
LLM-based semantic evaluation shows promise, it introduces
new challenges regarding evaluator bias and reproducibility
that must be rigorously studied. The community urgently needs
standardized, trustworthy protocols for assessing semantic
equivalence in structured outputs.
More critically, evaluation must expand along two neglected
dimensions: utility and trustworthiness. Firstly, intrinsic met-
rics should be complemented by extrinsicutility-based eval-
uation, measuring the quality of extraction by its tangible
impact on downstream applications (e.g., Does this EE system
measurably improve the accuracy of a quantitative financial
forecasting model?). Secondly, for real-world deployment,
metrics forcalibration and uncertaintyare essential. A useful
system must not only extract facts but also reliably signal its
own confidence levels, knowing when it does *not* know,
which is vital for ensuring safety and building user trust in
critical domains.
X. CONCLUSION
This survey traces the evolution of Event Extraction (EE)
and outlines promising directions for future work. We begin
by defining the core tasks in EE, covering a spectrum from
traditional text-based extraction to complex multimodal sce-
narios, and explore their applications across various domains.
The review then maps out the field’s methodological timeline,
from rule-based systems and classical machine learning to
deep learning and the current LLM-driven paradigm. We also
provide a detailed analysis of system architectures, feature
enhancement techniques, and essential resources like datasets,
metrics, and toolkits. In the current stage, the arrival of Large
Language Models has fundamentally changed the field, open-
ing up new possibilities while introducing significant chal-
lenges. Looking ahead, we identify critical research frontiers,
including the extraction of implicit events, the improvement of
cross-modal alignment, and the development of reliable gener-
ative models. We hope this comprehensive overview serves as
a clear guide to the field’s development, inspiring innovative
work that pushes the boundaries of event understanding.REFERENCES
[1] B. Sundheim, “Overview of the third message understanding evaluation
and conference,” inProceedings of MUC, 1991, pp. 3–16.
[2] J. Pustejovsky, P. Hanks, R. Sauri, A. See, R. Gaizauskas, A. Setzer,
D. Radev, B. Sundheim, D. Day, L. Ferroet al., “The timebank corpus,”
inCorpus linguistics, vol. 2003. Lancaster, UK, 2003, p. 40.
[3] A. Mitchell, S. Strassel, S. Huang, and R. Zakhary, “Ace 2004
multilingual training corpus,” Linguistic Data Consortium, 2005.
[4] C. Walker, S. Strassel, J. Medero, and K. Maeda, “Ace 2005 multilin-
gual training corpus,”(No Title), 2006.
[5] D. Ahn, “The stages of event extraction,” inProceedings of the
Workshop on Annotating and Reasoning about Time and Events, 2006,
pp. 1–8.
[6] J.-D. Kim, T. Ohta, S. Pyysalo, Y . Kano, and J. Tsujii, “Overview
of bionlp’09 shared task on event extraction,” inBioNLPHLT-NAACL,
2009, pp. 1–9.
[7] S. Liao and R. Grishman, “Using document level cross-event inference
to improve event extraction,” inProceedings of the 48th annual meeting
of the association for computational linguistics, 2010, pp. 789–797.
[8] Q. Li, H. Ji, and L. Huang, “Joint event extraction via structured
prediction with global features,” inProceedings of the 51st Annual
Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers), 2013, pp. 73–82.
[9] Y . Chen, L. Xu, K. Liu, D. Zeng, and J. Zhao, “Event extraction via
dynamic multi-pooling convolutional neural networks,” inProceedings
of the 53rd Annual Meeting of the Association for Computational
Linguistics and the 7th International Joint Conference on Natural
Language Processing (Volume 1: Long Papers), 2015, pp. 167–176.
[10] T. H. Nguyen and R. Grishman, “Event detection and domain adapta-
tion with convolutional neural networks,” inProceedings of the 53rd
Annual Meeting of the Association for Computational Linguistics and
the 7th International Joint Conference on Natural Language Processing
(Volume 2: Short Papers), 2015, pp. 365–371.
[11] Z. Song, A. Bies, S. Strassel, T. Riese, J. Mott, J. Ellis, J. Wright,
S. Kulick, N. Ryant, and X. Ma, “From light to rich ere: Annotation of
entities, relations, and events,” inProceedings of the 3rd workshop on
EVENTS: Definition, detection, coreference, and representation, 2015,
pp. 89–98.
[12] T. H. Nguyen, K. Cho, and R. Grishman, “Joint event extraction via
recurrent neural networks,” inProceedings of the 2016 conference
of the North American chapter of the association for computational
linguistics: human language technologies, 2016, pp. 300–309.
[13] B. Yang and T. Mitchell, “Joint extraction of events and entities within a
document context,” inProceedings of the 2016 Conference of the North
American Chapter of the Association for Computational Linguistics:
Human Language Technologies, 2016, pp. 289–299.
[14] R. Ghaeini, X. Z. Fern, L. Huang, and P. Tadepalli, “Event nugget
detection with forward-backward recurrent neural networks,” inPro-
ceedings of ACL, 2016.
[15] T. Nguyen and R. Grishman, “Graph convolutional networks with
argument-aware pooling for event detection,” inProceedings of the
AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018.
[16] X. Liu, Z. Luo, and H.-Y . Huang, “Jointly multiple events extraction
via attention-based graph information aggregation,” inProceedings
of the 2018 Conference on Empirical Methods in Natural Language
Processing, 2018, pp. 1247–1256.
[17] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training
of deep bidirectional transformers for language understanding,” in
Proceedings of the 2019 conference of the North American chapter
of the association for computational linguistics: human language
technologies, volume 1 (long and short papers), 2019, pp. 4171–4186.
[18] X. Wang, Z. Wang, X. Han, Z. Liu, J. Li, P. Li, M. Sun, J. Zhou, and
X. Ren, “Hmeae: Hierarchical modular event argument extraction,” in
EMNLP/IJCNLP, 2019, pp. 5776–5782.
[19] X. Wang, Z. Wang, X. Han, W. Jiang, R. Han, Z. Liu, J. Li, P. Li, Y . Lin,
and J. Zhou, “MA VEN: A Massive General Domain Event Detection
Dataset,” inProceedings of the 2020 Conference on Empirical Methods
in Natural Language Processing (EMNLP), 2020, pp. 1652–1671.
[20] Y . Lin, H. Ji, F. Huang, and L. Wu, “A joint neural model for
information extraction with global features,” inProceedings of the 58th
annual meeting of the association for computational linguistics, 2020,
pp. 7999–8009.
[21] M. Li, A. Zareian, Q. Zeng, S. Whitehead, D. Lu, H. Ji, and S.-F.
Chang, “Cross-media structured common space for multimedia event
extraction,” inProceedings of ACL, 2020, pp. 2557–2568.

LIet al.: EVENT EXTRACTION 27
[22] S. Ebner, P. Xia, R. Culkin, K. Rawlins, and B. Van Durme, “Multi-
sentence argument linking,” inProceedings of the 58th Annual Meeting
of the Association for Computational Linguistics, 2020, pp. 8057–8077.
[23] J. Liu, Y . Chen, K. Liu, W. Bi, and X. Liu, “Event extraction as machine
reading comprehension,” inProceedings of the 2020 conference on
empirical methods in natural language processing (EMNLP), 2020,
pp. 1641–1651.
[24] X. Du and C. Cardie, “Event extraction by answering (almost) natural
questions,” inProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP), 2020, pp. 671–
683.
[25] T. Satyapanich, F. Ferraro, and T. Finin, “Casie: Extracting cyber-
security event information from text,” inProceedings of the AAAI
conference on artificial intelligence, vol. 34, no. 05, 2020, pp. 8749–
8757.
[26] X. Du and C. Cardie, “Document-level event role filler extrac-
tion using multi-granularity contextualized encoding,”arXiv preprint
arXiv:2005.06579, 2020.
[27] S. Li, H. Ji, and J. Han, “Document-level event argument extraction
by conditional generation,” inProceedings of the 2021 Conference
of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies, 2021, pp. 894–908.
[28] B. Chen, X. Lin, C. Thomas, M. Li, S. Yoshida, L. Chum, H. Ji, and S.-
F. Chang, “Joint multimedia event extraction from video and article,”
inFindings of the Association for Computational Linguistics: EMNLP
2021, 2021, pp. 74–88.
[29] Y . Lu, H. Lin, J. Xu, X. Han, J. Tang, A. Li, L. Sun, M. Liao, and
S. Chen, “Text2event: Controllable sequence-to-structure generation
for end-to-end event extraction,” inProceedings of the 59th Annual
Meeting of the Association for Computational Linguistics and the
11th International Joint Conference on Natural Language Processing
(Volume 1: Long Papers), 2021, pp. 2795–2806.
[30] G. Paolini, B. Athiwaratkun, J. Krone, J. Ma, A. Achille, R. Anubhai,
C. N. dos Santos, B. Xiang, and S. Soatto, “Structured prediction as
translation between augmented natural languages,” inProceedings of
ICLR, 2021.
[31] I.-H. Hsu, K.-H. Huang, E. Boschee, S. Miller, P. Natarajan, K.-W.
Chang, and N. Peng, “Degree: A data-efficient generation-based event
extraction model,” inProceedings of the 2022 Conference of the North
American Chapter of the Association for Computational Linguistics:
Human Language Technologies, 2022, pp. 1890–1908.
[32] H. Cao, J. Li, F. Su, F. Li, H. Fei, S. Wu, B. Li, L. Zhao, and D. Ji,
“Oneee: A one-stage framework for fast overlapping and nested event
extraction,”arXiv preprint arXiv:2209.02693, 2022.
[33] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhari-
wal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal,
A. Herbert-V oss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M.
Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin,
S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford,
I. Sutskever, and D. Amodei, “Language models are few-shot learners,”
inNeurIPS, 2020.
[34] X. Wei, X. Cui, N. Cheng, X. Wang, X. Zhang, S. Huang, P. Xie, J. Xu,
Y . Chen, M. Zhanget al., “Chatie: Zero-shot information extraction via
chatting with chatgpt,”arXiv preprint arXiv:2302.10205, 2023.
[35] H. Peng, X. Wang, F. Yao, Z. Wang, C. Zhu, K. Zeng, L. Hou, and
J. Li, “Omnievent: A comprehensive, fair, and easy-to-use toolkit for
event understanding,”arXiv preprint arXiv:2309.14258, 2023.
[36] Z. Du, Y . Li, X. Guo, Y . Sun, and B. Li, “Training multimedia event
extraction with generated images and captions,” inProceedings of the
31st ACM international conference on multimedia, 2023, pp. 5504–
5513.
[37] H. Singh, P. Zhang, Q. Wang, M. Wang, W. Xiong, J. Du, and
Y . Chen, “Coarse-to-fine contrastive learning in image-text-graph space
for improved vision-language compositionality,” inProceedings of
the 2023 Conference on Empirical Methods in Natural Language
Processing, 2023, pp. 869–893.
[38] X. Wang, W. Zhou, C. Zu, H. Xia, T. Chen, Y . Zhang, R. Zheng,
J. Ye, Q. Zhang, T. Guiet al., “Instructuie: Multi-task instruction tuning
for unified information extraction,”arXiv preprint arXiv:2304.08085,
2023.
[39] Q. Gao, B. Li, Z. Meng, Y . Li, J. Zhou, F. Li, C. Teng, and D. Ji,
“Enhancing cross-document event coreference resolution by discourse
structure and semantic information,” inLREC/COLING, 2024, pp.
5907–5921.
[40] Q. Gao, Z. Meng, B. Li, J. Zhou, F. Li, C. Teng, and D. Ji, “Har-
vesting events from multiple sources: Towards a cross-document eventextraction paradigm,” inFindings of the Association for Computational
Linguistics: ACL 2024, 2024, pp. 1913–1927.
[41] O. Sharif, J. Gatto, M. Basak, and S. M. Preum, “Explicit, implicit, and
scattered: Revisiting event extraction to capture complex arguments,” in
Proceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing, 2024, pp. 12 061–12 081.
[42] H. Zhou, J. Qian, Z. Feng, L. Hui, Z. Zhu, and K. Mao, “Llms learn task
heuristics from demonstrations: A heuristic-driven prompting strategy
for document-level event argument extraction,” inProceedings of the
62nd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), 2024, pp. 11 972–11 990.
[43] X. F. Zhang, C. Blum, T. Choji, S. Shah, and A. Vempala, “Ultra: Un-
leash llms’ potential for event argument extraction through hierarchical
modeling and pair-wise self-refinement,” inFindings of the Association
for Computational Linguistics ACL 2024, 2024, pp. 8172–8185.
[44] L. Sun, K. Zhang, Q. Li, and R. Lou, “Umie: Unified multimodal
information extraction with instruction tuning,” inProceedings of the
AAAI Conference on Artificial Intelligence, vol. 38, no. 17, 2024, pp.
19 062–19 070.
[45] R. Lin, Y . Liu, Y . Gan, Y . Cai, T. Lan, and Q. Liu, “Gems: Generation-
based event argument extraction via multi-perspective prompts and
ontology steering,” inFindings of the Association for Computational
Linguistics: ACL 2025, 2025, pp. 26 392–26 409.
[46] J. Cao, Y . Hu, Z. Tan, and X. Zhao, “Cross-modal multi-task learning
for multimedia event extraction,” inProceedings of the AAAI Confer-
ence on Artificial Intelligence, vol. 39, no. 11, 2025, pp. 11 454–11 462.
[47] Z. Qiu, C. Ma, J. Wu, and J. Yang, “Text is all you need: Llm-enhanced
incremental social event detection,” inProceedings of ACL, 2025, pp.
4666–4680.
[48] C. Ma, Y . Wang, J. Wu, J. Yang, J. Du, Z. Qiu, Q. Li, H. Wang, and
P. Nakov, “Explicit and implicit data augmentation for social event
detection,” inProceedings of ACL, 2025, pp. 8402–8415.
[49] Y .-F. Lu, X.-L. Mao, T. Lan, T. Zhang, Y .-S. Zhu, and H. Huang,
“SEOE: A scalable and reliable semantic evaluation framework for
open domain event detection,” inProceedings of the 63rd Annual
Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers), 2025, pp. 7201–7218.
[50] OpenAI, “Gpt-4 technical report,” inCoRR, vol. abs/2303.08774, 2023.
[51] Deepseek, “DeepSeek-R1 incentivizes reasoning in LLMs through
reinforcement learning,”Nature, vol. 645, pp. 633–638, 2025. [Online].
Available: https://doi.org/10.1038/s41586-025-09422-z
[52] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen,
W. Peng, X. Feng, B. Qin, and T. Liu, “A survey on hallucination
in large language models: Principles, taxonomy, challenges, and open
questions,”CoRR, vol. abs/2311.05232, 2023. [Online]. Available:
https://arxiv.org/abs/2311.05232
[53] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi,
Q. Le, and D. Zhou, “Chain-of-thought prompting elicits reasoning in
large language models,”CoRR, vol. abs/2201.11903, 2022. [Online].
Available: https://arxiv.org/abs/2201.11903
[54] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K”uttler, M. Lewis, W.-t. Yih, T. Rockt”aschel, S. Riedel, and
D. Kiela, “Retrieval-augmented generation for knowledge-intensive
NLP tasks,” inAdvances in Neural Information Processing Systems,
2020. [Online]. Available: https://arxiv.org/abs/2005.11401
[55] B. Peng, Y . Zhu, Y . Liu, X. Bo, H. Shi, C. Hong, Y . Zhang, and
S. Tang, “Graph retrieval-augmented generation: A survey,”CoRR,
vol. abs/2408.08921, 2024. [Online]. Available: https://arxiv.org/abs/
2408.08921
[56] J. S. Park, J. C. O’Brien, C. J. Cai, M. R. Morris, P. Liang,
and M. S. Bernstein, “Generative agents: Interactive simulacra of
human behavior,” inProceedings of the 36th Annual ACM Symposium
on User Interface Software and Technology (UIST 2023). San
Francisco, CA, USA: ACM, 2023, pp. 2:1–2:22. [Online]. Available:
https://doi.org/10.1145/3586183.3606763
[57] G. Doddington, A. Mitchell, M. Przybocki, L. Ramshaw, S. Strassel,
and R. Weischedel, “The automatic content extraction (ACE) program –
tasks, data, and evaluation,” inProceedings of the Fourth International
Conference on Language Resources and Evaluation (LREC’04), 2004.
[58] S. Deng, N. Zhang, J. Kang, Y . Zhang, W. Zhang, and H. Chen,
“Meta-learning with dynamic-memory-based prototypical network for
few-shot event detection,” inProceedings of the 13th International
Conference on Web Search and Data Mining, 2020, p. 151–159.
[59] Z. Sun, J. Li, G. Pergola, B. Wallace, B. John, N. Greene, J. Kim, and
Y . He, “PHEE: A dataset for pharmacovigilance event extraction from
text,” inProceedings of the 2022 Conference on Empirical Methods in
Natural Language Processing, 2022, pp. 5571–5587.

28 ARXIV PREPRINT
[60] T. Parekh, I.-H. Hsu, K.-H. Huang, K.-W. Chang, and N. Peng,
“GENEV A: Benchmarking generalizability for event argument extrac-
tion with hundreds of event types and argument roles,” inProceedings
of the 61st Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), 2023, pp. 3664–3686.
[61] A. Pouran Ben Veyseh, M. V . Nguyen, F. Dernoncourt, and T. Nguyen,
“MINION: a large-scale and diverse dataset for multilingual event
detection,” inProceedings of the 2022 Conference of the North
American Chapter of the Association for Computational Linguistics:
Human Language Technologies, 2022, pp. 2286–2299.
[62] S. Zheng, W. Cao, W. Xu, and J. Bian, “Doc2edag: An end-to-end
document-level framework for chinese financial event extraction,” in
Proceedings of the 2019 Conference on Empirical Methods in Natural
Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP), 2019, pp. 337–346.
[63] H. Yang, D. Sui, Y . Chen, K. Liu, J. Zhao, and T. Wang, “Document-
level event extraction via parallel prediction networks,” inProceedings
of the 59th Annual Meeting of the Association for Computational
Linguistics and the 11th International Joint Conference on Natural
Language Processing (Volume 1: Long Papers), 2021, pp. 6298–6308.
[64] C. Wang, X. Liu, Z. Chen, H. Hong, J. Tang, and D. Song, “DeepStruct:
Pretraining of language models for structure prediction,” inFindings
of the Association for Computational Linguistics: ACL 2022, 2022, pp.
803–823.
[65] Y . Jiao, S. Li, Y . Xie, M. Zhong, H. Ji, and J. Han, “Open-
vocabulary argument role prediction for event extraction,”arXiv
preprint arXiv:2211.01577, 2022.
[66] G. Li, P. Wang, J. Xie, R. Cui, and Z. Deng, “Feed: A chinese
financial event extraction dataset constructed by distant supervision,” in
Proceedings of the 10th International Joint Conference on Knowledge
Graphs, 2022.
[67] X. Li, F. Li, L. Pan, Y . Chen, W. Peng, Q. Wang, Y . Lyu, and Y . Zhu,
“Duee: a large-scale dataset for chinese event extraction in real-world
scenarios,” inCCF International Conference on Natural Language
Processing and Chinese Computing. Springer, 2020, pp. 534–545.
[68] Y . Chen, H. Yang, K. Liu, J. Zhao, and Y . Jia, “Collective event
detection via a hierarchical and bias tagging networks with gated multi-
level attention mechanisms,” inProceedings of EMNLP, 2018, pp.
1267–1276.
[69] T. Caselli and P. V ossen, “The storyline annotation and representation
scheme (StaR): A proposal,” inProceedings of the 2nd Workshop on
Computing News Storylines (CNS 2016), 2016, pp. 67–72.
[70] A. Eirew, A. Cattan, and I. Dagan, “WEC: Deriving a large-scale cross-
document event coreference dataset from Wikipedia,” inProceedings of
the 2021 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies, 2021,
pp. 2498–2510.
[71] J. Zhao, J. Tu, B. Ye, X. Hu, N. Xue, and J. Pustejovsky, “Be-
yond benchmarks: Building a richer cross-document event coreference
dataset with decontextualization,” inProceedings of the 2025 Con-
ference of the Nations of the Americas Chapter of the Association for
Computational Linguistics: Human Language Technologies (Volume 1:
Long Papers), 2025, pp. 3499–3513.
[72] A. Cybulska and P. V ossen, “Using a sledgehammer to crack a nut?
lexical diversity and event coreference resolution,” inProceedings
of the Ninth International Conference on Language Resources and
Evaluation (LREC’14), 2014, pp. 4545–4552.
[73] A. Pouran Ben Veyseh, V . D. Lai, C. Nguyen, F. Dernoncourt, and
T. Nguyen, “MCECR: A novel dataset for multilingual cross-document
event coreference resolution,” inFindings of the Association for Com-
putational Linguistics: NAACL 2024, 2024, pp. 3869–3880.
[74] M. Bugert, N. Reimers, and I. Gurevych, “Generalizing cross-document
event coreference resolution across multiple corpora,”Computational
Linguistics, vol. 47, no. 3, pp. 575–614, 2021.
[75] K. Wei, X. Shi, J. Tong, S. R. Reddy, A. Natarajan, R. Jain,
A. Garimella, and R. Huang, “LegalCore: A dataset for event corefer-
ence resolution in legal documents,” inFindings of the Association for
Computational Linguistics: ACL 2025, 2025, pp. 25 044–25 059.
[76] P. V ossen, F. Ilievski, M. Postma, and R. Segers, “Don’t annotate, but
validate: a data-to-text method for capturing event data,” inProceedings
of the Eleventh International Conference on Language Resources and
Evaluation (LREC 2018), 2018.
[77] P. K. Choubey and R. Huang, “Tamu at kbp 2017: Event nugget de-
tection and coreference resolution,”arXiv preprint arXiv:1711.02162,
2017.[78] Z. Chen and H. Ji, “Graph-based event coreference resolution,” in
Graph-based Methods for Natural Language Processing, 2009, pp. 54–
57.
[79] H. Lee, M. Recasens, A. X. Chang, M. Surdeanu, and D. Jurafsky,
“Joint entity and event coreference resolution across documents,” in
EMNLP-CoNLL, 2012, pp. 489–500.
[80] S. Barhom, V . Shwartz, A. Eirew, M. Bugert, N. Reimers, and I. Da-
gan, “Revisiting joint modeling of cross-document entity and event
coreference resolution,” inProceedings of ACL, 2019, pp. 4179–4189.
[81] Q. Ning, H. Wu, and D. Roth, “A multi-axis annotation scheme for
event temporal relations,” inProceedings of the 56th Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long
Papers), 2018, pp. 1318–1328.
[82] P. Mirza and S. Tonelli, “An analysis of causality between events and its
relation to temporal information,” inProceedings of COLING 2014, the
25th International Conference on Computational Linguistics: Technical
Papers, 2014, pp. 2097–2106.
[83] G. Glava ˇs, J. ˇSnajder, M.-F. Moens, and P. Kordjamshidi, “HiEve: A
corpus for extracting event hierarchies from news stories,” inProceed-
ings of the Ninth International Conference on Language Resources and
Evaluation (LREC’14), 2014, pp. 3678–3683.
[84] T. Cassidy, B. McDowell, N. Chambers, and S. Bethard, “An annotation
framework for dense event ordering,” inProceedings of the 52nd
Annual Meeting of the Association for Computational Linguistics
(Volume 2: Short Papers), 2014, pp. 501–506.
[85] A. Naik, L. Breitfeller, and C. Rose, “TDDiscourse: A dataset for
discourse-level temporal ordering of events,” inProceedings of the 20th
Annual SIGdial Meeting on Discourse and Dialogue, 2019, pp. 239–
249.
[86] V . D. Lai, A. P. B. Veyseh, M. V . Nguyen, F. Dernoncourt, and
T. H. Nguyen, “MECI: A multilingual dataset for event causality
identification,” inProceedings of the 29th International Conference
on Computational Linguistics, 2022, pp. 2346–2356.
[87] S. Deng, N. Zhang, L. Li, C. Hui, T. Huaixiao, M. Chen, F. Huang,
and H. Chen, “OntoED: Low-resource event detection with ontology
embedding,” inProceedings of the 59th Annual Meeting of the Asso-
ciation for Computational Linguistics and the 11th International Joint
Conference on Natural Language Processing (Volume 1: Long Papers),
2021, pp. 2828–2839.
[88] Z. Song, A. Bies, S. Strassel, T. Riese, J. Mott, J. Ellis, J. Wright,
S. Kulick, N. Ryant, and X. Ma, “From light to rich ERE: Annotation
of entities, relations, and events,” inProceedings of the 3rd Workshop
on EVENTS: Definition, Detection, Coreference, and Representation,
2015, pp. 89–98.
[89] M. D. Ma, A. Taylor, W. Wang, and N. Peng, “DICE: Data-efficient
clinical event extraction with generative models,” inProceedings of the
61st Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), 2023, pp. 15 898–15 917.
[90] S. Pyysalo, T. Ohta, M. Miwa, H.-C. Cho, J. Tsujii, and S. Ananiadou,
“Event extraction across multiple levels of biological organization,”
Bioinformatics, vol. 28, no. 18, pp. i575–i581, 2012.
[91] B. M. Sundheim, “Overview of the fourth Message Understanding
Evaluation and Conference,” inFourth Message Understanding Con-
ference (MUC-4): Proceedings of a Conference Held in McLean,
Virginia, June 16-18, 1992, 1992.
[92] M. Yatskar, L. Zettlemoyer, and A. Farhadi, “Situation recognition:
Visual semantic role labeling for image understanding,” inProceedings
of the IEEE conference on computer vision and pattern recognition,
2016, pp. 5534–5542.
[93] A. Sadhu, T. Gupta, M. Yatskar, R. Nevatia, and A. Kembhavi, “Visual
semantic role labeling for video understanding,” inProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2021, pp. 5589–5600.
[94] S. Pratt, M. Yatskar, L. Weihs, A. Farhadi, and A. Kembhavi,
“Grounded situation recognition,” inEuropean Conference on Com-
puter Vision. Springer, 2020, pp. 314–332.
[95] Y . Bitton, R. Yosef, E. Strugo, D. Shahaf, R. Schwartz, and
G. Stanovsky, “Vasr: Visual analogies of situation recognition,” in
Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37,
no. 1, 2023, pp. 241–249.
[96] Z. Khan, C. Jawahar, and M. Tapaswi, “Grounded video situation
recognition,”Advances in Neural Information Processing Systems,
vol. 35, pp. 8199–8210, 2022.
[97] Z. Liu, J. Ning, Y . Cao, Y . Wei, Z. Zhang, S. Lin, and H. Hu, “Video
swin transformer,” inProceedings of the IEEE/CVF conference on
computer vision and pattern recognition, 2022, pp. 3202–3211.

LIet al.: EVENT EXTRACTION 29
[98] C. Feichtenhofer, H. Fan, J. Malik, and K. He, “Slowfast networks
for video recognition,” inProceedings of the IEEE/CVF international
conference on computer vision, 2019, pp. 6202–6211.
[99] J. Carreira and A. Zisserman, “Quo vadis, action recognition? a new
model and the kinetics dataset,” inproceedings of the IEEE Conference
on Computer Vision and Pattern Recognition, 2017, pp. 6299–6308.
[100] B. Wang, M. Zhang, H. Fei, Y . Zhao, B. Li, S. Wu, W. Ji, and M. Zhang,
“Speechee: A novel benchmark for speech event extraction,” inPro-
ceedings of the 32nd ACM international conference on multimedia,
2024, pp. 10 449–10 458.
[101] S. Ghannay, A. Caubriere, Y . Esteve, A. Laurent, and E. Morin,
“End-to-end named entity extraction from speech,”arXiv preprint
arXiv:1805.12045, 2018.
[102] Y . Chiba and R. Higashinaka, “Dialogue situation recognition for
everyday conversation using multimodal information.” inInterspeech,
2021, pp. 241–245.
[103] A. Baevski, Y . Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A
framework for self-supervised learning of speech representations,”Ad-
vances in neural information processing systems, vol. 33, pp. 12 449–
12 460, 2020.
[104] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and
I. Sutskever, “Robust speech recognition via large-scale weak supervi-
sion,” inInternational conference on machine learning. PMLR, 2023,
pp. 28 492–28 518.
[105] T. Wu, G. Wang, J. Zhao, Z. Liu, G. Qi, Y .-F. Li, and G. Haffari,
“Towards relation extraction from speech,” inProceedings of the 2022
Conference on Empirical Methods in Natural Language Processing,
2022, pp. 10 751–10 762.
[106] M. Li, R. Xu, S. Wang, L. Zhou, X. Lin, C. Zhu, M. Zeng,
H. Ji, and S. Chang, “Clip-event: Connecting text and images with
event structures,” inIEEE/CVF Conference on Computer Vision and
Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June
18-24, 2022. IEEE, 2022, pp. 16 399–16 408. [Online]. Available:
https://doi.org/10.1109/CVPR52688.2022.01593
[107] M. Zhang, H. Fei, B. Wang, S. Wu, Y . Cao, F. Li, and M. Zhang, “Rec-
ognizing everything from all modalities at once: Grounded multimodal
universal information extraction,” inACL (Findings), 2024.
[108] M. Liu, B. Zhou, H. Hu, C. Qiu, and X. Zhang, “Cross-modal event
extraction via visual event grounding and semantic relation filling,”
Information Processing & Management, vol. 62, no. 3, p. 104027,
2025.
[109] H. Ayyubi, C. Thomas, L. Chum, R. Lokesh, L. Chen, Y . Niu, X. Lin,
X. Feng, J. Koo, S. Rayet al., “Beyond grounding: extracting fine-
grained event hierarchies across modalities,” inProceedings of the
AAAI Conference on Artificial Intelligence, vol. 38, no. 16, 2024, pp.
17 664–17 672.
[110] J. Ma, J. Wang, J. Luo, P. Yu, and G. Zhou, “Sherlock: Towards
multi-scene video abnormal event extraction and localization via a
global-local spatial-sensitive llm,” inProceedings of the ACM on Web
Conference 2025, 2025, pp. 4004–4013.
[111] H. Yang, Y . Chen, K. Liu, Y . Xiao, and J. Zhao, “DCFEE: A
document-level Chinese financial event extraction system based on
automatically labeled training data,” inProceedings of ACL 2018,
System Demonstrations, F. Liu and T. Solorio, Eds., 2018, pp. 50–55.
[112] A. Ramponi, R. Van Der Goot, R. Lombardo, and B. Plank, “Biomed-
ical event extraction as sequence labeling,” inProceedings of the 2020
Conference on Empirical Methods in Natural Language Processing
(EMNLP), 2020, pp. 5357–5367.
[113] Y . Zeng, Y . Feng, R. Ma, Z. Wang, R. Yan, C. Shi, and D. Zhao, “Scale
up event extraction learning via automatic training data generation,” in
Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32,
no. 1, 2018.
[114] W. Lu and D. Roth, “Automatic event extraction with structured
preference modeling,” inProceedings of the 50th Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers),
2012, pp. 835–844.
[115] J. Ferguson, C. Lockard, D. S. Weld, and H. Hajishirzi, “Semi-
supervised event extraction with paraphrase clusters,” inProceedings of
the 2018 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies, Volume
2 (Short Papers), 2018, pp. 359–364.
[116] E.-d. El-allaly, M. Sarrouti, N. En-Nahnahi, and S. O. El Alaoui,
“Mttlade: A multi-task transfer learning-based method for adverse drug
events extraction,”Information Processing & Management, vol. 58,
no. 3, p. 102473, 2021.
[117] S. Duan, R. He, and W. Zhao, “Exploiting document level information
to improve event detection via recurrent neural networks,” inProceed-ings of the Eighth International Joint Conference on Natural Language
Processing (Volume 1: Long Papers), 2017, pp. 352–361.
[118] D. Wadden, U. Wennberg, Y . Luan, and H. Hajishirzi, “Entity, relation,
and event extraction with contextualized span representations,” in
Proceedings of the 2019 Conference on Empirical Methods in Natural
Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP), 2019, pp. 5784–
5789.
[119] S. Yang, D. Feng, L. Qiao, Z. Kan, and D. Li, “Exploring pre-trained
language models for event extraction and generation,” inProceedings
of the 57th annual meeting of the association for computational
linguistics, 2019, pp. 5284–5294.
[120] F. Li, W. Peng, Y . Chen, Q. Wang, L. Pan, Y . Lyu, and Y . Zhu,
“Event extraction as multi-turn question answering,” inFindings of
the Association for Computational Linguistics: EMNLP 2020, 2020,
pp. 829–838.
[121] Y . Ma, Z. Wang, Y . Cao, M. Li, M. Chen, K. Wang, and J. Shao,
“Prompt for extraction? paie: Prompting argument interaction for event
argument extraction,” inProceedings of the 60th Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers),
2022, pp. 6759–6774.
[122] J. Sheng, S. Guo, B. Yu, Q. Li, Y . Hei, L. Wang, T. Liu, and
H. Xu, “Casee: A joint learning framework with cascade decoding
for overlapping event extraction,” inFindings of the Association for
Computational Linguistics: ACL-IJCNLP 2021, 2021, pp. 164–174.
[123] Y . Liang, Z. Jiang, D. Yin, and B. Ren, “Raat: Relation-augmented
attention transformer for relation modeling in document-level event
extraction,”arXiv preprint arXiv:2206.03377, 2022.
[124] Z. Zhang, W. Xu, and Q. Chen, “Joint event extraction based on skip-
window convolutional neural networks,” inInternational Conference
on Computer Processing of Oriental Languages. Springer, 2016, pp.
324–334.
[125] L. Li, Y . Liu, and M. Qin, “Extracting biomedical events with parallel
multi-pooling convolutional neural networks,”IEEE/ACM transactions
on computational biology and bioinformatics, vol. 17, no. 2, pp. 599–
607, 2018.
[126] L. Sha, F. Qian, B. Chang, and Z. Sui, “Jointly extracting event
triggers and arguments by dependency-bridge rnn and tensor-based
argument interaction,” inProceedings of the AAAI conference on
artificial intelligence, vol. 32, no. 1, 2018.
[127] Y . Lu, Q. Liu, D. Dai, X. Xiao, H. Lin, X. Han, L. Sun, and H. Wu,
“Unified structure generation for universal information extraction,”
arXiv preprint arXiv:2203.12277, 2022.
[128] H. Fei, S. Wu, J. Li, B. Li, F. Li, L. Qin, M. Zhang, M. Zhang, and T.-
S. Chua, “Lasuie: Unifying information extraction with latent adaptive
structure-aware generative language model,” inNeurIPS, 2022.
[129] Q. Wan, C. Wan, K. Xiao, D. Liu, C. Li, B. Zheng, X. Liu, and R. Hu,
“Joint document-level event extraction via token-token bidirectional
event completed graph,” inProceedings of the 61st Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long
Papers), 2023, pp. 10 481–10 492.
[130] S. Cui, J. Sheng, X. Cong, Q. Li, T. Liu, and J. Shi, “Event causality
extraction with event argument correlations,” inProceedings of the
29th International Conference on Computational Linguistics, 2022, pp.
2300–2312.
[131] J. Ning, Z. Yang, Z. Wang, Y . Sun, and H. Lin, “Odee: A one-
stage object detection framework for overlapping and nested event
extraction,” inProceedings of the Thirty-Second International Joint
Conference on Artificial Intelligence, 2023, pp. 5170–5178.
[132] R. Pu, Y . Li, J. Zhao, S. Wang, D. Li, J. Liao, and J. Zheng, “A joint
framework with heterogeneous-relation-aware graph and multi-channel
label enhancing strategy for event causality extraction,” inProceedings
of the AAAI Conference on Artificial Intelligence, vol. 38, no. 17, 2024,
pp. 18 879–18 887.
[133] X. Liu, H.-Y . Huang, G. Shi, and B. Wang, “Dynamic prefix-tuning
for generative template-based event extraction,” inProceedings of the
60th Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), 2022, pp. 5216–5228.
[134] W. Liu, S. Cheng, D. Zeng, and H. Qu, “Enhancing document-level
event argument extraction with contextual clues and role relevance,”
arXiv preprint arXiv:2310.05991, 2023.
[135] I. Hsu, Z. Xie, K.-H. Huang, P. Natarajan, N. Penget al., “Am-
pere: Amr-aware prefix for generation-based event argument extraction
model,”arXiv preprint arXiv:2305.16734, 2023.
[136] Y . Qi, H. Peng, X. Wang, B. Xu, L. Hou, and J. Li, “ADELIE: Aligning
large language models on information extraction,” inProceedings of

30 ARXIV PREPRINT
the 2024 Conference on Empirical Methods in Natural Language
Processing, 2024, pp. 7371–7387.
[137] C. Yuan, Q. Xie, J. Huang, and S. Ananiadou, “Back to the future:
Towards explainable temporal reasoning with large language models,”
inProceedings of the ACM Web Conference 2024, 2024, p. 1963–1974.
[138] Z. Hong and J. Liu, “Towards better question generation in QA-based
event extraction,” inFindings of the Association for Computational
Linguistics: ACL 2024, 2024, pp. 9025–9038.
[139] X. Wei, Y . Chen, N. Cheng, X. Cui, J. Xu, and W. Han, “Collabkg:
A learnable human-machine-cooperative information extraction toolkit
for (event) knowledge graph construction,” inProceedings of the
2024 Joint International Conference on Computational Linguistics,
Language Resources and Evaluation (LREC-COLING 2024), 2024, pp.
3490–3506.
[140] X. Wang, S. Li, and H. Ji, “Code4Struct: Code generation for few-
shot event structure prediction,” inProceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers), 2023, pp. 3640–3663.
[141] Z. Hu, Z. Li, X. Jin, L. Bai, J. Guo, and X. Cheng, “Large language
model-based event relation extraction with rationales,” inProceedings
of the 31st International Conference on Computational Linguistics,
2025, pp. 7484–7496.
[142] S. Srivastava, S. Pati, and Z. Yao, “Instruction-tuning LLMs for event
extraction with annotation guidelines,” inFindings of the Association
for Computational Linguistics: ACL 2025, 2025, pp. 13 055–13 071.
[143] B. Li, G. Fang, Y . Yang, Q. Wang, W. Ye, W. Zhao, and S. Zhang,
“Evaluating chatgpt’s information extraction capabilities: An assess-
ment of performance, explainability, calibration, and faithfulness,”
arXiv preprint arXiv:2304.11633, 2023.
[144] Y . Ma, Y . Cao, Y . Hong, and A. Sun, “Large language model is not
a good few-shot information extractor, but a good reranker for hard
samples!” inFindings of the Association for Computational Linguistics:
EMNLP 2023, 2023, pp. 10 572–10 601.
[145] C. Pang, Y . Cao, Q. Ding, and P. Luo, “Guideline learning for in-context
information extraction,” inProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing, 2023, pp. 15 372–
15 389.
[146] K. Bao and N. Wang, “General collaborative framework between large
language model and experts for universal information extraction,” in
Findings of the Association for Computational Linguistics: EMNLP
2024, 2024, pp. 52–77.
[147] O. Sainz, I. Garc ´ıa-Ferrero, R. Agerri, O. L. de Lacalle, G. Rigau,
and E. Agirre, “GoLLIE: Annotation guidelines improve zero-shot
information-extraction,” inThe Twelfth International Conference
on Learning Representations, 2024. [Online]. Available: https:
//openreview.net/forum?id=Y3wpuxd7u9
[148] M. Zhu, K. Zeng, J. JibingWu, L. Liu, H. Huang, L. Hou, and J. Li,
“LC4EE: LLMs as good corrector for event extraction,” inFindings of
the Association for Computational Linguistics: ACL 2024, 2024, pp.
12 028–12 038.
[149] X. Liao, J. Duan, Y . Huang, and J. Wang, “RUIE: Retrieval-based
unified information extraction using large language model,” inProceed-
ings of the 31st International Conference on Computational Linguistics,
2025, pp. 9640–9655.
[150] M. D. Ma, X. Wang, P.-N. Kung, P. J. Brantingham, N. Peng,
and W. Wang, “Star: boosting low-resource information extraction
by structure-to-text data generation with large language models,” in
Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38,
no. 17, 2024, pp. 18 751–18 759.
[151] M. Chen, Y . Ma, K. Song, Y . Cao, Y . Zhang, and D. Li, “Improving
large language models in event relation logical prediction,” inProceed-
ings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), 2024, pp. 9451–9478.
[152] S. Wang and L. Huang, “Debate as optimization: Adaptive conformal
prediction and diverse retrieval for event extraction,” inFindings of
the Association for Computational Linguistics: EMNLP 2024, 2024,
pp. 16 422–16 435.
[153] ——, “Targeted augmentation for low-resource event extraction,” in
Findings of the Association for Computational Linguistics: NAACL
2024, 2024, pp. 4414–4428.
[154] X. Jin and H. Ji, “Schema-based data augmentation for event ex-
traction,” inProceedings of the 2024 Joint International Conference
on Computational Linguistics, Language Resources and Evaluation
(LREC-COLING 2024), 2024, pp. 14 382–14 392.
[155] Y . Zhou, Y . Chen, J. Zhao, Y . Wu, J. Xu, and J. Li, “What the role
is vs. what plays the role: Semi-supervised event argument extractionvia dual question answering,” inProceedings of the AAAI conference
on artificial intelligence, vol. 35, no. 16, 2021, pp. 14 638–14 646.
[156] X. Yang, Y . Lu, and L. Petzold, “Few-shot document-level event
argument extraction,”arXiv preprint arXiv:2209.02203, 2022.
[157] S. He, X. Peng, Y . Cai, X. Li, Z. Yuan, W. Du, and W. Yang, “ZSEE:
A dataset based on zeolite synthesis event extraction for automated
synthesis platform,” inFindings of the Association for Computational
Linguistics: NAACL 2024, 2024, pp. 1791–1808.
[158] H. Deng, Y . Zhang, Y . Zhang, W. Ying, C. Yu, J. Gao, W. Wang, X. Bai,
N. Yang, J. Ma, X. Chen, and T. Zhou, “Title2Event: Benchmarking
open event extraction with a large-scale Chinese title dataset,” in
Proceedings of the 2022 Conference on Empirical Methods in Natural
Language Processing, 2022, pp. 6511–6524.
[159] H. Ji and R. Grishman, “Refining event extraction through cross-
document inference,” in46th Annual Meeting of the Association for
Computational Linguistics: Human Language Technologies, ACL-08:
HLT, 2008, pp. 254–262.
[160] K.-H. Huang, I. Hsu, T. Parekh, Z. Xie, Z. Zhang, P. Natarajan, K.-
W. Chang, N. Peng, H. Jiet al., “Textee: Benchmark, reevaluation,
reflections, and future challenges in event extraction,”arXiv preprint
arXiv:2311.09562, 2023.
[161] J. Dutkiewicz, M. Nowak, and C. Jedrzejek, “R2e: Rule-based event
extractor.” inChallenge+ DC@ RuleML, 2014.
[162] J.-D. Kim, Y . Wang, T. Takagi, and A. Yonezawa, “Overview of Genia
event task in BioNLP shared task 2011,” inProceedings of BioNLP
Shared Task 2011 Workshop, 2011, pp. 7–15.
[163] H. Huang, X. Liu, G. Shi, and Q. Liu, “Event extraction with dynamic
prefix tuning and relevance retrieval,”IEEE Transactions on Knowledge
and Data Engineering, vol. 35, no. 10, pp. 9946–9958, 2023.
[164] X. Du, A. M. Rush, and C. Cardie, “Grit: Generative role-filler
transformers for document-level event entity extraction,”arXiv preprint
arXiv:2008.09249, 2020.
[165] T. Parekh, I. Hsu, K.-H. Huang, K.-W. Chang, N. Penget al.,
“Geneva: Benchmarking generalizability for event argument extraction
with hundreds of event types and argument roles,”arXiv preprint
arXiv:2205.12505, 2022.
[166] Y . Feng, C. Li, and V . Ng, “Legal judgment prediction via event ex-
traction with constraints,” inProceedings of the 60th annual meeting of
the association for computational linguistics (volume 1: long papers),
2022, pp. 648–664.
[167] J. Zhao, L. Li, W. Ning, J. Hao, Y . Fei, and J. Huang, “Multiple
optimization with retrieval-augmented generation and fine-grained de-
noising for biomedical event causal relation extraction,” in2024 IEEE
International Conference on Bioinformatics and Biomedicine (BIBM),
2024, pp. 4040–4043.
[168] K. Shuang, Z. Zhouji, W. Qiwei, and J. Guo, “Thinking about how
to extract: Energizing LLMs’ emergence capabilities for document-
level event argument extraction,” inFindings of the Association for
Computational Linguistics: ACL 2024, 2024, pp. 5520–5532.
[169] K.-H. Huang, M. Yang, and N. Peng, “Biomedical event extraction
with hierarchical knowledge graphs,”arXiv preprint arXiv:2009.09335,
2020.
[170] A. P. B. Veyseh, T. N. Nguyen, and T. H. Nguyen, “Graph transformer
networks with syntactic and semantic structures for event argument
extraction,”arXiv preprint arXiv:2010.13391, 2020.
[171] V . D. Lai, M. Van Nguyen, H. Kaufman, and T. H. Nguyen, “Event
extraction from historical texts: A new dataset for black rebellions,”
inFindings of the Association for Computational Linguistics: ACL-
IJCNLP 2021, 2021, pp. 2390–2400.
[172] H. Sun, J. Zhou, L. Kong, Y . Gu, and W. Qu, “Seq2eg: a novel and
effective event graph parsing approach for event extraction,”Knowledge
and Information Systems, vol. 65, no. 10, pp. 4273–4294, 2023.
[173] M. Joshi, D. Chen, Y . Liu, D. S. Weld, L. Zettlemoyer, and O. Levy,
“Spanbert: Improving pre-training by representing and predicting
spans,”Transactions of the association for computational linguistics,
vol. 8, pp. 64–77, 2020.
[174] Y . Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis,
L. Zettlemoyer, and V . Stoyanov, “Roberta: A robustly optimized bert
pretraining approach,”arXiv preprint arXiv:1907.11692, 2019.
[175] M. Lewis, Y . Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy,
V . Stoyanov, and L. Zettlemoyer, “Bart: Denoising sequence-to-
sequence pre-training for natural language generation, translation, and
comprehension,”arXiv preprint arXiv:1910.13461, 2019.
[176] W. U. Ahmad, N. Peng, and K.-W. Chang, “Gate: graph attention
transformer encoder for cross-lingual relation and event extraction,” in
Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35,
no. 14, 2021, pp. 12 462–12 470.

LIet al.: EVENT EXTRACTION 31
[177] Q. Xi, Y . Ren, L. Kou, Y . Cui, Z. Chen, L. Yuan, and D. Wang, “Eabert:
An event annotation enhanced bert framework for event extraction,”
Mobile Networks and Applications, vol. 28, no. 5, pp. 1818–1830,
2023.
[178] L. Ding, X. Chen, J. Wei, and Y . Xiang, “Mabert: mask-attention-
based bert for chinese event extraction,”ACM Transactions on Asian
and Low-Resource Language Information Processing, vol. 22, no. 7,
pp. 1–21, 2023.
[179] R. Xu, P. Wang, T. Liu, S. Zeng, B. Chang, and Z. Sui, “A two-stream
amr-enhanced model for document-level event argument extraction,” in
Proceedings of the 2022 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language
Technologies, 2022, pp. 5025–5036.
[180] Y . Yang, Q. Guo, X. Hu, Y . Zhang, X. Qiu, and Z. Zhang, “An amr-
based link prediction approach for document-level event argument ex-
traction,” inProceedings of the 61st Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), 2023, pp.
12 876–12 889.
[181] Q. Wan, L. Liutao, C. Wan, R. Hu, K. Xiao, and Y . Shuai, “Event
pattern-instance graph: A multi-round role representation learning
strategy for document-level event argument extraction,” inFindings of
the Association for Computational Linguistics: ACL 2025, 2025, pp.
1865–1877.
[182] H. Trieu, T. T. Tran, K. Duong, A.-T. Nguyen, M. Miwa,
M. Miwa, and S. Ananiadou, “Deepeventmine: end-to-end neural
nested event extraction from biomedical texts,”Bioinformatics,
vol. 36, pp. 4910 – 4917, 2020. [Online]. Available: https:
//api.semanticscholar.org/CorpusID:221471677
[183] T. Parekh, J. Kwan, J. Yu, S. Johri, H. Ahn, S. Muppalla, K.-W. Chang,
W. Wang, and N. Peng, “SPEED++: A multilingual event extraction
framework for epidemic prediction and preparedness,” inProceedings
of the 2024 Conference on Empirical Methods in Natural Language
Processing, 2024, pp. 12 936–12 965.
[184] K. D’Oosterlinck, F. Remy, J. Deleu, T. Demeester, C. Develder,
K. Zaporojets, A. Ghodsi, S. Ellershaw, J. Collins, and C. Potts,
“BioDEX: Large-scale biomedical adverse drug event extraction for
real-world pharmacovigilance,” inFindings of the Association for
Computational Linguistics: EMNLP 2023, 2023, pp. 13 425–13 454.
[185] D. Kim, R. Yoo, S. Yang, H. Yang, and J. Choo, “AniEE: A dataset of
animal experimental literature for event extraction,” inFindings of the
Association for Computational Linguistics: EMNLP 2023, 2023, pp.
12 959–12 971.
[186] H. Peng, R. Zhang, S. Li, Y . Cao, S. Pan, and P. S. Yu,
“Reinforced, incremental and cross-lingual event detection from
social messages,”IEEE Transactions on Pattern Analysis and
Machine Intelligence, vol. 45, pp. 980–998, 2022. [Online]. Available:
https://api.semanticscholar.org/CorpusID:246286553
[187] T. Zhang, M. Liu, and B. Zhou, “Cfindee: A chinese fine-grained
financial dataset for document-level event extraction,” inCompanion
Proceedings of the ACM Web Conference 2024, 2024, p. 1511–1520.
[188] Q. Wan, C. Wan, R. Hu, D. Liu, X. Wenwu, K. Xu, Z. Meihua, L. Tao,
J. Yang, and Z. Xiong, “OEE-CFC: A dataset for open event extraction
from Chinese financial commentary,” inFindings of the Association for
Computational Linguistics: EMNLP 2024, 2024, pp. 4446–4459.
[189] M. Wu, M. Liu, L. Wang, and H. Hu, “A chinese fine-grained financial
event extraction dataset,” inCompanion Proceedings of the ACM Web
Conference 2023, 2023, p. 1229–1235.
[190] F. Yao, C. Xiao, X. Wang, Z. Liu, L. Hou, C. Tu, J. Li,
Y . Liu, W. Shen, and M. Sun, “Leven: A large-scale chinese legal
event detection dataset,” inFindings, 2022. [Online]. Available:
https://api.semanticscholar.org/CorpusID:247476106
[191] Z. Xue, H. Liu, Y . Hu, K. Kong, C. Wang, Y . Liu, and W. Shen, “Leec:
A legal element extraction dataset with an extensive domain-specific
label system,”ArXiv, vol. abs/2310.01271, 2023. [Online]. Available:
https://api.semanticscholar.org/CorpusID:263605927
[192] B. Min, B. Rozonoyer, H. Qiu, A. Zamanian, N. Xue, and J. MacBride,
“Excavatorcovid: Extracting events and relations from text corpora for
temporal and causal analysis for covid-19,” inProceedings of the 2021
Conference on Empirical Methods in Natural Language Processing:
System Demonstrations, 2021, pp. 63–71.
[193] I. Guellil, S. Andres, A. Anand, B. Guthrie, H. Zhang, A. Hasan,
H. Wu, and B. Alex, “Adverse event extraction from discharge
summaries: A new dataset, annotation scheme, and initial findings,”
inProceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), 2025, pp. 28 532–
28 562.[194] N. Zhang, X. Xu, L. Tao, H. Yu, H. Ye, S. Qiao, X. Xie, X. Chen,
Z. Li, L. Liet al., “Deepke: A deep learning based knowledge
extraction toolkit for knowledge base population,”arXiv preprint
arXiv:2201.03335, 2022.
[195] Z. Kan, L. Peng, Y . Gao, N. Liu, L. Qiao, and D. Li, “Lfde: A
lighter, faster and more data-efficient pre-training framework for event
extraction,” inProceedings of the ACM Web Conference 2024, 2024,
pp. 3964–3975.
[196] M. Eberts and A. Ulges, “Span-based joint entity and relation extraction
with transformer pre-training,”arXiv preprint arXiv:1909.07755, 2019.
[197] H. Wen, Y . Lin, T. Lai, X. Pan, S. Li, X. Lin, B. Zhou, M. Li,
H. Wang, H. Zhanget al., “Resin: A dockerized schema-guided cross-
document cross-lingual cross-media information extraction and event
tracking system,” inProceedings of the 2021 Conference of the North
American Chapter of the Association for Computational Linguistics:
Human Language Technologies: Demonstrations, 2021, pp. 133–143.
[198] F. Hogenboom, F. Frasincar, U. Kaymak, and F. de Jong, “An overview
of event extraction from text,” inDeRiVE@ISWC, 2011, pp. 48–57.
[199] D. Le and T. H. Nguyen, “Fine-grained event trigger detection,” in
Proceedings of EACL, 2021, pp. 2745–2752.
[200] H. Wang, M. Chen, H. Zhang, and D. Roth, “Joint constrained learning
for event-event relation extraction,” inProceedings of EMNLP, 2020,
pp. 696–706.
[201] S. Liu, Y . Chen, K. Liu, and J. Zhao, “Exploiting argument information
to improve event detection via supervised attention mechanisms,” in
Proceedings of ACL, 2017, pp. 1789–1798.
[202] Z. Liu, T. Mitamura, and E. H. Hovy, “Graph based decoding for event
sequencing and coreference resolution,” inProceedings of COLING,
2018, pp. 3645–3657.
[203] J. Lu and V . Ng, “Event coreference resolution: A survey of two
decades of research,” inProceedings of IJCAI, 2018, pp. 5479–5486.
[204] P. P.-S. Chen, “Entity-relationship modeling: Historical events, future
trends, and lessons learned,” inSoftware Pioneers, 2002, pp. 296–310.
[205] Z. Wang, Y . Zhang, and C.-Y . Chang, “Integrating order information
and event relation for script event prediction,” inProceedings of
EMNLP, 2017, pp. 57–67.
[206] J. Pustejovsky, R. Ingria, R. Saur ´ı, J. M. Casta ˜no, J. Littman, R. J.
Gaizauskas, A. Setzer, G. Katz, and I. Mani, “The specification
language timeml,” inThe Language of Time - A Reader, 2005, pp.
545–558.
[207] P. Mirza, “Extracting temporal and causal relations between events,”
inProceedings of ACL, 2014, pp. 10–17.
[208] J. Araki, Z. Liu, E. H. Hovy, and T. Mitamura, “Detecting subevent
structure for event coreference resolution,” inProceedings of LREC,
2014, pp. 4553–4558.
[209] G. Yang, M. Li, J. Zhang, X. Lin, H. Ji, and S.-F. Chang, “Video event
extraction via tracking visual states of arguments,” inProceedings of
the AAAI conference on artificial intelligence, vol. 37, no. 3, 2023, pp.
3136–3144.
[210] H. Kilicoglu and S. Bergler, “Syntactic dependency based heuristics
for biological event extraction,” inProceedings of the BioNLP 2009
Workshop Companion Volume for Shared Task, 2009, pp. 119–127.
[211] R. Sun, W. Zhou, and Z.-T. Liu, “Using rules to extract event informa-
tion from sentences,”Journal of Chinese Computer Systems, vol. 32,
no. 11, pp. 2309–2314, 2011.
[212] M. A. Valenzuela-Esc ´arcega, G. Hahn-Powell, M. Surdeanu, and
T. Hicks, “A domain-independent rule-based framework for event ex-
traction,” inProceedings of ACL-IJCNLP 2015 system demonstrations,
2015, pp. 127–132.
[213] V . Danilova and S. Popova, “Socio-political event extraction using
a rule-based approach,” inOTM Confederated International Confer-
ences” On the Move to Meaningful Internet Systems”. Springer, 2014,
pp. 537–546.
[214] R. Nitschke, Y . Wang, C. Chen, A. Pyarelal, and R. Sharp, “Rule based
event extraction for artificial social intelligence,” inProceedings of the
First Workshop on Pattern-based Approaches to NLP in the Age of
Deep Learning, 2022, pp. 71–84.
[215] A. Kova ˇcevi´c, A. Dehghan, M. Filannino, J. A. Keane, and G. Nenadic,
“Combining rules and machine learning for extraction of temporal ex-
pressions and events from clinical narratives,”Journal of the American
Medical Informatics Association, vol. 20, no. 5, pp. 859–866, 2013.
[216] V . Guda and S. Sanampudi, “A hybrid method for extraction of
events from natural language text,” inData Engineering and Intelligent
Computing: Proceedings of IC3T 2016. Springer, 2017, pp. 301–307.
[217] X. Gao, Z. Diao, K. Wei, Y . Yang, and L. Li, “Event extraction via rules
and machine learning,” in2019 IEEE 6th International Conference on

32 ARXIV PREPRINT
Cloud Computing and Intelligence Systems (CCIS). IEEE, 2019, pp.
41–46.
[218] V . Guda and S. K. Sanampudi, “Rules based event extraction from
natural language text,” in2016 IEEE International Conference on Re-
cent Trends in Electronics, Information & Communication Technology
(RTEICT). IEEE, 2016, pp. 9–13.
[219] M. Naughton, N. Kushmerick, and J. Carthy, “Event extraction from
heterogeneous news sources,” inproceedings of the AAAI workshop
event extraction and synthesis, 2006, pp. 1–6.
[220] H. Tanev, J. Piskorski, and M. Atkinson, “Real-time news event
extraction for global crisis monitoring,” inInternational Conference on
Application of Natural Language to Information Systems. Springer,
2008, pp. 207–218.
[221] T. Sakaki, Y . Matsuo, T. Yanagihara, N. P. Chandrasiri, and K. Nawa,
“Real-time event extraction for driving information from social sen-
sors,” in2012 IEEE International Conference on Cyber Technology in
Automation, Control, and Intelligent Systems (CYBER). IEEE, 2012,
pp. 221–226.
[222] S. Patwardhan and E. Riloff, “A unified model of phrasal and sentential
evidence for information extraction,” inProceedings of the 2009
conference on empirical methods in natural language processing, 2009,
pp. 151–160.
[223] M. Miwa, R. Sætre, J.-D. Kim, and J. Tsujii, “Event extraction
with complex event classification using rich features,”Journal of
bioinformatics and computational biology, vol. 8, no. 01, pp. 131–146,
2010.
[224] Y . Hong, J. Zhang, B. Ma, J. Yao, G. Zhou, and Q. Zhu, “Using cross-
entity inference to improve event extraction,” inProceedings of the
49th annual meeting of the association for computational linguistics:
human language technologies, 2011, pp. 1127–1136.
[225] Z. Li, F. Liu, L. Antieau, Y . Cao, and H. Yu, “Lancet: a high precision
medication event extraction system for clinical text,”Journal of the
American Medical Informatics Association, vol. 17, no. 5, pp. 563–
567, 2010.
[226] J. Bj ¨orne and T. Salakoski, “Generalizing biomedical event extraction,”
inProceedings of BioNLP Shared Task 2011 Workshop, 2011, pp. 183–
191.
[227] Y . Peng, M. Moh, and T.-S. Moh, “Efficient adverse drug event extrac-
tion using twitter sentiment analysis,” in2016 IEEE/ACM International
Conference on Advances in Social Networks Analysis and Mining
(ASONAM). IEEE, 2016, pp. 1011–1018.
[228] M. Smadi and O. Qawasmeh, “A supervised machine learning approach
for events extraction out of arabic tweets,” in2018 Fifth International
Conference on Social Networks Analysis, Management and Security
(SNAMS). IEEE, 2018, pp. 114–119.
[229] S. Henn, A. Sticha, T. Burley, E. Verdeja, and P. Brenner, “Visualization
techniques to enhance automated event extraction,”arXiv preprint
arXiv:2106.06588, 2021.
[230] D. Kodelja, R. Besanc ¸on, and O. Ferret, “Exploiting a more global
context for event detection through bootstrapping,” inEuropean con-
ference on information retrieval. Springer, 2019, pp. 763–770.
[231] Y . Chen, S. Liu, S. He, K. Liu, and J. Zhao, “Event extraction
via bidirectional long short-term memory tensor neural networks,” in
International Symposium on Natural Language Processing Based on
Naturally Annotated Big Data. Springer, 2016, pp. 190–203.
[232] D. Li, L. Huang, H. Ji, and J. Han, “Biomedical event extraction based
on knowledge-driven tree-lstm,” inProceedings of the 2019 Conference
of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies, Volume 1 (Long and Short
Papers), 2019, pp. 1421–1430.
[233] Y . Zhang, G. Xu, Y . Wang, X. Liang, L. Wang, and T. Huang,
“Empower event detection with bi-directional neural language model,”
Knowledge-Based Systems, vol. 167, pp. 87–97, 2019.
[234] S. Scaboro, B. Portelli, E. Chersoni, E. Santus, and G. Serra, “Extensive
evaluation of transformer-based architectures for adverse drug events
extraction,”Knowledge-Based Systems, vol. 275, p. 110675, 2023.
[235] A. Aljabari, L. Duaibes, M. Jarrar, and M. Khalilia, “Event-arguments
extraction corpus and modeling using bert for arabic,” inProceedings
of The Second Arabic Natural Language Processing Conference, 2024,
pp. 309–319.
[236] H. Yan, X. Jin, X. Meng, J. Guo, and X. Cheng, “Event detection
with multi-order graph convolution and aggregated attention,” inPro-
ceedings of the 2019 conference on empirical methods in natural
language processing and the 9th international joint conference on
natural language processing (EMNLP-IJCNLP), 2019, pp. 5766–5770.
[237] V . D. Lai, T. N. Nguyen, and T. H. Nguyen, “Event detection:
Gate diversity and syntactic importance scores for graph convolutionneural networks,” inProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP), 2020, pp. 5405–
5411.
[238] A. Balali, M. Asadpour, R. Campos, and A. Jatowt, “Joint event
extraction along shortest dependency paths using graph convolutional
networks,”Knowledge-Based Systems, vol. 210, p. 106492, 2020.
[239] Q. Wan, C. Wan, K. Xiao, R. Hu, and D. Liu, “A multi-channel
hierarchical graph attention network for open event extraction,”ACM
Transactions on Information Systems, vol. 41, no. 1, pp. 1–27, 2023.
[240] Q. Wan, C. Wan, K. Xiao, R. Hu, D. Liu, G. Liao, X. Liu, and
Y . Shuai, “A multifocal graph-based neural network scheme for topic
event extraction,”ACM Transactions on Information Systems, vol. 43,
no. 1, pp. 1–36, 2024.
[241] S. Liu, K. Liu, S. He, and J. Zhao, “A probabilistic soft logic
based approach to exploiting latent and global information in event
classification,” inProceedings of the AAAI Conference on Artificial
Intelligence, vol. 30, no. 1, 2016.
[242] S. Liu, Y . Chen, S. He, K. Liu, and J. Zhao, “Leveraging framenet to
improve automatic event detection,” inProceedings of ACL, 2016.
[243] T. M. Nguyen and T. H. Nguyen, “One for all: Neural joint modeling
of entities and events,” inAAAI, 2019, pp. 6851–6858.
[244] T. Zhang, H. Ji, and A. Sil, “Joint entity and event extraction with
generative adversarial imitation learning,”Data Intell., vol. 1, no. 2,
pp. 99–120, 2019.
[245] Z. Li, Y . Zeng, Y . Zuo, W. Ren, W. Liu, M. Su, Y . Guo, Y . Liu,
L. Lixiang, Z. Hu, L. Bai, W. Li, Y . Liu, P. Yang, X. Jin, J. Guo, and
X. Cheng, “KnowCoder: Coding structured knowledge into LLMs for
universal information extraction,” inProceedings of the 62nd Annual
Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers), 2024, pp. 8758–8779.
[246] X. F. Zhang, C. Blum, T. Choji, S. Shah, and A. Vempala, “UL-
TRA: Unleash LLMs’ potential for event argument extraction through
hierarchical modeling and pair-wise self-refinement,” inFindings of
the Association for Computational Linguistics: ACL 2024, Bangkok,
Thailand, 2024, pp. 8172–8185.
[247] Z. Cai, P.-N. Kung, A. Suvarna, M. Ma, H. Bansal, B. Chang, P. J.
Brantingham, W. Wang, and N. Peng, “Improving event definition
following for zero-shot event detection,” inProceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), 2024, pp. 2842–2863.
[248] M. Li, R. Xu, S. Wang, L. Zhou, X. Lin, C. Zhu, M. Zeng, H. Ji,
and S. Chang, “Clip-event: Connecting text and images with event
structures,” inProceedings of CVPR, 2022, pp. 16 399–16 408.
[249] J. Liu, Y . Chen, and J. Xu, “Multimedia event extraction from news
with a unified contrastive learning framework,” inProceedings of the
30th ACM international conference on multimedia, 2022, pp. 1945–
1953.
[250] P. Seeberger, D. Wagner, and K. Riedhammer, “Mmutf: Multimodal
multimedia event argument extraction with unified template filling,”
inFindings of the Association for Computational Linguistics: EMNLP
2024, 2024, pp. 6539–6548.
[251] Q. Team, “Qwen2-vl technical report,” https://qwenlm.github.io/blog/
qwen2-vl/, August 2024. [Online]. Available: https://qwenlm.github.
io/blog/qwen2-vl/
[252] J. Yu, Y . Lin, Z. Gao, X. Qiu, and L. Rui, “Multimedia event extraction
with LLM knowledge editing,” inProceedings of the 2025 Conference
on Empirical Methods in Natural Language Processing, Suzhou, China,
2025, pp. 4116–4124.
[253] S. He, Y . Hong, S. Yang, J. Yao, and G. Zhou, “Demonstration retrieval-
augmented generative event argument extraction,” inProceedings of
the 2024 Joint International Conference on Computational Linguistics,
Language Resources and Evaluation (LREC-COLING 2024), 2024, pp.
4617–4625.
[254] Z. Kan, L. Peng, L. Qiao, and D. Li, “Emancipating event extraction
from the constraints of long-tailed distribution data utilizing large
language models,” inProceedings of the 2024 Joint International
Conference on Computational Linguistics, Language Resources and
Evaluation (LREC-COLING 2024), 2024, pp. 5644–5653.
[255] H. Zhou, J. Qian, Z. Feng, L. Hui, Z. Zhu, and K. Mao, “LLMs
learn task heuristics from demonstrations: A heuristic-driven prompting
strategy for document-level event argument extraction,” inProceedings
of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), 2024, pp. 11 972–11 990.
[256] Y . Fu, Y . Cao, Q. Wang, and Y . Liu, “TISE: A tripartite in-context
selection method for event argument extraction,” inProceedings of the
2024 Conference of the North American Chapter of the Association for

LIet al.: EVENT EXTRACTION 33
Computational Linguistics: Human Language Technologies (Volume 1:
Long Papers), 2024, pp. 1801–1818.
[257] W. Zhang, W. Lu, J. Wang, Y . Wang, L. Chen, H. Jiang, J. Liu, and
T. Ruan, “Unexpected phenomenon: LLMs’ spurious associations in
information extraction,” inFindings of the Association for Computa-
tional Linguistics: ACL 2024, 2024, pp. 9176–9190.
[258] Y . Wei, K. Shuang, Z. Li, and C. Mao, “How do LLMs’ preferences
affect event argument extraction? CAT: Addressing preference traps in
unsupervised EAE,” inFindings of the Association for Computational
Linguistics: ACL 2025, 2025, pp. 19 529–19 543.
[259] Y . Guan, H. Peng, L. Hou, and J. Li, “MMD-ERE: Multi-agent multi-
sided debate for event relation extraction,” inProceedings of the
31st International Conference on Computational Linguistics, 2025, pp.
6889–6896.
[260] M. Choudhary and X. Du, “QAEVENT: Event extraction as question-
answer pairs generation,” inFindings of the Association for Computa-
tional Linguistics: EACL 2024, 2024, pp. 1860–1873.
[261] R. Chen, C. Qin, W. Jiang, and D. Choi, “Is a large language model
a good annotator for event extraction?” inProceedings of the AAAI
conference on artificial intelligence, vol. 38, no. 16, 2024, pp. 17 772–
17 780.
[262] M. N. Uddin, E. George, E. Blanco, and S. Corman, “Generating
uncontextualized and contextualized questions for document-level event
argument extraction,” inProceedings of the 2024 Conference of the
North American Chapter of the Association for Computational Linguis-
tics: Human Language Technologies (Volume 1: Long Papers), 2024,
pp. 5612–5627.
[263] Z. Meng, T. Liu, H. Zhang, K. Feng, and P. Zhao, “CEAN: Contrastive
event aggregation network with LLM-based augmentation for event
extraction,” inProceedings of the 18th Conference of the European
Chapter of the Association for Computational Linguistics (Volume 1:
Long Papers), 2024, pp. 321–333.
[264] J. Zhou, K. Shuang, Q. Wang, B. Qian, and J. Guo, “Bi-directional
feature learning-based approach for zero-shot event argument extrac-
tion,”Information Processing & Management, vol. 62, no. 5, p. 104199,
2025.
[265] X. Bao, J. Gu, Z. Wang, M. Qiang, and C.-R. Huang, “Employing
glyphic information for Chinese event extraction with vision-language
model,” inFindings of the Association for Computational Linguistics:
EMNLP 2024, 2024, pp. 1068–1080.
[266] X. Bao, Z. Wang, J. Gu, and C.-R. Huang, “Revisiting classical Chinese
event extraction with ancient literature information,” inProceedings
of the 63rd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), 2025, pp. 8440–8451.
[267] S. Zhao, T. Liu, S. Zhao, Y . Chen, and J.-Y . Nie, “Event causality
extraction based on connectives analysis,”Neurocomputing, vol. 173,
pp. 1943–1950, 2016.
[268] C. Chen and V . Ng, “Joint modeling for chinese event extraction with
rich linguistic features,” inProceedings of COLING 2012, 2012, pp.
529–544.
[269] D. McClosky, M. Surdeanu, and C. D. Manning, “Event extraction
as dependency parsing,” inProceedings of the 49th annual meeting
of the association for computational linguistics: human language
technologies, 2011, pp. 1626–1635.
[270] Z. Wang, X. Wang, X. Han, Y . Lin, L. Hou, Z. Liu, P. Li, J. Li,
and J. Zhou, “Cleve: Contrastive pre-training for event extraction,”
inProceedings of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International Joint Conference
on Natural Language Processing (Volume 1: Long Papers), 2021, pp.
6283–6297.
[271] L. Huang, T. Cassidy, X. Feng, H. Ji, C. V oss, J. Han, and A. Sil,
“Liberal event extraction and event schema induction,” inProceedings
of the 54th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), 2016, pp. 258–268.
[272] R. Han, Q. Ning, and N. Peng, “Joint event and temporal relation
extraction with shared representations and structured prediction,” in
Proceedings of the 2019 Conference on Empirical Methods in Natural
Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP), 2019, pp. 434–444.
[273] H. Yang, Y . Chen, K. Liu, Y . Xiao, and J. Zhao, “Dcfee: A document-
level chinese financial event extraction system based on automatically
labeled training data,” inProceedings of ACL 2018, System Demon-
strations, 2018, pp. 50–55.
[274] R. He, Y . Zhang, T. Li, and Q. Hu, “Improved conditional random fields
model with multi-trigger embedding for chinese event extraction,”
World Wide Web, vol. 17, no. 5, pp. 1029–1049, 2014.[275] K. Wei, X. Sun, Z. Zhang, J. Zhang, G. Zhi, and L. Jin, “Trigger
is not sufficient: Exploiting frame-aware knowledge for implicit event
argument extraction,” inProceedings of the 59th Annual Meeting of the
Association for Computational Linguistics and the 11th International
Joint Conference on Natural Language Processing (Volume 1: Long
Papers), 2021, pp. 4672–4682.
[276] J. Liu, Y . Chen, and J. Xu, “Machine reading comprehension as data
augmentation: A case study on implicit event argument extraction,” in
Proceedings of the 2021 Conference on Empirical Methods in Natural
Language Processing, 2021, pp. 2716–2725.
[277] H. Chen, “Dynamic grid tagging scheme for event causality identifi-
cation and classification,”IEEE Transactions on Audio, Speech and
Language Processing, vol. 33, pp. 2516–2526, 2025.
[278] J. Xu, M. Sun, Z. Zhang, and J. Zhou, “Maqinstruct: Instruction-based
unified event relation extraction,” inCompanion Proceedings of the
ACM on Web Conference 2025, 2025, pp. 1441–1445.
[279] X. Du and H. Ji, “Retrieval-augmented generative question answering
for event argument extraction,” inProceedings of the 2022 Conference
on Empirical Methods in Natural Language Processing, 2022, pp.
4649–4666.
[280] D. Lu, S. Ran, J. Tetreault, and A. Jaimes, “Event extraction as question
generation and answering,” inProceedings of the 61st Annual Meeting
of the Association for Computational Linguistics (Volume 2: Short
Papers), 2023, pp. 1666–1688.
[281] Z. Hong and J. Liu, “Towards better question generation in qa-based
event extraction,” inFindings of the Association for Computational
Linguistics ACL 2024, 2024, pp. 9025–9038.
[282] R. Wang, D. Zhou, and Y . He, “Open event extraction from online text
using a generative adversarial network,” inProceedings of the 2019
Conference on Empirical Methods in Natural Language Processing and
the 9th International Joint Conference on Natural Language Processing
(EMNLP-IJCNLP), 2019, pp. 282–291.
[283] Y . Ren, Y . Cao, P. Guo, F. Fang, W. Ma, and Z. Lin, “Retrieve-and-
sample: Document-level event argument extraction via hybrid retrieval
augmentation,” inProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers),
2023, pp. 293–306.
[284] R. Xu, T. Liu, L. Li, and B. Chang, “Document-level event extraction
via heterogeneous graph-based interaction model with a tracker,”
inProceedings of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International Joint Conference
on Natural Language Processing (Volume 1: Long Papers), 2021, pp.
3533–3546.
[285] L. Gao, Z. Yang, J. Ning, K. Ma, L. Wang, W. Liu, Y . Zhang,
L. Luo, B. Xu, J. Wanget al., “Biomedical event extraction as
semantic segmentation,” in2024 IEEE International Conference on
Bioinformatics and Biomedicine (BIBM). IEEE, 2024, pp. 3218–3221.
[286] D. Duki ´c, K. Gashteovski, G. Glava ˇs, and J. ˇSnajder, “Leveraging open
information extraction for more robust domain transfer of event trigger
detection,”arXiv preprint arXiv:2305.14163, 2023.
[287] S. Abdulkadhar, B. Bhasuran, and J. Natarajan, “Multiscale laplacian
graph kernel combined with lexico-syntactic patterns for biomedical
event extraction from literature,”Knowledge and Information Systems,
vol. 63, no. 1, pp. 143–173, 2021.
[288] Y . Ren and Z. Wang, “A tree-structured neural network model for
joint extraction of adverse drug events,” in2023 IEEE International
Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2023,
pp. 4938–4941.
[289] Z. Zhang, E. Strubell, and E. Hovy, “Transfer learning from semantic
role labeling to event argument extraction with template-based slot
querying,” inProceedings of the 2022 Conference on Empirical Meth-
ods in Natural Language Processing, 2022, pp. 2627–2647.
[290] M. Van Nguyen, B. Min, F. Dernoncourt, and T. Nguyen, “Learning
cross-task dependencies for joint extraction of entities, events, event
arguments, and relations,” inProceedings of the 2022 conference on
empirical methods in natural language processing, 2022, pp. 9349–
9360.
[291] N. Ding, C. Hu, K. Sun, S. Mensah, and R. Zhang, “Explicit role
interaction network for event argument extraction,” inFindings of the
Association for Computational Linguistics: EMNLP 2022, 2022, pp.
3475–3485.
[292] J. Tao, Y . Pan, X. Li, B. Hu, W. Peng, C. Han, and X. Wang, “Multi-role
event argument extraction as machine reading comprehension with ar-
gument match optimization,” inICASSP 2022-2022 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP).
IEEE, 2022, pp. 6347–6351.

34 ARXIV PREPRINT
[293] S. Menini, “Semantic frame extraction in multilingual olfactory events,”
inProceedings of the 2024 joint international conference on com-
putational linguistics, language resources and evaluation (lrec-coling
2024), 2024, pp. 14 622–14 627.
[294] X. Huang, R. He, and F. Huang, “Dual-level amr injection for prompt-
based event argument extraction,” inICASSP 2025-2025 IEEE In-
ternational Conference on Acoustics, Speech and Signal Processing
(ICASSP). IEEE, 2025, pp. 1–5.
[295] J. Tang, H. Lin, M. Liao, Y . Lu, X. Han, L. Sun, W. Xie, and J. Xu,
“From discourse to narrative: Knowledge projection for event relation
extraction,”arXiv preprint arXiv:2106.08629, 2021.
[296] H. Wen, Y . Qu, H. Ji, Q. Ning, J. Han, A. Sil, H. Tong, and D. Roth,
“Event time extraction and propagation via graph attention networks,”
inProceedings of the 2021 conference of the North American chapter
of the association for computational linguistics: human language
technologies, 2021, pp. 62–73.
[297] Y . Ren, Y . Cao, F. Fang, P. Guo, Z. Lin, W. Ma, and Y . Liu, “Clio:
Role-interactive multi-event head attention network for document-level
event extraction,” inProceedings of the 29th International Conference
on Computational Linguistics, 2022, pp. 2504–2514.
[298] H. Li, Y . Cao, Y . Ren, F. Fang, L. Zhang, Y . Li, and S. Wang,
“Intra-event and inter-event dependency-aware graph network for event
argument extraction,” inFindings of the Association for Computational
Linguistics: EMNLP 2023, 2023, pp. 6362–6372.
[299] J. Zhang, C. Yang, H. Zhu, Q. Lin, F. Xu, and J. Liu, “A semantic
mention graph augmented model for document-level event argument
extraction,”arXiv preprint arXiv:2403.09721, 2024.
[300] L. Zhuang, H. Fei, and P. Hu, “Syntax-based dynamic latent graph
for event relation extraction,”Information Processing & Management,
vol. 60, no. 5, p. 103469, 2023.
[301] A. Hao, J. Su, S. Sun, and T. Y . Sen, “Soft syntactic reinforcement for
neural event extraction,” inProceedings of the 2025 Conference of the
Nations of the Americas Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Papers),
2025, pp. 9466–9478.
[302] C. Yao and Y . Guo, “Cascadepaie: Reallocating relevance for event
roles and event text in event argument extraction,” inICASSP 2025-
2025 IEEE International Conference on Acoustics, Speech and Signal
Processing (ICASSP). IEEE, 2025, pp. 1–5.
[303] Z. Wang, X. Wang, and W. Hu, “Continual event extraction with
semantic confusion rectification,”arXiv preprint arXiv:2310.15470,
2023.
[304] Z. Zhang, P. Balsebre, S. Luo, Z. Hai, and J. Huang, “Structam: En-
hancing address matching through semantic understanding of structure-
aware information,” inProceedings of the 2024 Joint International
Conference on Computational Linguistics, Language Resources and
Evaluation (LREC-COLING 2024), 2024, pp. 15 350–15 361.
[305] E. Fane, M. N. Uddin, O. Ikumariegbe, D. Kashif, E. Blanco, and
S. Corman, “Bemeae: Moving beyond exact span match for event
argument extraction,” inProceedings of the 2025 Conference of the
Nations of the Americas Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Papers),
2025, pp. 5734–5749.
[306] R. Han, Y . Zhou, and N. Peng, “Domain knowledge empowered
structured neural net for end-to-end event temporal relation extraction,”
arXiv preprint arXiv:2009.07373, 2020.
[307] S. Shen, G. Qi, Z. Li, S. Bi, and L. Wang, “Hierarchical chinese legal
event extraction via pedal attention mechanism,” inProceedings of the
28th international conference on computational linguistics, 2020, pp.
100–113.
[308] S. Wang, M. Yu, S. Chang, L. Sun, and L. Huang, “Query and extract:
Refining event extraction as type-oriented binary decoding,”arXiv
preprint arXiv:2110.07476, 2021.
[309] X. Zhang, L. Zang, P. Cheng, Y . Wang, and S. Hu, “A knowledge/data
enhanced method for joint event and temporal relation extraction,”
inICASSP 2022-2022 IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP). IEEE, 2022, pp. 6362–6366.
[310] A. P. B. Veyseh, V . D. Lai, F. Dernoncourt, and T. H. Nguyen, “Event
extraction in video transcripts,” inProceedings of the 29th International
Conference on Computational Linguistics, 2022, pp. 7156–7165.
[311] Y . Huang and W. Jia, “Exploring sentence community for document-
level event extraction,” inFindings of the Association for Computa-
tional Linguistics: EMNLP 2021, 2021, pp. 340–351.
[312] L. Zhuang, H. Fei, and P. Hu, “Knowledge-enhanced event relation
extraction via event ontology prompt,”Information Fusion, vol. 100,
p. 101919, 2023.[313] H. Zhang, W. Yao, and D. Yu, “Efficient zero-shot event extraction
with context-definition alignment,”arXiv preprint arXiv:2211.05156,
2022.
[314] N. Popovic and M. F ¨arber, “Few-shot document-level relation extrac-
tion,”arXiv preprint arXiv:2205.02048, 2022.
[315] C. Han, J. Zhang, X. Li, G. Xu, W. Peng, and Z. Zeng, “Duee-fin:
A large-scale dataset for document-level event extraction,” inNatural
Language Processing and Chinese Computing: 11th CCF International
Conference, NLPCC 2022, Guilin, China, September 24–25, 2022,
Proceedings, Part I, 2022, p. 172–183.
[316] H. Deng, Y . Zhang, Y . Zhang, W. Ying, C. Yu, J. Gao, W. Wang,
X. Bai, N. Yang, J. Maet al., “2event: Benchmarking open event
extraction with a large-scale chinese title dataset,”arXiv preprint
arXiv:2211.00869, 2022.
[317] C. Siagian and A. Shabbeer, “Entity and event topic extraction from
podcast episode title and description using entity linking,” inCompan-
ion Proceedings of the ACM Web Conference 2023, 2023, pp. 768–772.
[318] B. Zhang, L. Li, Q. Yang, and D. Feng, “Automatic large-scale data
generation for open-topic biomedical event relation extraction,” in2023
IEEE International Conference on Bioinformatics and Biomedicine
(BIBM). IEEE, 2023, pp. 4990–4992.
[319] D. Kim, R. Yoo, S. Yang, H. Yang, and J. Choo, “Aniee: A dataset of
animal experimental literature for event extraction,” inFindings of the
Association for Computational Linguistics: EMNLP 2023, 2023, pp.
12 959–12 971.
[320] S. Yan and K.-C. Wong, “Context awareness and embedding for
biomedical event extraction,”Bioinformatics, vol. 36, no. 2, pp. 637–
643, 2020.
[321] F. Su, C. Teng, F. Li, B. Li, J. Zhou, and D. Ji, “Generative biomed-
ical event extraction with constrained decoding strategy,”IEEE/ACM
Transactions on Computational Biology and Bioinformatics, 2024.
[322] C. Nguyen, H. Man, and T. Nguyen, “Contextualized soft prompts
for extraction of event arguments,” inFindings of the Association for
Computational Linguistics: ACL 2023, 2023, pp. 4352–4361.
[323] H. Ma, Q. Tang, N. Zhang, R. Xu, Y . Shao, W. Yan, and Y . Wang,
“Spteae: A soft prompt transfer model for zero-shot cross-lingual
event argument extraction,” inICASSP 2023-2023 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP).
IEEE, 2023, pp. 1–5.
[324] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena,
Y . Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning
with a unified text-to-text transformer,”Journal of machine learning
research, vol. 21, no. 140, pp. 1–67, 2020.
[325] K. Simonyan and A. Zisserman, “Very deep convolutional networks for
large-scale image recognition,”arXiv preprint arXiv:1409.1556, 2014.
[326] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” inProceedings of the IEEE conference on computer vision
and pattern recognition, 2016, pp. 770–778.
[327] M. Yatskar, V . Ordonez, L. Zettlemoyer, and A. Farhadi, “Commonly
uncommon: Semantic sparsity in situation recognition,” inProceedings
of the IEEE Conference on Computer Vision and Pattern Recognition,
2017, pp. 7196–7205.
[328] A. Mallya and S. Lazebnik, “Recurrent models for situation recogni-
tion,” inProceedings of the IEEE international conference on computer
vision, 2017, pp. 455–463.
[329] R. Li, M. Tapaswi, R. Liao, J. Jia, R. Urtasun, and S. Fidler, “Situation
recognition with graph neural networks,” inProceedings of the IEEE
international conference on computer vision, 2017, pp. 4173–4182.
[330] M. Suhail and L. Sigal, “Mixture-kernel graph attention network for
situation recognition,” inProceedings of the IEEE/CVF international
conference on computer vision, 2019, pp. 10 363–10 372.
[331] T. Cooray, N.-M. Cheung, and W. Lu, “Attention-based context aware
reasoning for situation recognition,” inProceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2020, pp.
4736–4745.
[332] S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: Towards real-time
object detection with region proposal networks,”IEEE transactions on
pattern analysis and machine intelligence, vol. 39, no. 6, pp. 1137–
1149, 2016.
[333] J. Cho, Y . Yoon, H. Lee, and S. Kwak, “Grounded situation recognition
with transformers,”arXiv preprint arXiv:2111.10135, 2021.
[334] J. Cho, Y . Yoon, and S. Kwak, “Collaborative transformers for
grounded situation recognition,” inProceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2022, pp.
19 659–19 668.
[335] W. Yu, H. Wang, G. Li, N. Xiao, and B. Ghanem, “Knowledge-
aware global reasoning for situation recognition,”IEEE Transactions

LIet al.: EVENT EXTRACTION 35
on Pattern Analysis and Machine Intelligence, vol. 45, no. 7, pp. 8621–
8633, 2023.
[336] C. Wang, J. Yang, and Q. He, “Event extraction for visual: De-
biasing with causality-guided attention mechanism,”Neurocomputing,
p. 130783, 2025.
[337] K. Gao, L. Chen, Y . Huang, and J. Xiao, “Video relation detection
via tracklet based visual transformer,” inProceedings of the 29th ACM
international conference on multimedia, 2021, pp. 4833–4837.
[338] Y . Zhao, H. Fei, Y . Cao, B. Li, M. Zhang, J. Wei, M. Zhang, and T.-
S. Chua, “Constructing holistic spatio-temporal scene graph for video
semantic role labeling,” inProceedings of the 31st ACM international
conference on multimedia, 2023, pp. 5281–5291.
[339] Y . Liu, F. Liu, L. Jiao, Q. Bao, L. Sun, S. Li, L. Li, and X. Liu, “Multi-
grained gradual inference model for multimedia event extraction,”IEEE
Transactions on Circuits and Systems for Video Technology, vol. 34,
no. 10, pp. 10 507–10 520, 2024.
[340] C. Sugandhika, C. Li, D. Rajan, and B. Fernando, “Situational scene
graph for structured human-centric situation understanding,” in2025
IEEE/CVF Winter Conference on Applications of Computer Vision
(WACV). IEEE, 2025, pp. 9215–9225.
[341] Y . Chen, X. Wang, M. Li, D. Hoiem, and H. Ji, “Vistruct: Visual
structural knowledge extraction via curriculum guided code-vision
representation,”arXiv preprint arXiv:2311.13258, 2023.
[342] J. Liu, B. Li, X. Yang, N. Yang, H. Fei, M. Zhang, F. Li, and D. Ji,
“M3d: A multimodal, multilingual and multitask dataset for grounded
document-level information extraction,”IEEE Transactions on Pattern
Analysis and Machine Intelligence, 2025.
[343] X. Zhang, Z. Wang, and P. Li, “Multimodal chinese event extraction
on text and audio,” in2023 International Joint Conference on Neural
Networks (IJCNN). IEEE, 2023, pp. 1–8.
[344] C. J. Fillmore, C. R. Johnson, and M. R. Petruck, “Background to
framenet,”International journal of lexicography, vol. 16, no. 3, pp.
235–250, 2003.
[345] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma,
Z. Huang, A. Karpathy, A. Khosla, M. Bernsteinet al., “Imagenet large
scale visual recognition challenge,”International journal of computer
vision, vol. 115, no. 3, pp. 211–252, 2015.
[346] M. Vilain, J. Burger, J. Aberdeen, D. Connolly, and L. Hirschman,
“A model-theoretic coreference scoring scheme,” inSixth Message
Understanding Conference (MUC-6): Proceedings of a Conference
Held in Columbia, Maryland, November 6-8, 1995, 1995.
[347] A. Bagga and B. Baldwin, “Entity-based cross-document coreferencing
using the vector space model,” inCOLING 1998 Volume 1: The 17th
international conference on computational linguistics, 1998.
[348] X. Luo, “On coreference resolution performance metrics,” inProceed-
ings of human language technology conference and conference on
empirical methods in natural language processing, 2005, pp. 25–32.
[349] M. Recasens and E. Hovy, “Blanc: Implementing the rand index for
coreference evaluation,”Natural language engineering, vol. 17, no. 4,
pp. 485–510, 2011.
[350] Y .-F. Lu, X.-L. Mao, T. Lan, H. Huang, C. Xu, and X. Gao, “Be-
yond exact match: Semantically reassessing event extraction by large
language models,”arXiv preprint arXiv:2410.09418, 2024.
[351] H. Gui, S. Qiao, J. Zhang, H. Ye, M. Sun, L. Liang, J. Z. Pan,
H. Chen, and N. Zhang, “Instructie: A bilingual instruction-based infor-
mation extraction dataset,” inInternational Semantic Web Conference.
Springer, 2024, pp. 59–79.
[352] M. Tong, B. Xu, S. Wang, M. Han, Y . Cao, J. Zhu, S. Chen, L. Hou, and
J. Li, “Docee: a large-scale and fine-grained benchmark for document-
level event extraction,” inProceedings of NAACL, 2022, p. 3970–3982.
[353] T. Parekh, J. Kwan, J. Yu, S. Johri, H. Ahn, S. Muppalla, K.-W. Chang,
W. Wang, and N. Peng, “Speed++: A multilingual event extraction
framework for epidemic prediction and preparedness,”arXiv preprint
arXiv:2410.18393, 2024.
[354] G. R. Doddington, A. Mitchell, M. A. Przybocki, L. A. Ramshaw, S. M.
Strassel, R. M. Weischedelet al., “The automatic content extraction
(ace) program-tasks, data, and evaluation.” inLrec, vol. 2, no. 1.
Lisbon, 2004, pp. 837–840.
[355] H. Ji and R. Grishman, “Knowledge base population: Successful
approaches and challenges,” inProceedings of the 49th annual meeting
of the association for computational linguistics: Human language
technologies, 2011, pp. 1148–1158.
[356] M. Surdeanu and H. Ji, “Overview of the english slot filling track at the
tac2014 knowledge base population evaluation,” inProc. Text Analysis
Conference (TAC2014), 2014.
[357] X. Wang, Y . Chen, N. Ding, H. Peng, Z. Wang, Y . Lin, X. Han, L. Hou,
J. Li, Z. Liuet al., “Maven-ere: A unified large-scale dataset for eventcoreference, temporal, causal, and subevent relation extraction,”arXiv
preprint arXiv:2211.07342, 2022.
[358] P. Stenetorp, S. Pyysalo, G. Topi ´c, T. Ohta, S. Ananiadou, and
J. Tsujii, “Brat: a web-based tool for nlp-assisted text annotation,”
inProceedings of the Demonstrations at the 13th Conference of the
European Chapter of the Association for Computational Linguistics,
2012, pp. 102–107.
[359] S. Zheng, W. Cao, W. Xu, and J. Bian, “Revisiting the evaluation
of end-to-end event extraction,” inFindings of the Association for
Computational Linguistics: ACL-IJCNLP 2021, 2021, pp. 4609–4617.
[360] J. Xu, J. Gallifant, A. E. Johnson, and M. McDermott, “Aces: Au-
tomatic cohort extraction system for event-stream datasets,”arXiv
preprint arXiv:2406.19653, 2024.
[361] Y . He, J. Hu, and B. Tang, “Revisiting event argument extraction: Can
eae models learn better when being aware of event co-occurrences?”
arXiv preprint arXiv:2306.00502, 2023.
[362] T.-N. Nguyen, B. T. Tran, T.-N. Luu, T. H. Nguyen, and K.-H. Nguyen,
“Bkee: Pioneering event extraction in the vietnamese language,” in
Proceedings of the 2024 Joint International Conference on Com-
putational Linguistics, Language Resources and Evaluation (LREC-
COLING 2024), 2024, pp. 2421–2427.
[363] Y . Wei, K. Shuang, Z. Li, and C. Mao, “How do llms’ preferences
affect event argument extraction? cat: Addressing preference traps in
unsupervised eae,” inFindings of the Association for Computational
Linguistics: ACL 2025, 2025, pp. 19 529–19 543.
[364] A. P. B. Veyseh, J. Ebrahimi, F. Dernoncourt, and T. H. Nguyen,
“Mee: A novel multilingual event extraction dataset,”arXiv preprint
arXiv:2211.05955, 2022.
[365] K.-H. Huang, I. Hsu, P. Natarajan, K.-W. Chang, N. Penget al.,
“Multilingual generative language models for zero-shot cross-lingual
event argument extraction,”arXiv preprint arXiv:2203.08308, 2022.
[366] C. Jenkins, S. Agarwal, J. Barry, S. Fincke, and E. Boschee, “Mas-
sively multi-lingual event understanding: Extraction, visualization, and
search,”arXiv preprint arXiv:2305.10561, 2023.
[367] N. Borenstein, N. d. S. Perez, and I. Augenstein, “Multilingual
event extraction from historical newspaper adverts,”arXiv preprint
arXiv:2305.10928, 2023.
[368] Q. Lyu, H. Zhang, E. Sulem, and D. Roth, “Zero-shot event extraction
via transfer learning: Challenges and insights,” inProceedings of the
59th Annual Meeting of the Association for Computational Linguistics
and the 11th International Joint Conference on Natural Language
Processing (Volume 2: Short Papers), 2021, pp. 322–332.
[369] M. Van Nguyen, T. N. Nguyen, B. Min, and T. H. Nguyen, “Crosslin-
gual transfer learning for relation and event extraction via word
category and class alignments,” inProceedings of the 2021 conference
on empirical methods in natural language processing, 2021, pp. 5414–
5426.
[370] J. Zhou, Q. Zhang, Q. Chen, L. He, and X. Huang, “A multi-format
transfer learning model for event argument extraction via variational
information bottleneck,”arXiv preprint arXiv:2208.13017, 2022.
[371] S. Fincke, S. Agarwal, S. Miller, and E. Boschee, “Language model
priming for cross-lingual event extraction,” inProceedings of the AAAI
Conference on Artificial Intelligence, vol. 36, no. 10, 2022, pp. 10 627–
10 635.
[372] P. Cao, Z. Jin, Y . Chen, K. Liu, and J. Zhao, “Zero-shot cross-lingual
event argument extraction with language-oriented prefix-tuning,” in
Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37,
no. 11, 2023, pp. 12 589–12 597.
[373] X. Yi, X. Zhu, and P. Li, “Enhancing zero-shot cross-lingual event ar-
gument extraction with language-independent information,” inICASSP
2025-2025 IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP). IEEE, 2025, pp. 1–5.
[374] E. Cai and B. O’Connor, “Evaluating zero-shot event structures:
Recommendations for automatic content extraction (ace) annotations,”
inProceedings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 2: Short Papers), 2023, pp. 1651–
1665.
[375] Z. Chen and H. Ji, “Language specific issue and feature exploration
in chinese event extraction,” inProceedings of Human Language
Technologies: The 2009 Annual Conference of the North American
Chapter of the Association for Computational Linguistics, Companion
Volume: Short Papers, 2009, pp. 209–212.
[376] J. Li, A. Ritter, C. Cardie, and E. Hovy, “Major life event extraction
from twitter based on congratulations/condolences speech acts,” in
Proceedings of the 2014 conference on empirical methods in natural
language processing (EMNLP), 2014, pp. 1997–2007.

36 ARXIV PREPRINT
[377] S. Riedel and A. McCallum, “Fast and robust joint models for
biomedical event extraction,” inProceedings of the 2011 Conference on
Empirical Methods in Natural Language Processing, 2011, pp. 1–12.
[378] A. Ritter, E. Wright, W. Casey, and T. Mitchell, “Weakly supervised
extraction of computer security events from twitter,” inProceedings
of the 24th international conference on world wide web, 2015, pp.
896–905.
[379] D. Zhou, L. Chen, and Y . He, “An unsupervised framework of
exploring events on twitter: Filtering, extraction and categorization,” in
Proceedings of the AAAI conference on artificial intelligence, vol. 29,
no. 1, 2015.
[380] Y . Chen, S. Liu, X. Zhang, K. Liu, and J. Zhao, “Automatically labeled
data generation for large scale event extraction,” inProceedings of the
55th Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), 2017, pp. 409–419.
[381] J. Peng, W. Yang, F. Wei, and L. He, “Prompt for extraction: Mul-
tiple templates choice model for event extraction,”Knowledge-based
systems, vol. 289, p. 111544, 2024.
[382] B. Tsolmon and K.-S. Lee, “An event extraction model based on
timeline and user analysis in latent dirichlet allocation,” inProceedings
of the 37th international ACM SIGIR conference on Research &
development in information retrieval, 2014, pp. 1187–1190.
[383] A. Intxaurrondo, E. Agirre, O. L. De Lacalle, and M. Surdeanu,
“Diamonds in the rough: Event extraction from imperfect microblog
data,” inProceedings of the 2015 Conference of the North American
Chapter of the Association for Computational Linguistics: Human
Language Technologies, 2015, pp. 641–650.
[384] H. Li and H. Ji, “Cross-genre event extraction with knowledge enrich-
ment,” inProceedings of the 2016 Conference of the North American
Chapter of the Association for Computational Linguistics: Human
Language Technologies, 2016, pp. 1158–1162.
[385] P. Jin, L. Mu, L. Zheng, J. Zhao, and L. Yue, “News feature extraction
for events on social network platforms,” inProceedings of the 26th
International Conference on World Wide Web Companion, 2017, pp.
69–78.
[386] X. Liu, H.-Y . Huang, and Y . Zhang, “Open domain event extraction
using neural latent variable models,” inProceedings of the 57th Annual
Meeting of the Association for Computational Linguistics, 2019, pp.
2860–2871.
[387] J. Eisenberg and M. Sheriff, “Automatic extraction of personal events
from dialogue,” inProceedings of the First Joint Workshop on Narra-
tive Understanding, Storylines, and Events, 2020, pp. 63–71.
[388] T. Mitamura, Z. Liu, and E. H. Hovy, “Overview of tac kbp 2015 event
nugget track,”Theory and Applications of Categories, 2015. [Online].
Available: https://api.semanticscholar.org/CorpusID:43923561
[389] Z. Du, Y . Li, X. Guo, Y . Sun, and B. A. Li, “Training multimedia event
extraction with generated images and captions,”Proceedings of the
31st ACM International Conference on Multimedia, 2023. [Online].
Available: https://api.semanticscholar.org/CorpusID:259165371
[390] F. Moghimifar, F. Shiri, V . Nguyen, Y .-F. Li, and G. Haffari, “Theia:
Weakly supervised multimodal event extraction from incomplete data,”
inInternational Joint Conference on Natural Language Processing,
2023. [Online]. Available: https://api.semanticscholar.org/CorpusID:
267406869
[391] J. Li, C. Zhang, M. Du, D. Min, Y . Chen, and G. Qi, “Three stream
based multi-level event contrastive learning for text-video event extrac-
tion,” inProceedings of the 2023 Conference on Empirical Methods in
Natural Language Processing, 2023, pp. 1666–1676.
[392] S. Srivastava, S. Pati, and Z. Yao, “Instruction-tuning llms for event
extraction with annotation guidelines,”ArXiv, vol. abs/2502.16377,
2025. [Online]. Available: https://api.semanticscholar.org/CorpusID:
276575342
[393] S. Li, Q. Zhan, K. Conger, M. Palmer, H. Ji, and J. Han, “GLEN:
general-purpose event detection for thousands of types,” inProceedings
of EMNLP, 2023, pp. 2823–2838.
[394] R. Bossy, W. Golik, Z. Ratkovi ´c, P. Bessi `eres, and C. N ´edellec, “Bionlp
shared task 2013 – an overview of the bacteria biotope task,” in
BioNLP@ACL, 2013. [Online]. Available: https://api.semanticscholar.
org/CorpusID:826766
[395] L. Del ´eger, R. Bossy, E. Chaix, M. Ba, A. Ferr ´e, P. Bessi `eres, and
C. N ´edellec, “Overview of the bacteria biotope task at bionlp shared
task 2016,” inWorkshop on Biomedical Natural Language Processing,
2016. [Online]. Available: https://api.semanticscholar.org/CorpusID:
17909922
[396] W. Zhao, Y . Zhao, X. Jiang, T. He, F. Liu, and N. Li, “A novel method
for multiple biomedical events extraction with reinforcement learning
and knowledge bases,”2020 IEEE International Conference onBioinformatics and Biomedicine (BIBM), pp. 402–407, 2020. [Online].
Available: https://api.semanticscholar.org/CorpusID:231618759
[397] Y . Zhao, W. Zhao, X. Jiang, T. He, and B. Su, “An improved rl-based
framework for multiple biomedical event extraction via self-supervised
learning,”2021 IEEE International Conference on Bioinformatics
and Biomedicine (BIBM), pp. 619–624, 2021. [Online]. Available:
https://api.semanticscholar.org/CorpusID:245934329
[398] X. He, P. Tai, H. Lu, X. Huang, and Y . Ren, “A biomedical
event extraction method based on fine-grained and attention
mechanism,”BMC Bioinformatics, vol. 23, 2022. [Online]. Available:
https://api.semanticscholar.org/CorpusID:251163053
[399] L. Wang, H. Cao, L. Yuan, X. Guo, and Y . Cui, “Child-sum
eatree-lstms: enhanced attentive child-sum tree-lstms for biomedical
event extraction,”BMC Bioinformatics, vol. 24, 2023. [Online].
Available: https://api.semanticscholar.org/CorpusID:259156673
[400] B. Zhang, L. Li, D. Song, and Y . Zhao, “Biomedical event causal
relation extraction based on a knowledge-guided hierarchical graph
network,”Soft Computing, vol. 27, pp. 17 369–17 386, 2023. [Online].
Available: https://api.semanticscholar.org/CorpusID:260213698
[401] J.-H. Hu, B. Tang, N. Lyu, Y . He, and Y . Xiong, “Cmbee: A constraint-
based multi-task learning framework for biomedical event extraction,”
Journal of biomedical informatics, p. 104599, 2024. [Online].
Available: https://api.semanticscholar.org/CorpusID:267240635
[402] L. Gao, Z. Yang, J. Ning, K. Ma, L. Wang, W. Liu, Y . Zhang,
L. Luo, B. Xu, J. Wang, Y . Yang, Z. Zhao, Y . Sun, and H. Lin,
“Biomedical event extraction as semantic segmentation,”2024 IEEE
International Conference on Bioinformatics and Biomedicine (BIBM),
pp. 3218–3221, 2024. [Online]. Available: https://api.semanticscholar.
org/CorpusID:275439056
[403] I. Guellil, S. Andres, A. Anand, B. Guthrie, H. Zhang, A. Hasan,
H. Wu, and B. Alex, “Adverse event extraction from discharge
summaries: A new dataset, annotation scheme, and initial findings,”
ArXiv, vol. abs/2506.14900, 2025. [Online]. Available: https://api.
semanticscholar.org/CorpusID:279447402
[404] J. Xu, J. Gallifant, A. E. W. Johnson, and M. B. A. McDermott,
“Aces: Automatic cohort extraction system for event-stream datasets,”
ArXiv, vol. abs/2406.19653, 2024. [Online]. Available: https://api.
semanticscholar.org/CorpusID:270845890
[405] S. Yadav, P. Ramteke, A. Ekbal, S. Saha, and P. Bhattacharyya,
“Exploring disorder-aware attention for clinical event extraction,”
ACM Transactions on Multimedia Computing, Communications, and
Applications (TOMM), vol. 16, pp. 1 – 21, 2020. [Online]. Available:
https://api.semanticscholar.org/CorpusID:218489442
[406] Q.-Z. Wan, C. Wan, K. Xiao, R. Hu, D. Liu, and X. Liu,
“Cfere: Multi-type chinese financial event relation extraction,”
Inf. Sci., vol. 630, pp. 119–134, 2023. [Online]. Available:
https://api.semanticscholar.org/CorpusID:256661205
[407] P. Chen, K. Liu, Y . Chen, T. Wang, and J. Zhao, “Probing into
the root: A dataset for reason extraction of structural events from
financial documents,” inConference of the European Chapter of the
Association for Computational Linguistics, 2021. [Online]. Available:
https://api.semanticscholar.org/CorpusID:233189529
[408] D. Mariko, H. A. Akl, E. Labidurie, S. Durfort, H. de Mazancourt,
and M. El-Haj, “The financial document causality detection shared
task (fincausal 2020),”ArXiv, vol. abs/2012.02505, 2020. [Online].
Available: https://api.semanticscholar.org/CorpusID:227231694
[409] L. Yue, Q. Liu, L. Zhao, L. Wang, W. Gao, and Y . An, “Event grounded
criminal court view generation with cooperative (large) language
models,”Proceedings of the 47th International ACM SIGIR Conference
on Research and Development in Information Retrieval, 2024. [Online].
Available: https://api.semanticscholar.org/CorpusID:269033454
[410] N. Lagos, F. Segond, S. Castellani, and J. O’Neill, “Event extraction
for legal case building and reasoning,” inIFIP International
Conference on Intelligent Information Processing, 2010. [Online].
Available: https://api.semanticscholar.org/CorpusID:16813422
[411] K. Le-Duc, Q.-A. Dang, T.-H. Pham, and T.-S. Hy, “wav2graph: A
framework for supervised learning knowledge graph from speech,”
arXiv preprint arXiv:2408.04174, 2024.
[412] J. Kang, T. Wu, J. Zhao, G. Wang, Y . Wei, H. Yang, G. Qi, Y .-F. Li,
and G. Haffari, “Double mixture: Towards continual event detection
from speech,”arXiv preprint arXiv:2404.13289, 2024.
[413] M. Gedeon, “Retrieval-enhanced few-shot prompting for speech event
extraction,”arXiv preprint arXiv:2504.21372, 2025.
[414] Y . Chiba and R. Higashinaka, “Dialogue situation recognition in
everyday conversation from audio, visual, and linguistic information,”
IEEE Access, vol. 11, pp. 70 819–70 832, 2023.

LIet al.: EVENT EXTRACTION 37
[415] J. Kang, T. Wu, J. Zhao, G. Wang, G. Qi, Y .-F. Li, and G. Haffari,
“Towards event extraction from speech with contextual clues,”arXiv
preprint arXiv:2401.15385, 2024.
[416] C. Wang, D. Yogatama, A. Coates, T. Han, A. Hannun, and B. Xiao,
“Lookahead convolution layer for unidirectional recurrent neural net-
works,” 2016.
[417] D. Amodei, S. Ananthanarayanan, R. Anubhai, J. Bai, E. Battenberg,
C. Case, J. Casper, B. Catanzaro, Q. Cheng, G. Chenet al., “Deep
speech 2: End-to-end speech recognition in english and mandarin,”
inInternational conference on machine learning. PMLR, 2016, pp.
173–182.
[418] Z. Zhang and H. Ji, “Abstract meaning representation guided graph
encoding and decoding for joint information extraction,” inProc. The
2021 Conference of the North American Chapter of the Association for
Computational Linguistics-Human Language Technologies (NAACL-
HLT2021), 2021.
[419] M. Nikolaus, E. Salin, S. Ayache, A. Fourtassi, and B. Favre,
“Do vision-and-language transformers learn grounded predicate-
noun dependencies?” inProceedings of the 2022 Conference
on Empirical Methods in Natural Language Processing, EMNLP
2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022,
Y . Goldberg, Z. Kozareva, and Y . Zhang, Eds. Association for
Computational Linguistics, 2022, pp. 1538–1555. [Online]. Available:
https://doi.org/10.18653/v1/2022.emnlp-main.100
[420] K. Kanagaraj and G. L. Priya, “Curvelet transform based feature
extraction and selection for multimedia event classification,”Journal
of King Saud University-Computer and Information Sciences, vol. 34,
no. 2, pp. 375–383, 2022.
[421] X. Bao, J. Gu, Z. Wang, M. Qiang, and C.-R. Huang, “Employing
glyphic information for chinese event extraction with vision-language
model,” inFindings of the Association for Computational Linguistics:
EMNLP 2024, 2024, pp. 1068–1080.
[422] H. Bansal, P.-N. Kung, P. J. Brantingham, K.-W. Chang, and N. Peng,
“Genearl: A training-free generative framework for multimodal event
argument role labeling,”arXiv preprint arXiv:2404.04763, 2024.
[423] Y . Cui, B. Sun, T. Jiang, and H. Cui, “Multimedia event extraction
based on multimodal low-dimensional feature representation space,”
Signal, Image and Video Processing, vol. 19, no. 5, p. 397, 2025.
[424] K. Wei, R. Du, L. Jin, J. Liu, J. Yin, L. Zhang, J. Liu, N. Liu, J. Zhang,
and Z. Guo, “Video event extraction with multi-view interaction
knowledge distillation,” inProceedings of the AAAI Conference on
Artificial Intelligence, vol. 38, no. 17, 2024, pp. 19 224–19 233.
[425] T.-T. Chu, A.-Z. Yen, W.-H. Ang, H.-H. Huang, and H.-H. Chen,
“Vidlife: A dataset for life event extraction from videos,” inPro-
ceedings of the 30th ACM International Conference on Information
& Knowledge Management, 2021, pp. 4436–4444.
[426] T. Zhang, S. Whitehead, H. Zhang, H. Li, J. Ellis, L. Huang, W. Liu,
H. Ji, and S.-F. Chang, “Improving event extraction via multimodal
integration,” inProceedings of the 25th ACM international conference
on Multimedia, 2017, pp. 270–278.
[427] M. Tong, S. Wang, Y . Cao, B. Xu, J. Li, L. Hou, and T.-S. Chua,
“Image enhanced event detection in news articles,” inProceedings of
the AAAI Conference on Artificial Intelligence, vol. 34, no. 05, 2020,
pp. 9040–9047.
[428] R. Huang and E. Riloff, “Modeling textual cohesion for event extrac-
tion,” inProceedings of the AAAI conference on artificial intelligence,
vol. 26, no. 1, 2012, pp. 1664–1670.
[429] K. Sanders, R. Kriz, A. Liu, and B. Van Durme, “Ambiguous images
with human judgments for robust visual event classification,”Advances
in Neural Information Processing Systems, vol. 35, pp. 2637–2650,
2022.
[430] X. Tan, G. Pergola, and Y . He, “Event temporal relation extraction with
bayesian translational model,”arXiv preprint arXiv:2302.04985, 2023.
[431] J.-D. Kim, Y . Wang, and Y . Yasunori, “The Genia event extraction
shared task, 2013 edition - overview,” inProceedings of the BioNLP
Shared Task 2013 Workshop, 2013, pp. 8–15.
[432] T. Parekh, A. Mac, J. Yu, Y . Dong, S. Shahriar, B. Liu, E. Yang, K.-
H. Huang, W. Wang, N. Peng, and K.-W. Chang, “Event detection
from social media for epidemic prediction,” inProceedings of the
2024 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies (Volume 1:
Long Papers), 2024, pp. 5758–5783.
[433] A. Pouran Ben Veyseh, J. Ebrahimi, F. Dernoncourt, and T. Nguyen,
“MEE: A novel multilingual event extraction dataset,” inProceedings
of the 2022 Conference on Empirical Methods in Natural Language
Processing, 2022, pp. 9603–9613.[434] M. Zubillaga, O. Sainz, A. Estarrona, O. Lopez de Lacalle, and
E. Agirre, “Event extraction in Basque: Typologically motivated cross-
lingual transfer-learning analysis,” inProceedings of the 2024 Joint
International Conference on Computational Linguistics, Language Re-
sources and Evaluation (LREC-COLING 2024), 2024, pp. 6607–6621.
[435] T.-N. Nguyen, B. T. Tran, T.-N. Luu, T. H. Nguyen, and K.-H. Nguyen,
“BKEE: Pioneering event extraction in the Vietnamese language,”
inProceedings of the 2024 Joint International Conference on Com-
putational Linguistics, Language Resources and Evaluation (LREC-
COLING 2024), 2024, pp. 2421–2427.
[436] B. M. Sundheim, “Overview of the third Message Understanding Eval-
uation and Conference,” inThird Message Understanding Conference
(MUC-3): Proceedings of a Conference Held in San Diego, California,
May 21-23, 1991, 1991.
[437] I. Soboroff, “The better cross-language datasets,” inProceedings of
the 46th International ACM SIGIR Conference on Research and
Development in Information Retrieval, 2023, p. 3047–3053.
[438] Y . Ren, Y . Cao, H. Li, Y . Li, Z. Z. Ma, F. Fang, P. Guo, and W. Ma,
“DEIE: Benchmarking document-level event information extraction
with a large-scale Chinese news dataset,” inProceedings of the
2024 Joint International Conference on Computational Linguistics,
Language Resources and Evaluation (LREC-COLING 2024), 2024, pp.
4592–4604.
[439] M. Zhu, Z. Xu, K. Zeng, K. Xiao, M. Wang, W. Ke, and H. Huang,
“CMNEE:a large-scale document-level event extraction dataset based
on open-source Chinese military news,” inProceedings of the 2024
Joint International Conference on Computational Linguistics, Lan-
guage Resources and Evaluation (LREC-COLING 2024), 2024, pp.
3367–3379.
[440] T. Mikolov, K. Chen, G. Corrado, and J. Dean, “Efficient estimation of
word representations in vector space,”arXiv preprint arXiv:1301.3781,
2013.
[441] J. Pennington, R. Socher, and C. D. Manning, “Glove: Global vectors
for word representation,” inProceedings of the 2014 conference on
empirical methods in natural language processing (EMNLP), 2014,
pp. 1532–1543.
[442] C. Mih ˘ail˘a, T. Ohta, S. Pyysalo, and S. Ananiadou, “Biocause: An-
notating and analysing causality in the biomedical domain,”BMC
bioinformatics, vol. 14, no. 1, p. 2, 2013.
[443] G. Wang, D. Liu, J.-Y . Nie, Q. Wan, R. Hu, X. Liu, W. Liu, and
J. Liu, “Degap: Dual event-guided adaptive prefixes for templated-
based event argument extraction with slot querying,”arXiv preprint
arXiv:2405.13325, 2024.
[444] S. Srivastava, G. Singh, S. Matsumoto, A. Raz, P. Costa, J. Poore, and
Z. Yao, “MailEx: Email event and argument extraction,” inProceedings
of the 2023 Conference on Empirical Methods in Natural Language
Processing, 2023, pp. 12 964–12 987.
[445] Y .-P. Chen, A.-Z. Yen, H.-H. Huang, H. Nakayama, and H.-H. Chen,
“LED: A dataset for life event extraction from dialogs,” inFindings of
the Association for Computational Linguistics: EACL 2023, 2023, pp.
384–398.
[446] G. Frisoni, G. Moro, and L. Balzani, “Text-to-text extraction and
verbalization of biomedical event graphs,” inProceedings of the 29th
International Conference on Computational Linguistics, 2022, pp.
2692–2710.
[447] N. Park, K. Lybarger, G. K. Ramachandran, S. Lewis, A. Damani,
¨O. Uzuner, M. Gunn, and M. Yetisgen, “A novel corpus of anno-
tated medical imaging reports and information extraction results using
BERT-based language models,” inProceedings of the 2024 Joint
International Conference on Computational Linguistics, Language Re-
sources and Evaluation (LREC-COLING 2024), 2024, pp. 1280–1292.
[448] S. Touileb, J. Murstad, P. Mæhlum, L. Steskal, L. C. Storset, H. You,
and L. Øvrelid, “EDEN: A dataset for event detection in Norwegian
news,” inProceedings of the 2024 Joint International Conference
on Computational Linguistics, Language Resources and Evaluation
(LREC-COLING 2024), 2024, pp. 5495–5506.
[449] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “Bleu: a method for
automatic evaluation of machine translation,” inProceedings of the
40th annual meeting of the Association for Computational Linguistics,
2002, pp. 311–318.
[450] C.-Y . Lin, “Rouge: A package for automatic evaluation of summaries,”
inText summarization branches out, 2004, pp. 74–81.
[451] R. Rei, C. Stewart, A. C. Farinha, and A. Lavie, “Comet: A neural
framework for mt evaluation,” inProceedings of the 2020 Conference
on Empirical Methods in Natural Language Processing (EMNLP),
2020, pp. 2685–2702.

38 ARXIV PREPRINT
[452] G. Doddington, “Automatic evaluation of machine translation quality
using n-gram co-occurrence statistics,” inProceedings of the second
international conference on Human Language Technology Research,
2002, pp. 138–145.
[453] T. Sellam, D. Das, and A. Parikh, “Bleurt: Learning robust metrics
for text generation,” inProceedings of the 58th Annual Meeting of the
Association for Computational Linguistics, 2020, pp. 7881–7892.
[454] T. Zhang*, V . Kishore*, F. Wu*, K. Q. Weinberger, and Y . Artzi,
“Bertscore: Evaluating text generation with bert,” inInternational
Conference on Learning Representations, 2020.
[455] OpenAI, “Gpt-4o system card,” inCoRR, vol. abs/2410.21276, 2024.
[456] Google, “Gemini: A family of highly capable multimodal models,” in
CoRR, vol. abs/2312.11805, 2023.
[457] H. Liu, C. Li, Q. Wu, and Y . J. Lee, “Visual instruction tuning,” in
NeurIPS, 2023.
[458] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen,
W. Peng, X. Feng, B. Qinet al., “A survey on hallucination in
large language models: Principles, taxonomy, challenges, and open
questions,”ACM Transactions on Information Systems, vol. 43, no. 2,
pp. 1–55, 2025.
[459] T. G. Clark, M. J. Bradburn, S. B. Love, and D. G. Altman, “Survival
analysis part i: basic concepts and first analyses,”British journal of
cancer, vol. 89, no. 2, pp. 232–238, 2003.
[460] A. Dwaraki, S. Kumary, and T. Wolf, “Automated event identification
from system logs using natural language processing,” inProceedings
of ICNC, 2020, pp. 209–215.
[461] V . Sundar, M. Dutson, A. Ardelean, C. Bruschini, E. Charbon, and
M. Gupta, “Generalized event cameras,” inProceedings of CVPR,
2024, pp. 25 007–25 017.
[462] H. Man, N. T. Ngo, L. N. Van, and T. H. Nguyen, “Selecting optimal
context sentences for event-event relation extraction,” inProceedings
of AAAI, 2022, pp. 11 058–11 066.
[463] E. Hwang, J.-Y . Lee, T. Yang, D. Patel, D. Zhang, and A. McCallum,
“Event-event relation extraction using probabilistic box embedding,” in
Proceedings of ACL, 2022, pp. 235–244.
[464] S. R. Ahmed, Z. E. Wang, G. A. Baker, K. Stowe, and J. H. Mar-
tin, “Generating harder cross-document event coreference resolution
datasets using metaphoric paraphrasing,” inProceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics
(Volume 2: Short Papers), 2024, pp. 276–286.
[465] J. Gong and Q. Hu, “Extracting military event temporal relations
via relative event time prediction and virtual adversarial training,” in
Findings of the Association for Computational Linguistics: NAACL
2025, 2025, pp. 3305–3317.
[466] A. Romanou, S. Montariol, D. Paul, L. Laugier, K. Aberer, and
A. Bosselut, “CRAB: Assessing the strength of causal relationships
between real-world events,” inProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing, 2023, pp. 15 198–
15 216.
[467] S. Alsayyahi and R. Batista-Navarro, “TIMELINE: Exhaustive annota-
tion of temporal relations supporting the automatic ordering of events
in news articles,” inProceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, 2023, pp. 16 336–16 348.
[468] H. Fei, M. Zhang, M. Zhang, and T.-S. Chua, “XNLP: An interactive
demonstration system for universal structured NLP,” inProceedings of
the ACL (System Demonstrations), 2024, pp. 19–30.
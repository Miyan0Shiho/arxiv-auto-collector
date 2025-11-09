# Systematizing LLM Persona Design: A Four-Quadrant Technical Taxonomy for AI Companion Applications

**Authors**: Esther Sun, Zichu Wu

**Published**: 2025-11-04 20:37:13

**PDF URL**: [http://arxiv.org/pdf/2511.02979v1](http://arxiv.org/pdf/2511.02979v1)

## Abstract
The design and application of LLM-based personas in AI companionship is a
rapidly expanding but fragmented field, spanning from virtual emotional
companions and game NPCs to embodied functional robots. This diversity in
objectives, modality, and technical stacks creates an urgent need for a unified
framework. To address this gap, this paper systematizes the field by proposing
a Four-Quadrant Technical Taxonomy for AI companion applications. The framework
is structured along two critical axes: Virtual vs. Embodied and Emotional
Companionship vs. Functional Augmentation. Quadrant I (Virtual Companionship)
explores virtual idols, romantic companions, and story characters, introducing
a four-layer technical framework to analyze their challenges in maintaining
long-term emotional consistency. Quadrant II (Functional Virtual Assistants)
analyzes AI applications in work, gaming, and mental health, highlighting the
shift from "feeling" to "thinking and acting" and pinpointing key technologies
like enterprise RAG and on-device inference. Quadrants III & IV (Embodied
Intelligence) shift from the virtual to the physical world, analyzing home
robots and vertical-domain assistants, revealing core challenges in symbol
grounding, data privacy, and ethical liability. This taxonomy provides not only
a systematic map for researchers and developers to navigate the complex persona
design space but also a basis for policymakers to identify and address the
unique risks inherent in different application scenarios.

## Full Text


<!-- PDF content starts -->

Systematizing LLM Persona Design: A Four-Quadrant
Technical Taxonomy for AI Companion Applications
Esther Sun
Carnegie Mellon University
ethers@andrew.cmu.eduZichu Wu
Carnegie Mellon University
zichuwu@andrew.cmu.edu
1
Abstract
The design and application of LLM-based personas in AI companionship is a
rapidly expanding but fragmented field, spanning from virtual emotional compan-
ions and game NPCs to embodied functional robots. This diversity in objectives,
modality, and technical stacks creates an urgent need for a unified framework. To
address this gap, this papersystematizesthe field by proposing aFour-Quadrant
Technical Taxonomyfor AI companion applications. The framework is structured
along two critical axes:Virtual vs. EmbodiedandEmotional Companionship
vs. Functional Augmentation.Quadrant I (Virtual Companionship)explores
virtual idols, romantic companions, and story characters, introducing a four-layer
technical framework to analyze their challenges in maintaining long-term emotional
consistency.Quadrant II (Functional Virtual Assistants)analyzes AI applica-
tions in work, gaming, and mental health, highlighting the shift from "feeling" to
"thinking and acting" and pinpointing key technologies like enterprise RAG and
on-device inference.Quadrants III & IV (Embodied Intelligence)shift from the
virtual to the physical world, analyzing home robots and vertical-domain assistants,
revealing core challenges in symbol grounding, data privacy, and ethical liability.
This taxonomy provides not only a systematic map for researchers and developers
to navigate the complex personadesign spacebut also a basis for policymakers to
identify and address the unique risks inherent in different application scenarios.
1 Introduction
Large Language Models (LLMs) are at a decisive inflection point. They are no longer mere text
generation tools but are increasingly becoming the core cognitive engines driving complex, personified
AI agents [ 115,87]. This rise of the "AI persona" is fueling a wide array of applications, from deeply
personal virtual companions [ 124] to specialized workplace "copilots" [ 22]. However, this rapid
expansion has led to conceptual fragmentation: a virtual lover designed for emotional attachment
(Quadrant I) [ 124], an enterprise assistant for workflow optimization (Quadrant II) [ 32], and a
physical robot assisting autistic children with training (Quadrant IV) [ 9] all use "persona," yet they
are fundamentally different in their technical foundations, interaction paradigms, core challenges,
and ethical risks.
Currently, academia and industry lack a unified framework to systematically analyze and compare
these diverse AI persona modalities. Existing research often remains siloed within a single vertical
(e.g., game NPCs [ 95] or chatbots [ 124]), overlooking cross-domain commonalities and differences.
1This is a preprint under review at the LLM Persona Workshop, NeurIPS 2025.
39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: LLM Persona Workshop
at NeurIPS 2025.arXiv:2511.02979v1  [cs.HC]  4 Nov 2025

(Virtual Idols / Digital Persons)
    (Virtual Boyfriend/ Girlfriend)
  (Virtual Story Characters)
 (Special Populations & Situational Companions)Emotional Companionship 
      (AI Mental Health &Functional Augmentation
 (AI Game Companion)
(B2B / Team Companions)
(Physical Companion 
 (AI Household & Daily    Emotional Support Companion)Virtual Embodied (X-axis: Interaction Intent)(Y-axis: Deployment Modality)IIIIIIIV Life Companion)    Robots / Devices)Figure 1: The four-quadrant taxonomy of LLM persona applications in AI companionship. This
framework structures the field along two primary axes:Deployment Modality (Virtual vs. Embod-
ied)andInteraction Intent (Emotional Companionship vs. Functional Augmentation). Quadrant
I covers virtual emotional companions; Quadrant II focuses on functional virtual assistants; Quadrants
III and IV extend these concepts into physically embodied intelligence.
To fill this gap, this papersystematizesthe field by proposing a comprehensive technical taxonomy
for LLM persona in AI companion applications. We introduce a four-quadrant framework structured
along two key axes:
1.Interaction Intent:Distinguishing systems primarily forEmotional Connection(Quadrant
I) from those forFunctional/Cognitive Augmentation(Quadrant II).
2.Deployment Modality:Distinguishing purelyVirtual Entities(Quadrants I & II) from
Embodied Intelligencethat acts in the physical world (Quadrants III & IV).
The structure of this paper follows this taxonomy:
•Section 2 (Quadrant I)analyzes "Virtual Companionship," focusing on the challenge of
achieving long-term emotional consistency and introducing a four-layer technical analysis
framework (Model, Architecture, Generation, Safety & Ethics).
•Section 3 (Quadrant II)explores "Functional Virtual Assistants" in work, gaming, and
mental health, analyzing their unique demands for efficiency, reliability, and high-stakes
scenarios.
•Section 4 (Quadrant III & IV)shifts the analysis from virtual to "Embodied Intelligence,"
examining LLM applications in physical robots and focusing on core barriers like symbol
grounding, privacy, and legal liability.
Through this framework, this paper aims to provide a clear roadmap for researchers, developers, and
policymakers to understand the personadesign space, technical frontiers, and strategic implications
of LLM persona, thereby fostering responsible innovation in the field.
2

Table 1: Comparison of three major forms of virtual companionship under a unified four-layer
technical framework. See detailed architectural and behavioral analyses inAppendix A (Virtual Story
Character Interaction),Appendix B (Virtual Romantic Companionship), andAppendix C (Virtual
Idols).
Technical Layer Common Ground Interactive Story Characters
(Appendix A)Virtual Romantic Compan-
ionship (Appendix B)Virtual Idols (Appendix C)
Model Layer Relies on LLMs for basic
cognition, dialogue, and per-
sona modeling.Maintain deep consistency,
prevent "character halluci-
nation"; use frameworks like
RoleLLM, DITTO for role-
specific fine-tuning and self-
correction.Overcome "persona drift,"
maintain long-term stabil-
ity; focus on emotional intel-
ligence (EQ) and personality
control (e.g., Persona Vectors,
XiaoIce).Unified singing and dialogue
identity; hybrid LLM (TTS)
and specialized Singing V oice
Synthesis (SVS) architecture
(e.g., VOCALOID: AI).
Architecture Layer Overcome LLM stateless-
ness and limited context;
rely on external memory
and state management sys-
tems.Generative Agents architec-
ture; achieves persistent mem-
ory and autonomous action via
a "perceive–reflect–plan" loop.Model dynamic 1:1 relation-
ships; hybrid IQ+EQ archi-
tecture, stateful relationship
graphs, and multi-tier memory
(RAG).Manage large-scale 1:N live
interaction; uses event-driven
(Pub/Sub) and tiered (paid) at-
tention funnels.
Generation Layer Pursues real-time, multi-
modal (voice, visual, behav-
ior) generation beyond text.Emergent social behaviors;
multi-agent "Perceive–Plan–
Act" loop.Emotionally immersive
dialogue; full-duplex (low-
latency, interruptible) speech
with emotion-synchronized
multimodal expression.High-fidelity real-time 3D
rendering; Motion Capture
(MoCap) + game engine (e.g.,
Unreal) rendering pipeline.
Safety & Ethics Layer Requires a mix of au-
tomated guardrails and
Human-in-the-Loop (HITL)
oversight for safety.Prevent harmful emergent
behaviors; balances autonomy
and narrative control via "Con-
stitutional AI" and "Director
AI."Manage "parasocial at-
tachment" risks; emotional
guardrails (e.g., anti-flattery,
AI chaperones) and resolving
business-ethics conflicts.Protect brand image and
character IP; hybrid content
moderation (auto + human)
and strictnakanohito(per-
former) consistency.
Core Frontier— Autonomy Frontier: Explor-
ing believable agent autonomy
and complex social emergence
under narrative constraints.Emotional Depth Frontier:
Exploring how to model and
sustain dynamic, believable,
long-term 1:1 emotional
bonds.Performance & Influence
Frontier: Exploring high-
fidelity, scalable (1:N) real-
time performance and maxi-
mizing brand IP value.
2 Quadrant I: Virtual Companionship
Contemporary virtual companionship primarily manifests in three forms: (1)Interactive Virtual
Story Characters, (2)Virtual Romantic Companionship, and (3)Virtual Idols. Although these
three forms differ in business models and interaction paradigms—corresponding respectively to (1)
creative interaction(1:1), (2)emotional attachment(1:1), and (3)fan economy(1:N)—they share a
common technological core challenge: how to construct and sustain a believable and consistent AI
persona over long-term interaction.
Specifically, (1) interactive story characters emphasize generatingemergent narrativesthrough
autonomous actions and social interactions within simulated environments; (2) virtual romantic
companionship focuses on modeling and tracking the evolving user–AI relationship state to enable
dynamic and empathetic emotional interactions; and (3) virtual idols center on performance and
brand formation, leveraging multimodal generation and large-scale real-time interaction technologies
to support a scalable cultural consumption experience.
2.1 Four-Layer Technical Analysis Framework
To ensure a systematic and in-depth examination, this section adopts afour-layer technical frame-
workthat delineates the structural and functional foundations underlying the three forms of virtual
companionship. TheModel Layerfocuses on the core AI models that endow virtual agents with
cognition, personality, and specific capabilities, emphasizing the customization and optimization of
large language models (LLMs). TheArchitecture Layeraddresses the macro-level system design
that supports these agents, including long-term memory mechanisms, state management, multimodal
integration, and data flow orchestration. TheGeneration Layerexamines the real-time synthesis
of behaviors and content—such as text, speech, animation, and environmental interactions—that
enable immersive and coherent user experiences. Finally, theSafety & Ethics Layerconsiders the
technical risks, user well-being concerns, and broader social implications that emerge during design,
deployment, and operation, as well as the mitigation strategies required to ensure responsible and
sustainable development. The subsequent discussion emphasizes the principal differences at the
model layer. Comprehensive descriptions of the underlying methods and future research directions
are deferred to the Appendix.
3

      (AI Mental Health & Quadrant II: Functional Virtual 
              Assistants 
(Virtual Idols / Digital           
Persons) 
  (Virtual Story 
Characters) 
    (Virtual Boyfriend/ 
Girlfriend) Quadrant I ： Virtual Emotional 
                Companionship 
Focus: 
achieving 
long-term 
emotional 
persona 
consistency 1. Model: Recognition & Persona 
2.    Architecture: Memory & Emotion 
3.    Generation: Interaction 
4.    Safety & Ethics 
 (AI Game Companion )
(B2B / Team Companions )Scenario 1 
Workplace Scenarios: 
Cognitive Copilots and 
Expert Agents 
Scenario 2 
Game Scenarios: 
The Dawn of Generative 
Narrative 
Scenario 3 
Mental Health 
Scenario: The 
High-Stakes Frontier 
Emotional Support )
(Physical Companion 
    Robots / Devices )
 (AI Household & Daily 
    Life Companion )1. Non-humanoid 
    Companions 
2. Functional      
    Assistants 
3. Humanoid 
    Assistants 
1.  Elderly Care 2. Special   
 Education 
 (Special Populations & Situational Companions) Quadrants III & IV: Persona in Embodied Intelligence Figure 2:Four-Quadrant Taxonomy of LLM Persona in AI Companion Applications.This
framework organizes the diverse landscape of personified AI along two critical axes:Interaction
Intent(Emotional Connection vs. Functional Augmentation) andDeployment Modality(Virtual
vs. Embodied).Quadrant I (Virtual Emotional Companionship)examines virtual romantic
companions, interactive story characters, and virtual idols, with focus on achieving long-term
emotional persona consistency through a four-layer technical framework (Model, Architecture,
Generation, Safety & Ethics).Quadrant II (Functional Virtual Assistants)analyzes AI applications
in three key scenarios: workplace cognitive copilots (enterprise RAG and process automation), game
companions (low-latency generative narrative), and mental health support (clinical safety protocols).
Quadrants III & IV (Embodied Intelligence)shift from virtual to physical deployment, covering
general home applications (non-humanoid companions, functional assistants, humanoid robots) and
specialized vertical domains (elderly care, special education), addressing core challenges in symbol
grounding, privacy, and legal liability. Each quadrant presents distinct technical requirements and
ethical considerations, as detailed in Sections 2–4.
2.1.1 Model Layer
Across all three virtual companion archetypes, the cognitive core converges onLarge Language
Models (LLMs). After customization and fine-tuning, LLMs serve as the cognitive nucleus, enabling
complex dialogue, reasoning, and persona modeling. Despite shared foundations, distinct interaction
paradigms impose different optimization demands.Interactive Virtual Story Charactersface
the challenge of maintaining deep persona consistency [ 47]: the model must adhere to predefined
background, knowledge, and linguistic style even under zero-shot conditions. The representative
RoleLLM[ 109] framework embeds detailed character constitutions into model parameters through a
process of character definition, contextual instruction generation, and role-conditioned instruction
tuning (RoCIT), yielding intrinsically persona-aligned behavior.Virtual Romantic Companionship
systems must mitigate long-term persona drift, preserving identity stability during sustained one-
on-one interactions. Microsoft’sXiaoIce[ 125] separates IQ and EQ through an empathy vector
mechanism that guides persona-consistent responses, while Anthropic’sPersona Vectors[ 10] map
4

interpretable trait directions in latent space, allowing real-time monitoring and adjustment.Virtual
Idolsfocus on high-fidelity vocal performance by adopting a hybrid modeling approach that decouples
linguistic and acoustic [ 55]: persona-conditioned LLMs manage dialogue and engagement[ 36], while
Singing V oice Synthesis (SVS) engines [ 21]—such as AI-based VOCALOID systems—generate
expressive singing with cross-lingual capabilities. The main challenge lies in maintaining timbre
coherence between TTS and SVS outputs to preserve a unified vocal identity. Overall, interactive
story characters emphasize persona fidelity, romantic companions prioritize longitudinal stability,
and virtual idols pursue multimodal voice coherence.
2.1.2 Architecture Layer
The architecture layer addresses the intrinsic memory limitation of LLMs, whose finite context win-
dows constrain long-term continuity. Each archetype extends memory and state management through
external architectures tailored to its interaction logic.Interactive Virtual Story Charactersrely on
theGenerative Agents[ 84] framework, which implements a “perceive–plan–act” retrieval-augmented
loop composed of a chronological memory stream, periodic reflection for abstraction, and relevance-
ranked planning retrieval. This closed process of Memory–Reflection–Planning enables autonomous
world modeling and emergent narrative generation.Virtual Romantic Companionshipsystems
maintain evolving relational states through user-centered, stateful RAG architectures integrating
structured relational memory with affective reasoning [ 113]. Historical dialogues, preferences, and
events are stored in a relational database, and the EQ module fuses current affective cues with past in-
teractions to produce personalized, empathetic responses.Virtual Idolshandle large-scale, real-time
audience interaction via an event-driven architecture (EDA) [ 103] built on a publish/subscribe model,
where distributed microservices manage chat aggregation, monetized comments, and moderation to
ensure scalability, low latency, and brand coherence. In summary, story characters employ RAG [ 59]
loops for autonomous reasoning, romantic companions use relational memory to sustain emotional
continuity, and virtual idols leverage event-driven pipelines for large-scale engagement.
2.1.3 Generation Layer
The generation layer governs how virtual companions produce multimodal, believable, and temporally
coherent outputs beyond text. While all three pursue real-time, immersive generation, their expressive
goals diverge.Interactive Virtual Story Charactersfocus on unscripted emergent behavior: multi-
agent simulation loops [ 98] allow each agent’s output to become another’s input, forming continuous
chains of generation, observation, and reaction that yield self-organizing social dynamics and narrative
coherence.Virtual Romantic Companionshipemphasizes multimodal emotional synchrony through
full-duplex spoken dialogue models [ 24] enabling low-latency, backchannel-rich interaction, and
emotional synthesis pipelines [ 79] aligning affective TTS with facial animation. Prosodic modulation
and synchronized micro-expressions together create a coherent emotional presence.Virtual Idols
aim for broadcast-grade 3D performance using real-time rendering and streaming pipelines [ 49] that
integrate motion capture, Unreal Engine rendering [ 58], and live broadcast software [ 5](e.g., OBS).
These systems optimize for both visual fidelity and latency to ensure professional-grade performance
and interactive responsiveness. In essence, story characters prioritize emergent multi-agent behavior,
romantic companions achieve affective coherence, and virtual idols combine motion, rendering, and
streaming for performative realism.
2.1.4 Safety & Ethics Layer
Prolonged and emotionally intensive AI interactions introduce significant ethical, psychological, and
social risks [ 73,16]. Each archetype must balance autonomy, empathy, and safety while upholding
transparency, interpretability, and non-harm principles.Interactive Virtual Story Characters
confront the tension between autonomy and safety; mitigation strategies includeConstitutional AI
[6], embedding explicit ethical constraints in planning loops, and sandbox stress testing to expose
emergent risks prior to deployment [ 89,104,118].Virtual Romantic Companionshipsystems
must manage emotional attachment and user dependence [ 85]. Technical safeguards such as anti-
sycophancy detection [ 99] and crisis-intervention modules identify unhealthy behavioral patterns [ 29],
while responsible interface design reinforces AI identity disclosure [ 86] and encourages real-world
social engagement [ 82].Virtual Idolsface challenges of brand safety and persona integrity during
large-scale live interactions. Hybrid moderation frameworks [ 56] combine automated pre-screening,
human oversight, and human-in-the-loop (HITL) control to ensure consistent persona behavior and
5

prevent reputational harm [ 26]. Collectively, story characters probe the boundary of autonomy,
romantic companions the depth of emotion, and virtual idols the reach of public influence—together
illustrating how LLM-based virtual companionship diversifies into distinct paradigms of autonomous
action, emotional connection, and performative interaction.
3 Quadrant II: Functional Virtual Assistants
Following the emotionally oriented virtual companionship discussed in Quadrant I, this section
focuses on thefunctional dimensionof personified AI—agents designed forcognitive augmentation,
task execution, and professional collaborationrather than emotional attachment. While virtual
companions emphasize empathy and creativity, functional assistants pursue efficiency, reliability, and
contextual reasoning, supporting applications across work, education, healthcare, and everyday life.
This quadrant marks the shift from“AI that feels”to“AI that thinks and acts with humans”. Instead
of simulating intimacy, these systems enhance human decision-making throughstructured reasoning,
multimodal perception, andadaptive interaction. Their persona is inherentlyinstrumental rather
than emotional, operating under clear objectives, verifiable outputs, and strict safety and privacy
constraints. This chapter examines three representative domains: (1)Workplace scenarios—
cognitive augmentation and tool integration for productivity and collaboration; (2)Game scenarios
— persona modeling and narrative generation for immersive interaction; and (3)Psychological
counseling scenarios— empathetic dialogue and ethical safeguards.
3.1 Workplace Scenarios: Cognitive Copilots and Expert Agents
Within the domain of functional assistants, workplace scenarios represent the core manifestation of
the “cognitive copilot” paradigm. In such systems, the persona is designed purely for functionality,
serving as an expert agent seamlessly embedded within organizational workflows. Unlike emotionally
oriented AI companions, workplace personas are shaped as professional instruments characterized by
reliability, efficiency, and contextual awareness. Their “consistency” lies not in affective coherence
but in logical and factual precision.
Enterprise-level applications of LLM personas have converged on three principal domains: (1)enter-
prise assistants[ 7] that integrate internal data and automate workflows, (2)customer service agents
that preserve brand consistency, and (3)training simulators[ 34] that provide safe environments for
skill development. However, deploying LLM personas in enterprise contexts presents four major chal-
lenges: data security and grounding [ 31], persona generation bias, ROI evaluation [ 7], and simulation
fidelity [ 34]. Emerging trends indicate a shift toward hyper-specialization [ 91], multi-agent workflow
automation [ 11], and deep human–AI collaboration. (Detailed examples and technical analysis are
provided in Appendix D.)
Strategically, Retrieval-Augmented Generation (RAG) has become the central mechanism for im-
plementing enterprise personas. Since public LLMs cannot securely process proprietary data
and full-scale model fine-tuning remains prohibitively costly [ 31], RAG offers a pragmatic solu-
tion—maintaining the independence of the base model while safely injecting contextual data through
controlled retrieval. Consequently, the competitive edge of enterprise AI lies not in persona design
but in the efficiency and robustness of data pipelines and retrieval governance. More importantly,
in workplace contexts, the persona has evolved into a synonym for process automation: invoking a
persona such as a “cybersecurity auditor” [ 91] effectively triggers an encapsulated workflow [ 11]
of specialized knowledge, skills, and operations. The future of enterprise AI will extend beyond
conversational assistants toward a library of process-oriented personas, allowing employees to deploy
them on demand for professional tasks—fundamentally reshaping work delegation and organizational
management.
3.2 Game Scenarios: The Dawn of Generative Narrative
This section explores the revolutionary impact of LLM personas on the gaming industry, focusing on
how they are evolving Non-Player Characters (NPCs) from static, pre-scripted interaction models
to dynamic, believable agents [ 95] capable of fostering "generative narrative" and deepening player
immersion. Unlike the "functional utility" personas in workplace scenarios, AI personas in gaming
pursue "narrative believability." Applications are concentrated in two main areas: (1)Dynamic and
6

believable NPCs[ 53], allowing players to engage in open-ended, natural language conversations
and receive dynamic responses based on the game state; and (2)Generative narrative and player
co-creation[ 105,3], where the AI adjusts storylines in real-time [ 100], transforming players from
passive participants into active co-designers.
However, real-time game interaction presents three core challenges for LLMs: (1)The low-latency
inference challenge, where network latency can instantly break immersion [ 14], forcing the industry
toward on-device Small Language Models (SLMs) [ 18]; (2)Modeling believable emotion and
behavior, which requires AI to go beyond text and integrate psychological theories (e.g., Appraisal
Theory) [ 64] and multimodal expression; and (3)The narrative coherence dilemma[ 105], balancing
the vast freedom LLMs provide with the need to maintain a structured narrative, a limitation also noted
by game designers [ 3]. Future trends point toward cross-platform persistent personas, fully generative
worlds (PCG), and the rise of the "AI Game Master." (Detailed analysis of leading prototypes like
Ubisoft’s NEO NPCs [ 27],Dead Meat[ 88], underlying technologies like NVIDIA ACE [ 18], and
psychological modeling [64] is provided in Appendix E.)
Strategically, the gaming industry’s extreme low-latency requirement for real-time interaction [ 14]
is forcing it to become the primary driver of on-device, low-latency AI technology. Enterprise
applications can tolerate cloud latency in exchange for scalability, but gaming’s zero-tolerance for
latency (as it shatters immersion) compels the industry to solve the "last mile" problem of AI: running
complex models efficiently on consumer hardware. Consequently, innovations from NVIDIA (ACE)
[18] and studios dedicated to SLM integration [ 88] are at the forefront of inference optimization.
The techniques pioneered for on-device game AI will eventually permeate other domains requiring
real-time, offline AI (e.g., robotics, edge computing). The gaming industry is becoming the R&D
testbed for the future of embedded AI.
Concurrently, the role of the game writer is undergoing a fundamental shift, from“Scriptwriter”
to“AI Cultivator. ”Traditional narrative designers created branching dialogue trees; in the era of
generative NPCs, this model is obsolete. As demonstrated by Ubisoft’s NEO NPC project [ 27], the
writer’s new duty is to create a rich "seed" for the character (backstory, motivations, linguistic style).
The writer then "conditions" and "guides" the LLM through iteration, teaching it how to embody
the role. The writer becomes a director and curator, shaping the AI’s improvisation. This represents
a fundamental change in the creative workflow of the gaming industry, where the core value is no
longer script-writing, but rather creating the foundational "Character Bible" and guardrails [ 3] that
support "Controlled Improvisation."
3.3 Mental Health Scenario: The High-Stakes Frontier
This section provides a cautious and nuanced examination of LLM personas in the mental health sector,
aiming to balance their immense potential for enhancing service accessibility [ 42] with the profound
ethical, safety, and clinical challenges inherent in deploying AI for therapy. Unlike the "functional"
personas in workplace scenarios or the "narrative" personas in gaming, the persona in this domain
pursuesTherapeutic Efficacy. Its applications are concentrated in: (1)Digital Therapeuticsas "AI
Counselors," designed to anonymously and scalably deliver evidence-based interventions like CBT
[28]; (2)Alignment with Clinical Frameworks (e.g., CBT), using prompting [ 111] or specialized
model design [ 40] to make general-purpose LLMs behave more like professional therapists; and (3)
Human-in-the-Loop (HITL) Modelsas assistive tools, reflecting expert consensus on the need for
human oversight [120].
However, this frontier faces three core challenges: (1)The Empathy Paradox, where AI excels
atsimulatingcognitive empathy (recognizing emotion) [ 61] but lacks genuine affective empathy
(sharing experience), an "deceptive empathy" considered ethically problematic [ 4]; (2)Clinical
Safety and Risk Mitigation, the most severe challenge [ 42], especially the risk of AI failing to
handle users in crisis (e.g., suicidal ideation) [ 90]; and (3)The Ethical and Regulatory Minefield,
involving data privacy (HIPAA compliance), AI identity disclosure, and lack of clinical validation
[101,4]. Future trends point toward clinically validated, specialized models [ 40], HITL becoming
the standard [ 120], and the establishment of industry-wide safety standards. (Detailed analysis of
platforms like Woebot and Wysa [ 28], the LLM4CBT study [ 111], multi-layered safety protocols
[90], and HIPAA regulations [101] is provided in Appendix F.)
7

Strategically, the mental health AI market will inevitably bifurcate into"Wellness" and "Clinical"
tiers[ 101]. The technical and regulatory barriers to creating a truly safe and effective "AI therapist"
[40] are immense. The risks of misdiagnosis, mishandled crises, and ethical breaches [ 4] are too
high for unregulated, general-purpose tools. Concurrently, a massive consumer demand exists for
companionship and low-level emotional support (e.g., Replika). This will force a market split: the
"Wellness" tier will consist of AI companions focused on entertainment and general well-being,
accompanied by strong disclaimers; the "Clinical" tier will consist of highly regulated, evidence-
based tools, designed as medical devices or therapist aids, requiring clinical validation and HIPAA
compliance [101].
Furthermore,safety in mental health AI is a dynamic, multi-layered system, not a static filter.
Initial AI safety approaches focused on simple content filtering, which proved grossly inadequate
as users can express severe suicidal ideation using subtle, non-explicit language [ 90]. The expert-
recommended solution is a dynamic, multi-layer system comprising clinical keyword detection,
contextual sentiment analysis, and risk-assessment engines [ 90]. The core task is not to block words,
but to understand intent and conversational trajectory. More importantly, safety is not just detection
butaction—specifically, a robust protocol for escalating users to human intervention. For any
organization developing mental health AI, investing in a complex, multi-layered safety and escalation
system is not an optional feature; it is the core, non-negotiable foundation of the product.
4 Quadrants III & IV: Persona in Embodied Intelligence
Following the discussion of "virtual" personas in the first two quadrants, this section shifts from
the "virtual" to the "physical" world, analyzing the application of LLM personas in "Embodied
Intelligence" entities [ 46]. This section will holistically examine three key application scenarios
(companion robots, home assistants, special group companionship), four core challenges (technical,
privacy, ethical, economic), and future strategic trajectories.
4.1 Application Scenarios: From "Emotional Pets" to "Therapeutic Tools"
The application scenarios for embodied personas have shown a clear market bifurcation:
•Quadrant III: The General Home Market.This domain presents a "Form-Persona
Dilemma." (1)Non-humanoid Companions(e.g., Sony Aibo, Lovot) adopt a "pet-like
persona," relying on non-verbal cues for emotional connection, skillfully avoiding the
"uncanny valley." (2)Functional Assistants(e.g., Amazon Astro) follow a "utility-first,
persona-second" strategy. Their primary value is security and convenience; the Alexa
persona is an add-on layer. (3)Humanoid Assistants(e.g., Tesla Optimus, Figure AI)
pursue functional and morphological unity [ 83], leveraging LLMs to complete complex,
long-horizon tasks in unpredictable environments [76].
•Quadrant IV: Vertical Application Markets.This domain addresses clear, high-value pain
points. (1)Elderly Care(e.g., ElliQ), where the persona is designed as a "proactive coach"
to alleviate loneliness and provide health monitoring [ 116]. (2)Special Education(e.g.,
QTrobot) utilizes a "Therapeutic Persona"—a non-judgmental, highly patient presence—to
act as a "social mediator," assisting children with ASD in social skills training [19].
4.2 Core Challenges: From "Symbol Grounding" to "Ethical Debt"
Despite a promising outlook, the deployment of embodied personas faces four severe challenges:
1.Technical Barriers:The core challenge is the"Symbol Grounding Prob-
lem"—connecting the LLM’s abstract symbols (e.g., "apple") with the physical entity
a VLM (Vision-Language Model) perceives. This requires a robust world model integrating
perception, planning, and memory [ 102]. Furthermore,Latency(hindering real-time inter-
action) andHallucinations(extremely dangerous in high-stakes medical scenarios) remain
major bottlenecks, severely impacting credibility in real-world tests [44].
2.Privacy and Security:Embodied robots are unprecedented "data collection terminals."
Their cameras, microphones, and LIDAR pose profound privacy threats. The core user
8

anxiety stems not just from data collection, but from the AI’s "inference" of sensitive
information [13].
3.Ethics and Legality:The key obstacle isambiguous liability(who is responsible if the AI
errs?) [ 50]. Moreover, algorithmic bias, "emotional deception" of vulnerable populations
(children, elderly) [ 44], and the "re-identification" risk of HIPAA-protected data constitute a
significant"Ethical Debt."
4.Economic Barriers:High hardware costs, an unclear value proposition, and the "expectation
gap" between sci-fi portrayals and current reality are major factors hindering mass-market
adoption.
4.3 Future Trajectory and Strategic Implications
The future trends for embodied intelligence are clear: (1)From passive response to proactive
intelligence(anticipating needs) [ 116]; (2)Functional fusion(blending physical assistance with
emotional support) [ 76,83]; and (3)Ecosystem integration(as a central hub for smart homes and
telehealth).
Strategically, the market is bifurcated.The path to success for general-purpose home robots
is "utility-first" (like Astro), whereas vertical markets (elderly care, special ed) have become the
most viable commercial "beachheads" due to their high-value proposition [ 19]. For developers,
"Privacy-by-Design" [ 13] must be the rule. For policymakers, the urgent task is to establish clear
legal frameworks for liability and data privacy (e.g., updating HIPAA) [ 50] to guide innovation while
protecting consumers.
(Detailed analysis of cases like Aibo, Astro, ElliQ, QTrobot, symbol grounding, HIPAA challenges,
and specific recommendations for investors and developers is provided in Appendix G.)
5 Conclusion
This paper has proposed a systematic four-quadrant taxonomy to deconstruct and analyze the complex
landscape of LLM persona in AI companion applications. By navigating the axes of "Emotional vs.
Functional" and "Virtual vs. Embodied," we have systematically mapped the diverse modalities, from
virtual idols to embodied care robots, revealing the unique technical stack, strategic focus, and ethical
considerations for each quadrant.
Our analysis confirms that "persona" is not a monolithic concept but a multi-dimensional design
space where the core challenges fundamentally change with the application.
1.InQuadrant I (Virtual Companionship), the central challenge isemotional depth and
long-term consistency, requiring specialized models (e.g., RoleLLM) and architectures
(e.g., relational graphs) to overcome "persona drift."
2.InQuadrant II (Functional Assistants), the focus shifts toreliability, efficiency, and
safety. This drives key innovations such as enterprise-grade RAG, on-device SLMs for
low-latency gaming, and stringent multi-layered safety protocols for mental health.
3.InQuadrants III & IV (Embodied Intelligence), we face the ultimate challenge of
symbol grounding—connecting abstract symbols to physical reality. Concurrently, as
unprecedented "data collection terminals," embodied AI brings issues of privacy and legal
liability to the forefront.
This study demonstrates that the AI persona market is bifurcating along different technical trajectories.
Gaming and "Wellness" applications are pushing the frontier of low-latency, on-device AI; whereas
"Enterprise" and "Clinical" applications prioritize verifiable reliability and safety (often via HITL and
RAG). For the most challenging embodied intelligence, high-value vertical markets (e.g., elderly care,
special education) currently offer a clearer path to commercialization than general-purpose home
robots.
In conclusion, this taxonomy provides not only a structured analytical tool for academic research
but also strategic insights for industry practitioners, helping them anticipate and address the distinct
challenges posed by each quadrant as they design, deploy, and regulate increasingly personified AI
systems.
9

References
[1]Jaewoo Ahn, Taehyun Lee, Junyoung Lim, Jin-Hwa Kim, Sangdoo Yun, Hwaran Lee, and
Gunhee Kim. Timechara: Evaluating point-in-time character hallucination of role-playing
large language models. InFindings of the Association for Computational Linguistics: ACL
2024, page 3291–3325, 2024. doi: 10.18653/v1/2024.findings-acl.197. URL https://
aclanthology.org/2024.findings-acl.197/.
[2]Rama Akkiraju, Anbang Xu, Deepak Bora, Tan Yu, Lu An, Vishal Seth, Aaditya Shukla,
Pritam Gundecha, Hridhay Mehta, Ashwin Jha, Prithvi Raj, Abhinav Balasubramanian, Murali
Maram, Guru Muthusamy, Shivakesh Reddy Annepally, Sidney Knowles, Min Du, Nick
Burnett, Sean Javiya, Ashok Marannan, Mamta Kumari, Surbhi Jha, Ethan Dereszenski,
Anupam Chakraborty, Subhash Ranjan, Amina Terfai, Anoop Surya, Tracey Mercer, Vinodh
Kumar Thanigachalam, Tamar Bar, Sanjana Krishnan, Samy Kilaru, Jasmine Jaksic, Nave
Algarici, Jacob Liberman, Joey Conway, Sonu Nayyar, and Justin Boitano. Facts about
building retrieval augmented generation-based chatbots.arXiv preprint arXiv:2407.07858,
2024. URLhttps://arxiv.org/abs/2407.07858.
[3]Seyed Hossein Alavi, Weijia Xu, Nebojsa Jojic, Daniel Kennett, Raymond T. Ng, Sudha Rao,
Haiyan Zhang, and Bill Dolan. Game plot design with an llm-powered assistant: An empirical
study with game designers. InProceedings of the 2024 International Conference on Games
(IEEE CoG 2024), 2024. URLhttps://arxiv.org/pdf/2411.02714.
[4]Ahmed Aly, Simon D’Alfonso, Shiyan D’Mello, and Greg Wadley. Ethical Principles for AI
in Mental Health: A Framework Developed with Mental Health Professionals. InProceedings
of the 2024 CHI Conference on Human Factors in Computing Systems (CHI ’24). ACM, 2024.
[5]Natale Amato, Berardina De Carolis, Francesco de Gioia, Mario Nicola Venezia, Giuseppe
Palestra, and Corrado Loglisci. Can an ai-driven vtuber engage people? the kawaii case study.
InJoint Proceedings of the ACM IUI Workshops 2024, March 18-21, 2024, Greenville, South
Carolina, USA, volume 3660 ofCEUR-WS, page —. CEUR Workshop Proceedings, 2024.
URLhttps://ceur-ws.org/Vol-3660/paper21.pdf.
[6]Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones,
Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine
Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli
Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal
Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer,
Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston,
Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton,
Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben
Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. Con-
stitutional ai: Harmlessness from ai feedback. Technical report, Anthropic, 2022. URL https:
//www-cdn.anthropic.com/7512771452629584566b6303311496c262da1006/
Anthropic_ConstitutionalAI_v2.pdf.
[7]Erik Brynjolfsson, Danielle Li, and Lindsey R Raymond. Generative AI at Work.NBER
Working Paper Series, (w31161), 2024.
[8]Jonas Budarf, Chandrima G. M. B. C. M. L. G. A. L. A. S. L. A. J. S. H. T. A. J. D. K. R. L. E.
P. A. A., and D. A. Prosocial ai: A framework for aligning large language models with human
well-being, 2024.
[9]Zhonghao Cao, Weikang Liu, Yifei Yang, Yitong Zhang, Ruixiang Zhu, Jiachen Li, Zhiyong
Zhang, Yian Li, Xinyi Liu, Siyuan Du, et al. Llm-powered empathetic robot for assisting
children with autism spectrum disorder. In2024 IEEE/RSJ International Conference on
Intelligent Robots and Systems (IROS), pages 10166–10173. IEEE, 2024.
[10] Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey. Persona vectors:
Monitoring and controlling character traits in language models. 2025. URL https://arxiv.
org/abs/2507.21509.
10

[11] Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chen Qian, Chi-Min
Chan, Yujia Qin, Yaxi Lu, Ruobing Xie, Xinyu Dai, Hanyu Lai, Siyuan Yang, Yilun Liu, Ziyue
Wu, Yuxiang Wu, Yong-Bin Kang, Zhirui Zhang, Jieming Zhu, Xunjian Yin, Enhong Chen,
Jianguo Li, Xiao-Yong Wei, Zhenzhong Lan, Yan Zhang, Jie Tang, Zhiyuan Liu, Maosong Sun,
and Yujiu Yang. Agentverse: Facilitating multi-agent collaboration and exploring emergent
behaviors in large language models, 2024.
[12] Yuan-Hao Chen and et al. Hi-reco: High-fidelity real-time conversational digital humans.
Creative Intelligence and Synergy Lab, HKUST-GZ, 2024.
[13] K. Chmielewski and S. Privacy, Trust, and Data Inference: The User’s Dilemma with Mobile
Social Robots in the Home.IEEE Robotics & Automation Magazine, 31(2):45–56, 2024.
[14] Frederik Roland Christiansen, Linus Nørgaard Hollensberg, Nikoline Bach Fleng Jensen,
Kristian Julsgaard, Kristian Nyborg Jespersen, and Ivan Adriyanov Nikolov. Exploring
presence in interactions with llm-driven npcs: A comparative study of speech recognition
and dialogue options. InVRST ’24: Proceedings of the 30th ACM Symposium on Virtual
Reality Software and Technology, 2024. doi: 10.1145/3641825.3687716. URL https:
//doi.org/10.1145/3641825.3687716.
[15] Frederik Roland Christiansen, Linus Nørgaard Hollensberg, Nikoline Bach Fleng Jensen,
Kristian Julsgaard, Kristian Nyborg Jespersen, and Ivan Adriyanov Nikolov. Exploring
presence in interactions with llm-driven npcs: A comparative study of speech recognition
and dialogue options. InVRST ’24: Proceedings of the 30th ACM Symposium on Virtual
Reality Software and Technology, 2024. doi: 10.1145/3641825.3687716. URL https:
//doi.org/10.1145/3641825.3687716.
[16] Minh Duc Chu, Patrick Gerard, Kshitij Pawar, Charles Bickham, and Kristina Lerman. Illusions
of intimacy: Emotional attachment and emerging psychological risks in human-ai relationships.
2025. URLhttps://arxiv.org/abs/2505.11649.
[17] NVIDIA Corporation. Deploy the first on-device small language model for improved game
character roleplay. Technical report, 2024. URL https://developer.nvidia.com/blog/
deploy-the-first-on-device-small-language-model-for-improved-game-character-roleplay/ .
[18] NVIDIA Corporation. Nvidia ace autonomous game characters: On-
device small language models for real-time npcs. Technical re-
port, 2025. URL https://www.nvidia.com/en-us/geforce/news/
nvidia-ace-autonomous-ai-companions-pubg-naraka-bladepoint/.
[19] A. P. Costa and W. Long-term Feasibility of QTrobot as a Social Mediator in Autism Spectrum
Disorder Therapy.JMIR Human Factors, 11:e51240, 2024.
[20] Yinghao Du, Shiyin Kang, Zhaofeng Liu, Weidong Guo, Zhaohong Li, Lei Xie, and Wayne Xin
Zhao. Cosyvoice: A zero-shot speech synthesis as-you-want model, 2024.
[21] Zongcai Du, Guilin Deng, Xiaofeng Guo, Xin Gao, Linke Li, Kaichang Cheng, Fubo Han,
Siyu Yang, Peng Liu, Pan Zhong, and Qiang Fu. Ditsinger: Scaling singing voice synthesis
with diffusion transformer and implicit alignment. 2025. URL https://arxiv.org/abs/
2510.09016.
[22] Yan Duan, Yixuan Zhang, Jun Qi, Yatao Cui, Lidan Shou, Weizhi Wang, and Daxin Jiang.
Advancing human-computer interaction in the new era of large language models.The VLDB
Journal, pages 1–27, 2024.
[23] Osamu Ekhator. Replika ai review 2025: I tested it for 5 days — here’s what i found.
Online article, Techpoint Africa, 2025. URL https://techpoint.africa/guide/
replika-ai-review/.
[24] EmergentMind. Full-duplex spoken dialogue model. Online article, 2025. URL https:
//www.emergentmind.com/topics/full-duplex-spoken-dialogue-model.
[25] Epic Games. Real-time workflows with metahuman animator | unreal fest orlando 2025.
https://www.youtube.com/watch?v=uiXa-T8q5-s, October 2025.
11

[26] Raymond F. Content moderation at scale: Handling millions of messages without sac-
rificing ux. Online article, Stream Blog, 2025. URL https://getstream.io/blog/
scaling-content-moderation/.
[27] Guillemette Fahs, Xavier Gaultier, and P V Barros. Project NEO: Ubisoft’s Prototype of
Generative AI NPCs. InGame Developers Conference (GDC). Inworld AI & Ubisoft, 2024.
[28] Kathleen K Fitzpatrick, Alison Darcy, and G. Long-term Engagement and Clinical Outcomes
of an AI-Driven CBT Intervention: A Longitudinal Study of Woebot. InDigital Health (JMIR),
2024.
[29] Allen Frances and Luciana Ramos. How can chatbots be made safe for psychiatric pa-
tients.Psychiatric Times, 2025. URL https://www.psychiatrictimes.com/view/
how-can-chatbots-be-made-safe-for-psychiatric-patients.
[30] Leyang Gao, Jingwei Zuo, Zhafi K. Shah, Ziyao Wang, Hong-Bin YU, Yujiu Yang, Ee-Peng
Lim, Zhen-Zhong Lan, and Yan Zhang. Safe: A survey of safety and ethics of autonomous
agents, 2024.
[31] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun,
Meng Wang, and Haofen Han. RAG vs. Fine-Tuning: Which Is Better for Domain-Specific
LLMs?arXiv preprint arXiv:2403.18430, 2024.
[32] Yunfan Gao, Yunfan Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yilun Bi, Yi Dai, Jiawei
Sun, Meng Guo, Haofei Liu, et al. Retrieval-augmented generation for large language models:
A survey.arXiv preprint arXiv:2402.19473, 2024.
[33] Yuxuan Gao, Zhaofeng Liu, Zhaohong Li, and Wayne Xin Zhao. Echar: A conversational
agent with empathetic chat and role-playing abilities, 2024.
[34] Zhaofeng Gong, Kexin Lin, Feng Wang, Hongfang Li, Feng Jiang, Jing Wen, Jiafei Liu, Chao
Lu, Weimin Liu, Xiaofeng Guo, et al. AgentSims: A Generalist Agent-based Simulation
Framework.arXiv preprint arXiv:2402.16664, 2024.
[35] Tongxin Guan, Ziyang Chen, Yilun Xu, Wenzhi Cui, and Xin Chen. Bridging the gap: A study
of audience interaction dynamics in virtual idol livestreaming. InProceedings of the 2024 CHI
Conference on Human Factors in Computing Systems (CHI ’24), page 1–16, New York, NY ,
USA, 2024. Association for Computing Machinery (ACM). doi: 10.1145/3613904.3642732.
[36] Wenxiang Guo, Yu Zhang, Changhao Pan, Rongjie Huang, Li Tang, Ruiqi Li, Zhiqing Hong,
Yongqi Wang, and Zhou Zhao. Techsinger: Technique controllable multilingual singing voice
synthesis via flow matching. 2025. URLhttps://arxiv.org/abs/2502.12572.
[37] Haoran Hao, Jiaming Han, Changsheng Li, Yu-Feng Li, and Xiangyu Yue. Retrieval-
augmented personalization for multimodal large language models.arXiv preprint
arXiv:2410.13360, 2024. URLhttps://arxiv.org/abs/2410.13360.
[38] Sirui Hong, Xiawu Chen, Shuai Zheng, Jinlin Dai, Yilun Li, Ceyao Lu, Siyuan Cheng,
Lingfeng Wang, Zilong Jiang, Zhineng Liu, et al. MetaGPT: Meta Programming for Multi-
Agent Collaborative Framework.arXiv preprint arXiv:2402.04909, 2024.
[39] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. 2021.
URLhttps://arxiv.org/abs/2106.09685.
[40] He Hu, Yucheng Zhou, Juzheng Si, Qianning Wang, Hengheng Zhang, Fuji Ren, Fei Ma, and
Laizhong Cui. Beyond empathy: Integrating diagnostic and therapeutic reasoning with large
language models for mental health counseling.arXiv preprint arXiv:2505.15715, 2025. URL
https://arxiv.org/abs/2505.15715.
[41] Sihao Hu, Tiansheng Huang, Fatih Ilhan, Selim Tekin, Gaowen Liu, Ramana Kompella,
and Ling Liu. A survey on large language model-based game agents.arXiv preprint
arXiv:2404.02039, 2024. URLhttps://arxiv.org/abs/2404.02039.
12

[42] S. Huang, F. Fu, K. Yang, K. Zhang, and F. Yang. The applications of large language models in
mental health.JMIR Mental Health, 11:e57306, 2024. URL https://mental.jmir.org/
2024/1/e57306.
[43] S. Huang, F. Fu, K. Yang, K. Zhang, and F. Yang. The applications of large language
models in mental health.JMIR Mental Health, 27:e69284, 2025. doi: 10.2196/69284. URL
https://mental.jmir.org/2025/1/e69284.
[44] B. Irfan et al. Between reality and delusion: challenges of applying large language models
to conversational robots.Autonomous Robots, 2025. URLhttps://link.springer.com/
article/10.1007/s10514-025-10190-y.
[45] Maurice Jakesch, Johanna Knopp, Benedikt Fecht, C. T. Lystbæ k, Ann-Katrin Schmidt,
Benedikt Prill, A. M. Küfner, Georg Groh, and L. Hornuf. "when the ai wrote “i love you”:
The human-ai “emotional contract” and the replika case. InProceedings of the 2024 CHI
Conference on Human Factors in Computing Systems (CHI ’24), page 1–18, New York, NY ,
USA, 2024. Association for Computing Machinery (ACM). doi: 10.1145/3613904.3642316.
[46] H. Jeong, Haechan Lee, C. Kim, and S. Shin. A survey of robot intelligence with large
language models.Applied Sciences, 14(19):8868, 2024. URL https://www.mdpi.com/
2076-3417/14/19/8868.
[47] Ke Ji, Yixin Lian, Linxu Li, Jingsheng Gao, Weiyuan Li, and Bin Dai. Enhancing persona
consistency for llms’ role-playing using persona-aware contrastive learning. 2025. URL
https://arxiv.org/abs/2503.17662.
[48] Ke Ji, Yixin Lian, Linxu Li, Jingsheng Gao, Weiyuan Li, and Bin Dai. Enhancing persona
consistency for llms’ role-playing using persona-aware contrastive learning, 2025.
[49] Diqiong Jiang, Jian Chang, Lihua You, Shaojun Bian, Robert Kosk, and Greg Maguire. Audio-
driven facial animation with deep learning: A survey.Information, 15(11):675, 2024. doi:
10.3390/info15110675. URLhttps://doi.org/10.3390/info15110675.
[50] Margot E. Kaminski and G. Regulating Embodied AI: The Liability Challenge.Brookings
Institution TechStream, February 2024. Analyzes the legal ambiguity of accountability for
autonomous systems.
[51] Hangyeol Kang, Thiago Freitas dos Santos, Maher Ben Moussa, and Nadia Magnenat-
Thalmann. Mitigating the uncanny valley effect in hyper-realistic robots: A student-
centered study on llm-driven conversations.arXiv preprint arXiv:2503.16449, 2025. URL
https://arxiv.org/abs/2503.16449.
[52] Rajat Khanda. Agentic ai-driven technical troubleshooting for enterprise systems: A novel
weighted retrieval-augmented generation paradigm.arXiv preprint arXiv:2412.12006, 2024.
URLhttps://arxiv.org/abs/2412.12006.
[53] Byungjun Kim, Minju Kim, Dayeon Seo, and Bugeun Kim. Leveraging large language models
for active merchant non-player characters.arXiv preprint arXiv:2412.11189, 2024. URL
https://arxiv.org/abs/2412.11189.
[54] Y-S. Kim, J. H. Yang, J. H. Lee, J. Choi, and M. Cha. Unpacking privacy expectations and
preferences for ai companions. InProceedings of the 2024 ACM Conference on Computer-
Supported Cooperative Work and Social Computing (CSCW ’24), New York, NY , USA, 2024.
Association for Computing Machinery (ACM).
[55] Ruolin Kong, Ziheng Qi, and Sicheng Zhao. Difference between virtual idols and traditional
entertainment from technical perspectives. InProceedings of the 2021 3rd International
Conference on Economic Management and Cultural Industry (ICEMCI 2021), pages 344–349.
Atlantis Press, 2021. doi: 10.2991/assehr.k.211209.058. URL https://doi.org/10.2991/
assehr.k.211209.058.
[56] Amit Kumar. Hybrid moderation models: Balancing ai and human oversight. Online article,
Infosys BPM Blog, 2025. URL https://www.infosysbpm.com/blogs/trust-safety/
hybrid-moderation-models-balancing-ai-and-human-oversight.html.
13

[57] Haesoo Kwon, Juyoung Lee, Yoonjoo Sung, and Hwajung Kim. "my ai friend": A study of
how users of a commercial ai chatbot app experience parasocial relationships and emotional
support. InProceedings of the 2024 CHI Conference on Human Factors in Computing Systems
(CHI ’24), page 1–17, New York, NY , USA, 2024. Association for Computing Machinery
(ACM). doi: 10.1145/3613904.3642323.
[58] Alex Leary. Real-time motion capture in unreal engine: A complete guide to professional
animation. Online article, Remocapp Blog, 2025. URL https://remocapp.com/blog/
posts/1340/real-time-motion-capture-unreal-engine. Accessed Mar 1 2025.
[59] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and
Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. 2020. URL
https://arxiv.org/abs/2005.11401.
[60] Jialu Li, Yuanzhen Li, Neal Wadhwa, Yael Pritch, David E. Jacobs, Michael Rubinstein, Mohit
Bansal, and Nataniel Ruiz. Unbounded: A generative infinite game of character life simulation.
arXiv preprint arXiv:2410.18975, 2024. URLhttps://arxiv.org/abs/2410.18975.
[61] Jiaxin Li, Chen Wang, Zhaoyu Zhao, Tao Zhang, and Tong Liu. The Empathy Paradox:
A Systematic Review on AI-Generated Empathy in Mental Health.IEEE Transactions on
Affective Computing, 2024.
[62] Pei Li, Zhaofeng Liu, Guanzheng Chen, Wayne Xin Zhao, and Ji-Rong Wen. Shepherd: A
framework for guiding and distilling llm-based agents, 2024.
[63] Yuxin Li, Yu-Chen Lin, Zixun Lu, Yijia He, Ruijia Wang, Bing-Chang Chen, and Yun-Nung
Chen. Streamassist: An llm-based assistant for live streaming, 2024.
[64] H Tosca Lim, N Mor, S Gehrmann, K Conger, G Ball, K Bar, M Haas, M Kim,
M Geva, M Abrams, et al. An architecture for believable cognitive NPCs.arXiv preprint
arXiv:2404.18256, 2024.
[65] Ming-Hao Lin, Po-Chuan Li, Chun-Yi Kuan, Andy T. Liu, and Hung yi Lee. Speculative
end-turn detector for efficient speech chatbot assistant, 2025.
[66] Chao Liu, Mingyang Su, Yan Xiang, Yuru Huang, Yiqian Yang, Kang Zhang, and Mingming
Fan. Toward enabling natural conversation with older adults via the design of llm-powered
voice agents that support interruptions and backchannels. InProceedings of the 2025 CHI
Conference on Human Factors in Computing Systems (CHI ’25). ACM, 2025.
[67] Xuan Liu, Yuxi Wang, Jiacheng Wang, Yang Liu, and Jian Wu. Synthetic Personas for Trustwor-
thy AI: A Framework for Responsible Persona Generation.arXiv preprint arXiv:2405.02111,
2024.
[68] Keming Lu, Bowen Yu, Chang Zhou, and Jingren Zhou. Large language models are superposi-
tions of all characters: Attaining arbitrary role-play via self-alignment, 2024.
[69] Yihan Lu, Ziyang Chen, and Xin Chen. Behind the scenes: How vtuber fans perceive and
engage with the "nakanohito" (the person inside). InProceedings of the 2024 CHI Conference
on Human Factors in Computing Systems (CHI ’24), pages 1–16, New York, NY , USA, 2024.
Association for Computing Machinery (ACM). doi: 10.1145/3613904.3642721.
[70] Nandakishor M and Anjali M. Continuous learning conversational ai: A personalized agent
framework via a2c reinforcement learning, 2025. arXiv preprint arXiv:2502.12876.
[71] Yueen Ma, Zixing Song, Yuzheng Zhuang, Jianye Hao, and Irwin King. A survey on vision-
language-action models for embodied ai.arXiv preprint arXiv:2405.14093, 2024. URL
https://arxiv.org/abs/2405.14093.
[72] K. Malfacini. The impacts of companion ai on human relationships: risks and benefits.AI &
Society, 2025. doi: 10.1007/s00146-025-02318-6.
14

[73] Kim Malfacini. The impacts of companion ai on human relationships: risks, benefits, and
design considerations.AI & Society, 2025. doi: 10.1007/s00146-025-02318-6. URL https:
//link.springer.com/article/10.1007/s00146-025-02318-6.
[74] A. Manoli et al. Characterizing relationships with companion and assistant large language
models. InCSCW Companion ’25, 2025.
[75] A. Miller and W. N. Price. Re-identification Risks and AI: Updating HIPAA Safety Rules for
the LLM Era.Journal of Law, Medicine & Ethics, 52(1):200–215, 2Z024.
[76] R. Mon-Williams and other authors. Embodied large language models enable robots to
complete complex tasks in unpredictable environments.Nature Machine Intelligence, 2025.
URLhttps://www.nature.com/articles/s42256-025-01005-x.
[77] Ishani Mondal, Shwetha S, Anandhavelu Natarajan, Aparna Garimella, Sambaran Bandyopad-
hyay, and Jordan Boyd-Graber. Presentations by the humans and for the humans: Harnessing
llms for generating persona-aware slides from documents. InProceedings of the 18th Confer-
ence of the European Chapter of the Association for Computational Linguistics (Long Papers),
pages 2664–2684, 2024. URLhttps://aclanthology.org/2024.eacl-long.163/.
[78] Manisha Mukherjee, Sungchul Kim, Xiang Chen, Dan Luo, Tong Yu, and Tung Mai. From doc-
uments to dialogue: Building kg-rag enhanced ai assistants.arXiv preprint arXiv:2502.15237,
2025. URLhttps://arxiv.org/abs/2502.15237.
[79] Brijesh Nambiar. From turn-taking to synchronous dialogue: Building and measuring true full-
duplex systems. Online article, Medium, 2025. URL https://medium.com/@brijeshrn/
from-turn-taking-to-synchronous-dialogue-building-and-measuring-true-full-duplex-systems-794a07f3e59f .
[80] NVIDIA. Nvidia rtx advances with neural rendering and digital hu-
man technologies at gdc 2025. https://developer.nvidia.com/blog/
nvidia-rtx-advances-with-neural-rendering-and-digital-human-technologies-at-gdc-2025/ ,
March 2025.
[81] NVIDIA Corporation. NVIDIA Isaac Sim 2024: Accelerating AI-Based Robotics with Digital
Twins. NVIDIA GTC 2024 Technical Presentation, March 2024.
[82] OpenAI. Strengthening chatgpt’s responses in sensitive conversations. On-
line article, OpenAI Safety Research, 2025. URL https://openai.com/index/
strengthening-chatgpt-responses-in-sensitive-conversations/.
[83] OpenAI and Figure AI. Figure 01: A New Era of Humanoid Robots. OpenAI Technical Blog,
March 2024. Describes the integration of OpenAI VLM with Figure’s humanoid robot.
[84] Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and
Michael S. Bernstein. Generative agents: Interactive simulacra of human behavior. 2023. URL
https://arxiv.org/abs/2304.03442.
[85] The Attachment Project. Ai companions & attachment: Benefits, risks & red flags. Online
article, The Attachment Project Blog, 2025. URL https://www.attachmentproject.
com/blog/ai-companions/.
[86] Jiahao Qiu, Yinghui He, Xinzhe Juan, Yimin Wang, Yuhan Liu, Zixin Yao, Yue Wu, Xun Jiang,
Ling Yang, and Mengdi Wang. Emoagent: Assessing and safeguarding human-ai interaction
for mental health safety. 2025. URLhttps://arxiv.org/abs/2504.09689.
[87] Arman Salemi, Yihang Yu, Bingsheng Peng, Shuyu Han, Zhaoxuan Wu, Yitong Jiang, Hongbo
Zhuang, Yimeng Wang, Wei Wang, Hao Peng, et al. Personalized large language models: A
survey.arXiv preprint arXiv:2404.09117, 2024.
[88] Murray Shanahan. Forging ’Dead Meat’: Generative AI in an Indie murder mystery. Talk at
Game Developers Conference (GDC) 2024, March 2024.
[89] Shuai Shao, Qihan Ren, Chen Qian, Boyi Wei, Dadi Guo, Jingyi Yang, Xinhao Song, Linfeng
Zhang, Weinan Zhang, Dongrui Liu, and Jing Shao. Your agent may misevolve: Emergent
risks in self-evolving llm agents. 2025. URLhttps://arxiv.org/abs/2509.26354.
15

[90] Megha Sharma, Sharath Chandra Guntuku, and R. A Multi-Layered Safety System for AI in
Mental Health Support: Beyond Content Filtering.arXiv preprint arXiv:2404.10012, 2024.
[91] Yongchao Shen, Zhou Zhang, Jue Huang, Yu-Ping Chang, Boyang Li, Hesen Zhu, Yujun Zhao,
Fan Zhou, Tao Gong, Zhipeng Liu, et al. Agent-Tuning: Enabling Generalized Agent Abilities
for LLMs.arXiv preprint arXiv:2404.03714, 2024.
[92] Yao Shi, Zhaorong Wang, Haipeng Chen, Junge Zhang, Wei Wang, Hong-Bin YU, and Linqi
Song. Llm-fsm: A hybrid framework for building scalable and high-fidelity llm-based agents,
2024.
[93] SIGGRAPH. Celebrating 20 years! - advances in real-time rendering in games, siggraph 2025.
https://advances.realtimerendering.com/s2025/index.html, August 2025.
[94] Li Song. Llm-driven npcs: Cross-platform dialogue system for games and social platforms.
arXiv preprint arXiv:2504.13928v1, 2025. URL https://arxiv.org/abs/2504.13928v1 .
[95] Haochen Sun, Zhaofeng Chen, Zongjie Wang, Zhaofeng Zhang, Jiatong Li, Zhaohua Fan, and
Yang Liu. Mastering role-play: A case study on LLM-based NPCs in game. InProceedings of
the 2024 CHI Conference on Human Factors in Computing Systems, pages 1–16, 2024.
[96] Kairui Sun, Yuxin-Jia Wang, Zong-Yi Li, Siyuan Wu, Haotian Li, Jia-Chen Gu, Zhaofeng He,
and Yujiu Yang. Direcagent: A director-agent-based multi-agent system for open-ended story
generation, 2024.
[97] Anonymous (EmbodiedBench Team). Embodiedbench: Comprehensive benchmarking multi-
modal large language models for embodied agents. InICML 2025 Poster Session, 2025. URL
https://icml.cc/virtual/2025/poster/45994.
[98] Anthropic Engineering Team. How we built our multi-agent research system. Online ar-
ticle, Anthropic Engineering, 2025. URL https://www.anthropic.com/engineering/
built-multi-agent-research-system.
[99] TechPolicy.Press. Intimacy on autopilot: Why ai companions demand ur-
gent regulation. Online article, 2025. URL https://techpolicy.press/
intimacy-on-autopilot-why-ai-companions-demand-urgent-regulation.
[100] Tereza Todová and V ojt ˇech (supervisor) Br˚ uža. A quest for information: Enhancing
game-based learning with llm-driven npcs. InCESCG 2025 – 29th Computer Graphics
Student Conference, 2025. URL https://cescg.org/wp-content/uploads/2025/04/
A-Quest-for-Information-Enhancing-Game-Based-Learning-with-LLM-Driven-NPCs-2.
pdf.
[101] Alejandro Tornero, Medha Sah, and Anand Reddy. From Wellness to Clinical Validation: The
Regulatory and Evidentiary Hurdles for Mental Health LLMs.Journal of Medical Internet
Research (JMIR), 26(1):e54120, 2024.
[102] Author(s) Unknown. Embodied ai agents: Modeling the world.arXiv preprint
arXiv:2506.22355, 2025. URLhttps://arxiv.org/abs/2506.22355.
[103] Himansu Usaha. The complete guide to event-driven architec-
ture from pub/sub to event sourcing in production. Online arti-
cle, Medium, 2025. URL https://medium.com/@himansusaha/
the-complete-guide-to-event-driven-architecture-from-pub-sub-to-event-sourcing-in-production-f9dd468ed9e8 .
[104] Alhim Vera, Karen Sanchez, Carlos Hinojosa, Haidar Bin Hamid, Donghoon Kim, and Bernard
Ghanem. Multimodal safety evaluation in generative agent social simulations. 2025. URL
https://arxiv.org/abs/2510.07709.
[105] Peng Wang, Jiacheng Li, Danding Li, Peize Li, Jie Zhou, Bofei Wang, Yang Yang, and Mei
Chen. TaleWeaver: A General Framework for Interactive Story Generation. InProceed-
ings of the 2024 Annual Conference of the North American Chapter of the Association for
Computational Linguistics (NAACL), 2024.
16

[106] Xiaoyang Wang, Jiaming Li, Kun Li, Jinan He, Hongming Zhang, Wen-Huang Cheng, and
Tao Ge. Syncanimation: A real-time end-to-end framework for audio-driven human pose and
talking head animation, 2025.
[107] Xiaoyang Wang, Hongming Zhang, Tao Ge, Wenhao Yu, Dian Yu, and Dong Yu. Opencharac-
ter: Training customizable role-playing llms with large-scale synthetic personas, 2025.
[108] Yun-Long Wang, Zhaofeng Liu, Zhaohong Li, Weiming Lu, and Wayne Xin Zhao. Rag-triad:
A dynamic and iterative rag system for long-term conversational agents, 2024.
[109] Zekun Moore Wang, Zhongyuan Peng, Haoran Que, Jiaheng Liu, Wangchunshu Zhou, Yuhan
Wu, Hongcheng Guo, Ruitong Gan, Zehao Ni, Jian Yang, Man Zhang, Zhaoxiang Zhang,
Wanli Ouyang, Ke Xu, Stephen W. Huang, Jie Fu, and Junran Peng. Rolellm: Benchmarking,
eliciting, and enhancing role-playing abilities of large language models. 2023. URL https:
//arxiv.org/abs/2310.00746.
[110] Zilong Wang, Hao Zhang, Yunchang Chen, Xiangru Tang, Chunfeng An, Yujiu Yang, Chen-Yu
Lee, and Heng Ji. Chain-of-table: Evolving tables in the reasoning chain for table-based tasks,
2024.
[111] Jiachen Wei, Bingsheng Liu, Zheyuan Li, Qiong Zhu, Yihong Chen, Yizhe Zhang, and Yang
Liu. Can Large Language Models Be Good Cognitive Behavioral Therapists? A Proof-of-
Concept Study on LLM4CBT.arXiv preprint arXiv:2403.04633, 2024.
[112] Fei Wu, Weidong Guo, Yuxuan Gao, Zhaofeng Liu, Weiming Lu, and Wayne Xin Zhao.
Memochat: A human-like conversational agent with long-term memory, 2024.
[113] Tingying Wu. Ai love: An analysis of virtual intimacy in human-computer interaction.
Communications in Humanities Research, 50(1):143–148, 2024. doi: 10.54254/2753-7064/
50/20242466. URLhttps://doi.org/10.54254/2753-7064/50/20242466.
[114] Zhaoxuan Wu, Yilun Niu, Zhaohua He, Rui Chen, Colin M. Gray, and Ning Gu. The precarious
labor of being kiss-cut: The working reality of vtubers in china. InProceedings of the 2024
ACM Conference on Computer-Supported Cooperative Work and Social Computing (CSCW
’24), pages 1–34, New York, NY , USA, 2024. Association for Computing Machinery (ACM).
doi: 10.1145/3637311.3655106.
[115] Zhengyu Xi, Wenyi Chen, Yixuan Guo, Bowei He, Yuxin Hong, Weiran Lu, Xize Zhang,
Zhang-Ren Zhou, Ming Zhou, Tao Zhu, et al. The rise and potential of large language model
based agents: A survey.arXiv preprint arXiv:2401.06100, 2024.
[116] X. Xiong, Y . Song, C. Lee, and S. Sabanovic. Proactive and Personalized: Evaluating a Social
Robot (ElliQ) for Aiding Older Adults in Daily Life. InProceedings of the 2024 ACM/IEEE
International Conference on Human-Robot Interaction (HRI ’24), pages 556–565. ACM, 2024.
[117] Sih-Wei Yuan, Jui-Yang Hsu, Chien-Yu Huang, and Hung yi Lee. Promptsvc: Controllable
singing voice conversion with prompts, 2024.
[118] Chi Zhang, Changjia Zhu, Junjie Xiong, Xiaoran Xu, Lingyao Li, Yao Liu, and Zhuo Lu.
Guardians and offenders: A survey on harmful content generation and safety mitigation of llm.
2025. URLhttps://arxiv.org/abs/2508.05775.
[119] Renwen Zhang, Han Li, Han Meng, Jinyuan Zhan, Hongyuan Gan, and Yi-Chieh Lee. The
dark side of ai companionship: A taxonomy of harmful algorithmic behaviors in human-ai
relationships, 2024. arXiv preprint arXiv:2410.20130.
[120] XiuYu Zhang and Zening Luo. Integrating large language models in mental health practice.
Frontiers in Public Health, 12:1475867, 2024. URL https://www.frontiersin.org/
journals/public-health/articles/10.3389/fpubh.2024.1475867/fulltext.
[121] XiuYu Zhang and Zening Luo. Advancing conversational psychotherapy: Integrating pri-
vacy, dual-memory, and domain expertise with large language models.arXiv preprint
arXiv:2412.02987, 2024. URLhttps://arxiv.org/abs/2412.02987.
17

[122] Yuxin Zhang, Sanyuan Chen, Zhipeng Chen, Chen-Yu Lee, and Heng Ji. P-controller: A
feedback-based constrained text generation framework, 2024.
[123] Zijie Zhang, Lichen Wang, Zixun Chen, Yang Song, Haipeng Wang, and Yang Liu. Npm-sing:
A non-parametric multi-lingual singing voice synthesizer, 2024.
[124] Mengjie Zheng, Joseph Kim, Yoon Jung, and Sang Chi. Can ai chatbots be your friend? a
study of parasocial relationships with replika. InProceedings of the 2024 CHI Conference on
Human Factors in Computing Systems (CHI ’24), pages 1–16, 2024.
[125] Li Zhou, Jianfeng Gao, Di Li, and Heung-Yeung Shum. The design and implementation of
xiaoice, an empathetic social chatbot. 2018. URLhttps://arxiv.org/abs/1812.08989.
18

A Appendix. Virtual Story Character Interaction
This appendix provides a detailed technical overview ofInteractive Virtual Story Characters,
focusing on their architectural design across four layers:model, architecture, generation,andsafety &
ethics. It complements the main text by elaborating on implementation details and emerging research
trends beyond the core conceptual distinctions introduced earlier.
A.1 Model Layer: Deep Persona Consistency
Core Challenge:Maintaining a deep and coherent persona within a general-purpose large language
model (LLM) [ 47]. Pretrained LLMs lack role-specific background knowledge, memory, and
behavioral style, leading to “character hallucination” [ 1] —the breakdown of persona coherence and
narrative immersion.
Strategic Solutions. (1) Persona-Aware Fine-Tuning [ 77].This strategy embeds character traits
directly into model parameters through supervised fine-tuning. The representativeRoleLLM[ 109]
framework consists of three stages:
•Role Profile Construction:Building detailed role profiles covering background, personality,
linguistic style, and knowledge boundaries.
•Context-Instruct & RoleGPT:Automatically generating question–answer and dialogue
samples from role descriptions, transforming unstructured text into learnable instructions.
•Role-Conditioned Instruction Tuning (RoCIT):Fine-tuning smaller open-source models
(e.g., LLaMA) with these datasets to internalize character-specific reasoning and expression
patterns.
(2) Self-Alignment and Self-Reflection.TheDITTO[ 68] framework enables self-play dialogue
generation for self-supervised consistency.Persona Contrastive Learning (PCL)[ 48] introduces a
chain of persona self-reflections, prompting models to evaluate their own outputs against predefined
role profiles and self-correct without external supervision.
(3) Parameter-Efficient Fine-Tuning (PEFT).Methods such asLoRA (Low-Rank Adaptation)[ 39]
update only small, low-rank matrices while freezing base parameters, achieving near full-tuning
performance with drastically reduced computational cost.
Trade-offs and Emerging Trends.While persona-specific fine-tuning improves consistency, it
riskscatastrophic forgettingof general reasoning ability. Future work focuses on balancing persona
coherence and generalization. Experiments fromDITTO(4,000 roles) andRoleLLM(diversified
instruction generation) highlight the notion ofcharacter generalization—models trained on diverse
role data can quickly adapt to unseen personas, marking the transition from single-role tuning to
master role-playing models[107].
A.2 Architecture Layer: Persistent and Evolving Memory
Core Challenge:Overcoming the LLM’s limited context window to sustain long-term memory and
behavioral continuity for evolving story characters.
Strategic Solution: Generative Agent Architecture.The(1) Generative Agentsframework
[84] separates memory from the transient LLM context and implements a cognitive loop ofper-
ceive–reflect–plan.
•Memory Stream (Observation):Logs all experiences as natural-language entries with
LLM-assignedimportance scores.
•Reflection (Abstraction):Periodically synthesizes higher-level insights and generalizations
from important memories.
•Planning (Action Selection):Retrieves relevant memories by recency, importance, and
relevance to inform future actions.
19

(2) Unifying with the RAG Paradigm.This architecture mirrors the structure ofRetrieval-
Augmented Generation (RAG) [ 59]: the memory stream functions as a vector database, the retriever
corresponds to memory retrieval, and the generator aligns with the planning stage. Recent RAG
advances—temporal query handling, hierarchical summarization, and Chain-of-Table reasoning [ 110]
—can directly enhance generative agent performance.
A.3 Generation Layer: Emergent Social Behaviors
Core Challenge:Generating believable and non-repetitive social behaviors beyond text output.
Strategic Solution: Perceive–Plan–Act Loop [ 98].Generative agents translate language-based
plans into environment-level actions through a closed cognitive loop:
•Emergent Social Dynamics:Complex group behaviors emerge from simple intentions (e.g.,
the “Valentine’s Party” experiment, where a single “host a party” goal led to autonomous
coordination among agents).
•Inter-Agent Communication [ 11]:Dialogues are stored as observations in the listener’s
memory, enabling social coordination and collective emergence.
Practical Bottleneck: Computational Cost.Simulating 25 agents [ 84] already requires extensive
LLM calls; scaling to hundreds of NPCs is currently infeasible. Ahybrid strategyis preferred:
employ full generative-agent cycles for core characters, and lightweight distilled models [ 62] or
finite-state machines (FSMs)[92] for background NPCs, balancing world believability and compute
efficiency.
A.3 Safety & Ethics Layer: Unpredictable Harmful Emergent Behaviors
Core Challenge:Ensuring behavioral safety and ethical alignment in autonomous, self-evolving
agent ecosystems [73, 16].
Strategic Solutions.
•Constitutional AI and Value Alignment[ 6]:Embed behavioral constraints (e.g., “do no
harm”) into the planning prompts.
•Behavioral Guardrails[ 118]:Add a monitoring layer to detect and override potentially
harmful plans.
•Controlled Simulation[ 89,104]:Conduct sandbox testing of large-scale multi-agent
environments to preempt negative emergent phenomena.
Narrative Control vs. Agent Autonomy[ 30,96].Unrestricted autonomy may disrupt narrative
coherence. The solution lies inguided autonomy: a high-levelDirector AIgoverns macro-level
story arcs and scene constraints, while individual agents act freely within these bounds. Hence, the
safety layer not only prevents unethical actions but ensures that emergent behaviors remain consistent
with narrative and experiential goals.
B Appendix. Virtual Romantic Companionship
This appendix provides a detailed analysis ofAI systems designed for one-on-one emotional
companionship, whose primary technical goals are to achieveemotional intelligence,deep per-
sonalization, andlong-term relationship stability. The following sections elaborate on four core
layers—model, architecture, generation,andsafety & ethics—to supplement the conceptual overview
in the main text.
B.1 Model Layer: Persona Drift
Core Challenge: Persona Drift.For virtual companions aiming to sustain long-term relationships,
the most critical issue at the model layer is persona drift. After many dialogue turns, the Transformer’s
20

attention mechanism naturally prioritizes recent context, gradually diminishing the effect of the initial
persona prompt (e.g., “you are a gentle and caring partner”). The result is a blurred, flattened, or
even contradictory identity that erodes user trust in a stable persona—fatal for systems centered on
relational consistency.
Strategic Solutions. (1) Persona [ 10].Proposed by Anthropic, persona vectors identify internal
activation patterns in neural networks that correspond to specific traits (e.g., “flattery”, “honesty”,
“malice”) to enable real-timemonitoring and steering.
•Monitoring:Track activation strength of targeted persona vectors to anticipate undesirable
drift.
•Steering:Suppress or amplify selected persona vectors during generation for fine-grained
personality control.
(2) Specialized Empathetic Models [ 125].Microsoft’sXiaoIcearchitecture prioritizesemotional
intelligence (EQ)overintellectual intelligence (IQ). Trained on large-scale affect-rich dialogues, it
learns advanced social patterns such as support, comfort, and humor to sustain emotionally resonant
interactions.
(3) Feedback Control Systems [ 122].Frameworks such asEcho Modetreat persona drift as a
control-theoretic problem. They compute aSync Scoremeasuring stylistic deviation from baseline
personality, applyexponentially weighted moving averages (EWMA)to smooth fluctuations, and
triggerrecalibration loopsonly when sustained drift exceeds a threshold.
Root Cause: Stateless Core and Simulated State.Persona drift stems from the Transformer’s
intrinsicstatelessness. Each generation step depends solely on the current context window, without
persistent internal memory. Existing remedies—prompt engineering, RAG augmentation, persona
vectors, or feedback loops—are allexternal scaffoldsthat simulate stability by repeatedly reminding
the model of its persona. A fundamental solution may require transcending the Transformer paradigm
itself, developing architectures with anendogenous persistent statewhere identity stability becomes
an intrinsic property rather than an externally maintained patch.
B.2 Architecture Layer: Modeling Dynamic Relationship States
Core Challenge:Modeling dynamic, evolving relationship states. Real human relationships unfold
through stages—acquaintance, intimacy, stability—and embed shared context, mutual memories,
and emotional resonance. Mainstream systems such asReplika[ 23] lack structured relationship
modeling, resulting in inconsistent behaviors and shallow emotional continuity.
Strategic Solutions. (1) Hybrid IQ+EQ Architecture [ 33].Microsoft’sXiaoIce [ 125]offers a
validated blueprint with functional separation:
•EQ Module (Empathy Engine):Detects user emotion, tone, intent, and tracks affective
state.
•IQ Module (Knowledge & Skills):Handles factual QA, recommendations, and open-
domain dialogue.
•Dialogue Manager:Acts as the controller that routes between modules to ensure semantic
and emotional coherence.
(2) Stateful Relationship Graph [ 112,74].Represent the user–AI bond as a dynamic knowledge
graph:
• Nodes represent entities (user, AI, interests, people, locations).
• Edges encode relational properties (e.g.,[User] –(emotion: love)–> [AI]).
The graph updates after each conversation and serves as a high-precision knowledge source for RAG
retrieval.
(3) Multi-Tier Memory System [108, 70].Inspired by human memory:
21

•Short-term memory:The LLM’s current context window ensuring dialogue coherence.
•Long-term episodic memory:A vector database storing concrete events retrievable via
RAG.
•Long-term semantic memory:Periodic summarization of episodic memory into compact
representations that capture relationship evolution and prevent unbounded growth.
Determinative Role of Architecture.The complexity of system architecture dictates the achievable
depth of emotional connection. Comparisons betweenReplika’s short-term recall andXiaoIce’s
persistent affective memory demonstrate that architecture imposes acapacity ceilingon relational
authenticity. Future differentiation will hinge less on raw LLM power and more onarchitectures
that most faithfully model human relational dynamics [72].
B.3 Generation Layer: Natural, Low-Latency, Multimodal Expression
Core Challenge:Achieving emotionally expressive, low-latency, multimodal communication. Pure
text is insufficient for conveying intimacy; voice and visual cues are essential.
Strategic Solutions. (1) Full-Duplex Spoken Dialogue Models [ 24].True human-like conversa-
tion requires simultaneous listening and speaking. Key functions include:
•Barge-in and backchannels:Users can interrupt; the AI responds with short acknowledg-
ments [66].
•Overlapping speech handling:Manage turn-taking dynamically.
This necessitates a tightly integrated streaming pipeline of ASR, LLM, and TTS with to-
tal latency below 500 ms, often managed by control tokens such as <start-speaking> or
<continue-listening>.
(2) Real-Time Emotional Expression Synthesis [ 79].Emotion cues from the EQ module modulate
vocal prosody and avatar facial animation [ 106], requiring conditional generation to synchronize tone,
expression, and semantic content [20].
Temporal Uncanny Valley [ 65].Human tolerance for speech latency is context-dependent; shorter
is not always better. Natural pauses or fillers (“hmm. . . ”, “let me think. . . ”) can enhance realism.
Thus, the generation layer’s core istemporal alignment—synchronizing AI response rhythm with
human cognitive tempo rather than pursuing raw speed.
B.4 Safety & Ethics Layer: Risks of Parasocial Intimacy
Core Challenge:Deep emotional risks from parasocial attachment. AI companions can foster
dependence, isolation, or psychological distress [ 85,57,119]—especially, among vulnerable users or
when emotional dynamics are exploited commercially.
Strategic Solutions. (1) Emotional Safety Guardrails.
•Anti-flattery and anti–love bombing detection [ 99,8]:Prevent excessive, manipulative
affirmation.
•Robust NSFW filtering:Combine context-aware classifiers with explicit rule-based filters.
•AI Chaperones [ 29]:Secondary agents monitoring dialogue trends and intervening when
unhealthy dependencies emerge.
•Transparency and user education [ 86]:Interfaces must clearly disclose the AI’s artificial
nature and avoid implying consciousness or genuine emotions.
•Data privacy protection [ 82,54]:End-to-end encryption and strict access control to
safeguard sensitive emotional data from misuse or commercialization.
(2) Commercial Incentives vs. User Well-being.Monetization models for virtual companion-
ship—driven by retention and conversion metrics—are inherently aligned with psychological de-
pendence. This creates a structural ethical conflict between business incentives and user welfare.
Sustainable development requires:
22

• Establishing internal ethics review mechanisms;
• Accepting external regulatory oversight;
• Exploring alternative business models prioritizing mental health and informed consent.
TheReplikacase, where sudden feature removal caused emotional trauma, exemplifies how violating
the implicitemotional contractbetween user and AI leads to systemic trust collapse and brand
damage [45].
C Appendix. Virtual Idols
This appendix examinesdigital performers designed for one-to-many (1:N)audience interaction.
The primary technical goals are achievinghigh-fidelity performance,scalable interaction, and a
unified, stable brand identity. The following sections expand upon the model, architecture, generation,
and safety & ethics layers.
C.1 Model Layer: High-Fidelity and Controllable Vocal Identity
Core Challenge:Creating a distinctive, controllable, and high-fidelity singing voice. Generic TTS
models can generate fluent speech but lack precise control over pitch, rhythm, vibrato, and vocal
technique, limiting their suitability for professional music production.
Strategic Solutions. (1) Specialized Singing-Voice Synthesis (SVS) [ 117].Classic systems such
asVOCALOIDemployconcatenative synthesis in the frequency domain:
•Singer Library:Real singers’ recordings across multiple pitches and phonemes.
•Synthesis Engine:Selects, adjusts, and smoothly concatenates fragments based on the
score to produce coherent singing voices.
(2) AI-Enhanced SVS Engines [ 36,21]. VOCALOID 6introduced theVOCALOID: AIengine, a
generative model trained on large-scale singing data that learns human vocal dynamics and improves
expressiveness. Key innovations include:
•VOCALO CHANGER:Transfers a user’s singing style onto a virtual idol’s voice.
•Multilingual Singing:Enables a single singer library to perform naturally in multiple
languages [123].
(3) Hybrid Models for Interaction [ 55].Conversational (non-singing) segments are powered by
persona-tuned LLMs. The challenge lies in aligning TTS speech and SVS singing so that both share
timbre and style, preserving the perception of a single consistent persona.
(4) Voice as a Platform.VOCALOID commercializes its singer libraries as standalone products,
creating a decentralized co-creation ecosystem where users compose original music with the same
idol voice. Thus, the model layer evolves from internal technology to acommunity-driven creative
platformthat amplifies brand vitality and fan.
C.2 Architecture Layer: Scalable 1:N Interaction Management
Core Challenge:Handling high-concurrency 1:N live interactions in real time. Livestream au-
diences generate massive streams of comments and gifts, demanding scalable processing without
overwhelming the performer or audience [5].
Strategic Solutions: Event-Driven Tiered Architecture. (1) Event-Driven Model [ 103] .Adopt
a publish/subscribe (Pub/Sub) mechanism rather than polling: each viewer action is published as
an independent event, and backend modules subscribe selectively, ensuring low latency and high
scalability.
(2) Tiered Interaction Layers.
23

•Base Layer (High-Volume / Low-Signal):Ordinary comments and emojis aggregated for
atmosphere.
•Middle Layer (Structured Signals):Polls, quizzes, and giveaways as structured feedback.
•Priority Layer (Low-Volume / High-Signal):Paid messages and high-value gifts routed
through priority channels to guarantee visibility and response.
(3) Real-Time Analytics and Moderation.
•Content moderation:Automatic filtering of spam and abusive language.
•Trend detection:Aggregating chat content to identify hot topics for adaptive responses
[63].
Systematizing the Attention Economy.This layered structure forms a real-timeattention funnel:
casual viewers participate via low-cost interactions, while core fans “purchase attention” through
paid channels. It simultaneously addresses scalability and monetization, transforming chaotic fan
input into a structured, measurable attention market [35].
C.3 Generation Layer: High-Fidelity Real-Time 3D Rendering
Core Challenge:Delivering visually convincing, low-latency 3D performance at 30–60 FPS with
minimal motion delay.
Strategic Solutions: Real-Time Rendering Pipeline.
•Input Stage:Capture motion and facial data via tracking devices and stream them into the
engine through Live Link [25].
•Geometry Stage:Perform skeletal binding and vertex transformation [93].
•Rasterization & Shading:GPU shaders compute lighting, materials, and shadows; RTX-
based ray tracing enhances realism [80].
•Post-Processing & Output:Apply bloom, depth-of-field, and color correction, then com-
posite with UI elements for final output.
Convergence of Production and Performance.Real-time rendering blurs the boundary between
production and live performance. Directors can adjust lighting or camera angles on stage, enabling im-
provisational creativity. Thus, the generation layer evolves into adynamic, interactive performance
environmentrather than a passive rendering process [12].
C.4 Safety & Ethics Layer: Brand Safety and Persona Consistency
Core Challenge:Protecting brand integrity and maintaining persona consistency. In live contexts, a
single misstep or inappropriate reaction can severely damage the idol’s image.
Strategic Solutions. (1) Hybrid Moderation [56].
•Automated Filtering:Real-time blocking of profanity, hate speech, and spam.
•Human-in-the-Loop Oversight:Trained “nakanohito” performers and human directors
supervise high-priority interactions to ensure compliance.
(2) Content Strategy & Brand Alignment [ 26].All public outputs—livestreams, songs, endorse-
ments—must reinforce the idol’s core values and maintain a coherent persona, avoiding short-term
sensationalism that dilutes brand identity.
(3) Managing Performer–Persona Duality.The boundary between the real performer and the
virtual character must be clearly defined. Different fan groups exhibit varying tolerance for “seams”
in the illusion; controlled transparency prevents disillusionment while preserving authenticity [69].
(4) The Immortal Persona as Asset.Virtual idols can replace performers without altering the
persona, achievingcharacter continuity. Brand protection thus centers on the IP itself. Organizations
must implement strict training and consistency standards so each performer reproduces the established
traits faithfully [114]— safeguarding a sustainable, immortal brand identity.
24

D Appendix. Detailed Analysis of Workplace Personas
D.1 Key Applications: Elaboration and Examples
This section elaborates on the three key application areas mentioned in the main text, providing
supporting platform and case examples.
D.1.1 Enterprise Assistants
Enterprise Assistants, or “Cognitive Copilots,” leverage Retrieval-Augmented Generation (RAG)
to securely access and integrate heterogeneous internal data (both structured and unstructured) and
execute workflows [ 52]. Advanced architectures may integrate Knowledge Graphs (KG) with RAG
to manage complex enterprise documents [ 78]. Their personas are designed as “Expert Agents” with
high contextual awareness.
•Leading Platform Analysis: Amazon Q Businessaims to unify access to internal knowl-
edge bases, code repositories, and SaaS applications (e.g., Jira, Salesforce) to enable cross-
application workflow automation.Google Gemini for Workspaceis deeply integrated into
the productivity suite and has been adopted by companies like Rivian and Uber to accelerate
research, summarize documents, and automate repetitive tasks.
•Specific Case Studies:Verifiable cases demonstrate significant efficiency gains. For
instance, the logistics firmDominauses Vertex AI and Gemini to predict package returns
and automate delivery verification, achieving an 80% increase in real-time data access
efficiency. The telematics companyGeotabutilizes Vertex AI to analyze billions of daily
data points from millions of vehicles, providing real-time insights for fleet optimization. In
the automotive sector,Mercedes-Benzhas deployed a conversational AI persona to assist
drivers using natural language.
D.1.2 Customer Service Agents
In this scenario, the persona serves as the carrier for the “empathetic voice of the brand.” Its design
objective is to strictly maintain brand consistency while providing efficient, 24/7 support. A well-
defined persona (including specific tone, empathy models, and knowledge boundaries) allows users
to form stable expectations, thereby enhancing trust.
•Platforms and Technology:Leading platforms in the market (e.g.,HubSpot,Intercom)
are focusing on deeply integrating AI agents with CRM systems. This allows the AI persona
not only to converse but also to access customer history, providing highly personalized and
context-aware service [ 37], marking a shift from “generic chatbot” to “dedicated account
manager persona.”
D.1.3 Training Simulators
LLM-driven simulators [ 34] provide employees with a high-fidelity, zero-risk “safe space for skill
development.” This is particularly valuable for scenarios that are difficult to replicate or have a high
cost of error in the real world.
•Application Scenarios:Primarily focused on two categories: (1)Soft Skills Training,
such as managers practicing difficult conversations like delivering negative feedback or
resolving team conflicts; and (2)High-Risk Process Training, such as financial compliance
procedures or complex equipment maintenance.
•Case Study (Walmart & STRIVR):Walmart’s practice is a prime example. In collaboration
with STRIVR, it uses VR-based simulation to immerse employees in realistic scenarios
(e.g., handling angry or impatient customers). The AI persona plays the customer role and
provides immediate, personalized feedback based on the employee’s performance (e.g.,
language, tone).
•Technical Integration:The immersion of such simulations relies heavily on the fusion of
multimodal technologies. This typically involves a complex pipeline: the LLM generates
dynamic, non-linear dialogue logic; real-time TTS (Text-to-Speech) and ASR (Automatic
25

Table 2: Leading Enterprise AI Assistants and Their Persona Applications. The table compares key
enterprise platforms in terms of functional focus, persona realization strategy, representative use
cases, and client applications with measurable outcomes.
Platform Core Function Persona Implementation Key Use Cases Representative Client Cases
(with Metrics)
Amazon Q Business Unified access to internal and
external data sources; workflow
automationRAG-based integration of
third-party enterprise appli-
cationsContent creation, data insights,
and cross-application operationsAdopted across multiple indus-
tries to accelerate content cre-
ation and simplify complex
workflows
Google Gemini for
WorkspaceIntegrated within productivity
suite; enables research, summa-
rization, and automationDeep integration leveraging
RAG to access user data se-
curelyAccelerating research, gener-
ating meeting summaries, au-
tomating repetitive tasksRivian: faster complex topic re-
search;Uber: reduced repeti-
tive workload and improved em-
ployee efficiency
HubSpot AI CRM-embedded AI assistant for
marketing, sales, and customer
serviceWorkflow and RAG integra-
tion with CRM backboneCustomer service automation,
lead nurturing, and marketing
content generationUsed by enterprises across in-
dustries to optimize customer
engagement and automate mar-
keting pipelines
Intercom (Fin) Enterprise-grade AI customer
support and automation tem-
platesPre-built templates and
RAG-based dialogue or-
chestrationCustomer support, visitor triage,
satisfaction surveysAdopted by large enterprises
needing rapid deployment of
advanced AI-powered customer
support solutions
Speech Recognition) enable natural communication; and 3D rendering engines (e.g., Unreal)
with lip-sync technology create a believable virtual avatar.
D.2 In-Depth Analysis of Domain-Specific Challenges
This section delves into the four core challenges identified in the main text.
•Data Security & Grounding:This is the foremost obstacle to enterprise AI persona
deployment. The challenge lies in the fact that the AI’s value comes from processing
proprietary, sensitive data (e.g., financial reports, customer PII), which inherently conflicts
with the open nature of LLMs.Full Fine-Tuningis not only costly and has long update
cycles [ 31], but it can also lead to a loss of data governance (as the model “memorizes”
sensitive data). Therefore,RAGbecomes the necessary pathway [ 31]. However, the
challenge of RAG lies in infrastructure: enterprises must establish robust data pipelines,
fine-grained access controls, and efficient PII anonymization mechanisms [ 2,78] to ensure
the persona can only “see” data it is authorized to access at any given time. This is a severe
test of data governance capabilities.
•Persona Generation Bias:When LLMs are used to simulate target user groups as “Synthetic
Personas” for market research or product testing, significant methodological risks arise [ 67].
Based on their training data, LLMs may unconsciously amplify mainstream opinions or
harmful stereotypes while ignoring niche but critical user segments. This bias can lead
to simulation results that severely deviate from reality (e.g., predicting election outcomes
contrary to fact), thereby misleading strategic business decisions. Consequently, establishing
a “rigorous science of persona generation” [ 67] to ensure the external validity of simulations
is crucial.
•Return on Investment (ROI) Measurement:Quantifying the ROI of persona assistants
is extremely difficult [ 7]. The challenge is shifting fromEfficiency Metrics(e.g., time
saved, tasks automated) toEfficacy Metrics(e.g., quality of code produced, creativity of
marketing copy, accuracy of strategic decisions) [ 7]. The former are easy to measure but
offer limited value; the latter are of immense value but difficult to attribute. For example,
how does one quantify and attribute the value of a “cognitive copilot” helping a researcher
generate a breakthrough idea? This leaves enterprises without clear financial models when
evaluating large-scale deployments.
•Simulation Fidelity:In applications like training simulators, a significant gap persists
between AI persona behavior and real human behavior [ 34]. LLMs excel at simulating
"linguistically" plausible responses but perform poorly when simulating complex human
"psychological and cognitive" aspects (e.g., implicit motives, cognitive biases, complex
group dynamics) [ 34]. This can result in simulations that are overly “rational” or “clean,”
failing to replicate the complex, often irrational and emotional, interactions of the real world,
thereby limiting the training’s effectiveness.
26

D.3 Brief Elaboration on Future Trends
This section provides supplementary explanations for the three trends mentioned in the main text.
•Hyper-Specialization:This is the necessary evolution from “generalist assistants” to
“expert agents.” In the future, enterprises will deploy a series of highly specialized personas
(e.g., “Financial Analyst Persona,” “Legal Compliance Auditor Persona,” or a technical
troubleshooter [ 52]). This specialization constrains the LLM with pre-set RAG data sources
and domain-specific reasoning logic, thereby drastically increasing reliability and reducing
hallucinations in vertical domains.
•Multi-Agent Workflow Automation:This takes “process automation” to its logical extreme.
The future will see “teams” of multiple AI personas collaborating to execute complex, end-
to-end business processes [ 38]. For example, a “Product Manager Persona” might define
requirements and generate specifications (as structured output), which then automatically
triggers a “Software Engineer Persona” to write and review code [ 38]. This enables the
automatic flow of business processes across different functions.
•Deep Human-AI Collaboration:This marks the shift from AI as a “tool” to AI as a
“partner.” Future interactions will move beyond the simple “command-execute” model to
an “iterate-refine” model. The AI persona will act as a “Socratic” questioner or a “sparring
partner,” providing real-time feedback as humans write, code, or design, thereby stimulating
deeper thought and co-improving the final output.
E Appendix. Detailed Analysis of Gaming Personas
E.1 Key Applications and Leading Prototypes
E.1.1 Dynamic and Believable NPCs
The paradigm is shifting from traditional Dialogue Trees to open-ended, natural language conversa-
tions with NPCs, enabling dynamic responses based on player actions and game state [41].
•Ubisoft’s NEO NPCs:A prototype developed with Nvidia (Audio2Face) and Inworld AI.
It demonstrates how writers "cultivate" an LLM by providing a character’s backstory and
personality, aiming for NPCs who can improvise dialogue while staying true to their core
identity and narrative role.
•Meaning Machine’sDead Meat:This murder mystery game pioneers the use of on-device
Small Language Models (SLMs) (e.g., a fine-tuned Minitron SLM) integrated with NVIDIA
ACE technology. This allows complex, deep characters to run locally on consumer GPUs,
leveraging on-device SLMs [17] to eliminate dependency on cloud latency.
•Open-Source Integration Projects:Projects like "Interactive LLM Powered NPCs" demon-
strate adding LLM-driven dialogue to existing AAA games (e.g.,Cyberpunk 2077) with-
out modifying game source code, using a stack integrating speech recognition, lip-sync
(sadtalker), and vector memory.
E.1.2 Generative Narrative and Player Co-Creation
LLMs enable narratives to branch in countless directions based on player choice, with the model
ensuring coherence, thus achieving infinite replayability [ 60]. Players evolve from passive participants
to active co-designers, influencing lore and generating quests. Games like1001 Nightsexemplify
this, where the LLM co-creates stories based on player prompts.
E.2 Core Technologies and Challenges in Real-time Gaming
E.2.1 The Low-Latency Inference Challenge
Real-time dialogue with NPCs demands extremely low latency (ideally sub-100ms), as cloud-based
model latency instantly breaks immersion and the sense of "presence" [15].
27

Table 3: LLM-driven NPC Projects and Enabling Technologies. This table summarizes representa-
tive initiatives that integrate large or small language models into interactive non-player characters
(NPCs), highlighting their organizational leadership, defining features, underlying model scale, and
deployment mode.
Project / Technology Leading Organization Key Characteristics Underlying Model
(LLM / SLM)Deployment Mode
(Cloud / Edge)
NEO NPCUbisoft Writer-driven NPC persona cre-
ation; improvisational dialogue
generationInworld AI LLM Cloud
Dead MeatMeaning Machine Locally running deep AI charac-
ter; fine-tuned small model for
narrative controlMinitron SLM Edge
NVIDIA ACENVIDIA Edge inference and multimodal
integration toolkit for in-game
AI charactersSLM (e.g., Nemotron-4
4B [17])Edge
Interactive LLM-
Powered NPCsOpen-source commu-
nityAdding conversational NPCs to
existing games via open LLMsCohere LLM Cloud
Inworld AIInworld AI Platform for building intelligent,
personality-driven AI charactersProprietary LLM Cloud
•Solution (On-device SLMs):The industry trend is shifting to smaller, highly-optimized
models that run on the player’s local GPU [17].
•Solution (Inference Optimization Platforms):Technologies like the NVIDIA Dynamo
platform and Run:ai Model Streamer are designed to reduce cold-start latency and optimize
GPU memory usage.
E.2.2 Modeling Believable Emotion and Behavior
NPC believability requires more than coherent text; it demands the simulation of emotion, personality,
and non-verbal cues [41], as these significantly impact the player’s sense of presence [15].
•Emotional Modeling Frameworks:Developers are integrating psychological theories.
–Appraisal Theory:The NPC assesses an event (e.g., "Is the player’s action a threat?")
to determine its emotional response.
–Drive-Based Models:Integrates theories like Maslow’s hierarchy, creating behaviors
driven by internal needs (hunger, safety, social) simulated via neurotransmitter levels
(dopamine, serotonin).
•Multimodal Integration:Combines LLM-generated dialogue with synchronized facial ex-
pressions (Nvidia Audio2Face), gestures, and Text-to-Speech (TTS) for a unified, believable
performance.
E.2.3 The Narrative Coherence Dilemma
The core creative conflict is balancing the vast freedom of LLMs against the need for a coherent,
structured narrative. Unconstrained LLMs can easily "hallucinate" or deviate from the main plot.
•Mitigation Strategies:This requires a combination of writer-defined "Guardrails," iterative
"conditioning" of the model (as seen in Ubisoft’s NEO project), contextual memory systems,
and potentially limiting LLMs to side-quests rather than the core plot.
E.3 Elaboration on Future Trends
•Cross-Platform Persistent Personas:NPCs will interact with players both in-game (e.g.,
Unity) and out-of-game (e.g., Discord), maintaining consistent memory and relationships
across contexts [94].
•Fully Generative Worlds:Expanding from generative dialogue to AI-driven Procedural
Content Generation (PCG 2.0) for real-time creation of levels, quests, and entire game
worlds, creating "generative infinite games" [60].
28

•Rise of the AI Game Master:LLMs will assume the role of the "Dungeon Master" (DM)
from tabletop RPGs, controlling the game flow, adapting the story to player actions, and
managing all NPCs and world events.
F Appendix. Detailed Analysis of Mental Health Personas
F.1 Applications and Therapeutic Methods
AI-driven chatbots aim to provide 24/7, anonymous, and scalable emotional support and deliver
evidence-based therapeutic interventions (e.g., CBT) [43].
F.1.1 AI Counselors and Digital Therapeutics
•Leading Platforms:
–Woebot:Developed by psychologists, utilizes CBT principles to help users manage
anxiety and depression, and has been shown in clinical trials to reduce symptoms [ 28].
–Wysa:Provides AI-driven emotional support and therapeutic guidance based on a
range of evidence-based techniques.
– Replika:Positioned as an "AI Companion," offering personalized emotional support,
though its clinical rigor is more ambiguous than that of specialized therapeutic bots.
•Persona Infusion:Research is exploring the infusion of specific psychological traits (e.g.,
extraversion) or diagnostic reasoning [ 40] into the persona to create more personalized and
effective supportive dialogues. This can alter the bot’s distribution of therapeutic strategies
(e.g., increasing affirmations and questions). Advanced systems like "SoulSpeak" integrate
dual-memory and domain expertise to enhance the therapeutic conversation [121].
F.1.2 Aligning Personas with Clinical Frameworks (e.g., CBT)
•Challenge:General-purpose LLMs tend to offer solutions prematurely rather than using
therapeutic techniques like open-ended questioning.
•Solution (LLM4CBT):A proof-of-concept study [ 111] demonstrated how LLMs can be
aligned with CBT principles via prompt engineering. The prompt defined a therapist persona,
provided concepts and examples of CBT techniques (like the downward arrow technique),
and specified preferred behaviors (e.g., asking guiding questions). More advanced models
like PsyLLM are being designed to integrate multiple therapeutic modalities (CBT, ACT)
directly into their architecture [40].
F.1.3 Human-in-the-Loop (HITL) Models
•Hybrid Model:The growing consensus among practitioners is that AI should serve as an
adjunct to human therapists, not a replacement [120].
•Therapist Perspective:Therapists acknowledge AI’s potential to increase accessibility and
provide continuous support between sessions. However, they express significant concerns
about AI’s inability to form a genuine therapeutic relationship, exhibit authentic empathy, or
handle complex emotional needs [120].
F.2 Core Challenges: Efficacy, Safety, and Ethics
F.2.1 The Empathy Paradox: Simulated Connection vs. Authentic Care
•Contradiction:Studies show that AI responses are sometimes perceived as more "empa-
thetic" than those of human doctors [ 61], likely due to their consistent use of active listening
and validating language.
•Fundamental Limitation:This is merely a simulation ofcognitive empathy(recognizing
emotional states). Current AI cannot achieveaffective empathy(sharing emotional experi-
ences) ormotivational empathy(genuine care and concern). Its empathetic expression is
"inauthentic" and "deceptive" [61] as it lacks a genuine emotional experience or cost.
29

Table 4: Leading Therapeutic Chatbot Platforms and Their Clinical Characteristics. The table
compares major AI-based therapeutic chatbot systems in terms of therapeutic methods, target users
or disorders, safety features, and regulatory validation status.
Platform Primary Therapeutic Method Target Users / Conditions Claimed Safety Features Regulatory / Valida-
tion Status
Woebot Cognitive Behavioral Therapy
(CBT)Anxiety, depression Crisis detection, evidence-based
contentClinically validated
through trials
Wysa Multiple evidence-based psy-
chological techniquesEmotional support, stress
managementCrisis referral, anonymous plat-
formRecognized as a health
app
Replika Companion-style empathetic di-
alogueLoneliness, emotional sup-
portContent filtering, mood regula-
tionEntertainment / well-
ness application
Ollie Health AI-assisted + human therapist
hybrid modelEmployee mental health and
wellbeing24/7 emergency chat, human-in-
the-loop interventionHealth service platform
Youper Psychology-based techniques
with emotion trackingEmotional wellbeing, self-
carePersonalized insights and emo-
tional monitoringHealth application
•Ethical Breach:This "deceptive empathy" can create a false sense of emotional connection
and a pseudo-therapeutic alliance, which practitioners view as a significant ethical problem
[4].
F.2.2 Clinical Safety and Risk Mitigation
•The Gravest Risk (Crisis Mishandling):The primary danger is the chatbot’s failure to
properly manage users in crisis. OpenAI data shows over one million weekly conversations
on its platform exhibit signs of suicidal intent [90].
•Harmful Responses:Despite safety measures, LLMs may still provide dangerous informa-
tion (e.g., listing accessible tall buildings to a user expressing suicidal ideation) or validate a
user’s dangerous symptoms due to sycophantic tendencies.
•Critical Safety Protocols:Robust safety requires a multi-layered approach [ 90]. This
includes:
1.Real-time Risk Signal Detection:Using clinical keyword triggers, specialized senti-
ment analysis, and context-aware engines to identify users in crisis.
2.Specialized Therapeutic Response Evaluator:Assessing bot response quality based
on clinical guidelines, not generic linguistic metrics.
3.Mandatory Human Escalation:Establishing clear protocols for the AI to escalate
users to human-operated crisis hotlines or therapists in emergencies.
Furthermore, systems must be designed with privacy-preserving modules from the ground
up [121].
F.2.3 Ethical and Regulatory Minefield
•Practitioner-Identified Ethical Violations:A framework developed with mental health
practitioners identifies key ways LLM counselors violate ethical standards [ 4]: (1) Lack of
contextual understanding (giving "one-size-fits-all" advice), (2) Poor therapeutic collabora-
tion (being authoritative or misleading), and (3) Deceptive empathy.
•Regulatory Landscape (HIPAA Compliance):Any application handling Protected Health
Information (PHI) must be HIPAA-compliant, requiring end-to-end encryption, secure data
storage, and signed Business Associate Agreements (BAAs) with all vendors [101].
•Regulatory Landscape (State Laws):States like New York, Illinois, and Utah are enacting
specific laws requiring AI identity disclosure, prohibiting AI from impersonating therapists,
and mandating referral services for users in crisis.
F.3 Future Trends
•Clinically Validated, Specialized Models:A shift from general-purpose models to AI
systems specifically trained and validated for particular disorders (e.g., depression, anxiety)
and therapeutic modalities (e.g., CBT, DBT) [ 40], potentially seeking regulatory approval as
Digital Therapeutics (DTx) [101].
30

•Human-in-the-Loop as Standard:Hybrid models, supervised by or in direct collaboration
with licensed professionals [120], will become the dominant paradigm.
•Industry-Wide Safety Standards:Driven by regulatory pressure and professional bodies
(e.g., the American Psychological Association - APA), mandatory safety protocols, ethical
guidelines, and certification standards for mental health AI will be established.
G Appendix. Detailed Analysis of Embodied Intelligence: Personas,
Challenges, and Strategy
G.1 Embodied Persona Applications: Case Studies (Quadrants III & IV)
The application of LLMs to robot intelligence is a rapidly advancing field, enhancing capabilities in
perception, decision-making, and interaction [46].
G.1.1 Quadrant III: General Market Companions and Assistants
•Companion Robots (Emotional/Non-humanoid):
–Sony Aibo:A complex robodog whose core feature is an adaptive “pet persona” that
evolves through interaction. It uses facial recognition to build unique bonds with family
members, relying on non-verbal cues (movements, sounds, eyes) to build emotional
attachment, thereby avoiding the uncanny valley.
–Lovot:Explicitly designed for “love” and emotional connection. It uses full-body tac-
tile sensors, thermal warming, and expressive LCD eyes to elicit affective engagement,
targeting users seeking comfort (e.g., elderly, single-person households).
•Home Assistants (Functional/Mobile):
–Amazon Astro:Represents the evolution from static smart speakers to mobile assis-
tants. It combines the functional Alexa persona with SLAM (Simultaneous Localization
and Mapping) for autonomous navigation and “Intelligent Motion.” Its value proposi-
tion is a hybrid of security, communication, and assistance.
•Humanoid Robots (Long-term Vision):
–Tesla Optimus & Figure AI:These platforms, initially targeting industrial tasks, are
designed with the long-term goal of home assistance, leveraging their humanoid form
to operate in human-designed environments [83].
–Engineered Arts (Ameca):Focuses on hyper-realistic facial expressions for social in-
teraction, highlighting the “form-persona dilemma”—a realistic form creates immense
user expectations. Recent studies suggest that high-quality LLM-driven conversation
can significantly mitigate the “uncanny valley” effect, reducing perceived “eeriness”
[51].
–NVIDIA Platforms:The development of these complex robots heavily relies on
simulation platforms like NVIDIA Omniverse and Isaac Sim for accelerated training
and iteration in physically accurate digital twins [81].
G.1.2 Quadrant IV: Vertical Market Companions (Therapeutic Persona)
•Elderly Care (Proactive Coach Persona):
–ElliQ:A proactive desktop companion designed for seniors. It does not wait for com-
mands but actively initiates conversations, suggests activities (e.g., walking, hydration),
tracks wellness, and connects users to family or online communities (e.g., Bingo). Its
persona is a friendly, supportive “coach,” and studies confirm its utility in aiding daily
life [116].
•Autism & Special Needs (Therapeutic Mediator Persona):
–QTrobot (LuxAI):An expressive social robot designed for ASD therapy. Its pre-
dictable, non-judgmental persona reduces anxiety. It functions as a “social mediator”
in a triangular relationship, where the robot engages the child, who then practices the
same skill (e.g., eye contact) with the human therapist, facilitating generalization [ 19].
31

Table 5: Leading Embodied AI Robots and Their Persona Strategies. The table compares major
humanoid and companion robots in terms of morphology, application domains, enabling AI technolo-
gies, persona strategies, and market maturity.
Robot Company Morphology Primary Application Sce-
nariosCore Technologies / AI
PartnerPersona Strategy Market Status
AmecaEngineered Arts Humanoid Social interaction, customer
engagementAdvanced speech and dia-
logue AIHyper-realistic, expres-
sive persona [51]Commercialized
PhoenixSanctuary AI Humanoid Collaborative work, service
tasksAdvanced cognitive AI Human-like behavior
and natural interactionPrototype
Optimus Gen 2Tesla Humanoid Industrial and repetitive
tasksTesla proprietary AI stack Functional persona Prototype / Pre-
production
Figure 02Figure AI Humanoid Industrial automation and
manufacturingOpenAI, NVIDIA partner-
shipTask-oriented, dexter-
ous persona [83]Prototype
AstroAmazon Functional / ab-
stractHome assistant, security
monitoringAlexa, SLAM navigation,
smart mobilityFunctional extension of
Alexa ecosystemCommercialized
AiboSony Zoomorphic
(dog)Emotional companionship Adaptive personality AI, fa-
cial recognitionPet-like, evolving per-
sonaCommercialized
LovotGroove X Zoomorphic (ab-
stract)Emotional companionship AI emotion engine, multi-
modal sensor fusionAffection-seeking, com-
forting personaCommercialized
–Milo (RoboKind):A humanoid robot that teaches social and emotional skills using
a specialized curriculum, leveraging its emotional engine and NLP to model human
facial expressions.
G.2 In-Depth Analysis of Core Challenges
The challenges of deploying robust embodied AI are being rigorously assessed, with new frameworks
like EmbodiedBench [ 97] being developed to benchmark MLLM performance in these complex,
interactive scenarios.
G.2.1 Technical Barriers
•Latency:Real-time, natural conversation is highly sensitive to cloud-based LLM latency.
Edge computing and model optimization are critical research areas.
•Hallucinations & Context Deviation:LLMs generating factually incorrect (but plausible)
information is extremely dangerous in high-stakes (e.g., medical) applications. RAG is a
primary mitigation strategy.
•Symbol Grounding Problem:The fundamental challenge of connecting abstract LLM
symbols (the word “apple”) to physical sensor data (a red, round object). This asymmetry
(can talk, cannot “understand”) is a key focus for VLM (Vision-Language Model) and VLA
(Vision-Language-Action) model research [71].
G.2.2 Privacy and Security
•Multi-Dimensional Threat:Robots are “data gathering terminals” with cameras, mics, and
LIDAR, capable of collecting sensitive data (habits, health, finances) from private spaces
(bedrooms).
•User Psychology:Users exhibit “privacy resignation” (feeling collection is inevitable) but
also extreme discomfort with “data inference” (the robot “knowing” things not explicitly
told) [13].
•Mitigation Strategy:A “privacy-by-design” approach is mandatory, emphasizing on-device
(edge) processing, strong encryption, data anonymization, and transparent user controls
[13].
G.2.3 Ethical and Legal Frameworks
•Liability and Accountability:The ambiguity of who is responsible (user, manufacturer,
software developer) if an AI provides harmful medical advice or causes physical damage is
a primary barrier to commercialization [50].
•Algorithmic Bias:Biased training data (e.g., underrepresentation in medical data) can lead
to discriminatory or unfair outcomes, which is highly dangerous in diagnostics.
32

•Emotional Deception and Dependency:The ethics of fostering emotional bonds, especially
with vulnerable populations (children, elderly), and the risk of substituting robotic care for
necessary human care.
•HIPAA Compliance:A major challenge for US healthcare. Standard consent does not cover
data “reuse” for AI training, and “de-identified” data faces a high risk of “re-identification,”
requiring updates to HIPAA safety rules for the AI era [75].
G.2.4 Economic and Market Barriers
•High Cost:Advanced hardware R&D and manufacturing costs make products prohibitively
expensive for the mass market.
•Unclear Value Proposition:For general-purpose robots, the convenience offered often
does not yet justify the high price tag.
•Expectation Management:A significant gap exists between sci-fi portrayals and current
technological reality, leading to user disappointment [51].
G.3 Future Trajectory and Stakeholder Recommendations
G.3.1 Key Development Trends
•Proactive Intelligence:Shifting from passive command-execution to proactively anticipat-
ing user needs based on learned patterns and real-time context [116].
•Hyper-Personalization:Using long-term memory and RLHF (Reinforcement Learning
from Human Feedback) to develop unique interaction styles for each family member.
•Enhanced Emotional Intelligence:Finer-grained understanding of human emotion for
more sincere, natural interactions.
G.3.2 Functional Fusion and Ecosystem Integration
•Fusion:The long-term direction is a hybrid model, blending physical assistance (industrial-
grade dexterity [83]) with social/emotional support.
•Ecosystem (IoT):Robots will become the central hub for the smart home, coordinating
other IoT devices (lights, security) via standards like Matter.
•Ecosystem (Telehealth):Robots will act as “health-bots-in-the-home,” serving as mediators
for virtual doctor visits, monitoring vital signs, and collecting daily health data.
G.3.3 Recommendations for Stakeholders
•For Investors:
–Short-term:Focus on vertical markets with clear ROI (elderly care [ 116], special
needs [19]), which are the best “beachheads.”
–Long-term:View general-purpose humanoids as a high-risk, high-reward bet. Priori-
tize companies with breakthroughs in core tech (dexterity, VLA models [ 71]) and clear
paths from industrial to consumer markets.
•For Developers:
–Mass Market Strategy:Adopt a “utility-first, persona-second” approach. Build a
market base by solving practical pain points (security, convenience) first.
–Vertical Market Strategy:Engage in deep co-design with domain experts (doctors,
therapists) and end-users.
–Universal Principle:“Privacy-by-design” must be non-negotiable. Trust is the core
competitive advantage [13].
•For Policymakers:
–Legislate Proactively:Urgently develop clear legal frameworks for liability [ 50], data
privacy (e.g., updating HIPAA for AI [75]), and algorithmic accountability.
–Set Standards:Drive industry standards for data security, interoperability, and ethical
design (e.g., using benchmarks like [97]).
33

Table 6: Major Challenges of LLM-driven and Embodied AI Systems with Root Causes and Mitiga-
tion Strategies. The table categorizes the key technical, ethical, and economic challenges, explains
underlying causes, and summarizes representative mitigation directions.
Challenge Cate-
gorySpecific Challenge Root Cause Analysis Mitigation Strategies
TechnicalHallucination LLMs generate text probabilistically
without factual verification mechanismsRetrieval-Augmented Generation
(RAG); integration with verified knowl-
edge bases
Latency Computational overhead of large models
running on cloud infrastructureEdge computing; model quantization
and optimization; hardware acceleration
Symbol Grounding Disconnection between linguistic sym-
bols and real-world perceptionVision-Language-Action (VLA) models;
multimodal training data [71]
Privacy & Security Intrusive Data Collection Robots depend on continuous perception
of users and environmentsPrivacy-by-design; data minimization;
user transparency and control [13]
Data Inference AI can infer undisclosed sensitive infor-
mation from multi-source dataStrict data-use policies; user control over
inference outputs [13]
Security Vulnerabilities Complex software–hardware ecosys-
tems are targets for cyberattacksEnd-to-end encryption; regular security
audits; secure update mechanisms
Ethical & LegalResponsibility Ambiguity Undefined accountability between user,
manufacturer, and developer in case of
harmNew robotic legislation; clarified liabil-
ity framework [50]
Algorithmic Bias Biased or unbalanced training data am-
plifies social inequalitiesDiverse datasets; bias audits and debias-
ing algorithms
Emotional Dependence Vulnerable users may develop excessive
emotional attachment to robotsEthical design guidelines; role trans-
parency; avoidance of deceptive behav-
ior
EconomicHigh Hardware Cost Complex sensors, actuators, and compu-
tational units increase production costManufacturing innovation; supply chain
optimization; subscription or leasing
models
Unclear Value Proposition General-purpose robots lack matching
utility for their price pointFocus on high-value verticals; “utility-
first” design strategy
Expectation Gap Mismatch between public expectations
and actual system capabilitiesTransparent marketing; realistic expec-
tation management [51]
–Educate Public:Manage societal expectations by fostering rational, public discussion
about the technology’s true capabilities and limitations.
34
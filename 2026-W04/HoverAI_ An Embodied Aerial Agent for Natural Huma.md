# HoverAI: An Embodied Aerial Agent for Natural Human-Drone Interaction

**Authors**: Yuhua Jin, Nikita Kuzmin, Georgii Demianchuk, Mariya Lezina, Fawad Mehboob, Issatay Tokmurziyev, Miguel Altamirano Cabrera, Muhammad Ahsan Mustafa, Dzmitry Tsetserukou

**Published**: 2026-01-20 09:59:49

**PDF URL**: [https://arxiv.org/pdf/2601.13801v1](https://arxiv.org/pdf/2601.13801v1)

## Abstract
Drones operating in human-occupied spaces suffer from insufficient communication mechanisms that create uncertainty about their intentions. We present HoverAI, an embodied aerial agent that integrates drone mobility, infrastructure-independent visual projection, and real-time conversational AI into a unified platform. Equipped with a MEMS laser projector, onboard semi-rigid screen, and RGB camera, HoverAI perceives users through vision and voice, responding via lip-synced avatars that adapt appearance to user demographics. The system employs a multimodal pipeline combining VAD, ASR (Whisper), LLM-based intent classification, RAG for dialogue, face analysis for personalization, and voice synthesis (XTTS v2). Evaluation demonstrates high accuracy in command recognition (F1: 0.90), demographic estimation (gender F1: 0.89, age MAE: 5.14 years), and speech transcription (WER: 0.181). By uniting aerial robotics with adaptive conversational AI and self-contained visual output, HoverAI introduces a new class of spatially-aware, socially responsive embodied agents for applications in guidance, assistance, and human-centered interaction.

## Full Text


<!-- PDF content starts -->

HoverAI: An Embodied Aerial Agent for Natural Human-Drone
Interaction
Yuhua Jin∗
Chinese University of Hong Kong,
Shenzhen
Guangdong, China
yuhuajin@cuhk.edu.cnNikita Kuzmin∗
Skolkovo Institute of Science and
Technology
Moscow, Russia
Nikita.Kuzmin@skoltech.ruGeorgii Demianchuk†
Skolkovo Institute of Science and
Technology
Moscow, Russia
Georgii.Demianchuk@skoltech.ru
Mariya Lezina†
Skolkovo Institute of Science and
Technology
Moscow, Russia
Mariya.Lezina@skoltech.ruFawad Mehboob
Skolkovo Institute of Science and
Technology
Moscow, Russia
Fawad.Mehboob@skoltech.ruIssatay Tokmurziyev
Skolkovo Institute of Science and
Technology
Moscow, Russia
issatay.tokmurziyev@skoltech.ru
Miguel Altamirano Cabrera
Skolkovo Institute of Science and
Technology
Moscow, Russia
m.altamirano@skoltech.ruMuhammad Ahsan Mustafa
Skolkovo Institute of Science and
Technology
Moscow, Russia
Ahsan.Mustafa@skoltech.ruDzmitry Tsetserukou
Skolkovo Institute of Science and
Technology
Moscow, Russia
d.tsetserukou@skoltech.ru
Figure 1: The HoverAI interactive aerial interface: (a) The HoverAI drone featuring an RGB camera for user perception and
a front-mounted semi-rigid projection screen for displaying the interactive avatar; (b) User interacting with HoverAI in an
experimental environment, wearing headphones for audio input/output while the drone hovers and displays a projected avatar.
Abstract
Drones operating in human-occupied spaces suffer from insuffi-
cient communication mechanisms that create uncertainty about
their intentions. We present HoverAI, an embodied aerial agent
that integrates drone mobility, infrastructure-independent visual
projection, and real-time conversational AI into a unified platform.
Equipped with a MEMS laser projector, onboard semi-rigid screen,
and RGB camera, HoverAI perceives users through vision and voice,
responding via lip-synced avatars that adapt appearance to user
demographics. The system employs a multimodal pipeline com-
bining VAD, ASR (Whisper), LLM-based intent classification, RAG
for dialogue, face analysis for personalization, and voice synthesis
(XTTS v2). Evaluation demonstrates high accuracy in command
recognition (F1: 0.90), demographic estimation (gender F1: 0.89, age
∗These authors contributed equally to this work.
†These authors contributed equally to this work.MAE: 5.14 years), and speech transcription (WER: 0.181). By uniting
aerial robotics with adaptive conversational AI and self-contained
visual output, HoverAI introduces a new class of spatially-aware,
socially responsive embodied agents for applications in guidance,
assistance, and human-centered interaction.
CCS Concepts
•Human-centered computing →Collaborative interaction;
User interface design;•Computing methodologies →Vision
for robotics;•Computer systems organization →Robotic con-
trol.
Keywords
Human-drone interaction, MEMS projection, Embodied AI agents,
Aerial displaysarXiv:2601.13801v1  [cs.RO]  20 Jan 2026

Conference’17, July 2017, Washington, DC, USA Jin et al.
1 Introduction
Digital interfaces are increasingly moving beyond fixed screens
into physical environments, reshaping how people interact with
computational systems. Embodied AI agents offer a promising par-
adigm for more natural and spatially-aware human-computer inter-
action [ 6,15]. As spatial computing evolves, such agents can make
digital information accessible exactly where it is needed, enabling
seamless integration of content into shared environments. How-
ever, challenges in mobility, perception, communication, and social
presence must be addressed to fully realize this vision.
Traditional interfaces such as smartphones and monitors confine
users to fixed locations and require explicit attention, limiting their
suitability for dynamic, multi-user settings. Augmented and virtual
reality systems provide immersive experiences but often isolate
users behind head-mounted displays and depend on controlled en-
vironments. These limitations motivate the development of mobile
platforms that can interact with people directly within everyday
physical spaces.
Ground robots are a common solution, yet their dependence on
floor-based navigation restricts mobility. They must avoid obstacles,
navigate through doorways, and cope with constantly changing lay-
outs, making them unreliable in crowded or dynamic environments
such as airports or museums. Drones, in contrast, provide unre-
stricted three-dimensional mobility: they can hover at eye level,
bypass obstacles from above, and accompany users without ob-
structing pedestrian flow. These properties make drones attractive
as interactive embodied agents. Recent work has demonstrated the
feasibility of integrating advanced AI models onboard UAVs for
real-time reasoning [ 12], but natural human-drone communication
remains largely unresolved. Studies in human-drone interaction
show that uncertainty about drone intentions and state significantly
reduces trust and usability in public spaces [10, 11].
Prior efforts to create drone-based visual interfaces illustrate
both the potential and limitations of current approaches. Large-
scale drone light shows enable impressive aerial displays but are
designed for passive viewing rather than close interaction [ 5]. Sys-
tems such as BitDrones attached small displays to drones, yet these
solutions are constrained by weight, power, and viewing-angle limi-
tations [ 8]. Projection-based approaches like LightAir and MaskBot
enable expressive visuals, but rely on external surfaces or infrastruc-
ture, preventing fully mobile interaction [ 2,13]. Flying Light Specks
achieved detailed visualizations using coordinated drones and fixed
projectors, but similarly requires pre-installed equipment [ 7]. Re-
cent work on expressive drone avatars explored social presence
through digital faces [ 1], yet lacked real-time conversational intel-
ligence and infrastructure-independent output. To date, no aerial
system has combined onboard conversational AI, adaptive visual
projection, and real-time social interaction in a single lightweight
platform.
We present HoverAI, an interactive aerial agent that combines
drone mobility, ultra-light MEMS laser projection, real-time speech
understanding, and adaptive avatar expression in a self-contained
system. Using an onboard RGB camera and projection screen, Hov-
erAI perceives users and communicates via voice and projected
imagery without relying on external infrastructure. A multimodalpipeline (VAD, ASR, lightweight LLMs, and RAG) enables contex-
tual dialogue, while real-time face analysis personalizes a lip-synced
avatar. This closed-loop design allows HoverAI to function as a
socially present, spatially aware embodied agent capable of natural
interaction in shared environments.
2 System Architecture
Currently, no aerial system combines onboard conversational AI,
environment-independent visual output, and real-time social adap-
tation in a single lightweight platform. HoverAI addresses this gap
through a self-contained quadrotor that integrates flight, visual pro-
jection, environmental perception, and conversational interaction.
As shown in Fig. 2, the 1.2 kg platform comprises: an Orange Pi 5
single-board computer for real-time processing, a front-facing RGB
camera (1080p, 30 fps), and a MEMS laser projector (720p, 30 fps, 85
g) paired with a semi-rigid projection film, enabling infrastructure-
free visual display during flight.
Figure 2: HoverAI quadcopter key hardware components.
2.1 Hardware Architecture
The Laser Scanning Projection (LSP) module features a 2D MEMS
scanning mirror delivering 720p resolution without the weight
or power constraints of conventional displays. A semi-rigid poly-
carbonate film (0.3 mm, 40 g) suspended 15 cm in front of the
projector remains stable under airflow and vibrations, projecting
clear visuals without requiring external surfaces. The Orange Pi
5 manages flight control via a Speedybee F405V4 flight controller
and communicates wirelessly with a ground station PC over WiFi
(5 GHz, 50 ms latency). User speech is captured via a close-talking
headphone microphone to maximize signal-to-noise ratio in indoor
environments.
2.2 Interaction Pipeline
As illustrated in Fig. 3, HoverAI processes two parallel input streams.
Audio is transmitted via WiFi to the ground station, where it un-
dergoes spectral noise reduction, Voice Activity Detection (VAD)
[16], and Automatic Speech Recognition using Whisper-medium.en
[14] (average WER: 0.181). Transcripts are classified by gemma:7b-
instruct [4] into:

HoverAI: An Embodied Aerial Agent for Natural Human-Drone Interaction Conference’17, July 2017, Washington, DC, USA
Figure 3: HoverAI pipeline with audio analysis, video-based
face processing, and TTS-driven avatar output.
•Structured commands(“follow,” “land,” “stay,” “explore
face”) forwarded to the drone control module for immediate
execution.
•Conversational queriesrouted to a Retrieval-Augmented
Generation (RAG) system with a domain-specific knowledge
base (museum artefacts, FAQ), ensuring grounded responses
and minimizing hallucination.
Generated responses are synthesized via XTTS v2 Text-to-Speech,
which adapts voice characteristics (pitch, timbre) to match the se-
lected avatar demographic [3].
Concurrently, the video stream undergoes face analysis via In-
sightFace [ 9], estimating age and gender to select among four pre-
defined avatars: young woman ( <30), adult woman (≥30), young
man ( <30), and adult man ( ≥30). When no face is detected, a
gender-neutral default avatar is displayed.
The synthesized speech plays through user’s headphones while
a lip-synced avatar with subtitles renders on the drone screen at
∼25fps (total pipeline latency: 800–1200 ms). This closed-loop
architecture enables HoverAI to function as a socially present, em-
bodied agent that adapts appearance, voice, and behavior to user
input, supporting natural turn-taking and co-presence.
The modular design supports future extensions, including SLAM-
based autonomous navigation, multi-drone swarm coordination for
large-scale displays, and expanded knowledge bases with real-time
web querying while maintaining response reliability.
3 Evaluation
To validate HoverAI’s multimodal interaction capabilities, we con-
ducted a benchmark evaluation measuring performance across
speech recognition, intent classification, and demographic estima-
tion.
3.1 Experimental Setup
We recruited 12 participants (6 male, 6 female, aged 22-48) to inter-
act with HoverAI in a controlled indoor laboratory environment
(6×6 m). Each participant completed a 5-minute interaction session
involving:
•Speech tasks: 20 conversational queries (general knowledge,
navigation) and 10 structured commands (“follow me, ” “land, ”
“stay,” “explore face”)
•Vision tasks: Continuous face tracking during interaction
for demographic estimationSpeech was captured via close-talking headphones in ambient
noise conditions (45-50 dB). All sessions were recorded with in-
formed consent. The RAG knowledge base contained 150 curated
facts about robotics and museum artifacts.
3.2 Results
As shown in Fig. 4, HoverAI achieved strong performance across
all modalities:
•Speech Transcription (WER: 0.181): Whisper-medium.en
demonstrated reliable ASR despite ambient noise, with most
errors occurring on technical terminology.
•Command Recognition (F1: 0.90): The gemma:7b classi-
fier correctly distinguished commands from queries in 90%
of cases, with confusion primarily between “stay” and con-
versational “wait” statements.
•Gender Estimation (F1: 0.89): InsightFace achieved robust
classification across lighting conditions and viewing angles
(±30◦).
•Age Estimation (MAE: 5.14 years): While absolute age
error averaged 5.14 years, binary classification ( <30vs.≥30)
for avatar selection achieved 91.7% accuracy, sufficient for
demographic adaptation.
Figure 4: Performance metrics averaged across 12 partici-
pants: speech transcription (WER: 0.181), command recog-
nition (F1: 0.90), gender classification (F1: 0.89), and age esti-
mation (MAE: 5.14 years).
End-to-end pipeline latency averaged 950 ms ( ±120 ms) from
speech onset to avatar response, supporting natural conversational
turn-taking. No system crashes occurred during 60 minutes of total
interaction time.
4 Applications and Use Cases
HoverAI’s combination of mobility, conversational AI, and adaptive
visual presence enables several application scenarios:
Museum and Educational Guidance: HoverAI can follow visi-
tors through exhibitions, projecting contextual narratives, multilin-
gual subtitles, or 3D reconstructions directly aligned with physical

Conference’17, July 2017, Washington, DC, USA Jin et al.
artefacts. Its spatial mobility enables optimal positioning for visi-
bility, offering an alternative to static displays or audio guides for
users with limited mobility.
Assistive Communication: For users with motor disabilities
or speech impairments, HoverAI can display speech-to-text output,
mirror smartphone content, or serve as a visual proxy during re-
mote calls. Its hands-free operation and human-scale presence are
valuable in home healthcare and eldercare contexts where conven-
tional devices may be inaccessible.
Personal Companion: HoverAI can provide ambient assistance
in everyday environments, hovering at eye level to deliver contex-
tual information or social engagement through natural conversa-
tion, creating a persistent visual presence without requiring touch
or wearable devices.
5 Discussion
5.1 Limitations
The system demonstrates robust technical performance, achieving
high accuracy in speech recognition, command classification, and
demographic estimation, and establishes a compelling foundation
for socially responsive human-drone interaction.
However, several technical and methodological limitations con-
strain the current prototype. Flight time is limited to ∼12minutes
by battery capacity, restricting deployment duration. Stable pro-
jection requires indoor operation with minimal wind ( <0.5 m/s)
and controlled lighting; outdoor use degrades visibility. WiFi-based
audio streaming limits operational range to ∼15m. The semi-rigid
screen occasionally exhibits vibration artefacts during aggressive
maneuvers, requiring conservative flight profiles.
Critically, our evaluation focused on technical performance met-
rics rather than user perception. While we demonstrated reliable
speech recognition and demographic estimation, we did not as-
sess whether the adaptive avatar actually reduces uncertainty or
enhances social presence. Future work should therefore include
controlled user studies measuring perceived safety, trust, and clar-
ity of drone intentions during interaction. Qualitative feedback and
comparative experiments with non-adaptive or non-visual drone
interfaces would help determine how much the projected avatar
contributes to usability and overall user experience.
Deploying a drone that adapts its appearance based on users’
age and gender and displays a human-like avatar raises important
ethical questions. Recording people’s faces and voices in public,
especially without their consent, can violate privacy. Mistakes in
estimating age or gender may lead to misrepresentation or bias, and
the lifelike avatar might make people think the drone has inten-
tions it does not actually have. To address these risks, future work
should include clear indicators when recording is active, obtain
user consent, handle data transparently, and comply with privacy
laws.
The RAG system is limited to pre-defined knowledge bases ( ∼150
facts) and cannot retrieve real-time information or handle out-
of-domain queries gracefully, occasionally producing generic re-
sponses.5.2 Future Directions
Near-term improvements include SLAM-based autonomous naviga-
tion for spatial positioning, extended battery life through optimized
power management, and expanded knowledge bases with vetted
external source integration.
Robustness to real-world acoustic environments is another criti-
cal direction: our current evaluation was conducted in controlled
indoor settings with moderate ambient noise, but practical deploy-
ments in museums, educational spaces, or urban outdoor areas
often involve higher noise levels and overlapping speech. Future
work should therefore evaluate and enhance the system’s speech
perception pipeline under diverse acoustic conditions, potentially
incorporating noise-robust ASR models to maintain performance.
Longer-term research directions include multi-drone swarm co-
ordination for large-scale collaborative displays, 3D volumetric
visualization, and distributed perception across coordinated units.
Investigating outdoor deployment with brighter projection and
stabilized screen mechanisms would broaden applicability.
6 Conclusion
We presented HoverAI, an embodied aerial agent that integrates
infrastructure-independent visual projection, real-time conversa-
tional AI, and demographic-adaptive avatar generation into a self-
contained mobile platform. By combining MEMS laser projection
with a semi-rigid screen, multimodal perception through vision and
speech, and closed-loop interaction via LLM-based dialogue and
face analysis, HoverAI demonstrates a new approach to spatially-
aware human-drone interaction. Evaluation across 12 participants
showed robust performance in speech recognition (WER: 0.181),
command classification (F1: 0.90), and demographic estimation,
establishing technical feasibility for applications in guidance, as-
sistance, and companionship. HoverAI represents a step toward
mobile, socially responsive interfaces that bring digital content into
shared physical spaces in more human-centered ways.
Acknowledgements
Research reported in this publication was financially supported by
the RSF grant No. 24-41-02039.
References
[1]Robin Bretin, Mohamed Khamis, Emily Cross, and Mohammad Obaid. 2025. The
Role of Drone’s Digital Facial Emotions and Gaze in Shaping Individuals’ Social
Proxemics and Interpretation.J. Hum.-Robot Interact.14, 3, Article 48 (May 2025),
34 pages. doi:10.1145/3714477
[2]Miguel Altamirano Cabrera, Igor Usachev, Juan Heredia, Jonathan Tirado, Aleksey
Fedoseev, and Dzmitry Tsetserukou. 2020. Maskbot: Real-time robotic projec-
tion mapping with head motion tracking. InSIGGRAPH Asia 2020 Emerging
Technologies. 1–2.
[3]Edresson Casanova, Kelly Davis, Eren Golge, Gorkem Goknar, Iulian Gulea,
Logan Hart, Aya Aljafari, Joshua Meyer, et al .2024.Xtts: a massively mul-
tilingual zero-shot text-to-speech model. arXiv:2406.04904 Retrieved from
https://arxiv.org/abs/2406.04904.
[4]Google DeepMind. 2024. gemma:7b-instruct: Open model for Ollama. https:
//ollama.com/library/gemma:7b-instruct. Accessed: 2025-12-08.
[5]Dronisos. 2025. Drone Light Shows indoor vs outdoor. https://www.dronisos.
com/post/drone-light-shows-indoor-vs-outdoor Accessed: 2025-12-08.
[6]Pascale Fung, Yoram Bachrach, Asli Celikyilmaz, Kamalika Chaudhuri, Delong
Chen, Willy Chung, Emmanuel Dupoux, Hongyu Gong, Hervé Jégou, et al .2025.
Embodied AI Agents: Modeling the World. arXiv:2506.22355 Retrieved from
https://arxiv.org/abs/2506.22355.

HoverAI: An Embodied Aerial Agent for Natural Human-Drone Interaction Conference’17, July 2017, Washington, DC, USA
[7]Shahram Ghandeharizadeh. 2022. Display of 3D Illuminations using Flying Light
Specks. InProc. of the 30th ACM Int. Conf. on Multimedia. New York, NY, USA,
2996–3005. doi:10.1145/3503161.3548250
[8]Antonio Gomes, Calvin Rubens, Sean Braley, and Roel Vertegaal. 2016. Bit-
Drones: Towards Using 3D Nanocopter Displays as Interactive Self-Levitating
Programmable Matter. InProc. of the 2016 CHI Conf. on Human Factors in Com-
puting Systems (CHI ’16). 770–780. doi:10.1145/2858036.2858519
[9]InsightFace. 2017. InsightFace: State-of-the-art 2D & 3D Face Analysis Project.
https://github.com/deepinsight/insightface. Accessed: 2025-12-08.
[10] Shiva Lingam, Rutger Verstegen, Sebastiaan Petermeijer, and Marieke Martens.
2025. Human Interactions With Delivery Drones in Public Spaces: Design Rec-
ommendations From Recipient and Bystander Perspectives. doi:10.13140/RG.2.2.
16544.70405
[11] Shiva Nischal Lingam, Mervyn Franssen, Sebastiaan M Petermeijer, and Marieke
Martens. 2025. Challenges and future directions for human-drone interaction
research: an expert perspective.International Journal of Human–Computer Inter-
action41, 12 (2025), 7905–7921.
[12] Artem Lykov, Valerii Serpiva, Muhammad Haris Khan, Oleg Sautenkov, Artyom
Myshlyaev, Grik Tadevosyan, Yasheerah Yaqoot, and Dzmitry Tsetserukou. 2025.CognitiveDrone: A VLA Model and Evaluation Benchmark for Real-Time Cogni-
tive Task Solving and Reasoning in UAVs. arXiv:2503.01378. Retrieved from
https://arxiv.org/abs/2503.01378.
[13] Mikhail Matrosov, Olga Volkova, and Dzmitry Tsetserukou. 2016. LightAir: a
novel system for tangible communication with quadcopters using foot gestures
and projected image. InACM SIGGRAPH 2016 Emerging Technologies(Anaheim,
California)(SIGGRAPH ’16). Association for Computing Machinery, New York,
NY, USA, Article 16, 2 pages. doi:10.1145/2929464.2932429
[14] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and
Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision.
InInt. conf. on machine learning. PMLR, 28492–28518.
[15] Jihun Shin, Hyeonjin Kim, Eunseong Lee, Donghwan Shin, Kwang Bin Lee, Taehei
Kim, Hyeshim Kim, Joonsik An, and Sung-Hee Lee. 2025. Situated Embodied XR
Agents via Spatial Reasoning and Prompting . In2025 IEEE Int. Symposium on
Mixed and Augmented Reality Adjunct (ISMAR-Adjunct). IEEE Computer Society,
Los Alamitos, CA, USA, 933–934. doi:10.1109/ISMAR-Adjunct68609.2025.00255
[16] Silero Team. 2024. Silero VAD: pre-trained enterprise-grade Voice Activity De-
tector (VAD), Number Detector and Language Classifier. https://github.com/
snakers4/silero-vad.